"""Real replay of a request trace by ONE process (S2 step 2.6; S3 step 3.1 = tiered runtime, all policies).

One process loads the model once, then serves the trace in order (closed loop, sequential). S3 interface (code/docs/policy_interface_S3.md
sec 4): per request, strictly in this order and with the SAME code the simulator uses,
    act = policy.decide(req, state)                          (the only decision point)
    mb  = cost.capture_pool_mb(arm_of(act), shape)           (cost-table PREDICTION; admission never reads measured memory)
    event, evicted = plan_step(act, state, shape, mb)        (evictions are decided "judge first, act later": no white eviction)
    runtime: release(evicted) -> main action (eager / capture / compile / compile_capture / run_bucket) -> timing
    state.register(shape, event, mb, evicted=evicted); policy.on_event(req, act, event, state)
Warm-up (B3'', B6, B6'): `policy.warmup(state, cost) -> [(Shape, Action)]` is executed before request 1 through the same plan/execute/register
path WITHOUT the steady run; its time counts into the service time (`service_s` spans first warm-up start -> last request end).
Policies: B0 B1 B2 B3' B4 B5 B6 B6' Ours Ours-NE (CLI aliases B3p, B6p). Runtime: p24.runners_tiered.TieredRuntime (E0 + G1 + C1 + C1+G1 in one
process; C5 for B2). B3' = C1 with recompile_limit 8: a request that hits the limit (FailOnRecompileLimitHit) is served ENTIRELY on the
un-compiled modules (event `eager_fallback_limit`). An out-of-memory capture is recorded (note capture_oom) and served eagerly (`capture_denied`).

Per-request columns: t_start_s / t_end_s (seconds since the first item started), evict_ms (release + empty_cache), input_ms (tokenisation + noise),
prepare_ms (compile / capture, synchronized), run_ms, dur_ms (= t_end - t_start); event = what actually happened, planned_event = plan_step's
answer (differs only on the two runtime fallbacks above); evicted / n_evicted; served_by; reserved_mb / alloc_mb (measured, record only);
graph_mem_mb (accounted Σ predicted pools), graph_pred_mb (= graph_mem_mb + overhead x n_graphs); phase (req | warmup).
meta.json: service_s (warm-up start -> last request end), service_s_excl_input, total_incl_load_s, pinned_fallback_count (number of
P24_PINNED_FALLBACK_USED lines on this process's stderr; always written), host fields (MemAvailable / Shmem / uptime before and after).
The stderr of the process is saved, success or failure, as <out>/<tag>.stderr.txt (run_protocol_S3 sec 5).
Hard memory cap per process: 20,480 MB (shared GPU) or 23,552 MB (--exclusive, exclusive block) via set_per_process_memory_fraction.

--cache-mode warm   per-policy warm compile cache <warm-root>/<dir>/ (B0 E0, B4 G1, B1/B3' C1, B2 C5, B5/B6/B6'/Ours/Ours-NE C1_G1; default
                    warm-root $P24_CACHE/warm_dev; formal runs pass --warm-root $P24_CACHE/warm_s2), never cleared (cold_diskcache)
--cache-mode empty  fresh dir $P24_CACHE/cold/<out.name>/serve_<policy>_<trace>_p<id>/ , deleted when the process ends (cold_empty)
Existing outputs are never overwritten (exit code 3). Output: <out>/<policy>_<trace stem>_p<id>.csv, .meta.json, .stderr.txt.
S2 `serve()` (used by the S2 validation and tests/test_sim.py) is kept unchanged; the S3 loop is `serve_s3()`.

NOT to be run while GPU0 is used by a formal timing block. Usage (from code/):
  python -m scripts.serve_trace --trace ../results/S3/workload/dev_traces/W2_N1000_seed0_A.jsonl --policy Ours --budget-tier tight \\
         --cache-mode warm --out ../results/S3/dev_runs --proc-id 0 --max-requests 200
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
import shutil
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import p24.policy as _pol
from p24.policy import (CAPTURE, COMPILE, COMPILE_CAPTURE, EAGER, RUN_BUCKET, ProcState, make_policy, plan_event,
                        shape_of)
from p24.sim.simulator import ARM_OF_ACTION, read_trace

COMPILE_DYNAMIC = getattr(_pol, "COMPILE_DYNAMIC", "compile_dynamic")  # interface v1 sec 1 (B2); added to base.py by P
ROW_COLS = ["idx", "req_id", "H", "W", "B", "cfg", "action", "event", "status", "t_start_s", "t_end_s", "input_ms",
            "prepare_ms", "run_ms", "dur_ms", "reserved_mb", "alloc_mb", "graph_mem_mb", "prep_json",
            # ---- S3 3.1 additions (appended: S2 CSV column order unchanged) ----
            "phase", "planned_event", "evicted", "n_evicted", "evict_ms", "served_by", "graph_pred_mb", "n_graphs", "note"]
PREPARE_EVENTS = ("capture", "compile", "compile_capture")

POLICY_NAMES = ["B0", "B1", "B2", "B3′", "B4", "B5", "B6", "B6′", "Ours", "Ours-NE"]
POLICY_ALIASES = {"B3p": "B3′", "B3'": "B3′", "B3prime": "B3′", "B6p": "B6′", "B6'": "B6′", "B6prime": "B6′"}
POLICY_CACHE_DIR = {"B0": "E0", "B1": "C1", "B2": "C5", "B3′": "C1", "B4": "G1", "B5": "C1_G1", "B6": "C1_G1", "B6′": "C1_G1",
                    "Ours": "C1_G1", "Ours-NE": "C1_G1"}
NO_COMPILE = {"B0", "B4"}                                  # policies that never build compiled artefacts
BUDGETED = {"B4", "B5", "B6", "B6′", "Ours", "Ours-NE"}    # graph-layer budget applies (default tier when none given)
MEM_CAP_SHARED_MB, MEM_CAP_EXCLUSIVE_MB = 20480.0, 23552.0
PAD_SEED0 = 99000  # filler seeds of padded samples (same as scripts/run_equivalence.py)


def norm_policy(name: str) -> str:
    n = POLICY_ALIASES.get(name, name)
    if n not in POLICY_NAMES:
        raise argparse.ArgumentTypeError(f"unknown policy {name!r}; choose from {POLICY_NAMES} (aliases {sorted(POLICY_ALIASES)})")
    return n


def file_policy(name: str) -> str:
    return name.replace("′", "p")


# ============================================================ S2 loop (unchanged; mock-runner tests in tests/test_sim.py)
def serve(requests: List[dict], runner, policy, make_inputs: Callable, clock: Callable[[], float],
          sync: Callable[[], None], mem_fn: Callable[[], Tuple[float, float]], mem_budget_mb: Optional[float] = None,
          graph_mb_fn: Optional[Callable] = None, eager_run: Optional[Callable] = None):
    """Serve `requests` in order. Dependencies are injected so the loop runs with a mock runner on CPU.
    make_inputs(request, shape) -> inputs object; mem_fn() -> (reserved_mb, allocated_mb) (called outside the timed span).
    Graph-memory budget (S2 2.6, AUTO_DECISIONS 2026-11-15 23:35): with mem_budget_mb set, the admission of a new capture
    uses the PREDICTED pool graph_mb_fn(shape) (cost table, the same number the simulator uses), so replay and simulator
    take identical capture / capture_denied decisions; a denied shape is served eagerly via eager_run(inputs, key).
    The registered graph memory is the measured capture_pool_mb.
    Returns (rows, status, error); a failing request is recorded (status 'fail') and stops the replay."""
    st = ProcState(mem_budget_mb=mem_budget_mb)
    rows: List[dict] = []
    origin = None
    for i, req in enumerate(requests):
        shape = shape_of(req)
        act = policy.decide(req, st)
        if act.kind == RUN_BUCKET:
            raise NotImplementedError("run_bucket (padding) is S3: use serve_s3")
        mb = graph_mb_fn(shape) if (graph_mb_fn is not None and act.kind in (CAPTURE, COMPILE_CAPTURE)) else 0.0
        ev = plan_event(act, st, shape, mb)
        key = shape.to_shapekey() if hasattr(runner, "p") else shape  # mock runners take the plain Shape
        sync()
        t0 = clock()
        origin = t0 if origin is None else origin
        row = dict(idx=i, req_id=req.get("req_id", str(i)), H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg,
                   action=act.kind, event=ev, status="ok", t_start_s=t0 - origin, prepare_ms=0.0, prep_json="{}")
        prep: Dict = {}
        try:
            inp = make_inputs(req, shape)
            sync()
            t_in = clock()
            if ev in PREPARE_EVENTS:
                prep = runner.prepare(key, inp) or {}
                sync()
            t_p = clock()
            if ev == "capture_denied":
                eager_run(inp, key)
            else:
                runner.run(inp, key)
            sync()
            t1 = clock()
            row.update(t_end_s=t1 - origin, input_ms=(t_in - t0) * 1e3, prepare_ms=(t_p - t_in) * 1e3,
                       run_ms=(t1 - t_p) * 1e3, dur_ms=(t1 - t0) * 1e3, prep_json=json.dumps(prep, default=float))
        except Exception as e:  # noqa: BLE001  failures are data (capture / compile limits), the replay stops
            row.update(status="fail", prep_json=json.dumps({"error": repr(e)[:500]}))
            rows.append(row)
            return rows, "fail", repr(e)[:2000]
        res, alc = mem_fn()
        graph_mb = float(prep.get("capture_pool_mb", 0.0) or 0.0)
        st.register(shape, ev, graph_mb)
        row.update(reserved_mb=res, alloc_mb=alc, graph_mem_mb=st.graph_mem_mb)
        rows.append(row)
    return rows, "ok", ""


# ============================================================ S3 loop
def plan_step_compat(action, state, shape, graph_mb: float = 0.0):
    """Stand-in for p24.policy.plan_step (interface v1 sec 2) while base.py has no S3 increment: evictions in `action.evict` are
    applied to a copy of the state first; if the capture still does not fit -> `capture_denied` and NOTHING is evicted."""
    evict = tuple(getattr(action, "evict", ()) or ())
    for s in evict:
        if s not in state.graphs:
            raise KeyError(f"evict: {s} not in state.graphs")
    if not evict:
        return plan_event(action, state, shape, graph_mb), ()
    tmp = dataclasses.replace(state, graphs={k: v for k, v in state.graphs.items() if k not in evict},
                              graph_kind={k: v for k, v in state.graph_kind.items() if k not in evict})
    ev = plan_event(action, tmp, shape, graph_mb)
    return ev, (() if ev == "capture_denied" else evict)


def get_plan_step():
    return getattr(_pol, "plan_step", None) or plan_step_compat


def register_compat(state, shape, event, graph_mb, evicted=(), served_bucket=None):
    """state.register(..., evicted=, served_bucket=) of interface v1 / P's base.py; on the S2 base.py (no such parameters) the evicted
    graphs are deleted first."""
    params = inspect.signature(state.register).parameters
    if "evicted" in params:
        kw = dict(evicted=tuple(evicted))
        if "served_bucket" in params:
            kw["served_bucket"] = served_bucket
        state.register(shape, event, graph_mb, **kw)
        return
    for s in evicted:
        state.graphs.pop(s, None)
        state.graph_kind.pop(s, None)
    state.register(shape, event, graph_mb)


def new_state(mem_budget_mb, policy=None):
    """S3 ProcState: overhead-aware admission (sum(pool + 22.33 x n_graphs) <= budget) and the policy's recompile limit (B3': 8)."""
    kw = {}
    if "overhead_aware" in {f.name for f in dataclasses.fields(ProcState)}:
        kw["overhead_aware"] = True
    st = ProcState(mem_budget_mb=mem_budget_mb, **kw)
    if hasattr(st, "recompile_limit"):
        st.recompile_limit = getattr(policy, "recompile_limit", None)
    return st


def _pred_pool_mb(cost, act, shape) -> float:
    if cost is None or act.kind not in (CAPTURE, COMPILE_CAPTURE):
        return 0.0
    return float(cost.capture_pool_mb(ARM_OF_ACTION[act.kind], shape))


def serve_s3(requests: List[dict], rt, policy, make_inputs: Callable, clock: Callable[[], float], sync: Callable[[], None],
             mem_fn: Callable[[], Tuple[float, float]], *, cost=None, mem_budget_mb: Optional[float] = None,
             make_padded_inputs: Optional[Callable] = None, plan_step_fn: Optional[Callable] = None,
             register_fn: Optional[Callable] = None, state=None, warmup_req: Optional[dict] = None,
             fallback_on_limit: bool = False, stats: Optional[dict] = None):
    """S3 replay loop (module docstring). `rt` implements: p (has make_inputs; mock runtimes may omit), LimitExc, CaptureFailed,
    run_eager(inp,key), prepare_compiled(key,inp), run_compiled(inp,key), prepare_graph(kind,key,inp), run_graph(inp,key), graph_kind(key),
    evict(key) -> {freed_mb, evict_s}. Keys: ShapeKey for the real runtime (has attribute `p`), plain Shape otherwise.
    `cost` = CostView (predicted numbers; also handed to policy.warmup). Warm-up items follow p24.sim.simulator exactly: same plan/execute/
    register path as a request (including one steady run: the simulator charges one-time + steady), no input_s, no policy.on_event, and
    n_served / seen_count / hits / last_used are restored afterwards (a warm-up is not a request); `state.warm_done` is set True.
    `stats` (dict, optional) is filled: n_limit_fallback, n_capture_failed, evict_log, warmup_s, n_warmup.
    Returns (rows, status, error); a failing item is recorded (status 'fail') and stops the replay."""
    plan_step_fn = plan_step_fn or get_plan_step()
    register_fn = register_fn or register_compat
    st = state if state is not None else new_state(mem_budget_mb, policy)
    stats = stats if stats is not None else {}
    stats.update(n_limit_fallback=0, n_capture_failed=0, evict_log=[], warmup_s=0.0, n_warmup=0)
    rows: List[dict] = []
    origin = [None]
    to_key = (lambda s: s.to_shapekey()) if hasattr(rt, "p") else (lambda s: s)

    def item(idx, req, shape, act, phase):
        mb = _pred_pool_mb(cost, act, shape)
        ev_plan, evicted = plan_step_fn(act, st, shape, mb)
        evicted = tuple(evicted or ())
        key = to_key(shape)
        sync()
        t0 = clock()
        origin[0] = t0 if origin[0] is None else origin[0]
        row = dict(idx=idx, req_id=req.get("req_id", str(idx)) if phase == "req" else f"warmup:{shape.tag()}",
                   H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, action=act.kind, event=ev_plan, planned_event=ev_plan,
                   status="ok", t_start_s=t0 - origin[0], prepare_ms=0.0, prep_json="{}", phase=phase,
                   evicted=";".join(s.tag() for s in evicted), n_evicted=len(evicted), evict_ms=0.0, served_by="", note="")
        ev, prep, notes, served = ev_plan, {}, [], ""
        try:
            for s in evicted:  # (1) evict BEFORE the main action (capture needs its whole pool free at once, #14)
                pred = float(st.graphs.get(s, 0.0))
                info = rt.evict(to_key(s)) or {}
                stats["evict_log"].append(dict(idx=idx, shape=s.tag(), pred_mb=pred, freed_mb=info.get("freed_mb"),
                                               evict_s=info.get("evict_s")))
            sync()
            t_e = clock()
            if act.kind == RUN_BUCKET:
                inp = make_padded_inputs(req, shape, act.bucket)
            else:
                inp = make_inputs(req, shape)
            sync()
            t_in = clock()
            t_p = [None]

            def prepared():
                sync()
                t_p[0] = clock()

            def eager():
                rt.run_eager(inp, key)

            k = act.kind
            if k == EAGER or ev_plan in ("capture_denied", "eager_fallback_limit"):
                served = "E0"
                prepared()
                eager()
            elif k in (CAPTURE, COMPILE_CAPTURE):
                kind = "G1" if k == CAPTURE else "C1+G1"
                failed = False
                if ev_plan in ("capture", "compile_capture"):
                    try:
                        prep = rt.prepare_graph(kind, key, inp) or {}
                    except rt.CaptureFailed as e:
                        failed = True
                        stats["n_capture_failed"] += 1
                        ev, served = "capture_denied", "E0"
                        notes.append("capture_oom:" + str(e)[:120].replace("\n", " "))
                prepared()
                if failed:
                    eager()
                else:
                    rt.run_graph(inp, key)
                    served = rt.graph_kind(key) or kind
            elif k in (COMPILE, COMPILE_DYNAMIC):
                try:
                    if ev_plan == "compile":
                        prep = rt.prepare_compiled(key, inp) or {}
                    prepared()
                    served = "C5" if k == COMPILE_DYNAMIC else "C1"
                    rt.run_compiled(inp, key)
                except Exception as e:  # noqa: BLE001  safety net: the runtime hit Dynamo's limit although plan_step did not predict it
                    if not (fallback_on_limit and isinstance(e, rt.LimitExc)):
                        raise
                    stats["n_limit_fallback"] += 1
                    ev, served = "eager_fallback_limit", "E0"
                    notes.append("recompile_limit_hit_unplanned")
                    prepared()
                    eager()
            elif k == RUN_BUCKET:
                served = "bucket:" + act.bucket.tag()
                prepared()
                rt.run_graph(inp, to_key(act.bucket))  # padding gate (equiv_pass_padB4, bucket faster than eager) is in decide()
            else:
                raise ValueError(f"unsupported action {k}")
            sync()
            t1 = clock()
            row.update(t_end_s=t1 - origin[0], evict_ms=(t_e - t0) * 1e3, input_ms=(t_in - t_e) * 1e3,
                       prepare_ms=(t_p[0] - t_in) * 1e3, run_ms=(t1 - t_p[0]) * 1e3, dur_ms=(t1 - t0) * 1e3,
                       prep_json=json.dumps(prep, default=float), event=ev, served_by=served, note=";".join(notes))
        except Exception as e:  # noqa: BLE001  failures are data (compile limits, #18, ...), the replay stops
            row.update(status="fail", prep_json=json.dumps({"error": repr(e)[:500]}))
            rows.append(row)
            return False, repr(e)[:2000]
        res, alc = mem_fn()
        register_fn(st, shape, ev, mb if ev in ("capture", "compile_capture") else 0.0, evicted,
                    act.bucket if k == RUN_BUCKET else None)
        if phase == "req":
            hook = getattr(policy, "on_event", None)
            if hook is not None:
                hook(req, act, ev, st)
        else:
            stats["warmup_s"] += row["dur_ms"] / 1e3
            stats["n_warmup"] += 1
        row.update(reserved_mb=res, alloc_mb=alc, graph_mem_mb=st.graph_mem_mb,
                   graph_pred_mb=st.graph_mem_mb + (float(getattr(st, "per_graph_overhead_mb", 0.0)) if getattr(st, "overhead_aware", False)
                                                    else 0.0) * len(st.graphs), n_graphs=len(st.graphs))
        rows.append(row)
        return True, ""

    wu = getattr(policy, "warmup", None)
    if wu is not None:
        items = list(wu(st, cost) or [])
        wreq = warmup_req if warmup_req is not None else (requests[0] if requests else {})
        snap = (st.n_served, dict(getattr(st, "seen_count", {})), dict(getattr(st, "hits", {})), dict(getattr(st, "last_used", {})))
        for j, (shape, act) in enumerate(items):
            ok, err = item(-1 - j, wreq, shape, act, "warmup")
            if not ok:
                return rows, "fail", err
        st.n_served = snap[0]
        if hasattr(st, "seen_count"):
            st.seen_count, st.hits, st.last_used = snap[1], snap[2], snap[3]
    if hasattr(st, "warm_done"):
        st.warm_done = True
    for i, req in enumerate(requests):
        ok, err = item(i, req, shape_of(req), policy.decide(req, st), "req")
        if not ok:
            return rows, "fail", err
    return rows, "ok", ""


def summarize_times(rows: List[dict], load_s: float) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return dict(service_s=float("nan"), service_s_excl_input=float("nan"), total_incl_load_s=float("nan"))
    span = ok[-1]["t_end_s"] - ok[0]["t_start_s"]
    excl = sum(r["prepare_ms"] + r["run_ms"] for r in ok) / 1e3
    counts: Dict[str, int] = {}
    for r in ok:
        if r.get("phase", "req") == "req":
            counts[r["event"]] = counts.get(r["event"], 0) + 1
    out = dict(service_s=span, service_s_excl_input=excl, input_s_total=sum(r["input_ms"] for r in ok) / 1e3,
               total_incl_load_s=load_s + span, n_served=sum(1 for r in ok if r.get("phase", "req") == "req"),
               event_counts=counts, peak_reserved_mb=max(r["reserved_mb"] for r in ok),
               peak_graph_mb=max(r["graph_mem_mb"] for r in ok))
    n_ev = sum(int(r.get("n_evicted") or 0) for r in ok)
    if any("phase" in r for r in ok):
        out.update(n_evicted_total=n_ev, n_denied=counts.get("capture_denied", 0),
                   warmup_s=sum(r["dur_ms"] for r in ok if r.get("phase") == "warmup") / 1e3)
    return out


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=ROW_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in ROW_COLS})


# ============================================================ process plumbing (stderr, host meta, policy construction)
class StderrTee:
    """Tee fd 2 (Python AND C-level writes, e.g. compat.py notices, CUDA / Triton warnings) into <path> (append) while still showing it.
    Counts P24_PINNED_FALLBACK_USED lines (run_protocol_S3 sec 5: stderr of every timed process is saved; meta carries the count)."""
    MARK = b"P24_PINNED_FALLBACK_USED"

    def __init__(self, path):
        self.path, self.count, self._tail = pathlib.Path(path), 0, b""
        self._orig = None

    def start(self):
        sys.stderr.flush()
        self._orig = os.dup(2)
        r, w = os.pipe()
        os.dup2(w, 2)
        os.close(w)
        self._r = r
        self._f = open(self.path, "ab")
        self._th = threading.Thread(target=self._pump, daemon=True)
        self._th.start()
        return self

    def _pump(self):
        buf = b""
        while True:
            try:
                chunk = os.read(self._r, 65536)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(self._orig, chunk)
            except OSError:
                pass
            self._f.write(chunk)
            self._f.flush()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            self.count += sum(1 for ln in lines if self.MARK in ln)
        if self.MARK in buf:
            self.count += 1

    def stop(self):
        try:
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os.dup2(self._orig, 2)  # closes the pipe's write end held as fd 2 (children of the process may keep it open: bounded join)
        self._th.join(timeout=5.0)
        os.close(self._orig)
        self._f.close()
        return self.count


def host_snapshot() -> dict:
    """MemAvailable / Shmem / swap / load / uptime (run_protocol_S3 sec 4)."""
    d = {}
    try:
        mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in open("/proc/meminfo")}
        d.update(memavailable_mb=mi["MemAvailable"] // 1024, shmem_mb=mi["Shmem"] // 1024,
                 swap_used_pct=(100 * (mi["SwapTotal"] - mi["SwapFree"]) // mi["SwapTotal"]) if mi.get("SwapTotal") else 0)
        d["uptime_s"] = int(float(open("/proc/uptime").read().split()[0]))
        d["load1"] = os.getloadavg()[0]
    except Exception as e:  # noqa: BLE001  never block a run on a metadata read
        d["error"] = repr(e)
    return d


def load_prior(path) -> Dict:
    """JSON {tag or 'HxWxBxcfg': probability} -> {Shape: p}."""
    from p24.policy import Shape
    out = {}
    for k, v in json.loads(pathlib.Path(path).read_text()).items():
        if "x" in k and "_b" not in k:
            h, w, b, c = (int(x) for x in k.split("x"))
        else:  # tag form 512x512_b1_cfg
            hw, b_, c_ = k.split("_")
            h, w = (int(x) for x in hw.split("x"))
            b, c = int(b_[1:]), int(c_ == "cfg")
        out[Shape(h, w, b, c)] = float(v)
    return out


def build_policy(name: str, ct=None, params: Optional[dict] = None, prior=None, cache_mode: str = "warm"):
    """make_policy with the S3 constructor injection (cost, params, prior) when p24.policy supports it, the S2 call otherwise.
    Returns (policy, cost_view)."""
    kw, cv = {}, ct
    sig = inspect.signature(make_policy)
    if "cost" in sig.parameters:
        cv_cls = getattr(_pol, "CostView", None)
        cv = cv_cls(ct, cache_mode) if (cv_cls is not None and ct is not None) else ct
        kw = dict(cost=cv, params=params, prior=prior)
    last = None
    for cand in (name, file_policy(name), name.replace("′", "'")):
        try:
            return make_policy(cand, **kw), cv
        except KeyError as e:
            last = e
    raise SystemExit(f"policy {name!r} is not provided by p24.policy yet ({last!r}); P writes p24/policy/{{ours,baselines}}.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--policy", required=True, type=norm_policy)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--cache-mode", default="warm", choices=["warm", "empty"])
    ap.add_argument("--warm-root", default=None)
    ap.add_argument("--max-requests", type=int, default=0, help="serve only the first N requests (development)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--require-idle", action="store_true", help="exit if another process is on GPU0 at start")
    ap.add_argument("--cost-table", default=None, help="cost_table.csv (default: frozen S2 merged table)")
    ap.add_argument("--mem-budget-mb", type=float, default=None, help="graph-layer budget; overrides --budget-tier")
    ap.add_argument("--budget-tier", default=None, choices=["tight", "mid", "default", "wide"],
                    help="tier of results/S3/budgets.json (budgeted policies default to 'default'; B0/B1/B2/B3' have no budget)")
    ap.add_argument("--budgets-json", default=None)
    ap.add_argument("--params-json", default=None, help="policy params (alpha, H_prior, ablation switches); frozen file results/S3/policy_params.json")
    ap.add_argument("--prior-json", default=None, help="shape prior {tag: p}")
    ap.add_argument("--exclusive", action="store_true", help="exclusive GPU0 block: hard cap 23,552 MB instead of 20,480 MB")
    ap.add_argument("--mem-cap-mb", type=float, default=None, help="hard per-process cap (overrides --exclusive / default)")
    a = ap.parse_args()

    root = pathlib.Path(os.environ["PROJ_ROOT"])
    trace_path = pathlib.Path(a.trace)
    requests = read_trace(trace_path)
    if a.max_requests:
        requests = requests[:a.max_requests]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{file_policy(a.policy)}_{trace_path.stem}_p{a.proc_id}"
    for suffix in (".csv", ".meta.json"):
        if (out / f"{tag}{suffix}").exists():
            print(f"REFUSING to overwrite existing {out / (tag + suffix)} (use a new --out or --proc-id)", file=sys.stderr, flush=True)
            sys.exit(3)
    tee = StderrTee(out / f"{tag}.stderr.txt").start()
    cleanup = None
    meta: dict = {}
    host0 = host_snapshot()
    try:
        p24 = pathlib.Path(os.environ["P24_CACHE"])
        arm_dir = POLICY_CACHE_DIR[a.policy]
        if a.cache_mode == "empty":
            cdir = p24 / "cold" / out.name / f"serve_{file_policy(a.policy)}_{trace_path.stem}_p{a.proc_id}"
            if cdir.exists():
                shutil.rmtree(cdir)
            cdir.mkdir(parents=True)
            cleanup = cdir
        else:
            warm_root = pathlib.Path(a.warm_root) if a.warm_root else p24 / "warm_dev"
            cdir = warm_root / arm_dir
            cdir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")  # must precede the torch / inductor import
        os.environ["TRITON_CACHE_DIR"] = str(cdir / "triton")

        import torch
        from p24 import diag
        from p24.bench import PROMPTS_DEFAULT, diag_columns, gpu0_other_procs, host_columns, meta_snapshot, smi
        from p24.runners_tiered import TieredRuntime
        from p24.sim.costs import CostTable

        if a.require_idle and gpu0_other_procs() != 0:
            sys.exit("GPU0 is not idle (--require-idle)")
        meta = meta_snapshot(a)
        torch.cuda.init()
        total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        cap_mb = a.mem_cap_mb or (MEM_CAP_EXCLUSIVE_MB if a.exclusive else MEM_CAP_SHARED_MB)
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_mb / total_mb), 0)

        ct_path = pathlib.Path(a.cost_table) if a.cost_table else root / "results/S2/cost_table/merged/cost_table.csv"
        ct = CostTable.from_csv(ct_path)
        ct_sha = hashlib.sha256(ct_path.read_bytes()).hexdigest()[:16]
        budget = a.mem_budget_mb
        tier = a.budget_tier
        budgets_path = pathlib.Path(a.budgets_json) if a.budgets_json else root / "results/S3/budgets.json"
        if budget is None and (tier or a.policy in BUDGETED):
            bj = json.loads(budgets_path.read_text())
            tier = tier or "default"
            budget = float(bj["tiers"][tier]["mb"])
            if bj.get("cost_table_sha16") != ct_sha:
                sys.exit(f"cost table SHA {ct_sha} != budgets.json cost_table_sha16 {bj.get('cost_table_sha16')}")
        params = json.loads(pathlib.Path(a.params_json).read_text()) if a.params_json else None
        if a.prior_json:
            prior = load_prior(a.prior_json)
        else:  # default prior = A-half shape distribution (p24.sim.prior; evaluation traces never enter it)
            try:
                from p24.sim.prior import prior_A
                prior = prior_A()
            except ImportError:
                prior = None
        policy, cv = build_policy(a.policy, ct, params, prior, "empty" if a.cache_mode == "empty" else "warm")
        meta.update(mem_cap_mb=cap_mb, mem_budget_mb=budget, budget_tier=tier, exclusive=bool(a.exclusive), cost_table=str(ct_path),
                    cost_table_sha256_16=ct_sha, policy=a.policy, params_json=a.params_json, prior_json=a.prior_json,
                    pinned_fallback_env=os.environ.get("P24_PINNED_FALLBACK"))
        limit_watch = diag.install_limit_watch()
        model_dir = a.model_dir or str(root / "data/models/sd15_lcm_fused")
        t0 = time.perf_counter()
        rt = TieredRuntime(model_dir, compile=a.policy not in NO_COMPILE, dynamic=(a.policy == "B2"),
                           recompile_limit=int(getattr(policy, "recompile_limit", None) or 64))
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        meta["recompile_limit_configured"] = rt.recompile_limit if rt.compile_enabled else None

        def make_inputs(req, shape):
            prompts = [req["prompt"]] * shape.B
            return rt.p.make_inputs(prompts, shape.to_shapekey(), seeds=[int(req["noise_seed"]) + j for j in range(shape.B)])

        def make_padded_inputs(req, shape, bucket):  # same filler convention as scripts/run_equivalence.py (outputs of fillers discarded)
            prompts = [req["prompt"]] * shape.B + [PROMPTS_DEFAULT[0]] * (bucket.B - shape.B)
            seeds = [int(req["noise_seed"]) + j for j in range(shape.B)] + [PAD_SEED0 + j for j in range(shape.B, bucket.B)]
            return rt.p.make_inputs(prompts, bucket.to_shapekey(), seeds=seeds)

        meta["policy_sig"] = dict(plan_step="p24.policy" if hasattr(_pol, "plan_step") else "compat", register_evicted=(
            "evicted" in inspect.signature(ProcState.register).parameters))
        stats: dict = {}
        host_start = host_snapshot()
        rows, status, err = serve_s3(
            requests, rt, policy, make_inputs, time.perf_counter, torch.cuda.synchronize, rt.mem_mb, cost=cv, mem_budget_mb=budget,
            make_padded_inputs=make_padded_inputs, fallback_on_limit=(a.policy == "B3′"), stats=stats)
        tier_label = "cold_empty" if a.cache_mode == "empty" else "cold_diskcache"
        dcols = diag_columns(rt, err, limit_watch, tier_label, meta["cpu_affinity"])
        meta.update(gpu0_other_procs_end=gpu0_other_procs(), loadavg_end=os.getloadavg(),
                    smi_end=smi("clocks.sm,temperature.gpu,power.draw"))
        trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        pkg = pathlib.Path(__file__).resolve().parents[1] / "p24"  # bench.code_sha() only globs p24/*.py and scripts/*.py
        meta["code_sha256_16_subpackages"] = {f"{d}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                                              for d in ("policy", "sim") for f in sorted((pkg / d).glob("*.py"))}
        ev_gaps = [abs(e["freed_mb"] - e["pred_mb"]) for e in stats["evict_log"] if e.get("freed_mb") is not None]
        meta.update(status=status, error=err, arm="TIERED", trace=str(trace_path), trace_sha256=trace_sha,
                    n_requests=len(requests), cache_mode=a.cache_mode, cache_dir=str(cdir), load_s=load_s,
                    host_cols=host_columns(meta), diag=dcols, recompile_limit_msgs=limit_watch.msgs,
                    n_limit_fallback=stats["n_limit_fallback"], n_capture_failed=stats["n_capture_failed"],
                    n_warmup_items=stats["n_warmup"], evict_log=stats["evict_log"][:200],
                    evict_gap_mb_max=max(ev_gaps) if ev_gaps else None, runtime_evict_log=rt.evict_log[:200],
                    **summarize_times(rows, load_s))
        write_rows(out / f"{tag}.csv", rows)
        meta["host"] = dict(before=host0, start=host_start, after=host_snapshot())
    except BaseException as e:  # noqa: BLE001  a crash still leaves a meta with pinned_fallback_count (record completeness)
        meta.setdefault("status", "crash")
        meta.setdefault("error", repr(e)[:2000])
        raise
    finally:
        if cleanup is not None:  # empty-cache runs leave nothing behind
            shutil.rmtree(cleanup, ignore_errors=True)
        n_pin = tee.stop()
        meta["pinned_fallback_count"] = n_pin
        meta["stderr_file"] = str(out / f"{tag}.stderr.txt")
        if meta:
            (out / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps({k: meta[k] for k in ("policy", "status", "n_requests", "load_s", "service_s", "service_s_excl_input",
                                                "event_counts", "pinned_fallback_count") if k in meta}), flush=True)


if __name__ == "__main__":
    main()
