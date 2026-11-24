"""Request-level discrete-event simulator of ONE serving process (closed loop, sequential requests).

Total service time = model loaded -> last request finished = sum of per-request durations (no idle gaps in a closed loop);
`load_s` is reported separately and as a second total (`total_incl_load_s`).

Per-request duration = [input_s] + steady time of the serving arm (cost table `steady_ms`) + one-time cost when the request
triggers an event on a NEW shape:
  eager_first   E0 one-time cost (`resid_s`: first_req - steady of E0 at this tier)
  capture       G1 one-time cost (eager warm-up + capture; `resid_s` of G1) ; graph memory += capture_pool_mb
  compile       C1: only the component keys NOT yet compiled in this process pay their first-call seconds
                (te key (H,W,B), UNet key (H,W,actual UNet batch), VAE key (H,W,B); see policy.base.component_keys),
                plus the shape-level residual (`resid_s`); optional post-compile transient excess (cold_empty tiers)
  compile_capture  same with the C1+G1 row (its residual contains the capture)
  capture_denied   capture would exceed the graph-memory budget -> served eagerly (E0), one-time eager cost only if new
Tier of the one-time cost: first shape the process handles -> cold_<cache_mode>; every later new shape -> inproc_<cache_mode>;
missing in-process tiers fall back to the same cache state's cold tier (costs._CHAIN; a missing cold tier raises MissingCost) and every fallback is recorded (event column `fallbacks`, summary counts).
The simulator makes no use of any real timing of the replay: only the cost table.

CLI (dev smoke): python -m p24.sim.simulator --cost-table T.csv --trace A.jsonl [B.jsonl ...] --policies B0,B4,B1 --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..policy import (CAPTURE, COMPILE, COMPILE_CAPTURE, COMPILE_DYNAMIC, COMPILE_RO, EAGER, EAGER_FALLBACK_LIMIT, RUN_BUCKET,
                      CostView, ProcState, component_keys, make_policy, plan_event, plan_step, shape_of)
from .costs import COMPILE_ARMS, CostTable, MissingCost, tier_for

ARM_OF_ACTION = {EAGER: "E0", CAPTURE: "G1", COMPILE: "C1", COMPILE_CAPTURE: "C1+G1", COMPILE_DYNAMIC: "C5", COMPILE_RO: "C2"}
DYN_KEYS = {"te": ("dyn", "te"), "unet": ("dyn", "unet"), "vae": ("dyn", "vae")}  # B2 (dynamic=True): one shape-generic key per component


@dataclass
class SimOptions:
    cache_mode: str = "warm"           # 'warm' (pre-warmed disk cache) | 'empty'
    load_s: float = 0.0                # model load seconds (reported separately)
    use_transient: bool = False        # add the post-compile transient excess (protocol sec 5) at cold_empty/inproc_empty
    input_s: float = 0.0               # per-request host input preparation seconds (replay column input_ms), default 0
    mem_budget_mb: Optional[float] = None
    te_by_resolution: bool = False  # te key = B (verified S-5; p24.policy.component_keys)
    # ---- S3 additions (defaults keep the S2 behaviour) ----
    overhead_aware: bool = False    # S3: admission counts +22.33 MB per graph (ProcState.overhead_aware)
    evict_s: float = 0.0            # seconds charged per evicted graph (release + empty_cache); not in the cost table -> 0 by default


@dataclass
class SimResult:
    policy: str
    events: List[dict]
    options: SimOptions
    summary: dict = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


def read_trace(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mem_base(ct: CostTable, arm: str, shape) -> Optional[float]:
    """Process memory without the graph pools: eager / compiled arms -> steady_peak_reserved_mb (weights + workspace);
    graph arms (G1, C1+G1) -> weights_alloc_mb (the pools are added from the ProcState)."""
    v = ct.row(arm, shape).get("weights_alloc_mb" if "G" in arm else "steady_peak_reserved_mb")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def simulate(requests: List[dict], policy, ct: CostTable, opt: Optional[SimOptions] = None) -> SimResult:
    """Simulate one process serving `requests` with `policy`. S3: optional policy.warmup(state, cost) entries run first (time counted,
    event rows idx -1, -2, ...), evict actions, rebuild costs, run_bucket bookkeeping, eager_fallback_limit (policy.recompile_limit)."""
    opt = opt or SimOptions()
    st = ProcState(mem_budget_mb=opt.mem_budget_mb, overhead_aware=opt.overhead_aware)
    st.recompile_limit = getattr(policy, "recompile_limit", None)
    cv = CostView(ct, opt.cache_mode, opt.te_by_resolution)
    events, t = [], 0.0
    acc = dict(peak_graph=0.0, peak_res=0.0, n_fb=0, n_mem_missing=0, warm_s=0.0)
    fb_kinds: Dict[str, int] = {}

    def step(idx, req_id, shape, act, is_warm):
        nonlocal t
        graph_arm = ARM_OF_ACTION.get(act.kind)
        mb = 0.0
        if act.kind in (CAPTURE, COMPILE_CAPTURE):
            mb = ct.capture_pool_mb(graph_arm, shape)
        ev, evicted = plan_step(act, st, shape, mb)
        tier = tier_for(opt.cache_mode, st.first_shape_pending)
        dur = 0.0 if is_warm else opt.input_s
        fbs: List[str] = []
        serve_arm = graph_arm
        if act.kind == RUN_BUCKET:
            serve_arm = "G1" if st.graph_kind[act.bucket] == "G1" else "C1+G1"
        elif ev in ("capture_denied", EAGER_FALLBACK_LIMIT):
            serve_arm = "E0"
        elif ev == "steady" and act.kind in (CAPTURE, COMPILE_CAPTURE) and shape in st.graph_kind:
            serve_arm = "G1" if st.graph_kind[shape] == "G1" else "C1+G1"
        steady_shape = act.bucket if act.kind == RUN_BUCKET else shape
        dur += ct.steady_s(serve_arm, steady_shape)
        rebuilt = shape in st.build_count and shape not in st.graphs and shape not in evicted  # graph of an evicted shape rebuilt
        if ev == "eager_first" or (ev in ("capture_denied", EAGER_FALLBACK_LIMIT) and shape not in st.eager_seen):
            ot = ct.one_time("E0", shape, tier)
            dur += ot.total_s
            fbs += ot.fallbacks
        elif ev == "capture":
            if rebuilt:
                dur += cv.rebuild_s("G1", shape)
            else:
                ot = ct.one_time("G1", shape, tier)
                dur += ot.total_s
                fbs += ot.fallbacks
        elif ev in ("compile", "compile_capture"):
            arm = graph_arm
            if ev == "compile_capture" and rebuilt and shape in st.compiled_shapes:
                dur += cv.rebuild_s(arm, shape)       # compiled code is kept; only the graph is captured again
            else:
                ot = ct.one_time(arm, shape, tier)
                dur += ot.resid_s
                keys = DYN_KEYS if act.kind == COMPILE_DYNAMIC else component_keys(shape, opt.te_by_resolution)
                for c, k in keys.items():
                    if k not in st.compiled_keys:
                        dur += ot.fc[c]
                fbs += ot.fallbacks
                if opt.use_transient and tier.endswith("empty"):
                    dur += ct.transient_s(arm, shape)
        dur += opt.evict_s * len(evicted)
        st.register(shape, ev, mb, opt.te_by_resolution, evicted=evicted,
                    served_bucket=act.bucket if act.kind == RUN_BUCKET else None)
        if act.kind == COMPILE_DYNAMIC and ev == "compile":
            st.compiled_keys.update(DYN_KEYS.values())
        base = _mem_base(ct, serve_arm, steady_shape)
        if base is None:
            acc["n_mem_missing"] += 1
            base = 0.0
        acc["peak_graph"] = max(acc["peak_graph"], st.graph_mem_mb)
        acc["peak_res"] = max(acc["peak_res"], base + st.graph_mem_mb)
        if fbs:
            acc["n_fb"] += 1
            for f in fbs:
                fb_kinds[f] = fb_kinds.get(f, 0) + 1
        t += dur
        events.append(dict(idx=idx, req_id=req_id, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg,
                           action=act.kind, event=ev, tier=tier if ev != "steady" else "", dur_s=dur, t_end_s=t,
                           graph_mem_mb=st.graph_mem_mb, fallbacks=";".join(fbs),
                           evicted=";".join(v.tag() for v in evicted), n_evicted=len(evicted)))
        return ev

    n_warm = 0
    if hasattr(policy, "warmup"):
        snap = (st.n_served, dict(st.seen_count), dict(st.hits), dict(st.last_used))
        for k, (wshape, wact) in enumerate(policy.warmup(st, cv)):
            ev = step(-(k + 1), f"warmup-{k}", wshape, wact, True)
            n_warm += 1
            if hasattr(policy, "on_event"):
                pass                                    # warm-up is not a request: no policy bookkeeping
        acc["warm_s"] = t
        st.n_served, st.seen_count, st.hits, st.last_used = snap[0], dict(snap[1]), dict(snap[2]), dict(snap[3])
    st.warm_done = True
    for i, req in enumerate(requests):
        shape = shape_of(req)
        act = policy.decide(req, st)
        ev = step(i, req.get("req_id", str(i)), shape, act, False)
        if hasattr(policy, "on_event"):
            policy.on_event(req, act, ev, st)
    counts: Dict[str, int] = {}
    for e in events:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    ereq = [e for e in events if e["idx"] >= 0]
    summary = dict(policy=policy.name, n_requests=len(requests), service_s=t, load_s=opt.load_s,
                   total_incl_load_s=t + opt.load_s, n_shapes=len({(e["H"], e["W"], e["B"], e["cfg"]) for e in ereq}),
                   peak_graph_mb=acc["peak_graph"], peak_reserved_est_mb=acc["peak_res"], n_mem_missing=acc["n_mem_missing"],
                   n_events_with_fallback=acc["n_fb"], fallbacks=json.dumps(fb_kinds, sort_keys=True),
                   cache_mode=opt.cache_mode, use_transient=int(opt.use_transient), input_s=opt.input_s,
                   n_warmup=n_warm, warmup_s=acc["warm_s"], n_evicted_total=st.evicted_total, n_denied_total=st.denied_total,
                   **{f"n_{k}": v for k, v in sorted(counts.items())})
    return SimResult(policy.name, events, opt, summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost-table", required=True)
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--policies", default="B0,B4,B1")
    ap.add_argument("--cache-mode", default="warm", choices=["warm", "empty"])
    ap.add_argument("--load-s", type=float, default=0.0)
    ap.add_argument("--input-s", type=float, default=0.0)
    ap.add_argument("--use-transient", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ct = CostTable.from_csv(a.cost_table)
    rows = []
    for tp in a.trace:
        reqs = read_trace(tp)
        name = pathlib.Path(tp).name.split("_")[0]
        for pn in a.policies.split(","):
            try:
                res = simulate(reqs, make_policy(pn), ct, SimOptions(a.cache_mode, a.load_s, a.use_transient, a.input_s))
            except MissingCost as e:  # a missing quantity is reported, never guessed
                rows.append(dict(trace=name, policy=pn, error=str(e)))
                continue
            res.frame().to_csv(out / f"sim_events_{name}_{pn}_{a.cache_mode}{a.tag}.csv", index=False)
            rows.append(dict(trace=name, **res.summary))
    pd.DataFrame(rows).to_csv(out / f"sim_summary_{a.cache_mode}{a.tag}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        df = pd.DataFrame(rows)
        cols = [c for c in ["trace", "policy", "n_requests", "n_shapes", "service_s", "total_incl_load_s", "peak_graph_mb",
                            "n_events_with_fallback", "error"] if c in df]
        print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
