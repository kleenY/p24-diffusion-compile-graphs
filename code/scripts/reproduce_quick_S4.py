"""S4 4.4b: `python -m p24.reproduce_main_figure --quick` -- real-GPU scaled-down reproduction from an EMPTY compile cache.

What it does (goals/S4_detailed_goals_v1.md 4.4 + Gate-2 revision 5; criteria below were written BEFORE any real run of this script):
  1. trace  : the first N_REQ requests of the frozen real-validation stream W2_N300_seed100_B (S3 3.4) that use one of the two micro-benchmark
              shapes SHAPES (the filter is declared in quick_trace.meta.json).
  2. micro  : cost micro-benchmark of the 2 shapes with the frozen S2 tool scripts/bench_seq.py (through scripts/quick_capped_S4.py so that the
              GPU memory cap applies), arms E0, C1, C1+G1, each process from a fresh EMPTY compile cache (first key = cold_empty, second = inproc).
  3. serve  : the short trace, one process each, policies E0 (=B0), B1, Ours, `scripts/serve_trace.py --cache-mode empty` (real replay, frozen
              policies and frozen params results/S3/real_validation/params_ours_flat.json, graph budget default tier), from an empty compile cache.
  4. predict: the request-level simulator (frozen cost table + S3 addendum, cache mode empty, A-half prior; same code as scripts/predict_S3_4.py)
              predicts the total service time of the same three runs; input preparation time from the S2 input summary.
  5. verdict: quick_table.csv, quick_verdict.json, reduced main figure fig_quick.{png,pdf}.
Pre-registered criteria (fixed before running):
  C1  wall time of the whole real part (micro + serve, incl. model loads and compiles) <= 15 min                        [target of 4.4b]
  C2  |predicted - measured| / measured of the total service time <= 15 % for each of E0, B1, Ours                       [S4 Gate-2 revision 5]
  C3  micro-benchmark steady median within 15 % of the frozen cost table steady_ms for each (arm, shape) (4 + 4 + 4 cells)  [4.4b]
  C4  GPU memory: max reserved of every process <= 14 GB (hard cap set by set_per_process_memory_fraction; OOM = failure)  [Gate-2 revision 5]
  C5  informational: measured ordering of E0 / B1 / Ours vs predicted ordering; pinned-memory fall-back count; event counts vs prediction.
A run that is not on an idle GPU0 (`--shared`) is a development trial, flagged `formal: false`; the formal run (exclusive, GPU0 idle, MemAvailable >= 25 GB)
is scheduled by the main session.

Usage (from code/, after `source ../ops/project_env.sh`):
  CPU, nothing runs on a GPU:  python -B -m p24.reproduce_main_figure --quick --dry-run
  real run (GPU0 only):        export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41 ; python -B -m p24.reproduce_main_figure --quick [--shared]
  tool self-test (CPU):        python -B scripts/reproduce_quick_S4.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_TRACE = ROOT / "results/S3/workload/real_validation/W2_N300_seed100_B.jsonl"
PARAMS_FLAT = ROOT / "results/S3/real_validation/params_ours_flat.json"
CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
DEFAULT_OUT = ROOT / "results/S4/reproduce/quick"
TMP_ROOT = pathlib.Path("tmp")
GPU0_UUID = "GPU-722679a6-175b-cb56-b764-e8124d262c41"

SHAPES = [(512, 512, 1, 0), (768, 512, 1, 0)]          # (H, W, B, cfg); the two most common single-image shapes of W2 with measured cold_empty costs
N_REQ = 60
CAP_MB = 14336                                          # 14 GB hard cap (S4 Gate-2 revision 5)
MICRO_ARMS = ["E0", "C1", "C1+G1"]
MICRO_REPS, MICRO_WARM = 5, 2
POLICIES = [("E0", "B0"), ("B1", "B1"), ("Ours", "Ours")]   # (label in figure/table, policy name in serve_trace / simulator)
TOL_TIME, TOL_STEADY, WALL_MAX_S = 0.15, 0.15, 15 * 60
MEM_MIN_MB, LOAD_MAX = 25600, 16
PIN = ["taskset", "-c", "0-15"]
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
key_str = lambda k: "x".join(str(v) for v in k)


# ------------------------------------------------------------------------------------------------------------------ trace
def build_trace(out: pathlib.Path) -> pathlib.Path:
    rows = [json.loads(l) for l in SRC_TRACE.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if (r["H"], r["W"], r["B"], r["cfg"]) in SHAPES][:N_REQ]
    assert len(keep) == N_REQ, f"source stream has only {len(keep)} requests in the two shapes"
    p = out / "quick_trace.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in keep))
    n_by = {key_str(k): sum(1 for r in keep if (r["H"], r["W"], r["B"], r["cfg"]) == k) for k in SHAPES}
    (out / "quick_trace.meta.json").write_text(json.dumps(dict(
        source=str(SRC_TRACE.relative_to(ROOT)), source_sha16=sha16(SRC_TRACE), rule=f"first {N_REQ} requests whose (H,W,B,cfg) is in {SHAPES}",
        n_requests=len(keep), n_by_shape=n_by, first_req_id=keep[0]["req_id"], last_req_id=keep[-1]["req_id"], sha16=sha16(p)), indent=1))
    return p


# ------------------------------------------------------------------------------------------------------------------ prediction
def predict(trace: pathlib.Path) -> pd.DataFrame:
    """Simulator prediction of the three serve runs (same cost basis / code path as scripts/predict_S3_4.py), cache mode `empty`."""
    sys.path.insert(0, str(ROOT / "code"))
    from scripts import predict_S3_4 as PR
    from p24.policy import CostView, make_policy, shape_of
    from p24.sim import simulator as SIM
    from p24.sim import addendum as ADD
    from p24.sim.prior import prior_A
    P = PR.load_params()
    ct = ADD.load_with_addendum(PR.CT_PATH, PR.ADDENDUM_PATH)
    ADD.install_simulator_hook()
    budget = float(json.loads(PR.BUDGETS_PATH.read_text())["tiers"]["default"]["mb"])
    inp = PR.InputModel(PR.INPUT_PATH)
    reqs = SIM.read_trace(trace)
    rows = []
    for label, pol in POLICIES:
        bud = budget if pol in PR.BUDGETED else None
        cv = CostView(ct, "empty", rebuild_override=getattr(ct, "rebuild_override", None))
        polobj = make_policy(pol, cost=cv, params=dict(P), prior=prior_A())
        res = SIM.simulate(reqs, polobj, ct, SIM.SimOptions(cache_mode="empty", mem_budget_mb=bud, overhead_aware=True))
        inp.n_extrap = 0
        tin = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
        ev = pd.DataFrame(res.events)
        ev = ev[ev.idx >= 0] if "idx" in ev else ev
        rows.append(dict(label=label, policy=pol, pred_excl_input_s=float(res.summary["service_s"]), pred_input_s=tin,
                         pred_total_s=float(res.summary["service_s"]) + tin, n_input_extrapolated=inp.n_extrap, mem_budget_mb=bud,
                         pred_event_counts=json.dumps(ev.event.value_counts().sort_index().to_dict(), sort_keys=True)))
    return pd.DataFrame(rows)


def frozen_steady() -> dict:
    d = pd.read_csv(CT_PATH)
    out = {}
    for arm in MICRO_ARMS:
        for k in SHAPES:
            r = d[(d.arm == arm) & (d.H == k[0]) & (d.W == k[1]) & (d.B == k[2]) & (d.cfg == k[3])]
            out[(arm, key_str(k))] = dict(steady_ms=float(r.steady_ms.iloc[0]), cold_empty_first_req_s=float(r.first_req_s__cold_empty.iloc[0]))
    return out


# ------------------------------------------------------------------------------------------------------------------ plan / run
def commands(out: pathlib.Path, trace: pathlib.Path, cache_root: pathlib.Path, shared: bool) -> list:
    """[(tag, argv)] in execution order: 3 micro-benchmark processes, then 3 serve processes."""
    py = [sys.executable, "-B"]
    cmds = []
    for arm in MICRO_ARMS:
        cmds.append((f"micro_{arm}", PIN + py + [str(ROOT / "code/scripts/quick_capped_S4.py"), "--cap-mb", str(CAP_MB), "--cache-root", str(cache_root),
                                                 "--cache-tag", f"micro_{arm.replace('+', '_')}", "--module", "scripts.bench_seq", "--",
                                                 "--arm", arm, "--keys", ",".join(key_str(k) for k in SHAPES), "--reps", str(MICRO_REPS),
                                                 "--warm", str(MICRO_WARM), "--cache", "env", "--cold-tier", "cold_empty", "--out", str(out / "micro")]))
    for label, pol in POLICIES:
        c = PIN + py + ["-m", "scripts.serve_trace", "--trace", str(trace), "--policy", pol, "--out", str(out / "serve"), "--proc-id", "0",
                        "--cache-mode", "empty", "--mem-cap-mb", str(CAP_MB), "--params-json", str(PARAMS_FLAT)]
        if not shared:
            c.append("--require-idle")
        cmds.append((f"serve_{label}", c))
    return cmds


def estimate_s(fz: dict, pred: pd.DataFrame) -> dict:
    """Wall-time estimate (s) from the frozen cost table: per process 25 s start-up (python, torch, model load) + compiles + requests."""
    start = 25.0
    est = {}
    for arm in MICRO_ARMS:
        t = start
        for i, k in enumerate(SHAPES):
            f = fz[(arm, key_str(k))]
            first = f["cold_empty_first_req_s"] if not np.isnan(f["cold_empty_first_req_s"]) else 45.0   # compile from an empty cache (both keys: new graphs)
            t += first + (MICRO_WARM + MICRO_REPS) * f["steady_ms"] / 1e3
        est[f"micro_{arm}"] = t
    for r in pred.itertuples():
        est[f"serve_{r.label}"] = start + r.pred_total_s
    est["gaps"] = 5.0 * (len(est) + 1)
    est["total"] = sum(est.values())
    return est


def gates(shared: bool) -> list:
    """Return the list of violated gates (empty = go)."""
    bad = []
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU0_UUID:
        bad.append(f"CUDA_VISIBLE_DEVICES must be exactly the GPU0 UUID {GPU0_UUID} (got {os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemAvailable", "SwapTotal", "SwapFree")}
    if mi["MemAvailable"] / 1024 < MEM_MIN_MB:
        bad.append(f"MemAvailable {mi['MemAvailable']/1024:.0f} MB < {MEM_MIN_MB}")
    if os.getloadavg()[0] > LOAD_MAX:
        bad.append(f"load1 {os.getloadavg()[0]:.1f} > {LOAD_MAX}")
    q = subprocess.run(["nvidia-smi", "-i", GPU0_UUID, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
    others = [l for l in q.stdout.splitlines() if l.strip()]
    if others and not shared:
        bad.append(f"GPU0 not idle (compute apps: {others}); use --shared for a development trial")
    if shared:
        u = subprocess.run(["nvidia-smi", "-i", GPU0_UUID, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        used = float(u.stdout.strip() or 0)
        if used + CAP_MB > 24000:
            bad.append(f"--shared: GPU0 already uses {used:.0f} MB; {used:.0f} + cap {CAP_MB} > 24000")
    return bad


def run_real(out: pathlib.Path, trace: pathlib.Path, shared: bool) -> dict:
    bad = gates(shared)
    if bad:
        sys.exit("GATE FAILED, nothing started:\n  " + "\n  ".join(bad))
    for sub in ("micro", "serve"):
        if (out / sub).exists():
            sys.exit(f"{out / sub} exists (tools refuse to overwrite raw results); use a new --out")
    cache_root = TMP_ROOT / f"p24_quick_cache_{time.strftime('%Y%m%d_%H%M%S')}"
    cache_root.mkdir(parents=True)
    env = dict(os.environ, P24_CACHE=str(cache_root), P24_PINNED_FALLBACK="1", TORCHINDUCTOR_COMPILE_THREADS="8", PROJ_ROOT=str(ROOT),
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "code") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    for v in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        env.pop(v, None)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    walls, t0 = {}, time.time()
    try:
        for tag, argv in commands(out, trace, cache_root, shared):
            t = time.time()
            with open(out / "logs" / f"{tag}.log", "w") as lg:
                r = subprocess.run(argv, env=env, cwd=str(ROOT / "code"), stdout=lg, stderr=subprocess.STDOUT, timeout=900)
            walls[tag] = dict(wall_s=time.time() - t, rc=r.returncode)
            print(f"{tag}: rc={r.returncode} {walls[tag]['wall_s']:.0f}s", flush=True)
            if r.returncode != 0:
                print(f"  FAILED, see {out / 'logs' / (tag + '.log')}; continuing with the next process", flush=True)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)       # temp compile caches: deleted whether the run succeeded or not
    walls["_total_wall_s"] = time.time() - t0
    walls["_cache_root_removed"] = not cache_root.exists()
    return walls


# ------------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(out: pathlib.Path, pred: pd.DataFrame, walls: dict, shared: bool) -> dict:
    fz = frozen_steady()
    rows, cells = [], []
    for r in pred.itertuples():
        mp = next(iter((out / "serve").glob(f"{r.policy}_quick_trace_p0.meta.json")), None)
        m = json.loads(mp.read_text()) if mp else {}
        ok = m.get("status") == "ok"
        meas = float(m["service_s"]) if ok else float("nan")
        rows.append(dict(label=r.label, policy=r.policy, pred_total_s=r.pred_total_s, meas_total_s=meas, rel_err=(r.pred_total_s - meas) / meas if ok else float("nan"),
                         status=m.get("status", "missing"), peak_reserved_mb=m.get("peak_reserved_mb"), pinned_fallback=m.get("pinned_fallback_count"),
                         pred_event_counts=r.pred_event_counts, meas_event_counts=json.dumps(m.get("event_counts", {}), sort_keys=True)))
    tab = pd.DataFrame(rows)
    for arm in MICRO_ARMS:
        mp = next(iter((out / "micro").glob(f"{arm}_seq_p0.meta.json")), None)
        pk = {x["key"]: x for x in json.loads(mp.read_text()).get("per_key", [])} if mp else {}
        for k in SHAPES:
            tag = f"{k[0]}x{k[1]}_b{k[2]}_{'cfg' if k[3] else 'nocfg'}"
            x = pk.get(tag)
            f = fz[(arm, key_str(k))]
            st = float(x["steady_med_ms"]) if x and x.get("status") == "ok" else float("nan")
            cells.append(dict(arm=arm, shape=key_str(k), frozen_steady_ms=f["steady_ms"], meas_steady_ms=st, rel_diff=(st - f["steady_ms"]) / f["steady_ms"],
                              frozen_cold_empty_first_req_s=f["cold_empty_first_req_s"], meas_first_req_s=float(x["first_req_s"]) if x else float("nan")))
    cell = pd.DataFrame(cells)
    tab.to_csv(out / "quick_table.csv", index=False)
    cell.to_csv(out / "quick_micro_vs_frozen.csv", index=False)
    wall = walls.get("_total_wall_s")
    mem_cells = [v for v in tab.peak_reserved_mb.dropna()] + []
    v = dict(
        formal=not shared,
        C1_wall_s=wall, C1_pass=bool(wall is not None and wall <= WALL_MAX_S),
        C2_max_abs_rel_err=float(np.nanmax(np.abs(tab.rel_err))) if tab.rel_err.notna().any() else None,
        C2_pass=bool(tab.rel_err.notna().all() and (tab.rel_err.abs() <= TOL_TIME).all()),
        C3_max_abs_rel_diff=float(np.nanmax(np.abs(cell.rel_diff))) if cell.rel_diff.notna().any() else None,
        C3_pass=bool(cell.rel_diff.notna().all() and (cell.rel_diff.abs() <= TOL_STEADY).all()),
        C4_max_peak_reserved_mb=float(max(mem_cells)) if mem_cells else None,
        C4_pass=bool(mem_cells and max(mem_cells) <= CAP_MB and all(w.get("rc") == 0 for k, w in walls.items() if not k.startswith("_"))),
        C5_order_pred=list(tab.sort_values("pred_total_s").label), C5_order_meas=list(tab.dropna(subset=["meas_total_s"]).sort_values("meas_total_s").label),
        pinned_fallback_counts=dict(zip(tab.label, tab.pinned_fallback)), walls=walls,
        note="formal=false: GPU0 was shared (development trial, timings not for the paper)" if shared else "exclusive-GPU0 run")
    v["all_pass"] = bool(v["C1_pass"] and v["C2_pass"] and v["C3_pass"] and v["C4_pass"])
    (out / "quick_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    draw_quick(tab, out / "fig_quick", v)
    return v


def draw_quick(tab: pd.DataFrame, path: pathlib.Path, v: dict | None):
    """Reduced main figure: total service time / E0 for prediction (hatched) and measurement; same palette/style as the S3 main figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "code"))
    from p24 import reproduce_main_figure as RF
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.edgecolor": RF.AXIS, "text.color": RF.INK})
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4), facecolor=RF.SURFACE)
    RF.style(ax)
    e0p = float(tab[tab.label == "E0"].pred_total_s.iloc[0])
    e0m = float(tab[tab.label == "E0"].meas_total_s.iloc[0])
    x = np.arange(len(tab))
    w = 0.36
    pr = tab.pred_total_s.to_numpy() / e0p
    ms = tab.meas_total_s.to_numpy() / e0m if np.isfinite(e0m) else np.full(len(tab), np.nan)
    ax.bar(x - w / 2, pr, w, color=RF.SERIES[0], alpha=0.45, hatch="//", edgecolor=RF.SERIES[0], linewidth=1.0, label="simulator prediction")
    ax.bar(x + w / 2, ms, w, color=RF.SERIES[0], label="real GPU measurement")
    ax.axhline(1.0, color=RF.MUTED, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(list(tab.label))
    ax.set_ylabel("Total service time / E0   (lower is better)", color=RF.INK2, fontsize=10)
    ax.set_title("Scaled-down reproduction (empty compile cache)", color=RF.INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for tx in leg.get_texts():
        tx.set_color(RF.INK)
    ax.set_xlabel(f"{N_REQ} requests, shapes {', '.join(key_str(k) for k in SHAPES)}", color=RF.INK2, fontsize=9)
    txt = "dry-run: prediction only, nothing measured" if v is None else (
        f"max |pred-meas|/meas {v['C2_max_abs_rel_err']:.1%}" if v.get("C2_max_abs_rel_err") is not None else "no measurement")
    if v is not None and not v.get("formal", True):
        txt += "  (shared GPU0: development trial)"
    fig.text(0.01, 0.01, txt, fontsize=7.5, color=RF.INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for f in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + f), dpi=200, facecolor=RF.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------ main
def selftest() -> int:
    """CPU: fabricate metas for a fake run in a temp dir and check evaluate() / draw_quick() plumbing (no GPU, no repo output touched)."""
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(dir=TMP_ROOT, prefix="quick_selftest_"))
    try:
        (tmp / "serve").mkdir()
        (tmp / "micro").mkdir()
        pred = predict(build_trace(tmp))
        for r in pred.itertuples():
            (tmp / "serve" / f"{r.policy}_quick_trace_p0.meta.json").write_text(json.dumps(
                dict(status="ok", service_s=r.pred_total_s * 1.05, peak_reserved_mb=5000, pinned_fallback_count=0, event_counts={"steady": 1})))
        fz = frozen_steady()
        for arm in MICRO_ARMS:
            (tmp / "micro" / f"{arm}_seq_p0.meta.json").write_text(json.dumps(dict(per_key=[
                dict(key=f"{k[0]}x{k[1]}_b{k[2]}_nocfg", status="ok", steady_med_ms=fz[(arm, key_str(k))]["steady_ms"] * 1.03, first_req_s=40.0) for k in SHAPES])))
        walls = {f"{k}_{a}": dict(wall_s=60, rc=0) for k in ("micro", "serve") for a in ("x",)}
        walls["_total_wall_s"] = 600.0
        v = evaluate(tmp, pred, walls, shared=True)
        assert v["C2_pass"] and v["C3_pass"] and v["C1_pass"] and v["C4_pass"], v
        assert (tmp / "fig_quick.png").stat().st_size > 5000
        print("selftest OK", {k: v[k] for k in ("C1_pass", "C2_max_abs_rel_err", "C3_max_abs_rel_diff", "C4_pass")})
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="CPU only: trace + prediction + command plan + time estimate + prediction-only figure; no GPU process")
    ap.add_argument("--out", default=None, help=f"output directory (default {DEFAULT_OUT.relative_to(ROOT)}; dry-run: <out>_dryrun)")
    ap.add_argument("--shared", action="store_true", help="development trial on a GPU0 that is not idle (result flagged formal:false)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    out = pathlib.Path(a.out).resolve() if a.out else DEFAULT_OUT   # 11-24: absolute, sub-processes run in another cwd
    if a.dry_run:
        out = out.with_name(out.name + "_dryrun")
    out.mkdir(parents=True, exist_ok=True)
    trace = build_trace(out)
    pred = predict(trace)
    fz = frozen_steady()
    est = estimate_s(fz, pred)
    pred.to_csv(out / "quick_prediction.csv", index=False)
    print(pred[["label", "pred_excl_input_s", "pred_input_s", "pred_total_s", "n_input_extrapolated", "pred_event_counts"]].round(2).to_string(index=False))
    print("estimated wall (s): " + ", ".join(f"{k}={v:.0f}" for k, v in est.items()), f"  -> {est['total']/60:.1f} min (target <= 15)")
    if a.dry_run:
        cache_root = TMP_ROOT / "p24_quick_cache_DRYRUN_NOT_CREATED"
        plan = [dict(tag=t, argv=c) for t, c in commands(out, trace, cache_root, a.shared)]
        (out / "quick_plan.json").write_text(json.dumps(dict(plan=plan, estimate_s=est, gates="CUDA_VISIBLE_DEVICES == GPU0 UUID; MemAvailable >= 25 GB; load1 <= 16; GPU0 idle unless --shared",
                                                             env="P24_CACHE=<fresh temp dir under .software/tmp, deleted afterwards>, P24_PINNED_FALLBACK=1, taskset -c 0-15"), indent=1, ensure_ascii=False))
        draw_quick(pred.assign(meas_total_s=np.nan), out / "fig_quick", None)
        for t, c in commands(out, trace, cache_root, a.shared):
            print(f"[plan] {t}: {' '.join(c)}")
        print(f"dry-run done; nothing was run on a GPU. Files in {out}")
        return 0
    walls = run_real(out, trace, a.shared)
    v = evaluate(out, pred, walls, a.shared)
    print(json.dumps({k: v[k] for k in v if k != "walls"}, indent=1, default=str))
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
