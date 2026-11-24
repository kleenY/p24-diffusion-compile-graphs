"""S4-P P5c SELFTEST ONLY: SYNTHETIC inputs for the P5c script chain (CPU, no torch).  NOTHING produced here is a measurement; every file name carries SYNTHETIC; the generator refuses
to write anywhere under results/ (use tmp/p5c_selftest/).  Real block-A / block-C data and evaluation seeds 100-129 are never read; the only real inputs are the DRY-RUN cost table
(results/S4P/p3/cost_v2_dryrun/cost_table_v2.csv, task-provided) and the DEVELOPMENT traces (seeds 0-4, half A), which are copied under eval-style names ONLY inside the temp root.
Layouts written under <root>:
  measure_A_aot/{runs_seq,runs_cold,build,runs_c1build}/        same JSON schema as window_S4P_A_aot.sh (AOTW_*.json per_key, build_*.json)
  prebuild/build/build_<tag>_SYNTHETIC.json                      32 synthetic prebuild build records (build_s_sum_comps)
  aot_fallback_shapes_SYNTHETIC.csv                              key,reason,detail (one fallback shape)
  eval_like/eval_traces/<W>_N<N>_seed<s>_B.jsonl + sha256_manifest_eval_SYNTHETIC.txt (0444)   dev traces re-labelled as seeds 120-129 (stand-ins to exercise the eval code path)
  streams/<W>_N300_seed101_B.jsonl                               first 300 requests of dev traces (stand-ins for the block-C streams)
  blockC/{raw,dryrun,logs}/                                      synthetic block-C processes (B0, B1, B1-AOT x 3 repeats x 3 streams), dry-run decision CSVs, block log
Usage (from code/): python -B -m scripts.synth_S4P_P5c --root ../tmp/p5c_selftest [--k-ratio 1.0] [--truncate-build] [--aot-ratio 0.35]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import aot_tables_S4P_P5c as T

ROOT = T.ROOT
DRY_TABLE = ROOT / "results/S4P/p3/cost_v2_dryrun/cost_table_v2.csv"
DEV_TRACES = ROOT / "results/S3/workload/dev_traces"
RES = [(512, 512), (512, 768), (768, 512), (768, 768)]
SHAPES = [T.Shape(h, w, b, c) for (h, w) in RES for b in (1, 2, 3, 4) for c in (0, 1)]
COLD7 = [T.Shape(512, 512, 1, 0), T.Shape(512, 512, 2, 1), T.Shape(512, 768, 3, 0), T.Shape(768, 512, 1, 1), T.Shape(768, 768, 1, 0), T.Shape(768, 768, 4, 1), T.Shape(512, 768, 2, 0)]
FALLBACK = T.Shape(512, 512, 3, 0)
CAL = [T.Shape(512, 512, 1, 0), T.Shape(768, 768, 4, 1)]


def guard_out(root: pathlib.Path) -> None:
    assert not T.inside(root, ROOT / "results"), f"SYNTHETIC data must not be written under results/: {root}"


def _wjson(p: pathlib.Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


_DRY = {}


def c1_steady_ms(sh: T.Shape) -> float:
    if "d" not in _DRY:
        _DRY["d"] = pd.read_csv(DRY_TABLE)
    d = _DRY["d"]
    r = d[(d.arm == "C1") & (d.H == sh.H) & (d.W == sh.W) & (d.B == sh.B) & (d.cfg == sh.cfg)]
    return float(r.steady_ms.iloc[0])


def synth_model(sh: T.Shape, rng: np.random.Generator):
    """Deterministic truth + small noise: steady = 0.995 x C1 steady; one-time inproc = 0.30 + 0.05 x batch; cold adds 0.17 s; build wall = 55 + 3 x batch s."""
    st = c1_steady_ms(sh) * 0.995
    bt = sh.B * (2 if sh.cfg else 1)
    ot_in = 0.30 + 0.05 * bt + float(rng.normal(0, 0.005))
    return st, ot_in, ot_in + 0.17 + float(rng.normal(0, 0.005)), 55.0 + 3.0 * bt


def _comps(w: float) -> dict:
    """three components whose export + AOTI-compile seconds sum to w (te 15 %, unet 65 %, vae 20 %)."""
    return {c: dict(status="ok", export_s=0.02 * f * w, aoti_compile_s=0.98 * f * w, package_bytes=int(1e7 * f)) for c, f in (("te", 0.15), ("unet", 0.65), ("vae", 0.2))}


def pk(sh: T.Shape, first_req_s: float, steady_ms: float, rng) -> dict:
    return dict(key=sh.tag(), status="ok", first_req_s=first_req_s, steady_n=10, steady_wall_median_ms=steady_ms, steady_peak_reserved_mb=2700.0 + float(rng.normal(0, 1)),
                prepare=dict(load_s=0.05, first_gen_s=first_req_s - 0.1), comp_median_ms={})


def proc_json(tag: str, pid: int, keys: List[dict], **kw) -> dict:
    j = dict(tag=tag, arm="AOTW", proc_id=pid, status="ok", per_key=keys, gpu0_other_procs_start=0, gpu0_other_procs_end=0, loadavg_start=[1.0, 1.0, 1.0], reps=10)
    j.update(kw)
    return j


def make_measure(root: pathlib.Path, k_ratio: float = 1.0, truncate_build: bool = False, seed: int = 0) -> dict:
    guard_out(root)
    rng = np.random.default_rng(seed)
    meas = root / "measure_A_aot"
    model = {sh: synth_model(sh, rng) for sh in SHAPES}
    firsts = [T.Shape(512, 512, 2, 0), T.Shape(512, 768, 1, 0), FALLBACK]
    for p in range(3):
        order = [s for s in SHAPES if s != firsts[p]]
        rng.shuffle(order)
        order = [firsts[p]] + order
        keys = []
        for i, sh in enumerate(order):
            st, oi, oc, _ = model[sh]
            ot = oc if i == 0 else oi + float(rng.normal(0, 0.003))
            keys.append(pk(sh, st / 1e3 + ot, st * (1 + float(rng.normal(0, 0.001))), rng))
        _wjson(meas / "runs_seq" / f"AOTW_seq32_p{p}_SYNTHETIC.json", proc_json("AOTW_seq32", p, keys))
    for r in range(3):
        for sh in COLD7:
            st, oi, oc, _ = model[sh]
            _wjson(meas / "runs_cold" / f"AOTW_{sh.tag()}_p{r}_SYNTHETIC.json", proc_json(f"AOTW_{sh.tag()}", r, [pk(sh, st / 1e3 + oc + float(rng.normal(0, 0.004)), st, rng)]))
    pre = {}
    for sh in SHAPES:
        w = model[sh][3]
        pre[sh] = w
        comps = _comps(w)
        total = sum(c["export_s"] + c["aoti_compile_s"] for c in comps.values())
        _wjson(root / "prebuild/build" / f"build_{sh.tag()}_SYNTHETIC.json", dict(key=sh.tag(), status="ok", build_s_sum_comps=total, total_s=total + 1.0, comps=comps,
                                                                                  gpu0_other_procs_start=0, gpu0_other_procs_end=0))
    cal_keys = CAL[:1] if truncate_build else CAL
    for sh in cal_keys:
        w = pre[sh] * k_ratio
        comps = _comps(w)
        total = sum(c["export_s"] + c["aoti_compile_s"] for c in comps.values())
        _wjson(meas / "build" / f"build_{sh.tag()}_SYNTHETIC.json", dict(key=sh.tag(), status="ok", build_s_sum_comps=total, total_s=total + 1.0, comps=comps))
    for i, sh in enumerate(CAL):
        _wjson(meas / "runs_c1build" / f"C1_seq1_p{100 + i}_SYNTHETIC.json", proc_json("C1_seq1", 100 + i, [pk(sh, 30.0 + i, c1_steady_ms(sh), rng)]))
    fb = root / "aot_fallback_shapes_SYNTHETIC.csv"
    fb.write_text("key,reason,detail\n" + f"{FALLBACK.H}x{FALLBACK.W}x{FALLBACK.B}x{FALLBACK.cfg},equiv_fail_frozen,SYNTHETIC\n")
    return dict(model={s.tag(): v for s, v in model.items()}, prebuild={s.tag(): w for s, w in pre.items()})


def make_eval_like(root: pathlib.Path, workloads=("W2", "W5"), Ns=(1000, 10000), seeds=tuple(range(120, 130))) -> pathlib.Path:
    """Dev traces (A half, seeds 0-4) copied as `<W>_N<N>_seed<120+i>_B.jsonl` (seed s -> dev seed s%5): exercises the eval path of sim_h6 (manifest, 0444, SHA check)."""
    guard_out(root)
    d = root / "eval_like"
    man = []
    for w in workloads:
        for n in Ns:
            for s in seeds:
                src = DEV_TRACES / f"{w}_N{n}_seed{s % 5}_A.jsonl"
                dst = d / "eval_traces" / f"{w}_N{n}_seed{s}_B.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    os.chmod(dst, 0o644)
                shutil.copyfile(src, dst)
                os.chmod(dst, 0o444)
                man.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  eval_traces/{dst.name}\n")
    mp = d / "sha256_manifest_eval_SYNTHETIC.txt"
    if mp.exists():
        os.chmod(mp, 0o644)
    mp.write_text("".join(sorted(man, key=lambda l: l.split()[1])))
    os.chmod(mp, 0o444)
    return mp


def make_streams(root: pathlib.Path) -> pathlib.Path:
    guard_out(root)
    d = root / "streams"
    d.mkdir(parents=True, exist_ok=True)
    for w in ("W2", "W3", "W5"):
        lines = open(DEV_TRACES / f"{w}_N1000_seed1_A.jsonl").read().splitlines()[:300]
        (d / f"{w}_N300_seed101_B.jsonl").write_text("\n".join(lines) + "\n")
    return d


def make_blockC(root: pathlib.Path, aot_ratio: float = 0.35, flag_b1aot: bool = False, bad_dynamo: bool = False) -> pathlib.Path:
    """Synthetic block-C processes.  Per request: steady = 0.995..1.0 x C1 steady; first request of a new shape: B0 0.3 s, B1 6.0 s, B1-AOT 0.4 s (fallback shape 0.45 s, served_by E0).
    aot_ratio scales the TOTAL of B1-AOT relative to B1 only through these numbers (the verdict is computed from the files, not from this parameter)."""
    guard_out(root)
    stream_dir = make_streams(root)
    raw, dry, logs = root / "blockC/raw", root / "blockC/dryrun", root / "blockC/logs"
    for d in (raw, dry, logs):
        d.mkdir(parents=True, exist_ok=True)
    steps = []
    k = 0
    for p in range(3):
        for w in ("W2", "W3", "W5"):
            stem = f"{w}_N300_seed101_B"
            reqs = [json.loads(l) for l in open(stream_dir / f"{stem}.jsonl")]
            for pol in ("B0", "B1", "B1-AOT"):
                rows, t, seen = [], 0.0, set()
                for i, q in enumerate(reqs):
                    sh = T.Shape(int(q["H"]), int(q["W"]), int(q["B"]), int(q["cfg"]))
                    st = c1_steady_ms(sh) * (1.0 if pol != "B1-AOT" or sh == FALLBACK else 0.995)
                    new = sh not in seen
                    seen.add(sh)
                    ev = "steady" if (not new or pol == "B0") else "compile"
                    if pol == "B0":
                        ev = "eager_first" if new else "steady"
                    first = 0.0
                    if new:
                        first = {"B0": 0.3, "B1": 6.0, "B1-AOT": 0.45 if sh == FALLBACK else 0.4}[pol] * (1 + 0.02 * p)
                    prep_ms = first * 1e3 * 0.9 if new else 0.0
                    run_ms = st + (first * 1e3 * 0.1 if new else 0.0)
                    inp = 5.0
                    dur = prep_ms + run_ms + inp
                    served = "E0" if (pol == "B0" or (pol == "B1-AOT" and sh == FALLBACK)) else "C1"
                    rows.append(dict(idx=i, req_id=f"r{i}", H=sh.H, W=sh.W, B=sh.B, cfg=sh.cfg, action="eager" if pol == "B0" else "compile", event=ev, status="ok",
                                     t_start_s=t, t_end_s=t + dur / 1e3, input_ms=inp, prepare_ms=prep_ms, run_ms=run_ms, dur_ms=dur, phase="req", evicted="", n_evicted=0,
                                     served_by=served, note="aot_fallback" if (pol == "B1-AOT" and sh == FALLBACK) else "",
                                     prep_json=json.dumps({"aot_fallback": 1}) if (new and pol == "B1-AOT" and sh == FALLBACK) else "{}"))
                    t += dur / 1e3
                tag = f"{pol}_{stem}_p{p}"
                with open(raw / f"{tag}.csv", "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
                evc = {}
                for r in rows:
                    evc[r["event"]] = evc.get(r["event"], 0) + 1
                meta = dict(trace=str(stream_dir / f"{stem}.jsonl"), policy=pol, cache_mode="warm", status="ok", service_s=t, service_s_excl_input=t - 5.0 * len(rows) / 1e3, input_s_total=5.0 * len(rows) / 1e3,
                            total_incl_load_s=t + 0.5, load_s=0.5, peak_reserved_mb=8000.0, event_counts=evc, pinned_fallback_count=0, n_requests=len(rows))
                if pol == "B1-AOT":
                    meta["aot"] = dict(dynamo_compile_events=3 if bad_dynamo else 0, rt_cache_files_after=0, pkg_manifest_mismatch=[], pkg_manifest="m.json")
                    meta["aot_fallback_requests"] = sum(1 for r in rows if r["note"])
                    meta["aot_fallback_shapes"] = [f"{FALLBACK.H}x{FALLBACK.W}x{FALLBACK.B}x{FALLBACK.cfg}"]
                (raw / f"{tag}.meta.json").write_text(json.dumps(meta))
                k += 1
                steps.append(f"STEP [{k}/27] {pol}_{w}_warm_p{p} rc=0 elapsed={k * 100}s")
                if flag_b1aot and pol == "B1-AOT" and p == 0 and w == "W2":
                    steps.insert(len(steps) - 1, "NOT_QUIET load1=17.2>16")
            if p == 0:
                for pol in ("B1", "B1-AOT"):
                    dec = pd.DataFrame([dict(idx=i, action="compile", event="compile" if i < 3 else "steady", evicted="") for i in range(10)])
                    dec.to_csv(dry / f"{pol}_{stem}_p0.dryrun_decisions.csv", index=False)
    (logs / "block_Cv2_ALL_SYNTHETIC.log").write_text("\n".join(steps) + "\n")
    return root / "blockC"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--k-ratio", type=float, default=1.0, help="exclusive / prebuild build wall-clock of the 2 calibration shapes")
    ap.add_argument("--truncate-build", action="store_true")
    ap.add_argument("--aot-ratio", type=float, default=0.35)
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    guard_out(root)
    make_measure(root, a.k_ratio, a.truncate_build)
    make_eval_like(root)
    make_blockC(root, a.aot_ratio)
    print("SYNTHETIC inputs written under", root)


if __name__ == "__main__":
    main()
