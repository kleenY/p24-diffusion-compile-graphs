"""input_s micro-benchmark (cost_protocol_S2_v2.md change 3): request input preparation = tokenisation + host noise draw
(per-sample CPU generators) + copy to device, i.e. LCMPipe.make_inputs exactly as scripts/serve_trace.py calls it.

One process loads the pipeline once, then for every shape of the validation streams (V1 ∪ V2 ∪ V3 = 12 shapes) times
`--warm` + `--reps` calls with prompts taken in order from the V3 stream (real DiffusionDB prompts, half A) and
noise seeds as in serve_trace; each call is bracketed by torch.cuda.synchronize. Shapes are visited in a rotated order
(proc_id) so position effects average out. Output: <out>/input_p<id>.csv (one row per call) + .meta.json.
Run >= 5 processes; the simulator uses the per-shape median over processes of the per-process median (input_summary.csv
written by --summarize). Needs GPU0 (copy to device); about 3 GB device memory; ~1 min per process.

Usage (from code/): python -B -m scripts.bench_input --out ../results/S2/input_s --proc-id 0
                    python -B -m scripts.bench_input --out ../results/S2/input_s --summarize
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import time

KEYS = ["512x512x1x0", "512x768x1x0", "768x512x1x0", "512x512x1x1", "512x512x2x0", "512x512x3x0", "512x768x1x1",
        "512x768x2x0", "768x512x1x1", "768x512x2x0", "512x512x2x1", "512x512x4x0"]


def summarize(out: pathlib.Path):
    import pandas as pd
    d = pd.concat([pd.read_csv(f) for f in sorted(out.glob("input_p*.csv"))])
    d = d[d.phase == "rep"]
    per = d.groupby(["H", "W", "B", "cfg", "proc_id"]).input_ms.median().reset_index()
    s = per.groupby(["H", "W", "B", "cfg"]).input_ms.agg(n_proc="count", input_ms="median", min_ms="min",
                                                          max_ms="max").reset_index()
    s.to_csv(out / "input_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trace", default=None, help="prompt source (default: V3 validation stream)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.summarize:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    tag = out / f"input_p{a.proc_id}"
    if tag.with_suffix(".csv").exists():
        sys.exit(f"REFUSING to overwrite {tag}.csv")
    import torch
    from p24.bench import gpu0_other_procs, meta_snapshot
    from p24.pipeline import LCMPipe, ShapeKey
    from p24.sim.simulator import read_trace

    root = pathlib.Path(os.environ["PROJ_ROOT"])
    trace = pathlib.Path(a.trace or root / "results/S2/workload/validation_streams/V3_N200_seed2_A.jsonl")
    prompts = [r["prompt"] for r in read_trace(trace)]
    meta = meta_snapshot(a)
    pipe = LCMPipe(a.model_dir or str(root / "data/models/sd15_lcm_fused"))
    torch.cuda.synchronize()
    rot = a.proc_id % len(KEYS)
    keys = KEYS[rot:] + KEYS[:rot]
    rows, pi = [], 0
    for k in keys:
        H, W, B, cfg = map(int, k.split("x"))
        key = ShapeKey(H, W, B, bool(cfg))
        for i in range(a.warm + a.reps):
            ps = [prompts[(pi + j) % len(prompts)] for j in range(B)]
            pi += B
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inp = pipe.make_inputs(ps, key, seeds=[1000 * pi + j for j in range(B)])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            del inp
            rows.append(dict(proc_id=a.proc_id, H=H, W=W, B=B, cfg=cfg, i=i, phase="warm" if i < a.warm else "rep",
                             input_ms=(t1 - t0) * 1e3))
    with open(tag.with_suffix(".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta.update(trace=str(trace), keys_order=keys, gpu0_other_procs_end=gpu0_other_procs(), loadavg_end=os.getloadavg())
    tag.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    med = {k: statistics.median(r["input_ms"] for r in rows if r["phase"] == "rep" and
                                f"{r['H']}x{r['W']}x{r['B']}x{r['cfg']}" == k) for k in keys}
    print(json.dumps({k: round(v, 3) for k, v in med.items()}))


if __name__ == "__main__":
    main()
