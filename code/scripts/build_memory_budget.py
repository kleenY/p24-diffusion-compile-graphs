"""S2 step 2.2 (closes L5): memory budget for the bucket cache, from the merged cost table (CPU only, no torch).

Inputs : results/S2/cost_table/merged/cost_table_procs.csv (one row per process x key, all blocks S-1..S-5)
         results/S2/inproc_seq/v3shapes_G1_disk/*.csv (one process accumulating G1 graph sets: additivity check)
Output : results/S2/memory_budget.json  (+ memory_budget_buckets.csv)

Definitions (all MiB; torch numbers are allocator numbers, device numbers come from cudaMemGetInfo):
  resident_mb      device used after model load (weights + CUDA context + allocator slack), mem_get_info, GPU0 exclusive
  weights_mb       torch.cuda.memory_allocated() after load
  eager_ws_mb      E0 steady max_memory_reserved - weights_mb  (eager workspace a fallback request needs on top)
  pool_mb          capture_pool_mb of one bucket (reserved growth during capture, G1 / C1+G1)
  static_io_mb     static input/output buffers of one bucket
Formula (plan 2.2):
  graph_budget_mb = cap_mb - resident_mb - max_eager_ws_mb - margin_mb
  margin_mb       = accounting_error_bound_mb = max(2 x cross-process fluctuation of pool_mb, 64)
  admit a new bucket iff  sum(pool_mb + overhead over live buckets) + pool_mb(new) + overhead <= graph_budget_mb
  overhead = per-bucket allocator overhead fitted on the S-5 G1 sequence (additivity_check)
"""
from __future__ import annotations

import glob
import hashlib
import json
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "S2"
CAP_MB = 20 * 1024          # project rule: all concurrent jobs <= ~20 GB on GPU0 (device 23.65 GiB, 4090)
DEVICE_TOTAL_MB = 23.65 * 1024
KEY = ["H", "W", "B", "cfg"]


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    src = R / "cost_table" / "merged" / "cost_table_procs.csv"
    p = pd.read_csv(src)
    p = p[p.steady_ms.notna()]
    resident = float(p.resident_mb.median()); weights = float(p.weights_alloc_mb.median())
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": str(src.relative_to(ROOT)), "source_sha16": sha16(src),
           "units": "MiB", "cap_mb": CAP_MB, "device_total_mb": round(DEVICE_TOTAL_MB),
           "resident_mb": resident, "resident_spread_mb": float(p.resident_mb.max() - p.resident_mb.min()),
           "weights_mb": weights, "context_and_slack_mb": resident - weights}

    # eager workspace per shape (E0, all processes incl. in-process rows; workload grid = 4 resolutions x B1-4 x cfg)
    e = p[p.arm == "E0"].copy(); e["eager_ws_mb"] = e.steady_peak_reserved_mb - e.weights_alloc_mb
    ews = e.groupby(KEY).eager_ws_mb.agg(["median", "min", "max", "count"]).reset_index()
    wl = ews[ews.H.isin([512, 768]) & ews.W.isin([512, 768])]
    imax = wl["max"].idxmax()
    out["eager_ws"] = {"max_over_workload_shapes_mb": float(wl.loc[imax, "max"]),
                       "argmax_shape": {k: int(wl.loc[imax, k]) for k in KEY},
                       "n_shapes": int(len(wl))}

    # graph pools per bucket
    rows = []
    for arm in ("G1", "C1+G1"):
        g = p[(p.arm == arm) & p.capture_pool_mb.notna()]
        for k, d in g.groupby(KEY):
            rows.append(dict(arm=arm, **dict(zip(KEY, map(int, k))), n=len(d), pool_median=d.capture_pool_mb.median(),
                             pool_min=d.capture_pool_mb.min(), pool_max=d.capture_pool_mb.max(),
                             pool_spread=d.capture_pool_mb.max() - d.capture_pool_mb.min(),
                             static_io_median=d.static_io_mb.median()))
    b = pd.DataFrame(rows)
    b.to_csv(R / "memory_budget_buckets.csv", index=False)
    fluct = float(b.pool_spread.max())
    acc = max(2 * fluct, 64.0)
    out["capture_pool"] = {"n_buckets": int(len(b)), "max_cross_process_spread_mb": fluct,
                           "median_cross_process_spread_mb": float(b.pool_spread.median()),
                           "argmax_spread": b.loc[b.pool_spread.idxmax(), ["arm"] + KEY].astype(str).to_dict(),
                           "table": "results/S2/memory_budget_buckets.csv"}
    out["accounting_error_bound_mb"] = acc
    out["accounting_error_bound_rule"] = "max(2 x max cross-process spread of capture_pool_mb, 64 MB) (plan 2.2; proposal for S3.1)"
    budget = CAP_MB - resident - out["eager_ws"]["max_over_workload_shapes_mb"] - acc
    out["graph_budget_mb"] = round(budget, 1)
    out["graph_budget_formula"] = "cap_mb - resident_mb - eager_ws.max_over_workload_shapes_mb - accounting_error_bound_mb"

    # additivity check: one G1 process accumulating buckets (S-5 v3 sequence; no eviction).
    # residual(n) = reserved - (weights + cumsum(pool)) is fitted as base + per_bucket * n (n = buckets alive)
    fits, maxres = [], []
    for pid in (0, 1, 2):
        fs = sorted(glob.glob(str(R / "inproc_seq" / "v3shapes_G1_disk" / f"G1_seq*_p{pid}.csv")))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs]).groupby("seq_idx").agg(
            pool=("capture_pool_mb", "max"), res=("steady_peak_reserved_mb", "max")).dropna()
        resid = d.res - (weights + d.pool.cumsum())
        n = np.arange(1, len(d) + 1)
        slope, icpt = np.polyfit(n, resid, 1)
        fits.append((slope, icpt)); maxres.append(float(np.abs(resid - (icpt + slope * n)).max()))
    per_bucket = float(np.median([f[0] for f in fits])); base = float(np.median([f[1] for f in fits]))
    out["additivity_check"] = {"what": "G1 one process, 9 buckets accumulated: reserved - (weights + cumsum(pool)) = base + per_bucket*n",
                               "per_bucket_overhead_mb": round(per_bucket, 2), "base_overhead_mb": round(base, 2),
                               "max_abs_err_of_linear_model_mb_per_process": maxres,
                               "source": "results/S2/inproc_seq/v3shapes_G1_disk"}
    out["admission_rule"] = ("admit new bucket iff sum over live buckets of (pool_mb + per_bucket_overhead_mb) + "
                             "pool_mb(new) + per_bucket_overhead_mb <= graph_budget_mb")

    # how many buckets fit (illustration for S3): G1 vs C1+G1 at 512x512 B1 no-CFG and 768x768 B4 CFG
    fit = {}
    for arm in ("G1", "C1+G1"):
        for k in [(512, 512, 1, 0), (512, 512, 4, 0), (768, 768, 4, 1)]:
            r = b[(b.arm == arm) & (b.H == k[0]) & (b.W == k[1]) & (b.B == k[2]) & (b.cfg == k[3])]
            if len(r):
                per = float(r.pool_median.iloc[0]) + per_bucket
                fit[f"{arm}_{k[0]}x{k[1]}_b{k[2]}_cfg{k[3]}"] = {"per_bucket_mb": per, "n_fit": int(budget // per)}
    out["buckets_that_fit"] = fit
    out["evidence_unbounded_g1"] = ("S-5: G1 without budget OOMs at the 10th V3 shape (19.24 GiB private pools); "
                                    "failure catalogue #19")
    (R / "memory_budget.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("resident_mb", "weights_mb", "eager_ws", "accounting_error_bound_mb",
                                          "graph_budget_mb", "additivity_check", "buckets_that_fit")}, indent=1))


if __name__ == "__main__":
    main()
