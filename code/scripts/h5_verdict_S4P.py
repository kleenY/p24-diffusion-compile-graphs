"""S4-P P1: H5a / H5b / H5c verdict tables from a sim_eval_S4P raw csv (CPU, seconds).  Thresholds are those of goals/S4P_uplift_line_prereg_v1.md:
  H5a (not worse)      W2, N=1e4, disk level: per-seed v3/B1 <= 1.01 in >= 8/10 seeds
  H5b (worst case)     over the table (5 workloads x 3 cold levels x 4 N; cell = mean over seeds): max(v3/B1) <= 1.02 AND max(v3/best online) <= 1.03
  H5c (better, bonus)  W2, N=1e4, disk: per-seed v3/B1 < 1 in >= 8/10 seeds; mean ratio + percentile bootstrap CI over seeds (2000 resamples, rng 0)
  also reported: v3 relative to B1 for small N / cold start, and relative to S3 Ours v2 ("Ours").
Best online baseline = B1,B2,B3,B3p,B3pp,B4,B5,B6,B6p as scripts/h4_verdict_S3.py (B0 = plain eager, B7 = oracle and Ours-NE are not online baselines).
With fewer than 10 seeds (dev data: 5) the "8/10" counts are reported as k/n and the verdict is labelled 'dev-style' (no pass/fail).
Usage: python code/scripts/h5_verdict_S4P.py --raw main_raw.csv --v3 Ours-v3 [--out verdict.json]   (importable: verdict(df, v3_label))
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

ONLINE = ["B1", "B2", "B3", "B3p", "B3pp", "B4", "B5", "B6", "B6p"]
COLDS = ["empty", "disk", "hot"]


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.status == "ok"]
    return d.pivot_table(index=["workload", "cold", "N"], columns="variant", values="service_s", aggfunc="mean")


def cell_ratios(df: pd.DataFrame, v3: str) -> pd.DataFrame:
    cm = cell_means(df)
    out = pd.DataFrame(index=cm.index)
    out["v3_s"] = cm[v3]
    out["v3/B1"] = cm[v3] / cm["B1"]
    out["v3/B0"] = cm[v3] / cm["B0"]
    if "Ours" in cm:
        out["v3/v2"] = cm[v3] / cm["Ours"]
    on = [c for c in ONLINE if c in cm]
    out["best_online"] = cm[on].idxmin(axis=1)
    out["v3/best_online"] = cm[v3] / cm[on].min(axis=1)
    return out


def _seed_ratios(df: pd.DataFrame, v3: str, ref: str, w="W2", n=10000, cold="disk") -> pd.Series:
    d = df[(df.status == "ok") & (df.workload == w) & (df.N == n) & (df.cold == cold)]
    p = d.pivot_table(index="seed", columns="variant", values="service_s", aggfunc="mean")
    return (p[v3] / p[ref]).sort_index()


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def verdict(df: pd.DataFrame, v3: str, w="W2", n=10000, cold="disk") -> dict:
    r = _seed_ratios(df, v3, "B1", w, n, cold)
    ns = len(r)
    full = ns >= 10
    k_a, k_c = int((r <= 1.01).sum()), int((r < 1.0).sum())
    need = math.ceil(0.8 * ns)
    ct = cell_ratios(df, v3)
    worst_b1, worst_best = ct["v3/B1"].idxmax(), ct["v3/best_online"].idxmax()
    ci = boot_ci(r.to_numpy()) if ns > 1 else (float("nan"), float("nan"))
    r2 = _seed_ratios(df, v3, "Ours", w, n, cold) if "Ours" in set(df.variant) else None
    out = dict(
        v3=v3, n_seeds=ns, seeds=[int(s) for s in r.index],
        H5a=dict(per_seed_v3_over_B1=[float(x) for x in r], seeds_le_1p01=k_a, need=need, mean_ratio=float(r.mean()),
                 verdict=("supported" if k_a >= need else "not supported") if full else f"dev-style {k_a}/{ns}"),
        H5c=dict(seeds_lt_1=k_c, need=need, mean_ratio=float(r.mean()), mean_gain_pct=float(100 * (1 - r.mean())), boot_ci95_mean_ratio=ci,
                 verdict=("supported" if k_c >= need else "not supported") if full else f"dev-style {k_c}/{ns}"),
        H5b=dict(max_v3_over_B1=float(ct["v3/B1"].max()), cell_max_B1=[str(c) for c in worst_b1],
                 max_v3_over_best_online=float(ct["v3/best_online"].max()), cell_max_best=[str(c) for c in worst_best],
                 best_online_at_cell_max=str(ct.loc[worst_best, "best_online"]),
                 pass_B1=bool(ct["v3/B1"].max() <= 1.02), pass_best=bool(ct["v3/best_online"].max() <= 1.03),
                 verdict=("supported" if (ct["v3/B1"].max() <= 1.02 and ct["v3/best_online"].max() <= 1.03) else "not supported") if full else
                 f"dev-style: max v3/B1 {ct['v3/B1'].max():.3f}, max v3/best {ct['v3/best_online'].max():.3f}"),
        vs_v2_W2_1e4_disk=None if r2 is None else dict(per_seed=[float(x) for x in r2], mean=float(r2.mean())),
    )
    return out


def cells_table(df: pd.DataFrame, v3: str) -> pd.DataFrame:
    return cell_ratios(df, v3).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--v3", default="Ours-v3")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.read_csv(a.raw)
    v = verdict(df, a.v3)
    print(json.dumps(v, indent=1, ensure_ascii=False))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(v, f, indent=1, ensure_ascii=False)
        cells_table(df, a.v3).to_csv(a.out.replace(".json", "_cells.csv"), index=False)


if __name__ == "__main__":
    main()
