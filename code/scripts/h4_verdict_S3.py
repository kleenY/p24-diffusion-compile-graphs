"""S3 H4 verdict (goals/S3_detailed_goals_v2.md section G items 17-18). Reads the raw CSV of `sim_eval_S3.py --experiment main` (and optionally the
bootstrap raw CSV) and writes h4_verdict.json. No simulation is run here.

Rule (written before any evaluation number was seen):
  * Condition: default budget, cold level = disk cache, primary N = 1e4 (also reported for N = 1e3), 10 evaluation seeds.
  * "Best online baseline": ONE policy chosen among B1, B2, B3 (truncated = budget-limited simulation), B3', B3'', B4 (truncated), B5, B6, B6'
    by the lowest MEAN total service time over the seeds on W2 at that N (a baseline with any non-ok seed is excluded and listed). B0 (E0),
    B7 (oracle) and Ours-NE are not online baselines. Then compare PER SEED.
  * H4a (W2): Ours lower than the best online baseline by >= 2% (Ours <= 0.98 x best) in >= 8/10 seeds -> supported; Ours higher by >= 2%
    (Ours >= 1.02 x best) in >= 8/10 seeds -> refuted; otherwise undetermined. Bootstrap label (reported, does NOT change the verdict): the
    95% percentile interval of mean(Ours)/mean(best) - 1 over the cost-table bootstrap replicates excludes 0 -> "robust".
  * H4b (W3): Ours <= 1.02 x B0 in >= 8/10 seeds -> supported, else refuted.
  * H4c (W1): Ours <= 1.02 x B1 in >= 8/10 seeds -> supported, else refuted.
  * Also reported: Ours / B7 (mean over seeds).
With a seed count other than 10 (development test) the threshold is ceil(0.8 n) and the output is flagged test_only.

Usage: python code/scripts/h4_verdict_S3.py --raw results/S3/sim_eval/main_raw.csv [--bootstrap results/S3/sim_eval/bootstrap_raw.csv] --out results/S3/sim_eval/h4_verdict.json
       python code/scripts/h4_verdict_S3.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

ONLINE = ["B1", "B2", "B3", "B3p", "B3pp", "B4", "B5", "B6", "B6p"]
TOL = 0.02
FRAC = 0.8
ALIAS = {"B3p": "B3'", "B3pp": "B3''", "B6p": "B6'"}


def thr(n_seeds: int) -> int:
    return 8 if n_seeds == 10 else int(math.ceil(FRAC * n_seeds - 1e-9))


def wide(d: pd.DataFrame, workload: str, n: int, cold: str, budget: str) -> pd.DataFrame:
    """service_s per (seed x variant) for rows with status ok; variants with any non-ok seed are marked NaN."""
    s = d[(d.workload == workload) & (d.N == n) & (d.cold == cold) & (d.budget == budget) & (d.boot == -1) & (d.prior_kind == "A")]
    if "trace_kind" in s:
        s = s[s.trace_kind.isin(["file", "gen_default"])]
    s = s.assign(service_s=np.where(s.status == "ok", s.service_s, np.nan))
    return s.pivot_table(index="seed", columns="variant", values="service_s", aggfunc="first", dropna=False)


def best_online(w: pd.DataFrame):
    cands, excluded = {}, {}
    for b in ONLINE:
        if b not in w:
            excluded[b] = "not in the raw CSV"
        elif w[b].isna().any():
            excluded[b] = f"{int(w[b].isna().sum())} seed(s) without a value (missing cost / error)"
        else:
            cands[b] = float(w[b].mean())
    best = min(cands, key=cands.get)
    return best, cands, excluded


def seeds_count(mask) -> int:
    return int(np.sum(np.asarray(mask)))


def h4a(w: pd.DataFrame, nseeds: int):
    best, cands, excl = best_online(w)
    rel = (w["Ours"] / w[best] - 1.0).to_numpy()
    lower, higher = seeds_count(rel <= -TOL + 1e-12), seeds_count(rel >= TOL - 1e-12)
    t = thr(nseeds)
    verdict = "supported" if lower >= t else ("refuted" if higher >= t else "undetermined")
    return dict(best_online=best, candidate_means_s=cands, excluded=excl, per_seed_rel_diff=[float(x) for x in rel],
                seeds_ours_lower_by_2pct=lower, seeds_ours_higher_by_2pct=higher, threshold_seeds=t, n_seeds=nseeds,
                mean_ours_s=float(w["Ours"].mean()), mean_best_s=float(w[best].mean()),
                mean_rel_diff=float(w["Ours"].mean() / w[best].mean() - 1), verdict=verdict,
                ours_over_B7_mean=float(w["Ours"].mean() / w["B7"].mean()) if "B7" in w and w["B7"].notna().all() else None)


def h4_ratio(w: pd.DataFrame, ref: str, nseeds: int):
    rel = (w["Ours"] / w[ref]).to_numpy()
    ok = seeds_count(rel <= 1 + TOL + 1e-12)
    t = thr(nseeds)
    return dict(reference=ref, per_seed_ratio=[float(x) for x in rel], seeds_within_1p02=ok, threshold_seeds=t, n_seeds=nseeds,
                mean_ratio=float(np.mean(rel)), verdict="supported" if ok >= t else "refuted",
                ours_over_B7_mean=float(w["Ours"].mean() / w["B7"].mean()) if "B7" in w and w["B7"].notna().all() else None)


def bootstrap_label(bd: pd.DataFrame, workload: str, n: int, cold: str, budget: str, best: str, ref_kind: str, point_sign: float):
    """95% percentile interval over replicates of mean(Ours)/mean(ref) - 1 (ref = the best online baseline fixed by the point estimate)."""
    s = bd[(bd.workload == workload) & (bd.N == n) & (bd.cold == cold) & (bd.budget == budget) & (bd.boot >= 0) & (bd.status == "ok")]
    if s.empty:
        return None
    rows = []
    for b, g in s.groupby("boot"):
        p = g.pivot_table(index="seed", columns="variant", values="service_s", aggfunc="first")
        if "Ours" not in p or best not in p:
            continue
        rows.append(dict(boot=b, rel=float(p["Ours"].mean() / p[best].mean() - 1),
                         best_identity=min([c for c in ONLINE if c in p and p[c].notna().all()], key=lambda c: p[c].mean())))
    r = pd.DataFrame(rows)
    lo, hi = np.percentile(r.rel, [2.5, 97.5])
    return dict(n_boot=int(len(r)), rel_diff_mean=float(r.rel.mean()), ci95=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0),
                label="robust" if (lo > 0 or hi < 0) else "not robust", frac_ours_lower=float((r.rel < 0).mean()),
                frac_sign_same_as_point=float((np.sign(r.rel) == point_sign).mean()),
                frac_best_identity_same=float((r.best_identity == best).mean()))


def verdict(raw: pd.DataFrame, boot: pd.DataFrame | None, cold="disk", budget="default"):
    seeds = sorted(raw.seed.unique())
    n = len(seeds)
    split = str(raw.split.iloc[0]) if "split" in raw else "?"
    out = dict(rule_version="G18 (S3_detailed_goals_v2.md), thresholds fixed before evaluation", cold=cold, budget=budget, seeds=[int(s) for s in seeds],
               n_seeds=n, split=split, test_only=bool(n != 10 or split != "eval"), tolerance=TOL, per_N={})
    for N in (10000, 1000):
        res = {}
        w2, w3, w1 = (wide(raw, w, N, cold, budget) for w in ("W2", "W3", "W1"))
        if len(w2):
            res["H4a_W2"] = h4a(w2, n)
        if len(w3) and "B0" in w3:
            res["H4b_W3"] = h4_ratio(w3, "B0", n)
        if len(w1) and "B1" in w1 and w1["B1"].notna().all():
            res["H4c_W1"] = h4_ratio(w1, "B1", n)
        if boot is not None and "H4a_W2" in res:
            a = res["H4a_W2"]
            res["H4a_bootstrap"] = bootstrap_label(boot, "W2", N, cold, budget, a["best_online"], "best", float(np.sign(a["mean_rel_diff"])))
        out["per_N"][str(N)] = res
    out["primary_N"] = 10000
    out["primary"] = {k: v["verdict"] for k, v in out["per_N"].get("10000", {}).items() if isinstance(v, dict) and "verdict" in v}
    return out


def _synthetic(ours_rel, n=10, b0_ratio=1.0):
    """Synthetic raw frame for the self-test: W2 best online baseline B1 = 100; Ours = 100*(1+rel[i]); other baselines worse."""
    rows = []
    for w in ("W1", "W2", "W3"):
        for s in range(n):
            vals = {"B0": 120.0, "B7": 90.0, "B1": 100.0, "B2": 110.0, "B3": 112.0, "B3p": 111.0, "B3pp": 115.0, "B4": 119.0, "B5": 105.0,
                    "B6": 108.0, "B6p": 107.0}
            vals["Ours"] = 100.0 * (1 + ours_rel[s]) if w == "W2" else (120.0 * b0_ratio if w == "W3" else 100.0 * b0_ratio)
            for v, x in vals.items():
                rows.append(dict(split="eval", variant=v, workload=w, N=10000, seed=100 + s, budget="default", cold="disk", prior_kind="A",
                                 trace_kind="file", boot=-1, status="ok", service_s=x))
    return pd.DataFrame(rows)


def selftest():
    r = verdict(_synthetic([-0.03] * 8 + [0.0] * 2), None)["per_N"]["10000"]
    assert r["H4a_W2"]["verdict"] == "supported" and r["H4a_W2"]["best_online"] == "B1", r["H4a_W2"]
    r = verdict(_synthetic([-0.03] * 7 + [0.0] * 3), None)["per_N"]["10000"]
    assert r["H4a_W2"]["verdict"] == "undetermined"
    r = verdict(_synthetic([0.03] * 8 + [0.0] * 2), None)["per_N"]["10000"]
    assert r["H4a_W2"]["verdict"] == "refuted"
    r = verdict(_synthetic([-0.0199] * 10), None)["per_N"]["10000"]
    assert r["H4a_W2"]["verdict"] == "undetermined"        # 1.99% is not >= 2%
    r = verdict(_synthetic([0.0] * 10, b0_ratio=1.0199), None)["per_N"]["10000"]
    assert r["H4b_W3"]["verdict"] == "supported" and r["H4c_W1"]["verdict"] == "supported"
    r = verdict(_synthetic([0.0] * 10, b0_ratio=1.03), None)["per_N"]["10000"]
    assert r["H4b_W3"]["verdict"] == "refuted" and r["H4c_W1"]["verdict"] == "refuted"
    d = _synthetic([-0.03] * 10)
    d.loc[(d.variant == "B1") & (d.workload == "W2") & (d.seed == 105), "status"] = "missing_cost"      # a baseline with a missing seed is excluded
    r = verdict(d, None)["per_N"]["10000"]["H4a_W2"]
    assert r["best_online"] == "B5" and "B1" in r["excluded"], r["best_online"]
    assert verdict(_synthetic([-0.03] * 10), None)["test_only"] is False
    print("h4_verdict selftest OK (7 logic cases)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw")
    ap.add_argument("--bootstrap", default=None)
    ap.add_argument("--out")
    ap.add_argument("--cold", default="disk")
    ap.add_argument("--budget", default="default")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    assert a.raw and a.out
    raw = pd.read_csv(a.raw)
    boot = pd.read_csv(a.bootstrap) if a.bootstrap else None
    out = verdict(raw, boot, a.cold, a.budget)
    out["inputs"] = dict(raw=str(a.raw), bootstrap=a.bootstrap)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["primary"], ensure_ascii=False), "test_only=", out["test_only"])
    for N, r in out["per_N"].items():
        a_ = r.get("H4a_W2")
        if a_:
            print(f"N={N}: best online={a_['best_online']} Ours/best-1={a_['mean_rel_diff']:+.2%} lower_by_2%={a_['seeds_ours_lower_by_2pct']}/{out['n_seeds']} "
                  f"higher_by_2%={a_['seeds_ours_higher_by_2pct']} -> {a_['verdict']}")


if __name__ == "__main__":
    main()
