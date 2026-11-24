"""S-6 verdict (plan 2.6 as replaced by §9.9). Written before any S-6 measurement was seen (2026-11-15 23:59). CPU only.

Inputs : predictions CSV from scripts/predict_S2_6.py (pred_total_s = simulated one-time + steady + per-shape input_s),
         replay metas results/S2/sim_validation/raw/<policy>_<stream>_p<id>.meta.json (+ raw_empty for V3 x B1 empty).
Measured total service time = meta `service_s` (model loaded -> last request finished, includes input preparation);
the load-inclusive total `total_incl_load_s` is reported alongside.
Rules (§9.9):
  error = |sim - median(measured over 3 processes)| / median; pass iff error <= 10% for EVERY (stream, policy) combination
  of the 9 disk-cache combinations; the V3 x B1 empty-cache run (1 process) is reported, not judged.
  tie: delta = max over the two policies of (max - min)/median of their 3 process values; policies a, b are tied when
       |Ta - Tb| / min(Ta, Tb) < max(2%, 2 * delta). Ranking check: for every NON-tied pair within a stream the simulated
       order must equal the measured order.
Output : results/S2/sim_validation/error_table.csv, ranking_table.csv, verdict.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
V = ROOT / "results" / "S2" / "sim_validation"


def measured(raw: pathlib.Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.meta.json")):
        m = json.loads(f.read_text())
        stream = pathlib.Path(m["trace"]).stem
        rows.append(dict(stream=stream, policy=m["policy"], cache_mode=m["cache_mode"], proc=f.stem.rsplit("_p", 1)[1].split(".")[0],
                         status=m["status"], service_s=m.get("service_s"), service_s_excl_input=m.get("service_s_excl_input"),
                         input_s_total=m.get("input_s_total"), total_incl_load_s=m.get("total_incl_load_s"),
                         load_s=m.get("load_s"), peak_reserved_mb=m.get("peak_reserved_mb"),
                         event_counts=json.dumps(m.get("event_counts"), sort_keys=True), file=f.name))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pred = pd.read_csv(a.pred)
    meas = pd.concat([measured(V / "raw"), measured(V / "raw_empty")], ignore_index=True)
    meas.to_csv(V / f"measured_processes{a.tag}.csv", index=False)
    rows = []
    for (s, p, cm), g in meas.groupby(["stream", "policy", "cache_mode"]):
        ok = g[g.status == "ok"]
        vals = ok.service_s.to_numpy(float)
        pr = pred[(pred.stream == s) & (pred.policy == p) & (pred.cache_mode == cm)]
        sim = float(pr.pred_total_s.iloc[0]) if len(pr) and "pred_total_s" in pr else np.nan
        med = float(np.median(vals)) if len(vals) else np.nan
        rows.append(dict(stream=s, policy=p, cache_mode=cm, n_ok=len(vals), n_fail=int((g.status != "ok").sum()),
                         measured_values_s=";".join(f"{v:.3f}" for v in vals), measured_median_s=med,
                         spread_rel=(vals.max() - vals.min()) / med if len(vals) > 1 else np.nan,
                         sim_s=sim, error=abs(sim - med) / med if med == med else np.nan,
                         judged=int(cm == "warm"),
                         measured_incl_load_median_s=float(np.median(ok.total_incl_load_s)) if len(ok) else np.nan,
                         sim_excl_input_s=float(pr.pred_excl_input_s.iloc[0]) if len(pr) else np.nan,
                         measured_excl_input_median_s=float(np.median(ok.service_s_excl_input)) if len(ok) else np.nan))
    err = pd.DataFrame(rows)
    err["pass_10pct"] = (err.error <= 0.10).astype(int)
    err.to_csv(V / f"error_table{a.tag}.csv", index=False)
    rk = []
    for s, g in err[err.judged == 1].groupby("stream"):
        g = g.set_index("policy")
        for pa, pb in itertools.combinations(sorted(g.index), 2):
            ta, tb = g.loc[pa, "measured_median_s"], g.loc[pb, "measured_median_s"]
            delta = max(g.loc[pa, "spread_rel"], g.loc[pb, "spread_rel"])
            thr = max(0.02, 2 * delta)
            gap = abs(ta - tb) / min(ta, tb)
            tied = gap < thr
            sa, sb = g.loc[pa, "sim_s"], g.loc[pb, "sim_s"]
            same = (np.sign(ta - tb) == np.sign(sa - sb))
            rk.append(dict(stream=s, a=pa, b=pb, measured_a=ta, measured_b=tb, sim_a=sa, sim_b=sb, rel_gap=gap,
                           tie_threshold=thr, tied=int(tied), order_agrees=int(same), counts=int(not tied)))
    rk = pd.DataFrame(rk)
    rk.to_csv(V / f"ranking_table{a.tag}.csv", index=False)
    judged = err[err.judged == 1]
    verdict = dict(n_judged=int(len(judged)), all_within_10pct=bool((judged.error <= 0.10).all() and len(judged) == 9),
                   max_error=float(judged.error.max()), n_nontied_pairs=int(rk.counts.sum()),
                   ranking_agrees=bool(rk[rk.counts == 1].order_agrees.all()) if len(rk) else None,
                   n_failed_processes=int(err.n_fail.sum()), pred_file=a.pred)
    verdict["pass"] = bool(verdict["all_within_10pct"] and verdict["ranking_agrees"] is not False)
    (V / f"verdict{a.tag}.json").write_text(json.dumps(verdict, indent=2))
    print(err[["stream", "policy", "cache_mode", "n_ok", "measured_median_s", "sim_s", "error", "spread_rel"]].round(4).to_string(index=False))
    print(rk[["stream", "a", "b", "rel_gap", "tie_threshold", "tied", "order_agrees"]].round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
