"""Simulator predictions for the S-6 validation runs (plan 2.6 / §9.9). CPU only.

Predicted total service time = simulated sum of (one-time + steady) per request  [= serve_trace `service_s_excl_input`]
                             + sum over requests of input_ms(shape)              [input_s micro-benchmark, per shape]
The first part is written BEFORE S-6 (file made read-only, SHA in the log); the input part is added once
results/S2/input_s/input_summary.csv exists (--input-summary). Graph budget of B4 per stream as in scripts/window_S2_6.sh.

Usage (from code/): python -B -m scripts.predict_S2_6 --out ../results/S2/sim_validation/pred_excl_input.csv
                    python -B -m scripts.predict_S2_6 --input-summary ../results/S2/input_s/input_summary.csv --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import pandas as pd

from p24.policy import make_policy, shape_of
from p24.sim.costs import CostTable, MissingCost
from p24.sim.simulator import SimOptions, read_trace, simulate

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "S2"
STREAMS = {"V1_N400_seed0_A": 16633.0, "V2_N300_seed1_A": 13065.0, "V3_N200_seed2_A": 13065.0}
RUNS = [(s, p, "warm") for s in STREAMS for p in ("B0", "B4", "B1")] + [("V3_N200_seed2_A", "B1", "empty")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-table", default=str(R / "cost_table/merged/cost_table.csv"))
    ap.add_argument("--input-summary", default=None)
    ap.add_argument("--use-transient", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ct = CostTable.from_csv(a.cost_table)
    inp = None
    if a.input_summary:
        d = pd.read_csv(a.input_summary)
        inp = {(int(r.H), int(r.W), int(r.B), int(r.cfg)): float(r.input_ms) for r in d.itertuples()}
    rows = []
    for s, pol, mode in RUNS:
        reqs = read_trace(R / "workload/validation_streams" / f"{s}.jsonl")
        bud = STREAMS[s] if pol == "B4" else None
        try:
            res = simulate(reqs, make_policy(pol), ct, SimOptions(cache_mode=mode, mem_budget_mb=bud,
                                                                   use_transient=a.use_transient))
        except MissingCost as e:
            rows.append(dict(stream=s, policy=pol, cache_mode=mode, mem_budget_mb=bud, n_requests=len(reqs), error=str(e)))
            continue
        sm = res.summary
        ev = pd.DataFrame(res.events)
        row = dict(stream=s, policy=pol, cache_mode=mode, mem_budget_mb=bud, n_requests=len(reqs),
                   pred_excl_input_s=float(sm["service_s"]), event_counts=json.dumps(ev.event.value_counts().to_dict()),
                   n_events_with_fallback=sm.get("n_events_with_fallback"), fallbacks=sm.get("fallbacks"),
                   peak_graph_mb=sm.get("peak_graph_mb"), peak_reserved_est_mb=sm.get("peak_reserved_est_mb"))
        if inp is not None:
            tot_in = sum(inp[(shape_of(r).H, shape_of(r).W, shape_of(r).B, int(shape_of(r).cfg))] for r in reqs) / 1e3
            row.update(pred_input_s=tot_in, pred_total_s=row["pred_excl_input_s"] + tot_in)
        rows.append(row)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.drop(columns=["fallbacks"]).to_string(index=False))
    print("cost_table_sha16", hashlib.sha256(pathlib.Path(a.cost_table).read_bytes()).hexdigest()[:16],
          "out_sha16", hashlib.sha256(out.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
