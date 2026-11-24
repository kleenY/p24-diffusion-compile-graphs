"""[v2 2026-11-18: identical to predict_S4P_p3.py except `import s4p_p3_common_v2` and a guard that --v3-params-json is required (no silent S3-params fallback for v3) -- v3 is built via factory_v3 (Ours-v3) so the v3 row is no longer omitted; use --v3-params-json results/S4P/v3/policy_params_v3_flat.json. Original file untouched.]
S4-P P3-c: simulator predictions for the S4-P true-GPU validation block (window_S4P_B.sh) with cost table v2. CPU only. Reuses predict_S3_4 (InputModel, params, prior,
budget, output columns), changes none of it.

Runs (disk-cache tier, = serve_trace --cache-mode warm): streams W2, W3, W5 (N=300, seed 101, half B; --seed 100 re-simulates the S3 3.4 streams for the informative
cross-check) x {B0, B1, B4, Ours, v3} + W3 x B3'.  Predicted total = simulated service seconds (one-time + steady) + per-request input seconds (as predict_S3_4).
Tables: world = v2 (--table-csv / --addendum-csv = build_cost_table_v2_S4P.py outputs: charges one-time costs); policy view = v1 (frozen S2 + S3 addendum, as the real
runtime builds its policy) unless --policy-view v2. Policy `v3` is a PLACEHOLDER: it is simulated only if p24.policy.make_policy knows the name `v3` (P1 freezes it and wires it in);
else the row is omitted and meta.v3_status says so (--v3-stand-in <name> uses another policy under the name v3: DRY RUN ONLY, role is marked).
Params: Ours = frozen S3 ours.params; v3 = --v3-params-json (flat params dict) if given, else the S3 params. Budget: tier `default` of results/S3/budgets.json for B4/Ours/v3.
--freeze chmods the CSV and meta to 0444 (do it BEFORE window_S4P_B.sh); refuses to overwrite.
Usage (from code/): python -B -m scripts.predict_S4P_p3_v2 --table-csv ../results/S4P/p3/cost_v2/cost_table_v2.csv --addendum-csv ../results/S4P/p3/cost_v2/addendum_rebuild_v2.csv \\
                        --out ../results/S4P/p3/real_validation/pred_S4P_v2.csv --freeze
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd

from p24.policy import shape_of
from p24.sim.costs import MissingCost
from scripts import s4p_p3_common_v2 as C

P34 = C.P34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--addendum-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101, choices=[100, 101])
    ap.add_argument("--policy-view", choices=["v1", "v2"], default="v1")
    ap.add_argument("--v3-params-json", default=None, help="flat params dict of the frozen v3 (results/S4P/v3/...); default: the S3 params")
    ap.add_argument("--v3-stand-in", default=None, help="DRY RUN ONLY: simulate this policy under the name v3")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--dryrun", action="store_true", help="label meta dryrun=true (synthetic / stand-in inputs); never frozen")
    a = ap.parse_args()
    if a.dryrun and a.freeze:
        sys.exit("--dryrun output must not be frozen")
    if a.v3_stand_in and not a.dryrun:
        sys.exit("--v3-stand-in only with --dryrun")
    if not a.v3_params_json and not a.v3_stand_in:
        sys.exit("v2: --v3-params-json (results/S4P/v3/policy_params_v3_flat.json) is required")
    out = pathlib.Path(a.out)
    if out.exists():
        sys.exit(f"REFUSING to overwrite {out}")
    assert (P34.PARAMS_PATH.stat().st_mode & 0o222) == 0, "policy_params.json must be read-only (0444)"
    P = P34.load_params()
    Pv3 = dict(P)
    if a.v3_params_json:
        Pv3 = json.loads(pathlib.Path(a.v3_params_json).read_text())
        if "tiers" in Pv3:
            Pv3["tiers"] = tuple(Pv3["tiers"])
    pj = json.loads(P34.PARAMS_PATH.read_text())
    assert P34.sha16(C.CT_V1) == pj["cost_table_sha16"], "frozen cost table differs from policy_params.json"
    budget = float(json.loads(P34.BUDGETS_PATH.read_text())["tiers"]["default"]["mb"])
    ct_v1 = C.load_v1()
    ct_v2 = C.load_v2(a.table_csv, a.addendum_csv)
    ct_pol = ct_v1 if a.policy_view == "v1" else ct_v2
    prior = C.prior_A()
    inp = P34.InputModel(P34.INPUT_PATH)
    stems = C.STREAMS4 if a.seed == 101 else C.STREAMS3
    rows, traces_sha, cache, v3_status = [], {}, {}, "ok"
    for (w, pol, mode) in C.make_plan():
        stem = stems[w]
        path = C.stream_path(w, a.seed)
        if stem not in cache:
            cache[stem] = C.SIM.read_trace(path)
            traces_sha[stem] = C.sha16(path)
        reqs = cache[stem]
        role = pol if pol != "v3" else ("v3" if not a.v3_stand_in else f"v3(STAND-IN {a.v3_stand_in}; dry run)")
        row = dict(stream=stem, policy=pol, cache_mode=mode, role=role, mem_budget_mb=(budget if (pol in P34.BUDGETED or pol == "v3") else None), n_requests=len(reqs))
        try:
            res = C.simulate_run(ct_v2, reqs, pol, Pv3 if pol == "v3" else P, prior, budget, mode, v3_stand_in=a.v3_stand_in, ct_policy=ct_pol)
        except MissingCost as e:
            row.update(error=f"missing_cost: {str(e)[:200]}")
            rows.append(row)
            continue
        if res is None:
            v3_status = "OMITTED: p24.policy has no policy named v3 yet (P1 not frozen / not wired)"
            continue
        sm = res.summary
        inp.n_extrap = 0
        tot_in = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
        ev = pd.DataFrame(res.events)
        ev = ev[ev.idx >= 0] if "idx" in ev else ev
        row.update(pred_excl_input_s=float(sm["service_s"]), pred_input_s=tot_in, pred_total_s=float(sm["service_s"]) + tot_in, n_input_extrapolated=inp.n_extrap,
                   pred_warmup_s=float(sm.get("warmup_s", 0.0)), pred_peak_graph_mb=sm.get("peak_graph_mb"), pred_peak_reserved_est_mb=sm.get("peak_reserved_est_mb"),
                   pred_n_evicted=sm.get("n_evicted_total"), pred_n_denied=sm.get("n_denied_total"), n_events_with_fallback=sm.get("n_events_with_fallback"),
                   event_counts=json.dumps(ev.event.value_counts().sort_index().to_dict(), sort_keys=True))
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = dict(time=time.strftime("%FT%T"), argv=sys.argv, dryrun=a.dryrun, seed=a.seed, policy_view=a.policy_view, v3_status=v3_status, v3_stand_in=a.v3_stand_in,
                table_v2=dict(path=a.table_csv, sha16=C.sha16(a.table_csv)), addendum_rebuild_v2=dict(path=a.addendum_csv, sha16=C.sha16(a.addendum_csv)),
                table_v1=dict(path=str(C.CT_V1), sha16=C.sha16(C.CT_V1), addendum=str(C.ADD_V1), addendum_sha16=C.sha16(C.ADD_V1)),
                policy_params_sha16=C.sha16(P34.PARAMS_PATH), params_used_ours=P, params_used_v3=(None if a.v3_params_json is None else dict(path=a.v3_params_json, sha16=C.sha16(a.v3_params_json))),
                budget_default_mb=budget, traces_sha16=traces_sha, input_summary_sha16=C.sha16(P34.INPUT_PATH), out_sha16=C.sha16(out),
                script_sha16=C.sha16(pathlib.Path(__file__).resolve()),
                code_sha16={f: C.sha16(C.ROOT / "code/p24" / f) for f in ("policy/base.py", "policy/ours.py", "policy/baselines.py", "policy/factory.py", "policy/costview.py", "sim/simulator.py", "sim/prior.py")})
    mp = out.with_name(out.stem + "_meta.json")
    mp.write_text(json.dumps(meta, indent=2, default=list))
    if a.freeze:
        for p in (out, mp):
            os.chmod(p, 0o444)
    show = [c for c in ("stream", "policy", "role", "pred_excl_input_s", "pred_input_s", "pred_total_s", "pred_peak_reserved_est_mb", "n_events_with_fallback", "error") if c in df]
    print(df[show].round(3).to_string(index=False))
    print("v3_status:", v3_status)
    print("out", out, "out_sha16", C.sha16(out), "frozen" if a.freeze else "NOT frozen", "meta", mp)


if __name__ == "__main__":
    main()
