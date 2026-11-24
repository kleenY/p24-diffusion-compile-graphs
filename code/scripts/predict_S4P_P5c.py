"""S4-P P5c step 5 = Pred-1: scripts/predict_S4P_p3_v2.py (edd6bdfaeb6be4a6, NOT modified) + the B1-AOT arm.  CPU only.
Preregistration goals/S4P_P5c_AOT_confirmatory_prereg_v1.md (0444, SHA16 c14507708a968699) sec 4 "frozen prediction / Pred-1": after block A, with cost_table_v2_aot.csv (0444), predict the total service time of B1-AOT on the three validation
streams (W2/W3/W5, N=300, seed 101, half B) next to the old rows; 0444 BEFORE block C (--freeze).  The error |measured - predicted| / measured is judged by scripts/compare_S3_4.py rules (<= 10 % per combination) in
a123_verdict_S4P_P5c.py, reported as "simulator v2 on the new arm", NOT part of A1-A3.
What is identical to predict_S4P_p3_v2.py: the whole plan loop (B0, B1, B4, Ours, v3 for W2/W3/W5, W3 also B3'), the world table = the table given here (the AOT rows are extra rows of arm AOT: no old arm sees them),
the policy view (v1 default), parameters, budget, input model, columns -> the B0/B1/B4/Ours/v3/B3' rows are bit-for-bit the rows the original script writes for the same table without AOT rows (checked in tests/test_p5c_pipeline.py).
What is added: rows `B1-AOT` (policy B1 on the table with the C1 rows swapped for the AOT rows, aot_tables_S4P_P5c.swap_c1; fallback shapes (aot_fallback_shapes.csv) priced as E0), appended after the old rows.
Same --freeze rule as the original (0444 on csv + meta; refuses to overwrite); here additionally the table and the fallback list must already be 0444 and the preregistration SHA must match.
--stream-dir (selftest only, with --dryrun): read the streams <dir>/<W>_N300_seed101_B.jsonl instead of results/S4P/workload/real_validation.
Usage (from code/): python -B -m scripts.predict_S4P_P5c --table-csv ../results/S4P/p5c/cost_table_v2_aot.csv --addendum-csv ../results/S4P/p3/cost_v2/addendum_rebuild_v2.csv \\
                        --v3-params-json ../results/S4P/v3/policy_params_v3_flat.json --out ../results/S4P/p3/real_validation/pred_S4P_P5c.csv --freeze
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
from scripts import aot_tables_S4P_P5c as T
from scripts import s4p_p3_common_v2 as C

P34 = C.P34
OFFICIAL_OUT_DIR = C.ROOT / "results/S4P/p3/real_validation"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True, help="cost_table_v2_aot.csv (world table: v2 rows + AOT rows)")
    ap.add_argument("--addendum-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback-csv", default=str(T.FALLBACK_CSV))
    ap.add_argument("--seed", type=int, default=101, choices=[100, 101])
    ap.add_argument("--policy-view", choices=["v1", "v2"], default="v1")
    ap.add_argument("--v3-params-json", default=None, help="flat params dict of the frozen v3 (results/S4P/v3/...)")
    ap.add_argument("--v3-stand-in", default=None, help="DRY RUN ONLY: simulate this policy under the name v3")
    ap.add_argument("--stream-dir", default=None, help="SELFTEST ONLY (needs --dryrun): directory of stand-in streams")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--dryrun", action="store_true", help="label meta dryrun=true (synthetic / stand-in inputs); never frozen; output must not be under results/")
    a = ap.parse_args(argv)
    if a.dryrun and a.freeze:
        sys.exit("--dryrun output must not be frozen")
    if a.v3_stand_in and not a.dryrun:
        sys.exit("--v3-stand-in only with --dryrun")
    if a.stream_dir and not a.dryrun:
        sys.exit("--stream-dir only with --dryrun")
    if not a.v3_params_json and not a.v3_stand_in:
        sys.exit("v2: --v3-params-json (results/S4P/v3/policy_params_v3_flat.json) is required")
    out = pathlib.Path(a.out)
    if a.dryrun:
        assert not T.inside(out, C.ROOT / "results"), f"--dryrun outputs must not be written under results/: {out}"
    else:
        T.check_prereg()
        assert T.is_ro(a.table_csv), f"{a.table_csv} must be 0444 before Pred-1 is computed"
        assert T.is_ro(a.fallback_csv), f"{a.fallback_csv} must be 0444"
        assert T.inside(out, OFFICIAL_OUT_DIR), f"official Pred-1 must be written under {OFFICIAL_OUT_DIR}"
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
    sdir = pathlib.Path(a.stream_dir) if a.stream_dir else None
    spath = (lambda w, stem: sdir / f"{stem}.jsonl") if sdir else (lambda w, stem: C.stream_path(w, a.seed))
    rows, traces_sha, cache, v3_status = [], {}, {}, "ok"
    # ------------------------------------------------------------------ the original plan (code identical to predict_S4P_p3_v2.main)
    for (w, pol, mode) in C.make_plan():
        stem = stems[w]
        path = spath(w, stem)
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
    # ------------------------------------------------------------------ new: B1-AOT (policy B1 on the table with C1 := AOT rows; fallback shapes = E0)
    fallback = T.read_fallback(a.fallback_csv)
    ct_aot = T.swap_c1(ct_v2, T.aot_provider(ct_v2), fallback)
    ct_aot_pol = ct_aot if a.policy_view == "v2" else ct_v1            # B1 needs no cost view; kept symmetric with the old rows
    for w in C.WLS:
        stem = stems[w]
        reqs = cache[stem]
        row = dict(stream=stem, policy="B1-AOT", cache_mode="warm", role="B1-AOT", mem_budget_mb=None, n_requests=len(reqs))
        try:
            res = C.simulate_run(ct_aot, reqs, "B1", P, prior, budget, "warm", ct_policy=ct_aot_pol)
        except MissingCost as e:
            row.update(error=f"missing_cost: {str(e)[:200]}")
            rows.append(row)
            continue
        sm = res.summary
        inp.n_extrap = 0
        tot_in = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
        ev = pd.DataFrame(res.events)
        ev = ev[ev.idx >= 0] if "idx" in ev else ev
        n_fb_req = int(sum(1 for r in reqs if shape_of(r) in fallback))
        row.update(pred_excl_input_s=float(sm["service_s"]), pred_input_s=tot_in, pred_total_s=float(sm["service_s"]) + tot_in, n_input_extrapolated=inp.n_extrap,
                   pred_warmup_s=float(sm.get("warmup_s", 0.0)), pred_peak_graph_mb=sm.get("peak_graph_mb"), pred_peak_reserved_est_mb=sm.get("peak_reserved_est_mb"),
                   pred_n_evicted=sm.get("n_evicted_total"), pred_n_denied=sm.get("n_denied_total"), n_events_with_fallback=sm.get("n_events_with_fallback"),
                   event_counts=json.dumps(ev.event.value_counts().sort_index().to_dict(), sort_keys=True), n_fallback_requests=n_fb_req)
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = dict(time=time.strftime("%FT%T"), argv=sys.argv if argv is None else list(argv), dryrun=a.dryrun, seed=a.seed, policy_view=a.policy_view, v3_status=v3_status, v3_stand_in=a.v3_stand_in,
                prereg_sha16=T.PREREG_SHA16, pred_role="Pred-1 (preregistration sec 4): B1-AOT row = policy B1 on the table with C1 := AOT rows; the other rows = predict_S4P_p3_v2.py rows",
                table_v2_aot=dict(path=a.table_csv, sha16=C.sha16(a.table_csv), mode=oct(T.mode_of(a.table_csv))), addendum_rebuild_v2=dict(path=a.addendum_csv, sha16=C.sha16(a.addendum_csv)),
                aot_fallback=dict(path=a.fallback_csv, sha16=C.sha16(a.fallback_csv), shapes=sorted(s.tag() for s in fallback)),
                table_v1=dict(path=str(C.CT_V1), sha16=C.sha16(C.CT_V1), addendum=str(C.ADD_V1), addendum_sha16=C.sha16(C.ADD_V1)),
                policy_params_sha16=C.sha16(P34.PARAMS_PATH), params_used_ours=P, params_used_v3=(None if a.v3_params_json is None else dict(path=a.v3_params_json, sha16=C.sha16(a.v3_params_json))),
                budget_default_mb=budget, traces_sha16=traces_sha, input_summary_sha16=C.sha16(P34.INPUT_PATH), out_sha16=C.sha16(out),
                script_sha16=C.sha16(pathlib.Path(__file__).resolve()), aot_tables_sha16=C.sha16(C.ROOT / "code/scripts/aot_tables_S4P_P5c.py"),
                code_sha16={f: C.sha16(C.ROOT / "code/p24" / f) for f in ("policy/base.py", "policy/ours.py", "policy/baselines.py", "policy/factory.py", "policy/costview.py", "sim/simulator.py", "sim/prior.py")})
    mp = out.with_name(out.stem + "_meta.json")
    mp.write_text(json.dumps(meta, indent=2, default=list))
    if a.freeze:
        for p in (out, mp):
            os.chmod(p, 0o444)
    show = [c for c in ("stream", "policy", "role", "pred_excl_input_s", "pred_input_s", "pred_total_s", "pred_peak_reserved_est_mb", "n_events_with_fallback", "n_fallback_requests", "error") if c in df]
    print(df[show].round(3).to_string(index=False))
    print("v3_status:", v3_status)
    print("out", out, "out_sha16", C.sha16(out), "frozen" if a.freeze else "NOT frozen", "meta", mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
