"""Simulator predictions for the S3-B real-GPU validation (goals/S3_detailed_goals_v2.md 3.4; criteria = S2 sec 9.9). CPU only.

Runs (disk-cache tier = serve_trace --cache-mode warm):
  3 real-validation traces (W2, W3, W5; N=300, seed 100, half B) x {B0, B1, B4, Ours, BEST}   (BEST = --best-online, the best online baseline of 3.3)
  + W3 x B3' (PyTorch default recompile limit 8)
  + W5 x Ours with an EMPTY cache (reported only, never judged)
Predicted total service time = simulated sum of (one-time + steady) per request  [= serve_trace `service_s_excl_input`]
                             + sum over requests of input_ms(shape)               [S2 input_summary.csv, see input_ms() for uncovered shapes]
which is what serve_trace reports as meta `service_s` (first item start -> last request end, includes input preparation and B6/B6' warm-up).
Same configuration as the real run (scripts/window_S3_B.sh): frozen S2 cost table, ours.params of the frozen results/S3/policy_params.json for
every policy that reads params (Ours, B5, B6, B6'), A-half prior (p24.sim.prior.prior_A), SimOptions(overhead_aware=True), graph budget = tier
`default` of results/S3/budgets.json for the budgeted policies (B4, B5, B6, B6', Ours, Ours-NE), none for B0/B1/B2/B3'.

INPUT_MS: the input_summary covers 12 shapes. Any other shape is estimated (never silently): same (H, W) measured -> mean(input_ms / B) * B;
otherwise (768x768) -> mean per-pixel-per-sample rate over all measured shapes * H*W*B. `n_input_extrapolated` counts such requests.
The extrapolated part is ~1 % of the total (input_ms is 1-5 ms per request against 0.2-3 s of compute).

Usage (from code/, CPU: export CUDA_VISIBLE_DEVICES=''):
  python -B -m scripts.predict_S3_4 --best-online B5 --out ../results/S3/real_validation/pred_S3_4.csv --freeze
     --freeze chmods the CSV and its pred_meta.json to 0444 and prints SHA256 (do this BEFORE the S3-B block; refuses to overwrite).
  python -B -m scripts.predict_S3_4 --best-online B5 --dev-check --out <scratch>/pred_dev.csv
     dev-check: the same code on development traces (half A, seed 0, N=1000 cut to the first 300) for a tool check / duration estimate;
     never touches the real-validation traces, never freezes.
BEST may be any runnable online baseline: B1 B2 B3' B4 B5 B6 B6' (B1/B4 already in the set -> BEST run is dropped as duplicate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import pandas as pd

from p24.policy import CostView, make_policy, shape_of
from p24.sim import simulator as SIM
from p24.sim import addendum as ADD  # 3.1b cost addendum overlay (same cost basis as the 3.3 evaluation)
from p24.sim.costs import CostTable, MissingCost
from p24.sim.prior import prior_A

ROOT = pathlib.Path(__file__).resolve().parents[2]
R3 = ROOT / "results" / "S3"
CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
ADDENDUM_PATH = R3 / "cost_addendum/cost_addendum.csv"   # default --addendum; `--addendum none` = S2 table only
PARAMS_PATH = R3 / "policy_params.json"
BUDGETS_PATH = R3 / "budgets.json"
INPUT_PATH = ROOT / "results/S2/input_s/input_summary.csv"
RV = R3 / "workload/real_validation"
DEV = R3 / "workload/dev_traces"
STREAMS = {"W2": "W2_N300_seed100_B", "W3": "W3_N300_seed100_B", "W5": "W5_N300_seed100_B"}
DEV_STREAMS = {"W2": "W2_N1000_seed0_A", "W3": "W3_N1000_seed0_A", "W5": "W5_N1000_seed0_A"}
BASE = ["B0", "B1", "B4", "Ours"]
ALIAS = {"B3p": "B3′", "B3'": "B3′", "B3prime": "B3′", "B6p": "B6′", "B6'": "B6′", "B6prime": "B6′"}
RUNNABLE_BEST = ["B1", "B2", "B3′", "B4", "B5", "B6", "B6′"]
BUDGETED = {"B4", "B5", "B6", "B6′", "Ours", "Ours-NE"}
SIM_NAME = {"B3′": "B3p", "B6′": "B6p"}  # p24.policy.make_policy accepts the alias form


def canon(n: str) -> str:
    return ALIAS.get(n, n)


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def load_params() -> dict:
    P = json.loads(PARAMS_PATH.read_text())["ours"]["params"]
    P = dict(P)
    P["tiers"] = tuple(P["tiers"])
    return P


class InputModel:
    def __init__(self, path):
        d = pd.read_csv(path)
        self.exact = {(int(r.H), int(r.W), int(r.B), int(r.cfg)): float(r.input_ms) for r in d.itertuples()}
        per = {}
        for (h, w, b, c), v in self.exact.items():
            per.setdefault((h, w), []).append(v / b)
        self.rate_hw = {k: sum(v) / len(v) for k, v in per.items()}
        rp = [v / (h * w) for (h, w), v in self.rate_hw.items()]
        self.rate_px = sum(rp) / len(rp)
        self.n_extrap = 0

    def ms(self, s) -> float:
        k = (s.H, s.W, s.B, int(s.cfg))
        if k in self.exact:
            return self.exact[k]
        self.n_extrap += 1
        if (s.H, s.W) in self.rate_hw:
            return self.rate_hw[(s.H, s.W)] * s.B
        return self.rate_px * s.H * s.W * s.B


def plan(best: str | None):
    """[(stream, policy, cache_mode, role)]"""
    runs = []
    for w in STREAMS:
        pols = list(BASE)
        role = {p: p for p in pols}
        if best and best not in pols:
            pols.append(best)
            role[best] = "BEST"
        elif best:
            role[best] = f"{best}(=BEST)"
        for p in pols:
            runs.append((w, p, "warm", role[p]))
        if w == "W3" and "B3′" not in pols:
            runs.append((w, "B3′", "warm", "B3prime"))
    runs.append(("W5", "Ours", "empty", "Ours_emptycache_report_only"))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-online", required=True, help="best online baseline of 3.3 (B1 B2 B3' B4 B5 B6 B6'; aliases B3p B6p accepted)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-check", action="store_true", help="use dev traces (first 300 of N=1000 seed 0, half A); tool check only")
    ap.add_argument("--addendum", default=str(ADDENDUM_PATH),
                    help="S3-A cost addendum csv (default results/S3/cost_addendum/cost_addendum.csv; a missing file is an ERROR); 'none' = S2 table only")
    ap.add_argument("--freeze", action="store_true", help="chmod outputs 0444 after writing (for the real prediction)")
    a = ap.parse_args()
    best = canon(a.best_online)
    if best not in RUNNABLE_BEST:
        sys.exit(f"--best-online {a.best_online!r} not runnable in 3.4; choose from {RUNNABLE_BEST}")
    if a.dev_check and a.freeze:
        sys.exit("--dev-check output must not be frozen")
    out = pathlib.Path(a.out)
    if out.exists():
        sys.exit(f"REFUSING to overwrite {out}")
    if not a.dev_check:
        assert (PARAMS_PATH.stat().st_mode & 0o222) == 0, "policy_params.json must be read-only (0444)"
    P = load_params()
    pj = json.loads(PARAMS_PATH.read_text())
    addendum = None if str(a.addendum).lower() == "none" else pathlib.Path(a.addendum)
    if addendum is not None and not addendum.is_file():
        sys.exit(f"ERROR: cost addendum not found: {addendum} (the S3-A block has not produced it yet; pass --addendum none to run on the S2 table only)")
    ct = ADD.load_with_addendum(CT_PATH, addendum) if addendum is not None else CostTable.from_csv(CT_PATH)
    if addendum is not None:
        ADD.install_simulator_hook()        # the simulator's own CostView must charge the same rebuild seconds the policy sees
    assert sha16(CT_PATH) == pj["cost_table_sha16"], "cost table differs from the one frozen in policy_params.json"
    bj = json.loads(BUDGETS_PATH.read_text())
    assert bj["cost_table_sha16"] == sha16(CT_PATH)
    budget = float(bj["tiers"]["default"]["mb"])
    prior = prior_A()
    inp = InputModel(INPUT_PATH)
    rows, traces_sha = [], {}
    cache = {}
    for (w, pol, mode, role) in plan(best):
        stem = (DEV_STREAMS if a.dev_check else STREAMS)[w]
        path = (DEV if a.dev_check else RV) / f"{stem}.jsonl"
        if stem not in cache:
            reqs = SIM.read_trace(path)
            cache[stem] = reqs[:300] if a.dev_check else reqs
            traces_sha[stem] = sha16(path)
        reqs = cache[stem]
        bud = budget if pol in BUDGETED else None
        row = dict(stream=stem, policy=pol, cache_mode=mode, role=role, mem_budget_mb=bud, n_requests=len(reqs))
        try:
            cv = CostView(ct, mode, rebuild_override=getattr(ct, "rebuild_override", None))
            polobj = make_policy(SIM_NAME.get(pol, pol), cost=cv, params=dict(P), prior=prior)
            res = SIM.simulate(reqs, polobj, ct, SIM.SimOptions(cache_mode=mode, mem_budget_mb=bud, overhead_aware=True))
        except MissingCost as e:
            row.update(error=f"missing_cost: {str(e)[:200]}")
            rows.append(row)
            continue
        sm = res.summary
        inp.n_extrap = 0
        tot_in = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
        ev = pd.DataFrame(res.events)
        ev = ev[ev.idx >= 0] if "idx" in ev else ev
        row.update(pred_excl_input_s=float(sm["service_s"]), pred_input_s=tot_in, pred_total_s=float(sm["service_s"]) + tot_in,
                   n_input_extrapolated=inp.n_extrap, pred_warmup_s=float(sm.get("warmup_s", 0.0)),
                   pred_peak_graph_mb=sm.get("peak_graph_mb"), pred_peak_reserved_est_mb=sm.get("peak_reserved_est_mb"),
                   pred_n_evicted=sm.get("n_evicted_total"), pred_n_denied=sm.get("n_denied_total"),
                   n_events_with_fallback=sm.get("n_events_with_fallback"),
                   event_counts=json.dumps(ev.event.value_counts().sort_index().to_dict(), sort_keys=True))
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = dict(time=time.strftime("%FT%T"), argv=sys.argv, best_online=best, dev_check=a.dev_check, cost_table_sha16=sha16(CT_PATH),
                addendum=(dict(path=str(addendum), sha16=sha16(addendum), **{k: v for k, v in ct.addendum_report.items() if k.startswith("n_")})
                          if addendum is not None else None),
                policy_params_sha16=sha16(PARAMS_PATH), params_used=P, budget_default_mb=budget, traces_sha16=traces_sha,
                input_summary_sha16=sha16(INPUT_PATH), out_sha16=sha16(out),
                code_sha16={f: sha16(ROOT / "code/p24" / f) for f in ("policy/base.py", "policy/ours.py", "policy/baselines.py",
                                                                      "policy/factory.py", "policy/costview.py", "sim/simulator.py", "sim/prior.py")})
    mp = out.with_name(out.stem + "_meta.json")
    mp.write_text(json.dumps(meta, indent=2, default=list))
    if a.freeze:
        for p in (out, mp):
            os.chmod(p, 0o444)
    show = [c for c in ("stream", "policy", "cache_mode", "role", "pred_excl_input_s", "pred_input_s", "pred_total_s", "pred_warmup_s",
                        "pred_peak_reserved_est_mb", "n_input_extrapolated", "pred_n_evicted", "pred_n_denied", "error") if c in df]
    print(df[show].round(3).to_string(index=False))
    print("out", out, "out_sha16", sha16(out), "frozen" if a.freeze else "NOT frozen", "meta", mp)


if __name__ == "__main__":
    main()
