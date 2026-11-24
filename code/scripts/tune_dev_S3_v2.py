"""S3 3.2b development tuning of Ours v2 (CPU only, simulator). Pre-registered in STATE/AUTO_DECISIONS.md ("S3 3.2 policy design revision v2", 2026-11-17):
  3 candidate groups only: {alpha=1,H=1e3}, {alpha=1,H=1e4}, {alpha=0.5,H=1e3}, all with (a) online horizon H_eff=max(H_prior,n_served)
  and (b) four tiers E->G1->C1->C1+G1; objective = geometric mean of total service time over W1-W5 x N in {1e3,1e4} x seeds 0-4
  (50 dev traces, A half only), default budget, pre-warmed disk-cache tier; first group unless another is better by >= 2%.
The v1_spec (3 tiers, fixed horizon) three groups are re-run as history and must reproduce REPORT_3.2 (602.11 / 627.92 / 601.06).
Only results/S3/workload/dev_traces/*_A.jsonl are read; evaluation traces are never touched.
Usage: python scripts/tune_dev_S3_v2.py [--out results/S3/sim_dev/v2]   (nice -n 10; single process)
"""
import argparse, collections, hashlib, json, os, pathlib, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.policy import CostView, make_policy, shape_of, NAMES  # noqa: E402
from p24.policy.ours import V1_SPEC_PARAMS  # noqa: E402
from p24.sim.costs import CostTable  # noqa: E402
from p24.sim.prior import prior_A  # noqa: E402
from p24.sim.simulator import SimOptions, read_trace, simulate  # noqa: E402

CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
DEV = ROOT / "results/S3/workload/dev_traces"
GROUPS = {"g1_a1_H1e3": dict(alpha=1.0, H_prior=1e3), "g2_a1_H1e4": dict(alpha=1.0, H_prior=1e4), "g3_a0.5_H1e3": dict(alpha=0.5, H_prior=1e3)}
VARIANTS = {"v2": dict(), "v1_spec": dict(V1_SPEC_PARAMS)}
WLS, NS, SEEDS = ["W1", "W2", "W3", "W4", "W5"], [1000, 10000], [0, 1, 2, 3, 4]
V1_REPORTED = {"g1_a1_H1e3": 602.11, "g2_a1_H1e4": 627.92, "g3_a0.5_H1e3": 601.06}     # REPORT_3.2 section 2
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def trace_path(w, n, s):
    p = DEV / f"{w}_N{n}_seed{s}_A.jsonl"
    assert p.name.endswith("_A.jsonl") and p.parent == DEV and p.exists(), p
    return p


def geomean(x):
    return float(np.exp(np.log(np.asarray(list(x), float)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/S3/sim_dev/v2"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert sha16(CT_PATH) == "aa691bbea98d15c6", sha16(CT_PATH)
    assert sha16(ROOT / "results/S3/workload/sha256_manifest.txt") == "7d7b993c4d5402b0"
    ct = CostTable.from_csv(CT_PATH)
    cv = CostView(ct, "warm")
    prior = prior_A()
    bj = json.loads((ROOT / "results/S3/budgets.json").read_text())["tiers"]
    opt = SimOptions(cache_mode="warm", mem_budget_mb=bj["default"]["mb"], overhead_aware=True)
    traces = {(w, n, s): read_trace(trace_path(w, n, s)) for w in WLS for n in NS for s in SEEDS}
    for (w, n, s), r in traces.items():
        assert len(r) == n
    rows, t0 = [], time.time()
    for var, vx in VARIANTS.items():
        for g, gp in GROUPS.items():
            for (w, n, s), reqs in traces.items():
                r = simulate(reqs, make_policy("Ours", cost=cv, params={**gp, **vx}, prior=prior), ct, opt).summary
                rows.append(dict(variant=var, group=g, W=w, N=n, seed=s, service_s=r["service_s"], n_evicted=r["n_evicted_total"],
                                 n_denied=r["n_denied_total"], peak_graph_mb=r["peak_graph_mb"]))
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "tuning_ours_v2_raw.csv", index=False)
    print(f"tuning sims done in {time.time()-t0:.0f}s")
    summ, first = {}, list(GROUPS)[0]
    for var in VARIANTS:
        gm = {g: geomean(raw[(raw.variant == var) & (raw.group == g)].service_s) for g in GROUPS}
        gmN = {g: {str(n): geomean(raw[(raw.variant == var) & (raw.group == g) & (raw.N == n)].service_s) for n in NS} for g in GROUPS}
        best = min(gm, key=gm.get)
        chosen = first if gm[first] <= 1.02 * gm[best] else best
        summ[var] = dict(objective_geomean_service_s=gm, objective_by_N=gmN, best_group=best, chosen_group=chosen,
                         first_vs_best_pct=100 * (gm[first] / gm[best] - 1), n_traces=int(len(raw[(raw.variant == var) & (raw.group == first)])))
        print(var, {g: round(v, 2) for g, v in gm.items()}, "->", chosen, f"(first vs best {summ[var]['first_vs_best_pct']:.2f}%)")
    for g, v in V1_REPORTED.items():       # history must reproduce the previous executor's numbers
        assert abs(summ["v1_spec"]["objective_geomean_service_s"][g] - v) < 0.006, (g, summ["v1_spec"]["objective_geomean_service_s"][g], v)
    (out / "tuning_ours_v2_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))

    # ---- dev ordering (dev information only; evaluation is 3.3). v2 group for Ours v2 and the Ours-derived baselines.
    g2, g1 = GROUPS[summ["v2"]["chosen_group"]], GROUPS[summ["v1_spec"]["chosen_group"]]
    extra = {"Ours v1_spec": ("Ours", {**g1, **V1_SPEC_PARAMS}),
             "Ours-fixedH (ablation: no a)": ("Ours", {**g2, "horizon": "fixed"}),
             "Ours-3tier (ablation: no b)": ("Ours", {**g2, "tiers": ("E", "G1", "C1+G1")})}
    rows = []
    for (w, n, s), reqs in traces.items():
        oc = dict(collections.Counter(shape_of(q) for q in reqs))
        for bk in ("default", "tight"):
            o = SimOptions(cache_mode="warm", mem_budget_mb=bj[bk]["mb"], overhead_aware=True)
            plist = [(nm if nm != "Ours" else "Ours v2", nm, g2) for nm in NAMES] + [(lab, nm, pp) for lab, (nm, pp) in extra.items()]
            for lab, nm, pp in plist:
                r = simulate(reqs, make_policy(nm, cost=cv, params=pp, prior=prior, oracle_counts=oc), ct, o).summary
                rows.append(dict(policy=lab, budget=bk, W=w, N=n, seed=s, service_s=r["service_s"], warmup_s=r["warmup_s"],
                                 n_evicted=r["n_evicted_total"], n_denied=r["n_denied_total"], peak_graph_mb=r["peak_graph_mb"]))
    od = pd.DataFrame(rows)
    od.to_csv(out / "dev_policy_ordering_v2_raw.csv", index=False)
    key = ["budget", "W", "N", "seed"]
    od["ratio_B0"] = od.service_s / od.set_index(key).index.map(od[od.policy == "B0"].set_index(key).service_s)
    od["ratio_B1"] = od.service_s / od.set_index(key).index.map(od[od.policy == "B1"].set_index(key).service_s)
    for bk in ("default", "tight"):
        for nsel in ("all",) + tuple(NS):
            d = od[(od.budget == bk) & ((od.N == nsel) if nsel != "all" else True)]
            t = d.groupby("policy").agg(geomean_ratio_vs_B0=("ratio_B0", geomean), geomean_ratio_vs_B1=("ratio_B1", geomean),
                                        geomean_service_s=("service_s", geomean), mean_evicted=("n_evicted", "mean"),
                                        mean_denied=("n_denied", "mean"), max_peak_graph_mb=("peak_graph_mb", "max")).sort_values("geomean_ratio_vs_B0")
            t.to_csv(out / f"dev_policy_ordering_v2_{bk}_N{nsel}.csv")
            print(f"--- dev ordering budget={bk} N={nsel}: geometric mean over {len(d)//d.policy.nunique()} dev traces")
            print(t.round(4).to_string())
    pv = od[od.budget == "default"].pivot_table(index=["W", "N"], columns="policy", values="ratio_B0", aggfunc=geomean)
    pv.to_csv(out / "dev_policy_ordering_v2_by_scenario_default.csv")
    # paired comparison Ours v2 vs B1 per scenario (default)
    pv1 = od[od.budget == "default"].pivot_table(index=["W", "N"], columns="policy", values="ratio_B1", aggfunc=geomean)
    pv1.to_csv(out / "dev_policy_ordering_v2_by_scenario_default_vsB1.csv")
    print(pv1[["Ours v2", "Ours v1_spec", "B0", "B7"]].round(3).to_string())


if __name__ == "__main__":
    main()
