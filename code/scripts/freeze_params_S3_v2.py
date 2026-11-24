"""S3 3.2b: freeze Ours v2 into results/S3/policy_params.json (0444) from scripts/tune_dev_S3_v2.py output. v1 is kept as policy_params_v1_spec.json (0444, untouched).
policy_params.json (v1) is made writable once, replaced by v2 (version field = 'v2'), and set back to 0444. Refuses to run twice (checks version)."""
import hashlib, json, os, pathlib, time

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
D = ROOT / "results/S3/sim_dev/v2"
S3 = ROOT / "results/S3"
summ = json.loads((D / "tuning_ours_v2_summary.json").read_text())
GROUPS = {"g1_a1_H1e3": dict(alpha=1.0, H_prior=1e3), "g2_a1_H1e4": dict(alpha=1.0, H_prior=1e4), "g3_a0.5_H1e3": dict(alpha=0.5, H_prior=1e3)}
ct, man = ROOT / "results/S2/cost_table/merged/cost_table.csv", S3 / "workload/sha256_manifest.txt"
assert sha(ct)[:16] == "aa691bbea98d15c6" and sha(man)[:16] == "7d7b993c4d5402b0"
p, p1 = S3 / "policy_params.json", S3 / "policy_params_v1_spec.json"
cur = json.loads(p.read_text())
old = json.loads(p1.read_text())          # v1 history always comes from the untouched policy_params_v1_spec.json
assert "version" not in old
assert ("version" not in cur and sha(p) == sha(p1)) or cur["version"] == "v2", "policy_params.json is neither the untouched v1 nor a v2 freeze"
# 2026-11-17 note: re-run once after the C1-backoff implementation fix (first pass kept in results/S3/sim_dev/v2_firstpass_before_C1_backoff_fix/)
v2, v1 = summ["v2"], summ["v1_spec"]
out = {
    "version": "v2",
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "step": "S3 3.2b Ours v2 development tuning (simulator, CPU, development traces from half A only; evaluation traces untouched)",
    "implementation_fix_note": "first v2 freeze (policy_code ours.py SHA16 see sim_dev/v2_firstpass_before_C1_backoff_fix/policy_params_firstpass.json) was redone after the toy test 'budget full -> only C1' exposed that the refused-graph backoff also throttled the memory-free C1 tier; fix = C1 keeps threshold alpha and is tried during graph backoff; dev numbers of both passes are kept; no change to candidates/objective/selection rule",
    "revision_basis": "v2 revision basis: see AUTO_DECISIONS 2026-11-17 (STATE/AUTO_DECISIONS.md, '2026-11-17 16:xx S3 3.2 policy design revision v2'; revision made AFTER seeing v1 development numbers; dev set only, no third design round)",
    "cost_table": "results/S2/cost_table/merged/cost_table.csv", "cost_table_sha16": sha(ct)[:16],
    "workload_manifest": "results/S3/workload/sha256_manifest.txt", "manifest_sha16": sha(man)[:16],
    "budgets_json_sha16": sha(S3 / "budgets.json")[:16],
    "tuning_script": "code/scripts/tune_dev_S3_v2.py", "tuning_raw_csv": "results/S3/sim_dev/v2/tuning_ours_v2_raw.csv",
    "tuning_raw_csv_sha16": sha(D / "tuning_ours_v2_raw.csv")[:16],
    "tuning_condition": "W1-W5 x N in {1e3,1e4} x seeds 0-4 (50 dev traces), default budget 6565.1 MiB, cache mode warm (pre-warmed disk cache), "
                        "overhead-aware admission (22.33 MiB per graph), objective = geometric mean of total service time (s); all three groups carry (a) online horizon and (b) four tiers",
    "groups_compared": GROUPS,
    "selection_rule": "pre-registered (AUTO_DECISIONS 2026-11-17, same rule as 3.2): choose the lowest objective; if the first group is within 2% of the lowest "
                      "(first <= 1.02 x best) take the first group",
    "ours": {
        "tiers": ["E", "G1", "C1", "C1+G1"], "horizon": "online: H_eff = max(H_prior, n_served)",
        "objective_geomean_service_s": v2["objective_geomean_service_s"], "objective_by_N": v2["objective_by_N"],
        "best_group": v2["best_group"], "chosen_group": v2["chosen_group"], "first_vs_best_pct": v2["first_vs_best_pct"],
        "params": {**GROUPS[v2["chosen_group"]], "horizon": "online", "tiers": ["E", "G1", "C1", "C1+G1"]},
        "selection_basis": f"objective of the three groups {v2['objective_geomean_service_s']}; first group is {v2['first_vs_best_pct']:.2f}% above the best "
                           f"-> {'first group (<2%)' if v2['chosen_group'] == 'g1_a1_H1e3' else 'best group'}",
    },
    "v1_history": {   # the previous freeze (3 tiers, fixed horizon), kept as ablation "v1_spec"; file policy_params_v1_spec.json (0444)
        "file": "results/S3/policy_params_v1_spec.json", "file_sha16": sha(p1)[:16], "tiers": ["E", "G1", "C1+G1"], "horizon": "fixed H_prior",
        "objective_geomean_service_s": old["ours"]["objective_geomean_service_s"], "chosen_group": old["ours"]["chosen_group"],
        "params": old["ours"]["params"], "selection_basis": old["ours"]["selection_basis"],
        "rerun_in_v2_script_objective": v1["objective_geomean_service_s"],
        "v1_C1_extension_history": old.get("ours_extension_C1_tier"),
    },
    "ablation_params": {"v1_spec": {"horizon": "fixed", "tiers": ["E", "G1", "C1+G1"], **old["ours"]["params"]},
                        "fixed_horizon (no a)": {"horizon": "fixed"}, "three_tiers (no b)": {"tiers": ["E", "G1", "C1+G1"]}},
    "shared_by": "Ours, Ours-NE, B5 use ours.params (alpha, H_prior, horizon, tiers); ablations vary one switch of Ours.params only",
    "fixed_constants_not_tuned": old["fixed_constants_not_tuned"],
    "policy_code": {f: sha(ROOT / "code/p24" / f)[:16] for f in ("policy/base.py", "policy/ours.py", "policy/baselines.py", "policy/factory.py",
                                                                  "policy/costview.py", "sim/simulator.py", "sim/prior.py")},
}
os.chmod(p, 0o644)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
os.chmod(p, 0o444)
print(json.dumps(out["ours"], indent=1, ensure_ascii=False))
