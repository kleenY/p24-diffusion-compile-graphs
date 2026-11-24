"""S3 3.0: write results/S3/budgets.json (4 graph-layer budget tiers, MiB).  CPU, no torch.
tight = sum of the C1+G1 pools (pool + 22.33 MB per graph overhead) of the 3 most frequent half-A shape buckets; mid = tight x2; default = 6565.1 of S2 memory_budget.json;
wide = 23552 - resident - largest eager workspace - 64 - 1024 (simulation only).  Half-A shape probability = A mapped-resolution distribution of S2 workload_stats.json x declared B/CFG distribution.
"""
import hashlib, json, os, sys
from pathlib import Path
import pandas as pd
ROOT = Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.workload import generate as G

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ctp = ROOT / "results/S2/cost_table/merged/cost_table.csv"
mb = json.loads((ROOT / "results/S2/memory_budget.json").read_text())
st = json.loads((ROOT / "results/S2/workload/workload_stats.json").read_text())
d = st["A_mapped_shape_dist"]
pr = {}
for H, W in G.RESOLUTIONS:
    for b, pb in zip(G.B_VALUES, G.B_PROBS):
        for c, pc in ((0, 1 - G.CFG_PROB), (1, G.CFG_PROB)):
            pr[(H, W, b, c)] = d[f"H{H}xW{W}"] * pb * pc
top = sorted(pr, key=lambda s: (-pr[s], s))[:3]
ct = pd.read_csv(ctp)
ov = mb["additivity_check"]["per_bucket_overhead_mb"]
rows, tight = [], 0.0
for s in top:
    r = ct[(ct.arm == "C1+G1") & (ct.H == s[0]) & (ct.W == s[1]) & (ct.B == s[2]) & (ct.cfg == s[3])]
    assert len(r) == 1
    pool = float(r.capture_pool_mb.iloc[0]); tight += pool + ov
    rows.append(dict(shape=dict(H=s[0], W=s[1], B=s[2], cfg=s[3]), p_A=round(pr[s], 6), c1g1_pool_mb=pool, per_graph_overhead_mb=ov, bucket_mb=round(pool + ov, 2)))
tight = round(tight, 2)
wide = round(23552 - mb["resident_mb"] - mb["eager_ws"]["max_over_workload_shapes_mb"] - mb["accounting_error_bound_mb"] - 1024, 2)
out = {
  "units": "MiB", "cost_table": "results/S2/cost_table/merged/cost_table.csv", "cost_table_sha16": sha(ctp)[:16],
  "memory_budget_json_sha16": sha(ROOT / "results/S2/memory_budget.json")[:16],
  "admission_rule": mb["admission_rule"],
  "tiers": {
    "tight": {"mb": tight, "rule": "sum over the 3 most probable half-A shape buckets (H,W,B,cfg) of one C1+G1 pool + 22.33 MB per-graph overhead each; i.e. only the 3 most common buckets fit", "buckets": rows,
              "calc": " + ".join(f"{r['bucket_mb']}" for r in rows) + f" = {tight}"},
    "mid": {"mb": round(2 * tight, 2), "rule": "tight x 2", "calc": f"2 x {tight}"},
    "default": {"mb": mb["graph_budget_mb"], "rule": "S2 memory_budget.json graph_budget_mb (20480 - resident - largest eager workspace - 64)",
                "calc": f"20480 - {mb['resident_mb']} - {mb['eager_ws']['max_over_workload_shapes_mb']} - {mb['accounting_error_bound_mb']} = {mb['graph_budget_mb']}"},
    "wide": {"mb": wide, "rule": "23552 (hard cap of an exclusive GPU0 process) - resident - largest eager workspace - 64 - 1024 (whole-process margin for the 716 MB underestimate x ~1.4); simulated only, not run on a real GPU",
             "calc": f"23552 - {mb['resident_mb']} - {mb['eager_ws']['max_over_workload_shapes_mb']} - 64 - 1024 = {wide}"},
  },
  "notes": "the tight tier uses half-A frequencies (evaluation traces do not take part); the 4 tiers are monotone tight < mid < default < wide; all are recomputed if the cost table changes.",
}
assert out["tiers"]["tight"]["mb"] < out["tiers"]["mid"]["mb"] < out["tiers"]["default"]["mb"] < out["tiers"]["wide"]["mb"]
p = ROOT / "results/S3/budgets.json"
txt = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
if p.exists():
    assert p.read_text() == txt, "already exists with different content"
else:
    p.write_text(txt); os.chmod(p, 0o444)
print({k: v["mb"] for k, v in out["tiers"].items()}, out["cost_table_sha16"])
