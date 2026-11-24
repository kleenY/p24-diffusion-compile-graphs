"""X1 screen analysis: per (shape, arm) process-median -> arm median (median over processes); per pair diff % (wall_ms,
negative = faster); X1 prereg criterion: candidate >=3% faster on one main shape and not >1% slower on the other.
Writes ../results/S3/accel/X1_screen_result_core.csv (equivalence / one-time-cost columns are merged later by hand)."""
import csv, glob, json, math, os, pathlib, statistics as st, sys
root = pathlib.Path(os.environ["PROJ_ROOT"]) / "results/S3/accel"
SHAPES = {"512x512_b1_nocfg": "s512", "768x768_b4_cfg": "s768"}
PAIRS = [("E0", "E0-cl"), ("G1", "G1-cl"), ("E0", "E0-cb"), ("C1+G1", "C1+G1-cb"), ("C1+G1", "C1+G1-recipe")]
data = {}  # (shape, arm) -> list of per-process dicts
for shape, d in SHAPES.items():
    for f in sorted(glob.glob(str(root / "screen" / d / f"*_{shape}_p*.csv"))):
        rows = list(csv.DictReader(open(f)))
        st_rows = [r for r in rows if r["row_kind"] == "steady" and r["status"] == "ok" and r["wall_ms"] not in ("", "nan")]
        if not st_rows:
            continue
        arm, pid = st_rows[0]["arm"], int(st_rows[0]["proc_id"])
        w = [float(r["wall_ms"]) for r in st_rows]; g = [float(r["gpu_ms"]) for r in st_rows]
        r0 = st_rows[0]
        data.setdefault((shape, arm), []).append(dict(
            pid=pid, wall=st.median(w), gpu=st.median(g), first_req_s=float(r0["first_req_s"]),
            compile_s=r0.get("compile_s"), cwu=r0.get("compile_and_warmup_s"), capture_s=r0.get("capture_s"),
            cpu_cont=int(float(r0["cpu_contended"])), cont=int(float(r0["contended"])), load1=r0["load1_start"],
            tier_viol=r0.get("tier_violation"), fxh=r0.get("fx_cache_hit"), fxm=r0.get("fx_cache_miss"), cold=r0["cold_tier"],
            peak=float(r0["steady_peak_alloc_mb"])))
def med(arm, shape, k):
    v = [p[k] for p in data.get((shape, arm), [])]
    return st.median(v) if v else float("nan")
out = []
res = {}
for base, cand in PAIRS:
    for shape in SHAPES:
        mb, mc = med(base, shape, "wall"), med(cand, shape, "wall")
        gb, gc = med(base, shape, "gpu"), med(cand, shape, "gpu")
        nb, nc = len(data.get((shape, base), [])), len(data.get((shape, cand), []))
        diff = (mc - mb) / mb * 100 if nb and nc else float("nan")
        gdiff = (gc - gb) / gb * 100 if nb and nc else float("nan")
        pw_b = [round(p["wall"], 2) for p in data.get((shape, base), [])]; pw_c = [round(p["wall"], 2) for p in data.get((shape, cand), [])]
        res[(base, cand, shape)] = diff
        out.append(dict(pair=f"{base} vs {cand}", shape=shape, base_median_ms=round(mb, 2), cand_median_ms=round(mc, 2),
                        diff_pct_wall=round(diff, 2), diff_pct_gpu=round(gdiff, 2), n_proc_base=nb, n_proc_cand=nc,
                        per_proc_base=pw_b, per_proc_cand=pw_c,
                        cpu_contended_procs=sum(p["cpu_cont"] for p in data.get((shape, base), []) + data.get((shape, cand), [])),
                        gpu_contended_procs=sum(p["cont"] for p in data.get((shape, base), []) + data.get((shape, cand), [])),
                        base_first_req_s=round(med(base, shape, "first_req_s"), 3), cand_first_req_s=round(med(cand, shape, "first_req_s"), 3),
                        cand_proc0_first_req_s=round(next((p["first_req_s"] for p in data.get((shape, cand), []) if p["pid"] == 0), float("nan")), 2)))
for r in out:
    base, cand = r["pair"].split(" vs ")
    other = [s for s in SHAPES if s != r["shape"]][0]
    d_this, d_other = res[(base, cand, r["shape"])], res[(base, cand, other)]
    r["meets_prereg"] = int(d_this <= -3.0 and d_other <= 1.0) if not (math.isnan(d_this) or math.isnan(d_other)) else ""
    r["pair_qualifies_any_shape"] = int(any(res[(base, cand, s)] <= -3.0 and res[(base, cand, [t for t in SHAPES if t != s][0])] <= 1.0 for s in SHAPES))
keys = list(out[0].keys())
with open(root / "X1_screen_result_core.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['pair']:22s} {r['shape']:18s} base={r['base_median_ms']:9.2f} cand={r['cand_median_ms']:9.2f} diff={r['diff_pct_wall']:+6.2f}% (gpu {r['diff_pct_gpu']:+6.2f}%) n={r['n_proc_base']}/{r['n_proc_cand']} meets={r['meets_prereg']} cpuCont={r['cpu_contended_procs']} gpuCont={r['gpu_contended_procs']}")
print("per-process medians:")
for k, v in sorted(data.items()):
    print(k, [(p['pid'], round(p['wall'],1), p['tier_viol'], p['fxh'], p['fxm']) for p in v])
