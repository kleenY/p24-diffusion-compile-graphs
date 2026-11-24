"""S3-X confirmation analysis (FROZEN BEFORE THE DATA; goals/S3_detailed_goals_v2.md lines 79-80 give the rule, this file only implements it).
Run from code/:  "$PROJ_PYTHON" -B -m scripts.accel_confirm_analyze            (real block: confirm/raw -> confirm/X_confirm_result.csv + X_confirm_verdict.txt)
Dry run on the screen data (format check only; file names get the suffix):
    "$PROJ_PYTHON" -B -m scripts.accel_confirm_analyze --raw ../results/S3/accel/screen/s512 ../results/S3/accel/screen/s768 --suffix _dryrun

Definitions (fixed here, before any confirmation data exists):
  * unit = independent process; per-process statistic = median of its steady-state wall_ms rows (row_kind=steady, status=ok; gpu_ms analogous, reported only).
  * per shape and arm: statistic = median over the process statistics (same as the X1 screen). REF = arm "C1+G1", R = arm "C1+G1-recipe".
  * speedup_pct = (REF - R) / REF * 100  (positive = R faster).
  * 95 % CI = process bootstrap: resample the REF processes and the R processes independently with replacement (n = their own count), 10000 draws,
    numpy default_rng seeded by SeedSequence([20261117, shape_index]); percentile interval 2.5 / 97.5 of speedup_pct. (A paired-by-round interval,
    resampling the process ids jointly, is written to the csv as a supplementary column; it is NOT used by the decision.)
  * main shapes = 512x512_b1_nocfg, 768x768_b4_cfg.  main_shape_hit = point speedup >= 3 and CI lower bound > 1.
  * significantly_slower(shape) = slowdown CI lower bound > 1 %, i.e. CI upper bound of speedup_pct < -1.
  * equivalence (8 prompts, RMSE ratio <= 2 and LPIPS <= 0.02 vs fp32 eager) = by_key[tag].equiv_pass from the equivalence jsons (screen json for the 2 main shapes,
    confirm/equiv/*.json for the other 4); all 6 shapes must be present and pass.
  * GO  <=>  at least one main_shape_hit  AND  no shape significantly slower  AND  equivalence passes on all 6 shapes.   Otherwise NO-GO.
    If any shape has fewer than 5 processes in either arm (time cut, failed process) the verdict is printed as INCOMPLETE together with the rule
    evaluation on the available data - it is then not a verdict and the main session decides.
  * One-time costs (first_req_s of each process, cold compile first_req_s from the x1_cold processes of prewarm/screen) are reported only, never part of the rule.
  * Flags per cell (cpu_contended, contended, tier_violation, fx_cache_miss, pinned_fallback_count > 0) are COUNTED and reported; no process is dropped except status != ok.
"""
import argparse, csv, glob, json, math, os, pathlib, statistics as st
import numpy as np

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
ACC = ROOT / "results/S3/accel"
SHAPES = ["512x512_b1_nocfg", "512x512_b4_cfg", "768x512_b1_nocfg", "512x768_b2_cfg", "768x768_b1_nocfg", "768x768_b4_cfg"]
MAIN = ["512x512_b1_nocfg", "768x768_b4_cfg"]
REF, CAND = "C1+G1", "C1+G1-recipe"
NBOOT, SEED = 10000, 20261117

ap = argparse.ArgumentParser()
ap.add_argument("--raw", nargs="+", default=[str(ACC / "confirm/raw")])
ap.add_argument("--equiv-json", nargs="*", default=None)
ap.add_argument("--cold-dirs", nargs="*", default=None)
ap.add_argument("--logs", default=str(ACC / "confirm/logs"))
ap.add_argument("--out-dir", default=str(ACC / "confirm"))
ap.add_argument("--suffix", default="")
ap.add_argument("--nrounds", type=int, default=5)
a = ap.parse_args()


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def tag_of(r):
    return f"{r['H']}x{r['W']}_b{r['B']}_{'cfg' if int(float(r['cfg'])) else 'nocfg'}"


# ---------- raw per-process statistics
procs = {}  # (shape, arm, pid) -> dict
for d in a.raw:
    for f in sorted(glob.glob(str(pathlib.Path(d) / "*.csv"))):
        rows = list(csv.DictReader(open(f)))
        ok = [r for r in rows if r.get("row_kind") == "steady" and r.get("status") == "ok" and r.get("wall_ms") not in ("", "nan", None)]
        if not ok:
            continue
        r0 = ok[0]
        if r0["arm"] not in (REF, CAND):
            continue
        shape, arm, pid = tag_of(r0), r0["arm"], int(float(r0["proc_id"]))
        pfc = None
        pf = pathlib.Path(a.logs) / f"{arm}_{shape}_p{pid}.pinned_fallback_count.txt"
        if pf.exists():
            pfc = int(pf.read_text().strip() or 0)
        procs[(shape, arm, pid)] = dict(
            wall=st.median(fl(r["wall_ms"]) for r in ok), gpu=st.median(fl(r["gpu_ms"]) for r in ok), n_rows=len(ok),
            first_req_s=fl(r0["first_req_s"]), cpu_cont=int(fl(r0.get("cpu_contended", 0)) or 0), cont=int(fl(r0.get("contended", 0)) or 0),
            tier_viol=int(fl(r0.get("tier_violation", 0)) or 0), fx_miss=int(fl(r0.get("fx_cache_miss", 0)) or 0), pinned=pfc,
            tier=r0.get("cold_tier"))


def cell(shape, arm):
    return {pid: v for (s, ar, pid), v in procs.items() if s == shape and ar == arm}


def boot_ci(ref, cand, ref_ids, cand_ids, rng):
    ref, cand = np.asarray(ref), np.asarray(cand)
    i1 = rng.integers(0, len(ref), (NBOOT, len(ref))); i2 = rng.integers(0, len(cand), (NBOOT, len(cand)))
    mr, mc = np.median(ref[i1], axis=1), np.median(cand[i2], axis=1)
    sp = (mr - mc) / mr * 100
    lo, hi = np.percentile(sp, [2.5, 97.5])
    plo = phi = float("nan")
    common = sorted(set(ref_ids) & set(cand_ids))
    if len(common) >= 2:
        rm = dict(zip(ref_ids, ref)); cm = dict(zip(cand_ids, cand))
        R_ = np.array([rm[c] for c in common]); C_ = np.array([cm[c] for c in common])
        j = rng.integers(0, len(common), (NBOOT, len(common)))
        spp = (np.median(R_[j], axis=1) - np.median(C_[j], axis=1)) / np.median(R_[j], axis=1) * 100
        plo, phi = np.percentile(spp, [2.5, 97.5])
    return float(lo), float(hi), float(plo), float(phi)


# ---------- equivalence
ej = a.equiv_json
if ej is None:
    ej = [str(ACC / "equiv/C1+G1-recipe.json")] + sorted(glob.glob(str(ACC / "confirm/equiv/C1+G1-recipe_*.json")))
equiv = {}
for f in ej:
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for t, m in d.get("by_key", {}).items():
        equiv[t] = dict(rmse_ratio=m["rmse_ratio"], lpips=m["lpips_vs_E0"], ok=int(m["equiv_pass"]) if d.get("status") == "ok" else 0,
                        fx_miss=d.get("compile_cache_counters", {}).get("fx_miss"), src=os.path.basename(f))

# ---------- one-time cold compile cost (x1_cold processes of the recipe arm in prewarm / screen)
cold_dirs = a.cold_dirs if a.cold_dirs is not None else [str(ACC / "confirm/prewarm"), str(ACC / "screen/s512"), str(ACC / "screen/s768")]
cold = {}
for d in cold_dirs:
    for f in sorted(glob.glob(str(pathlib.Path(d) / f"{CAND}_*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if rows and rows[0].get("cold_tier") == "x1_cold":
            cold[tag_of(rows[0])] = fl(rows[0]["first_req_s"])

# ---------- per shape table
out = []
for i, shape in enumerate(SHAPES):
    rc, cc = cell(shape, REF), cell(shape, CAND)
    row = dict(shape=shape, main=int(shape in MAIN), n_ref=len(rc), n_cand=len(cc))
    if rc and cc:
        ids_r, ids_c = sorted(rc), sorted(cc)
        wr, wc = [rc[p]["wall"] for p in ids_r], [cc[p]["wall"] for p in ids_c]
        mr, mc = st.median(wr), st.median(wc)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
        lo, hi, plo, phi = boot_ci(wr, wc, ids_r, ids_c, rng)
        gr, gc = st.median(rc[p]["gpu"] for p in ids_r), st.median(cc[p]["gpu"] for p in ids_c)
        row.update(ref_median_ms=round(mr, 3), cand_median_ms=round(mc, 3), speedup_pct=round((mr - mc) / mr * 100, 3),
                   ci_lo=round(lo, 3), ci_hi=round(hi, 3), paired_ci_lo=round(plo, 3), paired_ci_hi=round(phi, 3),
                   speedup_pct_gpu_ms=round((gr - gc) / gr * 100, 3),
                   main_shape_hit=int(shape in MAIN and (mr - mc) / mr * 100 >= 3.0 and lo > 1.0),
                   significantly_slower=int(hi < -1.0),
                   per_proc_ref=[round(x, 2) for x in wr], per_proc_cand=[round(x, 2) for x in wc],
                   ref_first_req_s_median=round(st.median(rc[p]["first_req_s"] for p in ids_r), 2),
                   cand_first_req_s_median=round(st.median(cc[p]["first_req_s"] for p in ids_c), 2),
                   cold_compile_first_req_s=round(cold[shape], 1) if shape in cold else "")
    e = equiv.get(shape)
    row.update(equiv_rmse_ratio=round(e["rmse_ratio"], 4) if e else "", equiv_lpips=round(e["lpips"], 5) if e else "",
               equiv_pass=e["ok"] if e else "", equiv_fx_miss=e["fx_miss"] if e else "")
    allp = list(rc.values()) + list(cc.values())
    row.update(flag_cpu_contended=sum(p["cpu_cont"] for p in allp), flag_gpu_contended=sum(p["cont"] for p in allp),
               flag_tier_violation=sum(p["tier_viol"] for p in allp), flag_fx_miss=sum(p["fx_miss"] for p in allp),
               flag_pinned_fallback=sum((p["pinned"] or 0) for p in allp))
    out.append(row)

# ---------- verdict
have = [r for r in out if "speedup_pct" in r]
complete = len(have) == 6 and all(r["n_ref"] >= a.nrounds and r["n_cand"] >= a.nrounds for r in out)
hit = [r["shape"] for r in out if r.get("main_shape_hit")]
slower = [r["shape"] for r in out if r.get("significantly_slower")]
eq_missing = [s for s in SHAPES if s not in equiv]
eq_fail = [s for s in SHAPES if s in equiv and not equiv[s]["ok"]]
c1, c2, c3 = bool(hit), not slower, (not eq_missing and not eq_fail)
go = c1 and c2 and c3
incomplete_why = [f"{r['shape']}: n_ref={r['n_ref']} n_cand={r['n_cand']}" for r in out if r["n_ref"] < a.nrounds or r["n_cand"] < a.nrounds]
verdict = ("GO" if go else "NO-GO") if complete else "INCOMPLETE"
lines = [f"S3-X confirmation verdict{a.suffix}  (raw: {', '.join(a.raw)}; nboot={NBOOT}, seed={SEED}; statistic = median of per-process medians; speedup = (REF-R)/REF)",
         f"VERDICT: {verdict}" + ("" if complete else "  (data incomplete; rule evaluation below is NOT a verdict; incomplete cells: " + "; ".join(incomplete_why) + ")"),
         f"rule (a) >=1 main shape with point speedup >= 3% and CI lower bound > 1%: {'PASS' if c1 else 'FAIL'}  hits={hit}",
         f"rule (b) no shape significantly slower (slowdown CI lower bound > 1%): {'PASS' if c2 else 'FAIL'}  slower={slower}",
         f"rule (c) 8-prompt equivalence passes on all 6 shapes: {'PASS' if c3 else 'FAIL'}  missing={eq_missing}  failed={eq_fail}",
         f"rule evaluation on available data: {'GO' if go else 'NO-GO'}",
         "", "per shape: n_ref/n_cand | REF ms | R ms | speedup % [95% CI] | paired CI (supplementary) | gpu_ms speedup | equiv rmse/lpips/pass | first_req_s REF/R | cold compile s"]
for r in out:
    if "speedup_pct" in r:
        lines.append(f"{r['shape']:18s}{'*' if r['main'] else ' '} n={r['n_ref']}/{r['n_cand']} {r['ref_median_ms']:9.2f} {r['cand_median_ms']:9.2f} "
                     f"{r['speedup_pct']:+6.2f} [{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] paired[{r['paired_ci_lo']:+6.2f},{r['paired_ci_hi']:+6.2f}] gpu {r['speedup_pct_gpu_ms']:+6.2f} "
                     f"equiv {r['equiv_rmse_ratio']}/{r['equiv_lpips']}/{r['equiv_pass']} first_req {r['ref_first_req_s_median']}/{r['cand_first_req_s_median']} cold {r['cold_compile_first_req_s']} "
                     f"flags cpu{r['flag_cpu_contended']} gpu{r['flag_gpu_contended']} tier{r['flag_tier_violation']} fxmiss{r['flag_fx_miss']} pin{r['flag_pinned_fallback']}")
    else:
        lines.append(f"{r['shape']:18s}{'*' if r['main'] else ' '} no data (n={r['n_ref']}/{r['n_cand']}); equiv {r['equiv_rmse_ratio']}/{r['equiv_lpips']}/{r['equiv_pass']}")
lines.append("(* = main shape. One-time costs are reported, not part of the rule.)")

od = pathlib.Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
keys = ["shape", "main", "n_ref", "n_cand", "ref_median_ms", "cand_median_ms", "speedup_pct", "ci_lo", "ci_hi", "paired_ci_lo", "paired_ci_hi",
        "speedup_pct_gpu_ms", "main_shape_hit", "significantly_slower", "equiv_rmse_ratio", "equiv_lpips", "equiv_pass", "equiv_fx_miss",
        "ref_first_req_s_median", "cand_first_req_s_median", "cold_compile_first_req_s", "flag_cpu_contended", "flag_gpu_contended",
        "flag_tier_violation", "flag_fx_miss", "flag_pinned_fallback", "per_proc_ref", "per_proc_cand"]
with open(od / f"X_confirm_result{a.suffix}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in out:
        w.writerow({k: r.get(k, "") for k in keys})
(od / f"X_confirm_verdict{a.suffix}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
