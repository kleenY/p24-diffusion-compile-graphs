"""S3 sensitivity (additional analysis, NOT a main result): Ours v2 on a "star" cost table = C1+G1 arm with the official Inductor recipe (R).

Background: S3-X confirm block (results/S3/accel/confirm/X_confirm_result.csv, verdict GO) measured, on 6 shapes, R vs REF (= C1+G1):
steady-state 2.95-5.22 % faster, but the first request of a NEW process with a warm disk cache is 43-66 s (REF 6-8 s) and a cold compile
takes 309-534 s. goals/S3_detailed_goals_v2.md line 80: "only the compile arm is faster -> W2 main panel gets an 'Ours*' sensitivity curve (extrapolated, labelled)".
This script changes NO frozen file: policy_params.json is loaded read-only, the S2 cost table / addendum are only read, sim_eval_S3.py is imported
(its trace reader, guards, priors, cost-table preprocessing and worker are reused so the control run is the formal evaluation path).

Cost-table conventions (p24/sim/costs.py): one row per (arm,H,W,B,cfg); steady_ms; one-time cost of a NEW shape at a tier =
resid_s__<tier> + sum_c fc_<c>_s__<tier> (components te/unet/vae, NaN = component key shared with an earlier shape, credited via
CostTable._shared_key_fc); tiers: cold_empty / cold_diskcache (first shape of a process), inproc_empty / inproc_diskcache (later new shape;
chain inproc_X -> cold_X when inproc cell is NaN). Disk tier = cache_mode 'warm', empty tier = cache_mode 'empty'
(formal evaluation uses --missing impute: missing cold_empty cells of compile arms = disk value x median empty/disk ratio; the same
impute_empty() is applied BEFORE the star modification, so the control table == the formal one).

Star table (only arm C1+G1 rows change; E0 (=B0), G1, C1, C2.. , capture_pool_mb, rebuild overrides, equivalence columns untouched):
  steady      6 measured shapes: steady_ms x (R median / REF median) of that shape (confirm csv); the other 28 shapes x median of the six ratios (extrapolated).
  disk/inproc_diskcache tiers: resid_s__tier += d  (and first_req_s__tier += d) wherever the cell is not NaN, d = (R first-req - REF first-req) of the
              confirm block for the 6 measured shapes; other shapes d = median over the 6 shapes of d / (REF steady s) x (that shape's table steady s)
              (ratio to REF steady, as specified; extrapolated). d is put on resid_s (no component-sharing credit -> conservative).
  empty tiers (cold_empty, inproc_empty): resid and fc_* of the cell are multiplied by k so that resid+sum(fc) == (R cold-compile first request - R steady) for the
              6 measured shapes (k = target / old total of the imputed table); other shapes k = median of the six k (ratio to REF's own empty one-time cost;
              extrapolated). first_req_s__tier is moved by the same absolute amount.
  capture pool memory: unchanged (the confirm block did not measure the recorded-graph pool; stated in meta).

--d-mode const (variant, added 2026-11-17 on the main session's request; default --d-mode prop = the specification above, unchanged):
  the six measured d (R first request - REF first request, 37-59 s) are nearly constant while REF steady differs ~13x, so proportional extrapolation
  over-states d for large shapes. const: unmeasured shapes get d = median of the six measured d (disk/inproc tiers), and in the empty tiers the
  one-time cost of an unmeasured shape is replaced by the median of the six measured (R cold-compile first request - R steady) (k = that substitute / the
  cell's imputed old one-time total). Measured shapes and the steady-state treatment are identical to prop. Outputs get the suffix _dconst
  (ours_star_W2_dconst_raw.csv, cost_table_star_dconst.csv, star_derivation_dconst.csv); only the star arm is run (the ctrl arm does not depend on d-mode).

Runs (W2 panel, preset of `sim_eval_S3.py` experiment main: cold levels empty/disk/hot, default budget): policies on the ORIGINAL table (control: B0, B1, Ours) and
Ours on the STAR table. --split dev = half A traces, seeds 0-4, N 1e3/1e4; --split eval = half B traces, seeds 100-109, N 1e2..1e5 (needs policy_params.json 0444).
Outputs results/S3/accel/sensitivity/[dev/]{ours_star_W2_raw.csv, ours_ctrl_W2_raw.csv, *_meta.json, cost_table_star.csv, star_derivation.csv}.
CPU only, <= 4 worker processes, nice 10. Usage:
  export CUDA_VISIBLE_DEVICES=''; source ops/project_env.sh; nice -n 10 "$PROJ_PYTHON" -B code/scripts/sens_recipe_S3.py --split dev --procs 3
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/scripts"))
import sim_eval_S3 as S  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402
from p24.sim.costs import COMPONENTS  # noqa: E402

CONFIRM = ROOT / "results/S3/accel/confirm/X_confirm_result.csv"
OUTROOT = ROOT / "results/S3/accel/sensitivity"
sha16 = S.sha16
STAR_ARM = "C1+G1"
CTRL_POLICIES = ["B0", "B1", "Ours"]
WL = "W2"


# ----------------------------------------------------------------------------------------------------------- star table
def _parse_tag(t: str):
    m = re.fullmatch(r"(\d+)x(\d+)_b(\d+)_(cfg|nocfg)", t)
    return int(m[1]), int(m[2]), int(m[3]), int(m[4] == "cfg")


def _tot(row: pd.Series, tier: str) -> float:
    v = row.get(f"resid_s__{tier}")
    if pd.isna(v):
        return np.nan
    return float(v) + float(np.nansum([row.get(f"fc_{c}_s__{tier}", np.nan) for c in COMPONENTS]))


def build_star(df_imp: pd.DataFrame, d_mode: str = "prop"):
    """df_imp = S2 table after impute_empty (as in the formal evaluation). -> (star DataFrame, derivation DataFrame, summary dict)."""
    cf = pd.read_csv(CONFIRM)
    meas = {}
    for r in cf.to_dict("records"):
        k = _parse_tag(r["shape"])
        rs = float(r["ref_median_ms"]) / 1e3
        meas[k] = dict(ratio=float(r["cand_median_ms"]) / float(r["ref_median_ms"]),
                       d_first=float(r["cand_first_req_s_median"]) - float(r["ref_first_req_s_median"]),
                       cold_first=float(r["cold_compile_first_req_s"]), r_steady_s=float(r["cand_median_ms"]) / 1e3, ref_steady_s=rs)
        meas[k]["d_per_steady"] = meas[k]["d_first"] / rs
    star = df_imp.copy()
    m_arm = (star.arm == STAR_ARM)
    keyof = lambda r: (int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"]))
    # --- k for the empty tier on the measured shapes
    for k, d in meas.items():
        row = star[m_arm & (star.H == k[0]) & (star.W == k[1]) & (star.B == k[2]) & (star.cfg == k[3])]
        assert len(row) == 1, ("confirm shape missing in the table", k)
        old = _tot(row.iloc[0], "cold_empty")
        assert np.isfinite(old) and old > 0, ("no cold_empty one-time cost for measured shape", k)
        d["old_empty_total_s"] = old
        d["k_empty"] = (d["cold_first"] - d["r_steady_s"]) / old
        d["target_empty_s"] = d["cold_first"] - d["r_steady_s"]
    med_ratio = float(np.median([d["ratio"] for d in meas.values()]))
    med_dps = float(np.median([d["d_per_steady"] for d in meas.values()]))
    med_k = float(np.median([d["k_empty"] for d in meas.values()]))
    med_d = float(np.median([d["d_first"] for d in meas.values()]))
    med_target = float(np.median([d["target_empty_s"] for d in meas.values()]))
    deriv = []
    tiers_d = ["cold_diskcache", "inproc_diskcache"]
    tiers_e = ["cold_empty", "inproc_empty"]
    for i in star.index[m_arm]:
        r = star.loc[i]
        k = keyof(r)
        steady_s = float(r["steady_ms"]) / 1e3
        if k in meas:
            ratio, d, kk, src = meas[k]["ratio"], meas[k]["d_first"], meas[k]["k_empty"], "measured"
        else:
            if d_mode == "const":
                old_e = _tot(r, "cold_empty")
                ratio, d, kk, src = med_ratio, med_d, med_target / old_e, "extrapolated"
            else:
                ratio, d, kk, src = med_ratio, med_dps * steady_s, med_k, "extrapolated"
        for c in ("steady_ms", "steady_ci_lo", "steady_ci_hi"):
            if pd.notna(star.at[i, c]):
                star.at[i, c] = float(star.at[i, c]) * ratio
        for t in tiers_d:
            for col in (f"first_req_s__{t}", f"resid_s__{t}"):
                if pd.notna(star.at[i, col]):
                    star.at[i, col] = float(star.at[i, col]) + d
        for t in tiers_e:
            old = _tot(r, t)
            if not np.isfinite(old):
                continue
            for col in [f"resid_s__{t}"] + [f"fc_{c}_s__{t}" for c in COMPONENTS]:
                if pd.notna(star.at[i, col]):
                    star.at[i, col] = float(star.at[i, col]) * kk
            if pd.notna(star.at[i, f"first_req_s__{t}"]):
                star.at[i, f"first_req_s__{t}"] = float(star.at[i, f"first_req_s__{t}"]) + (kk - 1.0) * old
        deriv.append(dict(arm=STAR_ARM, H=k[0], W=k[1], B=k[2], cfg=k[3], status=src, steady_ratio=ratio, steady_ms_ref=float(r["steady_ms"]),
                          steady_ms_star=float(star.at[i, "steady_ms"]), d_disk_s=d, k_empty=kk,
                          cold_empty_onetime_ref_s=_tot(r, "cold_empty"), cold_empty_onetime_star_s=_tot(star.loc[i], "cold_empty"),
                          cold_disk_onetime_ref_s=_tot(r, "cold_diskcache"), cold_disk_onetime_star_s=_tot(star.loc[i], "cold_diskcache")))
    star["star_note"] = np.where(m_arm, "star:C1+G1=official_inductor_recipe(sensitivity)", "")
    dv = pd.DataFrame(deriv)
    summ = dict(d_mode=d_mode, median_d_first_s=med_d, median_empty_target_s=med_target, n_star_rows=int(m_arm.sum()), n_measured=int((dv.status == "measured").sum()), n_extrapolated=int((dv.status == "extrapolated").sum()),
                median_steady_ratio=med_ratio, median_d_per_REFsteady=med_dps, median_k_empty=med_k,
                measured={f"{k[0]}x{k[1]}_b{k[2]}_{'cfg' if k[3] else 'nocfg'}": {a: round(b, 6) for a, b in d.items()} for k, d in meas.items()})
    return star, dv, summ


# ----------------------------------------------------------------------------------------------------------- workers
def _init(cfg: dict):
    S._init_worker(cfg)                      # reads the S2 table, applies impute + addendum exactly like the formal evaluation
    if cfg.get("star_csv"):
        df = pd.read_csv(cfg["star_csv"])
        ct = S._mk_table(df)                 # addendum overlay on the (already imputed) star table
        S.W["ct0i"] = ct
        S.W["ct0"] = ct


def run_pool(cfg: dict, specs, procs: int):
    tasks = [(-1, [sp]) for sp in sorted(specs, key=lambda x: -x[1])]      # largest N first
    rows = []
    t0 = time.time()
    with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(S.run_task, tasks, chunksize=1)):
            rows += r
            print(f"  {i+1}/{len(tasks)} tasks {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--N", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--d-mode", choices=["prop", "const"], default="prop", help="extrapolation of the one-time cost for unmeasured shapes (see header)")
    ap.add_argument("--out", help="output dir (default results/S3/accel/sensitivity[/dev])")
    a = ap.parse_args()
    assert 1 <= a.procs <= 4, "at most 4 worker processes for this task"
    sp = S.SPLITS[a.split]
    seeds = a.seeds or sp["seeds"]
    Ns = a.N or sp["N"]
    assert set(seeds) <= set(sp["seeds"]) and set(Ns) <= set(sp["N"])
    out = pathlib.Path(a.out) if a.out else (OUTROOT / "dev" if a.split == "dev" else OUTROOT)
    assert "sim_eval" not in str(out) and "sim_dev" not in str(out), "never write into the formal evaluation / dev tool directories"
    out.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.d_mode == "prop" else "_dconst"
    f_star, f_ctrl = out / f"ours_star_W2{sfx}_raw.csv", out / "ours_ctrl_W2_raw.csv"
    for f in ((f_star,) if a.d_mode == "const" else (f_star, f_ctrl)):
        if f.exists():
            sys.exit(f"REFUSING to overwrite {f}")
    pj = S.load_params()
    info = S.guards(a.split, pj)                 # eval: policy_params.json 0444, cost-table / manifest / policy-code SHAs frozen
    add_path = S.ADDENDUM_PATH
    P = S.base_params(pj)
    # --- star table (same preprocessing as the formal evaluation: impute -> star modification -> addendum overlay in the worker)
    df_imp, ilog = S.impute_empty(pd.read_csv(S.CT_PATH))
    star, deriv, ssum = build_star(df_imp, a.d_mode)
    star_csv = out / f"cost_table_star{sfx}.csv"
    star.to_csv(star_csv, index=False)
    deriv.to_csv(out / f"star_derivation{sfx}.csv", index=False)
    star_sha = sha16(star_csv)
    # --- which shapes the panel actually uses
    store = S.TraceStore(a.split, S.read_manifest())
    used = {}
    for n in Ns:
        for s in seeds:
            for q in store.get(WL, n, s, "file"):
                used[(q["H"], q["W"], q["B"], q["cfg"])] = used.get((q["H"], q["W"], q["B"], q["cfg"]), 0) + 1
    ext = {(r.H, r.W, r.B, r.cfg) for r in deriv[deriv.status == "extrapolated"].itertuples()}
    ext_used = sorted(k for k in used if k in ext)
    req_ext = sum(used[k] for k in ext_used) / sum(used.values())
    print(f"star table sha16={star_sha}; measured={ssum['n_measured']} extrapolated={ssum['n_extrapolated']}; shapes used by {WL} traces "
          f"{len(used)}, extrapolated among them {len(ext_used)} ({100*req_ext:.1f} % of requests)", flush=True)
    bj = json.loads(S.BUDGETS_PATH.read_text())["tiers"]
    budgets = {k: float(v["mb"]) for k, v in bj.items()}
    colds = ["empty", "disk", "hot"]
    specs = [(WL, n, s, "file") for n in Ns for s in seeds]
    for tag, fpath, pols, star_flag in (("ctrl", f_ctrl, CTRL_POLICIES, False), ("star", f_star, ["Ours"], True)):
        if tag == "ctrl" and a.d_mode == "const":
            continue
        variants = [S.Variant(p, p, dict(P)) for p in pols]
        ex = S.Exp("main", [WL], Ns, seeds, ["default"], colds, variants)
        cfg = dict(split=a.split, manifest=S.read_manifest(), exp=ex, budgets=budgets, priors=S.make_priors(False), impute=True, need_procs=False,
                   addendum=str(add_path), star_csv=str(star_csv) if star_flag else None)
        print(f"[{tag}] split={a.split} policies={pols} N={Ns} seeds={seeds} procs={a.procs}", flush=True)
        t0 = time.time()
        rows = run_pool(cfg, specs, a.procs)
        df = pd.DataFrame(rows)
        df["table"] = ("star" + sfx) if star_flag else "orig"
        df = df.sort_values(["variant", "workload", "N", "seed", "cold"]).reset_index(drop=True)
        df.to_csv(fpath, index=False)
        st = df.status.value_counts().to_dict()
        meta = dict(kind=tag, split=a.split, workload=WL, N=Ns, seeds=seeds, colds=colds, budget="default", policies=pols, status_counts=st,
                    n_rows=len(df), wall_s=time.time() - t0, time=time.strftime("%Y-%m-%dT%H:%M:%S"), missing_mode="impute",
                    guards=info, script_sha16=sha16(__file__), sim_eval_script_sha16=sha16(S.__file__),
                    inputs=dict(cost_table_S2_sha16=sha16(S.CT_PATH), addendum_sha16=sha16(add_path), confirm_csv_sha16=sha16(CONFIRM),
                                policy_params_sha16=sha16(S.PARAMS_PATH), budgets_sha16=sha16(S.BUDGETS_PATH), manifest_sha16=sha16(S.MANIFEST_PATH)),
                    star_table=dict(path=str(star_csv), sha16=star_sha, **{k: v for k, v in ssum.items()}),
                    n_extrapolated_cells=dict(shape_rows_total=ssum["n_extrapolated"], steady=ssum["n_extrapolated"], onetime_disk_inproc=ssum["n_extrapolated"],
                                              onetime_empty=ssum["n_extrapolated"], shapes_used_by_panel=len(used),
                                              extrapolated_shapes_used_by_panel=len(ext_used), request_share_on_extrapolated_shapes=req_ext),
                    capture_pool_note="capture_pool_mb of C1+G1 unchanged (the confirm block did not measure the recorded-graph pool)",
                    imputation_log=ilog if False else "impute_empty() applied before the star modification, same as formal evaluation")
        fpath.with_name(fpath.name.replace("_raw.csv", "_meta.json")).write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
        print(f"[{tag}] rows={len(df)} status={st} wall={time.time()-t0:.0f}s -> {fpath}", flush=True)
        if a.split == "eval":
            os.chmod(fpath, 0o444)
            os.chmod(fpath.with_name(fpath.name.replace("_raw.csv", "_meta.json")), 0o444)


if __name__ == "__main__":
    main()
