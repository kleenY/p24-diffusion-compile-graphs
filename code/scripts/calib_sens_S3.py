"""S3 appendix: one-time-cost calibration sensitivity analysis (CPU only). Implements results/S3/calib_sens/SPEC_calib_sens.txt (frozen, SHA16 148f2f315fe7cbee).

Stages (python -B scripts/calib_sens_S3.py --stage <name>; run from code/ after `source ../ops/project_env.sh`; CUDA_VISIBLE_DEVICES='' nice -n 10):
  check0a    event-level decomposition (sim - measured) of the 3.4 error: steady / one-time-fallback / one-time-non-fallback
  check0b    S2 raw meta: load1 vs C1 (and C1+G1) cold_diskcache fc_unet (Spearman)
  tables     build calibrated cost-table COPIES (variant A = S2-only ratios, variant B = A + k_comp), write CSV + meta
  identity   implementation check: identity copy (no fill, k=1) must reproduce frozen main_raw.csv bit-for-bit on one grid cell
  run        --set A_i|A_ii|B_i|B_ii : the 'main' experiment grid on the calibrated table (perspective i = policy sees the frozen table,
             simulator charges the calibrated one; ii = both calibrated)
  analyze    H4 verdicts (same code as h4_verdict_S3), differences vs frozen, worst cases, flips, robust/not robust
  selfcheck  3.4 predictions on the calibrated tables + new error table (in-sample for B)
Nothing frozen is modified (cost_table.csv, costs.py, addendum, simulator, policies, main_raw.csv are read only); outputs go to results/S3/calib_sens/
and an existing output is never overwritten.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import multiprocessing as mp
import pathlib
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
import scripts.sim_eval_S3 as S  # noqa: E402
from p24.policy import CostView, component_keys, make_policy, shape_of  # noqa: E402
from p24.policy.base import Shape  # noqa: E402
from p24.sim import simulator as SIM  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402
from p24.sim.costs import COMPILE_ARMS, COMPONENTS, CostTable, MissingCost  # noqa: E402

OUT = ROOT / "results/S3/calib_sens"
RV = ROOT / "results/S3/real_validation"
TABDIR = OUT / "tables"
FROZEN_RAW = ROOT / "results/S3/sim_eval/main_raw.csv"
ARMS_POOL = ["C1", "C1+G1", "C2", "C5"]          # spec: pooled compile arms
ITEMS = ["resid_s"] + [f"fc_{c}_s" for c in COMPONENTS]
CACHES = ["diskcache", "empty"]
sha16 = S.sha16


def wr(path: pathlib.Path, text: str):
    path = pathlib.Path(path)
    if path.exists():
        sys.exit(f"REFUSING to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def wcsv(df: pd.DataFrame, path: pathlib.Path, **kw):
    path = pathlib.Path(path)
    if path.exists():
        sys.exit(f"REFUSING to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kw)


# ============================================================================================ frozen table (exactly as sim_eval_S3 main)
def frozen_table() -> CostTable:
    """impute_empty(S2 table) + addendum overlay = what sim_eval_S3 --missing impute uses (W['ct0i'])."""
    add = ADD.read_addendum(S.ADDENDUM_PATH)
    return ADD.load_with_addendum(S.impute_empty(pd.read_csv(S.CT_PATH))[0], add)


def table_from_df(df: pd.DataFrame, rebuild_override, report=None) -> CostTable:
    ct = ADD.AddendumCostTable(df)
    ct.rebuild_override = dict(rebuild_override)
    ct.addendum_report = report or {}
    return ct


# ============================================================================================ calibration ratios
def pair_ratios(raw: pd.DataFrame):
    """r tables from the S2 cost table (pre-impute, pre-addendum): ratio inproc/cold of the SAME cache state, SAME item, rows whose source is
    `measured` and where both tiers are measured (non-NaN, cold > 0). -> (r dict, detail DataFrame)."""
    meas = raw[raw.source == "measured"]
    pairs = {}
    for arm in sorted(raw.arm.unique()):
        for cache in CACHES:
            for item in ITEMS:
                ci, cc = f"{item}__inproc_{cache}", f"{item}__cold_{cache}"
                m = meas[(meas.arm == arm) & meas[ci].notna() & meas[cc].notna() & (meas[cc] > 0)]
                pairs[(arm, cache, item)] = (m[ci] / m[cc]).to_numpy(float)
    r, rows = {}, []
    for arm in sorted(raw.arm.unique()):
        for cache in CACHES:
            for item in ITEMS:
                own = pairs[(arm, cache, item)]
                if arm in ("E0", "G1"):
                    # E0/G1 never compile (cache independent): merged E0+G1 median. inproc_empty cells of E0/G1 do not exist in S2, the
                    # same merged inproc_diskcache ratio is used for the empty-cache state (physically cache independent).
                    vals = np.concatenate([pairs[("E0", "diskcache", item)], pairs[("G1", "diskcache", item)]]) if item == "resid_s" else np.array([])
                    scope = "E0+G1 merged (diskcache state)"
                elif len(own) >= 3:
                    vals, scope = own, f"own arm {arm}"
                else:
                    vals = np.concatenate([pairs[(a, cache, item)] for a in ARMS_POOL])
                    scope = "pooled compile arms C1,C1+G1,C2,C5"
                val = float(np.median(vals)) if len(vals) else np.nan
                r[(arm, cache, item)] = val
                rows.append(dict(arm=arm, cache=cache, item=item, r=val, n_pairs_used=int(len(vals)), n_pairs_own_arm=int(len(own)), scope=scope))
    return r, pd.DataFrame(rows)


def fill_inproc(df: pd.DataFrame, r: dict):
    """Fill every missing inproc_<cache> cell (resid_s__inproc_<cache> NaN) of df with cold_<cache> value x r (resid and fc_*, fc only if the cold
    value exists and the arm is a compile arm). Cells that already have an inproc resid are NOT touched. Returns (new df, log DataFrame)."""
    df = df.copy()
    log = []
    for cache in CACHES:
        res_in, res_cold = f"resid_s__inproc_{cache}", f"resid_s__cold_{cache}"
        miss = df[res_in].isna() & df[res_cold].notna()
        for arm in sorted(df.arm.unique()):
            m = miss & (df.arm == arm)
            n = int(m.sum())
            if not n:
                continue
            for item in ITEMS:
                if item != "resid_s" and arm not in COMPILE_ARMS:
                    continue
                rv = r[(arm, cache, item)]
                cold, inp = f"{item}__cold_{cache}", f"{item}__inproc_{cache}"
                if not np.isfinite(rv):
                    log.append(dict(arm=arm, cache=cache, item=item, n_filled=0, r=rv, note="no ratio"))
                    continue
                mm = m & df[cold].notna()
                df.loc[mm, inp] = df.loc[mm, cold] * rv
                log.append(dict(arm=arm, cache=cache, item=item, n_filled=int(mm.sum()), n_cells_missing=n, r=rv, note=""))
    return df, pd.DataFrame(log)


# ============================================================================================ k_comp from 3.4 B1 (non-fallback compile events)
def read_measured_events(pol: str, stream: str):
    """3 processes of 3.4 raw/<pol>_<stream>_p*.csv -> list of DataFrames (req phase, sorted by idx)."""
    out = []
    for f in sorted(glob.glob(str(RV / f"raw/{pol}_{stream}_p*.csv"))):
        d = pd.read_csv(f)
        d = d[d.phase == "req"].sort_values("idx").reset_index(drop=True)
        out.append((pathlib.Path(f).name, d))
    return out


STREAMS = {"W2": "W2_N300_seed100_B", "W3": "W3_N300_seed100_B", "W5": "W5_N300_seed100_B"}


def k_comp_events(ct_frozen: CostTable, P: dict) -> pd.DataFrame:
    """One row per (stream, process file, event idx, component) with measured first-call seconds and the table value the frozen simulator
    charges, for compile events of B1 whose component is NOT a fallback in the simulator event (fallbacks has no 'fc_<c>:' entry) and whose
    component key is new in both the simulation and the measurement."""
    rows = []
    prior = S.prior_A()
    for w, stem in STREAMS.items():
        reqs = SIM.read_trace(ROOT / "results/S3/workload/real_validation" / f"{stem}.jsonl")
        cv = CostView(ct_frozen, "warm", rebuild_override=ct_frozen.rebuild_override)
        res = SIM.simulate(reqs, make_policy("B1", cost=cv, params=dict(P), prior=prior), ct_frozen, SIM.SimOptions(cache_mode="warm", overhead_aware=True))
        ev = res.frame()
        ev = ev[ev.idx >= 0].reset_index(drop=True)
        # simulated key-new tracking (B1 compiles every new shape)
        seen, simrows = set(), []
        for e in ev.itertuples():
            shape = Shape(e.H, e.W, e.B, e.cfg)
            keys = component_keys(shape)
            if e.event == "compile":
                ot = ct_frozen.one_time("C1", shape, e.tier)
                for c in COMPONENTS:
                    fb = any(f.startswith(f"fc_{c}:") for f in str(e.fallbacks).split(";") if f)
                    simrows.append(dict(idx=e.idx, comp=c, tier=e.tier, table_s=ot.fc[c], fallback=fb, sim_new=keys[c] not in seen, shape=shape.tag()))
                seen.update(keys.values())
        sr = pd.DataFrame(simrows)
        for fname, d in read_measured_events("B1", stem):
            for r in sr.itertuples():
                m = d[d.idx == r.idx]
                if m.empty:
                    continue
                m = m.iloc[0]
                pj = json.loads(m.prep_json)
                rows.append(dict(workload=w, file=fname, idx=r.idx, comp=r.comp, tier=r.tier, shape=r.shape, sim_new=r.sim_new, fallback=r.fallback,
                                 meas_new=int(pj.get(f"first_call_{r.comp}_new_graphs", 0)) > 0, meas_s=float(pj.get(f"first_call_{r.comp}_s", np.nan)),
                                 table_s=r.table_s, meas_event=m.event))
    return pd.DataFrame(rows)


def k_comp_from_events(kev: pd.DataFrame):
    sel = kev[(~kev.fallback) & kev.sim_new & kev.meas_new & (kev.table_s > 0) & kev.meas_s.notna()].copy()
    sel["ratio"] = sel.meas_s / sel.table_s
    k, info = {}, {}
    for c in COMPONENTS:
        s = sel[sel.comp == c]
        k[c] = float(s.ratio.median())
        info[c] = dict(k=k[c], n=int(len(s)), n_files=int(s.file.nunique()), q25=float(s.ratio.quantile(.25)), q75=float(s.ratio.quantile(.75)),
                       n_first_shape_cold=int((s.tier == "cold_diskcache").sum()), n_inproc=int((s.tier != "cold_diskcache").sum()),
                       median_cold=float(s[s.tier == "cold_diskcache"].ratio.median()) if (s.tier == "cold_diskcache").any() else None,
                       median_inproc=float(s[s.tier != "cold_diskcache"].ratio.median()) if (s.tier != "cold_diskcache").any() else None)
    info["_n_excluded"] = dict(fallback=int(kev.fallback.sum()), key_mismatch=int((kev.sim_new != kev.meas_new).sum()), total=int(len(kev)))
    return k, info


def scale_c1(df: pd.DataFrame, k: dict) -> pd.DataFrame:
    """Variant B: fc_te/unet/vae of C1 and C1+G1, ALL tiers, times k_comp[component]."""
    df = df.copy()
    m = df.arm.isin(["C1", "C1+G1"])
    for c in COMPONENTS:
        for t in ["cold_empty", "cold_diskcache", "inproc_empty", "inproc_diskcache"]:
            col = f"fc_{c}_s__{t}"
            df.loc[m, col] = df.loc[m, col] * k[c]
    return df


def build_tables():
    raw = pd.read_csv(S.CT_PATH)
    add = ADD.read_addendum(S.ADDENDUM_PATH)
    r, rdf = pair_ratios(raw)
    imp, _ = S.impute_empty(raw)
    base, ov, rep = ADD.apply_addendum(imp, add)           # exactly the frozen pipeline's table, before the calibration fill
    dfA, logA = fill_inproc(base, r)
    P = S.base_params(S.load_params())
    ctf = frozen_table()
    kev = k_comp_events(ctf, P)
    k, kinfo = k_comp_from_events(kev)
    dfB = scale_c1(dfA, k)
    TABDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in (("A", dfA), ("B", dfB)):
        p = TABDIR / f"cost_table_calib_{name}.csv"
        wcsv(df, p)
        out[name] = p
    # implementation-check copies (not results): ID = frozen-pipeline table copied through CSV (no fill, k=1); FILL1 = fill with every r forced to 1
    wcsv(base, TABDIR / "cost_table_calib_ID.csv")
    r1 = {k_: 1.0 for k_ in r}
    wcsv(fill_inproc(base, r1)[0], TABDIR / "cost_table_calib_FILL1.csv")
    wcsv(rdf, TABDIR / "ratios_r.csv")
    wcsv(logA, TABDIR / "fill_log_A.csv")
    wcsv(kev, TABDIR / "k_comp_events.csv")
    # cells changed vs frozen
    key = ["arm", "H", "W", "B", "cfg"]
    fz = base.set_index(key)
    chg = {}
    for name, df in (("A", dfA), ("B", dfB)):
        d = df.set_index(key)
        n = 0
        for c in d.columns:
            if not pd.api.types.is_numeric_dtype(d[c]) or not pd.api.types.is_numeric_dtype(fz[c]):
                continue
            a, b = fz[c].astype(float), d[c].astype(float)
            n += int((~((a == b) | (a.isna() & b.isna()))).sum())
        chg[name] = n
    meta = dict(time=time.strftime("%FT%T"), spec_sha16="148f2f315fe7cbee", spec_file=str(OUT / "SPEC_calib_sens.txt"), cost_table_sha16=sha16(S.CT_PATH),
                addendum_sha16=sha16(S.ADDENDUM_PATH), script_sha16=sha16(__file__), tables={n: dict(path=str(p), sha16=sha16(p)) for n, p in out.items()},
                pipeline="S2 cost_table.csv -> sim_eval_S3.impute_empty (cold_empty imputation, as frozen main) -> addendum overlay (22 inproc cells + 68 rebuild overrides) "
                         "-> variant A: fill missing inproc_<cache> cells = cold_<cache> x r -> variant B: A with C1/C1+G1 fc_* x k_comp (all tiers)",
                r_table=[{**row, "r": (None if not np.isfinite(row["r"]) else row["r"])} for row in rdf.to_dict("records")],
                k_comp=kinfo, n_numeric_cells_changed_vs_frozen_pipeline_table=chg,
                note="r and k_comp are computed by this script from the tables / 3.4 raw (not typed in). empty-cache k_comp is a hypothesis (no 3.4 empty-cache B1 measurement).")
    wr(TABDIR / "calib_tables_meta.json", json.dumps(meta, indent=1, ensure_ascii=False, default=str))
    print(rdf.round(4).to_string())
    print(json.dumps(kinfo, indent=1))
    print("changed cells", chg)
    print(logA.groupby(["arm", "cache"]).n_filled.sum().to_string())


def load_cal_table(name: str) -> CostTable:
    """Load a calibrated table copy (CSV) + the frozen rebuild overrides. name in A | B | ID (identity copy)."""
    p = TABDIR / f"cost_table_calib_{name}.csv"
    ov = ADD.apply_addendum(pd.read_csv(S.CT_PATH), ADD.read_addendum(S.ADDENDUM_PATH))[1]
    return table_from_df(pd.read_csv(p), ov)


# ============================================================================================ main-grid runner on (policy table, sim table)
SET_DEF = {"A_i": ("A", "i"), "A_ii": ("A", "ii"), "B_i": ("B", "i"), "B_ii": ("B", "ii"), "ID_i": ("ID", "i"), "ID_ii": ("ID", "ii"),
           "FILL1_i": ("FILL1", "i")}


def _init(cfg: dict):
    S._init_worker(cfg)                                     # frozen table (impute+addendum), traces, priors, simulator hook
    ctf = S.W["ct0i"]
    W = S.W
    W["ct_policy"] = ctf if cfg["persp"] == "i" else None
    W["ct_sim"] = load_cal_table(cfg["table"])
    if cfg["persp"] == "ii":
        W["ct_policy"] = W["ct_sim"]


def run_task(task):
    """Same as sim_eval_S3.run_task (boot=-1), but the policy's CostView and the simulator's table may differ."""
    boot, traces = task
    W = S.W
    ex = W["exp"]
    cts, ctp = W["ct_sim"], W["ct_policy"]
    rows = []
    views = {}
    for (w, n, seed, tkind) in traces:
        reqs = W["store"].get(w, n, seed, tkind)
        import collections
        oc = dict(collections.Counter(shape_of(q) for q in reqs))
        for budget in ex.budgets:
            mb = W["budgets"][budget]
            for cold in ex.colds:
                cm, hot = S.COLDS[cold]
                S._set_hot(hot)
                cv = views.setdefault(cm, CostView(ctp, cm, rebuild_override=getattr(ctp, "rebuild_override", None)))
                opt = SIM.SimOptions(cache_mode=cm, mem_budget_mb=mb, overhead_aware=True)
                for v in ex.variants:
                    row = dict(experiment=ex.name, split=W["split"], variant=v.label, policy=v.policy, workload=w, N=n, seed=seed,
                               budget=budget, budget_mb=mb, cold=cold, prior_kind=v.prior, trace_kind=tkind, boot=boot,
                               imputed=int(W["impute"] and cm == "empty"))
                    t0 = time.perf_counter()
                    try:
                        pol = make_policy(v.policy, cost=cv, params=v.params, prior=W["priors"][v.prior], oracle_counts=oc)
                        s = SIM.simulate(reqs, pol, cts, opt).summary
                        row.update(status="ok", error="", service_s=s["service_s"], load_s=s["load_s"], warmup_s=s["warmup_s"],
                                   total_incl_load_s=s["total_incl_load_s"], peak_graph_mb=s["peak_graph_mb"],
                                   peak_reserved_est_mb=s["peak_reserved_est_mb"], n_requests=s["n_requests"], n_shapes=s["n_shapes"],
                                   n_capture=s.get("n_capture", 0), n_compile=s.get("n_compile", 0),
                                   n_compile_capture=s.get("n_compile_capture", 0), n_capture_denied=s.get("n_capture_denied", 0),
                                   n_eager_first=s.get("n_eager_first", 0), n_eager_fallback_limit=s.get("n_eager_fallback_limit", 0),
                                   n_evicted=s["n_evicted_total"], n_denied=s["n_denied_total"], n_steady=s.get("n_steady", 0),
                                   hit_rate=s.get("n_steady", 0) / max(1, s["n_requests"]), n_events_with_fallback=s["n_events_with_fallback"])
                    except MissingCost as e:
                        row.update(status="missing_cost", error=str(e)[:160])
                    except Exception as e:  # noqa: BLE001
                        row.update(status="error", error=f"{type(e).__name__}: {str(e)[:160]}")
                    row["sim_wall_s"] = time.perf_counter() - t0
                    rows.append(row)
    S._set_hot(False)
    return rows


def run_set(setname: str, out_dir: pathlib.Path, procs: int = 8, tiny=False, cell=None):
    table, persp = SET_DEF[setname]
    pj = S.load_params()
    info = S.guards("eval", pj)
    P = S.base_params(pj)
    a = types.SimpleNamespace(seeds=None, N=None, colds=None, workloads=None, budgets=None, policies=None)
    if cell:                                        # (workload, N, cold) restricted grid for the identity check
        a.workloads, a.N, a.colds = [cell[0]], [cell[1]], [cell[2]]
        a.seeds = [100]
    ex = S.build_exp("main", "eval", P, a)
    sp = S.SPLITS["eval"]
    bj = json.loads(S.BUDGETS_PATH.read_text())["tiers"]
    budgets = {k: float(v["mb"]) for k, v in bj.items()}
    priors = S.make_priors(False)
    manifest = S.read_manifest()
    specs = [(w, n, s, "file") for w in ex.workloads for n in ex.Ns for s in ex.seeds]
    tasks = [(-1, [sp_]) for sp_ in specs]
    cfg = dict(split="eval", manifest=manifest, exp=ex, budgets=budgets, priors=priors, impute=True, need_procs=False,
               addendum=str(S.ADDENDUM_PATH), persp=persp, table=table)
    csv_path = out_dir / f"calib_{setname}_raw.csv"
    if csv_path.exists():
        sys.exit(f"REFUSING to overwrite {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    print(f"[{setname}] table={table} perspective={persp} tasks={len(tasks)} procs={procs}", flush=True)
    with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
            rows += r
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f"[{setname}] {i+1}/{len(tasks)} tasks {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    key = ["experiment", "split", "variant", "workload", "N", "seed", "budget", "cold", "prior_kind", "trace_kind", "boot"]
    df = df.sort_values(key).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    meta = dict(set=setname, table=table, perspective=persp, n_rows=len(df), status_counts=df.status.value_counts().to_dict(), wall_s=time.time() - t0,
                time=time.strftime("%FT%T"), spec_sha16="148f2f315fe7cbee", script_sha16=sha16(__file__), guards=info, cell=cell,
                table_sha16=sha16(TABDIR / f"cost_table_calib_{table}.csv"), procs=procs,
                perspective_def="i: policies (and B7 oracle) built on the FROZEN cost table (impute+addendum), simulator charges the calibrated table; "
                                "ii: policies and simulator use the calibrated table")
    (out_dir / f"calib_{setname}_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    os.chmod(csv_path, 0o444)
    os.chmod(out_dir / f"calib_{setname}_meta.json", 0o444)
    print(f"[{setname}] rows={len(df)} status={meta['status_counts']} wall={time.time()-t0:.0f}s -> {csv_path}", flush=True)


# ============================================================================================ stage dispatch (the rest lives in calib_sens_S3_analysis.py)
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["tables", "run"])
    ap.add_argument("--set", choices=sorted(SET_DEF))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cell", help="WORKLOAD,N,COLD (restricted one-cell grid, seed 100) for the identity check")
    a = ap.parse_args()
    assert 1 <= a.procs <= 8
    if a.stage == "tables":
        build_tables()
    elif a.stage == "run":
        cell = None
        if a.cell:
            w, n, c = a.cell.split(",")
            cell = (w, int(n), c)
        run_set(a.set, pathlib.Path(a.out), a.procs, cell=cell)


if __name__ == "__main__":
    main()
