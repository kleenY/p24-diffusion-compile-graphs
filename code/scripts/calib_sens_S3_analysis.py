"""S3 appendix calibration sensitivity: checks and analysis stages (CPU only). Spec: results/S3/calib_sens/SPEC_calib_sens.txt (SHA16 148f2f315fe7cbee).
Companion of scripts/calib_sens_S3.py (tables + grid runs).

  python -B scripts/calib_sens_S3_analysis.py --stage check0a | check0b | identity | analyze | selfcheck
(run from code/ after `source ../ops/project_env.sh`; CUDA_VISIBLE_DEVICES=''). Output only to results/S3/calib_sens/, never overwrites.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
import scripts.calib_sens_S3 as C  # noqa: E402
import scripts.sim_eval_S3 as S  # noqa: E402
import scripts.h4_verdict_S3 as H4  # noqa: E402
import scripts.predict_S3_4 as P34  # noqa: E402
import scripts.compare_S3_4 as CMP  # noqa: E402
from p24.policy import CostView, component_keys, make_policy, shape_of  # noqa: E402
from p24.sim import simulator as SIM  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402
from p24.sim.costs import CostTable  # noqa: E402

OUT = C.OUT
RV = C.RV
fmt = lambda x: f"{x:+.2%}"


# ============================================================================================ 0a event-level decomposition
def plain_frozen_table() -> CostTable:
    """The table predict_S3_4.py used: S2 table + addendum, NO impute (3.4 prediction)."""
    return ADD.load_with_addendum(S.CT_PATH, ADD.read_addendum(S.ADDENDUM_PATH))


def sim_events(ct_sim, ct_policy, w, pol, cm, P, prior, budget_mb, inp):
    stem = P34.STREAMS[w]
    reqs = SIM.read_trace(P34.RV / f"{stem}.jsonl")
    bud = budget_mb if pol in P34.BUDGETED else None
    cv = CostView(ct_policy, cm, rebuild_override=getattr(ct_policy, "rebuild_override", None))
    polobj = make_policy(P34.SIM_NAME.get(pol, pol), cost=cv, params=dict(P), prior=prior)
    res = SIM.simulate(reqs, polobj, ct_sim, SIM.SimOptions(cache_mode=cm, mem_budget_mb=bud, overhead_aware=True))
    ev = res.frame()
    ev = ev[ev.idx >= 0].reset_index(drop=True)
    inp.n_extrap = 0
    ev["input_s"] = [inp.ms(shape_of(r)) / 1e3 for r in reqs]
    return res, ev, reqs


def measured_per_idx(rawdir: pathlib.Path, pol: str, stem: str):
    """-> (per-idx median over processes of dur_ms/1000, event labels (mode) per idx, n_proc)"""
    pn = pol.replace("′", "p").replace("'", "p")
    ds = []
    for f in sorted(glob.glob(str(rawdir / f"{pn}_{stem}_p*.csv"))):
        d = pd.read_csv(f)
        d = d[d.phase == "req"].sort_values("idx").set_index("idx")
        ds.append(d)
    dur = pd.concat([d.dur_ms for d in ds], axis=1).median(axis=1) / 1e3
    evt = pd.concat([d.event for d in ds], axis=1).mode(axis=1)[0]
    return dur, evt, len(ds)


def check0a():
    P = C.S.base_params(C.S.load_params())
    ct = plain_frozen_table()
    ADD.install_simulator_hook()
    prior = S.prior_A()
    inp = P34.InputModel(P34.INPUT_PATH)
    budget = float(json.loads(P34.BUDGETS_PATH.read_text())["tiers"]["default"]["mb"])
    errtab = pd.read_csv(RV / "error_table.csv")
    predtab = pd.read_csv(RV / "pred_S3_4.csv")
    runs = []
    for w in ("W2", "W3", "W5"):
        for pol in ("B0", "B1", "B4", "Ours"):
            runs.append((w, pol, "warm", RV / "raw"))
        if w == "W3":
            runs.append((w, "B3p", "warm", RV / "raw"))
    runs.append(("W5", "Ours", "empty", RV / "raw_empty"))
    rows, cls_rows = [], []
    for (w, pol, cm, rawdir) in runs:
        stem = P34.STREAMS[w]
        res, ev, reqs = sim_events(ct, ct, w, pol, cm, P, prior, budget, inp)
        pe = predtab[(predtab.stream == stem) & (predtab.policy.str.replace("′", "p") == pol) & (predtab.cache_mode == cm)].iloc[0]
        assert abs(res.summary["service_s"] - pe.pred_excl_input_s) < 1e-6, (w, pol, cm, res.summary["service_s"], pe.pred_excl_input_s)
        dur, evt, nproc = measured_per_idx(rawdir, pol, stem)
        ev["sim_total_s"] = ev.dur_s + ev.input_s
        ev["meas_s"] = ev.idx.map(dur)
        ev["meas_event"] = ev.idx.map(evt)
        ev["fb"] = ev.fallbacks.fillna("").str.len() > 0
        ev["klass"] = np.where(ev.event == "steady", "steady", np.where(ev.fb, "onetime_fallback", "onetime_nonfallback"))
        ev["diff_s"] = ev.sim_total_s - ev.meas_s
        mism = int((ev.event != ev.meas_event).sum())
        ee = errtab[(errtab.stream == stem) & (errtab.policy.str.replace("′", "p") == pol) & (errtab.cache_mode == cm)].iloc[0]
        total_err = float(ee.sim_s - ee.measured_median_s)           # official (sim - median of process totals)
        sum_diff = float(ev.diff_s.sum())
        for k, g in ev.groupby("klass"):
            cls_rows.append(dict(stream=w, policy=pol, cache=cm, klass=k, n_events=len(g), sim_s=g.sim_total_s.sum(), meas_s=g.meas_s.sum(),
                                 diff_s=g.diff_s.sum(), share_of_event_diff=g.diff_s.sum() / sum_diff if sum_diff else np.nan,
                                 share_of_official_total_error=g.diff_s.sum() / total_err if total_err else np.nan))
        for (k, e), g in ev.groupby(["klass", "event"]):
            if k != "steady":
                cls_rows.append(dict(stream=w, policy=pol, cache=cm, klass=f"{k}/{e}", n_events=len(g), sim_s=g.sim_total_s.sum(), meas_s=g.meas_s.sum(),
                                     diff_s=g.diff_s.sum(), share_of_event_diff=g.diff_s.sum() / sum_diff if sum_diff else np.nan,
                                     share_of_official_total_error=g.diff_s.sum() / total_err if total_err else np.nan))
        rows.append(dict(stream=w, policy=pol, cache=cm, n_proc=nproc, sim_total_s=float(ee.sim_s), meas_median_total_s=float(ee.measured_median_s),
                         official_error_s=total_err, official_error_rel=float(ee.error), sum_event_diff_s=sum_diff,
                         residual_s=total_err - sum_diff, n_event_type_mismatch=mism,
                         n_fallback_events=int(ev.fb.sum()), n_onetime_events=int((ev.event != "steady").sum())))
    tot = pd.DataFrame(rows)
    cls = pd.DataFrame(cls_rows)
    C.wcsv(tot, OUT / "check0a_totals.csv")
    C.wcsv(cls, OUT / "check0a_decomposition.csv")
    pd.set_option("display.width", 250)
    print(tot.round(3).to_string())
    main = cls[~cls.klass.str.contains("/")]
    print(main.round(3).to_string())
    # aggregate over the 13 judged warm combos (sum of seconds / sum of official errors)
    warm = main[main.cache == "warm"].groupby("klass")[["sim_s", "meas_s", "diff_s"]].sum()
    warm["share"] = warm.diff_s / warm.diff_s.sum()
    print(warm.round(3).to_string())


def check0a_comp():
    """B1 component-level split of the one-time error (supplement to 0a): per (stream, process file) sum over compile events of
    (table value the simulator charges - measured first call) for components whose key is new, split by component and by fallback flag;
    median over the 3 processes per stream, then summed over streams. (Residual `resid_s` part not split here: see 0a totals.)"""
    kev = pd.read_csv(OUT / "tables/k_comp_events.csv")
    k = kev[kev.sim_new & kev.meas_new].copy()
    k["diff_s"] = k.table_s - k.meas_s
    g = k.groupby(["workload", "file", "comp", "fallback"]).agg(n=("diff_s", "size"), table_s=("table_s", "sum"), meas_s=("meas_s", "sum"), diff_s=("diff_s", "sum")).reset_index()
    m = g.groupby(["workload", "comp", "fallback"]).agg(n=("n", "median"), table_s=("table_s", "median"), meas_s=("meas_s", "median"), diff_s=("diff_s", "median")).reset_index()
    tot = m.groupby(["comp", "fallback"])[["n", "table_s", "meas_s", "diff_s"]].sum().reset_index()
    tot["meas_over_table"] = tot.meas_s / tot.table_s
    C.wcsv(m, OUT / "check0a_B1_component_split_by_stream.csv")
    C.wcsv(tot, OUT / "check0a_B1_component_split.csv")
    print(tot.round(3).to_string())
    print("sum over components/streams: fallback-flag diff", round(tot[tot.fallback].diff_s.sum(), 2), "non-fallback diff", round(tot[~tot.fallback].diff_s.sum(), 2))


# ============================================================================================ 0b load vs fc_unet
def find_meta(block, src, arm, H, W, B, cfg, pid, seq_idx):
    S2 = ROOT / "results/S2"
    for base in (S2 / "cost_table" / block / src, S2 / "cold_tiers" / src, S2 / "inproc_seq" / src):
        if not base.is_dir():
            continue
        names = [f"{arm}_{H}x{W}_b{B}_{'cfg' if cfg else 'nocfg'}_p{pid}.meta.json", f"{arm}_seq_p{pid}.meta.json"]
        for n in names:
            if (base / n).exists():
                return base / n
    return None


def check0b():
    from scipy.stats import spearmanr
    procs = pd.read_csv(S.PROCS_PATH)
    d = procs[procs.arm.isin(["C1", "C1+G1"]) & (procs.tier == "cold_diskcache") & procs.first_call_unet_s.notna() & (procs.tier_violation == 0)].copy()
    rows = []
    for r in d.itertuples():
        mp_ = find_meta(r.block, r.src, r.arm, r.H, r.W, r.B, r.cfg, int(r.proc_id), r.seq_idx)
        if mp_ is None:
            rows.append(dict(block=r.block, src=r.src, arm=r.arm, H=r.H, W=r.W, B=r.B, cfg=r.cfg, proc_id=r.proc_id, meta="NOT FOUND"))
            continue
        m = json.loads(mp_.read_text())
        la = m.get("loadavg") or [np.nan]
        le = m.get("loadavg_end") or [np.nan]
        rows.append(dict(block=r.block, src=r.src, arm=r.arm, H=r.H, W=r.W, B=r.B, cfg=r.cfg, proc_id=r.proc_id, meta=str(mp_.relative_to(ROOT)),
                         load1_start=float(la[0]), load1_end=float(le[0]), fc_unet_s=float(r.first_call_unet_s), fc_te_s=float(r.first_call_te_s),
                         fc_vae_s=float(r.first_call_vae_s), first_req_s=float(r.first_req_s)))
    x = pd.DataFrame(rows)
    ok = x[x.meta != "NOT FOUND"].dropna(subset=["load1_start", "fc_unet_s"]).copy()
    ok["load1_mean"] = ok[["load1_start", "load1_end"]].mean(axis=1)
    # shape-normalised fc_unet (divide by the median of the same arm+shape) removes the shape confound
    ok["fc_unet_rel"] = ok.fc_unet_s / ok.groupby(["arm", "H", "W", "B", "cfg"]).fc_unet_s.transform("median")
    C.wcsv(x, OUT / "check0b_processes.csv")
    res = dict(n_rows_selected=int(len(d)), n_meta_not_found=int((x.meta == "NOT FOUND").sum()), n_used=int(len(ok)))
    for sub, g in (("C1+C1G1", ok), ("C1", ok[ok.arm == "C1"]), ("C1+G1", ok[ok.arm == "C1+G1"])):
        for ld in ("load1_start", "load1_end", "load1_mean"):
            for yv in ("fc_unet_s", "fc_unet_rel"):
                rho, p = spearmanr(g[ld], g[yv])
                res[f"{sub}|{ld}|{yv}"] = dict(rho=float(rho), p=float(p), n=int(len(g)))
    res["load1_start_range"] = [float(ok.load1_start.min()), float(ok.load1_start.max())]
    res["load1_start_quantiles_5_50_95"] = [float(v) for v in np.percentile(ok.load1_start, [5, 50, 95])]
    res["fc_unet_s_quantiles_5_50_95"] = [float(v) for v in np.percentile(ok.fc_unet_s, [5, 50, 95])]
    prim = res["C1+C1G1|load1_start|fc_unet_s"]
    res["primary"] = "C1+C1G1, load1_start vs fc_unet_s"
    res["verdict_rule"] = "|rho|<0.3 or p>0.05 -> report 'cause not verified'"
    res["verdict"] = "reason not verified (the load hypothesis is not supported by the data)" if (abs(prim["rho"]) < 0.3 or prim["p"] > 0.05) else "load is significantly correlated with fc_unet (|rho|>=0.3 and p<=0.05)"
    # also: S2 loads vs 3.4 loads, binned compile time by load
    ok["load_bin"] = pd.cut(ok.load1_start, [0, 2, 5, 8, 100])
    res["by_load_bin_median_fc_unet"] = {str(k): dict(n=int(len(g)), median_fc_unet=float(g.fc_unet_s.median())) for k, g in ok.groupby("load_bin")}
    # magnitude (not only significance): Theil-Sen slope of the shape-normalised fc_unet on load1_start, per-block rho (time confound),
    # and the (out-of-range!) extrapolation to the 3.4 load level
    from scipy.stats import theilslopes
    sl, ic, lo_, hi_ = theilslopes(ok.fc_unet_rel, ok.load1_start)
    l_ref = float(ok.load1_start.median())
    res["theil_sen_fc_unet_rel_per_load1"] = dict(slope=float(sl), ci95=[float(lo_), float(hi_)], load1_median_S2=l_ref,
                                                  fitted_rel_at_median_load=float(ic + sl * l_ref),
                                                  fitted_rel_at_load1_eq_1_EXTRAPOLATED=float(ic + sl * 1.0),
                                                  implied_ratio_load1_vs_median_load=float((ic + sl * 1.0) / (ic + sl * l_ref)),
                                                  note="S2 load1 range [3.6, 9.7]; load 1 (3.4 level) is outside the data range")
    res["per_block_spearman_load1_start_vs_fc_unet_rel"] = {
        str(b): dict(n=int(len(g)), rho=(float(spearmanr(g.load1_start, g.fc_unet_rel)[0]) if len(g) > 4 and g.load1_start.nunique() > 1 else None),
                     p=(float(spearmanr(g.load1_start, g.fc_unet_rel)[1]) if len(g) > 4 and g.load1_start.nunique() > 1 else None),
                     load1_median=float(g.load1_start.median()), fc_unet_rel_median=float(g.fc_unet_rel.median()))
        for b, g in ok.groupby("block")}
    C.wr(OUT / "check0b_result.json", json.dumps(res, indent=1, ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a_, ld in zip(ax, ("load1_start", "load1_end")):
            for arm, g in ok.groupby("arm"):
                a_.scatter(g[ld], g.fc_unet_s, s=12, label=arm, alpha=.7)
            a_.set_xlabel(ld)
            a_.set_ylabel("C1 cold_diskcache fc_unet (s)")
            a_.legend()
        fig.tight_layout()
        p = OUT / "check0b_scatter.png"
        if not p.exists():
            fig.savefig(p, dpi=110)
    except Exception as e:  # noqa: BLE001
        print("plot skipped", e)
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ============================================================================================ identity
def identity():
    fz = pd.read_csv(C.FROZEN_RAW)
    key = ["variant", "workload", "N", "seed", "cold"]
    rows = []
    for s in ("ID_i", "ID_ii", "FILL1_i"):
        for c in ("disk", "empty"):
            d = pd.read_csv(glob.glob(str(OUT / f"identity/{s}-{c}/calib_{s}_raw.csv"))[0])
            f = fz[(fz.workload == "W2") & (fz.N == 1000) & (fz.seed == 100) & (fz.cold == c)]
            m = d.merge(f, on=key, suffixes=("", "_f"))
            cols = [x for x in d.columns if x not in key + ["sim_wall_s"] and x + "_f" in m]
            bad = {}
            for x in cols:
                a, b = m[x], m[x + "_f"]
                ne = ~((a == b) | (a.isna() & b.isna()))
                if ne.any():
                    bad[x] = int(ne.sum())
            rows.append(dict(set=s, cell=f"W2,N=1000,seed=100,cold={c}", n_rows=len(m), n_cols_compared=len(cols), columns_differing=json.dumps(bad),
                             bit_identical=int(not bad),
                             max_abs_rel_service_diff=float(((m.service_s / m.service_s_f) - 1).abs().max())))
    df = pd.DataFrame(rows)
    C.wcsv(df, OUT / "identity_check.csv")
    print(df.to_string())


# ============================================================================================ analysis
SETS = ["A_i", "A_ii", "B_i", "B_ii"]
ONLINE = H4.ONLINE


def cell_table(raw: pd.DataFrame) -> pd.DataFrame:
    """per (workload, N, cold) at default budget: seed means of Ours/B1/B0/best-online and per-seed ratio extremes."""
    d = raw[(raw.budget == "default") & (raw.status == "ok") & (raw.boot == -1) & (raw.prior_kind == "A")]
    rows = []
    for (w, n, cold), g in d.groupby(["workload", "N", "cold"]):
        p = g.pivot_table(index="seed", columns="variant", values="service_s", aggfunc="first")
        onl = [c for c in ONLINE if c in p and p[c].notna().all()]
        means = {c: float(p[c].mean()) for c in onl}
        best = min(means, key=means.get)
        r = dict(workload=w, N=int(n), cold=cold, best_online=best,
                 ours=float(p.Ours.mean()), B1=float(p.B1.mean()), B0=float(p.B0.mean()), best=float(p[best].mean()),
                 ours_over_B1=float(p.Ours.mean() / p.B1.mean()), ours_over_B0=float(p.Ours.mean() / p.B0.mean()),
                 ours_over_best=float(p.Ours.mean() / p[best].mean()),
                 ours_over_B1_seed_max=float((p.Ours / p.B1).max()), ours_over_best_seed_max=float((p.Ours / p[best]).max()),
                 B1_over_ours_meanratio=float((p.B1 / p.Ours).mean()), B1_over_ours_seed_max=float((p.B1 / p.Ours).max()),
                 B1_over_ours_of_means=float(p.B1.mean() / p.Ours.mean()))
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["workload", "N", "cold"]).reset_index(drop=True)


def side(rel):
    """Ours vs B1: -1 Ours faster by >=2%, +1 B1 faster by >=2%, 0 tied (<2%)."""
    return np.where(rel <= -H4.TOL + 1e-12, -1, np.where(rel >= H4.TOL - 1e-12, 1, 0))


def analyze():
    frozen = pd.read_csv(C.FROZEN_RAW)
    fz_cell = cell_table(frozen)
    fz_v = H4.verdict(frozen, None)
    res = dict(frozen=dict(primary=fz_v["primary"]))
    lines = []
    summ_rows = []
    cell_all = []
    for s in SETS:
        raw = pd.read_csv(OUT / f"raw/calib_{s}_raw.csv")
        assert (raw.status == "ok").all(), (s, raw.status.value_counts().to_dict())
        v = H4.verdict(raw, None)
        C.wr(OUT / f"h4_verdict_calib_{s}.json", json.dumps(v, indent=1, ensure_ascii=False))
        cc = cell_table(raw)
        mrg = cc.merge(fz_cell, on=["workload", "N", "cold"], suffixes=("", "_fz"))
        for k in ("ours_over_B1", "ours_over_B0", "ours_over_best"):
            mrg[f"d_{k}"] = mrg[k] - mrg[f"{k}_fz"]
        mrg["rel_B1_fz"] = mrg.ours_over_B1_fz - 1
        mrg["rel_B1"] = mrg.ours_over_B1 - 1
        mrg["side_fz"], mrg["side_cal"] = side(mrg.rel_B1_fz.to_numpy()), side(mrg.rel_B1.to_numpy())
        mrg["flip"] = (mrg.side_fz * mrg.side_cal == -1)               # one side faster by >=2% -> the other side faster by >=2%
        mrg["changed_band"] = mrg.side_fz != mrg.side_cal
        mrg.insert(0, "set", s)
        cell_all.append(mrg)
        # H4 comparison
        hv = {}
        for N in ("10000", "1000"):
            a, b0, b1 = v["per_N"][N]["H4a_W2"], v["per_N"][N]["H4b_W3"], v["per_N"][N]["H4c_W1"]
            fa, fb0, fb1 = fz_v["per_N"][N]["H4a_W2"], fz_v["per_N"][N]["H4b_W3"], fz_v["per_N"][N]["H4c_W1"]
            hv[N] = dict(H4a=a["verdict"], H4a_fz=fa["verdict"], H4a_best=a["best_online"], H4a_best_fz=fa["best_online"], H4a_mean_rel=a["mean_rel_diff"], H4a_mean_rel_fz=fa["mean_rel_diff"],
                         H4a_seeds_lower=a["seeds_ours_lower_by_2pct"], H4a_seeds_lower_fz=fa["seeds_ours_lower_by_2pct"],
                         H4b=b0["verdict"], H4b_fz=fb0["verdict"], H4b_mean_ratio=b0["mean_ratio"], H4b_mean_ratio_fz=fb0["mean_ratio"], H4b_seeds=b0["seeds_within_1p02"], H4b_seeds_fz=fb0["seeds_within_1p02"],
                         H4c=b1["verdict"], H4c_fz=fb1["verdict"], H4c_mean_ratio=b1["mean_ratio"], H4c_mean_ratio_fz=fb1["mean_ratio"], H4c_seeds=b1["seeds_within_1p02"], H4c_seeds_fz=fb1["seeds_within_1p02"])
            hv[N]["unchanged"] = bool(hv[N]["H4a"] == hv[N]["H4a_fz"] and hv[N]["H4b"] == hv[N]["H4b_fz"] and hv[N]["H4c"] == hv[N]["H4c_fz"])
        # worst cases
        wst = dict(ours_over_B1_max_cellmean=mrg.loc[mrg.ours_over_B1.idxmax(), ["workload", "N", "cold", "ours_over_B1"]].tolist(),
                   ours_over_B1_max_seed=mrg.loc[mrg.ours_over_B1_seed_max.idxmax(), ["workload", "N", "cold", "ours_over_B1_seed_max"]].tolist(),
                   ours_over_best_max_cellmean=mrg.loc[mrg.ours_over_best.idxmax(), ["workload", "N", "cold", "ours_over_best"]].tolist(),
                   ours_over_best_max_seed=mrg.loc[mrg.ours_over_best_seed_max.idxmax(), ["workload", "N", "cold", "ours_over_best_seed_max"]].tolist(),
                   B1_over_ours_meanratio_max=mrg.loc[mrg.B1_over_ours_meanratio.idxmax(), ["workload", "N", "cold", "B1_over_ours_meanratio"]].tolist(),
                   B1_over_ours_seed_max=mrg.loc[mrg.B1_over_ours_seed_max.idxmax(), ["workload", "N", "cold", "B1_over_ours_seed_max"]].tolist())
        wfz = dict(ours_over_B1_max_cellmean=fz_cell.loc[fz_cell.ours_over_B1.idxmax(), ["workload", "N", "cold", "ours_over_B1"]].tolist(),
                   ours_over_B1_max_seed=fz_cell.loc[fz_cell.ours_over_B1_seed_max.idxmax(), ["workload", "N", "cold", "ours_over_B1_seed_max"]].tolist(),
                   ours_over_best_max_cellmean=fz_cell.loc[fz_cell.ours_over_best.idxmax(), ["workload", "N", "cold", "ours_over_best"]].tolist(),
                   ours_over_best_max_seed=fz_cell.loc[fz_cell.ours_over_best_seed_max.idxmax(), ["workload", "N", "cold", "ours_over_best_seed_max"]].tolist(),
                   B1_over_ours_meanratio_max=fz_cell.loc[fz_cell.B1_over_ours_meanratio.idxmax(), ["workload", "N", "cold", "B1_over_ours_meanratio"]].tolist(),
                   B1_over_ours_seed_max=fz_cell.loc[fz_cell.B1_over_ours_seed_max.idxmax(), ["workload", "N", "cold", "B1_over_ours_seed_max"]].tolist())
        # empty cache N=1e4: Ours faster than B1 by (1 - Ours/B1)
        em = mrg[(mrg.cold == "empty") & (mrg.N == 10000)].set_index("workload")
        emp = {w: dict(cal=float(1 - em.loc[w, "ours_over_B1"]), frozen=float(1 - em.loc[w, "ours_over_B1_fz"])) for w in em.index}
        emp_rng = [min(v["cal"] for v in emp.values()), max(v["cal"] for v in emp.values())]
        emp_rng_fz = [min(v["frozen"] for v in emp.values()), max(v["frozen"] for v in emp.values())]
        nflip = int(mrg.flip.sum())
        nband = int(mrg.changed_band.sum())
        robust = bool(all(hv[N]["unchanged"] for N in hv) and nflip == 0)
        res[s] = dict(h4=hv, worst=wst, worst_frozen=wfz, empty_N1e4_ours_faster_than_B1=emp, empty_range_cal=emp_rng, empty_range_frozen=emp_rng_fz,
                      n_cells=int(len(mrg)), n_flip_cells=nflip, n_cells_changed_2pct_band=nband,
                      flip_cells=mrg[mrg.flip][["workload", "N", "cold", "ours_over_B1_fz", "ours_over_B1"]].values.tolist(),
                      band_change_cells=mrg[mrg.changed_band][["workload", "N", "cold", "side_fz", "side_cal", "ours_over_B1_fz", "ours_over_B1"]].values.tolist(),
                      max_abs_delta_ours_over_B1=float(mrg.d_ours_over_B1.abs().max()), max_abs_delta_ours_over_B0=float(mrg.d_ours_over_B0.abs().max()),
                      max_abs_delta_ours_over_best=float(mrg.d_ours_over_best.abs().max()),
                      by_cold_mean_delta_ours_over_B1={c: float(g.d_ours_over_B1.mean()) for c, g in mrg.groupby("cold")},
                      robust=robust)
        summ_rows.append(dict(set=s, robust=robust, H4_unchanged_N1e4=hv["10000"]["unchanged"], H4_unchanged_N1e3=hv["1000"]["unchanged"], n_flip=nflip, n_band_change=nband))
    allc = pd.concat(cell_all, ignore_index=True)
    C.wcsv(allc, OUT / "cell_diff_vs_frozen.csv")
    C.wcsv(fz_cell, OUT / "cell_table_frozen.csv")
    C.wcsv(pd.DataFrame(summ_rows), OUT / "robustness_summary.csv")
    C.wr(OUT / "analysis_summary.json", json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(summ_rows, indent=1))
    for s in SETS:
        r = res[s]
        print("==", s, "robust", r["robust"], "flips", r["n_flip_cells"], "band changes", r["n_cells_changed_2pct_band"])
        for N in ("10000", "1000"):
            h = r["h4"][N]
            print(f"  N={N}: H4a {h['H4a_fz']}->{h['H4a']} (best {h['H4a_best_fz']}->{h['H4a_best']}, rel {h['H4a_mean_rel_fz']:+.3%}->{h['H4a_mean_rel']:+.3%}, "
                  f"seeds {h['H4a_seeds_lower_fz']}->{h['H4a_seeds_lower']}) | H4b {h['H4b_fz']}->{h['H4b']} ({h['H4b_mean_ratio_fz']:.4f}->{h['H4b_mean_ratio']:.4f}) | "
                  f"H4c {h['H4c_fz']}->{h['H4c']} ({h['H4c_mean_ratio_fz']:.4f}->{h['H4c_mean_ratio']:.4f})")
        print("  empty N1e4 Ours faster than B1:", {w: f"{v['frozen']:.1%}->{v['cal']:.1%}" for w, v in r["empty_N1e4_ours_faster_than_B1"].items()})
        print("  worst:", r["worst"])
    print("  worst frozen:", res["A_i"]["worst_frozen"])


# ============================================================================================ 3.4 self-check
def selfcheck():
    P = C.S.base_params(C.S.load_params())
    pj = C.S.load_params()
    prior = S.prior_A()
    inp = P34.InputModel(P34.INPUT_PATH)
    budget = float(json.loads(P34.BUDGETS_PATH.read_text())["tiers"]["default"]["mb"])
    ADD.install_simulator_hook()
    ctf = plain_frozen_table()
    add_ov = ADD.apply_addendum(pd.read_csv(S.CT_PATH), ADD.read_addendum(S.ADDENDUM_PATH))[1]
    plan = P34.plan("B1")
    frozen_pred = pd.read_csv(RV / "pred_S3_4.csv")
    out_tabs = {}
    # the calibrated copies are built from impute_empty(S2): for the warm (disk) predictions impute has no effect; for the W5 Ours empty report it
    # does (the frozen 3.4 prediction ran on the un-imputed table; reported separately and flagged)
    for name in ("A", "B"):
        cts = C.load_cal_table(name)
        for persp in ("i", "ii"):
            ctp = ctf if persp == "i" else cts
            rows = []
            for (w, pol, mode, role) in plan:
                stem = P34.STREAMS[w]
                reqs = SIM.read_trace(P34.RV / f"{stem}.jsonl")
                bud = budget if pol in P34.BUDGETED else None
                cv = CostView(ctp, mode, rebuild_override=getattr(ctp, "rebuild_override", None))
                polobj = make_policy(P34.SIM_NAME.get(pol, pol), cost=cv, params=dict(P), prior=prior)
                res = SIM.simulate(reqs, polobj, cts, SIM.SimOptions(cache_mode=mode, mem_budget_mb=bud, overhead_aware=True))
                inp.n_extrap = 0
                tot_in = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
                sm = res.summary
                ev = res.frame()
                ev = ev[ev.idx >= 0]
                rows.append(dict(stream=stem, policy=pol, cache_mode=mode, role=role, mem_budget_mb=bud, n_requests=len(reqs),
                                 pred_excl_input_s=float(sm["service_s"]), pred_input_s=tot_in, pred_total_s=float(sm["service_s"]) + tot_in,
                                 pred_peak_reserved_est_mb=sm.get("peak_reserved_est_mb"), n_events_with_fallback=sm.get("n_events_with_fallback"),
                                 event_counts=json.dumps(ev.event.value_counts().sort_index().to_dict(), sort_keys=True)))
            pdf = pd.DataFrame(rows)
            pp = OUT / f"selfcheck34/pred_calib_{name}_{persp}.csv"
            C.wcsv(pdf, pp)
            tag = f"_{name}_{persp}"
            CMP.run(pp, RV / "raw", RV / "raw_empty", OUT / "selfcheck34", tag)
            out_tabs[(name, persp)] = pd.read_csv(OUT / f"selfcheck34/error_table{tag}.csv")
    base = pd.read_csv(RV / "error_table.csv")
    base = base[["stream", "policy", "cache_mode", "judged", "measured_median_s", "sim_s", "error"]].rename(columns={"sim_s": "sim_frozen_s", "error": "err_frozen"})
    cmp_ = base.copy()
    for (name, persp), t in out_tabs.items():
        x = t[["stream", "policy", "cache_mode", "sim_s", "error"]].rename(columns={"sim_s": f"sim_{name}{persp}_s", "error": f"err_{name}{persp}"})
        cmp_ = cmp_.merge(x, on=["stream", "policy", "cache_mode"])
    C.wcsv(cmp_, OUT / "selfcheck34/error_table_compare.csv")
    pd.set_option("display.width", 250)
    show = ["stream", "policy", "cache_mode", "judged", "measured_median_s", "err_frozen"] + [c for c in cmp_.columns if c.startswith("err_") and c != "err_frozen"]
    print(cmp_[show].round(4).to_string())
    j = cmp_[cmp_.judged == 1]
    print("max judged error:", {c: float(j[c].max()) for c in cmp_.columns if c.startswith("err_")})
    print("mean abs signed error (sim-meas)/meas:", {c.replace("sim_", "").replace("_s", ""): float(((j[c] - j.measured_median_s) / j.measured_median_s).mean()) for c in cmp_.columns if c.startswith("sim_")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["check0a", "check0a_comp", "check0b", "identity", "analyze", "selfcheck"])
    a = ap.parse_args()
    {"check0a": check0a, "check0a_comp": check0a_comp, "check0b": check0b, "identity": identity, "analyze": analyze, "selfcheck": selfcheck}[a.stage]()


if __name__ == "__main__":
    main()
