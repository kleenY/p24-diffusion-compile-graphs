"""S4-X X3 step 1: choose the bounded set K of shapes that get an offline AOT package (CPU only, no torch, no trace is read).
Preregistration goals/S4X_X3_finite_bucket_AOT_prereg_v1.md sec 2 (written before any number was computed; the rule below is that text, nothing else):
  q_s  expected request share of shape s = scripts-frozen A-half prior (p24.sim.prior.prior_A, same as S3 / S4-P)
  d_s  per-request saving (s)      = (steady E0 - steady AOT)/1e3                 from cost_table_v2_aot.csv (world table of block C, +addendum_rebuild_v2.csv)
  e_s  extra one-time cost (s)     = one-time(AOT, tier inproc_diskcache) - one-time(E0, tier inproc_diskcache)   (the disk tier for every shape but the process' first one)
  b_s  offline build wall-clock    = process_total_s of the shape's build process (results/S4P/p5c/build_summary.csv; the three component rows of a shape carry the same
                                     per-process value -> counted ONCE per shape; 32 shapes, sum = full-build cost)
  g_s(N) = N*q_s*d_s - e_s  (N = 1000);  candidates = shapes with an AOT row, NOT in the fallback list (aot_fallback_shapes.csv), g_s(N) > 0;
  order by g_s desc (ties: shape key), take the prefix while cumulative sum b_s <= budget = 0.5 * sum b_s over ALL 32 packages (stop at the first shape that would exceed).
Interpretation choices where the card text allows two readings (recorded in K.json "interpretation"; the sensitivity block shows whether K would differ):
  * "until the sum of b reaches the budget": strict prefix with cumulative <= budget (never above 50 %).  Sensitivity: the set that also takes the shape crossing the line.
  * e_s: pure one-time extra (resid difference).  The literal "first request(AOT) - first request(E0)" would add (steady_AOT - steady_E0) = -d_s, i.e. count the first request's saving twice.  Sensitivity shown.
  * tier of e_s: inproc_diskcache.  Sensitivity: cold_diskcache.
Outputs (written once, refuses to overwrite; --freeze = 0444): results/S4X/x3/K_selection.csv (per shape), K.json (selected set + inputs' SHA16 + sensitivity).
Usage (from code/):  python -B -m scripts.select_buckets_S4X_X3 --freeze
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import pandas as pd

from scripts import aot_tables_S4P_P5c as T
from scripts import s4p_p3_common_v2 as C

ROOT = C.ROOT
OUT_DIR = ROOT / "results/S4X/x3"
BUILD_SUMMARY = ROOT / "results/S4P/p5c/build_summary.csv"
ADDENDUM = ROOT / "results/S4P/p3/cost_v2/addendum_rebuild_v2.csv"
PREREG = ROOT / "goals/S4X_X3_finite_bucket_AOT_prereg_v1.md"
N_DEFAULT = 1000
BUDGET_FRAC = 0.5
TIER_MAIN, TIER_ALT = "inproc_diskcache", "cold_diskcache"
sha16 = C.sha16


def key_of(sh) -> str:
    return f"{sh.H}x{sh.W}x{sh.B}x{sh.cfg}"


def build_wall(path=BUILD_SUMMARY) -> dict:
    """{HxWxBxcfg: process_total_s}; asserts the three component rows of a shape agree and all 96 rows are ok."""
    b = pd.read_csv(path)
    assert (b.status == "ok").all(), "build_summary has non-ok rows"
    g = b.groupby("key").process_total_s.agg(["nunique", "first", "count"])
    assert (g["nunique"] == 1).all() and (g["count"] == 3).all(), "per-shape process_total_s must be identical over te/unet/vae"
    return {k: float(v) for k, v in g["first"].items()}


def per_shape_table(N: int, table_csv, addendum_csv, fallback_csv) -> pd.DataFrame:
    ct = C.load_v2(str(table_csv), str(addendum_csv))
    fb = T.read_fallback(fallback_csv)
    cta = T.swap_c1(ct, T.aot_provider(ct), ())                       # AOT prices on the C1 rows, no fallback swap (the fallback shapes keep their AOT price here; they are excluded by rule)
    prior = C.prior_A()
    bw = build_wall()
    rows = []
    for sh in sorted(T.aot_shapes(ct.df), key=lambda s: (s.H, s.W, s.B, s.cfg)):
        k = key_of(sh)
        e0, aot = float(ct.row("E0", sh)["steady_ms"]), float(ct.row("AOT", sh)["steady_ms"])
        d = (e0 - aot) / 1e3
        r = dict(shape_key=k, H=sh.H, W=sh.W, B=sh.B, cfg=sh.cfg, q_s=prior[sh], steady_e0_ms=e0, steady_aot_ms=aot, d_s=d, b_s=bw[k], b_s_costtable=float(ct.row("AOT", sh)["aot_build_wall_s"]), in_fallback=int(sh in fb))
        for tag, tier in (("", TIER_MAIN), ("_cold", TIER_ALT)):
            r[f"onetime_e0_s{tag}"] = ct.one_time("E0", sh, tier).total_s
            r[f"onetime_aot_s{tag}"] = cta.one_time("C1", sh, tier).total_s
            r[f"e_s{tag}"] = r[f"onetime_aot_s{tag}"] - r[f"onetime_e0_s{tag}"]
            r[f"g_s{tag}"] = N * prior[sh] * d - r[f"e_s{tag}"]
        r["g_s_literal_e"] = N * prior[sh] * d - (r["e_s"] - d)       # e = first-request(AOT) - first-request(E0) = resid diff + (steady_AOT - steady_E0)
        rows.append(r)
    return pd.DataFrame(rows)


def pick(df: pd.DataFrame, gcol: str, budget: float, allow_cross: bool = False, bcol: str = "b_s") -> list:
    """Shape keys of the prefix of candidates (not fallback, g>0) ordered by g desc while cumulative b <= budget (allow_cross: also the first shape that crosses)."""
    cand = df[(df.in_fallback == 0) & (df[gcol] > 0)].sort_values([gcol, "shape_key"], ascending=[False, True])
    out, cum = [], 0.0
    for r in cand.itertuples():
        if cum + getattr(r, bcol) <= budget:
            out.append(r.shape_key)
            cum += getattr(r, bcol)
        else:
            if allow_cross:
                out.append(r.shape_key)
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--budget-frac", type=float, default=BUDGET_FRAC)
    ap.add_argument("--table-csv", default=str(T.COST_TABLE_AOT))
    ap.add_argument("--addendum-csv", default=str(ADDENDUM))
    ap.add_argument("--fallback-csv", default=str(T.FALLBACK_CSV))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--freeze", action="store_true", help="chmod 0444 the outputs")
    a = ap.parse_args(argv)
    assert a.N == N_DEFAULT and a.budget_frac == BUDGET_FRAC, "preregistered N=1000 and 50 % budget only"
    assert T.is_ro(a.table_csv) and T.is_ro(a.fallback_csv), "cost table / fallback list must be 0444"
    out = pathlib.Path(a.out_dir)
    csv_p, js_p = out / "K_selection.csv", out / "K.json"
    for p in (csv_p, js_p):
        if p.exists():
            sys.exit(f"REFUSING to overwrite {p}")
    df = per_shape_table(a.N, a.table_csv, a.addendum_csv, a.fallback_csv)
    total_b = float(df.b_s.sum())
    assert len(df) == 32
    budget = a.budget_frac * total_b
    K = pick(df, "g_s", budget)
    cum = 0.0
    order = {}
    for i, k in enumerate(pick(df, "g_s", 1e18)):                      # full g-ranking of the candidates (rank 1 = best)
        order[k] = i + 1
    df["rank_g"] = df["shape_key"].map(order)
    df["selected"] = df["shape_key"].isin(K).astype(int)
    df["cum_b_s_selected"] = float("nan")
    for k in K:
        cum += float(df.loc[df["shape_key"] == k, "b_s"].iloc[0])
        df.loc[df["shape_key"] == k, "cum_b_s_selected"] = cum
    df["g_positive"] = (df.g_s > 0).astype(int)
    df = df.sort_values(["selected", "g_s"], ascending=[False, False]).reset_index(drop=True)
    sum_bK = float(df[df.selected == 1].b_s.sum())
    qK = float(df[df.selected == 1].q_s.sum())
    q_nonfb = float(df[df.in_fallback == 0].q_s.sum())
    K_cross = pick(df, "g_s", budget, allow_cross=True)
    K_bw = pick(df, "g_s", a.budget_frac * float(df.b_s_costtable.sum()), bcol="b_s_costtable")
    sens = dict(K_build_wall_from_cost_table=K_bw, K_tier_cold_diskcache=pick(df, "g_s_cold", budget), K_literal_first_request_e=pick(df, "g_s_literal_e", budget), K_include_crossing_shape=K_cross,
                same_as_main=dict(build_wall_from_cost_table=set(K_bw) == set(K), tier_cold_diskcache=set(pick(df, "g_s_cold", budget)) == set(K), literal_first_request_e=set(pick(df, "g_s_literal_e", budget)) == set(K),
                                  include_crossing_shape=set(K_cross) == set(K)),
                next_shape_after_K=[k for k in pick(df, "g_s", 1e18) if k not in K][:1],
                cum_b_if_crossing_shape_added=(sum_bK + float(df[df["shape_key"] == [k for k in pick(df, "g_s", 1e18) if k not in K][0]].b_s.iloc[0])) if len(K) < len(pick(df, "g_s", 1e18)) else None)
    meta = dict(prereg=str(PREREG.relative_to(ROOT)), prereg_sha16_at_selection=sha16(PREREG), time=time.strftime("%FT%T"), N=a.N, budget_frac=a.budget_frac,
                n_shapes_total=int(len(df)), sum_b_all_s=total_b, budget_s=budget, K=K, n_K=len(K), sum_b_K_s=sum_bK, sum_b_K_over_all=sum_bK / total_b,
                q_K=qK, q_K_over_nonfallback=qK / q_nonfb, q_all=float(df.q_s.sum()), n_candidates_g_positive_nonfallback=int(((df.in_fallback == 0) & (df.g_s > 0)).sum()),
                n_g_nonpositive=[r.shape_key for r in df.itertuples() if r.g_s <= 0], fallback_shapes=[r.shape_key for r in df.itertuples() if r.in_fallback],
                interpretation=dict(stop_rule="prefix of the g-ranked candidates while cumulative b <= budget (stop at the first that would exceed; never above 50 %)",
                                    e_s="one-time(AOT)-one-time(E0), tier inproc_diskcache; fc components of the AOT rows are 0", d_s="(steady E0 - steady AOT)/1e3 s from the same table",
                                    b_s="process_total_s per shape (one value shared by the te/unet/vae rows), 32 shapes",
                                    budget="0.5 * sum of b_s over all 32 packages (the full-build arm builds 32 incl. the fallback shape)"),
                sensitivity=sens, inputs=dict(cost_table=dict(path=str(a.table_csv), sha16=sha16(a.table_csv)), addendum=dict(path=str(a.addendum_csv), sha16=sha16(a.addendum_csv)),
                                              fallback=dict(path=str(a.fallback_csv), sha16=sha16(a.fallback_csv)), build_summary=dict(path=str(BUILD_SUMMARY), sha16=sha16(BUILD_SUMMARY)),
                                              prior_stats=dict(path=str(ROOT / "results/S2/workload/workload_stats.json"), sha16=sha16(ROOT / "results/S2/workload/workload_stats.json"))),
                script_sha16=sha16(pathlib.Path(__file__).resolve()), aot_tables_sha16=sha16(ROOT / "code/scripts/aot_tables_S4P_P5c.py"),
                s4p_p3_common_v2_sha16=sha16(ROOT / "code/scripts/s4p_p3_common_v2.py"),
                no_trace_read="no trace (dev / eval / real-machine) is read by this script: inputs are the A-half prior and the frozen cost / build tables only")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_p, index=False)
    js_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=list))
    if a.freeze:
        for p in (csv_p, js_p):
            os.chmod(p, 0o444)
    show = ["shape_key", "q_s", "d_s", "e_s", "b_s", "g_s", "in_fallback", "rank_g", "selected", "cum_b_s_selected"]
    print(df[show].round(4).to_string(index=False))
    print(f"|K|={len(K)}  sum_b(K)={sum_bK:.1f}s  sum_b(all)={total_b:.1f}s  ratio={sum_bK / total_b:.4f}  q(K)={qK:.4f} (of non-fallback {q_nonfb:.4f}: {qK / q_nonfb:.4f})")
    print("K =", K)
    print("sensitivity same_as_main:", sens["same_as_main"], "next:", sens["next_shape_after_K"], "cum_b_if_crossing:", sens["cum_b_if_crossing_shape_added"])
    print("K_selection.csv sha16", sha16(csv_p), "K.json sha16", sha16(js_p), "frozen" if a.freeze else "NOT frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
