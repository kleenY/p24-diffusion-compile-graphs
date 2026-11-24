"""Aggregate raw per-process CSVs. The process is the statistical unit: per process take the median of
its steady-state requests, then report the median across processes with a percentile-bootstrap interval over processes. With n=5 this
interval equals [min, max] of the 5 per-process medians (an order-statistic interval with ~94% coverage, not 95%);
report it as such.

Usage: python -m p24.stats <bench_dir> [--out summary.csv] [--windows 4-23,24-43] [--transient]

Review 2026-11-15 (M4): one-time-cost medians are split by cold tier (`<col>_median_cold_empty`,
`<col>_median_cold_diskcache`; cache-independent arms use all processes in both); `--windows lo-hi,...` adds per-process
medians of the steady wall time over request-index windows (request index i = `req_idx` column; i=0 is the first request
that is part of first_req_s; steady row r of a process run with --warm w has i = w + r + 1; CSVs without the column are read
as w=3, i.e. i = rep + 4), reported per cell for all processes and per cold tier; `--transient` writes
`<out>_transient.csv` with the post-compile transient excess (protocol sec 5): per cell, sum over i in 1..43 of
(wall of a cold_empty process at i - median over the cell's cold_diskcache processes at i), process-bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd


CACHE_INDEPENDENT = {"E0", "E1", "G1", "G2"}  # arms whose one-time cost does not depend on the compile cache


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def load(bench_dir: pathlib.Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(bench_dir.glob("*.csv")) if not f.name.startswith("summary")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


NEW_PROC_COLS = ["resident_mb", "weights_alloc_mb", "steady_peak_alloc_mb", "steady_peak_reserved_mb", "reserved_mb",
                 "compile_s", "compile_and_warmup_s", "capture_s", "capture_pool_mb", "graph_mem_mb", "static_io_mb",
                 "first_call_te_s", "first_call_unet_s", "first_call_vae_s", "first_call_whole_s",
                 "dyn_recompiles", "dyn_unique_graphs", "fx_cache_hit", "fx_cache_miss", "aot_cache_hit",
                 "aot_cache_miss", "recompile_limit_hit", "tier_violation", "pcore_pinned"]
COMP_COLS = ["te_ms", "denoise_ms", "vae_ms"]
ONE_TIME_COLS = ["compile_s", "compile_and_warmup_s", "capture_s", "first_call_te_s", "first_call_unet_s",
                 "first_call_vae_s", "first_call_whole_s"]  # excluded for tier_violation processes (protocol sec 2)


def excluded_processes(df: pd.DataFrame, include_contended: bool = False,
                       include_cpu_contended: bool = False) -> pd.DataFrame:
    """Processes dropped from the reported numbers, with the reason (protocol sec 2 / sec 7: listed, never re-included).
    tier_violation is NOT an exclusion here (steady state is kept); it is only dropped from one-time-cost statistics."""
    if df.empty:
        return pd.DataFrame()
    steady = df[df.get("row_kind", pd.Series("steady", index=df.index)).fillna("steady") != "comp"]
    keys = ["arm", "H", "W", "B", "cfg", "proc_id", "cold_tier"]
    pp = steady.groupby(keys).agg(
        status=("status", "first"), contended=("contended", "max"),
        load_max=("load1_start", "max") if "load1_start" in steady else ("status", "first"),
        load_end=("load1_end", "max") if "load1_end" in steady else ("status", "first"),
        pcore=("pcore_pinned", "min") if "pcore_pinned" in steady else ("status", "first"),
        rlh=("recompile_limit_hit", "max") if "recompile_limit_hit" in steady else ("status", "first")).reset_index()
    rows = []
    for _, r in pp.iterrows():
        why = []
        if r.status != "ok":
            why.append("status!=ok")
        if not include_contended and r.contended != 0:
            why.append("contended")
        if not include_cpu_contended and "load1_start" in steady and max(r.load_max, r.load_end) > _cpu_max():
            why.append("cpu_load")
        if "pcore_pinned" in steady and r.pcore == 0:
            why.append("not_pcore_pinned")
        if "recompile_limit_hit" in steady and r.rlh == 1:
            why.append("recompile_limit_hit")
        if why:
            rows.append({**{k: r[k] for k in keys}, "reason": "+".join(why)})
    return pd.DataFrame(rows)


def _cpu_max():
    from .bench import CPU_LOAD_MAX
    return CPU_LOAD_MAX


DEFAULT_WARM = 3  # S1 / 2.0 default: CSVs without a req_idx column were run with --warm 3


def req_index(df: pd.DataFrame) -> pd.Series:
    """Request index i of each steady row (see module docstring); NaN for component rows / placeholder rows."""
    if "req_idx" in df and df["req_idx"].notna().any():
        ri = df["req_idx"].astype(float)
    else:
        ri = df["rep"].astype(float) + DEFAULT_WARM + 1
    ri = ri.where(df["rep"] >= 0)
    if "row_kind" in df:
        ri = ri.where(df["row_kind"].fillna("steady") != "comp")
    return ri


def parse_windows(spec) -> list:
    """'4-23,24-43' -> [(4, 23), (24, 43)]"""
    if not spec:
        return []
    if not isinstance(spec, str):
        return [tuple(w) for w in spec]
    return [tuple(int(v) for v in w.split("-")) for w in spec.split(",") if w.strip()]


def _filtered_steady(df: pd.DataFrame, include_contended: bool, include_cpu_contended: bool) -> pd.DataFrame:
    """Steady rows of the processes that survive the protocol sec 2 exclusions (same filters as summarize)."""
    ok = df[df.status == "ok"]
    if not include_contended:
        ok = ok[ok.contended == 0]
    if not include_cpu_contended and "load1_start" in ok:
        from .bench import CPU_LOAD_MAX
        busy = ok[["load1_start", "load1_end"]].max(axis=1) > CPU_LOAD_MAX
        ok = ok[~busy.fillna(False)]
    if "pcore_pinned" in ok:
        ok = ok[ok.pcore_pinned.fillna(1) == 1]
    if "recompile_limit_hit" in ok:
        ok = ok[ok.recompile_limit_hit.fillna(0) == 0]
    if "row_kind" in ok:
        ok = ok[ok.row_kind.fillna("steady") != "comp"]
    return ok


def transient_excess(df: pd.DataFrame, i_range=(1, 43), cold="cold_empty", ref="cold_diskcache",
                     include_contended: bool = False, include_cpu_contended: bool = False,
                     n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Post-compile transient excess (protocol sec 5 (b)), in seconds, per cell (arm, H, W, B, cfg):
    excess_p = sum_{i in range} (wall_p,i - median over the cell's `ref`-tier processes of wall_ref,i) for every `cold` process p;
    cell value = median over the cold processes, 95% CI by a process bootstrap that resamples the cold processes AND the
    reference processes (`n_boot` resamples, seed `seed`). A process missing any request index in the range is dropped."""
    ok = _filtered_steady(df, include_contended, include_cpu_contended)
    if ok.empty:
        return pd.DataFrame()
    ok = ok.assign(i=req_index(ok)).dropna(subset=["i"])
    lo, hi = i_range
    idx = list(range(lo, hi + 1))
    rows = []
    for cell, g in ok.groupby(["arm", "H", "W", "B", "cfg"]):
        def mat(tier):
            ps = []
            for _, gp in g[g.cold_tier == tier].groupby("proc_id"):
                w = gp.drop_duplicates("i").set_index("i").wall_ms.reindex(idx)
                if w.notna().all():
                    ps.append(w.to_numpy(float))
            return np.array(ps).reshape(len(ps), len(idx))
        C, D = mat(cold), mat(ref)
        row = dict(zip(["arm", "H", "W", "B", "cfg"], cell), n_cold=len(C), n_ref=len(D), i_lo=lo, i_hi=hi)
        if len(C) == 0 or len(D) == 0:
            row.update(excess_s_median=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))
            rows.append(row)
            continue
        refmed = np.median(D, axis=0)
        ex = (C - refmed).sum(axis=1) / 1e3
        row["excess_s_median"] = float(np.median(ex))
        row["excess_s_mean"] = float(np.mean(ex))
        if len(C) >= 2 and len(D) >= 2:
            rng = np.random.default_rng(seed)
            bs = np.empty(n_boot)
            for b in range(n_boot):
                cb = C[rng.integers(0, len(C), len(C))]
                db = D[rng.integers(0, len(D), len(D))]
                bs[b] = np.median((cb - np.median(db, axis=0)).sum(axis=1)) / 1e3
            row["ci_lo"], row["ci_hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        else:
            row["ci_lo"] = row["ci_hi"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, include_contended: bool = False, include_cpu_contended: bool = False,
              windows=None) -> pd.DataFrame:
    """include_cpu_contended: keep processes measured while the host CPU was busy (column absent in
    pre-2026-11-15 runs -> treated as unknown and kept; those runs are reported as CPU-contended).
    S2 (2026-11-15): also excludes non-P-core processes (pcore_pinned==0) and recompile_limit_hit processes when those
    columns exist; component rows (row_kind=="comp") are kept apart from the steady rows; tier_violation processes are
    dropped only from the one-time-cost statistics.
    M4 (2026-11-15): one-time-cost medians are split by cold tier; `windows` (e.g. [(4, 23), (24, 43)] or "4-23,24-43")
    adds the steady wall time over request-index windows (per-process median, then median across processes)."""
    wins = parse_windows(windows)
    ok = df[df.status == "ok"]
    if not include_contended:
        ok = ok[ok.contended == 0]
    load_unknown = ok["load1_start"].isna() if "load1_start" in ok else pd.Series(True, index=ok.index)
    ok = ok.assign(load_unknown=load_unknown.astype(int))
    if not include_cpu_contended and "load1_start" in ok:
        # recomputed from the raw load columns so the threshold lives in one place (bench.CPU_LOAD_MAX)
        from .bench import CPU_LOAD_MAX
        busy = ok[["load1_start", "load1_end"]].max(axis=1) > CPU_LOAD_MAX
        ok = ok[~busy.fillna(False)]
    if "pcore_pinned" in ok:  # protocol sec 2 item 2
        ok = ok[ok.pcore_pinned.fillna(1) == 1]
    if "recompile_limit_hit" in ok:  # protocol sec 2 item 4
        ok = ok[ok.recompile_limit_hit.fillna(0) == 0]
    comp = ok[ok.row_kind == "comp"] if "row_kind" in ok else ok.iloc[0:0]
    if "row_kind" in ok:
        ok = ok[ok.row_kind != "comp"]
    gk = ["arm", "H", "W", "B", "cfg", "proc_id", "cold_tier"]
    new_cols = [c for c in NEW_PROC_COLS if c in ok]
    aggs = dict(wall_med=("wall_ms", "median"), gpu_med=("gpu_ms", "median"),
                wall_p95=("wall_ms", lambda s: s.quantile(.95)),
                first_req_s=("first_req_s", "first"), prep_json=("prep_json", "first"),
                peak_alloc_mb=("peak_alloc_mb", "first"), load_unknown=("load_unknown", "max"))
    aggs.update({c: (c, "first") for c in new_cols})
    per_proc = ok.groupby(gk).agg(**aggs).reset_index()
    if wins:
        okr = ok.assign(i=req_index(ok))
        for lo, hi in wins:
            sub = okr[(okr.i >= lo) & (okr.i <= hi)]
            wm = sub.groupby(gk).wall_ms.median().rename(f"wall_i{lo}_{hi}").reset_index()
            per_proc = per_proc.merge(wm, on=gk, how="left")
    if len(comp):
        cc = [c for c in COMP_COLS if c in comp]
        cp = comp.groupby(gk)[cc].median().reset_index().rename(columns={c: c + "_med" for c in cc})
        per_proc = per_proc.merge(cp, on=gk, how="left")
    out = []
    for k, g in per_proc.groupby(["arm", "H", "W", "B", "cfg"]):
        lo, hi = boot_ci(g.wall_med)
        # one-time-cost statistics: tier_violation processes excluded (steady state above keeps them)
        gt = g[g.tier_violation.fillna(0) == 0] if "tier_violation" in g else g
        prep0 = gt[gt.cold_tier == "cold_empty"]
        prep_d = gt[gt.cold_tier == "cold_diskcache"]

        if k[0] in CACHE_INDEPENDENT:
            # no compile cache -> "cold tier" is meaningless; use all processes (review 2026-11-15, item 1)
            prep0 = prep_d = g
        row = dict(zip(["arm", "H", "W", "B", "cfg"], k), n_proc=len(g), n_proc_load_unknown=int(g.load_unknown.sum()),
                   steady_wall_ms=float(np.median(g.wall_med)), ci_lo=lo, ci_hi=hi,
                   steady_gpu_ms=float(np.median(g.gpu_med)), wall_p95_ms=float(np.median(g.wall_p95)),
                   first_req_s_cold_empty=float(prep0.first_req_s.median()) if len(prep0) else float("nan"),
                   first_req_s_diskcache=float(prep_d.first_req_s.median()) if len(prep_d) else float("nan"),
                   peak_alloc_mb=float(g.peak_alloc_mb.max()))
        # ---- S2 columns (only when the raw CSVs have them) ----
        for c in ("te_ms_med", "denoise_ms_med", "vae_ms_med"):
            if c in g:
                row[c.replace("_med", "_median")] = float(g[c].median())
        for c in ("resident_mb", "weights_alloc_mb", "steady_peak_alloc_mb", "steady_peak_reserved_mb",
                  "capture_pool_mb", "graph_mem_mb", "static_io_mb"):
            if c in g:
                row[c] = float(g[c].max())  # memory: worst process
        for c in ONE_TIME_COLS:  # split by cold tier (review M4); cache-independent arms: all processes in both
            if c in gt:
                row[c + "_median_cold_empty"] = float(prep0[c].median()) if len(prep0) else float("nan")
                row[c + "_median_cold_diskcache"] = float(prep_d[c].median()) if len(prep_d) else float("nan")
        for lo, hi in wins:  # request-index windows of the steady wall time
            wc = f"wall_i{lo}_{hi}"
            row[f"steady_wall_ms_i{lo}_{hi}"] = float(g[wc].median()) if g[wc].notna().any() else float("nan")
            for tname, sub in (("cold_empty", g[g.cold_tier == "cold_empty"] if k[0] not in CACHE_INDEPENDENT else g),
                               ("cold_diskcache", g[g.cold_tier == "cold_diskcache"] if k[0] not in CACHE_INDEPENDENT else g)):
                row[f"steady_wall_ms_i{lo}_{hi}_{tname}"] = (float(sub[wc].median()) if len(sub) and sub[wc].notna().any()
                                                              else float("nan"))
        for c in ("dyn_recompiles", "dyn_unique_graphs", "fx_cache_hit", "fx_cache_miss", "aot_cache_hit",
                  "aot_cache_miss"):
            if c in g:
                row[c + "_max"] = float(g[c].max())
        if "tier_violation" in g:
            row["n_tier_violation"] = int((g.tier_violation.fillna(0) == 1).sum())
        out.append(row)
    if not out:  # every process excluded (e.g. contended dev-mode runs)
        return pd.DataFrame(columns=["arm", "H", "W", "B", "cfg", "n_proc"])
    s = pd.DataFrame(out)
    # break-even vs eager E0 of the same key and same cold tier. Serving n requests costs
    # F + (n-1)*steady for each arm (F = first-request latency incl. compile/capture), so the arm wins once
    # n - 1 > (F_arm - F_E0) / (steady_E0 - steady_arm).
    e0 = s[s.arm == "E0"].set_index(["H", "W", "B", "cfg"])
    def be(row, col):
        try:
            base = e0.loc[(row.H, row.W, row.B, row.cfg)]
        except KeyError:
            return float("nan")
        if row.arm == "E0":
            return float("nan")
        saving_s = (base.steady_wall_ms - row.steady_wall_ms) / 1e3
        extra_s = row[col] - base[col]
        if saving_s <= 0:
            return float("inf")
        return 1 + max(extra_s, 0.0) / saving_s

    def extra(row, col):
        try:
            return float(row[col] - e0.loc[(row.H, row.W, row.B, row.cfg)][col])
        except KeyError:
            return float("nan")
    for tier in ("cold_empty", "diskcache"):
        col = f"first_req_s_{tier}"
        s[f"extra_s_{tier}"] = s.apply(lambda r: extra(r, col), axis=1)  # signed; <0 -> truncated to 0 in n
        s[f"breakeven_n_{tier}"] = s.apply(lambda r: be(r, col), axis=1)
    return s.sort_values(["H", "W", "B", "cfg", "steady_wall_ms"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-contended", action="store_true")
    ap.add_argument("--include-cpu-contended", action="store_true")
    ap.add_argument("--windows", default="", help="request-index windows, e.g. 4-23,24-43 (adds steady_wall_ms_i<lo>_<hi>*)")
    ap.add_argument("--transient", action="store_true", help="also write <out>_transient.csv (sec 5 transient excess)")
    a = ap.parse_args()
    d = pathlib.Path(a.bench_dir)
    raw = load(d)
    s = summarize(raw, a.include_contended, a.include_cpu_contended, windows=a.windows)
    outp = pathlib.Path(a.out) if a.out else d / "summary.csv"
    s.to_csv(outp, index=False)
    if a.transient:
        te = transient_excess(raw, include_contended=a.include_contended, include_cpu_contended=a.include_cpu_contended)
        te.to_csv(outp.with_name(outp.stem + "_transient.csv"), index=False)
        print(f"transient excess: {len(te)} cells -> {outp.stem}_transient.csv")
    ex = excluded_processes(raw, a.include_contended, a.include_cpu_contended)
    if len(ex):
        ex.to_csv(outp.with_name(outp.stem + "_excluded.csv"), index=False)
        print(f"excluded processes: {len(ex)} -> {outp.stem}_excluded.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s.round(3).to_string(index=False))
