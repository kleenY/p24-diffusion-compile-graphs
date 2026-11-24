"""Build the simulator cost table from raw bench CSVs (p24.bench / scripts.run_matrix output), in-process sequence runs
(scripts.bench_seq output) and optional p24.stats summaries. CPU only, no torch.

Outputs (in --out-dir): cost_table.csv (schema: p24/sim/costs.py), cost_table_procs.csv (one row per process x key: the
input of the process bootstrap in p24.sim.breakeven), b3_interp_check.csv (if a measured B3 exists next to B2/B4),
build_meta.json.

Rules
 * Process = statistical unit (protocol sec 7). Steady rows are filtered exactly as p24.stats (contended, cpu load, P-core,
   recompile limit, status ok); tier_violation processes are dropped from one-time statistics only.
 * Tiers: cold_empty / cold_diskcache come from the cold_tier column (--tier-map renames labels, e.g. prewarm=cold_diskcache);
   key i>0 of a bench_seq run is `inproc_empty` / `inproc_diskcache` according to its meta cache_mode. Cache-independent
   arms (E0, E1, G1, G2) fill BOTH cold tiers from all non-inproc processes.
 * Component first-call costs of compiled arms are kept only where the component key was NEW in that process (sequence order,
   p24.policy.component_keys): a component shared with an earlier shape contributes nothing to the intrinsic cost.
 * B3 = linear interpolation of the B2 and B4 rows (plan 9.3), source `interpolated_B3`; a measured B3 is kept (source
   `measured`) and the interpolation error is written to b3_interp_check.csv.
 * --extrapolate C2,C5,C6: shapes without a row get the C1 row scaled by the median arm/C1 ratio (source
   `extrapolated_ratio_C1`).
 * --fill-dev-synthetic: DEVELOPMENT ONLY. Fills the full 4-resolution x B1-4 x CFG grid for the arms present with a linear
   work model (source `dev_synthetic`) and splits missing component costs by global proportions (fc_source `dev_split`).
   Never use for results.

Usage (from code/): python -B -m scripts.build_cost_table --raw ../results/S2/cost_table/blockS1 [--raw DIR[=label]] \\
        [--seq DIR] [--summary CSV] [--transient CSV] --out-dir ../results/S2/breakeven
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from p24 import stats
from p24.policy import Shape, component_keys
from p24.sim.costs import CACHE_INDEPENDENT, COMPILE_ARMS, COMPONENTS, TIERS, proc_tier_label, proc_tier_mask
KEY = ["arm", "H", "W", "B", "cfg"]
RESOLUTIONS_HW = [(512, 512), (768, 768), (768, 512), (512, 768)]   # (H, W): portrait = 768x512 in H x W
FC_COLS = {"te": "first_call_te_s", "unet": "first_call_unet_s", "vae": "first_call_vae_s"}
PROC_COLS = ["capture_s", "capture_pool_mb", "static_io_mb", "steady_peak_reserved_mb", "weights_alloc_mb", "resident_mb"]
NUM_COLS = None  # set after columns are known


# ------------------------------------------------------------------ loading
def load_dir(path: pathlib.Path, block: str, seq: bool = False) -> pd.DataFrame:
    frames = []
    for f in sorted(path.glob("*.csv")):
        if f.name.startswith("summary"):
            continue
        d = pd.read_csv(f)
        d["block"] = block
        d["src"] = path.name  # process identity = (block, src dir, arm, proc_id); fix 2026-11-16 01:35
        if seq or "seq_idx" in d.columns and d["seq_idx"].notna().any():
            arm, pid = d["arm"].iloc[0], int(d["proc_id"].iloc[0])
            meta = path / f"{arm}_seq_p{pid}.meta.json"
            mode = json.loads(meta.read_text()).get("cache_mode", "env") if meta.exists() else "env"
            lab = {"empty": "inproc_empty", "disk": "inproc_diskcache"}.get(mode, "inproc_env")
            d["cold_tier"] = d["cold_tier"].where(d["cold_tier"] != "inproc_first", lab)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ per process
def per_process(df: pd.DataFrame, tier_map: dict, include_contended=False, include_cpu_contended=False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "src" not in df:
        df["src"] = ""
    df["cold_tier"] = df["cold_tier"].replace(tier_map)
    if "seq_idx" not in df:
        df["seq_idx"] = np.nan
    df["seq_idx"] = df["seq_idx"].fillna(-1).astype(int)
    for c in ["tier_violation", *FC_COLS.values(), *PROC_COLS]:
        if c not in df:
            df[c] = np.nan
    ok = stats._filtered_steady(df, include_contended, include_cpu_contended)
    ok = ok[ok.wall_ms.notna() & (ok.rep >= 0)]
    gk = ["block", "src", "arm", "H", "W", "B", "cfg", "proc_id", "seq_idx", "cold_tier"]
    agg = dict(steady_ms=("wall_ms", "median"), n_steady=("wall_ms", "size"), first_req_s=("first_req_s", "first"),
               tier_violation=("tier_violation", "first"))
    agg.update({c: (c, "first") for c in [*FC_COLS.values(), *PROC_COLS]})
    pp = ok.groupby(gk, dropna=False).agg(**agg).reset_index()
    pp["tier_violation"] = pp["tier_violation"].fillna(0).astype(int)
    # intrinsic component first calls: keep only components whose key is new in this process (sequence order)
    pp = pp.sort_values(["block", "src", "arm", "proc_id", "seq_idx"]).reset_index(drop=True)
    for c in COMPONENTS:
        pp[f"_shared_{c}"] = 0
    # one process = one (block, src dir, arm, proc_id); before 2026-11-16 01:35 processes of different sequence dirs
    # in the same block were merged here (component keys wrongly marked as shared; same-label rows merged)
    for (blk, src, arm, pid), g in pp.groupby(["block", "src", "arm", "proc_id"], sort=False):
        if arm not in COMPILE_ARMS or (g.seq_idx < 0).all():
            continue
        seen = set()
        for ix, r in g.iterrows():
            ks = component_keys(Shape(int(r.H), int(r.W), int(r.B), int(r.cfg)))
            for c in COMPONENTS:
                if ks[c] in seen:
                    pp.at[ix, FC_COLS[c]] = np.nan
                    pp.at[ix, f"_shared_{c}"] = 1
            seen.update(ks.values())
    pp["x_measured_s"] = pp.first_req_s - pp.steady_ms / 1e3
    fc_sum = pp[list(FC_COLS.values())].sum(axis=1, min_count=1).fillna(0.0)
    pp["resid_s"] = (pp.x_measured_s - fc_sum).clip(lower=0)
    comp = pp.arm.isin(COMPILE_ARMS)
    pp.loc[~comp, "resid_s"] = pp.loc[~comp, "x_measured_s"].clip(lower=0)
    # intrinsic one-time cost x_s: measured x unless a component was shared with an earlier shape (then NaN)
    shared_any = pp[[f"_shared_{c}" for c in COMPONENTS]].sum(axis=1) > 0
    pp["x_s"] = pp.x_measured_s.where(~(comp & shared_any))
    pp["tier"] = pp.cold_tier.map(proc_tier_label)
    return pp.drop(columns=[f"_shared_{c}" for c in COMPONENTS]).rename(columns={"cold_tier": "cold_tier_label"})


# ------------------------------------------------------------------ cells
def tier_subset(g: pd.DataFrame, arm: str, tier: str) -> pd.DataFrame:
    sub = g[proc_tier_mask(g.tier, arm, tier)]
    return sub[sub.tier_violation == 0]


def cells_from_procs(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in pp.groupby(KEY):
        arm = k[0]
        r = dict(zip(KEY, k))
        lo, hi = stats.boot_ci(g.steady_ms)
        r.update(n_proc=len(g), n_steady_min=int(g.n_steady.min()), steady_ms=float(g.steady_ms.median()),
                 steady_ci_lo=lo, steady_ci_hi=hi)
        for t in TIERS:
            s = tier_subset(g, arm, t)
            r[f"n__{t}"] = len(s)
            r[f"first_req_s__{t}"] = float(s.first_req_s.median()) if len(s) else np.nan
            r[f"resid_s__{t}"] = float(s.resid_s.median()) if len(s) else np.nan
            for c in COMPONENTS:
                v = s[FC_COLS[c]].dropna()
                r[f"fc_{c}_s__{t}"] = float(v.median()) if len(v) else np.nan
        r["capture_s"] = float(g.capture_s.median()) if g.capture_s.notna().any() else np.nan
        for c in ("capture_pool_mb", "static_io_mb", "steady_peak_reserved_mb", "weights_alloc_mb"):
            r[c] = float(g[c].max()) if g[c].notna().any() else np.nan
        r["equiv_pass"] = np.nan
        r["transient_excess_s"] = np.nan
        r["block"] = ";".join(sorted(g.block.unique()))
        r["source"] = "measured"
        r["fc_source"] = "measured" if any(np.isfinite(r[f"fc_{c}_s__{t}"]) for c in COMPONENTS for t in TIERS) else "missing"
        rows.append(r)
    return pd.DataFrame(rows)


def cells_from_summaries(paths, have: set) -> pd.DataFrame:
    rows = []
    for p in paths:
        s = pd.read_csv(p)
        for _, x in s.iterrows():
            key = (x.arm, int(x.H), int(x.W), int(x.B), int(x.cfg))
            if key in have or not np.isfinite(x.get("steady_wall_ms", np.nan)):
                continue
            r = dict(zip(KEY, key))
            r.update(n_proc=int(x.get("n_proc", 0)), n_steady_min=np.nan, steady_ms=x.steady_wall_ms,
                     steady_ci_lo=x.get("ci_lo", np.nan), steady_ci_hi=x.get("ci_hi", np.nan))
            for t in TIERS:
                for c in COMPONENTS:
                    r[f"fc_{c}_s__{t}"] = np.nan
                r[f"n__{t}"] = 0
                r[f"first_req_s__{t}"] = np.nan
                r[f"resid_s__{t}"] = np.nan
            for t, col in (("cold_empty", "first_req_s_cold_empty"), ("cold_diskcache", "first_req_s_diskcache")):
                v = x.get(col, np.nan)
                if np.isfinite(v):
                    r[f"first_req_s__{t}"] = v
                    r[f"resid_s__{t}"] = max(v - x.steady_wall_ms / 1e3, 0.0)
                    r[f"n__{t}"] = int(x.get("n_proc", 0))
            if x.arm in CACHE_INDEPENDENT:  # cache independent: same value in both cold tiers
                for col in ("first_req_s", "resid_s", "n"):
                    a, b = r.get(f"{col}__cold_empty"), r.get(f"{col}__cold_diskcache")
                    v = a if np.isfinite(a) else b
                    r[f"{col}__cold_empty"] = r[f"{col}__cold_diskcache"] = v
            r.update(capture_s=np.nan, capture_pool_mb=x.get("capture_pool_mb", np.nan),
                     static_io_mb=x.get("static_io_mb", np.nan),
                     steady_peak_reserved_mb=x.get("steady_peak_reserved_mb", np.nan),
                     weights_alloc_mb=x.get("weights_alloc_mb", np.nan), equiv_pass=np.nan, transient_excess_s=np.nan,
                     block=f"summary:{pathlib.Path(p).parent.name}/{pathlib.Path(p).name}", source="measured",
                     fc_source="missing")
            rows.append(r)
            have.add(key)
    return pd.DataFrame(rows)


def numeric_cols(df):
    skip = set(KEY) | {"block", "source", "fc_source", "equiv_pass"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def interpolate_b3(ct: pd.DataFrame):
    nums = numeric_cols(ct)
    new, chk = [], []
    for (arm, H, W, cfg), g in ct.groupby(["arm", "H", "W", "cfg"]):
        b = {int(r.B): r for _, r in g.iterrows()}
        if 2 not in b or 4 not in b:
            continue
        a2, a4 = b[2], b[4]
        interp = {c: (0.5 * (a2[c] + a4[c]) if np.isfinite(a2[c]) and np.isfinite(a4[c]) else np.nan) for c in nums}
        if 3 in b:
            chk.append(dict(arm=arm, H=H, W=W, cfg=cfg, steady_ms_measured=b[3].steady_ms,
                            steady_ms_interp=interp["steady_ms"],
                            rel_err=(interp["steady_ms"] - b[3].steady_ms) / b[3].steady_ms))
            continue
        r = dict(arm=arm, H=H, W=W, B=3, cfg=cfg, **interp)
        r.update(n_proc=min(a2.n_proc, a4.n_proc), block=f"interp({a2.block}|{a4.block})", source="interpolated_B3",
                 fc_source=a2.fc_source if a2.fc_source == a4.fc_source else "mixed", equiv_pass=np.nan)
        new.append(r)
    return pd.concat([ct, pd.DataFrame(new)], ignore_index=True) if new else ct, pd.DataFrame(chk)


def apply_equiv(ct: pd.DataFrame, path) -> pd.DataFrame:
    """Plan §5 / §9.8 (S2 acceptance item 2): equiv_pass = judged verdict of the arm served natively at that shape
    (1 / 0; NaN = not judged); equiv_pass_padB4 = verdict of a B3 shape served padded in the B4 bucket."""
    e = pd.read_csv(path)
    def key(k):
        hw, b, c = k.split("_")
        h, w = map(int, hw.split("x"))
        return h, w, int(b[1:]), int(c == "cfg")
    nat, pad = {}, {}
    for r in e.itertuples():
        h, w, b, c = key(r.key)
        (pad if str(r.padding).startswith("padded") else nat)[(r.arm, h, w, b, c)] = int(r.equiv_pass)
    ct = ct.copy()
    k = list(zip(ct.arm, ct.H.astype(int), ct.W.astype(int), ct.B.astype(int), ct.cfg.astype(int)))
    ct["equiv_pass"] = [nat.get(x, np.nan) for x in k]
    ct["equiv_pass_padB4"] = [pad.get(x, np.nan) for x in k]
    return ct


def extrapolate(ct: pd.DataFrame, arms) -> pd.DataFrame:
    c1 = ct[(ct.arm == "C1") & (ct.source != "dev_synthetic")].set_index(["H", "W", "B", "cfg"])
    nums = numeric_cols(ct)
    one_time = [c for c in nums if c.startswith(("first_req_s__", "resid_s__", "fc_"))]
    new = []
    for arm in arms:
        x = ct[ct.arm == arm].set_index(["H", "W", "B", "cfg"])
        both = x.index.intersection(c1.index)
        if len(both) == 0:
            print(f"extrapolate: no shape with both {arm} and C1, skipped", file=sys.stderr)
            continue
        rs = float(np.median((x.loc[both, "steady_ms"] / c1.loc[both, "steady_ms"]).to_numpy(float)))
        # one-time costs: per tier, median ratio of the MEASURED first-request times arm / C1 over shared shapes
        # (fix 2026-11-16 01:35, S2 acceptance item 3: the old single ratio of resid_s__cold_diskcache scaled every
        # one-time column, giving C2 ~10x and C5 ~1/5 of their measured costs)
        rf = {}
        for t in TIERS:
            col = f"first_req_s__{t}"
            if col not in x or col not in c1:
                rf[t] = np.nan
                continue
            rr = (x.loc[both, col] / c1.loc[both, col]).replace([np.inf, -np.inf], np.nan)
            rf[t] = float(np.nanmedian(rr.to_numpy(float))) if rr.notna().any() else np.nan
        for idx, r in c1.iterrows():
            if idx in x.index:
                continue
            d = {c: r[c] for c in nums}
            d["steady_ms"] = r.steady_ms * rs
            for c in ("steady_ci_lo", "steady_ci_hi"):
                d[c] = r[c] * rs
            for c in one_time:
                t = c.split("__", 1)[1] if "__" in c else None
                f = rf.get(t, np.nan) if t else np.nan
                d[c] = r[c] * f if np.isfinite(f) else np.nan
            new.append(dict(arm=arm, H=idx[0], W=idx[1], B=idx[2], cfg=idx[3], **d, block=f"extrap:C1:{r.block}",
                            source="extrapolated_ratio_C1", fc_source=r.fc_source, equiv_pass=np.nan))
    return pd.concat([ct, pd.DataFrame(new)], ignore_index=True) if new else ct


def fill_dev_synthetic(ct: pd.DataFrame) -> pd.DataFrame:
    """DEVELOPMENT ONLY (see module docstring)."""
    nums = numeric_cols(ct)
    ct = ct.copy()
    # 1) component split for rows without component first calls
    for arm in sorted(set(ct.arm) & COMPILE_ARMS):
        m = ct.arm == arm
        have = ct[m & ct[[f"fc_{c}_s__cold_diskcache" for c in COMPONENTS]].notna().all(axis=1)]
        if len(have):
            tot = have[[f"fc_{c}_s__cold_diskcache" for c in COMPONENTS]].sum(axis=1)
            prop = {c: float((have[f"fc_{c}_s__cold_diskcache"] / tot).median()) for c in COMPONENTS}
        else:
            prop = {"te": 0.15, "unet": 0.55, "vae": 0.30}
        s = sum(prop.values())
        for ix in ct[m].index:
            for t in TIERS:
                if all(np.isnan(ct.at[ix, f"fc_{c}_s__{t}"]) for c in COMPONENTS) and np.isfinite(ct.at[ix, f"resid_s__{t}"]):
                    x = ct.at[ix, f"resid_s__{t}"]
                    for c in COMPONENTS:
                        ct.at[ix, f"fc_{c}_s__{t}"] = 0.98 * x * prop[c] / s
                    ct.at[ix, f"resid_s__{t}"] = 0.02 * x
                    ct.at[ix, "fc_source"] = "dev_split"
    # 2) full grid
    def work(H, W, B, cfg):
        return B * (2 if cfg else 1) * H * W / (512 * 512)
    new = []
    for arm in sorted(ct.arm.unique()):
        a = ct[(ct.arm == arm) & (ct.source != "dev_synthetic")]
        if a.empty:
            continue
        w = np.array([work(r.H, r.W, r.B, r.cfg) for r in a.itertuples()])
        y = a.steady_ms.to_numpy(float)
        if len(set(np.round(w, 6))) >= 2:
            b_, a_ = np.polyfit(w, y, 1)
        else:
            b_, a_ = y.mean() / w.mean(), 0.0
        have = {(r.H, r.W, r.B, r.cfg) for r in a.itertuples()}
        for (H, W) in RESOLUTIONS_HW:
            for B in (1, 2, 3, 4):
                for cfg in (0, 1):
                    if (H, W, B, cfg) in have:
                        continue
                    wn = work(H, W, B, cfg)
                    j = int(np.argmin(np.abs(np.log(w / wn))))
                    near = a.iloc[j]
                    d = {c: near[c] for c in nums}
                    d["steady_ms"] = max(a_ + b_ * wn, 1.0)
                    d["steady_ci_lo"] = d["steady_ci_hi"] = np.nan
                    for c in ("capture_pool_mb", "static_io_mb", "steady_peak_reserved_mb"):
                        okm = np.isfinite(a[c].to_numpy(float))  # nearest row that HAS this quantity, scaled by work
                        if okm.any():
                            jj = int(np.argmin(np.where(okm, np.abs(np.log(w / wn)), np.inf)))
                            d[c] = float(a.iloc[jj][c]) * wn / w[jj]
                        else:
                            d[c] = np.nan
                    new.append(dict(arm=arm, H=H, W=W, B=B, cfg=cfg, **d, block="dev_synthetic", source="dev_synthetic",
                                    fc_source=near.fc_source, equiv_pass=np.nan))
    return pd.concat([ct, pd.DataFrame(new)], ignore_index=True)


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="append", default=[], help="raw bench dir[=block label]; repeatable")
    ap.add_argument("--seq", action="append", default=[], help="bench_seq output dir[=block label]; repeatable")
    ap.add_argument("--summary", action="append", default=[], help="p24.stats summary csv (cells without raw data); repeatable")
    ap.add_argument("--transient", default=None, help="p24.stats --transient csv -> column transient_excess_s")
    ap.add_argument("--tier-map", default="", help="e.g. prewarm=cold_diskcache,cold_partial=other")
    ap.add_argument("--equiv-matrix", default=None,
                    help="results/S2/equiv/equiv_matrix.csv: equiv_pass (native verdict) and equiv_pass_padB4 (B3 served in B4)")
    ap.add_argument("--extrapolate", default="", help="e.g. C2,C5,C6")
    ap.add_argument("--fill-dev-synthetic", action="store_true", help="DEVELOPMENT ONLY")
    ap.add_argument("--include-contended", action="store_true")
    ap.add_argument("--include-cpu-contended", action="store_true")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    tier_map = dict(kv.split("=") for kv in a.tier_map.split(",") if kv)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("cost_table.csv", "cost_table_procs.csv"):
        if (out / f).exists():
            sys.exit(f"REFUSING to overwrite {out / f} (use a new --out-dir)")
    frames, inputs = [], []
    for spec, seq in [(s, False) for s in a.raw] + [(s, True) for s in a.seq]:
        path, _, label = spec.partition("=")
        p = pathlib.Path(path)
        label = label or p.name
        d = load_dir(p, label, seq=seq)
        inputs.append(dict(path=str(p), block=label, seq=seq, n_rows=len(d)))
        frames.append(d)
    raw = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    pp = per_process(raw, tier_map, a.include_contended, a.include_cpu_contended)
    ct = cells_from_procs(pp) if len(pp) else pd.DataFrame()
    have = set(map(tuple, ct[KEY].to_numpy().tolist())) if len(ct) else set()
    if a.summary:
        cs = cells_from_summaries(a.summary, have)
        for p in a.summary:
            inputs.append(dict(path=str(p), kind="summary", sha256_16=sha16(p)))
        ct = pd.concat([ct, cs], ignore_index=True) if len(ct) else cs
    if ct.empty:
        sys.exit("no cost rows built")
    ct, chk = interpolate_b3(ct)
    if a.extrapolate:
        ct = extrapolate(ct, [x for x in a.extrapolate.split(",") if x])
    if a.transient:
        tr = pd.read_csv(a.transient)
        m = ct.merge(tr[KEY + ["excess_s_median"]], on=KEY, how="left")
        ct["transient_excess_s"] = m["excess_s_median"].to_numpy()
        inputs.append(dict(path=a.transient, kind="transient", sha256_16=sha16(a.transient)))
    if a.equiv_matrix:
        ct = apply_equiv(ct, a.equiv_matrix)
        inputs.append(dict(path=a.equiv_matrix, kind="equiv_matrix", sha256_16=sha16(a.equiv_matrix)))
    if a.fill_dev_synthetic:
        ct = fill_dev_synthetic(ct)
    ct = ct.sort_values(["H", "W", "B", "cfg", "arm"]).reset_index(drop=True)
    lead = KEY + ["source", "block", "n_proc", "n_steady_min", "steady_ms", "steady_ci_lo", "steady_ci_hi"]
    ct = ct[lead + [c for c in ct.columns if c not in lead]]
    ct.to_csv(out / "cost_table.csv", index=False)
    pp.to_csv(out / "cost_table_procs.csv", index=False)
    if len(chk):
        chk.to_csv(out / "b3_interp_check.csv", index=False)
    code = pathlib.Path(__file__).resolve()
    (out / "build_meta.json").write_text(json.dumps(dict(
        time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, inputs=inputs, tier_map=tier_map,
        dev_synthetic=bool(a.fill_dev_synthetic), n_rows=len(ct), by_source=ct.source.value_counts().to_dict(),
        script_sha256_16=sha16(code), n_procs=len(pp)), indent=2, default=str))
    print(f"cost_table.csv: {len(ct)} rows {ct.source.value_counts().to_dict()}; procs: {len(pp)}; -> {out}")


if __name__ == "__main__":
    main()
