"""S4-P P5c step 1 (CPU, no torch): block-A cost table v2 + the AOT rows -> results/S4P/p5c/cost_table_v2_aot.csv (0444).
Preregistration goals/S4P_P5c_AOT_confirmatory_prereg_v1.md (0444, SHA16 c14507708a968699) sec 2 (rules written there, NOT chosen here):
  arm name "AOT", one row per shape that has a prebuilt package (32).  One-time cost = first request (package load + first full 4-step generation, bench convention first_req_s) - the same process's steady median.
  inproc_diskcache : median over the AOTseq processes of (first_req - own steady), POSITIONS >= 2 of each sequence only (position 1 of a process is a new-process first request, tier cold)   [n = 3, 2 for the three first shapes]
  cold_diskcache   : the 7 shapes of AOTcold (3 rounds, one new process each): median of (first_req - own steady); the other shapes = their inproc one-time + delta_cold,
                     delta_cold = MEDIAN over the AOTcold shapes of (cold - inproc) one-time
  cold_empty       : build wall-clock + the shape's cold_diskcache one-time;  inproc_empty: build wall-clock + the shape's inproc_diskcache one-time (upper bound, no sharing between shapes)
  build wall-clock : prebuild (results/S4P/p5c/prebuild/build/build_<tag>.json, `build_s_sum_comps` = export + AOTI compile + package of the three components, non-exclusive GPU but quiet gate);
                     calibration AOTbuild (measure_A_aot/build, exclusive): k = mean over the 2 shapes 512x512x1x0 / 768x768x4x1 of (exclusive / prebuild) build wall-clock;
                     |k-1| > 0.15 -> every build wall-clock x k, else as measured.  AOTbuild truncated (a calibration file missing) -> prebuild values as measured, rows flagged `build_nonexclusive`.
  steady_ms        : median over the AOTseq processes of the per-process steady median (all positions)
  sensitivity      : aot_resid_s__inproc_empty_sens = build wall-clock x (C1's own median inproc_empty/cold_empty one-time ratio, from the v2 table) + inproc_diskcache one-time
Components are not measured separately: everything is in resid_s, fc_* are NaN (the simulator gets no component-key sharing credit for AOT).
Outputs (never overwritten; 0444 after writing): <out> (= v2 table rows byte-for-byte unchanged + the AOT rows; extra columns aot_*), <out stem>_meta.json.
Flags are written, nothing is dropped silently: contended processes (other GPU0 process / load1 > 16 at start), n<3, build non-exclusive, fallback shape (aot_fallback=1; numbers unchanged, the simulator swaps in E0).
Cross-check: if <measure>/aot_seq_cells.csv (scripts/aot_cost_rows_S4P_P5c.py, not modified) exists, its per-shape pooled medians are compared with the pooled medians recomputed here (warnings in meta).
Usage (from code/, after the block-A add-on and `aot_cost_rows_S4P_P5c` on runs_seq / runs_cold):
  python -B -m scripts.aot_cost_table_S4P_P5c --v2-table ../results/S4P/p3/cost_v2/cost_table_v2.csv
  --synthetic (selftest only): output must NOT be under results/, raw files must be named *SYNTHETIC*; official mode refuses any raw file with SYNTHETIC in its name.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from p24.sim.costs import CostTable, MissingCost
from scripts import aot_tables_S4P_P5c as T

CALIB_TAGS = ("512x512_b1_nocfg", "768x768_b4_cfg")        # prereg sec 2: 512x512x1x0 and 768x768x4x1
K_BAND = 0.15
LOAD_MAX = 16.0
med = lambda xs: float(np.median(xs))


def load_proc_jsons(d: pathlib.Path) -> List[dict]:
    """AOTW_*.json of a runs dir -> list of dict(file, proc_id, ok, flags, keys=[per_key dict in sequence order])."""
    out = []
    for f in sorted(glob.glob(str(d / "AOTW_*.json"))):
        j = json.load(open(f))
        fl = []
        if (j.get("gpu0_other_procs_start") or 0) > 0 or (j.get("gpu0_other_procs_end") or 0) > 0:
            fl.append("other_gpu0_proc")
        l1 = (j.get("loadavg_start") or [0.0])[0]
        if l1 is not None and l1 > LOAD_MAX:
            fl.append(f"load1={l1:.1f}")
        out.append(dict(file=os.path.basename(f), proc_id=j.get("proc_id"), ok=j.get("status") == "ok", flags=fl, keys=j.get("per_key", [])))
    return out


def observations(procs: List[dict]) -> List[dict]:
    """One dict per (process, key) with status ok and a first request: shape, pos (0-based position in the process), first_req_s, steady_ms, steady_n, peak, flags."""
    obs = []
    for p in procs:
        if not p["ok"]:                      # same rule as aot_cost_rows_S4P_P5c.py: a process whose overall status is not ok contributes nothing (counted in meta.warnings)
            continue
        for pos, k in enumerate(p["keys"]):
            if k.get("status") != "ok" or k.get("first_req_s") is None or k.get("steady_wall_median_ms") is None:
                continue
            obs.append(dict(shape=T.shape_from_tag(k["key"]), pos=pos, first=float(k["first_req_s"]), steady_ms=float(k["steady_wall_median_ms"]),
                            steady_n=k.get("steady_n"), peak=k.get("steady_peak_reserved_mb"), proc=p["proc_id"], file=p["file"], flags=p["flags"],
                            proc_ok=p["ok"]))
    return obs


def load_builds(d: pathlib.Path) -> Dict[str, dict]:
    """build_<tag>.json (not .partial) -> {tag: dict(wall, bytes, status)}; wall = build_s_sum_comps = export + AOTI compile (+package) of the three components."""
    res = {}
    for f in sorted(glob.glob(str(d / "build_*.json"))):
        if f.endswith(".partial.json"):
            continue
        j = json.load(open(f))
        comps = j.get("comps", {})
        res[j["key"]] = dict(wall=float(j["build_s_sum_comps"]), status=j.get("status"), bytes=sum(int(c.get("package_bytes", 0) or 0) for c in comps.values()),
                             total_s=j.get("total_s"), other_procs=(j.get("gpu0_other_procs_start"), j.get("gpu0_other_procs_end")))
    return res


def c1_empty_ratio(v2: pd.DataFrame):
    """Median over the C1 rows with BOTH empty tiers measured of (inproc_empty one-time / cold_empty one-time) (v2 table, no imputation)."""
    ct = CostTable(v2)
    vals = []
    for r in v2[v2.arm == "C1"].itertuples():
        if not (pd.notna(getattr(r, "resid_s__cold_empty")) and pd.notna(getattr(r, "resid_s__inproc_empty"))):
            continue
        sh = T.Shape(int(r.H), int(r.W), int(r.B), int(r.cfg))
        try:
            a, b = ct.one_time("C1", sh, "inproc_empty").total_s, ct.one_time("C1", sh, "cold_empty").total_s
        except MissingCost:
            continue
        if b > 0 and np.isfinite(a) and np.isfinite(b):
            vals.append(a / b)
    return (med(vals) if vals else float("nan")), len(vals)


def build_rows(v2: pd.DataFrame, seq_obs, cold_obs, pre: Dict[str, dict], cal: Dict[str, dict], fallback, synthetic: bool):
    """-> (DataFrame of AOT rows, meta dict)."""
    meta: Dict[str, object] = {}
    warnings: List[str] = []
    pre_ok = {k: v for k, v in pre.items() if v["status"] == "ok"}
    bad_pre = sorted(set(pre) - set(pre_ok))
    if bad_pre:
        raise SystemExit(f"prebuild failed for {bad_pre}: AOT rows cannot be built (decide in the main session)")
    shapes = sorted({T.shape_from_tag(t) for t in pre_ok}, key=lambda s: (s.H, s.W, s.B, s.cfg))
    # ---- seq cells
    miss, seq_by = [], {}
    for sh in shapes:
        o = [x for x in seq_obs if x["shape"] == sh]
        inp = [x for x in o if x["pos"] > 0]
        if not o or not inp:
            miss.append(f"{sh.H}x{sh.W}x{sh.B}x{sh.cfg}")
        seq_by[sh] = (o, inp)
    if miss:
        raise SystemExit(f"AOTseq has no usable (position>=2) observation for {len(miss)} of {len(shapes)} prebuilt shapes: {miss} -> not guessing; main session decides")
    extra = sorted({x["shape"].tag() for x in seq_obs} - {T.shape_from_tag(t).tag() for t in pre_ok})
    if extra:
        warnings.append(f"AOTseq shapes without a prebuild file: {extra}")
    # ---- cold cells and delta_cold
    cold_cell, delta = {}, []
    for sh in sorted({x["shape"] for x in cold_obs}, key=lambda s: (s.H, s.W, s.B, s.cfg)):
        if sh not in seq_by:
            warnings.append(f"AOTcold shape {sh.tag()} not in the prebuilt set (ignored)")
            continue
        c = [x["first"] - x["steady_ms"] / 1e3 for x in cold_obs if x["shape"] == sh and x["pos"] == 0]
        if not c:
            continue
        cold_cell[sh] = (med(c), len(c), med([x["first"] for x in cold_obs if x["shape"] == sh]))
        delta.append(cold_cell[sh][0] - med([x["first"] - x["steady_ms"] / 1e3 for x in seq_by[sh][1]]))
    if not cold_cell:
        raise SystemExit("AOTcold has no usable observation: delta_cold undefined (the prereg gives no rule for a fully truncated AOTcold) -> main session decides")
    delta_cold = med(delta)
    if len(cold_cell) < 7:
        warnings.append(f"delta_cold from {len(cold_cell)} < 7 shapes")
    meta.update(delta_cold_s=delta_cold, delta_cold_per_shape={s.tag(): d for s, d in zip(sorted(cold_cell, key=lambda s: (s.H, s.W, s.B, s.cfg)), delta)},
                n_cold_shapes=len(cold_cell), n_seq_procs=len({x["file"] for x in seq_obs}))
    # ---- build wall-clock and k
    k_info = dict(rule="k = mean over the 2 calibration shapes of (exclusive/prebuild) build wall-clock; |k-1| > 0.15 -> all build wall-clocks x k, else as measured")
    cal_ok = all(t in cal and cal[t]["status"] == "ok" and t in pre_ok for t in CALIB_TAGS)
    if cal_ok:
        ratios = {t: cal[t]["wall"] / pre_ok[t]["wall"] for t in CALIB_TAGS}
        k = float(np.mean(list(ratios.values())))
        k_app = k if abs(k - 1.0) > K_BAND else 1.0
        k_info.update(ratios=ratios, k=k, k_applied=k_app, applied=bool(k_app != 1.0), status="calibrated (exclusive AOTbuild complete)")
    else:
        k, k_app = float("nan"), 1.0
        k_info.update(k=None, k_applied=1.0, applied=False, status="AOTbuild truncated or failed: prebuild values as measured (non-exclusive)",
                      present=sorted(cal))
    meta["build_calibration"] = k_info
    ratio_c1, n_ratio = c1_empty_ratio(v2)
    meta["c1_empty_ratio_inproc_over_cold"] = dict(median=ratio_c1, n_shapes=n_ratio)
    # ---- rows
    c1w = {(int(r.H), int(r.W), int(r.B), int(r.cfg)): r for r in v2[v2.arm == "C1"].itertuples()}
    rows = []
    for sh in shapes:
        o, inp = seq_by[sh]
        tag = sh.tag()
        steady_ms = med([x["steady_ms"] for x in o])
        res_in = med([x["first"] - x["steady_ms"] / 1e3 for x in inp])
        fr_in = med([x["first"] for x in inp])
        if sh in cold_cell:
            res_cold, n_cold, fr_cold = cold_cell[sh]
            cold_src = "measured"
        else:
            res_cold, n_cold, fr_cold, cold_src = res_in + delta_cold, 0, res_in + delta_cold + steady_ms / 1e3, "inproc+delta_cold"
        build_raw = pre_ok[tag]["wall"]
        build = build_raw * k_app
        flags = []
        for x in o + [y for y in cold_obs if y["shape"] == sh]:
            flags += x["flags"]
        flags = sorted(set(flags))
        if len(inp) < 3:
            flags.append(f"n_inproc={len(inp)}")
        if sh in cold_cell and cold_cell[sh][1] < 3:
            flags.append(f"n_cold={cold_cell[sh][1]}")
        if not cal_ok:
            flags.append("build_nonexclusive")
        if sh in fallback:
            flags.append("aot_fallback")
        if res_cold < 0 or res_in < 0:
            flags.append("negative_one_time")
        r = dict(arm=T.AOT_ARM, H=sh.H, W=sh.W, B=sh.B, cfg=sh.cfg, source="measured_aot_p5c", block="S4P-A-AOT", n_proc=len({x["file"] for x in o}),
                 n_steady_min=min([x["steady_n"] for x in o if x["steady_n"] is not None] or [np.nan]), steady_ms=steady_ms,
                 steady_peak_reserved_mb=med([x["peak"] for x in o if x["peak"] is not None]) if any(x["peak"] is not None for x in o) else np.nan,
                 weights_alloc_mb=getattr(c1w.get((sh.H, sh.W, sh.B, sh.cfg)), "weights_alloc_mb", np.nan), fc_source="aot_all_in_resid")
        r.update({"n__inproc_diskcache": len(inp), "first_req_s__inproc_diskcache": fr_in, "resid_s__inproc_diskcache": res_in,
                  "n__cold_diskcache": n_cold, "first_req_s__cold_diskcache": fr_cold, "resid_s__cold_diskcache": res_cold,
                  "resid_s__cold_empty": build + res_cold, "first_req_s__cold_empty": build + res_cold + steady_ms / 1e3,
                  "resid_s__inproc_empty": build + res_in, "first_req_s__inproc_empty": build + res_in + steady_ms / 1e3})
        r.update(aot_build_wall_raw_s=build_raw, aot_build_wall_s=build, aot_k_applied=k_app, aot_n_inproc=len(inp), aot_n_cold=n_cold, aot_cold_source=cold_src,
                 aot_delta_cold_s=delta_cold, aot_fallback=int(sh in fallback), aot_pkg_bytes=pre_ok[tag]["bytes"],
                 aot_resid_s__inproc_empty_sens=(build * ratio_c1 + res_in) if np.isfinite(ratio_c1) else np.nan, aot_flag=";".join(flags), aot_synthetic=int(synthetic))
        rows.append(r)
    df = pd.DataFrame(rows)
    meta.update(n_shapes=len(df), shapes_fallback=[s.tag() for s in shapes if s in fallback], build_total_wall_s=float(df.aot_build_wall_s.sum()),
                build_total_wall_raw_s=float(df.aot_build_wall_raw_s.sum()), pkg_total_bytes=int(df.aot_pkg_bytes.sum()),
                n_rows_flagged=int((df.aot_flag != "").sum()), flags=sorted({f for s in df.aot_flag for f in s.split(";") if f}), warnings=warnings)
    return df, meta


def cross_check(measure: pathlib.Path, seq_obs, warnings: List[str]) -> None:
    """Compare pooled medians (all positions, all ok processes) with the cells csv of the existing aot_cost_rows_S4P_P5c.py, when present."""
    f = measure / "aot_seq_cells.csv"
    if not f.exists():
        warnings.append("cross-check skipped: aot_seq_cells.csv not found (run scripts.aot_cost_rows_S4P_P5c on runs_seq first)")
        return
    cells = pd.read_csv(f).set_index("key")
    for tag, r in cells.iterrows():
        try:
            sh = T.shape_from_tag(str(tag))
        except ValueError:
            continue
        o = [x for x in seq_obs if x["shape"] == sh]
        if not o:
            warnings.append(f"cross-check: {tag} present in aot_seq_cells.csv but no observation here")
            continue
        a, b = med([x["steady_ms"] for x in o]), float(r["steady_wall_median_ms"])
        c, d = med([x["first"] for x in o]), float(r["first_req_s_median"])
        if not (np.isclose(a, b, rtol=1e-9) and np.isclose(c, d, rtol=1e-9)):
            warnings.append(f"cross-check MISMATCH {tag}: steady {a} vs {b}; first_req {c} vs {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v2-table", default=str(T.ROOT / "results/S4P/p3/cost_v2/cost_table_v2.csv"))
    ap.add_argument("--measure-dir", default=str(T.P5C / "measure_A_aot"))
    ap.add_argument("--prebuild-dir", default=str(T.P5C / "prebuild/build"))
    ap.add_argument("--fallback-csv", default=str(T.FALLBACK_CSV))
    ap.add_argument("--out", default=str(T.COST_TABLE_AOT))
    ap.add_argument("--synthetic", action="store_true", help="selftest: inputs/outputs outside results/, raw files named *SYNTHETIC*")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if not a.synthetic:
        T.check_prereg()
        assert T.is_ro(a.fallback_csv), f"{a.fallback_csv} must be 0444 (frozen before block A)"
    T.check_not_official_output(out, a.synthetic)
    if out.exists():
        sys.exit(f"REFUSING to overwrite {out}")
    meas = pathlib.Path(a.measure_dir)
    names = [p.name for p in meas.rglob("*") if p.is_file()]
    if not a.synthetic and any("SYNTHETIC" in n for n in names):
        sys.exit("REFUSED: SYNTHETIC files inside the measurement dir (official mode)")
    if a.synthetic and not any("SYNTHETIC" in n for n in names):
        sys.exit("REFUSED: --synthetic but no file named *SYNTHETIC* in the measurement dir")
    v2 = pd.read_csv(a.v2_table)
    assert not (v2.arm == T.AOT_ARM).any(), "the input table already has AOT rows"
    seq_p, cold_p = load_proc_jsons(meas / "runs_seq"), load_proc_jsons(meas / "runs_cold")
    seq_obs, cold_obs = observations(seq_p), observations(cold_p)
    pre = load_builds(pathlib.Path(a.prebuild_dir))
    cal = load_builds(meas / "build") if (meas / "build").is_dir() else {}
    fallback = T.read_fallback(a.fallback_csv)
    aot, meta = build_rows(v2, seq_obs, cold_obs, pre, cal, fallback, a.synthetic)
    for nm, ps in (("runs_seq", seq_p), ("runs_cold", cold_p)):
        bad = [p["file"] for p in ps if not p["ok"]]
        if bad:
            meta["warnings"].append(f"{nm}: {len(bad)} process(es) with overall status != ok ignored: {bad}")
    cross_check(meas, seq_obs, meta["warnings"])
    # C1 empty-cache first requests of the calibration (reported only)
    c1b = []
    for f in sorted(glob.glob(str(meas / "runs_c1build" / "C1_seq1_p*.json"))):
        try:
            j = json.load(open(f))
            for k in j.get("per_key", []):
                c1b.append(dict(key=k.get("key"), first_req_s=k.get("first_req_s"), steady_ms=k.get("steady_wall_median_ms")))
        except Exception as e:  # noqa: BLE001  (informational only)
            meta["warnings"].append(f"runs_c1build unreadable {os.path.basename(f)}: {e!r}")
    meta["c1_empty_first_requests_calibration"] = c1b
    full = pd.concat([v2, aot], ignore_index=True, sort=False)
    full = full[list(v2.columns) + [c for c in full.columns if c not in v2.columns]]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    full.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    pd.testing.assert_frame_equal(back[back.arm != T.AOT_ARM][list(v2.columns)].reset_index(drop=True), v2.reset_index(drop=True), check_dtype=False,
                                  obj="non-AOT rows (must be the input rows unchanged)")
    os.chmod(tmp, 0o444)
    tmp.rename(out)
    meta.update(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv if argv is None else list(argv), synthetic=a.synthetic, prereg_sha16=T.PREREG_SHA16,
                inputs=dict(v2_table=dict(path=str(a.v2_table), sha16=T.sha16(a.v2_table), mode=oct(T.mode_of(a.v2_table))), fallback_csv=dict(path=str(a.fallback_csv), sha16=T.sha16(a.fallback_csv)),
                            measure_dir=str(meas), prebuild_dir=str(a.prebuild_dir), n_seq_obs=len(seq_obs), n_cold_obs=len(cold_obs), n_prebuild=len(pre), n_calibration_build=len(cal)),
                out=dict(path=str(out), sha16=T.sha16(out), n_rows=len(full), n_aot_rows=len(aot)), script_sha16=T.sha16(pathlib.Path(__file__).resolve()),
                tables=dict(aot_cols=[c for c in full.columns if c not in v2.columns]))
    mp = out.with_name(out.stem + "_meta.json")
    mp.write_text(json.dumps(meta, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o), ensure_ascii=False))
    os.chmod(mp, 0o444)
    print(f"-> {out} (0444) rows={len(full)} (AOT {len(aot)}) sha16={T.sha16(out)}  k={meta['build_calibration'].get('k')} k_applied={meta['build_calibration']['k_applied']} "
          f"delta_cold={meta['delta_cold_s']:.4f}s build_total={meta['build_total_wall_s']:.0f}s flagged_rows={meta['n_rows_flagged']} warnings={len(meta['warnings'])}")
    for w in meta["warnings"]:
        print("WARNING", w)


if __name__ == "__main__":
    main()
