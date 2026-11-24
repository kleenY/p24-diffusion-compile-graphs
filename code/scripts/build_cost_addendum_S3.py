"""S3 3.1b: build the cost ADDENDUM from the S3-A block output (CPU only, no torch). Does not touch the S2 cost table.

Input (default = what scripts/window_S3_A.sh writes, root results/S3/cost_addendum/):
  <raw>/inproc_seq/v3shapes_<ARM>_disk/<ARM>_seq<i>_<shape>_p<id>.csv + <ARM>_seq_p<id>.meta.json   Part 1, ARM in C2, C5, proc-id 0-2 only
        (warm-pass dirs warmpass_<ARM> / proc-id 9 are NOT read)
  <raw>/rebuild/rebuild_<ARM>_p0.csv + .meta.json                                                  Part 2, ARM in G1, C1+G1
  <raw>/S3A_logs/<tag>.pinned_fallback_count.txt, <tag>.hostmeta.json, block.log                   flags (optional)
Output: <out-dir>/cost_addendum.csv + meta.json (input SHAs, process counts, rules, OOM keys, memory fall-back, pinned fall-back count).

Rules (main-session decisions, S3 prep 2026-11-17)
 * inproc_cost rows: per (arm, shape) median over the standard processes of first_req_s / resid_s / component first calls, computed with the
   SAME per-process machinery as scripts/build_cost_table.py (component keys shared with an earlier shape of the process are NaN, resid_s =
   first_req - steady - sum of kept component calls). Key 0 of a sequence is the cold tier (not inproc) -> no row. source=inproc_measured_S3.
   A key that failed (status=fail, e.g. the C2 cumulative-pool OOM) in ANY standard process of the arm, and every key after it in that
   process, gets NO row (stays "extrapolated" in the table); the failing key/error go to meta `oom_keys`.
 * rebuild rows: per (arm, shape) median over ok rounds of rebuild_wall_s (re-capture wall incl. the warm-up calls of runner.prepare) and of
   cap1_capture_s; ratio = rebuild_wall_med / capture_s_med; ratio > --ratio-thr (1.5) -> rebuild_override_s = rebuild_wall_med, else NaN.
   ratio_vs_table_capture_s (same numerator over the S2 table capture_s) is informative only.
 * extrapolated rebuild rows (main-session decision 2026-11-17): for an arm, the measured shapes whose ratio passed the gate (> --ratio-thr) give
   r_i = rebuild_wall_med / steady_s(arm, shape_i) (S2 table steady_ms); every table shape of that arm WITHOUT a measured rebuild row gets
   rebuild_override_s = median(r_i) x steady_s(arm, shape), source `extrapolated_S3_rebuild` (rationale: the rebuild wall is dominated by the
   warm-up calls of the re-prepare, proportional to one steady request). No passing shape -> no extrapolation. Ratios/shapes used go to meta.
 * flags (rows are kept, never dropped): pinned_fallback=<n> (process count file / meta), not_quiet_gate (NOT_QUIET printed by the block gate
   before that process), contended_rows (a row of the process with contended=1 or load1 > CPU_LOAD_MAX), nproc<3, partial_rounds.

Usage (from code/): python -B -m scripts.build_cost_addendum_S3 [--raw ../results/S3/cost_addendum] [--out-dir ../results/S3/cost_addendum]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

from p24.sim.addendum import ADDENDUM_COLS
from scripts import build_cost_table as BCT

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
DEF_RAW = ROOT / "results/S3/cost_addendum"
DEF_TABLE = ROOT / "results/S2/cost_table/merged/cost_table.csv"
SEQ_ARMS = ("C2", "C5")
RB_ARMS = ("G1", "C1+G1")
STD_PROCS = (0, 1, 2)
TIER = "inproc_diskcache"
RECOVER_TOL_MB = 64.0
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def _cpu_load_max() -> float:
    try:
        from p24.bench import CPU_LOAD_MAX
        return float(CPU_LOAD_MAX)
    except Exception:  # noqa: BLE001
        return 16.0


def _fallback_count(logs: pathlib.Path, tag: str, meta: dict | None = None):
    f = logs / f"{tag}.pinned_fallback_count.txt"
    if f.exists():
        try:
            return int(f.read_text().strip() or 0)
        except ValueError:
            return None
    if meta and "pinned_fallback_count" in meta:
        return int(meta["pinned_fallback_count"])
    return None


def _not_quiet_by_tag(block_log: pathlib.Path) -> dict:
    """tag -> list of NOT_QUIET lines printed after the previous STEP line and before this STEP line (the gate waited until quiet again)."""
    out, pend = {}, []
    if not block_log.exists():
        return out
    for l in block_log.read_text(errors="replace").splitlines():
        if "NOT_QUIET" in l:
            pend.append(l.strip())
        m = re.match(r"^STEP (\S+) rc=", l)
        if m:
            if pend:
                out[m.group(1)] = pend
            pend = []
    return out


# ------------------------------------------------------------------------------------------------ Part 1
def build_inproc(raw: pathlib.Path, logs: pathlib.Path, nq: dict, meta: dict) -> list:
    seq_root = raw / "inproc_seq"
    rows, minfo = [], {}
    for arm in SEQ_ARMS:
        d = seq_root / f"v3shapes_{arm}_disk"
        info = dict(dir=str(d), processes=[], oom_keys=[], n_rows=0)
        minfo[arm] = info
        if not d.is_dir():
            info["note"] = "directory missing: no rows (all shapes stay extrapolated)"
            continue
        metas = {}
        for p in STD_PROCS:
            mp = d / f"{arm}_seq_p{p}.meta.json"
            if mp.exists():
                metas[p] = json.loads(mp.read_text())
        # failure points per process: idx of the first non-ok key
        fail_idx, flags_by_proc, files = {}, {}, {}
        for p, m in metas.items():
            bad = [r for r in m.get("per_key", []) if r["status"] != "ok"]
            fail_idx[p] = bad[0]["idx"] if bad else None
            tag = f"v3shapes_{arm}_disk_p{p}"
            fl = []
            n = _fallback_count(logs, tag, m)
            if n:
                fl.append(f"pinned_fallback={n}")
            if tag in nq:
                fl.append("not_quiet_gate")
            flags_by_proc[p] = fl
            info["processes"].append(dict(proc=p, status=m.get("status"), n_keys=len(m.get("per_key", [])), fail_idx=fail_idx[p],
                                          pinned_fallback_count=n, not_quiet=nq.get(tag, []), meta_sha16=sha16(d / f"{arm}_seq_p{p}.meta.json")))
            for r in bad:
                info["oom_keys"].append(dict(proc=p, idx=r["idx"], key=r["key"], error=str(r.get("error"))[:300]))
        if not metas:
            info["note"] = "no standard-process meta found: no rows"
            continue
        for f in sorted(d.glob("*.csv")):
            files[f.name] = sha16(f)
        info["csv_sha16"] = files
        raw_df = BCT.load_dir(d, d.name, seq=True)
        raw_df = raw_df[raw_df.proc_id.isin(list(metas))]
        if raw_df.empty:
            info["note"] = "no csv rows"
            continue
        # contended rows per process
        cm = _cpu_load_max()
        for p in metas:
            g = raw_df[raw_df.proc_id == p]
            busy = (g[["load1_start", "load1_end"]].max(axis=1) > cm).any() if "load1_start" in g else False
            if (g.get("contended", pd.Series([0])).fillna(0) != 0).any() or busy:
                flags_by_proc[p].append("contended_rows")
        pp = BCT.per_process(raw_df, {})
        pp = pp[(pp.seq_idx > 0) & (pp.arm == arm)]
        n_std = len(metas)
        for (H, W, B, cfg), g in pp.groupby(["H", "W", "B", "cfg"]):
            idx = int(g.seq_idx.iloc[0])
            # a key is covered only if no standard process failed at or before it
            if any(fi is not None and idx >= fi for fi in fail_idx.values()):
                continue
            s = BCT.tier_subset(g, arm, TIER)
            if s.empty:
                continue
            fl = sorted({x for p in s.proc_id.astype(int) for x in flags_by_proc.get(p, [])})
            if len(s) < 3:
                fl.append(f"nproc<3({len(s)} of {n_std} std procs)")
            row = dict(kind="inproc_cost", arm=arm, H=int(H), W=int(W), B=int(B), cfg=int(cfg), source="inproc_measured_S3",
                       flag=";".join(fl), n_proc=len(s), first_req_s=float(s.first_req_s.median()), resid_s=float(s.resid_s.median()))
            for c in BCT.COMPONENTS:
                v = s[BCT.FC_COLS[c]].dropna()
                row[f"fc_{c}_s"] = float(v.median()) if len(v) else np.nan
            rows.append(row)
            info["n_rows"] += 1
    meta["inproc"] = minfo
    return rows


# ------------------------------------------------------------------------------------------------ Part 2
def build_rebuild(raw: pathlib.Path, logs: pathlib.Path, nq: dict, thr: float, table: pd.DataFrame | None, meta: dict) -> list:
    rows, minfo = [], {}
    for arm in RB_ARMS:
        cp, mp = raw / "rebuild" / f"rebuild_{arm}_p0.csv", raw / "rebuild" / f"rebuild_{arm}_p0.meta.json"
        info = dict(csv=str(cp), n_rows=0, shapes={})
        minfo[arm] = info
        if not cp.exists():
            info["note"] = "csv missing: no rows"
            continue
        m = json.loads(mp.read_text()) if mp.exists() else {}
        tag = f"rebuild_{arm}_p0"
        n_fb = _fallback_count(logs, tag, m)
        info.update(csv_sha16=sha16(cp), meta_sha16=sha16(mp) if mp.exists() else None, pinned_fallback_count=n_fb,
                    not_quiet=nq.get(tag, []), any_fail=m.get("any_fail"))
        d = pd.read_csv(cp)
        for (H, W, B, cfg), g in d.groupby(["H", "W", "B", "cfg"], sort=False):
            ok = g[g.status == "ok"]
            nfail = int((g.status != "ok").sum())
            key = f"{H}x{W}x{B}x{cfg}"
            if ok.empty:
                info["shapes"][key] = dict(n_ok=0, n_fail=nfail, error=str(g.error.iloc[0])[:300])
                continue
            reb = float(ok.rebuild_wall_s.median())
            cap = float(ok.cap1_capture_s.median())
            ratio = reb / cap if cap > 0 else np.nan
            ov = reb if (np.isfinite(ratio) and ratio > thr) else np.nan
            fl = []
            if n_fb:
                fl.append(f"pinned_fallback={n_fb}")
            if tag in nq:
                fl.append("not_quiet_gate")
            cm = _cpu_load_max()
            if (g[["load1_start", "load1_end"]].max(axis=1) > cm).any() or (g[["gpu0_other_procs_start", "gpu0_other_procs_end"]].max(axis=1) > 0).any():
                fl.append("contended_rows")
            if nfail:
                fl.append("partial_rounds")
            if not (ok.output_equal_after_rebuild == 1).all():
                fl.append("output_not_equal_after_rebuild")
            resid = ok.reserved_residual_mb
            rec = float((ok.reserved_recovered == 1).mean())
            row = dict(kind="rebuild", arm=arm, H=int(H), W=int(W), B=int(B), cfg=int(cfg), source="rebuild_measured_S3",
                       flag=";".join(fl), n_proc=1, n_rounds=len(ok), rebuild_wall_med_s=reb, capture_s_med=cap, rebuild_over_capture=ratio,
                       rebuild_override_s=ov, reserved_residual_med_mb=float(resid.median()), recovered_frac=rec, n_fail_rounds=nfail)
            rows.append(row)
            info["n_rows"] += 1
            sh = dict(n_ok=len(ok), n_fail=nfail, rebuild_wall_med_s=reb, capture_s_med=cap, ratio=ratio, override=bool(np.isfinite(ov)),
                      reserved_residual_med_mb=float(resid.median()), reserved_residual_max_abs_mb=float(resid.abs().max()),
                      recovered_frac=rec, memory_reserved_falls_back=bool(rec == 1.0))
            if table is not None:
                t = table[(table.arm == arm) & (table.H == H) & (table.W == W) & (table.B == B) & (table.cfg == cfg)]
                if len(t) and pd.notna(t.capture_s.iloc[0]):
                    sh["table_capture_s"] = float(t.capture_s.iloc[0])
                    sh["ratio_vs_table_capture_s"] = reb / float(t.capture_s.iloc[0])
            info["shapes"][key] = sh
        rs = [s["ratio"] for s in info["shapes"].values() if "ratio" in s and np.isfinite(s["ratio"])]
        info["median_ratio_over_shapes"] = float(np.median(rs)) if rs else None   # informative: NOT applied to unmeasured shapes
    if table is not None:
        rows += _extrapolate_rebuild(rows, table, minfo)
    else:
        for info in minfo.values():
            info["extrapolation"] = "skipped: no cost table (steady_s needed)"
    meta["rebuild"] = minfo
    return rows


def _extrapolate_rebuild(rows: list, table: pd.DataFrame, minfo: dict) -> list:
    out = []
    for arm in RB_ARMS:
        info = minfo.get(arm)
        meas = [r for r in rows if r["kind"] == "rebuild" and r["arm"] == arm]
        t = table[table.arm == arm].copy()
        t["steady_s"] = t.steady_ms / 1e3
        steady = {(int(r.H), int(r.W), int(r.B), int(r.cfg)): float(r.steady_s) for r in t.itertuples() if np.isfinite(r.steady_s)}
        used = {}
        for r in meas:
            k = (r["H"], r["W"], r["B"], r["cfg"])
            if np.isfinite(r["rebuild_override_s"]) and k in steady:          # same gate as the measured override
                used[f"{k[0]}x{k[1]}x{k[2]}x{k[3]}"] = r["rebuild_wall_med_s"] / steady[k]
        ex = dict(ratios_rebuild_over_steady=used, median_ratio=None, n_extrapolated=0)
        if used:
            med = float(np.median(list(used.values())))
            ex["median_ratio"] = med
            done = {(r["H"], r["W"], r["B"], r["cfg"]) for r in meas}
            for k, st in sorted(steady.items()):
                if k in done:
                    continue
                row = {c: np.nan for c in ADDENDUM_COLS}
                row.update(kind="rebuild", arm=arm, H=k[0], W=k[1], B=k[2], cfg=k[3], source="extrapolated_S3_rebuild", flag="",
                           n_proc=0, rebuild_override_s=med * st)
                out.append(row)
                ex["n_extrapolated"] += 1
        else:
            ex["note"] = "no measured shape passed the ratio gate (or no measurement): nothing extrapolated"
        if info is not None:
            info["extrapolation"] = ex
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(DEF_RAW), help="S3-A raw root (contains inproc_seq/, rebuild/, S3A_logs/)")
    ap.add_argument("--out-dir", default=str(DEF_RAW))
    ap.add_argument("--ratio-thr", type=float, default=1.5)
    ap.add_argument("--cost-table", default=str(DEF_TABLE), help="S2 table, read only for the informative ratio_vs_table_capture_s")
    a = ap.parse_args(argv)
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out_dir)
    if not raw.is_dir():
        sys.exit(f"raw dir not found: {raw}")
    out.mkdir(parents=True, exist_ok=True)
    csv_p, meta_p = out / "cost_addendum.csv", out / "meta.json"
    for f in (csv_p, meta_p):
        if f.exists():
            sys.exit(f"REFUSING to overwrite {f} (use a new --out-dir)")
    logs = raw / "S3A_logs"
    nq = _not_quiet_by_tag(logs / "block.log")
    table = pd.read_csv(a.cost_table) if a.cost_table and pathlib.Path(a.cost_table).exists() else None
    meta = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, raw=str(raw), ratio_thr=a.ratio_thr, std_procs=list(STD_PROCS),
                rules=__doc__.split("Rules")[1].split("Usage")[0].strip(),
                cost_table=dict(path=a.cost_table, sha16=sha16(a.cost_table) if table is not None else None),
                block_log_sha16=sha16(logs / "block.log") if (logs / "block.log").exists() else None)
    rows = build_inproc(raw, logs, nq, meta) + build_rebuild(raw, logs, nq, a.ratio_thr, table, meta)
    df = pd.DataFrame(rows, columns=ADDENDUM_COLS)
    df.to_csv(csv_p, index=False)
    meta.update(n_rows=len(df), by_kind=df.kind.value_counts().to_dict(),
                n_inproc_rows_by_arm=df[df.kind == "inproc_cost"].arm.value_counts().to_dict(),
                n_override=int(df.rebuild_override_s.notna().sum()),
                n_override_measured=int((df.rebuild_override_s.notna() & (df.source == "rebuild_measured_S3")).sum()),
                n_override_extrapolated=int((df.rebuild_override_s.notna() & (df.source == "extrapolated_S3_rebuild")).sum()), n_flagged=int((df.flag.fillna("") != "").sum()),
                oom_keys={arm: v["oom_keys"] for arm, v in meta.get("inproc", {}).items()},
                script_sha16=sha16(pathlib.Path(__file__).resolve()), addendum_csv_sha16=sha16(csv_p))
    meta_p.write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    print(f"cost_addendum.csv: {len(df)} rows {meta['by_kind']} overrides={meta['n_override']} flagged={meta['n_flagged']} -> {out}")


if __name__ == "__main__":
    main()
