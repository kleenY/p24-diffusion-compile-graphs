"""S4-X X9 (CPU): verdict H1 / H2 / H3 + validity for the N=10000 real-machine long replay that ADDS the eager (B0) and AOT (B1-AOT) arms (W2, seed 133).
Rules = goals/S4X_X9_N1e4_eager_AOT_arms_prereg_v1.md sec 3 / 4 (the only basis; written BEFORE any X9 number).  Imports (unchanged) load helpers from scripts/a123_verdict_S4P_P5c.py (A.norm, A.boot_ratios, A._clean).
  T = process total service time (meta `service_s`).  X9 raw: B0 p0-p2, B1-AOT p0-p2, BRIDGE B1 p3, BRIDGE Ours p3.  X5 raw (real/raw of the X5 block, read only): B1 p0-p2, Ours p0-p2.  Per arm the mean of its repeats.
  H1  T_B1(X5 mean)   / T_B0     <= 0.95        (predicted 0.890)
  H2  T_Ours(X5 mean) / T_B0     <= 0.98        (predicted 0.929)
  H3  T_B1-AOT / T_B0 <= 0.90 on the arm means AND all three p-paired ratios (AOT_p / B0_p, p = 0..2) <= 0.90  -> holds ; means ok but a paired ratio misses -> partially holds ; mean line missed -> does not hold   (predicted 0.843)
  Bridge (not an H; decides how H1/H2 are written): |bridge B1 / X5 B1 mean - 1| <= 2 % AND |bridge Ours / X5 Ours mean - 1| <= 2 %  -> comparable across days, else DRIFT.  H1/H2 are always computed from the X5 means (card);
        on DRIFT the version with the bridge processes in place of the X5 means is attached (both versions reported; `bridge.substituted_version`, computed whenever a bridge process exists, `attached_by_card = DRIFT`).
  Reported only (not judged): prediction error of B0 and B1-AOT vs the frozen X5 prediction (|T_real - T_pred| / T_real, flag > 10 %, X5 line; signed value too), B1-AOT/B1 and B1-AOT/Ours (cross-day: X9 vs X5; bridge version too),
        first-request latency, compile / capture counts, peak_reserved (torch) and the device-level NVML peak when <tag>.nvml.csv exists (NVML_SAMPLE=1 in the window; else null), Ours eviction / denial counts,
        bootstrap 95 % CI of every ratio (10 000 resamples, default_rng(0), same function as X8; n = 3 / 1 is coarse).
Validity (card sec 4; failed -> the numbers are reported, NOT as H verdicts):
  V1 completeness: B0 p0-p2 and B1-AOT p0-p2 each ok with no request row status != ok -> a missing / not-ok B0 or B1-AOT process = UNFINISHED (no judgement; finished ones listed);
     bridge B1 p3 and bridge Ours p3 ok and clean, X5 B1 / Ours p0-p2 ok (the card asks for 8 ok processes; a missing / not-ok BRIDGE process is not covered by the card sentence on UNFINISHED -> read literally as INVALID, see report)
  V2 pinned_fallback_count == 0 for every X9 process (meta and <tag>.pinned_fallback_count.txt)                 -> INVALID
  V3 the prediction file is the frozen one (0444) and has the four arms of the stream; X5 side loaded                -> INVALID
  V4 NOT_QUIET (gate line before the STEP line of block_X9_*.log, or meta load1_start / load1_end > 16) = FLAGGED: verdict WITH the flagged processes (marked) AND an `excluding_flagged` variant; a flag alone does not void the judgement.
     X5-side flags (known, disclosed in results/S4X/x5/verdict/NOTE_not_quiet_attribution_20261121.txt) are only listed.
  status = UNFINISHED | INVALID | JUDGED.
Usage: "$PROJ_PYTHON" -B -m scripts.h123_verdict_S4X_X9 [--raw ../results/S4X/x9/real/raw --log-dir ../results/S4X/x9/real/logs --x5-raw ../results/S4X/x5/real/raw --x5-log-dir ../results/S4X/x5/real/logs
                                                         --pred ../results/S4X/x5/pred_X5.csv --out-dir ../results/S4X/x9/verdict]      (official: needs results/S4X/x9/PREREG_SHA16.txt matching the 0444 card; outputs set 0444)
       "$PROJ_PYTHON" -B -m scripts.h123_verdict_S4X_X9 --synthetic [--synthetic-dir tmp/x9_selftest]   (self-test: the real X5 B1 / Ours raw + invented B0 / B1-AOT / bridge data -> holds / does not hold / partially holds / DRIFT / UNFINISHED / INVALID / NOT_QUIET ...;
                                                         case outputs are kept under tmp/x9_selftest/<case>/, never under results/)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import pathlib
import re
import shutil
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import a123_verdict_S4P_P5c as A
from scripts import aot_tables_S4P_P5c as T

ROOT = T.ROOT
X9 = ROOT / "results/S4X/x9"
X5 = ROOT / "results/S4X/x5"
PREREG = ROOT / "goals/S4X_X9_N1e4_eager_AOT_arms_prereg_v1.md"
PREREG_SHA_FILE = X9 / "PREREG_SHA16.txt"
STREAM = "W2_N10000_seed133_B"
MAIN_ARMS = ["B0", "B1-AOT"]          # X9 p0-p2
BRIDGE_ARMS = ["B1", "Ours"]          # X9 p3 (bridge) and X5 p0-p2 (numerators of H1 / H2)
ALL_ARMS = ["B0", "B1-AOT", "B1", "Ours"]
N_PROC, BRIDGE_PROC = 3, 3
H1_LINE, H2_LINE, H3_LINE = 0.95, 0.98, 0.90
BRIDGE_TOL, ERR_FLAG, LOAD_MAX = 0.02, 0.10, 16.0
GROUPS = [("B0", "X9", "B0", range(N_PROC)), ("B1-AOT", "X9", "B1-AOT", range(N_PROC)), ("B1_X5", "X5", "B1", range(N_PROC)), ("Ours_X5", "X5", "Ours", range(N_PROC)),
          ("B1_bridge", "X9", "B1", [BRIDGE_PROC]), ("Ours_bridge", "X9", "Ours", [BRIDGE_PROC])]


def check_prereg() -> str:
    assert PREREG.is_file() and T.is_ro(PREREG), f"preregistration missing or not 0444: {PREREG}"
    assert PREREG_SHA_FILE.is_file(), f"{PREREG_SHA_FILE} missing (the main session writes it after the card is frozen)"
    exp, got = PREREG_SHA_FILE.read_text().strip(), T.sha16(PREREG)
    assert got == exp, f"preregistration SHA16 {got} != recorded {exp}"
    return got


def not_quiet_tags(log_dir: pathlib.Path, prefix: str) -> Dict[str, List[str]]:
    """tag -> NOT_QUIET lines printed (by the gate, while waiting) before that process's STEP line (block_<prefix>_*.log of the window script)."""
    out: Dict[str, List[str]] = {}
    pend: List[str] = []
    for f in sorted(glob.glob(str(log_dir / f"block_{prefix}_*.log"))):
        for l in pathlib.Path(f).read_text(errors="replace").splitlines():
            if "NOT_QUIET" in l:
                pend.append(l.strip())
            m = re.match(r"^STEP (?:\[\d+/\d+\] )?(\S+) rc=", l)
            if m:
                if pend:
                    out.setdefault(m.group(1), []).extend(pend)
                pend = []
    return out


def _nvml_peak(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path, usecols=["used_mb"])
        return float(d.used_mb.max()) if len(d) else None
    except Exception:
        return None


def load_procs(raw: pathlib.Path, log_dir: pathlib.Path, source: str) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.meta.json")):
        m = json.loads(f.read_text())
        stem = pathlib.Path(m["trace"]).stem
        pol = A.norm(m["policy"])
        proc = int(f.name.rsplit("_p", 1)[1].split(".")[0])
        tag = f"{m['policy']}_W2_warm_p{proc}"
        csvp = f.with_name(f.name.replace(".meta.json", ".csv"))
        pcf = log_dir / f"{tag}.pinned_fallback_count.txt"
        first_s, n_bad = None, 0
        if csvp.exists():
            first = None
            for q in csv.DictReader(open(csvp, newline="", encoding="utf-8")):
                if q.get("status", "ok") != "ok":
                    n_bad += 1
                if first is None and q.get("phase", "req") == "req":
                    first = q
            if first is not None:
                first_s = float(first["t_end_s"]) - float(first["t_start_s"])
        ec = m.get("event_counts") or {}
        hc = m.get("host_cols") or {}
        ao = m.get("aot") or {}
        rows.append(dict(source=source, stream=stem, policy=pol, proc=proc, tag=tag, status=m.get("status"), service_s=m.get("service_s"), pinned=m.get("pinned_fallback_count"),
                         pinned_file=(int(pcf.read_text().strip()) if pcf.exists() and pcf.read_text().strip().isdigit() else None), first_req_s=first_s,
                         n_compile=int(ec.get("compile", 0)) + int(ec.get("compile_capture", 0)), n_capture=int(ec.get("capture", 0)) + int(ec.get("compile_capture", 0)),
                         n_eager_first=int(ec.get("eager_first", 0)), n_capture_denied_events=int(ec.get("capture_denied", 0)), event_counts=json.dumps(ec, sort_keys=True),
                         peak_reserved_mb=m.get("peak_reserved_mb"), peak_graph_mb=m.get("peak_graph_mb"), nvml_peak_mb=_nvml_peak(log_dir / f"{tag}.nvml.csv"),
                         n_evicted_total=m.get("n_evicted_total"), n_denied=m.get("n_denied"), evict_log_len=len(m.get("evict_log") or []), runtime_evict_log_len=len(m.get("runtime_evict_log") or []),
                         aot_fb_requests=m.get("aot_fallback_requests"), aot_fb_shapes=json.dumps(m.get("aot_fallback_shapes")) if m.get("aot_fallback_shapes") is not None else None,
                         aot_dynamo_compile_events=ao.get("dynamo_compile_events"), aot_rt_cache_files_after=ao.get("rt_cache_files_after"),
                         load1_start=hc.get("load1_start"), load1_end=hc.get("load1_end"), load_s=m.get("load_s"), n_bad_rows=n_bad, csv=str(csvp) if csvp.exists() else "", file=f.name))
    return pd.DataFrame(rows)


def pred_info(pred: pathlib.Path) -> dict:
    d = pd.read_csv(pred)
    d["policy"] = d.policy.map(A.norm)
    d = d[d.stream == STREAM]
    col = lambda c: {r.policy: (float(getattr(r, c)) if c in d and pd.notna(getattr(r, c)) else None) for r in d.itertuples()}
    out = dict(T_pred=col("pred_total_s"), pred_peak_reserved_est_mb=col("pred_peak_reserved_est_mb"), pred_n_evicted=col("pred_n_evicted"), pred_n_denied=col("pred_n_denied"),
               event_counts_pred={r.policy: (r.event_counts if "event_counts" in d else None) for r in d.itertuples()})
    out["complete"] = all(a in out["T_pred"] and out["T_pred"][a] is not None for a in ALL_ARMS)
    if out["complete"]:
        t = out["T_pred"]
        out["ratios_pred"] = dict(B1_over_B0=t["B1"] / t["B0"], Ours_over_B0=t["Ours"] / t["B0"], AOT_over_B0=t["B1-AOT"] / t["B0"], AOT_over_B1=t["B1-AOT"] / t["B1"], AOT_over_Ours=t["B1-AOT"] / t["Ours"])
    return out


def _grp(P: pd.DataFrame, source: str, pol: str, procs, drop: set) -> pd.DataFrame:
    return P[(P.source == source) & (P.policy == pol) & (P.proc.isin(list(procs))) & (P.status == "ok") & (~P.tag.isin(drop))].sort_values("proc").set_index("proc")


def _ci(num: np.ndarray, den: np.ndarray):
    ci, _ = A.boot_ratios([np.asarray(num, float)], [np.asarray(den, float)])
    return ci[0]


def _mean_or_none(s: pd.Series):
    s = s.dropna()
    return float(s.mean()) if len(s) else None


def _max_or_none(s: pd.Series):
    s = s.dropna()
    return float(s.max()) if len(s) else None


def judge(P: pd.DataFrame, drop: set, pi: dict) -> dict:
    g = {name: _grp(P, src, pol, procs, drop) for name, src, pol, procs in GROUPS}
    res = dict(n_ok={k: int(len(v)) for k, v in g.items()})
    if any(len(g[k]) == 0 for k in ("B0", "B1-AOT", "B1_X5", "Ours_X5")):
        res["error"] = "an arm has no ok process"
        return res
    v = {k: x.service_s.to_numpy(float) for k, x in g.items()}
    Tm = {k: float(x.mean()) for k, x in v.items() if len(x)}
    t0 = Tm["B0"]
    r1, r2, r3 = Tm["B1_X5"] / t0, Tm["Ours_X5"] / t0, Tm["B1-AOT"] / t0
    common = sorted(set(g["B0"].index) & set(g["B1-AOT"].index))
    paired = {int(p): float(g["B1-AOT"].service_s[p] / g["B0"].service_s[p]) for p in common}
    h1, h2 = bool(r1 <= H1_LINE), bool(r2 <= H2_LINE)
    h3_mean = bool(r3 <= H3_LINE)
    h3_paired = bool(len(paired) > 0 and all(x <= H3_LINE for x in paired.values()))
    h3_label = "holds" if (h3_mean and h3_paired) else ("partially holds" if h3_mean else "does not hold")
    lab = lambda ok: "holds" if ok else "does not hold"
    res.update(T_mean=Tm, T_median={k: float(np.median(x)) for k, x in v.items() if len(x)}, T_values={k: [float(y) for y in x] for k, x in v.items()},
               H1=dict(passed=h1, label=lab(h1), ratio_B1X5_over_B0=float(r1), line=f"<= {H1_LINE}", ci95=_ci(v["B1_X5"], v["B0"]), equivalent_T_B0_min=float(Tm["B1_X5"] / H1_LINE)),
               H2=dict(passed=h2, label=lab(h2), ratio_OursX5_over_B0=float(r2), line=f"<= {H2_LINE}", ci95=_ci(v["Ours_X5"], v["B0"]), equivalent_T_B0_min=float(Tm["Ours_X5"] / H2_LINE)),
               H3=dict(passed=bool(h3_mean and h3_paired), label=h3_label, mean_line_ok=h3_mean, all_paired_ok=h3_paired, ratio_AOT_over_B0=float(r3), line=f"<= {H3_LINE}", paired_ratios_by_p=paired, ci95=_ci(v["B1-AOT"], v["B0"])))
    # ---- bridge
    br = {}
    for a in BRIDGE_ARMS:
        b, ref = g[f"{a}_bridge"], Tm[f"{a}_X5"]
        if len(b):
            bv = float(b.service_s.iloc[0])
            br[a] = dict(T_bridge=bv, T_X5_mean=ref, rel_diff_to_X5_mean=float(bv / ref - 1.0), within_2pct=bool(abs(bv / ref - 1.0) <= BRIDGE_TOL), X5_values=[float(y) for y in v[f"{a}_X5"]])
        else:
            br[a] = dict(T_bridge=None, T_X5_mean=ref, rel_diff_to_X5_mean=None, within_2pct=None)
    if all(br[a]["T_bridge"] is not None for a in BRIDGE_ARMS):
        bstat = "comparable across days" if all(br[a]["within_2pct"] for a in BRIDGE_ARMS) else "DRIFT"
    else:
        bstat = "NOT_AVAILABLE"
    sub = {}
    if br["B1"]["T_bridge"] is not None:
        sub["H1_bridge"] = dict(ratio_B1bridge_over_B0=float(br["B1"]["T_bridge"] / t0), passed=bool(br["B1"]["T_bridge"] / t0 <= H1_LINE), label=lab(br["B1"]["T_bridge"] / t0 <= H1_LINE), ci95=_ci(v["B1_bridge"], v["B0"]))
    if br["Ours"]["T_bridge"] is not None:
        sub["H2_bridge"] = dict(ratio_OursBridge_over_B0=float(br["Ours"]["T_bridge"] / t0), passed=bool(br["Ours"]["T_bridge"] / t0 <= H2_LINE), label=lab(br["Ours"]["T_bridge"] / t0 <= H2_LINE), ci95=_ci(v["Ours_bridge"], v["B0"]))
    res["bridge"] = dict(status=bstat, tol=BRIDGE_TOL, per_arm=br, substituted_version=sub, attached_by_card=bool(bstat == "DRIFT"),
                         note="diff = bridge / X5 mean - 1; H1 / H2 above use the X5 means (card); on DRIFT the substituted version (bridge process in place of the X5 mean) is the second version to report")
    # ---- reported only
    errs = {}
    for a, k in (("B0", "B0"), ("B1-AOT", "B1-AOT")):
        tp = pi["T_pred"].get(a) if pi.get("T_pred") else None
        if tp is not None:
            e = (tp - Tm[k]) / Tm[k]
            errs[a] = dict(T_real_mean=Tm[k], T_pred=tp, signed_pred_minus_real_over_real=float(e), abs_error=float(abs(e)), flag_gt_10pct=bool(abs(e) > ERR_FLAG))
    res["prediction_error_reported"] = errs
    rep = dict(AOT_over_B1_X5=dict(ratio=float(Tm["B1-AOT"] / Tm["B1_X5"]), ci95=_ci(v["B1-AOT"], v["B1_X5"]), cross_day=True),
               AOT_over_Ours_X5=dict(ratio=float(Tm["B1-AOT"] / Tm["Ours_X5"]), ci95=_ci(v["B1-AOT"], v["Ours_X5"]), cross_day=True))
    if "B1_bridge" in Tm:
        rep["AOT_over_B1_bridge"] = dict(ratio=float(Tm["B1-AOT"] / Tm["B1_bridge"]), ci95=_ci(v["B1-AOT"], v["B1_bridge"]), cross_day=False)
    if "Ours_bridge" in Tm:
        rep["AOT_over_Ours_bridge"] = dict(ratio=float(Tm["B1-AOT"] / Tm["Ours_bridge"]), ci95=_ci(v["B1-AOT"], v["Ours_bridge"]), cross_day=False)
    if pi.get("ratios_pred"):
        rep["ratios_pred"] = pi["ratios_pred"]
    res["reported"] = rep
    res["groups"] = {k: dict(n=int(len(x)), procs=[int(p) for p in x.index], first_request_s_mean=_mean_or_none(x.first_req_s), n_compile_mean=_mean_or_none(x.n_compile), n_capture_mean=_mean_or_none(x.n_capture),
                             n_eager_first_mean=_mean_or_none(x.n_eager_first), peak_reserved_mb_max=_max_or_none(x.peak_reserved_mb), peak_graph_mb_max=_max_or_none(x.peak_graph_mb),
                             nvml_peak_mb_max=_max_or_none(x.nvml_peak_mb), n_evicted_total={int(p): (None if pd.isna(r.n_evicted_total) else int(r.n_evicted_total)) for p, r in x.iterrows()},
                             n_denied={int(p): (None if pd.isna(r.n_denied) else int(r.n_denied)) for p, r in x.iterrows()}, capture_denied_events={int(p): int(r.n_capture_denied_events) for p, r in x.iterrows()},
                             evict_log_len={int(p): int(r.evict_log_len) for p, r in x.iterrows()}, runtime_evict_log_len={int(p): int(r.runtime_evict_log_len) for p, r in x.iterrows()})
                     for k, x in g.items()}
    res["groups"]["B1-AOT"]["aot_fallback_requests"] = {int(p): (None if pd.isna(r.aot_fb_requests) else int(r.aot_fb_requests)) for p, r in g["B1-AOT"].iterrows()}
    res["groups"]["B1-AOT"]["aot_dynamo_compile_events"] = {int(p): (None if pd.isna(r.aot_dynamo_compile_events) else int(r.aot_dynamo_compile_events)) for p, r in g["B1-AOT"].iterrows()}
    res["groups"]["B1-AOT"]["aot_rt_cache_files_after"] = {int(p): (None if pd.isna(r.aot_rt_cache_files_after) else int(r.aot_rt_cache_files_after)) for p, r in g["B1-AOT"].iterrows()}
    res["groups"]["B1-AOT"]["note"] = "events labelled `compile` in the B1-AOT rows are AOT package loads (no Dynamo / Inductor compile: aot_dynamo_compile_events must be 0)"
    res["pred_ours_evictions"] = dict(pred_n_evicted=pi.get("pred_n_evicted", {}).get("Ours"), pred_n_denied=pi.get("pred_n_denied", {}).get("Ours"))
    return res


def validity(P: pd.DataFrame, pred: pathlib.Path, pi: dict, nq9: Dict[str, List[str]], nq5: Dict[str, List[str]]) -> dict:
    v = {}
    X = P[(P.source == "X9") & (P.stream == STREAM)]
    X5r = P[(P.source == "X5") & (P.stream == STREAM)]
    ok_ids = {}
    for a in MAIN_ARMS:
        ok_ids[a] = sorted(int(p) for p in X[(X.policy == a) & (X.status == "ok") & (X.proc.isin(range(N_PROC)))].proc)
    for a in BRIDGE_ARMS:
        ok_ids[f"{a}_bridge"] = sorted(int(p) for p in X[(X.policy == a) & (X.status == "ok") & (X.proc == BRIDGE_PROC)].proc)
        ok_ids[f"{a}_X5"] = sorted(int(p) for p in X5r[(X5r.policy == a) & (X5r.status == "ok") & (X5r.proc.isin(range(N_PROC)))].proc)
    want = {a: list(range(N_PROC)) for a in MAIN_ARMS}
    want.update({f"{a}_bridge": [BRIDGE_PROC] for a in BRIDGE_ARMS})
    want.update({f"{a}_X5": list(range(N_PROC)) for a in BRIDGE_ARMS})
    missing = {k: sorted(set(w) - set(ok_ids[k])) for k, w in want.items() if ok_ids[k] != w}
    bad_status = [r.tag for r in X.itertuples() if r.status != "ok"]
    bad_rows = [f"{r.tag}:{r.n_bad_rows}" for r in X.itertuples() if r.n_bad_rows]
    bad_status_x5 = [r.tag for r in X5r.itertuples() if r.status != "ok"]
    main_tags = {r.tag for r in X.itertuples() if r.policy in MAIN_ARMS}
    main_bad = sorted({t for t in bad_status} & main_tags) + sorted({s.split(":")[0] for s in bad_rows} & main_tags)
    main_missing = {k: m for k, m in missing.items() if k in MAIN_ARMS}
    other_missing = {k: m for k, m in missing.items() if k not in MAIN_ARMS}
    bridge_bad = sorted(({t for t in bad_status} | {s.split(":")[0] for s in bad_rows}) - main_tags)
    v["V1"] = dict(passed=bool(not missing and not bad_status and not bad_rows and not bad_status_x5), ok_proc_ids=ok_ids, missing=missing, bad_status=bad_status, rows_not_ok=bad_rows, x5_bad_status=bad_status_x5,
                   unfinished_main=bool(main_missing or main_bad), main_missing=main_missing, main_bad=main_bad, other_missing_or_bad=dict(missing=other_missing, bad=bridge_bad),
                   note="UNFINISHED = a B0 / B1-AOT process missing or not clean (card sec 4); a missing / not-ok bridge or X5 process -> INVALID (literal reading of '8 processes ok')")
    bad_pin = [r.tag for r in X.itertuples() if (r.pinned not in (0, "0")) or (r.pinned_file not in (None, 0))]
    v["V2"] = dict(passed=not bad_pin, pinned_nonzero=bad_pin)
    v["V3"] = dict(passed=bool(T.is_ro(pred) and pi.get("complete")), pred_frozen_0444=bool(T.is_ro(pred)), pred_has_four_arms=bool(pi.get("complete")))
    flagged = {}
    for r in X.itertuples():
        why = list(nq9.get(r.tag, []))
        for k in ("load1_start", "load1_end"):
            x = getattr(r, k)
            if x is not None and pd.notna(x) and float(x) > LOAD_MAX:
                why.append(f"meta {k}={float(x):.1f} > {LOAD_MAX:.0f}")
        if why:
            flagged[r.tag] = why
    flagged_x5 = {}
    for r in X5r.itertuples():
        why = list(nq5.get(r.tag, []))
        for k in ("load1_start", "load1_end"):
            x = getattr(r, k)
            if x is not None and pd.notna(x) and float(x) > LOAD_MAX:
                why.append(f"meta {k}={float(x):.1f} > {LOAD_MAX:.0f}")
        if why:
            flagged_x5[r.tag] = why
    v["V4_not_quiet"] = dict(flagged=sorted(flagged), lines=flagged, x5_side_flagged_info_only=dict(sorted(flagged_x5.items())),
                             note="a flag alone does not void the judgement (verdict WITH flagged processes + excluding_flagged variant); X5-side flags are the already disclosed ones (NOTE_not_quiet_attribution_20261121.txt), listed only")
    v["unfinished"] = bool(v["V1"]["unfinished_main"])
    v["all"] = bool(all(v[k]["passed"] for k in ("V1", "V2", "V3")))
    return v


def analyze(raw: pathlib.Path, log_dir: pathlib.Path, x5_raw: pathlib.Path, x5_log_dir: pathlib.Path, pred: pathlib.Path, out: pathlib.Path, synthetic: bool) -> dict:
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    P9 = load_procs(raw, log_dir, "X9")
    P5 = load_procs(x5_raw, x5_log_dir, "X5")
    assert len(P5), f"no X5 *.meta.json in {x5_raw}"
    P = pd.concat([P9, P5], ignore_index=True) if len(P9) else P5
    if not len(P9):
        sys.exit(f"no *.meta.json in {raw}")
    pi = pred_info(pred)
    V = validity(P, pred, pi, not_quiet_tags(log_dir, "X9"), not_quiet_tags(x5_log_dir, "X5"))
    flagged = set(V["V4_not_quiet"]["flagged"])
    full = judge(P, set(), pi)
    res = dict(validity=V, verdict_all_processes=full, prediction=pi, time=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=bool(synthetic), script_sha16=T.sha16(pathlib.Path(__file__).resolve()),
               raw=str(raw), x5_raw=str(x5_raw), n_processes_x9=int(len(P9)), n_processes_x5=int(len(P5)))
    if flagged:
        res["verdict_excluding_flagged"] = judge(P, flagged, pi)
        res["excluded_processes"] = sorted(flagged)
    if V["unfinished"]:
        status = "UNFINISHED"
    elif not V["all"] or "error" in full:
        status = "INVALID"
    else:
        status = "JUDGED"
    res["status"] = status
    if status == "JUDGED":
        res["conclusion"] = dict(H1=full["H1"]["label"], H2=full["H2"]["label"], H3=full["H3"]["label"], bridge=full["bridge"]["status"], ratio_B1X5_over_B0=full["H1"]["ratio_B1X5_over_B0"],
                                 ratio_OursX5_over_B0=full["H2"]["ratio_OursX5_over_B0"], ratio_AOT_over_B0=full["H3"]["ratio_AOT_over_B0"], H3_paired=full["H3"]["paired_ratios_by_p"],
                                 substituted_version_attached=full["bridge"]["substituted_version"] if full["bridge"]["status"] == "DRIFT" else None)
    else:
        res["conclusion"] = f"NOT JUDGED ({status}): a validity prerequisite failed (see validity); numbers are reported as such, not as H1 / H2 / H3 verdicts"
    (out / "h123_verdict.json").write_text(json.dumps(A._clean(res), indent=1, ensure_ascii=False, default=str))
    P.to_csv(out / "h123_processes.csv", index=False)
    if not synthetic:
        for f in out.rglob("*"):
            if f.is_file():
                os.chmod(f, 0o444)
    return res


def summary_lines(res: dict) -> List[str]:
    V, F = res["validity"], res["verdict_all_processes"]
    lb = (lambda x: x) if res["status"] == "JUDGED" else (lambda x: f"(not judged:{x})")   # numbers of a non-JUDGED status are reports, not verdicts
    L = [f"STATUS={res['status']} V1={V['V1']['passed']} V2={V['V2']['passed']} V3={V['V3']['passed']} NOT_QUIET flagged={V['V4_not_quiet']['flagged']}"]
    if "H1" in F:
        pe = F["prediction_error_reported"]
        T_ = F["T_mean"]
        L.append(f"T_B0={T_['B0']:.1f} T_B1-AOT={T_['B1-AOT']:.1f} T_B1(X5)={T_['B1_X5']:.1f} T_Ours(X5)={T_['Ours_X5']:.1f}" + (f" bridge B1={T_['B1_bridge']:.1f} Ours={T_['Ours_bridge']:.1f}" if "B1_bridge" in T_ and "Ours_bridge" in T_ else ""))
        L.append(f"H1 B1/B0={F['H1']['ratio_B1X5_over_B0']:.4f} (<=0.95) {lb(F['H1']['label'])} ci95={['%.4f' % x for x in F['H1']['ci95']]} | H2 Ours/B0={F['H2']['ratio_OursX5_over_B0']:.4f} (<=0.98) {lb(F['H2']['label'])} ci95={['%.4f' % x for x in F['H2']['ci95']]}")
        L.append(f"H3 AOT/B0={F['H3']['ratio_AOT_over_B0']:.4f} (<=0.90) {lb(F['H3']['label'])} paired={['%.4f' % x for x in F['H3']['paired_ratios_by_p'].values()]} ci95={['%.4f' % x for x in F['H3']['ci95']]}")
        b = F["bridge"]
        L.append(f"bridge={b['status']} " + " ".join(f"{a}: {x['rel_diff_to_X5_mean']:+.4f}" for a, x in b["per_arm"].items() if x["rel_diff_to_X5_mean"] is not None)
                 + (f" | substituted: {json.dumps({k: v_.get('label') for k, v_ in b['substituted_version'].items()}, ensure_ascii=False)}" if b["status"] == "DRIFT" else ""))
        L.append("pred error (reported) " + ", ".join(f"{a} {pe[a]['signed_pred_minus_real_over_real']:+.3f}{'(>10%!)' if pe[a]['flag_gt_10pct'] else ''}" for a in pe)
                 + f" | AOT/B1(X5)={F['reported']['AOT_over_B1_X5']['ratio']:.4f} AOT/Ours(X5)={F['reported']['AOT_over_Ours_X5']['ratio']:.4f}")
    return L


# ------------------------------------------------------------------------------------------------ synthetic self-test (real X5 raw + invented B0 / B1-AOT / bridge)
JIT = (1.01, 0.99, 1.00)


def _write_proc(raw: pathlib.Path, logs: pathlib.Path, pol: str, p: int, t: float, *, pinned=0, bad_row=False, load_end=None, nvml=None):
    with open(raw / f"{pol}_{STREAM}_p{p}.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["idx", "H", "W", "B", "cfg", "event", "status", "t_start_s", "t_end_s", "phase"])
        for i in range(5):
            cw.writerow([i, 512, 512, 1, 0, "compile" if i == 0 else "steady", "error" if (bad_row and i == 2) else "ok", i * 0.1, i * 0.1 + (0.8 if i == 0 else 0.1), "req"])
    ec = {"B0": {"eager_first": 32, "steady": 9968}, "B1": {"compile": 32, "steady": 9968}, "B1-AOT": {"compile": 32, "steady": 9968}, "Ours": {"capture": 6, "capture_denied": 56, "compile": 14, "compile_capture": 12, "eager_first": 28, "steady": 9884}}[pol]
    meta = dict(trace=f"x/{STREAM}.jsonl", policy=pol, status="ok", service_s=t, pinned_fallback_count=pinned, event_counts=ec, peak_reserved_mb=(20540.0 if pol == "Ours" else 8700.0), peak_graph_mb=(6408.0 if pol == "Ours" else 0.0),
                n_evicted_total=(6 if pol == "Ours" else 0), n_denied=(56 if pol == "Ours" else 0), evict_log=[{}] * (6 if pol == "Ours" else 0), runtime_evict_log=[{}] * (1 if pol == "Ours" else 0),
                host_cols=dict(load1_start=3.0, load1_end=(load_end if load_end is not None else 3.5)))
    if pol == "B1-AOT":
        meta.update(aot_fallback_requests=337, aot_fallback_shapes=["512x512x3x0"], aot=dict(dynamo_compile_events=0, rt_cache_files_after=0, pkg_manifest_mismatch=[], pkg_manifest="x"))
    (raw / f"{pol}_{STREAM}_p{p}.meta.json").write_text(json.dumps(meta))
    tag = f"{pol}_W2_warm_p{p}"
    (logs / f"{tag}.pinned_fallback_count.txt").write_text(f"{pinned}\n")
    if nvml:
        (logs / f"{tag}.nvml.csv").write_text("t_unix,used_mb,procs\n0,500,\n1,%s,\n2,600,\n" % nvml)


def _write_synth(root: pathlib.Path, *, b0=(2140.0 * 1.01, 2140.0 * 0.99, 2140.0), aot=(1800.0 * 1.01, 1800.0 * 0.99, 1800.0), br_b1=1935.0, br_ou=2020.0, drop=(), pinned=None, bad_row=None, not_quiet_tag=None, load_end=None, nvml_tag=None):
    raw, logs = root / "raw", root / "logs"
    for d in (raw, logs):
        d.mkdir(parents=True)
    plan = [("B0", 0, b0[0]), ("B1-AOT", 0, aot[0]), ("B1", 3, br_b1), ("B1-AOT", 1, aot[1]), ("B0", 1, b0[1]), ("Ours", 3, br_ou), ("B0", 2, b0[2]), ("B1-AOT", 2, aot[2])]
    lines = []
    for i, (pol, p, t) in enumerate(plan, 1):
        if (pol, p) in drop:
            continue
        tag = f"{pol}_W2_warm_p{p}"
        _write_proc(raw, logs, pol, p, t, pinned=(1 if pinned == (pol, p) else 0), bad_row=(bad_row == (pol, p)), load_end=((load_end or {}).get(tag)), nvml=(9000 if nvml_tag == tag else None))
        if not_quiet_tag == tag:
            lines.append("NOT_QUIET load1=17")
        lines.append(f"STEP [{i}/8] {tag} rc=0 elapsed={i * 100}s")
    (logs / "block_X9_ALL.log").write_text("\n".join(lines) + "\n")
    return raw, logs


def selftest(synth_dir: pathlib.Path) -> int:
    assert not T.inside(synth_dir, ROOT / "results"), "SYNTHETIC outputs must not be written under results/"
    assert T.inside(synth_dir, ROOT / "tmp"), "SYNTHETIC outputs go under tmp/"
    pred = X5 / "pred_X5.csv"
    assert pred.is_file() and T.is_ro(pred), "frozen X5 prediction missing / not 0444"
    x5_raw, x5_logs = X5 / "real/raw", X5 / "real/logs"
    x5 = load_procs(x5_raw, x5_logs, "X5")
    x5b1, x5ou = float(x5[(x5.policy == "B1")].service_s.mean()), float(x5[(x5.policy == "Ours")].service_s.mean())
    print(f"X5 real: T_B1={x5b1:.1f}  T_Ours={x5ou:.1f}  (read-only); frozen prediction {pred.name}; synthetic dir {synth_dir}")
    if synth_dir.exists():
        assert synth_dir.name == "x9_selftest"
        shutil.rmtree(synth_dir)
    synth_dir.mkdir(parents=True)
    b = lambda base: tuple(base * j for j in JIT)
    cases = [
        ("01_holds_all", dict(b0=b(2140.0), aot=b(1800.0)), dict(status="JUDGED", H1="holds", H2="holds", H3="holds", bridge="comparable across days")),
        ("02_all_fail", dict(b0=b(2000.0), aot=b(1850.0), br_b1=1932.0, br_ou=2018.0), dict(status="JUDGED", H1="does not hold", H2="does not hold", H3="does not hold", bridge="comparable across days")),
        ("03_H1_ok_H2_fails_at_line (B0 2040: 0.9466 / 0.9888)", dict(b0=b(2040.0), aot=b(1800.0)), dict(status="JUDGED", H1="holds", H2="does not hold", H3="holds")),
        ("04_H3_partial (means 0.87, one pair 0.907)", dict(b0=(2100.0, 2150.0, 2120.0), aot=(1800.0, 1950.0, 1790.0)), dict(status="JUDGED", H3="partially holds")),
        ("05_H3_mean_just_fails (AOT/B0 0.905)", dict(b0=b(2000.0), aot=b(1810.0)), dict(status="JUDGED", H3="does not hold")),
        ("06_DRIFT_both (bridge B1 +5%, Ours +3%)", dict(b0=b(2140.0), aot=b(1800.0), br_b1=x5b1 * 1.05, br_ou=x5ou * 1.03), dict(status="JUDGED", bridge="DRIFT", subst=True)),
        ("07_DRIFT_one_arm (Ours bridge +2.5%)", dict(b0=b(2140.0), aot=b(1800.0), br_b1=x5b1 * 1.005, br_ou=x5ou * 1.025), dict(status="JUDGED", bridge="DRIFT")),
        ("08_bridge_just_inside_2pct (B1 +1.9%, Ours -1.9%)", dict(b0=b(2140.0), aot=b(1800.0), br_b1=x5b1 * 1.019, br_ou=x5ou * 0.981), dict(status="JUDGED", bridge="comparable across days")),
        ("09_UNFINISHED_K3_not_run (B0 p2, B1-AOT p2 missing)", dict(drop=(("B0", 2), ("B1-AOT", 2))), dict(status="UNFINISHED")),
        ("10_UNFINISHED_only_AOT_p2_missing", dict(drop=(("B1-AOT", 2),)), dict(status="UNFINISHED")),
        ("11_UNFINISHED_request_row_not_ok_in_B0_p1", dict(bad_row=("B0", 1)), dict(status="UNFINISHED")),
        ("12_INVALID_pinned_fallback_on_B0_p0 (all 8 present)", dict(pinned=("B0", 0)), dict(status="INVALID")),
        ("13_INVALID_bridge_Ours_missing (B0 / AOT complete; literal reading)", dict(drop=(("Ours", 3),)), dict(status="INVALID")),
        ("14_INVALID_request_row_not_ok_in_bridge_B1", dict(bad_row=("B1", 3)), dict(status="INVALID")),
        ("15_NOT_QUIET_gate_line_on_AOT_p1 -> flag + excluding variant", dict(not_quiet_tag="B1-AOT_W2_warm_p1"), dict(status="JUDGED", H3="holds", flagged=True, excl_n_aot=2)),
        ("16_NOT_QUIET_meta_load1_end_20_on_B0_p2", dict(load_end={"B0_W2_warm_p2": 20.0}), dict(status="JUDGED", flagged=True, excl_n_b0=2)),
        ("17_pred_error_flag (B0 real ~2450 vs pred 2133 = 13 %)", dict(b0=b(2450.0), aot=b(1800.0)), dict(status="JUDGED", flag_B0=True)),
    ]
    bad = 0
    for name, kw, exp in cases:
        cd = synth_dir / name.split(" ")[0]
        raw, logs = _write_synth(cd, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            res = analyze(raw, logs, x5_raw, x5_logs, pred, cd / "out", synthetic=True)
        F = res["verdict_all_processes"]
        got = dict(status=res["status"])
        if "H1" in F:
            got.update(H1=F["H1"]["label"], H2=F["H2"]["label"], H3=F["H3"]["label"], bridge=F["bridge"]["status"])
            got["subst"] = bool(F["bridge"]["substituted_version"].get("H1_bridge") and F["bridge"]["substituted_version"].get("H2_bridge"))
            got["flag_B0"] = bool(F["prediction_error_reported"]["B0"]["flag_gt_10pct"])
        if exp.get("flagged"):
            got["flagged"] = bool(res["validity"]["V4_not_quiet"]["flagged"]) and "verdict_excluding_flagged" in res
            ex = res.get("verdict_excluding_flagged", {})
            got["excl_n_aot"] = ex.get("n_ok", {}).get("B1-AOT")
            got["excl_n_b0"] = ex.get("n_ok", {}).get("B0")
        ok = all(got.get(k) == v for k, v in exp.items())
        bad += (not ok)
        print(("PASS " if ok else "FAIL ") + name + f"  got={{{', '.join(f'{k}: {got[k]}' for k in exp)}}}")
    # ---- numeric checks on case 01 (hand calculation from the invented values and the REAL X5 means)
    cd = synth_dir / "01_holds_all"
    res = json.loads((cd / "out/h123_verdict.json").read_text())
    F = res["verdict_all_processes"]
    t_b0, t_aot = 2140.0, 1800.0
    pr = pd.read_csv(pred)
    tp = {r.policy: float(r.pred_total_s) for r in pr.itertuples()}
    chk = [abs(F["H1"]["ratio_B1X5_over_B0"] - x5b1 / t_b0) < 1e-9, abs(F["H2"]["ratio_OursX5_over_B0"] - x5ou / t_b0) < 1e-9, abs(F["H3"]["ratio_AOT_over_B0"] - t_aot / t_b0) < 1e-9,
           all(abs(x - t_aot / t_b0) < 1e-9 for x in F["H3"]["paired_ratios_by_p"].values()) and len(F["H3"]["paired_ratios_by_p"]) == 3,
           abs(F["bridge"]["per_arm"]["B1"]["rel_diff_to_X5_mean"] - (1935.0 / x5b1 - 1)) < 1e-9, abs(F["bridge"]["per_arm"]["Ours"]["rel_diff_to_X5_mean"] - (2020.0 / x5ou - 1)) < 1e-9,
           abs(F["prediction_error_reported"]["B0"]["abs_error"] - abs(t_b0 - tp["B0"]) / t_b0) < 1e-9, abs(F["prediction_error_reported"]["B1-AOT"]["signed_pred_minus_real_over_real"] - (tp["B1-AOT"] - t_aot) / t_aot) < 1e-9,
           abs(F["reported"]["AOT_over_B1_X5"]["ratio"] - t_aot / x5b1) < 1e-9, abs(F["reported"]["AOT_over_Ours_X5"]["ratio"] - t_aot / x5ou) < 1e-9,
           abs(F["H1"]["equivalent_T_B0_min"] - x5b1 / 0.95) < 1e-9, len(F["H1"]["ci95"]) == 2 and F["H1"]["ci95"][0] < F["H1"]["ci95"][1],
           F["groups"]["Ours_X5"]["n_evicted_total"]["0"] == 6,
           F["groups"]["B1-AOT"]["aot_dynamo_compile_events"]["0"] == 0, F["groups"]["B1-AOT"]["n_compile_mean"] == 32.0, abs(F["groups"]["B1-AOT"]["first_request_s_mean"] - 0.8) < 1e-9,
           F["groups"]["Ours_X5"]["n"] == 3 and F["groups"]["B1_X5"]["n"] == 3 and F["groups"]["B0"]["n"] == 3 and F["groups"]["B1_bridge"]["n"] == 1,
           res["status"] == "JUDGED" and res["synthetic"] is True]
    ok = all(chk)
    bad += (not ok)
    print(("PASS " if ok else "FAIL ") + f"numeric checks case 01 (ratios, paired, bridge diffs, pred error, AOT/B1, AOT/Ours, equivalent T_B0, groups) {chk}")
    # ---- DRIFT substituted version numbers (case 06)
    res = json.loads((synth_dir / "06_DRIFT_both/out/h123_verdict.json").read_text())
    F = res["verdict_all_processes"]
    s = F["bridge"]["substituted_version"]
    chk = [abs(s["H1_bridge"]["ratio_B1bridge_over_B0"] - x5b1 * 1.05 / 2140.0) < 1e-9, abs(s["H2_bridge"]["ratio_OursBridge_over_B0"] - x5ou * 1.03 / 2140.0) < 1e-9, F["bridge"]["attached_by_card"] is True,
           res["conclusion"]["substituted_version_attached"] is not None]
    ok = all(chk)
    bad += (not ok)
    print(("PASS " if ok else "FAIL ") + f"DRIFT substituted-version numbers (case 06) {chk}")
    # ---- NVML peak read from <tag>.nvml.csv (opt-in sampler output) and the official-output guard
    cd = synth_dir / "18_nvml"
    raw, logs = _write_synth(cd, nvml_tag="B0_W2_warm_p0")
    with contextlib.redirect_stdout(io.StringIO()):
        res = analyze(raw, logs, x5_raw, x5_logs, pred, cd / "out", synthetic=True)
    nv = res["verdict_all_processes"]["groups"]["B0"]["nvml_peak_mb_max"]
    ok = bool(nv == 9000.0 and res["verdict_all_processes"]["groups"]["B1-AOT"]["nvml_peak_mb_max"] is None)
    bad += (not ok)
    print(("PASS " if ok else "FAIL ") + f"NVML peak from <tag>.nvml.csv (B0 {nv}; arms without a csv -> null)")
    cd = synth_dir / "01_holds_all"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            analyze(cd / "raw", cd / "logs", x5_raw, x5_logs, pred, cd / "out", synthetic=True)
        ok = False
    except SystemExit:
        ok = True
    bad += (not ok)
    print(("PASS " if ok else "FAIL ") + "analyze refuses to write into a non-empty output dir")
    print("SELFTEST", "FAIL" if bad else "PASS", f"({len(cases) + 4} checks; case outputs in {synth_dir})")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(X9 / "real/raw"))
    ap.add_argument("--log-dir", default=str(X9 / "real/logs"))
    ap.add_argument("--x5-raw", default=str(X5 / "real/raw"))
    ap.add_argument("--x5-log-dir", default=str(X5 / "real/logs"))
    ap.add_argument("--pred", default=str(X5 / "pred_X5.csv"))
    ap.add_argument("--out-dir", default=str(X9 / "verdict"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-dir", default=str(ROOT / "tmp/x9_selftest"))
    a = ap.parse_args(argv)
    if a.synthetic:
        return selftest(pathlib.Path(a.synthetic_dir))
    out = pathlib.Path(a.out_dir)
    check_prereg()
    assert T.inside(out, X9), f"official output must be under {X9}"
    assert T.is_ro(a.pred), f"{a.pred} (frozen prediction) must be 0444"
    res = analyze(pathlib.Path(a.raw), pathlib.Path(a.log_dir), pathlib.Path(a.x5_raw), pathlib.Path(a.x5_log_dir), pathlib.Path(a.pred), out, False)
    print("\n".join(summary_lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
