"""S4-X X8 (CPU): verdict H1 / H2 + validity for the cold-start (EMPTY compile cache) real-machine replay (W2, N=1000, seed 135; arms B0, B1, Ours; 3 processes each).
Rules = goals/S4X_X8_cold_start_replay_prereg_v1.md sec 2/3 (written BEFORE any X8 number).
  T = process total service time (meta `service_s`); per arm the mean of the 3 repeats (p0-p2).
  H1 (cold start does not hurt Ours)        T_Ours / T_B0 <= 1.05
  H2 (compile-on-first-sight loses badly)   T_B1 / T_B0 >= 2.0
  holds    = H1 and H2 on the arm means AND all three p-paired ratios (Ours_p/B0_p <= 1.05, B1_p/B0_p >= 2.0) satisfy the same lines
  partial  = H1 and H2 hold on the means but not every paired ratio does (reported as such)
  H1_fails / H2_fails / both_fail = the corresponding mean-ratio line is missed
  Reported only (not judged): per-arm prediction error (flag > 15 %, same as the --quick C2 line), Ours/B1, first-request latency, compile / capture counts, peak reserved memory, FX-cache hit/miss,
  comparison with disk-cache-tier arms (P5d W2 N1000 seed 130, X3 W2 N1000 seed 131 [B0], X5 W2 N10000 seed 133 [B1, Ours]; N differs for X5).
Validity (card sec 3; if one fails H1/H2 are NOT judged and the failure is reported as such):
  V1 each arm has exactly 3 ok processes (p0, p1, p2) and no request row with status != ok     -> missing / not ok processes: UNFINISHED (no judgement; numbers of the finished ones are listed)
  V2 empty-cache evidence per process: <tag>.empty_evidence.json with empty_at_start (cache root had 0 files, expected cache dir absent) AND meta cache_mode == "empty" AND meta cache_dir below the
     evidenced cache root; a process without this record is invalid                              -> INVALID
  V3 pinned_fallback_count == 0 for every process (meta and the pinned_fallback_count.txt)       -> INVALID (the card says "flag and report; a pinned allocation failure stops the block = unfinished")
  V4 NOT_QUIET (gate line before the STEP line, or meta load1_start/load1_end > 16) = FLAGGED: verdict computed WITH the flagged processes (marked) and an `excluding_flagged` variant; a flag alone
     does not void the judgement.  (Cache removal after the process is reported, not a validity item.)
Usage: "$PROJ_PYTHON" -B -m scripts.h12_verdict_S4X_X8 [--raw ../results/S4X/x8/real/raw --log-dir ../results/S4X/x8/real/logs --pred ../results/S4X/x8/pred_X8.csv --out-dir ../results/S4X/x8/verdict]
       "$PROJ_PYTHON" -B -m scripts.h12_verdict_S4X_X8 --selftest      (synthetic data in a temp dir: holds / H1 fails / H2 fails / partial / INVALID (no empty evidence) / NOT_QUIET / unfinished / pinned)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts import a123_verdict_S4P_P5c as A
from scripts import aot_tables_S4P_P5c as T

ROOT = T.ROOT
X8 = ROOT / "results/S4X/x8"
PREREG = ROOT / "goals/S4X_X8_cold_start_replay_prereg_v1.md"
PREREG_SHA_FILE = X8 / "PREREG_SHA16.txt"
STREAM = "W2_N1000_seed135_B"
ARMS = ["B0", "B1", "Ours"]
N_PROC = 3
H1_LINE, H2_LINE, ERR_FLAG = 1.05, 2.0, 0.15
LOAD_MAX = 16.0
REFS = {"P5d_W2_N1000_seed130 (warm disk cache)": (ROOT / "results/S4P/p5d/real/raw", "W2_N1000_seed130_B"),
        "X3_W2_N1000_seed131 (warm disk cache)": (ROOT / "results/S4X/x3/real/raw", "W2_N1000_seed131_B"),
        "X5_W2_N10000_seed133 (warm disk cache, N=10000)": (ROOT / "results/S4X/x5/real/raw", "W2_N10000_seed133_B")}


def check_prereg() -> str:
    assert PREREG.is_file() and T.is_ro(PREREG), f"preregistration missing or not 0444: {PREREG}"
    assert PREREG_SHA_FILE.is_file(), f"{PREREG_SHA_FILE} missing"
    exp, got = PREREG_SHA_FILE.read_text().strip(), T.sha16(PREREG)
    assert got == exp, f"preregistration SHA16 {got} != recorded {exp}"
    return got


def not_quiet_tags(log_dir: pathlib.Path) -> Dict[str, List[str]]:
    """tag -> NOT_QUIET lines printed (by the gate, while waiting) before that process's STEP line (block_X8_*.log of window_S4X_X8.sh)."""
    out: Dict[str, List[str]] = {}
    pend: List[str] = []
    for f in sorted(glob.glob(str(log_dir / "block_X8_*.log"))):
        for l in pathlib.Path(f).read_text(errors="replace").splitlines():
            if "NOT_QUIET" in l:
                pend.append(l.strip())
            m = re.match(r"^STEP (?:\[\d+/\d+\] )?(\S+) rc=", l)
            if m:
                if pend:
                    out.setdefault(m.group(1), []).extend(pend)
                pend = []
    return out


def load_procs(raw: pathlib.Path, log_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.meta.json")):
        m = json.loads(f.read_text())
        stem = pathlib.Path(m["trace"]).stem
        pol = A.norm(m["policy"])
        proc = int(f.name.rsplit("_p", 1)[1].split(".")[0])
        tag = f"{m['policy']}_W2_empty_p{proc}"
        csvp = f.with_name(f.name.replace(".meta.json", ".csv"))
        ev = {}
        evp = log_dir / f"{tag}.empty_evidence.json"
        if evp.exists():
            ev = json.loads(evp.read_text())
        pcf = log_dir / f"{tag}.pinned_fallback_count.txt"
        first_s = None
        n_bad = 0
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
        dg = m.get("diag") or {}
        rows.append(dict(stream=stem, policy=pol, proc=proc, tag=tag, status=m.get("status"), service_s=m.get("service_s"), pinned=m.get("pinned_fallback_count"),
                         pinned_file=(int(pcf.read_text().strip()) if pcf.exists() and pcf.read_text().strip().isdigit() else None),
                         cache_mode=m.get("cache_mode"), cache_dir=m.get("cache_dir"), evidence_present=bool(ev), empty_at_start=bool(ev.get("empty_at_start", False)),
                         evid_root=ev.get("cache_root"), evid_files_at_start=ev.get("files_in_cache_root_at_start"), evid_cdir_removed=ev.get("expected_cache_dir_removed_by_process"),
                         evid_root_exists_after=ev.get("cache_root_exists_after_process"), evid_max_kb=ev.get("max_kb_sampled"), evid_first_sample_files=ev.get("first_sample_files"),
                         first_req_s=first_s, n_compile=int(ec.get("compile", 0)) + int(ec.get("compile_capture", 0)), n_capture=int(ec.get("capture", 0)) + int(ec.get("compile_capture", 0)),
                         n_eager_first=int(ec.get("eager_first", 0)), event_counts=json.dumps(ec, sort_keys=True), peak_reserved_mb=m.get("peak_reserved_mb"),
                         fx_hit=dg.get("fx_cache_hit"), fx_miss=dg.get("fx_cache_miss"), load1_start=hc.get("load1_start"), load1_end=hc.get("load1_end"), load_s=m.get("load_s"),
                         n_bad_rows=n_bad, csv=str(csvp) if csvp.exists() else "", file=f.name))
    return pd.DataFrame(rows)


def pred_info(pred: pathlib.Path) -> dict:
    d = pd.read_csv(pred)
    d["policy"] = d.policy.map(A.norm)
    d = d[d.stream == STREAM]
    out = dict(T_pred={r.policy: float(r.pred_total_s) for r in d.itertuples()},
               first_req_pred={r.policy: (float(r.pred_first_req_s) if "pred_first_req_s" in d and pd.notna(r.pred_first_req_s) else None) for r in d.itertuples()},
               n_compile_pred={r.policy: (float(r.pred_n_compile) if "pred_n_compile" in d else None) for r in d.itertuples()},
               n_capture_pred={r.policy: (float(r.pred_n_capture) if "pred_n_capture" in d else None) for r in d.itertuples()},
               event_counts_pred={r.policy: (r.event_counts if "event_counts" in d else None) for r in d.itertuples()})
    out["complete"] = all(a in out["T_pred"] for a in ARMS)
    return out


def judge(P: pd.DataFrame, drop: set, pi: dict) -> dict:
    by = {a: P[(P.stream == STREAM) & (P.policy == a) & (P.status == "ok") & (~P.tag.isin(drop))].sort_values("proc").set_index("proc") for a in ARMS}
    res = dict(n_ok={a: int(len(by[a])) for a in ARMS})
    if any(len(by[a]) == 0 for a in ARMS):
        res["error"] = "an arm has no ok process"
        return res
    v = {a: by[a].service_s.to_numpy(float) for a in ARMS}
    Tm = {a: float(v[a].mean()) for a in ARMS}
    r1, r2 = Tm["Ours"] / Tm["B0"], Tm["B1"] / Tm["B0"]
    common = sorted(set(by["B0"].index) & set(by["Ours"].index) & set(by["B1"].index))
    paired = {int(p): dict(Ours_over_B0=float(by["Ours"].service_s[p] / by["B0"].service_s[p]), B1_over_B0=float(by["B1"].service_s[p] / by["B0"].service_s[p])) for p in common}
    h1, h2 = bool(r1 <= H1_LINE), bool(r2 >= H2_LINE)
    p1 = bool(len(paired) > 0 and all(x["Ours_over_B0"] <= H1_LINE for x in paired.values()))
    p2 = bool(len(paired) > 0 and all(x["B1_over_B0"] >= H2_LINE for x in paired.values()))
    if h1 and h2 and p1 and p2:
        label = "holds"
    elif h1 and h2:
        label = "partial"
    elif not h1 and not h2:
        label = "both_fail"
    elif not h1:
        label = "H1_fails"
    else:
        label = "H2_fails"
    ci, _ = A.boot_ratios([v["Ours"], v["B1"]], [v["B0"], v["B0"]])
    res.update(T_mean=Tm, T_median={a: float(np.median(v[a])) for a in ARMS}, T_values={a: [float(x) for x in v[a]] for a in ARMS},
               H1=dict(passed=h1, ratio_Ours_over_B0=float(r1), line=f"<= {H1_LINE}", ci95=ci[0]), H2=dict(passed=h2, ratio_B1_over_B0=float(r2), line=f">= {H2_LINE}", ci95=ci[1]),
               paired_ratios_by_p=paired, paired_H1_all=p1, paired_H2_all=p2, label=label, holds=bool(label == "holds"),
               reported=dict(Ours_over_B1_real=float(Tm["Ours"] / Tm["B1"]), Ours_over_B1_pred=float(pi["T_pred"]["Ours"] / pi["T_pred"]["B1"]) if pi.get("complete") else None))
    errs = {}
    for a in ARMS:
        if a in pi["T_pred"]:
            e = (pi["T_pred"][a] - Tm[a]) / Tm[a]
            errs[a] = dict(T_real_mean=Tm[a], T_pred=pi["T_pred"][a], signed_pred_minus_real_over_real=float(e), abs_error=float(abs(e)), flag_gt_15pct=bool(abs(e) > ERR_FLAG))
    res["prediction_error_reported"] = errs
    if pi.get("complete"):
        res["reported"].update(r1_pred=float(pi["T_pred"]["Ours"] / pi["T_pred"]["B0"]), r2_pred=float(pi["T_pred"]["B1"] / pi["T_pred"]["B0"]))
    res["first_request_s_mean"] = {a: (float(by[a].first_req_s.dropna().mean()) if by[a].first_req_s.notna().any() else None) for a in ARMS}
    res["first_request_s_pred"] = pi.get("first_req_pred")
    res["n_compile_mean"] = {a: float(by[a].n_compile.mean()) for a in ARMS}
    res["n_capture_mean"] = {a: float(by[a].n_capture.mean()) for a in ARMS}
    res["n_compile_pred"] = pi.get("n_compile_pred")
    res["n_capture_pred"] = pi.get("n_capture_pred")
    res["peak_reserved_mb_max"] = {a: (float(by[a].peak_reserved_mb.dropna().max()) if by[a].peak_reserved_mb.notna().any() else None) for a in ARMS}
    return res


def validity(P: pd.DataFrame, nq_gate: Dict[str, List[str]], log_dir: pathlib.Path) -> dict:
    v = {}
    arms = P[P.stream == STREAM]
    ok_ids = {a: sorted(int(p) for p in arms[(arms.policy == a) & (arms.status == "ok")].proc) for a in ARMS}
    bad_status = [r.tag for r in arms.itertuples() if r.status != "ok"]
    bad_rows = [f"{r.tag}:{r.n_bad_rows}" for r in arms.itertuples() if r.n_bad_rows]
    missing = {a: sorted(set(range(N_PROC)) - set(ok_ids[a])) for a in ARMS if ok_ids[a] != list(range(N_PROC))}
    v["V1"] = dict(passed=bool(not missing and not bad_status and not bad_rows), ok_proc_ids=ok_ids, missing=missing, bad_status=bad_status, rows_not_ok=bad_rows)
    bad_ev = []
    for r in arms.itertuples():
        under = bool(r.evid_root and r.cache_dir and str(r.cache_dir).startswith(str(r.evid_root).rstrip("/") + "/"))
        good = bool(r.evidence_present and r.empty_at_start and r.cache_mode == "empty" and under and r.evid_files_at_start == 0)
        if not good:
            bad_ev.append(dict(tag=r.tag, evidence_present=bool(r.evidence_present), empty_at_start=bool(r.empty_at_start), cache_mode=r.cache_mode, cache_dir_under_evidenced_root=under))
    v["V2"] = dict(passed=not bad_ev, no_or_bad_evidence=bad_ev,
                   cache_removed_after={r.tag: dict(expected_dir_removed=r.evid_cdir_removed, root_exists_after=r.evid_root_exists_after, max_kb_sampled=r.evid_max_kb, first_sample_files=r.evid_first_sample_files) for r in arms.itertuples()},
                   note="removal after the process is reported, not a validity item")
    bad_pin = [r.tag for r in arms.itertuples() if (r.pinned not in (0, "0")) or (r.pinned_file not in (None, 0))]
    v["V3"] = dict(passed=not bad_pin, pinned_nonzero=bad_pin)
    flagged = {}
    for r in arms.itertuples():
        why = list(nq_gate.get(r.tag, []))
        for k in ("load1_start", "load1_end"):
            x = getattr(r, k)
            if x is not None and pd.notna(x) and float(x) > LOAD_MAX:
                why.append(f"meta {k}={float(x):.1f} > {LOAD_MAX:.0f}")
        if why:
            flagged[r.tag] = why
    v["V4_not_quiet"] = dict(flagged=sorted(flagged), lines=flagged, note="a flag alone does not void the judgement (verdict WITH flagged processes + excluding_flagged variant)")
    v["unfinished"] = bool(not v["V1"]["passed"])
    v["all"] = bool(all(v[k]["passed"] for k in ("V1", "V2", "V3")))
    return v


def reference_rows() -> dict:
    out = {}
    for name, (d, stem) in REFS.items():
        if not d.is_dir():
            continue
        r = {}
        for f in sorted(d.glob(f"*_{stem}_p*.meta.json")):
            m = json.loads(f.read_text())
            if m.get("status") == "ok":
                r.setdefault(A.norm(m["policy"]), []).append(float(m["service_s"]))
        if r:
            out[name] = {p: dict(mean_service_s=float(np.mean(v)), n=len(v)) for p, v in r.items()}
            if "B0" in r and "Ours" in r:
                out[name]["Ours_over_B0"] = float(np.mean(r["Ours"]) / np.mean(r["B0"]))
            if "B1" in r and "Ours" in r:
                out[name]["Ours_over_B1"] = float(np.mean(r["Ours"]) / np.mean(r["B1"]))
            if "B1" in r and "B0" in r:
                out[name]["B1_over_B0"] = float(np.mean(r["B1"]) / np.mean(r["B0"]))
    return out


def analyze(raw: pathlib.Path, log_dir: pathlib.Path, pred: pathlib.Path, out: pathlib.Path, synthetic: bool) -> dict:
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    P = load_procs(raw, log_dir)
    assert len(P), f"no *.meta.json in {raw}"
    pi = pred_info(pred)
    V = validity(P, not_quiet_tags(log_dir), log_dir)
    flagged = set(V["V4_not_quiet"]["flagged"])
    full = judge(P, set(), pi)
    res = dict(validity=V, verdict_all_processes=full, prediction=pi, time=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=bool(synthetic), script_sha16=T.sha16(pathlib.Path(__file__).resolve()),
               raw=str(raw), n_processes=int(len(P)), disk_cache_reference_not_judged=reference_rows() if not synthetic else {})
    if flagged:
        res["verdict_excluding_flagged"] = judge(P, flagged, pi)
        res["excluded_processes"] = sorted(flagged)
    if V["unfinished"]:
        status = "UNFINISHED"
    elif not V["all"]:
        status = "INVALID"
    else:
        status = "JUDGED"
    res["status"] = status
    res["conclusion"] = (dict(label=full["label"], H1=full["H1"]["passed"], H2=full["H2"]["passed"], r_Ours_over_B0=full["H1"]["ratio_Ours_over_B0"], r_B1_over_B0=full["H2"]["ratio_B1_over_B0"],
                              paired_H1_all=full["paired_H1_all"], paired_H2_all=full["paired_H2_all"], holds=full["holds"]) if status == "JUDGED"
                         else f"NOT JUDGED ({status}): a validity prerequisite failed (see validity); numbers are reported as such, not as H1/H2 verdicts")
    (out / "h12_verdict.json").write_text(json.dumps(A._clean(res), indent=1, ensure_ascii=False, default=str))
    P.to_csv(out / "h12_processes.csv", index=False)
    if not synthetic:
        for f in out.rglob("*"):
            if f.is_file():
                os.chmod(f, 0o444)
    return res


def summary_lines(res: dict) -> List[str]:
    V, F = res["validity"], res["verdict_all_processes"]
    L = [f"STATUS={res['status']} V1={V['V1']['passed']} V2={V['V2']['passed']} V3={V['V3']['passed']} NOT_QUIET flagged={V['V4_not_quiet']['flagged']}"]
    if "H1" in F:
        pe = F["prediction_error_reported"]
        L.append(f"T_B0={F['T_mean']['B0']:.1f} T_B1={F['T_mean']['B1']:.1f} T_Ours={F['T_mean']['Ours']:.1f} | Ours/B0={F['H1']['ratio_Ours_over_B0']:.4f} (H1 <=1.05: {F['H1']['passed']}) "
                 f"B1/B0={F['H2']['ratio_B1_over_B0']:.4f} (H2 >=2.0: {F['H2']['passed']}) | paired={ {p: (round(x['Ours_over_B0'], 3), round(x['B1_over_B0'], 3)) for p, x in F['paired_ratios_by_p'].items()} }")
        L.append(f"label={F['label']} holds={F['holds']} | pred error " + ", ".join(f"{a} {pe[a]['signed_pred_minus_real_over_real']:+.3f}{'(>15%!)' if pe[a]['flag_gt_15pct'] else ''}" for a in pe))
    return L


# ------------------------------------------------------------------------------------------------ selftest (synthetic)
PRED_T = {"B0": 222.3, "B1": 1070.2, "Ours": 221.6}


def _write_synth(root: pathlib.Path, *, T_real=None, jit=None, pinned=0, drop=None, no_evidence=None, not_quiet_tag=None, bad_cache_mode=False, nonempty_start=False, bad_row=False, load_meta=None):
    raw, logs = root / "raw", root / "logs"
    for d in (raw, logs):
        d.mkdir(parents=True)
    T_real = dict(PRED_T if T_real is None else T_real)
    jit = jit or {0: 1.01, 1: 0.99, 2: 1.00}
    step_lines = []
    for p in range(3):
        for pol in ARMS:
            if drop == (pol, p):
                continue
            t = T_real[pol] * jit[p]
            tag = f"{pol}_W2_empty_p{p}"
            with open(raw / f"{pol}_{STREAM}_p{p}.csv", "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["idx", "H", "W", "B", "cfg", "event", "status", "t_start_s", "t_end_s", "phase"])
                for i in range(5):
                    cw.writerow([i, 512, 512, 1, 0, "compile" if (pol == "B1" and i == 0) else "steady", "error" if (bad_row and pol == "Ours" and p == 1 and i == 2) else "ok", i * 0.1, i * 0.1 + (40.0 if (pol == "B1" and i == 0) else 0.1), "req"])
            cr = f"/x/x8_empty/{pol}_p{p}"
            ec = {"B0": {"eager_first": 31, "steady": 969}, "B1": {"compile": 31, "steady": 969}, "Ours": {"capture": 1, "eager_first": 30, "steady": 969}}[pol]
            meta = dict(trace=f"x/{STREAM}.jsonl", policy=pol, status="ok", service_s=t, pinned_fallback_count=pinned if (pol == "Ours" and p == 2) else 0,
                        cache_mode=("warm" if (bad_cache_mode and pol == "B1" and p == 1) else "empty"), cache_dir=f"{cr}/cold/raw/serve_{pol}_{STREAM}_p{p}", event_counts=ec, peak_reserved_mb=5000.0,
                        diag=dict(fx_cache_hit=0, fx_cache_miss=(31 if pol == "B1" else 0)), host_cols=dict(load1_start=(load_meta if (load_meta and pol == "Ours" and p == 0) else 3.0), load1_end=3.5), load_s=0.5)
            (raw / f"{pol}_{STREAM}_p{p}.meta.json").write_text(json.dumps(meta))
            if no_evidence != (pol, p):
                ev = dict(tag=tag, cache_root=cr, expected_cache_dir=meta["cache_dir"], files_in_cache_root_at_start=(7 if (nonempty_start and pol == "B1" and p == 0) else 0),
                          empty_at_start=not (nonempty_start and pol == "B1" and p == 0), expected_cache_dir_removed_by_process=True, cache_root_exists_after_process=False, max_kb_sampled=(5000 if pol == "B1" else 0), first_sample_files=0)
                (logs / f"{tag}.empty_evidence.json").write_text(json.dumps(ev))
            (logs / f"{tag}.pinned_fallback_count.txt").write_text(str(pinned if (pol == "Ours" and p == 2) else 0) + "\n")
            if not_quiet_tag == tag:
                step_lines.append("NOT_QUIET load1=17")
            step_lines.append(f"STEP [1/9] {tag} rc=0")
    (logs / "block_X8_ALL.log").write_text("\n".join(step_lines) + "\n")
    rows = [dict(stream=STREAM, policy=pol, pred_total_s=v, pred_first_req_s=(41.8 if pol == "B1" else 0.5), pred_n_compile=(31 if pol == "B1" else 0), pred_n_capture=(1 if pol == "Ours" else 0),
                 event_counts="{}") for pol, v in PRED_T.items()]
    pd.DataFrame(rows).to_csv(root / "pred.csv", index=False)
    os.chmod(root / "pred.csv", 0o444)
    return raw, logs, root / "pred.csv"


def selftest() -> int:
    c = lambda b0, b1, o: dict(B0=b0, B1=b1, Ours=o)
    cases = [
        ("holds: Ours/B0 1.003, B1/B0 4.8", dict(T_real=c(222.0, 1075.0, 222.7)), dict(status="JUDGED", label="holds", H1=True, H2=True)),
        ("holds at the lines (Ours/B0 1.049, B1/B0 2.05) with the same jitter", dict(T_real=c(300.0, 615.0, 314.7)), dict(status="JUDGED", label="holds")),
        ("H1 fails (Ours/B0 1.08)", dict(T_real=c(222.0, 1075.0, 239.8)), dict(status="JUDGED", label="H1_fails", H1=False, H2=True)),
        ("H2 fails (B1/B0 1.8)", dict(T_real=c(222.0, 400.0, 222.7)), dict(status="JUDGED", label="H2_fails", H1=True, H2=False)),
        ("both fail", dict(T_real=c(222.0, 400.0, 250.0)), dict(status="JUDGED", label="both_fail", H1=False, H2=False)),
        ("partial: means hold but one p-pair Ours/B0 > 1.05", dict(T_real=c(222.0, 1075.0, 222.7), jit=None, _special="partial"), dict(status="JUDGED", label="partial", H1=True, H2=True)),
        ("INVALID: one process has no empty-cache evidence", dict(T_real=c(222.0, 1075.0, 222.7), no_evidence=("B1", 1)), dict(status="INVALID")),
        ("INVALID: evidence says cache not empty at start", dict(T_real=c(222.0, 1075.0, 222.7), nonempty_start=True), dict(status="INVALID")),
        ("INVALID: meta cache_mode is warm", dict(T_real=c(222.0, 1075.0, 222.7), bad_cache_mode=True), dict(status="INVALID")),
        ("INVALID: pinned fallback != 0", dict(T_real=c(222.0, 1075.0, 222.7), pinned=1), dict(status="INVALID")),
        ("UNFINISHED: a process missing", dict(T_real=c(222.0, 1075.0, 222.7), drop=("Ours", 2)), dict(status="UNFINISHED")),
        ("UNFINISHED: a request row not ok", dict(T_real=c(222.0, 1075.0, 222.7), bad_row=True), dict(status="UNFINISHED")),
        ("NOT_QUIET (gate line): judged WITH flag + excluding variant", dict(T_real=c(222.0, 1075.0, 222.7), not_quiet_tag="Ours_W2_empty_p1"), dict(status="JUDGED", label="holds", flagged=True)),
        ("NOT_QUIET (meta load1_start 18): judged WITH flag + excluding variant", dict(T_real=c(222.0, 1075.0, 222.7), load_meta=18.0), dict(status="JUDGED", label="holds", flagged=True)),
    ]
    bad = 0
    for name, kw, exp in cases:
        kw = dict(kw)
        special = kw.pop("_special", None)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            raw, logs, pred = _write_synth(pathlib.Path(td), **kw)
            if special == "partial":   # Ours p1 is 8 % slower than B0 p1 while the means stay <= 1.05: shift B0 p1 down, Ours p1 up
                for pol, f in (("Ours", 1.0), ("B0", 1.0)):
                    mp = raw / f"{pol}_{STREAM}_p1.meta.json"
                    m = json.loads(mp.read_text())
                    m["service_s"] = m["service_s"] * (1.095 if pol == "Ours" else 0.94)
                    mp.write_text(json.dumps(m))
                    ref = raw / f"{pol}_{STREAM}_p0.meta.json"
                    m0 = json.loads(ref.read_text())
                    m0["service_s"] = m0["service_s"] * (0.95 if pol == "Ours" else 1.0)
                    ref.write_text(json.dumps(m0))
            with contextlib.redirect_stdout(io.StringIO()):
                res = analyze(raw, logs, pred, pathlib.Path(td) / "out", synthetic=True)
        got = dict(status=res["status"])
        vd = res["verdict_all_processes"]
        if "H1" in vd and res["status"] == "JUDGED":
            got.update(label=vd["label"], H1=vd["H1"]["passed"], H2=vd["H2"]["passed"])
        if exp.get("flagged"):
            got["flagged"] = bool(res["validity"]["V4_not_quiet"]["flagged"]) and "verdict_excluding_flagged" in res
        ok = all(got.get(k) == v for k, v in exp.items())
        bad += (not ok)
        print(("PASS " if ok else "FAIL ") + name + f"  got={got}")
    # numeric checks
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        raw, logs, pred = _write_synth(pathlib.Path(td), T_real=c(222.0, 1075.0, 222.7))
        with contextlib.redirect_stdout(io.StringIO()):
            res = analyze(raw, logs, pred, pathlib.Path(td) / "out", synthetic=True)
        v = res["verdict_all_processes"]
        chk = [abs(v["H1"]["ratio_Ours_over_B0"] - 222.7 / 222.0) < 1e-9, abs(v["H2"]["ratio_B1_over_B0"] - 1075.0 / 222.0) < 1e-9, len(v["paired_ratios_by_p"]) == 3,
               all(abs(x["B1_over_B0"] - 1075.0 / 222.0) < 1e-9 for x in v["paired_ratios_by_p"].values()), abs(v["reported"]["Ours_over_B1_real"] - 222.7 / 1075.0) < 1e-9,
               abs(v["prediction_error_reported"]["B1"]["signed_pred_minus_real_over_real"] - (1070.2 - 1075.0) / 1075.0) < 1e-9, not v["prediction_error_reported"]["B1"]["flag_gt_15pct"],
               abs(v["first_request_s_mean"]["B1"] - 40.0) < 1e-9, v["n_compile_mean"]["B1"] == 31.0, v["n_capture_mean"]["Ours"] == 1.0]
        ok = all(chk)
        bad += (not ok)
        print(("PASS " if ok else "FAIL ") + f"numeric checks (ratios, paired ratios, Ours/B1, pred error, first request, counts) {chk}")
    # prediction error flag > 15 %
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        raw, logs, pred = _write_synth(pathlib.Path(td), T_real=c(222.0, 1300.0, 222.7))   # B1 real 1300 vs pred 1070.2: error -17.7 %
        with contextlib.redirect_stdout(io.StringIO()):
            res = analyze(raw, logs, pred, pathlib.Path(td) / "out", synthetic=True)
        fl = res["verdict_all_processes"]["prediction_error_reported"]["B1"]["flag_gt_15pct"]
        ok = bool(fl) and res["status"] == "JUDGED"
        bad += (not ok)
        print(("PASS " if ok else "FAIL ") + f"prediction error > 15 % is flagged (B1 flag={fl}), judgement unaffected")
    print("SELFTEST", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(X8 / "real/raw"))
    ap.add_argument("--log-dir", default=str(X8 / "real/logs"))
    ap.add_argument("--pred", default=str(X8 / "pred_X8.csv"))
    ap.add_argument("--out-dir", default=str(X8 / "verdict"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    out = pathlib.Path(a.out_dir)
    if a.synthetic:
        assert not T.inside(out, ROOT / "results"), "SYNTHETIC outputs must not be written under results/"
    else:
        check_prereg()
        assert T.inside(out, X8), f"official output must be under {X8}"
        assert T.is_ro(a.pred), f"{a.pred} (frozen prediction) must be 0444 before the block"
    res = analyze(pathlib.Path(a.raw), pathlib.Path(a.log_dir), pathlib.Path(a.pred), out, bool(a.synthetic))
    print("\n".join(summary_lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
