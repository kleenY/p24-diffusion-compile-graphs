"""S4-X X2 confirmation verdict (CPU only): M1-M4 + validity, exactly as in the preregistration goals/S4X_X2_whole_process_memory_bound_prereg_v1.md sec 5 and the sec 7 revision v1.1.
Written BEFORE any X2 confirmation data (DRY self-test only).  Reads the raw tree of scripts/window_S4X_X2.sh (results/S4X/x2/confirm/raw/{cap,uncap}/) through the dev analyzer's loader (scripts/analyze_S4X_X2_v2.py, 0444, unchanged).

Unit = one (stream, arm) cell = 3 capped processes (proc id 0-2) + 3 uncapped processes.  Streams W2, W5.  Judged arms: Ours, B4.  B1-AOT = control (cap does not act on it): M2/M3/M4 (and the peaks) are REPORTED, never judged.
Frozen inputs: C and delta = the first two numbers of the LAST LINE of results/S4X/x2/dev/summary/DEV_RECOMMENDATION.txt ("C=.. delta=.. g=.. margin=.. B_pool=..", the card sec 6 values) -- or --c/--delta.

Definitions (all fixed here, before the data):
  M1  bounded     : every capped process of the cell has NVML device-level peak (nvml.window.json peak_used_mb over the process lifetime) <= C + delta.            (missing peak = not satisfied)
  M2  no crash    : every capped process of the cell ended with status ok (process non-ok = crash-level failure; crash_oom = those whose error text is an out-of-memory).  Reported, not judged: n_capture_failed
                    ("capture OOM -> eager" count of the frozen driver, with the capture fix on) and n_denied (capture_denied events = refused recordings), summed over the capped processes.
  M3  same output : for each capped process p, against the uncapped process with the SAME proc id (same arm, same stream): only requests served by the SAME tier (frozen driver csv `served_by`) are compared;
                    their output hashes must be equal (hashed requests = first 5 + last 5 of the process + the first request of every new shape).  Tier-changed requests are counted and reported (the expected price,
                    visible in M4), never compared.  Satisfied iff, for all 3 pairs: both processes ok, hash files present, identical hashed index sets, n_same_tier >= 1 (a pair with NO same-tier hashed request is
                    "no comparison" = not satisfied) and n_same_tier_diff == 0.  A missing `served_by` column / value is an INTEGRITY error (TierFieldMissing): the whole verdict is INVALID.
  M4  cost        : median(service_s of the ok capped processes) / median(service_s of the ok uncapped processes) <= 1.03   (service_s = meta.json; 3 processes each).
  validity ("the cap acts"): a cell ACTS iff every ok uncapped process of the cell has NVML peak > C + delta.  A judged-arm cell that does not act is labelled NOT_ACTING: it is excluded from the verdict
                    (but its M1-M4 numbers are printed).  An arm "acts" if it has >= 1 acting cell; a judged arm with NO acting cell => the experiment has no discriminating power (NO_POWER).
  integrity       : a record whose label / policy / trace stem / cap (capped: cap == C; uncapped: none) does not match its place in the design => INVALID (listed).
  cell status     : PASS = M1 and M2 and M3 and M4 (and complete: 3 capped + 3 uncapped processes present; uncapped ones ok);  FAIL = at least one of M1-M4 not satisfied on complete data;
                    INCOMPLETE = processes missing / uncapped not ok.  A DEFINITIVE failure (M1, M2 or M3 not satisfied on the available data) in an incomplete judged cell still counts as FAIL.
  verdict         : INVALID (integrity / tier-field errors)  >  FAILS (any acting judged cell FAIL or definitive failure)  >  INCOMPLETE (a judged cell incomplete)  >  NO_POWER (a judged arm without acting cell)  >  HOLDS
                    (all acting judged cells PASS, every judged arm acts in >= 1 stream).  HOLDS licenses the card's sentence (sec 5): "with an allocator cap plus pool-budget accounting, whole-process device memory
                    stayed within C+delta at <=3% time cost"; every other verdict is reported as it is (which condition failed, with the cost), per the card.
Outputs (--out, default results/S4X/x2/confirm/verdict/): verdict_x2.json, cell_table.csv, process_table.csv, VERDICT_X2.txt (last line  X2_VERDICT=<HOLDS|FAILS|NO_POWER|INCOMPLETE|INVALID> ).
Usage (from code/):  CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B -m scripts.m1234_verdict_S4X_X2 [--raw R] [--rec DEV_RECOMMENDATION.txt] [--c C --delta D] [--out O]      /     --selftest
Exit code: 0 a verdict was computed (any of HOLDS/FAILS/NO_POWER/INCOMPLETE), 2 INVALID / unreadable input.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import statistics
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.analyze_S4X_X2_v2 import TierFieldMissing, discover, hash_compare  # noqa: E402

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
DEFAULT_RAW = ROOT / "results/S4X/x2/confirm/raw"
DEFAULT_REC = ROOT / "results/S4X/x2/dev/summary/DEV_RECOMMENDATION.txt"
DEFAULT_OUT = ROOT / "results/S4X/x2/confirm/verdict"
STREAMS = ("W2", "W5")
ARMS = ("Ours", "B4", "B1-AOT")
JUDGED = ("Ours", "B4")
PROCS = (0, 1, 2)
RATIO_MAX = 1.03
LABEL_RE = re.compile(r"^(Ours|B4|B1-AOT)_(cap|uncap)_(W2|W5)_p(\d+)$")
REC_RE = re.compile(r"^\s*C=([0-9]+(?:\.[0-9]+)?)\s+delta=([0-9]+(?:\.[0-9]+)?)\s+g=(0\.6|0\.8|none)\s+margin=([0-9]+(?:\.[0-9]+)?)\s+B_pool=([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_rec(path) -> dict:
    lines = [l for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    m = REC_RE.match(lines[-1]) if lines else None
    if not m:
        raise ValueError(f"last line of {path} is not 'C=<n> delta=<n> g=<0.6|0.8|none> margin=<n> B_pool=<n>': {lines[-1] if lines else '(empty)'!r}")
    return dict(C=float(m[1]), delta=float(m[2]), g=m[3], margin=float(m[4]), B_pool=float(m[5]))


def _med(v):
    return statistics.median(v) if v else None


def _norm_policy(s) -> str:
    return (s or "").replace("′", "p")


# ------------------------------------------------------------------------------------------------------------------ loading and integrity
def load_design(raw_root, C: float) -> dict:
    """-> dict[(w, arm, ver, p)] = record, plus a list of integrity problems."""
    recs = discover(raw_root)
    design: Dict[tuple, dict] = {}
    bad: List[str] = []
    for r in recs:
        lab = r.get("label") or ""
        m = LABEL_RE.match(lab)
        if not m:
            bad.append(f"{r['tag']}: label {lab!r} is not <arm>_<cap|uncap>_<W2|W5>_p<k> (not a confirmation process)")
            continue
        arm, ver, w, p = m[1], m[2], m[3], int(m[4])
        key = (w, arm, ver, p)
        why = []
        if key in design:
            why.append("duplicate label")
        if r.get("policy") and _norm_policy(r["policy"]) != arm:
            why.append(f"meta policy {r['policy']!r} != arm {arm}")
        if r.get("trace") and not str(r["trace"]).startswith(f"{w}_N300_seed132_B"):
            why.append(f"trace stem {r['trace']!r} is not the {w} seed-132 stream")
        if ver == "cap":
            if r.get("cap_mb") is None or abs(r["cap_mb"] - C) > 0.5:
                why.append(f"capped process has cap {r.get('cap_mb')} != C {C}")
        elif r.get("cap_mb") is not None or r.get("budget_mb") is not None or r.get("gc") is not None:
            why.append("uncapped process carries cap / budget / g")
        if why:
            bad.append(f"{lab} ({r['tag']}): " + "; ".join(why))
        design[key] = r
    return dict(design=design, integrity=bad, n_records=len(recs))


# ------------------------------------------------------------------------------------------------------------------ one cell
def eval_cell(design: dict, w: str, arm: str, C: float, delta: float) -> dict:
    lim = C + delta
    cap = {p: design.get((w, arm, "cap", p)) for p in PROCS}
    unc = {p: design.get((w, arm, "uncap", p)) for p in PROCS}
    cap_p = [r for r in cap.values() if r]
    unc_p = [r for r in unc.values() if r]
    missing = [f"cap p{p}" for p in PROCS if not cap[p]] + [f"uncap p{p}" for p in PROCS if not unc[p]]
    cap_ok = [r for r in cap_p if not r["crash"]]
    unc_ok = [r for r in unc_p if not r["crash"]]
    out: dict = dict(stream=w, arm=arm, judged=arm in JUDGED, missing=missing, n_cap=len(cap_p), n_uncap=len(unc_p), limit_mb=lim, reasons=[], tier_errors=[])
    # validity
    unc_peaks = [r.get("nvml_peak_mb") for r in unc_ok]
    out["uncap_peaks_mb"] = unc_peaks
    out["uncap_not_ok"] = [r["tag"] for r in unc_p if r["crash"]]
    out["acts"] = bool(unc_ok) and all(v is not None and v > lim for v in unc_peaks)
    out["acts_known"] = bool(unc_ok)
    # M1
    cap_peaks = [r.get("nvml_peak_mb") for r in cap_p]
    out["cap_peaks_mb"] = cap_peaks
    out["M1"] = bool(cap_p) and all(v is not None and v <= lim for v in cap_peaks)
    if cap_p and not out["M1"]:
        out["reasons"].append("M1: capped NVML peaks " + ", ".join("none" if v is None else f"{v:.0f}" for v in cap_peaks) + f" vs C+delta {lim:.0f}")
    # M2
    out["M2"] = bool(cap_p) and not any(r["crash"] for r in cap_p)
    out["n_crash"] = sum(1 for r in cap_p if r["crash"])
    out["n_crash_oom"] = sum(1 for r in cap_p if r["crash_oom"])
    out["n_capture_failed"] = sum((r.get("n_capture_failed") or 0) for r in cap_p)
    out["n_denied"] = sum((r.get("n_denied") or 0) for r in cap_p)
    out["uncap_n_denied"] = sum((r.get("n_denied") or 0) for r in unc_p)
    if cap_p and not out["M2"]:
        out["reasons"].append("M2: capped process not ok: " + ", ".join(f"{r['tag']}:{r['status']}" for r in cap_p if r["crash"]))
    # M3
    pairs = []
    m3_ok = bool(cap_p)
    tot = dict(n_tier_changed_all=0, n_tier_changed_hashed=0, n_same_tier=0, n_same_tier_equal=0, n_same_tier_diff=0)
    for p in PROCS:
        c, u = cap[p], unc[p]
        if c is None or u is None:
            m3_ok = False
            pairs.append(dict(p=p, state="missing process"))
            continue
        if c["crash"] or u["crash"]:
            m3_ok = False
            pairs.append(dict(p=p, state="a process of the pair is not ok"))
            continue
        try:
            h = hash_compare(u, c)
        except TierFieldMissing as e:
            out["tier_errors"].append(str(e))
            m3_ok = False
            pairs.append(dict(p=p, state="TIER FIELD MISSING"))
            continue
        if not h.get("available"):
            m3_ok = False
            pairs.append(dict(p=p, state="no hash file"))
            out["reasons"].append(f"M3: p{p} hash file missing")
            continue
        for k in tot:
            tot[k] += h[k]
        state = "ok"
        if not h["same_index_set"]:
            state = "hashed index sets differ"
        elif h["n_same_tier"] < 1:
            state = "no same-tier hashed request (no comparison)"
        elif h["n_same_tier_diff"] > 0:
            state = f"{h['n_same_tier_diff']} same-tier hash difference(s)"
        if state != "ok":
            m3_ok = False
            out["reasons"].append(f"M3 p{p}: {state}")
        pairs.append(dict(p=p, state=state, **{k: h[k] for k in ("n_common", "n_tier_changed_all", "n_tier_changed_hashed", "n_same_tier", "n_same_tier_equal", "n_same_tier_diff")}))
    out["M3"] = bool(m3_ok) and not out["tier_errors"]
    out["M3_pairs"] = pairs
    out.update(m3_totals=tot)
    # M4
    ct = [r["service_s"] for r in cap_ok if r.get("service_s")]
    ut = [r["service_s"] for r in unc_ok if r.get("service_s")]
    out["cap_service_s"] = ct
    out["uncap_service_s"] = ut
    out["time_ratio"] = (_med(ct) / _med(ut)) if (ct and ut) else None
    out["M4"] = out["time_ratio"] is not None and out["time_ratio"] <= RATIO_MAX
    if out["time_ratio"] is None:
        out["reasons"].append("M4: no ok capped or uncapped service time")
    elif not out["M4"]:
        out["reasons"].append(f"M4: time ratio {out['time_ratio']:.4f} > {RATIO_MAX}")
    # status
    complete = not missing and not out["uncap_not_ok"]
    allm = out["M1"] and out["M2"] and out["M3"] and out["M4"]
    definitive_fail = bool(cap_p) and (not out["M1"] or not out["M2"] or any(pp["state"] not in ("ok", "missing process", "a process of the pair is not ok", "TIER FIELD MISSING") for pp in pairs))
    out["definitive_fail"] = definitive_fail
    if out["judged"] and not out["acts"] and out["acts_known"] and complete:
        out["status"] = "NOT_ACTING"
    elif not complete:
        out["status"] = "INCOMPLETE"
    else:
        out["status"] = "PASS" if allm else "FAIL"
    if not out["judged"]:
        out["role"] = "REPORT_ONLY (control: the cap does not act on it)"
    return out


# ------------------------------------------------------------------------------------------------------------------ verdict
def verdict(raw_root, C: float, delta: float) -> dict:
    ld = load_design(raw_root, C)
    design = ld["design"]
    cells = [eval_cell(design, w, arm, C, delta) for w in STREAMS for arm in ARMS]
    tier_errors = [e for c in cells for e in c["tier_errors"]]
    integrity = list(ld["integrity"]) + [f"TIER FIELD: {e}" for e in tier_errors]
    judged = [c for c in cells if c["judged"]]
    fails = [c for c in judged if (c["status"] == "FAIL") or (c["definitive_fail"] and c["status"] in ("INCOMPLETE", "FAIL"))]
    incomplete = [c for c in judged if c["status"] == "INCOMPLETE"]
    no_power_arms = [a for a in JUDGED if not any(c["arm"] == a and c["status"] in ("PASS", "FAIL") and c["acts"] for c in judged)]
    unknown_arms = [a for a in JUDGED if any(c["arm"] == a and not c["acts_known"] for c in judged)]
    if integrity:
        v = "INVALID"
    elif fails:
        v = "FAILS"
    elif incomplete:
        v = "INCOMPLETE"
    elif no_power_arms:
        v = "NO_POWER"
    else:
        v = "HOLDS"
    return dict(C=C, delta=delta, limit_mb=C + delta, verdict=v, integrity=integrity, n_records=ld["n_records"], cells=cells,
                failing_cells=[f"{c['stream']}/{c['arm']}" for c in fails], incomplete_cells=[f"{c['stream']}/{c['arm']}" for c in incomplete],
                not_acting_cells=[f"{c['stream']}/{c['arm']}" for c in judged if c["status"] == "NOT_ACTING"], no_power_arms=no_power_arms, acts_unknown_arms=unknown_arms,
                paper_sentence=("with an allocator cap plus pool-budget accounting, whole-process device memory stayed within C+delta at <=3% time cost" if v == "HOLDS" else None))


CELL_COLS = ["stream", "arm", "judged", "status", "acts", "uncap_peaks_mb", "limit_mb", "cap_peaks_mb", "M1", "M2", "n_crash", "n_crash_oom", "n_capture_failed", "n_denied", "uncap_n_denied", "M3",
             "n_tier_changed_all", "n_tier_changed_hashed", "n_same_tier", "n_same_tier_equal", "n_same_tier_diff", "time_ratio", "M4", "missing", "reasons"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, list):
        return ";".join(_fmt(x) for x in v)
    return str(v)


def write_outputs(V: dict, recs_design: dict, out_dir) -> str:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict_x2.json").write_text(json.dumps(V, indent=1, default=str, ensure_ascii=False))
    with open(out / "cell_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(CELL_COLS)
        for c in V["cells"]:
            row = dict(c)
            row.update(c["m3_totals"])
            row["reasons"] = " | ".join(c["reasons"])
            wr.writerow([_fmt(row.get(k)) for k in CELL_COLS])
    with open(out / "process_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        cols = ["label", "tag", "status", "rc", "service_s", "nvml_peak_mb", "torch_peak_res_mb", "n_capture_failed", "n_denied", "num_ooms", "nvml_hz"]
        wr.writerow(cols)
        for k in sorted(recs_design):
            r = recs_design[k]
            wr.writerow([_fmt(r.get(c)) for c in cols])
    L = [f"S4-X X2 confirmation verdict   C={V['C']:.0f}  delta={V['delta']:.0f}  C+delta={V['limit_mb']:.0f} MB   records={V['n_records']}", ""]
    for c in V["cells"]:
        tag = "JUDGED" if c["judged"] else "REPORT_ONLY"
        L.append(f"[{c['stream']}/{c['arm']}] {tag} status={c['status']} acts={c['acts']} (uncapped NVML peaks {_fmt(c['uncap_peaks_mb'])})" + (f" missing={c['missing']}" if c["missing"] else ""))
        t = c["m3_totals"]
        L.append(f"    M1 {c['M1']} (capped peaks {_fmt(c['cap_peaks_mb'])})   M2 {c['M2']} (crash {c['n_crash']}, crash-OOM {c['n_crash_oom']}; capture OOM->eager {c['n_capture_failed']}, recordings denied {c['n_denied']} [uncapped: {c['uncap_n_denied']}])")
        L.append(f"    M3 {c['M3']} (same-tier equal {t['n_same_tier_equal']}/{t['n_same_tier']}, diff {t['n_same_tier_diff']}; tier-changed requests {t['n_tier_changed_all']} of all, {t['n_tier_changed_hashed']} of hashed)   "
                 f"M4 {c['M4']} (ratio {_fmt(c['time_ratio'])}; cap {_fmt(c['cap_service_s'])} s / uncap {_fmt(c['uncap_service_s'])} s)")
        for r in c["reasons"]:
            L.append("    - " + r)
    L.append("")
    if V["integrity"]:
        L.append("INTEGRITY problems (verdict INVALID):")
        L += ["  - " + x for x in V["integrity"]]
    L.append(f"failing cells: {V['failing_cells'] or 'none'}; incomplete: {V['incomplete_cells'] or 'none'}; not acting (excluded): {V['not_acting_cells'] or 'none'}; arms without an acting cell: {V['no_power_arms'] or 'none'}")
    if V["paper_sentence"]:
        L.append("paper sentence (card sec 5, holds): " + V["paper_sentence"].replace("C+delta", f"C+delta (= {V['limit_mb']:.0f} MB)"))
    L.append(f"X2_VERDICT={V['verdict']}")
    text = "\n".join(L) + "\n"
    (out / "VERDICT_X2.txt").write_text(text)
    return text


# ------------------------------------------------------------------------------------------------------------------ synthetic data for --selftest
def _w(path, text):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(text)


def write_proc(root, spec: dict):
    arm, ver, w, p = spec["arm"], spec["ver"], spec["w"], spec["p"]
    d = pathlib.Path(root) / ver
    stem = f"{w}_N300_seed132_B"
    tag = f"{arm}_{stem}_p{p}"
    n = 300
    if spec.get("omit"):
        return
    status = spec.get("status", "ok")
    meta = dict(status=status, policy=spec.get("meta_policy", arm), trace=f"../results/S4X/x2/confirm/workload/{stem}.jsonl", service_s=spec["service_s"], n_requests=n,
                n_capture_failed=spec.get("n_capture_failed", 0), event_counts=spec.get("events", {"steady": 270, "capture_denied": spec.get("n_denied", 0)}), error=spec.get("error", ""), pinned_fallback_count=0)
    _w(d / f"{tag}.meta.json", json.dumps(meta))
    _w(d / f"{tag}.x2.json", json.dumps(dict(calls_match_rows=True, torch_begin=dict(allocated_mb=2056.6), torch_end=dict(max_allocated_mb=9000.0, max_reserved_mb=(spec["peak"] or 0) - 700, num_ooms=0))))
    var = dict(LABEL=f"{arm}_{ver}_{w}_p{p}", POLICY=arm, PROC=str(p), TRACE=f"x/{stem}.jsonl", CAP_MB=(str(spec["cap_mb"]) if spec.get("cap_mb") is not None else ""),
               GC=(spec.get("gc") or ""), BUDGET_MB=(str(spec["budget"]) if spec.get("budget") is not None else ""))
    _w(d / f"{tag}.x2run.json", json.dumps(dict(tag=tag, rc=0 if status == "ok" else 1, t0=0.0, t1=100.0, variant=var, meta_status=status)))
    _w(d / f"{tag}.nvml.window.json", json.dumps(dict(peak_used_mb=spec["peak"], baseline_used_mb=500.0, achieved_hz=20.0, n_samples=2000)))
    tiers = spec["tiers"]
    cols = "idx,req_id,H,W,B,cfg,status,phase,served_by,graph_mem_mb" if not spec.get("no_served_by") else "idx,req_id,H,W,B,cfg,status,phase,graph_mem_mb"
    rows = [cols]
    for i in range(n):
        rows.append(f"{i},r{i},512,512,1,0,ok,req," + ("" if spec.get("no_served_by") else f"{tiers[i]},") + "0.0")
    _w(d / f"{tag}.csv", "\n".join(rows) + "\n")
    hrows = ["idx,req_id,req_shape,served_key,call_kind,dtype,tensor_shape,nbytes,sha256"]
    for i in sorted(spec["hash_idx"]):
        hrows.append(f"{i},r{i},512x512x1x0,k,E0,float16,1x3,6,{spec['hashes'][i]}")
    _w(d / f"{tag}.x2hash.csv", "\n".join(hrows) + "\n")


def synth(root, C=16384.0, delta=1000.0, mutate=None):
    """A complete 36-process design that HOLDS.  mutate(spec) may edit a process spec in place (keys: arm ver w p status service_s peak tiers hashes cap_mb ...)."""
    hidx = list(range(0, 5)) + list(range(295, 300)) + [10, 20, 30]
    for w in STREAMS:
        for arm in ARMS:
            for ver in ("cap", "uncap"):
                for p in PROCS:
                    jit = 1.0 + 0.002 * p
                    if arm == "B1-AOT":
                        peak = 10800.0
                    elif ver == "uncap":
                        peak = 20000.0 if arm == "Ours" else 20800.0
                    else:
                        peak = 16900.0 if arm == "Ours" else 17100.0
                    base = 68.0 * jit
                    spec = dict(arm=arm, ver=ver, w=w, p=p, service_s=base * (1.01 if ver == "cap" else 1.0), peak=peak, status="ok",
                                tiers=["G"] * 300, hash_idx=hidx, hashes={i: f"h{i}" for i in hidx},
                                cap_mb=(C if ver == "cap" else None), gc=("0.8" if ver == "cap" else None), budget=((990.0 if arm != "B1-AOT" else None) if ver == "cap" else None))
                    if ver == "cap" and arm in JUDGED:       # a few requests change tier under the cap (hashed ones differ, which is allowed)
                        for i in (10, 20, 30):
                            spec["tiers"][i] = "E0"
                            spec["hashes"][i] = f"hcap{i}"
                    if mutate:
                        mutate(spec)
                    write_proc(root, spec)


def _run(mutate=None, C=16384.0, delta=1000.0):
    with tempfile.TemporaryDirectory() as td:
        synth(td, C, delta, mutate)
        return verdict(td, C, delta)


def selftest() -> int:
    ok_all = True

    def check(name, cond, info=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))

    def cell(V, w, arm):
        return next(c for c in V["cells"] if c["stream"] == w and c["arm"] == arm)

    # 0 rec line parsing
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "R.txt"
        p.write_text("x\nC=16384 delta=1156 g=0.8 margin=512.0 B_pool=990.4\n")
        r = parse_rec(p)
        check("parse_rec reads the last line", r["C"] == 16384 and r["delta"] == 1156 and r["g"] == "0.8" and abs(r["B_pool"] - 990.4) < 1e-9)
        p.write_text("C=16384 delta=1156 g=none margin=0 B_pool=0\n\n")
        check("parse_rec g=none and trailing blank line", parse_rec(p)["g"] == "none")
        p.write_text("C=16384 delta=1156\n")
        try:
            parse_rec(p); check("parse_rec rejects a malformed last line", False)
        except ValueError:
            check("parse_rec rejects a malformed last line", True)
    # 1 holds
    V = _run()
    check("holds: verdict HOLDS", V["verdict"] == "HOLDS", V["verdict"] + " " + str(V["integrity"][:2]))
    check("holds: all 4 judged cells PASS and act", all(c["status"] == "PASS" and c["acts"] for c in V["cells"] if c["judged"]))
    check("holds: B1-AOT reported, not judged, not acting", all((not c["judged"]) and (not c["acts"]) for c in V["cells"] if c["arm"] == "B1-AOT"))
    check("holds: tier-changed requests are counted (3 hashed per capped pair)", cell(V, "W2", "Ours")["m3_totals"]["n_tier_changed_all"] == 9 and cell(V, "W2", "Ours")["m3_totals"]["n_same_tier_diff"] == 0)
    check("holds: paper sentence present", V["paper_sentence"] is not None)

    # 2 M1 not satisfied
    def m1(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "cap", "W2", 1): s["peak"] = 17500.0
    V = _run(m1)
    check("M1 violated (one capped Ours/W2 process 17500 > 17384) -> FAILS", V["verdict"] == "FAILS" and not cell(V, "W2", "Ours")["M1"] and cell(V, "W2", "Ours")["M2"] and cell(V, "W2", "Ours")["M3"])
    # 2b M1 boundary: exactly C+delta passes
    def m1b(s):
        if s["ver"] == "cap" and s["arm"] == "Ours": s["peak"] = 17384.0
    check("M1 boundary: peak == C+delta passes", _run(m1b)["verdict"] == "HOLDS")

    # 3 M2 crash (OOM)
    def m2(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "cap", "W5", 2): s.update(status="fail", error="CUDA out of memory. Tried to allocate")
    V = _run(m2)
    c = cell(V, "W5", "B4")
    check("M2 violated (capped B4/W5 crash with OOM) -> FAILS, crash-OOM counted", V["verdict"] == "FAILS" and not c["M2"] and c["n_crash"] == 1 and c["n_crash_oom"] == 1)

    # 3b capture OOM -> eager: reported, not a crash
    def m2b(s):
        if s["ver"] == "cap" and s["arm"] == "Ours": s.update(n_capture_failed=2, n_denied=40)
    V = _run(m2b)
    check("capture OOM->eager and denied recordings are reported, not judged", V["verdict"] == "HOLDS" and cell(V, "W2", "Ours")["n_capture_failed"] == 6 and cell(V, "W2", "Ours")["n_denied"] == 120)

    # 4 M3: same-tier hash differs
    def m3(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "cap", "W5", 0): s["hashes"][2] = "DIFFERENT"
    V = _run(m3)
    check("M3 violated (same-tier hash differs) -> FAILS", V["verdict"] == "FAILS" and not cell(V, "W5", "Ours")["M3"] and cell(V, "W5", "Ours")["m3_totals"]["n_same_tier_diff"] == 1)

    # 4b M3: different hash only on a tier-changed request is fine (that is the default synth) ; vacuous comparison
    def m3v(s):
        if s["ver"] == "cap" and s["arm"] == "B4" and s["w"] == "W2" and s["p"] == 0:
            s["tiers"] = ["E0"] * 300
            for i in s["hash_idx"]: s["hashes"][i] = "x" + str(i)
    V = _run(m3v)
    check("M3 vacuous (no same-tier hashed request in a pair) -> not satisfied", V["verdict"] == "FAILS" and not cell(V, "W2", "B4")["M3"] and any("no comparison" in r for r in cell(V, "W2", "B4")["reasons"]))

    # 4c M3: hashed index set differs
    def m3i(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "cap", "W2", 1):
            s["hash_idx"] = [i for i in s["hash_idx"] if i != 20]
    V = _run(m3i)
    check("M3 hashed index sets differ -> not satisfied", V["verdict"] == "FAILS" and not cell(V, "W2", "B4")["M3"])

    # 5 M4
    def m4(r):
        def f(s):
            if s["ver"] == "cap" and s["arm"] == "Ours" and s["w"] == "W2": s["service_s"] = s["service_s"] / 1.01 * r
        return f
    V = _run(m4(1.05))
    check("M4 violated (ratio ~1.05) -> FAILS", V["verdict"] == "FAILS" and not cell(V, "W2", "Ours")["M4"] and abs(cell(V, "W2", "Ours")["time_ratio"] - 1.05) < 1e-6)
    check("M4 just under the line (1.029) holds", _run(m4(1.029))["verdict"] == "HOLDS")
    check("M4 just over the line (1.031) fails", _run(m4(1.031))["verdict"] == "FAILS")

    # 5b M4 uses the median: one slow capped process out of three does not break it
    def m4m(s):
        if s["ver"] == "cap" and s["arm"] == "Ours" and s["w"] == "W2" and s["p"] == 2: s["service_s"] = 68.0 * 1.5
    check("M4 is a median over 3 (one slow outlier tolerated)", _run(m4m)["verdict"] == "HOLDS")

    # 6 no discriminating power: Ours never acts
    def nop(s):
        if s["ver"] == "uncap" and s["arm"] == "Ours": s["peak"] = 17000.0
    V = _run(nop)
    check("no power: Ours uncapped peak <= C+delta in both streams -> NO_POWER", V["verdict"] == "NO_POWER" and V["no_power_arms"] == ["Ours"] and cell(V, "W2", "Ours")["status"] == "NOT_ACTING")
    # 6b B4 never acts
    def nop4(s):
        if s["ver"] == "uncap" and s["arm"] == "B4": s["peak"] = 17384.0
    check("no power: B4 uncapped peak == C+delta (not strictly greater) -> NO_POWER", _run(nop4)["verdict"] == "NO_POWER")
    # 6c one stream does not act: excluded, still HOLDS
    def one(s):
        if s["ver"] == "uncap" and s["arm"] == "Ours" and s["w"] == "W5": s["peak"] = 17000.0
    V = _run(one)
    check("Ours/W5 not acting but Ours/W2 acts -> that cell excluded, HOLDS", V["verdict"] == "HOLDS" and V["not_acting_cells"] == ["W5/Ours"])
    # 6d a not-acting cell does not hide a failure in an acting cell of the other arm
    def mix(s):
        nop(s)
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "cap", "W2", 0): s["peak"] = 19000.0
    V = _run(mix)
    check("failure of an acting cell dominates NO_POWER of the other arm -> FAILS", V["verdict"] == "FAILS" and V["no_power_arms"] == ["Ours"])

    # 7 B1-AOT cannot change the verdict
    def aot(s):
        if s["arm"] == "B1-AOT" and s["ver"] == "cap":
            s.update(status="fail", error="out of memory", service_s=999.0, peak=30000.0)
    V = _run(aot)
    check("B1-AOT crash / slowdown / high peak is reported only -> still HOLDS", V["verdict"] == "HOLDS" and cell(V, "W2", "B1-AOT")["n_crash"] == 3 and V["failing_cells"] == [])

    # 8 incomplete
    def inc(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "cap", "W2", 2): s["omit"] = True
    V = _run(inc)
    check("a missing capped process -> INCOMPLETE", V["verdict"] == "INCOMPLETE" and V["incomplete_cells"] == ["W2/B4"])
    def inc2(s):
        inc(s)
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "cap", "W2", 0): s["peak"] = 19000.0
    V = _run(inc2)
    check("incomplete cell with a definitive M1 failure -> FAILS", V["verdict"] == "FAILS")
    def inc3(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "uncap", "W5", 1): s["omit"] = True
    check("a missing UNCAPPED process -> INCOMPLETE", _run(inc3)["verdict"] == "INCOMPLETE")
    def empty(s):
        if s["w"] == "W5": s["omit"] = True
    check("a whole stream missing -> INCOMPLETE", _run(empty)["verdict"] == "INCOMPLETE")

    # 9 integrity
    def nots(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "cap", "W2", 0): s["no_served_by"] = True
    with tempfile.TemporaryDirectory() as td:
        synth(td, 16384.0, 1000.0, nots)
        try:
            V = verdict(td, 16384.0, 1000.0)
            check("served_by column missing -> INVALID (not silently same tier)", V["verdict"] == "INVALID" and any("TIER FIELD" in x for x in V["integrity"]), V["verdict"])
            txt = write_outputs(V, load_design(td, 16384.0)["design"], pathlib.Path(td) / "_verdict")
            check("outputs written, last line X2_VERDICT=INVALID", txt.rstrip().splitlines()[-1] == "X2_VERDICT=INVALID" and (pathlib.Path(td) / "_verdict/cell_table.csv").exists())
        except Exception as e:  # noqa: BLE001
            check("served_by column missing -> INVALID", False, repr(e))
    def badcap(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "cap", "W2", 0): s["cap_mb"] = 18432.0
    V = _run(badcap)
    check("a capped process whose recorded cap != C -> INVALID", V["verdict"] == "INVALID" and any("cap" in x for x in V["integrity"]))
    def badpol(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "uncap", "W5", 0): s["meta_policy"] = "Ours"
    check("policy in meta.json != arm in label -> INVALID", _run(badpol)["verdict"] == "INVALID")
    def capun(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("B4", "uncap", "W5", 0): s["cap_mb"] = 14336.0
    check("an uncapped process that carries a cap -> INVALID", _run(capun)["verdict"] == "INVALID")

    # 10 NVML peak missing -> M1 not satisfied
    def nonv(s):
        if (s["arm"], s["ver"], s["w"], s["p"]) == ("Ours", "cap", "W2", 0): s["peak"] = None
    with tempfile.TemporaryDirectory() as td:
        synth(td, 16384.0, 1000.0, nonv)
        V = verdict(td, 16384.0, 1000.0)
        check("missing NVML peak -> M1 not satisfied -> FAILS", V["verdict"] == "FAILS" and not cell(V, "W2", "Ours")["M1"])

    print("SELFTEST_OK" if ok_all else "SELFTEST_FAIL")
    return 0 if ok_all else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--rec", default=str(DEFAULT_REC))
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.c is None or a.delta is None:
        try:
            r = parse_rec(a.rec)
        except Exception as e:  # noqa: BLE001
            print(f"X2_VERDICT_ERROR cannot read C and delta: {e}")
            return 2
        C, delta = (a.c if a.c is not None else r["C"]), (a.delta if a.delta is not None else r["delta"])
    else:
        C, delta = a.c, a.delta
    V = verdict(a.raw, C, delta)
    text = write_outputs(V, load_design(a.raw, C)["design"], a.out)
    print(text, end="")
    return 2 if V["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    sys.exit(main())
