"""S4-X X2b confirmation verdict (CPU only): M1-M4 + validity as in the preregistration goals/S4X_X2b_whole_process_memory_bound_expandable_prereg_v1.md (v1.1) sec 5.  Written BEFORE any X2b confirmation data (self-test on synthetic data only).
Derived from scripts/m1234_verdict_S4X_X2.py (0444, unchanged; SHA16 7df9180683df4684).  The M1-M4 judging function `eval_cell` and the helper `_med` are COPIED VERBATIM from it (proved by a source diff in tests/test_verdict_S4X_X2b.py);
what changes is the design around it:
  * seed 134: every judged process must have run the seed-134 stream of its stream name (trace stem <W>_N300_seed134_B), else the verdict is INVALID.
  * allocator configuration of every judged process is CHECKED against the PYTORCH_CUDA_ALLOC_CONF value seen BY the process (x2.json alloc_conf_env): version cap must be exactly
      garbage_collection_threshold:0.8,expandable_segments:True ; version uncap (reference) must be empty ; version uncapxe (attribution arm) must be exactly expandable_segments:True.  A mismatch makes the process invalid -> INVALID verdict
      (listed).  Capped processes also need cap == C and, for Ours/B4, budget == B_pool (both from --rec); uncapped / uncapxe processes carry no cap / budget.
  * ARMS = (Ours, B4) (B1-AOT removed); both are judged.  The report-only trigger arm "trig" (X2 original recipe: cap + g0.8, NO expandable segments, one process per (arm, stream)) lives in its OWN directory (--report-raw) and is
    never read by the judged part: a trig process inside the judged tree is an integrity error, and the trig data cannot change the verdict.
  * the frozen parameters come ONLY from --rec (REQUIRED, no default, no --c/--delta override): the LAST line "C=.. delta=.. g=0.8 margin=.. B_pool=..".
Definitions (unit = one (stream, arm) cell = 3 capped (cap) + 3 uncapped reference (uncap) processes, proc ids 0-2; the attribution processes (uncapxe) feed only the report section below):
  M1  bounded     : every cap process of the cell has NVML device-level peak (nvml.window.json peak_used_mb over the process lifetime) <= C + delta.            (missing peak = not satisfied)
  M2  no crash    : every cap process ended with status ok.  Reported, not judged: n_capture_failed ("capture OOM -> eager") and n_denied (refused recordings).
  M3  same output : for each cap process p against the uncap process with the SAME proc id: only requests served by the SAME tier (driver csv `served_by`) are compared; hashes must be equal (identical hashed index sets, >= 1 same-tier
                    comparison, 0 same-tier differences; X2 v1.1 rules); a missing `served_by` is an INTEGRITY error (INVALID).
  M4  cost        : median(service_s of the ok cap processes) / median(service_s of the ok uncap processes) <= 1.03.
  "acts"          : a cell ACTS iff every ok uncap (reference, original configuration) process of the cell has NVML peak > C + delta; a cell that does not act is NOT_ACTING (excluded, numbers printed).  A judged arm with no acting cell in
                    either stream => NO_POWER ("no discriminating power").
  cell status     : PASS = M1 and M2 and M3 and M4 on complete data (3 + 3 processes);  FAIL = a failure on complete data;  INCOMPLETE = processes missing; a definitive failure (M1, M2 or M3) in an incomplete cell still counts as FAIL.
  verdict         : INVALID > FAILS > INCOMPLETE > NO_POWER > HOLDS  (HOLDS = all acting cells PASS, every arm acts in >= 1 stream).
Report sections (do NOT change the verdict; the card's wording table decides the sentence):
  bound touched   : per cell: the attribution arm (uncapxe, 3 processes) has NVML peak > C + delta in ALL 3 processes, and/or the cap processes show allocator OOMs (num_ooms) or allocator retries (num_alloc_retries) > 0.  The cap counts as
                    TOUCHED when at least one acting judged cell is touched (per-cell flags are printed).  attribution outputs vs the reference are also compared (info).
  trigger / repair: per (stream, arm) the trig process: allocator OOMs, same-tier hash differences vs the uncap process of the same stream/arm (proc 0), crash.  triggered = OOM > 0 or hash difference or crash.  No cell triggered ->
                    "the new seed did not trigger the failure; X2b holds but is not repair evidence"; a cell triggered while its cap cell is clean (no cap OOM, M3 satisfied) -> repair evidence; triggered and cap not clean -> not repaired.
Sentences (card sec 5 wording table, fixed before the data; "GB" = GiB as in the card's "16 GiB cap"):
  HOLDS and cap touched  : "With the allocator cap, garbage-collection threshold and expandable segments, whole-process device memory stayed within C+δ with bit-identical outputs and ≤3% time cost."
  HOLDS and not touched  : "Switching the allocator to expandable segments removed fragmentation and lowered the whole-process peak from about X GB to Y GB; the <C/1024:g> GiB cap was never reached and acted only as a safety net, so outputs
                           stayed bit-identical. When a cap actually binds a convolution workspace, cuDNN falls back to another algorithm and outputs can change (X2)."   (X = median uncap peak, Y = median cap peak, over the acting cells)
  not HOLDS              : no sentence; which condition failed, with the numbers (the card: report as it is).
Outputs (--out, default results/S4X/x2b/confirm/verdict/): verdict_x2b.json, cell_table.csv, process_table.csv, VERDICT_X2b.txt (last line  X2B_VERDICT=<HOLDS|FAILS|NO_POWER|INCOMPLETE|INVALID> ).
Usage (from code/):  CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B -m scripts.m1234_verdict_S4X_X2b --rec <frozen file> [--raw R] [--report-raw RR] [--out O]
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
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.analyze_S4X_X2_v2 import TierFieldMissing, discover, hash_compare  # noqa: E402

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
DEFAULT_RAW = ROOT / "results/S4X/x2b/confirm/raw"
DEFAULT_REPORT_RAW = ROOT / "results/S4X/x2b/confirm/raw_report"
DEFAULT_OUT = ROOT / "results/S4X/x2b/confirm/verdict"
SEED = 134
STREAMS = ("W2", "W5")
ARMS = ("Ours", "B4")
JUDGED = ("Ours", "B4")
PROCS = (0, 1, 2)
RATIO_MAX = 1.03
ALLOC_CAP = "garbage_collection_threshold:0.8,expandable_segments:True"
ALLOC_XE = "expandable_segments:True"
ALLOC_TRIG = "garbage_collection_threshold:0.8"
WANT_ALLOC = {"cap": ALLOC_CAP, "uncap": None, "uncapxe": ALLOC_XE}
LABEL_RE = re.compile(r"^(Ours|B4)_(cap|uncap|uncapxe)_(W2|W5)_p(\d+)$")
TRIG_RE = re.compile(r"^(Ours|B4)_trig_(W2|W5)_p(\d+)$")
REC_RE = re.compile(r"^\s*C=([0-9]+(?:\.[0-9]+)?)\s+delta=([0-9]+(?:\.[0-9]+)?)\s+g=(0\.8)\s+margin=([0-9]+(?:\.[0-9]+)?)\s+B_pool=([0-9]+(?:\.[0-9]+)?)\s*$")
SENT_TOUCHED = "With the allocator cap, garbage-collection threshold and expandable segments, whole-process device memory stayed within C+δ with bit-identical outputs and ≤3% time cost."
SENT_UNTOUCHED = ("Switching the allocator to expandable segments removed fragmentation and lowered the whole-process peak from about {X} GB to {Y} GB; the {CG} GiB cap was never reached and acted only as a safety net, "
                  "so outputs stayed bit-identical. When a cap actually binds a convolution workspace, cuDNN falls back to another algorithm and outputs can change (X2).")


def parse_rec(path) -> dict:
    lines = [l for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    m = REC_RE.match(lines[-1]) if lines else None
    if not m:
        raise ValueError(f"last line of {path} is not 'C=<n> delta=<n> g=0.8 margin=<n> B_pool=<n>': {lines[-1] if lines else '(empty)'!r}")
    return dict(C=float(m[1]), delta=float(m[2]), g=m[3], margin=float(m[4]), B_pool=float(m[5]))


def _med(v):
    return statistics.median(v) if v else None


def _norm_policy(s) -> str:
    return (s or "").replace("′", "p")


def _alloc(r) -> Optional[str]:
    return r.get("alloc_conf") or None


def _rerecords(r) -> Optional[int]:
    """Report only: number of re-recordings (a capture event of a shape evicted earlier in the same process) from the driver csv; None when the csv is missing."""
    f = r.get("csv_file")
    if not f:
        return None
    try:
        rows = [x for x in csv.DictReader(open(f, newline="")) if x.get("phase", "req") == "req"]
        evicted, n = set(), 0
        for x in rows:
            t = f'{x["H"]}x{x["W"]}_b{x["B"]}_{"cfg" if int(float(x["cfg"])) else "nocfg"}'
            if x.get("event") in ("capture", "compile_capture") and t in evicted:
                n += 1
                evicted.discard(t)
            for s in filter(None, (x.get("evicted") or "").split(";")):
                evicted.add(s)
        return n
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------------------------ loading and integrity
def load_design(raw_root, C: float, B_pool: float) -> dict:
    """-> dict[(w, arm, ver, p)] = record (ver in cap / uncap / uncapxe), plus a list of integrity problems."""
    recs = discover(raw_root)
    design: Dict[tuple, dict] = {}
    bad: List[str] = []
    for r in recs:
        lab = r.get("label") or ""
        m = LABEL_RE.match(lab)
        if not m:
            bad.append(f"{r['tag']}: label {lab!r} is not <arm>_<cap|uncap|uncapxe>_<W2|W5>_p<k> (not a judged confirmation process; report-only arms belong in --report-raw)")
            continue
        arm, ver, w, p = m[1], m[2], m[3], int(m[4])
        key = (w, arm, ver, p)
        why = []
        if key in design:
            why.append("duplicate label")
        if r.get("policy") and _norm_policy(r["policy"]) != arm:
            why.append(f"meta policy {r['policy']!r} != arm {arm}")
        if r.get("trace") and not str(r["trace"]).startswith(f"{w}_N300_seed{SEED}_B"):
            why.append(f"trace stem {r['trace']!r} is not the {w} seed-{SEED} stream")
        want = WANT_ALLOC[ver]
        if _alloc(r) != want:
            why.append(f"PYTORCH_CUDA_ALLOC_CONF seen by the process {_alloc(r)!r} != {want!r}")
        if ver == "cap":
            if r.get("cap_mb") is None or abs(r["cap_mb"] - C) > 0.5:
                why.append(f"capped process has cap {r.get('cap_mb')} != C {C}")
            if r.get("budget_mb") is None or abs(r["budget_mb"] - B_pool) > 0.5:
                why.append(f"capped process has budget {r.get('budget_mb')} != B_pool {B_pool}")
        elif r.get("cap_mb") is not None or r.get("budget_mb") is not None or r.get("gc") is not None:
            why.append(f"{ver} process carries cap / budget / g")
        if why:
            bad.append(f"{lab} ({r['tag']}): " + "; ".join(why))
        design[key] = r
    return dict(design=design, integrity=bad, n_records=len(recs))


def load_report(report_root, C: float, B_pool: float) -> dict:
    """The report-only trigger arm: dict[(w, arm, p)] = record, notes (never an integrity error of the verdict)."""
    out: Dict[tuple, dict] = {}
    notes: List[str] = []
    if not pathlib.Path(report_root).exists():
        return dict(trig=out, notes=[f"report-only directory {report_root} does not exist (trig arm not run)"])
    for r in discover(report_root):
        m = TRIG_RE.match(r.get("label") or "")
        if not m:
            notes.append(f"{r['tag']}: label {r.get('label')!r} is not <arm>_trig_<W2|W5>_p<k> (ignored)")
            continue
        key = (m[2], m[1], int(m[3]))
        why = []
        if r.get("trace") and not str(r["trace"]).startswith(f"{m[2]}_N300_seed{SEED}_B"):
            why.append(f"trace {r['trace']!r} is not the seed-{SEED} stream")
        if _alloc(r) != ALLOC_TRIG:
            why.append(f"allocator conf {_alloc(r)!r} != {ALLOC_TRIG!r}")
        if r.get("cap_mb") is None or abs(r["cap_mb"] - C) > 0.5:
            why.append(f"cap {r.get('cap_mb')} != C {C}")
        if why:
            notes.append(f"{r['label']}: configuration differs from the trigger recipe: " + "; ".join(why))
        out[key] = r
    return dict(trig=out, notes=notes)


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


# ------------------------------------------------------------------------------------------------------------------ report sections (never change the verdict)
def bound_report(design: dict, cells: List[dict], C: float, delta: float) -> dict:
    lim = C + delta
    per = []
    for c in cells:
        w, arm = c["stream"], c["arm"]
        xe = [design.get((w, arm, "uncapxe", p)) for p in PROCS]
        xe_ok = [r for r in xe if r and not r["crash"]]
        peaks = [r.get("nvml_peak_mb") for r in xe_ok]
        attr_exceed = len(xe_ok) == len(PROCS) and all(v is not None and v > lim for v in peaks)
        cap_p = [design.get((w, arm, "cap", p)) for p in PROCS]
        cap_p = [r for r in cap_p if r]
        ooms = sum((r.get("num_ooms") or 0) for r in cap_p)
        retries = sum((r.get("num_alloc_retries") or 0) for r in cap_p)
        touched = bool(attr_exceed or ooms > 0 or retries > 0)
        attr_hash = dict(n_same_tier=0, n_same_tier_diff=0, n_tier_changed_all=0, pairs=0)
        for p in PROCS:
            u, x = design.get((w, arm, "uncap", p)), design.get((w, arm, "uncapxe", p))
            if u and x and not u["crash"] and not x["crash"]:
                try:
                    h = hash_compare(u, x)
                except TierFieldMissing:
                    continue
                if h.get("available"):
                    attr_hash["pairs"] += 1
                    for k in ("n_same_tier", "n_same_tier_diff", "n_tier_changed_all"):
                        attr_hash[k] += h[k]
        per.append(dict(stream=w, arm=arm, acts=c["acts"], status=c["status"], uncapxe_peaks_mb=peaks, uncapxe_exceeds_all3=attr_exceed, cap_num_ooms=ooms, cap_num_alloc_retries=retries,
                        touched=touched, uncapxe_vs_uncap_hash=attr_hash))
    counted = [e for e in per if e["acts"] and e["status"] in ("PASS", "FAIL")]
    return dict(limit_mb=lim, cells=per, touched_any=any(e["touched"] for e in counted), rule="touched = at least one ACTING judged cell has (uncapxe NVML peaks > C+delta in all 3 processes) or (cap processes with allocator OOMs / retries)")


def trig_report(trig: dict, design: dict, cells: List[dict]) -> dict:
    per = []
    for w in STREAMS:
        for arm in ARMS:
            r = trig.get((w, arm, 0))
            cell = next(c for c in cells if c["stream"] == w and c["arm"] == arm)
            e = dict(stream=w, arm=arm, present=r is not None)
            if r is None:
                per.append(e)
                continue
            u = design.get((w, arm, "uncap", 0))
            h = None
            if u and not u["crash"] and not r["crash"]:
                try:
                    hh = hash_compare(u, r)
                    h = {k: hh.get(k) for k in ("n_common", "n_same_tier", "n_same_tier_equal", "n_same_tier_diff", "n_tier_changed_all", "same_index_set")} if hh.get("available") else None
                except TierFieldMissing as ex:
                    h = dict(error=str(ex))
            diff = (h or {}).get("n_same_tier_diff") or 0
            e.update(status=r["status"], crash=r["crash"], num_ooms=r.get("num_ooms"), num_alloc_retries=r.get("num_alloc_retries"), nvml_peak_mb=r.get("nvml_peak_mb"), hash_vs_uncap_p0=h,
                     triggered=bool(r["crash"] or (r.get("num_ooms") or 0) > 0 or diff > 0))
            cap_p = [design.get((w, arm, "cap", p)) for p in PROCS]
            cap_clean = bool(all(cap_p) and all((x.get("num_ooms") or 0) == 0 for x in cap_p) and cell["M3"])
            e["x2b_cap_cell_clean"] = cap_clean
            e["repair_evidence"] = bool(e["triggered"] and cap_clean)
            per.append(e)
    present = [e for e in per if e["present"]]
    anyt = any(e.get("triggered") for e in present)
    if len(present) < len(per):
        text = f"trig data incomplete ({len(present)} of {len(per)} cells present): no statement on triggering"
    elif not anyt:
        text = "the new seed did not trigger the failure; X2b holds but is not repair evidence"
    elif any(e.get("repair_evidence") for e in present):
        text = "repair evidence: the original X2 recipe was triggered (allocator OOM > 0 or hash difference) in " + ", ".join(f"{e['stream']}/{e['arm']}" for e in present if e["repair_evidence"]) + " while the X2b recipe was clean there"
    else:
        text = "the original X2 recipe was triggered but the X2b recipe was not clean in those cells: NOT a repair"
    return dict(cells=per, any_triggered=anyt, statement=text)


# ------------------------------------------------------------------------------------------------------------------ verdict
def verdict(raw_root, report_root, C: float, delta: float, B_pool: float) -> dict:
    ld = load_design(raw_root, C, B_pool)
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
    br = bound_report(design, cells, C, delta)
    rep = load_report(report_root, C, B_pool)
    tr = trig_report(rep["trig"], design, cells)
    sentence = None
    if v == "HOLDS":
        if br["touched_any"]:
            sentence = SENT_TOUCHED
        else:
            acting = [c for c in judged if c["acts"] and c["status"] == "PASS"]
            ux = [_med(c["uncap_peaks_mb"]) for c in acting]
            cy = [_med(c["cap_peaks_mb"]) for c in acting]
            X = _med(ux) / 1024.0
            Y = _med(cy) / 1024.0
            sentence = SENT_UNTOUCHED.format(X=f"{X:.1f}", Y=f"{Y:.1f}", CG=f"{C / 1024.0:g}")
    return dict(C=C, delta=delta, B_pool=B_pool, seed=SEED, limit_mb=C + delta, verdict=v, integrity=integrity, n_records=ld["n_records"], cells=cells,
                failing_cells=[f"{c['stream']}/{c['arm']}" for c in fails], incomplete_cells=[f"{c['stream']}/{c['arm']}" for c in incomplete],
                not_acting_cells=[f"{c['stream']}/{c['arm']}" for c in judged if c["status"] == "NOT_ACTING"], no_power_arms=no_power_arms, acts_unknown_arms=unknown_arms,
                bound_touched=br, trigger=tr, report_notes=rep["notes"], paper_sentence=sentence,
                sentence_class=(None if v != "HOLDS" else ("holds_cap_touched" if br["touched_any"] else "holds_cap_not_touched")), trig_records=rep["trig"])


CELL_COLS = ["stream", "arm", "judged", "status", "acts", "uncap_peaks_mb", "limit_mb", "cap_peaks_mb", "M1", "M2", "n_crash", "n_crash_oom", "n_capture_failed", "n_denied", "uncap_n_denied", "M3",
             "n_tier_changed_all", "n_tier_changed_hashed", "n_same_tier", "n_same_tier_equal", "n_same_tier_diff", "time_ratio", "M4", "missing", "reasons"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, list):
        return ";".join(_fmt(x) for x in v)
    return str(v)


def write_outputs(V: dict, design: dict, out_dir) -> str:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trig_recs = V.pop("trig_records")
    (out / "verdict_x2b.json").write_text(json.dumps(V, indent=1, default=str, ensure_ascii=False))
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
        cols = ["label", "version", "tag", "status", "rc", "service_s", "nvml_peak_mb", "torch_peak_res_mb", "n_capture_failed", "n_denied", "num_ooms", "num_alloc_retries", "n_evicted_total", "n_rerecord", "alloc_conf", "nvml_hz"]
        wr.writerow(cols)
        for k in sorted(design):
            r = design[k]
            wr.writerow([_fmt(r.get("label") if c == "label" else (k[2] if c == "version" else (_rerecords(r) if c == "n_rerecord" else r.get(c)))) for c in cols])
        for k in sorted(trig_recs):
            r = trig_recs[k]
            wr.writerow([_fmt(r.get("label") if c == "label" else ("trig(report-only)" if c == "version" else (_rerecords(r) if c == "n_rerecord" else r.get(c)))) for c in cols])
    L = [f"S4-X X2b confirmation verdict   seed {V['seed']}   C={V['C']:.0f}  delta={V['delta']:.0f}  C+delta={V['limit_mb']:.0f} MB  B_pool={V['B_pool']:.1f}   records={V['n_records']}", ""]
    for c in V["cells"]:
        L.append(f"[{c['stream']}/{c['arm']}] JUDGED status={c['status']} acts={c['acts']} (uncapped reference NVML peaks {_fmt(c['uncap_peaks_mb'])})" + (f" missing={c['missing']}" if c["missing"] else ""))
        t = c["m3_totals"]
        L.append(f"    M1 {c['M1']} (cap peaks {_fmt(c['cap_peaks_mb'])})   M2 {c['M2']} (crash {c['n_crash']}, crash-OOM {c['n_crash_oom']}; capture OOM->eager {c['n_capture_failed']}, recordings denied {c['n_denied']} [uncapped: {c['uncap_n_denied']}])")
        L.append(f"    M3 {c['M3']} (same-tier equal {t['n_same_tier_equal']}/{t['n_same_tier']}, diff {t['n_same_tier_diff']}; tier-changed requests {t['n_tier_changed_all']} of all, {t['n_tier_changed_hashed']} of hashed)   "
                 f"M4 {c['M4']} (ratio {_fmt(c['time_ratio'])}; cap {_fmt(c['cap_service_s'])} s / uncap {_fmt(c['uncap_service_s'])} s)")
        for r in c["reasons"]:
            L.append("    - " + r)
    L.append("")
    if V["integrity"]:
        L.append("INTEGRITY problems (verdict INVALID):")
        L += ["  - " + x for x in V["integrity"]]
    L.append(f"failing cells: {V['failing_cells'] or 'none'}; incomplete: {V['incomplete_cells'] or 'none'}; not acting (excluded): {V['not_acting_cells'] or 'none'}; arms without an acting cell: {V['no_power_arms'] or 'none'}")
    br = V["bound_touched"]
    L.append(f"BOUND TOUCHED (report; decides the sentence): {br['touched_any']}   [{br['rule']}]")
    for e in br["cells"]:
        L.append(f"    {e['stream']}/{e['arm']}: uncapxe NVML peaks {_fmt(e['uncapxe_peaks_mb'])} (all 3 > C+delta {e['uncapxe_exceeds_all3']}); cap allocator OOMs {e['cap_num_ooms']}, retries {e['cap_num_alloc_retries']}; touched {e['touched']}; "
                 f"uncapxe vs uncap hashes: same-tier diff {e['uncapxe_vs_uncap_hash']['n_same_tier_diff']} of {e['uncapxe_vs_uncap_hash']['n_same_tier']} ({e['uncapxe_vs_uncap_hash']['pairs']} pairs)")
    tr = V["trigger"]
    L.append(f"TRIGGER / REPAIR (report only; the original X2 recipe on seed 134): {tr['statement']}")
    for e in tr["cells"]:
        if not e["present"]:
            L.append(f"    {e['stream']}/{e['arm']}: trig process missing")
        else:
            L.append(f"    {e['stream']}/{e['arm']}: status {e['status']}, allocator OOMs {e['num_ooms']}, retries {e['num_alloc_retries']}, hash vs uncap p0 {e['hash_vs_uncap_p0']}, triggered {e['triggered']}, X2b cap cell clean {e['x2b_cap_cell_clean']}, repair evidence {e['repair_evidence']}")
    for n in V["report_notes"]:
        L.append("    note: " + n)
    if V["paper_sentence"]:
        L.append(f"sentence ({V['sentence_class']}): " + V["paper_sentence"].replace("C+δ", f"C+δ (= {V['limit_mb']:.0f} MB)"))
    else:
        L.append("no sentence (verdict is not HOLDS): report which condition failed, with the numbers above (card sec 5)")
    L.append(f"X2B_VERDICT={V['verdict']}")
    text = "\n".join(L) + "\n"
    (out / "VERDICT_X2b.txt").write_text(text)
    V["trig_records"] = trig_recs
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rec", required=True, help="the frozen parameter file whose LAST line is 'C=.. delta=.. g=0.8 margin=.. B_pool=..' (required, no default)")
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--report-raw", default=str(DEFAULT_REPORT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    try:
        r = parse_rec(a.rec)
    except Exception as e:  # noqa: BLE001
        print(f"X2B_VERDICT_ERROR cannot read the frozen parameters: {e}")
        return 2
    V = verdict(a.raw, a.report_raw, r["C"], r["delta"], r["B_pool"])
    text = write_outputs(V, load_design(a.raw, r["C"], r["B_pool"])["design"], a.out)
    print(text, end="")
    return 2 if V["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    sys.exit(main())
