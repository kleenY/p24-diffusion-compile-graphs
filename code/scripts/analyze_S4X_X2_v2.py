"""S4-X X2 analysis v2 (CPU only).  v1 (analyze_S4X_X2.py, 0444) is NOT changed; this is the copy revised for card sec 7 (goals/S4X_X2_whole_process_memory_bound_prereg_v1.md, revision v1.1) -- written BEFORE any X2 GPU data.
Reads the output of scripts/run_capped_S4X_X2.sh (per-process records), compares output hashes, and produces the development-block recommendation (C, delta, g, margin, B_pool).  Nothing here changes a run.

A process = the files  <dir>/<tag>.{csv,meta.json,x2.json,x2items.csv,x2hash.csv,nvml.csv,nvml.window.json,x2run.json}  (any may be missing -> the field is None).
Definitions (all MiB) -- unchanged from v1 unless marked v2:
  W            torch allocated after the model load (x2.json torch_begin.allocated_mb) of the E0 probe  (weights + static buffers; cost table: 2056.6)
  M_e0         E0 probe (policy B0, only the max-shape request x12, default allocator): torch max_memory_allocated at the end - W
  M_c1g1       C1+G1 probe (policy B5 forced to C1+G1): torch max_memory_allocated - W - max(graph_mem_mb of its rows)
  M_work_max   max(M_e0, M_c1g1);  need = W + M_work_max  (a candidate C below `need` is not eligible: the max shape would not fit eagerly)
  delta        (NVML used - torch reserved), joined per item (x2items.csv); delta_peak_gap = NVML peak - torch max_reserved; delta_dev = max over every dev process of max(item max, peak gap);
               frozen delta = ceil(delta_dev) + 128.
  g            of {0.6, 0.8}: processes ok, run calls == ok rows, no same-tier hash difference vs the no-GC reference; smaller mean service-time ratio wins; none ok -> "none".
  B_pool       C - W - M_work_max - margin (>= 0), capped at the default tier 6,565.1 MB (flags when the cap binds / when below the 'tight' tier 2,204.99)
  margin       smallest tried margin (256 MB steps, ascending from 0; trials at the middle C) whose Ours process is ok, has no capture OOM and NVML peak <= C + delta (C = that process' own cap).
v2 additions (card sec 7.3 / 7.4; thresholds written here before any data):
  per (arm in {Ours, B4}, candidate C): the capped process (label ours_C<C> / b4_C<C>) vs the uncapped one (ours_uncap / b4_uncap) of the same dev stream:
     peak_ok   NVML peak <= C + delta        crash-free  process status ok (capture OOM -> eager is NOT a crash; counted and reported)        ratio_ok  service_s(capped)/service_s(uncapped) <= 1.02
     passes    = peak_ok and crash-free and ratio_ok
  C RULE (card sec 7.3 item 3): the smallest candidate C that is eligible (C >= need) and where BOTH Ours and B4 pass; none -> C = 18432 and flag NONE_PASSED.
  M3 (card sec 7.4): only requests served by the SAME tier in the capped and the uncapped run are compared (tier = the frozen driver's per-request `served_by` field in <tag>.csv);
     n_tier_changed (all requests, and among the hashed ones), n_same_tier, n_same_tier_equal, n_same_tier_diff are reported.  A missing `served_by` column / csv / value is an ERROR (TierFieldMissing; the CLI ends with code 2),
     never silently treated as 'same tier'.
  DEV_RECOMMENDATION.txt: the LAST line is exactly  C=<C> delta=<delta> g=<g> margin=<margin> B_pool=<B_pool>
Subcommands (each prints KEY=VALUE lines for the shell, plus a JSON line):
  proc      --dir D --tag T                           one process record
  probes    --raw R [--c0 14336] [--cap-max 18432]    W, M_e0, M_c1g1, M_WORK, NEED, C_USE (v1 C rule, informational in v2), delta so far
  bpool     --c C --w W --m M --margin X              B_POOL
  margin-check --raw R --label L --c C --delta D      MARGIN_PASS=1|0 (+ reasons)
  select-g  --raw R                                   G_SEL
  delta     --raw R                                   DELTA_DEV, DELTA_FROZEN
  dev-summary --raw R --out S [--c0 14336]            dev_summary.json, dev_table.csv, dev_c_table.csv, DEV_RECOMMENDATION.txt   (exit 2 when a tier-field error occurred)
  hashcmp   --a DIR:TAG --b DIR:TAG                   hash comparison of two processes (a = reference, b = capped; raises TierFieldMissing without the field)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.nvml_sampler_S4X_X2 import read_csv as read_nvml_csv  # noqa: E402

DEFAULT_TIER_MB = 6565.1
TIGHT_TIER_MB = 2204.99
CAP_MAX_MB = 18432
DELTA_PAD_MB = 128
JOIN_DT_S = 0.10
STEADY_STEP_MB = 32.0


def _j(p) -> Optional[dict]:
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _f(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------------------------------------------ one process
def read_items(path) -> List[tuple]:
    out = []
    try:
        with open(path) as f:
            next(f)
            for line in f:
                p = line.strip().split(",")
                out.append((float(p[1]), float(p[2]), float(p[3])))
    except Exception:  # noqa: BLE001
        pass
    return out


def delta_series(items: List[tuple], nvml: List[tuple], max_dt: float = JOIN_DT_S) -> List[dict]:
    ts = [r[0] for r in nvml]
    out = []
    prev_res = None
    for k, (t, res, alc) in enumerate(items):
        i = bisect.bisect_left(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts) and abs(ts[j] - t) <= max_dt and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
                best = j
        steady = prev_res is not None and k >= 3 and abs(res - prev_res) <= STEADY_STEP_MB
        prev_res = res
        if best is None:
            continue
        used = nvml[best][1]
        out.append(dict(item=k, t=t, reserved=res, alloc=alc, used=used, delta=used - res, steady=steady))
    return out


def _med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def read_rows(path) -> List[dict]:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def load_proc(d, tag: str) -> dict:
    d = pathlib.Path(d)
    run = _j(d / f"{tag}.x2run.json") or {}
    meta = _j(d / f"{tag}.meta.json") or {}
    x = _j(d / f"{tag}.x2.json") or {}
    win = _j(d / f"{tag}.nvml.window.json") or {}
    var = run.get("variant") or {}
    items = read_items(d / f"{tag}.x2items.csv")
    nv = read_nvml_csv(d / f"{tag}.nvml.csv") if (d / f"{tag}.nvml.csv").exists() else []
    if run.get("t0") and run.get("t1"):
        nv_w = [r for r in nv if run["t0"] <= r[0] <= run["t1"]]
    else:
        nv_w = nv
    ds = delta_series(items, nv_w)
    tb, te = x.get("torch_begin") or {}, x.get("torch_end") or {}
    status = meta.get("status")
    err = meta.get("error") or ""
    rows = read_rows(d / f"{tag}.csv")
    ev = meta.get("event_counts") or {}
    rec = dict(tag=tag, dir=str(d), label=var.get("LABEL"), policy=meta.get("policy") or var.get("POLICY"), trace=pathlib.Path(meta.get("trace") or var.get("TRACE") or "").stem or None,
               cap_mb=_f(var.get("CAP_MB") or None), gc=_f(var.get("GC") or None), budget_mb=_f(var.get("BUDGET_MB") or None), rc=run.get("rc"),
               status=status, error=err[:300], crash=(status != "ok"), crash_oom=(status != "ok" and ("out of memory" in err.lower() or "outofmemory" in err.lower())),
               service_s=_f(meta.get("service_s")), n_requests=meta.get("n_requests"), event_counts=ev, n_capture_failed=meta.get("n_capture_failed"), n_denied=ev.get("capture_denied", 0),
               n_evicted_total=meta.get("n_evicted_total"), calls_match_rows=x.get("calls_match_rows"),
               w_alloc_mb=_f(tb.get("allocated_mb")), torch_peak_alloc_mb=_f(te.get("max_allocated_mb")), torch_peak_res_mb=_f(te.get("max_reserved_mb")),
               torch_begin_peak_res_mb=_f(tb.get("max_reserved_mb")), num_ooms=te.get("num_ooms"), num_alloc_retries=te.get("num_alloc_retries"), num_device_free=te.get("num_device_free"),
               alloc_conf=x.get("alloc_conf_env"), capture_fix=x.get("capture_fix"),
               nvml_peak_mb=_f(win.get("peak_used_mb")), nvml_baseline_mb=_f(win.get("baseline_used_mb")), nvml_hz=_f(win.get("achieved_hz")), nvml_n=win.get("n_samples"),
               nvml_max_gap_s=_f(win.get("max_gap_s")), n_items_joined=len(ds),
               delta_all_max=max((r["delta"] for r in ds), default=None), delta_all_median=_med([r["delta"] for r in ds]),
               delta_steady_max=max((r["delta"] for r in ds if r["steady"]), default=None), delta_steady_median=_med([r["delta"] for r in ds if r["steady"]]),
               graph_mem_mb_max=max((float(r["graph_mem_mb"]) for r in rows if r.get("graph_mem_mb") not in (None, "")), default=None),
               hash_file=str(d / f"{tag}.x2hash.csv") if (d / f"{tag}.x2hash.csv").exists() else None, csv_file=str(d / f"{tag}.csv") if (d / f"{tag}.csv").exists() else None)
    rec["delta_peak_gap"] = (rec["nvml_peak_mb"] - rec["torch_peak_res_mb"]) if (rec["nvml_peak_mb"] is not None and rec["torch_peak_res_mb"] is not None) else None
    rec["delta_dev_candidate"] = max([v for v in (rec["delta_all_max"], rec["delta_peak_gap"]) if v is not None], default=None)
    return rec


def discover(raw_root) -> List[dict]:
    recs = []
    for p in sorted(pathlib.Path(raw_root).rglob("*.x2run.json")):
        recs.append(load_proc(p.parent, p.name[:-len(".x2run.json")]))
    return recs


def by_label(recs: List[dict]) -> Dict[str, dict]:
    return {r["label"]: r for r in recs if r.get("label")}


# ------------------------------------------------------------------------------------------------------------------ hash comparison (v2: same-tier only, strict)
class TierFieldMissing(ValueError):
    """The executed-tier field (`served_by` of the frozen driver csv) is missing: M3 cannot be evaluated."""


def served_by_map(rec: dict) -> Dict[int, str]:
    f = rec.get("csv_file")
    if not f:
        raise TierFieldMissing(f"{rec.get('tag')}: driver csv missing -> no served_by field")
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or "served_by" not in rd.fieldnames:
            raise TierFieldMissing(f"{rec.get('tag')}: column served_by absent in {f}")
        return {int(r["idx"]): (r.get("served_by") or "") for r in rd if r.get("phase", "req") == "req"}


def hash_compare(a: dict, b: dict) -> dict:
    """a = reference (uncapped / no-GC), b = the other process of the same stream.  Only requests served by the SAME tier in both runs are compared (card sec 7.4); tier changes are counted."""
    if not (a.get("hash_file") and b.get("hash_file")):
        return dict(available=False)
    ha = {int(r["idx"]): r for r in read_rows(a["hash_file"])}
    hb = {int(r["idx"]): r for r in read_rows(b["hash_file"])}
    sa, sb = served_by_map(a), served_by_map(b)
    common = sorted(set(ha) & set(hb))
    for i in common:
        if not sa.get(i) or not sb.get(i):
            raise TierFieldMissing(f"{a.get('tag')} / {b.get('tag')}: request {i} is hashed but has no served_by value")
    same = [i for i in common if sa[i] == sb[i]]
    same_diff = [i for i in same if ha[i]["sha256"] != hb[i]["sha256"]]
    changed_all = [i for i in set(sa) & set(sb) if sa[i] != sb[i] and sa[i] and sb[i]]
    return dict(available=True, n_a=len(ha), n_b=len(hb), n_common=len(common), same_index_set=(set(ha) == set(hb)),
                n_diff=sum(1 for i in common if ha[i]["sha256"] != hb[i]["sha256"]),
                n_tier_changed_all=len(changed_all), n_tier_changed_hashed=len(common) - len(same),
                n_same_tier=len(same), n_same_tier_equal=len(same) - len(same_diff), n_same_tier_diff=len(same_diff), same_tier_diff_idx_head=same_diff[:10])


# ------------------------------------------------------------------------------------------------------------------ development-block quantities
def ceil512(x: float) -> int:
    return int(math.ceil(x / 512.0) * 512)


def probes(L: Dict[str, dict], c0: float = 14336.0, cap_max: float = CAP_MAX_MB) -> dict:
    e0, c1, e0c = L.get("probe_e0_uncap"), L.get("probe_c1g1_uncap"), L.get("probe_e0_cap")
    out: dict = dict(C0=c0, available=dict(e0_uncap=bool(e0), c1g1_uncap=bool(c1), e0_cap=bool(e0c), c1g1_cap=bool(L.get("probe_c1g1_cap"))))
    W = (e0 or {}).get("w_alloc_mb") or (c1 or {}).get("w_alloc_mb")
    out["W"] = W
    M_e0 = (e0["torch_peak_alloc_mb"] - W) if (e0 and e0.get("torch_peak_alloc_mb") is not None and W is not None) else None
    M_c1 = None
    if c1 and c1.get("torch_peak_alloc_mb") is not None and W is not None:
        M_c1 = c1["torch_peak_alloc_mb"] - W - (c1.get("graph_mem_mb_max") or 0.0)
    out.update(M_e0=M_e0, M_c1g1=M_c1)
    Ms = [m for m in (M_e0, M_c1) if m is not None]
    M = max(Ms) if Ms else None
    out["M_WORK"] = M
    flags = []
    if W is None or M is None:
        out.update(C_USE=int(c0) if float(c0).is_integer() else c0, flags=["probes incomplete: C stays C0"])
        return out
    need = W + M
    cap_failed = bool(e0c and e0c.get("crash"))
    if cap_failed and e0 and e0.get("torch_peak_res_mb") is not None:
        need = max(need, e0["torch_peak_res_mb"])
        flags.append("capped E0 probe at C0 FAILED -> need = max(W+M, uncapped E0 reserved peak)")
    if e0c is None:
        flags.append("capped E0 probe missing: fit at C0 not verified")
    C = (int(c0) if float(c0).is_integer() else c0) if need <= c0 else ceil512(need)
    if C > cap_max:
        flags.append(f"max shape needs {need:.0f} MB > cap_max {cap_max}: C limited to {cap_max}; the max shape cannot be served eagerly under the cap (limit that shape's batch or accept failure)")
        C = int(cap_max)
    out.update(need_mb=need, C_USE=C, C_raised=(C != c0), flags=flags, e0_cap_status=(e0c or {}).get("status"), e0_uncap_torch_peak_res=(e0 or {}).get("torch_peak_res_mb"))
    return out


def delta_dev(recs: List[dict]) -> dict:
    cands = [(r["delta_dev_candidate"], r["label"] or r["tag"]) for r in recs if r.get("delta_dev_candidate") is not None]
    if not cands:
        return dict(delta_dev_max=None, delta_frozen=None)
    mx, who = max(cands)
    steady = [r["delta_steady_median"] for r in recs if r.get("delta_steady_median") is not None]
    return dict(delta_dev_max=mx, delta_dev_from=who, delta_frozen=int(math.ceil(mx)) + DELTA_PAD_MB, delta_steady_median_range=[min(steady), max(steady)] if steady else None,
                n_processes=len(cands))


def bpool(c: float, w: float, m: float, margin: float, default_tier: float = DEFAULT_TIER_MB) -> dict:
    raw = c - w - m - margin
    b = max(0.0, raw)
    capped = min(b, default_tier)
    fl = []
    if raw < 0:
        fl.append("C - W - M_work - margin < 0: no graph budget at all")
    if capped < TIGHT_TIER_MB:
        fl.append(f"B_pool {capped:.0f} MB < tight tier {TIGHT_TIER_MB} MB: the graph layer is nearly empty (Ours degrades towards eager)")
    if b > default_tier:
        fl.append("B_pool capped at the default tier (decisions identical to the uncapped arm)")
    return dict(B_POOL=round(capped, 2), B_pool_raw=round(raw, 2), flags=fl)


def margin_check(rec: Optional[dict], c: float, delta: float) -> dict:
    if rec is None:
        return dict(MARGIN_PASS=0, reasons=["no record"])
    why = []
    if rec["crash"]:
        why.append(f"status={rec['status']} ({rec['error'][:80]})")
    if rec.get("n_capture_failed"):
        why.append(f"n_capture_failed={rec['n_capture_failed']}")
    if rec.get("nvml_peak_mb") is None:
        why.append("no NVML peak")
    elif rec["nvml_peak_mb"] > c + delta:
        why.append(f"NVML peak {rec['nvml_peak_mb']:.0f} > C+delta {c + delta:.0f}")
    return dict(MARGIN_PASS=0 if why else 1, reasons=why, nvml_peak_mb=rec.get("nvml_peak_mb"), limit_mb=c + delta, n_capture_failed=rec.get("n_capture_failed"))


def select_g(L: Dict[str, dict]) -> dict:
    refs = {"ours": L.get("ours_cap_nogc"), "b1": L.get("b1_uncap")}
    res: dict = {}
    for g in ("0.6", "0.8"):
        entry, ok = {}, True
        for arm in ("ours", "b1"):
            r = L.get(f"{arm}_g{g}")
            ref = refs[arm]
            if r is None:
                entry[arm] = "missing"
                if arm == "ours":
                    ok = False
                continue
            bad = []
            try:
                hc = hash_compare(ref, r) if ref else dict(available=False)
            except TierFieldMissing as e:
                hc = dict(available=False)
                bad.append(f"TIER FIELD MISSING: {e}")
            if r["crash"]:
                bad.append(f"status={r['status']}")
            if r.get("calls_match_rows") is False:
                bad.append("run calls != ok rows")
            if hc.get("available") and hc["n_same_tier_diff"]:
                bad.append(f"{hc['n_same_tier_diff']} same-tier hash differences")
            ratio = (r["service_s"] / ref["service_s"]) if (ref and ref.get("service_s") and r.get("service_s")) else None
            entry[arm] = dict(ok=not bad, problems=bad, time_ratio=ratio, hash=hc, num_ooms=r.get("num_ooms"), num_alloc_retries=r.get("num_alloc_retries"), num_device_free=r.get("num_device_free"),
                              ref_num_device_free=(ref or {}).get("num_device_free"))
            ok = ok and not bad
        ratios = [e["time_ratio"] for e in entry.values() if isinstance(e, dict) and e.get("time_ratio")]
        entry["ok"] = ok
        entry["mean_time_ratio"] = sum(ratios) / len(ratios) if ratios else None
        res[g] = entry
    good = [(res[g]["mean_time_ratio"] if res[g]["mean_time_ratio"] is not None else 9e9, g) for g in res if res[g]["ok"]]
    return dict(G_SEL=min(good)[1] if good else "none", per_g=res)


RATIO_MAX_DEV = 1.02          # card sec 7.3 item 3 (1% below the confirmation line 1.03)
ARMS = (("ours", "Ours"), ("b4", "B4"))
MARGIN_FALLBACK_MB = 1024.0   # B_pool margin when no margin trial passed (same as the block's MARGIN0)


def arm_eval(L: Dict[str, dict], pre: str, C: int, delta: Optional[float]) -> dict:
    """One (arm, C) cell: capped process label <pre>_C<C> against <pre>_uncap."""
    capped, ref = L.get(f"{pre}_C{C}"), L.get(f"{pre}_uncap")
    out: dict = dict(arm=dict(ARMS)[pre], C=C, present=capped is not None, reasons=[])
    if capped is None:
        out.update(passed=False, peak_ok=False, crash_free=False, ratio_ok=False, reasons=["capped process missing (skipped or not run)"])
        return out
    peak = capped.get("nvml_peak_mb")
    limit = (C + delta) if delta is not None else None
    peak_ok = peak is not None and limit is not None and peak <= limit
    crash_free = not capped["crash"]
    ratio = (capped["service_s"] / ref["service_s"]) if (ref and ref.get("service_s") and capped.get("service_s") and not ref["crash"]) else None
    ratio_ok = ratio is not None and ratio <= RATIO_MAX_DEV
    why = []
    if not peak_ok:
        why.append("no NVML peak / delta" if (peak is None or limit is None) else f"NVML peak {peak:.0f} > C+delta {limit:.0f}")
    if not crash_free:
        why.append(f"crash: status={capped['status']} ({(capped['error'] or '')[:60]})")
    if ratio is None:
        why.append("no uncapped reference time" if not ref else "no service time")
    elif not ratio_ok:
        why.append(f"time ratio {ratio:.4f} > {RATIO_MAX_DEV}")
    out.update(passed=bool(peak_ok and crash_free and ratio_ok), peak_ok=bool(peak_ok), crash_free=crash_free, ratio_ok=bool(ratio_ok), reasons=why, nvml_peak_mb=peak, limit_mb=limit,
               torch_peak_res_mb=capped.get("torch_peak_res_mb"), time_ratio=ratio, budget_mb=capped.get("budget_mb"), n_capture_failed=capped.get("n_capture_failed"), n_denied=capped.get("n_denied"),
               uncapped_nvml_peak_mb=(ref or {}).get("nvml_peak_mb"), acts=((ref or {}).get("nvml_peak_mb") is not None and ref["nvml_peak_mb"] > C), m3=None, m3_error=None)
    if ref and not ref["crash"] and not capped["crash"]:
        try:
            h = hash_compare(ref, capped)
            out["m3"] = h if h.get("available") else None
        except TierFieldMissing as e:
            out["m3_error"] = str(e)
    return out


def select_c(L: Dict[str, dict], delta: Optional[float], need: Optional[float], cap_max: int = CAP_MAX_MB) -> dict:
    """Card sec 7.3 item 3: smallest eligible C where Ours AND B4 pass; none -> cap_max + NONE_PASSED."""
    cs = sorted({int(lab.rsplit("_C", 1)[1]) for lab in L if lab.rsplit("_C", 1)[0] in ("ours", "b4") and "_C" in lab and lab.rsplit("_C", 1)[1].isdigit()})
    per_c, sel = {}, None
    for C in cs:
        cells = {pre: arm_eval(L, pre, C, delta) for pre, _ in ARMS}
        eligible = need is None or C >= need
        passed = eligible and all(c["passed"] for c in cells.values())
        per_c[C] = dict(eligible=eligible, passed=passed, cells=cells)
        if sel is None and passed:
            sel = C
    return dict(C_SEL=sel if sel is not None else int(cap_max), none_passed=sel is None, candidates=cs, per_C=per_c)


def dev_summary(raw_root, c0: float = 14336.0) -> dict:
    recs = discover(raw_root)
    L = by_label(recs)
    pr = probes(L, c0)
    dd = delta_dev(recs)
    gs = select_g(L)
    delta = dd.get("delta_frozen")
    W, M, need = pr.get("W"), pr.get("M_WORK"), pr.get("need_mb")
    margins = sorted([(float(l.rsplit("_", 1)[1]), r) for l, r in L.items() if l.startswith("ours_margin_")], key=lambda t: t[0])
    msel, mtab = None, []
    for m, r in margins:
        Cm = r.get("cap_mb")
        mc = margin_check(r, Cm, delta) if (delta is not None and Cm is not None) else dict(MARGIN_PASS=0, reasons=["delta or cap unknown"])
        mtab.append(dict(margin=m, tag=r["tag"], cap_mb=Cm, budget_mb=r["budget_mb"], **mc))
        if msel is None and mc["MARGIN_PASS"]:
            msel = m
    margin_used = msel if msel is not None else MARGIN_FALLBACK_MB
    sc = select_c(L, delta, need)
    C = sc["C_SEL"]
    bp = bpool(C, W, M, margin_used) if (W is not None and M is not None) else None
    bp_per_c = {c: (bpool(c, W, M, margin_used)["B_POOL"] if (W is not None and M is not None) else None) for c in sc["candidates"]}
    det = None
    if L.get("ours_uncap") and L.get("ours_uncap_rep"):
        try:
            det = hash_compare(L["ours_uncap"], L["ours_uncap_rep"])
        except TierFieldMissing as e:
            det = dict(available=False, error=str(e))
        det["time_ratio_rep_vs_first"] = (L["ours_uncap_rep"]["service_s"] / L["ours_uncap"]["service_s"]) if (L["ours_uncap"].get("service_s") and L["ours_uncap_rep"].get("service_s")) else None
    errors = [f"{c['arm']} C={cc}: {c['m3_error']}" for cc, v in sc["per_C"].items() for c in v["cells"].values() if c.get("m3_error")]
    errors += [f"g={g} {arm}: {p}" for g, e in gs["per_g"].items() for arm in ("ours", "b1") if isinstance(e.get(arm), dict) for p in e[arm]["problems"] if "TIER FIELD MISSING" in p]
    if det and det.get("error"):
        errors.append("determinism control: " + det["error"])
    return dict(c0=c0, probes=pr, delta=dd, g=gs, margin_table=mtab, margin_selected=msel, margin_used=margin_used, B_pool=bp, B_pool_per_C=bp_per_c, c_selection=sc, determinism_uncapped_repeat=det,
                n_processes=len(recs), crashes=[r["tag"] + ":" + str(r["status"]) for r in recs if r["crash"]], errors=errors,
                frozen_values_proposal=dict(C=C, delta=delta, g=gs["G_SEL"], margin=margin_used, B_pool=(bp or {}).get("B_POOL"), none_passed=sc["none_passed"]))


TABLE_COLS = ["label", "tag", "policy", "trace", "cap_mb", "gc", "budget_mb", "status", "crash_oom", "service_s", "n_requests", "n_capture_failed", "n_denied", "w_alloc_mb", "torch_peak_alloc_mb", "torch_peak_res_mb",
              "nvml_peak_mb", "nvml_baseline_mb", "delta_peak_gap", "delta_all_max", "delta_all_median", "delta_steady_max", "delta_steady_median", "num_ooms", "num_alloc_retries", "num_device_free", "nvml_hz",
              "nvml_max_gap_s", "calls_match_rows", "event_counts"]
C_TABLE_COLS = ["C", "arm", "eligible", "cell_passed", "C_passed", "budget_mb", "uncapped_nvml_peak_mb", "acts", "nvml_peak_mb", "limit_mb", "peak_ok", "crash_free", "time_ratio", "ratio_ok", "n_capture_failed", "n_denied",
                "n_tier_changed_all", "n_tier_changed_hashed", "n_same_tier", "n_same_tier_equal", "n_same_tier_diff", "m3_error", "reasons"]


def c_table_rows(S: dict) -> List[dict]:
    rows = []
    for C, v in S["c_selection"]["per_C"].items():
        for pre, _ in ARMS:
            c = v["cells"][pre]
            m3 = c.get("m3") or {}
            rows.append(dict(C=C, arm=c["arm"], eligible=v["eligible"], cell_passed=c["passed"], C_passed=v["passed"], budget_mb=c.get("budget_mb"), uncapped_nvml_peak_mb=c.get("uncapped_nvml_peak_mb"), acts=c.get("acts"),
                             nvml_peak_mb=c.get("nvml_peak_mb"), limit_mb=c.get("limit_mb"), peak_ok=c["peak_ok"], crash_free=c["crash_free"], time_ratio=c.get("time_ratio"), ratio_ok=c["ratio_ok"],
                             n_capture_failed=c.get("n_capture_failed"), n_denied=c.get("n_denied"), n_tier_changed_all=m3.get("n_tier_changed_all"), n_tier_changed_hashed=m3.get("n_tier_changed_hashed"),
                             n_same_tier=m3.get("n_same_tier"), n_same_tier_equal=m3.get("n_same_tier_equal"), n_same_tier_diff=m3.get("n_same_tier_diff"), m3_error=c.get("m3_error"), reasons="; ".join(c["reasons"])))
    return rows


def _num(v):
    if v is None:
        return "None"
    return str(int(v)) if float(v).is_integer() else str(v)


def recommendation_text(S: dict) -> str:
    pr, dd, gs, bp, sc = S["probes"], S["delta"], S["g"], S["B_pool"], S["c_selection"]
    f = lambda v, n=0: "n/a" if v is None else f"{v:.{n}f}"  # noqa: E731
    L = []
    L.append("X2 development block v2 - recommendation (numbers from this block only; the card sec 6 is filled by the main session)")
    L.append(f"  W (weights, torch allocated after load) = {f(pr.get('W'), 1)} MB ; M_e0 = {f(pr.get('M_e0'), 1)} ; M_c1g1 (above weights and graph pool) = {f(pr.get('M_c1g1'), 1)} ; M_work_max = {f(pr.get('M_WORK'), 1)} MB ; need = W + M_work_max = {f(pr.get('need_mb'), 1)} MB")
    L.append(f"  capped E0 probe at C0={S['c0']:.0f}: status {pr.get('e0_cap_status')}")
    for fl in pr.get("flags", []):
        L.append(f"    FLAG {fl}")
    L.append(f"  delta: dev max {f(dd.get('delta_dev_max'), 1)} MB (from {dd.get('delta_dev_from')}) -> frozen delta = {dd.get('delta_frozen')} MB (+{DELTA_PAD_MB}); steady medians {dd.get('delta_steady_median_range')}; processes {dd.get('n_processes')}")
    L.append(f"  g: selected {gs['G_SEL']}  " + "; ".join(f"g={g}: ok={e['ok']} mean time ratio {f(e['mean_time_ratio'], 4)}" for g, e in gs["per_g"].items()))
    for g, e in gs["per_g"].items():
        for arm in ("ours", "b1"):
            if isinstance(e.get(arm), dict) and e[arm]["problems"]:
                L.append(f"    g={g} {arm}: " + "; ".join(e[arm]["problems"]))
    L.append(f"  margin (trials at the middle C): selected {S['margin_selected']}  table " + "; ".join(f"m={t['margin']:.0f}@C{_num(t.get('cap_mb'))}: pass={t['MARGIN_PASS']} {t.get('reasons')}" for t in S["margin_table"]))
    if S["margin_selected"] is None:
        L.append(f"    FLAG no margin trial passed: margin for B_pool falls back to {S['margin_used']:.0f} MB")
    L.append(f"  C selection (rule: smallest eligible C where Ours AND B4 both: NVML peak <= C+delta, 0 crashes, time ratio capped/uncapped <= {RATIO_MAX_DEV}; none -> 18432 + NONE_PASSED)")
    for C in sc["candidates"]:
        v = sc["per_C"][C]
        L.append(f"   C={C}  eligible(C>=need)={v['eligible']}  B_pool(formula)={S['B_pool_per_C'].get(C)}  -> {'PASS' if v['passed'] else 'fail'}")
        for pre, _ in ARMS:
            c = v["cells"][pre]
            m3 = c.get("m3") or {}
            L.append(f"      {c['arm']:<4}: peak {f(c.get('nvml_peak_mb'))} (limit {f(c.get('limit_mb'))}) ratio {f(c.get('time_ratio'), 4)} capture_failed {c.get('n_capture_failed')} denied {c.get('n_denied')} cap acts (uncapped peak {f(c.get('uncapped_nvml_peak_mb'))}) {c.get('acts')} | "
                     f"tier-changed requests {m3.get('n_tier_changed_all', 'n/a')} (hashed {m3.get('n_tier_changed_hashed', 'n/a')}) same-tier hashes equal {m3.get('n_same_tier_equal', 'n/a')}/{m3.get('n_same_tier', 'n/a')} "
                     f"-> {'pass' if c['passed'] else 'FAIL: ' + '; '.join(c['reasons'])}")
    if sc["none_passed"]:
        L.append("  FLAG NONE_PASSED: no candidate C satisfied the rule -> C = 18432 (card sec 7.3 item 3); report the failures as they are")
    if bp:
        L.append(f"  B_pool at the selected C (C - W - M_work - margin, capped at the default tier {DEFAULT_TIER_MB}) = {bp['B_POOL']} MB (raw {bp['B_pool_raw']})")
        for fl in bp["flags"]:
            L.append(f"    FLAG {fl}")
    d = S.get("determinism_uncapped_repeat")
    if d and d.get("available"):
        L.append(f"  determinism control (Ours uncapped, run twice): {d['n_equal'] if 'n_equal' in d else d['n_common'] - d['n_diff']}/{d['n_common']} hashes equal, {d['n_diff']} differ, service time of the repeat / first = {f(d.get('time_ratio_rep_vs_first'), 4)}")
    L.append(f"  crashes: {S['crashes'] or 'none'}")
    for e in S["errors"]:
        L.append(f"  ERROR (M3 tier field): {e}")
    fv = S["frozen_values_proposal"]
    L.append("C=%s delta=%s g=%s margin=%s B_pool=%s" % (_num(fv["C"]), _num(fv["delta"]), fv["g"], _num(fv["margin"]), _num(fv["B_pool"])))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------ CLI
def _kv(d: dict, keys):
    for k in keys:
        print(f"{k}={d.get(k)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proc"); p.add_argument("--dir", required=True); p.add_argument("--tag", required=True)
    p = sub.add_parser("probes"); p.add_argument("--raw", required=True); p.add_argument("--c0", type=float, default=14336.0); p.add_argument("--cap-max", type=float, default=CAP_MAX_MB)
    p = sub.add_parser("bpool"); [p.add_argument(f"--{k}", type=float, required=True) for k in ("c", "w", "m", "margin")]
    p = sub.add_parser("margin-check"); p.add_argument("--raw", required=True); p.add_argument("--label", required=True); p.add_argument("--c", type=float, required=True); p.add_argument("--delta", type=float, required=True)
    p = sub.add_parser("select-g"); p.add_argument("--raw", required=True)
    p = sub.add_parser("delta"); p.add_argument("--raw", required=True)
    p = sub.add_parser("dev-summary"); p.add_argument("--raw", required=True); p.add_argument("--out", required=True); p.add_argument("--c0", type=float, default=14336.0)
    p = sub.add_parser("hashcmp"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "proc":
        r = load_proc(a.dir, a.tag)
        print(json.dumps(r, indent=1, default=str))
    elif a.cmd == "probes":
        r = probes(by_label(discover(a.raw)), a.c0, a.cap_max)
        _kv({**r, "W_MB": r.get("W"), "M_WORK_MB": r.get("M_WORK"), "NEED_MB": r.get("need_mb")}, ("W_MB", "M_WORK_MB", "NEED_MB", "C_USE"))
        print("PROBES_JSON=" + json.dumps(r, default=str))
    elif a.cmd == "bpool":
        r = bpool(a.c, a.w, a.m, a.margin)
        print(f"B_POOL={r['B_POOL']}")
        print("BPOOL_JSON=" + json.dumps(r))
    elif a.cmd == "margin-check":
        L = by_label(discover(a.raw))
        r = margin_check(L.get(a.label), a.c, a.delta)
        print(f"MARGIN_PASS={r['MARGIN_PASS']}")
        print("MARGIN_JSON=" + json.dumps(r, default=str))
    elif a.cmd == "select-g":
        r = select_g(by_label(discover(a.raw)))
        print(f"G_SEL={r['G_SEL']}")
        print("G_JSON=" + json.dumps(r, default=str))
    elif a.cmd == "delta":
        r = delta_dev(discover(a.raw))
        print(f"DELTA_DEV={r.get('delta_dev_max')}")
        print(f"DELTA_FROZEN={r.get('delta_frozen')}")
    elif a.cmd == "dev-summary":
        S = dev_summary(a.raw, a.c0)
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dev_summary.json").write_text(json.dumps(S, indent=1, default=str))
        recs = discover(a.raw)
        with open(out / "dev_table.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TABLE_COLS)
            for r in recs:
                w.writerow([json.dumps(r[c]) if isinstance(r.get(c), dict) else r.get(c) for c in TABLE_COLS])
        with open(out / "dev_c_table.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(C_TABLE_COLS)
            for r in c_table_rows(S):
                w.writerow([r.get(c) for c in C_TABLE_COLS])
        txt = recommendation_text(S)
        (out / "DEV_RECOMMENDATION.txt").write_text(txt)
        print(txt)
        if S["errors"]:
            print("ERRORS: tier field missing / unusable (see DEV_RECOMMENDATION.txt)", file=sys.stderr)
            return 2
    elif a.cmd == "hashcmp":
        da, ta = a.a.rsplit(":", 1)
        db, tb = a.b.rsplit(":", 1)
        try:
            print(json.dumps(hash_compare(load_proc(da, ta), load_proc(db, tb)), indent=1))
        except TierFieldMissing as e:
            print(f"ERROR TierFieldMissing: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
