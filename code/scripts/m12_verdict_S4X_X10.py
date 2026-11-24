"""S4-X X10 (CPU): verdict M1 / M2 (judged) and M3 (reported) for the SD-Turbo mechanism check.
Rules = goals/S4X_X10_second_model_mechanism_prereg_v1.md sec 2 / 3 (written BEFORE any X10 number).  Judged on 512x512 B1 (no CFG) only; 384x384 B1 and 512x512 B4 only reported.
  M1  compile saves small kernels (Nsight kernel classes of scripts/ceiling_S4X.py, E0 vs C1, per request):
        (a) C1 GPU kernel total < E0 kernel total (net reduction D > 0)
        (b) |big_C1 - big_E0| / big_E0 <= 3 %          big = conv + gemm + attention
        (c) small reduction >= 60 % of D                small = layout_convert + norm_reduce + elementwise_copy, reduction = small_E0 - small_C1
            [card wording taken literally: the Inductor-generated `triton_fused` and `other` classes are NOT in "small"; the alternative reading
             small' = all non-big classes (incl. triton_fused, other) is computed and printed too (`share_small_all_nonbig`), NOT judged]
  M2  capture saves about the GPU idle time: (E0 - G1) steady wall ms  /  E0 GPU idle ms per request in [0.5, 1.5].
        idle = the S4X derived idle (scripts/ceiling_S4X.py q1_accounting: untraced E0 steady wall - traced GPU busy union of the E0 Nsight trace; the plain traced idle is inflated by
        tracing, see results/S2/nsight/README_nsight.txt) -> judged; the plain traced idle ratio is printed too (not judged).
  M3  G1 relative gain r = (E0 - G1) / E0 over the three shapes is non-increasing with E0 latency (reported only; shapes whose G1 equivalence fails are left out).
  holds = M1 and M2 | partial = exactly one | fails = neither.   Reported only: C1 and C1+G1 speed-up over E0, compile time (empty cache = pre-warm process; disk cache = steady processes),
  capture time, break-even request count (same formula as p24.stats / S2: n = 1 + max(F_arm - F_E0, 0) / ((E0 - arm) / 1000), F = first-request seconds, empty- and disk-cache tier).
Validity (card sec 3; any failure -> INVALID, nothing re-interpreted): V1 equivalence C1 and G1 pass at 512x512 B1 (result file status ok, equiv_pass == 1);
  V2 profile traces of E0 and C1 complete (>= 3 NVTX request ranges, kernels > 0); V3 every steady process of the 512x512 B1 cell has status ok and each arm keeps >= 3 processes after the
  p24.stats exclusions (contended / host load > 16 / not P-core / recompile limit); V4 pinned fallback count 0 in every record and at least one record per steady / pre-warm process.
Usage: "$PROJ_PYTHON" -B -m scripts.m12_verdict_S4X_X10 [--steady DIR --prewarm DIR --equiv DIR --profile-json FILE --log-dir DIR --out-dir DIR]
       "$PROJ_PYTHON" -B -m scripts.m12_verdict_S4X_X10 --synthetic      (self-test in a temp dir: holds / partial / fails / INVALID x4 / M3 monotone and not)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from p24 import stats

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
X10 = ROOT / "results/S4X/x10"
JUDGED = (512, 512, 1, 0)
SHAPES = [(384, 384, 1, 0), (512, 512, 1, 0), (512, 512, 4, 0)]
BIG = ("conv", "gemm", "attention")
SMALL = ("layout_convert", "norm_reduce", "elementwise_copy")
BIG_TOL, SMALL_SHARE, RATIO_LO, RATIO_HI = 0.03, 0.60, 0.5, 1.5
MIN_PROC = 3


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def tag_of(h, w, b, c=0):
    return f"{h}x{w}_b{b}_{'cfg' if c else 'nocfg'}"


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ------------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(steady: pathlib.Path, prewarm: pathlib.Path, equiv_dir: pathlib.Path, profile_json: pathlib.Path, log_dir: pathlib.Path) -> dict:
    inv, notes = [], []
    raw = stats.load(steady)
    if raw.empty:
        return dict(verdict="INVALID", invalid_reasons=["no steady CSV rows"], notes=[])
    summ = stats.summarize(raw)
    excl = stats.excluded_processes(raw)
    S = {(r.arm, r.H, r.W, r.B, r.cfg): r for r in summ.itertuples()}

    def st(arm, key):
        r = S.get((arm,) + key)
        return None if r is None else float(r.steady_wall_ms)

    # ---- V3 steady processes (512x512 B1 judged cell)
    jr = raw[(raw.H == JUDGED[0]) & (raw.W == JUDGED[1]) & (raw.B == JUDGED[2]) & (raw.cfg == JUDGED[3])]
    for arm in ("E0", "G1", "C1", "C1+G1"):
        sub = jr[jr.arm == arm]
        if sub.empty:
            inv.append(f"V3 no processes of {arm} at 512x512 B1"); continue
        bad = sub[sub.status != "ok"]
        if len(bad):
            inv.append(f"V3 {arm} 512x512 B1: processes with status != ok: {sorted(bad.proc_id.unique().tolist())}")
        r = S.get((arm,) + JUDGED)
        n = 0 if r is None else int(r.n_proc)
        if n < MIN_PROC:
            inv.append(f"V3 {arm} 512x512 B1: {n} processes after exclusions (< {MIN_PROC})")
    # ---- V4 pinned records
    pc = sorted(pathlib.Path(log_dir).glob("*.pinned_fallback_count.txt")) if pathlib.Path(log_dir).exists() else []
    nonzero = [p.name for p in pc if p.read_text().strip() != "0"]
    if nonzero:
        inv.append(f"V4 pinned fallback > 0 in {nonzero[:5]}")
    pw_raw = stats.load(prewarm) if pathlib.Path(prewarm).exists() else pd.DataFrame()
    n_proc_expected = len(raw.groupby(["arm", "H", "W", "B", "cfg", "proc_id"])) + (len(pw_raw.groupby(["arm", "H", "W", "B", "cfg", "proc_id"])) if len(pw_raw) else 0)
    if len(pc) < n_proc_expected:
        inv.append(f"V4 only {len(pc)} pinned-count records for {n_proc_expected} steady + pre-warm processes")
    # ---- V1 equivalence
    eq = {}
    for arm in ("C1", "G1", "C1+G1"):
        f = pathlib.Path(equiv_dir) / f"{arm}.json"
        if f.exists():
            eq[arm] = json.loads(f.read_text())
    for arm in ("C1", "G1"):
        e = eq.get(arm)
        if e is None or e.get("status") != "ok":
            inv.append(f"V1 equivalence result of {arm} missing / not ok"); continue
        bk = e.get("by_key", {}).get(tag_of(*JUDGED))
        if not bk or int(bk.get("equiv_pass", 0)) != 1:
            inv.append(f"V1 equivalence {arm} at 512x512 B1 not passed ({None if not bk else dict(rmse_ratio=bk.get('rmse_ratio'), lpips=bk.get('lpips_vs_E0'))})")
    eq_pass = {arm: {k: int(v.get("equiv_pass", 0)) for k, v in e.get("by_key", {}).items()} for arm, e in eq.items()}
    # ---- V2 profile
    prof = json.loads(pathlib.Path(profile_json).read_text()) if pathlib.Path(profile_json).exists() else {"arms": {}}
    pa = prof.get("arms", {})
    for arm in ("E0", "C1"):
        if not pa.get(arm, {}).get("complete"):
            inv.append(f"V2 profile trace of {arm} missing / incomplete ({pa.get(arm, {}).get('error', '')})")
    res = dict(judged_cell="512x512 B1 nocfg", invalid_reasons=inv, n_excluded_processes=int(len(excl)),
               excluded=excl.to_dict("records") if len(excl) else [], equiv_pass=eq_pass, notes=notes)
    if "diffusers_check" in json.loads(pathlib.Path(equiv_dir, "E0.json").read_text() if pathlib.Path(equiv_dir, "E0.json").exists() else "{}"):
        res["diffusers_check_E0"] = json.loads(pathlib.Path(equiv_dir, "E0.json").read_text())["diffusers_check"]
    # ---- steady table (all shapes, reported)
    rows = []
    for key in SHAPES:
        e0, g1, c1, cg = (st(a, key) for a in ("E0", "G1", "C1", "C1+G1"))
        rows.append(dict(shape=tag_of(*key), E0_ms=e0, G1_ms=g1, C1_ms=c1, C1G1_ms=cg,
                         r_G1=(None if e0 is None or g1 is None else (e0 - g1) / e0),
                         speedup_C1=(None if e0 is None or c1 is None else e0 / c1), speedup_C1G1=(None if e0 is None or cg is None else e0 / cg),
                         n_proc={a: (int(S[(a,) + key].n_proc) if (a,) + key in S else 0) for a in ("E0", "G1", "C1", "C1+G1")}))
    res["steady"] = rows
    # ---- one-time costs and break-even (reported)
    onetime = []
    for key in SHAPES:
        for arm in ("G1", "C1", "C1+G1"):
            r = S.get((arm,) + key)
            if r is None:
                continue
            d = {"shape": tag_of(*key), "arm": arm}
            for c in ("compile_s", "compile_and_warmup_s", "capture_s", "first_req_s"):
                v = getattr(r, f"{c}_median_cold_diskcache", None) if c != "first_req_s" else getattr(r, "first_req_s_diskcache", None)
                d[f"{c}_disk"] = _f(v)
            d["breakeven_n_disk"] = _f(getattr(r, "breakeven_n_diskcache", None))
            pwv = {}
            if len(pw_raw):
                p = pw_raw[(pw_raw.arm == arm) & (pw_raw.H == key[0]) & (pw_raw.W == key[1]) & (pw_raw.B == key[2]) & (pw_raw.cfg == key[3]) & (pw_raw.status == "ok")]
                p = p.groupby("proc_id").first() if len(p) else p
                if len(p):
                    for c in ("compile_s", "compile_and_warmup_s", "capture_s", "first_req_s"):
                        if c in p:
                            pwv[c] = float(p[c].median())
            for c in ("compile_s", "compile_and_warmup_s", "capture_s", "first_req_s"):
                d[f"{c}_empty"] = _f(pwv.get(c))
            e0 = S.get(("E0",) + key)
            if e0 is not None and "first_req_s" in pwv:
                sav = (float(e0.steady_wall_ms) - float(r.steady_wall_ms)) / 1e3
                fe0 = float(e0.first_req_s_diskcache)
                d["breakeven_n_empty"] = (float("inf") if sav <= 0 else 1 + max(pwv["first_req_s"] - fe0, 0.0) / sav)
                d["breakeven_n_empty"] = None if (isinstance(d["breakeven_n_empty"], float) and math.isinf(d["breakeven_n_empty"])) else d["breakeven_n_empty"]
            onetime.append(d)
    res["one_time"] = onetime
    # ---- M1
    m1 = dict(judged=True)
    if pa.get("E0", {}).get("complete") and pa.get("C1", {}).get("complete"):
        e, c = pa["E0"], pa["C1"]
        D = e["kernel_total_ms"] - c["kernel_total_ms"]
        bigE, bigC = sum(e["ms"][k] for k in BIG), sum(c["ms"][k] for k in BIG)
        smE, smC = sum(e["ms"][k] for k in SMALL), sum(c["ms"][k] for k in SMALL)
        nbE = e["kernel_total_ms"] - bigE
        nbC = c["kernel_total_ms"] - bigC
        m1.update(E0_kernel_total_ms=e["kernel_total_ms"], C1_kernel_total_ms=c["kernel_total_ms"], net_reduction_ms=D,
                  big_E0_ms=bigE, big_C1_ms=bigC, big_change_pct=(bigC - bigE) / bigE * 100 if bigE else None,
                  small_E0_ms=smE, small_C1_ms=smC, small_reduction_ms=smE - smC, share_small=((smE - smC) / D if D > 0 else None),
                  share_small_all_nonbig=((nbE - nbC) / D if D > 0 else None),
                  per_class_delta_E0_minus_C1_ms={k: e["ms"][k] - c["ms"][k] for k in e["ms"]}, layout_convert_share=((e["ms"]["layout_convert"] - c["ms"]["layout_convert"]) / D if D > 0 else None),
                  other_fraction_E0=e.get("other_fraction"), other_fraction_C1=c.get("other_fraction"))
        m1["a_net_reduction_positive"] = bool(D > 0)
        m1["b_big_unchanged"] = bool(bigE > 0 and abs(bigC - bigE) / bigE <= BIG_TOL)
        m1["c_small_share"] = bool(D > 0 and (smE - smC) >= SMALL_SHARE * D)
        m1["holds"] = bool(m1["a_net_reduction_positive"] and m1["b_big_unchanged"] and m1["c_small_share"])
        if max(e.get("other_fraction") or 0, c.get("other_fraction") or 0) > 0.05:
            notes.append("classification gap: class `other` > 5 % of kernel time in E0 or C1 (see profile_summary.json top_other) - report to the main session")
    else:
        m1["holds"] = None
    res["M1"] = m1
    # ---- M2
    m2 = dict(judged=True)
    e0, g1 = st("E0", JUDGED), st("G1", JUDGED)
    if e0 is not None and g1 is not None and pa.get("E0", {}).get("complete"):
        busy, twall = pa["E0"]["busy_ms"], pa["E0"]["traced_wall_ms"]
        idle_d, idle_t = e0 - busy, twall - busy
        sav = e0 - g1
        m2.update(E0_steady_ms=e0, G1_steady_ms=g1, saving_ms=sav, E0_traced_busy_ms=busy, E0_traced_wall_ms=twall, idle_derived_ms=idle_d, idle_traced_ms=idle_t,
                  ratio_derived=(sav / idle_d if idle_d > 0 else None), ratio_traced=(sav / idle_t if idle_t > 0 else None))
        m2["holds"] = bool(idle_d > 0 and RATIO_LO <= sav / idle_d <= RATIO_HI)
    else:
        m2["holds"] = None
    res["M2"] = m2
    # ---- M3
    pts = []
    for key in SHAPES:
        a, b = st("E0", key), st("G1", key)
        if a is None or b is None:
            continue
        if eq_pass.get("G1", {}).get(tag_of(*key), 1) != 1:
            notes.append(f"M3: {tag_of(*key)} left out (G1 equivalence failed)"); continue
        pts.append((a, (a - b) / a, tag_of(*key)))
    pts.sort()
    res["M3"] = dict(judged=False, points=[dict(shape=s, E0_ms=a, r=r) for a, r, s in pts],
                     monotone_non_increasing=(all(pts[i + 1][1] <= pts[i][1] for i in range(len(pts) - 1)) if len(pts) >= 2 else None))
    # ---- verdict
    if inv:
        res["verdict"] = "INVALID"
    elif m1["holds"] is None or m2["holds"] is None:
        res["verdict"] = "INVALID"
        inv.append("M1 / M2 inputs missing")
    else:
        res["verdict"] = "holds" if (m1["holds"] and m2["holds"]) else ("partial" if (m1["holds"] or m2["holds"]) else "fails")
    return res


def render(res: dict) -> str:
    L = [f"X10 verdict: {res['verdict'].upper()}   (judged cell {res.get('judged_cell')})"]
    for r in res.get("invalid_reasons", []):
        L.append(f"  INVALID: {r}")
    m1, m2, m3 = res.get("M1", {}), res.get("M2", {}), res.get("M3", {})
    if m1.get("E0_kernel_total_ms") is not None:
        L.append(f"M1 {m1['holds']}: kernel total E0 {m1['E0_kernel_total_ms']:.2f} -> C1 {m1['C1_kernel_total_ms']:.2f} ms (D {m1['net_reduction_ms']:.2f}); big {m1['big_E0_ms']:.2f} -> {m1['big_C1_ms']:.2f} ({m1['big_change_pct']:+.2f} %, line 3 %); "
                 f"small3 reduction {m1['small_reduction_ms']:.2f} ms = {m1['share_small'] * 100:.0f} % of D (line 60 %); all-non-big reading {m1['share_small_all_nonbig'] * 100:.0f} %; layout_convert alone {m1['layout_convert_share'] * 100:.0f} %")
    if m2.get("saving_ms") is not None:
        rd = m2["ratio_derived"]
        rt = m2["ratio_traced"]
        L.append(f"M2 {m2['holds']}: (E0-G1) {m2['saving_ms']:.2f} ms / derived idle {m2['idle_derived_ms']:.2f} ms = {('%.2f' % rd) if rd is not None else 'n/a'} (line [0.5, 1.5]); traced idle {m2['idle_traced_ms']:.2f} ms ratio {('%.2f' % rt) if rt is not None else 'n/a'} (not judged)")
    L.append("M3 (reported): " + "; ".join(f"{p['shape']} E0 {p['E0_ms']:.1f} ms r={p['r'] * 100:.1f}%" for p in m3.get("points", [])) + f" -> non-increasing: {m3.get('monotone_non_increasing')}")
    for r in res.get("steady", []):
        g = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        L.append(f"steady {r['shape']}: E0 {g(r['E0_ms'])} G1 {g(r['G1_ms'])} C1 {g(r['C1_ms'])} C1+G1 {g(r['C1G1_ms'])} ms; speedup C1 {g(r['speedup_C1'])} C1+G1 {g(r['speedup_C1G1'])}; r_G1 {g(r['r_G1'])}; n_proc {r['n_proc']}")
    for o in res.get("one_time", []):
        L.append("one-time " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()}))
    for n in res.get("notes", []):
        L.append("NOTE " + n)
    return "\n".join(L)


def write_outputs(res: dict, out: pathlib.Path, synthetic: bool):
    out.mkdir(parents=True, exist_ok=True)
    res = dict(res, time=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=bool(synthetic), script_sha16=sha16(pathlib.Path(__file__).resolve()))
    jf, tf = out / "verdict_X10.json", out / "verdict_X10.txt"
    jf.write_text(json.dumps(res, indent=1, default=str))
    tf.write_text(render(res) + "\n")
    if not synthetic:
        for f in (jf, tf):
            os.chmod(f, 0o444)
    return res


# ------------------------------------------------------------------------------------------------------------------ synthetic self-test
def _csv(path, arm, key, pid, wall, tier="cold_diskcache", status="ok", n=20, first_req=10.0, compile_s=float("nan"), capture_s=float("nan")):
    H, W, B, cfg = key
    rng = np.random.default_rng(hash((arm, key, pid)) % 2**32)
    rows = []
    for r in range(n):
        rows.append(dict(arm=arm, H=H, W=W, B=B, cfg=cfg, proc_id=pid, cold_tier=tier, status=status, rep=r, wall_ms=wall * (1 + rng.normal(0, 0.003)),
                         gpu_ms=wall * 0.97, first_req_s=first_req, prep_json="{}", peak_alloc_mb=3000.0, contended=0, load1_start=3.0, load1_end=3.0,
                         row_kind="steady", pcore_pinned=1, recompile_limit_hit=0, req_idx=4 + r, compile_s=compile_s, compile_and_warmup_s=compile_s,
                         capture_s=capture_s, tier_violation=0))
    pd.DataFrame(rows).to_csv(path / f"{arm}_{tag_of(H, W, B, cfg)}_p{pid}.csv", index=False)


def _profile(E0_total=36.0, big_E=20.0, small_E=15.0, C_total=29.0, big_C=20.2, small_C=2.0, busy=38.0, wall=41.0, complete=True):
    def arm(total, big, small, b, w):
        cats = dict(conv=big * 0.5, gemm=big * 0.3, attention=big * 0.2, layout_convert=small * 0.5, norm_reduce=small * 0.3, elementwise_copy=small * 0.2, triton_fused=0.0, other=0.0)
        cats["triton_fused"] = total - big - small
        return dict(complete=complete, n_requests=3, kernel_total_ms=total, busy_ms=b, traced_wall_ms=w, ms=cats, big_ms=big, other_fraction=0.0, traced_idle_ms=w - b)
    return dict(arms={"E0": arm(E0_total, big_E, small_E, busy, wall), "C1": arm(C_total, big_C, small_C, busy - 7, wall - 10)})


def _make(td: pathlib.Path, e0=40.0, g1=38.0, c1=33.0, cg=32.0, procs=5, g1_procs=None, equiv_c1=True, pinned=0, profile=None, shapes_extra=None):
    st, pw, eqd, lg = td / "steady", td / "prewarm", td / "equiv", td / "logs"
    for d in (st, pw, eqd, lg):
        d.mkdir(parents=True)
    scale = {(384, 384, 1, 0): 0.7, (512, 512, 1, 0): 1.0, (512, 512, 4, 0): 2.4}
    n = 0
    for key, s in scale.items():
        for arm, w in (("E0", e0), ("G1", g1), ("C1", c1), ("C1+G1", cg)):
            if shapes_extra and (arm, key) in shapes_extra:
                ww = shapes_extra[(arm, key)]
            else:
                ww = w * s
            for pid in range(g1_procs if (arm == "G1" and key == (512, 512, 1, 0) and g1_procs is not None) else procs):
                _csv(st, arm, key, pid, ww, first_req=(8.0 if arm in ("E0", "G1") else 15.0), compile_s=(12.0 if "C1" in arm else float("nan")), capture_s=(0.4 if "G1" in arm else float("nan")))
                (lg / f"{arm}_{tag_of(*key)}_p{pid}.pinned_fallback_count.txt").write_text(f"{pinned if n == 0 else 0}\n")
                n += 1
    for arm in ("C1", "C1+G1"):
        for key in scale:
            _csv(pw, arm, key, 0, 50.0, tier="prewarm", n=1, first_req=60.0, compile_s=55.0)
            (lg / f"pw_{arm}_{tag_of(*key)}.pinned_fallback_count.txt").write_text("0\n")
    for arm in ("C1", "G1", "C1+G1"):
        ok = equiv_c1 or arm != "C1"
        (eqd / f"{arm}.json").write_text(json.dumps(dict(arm=arm, status="ok", by_key={tag_of(*k): dict(equiv_pass=int(ok), rmse_ratio=1.1 if ok else 3.0, lpips_vs_E0=0.003 if ok else 0.05) for k in scale})))
    (td / "profile_summary.json").write_text(json.dumps(profile or _profile()))
    return st, pw, eqd, td / "profile_summary.json", lg


def selftest() -> int:
    cases = {
        "holds": (dict(), "holds"),
        "partial (M1 ok, M2 saving too large)": (dict(g1=30.0), "partial"),
        "partial (M2 ok, M1 big ops change 10%)": (dict(profile=_profile(big_C=22.0, C_total=30.8)), "partial"),
        "fails (both)": (dict(g1=30.0, profile=_profile(big_C=22.0, C_total=30.8)), "fails"),
        "INVALID (C1 equivalence fails)": (dict(equiv_c1=False), "INVALID"),
        "INVALID (pinned > 0)": (dict(pinned=1), "INVALID"),
        "INVALID (profile incomplete)": (dict(profile=_profile(complete=False)), "INVALID"),
        "INVALID (G1 only 2 processes)": (dict(g1_procs=2), "INVALID"),
        "M3 monotone (r falls with E0 latency)": (dict(shapes_extra={("G1", (384, 384, 1, 0)): 24.0, ("G1", (512, 512, 4, 0)): 94.0}), "holds"),
        "M3 not monotone (reported only, verdict unchanged)": (dict(shapes_extra={("G1", (384, 384, 1, 0)): 27.5, ("G1", (512, 512, 4, 0)): 80.0}), "holds"),
    }
    bad = 0
    for name, (kw, want) in cases.items():
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            args = _make(td, **kw)
            r = evaluate(*args)
            r2 = write_outputs(r, td / "out", synthetic=True)
            m3 = r["M3"]["monotone_non_increasing"] if "M3" in r else None
            ok = r["verdict"] == want
            if name.startswith("M3 monotone"):
                ok = ok and m3 is True
            if name.startswith("M3 not"):
                ok = ok and m3 is False
            bad += 0 if ok else 1
            extra = "" if r["verdict"] == "INVALID" else f" M1={r['M1']['holds']} M2={r['M2']['holds']} ratio={r['M2'].get('ratio_derived') and round(r['M2']['ratio_derived'], 2)} M3mono={m3}"
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: verdict={r['verdict']} (want {want}){extra}" + (f" invalid={r['invalid_reasons']}" if r['verdict'] == 'INVALID' else ""))
    print("SYNTHETIC SELFTEST", "OK" if bad == 0 else f"FAILED ({bad})")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steady", default=str(X10 / "steady"))
    ap.add_argument("--prewarm", default=str(X10 / "prewarm"))
    ap.add_argument("--equiv", default=str(X10 / "equiv"))
    ap.add_argument("--profile-json", default=str(X10 / "profile/profile_summary.json"))
    ap.add_argument("--log-dir", default=str(X10 / "logs"))
    ap.add_argument("--out-dir", default=str(X10 / "verdict"))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    if a.synthetic:
        return selftest()
    res = evaluate(pathlib.Path(a.steady), pathlib.Path(a.prewarm), pathlib.Path(a.equiv), pathlib.Path(a.profile_json), pathlib.Path(a.log_dir))
    res = write_outputs(res, pathlib.Path(a.out_dir), synthetic=False)
    print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
