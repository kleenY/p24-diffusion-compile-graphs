"""S4-P P5c step 6 (CPU): block-C verdict A1 / A2 / A3 + validity prerequisites V1-V4 + prediction errors (Pred-0 and Pred-1) for the B1-AOT arm.
Rules = goals/S4P_P5c_AOT_confirmatory_prereg_v1.md sec 4 (0444, SHA16 c14507708a968699), written BEFORE any block-C number:
  T = process total service time (meta `service_s`, includes all one-time costs and input preparation, as scripts/compare_S3_4.py); per stream the mean of the 3 repeats (processes p0-p2); combined = geometric mean of the 3 stream ratios.
  A1 (vs B1)  T(B1-AOT)/T(B1) <= 0.60 on every stream AND combined <= 0.55.
  A2 (first request)  per stream the MEDIAN over the pooled first requests of new shapes (event 'compile', all repeats, ~90 values) of `prepare_ms + run_ms` (the same quantity as the P5c smoke check; `dur_ms` incl. input prep reported alongside):
        B1-AOT <= 1.0 s AND <= 0.30 x the median of B1 of the same stream; all three streams.
  A3 (vs B0)  T(B1-AOT)/T(B0) <= 0.92 on every stream AND combined <= 0.90.
  Real machine holds = A1 AND A2 AND A3.  Only "simulation main conclusion AND real machine" allows the paper to claim an end-to-end speed-up (decided in the main session from h6_verdict + this file).
  Statistics: per-stream ratio + bootstrap 95 % CI (each arm's 3 repeat values resampled with replacement, 10 000 resamples, numpy default_rng(0) per ratio; the combined CI uses the same draws); 3 rounds paired by p listed.
  Reported, not judged: ratios of B1-AOT to v3, Ours, B4, B3p, B1-R when present.
  Validity (if any fails, A1-A3 are NOT judged and the failure is reported as such): V1 dry-run decision CSVs of B1-AOT and B1 byte-identical on the three streams (+ real-run decisions of every B1-AOT/B1 pair equal, informational);
        V2 every B0/B1/B1-AOT process: meta status ok, pinned_fallback_count 0, no NOT_QUIET printed by the gate before its STEP line (flagged processes: verdict computed with them AND in an `excluding_flagged` variant);
        V3 every B1-AOT process: meta aot.dynamo_compile_events == 0, rt_cache_files_after == 0, no package manifest mismatch; V4 no unexpected exception: all rows status ok, 3 processes per (stream, arm); fallback shapes/requests listed.
  Prediction errors (compare_S3_4 rules, <= 10 % per combination, reported as "simulator v2 on the new arm", NOT part of A1-A3): Pred-1 = frozen pred_S4P_P5c.csv restricted to B1-AOT/B1/B0;
        Pred-0 = the rough estimate written in the preregistration (B1-AOT: mid-point of the range, B1/B0 as printed there).
  Sensitivity (not judged): A2 without the fallback shapes' first requests.
Usage: "$PROJ_PYTHON" -B -m scripts.a123_verdict_S4P_P5c --raw ../results/S4P/p3/real_validation/raw --pred1 ../results/S4P/p3/real_validation/pred_S4P_P5c.csv --out-dir ../results/S4P/p5c/blockC_verdict [--synthetic ...]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import math
import os
import pathlib
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts import aot_tables_S4P_P5c as T
from scripts import compare_S3_4 as CMP

RV = T.ROOT / "results/S4P/p3/real_validation"
WLS = ["W2", "W3", "W5"]
STEM = {w: f"{w}_N300_seed101_B" for w in WLS}
ARM, B1, B0 = "B1-AOT", "B1", "B0"
A1_LINES, A2_ABS_S, A2_REL, A3_LINES = (0.60, 0.55), 1.0, 0.30, (0.92, 0.90)
BOOT_N = 10000
OTHERS = ["v3", "Ours", "B4", "B3p", "B1-R"]
# Pred-0 as printed in the preregistration sec 4 (B1-AOT range s; B1, B0 s)
PRED0 = {"W2_N300_seed101_B": dict(aot=(58.9, 61.3), B1=184.2, B0=72.8), "W3_N300_seed101_B": dict(aot=(88.9, 91.7), B1=225.0, B0=106.8),
         "W5_N300_seed101_B": dict(aot=(61.5, 64.0), B1=197.6, B0=75.7)}


def norm(p: str) -> str:
    return CMP.norm(p)


def _clean(o):
    """recursively: numpy scalars -> python, non-finite floats -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def load_procs(raw: pathlib.Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.meta.json")):
        m = json.loads(f.read_text())
        stem = pathlib.Path(m["trace"]).stem
        pol = norm(m["policy"])
        proc = int(f.name.rsplit("_p", 1)[1].split(".")[0])
        w = stem.split("_")[0]
        csvp = f.with_name(f.name.replace(".meta.json", ".csv"))
        rows.append(dict(stream=stem, workload=w, policy=pol, proc=proc, status=m.get("status"), service_s=m.get("service_s"), pinned=m.get("pinned_fallback_count"),
                         tag=f"{m['policy']}_{w}_warm_p{proc}", aot=m.get("aot"), aot_fb_requests=m.get("aot_fallback_requests"), aot_fb_shapes=m.get("aot_fallback_shapes"),
                         csv=str(csvp) if csvp.exists() else "", file=f.name))
    return pd.DataFrame(rows)


def not_quiet_tags(log_dir: pathlib.Path) -> Dict[str, List[str]]:
    """tag -> NOT_QUIET lines printed (by the gate, while waiting) before that process's STEP line; handles 'STEP [i/N] tag rc=' (window_S4P_C_v2.sh) and 'STEP tag rc='."""
    out: Dict[str, List[str]] = {}
    pend: List[str] = []
    for f in sorted(glob.glob(str(log_dir / "block_Cv2_*.log"))) + sorted(glob.glob(str(log_dir / "block.log"))):
        for l in pathlib.Path(f).read_text(errors="replace").splitlines():
            if "NOT_QUIET" in l:
                pend.append(l.strip())
            m = re.match(r"^STEP (?:\[\d+/\d+\] )?(\S+) rc=", l)
            if m:
                if pend:
                    out.setdefault(m.group(1), []).extend(pend)
                pend = []
    return out


def boot_ratios(a: List[np.ndarray], b: List[np.ndarray], n: int = BOOT_N, seed: int = 0):
    """per-stream ratio and geometric-mean ratio bootstrap (each arm's repeat values resampled with replacement; same draws for the combined ratio) -> (per-stream (lo,hi) list, combined (lo,hi))."""
    rng = np.random.default_rng(seed)
    rs = []
    for x, y in zip(a, b):
        ix = rng.integers(0, len(x), size=(n, len(x)))
        iy = rng.integers(0, len(y), size=(n, len(y)))
        rs.append(x[ix].mean(axis=1) / y[iy].mean(axis=1))
    rs = np.vstack(rs)
    g = np.exp(np.log(rs).mean(axis=0))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return [ci(r) for r in rs], ci(g)


def values(P: pd.DataFrame, stream: str, pol: str, drop: set) -> np.ndarray:
    g = P[(P.stream == stream) & (P.policy == pol) & (P.status == "ok") & (~P.tag.isin(drop))]
    return g.sort_values("proc").service_s.to_numpy(float)


def first_requests(P: pd.DataFrame, stream: str, pol: str, drop: set, exclude_shapes: set = frozenset()) -> pd.DataFrame:
    rows = []
    for r in P[(P.stream == stream) & (P.policy == pol) & (~P.tag.isin(drop))].itertuples():
        if not r.csv:
            continue
        for q in csv.DictReader(open(r.csv, newline="", encoding="utf-8")):
            if q.get("phase", "req") != "req" or q.get("event") != "compile" or q.get("status", "ok") != "ok":
                continue
            sk = f"{int(q['H'])}x{int(q['W'])}x{int(q['B'])}x{int(float(q['cfg']))}"
            if sk in exclude_shapes:
                continue
            pr, rn, dur = float(q["prepare_ms"]), float(q["run_ms"]), float(q.get("dur_ms") or "nan")
            rows.append(dict(stream=stream, policy=pol, proc=r.proc, idx=int(q["idx"]), shape=sk, first_s=(pr + rn) / 1e3, dur_s=dur / 1e3))
    return pd.DataFrame(rows, columns=["stream", "policy", "proc", "idx", "shape", "first_s", "dur_s"])


def judge(P: pd.DataFrame, drop: set, fb_shapes: set) -> dict:
    """A1/A2/A3 numbers and pass flags for the process set P minus `drop`."""
    res = dict(streams={}, other_ratios={})
    va = [values(P, STEM[w], ARM, drop) for w in WLS]
    v1 = [values(P, STEM[w], B1, drop) for w in WLS]
    v0 = [values(P, STEM[w], B0, drop) for w in WLS]
    ok = all(len(x) > 0 for x in va + v1 + v0)
    res["complete"] = bool(ok and all(len(x) == 3 for x in va + v1 + v0))
    if not ok:
        res["error"] = "an arm/stream has no ok process"
        return res
    r1 = [x.mean() / y.mean() for x, y in zip(va, v1)]
    r3 = [x.mean() / y.mean() for x, y in zip(va, v0)]
    g = lambda r: float(np.exp(np.mean(np.log(r))))
    ci1, cig1 = boot_ratios(va, v1)
    ci3, cig3 = boot_ratios(va, v0)
    for i, w in enumerate(WLS):
        a_by, b_by = (P[(P.stream == STEM[w]) & (P.policy == pol) & (P.status == "ok") & (~P.tag.isin(drop))].set_index("proc").service_s for pol in (ARM, B1))
        paired = {int(p): float(a_by[p] / b_by[p]) for p in sorted(set(a_by.index) & set(b_by.index))}
        res["streams"][w] = dict(T_aot=float(va[i].mean()), T_b1=float(v1[i].mean()), T_b0=float(v0[i].mean()), n_aot=len(va[i]), n_b1=len(v1[i]), n_b0=len(v0[i]),
                                 A1_ratio=float(r1[i]), A1_ci95=ci1[i], A3_ratio=float(r3[i]), A3_ci95=ci3[i], paired_A1_by_p=paired,
                                 values_aot=[float(x) for x in va[i]], values_b1=[float(x) for x in v1[i]], values_b0=[float(x) for x in v0[i]])
    A1 = dict(per_stream=[float(x) for x in r1], combined=g(r1), combined_ci95=cig1, lines=dict(stream=A1_LINES[0], combined=A1_LINES[1]))
    A1["passed"] = bool(all(x <= A1_LINES[0] for x in r1) and A1["combined"] <= A1_LINES[1])
    A3 = dict(per_stream=[float(x) for x in r3], combined=g(r3), combined_ci95=cig3, lines=dict(stream=A3_LINES[0], combined=A3_LINES[1]))
    A3["passed"] = bool(all(x <= A3_LINES[0] for x in r3) and A3["combined"] <= A3_LINES[1])
    a2, a2x = {}, {}
    for w in WLS:
        fa, fb = first_requests(P, STEM[w], ARM, drop), first_requests(P, STEM[w], B1, drop)
        ma, mb = (float(fa.first_s.median()) if len(fa) else float("nan")), (float(fb.first_s.median()) if len(fb) else float("nan"))
        a2[w] = dict(n_aot=len(fa), n_b1=len(fb), median_aot_s=ma, median_b1_s=mb, ratio=ma / mb if mb == mb and mb > 0 else float("nan"), dur_median_aot_s=float(fa.dur_s.median()) if len(fa) else float("nan"),
                     dur_median_b1_s=float(fb.dur_s.median()) if len(fb) else float("nan"), pass_abs=bool(ma <= A2_ABS_S), pass_rel=bool(ma <= A2_REL * mb))
        fa2, fb2 = first_requests(P, STEM[w], ARM, drop, fb_shapes), first_requests(P, STEM[w], B1, drop, fb_shapes)
        a2x[w] = dict(n_aot=len(fa2), median_aot_s=float(fa2.first_s.median()) if len(fa2) else float("nan"), median_b1_s=float(fb2.first_s.median()) if len(fb2) else float("nan"))
    A2 = dict(per_stream=a2, lines=dict(abs_s=A2_ABS_S, rel_of_b1=A2_REL), passed=bool(all(v["pass_abs"] and v["pass_rel"] for v in a2.values())), sensitivity_excluding_fallback_shapes=a2x)
    res.update(A1=A1, A2=A2, A3=A3, real_machine_holds=bool(A1["passed"] and A2["passed"] and A3["passed"]))
    for pol in OTHERS:
        vp = [values(P, STEM[w], pol, drop) for w in WLS]
        if all(len(x) for x in vp):
            r = [x.mean() / y.mean() for x, y in zip(va, vp)]
            res["other_ratios"][f"B1-AOT/{pol}"] = dict(per_stream=[float(x) for x in r], combined=g(r))
    return res


def run_cmp(pred_csv: pathlib.Path, raw: pathlib.Path, out: pathlib.Path, name: str, policies: Optional[List[str]] = None) -> dict:
    d = pd.read_csv(pred_csv)
    d["policy"] = d.policy.map(norm)
    if policies:
        d = d[d.policy.isin(policies)]
    if "event_counts" not in d:
        d["event_counts"] = ""
    sub = out / f"{name}_subset.csv"
    d.to_csv(sub, index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v = CMP.run(sub, raw, out / "__none__", out / name, "")
    (out / f"{name}_compare_S3_4_stdout.txt").write_text(buf.getvalue())
    et = pd.read_csv(out / name / "error_table.csv")
    rows = et[et.policy == ARM]
    return dict(all_within_10pct=bool(v["all_within_10pct"]), max_error=v["max_error"], worst_combo=v["worst_combo"], ranking_agrees=v["ranking_agrees"], n_judged=v["n_judged"],
                aot_rows=[dict(stream=r.stream, pred_s=float(r.sim_s), measured_median_s=float(r.measured_median_s), error=float(r.error), pass_10pct=int(r.pass_10pct)) for r in rows.itertuples()],
                simulator_v2_passes_on_new_arm=bool(len(rows) == 3 and (rows.pass_10pct == 1).all()))


def write_pred0(out: pathlib.Path) -> pathlib.Path:
    rows = []
    for stem, v in PRED0.items():
        for pol, val in ((ARM, sum(v["aot"]) / 2), (B1, v["B1"]), (B0, v["B0"])):
            rows.append(dict(stream=stem, policy=pol, cache_mode="warm", role="Pred-0", pred_total_s=val, pred_excl_input_s=val, event_counts=""))
    p = out / "pred0_prereg_values.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def validity(P: pd.DataFrame, dryrun_dir: pathlib.Path, nq: Dict[str, List[str]]) -> dict:
    v = {}
    # V1
    v1 = {}
    for w in WLS:
        fa, fb = dryrun_dir / f"B1-AOT_{STEM[w]}_p0.dryrun_decisions.csv", dryrun_dir / f"B1_{STEM[w]}_p0.dryrun_decisions.csv"
        if fa.exists() and fb.exists():
            v1[w] = bool(hashlib.sha256(fa.read_bytes()).digest() == hashlib.sha256(fb.read_bytes()).digest())
        else:
            v1[w] = None
    real_dec = {}
    for w in WLS:
        for p in sorted(P[P.stream == STEM[w]].proc.unique()):
            a, b = (P[(P.stream == STEM[w]) & (P.policy == pol) & (P.proc == p)] for pol in (ARM, B1))
            if len(a) and len(b) and a.csv.iloc[0] and b.csv.iloc[0]:
                da, db = ([(q["idx"], q["action"], q["event"], q.get("evicted", "")) for q in csv.DictReader(open(c, newline="", encoding="utf-8")) if q.get("phase", "req") == "req"] for c in (a.csv.iloc[0], b.csv.iloc[0]))
                real_dec[f"{w}_p{p}"] = bool(da == db)
    v["V1"] = dict(passed=bool(all(x is True for x in v1.values())), dryrun_byte_identical=v1, real_run_decisions_equal_info=real_dec)
    # V2
    arms = P[P.policy.isin([ARM, B1, B0])]
    bad_status = [r.tag for r in arms.itertuples() if r.status != "ok"]
    bad_pin = [r.tag for r in arms.itertuples() if r.pinned not in (0, "0")]
    flagged = sorted({r.tag for r in arms.itertuples() if r.tag in nq})
    v["V2"] = dict(passed=bool(not bad_status and not bad_pin and not flagged), bad_status=bad_status, pinned_nonzero=bad_pin, not_quiet_flagged=flagged, not_quiet_lines={t: nq[t] for t in flagged})
    # V3
    bad3 = []
    for r in P[P.policy == ARM].itertuples():
        a = r.aot if isinstance(r.aot, dict) else {}
        if not a or a.get("dynamo_compile_events") != 0 or a.get("rt_cache_files_after") != 0 or a.get("pkg_manifest_mismatch") not in ([], None):
            bad3.append(r.tag)
    v["V3"] = dict(passed=bool(not bad3 and (P.policy == ARM).any()), bad=bad3)
    # V4
    bad_rows = []
    for r in arms.itertuples():
        if r.csv:
            n_bad = sum(1 for q in csv.DictReader(open(r.csv, newline="", encoding="utf-8")) if q.get("status", "ok") != "ok")
            if n_bad:
                bad_rows.append(f"{r.tag}:{n_bad}")
    counts = {f"{w}/{pol}": int(((P.stream == STEM[w]) & (P.policy == pol) & (P.status == "ok")).sum()) for w in WLS for pol in (ARM, B1, B0)}
    fbs = sorted({s for r in P[P.policy == ARM].itertuples() for s in (r.aot_fb_shapes or [])})
    fbr = {f"{r.workload}_p{r.proc}": r.aot_fb_requests for r in P[P.policy == ARM].itertuples()}
    v["V4"] = dict(passed=bool(not bad_rows and all(c == 3 for c in counts.values())), rows_not_ok=bad_rows, ok_processes_per_stream_arm=counts, fallback_shapes=fbs, fallback_requests_by_process=fbr)
    v["all"] = bool(all(v[k]["passed"] for k in ("V1", "V2", "V3", "V4")))
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(RV / "raw"))
    ap.add_argument("--log-dir", default=str(RV / "logs"))
    ap.add_argument("--dryrun-dir", default=str(T.P5C / "dryrun_final"))
    ap.add_argument("--pred1", default=str(RV / "pred_S4P_P5c.csv"))
    ap.add_argument("--out-dir", default=str(T.P5C / "blockC_verdict"))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out_dir)
    if a.synthetic:
        assert not T.inside(out, T.ROOT / "results"), "SYNTHETIC outputs must not be written under results/"
    else:
        T.check_prereg()
        assert T.inside(out, T.P5C), f"official output must be under {T.P5C}"
        assert T.is_ro(a.pred1), f"{a.pred1} (Pred-1) must be frozen 0444 before block C"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSING to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    P = load_procs(raw)
    assert len(P), f"no *.meta.json in {raw}"
    nq = not_quiet_tags(pathlib.Path(a.log_dir))
    V = validity(P, pathlib.Path(a.dryrun_dir), nq)
    fb = set(V["V4"]["fallback_shapes"])
    flagged = set(V["V2"]["not_quiet_flagged"]) | set(V["V2"]["pinned_nonzero"]) | set(V["V2"]["bad_status"])
    full = judge(P, set(), fb)
    res = dict(validity=V, verdict_all_processes=full, time=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=bool(a.synthetic), prereg_sha16=T.PREREG_SHA16, script_sha16=T.sha16(pathlib.Path(__file__).resolve()),
               raw=str(raw), n_processes=int(len(P)))
    if flagged:
        res["verdict_excluding_flagged"] = judge(P, flagged, fb)
        res["excluded_processes"] = sorted(flagged)
    judged = bool(V["all"])
    res["A123_judged"] = judged
    res["conclusion"] = (dict(A1=full.get("A1", {}).get("passed"), A2=full.get("A2", {}).get("passed"), A3=full.get("A3", {}).get("passed"), real_machine_holds=full.get("real_machine_holds"))
                         if judged else "NOT JUDGED: a validity prerequisite failed (see validity); numbers are reported as a failure, not as A1-A3 verdicts")
    # ---- prediction errors
    p0 = write_pred0(out)
    res["pred0_errors"] = run_cmp(p0, raw, out, "pred0", [ARM, B1, B0])
    pr1 = pathlib.Path(a.pred1)
    res["pred1_errors"] = run_cmp(pr1, raw, out, "pred1", [ARM, B1, B0]) if pr1.exists() else "Pred-1 file missing"
    res["pred1_file"] = dict(path=str(pr1), sha16=T.sha16(pr1)) if pr1.exists() else None
    (out / "a123_verdict.json").write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False, default=str))
    P.drop(columns=["aot"]).to_csv(out / "a123_processes.csv", index=False)
    srows = [dict(workload=w, **{k: v for k, v in s.items() if not k.startswith("values_") and k != "paired_A1_by_p"}) for w, s in full.get("streams", {}).items()]
    pd.DataFrame(srows).to_csv(out / "a123_streams.csv", index=False)
    if not a.synthetic:
        for f in out.rglob("*"):
            if f.is_file():
                os.chmod(f, 0o444)
    A = full
    print(f"V1..V4 all={V['all']} ({', '.join(k + '=' + str(V[k]['passed']) for k in ('V1', 'V2', 'V3', 'V4'))}); A123 judged={judged}")
    if "A1" in A:
        print(f"A1 {['%.3f' % x for x in A['A1']['per_stream']]} comb {A['A1']['combined']:.3f} pass={A['A1']['passed']} | A3 {['%.3f' % x for x in A['A3']['per_stream']]} comb {A['A3']['combined']:.3f} pass={A['A3']['passed']} | "
              f"A2 pass={A['A2']['passed']} medians(aot) {[round(v['median_aot_s'], 3) for v in A['A2']['per_stream'].values()]}  real machine holds={A['real_machine_holds']}")
    print("Pred-1:", res["pred1_errors"] if not isinstance(res["pred1_errors"], dict) else {k: res["pred1_errors"][k] for k in ("all_within_10pct", "max_error", "simulator_v2_passes_on_new_arm")},
          "| Pred-0:", {k: res["pred0_errors"][k] for k in ("all_within_10pct", "max_error", "simulator_v2_passes_on_new_arm")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
