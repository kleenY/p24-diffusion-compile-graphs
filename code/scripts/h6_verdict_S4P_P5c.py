"""S4-P P5c step 4 (CPU, seconds): H6a-H6f verdict tables from a sim_h6_S4P_P5c raw csv.  Rules = goals/S4P_P5c_AOT_confirmatory_prereg_v1.md sec 3 (0444, SHA16 c14507708a968699), written BEFORE any evaluation number:
  per cell (workload W, N, cold level): per-seed paired ratio r_s = service_s(numerator arm) / service_s(denominator arm) on the SAME trace; reported: mean, median, min, max, seed-resampling percentile
  bootstrap 95 % CI of the mean (10 000 resamples, numpy default_rng(0), one fresh rng per cell), number of seeds at or below the line.  PASS of a cell = mean <= line AND >= 8/10 seeds <= line.
  H6a  W2, W5; N=1e3; disk;  B1-AOT / B1                       <= 0.80   (W2 and W5 must both pass)
  H6b  W2, W5; N=1e4; disk;  B1-AOT / B1                       <= 0.97
  H6c  W2, W5; N in {1e3, 1e4}; disk; B1-AOT / best old policy  <= 1.00   (all four cells; best old per seed = min over {B0, B1, B2, B3, B3p, B3pp, B4, B5, B6, B6p, Ours (v2), Ours-v3} on that trace)
  H6d  (secondary, simulation only) W2, W5; disk; v2-AOT / Ours (v2): N=1e3 <= 0.99, N=1e4 <= 0.95
  H6e  (no harm) disk; W1-W5 x N in {1e2,1e3,1e4,1e5}: max over the 20 cells of the cell mean of B1-AOT / B1 <= 1.005 (cell mean = mean of the per-seed ratios; the ratio of the cell means is reported next to it)
  H6f  (expected negative, no line) W2, W5; N in {1e3, 1e4}; empty-cache level; B1-AOT / B1
  Main simulation conclusion = H6a AND H6b AND H6c AND H6e;  H6d and H6f are reported alone.
Also reported (not judged): full table 3 cold levels x 5 W x 4 N of B1-AOT / B1, / B0, / best old; break-even K* = offline build seconds / per-process disk-level saving (B1 - B1-AOT) and the extra-offline-cost variant
  (build - C1 pre-warm, floored at 0) / saving (numbers from main_meta.json 'offline'; the C1 pre-warm definition is the executor's, see sim_h6_S4P_P5c.py); sensitivities nofb / excl / sens / H6c with nofb.
Confirmatory labelling: the verdict is confirmatory only when the raw frame is the eval split with 10 seeds in every judged cell and the full grid; otherwise `confirmatory=false` and the verdicts are labelled dev-style.
Usage: "$PROJ_PYTHON" -B -m scripts.h6_verdict_S4P_P5c --raw ../results/S4P/p5c/eval_sim/main_raw.csv --out-dir ../results/S4P/p5c/eval_sim/verdict [--synthetic]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts import aot_tables_S4P_P5c as T

OLD_SET = list(T.OLD_SET)
WL_ALL = ["W1", "W2", "W3", "W4", "W5"]
N_ALL = [100, 1000, 10000, 100000]
COLDS = ["empty", "disk", "hot"]
BOOT_N = 10000
HOLD = {"a": dict(N=1000, line=0.80), "b": dict(N=10000, line=0.97)}
H6D_LINES = {1000: 0.99, 10000: 0.95}
H6E_LINE = 1.005
NEED_FRAC_NUM, NEED_FRAC_DEN = 8, 10        # >= 8/10 of the seeds (integer ceil below)


def boot_ci(x: np.ndarray, n: int = BOOT_N, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stat(r: pd.Series, line: Optional[float] = None) -> dict:
    x = r.to_numpy(float)
    n = len(x)
    if n == 0:
        return dict(n_seeds=0, mean=np.nan, median=np.nan, min=np.nan, max=np.nan, ci_lo=np.nan, ci_hi=np.nan, line=line, k_le_line=0, need=0, pass_mean=False, pass_k=False, passed=False)
    lo, hi = boot_ci(x) if n > 1 else (float("nan"), float("nan"))
    d = dict(n_seeds=n, mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()), max=float(x.max()), ci_lo=lo, ci_hi=hi)
    if line is not None:
        k = int((x <= line).sum())
        need = (NEED_FRAC_NUM * n + NEED_FRAC_DEN - 1) // NEED_FRAC_DEN          # ceil(0.8 n); 8 for 10 seeds
        pm, pk = bool(x.mean() <= line), bool(k >= need)
        d.update(line=line, k_le_line=k, need=need, pass_mean=pm, pass_k=pk, passed=bool(pm and pk))
    return d


class Frame:
    """service_s pivot: index (workload, N, cold, seed) x columns variant."""

    def __init__(self, df: pd.DataFrame):
        ok = df[df.status == "ok"]
        self.piv = ok.pivot_table(index=["workload", "N", "cold", "seed"], columns="variant", values="service_s", aggfunc="mean")
        self.has = set(self.piv.columns)

    def cell(self, w, n, cold) -> pd.DataFrame:
        try:
            return self.piv.xs((w, n, cold), level=("workload", "N", "cold"))
        except KeyError:
            return pd.DataFrame(columns=self.piv.columns)

    def best_old(self, sub: pd.DataFrame):
        cols = [c for c in OLD_SET if c in sub.columns]
        if len(cols) < len(OLD_SET):
            return pd.Series(np.nan, index=sub.index), pd.Series("", index=sub.index)
        v = sub[cols]
        bo = v.min(axis=1, skipna=False)
        nm = v.idxmin(axis=1).where(bo.notna(), "")
        return bo, nm

    def ratio(self, w, n, cold, num: str, den: str):
        """per-seed ratio Series (index seed) and the name of the best old policy per seed when den == 'best_old'."""
        sub = self.cell(w, n, cold)
        if num not in sub.columns or (den != "best_old" and den not in sub.columns):
            return pd.Series(dtype=float), None
        names = None
        if den == "best_old":
            d, names = self.best_old(sub)
        else:
            d = sub[den]
        return (sub[num] / d).dropna().sort_index(), names


def cells(F: Frame, wls, Ns, cold, num, den, lines=None, tag="") -> List[dict]:
    out = []
    for w in wls:
        for n in Ns:
            r, names = F.ratio(w, n, cold, num, den)
            line = None if lines is None else (lines[n] if isinstance(lines, dict) else lines)
            d = dict(tag=tag, workload=w, N=n, cold=cold, numerator=num, denominator=den, **stat(r, line))
            if names is not None and len(r):
                d["best_old_counts"] = json.dumps(pd.Series(names.loc[r.index]).value_counts().to_dict(), sort_keys=True)
            d["seeds"] = [int(s) for s in r.index]
            out.append(d)
    return out


def verdict(df: pd.DataFrame, meta: Optional[dict] = None) -> dict:
    F = Frame(df)
    split = ",".join(sorted(df.split.unique())) if "split" in df else "?"
    need_vars = {"B0", "B1", "B1-AOT", "Ours", "v2-AOT"}
    missing = sorted((need_vars | set(OLD_SET)) - F.has)
    H = {}
    H["H6a"] = cells(F, ["W2", "W5"], [1000], "disk", "B1-AOT", "B1", HOLD["a"]["line"], "H6a")
    H["H6b"] = cells(F, ["W2", "W5"], [10000], "disk", "B1-AOT", "B1", HOLD["b"]["line"], "H6b")
    H["H6c"] = cells(F, ["W2", "W5"], [1000, 10000], "disk", "B1-AOT", "best_old", 1.00, "H6c")
    H["H6d"] = cells(F, ["W2", "W5"], [1000, 10000], "disk", "v2-AOT", "Ours", H6D_LINES, "H6d")
    H["H6f"] = cells(F, ["W2", "W5"], [1000, 10000], "empty", "B1-AOT", "B1", None, "H6f")
    # H6e: all disk cells of the full grid
    e_cells = cells(F, WL_ALL, N_ALL, "disk", "B1-AOT", "B1", None, "H6e")
    for c in e_cells:
        w, n = c["workload"], c["N"]
        sub = F.cell(w, n, "disk")
        c["ratio_of_means"] = float(sub["B1-AOT"].mean() / sub["B1"].mean()) if {"B1-AOT", "B1"} <= set(sub.columns) and len(sub) else float("nan")
    seeds_expected = 10
    judged = H["H6a"] + H["H6b"] + H["H6c"] + H["H6d"]
    complete_seeds = all(c["n_seeds"] == seeds_expected for c in judged + e_cells)
    full_grid = all(c["n_seeds"] > 0 for c in e_cells) and not missing
    confirmatory = bool(split == "eval" and complete_seeds and full_grid and (meta is None or not meta.get("synthetic", False)))
    e_max = max((c["mean"] for c in e_cells if c["n_seeds"]), default=float("nan"))
    e_max_cell = max((c for c in e_cells if c["n_seeds"]), key=lambda c: c["mean"], default=None)
    e_max_rom = max((c["ratio_of_means"] for c in e_cells if c["n_seeds"]), default=float("nan"))
    res = dict(split=split, confirmatory=confirmatory, missing_variants=missing, seeds_expected=seeds_expected, complete_seeds=bool(complete_seeds), full_grid=bool(full_grid),
               label="confirmatory (pre-registered, new seeds)" if confirmatory else "NOT confirmatory (dev-style / synthetic / incomplete): numbers only, no pass/fail claim")
    for k in ("H6a", "H6b", "H6c", "H6d"):
        res[k] = dict(cells=H[k], passed=bool(len(H[k]) > 0 and all(c["passed"] for c in H[k])))
    res["H6e"] = dict(cells=e_cells, n_cells=len(e_cells), max_cell_mean=e_max, max_cell=None if e_max_cell is None else [e_max_cell["workload"], e_max_cell["N"]], max_ratio_of_means=e_max_rom, line=H6E_LINE,
                      passed=bool(len(e_cells) == 20 and all(c["n_seeds"] > 0 for c in e_cells) and e_max <= H6E_LINE),
                      pass_if_ratio_of_means=bool(e_max_rom <= H6E_LINE), note="primary = mean of per-seed ratios; ratio of means shown because the preregistration wording 'cell mean' is not explicit")
    res["H6f"] = dict(cells=H["H6f"], note="expected > 1; reported, no pass line")
    res["main_conclusion"] = dict(H6a=res["H6a"]["passed"], H6b=res["H6b"]["passed"], H6c=res["H6c"]["passed"], H6e=res["H6e"]["passed"],
                                  all=bool(res["H6a"]["passed"] and res["H6b"]["passed"] and res["H6c"]["passed"] and res["H6e"]["passed"]),
                                  rule="H6a AND H6b AND H6c AND H6e (H6d alone, H6f as the cost)")
    # ---- sensitivities (not judged)
    sens = []
    for lab, num, den, ws, ns, cold, lines in (
            ("nofb_vs_B1", "B1-AOT-nofb", "B1", ["W2", "W5"], [1000, 10000], "disk", {1000: 0.80, 10000: 0.97}),
            ("nofb_vs_best_old", "B1-AOT-nofb", "best_old", ["W2", "W5"], [1000, 10000], "disk", 1.00),
            ("excl_B1AOT_over_B1", "B1-AOT-excl", "B1-excl", ["W2", "W5"], [1000, 10000], "disk", {1000: 0.80, 10000: 0.97}),
            ("empty_sens_inproc", "B1-AOT-sens", "B1", ["W2", "W5"], [1000, 10000], "empty", None)):
        sens += cells(F, ws, ns, cold, num, den, lines, lab)
    sens += cells(F, WL_ALL, N_ALL, "disk", "B1-AOT-excl", "B1-excl", None, "excl_H6e_cells")
    res["sensitivity"] = dict(cells=sens, note="reported, not judged; lines shown only as reference")
    # ---- full table + break-even
    off = (meta or {}).get("offline", {}) if meta else {}
    rows = []
    for cold in COLDS:
        for w in WL_ALL:
            for n in N_ALL:
                sub = F.cell(w, n, cold)
                if not len(sub) or "B1-AOT" not in sub:
                    continue
                r1, _ = F.ratio(w, n, cold, "B1-AOT", "B1")
                r0, _ = F.ratio(w, n, cold, "B1-AOT", "B0")
                rb, nm = F.ratio(w, n, cold, "B1-AOT", "best_old")
                row = dict(workload=w, N=n, cold=cold, n_seeds=len(sub), B1AOT_over_B1=float(r1.mean()) if len(r1) else np.nan, B1AOT_over_B0=float(r0.mean()) if len(r0) else np.nan,
                           B1AOT_over_best_old=float(rb.mean()) if len(rb) else np.nan, B1AOT_s=float(sub["B1-AOT"].mean()), B1_s=float(sub["B1"].mean()) if "B1" in sub else np.nan,
                           B0_s=float(sub["B0"].mean()) if "B0" in sub else np.nan)
                if cold == "disk" and "B1" in sub:
                    save = float(sub["B1"].mean() - sub["B1-AOT"].mean())
                    ab, cp = off.get("aot_build_total_s"), off.get("c1_prewarm_total_s_inproc_empty")
                    row.update(saving_per_process_s=save, K_star=(ab / save if (ab and save > 0) else np.nan),
                               K_star_extra_vs_C1_prewarm=(max(0.0, ab - cp) / save if (ab and cp is not None and save > 0) else np.nan))
                rows.append(row)
    res["full_table_rows"] = rows
    res["offline"] = off
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", help="main_meta.json of the sim run (default: next to --raw)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    raw = pathlib.Path(a.raw)
    out = pathlib.Path(a.out_dir)
    T.check_not_official_output(out, a.synthetic)
    metap = pathlib.Path(a.meta) if a.meta else raw.with_name("main_meta.json")
    meta = json.loads(metap.read_text()) if metap.exists() else None
    if not a.synthetic:
        T.check_prereg()
        assert meta is not None and not meta.get("synthetic"), "official verdict needs the real main_meta.json (synthetic=false)"
    elif meta is not None:
        meta = dict(meta, synthetic=True)
    df = pd.read_csv(raw)
    res = verdict(df, meta)
    res.update(raw=dict(path=str(raw), sha16=T.sha16(raw), n_rows=len(df)), script_sha16=T.sha16(pathlib.Path(__file__).resolve()), prereg_sha16=T.PREREG_SHA16,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"), bootstrap=dict(n=BOOT_N, rng="numpy default_rng(0) per cell"), synthetic=bool(a.synthetic))
    out.mkdir(parents=True, exist_ok=True)
    files = {"h6_verdict.json": None, "h6_cells_judged.csv": None, "h6_full_table.csv": None, "h6_sensitivity.csv": None}
    for f in files:
        if (out / f).exists():
            sys.exit(f"REFUSING to overwrite {out / f}")
    cl = _clean(res)
    (out / "h6_verdict.json").write_text(json.dumps(cl, indent=1, ensure_ascii=False))
    jr = [c for k in ("H6a", "H6b", "H6c", "H6d", "H6f") for c in res[k]["cells"]]
    pd.DataFrame([{k: v for k, v in c.items() if k != "seeds"} for c in jr]).to_csv(out / "h6_cells_judged.csv", index=False)
    pd.DataFrame(res["full_table_rows"]).to_csv(out / "h6_full_table.csv", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k != "seeds"} for c in res["sensitivity"]["cells"]]).to_csv(out / "h6_sensitivity.csv", index=False)
    if res["confirmatory"] and not a.synthetic:
        for f in files:
            os.chmod(out / f, 0o444)
    m = res["main_conclusion"]
    print(f"split={res['split']} confirmatory={res['confirmatory']} H6a={m['H6a']} H6b={m['H6b']} H6c={m['H6c']} H6e={m['H6e']} => main={m['all']}   H6d={res['H6d']['passed']}   "
          f"H6e max cell mean={res['H6e']['max_cell_mean']:.4f} at {res['H6e']['max_cell']}")
    for k in ("H6a", "H6b", "H6c", "H6d", "H6f"):
        for c in res[k]["cells"]:
            print(f"  {k} {c['workload']} N={c['N']:>6d} {c['cold']:5s} {c['numerator']}/{c['denominator']:9s} mean={c['mean']:.4f} [{c['ci_lo']:.4f},{c['ci_hi']:.4f}] k<=line {c.get('k_le_line')}/{c['n_seeds']} pass={c.get('passed')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
