"""Tiered break-even numbers (plan 4.5) with a process bootstrap.

For a shape s (all times in seconds, steady latencies converted from ms):
    e = eager (E0) steady, g = capture-only (G1) steady, o = compile+capture (C1+G1) steady
    Cg = one-time cost of capture beyond eager   = x_G1(tier) - x_E0(tier)
    Cc = one-time cost of compile beyond eager   = x_C1(tier) - x_E0(tier)       (compile-only arm, no graph)
    n_capture = Cg / (e - g)            n_compile = (Cc + Cg) / (e - o)            n_upgrade = Cc / (g - o)
with x_arm(tier) = first_req_s - steady (the intrinsic one-time cost of a new shape), numerators clipped at 0, and the value
inf when the denominator is <= 0 (the arm is not faster than the reference). `n_compile_direct` uses the measured one-time cost
of the C1+G1 arm, x_C1+G1(tier) - x_E0(tier), instead of Cc + Cg (cross-check of the additive model).
Tiers: cold_empty, cold_diskcache (first shape of a fresh process), inproc_empty, inproc_diskcache (later new shape; E0/G1
fall back to their cold values when no in-process runs exist, flag `eg_fallback`).

Process bootstrap (`n_boot`=10 000, seed 0): the processes of each (arm, role) group are resampled with replacement
independently (steady medians from all non-inproc processes of the arm; one-time costs from the tier's processes), the statistic
is recomputed from the medians of the resampled processes, CI = [2.5%, 97.5%] order statistics (no interpolation, so inf values are
kept as inf). Rows without process data (interpolated B3, summary-only cells) get the point value from the cost table and NaN CI.

CLI: python -m p24.sim.breakeven --cost-table T.csv [--procs P.csv] --out breakeven.csv
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..policy.base import Shape
from .costs import COMPILE_ARMS, TIERS, CostTable, proc_tier_mask

INF = float("inf")


def _div(num, den):
    """num/den elementwise; inf where den <= 0; numerator clipped at 0."""
    num = np.maximum(np.asarray(num, float), 0.0)
    den = np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), INF)
    return np.where(np.isnan(num) | np.isnan(den), np.nan, out)


def n_capture(e, g, cg):
    return _div(cg, np.asarray(e, float) - np.asarray(g, float))


def n_compile(e, o, cc, cg):
    return _div(np.asarray(cc, float) + np.asarray(cg, float), np.asarray(e, float) - np.asarray(o, float))


def n_upgrade(g, o, cc):
    return _div(cc, np.asarray(g, float) - np.asarray(o, float))


def _scalar(x):
    x = float(np.asarray(x))
    return x


def point(e, g, o, cg, cc, c_direct=None) -> Dict[str, float]:
    d = dict(n_capture=_scalar(n_capture(e, g, cg)), n_compile=_scalar(n_compile(e, o, cc, cg)),
             n_upgrade=_scalar(n_upgrade(g, o, cc)))
    d["n_compile_direct"] = _scalar(_div(c_direct, np.asarray(e, float) - np.asarray(o, float))) if c_direct is not None else np.nan
    return d


def _ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(v, 2.5, method="lower")), float(np.percentile(v, 97.5, method="higher")))


def _boot_median(arr: np.ndarray, rng, n_boot: int) -> np.ndarray:
    """arr (n,) or (n, k) -> (n_boot,) or (n_boot, k) medians of process resamples."""
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.median(arr[idx], axis=1)


def bootstrap(groups: Dict[str, np.ndarray], n_boot: int = 10000, seed: int = 0) -> Optional[Dict[str, tuple]]:
    """groups: e_s, x_e, g_s, x_g, o_s, x_o, x_c (1-D arrays of per-process values, seconds)."""
    if any(len(v) < 2 for v in groups.values()):
        return None
    rng = np.random.default_rng(seed)
    b = {k: _boot_median(groups[k], rng, n_boot) for k in ("e_s", "x_e", "g_s", "x_g", "o_s", "x_o", "x_c")}
    cg, cc = b["x_g"] - b["x_e"], b["x_c"] - b["x_e"]
    return dict(n_capture=_ci(n_capture(b["e_s"], b["g_s"], cg)),
                n_compile=_ci(n_compile(b["e_s"], b["o_s"], cc, cg)),
                n_upgrade=_ci(n_upgrade(b["g_s"], b["o_s"], cc)),
                n_compile_direct=_ci(_div(b["x_o"] - b["x_e"], b["e_s"] - b["o_s"])))


def _tier_x(g: pd.DataFrame, arm: str, tier: str):
    """per-process intrinsic one-time cost x_s of an arm at a tier; (array, tier_used). E0 / G1 fall back to their cold
    values when no in-process run exists."""
    def sub(t):
        s = g[proc_tier_mask(g.tier, arm, t)]
        return s[(s.tier_violation == 0) & s.x_s.notna()]
    s = sub(tier)
    if len(s) or not tier.startswith("inproc") or arm in COMPILE_ARMS:
        return s.x_s.to_numpy(float), tier
    cold = "cold_empty" if tier.endswith("empty") else "cold_diskcache"
    return sub(cold).x_s.to_numpy(float), cold


def _steady(g: pd.DataFrame) -> np.ndarray:
    s = g[~g.tier.str.startswith("inproc")]
    s = s if len(s) else g
    return s.steady_ms.to_numpy(float) / 1e3


def breakeven_table(ct: CostTable, procs: Optional[pd.DataFrame] = None, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """One row per (shape, tier) for every shape that has E0, G1, C1 and C1+G1 rows."""
    rows = []
    shapes = [s for s in ct.shapes("E0") if all(ct.has(a, s) for a in ("G1", "C1", "C1+G1"))]
    pindex = {}
    if procs is not None and len(procs):
        for k, g in procs.groupby(["arm", "H", "W", "B", "cfg"]):
            pindex[k] = g
    for s in sorted(shapes):
        for tier in TIERS:
            try:
                rc = ct.row("C1", s)
                if not np.isfinite(rc.get(f"resid_s__{tier}", np.nan)):
                    continue
                x = {}
                eg_fb = 0
                for arm in ("E0", "G1", "C1", "C1+G1"):
                    try:
                        ot = ct.one_time(arm, s, tier)
                    except KeyError:
                        ot = None
                    if ot is None:
                        break
                    x[arm] = ot.total_s
                    if arm in ("E0", "G1") and ot.tier != tier:
                        eg_fb = 1
                if len(x) < 4:
                    continue
            except KeyError:
                continue
            e, g, o = (ct.steady_s(a, s) for a in ("E0", "G1", "C1+G1"))
            cg, cc = x["G1"] - x["E0"], x["C1"] - x["E0"]
            row = dict(H=s.H, W=s.W, B=s.B, cfg=s.cfg, tier=tier, e_ms=e * 1e3, g_ms=g * 1e3, o_ms=o * 1e3,
                       cg_s=cg, cc_s=cc, eg_fallback=eg_fb, source=ct.row("C1+G1", s)["source"])
            row.update(point(e, g, o, cg, cc, c_direct=x["C1+G1"] - x["E0"]))
            groups = None
            keys = {a: (a, s.H, s.W, s.B, s.cfg) for a in ("E0", "G1", "C1", "C1+G1")}
            if pindex and all(k in pindex for k in keys.values()):
                pe, pg, po, pc = (pindex[keys[a]] for a in ("E0", "G1", "C1+G1", "C1"))
                xe, _ = _tier_x(pe, "E0", tier)
                xg, _ = _tier_x(pg, "G1", tier)
                xo, _ = _tier_x(po, "C1+G1", tier)
                xc, _ = _tier_x(pc, "C1", tier)
                groups = dict(e_s=_steady(pe), x_e=xe, g_s=_steady(pg), x_g=xg, o_s=_steady(po), x_o=xo, x_c=xc)
                row.update(n_proc_e=len(groups["e_s"]), n_proc_g=len(groups["g_s"]), n_proc_o=len(groups["o_s"]),
                           n_proc_xc=len(xc))
            ci = bootstrap(groups, n_boot, seed) if groups else None
            for k in ("n_capture", "n_compile", "n_upgrade", "n_compile_direct"):
                row[f"{k}_lo"], row[f"{k}_hi"] = ci[k] if ci else (np.nan, np.nan)
            row["ci_note"] = "process bootstrap" if ci else "no process-level data (point value from cost table)"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost-table", required=True)
    ap.add_argument("--procs", default=None, help="cost_table_procs.csv from build_cost_table (default: next to the cost table)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"REFUSING to overwrite {out}")
    ctp = pathlib.Path(a.cost_table)
    pp = pathlib.Path(a.procs) if a.procs else ctp.with_name("cost_table_procs.csv")
    procs = pd.read_csv(pp) if pp.exists() else None
    t = breakeven_table(CostTable.from_csv(ctp), procs, a.n_boot, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print(f"{out}: {len(t)} rows")


if __name__ == "__main__":
    main()
