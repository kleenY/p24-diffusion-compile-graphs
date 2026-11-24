"""S4-P P5c shared helpers (CPU only, no torch): preregistration SHA guard, fallback list reader, shape-key parsing, and the ONE function that turns a cost table into a "B1-AOT / v2-AOT" table.

Preregistration goals/S4P_P5c_AOT_confirmatory_prereg_v1.md (0444, SHA16 c14507708a968699): the AOT arm is simulated by swapping the one-time and steady cells of the C1 rows of the table
(the simulator maps the `compile` action to arm "C1"), so B1-AOT = policy B1 on the swapped table and v2-AOT = policy Ours on the swapped table (C1+G1 rows untouched).  Rows of shapes in the fallback
list (aot_fallback_shapes.csv) get the E0 steady and E0 one-time cells of the SAME table (sec 1: "fallback ... served fully in eager (E0); in the simulation the shapes of the same list use E0 costs").
`swap_c1` is also used with the P5 step-2 mapping providers (probe_aot_sim_S4P_P5.py) to reproduce step2_raw.csv bit for bit (sim_h6_S4P_P5c.py --repro-p5-step2).
Nothing here modifies a file.
"""
from __future__ import annotations

import csv
import hashlib
import os
import pathlib
import re
import stat
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from p24.policy.base import Shape
from p24.sim import addendum as ADD
from p24.sim.costs import COMPONENTS, TIERS, MissingCost

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
PREREG = ROOT / "goals/S4P_P5c_AOT_confirmatory_prereg_v1.md"
PREREG_SHA16 = "c14507708a968699"
P5C = ROOT / "results/S4P/p5c"
FALLBACK_CSV = P5C / "aot_fallback_shapes.csv"
COST_TABLE_AOT = P5C / "cost_table_v2_aot.csv"
EVAL_SEEDS = list(range(120, 130))
FORBIDDEN_SEEDS = set(range(0, 5)) | set(range(100, 120)) | {101}     # dev 0-4, S3 eval 100-109, S4-P eval 110-119, real-machine stream 101 (prereg sec 3)
AOT_ARM = "AOT"
OLD_ONLINE = ["B1", "B2", "B3", "B3p", "B3pp", "B4", "B5", "B6", "B6p"]            # H5b online set (scripts/h5_verdict_S4P.py)
OLD_SET = ["B0"] + OLD_ONLINE + ["Ours", "Ours-v3"]                                  # H6c "best old policy" set (prereg sec 3)
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def mode_of(p) -> int:
    return stat.S_IMODE(pathlib.Path(p).stat().st_mode)


def is_ro(p) -> bool:
    """0444 exactly (the 'frozen' convention of this project)."""
    return mode_of(p) == 0o444


def check_prereg() -> str:
    assert PREREG.is_file(), f"preregistration missing: {PREREG}"
    assert is_ro(PREREG), f"preregistration must be 0444: {PREREG}"
    got = sha16(PREREG)
    assert got == PREREG_SHA16, f"preregistration SHA16 {got} != frozen {PREREG_SHA16}"
    return got


def inside(path, root) -> bool:
    p, r = pathlib.Path(path).resolve(), pathlib.Path(root).resolve()
    return p == r or r in p.parents


def check_not_official_output(out, synthetic: bool) -> None:
    """Synthetic outputs must never land under results/ ; real outputs must land under results/S4P/p5c (or a sub-folder)."""
    if synthetic:
        assert not inside(out, ROOT / "results"), f"SYNTHETIC outputs must not be written under results/: {out}"
    else:
        assert inside(out, P5C), f"official outputs must be under {P5C}: {out}"


# ------------------------------------------------------------------------------------------------ keys
_TAG = re.compile(r"^(\d+)x(\d+)_b(\d+)_(cfg|nocfg)$")


def shape_from_tag(tag: str) -> Shape:
    """'512x512_b1_nocfg' (ShapeKey.tag, as in the AOTW / build JSON files) -> Shape."""
    m = _TAG.match(tag)
    if not m:
        raise ValueError(f"not a shape tag: {tag!r}")
    return Shape(int(m.group(1)), int(m.group(2)), int(m.group(3)), 1 if m.group(4) == "cfg" else 0)


def shape_from_key(key: str) -> Shape:
    """'512x768x3x1' (HxWxBxcfg, as in aot_fallback_shapes.csv) -> Shape."""
    h, w, b, c = (int(x) for x in key.split("x"))
    return Shape(h, w, b, c)


def read_fallback(path) -> Set[Shape]:
    """aot_fallback_shapes.csv (columns key,reason,detail; header-only = empty list) -> set of Shape."""
    out: Set[Shape] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("key") or "").strip():
                out.add(shape_from_key(r["key"].strip()))
    return out


# ------------------------------------------------------------------------------------------------ table swap
def _nz(x) -> bool:
    return x is not None and not (isinstance(x, float) and x != x)


Provider = Callable[[int, pd.Series, Shape], Optional[Tuple[Optional[float], Dict[str, Tuple[float, Optional[Dict[str, float]]]]]]]


def swap_c1(ct0: ADD.AddendumCostTable, provider: Provider, fallback: Iterable[Shape] = (), *, nan_if_none: bool = True) -> ADD.AddendumCostTable:
    """New table in which every row of arm C1 carries the AOT (or fallback-E0) prices.  `ct0` is not modified.
    provider(i, row, shape) -> (steady_ms or None, {tier: (resid_s, fc dict or None)}) or None (no AOT price for this shape).
      * steady_ms None  : keep the C1 steady cell (P5 convention);
      * fc dict None    : fc_{te,unet,vae} = 0.0 for that tier (everything is in resid_s; first_req_s__<tier> := NaN, not read by the simulator);
      * provider -> None: nan_if_none=True blanks steady and all four tiers of the row -> MissingCost if a trace asks for it (never a silent C1 price); False leaves the row unchanged.
    Fallback shapes (E0 service): steady = E0 steady of the same table, one-time at each tier = ct0.one_time('E0', shape, tier).total_s (same tier chain as the simulator's eager_first).
    The rebuild overrides of ct0 are carried over (the addendum is NOT applied a second time)."""
    d = ct0.df.copy()
    fb = set(fallback)
    for i in d.index[d.arm == "C1"]:
        row = d.loc[i]
        sh = Shape(int(row["H"]), int(row["W"]), int(row["B"]), int(row["cfg"]))
        if sh in fb:
            steady = float(ct0.row("E0", sh)["steady_ms"])
            vals = {t: (ct0.one_time("E0", sh, t).total_s, None) for t in TIERS}
        else:
            got = provider(i, row, sh)
            if got is None:
                if nan_if_none:
                    d.loc[i, "steady_ms"] = np.nan
                    for t in TIERS:
                        d.loc[i, f"resid_s__{t}"] = np.nan
                        d.loc[i, f"first_req_s__{t}"] = np.nan
                        for c in COMPONENTS:
                            d.loc[i, f"fc_{c}_s__{t}"] = np.nan
                continue
            steady, vals = got
        if steady is not None:
            d.loc[i, "steady_ms"] = steady
            for c in ("steady_ci_lo", "steady_ci_hi"):
                if c in d.columns:
                    d.loc[i, c] = np.nan
        for t, (resid, fc) in vals.items():
            d.loc[i, f"resid_s__{t}"] = resid
            for c in COMPONENTS:
                d.loc[i, f"fc_{c}_s__{t}"] = 0.0 if fc is None else fc[c]
            d.loc[i, f"first_req_s__{t}"] = np.nan
    ct = ADD.AddendumCostTable(d)
    ct.rebuild_override = dict(getattr(ct0, "rebuild_override", {}) or {})
    ct.addendum_report = getattr(ct0, "addendum_report", {})
    return ct


def aot_provider(ct0, arm: str = AOT_ARM, resid_col_override: Optional[Dict[str, str]] = None) -> Provider:
    """Prices from the rows of arm `arm` of the same table (cost_table_v2_aot.csv).  resid_col_override {tier: column} reads that tier from another column
    (e.g. {'inproc_empty': 'aot_resid_s__inproc_empty_sens'} for the build-wall sensitivity)."""
    rows = {(int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"])): r for r in ct0.df[ct0.df.arm == arm].to_dict("records")}
    ov = resid_col_override or {}

    def prov(i, row, sh):
        r = rows.get((sh.H, sh.W, sh.B, sh.cfg))
        if r is None:
            return None
        vals = {}
        for t in TIERS:
            v = r.get(ov.get(t, f"resid_s__{t}"))
            if not _nz(v):
                return None
            vals[t] = (float(v), None)
        if not _nz(r.get("steady_ms")):
            return None
        return float(r["steady_ms"]), vals

    return prov


def aot_shapes(df: pd.DataFrame, arm: str = AOT_ARM) -> Set[Shape]:
    return {Shape(int(r.H), int(r.W), int(r.B), int(r.cfg)) for r in df[df.arm == arm].itertuples()}
