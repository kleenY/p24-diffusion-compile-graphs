"""S3 3.1b cost addendum: overlay of the S3-A measurements on the frozen S2 cost table (CPU only).

`results/S3/cost_addendum/cost_addendum.csv` (written by scripts/build_cost_addendum_S3.py) holds two kinds of rows:
  kind=inproc_cost   one row per (C2|C5, shape): median over the standard processes of the FIRST-call cost of a shape served by an already
                     running process with a pre-warmed disk cache -> replaces the tier `inproc_diskcache` cells
                     (first_req_s__, resid_s__, fc_{te,unet,vae}_s__, n__) of that (arm, shape) row of the table; source `inproc_measured_S3`
  kind=rebuild       one row per (G1|C1+G1, shape): measured eviction -> re-capture seconds; `rebuild_override_s` is set only where
                     rebuild wall / capture_s > 1.5 (NaN otherwise) -> CostView.rebuild_override
Only cells covered by the addendum are replaced; every other cell of the table is untouched (the S2 table file is never written).
`load_with_addendum` returns an `AddendumCostTable` (a CostTable with `.rebuild_override` and `.addendum_report`).
`install_simulator_hook()` makes the simulator's internal CostView (p24.sim.simulator builds its own `CostView(ct, ...)`, which cannot
receive the override) use `ct.rebuild_override`, so policy decisions and simulator charges see the SAME rebuild seconds; no frozen file changes.
"""
from __future__ import annotations

import math
import pathlib
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..policy.base import Shape
from ..policy.costview import CostView
from .costs import COMPONENTS, CostTable

INPROC_TIER = "inproc_diskcache"
INPROC_COLS = [f"first_req_s__{INPROC_TIER}", f"resid_s__{INPROC_TIER}", *[f"fc_{c}_s__{INPROC_TIER}" for c in COMPONENTS],
               f"n__{INPROC_TIER}"]
ADDENDUM_COLS = ["kind", "arm", "H", "W", "B", "cfg", "source", "flag", "n_proc",
                 "first_req_s", "resid_s", "fc_te_s", "fc_unet_s", "fc_vae_s",
                 "n_rounds", "rebuild_wall_med_s", "capture_s_med", "rebuild_over_capture", "rebuild_override_s",
                 "reserved_residual_med_mb", "recovered_frac", "n_fail_rounds"]
KEY = ["arm", "H", "W", "B", "cfg"]


def _nz(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class AddendumCostTable(CostTable):
    rebuild_override: Dict[Tuple[str, Shape], float]
    addendum_report: dict


def read_addendum(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cost addendum not found: {p} (run scripts/build_cost_addendum_S3.py, or pass --addendum none)")
    d = pd.read_csv(p)
    miss = {"kind", *KEY} - set(d.columns)
    if miss:
        raise ValueError(f"{p}: not a cost addendum csv, missing columns {sorted(miss)}")
    return d


def apply_addendum(table: pd.DataFrame, add: pd.DataFrame):
    """-> (new table DataFrame, rebuild_override {(arm, Shape): s}, report). The input frames are not modified."""
    df = table.copy().reset_index(drop=True)
    if "addendum_source" not in df.columns:
        df["addendum_source"] = ""
    df["addendum_source"] = df["addendum_source"].fillna("").astype(str)
    idx = {(r.arm, int(r.H), int(r.W), int(r.B), int(r.cfg)): i for i, r in enumerate(df[KEY].itertuples(index=False))}
    rep = dict(n_inproc_rows=0, n_inproc_replaced=0, inproc_skipped_no_table_row=[], inproc_replaced=[],
               n_rebuild_rows=0, n_override=0, overrides=[])
    for r in add[add.kind == "inproc_cost"].to_dict("records"):
        rep["n_inproc_rows"] += 1
        k = (r["arm"], int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"]))
        i = idx.get(k)
        if i is None:
            rep["inproc_skipped_no_table_row"].append(list(k))
            continue
        df.loc[i, f"first_req_s__{INPROC_TIER}"] = r["first_req_s"]
        df.loc[i, f"resid_s__{INPROC_TIER}"] = r["resid_s"]
        for c in COMPONENTS:      # NaN stays NaN: key shared with an earlier shape of the sequence (same convention as build_cost_table)
            df.loc[i, f"fc_{c}_s__{INPROC_TIER}"] = r.get(f"fc_{c}_s", np.nan)
        df.loc[i, f"n__{INPROC_TIER}"] = r["n_proc"]
        df.loc[i, "addendum_source"] = str(r.get("source", "inproc_measured_S3"))
        rep["n_inproc_replaced"] += 1
        rep["inproc_replaced"].append(list(k))
    ov: Dict[Tuple[str, Shape], float] = {}
    for r in add[add.kind == "rebuild"].to_dict("records"):
        rep["n_rebuild_rows"] += 1
        v = r.get("rebuild_override_s", np.nan)
        if _nz(v):
            ov[(r["arm"], Shape(int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"])))] = float(v)
            rep["overrides"].append([r["arm"], int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"]), float(v)])
    rep["n_override"] = len(ov)
    return df, ov, rep


def load_with_addendum(cost_table: Union[str, pathlib.Path, pd.DataFrame, CostTable],
                       addendum_csv: Union[str, pathlib.Path, pd.DataFrame]) -> AddendumCostTable:
    """cost_table: csv path / DataFrame / CostTable; addendum_csv: csv path or DataFrame. Missing addendum file -> FileNotFoundError."""
    if isinstance(cost_table, CostTable):
        base = cost_table.df
    elif isinstance(cost_table, pd.DataFrame):
        base = cost_table
    else:
        base = pd.read_csv(cost_table)
    add = addendum_csv if isinstance(addendum_csv, pd.DataFrame) else read_addendum(addendum_csv)
    df, ov, rep = apply_addendum(base, add)
    ct = AddendumCostTable(df)
    ct.rebuild_override = ov
    ct.addendum_report = rep
    return ct


class AddendumCostView(CostView):
    """CostView that takes the rebuild override from the cost table when none is passed (used inside the simulator)."""

    def __init__(self, ct, cache_mode: str = "warm", te_by_resolution: bool = False, rebuild_override=None):
        if rebuild_override is None:
            rebuild_override = getattr(ct, "rebuild_override", None)
        super().__init__(ct, cache_mode, te_by_resolution, rebuild_override)


def install_simulator_hook() -> None:
    """Swap the CostView name used by p24.sim.simulator.simulate for AddendumCostView (idempotent; plain tables are unaffected)."""
    from . import simulator as SIM
    SIM.CostView = AddendumCostView
