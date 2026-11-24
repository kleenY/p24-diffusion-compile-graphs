"""Cost table access for the simulator / break-even model.

cost_table.csv (written by scripts/build_cost_table.py): one row per (arm, H, W, B, cfg).
  steady_ms, steady_ci_lo, steady_ci_hi        median over processes of the per-process steady median (ms), process bootstrap CI
  first_req_s__<tier>                          first request (prepare + first call), median over processes of that tier
  resid_s__<tier>                              one-time cost beyond the components: first_req - steady - sum(component first calls);
                                               for non-compiled arms (E0, G1) the whole one-time cost first_req - steady
  fc_te_s__<tier>, fc_unet_s__<tier>, fc_vae_s__<tier>
                                               first-call seconds of each compiled component, ONLY from observations where that
                                               component key was new (shared components are not averaged in as 0)
  capture_s, capture_pool_mb, static_io_mb, steady_peak_reserved_mb, weights_alloc_mb, equiv_pass, transient_excess_s
  block, source (measured / interpolated_B3 / extrapolated_ratio_C1 / dev_synthetic), fc_source
tiers: cold_empty, cold_diskcache (first shape of a fresh process), inproc_empty, inproc_diskcache (a later new shape in a
process whose compile cache is empty / pre-warmed). Cache-independent arms (E0, G1) fill both cold tiers from all processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..policy.base import Shape

TIERS = ["cold_empty", "cold_diskcache", "inproc_empty", "inproc_diskcache"]
COMPONENTS = ["te", "unet", "vae"]
COMPILE_ARMS = {"C1", "C1+G1", "C1+G2", "C2", "C3", "C5", "C6"}
CACHE_INDEPENDENT = {"E0", "E1", "G1", "G2"}  # no compile cache: their cold tiers use all non-inproc processes

# tier -> fallback chain (first non-NaN wins; anything but the first entry is reported as a fallback). Only the safe direction
# is allowed: a later shape of a process with the SAME cache state may use the first-shape cost of that cache state (an upper
# bound: library start-up is included there). A cold tier never falls back to another cache state (an empty cache costs
# ~10x a pre-warmed one), so a missing cold tier raises MissingCost instead of silently under- or over-predicting.
_CHAIN = {
    "cold_empty": ["cold_empty"],
    "cold_diskcache": ["cold_diskcache"],
    "inproc_empty": ["inproc_empty", "cold_empty"],
    "inproc_diskcache": ["inproc_diskcache", "cold_diskcache"],
}


class MissingCost(KeyError):
    pass


def tier_for(cache_mode: str, first_shape: bool) -> str:
    """cache_mode: 'warm' (disk cache pre-warmed) or 'empty'. First shape of the process -> cold tier, later -> inproc tier."""
    base = "empty" if cache_mode == "empty" else "diskcache"
    return f"cold_{base}" if first_shape else f"inproc_{base}"


def proc_tier_label(label: str) -> str:
    """cold_tier label of a process row -> tier name (TIERS), 'inproc_any' (in-process run whose cache mode is unknown, valid
    for cache-independent arms only) or 'other' (na / cold_partial / unknown)."""
    if label in TIERS:
        return label
    return "inproc_any" if str(label).startswith("inproc") else "other"


def proc_tier_mask(tier_col: pd.Series, arm: str, tier: str) -> pd.Series:
    """Rows of a per-process frame that belong to `tier` for `arm`: cache-independent arms use every non-inproc process
    for the cold tiers and every in-process run (any cache mode) for the inproc tiers."""
    if arm in CACHE_INDEPENDENT:
        if tier.startswith("cold"):
            return ~tier_col.str.startswith("inproc")
        return tier_col.isin([tier, "inproc_any"])
    return tier_col == tier


def _nz(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


@dataclass
class OneTime:
    resid_s: float
    fc: Dict[str, float]
    tier: str
    fallbacks: List[str] = field(default_factory=list)  # e.g. ["resid:inproc_diskcache->cold_diskcache"]

    @property
    def total_s(self) -> float:
        return self.resid_s + sum(self.fc.values())


class CostTable:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._rows = {}
        for r in self.df.to_dict("records"):
            self._rows[(r["arm"], int(r["H"]), int(r["W"]), int(r["B"]), int(r["cfg"]))] = r

    @classmethod
    def from_csv(cls, path) -> "CostTable":
        return cls(pd.read_csv(path))

    @staticmethod
    def _k(arm, shape: Shape):
        return (arm, int(shape.H), int(shape.W), int(shape.B), int(shape.cfg))

    def has(self, arm: str, shape: Shape) -> bool:
        return self._k(arm, shape) in self._rows

    def row(self, arm: str, shape: Shape) -> dict:
        try:
            return self._rows[self._k(arm, shape)]
        except KeyError:
            raise MissingCost(f"cost table has no row for arm={arm} shape={shape.tag()}") from None

    def shapes(self, arm: str) -> List[Shape]:
        return [Shape(h, w, b, c) for (a, h, w, b, c) in self._rows if a == arm]

    def steady_s(self, arm: str, shape: Shape) -> float:
        v = self.row(arm, shape)["steady_ms"]
        if not _nz(v):
            raise MissingCost(f"no steady_ms for arm={arm} shape={shape.tag()}")
        return float(v) / 1e3

    def capture_pool_mb(self, arm: str, shape: Shape) -> float:
        v = self.row(arm, shape).get("capture_pool_mb")
        return float(v) if _nz(v) else 0.0

    def transient_s(self, arm: str, shape: Shape) -> float:
        v = self.row(arm, shape).get("transient_excess_s")
        return float(v) if _nz(v) else 0.0

    def _pick(self, row: dict, prefix: str, tier: str):
        """First non-NaN `<prefix>__<t>` along the tier's fallback chain -> (value, tier_used) or (None, None)."""
        for t in _CHAIN[tier]:
            v = row.get(f"{prefix}__{t}")
            if _nz(v):
                return float(v), t
        return None, None

    def _shared_key_fc(self, arm: str, shape: Shape, comp: str, tier: str) -> Optional[float]:
        """Median `fc_<comp>_s__<tier>` over the other measured shapes of `arm` with the same component cache key."""
        from ..policy import component_keys
        k = component_keys(shape)[comp]
        vals = [float(r[f"fc_{comp}_s__{tier}"]) for (a, h, w, b, c), r in self._rows.items()
                if a == arm and (h, w, b, c) != self._k(arm, shape)[1:] and r.get("source") == "measured"
                and _nz(r.get(f"fc_{comp}_s__{tier}")) and component_keys(Shape(h, w, b, c))[comp] == k]
        return float(np.median(vals)) if vals else None

    def one_time(self, arm: str, shape: Shape, tier: str, need_components: Optional[bool] = None) -> OneTime:
        """One-time cost (seconds, excluding the steady request itself) for a NEW shape at `tier`."""
        row = self.row(arm, shape)
        comp = (arm in COMPILE_ARMS) if need_components is None else need_components
        fb: List[str] = []
        v, used = self._pick(row, "resid_s", tier)
        if v is None:
            raise MissingCost(f"no one-time cost (resid_s) for arm={arm} shape={shape.tag()} (any tier)")
        if used != tier:
            fb.append(f"resid:{tier}->{used}")
        fc = {}
        if comp and row.get("fc_source") == "missing":
            # no component breakdown was measured for this cell (e.g. S1 raw data): the whole cost sits in resid_s,
            # component sharing between shapes cannot be credited (recorded as a fallback)
            fc = {c: 0.0 for c in COMPONENTS}
            fb.append("fc:none(all in resid)")
        elif comp:
            for c in COMPONENTS:
                x, u = self._pick(row, f"fc_{c}_s", tier)
                if u != tier:
                    # the first-call cost belongs to the COMPONENT cache key, not to the request shape: in the measured
                    # sequence the key may have been introduced by another shape (then this shape credited 0 / NaN)
                    y = self._shared_key_fc(arm, shape, c, tier)
                    if y is not None:
                        fc[c] = y
                        fb.append(f"fc_{c}:shared_key")
                        continue
                if x is None:
                    raise MissingCost(f"no first-call cost fc_{c}_s for arm={arm} shape={shape.tag()}")
                fc[c] = x
                if u != tier:
                    fb.append(f"fc_{c}:{tier}->{u}")
        return OneTime(v, fc, used, fb)
