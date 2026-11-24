"""CostView: read-only cost interface handed to S3 policies at construction time (interface v1 sec. 3).

Wraps `p24.sim.costs.CostTable` (duck-typed, no import of p24.sim here so policy/ stays import-light and cycle-free). A policy
only ever sees PREDICTED numbers from the (frozen) cost table: never a measured timing, never the true N.
`cache_mode` ('warm' = pre-warmed disk cache, 'empty') is a deployment fact the operator knows; it fixes the tier used for
the one-time cost of a NEW shape of an already running process ("inproc_<cache>").
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

from .base import Shape, component_keys

COMPILE_ARMS = {"C1", "C1+G1", "C1+G2", "C2", "C3", "C5", "C6"}


def _nz(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class CostView:
    def __init__(self, ct, cache_mode: str = "warm", te_by_resolution: bool = False,
                 rebuild_override: Optional[Dict[Tuple[str, Shape], float]] = None):
        assert cache_mode in ("warm", "empty")
        self.ct = ct
        self.cache_mode = cache_mode
        self.te_by_resolution = te_by_resolution
        self.rebuild_override = dict(rebuild_override or {})   # 3.1b addendum: measured re-capture seconds per (arm, shape)
        self._ot: Dict[tuple, object] = {}
        self._steady: Dict[tuple, float] = {}

    # ---- helpers -------------------------------------------------------------------------------------------------
    @property
    def inproc_tier(self) -> str:
        return "inproc_empty" if self.cache_mode == "empty" else "inproc_diskcache"

    def has(self, arm: str, shape: Shape) -> bool:
        return self.ct.has(arm, shape)

    # ---- interface -----------------------------------------------------------------------------------------------
    def steady_s(self, arm: str, shape: Shape) -> float:
        k = (arm, shape)
        v = self._steady.get(k)
        if v is None:
            v = self._steady[k] = self.ct.steady_s(arm, shape)
        return v

    def one_time_s(self, arm: str, shape: Shape, tier: Optional[str] = None,
                   compiled_keys: Optional[Iterable[tuple]] = None) -> float:
        """One-time seconds of building `arm` for a NEW shape (excluding the request's own steady time).
        For compile arms with `compiled_keys` given, component first-call costs of keys already compiled are not charged
        (component-key sharing; compiled_keys=None -> full cost). Missing cost -> +inf (policy treats as unaffordable)."""
        tier = tier or self.inproc_tier
        k = (arm, shape, tier)
        ot = self._ot.get(k)
        if ot is None:
            try:
                ot = self.ct.one_time(arm, shape, tier)
            except KeyError:        # MissingCost subclasses KeyError
                ot = False
            self._ot[k] = ot
        if ot is False:
            return math.inf
        if compiled_keys is None or arm not in COMPILE_ARMS or not ot.fc:
            return ot.total_s
        keys = component_keys(shape, self.te_by_resolution)
        return ot.resid_s + sum(v for c, v in ot.fc.items() if keys[c] not in compiled_keys)

    def capture_pool_mb(self, arm: str, shape: Shape) -> float:
        """Predicted graph pool (MiB). C2 (reduce-overhead) has no measured pool in the table: its C1+G1 pool is used as a proxy."""
        if arm == "C2":
            arm = "C1+G1"
        return self.ct.capture_pool_mb(arm, shape)

    def rebuild_s(self, arm: str, shape: Shape) -> float:
        """Seconds to re-capture a graph of an already built (evicted) shape: measured override, else cost table capture_s."""
        v = self.rebuild_override.get((arm, shape))
        if v is not None:
            return float(v)
        x = self.ct.row(arm, shape).get("capture_s")
        return float(x) if _nz(x) else 0.0

    def equiv_pass_padB4(self, shape: Shape, arm: Optional[str] = None) -> bool:
        """True iff serving `shape` (a B3 shape) padded into the B4 bucket passed the equivalence check (cost table column
        equiv_pass_padB4 of that shape's row for `arm`; arm=None -> every graph arm with a verdict must pass)."""
        arms = [arm] if arm else ["G1", "C1+G1"]
        vals = []
        for a in arms:
            if not self.ct.has(a, shape):
                continue
            v = self.ct.row(a, shape).get("equiv_pass_padB4")
            if _nz(v):
                vals.append(float(v) == 1.0)
        return bool(vals) and all(vals)
