"""make_policy for S3 (interface v1 sec. 3): `make_policy(name, *, cost=None, params=None, prior=None, oracle_counts=None)`.
Old calls `make_policy("B0"|"B1"|"B4")` keep working (fixed.make_policy)."""
from __future__ import annotations

from typing import Dict, Optional

from .base import Shape
from .baselines import (B2Dynamic, B3PostWarm, B3PrimeLimit8, B3ReduceOverhead, B6BatchBuckets, B6PrimeTopK, B7Knapsack,
                        make_B5, make_Ours_NE)
from .fixed import POLICIES as FIXED
from .ours import Ours

ALIASES = {"B3'": "B3p", "B3′": "B3p", "B3''": "B3pp", "B3″": "B3pp", "B6'": "B6p", "B6′": "B6p"}
NAMES = ["B0", "B1", "B2", "B3", "B3p", "B3pp", "B4", "B5", "B6", "B6p", "B7", "Ours", "Ours-NE"]


def make_policy(name: str, *, cost=None, params: Optional[dict] = None, prior: Optional[Dict[Shape, float]] = None,
                oracle_counts: Optional[Dict[Shape, int]] = None):
    name = ALIASES.get(name, name)
    if name in FIXED:
        return FIXED[name]()
    if name == "B2":
        return B2Dynamic()
    need = {"B3": ("cost",), "B3pp": ("cost", "prior"), "B5": ("cost",), "B6": ("cost", "prior"), "B6p": ("cost", "prior"),
            "B7": ("cost", "oracle_counts"), "Ours": ("cost",), "Ours-NE": ("cost",)}
    got = dict(cost=cost, prior=prior, oracle_counts=oracle_counts)
    for k in need.get(name, ()):
        if got[k] is None:
            raise ValueError(f"policy {name} needs `{k}`")
    if name == "B3":
        return B3ReduceOverhead(cost)
    if name == "B3p":
        return B3PrimeLimit8()
    if name == "B3pp":
        return B3PostWarm(cost, prior, params)
    if name == "B5":
        return make_B5(cost, params, prior)
    if name == "B6":
        return B6BatchBuckets(cost, prior, params)
    if name == "B6p":
        return B6PrimeTopK(cost, prior, params)
    if name == "B7":
        return B7Knapsack(cost, oracle_counts, params)
    if name == "Ours":
        return Ours(cost, params, prior)
    if name == "Ours-NE":
        return make_Ours_NE(cost, params, prior)
    raise KeyError(name)
