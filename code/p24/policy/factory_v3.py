"""make_policy for S4-P: the S3 factory (p24/policy/factory.py, unchanged, frozen SHA) plus "Ours-v3" (ours_v3.OursV3).
`from p24.policy.factory_v3 import make_policy` is a drop-in superset; the simulator takes policy OBJECTS, so no existing file has to change.
(Registering "Ours-v3" in factory.py / __init__.py would alter files whose SHAs are frozen in results/S3/policy_params.json.)"""
from __future__ import annotations

from typing import Dict, Optional

from .base import Shape
from .factory import ALIASES, NAMES as NAMES_S3, make_policy as _make_s3
from .ours_v3 import OursV3

V3_NAMES = ("Ours-v3", "OursV3", "Ours_v3")
NAMES = list(NAMES_S3) + ["Ours-v3"]


def make_policy(name: str, *, cost=None, params: Optional[dict] = None, prior: Optional[Dict[Shape, float]] = None,
                oracle_counts: Optional[Dict[Shape, int]] = None):
    if name in V3_NAMES:
        if cost is None:
            raise ValueError("policy Ours-v3 needs `cost`")
        return OursV3(cost, params, prior)
    return _make_s3(name, cost=cost, params=params, prior=prior, oracle_counts=oracle_counts)
