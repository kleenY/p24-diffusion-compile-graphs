"""A-half shape prior p_A(shape) for the S3 policies = S2 `workload_stats.json` A mapped resolution distribution x declared
B / CFG distribution (the same construction as scripts/make_budgets_S3.py). A half only; evaluation traces never enter."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from ..policy.base import Shape
from ..workload import generate as G

ROOT = Path(os.environ.get("PROJ_ROOT", "/data/kunlin-data/HPC 课程/project"))
STATS = ROOT / "结果/S2/workload/workload_stats.json"


def prior_A(stats_path=STATS) -> Dict[Shape, float]:
    d = json.loads(Path(stats_path).read_text())["A_mapped_shape_dist"]
    pr: Dict[Shape, float] = {}
    for H, W in G.RESOLUTIONS:
        for b, pb in zip(G.B_VALUES, G.B_PROBS):
            for c, pc in ((0, 1 - G.CFG_PROB), (1, G.CFG_PROB)):
                pr[Shape(H, W, b, c)] = d[f"H{H}xW{W}"] * pb * pc
    s = sum(pr.values())
    assert abs(s - 1.0) < 1e-6, s
    return pr
