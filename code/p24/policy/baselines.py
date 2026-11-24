"""S3 baselines B2, B3 (truncated), B3', B3'', B5, B6, B6', B7 (B0/B1/B4 live in fixed.py).  Interface v1 sec. 5.

Every baseline is built from the construction-time injected CostView / prior / params (predicted numbers only);
none reads the true N. B7 alone receives `oracle_counts` (known per-shape request counts; "not the true optimum, known counts").
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import (CAPTURE, COMPILE, COMPILE_CAPTURE, COMPILE_DYNAMIC, COMPILE_RO, EAGER, RUN_BUCKET, Action, ProcState,
                   Shape, shape_of)
from .fixed import Policy
from .ours import ARM, Ours

OV_DEFAULT = 22.33


def _ov(state: ProcState) -> float:
    return state.per_graph_overhead_mb if state.overhead_aware else 0.0


def prior_order(prior: Dict[Shape, float]) -> List[Shape]:
    return sorted(prior, key=lambda s: (-prior[s], s))


# ------------------------------------------------------------------------------------------------------------------ B2
class B2Dynamic(Policy):
    """B2: torch.compile dynamic=True (C5), compile on first encounter."""
    name, arm = "B2", "C5"

    def decide(self, request, state):
        return Action(COMPILE_DYNAMIC)


# ------------------------------------------------------------------------------------------------------------------ B3
class B3ReduceOverhead(Policy):
    """B3 (truncated): reduce-overhead compile per shape (C2). Pools cannot be released per shape (failure #15): once the
    cumulative PREDICTED pool (+22.33 MB per shape) would exceed the budget, a NEW shape is served eagerly. Simulator only."""
    name, arm = "B3", "C2"

    def __init__(self, cost):
        self.cost = cost
        self.pool_sum_mb = 0.0
        self.n_pools = 0

    def decide(self, request, state):
        s = shape_of(request)
        if s in state.compiled_shapes:
            return Action(COMPILE_RO)
        if state.mem_budget_mb is not None:
            need = self.cost.capture_pool_mb("C2", s) + OV_DEFAULT
            if self.pool_sum_mb + self.n_pools * OV_DEFAULT + need > state.mem_budget_mb + 1e-9:
                return Action(EAGER)
        return Action(COMPILE_RO)

    def on_event(self, request, act, event, state):
        if act.kind == COMPILE_RO and event == "compile":
            self.pool_sum_mb += self.cost.capture_pool_mb("C2", shape_of(request))
            self.n_pools += 1


# ------------------------------------------------------------------------------------------------------------------ B3'
class B3PrimeLimit8(Policy):
    """B3' = C1 + PyTorch default recompile limit 8: compile on first encounter; if compiling a new shape would put any
    component over 8 keys the WHOLE request is served eagerly (event eager_fallback_limit, produced by plan_step from
    state.recompile_limit which the simulator/runtime copy from `recompile_limit`)."""
    name, arm = "B3p", "C1"
    recompile_limit = 8

    def decide(self, request, state):
        return Action(COMPILE)


# ------------------------------------------------------------------------------------------------------------------ B3''
class B3PostWarm(Policy):
    """B3'': warm the prior's top-k shapes with C1 at start-up (time counted), then lock (eager_on_recompile): compiled shape ->
    compiled, any new shape -> eager. Simulator only."""
    name, arm = "B3pp", "C1"

    def __init__(self, cost, prior, params=None):
        self.cost = cost
        self.prior = dict(prior)
        self.k = int((params or {}).get("k", 8))

    def warmup(self, state, cost):
        return [(s, Action(COMPILE)) for s in prior_order(self.prior)[:self.k]]

    def decide(self, request, state):
        return Action(COMPILE) if shape_of(request) in state.compiled_shapes else Action(EAGER)


# ------------------------------------------------------------------------------------------------------------------ B5
def make_B5(cost, params=None, prior=None) -> Ours:
    """B5 LRU: identical admission/tiers/padding as Ours, only the eviction is replaced by least-recently-used."""
    p = dict(params or {})
    p["evict_score"] = "lru"
    p["evict"] = True
    return Ours(cost, p, prior, name="B5")


def make_Ours_NE(cost, params=None, prior=None) -> Ours:
    p = dict(params or {})
    p["evict"] = False
    return Ours(cost, p, prior, name="Ours-NE")


# ------------------------------------------------------------------------------------------------------------------ B6
class B6BatchBuckets(Policy):
    """B6 vLLM-style: per resolution x CFG the batch buckets {1,2,4} are pre-built (C1+G1) at start-up (time counted), ordered by
    prior mass and truncated to the budget; B=3 requests are padded into the B4 bucket only if equiv_pass_padB4 == 1.
    No bucket (not built / not allowed) -> eager. No dynamic building, no eviction."""
    name, arm = "B6", "C1+G1"
    BUCKETS = (1, 2, 4)

    def __init__(self, cost, prior, params=None):
        self.cost = cost
        self.prior = dict(prior)

    def _bucket_of(self, s: Shape) -> Optional[Shape]:
        if s.B in self.BUCKETS:
            return s
        if s.B == 3 and self.cost.equiv_pass_padB4(s, "C1+G1"):
            return Shape(s.H, s.W, 4, s.cfg)
        return None

    def warmup(self, state, cost):
        mass: Dict[Shape, float] = {}
        for s, p in self.prior.items():
            b = self._bucket_of(s)
            if b is not None:
                mass[b] = mass.get(b, 0.0) + p
        for H, W, c in {(s.H, s.W, s.cfg) for s in self.prior}:
            for b in self.BUCKETS:
                mass.setdefault(Shape(H, W, b, c), 0.0)
        out, used = [], 0.0
        budget = state.mem_budget_mb
        for b in sorted(mass, key=lambda x: (-mass[x], x)):
            need = cost.capture_pool_mb("C1+G1", b) + _ov(state)
            if budget is not None and used + need > budget + 1e-9:
                continue
            used += need
            out.append((b, Action(COMPILE_CAPTURE)))
        return out

    def decide(self, request, state):
        s = shape_of(request)
        b = self._bucket_of(s)
        if b is None or b not in state.graphs:
            return Action(EAGER)
        return Action(COMPILE_CAPTURE) if b == s else Action(RUN_BUCKET, bucket=b)


class B6PrimeTopK(Policy):
    """B6': pre-build the prior's top-k EXACT shapes with C1+G1 at start-up (time counted, truncated to the budget); others eager."""
    name, arm = "B6p", "C1+G1"

    def __init__(self, cost, prior, params=None):
        self.cost = cost
        self.prior = dict(prior)
        self.k = int((params or {}).get("k", 8))

    def warmup(self, state, cost):
        out, used, budget = [], 0.0, state.mem_budget_mb
        for s in prior_order(self.prior)[:self.k]:
            need = cost.capture_pool_mb("C1+G1", s) + _ov(state)
            if budget is not None and used + need > budget + 1e-9:
                continue
            used += need
            out.append((s, Action(COMPILE_CAPTURE)))
        return out

    def decide(self, request, state):
        return Action(COMPILE_CAPTURE) if shape_of(request) in state.graphs else Action(EAGER)


# ------------------------------------------------------------------------------------------------------------------ B7
def solve_mck(options: Sequence[Sequence[Tuple[str, float, float]]], budget: float, unit: float = 0.1):
    """Multiple-choice knapsack. options[i] = [(label, weight, value), ...] for item i (choose at most one; nothing = value 0).
    Maximise total value s.t. total weight <= budget. Weights are rounded UP to the `unit` grid and the budget DOWN, so the plan is
    feasible for the exact weights. Exact DP on the grid. Returns (best_value, [label or None per item])."""
    cap = int(math.floor(budget / unit + 1e-9))
    n = len(options)
    W = [[int(math.ceil(w / unit - 1e-9)) for (_, w, _) in opt] for opt in options]
    tables = [np.zeros(cap + 1)]
    for i, opt in enumerate(options):
        prev = tables[-1]
        new = prev.copy()
        for j, (_, _, v) in enumerate(opt):
            wj = W[i][j]
            if wj > cap or v <= 0:
                continue
            cand = np.full(cap + 1, -np.inf)
            cand[wj:] = prev[:cap + 1 - wj] + v
            new = np.maximum(new, cand)
        tables.append(new)
    labels: List[Optional[str]] = [None] * n
    c = cap
    for i in range(n - 1, -1, -1):
        if tables[i + 1][c] <= tables[i][c] + 1e-12:
            continue                                     # item i not chosen in an optimal plan
        for j, (lab, _, v) in enumerate(options[i]):
            wj = W[i][j]
            if wj <= c and v > 0 and abs(tables[i][c - wj] + v - tables[i + 1][c]) < 1e-9:
                labels[i] = lab
                c -= wj
                break
    return float(tables[-1][cap]), labels


class B7Knapsack(Policy):
    """B7: oracle multiple-choice knapsack. Knowing the per-shape request counts of the trace (`oracle_counts`), choose for every shape
    eager / G1 / C1+G1 to maximise total saving n x (eager - steady) - one-time cost under the graph budget (weight = pool + 22.33 MB;
    compile cost counted per shape in full, i.e. no shared-key credit). Planned shapes are pre-built at start-up (time counted);
    no eviction. NOT the true optimum (shared compile keys and tiers are approximated) -> labelled 'known counts'."""
    name, arm = "B7", None

    def __init__(self, cost, oracle_counts, params=None):
        assert oracle_counts is not None, "B7 needs oracle_counts"
        self.cost = cost
        self.counts = dict(oracle_counts)
        self.plan: Dict[Shape, str] = {}
        self.plan_value_s = 0.0

    def make_plan(self, budget_mb: Optional[float], ov: float = OV_DEFAULT):
        shapes = sorted(self.counts, key=lambda s: (-self.counts[s], s))
        opts = []
        for s in shapes:
            n = self.counts[s]
            e = self.cost.steady_s("E0", s)
            o = []
            for lab, arm in (("G1", "G1"), ("C1+G1", "C1+G1")):
                U = self.cost.one_time_s(arm, s)
                if math.isinf(U):
                    continue
                o.append((lab, self.cost.capture_pool_mb(arm, s) + ov, n * (e - self.cost.steady_s(arm, s)) - U))
            opts.append(o)
        budget = budget_mb if budget_mb is not None else 1e7
        best, labels = solve_mck(opts, budget)
        self.plan = {s: lab for s, lab in zip(shapes, labels) if lab}
        self.plan_value_s = best
        return self.plan

    def warmup(self, state, cost):
        self.make_plan(state.mem_budget_mb, state.per_graph_overhead_mb)
        out = []
        for s in sorted(self.plan, key=lambda s: (-self.counts[s], s)):
            out.append((s, Action(CAPTURE if self.plan[s] == "G1" else COMPILE_CAPTURE)))
        return out

    def decide(self, request, state):
        lab = self.plan.get(shape_of(request))
        return Action(EAGER) if lab is None else Action(CAPTURE if lab == "G1" else COMPILE_CAPTURE)
