"""Our policy (S3 sec. 5/10), version v2 (2026-11-17, AUTO_DECISIONS "S3 3.2 policy design revision v2"; the v1 backup is kept in the internal tree):
FOUR tiers eager -> G1 (manual CUDA graph) -> C1 (compile only, no graph memory) -> C1+G1 (compile + graph), graph-memory budget,
prior admission, online (ski-rental) upgrade, GreedyDual-Size-style eviction of GRAPHS (the compile layer is never evicted: a shape
whose graph was evicted keeps its compiled code and is served by it) and batch padding.
v2 = (a) online horizon estimate H_eff = max(H_prior, n_served) instead of the fixed H_prior  [params horizon="online"]
     (b) the four-tier ladder (when the budget is full a shape can still be upgraded to C1 only)  [params tiers=(E,G1,C1,C1+G1)].
v1_spec (3 tiers, fixed horizon; the policy frozen in policy_params_v1_spec.json) = params V1_SPEC_PARAMS (horizon="fixed", tiers=(E,G1,C1+G1));
ablations "fixed horizon" = horizon="fixed" only, "three tiers" = tiers=(E,G1,C1+G1) only.

All ablations are `params` switches of ONE class (no new classes): B5 = Ours with evict_score='lru' (shared admission, LRU eviction),
Ours-NE = Ours with evict=False (no eviction; over budget -> eager / capture_denied).
Everything is built from the construction-time injected CostView (predicted numbers only), a prior over shapes and `params`.
The policy never sees the true number of requests N: `H_prior` is a parameter tuned on development seeds and H_eff only uses the
number of requests served so far (n_served).
No `if simulator/runtime` branches: the same code runs in p24.sim.simulator and scripts/serve_trace.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .base import CAPTURE, COMPILE, COMPILE_CAPTURE, EAGER, RUN_BUCKET, Action, ProcState, Shape, shape_of
from .fixed import Policy

E, G1, C1, C1G1 = "E", "G1", "C1", "C1+G1"
ARM = {E: "E0", G1: "G1", C1: "C1", C1G1: "C1+G1"}
KIND = {G1: CAPTURE, C1: COMPILE, C1G1: COMPILE_CAPTURE}
RANK = {E: 0, G1: 1, C1: 2, C1G1: 3}       # = ranking by steady speed in the cost table (E0 slowest ... C1+G1 fastest)
DENY = object()

DEFAULT_PARAMS = dict(
    alpha=1.0,            # upgrade when cumulative saved seconds >= alpha x one-time upgrade cost (ski-rental)
    H_prior=1000.0,       # horizon floor (requests) used by the prior admission and the eviction value; NOT the true N
    horizon="online",     # v2 (a): H_eff = max(H_prior, n_served) (online estimate of the remaining requests); "fixed": H_eff = H_prior (v1)
    prior_admit=True,     # ablation "no prior": False -> only the online (saved_s) path; value estimates use online counts only
    pad=True,             # ablation "no padding": False -> never run_bucket
    evict=True,           # Ours-NE: False
    evict_score="gdsf",   # 'gdsf' (expected remaining saving x rebuild cost / memory) | 'lru' (= B5)
    key_decomp=True,      # ablation "no component-key decomposition": False -> compile cost counted in full (no shared-key credit)
    tiers=(E, G1, C1, C1G1),  # v2 (b): eager -> G1 -> C1 (compile only, memory-free) -> C1+G1. v1 ladder (E, G1, C1+G1) = ablation "three tiers".
                          # A shape holding a G1 graph moves only to C1+G1 (never to C1: it would keep a useless graph).
    prior_weight=50.0,    # pseudo-count weight of the prior in the online frequency estimate (fixed, not tuned)
    backoff0=8,           # after a refused admission, retry after backoff0 * 2**fails more requests of the process
)


V1_SPEC_PARAMS = dict(horizon="fixed", tiers=(E, G1, C1G1))     # the v1 policy (three tiers, fixed horizon): ablation / history


def merge_params(params: Optional[dict]) -> dict:
    p = dict(DEFAULT_PARAMS)
    for k, v in (params or {}).items():
        if k not in DEFAULT_PARAMS:
            raise KeyError(f"unknown policy param {k!r}")
        p[k] = v
    p["tiers"] = tuple(p["tiers"])
    assert p["tiers"][0] == E and set(p["tiers"]) <= set(RANK), p["tiers"]
    assert list(p["tiers"]) == sorted(p["tiers"], key=RANK.get), p["tiers"]
    assert p["evict_score"] in ("gdsf", "lru")
    assert p["horizon"] in ("online", "fixed"), p["horizon"]
    return p


class Ours(Policy):
    name, arm = "Ours", None

    def __init__(self, cost, params: Optional[dict] = None, prior: Optional[Dict[Shape, float]] = None, name: Optional[str] = None):
        assert cost is not None, "Ours needs a CostView"
        self.cost = cost
        self.p = merge_params(params)
        if name:
            self.name = name
        self.prior: Dict[Shape, float] = dict(prior or {})
        self.ladder = self.p["tiers"]
        self._w = self.p["prior_weight"] if self.p["prior_admit"] else 0.0
        self._sv: Dict[Shape, Dict[str, float]] = {}   # seconds saved so far at the shape's current tier, had each HIGHER tier been in place
        self._fails: Dict[Shape, int] = {}
        self._next_try: Dict[Shape, int] = {}

    # ---------------------------------------------------------------- small helpers (predicted numbers only)
    def _tier(self, s: Shape, state: ProcState) -> str:
        k = state.graph_kind.get(s)
        if k == "C1+G1":
            return C1G1
        if k == "G1":
            return G1
        return C1 if s in state.compiled_shapes else E      # compile layer is never evicted: a shape whose graph was released stays compiled

    def _steady(self, tier: str, s: Shape) -> float:
        return self.cost.steady_s(ARM[tier], s)

    def _U(self, tier: str, s: Shape, state: ProcState) -> float:
        """Predicted one-time cost of moving shape s to `tier` now (same rule the simulator charges): re-capture only (capture_s) for a
        shape whose graph was evicted (G1: any time; C1+G1: if its compiled code is kept), else the cost-table one-time cost."""
        if tier != C1 and s in state.build_count and s not in state.graphs and (tier == G1 or s in state.compiled_shapes):
            return self.cost.rebuild_s(ARM[tier], s)
        keys = state.compiled_keys if self.p["key_decomp"] else None
        return self.cost.one_time_s(ARM[tier], s, compiled_keys=keys)

    def _above(self, cur: str) -> List[str]:
        out = [t for t in self.ladder if RANK[t] > RANK[cur]]
        return [t for t in out if not (cur == G1 and t == C1)]      # G1 -> C1 would leave a live graph unused: only G1 -> C1+G1

    def _p_hat(self, s: Shape, state: ProcState) -> float:
        n = state.seen_count.get(s, 0)
        return (n + self._w * self.prior.get(s, 0.0)) / max(state.n_served + self._w, 1.0)

    def _h_eff(self, state: ProcState) -> float:
        """Expected number of requests still to come. v2: max(H_prior, n_served) (n_served = requests served so far; never the true N)."""
        h = self.p["H_prior"]
        return max(h, float(state.n_served)) if self.p["horizon"] == "online" else h

    def _n_hat(self, s: Shape, state: ProcState) -> float:
        return self._p_hat(s, state) * self._h_eff(state)

    def _kind_tier(self, v: Shape, state: ProcState) -> str:
        return C1G1 if state.graph_kind.get(v) == "C1+G1" else G1

    def _saving(self, v: Shape, state: ProcState) -> float:
        """Per-request saving the live graph of v still provides: what the shape would fall back to without it (compiled code if
        the shape is compiled, else eager) minus the graph's steady time."""
        fb = C1 if v in state.compiled_shapes else E
        return max(self._steady(fb, v) - self._steady(self._kind_tier(v, state), v), 0.0)

    def _prio(self, v: Shape, state: ProcState) -> float:
        """GreedyDual-Size style keep value: expected remaining saving x rebuild cost / memory held."""
        mem = state.graphs[v] + (state.per_graph_overhead_mb if state.overhead_aware else 0.0)
        reb = self.cost.rebuild_s(ARM[self._kind_tier(v, state)], v)
        return self._n_hat(v, state) * self._saving(v, state) * max(reb, 1e-6) / max(mem, 1.0)

    # ---------------------------------------------------------------- decision
    def decide(self, request: dict, state: ProcState) -> Action:
        s = shape_of(request)
        cur = self._tier(s, state)
        if cur != self.ladder[-1]:
            backing_off = state.n_served < self._next_try.get(s, 0)      # after a refused graph admission: only the memory-free tier may be tried
            denied = None
            for tgt in self._targets(s, cur, state):
                if backing_off and tgt != C1:
                    continue
                act = self._admit(s, cur, tgt, state)
                if act is DENY:
                    denied = tgt
                elif act is not None:
                    return act
            if denied is not None and cur == E:
                return Action(KIND[denied])          # a NEW shape asks anyway: plan_step answers capture_denied (counted) -> eager
        if cur == C1G1:
            return Action(COMPILE_CAPTURE)
        if cur == G1:
            return Action(CAPTURE)
        if cur == C1:
            return Action(COMPILE)
        if self.p["pad"]:
            b = self._pad_bucket(s, state)
            if b is not None:
                return Action(RUN_BUCKET, bucket=b)
        return Action(EAGER)

    def _pad_bucket(self, s: Shape, state: ProcState) -> Optional[Shape]:
        """Complement (padding): a B<4 shape served by the existing B4 graph of the same resolution/CFG, only if the cost table
        says equiv_pass_padB4 == 1 for that bucket's arm AND the bucket's steady latency < this shape's eager latency."""
        if s.B >= 4:
            return None
        b = Shape(s.H, s.W, 4, s.cfg)
        if b not in state.graphs:
            return None
        arm = ARM[self._kind_tier(b, state)]
        if not self.cost.equiv_pass_padB4(s, arm):
            return None
        return b if self.cost.steady_s(arm, b) < self.cost.steady_s("E0", s) else None

    def _targets(self, s: Shape, cur: str, state: ProcState) -> List[str]:
        """Upgrade targets in order of preference (possibly empty)."""
        cands = self._above(cur)
        if not cands:
            return []
        x = self._steady(cur, s)
        out: List[str] = []
        # (a) prior admission: p_A x H_prior x per-request saving >= one-time cost of the tier (= prior hits >= break-even)
        if self.p["prior_admit"] and s in self.prior:
            nh = self.prior[s] * self._h_eff(state)
            scored = []
            for t in cands:
                sav = x - self._steady(t, s)
                if sav <= 0:
                    continue
                net = nh * sav - self._U(t, s, state)
                if net > 0:
                    scored.append((-net, RANK[t], t))
            out += [t for _, _, t in sorted(scored)]
        if out:
            return out
        # (b) online path: cumulative saved seconds (vs tier t) >= alpha x upgrade cost of t (doubling threshold after refused
        #     admissions); highest qualifying tier first
        thr = self.p["alpha"] * (2.0 ** self._fails.get(s, 0))           # refusals concern graph memory only: C1 (no memory) keeps threshold alpha
        sv = self._sv.get(s, {})
        for t in reversed(cands):
            if sv.get(t, 0.0) >= (self.p["alpha"] if t == C1 else thr) * self._U(t, s, state):
                out.append(t)
        return out

    # ---------------------------------------------------------------- admission / eviction
    def _admit(self, s: Shape, cur: str, tgt: str, state: ProcState):
        """-> Action (admitted, possibly with evictions) | DENY (no room and no worthwhile eviction) | None."""
        if tgt == C1:
            return Action(COMPILE)                    # memory-free tier
        pool = self.cost.capture_pool_mb(ARM[tgt], s)
        free0 = (s,) if s in state.graphs else ()
        if state.can_admit(pool, free0):
            return Action(KIND[tgt])
        victims = self._victims(s, cur, tgt, pool, free0, state) if self.p["evict"] else None
        if victims is not None:
            return Action(KIND[tgt], evict=tuple(victims))
        f = self._fails[s] = self._fails.get(s, 0) + 1                  # refused: back off before asking again
        self._next_try[s] = state.n_served + self.p["backoff0"] * (2 ** f)
        return DENY

    def _victims(self, s, cur, tgt, pool, free0, state) -> Optional[List[Shape]]:
        cand = [v for v in state.graphs if v not in free0]
        if self.p["evict_score"] == "lru":
            order = sorted(cand, key=lambda v: (state.last_used.get(v, 0), v))
        else:
            order = sorted(cand, key=lambda v: (self._prio(v, state), v))
        chosen: List[Shape] = []
        free = list(free0)
        for v in order:
            chosen.append(v)
            free.append(v)
            if state.can_admit(pool, tuple(free)):
                break
        else:
            return None
        if self.p["evict_score"] == "lru":
            return chosen
        gain = self._n_hat(s, state) * (self._steady(cur, s) - self._steady(tgt, s)) - self._U(tgt, s, state)
        loss = sum(self._n_hat(v, state) * self._saving(v, state) for v in chosen)
        return chosen if gain > loss else None

    # ---------------------------------------------------------------- bookkeeping after the event
    def on_event(self, request: dict, act: Action, event: str, state: ProcState) -> None:
        s = shape_of(request)
        if event in ("capture", "compile_capture", "compile"):
            state.saved_s[s] = 0.0
            self._sv[s] = {}
            self._fails.pop(s, None)
            return
        if act.kind == COMPILE_CAPTURE and event == "steady":
            tier, x = C1G1, self._steady(C1G1, s)
        elif act.kind == CAPTURE and event == "steady":
            tier, x = G1, self._steady(G1, s)
        elif act.kind == COMPILE and event == "steady":
            tier, x = C1, self._steady(C1, s)
        elif act.kind == RUN_BUCKET:
            tier, x = E, self.cost.steady_s(ARM[self._kind_tier(act.bucket, state)], act.bucket)
        else:
            tier, x = E, self._steady(E, s)
        up = self._above(tier)
        if up:
            state.saved_s[s] = state.saved_s.get(s, 0.0) + x - self._steady(up[0], s)
            d = self._sv.setdefault(s, {})
            for t in up:
                d[t] = d.get(t, 0.0) + x - self._steady(t, s)
