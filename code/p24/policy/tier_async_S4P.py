"""A-tier (S4-P P4b probe, exploratory; card goals/S4P_P4b_tiered_async_probe_card_v1.md, candidates results/S4P/p4b/candidates.json): layered asynchronous serving.

  * G1 recording: the v2 policy (Ours, S3 frozen params, graph budget / prior / gdsf eviction / B4 padding) with the ladder restricted to (E, G1):
    it decides, synchronously on the main thread, which shapes get a manually captured CUDA graph.  A compile never creates a graph by itself.
  * Background C1 compile: queued by a TRIGGER rule (candidates: k-th request of the shape, k in {1,2,8}; or the v2 ski-rental cumulative-saving threshold,
    non-blocking).  The simulator (p24/sim/simulator_tier_async_S4P.py) runs one compile at a time, adds the interference of the compile to the main thread
    and switches a finished shape to C1+G1 (graph-holding shapes) / C1 (graph-less shapes) lazily at its next request.
Uses no true N, no future requests, no measured timings; predicted numbers only (CostView).  New file; no frozen file is touched.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from .base import CAPTURE, COMPILE, COMPILE_CAPTURE, EAGER, RUN_BUCKET, Action, ProcState, Shape, component_keys, shape_of
from .ours import C1, C1G1, E, G1, Ours

TRIGGERS = ("kth_request", "ski_rental")


class ATier:
    arm = None

    def __init__(self, cost, params: Optional[dict] = None, prior: Optional[Dict[Shape, float]] = None, trigger: str = "kth_request",
                 k: int = 1, name: Optional[str] = None):
        assert trigger in TRIGGERS and k >= 1
        self.cost = cost
        self.trigger, self.k = trigger, int(k)
        p = dict(params or {})
        p["tiers"] = (E, G1)                       # v2 rule for G1 recording only; compile is the trigger's business
        self.g1 = Ours(cost, p, prior)
        self.alpha = self.g1.p["alpha"]
        self.name = name or (f"A-tier:k{self.k}" if trigger == "kth_request" else "A-tier:ski")
        self._sc: Dict[Shape, float] = {}          # ski-rental: saving so far vs the C1 target (graph-less shape)
        self._scg: Dict[Shape, float] = {}         # ski-rental: saving so far vs the C1+G1 target (shape holding a G1 graph)

    # ------------------------------------------------------------------------------------------ decision
    def decide(self, request: dict, state: ProcState) -> Action:
        return self.g1.decide(request, state)      # eager / capture / compile(steady, compiled shape) / compile_capture(steady) / run_bucket

    def _U(self, arm: str, s: Shape, state: ProcState, queued_keys: Set[tuple]) -> float:
        """predicted blocking one-time seconds of building `arm` for s (v2 _U), component keys already compiled or queued credited"""
        return self.cost.one_time_s(arm, s, compiled_keys=set(state.compiled_keys) | set(queued_keys))

    def wants_compile(self, request: dict, state: ProcState, queued: Set[Shape], queued_keys: Set[tuple]) -> bool:
        s = shape_of(request)
        if s in queued or s in state.compiled_shapes:
            return False
        if self.trigger == "kth_request":
            return state.seen_count.get(s, 0) + 1 >= self.k          # this request is the (seen+1)-th of the shape
        if state.graph_kind.get(s) == "G1":                           # holds a G1 graph -> target C1+G1
            return self._scg.get(s, 0.0) >= self.alpha * self._U("C1+G1", s, state, queued_keys)
        return self._sc.get(s, 0.0) >= self.alpha * self._U("C1", s, state, queued_keys)   # graph-less -> target C1

    # ------------------------------------------------------------------------------------------ bookkeeping (same cases as Ours.on_event)
    def on_event(self, request: dict, act: Action, event: str, state: ProcState) -> None:
        s = shape_of(request)
        self.g1.on_event(request, act, event, state)
        if event in ("capture", "compile_capture", "compile"):
            self._sc.pop(s, None)
            self._scg.pop(s, None)
            return
        if self.trigger != "ski_rental" or s in state.compiled_shapes:
            return
        st = self.cost.steady_s
        if act.kind == CAPTURE and event == "steady":                 # G1 replay of a graph-holding shape
            x = st("G1", s)
            self._scg[s] = self._scg.get(s, 0.0) + x - st("C1+G1", s)
            return
        if act.kind == RUN_BUCKET:
            kind = state.graph_kind.get(act.bucket)
            x = st("G1" if kind == "G1" else "C1+G1", act.bucket)
        else:
            x = st("E0", s)
        self._sc[s] = self._sc.get(s, 0.0) + x - st("C1", s)
