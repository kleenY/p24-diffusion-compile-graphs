"""A-async (S4-P P4 probe, exploratory; card goals/S4P_P4_async_compile_probe_card_v1.md): background compilation + eager fallback.

Rule (card step 1): every NEW shape triggers a background C1 compile at its first request (one compile at a time, FIFO queue);
requests of any shape are served eagerly (E0) until the compile of THEIR shape has finished; afterwards that shape is served by C1.
Only C1 is used (no CUDA-graph tier in the probe).  The policy decides; the timing (compile wall clock, interference f, factor g) lives in the
simulator p24/sim/simulator_async_S4P.py.  Uses no true N, no future requests, no measured timings.
New file; no frozen file is touched.
"""
from __future__ import annotations

from .base import COMPILE, EAGER, Action, shape_of


class AAsync:
    name, arm = "A-async", None

    def wants_compile(self, request: dict, state, queued) -> bool:
        """Enqueue a background compile of the request's shape iff it was never queued (first sight)."""
        return shape_of(request) not in queued

    def decide(self, request: dict, state) -> Action:
        """C1 once the shape's background compile has finished (state.compiled_shapes is filled by the simulator), else eager."""
        return Action(COMPILE) if shape_of(request) in state.compiled_shapes else Action(EAGER)
