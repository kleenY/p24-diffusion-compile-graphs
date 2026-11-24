"""Request-level simulator with ONE background compile worker (S4-P P4 probe step 1; new file, p24/sim/simulator.py is not modified).

Time model (closed loop, sequential requests, like p24.sim.simulator):
  * A policy with `wants_compile(req, state, queued)` may enqueue a background C1 compile of the request's shape at the START of that request.
    Compile wall clock  Wc = g x (resid_s + sum of the first-call seconds of the component keys NOT yet queued)  from the cost table C1 row, at the
    tier of that moment (first shape of the process -> cold_<cache>, later -> inproc_<cache>; the same convention as B1 in the simulator).
    One compile at a time (FIFO): start = max(now, end of the previous compile).  Component keys of earlier queued compiles are credited
    (they are finished when the later compile starts).
  * Compile interference: eager work that overlaps a compile interval runs at rate 1/(1+f)  (exact piecewise: a request that straddles the end of
    a compile is slowed only for the overlapped part).  f_scope='eager' (card default: only eager requests are slowed) or 'all' (sensitivity:
    C1 requests during a compile are slowed too).
  * Eager request time = E0 steady + (E0 one-time cost of a NEW shape at the process tier).  C1 request (shape finished compiling at or before the
    request start) = C1 steady, no one-time cost (the compile + first call were done in the background); optional `switch_s` seconds per shape at
    its first C1 request (default 0; the card's step 2 measures the real switch).
  * The decision (eager or C1) is taken at request start from the finished compiles.  Compiles still running at the end of the stream are abandoned.
Memory / graph tiers are not modelled (C1 has no graph pool).  Only the cost table is used (no real timing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from ..policy import ProcState, component_keys, shape_of
from ..policy.base import COMPILE, EAGER
from .costs import tier_for


@dataclass
class AsyncOptions:
    cache_mode: str = "warm"      # 'warm' (pre-warmed disk cache) | 'empty'
    f: float = 0.0                # interference: eager work during a compile runs (1+f) x slower
    g: float = 1.0                # background compile wall clock = g x the cost-table one-time cost
    f_scope: str = "eager"        # 'eager' (card) | 'all' (sensitivity)
    switch_s: float = 0.0         # seconds charged once per shape at its first C1 request
    input_s: float = 0.0
    # subprocess alternative (card step 2 fallback): seconds the MAIN process is blocked when it first loads a shape's compiled code from the disk cache
    # filled by the helper process: switch_fn(shape, loaded_keys) -> s (loaded_keys = component keys already loaded in the main process); None = off
    switch_fn: Optional[Callable] = None


def run_work(t: float, w: float, f: float, busy: List[tuple]) -> float:
    """End time of `w` seconds of work started at `t`; inside the busy intervals the work progresses at rate 1/(1+f). busy: sorted, disjoint."""
    if f == 0.0 or w <= 0.0:
        return t + w
    inv = 1.0 / (1.0 + f)
    for a, b in busy:
        if b <= t:
            continue
        if a > t:
            gap = a - t
            if w <= gap:
                return t + w
            w -= gap
            t = a
        cap = (b - t) * inv
        if w <= cap:
            return t + w * (1.0 + f)
        w -= cap
        t = b
    return t + w


def simulate_async(requests: List[dict], policy, ct, opt: AsyncOptions) -> dict:
    """-> summary dict (service_s, counts, ...) and events list under key 'events'."""
    assert opt.f >= 0.0 and opt.g > 0.0 and opt.f_scope in ("eager", "all")
    st = ProcState()
    t = 0.0
    busy: List[tuple] = []           # compile intervals (start, end), appended in order
    pending: List[tuple] = []        # (ready_at, shape) FIFO
    queued, queued_keys = set(), set()
    last_end = 0.0
    switched = set()
    loaded_keys: Set[tuple] = set()
    events, n_eager, n_c1, comp_wall, comp_log = [], 0, 0, 0.0, []
    for i, req in enumerate(requests):
        shape = shape_of(req)
        while pending and pending[0][0] <= t + 1e-12:     # compiles finished by the start of this request
            _, s = pending.pop(0)
            st.compiled_shapes.add(s)
            st.compiled_keys.update(component_keys(s).values())
        tier = tier_for(opt.cache_mode, st.first_shape_pending)
        if policy.wants_compile(req, st, queued):
            ot = ct.one_time("C1", shape, tier)
            keys = component_keys(shape)
            wc = opt.g * (ot.resid_s + sum(ot.fc[c] for c, k in keys.items() if k not in queued_keys))
            start = max(t, last_end)
            end = start + wc
            busy.append((start, end))
            last_end = end
            pending.append((end, shape))
            queued.add(shape)
            queued_keys.update(keys.values())
            comp_wall += wc
            comp_log.append((i, shape.tag(), tier, wc, start, end))
        act = policy.decide(req, st)
        active = any(a <= t < b for a, b in busy)
        if act.kind == COMPILE:
            w = ct.steady_s("C1", shape) + opt.input_s
            if shape not in switched:
                w += opt.switch_s
                if opt.switch_fn is not None:
                    w += float(opt.switch_fn(shape, loaded_keys))
                    loaded_keys.update(component_keys(shape).values())
                switched.add(shape)
            ev = "c1"
            t_end = run_work(t, w, opt.f if opt.f_scope == "all" else 0.0, busy)
            n_c1 += 1
        else:
            assert act.kind == EAGER, act.kind
            w = ct.steady_s("E0", shape) + opt.input_s
            ev = "steady_eager"
            if shape not in st.eager_seen:
                ev = "eager_first"
                w += ct.one_time("E0", shape, tier).total_s
            t_end = run_work(t, w, opt.f, busy)
            n_eager += 1
        st.register(shape, "eager_first" if ev == "eager_first" else "steady")   # bookkeeping: shapes_seen / eager_seen / counters
        events.append(dict(idx=i, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, event=ev, compile_active=int(active), work_s=w,
                           dur_s=t_end - t, t_end_s=t_end))
        t = t_end
    n_ready = len(st.compiled_shapes)
    return dict(policy=policy.name, n_requests=len(requests), service_s=t, n_eager=n_eager, n_c1=n_c1, n_shapes=len(queued),
                n_compiled_by_end=n_ready, bg_compile_wall_s=comp_wall, bg_compile_end_s=last_end, f=opt.f, g=opt.g, f_scope=opt.f_scope,
                cache_mode=opt.cache_mode, events=events, compile_log=comp_log)
