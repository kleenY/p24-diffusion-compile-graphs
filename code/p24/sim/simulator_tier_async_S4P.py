"""Request-level simulator of the layered asynchronous design A-tier (S4-P P4b probe step 1; new file, p24/sim/simulator.py and
simulator_async_S4P.py are not modified).  Operational definitions: results/S4P/p4b/candidates.json (0444, written before any run).

Same cost machinery as p24.sim.simulator (G1 one-time cost / graph pool budget / eviction / padding via plan_step, cost-table steady times),
plus ONE background C1 compile worker as in simulator_async_S4P:
  * the policy (p24.policy.tier_async_S4P.ATier) may queue a background C1 compile of the request's shape at the START of a request;
    compile wall = g x (C1 resid_s + first-call seconds of the component keys not yet queued/compiled), tier cold_<cache> iff the shape is the process'
    first shape else inproc_<cache>; FIFO, one at a time (start = max(now, end of previous compile)).
  * main-thread work that overlaps a compile interval runs at rate 1/(1+f), exact piecewise.  f_g1: CUDA-graph replay steady (G1, C1+G1);
    f_eager: every other main-thread work (eager steady + E0 one-time, G1 capture one-time, C1+G1 re-capture at the switch, graph-less C1 steady).
  * LAZY switch at the first request of a shape after its compile finished: graph-holding shape -> re-capture C1+G1 (cost-table rebuild seconds,
    pool swap under the graph budget with its own G1 freed; does not fit -> graph released, served by C1); graph-less shape -> C1 (switch_s seconds).
  * compiles still running at the end of the stream are abandoned.
Everything else (E0 one-time for a new shape, G1 one-time = resid of the G1 row, `rebuilt` captures, padding) follows p24.sim.simulator.step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..policy import CostView, ProcState, component_keys, plan_step, shape_of
from ..policy.base import CAPTURE, COMPILE, COMPILE_CAPTURE, EAGER, RUN_BUCKET
from .costs import tier_for
from .simulator_async_S4P import run_work

GRAPH_ARMS = ("G1", "C1+G1")


@dataclass
class TierOptions:
    cache_mode: str = "warm"            # 'warm' (pre-warmed disk cache) | 'empty'
    f_g1: float = 0.197                 # interference on graph replay while a compile runs
    f_eager: float = 8.6                # interference on every other main-thread work while a compile runs
    g: float = 2.76                     # compile wall = g x cost-table one-time cost
    mem_budget_mb: Optional[float] = None
    overhead_aware: bool = True
    switch_s: float = 0.0               # seconds charged once per graph-less shape at its first C1 request
    input_s: float = 0.0
    keep_events: bool = False


def run_pieces(t: float, pieces: List[Tuple[float, float]], busy: List[tuple]) -> Tuple[float, float]:
    """Sequential (work_s, f) pieces from time t -> (end time, total interference seconds = elapsed - nominal work)."""
    t0, nominal = t, 0.0
    for w, f in pieces:
        if w <= 0.0:
            continue
        nominal += w
        t = run_work(t, w, f, busy)
    return t, (t - t0) - nominal


def simulate_tier_async(requests: List[dict], policy, ct, opt: TierOptions) -> dict:
    assert opt.f_g1 >= 0.0 and opt.f_eager >= 0.0 and opt.g > 0.0
    st = ProcState(mem_budget_mb=opt.mem_budget_mb, overhead_aware=opt.overhead_aware)
    cv = policy.cost if isinstance(policy.cost, CostView) else CostView(ct, opt.cache_mode)
    t = 0.0
    busy: List[tuple] = []                       # compile intervals (start, end), appended in order
    pending: List[tuple] = []                    # (end, shape) FIFO
    queued: Set = set()
    queued_keys: Set[tuple] = set()
    ready: Set = set()                           # compile finished, not yet switched
    last_end, comp_wall, comp_log = 0.0, 0.0, []
    first_shape = None
    events: List[dict] = []
    c = dict(eager=0, g1=0, c1g1=0, c1=0, bucket=0, capture=0, capture_denied=0, recapture=0, released_at_switch=0, switched_c1only=0,
             interf_eager_s=0.0, interf_g1_s=0.0, nominal_s=0.0, peak_graph_mb=0.0)
    for i, req in enumerate(requests):
        shape = shape_of(req)
        if first_shape is None:
            first_shape = shape
        while pending and pending[0][0] <= t + 1e-12:         # compiles finished by the start of this request
            _, s = pending.pop(0)
            ready.add(s)
        pieces: List[Tuple[float, float, str]] = []           # (work_s, f, class) class in {'e','g'}
        # ---- lazy switch of a finished shape
        if shape in ready:
            ready.discard(shape)
            st.compiled_shapes.add(shape)
            st.compiled_keys.update(component_keys(shape).values())
            if st.graph_kind.get(shape) == "G1":
                mb = ct.capture_pool_mb("C1+G1", shape)
                if st.can_admit(mb, (shape,)):
                    del st.graphs[shape]
                    st.graphs[shape] = mb
                    st.graph_kind[shape] = "C1+G1"
                    st.build_count[shape] = st.build_count.get(shape, 0) + 1
                    pieces.append((cv.rebuild_s("C1+G1", shape), opt.f_eager, "e"))
                    c["recapture"] += 1
                else:
                    del st.graphs[shape]
                    st.graph_kind.pop(shape, None)
                    c["released_at_switch"] += 1
            elif shape not in st.graphs and opt.switch_s > 0.0:
                pieces.append((opt.switch_s, opt.f_eager, "e"))
                c["switched_c1only"] += 1
        tier = tier_for(opt.cache_mode, st.first_shape_pending)
        # ---- trigger: queue a background compile
        if policy.wants_compile(req, st, queued, queued_keys):
            ctier = tier_for(opt.cache_mode, shape == first_shape)
            ot = ct.one_time("C1", shape, ctier)
            keys = component_keys(shape)
            known = queued_keys | st.compiled_keys
            wc = opt.g * (ot.resid_s + sum(ot.fc[cc] for cc, k in keys.items() if k not in known))
            start = max(t, last_end)
            end = start + wc
            busy.append((start, end))
            last_end = end
            pending.append((end, shape))
            queued.add(shape)
            queued_keys.update(keys.values())
            comp_wall += wc
            comp_log.append((i, shape.tag(), ctier, wc, start, end))
        # ---- decide + serve (same bookkeeping as simulator.step)
        act = policy.decide(req, st)
        assert act.kind in (EAGER, CAPTURE, COMPILE, COMPILE_CAPTURE, RUN_BUCKET), act.kind
        graph_arm = {EAGER: "E0", CAPTURE: "G1", COMPILE: "C1", COMPILE_CAPTURE: "C1+G1"}.get(act.kind)
        mb = ct.capture_pool_mb(graph_arm, shape) if act.kind in (CAPTURE, COMPILE_CAPTURE) else 0.0
        ev, evicted = plan_step(act, st, shape, mb)
        assert ev in ("steady", "eager_first", "capture", "capture_denied"), (ev, act.kind, shape.tag())   # no synchronous compile in A-tier
        serve_arm = graph_arm
        if act.kind == RUN_BUCKET:
            serve_arm = "G1" if st.graph_kind[act.bucket] == "G1" else "C1+G1"
        elif ev == "capture_denied":
            serve_arm = "E0"
        elif ev == "steady" and act.kind in (CAPTURE, COMPILE_CAPTURE) and shape in st.graph_kind:
            serve_arm = "G1" if st.graph_kind[shape] == "G1" else "C1+G1"
        steady_shape = act.bucket if act.kind == RUN_BUCKET else shape
        if opt.input_s > 0.0:
            pieces.append((opt.input_s, opt.f_eager, "e"))
        rebuilt = shape in st.build_count and shape not in st.graphs and shape not in evicted
        if ev == "eager_first" or (ev == "capture_denied" and shape not in st.eager_seen):
            pieces.append((ct.one_time("E0", shape, tier).total_s, opt.f_eager, "e"))
        elif ev == "capture":
            ot_s = cv.rebuild_s("G1", shape) if rebuilt else ct.one_time("G1", shape, tier).total_s
            pieces.append((ot_s, opt.f_eager, "e"))
        w_steady = ct.steady_s(serve_arm, steady_shape)
        pieces.append((w_steady, opt.f_g1 if serve_arm in GRAPH_ARMS else opt.f_eager, "g" if serve_arm in GRAPH_ARMS else "e"))
        st.register(shape, ev, mb, evicted=evicted, served_bucket=act.bucket if act.kind == RUN_BUCKET else None)
        policy.on_event(req, act, ev, st)
        # ---- time
        t_end = t
        for w, f, cl in pieces:
            if w <= 0.0:
                continue
            t_new = run_work(t_end, w, f, busy)
            c["nominal_s"] += w
            c["interf_g1_s" if cl == "g" else "interf_eager_s"] += (t_new - t_end) - w
            t_end = t_new
        c["peak_graph_mb"] = max(c["peak_graph_mb"], st.graph_mem_mb)
        if ev == "capture":
            c["capture"] += 1
        elif ev == "capture_denied":
            c["capture_denied"] += 1
        if act.kind == RUN_BUCKET:
            c["bucket"] += 1
        elif serve_arm == "E0":
            c["eager"] += 1
        elif serve_arm == "G1":
            c["g1"] += 1
        elif serve_arm == "C1+G1":
            c["c1g1"] += 1
        else:
            c["c1"] += 1
        if opt.keep_events:
            events.append(dict(idx=i, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, action=act.kind, event=ev, serve_arm=serve_arm,
                               compile_active=int(any(a <= t < b for a, b in busy)), dur_s=t_end - t, t_end_s=t_end, graph_mem_mb=st.graph_mem_mb))
        t = t_end
    # overlap of the stream with the compile windows
    overlap = sum(max(0.0, min(b, t) - a) for a, b in busy if a < t)
    out = dict(policy=policy.name, n_requests=len(requests), service_s=t, n_shapes=len({shape_of(r) for r in requests}),
               n_queued=len(queued), n_compiled_by_end=len(st.compiled_shapes) + len(ready), bg_compile_wall_s=comp_wall, bg_compile_end_s=last_end,
               window_overlap_s=overlap, f_g1=opt.f_g1, f_eager=opt.f_eager, g=opt.g, cache_mode=opt.cache_mode, compile_log=comp_log, **c)
    if opt.keep_events:
        out["events"] = events
    return out
