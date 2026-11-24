"""Policy interface shared by the request-level simulator (p24.sim) and the real replay (scripts/serve_trace.py).

    policy.decide(request, state) -> Action

`state` is a `ProcState` (what the single serving process holds: eager-seen shapes, compiled shapes / component keys,
captured graphs with their memory, optional graph-memory budget). Both the simulator and the real replay own one
ProcState and update it with `ProcState.register`, so a policy sees the same state in both.

Actions (S3 policies return the same set):
  eager            run eagerly (E0)
  capture          ensure a manual per-component CUDA graph (G1) for the request's own shape exists, then replay it
  compile          ensure torch.compile (C1, dynamic=False) artefacts for the request's shape exist, then run compiled
  compile_capture  C1 + G1 (graph captured over compiled components)
  run_bucket(key)  replay an existing graph of another shape `key` (batch padding; S3)
Event types (what happened to this request; identical definitions in simulator and replay):
  steady           nothing new was built
  eager_first      first eager call of this shape in this process
  capture          graph captured for this shape (includes the G1 warm-up)
  compile          first compiled call of this shape (new shape for the compiled runner)
  compile_capture  compile + capture
  capture_denied   capture was requested but would exceed the graph-memory budget -> served eagerly
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Set, Tuple

EAGER, CAPTURE, COMPILE, COMPILE_CAPTURE, RUN_BUCKET = "eager", "capture", "compile", "compile_capture", "run_bucket"
# --- S3 additions (interface v1, 代码/docs/policy_interface_S3.md): appended, nothing above/below is removed ---
EVICT = "evict"                        # event / ledger tag: release one shape's graph (release pool + empty_cache); compile layer is never evicted
COMPILE_DYNAMIC = "compile_dynamic"    # C5 (B2): torch.compile dynamic=True
COMPILE_RO = "compile_ro"              # C2 (B3): torch.compile mode=reduce-overhead (simulator only)
EAGER_FALLBACK_LIMIT = "eager_fallback_limit"  # event: B3' request served eagerly because a component's key count would exceed the limit
ALL_KINDS = (EAGER, CAPTURE, COMPILE, COMPILE_CAPTURE, RUN_BUCKET, COMPILE_DYNAMIC, COMPILE_RO)


class Shape(NamedTuple):
    H: int
    W: int
    B: int
    cfg: int  # 0 / 1

    @property
    def unet_batch(self) -> int:
        return self.B * (2 if self.cfg else 1)

    def tag(self) -> str:
        return f"{self.H}x{self.W}_b{self.B}_{'cfg' if self.cfg else 'nocfg'}"

    def to_shapekey(self):  # lazy: p24.pipeline imports torch / diffusers
        from ..pipeline import ShapeKey
        return ShapeKey(self.H, self.W, self.B, bool(self.cfg))


def shape_of(request: dict) -> Shape:
    return Shape(int(request["H"]), int(request["W"]), int(request["B"]), int(bool(request["cfg"])))


def component_keys(shape: Shape, te_by_resolution: bool = False) -> Dict[str, tuple]:
    """Dynamo cache keys of the three compiled components (plan 9.4, measured in
    结果/S2/pilot/cache_keys_v2_inference_inputs.json): UNet is keyed by its ACTUAL batch (2B with CFG, else B) and the
    resolution, so "B1 CFG-on" and "B2 CFG-off" share it; text encoder and VAE are keyed by B (and resolution).
    cfg is NOT part of the text-encoder key (measured: B2 no-CFG -> B2 CFG-on reused the text encoder graph).
    Text-encoder key = B only (default since 2026-11-15 23:40): its input is (B,77); verified in the S-5 12-shape C1 sequences
    (结果/S2/inproc_seq/v3shapes_C1_*: 512x768 B1 after 512x512 B1 -> te first call 0.002 s, disk and empty cache).
    te_by_resolution=True restores the conservative (H, W, B) key used before that check."""
    te = ("te", shape.H, shape.W, shape.B) if te_by_resolution else ("te", shape.B)
    return {"te": te, "unet": ("unet", shape.H, shape.W, shape.unet_batch), "vae": ("vae", shape.H, shape.W, shape.B)}


@dataclass(frozen=True)
class Action:
    kind: str
    bucket: Optional[Shape] = None  # only for run_bucket
    evict: Tuple[Shape, ...] = ()   # S3: graphs released (in order) BEFORE the main action of the same request; each must be in state.graphs

    def __post_init__(self):
        assert self.kind in ALL_KINDS, self.kind
        assert (self.bucket is not None) == (self.kind == RUN_BUCKET)
        assert isinstance(self.evict, tuple)


@dataclass
class ProcState:
    """What one serving process holds. `graphs`: shape -> graph pool MB; `graph_kind`: shape -> 'G1' | 'C1+G1'."""
    mem_budget_mb: Optional[float] = None       # graph-layer budget (None = unlimited; all S2 policies)
    eager_seen: Set[Shape] = field(default_factory=set)
    compiled_shapes: Set[Shape] = field(default_factory=set)
    compiled_keys: Set[tuple] = field(default_factory=set)
    graphs: Dict[Shape, float] = field(default_factory=dict)
    graph_kind: Dict[Shape, str] = field(default_factory=dict)
    n_served: int = 0
    shapes_seen: Set[Shape] = field(default_factory=set)  # any shape this process has handled (tier selection)
    # ---- S3 additions (all defaulted) ----
    per_graph_overhead_mb: float = 22.33        # admission: sum(pool + overhead x n_graphs) <= budget (only if overhead_aware)
    overhead_aware: bool = False                # False = S2 accounting (pools only; S2 tests / validation); S3 states pass True
    last_used: Dict[Shape, int] = field(default_factory=dict)     # request index (n_served after the request) -> LRU
    hits: Dict[Shape, int] = field(default_factory=dict)          # steady requests served from a graph of / for this shape
    seen_count: Dict[Shape, int] = field(default_factory=dict)    # arrivals of this shape
    saved_s: Dict[Shape, float] = field(default_factory=dict)     # policy-written: time saved had the next tier been in place
    build_count: Dict[Shape, int] = field(default_factory=dict)   # graph (re)builds of this shape
    evicted_total: int = 0
    denied_total: int = 0
    warm_done: bool = False
    recompile_limit: Optional[int] = None       # B3': max distinct keys per compiled component (None = unlimited / 64 for B1)

    @property
    def graph_mem_mb(self) -> float:
        return float(sum(self.graphs.values()))

    @property
    def first_shape_pending(self) -> bool:
        """True until the process has handled its first (new) shape: that one is charged at the process cold tier."""
        return not self.shapes_seen

    def can_admit(self, mb: float, free: Tuple[Shape, ...] = ()) -> bool:
        """Would a new graph of `mb` fit after the graphs of `free` were released? With overhead_aware every live graph and
        the new one also cost `per_graph_overhead_mb` (S3 accounting); without it this is exactly the S2 rule."""
        if self.mem_budget_mb is None:
            return True
        ov = self.per_graph_overhead_mb if self.overhead_aware else 0.0
        live = [s for s in self.graphs if s not in free]
        used = sum(self.graphs[s] for s in live) + ov * len(live)
        return used + mb + ov <= self.mem_budget_mb + 1e-9

    @property
    def graph_alloc_mb(self) -> float:
        """Graph layer incl. per-graph overhead (S3 accounting; equals graph_mem_mb unless overhead_aware)."""
        return self.graph_mem_mb + (self.per_graph_overhead_mb * len(self.graphs) if self.overhead_aware else 0.0)

    def can_capture(self, mb: float) -> bool:
        return self.can_admit(mb)

    def register(self, shape: Shape, event: str, graph_mb: float = 0.0, te_by_resolution: bool = False,
                 evicted: Tuple[Shape, ...] = (), served_bucket: Optional[Shape] = None):
        """Update the state after a request was served with `event` (see module docstring).
        S3: `evicted` graphs are released first (graphs/graph_kind), then the original logic runs; `served_bucket` = the
        bucket replayed by a run_bucket request (LRU/hit bookkeeping goes to it)."""
        for v in evicted:
            del self.graphs[v]
            self.graph_kind.pop(v, None)
        self.evicted_total += len(evicted)
        self.n_served += 1
        self.seen_count[shape] = self.seen_count.get(shape, 0) + 1
        self.last_used[shape] = self.n_served
        if served_bucket is not None:
            self.last_used[served_bucket] = self.n_served
        if event == "steady":
            tgt = served_bucket if served_bucket is not None else shape
            if tgt in self.graphs:
                self.hits[tgt] = self.hits.get(tgt, 0) + 1
        if event in ("capture", "compile_capture"):
            self.build_count[shape] = self.build_count.get(shape, 0) + 1
        if event == "capture_denied":
            self.denied_total += 1
        self.shapes_seen.add(shape)
        if event in ("eager_first", "capture_denied"):
            self.eager_seen.add(shape)
        elif event == "capture":
            self.graphs[shape] = graph_mb
            self.graph_kind[shape] = "G1"
            self.eager_seen.add(shape)  # the G1 warm-up runs the shape eagerly
        elif event == EAGER_FALLBACK_LIMIT:
            self.eager_seen.add(shape)
        elif event in ("compile", "compile_capture"):
            self.compiled_shapes.add(shape)
            self.compiled_keys.update(component_keys(shape, te_by_resolution).values())
            if event == "compile_capture":
                self.graphs[shape] = graph_mb
                self.graph_kind[shape] = "C1+G1"


def plan_event(action: Action, state: ProcState, shape: Shape, graph_mb: float = 0.0) -> str:
    """Event type the request will trigger, given the state BEFORE serving it."""
    k = action.kind
    if k == EAGER:
        return "steady" if shape in state.eager_seen else "eager_first"
    if k == CAPTURE:
        if shape in state.graphs:
            return "steady"
        return "capture" if state.can_capture(graph_mb) else "capture_denied"
    if k == COMPILE:
        return "steady" if shape in state.compiled_shapes else "compile"
    if k == COMPILE_CAPTURE:
        if shape in state.graphs and state.graph_kind.get(shape) == "C1+G1":
            return "steady"
        return "compile_capture" if state.can_capture(graph_mb) else "capture_denied"
    if k == RUN_BUCKET:
        if action.bucket not in state.graphs:
            raise KeyError(f"run_bucket: no graph for {action.bucket}")
        return "steady"
    raise ValueError(k)


# ----------------------------------------------------------------------------------------------------------------------
# S3: plan_step (shared by simulator and runtime; interface sec. 2 / 4)
def limit_would_be_exceeded(state: ProcState, shape: Shape, te_by_resolution: bool = False) -> bool:
    """B3': True if compiling `shape` would push any component's key count above state.recompile_limit."""
    lim = state.recompile_limit
    if lim is None:
        return False
    for comp, key in component_keys(shape, te_by_resolution).items():
        n = sum(1 for k in state.compiled_keys if k[0] == comp)
        if key not in state.compiled_keys and n + 1 > lim:
            return True
    return False


def plan_step(action: Action, state: ProcState, shape: Shape, graph_mb: float = 0.0):
    """-> (event, evicted_shapes). Event semantics = plan_event for the five S2 actions, plus:
      * `action.evict` shapes are released first, but only if the request's own build then fits ("judge first, act later":
        otherwise `capture_denied` and NOTHING is evicted); every evict shape must be in state.graphs (else KeyError);
      * compile / compile_capture of a new shape with state.recompile_limit set and a component over the limit
        -> `eager_fallback_limit` (served eagerly);
      * compile_dynamic / compile_ro: `steady` if the shape was compiled by this process, else `compile`.
    The graph admission uses ProcState.can_admit (overhead-aware only when state.overhead_aware)."""
    ev = tuple(action.evict)
    for v in ev:
        if v not in state.graphs:
            raise KeyError(f"evict: no graph for {v}")
    k = action.kind
    if k in (COMPILE, COMPILE_CAPTURE) and shape not in state.compiled_shapes and limit_would_be_exceeded(state, shape):
        return EAGER_FALLBACK_LIMIT, ()
    if k in (COMPILE_DYNAMIC, COMPILE_RO):
        return ("steady" if shape in state.compiled_shapes else "compile"), ev
    if k == RUN_BUCKET:
        return plan_event(action, state, shape, graph_mb), ev
    if k == CAPTURE:
        if shape in state.graphs:
            return "steady", ev
        return ("capture", ev) if state.can_admit(graph_mb, ev) else ("capture_denied", ())
    if k == COMPILE_CAPTURE:
        if shape in state.graphs and state.graph_kind.get(shape) == "C1+G1":
            return "steady", ev
        free = ev + ((shape,) if shape in state.graphs and shape not in ev else ())  # upgrade releases the shape's own G1 first
        return ("compile_capture", ev) if state.can_admit(graph_mb, free) else ("capture_denied", ())
    return plan_event(action, state, shape, graph_mb), ev
