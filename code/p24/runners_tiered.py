"""Tiered runtime (S3 step 3.1): ONE process holds every execution tier at once.

  E0      the un-compiled modules (eager path; also the fallback of every policy)
  G1      manual per-component CUDA graphs, one private pool per shape key (cheap to build / release)
  C1      torch.compile(mode="default", fullgraph=True, dynamic=False) of the three components (Dynamo cache keyed per component:
          text encoder by B, UNet by (H, W, UNet batch), VAE by (H, W, B); see p24.policy.component_keys)
  C1+G1   manual per-component graphs captured over the SAME compiled components (compile layer + graph layer)
  C5      (B2 only) the same compiled components with dynamic=True

The three compiled callables are created ONCE and shared by C1 and C1+G1, so a shape compiled for one tier is a Dynamo cache hit for the
other (this is what the simulator assumes: p24.policy.ProcState.compiled_keys is shared by `compile` and `compile_capture`).
S2 runners (p24/runners.py, frozen) are reused unchanged: their classes are instantiated with the shared components injected.

Graph layer rules (failure catalogue #14, #15, #19):
  * a graph set owns a private pool; `evict(key)` = g.reset() for every graph + drop static buffers + torch.cuda.empty_cache();
  * the compile layer is never evicted (the Dynamo cache has no per-shape release, #15);
  * a G1 -> C1+G1 upgrade of a shape first releases its old graph set (before the new compile / capture);
  * capture needs its whole pool at once, so the caller evicts BEFORE capturing (serve_trace.serve_s3 does: evict -> main action).
B3' (default recompile limit): constructed with `recompile_limit=8` (C1 keeps fullgraph=True). Hitting the limit raises
torch._dynamo.exc.FailOnRecompileLimitHit (no automatic fallback, precheck 11-17); the caller catches `rt.LimitExc` and serves the
WHOLE request on the un-compiled modules (event `eager_fallback_limit`).
Accounting convention (interface v1 sec 2/4): admission uses the cost-table prediction only; everything measured here (reserved memory
before/after an eviction, measured pool of a capture) is only reported.
"""
from __future__ import annotations

import gc
import time
from typing import Dict, Optional

import torch

from . import compat  # noqa: F401  (P24_PINNED_FALLBACK=1 -> failure catalogue #18 workaround)
from . import diag
from .pipeline import LCMPipe, RequestInputs, ShapeKey
from .runners import CompiledRunner, GraphRunner, Runner

MiB = 2 ** 20


class CaptureFailed(RuntimeError):
    """Graph capture ran out of memory (per-process cap or device). Policy rule: capture failure -> serve eagerly."""


class _SharedCompiled(CompiledRunner):
    """CompiledRunner (S2) whose compiled components are injected, not created (shared with the C1+G1 graph runner)."""

    def __init__(self, pipe: LCMPipe, comps, dynamic: bool = False):
        Runner.__init__(self, pipe)
        self.mode, self.dynamic, self.regional = "default", dynamic, False
        self.name = "compile[default,dyn]" if dynamic else "compile[default]"
        self._te, self._unet, self._vae_decode = comps
        self.cudagraphs = False


def _graph_runner(pipe: LCMPipe, comps=None) -> GraphRunner:
    """G1 (comps=None: un-compiled modules) or C1+G1 (comps = shared compiled components)."""
    r = GraphRunner(pipe, whole=False, compiled=False)  # compiled=False: does not create its own torch.compile objects
    if comps is not None:
        r.compiled = r.uses_compile = True
        r.name = "C1+G1"
        r._te, r._unet, r._vdec = comps
    return r


class TieredRuntime:
    """See module docstring. All timing is done by the caller (serve_trace.serve_s3); methods only synchronise where memory is read."""

    LimitExc = None  # set in __init__ (torch._dynamo.exc.FailOnRecompileLimitHit)

    def __init__(self, model_dir: Optional[str] = None, *, pipe: Optional[LCMPipe] = None, compile: bool = True,
                 dynamic: bool = False, recompile_limit: int = 64):
        from torch._dynamo.exc import FailOnRecompileLimitHit
        self.LimitExc = FailOnRecompileLimitHit
        self.p = pipe if pipe is not None else LCMPipe(model_dir)
        # Timestep buffers as INFERENCE tensors (failure catalogue #17 again): LCMPipe builds t_bufs outside inference_mode (normal tensors),
        # while the graph runner captures with an inference-mode clone (GraphRunner.prepare is @inference_mode). Dynamo guards on the tensor
        # kind, so C1 followed by C1+G1 of the SAME shape recompiled the UNet once (measured 11-17: +1 graph, 3.8 s FX-cache hit, 256^2 B1).
        # With inference buffers both tiers share one Dynamo entry per component key (0 recompiles; FX cache still hits). Values unchanged.
        with torch.inference_mode():
            self.p.t_bufs = [t.clone() for t in self.p.t_bufs]
        self.recompile_limit = int(recompile_limit)
        self.compile_enabled = bool(compile)
        self.dynamic = bool(dynamic)
        self.uses_compile = self.compile_enabled
        self.arm = "TIERED"
        self._g1 = _graph_runner(self.p)
        self._cg: Optional[GraphRunner] = None
        self._c1: Optional[_SharedCompiled] = None
        self._kind: Dict[ShapeKey, str] = {}  # key -> 'G1' | 'C1+G1' (graph sets alive in this process)
        self.n_evictions = 0
        self.evict_log = []  # one dict per evict(): freed MB measured by memory_reserved, time
        if self.compile_enabled:
            diag.set_recompile_limit(self.recompile_limit)  # C1/C1+G1: 64; B3': 8
            kw = dict(fullgraph=True, dynamic=self.dynamic)
            comps = (torch.compile(self.p.te, **kw), torch.compile(self.p.unet, **kw),
                     torch.compile(self.p.vae.decode, **kw))
            self._c1 = _SharedCompiled(self.p, comps, dynamic=self.dynamic)
            self._cg = _graph_runner(self.p, comps)

    # ------------------------------------------------------------------ queries
    def graph_kind(self, key: ShapeKey) -> Optional[str]:
        return self._kind.get(key)

    def has_graph(self, key: ShapeKey) -> bool:
        return key in self._kind

    def graph_keys(self):
        return list(self._kind)

    @staticmethod
    def mem_mb():
        return torch.cuda.memory_reserved() / MiB, torch.cuda.memory_allocated() / MiB

    # ------------------------------------------------------------------ execution
    @torch.inference_mode()
    def run_eager(self, inp: RequestInputs, key: ShapeKey):
        return self.p.run(inp, key.cfg)

    def prepare_compiled(self, key: ShapeKey, inp: RequestInputs) -> dict:
        """First compiled call of this shape (C1 / C5). May raise self.LimitExc when the recompile limit is hit."""
        if self._c1 is None:
            raise RuntimeError("runtime built with compile=False")
        return self._c1.prepare(key, inp) or {}

    def run_compiled(self, inp: RequestInputs, key: ShapeKey):
        if self._c1 is None:
            raise RuntimeError("runtime built with compile=False")
        return self._c1.run(inp, key)

    def prepare_graph(self, kind: str, key: ShapeKey, inp: RequestInputs) -> dict:
        """Build the graph set `kind` ('G1' | 'C1+G1') for `key`. An existing graph set of this key (any kind) is released first
        (G1 -> C1+G1 upgrade). OOM during capture -> CaptureFailed (partial capture cleaned up)."""
        if kind not in ("G1", "C1+G1"):
            raise ValueError(kind)
        if kind == "C1+G1" and self._cg is None:
            raise RuntimeError("runtime built with compile=False")
        extra = {}
        if key in self._kind:
            info = self.evict(key)
            extra = {"upgrade_released_kind": info["kind"], "upgrade_released_mb": info["freed_mb"],
                     "upgrade_release_s": info["evict_s"]}
        runner = self._g1 if kind == "G1" else self._cg
        failed = None
        out = {}
        try:
            out = runner.prepare(key, inp)
        except torch.OutOfMemoryError as e:  # leave the except block first: the traceback pins the half-built graph set
            failed = str(e)[:300]
        if failed is not None:
            runner.sets.pop(key, None)
            gc.collect()
            torch.cuda.empty_cache()
            raise CaptureFailed(failed)
        self._kind[key] = kind
        return {**out, **extra}

    def run_graph(self, inp: RequestInputs, key: ShapeKey):
        kind = self._kind.get(key)
        if kind is None:
            raise KeyError(f"no graph for {key.tag()}")
        return (self._g1 if kind == "G1" else self._cg).run(inp, key)

    def run_bucket(self, inp_padded: RequestInputs, bucket: ShapeKey, real_B: int):
        """Serve a request of `real_B` samples from the graph of `bucket` (same H, W, cfg; larger B). inp_padded is built for the
        BUCKET shape (filler prompts / seeds, see scripts/run_equivalence.py); the first real_B samples are the answer."""
        return self.run_graph(inp_padded, bucket)[:real_B]

    # ------------------------------------------------------------------ eviction
    def evict(self, key: ShapeKey) -> dict:
        """Release the graph set of `key` (reset every graph, drop buffers, empty_cache). Compile layer untouched (#15)."""
        kind = self._kind.pop(key, None)
        if kind is None:
            raise KeyError(f"evict: no graph for {key.tag()}")
        torch.cuda.synchronize()
        r0 = torch.cuda.memory_reserved()
        t0 = time.perf_counter()
        # flush the ordinary allocator cache first (eager workspaces of earlier requests; it would be flushed by the final empty_cache
        # anyway) so that the next difference is the graph's footprint alone
        torch.cuda.empty_cache()
        r1 = torch.cuda.memory_reserved()
        # g.reset() x3, bufs.clear(), empty_cache(). NO gc.collect(): measured 11-17 it costs 0.26-0.6 s in a process holding the Dynamo/Inductor
        # heap (release itself 8 ms) and is not needed (graph objects are freed by reference counting; the freed MB check in the tests passes).
        (self._g1 if kind == "G1" else self._cg).release(key)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        info = {"shape": key.tag(), "kind": kind, "freed_mb": (r1 - torch.cuda.memory_reserved()) / MiB,
                "cache_flushed_mb": (r0 - r1) / MiB, "evict_s": dt}
        self.n_evictions += 1
        self.evict_log.append(info)
        return info

    def release_all(self) -> int:
        n = 0
        for k in list(self._kind):
            self.evict(k)
            n += 1
        return n
