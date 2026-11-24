"""Compile / cache diagnostics shared by bench, bench_seq, check_cache_keys (S2 step 2.0, protocol sections 1, 5, 6).

Counter sources (torch 2.9.1, verified in torch/_dynamo/utils.py, output_graph.py, convert_frame.py, codecache.py):
  torch._dynamo.utils.counters["stats"]["unique_graphs"]      one per compiled FX graph (first compile AND recompile)
  torch._dynamo.convert_frame.FRAME_COMPILE_COUNTER[frame_id] number of compiles of that frame -> recompiles = sum(n-1)
  torch._dynamo.utils.counters["inductor"]["fxgraph_cache_hit|miss|bypass"]
  torch._dynamo.utils.counters["aot_autograd"]["autograd_cache_hit|miss|bypass"]
  torch._dynamo.config.recompile_limit (per-code-object, default 8; protocol sets 64), accumulated_recompile_limit
Recompile-limit hit (torch 2.9.1, convert_frame.py ~1349-1379; corrected 2026-11-15 after review S6): with fullgraph=True
(all compiled arms) hitting the limit logs the warning "torch._dynamo hit config.recompile_limit (N)" and raises
FailOnRecompileLimitHit("recompile_limit reached with fullgraph=True ..."); nothing is written to
counters["unimplemented"]["recompile_limit reached"] on that path (no such key is written anywhere in torch). The process
then ends with status=fail and recompile_limit_hit=1, which p24.bench derives from (a) the exception text in `err`
("FailOnRecompileLimitHit"), (b) the LimitWatch log handler, (c) counters["unimplemented"] keys containing "cache limit
exceeded" (the non-fullgraph unimplemented_v2 path, not used by our arms). tests/test_recompile_limit.py exercises (a)+(b)
end to end with recompile_limit=1. Never use the recompile count: the failed attempt is counted in FRAME_COMPILE_COUNTER too.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import torch

RECOMPILE_LIMIT = 64  # protocol sec 1 / plan 9.5


def set_recompile_limit(n: int = RECOMPILE_LIMIT) -> int:
    import torch._dynamo.config as dc
    dc.recompile_limit = n
    return dc.recompile_limit


def dyn_snapshot() -> Dict[str, int]:
    """Cheap monotone counters; subtract two snapshots for a delta."""
    from torch._dynamo.utils import counters
    try:
        from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER
        compiles = int(sum(FRAME_COMPILE_COUNTER.values()))
        recompiles = int(sum(max(int(v) - 1, 0) for v in FRAME_COMPILE_COUNTER.values()))
    except Exception:  # pragma: no cover
        compiles = recompiles = -1
    ind, aot, st = counters["inductor"], counters["aot_autograd"], counters["stats"]
    return {
        "unique_graphs": int(st["unique_graphs"]), "frame_compiles": compiles, "recompiles": recompiles,
        "fx_hit": int(ind["fxgraph_cache_hit"]), "fx_miss": int(ind["fxgraph_cache_miss"]),
        "fx_bypass": int(ind["fxgraph_cache_bypass"]),
        "aot_hit": int(aot["autograd_cache_hit"]), "aot_miss": int(aot["autograd_cache_miss"]),
        "aot_bypass": int(aot["autograd_cache_bypass"]),
        # secondary limit-hit marker: only the non-fullgraph path (unimplemented_v2 "Dynamo cache limit exceeded") writes
        # counters["unimplemented"]; the fullgraph path used by our arms raises FailOnRecompileLimitHit instead (see module
        # docstring), so the primary signals are the exception and the log handler. NOT derived from the recompile count,
        # because the failed attempt after the limit is also counted in FRAME_COMPILE_COUNTER (+1).
        "limit_reached": int(sum(v for k, v in counters["unimplemented"].items()
                                 if "cache limit exceeded" in str(k) or "recompile_limit reached" in str(k))),
    }


def dyn_delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: b[k] - a[k] for k in b}


class LimitWatch(logging.Handler):
    """Catches torch._dynamo's 'hit config.recompile_limit' warning (also raised as an exception under fullgraph)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.hit = False
        self.msgs = []

    def emit(self, record):
        m = record.getMessage()
        if "hit config." in m and "limit" in m:
            self.hit = True
            self.msgs.append(m[:300])


def install_limit_watch() -> LimitWatch:
    h = LimitWatch()
    logging.getLogger("torch._dynamo").addHandler(h)
    return h


class FirstCall:
    """Synchronised timing of the FIRST invocation of a callable per `store` (used only inside prepare(); the
    steady path never goes through this wrapper). store[name] = seconds, store[name+'_diag'] = counter delta."""

    def __init__(self, fn, name: str, store: dict, enabled: bool = True, with_diag: bool = False):
        self.fn, self.name, self.store = fn, name, store
        self.enabled, self.with_diag = enabled, with_diag

    def __call__(self, *a, **kw):
        if not self.enabled or self.name in self.store:
            return self.fn(*a, **kw)
        torch.cuda.synchronize()
        d0 = dyn_snapshot() if self.with_diag else None
        t0 = time.perf_counter()
        out = self.fn(*a, **kw)
        torch.cuda.synchronize()
        self.store[self.name] = time.perf_counter() - t0
        if d0 is not None:
            self.store[self.name + "_diag"] = dyn_delta(d0, dyn_snapshot())
        return out


def flatten_first_call(store: dict, whole: Optional[float] = None) -> Dict[str, float]:
    """store -> flat prepare-dict fields first_call_{te,unet,vae}_s (+ new-graph counts)."""
    nan = float("nan")
    out = {}
    for n in ("te", "unet", "vae"):
        out[f"first_call_{n}_s"] = store.get(n, nan)
        d = store.get(n + "_diag")
        out[f"first_call_{n}_new_graphs"] = d["unique_graphs"] if d else 0
    if whole is not None:
        out["first_call_whole_s"] = whole
    return out
