"""Execution configurations (arms). Every arm runs the same LCMPipe math on the same fused weights.

E0/E1  eager (E1 = channels_last + fused QKV, set on LCMPipe construction)
C*     torch.compile on the three components (text encoder, UNet, VAE); step math stays eager
G1     manual per-component CUDA graphs (text / UNet-single-step / VAE), one private pool per shape key
G2     manual whole-pipeline CUDA graph (4 unrolled steps + VAE), one private pool per shape key
C1+G   = G1 or G2 built on top of torch.compile(mode="default") components (compile layer + graph layer)
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from . import compat  # noqa: F401  (P24_PINNED_FALLBACK=1 -> failure catalogue #18 workaround)
from . import diag
from .pipeline import LCMPipe, RequestInputs, ShapeKey


def _sync():
    torch.cuda.synchronize()


class Runner:
    name = "base"
    uses_compile = False

    def __init__(self, pipe: LCMPipe):
        self.p = pipe

    def prepare(self, key: ShapeKey, example: RequestInputs) -> Dict[str, float]:
        """One-time per-shape work (compile and/or capture). Returns timing breakdown in seconds."""
        return {}

    def run(self, inp: RequestInputs, key: ShapeKey, marks=None) -> torch.Tensor:
        """marks: optional callback marks(name), name in start / te_end / denoise_end / vae_end (caller records a
        CUDA event). marks=None keeps the original timed path (S2 step 2.0, protocol section 3 step 6)."""
        raise NotImplementedError

    def release(self, key: ShapeKey) -> None:
        pass

    def has(self, key: ShapeKey) -> bool:
        return True


class EagerRunner(Runner):
    name = "eager"

    @torch.inference_mode()
    def run(self, inp, key, marks=None):
        if marks is not None:
            return self.p.run(inp, key.cfg, marks)
        return self.p.run(inp, key.cfg)


class CompiledRunner(Runner):
    """torch.compile on components. mode in {default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs}."""
    uses_compile = True

    def __init__(self, pipe: LCMPipe, mode: str = "default", dynamic: Optional[bool] = False,
                 regional: bool = False):
        super().__init__(pipe)
        diag.set_recompile_limit()  # protocol sec 1: recompile_limit = 64 for every compiled arm
        self.mode, self.dynamic, self.regional = mode, dynamic, regional
        self.name = f"compile[{mode}{',dyn' if dynamic else ''}{',regional' if regional else ''}]"
        kw = dict(fullgraph=True, dynamic=dynamic)
        if mode != "default":
            kw["mode"] = mode
        if regional:
            # compile only the repeated transformer/resnet blocks (diffusers >=0.35 helper)
            pipe.unet.compile_repeated_blocks(**kw)
            self._unet = pipe.unet.forward
        else:
            self._unet = torch.compile(pipe.unet, **kw)
        self._te = torch.compile(pipe.te, **kw)
        self._vae_decode = torch.compile(pipe.vae.decode, **kw)
        self.cudagraphs = "reduce-overhead" in mode or mode == "max-autotune"

    def _call(self, inp: RequestInputs, key: ShapeKey):
        p = self.p
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        emb = p.encode(inp.ids, key.cfg, te=self._te)
        lat = inp.latents
        for i in range(p.steps):
            x = torch.cat([lat] * 2) if key.cfg else lat
            eps = self._unet(x, p.t_bufs[i], encoder_hidden_states=emb, return_dict=False)[0]
            lat = p.step(i, eps, lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        img = self._vae_decode(lat / p.vae.config.scaling_factor, return_dict=False)[0]
        out = (img / 2 + 0.5).clamp(0, 1)
        return out.clone() if self.cudagraphs else out

    def _call_gen(self, inp: RequestInputs, key: ShapeKey, te, unet, vde, marks=None):
        """Same math/order as _call with injectable component callables (first-call timers in prepare) and optional
        event marks. Used only by prepare() and marked requests; the unmarked steady path is _call."""
        p = self.p
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        if marks is not None:
            marks("start")
        emb = p.encode(inp.ids, key.cfg, te=te)
        if marks is not None:
            marks("te_end")
        lat = inp.latents
        for i in range(p.steps):
            x = torch.cat([lat] * 2) if key.cfg else lat
            eps = unet(x, p.t_bufs[i], encoder_hidden_states=emb, return_dict=False)[0]
            lat = p.step(i, eps, lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        if marks is not None:
            marks("denoise_end")
        img = vde(lat / p.vae.config.scaling_factor, return_dict=False)[0]
        out = (img / 2 + 0.5).clamp(0, 1)
        if marks is not None:
            marks("vae_end")
        return out.clone() if self.cudagraphs else out

    @torch.inference_mode()
    def prepare(self, key, example):
        _sync()
        d0 = diag.dyn_snapshot()
        t0 = time.perf_counter()
        fc: dict = {}
        self._call_gen(example, key, diag.FirstCall(self._te, "te", fc, with_diag=True),
                       diag.FirstCall(self._unet, "unet", fc, with_diag=True),
                       diag.FirstCall(self._vae_decode, "vae", fc, with_diag=True))
        _sync()
        t1 = time.perf_counter()
        extra = 0.0
        if self.cudagraphs:  # cudagraph trees: warmup run then record run
            for _ in range(2):
                self._call(example, key)
            _sync()
            extra = time.perf_counter() - t1
        dd = diag.dyn_delta(d0, diag.dyn_snapshot())
        return {"compile_s": t1 - t0, "capture_s": extra, **diag.flatten_first_call(fc),
                "prep_new_graphs": dd["unique_graphs"], "prep_new_recompiles": dd["recompiles"]}

    @torch.inference_mode()
    def run(self, inp, key, marks=None):
        if marks is not None:
            return self._call_gen(inp, key, self._te, self._unet, self._vae_decode, marks)
        return self._call(inp, key)


class _GraphSet:
    """Static buffers + CUDA graphs for one shape key, sharing one private memory pool."""

    def __init__(self):
        self.graphs = []
        self.bufs: Dict[str, torch.Tensor] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.mem_bytes = 0


class GraphRunner(Runner):
    """Manual torch.cuda.CUDAGraph capture. whole=False -> G1 (per component), whole=True -> G2.
    If `compiled` is True the captured callables are torch.compile(mode='default') components (C1+G)."""

    def __init__(self, pipe: LCMPipe, whole: bool = False, compiled: bool = False, warmup: int = 3):
        super().__init__(pipe)
        self.whole, self.compiled, self.warmup = whole, compiled, warmup
        self.uses_compile = compiled
        self.name = ("C1+" if compiled else "") + ("G2" if whole else "G1")
        if compiled:
            diag.set_recompile_limit()  # protocol sec 1
            kw = dict(fullgraph=True, dynamic=False)
            self._te = torch.compile(pipe.te, **kw)
            self._unet = torch.compile(pipe.unet, **kw)
            self._vdec = torch.compile(pipe.vae.decode, **kw)
        else:
            self._te, self._unet, self._vdec = pipe.te, pipe.unet, pipe.vae.decode
        self.sets: Dict[ShapeKey, _GraphSet] = {}

    # pure functions over static buffers
    def _encode(self, ids, cfg):
        return self.p.encode(ids, cfg, te=self._te)

    def _eps(self, x, t, emb):
        return self._unet(x, t, encoder_hidden_states=emb, return_dict=False)[0]

    def _decode(self, lat):
        img = self._vdec(lat / self.p.vae.config.scaling_factor, return_dict=False)[0]
        return (img / 2 + 0.5).clamp(0, 1)

    def _whole(self, b, key):
        p = self.p
        emb = self._encode(b["ids"], key.cfg)
        lat = b["lat"]
        for i in range(p.steps):
            x = torch.cat([lat] * 2) if key.cfg else lat
            eps = self._eps(x, p.t_bufs[i], emb)
            lat = p.step(i, eps, lat, key.cfg, b["noise"][i] if i < p.steps - 1 else None)
        return self._decode(lat)

    def has(self, key):
        return key in self.sets

    @torch.inference_mode()
    def prepare(self, key, example):
        p = self.p
        gs = _GraphSet()
        b = gs.bufs
        b["ids"] = example.ids.clone()
        b["lat"] = example.latents.clone()
        b["noise"] = example.noises.clone()
        h, w = key.latent_hw
        b["x"] = torch.empty((key.unet_batch, 4, h, w), device=p.device, dtype=p.dtype)
        b["t"] = p.t_bufs[0].clone()
        mem0 = torch.cuda.memory_reserved()
        _sync()
        d0 = diag.dyn_snapshot() if self.compiled else None
        t0 = time.perf_counter()
        fc: dict = {}
        whole_first = None
        # ---- warmup on a side stream (also triggers torch.compile when compiled=True) ----
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for it in range(self.warmup):
                if self.whole:
                    if it == 0:  # whole-pipeline graph: no component split, only the total first call
                        _sync()
                        tw = time.perf_counter()
                        self._whole(b, key)
                        _sync()
                        whole_first = time.perf_counter() - tw
                    else:
                        self._whole(b, key)
                else:
                    wd = self.compiled
                    enc = diag.FirstCall(self._encode, "te", fc, enabled=(it == 0), with_diag=wd)
                    eps_f = diag.FirstCall(self._eps, "unet", fc, enabled=(it == 0), with_diag=wd)
                    dec = diag.FirstCall(self._decode, "vae", fc, enabled=(it == 0), with_diag=wd)
                    emb = enc(b["ids"], key.cfg)
                    b["x"].copy_(torch.cat([b["lat"]] * 2) if key.cfg else b["lat"])
                    eps = eps_f(b["x"], b["t"], emb)
                    dec(b["lat"])
        torch.cuda.current_stream().wait_stream(s)
        _sync()
        t1 = time.perf_counter()
        torch.cuda.empty_cache()  # drop warm-up cache blocks so the next delta is the capture pool itself
        mem_w = torch.cuda.memory_reserved()
        # ---- capture ----
        if self.whole:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=gs.pool):
                b["out"] = self._whole(b, key)
            gs.graphs = [g]
        else:
            g_te, g_un, g_vae = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_te, pool=gs.pool):
                b["emb"] = self._encode(b["ids"], key.cfg)
            with torch.cuda.graph(g_un, pool=gs.pool):
                b["eps"] = self._eps(b["x"], b["t"], b["emb"])
            with torch.cuda.graph(g_vae, pool=gs.pool):
                b["out"] = self._decode(b["lat"])
            gs.graphs = [g_te, g_un, g_vae]
        _sync()
        t2 = time.perf_counter()
        gs.mem_bytes = torch.cuda.memory_reserved() - mem_w
        self.sets[key] = gs
        # capture_pool_mb: reserved growth during capture only (private pool + static outputs; added 2026-11-15).
        # graph_mem_mb (kept for continuity with S1 block A/B): reserved growth over the whole prepare().
        # static_io_mb: bytes of all static tensors held in gs.bufs (inputs ids/lat/noise/x/t AND captured outputs
        # emb/eps/out; the outputs live inside the private pool, so static_io_mb is NOT additive to capture_pool_mb).
        # static_in_mb: the input buffers only.
        out_names = ("emb", "eps", "out")
        io_b = sum(v.numel() * v.element_size() for v in b.values())
        in_b = sum(v.numel() * v.element_size() for k_, v in b.items() if k_ not in out_names)
        ret = {"compile_and_warmup_s": t1 - t0, "capture_s": t2 - t1, "capture_pool_mb": gs.mem_bytes / 2**20,
               "graph_mem_mb": (torch.cuda.memory_reserved() - mem0) / 2**20,
               "static_io_mb": io_b / 2**20, "static_in_mb": in_b / 2**20,
               **diag.flatten_first_call(fc, whole=whole_first)}
        if d0 is not None:
            dd = diag.dyn_delta(d0, diag.dyn_snapshot())
            ret.update(prep_new_graphs=dd["unique_graphs"], prep_new_recompiles=dd["recompiles"])
        return ret

    @torch.inference_mode()
    def run(self, inp, key, marks=None):
        if marks is not None and not self.whole:  # whole-pipeline graphs have no component boundaries (NaN)
            return self._run_marked(inp, key, marks)
        p, gs = self.p, self.sets[key]
        b = gs.bufs
        b["ids"].copy_(inp.ids)
        if self.whole:
            b["lat"].copy_(inp.latents)
            b["noise"].copy_(inp.noises)
            gs.graphs[0].replay()
            return b["out"].clone()
        g_te, g_un, g_vae = gs.graphs
        g_te.replay()
        lat = inp.latents
        for i in range(p.steps):
            b["x"].copy_(torch.cat([lat] * 2) if key.cfg else lat)
            b["t"].copy_(p.t_bufs[i])
            g_un.replay()
            lat = p.step(i, b["eps"], lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        b["lat"].copy_(lat)
        g_vae.replay()
        return b["out"].clone()

    def _run_marked(self, inp, key, marks):
        """Same ops/order as run() (G1 only) with CUDA-event marks between components."""
        p, gs = self.p, self.sets[key]
        b = gs.bufs
        marks("start")
        b["ids"].copy_(inp.ids)
        g_te, g_un, g_vae = gs.graphs
        g_te.replay()
        marks("te_end")
        lat = inp.latents
        for i in range(p.steps):
            b["x"].copy_(torch.cat([lat] * 2) if key.cfg else lat)
            b["t"].copy_(p.t_bufs[i])
            g_un.replay()
            lat = p.step(i, b["eps"], lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        marks("denoise_end")
        b["lat"].copy_(lat)
        g_vae.replay()
        out = b["out"].clone()
        marks("vae_end")
        return out

    def release(self, key):
        gs = self.sets.pop(key, None)
        if gs is not None:
            for g in gs.graphs:
                g.reset()
            gs.graphs.clear()
            gs.bufs.clear()
            del gs
            torch.cuda.empty_cache()


def make_runner(arm: str, model_dir: str) -> Runner:
    """Arm registry (names used in CSVs)."""
    E1 = dict(channels_last=True, fuse_qkv=True)
    table = {
        "E0": lambda: EagerRunner(LCMPipe(model_dir)),
        "E1": lambda: EagerRunner(LCMPipe(model_dir, **E1)),
        "G1": lambda: GraphRunner(LCMPipe(model_dir), whole=False),
        "G2": lambda: GraphRunner(LCMPipe(model_dir), whole=True),
        "C1": lambda: CompiledRunner(LCMPipe(model_dir), "default"),
        "C2": lambda: CompiledRunner(LCMPipe(model_dir), "reduce-overhead"),
        "C3": lambda: CompiledRunner(LCMPipe(model_dir), "max-autotune"),
        "C5": lambda: CompiledRunner(LCMPipe(model_dir), "default", dynamic=True),
        "C6": lambda: CompiledRunner(LCMPipe(model_dir), "default", regional=True),
        "C1+G1": lambda: GraphRunner(LCMPipe(model_dir), whole=False, compiled=True),
        "C1+G2": lambda: GraphRunner(LCMPipe(model_dir), whole=True, compiled=True),
    }
    r = table[arm]()
    r.arm = arm
    return r
