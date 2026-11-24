"""S4-P P5 probe (AOTInductor packages): shared pieces. New module; no existing file is touched.

Card: goals/S4P_P5_AOT_artifact_probe_card_v1.md (0444).  Exploration level, numbers are go/no-go only.

* Wrapper modules that expose the three components with the exact call structure used by p24.runners.CompiledRunner
  (text encoder: ids -> hidden states; UNet: (x, t, emb) -> eps; VAE decode: latents -> image), so that the AOT arm runs the SAME
  LCMPipe math (step maths, CFG split of the text encoder calls, scaling factor) eagerly around three AOT-compiled components.
* AOTRunner: Runner subclass; prepare() = aoti_load_package x 3 + first full generation; run() = same op order as CompiledRunner._call.
* `adapters()` give call-compatible callables so that GraphRunner (G1) can be built on top of the AOT modules (side report).
"""
from __future__ import annotations

import pathlib
import time
from typing import Dict

import torch

from p24 import compat  # noqa: F401  (P24_PINNED_FALLBACK=1)
from p24.pipeline import LCMPipe, RequestInputs, ShapeKey
from p24.runners import Runner

COMPONENTS = ("te", "unet", "vae")


class TEWrap(torch.nn.Module):
    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, ids):
        return self.te(ids, attention_mask=None)[0]


class UNetWrap(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, emb):
        return self.unet(x, t, encoder_hidden_states=emb, return_dict=False)[0]


class VAEWrap(torch.nn.Module):
    """vae.decode(latents/scaling_factor)[0]; the scaling division and the (img/2+0.5).clamp stay outside, as in C1."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, lat):
        return self.vae.decode(lat, return_dict=False)[0]


def te_batch(key: ShapeKey) -> int:
    """Text encoder batch per call: with CFG LCMPipe.encode makes two calls (uncond, cond) of B each."""
    return key.B


def export_args(p: LCMPipe, key: ShapeKey, comp: str):
    """Example inputs for torch.export (static shapes, same device/dtype as the live path)."""
    h, w = key.latent_hw
    dev, dt = p.device, p.dtype
    if comp == "te":
        return (torch.randint(0, 1000, (te_batch(key), p.tok.model_max_length), dtype=torch.int64, device=dev),)
    if comp == "unet":
        ub = key.unet_batch
        return (torch.randn(ub, 4, h, w, dtype=dt, device=dev), p.t_bufs[0].clone(),
                torch.randn(ub, 77, p.te.config.hidden_size, dtype=dt, device=dev))
    if comp == "vae":
        return (torch.randn(key.B, 4, h, w, dtype=dt, device=dev),)
    raise ValueError(comp)


def wrapper(p: LCMPipe, comp: str) -> torch.nn.Module:
    return {"te": lambda: TEWrap(p.te), "unet": lambda: UNetWrap(p.unet), "vae": lambda: VAEWrap(p.vae)}[comp]()


def pkg_path(pkg_root, key: ShapeKey, comp: str) -> pathlib.Path:
    return pathlib.Path(pkg_root) / key.tag() / f"{comp}.pt2"


def _first(o):
    return o[0] if isinstance(o, (list, tuple)) else o


class AOTRunner(Runner):
    """Three AOT components + eager step maths. name 'AOT'."""
    name = "AOT"
    uses_compile = False  # no torch.compile/Dynamo in this process

    def __init__(self, pipe: LCMPipe, pkg_root):
        super().__init__(pipe)
        self.pkg_root = pathlib.Path(pkg_root)
        self.mods: Dict[ShapeKey, dict] = {}
        self.arm = "AOT"

    # ---- call-compatible adapters (same call signatures as the eager/compiled modules) ----
    def adapters(self, key: ShapeKey):
        m = self.mods[key]

        def te(ids, attention_mask=None):
            return (m["te"](ids),)

        def unet(x, t, encoder_hidden_states=None, return_dict=False):
            return (_first(m["unet"](x, t, encoder_hidden_states)),)

        def vdec(lat, return_dict=False):
            return (_first(m["vae"](lat)),)
        return te, unet, vdec

    def has(self, key):
        return key in self.mods

    def _load(self, key: ShapeKey) -> dict:
        from torch._inductor import aoti_load_package
        t = {}
        mods = {}
        for c in COMPONENTS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mods[c] = aoti_load_package(str(pkg_path(self.pkg_root, key, c)))
            torch.cuda.synchronize()
            t[f"load_{c}_s"] = time.perf_counter() - t0
        self.mods[key] = mods
        t["load_s"] = sum(t.values())
        return t

    def _call(self, inp: RequestInputs, key: ShapeKey):
        p = self.p
        m = self.mods[key]
        if key.cfg:
            n = inp.ids.shape[0] // 2
            emb = torch.cat([_first(m["te"](inp.ids[:n].contiguous())), _first(m["te"](inp.ids[n:].contiguous()))])
        else:
            emb = _first(m["te"](inp.ids))
        lat = inp.latents
        for i in range(p.steps):
            x = torch.cat([lat] * 2) if key.cfg else lat
            eps = _first(m["unet"](x, p.t_bufs[i], emb))
            lat = p.step(i, eps, lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        img = _first(m["vae"](lat / p.vae.config.scaling_factor))
        return (img / 2 + 0.5).clamp(0, 1)

    @torch.inference_mode()
    def prepare(self, key, example):
        torch.cuda.synchronize()  # eager weights (if freed) were moved to host by the driver BEFORE timing starts
        t0 = time.perf_counter()
        ld = self._load(key)
        t1 = time.perf_counter()
        self._call(example, key)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {"load_s": t1 - t0, "first_gen_s": t2 - t1, "load_plus_first_gen_s": t2 - t0,
                "compile_s": t2 - t0, **{k: v for k, v in ld.items() if k != "load_s"}}

    @torch.inference_mode()
    def run(self, inp, key, marks=None):
        if marks is not None:  # component marks (te_end/denoise_end/vae_end) around the AOT calls
            p, m = self.p, self.mods[key]
            marks("start")
            if key.cfg:
                n = inp.ids.shape[0] // 2
                emb = torch.cat([_first(m["te"](inp.ids[:n].contiguous())), _first(m["te"](inp.ids[n:].contiguous()))])
            else:
                emb = _first(m["te"](inp.ids))
            marks("te_end")
            lat = inp.latents
            for i in range(p.steps):
                x = torch.cat([lat] * 2) if key.cfg else lat
                eps = _first(m["unet"](x, p.t_bufs[i], emb))
                lat = p.step(i, eps, lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
            marks("denoise_end")
            img = _first(m["vae"](lat / p.vae.config.scaling_factor))
            out = (img / 2 + 0.5).clamp(0, 1)
            marks("vae_end")
            return out
        return self._call(inp, key)

    def release(self, key):
        self.mods.pop(key, None)
        torch.cuda.empty_cache()


def free_eager_weights(pipe: LCMPipe):
    """Move the eager UNet/text-encoder/VAE weights to host so that the AOT process holds one weight copy on the GPU
    (done before any timing; the scaling factor, tokenizer and schedule constants stay)."""
    for m in (pipe.unet, pipe.te, pipe.vae):
        m.to("cpu")
    torch.cuda.empty_cache()
