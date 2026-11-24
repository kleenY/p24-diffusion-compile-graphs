"""S4-P P5c: tiered runtime whose "compiled" tier is the weight-free AOTInductor package of P5b (TE / UNet / VAE; the weights are inputs and are the ONE live fp16 copy
of the LCMPipe).  New module; imports (does not modify) p24.runners_tiered, scripts.aot_common_S4P_P5 / _P5b.

AOTRunnerP5c(W.AOTRunnerWFI)  same packages / same op order as the P5b arm; the ONLY change: with CFG the text-encoder call on the cond half takes
                              `ids[n:].clone()` instead of `ids[n:].contiguous()` (a contiguous slice at byte offset B*616 is returned unchanged; for odd B the offset is not
                              16-byte aligned and AOTInductor copies the input to an aligned tensor on every call with a warning, P5b report sec 6).  Values are identical.
AOTTieredRuntime(TieredRuntime)  compile=False (no torch.compile objects, `uses_compile` False); prepare_compiled(key, inp) = load the 3 packages of the shape + one full first
                              generation (AOTRunnerP5c.prepare), run_compiled = AOT run.  FALLBACK to eager (no AOT work at all, run_compiled -> run_eager) when the package of the
                              shape is missing (any of the 3 files) or the shape is in the equivalence quick-check failure list; the prep_json of the shape's first request carries
                              aot_fallback=1 and aot_fallback_reason.  A package that exists but fails to load / run is NOT a fallback: the exception propagates (the replay records it
                              as a failed item, as for any compile failure).  Graph layer (prepare_graph / run_graph / evict) is inherited unchanged (B1 never uses it).
"""
from __future__ import annotations

import pathlib
from typing import Dict, Iterable, Optional

import torch

from p24.pipeline import LCMPipe, RequestInputs, ShapeKey
from p24.runners_tiered import TieredRuntime
from scripts import aot_common_S4P_P5 as C
from scripts import aot_common_S4P_P5b as W

_first = C._first


class AOTRunnerP5c(W.AOTRunnerWFI):
    """P5b weight-free AOT runner + aligned (cloned) cond half of the CFG text-encoder input."""
    name = "AOTW"

    @staticmethod
    def _emb(m, inp: RequestInputs, key: ShapeKey):
        if key.cfg:
            n = inp.ids.shape[0] // 2
            return torch.cat([_first(m["te"](inp.ids[:n].contiguous())), _first(m["te"](inp.ids[n:].clone()))])
        return _first(m["te"](inp.ids))

    def _call(self, inp: RequestInputs, key: ShapeKey):
        p, m = self.p, self.mods[key]
        emb = self._emb(m, inp, key)
        lat = inp.latents
        for i in range(p.steps):
            x = torch.cat([lat] * 2) if key.cfg else lat
            eps = _first(m["unet"](x, p.t_bufs[i], emb))
            lat = p.step(i, eps, lat, key.cfg, inp.noises[i] if i < p.steps - 1 else None)
        img = _first(m["vae"](lat / p.vae.config.scaling_factor))
        return (img / 2 + 0.5).clamp(0, 1)

    @torch.inference_mode()
    def run(self, inp, key, marks=None):
        if marks is None:
            return self._call(inp, key)
        p, m = self.p, self.mods[key]
        marks("start")
        emb = self._emb(m, inp, key)
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


def missing_components(pkg_root, key: ShapeKey):
    """Components (te/unet/vae) whose package file is absent for this shape."""
    return [c for c in W.COMPONENTS if not W.wfi_pkg_path(pkg_root, key, c).exists()]


def read_fallback_csv(path) -> Dict[str, str]:
    """aot_fallback_shapes.csv (columns key,reason,...; key = HxWxBxcfg, e.g. 512x768x3x1) -> {key: reason}. A header-only file = empty list."""
    import csv
    out: Dict[str, str] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["key"].strip()] = r.get("reason", "") or "listed"
    return out


def shapekey_of(tag: str) -> ShapeKey:
    h, w, b, c = (int(v) for v in tag.split("x"))
    return ShapeKey(h, w, b, bool(c))


def tag_of(key: ShapeKey) -> str:
    return f"{key.H}x{key.W}x{key.B}x{int(key.cfg)}"


class AOTTieredRuntime(TieredRuntime):
    def __init__(self, model_dir: Optional[str] = None, *, pipe: Optional[LCMPipe] = None, pkg_root,
                 fallback: Optional[Dict[str, str]] = None, aot_runner=None, recompile_limit: int = 64):
        super().__init__(model_dir, pipe=pipe, compile=False, dynamic=False, recompile_limit=recompile_limit)
        self.arm = "TIERED-AOT"
        self.pkg_root = pathlib.Path(pkg_root)
        self.fallback_list: Dict[str, str] = dict(fallback or {})          # equivalence quick-check failure list: {HxWxBxc: reason}
        self._aot = aot_runner if aot_runner is not None else AOTRunnerP5c(self.p, self.pkg_root)
        self.aot_shapes: Dict[ShapeKey, dict] = {}       # shapes served through their AOT package (value: prepare dict)
        self.fallback_shapes: Dict[ShapeKey, str] = {}   # shapes served eagerly instead (value: reason)
        self.n_aot_prepared = 0
        self.n_aot_requests = 0
        self.n_fallback_requests = 0

    # ------------------------------------------------------------------ AOT tier
    def aot_fallback_reason(self, key: ShapeKey) -> Optional[str]:
        t = tag_of(key)
        if t in self.fallback_list:
            return "equiv_fail_list:" + str(self.fallback_list[t])[:80]
        miss = missing_components(self.pkg_root, key)
        if miss:
            return "missing_pkg:" + ",".join(miss)
        return None

    def prepare_compiled(self, key: ShapeKey, inp: RequestInputs) -> dict:
        """First 'compiled' call of this shape: load the 3 packages + one full generation (included in the replay's prepare time), or fall back to eager."""
        reason = self.aot_fallback_reason(key)
        if reason is not None:
            self.fallback_shapes[key] = reason
            return {"aot_fallback": 1, "aot_fallback_reason": reason}
        info = self._aot.prepare(key, inp) or {}
        self.aot_shapes[key] = dict(info)
        self.n_aot_prepared += 1
        return {"aot_fallback": 0, **info}

    def run_compiled(self, inp: RequestInputs, key: ShapeKey):
        if key in self.fallback_shapes:
            self.n_fallback_requests += 1
            return self.run_eager(inp, key)
        self.n_aot_requests += 1
        return self._aot.run(inp, key)

    # ------------------------------------------------------------------ reporting
    def aot_summary(self) -> dict:
        return {"n_aot_shapes": len(self.aot_shapes), "n_fallback_shapes": len(self.fallback_shapes),
                "fallback_shapes": {tag_of(k): v for k, v in self.fallback_shapes.items()},
                "aot_shapes": sorted(tag_of(k) for k in self.aot_shapes),
                "n_aot_requests": self.n_aot_requests, "n_fallback_requests": self.n_fallback_requests,
                "n_aot_prepared": self.n_aot_prepared,
                "load_log": getattr(self._aot, "load_log", [])}
