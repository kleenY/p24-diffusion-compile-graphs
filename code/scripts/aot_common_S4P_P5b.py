"""S4-P P5b (card goals/S4P_P5b_AOT_memory_gate_probe_card_v1.md, 0444): AOTInductor packages with the WEIGHTS AS INPUTS for all three components
(text encoder, UNet, VAE decoder), so that every shape package is weight-free (a few MB) and all packages of one process share the one live
fp16 weight copy of the LCMPipe.  New module; imports (does not modify) scripts.aot_common_S4P_P5 and p24.*.

* wfi_names / wfi_weights: the ordered weight names/tensors of a component (wrapper state_dict; VAE: decoder + post_quant_conv only).
* WFIExport: nn.Module whose forward(*inputs, *weights) = functional_call(wrapper, weights, inputs); this is what torch.export sees.
* WFIMod: call-compatible object around a loaded package: m(x, ...) -> loader.boxed_run([x, ..., *weights]).
* AOTRunnerWFI: C.AOTRunner with WFIMod modules (same op order / step maths as the P5 AOT arm) + load-time device-memory bookkeeping.
"""
from __future__ import annotations

import pathlib
import subprocess
import time

import torch
from torch.func import functional_call

from p24.pipeline import LCMPipe, ShapeKey
from scripts import aot_common_S4P_P5 as C

COMPONENTS = C.COMPONENTS
N_INPUTS = {"te": 1, "unet": 3, "vae": 1}


def dev_used_mb() -> float:
    """Device-level used memory of GPU0 (cudaMemGetInfo; includes the CUDA context and everything any process holds)."""
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 2**20


def smi_used_mb() -> float:
    """nvidia-smi memory.used of GPU0 (visible device 0 of this process; one query, ~50 ms; never call inside a timed region)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", "GPU-722679a6-175b-cb56-b764-e8124d262c41"], capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return float("nan")


def mem_snapshot(label: str) -> dict:
    torch.cuda.synchronize()
    return {"label": label, "dev_used_mb": dev_used_mb(), "smi_used_mb": smi_used_mb(),
            "torch_reserved_mb": torch.cuda.memory_reserved() / 2**20, "torch_alloc_mb": torch.cuda.memory_allocated() / 2**20}


def wfi_names_and_weights(p: LCMPipe, comp: str):
    """Ordered names (relative to the component wrapper) and the live weight tensors (same storage as the eager module)."""
    sd = C.wrapper(p, comp).state_dict()
    names = list(sd.keys())
    if comp == "vae":  # decode path only: the encoder weights are not inputs of the package
        names = [n for n in names if n.startswith(("vae.post_quant_conv.", "vae.decoder."))]
    return names, [sd[n] for n in names]


class WFIExport(torch.nn.Module):
    def __init__(self, p: LCMPipe, comp: str):
        super().__init__()
        self.inner = C.wrapper(p, comp)
        self.names, _ = wfi_names_and_weights(p, comp)
        self.n_in = N_INPUTS[comp]

    def forward(self, *args):
        ins, ws = args[: self.n_in], args[self.n_in:]
        return functional_call(self.inner, dict(zip(self.names, ws)), tuple(ins))


def wfi_pkg_path(pkg_root, key: ShapeKey, comp: str) -> pathlib.Path:
    return C.pkg_path(pkg_root, key, comp)


class WFIMod:
    """Call-compatible stand-in for an embedded-weights AOTICompiledModel: forwards the live weights as trailing inputs."""

    def __init__(self, loaded, ws):
        self.m, self.loader, self.ws = loaded, loaded.loader, ws

    def __call__(self, *args):
        return self.loader.boxed_run([*args, *self.ws])


class AOTRunnerWFI(C.AOTRunner):
    """Three weight-free AOT components + eager step maths. name 'AOTW'. The eager weights stay on the GPU and are THE single copy."""
    name = "AOTW"

    def __init__(self, pipe: LCMPipe, pkg_root):
        super().__init__(pipe, pkg_root)
        self.arm = "AOTW"
        self.ws = {c: wfi_names_and_weights(pipe, c)[1] for c in COMPONENTS}
        self.load_log = []

    def _load(self, key: ShapeKey) -> dict:
        from torch._inductor import aoti_load_package
        t, mods = {}, {}
        torch.cuda.synchronize()
        d0, r0 = dev_used_mb(), torch.cuda.memory_reserved() / 2**20
        for c in COMPONENTS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mods[c] = WFIMod(aoti_load_package(str(wfi_pkg_path(self.pkg_root, key, c))), self.ws[c])
            torch.cuda.synchronize()
            t[f"load_{c}_s"] = time.perf_counter() - t0
        self.mods[key] = mods
        t["load_s"] = sum(t.values())
        t["load_dev_delta_mb"] = dev_used_mb() - d0
        t["load_reserved_delta_mb"] = torch.cuda.memory_reserved() / 2**20 - r0
        self.load_log.append({"key": key.tag(), **t})
        return t
