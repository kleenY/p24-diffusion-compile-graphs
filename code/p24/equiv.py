"""Equivalence metrics: arm output vs fp32-eager reference and vs fp16-eager (E0)."""
from __future__ import annotations

import math
import torch


def rmse(a, b):
    return (a.float() - b.float()).pow(2).mean().sqrt().item()


def psnr(a, b):
    m = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if m == 0 else 10 * math.log10(1.0 / m)


_lp = None


def lpips_mean(a, b):
    global _lp
    import lpips
    if _lp is None:
        _lp = lpips.LPIPS(net="alex", verbose=False).to(a.device).eval()
    with torch.no_grad():
        return _lp(a.float() * 2 - 1, b.float() * 2 - 1).mean().item()


def compare(x, e0, ref32):
    """x, e0, ref32: (N,3,H,W) in [0,1]."""
    r_x, r_e0 = rmse(x, ref32), rmse(e0, ref32)
    return {
        "maxabs_vs_ref32": (x.float() - ref32.float()).abs().max().item(),
        "rmse_vs_ref32": r_x, "rmse_E0_vs_ref32": r_e0,
        "rmse_ratio": r_x / r_e0 if r_e0 > 0 else (0.0 if r_x == 0 else float("inf")),
        "psnr_vs_ref32": psnr(x, ref32), "psnr_E0_vs_ref32": psnr(e0, ref32),
        "maxabs_vs_E0": (x.float() - e0.float()).abs().max().item(),
        "lpips_vs_E0": lpips_mean(x, e0),
    }
