"""Host-specific workarounds (failure catalogue #18, diagnosed 2026-11-16).

On 2026-11-15/16 this host (driver 535.309.01, kernel 6.11, IOMMU on; up since 10-08) refused every single PINNED host
allocation larger than 1 GiB with cudaErrorInvalidValue (cudaHostAlloc and cudaHostRegister alike; repro in
results/S2/diag/pinned/). Revision 2026-11-17: after the reboot of 11-17 the same 1152 MiB pinned allocation succeeds, so this
is a host-STATE condition (it coincided with ~56 GB of host memory pinned by no visible process), not a fixed limit. Inductor's runtime
autotuner backs up an in-place kernel argument into pinned host memory when GPU headroom (max_allocated - allocated) is
smaller than the argument; at 768² B4 CFG the VAE-decoder buffer (4,256,768,768) fp16 = 1.15 GiB hits this, so a cold
compile fails whenever the headroom happens to be small.
apply_pinned_fallback(): if that pinned allocation fails, back the argument up in pageable host memory instead (synchronous
copy). Only the backup during autotuning changes; kernels, chosen configs and numerics are untouched.
Enabled by P24_PINNED_FALLBACK=1 (recorded in run meta); off by default so earlier measurements stay reproducible."""
from __future__ import annotations

import os

APPLIED = False


def apply_pinned_fallback() -> bool:
    global APPLIED
    if APPLIED:
        return True
    import torch
    from torch._inductor.runtime import triton_heuristics as th
    from torch._inductor.runtime.triton_heuristics import compute_required_storage_length

    def copy_args_to_cpu_if_needed(self, *args, **kwargs):
        if not self.optimize_mem:
            return {}
        copies = {}
        budget = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()

        def maybe_copy(name, arg):
            nonlocal budget
            if name in self.mutated_arg_names and arg.is_cuda:
                n = compute_required_storage_length(arg.size(), arg.stride(), 0)
                size = n * arg.element_size()
                if size > budget:
                    try:
                        cpu_arg = torch.empty_strided((n,), (1,), dtype=arg.dtype, device="cpu", pin_memory=True)
                        cpu_arg.copy_(arg.as_strided((n,), (1,)), non_blocking=True)
                    except Exception:  # pinned allocation refused by this host -> pageable memory
                        import sys
                        print(f"P24_PINNED_FALLBACK_USED kernel={self.inductor_meta.get('kernel_name')} arg={name} "
                              f"bytes={size}", file=sys.stderr, flush=True)
                        torch.cuda.synchronize()
                        cpu_arg = torch.empty_strided((n,), (1,), dtype=arg.dtype, device="cpu", pin_memory=False)
                        cpu_arg.copy_(arg.as_strided((n,), (1,)))
                    copies[name] = (arg, cpu_arg)
                else:
                    budget -= size

        for name, arg in zip(self.fn.arg_names, args):
            maybe_copy(name, arg)
        for name, arg in kwargs.items():
            maybe_copy(name, arg)
        return copies

    th.CachingAutotuner.copy_args_to_cpu_if_needed = copy_args_to_cpu_if_needed
    APPLIED = True
    return True


if os.environ.get("P24_PINNED_FALLBACK") == "1":
    apply_pinned_fallback()
