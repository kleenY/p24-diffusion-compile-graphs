"""Run another scripts/* module in this process after setting a hard per-process GPU memory cap (S4 4.4b `--quick`).

    python -B scripts/quick_capped_S4.py --cap-mb 14336 --cache-root DIR --module scripts.bench_seq -- <args of that module>

Why a wrapper: scripts/bench_seq.py has no memory-cap option and is frozen (S2). The wrapper
  1. sets TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR to DIR/<arm>/{inductor,triton} BEFORE torch is imported (fresh, empty directory => empty compile
     cache; the caller deletes DIR afterwards),
  2. calls torch.cuda.set_per_process_memory_fraction(cap / total) on the visible GPU (the caller exports CUDA_VISIBLE_DEVICES = GPU0 only),
  3. runs the module with runpy as __main__ (argv = the arguments after `--`; the module is run with `--cache env`).
Nothing in the wrapped module is changed.  CPU-only importability is checked by the dry-run (nothing is imported from torch there).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mb", type=float, required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--cache-tag", required=True, help="sub-directory of --cache-root (one per process => every process starts from an empty cache)")
    ap.add_argument("--module", required=True)
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    cdir = pathlib.Path(a.cache_root) / a.cache_tag
    if cdir.exists() and any(cdir.iterdir()):
        sys.exit(f"cache dir {cdir} is not empty (the --quick run must start from an EMPTY compile cache)")
    cdir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cdir / "triton")
    import torch
    torch.cuda.init()
    total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_mb / total_mb), 0)
    print(f"quick_capped: cap {a.cap_mb:.0f} MB of {total_mb:.0f} MB, cache {cdir}", flush=True)
    sys.argv = [a.module] + rest
    runpy.run_module(a.module, run_name="__main__")


if __name__ == "__main__":
    main()
