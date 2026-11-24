#!/bin/bash
# S2 step 2.7 (closes L3/L4): equivalence extension, DEVELOPMENT mode (not a timed block; GPU0, nothing else of mine on it).
# Plan 2.7 + §9.8: 8 prompts, seeds 7000+i, fp16 noise; RMSE ratio ≤2.0 and LPIPS ≤0.02 vs fp32 eager (tests/tolerances.json).
# Every process is hard-capped at 20480 MB (set_per_process_memory_fraction via a runpy wrapper; run_equivalence.py unchanged);
# an OOM is recorded as a failed cell, never retried with a larger cap. Compiled arms use warm_s2 (judged numerics = timed kernels).
set -u
source ../ops/project_env.sh
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41 TORCHINDUCTOR_COMPILE_THREADS=8
OUT=../results/S2/equiv; mkdir -p $OUT; W=$P24_CACHE/warm_s2
EQ() { taskset -c 0-15 $PROJ_PYTHON -B -c '
import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(min(1.0, 20480 / (torch.cuda.get_device_properties(0).total_memory / 2**20)), 0)
sys.argv = ["run_equivalence"] + sys.argv[1:]
runpy.run_module("scripts.run_equivalence", run_name="__main__")' "$@"; }
# (a) main arms: S1 cells re-run with SHA (512² B1 off/on), B=4, every resolution at B1 and B4 (CFG on at B4), probe 256²
K=256x256x1x0,512x512x1x0,512x512x1x1,512x512x4x0,512x512x4x1,768x512x1x0,512x768x1x0,768x512x4x1,512x768x4x1,768x768x1x0
# 768² B4 CFG in its own process per arm: a failure (e.g. fp32 reference OOM under the cap) must not void the other cells
for a in E0 G1 C1 C1+G1 C2; do EQ --arm $a --keys $K --out $OUT --warm-root $W --tag _main > $OUT/log_${a}_main.txt 2>&1; echo "equiv $a main rc=$? $(date +%T)"; done
for a in E0 G1 C1 C1+G1 C2; do EQ --arm $a --keys 768x768x4x1 --out $OUT --warm-root $W --tag _768b4 > $OUT/log_${a}_768b4.txt 2>&1; echo "equiv $a 768b4 rc=$? $(date +%T)"; done
# (b) padding B3 -> B4 (arm under test padded; ref32 and E0 native), CFG off and on
for a in G1 C1 C1+G1; do EQ --arm $a --keys 512x512x3x0,512x512x3x1 --pad-b3 4 --out $OUT --warm-root $W --tag _pad > $OUT/log_${a}_pad.txt 2>&1; echo "equiv $a pad rc=$? $(date +%T)"; done
# (c) baselines: C3 at 512² B1; C5, C6 at 768² B1
EQ --arm C3 --keys 512x512x1x0 --out $OUT --warm-root $W --tag _base > $OUT/log_C3_base.txt 2>&1; echo "equiv C3 rc=$? $(date +%T)"
for a in C5 C6; do EQ --arm $a --keys 768x768x1x0 --out $OUT --warm-root $W --tag _base > $OUT/log_${a}_base.txt 2>&1; echo "equiv $a rc=$? $(date +%T)"; done
echo "EQUIV_S2_7_END $(date +%FT%T)"
