#!/bin/bash
# S2 formal block S-6 (plan 2.6 + §9.9; protocol addendum v2 change 3): ≤2 h exclusive GPU0.
#   (1) input_s micro-benchmark: 5 processes (scripts/bench_input.py) -> results/S2/input_s
#   (2) in-process gap: 512² B1 no-CFG as a LATER shape after 512² B2 (E0, G1, C1 disk; C1 empty) x3 -> results/S2/inproc_seq/gap_*
#       (the 12-shape sequences started with it, so its inproc tier was missing)
#   (3) validation: V1/V2/V3 x B0/B4/B1 x 3 processes (disk cache warm_s2), interleaved by process round;
#       B4 with a graph budget (AUTO_DECISIONS 2026-11-15 23:35 / 23:40): V1 16633 MB, V2/V3 13065 MB,
#       predicted pools from the merged cost table; every process hard-capped at 20480 MB (set_per_process_memory_fraction).
#       + V3 x B1 empty cache x1 (plan 2.6).
set -u
source ../ops/project_env.sh; source scripts/gate_quiet.sh
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41 TORCHINDUCTOR_COMPILE_THREADS=8
export LOAD_MAX=16
wait_quiet ${GATE_TIMEOUT_S:-3600} || exit 3
echo "WINDOW_S2_6_START $(date +%FT%T)"
R=../results/S2; W=$P24_CACHE/warm_s2; CT=$R/cost_table/merged/cost_table.csv
TS="taskset -c 0-15 $PROJ_PYTHON -B -m"
for p in 0 1 2 3 4; do $TS scripts.bench_input --out $R/input_s --proc-id $p > /dev/null; echo "input p$p rc=$? $(date +%T)"; done
$TS scripts.bench_input --out $R/input_s --summarize
G2=512x512x2x0,512x512x1x0   # B2 first: B1 text encoder, UNet and VAE of 512² B1 are new in-process; key 0 in the
#   empty-cache run is also the C1 cold_empty cost of 512² B2 (first shape of V3; missing from the cost table)
for p in 0 1 2; do
  for a in E0 G1 C1; do $TS scripts.bench_seq --arm $a --keys $G2 --cache disk --warm-root $W --out $R/inproc_seq/gap_${a}_disk --proc-id $p; echo "gap $a disk p$p rc=$? $(date +%T)"; done
  $TS scripts.bench_seq --arm C1 --keys $G2 --cache empty --out $R/inproc_seq/gap_C1_empty --proc-id $p; echo "gap C1 empty p$p rc=$? $(date +%T)"
done
VS=$R/workload/validation_streams; OUT=$R/sim_validation/raw
declare -A BUD=([V1_N400_seed0_A]=16633 [V2_N300_seed1_A]=13065 [V3_N200_seed2_A]=13065)
for p in 0 1 2; do
  quick_quiet || echo "NOT_QUIET p$p load=$(cut -d' ' -f1 /proc/loadavg)"
  for s in V1_N400_seed0_A V2_N300_seed1_A V3_N200_seed2_A; do
    for pol in B0 B4 B1; do
      X=""; [ $pol = B4 ] && X="--cost-table $CT --mem-budget-mb ${BUD[$s]}"
      $TS scripts.serve_trace --trace $VS/$s.jsonl --policy $pol --cache-mode warm --warm-root $W --out $OUT \
        --proc-id $p --require-idle $X | tail -1
      echo "serve $s $pol p$p rc=${PIPESTATUS[0]} $(date +%T)"
    done
  done
done
$TS scripts.serve_trace --trace $VS/V3_N200_seed2_A.jsonl --policy B1 --cache-mode empty --out $R/sim_validation/raw_empty \
  --proc-id 0 --require-idle | tail -1
echo "serve V3 B1 empty rc=${PIPESTATUS[0]} $(date +%T)"
echo "WINDOW_S2_6_END $(date +%FT%T)"
