#!/bin/bash
# X1 screen block (dev mode, 2026-11-17). Usage: accel_screen.sh <512|768> [rounds=3]
# 8 arms interleaved round-robin (forward in even rounds, reverse in odd rounds), 3 processes per arm, reps 20 warm 3 comp-reps 0.
# E0-cl/E0-cb share baseline E0; C1+G1-cb/C1+G1-recipe share baseline C1+G1 (deviation from "5 separate ABAB pairs": saves processes).
# C1+G1 / C1+G1-cb use warm_s2/C1_G1 caches directly (hits only; checked by file listing before/after); recipe uses x1_recipe.
set -u
source ../ops/project_env.sh; source scripts/gate_quiet_v2.sh
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41 TORCHINDUCTOR_COMPILE_THREADS=8 LOAD_MAX=16
SH=${1:?}; RO=${2:-3}
if [ "$SH" = 512 ]; then H=512; B=1; CFG=0; else H=768; B=4; CFG=1; fi
OUT=../results/S3/accel/screen/s${SH}; mkdir -p $OUT
WS=$P24_CACHE/warm_s2/C1_G1; XR=$P24_CACHE/x1_recipe; mkdir -p $XR/inductor $XR/triton
ARMS=("E0:none" "E0:cl" "E0:cb" "G1:none" "G1:cl" "C1+G1:none" "C1+G1:cb" "C1+G1:recipe")
find $WS -type f -printf '%p %s %T@\n' | sort > $OUT/warm_s2_C1G1_listing_before.txt
echo "SCREEN_START $SH $(date +%FT%T) arms=${#ARMS[@]} rounds=$RO"; mem_log start
wait_quiet_mem 1800 || exit 3
block_mem_start
for ((p=0; p<RO; p++)); do
  if (( p % 2 == 0 )); then idx=$(seq 0 $((${#ARMS[@]}-1))); else idx=$(seq $((${#ARMS[@]}-1)) -1 0); fi
  for i in $idx; do
    a=${ARMS[$i]}; arm=${a%%:*}; knob=${a##*:}; name=$arm; [ $knob != none ] && name=$arm-$knob
    tier=na
    case "$arm:$knob" in
      "C1+G1:none"|"C1+G1:cb") tier=cold_diskcache; export TORCHINDUCTOR_CACHE_DIR=$WS/inductor TRITON_CACHE_DIR=$WS/triton;;
      "C1+G1:recipe") export TORCHINDUCTOR_CACHE_DIR=$XR/inductor TRITON_CACHE_DIR=$XR/triton
           if [ "$(find $XR -type f | wc -l)" -lt 5 ] || ! ls $XR/inductor | grep -q . ; then tier=x1_cold; else tier=x1_warm; fi;;
      *) export TORCHINDUCTOR_CACHE_DIR=$P24_CACHE/inductor TRITON_CACHE_DIR=$P24_CACHE/triton;;
    esac
    # recipe: tier from whether this shape was compiled before in x1_recipe (marker file per shape)
    if [ "$knob" = recipe ]; then [ -f $XR/done_$SH ] && tier=x1_warm || tier=x1_cold; fi
    quick_quiet_mem || wait_quiet_mem 1800 || { echo "ABORT not quiet before $name p$p"; exit 3; }
    echo "RUN $name p$p tier=$tier $(date +%T)"
    run_logged_proc "$OUT/${name}_p$p" nice -n 10 taskset -c 0-15 $PROJ_PYTHON -B -m scripts.accel_bench --knob $knob --arm $arm \
      --H $H --W $H --B $B --cfg $CFG --reps 20 --warm 3 --comp-reps 0 --out $OUT --proc-id $p --cold-tier $tier | tail -1
    rc=${PIPESTATUS[0]}; echo "done $name p$p rc=$rc $(date +%T)"
    [ "$knob" = recipe ] && [ $rc = 0 ] && touch $XR/done_$SH
    block_mem_guard || { echo "ABORT mem guard"; exit 5; }
  done
done
find $WS -type f -printf '%p %s %T@\n' | sort > $OUT/warm_s2_C1G1_listing_after.txt
diff $OUT/warm_s2_C1G1_listing_before.txt $OUT/warm_s2_C1G1_listing_after.txt > $OUT/warm_s2_C1G1_listing_diff.txt && echo "WARM_S2_UNCHANGED" || echo "WARM_S2_CHANGED (see listing_diff)"
echo "SCREEN_END $SH $(date +%FT%T)"
