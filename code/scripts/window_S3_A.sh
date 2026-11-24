#!/bin/bash
# S3-A exclusive block (goals/S3_detailed_goals_v2.md 3.1b): in-process new-shape costs of C2 / C5 and the eviction -> re-capture micro-benchmark.
# WRITTEN IN S3 PREP, NOT RUN. Needs GPU0 exclusively (QSA alternates; announce 1 h before, announce "GPU0 returned" after). Run from code/:
#     bash scripts/window_S3_A.sh            (or  DRY_RUN=1 bash scripts/window_S3_A.sh  to print the plan only: no gate, no GPU, no output)
#
# EXPECTED DURATION  <= 1.5 h  (hard stop: BLOCK_MAX_S=5400 s; steps that have not started by then are skipped and logged, never cut in the middle)
#   gate + 1.5 GiB pinned self-test                                  ~  3 min
#   Part 2  rebuild micro-benchmark, G1 and C1+G1 (2 processes)       ~ 12-18 min  (30 rounds, 60 prepares, 300 steady requests, 2 model loads)
#   Part 1  V3 12 shapes x {C2, C5} x 3 processes in-process seq
#           + 1 warm-up pass per arm (WARM_PASS=1)                    ~ 45-60 min (8 processes x 5-8 min; the C2 sequence ends early at the OOM)
#   -> total estimate ~ 65-80 min.  Part 2 runs first on purpose: its length is fixed, Part 1 is the part the time guard may truncate.
#
# PART 1 (3.1b first part). One process serves the 12 V3 shapes in ascending UNet work (same K12 order as window_S2_5.sh) with the arm C2
#   (reduce-overhead) or C5 (dynamic=True), disk-cache tier, 3 independent processes per arm (proc-id 0-2) -> every shape's FIRST call cost in a
#   running process is recorded (bench_seq: per-key CSV + meta, key i>0 labelled inproc_first, meta cache_mode=disk -> inproc_diskcache in build_cost_table).
#   The pre-warmed per-arm compile dirs ($P24_CACHE/warm_s2/C2, /C5) may lack some of the 12 shapes: with WARM_PASS=1 one extra process per arm
#   (proc-id 9, dir warmpass_<arm>, NOT for the cost table) first compiles whatever is missing so that the 3 measured processes are disk-warm.
#   C2 keeps every shape's CUDA graph pool in one cudagraph tree and cannot release one shape: the cumulative pool is expected to hit
#   torch.OutOfMemoryError at some shape. bench_seq records that as status=fail with the error text and stops that process. This is EVIDENCE
#   for failure-catalogue entry #19 (graph pools accumulate), NOT a block failure: the block continues, the OOM key/error is copied to
#   S3A_logs/oom_evidence.txt, and keys after the OOM stay "extrapolated" in the cost table (acceptance: rows with `inproc_measured_S3`).
# PART 2 (3.1b second part). scripts/bench_rebuild_S3.py: G1 and C1+G1, shapes 512x512x1x0, 768x512x1x0, 512x512x4x1, 5 rounds each:
#   capture -> steady -> release + empty_cache -> re-capture; records capture_s, rebuild seconds and whether memory_reserved falls back (<= 64 MB).
#   Rule for the simulator: rebuild / capture_s > 1.5 -> CostView.rebuild_override with the measured rebuild seconds.
#
# GATES (code/docs/run_protocol_S3.md, gate_quiet_v2.sh): before every timed process MemAvailable >= 25 GB, swap < 50 %, load1 <= 16, GPU0 has no other
# process (wait <= 10 min else stop the block); 1.5 GiB pinned-memory self-test once before the first process (stop + tell the user on failure);
# leak guard after every process (MemAvailable more than 4 GB below the block start and not recovered in 60 s -> stop); stderr of EVERY process
# saved as <tag>.stderr.txt; <tag>.pinned_fallback_count.txt and <tag>.hostmeta.json (pinned_fallback_count, MemAvailable, Shmem, uptime) per process;
# bench_rebuild_S3.py additionally writes pinned_fallback_count into its own meta json. P24_PINNED_FALLBACK=1 is kept as the safety net (as S2 L2 r2).
# Timing discipline: taskset -c 0-15 (P cores), load <= 16; a process with pinned_fallback_count > 0 or NOT_QUIET at start is flagged in the log and
# decided on at acceptance, never silently used.
set -u
cd "$(dirname "$0")/.." || exit 2
source ../ops/project_env.sh
source scripts/gate_quiet_v2.sh
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41 TORCHINDUCTOR_COMPILE_THREADS=8 P24_PINNED_FALLBACK=1
export LOAD_MAX=16

BLOCK_MAX_S=${BLOCK_MAX_S:-5400}
WARM_PASS=${WARM_PASS:-1}
DRY_RUN=${DRY_RUN:-0}
PARTS=${PARTS:-"2 1"}                      # order of the parts; "1" or "2" runs one part only
R=../results/S3/cost_addendum
SEQ=$R/inproc_seq                          # Part 1 raw (bench_seq layout, like results/S2/inproc_seq)
RB=$R/rebuild                              # Part 2 raw
LOG=$R/S3A_logs
W=$P24_CACHE/warm_s2
K12=512x512x1x0,512x768x1x0,768x512x1x0,512x512x1x1,512x512x2x0,512x512x3x0,512x768x1x1,512x768x2x0,768x512x1x1,768x512x2x0,512x512x2x1,512x512x4x0
PIN="taskset -c 0-15"
ARMS="C2 C5"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN plan (nothing is executed):"
  echo " block max ${BLOCK_MAX_S}s; parts order: $PARTS; warm pass: $WARM_PASS; raw -> $R ; logs -> $LOG"
  for part in $PARTS; do
    if [ "$part" = "2" ]; then
      for arm in G1 C1+G1; do echo " [part2] $PIN $PROJ_PYTHON -B -m scripts.bench_rebuild_S3 --arm $arm --out $RB --proc-id 0 --cache disk --warm-root $W"; done
    else
      for arm in $ARMS; do [ "$WARM_PASS" = "1" ] && echo " [part1] warm pass: $PIN $PROJ_PYTHON -B -m scripts.bench_seq --arm $arm --keys K12 --cache disk --warm-root $W --out $SEQ/warmpass_$arm --proc-id 9"; done
      for p in 0 1 2; do for arm in $ARMS; do echo " [part1] $PIN $PROJ_PYTHON -B -m scripts.bench_seq --arm $arm --keys K12 --cache disk --warm-root $W --out $SEQ/v3shapes_${arm}_disk --proc-id $p"; done; done
    fi
  done
  echo " K12=$K12"
  exit 0
fi

mkdir -p "$LOG" "$SEQ" "$RB"
exec > >(tee -a "$LOG/block.log") 2>&1
T0=$(date +%s)
echo "S3A_START $(date +%FT%T) block_max=${BLOCK_MAX_S}s parts='$PARTS' warm_pass=$WARM_PASS"
wait_quiet_mem 600 || { echo "S3A_STOP gate_timeout_at_start"; exit 3; }
block_pinned_selftest || { echo "S3A_STOP pinned_selftest_failed -> tell the user (failure #18 host state)"; exit 6; }
block_mem_start

over_time() { [ $(( $(date +%s) - T0 )) -ge "$BLOCK_MAX_S" ]; }

# step <tag> <cmd...> : gated + logged timed process. Always returns 0 unless the block must stop (exit inside).
step() {
  local tag=$1; shift
  if over_time; then echo "SKIPPED_TIME_BUDGET $tag elapsed=$(( $(date +%s) - T0 ))s"; return 0; fi
  quick_quiet_mem || wait_quiet_mem 600 || { echo "S3A_STOP gate_timeout_before_$tag"; exit 3; }
  mem_log "before_$tag"
  run_logged_proc "$LOG/$tag" "$@"
  local rc=$?
  mem_log "after_$tag"
  echo "STEP $tag rc=$rc elapsed=$(( $(date +%s) - T0 ))s"
  block_mem_guard || { echo "S3A_STOP leak_guard_after_$tag"; exit 5; }
  return 0
}

# oom_note <arm> <proc> : copy the first non-ok key of a finished sequence (C2 cumulative-pool OOM = #19 evidence) to oom_evidence.txt
oom_note() {
  local meta="$SEQ/v3shapes_${1}_disk/${1}_seq_p${2}.meta.json"
  [ -f "$meta" ] || { echo "OOM_NOTE no meta for $1 p$2" >> "$LOG/oom_evidence.txt"; return 0; }
  "$PROJ_PYTHON" -B - "$meta" "$1" "$2" >> "$LOG/oom_evidence.txt" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
n_ok = sum(1 for r in m.get("per_key", []) if r["status"] == "ok")
bad = [r for r in m.get("per_key", []) if r["status"] != "ok"]
if bad:
    r = bad[0]
    print(f"{sys.argv[2]} p{sys.argv[3]}: {n_ok} shapes ok, FAIL at idx {r['idx']} key {r['key']}: {str(r['error'])[:300]}")
else:
    print(f"{sys.argv[2]} p{sys.argv[3]}: all {n_ok} shapes ok (no OOM)")
PY
  tail -n 1 "$LOG/oom_evidence.txt"
}

for part in $PARTS; do
  if [ "$part" = "2" ]; then
    echo "=== PART 2: eviction -> re-capture micro-benchmark ==="
    for arm in G1 C1+G1; do
      step "rebuild_${arm}_p0" $PIN $PROJ_PYTHON -B -m scripts.bench_rebuild_S3 --arm "$arm" --out $RB --proc-id 0 --cache disk --warm-root $W
    done
  elif [ "$part" = "1" ]; then
    echo "=== PART 1: V3 12 shapes x C2/C5 x 3 processes (in-process sequences) ==="
    if [ "$WARM_PASS" = "1" ]; then
      for arm in $ARMS; do
        step "warmpass_${arm}_p9" $PIN $PROJ_PYTHON -B -m scripts.bench_seq --arm $arm --keys $K12 --cache disk --warm-root $W --out $SEQ/warmpass_$arm --proc-id 9
      done
    fi
    for p in 0 1 2; do
      for arm in $ARMS; do
        step "v3shapes_${arm}_disk_p${p}" $PIN $PROJ_PYTHON -B -m scripts.bench_seq --arm $arm --keys $K12 --cache disk --warm-root $W --out $SEQ/v3shapes_${arm}_disk --proc-id $p
        [ "$arm" = "C2" ] && oom_note C2 $p
      done
    done
  fi
done

# record completeness: every timed process needs stderr + pinned-fallback files
miss=0
for f in "$LOG"/*.hostmeta.json; do
  [ -f "$f" ] || continue
  b=${f%.hostmeta.json}
  [ -f "$b.stderr.txt" ] && [ -f "$b.pinned_fallback_count.txt" ] || { echo "RECORD_INCOMPLETE $b"; miss=1; }
  c=$(cat "$b.pinned_fallback_count.txt" 2>/dev/null || echo "?")
  [ "$c" != "0" ] && echo "PINNED_FALLBACK_FLAG $(basename "$b") count=$c (compile seconds of this process are flagged; decide at acceptance)"
done
[ $miss -eq 0 ] && echo "GATE_RECORDS_COMPLETE" || echo "GATE_RECORDS_INCOMPLETE (block marked incomplete)"
mem_log block_end
echo "S3A_END $(date +%FT%T) elapsed=$(( $(date +%s) - T0 ))s  -> tell the user: GPU0 returned (S3-A)"
