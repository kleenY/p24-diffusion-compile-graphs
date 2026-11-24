#!/bin/bash
# S3-B exclusive block (goals/S3_detailed_goals_v2.md 3.4): REAL GPU validation of the simulator on 3 evaluation-half (half B, seed 100) streams, N=300.
# WRITTEN IN S3 PREP, NOT RUN. Needs GPU0 exclusively (QSA alternates; announce 1 h before, announce "GPU0 returned" after). Run from code/:
#     BEST=<policy> bash scripts/window_S3_B.sh            full block (PART=ALL)
#     BEST=<policy> PART=B1 bash scripts/window_S3_B.sh    split block 1 = W2 + W5 (all policies) + W5 x Ours empty cache      (wrapper: window_S3_B1.sh)
#     BEST=<policy> PART=B2 bash scripts/window_S3_B.sh    split block 2 = W3 (all policies incl. B3')                          (wrapper: window_S3_B2.sh)
#     DRY_RUN=1 BEST=B5 bash scripts/window_S3_B.sh        prints the plan only (no gate, no GPU, no output files)
#   BEST = "best online baseline" named by 3.3 (one of B1 B2 B3' B4 B5 B6 B6'; B3p/B6p aliases accepted). BEST in {B1,B4} is already in the set -> dropped.
#
# RUNS (disk-cache tier = serve_trace --cache-mode warm --warm-root $WARM_ROOT; every run: --exclusive = 23,552 MB hard cap, --require-idle, taskset 0-15):
#   W2, W3, W5 x {B0, B1, B4, Ours, BEST} x 3 processes (proc-id 0-2)            = 45 processes (BEST duplicates dropped)
#   W3 x B3' x 3 processes (PyTorch default recompile limit 8)                   =  3
#   W5 x Ours with an EMPTY cache x 1 process (reported only, never judged)      =  1      -> 49 processes in total
#   Order: process round p (0,1,2) -> workload (W2, W3, W5) -> policies in a list ROTATED by (p + workload index), so no policy is run twice in a
#   row and each policy meets different neighbours in each round (cache / clocks / thermal drift is spread); the empty-cache run is last.
#   Ours uses the frozen ours.params (flat copy of policy_params.json written once to <OUT>/params_ours_flat.json); budgeted policies use budget tier
#   `default` (serve_trace default; B0/B1/B2/B3' have none) = exactly what scripts/predict_S3_4.py simulates.
#
# EXPECTED DURATION (estimate; sim of the DEVELOPMENT analogues of the same three workloads, first 300 requests of half A, 15 s per process for python/
#   import/gate/log overhead (3.1 dev runs: 4-5 s between consecutive CLI runs), 3 min gate + 1.5 GiB pinned self-test; real/sim for these runs was 0.92-1.0 in 3.1):
#     per round, sim seconds:  W2 {B0 68, B1 180, B4 69, Ours 69}  W3 {B0 112, B1 230, B4 113, Ours 113, B3' 155}  W5 {B0 74, B1 195, B4 74, Ours 74}
#                              = 1,525 s without BEST;  BEST adds 255 s (B5, = Ours), 338 (B6), 430 (B2), 0 duplicate (B1/B4/B3')  per round
#     ALL : 3 x (1,525 + BEST) s = 76-98 min service + 49 x 15 s overhead (12 min) + 3 min block start + empty-cache run (~2 min) = ~95-115 min
#           -> 1.6-1.9 h nominal; with a +10 % real/sim error up to ~2.1 h: BORDERLINE vs the 2 h cap  => the split below is recommended.
#     B1  : W2 + W5 + empty: 3 x (804 + BEST(2 workloads)) s = ~47-53 min + 31 x 15 s (8 min) + 5 min = ~60-65 min
#     B2  : W3: 3 x (721 + BEST) s = ~42-45 min + 18 x 15 s (4.5 min) + 3 min = ~50 min
#   Hard stop: BLOCK_MAX_S (default 7200 for ALL, 5400 for B1/B2): a process that has not started when the time is up is skipped and logged
#   (SKIPPED_TIME_BUDGET), a running process is never cut. A skipped process leaves a hole in the balanced design -> decide at acceptance.
#
# PRECONDITIONS (checked here unless DRY_RUN): BEST valid; frozen prediction $PRED exists and is read-only (0444, written by predict_S3_4.py --freeze);
#   warm cache copy $WARM_ROOT exists (scripts/prewarm_S3_4.sh) and scripts/check_prewarm_S3_4.py --strict reports nothing missing.
# GATES (code/docs/run_protocol_S3.md, gate_quiet_v2.sh): before every timed process MemAvailable >= 25 GB, swap < 50 %, load1 <= 16, GPU0 has no other
#   process (wait <= 10 min else stop the block); 1.5 GiB pinned self-test once before the first process (stop + tell the user on failure); leak guard after
#   every process (MemAvailable > 4 GB below block start and not recovered in 60 s -> stop); stderr of EVERY process saved (<tag>.stderr.txt both by
#   serve_trace in raw/ and by run_logged_proc in logs/), <tag>.pinned_fallback_count.txt, <tag>.hostmeta.json; serve_trace meta carries pinned_fallback_count.
#   P24_PINNED_FALLBACK=1 stays on as the safety net (as S3-A / S2 L2 r2); a process with count > 0 is flagged and decided at acceptance, never silently used.
# AFTER THE BLOCK (CPU): python -B -m scripts.compare_S3_4 --pred <PRED> [--tag _B1]  ; then tell the user "GPU0 returned (S3-B)".
set -u
cd "$(dirname "$0")/.." || exit 2
source ../ops/project_env.sh
source scripts/gate_quiet_v2.sh
export TORCHINDUCTOR_COMPILE_THREADS=8 P24_PINNED_FALLBACK=1
export LOAD_MAX=16

PART=${PART:-ALL}
DRY_RUN=${DRY_RUN:-0}
BEST=${BEST:-}
case "$PART" in ALL) WLS="W2 W3 W5"; EMPTY=1; BLOCK_MAX_S=${BLOCK_MAX_S:-7200};;
                B1)  WLS="W2 W5";    EMPTY=1; BLOCK_MAX_S=${BLOCK_MAX_S:-5400};;
                B2)  WLS="W3";       EMPTY=0; BLOCK_MAX_S=${BLOCK_MAX_S:-5400};;
                *) echo "PART must be ALL, B1 or B2"; exit 2;; esac
case "$BEST" in B3p|"B3'"|B3prime) BEST=B3p;; B6p|"B6'"|B6prime) BEST=B6p;; esac
if [ -z "$BEST" ]; then
  if [ "$DRY_RUN" = "1" ]; then BEST="<BEST>"; else echo "BEST is required (best online baseline of 3.3: B1 B2 B3p B4 B5 B6 B6p)"; exit 2; fi
elif ! [[ " B1 B2 B3p B4 B5 B6 B6p " == *" $BEST "* ]]; then echo "BEST=$BEST is not a runnable online baseline"; exit 2; fi

R=../results/S3/real_validation
OUT=$R; RAW=$R/raw; RAWE=$R/raw_empty; LOG=$R/logs
TR=../results/S3/workload/real_validation
WARM_ROOT=${WARM_ROOT:-$P24_CACHE/warm_s3b}
PRED=${PRED:-$R/pred_S3_4.csv}
PJ_FLAT=$R/params_ours_flat.json
declare -A STEM=([W2]=W2_N300_seed100_B [W3]=W3_N300_seed100_B [W5]=W5_N300_seed100_B)
declare -A WIDX=([W2]=0 [W3]=1 [W5]=2)
PIN="taskset -c 0-15"

# ---- plan: ordered list of "workload policy proc mode"
policies_of() {   # $1 = workload -> space-separated policy list (before rotation)
  local l="B0 B1 B4 Ours"
  case " $l " in *" $BEST "*) ;; *) l="$l $BEST";; esac
  if [ "$1" = "W3" ]; then case " $l " in *" B3p "*) ;; *) l="$l B3p";; esac; fi
  echo "$l"
}
PLAN=()
for p in 0 1 2; do
  for w in $WLS; do
    read -ra L <<< "$(policies_of $w)"
    n=${#L[@]}; off=$(( (p + 2 * WIDX[$w]) % n ))
    if [ ${#PLAN[@]} -gt 0 ]; then read -r _w last _p _m <<< "${PLAN[-1]}"; [ "${L[$off]}" = "$last" ] && off=$(( (off + 1) % n )); fi  # never the same policy twice in a row
    for ((i=0;i<n;i++)); do PLAN+=("$w ${L[$(( (i+off) % n ))]} $p warm"); done
  done
done
[ "$EMPTY" = "1" ] && PLAN+=("W5 Ours 0 empty")
prev=""; for e in "${PLAN[@]}"; do read -r _w pol _p _m <<< "$e"; [ "$pol" = "$prev" ] && echo "WARNING adjacent duplicate policy $pol in the plan (design: no policy twice in a row)"; prev=$pol; done

serve_cmd() {   # $1 w  $2 pol  $3 proc  $4 mode -> command line (string)
  local w=$1 pol=$2 p=$3 mode=$4 extra
  if [ "$mode" = "empty" ]; then extra="--cache-mode empty --out $RAWE"; else extra="--cache-mode warm --warm-root $WARM_ROOT --out $RAW"; fi
  echo "$PIN $PROJ_PYTHON -B -m scripts.serve_trace --trace $TR/${STEM[$w]}.jsonl --policy $pol $extra --proc-id $p --exclusive --require-idle --params-json $PJ_FLAT"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN plan (nothing is executed): PART=$PART BEST=$BEST workloads='$WLS' empty_cache_run=$EMPTY block_max=${BLOCK_MAX_S}s"
  echo " warm root: $WARM_ROOT ; prediction: $PRED ; raw: $RAW (empty: $RAWE) ; logs: $LOG ; GPU0 device env exported in the real run"
  echo " steps: wait_quiet_mem 600 -> block_pinned_selftest -> block_mem_start -> per process: quick_quiet_mem||wait_quiet_mem 600, mem_log, run_logged_proc, mem_log, block_mem_guard"
  i=0; for e in "${PLAN[@]}"; do i=$((i+1)); read -r w pol p mode <<< "$e"; echo " [$i] $w $pol p$p $mode :: $(serve_cmd $w $pol $p $mode)"; done
  echo " total processes: ${#PLAN[@]}"
  exit 0
fi

# ---- preconditions (CPU only)
[ -f "$PRED" ] || { echo "S3B_STOP missing frozen prediction $PRED (run predict_S3_4.py --freeze first)"; exit 4; }
[ ! -w "$PRED" ] && [ "$(stat -c %a "$PRED")" = "444" ] || { echo "S3B_STOP prediction $PRED must be 0444"; exit 4; }
[ -d "$WARM_ROOT/C1" ] || { echo "S3B_STOP warm cache $WARM_ROOT missing (scripts/prewarm_S3_4.sh)"; exit 4; }
CUDA_VISIBLE_DEVICES='' nice -n 10 "$PROJ_PYTHON" -B -m scripts.check_prewarm_S3_4 --best-online "$BEST" --streams "${WLS// /,}" --strict > /dev/null \
  || { echo "S3B_STOP warm cache incomplete: run scripts/check_prewarm_S3_4.py --best-online $BEST to list what is missing"; exit 4; }
mkdir -p "$LOG" "$RAW" "$RAWE"
if [ ! -f "$PJ_FLAT" ]; then
  CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B - "$PJ_FLAT" <<'PY'
import json, os, sys
P = json.load(open("../results/S3/policy_params.json"))["ours"]["params"]
json.dump(P, open(sys.argv[1], "w"), indent=1)
os.chmod(sys.argv[1], 0o444)
PY
fi
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41
exec > >(tee -a "$LOG/block_${PART}.log") 2>&1
T0=$(date +%s)
echo "S3B_START $(date +%FT%T) part=$PART best=$BEST block_max=${BLOCK_MAX_S}s processes=${#PLAN[@]} warm_root=$WARM_ROOT pred=$PRED pred_sha16=$(sha256sum "$PRED" | cut -c1-16) params_flat_sha16=$(sha256sum "$PJ_FLAT" | cut -c1-16)"
wait_quiet_mem 600 || { echo "S3B_STOP gate_timeout_at_start"; exit 3; }
block_pinned_selftest || { echo "S3B_STOP pinned_selftest_failed -> tell the user (failure #18 host state)"; exit 6; }
block_mem_start

over_time() { [ $(( $(date +%s) - T0 )) -ge "$BLOCK_MAX_S" ]; }
i=0
for e in "${PLAN[@]}"; do
  i=$((i+1)); read -r w pol p mode <<< "$e"
  tag="${pol}_${w}_${mode}_p${p}"
  if over_time; then echo "SKIPPED_TIME_BUDGET [$i/${#PLAN[@]}] $tag elapsed=$(( $(date +%s) - T0 ))s"; continue; fi
  quick_quiet_mem || wait_quiet_mem 600 || { echo "S3B_STOP gate_timeout_before_$tag"; exit 3; }
  mem_log "before_$tag"
  # shellcheck disable=SC2046
  run_logged_proc "$LOG/$tag" $(serve_cmd $w $pol $p $mode)
  rc=$?
  mem_log "after_$tag"
  echo "STEP [$i/${#PLAN[@]}] $tag rc=$rc elapsed=$(( $(date +%s) - T0 ))s"
  block_mem_guard || { echo "S3B_STOP leak_guard_after_$tag"; exit 5; }
done

# ---- record completeness: stderr + pinned_fallback_count per process, meta field pinned_fallback_count, all statuses
miss=0
for f in "$LOG"/*.hostmeta.json; do
  [ -f "$f" ] || continue
  b=${f%.hostmeta.json}
  [ -f "$b.stderr.txt" ] && [ -f "$b.pinned_fallback_count.txt" ] || { echo "RECORD_INCOMPLETE $b"; miss=1; }
  c=$(cat "$b.pinned_fallback_count.txt" 2>/dev/null || echo "?")
  [ "$c" != "0" ] && echo "PINNED_FALLBACK_FLAG $(basename "$b") count=$c (compile seconds of this process are flagged; decide at acceptance)"
done
CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B - "$RAW" "$RAWE" <<'PY' || miss=1
import glob, json, sys
n = bad = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*.meta.json")):
        m = json.load(open(f)); n += 1
        if "pinned_fallback_count" not in m or m.get("status") != "ok":
            bad += 1; print("META_FLAG", f.rsplit("/", 1)[-1], "status=", m.get("status"), "pinned_fallback_count=", m.get("pinned_fallback_count", "MISSING"))
print(f"META_CHECK n_meta={n} flagged={bad}")
sys.exit(1 if bad else 0)
PY
[ $miss -eq 0 ] && echo "GATE_RECORDS_COMPLETE" || echo "GATE_RECORDS_INCOMPLETE (block marked incomplete)"
mem_log block_end
echo "S3B_END $(date +%FT%T) part=$PART elapsed=$(( $(date +%s) - T0 ))s  -> tell the user: GPU0 returned (S3-B)"
