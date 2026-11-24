#!/bin/bash
# S4-X X10 exclusive window = SD-Turbo (SD2.1 distilled, 1 step, guidance 0, fp16) MECHANISM CHECK on GPU0, <= 1 h (preregistration goals/S4X_X10_second_model_mechanism_prereg_v1.md = the only basis).
# Same gates / STOP rules / records as window_S4X_X8.sh (that file is UNCHANGED): per timed process MemAvailable >= 25 GB, swap < 50 %, load1 <= 16, GPU0 has no other process (wait <= 10 min else stop);
#   1.5 GiB pinned self-test once per block; leak guard (MEM_DROP_MB default 12288); stderr of EVERY process saved, <tag>.pinned_fallback_count.txt, <tag>.hostmeta.json; P24_PINNED_FALLBACK=1;
#   taskset -c 0-15; TORCHINDUCTOR_COMPILE_THREADS=8.  STOP (no retry) on pinned self-test failure, pinned count > 0, rc != 0, bench meta status != ok, a skipped (gate-timeout) matrix process, leak guard
#   -> records finished, exit code 5/6/7/8/9; the verdict then reads INVALID.
# Measurement code = the S2 code UNCHANGED, only the model class swapped (scripts/x10_common.py): scripts/x10_prewarm.py (runs prewarm_s2.py), scripts/x10_run_matrix.py (runs run_matrix.py, S2 formal
#   arguments: --cache-mode warm --order shape --formal, reps 20, warm 3, comp-reps 5 = the bench defaults), scripts/x10_bench.py (runs p24.bench.main()), scripts/x10_equiv.py (copy of run_equivalence.py logic),
#   Nsight Systems E0 / C1 traces as in window_S2_1.sh (nsys profile -t cuda,nvtx --cuda-graph-trace=node, --nsys-window 3) + scripts/x10_profile_analyze.py (imports ceiling_S4X classify / load_trace).
# ORDER (card sec 1: equivalence -> steady -> profile; the compile caches of C1 / C1+G1 must be pre-warmed first, exactly as S2 formal: the pre-warm process of each key is also the cold-EMPTY-cache measurement):
#   [1] prewarm C1, C1+G1 x 3 keys (cold compile)   [2-5] equivalence E0 (+ official-diffusers check), C1, G1, C1+G1 on warm caches   [6] steady E0,G1,C1,C1+G1 x 3 keys x PROCS procs (shape-major, rotating arm order)
#   [7-8] Nsight E0, C1 at 512x512 B1 no-CFG.   Keys: 512x512x1x0 (judged), 384x384x1x0, 512x512x4x0 (reported).
# Time: estimate printed by DRY_RUN (S2 measured per-process seconds + assumptions); the block is REFUSED when the x1.5 conservative estimate is > 1 h (BLOCK_LIMIT_S, default 3600) unless ALLOW_LONG=1 after telling
#   the user; reduce PROCS (card: >= 3 processes per cell; default 5 = S2) to shorten.  BLOCK_MAX_S (default 3600): no new step starts after this many seconds.
# WRITTEN IN S4-X X10 PREP, NOT RUN.  GPU0 exclusive for <= 1 h (announce to QSA papers 1/2 first, host CPU load <= 16 during timing, register STATE/GPU_QUEUE.md; say "GPU0 returned (S4X-X10)" after).  From code/:
#     DRY_RUN=1 bash scripts/window_S4X_X10.sh      plan + estimate + CPU checks (--help of every script, config.json + scheduler check); no gate / GPU / output under results/.
#     bash scripts/window_S4X_X10.sh                (PROCS=3 bash ... for the shorter variant)
# NEEDS before a real block: card 0444 + results/S4X/x10/PREREG_SHA16.txt (0444; = first 16 hex of sha256 of the card, written by the main session), all x10 scripts + this file 0444, results/S4X/x10/revision.json 0444.
# AFTER THE BLOCK (CPU): "$PROJ_PYTHON" -B -m scripts.x10_profile_analyze ; "$PROJ_PYTHON" -B -m scripts.m12_verdict_S4X_X10 ; then tell the user "GPU0 returned (S4X-X10)".
set -u
cd "$(dirname "$0")/.." || exit 2
source ../ops/project_env.sh
export MEM_DROP_MB=${MEM_DROP_MB:-12288}
source scripts/gate_quiet_v2.sh
export TORCHINDUCTOR_COMPILE_THREADS=8 P24_PINNED_FALLBACK=1 LOAD_MAX=16
export X10_MEM_MIN_MB=$MEM_MIN_MB X10_MEM_DROP_MB=$MEM_DROP_MB

DRY_RUN=${DRY_RUN:-0}
ALLOW_LONG=${ALLOW_LONG:-0}
PROCS=${PROCS:-5}
BLOCK_MAX_S=${BLOCK_MAX_S:-3600}
BLOCK_LIMIT_S=${BLOCK_LIMIT_S:-3600}
[ "$PROCS" -ge 3 ] 2>/dev/null || { echo "PROCS must be an integer >= 3 (card: >= 3 processes per cell)"; exit 2; }

X10D=../results/S4X/x10
PW=$X10D/prewarm; EQ=$X10D/equiv; ST=$X10D/steady; PF=$X10D/profile; LG=$X10D/logs; VD=$X10D/verdict
PREREG=../goals/S4X_X10_second_model_mechanism_prereg_v1.md
PREREG_SHA_FILE=$X10D/PREREG_SHA16.txt
REVJSON=$X10D/revision.json
WARM=${X10_WARM_ROOT:-$P24_CACHE/x10/warm_s2}            # no spaces; its basename must be warm_s2 (run_matrix --formal self-check); fresh per X10 (SD-Turbo never shares a cache with SD1.5)
KEYS=512x512x1x0,384x384x1x0,512x512x4x0
NKEYS=3
PY="$PROJ_PYTHON"
PYC() { CUDA_VISIBLE_DEVICES='' nice -n 10 "$PROJ_PYTHON" -B "$@"; }
FROZEN=(scripts/x10_common.py scripts/x10_bench.py scripts/x10_run_matrix.py scripts/x10_prewarm.py scripts/x10_equiv.py scripts/x10_profile_analyze.py scripts/m12_verdict_S4X_X10.py scripts/window_S4X_X10.sh)
NSYS=/usr/local/cuda/bin/nsys

# ---- fixed design: "<name>"
STEPS=(prewarm equiv_E0 equiv_C1 equiv_G1 equiv_C1+G1 steady profile_E0 profile_C1)
step_cmd() {   # $1 name -> command line (no spaces inside words)
  local n=$1 a
  case "$n" in
    prewarm)  echo "taskset -c 0-15 $PY -B -m scripts.x10_prewarm --arms C1,C1+G1 --keys $KEYS --out $PW --max-wait-s 600 --warm-root $WARM";;
    equiv_*)  a=${n#equiv_}; echo "taskset -c 0-15 $PY -B -m scripts.x10_equiv --arm $a --keys $KEYS --out $EQ --warm-root $WARM$([ "$a" = E0 ] && echo ' --diffusers-check')";;
    steady)   echo "taskset -c 0-15 $PY -B -m scripts.x10_run_matrix --arms E0,G1,C1,C1+G1 --keys $KEYS --procs $PROCS --reps 20 --out $ST --require-idle --max-wait-s 900 --cache-mode warm --order shape --formal --warm-root $WARM";;
    profile_*) a=${n#profile_}
      local envp=""; [ "$a" = C1 ] && envp="env TORCHINDUCTOR_CACHE_DIR=$WARM/C1/inductor TRITON_CACHE_DIR=$WARM/C1/triton"
      echo "$envp timeout 900 taskset -c 0-15 $NSYS profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o $PF/${a}_512x512_b1_cfg0 $PY -B -m scripts.x10_bench --arm $a --H 512 --W 512 --B 1 --cfg 0 --reps 3 --warm 2 --comp-reps 0 --nsys-window 3 --out $PF/bench --proc-id 0 --cold-tier profile";;
  esac
}

estimate() {   # CPU: S2 measured per-process seconds + assumptions (printed)
  PYC - "$PROCS" <<'PY'
import collections, json, statistics, sys
procs = int(sys.argv[1])
keys = [(512, 512, 1, 0), (384, 384, 1, 0), (512, 512, 4, 0)]
d = collections.defaultdict(list)
for l in open("../results/S2/cost_table/S-1/main/driver_log.jsonl"):
    r = json.loads(l)
    if "arm" in r and "secs" in r and r.get("rc") == 0: d[(r["arm"], tuple(r["key"]))].append(r["secs"])
pw = collections.defaultdict(list)
for l in open("../results/S2/prewarm_final/prewarm_log.jsonl"):
    r = json.loads(l)
    if r.get("secs") and r.get("key") and r.get("rc") == 0: pw[(r["arm"], tuple(r["key"]))].append(r["secs"])
med = lambda dd, k: statistics.median(dd[k])
pre = sum(med(pw, (a, k)) for a in ("C1", "C1+G1") for k in keys) + 6 * 5
rnd = sum(med(d, (a, k)) + 3 for k in keys for a in ("E0", "G1", "C1", "C1+G1"))
steady = rnd * procs + 10
eq = 240 + 3 * 150            # ASSUMED: E0 proc (3 fp32 refs + 3 E0 + diffusers check + LPIPS load) 240 s; C1 / G1 / C1+G1 150 s each (model load + warm-cache compile + LPIPS)
prof = 2 * 150                # ASSUMED: nsys E0 / C1 (C1 loads the warm cache) 150 s each
start = 120                   # 3 x 30 s quiet confirmations + 1.5 GiB pinned self-test
tot = start + pre + eq + steady + prof
print(f"estimate PROCS={procs}: block start {start} s + prewarm {pre:.0f} s (S2 prewarm_final median secs of C1 / C1+G1 at the same keys) + equivalence {eq} s (assumed) + steady {steady:.0f} s "
      f"({rnd:.0f} s per round of 12 processes from S2 driver_log medians + 3 s/process, x {procs} + warm-up process) + Nsight {prof} s (assumed) = {tot:.0f} s = {tot/60:.0f} min")
print(f"  conservative x1.5 = {tot*1.5/60:.0f} min ({tot*1.5/3600:.2f} h)   [SD2.1 UNet cold compile may differ from SD1.5; the S2 numbers are SD1.5]")
print("  OK: x1.5 estimate <= 1 h" if tot * 1.5 <= 3600 else "  WARNING block estimated time (x1.5) > 1 h: reduce PROCS (>= 3) or set ALLOW_LONG=1 after telling the user")
PY
}

preconditions() {
  [ -f "$PREREG_SHA_FILE" ] || { echo "X10_STOP missing $PREREG_SHA_FILE (card must be final: write sha16 of the card there, 0444)"; return 4; }
  local sha; sha=$(cat "$PREREG_SHA_FILE")
  [ "$(sha256sum "$PREREG" | cut -c1-16)" = "$sha" ] && [ "$(stat -c %a "$PREREG")" = "444" ] || { echo "X10_STOP preregistration $PREREG missing / not 0444 / SHA16 != $sha"; return 4; }
  local f; for f in "${FROZEN[@]}" "$REVJSON"; do
    [ "$(stat -c %a "$f" 2>/dev/null)" = "444" ] || { echo "X10_STOP $f missing / not 0444"; return 4; }
  done
  case "$WARM" in *" "*) echo "X10_STOP warm root must not contain spaces"; return 4;; esac
  [ "$(basename "$WARM")" = "warm_s2" ] || { echo "X10_STOP warm root basename must be warm_s2 (run_matrix --formal)"; return 4; }
  local md; md=$(PYC -c "from scripts import x10_common as X; print(X.model_dir())") || return 4
  [ -f "$md/unet/diffusion_pytorch_model.fp16.safetensors" ] || { echo "X10_STOP SD-Turbo snapshot incomplete: $md"; return 4; }
  return 0
}

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN plan (nothing is executed on GPU): PROCS=$PROCS keys=$KEYS block_max=${BLOCK_MAX_S}s MEM_DROP_MB=$MEM_DROP_MB  [X10 SD-Turbo mechanism check]"
  echo " INFO prereg $PREREG mode=$(stat -c %a "$PREREG" 2>/dev/null) sha16=$(sha256sum "$PREREG" | cut -c1-16) ; expected (file $PREREG_SHA_FILE): $([ -f "$PREREG_SHA_FILE" ] && cat "$PREREG_SHA_FILE" || echo ABSENT) $([ -f "$PREREG_SHA_FILE" ] && { [ "$(sha256sum "$PREREG" | cut -c1-16)" = "$(cat "$PREREG_SHA_FILE")" ] && echo MATCH || echo MISMATCH; })"
  echo " INFO revision $REVJSON mode=$(stat -c %a "$REVJSON" 2>/dev/null) ; model dir $(PYC -c 'from scripts import x10_common as X; print(X.model_dir())')"
  for f in "${FROZEN[@]}"; do echo " INFO $f mode=$(stat -c %a "$f") sha16=$(sha256sum "$f" | cut -c1-16)"; done
  echo " INFO warm root $WARM (exists now: $([ -e "$WARM" ] && echo yes || echo no)) ; prewarm $PW ; equiv $EQ ; steady $ST ; profile $PF ; logs $LG ; verdict $VD"
  echo " steps: preconditions -> wait_quiet_mem 600 -> block_pinned_selftest -> block_mem_start -> per step: quick_quiet_mem||wait_quiet_mem 600, run_logged_proc, STEP line, stop on pinned/rc/skip/leak"
  i=0; for n in "${STEPS[@]}"; do i=$((i+1)); echo " [$i/${#STEPS[@]}] $n :: $(step_cmd "$n" | cut -c1-560)"; done
  estimate
  echo " CPU command checks (argument parsing only; no weights, no GPU):"
  for m in x10_bench x10_run_matrix x10_prewarm x10_equiv x10_profile_analyze; do echo "  $m --help: $(PYC -m scripts.$m --help 2>&1 | head -1 | cut -c1-160)"; done
  echo "  m12_verdict --help: $(PYC -m scripts.m12_verdict_S4X_X10 --help 2>&1 | head -1 | cut -c1-160)"
  echo " config.json / scheduler check (CPU, reads config files only):"
  PYC -m scripts.x10_common 2>&1 | sed 's/^/  /' | cut -c1-330
  exit 0
fi

# ---- preconditions (CPU only)
preconditions || exit 4
mkdir -p "$LG" "$PW" "$EQ" "$PF" "$VD"
EST_OUT=$(estimate)
echo "$EST_OUT"
if grep -q "WARNING block" <<< "$EST_OUT" && [ "$ALLOW_LONG" != "1" ]; then
  echo "X10_STOP estimated time (x1.5) > 1 h: reduce PROCS (>= 3) or set ALLOW_LONG=1 after telling the user"; exit 2
fi
if [ -e "$WARM" ] && ! ls "$PW"/*.csv >/dev/null 2>&1; then
  echo "X10_STOP warm root $WARM exists but $PW has no pre-warm record: a stale compile cache would fake the cold-compile time; move it aside by hand (never delete)"; exit 9
fi
mkdir -p "$(dirname "$WARM")"
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41
export X10_LOG_DIR="$LG"
exec > >(tee -a "$LG/block_X10.log") 2>&1
T0=$(date +%s)
echo "X10_START $(date +%FT%T) procs=$PROCS keys=$KEYS block_max=${BLOCK_MAX_S}s mem_drop_mb=$MEM_DROP_MB prereg_sha16=$(cat "$PREREG_SHA_FILE") revision=$(PYC -c "import json; print(json.load(open('$REVJSON'))['revision'])") $(for f in "${FROZEN[@]}"; do printf '%s=%s ' "$(basename "$f")" "$(sha256sum "$f" | cut -c1-16)"; done)"

finish_records() {
  local c miss=0 f
  for f in "$LG"/*.pinned_fallback_count.txt; do
    [ -f "$f" ] || continue
    c=$(cat "$f"); [ "$c" != "0" ] && { echo "PINNED_FALLBACK_FLAG $(basename "$f") count=$c"; miss=1; }
  done
  echo "RECORDS prewarm_csv=$(ls "$PW"/*.csv 2>/dev/null | wc -l) steady_csv=$(ls "$ST"/*.csv 2>/dev/null | wc -l) equiv_json=$(ls "$EQ"/*.json 2>/dev/null | wc -l) nsys_rep=$(ls "$PF"/*.nsys-rep 2>/dev/null | wc -l) pinned_records=$(ls "$LG"/*.pinned_fallback_count.txt 2>/dev/null | wc -l)"
  PYC - "$PW" "$ST" "$PF/bench" <<'PY' || miss=1
import glob, json, sys
n = bad = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*.meta.json")):
        m = json.load(open(f)); n += 1
        if m.get("status") != "ok":
            bad += 1; print("META_FLAG", f.rsplit("/", 1)[-1], "status=", m.get("status"), (m.get("error") or "")[:160])
print(f"META_CHECK n_meta={n} flagged={bad}")
sys.exit(1 if bad else 0)
PY
  [ $miss -eq 0 ] && echo "GATE_RECORDS_COMPLETE" || echo "GATE_RECORDS_INCOMPLETE (block marked incomplete)"
  mem_log block_end
}
stop_block() {   # $1 reason  $2 exit code
  echo "X10_STOP $1 elapsed=$(( $(date +%s) - T0 ))s (no retry; finished records stay; the verdict will read INVALID / unfinished)"
  finish_records
  exit "$2"
}
over_time() { [ $(( $(date +%s) - T0 )) -ge "$BLOCK_MAX_S" ]; }
any_pinned() { local f c; for f in "$LG"/*.pinned_fallback_count.txt; do [ -f "$f" ] || continue; c=$(cat "$f"); [ "$c" != "0" ] && { echo "$(basename "$f") count=$c"; return 0; }; done; return 1; }
step_done() {   # $1 name -> 0 when the step's product already exists and is ok (resume); stops the block on a non-ok leftover
  local n=$1 a
  case "$n" in
    prewarm) [ "$(ls "$PW"/*.csv 2>/dev/null | wc -l)" -ge 6 ] && PYC -c "
import glob, json, sys
sys.exit(0 if all(json.load(open(f)).get('status') == 'ok' for f in glob.glob('$PW/*.meta.json')) and len(glob.glob('$PW/*.meta.json')) >= 6 else 1)";;
    equiv_*) a=${n#equiv_}; [ -f "$EQ/$a.json" ] || return 1
             PYC -c "import json,sys; sys.exit(0 if json.load(open('$EQ/$a.json')).get('status') == 'ok' else 1)" || stop_block "existing non-ok equivalence result $EQ/$a.json: move it aside by hand, never delete" 8;;
    steady)  ls "$ST"/*.csv >/dev/null 2>&1 || return 1
             [ "$(ls "$ST"/*.csv | wc -l)" -ge $((4 * NKEYS * PROCS)) ] && PYC -c "
import glob, json, sys
ms = glob.glob('$ST/*.meta.json')
sys.exit(0 if len(ms) >= $((4 * NKEYS * PROCS)) and all(json.load(open(f)).get('status') == 'ok' for f in ms) else 1)" && return 0
             stop_block "steady dir $ST has an unfinished / non-ok earlier run: move it aside by hand (run_matrix refuses to overwrite)" 8;;
    profile_*) a=${n#profile_}; [ -f "$PF/${a}_512x512_b1_cfg0.nsys-rep" ];;
  esac
}

wait_quiet_mem 600 || { echo "X10_STOP gate_timeout_at_start"; exit 3; }
block_pinned_selftest || stop_block "pinned_selftest_failed -> tell the user (failure #18 host state)" 6
block_mem_start
i=0
for n in "${STEPS[@]}"; do
  i=$((i+1))
  if step_done "$n"; then echo "SKIP_DONE [$i/${#STEPS[@]}] $n"; continue; fi
  if over_time; then echo "SKIPPED_TIME_BUDGET [$i/${#STEPS[@]}] $n elapsed=$(( $(date +%s) - T0 ))s"; continue; fi
  quick_quiet_mem || wait_quiet_mem 600 || { echo "X10_STOP gate_timeout_before_$n"; finish_records; exit 3; }
  mem_log "before_$n"
  # shellcheck disable=SC2046
  run_logged_proc "$LG/step_$n" $(step_cmd "$n")
  rc=$?
  mem_log "after_$n"
  echo "STEP [$i/${#STEPS[@]}] $n rc=$rc elapsed=$(( $(date +%s) - T0 ))s"
  pc=$(cat "$LG/step_$n.pinned_fallback_count.txt" 2>/dev/null || echo "?")
  [ "$pc" = "0" ] || stop_block "pinned_fallback_count=$pc after $n (pinned-memory allocation failure: stop, no retry, tell the user)" 7
  w=$(any_pinned) && stop_block "pinned fallback in a child process: $w" 7
  [ "$rc" -eq 0 ] || stop_block "step $n rc=$rc" 7
  if [ "$n" = steady ] && grep -q '"skip"' "$ST/driver_log.jsonl" 2>/dev/null; then stop_block "run_matrix skipped process(es) (gate timeout): see $ST/driver_log.jsonl" 8; fi
  block_mem_guard || stop_block "leak_guard_after_$n" 5
done
finish_records
echo "X10_END $(date +%FT%T) elapsed=$(( $(date +%s) - T0 ))s  -> run x10_profile_analyze + m12_verdict_S4X_X10 (CPU), then tell the user: GPU0 returned (S4X-X10)"
