#!/bin/bash
# S4-X X9 exclusive window = N=10000 real-machine long replay, ADDING the eager (B0) and AOT (B1-AOT) arms plus two bridge processes (B1, Ours) on W2 seed 133
# (preregistration goals/S4X_X9_N1e4_eager_AOT_arms_prereg_v1.md sec 2 = the only basis; frozen prediction = the X5 one, results/S4X/x5/pred_X5.csv, 0444, NOT re-predicted).
# Built from window_S4X_X5.sh (B1 / Ours command, gates, leak guard, pinned self-test, SKIP_DONE, DRY_RUN) + window_S4P_P5d.sh (B0 and B1-AOT commands, AOT checks) + window_S4X_X8.sh
# (PART blocks, STOP rules, record completeness, X8-style START / STEP / META_CHECK / END lines).  All three templates are UNCHANGED.  Nothing signed-off is modified.
# ONE stream, read only: results/S4X/x5/workload/W2_N10000_seed133_B.jsonl (not copied).  raw / logs: results/S4X/x9/real/{raw,logs}.  Commands = the P5d / X5 commands of the arm, only --out differs
#   (B0, B1, Ours: scripts/serve_trace_S4P.py --cache-mode warm --warm-root warm_s4p_B --params-json params_ours_flat.json --exclusive --require-idle;
#    B1-AOT: scripts/serve_trace_S4P_v2.py --policy B1-AOT --cache-mode warm --aot-root $P24_CACHE/aot_p5c/pkg --aot-fallback-csv results/S4P/p5c/aot_fallback_shapes.csv (512x512_b3_nocfg is served eagerly) --exclusive --require-idle).
# 8 processes = B0 p0-p2, B1-AOT p0-p2, BRIDGE B1 p3 and BRIDGE Ours p3 (proc-id 3 on purpose: never confused with X5's p0-p2).  Blocks (card sec 2; never the same arm twice in a row):
#   PART=K1 : B0 p0 -> B1-AOT p0 -> B1 p3      PART=K2 : B1-AOT p1 -> B0 p1 -> Ours p3      PART=K3 : B0 p2 -> B1-AOT p2      PART=ALL = all 8 (> 2 h: refused unless ALLOW_LONG=1 after telling the user)
# Gates (code/docs/run_protocol_S3.md, gate_quiet_v2.sh): before every timed process MemAvailable >= 25 GB, swap < 50 %, load1 <= 16, GPU0 has no other process (wait <= 10 min else stop);
#   1.5 GiB pinned self-test once per block; leak guard after every process (MEM_DROP_MB default 12288); stderr of EVERY process saved, <tag>.pinned_fallback_count.txt, <tag>.hostmeta.json;
#   P24_PINNED_FALLBACK=1; taskset -c 0-15; TORCHINDUCTOR_COMPILE_THREADS=8; SKIP_DONE only for a process whose raw meta.json has status ok (a non-ok meta stops the block: move it aside by hand, never delete).
# STOP RULES (global rule, no retry): pinned self-test failure, pinned_fallback_count > 0 after a process, a process with rc != 0, a leak-guard hit, an incomplete pre-warm ledger / AOT package set
#   stop the block at once (records finished, exit code 5/6/7/8/9); finished processes stay in raw/ and are reported; the verdict then reads UNFINISHED / INVALID.
# Pre-warm ledger: scripts.check_prewarm_S4X_X5 --strict (the X5 stream IS this stream: C1 for B1, C1 + G1 for Ours) AND the AOT package preflight of serve_trace_S4P_v2 (package set + fallback list for B1-AOT); incomplete -> block not started.
# Block length: estimate (pred_X5.csv x measured slope + measured per-process overhead; nominal 15 s / process + 3 min start; x1.10) must be <= 2 h = 7200 s, else refused unless ALLOW_LONG=1.
# NVML_SAMPLE=1 (default 0) starts scripts/nvml_sampler_S4X_X2.py (core 31, 10 Hz, not in the timed path) per process -> <tag>.nvml.csv (device-level peak for the report).  Default OFF because the X5 arms
#   B1 / Ours (numerators of H1 / H2) were measured without a sampler; tell the main session before switching it on.
# WRITTEN IN S4-X X9 PREP, NOT RUN.  GPU0 exclusive per block (<= 2 h; announce to QSA papers 1/2 first, "host CPU load <= 16 during timing", register STATE/GPU_QUEUE.md; say "GPU0 returned (S4X-X9)" after).  From code/:
#     DRY_RUN=1 PART=K1 bash scripts/window_S4X_X9.sh    plan + estimate + CPU checks (serve_trace_S4P / _v2 --dry-run, AOT preflight, pre-warm ledger); no gate / GPU / output under results/.  K2 / K3 / ALL likewise.
#     PART=K1 bash scripts/window_S4X_X9.sh ; PART=K2 ... ; PART=K3 ...
# NEEDS before a real block: card 0444 + results/S4X/x9/PREREG_SHA16.txt (0444; = first 16 hex of sha256 of the card, written by the main session after the card is frozen), pred_X5.csv 0444, trace 0444, params_ours_flat.json 0444.
# AFTER THE BLOCKS (CPU): "$PROJ_PYTHON" -B -m scripts.h123_verdict_S4X_X9 , then tell the user "GPU0 returned (S4X-X9)".
set -u
cd "$(dirname "$0")/.." || exit 2
source ../ops/project_env.sh
export MEM_DROP_MB=${MEM_DROP_MB:-12288}
source scripts/gate_quiet_v2.sh
export TORCHINDUCTOR_COMPILE_THREADS=8 P24_PINNED_FALLBACK=1 LOAD_MAX=16

PART=${PART:-ALL}
DRY_RUN=${DRY_RUN:-0}
ALLOW_LONG=${ALLOW_LONG:-0}
NVML_SAMPLE=${NVML_SAMPLE:-0}
case "$PART" in ALL)      BLOCK_MAX_S=${BLOCK_MAX_S:-25000};;     # only reachable with ALLOW_LONG=1
                K1|K2|K3) BLOCK_MAX_S=${BLOCK_MAX_S:-5000};;      # start gate: a process may START only before this (a ~2100 s process then still ends inside 2 h = 7200 s)
                *) echo "PART must be ALL, K1, K2 or K3"; exit 2;; esac

X9D=../results/S4X/x9
R=$X9D/real
RAW=$R/raw; LOG=$R/logs
X5D=../results/S4X/x5
TR=$X5D/workload
STEM=W2_N10000_seed133_B
PREREG=../goals/S4X_X9_N1e4_eager_AOT_arms_prereg_v1.md
PREREG_SHA_FILE=$X9D/PREREG_SHA16.txt
[ "$DRY_RUN" = "1" ] && PREREG_SHA_FILE=${PREREG_SHA_FILE_OVERRIDE:-$PREREG_SHA_FILE}      # the override exists for the dry-run self-test only; a real block always reads the fixed path
PRED=$X5D/pred_X5.csv
WARM_ROOT=$P24_CACHE/warm_s4p_B
PJ_OURS=../results/S3/real_validation/params_ours_flat.json
AOT_ROOT=$P24_CACHE/aot_p5c/pkg
AOT_FALLBACK=../results/S4P/p5c/aot_fallback_shapes.csv     # relative on purpose: the commands are word-split (the project path has a space)
AOT_MANIFEST=$P24_CACHE/aot_p5c/pkg_manifest.json
P5D_RAW=../results/S4P/p5d/real/raw; P5D_LOG=../results/S4P/p5d/real/logs; P5D_PRED=../results/S4P/p5d/pred_P5d.csv
X5_RAW=$X5D/real/raw
PIN="taskset -c 0-15"
PYC() { CUDA_VISIBLE_DEVICES='' nice -n 10 "$PROJ_PYTHON" -B "$@"; }

# fixed design (card sec 2): "<workload> <arm> <proc-id>"; bridge = proc-id 3
PLAN_ALL=("W2 B0 0" "W2 B1-AOT 0" "W2 B1 3"   "W2 B1-AOT 1" "W2 B0 1" "W2 Ours 3"   "W2 B0 2" "W2 B1-AOT 2")
prev=""; for e in "${PLAN_ALL[@]}"; do read -r _w pol _p <<< "$e"; [ "$pol" = "$prev" ] && echo "WARNING adjacent duplicate arm $pol in the plan (design: no arm twice in a row)"; prev=$pol; done
filter_plan() {   # $1 PART -> entries
  case "$1" in ALL) printf '%s\n' "${PLAN_ALL[@]}";; K1) printf '%s\n' "${PLAN_ALL[@]:0:3}";; K2) printf '%s\n' "${PLAN_ALL[@]:3:3}";; K3) printf '%s\n' "${PLAN_ALL[@]:6:2}";; esac
}
mapfile -t PLAN < <(filter_plan "$PART")

serve_cmd() {   # $1 w  $2 pol  $3 proc  [$4 out dir] -> command line (the P5d / X5 command of the arm; only --out differs)
  local pol=$2 p=$3 out=${4:-$RAW}
  if [ "$pol" = "B1-AOT" ]; then   # P5d: weight-free AOT packages, own driver; no warm root / params
    echo "$PIN $PROJ_PYTHON -B -m scripts.serve_trace_S4P_v2 --trace $TR/$STEM.jsonl --policy B1-AOT --cache-mode warm --aot-root $AOT_ROOT --aot-fallback-csv $AOT_FALLBACK --out $out --proc-id $p --exclusive --require-idle"
  else   # B0 / B1 / Ours: X5 / P5d block-B command
    echo "$PIN $PROJ_PYTHON -B -m scripts.serve_trace_S4P --trace $TR/$STEM.jsonl --policy $pol --cache-mode warm --warm-root $WARM_ROOT --out $out --proc-id $p --exclusive --require-idle --params-json $PJ_OURS"
  fi
}

estimate() {   # $1 label $2.. plan entries.  CPU: pred_X5.csv x measured slope + measured per-process overhead (P5d / X5 wall clocks)
  local lab=$1; shift
  PYC - "$PRED" "$lab" "$STEM" "$P5D_PRED" "$P5D_RAW" "$P5D_LOG" "$X5_RAW" "$@" <<'PY'
import glob, json, re, sys
import pandas as pd
pred = pd.read_csv(sys.argv[1]); lab = sys.argv[2]; stem = sys.argv[3]; p5pred = pd.read_csv(sys.argv[4]); raw = sys.argv[5]; logd = sys.argv[6]; x5raw = sys.argv[7]; plan = [a.split() for a in sys.argv[8:]]
pred = pred[pred.stream == stem]
# --- P5d measured wall clock: per process wall = difference of consecutive STEP elapsed (first STEP of a block also contains the block start); same method as window_S4X_X5.sh
svc = {}
for f in glob.glob(raw + "/*.meta.json"):
    m = json.load(open(f)); svc[(m["policy"], m["trace"].rsplit("/", 1)[-1][:-6], int(f.rsplit("_p", 1)[1].split(".")[0]))] = float(m["service_s"])
ovh, start = [], []
for lf in sorted(glob.glob(logd + "/block_P5d_*.log")):
    last = None; first = True
    for l in open(lf, errors="replace"):
        if l.startswith("S4PD_START"):
            last, first = 0.0, True; continue
        m = re.match(r"^STEP \[\d+/\d+\] (\S+)_(W\d)_warm_p(\d) rc=0 elapsed=(\d+)s", l)
        if not m or last is None: continue
        pol, w, p, el = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        s = svc.get((pol, f"{w}_N1000_seed130_B", p)); wall = el - last; last = el
        if s is None: continue
        if first: start.append(wall - s - (sum(ovh) / len(ovh) if ovh else 4.5)); first = False
        else: ovh.append(wall - s)
ov = sum(ovh) / len(ovh) if ovh else 9.5; st = max(sum(start) / len(start), 0.0) if start else 56.0
slope, src = {}, {}
for pol in ("B0", "B1-AOT"):   # slope = real service_s / predicted pred_total_s on the P5d streams (N=1000)
    r = []
    for (pl, s_, p), s in svc.items():
        q = p5pred[(p5pred.stream == s_) & (p5pred.policy == pl)]
        if pl == pol and len(q): r.append(s / float(q.pred_total_s.iloc[0]))
    slope[pol] = sum(r) / len(r) if r else 1.0; src[pol] = f"P5d N=1000 mean real/pred, n={len(r)}"
x5 = {}
for f in glob.glob(x5raw + "/*.meta.json"):
    m = json.load(open(f))
    if m.get("status") == "ok": x5.setdefault(m["policy"], []).append(float(m["service_s"]))
for pol in ("B1", "Ours"):   # slope = measured X5 mean / predicted (same stream, N=10000)
    q = pred[pred.policy == pol]
    if pol in x5 and len(q): slope[pol] = (sum(x5[pol]) / len(x5[pol])) / float(q.pred_total_s.iloc[0]); src[pol] = f"X5 measured mean/pred, n={len(x5[pol])}"
    else: slope[pol] = 1.0; src[pol] = "no X5 data: 1.0"
def P(pol):
    r = pred[pred.policy == pol]; assert len(r) == 1, pol
    return float(r.pred_total_s.iloc[0])
tot = 0.0; per = []
for w, pol, p in plan:
    t = P(pol) * slope[pol] + ov; per.append(f"{pol} p{p} {t/60:.1f} min"); tot += t
est = tot + st; nom = sum(P(pol) for _, pol, _ in plan) + 15 * len(plan) + 180   # nominal = pred + 15 s/process + 3 min block start (P5d report rule)
print(f"estimate {lab}: {len(plan)} processes [{'; '.join(per)}]")
print(f"  slopes {{{', '.join(f'{k} {v:.4f} ({src[k]})' for k, v in slope.items())}}} + {ov:.1f} s/process overhead (P5d, n={len(ovh)}) ; block start {st:.0f} s (n={len(start)}) -> {est/60:.0f} min ({est/3600:.2f} h)")
print(f"  nominal (pred + 15 s/process + 3 min start): {nom/60:.0f} min ({nom/3600:.2f} h) ; x1.10 conservative: {nom*1.1/60:.0f} min ({nom*1.1/3600:.2f} h)")
worst = max(est, nom, nom * 1.1)
if worst > 7200:
    print(f"  WARNING block '{lab}' estimated time {worst/3600:.2f} h > 2 h exclusive limit: use PART=K1 / K2 / K3 (card sec 2), or set ALLOW_LONG=1 after telling the user")
else:
    print(f"  OK: below 2 h in all of central / nominal / x1.10 (worst {worst/60:.0f} min = {worst/3600:.2f} h)")
PY
}

aot_checks() {   # CPU only: B1-AOT preconditions (prints AOT_CHECK lines; returns 1 on the first failure)
  grep -q '"B1-AOT"' scripts/serve_trace_S4P_v2.py || { echo "X9_STOP serve_trace_S4P_v2.py does not know B1-AOT"; return 1; }
  [ -d "$AOT_ROOT" ] || { echo "X9_STOP AOT package root $AOT_ROOT missing (scripts/run_aot_prebuild_S4P_P5c.sh)"; return 1; }
  [ -f "$AOT_FALLBACK" ] && [ "$(stat -c %a "$AOT_FALLBACK")" = "444" ] || { echo "X9_STOP fallback list $AOT_FALLBACK missing or not 0444"; return 1; }
  [ -f "$AOT_MANIFEST" ] || { echo "X9_STOP package manifest $AOT_MANIFEST missing"; return 1; }
  echo "AOT_CHECK fallback list $(basename "$AOT_FALLBACK") sha16=$(sha256sum "$AOT_FALLBACK" | cut -c1-16): $(( $(wc -l < "$AOT_FALLBACK") - 1 )) shape(s) listed -> served eagerly; manifest sha16=$(sha256sum "$AOT_MANIFEST" | cut -c1-16)"
  local out rc
  out=$(PYC -m scripts.serve_trace_S4P_v2 --preflight --trace "$TR/$STEM.jsonl" --policy B1-AOT --aot-root "$AOT_ROOT" --aot-fallback-csv "$AOT_FALLBACK"); rc=$?
  echo "AOT_CHECK preflight W2 rc=$rc $(echo "$out" | tail -1 | cut -c1-300)"
  [ $rc -eq 0 ] || { echo "X9_STOP AOT packages incomplete for the X9 stream (shape without a package and not in the fallback list)"; return 1; }
  return 0
}

cmdcheck() {   # CPU: the exact serve command line of arm $1 proc $2 run with --dry-run (argument parsing + policy build + mock replay; no torch / GPU); output into the scratch dir under tmp/, never results/
  local pol=$1 p=$2 outd=${DRYRUN_DIR:-../tmp/x9_cmdcheck} c   # relative, no space (the command line is word-split)
  mkdir -p "$outd"
  c=$(serve_cmd W2 "$pol" "$p" "$outd")
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES='' nice -n 10 $c --dry-run < /dev/null 2>&1 | cut -c1-330
}

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN plan (nothing is executed on GPU): PART=$PART stream=$STEM block_max=${BLOCK_MAX_S}s MEM_DROP_MB=$MEM_DROP_MB NVML_SAMPLE=$NVML_SAMPLE  [X9]"
  echo " INFO prereg $PREREG mode=$(stat -c %a "$PREREG" 2>/dev/null) sha16=$(sha256sum "$PREREG" | cut -c1-16) ; expected (file $PREREG_SHA_FILE): $([ -f "$PREREG_SHA_FILE" ] && cat "$PREREG_SHA_FILE" || echo ABSENT) $([ -f "$PREREG_SHA_FILE" ] && { [ "$(sha256sum "$PREREG" | cut -c1-16)" = "$(cat "$PREREG_SHA_FILE")" ] && echo MATCH || echo MISMATCH; })"
  echo " INFO frozen prediction $PRED $([ -f "$PRED" ] && echo "present mode=$(stat -c %a "$PRED") sha16=$(sha256sum "$PRED" | cut -c1-16)" || echo "ABSENT"); warm root $WARM_ROOT $([ -d "$WARM_ROOT/C1" ] && [ -d "$WARM_ROOT/C1_G1" ] && echo "present (C1, C1_G1)" || echo "ABSENT")"
  echo " INFO trace $TR/$STEM.jsonl mode=$(stat -c %a "$TR/$STEM.jsonl" 2>/dev/null) sha16=$(sha256sum "$TR/$STEM.jsonl" | cut -c1-16) ; params $PJ_OURS mode=$(stat -c %a "$PJ_OURS" 2>/dev/null) ; raw $RAW ; logs $LOG ; verdict dir $X9D/verdict"
  echo " INFO AOT root $AOT_ROOT ; fallback $AOT_FALLBACK ; manifest $AOT_MANIFEST ; drivers scripts/serve_trace_S4P.py sha16=$(sha256sum scripts/serve_trace_S4P.py | cut -c1-16), scripts/serve_trace_S4P_v2.py sha16=$(sha256sum scripts/serve_trace_S4P_v2.py | cut -c1-16)"
  aot_checks || echo " WARN AOT checks failed (the real block would stop: X9_STOP)"
  PYC -m scripts.check_prewarm_S4X_X5 --strict > /dev/null && echo " INFO pre-warm ledger check: stream shapes fully covered for C1 and C1+G1" || echo " WARN pre-warm ledger check: shapes missing (the real block would stop: X9_STOP)"
  echo " steps: preconditions -> estimate (> 2 h refused unless ALLOW_LONG=1) -> wait_quiet_mem 600 -> block_pinned_selftest -> block_mem_start -> per process: quick_quiet_mem||wait_quiet_mem 600, mem_log, run_logged_proc, mem_log, STEP, stop on pinned>0 / rc!=0, block_mem_guard"
  estimate "PART=$PART" "${PLAN[@]}"
  if [ "$PART" = "ALL" ]; then
    for q in K1 K2 K3; do mapfile -t SUB < <(filter_plan "$q"); estimate "block $q" "${SUB[@]}"; done
  fi
  i=0; for e in "${PLAN[@]}"; do i=$((i+1)); read -r w pol p <<< "$e"; echo " [$i] $w $pol p$p :: $(serve_cmd $w $pol $p | cut -c1-520)"; done
  echo " total processes: ${#PLAN[@]}  (B0: $(printf '%s\n' "${PLAN[@]}" | awk '$2=="B0"' | wc -l), B1-AOT: $(printf '%s\n' "${PLAN[@]}" | awk '$2=="B1-AOT"' | wc -l), bridge B1: $(printf '%s\n' "${PLAN[@]}" | awk '$2=="B1"' | wc -l), bridge Ours: $(printf '%s\n' "${PLAN[@]}" | awk '$2=="Ours"' | wc -l))"
  echo " command check (serve_trace_S4P / _v2 --dry-run, CPU, same arguments, scratch out dir ${DRYRUN_DIR:-../tmp/x9_cmdcheck}; first process of each arm in the plan):"
  while read -r pol p; do echo "  $pol p$p: $(cmdcheck "$pol" "$p" | tail -1)"; done < <(printf '%s\n' "${PLAN[@]}" | awk '!s[$2]++{print $2, $3}')
  exit 0
fi

[ "$DRY_RUN" = "1" ] && { echo "X9_STOP the DRY_RUN block above aborted on an error; nothing was run"; exit 4; }   # never fall through into the real run
# ---- preconditions (CPU only)
[ -f "$PREREG_SHA_FILE" ] && [ "$(stat -c %a "$PREREG_SHA_FILE")" = "444" ] || { echo "X9_STOP missing or not 0444: $PREREG_SHA_FILE (card must be final: sha16 of the card there, 0444)"; exit 4; }
PREREG_SHA16=$(cat "$PREREG_SHA_FILE")
[ "$(sha256sum "$PREREG" | cut -c1-16)" = "$PREREG_SHA16" ] && [ "$(stat -c %a "$PREREG")" = "444" ] || { echo "X9_STOP preregistration $PREREG missing / not 0444 / SHA16 != $PREREG_SHA16"; exit 4; }
[ -f "$PRED" ] || { echo "X9_STOP missing frozen prediction $PRED"; exit 4; }
[ ! -w "$PRED" ] && [ "$(stat -c %a "$PRED")" = "444" ] || { echo "X9_STOP prediction $PRED must be 0444"; exit 4; }
[ -d "$WARM_ROOT/C1" ] && [ -d "$WARM_ROOT/C1_G1" ] || { echo "X9_STOP warm cache $WARM_ROOT missing (C1, C1_G1)"; exit 4; }
PYC -m scripts.check_prewarm_S4X_X5 --strict > /dev/null || { echo "X9_STOP pre-warm ledger incomplete for the X9 stream (scripts/check_prewarm_S4X_X5.py)"; exit 4; }
aot_checks || exit 4
[ "$(stat -c %a "$TR/$STEM.jsonl")" = "444" ] || { echo "X9_STOP trace $STEM must be 0444"; exit 4; }
[ "$(stat -c %a "$PJ_OURS")" = "444" ] || { echo "X9_STOP $PJ_OURS must be 0444"; exit 4; }
mkdir -p "$LOG" "$RAW"
EST_OUT=$(estimate "PART=$PART" "${PLAN[@]}")
echo "$EST_OUT"
if grep -q "WARNING block" <<< "$EST_OUT" && [ "$ALLOW_LONG" != "1" ]; then
  echo "X9_STOP estimated time > 2 h: use PART=K1 / K2 / K3, or set ALLOW_LONG=1 after telling the user"; exit 2
fi
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41
exec > >(tee -a "$LOG/block_X9_${PART}.log") 2>&1
T0=$(date +%s)
echo "X9_START $(date +%FT%T) part=$PART block_max=${BLOCK_MAX_S}s processes=${#PLAN[@]} mem_drop_mb=$MEM_DROP_MB nvml_sample=$NVML_SAMPLE warm_root=$WARM_ROOT pred_sha16=$(sha256sum "$PRED" | cut -c1-16) prereg_sha16=$PREREG_SHA16 serve_trace_S4P_sha16=$(sha256sum scripts/serve_trace_S4P.py | cut -c1-16) serve_trace_S4P_v2_sha16=$(sha256sum scripts/serve_trace_S4P_v2.py | cut -c1-16) trace_sha16=$(sha256sum "$TR/$STEM.jsonl" | cut -c1-16) aot_root=$AOT_ROOT aot_fallback_sha16=$(sha256sum "$AOT_FALLBACK" | cut -c1-16) aot_manifest_sha16=$(sha256sum "$AOT_MANIFEST" | cut -c1-16)"

finish_records() {   # record completeness + META_CHECK (also on a stop)
  local miss=0 f b c
  for f in "$LOG"/*.hostmeta.json; do
    [ -f "$f" ] || continue
    b=${f%.hostmeta.json}
    [ -f "$b.stderr.txt" ] && [ -f "$b.pinned_fallback_count.txt" ] || { echo "RECORD_INCOMPLETE $b"; miss=1; }
    c=$(cat "$b.pinned_fallback_count.txt" 2>/dev/null || echo "?")
    [ "$c" != "0" ] && echo "PINNED_FALLBACK_FLAG $(basename "$b") count=$c"
  done
  PYC - "$RAW" <<'PY' || miss=1
import glob, json, sys
n = bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.meta.json")):
    m = json.load(open(f)); n += 1
    if "pinned_fallback_count" not in m or m.get("status") != "ok" or m.get("cache_mode") != "warm":
        bad += 1; print("META_FLAG", f.rsplit("/", 1)[-1], "status=", m.get("status"), "cache_mode=", m.get("cache_mode"), "pinned_fallback_count=", m.get("pinned_fallback_count", "MISSING"))
    if m.get("policy") == "B1-AOT":
        ao = m.get("aot") or {}
        if not ao or ao.get("rt_cache_files_after") != 0 or ao.get("dynamo_compile_events") != 0 or ao.get("pkg_manifest_mismatch") not in ([], None) or ao.get("pkg_manifest") is None:
            bad += 1; print("META_FLAG", f.rsplit("/", 1)[-1], "AOT arm: no aot block / Inductor cache files created (rt_cache_files_after != 0) / Dynamo-Inductor compile events != 0 / package SHA differs from the manifest / no manifest")
        if m.get("aot_fallback_requests"):
            print("AOT_FALLBACK_INFO", f.rsplit("/", 1)[-1], "fallback shapes", m.get("aot_fallback_shapes"), "eager requests", m.get("aot_fallback_requests"))
print(f"META_CHECK n_meta={n} flagged={bad}")
sys.exit(1 if bad else 0)
PY
  [ $miss -eq 0 ] && echo "GATE_RECORDS_COMPLETE" || echo "GATE_RECORDS_INCOMPLETE (block marked incomplete)"
  mem_log block_end
}
stop_block() {   # $1 reason  $2 exit code
  echo "X9_STOP $1 elapsed=$(( $(date +%s) - T0 ))s (no retry; finished processes stay in $RAW; the verdict will read UNFINISHED / INVALID)"
  finish_records
  exit "$2"
}
NVML_PID=""
nvml_start() {   # $1 = <logs>/<tag>.nvml (opt-in, default off)
  [ "$NVML_SAMPLE" = "1" ] || return 0
  rm -f "$1.stop"
  taskset -c 31 "$PROJ_PYTHON" -B scripts/nvml_sampler_S4X_X2.py --out "$1.csv" --hz 10 --stop-file "$1.stop" --max-s 7000 > "$1.sampler.log" 2>&1 &
  NVML_PID=$!
}
nvml_stop() {    # $1 = <logs>/<tag>.nvml
  [ -n "$NVML_PID" ] || return 0
  touch "$1.stop"; wait "$NVML_PID" 2>/dev/null; NVML_PID=""
}

wait_quiet_mem 600 || { echo "X9_STOP gate_timeout_at_start"; exit 3; }
block_pinned_selftest || stop_block "pinned_selftest_failed -> tell the user (failure #18 host state)" 6
block_mem_start
over_time() { [ $(( $(date +%s) - T0 )) -ge "$BLOCK_MAX_S" ]; }
i=0
for e in "${PLAN[@]}"; do
  i=$((i+1)); read -r w pol p <<< "$e"
  tag="${pol}_${w}_warm_p${p}"
  mf="$RAW/${pol}_${STEM}_p${p}.meta.json"
  if [ -f "$mf" ]; then
    if PYC - "$mf" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "ok" else 1)
PY
    then echo "SKIP_DONE [$i/${#PLAN[@]}] $tag"; continue
    else stop_block "existing non-ok meta for $tag ($mf): move it (and its csv/stderr) aside by hand, never delete; the server refuses to overwrite" 8; fi
  fi
  if over_time; then echo "SKIPPED_TIME_BUDGET [$i/${#PLAN[@]}] $tag elapsed=$(( $(date +%s) - T0 ))s"; continue; fi
  quick_quiet_mem || wait_quiet_mem 600 || { echo "X9_STOP gate_timeout_before_$tag"; finish_records; exit 3; }
  mem_log "before_$tag"
  nvml_start "$LOG/$tag.nvml"
  # shellcheck disable=SC2046
  run_logged_proc "$LOG/$tag" $(serve_cmd $w $pol $p)
  rc=$?
  nvml_stop "$LOG/$tag.nvml"
  mem_log "after_$tag"
  echo "STEP [$i/${#PLAN[@]}] $tag rc=$rc elapsed=$(( $(date +%s) - T0 ))s"
  pc=$(cat "$LOG/$tag.pinned_fallback_count.txt" 2>/dev/null || echo "?")
  [ "$pc" = "0" ] || stop_block "pinned_fallback_count=$pc after $tag (pinned-memory allocation failure: stop, no retry, tell the user)" 7
  [ "$rc" -eq 0 ] || stop_block "process $tag rc=$rc" 7
  block_mem_guard || stop_block "leak_guard_after_$tag" 5
done
finish_records
echo "X9_END $(date +%FT%T) part=$PART elapsed=$(( $(date +%s) - T0 ))s  -> tell the user: GPU0 returned (S4X-X9)"
