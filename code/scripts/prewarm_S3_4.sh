#!/bin/bash
# S3 3.4 pre-warm (DEVELOPMENT mode, not a timing block; GPU0 shared-use rules apply: nothing else of mine on GPU0, wait if a classmate process is there).
# WRITTEN IN S3 PREP, NOT RUN. The main session schedules GPU0 and decides BEST first. Run from code/:
#     BEST=<policy> bash scripts/prewarm_S3_4.sh            copy + pre-warm what is missing
#     DRY_RUN=1 BEST=B5 bash scripts/prewarm_S3_4.sh        CPU only: copy plan + the list of missing (arm, shape) pairs + the pre-warm commands, nothing written except the list
#
# WHAT: the 3 real-validation traces (results/S3/workload/real_validation/{W2,W3,W5}_N300_seed100_B.jsonl) use shapes that the S2 pre-warm (warm_s2) never compiled
#   (B=3 shapes at 512x768 / 768x512 / 768x768 are interpolation rows of the cost table, and C1 768x768 B4 CFG/no-CFG FAILED in the S2 pre-warm, failure #18).
#   scripts/check_prewarm_S3_4.py lists exactly which (arm, shape) pairs are missing (ledger = S2 prewarm log + S3 prewarm log + S3-A C2/C5 sequences);
#   this script pre-warms them with the S2 driver scripts/prewarm_s2.py (one bench process per arm x shape, GPU0 must be idle, pinned to cores 0-15).
# WHERE: a COPY of warm_s2 -> $P24_CACHE/warm_s3b (dirs C1, C1_G1, C5 = 5.6 GB; E0/G1 are cache-free).  warm_s2 itself is NEVER written.
#   The S3-B block (window_S3_B.sh, WARM_ROOT default = warm_s3b) and nothing else uses warm_s3b.  Make the copy AFTER S3-A has run (S3-A writes C2/C5 into warm_s2);
#   the copy time is stored in warm_s3b/PREWARM_COPY.json and the checker uses it.  Ledger of caches: owner = P24 S3 3.4 / purpose = S3-B formal runs /
#   rebuildable from warm_s2 + this script / delete after S3 sign-off.
# OUTPUT: results/S3/prewarm_S3_4/{missing_<BEST>.json, prewarm_log.jsonl, per-process csv/meta}, log logs/S3_3.4_prewarm.log.
# DURATION: checker prints an upper bound (S2 per-process secs of the nearest shape; each process = model load + compile of the components NOT already in the cache,
#   so real time is shorter). Typical BEST (B1,B3',B4,B5,B6,B6'): 14 missing pairs, upper bound ~10 min. BEST=B2 (C5 = dynamic=True, shape-generic code, most shapes
#   expected to be cache hits but not guaranteed): 42 pairs, upper bound ~99 min -> decide with the main session before running.
# REDO=1 re-runs entries whose csv already exists in the output dir (needed after a FAILED entry; the key list only holds missing pairs).
# Order: C1 first (it contains the two pairs that failed in S2: 768x768 B4; run them while the host is quiet, wait_quiet_mem gate before each arm group).
set -u
cd "$(dirname "$0")/.." || exit 2
source ../ops/project_env.sh
source scripts/gate_quiet_v2.sh
export TORCHINDUCTOR_COMPILE_THREADS=8 P24_PINNED_FALLBACK=1
export LOAD_MAX=16
DRY_RUN=${DRY_RUN:-0}
BEST=${BEST:-}
case "$BEST" in B3p|"B3'"|B3prime) BEST=B3p;; B6p|"B6'"|B6prime) BEST=B6p;; esac
[ -n "$BEST" ] || { echo "BEST is required (B1 B2 B3p B4 B5 B6 B6p): the arm set to pre-warm depends on it"; exit 2; }
S2W=$P24_CACHE/warm_s2
S3W=$P24_CACHE/warm_s3b
OUT=../results/S3/prewarm_S3_4
MISS=$OUT/missing_${BEST}.json
LOGF=../logs/S3_3.4_prewarm.log
mkdir -p "$OUT"

echo "== 1. CPU check: what is missing (BEST=$BEST)"
CUDA_VISIBLE_DEVICES='' nice -n 10 "$PROJ_PYTHON" -B -m scripts.check_prewarm_S3_4 --best-online "$BEST" --out "$MISS.new"
[ -f "$MISS.new" ] || { echo "checker failed"; exit 2; }
mv -f "$MISS.new" "$MISS"
# per-arm key lists
ARMLINES=$(CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B - "$MISS" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1])); g = collections.OrderedDict()
for m in sorted(d["missing"], key=lambda m: (m["arm"] != "C1", m["arm"], -m["est_secs_upper"])):
    g.setdefault(m["arm"], []).append(m["key"])
for arm, ks in g.items():
    print(arm, ",".join(ks))
PY
)
NMISS=$(CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B -c "import json,sys;print(json.load(open(sys.argv[1]))['n_missing'])" "$MISS")

echo "== 2. copy plan: $S2W/{C1,C1_G1,C5} -> $S3W   (cp -a, only if $S3W does not exist yet; E0 and G1 are cache-free; C2/C3/C6 are not used by serve_trace)"
echo "== 3. pre-warm commands (one per arm; keys per arm below); missing pairs: $NMISS"
while read -r arm keys; do
  [ -n "$arm" ] || continue
  echo "   $arm: taskset-pinned scripts.prewarm_s2 --arms '$arm' --keys $keys --out $OUT --warm-root $S3W --max-wait-s 900"
done <<< "$ARMLINES"
if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN: nothing copied, no GPU used (list written to $MISS)"; exit 0; fi

# ---------------------------------------------------------------- real run
exec > >(tee -a "$LOGF") 2>&1
echo "PREWARM_S3_4_START $(date +%FT%T) best=$BEST missing=$NMISS"
if [ ! -d "$S3W" ]; then
  [ -d ../results/S3/cost_addendum/inproc_seq ] || echo "WARNING: no S3-A results found (results/S3/cost_addendum/inproc_seq); the copy is made without them (affects BEST=B2 only)"
  mkdir -p "$S3W"
  for d in C1 C1_G1 C5; do nice -n 10 cp -a "$S2W/$d" "$S3W/$d" || { echo "copy of $d failed"; exit 2; }; done
  mkdir -p "$S3W/E0" "$S3W/G1"
  "$PROJ_PYTHON" -B -c "import json,time,sys;json.dump({'copied_at':time.time(),'source':sys.argv[1]},open(sys.argv[2],'w'))" "$S2W" "$S3W/PREWARM_COPY.json"
  echo "copied warm_s2 -> $S3W ($(du -sh "$S3W" | cut -f1))"
else
  echo "$S3W exists: reused (not overwritten)"
fi
# the checker must now count only the copy marker's S3-A entries; re-list (CPU) and pre-warm
CUDA_VISIBLE_DEVICES='' nice -n 10 "$PROJ_PYTHON" -B -m scripts.check_prewarm_S3_4 --best-online "$BEST" --out "$MISS.new" > /dev/null
mv -f "$MISS.new" "$MISS"
ARMLINES=$(CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B - "$MISS" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1])); g = collections.OrderedDict()
for m in sorted(d["missing"], key=lambda m: (m["arm"] != "C1", m["arm"], -m["est_secs_upper"])):
    g.setdefault(m["arm"], []).append(m["key"])
for arm, ks in g.items():
    print(arm, ",".join(ks))
PY
)
export CUDA_VISIBLE_DEVICES=GPU-722679a6-175b-cb56-b764-e8124d262c41
while read -r arm keys; do
  [ -n "$arm" ] || continue
  wait_quiet_mem 600 || { echo "PREWARM_STOP gate_timeout_before_$arm"; exit 3; }
  echo "--- pre-warm $arm : $keys  $(date +%T)"
  "$PROJ_PYTHON" -B -m scripts.prewarm_s2 --arms "$arm" --keys "$keys" --out "$OUT" --warm-root "$S3W" --max-wait-s 900 ${REDO:+--redo}
  echo "--- done $arm rc=$? $(date +%T)"
done <<< "$ARMLINES"
echo "== 4. re-check (CPU, strict)"
CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B -m scripts.check_prewarm_S3_4 --best-online "$BEST" --strict && echo "PREWARM_COMPLETE (nothing missing)" || echo "PREWARM_INCOMPLETE: see the MISSING lines above (failed / skipped entries: re-run this script; finished entries are not repeated)"
echo "PREWARM_S3_4_END $(date +%FT%T)"
