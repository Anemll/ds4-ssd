#!/usr/bin/env bash
# M1 Max ANE knob OFAT sweep (streaming). At a fixed operating point
# (PROMPT/CHUNK/MIN_REFS) run the GPU/ALU baseline once, a single-cluster ANE
# "base", then one run per knob variant (one factor changed) to see if any
# concurrency / tail-pack / overlap / kernel-variant / batching knob flips ANE
# past the GPU baseline. Hot single runs -- trends robust, exact % +-a few.
#
#   PROMPT=8k CHUNK=16384 MIN_REFS=512 DS4_SLOTS=96 \
#     ./moe-batch-bench/sweep_ane_knobs_M1MAX64.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROMPT="${PROMPT:-8k}"
CHUNK="${CHUNK:-16384}"
MIN_REFS="${MIN_REFS:-512}"
DS4_SLOTS="${DS4_SLOTS:-64}"
PROMPT_DIR="${PROMPT_DIR:-$ROOT/tests/test-vectors/prompts/coding}"
PROMPT_FILE="${PROMPT_FILE:-$PROMPT_DIR/coding_${PROMPT}.txt}"
[[ -f "$PROMPT_FILE" ]] || { echo "no prompt: $PROMPT_FILE" >&2; exit 2; }

case "$PROMPT" in
  1k) CTX=3072;; 4k) CTX=6144;; 6k) CTX=8192;; 8k) CTX=10000;;
  10k) CTX=12000;; 12k) CTX=14336;; 14k) CTX=17000;; 16k) CTX=20000;;
  *) CTX="${CTX:-20000}";;
esac

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/moe-batch-bench/profile_runs/ane_knobs_m1max64_${PROMPT}_$SWEEP_ID"
mkdir -p "$OUT"
RESULTS="$OUT/results.csv"
echo "variant,prefill_tps,env" > "$RESULTS"

# Single-cluster base, applied to every ANE variant (M1 Max: DUAL=0 THREADS=1,
# shared/oproj off). Per-variant env is appended and overrides these.
COMMON_ANE="DS4_FLASH_MOE_ANE_DUAL=0 DS4_FLASH_MOE_ANE_THREADS=1 \
DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0 \
DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=$MIN_REFS DS4_FLASH_MOE_ANE_MIN_REFS=$MIN_REFS"

# variant_name | extra env (one factor vs base)
VARIANTS=(
  "base|"
  "threads2|DS4_FLASH_MOE_ANE_THREADS=2"
  "multiactive0|DS4_FLASH_MOE_ANE_MULTI_ACTIVE=0"
  "outq8|DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=8"
  "outq2|DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=2"
  "scalarpack|DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=1 DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=0"
  "preflush1|DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=1"
  "preflush16|DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=16"
  "sched_off|DS4_FLASH_MOE_OVERLAP_SCHEDULER=0"
  "overlap_off|DS4_FLASH_MOE_OVERLAP_PREFILL=0 DS4_FLASH_MOE_OVERLAP_SCHEDULER=0"
  "pipeline_off|DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0"
  "defer_dequant|DS4_FLASH_MOE_ANE_DEFER_DEQUANT_COMMIT=1"
  "fused_dequant|DS4_FLASH_MOE_ANE_FUSED_DEQUANT=1"
  "full_fused|DS4_FLASH_MOE_ANE_I8I8_FULL_FUSED_PREFILL=1 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0"
  "fp16x_i8w|DS4_FLASH_MOE_ANE_FP16X_INT8W=1 DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0"
  "maxrefs512|DS4_FLASH_MOE_ANE_MAX_REFS=512"
  "maxrefs128|DS4_FLASH_MOE_ANE_MAX_REFS=128"
  "batches512|DS4_FLASH_MOE_ANE_BATCHES=512"
)

extract_tps() { awk '/prefill:/ { v=$3; sub(/,/,"",v) } END { print v }' "$1"; }

run_one() {
  local name="$1" script="$2"; shift 2
  local rn="knob_${name}"
  local sum="$OUT/$rn.summary.txt"
  echo "==> $name"
  if env DS4_LOG_DIR="$OUT" DS4_RUN_NAME="$rn" DS4_PROMPT_FILE="$PROMPT_FILE" \
      DS4_CTX="$CTX" DS4_PREFILL_CHUNK="$CHUNK" DS4_SLOTS="$DS4_SLOTS" \
      "$@" "$script" >"$OUT/$rn.driver.log" 2>&1; then
    local tps; tps="$(extract_tps "$sum" 2>/dev/null)"
    printf '%s,%s,"%s"\n' "$name" "${tps:-NA}" "$*" >> "$RESULTS"
    echo "    $name -> ${tps:-NA} t/s"
  else
    printf '%s,FAIL,"%s"\n' "$name" "$*" >> "$RESULTS"
    echo "    $name -> FAIL (see $OUT/$rn.driver.log)"
  fi
}

echo "operating point: prompt=$PROMPT ctx=$CTX chunk=$CHUNK min_refs=$MIN_REFS slots=$DS4_SLOTS"
echo "out: $OUT"; echo

# GPU/ALU baseline (once)
run_one "gpu_alu" "./run_gpu_prefill_profile_M1MAX64.sh"

# ANE variants
for v in "${VARIANTS[@]}"; do
  name="${v%%|*}"; extra="${v#*|}"
  # shellcheck disable=SC2086
  run_one "ane_$name" "./run_ane_prefill_profile_M1MAX64.sh" $COMMON_ANE $extra
done

echo
echo "=== RANKED (best first) ==="
sort -t, -k2 -gr "$RESULTS" | awk -F, 'NR>=1{printf "%-22s %s t/s\n",$1,$2}'
echo "results: $RESULTS"
