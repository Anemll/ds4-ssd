#!/usr/bin/env bash
# M5 Max context-depth prefill/decode sweep via ds4-bench, mirroring
# run_ane_prefill_profile_m5max.sh (single-ANE i8i8 tiled-fused + parallel pread)
# so the benchmark and the M5 production prefill path stay in lockstep.
# This is the M5 counterpart of run_bench_sweep_m3u.sh; key M5 differences:
#   DS4_FLASH_MOE_ANE_DUAL=0   (M5 Max has ONE ANE cluster, not two)
#   model/sidecar under /Users/anemll/Models/flash/...
#   no DS4_FLASH_MOE_ANE_THREADS override (binary default for single cluster)
#
# Every knob is ${VAR:-default} so the outer shell can override any of them.
# For the GPU baseline, set the ANE/MPP knobs the opposite way (see comment at end).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4-bench}"
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DS4_SLOTS="${DS4_SLOTS:-32}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"

CTX_START="${CTX_START:-16384}"
CTX_MAX="${CTX_MAX:-131072}"
STEP_INCR="${STEP_INCR:-16384}"
GEN_TOKENS="${GEN_TOKENS:-1}"
CSV="${CSV:-/tmp/ds4_bench_m5max.csv}"

PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16384}"
RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-16896}"

if [[ -z "${DS4_PROMPT_FILE:-}" ]]; then
  for cand in \
    "$ROOT/moe-batch-bench/coding_187k.txt" \
    /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_64k.txt \
    "$ROOT/tests/long_context_story_prompt.txt"; do
    [[ -f "$cand" ]] && DS4_PROMPT_FILE="$cand" && break
  done
  [[ -z "${DS4_PROMPT_FILE:-}" ]] && { echo "error: set DS4_PROMPT_FILE (need >= CTX_MAX tokens)" >&2; exit 2; }
fi

echo "csv: $CSV"
echo "sweep: ctx $CTX_START..$CTX_MAX step $STEP_INCR gen $GEN_TOKENS  chunk=$PREFILL_CHUNK slots=$DS4_SLOTS  (M5 Max)"

EXTRA_ENV=()
[[ -n "${MOE_PREFETCH_TOPK:-}" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK="$MOE_PREFETCH_TOPK")
[[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_DECODE_PREFETCH=1)

env \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="$RAW_CAP" \
  `# --- parallel SSD reader pool ---` \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  `# --- route expert work through ANE (single cluster on M5 Max) ---` \
  DS4_FLASH_MOE_ANE_PREFILL="${DS4_FLASH_MOE_ANE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL="${DS4_FLASH_MOE_ANE_PIPELINE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_PREFILL="${DS4_FLASH_MOE_OVERLAP_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER="${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-0}" \
  `# --- M5 Max single-cluster ANE (no DUAL) ---` \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-0}" \
  `# M5 has ONE physical ANE cluster, but 2 software workers (oversubscribed)` \
  `# was reported faster than 1 -- the binary default with DUAL=0 is only 1, so` \
  `# force 2 here. A/B with DS4_FLASH_MOE_ANE_THREADS=1 on the real box.` \
  DS4_FLASH_MOE_ANE_THREADS="${DS4_FLASH_MOE_ANE_THREADS:-2}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="${DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS:-384}" \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  `# dense-on-ANE off by default on M5 (untested); set =1 to try.` \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-0}" \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-0}" \
  "$DS4_BIN" -m "$DS4_MODEL" --metal \
    --moe-sidecar "$DS4_SIDECAR" --moe-mode slot-bank --moe-slot-bank "$DS4_SLOTS" \
    --prompt-file "$DS4_PROMPT_FILE" \
    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-incr "$STEP_INCR" \
    --gen-tokens "$GEN_TOKENS" --csv "$CSV"

echo "done -> $CSV"

# GPU baseline: prepend these to flip ANE off / MPP on:
#   DS4_FLASH_MOE_ANE_PREFILL=0 DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0 \
#   DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0 \
#   DS4_FLASH_MOE_MPP_INT8_PREFILL=1 DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1 \
#   DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0
