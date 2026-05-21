#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# M3U defaults for GPU-only MPP i8w-i8x fused prefill profiling.
# Local 8K sweep winner: async pread, prefetch 3, issue preads after stage.
# Individual values can be overridden in the environment before invoking this script.
DS4_BIN="${DS4_BIN:-./ds4}"
DS4_MODEL="${DS4_MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
DS4_CTX="${DS4_CTX:-9000}"
DS4_TOKENS="${DS4_TOKENS:-1}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16384}"
DS4_SLOTS="${DS4_SLOTS:-4}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
DS4_LOG_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs}"
DS4_RUN_NAME="${DS4_RUN_NAME:-m3u_gpuonly_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${DS4_PROMPT_FILE:-}" ]]; then
  if [[ -f /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt ]]; then
    DS4_PROMPT_FILE=/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
  elif [[ -f /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt ]]; then
    DS4_PROMPT_FILE=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
  elif [[ -f "$ROOT/tests/long_context_story_prompt.txt" ]]; then
    DS4_PROMPT_FILE="$ROOT/tests/long_context_story_prompt.txt"
  else
    echo "error: set DS4_PROMPT_FILE to a prompt text file" >&2
    exit 2
  fi
fi

mkdir -p "$DS4_LOG_DIR"
LOG="$DS4_LOG_DIR/$DS4_RUN_NAME.log"
SUMMARY="$DS4_LOG_DIR/$DS4_RUN_NAME.summary.txt"

DS4_RUN_ARGS=(
  -m "$DS4_MODEL"
  --moe-sidecar "$DS4_SIDECAR"
  --moe-mode slot-bank
  --metal
  --ctx "$DS4_CTX"
  --tokens "$DS4_TOKENS"
  --temp 0
  --prompt-file "$DS4_PROMPT_FILE"
)

echo "log: $LOG"
echo "summary: $SUMMARY"

env \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-8704}" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  DS4_FLASH_MOE_ANE_PREFILL=0 \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER=0 \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0 \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  DS4_FLASH_MOE_STAGE_STATS="${DS4_FLASH_MOE_STAGE_STATS:-1}" \
  DS4_FLASH_MOE_PROFILE="${DS4_FLASH_MOE_PROFILE:-1}" \
  DS4_PREFILL_PROFILE_DETAIL="${DS4_PREFILL_PROFILE_DETAIL:-1}" \
  DS4_METAL_GRAPH_PREFILL_PROFILE="${DS4_METAL_GRAPH_PREFILL_PROFILE:-1}" \
  "$DS4_BIN" "${DS4_RUN_ARGS[@]}" \
  >"$LOG" 2>&1

{
  echo "run=$DS4_RUN_NAME"
  echo "log=$LOG"
  echo "model=$DS4_MODEL"
  echo "sidecar=$DS4_SIDECAR"
  echo "prompt=$DS4_PROMPT_FILE"
  echo "ctx=$DS4_CTX tokens=$DS4_TOKENS chunk=$DS4_PREFILL_CHUNK slots=$DS4_SLOTS"
  echo "async_pread=${DS4_FLASH_MOE_ASYNC_PREAD:-1} prefetch=${DS4_FLASH_MOE_PREFETCH:-3} after_stage=${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}"
  echo
  grep -E \
    'prefill:|decode:|Flash-MoE prefill dedup|Flash-MoE prefill stage stats|Flash-MoE prefill pread issue stats|Flash-MoE prefill pread bucket|Flash-MoE slot-bank stats|Flash-MoE layer=|gpu (graph|chunked|layer-major) prefill|prefill detail|MPP|ANE prefill stats' \
    "$LOG" || true
} | tee "$SUMMARY"
