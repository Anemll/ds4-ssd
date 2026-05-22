#!/usr/bin/env bash
# M3 Ultra ANE prefill profile — dual-cluster optimal config.
#
# Defaults capture the tuned dual-cluster ANE prefill path for DSv4 IQ2_XXS.
# Expected prefill throughput at the 8K coding prompt:
#   prefill ~275-282 t/s  (vs ~220 t/s GPU-only, ~218 t/s single-cluster ANE)
# This is ~1.226x over GPU-only and ~1.36x over the original m3u defaults.
#
# Every knob below is `${VAR:-default}` so the outer shell can override any
# of them.  See moe-batch-bench/DUAL_ANE_CLUSTER_OPTIMIZATION.md for what
# each setting buys and why.
#
# To force GPU-only for an A/B comparison, use run_gpu_prefill_profile_m3u.sh
# (it sets the ANE/MPP flags the opposite way).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_MODEL="${DS4_MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
DS4_CTX="${DS4_CTX:-9000}"
DS4_TOKENS="${DS4_TOKENS:-1}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16384}"
DS4_SLOTS="${DS4_SLOTS:-4}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
DS4_LOG_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs}"
DS4_RUN_NAME="${DS4_RUN_NAME:-m3u_ane_prefill_$(date +%Y%m%d_%H%M%S)}"

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
  `# --- Infra ---` \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-8704}" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  `# --- Route expert work through ANE (master gates) ---` \
  `# These select the ANE i8i8 tiled-fused path.  Setting any of them to 0 ` \
  `# disables ANE prefill — useful for an A/B vs the GPU script. ` \
  DS4_FLASH_MOE_ANE_PREFILL="${DS4_FLASH_MOE_ANE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL="${DS4_FLASH_MOE_ANE_PIPELINE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_PREFILL="${DS4_FLASH_MOE_OVERLAP_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER="${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" \
  `# Force GPU-MPP path off so the hybrid scheduler doesn't steal expert work ` \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-0}" \
  `# --- Dual-cluster ANE optimizations (M3 Ultra has 2 ANEx16 clusters) ---` \
  `# DUAL=1: two ANE workers per call, one per cluster, splitting chunks ` \
  `# MULTI_ACTIVE=1: scheduler skips wait_predict join so eval(N+1) starts ` \
  `#                during eval(N) — both clusters compute in parallel ` \
  `# OUTPUT_QUEUE=4: post-eval queue depth, lets writeback overlap with next eval ` \
  `# GPU_OUTPUT_PACK=1: GPU does the f16->f32 + route-weight scaling ` \
  `# PREFLUSH_EVERY=4: batch every 4 calls' dequant kernels into one CB commit ` \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-1}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  `# Batch + refs tuning: single batch size keeps the ctx cache simple. ` \
  `# Sweep peak is BATCHES=256, MAX_REFS=256, HYBRID_ANE_MIN_REFS=384. ` \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_ANE_MIN_REFS="${DS4_FLASH_MOE_ANE_MIN_REFS:-32}" \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="${DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS:-384}" \
  `# Scheduler routing weights between ANE and GPU paths ` \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  `# --- Staging / pread tuning (8K winner: async + prefetch=3 + after-stage) ---` \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  `# Quantization scales (still set so the MPP path is correctly tuned if invoked) ` \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  `# --- Profiling / stats output (cheap; off in production binaries) ---` \
  DS4_FLASH_MOE_ANE_STATS="${DS4_FLASH_MOE_ANE_STATS:-1}" \
  DS4_FLASH_MOE_SCHED_STATS="${DS4_FLASH_MOE_SCHED_STATS:-1}" \
  DS4_FLASH_MOE_HYBRID_STATS="${DS4_FLASH_MOE_HYBRID_STATS:-1}" \
  DS4_FLASH_MOE_CONCURRENT_STATS="${DS4_FLASH_MOE_CONCURRENT_STATS:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_STATS="${DS4_FLASH_MOE_ANE_PIPELINE_STATS:-1}" \
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
    'prefill:|decode:|Flash-MoE prefill dedup|Flash-MoE prefill stage stats|Flash-MoE prefill pread issue stats|Flash-MoE prefill pread bucket|Flash-MoE slot-bank stats|Flash-MoE hybrid prefill|Flash-MoE overlap plan|ANE prefill stats|ANE quant stats|ANE i8i8 hidden quant stats|ANE prefill timing|ANE prefill chunks|ANE prefill batch_hist|gpu (graph|chunked|layer-major) prefill|prefill detail' \
    "$LOG" || true
} | tee "$SUMMARY"
