#!/usr/bin/env bash
# M3 Ultra context-depth prefill/decode sweep via ds4-bench, with the tuned
# dual-cluster ANE + parallel-pread env.  Mirrors run_ane_prefill_profile_m3u.sh
# so the benchmark and the production prefill path stay in lockstep.
#
# Every knob is ${VAR:-default} so the outer shell can override any of them.
# Notable M3U-tuned defaults:
#   DS4_FLASH_MOE_PREAD_THREADS=6      parallel SSD reader pool (universal win)
#   DS4_FLASH_MOE_ANE_THREADS=2        dual cluster (3/4 within noise on M3U)
#   DS4_FLASH_MOE_SCHED_ANE_REL_SPEED=99  force ANE-heavy expert routing
#   DS4_METAL_PREFILL_CHUNK=16384      big chunk => far fewer expert-stream passes
#
# Sweep defaults reproduce the standard 4096-step frontier walk to 64k.
# Tip: --moe-slot-bank 8 is right for a *prefill* sweep (slot bank is a decode
# cache; crank it for decode-heavy measurement).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4-bench}"
DS4_MODEL="${DS4_MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
DS4_SLOTS="${DS4_SLOTS:-8}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"

CTX_START="${CTX_START:-4096}"
CTX_MAX="${CTX_MAX:-65536}"
STEP_INCR="${STEP_INCR:-4096}"
GEN_TOKENS="${GEN_TOKENS:-128}"
CSV="${CSV:-/tmp/ds4_bench_m3u.csv}"

# Big prefill chunk + matching raw-KV cap so each step is a single expert-stream
# pass (raw_chunk_cap must exceed the chunk; window is DS4_N_SWA=128).
PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16384}"
RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-16896}"

if [[ -z "${DS4_PROMPT_FILE:-}" ]]; then
  for cand in \
    /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_64k.txt \
    /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_64k.txt \
    "$ROOT/tests/long_context_story_prompt.txt"; do
    [[ -f "$cand" ]] && DS4_PROMPT_FILE="$cand" && break
  done
  [[ -z "${DS4_PROMPT_FILE:-}" ]] && { echo "error: set DS4_PROMPT_FILE" >&2; exit 2; }
fi

echo "csv: $CSV"
echo "sweep: ctx $CTX_START..$CTX_MAX step $STEP_INCR gen $GEN_TOKENS  chunk=$PREFILL_CHUNK slots=$DS4_SLOTS"

env \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="$RAW_CAP" \
  `# --- parallel SSD reader pool (short/long prefill win) ---` \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  `# --- route expert work through ANE ---` \
  DS4_FLASH_MOE_ANE_PREFILL="${DS4_FLASH_MOE_ANE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL="${DS4_FLASH_MOE_ANE_PIPELINE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_PREFILL="${DS4_FLASH_MOE_OVERLAP_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER="${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-0}" \
  `# --- dual-cluster ANE + size-adaptive worker cap ---` \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-1}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_THREADS="${DS4_FLASH_MOE_ANE_THREADS:-2}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  "$DS4_BIN" -m "$DS4_MODEL" --metal \
    --moe-sidecar "$DS4_SIDECAR" --moe-mode slot-bank --moe-slot-bank "$DS4_SLOTS" \
    --prompt-file "$DS4_PROMPT_FILE" \
    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-incr "$STEP_INCR" \
    --gen-tokens "$GEN_TOKENS" --csv "$CSV"

echo "done -> $CSV"
