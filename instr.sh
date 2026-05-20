#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS4_BIN="${SCRIPT_DIR}/ds4"

export DS4_LOCK_FILE=/tmp/ds4-codex.lock
export DS4_METAL_PREFILL_CHUNK=32000
export DS4_FLASH_MOE_PREFETCH=3
export DS4_FLASH_MOE_ANE_STATS=1
export DS4_FLASH_MOE_SCHED_STATS=1
export DS4_FLASH_MOE_ANE_PIPELINE_STATS=1
export DS4_FLASH_MOE_CONCURRENT_STATS=1
export DS4_FLASH_MOE_PROFILE=1
export DS4_FLASH_MOE_ANE_PREFILL=1
export DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1
export DS4_FLASH_MOE_OVERLAP_PREFILL=1
export DS4_FLASH_MOE_OVERLAP_SCHEDULER=1
export DS4_FLASH_MOE_ANE_BATCHES=64,128,256,512
export DS4_FLASH_MOE_ANE_MAX_REFS=512
export DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS=1
export DS4_FLASH_MOE_ANE_MIN_REFS=32
export DS4_FLASH_MOE_MPP_INT8_PREFILL=1
export DS4_FLASH_MOE_MPP_I8I8_PREFILL=1
export DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1
export DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1
export DS4_FLASH_MOE_MPP_INT8_QSCALE=512
export DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32
export DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32

exec "${DS4_BIN}" -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank --metal --ctx 98192 --tokens 1 --temp 0 \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_1k.txt
