#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# M5 Max defaults for ANE-only i8w-i8x tiled-fused prefill profiling.
# Individual values can be overridden in the environment before invoking this script.
export DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
export DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
export DS4_CTX="${DS4_CTX:-98192}"
export DS4_TOKENS="${DS4_TOKENS:-1}"
export DS4_SLOTS="${DS4_SLOTS:-32}"
export DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-32000}"
export DS4_METAL_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-8704}"

if [[ -z "${DS4_PROMPT_FILE:-}" ]]; then
  export DS4_PROMPT_FILE=/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
fi

# Current M5 Max best ANE-only profile settings from the 8K production sweep.
export DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}"
export DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}"
export DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}"
export DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}"
export DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}"
export DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}"
export DS4_FLASH_MOE_ANE_MIN_REFS="${DS4_FLASH_MOE_ANE_MIN_REFS:-32}"
export DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-0}"
export DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}"
export DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}"
export DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}"
export DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="${DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS:-384}"

# GPU output pack was faster on M5 Max for this ANE-only prefill path.
export DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}"
export DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK:-0}"

export DS4_RUN_NAME="${DS4_RUN_NAME:-m5max_aneonly_b256_$(date +%Y%m%d_%H%M%S)}"

exec "$ROOT/run_ane_prefill_profile_m4pro.sh"
