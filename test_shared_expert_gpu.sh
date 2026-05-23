#!/usr/bin/env bash
# Test B: GPU shared expert (baseline; current shipped config).
# Same routed-expert ANE setup as test_shared_expert_ane.sh — only the
# shared expert path differs (GPU q8_0 matmul + swiglu on this side).
# Runs prefill + 50 decode tokens.  Pair with test_shared_expert_ane.sh
# for the A/B comparison.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_RUN_NAME="${DS4_RUN_NAME:-shared_gpu_$(date +%Y%m%d_%H%M%S)}"
export DS4_RUN_NAME
export DS4_TOKENS="${DS4_TOKENS:-50}"
export DS4_SLOTS="${DS4_SLOTS:-8}"
# Explicitly disable to make the intent obvious in profile_runs/.
export DS4_FLASH_MOE_ANE_SHARED_EXPERT=0
unset DS4_FLASH_MOE_SKIP_SHARED_EXPERT

./run_ane_prefill_profile_m3u.sh
