#!/usr/bin/env bash
# Test A: ANE shared expert (DS4_FLASH_MOE_ANE_SHARED_EXPERT=1).
# Runs prefill + 50 decode tokens with the synchronous ANE shared-expert
# evaluator replacing the GPU q8_0 matmul + swiglu chain.  The full ANE
# stack (THREADS=2 dual-cluster routed expert + ANE shared expert) is
# active.  Pair with test_shared_expert_gpu.sh for the A/B comparison.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_RUN_NAME="${DS4_RUN_NAME:-shared_ane_$(date +%Y%m%d_%H%M%S)}"
export DS4_RUN_NAME
export DS4_TOKENS="${DS4_TOKENS:-50}"
export DS4_SLOTS="${DS4_SLOTS:-8}"
export DS4_FLASH_MOE_ANE_SHARED_EXPERT=1

./run_ane_prefill_profile_m3u.sh
