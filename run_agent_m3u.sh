#!/usr/bin/env bash
# Launch the interactive ds4-agent with the tuned M3 Ultra prefill stack.
# Mirrors run_ane_prefill_profile_m3u.sh / run_bench_sweep_m3u.sh so the agent,
# the profiler, and the benchmark all use the same env (no drifting knobs).
#
# Validated on M3U @ ctx=100000, 76k-token turn: prefill ~266 t/s, peak ~38 GiB
# of 96, zero swaps, coherent output.
#
# Usage:
#   ./run_agent_m3u.sh                 # fresh interactive session, ctx 100000
#   ./run_agent_m3u.sh --resume <SHA>  # resume a saved session
#   CTX=200000 SLOTS=128 ./run_agent_m3u.sh   # override ctx / slot bank
#   any extra args are passed straight through to ds4-agent.
#
# Memory/decode notes (96 GiB box): model ~8.4 + slot-bank gpu-bank + ctx + scratch.
# Default raised to SLOTS=96 (~27.9 GiB gpu-bank) per request. CAUTION: 64 was the
# previously-measured sweet spot (decode ~4.6 t/s @16k, 51% hit-rate). Raising
# slots eventually hits a wired-memory cliff -- 128 (~37 GiB bank) COLLAPSES decode
# to ~0.7 t/s (measured) despite a higher hit-rate. 96 sits between the two and is
# NOT yet validated for sustained decode; if decode tanks, fall back to SLOTS=64.
# (prefill is unaffected -- the slot bank is a decode cache.)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_MODEL="${DS4_MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
CTX="${CTX:-100000}"
SLOTS="${SLOTS:-96}"

# Prefetch pass-through (the engine reads these env vars directly; the agent's
# --moe-prefetch-* flags just set the same). Mirrors run_*_ssd_agent_m5max.sh:
#   MOE_PREFETCH_TOPK=N      -> DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=N (prefill slot-cache top-k)
#   MOE_PREFETCH_TEMPORAL=1  -> DS4_FLASH_MOE_DECODE_PREFETCH=1        (decode prefetch)
EXTRA_ENV=()
[[ -n "${MOE_PREFETCH_TOPK:-}" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK="$MOE_PREFETCH_TOPK")
[[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_DECODE_PREFETCH=1)

env \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16384}" \
  DS4_METAL_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-16896}" \
  `# parallel SSD reader pool (the prefill win)` \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  `# route experts through ANE` \
  DS4_FLASH_MOE_ANE_PREFILL="${DS4_FLASH_MOE_ANE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL="${DS4_FLASH_MOE_ANE_PIPELINE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_PREFILL="${DS4_FLASH_MOE_OVERLAP_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER="${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-0}" \
  `# dual-cluster ANE` \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-1}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_THREADS="${DS4_FLASH_MOE_ANE_THREADS:-2}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  `# the knobs that actually push work onto ANE (binary default keeps it on GPU)` \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  `# dense-on-ANE (M3 Ultra dual cluster). Default off; flip to 1 to test moving` \
  `# the shared-expert FFN and/or attention O-proj onto the ANE.` \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-1}" \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-1}" \
  ./ds4-agent --model "$DS4_MODEL" --moe-sidecar "$DS4_SIDECAR" \
    --moe-mode slot-bank --moe-slot-bank "$SLOTS" \
    --metal --ctx "$CTX" "$@"
