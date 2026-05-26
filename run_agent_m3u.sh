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
# Memory/decode notes (96 GiB box): model ~8.4 + slot-bank(64)=18.5 GB gpu-bank
# + ctx(100k)=~5.9 + scratch ~= 38 GiB.  SLOTS=64 is the measured sweet spot:
# decode ~4.6 t/s @16k ctx (51% expert hit-rate).  Do NOT raise to 128 -- the
# 37 GiB gpu-bank hits a wired-memory cliff and decode COLLAPSES to ~0.7 t/s
# (measured) despite a higher hit-rate; prefill is unaffected (slot bank is a
# decode cache).  If anything, sweep DOWN (32/48) if you need more headroom.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DS4_MODEL="${DS4_MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
CTX="${CTX:-100000}"
SLOTS="${SLOTS:-64}"

env \
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
  ./ds4-agent --model "$DS4_MODEL" --moe-sidecar "$DS4_SIDECAR" \
    --moe-mode slot-bank --moe-slot-bank "$SLOTS" \
    --metal --ctx "$CTX" "$@"
