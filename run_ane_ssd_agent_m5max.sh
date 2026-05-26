#!/usr/bin/env bash
# run_ane_ssd_agent_m5max.sh
# ds4-agent on M5 Max, SSD slot-bank Dedup-MoE, ANE strategy:
#   - dense projections (q/kv/shared/O-proj) on fp16-NAX  (DS4_GPU_DENSE_NAX, default-ON on M5+)
#   - routed experts on ANE (i8i8 tiled-fused) overlapped with GPU dense  (the dominant slice -> ANE)
#   - shared-expert / O-proj ANE OFF on M5 (single ANE loses; the cluster is busy with routed)
# Compare against run_nax_ssd_agent_m5max.sh (routed on GPU).
# M3 ULTRA: set DS4_FLASH_MOE_ANE_DUAL=1 (dual cluster) and optionally ANE_SHARED_EXPERT=1.
# All knobs env-overridable. Extra ds4-agent args pass through ("$@").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4-agent}"
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DS4_SLOTS="${DS4_SLOTS:-48}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16000}"

# Optional flags from the SSD-prefetch-ANE branch (omitted unless set; safe on codex branch).
# Set once that branch is merged:
#   MOE_PREFETCH_TEMPORAL=1  -> --moe-prefetch-temporal (enables DECODE prefetch). NOTE: for ANE
#                              prefill this does NOT auto-enable prefill top-k slot population (it
#                              benchmarked slower) -> you must force it explicitly below.
#   MOE_PREFETCH_TOPK=N      -> --moe-prefetch-topk N. For ANE prefill, force N=32 (with --moe-slot-bank
#                              >=64 the clamp-to-half makes N=64 run as 32; with slot-bank 48 it clamps to 24).
EXTRA_ARGS=()
if [[ -n "${MOE_PREFETCH_TOPK:-}" ]]; then EXTRA_ARGS+=(--moe-prefetch-topk "$MOE_PREFETCH_TOPK"); fi
if [[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]]; then EXTRA_ARGS+=(--moe-prefetch-temporal); fi

env \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-8704}" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  `# ---- dense projections: fp16-NAX (now default-ON on M5+; explicit for clarity) ----` \
  DS4_GPU_DENSE_NAX="${DS4_GPU_DENSE_NAX:-1}" \
  DS4_GPU_DENSE_I8="${DS4_GPU_DENSE_I8:-0}" \
  `# ---- routed experts on ANE (i8i8 tiled-fused), overlapped with GPU ----` \
  DS4_FLASH_MOE_ANE_PREFILL=1 \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=1 \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER=1 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1 \
  `# ---- GPU MPP int8 routed OFF (routed goes to ANE) ----` \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0 \
  `# ---- ANE batch / refs / scheduler (tuned) ----` \
  DS4_FLASH_MOE_ANE_BATCH="${DS4_FLASH_MOE_ANE_BATCH:-256}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_ANE_MIN_REFS="${DS4_FLASH_MOE_ANE_MIN_REFS:-32}" \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="${DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS:-384}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-0}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK:-0}" \
  `# ---- shared-expert + O-proj ANE OFF on M5 (loses on single ANE) ----` \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-0}" \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-0}" \
  "$DS4_BIN" \
    --model "$DS4_MODEL" \
    --moe-sidecar "$DS4_SIDECAR" \
    --moe-mode slot-bank \
    --moe-slot-bank "$DS4_SLOTS" \
    --metal \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@"
