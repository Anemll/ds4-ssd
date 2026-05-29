#!/usr/bin/env bash
# run_nax_ssd_agent_m5max.sh
# ds4-agent on M5 Max, SSD slot-bank Dedup-MoE, GPU/NAX strategy:
#   - dense projections (q/kv/shared/O-proj) on fp16-NAX  (DS4_GPU_DENSE_NAX, default-ON on M5+)
#   - routed experts on the GPU MPP int8 dedup path       (no ANE)
# This is the "all-GPU" path. Compare against run_ane_ssd_agent_m5max.sh (routed->ANE).
# All knobs env-overridable. Extra ds4-agent args pass through ("$@").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4-agent}"
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DS4_SLOTS="${DS4_SLOTS:-48}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16000}"

# Optional flags from the SSD-prefetch-ANE branch (omitted unless set, so this is safe
# on codex/integrate-ds4-agent which doesn't parse them yet; set once that branch is merged):
#   MOE_PREFETCH_TOPK=N      -> --moe-prefetch-topk N  (prefill top-k slot population from the
#                              already-read buffer -> avoids a 2nd SSD read; clamped to half slot-bank).
#                              For the GPU-routed path this can cut sidecar preads; try N=slot-bank.
#   MOE_PREFETCH_TEMPORAL=1  -> --moe-prefetch-temporal (enables decode prefetch).
EXTRA_ARGS=()
if [[ -n "${MOE_PREFETCH_TOPK:-}" ]]; then EXTRA_ARGS+=(--moe-prefetch-topk "$MOE_PREFETCH_TOPK"); fi
if [[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]]; then EXTRA_ARGS+=(--moe-prefetch-temporal); fi

env \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  `# ---- dense projections: fp16-NAX (now default-ON on M5+; explicit for clarity) ----` \
  DS4_GPU_DENSE_NAX="${DS4_GPU_DENSE_NAX:-1}" \
  DS4_GPU_DENSE_I8="${DS4_GPU_DENSE_I8:-0}" \
  `# ---- routed experts on GPU MPP int8 dedup (NOT ANE) ----` \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-1}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  `# ---- ANE fully OFF for this path ----` \
  DS4_FLASH_MOE_ANE_PREFILL=0 \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER=0 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0 \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0 \
  "$DS4_BIN" \
    --model "$DS4_MODEL" \
    --moe-sidecar "$DS4_SIDECAR" \
    --moe-mode slot-bank \
    --moe-slot-bank "$DS4_SLOTS" \
    --metal \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@"
