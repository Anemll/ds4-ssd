#!/usr/bin/env bash
# run_nax_resident_agent_m5max.sh
# ds4-agent on M5 Max, RESIDENT MoE (all experts in RAM, no SSD sidecar).
#   - full GGUF mmaped resident (~81 GB); --moe-mode off (NO slot-bank, NO sidecar)
#   - routed experts on GPU NAX: "Plan A" = h_h_f fused gate+up+swiglu, MIN_REFS=0
#     (the resident prefill champion: ~573 t/s @16K vs ~532 baseline / ~381 mul_mm_id)
#   - dense projections on fp16-NAX (DS4_GPU_DENSE_NAX, default-ON on M5+)
# Contrast with run_ane_ssd_agent_m5max.sh (SSD slot-bank + routed-on-ANE).
# Needs ~81 GB RAM for the model + KV; M5 Max 137 GB is fine.
# All knobs env-overridable. Extra ds4-agent args pass through ("$@").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

DS4_BIN="${DS4_BIN:-./ds4-agent}"
# Full resident GGUF (experts embedded), NOT the dense-only sidecar model.
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
DS4_CTX="${DS4_CTX:-32768}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16384}"

env \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  `# ---- dense projections: fp16-NAX (default-ON on M5+; explicit for clarity) ----` \
  DS4_GPU_DENSE_NAX="${DS4_GPU_DENSE_NAX:-1}" \
  DS4_GPU_DENSE_I8="${DS4_GPU_DENSE_I8:-0}" \
  `# ---- resident routed MoE: GPU MPP/NAX compact bridge ----` \
  DS4_RESIDENT_MOE_MPP_INT8_PREFILL="${DS4_RESIDENT_MOE_MPP_INT8_PREFILL:-1}" \
  DS4_RESIDENT_MOE_MPP_FORCE="${DS4_RESIDENT_MOE_MPP_FORCE:-1}" \
  DS4_RESIDENT_MOE_MPP_MIN_TOKENS="${DS4_RESIDENT_MOE_MPP_MIN_TOKENS:-64}" \
  DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS="${DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS:-64}" \
  DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE="${DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE:-1}" \
  DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT="${DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT:-1}" \
  `# ---- Plan A: NAX-half fused gate+up+swiglu, engage on ALL experts (MIN_REFS=0) ----` \
  DS4_RESIDENT_MOE_NAX_HALF="${DS4_RESIDENT_MOE_NAX_HALF:-1}" \
  DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP="${DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP:-1}" \
  DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS="${DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS:-0}" \
  "$DS4_BIN" \
    --model "$DS4_MODEL" \
    --moe-mode off \
    --metal \
    --warm-weights \
    --ctx "$DS4_CTX" \
    "$@"
