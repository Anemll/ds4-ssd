#!/usr/bin/env bash
# run_ssd_kernel_bench_m5.sh
# SSD slot-bank forced-backend bench + bottleneck breakdown for the 32 GB M5
# (resident NAX-half/ALU need 81 GB -> N/A here; SSD routed compute is ANE or
# GPU-int8). Diagnoses "prefill is slow": stage stats show whether the wall is
# SSD I/O (pread/upload) or compute (gate/up/down) — ANE only helps compute.
#
# Forces, per run, with DS4_FLASH_MOE_STAGE_STATS=1:
#   gpu_int8 - routed experts on GPU MPP-int8 (ANE off)
#   ane      - routed experts on ANE (overlapped with GPU dense)
#   ane_all  - ANE with every expert forced on (HYBRID_ANE_MIN_REFS=0, MIN_UTIL=0)
#
# Usage: ./run_ssd_kernel_bench_m5.sh [ctx]   (default ctx 8192)
# Env: DS4_SLOTS (default 8), DS4_PROMPT, COOLDOWN (default 60).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

CTX="${1:-8192}"
DS4_SLOTS="${DS4_SLOTS:-8}"
DS4_PROMPT="${DS4_PROMPT:-tests/test-vectors/prompts/long_code_audit.txt}"
COOLDOWN="${COOLDOWN:-60}"
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
[ -f "$DS4_PROMPT" ] || { echo "prompt $DS4_PROMPT missing" >&2; exit 1; }

run_one() {  # $1=label  $2..=extra env "K=V"
  local label="$1"; shift
  sleep "$COOLDOWN"
  echo "============================== $label =============================="
  { cat "$DS4_PROMPT"; printf '\n/quit\n'; } | env \
    DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16000}" \
    DS4_METAL_GRAPH_RAW_CAP=8704 \
    DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
    DS4_FLASH_MOE_PREFETCH=3 DS4_FLASH_MOE_ASYNC_PREAD=1 DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE=1 \
    DS4_GPU_DENSE_NAX=1 \
    DS4_FLASH_MOE_STAGE_STATS=1 \
    "$@" \
    ./ds4-agent --model "$DS4_MODEL" --moe-sidecar "$DS4_SIDECAR" \
      --moe-mode slot-bank --moe-slot-bank "$DS4_SLOTS" --metal \
      --ctx "$CTX" --non-interactive -n 4 2>&1 \
    | grep -iE "routed experts =|prefill .*avg=|prefill resume:|stage stats|pread|upload|gate=|down=" | tail -8
}

echo "M5 SSD forced-kernel bench  ctx=$CTX slots=$DS4_SLOTS prompt=$(basename "$DS4_PROMPT")"

# 1) GPU-int8 routed
run_one "gpu_int8 (routed->GPU MPP-int8)" \
  DS4_FLASH_MOE_ANE_PREFILL=0 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=1 DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1

# 2) ANE routed (default 384 floor)
run_one "ane (routed->ANE, HYBRID_MIN_REFS=384)" \
  DS4_FLASH_MOE_ANE_PREFILL=1 DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=1 DS4_FLASH_MOE_OVERLAP_SCHEDULER=1 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=384

# 3) ANE all experts (slow-GPU should favor more ANE)
run_one "ane_all (routed->ANE, force every expert)" \
  DS4_FLASH_MOE_ANE_PREFILL=1 DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=1 DS4_FLASH_MOE_OVERLAP_SCHEDULER=1 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=0 DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL=0

cat <<'NOTES'

-------------------------------------------------------------------------------
DIAGNOSING SLOW PREFILL (read the stage stats line)
-------------------------------------------------------------------------------
  pread_ms / upload_ms  >> gate+up+down  => SSD I/O bound (slot bank too small for
       the routed set -> thrashing). Try larger DS4_SLOTS (if RAM allows),
       DS4_FLASH_MOE_PREFETCH=, or accept I/O is the floor on this disk.
  gate+up+down dominates => compute bound => ANE offload should help; pick the
       fastest of the three runs above and lower HYBRID_ANE_MIN_REFS toward it.
  On a 32 GB M5 with only 8 slots, expect I/O to dominate at large ctx — that is
  the likely cause of "really slow", not the kernel choice.
NOTES