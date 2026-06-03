#!/usr/bin/env bash
# Decisive test: does a resident crossover table (ane_gpu for small chunks,
# nax_int8 for large) regress the large nax_int8 chunk because the ane* segment
# forces ANE_HYBRID -> global 2048 scratch tile -> sync bridge?
#
# Measures the 16384 chunk (16k prompt) under:
#   A) CONTROL  : pure nax_int8 table          (no ANE flags)         -> expect ~631 t/s
#   B) CROSSOVER: "8191:ane_gpu,99999:nax_int8" + ANE defaults (trap) -> the unknown
# If B ~= A  -> trap does NOT bite -> use a multi-segment resident table.
# If B << A  -> trap bites -> keep single-segment nax_int8.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

MODEL="/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
PROMPT="/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_16k.txt"
LOCK="/tmp/ds4-m5max-profile.lock"
OUT="$ROOT/moe-batch-bench/profile_runs/resident_crossover_trap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

run() {  # $1=name  $2..=extra env (KEY=VAL)
  local name="$1"; shift
  echo "==> $name"
  env DS4_PROFILE=none DS4_LOCK_FILE="$LOCK" \
    DS4_METAL_PREFILL_CHUNK=16384 DS4_METAL_GRAPH_RAW_CAP=8704 \
    DS4_GPU_DENSE_NAX=1 \
    DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1 DS4_RESIDENT_MOE_MPP_FORCE=1 \
    DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64 DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS=64 \
    DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=1 DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT=1 \
    "$@" \
    ./ds4 -m "$MODEL" --metal --moe-mode off --ctx 20000 --tokens 1 --temp 0 \
      --prompt-file "$PROMPT" > "$OUT/$name.log" 2>&1 || echo "  (exit $?)"
  echo -n "  tps="; awk '/prefill:/{v=$3;sub(/,/,"",v)} END{print v}' "$OUT/$name.log"
  grep -hE "resident-NAX tile|sync-bridge|prefill compute|prefill routed-MoE" "$OUT/$name.log" | head -3 | sed 's/^/  /'
}

# A) control: pure nax_int8, no ANE anywhere
run control_nax_int8

# B) crossover: ane_gpu<8192 + nax_int8>=8192. Replicate apply_resident_ane_prefill_defaults()
#    (the loader sets these whenever an ane* segment is present).
run crossover_ane_lt8192 \
  DS4_RESIDENT_MOE_PREFILL_BY_TOKENS="8191:ane_gpu,99999:nax_int8" \
  DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=1 DS4_RESIDENT_MOE_ANE_HYBRID=1 \
  DS4_RESIDENT_MOE_ANE_HYBRID_OUTER=0 DS4_RESIDENT_MOE_COMPACT_SCRATCH=1 \
  DS4_RESIDENT_MOE_MPP_MIN_TILE_UTIL=0.0 \
  DS4_RESIDENT_MOE_ANE_MIN_REFS=64 DS4_RESIDENT_MOE_ANE_MAX_REFS=1024 DS4_RESIDENT_MOE_ANE_QUEUE=8 \
  DS4_RESIDENT_MOE_ANE_GPU_TAIL_GATHER_SCATTER=1 DS4_RESIDENT_MOE_ANE_GPU_TAIL_OVERLAP=1 \
  DS4_METAL_LAZY_MODEL_VIEWS=1 DS4_METAL_DECODE_RESIDENCY=1 \
  DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE=1 DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1

echo "logs: $OUT"
