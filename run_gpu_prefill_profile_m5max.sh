#!/usr/bin/env bash
# M5 Max GPU prefill profile. Defaults to Flash-MoE SSD/slot-bank; pass
# --resident to run the same GPU baseline against the full resident GGUF.
#
# On M5+ the meaningful "GPU route" is NAX-int8 (MPP int8 matmul2d, W8A8) -- the
# real competitor to ANE. That is the DEFAULT here. Pass --alu to force the plain
# GPU/ALU mul_mm_id baseline (the M3U gpu-script behaviour) instead.
# Compare against run_ane_prefill_profile_m5max.sh (routed -> ANE).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./run_gpu_prefill_profile_m5max.sh [options]

GPU prefill profile for M5 Max. Defaults to Flash-MoE SSD/slot-bank, routed on
NAX-int8 (W8A8). Use --resident for the full resident GGUF; --alu for plain GPU/ALU.

Routes:
  (default)                 routed experts on NAX-int8 (MPP int8 matmul2d, W8A8)
  --alu                     routed experts on plain GPU/ALU (mulmm / mul_mm_id), MPP/NAX off

Modes:
  --ssd, --flash-moe        Flash-MoE sidecar + slot-bank mode (default)
  --resident               Full resident model, no sidecar, --moe-mode off

Run shape:
  --prefill-size N          Prompt size selector, e.g. 1k, 4k, 16k.
                            Selects coding_N.txt unless --prompt-file is set.
                            Also sets --ctx to N + tokens + 1024 unless --ctx is set.
  --prompt-file FILE        Use an explicit prompt file
  --ctx N                   Context allocation for ds4 (supports k suffix)
  --tokens N                Decode tokens after prefill. Default: 1
  --chunk N                 DS4_METAL_PREFILL_CHUNK (supports k suffix)
  --raw-cap N               DS4_METAL_GRAPH_RAW_CAP (supports k suffix)

Paths:
  --model FILE              Override model path
  --resident-model FILE     Default model used by --resident
  --sidecar DIR             Flash-MoE sidecar dir
  --bin FILE                ds4 binary. Default: ./ds4
  --log-dir DIR             Log output directory
  --run-name NAME           Log basename
  --lock-file FILE          DS4_LOCK_FILE
  --slots N                 Slot-bank slots for SSD mode. Default: 96

Examples:
  ./run_gpu_prefill_profile_m5max.sh --prefill-size 4k
  ./run_gpu_prefill_profile_m5max.sh --alu --prefill-size 16k
  ./run_gpu_prefill_profile_m5max.sh --resident --prefill-size 16k --ctx 20k
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

need_arg() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires an argument"
  printf '%s\n' "$2"
}

parse_size() {
  local raw="$1"
  local s
  s="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  local mult=1
  case "$s" in
    *k) mult=1024; s="${s%k}" ;;
    *m) mult=1048576; s="${s%m}" ;;
  esac
  [[ "$s" =~ ^[0-9]+$ ]] || die "invalid numeric value: $raw"
  printf '%d\n' "$((s * mult))"
}

prompt_label() {
  local n
  n="$(parse_size "$1")"
  if (( n % 1024 == 0 )); then
    printf '%dk\n' "$((n / 1024))"
  else
    printf '%d\n' "$n"
  fi
}

find_prompt_file() {
  [[ -n "${DS4_PROMPT_FILE:-}" ]] && return
  local label="$1"
  local cand
  for cand in \
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_${label}.txt" \
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_${label}.txt" \
    "$ROOT/tests/long_context_story_prompt.txt"; do
    [[ -f "$cand" ]] && DS4_PROMPT_FILE="$cand" && return
  done
  die "set DS4_PROMPT_FILE or provide a prompt that exists for --prefill-size ${label}"
}

MODE="${DS4_RUN_MODE:-flash-moe}"
ROUTE="${DS4_GPU_ROUTE:-nax_int8}"   # nax_int8 (default) | alu
DS4_BIN="${DS4_BIN:-./ds4}"
DS4_SSD_MODEL_DEFAULT="${DS4_SSD_MODEL_DEFAULT:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_RESIDENT_MODEL="${DS4_RESIDENT_MODEL:-/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DS4_PREFILL_SIZE="${DS4_PREFILL_SIZE:-8k}"
DS4_CTX_SET=0
[[ -n "${DS4_CTX+x}" ]] && DS4_CTX_SET=1
DS4_CTX="${DS4_CTX:-}"
DS4_TOKENS="${DS4_TOKENS:-1}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16384}"
DS4_GRAPH_RAW_CAP_SET=0
[[ -n "${DS4_METAL_GRAPH_RAW_CAP+x}" ]] && DS4_GRAPH_RAW_CAP_SET=1
DS4_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-}"
DS4_SLOTS="${DS4_SLOTS:-96}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-m5max-profile.lock}"
DS4_LOG_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs}"
DS4_RUN_NAME="${DS4_RUN_NAME:-}"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ssd|--flash-moe) MODE="flash-moe"; shift ;;
    --resident) MODE="resident"; shift ;;
    --alu) ROUTE="alu"; shift ;;
    --nax-int8|--nax) ROUTE="nax_int8"; shift ;;
    --prefill-size) DS4_PREFILL_SIZE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --prompt-file) DS4_PROMPT_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --ctx) DS4_CTX="$(parse_size "$(need_arg "$1" "${2:-}")")"; DS4_CTX_SET=1; shift 2 ;;
    --tokens|-n) DS4_TOKENS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --chunk) DS4_PREFILL_CHUNK="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --raw-cap) DS4_GRAPH_RAW_CAP="$(parse_size "$(need_arg "$1" "${2:-}")")"; DS4_GRAPH_RAW_CAP_SET=1; shift 2 ;;
    --model|-m) DS4_MODEL="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --resident-model) DS4_RESIDENT_MODEL="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --sidecar) DS4_SIDECAR="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --bin) DS4_BIN="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --log-dir) DS4_LOG_DIR="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --run-name) DS4_RUN_NAME="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --lock-file) DS4_LOCK_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --slots) DS4_SLOTS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$ROUTE" == "nax_int8" || "$ROUTE" == "alu" ]] || die "invalid route: $ROUTE (nax_int8|alu)"

PREFILL_TOKENS="$(parse_size "$DS4_PREFILL_SIZE")"
PROMPT_LABEL="$(prompt_label "$DS4_PREFILL_SIZE")"
find_prompt_file "$PROMPT_LABEL"

if [[ -z "$DS4_CTX" || "$DS4_CTX_SET" == "0" ]]; then
  DS4_CTX="$((PREFILL_TOKENS + DS4_TOKENS + 1024))"
fi
if [[ -z "$DS4_GRAPH_RAW_CAP" || "$DS4_GRAPH_RAW_CAP_SET" == "0" ]]; then
  DS4_GRAPH_RAW_CAP="$((PREFILL_TOKENS + 512))"
fi
if [[ -z "${DS4_MODEL:-}" ]]; then
  if [[ "$MODE" == "resident" ]]; then
    DS4_MODEL="$DS4_RESIDENT_MODEL"
  else
    DS4_MODEL="$DS4_SSD_MODEL_DEFAULT"
  fi
fi
if [[ -z "$DS4_RUN_NAME" ]]; then
  DS4_RUN_NAME="m5max_${MODE}_gpu_${ROUTE}_${PROMPT_LABEL}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$DS4_LOG_DIR"
LOG="$DS4_LOG_DIR/$DS4_RUN_NAME.log"
SUMMARY="$DS4_LOG_DIR/$DS4_RUN_NAME.summary.txt"

DS4_RUN_ARGS=(
  -m "$DS4_MODEL"
  --metal
  --ctx "$DS4_CTX"
  --tokens "$DS4_TOKENS"
  --temp 0
  --prompt-file "$DS4_PROMPT_FILE"
)

# Route -> routed-expert backend env (flash-moe and resident).
if [[ "$ROUTE" == "nax_int8" ]]; then
  FM_MPP_INT8=1; FM_MPP_I8I8=1; FM_MPP_FUSED=1
  RESIDENT_BACKEND="nax_int8"
  ROUTE_DESC="NAX-int8 (MPP int8 matmul2d, W8A8)"
else
  FM_MPP_INT8=0; FM_MPP_I8I8=0; FM_MPP_FUSED=0
  RESIDENT_BACKEND="mulmm"
  ROUTE_DESC="plain GPU/ALU (mul_mm_id)"
fi

if [[ "$MODE" == "resident" ]]; then
  DS4_RUN_ARGS+=(--moe-mode off)
  PREFILL_KERNEL="resident GPU routed-MoE (M5): ${ROUTE_DESC}, ANE off | dense proj = fp16-NAX"
else
  DS4_RUN_ARGS+=(--moe-sidecar "$DS4_SIDECAR" --moe-mode slot-bank --moe-slot-bank "$DS4_SLOTS")
  PREFILL_KERNEL="Flash-MoE SSD GPU routed-MoE (M5): ${ROUTE_DESC}, ANE off | dense proj = fp16-NAX"
fi

echo "mode: $MODE"
echo "route: $ROUTE ($ROUTE_DESC)"
echo "prefill_kernel: $PREFILL_KERNEL"
echo "log: $LOG"
echo "summary: $SUMMARY"

SUMMARY_GREP='ds4: -+|prefill routed-MoE|prefill compute|prefill:|decode:|Flash-MoE prefill dedup|Flash-MoE prefill stage stats|Flash-MoE prefill pread issue stats|Flash-MoE prefill pread bucket|Flash-MoE slot-bank stats|resident routed MoE|resident ANE|MPP/NAX|(gpu|metal) (graph|chunked|layer-major) prefill|prefill detail'
if [[ "${DS4_PREFILL_TEST_VERBOSE:-0}" == "1" ]]; then
  SUMMARY_GREP="$SUMMARY_GREP|resident MPP/NAX prefill layer=|Flash-MoE layer=|Flash-MoE hybrid prefill|Flash-MoE overlap plan"
fi

EXTRA_ENV=()
[[ -n "${MOE_PREFETCH_TOPK:-}" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK="$MOE_PREFETCH_TOPK")
[[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_DECODE_PREFETCH=1)

RESIDENT_ENV=()
if [[ "$MODE" == "resident" ]]; then
  RESIDENT_ENV+=(
    DS4_RESIDENT_MOE_ANE_HYBRID="${DS4_RESIDENT_MOE_ANE_HYBRID:-0}"
    DS4_METAL_NO_PREFILL_KERNEL_WARMUP="${DS4_METAL_NO_PREFILL_KERNEL_WARMUP:-1}"
    DS4_METAL_GPU_BATCH_EMBED_MIN="${DS4_METAL_GPU_BATCH_EMBED_MIN:-1048576}"
  )
  if [[ "$ROUTE" == "nax_int8" ]]; then
    # Validated resident NAX-int8 path (matches ds4_profile.json M5 Max env).
    # BACKEND=nax_int8 alone is NOT enough -- it falls back to the grouped MPP/NAX
    # path and segfaults on the sync bridge. The MPP int8 + compact-bridge +
    # fused-dequant knobs are what actually engage the W8A8 matmul2d.
    RESIDENT_ENV+=(
      DS4_RESIDENT_MOE_MPP_INT8_PREFILL="${DS4_RESIDENT_MOE_MPP_INT8_PREFILL:-1}"
      DS4_RESIDENT_MOE_MPP_FORCE="${DS4_RESIDENT_MOE_MPP_FORCE:-1}"
      DS4_RESIDENT_MOE_MPP_MIN_TOKENS="${DS4_RESIDENT_MOE_MPP_MIN_TOKENS:-64}"
      DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS="${DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS:-64}"
      DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE="${DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE:-1}"
      DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT="${DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT:-1}"
    )
  else
    RESIDENT_ENV+=(
      DS4_RESIDENT_MOE_BACKEND="${DS4_RESIDENT_MOE_BACKEND:-mulmm}"
      DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL="${DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL:-0}"
    )
  fi
fi

env \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  ${RESIDENT_ENV[@]+"${RESIDENT_ENV[@]}"} \
  DS4_PROFILE="${DS4_PROFILE:-none}" \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="$DS4_GRAPH_RAW_CAP" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  `# ---- dense projections: fp16-NAX (default-ON on M5+; explicit) ----` \
  DS4_GPU_DENSE_NAX="${DS4_GPU_DENSE_NAX:-1}" \
  DS4_GPU_DENSE_I8="${DS4_GPU_DENSE_I8:-0}" \
  `# ---- ANE fully OFF for this path ----` \
  DS4_FLASH_MOE_ANE_PREFILL=0 \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER=0 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0 \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0 \
  `# ---- routed route: NAX-int8 (default) or plain ALU (--alu) ----` \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="$FM_MPP_INT8" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="$FM_MPP_I8I8" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="$FM_MPP_FUSED" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-4}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  DS4_RESIDENT_MOE_MPP_STATS="${DS4_RESIDENT_MOE_MPP_STATS:-0}" \
  DS4_FLASH_MOE_STAGE_STATS="${DS4_FLASH_MOE_STAGE_STATS:-1}" \
  DS4_FLASH_MOE_PROFILE="${DS4_FLASH_MOE_PROFILE:-1}" \
  DS4_PREFILL_PROFILE_DETAIL="${DS4_PREFILL_PROFILE_DETAIL:-1}" \
  DS4_METAL_GRAPH_PREFILL_PROFILE="${DS4_METAL_GRAPH_PREFILL_PROFILE:-1}" \
  "$DS4_BIN" "${DS4_RUN_ARGS[@]}" \
  >"$LOG" 2>&1

{
  echo "run=$DS4_RUN_NAME"
  echo "mode=$MODE"
  echo "route=$ROUTE ($ROUTE_DESC)"
  echo "log=$LOG"
  echo "model=$DS4_MODEL"
  [[ "$MODE" == "flash-moe" ]] && echo "sidecar=$DS4_SIDECAR"
  echo "prefill_kernel=$PREFILL_KERNEL"
  echo "prompt=$DS4_PROMPT_FILE"
  echo "prefill_size=$DS4_PREFILL_SIZE ctx=$DS4_CTX tokens=$DS4_TOKENS chunk=$DS4_PREFILL_CHUNK raw_cap=$DS4_GRAPH_RAW_CAP slots=$DS4_SLOTS"
  echo "async_pread=${DS4_FLASH_MOE_ASYNC_PREAD:-1} prefetch=${DS4_FLASH_MOE_PREFETCH:-3} after_stage=${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}"
  echo "summary_detail=$([[ "${DS4_PREFILL_TEST_VERBOSE:-0}" == "1" ]] && echo verbose || echo concise)"
  echo
  grep -E "$SUMMARY_GREP" "$LOG" || true
} | tee "$SUMMARY"
