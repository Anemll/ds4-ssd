#!/usr/bin/env bash
# M1 Max (64 GiB) GPU/ALU prefill profile. STREAMING ONLY (Flash-MoE SSD/slot-bank).
# Derived from run_gpu_prefill_profile_m3u.sh with the resident path removed:
# M1 Max has no NAX (M5-only) and a weak ANE, so the routed-MoE prefill runs on
# the mul_mm_id grouped GPU/ALU baseline that works on every chip.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./run_gpu_prefill_profile_M1MAX64.sh [options]

GPU/ALU prefill profile for M1 Max (64 GiB). Streaming only: Flash-MoE
sidecar + slot-bank. Routed-MoE prefill runs on mul_mm_id grouped GPU/ALU
(ANE off, MPP/NAX off).

Run shape:
  --prefill-size N          Prompt size selector, e.g. 1k, 4k, 8k, 16k.
                            Selects coding_N.txt unless --prompt-file is set.
                            Also sets --ctx to N + tokens + 1024 unless --ctx is set.
  --prompt-file FILE        Use an explicit prompt file
  --ctx N                   Context allocation for ds4 (supports k suffix)
  --tokens N                Decode tokens after prefill. Default: 1
  --chunk N                 DS4_METAL_PREFILL_CHUNK (supports k suffix)
  --raw-cap N               DS4_METAL_GRAPH_RAW_CAP (supports k suffix)

Paths:
  --model FILE              Override dense model path
  --sidecar DIR             Flash-MoE sidecar dir
  --bin FILE                ds4 binary. Default: ./ds4
  --log-dir DIR             Log output directory
  --run-name NAME           Log basename
  --lock-file FILE          DS4_LOCK_FILE
  --slots N                 Slot-bank slots. Default: 64

Noise:
  DS4_PREFILL_TEST_VERBOSE=1
                            Include generated per-layer planner lines in the
                            summary. Default is concise.

Examples:
  ./run_gpu_prefill_profile_M1MAX64.sh --prefill-size 8k
  ./run_gpu_prefill_profile_M1MAX64.sh --slots 64 --prefill-size 4k
  DS4_PROMPT_FILE=/tmp/p.txt ./run_gpu_prefill_profile_M1MAX64.sh
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
    "$ROOT/tests/test-vectors/prompts/coding/coding_${label}.txt" \
    "$ROOT/tests/test-vectors/prompts/long_code_audit.txt" \
    "$ROOT/tests/long_context_story_prompt.txt"; do
    [[ -f "$cand" ]] && DS4_PROMPT_FILE="$cand" && return
  done
  die "set DS4_PROMPT_FILE or provide a prompt that exists for --prefill-size ${label}"
}

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_SSD_MODEL_DEFAULT="${DS4_SSD_MODEL_DEFAULT:-$HOME/Models/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-$HOME/Models/dsv4-iq2xxs-expert-major}"
DS4_PREFILL_SIZE="${DS4_PREFILL_SIZE:-8k}"
DS4_CTX_SET=0
[[ -n "${DS4_CTX+x}" ]] && DS4_CTX_SET=1
DS4_CTX="${DS4_CTX:-}"
DS4_TOKENS="${DS4_TOKENS:-1}"
DS4_PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-16384}"
DS4_GRAPH_RAW_CAP_SET=0
[[ -n "${DS4_METAL_GRAPH_RAW_CAP+x}" ]] && DS4_GRAPH_RAW_CAP_SET=1
DS4_GRAPH_RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-}"
DS4_SLOTS="${DS4_SLOTS:-64}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
DS4_LOG_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs}"
DS4_RUN_NAME="${DS4_RUN_NAME:-}"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ssd|--flash-moe) shift ;;  # accepted for symmetry; streaming is the only mode
    --prefill-size) DS4_PREFILL_SIZE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --prompt-file) DS4_PROMPT_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --ctx) DS4_CTX="$(parse_size "$(need_arg "$1" "${2:-}")")"; DS4_CTX_SET=1; shift 2 ;;
    --tokens|-n) DS4_TOKENS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --chunk) DS4_PREFILL_CHUNK="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --raw-cap) DS4_GRAPH_RAW_CAP="$(parse_size "$(need_arg "$1" "${2:-}")")"; DS4_GRAPH_RAW_CAP_SET=1; shift 2 ;;
    --model|-m) DS4_MODEL="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --sidecar) DS4_SIDECAR="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --bin) DS4_BIN="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --log-dir) DS4_LOG_DIR="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --run-name) DS4_RUN_NAME="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --lock-file) DS4_LOCK_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --slots) DS4_SLOTS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

PREFILL_TOKENS="$(parse_size "$DS4_PREFILL_SIZE")"
PROMPT_LABEL="$(prompt_label "$DS4_PREFILL_SIZE")"
find_prompt_file "$PROMPT_LABEL"

if [[ -z "$DS4_CTX" || "$DS4_CTX_SET" == "0" ]]; then
  DS4_CTX="$((PREFILL_TOKENS + DS4_TOKENS + 1024))"
fi
if [[ -z "$DS4_GRAPH_RAW_CAP" || "$DS4_GRAPH_RAW_CAP_SET" == "0" ]]; then
  DS4_GRAPH_RAW_CAP="$((PREFILL_TOKENS + 512))"
fi
DS4_MODEL="${DS4_MODEL:-$DS4_SSD_MODEL_DEFAULT}"
if [[ -z "$DS4_RUN_NAME" ]]; then
  DS4_RUN_NAME="m1max64_flash-moe_gpuonly_${PROMPT_LABEL}_$(date +%Y%m%d_%H%M%S)"
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
  --moe-sidecar "$DS4_SIDECAR"
  --moe-mode slot-bank
  --moe-slot-bank "$DS4_SLOTS"
)
PREFILL_KERNEL="Flash-MoE SSD GPU/ALU routed-MoE: mul_mm_id grouped GPU/ALU, ANE and MPP/NAX off"

echo "mode: flash-moe (streaming)"
echo "prefill_kernel: $PREFILL_KERNEL"
echo "log: $LOG"
echo "summary: $SUMMARY"

SUMMARY_GREP='ds4: -+|prefill routed-MoE|prefill compute|prefill:|decode:|Flash-MoE prefill dedup|Flash-MoE prefill stage stats|Flash-MoE prefill pread issue stats|Flash-MoE prefill pread bucket|Flash-MoE slot-bank stats|(gpu|metal) (graph|chunked|layer-major) prefill|prefill detail'
if [[ "${DS4_PREFILL_TEST_VERBOSE:-0}" == "1" ]]; then
  SUMMARY_GREP="$SUMMARY_GREP|Flash-MoE layer=|Flash-MoE hybrid prefill|Flash-MoE overlap plan"
fi

EXTRA_ENV=()
[[ -n "${MOE_PREFETCH_TOPK:-}" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK="$MOE_PREFETCH_TOPK")
[[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_DECODE_PREFETCH=1)

env \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  DS4_PROFILE="${DS4_PROFILE:-none}" \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$DS4_PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="$DS4_GRAPH_RAW_CAP" \
  DS4_FLASH_MOE_SLOT_BANK_SLOTS="$DS4_SLOTS" \
  DS4_FLASH_MOE_ANE_PREFILL=0 \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_PREFILL=0 \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER=0 \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0 \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=0 \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0 \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0 \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  DS4_FLASH_MOE_STAGE_STATS="${DS4_FLASH_MOE_STAGE_STATS:-1}" \
  DS4_FLASH_MOE_PROFILE="${DS4_FLASH_MOE_PROFILE:-1}" \
  DS4_PREFILL_PROFILE_DETAIL="${DS4_PREFILL_PROFILE_DETAIL:-1}" \
  DS4_METAL_GRAPH_PREFILL_PROFILE="${DS4_METAL_GRAPH_PREFILL_PROFILE:-1}" \
  "$DS4_BIN" "${DS4_RUN_ARGS[@]}" \
  >"$LOG" 2>&1

{
  echo "run=$DS4_RUN_NAME"
  echo "mode=flash-moe (streaming)"
  echo "log=$LOG"
  echo "model=$DS4_MODEL"
  echo "sidecar=$DS4_SIDECAR"
  echo "prefill_kernel=$PREFILL_KERNEL"
  echo "prompt=$DS4_PROMPT_FILE"
  echo "prefill_size=$DS4_PREFILL_SIZE ctx=$DS4_CTX tokens=$DS4_TOKENS chunk=$DS4_PREFILL_CHUNK raw_cap=$DS4_GRAPH_RAW_CAP slots=$DS4_SLOTS"
  echo "async_pread=${DS4_FLASH_MOE_ASYNC_PREAD:-1} prefetch=${DS4_FLASH_MOE_PREFETCH:-3} after_stage=${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}"
  echo "summary_detail=$([[ "${DS4_PREFILL_TEST_VERBOSE:-0}" == "1" ]] && echo verbose || echo concise)"
  echo
  grep -E "$SUMMARY_GREP" "$LOG" || true
} | tee "$SUMMARY"
