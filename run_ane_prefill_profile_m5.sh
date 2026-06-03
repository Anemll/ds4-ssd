#!/usr/bin/env bash
# M5 (32 GB) ANE prefill profile. Flash-MoE SSD/slot-bank only.
#
# This box has 32 GB RAM, so there is NO resident path (the full ~81 GB GGUF
# does not fit). Routed experts stream from the SSD sidecar.
#
# M5 is a single ANE cluster, so this script defaults to:
#   - DS4_FLASH_MOE_ANE_DUAL=0, DS4_FLASH_MOE_ANE_THREADS=1 (one cluster)
#   - shared-expert + O-proj ANE OFF (single ANE loses; cluster is busy w/ routed)
#   - dense projections on fp16-NAX (DS4_GPU_DENSE_NAX, default-ON on M5+)
# On M3 Ultra use run_ane_prefill_profile_m3u.sh instead (dual cluster, resident).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./run_ane_prefill_profile_m5.sh [options]

ANE prefill profile for M5 (32 GB). Flash-MoE sidecar + slot-bank only.
No resident mode: 32 GB cannot host the full model.

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
  --slots N                 Slot-bank slots. Default: 48

Noise:
  DS4_PREFILL_TEST_VERBOSE=1
                            Include generated per-layer planner lines in the
                            summary. Default is concise.

Examples:
  ./run_ane_prefill_profile_m5.sh --prefill-size 8k
  ./run_ane_prefill_profile_m5.sh --prefill-size 16k --ctx 20k
  DS4_FLASH_MOE_ANE_MIN_REFS=32 ./run_ane_prefill_profile_m5.sh
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

# Generate a prompt of approximately N tokens by cycling words from a source
# text. Measured ~1.2 tokens/word on the story corpus, so 0.83 words/token
# lands close to N tokens while staying inside the auto ctx (N + tokens + 1024).
gen_sized_prompt() {
  local tokens="$1" label="$2"
  local src
  for src in \
    "$ROOT/tests/long_context_story_prompt.txt" \
    "$ROOT/tests/long_context_security_prompt.txt"; do
    [[ -f "$src" ]] && break
  done
  [[ -f "$src" ]] || return 1
  local words=$(( tokens * 83 / 100 ))
  local outdir="$ROOT/tests/generated"
  local out="$outdir/prompt_${label}_${words}w.txt"
  if [[ ! -s "$out" ]]; then
    mkdir -p "$outdir"
    awk -v need="$words" '
      { for (i = 1; i <= NF; i++) { buf[n++] = $i } }
      END {
        out = 0
        while (out < need) {
          for (i = 0; i < n && out < need; i++) {
            printf "%s%s", (out ? " " : ""), buf[i]
            out++
          }
        }
        printf "\n"
      }' "$src" > "$out" || return 1
  fi
  DS4_PROMPT_FILE="$out"
}

find_prompt_file() {
  [[ -n "${DS4_PROMPT_FILE:-}" ]] && return
  local label="$1"
  local cand
  for cand in \
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_${label}.txt" \
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_${label}.txt"; do
    [[ -f "$cand" ]] && DS4_PROMPT_FILE="$cand" && return
  done
  # No coding_N.txt on this box: synthesize a size-matched prompt.
  gen_sized_prompt "$PREFILL_TOKENS" "$label" && return
  die "set DS4_PROMPT_FILE or provide a prompt that exists for --prefill-size ${label}"
}

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_SSD_MODEL_DEFAULT="${DS4_SSD_MODEL_DEFAULT:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
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
DS4_SLOTS="${DS4_SLOTS:-48}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
DS4_LOG_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs}"
DS4_RUN_NAME="${DS4_RUN_NAME:-}"

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ssd|--flash-moe) shift ;;
    --resident) die "--resident not supported on this 32 GB M5 (model does not fit); SSD slot-bank only" ;;
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
  DS4_RUN_NAME="m5_flash-moe_ane_prefill_${PROMPT_LABEL}_$(date +%Y%m%d_%H%M%S)"
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

if [[ "${DS4_FLASH_MOE_GPU_DEDUP:-1}" == "0" ]]; then
  PREFILL_KERNEL="Flash-MoE SSD ANE routed-MoE: CPU dedup + sequential ANE/GPU experts"
elif [[ "${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" == "0" ]]; then
  PREFILL_KERNEL="Flash-MoE SSD ANE routed-MoE: GPU dedup + basic ANE/GPU overlap, scheduler off"
else
  PREFILL_KERNEL="Flash-MoE SSD ANE routed-MoE: GPU dedup + ANE/GPU overlap scheduler"
fi

echo "mode: flash-moe"
echo "prefill_kernel: $PREFILL_KERNEL"
echo "log: $LOG"
echo "summary: $SUMMARY"

SUMMARY_GREP='ds4: -+|prefill routed-MoE|prefill compute|prefill:|decode:|Flash-MoE prefill dedup|Flash-MoE prefill stage stats|Flash-MoE prefill pread issue stats|Flash-MoE prefill pread bucket|Flash-MoE slot-bank stats|ANE prefill stats|ANE quant stats|ANE i8i8 hidden quant stats|ANE prefill timing|ANE prefill chunks|ANE prefill batch_hist|gpu chunked prefill start=|metal layer-major prefill total|metal graph prefill|gpu graph prefill|prefill detail'
if [[ "${DS4_PREFILL_TEST_VERBOSE:-0}" == "1" ]]; then
  SUMMARY_GREP="$SUMMARY_GREP|Flash-MoE hybrid prefill|Flash-MoE overlap plan"
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
  `# ---- dense projections: fp16-NAX (default-ON on M5+; explicit for clarity) ----` \
  DS4_GPU_DENSE_NAX="${DS4_GPU_DENSE_NAX:-1}" \
  DS4_GPU_DENSE_I8="${DS4_GPU_DENSE_I8:-0}" \
  DS4_FLASH_MOE_ANE_PREFILL="${DS4_FLASH_MOE_ANE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL="${DS4_FLASH_MOE_ANE_PIPELINE_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_PREFILL:-1}" \
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL="${DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL:-1}" \
  DS4_FLASH_MOE_GPU_DEDUP="${DS4_FLASH_MOE_GPU_DEDUP:-1}" \
  DS4_FLASH_MOE_OVERLAP_PREFILL="${DS4_FLASH_MOE_OVERLAP_PREFILL:-1}" \
  DS4_FLASH_MOE_OVERLAP_SCHEDULER="${DS4_FLASH_MOE_OVERLAP_SCHEDULER:-1}" \
  DS4_FLASH_MOE_MPP_INT8_PREFILL="${DS4_FLASH_MOE_MPP_INT8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_PREFILL:-0}" \
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL="${DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL:-0}" \
  `# ---- M5: single ANE cluster ----` \
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-0}" \
  DS4_FLASH_MOE_ANE_THREADS="${DS4_FLASH_MOE_ANE_THREADS:-1}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK:-0}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_ANE_MIN_REFS="${DS4_FLASH_MOE_ANE_MIN_REFS:-32}" \
  DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="${DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS:-384}" \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  `# ---- shared-expert + O-proj ANE OFF on M5 (loses on single ANE) ----` \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-0}" \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-0}" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  DS4_FLASH_MOE_ANE_STATS="${DS4_FLASH_MOE_ANE_STATS:-1}" \
  DS4_FLASH_MOE_SCHED_STATS="${DS4_FLASH_MOE_SCHED_STATS:-1}" \
  DS4_FLASH_MOE_HYBRID_STATS="${DS4_FLASH_MOE_HYBRID_STATS:-1}" \
  DS4_FLASH_MOE_CONCURRENT_STATS="${DS4_FLASH_MOE_CONCURRENT_STATS:-1}" \
  DS4_FLASH_MOE_ANE_PIPELINE_STATS="${DS4_FLASH_MOE_ANE_PIPELINE_STATS:-1}" \
  DS4_FLASH_MOE_PROFILE="${DS4_FLASH_MOE_PROFILE:-1}" \
  DS4_PREFILL_PROFILE_DETAIL="${DS4_PREFILL_PROFILE_DETAIL:-1}" \
  DS4_METAL_GRAPH_PREFILL_PROFILE="${DS4_METAL_GRAPH_PREFILL_PROFILE:-1}" \
  "$DS4_BIN" "${DS4_RUN_ARGS[@]}" \
  >"$LOG" 2>&1

{
  echo "run=$DS4_RUN_NAME"
  echo "mode=flash-moe"
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
