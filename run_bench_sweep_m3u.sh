#!/usr/bin/env bash
# M3 Ultra context-depth prefill/decode sweep via ds4-bench.
# Defaults to Flash-MoE SSD/slot-bank. Pass --resident for full-model resident
# ANE prefill with no sidecar.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./run_bench_sweep_m3u.sh [options]

M3 Ultra ds4-bench sweep. Defaults to Flash-MoE SSD/slot-bank with the tuned
dual-cluster ANE + parallel-pread env. Use --resident to run resident/full-model
ANE prefill instead.

Modes:
  --ssd, --flash-moe        Flash-MoE sidecar + slot-bank mode (default)
  --resident               Full resident model, no sidecar, --resident-ane-prefill

Sweep:
  --prefill-size N          One-shot row at N tokens, e.g. 4k, 8k, 16k.
                            Selects coding_N.txt unless --prompt-file is set.
                            Sets ctx-start=ctx-max=N.
  --ctx-start N             First measured frontier. Default: 4096
  --ctx-max N               Last measured frontier. Default: 65536
  --ctx-alloc N             Allocated context passed to ds4-bench
  --step-incr N             Linear step. Default: 4096
  --gen-tokens N            Decode tokens per frontier. Default: 128
  --full-prefill-each-frontier

Runtime:
  --prompt-file FILE        Use an explicit prompt file
  --chunk N                 DS4_METAL_PREFILL_CHUNK (supports k suffix)
  --raw-cap N               DS4_METAL_GRAPH_RAW_CAP (supports k suffix)
  --slots N                 Slot-bank slots for SSD mode. Default: 96
  --csv FILE                CSV output. Default: /tmp/ds4_bench_m3u.csv

Paths:
  --model FILE              Override model path
  --resident-model FILE     Default model used by --resident
  --sidecar DIR             Flash-MoE sidecar dir
  --bin FILE                ds4-bench binary. Default: ./ds4-bench
  --lock-file FILE          DS4_LOCK_FILE

Noise:
  DS4_RESIDENT_MOE_MPP_STATS=1
                            Include resident per-layer MPP/NAX stats.
                            Default is off for this harness.

Examples:
  ./run_bench_sweep_m3u.sh --prefill-size 8k --gen-tokens 1
  ./run_bench_sweep_m3u.sh --resident --prefill-size 16k --ctx-alloc 20k --gen-tokens 1
  ./run_bench_sweep_m3u.sh --ctx-start 4k --ctx-max 32k --step-incr 4k
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
  die "set DS4_PROMPT_FILE or provide a prompt that exists for ${label}"
}

MODE="${DS4_RUN_MODE:-flash-moe}"
DS4_BIN="${DS4_BIN:-./ds4-bench}"
DS4_SSD_MODEL_DEFAULT="${DS4_SSD_MODEL_DEFAULT:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
DS4_RESIDENT_MODEL="${DS4_RESIDENT_MODEL:-/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf}"
DS4_SIDECAR="${DS4_SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
DS4_SLOTS="${DS4_SLOTS:-96}"
DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
CTX_START="${CTX_START:-4096}"
CTX_MAX="${CTX_MAX:-65536}"
STEP_INCR="${STEP_INCR:-4096}"
GEN_TOKENS="${GEN_TOKENS:-128}"
CTX_ALLOC="${CTX_ALLOC:-}"
CSV="${CSV:-/tmp/ds4_bench_m3u.csv}"
PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16384}"
RAW_CAP="${DS4_METAL_GRAPH_RAW_CAP:-16896}"
PREFILL_SIZE=""
FULL_PREFILL=0

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ssd|--flash-moe) MODE="flash-moe"; shift ;;
    --resident) MODE="resident"; shift ;;
    --prefill-size) PREFILL_SIZE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --ctx-start) CTX_START="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --ctx-max) CTX_MAX="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --ctx-alloc) CTX_ALLOC="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --step-incr) STEP_INCR="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --gen-tokens|--tokens|-n) GEN_TOKENS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --full-prefill-each-frontier) FULL_PREFILL=1; shift ;;
    --prompt-file) DS4_PROMPT_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --chunk) PREFILL_CHUNK="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --raw-cap) RAW_CAP="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --slots) DS4_SLOTS="$(parse_size "$(need_arg "$1" "${2:-}")")"; shift 2 ;;
    --csv) CSV="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --model|-m) DS4_MODEL="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --resident-model) DS4_RESIDENT_MODEL="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --sidecar) DS4_SIDECAR="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --bin) DS4_BIN="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    --lock-file) DS4_LOCK_FILE="$(need_arg "$1" "${2:-}")"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ -n "$PREFILL_SIZE" ]]; then
  CTX_START="$(parse_size "$PREFILL_SIZE")"
  CTX_MAX="$CTX_START"
  STEP_INCR="$CTX_START"
  PROMPT_LABEL="$(prompt_label "$PREFILL_SIZE")"
else
  PROMPT_LABEL="$(prompt_label "$CTX_MAX")"
fi
find_prompt_file "$PROMPT_LABEL"

if [[ -z "${DS4_MODEL:-}" ]]; then
  if [[ "$MODE" == "resident" ]]; then
    DS4_MODEL="$DS4_RESIDENT_MODEL"
  else
    DS4_MODEL="$DS4_SSD_MODEL_DEFAULT"
  fi
fi

BENCH_ARGS=(
  -m "$DS4_MODEL"
  --metal
  --prompt-file "$DS4_PROMPT_FILE"
  --ctx-start "$CTX_START"
  --ctx-max "$CTX_MAX"
  --step-incr "$STEP_INCR"
  --gen-tokens "$GEN_TOKENS"
  --csv "$CSV"
)
[[ -n "$CTX_ALLOC" ]] && BENCH_ARGS+=(--ctx-alloc "$CTX_ALLOC")
[[ "$FULL_PREFILL" == "1" ]] && BENCH_ARGS+=(--full-prefill-each-frontier)

if [[ "$MODE" == "resident" ]]; then
  BENCH_ARGS+=(--moe-mode off --resident-ane-prefill)
  [[ "${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-1}" != "0" ]] && BENCH_ARGS+=(--resident-ane-shared-expert)
  [[ "${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-1}" != "0" ]] && BENCH_ARGS+=(--resident-ane-oproj)
  PREFILL_KERNEL="resident ANE routed-MoE: ds4-bench --resident-ane-prefill (shared/oproj ANE unless disabled)"
else
  BENCH_ARGS+=(--moe-sidecar "$DS4_SIDECAR" --moe-mode slot-bank --moe-slot-bank "$DS4_SLOTS")
  PREFILL_KERNEL="Flash-MoE SSD ANE routed-MoE: GPU dedup + ANE/GPU overlap scheduler"
fi

echo "mode: $MODE"
echo "prefill_kernel: $PREFILL_KERNEL"
echo "csv: $CSV"
echo "sweep: ctx $CTX_START..$CTX_MAX step $STEP_INCR gen $GEN_TOKENS chunk=$PREFILL_CHUNK raw_cap=$RAW_CAP slots=$DS4_SLOTS"
[[ -n "$CTX_ALLOC" ]] && echo "ctx_alloc: $CTX_ALLOC"

EXTRA_ENV=()
[[ -n "${MOE_PREFETCH_TOPK:-}" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK="$MOE_PREFETCH_TOPK")
[[ "${MOE_PREFETCH_TEMPORAL:-0}" == "1" ]] && EXTRA_ENV+=(DS4_FLASH_MOE_DECODE_PREFETCH=1)

env \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  DS4_PROFILE="${DS4_PROFILE:-none}" \
  DS4_LOCK_FILE="$DS4_LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK="$PREFILL_CHUNK" \
  DS4_METAL_GRAPH_RAW_CAP="$RAW_CAP" \
  DS4_FLASH_MOE_PREAD_THREADS="${DS4_FLASH_MOE_PREAD_THREADS:-6}" \
  DS4_FLASH_MOE_ASYNC_READAHEAD="${DS4_FLASH_MOE_ASYNC_READAHEAD:-12}" \
  DS4_FLASH_MOE_ASYNC_PREAD="${DS4_FLASH_MOE_ASYNC_PREAD:-1}" \
  DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE="${DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE:-1}" \
  DS4_FLASH_MOE_PREFETCH="${DS4_FLASH_MOE_PREFETCH:-3}" \
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
  DS4_FLASH_MOE_ANE_DUAL="${DS4_FLASH_MOE_ANE_DUAL:-1}" \
  DS4_FLASH_MOE_ANE_MULTI_ACTIVE="${DS4_FLASH_MOE_ANE_MULTI_ACTIVE:-1}" \
  DS4_FLASH_MOE_ANE_THREADS="${DS4_FLASH_MOE_ANE_THREADS:-2}" \
  DS4_FLASH_MOE_ANE_OUTPUT_QUEUE="${DS4_FLASH_MOE_ANE_OUTPUT_QUEUE:-4}" \
  DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK="${DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK:-1}" \
  DS4_FLASH_MOE_ANE_PREFLUSH_EVERY="${DS4_FLASH_MOE_ANE_PREFLUSH_EVERY:-4}" \
  DS4_FLASH_MOE_ANE_BATCHES="${DS4_FLASH_MOE_ANE_BATCHES:-256}" \
  DS4_FLASH_MOE_ANE_MAX_REFS="${DS4_FLASH_MOE_ANE_MAX_REFS:-256}" \
  DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS="${DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS:-1}" \
  DS4_FLASH_MOE_MPP_INT8_QSCALE="${DS4_FLASH_MOE_MPP_INT8_QSCALE:-512}" \
  DS4_FLASH_MOE_MPP_INT8_X_QSCALE="${DS4_FLASH_MOE_MPP_INT8_X_QSCALE:-32}" \
  DS4_FLASH_MOE_MPP_INT8_MID_QSCALE="${DS4_FLASH_MOE_MPP_INT8_MID_QSCALE:-32}" \
  DS4_RESIDENT_MOE_MPP_STATS="${DS4_RESIDENT_MOE_MPP_STATS:-0}" \
  DS4_FLASH_MOE_SCHED_ANE_REL_SPEED="${DS4_FLASH_MOE_SCHED_ANE_REL_SPEED:-99}" \
  DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL="${DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL:-0.0}" \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-1}" \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-1}" \
  "$DS4_BIN" "${BENCH_ARGS[@]}"

echo "done -> $CSV"
