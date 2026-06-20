#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_ds4_agent_m5m_stable.sh [options] [-- extra ds4-agent args...]

Runs ds4-agent with the stable M5 Max MXFP4 Flash-MoE grow/recovery profile.

Common options:
  -m, --model PATH              Model path
  --ctx N                       Context size
  -n, --tokens N                Max generation tokens
  --temp F                      Temperature
  --moe-slot-bank N             Initial/steady slot bank
  --recovery-slot-bank N        Slot bank after compression recovery
  --debug | --no-debug          Show/hide live footer slot/compression status

Logging / diagnostics:
  --snapshot-file PATH          Slot snapshot log path
  --snapshot-tokens N           Slot snapshot interval in decode tokens
  --snapshot-hot-tokens N       Hot-window size for snapshot tail accounting
  --stats N                     DS4_FLASH_MOE_RESIDENCY_STATS interval
  --start-stats | --no-start-stats

Recovery / grow tuning:
  --grow-step N
  --grow-warmup-tokens N
  --grow-interval-tokens N
  --grow-max-gpu-mb N
  --grow-max-sys-mb N
  --recovery-gpu-mb N
  --recovery-sys-mb N
  --recovery-samples N
  --recovery-interval N
  --slow-dump | --no-slow-dump
  --slow-seconds N
  --slow-tps N
  --slow-min-tokens N
  --slow-min-resident-pct N
  --slow-max-dumps N
  --slow-slot-bank N
  --prefill-recovery | --no-prefill-recovery
  --kv-touch | --no-kv-touch

Other:
  --lock-file PATH
  --agent PATH                  ds4-agent binary
  --no-time                     Do not wrap with /usr/bin/time -lp
  --dry-run                     Print command/env and exit
  -h, --help

Examples:
  scripts/run_ds4_agent_m5m_stable.sh --ctx 132768 --debug
  scripts/run_ds4_agent_m5m_stable.sh --moe-slot-bank 128 --recovery-slot-bank 96
  scripts/run_ds4_agent_m5m_stable.sh -- --prompt-file /tmp/prompt.txt
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AGENT="${DS4_AGENT_BIN:-./ds4-agent}"
MODEL="${DS4_MODEL:-/Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash}"
CTX="${DS4_CTX:-132768}"
TOKENS="${DS4_TOKENS:-20000}"
TEMP="${DS4_TEMP:-0}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-cli-grow-no-prefill-compact.lock}"
TIME_WRAP=1
DRY_RUN=0
DEBUG_STATUS="${DS4_AGENT_DEBUG_STATUS:-1}"

MOE_SLOT_BANK="${DS4_FLASH_MOE_SLOT_BANK_DEFAULT:-134}"
RECOVERY_SLOT_BANK="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_SLOT_BANK:-96}"

GROW_STEP="${DS4_FLASH_MOE_GROW_SLOT_BANK_STEP:-16}"
GROW_WARMUP_TOKENS="${DS4_FLASH_MOE_GROW_SLOT_BANK_WARMUP_TOKENS:-128}"
GROW_INTERVAL_TOKENS="${DS4_FLASH_MOE_GROW_SLOT_BANK_INTERVAL_TOKENS:-128}"
GROW_MIN_RESIDENT_PCT="${DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_RESIDENT_PCT:-99}"
GROW_MIN_HIT_PCT="${DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_HIT_PCT:-80}"
GROW_MAX_GPU_MB="${DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_COMPRESSED_MB:-5000}"
GROW_MAX_SYS_MB="${DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_SYS_COMPRESSED_MB:-8192}"

RECOVERY_INTERVAL="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_INTERVAL:-16}"
RECOVERY_GPU_MB="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_MB:-1024}"
RECOVERY_SYS_MB="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_SYS_MB:-8192}"
RECOVERY_SAMPLES="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_SAMPLES:-1}"
SLOW_DECODE_DUMP="${DS4_FLASH_MOE_SLOW_DECODE_DUMP_CACHE:-1}"
SLOW_DECODE_SECONDS="${DS4_FLASH_MOE_SLOW_DECODE_SECONDS:-5}"
SLOW_DECODE_TPS="${DS4_FLASH_MOE_SLOW_DECODE_TPS:-1}"
SLOW_DECODE_MIN_TOKENS="${DS4_FLASH_MOE_SLOW_DECODE_MIN_TOKENS:-0}"
SLOW_DECODE_MIN_RESIDENT_PCT="${DS4_FLASH_MOE_SLOW_DECODE_MIN_RESIDENT_PCT:-50}"
SLOW_DECODE_MAX_DUMPS="${DS4_FLASH_MOE_SLOW_DECODE_MAX_DUMPS:-1}"
SLOW_DECODE_SLOT_BANK="${DS4_FLASH_MOE_SLOW_DECODE_SLOT_BANK:-}"
PREFILL_RECOVERY="${DS4_FLASH_MOE_COMPRESSION_RECOVERY_ON_PREFILL_START:-1}"
KV_TOUCH="${DS4_METAL_KV_TOUCH_ON_DECODE_START:-1}"

SNAPSHOT_TOKENS="${DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS:-16}"
SNAPSHOT_FILE="${DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE:-/tmp/ds4-slot-grow-no-prefill-compact.log}"
SNAPSHOT_HOT_TOKENS="${DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS:-128}"
RESIDENCY_STATS="${DS4_FLASH_MOE_RESIDENCY_STATS:-0}"
RESIDENCY_STATS_START="${DS4_FLASH_MOE_RESIDENCY_STATS_START:-1}"

EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    -m|--model)
      MODEL="${2:?missing value for $1}"; shift 2 ;;
    --ctx)
      CTX="${2:?missing value for $1}"; shift 2 ;;
    -n|--tokens)
      TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --temp)
      TEMP="${2:?missing value for $1}"; shift 2 ;;
    --moe-slot-bank)
      MOE_SLOT_BANK="${2:?missing value for $1}"; shift 2 ;;
    --recovery-slot-bank)
      RECOVERY_SLOT_BANK="${2:?missing value for $1}"; shift 2 ;;
    --debug)
      DEBUG_STATUS=1; shift ;;
    --no-debug)
      DEBUG_STATUS=0; shift ;;
    --snapshot-file)
      SNAPSHOT_FILE="${2:?missing value for $1}"; shift 2 ;;
    --snapshot-tokens)
      SNAPSHOT_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --snapshot-hot-tokens)
      SNAPSHOT_HOT_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --stats|--residency-stats)
      RESIDENCY_STATS="${2:?missing value for $1}"; shift 2 ;;
    --start-stats)
      RESIDENCY_STATS_START=1; shift ;;
    --no-start-stats)
      RESIDENCY_STATS_START=0; shift ;;
    --grow-step)
      GROW_STEP="${2:?missing value for $1}"; shift 2 ;;
    --grow-warmup-tokens)
      GROW_WARMUP_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --grow-interval-tokens)
      GROW_INTERVAL_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --grow-max-gpu-mb)
      GROW_MAX_GPU_MB="${2:?missing value for $1}"; shift 2 ;;
    --grow-max-sys-mb)
      GROW_MAX_SYS_MB="${2:?missing value for $1}"; shift 2 ;;
    --recovery-gpu-mb)
      RECOVERY_GPU_MB="${2:?missing value for $1}"; shift 2 ;;
    --recovery-sys-mb)
      RECOVERY_SYS_MB="${2:?missing value for $1}"; shift 2 ;;
    --recovery-samples)
      RECOVERY_SAMPLES="${2:?missing value for $1}"; shift 2 ;;
    --recovery-interval)
      RECOVERY_INTERVAL="${2:?missing value for $1}"; shift 2 ;;
    --slow-dump)
      SLOW_DECODE_DUMP=1; shift ;;
    --no-slow-dump)
      SLOW_DECODE_DUMP=0; shift ;;
    --slow-seconds)
      SLOW_DECODE_SECONDS="${2:?missing value for $1}"; shift 2 ;;
    --slow-tps)
      SLOW_DECODE_TPS="${2:?missing value for $1}"; shift 2 ;;
    --slow-min-tokens)
      SLOW_DECODE_MIN_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --slow-min-resident-pct)
      SLOW_DECODE_MIN_RESIDENT_PCT="${2:?missing value for $1}"; shift 2 ;;
    --slow-max-dumps)
      SLOW_DECODE_MAX_DUMPS="${2:?missing value for $1}"; shift 2 ;;
    --slow-slot-bank)
      SLOW_DECODE_SLOT_BANK="${2:?missing value for $1}"; shift 2 ;;
    --prefill-recovery)
      PREFILL_RECOVERY=1; shift ;;
    --no-prefill-recovery)
      PREFILL_RECOVERY=0; shift ;;
    --kv-touch)
      KV_TOUCH=1; shift ;;
    --no-kv-touch)
      KV_TOUCH=0; shift ;;
    --lock-file)
      LOCK_FILE="${2:?missing value for $1}"; shift 2 ;;
    --agent)
      AGENT="${2:?missing value for $1}"; shift 2 ;;
    --no-time)
      TIME_WRAP=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

cd "$ROOT"

if [[ -z "$SLOW_DECODE_SLOT_BANK" ]]; then
  SLOW_DECODE_SLOT_BANK="$RECOVERY_SLOT_BANK"
fi

ENV_ARGS=(
  "DS4_LOCK_FILE=$LOCK_FILE"
  "DS4_FLASH_MOE_GROW_AFTER_COMPRESSION_RECOVERY=1"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_DURING_DECODE=1"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_CARRY=1"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_STEP=$GROW_STEP"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_WARMUP_TOKENS=$GROW_WARMUP_TOKENS"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_INTERVAL_TOKENS=$GROW_INTERVAL_TOKENS"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_RESIDENT_PCT=$GROW_MIN_RESIDENT_PCT"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_HIT_PCT=$GROW_MIN_HIT_PCT"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_COMPRESSED_MB=$GROW_MAX_GPU_MB"
  "DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_SYS_COMPRESSED_MB=$GROW_MAX_SYS_MB"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY=1"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_INTERVAL=$RECOVERY_INTERVAL"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_MB=$RECOVERY_GPU_MB"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_SYS=1"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_SYS_MB=$RECOVERY_SYS_MB"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_SAMPLES=$RECOVERY_SAMPLES"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_SLOT_BANK=$RECOVERY_SLOT_BANK"
  "DS4_FLASH_MOE_COMPRESSION_RECOVERY_ON_PREFILL_START=$PREFILL_RECOVERY"
  "DS4_FLASH_MOE_SLOW_DECODE_DUMP_CACHE=$SLOW_DECODE_DUMP"
  "DS4_FLASH_MOE_SLOW_DECODE_SECONDS=$SLOW_DECODE_SECONDS"
  "DS4_FLASH_MOE_SLOW_DECODE_TPS=$SLOW_DECODE_TPS"
  "DS4_FLASH_MOE_SLOW_DECODE_MIN_TOKENS=$SLOW_DECODE_MIN_TOKENS"
  "DS4_FLASH_MOE_SLOW_DECODE_MIN_RESIDENT_PCT=$SLOW_DECODE_MIN_RESIDENT_PCT"
  "DS4_FLASH_MOE_SLOW_DECODE_MAX_DUMPS=$SLOW_DECODE_MAX_DUMPS"
  "DS4_FLASH_MOE_SLOW_DECODE_SLOT_BANK=$SLOW_DECODE_SLOT_BANK"
  "DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS=$SNAPSHOT_TOKENS"
  "DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE=$SNAPSHOT_FILE"
  "DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS=$SNAPSHOT_HOT_TOKENS"
  "DS4_FLASH_MOE_RESIDENCY_STATS=$RESIDENCY_STATS"
  "DS4_FLASH_MOE_RESIDENCY_STATS_START=$RESIDENCY_STATS_START"
  "DS4_MXFP4_NATIVE=1"
  "DS4_FLASH_MOE_ANE_PREFILL=1"
  "DS4_METAL_PREFILL_CHUNK=4096"
  "DS4_METAL_KV_TOUCH_ON_DECODE_START=$KV_TOUCH"
  "DS4_METAL_KV_TOUCH_ON_PREFILL_START=$KV_TOUCH"
  "DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1"
  "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO=1"
  "DS4_FLASH_MOE_SLOTWISE_DECODE=1"
)

CMD=(
  "$AGENT"
  -m "$MODEL"
  --ctx "$CTX"
  -n "$TOKENS"
  --temp "$TEMP"
  --nothink
  --moe-slot-bank "$MOE_SLOT_BANK"
)

if [[ "$DEBUG_STATUS" != "0" ]]; then
  CMD+=(--debug-status)
fi
if ((${#EXTRA_ARGS[@]})); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if ((DRY_RUN)); then
  printf 'cd %q\n' "$ROOT"
  if ((TIME_WRAP)); then
    printf '/usr/bin/time -lp env'
  else
    printf 'env'
  fi
  for kv in "${ENV_ARGS[@]}"; do
    printf ' %q' "$kv"
  done
  for arg in "${CMD[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  exit 0
fi

if ((TIME_WRAP)); then
  exec /usr/bin/time -lp env "${ENV_ARGS[@]}" "${CMD[@]}"
else
  exec env "${ENV_ARGS[@]}" "${CMD[@]}"
fi
