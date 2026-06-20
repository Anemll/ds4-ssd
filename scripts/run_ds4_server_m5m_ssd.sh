#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_ds4_server_m5m_ssd.sh [options] [-- extra ds4-server args...]

Runs ds4-server with the stable M5 Max MXFP4 Flash-MoE grow/recovery profile.

Common options:
  -m, --model PATH              Model path
  --ctx N                       Context size
  -n, --tokens N                Default max output tokens when client omits it
  --moe-slot-bank N             Initial/steady slot bank
  --ssd-cache BYTES|auto        Use server --ssd-cache instead of fixed slots
  --recovery-slot-bank N        Slot bank after compression recovery
  --prefill-chunk N             DS4_METAL_PREFILL_CHUNK. Default: 16384

HTTP server:
  --host HOST                   Bind address
  --port N                      Bind port
  --cors | --no-cors            Enable/disable browser CORS headers
  --trace FILE                  Server trace file

Disk KV cache:
  --kv-disk-dir DIR
  --kv-disk-space-mb N
  --kv-cache-min-tokens N
  --kv-cache-cold-max-tokens N
  --kv-cache-continued-interval-tokens N

Logging / diagnostics:
  --snapshot-file PATH          Slot snapshot log path
  --snapshot-tokens N           Slot snapshot interval in decode tokens
  --snapshot-hot-tokens N       Hot-window size for snapshot tail accounting
  --stdout-log PATH             Tee server stdout/stderr to this log
  --no-stdout-log               Do not tee server stdout/stderr
  --rotate-logs | --no-rotate-logs
                                Rename existing snapshot/stdout logs at startup
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
  --server PATH                 ds4-server binary
  --backend NAME                metal, cuda, or cpu
  --no-time                     Do not wrap with /usr/bin/time -lp
  --dry-run                     Print command/env and exit
  -h, --help

Notes:
  ds4-server does not have agent-only --debug-status, --temp, or --nothink
  flags. Sampling and thinking mode are controlled by each API request.

Examples:
  scripts/run_ds4_server_m5m_ssd.sh --ctx 132768 --port 8000
  scripts/run_ds4_server_m5m_ssd.sh --moe-slot-bank 128 --recovery-slot-bank 96
  scripts/run_ds4_server_m5m_ssd.sh --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER="${DS4_SERVER_BIN:-./ds4-server}"
MODEL="${DS4_MODEL:-/Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash}"
CTX="${DS4_CTX:-132768}"
TOKENS="${DS4_TOKENS:-20000}"
HOST="${DS4_SERVER_HOST:-127.0.0.1}"
PORT="${DS4_SERVER_PORT:-8000}"
CORS="${DS4_SERVER_CORS:-0}"
TRACE="${DS4_SERVER_TRACE:-}"
BACKEND="${DS4_BACKEND:-}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-server-m5m-ssd.lock}"
TIME_WRAP=1
DRY_RUN=0
ROTATE_LOGS="${DS4_RUNNER_ROTATE_LOGS:-1}"
STDOUT_LOG="${DS4_SERVER_STDOUT_LOG:-/tmp/ds4-server-m5m-ssd.stdout.log}"

MOE_SLOT_BANK="${DS4_FLASH_MOE_SLOT_BANK_DEFAULT:-134}"
SSD_CACHE="${DS4_SSD_CACHE:-}"
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
PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-16384}"
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
SNAPSHOT_FILE="${DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE:-/tmp/ds4-slot-server-m5m-ssd.log}"
SNAPSHOT_HOT_TOKENS="${DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS:-128}"
RESIDENCY_STATS="${DS4_FLASH_MOE_RESIDENCY_STATS:-0}"
RESIDENCY_STATS_START="${DS4_FLASH_MOE_RESIDENCY_STATS_START:-1}"

KV_DISK_DIR="${DS4_KV_DISK_DIR:-}"
KV_DISK_SPACE_MB="${DS4_KV_DISK_SPACE_MB:-}"
KV_CACHE_MIN_TOKENS="${DS4_KV_CACHE_MIN_TOKENS:-}"
KV_CACHE_COLD_MAX_TOKENS="${DS4_KV_CACHE_COLD_MAX_TOKENS:-}"
KV_CACHE_CONTINUED_INTERVAL_TOKENS="${DS4_KV_CACHE_CONTINUED_INTERVAL_TOKENS:-}"

EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    -m|--model)
      MODEL="${2:?missing value for $1}"; shift 2 ;;
    --ctx)
      CTX="${2:?missing value for $1}"; shift 2 ;;
    -n|--tokens)
      TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --moe-slot-bank)
      MOE_SLOT_BANK="${2:?missing value for $1}"; SSD_CACHE=""; shift 2 ;;
    --ssd-cache)
      SSD_CACHE="${2:?missing value for $1}"; shift 2 ;;
    --recovery-slot-bank)
      RECOVERY_SLOT_BANK="${2:?missing value for $1}"; shift 2 ;;
    --prefill-chunk)
      PREFILL_CHUNK="${2:?missing value for $1}"; shift 2 ;;
    --host)
      HOST="${2:?missing value for $1}"; shift 2 ;;
    --port)
      PORT="${2:?missing value for $1}"; shift 2 ;;
    --cors)
      CORS=1; shift ;;
    --no-cors)
      CORS=0; shift ;;
    --trace)
      TRACE="${2:?missing value for $1}"; shift 2 ;;
    --kv-disk-dir)
      KV_DISK_DIR="${2:?missing value for $1}"; shift 2 ;;
    --kv-disk-space-mb)
      KV_DISK_SPACE_MB="${2:?missing value for $1}"; shift 2 ;;
    --kv-cache-min-tokens)
      KV_CACHE_MIN_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --kv-cache-cold-max-tokens)
      KV_CACHE_COLD_MAX_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --kv-cache-continued-interval-tokens)
      KV_CACHE_CONTINUED_INTERVAL_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --snapshot-file)
      SNAPSHOT_FILE="${2:?missing value for $1}"; shift 2 ;;
    --snapshot-tokens)
      SNAPSHOT_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --snapshot-hot-tokens)
      SNAPSHOT_HOT_TOKENS="${2:?missing value for $1}"; shift 2 ;;
    --stdout-log)
      STDOUT_LOG="${2:?missing value for $1}"; shift 2 ;;
    --no-stdout-log)
      STDOUT_LOG=""; shift ;;
    --rotate-logs)
      ROTATE_LOGS=1; shift ;;
    --no-rotate-logs)
      ROTATE_LOGS=0; shift ;;
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
    --server)
      SERVER="${2:?missing value for $1}"; shift 2 ;;
    --backend)
      BACKEND="${2:?missing value for $1}"; shift 2 ;;
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

rotate_log_file() {
  local path="$1"
  [[ -n "$path" && "$path" != "-" && -f "$path" ]] || return 0
  local dir base stem ext archive_dir stamp dest
  dir="$(dirname "$path")"
  base="$(basename "$path")"
  if [[ "$base" == *.* ]]; then
    stem="${base%.*}"
    ext=".${base##*.}"
  else
    stem="$base"
    ext=""
  fi
  archive_dir="${DS4_LOG_ARCHIVE_DIR:-$dir/ds4-logs}"
  stamp="$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$archive_dir"
  dest="$archive_dir/${stem}-${stamp}${ext}"
  if [[ -e "$dest" ]]; then
    dest="$archive_dir/${stem}-${stamp}-$$${ext}"
  fi
  mv "$path" "$dest"
  printf 'ds4-runner: archived %s -> %s\n' "$path" "$dest" >&2
}

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
  "DS4_METAL_PREFILL_CHUNK=$PREFILL_CHUNK"
  "DS4_METAL_KV_TOUCH_ON_DECODE_START=$KV_TOUCH"
  "DS4_METAL_KV_TOUCH_ON_PREFILL_START=$KV_TOUCH"
  "DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1"
  "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO=1"
  "DS4_FLASH_MOE_SLOTWISE_DECODE=1"
)

CMD=(
  "$SERVER"
  -m "$MODEL"
  --ctx "$CTX"
  -n "$TOKENS"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$SSD_CACHE" ]]; then
  CMD+=(--ssd-cache "$SSD_CACHE")
else
  CMD+=(--moe-slot-bank "$MOE_SLOT_BANK")
fi
if [[ "$CORS" != "0" ]]; then
  CMD+=(--cors)
fi
if [[ -n "$TRACE" ]]; then
  CMD+=(--trace "$TRACE")
fi
if [[ -n "$BACKEND" ]]; then
  CMD+=(--backend "$BACKEND")
fi
if [[ -n "$KV_DISK_DIR" ]]; then
  CMD+=(--kv-disk-dir "$KV_DISK_DIR")
fi
if [[ -n "$KV_DISK_SPACE_MB" ]]; then
  CMD+=(--kv-disk-space-mb "$KV_DISK_SPACE_MB")
fi
if [[ -n "$KV_CACHE_MIN_TOKENS" ]]; then
  CMD+=(--kv-cache-min-tokens "$KV_CACHE_MIN_TOKENS")
fi
if [[ -n "$KV_CACHE_COLD_MAX_TOKENS" ]]; then
  CMD+=(--kv-cache-cold-max-tokens "$KV_CACHE_COLD_MAX_TOKENS")
fi
if [[ -n "$KV_CACHE_CONTINUED_INTERVAL_TOKENS" ]]; then
  CMD+=(--kv-cache-continued-interval-tokens "$KV_CACHE_CONTINUED_INTERVAL_TOKENS")
fi
if ((${#EXTRA_ARGS[@]})); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if ((DRY_RUN)); then
  printf 'cd %q\n' "$ROOT"
  printf '# snapshot-log %q\n' "$SNAPSHOT_FILE"
  if [[ -n "$STDOUT_LOG" ]]; then
    printf '# stdout-log %q\n' "$STDOUT_LOG"
  fi
  printf '# rotate-logs %q\n' "$ROTATE_LOGS"
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

if [[ "$ROTATE_LOGS" != "0" ]]; then
  rotate_log_file "$SNAPSHOT_FILE"
  if [[ -n "$STDOUT_LOG" && "$STDOUT_LOG" != "$SNAPSHOT_FILE" ]]; then
    rotate_log_file "$STDOUT_LOG"
  fi
fi

if ((TIME_WRAP)); then
  RUN_CMD=(/usr/bin/time -lp env "${ENV_ARGS[@]}" "${CMD[@]}")
else
  RUN_CMD=(env "${ENV_ARGS[@]}" "${CMD[@]}")
fi

if [[ -n "$STDOUT_LOG" ]]; then
  mkdir -p "$(dirname "$STDOUT_LOG")"
  "${RUN_CMD[@]}" 2>&1 | tee -a "$STDOUT_LOG"
  exit "${PIPESTATUS[0]}"
fi

exec "${RUN_CMD[@]}"
