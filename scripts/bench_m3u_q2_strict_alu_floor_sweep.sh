#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIDECAR=${SIDECAR:-/Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2}
PROMPT=${PROMPT:-$ROOT/tests/test-vectors/prompts/sidecar_16k.txt}
OUT_DIR=${OUT_DIR:-$ROOT/bench-results/m3u-q2-strict-alu-floor-sweep}
FRONTIERS=${FRONTIERS:-"128 512 1024 2048 4000"}
FLOORS=${FLOORS:-"16 24 32 48 64"}
SLOTS=${SLOTS:-32}
COOLDOWN_SEC=${COOLDOWN_SEC:-0}
LOCK_DIR=${DS4_M3U_PROFILE_LOCK:-/tmp/ds4-m3u-profile.lock}
MIN_FREE_PERCENT=${DS4_M3U_MIN_FREE_PERCENT:-50}
MIN_DISK_MIB=${DS4_M3U_MIN_DISK_MIB:-256}

assert_idle() {
    local active
    active=$(ps -axo pid=,comm=,args= | awk '
        $2 ~ /(^|\/)ds4(-agent|-server|-bench)?$/ ||
        $2 ~ /(^|\/)ane_[[:alnum:]_]*smoke$/ ||
        $0 ~ /(^|[[:space:]])(\.\/)?tests\/ane_[^[:space:]]*smoke([[:space:]]|$)/ {
            print
        }
    ')
    if [[ -n "$active" ]]; then
        printf 'Refusing to overlap M3 Ultra profiling; active DS4/ANE-smoke process(es):\n%s\n' "$active" >&2
        return 1
    fi
}

assert_memory_ready() {
    local report free_percent
    [[ "$MIN_FREE_PERCENT" =~ ^[0-9]+$ ]] && (( MIN_FREE_PERCENT <= 100 )) || return 1
    report=$(memory_pressure -Q) || return 1
    free_percent=$(awk '/System-wide memory free percentage:/ {gsub(/%/, "", $5); print $5; exit}' <<< "$report")
    [[ "$free_percent" =~ ^[0-9]+$ ]] || return 1
    if (( free_percent < MIN_FREE_PERCENT )); then
        printf 'Refusing M3 Ultra profiling at %s%% free memory (minimum %s%%)\n' \
            "$free_percent" "$MIN_FREE_PERCENT" >&2
        return 1
    fi
}

assert_m3u_ready() {
    local disk_free_mib
    [[ -d "$LOCK_DIR" ]] || return 1
    assert_idle
    assert_memory_ready
    disk_free_mib=$(df -Pm "$ROOT" | awk 'NR == 2 {print $4}')
    [[ "$MIN_DISK_MIB" =~ ^[0-9]+$ && "$disk_free_mib" =~ ^[0-9]+$ ]] || return 1
    (( disk_free_mib >= MIN_DISK_MIB )) || {
        printf 'Refusing M3 Ultra profiling with %s MiB disk free (minimum %s MiB)\n' \
            "$disk_free_mib" "$MIN_DISK_MIB" >&2
        return 1
    }
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    printf 'M3 Ultra profile lock already exists: %s\n' "$LOCK_DIR" >&2
    exit 1
fi
cleanup_lock() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assert_m3u_ready
[[ -x "$ROOT/ds4-bench" && -d "$SIDECAR" && -f "$PROMPT" ]] || exit 1
mkdir -p "$OUT_DIR"

common_env=(
    DS4_ANE=0
    DS4_GPU_DENSE_NAX=0
    DS4_GPU_DENSE_I8=0
    DS4_GPU_INDEXER_NAX=0
    DS4_MPP_NAX_FORCE_NON_M5=0
    DS4_FLASH_MOE_NAX_FORCE_NON_M5=0
    DS4_FLASH_MOE_MPP_FORCE_NON_M5=0
    DS4_FLASH_MOE_ANE_PREFILL=0
    DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
    DS4_FLASH_MOE_HYBRID_PREFILL=0
    DS4_FLASH_MOE_CONCURRENT_PREFILL=0
    DS4_FLASH_MOE_OVERLAP_PREFILL=0
    DS4_FLASH_MOE_OVERLAP_SCHEDULER=0
    DS4_FLASH_MOE_MPP_INT8_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_FULL_FUSED_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_TILED_FUSED_PREFILL=0
    DS4_RESIDENT_MOE_MPP_INT8_PREFILL=0
    DS4_RESIDENT_MOE_NAX_INT8_PREFILL=0
    DS4_METAL_PREFILL_CHUNK=4096
    DS4_FLASH_MOE_SLOT_BANK_SLOTS="$SLOTS"
    DS4_FLASH_MOE_KERNEL_LOG=1
)

run_floor() {
    local seq=$1
    local frontier=$2
    local floor=$3
    assert_m3u_ready
    printf 'run=%s frontier=%s floor=%s\n' "$seq" "$frontier" "$floor"
    env -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        DS4_METAL_FLASH_MOE_FORCE_MM_ID=1 \
        DS4_METAL_MOE_MM_MIN_REFS="$floor" \
        "$ROOT/ds4-bench" \
            -m "$SIDECAR" \
            --metal \
            --prompt-file "$PROMPT" \
            --moe-sidecar "$SIDECAR" \
            --moe-mode slot-bank \
            --moe-slot-bank "$SLOTS" \
            --ctx-start "$frontier" \
            --ctx-max "$frontier" \
            --ctx-alloc 16384 \
            --full-prefill-each-frontier \
            --gen-tokens 1 \
            --csv "$OUT_DIR/${seq}_${frontier}_r${floor}.csv" \
            2>"$OUT_DIR/${seq}_${frontier}_r${floor}.log"
    if (( COOLDOWN_SEC > 0 )); then sleep "$COOLDOWN_SEC"; fi
}

read -r -a floor_values <<< "$FLOORS"
run_floor 00 128 32
for frontier in $FRONTIERS; do
    seq=1
    for floor in "${floor_values[@]}"; do
        printf -v label '%02d' "$seq"
        run_floor "$label" "$frontier" "$floor"
        ((seq += 1))
    done
    for ((i=${#floor_values[@]}-1; i>=0; i--)); do
        printf -v label '%02d' "$seq"
        run_floor "$label" "$frontier" "${floor_values[i]}"
        ((seq += 1))
    done
done

printf 'Results: %s\n' "$OUT_DIR"
