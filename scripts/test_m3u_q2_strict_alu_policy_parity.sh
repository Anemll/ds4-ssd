#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIDECAR=${SIDECAR:-/Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2}
PROMPT=${PROMPT:-$ROOT/tests/test-vectors/prompts/long_code_audit.txt}
OUT_DIR=${OUT_DIR:-$ROOT/bench-results/m3u-q2-strict-alu-policy-parity}
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
    DS4_FLASH_MOE_SLOT_BANK_SLOTS=32
    DS4_FLASH_MOE_KERNEL_LOG=1
)

run_arm() {
    local arm=$1
    local arm_env=()
    case "$arm" in
        policy)
            arm_env+=(DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MIN_TOKENS=128)
            arm_env+=(DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MAX_TOKENS=8191)
            ;;
        force)
            arm_env+=(DS4_METAL_FLASH_MOE_FORCE_MM_ID=1)
            ;;
        *) return 2 ;;
    esac
    assert_m3u_ready
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MIN_TOKENS \
        -u DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MAX_TOKENS \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        "${arm_env[@]}" \
        "$ROOT/ds4" \
            -m "$SIDECAR" \
            --metal \
            --prompt-file "$PROMPT" \
            --moe-sidecar "$SIDECAR" \
            --moe-mode slot-bank \
            --moe-slot-bank 32 \
            --ctx 8192 \
            -sys '' \
            --nothink \
            --temp 0 \
            -n 16 \
            --logprobs-top-k 20 \
            --dump-logprobs "$OUT_DIR/${arm}.json" \
            >"$OUT_DIR/${arm}.stdout" \
            2>"$OUT_DIR/${arm}.log"
}

run_arm policy
run_arm force
cmp "$OUT_DIR/policy.json" "$OUT_DIR/force.json"
printf 'Strict ALU policy parity: byte-identical\nResults: %s\n' "$OUT_DIR"
