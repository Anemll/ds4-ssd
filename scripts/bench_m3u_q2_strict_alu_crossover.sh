#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIDECAR=${SIDECAR:-/Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2}
PROMPT=${PROMPT:-$ROOT/tests/test-vectors/prompts/sidecar_16k.txt}
OUT_DIR=${OUT_DIR:-$ROOT/bench-results/m3u-q2-strict-alu-crossover}
FRONTIERS=${FRONTIERS:-"128 256 384 512 768 1024 1536 2048 3072 4000"}
SLOTS=${SLOTS:-32}
PREFILL_CHUNK=${PREFILL_CHUNK:-4096}
COOLDOWN_SEC=${COOLDOWN_SEC:-0}
LOCK_DIR=${DS4_M3U_PROFILE_LOCK:-/tmp/ds4-m3u-profile.lock}
MIN_FREE_PERCENT=${DS4_M3U_MIN_FREE_PERCENT:-50}
MIN_DISK_MIB=${DS4_M3U_MIN_DISK_MIB:-256}
SWEEP_IN_ONE_PROCESS=${SWEEP_IN_ONE_PROCESS:-0}
IN_PROCESS_ABBA=${IN_PROCESS_ABBA:-0}
SWEEP_START=${SWEEP_START:-128}
SWEEP_MAX=${SWEEP_MAX:-4000}
SWEEP_STEP_MUL=${SWEEP_STEP_MUL:-2}

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
    [[ "$MIN_FREE_PERCENT" =~ ^[0-9]+$ ]] && (( MIN_FREE_PERCENT <= 100 )) || {
        printf 'Invalid DS4_M3U_MIN_FREE_PERCENT: %s\n' "$MIN_FREE_PERCENT" >&2
        return 1
    }
    report=$(memory_pressure -Q) || return 1
    free_percent=$(awk '/System-wide memory free percentage:/ {gsub(/%/, "", $5); print $5; exit}' <<< "$report")
    [[ "$free_percent" =~ ^[0-9]+$ ]] || {
        printf 'Unable to parse M3 Ultra memory pressure:\n%s\n' "$report" >&2
        return 1
    }
    if (( free_percent < MIN_FREE_PERCENT )); then
        printf 'Refusing M3 Ultra profiling at %s%% free memory (minimum %s%%)\n' \
            "$free_percent" "$MIN_FREE_PERCENT" >&2
        return 1
    fi
}

assert_m3u_ready() {
    local disk_free_mib
    [[ -d "$LOCK_DIR" ]] || {
        printf 'Canonical M3 Ultra profile lock was lost: %s\n' "$LOCK_DIR" >&2
        return 1
    }
    assert_idle
    assert_memory_ready
    disk_free_mib=$(df -Pm "$ROOT" | awk 'NR == 2 {print $4}')
    [[ "$MIN_DISK_MIB" =~ ^[0-9]+$ && "$disk_free_mib" =~ ^[0-9]+$ ]] || return 1
    if (( disk_free_mib < MIN_DISK_MIB )); then
        printf 'Refusing M3 Ultra profiling with %s MiB disk free (minimum %s MiB)\n' \
            "$disk_free_mib" "$MIN_DISK_MIB" >&2
        return 1
    fi
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
[[ -x "$ROOT/ds4-bench" && -d "$SIDECAR" && -f "$PROMPT" ]] || {
    printf 'Missing ds4-bench, sidecar, or prompt input\n' >&2
    exit 1
}
mkdir -p "$OUT_DIR"

# Keep the production sidecar/I/O setup, but make compute strictly M3U Metal
# GPU/ALU. In particular, an M3U build may compile the experimental Metal-4
# direct-RHS library; force it off so the absolute numbers cannot be labeled
# NAX by the backend resolver.
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
    DS4_METAL_PREFILL_CHUNK="$PREFILL_CHUNK"
    DS4_FLASH_MOE_SLOT_BANK_SLOTS="$SLOTS"
    DS4_FLASH_MOE_KERNEL_LOG=1
)

run_bench() {
    local seq=$1
    local frontier=$2
    local arm=$3
    local force_mm=0
    case "$arm" in
        current) ;;
        grouped) force_mm=1 ;;
        *) printf 'Unknown arm: %s\n' "$arm" >&2; return 2 ;;
    esac

    local arm_env=()
    if (( force_mm != 0 )); then
        arm_env+=(DS4_METAL_FLASH_MOE_FORCE_MM_ID=1)
    fi

    assert_m3u_ready
    printf 'run=%s frontier=%s arm=%s\n' "$seq" "$frontier" "$arm"
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        "${arm_env[@]}" \
        DS4_MATMUL_PATH_CSV="$OUT_DIR/${seq}_${frontier}_${arm}.matmul.csv" \
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
            --csv "$OUT_DIR/${seq}_${frontier}_${arm}.csv" \
            2>"$OUT_DIR/${seq}_${frontier}_${arm}.log"
    if (( COOLDOWN_SEC > 0 )); then
        sleep "$COOLDOWN_SEC"
    fi
}

run_sweep() {
    local seq=$1
    local arm=$2
    local arm_env=()
    case "$arm" in
        current) ;;
        grouped) arm_env+=(DS4_METAL_FLASH_MOE_FORCE_MM_ID=1) ;;
        *) printf 'Unknown arm: %s\n' "$arm" >&2; return 2 ;;
    esac

    assert_m3u_ready
    printf 'run=%s sweep=%s..%s arm=%s\n' "$seq" "$SWEEP_START" "$SWEEP_MAX" "$arm"
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        "${arm_env[@]}" \
        DS4_MATMUL_PATH_CSV="$OUT_DIR/${seq}_${arm}.matmul.csv" \
        "$ROOT/ds4-bench" \
            -m "$SIDECAR" \
            --metal \
            --prompt-file "$PROMPT" \
            --moe-sidecar "$SIDECAR" \
            --moe-mode slot-bank \
            --moe-slot-bank "$SLOTS" \
            --ctx-start "$SWEEP_START" \
            --ctx-max "$SWEEP_MAX" \
            --ctx-alloc 16384 \
            --step-mul "$SWEEP_STEP_MUL" \
            --full-prefill-each-frontier \
            --gen-tokens 1 \
            --csv "$OUT_DIR/${seq}_${arm}.csv" \
            2>"$OUT_DIR/${seq}_${arm}.log"
    if (( COOLDOWN_SEC > 0 )); then sleep "$COOLDOWN_SEC"; fi
}

run_internal_abba() {
    assert_m3u_ready
    printf 'run=in-process-abba sweep=%s..%s\n' "$SWEEP_START" "$SWEEP_MAX"
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        -u DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MIN_TOKENS \
        -u DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MAX_TOKENS \
        "${common_env[@]}" \
        DS4_MATMUL_PATH_CSV="$OUT_DIR/in_process_abba.matmul.csv" \
        "$ROOT/ds4-bench" \
            -m "$SIDECAR" \
            --metal \
            --prompt-file "$PROMPT" \
            --moe-sidecar "$SIDECAR" \
            --moe-mode slot-bank \
            --moe-slot-bank "$SLOTS" \
            --ctx-start "$SWEEP_START" \
            --ctx-max "$SWEEP_MAX" \
            --ctx-alloc 16384 \
            --step-mul "$SWEEP_STEP_MUL" \
            --prefill-abba-force-mm-id \
            --gen-tokens 1 \
            --csv "$OUT_DIR/in_process_abba.csv" \
            2>"$OUT_DIR/in_process_abba.log"
}

if (( IN_PROCESS_ABBA != 0 )); then
    run_internal_abba
    printf 'Results: %s\n' "$OUT_DIR"
    exit 0
fi

if (( SWEEP_IN_ONE_PROCESS != 0 )); then
    run_sweep 01 current
    run_sweep 02 grouped
    run_sweep 03 grouped
    run_sweep 04 current
    printf 'Results: %s\n' "$OUT_DIR"
    exit 0
fi

# Warm both routed kernels before measuring the smallest frontier. This avoids
# assigning first-use Metal compilation/page-fault cost to A1.
run_bench 00a 128 current
run_bench 00b 128 grouped

for frontier in $FRONTIERS; do
    run_bench 01 "$frontier" current
    run_bench 02 "$frontier" grouped
    run_bench 03 "$frontier" grouped
    run_bench 04 "$frontier" current
done

printf 'Results: %s\n' "$OUT_DIR"
