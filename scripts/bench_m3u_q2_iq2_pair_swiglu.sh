#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIDECAR=${SIDECAR:-/Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2}
PROMPT=${PROMPT:-$ROOT/tests/test-vectors/prompts/sidecar_16k.txt}
PARITY_PROMPT=${PARITY_PROMPT:-$ROOT/tests/test-vectors/prompts/long_code_audit.txt}
OUT_DIR=${OUT_DIR:-$ROOT/bench-results/m3u-q2-iq2-pair-swiglu}
SLOTS=${SLOTS:-32}
COOLDOWN_SEC=${COOLDOWN_SEC:-0}
RUN_FUSION_CONTROL=${RUN_FUSION_CONTROL:-1}
RUN_PARITY=${RUN_PARITY:-1}
RUN_METAL_TESTS=${RUN_METAL_TESTS:-1}
LOCK_DIR=${DS4_M3U_PROFILE_LOCK:-/tmp/ds4-m3u-profile.lock}

assert_idle() {
    local active
    active=$(ps -axo pid=,comm=,args= | awk '
        $2 ~ /(^|\/)ds4(-agent|-server|-bench)?$/ { print }
    ')
    if [[ -n "$active" ]]; then
        printf 'Refusing to overlap M3 Ultra profiling; active DS4 process(es):\n%s\n' "$active" >&2
        return 1
    fi
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    printf 'M3 Ultra profile lock already exists: %s\n' "$LOCK_DIR" >&2
    exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

assert_idle
[[ -x "$ROOT/ds4" && -x "$ROOT/ds4-bench" ]] || {
    printf 'Build ds4 and ds4-bench first in %s\n' "$ROOT" >&2
    exit 1
}
[[ -d "$SIDECAR" && -f "$PROMPT" && -f "$PARITY_PROMPT" ]] || {
    printf 'Missing sidecar or prompt input\nSIDECAR=%s\nPROMPT=%s\nPARITY_PROMPT=%s\n' \
        "$SIDECAR" "$PROMPT" "$PARITY_PROMPT" >&2
    exit 1
}
mkdir -p "$OUT_DIR"

common_env=(
    DS4_ANE=0
    DS4_FLASH_MOE_ANE_PREFILL=0
    DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
    DS4_FLASH_MOE_OVERLAP_PREFILL=0
    DS4_FLASH_MOE_OVERLAP_SCHEDULER=0
    DS4_METAL_PREFILL_CHUNK=8192
    DS4_FLASH_MOE_SLOT_BANK_SLOTS="$SLOTS"
    DS4_FLASH_MOE_KERNEL_LOG=1
)

run_bench() {
    local seq=$1
    local arm=$2
    local frontier=$3
    local force_mm=0
    local fusion=0
    local arm_env=()
    case "$arm" in
        current)  ;;
        unfused) force_mm=1 ;;
        fused)   force_mm=1; fusion=1 ;;
        *) printf 'Unknown arm: %s\n' "$arm" >&2; return 2 ;;
    esac
    if (( force_mm != 0 )); then
        arm_env+=(DS4_METAL_FLASH_MOE_FORCE_MM_ID=1)
    fi
    if (( fusion != 0 )); then
        arm_env+=(DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU=1)
    fi

    assert_idle
    printf 'run=%s frontier=%s arm=%s force_mm=%s fusion=%s\n' \
        "$seq" "$frontier" "$arm" "$force_mm" "$fusion"
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        "${arm_env[@]}" \
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

run_parity() {
    local arm=$1
    local force_mm=0
    local fusion=0
    local arm_env=()
    case "$arm" in
        current)  ;;
        unfused) force_mm=1 ;;
        fused)   force_mm=1; fusion=1 ;;
        *) printf 'Unknown arm: %s\n' "$arm" >&2; return 2 ;;
    esac
    if (( force_mm != 0 )); then
        arm_env+=(DS4_METAL_FLASH_MOE_FORCE_MM_ID=1)
    fi
    if (( fusion != 0 )); then
        arm_env+=(DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU=1)
    fi

    assert_idle
    env -u DS4_METAL_FLASH_MOE_FORCE_MM_ID \
        -u DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU \
        "${common_env[@]}" \
        "${arm_env[@]}" \
        "$ROOT/ds4" \
            -m "$SIDECAR" \
            --metal \
            --prompt-file "$PARITY_PROMPT" \
            --moe-sidecar "$SIDECAR" \
            --moe-mode slot-bank \
            --moe-slot-bank "$SLOTS" \
            --ctx 8192 \
            -sys '' \
            --nothink \
            --temp 0 \
            -n 16 \
            --logprobs-top-k 20 \
            --dump-logprobs "$OUT_DIR/parity_${arm}.json" \
            >"$OUT_DIR/parity_${arm}.stdout" \
            2>"$OUT_DIR/parity_${arm}.log"
}

if (( RUN_METAL_TESTS != 0 )); then
    assert_idle
    "$ROOT/ds4_test" --metal-kernels \
        >"$OUT_DIR/metal-kernels.stdout" \
        2>"$OUT_DIR/metal-kernels.stderr"
fi

if (( RUN_PARITY != 0 )); then
    run_parity current
    run_parity unfused
    run_parity fused
    if cmp -s "$OUT_DIR/parity_unfused.json" "$OUT_DIR/parity_fused.json"; then
        printf 'PARITY grouped-unfused vs fused: byte-identical\n'
    else
        printf 'PARITY grouped-unfused vs fused: DIFFERENT\n' >&2
        exit 3
    fi
fi

# Outcome comparison requested by the task: current, fused, fused, current at
# each exact frontier. The second ABBA holds grouped MM constant and changes
# only the fusion bit.
for frontier in 128 512 2048 4000 8192; do
    run_bench 01 current "$frontier"
    run_bench 02 fused "$frontier"
    run_bench 03 fused "$frontier"
    run_bench 04 current "$frontier"

    if (( RUN_FUSION_CONTROL != 0 )); then
        run_bench 05 unfused "$frontier"
        run_bench 06 fused "$frontier"
        run_bench 07 fused "$frontier"
        run_bench 08 unfused "$frontier"
    fi
done

printf 'Results: %s\n' "$OUT_DIR"
