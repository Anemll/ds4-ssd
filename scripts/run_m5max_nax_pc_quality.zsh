#!/bin/zsh
set -euo pipefail

if (( $# != 2 )); then
    print -u2 "usage: $0 smoke|quality|smoke-i8i8|quality-i8i8|smoke-resident|quality-resident RUNROOT"
    exit 64
fi

MODE="$1"
RUNROOT="$2"
REPO="${REPO:-/Users/anemll/Documents/Codex/2026-07-17/ds4-nax-pc-m5max}"
SIDECAR="${SIDECAR:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major-ane-i8-v2}"
MODEL="$SIDECAR/dense/model-dense.gguf"
MANIFEST="$REPO/gguf-tools/quality-testing/data/flash/manifest.tsv"
PROFILE_LOCK="${PROFILE_LOCK:-/tmp/ds4-m5max-exclusive-profile.lock}"

[[ "$(sysctl -n machdep.cpu.brand_string)" == *"M5 Max"* ]]
[[ -x "$REPO/gguf-tools/quality-testing/score_official" ]]
[[ -f "$SIDECAR/manifest.json" ]]
[[ -f "$MODEL" ]]
[[ -f "$MANIFEST" ]]

if pgrep -fl '(^|/)(ds4|ds4-agent|ds4-bench|score_official)( |$)' >/dev/null; then
    print -u2 "another DS4 scorer/benchmark process is active"
    pgrep -fl '(^|/)(ds4|ds4-agent|ds4-bench|score_official)( |$)' >&2 || true
    exit 75
fi

if ! mkdir "$PROFILE_LOCK"; then
    print -u2 "M5 Max profiling lock is already held: $PROFILE_LOCK"
    exit 75
fi
cleanup_profile_lock() {
    command rmdir "$PROFILE_LOCK" >/dev/null 2>&1 || true
}
trap cleanup_profile_lock EXIT HUP INT TERM

mkdir -p "$RUNROOT/smoke" "$RUNROOT/quality"

COMMON_ENV=(
    DS4_ANE=0
    DS4_FLASH_MOE_ANE_PREFILL=0
    DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
    DS4_FLASH_MOE_ANE_PER_CHANNEL=0
    DS4_FLASH_MOE_ANE_REQUIRE=0
    DS4_FLASH_MOE_ANE_I8I8_PREFILL=0
    DS4_FLASH_MOE_ANE_I8I8_FUSED_PREFILL=0
    DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0
    DS4_FLASH_MOE_ANE_I8I8_FULL_FUSED_PREFILL=0
    DS4_FLASH_MOE_HYBRID_PREFILL=0
    DS4_FLASH_MOE_HYBRID_CONCURRENT_PREFILL=0
    DS4_FLASH_MOE_CONCURRENT_PREFILL=0
    DS4_FLASH_MOE_OVERLAP_PREFILL=0
    DS4_RESIDENT_MOE_ANE_NAX_HYBRID=0
    DS4_RESIDENT_MOE_NAX_HALF=0
    DS4_RESIDENT_MOE_NAX_HALF_MAX_TOKENS=0
    DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=0
    DS4_FLASH_MOE_RESIDENT_GROUPED_PREFILL=0
    DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=0
    DS4_RESIDENT_MOE_NAX_DEDUP_PREFILL=0
    DS4_RESIDENT_MOE_MPP_INT8_PREFILL=0
    DS4_RESIDENT_MPP_INT8_PREFILL=0
    DS4_FLASH_MOE_MPP_INT8_ACT=0
    DS4_FLASH_MOE_MPP_I8I8_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_TILED_FUSED_PREFILL=0
    DS4_FLASH_MOE_MPP_I8I8_FULL_FUSED_PREFILL=0
    DS4_FLASH_MOE_MPP_INT8_QSCALE=512
    DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32
    DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32
    DS4_FLASH_MOE_KERNEL_LOG=1
    DS4_FLASH_MOE_SCHED_STATS=1
)

RESIDENT_ENV=(
    DS4_PROFILE=none
    DS4_FLASH_MOE_FORCE_MIXED_SLOT_BANK=1
    DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0
    DS4_FLASH_MOE_SLOT_BANK_RESIDENCY=1
    DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1
    DS4_FLASH_MOE_PRELOAD_SLOT_BANK=1
    DS4_FLASH_MOE_ASYNC_PREAD=0
    DS4_FLASH_MOE_GPU_DEDUP=1
)

cd "$REPO"

check_strict_log() {
    local log="$1"
    local expected_mode="${2:-h-i8-pc}"
    grep -F "Flash-MoE NAX INT8 per-channel active: packed gate/up/down count=8192 (2048/2048/4096) mode=$expected_mode" "$log" >/dev/null
    grep -E 'strict NAX INT8 per-channel M-tile padding active: logical_refs=[1-9][0-9]* dispatch_refs=64 tile=64' "$log" >/dev/null
    grep -F "Flash-MoE MPP prefill engaged: tokens=64 mode=$expected_mode" "$log" >/dev/null
    if grep -En 'NAX INT8 per-channel (scales )?unavailable|using scalar NAX INT8 weight scale|ERROR:|unpadded partial M tile|MPP/NAX int8 partial-tile workaround active|evaluation failed|write failed|reopen failed|reject(ed)?=[1-9][0-9]*|fail(ure)?s?=[1-9][0-9]*' "$log"; then
        print -u2 "strict per-channel run reported failure or fallback"
        exit 1
    fi
}

check_resident_pc_log() {
    local log="$1"
    local min_layer_markers="${2:-43}"
    grep -F "Flash-MoE preloaded mixed slot bank: layers=43 slots=256 identity-mapped experts=256" "$log" >/dev/null
    grep -F "resident sidecar identity prefill active: slots=256 weights=zero-copy-slot-views transient_expert_stage=off" "$log" >/dev/null
    grep -F "resident sidecar NAX INT8 per-channel active: mode=h-i8-pc weights=preloaded-identity-bank scales=F16-resident-record scale_count=8192 transient_expert_stage=off scale_upload=per-expert" "$log" >/dev/null
    local layer_markers
    layer_markers="$(grep -Ec 'resident sidecar prefill layer=[0-9]+ mode=h-i8-pc .*mpp_groups=[1-9][0-9]* .*slot_installs=0 transient_stage_calls=0 transient_stage_bytes=0' "$log")"
    if (( layer_markers < min_layer_markers )); then
        print -u2 "resident per-channel run covered only $layer_markers layer calls (expected at least $min_layer_markers)"
        exit 1
    fi
    if grep -En 'resident sidecar prefill .*slot_installs=[1-9][0-9]*|transient_stage_calls=[1-9][0-9]*|transient_stage_bytes=[1-9][0-9]*|Flash-MoE prefill stage stats calls=[1-9][0-9]*|resident (sidecar )?(NAX INT8 per-channel )?identity slot lost|refusing transient-stage fallback' "$log"; then
        print -u2 "resident per-channel run reported an install or transient stage"
        exit 1
    fi
}

case "$MODE" in
    smoke)
        [[ ! -e "$RUNROOT/smoke/pc.tsv" ]]
        mkdir -p "$RUNROOT/smoke/pc-logits"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --limit 1 --first-token-only \
            --dump-logits-dir "$RUNROOT/smoke/pc-logits" \
            "$SIDECAR" "$MANIFEST" "$RUNROOT/smoke/pc.tsv" \
            >"$RUNROOT/smoke/pc.stdout.log" \
            2>"$RUNROOT/smoke/pc.log"
        check_strict_log "$RUNROOT/smoke/pc.log"
        print "strict M5 Max smoke: PASS"
        ;;
    smoke-i8i8)
        [[ ! -e "$RUNROOT/smoke/i8i8.tsv" ]]
        mkdir -p "$RUNROOT/smoke/i8i8-logits"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_MPP_INT8_ACT=1 \
            DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --limit 1 --first-token-only \
            --dump-logits-dir "$RUNROOT/smoke/i8i8-logits" \
            "$SIDECAR" "$MANIFEST" "$RUNROOT/smoke/i8i8.tsv" \
            >"$RUNROOT/smoke/i8i8.stdout.log" \
            2>"$RUNROOT/smoke/i8i8.log"
        check_strict_log "$RUNROOT/smoke/i8i8.log" i8i8-pc
        print "strict M5 Max i8i8 smoke: PASS"
        ;;
    smoke-resident)
        [[ ! -e "$RUNROOT/smoke/resident-pc.tsv" ]]
        mkdir -p "$RUNROOT/smoke/resident-pc-logits"
        env "${COMMON_ENV[@]}" "${RESIDENT_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --resident --moe-slot-bank 256 \
            --limit 1 --first-token-only \
            --dump-logits-dir "$RUNROOT/smoke/resident-pc-logits" \
            "$SIDECAR" "$MANIFEST" "$RUNROOT/smoke/resident-pc.tsv" \
            >"$RUNROOT/smoke/resident-pc.stdout.log" \
            2>"$RUNROOT/smoke/resident-pc.log"
        check_strict_log "$RUNROOT/smoke/resident-pc.log"
        check_resident_pc_log "$RUNROOT/smoke/resident-pc.log" 43
        print "strict resident M5 Max per-channel smoke: PASS"
        ;;
    quality)
        Q="$RUNROOT/quality"
        [[ ! -e "$Q/gpu.tsv" && ! -e "$Q/scalar-nax.tsv" && ! -e "$Q/pc-nax.tsv" ]]
        mkdir -p "$Q/gpu-logits" "$Q/scalar-nax-logits" "$Q/pc-nax-logits"

        print "quality 1/3: GPU/no-int8"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=1 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=0 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=0 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --no-int8 --dump-logits-dir "$Q/gpu-logits" \
            "$SIDECAR" "$MANIFEST" "$Q/gpu.tsv" \
            >"$Q/gpu.stdout.log" 2>"$Q/gpu.log"

        print "quality 2/3: scalar NAX"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=0 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=0 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --dump-logits-dir "$Q/scalar-nax-logits" \
            "$SIDECAR" "$MANIFEST" "$Q/scalar-nax.tsv" \
            >"$Q/scalar-nax.stdout.log" 2>"$Q/scalar-nax.log"

        print "quality 3/3: strict per-channel NAX"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --dump-logits-dir "$Q/pc-nax-logits" \
            "$SIDECAR" "$MANIFEST" "$Q/pc-nax.tsv" \
            >"$Q/pc-nax.stdout.log" 2>"$Q/pc-nax.log"
        check_strict_log "$Q/pc-nax.log"

        python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
            "$Q/gpu.tsv" "$Q/scalar-nax.tsv" >"$Q/gpu-vs-scalar-nax-scores.txt"
        python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
            "$Q/gpu.tsv" "$Q/pc-nax.tsv" >"$Q/gpu-vs-pc-nax-scores.txt"
        python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
            "$Q/scalar-nax.tsv" "$Q/pc-nax.tsv" >"$Q/scalar-vs-pc-nax-scores.txt"

        python3 "$REPO/scripts/compare_logits.py" \
            "$Q/gpu-logits" "$Q/scalar-nax-logits" \
            --per-case "$Q/gpu-vs-scalar-nax-logits.tsv" \
            >"$Q/gpu-vs-scalar-nax-logits.txt"
        python3 "$REPO/scripts/compare_logits.py" \
            "$Q/gpu-logits" "$Q/pc-nax-logits" \
            --per-case "$Q/gpu-vs-pc-nax-logits.tsv" \
            >"$Q/gpu-vs-pc-nax-logits.txt"
        python3 "$REPO/scripts/compare_logits.py" \
            "$Q/scalar-nax-logits" "$Q/pc-nax-logits" \
            --per-case "$Q/scalar-vs-pc-nax-logits.tsv" \
            >"$Q/scalar-vs-pc-nax-logits.txt"
        print "M5 Max 100-case quality matrix: PASS"
        ;;
    quality-i8i8)
        Q="$RUNROOT/quality"
        [[ -e "$Q/gpu.tsv" && -d "$Q/gpu-logits" ]]
        [[ ! -e "$Q/i8i8-nax.tsv" ]]
        mkdir -p "$Q/i8i8-nax-logits"

        print "quality i8i8: strict per-channel W8A8 NAX"
        env "${COMMON_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_MPP_INT8_ACT=1 \
            DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --dump-logits-dir "$Q/i8i8-nax-logits" \
            "$SIDECAR" "$MANIFEST" "$Q/i8i8-nax.tsv" \
            >"$Q/i8i8-nax.stdout.log" 2>"$Q/i8i8-nax.log"
        check_strict_log "$Q/i8i8-nax.log" i8i8-pc

        python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
            "$Q/gpu.tsv" "$Q/i8i8-nax.tsv" >"$Q/gpu-vs-i8i8-nax-scores.txt"
        python3 "$REPO/scripts/compare_logits.py" \
            "$Q/gpu-logits" "$Q/i8i8-nax-logits" \
            --per-case "$Q/gpu-vs-i8i8-nax-logits.tsv" \
            >"$Q/gpu-vs-i8i8-nax-logits.txt"
        print "M5 Max 100-case i8i8 quality arm: PASS"
        ;;
    quality-resident)
        Q="$RUNROOT/quality"
        [[ -e "$Q/pc-nax.tsv" && -d "$Q/pc-nax-logits" ]]
        [[ ! -e "$Q/resident-pc-nax.tsv" ]]
        mkdir -p "$Q/resident-pc-nax-logits"

        print "quality resident: strict per-channel NAX from preloaded identity bank"
        env "${COMMON_ENV[@]}" "${RESIDENT_ENV[@]}" \
            DS4_NO_INT8=0 \
            DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
            DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
            "$REPO/gguf-tools/quality-testing/score_official" \
            --ctx 4096 --resident --moe-slot-bank 256 \
            --dump-logits-dir "$Q/resident-pc-nax-logits" \
            "$SIDECAR" "$MANIFEST" "$Q/resident-pc-nax.tsv" \
            >"$Q/resident-pc-nax.stdout.log" \
            2>"$Q/resident-pc-nax.log"
        check_strict_log "$Q/resident-pc-nax.log"
        check_resident_pc_log "$Q/resident-pc-nax.log" 4300

        python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
            "$Q/pc-nax.tsv" "$Q/resident-pc-nax.tsv" \
            >"$Q/streaming-vs-resident-pc-nax-scores.txt"
        python3 "$REPO/scripts/compare_logits.py" \
            "$Q/pc-nax-logits" "$Q/resident-pc-nax-logits" \
            --per-case "$Q/streaming-vs-resident-pc-nax-logits.tsv" \
            >"$Q/streaming-vs-resident-pc-nax-logits.txt"
        print "M5 Max 100-case resident per-channel quality arm: PASS"
        ;;
    *)
        print -u2 "unknown mode: $MODE"
        exit 64
        ;;
esac
