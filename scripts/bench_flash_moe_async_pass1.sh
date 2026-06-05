#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4_BIN="${DS4_BIN:-$ROOT/ds4}"
MODEL="${MODEL:-$HOME/Models/flash/dsv4-iq2xxs-expert-major}"
SLOT_BANK="${SLOT_BANK:-64}"
SSD_CACHE="${SSD_CACHE:-}"
CTX="${CTX:-60768}"
N_TOKENS="${N_TOKENS:-500}"
TEMP="${TEMP:-0}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-45}"
PROMPT="${PROMPT:-make a game of Space invaders in PyGame}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_DIR="${OUT_DIR:-/tmp/ds4_flash_moe_async_pass1_$(date +%Y%m%d_%H%M%S)}"
INCLUDE_SPLIT="${INCLUDE_SPLIT:-0}"
INCLUDE_PROFILE="${INCLUDE_PROFILE:-0}"
INCLUDE_PREPROTECT="${INCLUDE_PREPROTECT:-0}"

if [ "$(basename "$DS4_BIN")" = "ds4-agent" ] && [[ " $EXTRA_ARGS " != *" --non-interactive "* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--non-interactive"
fi

mkdir -p "$OUT_DIR"

run_row() {
    local name="$1"
    shift
    local out="$OUT_DIR/${name}.out"
    local log="$OUT_DIR/${name}.log"
    printf '\n== %s ==\n' "$name" | tee -a "$OUT_DIR/summary.txt"
    printf 'start: %s\n' "$(date)" | tee -a "$OUT_DIR/summary.txt"
    local cache_arg_name="--moe-slot-bank"
    local cache_arg_value="$SLOT_BANK"
    if [ -n "$SSD_CACHE" ]; then
        cache_arg_name="--ssd-cache"
        cache_arg_value="$SSD_CACHE"
    fi
    (
        set -x
        if [ -n "$EXTRA_ARGS" ]; then
            env "$@" "$DS4_BIN" \
                -m "$MODEL" \
                "$cache_arg_name" "$cache_arg_value" \
                --ctx "$CTX" \
                --temp "$TEMP" \
                -p "$PROMPT" \
                -n "$N_TOKENS" \
                $EXTRA_ARGS
        else
            env "$@" "$DS4_BIN" \
                -m "$MODEL" \
                "$cache_arg_name" "$cache_arg_value" \
                --ctx "$CTX" \
                --temp "$TEMP" \
                -p "$PROMPT" \
                -n "$N_TOKENS"
        fi
    ) >"$out" 2>"$log"
    shasum -a 256 "$out" | tee -a "$OUT_DIR/summary.txt"
    grep -E 'decode  I/O|generation:|Flash-MoE async handout|miss-hist|slot-bank stats|stable replay' "$log" \
        | tee -a "$OUT_DIR/summary.txt" || true
    printf 'end: %s\n' "$(date)" | tee -a "$OUT_DIR/summary.txt"
}

cooldown() {
    if [ "$COOLDOWN_SECONDS" -gt 0 ]; then
        printf '\n-- cooldown %ss --\n' "$COOLDOWN_SECONDS" | tee -a "$OUT_DIR/summary.txt"
        sleep "$COOLDOWN_SECONDS"
    fi
}

printf 'output dir: %s\n' "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
printf 'git: %s\n' "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)" | tee -a "$OUT_DIR/summary.txt"
printf 'model: %s\nslot_bank: %s\nssd_cache: %s\nctx: %s\nn_tokens: %s\nprompt: %s\nextra_args: %s\n' \
    "$MODEL" "$SLOT_BANK" "$SSD_CACHE" "$CTX" "$N_TOKENS" "$PROMPT" "$EXTRA_ARGS" | tee -a "$OUT_DIR/summary.txt"

run_row grouped_first \
    DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
    DS4_FLASH_MOE_STABLE_REPLAY=0 \
    DS4_FLASH_MOE_BAKED_SLOT_DECODE=0 \
    DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
    DS4_FLASH_MOE_ICB_REPLAY=0
cooldown

run_row stable_slot \
    DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
    DS4_FLASH_MOE_STABLE_REPLAY=1 \
    DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
    DS4_FLASH_MOE_ICB_REPLAY=0
cooldown

run_row async_conservative \
    DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
    DS4_FLASH_MOE_STABLE_REPLAY=1 \
    DS4_FLASH_MOE_ASYNC_HANDOUT=1 \
    DS4_FLASH_MOE_ICB_REPLAY=0
cooldown

if [ "$INCLUDE_SPLIT" != "0" ]; then
    run_row async_forced_split \
        DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
        DS4_FLASH_MOE_STABLE_REPLAY=1 \
        DS4_FLASH_MOE_ASYNC_HANDOUT=1 \
        DS4_FLASH_MOE_ASYNC_HANDOUT_OVERLAP_MISSES=1 \
        DS4_FLASH_MOE_ASYNC_HANDOUT_SPLIT_MISS_MIN=1 \
        DS4_FLASH_MOE_ICB_REPLAY=0
    cooldown
fi

if [ "$INCLUDE_PROFILE" != "0" ]; then
    run_row async_profile \
        DS4_FLASH_MOE_PROFILE=1 \
        DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
        DS4_FLASH_MOE_STABLE_REPLAY=1 \
        DS4_FLASH_MOE_ASYNC_HANDOUT=1 \
        DS4_FLASH_MOE_ICB_REPLAY=0
    cooldown
fi

if [ "$INCLUDE_PREPROTECT" != "0" ]; then
    run_row grouped_preprotect \
        DS4_FLASH_MOE_PREPROTECT_TOPK=1 \
        DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
        DS4_FLASH_MOE_STABLE_REPLAY=0 \
        DS4_FLASH_MOE_BAKED_SLOT_DECODE=0 \
        DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
        DS4_FLASH_MOE_ICB_REPLAY=0
    cooldown
fi

run_row grouped_last \
    DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
    DS4_FLASH_MOE_STABLE_REPLAY=0 \
    DS4_FLASH_MOE_BAKED_SLOT_DECODE=0 \
    DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
    DS4_FLASH_MOE_ICB_REPLAY=0

printf '\nsummary: %s\n' "$OUT_DIR/summary.txt"
