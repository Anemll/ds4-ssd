#!/usr/bin/env bash
set -euo pipefail

# Run ds4 outside any inherited Codex filesystem-sandbox marker so Metal stays visible.
unset CODEX_SANDBOX

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DRAFT="${DRAFT:-/Users/anemll/Models/DSv4-Flash-DSpark-draft}"
PROMPT="${PROMPT:-Make a game of Space Invader in Pygame}"
N="${N:-160}"
CTX="${CTX:-4096}"
BUDGETS="${BUDGETS:-1 2 3 4 5}"
OUT_DIR="${OUT_DIR:-bench-results/dspark_phase0_$(date +%Y%m%d_%H%M%S)}"
RUN_BASELINE="${RUN_BASELINE:-1}"
ROUTE_OVERLAP="${ROUTE_OVERLAP:-1}"
DISPATCH_PROFILE="${DISPATCH_PROFILE:-1}"
BLOCK_TIMING="${BLOCK_TIMING:-1}"
RESIDENT="${RESIDENT:-1}"
PLAN_C_BACKENDS="${PLAN_C_BACKENDS:-}"
TARGET_FORWARD_PROFILE="${TARGET_FORWARD_PROFILE:-1}"
SHARED_PREFIX_PROFILE="${SHARED_PREFIX_PROFILE:-1}"
SHARED_PREFIX_PROFILE_ALL="${SHARED_PREFIX_PROFILE_ALL:-0}"
ATTN_ROWS_SHAPE_PROFILE="${ATTN_ROWS_SHAPE_PROFILE:-0}"
ATTN_ROWS_SHAPE_PROFILE_ALL="${ATTN_ROWS_SHAPE_PROFILE_ALL:-0}"

mkdir -p "$OUT_DIR"

resident_args=()
if [[ "$RESIDENT" != "0" ]]; then
    resident_args+=(--resident)
fi

common_args=(
    -m "$BASE"
    --temp 0
    --nothink
    -n "$N"
    -p "$PROMPT"
    -c "$CTX"
    "${resident_args[@]}"
)

extract_generation_tps() {
    sed -nE 's/.*generation: ([0-9.]+) t\/s.*/\1/p' "$1" | tail -1
}

write_run_manifest() {
    {
        printf 'BASE=%s\n' "$BASE"
        printf 'DRAFT=%s\n' "$DRAFT"
        printf 'PROMPT=%s\n' "$PROMPT"
        printf 'N=%s\n' "$N"
        printf 'CTX=%s\n' "$CTX"
        printf 'BUDGETS=%s\n' "$BUDGETS"
        printf 'RUN_BASELINE=%s\n' "$RUN_BASELINE"
        printf 'ROUTE_OVERLAP=%s\n' "$ROUTE_OVERLAP"
        printf 'DISPATCH_PROFILE=%s\n' "$DISPATCH_PROFILE"
        printf 'BLOCK_TIMING=%s\n' "$BLOCK_TIMING"
        printf 'RESIDENT=%s\n' "$RESIDENT"
        printf 'PLAN_C_BACKENDS=%s\n' "$PLAN_C_BACKENDS"
        printf 'TARGET_FORWARD_PROFILE=%s\n' "$TARGET_FORWARD_PROFILE"
        printf 'SHARED_PREFIX_PROFILE=%s\n' "$SHARED_PREFIX_PROFILE"
        printf 'SHARED_PREFIX_PROFILE_ALL=%s\n' "$SHARED_PREFIX_PROFILE_ALL"
        printf 'ATTN_ROWS_SHAPE_PROFILE=%s\n' "$ATTN_ROWS_SHAPE_PROFILE"
        printf 'ATTN_ROWS_SHAPE_PROFILE_ALL=%s\n' "$ATTN_ROWS_SHAPE_PROFILE_ALL"
    } > "$OUT_DIR/manifest.env"
}

write_run_manifest

baseline_tps="${BASELINE_TPS:-}"
baseline_err=""
if [[ "$RUN_BASELINE" != "0" ]]; then
    echo "dspark phase0: baseline -> $OUT_DIR/baseline.err" >&2
    ./ds4 "${common_args[@]}" > "$OUT_DIR/baseline.out" 2> "$OUT_DIR/baseline.err"
    baseline_err="$OUT_DIR/baseline.err"
    baseline_tps="$(extract_generation_tps "$baseline_err")"
    if [[ -z "$baseline_tps" ]]; then
        echo "dspark phase0: could not parse baseline generation t/s" >&2
        exit 1
    fi
fi

if [[ -z "$baseline_tps" ]]; then
    echo "dspark phase0: set BASELINE_TPS or keep RUN_BASELINE=1" >&2
    exit 1
fi

echo "dspark phase0: baseline_tps=$baseline_tps" >&2
printf '%s\n' "$baseline_tps" > "$OUT_DIR/baseline_tps.txt"

logs=()
if [[ -n "$baseline_err" ]]; then
    logs+=("$baseline_err")
fi

for budget in $BUDGETS; do
    echo "dspark phase0: draft budget=$budget -> $OUT_DIR/dspark_b${budget}.err" >&2
    env_args=(
        "DS4_DSPARK_PERF=1"
        "DS4_DSPARK_BASELINE_TPS=$baseline_tps"
    )
    if [[ "$ROUTE_OVERLAP" != "0" ]]; then
        env_args+=("DS4_DSPARK_ROUTE_OVERLAP_LOG=1")
    fi
    if [[ "$DISPATCH_PROFILE" != "0" ]]; then
        env_args+=("DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1")
    fi
    if [[ "$BLOCK_TIMING" != "0" ]]; then
        env_args+=("DS4_DSPARK_BLOCK_TIMING=1")
    fi
    if [[ "$SHARED_PREFIX_PROFILE" != "0" ]]; then
        env_args+=("DS4_DSPARK_SHARED_PREFIX_PROFILE=1")
    fi
    if [[ "$SHARED_PREFIX_PROFILE_ALL" != "0" ]]; then
        env_args+=("DS4_DSPARK_SHARED_PREFIX_PROFILE_ALL=1")
    fi
    if [[ "$ATTN_ROWS_SHAPE_PROFILE" != "0" ]]; then
        env_args+=("DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE=1")
    fi
    if [[ "$ATTN_ROWS_SHAPE_PROFILE_ALL" != "0" ]]; then
        env_args+=("DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE_ALL=1")
    fi

    env "${env_args[@]}" \
        ./ds4 "${common_args[@]}" \
        --draft dspark \
        --draft-path "$DRAFT" \
        --draft-verify "$budget" \
        > "$OUT_DIR/dspark_b${budget}.out" \
        2> "$OUT_DIR/dspark_b${budget}.err"
    logs+=("$OUT_DIR/dspark_b${budget}.err")
done

for backend in $PLAN_C_BACKENDS; do
    safe_backend="${backend//[^A-Za-z0-9_.-]/_}"
    for budget in $BUDGETS; do
        echo "dspark phase0: unified backend=$backend budget=$budget -> $OUT_DIR/dspark_unified_${safe_backend}_b${budget}.err" >&2
        env_args=(
            "DS4_DSPARK_PERF=1"
            "DS4_DSPARK_BASELINE_TPS=$baseline_tps"
            "DS4_TARGET_FORWARD_UNIFIED_BACKEND=$backend"
        )
        if [[ "$TARGET_FORWARD_PROFILE" != "0" ]]; then
            env_args+=("DS4_TARGET_FORWARD_UNIFIED_PROFILE=1")
        fi
        if [[ "$ROUTE_OVERLAP" != "0" ]]; then
            env_args+=("DS4_DSPARK_ROUTE_OVERLAP_LOG=1")
        fi
        if [[ "$DISPATCH_PROFILE" != "0" ]]; then
            env_args+=("DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1")
        fi
        if [[ "$BLOCK_TIMING" != "0" ]]; then
            env_args+=("DS4_DSPARK_BLOCK_TIMING=1")
        fi
        if [[ "$SHARED_PREFIX_PROFILE" != "0" ]]; then
            env_args+=("DS4_DSPARK_SHARED_PREFIX_PROFILE=1")
        fi
        if [[ "$SHARED_PREFIX_PROFILE_ALL" != "0" ]]; then
            env_args+=("DS4_DSPARK_SHARED_PREFIX_PROFILE_ALL=1")
        fi
        if [[ "$ATTN_ROWS_SHAPE_PROFILE" != "0" ]]; then
            env_args+=("DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE=1")
        fi
        if [[ "$ATTN_ROWS_SHAPE_PROFILE_ALL" != "0" ]]; then
            env_args+=("DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE_ALL=1")
        fi

        env "${env_args[@]}" \
            ./ds4 "${common_args[@]}" \
            --draft dspark \
            --draft-path "$DRAFT" \
            --draft-mode unified \
            --draft-verify "$budget" \
            > "$OUT_DIR/dspark_unified_${safe_backend}_b${budget}.out" \
            2> "$OUT_DIR/dspark_unified_${safe_backend}_b${budget}.err"
        logs+=("$OUT_DIR/dspark_unified_${safe_backend}_b${budget}.err")
    done
done

python3 scripts/parse_dspark_phase0.py "${logs[@]}" > "$OUT_DIR/summary.tsv"

echo "dspark phase0: wrote $OUT_DIR/summary.tsv" >&2
cat "$OUT_DIR/summary.tsv"

if [[ "$RUN_BASELINE" != "0" && -f "$OUT_DIR/baseline.out" ]]; then
    {
        printf 'label\tcmp_vs_baseline\n'
        for out in "$OUT_DIR"/dspark*.out; do
            [[ -e "$out" ]] || continue
            label="$(basename "$out" .out)"
            if cmp -s "$OUT_DIR/baseline.out" "$out"; then
                cmp_status=0
            else
                cmp_status=1
            fi
            printf '%s\t%s\n' "$label" "$cmp_status"
        done
    } > "$OUT_DIR/cmp.tsv"
    echo "dspark phase0: wrote $OUT_DIR/cmp.tsv" >&2
    cat "$OUT_DIR/cmp.tsv"
fi
