#!/usr/bin/env bash
# run_resident_variant_sweep.sh
# Clean, cooldown-separated resident-MoE backend sweep to (re)establish the
# correct gates/settings. Compares, per context frontier (2K..32K):
#   mulmm   - upstream mul_mm_id GPU routed (no resident MPP envs)   [reference]
#   int8    - NAX-int8 fused-dequant matmul2d (compact bridge)
#   nax     - NAX-half "Plan A" (h_h_f fused gate+up+swiglu, MIN_REFS=0)
#   alu     - Path C i8 in-kernel NAX∥ALU fused (DS4_RESIDENT_MOE_NAX_FULL_FUSED)
#
# RESIDENT mode = full GGUF in RAM (--moe-mode off). Needs ~81 GB (M5 Max).
# Variants present depends on branch:
#   agent-clean: mulmm, int8, (nax only after Plan A wiring is finished)
#   bad-stat-NAX_ALU-ANE: all four (full session work)
# Unavailable variants are auto-skipped (binary lacks the env hook -> same as int8;
# the script flags suspiciously-equal rows).
#
# METHODOLOGY (matters — M5 thermals + concurrent GPU users skew results):
#   - 90s cooldown before each run (overridable: COOLDOWN=).
#   - aborts if another ds4/moe-batch/ane process is using the GPU.
#   - single gen token; prefill t/s is the measured quantity.
#   - run it ALONE; close other GPU apps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
# shellcheck source=ds4_backend_env.sh
. "$ROOT/ds4_backend_env.sh"   # ds4_backend_common_env / ds4_backend_variant_env

DS4_BENCH="${DS4_BENCH:-./ds4-bench}"
DS4_MODEL="${DS4_MODEL:-/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
DS4_PROMPT="${DS4_PROMPT:-/tmp/big_prompt.txt}"     # must be >= largest ctx in tokens
CTXS="${CTXS:-2048 4096 8192 16384 32768}"
VARIANTS="${VARIANTS:-mulmm int8 nax alu}"
COOLDOWN="${COOLDOWN:-90}"
LOCK="${DS4_LOCK_FILE:-/tmp/ds4-sweep.lock}"
OUT="${OUT:-resident_variant_sweep_$(date +%Y%m%d_%H%M%S).csv}"

# --- contention guard ---
if ps -A -o command | grep -iE "ds4-(bench|agent|server)|moe-batch|ane_ds4" | grep -v grep | grep -vq "$$"; then
  echo "ABORT: another ds4/moe-batch/ane process is running — GPU contention will skew results." >&2
  ps -A -o pid,command | grep -iE "ds4-(bench|agent|server)|moe-batch|ane_ds4" | grep -v grep >&2 || true
  exit 1
fi
[ -f "$DS4_PROMPT" ] || { echo "ABORT: prompt file $DS4_PROMPT missing (need >= 32K tokens). Set DS4_PROMPT=." >&2; exit 1; }

# Common resident MPP/NAX enablement (from the shared snippet; DS4_LOCK_FILE added).
COMMON=( DS4_LOCK_FILE="$LOCK" )
while IFS= read -r kv; do [ -n "$kv" ] && COMMON+=( "$kv" ); done < <(ds4_backend_common_env)

variant_env() { ds4_backend_variant_env "$1"; }  # mulmm/int8/nax/alu aliases handled in snippet

echo "ctx,variant,prefill_tps" | tee "$OUT"
for ctx in $CTXS; do
  chunk=$ctx; [ "$chunk" -gt 16384 ] && chunk=16384
  for v in $VARIANTS; do
    sleep "$COOLDOWN"
    extra="$(variant_env "$v")"
    args=( -m "$DS4_MODEL" --metal --moe-mode off --warm-weights
           --ctx-start "$ctx" --ctx-max "$ctx" --gen-tokens 1 --prompt-file "$DS4_PROMPT" )
    if [ "$extra" = "MULMM" ]; then
      tps=$(env DS4_LOCK_FILE="$LOCK" DS4_METAL_PREFILL_CHUNK="$chunk" \
            "$DS4_BENCH" "${args[@]}" 2>&1 | grep "prefill full:" | grep -oE '[0-9]+\.[0-9]+ t/s' | head -1 | grep -oE '[0-9]+\.[0-9]+')
    else
      tps=$(env "${COMMON[@]}" DS4_METAL_PREFILL_CHUNK="$chunk" $extra \
            "$DS4_BENCH" "${args[@]}" 2>&1 | grep "prefill full:" | grep -oE '[0-9]+\.[0-9]+ t/s' | head -1 | grep -oE '[0-9]+\.[0-9]+')
    fi
    echo "$ctx,$v,${tps:-FAIL}" | tee -a "$OUT"
  done
done
echo
echo "=== sweep complete -> $OUT ==="
echo "Gate guidance:"
echo " - resident MPP crossover: first ctx where int8 > mulmm => set DS4_RESIDENT_MOE_MPP_MIN_TOKENS just below it."
echo " - Plan A: if nax > int8 at a ctx, NAX-half wins there; MIN_REFS=0 was best post-cooldown."
echo " - alu(Path C) is expected ~= or below int8 (M5 has no cross-kernel NAX∥ALU overlap)."