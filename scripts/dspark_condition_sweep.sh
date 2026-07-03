#!/usr/bin/env bash
#
# dspark_condition_sweep.sh — condition-aware kernel A/B harness.
#
# Purpose: many DSpark verifier optimizations win only under certain conditions
# (context length / n_keys, batch M, NA:ALU balance, route reuse). Static flags
# or always-on/off gates leave performance on the table. This harness runs the
# REAL binary for a set of kernel VARIANTS, captures the unfenced per-block
# verify-ms-vs-position curve (DS4_DSPARK_BLOCK_TIMING) for each, and the parser
# compares variants across the context axis to find break-even points → the
# thresholds a runtime kernel selector should gate on.
#
# Why real runs (not synthetic microbench / fenced profile): the fenced
# subprofile is an ~8x artifact on this verifier; only real unfenced generation
# reflects true kernel cost. One long run per variant yields the whole
# context curve (position grows through the run), so the matrix is
# variants x 1 long run, not variants x contexts.
#
# Portability: parameterized paths. To run on another system, override BASE /
# DRAFT / BIN. Everything else (thermal cooldown, gate) travels.
#
# Gate: uses a unique lock and refuses to start a run while another ds4 /
# ds4-agent / ds4-server is active (other agents share the GPU; concurrency +
# M5 thermals destroy bench validity).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-./ds4}"
BASE="${BASE:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DRAFT="${DRAFT:-/Users/anemll/Models/DSv4-Flash-DSpark-draft}"
# A long, neutral prompt so the run reaches deep context (attention-dominated).
PROMPT="${PROMPT:-Write a detailed technical essay about how speculative decoding works in large language models, covering draft models, verification, acceptance criteria, tree attention, and practical engineering trade-offs.}"
N="${N:-2500}"            # generation length — long enough to sweep context
CTX="${CTX:-4096}"
BUDGET="${BUDGET:-5}"      # --draft-verify
REPEATS="${REPEATS:-1}"   # runs per variant (parser takes median per position bin)
COOLDOWN="${COOLDOWN:-30}" # seconds between runs — thermal recovery (M5 Max)
RESIDENT="${RESIDENT:-1}"
LOCK="${DS4_LOCK_FILE:-/tmp/ds4-condition-sweep.lock}"
OUT_DIR="${OUT_DIR:-bench-results/condition_sweep_$(date +%Y%m%d_%H%M%S)}"

# Variants: "name|ENV FLAGS for this variant".  Override VARIANTS to add kernels
# (e.g. MoE dedup, AV mode, prefill-indexer-NAX gate) without editing the script.
# The flags are exported verbatim before the run.
DEFAULT_VARIANTS=(
  "strict|"
  "caseE_fastav|DS4_DSPARK_ATTN_NAX=1 DS4_DSPARK_ATTN_NAX_FAST_AV=1"
  "caseE_simpleav|DS4_DSPARK_ATTN_NAX=1"
)
# To override: export VARIANTS as a newline-separated list of "name|FLAGS".
if [[ -n "${VARIANTS:-}" ]]; then
  IFS=$'\n' read -r -d '' -a VARIANT_LIST < <(printf '%s\0' "$VARIANTS") || true
else
  VARIANT_LIST=("${DEFAULT_VARIANTS[@]}")
fi

mkdir -p "$OUT_DIR"
echo "condition-sweep -> $OUT_DIR"
echo "  bin=$BIN base=$BASE n=$N ctx=$CTX budget=$BUDGET repeats=$REPEATS cooldown=${COOLDOWN}s"
echo "  variants: ${VARIANT_LIST[*]}"

resident_args=()
[[ "$RESIDENT" != "0" ]] && resident_args+=(--resident)

gate_check() {
  # Refuse to run while another ds4 process is active. Exact process-name
  # match (pgrep -x): immune to unrelated command lines that merely contain
  # './ds4' text (a ps|grep gate once deadlocked on an orphaned wrapper).
  if pgrep -x ds4 >/dev/null || pgrep -x ds4-agent >/dev/null || pgrep -x ds4-server >/dev/null; then
    echo "GATE: another ds4 process is active — aborting to protect bench validity." >&2
    pgrep -lx ds4 >&2 || true; pgrep -lx ds4-agent >&2 || true; pgrep -lx ds4-server >&2 || true
    return 1
  fi
  return 0
}

run_one() {
  local name="$1" flags="$2" rep="$3"
  local log="$OUT_DIR/${name}.r${rep}.log"
  echo "  [$(date +%H:%M:%S)] run variant=$name rep=$rep -> $log"
  gate_check || return 1
  # shellcheck disable=SC2086
  env DS4_LOCK_FILE="$LOCK" DS4_DSPARK_BLOCK_TIMING=1 $flags \
    "$BIN" -m "$BASE" \
      --draft dspark --draft-path "$DRAFT" \
      --draft-verify "$BUDGET" --temp 0 --nothink -n "$N" \
      -p "$PROMPT" "${resident_args[@]}" -c "$CTX" \
      >"$log" 2>&1 || { echo "    run FAILED (exit $?) — see $log" >&2; return 1; }
  grep -E "generation:|dspark perf:|acceptance:" "$log" | sed 's/^/    /' || true
}

first=1
for entry in "${VARIANT_LIST[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  for ((r=1; r<=REPEATS; r++)); do
    [[ $first -eq 0 ]] && { echo "  cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }
    first=0
    run_one "$name" "$flags" "$r" || true
  done
done

echo "=== parsing ==="
python3 "$ROOT/scripts/dspark_condition_parse.py" "$OUT_DIR" "${BASELINE:-strict}" | tee "$OUT_DIR/summary.txt"
echo "done -> $OUT_DIR/summary.txt"
