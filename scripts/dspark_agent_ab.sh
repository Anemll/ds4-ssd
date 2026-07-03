#!/usr/bin/env bash
#
# dspark_agent_ab.sh — automated agentic A/B for DSpark draft variants.
#
# The isolated CLI canary (Pygame prompt) is only generic validation. Real
# DSpark speed/correctness claims must also hold in an agentic workflow:
# multi-round tool use, growing context, mixed decode/prefill. This harness
# runs ds4-agent non-interactively on a deterministic tool task for each
# draft variant and reports:
#   - agentic generation t/s (aggregated from ds4-agent turn-stats)
#   - dspark acceptance / full-accept footers
#   - malformed-tool-syntax check (hard fail)
#   - trajectory byte-compare against the no-draft reference (strict
#     verification at temp 0 with deterministic tools must reproduce the
#     exact no-draft trajectory; timing lines are filtered before diff)
#
# Each run executes in a freshly seeded sandbox directory so tool output
# (list/read/search) is deterministic across variants.
#
# Usage:
#   scripts/dspark_agent_ab.sh                # nodraft + strict_b5
#   VARIANTS=$'strict_b5|\nfrontier|DS4_DSPARK_FRONTIER_DRAFT=1' \
#     scripts/dspark_agent_ab.sh              # custom variants "name|ENV|ARGS"
#   TOKENS=4096 PROMPT='...' scripts/dspark_agent_ab.sh
#
# Bench discipline: single-process gate, thermal cooldown between runs,
# unfenced real runs only.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-$ROOT/ds4-agent}"
BASE="${BASE:-/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major}"
DRAFT="${DRAFT:-/Users/anemll/Models/DSv4-Flash-DSpark-draft}"
TOKENS="${TOKENS:-2048}"
CTX="${CTX:-8192}"
BUDGET="${BUDGET:-5}"
COOLDOWN="${COOLDOWN:-45}"
OUT_DIR="${OUT_DIR:-$ROOT/bench-results/agent_ab_$(date +%Y%m%d_%H%M%S)}"
# Runs cd into a sandbox; OUT_DIR must be absolute or redirects land in it.
[[ "$OUT_DIR" != /* ]] && OUT_DIR="$ROOT/$OUT_DIR"
PROMPT="${PROMPT:-Use local tools only, one tool call at a time, with no narration before tool calls: list the current directory, read notes.txt, search for TODO in the directory, create summary.txt containing a one-paragraph summary of notes.txt, then write a complete single-file Space Invaders game in pygame to game.py, then stop.}"

# Variants: "name|ENV assignments|extra ds4-agent args". nodraft runs first as
# the trajectory reference. Override VARIANTS (newline-separated) to test
# schedulers/kernels; do NOT add relaxed-accept variants here — they violate
# the strict contract this harness exists to enforce.
DEFAULT_VARIANTS=(
  "strict_b5||"
  "dynamic||--draft-verify-dynamic"
)
if [[ -n "${VARIANTS:-}" ]]; then
  IFS=$'\n' read -r -d '' -a VARIANT_LIST < <(printf '%s\0' "$VARIANTS") || true
else
  VARIANT_LIST=("${DEFAULT_VARIANTS[@]}")
fi

mkdir -p "$OUT_DIR"
echo "agent-ab -> $OUT_DIR"
echo "  bin=$BIN base=$BASE tokens=$TOKENS ctx=$CTX budget=$BUDGET cooldown=${COOLDOWN}s"

gate_check() {
  # Wait up to GATE_WAIT seconds for other ds4 processes to exit, then abort.
  # Exact process-name match (pgrep -x): immune to unrelated command lines
  # that merely contain './ds4' text (a ps|grep gate once deadlocked on an
  # orphaned wrapper for an hour).
  local waited=0 max="${GATE_WAIT:-120}"
  while pgrep -x ds4 >/dev/null || pgrep -x ds4-agent >/dev/null || pgrep -x ds4-server >/dev/null; do
    if (( waited >= max )); then
      echo "GATE: another ds4 process is still active after ${max}s — aborting run." >&2
      pgrep -lx ds4 >&2 || true; pgrep -lx ds4-agent >&2 || true; pgrep -lx ds4-server >&2 || true
      return 1
    fi
    sleep 5; waited=$((waited+5))
  done
  return 0
}

# The agent runs from a sandbox cwd, but ds4 resolves metal/*.metal relative to
# cwd. Export absolute per-file source overrides so kernels load from ROOT.
export_metal_sources() {
  local pairs=(
    "DS4_METAL_FLASH_ATTN_SOURCE:flash_attn.metal" "DS4_METAL_DENSE_SOURCE:dense.metal"
    "DS4_METAL_GLM_SOURCE:glm.metal" "DS4_METAL_GLM_QUANT_TABLES_SOURCE:glm_quant_tables.metal"
    "DS4_METAL_MOE_SOURCE:moe.metal" "DS4_METAL_DSV4_HC_SOURCE:dsv4_hc.metal"
    "DS4_METAL_UNARY_SOURCE:unary.metal" "DS4_METAL_DSV4_KV_SOURCE:dsv4_kv.metal"
    "DS4_METAL_DSV4_ROPE_SOURCE:dsv4_rope.metal" "DS4_METAL_DSV4_MISC_SOURCE:dsv4_misc.metal"
    "DS4_METAL_ARGSORT_SOURCE:argsort.metal" "DS4_METAL_CPY_SOURCE:cpy.metal"
    "DS4_METAL_CONCAT_SOURCE:concat.metal" "DS4_METAL_GET_ROWS_SOURCE:get_rows.metal"
    "DS4_METAL_SUM_ROWS_SOURCE:sum_rows.metal" "DS4_METAL_SOFTMAX_SOURCE:softmax.metal"
    "DS4_METAL_REPEAT_SOURCE:repeat.metal" "DS4_METAL_GLU_SOURCE:glu.metal"
    "DS4_METAL_NORM_SOURCE:norm.metal" "DS4_METAL_BIN_SOURCE:bin.metal"
    "DS4_METAL_SET_ROWS_SOURCE:set_rows.metal"
  )
  local p
  for p in "${pairs[@]}"; do
    export "${p%%:*}=$ROOT/metal/${p##*:}"
  done
  # These three load through separate code paths with their own override envs.
  export DS4_METAL_NAX_FUSED_SOURCE="${DS4_METAL_NAX_FUSED_SOURCE:-$ROOT/metal/nax_fused.metal}"
  export DS4_METAL_MXFP4_COMMON_SOURCE="${DS4_METAL_MXFP4_COMMON_SOURCE:-$ROOT/metal/mxfp4_common.h}"
  export DS4_METAL_MXFP4_NATIVE_SOURCE="${DS4_METAL_MXFP4_NATIVE_SOURCE:-$ROOT/metal/mxfp4_native.metal}"
}
export_metal_sources

seed_sandbox() {
  local dir="$1"
  rm -rf "$dir"
  mkdir -p "$dir"
  cat >"$dir/notes.txt" <<'EOF'
Project notes: the decode pipeline has three stages.
TODO: profile the draft wall.
Stage one loads weights, stage two drafts tokens, stage three verifies them.
TODO: automate the agent benchmark.
The verifier must remain exact for correctness.
EOF
  printf 'placeholder\n' >"$dir/data.txt"
}

# Strip nondeterministic content (timestamps, timings, stats, run paths)
# before diff so only the semantic trajectory is compared.
filter_trace() {
  # Also normalize KV bookkeeping (cached=/suffix=/start=): the speculative
  # path legitimately caches one extra evaluated token per round boundary;
  # trajectory equality is about tool calls + emitted text, not KV counters.
  sed -E -e 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:.]+ //' \
         -e 's/cached=[0-9]+ suffix=[0-9]+/cached=~ suffix=~/' \
         -e 's/start=[0-9]+ len=/start=~ len=/' "$1" | \
    grep -Ev 'turn-stats|t/s|tok/s| ms|decode_s|elapsed|seconds|worker start|trace=' || true
}

run_one() {
  local name="$1" envs="$2" extra="$3" draft_args=()
  local sandbox="$OUT_DIR/sandbox_${name}"
  local trace="$OUT_DIR/${name}.trace"
  local err="$OUT_DIR/${name}.err"
  [[ "$name" != "nodraft" ]] && draft_args=(--draft dspark --draft-path "$DRAFT" --draft-verify "$BUDGET")
  seed_sandbox "$sandbox"
  gate_check || return 1
  echo "  [$(date +%H:%M:%S)] variant=$name env='$envs' extra='$extra'"
  # shellcheck disable=SC2086
  (cd "$sandbox" && env DS4_AGENT_TURN_STATS=1 DS4_DSPARK_PERF=1 $envs \
    "$BIN" -m "$BASE" ${draft_args[@]+"${draft_args[@]}"} \
      --non-interactive --nothink --temp 0 --tokens "$TOKENS" -c "$CTX" \
      --resident --trace "$trace" -p "$PROMPT" $extra \
      >"$OUT_DIR/${name}.out" 2>"$err")
  local rc=$?
  echo "    exit=$rc"
  return 0
}

summarize_one() {
  local name="$1"
  local err="$OUT_DIR/${name}.err" out="$OUT_DIR/${name}.out" trace="$OUT_DIR/${name}.trace"
  echo "== $name"
  # Aggregate turn-stats across rounds; lines may land in stdout, stderr, or
  # the trace, so dedupe identical lines before summing.
  cat "$out" "$err" "$trace" 2>/dev/null | grep 'ds4-agent: turn-stats' | sort -u | \
  awk '{
         for (i=1;i<=NF;i++) {
           if ($i ~ /^generated=/) { split($i,a,"="); g+=a[2] }
           if ($i ~ /^decode_s=/)  { split($i,b,"="); s+=b[2] }
         }
         r++
       }
       END { if (s>0) printf "  agent decode: %d tokens / %.2fs = %.2f t/s over %d rounds\n", g, s, g/s, r;
             else print "  agent decode: no turn-stats found" }'
  grep -E "dspark perf:|dspark acceptance:|full-accept" "$err" 2>/dev/null | tail -3 | sed 's/^/  /'
  if grep -E "dsml error|glm_tool error|foreign tool syntax|unsupported tool-call syntax|invalid DSML" "$trace" >/dev/null 2>&1; then
    echo "  TOOLS: MALFORMED tool syntax detected (FAIL)"
  else
    echo "  TOOLS: clean"
  fi
  if [[ "$name" != "nodraft" && -f "$OUT_DIR/nodraft.trace" ]]; then
    if diff -q <(filter_trace "$OUT_DIR/nodraft.trace") <(filter_trace "$trace") >/dev/null 2>&1; then
      echo "  TRAJECTORY: identical to no-draft reference"
    else
      echo "  TRAJECTORY: DIVERGES from no-draft reference (inspect ${name}.trace)"
    fi
  fi
}

first=1
ALL=("nodraft||" "${VARIANT_LIST[@]}")
for entry in "${ALL[@]}"; do
  name="${entry%%|*}"; rest="${entry#*|}"
  envs="${rest%%|*}"; extra="${rest#*|}"
  [[ $first -eq 0 ]] && { echo "  cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"; }
  first=0
  run_one "$name" "$envs" "$extra" || true
done

echo "=== summary ==="
for entry in "${ALL[@]}"; do
  summarize_one "${entry%%|*}"
done | tee "$OUT_DIR/summary.txt"
echo "done -> $OUT_DIR/summary.txt"
