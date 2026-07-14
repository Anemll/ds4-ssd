#!/usr/bin/env bash
# DSpark strict-contract smoke: DSpark output must be byte-identical to
# no-draft at temp 0 (same binary, same prompt). Catches verify-arm exactness
# regressions that DSpark-vs-DSpark A/B gates cannot see (2026-07-08: the
# "monolithic row-exact tiny batch" MoE path broke this silently for 3 days).
#
# Usage: tests/dspark_parity_smoke.sh [n_tokens]
# Env: DS4_PARITY_MODEL (gguf path), DS4_PARITY_DRAFT (dspark draft package)
set -uo pipefail
cd "$(dirname "$0")/.."
GG=${DS4_PARITY_MODEL:-/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}
DR=${DS4_PARITY_DRAFT:-/Users/anemll/Models/DSv4-Flash-DSpark-draft}
N=${1:-300}
PROMPT="Space Invaders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

./ds4 --model "$GG" -c 40000 -n "$N" --temp 0 --nothink -p "$PROMPT" > "$T/nodraft.out" 2>/dev/null
rc=0
for SCHED in static confidence rate; do
  ./ds4 --model "$GG" --draft dspark --draft-path "$DR" --draft-scheduler "$SCHED" \
    -c 40000 -n "$N" --temp 0 --nothink -p "$PROMPT" > "$T/$SCHED.out" 2>/dev/null
  NZ=$(stat -f%z "$T/nodraft.out"); DZ=$(stat -f%z "$T/$SCHED.out")
  if [ "$NZ" -lt 100 ] || [ "$DZ" -lt 100 ]; then
    echo "dspark-parity: $SCHED RUN FAILED (nodraft=$NZ bytes, dspark=$DZ bytes)"; rc=1; continue
  fi
  MIN=$(( NZ < DZ ? NZ : DZ ))
  if head -c "$MIN" "$T/nodraft.out" | cmp -s - <(head -c "$MIN" "$T/$SCHED.out"); then
    echo "dspark-parity: $SCHED OK ($MIN bytes)"
  else
    echo "dspark-parity: $SCHED BYTE MISMATCH: $(head -c "$MIN" "$T/nodraft.out" | cmp - <(head -c "$MIN" "$T/$SCHED.out") 2>&1 | head -1)"
    rc=1
  fi
done
exit $rc
