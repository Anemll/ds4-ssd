#!/usr/bin/env bash
# Byte-parity triage on the current build:
#  (1) no-draft self-determinism (x2)
#  (2) static DSpark self-determinism (x2)
#  then cross-compares. All on the seg prompt where divergence is at char 74.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
until grep -q "ALL DONE" /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/bench-results/rate-sched-20260707/progress.log 2>/dev/null; do sleep 10; done
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
PROMPT=$(cat bench-results/sidecar-gap-ab-20260707/p_seg.txt)

run_one() {
  name="$1"; shift
  echo "===== $name $(date) =====" >> "$B/progress.log"
  "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  sleep 15
}

run_one nodraft-a ./ds4 --model "$GGUF" -c 40000 -n 1200 --temp 0 --nothink -p "$PROMPT"
run_one nodraft-b ./ds4 --model "$GGUF" -c 40000 -n 1200 --temp 0 --nothink -p "$PROMPT"
run_one static-a ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" --draft-scheduler static -c 40000 -n 1200 --temp 0 --nothink -p "$PROMPT"
run_one static-b ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" --draft-scheduler static -c 40000 -n 1200 --temp 0 --nothink -p "$PROMPT"

{
  cmp -s "$B/nodraft-a.out" "$B/nodraft-b.out" && echo "nodraft self: MATCH" || echo "nodraft self: DIFF $(cmp "$B/nodraft-a.out" "$B/nodraft-b.out" 2>&1 | head -1)"
  cmp -s "$B/static-a.out" "$B/static-b.out" && echo "static self: MATCH" || echo "static self: DIFF $(cmp "$B/static-a.out" "$B/static-b.out" 2>&1 | head -1)"
  cmp -s "$B/nodraft-a.out" "$B/static-a.out" && echo "cross: MATCH" || echo "cross: DIFF $(cmp "$B/nodraft-a.out" "$B/static-a.out" 2>&1 | head -1)"
  echo "TRIAGE DONE $(date)"
} >> "$B/progress.log"
