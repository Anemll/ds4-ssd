#!/usr/bin/env bash
# Sidecar-resident vs GGUF under DSpark, static scheduler (5 rows fixed) —
# localize whether the sidecar deficit is verify cost or tau/acceptance.
# Plus one no-draft pair on p_seg to anchor the backend baseline gap post prefetch-fix.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
SIDE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

run_one() {
  name="$1"; shift
  echo "===== $name $(date) =====" >> "$B/progress.log"
  /usr/bin/time -p "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  grep -h -E 'ds4: (decode-progress:|prefill:|dspark perf:|dspark verify GPU-busy|dspark acceptance:|dspark full-accept:|dspark avg scheduled:)' "$B/$name.err" >> "$B/progress.log" || true
  sleep 20
}

for P in seg json prose; do
  PROMPT=$(cat "$B/p_$P.txt")
  run_one "side-$P" env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
    ./ds4 -m "$SIDE" --draft dspark --draft-path "$DRAFT" --draft-scheduler static \
    -c 40000 -n 2500 --temp 0 --nothink --resident -p "$PROMPT"
  run_one "gguf-$P" env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
    ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" --draft-scheduler static \
    -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"
done

PROMPT=$(cat "$B/p_seg.txt")
run_one side-seg-nodraft env DS4_PROGRESS_1K=1 \
  ./ds4 -m "$SIDE" -c 40000 -n 2500 --temp 0 --nothink --resident -p "$PROMPT"
run_one gguf-seg-nodraft env DS4_PROGRESS_1K=1 \
  ./ds4 --model "$GGUF" -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"

echo "ALL DONE $(date)" >> "$B/progress.log"
