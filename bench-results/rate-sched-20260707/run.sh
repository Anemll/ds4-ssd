#!/usr/bin/env bash
# Rate scheduler (k=0 skip + online calibration + dormant hysteresis) vs
# confidence-0.4 / confidence-cost on the new binary.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
AB=/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/bench-results/sidecar-gap-ab-20260707

run_one() {
  name="$1"; shift
  echo "===== $name $(date) =====" >> "$B/progress.log"
  /usr/bin/time -p "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  grep -h -E 'ds4: (decode-progress:|prefill:|dspark perf:|dspark acceptance:|dspark full-accept:|dspark avg scheduled:)' "$B/$name.err" >> "$B/progress.log" || true
  sleep 20
}

for P in seg json prose; do
  PROMPT=$(cat "$AB/p_$P.txt")
  run_one "$P-rate" env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 DS4_DSPARK_CONF_CALIB=1 \
    ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" \
    --draft-scheduler rate -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"
done

PROMPT=$(cat "$AB/p_seg.txt")
run_one seg-cost env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
  ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" \
  --draft-scheduler confidence-cost -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"
run_one seg-conf04-newbin env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
  ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" \
  --draft-scheduler confidence --draft-conf-threshold 0.4 \
  -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"

echo "ALL DONE $(date)" >> "$B/progress.log"
