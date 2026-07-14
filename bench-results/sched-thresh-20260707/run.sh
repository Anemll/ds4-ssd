#!/usr/bin/env bash
# Confidence-threshold do-no-harm sweep at the NEW faster decode baseline.
# Question: how much of the DSpark loss on low-tau content (json/prose) does
# aggressive trimming recover, and what does it cost on seg (tau ~3)?
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
  grep -h -E 'ds4: (decode-progress:|prefill:|dspark perf:|dspark acceptance:|dspark full-accept:|dspark avg scheduled:|dspark conf)' "$B/$name.err" >> "$B/progress.log" || true
  sleep 20
}

for P in seg json prose; do
  PROMPT=$(cat "$AB/p_$P.txt")
  for TH in 0.4 0.9; do
    run_one "$P-th$TH" env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 DS4_DSPARK_CONF_CALIB=1 \
      ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" \
      --draft-scheduler confidence --draft-conf-threshold "$TH" \
      -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"
  done
done

for P in json prose; do
  PROMPT=$(cat "$AB/p_$P.txt")
  run_one "$P-nodraft" env DS4_PROGRESS_1K=1 \
    ./ds4 --model "$GGUF" -c 40000 -n 2500 --temp 0 --nothink -p "$PROMPT"
done

echo "ALL DONE $(date)" >> "$B/progress.log"
