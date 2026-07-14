#!/usr/bin/env bash
set -euo pipefail
B="$1"
PROMPT=$(cat "$B/prompt.txt")
SIDE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
run_one() {
  name="$1"; shift
  echo "===== $name $(date) ====="
  /usr/bin/time -p "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  grep -H -E 'ds4: (decode-progress:|prefill:|ttf:|dspark perf:|dspark acceptance:|dspark full-accept:|dspark avg scheduled:|dspark overlap-draft:)|^real |^user |^sys ' "$B/$name.err" || true
}
run_one sidecar-dspark env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 DS4_DSPARK_FRONTIER_DRAFT=1 DS4_DSPARK_OVERLAP_DRAFT=1 \
  ./ds4 -m "$SIDE" --draft dspark --draft-path "$DRAFT" -c 40000 -n 7000 --temp 0 --nothink --resident -p "$PROMPT"
sleep 20
run_one gguf-dspark env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 DS4_DSPARK_FRONTIER_DRAFT=1 DS4_DSPARK_OVERLAP_DRAFT=1 \
  ./ds4 --model "$GGUF" --draft dspark --draft-path "$DRAFT" -c 40000 -n 7000 --temp 0 --nothink -p "$PROMPT"
sleep 20
run_one upstream-gguf env DS4_PROGRESS_1K=1 \
  ../ds4/ds4 --model "$GGUF" -c 40000 -n 7000 --temp 0 --nothink -p "$PROMPT"
