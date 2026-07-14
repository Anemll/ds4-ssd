#!/usr/bin/env bash
# User's exact pair: sidecar resident, DSpark default vs no-draft, -c 20000 -n 4000.
# +DS4_PROGRESS_1K on both for ctx-shape. Third run: rate scheduler.
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
SIDE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
DR=/Users/anemll/Models/DSv4-Flash-DSpark-draft
PROMPT="aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics"

run_one() {
  name="$1"; shift
  echo "===== $name $(date) =====" >> "$B/progress.log"
  "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  grep -h -E 'ds4: (decode-progress:|prefill:|dspark perf:|dspark verify GPU-busy|dspark acceptance:|dspark full-accept:|dspark avg scheduled:)' "$B/$name.err" >> "$B/progress.log" || true
  sleep 20
}

run_one dspark env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
  ./ds4 -m "$SIDE" --draft dspark --draft-path "$DR" --draft-verify 5 \
  --nothink --temp 0 --resident -c 20000 -n 4000 -p "$PROMPT"
run_one nodraft env DS4_PROGRESS_1K=1 DS4_AGENT_TURN_STATS=1 DS4_DSPARK_PERF=1 \
  ./ds4 -m "$SIDE" --nothink --temp 0 --resident -c 20000 -n 4000 -p "$PROMPT"
run_one dspark-rate env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
  ./ds4 -m "$SIDE" --draft dspark --draft-path "$DR" --draft-scheduler rate \
  --nothink --temp 0 --resident -c 20000 -n 4000 -p "$PROMPT"
echo "ALL DONE $(date)" >> "$B/progress.log"
