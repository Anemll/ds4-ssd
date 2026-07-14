#!/usr/bin/env bash
# Champion promotion gates:
#  (1) byte parity vs no-draft (GGUF, seg n=300) under champion envs
#  (2) cross-content perf: {json, prose} x {champion, nodraft}, sidecar resident n=2500
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
GG=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
SIDE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
DR=/Users/anemll/Models/DSv4-Flash-DSpark-draft
AB=bench-results/sidecar-gap-ab-20260707
CH="DS4_DSPARK_FORCE_TARGET_FIRST=1 DS4_DSPARK_CONF_SCALE=0.85 DS4_DSPARK_CONF_THRESHOLD=0.50"

log() { echo "$*" >> "$B/progress.log"; }
run_one() {
  name="$1"; shift
  log "===== $name $(date)"
  "$@" > "$B/$name.out" 2> "$B/$name.err" || true
  grep -h -E 'ds4: (decode-progress:|prefill:|dspark perf:|dspark acceptance:|dspark full-accept:|dspark avg scheduled:)' "$B/$name.err" >> "$B/progress.log" || true
  sleep 20
}

# Gate 1: parity (GGUF seg n=300)
PROMPT=$(cat $AB/p_seg.txt)
run_one par-nodraft ./ds4 --model $GG -c 40000 -n 300 --temp 0 --nothink -p "$PROMPT"
run_one par-champ env DS4_DSPARK_FORCE_TARGET_FIRST=1 DS4_DSPARK_CONF_SCALE=0.85 DS4_DSPARK_CONF_THRESHOLD=0.50 \
  ./ds4 --model $GG --draft dspark --draft-path $DR -c 40000 -n 300 --temp 0 --nothink -p "$PROMPT"
NZ=$(stat -f%z $B/par-nodraft.out); DZ=$(stat -f%z $B/par-champ.out); MIN=$((NZ<DZ?NZ:DZ))
if [ "$MIN" -gt 100 ] && head -c $MIN $B/par-nodraft.out | cmp -s - <(head -c $MIN $B/par-champ.out); then
  log "GATE1 PARITY: OK"
else
  log "GATE1 PARITY: BROKEN $(head -c $MIN $B/par-nodraft.out | cmp - <(head -c $MIN $B/par-champ.out) 2>&1 | head -1)"
fi

# Gate 2: cross-content (sidecar resident n=2500)
for P in json prose; do
  PROMPT=$(cat $AB/p_$P.txt)
  run_one "$P-champ" env DS4_PROGRESS_1K=1 DS4_DSPARK_PERF=1 \
    DS4_DSPARK_FORCE_TARGET_FIRST=1 DS4_DSPARK_CONF_SCALE=0.85 DS4_DSPARK_CONF_THRESHOLD=0.50 \
    ./ds4 -m $SIDE --draft dspark --draft-path $DR --nothink --temp 0 --resident -c 20000 -n 2500 -p "$PROMPT"
  run_one "$P-nodraft" env DS4_PROGRESS_1K=1 \
    ./ds4 -m $SIDE --nothink --temp 0 --resident -c 20000 -n 2500 -p "$PROMPT"
done
log "ALL DONE $(date)"
