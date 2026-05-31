#!/usr/bin/env bash
# tune_profile.sh — find good DS4 tuning knobs for THIS machine and (optionally)
# write them into ds4_profile.json so every ds4 executable picks them up.
#
# It sweeps the routed-ANE prefill threshold (and reports decode t/s) with the
# resident ANE prefill path, picks the best by prefill throughput, decides a
# CTX_GROW block from available RAM, prints the resulting JSON profile entry, and
# offers to merge it into ds4_profile.json (env in the profile is a *default* —
# anything you export at runtime still wins).
#
# Usage:
#   ./tune_profile.sh -m MODEL.gguf -p PROMPT.txt [options]
# Options:
#   --ctx N             context to prefill to (default 8192)
#   --chunk N           DS4_METAL_PREFILL_CHUNK (default = ctx, capped at 16384)
#   --gen N             decode tokens to measure (default 64)
#   --min-refs "L"      space list of ANE min_refs to sweep (default "32 64 128 256")
#   --bin PATH          ds4-bench path (default ./ds4-bench)
#   --apply             merge the winning entry into ds4_profile.json without prompting
#   --profile PATH      profile file to write (default ./ds4_profile.json)
#   -h|--help
set -u

BENCH=./ds4-bench
MODEL=""; PROMPT=""; CTX=8192; CHUNK=""; GEN=64
MINREFS="32 64 128 256"; APPLY=0; PROFILE=./ds4_profile.json

while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -p|--prompt-file) PROMPT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --chunk) CHUNK="$2"; shift 2;;
    --gen) GEN="$2"; shift 2;;
    --min-refs) MINREFS="$2"; shift 2;;
    --bin) BENCH="$2"; shift 2;;
    --apply) APPLY=1; shift;;
    --profile) PROFILE="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$MODEL" ] || { echo "error: -m MODEL.gguf required" >&2; exit 2; }
[ -x "$BENCH" ] || { echo "error: $BENCH not found/executable (run 'make ds4-bench')" >&2; exit 2; }
[ -n "$PROMPT" ] || { echo "error: -p PROMPT.txt required (a prompt long enough to fill --ctx)" >&2; exit 2; }
[ -n "$CHUNK" ] || { CHUNK=$CTX; [ "$CHUNK" -gt 16384 ] && CHUNK=16384; }

CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
RAM_GIB=$(( RAM_BYTES / 1073741824 ))
# RAM bucket for the match{min_ram_gib}: floor to a round step below actual.
if   [ "$RAM_GIB" -ge 256 ]; then RAM_MATCH=256
elif [ "$RAM_GIB" -ge 128 ]; then RAM_MATCH=128
elif [ "$RAM_GIB" -ge 96  ]; then RAM_MATCH=64
elif [ "$RAM_GIB" -ge 64  ]; then RAM_MATCH=64
else RAM_MATCH=0; fi
# CTX_GROW block: tight RAM -> small (max model-resident headroom); ample -> larger.
if [ "$RAM_GIB" -ge 128 ]; then GROW_BLOCK=16384; else GROW_BLOCK=2048; fi

echo "== tune_profile: chip='$CHIP' ram=${RAM_GIB}GiB ctx=$CTX chunk=$CHUNK gen=$GEN =="
echo "== sweeping DS4_RESIDENT_MOE_ANE_MIN_REFS in: $MINREFS (each run loads the model; this takes a while) =="

OUT=$(mktemp); LOCK="${TMPDIR:-/tmp}/ds4-tune.lock"
GRAPH_RAW_CAP=$(( CHUNK + 512 ))
best_refs=""; best_pre="-1"; best_gen="0"
printf "%-10s %-12s %-12s\n" "min_refs" "prefill_tps" "decode_tps"
for mr in $MINREFS; do
  CSV="${TMPDIR:-/tmp}/ds4_tune_${mr}.csv"
  DS4_LOCK_FILE="$LOCK" \
  DS4_METAL_PREFILL_CHUNK="$CHUNK" DS4_METAL_GRAPH_RAW_CAP="$GRAPH_RAW_CAP" \
  DS4_RESIDENT_MOE_ANE_MIN_REFS="$mr" DS4_RESIDENT_MOE_ANE_MAX_REFS=1024 DS4_RESIDENT_MOE_ANE_QUEUE=8 \
  "$BENCH" -m "$MODEL" --prompt-file "$PROMPT" --metal --moe-mode off \
    --resident-ane-prefill --no-decode-split \
    --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens "$GEN" \
    --csv "$CSV" >/dev/null 2>"$OUT"
  line=$(grep -E "^[0-9]+," "$CSV" 2>/dev/null | tail -1)
  pre=$(echo "$line" | cut -d, -f3); gen=$(echo "$line" | cut -d, -f5)
  [ -n "$pre" ] || { pre="0"; gen="0"; echo "  (min_refs=$mr failed; see below)"; tail -3 "$OUT"; }
  printf "%-10s %-12s %-12s\n" "$mr" "${pre:-0}" "${gen:-0}"
  if awk "BEGIN{exit !(${pre:-0} > ${best_pre})}"; then best_pre="$pre"; best_gen="$gen"; best_refs="$mr"; fi
done
rm -f "$OUT"

[ -n "$best_refs" ] || { echo "error: all sweep runs failed" >&2; exit 1; }
echo "== best: min_refs=$best_refs (prefill ${best_pre} t/s, decode ${best_gen} t/s) =="

# Build the profile entry JSON for this machine.
ENTRY=$(cat <<JSON
    {
      "match": { "chip": "$CHIP", "min_ram_gib": $RAM_MATCH },
      "env": {
        "DS4_CTX_GROW": "1",
        "DS4_CTX_GROW_BLOCK": "$GROW_BLOCK",
        "DS4_RESIDENT_MOE_ANE_MIN_REFS": "$best_refs",
        "DS4_RESIDENT_MOE_ANE_MAX_REFS": "1024",
        "DS4_RESIDENT_MOE_ANE_QUEUE": "8"
      }
    }
JSON
)
echo "== proposed profile entry for [$CHIP / ${RAM_MATCH}GiB+] =="
echo "$ENTRY"

if [ "$APPLY" != "1" ]; then
  printf "Merge this into %s? [y/N] " "$PROFILE"; read -r ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not applied."; exit 0;; esac
fi

# Merge with python3 (replace any existing entry with the same chip+ram, else prepend).
CHIP="$CHIP" RAM_MATCH="$RAM_MATCH" GROW_BLOCK="$GROW_BLOCK" BEST_REFS="$best_refs" PROFILE="$PROFILE" \
python3 - <<'PY'
import json, os, sys
path=os.environ["PROFILE"]; chip=os.environ["CHIP"]; ram=int(os.environ["RAM_MATCH"])
entry={"match":{"chip":chip,"min_ram_gib":ram},
       "env":{"DS4_CTX_GROW":"1","DS4_CTX_GROW_BLOCK":os.environ["GROW_BLOCK"],
              "DS4_RESIDENT_MOE_ANE_MIN_REFS":os.environ["BEST_REFS"],
              "DS4_RESIDENT_MOE_ANE_MAX_REFS":"1024","DS4_RESIDENT_MOE_ANE_QUEUE":"8"}}
try:
    with open(path) as f: doc=json.load(f)
except Exception: doc={"version":1,"profiles":[]}
profs=doc.setdefault("profiles",[])
def same(m): return m.get("chip")==chip and int(m.get("min_ram_gib",0))==ram
profs=[p for p in profs if not same(p.get("match",{}))]
profs.insert(0, entry)  # most-specific first
doc["profiles"]=profs
with open(path,"w") as f: json.dump(doc,f,indent=2); f.write("\n")
print("== wrote %s (entry for %s / %dGiB+ now first) ==" % (path, chip, ram))
PY
