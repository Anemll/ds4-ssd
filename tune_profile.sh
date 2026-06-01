#!/usr/bin/env bash
# tune_profile.sh — find good DS4 tuning knobs for THIS machine and (optionally)
# write them into ds4_profile.json so every ds4 executable picks them up.
#
# Two modes, picked automatically by chip:
#  * NAX-capable chips (Apple M5+ / DS4_NAX_OK=1): sweep the routed-MoE prefill
#    *compute backend* (ALU baseline / NAX-half / NAX-half+ALU / NAX-int8 /
#    ANE+GPU [/ ANE+NAX once built]) and, for ANE backends, the ANE min_refs
#    threshold; bake the winning backend's env block into the profile.
#  * Other chips: sweep just the routed-ANE prefill threshold (min_refs) on the
#    resident ANE prefill path (the original behaviour — unchanged).
# In both modes it decides a CTX_GROW block from RAM, prints the resulting JSON
# profile entry, and offers to merge it into ds4_profile.json (env in the profile
# is a *default* — anything you export at runtime still wins).
#
# Usage:
#   ./tune_profile.sh -m MODEL.gguf -p PROMPT.txt [options]
# Options:
#   --ctx N             context to prefill to (default 8192)
#   --chunk N           DS4_METAL_PREFILL_CHUNK (default = ctx, capped at 16384)
#   --gen N             decode tokens to measure (default 64)
#   --min-refs "L"      space list of ANE min_refs to sweep (default "32 64 128 256")
#   --backends "L"      (NAX chips) space list of backends to sweep
#                       (default: mulmm nax_int8 nax_half nax_half_alu ane_gpu).
#                       Names + env are defined in ds4_backend_env.sh.
#   --no-backend-sweep  force the legacy min_refs-only mode even on a NAX chip
#   --cooldown N        seconds to idle before each measured run (default 0; use
#                       90 on a thermally-touchy M5 Max to keep results valid)
#   --bin PATH          ds4-bench path (default ./ds4-bench)
#   --apply             merge the winning entry into ds4_profile.json without prompting
#   --profile PATH      profile file to write (default ./ds4_profile.json)
#   -h|--help
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ds4_backend_env.sh
. "$ROOT/ds4_backend_env.sh"   # ds4_backend_common_env / ds4_backend_variant_env / ds4_backend_default_list

BENCH=./ds4-bench
MODEL=""; PROMPT=""; CTX=8192; CHUNK=""; GEN=64
MINREFS="32 64 128 256"; APPLY=0; PROFILE=./ds4_profile.json
# 45s default cooldown between measured runs: M5 Max thermals + a warm GPU skew
# prefill t/s; the established gap from prior valid sweeps is 45s. Override with
# --cooldown 0 only for throwaway/correctness runs where thermals don't matter.
BACKENDS=""; NO_BACKEND_SWEEP=0; COOLDOWN=45; RESULTS=./ds4_backend_sweep_results.json

while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -p|--prompt-file) PROMPT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --chunk) CHUNK="$2"; shift 2;;
    --gen) GEN="$2"; shift 2;;
    --min-refs) MINREFS="$2"; shift 2;;
    --backends) BACKENDS="$2"; shift 2;;
    --no-backend-sweep) NO_BACKEND_SWEEP=1; shift;;
    --cooldown) COOLDOWN="$2"; shift 2;;
    --results) RESULTS="$2"; shift 2;;
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

# NAX (matmul2d) capability: Apple M5 and newer, or forced via DS4_NAX_OK=1.
nax_capable() {
  [ "${DS4_NAX_OK:-0}" = "1" ] && return 0
  # "Apple M<gen> ..." with gen >= 5 (M5, M6, ... and M10+).
  echo "$CHIP" | grep -qE 'Apple M([5-9]|[1-9][0-9])( |$)'
}
GRAPH_RAW_CAP=$(( CHUNK + 512 ))
LOCK="${TMPDIR:-/tmp}/ds4-tune.lock"

# measure_prefill "<extra env, space-separated KEY=VAL>" -> echoes prefill t/s (or empty)
# Uses the validated resident harness from run_resident_variant_sweep.sh: full GGUF
# in RAM, single gen token, prefill t/s is the measured quantity. The MULMM sentinel
# runs with NO resident envs (pure upstream GPU reference).
# measure_prefill writes the full run to $MEASURE_LOG and echoes prefill t/s. The
# caller then asks engage_note (which reads that same log — measure_prefill runs in a
# $()-subshell, so it can't export state) whether the resident ANE/MPP dedup path
# actually engaged: "FELL-BACK" = bailed to the grouped fallback (an ANE row that
# silently ran GPU-only is flagged, not trusted), "NO-ANE" = dedup ran but ANE never
# did (ane_groups==0), "" = fine.
MEASURE_LOG="${TMPDIR:-/tmp}/ds4_tune_measure.log"
measure_prefill() {
  local extra="$1"; local envs=()
  [ "$COOLDOWN" -gt 0 ] && sleep "$COOLDOWN"
  envs=( DS4_LOCK_FILE="$LOCK" DS4_METAL_PREFILL_CHUNK="$CHUNK" DS4_METAL_GRAPH_RAW_CAP="$GRAPH_RAW_CAP"
         DS4_RESIDENT_MOE_MPP_STATS=1 )
  if [ "$extra" != "MULMM" ]; then
    while IFS= read -r kv; do [ -n "$kv" ] && envs+=( "$kv" ); done < <(ds4_backend_common_env)
    # shellcheck disable=SC2206
    [ -n "$extra" ] && envs+=( $extra )
  fi
  env "${envs[@]}" "$BENCH" -m "$MODEL" --metal --moe-mode off --warm-weights \
      --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens 1 --prompt-file "$PROMPT" >"$MEASURE_LOG" 2>&1
  grep "prefill full:" "$MEASURE_LOG" | grep -oE '[0-9]+\.[0-9]+ t/s' | head -1 | grep -oE '[0-9]+\.[0-9]+'
}
engage_note() {  # "$1"=extra env ; echoes "FELL-BACK" / "NO-ANE" / ""
  if grep -q "DeDup prefill unavailable" "$MEASURE_LOG" 2>/dev/null; then
    echo "FELL-BACK"; return
  fi
  if printf '%s' "$1" | grep -q "ANE_HYBRID=1"; then
    local ag; ag="$(grep -oE 'ane_groups=[0-9]+' "$MEASURE_LOG" 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')"
    [ "${ag:-0}" -gt 0 ] || echo "NO-ANE"
  fi
}

# Outputs (filled by one of the two sweep modes below):
WINNER_DESC=""        # human label for the bake summary
WINNER_ENV_KV=""      # newline KEY=VAL block to merge into the profile env map

if nax_capable && [ "$NO_BACKEND_SWEEP" != "1" ]; then
  [ -n "$BACKENDS" ] || BACKENDS="$(ds4_backend_default_list)"
  echo "== tune_profile (NAX backend sweep): chip='$CHIP' ram=${RAM_GIB}GiB ctx=$CTX chunk=$CHUNK =="
  echo "== backends: $BACKENDS  (ANE backends also sweep min_refs: $MINREFS) =="
  echo "== each run loads the full model in RAM; this takes a while. Run ALONE on the box. =="
  printf "%-14s %-10s %-12s %-10s\n" "backend" "min_refs" "prefill_tps" "note"
  best_tps="-1"; best_backend=""; best_refs="-"; best_extra=""
  RESULT_ROWS=""   # newline list of: backend<TAB>min_refs<TAB>tps<TAB>note
  record() {  # backend min_refs tps note
    RESULT_ROWS="${RESULT_ROWS}${1}	${2}	${3:-FAIL}	${4}
"
  }
  # A flagged ANE row (FELL-BACK to grouped, or NO-ANE) is NOT a real ANE measurement,
  # so it is recorded for transparency but barred from winning the bake.
  consider() {  # backend min_refs tps extra note
    [ -n "$3" ] || return 0
    [ -z "$5" ] || return 0
    if awk "BEGIN{exit !($3 > $best_tps)}"; then
      best_tps="$3"; best_backend="$1"; best_refs="$2"; best_extra="$4"
    fi
  }
  for b in $BACKENDS; do
    extra="$(ds4_backend_variant_env "$b")"
    if [ "$extra" = "__UNKNOWN__" ]; then echo "  (skip unknown backend '$b')"; continue; fi
    if ds4_backend_is_ane "$b"; then
      for mr in $MINREFS; do
        e="$extra DS4_RESIDENT_MOE_ANE_MIN_REFS=$mr"
        tps="$(measure_prefill "$e")"; note="$(engage_note "$e")"
        printf "%-14s %-10s %-12s %-10s\n" "$b" "$mr" "${tps:-FAIL}" "$note"
        record "$b" "$mr" "$tps" "$note"
        consider "$b" "$mr" "$tps" "$e" "$note"
      done
    else
      tps="$(measure_prefill "$extra")"; note="$(engage_note "$extra")"
      printf "%-14s %-10s %-12s %-10s\n" "$b" "-" "${tps:-FAIL}" "$note"
      record "$b" "-" "$tps" "$note"
      consider "$b" "-" "$tps" "$extra" "$note"
    fi
  done
  [ -n "$best_backend" ] || { echo "error: all backend runs failed" >&2; exit 1; }
  echo "== best backend: $best_backend${best_refs:+ (min_refs=$best_refs)} @ ${best_tps} t/s prefill =="
  # Results JSON: every measured row + the winner + machine context.
  CHIP="$CHIP" RAM_GIB="$RAM_GIB" CTX="$CTX" CHUNK="$CHUNK" \
  BEST_BACKEND="$best_backend" BEST_REFS="$best_refs" BEST_TPS="$best_tps" \
  RESULT_ROWS="$RESULT_ROWS" RESULTS="$RESULTS" python3 - <<'PY'
import json, os
rows=[]
for line in os.environ.get("RESULT_ROWS","").splitlines():
    if not line.strip(): continue
    b,mr,tps,note=(line.split("\t")+["","","",""])[:4]
    rows.append({"backend":b, "min_refs":(None if mr=="-" else mr),
                 "prefill_tps":(None if tps in ("","FAIL") else float(tps)),
                 "note":(note or None)})
doc={"chip":os.environ["CHIP"], "ram_gib":int(os.environ["RAM_GIB"]),
     "ctx":int(os.environ["CTX"]), "chunk":int(os.environ["CHUNK"]),
     "winner":{"backend":os.environ["BEST_BACKEND"],
               "min_refs":(None if os.environ["BEST_REFS"]=="-" else os.environ["BEST_REFS"]),
               "prefill_tps":float(os.environ["BEST_TPS"])},
     "results":rows}
with open(os.environ["RESULTS"],"w") as f: json.dump(doc,f,indent=2); f.write("\n")
print("== wrote results JSON -> %s (%d rows) ==" % (os.environ["RESULTS"], len(rows)))
PY
  WINNER_DESC="backend=$best_backend min_refs=$best_refs prefill=${best_tps} t/s"
  # Bake the winner's full env, EXCEPT the MULMM sentinel which means "no resident
  # backend envs" (pure GPU) -> bake nothing backend-specific.
  if [ "$best_extra" != "MULMM" ]; then
    { ds4_backend_common_env; printf '%s\n' $best_extra; } > /dev/null  # validate splitting
    WINNER_ENV_KV="$(ds4_backend_common_env; for kv in $best_extra; do printf '%s\n' "$kv"; done)"
  fi
else
  # ---- legacy min_refs-only sweep (non-NAX chips), original behaviour ----
  echo "== tune_profile: chip='$CHIP' ram=${RAM_GIB}GiB ctx=$CTX chunk=$CHUNK gen=$GEN =="
  echo "== sweeping DS4_RESIDENT_MOE_ANE_MIN_REFS in: $MINREFS (each run loads the model; this takes a while) =="
  OUT=$(mktemp)
  best_refs=""; best_pre="-1"; best_gen="0"
  printf "%-10s %-12s %-12s\n" "min_refs" "prefill_tps" "decode_tps"
  for mr in $MINREFS; do
    [ "$COOLDOWN" -gt 0 ] && sleep "$COOLDOWN"
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
  WINNER_DESC="min_refs=$best_refs prefill=${best_pre} t/s decode=${best_gen} t/s"
  WINNER_ENV_KV=$(printf '%s\n' \
    "DS4_RESIDENT_MOE_ANE_MIN_REFS=$best_refs" \
    "DS4_RESIDENT_MOE_ANE_MAX_REFS=1024" \
    "DS4_RESIDENT_MOE_ANE_QUEUE=8")
fi

# ---- build + show the profile entry (CTX_GROW knobs + winner env) ----
echo "== proposed profile entry for [$CHIP / ${RAM_MATCH}GiB+] : $WINNER_DESC =="
CHIP="$CHIP" RAM_MATCH="$RAM_MATCH" GROW_BLOCK="$GROW_BLOCK" WINNER_ENV_KV="$WINNER_ENV_KV" \
python3 - <<'PY'
import json, os
chip=os.environ["CHIP"]; ram=int(os.environ["RAM_MATCH"]); grow=os.environ["GROW_BLOCK"]
env={"DS4_CTX_GROW":"1","DS4_CTX_GROW_BLOCK":grow}
for line in os.environ.get("WINNER_ENV_KV","").splitlines():
    line=line.strip()
    if not line or "=" not in line: continue
    k,v=line.split("=",1); env[k]=v
print(json.dumps({"match":{"chip":chip,"min_ram_gib":ram},"env":env}, indent=2))
PY

if [ "$APPLY" != "1" ]; then
  printf "Merge this into %s? [y/N] " "$PROFILE"; read -r ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not applied."; exit 0;; esac
fi

# Merge with python3 (replace any existing entry with the same chip+ram, else prepend).
CHIP="$CHIP" RAM_MATCH="$RAM_MATCH" GROW_BLOCK="$GROW_BLOCK" WINNER_ENV_KV="$WINNER_ENV_KV" PROFILE="$PROFILE" \
python3 - <<'PY'
import json, os
path=os.environ["PROFILE"]; chip=os.environ["CHIP"]; ram=int(os.environ["RAM_MATCH"]); grow=os.environ["GROW_BLOCK"]
env={"DS4_CTX_GROW":"1","DS4_CTX_GROW_BLOCK":grow}
for line in os.environ.get("WINNER_ENV_KV","").splitlines():
    line=line.strip()
    if not line or "=" not in line: continue
    k,v=line.split("=",1); env[k]=v
entry={"match":{"chip":chip,"min_ram_gib":ram},"env":env}
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
