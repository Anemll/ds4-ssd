#!/usr/bin/env bash
# validate_ane_nax.sh — correctness gate for the ANE+NAX resident prefill backend
# (DS4_RESIDENT_MOE_ANE_NAX_HYBRID). Run on the M5 BEFORE trusting ANE+NAX throughput
# or adding `ane_nax` to the tune_profile.sh backend sweep.
#
# Method — controlled greedy-generation comparison. ane_gpu and ane_nax run the SAME
# resident ANE dedup prefill with the SAME ANE hot-expert split; they differ ONLY in
# the cold-expert ("below ANE min_refs") GPU kernel: ane_gpu uses classic int8, ane_nax
# uses NAX (matmul2d half). So if ane_nax's greedy output matches the ane_gpu reference,
# the new NAX cold-expert routing + scatter is correct. For each NAX_HALF_TILE x
# FUSED_GATE_UP combo a run PASSES iff:
#   1. it didn't crash, and ane_scatter_failures == 0, and
#   2. the NAX cold path actually ran (mpp_groups > 0), and
#   3. its greedy continuation has >= 50% first-line word overlap with the ane_gpu
#      reference. (Exact match is too strict: cold experts are the bulk of compute, so
#      NAX-half vs int8 precision can legitimately flip a token. A real plumbing/scatter
#      bug — wrong expert, double-scatter, tile corruption — yields garbage, i.e. near-0
#      overlap, so 50% cleanly separates "correct" from "broken".)
#
# Uses RESIDENT mode (full GGUF in RAM, --moe-mode off). The prompt is auto-trimmed to
# fit --ctx. min_refs defaults to 256 so a large band of cold experts (refs in [64,256))
# exercises the NAX path. Run ALONE on the box (correctness doesn't need cooldown).
#
# Usage: ./validate_ane_nax.sh -m MODEL.gguf -p PROMPT.txt [opts]
#   --ctx N        prefill/context size (default 4096)
#   --bytes N      bytes of PROMPT to use (auto-trimmed; default 10000 ~ 2.5K tok)
#   --tokens N     greedy tokens to generate for the comparison (default 24)
#   --tiles "L"    NAX_HALF_TILE values to sweep (default "32 64 128 256")
#   --fused "L"    NAX_FUSED_GATE_UP values to sweep (default "0 1")
#   --min-refs N   ANE min_refs (default 256)
#   --bin PATH     ds4 binary (default ./ds4)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
# shellcheck source=ds4_backend_env.sh
. "$ROOT/ds4_backend_env.sh"

BIN=./ds4; MODEL=""; PROMPT=""; CTX=4096; PBYTES=10000; TOKENS=24
TILES="32 64 128 256"; FUSED="0 1"; MINREFS=256
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -p|--prompt-file) PROMPT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --bytes) PBYTES="$2"; shift 2;;
    --tokens) TOKENS="$2"; shift 2;;
    --tiles) TILES="$2"; shift 2;;
    --fused) FUSED="$2"; shift 2;;
    --min-refs) MINREFS="$2"; shift 2;;
    --bin) BIN="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] || { echo "error: -m MODEL.gguf required" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "error: $BIN not found/executable (run 'make ds4')" >&2; exit 2; }
[ -n "$PROMPT" ] && [ -f "$PROMPT" ] || { echo "error: -p PROMPT.txt required" >&2; exit 2; }

CHUNK=$CTX; [ "$CHUNK" -gt 16384 ] && CHUNK=16384
GRAPH_RAW_CAP=$(( CHUNK + 512 ))
LOCK="${TMPDIR:-/tmp}/ds4-validate.lock"
PFILE="${TMPDIR:-/tmp}/ds4_validate_prompt.txt"
head -c "$PBYTES" "$PROMPT" > "$PFILE"
OUTTXT="${TMPDIR:-/tmp}/ds4_validate_out.txt"
LOG="${TMPDIR:-/tmp}/ds4_validate_log.txt"

# run_gen "<variant>" "<extra env>" -> sets R_OUT (first generated line), R_MPP, R_ANE,
#   R_SCATTER_FAIL, R_EXIT. stdout = generated text, stderr = ds4 logs.
run_gen() {
  local variant="$1" extra="$2"; local envs=( DS4_LOCK_FILE="$LOCK" )
  while IFS= read -r kv; do [ -n "$kv" ] && envs+=( "$kv" ); done < <(ds4_backend_common_env)
  # shellcheck disable=SC2206
  for kv in $(ds4_backend_variant_env "$variant"); do envs+=( "$kv" ); done
  # shellcheck disable=SC2206
  [ -n "$extra" ] && envs+=( $extra )
  envs+=( DS4_RESIDENT_MOE_ANE_MIN_REFS="$MINREFS" DS4_RESIDENT_MOE_MPP_STATS=1
          DS4_METAL_PREFILL_CHUNK="$CHUNK" DS4_METAL_GRAPH_RAW_CAP="$GRAPH_RAW_CAP" )
  env "${envs[@]}" "$BIN" -m "$MODEL" --metal --moe-mode off --prompt-file "$PFILE" \
      --temp 0 --tokens "$TOKENS" --ctx "$CTX" >"$OUTTXT" 2>"$LOG"; R_EXIT=$?
  R_OUT="$(grep -vE '^[[:space:]]*$' "$OUTTXT" | head -1 | cut -c1-72)"
  R_MPP="$(grep -oE ' mpp_groups=[0-9]+' "$LOG" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')"
  R_ANE="$(grep -oE 'ane_groups=[0-9]+' "$LOG" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')"
  R_SCATTER_FAIL="$(grep -oE 'ane_scatter_failures=[0-9]+' "$LOG" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')"
}

echo "== validate ANE+NAX: model=$(basename "$MODEL") ctx=$CTX min_refs=$MINREFS tokens=$TOKENS =="
echo "== chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null) =="

echo "-- reference: ane_gpu (ANE hot + classic-int8 cold) --"
run_gen ane_gpu ""
REF_OUT="$R_OUT"
printf "  exit=%s ane_groups=%s mpp_groups=%s scatter_fail=%s\n  ref_out: %s\n" \
  "$R_EXIT" "$R_ANE" "$R_MPP" "$R_SCATTER_FAIL" "$REF_OUT"
if [ "$R_EXIT" != 0 ] || [ -z "$REF_OUT" ]; then
  echo "FAIL: ane_gpu reference itself did not produce output (check model/prompt/ctx)." >&2; exit 1
fi

echo "-- ANE+NAX sweep (tile x fused), compared to the ane_gpu reference --"
printf "  %-6s %-6s %-6s %-5s %-6s %-9s %-6s %-6s\n" "tile" "fused" "exit" "ane" "nax" "scat_fail" "overlap" "PASS"
fail=0; ran=0
for tile in $TILES; do
  for fu in $FUSED; do
    extra="DS4_RESIDENT_MOE_NAX_HALF_TILE=$tile"
    [ "$fu" = "1" ] && extra="$extra DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1 DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0"
    run_gen ane_nax "$extra"; ran=$((ran+1))
    # first-line word overlap with the reference (fraction of ref words present, in %)
    ovl="$(awk -v a="$REF_OUT" -v b="$R_OUT" 'BEGIN{
      na=split(a,A," "); for(i=1;i<=na;i++) seen[A[i]]=1;
      nb=split(b,B," "); hit=0; for(i=1;i<=nb;i++) if(B[i] in seen) hit++;
      printf "%d", (nb? 100*hit/nb : 0)}')"
    pass=1
    [ "$R_EXIT" = 0 ] || pass=0
    [ "${R_SCATTER_FAIL:-0}" = 0 ] || pass=0
    [ "${R_MPP:-0}" -gt 0 ] || pass=0
    [ "${ovl:-0}" -ge 50 ] || pass=0
    [ "$pass" = 1 ] || { fail=$((fail+1)); echo "    out: $R_OUT"; }
    printf "  %-6s %-6s %-6s %-5s %-6s %-9s %-6s %-6s\n" "$tile" "$fu" "$R_EXIT" "${R_ANE:-0}" "${R_MPP:-0}" "${R_SCATTER_FAIL:-0}" "${ovl}%" "$([ $pass = 1 ] && echo yes || echo NO)"
  done
done
rm -f "$OUTTXT" "$LOG" "$PFILE"
echo "== $((ran-fail))/$ran combos passed =="
if [ "$fail" -ne 0 ]; then
  echo "FAIL: $fail combo(s) failed (crash, scatter failure, NAX cold path never ran, or output != ref)." >&2
  echo "      Do NOT add ane_nax to the tune_profile.sh sweep until this is clean." >&2
  exit 1
fi
echo "PASS: ANE+NAX prefill matches the ane_gpu reference across all tiles/fused settings."
