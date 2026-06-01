#!/usr/bin/env bash
# Sweep the ANE‖NAX overlap + token-balance knobs and measure resident prefill t/s,
# hunting for a config where ANE-NAX beats BOTH ane_gpu and nax_int8.
#
# Each line of the config file (arg 1) is: LABEL key=val key=val ...
# Extra env beyond the ane_nax variant block. Results appended to RESULTS (arg 2).
# Baselines to beat are passed as ANE_GPU / NAX_INT8 env (defaults 315 / 531 @16384).
#
# Run ALONE on the box. 45s cooldown between configs (M5 thermals skew results).
set -u
cd "$(dirname "$0")"
. ./ds4_backend_env.sh

MODEL="${MODEL:-/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PROMPT="${PROMPT:-/tmp/big_prompt.txt}"
CTX="${CTX:-16384}"
COOLDOWN="${COOLDOWN:-45}"
ANE_GPU_REF="${ANE_GPU:-315}"
NAX_INT8_REF="${NAX_INT8:-531}"
CFG="${1:-/tmp/ane_nax_cfgs.txt}"
RESULTS="${2:-ane_nax_overlap_results.tsv}"

[ -f "$RESULTS" ] || printf 'label\tprefill_ts\tane_groups\tane_refs\tane_start_fail\tane_scatter_fail\tmpp_groups\tmpp_refs\troute\tbeats_gpu\tbeats_int8\tenv\n' >"$RESULTS"

sumcnt() { grep -oE "$1=[0-9]+" "$2" 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; }

run_cfg() {
  local label="$1"; shift
  local extra=( "$@" )
  local envs=( DS4_PROFILE=none DS4_LOCK_FILE=/tmp/ds4-bench.lock DS4_RESIDENT_MOE_MPP_STATS=1
               DS4_RESIDENT_MOE_KERNEL_LOG=1 )
  while IFS= read -r kv; do [ -n "$kv" ] && envs+=( "$kv" ); done < <(ds4_backend_common_env)
  for kv in $(ds4_backend_variant_env ane_nax); do envs+=( "$kv" ); done
  envs+=( "${extra[@]}" )
  local log=/tmp/anenax_${label}.log
  env "${envs[@]}" ./ds4-bench -m "$MODEL" --metal --moe-mode off --warm-weights \
      --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens 1 --prompt-file "$PROMPT" >"$log" 2>&1 &
  local pid=$! prev="" stable=0
  for _w in $(seq 1 50); do
    sleep 5
    grep -q "t/s" "$log" 2>/dev/null && break
    local cur=$(wc -l < "$log" 2>/dev/null)
    if [ "$cur" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi; prev="$cur"
    [ "$stable" -ge 6 ] && { echo "  ($label STALLED — killed)"; kill -9 $pid 2>/dev/null; break; }
  done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  local ts ane ane_refs ane_sf ane_scf mpp mpp_refs route
  ts=$(grep -oE '[0-9]+\.[0-9]+ t/s' "$log" | head -1 | grep -oE '[0-9]+\.[0-9]+')
  ane=$(sumcnt ane_groups "$log")
  ane_refs=$(sumcnt ane_refs "$log")
  ane_sf=$(sumcnt ane_start_failures "$log")
  ane_scf=$(sumcnt ane_scatter_failures "$log")
  mpp=$(grep -oE ' mpp_groups=[0-9]+' "$log" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
  mpp_refs=$(grep -oE ' mpp_refs=[0-9]+' "$log" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
  route=$(grep -oE 'route=[^ ]+' "$log" | sort -u | tr '\n' ',' | sed 's/,$//')
  [ -z "$ts" ] && ts=0
  [ -z "$route" ] && route="(none/ANE-only?)"
  local bg bi
  bg=$(awk -v a="$ts" -v b="$ANE_GPU_REF" 'BEGIN{print (a>b)?"Y":"n"}')
  bi=$(awk -v a="$ts" -v b="$NAX_INT8_REF" 'BEGIN{print (a>b)?"Y":"n"}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$ts" "$ane" "$ane_refs" "$ane_sf" "$ane_scf" "$mpp" "$mpp_refs" "$route" "$bg" "$bi" "${extra[*]}" >>"$RESULTS"
  printf '%-20s %8s t/s  ane_grp=%-5s ane_ref=%-7s ane_startfail=%-5s mpp_grp=%-5s beats_gpu=%s beats_int8=%s [%s]\n' \
    "$label" "$ts" "$ane" "$ane_refs" "$ane_sf" "$mpp" "$bg" "$bi" "$route"
}

i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  if [ "$i" -gt 0 ]; then sleep "$COOLDOWN"; fi
  # shellcheck disable=SC2086
  run_cfg $line
  i=$((i+1))
done < "$CFG"
echo "SWEEP_DONE ($i configs) -> $RESULTS"
