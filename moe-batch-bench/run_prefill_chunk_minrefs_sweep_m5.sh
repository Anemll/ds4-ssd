#!/usr/bin/env bash
# Sweep M5 (32 GB) routed-MoE ANE-vs-GPU prefill across prompt sizes, Metal
# prefill chunk sizes, and ANE routing thresholds (min_refs).
#
# SSD slot-bank only: this box cannot host the resident model. Prompts are
# synthesized to size by run_{ane,gpu}_prefill_profile_m5.sh (no coding_N.txt
# corpus on this box), so we drive them via DS4_PREFILL_SIZE, not a fixed file.
#
# Defaults match the M3U sweep segments:
#   - ANE routed expert path enabled, single-cluster M5 defaults
#   - dense shared-expert ANE and O-proj ANE forced off
#   - GPU baseline run once per prompt/chunk/repeat
#
# Segment 1 (fixed chunk=16K, vary prompt + min_refs):
#   PROMPTS="128 256 1k 4k 8k 16k" CHUNKS="16384" MIN_REFS="32 64 128 256 512" \
#     ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5.sh
#
# Segment 2 (8K prompt, vary chunk + min_refs):
#   PROMPTS="8k" CHUNKS="1024 2048 4096 8192 16384" MIN_REFS="32 64 128 256 512" \
#     ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

read -r -a PROMPT_LIST <<< "${PROMPTS:-128 256 1k 4k 8k 16k}"
read -r -a CHUNK_LIST <<< "${CHUNKS:-1024 2048 4096 8192 16384}"
read -r -a MIN_REF_LIST <<< "${MIN_REFS:-32 64 128 256 512}"

REPEATS="${REPEATS:-1}"
RUN_GPU="${RUN_GPU:-1}"
RUN_ANE="${RUN_ANE:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
QUIET="${QUIET:-1}"
SLOTS="${DS4_SLOTS:-48}"

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs/prefill_chunk_minrefs_m5_$SWEEP_ID}"
RESULTS="$OUT_DIR/results.csv"
mkdir -p "$OUT_DIR"

ctx_for_prompt() {
  local label="$1"
  if [[ -n "${SWEEP_CTX:-}" ]]; then
    echo "$SWEEP_CTX"
    return
  fi
  case "$label" in
    100|128|256|512) echo 2048 ;;
    1k) echo 3072 ;;
    2k) echo 4096 ;;
    4k) echo 6144 ;;
    6k) echo 8192 ;;
    8k) echo 10000 ;;
    10k) echo 12000 ;;
    12k) echo 14336 ;;
    16k) echo 20000 ;;
    *) echo "${DS4_CTX:-20000}" ;;
  esac
}

extract_prefill_tps() {
  awk '/prefill:/ { v=$3; sub(/,/, "", v) } END { print v }' "$1"
}

extract_kv_last() {
  local file="$1"
  local pattern="$2"
  local key="$3"
  awk -v pat="$pattern" -v key="$key" '
    $0 ~ pat {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" key "=")) {
          v = $i
          sub("^" key "=", "", v)
          sub(/[,)]$/, "", v)
        }
      }
    }
    END { print v }
  ' "$file"
}

append_results_row() {
  local mode="$1"
  local label="$2"
  local ctx="$3"
  local chunk="$4"
  local min_refs="$5"
  local repeat="$6"
  local status="$7"
  local summary="$8"
  local log="$9"

  local tps="" tokens=""
  local stage_calls="" stage_pread_ms="" stage_upload_ms="" stage_rate=""
  local ane_calls="" ane_ok="" ane_eval_calls=""
  local ane_chunk_refs="" ane_pad_util="" ane_wall_ms="" ane_eval_ms=""
  local finish_join_ane_ms="" dequant_wait_ms="" output_convert_ms=""
  local same_layer_post_ms="" cross_layer_post_ms="" hit_rate=""

  if [[ -f "$summary" ]]; then
    tps="$(extract_prefill_tps "$summary")"
    tokens="$(extract_kv_last "$summary" "prefill total tokens=" "tokens")"
    stage_calls="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "calls")"
    stage_pread_ms="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "pread")"
    stage_upload_ms="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "upload")"
    stage_rate="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "stage_rate")"
    ane_calls="$(extract_kv_last "$summary" "ANE prefill stats" "calls")"
    ane_ok="$(extract_kv_last "$summary" "ANE prefill stats" "ok")"
    ane_eval_calls="$(extract_kv_last "$summary" "ANE prefill stats" "eval_calls")"
    ane_chunk_refs="$(extract_kv_last "$summary" "ANE prefill chunks" "refs")"
    ane_pad_util="$(extract_kv_last "$summary" "ANE prefill chunks" "pad_util")"
    ane_wall_ms="$(extract_kv_last "$summary" "ANE prefill timing" "ane_wall_est")"
    ane_eval_ms="$(extract_kv_last "$summary" "ANE prefill timing" "ane_eval")"
    finish_join_ane_ms="$(extract_kv_last "$summary" "ANE prefill timing4" "finish_join_ane")"
    dequant_wait_ms="$(extract_kv_last "$summary" "ANE prefill timing3" "dequant_wait")"
    output_convert_ms="$(extract_kv_last "$summary" "ANE prefill timing3" "output_convert")"
    same_layer_post_ms="$(extract_kv_last "$summary" "Flash-MoE prefill pread issue stats" "same_layer_post")"
    cross_layer_post_ms="$(extract_kv_last "$summary" "Flash-MoE prefill pread issue stats" "cross_layer_post")"
    hit_rate="$(extract_kv_last "$summary" "Flash-MoE slot-bank stats" "hit-rate")"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$mode" "$label" "$ctx" "$chunk" "$min_refs" "$repeat" "$status" "$tps" "$tokens" \
    "$stage_calls" "$stage_pread_ms" "$stage_upload_ms" "$stage_rate" \
    "$ane_calls" "$ane_ok" "$ane_eval_calls" "$ane_chunk_refs" "$ane_pad_util" \
    "$ane_wall_ms" "$ane_eval_ms" "$finish_join_ane_ms" "$dequant_wait_ms" "$output_convert_ms" \
    "$same_layer_post_ms" "$cross_layer_post_ms" "$log" >> "$RESULTS"
}

run_driver() {
  local driver_log="$1"
  shift
  if [[ "$QUIET" == "1" ]]; then
    "$@" > "$driver_log" 2>&1
  else
    "$@"
  fi
}

run_profile() {
  local mode="$1"
  local label="$2"
  local ctx="$3"
  local chunk="$4"
  local min_refs="$5"
  local repeat="$6"
  local script=""
  local run_name=""

  if [[ "$mode" == "gpu" ]]; then
    script="./run_gpu_prefill_profile_m5.sh"
    run_name="sweep_gpu_${label}_c${chunk}_r${repeat}"
  else
    script="./run_ane_prefill_profile_m5.sh"
    run_name="sweep_ane_${label}_c${chunk}_mr${min_refs}_r${repeat}"
  fi

  local log="$OUT_DIR/$run_name.log"
  local summary="$OUT_DIR/$run_name.summary.txt"
  local driver_log="$OUT_DIR/$run_name.driver.log"
  local status="ok"

  echo "==> $mode prompt=$label ctx=$ctx chunk=$chunk min_refs=$min_refs repeat=$repeat"
  if [[ "$mode" == "gpu" ]]; then
    if ! run_driver "$driver_log" env \
      DS4_LOG_DIR="$OUT_DIR" \
      DS4_RUN_NAME="$run_name" \
      DS4_PREFILL_SIZE="$label" \
      DS4_CTX="$ctx" \
      DS4_PREFILL_CHUNK="$chunk" \
      DS4_SLOTS="$SLOTS" \
      "$script"; then
      status="fail"
    fi
  else
    if ! run_driver "$driver_log" env \
      DS4_LOG_DIR="$OUT_DIR" \
      DS4_RUN_NAME="$run_name" \
      DS4_PREFILL_SIZE="$label" \
      DS4_CTX="$ctx" \
      DS4_PREFILL_CHUNK="$chunk" \
      DS4_SLOTS="$SLOTS" \
      DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="$min_refs" \
      DS4_FLASH_MOE_ANE_MIN_REFS="$min_refs" \
      DS4_FLASH_MOE_ANE_SHARED_EXPERT="${DS4_FLASH_MOE_ANE_SHARED_EXPERT:-0}" \
      DS4_FLASH_MOE_ANE_OUTPUT_PROJ="${DS4_FLASH_MOE_ANE_OUTPUT_PROJ:-0}" \
      "$script"; then
      status="fail"
    fi
  fi

  append_results_row "$mode" "$label" "$ctx" "$chunk" "$min_refs" "$repeat" "$status" "$summary" "$log"
  if [[ "$status" != "ok" && "$CONTINUE_ON_FAIL" != "1" ]]; then
    echo "error: $run_name failed; see $log" >&2
    exit 1
  fi
}

printf 'mode,prompt,ctx,chunk,min_refs,repeat,status,prefill_tps,tokens,stage_calls,stage_pread_ms,stage_upload_ms,stage_rate_mibs,ane_calls,ane_ok,ane_eval_calls,ane_chunk_refs,ane_pad_util,ane_wall_ms,ane_eval_ms,finish_join_ane_ms,dequant_wait_ms,output_convert_ms,same_layer_post_ms,cross_layer_post_ms,log\n' > "$RESULTS"

echo "box: M5 (32 GB), SSD slot-bank only (slots=$SLOTS)"
echo "results: $RESULTS"
echo "prompts: ${PROMPT_LIST[*]}"
echo "chunks: ${CHUNK_LIST[*]}"
echo "min_refs: ${MIN_REF_LIST[*]}"
echo

for label in "${PROMPT_LIST[@]}"; do
  ctx="$(ctx_for_prompt "$label")"
  for chunk in "${CHUNK_LIST[@]}"; do
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
      if [[ "$RUN_GPU" == "1" ]]; then
        run_profile gpu "$label" "$ctx" "$chunk" 0 "$repeat"
      fi
      if [[ "$RUN_ANE" == "1" ]]; then
        for min_refs in "${MIN_REF_LIST[@]}"; do
          run_profile ane "$label" "$ctx" "$chunk" "$min_refs" "$repeat"
        done
      fi
    done
  done
done

echo
echo "completed: $RESULTS"
echo "best by prompt/chunk (mode,prompt,ctx,chunk,min_refs,...,prefill_tps):"
awk -F, '
  NR == 1 { next }
  $7 == "ok" && $8 != "" {
    key = $2 "," $4
    if (!(key in best) || $8 + 0 > best[key] + 0) {
      best[key] = $8
      brow[key] = $0
    }
    gkey = $1 "," $2 "," $4
    if (!(gkey in gbest) || $8 + 0 > gbest[gkey] + 0) {
      gbest[gkey] = $8
      grow[gkey] = $0
    }
  }
  END {
    for (key in brow) print brow[key]
  }
' "$RESULTS" | sort -t, -k2,2V -k4,4n
