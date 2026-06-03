#!/usr/bin/env bash
# Sweep M5 Max routed-MoE ANE prefill routing thresholds across prompt sizes
# and Metal prefill chunk sizes -- the M5 Max counterpart of
# run_prefill_chunk_minrefs_sweep_m3u.sh.
#
# M5 Max specifics:
#   - ANE path is SINGLE-CLUSTER (run_ane_prefill_profile_m5max.sh: DUAL=0, THREADS=1,
#     shared-expert + O-proj ANE off).
#   - GPU baseline is NAX-int8 (W8A8) by default -- the real M5 GPU competitor
#     (run_gpu_prefill_profile_m5max.sh). Set GPU_ROUTE=alu for the plain mulmm baseline.
#
# Example fixed-chunk prompt sweep (segment A):
#   PROMPTS="128 256 1k 4k 8k 16k" CHUNKS="16384" MIN_REFS="32 64 128 256 512" \
#     ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5max.sh
#
# Example chunk sweep at 8K (segment B):
#   PROMPTS="8k" CHUNKS="1024 2048 4096 8192 16384" MIN_REFS="32 64 128 256 512" \
#     ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5max.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROMPT_DIR="${PROMPT_DIR:-/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding}"
if [[ ! -d "$PROMPT_DIR" && -d /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding ]]; then
  PROMPT_DIR=/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding
fi

read -r -a PROMPT_LIST <<< "${PROMPTS:-128 256 1k 4k 8k 16k}"
read -r -a CHUNK_LIST <<< "${CHUNKS:-1024 2048 4096 8192 16384}"
read -r -a MIN_REF_LIST <<< "${MIN_REFS:-32 64 128 256 512}"

REPEATS="${REPEATS:-1}"
RUN_GPU="${RUN_GPU:-1}"
RUN_ANE="${RUN_ANE:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
QUIET="${QUIET:-1}"
DS4_SLOTS="${DS4_SLOTS:-96}"
GPU_ROUTE="${GPU_ROUTE:-nax_int8}"   # nax_int8 (default M5 GPU competitor) | alu
COOLDOWN="${COOLDOWN:-0}"            # seconds to sleep between runs (thermal)

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${DS4_LOG_DIR:-$ROOT/moe-batch-bench/profile_runs/prefill_chunk_minrefs_m5max_$SWEEP_ID}"
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
    14k) echo 17000 ;;
    16k) echo 20000 ;;
    *) echo "${DS4_CTX:-20000}" ;;
  esac
}

prompt_file_for_label() {
  local label="$1"
  if [[ "$label" == /* || "$label" == ./* ]]; then
    echo "$label"
    return
  fi
  echo "$PROMPT_DIR/coding_${label}.txt"
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

sum_kv() {
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
          sum += v + 0
          n++
        }
      }
    }
    END { if (n > 0) print sum; }
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

  local tps=""
  local stage_calls="" stage_pread_ms="" stage_upload_ms="" stage_rate=""
  local ane_calls="" ane_ok="" ane_eval_calls="" ane_refs="" ane_groups=""
  local ane_chunk_refs="" ane_pad_util="" ane_wall_ms="" ane_eval_ms=""
  local finish_join_ane_ms="" dequant_wait_ms="" output_convert_ms="" same_layer_post_ms="" cross_layer_post_ms=""

  if [[ -f "$summary" ]]; then
    tps="$(extract_prefill_tps "$summary")"
    stage_calls="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "calls")"
    stage_pread_ms="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "pread")"
    stage_upload_ms="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "upload")"
    stage_rate="$(extract_kv_last "$summary" "Flash-MoE prefill stage stats" "stage_rate")"
    ane_calls="$(extract_kv_last "$summary" "ANE prefill stats" "calls")"
    ane_ok="$(extract_kv_last "$summary" "ANE prefill stats" "ok")"
    ane_eval_calls="$(extract_kv_last "$summary" "ANE prefill stats" "eval_calls")"
    ane_refs="$(sum_kv "$summary" "Flash-MoE hybrid prefill layer" "ane_refs")"
    ane_groups="$(sum_kv "$summary" "Flash-MoE hybrid prefill layer" "ane_groups")"
    ane_chunk_refs="$(extract_kv_last "$summary" "ANE prefill chunks" "refs")"
    ane_pad_util="$(extract_kv_last "$summary" "ANE prefill chunks" "pad_util")"
    ane_wall_ms="$(extract_kv_last "$summary" "ANE prefill timing" "ane_wall_est")"
    ane_eval_ms="$(extract_kv_last "$summary" "ANE prefill timing" "ane_eval")"
    finish_join_ane_ms="$(extract_kv_last "$summary" "ANE prefill timing4" "finish_join_ane")"
    dequant_wait_ms="$(extract_kv_last "$summary" "ANE prefill timing3" "dequant_wait")"
    output_convert_ms="$(extract_kv_last "$summary" "ANE prefill timing3" "output_convert")"
    same_layer_post_ms="$(extract_kv_last "$summary" "Flash-MoE prefill pread issue stats" "same_layer_post")"
    cross_layer_post_ms="$(extract_kv_last "$summary" "Flash-MoE prefill pread issue stats" "cross_layer_post")"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$mode" "$label" "$ctx" "$chunk" "$min_refs" "$repeat" "$status" "$tps" \
    "$stage_calls" "$stage_pread_ms" "$stage_upload_ms" "$stage_rate" \
    "$ane_calls" "$ane_ok" "$ane_eval_calls" "$ane_refs" "$ane_groups" "$ane_chunk_refs" "$ane_pad_util" \
    "$ane_wall_ms" "$ane_eval_ms" "$finish_join_ane_ms" "$dequant_wait_ms" "$output_convert_ms" \
    "$same_layer_post_ms" "$cross_layer_post_ms" "$log" "$summary" >> "$RESULTS"
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
  local prompt_file="$3"
  local ctx="$4"
  local chunk="$5"
  local min_refs="$6"
  local repeat="$7"
  local script=""
  local run_name=""

  if [[ "$mode" == "gpu" ]]; then
    script="./run_gpu_prefill_profile_m5max.sh"
    run_name="sweep_gpu_${label}_c${chunk}_r${repeat}"
  else
    script="./run_ane_prefill_profile_m5max.sh"
    run_name="sweep_ane_${label}_c${chunk}_mr${min_refs}_r${repeat}"
  fi

  local log="$OUT_DIR/$run_name.log"
  local summary="$OUT_DIR/$run_name.summary.txt"
  local driver_log="$OUT_DIR/$run_name.driver.log"
  local status="ok"

  echo "==> $mode prompt=$label ctx=$ctx chunk=$chunk min_refs=$min_refs repeat=$repeat slots=$DS4_SLOTS"
  if [[ "$mode" == "gpu" ]]; then
    if ! run_driver "$driver_log" env \
      DS4_LOG_DIR="$OUT_DIR" \
      DS4_RUN_NAME="$run_name" \
      DS4_PROMPT_FILE="$prompt_file" \
      DS4_CTX="$ctx" \
      DS4_PREFILL_CHUNK="$chunk" \
      DS4_SLOTS="$DS4_SLOTS" \
      DS4_GPU_ROUTE="$GPU_ROUTE" \
      DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 \
      DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0 \
      "$script"; then
      status="fail"
    fi
  else
    if ! run_driver "$driver_log" env \
      DS4_LOG_DIR="$OUT_DIR" \
      DS4_RUN_NAME="$run_name" \
      DS4_PROMPT_FILE="$prompt_file" \
      DS4_CTX="$ctx" \
      DS4_PREFILL_CHUNK="$chunk" \
      DS4_SLOTS="$DS4_SLOTS" \
      DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS="$min_refs" \
      DS4_FLASH_MOE_ANE_MIN_REFS="$min_refs" \
      DS4_RESIDENT_MOE_ANE_MIN_REFS="$min_refs" \
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
  [[ "$COOLDOWN" -gt 0 ]] && sleep "$COOLDOWN" || true
}

printf 'mode,prompt,ctx,chunk,min_refs,repeat,status,prefill_tps,stage_calls,stage_pread_ms,stage_upload_ms,stage_rate_mibs,ane_calls,ane_ok,ane_eval_calls,ane_refs_routed,ane_groups_routed,ane_chunk_refs,ane_pad_util,ane_wall_ms,ane_eval_ms,finish_join_ane_ms,dequant_wait_ms,output_convert_ms,same_layer_post_ms,cross_layer_post_ms,log,summary\n' > "$RESULTS"

echo "results: $RESULTS"
echo "prompts: ${PROMPT_LIST[*]}"
echo "chunks: ${CHUNK_LIST[*]}"
echo "min_refs: ${MIN_REF_LIST[*]}"
echo "slots: $DS4_SLOTS gpu_route: $GPU_ROUTE cooldown: ${COOLDOWN}s"
echo

for label in "${PROMPT_LIST[@]}"; do
  prompt_file="$(prompt_file_for_label "$label")"
  if [[ ! -f "$prompt_file" ]]; then
    echo "error: prompt file not found for '$label': $prompt_file" >&2
    exit 2
  fi
  ctx="$(ctx_for_prompt "$label")"
  for chunk in "${CHUNK_LIST[@]}"; do
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
      if [[ "$RUN_GPU" == "1" ]]; then
        run_profile gpu "$label" "$prompt_file" "$ctx" "$chunk" 0 "$repeat"
      fi
      if [[ "$RUN_ANE" == "1" ]]; then
        for min_refs in "${MIN_REF_LIST[@]}"; do
          run_profile ane "$label" "$prompt_file" "$ctx" "$chunk" "$min_refs" "$repeat"
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
      row[key] = $0
    }
  }
  END {
    for (key in row) print row[key]
  }
' "$RESULTS" | sort -t, -k2,2V -k4,4n
