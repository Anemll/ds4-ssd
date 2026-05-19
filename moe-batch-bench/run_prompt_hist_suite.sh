#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/moe-batch-bench}"
MODEL="${MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
SIDECAR="${SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
PROMPT_DIR="${PROMPT_DIR:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding}"
CTX="${CTX:-32768}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"

mkdir -p "$OUT_DIR"

for size in 1k 4k 8k 16k; do
  hist="$OUT_DIR/prompt_hist_${size}.csv"
  log="$OUT_DIR/prompt_run_${size}.log"
  prompt="$PROMPT_DIR/coding_${size}.txt"
  rm -f "$hist" "$log"
  echo "running $size prompt"
  DS4_LOCK_FILE="$LOCK_FILE" \
  DS4_FLASH_MOE_HIST_CSV="$hist" \
  "$ROOT/ds4" \
    -m "$MODEL" \
    --moe-sidecar "$SIDECAR" \
    --moe-mode slot-bank \
    --metal \
    --ctx "$CTX" \
    --tokens 1 \
    --temp 0 \
    --prompt-file "$prompt" \
    > "$log" 2>&1
  tail -20 "$log"
  wc -l "$hist"
done

python3 "$ROOT/moe-batch-bench/analyze_prompt_hist.py" \
  "$OUT_DIR"/prompt_hist_{1k,4k,8k,16k}.csv \
  --qhybrid \
  "$OUT_DIR/qhybrid_amx16_seq.csv" \
  "$OUT_DIR/qhybrid_amx32_seq.csv"
