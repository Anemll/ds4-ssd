#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/moe-batch-bench}"
MODEL="${MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
SIDECAR="${SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
PROMPT="${PROMPT:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt}"
CTX="${CTX:-32768}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"

HIST="$OUT_DIR/amx_feas_8k_hist.csv"
LOG="$OUT_DIR/amx_feas_8k_prefill.log"
SIM="$OUT_DIR/amx_feas_8k_scheduler.csv"
SUMMARY="$OUT_DIR/amx_feas_8k_summary.txt"

mkdir -p "$OUT_DIR"
rm -f "$HIST" "$LOG" "$SIM" "$SUMMARY"

echo "running pinned 8K prefill baseline"
DS4_LOCK_FILE="$LOCK_FILE" \
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_FLASH_MOE_PREFETCH=3 \
DS4_FLASH_MOE_HIST_CSV="$HIST" \
"$ROOT/ds4" \
  -m "$MODEL" \
  --moe-sidecar "$SIDECAR" \
  --moe-mode slot-bank \
  --metal \
  --ctx "$CTX" \
  --tokens 1 \
  --temp 0 \
  --prompt-file "$PROMPT" \
  > "$LOG" 2>&1

python3 "$ROOT/moe-batch-bench/simulate_prefill_scheduler.py" \
  "$HIST" \
  --gpu "$OUT_DIR/ds4_quant_large_scheduler.csv" "$OUT_DIR/ds4_quant_scheduler.csv" \
  --amx "$OUT_DIR/amxq_cached_scheduler.csv" \
  --dequant-ms 0.96 \
  > "$SIM"

{
  echo "Pinned 8K prefill baseline:"
  grep -E "prefill:|Flash-MoE prefill dedup" "$LOG" || true
  echo
  echo "Best AMX feasibility rows from histogram simulation:"
  python3 - "$SIM" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        row["speedup"] = float(row["speedup"])
        row["hybrid_tps_est"] = float(row["hybrid_tps_est"])
        row["baseline_tps_est"] = float(row["baseline_tps_est"])
        row["cache_gib"] = float(row["cache_gib"])
        rows.append(row)
rows.sort(key=lambda r: r["speedup"], reverse=True)
for row in rows[:12]:
    print(
        f"{row['cache']} {row['amx_range']}: "
        f"speedup={row['speedup']:.3f} "
        f"baseline_est={row['baseline_tps_est']:.2f}t/s "
        f"hybrid_est={row['hybrid_tps_est']:.2f}t/s "
        f"cache={row['cache_gib']:.2f}GiB "
        f"fills={row['fill_count']} "
        f"amx_refs={row['amx_refs']} gpu_refs={row['gpu_refs']}"
    )
PY
  echo
  echo "Files:"
  echo "  log: $LOG"
  echo "  hist: $HIST"
  echo "  sim: $SIM"
} | tee "$SUMMARY"
