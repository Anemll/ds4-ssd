#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/moe-batch-bench}"
MODEL="${MODEL:-/Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf}"
SIDECAR="${SIDECAR:-/Volumes/optane/dsv4-iq2xxs-expert-major}"
PROMPT="${PROMPT:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt}"
CTX="${CTX:-32768}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-codex.lock}"
ANE_BATCH="${ANE_BATCH:-128}"
ANE_ITERS="${ANE_ITERS:-2200}"

BASE_LOG="$OUT_DIR/real_prefill_ane_base.log"
CONC_LOG="$OUT_DIR/real_prefill_ane_concurrent.log"
ANE_LOG="$OUT_DIR/real_prefill_ane_worker.log"
SUMMARY="$OUT_DIR/real_prefill_ane_summary.txt"

mkdir -p "$OUT_DIR"
rm -f "$BASE_LOG" "$CONC_LOG" "$ANE_LOG" "$SUMMARY"

run_prefill() {
  local log="$1"
  DS4_LOCK_FILE="$LOCK_FILE" \
  DS4_METAL_PREFILL_CHUNK=16384 \
  DS4_FLASH_MOE_PREFETCH=3 \
  "$ROOT/ds4" \
    -m "$MODEL" \
    --moe-sidecar "$SIDECAR" \
    --moe-mode slot-bank \
    --metal \
    --ctx "$CTX" \
    --tokens 1 \
    --temp 0 \
    --prompt-file "$PROMPT" \
    > "$log" 2>&1
}

run_ane_worker() {
  "$ROOT/moe-batch-bench/ane_ds4_mlp_inmem_bench_packed_split3" \
    -bench-shape 7168 18432 3 "$ANE_BATCH" 5 "$ANE_ITERS" \
    > "$ANE_LOG" 2>&1
}

echo "running real pinned prefill baseline"
t0=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
run_prefill "$BASE_LOG"
t1=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)

echo "running real pinned prefill with concurrent split-3 ANE worker"
t2=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
run_ane_worker &
ane_pid=$!
run_prefill "$CONC_LOG"
prefill_rc=$?
wait "$ane_pid"
t3=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
if [ "$prefill_rc" -ne 0 ]; then
  exit "$prefill_rc"
fi

python3 - "$BASE_LOG" "$CONC_LOG" "$ANE_LOG" "$t0" "$t1" "$t2" "$t3" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path

base_log, conc_log, ane_log = map(Path, sys.argv[1:4])
t0, t1, t2, t3 = map(float, sys.argv[4:8])
summary = Path(sys.argv[8])

def parse_prefill(path):
    text = path.read_text(errors="replace")
    tps = None
    dedup = None
    m = re.search(r"prefill:\s+([0-9.]+) t/s", text)
    if m:
        tps = float(m.group(1))
    m = re.search(r"Flash-MoE prefill dedup refs=([0-9]+) unique=([0-9]+).*reuse=([0-9.]+)x", text)
    if m:
        dedup = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    return tps, dedup

def parse_ane(path):
    text = path.read_text(errors="replace")
    m = re.search(r"eval \(B=.*?\):\s+([0-9.]+) ms/iter\s+\|\s+([0-9.]+) TFLOP/s", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

base_tps, base_dedup = parse_prefill(base_log)
conc_tps, conc_dedup = parse_prefill(conc_log)
ane = parse_ane(ane_log)
base_wall = t1 - t0
conc_wall = t3 - t2
ratio = conc_tps / base_tps if base_tps and conc_tps else 0.0

lines = [
    "Real prefill + ANE concurrency test",
    f"baseline_wall_s={base_wall:.3f}",
    f"concurrent_wall_s={conc_wall:.3f}",
    f"baseline_prefill_tps={base_tps}",
    f"concurrent_prefill_tps={conc_tps}",
    f"concurrent_vs_baseline_tps={ratio:.3f}",
    f"baseline_dedup={base_dedup}",
    f"concurrent_dedup={conc_dedup}",
    f"ane_eval={ane}",
    f"base_log={base_log}",
    f"concurrent_log={conc_log}",
    f"ane_log={ane_log}",
]
summary.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
