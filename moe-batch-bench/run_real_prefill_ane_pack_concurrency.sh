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
PACK_ITERS="${PACK_ITERS:-2200}"
PACK_PACE_US="${PACK_PACE_US:-18000}"

BASE_LOG="$OUT_DIR/real_prefill_ane_pack_base.log"
CONC_LOG="$OUT_DIR/real_prefill_ane_pack_concurrent.log"
ANE_LOG="$OUT_DIR/real_prefill_ane_pack_ane_worker.log"
PACK_LOG="$OUT_DIR/real_prefill_ane_pack_gpu_worker.log"
SUMMARY="$OUT_DIR/real_prefill_ane_pack_summary.txt"

mkdir -p "$OUT_DIR"
rm -f "$BASE_LOG" "$CONC_LOG" "$ANE_LOG" "$PACK_LOG" "$SUMMARY"

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

run_pack_worker() {
  DS4_ANE_PACK_PACE_US="$PACK_PACE_US" \
  "$ROOT/moe-batch-bench/moe-batch-bench" \
    --backend anepackgpu \
    --in 7168 \
    --mid 18432 \
    --batches "$ANE_BATCH" \
    --warmup 5 \
    --iters "$PACK_ITERS" \
    > "$PACK_LOG" 2>&1
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

echo "running real pinned prefill with concurrent split-3 ANE eval and GPU split-pack workers"
t2=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
run_ane_worker &
ane_pid=$!
run_pack_worker &
pack_pid=$!
run_prefill "$CONC_LOG"
prefill_rc=$?
wait "$ane_pid"
wait "$pack_pid"
t3=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
if [ "$prefill_rc" -ne 0 ]; then
  exit "$prefill_rc"
fi

python3 - "$BASE_LOG" "$CONC_LOG" "$ANE_LOG" "$PACK_LOG" "$t0" "$t1" "$t2" "$t3" "$PACK_PACE_US" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path

base_log, conc_log, ane_log, pack_log = map(Path, sys.argv[1:5])
t0, t1, t2, t3 = map(float, sys.argv[5:9])
pack_pace_us = int(sys.argv[9])
summary = Path(sys.argv[10])

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

def parse_pack(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ane_split3_pack_gpu,"):
            parts = line.split(",")
            rows.append((float(parts[6]), float(parts[7])))
    return rows[-1] if rows else None

base_tps, base_dedup = parse_prefill(base_log)
conc_tps, conc_dedup = parse_prefill(conc_log)
ane = parse_ane(ane_log)
pack = parse_pack(pack_log)
base_wall = t1 - t0
conc_wall = t3 - t2
ratio = conc_tps / base_tps if base_tps and conc_tps else 0.0

lines = [
    "Real prefill + ANE eval + GPU split-pack concurrency test",
    f"baseline_wall_s={base_wall:.3f}",
    f"concurrent_wall_s={conc_wall:.3f}",
    f"baseline_prefill_tps={base_tps}",
    f"concurrent_prefill_tps={conc_tps}",
    f"concurrent_vs_baseline_tps={ratio:.3f}",
    f"baseline_dedup={base_dedup}",
    f"concurrent_dedup={conc_dedup}",
    f"ane_eval_ms_tflops={ane}",
    f"gpu_pack_ms_gbps={pack}",
    f"pack_pace_us={pack_pace_us}",
    f"base_log={base_log}",
    f"concurrent_log={conc_log}",
    f"ane_log={ane_log}",
    f"pack_log={pack_log}",
]
summary.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
