# ANE int8 Combined Benchmark Procedure

Goal: measure whether ANE can add useful prefill throughput alongside the
existing quantized Metal DeDup GPU path, using streamed int8 weights/activations
and batched GPU dequant.

## Build

Run from the repo root:

```sh
make moe-batch-bench
make moe-batch-bench/ane_int8_w_input_probe
```

## 1. GPU Expert Baseline

Measure the current quantized Metal expert backend:

```sh
./moe-batch-bench/moe-batch-bench \
  --backend ds4 \
  --batches 8192,16384,32768 \
  --warmup 5 \
  --iters 100
```

This is the GPU-only DeDup expert MLP baseline.

## 2. Per-Tensor GPU Dequant

Compare BF16 materialization against int8 materialization:

```sh
./moe-batch-bench/moe-batch-bench \
  --backend dequantparts \
  --in 4096 --mid 2048 \
  --warmup 10 \
  --iters 100
```

m3u reference:

```text
BF16 full expert: ~1.35 ms
int8 full expert: ~1.20 ms
```

## 3. Batched Multi-Expert Dequant

Measure whether batching multiple experts reduces GPU dequant overhead:

```sh
./moe-batch-bench/moe-batch-bench \
  --backend dequantbatch \
  --in 4096 --mid 2048 \
  --batches 1,2,4,8,16 \
  --warmup 5 \
  --iters 50
```

Production-relevant fused-only mode:

```sh
DS4_DEQUANT_BATCH_FUSED_ONLY=1 \
./moe-batch-bench/moe-batch-bench \
  --backend dequantbatch \
  --in 4096 --mid 2048 \
  --batches 16 \
  --warmup 5 \
  --iters 50
```

m3u reference:

```text
single-ish expert: ~0.75 ms/expert
16 experts batched: ~0.45 ms/expert
```

## 4. ANE Streamed int8 MLP

Runtime inputs are int8 weights plus int8 activation. The MIL graph dequantizes
inside ANE, then runs matmuls.

```sh
./moe-batch-bench/ane_int8_w_input_probe \
  --mlp \
  --shape 4096 2048 128 \
  --warmup 5 \
  --iters 1000
```

m3u reference:

```text
B128: ~0.7-0.85 ms/eval
```

Note: this is streamed int8 input, but on m3u it is likely still fp16 matmul
after runtime dequant. On M5 Max, check whether this path maps to faster int8
compute.

## 5. GPU + ANE Coexistence

Run the existing GPU backend and ANE worker concurrently:

```sh
python3 - <<'PY'
import subprocess, time
gpu = [
    "./moe-batch-bench/moe-batch-bench",
    "--backend", "ds4",
    "--batches", "16384",
    "--warmup", "5",
    "--iters", "100",
]
ane = [
    "./moe-batch-bench/ane_int8_w_input_probe",
    "--mlp",
    "--shape", "4096", "2048", "128",
    "--warmup", "5",
    "--iters", "1600",
]
t = time.perf_counter()
p1 = subprocess.Popen(gpu)
p2 = subprocess.Popen(ane)
p1.wait()
p2.wait()
print("wall", time.perf_counter() - t)
PY
```

m3u result: the int8 ANE worker did not materially slow the GPU backend.

## 6. Full Synthetic Combined Path

Run three workers concurrently:

```text
1. GPU ds4 expert backend
2. ANE int8 streamed MLP
3. GPU batched IQ2_XXS/Q2_K -> int8 dequant for ANE weights
```

Compare against a GPU-only batch with the same total expert rows. On m3u, the
fused-only combined path produced small synthetic wins:

```text
GPU B8192:  1.011x
GPU B16384: 1.009x
GPU B32768: 1.036x
```

Interpretation: barely positive on m3u. M5 Max may improve if ANE streamed int8
uses faster int8 compute and the GPU remains fast enough.

## 7. Real Prefill Baseline

Always pin real prefill settings:

```sh
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_FLASH_MOE_PREFETCH=3 \
./ds4 \
  -m /Volumes/optane/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /Volumes/optane/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --metal \
  --ctx 32768 \
  --tokens 1 \
  --temp 0 \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
```

Use real prefill only after the synthetic combined benchmark looks positive.

## M5 Max Checklist

On M5 Max, answer these questions:

```text
1. Does ane_int8_w_input_probe get much faster than m3u?
2. Does dequantbatch fused stay below ~0.45 ms/expert?
3. Does combined synthetic speedup exceed ~1.10x?
4. Does real pinned 8K prefill stay flat or improve when ANE work is active?
```

If M5 Max reaches true int8 ANE throughput for the streamed-input path, the
combined path should show a stronger win than m3u.
