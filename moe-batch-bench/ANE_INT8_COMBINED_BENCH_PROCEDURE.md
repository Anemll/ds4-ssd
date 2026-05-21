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

M5 Max result, `H=4096 I=2048`, 2026-05-18. `GPU no quant` is the dense fp16
MPS proxy with resident weights. `GPU quant` is the real DS4 quantized Metal
expert backend. `ANE no producer` assumes int8 weights/activation are already
materialized. `ANE + producer` charges one fused GPU dequant/materialization
producer at ~0.56 ms/expert.

| B   | GPU no quant TFLOP/s | GPU quant TFLOP/s | ANE no producer ms | ANE no producer TFLOP/s | ANE + producer ms | ANE + producer TFLOP/s |
|----:|---------------------:|------------------:|-------------------:|------------------------:|------------------:|-----------------------:|
| 1   | 0.16                 | 0.22              | 0.388              | 0.13                    | 0.949             | 0.05                   |
| 8   | 1.39                 | 1.21              | 0.388              | 1.04                    | 0.949             | 0.42                   |
| 16  | 2.30                 | 1.90              | 0.390              | 2.06                    | 0.951             | 0.85                   |
| 32  | 3.99                 | 3.46              | 0.403              | 4.00                    | 0.964             | 1.67                   |
| 64  | 8.19                 | 5.26              | 0.411              | 7.83                    | 0.972             | 3.31                   |
| 96  | 9.55                 | 7.11              | 0.428              | 11.30                   | 0.989             | 4.89                   |
| 128 | 14.53                | 7.92              | 0.442              | 14.56                   | 1.003             | 6.42                   |
| 192 | 19.21                | 9.64              | 0.733              | 13.19                   | 1.294             | 7.47                   |
| 256 | 26.01                | 9.99              | 0.755              | 17.07                   | 1.316             | 9.79                   |
| 384 | 29.66                | 10.86             | 1.122              | 17.22                   | 1.683             | 11.48                  |
| 512 | 36.36                | 11.28             | 1.389              | 18.55                   | 1.950             | 13.21                  |

Interpretation: M5 Max gets a useful raw ANE curve, but the producer cost is
large enough that small ANE batches are not useful once quantized-bank
materialization is charged. Against the real quantized DS4 GPU path, B384-B512
are the first points where `ANE + producer` exceeds GPU DS4 throughput on a
per-expert TFLOP/s basis. Against the dense no-quant GPU proxy, ANE remains
below the GPU ceiling.

### M5 Max GPU dequant accounting

The ANE eval numbers above exclude the GPU producer that turns quantized
`IQ2_XXS`/`Q2_K` expert banks into streamed int8 gate/up/down inputs. On M5 Max,
single-tensor Metal dequant is:

| tensor | source quant | int8 output MB | Metal dequant ms |
|--------|--------------|---------------:|-----------------:|
| W_gate | IQ2_XXS      | 8.39           | 0.314            |
| W_up   | IQ2_XXS      | 8.39           | 0.320            |
| W_down | Q2_K         | 8.39           | 0.282            |

Separate kernels therefore cost about **0.916 ms/expert**. The fused batched
producer is the production-relevant estimate:

| experts/batch | total ms | ms/expert | output GB/s |
|--------------:|---------:|----------:|------------:|
| 1             | 0.783    | 0.783     | 32.15       |
| 2             | 1.267    | 0.633     | 39.74       |
| 4             | 2.358    | 0.589     | 42.70       |
| 8             | 4.566    | 0.571     | 44.09       |
| 16            | 8.977    | 0.561     | 44.85       |
| 32            | 17.782   | 0.556     | 45.29       |
| 64            | 35.255   | 0.551     | 45.69       |

For rough scheduling, use the fused producer's **~0.56 ms per ANE expert**
number from the table above. Whether that becomes an end-to-end prefill win
depends on keeping the GPU dequant producer overlapped with the main GPU expert
path without stealing too much bandwidth.

### MPP GPU int8 option

Apple's newer **Metal Performance Primitives (MPP)** tensor-op matmul is a
separate path from the older `MPSMatrixMultiplication`/MPSGraph proxy used in
the table above. The local macOS SDK header
`MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h` lists
the supported matmul type combinations explicitly. Relevant combinations:

```text
Left    Right   Destination
half    int8    half
half    int8    float
int8    half    half
int8    half    float
float   int8    float
int8    float   float
int8    int8    int32
uint8   uint8   int32
half    int4    half/float
int8    int4    int32
```

So yes, M5-class GPU kernels can multiply fp16 activations by int8 weights via
MPP, and can also do true W8A8 `int8 x int8 -> int32`. MPP matmul does not
understand DS4 quantization scales by itself, though: the operands are raw
numeric tensor elements. A production MoE-DeDup MPP path therefore needs one of
these accounting models:

1. **fp16 activation x int8 weight -> fp16/float output**: avoid activation
   quantization, but apply the expert/block scale either before or after the
   matmul. This is probably the easiest first prototype.
2. **int8 activation x int8 weight -> int32 output**: true integer GEMM, but
   requires activation quantization and a post-scale path before SiLU/mul/down.
3. **int8 activation x int4 weight -> int32 output**: attractive for bandwidth,
   but needs an int4 packing/layout experiment and the same post-scale handling.

The next useful benchmark is a synthetic MPP backend for the same
`H=4096 I=2048` MLP shape, starting with `half x int8 -> half/float`. If that
beats the current DS4 quantized Metal backend after scale handling is charged,
then the MoE-DeDup replacement point is the same per-expert call site used by
the existing GPU expert block.

Probe build/run:

```sh
make mpp-int8-bench
./moe-batch-bench/mpp_int8_matmul_probe \
  --shape 128 4096 2048 \
  --warmup 5 \
  --iters 200
```

The probe measures one raw MPP matmul, not the full fused MLP. Shape is
`M=B, K=4096, N=2048`, matching the gate/up projection orientation:

| B   | half x int8 -> half | half x int8 -> float | int8 x half -> half | int8 x half -> float | int8 x int8 -> int32 |
|----:|--------------------:|---------------------:|--------------------:|---------------------:|---------------------:|
| 1   | 0.14 TF/s           | 0.25 TF/s            | 0.25 TF/s           | 0.25 TF/s            | 0.40 TF/s            |
| 8   | 2.27 TF/s           | 2.29 TF/s            | 2.02 TF/s           | 2.07 TF/s            | 3.52 TF/s            |
| 16  | 4.60 TF/s           | 4.44 TF/s            | 4.25 TF/s           | 4.06 TF/s            | 6.57 TF/s            |
| 32  | 9.16 TF/s           | 8.95 TF/s            | 8.21 TF/s           | 8.38 TF/s            | 11.65 TF/s           |
| 64  | 17.77 TF/s          | 17.92 TF/s           | 16.35 TF/s          | 16.46 TF/s           | 23.15 TF/s           |
| 96  | 18.40 TF/s          | 18.04 TF/s           | 17.66 TF/s          | 17.99 TF/s           | 33.13 TF/s           |
| 128 | 24.18 TF/s          | 23.59 TF/s           | 23.93 TF/s          | 23.81 TF/s           | 44.22 TF/s           |
| 192 | 25.21 TF/s          | 24.84 TF/s           | 24.62 TF/s          | 24.90 TF/s           | 49.60 TF/s           |
| 256 | 26.17 TF/s          | 26.02 TF/s           | 26.51 TF/s          | 26.26 TF/s           | 45.33 TF/s           |
| 384 | 27.21 TF/s          | 26.97 TF/s           | 27.12 TF/s          | 27.16 TF/s           | 46.56 TF/s           |
| 512 | 28.94 TF/s          | 29.06 TF/s           | 29.56 TF/s          | 29.28 TF/s           | 48.04 TF/s           |

Down-projection orientation (`M=B, K=2048, N=4096`) spot checks:

| B   | half x int8 -> float | int8 x half -> float | int8 x int8 -> int32 |
|----:|---------------------:|---------------------:|---------------------:|
| 128 | 24.98 TF/s           | 24.29 TF/s           | 43.43 TF/s           |
| 256 | 29.04 TF/s           | 28.62 TF/s           | 50.48 TF/s           |
| 512 | 28.72 TF/s           | 29.26 TF/s           | 50.44 TF/s           |

Takeaway: MPP mixed `half/int8` matmul is already well above the current DS4
quantized backend's dense-equivalent throughput once batches are large enough,
and true W8A8 integer matmul is roughly 45-50 TFLOP/s on this M5 Max. The open
question is not raw GEMM speed; it is the end-to-end cost of scale handling,
activation quantization for W8A8, SiLU/mul fusion, and conversion back into the
existing scatter-add contract.

First full-MLP synthetic MPP path, still with fp/half activations:

```text
gate:  half activation x int8 weight -> float
up:    half activation x int8 weight -> float
mid:   SiLU(gate) * up in float
down:  float mid x int8 weight -> float
```

This matches the first production-adjacent DeDup plan: keep router/top-k/dedup
in the existing FP32 path, gather compacted expert rows as FP32, then cast only
that compacted per-expert batch to half at the actual DeDup execution point.
It does **not** quantize activations to int8 yet.

| B   | full MLP ms | full MLP TFLOP/s |
|----:|------------:|-----------------:|
| 32  | 0.405       | 3.97             |
| 64  | 0.257       | 12.54            |
| 96  | 0.362       | 13.35            |
| 128 | 0.361       | 17.83            |
| 192 | 0.510       | 18.93            |
| 256 | 0.669       | 19.27            |
| 384 | 0.981       | 19.70            |
| 512 | 1.271       | 20.27            |

This path avoids the W8A8 activation quantization cost and is therefore the
cleaner first correctness experiment. The remaining production cost to charge
is DS4 quantized-bank -> int8 MPP weight materialization plus any scale
correction needed to make the int8 weights numerically match the existing
expert output closely enough.

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

## 8. M5 Max MPP Int8 Prefill Smoke

Experimental GPU MPP int8 prefill is gated by:

```sh
DS4_FLASH_MOE_MPP_INT8_PREFILL=1
DS4_FLASH_MOE_MPP_INT8_QSCALE=64
```

Smoke command used on the M5 Max SSD copy:

```sh
DS4_LOCK_FILE=/tmp/ds4-codex.lock \
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_FLASH_MOE_PREFETCH=3 \
DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
DS4_FLASH_MOE_MPP_INT8_QSCALE=64 \
./ds4 \
  -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --metal \
  --ctx 32768 \
  --tokens 100 \
  --temp 0 \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
```

Real 8K prefill/generation comparison:

| Mode | Tokens | Prefill t/s | Generation t/s | Dedup unique | Reuse | Output |
|---|---:|---:|---:|---:|---:|---|
| Baseline GPU | 100 | 200.83 | 11.43 | 17,981 | 120.86x | Coherent plan text |
| MPP int8 prefill | 1 | 190.31 | 2.41 | 15,499 | 140.21x | Smoke pass |
| MPP int8 prefill | 100 | 170.32 | 14.10 | 15,499 | 140.21x | Diverges immediately; repetitive pattern |

The default qscale is experimental. A quick real-prompt qscale sweep with
16 generated tokens showed that the earlier qscale=8 result was a bad numeric
point, not a failure of MPP half/int8 arithmetic:

| Qscale | Tokens | Prefill t/s | Generation t/s | Dedup unique | Reuse | Output preview |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 193.17 | 7.10 | 15,426 | 140.87x | Readable but drifted |
| 2 | 16 | 188.95 | 8.04 | 15,430 | 140.84x | Odd punctuation |
| 4 | 16 | 188.09 | 7.75 | 15,416 | 140.97x | Odd punctuation |
| 8 | 16 | 188.31 | 8.54 | 15,499 | 140.21x | Repetitive pattern |
| 16 | 16 | 177.80 | 7.58 | 17,945 | 121.10x | Coherent |
| 32 | 16 | 171.65 | 7.66 | 18,777 | 115.73x | Coherent |

100-token follow-up:

| Qscale | Tokens | Prefill t/s | Generation t/s | Dedup unique | Reuse | Output |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 100 | 178.23 | 9.76 | 17,945 | 121.10x | Coherent DFlash plan; not exact baseline wording |
| 32 | 100 | 189.92 | 11.20 | 18,777 | 115.73x | Coherent DFlash plan |
| 64 | 100 | 189.15 | 10.97 | 18,893 | 115.02x | Coherent DFlash plan |
| 128 | 100 | 176.85 | 10.78 | 18,839 | 115.35x | Coherent DFlash plan, more drift |
| 256 | 100 | 188.26 | 11.42 | 18,930 | 114.80x | Coherent DFlash plan |

Status: the guarded path runs full prefill and generation. Correctness is now
scale-sensitive rather than simply broken. qscale=32 through qscale=256 are all
coherent in this prompt; qscale=64 is a good current default candidate because
it stays coherent while avoiding the lower-scale collapse. qscale=8 should not
be used for correctness testing.

## 9. M5 Max MPP Int8 x Int8 Prefill

Full int8 x int8 prefill is a separate experimental mode layered under the MPP
int8 gate:

```sh
DS4_FLASH_MOE_MPP_INT8_PREFILL=1
DS4_FLASH_MOE_MPP_I8I8_PREFILL=1
DS4_FLASH_MOE_MPP_INT8_QSCALE=64
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=16
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=2
```

This mode quantizes gathered FP32 activations to int8, runs gate/up as
int8 x int8 -> int32, rescales into the SwiGLU, quantizes the routed SwiGLU
mid activation to int8, runs down as int8 x int8 -> int32, then rescales back
to FP32 for the existing scatter-add.

Current test command:

```sh
DS4_LOCK_FILE=/tmp/ds4-codex.lock \
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_FLASH_MOE_PREFETCH=3 \
DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 \
DS4_FLASH_MOE_MPP_INT8_QSCALE=64 \
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=16 \
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=2 \
./ds4 \
  -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --metal \
  --ctx 32768 \
  --tokens 100 \
  --temp 0 \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt
```

Mid-scale sweep at weight qscale=64 and x qscale=16:

| Weight qscale | X qscale | Mid qscale | Tokens | Prefill t/s | Generation t/s | Dedup unique | Reuse | Output preview |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 16 | 0.25 | 16 | 186.88 | 7.34 | 16,552 | 131.29x | Coherent but drifted |
| 64 | 16 | 0.5 | 16 | 186.92 | 7.64 | 17,333 | 125.38x | Coherent |
| 64 | 16 | 1 | 16 | 185.85 | 7.68 | 17,962 | 120.99x | Readable, odd drift |
| 64 | 16 | 2 | 16 | 184.68 | 7.69 | 18,357 | 118.38x | Coherent |
| 64 | 16 | 4 | 16 | 185.55 | 7.87 | 18,330 | 118.56x | Coherent |
| 64 | 16 | 2 | 100 | 184.64 | 10.11 | 18,357 | 118.38x | Coherent DFlash plan; not exact baseline |

Status: the full int8 x int8 path is wired and can run real 8K prefill plus
generation. It is useful for scale search, but not yet faster than the current
baseline and still diverges from exact baseline wording.

### Experimental fused int8 x int8 mid path

The int8 x int8 path also has a separately gated fused mid step:

```sh
DS4_FLASH_MOE_MPP_INT8_PREFILL=1
DS4_FLASH_MOE_MPP_I8I8_PREFILL=1
DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1
DS4_FLASH_MOE_MPP_INT8_QSCALE=64
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=16
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=2
```

This fuses the post-MPP gate/up stage:

```text
gate_i32 + up_i32 -> weighted SwiGLU -> mid_i8
```

The MPP gate/up matmuls still materialize `int32` intermediates, and the down
projection remains a separate MPP `int8 x int8 -> int32` matmul followed by an
`int32 -> f32` scale kernel. This is the largest safe fused step in the current
MPP implementation because the MPP matmul epilogue is not customized here.

Tiny compile/runtime smoke used during bring-up:

```sh
DS4_LOCK_FILE=/tmp/ds4-codex.lock \
DS4_METAL_PREFILL_CHUNK=256 \
DS4_FLASH_MOE_PREFETCH=1 \
DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
DS4_FLASH_MOE_MPP_I8I8_PREFILL=1 \
DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=1 \
DS4_FLASH_MOE_MPP_INT8_QSCALE=64 \
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=16 \
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=2 \
./ds4 \
  -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --metal \
  --ctx 1024 \
  --tokens 1 \
  --temp 0 \
  -p "hello"
```

Smoke result: build passed with `make ds4`; tiny runtime completed with
`prefill: 4.20 t/s`, `generation: 2.38 t/s`, `unique=1528`, `reuse=1.69x`.

---

## 10. ANE-only async pipeline (2026-05-19/20)

**Critical env contract:** `DS4_FLASH_MOE_ANE_I8I8_PREFILL=1` MUST be set
explicitly when running ANE-only (`DS4_FLASH_MOE_MPP_I8I8_PREFILL=0`).
Without it, `ds4_gpu_ane_prefill_i8i8_enabled()` falls back to reading
`DS4_FLASH_MOE_MPP_I8I8_PREFILL` and the ANE path silently returns NULL
from `ane_start_tensor`. Work then falls through to the fp32 GPU MoE
legacy kernel — output is correct but ANE silicon stays idle and the
throughput numbers are misleading (they look reasonable because GPU fp32
is fast). Diagnose with `DS4_FLASH_MOE_TRACE_DISPATCH=1` which logs
per-group dispatch decision + `start_tensor` return for the first 8
groups in layers 0–1.

Reference env that engages the async pipeline + makes the scheduler try
to favor ANE (`DS4_FLASH_MOE_SCHED_ANE_REL_SPEED` accepts up to 1024
after the 2026-05-20 clamp lift):

```bash
DS4_FLASH_MOE_ANE_PREFILL=1 \
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 \
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1 \
DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1 \
DS4_FLASH_MOE_OVERLAP_PREFILL=1 \
DS4_FLASH_MOE_OVERLAP_SCHEDULER=1 \
DS4_FLASH_MOE_ANE_BATCHES=64,128,256,512 \
DS4_FLASH_MOE_ANE_MAX_REFS=512 \
DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS=1 \
DS4_FLASH_MOE_ANE_MIN_REFS=32 \
DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
DS4_FLASH_MOE_MPP_I8I8_PREFILL=0 \
DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0 \
DS4_FLASH_MOE_SCHED_ANE_REL_SPEED=99 \
DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL=0.0 \
./ds4 -m … --prompt-file …/coding_8k.txt --tokens 1 --temp 0
```

Reference results on M5 Max, coding_8k.txt (8423 tokens), one full prefill
chunk (`DS4_METAL_GRAPH_RAW_CAP=8704`) and ANE-only MLP dispatch
(`gpu_i8_groups=0`, `fp32_groups=0` on every layer):

| Config | Prefill t/s | Notes |
|---|---:|---|
| ANE-only, no async pread | 204.41 | Baseline with `DS4_FLASH_MOE_ASYNC_PREAD=0`, `PREFETCH=3`. |
| ANE-only, async pread tuned | **214.88** | Best M5 Max result so far: `ASYNC_PREAD=1`, `PREFETCH=1`, `ASYNC_PREAD_AFTER_STAGE=1`, `ANE_OUTPUT_QUEUE=1`. |
| Async + `ANE_OUTPUT_QUEUE=2` | 207.42 | Worse; queueing completed ANE jobs increases tail/convert pressure. |
| Async + GPU output pack | 186.73 | Removes CPU output convert time, but shared-buffer/copy/Metal scheduling overhead dominates. Do not enable by default. |
| Async + scalar output pack | 192.97 | Diagnostic only; proves NEON vector output conversion matters. |
| Async + `PREFETCH=0` | 193.67 | Too little read speculation; long cross-layer issue gaps return. |

Best current M5 Max ANE-only profile env:

```bash
DS4_FLASH_MOE_ASYNC_PREAD=1
DS4_FLASH_MOE_PREFETCH=1
DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE=1
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=1
```

Do **not** enable these by default on M5 Max:

```bash
DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=1      # slower end-to-end despite cheap GPU encode
DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=1   # diagnostic override; disables NEON conversion
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=2          # slower than queue depth 1 in this sweep
```

The async pread win is scheduling rather than raw read bandwidth. On M5 Max,
baseline pread time was 8.95 s and tuned async pread was 9.43 s, but pread span
fell from 39.10 s to 37.22 s and same-layer post-stage gap fell from 11.84 s
to 0.08 s. That produced a net +5.1% throughput improvement.

The remaining limiter is not raw output conversion alone. NEON vector output
packing reduces `output_pack` from 7.24 s (scalar diagnostic) to about 2.44 s,
but the tuned run still shows a large `post_wait`/pipeline tail. GPU output
packing reduced `output_pack` to 28.8 ms, but the full run dropped to 186.73
t/s because the extra shared-buffer and Metal-side costs lengthened the layer
pipeline.

For repeatable M5 Max profiling use:

```bash
./run_ane_prefill_profile_m5max.sh
```

Logs are written under `moe-batch-bench/profile_runs/`. Use
`DS4_RUN_NAME=...` to label a sweep and override individual env keys at the
command line.

**Interpretation.** ANE-only is now useful as a profiling mode and the async
sidecar read schedule is measurably better on M5 Max, but ANE still does not
beat the GPU-only MPP int8 path for this end-to-end prefill workload. See
Appendix C in `DSv4_MLP_ANE_Matmul_Investigation.md` for the broader
engine-selection conclusion and the multi-ctx pool negative result.

## 11. M5 Max GPU-only async pread profile (2026-05-21)

GPU-only MPP i8w-i8x fused prefill also benefits from the new async sidecar
pread path. The ANE output-pack flags are intentionally disabled for this mode:
they only affect ANE job output conversion and should not be part of a pure GPU
profile.

Reference results on M5 Max, coding_8k.txt (8423 tokens), one full prefill
chunk (`DS4_METAL_GRAPH_RAW_CAP=8704`), 32 slot-bank slots, GPU-only MLP
dispatch:

| Config | Prefill t/s | GPU layer-major total | Stage/pread notes |
|---|---:|---:|---|
| GPU-only, no async pread | 225.57 | 37.26 s | `ASYNC_PREAD=0`, `PREFETCH=3`; same-layer post-stage gap 11.18 s. |
| GPU-only, async pread tuned | **276.56** | 30.37 s | `ASYNC_PREAD=1`, `PREFETCH=1`, `ASYNC_PREAD_AFTER_STAGE=1`; same-layer post-stage gap 0.07 s. |
| GPU-only, async pread + `PREFETCH=3` | 275.50 | 30.48 s | Similar but slightly slower than prefetch 1 on this run. |
| GPU-only, async pread before-stage | 273.23 | 30.73 s | `ASYNC_PREAD_AFTER_STAGE=0`; slightly worse. |

Best current M5 Max GPU-only profile env:

```bash
DS4_FLASH_MOE_ASYNC_PREAD=1
DS4_FLASH_MOE_PREFETCH=1
DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE=1
DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=0
DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0
```

The gain is again scheduling rather than raw read bandwidth. Baseline pread was
6.96 s at about 10.19 GiB/s; tuned async pread was 6.68 s at about 10.63 GiB/s.
The important change is that read issue/post gaps are pulled out of the critical
same-layer path: total pread span fell from 36.64 s to 29.74 s and same-layer
post-stage gap fell from 11.18 s to 0.07 s. End-to-end prefill improved by
about 22.6%.

For repeatable M5 Max GPU-only profiling use:

```bash
./run_gpu_prefill_profile_m5max.sh
```

Override `DS4_SLOTS`, `DS4_PREFILL_CHUNK`, `DS4_PROMPT_FILE`,
`DS4_METAL_GRAPH_RAW_CAP`, or `DS4_RUN_NAME` in the environment when comparing
different machines or prompt sizes.
