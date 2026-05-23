# Attention output projection (O-proj) on ANE — investigation notes

## TL;DR

We moved DSv4's attention output projection (`attn_output_a × attn_output_b`,
a LoRA-style two-stage matmul, ~67 ms/layer GPU dispatch) onto ANE via two
paths, both off by default:

1. **Matmul path** — uses the existing fp16w (mode 1) or i8w-fp16x (mode 2)
   split ctx with a new linear two-matmul eval (no SiLU/mul activation).
2. **Constexpr conv path** — single-call conv2d-1×1 with weights baked into
   MIL via BLOBFILE, multi-cluster ANE worker support, per-channel int8
   quantization, GPU input conversion via per-chunk IOSurface array.

End state on M3 Ultra (8423-token prefill, 50 decode tokens):

| Config | Prefill t/s | Δ vs GPU |
|--------|-------------|----------|
| GPU baseline | **287.4** | — |
| ANE fp16 matmul | 184 | −103 |
| ANE int8 matmul (per-tensor scale) | 203 | −84 |
| ANE fp16 conv 2-cluster | 229 | −58 |
| ANE int8 conv 1-cluster | 218 | −69 |
| ANE int8 conv 2-cluster | 247 | −40 |
| ANE int8 conv 4-cluster | 254 | −33 |
| ANE int8 conv 4-cluster + NEON | 266 | −21 |
| **ANE int8 conv 2-cluster + per-thread NEON + GPU CVT** | **262** | **−25** |

GPU still wins on M3U. The remaining gap is in the per-call ANE eval
(189 ms/layer in production vs 82 ms predicted by the standalone smoke) —
likely SLC/bandwidth pressure from 86 in-flight constexpr ctxs.  The
optimizations DO compose: each one closes part of the gap, but the layer-
local dataflow (O-proj producer → consumer with no GPU work between) means
ANE-side wins don't fully translate to prefill wall time.

The same code may pay off on M5 family chips where ANE INT8 throughput
matches or exceeds GPU on this shape — see § M5 hand-off below.

---

## Env flag matrix

All flags require `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1` as the master switch.

| Flag | Effect |
|------|--------|
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1` | Master — enable O-proj on ANE (off = GPU baseline). |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1` | Use conv2d-1×1 + constexpr weights path (off = matmul + IOSurface inputs). |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_INT8=1` | Use int8 weights with per-channel fp16 scale (matmul: per-tensor; conv: per-channel via `constexpr_blockwise_shift_scale`). |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=1..4` | Number of parallel ANE workers per layer (conv only). 2 = dual cluster. |
| `DS4_FLASH_MOE_ANE_OPROJ_BATCH=N` | Compiled ANE batch size B (default 256). 33 chunks at B=256 for 8423-token prefill. |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_GPU_CVT=1` | Allocate 64 per-chunk fp16 IOSurfaces shared across all ctxs; GPU does input fp32→fp16 conversion via `kernel_cpy_f32_f16`; worker calls `eval_at_chunk(k)` so ANE reads chunk `k` from its own pre-written IOSurface (no NEON, no memcpy). Requires `_CONV=1 + _INT8=1`. |

---

## Two key architecture changes added this session

### 1. INT8 weights with per-channel scale (conv path, mode 11)

**What it does.** Replaces fp16 constexpr weights in the conv2d-1×1 MIL with
int8 weights + per-output-channel fp16 scales, baked at compile time via
`constexpr_blockwise_shift_scale`.  Halves weight storage in the compiled
program (4 MB int8 + ~16 KB scale per Wa/Wb vs 8 MB fp16) and meaningfully
reduces ANE-internal weight fetch bandwidth.

**Key files.**
- `moe-batch-bench/ane_ds4_mlp_int8w.m`:
  - `gen_mil_int8w_linear_constexpr_conv` — MIL generator using
    `constexpr_blockwise_shift_scale` with degenerate block-size = full
    in_dim (which ANE accepts as per-channel; ANE only supports per-tensor
    or per-channel scaling, per workflow doc).
  - `ane_build_blob_n` — generic N-chunk MIL weight blob builder.
  - `ds4_ane_mlp_int8w_linear_constexpr_create` / `_eval` — public API.
- `moe-batch-bench/ane_ds4_oproj_int8_smoke.m` — standalone harness with
  1/2/4 thread sweep.
- `ds4_metal.m`:
  - `ds4_quantize_f16_to_i8_per_channel` — host-side symmetric int8
    quantization with one fp16 scale per output channel.
  - `ds4_oproj_layer_cache.int8_conv_ctxs[4]` — per-layer, per-worker ctxs.
  - Worker dispatch picks int8 conv > fp16 conv > matmul.

**Standalone smoke result** (H=4096, I=8192, B=256):

| Threads | ms/iter | TFLOP/s | Scaling |
|---------|---------|---------|---------|
| 1 | 4.93 | 6.95 | 1.00× |
| 2 | 2.49 | **13.80** | 1.98× |
| 4 | 2.32 | 14.83 | 2.13× |

vs fp16 conv 2-thread: 3.91 ms/iter, 8.78 TFLOP/s.  Int8 path is **1.57×
faster** per call.  Dual-cluster scaling is nearly linear at 1.98×.

**Production result** (DSv4 IQ2_XXS, 8423-token prompt, 2-cluster):

| Path | Prefill | per-layer ANE eval |
|------|---------|---------------------|
| fp16 conv | 229 t/s | 300 ms |
| int8 conv | **247 t/s** | **192 ms** (−36%) |

Decode generation rate stays correct (3.7+ t/s, vs GPU baseline 3.8), so
per-channel int8 quantization preserves quality at this layer.

**ANE compute mode caveat.** `constexpr_blockwise_shift_scale` dequantizes
weights to fp16 **at compile time**, so the conv itself still runs as FP16
GEMM at runtime.  Memory pressure is the win, not compute throughput.  To
use ANE's native INT8 GEMM (which has higher peak TOPS on M5 and possibly
on M3U for some configs), the model would need int8 activations too (would
require a `quantize` op on X inside MIL + matching dequant on output).  Not
implemented; see § M5 hand-off.

### 2. Per-chunk IOSurface array + GPU input conversion (mode 11 + GPU_CVT)

**What it does.** Removes CPU input conversion entirely.  Pre-allocates a
pool of 64 per-chunk input IOSurfaces (size [B, in_dim] fp16), wraps each
as a Metal buffer via `newBufferWithBytesNoCopy(IOSurfaceGetBaseAddress)`.
Each int8 conv ctx gets 64 requests, each bound to one of the 64 IOSurfaces.
At eval time GPU encodes N `kernel_cpy_f32_f16` dispatches into the same CB
that produces `batch_heads`, each writing the chunk's fp32→fp16-converted
slice directly into the chunk's IOSurface.  The CB's completion handler
signals dep_done, and the ANE worker calls `eval_at_chunk(k)` — ANE reads
IOSurface k via the matching request.  No NEON conversion, no per-call
`write_surface` to ANE's input IOSurface, no per-job scratch buffer.

**Key files.**
- `moe-batch-bench/ane_ds4_mlp_int8w.m`:
  - `ds4_ane_mlp_int8w_linear_constexpr_attach_chunks` — bolts N input
    IOSurfaces onto an existing mode-11 ctx by creating one ANE request
    per IOSurface.
  - `ds4_ane_mlp_int8w_linear_constexpr_eval_at_chunk` — dispatches via the
    chunk-indexed request; no per-call IOSurface write.
- `moe-batch-bench/ane_per_chunk_iosurface_probe.m` — probe that validates
  ANE accepts N=33 requests on one compiled model with distinct input
  IOSurfaces.  Probe passes.
- `ds4_metal.m`:
  - `g_oproj_chunk_in_iosurfs` / `g_oproj_chunk_in_mtl_bufs` — global pool.
  - `ds4_oproj_chunk_in_pool_ensure` — lazy growth.
  - `ds4_oproj_ensure` attaches chunks to int8 conv ctxs at prewarm time.
  - `ds4_gpu_oproj_ane_async_start_tensor` encodes the N cpy_f32_f16
    dispatches into the current CB before flush.
  - Worker `chunk_io_active` branch skips input conversion entirely.

**Memory.** Pool = 64 IOSurfaces × 2 MB = 128 MB shared across all ctxs and
layers.  86 ctxs × 64 requests = 5504 ANE request objects (~50 MB).  Total
~180 MB persistent vs the prior ~16 MB transient per-job scratch +
hypothetical 67 MB shared staging.  Net: similar memory, but no per-call
alloc/free.

**Production result.** input_ms goes from 147 ms (NEON) → **0 ms** (GPU
does it).  Prefill at the same 262 t/s as the per-thread NEON variant —
the gain on prefill is within run-to-run noise because NEON had already
amortized the conversion cost effectively.  The architectural win (CPU
input path eliminated) is the value here, not raw prefill t/s.

---

## How to test

### Build

```bash
cd /path/to/ds4-ssd
make ds4
```

Also build the standalone smokes for ANE-only timing (independent of model
file / SSD IO):

```bash
make moe-batch-bench/ane_ds4_oproj_constexpr_smoke
make moe-batch-bench/ane_ds4_oproj_int8_smoke
make moe-batch-bench/ane_per_chunk_iosurface_probe
```

(or build each by hand if the Makefile target isn't wired:)
```bash
xcrun clang -fobjc-arc -O2 -Wall -I moe-batch-bench \
    -framework Foundation -framework IOSurface -ldl \
    -o moe-batch-bench/ane_ds4_oproj_int8_smoke \
    moe-batch-bench/ane_ds4_oproj_int8_smoke.m \
    moe-batch-bench/ane_ds4_mlp_int8w.o
```

### Standalone smoke (no model required)

```bash
# fp16 conv constexpr, batch sweep, multi-thread:
./moe-batch-bench/ane_ds4_oproj_constexpr_smoke -H 4096 -I 8192 -B 256 -iters 30 -threads 1,2,4

# int8 conv constexpr, same sweep:
./moe-batch-bench/ane_ds4_oproj_int8_smoke -H 4096 -I 8192 -B 256 -iters 30 -threads 1,2,4

# Per-chunk IOSurface probe (validates ANE accepts N requests on one model):
./moe-batch-bench/ane_per_chunk_iosurface_probe -H 4096 -I 8192 -B 256 -N 33
```

### Full prefill A/B (needs the DSv4 model)

Set `DS4_MODEL` and `DS4_SIDECAR` if not at the default `/Volumes/optane/...`
path.  The `run_ane_prefill_profile_m3u.sh` script (also referenced for M4
Pro / M5 Max in sibling scripts) handles the common knobs.

```bash
# GPU baseline:
DS4_RUN_NAME=gpu_baseline \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 \
  DS4_TOKENS=50 DS4_SLOTS=8 \
  ./run_ane_prefill_profile_m3u.sh

# Best O-proj ANE config we found on M3U:
DS4_RUN_NAME=oproj_best \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_INT8=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=2 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_GPU_CVT=1 \
  DS4_TOKENS=50 DS4_SLOTS=8 \
  ./run_ane_prefill_profile_m3u.sh

# Each variant in the table above flips one or more flags accordingly.
```

Numbers to read from the log:

```bash
log=$(ls -t moe-batch-bench/profile_runs/oproj_best_*.log | head -1)
grep -E "^ds4: prefill:|metal layer-major prefill total|O-proj calls=|O-proj prewarm" "$log"
```

Headline fields:

- `ds4: prefill: <X> t/s` — prefill throughput
- `O-proj calls=43 ... eval_avg=<X> ms/call` — per-layer ANE wall
- `O-proj ... join_wait_ms=<X> join_avg=<X> ms/call` — main-thread block per layer (real serial cost)
- `O-proj ... input_ms=<X>` — CPU input conversion (should be 0 with `_GPU_CVT=1`)
- `O-proj prewarm: 43/43 layers, <X> ms total` — one-time setup, OUTSIDE the prefill timer

---

## M5 hand-off

This work was developed and measured on **M3 Ultra (96 GB)**.  M5 Max has a
materially different ANE characteristic — notably, **INT8 GEMM appears to
run with dedicated int8 hardware** with higher peak TOPS than fp16, where
M3 Ultra effectively executes all the constexpr int8 paths as fp16 GEMM
(weights are dequantized at compile time per `constexpr_blockwise_shift_scale`).

### Prerequisites on m5m.local

- Repo synced to the `prefill-ANE` branch (`git pull` after pushing).
- DSv4 IQ2_XXS model + sidecar at the same path (or override
  `DS4_MODEL` / `DS4_SIDECAR` in your run command).
- The `run_ane_prefill_profile_m5max.sh` script already exists in the repo.

### Quick smoke comparison first

Run both standalone smokes on m5m at the same shape — these isolate ANE
throughput from GPU + IO completely:

```bash
make moe-batch-bench/ane_ds4_oproj_constexpr_smoke
make moe-batch-bench/ane_ds4_oproj_int8_smoke

./moe-batch-bench/ane_ds4_oproj_constexpr_smoke -H 4096 -I 8192 -B 256 -iters 30 -threads 1,2
./moe-batch-bench/ane_ds4_oproj_int8_smoke      -H 4096 -I 8192 -B 256 -iters 30 -threads 1,2
```

Expected on M3U (baseline):
- fp16 conv  2-thread: 3.91 ms/iter,  8.78 TFLOP/s
- int8 conv  2-thread: 2.49 ms/iter, 13.80 TFLOP/s (1.57× fp16)

**On M5 Max watch for:**
1. Does int8 conv hit substantially higher TFLOP/s than fp16 conv (e.g.,
   > 2× rather than 1.57×)?  That would indicate the constexpr int8 path is
   actually using INT8 GEMM hardware.
2. Does the 2-thread scaling stay near 2×?  On M3U it does.

### Full prefill A/B

Same env flags as M3U.  Use the M5 Max script:

```bash
# Baseline:
DS4_RUN_NAME=m5_gpu \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 \
  DS4_TOKENS=50 \
  ./run_ane_prefill_profile_m5max.sh

# All O-proj ANE optimizations stacked:
DS4_RUN_NAME=m5_oproj_best \
  DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_INT8=1 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=2 \
  DS4_FLASH_MOE_ANE_OUTPUT_PROJ_GPU_CVT=1 \
  DS4_TOKENS=50 \
  ./run_ane_prefill_profile_m5max.sh
```

On M5 Max with int8 GEMM acceleration, the ANE per-call eval may drop from
the ~189 ms/layer we see on M3U to something competitive with GPU.  If so,
the O-proj ANE path could match or beat the GPU baseline.

### If int8 GEMM doesn't kick in (constexpr path still fp16 GEMM)

The next experiment is **int8 activations**.  Currently the MIL takes X as
fp16, dequantizes constexpr int8 weights to fp16, and matmuls fp16 × fp16.
To run true INT8 GEMM we'd need:

- A `quantize` op on X inside MIL to produce int8 activations (per-tensor
  scale computed at runtime or fed as a per-call input).
- The conv op operating on int8 × int8 (the existing routed-expert i8w-i8x
  modes do this for a different shape).
- A `dequantize` op on the output to fp16 before the next stage.

The DSv4 routed-expert path already has this for its MLP shape (modes 3/4/5/6
in `ane_ds4_mlp_int8w.m`).  Adapting one of those for the linear two-stage
LoRA shape (Wa × Wb, no SiLU/mul) would be the natural next step.

Quality risk: int8 activations quantize harder than int8 weights (activations
are more dynamic).  Should be validated with a decode-quality comparison
(50+ tokens generated, compare against GPU baseline output).

### What's NOT in tree

- True INT8 GEMM (int8 activations).  See above.
- Output-side GPU conversion (`kernel_cpy_f16_f32` on the output IOSurface
  before downstream GPU ops read it).  Current path NEON-converts output
  fp16 → fp32 in the worker.  Same architecture as input conversion would
  work but wasn't built — output_ms is already small (~3 ms/layer) so the
  win is marginal on M3U.  May matter more on M5.
- M5-specific tuning of `DS4_FLASH_MOE_ANE_OPROJ_BATCH`.  The current B=256
  default is M3U-tuned.

---

## Earlier notes (for context)

### Why the GPU baseline still wins on M3U

GPU per-layer O-proj wall ≈ 67 ms (from `DS4_METAL_LAYER_STAGE_PROFILE=1`).
For ANE to beat GPU end-to-end:

- **Path A: faster ANE.** ANE per-layer needs ≤ 67 ms.  Smoke best case
  (int8 conv dual cluster) is 82 ms; production is 189 ms.  Both > 67 ms.
- **Path B: hide ANE behind GPU.** Requires layer-local GPU work between
  O-proj producer (`inv_rope`) and consumer (`hc_expand_split`).  None
  exists, so any ANE > 0 ms lands directly on the wall.  Pipelining across
  layers is also blocked (layer N+1's attention depends on layer N's
  residual which depends on layer N's `batch_attn_out`).

The constraint is structural (pipeline orchestration on M3U with this model
shape), not algorithmic.  On a chip where ANE INT8 GEMM beats GPU FP16 GEMM
per-call, Path A becomes reachable.

### Wait-time probes

The ANE prefill stats printer surfaces `dep_wait_ms` (worker idle waiting
for GPU dep_done) and `join_wait_ms` (main thread blocked in
`pthread_join` waiting for ANE worker).  Both are part of the `ds4: ANE
O-proj calls=...` line at end-of-prefill.  Use these to triangulate
whether changes affect the worker's critical path, the main thread's
serial cost, or neither.

### Files added by this investigation

- `moe-batch-bench/ane_ds4_oproj_constexpr_smoke.m` — fp16 conv linear smoke.
- `moe-batch-bench/ane_ds4_oproj_int8_smoke.m` — int8 conv linear smoke.
- `moe-batch-bench/ane_per_chunk_iosurface_probe.m` — multi-request probe.
- `moe-batch-bench/ane_ds4_mlp_int8w.{h,m}`:
  - `ds4_ane_mlp_fp16w_linear_eval` (mode 1, linear no-activation eval).
  - `ds4_ane_mlp_i8w_fp16x_linear_eval` + `_create` (mode 2, two-scale).
  - `gen_mil_fp16w_linear_constexpr_conv` + `_create` + `_eval` (mode 10).
  - `gen_mil_int8w_linear_constexpr_conv` + `_create` + `_eval` (mode 11).
  - `ane_build_blob_n` + `ane_build_fp16_blob_2` (blob builders).
  - `ds4_ane_mlp_int8w_linear_constexpr_attach_chunks` + `_eval_at_chunk`.
- `ds4_metal.m`:
  - `ds4_neon_f32_to_f16` / `ds4_neon_f16_to_f32` (NEON SIMD conversion).
  - `ds4_quantize_f16_to_i8_per_tensor` / `_per_channel` (int8 quant).
  - `ds4_oproj_layer_cache` + all the ensure/worker plumbing.
  - `g_oproj_chunk_in_iosurfs` + GPU input conversion path.
  - Stats counters surfaced in the prefill stats printer.
- `ds4_gpu.h`: `ds4_gpu_oproj_ane_{async_start_tensor, async_finish_tensor, prewarm}`.
- `ds4.c`: conditional dispatch at the prefill batched attention output;
  prewarm loop at engine open.
- `moe-batch-bench/FUSED_MLP_ANE_CONV.md` — companion doc for the shared-
  expert conv path (off by default for similar reasons).
