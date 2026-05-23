# Attention output projection (O-proj) on ANE — investigation notes

## TL;DR

For the DSv4 IQ2_XXS prefill on M3 Ultra, **moving the attention output
projection from GPU to ANE doesn't beat the GPU baseline** under any
combination of (matmul vs conv2d-1×1, single vs dual cluster, batch size
256–2048) we tested.  The standalone ANE smoke shows per-call performance is
fine; the loss happens at pipeline level because the O-proj producer
(`inv_rope`) feeds its consumer (`hc_expand_split`) directly with no
intervening GPU work that an ANE worker could overlap.

All variants stay in tree behind env flags but **off by default**.

## What's wired up

| Path | Mode | Env | Default |
|------|------|-----|---------|
| GPU q8_0 batched matmul (`ds4_gpu_attention_output_q8_batch_tensor`) | — | none | **on** |
| ANE matmul split (fp16w mode 1, `ds4_ane_mlp_fp16w_linear_eval`) | matmul | `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1` | off |
| ANE constexpr conv2d-1×1 (mode 10, `ds4_ane_mlp_fp16w_linear_constexpr_eval`) | conv | `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1` + `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1` | off |
| Multi-cluster ANE workers | both | `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=1..4` | 1 |
| Compiled-batch override | both | `DS4_FLASH_MOE_ANE_OPROJ_BATCH=N` | 256 |

## A/B results on M3U (8423-token prompt, DS4_TOKENS=2, single trial each)

| Config | Prefill | ANE per-layer eval | Main thread join wait per layer | Δ vs GPU baseline |
|--------|---------|---------------------|----------------------------------|--------------------|
| **GPU baseline (no O-proj ANE)** | **286 t/s** | — | — | — |
| ANE matmul, 1 worker | 184 t/s | 349 ms | 626 ms | **−102 t/s** |
| ANE conv, 1 cluster, B=256 | 193 t/s | 281 ms | 558 ms | −93 t/s |
| ANE conv, 2 clusters, B=256 | 229 t/s | 300 ms | 412 ms | −57 t/s |
| ANE conv, 2 clusters, B=512 | 225 t/s | 288 ms | 413 ms | −61 t/s |
| ANE conv, 2 clusters, B=1024 | 226 t/s | 295 ms | 421 ms | −60 t/s |
| ANE conv, 2 clusters, B=2048 | 218 t/s | 321 ms | 448 ms | −68 t/s |

## Standalone smoke at the O-proj shape (`ane_ds4_oproj_constexpr_smoke`)

H=4096 I=8192 B=256, 30 iters/thread:

| Threads | Wall (ms) | ms/iter | TFLOP/s | Scaling vs 1 |
|---------|-----------|---------|---------|--------------|
| 1 | 233.7 | 7.79 | 4.41 | 1.00× |
| 2 | 234.8 | **3.91** | **8.78** | **1.99× (dual cluster works)** |
| 4 | 445.4 | 3.71 | 9.26 | 2.10× |

Dual cluster scales nearly linearly — the existing dual-ANE-cluster
infrastructure works for this shape.  Per-call (3.91 ms × 33 chunks) implies
~129 ms/layer; production saw 295–321 ms/layer (~2.5× slower than smoke).
The gap likely comes from CPU-side conversion, dep_wait on the GPU command
buffer completion handler, and per-layer ctx swap (43 layers × 2 workers =
86 compiled constexpr ctxs in flight).

## Why the GPU still wins

GPU per-layer O-proj wall ≈ 67 ms (from `DS4_METAL_LAYER_STAGE_PROFILE=1`).
For ANE to beat GPU end-to-end:

- **Path A: faster ANE.** ANE per-layer needs ≤ 67 ms.  Best case (dual
  cluster + ideal smoke conditions) is 129 ms — already 2× too slow, and
  int8 weights via `constexpr_blockwise_shift_scale` dequant at compile
  time so ANE runtime stays fp16 GEMM (memory pressure helps; throughput
  doesn't).
- **Path B: hide ANE behind GPU.** Requires ≥ 300 ms of layer-local GPU work
  between O-proj producer and consumer.  The actual dataflow:

  ```
  attention → inv_rope → [O-proj here] → hc_expand_split → norm → ffn → residual
  ```

  `hc_expand_split` and everything after depends on `batch_attn_out`.
  Pipelining across layers also fails: layer N+1's attention depends on
  layer N's residual which depends on layer N's `batch_attn_out`.

Neither path is reachable inside the conversion alone.

## Where this code might still pay off

- **Configs where GPU is much slower** (smaller GPU, IO-bound prefill where
  GPU has slack, mobile chips with lower GPU/ANE ratio).
- **Configs where ANE is dramatically faster** (M5 Pro / M5 Max with newer
  ANE).
- **Decode** (single-token batch where the entire pipeline is sequential
  and per-layer cost matters more uniformly).  Not tested here.

## Files added by this investigation

- `moe-batch-bench/ane_ds4_oproj_constexpr_smoke.m` — standalone harness for
  O-proj shape, 1/2/4 thread sweep.
- `moe-batch-bench/ane_ds4_mlp_int8w.{h,m}` — `ds4_ane_mlp_fp16w_linear_eval`
  (mode-1 ctx, no activation between matmuls), `gen_mil_fp16w_linear_constexpr_conv`,
  `ane_build_fp16_blob_2`, `ds4_ane_mlp_fp16w_linear_constexpr_create`,
  `ds4_ane_mlp_fp16w_linear_constexpr_eval` (mode 10).
- `ds4_metal.m` — `ds4_oproj_layer_cache` with both matmul-shared-ctx and
  conv-per-layer-per-worker ctxs, `ds4_gpu_oproj_ane_{worker, stride_worker,
  async_start_tensor, async_finish_tensor, prewarm}`.  Stats counters
  `g_oproj_ane_{calls, total_ms, eval_ms, input_ms, output_ms, init_ms,
  dep_wait_ms, join_wait_ms}` surfaced in the ANE prefill stats printer.
- `ds4_gpu.h` — public API for the O-proj ANE start/finish/prewarm.
- `ds4.c` — conditional dispatch at line 15540 (prefill batched attention
  output); prewarm loop at engine open.

## What to try next

Not O-proj-on-ANE-with-yet-another-knob — the orchestration ceiling is the
real constraint.  Either:

- Pick a target with actual async opportunity (a GPU op that has substantial
  layer-local downstream GPU work between its producer and consumer).
- Or accept that the shared-expert work has captured the available GPU→ANE
  shift for this pipeline and focus elsewhere (decode path, IO, GPU kernel
  efficiency).
