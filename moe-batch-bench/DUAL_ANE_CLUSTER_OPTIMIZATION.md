# Dual-ANE Cluster Optimization (M3 Ultra)

## Context

The M3 Ultra has **two ANEx16 clusters**. The pre-existing ANE prefill path
used only one cluster — the per-call `predict` API issued from a single thread
serialised both clusters into a single device.  This document captures the
work to extract the second cluster's throughput in the production MoE prefill
path used by `run_ane_prefill_profile_m3u.sh`.

End-user prefill result: **220.49 t/s GPU-only → 270.26 t/s with dual-cluster
ANE = 1.226x speedup** on DSv4 IQ2_XXS at 8423-token prefill.

## Validation: smoke tests prove 2x is achievable

Two in-process pthread smokes establish that the M3 Ultra OS willingly
dispatches concurrent `predict` calls to two different clusters:

| Smoke | Engine | Shape | Solo (per cluster) | Concurrent (per thread) | Aggregate |
|---|---|---|---|---|---|
| `ane_ds4_mlp_split3_dual_smoke.m` | split3 | H=7168, I=18432, B=128 | 26.05 ms / 3.90 TF/s | 26.10 ms (+0.2%) | **7.75 TF/s, 1.99x** |
| `ane_ds4_mlp_int8w_dual_smoke.m` | i8i8 tiled-fused | H=4096, I=2048, B=128 | 1.31 ms / 4.93 TF/s | 1.34 ms (+2%) | **9.58 TF/s, 1.97x** |
| `ane_ds4_mlp_int8w_dual_smoke.m` | i8i8 tiled-fused | H=4096, I=2048, B=256 | 1.83 ms / 7.03 TF/s | 2.11 ms (+15% bw contention) | **12.19 TF/s, 1.74x** |

Both contexts must be separate handles; the underlying `_ANEInMemoryModel` is
a process-shared resource but per-context state (IOSurfaces + model handle)
keeps the clusters independent.  At B=128 the bandwidth headroom is generous;
by B=256 weight-upload contention costs ~15% per cluster but the aggregate is
still ~1.74x.

Build either smoke (sandbox must be off so the private ANE compile can write
its temp file):

```
cd moe-batch-bench
clang -fobjc-arc -O2 \
    ane_ds4_mlp_int8w.m ane_ds4_mlp_int8w_dual_smoke.m \
    -framework Foundation -framework IOSurface -lpthread \
    -o ane_ds4_mlp_int8w_dual_smoke
./ane_ds4_mlp_int8w_dual_smoke -batches 128,256 -warmup 5 -iters 80
```

## Integration architecture

The dual-ANE-cluster implementation is **split across two files**, and both
halves are required for the optimization to do anything.  Either alone is
inert:

- **`ds4_metal.m`** holds the worker-side mechanics: a per-job dequant scratch
  slot pool, a secondary context cache for cluster B, two ANE worker thunks
  (one per cluster), and the chunk-completion bitmap that lets the post
  thread stay in-order.
- **`ds4.c`** holds the scheduler-side prerequisite: the
  `DS4_FLASH_MOE_ANE_MULTI_ACTIVE` branch of `DS4_WAIT_ACTIVE_ANE_PREDICT`
  that defers the synchronous `wait_predict` join.  Without this, the
  scheduler still serialises ANE evals at the `active_ane_job` boundary and
  the cluster-B worker would never overlap with cluster-A's eval — making
  the slot pool correct but useless.

The production prefill scheduler in `ds4.c` was already designed around a
single `active_ane_job` plus a `ready_ane[]` post-processing queue.  Three
coordinated changes were needed to actually run two ANE jobs in parallel.

### 1. Per-job dequant scratch slot pool — `ds4_metal.m`

**Problem.** The dequant scratch buffers (`g_ane_prefill_gate_i8_buffer` and
friends) were global singletons.  Two in-flight ANE jobs would clobber each
other's weights: job N's dequant kernel would overwrite job N-1's bytes while
N-1's ANE worker was still reading them.

**Fix.** A 12-slot pool `g_ane_dequant_slots[DS4_ANE_DEQUANT_SLOTS]`, each
slot holding its own `gate_i8` / `up_i8` / `down_i8` / `x_i8` MTLBuffers and
(in a later commit) the output buffers `out_f16_all` / `out_f32` /
`route_weights`.  `ds4_gpu_ane_acquire_dequant_slot()` blocks on a condvar
when all slots are in use, so the slot count caps maximum concurrency.

The slot is released by the *last* ANE worker thread to finish reading from
it — workers atomically decrement `job->ane_threads_remaining` under the job
mutex and the thread that hits zero calls `ds4_gpu_ane_release_dequant_slot`.
Defensive cleanup also runs in `ds4_gpu_ane_prefill_job_free` in case a
worker errored before releasing.

### 2. Dual-cluster ANE worker pair — `ds4_metal.m`

The existing single-cluster worker
(`ds4_gpu_ane_prefill_tiled_fused_thread`) is left intact for the solo path.
Two new thunks plus a shared strided helper drive the dual case:

```
ds4_gpu_ane_prefill_tiled_fused_thread_dual_a  (waits for dequant_done, runs even chunks)
ds4_gpu_ane_prefill_tiled_fused_thread_dual_b  (optional stagger, runs odd chunks)
        \                  /
         ds4_gpu_ane_prefill_run_chunks_strided
```

Each worker uses its own context (`ane_ctx` for A, `ane_ctx_b` for B) and its
own per-cluster scratch (`x_i8_batch`/`out_f16_batch` vs `_b` variants).
Both write into disjoint regions of the shared `out_f16_all` (different
chunk-row offsets), so the post-thread is unchanged.

A secondary context cache `g_ane_ctx_cache_b` is required because the primary
cache `g_ane_ctx_cache` deduplicates by `(mode, H, I, B, scales)` — without
a parallel cache, asking for "another context for the same shape" would just
return the cluster-A context a second time.

**Chunk ordering & post-thread.** Today the post-thread treats `chunks_ready`
as a contiguous high-watermark.  With two workers, chunk completions arrive
out of order.  A `chunks_done_bitmap` byte-array tracks per-chunk completion;
each worker sets its bit under the mutex and advances `chunks_ready` as far
as the bitmap is contiguous from 0 — preserving the post-thread's in-order
semantics.

### 3. Multi-active scheduler — `ds4.c`

**Problem.** Even with per-job scratch, the scheduler's
`DS4_WAIT_ACTIVE_ANE_PREDICT` macro called `wait_predict_tensor` to join the
previous ANE thread *before* starting the next one.  This serialised ANE
evals at the scheduler level: only one ANE job actually computed at a time
across the prefill.

**Fix.** Guarded by `DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1`, the macro now skips
the synchronous join and transfers `active_ane_job` to the predicted-handle
slot without blocking.  The eventual `finish_tensor` (called when the job
pops from `ready_ane[]`) does the join.  Net effect: the previous job's ANE
evaluation overlaps with the next job's evaluation on the *other* cluster.

Telemetry confirms it works: `wait_predict_calls=0` after the change, and
`finish_join_ane_ms` ~ 434 ms vs the old `wait_predict_ms` ~ 7518 ms.

### 4. Auto chunk-split for dual

A single-chunk ANE call only feeds cluster A — the dual path requires at
least 2 chunks per call to engage cluster B.  In `start_tensor`, when
`DS4_FLASH_MOE_ANE_DUAL=1` and `n_tokens >= 128`, the selected `ane_batch`
is dropped to the smallest batch >= `ceil(n_tokens/2)`.  E.g. a call with
n_tokens=200 normally runs as one B=256 chunk; with dual it splits into two
B=128 chunks (one per cluster).

A second balancing tweak: for calls with `chunk_count == 2`, the chunk size
is set to `ceil(n_tokens/2)` so A and B do equal work instead of A doing
`ane_batch` refs while B does the remainder.

For *single-chunk* calls (n_tokens too small to split), a round-robin
counter sends alternating calls to ctx A vs ctx B so the cluster load
balances across calls.

### 5. MTLBuffer pool in the slot

Per-call `[g_device newBufferWithLength:]` for `out_f16_all_mtl`,
`out_f32_mtl`, and `route_weights_mtl` was ~500ms-2s of allocator churn per
prefill at production workload.  These buffers are now slot-owned and grown
via `ds4_gpu_ensure_scratch_buffer` on demand — no per-call newBufferWithLength.

### 6. Batched dequant CB commit

`DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=N` (default 4) skips the per-call pre-flush
`ds4_gpu_flush_commands()` on N-1 of every N calls.  Multiple ANE calls'
dequant kernels accumulate into the same batch CB, committed at the next
preflush.  Saves ~150-500ms of Metal commit syscall overhead per prefill.

### 7. Hybrid scheduler tuning

The biggest single end-user gain was env-only:
`DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=384`.  By routing expert groups with fewer
than 384 refs to the GPU instead of ANE, the per-call fixed overhead is
amortised over more work per ANE call.  Sweep peak: 384.  Performance
degrades both below (too many small calls cost more in overhead than they
save in ANE compute) and above (too much GPU load).

## Env knobs

All defaulted in `run_ane_prefill_profile_m3u.sh`:

| Env | Default | Effect |
|---|---|---|
| `DS4_FLASH_MOE_ANE_DUAL` | 1 | Enable dual-cluster ANE prefill |
| `DS4_FLASH_MOE_ANE_DUAL_STAGGER_US` | 0 | Phase-offset between clusters (smoke showed no benefit in production) |
| `DS4_FLASH_MOE_ANE_DUAL_SPLIT_THRESH` | 128 | n_tokens >= this triggers auto half-batch select |
| `DS4_FLASH_MOE_ANE_MULTI_ACTIVE` | 1 | Defer `wait_predict` join until `finish_tensor` |
| `DS4_FLASH_MOE_ANE_OUTPUT_QUEUE` | 4 | Max in-flight post-eval jobs |
| `DS4_FLASH_MOE_ANE_BATCHES` | 256 | ANE batch sizes considered (single-size keeps the cache simple) |
| `DS4_FLASH_MOE_ANE_MAX_REFS` | 256 | Cap on chunk refs |
| `DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK` | 1 | GPU does f16→f32 + route-weight scaling (skips post_thread) |
| `DS4_FLASH_MOE_ANE_PREFLUSH_EVERY` | 4 | Batch this many calls' dequants per CB commit |
| `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS` | 384 | Refs < this go to GPU instead of ANE |
| `DS4_FLASH_MOE_ANE_FUSED_DEQUANT` | 0 | Use fused gate+up+down dequant kernel (no measured benefit, see below) |

## Measured results

DSv4 IQ2_XXS, 8423-token prefill on the coding_8k prompt:

| Config | Prefill t/s | Speedup |
|---|---|---|
| GPU-only (`run_gpu_prefill_profile_m3u.sh`) | 220.49 | baseline |
| ANE-only Q=1, no DUAL, no MULTI (original) | ~218 | 0.99x (no gain) |
| ANE + DUAL + MULTI + all optimisations | **270.26** | **1.226x** |
| Best observed run | 283.97 | 1.288x |

Run-to-run variance is ~±5 t/s; sustained typical is ~275-280 t/s.

**ANE-side metrics** at the optimized config (1238 ANE prefill calls, 4945
chunks across 60 layers):

```
ane_eval_a = 9984 ms       (cluster A active window)
ane_eval_b = 6856 ms       (cluster B active window)
ane_wall_est = 9984 ms     (max(A, B))
sum(eval) = 16840 ms       (A + B, the cumulative compute work)
pad_util = 88%             (12% of ANE cycles spend on zero-work padding)
wait_predict_calls = 0     (multi-active bypass working)
finish_join_ane = 434 ms   (where the deferred join lands)
```

ANE compute rate: 56 / 9.984 = **5.6 sustained TOPS** (int8) aggregate across
both clusters.  Against the M3 Ultra ANE's ~38 TOPS nominal int8 peak,
that's ~15% utilization — vs ~32% in the tight-loop smoke.  The gap is
absorbed by padding waste, the cluster-B 30% idle window from single-chunk
calls, and per-call dispatch overhead that doesn't fully overlap with ANE
compute.

## What was tried and didn't help

**Fused gate+up+down dequant kernel.**  Implemented as
`kernel_dsv4_mpp_dequant_gate_up_down_i8` in `metal/moe.metal`.  Collapses
three separate dispatches into one (all three have `total = 524288` work
items on DSv4 so they share a single launch).  Three trials each: mean 279.6
t/s fused, 279.6 t/s unfused — fully within run-to-run noise.

The reason: Metal already shares the compute encoder across the three
standalone dispatches (`ds4_gpu_compute_encoder` returns `g_batch_enc` for
batch CBs and `ds4_gpu_end_compute_encoder` is a no-op).  So the
`setComputePipelineState` + `dispatchThreadgroups` overhead I expected to
save is essentially already zero.  Kept the kernel as a code path behind
`DS4_FLASH_MOE_ANE_FUSED_DEQUANT=1` for future shapes where the encoder
might not be shared.

**Single-chunk A↔B alternation alone (without changing the scheduler).**
Routing alternate single-chunk calls to ctx B gives no speedup because the
scheduler still serialises ANE evals — the two clusters can't compute
concurrently unless `MULTI_ACTIVE` is also on.  The optimization landed
naturally as part of the multi-active path.

**Increasing OUTPUT_QUEUE past 4 with MULTI_ACTIVE.**  Q=8 and Q=12 measured
within ±1 t/s of Q=4 — once the queue is deep enough to hide post-write-back
latency, more depth doesn't help.

## Future levers

The wall is now ~95% CPU-side Metal command encoding (~28 s of ~30 s total).
ANE wall is well overlapped with GPU encoding.  The remaining levers all
require larger refactors:

- **Skip `write_surface` memcpy in the ANE library.**  The int8w eval's first
  action is a memcpy from the caller's i8 buffer into the ANE IOSurface
  (~480 μs per call × 5000 calls = ~2.4 s/prefill).  Exposing the IOSurface
  as a Metal-writable buffer (so the dequant kernel writes directly into the
  IOSurface) would eliminate this entirely.  Requires changes to the ANE
  library's IOSurface ownership model.
- **Pre-quantize the hidden state once per layer.**  The same `x` is
  quantized per ANE call today; cached layer-level quantization would
  collapse 5000+ x-quant dispatches into 60.
- **Persistent ANE worker threads + job queue.**  `pthread_create`/`pthread_join`
  per ANE call is ~200 ms of overhead per prefill.  Persistent workers with
  a shared job queue would eliminate it and keep both ANE clusters
  continuously fed across call boundaries.

None of these are dual-cluster-specific; they would lift both GPU-only and
ANE+dual paths.

## Files touched

The dual-ANE-cluster optimization lives in **two source files**; both halves
are required.

| File | Role | What it adds |
|---|---|---|
| `ds4_metal.m` | **Dual-ANE-cluster — worker side** | Slot pool, secondary context cache, two ANE worker thunks + strided helper, chunk-done bitmap, A↔B balancing, auto chunk-split, MTLBuffer pool, preflush batching, fused kernel encoder |
| `ds4.c` | **Dual-ANE-cluster — scheduler side** | `DS4_FLASH_MOE_ANE_MULTI_ACTIVE` gate in `DS4_WAIT_ACTIVE_ANE_PREDICT` — defers `wait_predict` join so the two clusters can compute concurrently |
| `metal/moe.metal` | Optional / no-op on DSv4 | `kernel_dsv4_mpp_dequant_gate_up_down_i8` (fused dequant) |
| `run_ane_prefill_profile_m3u.sh` | Reproducibility | Dual-cluster optimal defaults |
| `moe-batch-bench/ane_ds4_mlp_split3_dual_smoke.m` | Validation | Smoke (dense MLP shape) |
| `moe-batch-bench/ane_ds4_mlp_int8w_dual_smoke.m` | Validation | Smoke (production expert shape) |

## Reproducing the result

```bash
bash run_ane_prefill_profile_m3u.sh
# expect ~275-282 t/s
```

To isolate the ANE contribution, also run the GPU-only counterpart and take
the ratio:

```bash
bash run_gpu_prefill_profile_m3u.sh
# expect ~218-225 t/s
# ratio: ~1.22-1.28x
```

The dual-cluster smokes confirm the underlying hardware capability and are
useful for sanity-checking ANE behavior after macOS or driver updates:

```bash
cd moe-batch-bench
./ane_ds4_mlp_int8w_dual_smoke -batches 128,256 -warmup 5 -iters 80
```
