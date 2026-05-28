# ANE (Apple Neural Engine) Implementation Analysis

## Overview

The ANE implementation in the ds4-ssd project provides a high-performance neural network inference path using Apple's private `ANEServices` framework on Apple Silicon. It replaces GPU-based matrix multiplications with ANE evaluation for the **shared expert FFN** (gate + up + SwiGLU + down) and **attention output projection** paths in the DSv4 MoE (Mixture of Experts) model.

The codebase spans three primary files:
- **`ds4_metal.m`** (~24,000 lines) — GPU/Metal dispatch, ANE job scheduling, profiling
- **`moe-batch-bench/ane_ds4_mlp_int8w.m`** (~2,777 lines) — MIL text generation, ANE model compilation/eval
- **`ds4_gpu.h`** — Public C API for async ANE dispatch

---

## 1. Architecture — Two-Level Design

### Level 1: MIL Backend (ane_ds4_mlp_int8w.m)

**MIL (Model Intermediate Language)** is Apple's neural network IR for the ANE. The code generates MIL text at runtime for various matrix multiplication configurations, then compiles and loads the model via private `_ANEInMemoryModel` APIs.

#### Supported Modes (via `mode` parameter)

| Mode | Name | Description |
|------|------|-------------|
| 0 | `int8w` | Legacy int8 weights, fp16 activations |
| 1 | `fp16w` | fp16 weights + activations (split gate/up, down) |
| 2 | `i8w-fp16x` | int8 weights, fp16 activations (split) |
| 3 | `i8w-i8x` | int8 weights + activations (split) |
| 4 | `i8w-i8x-fused` | **Single model**: 4-input (gate, up, down, x), 1-output fused MLP |
| 5 | `i8w-i8x-gateup-fused` | Gate+Up fused, then down split |
| **6** | **`i8w-i8x-tiled-fused`** | **Production path**: single fused MLP with tiling support |
| 7 | `i8w-i8x-tiled-fused-i8out` | Tiled-fused with int8 output (not used) |
| 8 | `fp16w-fused-conv` | fp16 fused conv (not used) |

#### Key APIs

```c
// Create a compiled ANE context for a specific shape + quantization scales
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_create(
    int H,        // input dimension (expert_in_dim)
    int I,        // hidden dimension (expert_mid_dim)
    int B,        // batch size
    float w_scale, // weight quantization scale
    float x_scale, // input quantization scale
    float mid_scale // intermediate quantization scale
);

// Evaluate the MLP on the ANE
int ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *gate_i8,  // gate weights (int8)
    const int8_t *up_i8,    // up weights (int8)
    const int8_t *down_i8,  // down weights (int8)
    const int8_t *x_i8,     // input (int8)
    uint16_t *out_f16       // output (fp16)
);
```

#### Compilation Flow

1. **MIL text generation** (`gen_mil_i8w_i8x_tiled_fused`) produces Apple Neural Engine intermediate language
2. **Model descriptor** created via `modelWithMILText:weights:optionsPlist:`
3. **In-memory model** created via `inMemoryModelWithDescriptor:`
4. **Compile** via `compileWithQoS:options:error:` (QoS 21 = highest priority)
5. **Load** via `loadWithQoS:options:error:`
6. **IOSurfaces** allocated for gate/up/down weights, input, and output
7. **Request** created via `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:`

### Level 2: GPU Dispatch & Async Pipeline (ds4_metal.m)

The GPU-side code handles **dequantization** (Q8_0 → int8), **ANE scheduling**, and **output packing**.

#### Job Structure (`ds4_gpu_ane_prefill_job`)

Key fields (from lines ~17550-17778):
- `n_tokens`, `ane_batch`, `ane_chunk_refs` — chunking parameters
- `dequant_slot` — pooled GPU dequantization buffer slot
- `ane_ctx`, `ane_ctx_b`, `ane_ctx_c`, `ane_ctx_d` — up to 4 per-cluster ANE contexts
- `x_i8_batch`, `out_f16_batch` — per-worker scratch buffers
- `chunks_done_bitmap` — tracks completion order for post-processing
- `n_workers` (1-4), `dual` flag — multi-cluster parallelism
- Synchronization: `mu` (mutex), `cv` (cond var), `dequant_done_flag`, `ane_done`, `ane_failed`

---

## 2. Data Flow — Async ANE Prefill Pipeline

```
Layer prefill start
│
├── GPU dequant (Q8_0 → int8) ← on GPU command buffer
│   ├── gate bank: IQ2_XXS → int8 (expert_mid_dim × expert_in_dim)
│   ├── up bank:   IQ2_XXS → int8 (expert_mid_dim × expert_in_dim)
│   ├── down bank: Q2_K → int8 (out_dim × expert_mid_dim)
│   └── x (input): fp32 → int8 quantize
│
├── Completion handler → signals dequant_done_flag
│
├── ANE worker threads (1-4 pthreads)
│   ├── Wait on dequant_done_flag
│   ├── For each chunk:
│   │   ├── memset scratch buffers
│   │   ├── memcpy x_i8 subset
│   │   ├── ds4_ane_mlp_i8w_i8x_tiled_fused_eval()
│   │   └── memcpy out_f16 → out_f16_all
│   ├── Atomically update chunks_done_bitmap
│   └── Signal ane_done when last worker completes
│
└── Post-processing (post_thread or GPU pack)
    ├── fp16 → fp32 conversion + route weights application
    ├── GPU output pack: Metal dispatch to pack fp16→fp32
    └── OR CPU scalar output pack (legacy)
```

### Chunking Strategy

- **Batch size** (`ane_batch`): default 256, configurable via `DS4_FLASH_MOE_ANE_BATCHES`
- **Chunk refs** (`ane_chunk_refs`): default = `min(ane_max_refs, ane_batch)` where `ane_max_refs` defaults to `ane_batch`
- **Balanced chunking**: when multi-worker, chunks are balanced across workers (`ceil(n_tokens / n_workers)`)
- **Dual-cluster split**: when `DS4_FLASH_MOE_ANE_DUAL=1` and `n_tokens ≥ 128`, batch is shrunk to `ceil(n_tokens / planned_workers)` so each worker gets a natural chunk

### Multi-Worker Parallelism (Up to 4 Workers)

Each worker gets a **strided** set of chunks (`start_idx + stride * n`):
- Worker A: chunks 0, 2, 4, 6... (stride = n_workers, start = 0)
- Worker B: chunks 1, 3, 5, 7... (stride = n_workers, start = 1)
- Worker C/D: same pattern with offsets 2/3

**Stagger**: Workers B/C/D sleep `stagger_us * 1/2/3` to spread ANE submission load across clusters.

**Solo balancing**: For single-chunk calls with multi-worker enabled, alternates which cluster (A vs B) the call lands on using a static counter.

---

## 3. GPU Dequantization

Before ANE evaluation, the quantized weights must be dequantized to int8:

| Weight Type | Dequant Kernel | Shape |
|-------------|----------------|-------|
| **Gate bank** (IQ2_XXS) | `ds4_gpu_encode_mpp_dequant_iq2_xxs_i8` | `expert_mid_dim × expert_in_dim` |
| **Up bank** (IQ2_XXS) | `ds4_gpu_encode_mpp_dequant_iq2_xxs_i8` | `expert_mid_dim × expert_in_dim` |
| **Down bank** (Q2_K) | `ds4_gpu_encode_mpp_dequant_q2_k_i8` | `out_dim × expert_mid_dim` |
| **Input x** (fp32) | `ds4_gpu_encode_mpp_quant_f32_i8` | `n_tokens × expert_in_dim` |

**Fused dequant** (`DS4_FLASH_MOE_ANE_FUSED_DEQUANT=1`): single Metal dispatch for all three weight dequant + input quant in one kernel.

### Dequant Slot Pooling

Multiple `ds4_ane_dequant_slot_t` slots are allocated to prevent overlapping scratch buffer use between concurrent ANE evaluations. The pool supports `DS4_FLASH_MOE_ANE_MULTI_ACTIVE` mode.

---

## 4. Output Packing

Two output pack strategies:

### GPU Output Pack (default on M5 Max)
```c
// Metal dispatch: fp16 → fp32 + apply route weights
// Uses scratch MTLBuffers for out_f16_all and route_weights
// Produces final float output in slot->out_f32
```

### CPU Scalar Output Pack
```c
// post_thread: memcpy row-by-row + fp16→fp32 + route weights
// Slower but doesn't need GPU compute for packing
```

---

## 5. Pre-Flush Batching

Controls how often the GPU command buffer is committed:

- `DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=N` — commit every Nth ANE call
- Default: 1 (commit every call = old behavior)
- Higher values batch N dequant dispatches into one commit, saving ~100-300 µs per commit × 5000 ANE calls

---

## 6. Context Caching

Compiled ANE models are cached per shape/scale combination:

```c
struct ds4_ane_ctx_cache_entry {
    ds4_ane_mlp_int8w_ctx *ctx;
    int mode, H, I, B;
    float w_scale, x_scale, mid_scale;
};
```

- 5 separate caches: `g_ane_ctx_cache`, `g_ane_ctx_cache_b`, `g_ane_ctx_cache_c`, `g_ane_ctx_cache_d`
- Each cache has `DS4_ANE_CTX_CACHE_MAX` (16) slots
- Compile limit: `DS4_FLASH_MOE_ANE_COMPILE_LIMIT` (default 8 attempts)
- **Warm-up**: shared expert weights can be pre-compiled via `ds4_gpu_shared_expert_ane_prewarm()`

---

## 7. Env Configuration (Production Settings)

### ANE-Only Prefill (M5 Max)
```
DS4_FLASH_MOE_ANE_PREFILL=1
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1          # ← critical for ANE-only
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1
DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1
DS4_FLASH_MOE_OVERLAP_PREFILL=1
DS4_FLASH_MOE_OVERLAP_SCHEDULER=1
DS4_FLASH_MOE_ANE_BATCHES=64,128,256,512
DS4_FLASH_MOE_ANE_MAX_REFS=256
DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS=1
DS4_FLASH_MOE_ANE_MIN_REFS=32
DS4_FLASH_MOE_ANE_DUAL=0                   # single-cluster (M5 Max)
DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=4
DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=4
DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=1
DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0
```

### Hybrid ANE/GPU (M3 Ultra)
```
DS4_FLASH_MOE_ANE_DUAL=1                   # dual-cluster
DS4_FLASH_MOE_ANE_THREADS=4               # 2 clusters × 2 workers
DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=384     # GPU spillover threshold
```

---

## 8. Shared Expert ANE Path (ds4_gpu.h)

Replaces 3 Q8_0 GPU matmuls + 1 SwiGLU with ANE evaluation:

```c
ds4_gpu_shared_expert_ane_async_start_tensor(in, gate_bank, up_bank, down_bank,
    weights, expert_in_dim, expert_mid_dim, out_dim, n_tokens);
ds4_gpu_shared_expert_ane_async_finish_tensor(job);
```

**Async variant**:
1. Attaches completion handler to GPU command buffer producing `in`
2. Flushes CB
3. Spawns worker thread: CPU fp32↔fp16 conversion + ANE eval + writeback
4. `finish_tensor` pthread_joins before downstream dispatch reads `out`

---

## 9. Attention Output Projection (O-Proj) ANE Path

Similar lifecycle to shared expert but for attention output:
```c
ds4_gpu_oproj_ane_async_start_tensor(in, attn_output_a, attn_output_b, n_tokens);
ds4_gpu_oproj_ane_async_finish_tensor(job);
```

Two-stage linear matmul (attn_output_a, attn_output_b). Gated behind `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1`.

---

## 10. MIL Text Generation

The MIL compiler generates Apple Neural Engine intermediate language at runtime. For mode 6 (tiled-fused), the generated MIL defines a single model with:

- **Inputs**: gate_i8, up_i8, down_i8, x_i8 (all int8)
- **Output**: out_f16 (fp16)
- **Operations**:
  - Gate matmul: `x_i8 × gate_i8^T` → int8 intermediate
  - Up matmul: `x_i8 × up_i8^T` → int8 intermediate
  - SwiGLU fusion: gate * sigmoid(gate) * up
  - Down matmul: intermediate × down_i8^T → fp16 output

The MIL is generated as a string at runtime, compiled by the ANE compiler, and cached by shape.

---

## 11. Error Handling & Fallbacks

- **Compile failure**: returns NULL, caller falls back to GPU path
- **Dequant failure**: `ane_failed=1`, workers terminate, job freed
- **Thread spawn failure**: phantom decrements ensure `ane_done` fires
- **Slot exhaustion**: `acquire_dequant_slot` returns -1, call skipped
- **Sticky failure per layer**: `prewarm` failure persists for the session

---

## 12. Profiling & Telemetry

Stats counters (`g_ane_prefill_calls`, `g_ane_prefill_skip_big_refs`, etc.) track ANE utilization. Per-call timing via `ds4_gpu_now_ms()` captures:
- `dequant_ms` — GPU dequantization time
- `eval_ms` — ANE evaluation time (per worker: a/b/c/d)
- `output_f16_copy_ms` — fp16 output copy time
- `post_wait_ms` — post-processing wait time

---

## Summary

The ANE implementation is a sophisticated async pipeline that:
1. **Dequantizes** Q8_0/IQ2_XXS/Q2_K weights to int8 on the GPU
2. **Evaluates** fused MLP (gate+up+SwiGLU+down) on the ANE via compiled MIL models
3. **Packs** fp16 output to fp32 on GPU or CPU
4. **Parallelizes** across up to 4 software workers on 1-2 ANE clusters
5. **Overlaps** GPU dequant with ANE eval via async command buffers and pthread synchronization

The production path (mode 6, tiled-fused) generates a single MIL model with all 4 weight inputs and 1 output, eliminating the split-gate/up + down two-model approach of earlier modes.
