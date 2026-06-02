---
name: ane-kernel-tuning
description: Tune or add an ANE (Apple Neural Engine) MLP / attention kernel in ds4 — pick MIL format (matmul vs conv, int8 vs fp16), validate with a no-model microbench, measure wall time, and integrate into the hybrid GPU+ANE prefill pipeline. Use when optimizing ANE offload for shared expert, attention O-proj, routed MoE, or lm_head projections on M3 Ultra (dual-cluster) or M5 Max (single-cluster).
---

# Tuning ANE kernels in ds4

The ANE is a **batch matrix engine** — it wins on large-batch GEMMs where the GPU command-encode overhead would stall. The trick is **overlap**: ANE compute runs concurrently with GPU routed-expert work, and the wall time is the max of the two, not the sum.

## Hardware context

| Chip | ANE clusters | Peak aggregate TF/s (INT8) | Key trait |
|---|---|---|---|
| M3 Ultra | **2** (UltraFused dies) | ~19 TF/s (dual-cluster) | ANE offload gives +22.6% prefill t/s |
| M5 Max | **1** | ~9.5 TF/s (single-cluster) | GPU command-encode caps throughput; ANE helps ~+6 t/s |
| M4 Pro / base M5 | **1** | ~7–9 TF/s | Single-cluster, limited ANE benefit |

## Procedure

### 1. Build a no-model microbench probe

Pattern in `moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke.m`, `ane_ds4_mlp_int8w_dual_smoke.m`:
- Load MIL from CoreML export (`moe-batch-bench/coreml_exports/`) or compile inline
- Use private `ANEServices` framework (`_ANEInMemoryModel`, `_ANECreateInputs`)
- Run on synthetic IOSurface buffers
- **Validate against a CPU/GPU reference** (bit-exact for int8; ~f16 tolerance for fp16)
- Print aggregate TF/s and per-cluster wall
- ~1 s/config

### 2. Choose the MIL format

| Mode | Create entry | Eval entry | Use case |
|---|---|---|---|
| **1 (matmul split)** | `ds4_ane_mlp_fp16w_create` | `ds4_ane_mlp_fp16w_eval` | Production default. 3 evaluate calls + CPU silu/mul per call. |
| **8 (conv, W-as-input)** | `ds4_ane_mlp_fp16w_fused_conv_create` | `ds4_ane_mlp_fp16w_fused_conv_eval` | Off by default. Single-call fused conv2d-1x1, weights as IOSurface inputs. Slower than mode 1 on M3U. |
| **9 (conv, constexpr W)** | `ds4_ane_mlp_fp16w_constexpr_create` | `ds4_ane_mlp_fp16w_constexpr_eval` | Off by default. Single-call conv2d-1x1 with weights baked into MIL as `constexpr` BLOBFILE refs. 2.6× faster ANE eval but no e2e win (ANE hidden behind GPU). |

Toggle via `DS4_SHARED_EXPERT_ANE_CONV=1` (mode 9 if set, mode 1 otherwise).

### 3. Sweep the knobs

**Batch size** (`DS4_FLASH_MOE_ANE_BATCHES`):
- {64, 128, 256, 512} — larger batches improve ANE throughput but increase latency
- The per-call batch auto-shrinks across N workers; use `DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH=1` to pin it

**Worker threads** (`DS4_FLASH_MOE_ANE_THREADS`):
- M3U dual-cluster: N=2 gives ~2× aggregate (peak at N=4 with 2 workers per cluster)
- M5/M4 single-cluster: N=1 ≈ N=2 ≈ N=3 ≈ N=4 (no cluster parallelism)

**INT8 vs FP16**:
- INT8 (i8i8 tiled-fused): higher TOPS, but activation drift + per-block-scale issues
- FP16 (fp16w matmul): lower TOPS, simpler precision
- Decision: `DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1` for int8, default fp16

**Quantization** (`DS4_FLASH_MOE_MPP_INT8_QSCALE`, `DS4_FLASH_MOE_MPP_INT8_X_QSCALE`, `DS4_FLASH_MOE_MPP_INT8_MID_QSCALE`):
- QSCALE=512 (default), X_QSCALE=32, MID_QSCALE=32 for the shared expert int8 path

**Prefetch** (`DS4_FLASH_MOE_PREFETCH`):
- Default 3 — number of layers to prefetch ahead

**Hybrid threshold** (`DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS`):
- Routes refs below threshold to GPU. M3U sweep was noise-flat across 128–512.

### 4. Measure ANE wall correctly

ANE wall = `max(ane_eval_a, ane_eval_b, ane_eval_c, ane_eval_d)` — the aggregate time the ANE evaluator spent inside `predict` calls across all workers. Use the profile scripts:

```bash
for n in 1 2 3 4; do
  DS4_RUN_NAME=ane_wall_threads${n} \
    DS4_FLASH_MOE_ANE_THREADS=$n \
    DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH=1 \
    ./run_ane_prefill_profile_m3u.sh
done
```

### 5. Wire behind env flag

Default off until benched. Validate generation coherence (top-k/score drift is benign for rank-only paths), then run the same-binary A/B `ds4-bench` slope.

### 6. Validate on both single and dual cluster

Bench ANE on **both** M5 Max (single-cluster) and M3 Ultra (dual-cluster) before judging. A kernel that loses to NAX on M5 can be faster than NAX on M3U due to dual-cluster aggregate.

## Gotchas (cost real time here)

- **Dual-cluster requires separate ANE handles** — both contexts must be separate handles; the underlying `_ANEInMemoryModel` is a process-shared resource but per-context state (IOSurfaces + model handle) keeps clusters independent.
- **Sandbox must be off** for private ANE compile to write its temp file.
- **SLC/bandwidth pressure** — 86 in-flight constexpr contexts compete for system-level cache, causing ANE eval to be slower than standalone smokes predict (189 ms/layer in production vs 82 ms predicted).
- **Overlap bubbles are the hard part, not kernel TFLOPs** — making ANE 2.6× faster only buys "ANE finishes earlier and waits longer for GPU" if GPU is the bottleneck.
- **GPU command-encode caps throughput** — on single-cluster M5 Max, the wall is GPU command-encode (~95% CPU-side, caps ~277 t/s), so ANE only helps by shedding GPU work.
- **The ANE conv path (mode 9) does not win for prefill** — the ANE eval is hidden behind GPU + pread (pread alone is ~310 ms/layer; ANE eval is ~94 ms in mode 9 or ~244 ms in mode 1 — both << the pread wall).
- **Reference for tuned kernels**: `ds4_metal.m` (`g_shared_ane_dep_wait_ms`, `g_shared_ane_join_wait_ms`) for wait-time probes; `moe-batch-bench/ane_ds4_mlp_int8w.m` for the microbench suite.

## Related docs

- `moe-batch-bench/ANE_INT8_COMBINED_BENCH_PROCEDURE.md` — full combined benchmark procedure
- `moe-batch-bench/ANE_WALL_MEASUREMENT.md` — how to measure ANE wall across chips
- `moe-batch-bench/DUAL_ANE_CLUSTER_OPTIMIZATION.md` — dual-cluster ANE on M3 Ultra
- `moe-batch-bench/OPROJ_ANE_INVESTIGATION.md` — attention O-proj on ANE
- `moe-batch-bench/FUSED_MLP_ANE_CONV.md` — fused MLP ANE conv2d-1x1 path
- `moe-batch-bench/SHORT_PREFILL_OPTIMIZATION.md` — short-prefill optimization (SSD-bound context)
- `NAX_ANE_PREFILL_SCOPE.md` — layer-wise NAX vs ANE decision framework
- `moe-batch-bench/RUN_ON_M5.md` — running ANE tests on new chips
