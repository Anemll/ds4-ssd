---
name: nax-kernel-tuning
description: Tune or add an M5 NAX (MetalPerformancePrimitives matmul2d / tensor_ops) kernel in ds4 — pick tile/dtype/post-proc knobs, validate in a no-model microbench, and autotune. Use when optimizing a Metal matmul2d kernel (indexer scores, dense Q8_0 projections, MoE), porting tensor_matmul, or chasing prefill throughput on M5/M6.
---

# Tuning M5 NAX matmul2d kernels in ds4

The `matmul2d` op is the easy part; the **wrapper** (tiling, staging, post-processing) decides speed —
a naive vs tuned version of the *same* op differed 5.8x here. Work the knobs, not the op.

## Procedure
1. **Build/keep a microbench probe** (no 86 GB model). Pattern in `moe-batch-bench/nax_*_probe.m`,
   `nax_indexer_test.m`: load inline Metal with `MTLLanguageVersion4_0`, run the kernel on synthetic
   buffers, **validate against a CPU reference** (bit-exact for int8; ~f16 tolerance for half), print GF/s.
   This is the autotuner's cost function; ~1 s/config.
2. **Sweep the knobs** (see `memory/nax-kernel-tuning-playbook.md` for directions/evidence):
   - token tile `TM` {16,32,64} — smaller often wins (occupancy); indexer won at 16.
   - K-tile `NK` {32,64,128,256} — smaller pipelines staging vs compute.
   - post-proc: `cooperative_tensor.store(threadgroup)` + flat loop — **never** call
     `get_multidimensional_index` in a hot loop.
   - dtype: `half×half→float` (antirez's choice everywhere; dequant weights to half) vs `int8×int8→int32`
     (higher TOPS, but activation drift + Q8_0 per-block-scale issues).
   - `transpose_left/right` to match natural memory layout; `mode` multiply vs multiply_accumulate.
   - `relaxed_precision` (descriptor arg) for rank-only kernels (e.g. indexer).
   - **tile walk order**: regular vs Morton/Z-order tgid→tile remap for L2 locality.
   - keep operand A device-contiguous (threadgroup-staged gather was ~9x slower).
3. **Autotune** with `moe-batch-bench/nax_autotune.m`: it templates the kernel source over the knobs,
   compiles+validates+times each, prints the ranked table, and emits the winning config.
4. **Gate by problem size.** NAX has fixed per-dispatch overhead; it wins only above a crossover
   (indexer: +4..+8.5% at ctx≥16k, −8% at 8k). Add an `n_comp`/`n_tokens` threshold env gate.
5. **Wire** behind an env flag (default off until benched), validate generation coherence (top-k/score
   drift is benign for rank-only paths), then run the same-binary A/B `ds4-bench` slope.

## Gotchas (cost real time here)
- threadgroup tile element type must be `int8_t`/`half`, not `char`.
- A operand must be non-`const` (cooperative tensor element-type match).
- multiply_accumulate is required to accumulate across manual K-tiles (multiply overwrites).
- The fused NAX library is a separate `MTLLanguageVersion4_0` lib (`metal/nax_fused.metal`); the main
  ds4 library is default-options, so NAX kernels can't live there in this fork.
- Reference for tuned kernels: `../ds4` (antirez) `metal/{dense,moe,dsv4_misc}.metal` (`#ifdef DS4_METAL_HAS_TENSOR`).
