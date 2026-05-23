# Fused MLP ANE conv2d-1x1 path (future investigation)

Two ANE MIL variants for the shared-expert fp16w MLP, both kept in tree behind
env flags but **off by default** because they don't win for the current
DSv4 IQ2_XXS prefill profile on M3U. Worth revisiting if/when ANE becomes the
main-thread critical path.

## What's implemented

| Mode | Create entry | Eval entry | Status |
|------|--------------|-----------|--------|
| 1 | `ds4_ane_mlp_fp16w_create` | `ds4_ane_mlp_fp16w_eval` | Production (matmul, 3 evaluate calls + CPU silu/mul per call) |
| 8 | `ds4_ane_mlp_fp16w_fused_conv_create` | `ds4_ane_mlp_fp16w_fused_conv_eval` | Off by default. Single-call fused conv2d-1x1 with weights as **IOSurface inputs**. Slower than mode 1. |
| 9 | `ds4_ane_mlp_fp16w_constexpr_create` | `ds4_ane_mlp_fp16w_constexpr_eval` | Off by default. Single-call fused conv2d-1x1 with weights baked into MIL as **`constexpr` BLOBFILE refs**. 2.6× faster on the ANE eval itself, but no end-to-end win. |

Toggled via `DS4_SHARED_EXPERT_ANE_CONV=1` (mode 9 if set, mode 1 otherwise).
Mode 8 has no production wiring — it's reachable only by direct API call.

## Results on M3U (mean of 3 trials, DS4_TOKENS=10, 8423-token prompt)

| Path | Prefill | Per-layer ANE eval | Total ANE work | Notes |
|------|---------|--------------------|-----------------|-------|
| Mode 1 (matmul split, prod default) | **286.30 t/s** | 244.4 ms | 10.45 s | Baseline |
| Mode 8 (conv, W-as-input) | 251.92 t/s | — | — | MIL transposes likely demoted off the optimized conv path |
| Mode 9 (conv, constexpr W) + warm | 281.48 t/s | **93.85 ms (2.6× faster)** | 4.03 s | ANE faster, prefill ~5 t/s slower |

## Why mode 9 doesn't win for prefill

The wait-time probes added in `ds4_metal.m` (`g_shared_ane_dep_wait_ms`,
`g_shared_ane_join_wait_ms`) make the answer concrete:

- `join_wait_ms` (main thread blocked in `pthread_join` waiting for ANE worker
  to finish) is **~0.2 ms total across the whole prefill** in *both* modes.
- The main encode loop never waits on ANE. ANE is fully hidden behind
  GPU + pread (pread alone is ~310 ms/layer; ANE eval is ~94 ms in mode 9 or
  ~244 ms in mode 1 — both <<).
- Making ANE 2.6× faster only buys "ANE finishes earlier and waits longer for
  GPU." It doesn't shorten the wall.

The original GPU→ANE shared-expert move *did* win (~+6 t/s) because it shed
GPU work, not because ANE was fast. Matmul vs constexpr conv on ANE produces
the same GPU savings, so the choice doesn't move prefill at this workload.

The residual ~5 t/s mode-9 regression is outside the shared-expert ANE path
itself (the probes prove that). Likely SLC/memory-bandwidth contention from
holding ~2 GB of compiled constexpr ANE programs while GPU runs, or driver
overhead from juggling 43 contexts.

## When mode 9 *would* pay off

- **Resident-mode MoE** (slot-bank fully populated, no pread on the critical
  path). With pread gone, ANE eval enters the wall budget and the 6.4 s of
  ANE work saved by conv would translate to real prefill t/s.
- **Smaller models / fewer layers** where ANE program memory pressure is low
  enough not to disturb GPU.
- **Configs where ANE is the bottleneck** for any reason (e.g., much faster
  SSDs that drop pread below ANE eval time).
- **Power / thermal-bound** scenarios where reducing total ANE work matters
  even when wall time doesn't change.

## Mode 9 architecture (canonical ANE pattern)

Per Apple's own ANE LLM exports (`../ane/INT4_MATMUL_ANE_WORKFLOW.md`,
section 4):

- **Weights as `constexpr` fp16 in MIL via `BLOBFILE` refs.** No per-call
  weight upload, no in-MIL transposes.
- **Weight blob format** (CoreML `MILBlob/Blob/StorageFormat.hpp`):
  64-byte `storage_header` (count, version=2) + per-blob 64-byte
  `blob_metadata` (sentinel `0xDEADBEEF`, dtype, sizeInBytes, offset) +
  raw 64-aligned data. `BLOBFILE(offset = uint64(N))` references the
  metadata; the reader follows `metadata.offset` to the raw data.
- **Both in-memory and on-disk weight channels are required**:
  - The `weights:` dict passed to
    `[_ANEInMemoryModelDescriptor modelWithMILText:weights:options:]` keyed
    by the full `@model_path/...` string with value `{offset, data}`. This
    affects the descriptor's `hexStringIdentifier` (content hash).
  - The same bytes written to disk at
    `NSTemporaryDirectory()/<hexId>/weights/weight.bin` so the post-descriptor
    compile pass can read them.
- **Per-layer compiled ctx** (43 for DSv4). One compile per layer at prewarm
  (~220 ms each, ~9.5 s total — paid before the prefill timer). Shape-shared
  ctx pooling is not possible because weights are baked in.
- **MIL skeleton**: only X is a function input; gate/up/down weights are
  `const(... BLOBFILE)` constants. Activation reshape to NCHW
  `[1, H, 1, B]`, three `conv` ops with `[O, I, 1, 1]` weights, `silu`+`mul`
  for SiLU-gated activation, output reshape back to `[B, H]`.

## Files

- `moe-batch-bench/ane_ds4_mlp_int8w.{h,m}` — modes 8 / 9 MIL generators,
  blob writer, create/eval entry points.
- `moe-batch-bench/ane_ds4_mlp_constexpr_smoke.m` — standalone harness that
  drives the mode-9 path with synthetic weights; confirms compile/load/eval
  works and reports per-call ms / TFLOP/s without needing the full ds4 binary
  or external model. At H=4096 I=2048 B=256: 197 ms create, 2.17 ms/eval,
  5.93 TFLOP/s.
- `ds4_metal.m` — `ds4_shared_expert_ensure` branches on
  `DS4_SHARED_EXPERT_ANE_CONV` to pick mode 1 or mode 9; worker eval call
  dispatches on `ds4_ane_mlp_int8w_mode(ctx)`.
- `ds4_metal.m` — wait-time probes (`g_shared_ane_dep_wait_ms`,
  `g_shared_ane_join_wait_ms`, `g_shared_ane_warm_ms`) printed in the prefill
  summary line `ds4: ANE shared-expert calls=...`.

## Related references

- `../ane/INT4_MATMUL_ANE_WORKFLOW.md` — the prior-art doc with the canonical
  pattern, the `weights:` dict format, and the table of what works vs what
  doesn't on `_ANEInMemoryModel`.
- `../ane/test_conv_w_input.m` — earlier probe of conv-with-W-as-input
  (the analogue of mode 8). Compile fails for shapes exceeding the ANE
  compiler's per-tensor limits (`N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384]`).
