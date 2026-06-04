# Apple Neural Engine Kernels

ANE support in DS4 is experimental. This file documents only the Apple Neural
Engine code paths implemented in `ds4_ane_mlp_int8w.{m,h}` and used from
`ds4_metal.m`. It is not a backend-routing guide.

The ANE path uses Apple's private, undocumented ANE interfaces
(`_ANEInMemoryModel`, `_ANERequest`, IOSurface-backed inputs/outputs). The
runtime profiles enable ANE only for prefill shapes where ANE has measured
favorably on that machine. `--quality` implies `--no-int8`, which currently
disables the production ANE prefill routes because they are int8-backed.

## Routed MLP Kernels

These kernels implement the DeepSeek routed expert MLP shape:
`gate(x)`, `up(x)`, `silu(gate) * up`, then `down(...)`.

- **mode 0: int8w routed MLP**
  `ds4_ane_mlp_int8w_create` / `ds4_ane_mlp_int8w_eval`.
  Original int8-weight route. Takes int8 gate/up/down weights, int8 input, an
  fp16 route vector, and returns fp16 output.

- **mode 1: fp16w split MLP**
  `ds4_ane_mlp_fp16w_create` / `ds4_ane_mlp_fp16w_eval`.
  Baseline fp16-weight MLP. It is simple and useful for validation, but it
  requires multiple ANE evaluate calls and host/runtime work between stages.

- **mode 2: i8w-fp16x MLP**
  `ds4_ane_mlp_i8w_fp16x_create` / `ds4_ane_mlp_i8w_fp16x_eval`.
  Int8 weights with fp16 activations. This reduces weight traffic while avoiding
  int8 activation quantization.

- **mode 3: i8w-i8x MLP**
  `ds4_ane_mlp_i8w_i8x_create` / `ds4_ane_mlp_i8w_i8x_eval`.
  Int8 weights and int8 activations. Higher throughput target, but activation
  quantization makes it a quality-sensitive path.

- **mode 4: i8w-i8x full-fused MLP**
  `ds4_ane_mlp_i8w_i8x_fused_create` /
  `ds4_ane_mlp_i8w_i8x_fused_eval`.
  Fuses more of the MLP into a single ANE graph/evaluate sequence. Kept as an
  experimental fused variant.

- **mode 5: i8w-i8x gate/up fused MLP**
  `ds4_ane_mlp_i8w_i8x_gateup_fused_create` /
  `ds4_ane_mlp_i8w_i8x_gateup_fused_eval`.
  Fuses the gate and up work while leaving the remaining stages separate.

- **mode 6: i8w-i8x tiled-fused MLP**
  `ds4_ane_mlp_i8w_i8x_tiled_fused_create` /
  `ds4_ane_mlp_i8w_i8x_tiled_fused_eval`.
  Current production-oriented routed MLP kernel family. It batches expert work
  into ANE-friendly tiles and is the path used by the prefill scheduler when
  ANE is enabled.

- **mode 7: i8w-i8x tiled-fused with int8 output**
  `ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_create` /
  `ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_eval`.
  Same tiled-fused shape as mode 6, but writes int8 output instead of fp16.
  This is an experiment for reducing output bandwidth.

- **mode 13: i8w-i8x tiled-fused routed MLP**
  `ds4_ane_mlp_i8w_i8x_tiled_fused_routed_create` /
  `ds4_ane_mlp_i8w_i8x_tiled_fused_routed_eval`.
  Tiled-fused int8 route with explicit fp16 routing weights supplied to the ANE
  eval path.

## FP16 Conv Kernels

These kernels lower MLP work into ANE-friendly 1x1-convolution patterns.

- **mode 8: fp16w fused conv MLP**
  `ds4_ane_mlp_fp16w_fused_conv_create` /
  `ds4_ane_mlp_fp16w_fused_conv_eval`.
  Single-call fused fp16-weight MLP with weights supplied as inputs. Useful for
  comparing ANE-native conv lowering against the split matmul-style path.

- **mode 9: fp16w constexpr conv MLP**
  `ds4_ane_mlp_fp16w_constexpr_create` /
  `ds4_ane_mlp_fp16w_constexpr_eval`.
  Fused fp16-weight MLP with weights baked into the compiled MIL as side-loaded
  blob data. Per-call input/output traffic is lower because weights are not
  uploaded for every eval.

## Linear Projection Kernels

These kernels implement two-stage linear projections without SwiGLU. They are
used for projection experiments such as attention output projection.

- **mode 10: fp16w linear constexpr**
  `ds4_ane_mlp_fp16w_linear_constexpr_create` /
  `ds4_ane_mlp_fp16w_linear_constexpr_eval`.
  Two fp16 linear layers with weights baked into MIL. Input and output are fp16.

- **mode 1 linear eval helper**
  `ds4_ane_mlp_fp16w_linear_eval`.
  Reuses a mode-1 fp16 split context for two-matmul linear projection. Weights
  are supplied per eval.

- **mode 2 linear eval helper**
  `ds4_ane_mlp_i8w_fp16x_linear_create` /
  `ds4_ane_mlp_i8w_fp16x_linear_eval`.
  Int8-weight, fp16-activation linear projection with separate scales for the
  two linear weights.

- **mode 11: int8w linear constexpr**
  `ds4_ane_mlp_int8w_linear_constexpr_create` /
  `ds4_ane_mlp_int8w_linear_constexpr_eval`.
  Int8 weights and per-output-channel scale data are baked into the MIL graph.
  Input and output are fp16.

- **mode 12: int8w-i8x linear constexpr**
  `ds4_ane_mlp_int8w_i8x_linear_constexpr_create` /
  `ds4_ane_mlp_int8w_i8x_linear_constexpr_eval`.
  Same constexpr int8-weight linear path as mode 11, but input is supplied as
  int8 and dequantized inside the ANE graph. Output remains fp16.

## Dispatch Helpers

These are not separate mathematical kernels, but they matter for measuring and
integrating ANE:

- `ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface` evaluates a tiled-fused
  MLP and leaves output in the ANE/IOSurface path.
- `ds4_ane_mlp_i8w_i8x_tiled_fused_eval_xonly` skips weight IOSurface writes
  after a previous call has already populated the weights. It is useful for
  measuring input-only dispatch overhead.
- `ds4_ane_mlp_int8w_linear_constexpr_attach_chunks` and
  `ds4_ane_mlp_int8w_linear_constexpr_eval_at_chunk` bind multiple external
  input IOSurfaces to a constexpr linear context so chunked eval can avoid
  per-call input surface rewrites.

## Current Release Use

The release ANE prefill path is the i8w-i8x tiled-fused routed MLP family, with
worker/thread policy controlled by runtime profile defaults and explicit ANE
environment knobs. Dual-cluster M3 Ultra can use multiple ANE contexts in
parallel; single-cluster chips use a smaller worker count. Shared expert and
linear projection ANE paths remain experimental.
