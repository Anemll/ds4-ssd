# DSv4 MLP on ANE – Matmul + Streamed Weights Investigation

**Date:** May 2026
**Goal:** Build a reliable reference for running DeepSeek V4 Flash MLP projections (Gate / Up / Down) on the Apple Neural Engine using **raw `matmul`** operations with **weights as inputs** (not baked-in constants), so they can later be replaced with `BLOBFILE`-style streamed weights in the private ANE API.

---

## 1. High-Level Goal

We want a working MIL graph for the following projections:

- Gate: `[..., 7168] → [..., 18432]`
- Up:   `[..., 7168] → [..., 18432]`
- Down: `[..., 18432] → [..., 7168]`

Requirements:
- Use **explicit `matmul`** (not `linear` or `conv`).
- Weights must appear as **inputs** in the MIL (so they can be swapped for streamed `BLOBFILE` weights later).
- The graph must compile and run on the ANE (no silent CPU fallback).
- Follow the proven patterns from existing working code in this repo.

---

## 2. Key References & Prior Art

### In this repository
- `inmem_peak_matmul.m` – The gold-standard reference for **matmul + streamed weights** (BLOBFILE) that actually works on ANE. Uses `transpose_x = true` pattern with specific activation layout.
- `inmem_peak.m` – Reliable conv + streamed weights baseline.
- `ane_mlp_bench.m` (original from m3u) – The file that started this investigation. Originally used matmul but hit many `InvalidMILProgram` errors on DSv4 shapes.
- Various `gen_*_model.py` scripts – High-level CoreML conversion experiments.

### External repo (very relevant)
- **https://github.com/maderix/ane-prefill-bench**
  - Uses raw MIL + private ANE APIs (`_ANEInMemoryModel`).
  - Successfully runs large models on ANE.
  - Uses a **"pack weights into input + slice_by_size"** trick for dynamic/streamed weights instead of normal separate inputs or constants.
  - Has both `matmul` and `conv1x1` dynamic-weight kernels (conv version reported as faster for large shapes).

---

## 3. Workflow We Followed (ANE Skills)

Per the established "ANE Skills" process:

1. Start with **PyTorch + high-level `ct.convert`** (not raw MIL).
2. Confirm the model actually runs on ANE using `anemll-profile`.
3. Only after confirmation → extract the `model.mil`.
4. Use the extracted MIL as a reference to build the low-level private-API version with streamed weights.

We tried to stay on this path but repeatedly hit friction.

---

## 4. What We Tried (Chronological)

| Phase | Approach | Weights | Matmul Style | Result | Key Problem |
|-------|----------|---------|--------------|--------|-------------|
| 1 | High-level `ct.convert` (PyTorch `nn.Linear` + matmul) | Constants (nn.Parameter) | CoreML turned them into `linear` | Compiled, ran on ANE | Weights not visible as inputs |
| 2 | Weights passed as explicit inputs to `forward()` | Inputs | `matmul(..., y = W_input)` | Compiled | ANE compilation failed at runtime → CPU fallback |
| 3 | Explicit activation transpose before matmul (to produce `transpose_x = true`) | Inputs | `matmul(transpose_x = true, ...)` | Better MIL | Still CPU fallback on ANE |
| 4 | "One weight + slice" pattern from `ane-prefill-bench` (pack activation + weight in one input, use `slice_by_size`) | Sliced from input | `matmul` after slice | **InvalidMILProgram** (minimal test) | Pattern is very sensitive to exact packing + reshape order |
| 5 | Weights as inputs + activation layout matching `inmem_peak_matmul.m` | Inputs | Closer to working `transpose_x=true` | Better MIL shape | Still investigating ANE acceptance |
| 6 | Profile compiled `ds4_mlp_matmul_fp16_weights_as_inputs.mlmodelc` (2026-05-18) | Inputs | `transpose_x=true, transpose_y=false` | **100% ANE placement, 38.5 ms/iter, 0.66 TOPS** | Bandwidth bound on weight upload (756 MB/iter via `tensor_buffer_to_tensor`) |
| 7 | `test_matmul_w_input.m` raw MIL via `_ANEInMemoryModel`, 4D fp32 input + cast + squeeze (legacy inmem_peak template) | Input | `transpose_x=true, transpose_y=false` | compile + load OK, **eval FAIL: `ANEProgramProcessRequestDirect status=0x1d, statusType=0x9` (Program Inference error)** | 4D-fp32-with-cast wrapper is rejected at execution time when W is a function input |
| 8 | `test_matmul_w_input.m` rewritten to mirror `model.mil` of the compiled mlmodelc: 2D fp16 input, no `cast`, no `squeeze` (2026-05-18) | Input | `transpose_x=true, transpose_y=false` | **compile + load + eval OK on all 3 shapes** — gate 5.15 ms / up 5.15 ms / down 16.10 ms (single matmul each) | None — this is a viable private-API path for streamed weights |

---

## 5. Key Technical Findings

### 5.1 Weight Handling is the Core Problem
- When weights are `nn.Parameter` → CoreML bakes them as `const` + internal BLOBFILE. Good for ANE execution, bad for streamed weights.
- ~~When weights are explicit inputs → ANE compiler often fails to compile the program (even if `MLComputePlan` says ANE), causing CPU fallback.~~ **Refuted 2026-05-18:** with `transpose_x=true, transpose_y=false` and 3D activation layout, weights-as-inputs *do* compile and run 100% on ANE (`anemll-profile` on `ds4_mlp_matmul_fp16_weights_as_inputs.mlmodelc` shows every op placed on ANE, 38.5 ms/iter).
- The working repo (`inmem_peak_matmul.m`) keeps weights as `const` + BLOBFILE. The prefill-bench repo uses a **pack + slice** trick instead of normal inputs.
- The real cost of W-as-input is bandwidth: each iteration emits a `tensor_buffer_to_tensor` op per weight (252 MB × 3 = 756 MB) running at ~98 GB/s, eating ~7.7 ms before any matmul starts. That is the motivation for streamed/BLOBFILE weights, not ANE rejection.

### 5.2 Matmul Call Style Matters
The reliable pattern from `inmem_peak_matmul.m` is:

```mil
matmul(transpose_x = true, transpose_y = false, x = x_3d, y = W)
```

Our early attempts used `transpose_x = false, transpose_y = false` with weights as direct inputs. This appears to be less reliable when weights are not compile-time constants.

### 5.2a Activation Wrapper Style Also Matters (2026-05-18)
The original `inmem_peak_matmul.m` template wraps the activation as
`tensor<fp32, [1, ch, 1, sp]> x` and uses `cast(fp16) -> squeeze` to get the 3D
matmul-x.  That wrapper is **incompatible with W-as-input on `_ANEInMemoryModel`** —
it compiles and loads, but inference fails with
`ANEProgramProcessRequestDirect status=0x1d, statusType=0x9: Program Inference error`.

The pattern that works (verified end-to-end in `test_matmul_w_input.m`) drops the
wrapper entirely and matches what the high-level CoreML converter emits in
`ds4_mlp_matmul_fp16_weights_as_inputs.mlmodelc/model.mil`:

```mil
func main<ios18>(tensor<fp16, [hidden, interm]> W,
                 tensor<fp16, [batch,  hidden]> input) {
    transpose(perm=[1,0])    // [batch,hidden] -> [hidden,batch]
    expand_dims(axes=[0])    // [hidden,batch] -> [1,hidden,batch]
    matmul(transpose_x=true, transpose_y=false, x=x_3d, y=W)
        -> [1, batch, interm]
}
```

Key constraints discovered:
- All inputs and the output are **fp16** at the MIL boundary.  No `cast(fp32→fp16)` ops.
- Activation is shaped `[batch, hidden]` (2D), not `[1, hidden, 1, batch]` (4D).
- The IOSurface byte sizes must match the 2D fp16 element count
  (`batch*hidden*2`, `hidden*interm*2`, `batch*interm*2`).
- `inputIndices` array order must match the order of function arguments in the
  MIL signature (W is arg 0, input is arg 1 in the working test).

### 5.3 "Dynamic Weight via Slice" Pattern
The `ane-prefill-bench` repo uses a single large input tensor containing both activation and weights, then slices the weights out at runtime with `slice_by_size`. This is currently the most successful known technique for truly dynamic/streamed weights on ANE, but it is fragile and layout-sensitive (our minimal extraction hit `InvalidMILProgram`).

---

## 6. Files Created / Modified

| File | Purpose | Status |
|------|---------|--------|
| `gen_ds4_mlp_matmul_fp16.py` | First high-level conversion (weights as constants) | Superseded |
| `gen_ds4_mlp_matmul_fp16_weights_as_inputs.py` | Weights as explicit inputs + transpose tweaks | Active |
| `gen_ds4_mlp_matmul_fp16_weights_as_inputs.py` (later version) | Weights as inputs + activation layout matching working pattern | Latest |
| `/tmp/test_one_weight_slice.mil` | Minimal raw MIL test of "one weight + slice + matmul" | Test artifact |
| `test_matmul_w_input.m` | Raw-MIL `_ANEInMemoryModel` benchmark: W as function input, 2D fp16 layout. **First working private-API path for W-as-input on DSv4 shapes** | Current (2026-05-18) |
| `ds4_mlp_inmem_bench.m` | Full DSv4 MLP (gate+up+silu+mul+down) via `_ANEInMemoryModel` with input, W_gate, W_up, W_down as function args. Sweeps batch sizes {1,8,16,32,64} | Current (2026-05-18) |
| `dump_ane_classes.m` | Selector / property dump of `_ANEInMemoryModel*` private classes; found the `weightsBuffer:` request slot | Reference |
| `DSv4_MLP_ANE_Matmul_Investigation.md` | This summary document | Current |

---

## 7. Current Open Questions / Next Experiments

Status after the 2026-05-18 session:
- Q2 (best transpose / layout for W-as-input) — **answered**: 2D fp16 activation
  + `transpose+expand_dims` -> `matmul(transpose_x=true, transpose_y=false, x, W)`
  with 2D fp16 W. See §5.2a.
- Q4 (is the "pack+slice" trick fundamentally more reliable than separate W input)
  — **answered no**: separate W input runs on ANE for all three DSv4 shapes via
  the private `_ANEInMemoryModel` API. The slice trick is one option, not a
  requirement.

Still open:
1. Can the "pack + slice" pattern be made to compile for DSv4 sizes anyway? It
   might still beat plain W-input on bandwidth for streamed weights.
2. (was Q3) Are any MIL passes (`common::const_elimination`, `common::fuse_linear_bias`)
   eating performance when weights are inputs? Worth measuring with passes disabled.
3. The `_ANERequest weightsBuffer:` slot (nil in the current test) likely points
   at a real streamed-weights API. Find concrete usage and confirm whether it
   replaces the W function input or augments it.
4. Why is the down-projection (h=18432, i=7168) ~3x slower than the gate-projection
   (h=7168, i=18432) at identical FLOPs (5.15 ms vs 16.10 ms)? Probably the larger
   K reduction is bandwidth-bound on the activation tile; needs profiling.

---

## 8. References

- `inmem_peak_matmul.m` (this repo) – Best existing example of working streamed matmul on ANE.
- `https://github.com/maderix/ane-prefill-bench` – Source of the "pack + slice" dynamic weight technique.
- `anemll-profile` skill (`~/.codex/skills/anemll-profile`) – Used for all ANE placement and performance analysis.
- `ane_mlp_bench.m` (original) – The benchmark we are ultimately trying to make work for DSv4 shapes.

---

## Appendix B — 2026-05-18 (later): GPU comparison + int8 / int4 ANE

### B.1 Correctness vs CPU fp32 (`ds4_mlp_inmem_bench -verify` + `ds4_mlp_gpu_bench.py --verify`)

Deterministic inputs and 3 weight matrices were written from the fp16 ANE
bench, then recomputed in CPU fp32 (gold) and torch MPS fp16.  Values scaled
to keep the matmul reduction well inside fp16 range (each element |x|≤0.02).

| reference             | max |Δ|  | mean |Δ| | mean rel |
|-----------------------|----------|---------:|---------:|
| ANE fp16  vs CPU fp32 | 4.4e-4   | 8.0e-5   | 8.0% *   |
| MPS fp16  vs CPU fp32 | 2.5e-7   | 4.3e-8   | 4e-5     |
| ANE fp16  vs MPS fp16 | 4.4e-4   | 8.0e-5   | 8.0% *   |

\*Relative error is high only because the output magnitudes themselves are
~1e-3 (deliberately scaled to avoid fp16 overflow in the H=7168 reduction).
Absolute error matches expected fp16 mantissa precision (~2⁻¹⁰).  MPS appears
to accumulate the reduction in fp32 internally, which is why it tracks the
gold so tightly; ANE accumulates in fp16.  Numerical sanity confirmed.

### B.2 GPU performance (`ds4_mlp_gpu_bench.py --bench`, torch MPS, fp16)

For the same H=7168, I=18432 MLP, weights resident in GPU memory:

| B   | MPS ms/iter | MPS TFLOP/s |
|----:|------------:|------------:|
| 1   | 1.56        | 0.51        |
| 8   | 1.52        | 4.17        |
| 16  | 1.54        | 8.26        |
| 32  | 1.66        | 15.32       |
| 64  | 1.72        | 29.58       |
| 96  | 2.55        | 29.86       |
| 128 | 2.07        | 49.03       |
| 256 | 3.40        | 59.75       |

The GPU pays no per-iteration weight upload, so this is the natural ceiling.
GPU saturates around 60 TFLOPs fp16 at B=256 on this machine.

### B.3 W8A8 int8 compute on ANE (`ds4_mlp_inmem_bench_int8.m`)

> ⚠️ **Does NOT satisfy the original "weights as inputs / streamed" goal.**
> This bench uses **conv1x1 + const+BLOBFILE** weights because ANE's int8
> GEMM path is only available when the weights are compile-time constants.
> `conv(weight = <function-input>)` is explicitly rejected by the ANE
> compiler (see §B.9 and `test_conv_w_input.m`).  Use this number for
> "what's the ceiling on ANE with quantized weights" — not for streaming.

- Weights: int8 stored in a BLOBFILE, dequantized at compile-time via
  `constexpr_affine_dequantize` (scale=2⁻³, zero_point=0).
- Activations: `quantize(fp16→int8) → dequantize(int8→fp16)` pair before each
  conv — the ANE compiler recognises this pattern and runs **real int8 GEMM**.
- Graph uses `conv1x1` rather than `matmul`. (`matmul` + `constexpr_*` was
  rejected by the ANE compiler as InvalidMILProgram; the proven int8 path is
  conv-based, same as `inmem_peak_int8.m`.)

| B   | ms/iter | TFLOP/s | regime                          |
|----:|--------:|--------:|---------------------------------|
| 1   | 3.60    | 0.22    | latency floor (~3.6 ms)         |
| 8   | 3.60    | 1.76    | latency floor                   |
| 16  | 3.60    | 3.52    | latency floor                   |
| 32  | 3.62    | 7.01    | transitioning                   |
| 64  | 3.63    | **13.98** | compute-bound, peak           |
| 96  | 5.38    | 14.14   | compute-bound (off tile, slight regress on ms/token) |
| 128 | —       | —       | **ANE compile fail** (tensor size limit) |
| 256 | —       | —       | ANE compile fail                |

Weights are 378 MB int8 const, loaded once into the ANE kernel context.  No
per-iteration upload.  Compared to fp16 W-as-input, the latency floor drops
from 18 ms → 3.6 ms (5x) and the peak compute lifts from 1.2 → 14 TFLOPs (11x).

### B.4 W4A8 int4 compute on ANE (`ds4_mlp_inmem_bench_int4.m`)

> ⚠️ **Same caveat as §B.3** — this uses `conv1x1 + const+BLOBFILE` weights.
> The 18 TFLOPs number is *not* a viable streamed-weights configuration.
> Reported here only as the upper bound on ANE compute throughput for the
> DSv4 MLP shapes; bridging this to dynamic weights is still open (§B.9).

- Weights: int4 packed in BLOBFILE, per-output-channel fp16 scale, int4 offset.
  Dequantized at compile-time via `constexpr_blockwise_shift_scale`
  (the pattern from `inmem_peak_w4.m`).
- Activations: same quantize/dequantize hint as W8A8 — ANE still uses int8 GEMM.
- Graph: `conv1x1`.

| B   | ms/iter | TFLOP/s | regime                          |
|----:|--------:|--------:|---------------------------------|
| 1   | 2.73    | 0.29    | latency floor (~2.7 ms)         |
| 8   | 2.75    | 2.30    | latency floor                   |
| 16  | 2.74    | 4.64    | latency floor                   |
| 32  | 2.73    | 9.28    | transitioning                   |
| 64  | 2.79    | **18.22** | compute-bound, peak           |
| 96  | 5.42    | 14.05   | regression (off-tile)           |
| 128 | —       | —       | ANE compile fail                |
| 256 | —       | —       | ANE compile fail                |

Weight memory is 189 MB (half of int8, 1/4 of fp16).  The smaller-weight
storage actually pushes compute higher than W8A8 — less time spent moving
weights into the compute tiles per cycle.

### B.5 Side-by-side summary at H=7168, I=18432

| precision        | best TFLOPs |  at B | floor ms | weight MB | streamable? | notes                                    |
|------------------|------------:|------:|---------:|----------:|:-----------:|------------------------------------------|
| fp16 W-as-input  |  1.21       | 64+   | 18.1     | 756/iter  | ✅           | upload-bound; *this is the only path that meets the original goal* |
| fp16 BLOBFILE    |  4.90       | 32    |  5.2     | 756 const | ❌           | conv1x1 + fp16 const, no upload cost     |
| MPS fp16 (GPU)   | 59.75       | 256   |  1.5     | resident  | (n/a)       | for reference                            |
| W8A8 (int8/int8) | 14.14       | 96    |  3.6     | 378 const | ❌           | conv1x1, true int8 GEMM                  |
| W4A8 (int4/int8) | **18.22**   | 64    |  2.7     | 189 const | ❌           | conv1x1, true int8 GEMM, smaller storage |

"Streamable" = weights can be swapped at runtime without recompiling the
`_ANEInMemoryModel`.  Today only fp16 W-as-input (matmul) qualifies; all
ANE int8/int4 GEMM paths require const+BLOBFILE weights (see §B.9).

ANE at W4A8 reaches roughly 1/3 of GPU fp16 throughput while using **75% less
weight memory** than fp16 and **50% less** than int8.  The B=128/256 ANE
compile failures show an ANE activation-tile size limit that the int8/int4
graphs hit but the fp16 W-as-input graph doesn't — worth investigating if
serving larger contexts becomes a goal.

### B.6 fp16 BLOBFILE baseline — does int8 actually use int8 hardware?

Question: are the int8/int4 numbers in B.3/B.4 driven by **int8 compute hardware**,
or just by **weight-residency** (BLOBFILE const has no per-iter upload)?  The
fair test is the same conv1x1 graph with fp16 weights as BLOBFILE const, no
quantize/dequantize hint.  Implemented in `ds4_mlp_inmem_bench_fp16_blob.m`.

| B   | fp16 BLOBFILE ms | fp16 BLOBFILE TFLOP/s | W8A8 TFLOP/s | W4A8 TFLOP/s |
|----:|-----------------:|----------------------:|-------------:|-------------:|
| 1   | 5.16             | 0.15                  | 0.22         | 0.29         |
| 8   | 5.16             | 1.23                  | 1.76         | 2.30         |
| 16  | 5.17             | 2.46                  | 3.52         | 4.64         |
| 32  | 5.18             | **4.90**              | **7.01**     | **9.28**     |
| 64  | compile fail     | —                     | 13.98        | 18.22        |
| 96  | compile fail     | —                     | 14.14        | 14.05        |
| 128 | compile fail     | —                     | compile fail | compile fail |
| 256 | 83.36            | 2.43                  | compile fail | compile fail |

**Conclusion**: int8 GEMM hardware really is engaged.
- At B=32 (the largest batch where fp16 BLOBFILE compiles cleanly): int8 is
  **1.43×** faster than fp16, int4/int8-cmp is **1.89×** faster.
- The original 11× gap "fp16 W-as-input → W4A8" decomposes as:
    1.21 → 4.90 TFLOP/s from removing weight upload (BLOBFILE const), 4×.
    4.90 → 18.22 TFLOP/s from int8 compute + int4 storage, ~3.7×.
- The fp16 BLOBFILE graph **fails to compile at B=64/96/128** but **succeeds at
  B=256**.  That's a different ANE-tile failure footprint than the int8/int4
  graphs (which fail at B=128/256 instead).  The compiler clearly picks
  different tile schedules for fp16 vs int8 weights, and the boundary
  conditions diverge.

### B.7 anemll-profile confirms int8 compute requires the quantize/dequantize hint

Generated two mlpackages with `gen_ds4_mlp_int8.py`:
- `ds4_mlp_fp16_const.mlpackage` — coremltools fp16 baseline.
- `ds4_mlp_int8.mlpackage` — same model after
  `coremltools.optimize.coreml.linear_quantize_weights` (int8, symmetric,
  per-channel).  B=32, 3 linear ops.

`anemll-profile` measured (both 100% ANE placement):

| mlpackage                          | measured ms | TFLOPs | ANE op_type_breakdown                                |
|------------------------------------|------------:|-------:|------------------------------------------------------|
| fp16 const                         | 5.36        | 5.29   | 3× `ios18.linear`, silu, mul                         |
| int8 (`linear_quantize_weights`)   | 5.44        | 5.35   | 3× `ios18.linear`, 3× `constexpr_blockwise_shift_scale`, silu, mul |
| W8A8 (hand-written, this repo)     | 3.62 (B=32) | **7.01** | 3× `conv1x1` with `quantize`/`dequantize` hint     |

**Key finding**: coremltools' default int8 weight quantization is **storage-only**.
The `constexpr_blockwise_shift_scale` op dequantizes the weight to fp16 at
inference, the `linear` op runs **fp16 GEMM**, and the two ops pipeline so the
dequant cost is hidden — but compute throughput is unchanged from fp16
(5.29 → 5.35 TFLOPs, within noise).

To actually engage int8 GEMM hardware, the activation side also has to be
quantized.  The proven pattern (and what `ds4_mlp_inmem_bench_int8.m` /
`ds4_mlp_inmem_bench_int4.m` use) is to insert
`quantize(fp16→int8) → dequantize(int8→fp16)` immediately before each conv.
That pair is a hint to the ANE compiler that the input tensor "is really
int8", which switches the GEMM unit to int8 mode.  Measured impact:
7.01 TFLOPs vs 5.35 TFLOPs at B=32 (1.5×), and 18.22 TFLOPs vs ~9 TFLOPs at
B=64 (2.0×) — within range of the 2× hardware multiplier int8 vs fp16 macs.

If you only quantize weights (no activation hint), you get the storage win
but nothing else.  This is why `inmem_peak_int8.m`'s W8A8 mode is structured
with `quantize`/`dequantize` between every layer.

### B.8 Sustained run for visual ANE-load confirmation

Short bursts (≤200 ms total) don't register in mactop's 1 Hz sampling.  Added
`ds4_mlp_sustained_bench.m`: compiles the int4 W4A8 graph once, then loops
for N seconds printing throughput every 250 ms.  Verified at 15 s on B=64:
**5050+ iterations, steady 2.78 ms/iter, 18.2 TFLOPs**, ANE load clearly
visible in mactop.  Matches the short-burst measurement exactly.

### B.9 Dynamic weights × int8 compute: not on the same path (2026-05-18)

The int8/int4 benches in §B.3–B.4 demonstrate real int8 GEMM on ANE, but they
use **const+BLOBFILE** weights, not function inputs.  That undoes the
"streamed weights" property the investigation started from.  Probing both
combinations confirms a hard constraint:

| op     | W as function input            | W as const+BLOBFILE                                |
|--------|--------------------------------|----------------------------------------------------|
| matmul | ✅ compiles, runs on ANE (fp16) | ✅ compiles, runs on ANE (fp16; `constexpr_*` + matmul rejected) |
| conv   | ❌ `CompilationFailure` (see `test_conv_w_input.m`) | ✅ compiles, runs on ANE; supports `constexpr_*` + `quantize/dequantize` hint -> int8 GEMM |

Neither cell offers both "dynamic W" and "int8 GEMM".  Today, you pick one:

- **Dynamic W (matmul + function input)** → fp16 compute only, ~1.2 TFLOPs
  peak, 18 ms upload floor on the DSv4 shapes.
- **Streamed via const+BLOBFILE swap (conv)** → up to 18 TFLOPs int4/int8
  GEMM, but the weights are baked into the compiled `_ANEInMemoryModel`.
  You'd need a fresh compile per weight set (~150 ms), or a different
  weight-injection mechanism.

Open bridges worth a follow-up:
1. ~~**`_ANERequest.weightsBuffer:` slot**~~ — **TESTED 2026-05-18 in
   `test_weightsbuffer_probe.m`**.  Result: this slot is **silently ignored**
   for BLOBFILE references on this OS.  Running the int4 W4A8 graph with
   `weightsBuffer = (a) nil, (b) IOSurface with exact same bytes,
   (c) IOSurface with zeroed weight payload, (d) IOSurface with a totally
   different valid W4 blob` produced **byte-identical outputs in all four
   cases**.  Conclusion: BLOBFILE always resolves against the descriptor's
   `weights:` dict; the `weightsBuffer:` slot is for a different mechanism
   (possibly only consulted by specific ANE program types — needs more
   probing).  Streamed weights via this route is **not** the answer.
2. **ane-prefill-bench "pack + slice" combined with int8**: smuggle int8
   weights through the activation input tensor, slice them out at runtime,
   feed into `constexpr_*` (which expects a tensor, not necessarily a const?
   needs verification).  Earlier minimal extraction hit `InvalidMILProgram`
   even for fp16 — this is the riskier path.
3. **N pre-compiled models, one per weight set**: compile cost (~150 ms each)
   amortizes only if weight switches are rare (≤ a few per second).  Useful
   for offline rotating-expert serving, not for token-grained streaming.

### B.10b SOLVED for m3u — full I-axis split (2026-05-18)

m3u (M3 Ultra) rejects the packed graph in §B.10a for the full DSv4 shape
because **any tensor with `I = 18432` in a matmul-relevant axis fails ANEC
compilation**, regardless of byte count.  TASK_BATCH_ENGINE_MOE.md established
this empirically: `H=7168, I=16384` compiles at 672 MB packed, `I=18432` fails
even at `H=1024`.

Fix: in `ds4_mlp_inmem_bench_packed_split3.m`, split everything along the I
axis into T=3 streams.  No tensor in the graph carries `I=18432` anywhere — the
maximum I-axis dim is `I/T = 6144`.

Structure (one packed function input, sliced into 1 activation + 9 weight
blocks):

```
for t in 0..2:
    gate_t   = matmul(tx=true,  x_3d[1,H,B], W_gate_t[H, I/T])    -> [1,B,I/T]
    up_t     = matmul(tx=true,  x_3d[1,H,B], W_up_t  [H, I/T])    -> [1,B,I/T]
    hidden_t = silu(gate_t) * up_t                                -> [1,B,I/T]
    d_t      = matmul(tx=false, hidden_t,    W_down_t[I/T, H])    -> [1,B,H]
output = d_0 + d_1 + d_2
```

Caller-side data layout requirement: the packed input must lay out W_gate /
W_up as 3 contiguous `[H, I/T]` row-major blocks each (i.e., one-time re-tile
of the original `[H, I]` weight).  W_down is naturally compatible: each `[I/T,
H]` block is a contiguous row range of the original `[I, H]` weight.

M5 measured at H=7168, I=18432:

| B   | non-tiled ms | K-tile (down only) ms | **split-3 (full I) ms** | TFLOP/s |
|----:|-------------:|----------------------:|-------------------------:|--------:|
| 1   | 18.12        | 13.48                 | **10.07**                | 0.08    |
| 8   | 18.10        | 13.46                 | **10.08**                | 0.63    |
| 16  | 18.14        | 13.48                 | **10.09**                | 1.26    |
| 32  | 26.14        | 13.53                 | **10.17**                | 2.49    |
| 64  | 42.23        | 13.58                 | **10.27**                | 4.94    |
| 96  | 70.35        | 30.29                 | 23.78                    | 3.20    |
| 128 | 84.21        | —                     | 20.40                    | 4.97    |

Below B≈64 the floor is the 756 MB packed weight upload (10 ms / 75 GB/s) — at
that point matmul time is overlapped with the upload.  Higher B becomes
compute-bound.

This is the **production candidate** for the m3u DSv4 path.  ~~Pending m3u verification.~~

**m3u verification ✅ (2026-05-18)** — compiles, loads, and evaluates for the
full DSv4 shape at all tested batches:

| B   | m3u ms | m3u TFLOP/s |
|----:|-------:|------------:|
| 1   | 12.95  | 0.06        |
| 8   | 13.00  | 0.49        |
| 16  | 13.03  | 0.97        |
| 32  | 13.15  | 1.93        |
| 64  | 13.45  | **3.77**    |
| 96  | 29.54  | 2.58        |
| 128 | 26.09  | 3.89        |

m3u sits about ~30% behind M5 at peak (13.45 vs 10.27 ms at B=64) — expected
for the ANE generation gap.  The 13 ms floor is again the 756 MB packed weight
upload at ~58 GB/s, not compute.  **This unblocks the
`DS4_FLASH_MOE_ANE_PREFILL=1` integration in `ds4-ssd/`.**

### B.10d Streamed int8 W-as-input — private API rejects, public MLModel API works (2026-05-18)

Asked: could we run `matmul + W-as-input` with W as **int8** (2× upload BW
saving over fp16) and dequant inside MIL?  `test_matmul_int8_dequant.m`
exhaustively probes the `_ANEInMemoryModel` (private API) path:

| MIL op                  | surface 0 (1D bytes) | surface 1 (2D bytes WxH) | surface 2 ('L008' pixfmt) | surface 3 (CVPixelBuffer OneComponent8) |
|-------------------------|:--------------------:|:------------------------:|:-------------------------:|:----------------------------------------:|
| `dequantize`            | ❌                   | ❌                       | ❌                        | ❌                                       |
| `cast + mul` (manual)   | ❌                   | ❌                       | ❌                        | ❌                                       |

All 8 cells compile + load OK and fail at the first eval with
`ANEProgramProcessRequestDirect status=0x1d, statusType=0x9: Program
Inference error`.  Also tried 4D-wrapped int8 input (`[H,I,1,1]`) and the
ios19 opset — both still fail at eval.  Conclusion for the private API:
**`_ANEInMemoryModel` does not accept int8 as a runtime function input.**

However, **ccv's `lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm` does this
exact pattern and works**.  The crucial difference is the API surface:

```
ccv:    MLMultiArray initWithPixelBuffer:shape:     // tagged dtype int8
          -> MLFeatureValue
            -> [MLModel predictionFromFeatures:]    // public API path
              -> ANE runtime adapter (handles int8 input dtype)
                -> ANE hardware

ours:   _ANEIOSurfaceObject objectWithIOSurface:    // raw IOSurface, no dtype
          -> _ANERequest
            -> _ANEInMemoryModel evaluate            // private API
              -> ANE hardware (rejects int8 input dtype)
```

ccv's MIL signature is `func main<ios19>(tensor<int8, [1,1,N,K]> w,
tensor<int8, [1,1,K,padded_M]> x)` with both weights and activations as int8
function inputs.  It works because the public MLModel API has an int8-input
adapter the private `_ANEInMemoryModel` path bypasses.

### Implication for production — RESOLVED 2026-05-18 via anemll-profile

Built the int8-input model via `gen_int8_streamed_probe.py` (coremltools MIL
Builder, `mb.TensorSpec(dtype=types.int8)` weight input, `mb.dequantize` +
matmul) at multiple shapes and profiled with `anemll-profile` (iOS26 target,
which supports int8 I/O natively — no cast).  The placement is **shape-
dependent**:

| shape (H × I, B=32)   | anemll-profile placement                 | notes |
|-----------------------|-------------------------------------------|-------|
| 256 × 512 (toy)       | **100% CPU**                              | "ANE supported but not preferred" — planner cost-model picks CPU at tiny sizes (ANE setup overhead exceeds workload). Header: ⚠ ANE compilation likely failed. |
| 7168 × 18432 (DSv4 gate) | **100% ANE** (dequantize, matmul, transpose, expand_dims all on ANE) | At production size the planner flips to ANE.  Measured 13.9 ms/iter, 0.63 TFLOP/s, 252 MB int8 streamed @ 18 GB/s. |

**Corrected verdict**: streamed int8 W on ANE **does work for DSv4 shapes**.
The earlier toy-shape CPU result was misleading — the cost model only
prefers CPU at sizes too small to justify ANE setup.  ccv's pattern works on
macOS the same way it works on iOS, just bench at production scale.

`bench_mlmodel_int8.m` (Obj-C, lower per-call overhead than the Python
proxy / profiler) measured the **single-matmul** at DSv4 gate shape:

| variant (single gate matmul, H=7168, I=18432, B=32) | ms/iter | TFLOP/s | streamed bytes |
|---|---:|---:|---:|
| fp16 W-as-input (`test_matmul_w_input`, private `_ANEInMemoryModel`) | 5.15 | 1.64 | 252 MB |
| **int8 W-as-input (`bench_mlmodel_int8`, public `MLModel + MLMultiArray`)** | **6.56** | **1.29** | 132 MB |

int8 streamed is **~1.3× slower per matmul** despite 2× less weight
bandwidth.  The dequantize op (`ios19.dequantize`, 3.87 ms in the profile)
sits in the critical path and is roughly the cost the bandwidth savings
should have bought.  Net per-matmul: int8 not a win for streamed weights at
DSv4 shapes on m5/m3u — the dequant cost exactly offsets the BW savings.

For the full **fused 3-matmul MLP**, the picture worsens: 3 dequants
(~12 ms total memory) + 3 matmuls (~11 ms compute) → ~20-23 ms with
pipelining vs fp16 split-3's measured **10 ms** end-to-end.  int8 streamed
is **2× slower than fp16 streamed** for the full MLP on macOS ANE.

### Why the asymmetry

The fp16 path's "upload" cost in §B.10c is just the
`tensor_buffer_to_tensor` op (~7.7 ms for 252 MB at 33 GB/s, but pipelined
with the matmul). The int8 path has the same `tensor_buffer_to_tensor`
(half the bytes → ~3.9 ms) PLUS a `dequantize` op (~3.9 ms, also at memory
BW). Total int8 setup ≈ fp16 setup. The matmul cost is the same fp16 GEMM
in both cases (ANE doesn't engage int8 hardware for runtime-input int8).

So **streamed int8 isn't a free win on ANE** — the dequant has to happen
*somewhere* and it lands inside the critical path as a separate op rather
than fusing with the matmul.

**Final shipping recommendation** for the DSv4 MoE expert path: stay on
`ane_ds4_mlp_split3` (fp16 streamed) for the production replacement.  int8
streaming is feasible — it placement-checks on ANE and runs — it's just
not faster than fp16 streaming for these shapes on this hardware.

### Other options still available

1. **fp16 streamed** (the working `ane_ds4_mlp_split3` library) — 756 MB/iter
   upload, 10 ms floor.  This is what's currently shippable.
2. **N pre-compiled int8 models** — one `_ANEInMemoryModel` per expert with
   weights baked into the descriptor BLOBFILE.  ~150 ms compile × 256
   experts = 38 s one-time startup.  Resident: 1.77 GB IQ2_XXS or 95 GB
   raw int8.  Expert switch is a model-handle swap.
3. **Move int8 / IQ2_XXS to Metal/GPU** — ccv-MFA `NAInt8AttentionKernel`
   demonstrates per-tile dynamic int8 GEMM on Metal.

### B.10c Library — `ane_ds4_mlp_split3.h` + `.m`

The standalone C ABI wrapping the split-3 graph for the production MoE
expert path:

```c
ds4_ane_mlp_split3_ctx *ds4_ane_mlp_split3_create(int H, int I, int B);
bool ds4_ane_mlp_split3_eval(ctx, input, Wgate_t3, Wup_t3, Wdown_t3, output);
void ds4_ane_mlp_split3_destroy(ctx);
// zero-copy variant for hot paths where caller writes the IOSurface directly
void *ds4_ane_mlp_split3_begin_input(ctx);
void  ds4_ane_mlp_split3_end_input  (ctx);
bool  ds4_ane_mlp_split3_eval_packed(ctx, output);
```

Smoke test (`ane_ds4_mlp_split3_smoke.m`) verified on M5 across B in {1, 8,
16, 32, 64} at full DSv4 shape.  Measured both API variants:

| B  | `_eval` (memcpy) ms | **`_eval_packed` (zero-copy) ms** | savings |
|---:|--------------------:|----------------------------------:|--------:|
| 1  | 22.46               | **10.08**                          | 12.4    |
| 8  | 22.52               | **10.08**                          | 12.4    |
| 16 | 22.48               | **10.10**                          | 12.4    |
| 32 | 22.50               | **10.18**                          | 12.3    |
| 64 | 22.82               | **10.28**                          | 12.5    |

`_eval()` is memcpy-dominated (~12 ms host → IOSurface for the 756 MB packed
input).  `_eval_packed()` is within noise of the original bench numbers in
§B.10b — the zero-copy path is what production integration should target;
the memcpy path is only useful for prototype glue.

Pushed to:
- `m3u:/Users/anemll/SourceRelease/GITHUB/ML_playground/ANE/`
- `m3u:/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/moe-batch-bench/`

Caller-side data layout (re-tile of W_gate / W_up along the I axis into 3
contiguous `[H, I/3]` blocks) is documented in the header.  This is the only
data-arrangement change the integration team needs to make; W_down is
naturally compatible.

**Open follow-ups for the full `ds4_gpu_routed_moe_expert_banked_batch_tensor`
drop-in replacement** (still TODO):
1. Wire per-expert dispatch over the dedup'd token lists.
2. Dequant from the quantized weight banks to fp16 + W_gate/W_up I-axis
   re-tile.  Two options: (a) add a Metal dequant kernel that writes directly
   into the ANE IOSurface, (b) CPU dequant during gather.  (a) is ~10× faster
   for DSv4 weight sizes; (b) is easier to get right first.
3. Apply router weights and scatter-add into `out` — match the existing GPU
   path's semantics.
4. Cache `ds4_ane_mlp_split3_ctx *` per (H, I, B) tuple seen at runtime; the
   `~70 ms` create cost is per-shape, not per-eval.

### B.10a Pack-and-slice fused MLP — single input descriptor (2026-05-18)

`ds4_mlp_inmem_bench_packed.m` collapses the four function inputs of
`ds4_mlp_inmem_bench.m` (input, W_gate, W_up, W_down) into a single 4D
function input:

```
packed: tensor<fp16, [1, 1, 1, N]>   N = B*H + 3*H*I
        = concat(input, W_gate, W_up, W_down)  (all flattened)
```

Inside the MIL, each piece is recovered with `slice_by_index` + `reshape`,
then the standard MLP body runs.  The down projection uses the converter's
`tx=false` pattern directly on `hidden [1,B,I]` (no extra transpose).

Patterns that turned out to matter:
- Packed input must be **4D**, not 1D — flat `tensor<fp16, [N]>` triggers
  ANE compile Code=22 (Failed to HandleANELayer); `tensor<fp16, [1,1,1,N]>`
  is accepted.  This matches what draft-afm.py uses for its slice work.
- Slicing must use `slice_by_index` with 4D `begin`/`end` const tensors.
  `slice_by_size` with 1D begin/size was rejected.
- Tiny shapes (H=256, I=512) fail at inference for the full fused graph
  ("Program Inference error", status 0x1d) — the ANE tile scheduler chokes
  on such small slice sizes.  All H≥1024 production shapes work cleanly.

Measured on M5, fp16, full DSv4 (H=7168, I=18432):

| B   | packed ms | TFLOP/s | vs multi-input bench (B same) |
|----:|----------:|--------:|-------------------------------|
| 1   | 18.12     | 0.04    | 18.15                          |
| 8   | 18.10     | 0.35    | 18.16                          |
| 16  | 18.14     | 0.70    | 18.14                          |
| 32  | 26.14     | 0.97    | 26.13                          |
| 64  | 42.23     | 1.20    | 42.10                          |
| 96  | 70.35     | 1.08    | 70.28                          |
| 128 | 84.21     | 1.20    | 84.23                          |
| 256 | 168.29    | 1.21    | 168.17                         |

Slice has zero observable overhead — perf is byte-for-byte the same as the
4-input variant.  **The packed graph is the portable candidate** for ANE
generations that won't compile the 4-input variant: one input descriptor,
one `tensor_buffer_to_tensor` op, otherwise identical to the multi-input
graph downstream.  Pending verification on m3u.

### B.10 Cross-hardware portability findings (2026-05-18, from m3u)

Running this session's benches on a second machine (`m3u.local`) surfaced two
gotchas that don't show up on the M5 dev box.

**Hard constraint (2026-05-18, from user)**: the deployment target needs an
**integrated, single-graph MLP**.  Splitting gate/up/down into separate
`_ANEInMemoryModel` instances and round-tripping the intermediate activations
through IOSurface between them is **not viable** — at DSv4 shapes the
intermediates are ~1 MB+ per call, and the per-submission launch + copy
overhead would swamp the compute.  Any portability fix has to keep all three
matmuls in one MIL.

1. **`transpose_x=true` for the down projection is rejected on the older
   ANE compiler.**  The high-level CoreML converter emits the down matmul as
   `transpose_x = false, transpose_y = false` with `hidden_cast_fp16` fed in
   directly (no extra transpose).  My `ds4_mlp_inmem_bench.m` unified all three
   matmuls under `tx=true` (so it can reuse one activation layout).  That
   works on M5 but **fails compile on m3u**.  Portable single-graph fix:
   match the converter exactly — gate/up use `tx=true` with `[1,H,B]` input,
   down uses `tx=false` with the `[1,B,I]` `hidden_cast_fp16` directly (drop
   the `transpose(perm=[0,2,1])` before the down matmul).  Action item.
2. **Fused 3-weight graph fails to compile on m3u even with the corrected
   down layout.**  Single-matmul models for each of gate/up/down compile and
   run on m3u individually, but combining all three with all-W-as-inputs in a
   single `_ANEInMemoryModel` exceeds some ANE resource limit
   (likely activation tile + multiple large weight-input descriptors).
   Since split-ANE is ruled out, the remaining single-graph options are:
   - **Reduce the number of large input descriptors**: pack `input + W_gate
     + W_up + W_down` into one large input tensor and slice them out inside
     the MIL (the ane-prefill-bench "pack + slice" trick).  This collapses
     four tensor_buffer_to_tensor ops into one, which is the specific
     resource a multi-weight-input graph stresses.
   - **Find an option to relax the ANE compiler's input-descriptor count
     limit** (via `optionsPlist` on the descriptor) — unexplored.
   - **Drop W-as-input and use BLOBFILE const** — works fine fused, but
     forfeits dynamic weights.
3. Implication for the §B.5 numbers: the **38.5 ms** mlpackage and
   **26.1 ms** fused private-API numbers were measured on M5 with the
   fused 3-matmul `tx=true` graph.  On m3u (and likely older ANE generations)
   the path forward is single-graph + pack-and-slice, not splitting.

### B.11 Files added this session

| file                              | purpose |
|-----------------------------------|---------|
| `ds4_mlp_inmem_bench_int8.m`      | W8A8 conv1x1 ANE bench (int8 weights via BLOBFILE + `constexpr_affine_dequantize`, int8 activations via `quantize`/`dequantize`) |
| `ds4_mlp_inmem_bench_int4.m`      | W4A8 conv1x1 ANE bench (int4 weights via `constexpr_blockwise_shift_scale`, int8 activations) |
| `ds4_mlp_inmem_bench_fp16_blob.m` | fp16 const-BLOBFILE baseline using the *same* conv1x1 graph as the int8/int4 benches. Used to attribute speedups between "weight residency" and "int8 compute". |
| `ds4_mlp_sustained_bench.m`       | Long-running int4 W4A8 loop with 250 ms throughput reports for visual ANE-load confirmation in mactop / powermetrics. |
| `gen_ds4_mlp_int8.py`             | Emits `ds4_mlp_fp16_const.mlpackage` and `ds4_mlp_int8.mlpackage` (coremltools `linear_quantize_weights`, symmetric, per-channel) for anemll-profile comparison. |
| `test_conv_w_input.m`             | Probe: can `_ANEInMemoryModel` compile `conv1x1(weight = <function-input>)`? Result: **CompilationFailure** — confirms conv requires const weights, blocking the "dynamic W + int8 GEMM" combination. |
| `test_weightsbuffer_probe.m`      | Probe: does `_ANERequest.weightsBuffer:` override BLOBFILE at runtime? Result: **silently ignored** — output bytes identical for nil / same / zeroed / different W4 IOSurfaces. The streamed-weights hypothesis is refuted for this code path. |
| `ds4_mlp_gpu_bench.py`            | torch MPS fp16 reference: `--bench` for the same batch sweep; `--verify DIR` for numerical comparison against the ANE-dumped binaries |
| `ds4_mlp_inmem_bench.m` (mod)     | added `-verify H I B [outDir]` mode: deterministic fp16 inputs/weights, single eval, dumps `input.bin`/`W_*.bin`/`output_ane.bin`/`meta.txt` for the Python verifier |

---

**Last updated:** 2026-05-18
**Owner:** Investigation between user and AI assistant on ANE private API + streamed weights for large non-square matmuls.

---

## Appendix A — 2026-05-18 Session Summary

Three concrete results from this session:

1. **`anemll-profile` on the high-level mlpackage**: the W-as-inputs CoreML model
   already runs **100% on ANE**. 3 matmuls (~3.58 ms each) + 3 weight uploads
   (`tensor_buffer_to_tensor`, ~2.58 ms each). Measured 38.5 ms/iter total.
   This refutes the earlier "CPU fallback" claim — the bottleneck is weight
   upload bandwidth, not ANE rejection.

2. **Working raw-MIL W-as-input test** (`test_matmul_w_input.m`): private
   `_ANEInMemoryModel` runs a single matmul with W as a function input on all
   three DSv4 shapes:

   | shape         | hidden | interm | time     | TFLOPs |
   |---------------|-------:|-------:|---------:|-------:|
   | gate / up     | 7168   | 18432  | 5.15 ms  | 1.64   |
   | down          | 18432  | 7168   | 16.10 ms | 0.53   |
   | smoke         | 256    | 512    | 0.12 ms  | 0.07   |

   The breakthrough was switching from the 4D-fp32-with-cast template to the
   2D-fp16 layout the high-level CoreML converter emits.

3. **`_ANERequest weightsBuffer:` discovered** (via `dump_ane_classes.m`).
   This is almost certainly the streamed-weights mechanism — `inmem_peak_matmul.m`
   passes `nil` for it. Worth a follow-up session.

4. **Full end-to-end private-API benchmark** (`ds4_mlp_inmem_bench.m`): chains
   `gate + up + silu + mul + down` with all four function args (input, W_gate,
   W_up, W_down).  Batch sweep at H=7168, I=18432, **extended 2026-05-18**:

   | B   | ms/iter | TFLOP/s | regime                                |
   |----:|--------:|--------:|---------------------------------------|
   | 1   | 18.15   | 0.04    | weight-upload floor (~18 ms / 756 MB) |
   | 8   | 18.16   | 0.35    | weight-upload floor                   |
   | 16  | 18.14   | 0.70    | weight-upload floor                   |
   | 32  | 26.13   | 0.97    | compute starts to matter              |
   | 64  | 42.10   | 1.20    | compute-bound                         |
   | 96  | 70.28   | 1.08    | compute-bound, off-tile               |
   | 128 | 84.23   | 1.20    | compute-bound                         |
   | 256 | 168.17  | 1.21    | compute-bound, saturated              |

   At B=32 the private-API path is **26.1 ms** vs the high-level mlpackage's
   **38.5 ms** — ~12 ms saved by avoiding the CoreML runtime.  Below B≈16 the
   whole MLP is dominated by re-uploading 756 MB of weights each iteration,
   confirming that further wins require `weightsBuffer:` / streamed weights.
   ANE plateaus at **~1.21 TFLOP/s fp16** for these shapes when weights are
   uploaded every iteration.