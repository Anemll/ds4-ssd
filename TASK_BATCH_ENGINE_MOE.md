# Task: Per-Expert Batch-Size Compute Engine Selection for MoE Token Grouping (d_dup / routed experts)

## Goal
Investigate and prototype dynamic selection of compute engine for MoE sub-functions that group tokens by expert (the "ddup" / dispatch / routed-expert path). After the router produces per-token expert assignments, tokens are grouped so that each expert processes a variable-sized batch. The selected engine for each expert group should depend on that group's effective batch size:

- **Larger per-expert batch** → Apple Neural Engine (ANE)
- **Medium / smaller per-expert batch** → GPU (Metal)
- **Very small (1–2 tokens)** → AMX (or CPU reference path)

The goal is to estimate performance/latency/throughput and decide whether a hybrid engine strategy is worthwhile.

## Background
- DeepSeek V4 Flash uses a routed MoE with expert grouping.
- Current code has static backend choice (`--metal` / `--cuda` / `--cpu`) and MoE sidecar handling.
- Token grouping by expert already happens; we now want to inspect the resulting per-expert batch sizes and route the computation accordingly.
- Apple platforms offer multiple accelerators: Metal GPU, ANE, and AMX on the CPU side.

## Requirements
1. After expert routing, measure the per-expert token batch size (number of tokens assigned to each routed expert in the current step).
2. Implement (or stub) a decision function that maps per-expert batch size → preferred engine:
   - Large batch (e.g. ≥ 8–16) → target ANE path (if available / implementable)
   - Medium batch → Metal GPU kernels
   - 1–2 tokens → AMX-accelerated or highly-optimized CPU path
3. Add instrumentation to log/report the observed per-expert batch sizes and chosen engines (under `--trace` or a dedicated flag).
4. Create micro-benchmarks or estimation code that measures latency/throughput for representative batch sizes on each engine.
5. Keep the change non-breaking: default to current Metal/GPU path; the new logic is opt-in or behind an experimental flag.
6. Produce estimates and recommendations (numbers, graphs, or tables) on whether the hybrid approach is beneficial.

## Files Likely to Change
- `ds4.c` – MoE forward pass, expert grouping / ddup logic, batch-size inspection, decision helper.
- `ds4_metal.m` / `metal/*.metal` – existing GPU kernels (baseline).
- New or experimental files for ANE and AMX paths (or thin wrappers).
- Possibly benchmark scripts under `speed-bench/` or a new `moe-batch-bench/` directory.

## Acceptance Criteria
- Code can report per-expert batch sizes during a normal run.
- Decision logic is implemented and can be exercised (even if ANE/AMX paths are initially stubbed).
- Micro-benchmarks or estimation harness produces comparable numbers for the three regimes.
- No regression on existing single-request or speed-bench workloads when the feature is disabled.

## Refined Approach (Pre-Graph Investigation)
Before touching the main inference graph, create standalone micro-benchmarks / execution harnesses for the core MLP sub-ops (up-projection, gate, down-projection) using the exact tensor sizes from DeepSeek V4 Flash.

Test three compute units:
- Apple Neural Engine (ANE)
- Metal GPU
- AMX (CPU)

Key measurement goals:
- Performance (latency & throughput) across a range of batch sizes (especially the per-expert batch sizes that appear after token grouping).
- Overhead of weight loading, cacheability, and "slot install" / setup costs (these can dominate on GPU and must be quantified).
- Emphasize near-zero-latency direct execution for the actual MAC operations.

For ANE, investigate two concrete paths:
1. Matmul API that supports streaming weights directly from host CPU memory **without** kernel compilation or model loading step (preferred for low-overhead dynamic use).
2. (To be explored) Any alternative lower-overhead or compiled path if the streaming path proves insufficient.

**Reference implementation for ANE in-memory streaming matmul:**
`/Users/anemll/SourceRelease/GITHUB/ML_playground/ANE/inmem_peak_matmul.m` (contains working `_ANEInMemoryModel` + MIL generation patterns for fp16/int8 matmul).

**Quantization considerations for ANE path:**
- ANE cannot perform quantization at runtime.
- Common strategy: Quantize on GPU first (usually fastest), then stream the already-quantized weights (INT8 or FP16) to ANE via the in-memory blob path.
- Alternative: CPU quantization (simpler, sometimes competitive for very small weights).
- Key decision: target INT8 (better speed, potential quality trade-off) vs FP16 (higher quality, higher bandwidth). This choice affects both the weight blob format and the final performance/quality numbers we collect.

**2-bit IQ quantization (current repo format):**
- The DS4 repo uses highly asymmetric 2-bit vector quantization for routed experts: `IQ2_XXS` for gate/up, `Q2_K` for down.
- The benchmark harness should also test this exact 2-bit IQ format (or a close emulation) to measure real-world speed/quality when feeding ANE from the same quantized weights used by the main engine.

**Hardware note – ANE INT8 acceleration:**
- Current ANE hardware is primarily FP16-optimized.
- Future silicon (M4/M5 class or later) is expected to have accelerated INT8 paths.
- When designing the benchmark and quantization flow, keep both FP16-only and accelerated-INT8 targets in mind so we do not prematurely close off the higher-performance INT8 route on newer hardware.

## DeepSeek V4 Flash MLP Workflow & Tensor Sizes

DeepSeek V4 Flash uses a SwiGLU-style routed MoE + shared experts.

**Hidden state size** (embedding_length):
- `n_embd = required_u32(m, "deepseek4.embedding_length")`   (typically 7168 for the Flash variant)

**Expert FFN size** (per expert):
- `n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length")`   (typically ~18432–18944)

**MLP sub-operations per routed expert** (the part we will benchmark):
1. `gate = x @ W_gate`   (hidden → intermediate)   shape: [batch, n_embd] × [n_embd, n_ff_exp]
2. `up   = x @ W_up`     (hidden → intermediate)   shape: [batch, n_embd] × [n_embd, n_ff_exp]
3. `act  = silu(gate) * up`  (element-wise)
4. `down = act @ W_down` (intermediate → hidden) shape: [batch, n_ff_exp] × [n_ff_exp, n_embd]

Shared experts follow the same three-projection pattern but are always active.

**d_dup / expert dispatch** (what we need to understand and instrument):
After the router produces logits for all experts, the top-k experts are selected per token. Tokens are then grouped by expert so that each expert receives a contiguous batch of tokens that chose it. "d_dup" (or dispatch duplication) refers to the data-movement / indexing step that prepares these per-expert batches (copying or striding the hidden states into expert-specific buffers). This step has non-trivial memory traffic and cache effects, especially when the resulting per-expert batch sizes are small and irregular.

**Additional quantities we should capture**:
- Number of active experts per token (usually 8 for DeepSeek V3/V4 Flash)
- Distribution of per-expert batch sizes after grouping (highly variable; many experts may receive 0–4 tokens while a few receive 16+)
- Memory layout of the hidden state buffers (`ffn_cur`, `ffn_moe`, etc. in ds4_session)

## Next Steps
1. Create a minimal standalone test harness (new small C / Objective-C program or extension under `speed-bench/`) that exercises the three projections (gate/up/down) at various batch sizes on ANE, Metal GPU, and AMX.
2. Instrument timing for both the pure compute (MAC) and all setup / weight-transfer / slot-install costs.
3. Collect data for batch sizes 1, 2, 4, 8, 16, … (and irregular sizes that appear in real d_dup grouping).
4. Measure and report the overhead of weight streaming vs. pre-installed weights on each engine.
5. Use the collected numbers to decide whether per-expert dynamic engine selection is worth integrating into the main graph.

## 2026-05-18 Benchmark Update: GPU + AMX With Background Dequant

Implemented a standalone `moe-batch-bench` harness and opt-in real-run histogram
tracing. The relevant new mode is:

```sh
./moe-batch-bench/moe-batch-bench --backend qhybrid --qhybrid-mode both --split-amx-batch 32 --batches 64,96,128,160 --iters 120
```

This runs one actual DS4 quantized Metal expert block on GPU concurrently with
one BNNS/AMX BF16 expert block. In `bgdequant` mode, a second GPU command queue
materializes the next AMX expert's dense BF16 weights into a ping-pong slot in
the same timed interval.

Key results on the tested M3 Ultra:

- Cached dense AMX weights are strongly useful: AMX B32 + GPU B96/B128/B160 is
  roughly 1.6-1.8x versus GPU-only sequential execution of those two experts.
- If AMX weights must be GPU-dequantized in the same interval, B16 is not useful
  (`~1.00x` best case). B32 is the first plausible AMX size.
- Fine sweep for B32 with background dequant:
  - first sustained `>=1.05x` at GPU B64,
  - best observed point around GPU B128 (`~1.10x`),
  - typical gains are single-digit once dequant cost is included.
- Therefore, foreground or one-shot dequant is not enough to justify a broad
  AMX path. The useful design is a dense BF16 cache populated ahead of need.

Prompt histogram runs using `/Volumes/optane/dsv4-iq2xxs-expert-major` and the
coding prompt suite:

```text
1K:   1,098 tokens,  9,541 active expert batches, reuse 29.69x
4K:   4,344 tokens, 27,886 active expert batches, reuse 40.19x
8K:   8,423 tokens, 46,698 active expert batches, reuse 46.54x
16K: 17,329 tokens, 87,222 active expert batches, reuse 51.26x
```

The longer prompts have very high chunk-to-chunk `(layer, expert)` locality:
4K/8K/16K repeated layer-expert refs account for `99.3% / 99.6% / 99.9%` of
routed references. This supports background dense-cache fill across prompt
chunks. However, dense BF16 materialization is about `48 MiB` per expert, so a
large per-layer cache is expensive: top-8 experts/layer is already about
`16.1 GiB`, and top-32/layer is about `64.5 GiB`.

Recommended experimental policy:

- Do not route B16 to AMX unless weights are already cached and the GPU queue is
  known to be the bottleneck.
- Route B32-ish expert batches to AMX only when dense BF16 weights are already
  cached, or when background dequant can be paired with a GPU expert batch of at
  least ~64 refs.
- Keep uncached experts on the existing fused quantized GPU path.
- Explore a small predictor/LRU dense cache for repeated layer-experts across
  prefill chunks before wiring AMX into the production graph.

This remains an investigation-first task focused on measurement and data.

## 2026-05-18 Follow-up: Prefill Staging Fix Beats Hybrid Projection

While rechecking the real prompt path, the current Flash-MoE prefill staging
pipeline showed a larger issue than the AMX/GPU split itself:

- Four transient prefill banks were allocated.
- Prefetch staged experts into all four banks.
- The compute path only read bank 0/1, so bank 2/3 prefetches were not usable.
- The current expert was then staged again even when a previous prefetch had
  already filled the correct bank.
- With `DS4_FLASH_MOE_PREFETCH=0`, the old code also staged the current expert
  once more through the "future" prefetch path.

The prototype fix rotates compute across all four prefill banks and tracks the
expert currently resident in each bank so duplicate staging is skipped. This is
still the same fused quantized GPU expert path; it does not introduce AMX into
production inference.

Patched real-prompt prefill measurements:

```text
1K:  p0 52.46 t/s, p1 52.47, p2 53.04, p3 53.57
4K:  p0 66.52 t/s, p1 67.39, p2 67.91, p3 68.67
8K:  default p3 77.08 t/s
16K: default p3 82.82 t/s
```

Install counts now match the unique expert-batch count closely:

```text
1K:  misses 9,799  vs unique 9,541
4K:  misses 28,144 vs unique 27,886
8K:  misses 46,956 vs unique 46,698
16K: misses 87,480 vs unique 87,222
```

This changes the priority order:

1. Keep the four-bank prefill staging fix; it is an immediate real prefill t/s
   win on the existing quantized GPU path.
2. Keep default `DS4_FLASH_MOE_PREFETCH=3`; after the bank fix it is the best
   tested setting for 1K and 4K.
3. Treat AMX as a second-stage optimization only if dense BF16 weights are
   already cached. The estimated AMX hybrid gains are smaller than the staging
   fix unless a large dense cache is available.

Dequant path measurements:

```text
AMX cached BF16 compute:       B16 0.84 ms, B32 0.77 ms
AMX + CPU scalar dequant:      B16 90.7 ms, B32 96.2 ms
AMX + GPU materialization:     B16 2.19 ms, B32 2.27 ms
QHybrid B32 background dequant best observed: ~1.10x at GPU B128
```

Conclusion: CPU foreground dequant is not viable. GPU materialization can make
AMX useful only when it is hidden in the background or amortized by a dense
cache. For prefill throughput on this system, fixing duplicate GPU-path staging
is the dominant optimization found so far.

### Negative result: naive quantized slot-cache for prefill

Added an opt-in experiment, `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=N`, that uses
the existing per-layer quantized GPU slot bank for the top-N expert batches in a
prefill layer call and leaves the rest on the transient four-bank path.

Measured result:

```text
1K top0:  53.95 t/s, hits 0
1K top8:  52.21 t/s, hits 37
1K top16: 51.71 t/s, hits 58
1K top32: 51.86 t/s, hits 79
4K top0:  69.53 t/s, hits 0
4K top32: 62.31 t/s, hits 1,142 (4.1%)
```

This loses because slot-cache installs must synchronize before they can safely
mutate the shared slot bank, and the naive per-chunk top-N policy does not
produce enough hits to pay for that synchronization. Leave this path opt-in and
disabled by default. A useful quantized cache would need a persistent predictor
and/or a non-evicting cache path separate from the decode slot bank.

### Synthetic FP4-LUT vs int8 fixed-point direct compute

Added `--backend gpuqsynthetic` to `moe-batch-bench`. It runs the same MLP shape
through simple Metal direct matmul kernels where weights stay compressed until
the inner loop:

- `fp4_lut_in_kernel`: two weights per byte, expanded through a 16-entry half
  LUT.
- `int8_fixed_in_kernel`: signed int8 weights with a fixed `1/127` scale.

This is intentionally synthetic. It is not the tuned DS4 quantized kernel, and
it is not a full MXFP4 decoder. The reference file
`/Volumes/SN8100/DS/DeepSeek-V4-Flash-FP4-FP8-native.gguf` was inspected enough
to confirm routed expert tensors use GGUF type `39`, which local gguf constants
identify as `MXFP4`:

```text
blk.0.ffn_down_exps.weight dims=(2048, 4096, 256) type=39
blk.0.ffn_gate_exps.weight dims=(4096, 2048, 256) type=39
blk.0.ffn_up_exps.weight   dims=(4096, 2048, 256) type=39
```

Measured synthetic results:

```text
B1:   FP4 LUT 4.51 ms, int8 fixed 3.57 ms
B2:   FP4 LUT 4.32 ms, int8 fixed 3.33 ms
B4:   FP4 LUT 4.15 ms, int8 fixed 3.27 ms
B8:   FP4 LUT 4.16 ms, int8 fixed 3.29 ms
B16:  FP4 LUT 4.18 ms, int8 fixed 3.39 ms
B32:  FP4 LUT 4.53 ms, int8 fixed 3.36 ms
B64:  FP4 LUT 5.62 ms, int8 fixed 4.06 ms
B128: FP4 LUT 9.65 ms, int8 fixed 7.22 ms
```

In this kernel, FP4's smaller reads do not beat int8. The nibble unpack and LUT
load cost is higher than the bandwidth saved. This supports an int8/fixed-point
intermediate experiment before investing in a full MXFP4 production kernel,
unless the MXFP4 block format enables a much more vector-friendly decode path
than the generic LUT test.

### ANE W-as-input MLP integration attempt

Integrated the private-API ANE samples into `moe-batch-bench`:

- `ane_ds4_mlp_inmem_bench.m`: full gate+up+silu+mul+down MLP with input and
  all three weight matrices as fp16 function inputs.
- `ane_ds4_mlp_inmem_bench_packed.m`: same fused MLP using one packed 4D
  function input `[1,1,1,N]`, then `slice_by_index` + `reshape` inside MIL.
- `ane_matmul_w_input_probe.m`: single-matmul probe for W-as-input shapes.
- `verify_ane_dump_numpy.py`: CPU fp32 NumPy verifier for dumps produced by
  `-verify`, because the default Python environment on this machine does not
  have `torch` installed for `ds4_mlp_gpu_bench.py --verify`.
- `make ane-mlp-bench`: builds the ANE probes.

Important result: the exact full DSv4-shape path from
`DSv4_MLP_ANE_Matmul_Investigation.md` does **not** currently compile on this
system as a monolithic W-input MLP. The gate/up single matmul still compiles,
but the down shape fails:

```text
single gate/up W-input: H=7168 I=18432 B=32 -> 6.95 ms, 1.22 TFLOP/s
single down W-input:    H=18432 I=7168 B=1/8/16/32 -> ANE CompilationFailure
full DSv4 W-input MLP:  H=7168 I=18432 B=1/8/16/32 -> ANE CompilationFailure
```

The current repo benchmark shape (`H=4096, I=2048`) does compile and run:

```text
B1:   1.311 ms, 0.04 TFLOP/s, 48 MB weights/iter
B8:   1.327 ms, 0.30 TFLOP/s, 48 MB weights/iter
B16:  1.290 ms, 0.62 TFLOP/s, 48 MB weights/iter
B32:  1.273 ms, 1.27 TFLOP/s, 48 MB weights/iter
B64:  1.206 ms, 2.67 TFLOP/s, 48 MB weights/iter
B128: 1.296 ms, 4.97 TFLOP/s, 48 MB weights/iter
```

Numerics for the 4096/2048 path match CPU fp32 closely:

```text
B1: max_abs=6.75e-05 mean_abs=1.50e-05
B8: max_abs=8.65e-05 mean_abs=1.57e-05
```

Decision for DeDup scheduling:

- ANE fp16 W-as-input is not a replacement for the fused quantized GPU path on
  the tested 4096/2048 shape. GPU DS4 quantized expert is faster at all shared
  batch points (`B32`: GPU 0.758 ms vs ANE 1.273 ms; `B128`: GPU 1.019 ms vs
  ANE 1.296 ms).
- ANE can beat uncached AMX with GPU materialization, and it is close to cached
  AMX around B64, but it streams 48 MB of fp16 weights per expert evaluation.
  That makes it primarily a memory-bandwidth path, not a clean extra compute
  engine.
- For actual DSv4 Flash `7168/18432`, do not wire ANE into production DeDup
  routing yet. The current system needs either a working down-projection
  W-input shape, a split/down workaround, or the `_ANERequest.weightsBuffer:`
  bridge from §B.9 before ANE can be considered a real GPU/AMX peer.

Packed-input follow-up on this m3u system (`Apple M3 Ultra`, `Mac15,14`):

- The M5 candidate `ds4_mlp_inmem_bench_packed.m` was copied into the bench
  tree and given `-bench-shape H I batches warmup iters` controls.
- Original upstream packed source, built and run directly on m3u, fails ANEC
  compilation for the full DSv4 shape at every tested batch:

```text
packed full MLP: H=7168 I=18432 B=1/8/16/32/64/96/128/256 -> ANE CompilationFailure
```

- Packed input itself is not rejected on m3u. Smaller/intermediate shapes
  compile and evaluate:

```text
H=1024 I=2048  B32 -> 1.229 ms, 0.33 TFLOP/s, 12.1 MB packed
H=2048 I=4096  B32 -> 1.032 ms, 1.56 TFLOP/s, 48.1 MB packed
H=4096 I=2048  B32 -> 1.219 ms, 1.32 TFLOP/s, 48.2 MB packed
H=4096 I=8192  B32 -> 6.278 ms, 1.03 TFLOP/s, 192.2 MB packed
H=7168 I=4096  B32 -> 3.737 ms, 1.51 TFLOP/s, 168.4 MB packed
H=7168 I=8192  B32 -> 11.094 ms, 1.02 TFLOP/s, 336.4 MB packed
H=7168 I=12288 B32 -> 37.679 ms, 0.45 TFLOP/s, 504.4 MB packed
H=7168 I=14336 B32 -> 46.386 ms, 0.43 TFLOP/s, 588.4 MB packed
H=7168 I=16384 B32 -> 43.267 ms, 0.52 TFLOP/s, 672.4 MB packed
```

- The rejection boundary is dimension-sensitive, not simply packed byte size:
  `H=7168,I=16384` compiles at ~672 MB packed, while `I=18432` fails even with
  `H=1024` and `H=2048`.
- Practical m3u conclusion: packed 4D input solves the "many large descriptors"
  class on M5, but does not make full DSv4 `I=18432` streamable on m3u. Also,
  the high-intermediate packed cases that do compile are too slow for prefill
  offload. For m3u, the viable ANE branch is therefore BLOBFILE-const/quantized
  ANE experiments or split-intermediate models below the failing dimension,
  not monolithic W-as-input streaming.

DeDup integration boundary:

- The production replacement point is inside
  `metal_graph_flash_moe_run_prefill_dedup()`, after each compacted expert's
  token list is available. The current flow is:

```text
dedup refs -> stage quantized expert -> gather token rows
           -> ds4_gpu_routed_moe_expert_banked_batch_tensor()
           -> scatter-add expert output
```

- A real ANE backend would replace only the per-expert MLP call while preserving
  GPU router/dedup/gather/scatter. It must therefore handle:
  1. moving gathered activations from Metal tensor storage to the ANE input,
  2. presenting the expert weights to ANE,
  3. returning per-token expert output for the existing scatter-add.
- The packed W-as-input ANE graph is not wired into DeDup on m3u because the
  full DSv4 expert shape fails compilation before runtime. Wiring it behind an
  env flag would only create a production path that deterministically aborts on
  this target. On an M5 target where the packed graph compiles, the next
  integration step is a guarded `DS4_FLASH_MOE_ANE_PREFILL=1` backend at this
  replacement point, with automatic fallback to
  `ds4_gpu_routed_moe_expert_banked_batch_tensor()` if ANE model creation or
  first eval fails.

Update: the split-3 packed ANE graph fixes the m3u full-shape compile failure
by splitting the entire intermediate axis into three `I/3=6144` streams. The
repo now includes `ane_ds4_mlp_inmem_bench_packed_split3.m` in
`make ane-mlp-bench`. Local m3u verification for `H=7168,I=18432`:

```text
B1:   13.117 ms, 0.06 TFLOP/s, 756.0 MB packed
B8:   13.066 ms, 0.49 TFLOP/s
B16:  13.148 ms, 0.96 TFLOP/s
B32:  13.216 ms, 1.92 TFLOP/s
B64:  13.109 ms, 3.87 TFLOP/s
B96:  29.303 ms, 2.60 TFLOP/s
B128: 25.817 ms, 3.93 TFLOP/s
```

The production hook is now present but intentionally falls back:

```text
DS4_FLASH_MOE_ANE_PREFILL=1
  -> ds4_gpu_routed_moe_expert_banked_batch_ane_tensor()
  -> fallback to ds4_gpu_routed_moe_expert_banked_batch_tensor()
```

The missing implementation is the worker body: materialize the current
quantized banked expert into one fp16 split-packed ANE input, with W_gate/W_up
retiled into three contiguous `[H,I/3]` blocks and W_down copied as three
contiguous `[I/3,H]` row ranges, then evaluate ANE and feed the result back to
the existing scatter-add path. Also note that this checkout's active compiled
Flash constants are `H=4096,I=2048`; the split-3 full-shape path targets the
larger `H=7168,I=18432` DSv4 variant.

### ANE+GPU concurrent execution probe

Added `moe-batch-bench/run_ane_gpu_concurrency_probe.py` to test whether the
split-3 ANE graph can overlap with the quantized Metal GPU expert path. This is
a process-level concurrency probe, not production DeDup integration: the GPU
side uses the repo's active `4096 x 2048` quantized expert benchmark, while the
ANE side uses the split-3 `7168 x 18432` full-shape packed graph.

Clean m3u results:

```text
GPU B2048 + ANE B64:
  solo GPU wall 3.984 s, GPU ms 7.272
  solo ANE wall 3.803 s, ANE ms 13.203
  concurrent wall 3.688 s
  speedup vs sequential wall 2.112x

GPU B4096 + ANE B64:
  solo GPU wall 4.277 s, GPU ms 13.739
  solo ANE wall 3.550 s, ANE ms 13.182
  concurrent wall 4.281 s
  speedup vs sequential wall 1.828x

GPU B2048 + ANE B128:
  solo GPU wall 3.682 s, GPU ms 7.146
  solo ANE wall 4.250 s, ANE ms 25.975
  concurrent wall 4.263 s
  speedup vs sequential wall 1.861x
```

Conclusion: the GPU and ANE work do overlap on this system. Concurrent wall
time is effectively the slower of the two solo runs for the clean B4096+B64 and
B2048+B128 points. This supports a production schedule that sends different
large expert batches to GPU and ANE concurrently. The remaining question is not
hardware overlap; it is whether quantized-bank -> fp16 split-packed
materialization plus result handoff can be made cheap enough for real DeDup.

Large GPU batch follow-up (`8k` through `64k` GPU expert batches):

```text
GPU-only smoke:
  B8192:  26.077 ms, 15.81 TFLOP/s
  B16384: 51.764 ms, 15.93 TFLOP/s
  B32768: 102.937 ms, 16.02 TFLOP/s
  B65536: 276.167 ms, 11.94 TFLOP/s

GPU B8192 + ANE B64:
  solo GPU 2.357 s, solo ANE 1.314 s, concurrent 2.346 s
  speedup vs sequential wall 1.565x

GPU B16384 + ANE B64:
  solo GPU 2.500 s, solo ANE 1.314 s, concurrent 2.514 s
  speedup vs sequential wall 1.517x

GPU B32768 + ANE B64:
  solo GPU 2.855 s, solo ANE 1.308 s, concurrent 2.868 s
  speedup vs sequential wall 1.451x

GPU B65536 + ANE B64:
  solo GPU 4.835 s, solo ANE 1.298 s, concurrent 4.784 s
  speedup vs sequential wall 1.282x

GPU B8192 + ANE B128:
  solo GPU 2.349 s, solo ANE 1.408 s, concurrent 2.361 s
  speedup vs sequential wall 1.591x

GPU B16384 + ANE B128:
  solo GPU 2.504 s, solo ANE 1.390 s, concurrent 2.496 s
  speedup vs sequential wall 1.560x

GPU B32768 + ANE B128:
  solo GPU 2.863 s, solo ANE 1.388 s, concurrent 2.854 s
  speedup vs sequential wall 1.489x

GPU B65536 + ANE B128:
  solo GPU 4.892 s, solo ANE 1.420 s, concurrent 4.854 s
  speedup vs sequential wall 1.300x
```

The overlap still holds at very large GPU batches: concurrent wall time tracks
the slower GPU run. However, the production value depends on how much useful
work can be assigned to ANE in the same window. If GPU gets an 8k-64k expert
batch while ANE only gets B64/B128, the incremental token count is small. For a
real win, scheduling should give ANE one or more large full-shape expert batches
whose total ANE time fits under the GPU-side critical path.

ANE-only scaling and multi-job probe:

```text
single split-3 ANE job:
  B64:  13.103 ms, 3.87 TFLOP/s
  B128: 25.891 ms, 3.92 TFLOP/s
  B192: 38.570 ms, 3.95 TFLOP/s
  B256: 51.334 ms, 3.95 TFLOP/s
  B384: 76.861 ms, 3.96 TFLOP/s
  B512: 102.457 ms, 3.96 TFLOP/s

two ANE jobs, B64 each:
  solo wall 1.317 s, solo eval 13.209 ms
  parallel wall 2.339 s, parallel evals 25.192 / 24.598 ms
  speedup vs sequential wall 1.126x

two ANE jobs, B128 each:
  solo wall 1.918 s, solo eval 25.900 ms
  parallel wall 3.478 s, parallel evals 49.465 / 50.324 ms
  speedup vs sequential wall 1.103x

three ANE jobs, B128 each:
  solo wall 1.385 s, solo eval 25.876 ms
  parallel wall 3.666 s, parallel evals 76.108 / 72.322 / 74.159 ms
  speedup vs sequential wall 1.133x
```

Conclusion: larger single ANE batches scale linearly after B128 at about
3.95 TFLOP/s; they do not get progressively more efficient beyond that.
Multiple ANE jobs do not provide useful parallelism on this system: eval time
roughly scales with the number of jobs, so the runtime/hardware is effectively
serializing or partitioning the same ANE resource. Production scheduling should
therefore issue one larger ANE batch at a time, not many concurrent ANE batches.

### Real production prefill + ANE coexistence test

Added `moe-batch-bench/run_real_prefill_ane_concurrency.sh`. This is not a fake
microbench: it runs the real pinned 8K `ds4` prefill path with sidecar,
router/dedup/gather/scatter, then repeats the same prefill while a split-3 ANE
B128 worker runs concurrently for roughly the full prefill window.

Settings:

```sh
DS4_METAL_PREFILL_CHUNK=16384
DS4_FLASH_MOE_PREFETCH=3
ANE_BATCH=128
ANE_ITERS=2200
```

Result:

```text
baseline_wall_s=61.849
concurrent_wall_s=61.980
baseline_prefill_tps=140.72
concurrent_prefill_tps=142.39
concurrent_vs_baseline_tps=1.012
baseline_dedup=(2173134, 18869, 115.17)
concurrent_dedup=(2173134, 18869, 115.17)
ane_eval=(26.216 ms, 3.87 TFLOP/s)
```

Conclusion: a real ANE split-3 stream can run alongside production pinned 8K
GPU prefill without reducing GPU prefill throughput on this system. This does
not yet prove an end-to-end speedup, because the ANE output is not integrated
into DeDup, but it removes the main contention concern. The next production
test should replace one or more large expert batches with ANE work and include
the quantized-bank -> fp16 split-packed materialization cost.

### Split-3 materialization cost

Added `--backend anepack` to `moe-batch-bench` to materialize the split-3 ANE
packed input from synthetic `IQ2_XXS` gate/up and `Q2_K` down blocks on CPU.
This emits the exact caller-side layout required by the split-3 graph:

```text
[input fp16 | gate_0..2 fp16 [H,I/3] | up_0..2 fp16 [H,I/3] | down_0..2 fp16 [I/3,H]]
```

Full-shape result:

```text
./moe-batch-bench/moe-batch-bench --backend anepack \
  --in 7168 --mid 18432 --batches 128 --warmup 1 --iters 1

ane_split3_pack_cpu,iq2xxs_q2k_to_fp16,B128,H7168,I18432:
  1609.317 ms, 0.494 GB/s packed-output bandwidth
```

Conclusion: CPU materialization is a non-starter. The split-3 ANE eval is about
26 ms at B128, while CPU pack/dequant is about 1.6 seconds. The real
`DS4_FLASH_MOE_ANE_PREFILL=1` worker must use GPU materialization into the
split-packed fp16 buffer, ideally overlapping materialization for the next ANE
expert with current GPU/ANE compute.

### Correction: AMX/ANE integration tests must use large prefill chunks

The proper prefill integration baseline on this system is:

```sh
DS4_METAL_PREFILL_CHUNK=16384 DS4_FLASH_MOE_PREFETCH=3
```

Earlier prompt-level conclusions used smaller effective chunks for some runs,
which overestimated the number of per-expert batches and therefore overstated
the opportunity for AMX/ANE side execution. With the correct env pinned, the
existing quantized GPU path gets much better multi-token batching and dedup:

```text
1K:   53.92 t/s, unique  9,541, reuse  29.69x
4K:  139.21 t/s, unique 10,440, reuse 107.35x
8K:  141.81 t/s, unique 18,869, reuse 115.17x
16K: 157.75 t/s, unique 30,205, reuse 148.02x
```

This is the baseline to use for any AMX/ANE integration claim. Compared with
the earlier smaller-chunk 16K run (`82.82 t/s`, `87,222` unique), the proper
chunk setting nearly doubles prefill throughput and cuts unique expert staging
by about 2.9x.

Updated scheduling implication:

- AMX/ANE should not be evaluated against small-chunk prompt runs.
- Large chunks make each routed expert batch larger and reduce staging count,
  so the fused quantized GPU path becomes more favorable.
- AMX still only makes sense for already-cached dense BF16 weights, and ANE
  W-input is even less attractive because it streams fp16 weights per expert.
- The next meaningful side-engine experiment must run under the pinned env and
  report combined prefill t/s, not isolated microbench speedup.

### 8K AMX feasibility test under pinned prefill settings

Added `moe-batch-bench/run_8k_amx_feasibility.sh` to make the 8K check
repeatable. It runs the real prefill path with:

```sh
DS4_METAL_PREFILL_CHUNK=16384 DS4_FLASH_MOE_PREFETCH=3
```

Then it feeds the fresh 8K dedup histogram into the existing scheduler
simulation using measured DS4 quantized GPU expert latencies and cached-BF16
AMX latencies. This is still not production AMX integration; it is the
best-case feasibility test before wiring a real AMX backend into DeDup.

Fresh 8K result:

```text
real pinned GPU prefill: 139.35 t/s
Flash-MoE refs=2,173,134 unique=18,869 reuse=115.17x
```

Best AMX scheduler cases:

```text
oracle-preload top32/layer, AMX B16-127:
  speedup=1.053x, cache=64.50 GiB, fills=0
  AMX refs=41,280, GPU refs=2,131,854

oracle-preload top16/layer, AMX B16-127:
  speedup=1.030x, cache=32.25 GiB, fills=0

dynamic-fill all experts, AMX B16-127:
  speedup=0.654x, fills=10,528, fill_ms=10,106.9
```

Conclusion for this system and this repo's active `4096 x 2048` Flash-MoE
shape: AMX is not worth wiring for 8K prefill unless dense BF16 expert weights
are already preloaded. Even the oracle-preloaded top32/layer case is only a
5.3% expert-MLP scheduling win while requiring ~64.5 GiB of dense AMX cache,
and real end-to-end prefill would see less than that because attention,
routing, staging, gather, and scatter remain outside the AMX win. Dynamic
materialization is clearly negative.

### Split-3 ANE materialization producer test

Added `--backend anepackgpu` to `moe-batch-bench/moe-batch-bench`. It expands
the quantized expert bank into the split-3 packed fp16 ANE input:

```text
[x fp16 | gate_0..2 [H,I/3] | up_0..2 [H,I/3] | down_0..2 [I/3,H]]
```

For the full DSv4 expert shape (`H=7168`, `I=18432`, `T=3`, `B=128`) on this
m3u host:

```text
CPU anepack:     1624.503 ms  (~0.49 GB/s packed-output rate)
Metal anepackgpu:   8.393 ms  (~94.7 GB/s packed-output rate)
ANE eval B128:     26.0-26.3 ms
```

The isolated producer+ANE overlap is good:

```text
solo_pack_wall_s=2.683 pack_ms=8.362
solo_ane_wall_s=2.957 ane_ms=25.970
concurrent_wall_s=3.011 pack_concurrent_ms=8.403 ane_concurrent_ms=26.511
overlap_speedup_vs_sequential=1.873
efficiency_vs_ideal_max_wall=0.982
```

But the real 8K prefill test shows scheduling contention with the current GPU
path:

```text
unpaced GPU pack + ANE worker:
  baseline_prefill_tps=147.79
  concurrent_prefill_tps=84.33
  gpu_pack_ms=9.884
  ane_eval_ms=26.562

paced GPU pack + ANE worker, PACK_PACE_US=18000:
  baseline_prefill_tps=149.54
  concurrent_prefill_tps=121.49
  gpu_pack_effective_ms=33.016
  ane_eval_ms=26.364
```

Conclusion: CPU materialization is impossible for production. Metal
materialization is fast enough in isolation and overlaps with ANE eval, but it
still steals enough GPU time from the current pinned prefill path to lose about
19% prefill throughput even when paced to roughly one pack per ANE eval.
Production ANE integration therefore needs a scheduler that only materializes
high-value expert batches, reuses cached packed inputs when possible, and keeps
the producer duty cycle below the point where it harms the main quantized GPU
prefill path.

## 2026-05-19/20 Update: Async Pipeline Engagement and the Hard Ceiling

Full per-call cost decomposition and the multi-ctx pool experiment are in
Appendix C of `DSv4_MLP_ANE_Matmul_Investigation.md`. Headline findings
relevant to this task's engine-selection question on M5 Max:

- **Per-call ANE wall at H=4096 I=2048 B=256** decomposes as 24% weight
  upload + 4% output read + 72% evaluate (1.65 ms total). Apple's framework
  hides the weight-upload cost when 4 ctxs run concurrently (per-call drops
  to 1.12 ms, 32% lower) but the ANE silicon itself is the single bottleneck
  — pipelining beyond the framework's natural 1.48× is not available without
  recompiling the CoreML model.

- **ANE silicon is genuinely slower per ref than the GPU MPP int8 + fp32
  fallback paths** for the DeepSeek-V3 routed expert MLP at int8. Forcing
  more refs to ANE via `DS4_FLASH_MOE_SCHED_ANE_REL_SPEED` (cap was raised
  from 4.0 to 1024.0 in ds4.c so the env actually takes effect) drops
  throughput monotonically: 186.50 t/s at 57% ANE share → 179.49 t/s at
  69% ANE share. The engine-selection policy "send everything that fits in
  ANE_MIN_REFS to ANE" is therefore wrong on this hardware/workload — the
  existing `build_flash_prefill_overlap_plan` cost-balancing assignment is
  approximately optimal.

- **The right engine for small expert groups is fp32 GPU MoE**, not ANE.
  The default scheduler already routes them there as long as
  `concurrent_prefill` is enabled. The pre-existing
  `concurrent_prefill = try_ane_prefill && try_mpp_int8_prefill && …` gate
  was too restrictive — relaxed to drop the MPP requirement so ANE-only
  configurations can use the same dispatch logic and pick up the same
  routing decisions.

- **The async pipeline is dead code in pure ANE-only mode unless
  `DS4_FLASH_MOE_ANE_I8I8_PREFILL=1` is set explicitly.** Without it
  `ane_start_tensor` silently returns NULL on every call (the `i8i8_enabled`
  helper falls back to reading the disabled MPP env) and work cascades
  through to fp32 GPU. Add `DS4_FLASH_MOE_TRACE_DISPATCH=1` for per-group
  dispatch tracing when diagnosing this kind of fall-through.

- **Multi-ctx pool**: tested, reverted. pool=2 deadlocks (3-slot transient
  peak in the scheduler); pool=4 regresses combined by 16% (memory pressure)
  and deadlocks ANE-only async (concurrent completion handlers + shared GPU
  command buffer queue). The standalone bench's 1.48× speedup doesn't
  translate to production.

- **Final landscape on coding_8k.txt:** GPU-only ~199 t/s, Combined default
  ~187 t/s, ANE-heavy async hybrid 179–186 t/s, ANE-only sync ~134 t/s.
  The default combined mode is the production sweet spot.

### 2026-05-23 agent/Flash-MoE instrumentation notes

- `ds4-agent` now accepts Flash-MoE sidecar options directly:
  `--moe-sidecar PATH`, `--moe-mode slot-bank`, and `--moe-slot-bank N`.
  This lets the native agent exercise the same memory-resident sidecar path as
  the CLI without an HTTP hop.
- Large `DS4_METAL_PREFILL_CHUNK` / raw-cap runs do not naturally emit frequent
  durable progress, because a whole chunk is processed before the KV checkpoint
  boundary is safe. The Metal prefill path now sends display-only
  `prefill_display` callbacks after each completed layer in the chunk, while
  keeping durable `prefill_chunk` callbacks only at true chunk boundaries.
- The agent footer consumes both callbacks. Its percentage and prefill t/s keep
  moving inside one large Flash-MoE chunk, but session save/resume accounting
  still uses the durable chunk boundary.
- Flash-MoE slot-bank shutdown stats now include actual resident slot
  occupancy: used/total slots, percent resident, and average/min/max slots per
  layer. This distinguishes configured slot-bank capacity from slots actually
  populated by the workload.
- `DS4_FLASH_MOE_ANE_THREADS=2` is the explicit routed-ANE prefill worker-count
  flag. If it is unset, `DS4_FLASH_MOE_ANE_DUAL=1` implies two workers.
  This does not split `DS4_FLASH_MOE_ANE_SHARED_EXPERT=1`: the shared-expert
  path currently starts one async worker for the shared-expert job and has no
  separate shared-expert thread-count flag.
- `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1` is ignored while routed ANE prefill is
  active unless `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1` or
  `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_WITH_ROUTED=1` is set. The default avoids a
  known serial dependency boundary between attention output projection and FFN.
