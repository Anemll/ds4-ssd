# MoE Batch Benchmark

Standalone per-expert batch benchmark for the routed MoE MLP shape.

The benchmark has two kinds of backend:

- Dense proxy backends measure a FP16, BF16, or FP32 approximation of one routed
  expert:

```text
gate = x @ W_gate
up   = x @ W_up
mid  = silu(gate) * up
out  = mid @ W_down
```

- `gpu_ds4_moe_block` calls the repo's actual Metal routed expert block as one
  timed operation, using `IQ2_XXS` gate/up banks, a `Q2_K` down bank, F32
  activations/output, and the current F32/F16 mid behavior from
  `ds4_gpu_routed_moe_expert_banked_batch_tensor`.

Default dimensions match this repo's fixed DS4 Flash shape:

```text
x:      [B, 4096]
gate:   [4096, 2048]
up:     [4096, 2048]
down:   [2048, 4096]
out:    [B, 4096]
```

Backends:

- `gpu_mps_matmul`: Metal Performance Shaders FP16/FP32 matrix multiplication.
- `gpu_mpsgraph_mlp`: MPSGraph BF16 MLP graph. `MPSMatrixMultiplication`
  exposes `MPSDataTypeBFloat16` in headers but rejects BF16 at runtime on this
  system, so BF16 GPU uses MPSGraph.
- `amx_bnns_matmul`: BNNS FP16/BF16 matrix multiplication through Accelerate.
- `amx_bnns_quant`: synthetic `IQ2_XXS` gate/up and `Q2_K` down weights
  dequantized to BF16 for BNNS. `cached` excludes dequant from the timed loop,
  `dequant` includes scalar CPU dequant each iteration, and `gpudequant`
  materializes BF16 weights with Metal each iteration before BNNS.
- `gpu_ds4_moe_block`: actual DS4 Metal quantized routed expert block.
- `qhybrid_ds4gpu_amx_*`: actual DS4 Metal quantized GPU expert compute run
  concurrently with BNNS/AMX BF16 expert compute. `cached` uses weights
  materialized by GPU before timing; `bgdequant` ping-pongs two AMX weight slots
  and runs GPU materialization for the next AMX expert in the same timed
  interval.

Build:

```sh
make moe-batch-bench
```

Run:

```sh
./moe-batch-bench/moe-batch-bench --batches 1,2,4,8,16,32,64,128 --iters 20
```

Run dense BF16 GPU/AMX:

```sh
./moe-batch-bench/moe-batch-bench --backend both --dtype bf16 --batches 1,2,4,8,16,32,64,128 --iters 20
```

Run dense FP32 GPU/AMX:

```sh
./moe-batch-bench/moe-batch-bench --backend both --dtype fp32 --batches 1,2,4,8,16,32,64,128 --iters 20
```

Run synchronized BF16 split mode, with GPU batch swept and AMX fixed. This
uses two different experts and also reports a GPU-sequential different-expert
baseline:

```sh
./moe-batch-bench/moe-batch-bench --backend split --split-amx-batch 32 --batches 128,192,256 --iters 120
```

Run quantized AMX cache sensitivity:

```sh
./moe-batch-bench/moe-batch-bench --backend amxq --amxq-mode both --batches 16,32 --iters 20
./moe-batch-bench/moe-batch-bench --backend amxq --amxq-mode gpudequant --batches 16,32 --iters 20
```

Run quantized GPU + AMX concurrent split with background GPU dequant:

```sh
./moe-batch-bench/moe-batch-bench --backend qhybrid --qhybrid-mode both --split-amx-batch 32 --batches 96,128,160,192,224,256 --iters 80
```

Run the actual DS4 Metal MoE block:

```sh
./moe-batch-bench/moe-batch-bench --backend ds4 --batches 1,2,4,8,16,32,64,128 --iters 20
```

Run synthetic packed-weight Metal kernels. This stores the three expert matrices
as either nibble-packed FP4 with a 16-entry LUT or signed int8 with a fixed scale,
then expands inside simple direct matmul kernels:

```sh
./moe-batch-bench/moe-batch-bench --backend gpuqsynthetic --batches 1,2,4,8,16,32,64,128 --iters 10
```

Build and run the private-API ANE W-as-input MLP benchmark:

```sh
make ane-mlp-bench
./moe-batch-bench/ane_ds4_mlp_inmem_bench -bench-shape 4096 2048 1,8,16,32,64,128 5 30
./moe-batch-bench/ane_ds4_mlp_inmem_bench -verify 4096 2048 8 /tmp/ds4_ane_verify_4096_8
./moe-batch-bench/verify_ane_dump_numpy.py /tmp/ds4_ane_verify_4096_8
```

Run the packed-input ANE variant, which collapses activation and all three
weights into one 4D function input before slicing in MIL:

```sh
./moe-batch-bench/ane_ds4_mlp_inmem_bench_packed -bench-shape 4096 2048 1,8,16,32,64,128 5 30
./moe-batch-bench/ane_ds4_mlp_inmem_bench_packed -bench-shape 7168 18432 1,8,16,32 3 20
```

Run the full-I split packed ANE variant for the larger DSv4 expert shape. This
splits `I=18432` into three `6144`-wide streams and is the candidate for m3u/M5
full-shape ANE prefill work:

```sh
./moe-batch-bench/ane_ds4_mlp_inmem_bench_packed_split3
```

Write CSV:

```sh
./moe-batch-bench/moe-batch-bench --csv moe-batch-bench/m4_max_fp16.csv
```

Notes:

- The dense proxy is not the current quantized Metal dedup kernel.
- The DS4 quantized backend currently has no BF16 activation input mode; it uses
  the production F32 activation contract and current F32/F16 mid behavior.
- The DS4 backend does not include router top-k, dedup histogram/compact, SSD
  sidecar staging, or scatter-add back to token rows. It measures the per-expert
  MLP block after dedup has already produced a single expert's token list.
- Prompt-level per-expert dedup histograms can be captured from the real runner
  with `DS4_FLASH_MOE_HIST_CSV=path.csv`; summarize them with
  `python3 moe-batch-bench/analyze_prompt_hist.py path.csv --qhybrid moe-batch-bench/qhybrid_amx16_seq.csv moe-batch-bench/qhybrid_amx32_seq.csv`.
- The prompt histogram suite for the local DS4 Flash sidecar can be run with
  `moe-batch-bench/run_prompt_hist_suite.sh`.
- A pinned 8K AMX feasibility check can be run with
  `moe-batch-bench/run_8k_amx_feasibility.sh`. It uses
  `DS4_METAL_PREFILL_CHUNK=16384 DS4_FLASH_MOE_PREFETCH=3`, captures a fresh
  8K histogram, and runs the cached-AMX scheduler simulation.
- A process-level ANE+GPU overlap probe can be run with
  `moe-batch-bench/run_ane_gpu_concurrency_probe.py`. It compares solo
  quantized GPU expert batches, solo split-3 ANE full-shape MLP, and both
  running concurrently. Large GPU batches can be tested with, for example:
  `./moe-batch-bench/run_ane_gpu_concurrency_probe.py --gpu-batch 8192 --ane-batch 128 --gpu-iters 80 --ane-iters 40`.
- Multi-ANE process behavior can be checked with
  `moe-batch-bench/run_ane_multi_probe.py`. On this m3u host, multiple ANE jobs
  mostly serialize/partition the same ANE resource, so one larger ANE batch is
  preferable to many concurrent ANE batches.
- Real production prefill coexistence with an ANE worker can be checked with
  `moe-batch-bench/run_real_prefill_ane_concurrency.sh`. It runs pinned 8K
  prefill alone, then pinned 8K prefill while a split-3 ANE B128 stream runs in
  parallel.
- Split-3 ANE input materialization can be measured with
  `./moe-batch-bench/moe-batch-bench --backend anepackgpu --in 7168 --mid 18432 --batches 128`.
  The CPU reference is `--backend anepack`. On this m3u host, B128 CPU
  materialization is about 1.62 s, while the Metal producer is about 8.4 ms for
  the 757.8 MB packed fp16 input.
- Real production prefill coexistence with both a split-3 ANE worker and the
  Metal split-pack producer can be checked with
  `moe-batch-bench/run_real_prefill_ane_pack_concurrency.sh`. The producer is
  paced by `PACK_PACE_US` (default `18000`) so it approximates one pack per ANE
  eval instead of an unrealistic continuous GPU stress loop.
- Dense-cache scheduling estimates can be generated from prompt histograms and
  latency CSVs with `moe-batch-bench/simulate_prefill_scheduler.py`.
- Real prefill t/s is sensitive to the transient prefill bank pipeline. Current
  `ds4` rotates all four prefill banks and skips duplicate staging when a
  prefetched bank already holds the required expert.
- `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=N` is an opt-in negative-result
  experiment for using the decode slot bank during prefill. On this system,
  `top32` was slower on 4K because the hit rate was too low to pay for the
  required synchronization.
- The FP4 reference file
  `/Volumes/SN8100/DS/DeepSeek-V4-Flash-FP4-FP8-native.gguf` uses GGUF tensor
  type `39` (`MXFP4` in the local gguf constants) for routed expert weights.
  The synthetic `gpuqsynthetic` backend is not a decoder for that exact MXFP4
  block format; it is a controlled packed-read experiment for FP4-LUT vs int8
  fixed-point in-kernel expansion.
- The ANE W-as-input path is integrated as a separate benchmark, not production
  routing. On this system the repo's `4096 x 2048` MLP shape compiles and runs
  from B1 through B128, but the full DSv4 `7168 x 18432` monolithic MLP and the
  `18432 x 7168` down-projection W-input matmul currently fail ANE compilation.
- The packed-input ANE path also builds as part of `make ane-mlp-bench`. On this
  m3u system it compiles smaller/intermediate shapes but still fails the full
  DSv4 `7168 x 18432` shape; `I=16384` compiles while `I=18432` fails even with
  smaller `H`, so this appears dimension-sensitive rather than simply a packed
  input byte limit.
- The split-3 packed ANE path also builds as part of `make ane-mlp-bench` and
  does compile/evaluate the full `7168 x 18432` shape on this m3u host. Local
  observed floor was about 13.1 ms/iter through B64 with a 756 MB packed input.
- `DS4_FLASH_MOE_ANE_PREFILL=1` is wired only as a guarded production hook with
  fallback today. The ANE worker still needs the quantized-bank to fp16
  split-packed materializer and IOSurface handoff before it can replace the
  Metal GPU expert batch path.
- Prompt-level AMX/ANE integration measurements must pin
  `DS4_METAL_PREFILL_CHUNK=16384 DS4_FLASH_MOE_PREFETCH=3`. Smaller chunks
  create too many expert batches and understate the fused GPU path's real
  multi-token dedup efficiency.
