# M1 Max routed-MoE prefill profile — ANE vs GPU/ALU

M1 Max is **single-cluster ANE** and has **no NAX** (matmul2d is M5+ only), so the
GPU baseline here is plain **`mul_mm_id` (ALU)** — not nax_int8 (M5) and not the
dual-cluster ANE story (M3 Ultra). 64 GB → **streaming only** (dense GGUF +
expert sidecar, `--moe-mode slot-bank`); the 81 G IQ2_XXS model can't be resident.

**Verdict (measured 2026-06-02): streaming routed-MoE prefill stays on GPU/ALU.
Single-cluster M1 ANE never beats `mul_mm_id` at any config we tried.** The profile
entry (`ds4_profile.json` → "Apple M1 Max") sets `DS4_FLASH_MOE_ANE_PREFILL=0`.

## Setup

- Box: Apple M1 Max, 64 GB. Model: `dsv4-iq2xxs-expert-major` (dense + sidecar).
- Streaming: `--moe-mode slot-bank --moe-slot-bank 96`, `DS4_METAL_PREFILL_CHUNK=16384`.
- Prompts: `tests/test-vectors/prompts/coding/coding_*.txt` (pulled from m3u;
  `coding_14k` truncated from 16k). Single gen token.
- Single-cluster ANE knobs: `DS4_FLASH_MOE_ANE_DUAL=0 THREADS=1`, shared-expert +
  O-proj ANE off (per PROFILE_NEW_MACHINE.md §1).
- Scripts: `run_{gpu,ane}_prefill_profile_M1MAX64.sh`,
  `moe-batch-bench/run_prefill_chunk_minrefs_sweep_M1MAX64.sh`,
  `moe-batch-bench/sweep_ane_knobs_M1MAX64.sh`.
- **Hot single runs unless noted** — M1 prefill is thermally sensitive; the only
  trustworthy comparisons are cooldown'd or thermally-matched (see §Thermal traps).

## Segment A — fixed chunk 16K, prompt sweep (tiled kernel, default MAX_REFS=256)

GPU = `mul_mm_id` (ALU). Best ANE = best over min_refs {32,64,128,256,512}.

| prompt | GPU | best ANE | min_refs | winner | note |
|-------:|----:|---------:|---------:|:-------|:-----|
| 1K  | 57.32 | 57.33 | 256 | tie | **ANE never engages at 1K** (ane_refs=0 at every min_refs) |
| 4K  | 87.34 | 89.02 | 512 | ANE +1.9% | within thermal noise |
| 8K  | 94.36 | 89.37 | 512 | GPU +5.3% | |
| 10K | 87.37 | 85.12 | 512 | GPU +2.6% | |
| 12K | 89.11 | 86.64 | 512 | GPU +2.8% | |
| 14K | 87.95 | 85.14 | 512 | GPU +3.2% | |
| 16K | 87.85 | 85.32 | 512 | GPU +2.9% | |

Best min_refs is **high (512)** — opposite of M5/M3U's 32–64. On weak single-cluster
M1 ANE you want to send *less* to the ANE.

## Segment B — 8K prompt, chunk sweep (tiled kernel, default MAX_REFS=256)

| chunk | GPU | best ANE | min_refs | winner |
|------:|----:|---------:|---------:|:-------|
| 1024  | 54.37 | 56.48 | 32  | ANE +3.9% |
| 2048  | 70.57 | 72.04 | 32  | ANE +2.1% |
| 4096  | 83.42 | 83.29 | 512 | tie −0.2% |
| 8192  | 88.61 | 82.71 | 64  | GPU +6.7% |
| 16384 | 92.74 | 75.20 | 512 | GPU +18.9% |

Crossover ≈ 4096: ANE wins *small* chunks (biggest SSD I/O bubble to hide behind),
GPU wins large. But **absolute throughput rises with chunk** (GPU 54→70→83→89→**92.7**),
so the production 16384 chunk is firmly GPU; small-chunk ANE wins are at far lower
absolute t/s and not worth dropping the chunk for.

## The knob sweep, and the trap it exposed (8K, chunk 16384)

OFAT over 18 knobs. **Two apparent "winners" were artifacts:**

| variant | t/s | sum ane_refs | real? |
|:--------|----:|-------------:|:------|
| fp16x_i8w  | 93.35 | **0** | ❌ ANE disabled → secret GPU |
| full_fused | 92.45 | **0** | ❌ ANE disabled → secret GPU |
| **maxrefs512** | **90.80** | 919,003 | ✅ real ANE |
| threads2   | 83.99 | 934,861 | ✅ |
| base (MAX_REFS=256) | 74.22 | 946,014 | ✅ |
| overlap_off | 46.22 | 2,173,134 | ✅ (no overlap → slow) |

**Routed-ANE prefill only works on the tiled-fused kernel.** `fp16x_i8w` and
`full_fused` were configured with `I8I8_TILED_FUSED_PREFILL=0`, which **silently
disables routed-ANE** — the `routed experts = ANE` header still prints, but the
per-layer split routes `ane_refs=0` and everything falls back to GPU. Always verify
`sum ane_refs > 0`.

The real lever is **`MAX_REFS` (ANE call batch size)**: tiled + `MAX_REFS=512` →
90.80 vs base `MAX_REFS=256` → 74.22. Bigger ANE calls amortize better.

## Padding is not the bottleneck (8K, min_refs=0)

Default B=256 wastes ~32% on padding (`pad_util` 67.7%). Multi-bucket batch
schedules cut the waste but **didn't raise t/s**:

| config | t/s | pad_util |
|:-------|----:|---------:|
| gpu_alu | **94.20** | — |
| b256 (default) | 83.76 | 67.7% |
| b128,256 | 84.47 | 76.1% |
| b64,128,256 | 84.20 | 79.5% |
| b64,128 / max128 | 77.01 | 85.5% |
| **b256,512 / max512** | **86.12** | 64.2% |

The *most*-padded config (max512, 64.2% util) is the **fastest**; the *least*-padded
(max128, 85.5%) is the **slowest**. **Throughput-bound, not padding-bound** — bigger
ANE calls win despite more padding.

## Overlap is already near-optimal

The ANE/GPU overlap scheduler (`OVERLAP_PREFILL=1 OVERLAP_SCHEDULER=1
ANE_PIPELINE_PREFILL=1`, all on by default in the ANE script) emits a per-layer plan:

```
overlap plan layer=0: planned_ane_groups=153 planned_gpu_groups=103
  est_gpu=25017 est_ane=24982 est_idle=0.14% overlap_gpu_groups=103
```

`est_idle ≈ 0.14–0.61%` at 8K — the GPU tail already runs concurrently with ANE,
near-zero idle. The `gpu_overlap=0` seen at 1K is only because ANE is idle there
(nothing to overlap), not a missing-overlap bug. **No overlap headroom to exploit.**

## Decisive test — best ANE vs thermally-matched cool GPU (8K, 25s settle)

| config | t/s | ane_refs |
|:-------|----:|---------:|
| **GPU (cool)** | **94.31** | — |
| ANE max512, min_refs 128 | 88.83 | 1.53M |
| ANE max512, min_refs 256 | 88.93 | 1.32M |
| ANE max512, min_refs 512 (best) | 90.63 | 907K |

Best-tuned ANE **90.63 < cool GPU 94.31 (GPU +4%)**. The earlier "+8% ANE win"
(90.80 vs 83.91) was a **thermal artifact** — measured against a heat-soaked GPU
baseline (83.91); the real cool GPU is 94.31.

## Sustained load (6 prefills back-to-back, no cooldown, 8K)

| run | 1 | 2 | 3 | 4 | 5 | 6 | mean |
|:----|--:|--:|--:|--:|--:|--:|-----:|
| GPU-only | 94.11 | 90.52 | 91.26 | 94.05 | 94.32 | 94.18 | ~93.1 |
| ANE (best) | 89.97 | 90.46 | 91.13 | 90.61 | 91.02 | 90.86 | ~90.7 |

**The GPU does not throttle under realistic sustained load** — it holds ~94 across 6
back-to-back runs. GPU wins sustained too (+3.6%). (The deep 94→84 droop seen
elsewhere only appears after ~80+ runs of accumulated soak, not normal use.)

## Thermal stress / power

ANE *does* lower GPU thermal/power load — the overlap plan splits routed-expert
matmul ~50/50 (`est_gpu ≈ est_ane`), so the GPU runs at roughly half utilization
with the rest on the more perf/watt-efficient ANE. But this buys **no throughput**
(GPU doesn't throttle), so it only matters for power/heat-sensitive scenarios
(fanless/quiet, battery, or GPU contended by other work). Direct GPU-vs-ANE watts:
`sudo powermetrics --samplers gpu_power,ane_power,thermal`.

## Memory budget — slots × ctx on 64 GB (slots=64 default)

**The slot count must fit RAM alongside the dense model + KV cache.** On 64 GB:

| component | approx |
|:----------|-------:|
| dense model | ~8.8 GB |
| 96-slot expert bank | ~28 GB |
| 64-slot expert bank | ~19 GB |
| KV cache | grows with ctx (DS4_CTX_GROW) |

`--moe-slot-bank 96` + `--ctx 100000` **over-commits 64 GB and swaps** — MEASURED:
prefill collapsed to **2.2 t/s** (≈40× slow) on a 119-token prompt purely from swap
thrash, not compute. Dropping to **`--moe-slot-bank 64` restored normal speed**.

So the M1 Max profile and `*_M1MAX64.sh` scripts default to **slots=64**. Raise
slots only at small ctx. (The recorded prefill benchmarks above used slots=96 at
small ctx ≤20K, which fits — so those throughput numbers are unaffected.) If a real
ctx-100K prompt still pressures RAM at slots=64, drop ctx or slots further and watch
`memory_pressure` / the `Flash-MoE slot-bank stats … installed=… MiB` line.

## Profile decision (ds4_profile.json → "Apple M1 Max")

**Streaming `sidecar_env`: GPU/ALU, ANE off.**

```jsonc
"DS4_FLASH_MOE_ANE_PREFILL": "0",      // GPU/ALU (mul_mm_id); no NAX on M1
"DS4_METAL_PREFILL_CHUNK": "16384",
"DS4_FLASH_MOE_SLOT_BANK_SLOTS": "64",   // 96 + large ctx OOMs on 64 GB (see Memory budget)
"DS4_CTX_GROW_BLOCK": "2048",          // conservative for 64 GB single-die
// + I/O: PREFETCH=3 ASYNC_PREAD=1 ASYNC_PREAD_AFTER_STAGE=1 PREAD_THREADS=6 ASYNC_READAHEAD=12
```

To A/B ANE on: `export DS4_FLASH_MOE_ANE_PREFILL=1` (+ tiled-fused kernel,
`DS4_FLASH_MOE_ANE_MAX_REFS=512`, `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=512`).

## Traps that cost real time (all hit during this work)

- **`I8I8_TILED_FUSED=0` silently disables routed-ANE** (ane_refs=0 = secret GPU run).
  Routed-ANE prefill only works on the tiled-fused kernel. Always assert
  `sum ane_refs > 0` per run before trusting an "ANE" number.
- **Thermal artifacts** — a hot GPU baseline (83.91) made ANE look +8%; the cool
  GPU is 94.31 and ANE actually loses. Cooldown or thermally-match every A/B.
- **Throughput-bound, not padding-bound** — reducing ANE padding (smaller batch
  buckets) does not help; bigger `MAX_REFS` wins despite more padding.
- **1K prompts never engage ANE** — no expert group reaches the min_refs threshold,
  so all "1K ANE" rows are pure GPU. Exclude from ANE judgments.
- **`ane_calls` ≠ `ane_refs`** — calls falls with min_refs (fewer, bigger calls);
  total refs routed is the real engagement metric. Mid-run logs under-report refs.
- **Overlap is already on** — `gpu_overlap=0` at low load means ANE is idle, not
  that overlap is broken.

## vs M5 Max / M3 Ultra

| | M1 Max (64G) | M5 Max | M3 Ultra |
|---|---|---|---|
| NAX (matmul2d) | no | yes | no |
| ANE clusters | 1 | 1 | 2 |
| GPU baseline | mul_mm_id (ALU) | nax_int8 | mul_mm_id (ALU) |
| streaming routed prefill | **GPU/ALU** (ANE loses) | ANE wins (+5–15%) | ANE wins (+30–40%) |

Same single-cluster *shape* as M5, but M1's GPU/ALU baseline is strong enough
relative to its weaker ANE that ANE never gets ahead — unlike M5 (where ANE beats
nax_int8) or dual-cluster M3U (where aggregate ANE throughput ~doubles).
