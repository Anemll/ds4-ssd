# MXFP4 Mixed-Residency Decode Plan

Date: 2026-06-12
Branch: `MXFP4`
Goal: keep the high-memory Flash-MoE cache residency benefits without paying
the large-bank decode dispatch cost.

## Current Evidence

The previous warm 90 GB cliff investigation split the problem into two costs:

- Cold miss cost: true SSD reads are slow when the wired bank evicts the OS
  page cache. This is real, but it is not the warm-big-bank mechanism.
- Warm dispatch cost: the old mixed per-layer bank bound a huge layer buffer
  during `routed_moe`. At 90 GB that made decode scale with bank size.

The first fix, committed as `eb386fe Auto-split large Flash-MoE slot banks`,
avoids the catastrophic warm cliff by automatically switching large banks
(>=44 GiB capacity) to per-slot expert buffers.

Observed after that fix:

| config | ctx | slots | layout | prefill | generation | note |
|---|---:|---:|---|---:|---:|---|
| `--ssd-cache 90GB` | 4096 | 168 | `per-slot-auto` | 7.38 t/s | 5.95 t/s | target met |
| `--ssd-cache 20GB` | 4096 | 37 | `slot-bank` | 7.46 t/s | 7.62 t/s | small-bank smoke |
| `--ssd-cache 90GB` | 32768 | 168 | `per-slot-auto` | 5.02 t/s | 4.53 t/s | user pasted run |
| `--ssd-cache 32GB` | 32768 | 59 | `slot-bank` | 4.93 t/s | 12.71 t/s | user pasted run |

Conclusion: the 90 GB auto-split path is now safe, but it is still slower than
a moderate mixed bank. Higher memory improves residency, but the per-slot MXFP4
decode path is not yet as efficient as the mixed-bank grouped path.

## First-Principles Model

For one decode token, each layer needs only the active routed experts, normally
6 experts. The high-level cost model is:

```text
token_time ~= route + active_expert_compute + miss_install + dispatch_binding
```

A good high-memory design should make:

- cache capacity scale with memory budget;
- miss installs decrease as residency improves;
- dispatch binding scale with active experts, not total bank capacity;
- kernel work scale with active experts, not total bank capacity;
- data movement stay below the saved miss-read cost.

The bad design binds the full layer bank, so dispatch_binding scales with
`slot_bank * expert_stride`. The safe-but-not-fast design binds only active
per-slot buffers, but MXFP4 currently appears to use per-route fallback kernels.

## Candidate Designs

### A. MXFP4 slots6 grouped path

Keep the high-memory per-slot cache. Add grouped six-buffer MXFP4 decode kernels
so each layer binds only the six active expert buffers but computes them in a
single grouped path rather than six per-route fallbacks.

Expected win:

- preserves 90 GB residency;
- dispatch footprint scales with 6 active buffers;
- no staging copy;
- should close much of the 90 GB per-slot vs 32 GB mixed gap.

Risk:

- requires Metal kernel and host pipeline work;
- may still trail mixed-bank if binding 18 small buffers per layer is expensive;
- needs exactness checks because MXFP4 block geometry differs from QK_K.

### B. Two-tier L1/L2 cache

Keep a fast mixed L1 bank, e.g. 32 GB, plus a large per-slot L2 cache using the
rest of the memory budget. Decode always runs from L1. If an active expert is
only in L2, promote/copy it into L1 instead of reading SSD.

Expected win:

- decode stays on the fastest mixed-bank kernel path;
- high-memory residency still reduces SSD misses;
- promotion from L2 is memory bandwidth, not storage bandwidth.

Risk:

- significantly more cache policy complexity;
- L1 slot churn can still cost if routing has high layer-local diversity;
- requires correctness around slot mapping, age, replay invalidation, and
  prefetch interactions.

### C. Active staging bank

Keep 90 GB per-slot storage, copy only the active six experts per layer into a
tiny mixed staging bank, then run existing mixed-bank decode kernels.

Expected win:

- reuses current fast mixed kernels;
- active dispatch footprint is tiny.

Risk:

- copy volume is about `6 * 43 * 12.75 MiB = 3.2 GiB/token` before accounting
  for gate/up/down family details;
- may trade driver overhead for memory bandwidth overhead;
- requires staging lifetime and synchronization care.

### D. Chunked mixed banks

Split each layer's mixed bank into smaller mixed buffers, then dispatch only
the chunks containing active experts.

Expected win:

- keeps mixed layout within chunks;
- reduces max buffer binding size.

Risk:

- active experts can span many chunks;
- host routing and selected-slot remapping get more complex;
- still binds inactive experts inside each selected chunk.

## Step-by-Step Test Plan

1. Source audit: prove which path MXFP4 per-slot decode actually takes.
2. Baseline profile at `--ctx 32768` for:
   - 90 GB `per-slot-auto`;
   - 32 GB mixed `slot-bank`.
3. If 90 GB per-slot spends routed time in per-route fallback, implement A
   first: MXFP4 `slots6` grouped gate/up/down path.
4. Compare:
   - 90 GB before/after MXFP4 slots6 grouped;
   - 32 GB mixed reference;
   - 90 GB legacy mixed only if needed with
     `DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1`.
5. If MXFP4 slots6 does not close the gap, prototype B or C with a narrow
   env-gated diagnostic before any production default.

Bench hygiene:

- one run at a time;
- cooldown between decode benches;
- keep prompt, ctx, `-n`, and temperature fixed inside each comparison set;
- record layout line, residency stats, stage profile when enabled, prefill t/s,
  and generation t/s;
- append results here instead of overwriting history.

## Progress Log

### 2026-06-12 - Initial plan

- Created this plan from the post-fix observation that `--ssd-cache 32GB`
  mixed-bank decode can outperform `--ssd-cache 90GB` per-slot-auto decode at
  long context.
- Current leading hypothesis: the high-memory path is slow because MXFP4
  per-slot decode lacks a grouped `slots6` path and falls back to per-route
  dispatches.

### 2026-06-12 - Step 1 source audit

Confirmed from source that MXFP4 cannot enter the grouped per-slot `slots6`
path yet:

- `ds4.c:metal_graph_flash_moe_compute_independent_slots6_grouped` returns
  false unless gate is `DS4_TENSOR_IQ2_XXS` and down is `DS4_TENSOR_Q2_K`.
  MXFP4/MXFP4 is excluded before the slot-buffer views are passed to Metal.
- `ds4_metal.m:ds4_gpu_routed_moe_one_slots6_tensor` independently returns
  false unless gate is `DS4_METAL_TENSOR_IQ2_XXS`, down is
  `DS4_METAL_TENSOR_Q2_K`, and the IQ2/Q2 slots6 pipelines exist.
- `metal/moe.metal` has `kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32`,
  `kernel_mul_mv_slots6_q2_K_f32`, and `kernel_mul_mv_slots6_q2_K_sum6_f32`,
  but no MXFP4 `slots6` kernels.

Implication: the 90 GB `per-slot-auto` MXFP4 path avoids the giant buffer bind,
but it must use the per-route fallback. Candidate A remains the first
implementation target unless profiling contradicts this.

### 2026-06-12 - Step 2 controlled baselines

Commands used the same prompt, context, and deterministic decode:

```bash
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache <SIZE> --ctx 32768 -n <N> --temp 0 \
  -p "What is Apple Neural Engine"
```

Profiling runs (`DS4_METAL_DECODE_STAGE_PROFILE=1`, `-n 16`) are useful for
stage attribution but perturb throughput heavily:

| config | layout | routed_moe avg | generation |
|---|---|---:|---:|
| 90 GB | `per-slot-auto` | 3.808 ms/layer | 3.77 t/s |
| 32 GB | `slot-bank` | 3.540 ms/layer | 3.89 t/s |

No-profile short controls (`-n 16`) show only a small layout gap:

| config | layout | tok16 residency | tok16 hit | generation |
|---|---|---:|---:|---:|
| 90 GB | `per-slot-auto` | 1763/7224 | 62.7% | 5.38 t/s |
| 32 GB | `slot-bank` | 1711/2537 | 62.6% | 5.81 t/s |

No-profile longer controls (`-n 64`) show the practical gap after warm-up:

| config | layout | tok64 residency | tok64 hit | generation |
|---|---|---:|---:|---:|
| 90 GB | `per-slot-auto` | 3779/7224 | 85.4% | 5.98 t/s |
| 32 GB | `slot-bank` | 2535/2537 | 77.3% | 7.37 t/s |

Interpretation:

- 90 GB has better cache residency and hit rate by token 64.
- 32 GB is still faster, so the remaining gap is not residency; it is the
  per-slot MXFP4 decode implementation.
- Stage profiling compresses the difference and should not be used as a
  throughput benchmark for this comparison.
- Next step: implement Candidate A behind the existing slots6 gate, adding
  MXFP4 grouped per-slot gate/up and down kernels.

### 2026-06-12 - Step 3 MXFP4 slots6 implementation

Implemented Candidate A:

- Added `kernel_mul_mv_slots6_mxfp4_pair_swiglu_f32` for six-buffer MXFP4
  gate/up plus fused SwiGLU/route weight.
- Added `kernel_mul_mv_slots6_mxfp4_f32` for six-buffer MXFP4 down projection.
- Wired Metal pipelines in `ds4_metal.m`.
- Opened `metal_graph_flash_moe_compute_independent_slots6_grouped` to
  MXFP4/MXFP4 in addition to the existing IQ2_XXS/Q2_K path.

Validation:

| test | result |
|---|---|
| build | `make ds4` passed; only pre-existing warnings plus existing unused ObjC helpers |
| path selection | 90 GB run prints `Flash-MoE separate slot buffers using grouped slots6 decode path` |
| greedy token check | grouped and fallback selected identical first 8 token IDs: `[2581, 1309, 304, 3287, 28, 582, 3085, 344]` |
| 90 GB `-n 16` | grouped slots6: 5.42 t/s generation |
| 90 GB `-n 64` | grouped slots6: 6.18 t/s generation |
| 90 GB `-n 64` fallback A/B | `DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1`: 6.02 t/s generation |
| 32 GB `-n 64` reference | mixed `slot-bank`: 7.41 t/s generation |

Interpretation:

- MXFP4 grouped slots6 is correct on the greedy smoke and gives a small win
  over fallback.
- It does not close the 90 GB per-slot-auto vs 32 GB mixed gap. The remaining
  gap is not simply "too many per-route dispatches".
- Candidate A should stay as a useful cleanup/optimization, but the next
  mixed-residency experiment should move to Candidate B or C:
  - B: mixed 32 GB L1 plus high-memory per-slot L2 promotion;
  - C: active six-expert staging into a tiny mixed bank before the existing
    mixed-bank kernel.

### 2026-06-12 - Step 4 mixed-residency probes after user pushback

User correctly pointed out that the 90 GB path was still slower than the 32 GB
bank, so the target is not merely "above 5 t/s"; it is preserving useful extra
residency without losing the 32 GB decode speed.

Additional controlled runs, current binary after Candidate A, same prompt and
deterministic decode:

| config | ctx | layout | tok64 residency | tok64 hit | generation | interpretation |
|---|---:|---|---:|---:|---:|---|
| 90 GB | 4096 | `per-slot-auto` | tok16 1763/7224 | tok16 62.7% | 5.47 t/s | matched no-staging reference |
| 90 GB + active staging prototype | 4096 | per-slot storage -> 6-slot staged bank | tok16 1763/7224 | tok16 62.7% | 5.16 t/s | negative; copy/stage loses |
| 90 GB + lazy per-slot | 4096 | `per-slot-auto-lazy` | tok16 1763/7224 | tok16 62.7% | 5.56 t/s | tiny positive, not enough |
| 90 GB + lazy per-slot | 32768 | `per-slot-auto-lazy` | 3779/7224 | 85.4% | 6.01 t/s | still behind 32 GB |
| 32 GB | 32768 | mixed `slot-bank` | 2535/2537 | 77.3% | 7.48 t/s | current-binary fast reference |
| 40 GB | 32768 | mixed `slot-bank` | 3078/3182 | 80.2% | 7.15 t/s | more hits, slower than 32 GB |

Active staging details:

- A narrow env-gated prototype copied the six active per-slot experts for each
  layer into six-slot family staging banks, remapped selected IDs to `0..5`,
  and ran the existing banked MoE path.
- It was correct on the smoke output but slower than the existing grouped
  slots6 per-slot path, so the scaffold was removed instead of committed.
- The result argues that adding a per-token copy layer is not the right way to
  recover the 32 GB speed.

Lazy per-slot details:

- `DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1` keeps the 168-slot logical capacity
  but allocates Metal buffers only for slots actually installed.
- It reduces upfront allocation (`0.0GB allocated, 89.9GB planned`) and gives a
  small short-run lift, but the long run remains ~20% slower than 32 GB while
  achieving the same 85.4% tok64 hit rate as the full 90 GB per-slot path.
- This suggests the residual cost is not only cold unused Metal allocation.

40 GB mixed-bank probe:

- 40 GB stays below the 44 GiB auto-split threshold and therefore uses the fast
  mixed expert-major bank.
- It improves tok64 hit rate over 32 GB (80.2% vs 77.3%) but still loses
  generation throughput (7.15 vs 7.48 t/s). The speed/residency optimum is
  already bending down by 40 GB on this prompt.

Updated conclusion:

- Extra 90 GB residency is real, but the saved misses do not pay for the extra
  decode overhead on this short-context workload.
- Candidate C is negative. Lazy allocation is also not sufficient.
- A production "mixed" solution likely needs a policy change, not just a
  kernel/dispatch shape change: decode should probably keep a moderate fast
  mixed bank (near 32 GB on this machine/prompt) and use additional memory only
  where it demonstrably reduces true SSD misses on longer or colder workloads.
  The next implementation should be explicit about that tradeoff, e.g. an
  auto-tuned decode bank cap or a real two-tier L1/L2 design with promotion
  only when L2 hits replace true disk reads.
