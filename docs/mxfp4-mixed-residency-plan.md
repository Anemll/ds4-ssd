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

### 2026-06-12 - Step 5 fast mixed-L1 decode cap prototype

Implemented the first policy-shaped prototype: a large explicit cache request
can now opt into a moderate mixed decode L1 without the large-bank per-slot
auto guard blocking the shrink.

Code behavior:

- If `DS4_FLASH_MOE_DECODE_SLOT_BANK=<N>` or
  `DS4_FLASH_MOE_DECODE_SSD_CACHE=<size>` requests a real post-prefill shrink,
  large-bank auto per-slot buffers are suppressed so the bank can remain mixed
  for prefill and shrink to the requested mixed decode bank.
- Added `DS4_FLASH_MOE_FAST_DECODE_L1=1` as a shorthand prototype policy. It
  defaults the decode L1 budget to 32 GB, overrideable with
  `DS4_FLASH_MOE_FAST_DECODE_SSD_CACHE=<size>` or
  `DS4_FLASH_MOE_DECODE_L1_SSD_CACHE=<size>`.
- `DS4_FLASH_MOE_DECODE_SLOT_BANK=0` still explicitly opts out and keeps the
  existing no-shrink behavior.

Validation:

```bash
DS4_FLASH_MOE_RESIDENCY_STATS=16 \
DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 64 --temp 0 \
  -p "What is Apple Neural Engine"
```

Result:

| config | prefill bank | decode bank | tok64 hit | prefill | generation |
|---|---:|---:|---:|---:|---:|
| 90 GB + decode L1 cap | 168 slots / 89.95 GiB | 59 slots / 31.59 GiB | 77.3% | 6.00 t/s | 7.51 t/s |
| 32 GB reference | 59 slots / 31.59 GiB | 59 slots / 31.59 GiB | 77.3% | 5.51 t/s | 7.48 t/s |
| 90 GB per-slot-auto | 168 slots / 89.95 GiB | 168 slots / per-slot | 85.4% | ~6 t/s | ~6.0-6.2 t/s |

Interpretation:

- This restores 32 GB-class decode speed for an explicit 90 GB request.
- It does not yet preserve the 90 GB decode hit rate; the shrink resets the
  slot cache and decode uses a 59-slot L1. This is therefore a fast-L1 policy
  prototype, not a real L2 cache.
- The next layer, if needed, is a true L2: keep the 59-slot mixed L1 for decode
  and retain/promote from a separate larger backing cache only when it replaces
  true SSD reads. Do not put the full 90 GB working set directly on the decode
  hot path.

### 2026-06-12 - Step 6 L1/L2 and six-slot baselines

User noticed that the first L1/L2 diagnostic was not really using high memory.
That was correct:

- `DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB DS4_FLASH_MOE_DECODE_L2=1`
  captured `0` resident prefill slots because the default prefill slot-cache
  policy is `slot-cache-topk=0`. The run generated 6.04 t/s, but it was an
  empty-L2 overhead test, not a full high-memory residency test.
- The CPU-L2 miss path was adjusted so an L2 miss can still read directly into
  the Metal L1 slot. That removes one avoidable scratch -> L1 copy on misses,
  but it does not solve the fundamental L2 cost.

Corrected high-RAM L1/L2 test:

```bash
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=84 \
DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB \
DS4_FLASH_MOE_DECODE_L2=1 \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 16 --temp 0 \
  -p "What is Apple Neural Engine"
```

Result:

| config | prefill bank | decode bank | L2 capture | prefill | generation |
|---|---:|---:|---:|---:|---:|
| 90 GB + 32 GB L1 + CPU L2, topk84 | 168 slots / 89.95 GiB | 59 slots / 31.59 GiB | 1951 records | 2.22 t/s | 0.22 t/s |
| same + `DECODE_PREFETCH_MAX_LOADS=6`, direct L2-backed prefetch | 168 slots / 89.95 GiB | 59 slots / 31.59 GiB | 1951 records | 2.23 t/s | 0.22 t/s |

Interpretation:

- Forcing the large bank to become a real populated L2 makes RAM use meaningful,
  but the decode path collapses. CPU-backed promotion/copy on the decode miss
  path is not viable for this workload.
- Overlapping CPU-L2 copies through the decode-prefetch machinery does not help;
  the result is still 0.22 t/s.
- A default-length run of the same config captured 2032 records and was killed
  after more than 12 minutes without reaching the final summary. That is enough
  to classify it as negative for the short workflow.

Six-slot baselines requested by the user:

| config | decode bank | reuse policy | generation |
|---|---:|---|---:|
| `DS4_FLASH_MOE_DECODE_SLOT_BANK=6 DS4_FLASH_MOE_SIX_SLOT_BASELINE=1` | 6 slots / 3.21 GiB | always reload active experts into slots 0..5 | 6.85 t/s |
| `DS4_FLASH_MOE_DECODE_SLOT_BANK=6` | 6 slots / 3.21 GiB | normal LRU reuse | 9.11 t/s |
| `DS4_FLASH_MOE_DECODE_SSD_CACHE=36GB` | 67 slots / 35.87 GiB | normal LRU reuse | 10.36 t/s |
| attached 32 GB control | 59 slots / 31.59 GiB | normal LRU reuse | 12.71 t/s |

Interpretation:

- "Just six active slots" is not the hidden fast path. No-reuse six-slot decode
  is too miss-heavy, and even normal six-slot LRU trails the 32 GB bank.
- Raising the fast L1 above 32 GB does not help on this prompt. The 36 GB
  point has a similar steady-state hit rate to 32 GB but remains around
  10.36 t/s, still below the attached 32 GB control.
- The 32 GB result is fast because it combines the mixed-bank grouped compute
  path with enough per-layer temporal reuse to avoid most repeated installs.
- Current evidence rules out CPU-L2 promotion and active staging for the short
  90 GB target. Any useful high-memory design must avoid both large-buffer
  binding and per-token/per-miss full-record copies on the hot path.

### 2026-06-12 - Step 7 chunked mixed no-copy probe

Re-tested Candidate D because no result was recorded in this plan. This keeps
the full 168-slot / 89.95 GiB logical bank and splits each layer into smaller
mixed expert-major buffers, then dispatches only chunks containing active
experts. It avoids CPU/GPU L2 copies and avoids binding the full 2.14 GiB
layer bank.

| config | chunks | chunk size | prefill | generation |
|---|---:|---:|---:|---:|
| `DS4_FLASH_MOE_CHUNKED_MIXED=1 DS4_FLASH_MOE_CHUNK_SLOTS=56` | 3 | 56 slots | 4.85 t/s | 5.48 t/s |
| `DS4_FLASH_MOE_CHUNKED_MIXED=1 DS4_FLASH_MOE_CHUNK_SLOTS=84` | 2 | 84 slots | 6.53 t/s | 4.41 t/s |
| `DS4_FLASH_MOE_CHUNKED_MIXED=1 DS4_FLASH_MOE_CHUNK_SLOTS=56 DS4_FLASH_MOE_CHUNKED_SLOTS6_GROUPED=1` | 3 | 56 slots | 6.41 t/s | 6.18 t/s |

Interpretation:

- Chunking is mechanically correct and uses the full 90 GB bank, but it does
  not recover 32 GB speed. Smaller chunks pay multiple routed-MoE dispatches
  and output accumulation; larger chunks reduce dispatch count but reintroduce
  larger-buffer overhead.
- The grouped chunked-slots6 variant removes the multi-dispatch accumulation
  penalty and still only reaches the same ~6 t/s class as per-slot grouped
  decode. Active-buffer binding/view validation remains the ceiling.
- Candidate D is negative for the short 90 GB target.

### 2026-06-12 - Step 8 Metal trace and chunk-bank slots6 probe

Metal System Trace export, 6-token decode, Instruments-overhead numbers only:

| trace | Metal allocation peak | command-buffer encoder time | ds4 GPU intervals |
|---|---:|---:|---:|
| 90 GB per-slot / record path | 116.8 GiB | 1.18 s | 1.10 s |
| 32 GB mixed bank | 58.4 GiB | 0.58 s | 0.96 s |

Interpretation: the trace does allocate high memory; the low-RAM concern was
specific to the first empty-L2 diagnostic. The 90 GB trace adds only about 15%
GPU interval time but about 2x command-buffer encoder time, so the remaining
cost is mostly CPU/driver resource binding and validation, not a GPU kernel that
loops over all 168 slots.

Diagnostic 32 GB A/B:

| config | generation |
|---|---:|
| attached 32 GB control, normal mixed-bank selected-id path | 12.71 t/s |
| `DS4_FLASH_MOE_MIXED_SLOTS6_GROUPED=1`, 32 GB | 4.95 t/s |

Forcing the normally fast 32 GB bank through the slots6 active-buffer path drops
it into the same slow class as full-90 per-slot/chunked slots6. The fast path is
specifically the mixed-bank selected-id kernel shape.

Prototype: `DS4_FLASH_MOE_CHUNKED_BANK_SLOTS6=1` keeps the full 168-slot /
89.95 GiB bank split into 56-slot mixed chunks, passes chunk id + local slot for
each of the six active experts, and avoids constructing six active views.

| config | chunk binding | prefill | generation |
|---|---|---:|---:|
| chunk-bank slots6 v1 | bind all 3 chunks | 6.40 t/s | 0.18 t/s |
| chunk-bank slots6 compact | bind only active chunks | 5.58 t/s | 4.72 t/s |

Result: negative. Binding all chunks recreates the original big-bank validation
cliff. Compact active-chunk binding works but is still slower than ordinary
chunked slots6 (6.18 t/s) and far below the 32 GB control. Treat slots6-shaped
full-residency paths as eliminated for the >12 t/s target.

### 2026-06-12 - Step 9 corrected high-memory GPU-L2 probe

The first GPU-L2 log was not a useful-residency test: it allocated the high
memory layout but reported `preserved L1=0 L2=0/0` because the short prompt did
not populate the prefill slot cache. Re-ran with explicit prefill cache fill:

```bash
DS4_FLASH_MOE_RESIDENCY_STATS=8 \
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=84 \
DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB \
DS4_FLASH_MOE_GPU_L2=1 \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 16 --temp 0 \
  -p "What is Apple Neural Engine"
```

Result:

| config | decode layout | preserved after prefill | tok16 L1 hit | prefill | generation |
|---|---:|---:|---:|---:|---:|
| 90 GB split GPU-L2, topk84 | 59-slot L1 / 109-slot GPU L2, total 89.95 GiB bank | L1=1922, L2=29/1951 | 75.7% | 2.12 t/s | 0.10 t/s |

Interpretation:

- This is the high-RAM case the low-memory concern asked for: total Metal bank
  allocation remains 89.95 GiB, with dense/context total at 100.9 GiB.
- It is dramatically slower than both the 32 GB control and the shrink-only L1.
  Even when the routed kernel binds only the 59-slot L1, retaining the large
  Metal L2 plus doing per-miss GPU blits/writebacks destroys decode throughput.
- GPU-owned high-residency backing is eliminated for the short >12 t/s target.
  The only fast memory backing seen so far is the OS file cache serving
  decode-miss reads while the Metal working set stays near the 32 GB class.

### 2026-06-12 - Step 10 MXFP4 record-table argument-buffer probe

Prototype:

- `DS4_FLASH_MOE_RECORD_TABLE=1` keeps the full 90 GB per-slot allocation.
- At slot-bank allocation time, each layer builds one Metal argument buffer
  mapping slot id -> full expert-record buffer.
- Decode uses new MXFP4 selected-slot kernels:
  `kernel_mul_mv_id_mxfp4_record_table_pair_swiglu_f32` and
  `kernel_mul_mv_id_mxfp4_record_table_sum6_f32`.
- The routed kernel binds one small table buffer plus the usual activation,
  selected-id, and weight buffers. It does not bind the huge mixed layer bank,
  and it does not bind six/eighteen active expert buffers.

Validation:

```bash
DS4_FLASH_MOE_RESIDENCY_STATS=8 \
DS4_FLASH_MOE_RECORD_TABLE=1 \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 16 --temp 0 \
  -p "What is Apple Neural Engine"
```

Result:

| config | decode layout | tok16 hit | prefill | generation |
|---|---:|---:|---:|---:|
| full 90 GB per-slot + record-table argument buffer | 168 slots / 89.95 GiB, total 100.9 GiB | 72.9% | 5.15 t/s | 4.16 t/s |

Interpretation:

- Mechanically works and selects the intended path, but it is slower than the
  previous full per-slot record-buffer path (5.27 t/s) and far below the 32 GB
  control.
- The likely remaining cost is Metal's indirect-resource/argument-buffer
  validation or GPU pointer indirection across the full table. This means the
  "bind one table, index records in-kernel" approach does not recover the
  selected-id mixed-bank speed.
- Treat record-table/argument-buffer full-residency as negative unless a future
  Metal trace proves a different bottleneck.
