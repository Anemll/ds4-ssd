# ANE+NAX hybrid resident — plan & current state
_2026-05-28 · ongoing /loop "until ANE+NAX is fastest"_

## State

- **Plan A (pure GPU NAX-half fused, MIN_REFS=0)**: 573.1 t/s @ 16K resident — current champion.
- **Existing ANE+GPU probe** (`moe-batch-bench/ane_gpu_probe_summary.txt`):
  - GPU solo (8192 batches): 33.4 ms
  - ANE solo (8192 batches): 20.2 ms
  - Concurrent vs sequential: **1.61× speedup**, 18% overhead vs ideal max(GPU,ANE).
  - Validates: ANE∥GPU concurrency is real on M5 Max for big-M workloads.

## Pivot: which path to wire

The original plan was staged-MPP (`ds4_gpu_routed_moe_batch_tensor`), but the
ANE eligibility diag (this loop iter) shows that path is called with
sub-chunks of ~192 tokens, giving per-expert M ≈ 32–282. **That's at or
below the ANE crossover threshold** — the win disappears in startup overhead.

**Better target: the compact bridge** (where Plan A's iq2_fused chain runs
through `use_fused` branch). At 16K, per-expert M ≈ 512 — well into the ANE
win region (probe shows 1.61× at M=8192). The trade-off: compact bridge uses
GPU-side counts (no host-side `counts[e]`); to gate ANE eligibility we need
a sync read-back of the counts buffer (~256 uints × 43 layers ≈ 11 KB total
per chunk = ~2 ms read-back overhead, negligible vs 30 s prefill).

## Per-M ANE vs GPU crossover (UNKNOWN — needs probe)

Next iteration: build a focused probe that measures, at fixed K (~4096) and
N (~2048), the GPU-h_h_f-fused time vs ANE-i8i8-tiled-fused time for
M ∈ {32, 64, 128, 256, 512, 1024}. The crossover M tells us the
`DS4_RESIDENT_MOE_ANE_MIN_REFS` threshold.

Rough expectation from existing data: crossover likely in the 200–500 range.
ANE startup is ~2 ms per submission; at M=200 with K=4096 N=2048 the GPU
runs in ~0.5 ms, so ANE loses below ~500. At M=500+ GPU is ~1.3 ms while
ANE is ~1.5 ms but runs concurrently → net win starts emerging.

## Update after iters 21-23 (2026-05-28) — direct-eval upper bound

User's first-principles framing (iter 21 message): win formula is
`producer_overhead + max(GPU_remaining, ANE_eval+ANE_output) < GPU_all`.
Said `_ane_start_tensor` makes producer_overhead too fat ("feeding 50 TOPS
through a straw"); recommended testing the direct-eval path bypassing the
wrapper as the upper bound.

**Implemented (iter 22)**: new `ds4_gpu_ane_direct_eval_one_expert` API
(`ds4_gpu.h` + `ds4_metal.m`). Bypasses `_ane_start_tensor`:
1. Encodes `ds4_gpu_encode_mpp_dequant_gud_i8` (fused gate+up+down → int8).
2. Encodes `ds4_gpu_encode_mpp_gather_token_f32_i8` (f32 acts → int8).
3. `ds4_gpu_synchronize` to drain cb.
4. Calls `ds4_ane_mlp_i8w_i8x_tiled_fused_eval` directly with raw
   `MTLBuffer.contents` pointers.

**Wired (iter 23)** behind `DS4_RESIDENT_MOE_ANE_DIRECT_EVAL=1` env flag.
With sorted K=1 min_refs=250 + FIXED_BATCH=1 + MAX_REFS=256:

| Path | Prefill t/s | vs Plan A 573 |
|---|---:|---:|
| `_ane_start_tensor` K=1 min_refs=250 (iter 21) | 562.6 | −1.8% |
| **Direct-eval K=1 min_refs=250 (iter 23)** | **561.4** | **−2.0%** |

**Verdict: bypassing the wrapper does NOT change the plateau.** Both paths
land at ~560 t/s. The bottleneck is NOT the wrapper API churn — it's that
`ds4_gpu_synchronize` (or `_ane_start_tensor`'s internal `flush_commands`)
drains the FULL shared cb, charging ANE producer_overhead the cost of all
pending GPU work. Per the win formula: producer_overhead is roughly equal
in both paths, dominating any concurrency gain.

**Per user's iter-20 decision criteria**: direct-eval doesn't clear 580 t/s,
so scatter-add cost would push the corrected/full path further below Plan A.
Stop pursuing this architecture.

**The genuine next-restructure** (NOT committed, multi-day, blocked on a
different design): SEPARATE COMMAND QUEUE for ANE producer work. ANE producer
kernels (dequant + i8 gather) go on a dedicated `MTLCommandQueue`,
independent of the main GPU routed-MoE queue. Producer flushes async,
completion handler fires ANE worker pthread. Main thread doesn't wait on
producer cb. Routed-MoE GPU work runs concurrently on main queue.

Plus the user's other suggestions (still valid): hot-weight int8 cache for
stable top-K experts (~1-4 GB), CPU producer probe to bypass GPU producer
entirely, multi-expert ANE graph for batched eval. Each is its own deep
investigation.

**Final state**: Plan A 573 t/s remains the production champion. ANE hybrid
infrastructure (outer-caller hook, sorted-K-select, skip-mask, scaffolding,
direct-eval entry point) all preserved in tree behind
`DS4_RESIDENT_MOE_ANE_HYBRID*` and `DS4_RESIDENT_MOE_ANE_DIRECT_EVAL` envs.

## Update after iters 17-20 (2026-05-28) — sorted K-sweep verdict

User's review of iter-17 data (K=1=560.7 looked promising) caught a real bug:
the K-cap was selecting by EXPERT ID order, not by refs DESCENDING. Iter 20
fixes that: build candidate list of in-window [min_refs, 256] experts, sort
descending by `per_expert_counts[]`, K caps the sorted list. Plus added per-call
diag: `picked-refs(topN): e<id>=<refs>` so we can verify the actual selection.

**Sorted K-sweep results @ 16K resident (with `THREADS_FIXED_BATCH=1` +
`MAX_REFS=256` to avoid the dual-split batch-shrink trap):**

| min_refs | K | Sorted refs picked | t/s |
|---:|---:|---|---:|
| 128 | 1 | e228=256 | **557.8** |
| 128 | 4 | 256, 254, 254, 254 | 547.3 |

Plan A baseline: **573 t/s**. Gap at sorted K=1: **−2.7%**.

**Decision criteria recap (user, iter-20)**:
- ≥573 t/s repeatable → finish single-hot-expert relief valve as production.
- 560-565 t/s → useful evidence but not production (no headroom for scatter-add).
- K=2 loses meaningfully → direct `_eval` is the only path forward.

Outcome: sorted K=1 at **557.8 t/s does not clear 573**, and adding scatter-add
for correctness (currently throwaway) would push it further below. K=2 already
loses ~10 t/s vs K=1. Both decision rules say: **stop here; do not finish the
single-hot path; do not start the direct-`_eval` restructure** unless someone
wants to invest the multi-day work for a speculative further gain.

**Final state of integration (all gated off by default)**:
- Outer-caller hook in `ds4.c` after router_select, before routed_moe_batch_tensor.
- CPU-side per-expert counts, sort-descending K selection, hids tensor wrap,
  filtered f32 acts/weights gather, per-expert weight views via
  `ds4_gpu_model_tensor_view`, ANE ctx fetch.
- `_ane_start_tensor` jobs deferred for concurrent GPU+ANE; pthread_join in
  post-GPU drain.
- Skip-mask wiring (new `ds4_gpu_set_ane_skip_mask` API in `ds4_gpu.h`).
- Precompile at warmup (layer-0 first-call cost moved out of prefill timer).

**Plan A (573 t/s) remains production champion.** Hybrid code preserved
in-tree as a working concurrency scaffold; reachable for future restructure if
needed, but not currently a win.

## State after 16 /loop iters (2026-05-28)

### What's working in tree (gated off by default)
- `DS4_RESIDENT_MOE_ANE_HYBRID=1` triggers precompile of i8i8 tiled-fused ANE
  ctxs at warmup time (layer-0 cost moved out of prefill timer).
- `DS4_RESIDENT_MOE_ANE_HYBRID_OUTER=1` activates the outer-caller hook in
  `ds4.c` (after router_select, before routed_moe_batch_tensor).
- Outer hook builds per-expert hids on CPU from `selected[]`, gathers filtered
  f32 acts + per-token routing weights via existing GPU gather kernels, builds
  per-expert weight views via `ds4_gpu_model_tensor_view`, fires
  `ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor` for every eligible
  expert (refs in [min_refs, 256]).
- ANE jobs run concurrent with GPU (pthread inside `_ane_start_tensor` returns
  after dequant+upload+spawn). `finish_tensor`/`pthread_join` deferred to AFTER
  `routed_moe_batch_tensor` returns.
- `DS4_RESIDENT_MOE_ANE_HYBRID_SKIP=1` plus the new
  `ds4_gpu_set_ane_skip_mask` API (declared in `ds4_gpu.h`) tells the per-expert
  GPU loop to skip ANE-handled experts.
- User CONFIRMED ANE active on monitor ("very active ANE now") at 16K resident.

### Numbers @ 16K resident, cooldown-separated triple-run

| Config | Prefill t/s | vs Plan A |
|---|---:|---:|
| Plan A baseline (no ANE) | 573 | — |
| Hybrid sequential (no skip) | 444 | −22.5% |
| Hybrid sequential + skip | 421 | −26.5% |
| Hybrid concurrent (deferred drain) + skip | 507 | −11.5% |
| Hybrid + precompile + skip (triple-run) | **509.6 / 509.6 / 510.2** | **−11%** |
| Hybrid + PREFLUSH_EVERY=32 | 508.8 | −11% |
| Hybrid without pre-ANE synchronize (iter-15) | 511.8 | −11% |

Per-layer ANE fire wall (warm): ~40 ms (87 experts), post-GPU drain ~14 ms.
First-layer wall now 34 ms with precompile (was 436 ms before).

### Where the −11% gap lives

Hybrid takes 32s vs Plan A 28.6s = 3.4s of overhead per 16K prefill. Verified
by experiments:
- Not the pre-ANE `synchronize` (removing made fire wall 40→97 ms, net same).
- Not the per-call `flush_commands` inside `_ane_start_tensor` (PREFLUSH_EVERY=32
  same prefill rate).
- Not the compile (precompile already at warmup).
- Likely cumulative Metal API call overhead from `_ane_start_tensor`'s
  monolithic per-expert work: dequant kernel encoding (×3 per expert: gate/up/down)
  + flush_commands + ANE worker pthread spawn = ~0.45 ms per call × 87 experts
  × 86 layer-calls ≈ 3.4 s. Matches the observed gap.

### NEXT BITE (multi-day; not started)

To close the gap: STOP calling `_ane_start_tensor` per-expert. Instead, the
"dequant-into-shared-cb" pattern user described:

1. For each layer-call: encode iq2/q2k→i8 dequant kernels for ALL eligible
   experts' weights directly into the shared cb (using existing
   `ds4_gpu_encode_mpp_dequant_iq2_xxs_i8` + `dequant_q2_k_i8` GPU kernels).
2. Encode the gather kernels for filtered acts.
3. ONE `flush_commands` (commits cb, no wait).
4. Spawn N pthreads (one per expert) that:
   - Wait for the cb to complete (`[cb waitUntilCompleted]` from any thread).
   - Call `ds4_ane_mlp_i8w_i8x_tiled_fused_eval` directly with CPU pointers to
     the now-populated int8 scratches (mapped MTLBuffer.contents).
5. Main thread proceeds: `begin_commands` + `routed_moe_batch_tensor` (with
   skip-mask set). This GPU work runs concurrent with ANE pthreads.
6. After `routed_moe_batch_tensor` returns: `pthread_join` all ANE threads.
7. Scatter ANE outputs to routed-out buffer (currently throwaway).

Plus the cheap-to-add bigger-B ctxs (B=512, B=1024) so refs>256 experts
(currently dropped) become eligible — expands ANE coverage from 34% → ~89%.

### Stop the loop reason

Sixteen iterations of focused work; infrastructure verified end-to-end; gap
diagnosed. The next bite (encode-dequant-in-cb + direct-eval pthread + scatter
+ bigger-B ctxs) is a single substantial restructure rather than another
inline iter. Better as a focused next session than as more /loop pumping.

## State of integration after 6 /loop iters (2026-05-28)

In-tree scaffolding in `ds4_metal.m::ds4_gpu_routed_moe_batch_tensor` per-expert loop (around line 23500+, behind `DS4_RESIDENT_MOE_ANE_HYBRID=1`):

- ✅ `ane_mask[]` host-side eligibility computation + DIAG env (`DS4_RESIDENT_MOE_ANE_HYBRID_DIAG=1`)
- ✅ Per-expert weight views via `ds4_gpu_model_tensor_view(model_map, model_size, expert_offset, expert_bytes)` for gate/up/down — `views_ok=1` confirmed
- ✅ Static scratches `g_resident_ane_x_scratch` (4 MB f32, 256 max-refs × 4096 in_dim) and `g_resident_ane_w_scratch` (1 KB f32) lazily allocated
- ✅ Per-expert hids tensor wrap from `g_moe_id_map_buffer + hids_off` (using DS4MetalTensor pattern from line 15769)
- ✅ `ds4_gpu_gather_rows_f32_tensor` for filtered x (refs × in_dim) and weights (refs × 1) — both `gather_ok=1`
- ✅ `ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor` + `_finish_tensor` calls compile and return `ane_submit_ok=1` on ad-hoc smoke
- ❌ **INLINE SUBMIT DEADLOCKS** in full prefill: ANE's internal `ds4_gpu_flush_commands()` conflicts with the surrounding per-expert GPU encoder cb. Naive `end_commands+begin_commands` dance around the ANE call also fails (`ane_submit_ok=0` — pre-flush rejects state). **Gated off** behind unset `DS4_RESIDENT_MOE_ANE_HYBRID_SUBMIT` env so default Plan A path (339 t/s @ 2K, 573 t/s @ 16K) is unaffected.

## Update after iter 7: cb-state diagnosis

The cb-conflict isn't really in ANE — it's that `routed_moe_batch_tensor` runs
inside a SHARED batch managed by its caller. `ds4_gpu_command_buffer(&owned)`
returns `g_batch_cb` (owned=0) when in a batch. Any `end_commands` /
`flush_commands` from inside this function ends the OUTER caller's batch
prematurely → corrupts state.

ANE function `_ane_start_tensor` works fine on M3 because flash hybrid calls it
from OUTSIDE such a shared batch (ds4.c:13526 is at the layer-loop level, after
explicit `end_commands`). So the API isn't broken — the integration point is
wrong.

Confirmed working in tree (iter 7):
- ✅ `ds4_gpu_ane_get_i8i8_tiled_ctx(H=4096, I=2048, B=256, scales)` returns
  non-NULL ctx (`ctx_ok=1` per diag).

## NEXT BITE (start here, multi-iter): MOVE ANE submit out of the shared batch

Inline submit can't work; need a 2-pass restructure:

1. **Pass 1** (in the per-expert GPU loop): for each ANE-eligible expert,
   collect `{expert, refs, gate/up/down model offsets}` into a `pending_ane[]`
   array. **Do not** submit yet, and **do not** dispatch GPU gate/up/swiglu/down
   for these experts (skip via `ane_mask[expert]`).
2. **After per-expert GPU loop completes** (all non-ANE work encoded but cb
   still open): `ds4_gpu_end_commands()` once. This is the natural drain point.
3. **Per-expert scratches**: the single-slot `g_resident_ane_x_scratch` /
   `_w_scratch` can't hold multiple experts' data — either:
   - (a) Bump to per-expert array of scratches (N × 4 MB = bounded).
   - (b) Process ANE experts one-at-a-time post-drain: gather→submit→finish
     per expert (no actual concurrency, but proves the path).
   - (c) Skip pass-1 gather entirely, do gather + submit + finish all
     post-drain. Best balance.
4. **Single-active-ANE submit loop**: for each pending expert, run gather
   (still on GPU; needs `ds4_gpu_begin_commands()`+`end_commands()` to drain
   before ANE), submit ANE, wait+finish previous job, advance.
5. **`ds4_gpu_begin_commands()`** before subsequent layers' work.
6. **Output scatter**: ANE writes to `out` directly (via finish_tensor's `out`
   arg). Verify per-token positions don't collide with GPU's writes for non-
   ANE experts. May need ANE to write to a separate scratch + scatter-add.
7. **Correctness**: layer-0 `ffn_moe_out` dump under hybrid vs pure-Plan A,
   max_abs should be at the int8 quant noise floor (~1e-2 range, matching
   `ane_vs_fp16_cpu max_abs=0.084` from `ane_ds4_mlp_i8i8_precision_smoke`).
8. **Threshold sweep**: `DS4_RESIDENT_MOE_ANE_MIN_REFS` ∈ {32, 64, 96, 128, 192}
   with cooldown-separated triples at 2K/8K/16K resident.

## Required envs (currently)

```bash
DS4_RESIDENT_MOE_ANE_HYBRID=1                  # engage the resident hybrid wiring
DS4_RESIDENT_MOE_ANE_HYBRID_DIAG=1             # diagnostic prints (optional)
DS4_RESIDENT_MOE_ANE_MIN_REFS=128              # eligibility threshold (tune)
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1               # required by ane_start_tensor
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1   # required by ane_start_tensor
# Plus the Plan A enablement:
DS4_RESIDENT_MOE_NAX_HALF=1 DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1 \
DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0 \
DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1 DS4_RESIDENT_MOE_MPP_FORCE=1 \
DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64 DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS=64 \
DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=1 DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT=1
```

## Implementation steps (compact bridge target)

1. **Per-layer per-chunk counts read-back**: after the counts buffer is
   populated, host reads 256 uints once. (Use existing `ds4_gpu_tensor_read`
   or a small blit-to-shared helper.)
2. **Eligibility mask** on host: experts where `counts[e] >= ane_min_refs`.
3. **Per-expert tensor views** for weight banks: `ds4_gpu_tensor_view(gate_tensor,
   gate_expert_off, gate_expert_bytes)` for each ANE-eligible expert.
4. **Filtered x gather**: needs a new gather kernel that produces `f32`/`f16`
   x output filtered by hids for this expert (existing gather writes int8 for
   GPU; ANE needs higher precision input). Or repurpose the existing
   `g->flash_prefill_x` infra if it's also resident-allocatable.
5. **Filtered weights view**: per-expert pair-weights (likely just an offset
   into the existing routing-weights buffer).
6. **Submit + drain loop**: single-active-ANE pattern matching flash hybrid.
   Submit ANE job for the next eligible expert, wait+finish previous.
7. **Skip ANE experts in compact bridge's gate/up/swiglu/down chain.**
8. **Scatter ANE output**: ANE produces full per-token contribution → scatter-add
   into the routed output buffer.
9. **Correctness**: layer-0 `ffn_moe_out` dump vs Plan A reference.
10. **Bench**: 16K resident with cooldowns, threshold sweep.

## Required env to engage ANE

- `DS4_FLASH_MOE_ANE_I8I8_PREFILL=1` — required (function checks).
- `DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1` — required.
- `DS4_RESIDENT_MOE_ANE_HYBRID=1` — the new wiring flag.
- `DS4_RESIDENT_MOE_ANE_MIN_REFS=<TBD>` — to be tuned via the M-crossover probe.

## Stop condition for the loop

Exit when one of:
- Hybrid GPU+ANE @ 16K resident **exceeds 573.1 t/s** (current Plan A peak)
  with at least one ANE submission per layer per chunk (i.e., "some ANE used").
- OR after enough investigation it's clearly infeasible (e.g., per-expert M
  ceiling stays below the ANE crossover even with compact bridge), in which
  case stop with that conclusion documented.

## CORRECTION 2026-05-28 — user pushed back on the "infeasible" verdict below

The original analysis assumed an ANE per-call wall of ~2 ms based on
speculation. **Actual measurement from `ane_parallel_ctx_bench`**
(H=4096 I=2048 B=256):
- serial single-ctx: **1.657 ms/call**
- xonly (weights pre-written): **1.256 ms/call**
- 4-ctx concurrent floor: **1.115 ms/call**

Re-running the optimal-K offload arithmetic with real numbers:

| ANE per-call wall | Optimal K | Wall saved | Resident gain |
|---:|---:|---:|---:|
| 1.66 ms (serial) | 10 | 0.7 ms/call | **+9%** |
| 1.26 ms (xonly)  | 13 | 0.9 ms/call | **+12%** |
| 1.11 ms (4-ctx)  | 15 | 1.05 ms/call | **+14%** |

Plan A 573 × 1.12 ≈ **640 t/s** achievable target if xonly-class wall holds
in production. The integration cost is still multi-day, but the upside is
real — **not infeasible**, just expensive.

The verdict text below was wrong about feasibility (used speculative ANE
wall). Keeping it for the record but supersede with the measurement above.

## ORIGINAL (superseded) verdict — loop stopped 2026-05-28, condition: infeasible upside

Investigation across 4 loop iterations leads to the second exit condition.
Three data points:

### 1. GPU h_h_f-fused per-M is FLAT
Sweep at K=4096, N=2048 (DS4 expert dims):

| M | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU fused ms | 0.082 | 0.070 | 0.073 | 0.067 | **0.072** | 0.098 | 0.102 | 0.095 |

Per-expert GPU runtime is essentially constant 0.07-0.10 ms across the entire
useful M range. The K-loop dominates and the matmul2d execution saturates
parallelism by M≈32. Implication: there is **no per-expert M region where
ANE can plausibly beat GPU** — GPU is sub-ms, ANE startup alone is ~1-2 ms.

### 2. Per-call ANE eligibility is small
At resident 16K with `DS4_RESIDENT_MOE_NAX_HALF=1`, the staged-MPP path is
called with sub-chunks of ~192 tokens × 8 top-k = 1536 refs per call:

- min_refs=129 (default): 0-3 ANE-eligible experts per call.
- min_refs=32: 9-16 ANE-eligible experts per call (78% of refs).
- min_refs=16: ~all hot experts.

Even at min_refs=32, only ~14 experts have enough M to plausibly amortize
ANE startup, and their refs (32-200) sit in the small-M region where ANE is
slowest relative to its peak.

### 3. Concurrency arithmetic doesn't close
GPU wall per call: ~256 experts × 0.07 ms = **18 ms**.
ANE wall for offloading 14 experts at ~2 ms each:
- 1 cluster: 28 ms (LOSES).
- 2 clusters: 14 ms (saves ~1 ms vs 18 ms baseline if it perfectly overlaps).

Even in the optimistic 2-cluster case, savings per call ≈ 1 ms. Multiplied
across ~3655 per-call invocations in a 16K prefill = ~3.6s saved. From 30s
total = 27.4s = **~598 t/s vs Plan A 573 = +4.4% best-case ceiling**.

The compact-bridge alternative (per-layer per-chunk, M ≈ 512) has the same
problem at scale: GPU h_h_f at M=512 is 0.098 ms × 256 experts = ~25 ms
GPU wall; ANE startup × 100 hot-experts wall ≥ 75 ms even with 2 clusters.
No headroom for ANE without slowdown.

### Why the 1.61× flash concurrent probe doesn't transfer
The existing `ane_gpu_probe_summary.txt` measured ANE at 8192-batches in a
single workload (one big ANE call). That's 32× larger than the largest
per-expert M in resident. ANE startup amortizes across 8192 tokens there
(~0.25 us/token). In resident, per-expert max is ~512 → 4-8 us/token but
the startup is fixed → effective per-token cost is dominated by overhead.

### Conclusion
Plan A (h_h_f fused + MIN_REFS=0) at **573 t/s @ 16K resident is at or near
the M5 Max compute-bound prefill ceiling for this model** — the previously
published 532 was a slight underestimate; Plan A's wider min-refs reach
unlocked the remaining ~8%. Adding ANE in the hybrid pattern cannot
materially exceed this for resident workloads because (a) GPU per-expert
work is already sub-ms, (b) ANE startup overhead is ~1-2 ms, and (c) the
resident per-call expert M distribution doesn't reach the regime where
ANE amortizes.

The only paths to further resident speedup are non-ANE:
- Multi-expert grouping (kernel + wrappers landed this session; production
  wiring TBD). Saves dispatch overhead, similar workload to ANE-startup-vs-
  GPU-work analysis but stays on GPU where overheads are lower.
- Lower-bit weight packing (e.g. ternary) reducing memory bandwidth.
- Compressed attention beyond the current indexer.

ANE remains genuinely useful for the **flash path** where token batches are
large (8192+) and amortization works — but is the wrong tool for resident.
