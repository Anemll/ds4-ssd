# Flash-MoE Stable Slot Progress

Date: 2026-06-05

Branch: `codex/stable-slot-replay-experiment`

## Goal

Restore the original stable slot-bank idea: a routed expert miss loads into a resident expert buffer, and later hits reuse that resident buffer and its execution plan until eviction. The goal is not to release cached experts; it is to release and rebuild the reusable execution object only when the resident expert is evicted.

## Current Finding

The current `ds4-ssd` decode path has a stable residency map, but not the full stable execution design.

- It tracks `slot -> expert`, `expert -> slot`, and LRU ages.
- It resolves true routed experts to slot IDs.
- Hits skip disk reads.
- Decode still executes the 6 routed experts as one banked routed-MoE call.
- The replay cache currently caches shallow encoder args/grid for that 6-expert call, not a reusable per-expert execution object.

The 6-expert signature is unlikely to repeat often, so caching the whole top-6 group is the wrong unit for stable replay.

## Stable Default Preset

The release-stable sidecar default is now the conservative fast path:

- grouped banked decode remains the default execution mode;
- mixed expert-major slots are default-on;
- direct miss installs into resident slots are default-on;
- decode router/scratch prefetch is default-off with
  `DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0`;
- stable/baked slot replay, async handout, and ICB replay are default-off;
- generic Metal decode replay remains default-on;
- routed-down `sum6` remains opt-in through
  `DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=1`.

The three comparison modes are:

```text
# 1. Stable release default / grouped baseline
DS4_FLASH_MOE_STABLE_REPLAY=0
DS4_FLASH_MOE_BAKED_SLOT_DECODE=0
DS4_FLASH_MOE_ASYNC_HANDOUT=0
DS4_FLASH_MOE_ICB_REPLAY=0
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0

# 2. Stable slot execution, no async
DS4_FLASH_MOE_STABLE_REPLAY=1
DS4_FLASH_MOE_ASYNC_HANDOUT=0
DS4_FLASH_MOE_ICB_REPLAY=0
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0

# 3. Dynamic grouped / async stable experiment
DS4_FLASH_MOE_STABLE_REPLAY=1
DS4_FLASH_MOE_ASYNC_HANDOUT=1
DS4_FLASH_MOE_ICB_REPLAY=0
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
```

Profile defaults are applied with `setenv(..., overwrite=0)`, so explicit shell
exports still win. The async/ICB paths stay available for experiments, but they
should not turn on from a bare `./ds4` sidecar run.

## Benchmark Clues

Small model runs showed slot count itself changes decode time even when memory is not exhausted.

- `--moe-slot-bank 32`: generation about 9.6 tok/s.
- `--moe-slot-bank 256`: generation about 4.3 tok/s.
- Decode replay stats were nearly identical, so this is not explained by replay misses.
- Stage profiling showed routed MoE was the moving part:
  - slot32 routed-MoE average about 1.8 ms.
  - slot256 routed-MoE average about 5.0 ms.
  - zero-miss routed calls also got slower with larger banks.

This points at the resident bank execution/memory shape, not just SSD or prefetch overlap.

These measurements are only guardrails until the real per-expert replay object
exists. The current branch still executes decode as one grouped banked routed-MoE
call for the six routed experts, so it has not yet implemented the original
stable replay unit.

## Reference Design

The original stable slot-bank plan from `anemll-flash-mlx` / llama.cpp has these properties:

- Resolve `expert_id` to a resident `slot_id`.
- On a hit, reuse the resident slot and stable consume path.
- On a miss, load into a victim slot and update ownership.
- The miss path and hit path are separate problems.
- The consume path should be stable with IDs as data.

The important correction for this branch: if the expert data is read into a pread-backed resident buffer, execution should point at that resident buffer. It should not require `pread -> temporary buffer -> second copy into a different slot slab`.

## Implementation Plan

1. Use a mixed expert-major resident slot buffer per layer.
   - Shape: `slot_bank * expert_stride`.
   - Miss path can pread the whole expert record into `slot * expert_stride`.
   - Gate/up/down become views into that buffer with family offsets.

2. Teach the current routed execution to use explicit slot stride.
   - Old separate-bank path: slot stride is `family_bytes`.
   - Mixed-bank path: slot stride is `expert_stride`.
   - Existing quant kernels can still work because they already use `args.nb02` as the expert/slot stride.

3. Benchmark before adding replay complexity.
   - Compare old three-bank path with `DS4_FLASH_MOE_MIXED_SLOT_BANK=0`.
   - Compare mixed path default with slot32 and slot256.
   - Keep `DS4_FLASH_MOE_DECODE_PREFETCH=0/1` A/B to separate read overlap from resident execution cost.

4. Test layer-wise resident slot locality before deeper fusion work.
   - Existing mixed mode already stores each layer as `slot_bank * expert_stride`.
   - Add an optional single layer-major slab:
     `DS4_FLASH_MOE_LAYER_SLOT_SLAB=1`.
   - Test slot32 first, then slot64 and slot128.
   - Treat slot256 as a deferred locality/falsifier track, not the routine
     fusion metric.

5. Complete the fused replay workflow after the locality check.
   - Keep the ordered route unit correct first: fused gate/up/SwiGLU, then down.
   - Remove unnecessary tail dispatches and interruptions where possible.
   - Only pursue a monolithic gate/up/SwiGLU/down kernel if the two-stage replay
     path remains dominated by dispatch/dependency cost.

6. Build the real per-expert replay object.
   - A true Metal command buffer is one-shot, so this likely means an ICB or prebuilt encoder plan.
   - The unit should be a single resident expert/slot, not the whole 6-expert routed set.
   - If expert is resident but replay object is missing, skip disk load and create the replay object.
   - On eviction, release/invalidate the replay object for that slot/expert.

7. Revisit compute-while-load.
   - Once the resident hit path is stable, compute resident experts while missing experts are loading.
   - Fold late miss experts into the output as they arrive.

## In-Progress Code Direction

The current experiment is wiring a mixed slot-bank path in `ds4.c`, `ds4_gpu.h`, and `ds4_metal.m`.

- `DS4_FLASH_MOE_MIXED_SLOT_BANK=1` is intended to be the default experiment path.
- `DS4_FLASH_MOE_MIXED_SLOT_BANK=0` should restore the old separate gate/up/down slot-bank layout.
- `DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0` should restore temporary-buffer loading where needed for A/B checks.

## Implementation Checkpoint

Current branch state:

- Added `flash_mixed_bank[layer]` as an expert-major resident slot buffer.
- Mixed mode allocates one `slot_bank * expert_stride` GPU tensor per layer.
- Gate/up/down bank tensors are views into the mixed buffer at the family offsets.
- Miss installs can pread directly into the resident mixed slot record.
- Separate-family mode remains available with `DS4_FLASH_MOE_MIXED_SLOT_BANK=0`.
- Banked decode now passes explicit gate/down slot strides into Metal execution.
- Non-banked model-map decode was restored to its original contiguous expert layout.
- `make ds4` passes on this branch; remaining warnings are pre-existing sign/unused warnings.

Short A/B runs on the small model:

- Separate-family layout with temp-buffer install remains the best current guardrail.
- Direct pread into resident Metal buffer costs throughput before replay exists.
- Mixed expert-major layout is useful for testing no-copy residency, but by itself
  is not the replay fix.

Conclusion: do not judge the design on these numbers yet. The missing piece is
the stable per-expert execution object.

## Correctness Checkpoint

Deterministic decode parity was checked with:

```text
./ds4 -m ~/Models/flash/dsv4-iq2xxs-expert-major --ctx 60768 -p "Who are you" --temp 0
```

Slot32, `-n 64`, exact stdout hash:

```text
ccf56d68b73aefc8147bdd15427c1391905945e5861f1c7e54def6d7a9f6653b
```

Matching modes:

- separate-family layout, temp-buffer install
- separate-family layout, direct resident pread
- mixed expert-major layout, temp-buffer install
- mixed expert-major layout, direct resident pread

Slot256, `-n 8`, exact stdout hash:

```text
efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04
```

Matching modes:

- separate-family layout, temp-buffer install
- separate-family layout, direct resident pread
- mixed expert-major layout, temp-buffer install
- mixed expert-major layout, direct resident pread

No `failed`, `undersized`, `mismatch`, `nan`, or `inf` patterns were found in
the stderr logs for these parity runs.

## Slotwise Decode Checkpoint

Added an env-gated slotwise decode path:

```text
DS4_FLASH_MOE_SLOTWISE_DECODE=1
```

This does not yet implement the final stable replay object. It changes the
execution unit from one grouped six-expert banked call to six one-expert banked
calls, writes per-expert down outputs into the existing expert scratch, then
sums them. This is the unit where a per-slot/per-expert replay object can attach.

Correctness checks:

- Slot32, `-n 32`, separate-family/temp install:
  - grouped hash: `ab89b0ff2f40cd77ba4863b8d438ae559c46e09d3fbc90dccfc9d12fa0f0f6ed`
  - slotwise hash: `ab89b0ff2f40cd77ba4863b8d438ae559c46e09d3fbc90dccfc9d12fa0f0f6ed`
- Slot32, `-n 32`, mixed/direct install:
  - slotwise hash matched the grouped reference above.
- Slot256, `-n 8`, mixed/direct install:
  - slotwise hash matched the grouped slot256 reference:
    `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`

No warning/error patterns were found in the slotwise stderr logs.

## Baked-Slot Plan Checkpoint

Added an env-gated baked-slot decode path:

```text
DS4_FLASH_MOE_STABLE_REPLAY=1
```

Alias:

```text
DS4_FLASH_MOE_BAKED_SLOT_DECODE=1
```

Important: this is still not true Metal replay. It still encodes normal Metal
commands each token. The difference is that each one-expert execution points
directly at the resolved resident slot by host buffer offset and feeds a zero ID
to the existing `*_id` kernels. That makes the execution independent of the
number of slots in the bank, and it is the shape a real per-slot ICB/replay
object should capture.

Also added a C-side replay ownership table:

- `layer,slot -> replay expert`
- `layer,slot -> replay valid`
- hit/miss/build/invalidation counters

Slot ownership and replay ownership are separate on purpose. If an expert is
resident but its replay plan is missing, the miss is counted as a replay-build
candidate without reloading the expert from disk. When a slot is evicted or
invalidated, the attached replay plan is invalidated too.

Correctness, `-n 8`, separate-family/temp layout:

- slot32 grouped hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot32 slotwise hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot32 baked-slot hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot256 grouped hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot256 slotwise hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot256 baked-slot hash:
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`

Short-run speed breadcrumbs from the same `-n 8` run:

- slot32 grouped: generation `6.47 t/s`
- slot32 slotwise: generation `5.84 t/s`
- slot32 baked-slot: generation `5.78 t/s`
- slot256 grouped: generation `0.32 t/s`
- slot256 slotwise: generation `0.33 t/s`
- slot256 baked-slot: generation `0.33 t/s`

These speed numbers are not proof of replay performance because there is no ICB
or reusable command object yet. The value is correctness: the host-offset,
zero-ID path produces identical output.

Mixed expert-major plus direct resident pread also matched the same `-n 8`
baseline hash in baked-slot mode:

- slot32 mixed/direct baked-slot: generation `5.15 t/s`
- slot256 mixed/direct baked-slot: generation `0.29 t/s`

With backend stats enabled on slot32 baked-slot `-n 8`:

```text
Flash-MoE stable replay plans hits=971 misses=1092 hit-rate=47.1% builds=1092 invalidations=24 live=1068/1376
```

Those are lifecycle counters only. They show how often a real replay object
would be reused or built under the current residency behavior.

## Current Next Step

Continue toward a real replay object. The current experiment has:

- ICB-capable routed matvec pipelines via
  `MTLComputePipelineDescriptor.supportIndirectCommandBuffers = YES`.
- Per-slot invalidation on eviction/install.
- A real ICB path for down projection.
- A real ICB path for fused gate/up/SwiGLU.
- A diagnostic two-command routed-unit ICB containing gate/up/SwiGLU then down.

Important finding: compute ICB dispatch commands are
`MTLIndirectCommandTypeConcurrentDispatch`. Executing a dependent two-command
range `[gate/up/SwiGLU, down]` as one concurrent range produced wrong output.
Split execution of command 0 then command 1 produced the baseline hash; split
execution without an explicit Metal buffer barrier also matched in the short
test. The reference llama.cpp branch's "one tensor prod" is graph/op-level
fused, not a single monolithic Metal kernel: it dispatches pair/SwiGLU, then
down, and includes a memory barrier before down.

The unsafe concurrent two-command route-unit execute path has been removed from
this experiment. If `DS4_FLASH_MOE_ICB_FUSED_UNIT=1` is enabled, it always
executes command 0 then command 1 as split dependent dispatches.

Correctness checkpoints:

- slot32 down-only ICB: hash matched
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot256 down-only ICB: hash matched
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot32 fused pair/SwiGLU + down in one concurrent ICB range: wrong hash
  `b8f7509db6ce9386779cbf23980ce429d0821ed694e869494971a8c699e110ae`
- slot32 fused pair/SwiGLU + down split ICB execution: baseline hash matched
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`
- slot32 fused pair/SwiGLU + down split ICB execution without explicit barrier:
  baseline hash matched
  `efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04`

Speed breadcrumbs from short `-n 8` runs should not be treated as final replay
performance yet:

- slot32 down-only ICB was correct around `5.6 t/s`.
- slot256 no-ICB baked-slot was around `0.38 t/s` in one same-build run.
- slot256 split fused-unit diagnostic was correct around `0.30 t/s`.

Interpretation: current replay is still too interrupted. The correct target is
not "two dependent concurrent commands in one ICB range"; it is either:

- a graph/op-level fused route unit that replays ordered stages cheaply, or
- a monolithic Metal kernel for the whole gate/up/SwiGLU/down unit if that
  proves practical.

The latter would be the only form that naturally removes the stage dependency
without split execution or a barrier.

## Latest Checkpoint

Current state is not fully fused. The slotwise decode path now has:

- one fused gate/up/SwiGLU kernel per routed expert;
- one down kernel per routed expert;
- no separate `sum_experts` tail in slotwise decode when the down type supports
  accumulated output.

The new Q2_K/Q4_K accumulated-down kernels reuse the existing down matvec math.
Route 0 writes into `routed_out`; routes 1..5 add into the same output buffer in
the same down dispatch. This preserves the old accumulation order while removing
the five extra add dispatches that used to combine per-expert scratch rows.

The ICB cache key for inherited-buffer replay is now route-independent. The
replay object is keyed by layer, resident slot, pipeline, canonical args, and
usage shape; route-local activation/output offsets are rebound at execution.
This is closer to the intended stable-slot behavior: a top-k position change no
longer forces a new ICB for the same resident expert slot.

Short `-n 8` correctness checks after this change:

- slot32 accumulated-down default: baseline hash matched,
  generation about `5.09 t/s`.
- slot256 accumulated-down default: baseline hash matched,
  generation about `0.31 t/s`.
- slot32 stable ICB replay: baseline hash matched,
  `hits=1798 misses=2330 builds=2330 execs=4128 invalidations=48`,
  generation about `4.63 t/s`.
- slot256 stable ICB replay: baseline hash matched,
  `hits=1808 misses=2320 builds=2320 execs=4128 invalidations=0`,
  generation about `0.28 t/s`.

Before route-independent inherited-buffer keys, the same slot256 stable ICB run
had about `hits=820 misses=3308`. The improvement confirms that route-specific
offsets were causing unnecessary rebuilds. Remaining misses are mostly first
use of layer/expert slots in this short trace; full-resident `slot == expert_id`
preload should remove eviction invalidations but will not remove first-use
builds unless replay objects are proactively built.

Plain full-resident layout note: if `--moe-slot-bank 256` preloads every layer
with `slot == expert_id`, resident weight offsets become constants and do not
need to be part of the dynamic handout. Route-local activation/output offsets
still vary with top-k position unless we also introduce per-slot activation
scratch or a different reduction layout.

Slot32 is now the main fused-workflow measurement. Slot256 remains useful as a
locality/data-layout falsifier, but should not be part of the routine fusion
loop because the current large-bank slowdown looks dominated by locality.

Slot32 `-n 50` checkpoint:

- accumulated-down default stdout hash:
  `802e0cf3941e4d3536081a4bf9a9899d500730146eeb0545191e4770322cb4e7`
- stable ICB replay stdout hash:
  `802e0cf3941e4d3536081a4bf9a9899d500730146eeb0545191e4770322cb4e7`
- `cmp` result: identical stdout
- accumulated-down default generation: `9.74 t/s`
- stable ICB replay generation: `9.61 t/s`
- stable ICB replay counters:
  `hits=16048 misses=9752 hit=62.2% builds=9752 execs=25800 failures=0 invalidations=6704`

## Layer-Major Slab Locality Checkpoint

Added an env-gated diagnostic layout:

```text
DS4_FLASH_MOE_LAYER_SLOT_SLAB=1
```

This only applies when mixed slot banks are enabled. The current default mixed
layout already stores a whole expert record in one resident slot per layer:
`[slot][gate,up,down]`. The slab mode changes allocation granularity by using
one large GPU allocation laid out as `[layer][slot][expert-record]`, then
creating the existing per-layer and per-family views from that slab.

Correctness, small model, `-n 50`, stable ICB replay:

- slot32 slab hash:
  `802e0cf3941e4d3536081a4bf9a9899d500730146eeb0545191e4770322cb4e7`
- slot64 default and slab hash:
  `2e7de0af3fd6ffbdc7cd10d88377ce5ba622a66b4a2f36c227c2913353bc3652`
  with `cmp=0`
- slot128 default and slab hash:
  `f6fb1e8767d750ca1baf3fb6f29686bcdd5faa776d9f2afa5768399d0a0af58d`
  with `cmp=0`

Speed breadcrumbs:

- slot32 default stable ICB from the prior run: `9.61 t/s`
- slot32 slab stable ICB: `9.97 t/s`
- slot64 default stable ICB: `9.72 t/s`
- slot64 slab stable ICB: `8.65 t/s`
- slot128 default stable ICB: `8.37 t/s`
- slot128 slab stable ICB: `8.00 t/s`

Replay counters matched within each same-slot comparison:

- slot64 default/slab:
  `hits=16736 misses=9064 hit=64.9% builds=9064 execs=25800 invalidations=864`
- slot128 default/slab:
  `hits=16506 misses=9294 hit=64.0% builds=9294 execs=25800 invalidations=0`

Interpretation: the single global layer-major slab is correctness-preserving,
but it is not the main locality fix. It slightly helped slot32 in one run, then
hurt slot64 and slot128. The stronger locality falsifier is still full-resident
preload with `slot == expert_id`, because that removes lazy install/eviction
noise rather than only changing allocation granularity.

## Prefill And Decode Slot Reuse

Resident slot meaning is now clear:

- In mixed/slab layout, one resident slot contains one whole expert record:
  gate, up, and down all live inside the same `slot * expert_stride` record.
- Decode reads from that resident slot and can skip disk when
  `expert_to_slot[expert]` is valid.
- Prefill has separate rotating staging banks for compute. Those are not
  resident slots; they are scratch.
- If prefill finds the expert already resident, it computes from the resident
  slot views.
- If prefill marks an expert for slot-cache install, it can compute from scratch
  and also install that same expert into the resident slot for decode reuse.
- If neither case applies, a prefill-loaded expert is scratch-only and is not
  preserved for decode.

The `-n 50` locality runs did not print a final `prefill slot-cache refs=...`
counter line, so do not assume prefill-installed resident reuse happened in
those tests. The `prefill I/O ... topk=16` banner is cross-layer async prefetch
top-K, not the resident slot-cache top-K.

TODO: make the prefill-to-decode reuse policy explicit in the next experiment.
Either force/test `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK`, or change cache
candidates to install into resident slots before/while computing so a prefill
expert read is not wasted. Separately, consider extending
`DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE` to release the Flash-MoE rotating
staging banks, with lazy reallocation for resumed prefill.

## Deferred Full-Resident Locality Test

After replay buffers are completely fused, run a naive full-resident locality
falsifier:

- require `--moe-slot-bank >= DS4_N_EXPERT`;
- preload every layer so `slot == expert_id`;
- mark all `expert_to_slot` / `slot_to_expert` entries resident;
- decode with no eviction and no SSD reads;
- compare separate-family layout against mixed expert-major layout.

This is not a memory-allocation change for `--moe-slot-bank 256`; that already
allocates the full per-layer slot table. The missing piece is preload policy:
today slot 256 still installs lazily, so the short `-n 8` runs only populate
about 25 experts per layer.

## Decode Prefetch Cliff Checkpoint

The large-bank cliff is strongly tied to decode temporal prefetch, especially
direct pread into resident Metal slot buffers.

Small model resident-bank sizes:

```text
slots  resident bank
32      9.07 GiB
64     18.14 GiB
96     27.21 GiB
128    36.28 GiB
160    45.35 GiB
192    54.42 GiB
224    63.49 GiB
232    65.76 GiB
234    66.33 GiB
235    66.61 GiB
236    66.89 GiB
240    68.03 GiB
256    72.56 GiB
```

Short `-n 10` cliff probes with default decode prefetch:

```text
128  36.28 GiB  4.48 t/s
136  38.55 GiB  4.45 t/s
144  40.82 GiB  3.74 t/s
160  45.35 GiB  3.65 t/s
192  54.42 GiB  3.60 t/s
224  63.49 GiB  2.94 t/s
232  65.76 GiB  3.25 t/s
234  66.33 GiB  3.21 t/s
235  66.61 GiB  1.85 t/s
236  66.89 GiB  0.31 t/s
240  68.03 GiB  0.32 t/s
256  72.56 GiB  0.28 t/s
```

Exact `-n 8` A/B at the boundary:

```text
236 default decode prefetch on, direct slot pread on: 0.41 t/s
236 decode prefetch off:                            3.00 t/s
236 all prefetch knobs off:                         2.53 t/s
236 decode prefetch on, shared-down overlap off:    1.95 t/s
236 decode prefetch on, io-split=1:                 2.75 t/s
236 decode prefetch on, direct slot pread off:      3.06 t/s
```

All rows above produced the same stdout hash:

```text
efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04
```

Interpretation:

- The cliff is not explained by replay misses; ICB replay counters are identical
  across the A/B rows.
- Default decode prefetch launches up to one read job per routed miss in each
  layer. In the `-n 8` profile it made `344` prefetch calls, loaded `1088`
  expert records, and read about `7344 MiB`.
- With direct slot pread enabled, those read threads write SSD data directly
  into scattered offsets of the huge resident `MTLBuffer`.
- Disabling only direct slot pread, while keeping decode prefetch and shared-down
  overlap enabled, removes the catastrophic cliff at 236 slots in the short
  test.
- Reducing `DS4_FLASH_MOE_CACHE_IO_SPLIT` from 4 to 1 also removes most of the
  cliff, which suggests the damage is from parallel CPU writes into the large
  shared Metal buffer while GPU work is in flight, not from allocation alone.

Added a targeted knob:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_DIRECT_SLOT_PREAD=0
```

Decode prefetch direct slot pread is now off by default on this experiment
branch. Blocking decode installs still use the broad
`DS4_FLASH_MOE_DIRECT_SLOT_PREAD` no-copy path unless that is disabled
separately, but the overlapped/speculative decode prefetch now defaults to:

```text
SSD -> temp CPU buffer -> controlled upload into resident Metal slot
```

This avoids background read threads writing directly into scattered offsets of a
huge resident `MTLBuffer` while GPU work is in flight.

The startup banner now calls this `router-prefetch` instead of
`temporal-prefetch`. "Temporal" was a confusing name from the earlier
prior-token/router-history idea. In the current non-oracle path, decode first
runs the router for the current token/layer, reads those true expert IDs back,
then prefetches any missing resident slots before routed-MoE execution.

Added another decode-prefetch throttle:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
```

Decode prefetch was changed to support scratch-prefetch:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_SCRATCH_ONLY=1
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_DECODE_PREFETCH_LAYER_STRIDE=1
```

Default is now `MAX_LOADS=0` (off). When enabled, the async part reads only into
a reusable decode scratch buffer shared across all layers. It does not write
directly into the resident slot bank. After the overlapped shared expert work is
committed, the scratch path uses the blocking flush path to drain pending GPU
work before installing any fully prefetched expert into the resident slot. This
keeps possible SSD read overlap but avoids CPU writes into the giant resident
`MTLBuffer` while GPU work is still active.

`MAX_LOADS>1` is sequential, not parallel: one scratch worker reads candidate
missing experts in order. The worker checks an interrupt flag before each expert
record and between 1 MiB chunks. When the compute window closes, unfinished
scratch reads are discarded and the normal synchronous decode install path reads
the expert if it is still needed.

Prefetch jobs are miss-driven. The code only starts a scratch read after
`metal_graph_flash_moe_reserve_decode_slot()` reports a real resident miss.
Resident hits and duplicate routed experts do not start prefetch jobs.

Setting `DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0` disables async decode
prefetch without needing to also set `DS4_FLASH_MOE_DECODE_PREFETCH=0`. Setting
it to `1..6` enables up to that many async expert-record scratch reads per
layer. If a layer has more misses than the throttle, decode prefetch completes
the overlapped scratch loads and then falls back to the normal synchronous
decode prepare/install path for the remaining experts before routed execution.
`DS4_FLASH_MOE_DECODE_PREFETCH_LAYER_STRIDE=N` attempts scratch-prefetch only on
every Nth layer.

First A/B after adding the throttle:

```text
236 max-loads=1, direct slot pread off: 3.09 t/s
256 max-loads=1, direct slot pread off: 0.40 t/s
256 max-loads=0:                        1.96 t/s
256 max-loads=1, shared-down off:        2.66 t/s
256 scratch-prefetch max-loads=1:        2.66 t/s
256 scratch-prefetch max-loads=2:        2.59 t/s
256 scratch-prefetch max-loads=3:        1.70 t/s
256 scratch-prefetch max-loads=6:        2.24 t/s
```

Those `-n 8` smoke tests were not enough. With 50-token runs after making
scratch-prefetch truly sequential/interruptible:

```text
256 max-loads=0:                         6.86 t/s
256 max-loads=1 layer-stride=1:           killed after >90s, too slow
256 max-loads=1 layer-stride=3:           killed after >60s, too slow
```

Interpretation: for this IQ2XXS expert-major model, one expert record is still
too large to prefetch inside the decode compute window. The scratch read often
does not finish, gets canceled, and then the synchronous path reads the same
expert again. Scratch-prefetch is therefore useful as an experiment but should
stay default-off for this model until we either shrink the prefetched unit or
find a longer compute window.

So even one overlapped decode-prefetch load can be destructive at 256 slots when
the CPU slot install overlaps shared expert GPU work. Turning shared-down overlap
off recovers the run, which points to a unified-memory/coherency/residency
interaction, not merely to SSD read count. Keep async decode prefetch
scratch-only until we redesign it as a truly narrow layer-adjacent scheduler or
replace it with a safer preload/reuse strategy.

Relevant mechanics:

- `ds4_gpu_flush_commands()` commits the shared expert work but does not wait.
- `ds4_gpu_tensor_write()` is a plain `memcpy()` into `[MTLBuffer contents]`.
- With `MAX_LOADS=1`, decode prefetch can therefore upload the prefetched slot
  into the resident bank before the flushed GPU work has drained.
- With `MAX_LOADS=0`, the normal decode prepare path calls `end_commands()`
  before installing experts, which drains pending command buffers first.

This explains why 32 slots can tolerate the overlap while 256 slots cannot: the
same 6.75 MiB expert install lands inside a much larger resident shared Metal
allocation (`9.07 GiB` vs `72.56 GiB`), and at that size the CPU write plus
pending GPU work appears to trigger a severe memory-system/residency penalty.

With decode prefetch off, slot count still matters in 50-token probes:

```text
32 slots   9.07 GiB  15.02 t/s
64 slots  18.14 GiB  14.26 t/s
128 slots 36.28 GiB  12.14 t/s
256 slots 72.56 GiB   6.86 t/s
```

So the remaining big-slot slowdown is no longer decode prefetch. It points at
resident-bank size/layout or routed execution against a very large shared Metal
allocation.

Longer 500-token probes, same prompt/settings, stdout redirected and hashes
checked:

```text
32 slots   9.07 GiB  17.18 t/s
64 slots  18.14 GiB  18.36 t/s
128 slots 36.28 GiB  16.70 t/s
256 slots 72.56 GiB  12.22 t/s
```

All four stdout hashes matched:

```text
5d4933a583f22ceccc9dfcaf026cef01c37e4eb4517ea9174520c4bd71f0abd1
```

Interpretation: longer runs warm the resident bank and are much faster than
50-token cold probes, but 256 slots still pays a real steady-state penalty.
The likely next falsifier is a resident-bank layout/execution experiment, not
more decode prefetch tuning.

Next experiment: rerun the 236/240/256 short probes with the new defaults, then
A/B `DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=6` and
`DS4_FLASH_MOE_DECODE_PREFETCH_DIRECT_SLOT_PREAD=1` independently.

## YOLO Slot64 No-Prefetch Checkpoint

All runs in this checkpoint used:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
./ds4 -m ~/Models/flash/dsv4-iq2xxs-expert-major \
  --ctx 60768 --temp 0 -p "Who are you" -n 500
```

ICB means Metal Indirect Command Buffer. Important flag correction:
`DS4_FLASH_MOE_ICB_FUSED_UNIT=1` alone does not enable ICB replay. The real
full routed-unit ICB stack is:

```text
DS4_FLASH_MOE_STABLE_REPLAY=1
DS4_FLASH_MOE_ICB_REPLAY=1
DS4_FLASH_MOE_ICB_PAIR_SWIGLU=1
DS4_FLASH_MOE_ICB_FUSED_UNIT=1
```

Without `DS4_FLASH_MOE_ICB_REPLAY=1`, the fused-unit flag is a no-op for the
ICB cache; a stats probe printed `builds=0 execs=0`.

Slot64 A/B before the mid-only activation-store change:

```text
grouped baseline                         16.52 t/s
slotwise                                 16.48 t/s
stable replay                            16.76 t/s
stable + ICB_REPLAY down-only            13.87 t/s
stable + ICB_REPLAY pair split           14.02 t/s
stable + ICB_REPLAY pair+down fused      14.53 t/s
stable + fused ICB, no explicit barrier  14.65 t/s
stable + fused ICB, route0 only          16.54 t/s
```

All slotwise/stable/ICB variants matched stdout hash:

```text
bf0d6057d316306de5f427b565edc1446737ff8105d23fb2ec389ca7b7b8e316
```

Grouped kept its separate hash:

```text
5d4933a583f22ceccc9dfcaf026cef01c37e4eb4517ea9174520c4bd71f0abd1
```

The real full-ICB stats probe at slot64 `-n 64` showed route/unit replay is not
currently a win:

```text
hits=6841 misses=9671 hit=41.4% builds=9671 execs=16512 failures=0 invalidations=1048 cache=4096/4096
```

Temporarily raising `DS4_FLASH_MOE_ICB_CACHE_MAX` to `32768` avoided the hard
cache cap but did not improve speed:

```text
n64 stats: hits=8206 misses=8306 builds=8306 execs=16512 invalidations=1917 cache=6389/32768
n500 speed: 13.34 t/s
```

Interpretation: current two-command ICB replay still bakes too much
route/top-k-local state and allocates/builds too much state to beat ordinary
encoding. The `route0 only` result recovering most speed strengthens that
diagnosis. For now, ICB should remain diagnostic, not the best-speed path.

Low-risk fuse improvement implemented after that: the IQ2_XXS fused
gate/up/SwiGLU kernel now skips dead gate/up scratch stores unless
`act.write_clamped != 0`. Normal decode only consumes `mid`, so this preserves
the output while reducing memory traffic.

Post-change slot64:

```text
grouped, no stable replay        17.87 t/s  hash 5d4933...
stable slotwise replay           17.34 t/s  hash bf0d60...
stable replay, down accum off    15.32 t/s  hash bf0d60...
```

Stable slotwise replay slot sweep after the same change:

```text
32 slots    9.07 GiB  16.16 t/s
64 slots   18.14 GiB  17.34 t/s
128 slots  36.28 GiB  16.70 t/s
256 slots  72.56 GiB  11.23 t/s
```

All stable slot counts produced the same stdout hash:

```text
bf0d6057d316306de5f427b565edc1446737ff8105d23fb2ec389ca7b7b8e316
```

Current best-speed flags for the stable-bank workflow:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_STABLE_REPLAY=1
--moe-slot-bank 64
```

Current fastest observed slot64 decode is grouped (`17.87 t/s`), but that is
still the grouped top-6 execution shape. The stable-bank correctness workflow is
stable slotwise replay (`17.34 t/s`) because it executes resident per-expert
slots and keeps the replay/cache unit aligned with individual experts.

## PyGame Prompt Benchmark Switch

The current benchmark prompt is now:

```text
make a game of Space invaders in PyGame
```

All runs in this checkpoint used:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
./ds4 -m ~/Models/flash/dsv4-iq2xxs-expert-major \
  --ctx 60768 --temp 0 -p "make a game of Space invaders in PyGame" -n 500
```

Original slot64 grouped vs stable before the correctness fix:

```text
grouped, no stable replay   17.60 t/s  hash 4936a2...
stable slotwise replay      17.39 t/s  hash ab4a2b...
```

Stable slotwise replay slot sweep:

```text
32 slots    9.07 GiB  14.68 t/s
48 slots   13.60 GiB  16.18 t/s
64 slots   18.14 GiB  17.39 t/s
96 slots   27.21 GiB  16.85 t/s
128 slots  36.28 GiB  17.11 t/s
```

All stable slot counts produced identical stdout:

```text
ab4a2b0851afcff87ee443ca42b146805d4d02e1abfc3b28d4ea6776dfa1495e
```

Grouped slot64 originally produced a different hash and longer output:

```text
stable slot64:  1641 bytes
grouped slot64: 1806 bytes
```

Correctness fix: grouped decode was using the specialized
`kernel_mul_mv_id_*_sum6_f32` down-reduction path by default. Disabling that
path made grouped output byte-identical to stable slotwise output, and was
faster in the PyGame prompt test. `sum6` is now opt-in:

```text
DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=1
```

Default grouped and stable after making `sum6` opt-in:

```text
PyGame prompt grouped slot64 default   19.58 t/s  hash ab4a2b...
PyGame prompt stable slot64 default    19.38 t/s  hash ab4a2b...
Who prompt grouped slot64 default      16.81 t/s  hash bf0d60...
Who prompt stable slot64 default       16.88 t/s  hash bf0d60...
```

Both prompts compared byte-identical between grouped and stable after the fix
(`cmp=0`). The earlier stable slot sweep still shows slot64 as the best
stable-bank point among the tested slot counts for the PyGame prompt, but the
headline correctness result is now grouped/stable parity with `sum6` default-off.

Logging caveat for the `19.58 t/s` result: the generated text was redirected to
`/tmp`, but the speed line was printed on stderr and was not captured in a
separate `.err` file at that time. The progress note above is therefore the
timing log; the `/tmp` files preserve output bytes and mtimes.

Artifact sequence around the fast run:

```text
02:24:12 /tmp/ds4_pygame_slot64_stable_n500.out
02:24:56 /tmp/ds4_pygame_slot64_grouped_n500.out
02:30:14 /tmp/ds4_correct_pygame_slot64_grouped_nopairfuse_n500.out
02:30:53 /tmp/ds4_correct_pygame_slot64_stable_nopairfuse_n500.out
02:32:00 /tmp/ds4_correct_pygame_slot64_grouped_nosum6_n500.out
02:33:09 /tmp/ds4_correct_pygame_slot64_grouped_sum6defaultoff_n500.out
02:33:55 /tmp/ds4_correct_pygame_slot64_stable_sum6defaultoff_n500.out
02:34:24 /tmp/ds4_correct_who_slot64_grouped_sum6defaultoff_n500.out
02:34:41 /tmp/ds4_correct_who_slot64_stable_sum6defaultoff_n500.out
```

Hypothesis: the `19.58 t/s` grouped run may have benefited from warm OS file
cache / driver state because several same-model slot64 runs happened
immediately before it. Current repeated warm-cache runs did not reproduce
`19.58` under today’s system state, so cache warmth is plausible but not enough
by itself.

Follow-up from direct Terminal, same current binary and normal command (no
explicit replay env flags):

```text
./ds4 \
  -m ~/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 64 \
  --ctx 60768 \
  -p "make a game of Space invaders in PyGame" \
  --temp 0 \
  -n 500

decode banner:
router-prefetch=off scratch-prefetch=off max-loads=0 layer-stride=1
miss-direct-slot-pread=on prefetch-direct-slot-pread=off shared-down=on
slots=64

Terminal run 1: generation 20.87 t/s
Terminal run 2: generation 20.82 t/s
```

This reproduces/exceeds the old `19.58` target outside the Codex tool-run
environment. The lower `exec_command` measurements around `17.9-18.9 t/s` are
therefore likely measurement-environment artifacts (background WindowServer /
Codex UI / process launch context / competing tools), not evidence of a grouped
code regression. Keep using direct Terminal or a standalone benchmark wrapper
for headline speed numbers, and capture `2>&1` logs for exact timing context.

Three-way Terminal comparison from `/tmp/ds4_cmp_*.log`, same prompt/slot64/n500:

```text
grouped baseline                  20.07 t/s  generated-text hash ab4a2b...
stable slot                       20.31 t/s  generated-text hash ab4a2b...
dynamic grouped / async stable    18.05 t/s  generated-text hash ab4a2b...
async wait-all then grouped       19.19 t/s  generated-text hash ab4a2b...
```

Interpretation: stable-slot execution itself is not the speed problem on this
trace; grouped and stable-slot are effectively tied and produce identical text.
The loss is specific to the async handout policy/path. With current defaults,
any layer with misses usually falls into the split route path: compute resident
routes, wait/upload missing experts, compute missing routes, then sum. That
replaces the optimized grouped MoE call with smaller route-level work and extra
submit/sum overhead. On this warm-cache slot64 trace, the missing-expert read
latency is apparently not large enough to pay for that split.

The wait-all grouped test used:

```text
DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_MAX=6
DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_FORCE=1
```

That recovered most, but not all, of the split-async loss. This means two
things at once:

- The current split selection is wrong for this warm slot64 trace.
- Even when async falls back to grouped after loading misses, the async handout
  machinery still has measurable overhead versus plain grouped/stable.

## Async Handout Checkpoint

Added a correctness-first async handout experiment:

```text
DS4_FLASH_MOE_ASYNC_HANDOUT=1
```

This is only active with stable/baked slot decode. It answers the grouped-vs-slot
question this way:

- if all routed experts are already resident, grouped execution can be the fast
  path because it submits the whole routed set together;
- if any routed expert is missing, grouped execution is the wrong scheduling
  unit because it waits for the missing pread before computing resident experts;
- the async handout path starts miss preads, computes resident routes first, then
  joins/uploads missing routes and computes them in route order;
- each route writes its down result into a stable scratch row, and a final
  ordered sum preserves the same accumulation order as the known-correct stable
  path.

The current implementation drains GPU commands before uploading a newly read
expert into the resident bank. That avoids background CPU writes into the large
shared Metal slot buffer while GPU work is still in flight. It is not yet the
fast design: v0 uses one-route submits plus a final sum, so it proves the
scheduler/correctness shape before optimizing submit count.

PyGame prompt, slot64, no decode prefetch:

```text
n32 async handout    hash e589d9c6...  cmp=0 vs default stable
n32 default stable   12.19 t/s
n32 async handout     8.57 t/s

n500 async handout   hash ab4a2b...  cmp=0 vs default stable
n500 default stable  19.38 t/s
n500 async handout   15.72 t/s
```

Interpretation: async handout is now correctness-clean for the 500-token PyGame
prompt, but speed still needs the next scheduler layer. The next optimization is
to measure the break-even between one grouped submit when all routed experts are
resident, per-route submits when misses exist, and small chunks such as two or
three submits when a partial resident set is available.

Adaptive update:

- When `DS4_FLASH_MOE_ASYNC_HANDOUT=1` and a layer has zero routed misses, the
  experiment now executes the grouped banked routed-MoE call immediately.
- When a layer has misses, the default async-handout path still computes resident
  routes first and late miss routes afterward.
- `DS4_FLASH_MOE_ASYNC_HANDOUT_OVERLAP_MISSES=0` keeps the all-resident grouped
  fast path but joins/uploads miss reads and falls through to the normal stable
  routed execution for miss layers. This is a diagnostic for submit overhead.

Correctness bug found while testing the fallback: decode reservation could pick
a victim slot for route A, then a later route B could still "hit" that reserved
slot before route A's eviction was committed. Full async happened to compute B
before uploading A, but the sync-miss fallback uploaded A first and then B read
the overwritten slot. `metal_graph_flash_moe_reserve_decode_slot()` now rejects
hits on slots already reserved by the same decode step.

After the reserved-slot fix, PyGame prompt, slot64, no decode prefetch,
500-token stats probes:

```text
async overlap        16.17 t/s  hash ab4a2b...  cmp=0
sync-miss fallback   16.66 t/s  hash ab4a2b...  cmp=0
```

Branch mix from the same 500-token trace:

```text
async handout calls=21500
grouped all-resident=10116 (47.1%)
miss-calls=11384
miss-routes=21025
avg-miss-routes=1.85
```

Clean no-stats sync-miss fallback:

```text
sync-miss fallback   16.74 t/s  hash ab4a2b...  cmp=0
```

Conclusion: the adaptive grouped all-resident fast path is correct, but it is
not yet a speed win over the ordinary corrected stable path on this model. The
miss path still dominates, and one-route async compute loses to submit overhead.
The next useful scheduler test is chunking resident routes into a small number
of submits, or falling back to grouped execution after misses are installed when
the expected overlap window is too small.

Submit parallelism check:

- The async handout route splits are not parallel Metal submits today.
- `ds4_gpu_end_commands()` calls `ds4_gpu_finish_command_buffer()`, which commits
  the command buffer, waits all pending command buffers, then waits the current
  command buffer.
- The route-split miss path uses `ds4_gpu_end_commands()` before uploads and
  between missing routes, so those route submits are serialized.
- A fusion-profile probe printed repeated
  `owned_commit+wait=1 flush=0 enc_end=1 wait=1` boundaries, confirming the
  hard waits.
- The only overlap in the current async-handout path is CPU pread threads
  running while resident GPU work is executing. The resident/miss Metal route
  submits themselves are not running in parallel.

Parallel-submit experiment:

```text
DS4_FLASH_MOE_ASYNC_HANDOUT_PARALLEL_SUBMITS=1
```

This replaces async route-split `end_commands()` hard waits with
`flush_commands()` no-wait submits. It still uses the same Metal command queue,
so this is not guaranteed multi-queue GPU concurrency; it is a no-wait queued
submit that allows CPU uploads/reads to proceed while earlier route work is in
flight.

Correctness and speed, PyGame prompt, slot64, no decode prefetch:

```text
n32  parallel submits      9.39 t/s  hash e589d9...  cmp=0
n500 parallel submits     17.61 t/s  hash ab4a2b...  cmp=0
n500 stats probe          17.78 t/s  hash ab4a2b...  cmp=0
```

Miss-count histogram from the 500-token stats probe:

```text
miss-hist[0..6]=10116,6292,2712,1118,584,449,229
```

So the scheduler sees many 0-miss grouped calls, but the miss side is dominated
by one- and two-miss layers. This means the best policy probably depends on miss
count and read timing, not a single global mode.

Chunked-miss experiments:

```text
DS4_FLASH_MOE_ASYNC_HANDOUT_CHUNK_MISSES=1
DS4_FLASH_MOE_ASYNC_HANDOUT_CHUNK_MISS_MIN=N
```

Chunking all miss routes after resident work was correct but not faster for the
500-token trace:

```text
n32  chunk all misses      10.74 t/s  hash e589d9...  cmp=0
n500 chunk all misses      17.47 t/s  hash ab4a2b...  cmp=0
n500 chunk miss >= 2       16.61 t/s  hash ab4a2b...  cmp=0
```

Important correction: grouped-first is still the baseline and still the fastest
known path. The async-handout experiments below are condition searches, not a
replacement for grouped execution. The right question is:

```text
When, if ever, does async split beat "install misses, then run grouped"?
```

So far that condition has not been found.

Current best async-handout scheduler for this model remains no chunking:
grouped for 0 misses, no-wait per-route submits for miss layers. But this is
still slower than grouped-first, so it should stay experimental.

Clarification after reviewing the miss-count policy: the chunk experiment above
only changed how already-missing routes are submitted after resident work. It
did not test the separate decision of whether to wait for a small number of
misses before starting resident work.

Added:

```text
DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_MAX=N
DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_FORCE=1
```

With `WAIT_MISS_MAX=1`, the scheduler can wait for a one-miss layer and run the
grouped banked call after upload. The normal read thread now marks jobs
complete, so the non-forced path only takes this grouped wait path when the
small miss read is already complete. `WAIT_MISS_FORCE=1` reproduces the
unconditional wait experiment.

Results, PyGame prompt, slot64, no decode prefetch:

```text
n32  wait-one forced/default-at-that-time  10.46 t/s  hash e589d9...  cmp=0
n500 wait-one forced/default-at-that-time  16.08 t/s  hash ab4a2b...  cmp=0
n500 wait-one ready-only                   16.95 t/s  hash ab4a2b...  cmp=0
n500 wait disabled                         17.09 t/s  hash ab4a2b...  cmp=0
```

Conclusion: the intuition is right that one missing expert changes the optimal
submit shape, but for this trace unconditional waiting sacrifices too much
overlap. Ready-only waiting was also not a win. The default is therefore
`WAIT_MISS_MAX=0`; the wait/grouped path remains as an explicit experiment.
For 2+ misses, starting resident work immediately is still the intended async
direction only if future measurements show it beats grouped-first.

Current-build grouped-first sanity check, same prompt/settings, after the async
and ICB experiments:

```text
grouped-first current run 1       16.57 t/s  hash ab4a2b...  cmp=0
grouped-first current run 2       16.73 t/s  hash ab4a2b...  cmp=0
async split current comparison    16.42 t/s  hash ab4a2b...  cmp=0
```

These current-run speeds are lower than the earlier 19.6-ish grouped baseline,
so we checked whether grouped degraded due to shared experiment overhead.
Separate-family layout was slower (`15.15 t/s`), so the mixed expert-major
layout was not the regression source. A 45-second cooldown recovered grouped to
`17.07 t/s`, which suggests thermal/run-state noise was part of it but not all
of it.

Stability / ICB replay finding:

- The async route helper is stable only in the resident-weight sense. It still
  creates route-local gate/up/mid/down/weight tensor views and encodes fresh
  commands.
- The two-command fused-unit ICB key includes route-local buffers/offsets and
  `route_index`, so it is not the stable per-slot execution object we want.
- Split pair/down ICB with inherited buffers is more stable, but still slower
  and creates two replay entries per layer/slot.
- Temporarily raised the compiled diagnostic ICB cache cap from 4096 to 32768
  so slot64 could hold the expected split entries (`43 * 64 * 2 = 5504`) for
  testing.

ICB stats, async parallel submits, PyGame prompt, slot64:

```text
n64 fused-unit ICB, cache 4096    hit=31.6%  generation 12.24 t/s  cmp=0
n64 split ICB, cache 4096         hit=56.9%  generation 11.87 t/s  cmp=0
n64 split ICB, cache 32768        hit=63.3%  generation 10.45 t/s  cmp=0
n500 split ICB, cache 32768       hit=69.2%  generation 14.94 t/s  cmp=0
```

Grouped regression check after reverting the compiled ICB cache cap back to
4096:

```text
grouped, cache 32768, cooldown    17.07 t/s  hash ab4a2b...  cmp=0
grouped, cache 4096, cooldown     18.52 t/s  hash ab4a2b...  cmp=0
grouped, cache 4096, repeat       18.66 t/s  hash ab4a2b...  cmp=0
```

Corrected diagnosis: grouped slot install/eviction was calling
`ds4_gpu_flash_moe_icb_invalidate_slot()` through the replay invalidation path.
That function scanned `DS4_FLASH_MOE_ICB_CACHE_MAX` entries, so the cache cap
affected grouped even when ICB replay was not being used. This was replay
maintenance leaking into the baseline.

Fix applied:

- `ds4_gpu_flash_moe_icb_invalidate_slot()` now returns immediately unless ICB
  replay is enabled.
- Flash-MoE replay-plan maintenance (`flash_decode_*` ids,
  `flash_replay_slot_*` state, replay-plan hit/miss counters) now runs only
  when baked/stable slot decode is enabled.

Post-gating baseline reset. Common command args:

```text
./ds4 \
  -m ~/Models/flash/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 64 \
  --ctx 60768 \
  -p "make a game of Space invaders in PyGame" \
  --temp 0 \
  -n 500
```

Explicit replay infrastructure off:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_STABLE_REPLAY=0
DS4_FLASH_MOE_BAKED_SLOT_DECODE=0
DS4_FLASH_MOE_ASYNC_HANDOUT=0
DS4_FLASH_MOE_ICB_REPLAY=0
DS4_FLASH_MOE_METAL_DECODE_REPLAY=0
generation 18.93 t/s  hash ab4a2b...  cmp=0
```

Normal current default after baseline gating:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
/* stable replay, baked slot decode, async handout, and ICB replay are unset */
/* generic Metal decode replay is left at its default */
generation 18.79 t/s  hash ab4a2b...  cmp=0
```

Normal current default after adding indexed ICB maintenance:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
/* same env as normal current default above; code state includes ICB generation/index hints */
generation 18.71 t/s  hash ab4a2b...  cmp=0
```

Clean `Pro-alpha`, old default hash path:

```text
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
/* on clean Pro-alpha this does not disable the old temporal prefetch path */
generation 17.97 t/s  hash 0ae6af...  cmp=1
```

Clean `Pro-alpha`, temporal prefetch off and old down-sum disabled:

```text
DS4_FLASH_MOE_DECODE_PREFETCH=0
DS4_METAL_DISABLE_ROUTED_DOWN_SUM6=1
generation 17.92 t/s  hash 4936a2...  cmp=1
```

Conclusion: baseline grouped is back above the clean branch under current run
conditions while preserving the corrected output. Keep the compiled ICB cap at
4096 until replay-on maintenance is rewritten. The next ICB attempt must avoid
hidden baseline overhead and must use indexed or lazy invalidation instead of a
global cache scan.

Replay-on maintenance reset:

- Added per-layer/slot ICB generations.
- Added small per-layer/slot/kind cache hints so hot ICB hits can avoid the
  global cache scan.
- ICB invalidation now bumps the slot generation and clears only the slot's
  local hints.

First indexed-ICB replay test, PyGame prompt, slot64, n64:

```text
grouped baseline after ICB index      14.97 t/s  hash 1828e1...  cmp=0
split ICB replay after ICB index      11.82 t/s  hash 1828e1...  cmp=0
ICB stats: hits=13014 misses=9846 hit=56.9% builds=9846 execs=22860
           failures=0 invalidations=4195 cache=4096/4096
```

Conclusion: the indexed maintenance fixes the hidden baseline-overhead class of
bug, but current ICB replay is still not a speed path. The remaining loss is
the replay object shape/execution overhead: too many ICB entries, too many
builds, and still a route-local execution plan. Do not optimize around ICB
until the replay key/object is redesigned around stable resident layer/slot
state.

Conclusion: current ICB replay is useful for diagnosing stability, but it is
not the speed path yet. A real stable execution object should be per resident
layer/slot and should treat route position, weight scalar, and scratch/output
offsets as rebound data rather than cache-key identity.

## Open Questions

- Does one expert-major resident buffer remove the slot32 vs slot256 slowdown?
- Is the slowdown dominated by memory layout/cache pressure, by many small dispatches, or by the 6-expert grouped execution unit?
- After mixed residency is correct, should the replay object be per expert, per slot, or per layer-slot-family bundle?
- Is a monolithic gate/up/SwiGLU/down kernel worth pursuing, or are two fused kernels plus per-expert replay sufficient?
- What is the submit break-even for grouped all-resident execution vs per-route
  async handout vs two/three route chunks?
- Which slot eviction policy wins on long traces: LRU, LFU, or a
  recency/frequency hybrid?
- Which slot64 prefetch strategy helps after correctness is stable: no prefetch,
  one scratch load, layer-strided scratch loads, or a narrower preloaded unit?
