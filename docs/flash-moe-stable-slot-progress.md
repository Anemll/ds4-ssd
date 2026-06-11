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

## 2026-06-05 Async Handout Review Plan

External review of the current dynamic grouped / async stable path agrees with
the local measurements:

```text
grouped baseline                  ~20.07 t/s
stable slot                       ~20.31 t/s
dynamic grouped / async stable    ~18.05 t/s
async wait-all then grouped       ~19.19 t/s
```

The combined diagnosis is that async handout loses for stacked reasons:

- the split decision happens after expensive setup: slot reservation, per-miss
  multi-MiB `xmalloc`, and per-miss `pthread_create`;
- miss layers are dominated by one or two misses, where splitting the optimized
  grouped call hides little read latency and adds route-level submit/sum cost;
- default split submits drain the GPU per missing route unless parallel submits
  are explicitly enabled;
- all reservation paths only protect routed experts incrementally, so an early
  miss can evict a later route's already-resident expert and inflate misses;
- the shared-expert work is currently the best overlap candidate because it does
  not break the grouped routed-MoE unit.

Implementation order:

1. Pre-protect the full routed top-k before victim selection.
   - Apply to `metal_graph_flash_moe_prepare_decode()`.
   - Apply to `metal_graph_flash_moe_prepare_decode_prefetch_ids()`.
   - Apply to `metal_graph_flash_moe_decode_async_handout()`.
   - Keep `reserved_slots` as the hard per-decode-step guard.

2. Refactor async handout into plan -> policy -> execute.
   - Planning resolves `true_ids`, `slot_ids`, `route_load_idx`, resident/miss
     flags, and evicted slots.
   - Planning must not allocate expert buffers or start read threads.
   - Record async miss histograms after planning so policy and stats match.

3. Make the async miss default conservative.
   - Zero-miss layers still run grouped/stable immediately.
   - Miss layers default to install/read misses and then fall through to the
     existing grouped/stable caller.
   - Route splitting is experimental only: require explicit opt-in and a
     minimum miss count, initially `n_loads >= 4`.
   - Cache `DS4_FLASH_MOE_ASYNC_HANDOUT_OVERLAP_MISSES` like the other async
     flags so the hot path does not call `getenv()` per layer.

4. Remove per-miss allocation churn.
   - Add reusable async read scratch buffers sized to `DS4_N_EXPERT_ACTIVE_USED`
     records, or share the decode-prefetch scratch pattern with a separate
     async allocation so `max-loads=0` does not disable it.
   - Start with `io_split=1` for async handout reads because misses are already
     parallel across experts; benchmark this against the old nested split-read
     fan-out before making it permanent.

5. Add minimal policy metrics.
   - Count planned miss layers by miss count.
   - Track whether joins found the read already complete.
   - Keep per-miss-count pread/upload timing so split can become a measured
     exception instead of a default.
   - Convert `job.complete` from `volatile int` to atomic or mutex-backed state
     before depending on it for ready/deadline policy.

6. Only after parity: overlap shared-expert work with miss reads.
   - Mirror the decode-prefetch ordering that already hash-matched:
     start reads, encode shared gate/up/down, flush, join/upload misses, then
     run one intact grouped/stable routed-MoE call.
   - Verify at slot64 first, then slot128/256 because CPU upload into resident
     slot buffers while flushed GPU work is in flight is the known hazard class.

7. Keep ICB and route-split kernel cleanup out of the default path.
   - ICB remains diagnostic-only.
   - Do not optimize the route-level tensor-view churn or sum-tail until split
     earns a measured cold-cache win.

Benchmark matrix for this sequence:

```text
A   grouped default, slot64, n500
B   stable slot, no async
C   current async, before refactor
D   async wait-all grouped
E   async + DS4_FLASH_MOE_CACHE_IO_SPLIT=1
E'  grouped default + DS4_FLASH_MOE_CACHE_IO_SPLIT=1
F   wait-all async + DS4_FLASH_MOE_CACHE_IO_SPLIT=1
G   after pre-protect only
H   after plan/policy + reusable scratch
I   after shared-work overlap
J   cold-cache rows A-I, if needed
```

Run direct Terminal only, one row at a time, with a cooldown and `stdout` hashes
captured. The PyGame prompt hash should remain `ab4a2b...`; grouped baseline
should be run first and last to bracket thermal drift.

## 2026-06-05 Async Handout Pass 1

Implemented the first conservative refactor pass:

- Added opt-in full top-k pre-protection before victim selection in:
  `metal_graph_flash_moe_prepare_decode()`,
  `metal_graph_flash_moe_prepare_decode_prefetch_ids()`, and
  `metal_graph_flash_moe_decode_async_handout()`.
  It is gated by `DS4_FLASH_MOE_PREPROTECT_TOPK=1` and defaults off after an
  outside-run report suggested grouped default may regress with it enabled.
- Async handout now plans all misses before starting any read thread.
- `DS4_FLASH_MOE_ASYNC_HANDOUT_OVERLAP_MISSES` is cached and now defaults off.
- Route split is opt-in and additionally gated by
  `DS4_FLASH_MOE_ASYNC_HANDOUT_SPLIT_MISS_MIN` (default `4`).
- Added `DS4_FLASH_MOE_ASYNC_HANDOUT_IO_SPLIT`; async handout reads default to
  `io_split=1` because misses are already parallelized across experts.
- Conservative async miss path reads directly into resident slots before GPU
  commands reopen, then falls through to the existing stable/grouped caller.
- Experimental split path uses lazy reusable async scratch instead of per-miss
  `xmalloc`.
- `job.complete` now uses acquire/release helper access instead of raw volatile
  reads.
- Backend stats now include async split count plus join-ready/join-wait counts.

Small correctness checks after this pass:

```text
PyGame prompt, slot64, n64:
grouped hash   1828e1f5088e...
stable hash    1828e1f5088e...
async hash     1828e1f5088e...

PyGame prompt, slot64, n32:
async conservative hash     e589d9c6337a...
forced split opt-in hash    e589d9c6337a...

PyGame prompt, slot64, n500:
async conservative hash     ab4a2b0851af...
```

Rough Codex tool-run n500 matrix after pass 1:

```text
grouped default            17.07 t/s  hash ab4a2b...
stable slot                19.22 t/s  hash ab4a2b...
async conservative         14.18 t/s  hash ab4a2b...
```

Stats probe (`DS4_FLASH_MOE_PROFILE=1`, PyGame slot64 n32):

```text
async handout calls=1376
grouped=267 (19.4%)
miss-calls=1109
miss-routes=2805
avg-miss-routes=2.53
sync-miss=1109
wait-grouped=0
split=0
join-ready=1102
join-wait=1703
miss-hist[0..6]=267,329,324,207,111,65,73
```

This confirms the new default does not enter the route-split branch. The tool-run
generation speeds are not representative of direct Terminal benchmarks, but the
relative result still shows conservative async has not yet become the speed path.

Next planned step: run the direct Terminal benchmark matrix after cooldown. If
async conservative is still below stable/grouped, implement shared-expert
overlap during miss reads before revisiting route split.

Outside-environment benchmark helper:

```sh
scripts/bench_flash_moe_async_pass1.sh
```

Default rows:

```text
grouped_first
stable_slot
async_conservative
grouped_last
```

Optional probes:

```sh
INCLUDE_SPLIT=1 scripts/bench_flash_moe_async_pass1.sh
INCLUDE_PROFILE=1 scripts/bench_flash_moe_async_pass1.sh
INCLUDE_PREPROTECT=1 scripts/bench_flash_moe_async_pass1.sh
```

Useful overrides:

```sh
N_TOKENS=32 COOLDOWN_SECONDS=0 scripts/bench_flash_moe_async_pass1.sh
N_TOKENS=500 COOLDOWN_SECONDS=45 scripts/bench_flash_moe_async_pass1.sh
```

Clean outside-Terminal run after killing stray PyGame processes:

```text
/tmp/ds4_flash_moe_async_pass1_20260605_111433
slot64, n500, all hashes ab4a2b...

grouped_first          20.29 t/s
stable_slot            20.73 t/s
async_conservative     19.16 t/s
grouped_preprotect     21.16 t/s
grouped_last           21.23 t/s
```

Interpretation: pre-protect is not the grouped-default regression source in
this run. The earlier low grouped numbers were run-state noise. Stable remains
slightly ahead of grouped, and async conservative is still behind; the next
speed item remains shared-expert overlap during miss reads.

Longer M5 Max outside-Terminal run:

```text
/tmp/ds4_flash_moe_async_pass1_20260605_112944
slot64, n2000, all hashes 80895569...

grouped_first          20.30 t/s
stable_slot            19.97 t/s
async_conservative     18.32 t/s
grouped_preprotect     20.39 t/s
grouped_last           20.32 t/s
```

Interpretation: the longer run removes the short-run ambiguity. Grouped is
stable at ~20.3 t/s, pre-protect is effectively neutral/slightly positive, and
async conservative is consistently slower. This strengthens the conclusion that
the current async miss path still adds overhead without enough overlap work.

M3 Ultra 96G run from attached logs:

```text
/Volumes/SN8100/DS/dsv4-iq2xxs-expert-major
slot64, n500, all row hashes 5c51868c...
decode shared-down banner: off

grouped_first          12.88 t/s
stable_slot            12.62 t/s
async_conservative     12.01 t/s
grouped_preprotect     12.98 t/s
grouped_last           12.86 t/s
```

Interpretation: M3U shows the same broad shape as M5 Max but at a lower
throughput level. Pre-protect is not a regression source here either; it is
slightly ahead of grouped_first/last. Async conservative remains behind grouped,
so the missing speed win is still not route splitting or wait-all; the next
candidate remains overlapping shared-expert work with miss reads while keeping
the grouped routed-MoE call intact. The first prefill row was much colder
(`3.49 t/s`) than later rows (`6.3-6.7 t/s`), so decode comparisons should use
generation speed, not prefill speed.

Pro short-run check:

```text
/tmp/ds4_flash_moe_async_pass1_20260605_150116
/Users/anemll/Models/DSv4Pro-flash
slot32, ctx32768, n32, all hashes dea9219a...
decode banner: miss-direct-slot-pread=on, prefetch-direct-slot-pread=off

grouped_first          2.20 t/s
stable_slot            2.25 t/s
async_conservative     2.40 t/s
grouped_last           2.25 t/s
```

Interpretation: correctness is good across the three rows. On this very short
Pro sample, async conservative is ahead of grouped/stable, but this should not
be treated as the final Pro policy until a longer `n500` or `n2000` run confirms
it. It is separate from the high-slot cliff: a Pro slot50 run at ctx32768/n500
collapsed to roughly `0.23 t/s`, and the small model with `--ssd-cache 64GB`
resolved to `slots=225`, `gpu-bank=63.78 GiB`, total about `76 GiB`, and also
ran at roughly `0.2 t/s`.

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

The high-slot cliff is not currently explained by an extra user-space copy:
the slow logs already showed `miss-direct-slot-pread=on`. In the default direct
path, `pread()` targets the CPU-visible Metal slot buffer returned by
`MTLBuffer.contents` plus the slot offset. The staged copy path only runs when
`DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0` or when DS4 cannot get slot pointers; that
path is `pread -> CPU scratch -> ds4_gpu_tensor_write() -> memcpy into
MTLBuffer.contents`.

DS4 currently allocates slot banks as shared `id<MTLBuffer>` tensors via
`newBufferWithLength`. The llama.cpp SSD branch uses CPU-visible slot writes too,
and its Metal backend additionally wraps page-aligned host memory with
`newBufferWithBytesNoCopy` and uses Metal residency sets for large model buffers.
To isolate whether DS4's high-slot collapse is page/residency related, this
branch now has two opt-in diagnostics:

```text
DS4_FLASH_MOE_SLOT_BANK_RESIDENCY=1
DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1
```

`DS4_FLASH_MOE_SLOT_BANK_RESIDENCY=1` requests a Metal residency set for the
slot-bank owner buffers after allocation. `DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1`
touches one byte per page at startup so first-touch page faults are moved out of
decode. A slot8 smoke test requested residency and touched `2.27 GiB` of slot
banks successfully; the real A/B still needs to run at the cliff size
(`--ssd-cache 64GB` / slot225, or a smaller binary-search point).

Follow-up high-slot run:

```text
small model, ds4-agent, --ssd-cache 64GB -> slots=225, total about 76.0GB
DS4_FLASH_MOE_SLOT_BANK_RESIDENCY=1
DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_STABLE_REPLAY=1
DS4_FLASH_MOE_ASYNC_HANDOUT=0
DS4_FLASH_MOE_ICB_REPLAY=0
--no-int8

decode status after reading ../ds4/ds4.c:
ctx 46.2k/60.8k, generation about 0.2 t/s
```

Residency plus page-touch did not move the cliff. That weakens the simple
"first-touch page fault" theory. `--no-int8` was intentional here to disable
ANE/int8 effects for the experiment; it routes Flash-MoE through NAX-half/GPU
fallback and is slower for large chunks, but we can leave that as a separate
compute-path issue. The other live axis was that the agent prompt had already
grown to about `46.2k` tokens after reading a large file, so high-slot tests
should compare both a tiny prompt and the long-context agent state.

Added one more falsifier:

```text
DS4_FLASH_MOE_RESET_SLOT_CACHE_AFTER_PREFILL=1
```

Alias:

```text
DS4_FLASH_MOE_CLEAR_SLOT_CACHE_AFTER_PREFILL=1
```

This clears slot ownership, replay metadata, decode slot IDs, slot ages, and ICB
slot generations after full or resume prefill, while keeping the same allocated
slot-bank buffers. It answers whether inference/decode throughput is hurt by
inheriting a prefill-populated resident cache. If this does not improve the
high-slot decode tokens/sec, the issue is more likely tied to the long-context
decode path, the huge tool-result prompt shape, or the `--no-int8` compute path
rather than prefill's slot-cache contents.

Short-context falsifier:

```text
same small model, ds4-agent, --ssd-cache 64GB -> slots=225
same residency/page-touch/no-int8/reset-after-prefill flags
prompt: "hi"
ctx 1.3k/60.8k, generation 37 tokens, 11.6 t/s
```

This shows the large slot-bank allocation itself is not sufficient to trigger
the `0.2 t/s` inference-throughput collapse. Also, about 40k prompt/context
tokens is not inherently fatal because the same class of prompt works with
smaller slot allocations. The measured failure is still decode/generation
speed; the current trigger is more likely the combination of high slot-bank
memory footprint and non-tiny/agent tool-result decode state. Next tests should
compare slot225 at a controlled long prompt against slot64/96 with the same
prompt and compute flags, and add per-layer decode stage/profile counters around
attention, indexer, routed MoE, and output head.

In the ds4-agent tool-result reproduction, `-p ./ds4.c` is a short literal
prompt first, then the agent reads the file as a tool result and resumes
prefill:

```text
slot cache reset after KV payload load: layers=43 slots=225 cleared=9675
session sync path=decode-suffix checkpoint=1274 prompt=1281 suffix=7
session sync path=resume-prefill checkpoint=1356 prompt=46225 suffix=44869
prefill 44869/44869 (100.0%) batch=191.8 t/s avg=285.6 t/s
slot cache reset after resume prefill: layers=43 slots=225 cleared=9675
decode still crawling after reset
```

That trace confirms the reset hook is active both after KV payload load and
after the 44.9k-token resume-prefill. Since decode still crawls after the reset,
prefill-populated slot metadata is ruled out. The next diagnostic should profile
decode stages directly, preferably one token at slot225 and the same prompt at
slot64/96.

Controlled prompt-file reproduction, outside the agent/tool loop:

```text
prompt: first 140000 bytes of ds4.c, about 46.7k token-dump lines
ctx: 60768
flags: no decode prefetch, stable replay on, async/ICB off, Metal replay on,
       --no-int8, reset-after-prefill on

slot225 / --ssd-cache 64GB:
prefill 302.97 t/s, generation 0.30 t/s

slot64:
prefill 312.02 t/s, generation 5.16 t/s
```

Decode-stage-only profile for one generated token on the same prompt:

```text
slot225:
routed_moe             2039.396 ms total, avg 47.428 ms/layer, max 134.807 ms
kv_path                 138.701 ms total
compressor_indexer       86.014 ms total
profiled generation       0.41 t/s

slot64:
routed_moe              302.350 ms total, avg 7.031 ms/layer, max 19.309 ms
compressor_indexer       20.407 ms total
kv_path                  16.073 ms total
profiled generation       2.33 t/s
```

This localizes the high-slot cliff to the routed-MoE banked decode kernel path,
not attention, indexer, KV, or decode prefetch. The routed-MoE cost grows by
about 6.7x between slot64 and slot225 on the same prompt, while prefill remains
healthy in both runs.

New falsifier to test the prefill-write/page-placement theory:

```text
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1
```

Alias:

```text
DS4_FLASH_MOE_RECREATE_SLOT_BANK_AFTER_PREFILL=1
```

This synchronizes after prefill/KV payload load, frees resident slot-bank Metal
buffers, recreates them with the same slot count/layout, then clears metadata.
If slot225 decode recovers, prefill writes/page placement are poisoning the
resident bank. If it does not recover, the slot225 routed-MoE kernel is slow
because of the large bank's live address/stride/working-set shape itself.

First slot225 reallocation test on the same controlled prompt:

```text
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1
slot225 / --ssd-cache 64GB
prefill 282.78 t/s, generation 2.60 t/s
```

This is a major recovery from `0.30 t/s` without reallocation, while still below
slot64's `5.16 t/s`. That points to prefill-time slot-bank writes/page placement
as a real part of the cliff. The next split is to disable prefill writes into the
resident slot bank and prefill lookahead/prefetch, then compare against the
reallocation recovery.

Related prefill knobs to isolate if reallocation helps:

```text
DS4_FLASH_MOE_PREFETCH=0
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=0
DS4_FLASH_MOE_XLAYER_PREFETCH=0
```

All three disabled, without slot-bank reallocation:

```text
slot225 / --ssd-cache 64GB
prefill I/O: bank-prefetch=0 xlayer=off
realloc-after-prefill=off
prefill 308.94 t/s, generation 1.69 t/s
```

This partially recovers the cliff without teardown. It is still below the
reallocation result (`2.60 t/s`), so prefill-side write/prefetch behavior is
implicated but may not be the only page-placement effect.

Important correction after adding the clearer prefill banner:

```text
prefill I/O: ... bank-prefetch=3 slot-cache-topk=0 xlayer=auto<=6k-tok topk=112
```

The old `topk=112` reading was the x-layer prefetch descriptor, not resident
slot-cache top-k. For this controlled long prompt, resident `slot-cache-topk`
is `0` unless explicitly configured. Therefore the `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=0`
row should be treated as a no-op/noisy run, not evidence that resident top-k
installs were active by default.

`DS4_FLASH_MOE_PREFETCH=0` only:

```text
slot225 / --ssd-cache 64GB
prefill I/O: bank-prefetch=0 slot-cache-topk=0 xlayer=auto<=6k-tok topk=112
realloc-after-prefill=off
prefill 306.46 t/s, generation 1.44 t/s
```

Bank-prefetch-off partially recovers from `0.30 t/s`, but less than reallocation.
The `1.44` vs `1.69` difference between bank-prefetch-off-only and "all off" is
not yet clean because resident slot-cache top-k appears to have been `0` already,
and x-layer auto is off for this >6k-token prompt.

`DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0`, leaving prefill-side behavior at default:

```text
slot225 / --ssd-cache 64GB
decode I/O: miss-direct-slot-pread=off
realloc-after-prefill=off
prefill 309.19 t/s, generation 1.50 t/s
```

This partially recovers from `0.30 t/s`, so direct pread into resident Metal slot
buffers participates in the bad interaction. It is not the whole issue because
full reallocation still performs better, and prefill-side knobs also matter.

Combined reallocation plus direct-slot-pread-off:

```text
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1
DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0
prefill I/O: bank-prefetch=3 slot-cache-topk=0 xlayer=auto<=6k-tok topk=112
decode I/O: miss-direct-slot-pread=off realloc-after-prefill=on
prefill 294.80 t/s, generation 2.47 t/s
```

This does not beat reallocation alone (`2.60 t/s`), so staged decode misses do
not improve the fresh-bank case. The strongest mitigation remains recreating the
resident slot bank after prefill.

Current recommended direction for the high-slot cliff:

- Treat reset-after-prefill as a failed metadata-only fix. It clears slot
  ownership/replay state, but decode remains slow at slot225.
- Treat reallocation-after-prefill as the strongest signal. It recovers slot225
  from `0.30 t/s` to `2.60 t/s`, so prefill-time writes/page placement of the
  resident Metal slot bank are part of the issue.
- For high-slot SSD-cache runs, prototype a production version where prefill
  uses transient scratch/staging only, then the decode resident slot bank is
  allocated or refreshed after prefill.
- If we want to preserve prefill-warmed experts, copy a small warm set into the
  fresh decode bank after prefill with a Metal/GPU copy path, not direct CPU
  `pread` into the final resident slots.
- Keep direct resident-slot `pread` as an A/B flag until we understand why the
  nominal no-copy path is slower than the staged path at high slot counts.

Code-path note:

- `metal_graph_flash_moe_install()` uses direct slot pread by default:
  `DS4_FLASH_MOE_DIRECT_SLOT_PREAD` defaults on, mixed layout gets the resident
  slot pointer and calls `flash_moe_pread_split(..., dst, expert_stride, ...)`.
- If resident prefill slot-cache top-k is enabled, prefill slot prefetch usually
  stages the expert first, then calls
  `metal_graph_flash_moe_prefetch_slot_from_buf()`, which reserves a resident
  slot and writes from `slot_prefetch_src` into that slot.
- The prefill I/O banner now prints `slot-cache-topk=N` separately from the
  x-layer prefetch `topk=N` descriptor to avoid confusing those two knobs.

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

## Pro 1.6T Slot-Bank Caveat

The smaller `dsv4-iq2xxs-expert-major` sweeps do not directly predict the best
slot-bank size for the 1.6T Pro Flash sidecar. For Pro, each expert record is
much larger and a bank around 50 slots/layer may still be fastest if the higher
expert hit rate outweighs residency/page-locality costs. Treat large-bank
warnings as calibration prompts, not as advice to force tiny banks; compare the
actual model/storage/device combination before drawing conclusions.

Follow-up Pro run on M5 Max, `~/Models/DSv4Pro-flash`, slot50, ctx32768,
PyGame prompt, n500:

```text
/tmp/ds4_flash_moe_async_pass1_20260605_131533
grouped_first hash f1878e3437c...
decode I/O: max-loads=0, miss-direct-slot-pread=on, shared-down=on, slots=50
prefill 0.52 t/s, generation 0.23 t/s
```

This run shows a severe slot50 Pro decode-throughput collapse under the current defaults. The
sweep helper failed after the first row because it was edited while Bash was
still executing it; the final helper passes `bash -n`. Rerun Pro comparisons
from a fresh helper process, preferably starting with shorter `N_TOKENS` and
explicit `--ssd-cache` budgets or lower slot counts to map the cliff before
spending another full 500-token row.

## SSD Cache Budget Flag

Added a shared `--ssd-cache BYTES|auto` frontend option for `ds4`, `ds4-agent`,
and `ds4-server`. It feeds the common `ds4_engine_options` path and resolves the
Flash-MoE slot count before Metal slot-bank allocation.

- Explicit values such as `25GB`, `25gb`, `25GiB`, `25000M`, or raw bytes are
  treated as the target GPU slot-bank cache budget.
- `auto` reads currently available memory, subtracts the dense mapped weight
  size and the context-buffer estimate for the selected `--ctx`, then assigns
  85% of the remainder to the slot bank.
- The resolver uses the current sidecar layout mode when converting budget to
  slots, so mixed expert-major and layer-slab overhead are reflected in the
  selected slot count.
- `--ssd-cache` is rejected unless Flash-MoE slot-bank mode is active, either
  explicitly or through package auto-detection.

Smoke checks while another Pro sweep owned a 50-slot bank:

```text
./ds4 -m ~/Models/DSv4Pro-flash --ssd-cache 25GB --ctx 32768 --inspect
resolved slots=23, gpu-bank=24.28 GiB, budget=25.00 GiB

./ds4 -m ~/Models/DSv4Pro-flash --ssd-cache 1GB --ctx 32768 --inspect
rejected: min-slots=6 needs 6.33 GiB

./ds4 -m ~/Models/DSv4Pro-flash --ssd-cache auto --ctx 32768 --inspect
rejected under active memory pressure:
available=26.56 GiB, dense=27.40 GiB, context=3.54 GiB
```

The `auto` result is pressure-sensitive by design; it should be measured after
the active benchmark/model process exits if the goal is startup sizing for a
clean machine.

## Open Questions

- Does full-resident one-allocation-per-semantic-expert remove the slot225
  routed-MoE cliff?
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

## Per-Semantic-Expert Buffer Diagnostic

User correction: do not test "one allocation per slot/stride" as the solution
candidate. The diagnostic now tests one allocation per semantic expert:

```text
layer L, expert E -> one MTLBuffer containing that expert's full record
```

Implemented knob:

```bash
DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1
# alias:
DS4_FLASH_MOE_FULL_RESIDENT_EXPERT_BUFFERS=1
```

Current behavior under this knob:

- overrides the effective decode resident slot count to `DS4_N_EXPERT`;
- preloads every `(layer, expert_id)` record into its own Metal buffer at graph
  creation;
- records `slot_id == expert_id`, so decode does not evict or install experts;
- disables decode prefetch and prefill slot-cache installs for this diagnostic;
- bypasses the grouped banked ABI and computes routed experts route-wise by
  binding the selected expert-owned buffer as a one-entry bank;
- disables ICB replay inside the route-wise helper by passing
  `layer_index=UINT32_MAX`, because local slot zero is not a stable semantic key;
- calls `didModifyRange` after CPU writes to Metal buffers, including direct
  `pread()` into the expert-owned preload buffers.

This is deliberately memory-heavy and intended first for the small IQ2XXS
sidecar. Full resident per-expert Pro likely does not fit a useful cache budget.

First test command shape:

```bash
DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1 \
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
DS4_FLASH_MOE_STABLE_REPLAY=1 \
DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
DS4_FLASH_MOE_ICB_REPLAY=0 \
DS4_METAL_DECODE_STAGE_PROFILE=1 \
./ds4 \
  -m "$HOME/Models/flash/dsv4-iq2xxs-expert-major" \
  --ctx 4096 \
  --temp 0 \
  -p "Who are you" \
  -n 8
```

If this recovers slot225/large-bank decode throughput, the high-slot cliff is
very likely tied to mixed allocation/subrange ownership rather than the core
expert math. The next step would be to reintroduce direct `pread()` into the
expert-owned buffer and compare against scratch -> upload/blit, without going
back to CPU writes in the middle of a giant layer bank.

## didModifyRange Diagnostic

Metal `didModifyRange` now has an explicit runtime knob:

```bash
DS4_FLASH_MOE_DID_MODIFY_RANGE=1  # default
DS4_FLASH_MOE_DID_MODIFY_RANGE=0  # reproduce old behavior
```

Implemented calls:

- `ds4_gpu_tensor_write()` marks the written view range;
- `ds4_gpu_tensor_fill_f32()` marks the filled range;
- direct slot-bank `pread()` marks the installed expert range;
- direct async/prefetch slot `pread()` marks the installed expert range;
- full-resident per-expert preload marks the whole expert-owned buffer;
- direct ANE output writes mark the output tensor range.

Important nuance: `didModifyRange` announces CPU-written bytes to Metal; it is
not a CPU/GPU ordering primitive. Tests should still avoid writing a range while
an in-flight command buffer may read that range.

Immediate high-slot matrix:

```text
A. slot225 direct pread, DS4_FLASH_MOE_DID_MODIFY_RANGE=0
B. slot225 direct pread, DS4_FLASH_MOE_DID_MODIFY_RANGE=1
C. B + strict handoff/drain if needed
D. realloc-after-prefill, didModifyRange off
E. realloc-after-prefill, didModifyRange on
F. DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1, didModifyRange on
```

Primary metric remains `DS4_METAL_DECODE_STAGE_PROFILE=1` routed-MoE time, not
only generation tokens/sec.

## 2026-06-05 Per-Expert Buffer Correctness Pass

Important testing correction: do not pack environment assignments into a single
zsh scalar such as `BASE_ENV='A=1 B=1'` and then run `env $BASE_ENV ...`.
Several early failures were caused by zsh not splitting that scalar into
separate assignments. Use explicit `env A=1 B=1 ... ./ds4 ...` commands, or a
proper shell array, so the logs show the expected kept environment count.

Implementation update after the first per-expert smoke failure:

- `DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1` remains decode-resident only for the
  slot-cache view of the world.
- Prefill no longer tries to satisfy its streaming routed-MoE reads from the
  resident slot cache when per-expert buffers are active.
- Prefill still uses its normal grouped/streaming path; decode reads the
  preloaded semantic expert buffers.
- Guarded breadcrumbs were added under `DS4_FLASH_MOE_PER_EXPERT_DEBUG=1` or
  `DS4_DEBUG_RESUME=1` for decode failures.

Validated command shape:

```bash
env \
  DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
  DS4_FLASH_MOE_STABLE_REPLAY=1 \
  DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
  DS4_FLASH_MOE_ICB_REPLAY=0 \
  DS4_FLASH_MOE_METAL_DECODE_REPLAY=1 \
  DS4_FLASH_MOE_DID_MODIFY_RANGE=1 \
  DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1 \
  ./ds4 \
    -m "$HOME/Models/flash/dsv4-iq2xxs-expert-major" \
    --moe-slot-bank 6 --ctx 60768 \
    -p "make a game of Space invaders in PyGame" \
    --temp 0 -n 32
```

Results directory:

```text
/tmp/ds4_flash_moe_expert_diag_20260605_174400
```

Hash comparison against the slot-bank path:

```text
PyGame prompt, ctx60768, n8:
slot225 grouped/stable bank     hash d6d70d53947a...  generation 2.71 t/s
per-expert semantic buffers     hash d6d70d53947a...  generation 1.29 t/s

PyGame prompt, ctx60768, n32:
slot225 grouped/stable bank     hash e589d9c6337a...  generation 5.79 t/s
per-expert semantic buffers     hash e589d9c6337a...  generation 2.15 t/s

ds4-agent smoke, ctx4096, n1:
per-expert semantic buffers     status 0, output "Hello"

Huge prompt, ctx60768, n5, `--no-int8`, no teardown/realloc:
per-expert semantic buffers     hash 6ac133b4cb9d...  prefill 301.73 t/s, generation 1.44 t/s
```

Conclusion:

- The per-semantic-expert allocation path is now correctness-clean for n8 and
  n32 on the long-context PyGame prompt.
- `ds4-agent --non-interactive` also runs through the per-expert path for a
  one-token smoke test.
- On the huge 46.7k-token prompt, full-resident semantic GPU buffers avoid the
  old slot225 `0.30 t/s` collapse, but the current diagnostic path only reaches
  `1.44 t/s`; it is still below slot-bank reallocation-after-prefill (`2.60 t/s`)
  because decode is route-wise instead of grouped/fused.
- The current per-expert path is not yet the speed path. It bypasses the grouped
  banked routed-MoE ABI and computes the six routed experts route-wise by
  binding one expert-owned buffer at a time.
- The next speed item is a grouped/fused execution path over separate semantic
  expert buffers: argument buffer/pointer table, compact descriptor list, or an
  equivalent Metal ABI that preserves one allocation per expert while avoiding
  six route-wise submissions.

## 2026-06-05 Exact Metal Dirty-Range Audit

Rule applied: only call `didModifyRange` after the CPU has actually written into
a Metal-backed buffer, and only for the exact written range.

Implemented/audited paths:

- direct slot `pread()` into Flash-MoE mixed/family Metal slot pointers marks the
  installed expert range after the read succeeds;
- scratch `pread()` followed by upload no longer double-marks in
  `metal_graph_flash_moe_install()`, because `ds4_gpu_tensor_write()` already
  marks the copied range;
- decode prefetch and async handout direct-slot reads mark after thread join;
- `ds4_gpu_tensor_write()` and `ds4_gpu_tensor_fill_f32()` mark their exact
  tensor view range;
- raw Objective-C `MTLBuffer.contents` CPU writes now mark exact ranges for
  transient RoPE/row/mask/index buffers, resident model copies, ANE skip masks,
  async shared-expert/O-proj outputs, and ANE prefill staging/output buffers.

High-slot retest used a trimmed `ds4.c` prompt file because the full source now
exceeds the 60.8k context as a raw CLI prompt:

```text
prompt bytes: 120000
prompt tokens: 39894
model: ~/Models/flash/dsv4-iq2xxs-expert-major
ctx: 60768
cache: --ssd-cache 64GB -> slots=225, gpu-bank=63.78 GiB
flags: no decode prefetch, stable replay on, reset-after-prefill on,
       slot-bank residency/touch-pages on, --no-int8
logs: /tmp/ds4_didmodify_retest_20260605_190746
```

Results:

```text
direct slot pread on:
prefill 289.64 t/s, generation 0.22 t/s

direct slot pread off:
prefill 290.58 t/s, generation 0.23 t/s
```

Conclusion:

- Exact dirty-range marking is necessary correctness hygiene, but it does not
  fix the high-slot decode cliff.
- Disabling direct slot pread no longer materially changes this 39.9k-token
  slot225 repro, so the current cliff is not explained solely by missing
  `didModifyRange` on direct pread into Metal pages.
- The remaining likely culprit is the large resident slot-bank GPU read path
  itself: allocation/page locality, sparse subrange access over 63.8 GiB, or the
  grouped routed-MoE ABI over that giant mixed bank.

## 2026-06-05 Per-Slot Shared Metal Buffers

Added an opt-in slot cache layout:

```text
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1
alias: DS4_FLASH_MOE_SEPARATE_SLOT_BUFFERS=1
```

This keeps the normal resident slot-cache policy and eviction semantics, but
allocates each `(layer, slot)` as its own `ds4_gpu_tensor_alloc(expert_stride)`.
`ds4_gpu_tensor_alloc()` creates a separate `MTLResourceStorageModeShared`
`MTLBuffer`, so this removes CPU writes into sparse subranges of the one huge
mixed layer bank.

Important distinction:

- `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1`: one shared Metal buffer per resident slot;
  slot ownership/eviction still work as before.
- `DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1` or
  `DS4_FLASH_MOE_FULL_RESIDENT_EXPERT_BUFFERS=1`: one shared Metal buffer per
  semantic expert id; diagnostic full-resident mode.

Current execution path:

- decode uses route-wise one-slot execution for per-slot buffers;
- grouped/fused execution over separate buffers is not implemented yet;
- async handout is disabled for this layout for now;
- direct slot `pread()` can write into the per-slot buffer and marks the exact
  expert range with `didModifyRange`.

Small correctness smoke:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: "Who are you"
ctx: 4096
slot_bank: 8
n: 8
flags: no decode prefetch
logs: /tmp/ds4_per_slot_buffers_20260605_201520

mixed bank hash:    9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18
per-slot hash:      9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18
mixed generation:   10.47 t/s
per-slot generation: 9.63 t/s
```

High-slot cliff diagnostic with a 100 KB `ds4.c` prompt file:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt file bytes: 100037
ctx: 60768
cache: --ssd-cache 64GB -> slots=225, gpu-bank=63.78 GiB
flags: --no-int8, no decode prefetch, stable replay on,
       async handout off, ICB replay off, Metal decode replay on
logs: /tmp/ds4_per_slot_fileprompt_20260605_202052

per-slot buffers:
  prefill 301.48 t/s, generation 8.97 t/s

mixed giant bank:
  prefill 301.62 t/s, generation 0.21 t/s
```

Interpretation:

- This is the strongest evidence so far that the slot225 cliff is caused by the
  huge mixed resident bank/subrange access pattern, not by SSD read bandwidth or
  prefill speed.
- Separate shared Metal resources per resident slot recover usable decode speed
  even before grouped/fused execution is implemented.
- The long-prompt outputs are both plausible summaries but diverge after the
  first few generated tokens, likely because per-slot route-wise execution and
  grouped mixed-bank execution are not bit-identical at long greedy decode.
- Next speed item: implement grouped/fused routed-MoE execution over separate
  slot/expert buffers, probably via an argument buffer or compact descriptor
  list, so we keep the recovered allocation behavior without paying route-wise
  dispatch overhead.

## 2026-06-05 Lazy Per-Slot Allocation Diagnostic

The sibling `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4` SSD streaming
cache does not preallocate its whole byte budget. It lazily allocates one
combined shared Metal buffer per cached `(layer, expert)`, fills it via
`pread()`, calls `didModifyRange`, publishes GPU addresses in per-layer address
tables, and reuses buffers after eviction. It also attempts `mlock()` once at
buffer allocation.

Added an opt-in Flash-MoE diagnostic that preserves per-slot cache semantics but
does not allocate all `layers * slots` buffers at startup:

```text
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1
DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1
alias: DS4_FLASH_MOE_LAZY_SLOT_BUFFERS=1
```

Behavior:

- startup computes planned cache capacity but allocates zero expert slot buffers;
- each per-slot shared `MTLBuffer` is allocated on first slot install/direct
  `pread()`;
- decode/prefill miss uploads still call exact `didModifyRange`;
- `DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1` is applied only to newly allocated
  lazy slot buffers, not the whole planned bank at startup;
- upfront slot-bank residency set is intentionally skipped in lazy mode, closer
  to the older SSD streaming cache behavior.

Smoke test:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: "Who are you"
ctx: 4096
slot_bank: 8
n: 8
logs: /tmp/ds4_per_slot_lazy_20260605_225412

eager per-slot hash: 9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18
lazy per-slot hash:  9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18

eager startup: gpu-bank=2.3GB
lazy startup:  gpu-bank=0.0GB allocated, planned=2.3GB
eager generation: 5.99 t/s
lazy generation:  10.00 t/s
```

High-slot prompt retest:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: /tmp/ds4_per_slot_fileprompt_20260605_202052/prompt.txt
ctx: 60768
cache: --ssd-cache 64GB -> slots=225, planned gpu-bank=63.8GB
flags: --no-int8, no decode prefetch, stable replay on,
       async handout off, ICB replay off, Metal decode replay on

lazy per-slot:
  logs /tmp/ds4_per_slot_lazy_high_20260605_225617
  startup gpu-bank=0.0GB allocated, planned=63.8GB
  prefill 307.67 t/s, generation 8.82 t/s

current eager per-slot:
  logs /tmp/ds4_per_slot_eager_high_20260605_225833
  startup gpu-bank=63.8GB
  prefill 308.72 t/s, generation 9.16 t/s
```

Interpretation:

- Lazy allocation is correctness-clean for the small deterministic smoke.
- Lazy allocation removes the huge startup allocation and gives a direct way to
  test whether preallocation/VM pressure is hurting interactive `ds4-agent`.
- On the 100KB high-slot CLI prompt, lazy allocation did not beat eager per-slot;
  both are in the same band and both remain far faster than the old mixed-bank
  collapse.
- The remaining major per-slot speed gap versus normal fast mixed-bank decode is
  still route-wise execution; grouped/fused separate-buffer execution remains
  the main architecture item.

## 2026-06-05 Separate Buffers Must Preserve Grouped Decode

Important correction: the first `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1`
implementation changed two variables at once:

- storage layout changed from one giant mixed bank to separate shared Metal
  buffers;
- decode execution also changed from grouped top-k execution to a per-route
  fallback loop.

That was a measurement bug. `PER_SLOT_BUFFERS=1` should be a residency/layout
choice, not a request to execute six routed experts as six independent MoE
calls.

Root cause:

- the existing grouped banked decode ABI expects one bank base pointer plus
  `slot * stride`;
- separate per-slot buffers do not fit that base+stride ABI;
- the branch therefore bound each expert buffer as a fake one-slot bank and
  called `metal_graph_flash_moe_compute_route_to_down()` once per route;
- this was not communicated clearly enough when the diagnostic was added.

Fix implemented:

- ported the sibling `ds4` direct `slots6` grouped kernel shape for the main
  Flash quant combo: IQ2_XXS gate/up and Q2_K down, top-k 6;
- added `ds4_gpu_routed_moe_one_slots6_tensor()`, which accepts six separate
  gate/up/down tensor views in route order;
- `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1` and `DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1`
  now try grouped slots6 decode first;
- the old route-wise path remains only as fallback/debug, forceable with
  `DS4_FLASH_MOE_FORCE_PER_ROUTE=1` or by disabling grouped with
  `DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1`;
- startup/decode now logs:

```text
ds4: Flash-MoE separate slot buffers using grouped slots6 decode path (direct 6-buffer IQ2_XXS/Q2_K)
```

Validation:

```text
logs: /tmp/ds4_slots6_fix_20260605_235148
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: "Who are you"
slot_bank: 8
ctx: 4096
n: 8

mixed bank hash:             efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04
per-slot grouped hash:       efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04
forced per-route hash:       efe53724915ea81458eed5b97818882eaf7fd094fcc8183b989d0da3c5721e04

mixed generation:            6.96 t/s
per-slot grouped generation: 9.82 t/s
forced per-route generation: 6.25 t/s
```

User-shaped 50-token prompt:

```text
logs: /tmp/ds4_slots6_fix_space_20260605_235428
prompt: "craete game of spsce invaders in PyGame, keep files and compile in /tmp/tmp1 compile, test, iterate"
cache: --ssd-cache 32GB -> slots=112
ctx: 32768
n: 50

mixed bank hash:             f944e748face7f4222bc9ab3229a489d44c49460a06b857327828c9d87e2a743
per-slot grouped hash:       f944e748face7f4222bc9ab3229a489d44c49460a06b857327828c9d87e2a743

mixed generation:            10.40 t/s
lazy per-slot grouped:       9.37 t/s
```

500-token Space Invaders prompt:

```text
logs: /tmp/ds4_slots6_fix_space500_20260605_235731
prompt: "make a game of Space invaders in PyGame"
cache: --ssd-cache 32GB -> slots=112
ctx: 32768
n: 500

mixed bank hash:             b424c5003c694a3bf52acf6cdab1bb31383dc3c64494b9d3f309920e4038ae78
per-slot grouped hash:       b424c5003c694a3bf52acf6cdab1bb31383dc3c64494b9d3f309920e4038ae78

mixed generation:            18.33 t/s
lazy per-slot grouped:       16.58 t/s
```

Interpretation:

- grouped execution itself was not the problem;
- the problem was that the first separate-buffer diagnostic bypassed grouped
  execution;
- direct slots6 grouped decode removes the accidental per-route execution
  penalty while preserving separate Metal resources;
- the remaining delta versus mixed bank on non-cliff settings is likely slot
  install/resource binding overhead and lazy allocation behavior, not six serial
  expert computes;
- next production improvement is still address-table/descriptor grouped decode
  or broader slots6 support for other quant combos, plus optional buffer reuse
  and `mlock()` diagnostics from the sibling streaming cache.

## 2026-06-06 YOLO Follow-Up: Slots6 Gated Off, Agent Small-Suffix Prefill Avoided

User reported two issues after the initial separate-buffer grouped work:

- `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1` was still slower than the mixed bank on
  normal 32GB/112-slot runs;
- `ds4-agent` could appear frozen after a short tool result, showing
  `prefill 0/90`.

Findings:

- The agent pause is explained by `ds4_session_sync()` taking `resume-prefill`
  for suffixes >= the default `DS4_METAL_RESUME_PREFILL_MIN=32`. A 90-token
  tool result therefore entered chunked prefill, and progress stayed at `0/90`
  until the whole small chunk completed.
- `ds4-agent` now defaults `DS4_METAL_RESUME_PREFILL_MIN=256` unless the user
  sets it explicitly. This keeps small tool-result continuations on
  `decode-suffix`, which is more responsive in the interactive tool loop.
- Added cached per-slot family views for separate slot buffers so each allocated
  slot gets persistent gate/up/down tensor views instead of rebuilding wrapper
  views for future direct-buffer grouped experiments.
- The custom direct slots6 grouped kernel failed a longer deterministic
  correctness check. It is now opt-in only with:

```bash
DS4_FLASH_MOE_ENABLE_SLOTS6_GROUPED=1
```

Default `PER_SLOT_BUFFERS=1` now falls back to the known-good route-wise path.
The previous force/disable flags still work:

```bash
DS4_FLASH_MOE_FORCE_PER_ROUTE=1
DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1
```

Correctness split:

```text
prompt: "craete game of spsce invaders in PyGame, keep files and compile in /tmp/tmp1a compile, test, iterate"
model: ~/Models/flash/dsv4-iq2xxs-expert-major
ctx: 32768
n: 100
temp: 0
decode prefetch: off

32GB/112-slot mixed:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  generation 13.48 t/s before rerun, 15.03 t/s after rebuild/rerun

32GB/112-slot per-slot with direct slots6 grouped enabled by old default:
  hash c9d0d5855ebc17629e8e5a71bd5f82789d582c79d35f11a30d9e2820c81adfc1
  generation 10.81 t/s
  cmp vs mixed = 1

32GB/112-slot per-slot forced route-wise:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  generation 13.36 t/s

32GB/112-slot per-slot corrected default:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  generation 13.14 t/s
  cmp vs mixed = 0
  logs: /tmp/ds4_yolo_32gb_default_20260606_002254
```

User-requested 64GB/225-slot smoke:

```text
command shape:
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1
DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1
DS4_FLASH_MOE_XLAYER_PREFETCH=0
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_PREPROTECT_TOPK=1
DS4_FLASH_MOE_STABLE_REPLAY=0
DS4_FLASH_MOE_BAKED_SLOT_DECODE=0
DS4_FLASH_MOE_ASYNC_HANDOUT=0
DS4_FLASH_MOE_ICB_REPLAY=0
./ds4 -m ~/Models/flash/dsv4-iq2xxs-expert-major --ssd-cache 64GB --ctx 32768 --temp 0 -n 100 -p ...

corrected default route-wise per-slot:
  slots=225
  prefill 0.72 t/s
  generation 13.13 t/s
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  logs: /tmp/ds4_yolo_64gb_default_20260606_002506

earlier same-shape mixed 64GB comparator:
  generation 9.76 t/s
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  logs: /tmp/ds4_yolo_mixed_20260606_001618
```

Agent trace check:

```text
logs: /tmp/ds4_yolo_agent_trace_20260606_002620
ds4: session sync path=decode-suffix checkpoint=1274 prompt=1278 suffix=4 resume-min=256
```

Follow-up UI note:

- The agent status label still says `prefill` while it is synchronizing prompt
  tokens into KV, even when the underlying sync path is `decode-suffix`.
- Before the follow-up patch, the GPU decode-suffix loop did not call the
  progress callback, so a slow suffix could display `prefill 0/N 0.0%` until
  it finished.
- Added the same per-token `prefill_chunk` callback used by the CPU/decode
  helper path after each GPU decode-suffix token is appended. The label is still
  `prefill`, but the bar/counter/tps should now advance for suffixes below the
  `DS4_METAL_RESUME_PREFILL_MIN` threshold.
- Runtime verification was attempted at
  `/tmp/ds4_agent_decode_suffix_progress_20260606_114600`, but the single
  process guard refused to start because an existing `ds4-agent` was running.

Related `ds4` CLI startup-prefill note:

- `ds4` had an analogous missing-progress case on startup full-prefill.
- Root cause: the CLI only registered `ds4_session_set_progress()`, and its
  callback ignored `prefill_display`. One-chunk/full-prefill reports useful
  per-layer progress through `ds4_session_set_display_progress()`, not through
  durable `prefill_chunk` boundaries.
- Updated the CLI callback to accept both `prefill_chunk` and
  `prefill_display`, and registered display progress in sampled generation,
  logprob dump, and interactive chat turns.
- Also registered display progress in the Flash-MoE
  `ds4_engine_generate_argmax()` session path.
- Expected user-visible effect: startup prompt processing should show advancing
  `processing N input tokens: x/N ...` progress during full-prefill rather than
  staying silent until the first chunk/done boundary.

Interpretation:

- The safe default is correctness-first route-wise separate buffers.
- At high slots, per-slot lazy route-wise still beats the giant mixed bank
  cliff in the 64GB/225-slot smoke (`13.13` vs `9.76 t/s`).
- At normal 32GB/112-slot settings, mixed remains faster (`15.03` vs
  `13.14 t/s`), so per-slot is not a universal default yet.
- The direct slots6 kernel needs a separate correctness fix before it can be a
  production grouped path. The first failing comparator is the 100-token prompt
  above; forced route-wise proves the per-slot cache/install path itself is
  correct.

Lazy allocation banner clarification:

- With `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1` and
  `DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1`, `--ssd-cache 32GB` sizes the planned
  slot capacity, but no slot `MTLBuffer`s are allocated at startup.
- Therefore startup can correctly print `gpu-bank=0.0GB allocated,
  planned=31.7GB`, while `Total allocated` is only dense mapped weights plus
  context buffers (`8.2GB + 2.8GB = 11.0GB` for the local small model run).
- Updated the banner to print both total allocated and total planned:

```text
Total <<<< 11.0GB allocated, 42.7GB planned >>>>
```

- `DS4_FLASH_MOE_PREPROTECT_TOPK=1` does not preallocate or prefill slots. It
  protects the token's currently routed experts from eviction while decode
  reserves/installs slots. Prefill-side resident installation is controlled by
  `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK` / `--moe-prefetch-topk`.

## 2026-06-06 Slots6 Correctness Fix and Grouped Exact-Down Path

User asked to investigate the direct separate-buffer `slots6` path instead of
leaving it gated off. The previous failure was:

```text
mixed / route-wise per-slot hash: 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
old direct slots6 hash:          c9d0d5855ebc17629e8e5a71bd5f82789d582c79d35f11a30d9e2820c81adfc1
```

Root cause:

- The first direct slots6 implementation changed two things at once:
  grouped separate-buffer gate/up and a direct six-expert down-sum kernel.
- The profile baseline has `DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=0`, so the
  correct path computes six routed-down rows and then runs the existing sum
  kernel. The old slots6 path always accumulated all six down experts inside
  one q2_K kernel, changing floating-point order enough to alter generated
  tokens.
- Per-slot cache/install was not the culprit: forced route-wise per-slot had
  already matched mixed exactly.

Fixes:

- `ds4_gpu_routed_moe_one_slots6_tensor()` now takes `routed_down`/expert
  scratch and, by default, preserves the baseline down shape.
- The old direct sum6 experiment is only used with:

```bash
DS4_FLASH_MOE_SLOTS6_DIRECT_DOWN_SUM=1
```

- Added `kernel_mul_mv_slots6_q2_K_f32`, a separate-buffer grouped down
  projection that binds six down buffers and writes six ordinary routed-down
  rows in one dispatch. The existing `moe_sum_experts` kernel then performs the
  exact baseline final sum.
- The grouped exact-down kernel is enabled by default for slots6; disable it
  for A/B with:

```bash
DS4_FLASH_MOE_SLOTS6_DISABLE_GROUPED_DOWN=1
```

- `DS4_FLASH_MOE_SLOTS6_FRESH_VIEWS=1` remains as an isolation diagnostic for
  cached per-slot family views.
- Corrected slots6 is now default-on for independent per-slot/per-expert
  buffers when the shape is supported (`topk=6`, IQ2_XXS gate/up, Q2_K down).
  Escape hatches:

```bash
DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1
DS4_FLASH_MOE_FORCE_PER_ROUTE=1
DS4_FLASH_MOE_ENABLE_SLOTS6_GROUPED=0
```

Validation:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: "craete game of spsce invaders in PyGame, keep files and compile in /tmp/tmp1a compile, test, iterate"
ctx: 32768
temp: 0
prefetch: DS4_FLASH_MOE_XLAYER_PREFETCH=0, DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
slot mode: DS4_FLASH_MOE_PER_SLOT_BUFFERS=1, DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1
```

Before grouped exact-down, after disabling direct sum6:

```text
n20, 32GB/112 slots:
  route fallback:  hash 5c2009e5a68b1718eccc3c8cc760e1134f9885ea46d84971c120e8f250903ba1, generation 8.42 t/s
  slots6 gate/up:  hash 5c2009e5a68b1718eccc3c8cc760e1134f9885ea46d84971c120e8f250903ba1, generation 8.82 t/s
  cmp=0

n100, 32GB/112 slots:
  route fallback:  hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 8.54 t/s
  slots6 gate/up:  hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 8.24 t/s
  cmp=0
```

After adding grouped exact-down:

```text
n50, 32GB/112 slots:
  route fallback:       hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 9.80 t/s
  default slots6 group: hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 11.01 t/s
  cmp=0
  logs: /tmp/ds4_slots6_grouped_down_n50_20260606_142236

n100, 32GB/112 slots:
  route fallback:       hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 9.06 t/s
  default slots6 group: hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 9.02 t/s
  cmp=0
  logs: /tmp/ds4_slots6_grouped_down_n100_20260606_142458

n50, 64GB/225 slots:
  route fallback:       hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 10.60 t/s
  default slots6 group: hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 10.90 t/s
  cmp=0
  logs: /tmp/ds4_slots6_grouped_down_64gb_n50_20260606_142732

n20 default-on sanity:
  default grouped:  hash 5c2009e5a68b1718eccc3c8cc760e1134f9885ea46d84971c120e8f250903ba1, generation 8.89 t/s
  disabled route:   hash 5c2009e5a68b1718eccc3c8cc760e1134f9885ea46d84971c120e8f250903ba1, generation 8.73 t/s
  cmp=0
  logs: /tmp/ds4_slots6_default_n20_20260606_143031
```

Interpretation:

- Correctness regression is fixed for the old n100 repro.
- The grouped exact-down path gives a clear n50 win and a small 64GB/225-slot
  win, but n100 at 32GB was effectively tied. Treat speed as promising but noisy
  until longer 500-token runs are repeated.
- The first visible `processing 0/N` pause in these tests is expected with an
  empty lazy slot cache: the first prompt token performs many slot installs
  before the progress counter can advance. This was previously mostly silent;
  the CLI display-progress fix now exposes it.

Next performance items:

- Repeat 500-token sweeps with default grouped slots6 at 32GB and 64GB.
- Add stage profiling around the new slots6 grouped-down dispatch versus six
  individual down dispatches to verify dispatch overhead is the actual win/loss.
- If the old direct sum6 path is ever reconsidered, it needs a numerical policy:
  it is faster-shaped but not baseline-exact because it changes accumulation
  order.

## 2026-06-06 Agent Tool/Prefill Status Clarity

User reported another freeze-shaped footer:

```text
Reading /tmp/timeout_staging/timeout.c 1:500...

ds4-agent>
ctx 6.4k/132.8k | prefill 0/4896 0.0% 0.0 t/s
```

The confusing part is that two different waits were both visually collapsed
into `prefill`:

- executing the tool itself;
- synchronizing the tool result back into the live DS4 session/KV before the
  assistant can continue.

Implemented UI/status changes in `ds4_agent.c`:

- added an explicit `AGENT_WORKER_TOOL` state;
- footer now shows active tool calls as
  `tool <name> <current>/<total> running <elapsed>`;
- bash tools additionally report live captured output size and line count while
  the process is being refreshed, for example
  `tool bash 1/1 running 12s 4.1KiB 37 lines`;
- ordinary user prompt sync remains `prefill`;
- tool continuation sync now displays as `sync tool result`;
- system prompt and compaction rebuild syncs display as `sync system`,
  `compact prompt`, or `sync compacted`;
- the initial accepted-turn handoff now displays `starting` instead of
  `prefill cached`;
- non-interactive stderr progress uses the same labels.

Important limitation:

- `read`, `more`, `write`, `edit`, `list`, and `search` tools are still
  synchronous. The footer now makes it clear which tool is executing, but there
  is no per-byte or per-line progress inside `read` yet. If `read` itself is
  materially slow, the next diagnostic is to make `agent_read_file_bytes()` or
  line rendering publish chunked status updates while it scans the file.

Verification:

```text
make ds4-agent
make
git diff --check -- ds4_agent.c docs/flash-moe-stable-slot-progress.md
./ds4-agent --help
status: passed
```

## 2026-06-06 Prefill Setup Verbose Trace

User observed another stall-looking interval:

```text
ctx 1.9k/132.8k | prefill 0/675 0.0% 0.0 t/s
```

This status is set just before `ds4_session_sync()` enters the backend. If it
sits at `0/N`, the worker has already tokenized the next prompt and is inside
session sync, but the backend has not emitted its first durable
`prefill_chunk`/`prefill_display` progress callback yet.

The pre-callback work can include:

- switching Metal model views for prefill;
- allocating/ensuring prefill scratch;
- uploading prompt token ids;
- warming Metal prefill kernels;
- precompiling ANE/i8i8 prefill contexts;
- growing compressed/raw KV context buffers.

Added display-progress phase events and verbose stderr timing around those
setup calls:

```text
prefill_model_views
prefill_setup
prefill_upload
prefill_warmup
prefill_ane_compile
prefill_ctx_grow
```

`ds4-agent` now maps those into footer labels such as:

```text
prefill warmup 0/675 ...
prefill ANE compile 0/675 ...
sync tool result ctx grow 0/4896 ...
```

Verbose trace flags:

```bash
DS4_SESSION_SYNC_TRACE=1
DS4_SESSION_SYNC_TRACE_VERBOSE=1
```

Equivalent extra aliases:

```bash
DS4_PREFILL_VERBOSE_TRACE=1
DS4_PREFILL_PHASE_TRACE=1
```

Example repro command:

```bash
DS4_SESSION_SYNC_TRACE=1 \
DS4_SESSION_SYNC_TRACE_VERBOSE=1 \
DS4_FLASH_MOE_XLAYER_PREFETCH=0 \
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
./ds4-agent \
  -m "$HOME/Models/flash/dsv4-iq2xxs-expert-major" \
  --ssd-cache 32GB \
  --ctx 132768 \
  --temp 0
```

Expected verbose lines:

```text
ds4: prefill trace scope=session-sync phase=model_views begin ...
ds4: prefill trace scope=chunked phase=warmup begin ...
ds4: prefill trace backend=ANE-precompile begin mode=...
ds4: prefill trace backend=ANE-precompile create begin mode=...
ds4: prefill trace backend=ANE-precompile create end mode=... elapsed=...
ds4: prefill trace scope=chunked phase=ANE compile end ...
```

Diagnostic interpretation:

- If `--no-int8` removes the stall, suspect the int8/ANE prefill setup branch.
- The current code confirms `--no-int8` forces `DS4_FLASH_MOE_ANE_PREFILL=0`
  and disables MPP/i8i8 prefill envs.
- If the next run stalls after `backend=ANE-precompile create begin`, the delay
  is inside ANE context creation/compilation.
- If it stalls after `phase=warmup begin`, the delay is Metal prefill kernel
  warmup before the ANE precompile branch.
- If it stalls after `phase=ctx grow begin`, the delay is KV/context buffer
  growth rather than prefill compute.

Follow-up after seeing `prefill ANE compile` in the footer:

- The agent footer now appends elapsed time for the current prefill phase, so a
  long ANE compile should display as `prefill ANE compile ... 37s` rather than
  only `0.0 t/s`.
- The ANE tiled-fused prefill path is cached in-process.
- Default profile sets `DS4_FLASH_MOE_ANE_BATCHES=256`, so ordinary precompile
  should compile one primary tiled context for batch 256.
- Cache capacity is `DS4_ANE_CTX_CACHE_MAX=12` per cache table.
- There is one primary cache plus secondary B/C/D cache tables for multi-worker
  ANE execution. Precompile warms the primary table; actual dual/quad ANE
  execution can later compile B/C/D contexts too.
- Verbose trace now logs cache hits and per-context compile begin/end:

```text
ds4: prefill trace backend=ANE-cache compile begin cache=primary mode=... slot=0 B=256 ...
ds4: prefill trace backend=ANE-cache compile end cache=primary mode=... slot=0 B=256 elapsed=...
ds4: prefill trace backend=ANE-cache hit cache=primary mode=... slot=0 B=256 ...
ds4: prefill trace backend=ANE-cache compile begin cache=B mode=... slot=0 B=256 ...
```

Current suspicion:

- If the footer stalls at `prefill ANE compile`, the expensive call is not the
  main prefill computation. It is compiling/loading the ANE/CoreML context for
  the configured routed-MoE prefill mode and batch shape.
- If `--no-int8` makes the same prompt immediate, that is consistent with this
  path because `--no-int8` disables `DS4_FLASH_MOE_ANE_PREFILL` and the
  MPP/i8i8 prefill envs.

Verification:

```text
make ds4-agent
make
git diff --check -- ds4.c ds4_agent.c ds4_metal.m docs/flash-moe-stable-slot-progress.md
status: passed
```

## 2026-06-06 ANE Prefill Compile Trace Gap

User observed:

```text
ds4: prefill trace scope=chunked phase=scratch end start=1274 n_tokens=675 prompt=1949 elapsed=0.014 ms
```

but no visible ANE compile duration. Root cause in the trace surface: the
chunked prefill path still ignored the `ds4_gpu_ane_prefill_precompile_from_env`
return value and only emitted a coarse `phase=ANE compile` bracket when the
ANE-precompile condition was true. If the condition was false, or if the backend
returned early, there was no explicit verbose line saying "skipped" or "returned
0", making the blank startup/pre-fill region ambiguous.

Trace additions:

- both layer-major and chunked prefill now emit:

```text
ds4: prefill trace scope=chunked phase=ANE-precompile check ... flash_ane=... resident_ane=... token_backend_ane=... result=...
ds4: prefill trace scope=chunked phase=ANE-precompile skip ...
ds4: prefill trace scope=chunked phase=ANE-precompile end ... result=<0|1> elapsed=...
```

- `ds4_gpu_ane_prefill_precompile_from_env()` now logs verbose early exits:

```text
ds4: prefill trace backend=ANE-precompile init-failed ...
ds4: prefill trace backend=ANE-precompile unsupported-shape ...
```

- ANE tiled context cache now logs silent failure cases:

```text
ds4: prefill trace backend=ANE-cache compile-budget-exhausted ...
ds4: prefill trace backend=ANE-cache cache-full ...
```

Expected diagnosis after this change:

- If startup still pauses after `phase=ANE-precompile check result=0`, the delay
  is not inside the explicit precompile call and the next suspect is warmup,
  ctx-grow, tokenization, or agent prompt/session sync.
- If it pauses after `backend=ANE-cache compile begin`, the delay is inside the
  CoreML/ANE context creation call.
- If it prints `phase=ANE-precompile end elapsed=<large>`, the coarse prefill
  phase is now correctly accounting for the ANE compile wall time.

Follow-up after terminal redraw ambiguity:

- Added `DS4_PREFILL_TRACE_LOG=/path/file.log` to mirror prefill/session trace
  lines to an append-only plain file. This avoids relying on the interactive
  agent footer, which can visually overwrite stderr lines.
- Added direct private ANE trace in `ds4_ane_mlp_int8w.m` around:

```text
modelWithMILText:weights:optionsPlist:
inMemoryModelWithDescriptor:
compileWithQoS:options:error:
loadWithQoS:options:error:
```

- This means a real ANE compile should now emit:

```text
ds4: prefill trace backend=ANE-inmemory ... stage=compileWithQoS begin ...
ds4: prefill trace backend=ANE-inmemory ... stage=compileWithQoS end ... elapsed=...
```

- If the log only contains `session-sync phase=model_views end` and does not
  contain `resume_prefill_call begin`, the process did not reach the prefill
  handoff in the rebuilt binary. If it contains `resume_prefill_call begin` but
  no `chunked phase=scratch begin`, the stall is inside the call boundary before
  the chunked prefill function body starts, which would be unexpected enough to
  inspect symbols/binary freshness first.

Smoke result with file logging:

```text
command: ctx=4096, --moe-slot-bank 6, prompt "hi", n=1
trace: /tmp/ds4_prefill_trace_smoke.log

tmp_cleanup:    54004.863 ms
compileWithQoS:   557.351 ms
loadWithQoS:       16.693 ms
ANE create total: 54585.561 ms
```

Conclusion:

- The observed ~45-55 second startup pause was not CoreML/ANE compile itself.
- It was `ane_cleanup_stale_tmp_dirs_once()` scanning/removing stale
  `NSTemporaryDirectory()` ANE model directories on the hot prefill path.
- The private `_ANEInMemoryModel compileWithQoS` call was sub-second in this
  smoke.

Interim fix:

- `DS4_ANE_TMP_CLEANUP` is now opt-in. Default behavior is no startup cleanup.
- To manually run the old cleanup behavior for diagnostics, set:

```bash
DS4_ANE_TMP_CLEANUP=1
```

Retest after making cleanup opt-in:

```text
command: same ctx=4096, --moe-slot-bank 6, prompt "hi", n=1
trace: /tmp/ds4_prefill_trace_smoke2.log

tmp_cleanup:       0.001 ms
compileWithQoS:  302.186 ms
loadWithQoS:      17.119 ms
ANE compile end: 324.203 ms
prefill call:   1424.945 ms
```

This confirms the visible 45-55 second startup pause was stale ANE temp cleanup,
not CoreML compile.

Follow-up fix after inspecting `$TMPDIR`:

- `$TMPDIR` currently contains roughly 1.18M top-level `tmp.*` entries. Any
  foreground cleanup that lists the parent temp directory is unsafe, even if it
  only deletes a narrow ANE pattern.
- ANE temp creation is now isolated under:

```text
$TMPDIR/ds4-ane/
```

- During private `_ANEInMemoryModel` descriptor/compile/load, DS4 temporarily
  sets process `TMPDIR` to that DS4-only root under a mutex, then restores the
  caller's `TMPDIR` immediately. This keeps both DS4's explicit
  `NSTemporaryDirectory()/hexId/model.mil` writes and CoreML's private
  temporary files inside the same DS4-owned root without leaking the altered
  temp directory to `ds4-agent` tool subprocesses.
- Cleanup is now non-blocking and DS4-root-only: the first ANE create schedules
  a detached background cleanup of stale children in `$TMPDIR/ds4-ane/`.
  It never scans `$TMPDIR/tmp.*` and does not block prefill startup.
- Runtime knobs:

```bash
DS4_ANE_TMP_CLEANUP=0              # disable background DS4-root cleanup
DS4_ANE_TMP_CLEANUP_AGE_SEC=7200   # stale age, default 2 hours
DS4_ANE_TMP_ROOT=/path/to/root     # override the DS4 ANE temp root
DS4_ANE_TMP_CLEANUP_DEBUG=1        # log cleanup summary/errors
```

Retest after dedicated-root/background cleanup:

```text
command: same ctx=4096, --moe-slot-bank 6, prompt "hi", n=1
trace: /tmp/ds4_prefill_trace_ds4ane.log
root: /var/folders/.../T/ds4-ane

tmp_cleanup:       0.008 ms
compileWithQoS:  331.837 ms
loadWithQoS:      16.163 ms
ANE create total: 349.531 ms
prefill call:   1487.016 ms
post-run DS4 root children: 0
```

Agent smoke:

```text
command: ds4-agent --non-interactive, ctx=4096, --moe-slot-bank 6, prompt "hi", n=1
trace: /tmp/ds4_agent_prefill_trace_ds4ane.log

system sync tokens: 1274
tmp_cleanup:        0.026 ms
compileWithQoS:   404.141 ms
loadWithQoS:       16.498 ms
ANE create total: 422.347 ms
system sync call: 7166.525 ms
post-run DS4 root children: 0
```

This validates the agent path too: the remaining multi-second startup work in
that smoke is real system-prompt prefill, not hidden ANE temp cleanup.

### 2026-06-06: Agent Footer ANE Prefill Color

Request: make the agent prefill progress bar red when ANE is the active prefill
chunk backend, instead of using the normal magenta fill.

Implementation:

- `ds4.c` now emits `prefill_display_ane` display-progress events when the
  current prefill work is ANE-backed:
  - full/layer-major prefill uses the existing ANE precompile/backend decision;
  - chunked prefill computes the decision per chunk from
    `DS4_FLASH_MOE_ANE_PREFILL`, `DS4_RESIDENT_MOE_ANE_HYBRID`, or the
    per-token backend table.
- `ds4-agent` stores this as `status.prefill_ane` and renders the filled
  progress-bar segment as bright red (`38;5;196`) while the event is active.
  Normal prefill remains the existing magenta (`38;5;201`).
- `ds4` CLI text progress accepts the new event so display-progress updates are
  not dropped, but it keeps its existing textual progress styling.

Validation:

```text
make: pass
git diff --check -- ds4.c ds4_cli.c ds4_agent.c: pass
DS4_FLASH_MOE_ANE_PREFILL=1 ./ds4 ... -p hi -n 1: pass
```

### 2026-06-07: Q4 60GB Mode Sweep and ANE Prefill Trace

Request: benchmark Q4 Flash model decode modes and check whether ANE is engaged
for prefill.

Model and prompt:

```bash
./ds4 \
  -m "$HOME/Models/DSv4-Flash-Q4KExperts-chat-v2-flash" \
  --ssd-cache 60GB \
  --ctx 132768 \
  --temp 0 \
  --nothink \
  -p "craete game of spsce invaders in PyGame, keep files and compile in /tmp/tmp_q4A compile, test, iterate"
```

Common decode env for the sweep:

```bash
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0
DS4_FLASH_MOE_XLAYER_PREFETCH=0
```

Short correctness smoke (`-n 16`):

```text
output dir: /tmp/ds4_q4_modes_20260607_182958
all five modes hash: d1cdb208c0c81258586c36afed56a3e144fef3d68adbffe1429b2fb8664b63e5

grouped:            prefill 8.80 t/s, generation 1.83 t/s
stable_slot label:  prefill 9.14 t/s, generation 3.24 t/s
slotwise:           prefill 9.82 t/s, generation 3.54 t/s
per_expert_buffers: prefill 3.14 t/s, generation 0.27 t/s
baked_slot:         prefill 8.59 t/s, generation 3.47 t/s
```

Important caveat from that smoke:

- `--ssd-cache 60GB` resolves the Q4 slot bank to `slots=105`, `gpu-bank=59.52 GiB`.
- `DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1` currently switches to full-resident
  semantic expert buffers: `slots=256`, `gpu-bank=145.12 GiB`. On the 128 GiB
  M5 Max this is not a valid apples-to-apples 60GB cache comparison and should
  be treated as over-commit/thrash, not a route-wise architecture result.

Longer baseline sweep (`-n 100`, before separating stable/non-baked from baked):

```text
output dir: /tmp/ds4_q4_modes_n100_20260607_183422
hash: 17e8aac33a15e9d069c56796e868c3397793153432ee8032f3a38bc1793abbf3

grouped:      prefill 9.74 t/s, generation 7.71 t/s
stable label: prefill 8.73 t/s, generation 7.16 t/s
slotwise:     prefill 8.76 t/s, generation 7.64 t/s
baked_slot:   prefill 8.80 t/s, generation 7.54 t/s
grouped_last: prefill 8.45 t/s, generation 7.67 t/s
```

Flag-separation fix:

- Found that `flash_moe_baked_slot_decode_enabled()` auto-enabled baked-slot
  whenever `DS4_FLASH_MOE_STABLE_REPLAY=1`; an explicit
  `DS4_FLASH_MOE_BAKED_SLOT_DECODE=0` did not override it.
- Patched this so an explicit baked env wins, while the default behavior is
  unchanged: if `DS4_FLASH_MOE_BAKED_SLOT_DECODE` is absent,
  `DS4_FLASH_MOE_STABLE_REPLAY=1` still implies baked-slot decode.

Fixed stable-vs-baked speed rerun (`-n 100`):

```text
output dir: /tmp/ds4_q4_stable_baked_n100_20260607_183939
hash: 17e8aac33a15e9d069c56796e868c3397793153432ee8032f3a38bc1793abbf3

stable_non_baked:
  env: DS4_FLASH_MOE_STABLE_REPLAY=1 DS4_FLASH_MOE_BAKED_SLOT_DECODE=0
  prefill 10.35 t/s, generation 7.78 t/s

baked_slot:
  env: DS4_FLASH_MOE_STABLE_REPLAY=1 DS4_FLASH_MOE_BAKED_SLOT_DECODE=1
  prefill 8.21 t/s, generation 7.68 t/s
```

Interpretation:

- For this Q4 model at 60GB cache / 105 slots, grouped, slotwise,
  stable-non-baked, and baked-slot are clustered around ~7.6-7.8 t/s for the
  100-token prompt.
- The earlier `-n 16` low grouped number was short-run/cold-start noise.
- Full per-expert semantic buffers need either a smaller model, a bigger memory
  machine, or a cache-budgeted per-expert implementation before speed numbers
  are meaningful.

Original ANE prefill trace:

```text
trace dir: /tmp/ds4_q4_ane_trace_20260607_183314
normal env:
  prefill compute: routed experts = ANE i8i8 (W8A8) | dense proj = fp16-NAX
  ANE-precompile check: flash_ane=on resident_ane=off token_backend_ane=off result=1
  compileWithQoS: 274.622 ms
  loadWithQoS: 16.053 ms
  ANE create total: 292.170 ms
  prefill call: 3696.971 ms

negative control:
  trace dir: /tmp/ds4_q4_ane_off_check_20260607_184124
  env: DS4_FLASH_MOE_ANE_PREFILL=0 DS4_RESIDENT_MOE_ANE_HYBRID=0
  prefill compute: routed experts = GPU fp32 | dense proj = fp16-NAX
  ANE-precompile check: flash_ane=off resident_ane=off token_backend_ane=off result=0
```

Corrected conclusion after checking actual ANE graph execution:

- The original trace only proved ANE precompile/load. It did **not** prove
  routed-MoE prefill used ANE evaluation.
- Q4 Flash routed experts are `Q4_K/Q4_K/Q4_K`. The current Flash-MoE ANE
  prefill entry supports `gate/up=IQ2_XXS` and `down=Q2_K/IQ2_XXS`, so Q4
  cannot execute that ANE path today.
- This explains the external monitor observation: mactop showed GPU 100% and
  ANE 0% during Q4 prefill because the actual graph ran on GPU.

Fixes added:

- Startup compute banner now separates requested ANE from executable ANE and
  prints the unsupported expert types.
- `ANE-precompile` trace now includes `flash_ane_executable=on/off`.
- Q4 unsupported Flash ANE no longer precompiles an unused ANE graph, no longer
  uses the red ANE prefill progress color, and no longer attempts/falls back per
  expert.
- Metal ANE stats now include `attempts`, `ane_evaluate_calls`, and bounded
  early-reject reason counters/logs.
- The debug fallback line now prints only after an actual ANE submission attempt,
  not merely because ANE was requested or theoretically possible.

Validation:

```text
Q4 unsupported path:
  dir: /tmp/ds4_q4_ane_final_20260607_203200
  banner: routed experts = GPU/MPP fallback
  trace: flash_ane=on flash_ane_executable=off result=0
  message: actual Flash-MoE ANE eval calls=0
  counts: ANE tensor enter=0, eval_ok=0, fallback=0
  prefill: 9.12 t/s

IQ2 default tiny-prompt policy:
  dir: /tmp/ds4_iq2_default_stats_final_20260607_203353
  trace: flash_ane=on flash_ane_executable=on result=1
  counts: ANE tensor enter=0, eval_ok=0, fallback=0
  stats: attempts=0 calls=0 compile_attempts=1 eval_calls=0
    ane_evaluate_calls=0
  note: default hybrid/concurrent policy did not submit ANE for this tiny prompt.

IQ2 forced sync ANE control:
  dir: /tmp/ds4_iq2_ane_forced_sync_20260607_203028
  env overrides: DS4_FLASH_MOE_HYBRID_PREFILL=0,
    DS4_FLASH_MOE_CONCURRENT_PREFILL=0,
    DS4_FLASH_MOE_HYBRID_CONCURRENT_PREFILL=0,
    DS4_FLASH_MOE_OVERLAP_PREFILL=0,
    DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
  counts: ANE tensor enter=2277, eval_ok=2277, fallback=0
  stats: attempts=2277 calls=2277 ok=2277 ane_evaluate_calls=2277
  prefill: 2.39 t/s for the tiny prompt (slow because every 1-3 ref group pays
    a full sync ANE call; this is only a path-validity control).
```

### 2026-06-07 Q4 Flash-MoE ANE Prefill Fix

Correction to the previous conclusion:

- Q4 was unsupported because the Flash-MoE ANE prefill staging layer only had
  IQ2_XXS and Q2_K dequant/transpose pack kernels.
- The ANE MLP itself already consumes dense i8/f16 staged weights; the missing
  piece was Q4_K raw expert -> staged ANE weight buffers.

Implementation:

- Added Q4_K pack kernels:
  - `kernel_dsv4_mpp_dequant_q4_k_transpose_i8`
  - `kernel_dsv4_ane_dequant_q4_k_transpose_f16`
- Wired Q4 pipelines into `ds4_gpu_ensure_mpp_int8_prefill_pipelines()`.
- Added typed dequant dispatch helpers so ANE prefill can pack IQ2_XXS, Q2_K,
  or Q4_K without accidentally using the IQ2/Q2 path.
- Enabled the Flash ANE eligibility helper for `Q4_K/Q4_K/Q4_K`.
- Left the fused GUD dequant fast path IQ2+Q2-only; Q4 currently uses three
  separate pack dispatches.

Validation:

```text
Q4 forced-sync ANE smoke:
  dir: /tmp/ds4_q4_ane_forced_sync_20260607_212045
  model: ~/Models/DSv4-Flash-Q4KExperts-chat-v2-flash
  env: disabled hybrid/concurrent/overlap/pipeline prefill
  stats: attempts=651 calls=651 ok=651 ane_evaluate_calls=651
  rejects: all zero
  prefill: 5.78 t/s on a tiny 21-token prompt

Q4 default/profile long prompt:
  dir: /tmp/ds4_q4_ane_default_long_20260607_212306
  prompt: 2211 tokens
  banner: routed experts = ANE i8i8 (W8A8)
  trace: flash_ane=on flash_ane_executable=on result=1
  stats: attempts=404 calls=404 ok=404 eval_calls=2184 ane_evaluate_calls=2184
  rejects: all zero
  prefill: 348.99 t/s

Q4 GPU fallback control:
  dir: /tmp/ds4_q4_gpu_default_long_20260607_212329
  env: DS4_FLASH_MOE_ANE_PREFILL=0
  banner: routed experts = GPU fp32
  prefill: 227.93 t/s

Q4 screenshot-like cache/context:
  dir: /tmp/ds4_q4_ane_60gb_ctx132k_20260607_212511
  args: --ssd-cache 60GB --ctx 132768
  allocation: slots=105 gpu-bank=59.5GB Dense=8.2GB Context=7.1GB Total=74.8GB
  stats: attempts=400 calls=400 ok=400 eval_calls=2182 ane_evaluate_calls=2182
  rejects: all zero
  prefill: 335.93 t/s
  generation: 0.56 t/s (large-slot decode cliff remains separate from ANE prefill)

Q4 short generation A/B:
  dir: /tmp/ds4_q4_ane_ab_20260607_212122
  n=50, temp=0
  GPU hash: 12e7fd35d0787bab247cce6477aed03c8ff25338ab0f7b65d9e50c7ceb6a9451
  ANE hash: d137f5114acfa680a0f20813d23a69f362627d98054922a0f81a634bbc1d4d43
  cmp=1
  note: both outputs were coherent Space Invaders code preambles; divergence
    appeared around imports (`sys` vs `math`) and is consistent with W8A8 ANE
    numeric drift versus exact GPU Q4 prefill, not an immediate packing failure.

IQ2 forced-sync regression:
  dir: /tmp/ds4_iq2_ane_regress_20260607_212358
  stats: attempts=2312 calls=2312 ok=2312 ane_evaluate_calls=2312
  rejects: all zero
```

Remaining follow-up:

- Add a fused Q4 gate/up/down pack dispatch if Q4 ANE dequant overhead shows up
  in long prefill profiles.
- If deterministic parity is required, compare logits or staged dequant buffers
  against the exact Q4 GPU path; token hashes are expected to drift with W8A8
  prefill.

### 2026-06-07 Q4 ANE Corruption Fix And Scale Sweep

The first Q4 ANE implementation produced corrupted text (`}<?_...`-style
output) across every tested weight qscale, which ruled out "just choose a better
scale" as the primary failure. The actual bug was in the Q4_K scale/min helper
used by the new staging kernels: it did not match the existing release-path
`get_scale_min_k4_just2()` indexing for the upper scale groups. The helper now
uses the same `j + k` / `j + 4 + k` / high-bit reconstruction as the existing
Q4 decode kernel.

Pre-fix qscale sweep:

```text
dir: /tmp/ds4_q4_wq_sweep_20260607_231442
model: ~/Models/DSv4-Flash-Q4KExperts-chat-v2-flash
qscales: 512,384,256,192,128,96,64,48,32
result: every ANE run produced corrupted/non-code output
conclusion: corruption was Q4_K dequant indexing, not qscale saturation
```

Post-fix focused check:

```text
dir: /tmp/ds4_q4_wq_fixcheck_20260607_231842
qscales: 512,256,128
result: all produced coherent PyGame code preambles
```

Post-fix qscale sweep:

```text
dir: /tmp/ds4_q4_wq_postsweep_20260607_232056
qscales: 768,640,512,384,320,256,192,160,128,96,64
best single-prompt text similarity to GPU fallback: 320
note: text similarity is noisy after the first greedy divergence; use it only as
  a smoke signal, not a numeric-quality metric.
```

Corrected true-GPU / Q4 ANE control:

```text
dir: /tmp/ds4_q4_ane_truegpu_20260607_233953
GPU control env:
  DS4_FLASH_MOE_ANE_PREFILL=0
  DS4_RESIDENT_MOE_ANE_HYBRID=0
  DS4_RESIDENT_MOE_PREFILL_BY_TOKENS=16384:mulmm
GPU row: routed experts = GPU fp32, prefill 206.09 t/s
Q4 ANE default row: routed experts = ANE i8i8, compiled w_scale=0.003125
  (w_qscale=320), prefill 221.64 t/s, attempts=996 calls=996 ok=996
Explicit qscales 256/320/384/512: all coherent; no control-character garbage.
```

Longer default-Q4 ANE smoke:

```text
dir: /tmp/ds4_q4_ane_default500_20260607_234543
n=500, temp=0
compiled scale: w_scale=0.003125 (w_qscale=320)
stats: attempts=871 calls=871 ok=871 eval_calls=1310 ane_evaluate_calls=1310
rejects: all zero
output: coherent Space Invaders/PyGame class code, bad_ctrl=0
prefill: 218.40 t/s
generation: 8.58 t/s
```

Startup logging fix:

- Q4/Q4/Q4 Flash ANE prefill now reports the same Q4-specific default in the
  `prefill quant scales` line that the ANE compile path uses.
- Default Q4 routed-expert weight scale is `w_qscale=320`; explicit
  `DS4_FLASH_MOE_ANE_Q4_INT8_QSCALE` / `DS4_FLASH_MOE_ANE_Q4_W_QSCALE` wins,
  and the generic `DS4_FLASH_MOE_ANE_INT8_QSCALE` is still honored when set.
- The generic startup `w_qscale=512` line was stale/misleading for Q4 and was
  the source of false qscale suspicion after the actual decode bug was fixed.

## High-slot cliff: quant-type independence A/B (2026-06-11, MXFP4 branch)

`--ssd-cache auto` (M5 Max 128GB, ctx 4096, n=256, temp 0, same essay prompt,
sequential runs with cooldown):

| package | slots | bank | prefill | decode |
|---|---|---|---|---|
| MXFP4-native (12.75 MiB records) | 131 | 70.1 GB | 4.92 t/s | **0.11 t/s** |
| Q4K chat-v2 (13.5 MiB records)   | 138 | 78.2 GB | 5.22 t/s | **0.09 t/s** |

Reference: both packages decode ~10 t/s at slot-bank 48 (25-26 GB bank) with
the same prompt class. The collapse is ~100x and hits both quant types
equally, so the cliff is independent of expert quant format, record size, and
the dequant kernel path (MXFP4 uses the new mul_mv_id_mxfp4 kernel, Q4K the
long-standing q4_K one). This kills any theory tied to a specific dequant
kernel or block layout; consistent with the earlier localization to the
banked decode path / decode temporal prefetch.

Practical guidance until the cliff is fixed: cap slot banks well below the
cliff (48 slots measured healthy) rather than using --ssd-cache auto on
128GB machines.

## Cliff root cause in the cold-decode regime: page-cache squeeze (2026-06-11)

Short-prompt (25-token) cold-decode repro at slot131 (70 GiB bank, MXFP4
package, n=32): baseline 0.21 t/s; DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0 gives
0.22; REALLOC_SLOT_BANK_AFTER_PREFILL=1 gives 0.21. Neither known mitigation
moves it — this regime is NOT prefill-write poisoning.

iostat during decode is decisive:
- slot48 (26 GiB bank): decode 8.44 t/s with the disk essentially idle —
  decode-miss preads (~2.7 GB/token) are served by the macOS file cache at
  RAM speed. The file cache, not the slot bank, is the effective MoE cache.
- slot131 (70 GiB bank): decode 0.22 t/s with sustained 280-400 MB/s of true
  disk reads for the whole run — the wired bank evicted the file cache, so
  every miss is a real SSD read.

So in this regime the "cliff" is the wired slot bank squeezing out the OS
file cache that was invisibly serving misses. It is quant-independent
(reproduced on Q4K and MXFP4) and insensitive to install-path flags because
the IO volume itself is the cost.

Fix landed on the MXFP4 branch: --ssd-cache auto now budgets
DS4_SSD_CACHE_AUTO_PCT% (default 40, was 85) of remaining memory.
Validation, same command/prompt: auto now resolves 62 slots / 33 GiB and
decodes at 6.52 t/s vs 0.21 before (~30x). slot48 remains slightly faster
(8.44) — the bank/cache tradeoff curve peaks below 40% on a 128 GiB M5 Max
with a 145 GiB sidecar.

Note for the original slot225/long-prompt investigation: the realloc-recovery
evidence there (0.30 -> 2.60) may still indicate an additional kernel- or
placement-side component in the warmed-bank regime, but a large share of
"high-slot decode collapse" is explained by cache squeeze, which also
explains why teardown/recreate (which momentarily frees 60+ GiB) helps.

Auto-pct sweep (same short-prompt cold-decode repro, MXFP4 package, n=32):

| auto pct | slots | bank | decode t/s |
|---|---|---|---|
| 10% | 15 | 8 GiB | 10.53 |
| 20% | 30 | 16 GiB | 9.99 |
| 30% | 45 | 24 GiB | 9.17 |
| 40% | 62 | 33 GiB | 6.52 |
| 85% | 131 | 70 GiB | 0.21 |

Monotone: in the cold-decode regime every GiB wired into the bank costs more
(file-cache loss) than it gains (slot hits). Default set to 20% — near-peak
decode while keeping a slot-bank-32-class bank for prefill streaming.
DS4_FLASH_MOE_EXPERT_MMAP=1 at 20%: 9.76 t/s — no measurable benefit (reads
remain pread); left off.
