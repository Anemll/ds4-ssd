# Flash-MoE High-Slot Decode Cliff Handoff

Date: 2026-06-05  
Branch: `codex/stable-slot-replay-experiment`  
Latest pushed commit with diagnostics: `6532867 diagnose flash moe prefill slot bank cliff`

## Request For Review

We need an outside/higher-agent review of a counterintuitive Metal/unified-memory
behavior in DS4 Flash-MoE slot-bank decode.

The short version:

- Decode/inference throughput collapses with a large resident Flash-MoE slot
  bank.
- The collapse localizes to the routed-MoE decode stage, not attention,
  indexer, or decode prefetch.
- Reallocating the resident slot-bank buffers after prefill recovers most of the
  lost speed.
- Disabling direct `pread` into resident Metal slot buffers also improves speed,
  even though this is the no-copy path we expected to be best.

Please advise whether this is likely Metal page ownership/placement, CPU cache
mode, unified-memory residency, or an implementation mistake in the slot-bank
write path.

## Current Recommendation

Teardown/recreate is the strongest result so far, so the next implementation
should stop letting prefill write the final decode resident bank for high-slot
runs. The simplest production-shaped version is:

- prefill uses transient streaming/staging buffers only;
- after prefill/resume-prefill, allocate or refresh the decode resident slot
  bank;
- decode fills the resident bank in a decode-friendly way;
- for high-slot banks, avoid direct CPU `pread` into final Metal slot buffers
  until we understand the page-placement/caching behavior.

If we want to preserve useful prefill-loaded experts, copy a small warm set into
the fresh decode bank after prefill, preferably through a GPU/Metal copy or
another path known not to poison later GPU reads. Resetting metadata alone is
not enough: `DS4_FLASH_MOE_CLEAR_SLOT_CACHE_AFTER_PREFILL=1` clears ownership
and replay state but the high-slot decode cliff remains.

## Reproduction Setup

Machine:

```text
Apple M5 Max, 128 GiB RAM
```

Model:

```text
~/Models/flash/dsv4-iq2xxs-expert-major
```

Controlled prompt:

```bash
head -c 140000 ds4.c > /tmp/ds4_prompt_140000.txt
```

This tokenizes to roughly the same size as the agent tool-result reproduction:
about `46.7k` token-dump lines. It fits `--ctx 60768`.

Common flags:

```bash
DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=0 \
DS4_FLASH_MOE_STABLE_REPLAY=1 \
DS4_FLASH_MOE_ASYNC_HANDOUT=0 \
DS4_FLASH_MOE_ICB_REPLAY=0 \
DS4_FLASH_MOE_METAL_DECODE_REPLAY=1 \
DS4_FLASH_MOE_SLOT_BANK_RESIDENCY=1 \
DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1 \
DS4_FLASH_MOE_CLEAR_SLOT_CACHE_AFTER_PREFILL=1 \
./ds4 \
  -m "$HOME/Models/flash/dsv4-iq2xxs-expert-major" \
  --ssd-cache 64GB \
  --ctx 60768 \
  --temp 0 \
  --no-int8 \
  --nothink \
  --prompt-file /tmp/ds4_prompt_140000.txt \
  -n 5
```

Note: decode prefetch is off. The remaining suspects are prefill-side writes into
resident slot-bank buffers and direct slot `pread`.

## Key Results

Same controlled prompt, same context, same model:

```text
slot64 baseline:
prefill 312.02 t/s, generation 5.16 t/s

slot225 baseline/default prefill/direct slot pread:
prefill 302.97 t/s, generation 0.30 t/s

slot225 + DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1:
prefill 282.78 t/s, generation 2.60 t/s

slot225 + prefill-side prefetch knobs off:
DS4_FLASH_MOE_PREFETCH=0
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=0
DS4_FLASH_MOE_XLAYER_PREFETCH=0
prefill 308.94 t/s, generation 1.69 t/s

slot225 + DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0:
prefill 309.19 t/s, generation 1.50 t/s

slot225 + DS4_FLASH_MOE_PREFETCH=0 only:
prefill 306.46 t/s, generation 1.44 t/s

slot225 + DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1
        + DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0:
prefill 294.80 t/s, generation 2.47 t/s
```

Interpretation:

- Large slot allocation alone is not sufficient to explain the collapse.
- 46k context alone is not sufficient either, because slot64 is fine.
- Teardown/recreate after prefill recovers from `0.30` to `2.60 t/s`.
- Turning off direct `pread` into resident slots recovers to `1.50 t/s`.
- Turning off prefill bank prefetch recovers to roughly `1.44-1.69 t/s`.
- Direct-slot-pread-off plus reallocation does not beat reallocation alone
  (`2.47 t/s` vs `2.60 t/s`).
- These partial recoveries suggest multiple write/page-placement paths can poison the
  resident bank's later GPU-read performance.
- Important correction: the prefill I/O banner now distinguishes
  `slot-cache-topk=N` from x-layer `topk=N`. The controlled runs show
  `slot-cache-topk=0`; the previously observed `topk=112` was x-layer topk,
  not resident slot-cache topk.

## Decode Stage Profile

One generated token, same prompt:

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

This strongly localizes the high-slot cliff to the routed-MoE banked decode
kernel path over resident slot-bank buffers.

## Relevant Code Paths

Main file:

```text
ds4.c
```

Direct slot `pread`:

```text
flash_moe_direct_slot_pread_enabled()
metal_graph_flash_moe_install()
metal_graph_flash_moe_direct_slot_ptrs()
metal_graph_flash_moe_mixed_slot_ptr()
metal_graph_flash_moe_pread_slot_direct()
```

`DS4_FLASH_MOE_DIRECT_SLOT_PREAD` defaults on. In mixed slot-bank mode,
`metal_graph_flash_moe_install()` obtains a CPU-visible pointer to the resident
Metal buffer slot:

```text
mixed_bank + slot * expert_stride
```

Then it calls:

```text
flash_moe_pread_split(fd, record_offset, dst, expert_stride, io_split)
```

So the no-copy path is literally `pread` into the resident Metal slot buffer.

Optional prefill top-k resident slot-cache path:

```text
get_prefill_slot_cache_target()
slot_cache_expert[]
metal_graph_flash_moe_prefetch_slot_from_buf()
metal_graph_flash_moe_upload_prefill_slot()
metal_graph_flash_moe_write_slot_from_buf()
```

This path usually stages the expert for prefill first, then uploads/copies from
`slot_prefetch_src` into the resident slot. It can poison the bank without using
direct slot `pread`, but in the controlled runs above `slot-cache-topk=0`, so it
was not the default active culprit.

New diagnostic:

```text
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1
DS4_FLASH_MOE_RECREATE_SLOT_BANK_AFTER_PREFILL=1
```

Implemented behavior:

- synchronize after full/resume prefill or KV payload load;
- free resident slot-bank Metal buffers;
- recreate them with the same slot count/layout;
- reset slot ownership/replay metadata.

This recovers slot225 decode from `0.30` to `2.60 t/s`, which is the strongest
evidence that prefill-time writes/page placement poison later decode.

## Current Hypotheses

1. CPU `pread` into `MTLBuffer.contents` faults or places pages in a way that is
   bad for later GPU reads. The no-copy path avoids an explicit copy but may make
   the unified-memory page locality/ownership worse.

2. Prefill bank-prefetch/direct staging writes and direct slot preads touch a
   huge 64 GiB bank. Reset-after-prefill clears metadata but cannot undo
   physical page placement/cache ownership.

3. Reallocation recovers because it gives decode a fresh Metal buffer allocation
   after prefill, avoiding prefill's CPU writes and page-touch pattern.

4. The remaining gap between slot225 reallocation (`2.60 t/s`) and slot64
   (`5.16 t/s`) may be the inherent large-bank routed-MoE stride/working-set
   shape, or direct-slot read/write behavior during decode misses after the fresh
   bank is created.

## Questions For Higher Agent

1. On Apple unified memory, is it expected that CPU `pread` into
   `MTLBuffer.contents` can make subsequent GPU reads much slower than staging
   into CPU memory and using `ds4_gpu_tensor_write`/Metal upload?

2. Should resident slot-bank buffers be allocated with a different Metal resource
   option, for example write-combined CPU cache mode, private storage plus blit,
   heap allocation, or an explicit managed/flush-like path?

3. Should direct `pread` into resident Metal buffers be disabled by default for
   high-slot banks, especially before/decode after large prefill?

4. Should prefill never touch the decode resident slot bank? A possible design:
   use transient prefill scratch only, then allocate or refresh the decode slot
   bank after prefill. This matches the reallocation recovery but costs startup
   time and loses prefill-warmed experts.

5. Is there a better way to preserve useful prefill-loaded experts without
   poisoning the resident bank? For example:
   - prefill into a separate scratch/cache buffer;
   - after prefill, GPU-copy selected experts into the decode bank;
   - allocate decode bank only after prefill;
   - or keep a small decode-warm set instead of writing across the huge bank.

6. Does the routed-MoE kernel itself have a slot-count-dependent access pattern
   that becomes pathological at 225 slots even when only six routed experts are
   selected? The stage profile suggests routed-MoE is the visible slow stage.

## Suggested Next Experiments

Run these on the same controlled prompt:

1. Combine reallocation with direct-slot-pread-off. This has been tested once:

```bash
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1 \
DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0 \
...
```

Result:

```text
prefill 294.80 t/s, generation 2.47 t/s
```

This does not beat reallocation alone (`2.60 t/s`), so staged decode misses do
not improve the fresh-bank case.

2. Reallocate after prefill and disable prefill-side knobs:

```bash
DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1 \
DS4_FLASH_MOE_PREFETCH=0 \
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=0 \
DS4_FLASH_MOE_XLAYER_PREFETCH=0 \
...
```

This tests whether reallocation fully dominates prefill-side knobs.

3. Stage-profile the best slot225 mitigation and compare routed-MoE totals:

```bash
DS4_METAL_DECODE_STAGE_PROFILE=1
```

Target comparison:

```text
slot225 baseline routed_moe total: 2039 ms
slot64 baseline routed_moe total:   302 ms
```

4. Re-run the baseline after the new banner to confirm the active prefill knobs
shown in logs. In current controlled runs, `slot-cache-topk=0`; the x-layer
descriptor still prints `topk=112`.

5. Test an implementation that allocates resident decode slot banks only after
prefill. This is functionally similar to reallocation, but avoids the first
large allocation and any prefill-time resident-bank page touches.

6. Compare three install paths into the final decode bank after prefill:

```text
A. direct pread into resident MTLBuffer.contents
B. pread into CPU scratch, then ds4_gpu_tensor_write/memcpy into the slot
C. pread into CPU scratch, then Metal blit/GPU copy into the slot
```

The surprising result is that A is not best at high slot counts. The higher
agent should advise whether C is the correct Metal-shaped path for preserving
GPU-read locality.

## Candidate Direction

Updated direction after the user correction:

Do not treat automatic teardown/reallocation after prefill as the solution. It is
only a diagnostic showing that the resident bank's allocation/write history can
poison decode throughput.

Do not start with one allocation per slot/stride. That still lets CPU writes land
inside a cache-managed slot abstraction and does not test the desired ownership
model cleanly.

The current diagnostic implementation is:

```bash
DS4_FLASH_MOE_DID_MODIFY_RANGE=1
DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1
# alias:
DS4_FLASH_MOE_FULL_RESIDENT_EXPERT_BUFFERS=1
```

This allocates one Metal buffer per `(layer, semantic expert_id)`, preloads every
expert record into its own buffer, sets `slot_id == expert_id`, disables decode
prefetch/prefill slot-cache writes, and runs routed decode route-wise by binding
that expert-owned buffer as a one-entry bank. It also calls `didModifyRange`
after CPU writes into Metal buffers, including direct `pread()` preloads.

What I would do next:

- first A/B direct slot `pread()` with `DS4_FLASH_MOE_DID_MODIFY_RANGE=0/1`;
- test the new per-semantic-expert mode on the small IQ2XXS sidecar;
- compare stage profile `routed_moe` against slot64, slot225 baseline, and
  slot225 reallocate-after-prefill;
- if per-expert buffers recover speed, reintroduce direct `pread()` only into
  expert-owned buffers and compare against scratch -> Metal write/blit;
- keep Pro separate: full resident per-expert is probably too large for Pro, but
  the small-model result should tell whether semantic ownership is the right
  architecture.

Longer-term design likely separates prefill streaming scratch from decode
resident semantic expert buffers. Prefill can stream experts efficiently, but
decode residency should avoid CPU writes into arbitrary subranges of a giant
shared layer bank.

## Latest Local Test Update

After implementing the per-semantic-expert diagnostic, the current local result
is correctness-positive but not speed-positive yet.

Current behavior under:

```bash
DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1
DS4_FLASH_MOE_DID_MODIFY_RANGE=1
```

- allocates one Metal buffer per `(layer, semantic expert_id)`;
- preloads every expert record into its own buffer and calls `didModifyRange`;
- sets `slot_id == expert_id`;
- disables decode prefetch and prefill slot-cache reuse for this diagnostic;
- keeps prefill on the normal grouped streaming path;
- runs decode routed experts route-wise by binding one expert-owned buffer at a
  time.

The explicit-env correctness matrix on the small IQ2XXS sidecar:

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

Result logs:

```text
/tmp/ds4_flash_moe_expert_diag_20260605_174400
/tmp/ds4_huge_perexpert_gpu_current_20260605_181515
```

Interpretation:

- One allocation per semantic expert is now a working, hash-matching diagnostic.
- It has not proved a speed win because the current execution path deliberately
  uses route-wise one-entry bank calls, not a grouped/fused kernel over separate
  expert buffers.
- On the huge no-teardown prompt, per-expert GPU buffers improve over the old
  slot225 `0.30 t/s` cliff but only reach `1.44 t/s`, so separate allocation
  alone is not enough while execution remains route-wise.
- The next review question should be how to represent a top-k list of separate
  expert buffers to Metal without rebuilding the giant mixed layer bank:
  argument buffer, pointer/descriptor table, or another grouped ABI that keeps
  semantic expert ownership.
- If that grouped separate-buffer path recovers routed-MoE time, the high-slot
  cliff is very likely giant allocation/subrange/page ownership. If it remains
  slow even with grouped separate buffers, the routed-MoE kernel or total working
  set is the remaining suspect.

## Exact Dirty-Range Retest

An audit tightened `didModifyRange` usage to exact CPU-written Metal ranges:
direct slot `pread()` marks after successful read, scratch upload relies on
`ds4_gpu_tensor_write()`, decode prefetch/async handout mark after join, and raw
`MTLBuffer.contents` CPU writes in `ds4_metal.m` now mark their exact ranges.

Retest on M5 Max with a 39,894-token trimmed `ds4.c` prompt, `--ssd-cache 64GB`
(`slots=225`, `gpu-bank=63.78 GiB`), decode prefetch off, reset-after-prefill on,
residency/touch-pages on, and `--no-int8`:

```text
direct slot pread on   prefill 289.64 t/s, generation 0.22 t/s
direct slot pread off  prefill 290.58 t/s, generation 0.23 t/s
logs: /tmp/ds4_didmodify_retest_20260605_190746
```

This means exact `didModifyRange` cleanup is not sufficient to fix the cliff.
The bad interaction is now more likely in the large resident bank GPU read path
or allocation/page locality than in only the direct `pread()` dirty-range path.

## Per-Slot Shared Buffer Diagnostic

Added an opt-in layout:

```bash
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1
# alias:
DS4_FLASH_MOE_SEPARATE_SLOT_BUFFERS=1
```

This is different from the existing full semantic expert diagnostic:

```bash
DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1
DS4_FLASH_MOE_FULL_RESIDENT_EXPERT_BUFFERS=1
```

Per-slot mode keeps the normal slot cache and eviction policy, but allocates
each resident `(layer, slot)` as its own `ds4_gpu_tensor_alloc(expert_stride)`.
That creates a separate `MTLResourceStorageModeShared` `MTLBuffer` per resident
slot instead of one huge mixed layer buffer with `slot * expert_stride`
subranges.

Validation:

```text
small prompt, slot8, n8:
mixed hash    9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18
per-slot hash 9b7d4e56be73293b81fcc0625aa4b382742f8e81a692f44dcadc6a06fda36f18
logs: /tmp/ds4_per_slot_buffers_20260605_201520
```

High-slot diagnostic:

```text
model: ~/Models/flash/dsv4-iq2xxs-expert-major
prompt: 100 KB ds4.c prompt file
ctx: 60768
cache: --ssd-cache 64GB -> slots=225, gpu-bank=63.78 GiB
flags: --no-int8, no decode prefetch, stable replay on,
       async handout off, ICB replay off, Metal decode replay on
logs: /tmp/ds4_per_slot_fileprompt_20260605_202052

per-slot buffers  prefill 301.48 t/s, generation 8.97 t/s
mixed giant bank   prefill 301.62 t/s, generation 0.21 t/s
```

Interpretation for reviewer:

- This strongly implicates the giant mixed resident bank/subrange access pattern.
- Separate shared Metal buffers per resident slot recover high-slot decode speed
  even though the current per-slot execution path is still route-wise and not
  grouped/fused.
- The next architecture question is how to run grouped routed-MoE over separate
  expert/slot buffers without rebuilding a huge mixed bank. Candidate Metal ABIs:
  argument buffers, pointer/descriptor tables, or a compact per-token descriptor
  list consumed by a fused routed-MoE kernel.

## Lazy Per-Slot Allocation Follow-Up

The sibling `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4` SSD streaming
cache allocates expert buffers lazily on miss, not as one upfront byte-budget
reservation. It also reuses evicted buffers and attempts `mlock()` once when a
buffer is allocated.

Added a matching Flash-MoE diagnostic:

```bash
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1
DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1
# alias:
DS4_FLASH_MOE_LAZY_SLOT_BUFFERS=1
```

This preserves per-slot cache semantics but allocates each separate shared
`MTLBuffer` only when a slot is first installed. Startup logs distinguish
allocated bytes from planned capacity:

```text
gpu-bank=0.0GB allocated, planned=63.8GB
slot bank layout: lazy per-slot expert buffers
decode I/O: resident=per-slot-lazy
```

Results:

```text
slot8 smoke, n8:
eager/lazy hashes match
lazy startup avoids 2.3GB preallocation

slot225, --ssd-cache 64GB, 100KB prompt, n50, --no-int8:
lazy per-slot   prefill 307.67 t/s, generation 8.82 t/s
eager per-slot  prefill 308.72 t/s, generation 9.16 t/s
mixed bank      generation 0.21 t/s from earlier same prompt
```

Interpretation:

- Lazy allocation is useful to test whether agent slowdown is caused by upfront
  preallocation/VM pressure.
- In the controlled CLI high-slot prompt it does not beat eager per-slot, so
  preallocation alone is not the only speed factor.
- The fast sibling allocation also has features still missing here: buffer reuse
  and optional `mlock()`, plus grouped execution via GPU address tables. Those
  are separate follow-up diagnostics.

## Correction: Per-Slot Buffers Now Preserve Grouped Decode

The first per-slot implementation had a major hidden variable: it allocated
separate Metal buffers but decoded route-wise. That made `PER_SLOT_BUFFERS=1`
look like a storage-only diagnostic while it had actually stopped using grouped
top-k execution.

Fixed in the current branch:

- added direct `slots6` grouped decode kernels for IQ2_XXS gate/up and Q2_K
  down;
- added `ds4_gpu_routed_moe_one_slots6_tensor()`;
- `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1` and
  `DS4_FLASH_MOE_PER_EXPERT_BUFFERS=1` now try grouped six-buffer decode first;
- route-wise execution remains only as fallback/debug:

```bash
DS4_FLASH_MOE_FORCE_PER_ROUTE=1
DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1
```

Successful grouped-path log:

```text
ds4: Flash-MoE separate slot buffers using grouped slots6 decode path (direct 6-buffer IQ2_XXS/Q2_K)
```

Validation:

```text
/tmp/ds4_slots6_fix_20260605_235148
slot8, ctx4096, n8:
mixed/per-slot/forced-route hashes all match
mixed              6.96 t/s
per-slot grouped   9.82 t/s
forced per-route   6.25 t/s

/tmp/ds4_slots6_fix_space_20260605_235428
--ssd-cache 32GB -> slots=112, ctx32768, n50:
mixed hash             f944e748face7f4222bc9ab3229a489d44c49460a06b857327828c9d87e2a743
per-slot grouped hash  f944e748face7f4222bc9ab3229a489d44c49460a06b857327828c9d87e2a743
mixed                  10.40 t/s
lazy per-slot grouped  9.37 t/s
```

Reviewer takeaway:

- grouped decode was not the root problem;
- the branch accidentally bypassed grouped decode for separate buffers;
- separate buffers plus grouped slots6 is the correct diagnostic baseline now;
- for high-slot cliff work, compare mixed bank versus per-slot grouped, not
  mixed bank versus route-wise separate buffers.

## 2026-06-06 Correction: Direct Slots6 Is Not Yet Correct

The direct slots6 grouped path above passed short smokes, but failed a longer
deterministic 100-token prompt. Do not treat it as the current baseline.

Current default has been changed:

```bash
# direct slots6 is opt-in only
DS4_FLASH_MOE_ENABLE_SLOTS6_GROUPED=1
```

Without that opt-in, `DS4_FLASH_MOE_PER_SLOT_BUFFERS=1` uses the known-good
route-wise fallback.

Failure split:

```text
prompt: "craete game of spsce invaders in PyGame, keep files and compile in /tmp/tmp1a compile, test, iterate"
model: ~/Models/flash/dsv4-iq2xxs-expert-major
cache: --ssd-cache 32GB -> slots=112
ctx: 32768
n: 100

mixed bank:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  generation 15.03 t/s

per-slot direct slots6:
  hash c9d0d5855ebc17629e8e5a71bd5f82789d582c79d35f11a30d9e2820c81adfc1
  generation 10.81 t/s
  cmp vs mixed = 1

per-slot route-wise fallback:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  generation 13.14 t/s
  cmp vs mixed = 0
  logs: /tmp/ds4_yolo_32gb_default_20260606_002254
```

High-slot 64GB smoke with corrected default:

```text
per-slot lazy route-wise, --ssd-cache 64GB -> slots=225:
  hash 92fd701d4cd59f7a64448e24b6cdf5f705ab67678c00d166e05d82bef60186d8
  prefill 0.72 t/s
  generation 13.13 t/s
  logs: /tmp/ds4_yolo_64gb_default_20260606_002506

same-shape mixed 64GB comparator:
  generation 9.76 t/s
  logs: /tmp/ds4_yolo_mixed_20260606_001618
```

Agent responsiveness change:

```text
ds4-agent now defaults DS4_METAL_RESUME_PREFILL_MIN=256
unless the user sets it explicitly.
```

Reason: the old library default `32` made short tool-result suffixes such as
`90` tokens take `resume-prefill`; the status footer sat at `prefill 0/90`
until the whole chunk completed. The new agent-only default keeps these small
continuations on `decode-suffix`.

Reviewer takeaway update:

- Current correctness baseline is mixed bank versus per-slot route-wise fallback.
- Direct slots6 needs a kernel correctness fix before being used for speed
  conclusions.
- The per-slot allocation/locality diagnostic remains valuable: even route-wise,
  it beats the giant-bank high-slot cliff in the 64GB/225-slot smoke.

## 2026-06-06 Update: Corrected Slots6 Is Default for Supported Separate Buffers

The direct slots6 failure above has been fixed. The bug was not the per-slot
cache/install path and not cached family views. The first slots6 implementation
also forced a direct six-expert q2_K down-sum kernel, while the profile baseline
keeps `DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=0` and sums six routed-down rows after
the down projection. That changed floating-point accumulation order and altered
tokens.

Current behavior:

```bash
# default for supported independent buffers:
#   topk=6, IQ2_XXS gate/up, Q2_K down
DS4_FLASH_MOE_PER_SLOT_BUFFERS=1

# escape hatches:
DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED=1
DS4_FLASH_MOE_FORCE_PER_ROUTE=1
DS4_FLASH_MOE_ENABLE_SLOTS6_GROUPED=0

# old direct sum6 remains experimental only:
DS4_FLASH_MOE_SLOTS6_DIRECT_DOWN_SUM=1
```

Implementation status:

- slots6 gate/up uses the six separate gate/up buffers in one grouped dispatch.
- down now defaults to `kernel_mul_mv_slots6_q2_K_f32`, which binds six separate
  down buffers and writes six normal routed-down rows in one dispatch.
- the existing exact `moe_sum_experts` kernel performs the final sum, preserving
  route-wise/mixed output.
- `DS4_FLASH_MOE_SLOTS6_DISABLE_GROUPED_DOWN=1` forces the slower six-dispatch
  down fallback for A/B.

Latest validation:

```text
n50, 32GB/112 slots:
  route fallback:       hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 9.80 t/s
  default slots6 group: hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 11.01 t/s
  cmp=0

n100, 32GB/112 slots:
  route fallback:       hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 9.06 t/s
  default slots6 group: hash 004407eae977e457fac5ef8fbcc147c2f53d5a8ee7a32fb7a15d3a9487ffadfc, generation 9.02 t/s
  cmp=0

n50, 64GB/225 slots:
  route fallback:       hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 10.60 t/s
  default slots6 group: hash d316b5d3a34eea56e017c76620b831334668f55d30e2c99307803bb3412d4766, generation 10.90 t/s
  cmp=0
```

Updated reviewer takeaway:

- Use default per-slot grouped slots6 as the current separate-buffer baseline.
- Keep the old direct sum6 disabled unless specifically testing numerical drift
  versus speed.
- Longer 500-token sweeps are still needed; short runs are dominated by empty
  lazy-slot first-token installs, visible as `processing 0/N` before the prompt
  counter advances.
