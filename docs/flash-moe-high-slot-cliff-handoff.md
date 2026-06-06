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

What I would do next:

- keep decode prefetch off while debugging this cliff;
- for large `--ssd-cache`/high-slot runs, allocate or recreate the decode slot
  bank after prefill by default;
- disable or reduce prefill bank-prefetch writes into the final resident bank;
- avoid direct `pread` into final resident Metal buffers for high-slot banks until
  Metal/page-placement behavior is understood;
- test a Metal-copy based warm-expert transfer from prefill scratch into the
  fresh decode bank.

The longer-term design should probably separate prefill streaming scratch from
decode resident slots. Prefill can stream experts efficiently, but the resident
decode bank should be filled in a decode-friendly way, ideally by GPU copy or a
Metal write path that preserves GPU read locality.
