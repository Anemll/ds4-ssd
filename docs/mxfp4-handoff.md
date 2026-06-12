# MXFP4 branch — handoff

Branch: `MXFP4` on https://github.com/Anemll/ds4-ssd (cut from
`codex/stable-slot-replay-experiment`). Everything below is committed and
pushed. Companion docs: `docs/mxfp4-native-sidecar-plan.md` (implementation +
precision audit), `docs/flash-moe-stable-slot-progress.md` (cliff
investigation, appended sections dated 2026-06-11).

## What works (validated)

- **Native MXFP4 routed experts** decode end to end. Enum 39 / `block_mxfp4`
  (17 B = e8m0 scale + 32 e2m1 nibbles, ggml split-half order) threaded
  through: manifest parse -> slot banks -> Metal kernels -> backend dispatch.
  - Kernels: `dequantize_mxfp4` + `mul_mm_id_mxfp4` (f32/f16, n64/128/256),
    `mul_mv_id_mxfp4_f32`, `mpp_dequant_mxfp4_transpose_i8` (NAX int8),
    `ane_dequant_mxfp4_transpose_f16` (ANE path).
  - MXFP4 qualifies for the same ANE i8i8 (W8A8) prefill backend as Q4_K.
- **FP8 dense converter** `fp4_samples/convert_native_dense_to_ds4.py`.
  The native package's dense GGUF is fork-specific FP8 (type 42 = {e8m0;
  e4m3fn[128]}, 129 B/block, scale-first) + BF16 + renamed tensors; no stock
  loader reads it. Converter rebuilds it template-driven against the chat-v2
  dense schema (Q8_0/F16, metadata copied verbatim incl. chat template,
  `exp_probs_b.bias` borrowed from chat-v2 — router gate weights are
  bit-identical between exports).
- **Precision audit vs HF original** (`/Volumes/TB36/Models/DS/DeepSeek-V4-Flash`):
  - HF is 4-tier: FP8 attn/shexp, native MXFP4 routed experts (stored as
    "I8" + e8m0 scales — safetensors has no FP4 dtype), BF16 embed/head/norms,
    F32 hc/sinks/bias.
  - Sidecar expert bytes: **bit-exact** with HF (verified, nibble orders
    reconciled). Native GGUF dense: **bit-exact** FP8 repack.
  - Only loss in the chain: dense Q8_0 re-encode, 0.55% relRMS (~4-5x below
    the FP8 grid's own step). Embeddings exact.
- **Quality**: graded QA (ds4-eval GPQA/SuperGPQA) 2/2, identical answers to
  Q4K. 2-turn resume-after-decode repro clean. ds4-server OpenAI API smoke
  clean.

## Package layout (IMPORTANT — differs between machines)

`DSv4-Flash-MXFP4-native-flash/`:
- `manifest.json` + `layer_000..042.bin` — untouched, bit-exact with HF.
- `dense/model-dense.gguf` — **on the M5 Max local copy this is the CONVERTED
  Q8_0 file** (8,806,710,848 B). Original FP8 preserved alongside as
  `dense/model-dense-fp8-native.gguf` (8,978,441,472 B).
- `dense/flashmoe-package.json` — updated to describe both files.
- **The SN8100/M3U copy may still have the FP8 original under the canonical
  name** — if ds4 prints "unsupported GGUF type 42", run the converter there
  or copy the converted dense over (only dense/ differs; never re-copy the
  145 GB of layer bins).

## Performance numbers (M5 Max 128 GB, slot-bank 48 unless noted)

| metric | MXFP4 native pkg | Q4K pkg |
|---|---|---|
| 16k prefill (ANE i8i8) | 315.9 t/s | 312.9 t/s |
| decode @16k ctx | 4.59 | 4.47 |
| short-ctx decode | ~10 | 9.8 |

ANE i8i8 beat GPU mul_mm_id A/B (315.9 vs 307.9). Wide tiles default-off.

## THE DECODE CLIFF (page-cache squeeze) — the active iteration topic

Root cause, established with iostat A/B on M5 Max and reproduced on the
96 GB M3U: **decode-miss preads of the sidecar are served by the macOS file
cache at RAM speed; the wired slot bank evicts that cache.** When sidecar
(145 GB) >> RAM, every GiB wired into the bank costs more than it gains.

- M5 Max sweep (cold decode t/s): 8 GB bank: 10.5 | 16: 10.0 | 24: 9.2 |
  33: 6.5 | 70: 0.21. Monotone — smaller bank = faster decode.
- M3U 96 GB: 32 GB bank -> 5.67 t/s; 48 GB bank -> ~0.1 t/s (58.6 GB wired,
  ~29 GB left for cache).
- NOT fixed by: `DS4_FLASH_MOE_DIRECT_SLOT_PREAD=0`, realloc-after-prefill,
  expert mmap (`DS4_FLASH_MOE_EXPERT_MMAP=1`), raising
  `iogpu.wired_limit_mb` (tested 84 GiB on M3U — no change; the working set
  was under the budget anyway; raising it points the wrong way).
- Warm bank doesn't help either (slot131 + 16k prefill: 0.15 both ways) —
  with max-loads=0 + ~50% miss IO, the IO volume itself is the cost.
- Prefill is UNAFFECTED — 16k prefill stays ~313-320 t/s even at slot131.

Mitigations landed:
- `--ssd-cache auto` now budgets `DS4_SSD_CACHE_AUTO_PCT`% (default 20,
  was 85) of remaining RAM.
- Startup prints the GPU working-set budget (`recommendedMaxWorkingSetSize`)
  with the `sudo sysctl iogpu.wired_limit_mb=<MB>` hint; warns when the bank
  exceeds it; big-bank warning text rewritten with the measured guidance.

Practical settings:
- M5 Max 128 GB: `--ssd-cache auto` or `--moe-slot-bank 32..48`.
- M3U 96 GB: any `--ssd-cache` is now decode-safe — banks ≥44 GiB auto-shrink
  for decode (see below). `--ssd-cache 48GB` measured **3.2 t/s** decode (was
  0.24); `32GB` left as-is at 2.7 t/s.
- `iogpu.wired_limit_mb`: only for models that FIT in RAM; irrelevant here.

## DECODE CLIFF — FIXED (2026-06-11): auto decode-bank shrink

The page-cache-squeeze cliff is fixed in production. Confirmed on the 96 GB M3U
with iostat: `--ssd-cache 48GB` decode = 0.24 t/s with the sidecar SSD read at a
sustained ~610 MB/s *during decode* (every miss a real SSD read — the wired
47.7 GiB bank had evicted the file cache). Both configs sat far under the
84 GiB GPU working-set budget (58.6 vs 42.6 GiB), so this is not a Metal
residency/coherency stall — it is genuine miss IO.

Fix (`metal_graph_flash_moe_*` in `ds4.c`): keep the requested bank for prefill,
then **shrink the slot bank after prefill** to a small decode bank so the OS
file cache repopulates and serves misses at RAM speed. Free + retarget
(`g->flash_slot_bank` and the sidecar `slot_bank`) + realloc + cache-reset; the
slot index arrays are allocated for the original (larger) bank so a smaller
stride only under-uses them. Prefill is unaffected by bank size, so nothing is
lost. This is the shrinking version of `REALLOC_SLOT_BANK_AFTER_PREFILL`.

- **Automatic**, fires only in the RAM-limited regime (sidecar > physical RAM)
  AND only for banks ≥ 44 GiB (the cliff-warning threshold). Memory-rich
  machines and already-small banks are untouched. Target = the same size
  `--ssd-cache auto` would pick: `DS4_SSD_CACHE_AUTO_PCT`% (default 20) of RAM
  left after dense+context. On the M3U, 48 GiB → 31 slots / 16.6 GiB decode.
- Knobs: `DS4_FLASH_MOE_DECODE_SLOT_BANK=<slots>` forces an exact decode bank
  (`=0` opts out / keeps the big bank); `DS4_FLASH_MOE_DECODE_SSD_CACHE=<size>`
  sets it by byte budget (e.g. `20GB`). Either overrides the automatic target.
- Measured (M3U 96 GB, 14-tok cold prompt, n=16): `48GB` 0.24 → **3.23 t/s**
  (89→31 slots), beating the plain `32GB` run (2.70). Output bit-coherent.
- **Prefill overflow clamp**: explicit `--ssd-cache` is clamped at startup so
  bank+dense+context ≤ `DS4_SSD_CACHE_MAX_PCT`% (default 85) of RAM —
  `--ssd-cache 80GB` on the M3U clamps to 70.6 GiB, prefills fine, shrinks
  131→31, decodes 2.74 t/s. Any requested size is now safe end-to-end.
- Negative results (so nobody re-runs them): decode-time LRU bank warming
  (`max-loads` 2/6) and a smaller post-shrink bank (6 slots) are both flat vs
  the defaults — post-shrink decode is file-cache-served and insensitive to
  decode-bank size. All variants bit-identical output.

## Next iteration candidates (in rough value order)

1. ~~**Bank-release-after-prefill**~~ — **DONE** (see "DECODE CLIFF — FIXED"
   above). Remaining polish: a decode-time iostat A/B confirming the disk goes
   idle post-shrink (predicted), and a server multi-turn check that the shrunk
   bank stays put across turns (it does — sidecar slot_bank is retargeted, so
   later prefills/shrinks are idempotent at 31 slots).
2. ~~**Decode-time bank warming**~~ — **TESTED, NO WIN** (M3U, post-shrink):
   max-loads 0/2/6 → 3.98/3.88/3.87 t/s. Post-shrink misses are served by the
   file cache at RAM speed, so installs are pure overhead. Default stays 0.
   Re-test only on a machine where decode misses hit true SSD even after the
   shrink.
3. **M3U bank sweep** (16/24/32 GB) to pin its decode optimum.
4. **mul_mv pair/pair_swiglu fused decode kernels for MXFP4** — q4_k has
   them, mxfp4 uses plain mv. Only worth it once decode stops being IO-bound
   (i.e., after item 1 on a warm cache).
5. **MPP 4.1 / macOS 27 native FP4 scale-plane matmul** (~2.6x MLP) —
   scaffolding plan in mxfp4-native-sidecar-plan.md; blocked on macOS 27.
6. **HF reupload**: only `dense/model-dense.gguf` (converted) +
   `dense/flashmoe-package.json` changed; layer bins identical. FP8 original
   optional (regenerable from DeepSeek-V4-Flash-FP4-FP8-native.gguf).

## Quick commands

```bash
# clone on a new machine
git clone -b MXFP4 https://github.com/Anemll/ds4-ssd.git && cd ds4-ssd && make

# convert the dense on a machine that still has the FP8 file canonical
mv <pkg>/dense/model-dense.gguf <pkg>/dense/model-dense-fp8-native.gguf
python3 fp4_samples/convert_native_dense_to_ds4.py \
  --template <chat-v2-pkg>/dense/model-dense.gguf \
  --native   <pkg>/dense/model-dense-fp8-native.gguf \
  --out      <pkg>/dense/model-dense.gguf

# run (M3U 96GB)
./ds4 -m <pkg> --ssd-cache 24GB -p "..."

# diagnose decode slowness: watch disk during decode
iostat -d -w 5   # sustained 300+ MB/s during decode = cache squeeze
```

## Gotchas that cost time (don't rediscover)

- ggml MXFP4 nibble order is split-half (qs[j] lo=elem j, hi=elem j+16);
  the HF safetensors packing is sequential-pair. Values identical, bytes not.
- E8M0 decode = `as_type<float>(uint(e)<<23)`; e=0 -> 0.0.
- `routed_expert_row_bytes_for_type` and the ANE-prefill row-bytes helper
  both assumed 256-elem blocks; MXFP4 is 32 (fixed, but watch for other
  QK_K assumptions when adding formats).
- Small slot banks still use one Metal buffer per LAYER ("mixed expert-major").
  Large banks (>=44 GiB capacity) now auto-split into per-slot buffers to avoid
  the warm O(bank) routed_moe cliff. `DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1`
  restores the legacy mixed-bank path for A/B only.
- M5 Max thermals + concurrent runs invalidate benchmarks; cooldown between
  runs, never bench in parallel.
- The first run after a cache-polluting run is penalized while the page
  cache repopulates; trust the second run at a given config.

## BREAKTHROUGH (2026-06-12): the warm-big-bank cliff is O(bank) in routed_moe, NOT miss IO

User pushback forced a re-test that overturned the cache-squeeze-only model:

- 90 GB bank, warm, 80% hit rate, only ~26 misses/token (~330 MB IO): 0.22 t/s.
- 20 GB bank, ~229 misses/token (~2.9 GB IO): 3.6-10 t/s.
  9x LESS IO but 30x SLOWER => IO volume cannot be the warm-regime mechanism.
- DS4_FLASH_MOE_DID_MODIFY_RANGE=0 A/B at 90 GB: identical 0.22 t/s
  (didModifyRange is a no-op on shared buffers; neither missing nor harmful).
- DS4_METAL_DECODE_STAGE_PROFILE=1 localizes it completely:
  routed_moe = 80.4 ms/layer at 90 GB vs 4.1 ms/layer at 20 GB (19x), all
  other stages flat. 80.4 x 43 = 3.5 s of the 4.5 s token. Matches the old
  slot225 finding (x6.7) - same mechanism, quant- and machine-independent.
- Cache squeeze remains real but only for the COLD regime (258 misses/token,
  iostat-confirmed disk-bound). Two mechanisms, not one.

Leads for the O(bank) routed_moe cost (next session, in order):
1. ICB replay encode calls [enc useResource:] per entry incl. the layer's
   mixed bank buffer (ds4_metal.m ~17313-17322, ~17686-17701). Driver
   validation of a 2.14 GB freshly-CPU-written wired buffer per layer per
   token would scale with bank bytes. Test: disable ICB replay at 90 GB
   (ds4_gpu_flash_moe_icb_replay_enabled env) and compare.
2. Hazard tracking: bank buffers are default-tracked; CPU writes + GPU reads
   on a huge tracked buffer may serialize. Test: allocate banks with
   MTLResourceHazardTrackingModeUntracked (fences already exist via
   ds4_gpu_synchronize at install boundaries).
3. First-GPU-touch page validation of CPU-written pages in the wired bank.
   Test: GPU-blit a dummy read over installed slots right after install,
   off the critical path.

Update (negatives, all at 90 GB warm bank, baseline 0.22-0.26 t/s):
- DS4_FLASH_MOE_SLOT_BANK_RESIDENCY is default OFF; =0 run was a no-op.
- RESIDENCY=1 + TOUCH_PAGES=1 (all 89.95 GiB wired+touched): 0.28 t/s. FLAT.
  Swap/pressure/page-fault-in-kernel theory dead.
- didModifyRange on/off: flat. ICB useResource: path default-off, never ran.
So the O(bank) routed_moe cost is NOT: miss IO, page cache, wiring, swap,
didModify, or ICB. Suspects entering the next session were:
1. Per-dispatch driver cost of binding the per-layer mixed-bank buffer
   (2.14 GB at 168 slots vs 0.26 GB at 44) — setBuffer/commit-time page-table
   work proportional to buffer size, 43 binds/token. Test: allocate the bank
   as N small per-slot-group buffers instead of one mixed buffer per layer
   (per_slot buffers mode exists: flash_moe_per_slot_buffers_enabled).
2. The banked decode kernel itself — read metal/moe.metal banked/slots6
   kernels for any loop bounded by slot_bank rather than n_active experts.
3. Decisive instrument: Instruments "Metal System Trace" on a 6-token decode
   at 90 GB vs 20 GB — splits routed_moe 80 ms into encode/driver/GPU-exec.
Result: suspect #1 was confirmed and fixed below. Do not re-run the eliminated
dead ends above.

## FIXED (2026-06-12): warm-big-bank routed_moe cliff avoided by split per-slot buffers

The first remaining suspect was correct. The huge mixed per-layer buffer is the
warm-regime cliff: binding a 2.14 GiB layer bank per layer/token makes
`routed_moe` scale with bank capacity. Splitting the bank into per-slot expert
buffers removes the giant bind from the decode dispatch path.

Validation on M5 Max 128 GB, explicit `--ssd-cache 90GB`, `--ctx 4096 -n 16`,
prompt `What is Apple Neural Engine`, one run at a time:

- Baseline from the handoff: mixed 90 GB warm bank = 0.22-0.28 t/s generation.
- Diagnostic A/B:
  `DS4_FLASH_MOE_RESIDENCY_STATS=8 DS4_FLASH_MOE_PER_SLOT_BUFFERS=1 ./ds4 ... --ssd-cache 90GB ...`
  => 5.32 t/s generation, 5.75 t/s prefill. Same 168-slot / 89.95 GiB logical
  bank; residency at tok16 was 1763/7224 slots (21.95 GiB resident), hit 62.7%.
- Production validation after the fix, no per-slot env:
  `DS4_FLASH_MOE_RESIDENCY_STATS=8 ./ds4 ... --ssd-cache 90GB ...`
  => `resident=per-slot-auto`, 5.95 t/s generation, 7.38 t/s prefill. This
  beats the >5 t/s target while honoring the explicit 90 GB bank through decode.
- Small-bank smoke after the fix:
  `DS4_FLASH_MOE_RESIDENCY_STATS=8 ./ds4 ... --ssd-cache 20GB ...`
  => `resident=slot-bank`, 7.62 t/s generation, 7.46 t/s prefill. The
  auto-split threshold leaves the fast mixed 20 GB path alone.

Implementation:

- `ds4.c` now auto-selects per-slot expert buffers when the requested Flash-MoE
  bank capacity is >=44 GiB and per-expert/per-slot modes were not explicitly
  requested. Small banks keep the mixed expert-major layer buffer.
- New kill switches for A/B: `DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1`
  or `DS4_FLASH_MOE_FORCE_MIXED_SLOT_BANK=1`.
- Startup logs now report `resident=per-slot-auto` and print a large-bank
  decode guard instead of the old page-cache-only warning when the split is
  active.

Negative / no-longer-needed:

- The banked decode-kernel loop-bound suspect did not need a fix for the
  target: per-slot split alone moves 90 GB from ~0.25 t/s to ~6 t/s.
- Instruments Metal System Trace is no longer required to decide the root
  cause, though it could still quantify encode vs driver time on the legacy
  mixed-bank path if someone wants a postmortem.

## UPDATE (2026-06-12): 90 GB is fixed, but still slower than the 32 GB optimum

User pushback was right: the per-slot-auto fix solved the catastrophic 90 GB
decode cliff and met the >5 t/s target, but it did **not** solve the stronger
goal of making high-residency 90 GB decode as fast as the 32 GB mixed bank.

Current M5 Max 128 GB controls, same prompt, `--ctx 32768 -n 64 --temp 0`:

| config | layout | tok64 residency | tok64 hit | generation |
|---|---|---:|---:|---:|
| 90 GB | `per-slot-auto` / grouped MXFP4 slots6 | 3779/7224 | 85.4% | ~6.0-6.2 t/s |
| 90 GB + lazy per-slot | `per-slot-auto-lazy` | 3779/7224 | 85.4% | 6.01 t/s |
| 40 GB | mixed `slot-bank` | 3078/3182 | 80.2% | 7.15 t/s |
| 32 GB | mixed `slot-bank` | 2535/2537 | 77.3% | 7.48 t/s |

Negative experiments added to the "do not re-run casually" list:

- MXFP4 grouped `slots6` per-slot decode is correct and a small win over
  fallback (`90GB -n64`: 6.18 vs 6.02 t/s), but it does not close the gap to
  32 GB mixed (`7.4+ t/s`).
- Active six-expert staging prototype (90 GB per-slot storage -> tiny staged
  bank -> existing banked MoE) was correct but slower: 5.16 t/s at
  `--ctx 4096 -n 16` vs 5.47 t/s matched no-staging. The copy layer loses.
  The scaffold was removed; result is recorded in
  `docs/mxfp4-mixed-residency-plan.md`.
- `DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC=1` reduces upfront Metal allocation and
  gives only a tiny short-run lift (5.56 vs 5.47 t/s at `--ctx 4096 -n 16`);
  long run remains slower than 32 GB (6.01 t/s at `--ctx 32768 -n 64`).
- 40 GB mixed improves hit rate over 32 GB but is already slower. The
  speed/residency optimum on this prompt is near 32 GB, not "as large as
  possible".

Updated direction:

- Treat the remaining problem as a policy/working-set tradeoff, not just a
  missing kernel. Extra 90 GB residency is real, but on this short prompt the
  saved misses do not pay for the larger decode working set.
- The next credible mixed solution should keep decode on a moderate fast mixed
  bank (around the measured optimum) and use extra memory only when it replaces
  true SSD reads: e.g. an auto-tuned decode bank cap, or a real two-tier
  mixed-L1 / large-L2 cache with promotion. Do not resurrect active staging
  unless a new workload proves the copy cost is cheaper than true disk misses.

## PROTOTYPE (2026-06-12): explicit 90 GB request with fast 32 GB mixed decode L1

The first policy prototype works: keep the initial large request, but make
decode run from a moderate mixed L1 instead of the full 90 GB hot set.

Implementation:

- An explicit decode-bank cap now suppresses the large-bank auto per-slot guard
  so the graph can allocate a mixed prefill bank and then shrink it after
  prefill.
- Existing knobs now work for large explicit banks:
  `DS4_FLASH_MOE_DECODE_SLOT_BANK=<slots>` or
  `DS4_FLASH_MOE_DECODE_SSD_CACHE=<size>`.
- New shorthand prototype:
  `DS4_FLASH_MOE_FAST_DECODE_L1=1` defaults the decode L1 budget to 32 GB.
  Override with `DS4_FLASH_MOE_FAST_DECODE_SSD_CACHE=<size>` or
  `DS4_FLASH_MOE_DECODE_L1_SSD_CACHE=<size>`.
- `DS4_FLASH_MOE_DECODE_SLOT_BANK=0` remains an explicit opt-out.

Validation on M5 Max 128 GB:

```bash
DS4_FLASH_MOE_RESIDENCY_STATS=16 \
DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 64 --temp 0 \
  -p "What is Apple Neural Engine"
```

Observed:

- Startup: 168-slot / 89.95 GiB mixed prefill bank.
- After prefill: shrinks 168 -> 59 slots / 31.59 GiB mixed decode bank.
- `generation: 7.51 t/s`, `prefill: 6.00 t/s`, tok64 hit 77.3%.
- Fresh 32 GB reference was `generation: 7.48 t/s`, tok64 hit 77.3%.

This is not a true 90 GB L2 yet: the shrink resets the slot cache, so decode
hit rate is the 59-slot L1 hit rate. But it gets explicit 90 GB requests back
to 32 GB-class decode speed and gives the next prototype a clean base: add a
separate L2/promotion path only for cases where it avoids true SSD reads.

## UPDATE (2026-06-12): L1/L2 and six-slot baselines are negative

User noticed correctly that the first CPU-L2 run had low RAM use. Cause:
default prefill uses `slot-cache-topk=0`, so the 90 GB prefill bank was not
populated before the shrink. That run (`DECODE_L2=1`, 32 GB L1) captured
`0` L2 records and generated 6.04 t/s; treat it only as an empty-L2 overhead
check, not as a high-residency test.

Corrected high-RAM L1/L2 test:

```bash
DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=84 \
DS4_FLASH_MOE_DECODE_SSD_CACHE=32GB \
DS4_FLASH_MOE_DECODE_L2=1 \
./ds4 -m ~/Models/DSv4-Flash-MXFP4-native-flash \
  --ssd-cache 90GB --ctx 32768 -n 16 --temp 0 \
  -p "What is Apple Neural Engine"
```

Result: 168-slot / 89.95 GiB mixed prefill bank -> 59-slot / 31.59 GiB
mixed decode L1, CPU L2 captured 1951 resident records, `prefill: 2.22 t/s`,
`generation: 0.22 t/s`. A default-length version captured 2032 records and was
killed after more than 12 minutes. CPU-backed L2 promotion/copy is therefore
not a viable path for the short 90 GB target.

Six-slot baselines:

- `DS4_FLASH_MOE_DECODE_SLOT_BANK=6 DS4_FLASH_MOE_SIX_SLOT_BASELINE=1`
  forces no slot reuse and reloads the active six experts into slots 0..5 every
  layer: 6.85 t/s.
- `DS4_FLASH_MOE_DECODE_SLOT_BANK=6` with normal LRU reuse: 9.11 t/s.
- Attached user 32 GB control, same workflow shape: 12.71 t/s.

Conclusion: the 32 GB speed comes from the mixed-bank grouped compute path plus
enough temporal slot reuse. "Just six slots" is too miss-heavy, and CPU-L2
promotion makes the decode hot path worse. Do not re-run CPU-L2, GPU-L2,
active staging, or six-slot no-reuse unless the workload changes radically.

Chunked mixed no-copy probe is also negative:

- `DS4_FLASH_MOE_CHUNKED_MIXED=1 DS4_FLASH_MOE_CHUNK_SLOTS=56`: full 168-slot /
  89.95 GiB bank split into 3 chunks, `generation: 5.48 t/s`.
- `DS4_FLASH_MOE_CHUNKED_MIXED=1 DS4_FLASH_MOE_CHUNK_SLOTS=84`: full bank split
  into 2 chunks, `generation: 4.41 t/s`.

Chunking avoids both the giant full-layer bind and L2 copies, but multiple
chunk dispatches plus accumulation still lose badly to the attached 32 GB
control (12.71 t/s). Treat chunked mixed as eliminated for the short 90 GB
target.
