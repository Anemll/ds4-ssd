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
- M3U 96 GB: `--ssd-cache 16GB..24GB` (predicted 6-8 t/s decode; 32 GB
  measured 5.67). Avoid 48 GB+.
- `iogpu.wired_limit_mb`: only for models that FIT in RAM; irrelevant here.

## Next iteration candidates (in rough value order)

1. **Bank-release-after-prefill (production fix for the cliff).** Big banks
   help prefill streaming and never hurt it; they only strangle decode.
   Prototype: after prefill completes, free/shrink the slot bank to a small
   decode bank (~32 slots), letting the file cache repopulate. This is the
   production version of `DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL`,
   but shrinking instead of same-size recreate. Would make `--ssd-cache 48GB`
   safe: large for prefill, small for decode.
2. **Decode-time bank warming** (`max-loads` defaults): with max-loads=0 the
   bank never warms during decode; misses re-read the same experts from disk
   forever on decode-heavy sessions. Cheap experiment:
   `DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=2..6`.
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
- Slot-bank allocation is one Metal buffer per LAYER ("mixed expert-major"),
  not per expert; per-expert/per-slot buffers are opt-in diagnostic modes.
- M5 Max thermals + concurrent runs invalidate benchmarks; cooldown between
  runs, never bench in parallel.
- The first run after a cache-polluting run is penalized while the page
  cache repopulates; trust the second run at a given config.
