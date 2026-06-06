# DS4 SSD Streaming And Prefill Knobs

This document covers the Flash-MoE SSD streaming path:

```sh
./ds4 \
  -m /path/to/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 8 \
  --ctx 8192
```

In this mode, the dense/shared model tensors stay mmap-backed while routed MoE
expert records are streamed from the sidecar and cached in per-layer Metal slot
banks. Most knobs below are environment variables because they are also used by
`ds4_profile.json`. Command-line flags exist only for the knobs that are stable
enough to expose directly.

Profiles are applied before the engine opens the model. User environment wins:
if an environment variable is already set, the profile leaves it alone.
`DS4_PROFILE=none` disables profile loading, and `DS4_PROFILE=/path/file.json`
uses a specific profile file. Sidecar runs use the matching profile's
`sidecar_env` block.

At startup, check these lines to confirm the effective configuration:

```text
ds4: applied sidecar tuning profile [...]
ds4: Flash-MoE sidecar loaded: ... (slot-bank=N, expert-record X MiB)
ds4: Flash-MoE slot banks allocated: layers=... slots=N gpu-bank=Y MiB
ds4: prefill compute: ...
ds4: prefill I/O: io-split=... async-pread=... pread-threads=... readahead=... bank-prefetch=... slot-cache-topk=... xlayer=...
ds4: decode  I/O: io-split=... router-prefetch=... scratch-prefetch=... max-loads=... miss-direct-slot-pread=... reset-after-prefill=... slots=N
```

## Experimental Pro Support

DeepSeek V4 Pro sidecar support is experimental. For Pro agent testing, start
with `--moe-slot-bank 32` or lower, add `--nothink`, and keep shared-down
decode prefetch enabled:

```sh
DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN=1 ./ds4-agent \
  -m ~/Models/DSv4Pro-flash/ \
  --moe-slot-bank 32 \
  --ctx 32768 \
  --nothink
```

Pro expert records are larger than Flash records, so high slot counts can make
decode slower even when SSD I/O looks low. Treat `32` slots as the upper
baseline for Pro until reuse and stall traces show that a larger bank helps.

To test fewer actual streamed experts, add `--moe-expert-topk 4` or export
`DS4_MOE_EXPERT_TOPK=4`. This applies to both prefill and decode routed refs.
It is a quality-changing diagnostic override, not a cache/prefetch hint.

## Profile-Style Block

This is the shape of a sidecar streaming/prefill tuning block. It is useful for
agent handoff notes, local A/B scripts, and `sidecar_env` entries in
`ds4_profile.json`.

```sh
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_FLASH_MOE_SLOT_BANK_SLOTS=48 \
DS4_FLASH_MOE_ANE_PREFILL=1 \
DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1 \
DS4_FLASH_MOE_OVERLAP_PREFILL=1 \
DS4_FLASH_MOE_OVERLAP_SCHEDULER=1 \
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 \
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1 \
DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
DS4_FLASH_MOE_MPP_I8I8_PREFILL=0 \
DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0 \
DS4_FLASH_MOE_ANE_DUAL=1 \
DS4_FLASH_MOE_ANE_THREADS=2 \
DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1 \
DS4_FLASH_MOE_ANE_BATCH=256 \
DS4_FLASH_MOE_ANE_BATCHES=256 \
DS4_FLASH_MOE_ANE_MAX_REFS=256 \
DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS=1 \
DS4_FLASH_MOE_ANE_MIN_REFS=32 \
DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=384 \
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=4 \
DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=4 \
DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=1 \
DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK=0 \
DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 \
DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0 \
DS4_FLASH_MOE_PREFETCH=3 \
DS4_FLASH_MOE_ASYNC_PREAD=1 \
DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE=1 \
DS4_FLASH_MOE_PREAD_THREADS=6 \
DS4_FLASH_MOE_ASYNC_READAHEAD=12 \
DS4_FLASH_MOE_SCHED_ANE_REL_SPEED=99 \
DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL=0.0 \
DS4_FLASH_MOE_MPP_INT8_QSCALE=512 \
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32 \
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32
```

Do not blindly pair `DS4_METAL_PREFILL_CHUNK=16384` with
`DS4_METAL_GRAPH_RAW_CAP=8704`. DS4's raw sliding window is 128 rows, and the
raw cap target for a true 16K chunk is `align_up(128 + 16384, 256) = 16640`.
With `DS4_METAL_GRAPH_RAW_CAP=8704`, the effective chunk is only `8576` tokens
and startup should warn that raw KV cap is clamping prefill. That can be useful
for an 8K-class A/B run, but it is not a 16K prefill setting.

Some older experiment blocks used `DS4_METAL_PREFILL_CHUNK=16000`. That is still
an explicit chunk size, but the current sidecar profile convention is `16384`.

## User Knobs

These are the knobs most likely to matter for an end user tuning RAM occupancy,
SSD behavior, or quality.

| Knob | CLI flag | Code default | Profile/default notes | Meaning |
| --- | --- | ---: | --- | --- |
| `DS4_FLASH_MOE_SLOT_BANK_SLOTS` | Use `--moe-slot-bank N` | Not consumed directly | Current profiles record values such as `96`, `64`, or `48` | Profile-side suggested slot count. The live slot bank is currently sized by `--moe-slot-bank`, not by this env var. |
| `--moe-slot-bank N` | `--moe-slot-bank N` | `32` for `ds4`, `ds4-server`, `ds4-agent`; `8` for `ds4-bench` | Valid range `6..256` | Number of cached routed expert slots per layer. Higher values cache more experts and use more unified/Metal memory; lower values use less RAM and stream/reload more often. |
| `--ssd-cache BYTES\|auto` | `--ssd-cache 25GB`, `--ssd-cache auto` | unset | Shared by `ds4`, `ds4-agent`, and `ds4-server` | Sizes the Flash-MoE slot bank from a memory budget instead of a slot count. Explicit sizes are the target slot-bank GPU cache budget. `auto` uses currently available memory, subtracts dense mapped weights and estimated context buffers for `--ctx`, then assigns 85% of the remainder to slots. |
| `DS4_METAL_PREFILL_CHUNK` | none | Whole prompt if `<=4096`, else `4096`; `0` means whole prompt | Current sidecar profiles use `16384` | Max Metal prefill chunk size. Use `16384` for the alpha sidecar path unless measuring another value. Do not assume `32768` chunks are supported. |
| `DS4_CTX_GROW` | none | `1` | Profiles set `1` | Enables high-water compressed-KV growth instead of eagerly allocating the full `--ctx` compressed KV cache. Usually leave on for streaming. |
| `DS4_CTX_GROW_BLOCK` | none | `2048` | Profiles use `16384` on larger-memory sidecar targets and `2048` on tighter targets | Context-token growth step for compressed KV. Larger values reduce grow events; smaller values keep memory occupancy tighter. |
| `DS4_METAL_GRAPH_RAW_CAP` | none | Auto | Usually unset | Raw KV row cap. Leave unset unless debugging chunk clamps. For true 16K prefill, use auto or at least `16640`; `8704` clamps the effective chunk to `8576`. |
| `--ctx N` | `-c N`, `--ctx N` | `32768` for CLI/server, `100000` for agent | User-selected | Total context ceiling. Larger contexts reserve or grow more KV memory, so slot-bank choices should leave room for it. |
| `DS4_NO_INT8` | `--no-int8`; `--quality` implies it | `0` | User opt-in | Disables current int8 dense, NAX, Flash-MoE, and ANE accelerator paths for quality-preserving runs. Streaming remains available, but prefill may fall back to GPU/NAX-half paths and slow down. |

## Sidecar Mode

| Knob | CLI flag | Code default | Meaning |
| --- | --- | ---: | --- |
| `-m DIR` package root | `-m DIR` | none | If `DIR` contains `manifest.json` and `dense/model-dense.gguf`, DS4 auto-detects SSD sidecar mode and uses `DIR` as the sidecar root. |
| `--moe-sidecar DIR` | `--moe-sidecar DIR` | none | Explicit sidecar directory containing `manifest.json` and expert records. Needed only when `-m` is not the package root. |
| `--moe-mode slot-bank` | `--moe-mode NAME` | `off` | Selects the streaming slot-bank routed expert source. Inferred automatically for package-root `-m DIR`; explicit `--moe-sidecar` still needs this flag except in `ds4-bench`. |
| `DS4_PROFILE` | none | auto-search | `none` or `0` disables profiles. Any other value is interpreted as a profile path. |

## I/O And Cache Knobs

These control SSD read scheduling, decode-side caching, and prefill staging.
They are safe to document, but most users should start with profile defaults.

| Knob | CLI flag | Code default | Profile/default notes | Meaning |
| --- | --- | ---: | --- | --- |
| `DS4_FLASH_MOE_GPU_DEDUP` | none | `1` | Profiles set `1` | Uses GPU kernels to compact routed expert references during prefill. Set `0` only for CPU-path A/B tests. |
| `DS4_FLASH_MOE_PREFETCH` | none | `3` | Profiles set `3` | Prefill staged-bank look-ahead. Four prefill banks are allocated, so values above `3` are clamped. |
| `DS4_FLASH_MOE_CACHE_IO_SPLIT` | `ds4-agent --moe-cache-io-split N` | `4` | Agent flag overrides env | Splits decode/slot-bank expert reads into up to `N` page-aligned concurrent reads. Clamped `1..16`; page-misaligned reads fall back to `1`. |
| `DS4_FLASH_MOE_PREFILL_IO_SPLIT` | `ds4-agent --moe-prefill-io-split N` | Inherits `DS4_FLASH_MOE_CACHE_IO_SPLIT` | Agent flag overrides env | Same split policy for prefill expert reads. |
| `DS4_FLASH_MOE_DECODE_PREFETCH` | `ds4-agent --moe-prefetch-temporal`, `--no-moe-prefetch-temporal` | `1` | Some profiles set `1` explicitly | Enables temporal decode prefetch so likely next expert records are read before the layer needs them. |
| `DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN` | none | `0` | M5-family sidecar profiles set `1` | Extends decode prefetch overlap through shared-down work. Set `0` for A/B tests against blocking install or gate/up-only overlap. |
| `DS4_MOE_EXPERT_TOPK` | `--moe-expert-topk N` | Model metadata (`deepseek4.expert_used_count`, usually `6`) | Experimental diagnostic; `DS4_FLASH_MOE_EXPERT_TOPK` is accepted as an env alias | Overrides the actual routed expert fanout. `4` means prefill emits `tokens*4` routed refs and decode streams/computes 4 experts per layer/token. Changes logits/quality. |
| `DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK` | `ds4-agent --moe-prefetch-topk N` | Usually `0`; explicit values are clamped to half the slot bank | Agent flag overrides env | During prefill, installs the top routed experts per layer into the decode slot cache for reuse. |
| `DS4_FLASH_MOE_PREFILL_SLOT_PREFETCH` | none | Auto | Auto can engage when decode prefetch is on and ANE prefill is off | Forces whether prefill should populate the decode slot cache. |
| `DS4_FLASH_MOE_DIRECT_SLOT_PREAD` | none | `1` | Usually leave on | On decode/prefill slot-cache misses, reads expert bytes directly into the CPU-visible Metal slot buffer. Set `0` to force the staged path: `pread` into a CPU scratch record, then `ds4_gpu_tensor_write` into the slot. |
| `DS4_FLASH_MOE_ASYNC_PREAD` | none | `0` | Sidecar profiles set `1` | Enables async prefill expert reads through a reader pool. |
| `DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE` | none | `0` | Sidecar profiles set `1` | Allows the async reader to keep running after staging begins. |
| `DS4_FLASH_MOE_PREAD_THREADS` | none | `4` | Profiles use `4` or `6` | Number of async prefill read worker threads. Clamped `1..8`. |
| `DS4_FLASH_MOE_ASYNC_READAHEAD` | none | `12` | Profiles set `12` | Number of expert reads queued ahead into async prefill buffers. |
| `DS4_FLASH_MOE_XLAYER_PREFETCH` | none | Auto: on for prefill chunks `<=6000` tokens, off above that | Force `0` or `1` for A/B tests | Cross-layer prefetch: predicts next layer's hot experts from the current layer and starts SSD reads early. |
| `DS4_FLASH_MOE_XLAYER_TOPK` | none | `slot_bank / 2` | Clamped to preserve reader headroom | Number of predicted next-layer experts to pre-stage. |
| `DS4_FLASH_MOE_XLAYER_ATTN_ONLY` | none | `1` | Usually leave on | Pauses speculative cross-layer reads during ANE expert evaluation so SSD/unified-memory traffic stays in the attention/dense/router window. |
| `DS4_FLASH_MOE_HIST_CSV` | none | unset | Diagnostic | Writes per-layer prefill expert histograms to CSV. |

`ds4-agent` also accepts `--moe-prefill-banks N`, but the current runtime uses a
fixed four-bank prefill allocation and does not read `DS4_FLASH_MOE_PREFILL_BANKS`.
Tune `DS4_FLASH_MOE_PREFETCH=0..3` instead.

## Compute And Accelerator Knobs

These choose the routed-MoE prefill compute path. Profiles set these for known
machines such as M3 Ultra and M5 Max. Change them only for controlled A/B runs.

| Knob | CLI flag | Code default | Profile/default notes | Meaning |
| --- | --- | ---: | --- | --- |
| `DS4_FLASH_MOE_ANE_PREFILL` | none | `0` | Sidecar profiles set `1` on machines where ANE is faster for the measured chunk shapes; M1 Max profile keeps it `0` | Enables ANE routed-MoE prefill for sidecar streaming. |
| `DS4_FLASH_MOE_ANE_PIPELINE_PREFILL` | none | `0` | Profiles set `1` with ANE sidecar prefill | Enables pipelined ANE prefill scheduling. |
| `DS4_FLASH_MOE_OVERLAP_PREFILL` | none | `0` | Profiles set `1` with ANE sidecar prefill | Allows ANE and GPU work to overlap during sidecar prefill. |
| `DS4_FLASH_MOE_OVERLAP_SCHEDULER` | none | `0` | Profiles set `1` with ANE sidecar prefill | Enables the overlap scheduler that assigns groups to ANE/GPU lanes. |
| `DS4_FLASH_MOE_MPP_INT8_PREFILL` | none | `0` | Profiles generally keep `0` for sidecar ANE path | Enables MPP/NAX int8 prefill path when supported. `--no-int8` disables it. |
| `DS4_FLASH_MOE_MPP_I8I8_PREFILL` | none | `0` | Profiles keep `0` | Experimental MPP/NAX W8A8 prefill variant. |
| `DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL` | none | `0` | Profiles keep `0` | Experimental fused MPP/NAX W8A8 prefill variant. |
| `DS4_FLASH_MOE_ANE_I8I8_PREFILL` | none | `0` unless profile/default helper sets it | Profiles set `1` for ANE-capable paths | Enables the ANE W8A8 MLP kernel family. |
| `DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL` | none | `0` unless profile/default helper sets it | Profiles set `1` for current ANE sidecar path | Selects the tiled-fused W8A8 ANE routed kernel. |
| `DS4_FLASH_MOE_ANE_DUAL` | none | `0` | M3 Ultra profiles set `1`; single-cluster systems set `0` | Enables dual ANE cluster scheduling for supported hardware. |
| `DS4_FLASH_MOE_ANE_THREADS` | none | `2` when `DS4_FLASH_MOE_ANE_DUAL=1`, else `1` | Profiles use `2` on M3 Ultra and `1` on single-cluster systems | Explicit ANE worker cap. Clamped `1..4`. |
| `DS4_FLASH_MOE_ANE_MULTI_ACTIVE` | none | `0` | Profiles set `1` for current ANE sidecar path | Allows multiple active ANE jobs instead of synchronously waiting after each prediction call. |
| `DS4_FLASH_MOE_ANE_BATCH` | none | `256` | `DS4_FLASH_MOE_ANE_BATCHES` usually supersedes it | Single fallback ANE batch size. |
| `DS4_FLASH_MOE_ANE_BATCHES` | none | fallback batch only | Profiles use `256` | Comma/colon/semicolon-separated allowed ANE batch sizes. |
| `DS4_FLASH_MOE_ANE_MAX_REFS` | none | current ANE batch size | Profiles use `256` | Maximum routed references per ANE call before chunking/rejection policy applies. |
| `DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS` | none | `0` | Profiles set `1` | Allows large routed expert groups to be split into multiple ANE calls. |
| `DS4_FLASH_MOE_ANE_MIN_REFS` | none | Scheduler fallback `129` via `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS` lookup | Profiles use `32` | Minimum routed references for ANE scheduling. |
| `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS` | none | `129` | Profiles use `64` or `384` depending on machine | Minimum references for hybrid/overlap ANE scheduling; falls back to `DS4_FLASH_MOE_ANE_MIN_REFS`. |
| `DS4_FLASH_MOE_SCHED_ANE_REL_SPEED` | none | `99.0` | Profiles use `99` | Relative ANE speed used by overlap planner. This is a scheduler cost-model knob, not a percentage. |
| `DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL` | none | `0.0` | Profiles use `0.0` | Minimum ANE batch utilization before low-util tails are split away. |
| `DS4_FLASH_MOE_SCHED_ANE_CALL_REFS` | none | `96.0` | Usually unset | Cost-model estimate for ANE call overhead in reference units. |
| `DS4_FLASH_MOE_SCHED_SSD_REFS` | none | `64.0` | Usually unset | Cost-model estimate for SSD staging cost in reference units. |
| `DS4_FLASH_MOE_SCHED_GPU_COVER` | none | `0.90` | Usually unset | How much GPU work can cover ANE cost in the overlap planner. |
| `DS4_FLASH_MOE_ANE_OUTPUT_QUEUE` | none | `1` | Profiles use `4` | Number of queued ANE outputs before host-side pack/writeback pressure stalls scheduling. |
| `DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK` | none | `0` | Profiles set `1` | Uses GPU packing for ANE outputs. |
| `DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK` | none | `0` | Profiles keep `0` | Forces scalar host output pack path; diagnostic. |
| `DS4_FLASH_MOE_ANE_PREFLUSH_EVERY` | none | `1` | Profiles use `4` | Skips some pre-flush command-buffer commits before ANE submissions to reduce commit overhead. |
| `DS4_FLASH_MOE_ANE_SHARED_EXPERT` | none | `0` | Profiles keep `0` for sidecar | Routes shared expert prefill to ANE. Experimental for sidecar. |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ` | none | `0` | M3 Ultra profiles may set `1`; single-cluster profiles generally keep `0` | Routes attention output projection to ANE. Automatically disabled with routed ANE prefill unless force knobs are set. |
| `DS4_FLASH_MOE_MPP_INT8_QSCALE` | none | `512` | Profiles use `512` | Weight quantization scale for int8 prefill paths. Leave at profile default unless recalibrating kernels. |
| `DS4_FLASH_MOE_MPP_INT8_X_QSCALE` | none | `32` | Profiles use `32` | Activation quantization scale for int8 prefill paths. |
| `DS4_FLASH_MOE_MPP_INT8_MID_QSCALE` | none | `32` | Profiles use `32` | Intermediate activation quantization scale for int8 prefill paths. |

## Debug And Measurement Knobs

| Knob | CLI flag | Default | Meaning |
| --- | --- | ---: | --- |
| `DS4_FLASH_MOE_PROFILE` | none | `0` | Enables high-level Flash-MoE timing/profile logs. |
| `DS4_FLASH_MOE_STAGE_STATS` | none | `0` | Enables prefill staging statistics. |
| `DS4_FLASH_MOE_SCHED_STATS` | none | `0` | Enables overlap scheduler statistics. |
| `DS4_FLASH_MOE_SCHED_DEBUG` | none | `0` | More verbose scheduler diagnostics. |
| `DS4_FLASH_MOE_HYBRID_STATS` | none | `0` | Hybrid prefill statistics. |
| `DS4_FLASH_MOE_CONCURRENT_STATS` | none | `0` | Concurrent prefill statistics. |
| `DS4_FLASH_MOE_ANE_PIPELINE_STATS` | none | `0` | ANE pipeline statistics. |
| `DS4_FLASH_MOE_ANE_STATS` | none | `0` | ANE batch/reference statistics. |
| `DS4_FLASH_MOE_ANE_DEBUG` | none | `0` | Verbose ANE routing diagnostics. |
| `DS4_FLASH_MOE_ASYNC_PREAD_DEBUG` | none | `0` | Verbose async pread diagnostics. |
| `DS4_FLASH_MOE_SLOT_BANK_RESIDENCY` | none | `0` | Metal-only diagnostic. Requests a Metal residency set for slot-bank owner buffers after allocation. Use for high-slot decode cliff A/B tests. |
| `DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES` | none | `0` | Metal-only diagnostic. Touches one byte per slot-bank page at startup so page faults happen before decode. Expensive for large banks; use only for cliff diagnosis. |
| `DS4_FLASH_MOE_RESET_SLOT_CACHE_AFTER_PREFILL` | none | `0` | Diagnostic. Clears slot ownership/replay metadata after full or resume prefill, so decode starts with an empty slot cache while keeping the same allocated slot-bank memory. Alias: `DS4_FLASH_MOE_CLEAR_SLOT_CACHE_AFTER_PREFILL`. |
| `DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL` | none | `auto` for high-slot banks | After full/resume prefill or KV payload load, synchronizes, frees the resident slot-bank Metal buffers, recreates them with the same slot count/layout, and resets slot metadata. Alias: `DS4_FLASH_MOE_RECREATE_SLOT_BANK_AFTER_PREFILL`. Unset follows the high-slot policy; explicit `0` disables it, explicit `1` forces it. |
| `DS4_FLASH_MOE_HIGH_SLOT_REALLOC_GB` | `44` | `44` | Slot-bank GPU-cache size threshold, in GiB, for the automatic `DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL` policy. Set `0` to disable only the automatic threshold; explicit `DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL=1` still forces reallocation. |
| `DS4_FLASH_MOE_DECODE_TRACE_OUT` | none | unset | Writes decode routed expert IDs as `pos layer expert...` rows for oracle/predictor A/B tests. |
| `DS4_FLASH_MOE_DECODE_ORACLE_IN` | none | unset | Replays a `DS4_FLASH_MOE_DECODE_TRACE_OUT` file as an exact decode prefetch oracle. Diagnostic only; it is not a production predictor. |

## Practical Starting Points

For a first streaming run:

```sh
./ds4 \
  -m /path/to/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 8 \
  --ctx 8192
```

Then tune only one axis at a time:

- If memory pressure is high, lower `--moe-slot-bank`.
- If decode stalls on SSD reads and memory pressure is low, raise
  `--moe-slot-bank`.
- If prefill does not show a `16384` chunk cap, check `DS4_METAL_PREFILL_CHUNK`
  and `DS4_METAL_GRAPH_RAW_CAP`.
- If A/B testing I/O, start with `DS4_FLASH_MOE_CACHE_IO_SPLIT`,
  `DS4_FLASH_MOE_PREFILL_IO_SPLIT`, `DS4_FLASH_MOE_PREAD_THREADS`, and
  `DS4_FLASH_MOE_ASYNC_READAHEAD`.
- If A/B testing compute, prefer profile values as the baseline and then change
  one `DS4_FLASH_MOE_ANE_*` or scheduler knob at a time.
