# DS4_CTX_GROW — high-water compressed-KV growth

## Problem

The Metal backend used to allocate the compressed KV cache **and** the indexer
scratch to the full `--ctx` ceiling at graph-init time, regardless of how much
context a run actually fills. On a machine where the model is a large fraction of
RAM (e.g. M3 Ultra 96 GiB + the 80 GB IQ2_XXS model = 74.7 GiB tensor data), a
large `--ctx` (say `900000`) reserved **~27 GB of anonymous GPU buffers** even
when the prompt was only a few thousand tokens. That stolen RAM evicted the
file-backed model pages, so every decode token had to page experts back in from
SSD → slow decode or outright `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

The KV itself is tiny (latent attention: 1 KV head + SWA-128 + compressed cache,
ratio 4/128) — only ~1.4 GiB even at 100K context. The eager *allocation to the
ceiling*, not the KV size, was the problem.

## What it does

`--ctx` becomes a **cap**, not an eager reservation. The ctx-scaled buffers start
at one block and grow in fixed context-token blocks toward the ceiling as the
context actually fills.

Two categories of ctx-scaled allocation (in `ds4.c`, Metal path):

| buffer | size at full ctx | kind | growth |
|---|---|---|---|
| per-layer kept compressed KV (`layer_attn_comp_cache`, `layer_index_comp_cache`, sized `layer_comp_cap[il] = ctx/ratio`) | ~12 GB @900K | **kept state** | **copy-on-grow** — `metal_graph_ctx_grow_layer()` allocates the larger buffer, blits the `layer_n_comp[il]` filled rows in a dedicated `begin/end_commands` batch (commit+wait), then frees the old one |
| shared indexer scratch (`indexer_scores`, `comp_mask`, each `comp_cap × prefill_cap`, `comp_cap = ctx/4`) | ~14.7 GB @900K | **transient** (recomputed every step) | **no-copy** — `metal_graph_ctx_grow_shared()` just free+alloc |

Growth algorithm (`metal_graph_ctx_grow_layer` / `_shared`):
- grow in steps of `block/ratio` compressed rows, clamped to the `--ctx` ceiling
  (`layer_comp_cap_max[il]` / `comp_cap_max`);
- only ever called at a **clean command-buffer boundary** (no batch open, prior
  step's GPU work complete), so the copy is safe and the old buffer can be freed;
- kernels pass the **actual** `n_comp`/`n_index_comp` as the row dimension — never
  `comp_cap` as a stride — so resizing the buffers is transparent to the kernels.

### Hook points (`metal_graph_ctx_grow_ensure(g, ctx_pos)`)

Growth fires at clean boundaries before any graph op that writes compressed KV:

- **Prefill** — `metal_graph_prefill_chunked_range` and `metal_graph_prefill_raw_swa`
  grow **once to the prompt length** before the chunk loop (the prompt length is
  known up front). The imatrix calibration prefill is covered too.
- **Decode** — `metal_graph_eval_token_raw_swa` (+ `_top`) grow to `pos+1` before
  opening the command batch; this is a no-op until a block boundary is crossed.
- **Speculative / MTP** — `ds4_session_eval_speculative_argmax` grows once per
  cycle to `checkpoint.len + draft_cap + 1` before the draft+verify, since the
  verify writes the main compressed KV for the accepted suffix.
- **Checkpoint restore** — the GPU snapshot-load path grows to fit the restored
  row counts before validating.

### Coverage across executables

The growth lives entirely in the shared core (`ds4.c`, linked as `CORE_OBJS`), so
it applies to **every** inference executable — `ds4`, `ds4-server`, `ds4-bench`,
`ds4-eval`, `ds4-agent` — because they all drive inference through the same
`ds4_session_sync` / `ds4_session_eval` / `ds4_session_eval_speculative_argmax`
API, which routes through the hooked prefill/decode/speculative functions.
(`ds4-server` and `ds4` CLI use the speculative path by default — that's why the
speculative hook matters.) The CPU backend's KV cache is a separate allocation and
is not yet grown (see follow-ups).

## Env flags & defaults

| flag | default | meaning |
|---|---|---|
| `DS4_CTX_GROW` | **1 (on)** | `0` restores the old eager full-ctx allocation |
| `DS4_CTX_GROW_BLOCK` | **2048** | growth granularity in context tokens (compressed step = `block/ratio`) |

Default-on is safe: when the context fits easily, the caches just grow a step or
two; when `--ctx` is large but underused, almost nothing is reserved.

## Verified results (M3 Ultra 96 GiB, 80 GB IQ2_XXS)

| run | result |
|---|---|
| `--ctx-alloc 900000`, fill 8K, **eager** (`DS4_CTX_GROW=0`) | 26.6 GB context buffers → **OOM on decode token 1** |
| `--ctx-alloc 900000`, fill 8K, **grow** | **19.5 t/s, no OOM**, model resident |
| prefill throughput (either) | **264–277 t/s — unaffected** (chunk/`prefill_cap` and the ANE batch are untouched) |
| tiny-block stress (`DS4_CTX_GROW_BLOCK=2048`, repeated grow+copy) | **0 errors, 20 t/s** |
| user agent check at `--ctx 900000` | coherent output, ~10% faster than full-ctx |

## Block-size tuning (and the M5 Max / bigger-RAM answer)

Block size only sets growth **granularity** — it does not change throughput
materially, because the compressed KV is tiny and attention is sparse (SWA-128 +
indexer top-512). Per growth step it adds roughly (block=2048, `prefill_cap`=8192):

- kept comp KV: `~(block/4)·HEAD_DIM·4·(#ratio-4 layers)` ≈ **~11 MB/step**
- shared scratch: `~2·(block/4)·prefill_cap·4` ≈ **~34 MB/step**
- → **~45 MB per step** at block 2048; ~8× that (~360 MB) at block 16384.

The trade-off:
- **Smaller block** → tighter memory (over-allocates ≤ one block beyond need →
  maximum model-resident headroom) but more realloc+`synchronize()` events.
- **Bigger block** → fewer realloc/sync events (marginally lower overhead) but
  over-allocates up to one block.

Each grow event costs one GPU pipeline flush + a tiny blit; even at block 2048 a
100K fill is only ~48 events total, so the difference is well under 1% either way.

**Recommendation — scale block with available headroom (RAM − model − overhead),
not raw RAM:**

| machine | headroom | recommended block |
|---|---|---|
| **M3 Ultra 96 GiB + 80 GB model** | tight (~10 GiB) | **2048 (default)** — keep over-allocation minimal |
| **M5 Max 128 GiB** (or same model, more RAM) | ample | **8192–16384** is fine — fewer sync events, over-allocation negligible |
| **Bigger-RAM M3 Ultra (256/512 GiB)** | very ample | **16384+**, or just leave it (or even `DS4_CTX_GROW=0`) since the model fits with room |

So: **yes, a bigger `DS4_CTX_GROW_BLOCK` is fine and marginally better on M5 Max /
bigger-RAM machines** — but the gain is small. Block size is primarily a
memory-tightness knob; the 96 GiB M3U is the case that actually needs it small.
(Note: M5 Max is single-cluster, which matters for ANE kernel selection, not for
KV growth.)

## Machine tuning profiles (`ds4_profile.json`)

CTX_GROW is one of several device-dependent knobs. Rather than hardcode them,
every ds4 executable (`ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`)
loads a per-machine **tuning profile** at startup (`ds4_profile.c`, called from
`ds4_engine_open`). A profile is a per-device map of env-var defaults:

```json
{
  "version": 1,
  "profiles": [
    { "match": { "chip": "Apple M3 Ultra", "min_ram_gib": 64 },
      "env": { "DS4_CTX_GROW": "1", "DS4_CTX_GROW_BLOCK": "2048",
               "DS4_RESIDENT_MOE_ANE_MIN_REFS": "64",
               "DS4_RESIDENT_MOE_ANE_MAX_REFS": "1024",
               "DS4_RESIDENT_MOE_ANE_QUEUE": "8" } }
  ]
}
```

- **Match**: first entry whose `chip` (substring of `machdep.cpu.brand_string`)
  and `min_ram_gib` (≤ `hw.memsize`) fit this machine wins. An empty `match: {}`
  is a catch-all; order entries most-specific first.
- **Apply**: each `env` value is set with `setenv(overwrite=0)` — a **default**.
  Anything you export yourself still wins, so one-off experiments are unaffected.
- **Search order**: `$DS4_PROFILE` (set to `none`/`0` to disable) → `./ds4_profile.json`
  → `<dir of executable>/ds4_profile.json` → `~/.config/ds4/ds4_profile.json`.
- **Log**: `ds4: applied tuning profile [<chip>] from <path> (N env defaults set, M kept from environment)`.

### Auto-tuner: `tune_profile.sh`

Find good knobs for the current machine and (optionally) write them into the
profile:

```bash
./tune_profile.sh -m MODEL.gguf -p PROMPT.txt --ctx 8192 [--gen 64] \
                  [--min-refs "32 64 128 256"] [--apply] [--profile ds4_profile.json]
```

It detects the chip + RAM, sweeps `DS4_RESIDENT_MOE_ANE_MIN_REFS` (the routed-ANE
prefill threshold) on the resident path, reports prefill/decode t/s per setting,
picks the best by prefill throughput, chooses a `DS4_CTX_GROW_BLOCK` from RAM
(tight → 2048, ample → 16384), prints the resulting profile entry, and on `--apply`
(or a `y` prompt) merges it into `ds4_profile.json` — replacing any existing entry
for the same chip+RAM and putting it first. The merge is done with `python3` and
round-trips cleanly through the C loader.

## Known follow-ups

- **CPU KV cache** (`s->cpu_cache`, `kv_cache_init`) is still eager ctx-sized
  (~few GB at huge ctx); applying the same growth there reclaims the last bit of
  headroom. This is why decode-phase wired sat ~86 GB rather than ~75 GB at 900K.
- Output was eyeball-verified coherent but not byte-compared against the eager
  path; a divergence-harness pass would make it airtight.
