# Resident Mode Decode Stutter Handoff

Date: 2026-05-31

Target checkout:

```text
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
```

This note is for the resident/full-model path, not the Flash-MoE sidecar
slot-bank path. Slot-bank findings are included only where they help avoid
mixing up symptoms.

---

## RESOLVED 2026-05-31 — root cause + fix (read this first)

Hardware/model under test: M3 Ultra, 96 GiB RAM, 80 GB IQ2_XXS GGUF
(76.5 GiB tensor data, 43 layers, 256 experts, 6 routed + 1 shared/token).

### Root cause (measured, not hypothesised)

The decode speed itself is NOT a regression: at 8K context both this checkout
and the known-good `ds4-ssd` decode the resident model at ~22 t/s / ~40 ms-per-
token GPU execute (identical). The handoff's old "110 t/s" comparison was a
different (slot-bank) mode, not the resident path.

The real defect is a memory-residency cliff in how decode wraps the mmap'd
weights:

- At the prefill→decode boundary the resident path switched the model views to
  **persistent decode mode**: the whole 76.5 GiB tensor range wrapped as two
  giant no-copy `MTLBuffer`s, kept GPU-resident. A decode command buffer that
  references those buffers needs their entire backing resident at once.
- At 8K that *barely* fit (wired peaked at ~86/96 GiB) and decoded smoothly.
- At 16K the larger KV cache (2.1 GiB) tipped it over: the **first decode token
  failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`**. That hard failure
  (plus the near-cliff memory-pressure oscillation observed as wired bouncing
  8→87→10 GiB while macOS reclaims/refaults) is the "stutter/freeze + swap" the
  user saw.
- Forcing `DS4_METAL_NO_RESIDENCY=1` did **not** help (still OOM'd at 16K).
- Lazy model views (`DS4_METAL_DECODE_PERSISTENT_VIEWS=0`) avoided the OOM —
  many small per-layer buffers page granularly instead of all-or-nothing — but
  the lazy wrap cache was cleared every command batch, so each decode token
  re-created ~130 no-copy buffers and the GPU re-validated their VM ranges:
  ~1.1 s/token (0.85 t/s) **even though the pages were already RAM-resident**
  (confirmed: inactive 56.7 GiB + wired 31.8 GiB ≈ the model). So that 1.1 s was
  residency/VM re-validation, not SSD I/O.

### Fix (in `ds4_metal.m`)

Persist the lazy model-wrap cache across decode command batches, and auto-select
lazy decode views for models that are a large fraction of RAM:

- New `g_model_cache_persist`: when set, `ds4_gpu_transient_resources_clear()`
  keeps `g_model_buffer_cache` (the no-copy wraps) instead of dropping it, so a
  decode step reuses last token's buffers and their established GPU residency.
  The cache is reset on any model-view mode change / remap
  (`ds4_gpu_model_buffer_cache_clear`, prefill prep, `ds4_gpu_model_views_clear`).
- `ds4_gpu_decode_persistent_views_requested(map_size)` now auto-returns lazy
  when `map_size > 5/8 of system RAM` (env `DS4_METAL_DECODE_PERSISTENT_VIEWS`
  still overrides). Small models that comfortably fit keep fast persistent views.

Result: granular per-layer buffers (no OOM cliff) + persisted residency (low
per-token cost).

### Verified results (ds4-bench, --gen-tokens 256, DS4_METAL_GRAPH_TOKEN_PROFILE)

| context | persistent (old default) | naive lazy | **fix (new default)** |
|---|---|---|---|
| 8K  | 22.4 t/s (smooth)        | 0.87 t/s   | **20.9 t/s, median 44 ms/token** |
| 16K | **OOM on token 1**       | 0.85 t/s   | **14.6 t/s, median 57 ms/token, no OOM** |

Only the first decode token is slow now (cache warmup: ~0.9 s at 8K, ~2.4 s at
16K); every subsequent token is uniform. Known follow-up: pre-warm the wrap
cache during the prefill→decode transition to hide that first-token pause, and
16K is still memory-pressured (working set churns near the cliff) so it runs
slower than 8K — but it no longer crashes.

Reproduce / regress-check:

```bash
P=/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_22k.txt
M=/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf
DS4_METAL_GRAPH_TOKEN_PROFILE=1 DS4_METAL_PREFILL_CHUNK=16384 DS4_METAL_GRAPH_RAW_CAP=16640 \
./ds4-bench -m "$M" --prompt-file "$P" --metal --moe-mode off --resident-ane-prefill \
  --no-decode-split --ctx-start 16384 --ctx-max 16384 --gen-tokens 256
# expect: completes 256 tokens, no kIOGPUCommandBufferCallbackErrorOutOfMemory
# analysis helper: profile_runs/analyze_tokens.py <stderr-log>
```

Everything below is the original investigation context, kept for history.

## Problem

Resident ANE prefill now looks materially better than the initial port, but
generation still has intermittent visible stutter around long-context resident
runs. The symptom is not just low average tokens/sec. It appears as periodic
decode stalls or freezes while memory pressure changes, and in one run macOS
showed swap/page activity.

The original visible bad comparison was:

```text
ds4-ssd-anemll: ctx 2.1k/20k prefill 398/779 51.1% 11.1 t/s
ds4-ssd:        ctx 2.3k/20k prefill 498/893 55.8% 110.4 t/s
```

After the prefill fixes, prefill was improved, but decode still stuttered. The
current open issue is decode smoothness, not the basic ability to prefill.

## Primary Repro

Use the full resident GGUF. This is the original stutter-prone shape because it
forces no full-model residency while using resident ANE prefill:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll

M=/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf

DS4_LOCK_FILE=/tmp/ds4-agent-ane-prefill.lock \
DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_PREFILL_CHUNK=8192 \
DS4_METAL_GRAPH_RAW_CAP=8704 \
DS4_RESIDENT_MOE_ANE_MIN_REFS=512 \
DS4_RESIDENT_MOE_ANE_MAX_REFS=1024 \
DS4_RESIDENT_MOE_ANE_QUEUE=8 \
./ds4-agent \
  -m "$M" \
  --metal --moe-mode off \
  --resident-ane-prefill \
  --no-decode-split \
  --ctx 20000
```

Repro steps:

1. Start the agent with the command above.
2. Use a normal multi-step prompt that produces a long answer.
3. Watch the terminal during generation, not only prefill.
4. In Activity Monitor or `asitop`, watch memory pressure, wired memory, and swap.

Expected bad symptom:

```text
generation pauses or visibly stutters despite prefill completing
memory graph shows dips/spikes; swap/page-in can appear
```

## Current Recommended Resident Run

This is the safer run shape after the latest fixes. The important difference is
that it does not force `DS4_METAL_NO_RESIDENCY=1`, and it lets
`--resident-ane-prefill` keep decode residency and prefill scratch release on by
default.

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll

M=/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf

DS4_LOCK_FILE=/tmp/ds4-agent-ane-prefill.lock \
DS4_METAL_PREFILL_CHUNK=8k \
DS4_METAL_GRAPH_RAW_CAP=8704 \
DS4_RESIDENT_MOE_ANE_MIN_REFS=64 \
DS4_RESIDENT_MOE_ANE_MAX_REFS=1024 \
DS4_RESIDENT_MOE_ANE_QUEUE=8 \
./ds4-agent \
  -m "$M" \
  --metal --moe-mode off \
  --resident-ane-prefill \
  --no-decode-split \
  --ctx 20000
```

`--resident-ane-prefill` currently sets these important defaults in
`ds4_agent.c`:

```text
DS4_RESIDENT_MOE_ANE_HYBRID=1
DS4_RESIDENT_MOE_ANE_HYBRID_OUTER=0
DS4_METAL_DECODE_RESIDENCY=1
DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE=1
DS4_RESIDENT_MOE_ANE_MIN_REFS=64
DS4_RESIDENT_MOE_ANE_MAX_REFS=1024
DS4_RESIDENT_MOE_ANE_QUEUE=8
```

## Diagnostic Runs

Use these only for short runs because they can add overhead and backend output.

To confirm prefill scratch is released before decode:

```bash
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_METAL_RELEASE_PREFILL_SCRATCH_TRACE=1 \
DS4_METAL_RELEASE_PREFILL_BACKEND_TRACE=1 \
...same resident command...
```

To profile decode stage timing around the stutter:

```bash
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_METAL_DECODE_STAGE_PROFILE=1 \
DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
...same resident command...
```

To A/B the old no-residency behavior against current decode residency:

```bash
# Bad/stress shape:
DS4_METAL_NO_RESIDENCY=1 ...same command...

# Current intended shape:
unset DS4_METAL_NO_RESIDENCY
DS4_METAL_DECODE_RESIDENCY=1 ...same command...
```

Optional dense residency experiment:

```bash
DS4_METAL_RESIDENT_DENSE=1 ...same command...
```

Notes:

- Resident dense cache is skipped by default when persistent decode model views
  are enabled, because the combination can be high-memory.
- To force that high-memory experiment, use:

```bash
DS4_METAL_RESIDENT_DENSE=1 \
DS4_METAL_RESIDENT_DENSE_FORCE=1 \
...same command...
```

Only use force mode as an A/B test; do not treat it as the default answer.

## What Was Tried

Resident prefill fixes reimplemented from the known-good DS4-SSD direction:

- IQ2_XXS down handling for resident routed ANE paths.
- Correct grouped GPU tail using an ANE skip mask instead of disabling grouped
  tail entirely.
- Prefill scratch release and re-ensure plumbing so decode does not carry the
  largest prefill scratch allocations.
- Prefill stage flush points for compact scratch on large chunks.
- Additional Metal skip-mask and ANE/predequant support in `ds4_metal.m` and
  `metal/moe.metal`.
- Backend debug spam suppression so interactive prefill does not corrupt the
  agent prompt.
- Numeric suffix parsing for env/CLI values such as `8k`, `16k`, and `200k`.
- Metal decode-replay descriptor cache was added for routed decode. This is
  useful for parity with the llama.cpp concept, but it is expected to affect CPU
  encode overhead, not SSD page-in or memory-pressure stalls.

Observed result:

- Prefill is no longer the original 10x regression shape.
- Decode still needs A/B work. The stutter persisted after the prefill path
  looked better.
- `DS4_METAL_NO_RESIDENCY=1` remains suspicious because it forces mmap-backed
  model view behavior during decode.

## Current Hypotheses

Most likely:

```text
decode stutter is memory/page-in pressure after prefill, not ANE eval time
```

Reasoning:

- The bad repro explicitly sets `DS4_METAL_NO_RESIDENCY=1`.
- The model is mmap-backed when full residency is disabled.
- Large prefill chunks plus long context can leave the system near the memory
  pressure cliff if scratch or model pages are not released/resident at the
  right boundary.
- User observed memory graph changes and swap during the bad decode run.

Less likely but still possible:

- A hidden command-buffer sync or stage flush still fires in decode.
- Dense/shared model views are being rewrapped or faulted too aggressively.
- The adjacent `ds4-ssd` checkout has a newer decode-residency fix that is not
  fully reimplemented here.

Probably not the main resident issue:

- Slot-bank size. `--moe-slot-bank 164` caused a separate decode collapse in
  sidecar mode because it allocated about 47.6 GiB of GPU bank. That is a
  sidecar/slot-bank memory cliff, not the same as resident full-GGUF decode.

## Next Tests

Run these in order and record actual generation smoothness plus average t/s:

1. Resident command with `DS4_METAL_NO_RESIDENCY=1`.
2. Same command with `DS4_METAL_NO_RESIDENCY` unset.
3. Same command with `DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE=0`.
4. Same command with `DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE=1`.
5. Optional: `DS4_METAL_RESIDENT_DENSE=1`.
6. Optional high-memory force: `DS4_METAL_RESIDENT_DENSE=1 DS4_METAL_RESIDENT_DENSE_FORCE=1`.
7. Short decode-stage profile only after reproducing the stutter without stats.

Minimum report for each run:

```text
command/env delta:
prompt/context at generation start:
prefill t/s:
generation t/s:
visible stutter: yes/no
swap observed: yes/no
memory peak:
notable backend log lines:
```

## Build Checks

Use this after any fix:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
git diff --check -- ds4.c ds4_agent.c ds4_bench.c ds4_gpu.h ds4_metal.m metal/moe.metal
make ds4 ds4-agent ds4-bench -j8
```

Do not stage generated binaries:

```text
ds4
ds4-agent
ds4-bench
*.log
*.csv
profile_runs*
```

