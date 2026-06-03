# M5 Max routed-MoE prefill profile — ANE vs GPU/NAX-int8

M5 Max is **single-cluster ANE** (M3 Ultra is dual). Two deployment regimes,
which behave oppositely and so need different profiles:

- **128 GB → resident** (full 81 G IQ2_XXS GGUF wired, `--moe-mode off`).
- **64 GB → streaming** (dense GGUF + expert sidecar, `--moe-mode slot-bank`;
  the 81 G model can't be resident in 64 G). No resident path on 64 G.

GPU baseline here is **NAX-int8 (W8A8 matmul2d)** — the real M5 competitor — NOT
the plain ALU `mul_mm_id` baseline the M3U sweep used. So M5 ANE deltas are
against a *stronger* baseline than the M3U doc's numbers.

## Scripts

```bash
# per-route single profile (rich stats)
run_ane_prefill_profile_m5max.sh   # routed -> ANE (single cluster: DUAL=0 THREADS=1, shared/oproj ANE off)
run_gpu_prefill_profile_m5max.sh   # routed -> NAX-int8 (default) | --alu for plain mul_mm_id

# the sweep harness (GPU baseline + ANE min_refs inner sweep)
moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5max.sh
#   streaming (default):  ./run_prefill_chunk_minrefs_sweep_m5max.sh
#   resident:             DS4_RUN_MODE=resident ... ./run_prefill_chunk_minrefs_sweep_m5max.sh
```

Common: `DS4_PROFILE=none` (so the JSON profile can't confound), single gen
token, `DS4_FLASH_MOE_ANE_SHARED_EXPERT=0 DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0`
(route-only ANE), slots=96, prompts from the `flashmoe-sidecar/prompts/coding`
set. Hot single runs (no inter-run cooldown) — trends are robust, exact % ±a few.

## Streaming (slot-bank, slots=96, ~40 GB RSS)

**Segment A — fixed chunk 16K, prompt sweep:** GPU=NAX-int8.

| prompt | GPU | best ANE | min_refs | ANE delta |
|-------:|----:|---------:|---------:|----------:|
| 128 | 46.50 | 48.55 | 512 | +4.4% (noise) |
| 256 | 79.74 | 75.63 | 32 | no win |
| 1K | 172.81 | 184.48 | 32 | +6.8% |
| 4K | 291.51 | 334.60 | 64 | **+14.8%** |
| 8K | 363.58 | 380.54 | 32 | +4.7% |
| 16K | 336.35 | 353.69 | 256 | +5.2% |

**Segment B — 8K prompt, chunk sweep:**

| chunk | GPU | best ANE | min_refs | ANE delta |
|------:|----:|---------:|---------:|----------:|
| 1024 | 146.46 | 167.13 | 64 | +14.1% |
| 2048 | 204.29 | 242.41 | 32 | **+18.7%** |
| 4096 | 263.56 | 305.97 | 64 | +16.1% |
| 8192 | 330.24 | 345.43 | 64 | +4.6% |
| 16384 | 373.61 | 384.88 | 64 | +3.0% |

Streaming verdict: **ANE wins almost everywhere** (except tiny prompts); biggest
in the mid-range (4K prompt +14.8%, 2K chunk +18.7%) and shrinks as chunk grows
(16K chunk only +3% — the SSD I/O bubble ANE hides behind closes up). Best
`min_refs` is firmly **32–64**.

## Resident (full 81 G GGUF, `--moe-mode off`)

**Segment A — fixed chunk 16K, prompt sweep:** GPU=NAX-int8.

| prompt | GPU NAX-int8 | best ANE | min_refs | winner |
|-------:|----:|---------:|---------:|:------|
| 128 | 38.62 | 94.65 | 32 | ANE +145% |
| 256 | 56.83 | 130.15 | 64 | ANE +129% |
| 1K | 154.95 | 250.26 | 256 | ANE +62% |
| 4K | 357.73 | 367.50 | 512 | tie (+3%) |
| 8K | **633.07** | 389.34 | 512 | **GPU +63%** |
| 16K | **512.82** | 366.32 | 512 | **GPU +40%** |

(1K spread across min_refs: 196/228/245/250/230 for 32/64/128/256/512 — clean
curve; an earlier 54 t/s smoke was a cold first-load outlier, discarded.)

**Segment B — 8K prompt, chunk sweep:** GPU=NAX-int8.

| chunk | GPU NAX-int8 | best ANE | min_refs | winner |
|------:|----:|---------:|---------:|:------|
| 1024 | 135.57 | 226.75 | 128 | ANE +67% |
| 2048 | 210.53 | 288.80 | 256 | ANE +37% |
| 4096 | 291.26 | 335.96 | 512 | ANE +15% |
| 8192 | **464.55** | 363.31 | 512 | **GPU +28%** |
| 16384 | **630.77** | 385.52 | 512 | **GPU +64%** |

Same crossover seen by **chunk size**: ANE wins <8K chunk, NAX-int8 wins ≥8K
chunk. Since the agent prefills 16384-token chunks, resident lands firmly in the
NAX-int8 regime.

Resident verdict: a **prompt/chunk-size crossover ≈ 4–8K**. ANE wins short prompts
(<4K, up to +145%); NAX-int8 wins large (≥8K, up to +63%) because resident
NAX-int8 is slow at tiny batch (39–155) but explodes at large chunk (633) while
ANE is flat. `min_refs` **inverts vs streaming** — resident prefers **256–512**.

## Cross-mode summary

| | streaming (64 GB) | resident (128 GB) |
|---|---|---|
| ANE wins | broadly; best mid-range | only short chunks (<4K) |
| NAX-int8 wins | only very large chunk | large chunks (≥8K) |
| best min_refs | 32–64 (low) | 256–512 (high) |
| why | ANE hides behind SSD pread | no I/O bubble; NAX-int8 huge at large batch |

vs M3 Ultra: same *shape* but smaller magnitude (single vs dual ANE cluster:
M3U routed-ANE gains were +37–38% where M5 is +5–15%), and M5's GPU baseline is
NAX-int8 not ALU, so M5 ANE deltas are against a tougher baseline.

## Profile decisions (ds4_profile.json)

- **M5 Max 128 GB / 96 GB (resident): stays single-segment `nax_int8`.** The
  resident <8K-chunk ANE win is REAL per-chunk but **NOT exploitable** via the
  table. Putting `ane_gpu` in `prefill_by_tokens` auto-enables
  `DS4_RESIDENT_MOE_ANE_HYBRID`, and `compact_scratch_requested()` (which
  includes ANE_HYBRID) hard-caps the resident tile to 2048 **globally** → the
  sync bridge turns ON for the `nax_int8` ≥8K chunks too.
  **Measured directly (2026-06-02, `test_resident_crossover_trap_m5max.sh`, same
  17,329-tok prompt):** crossover `8191:ane_gpu,99999:nax_int8` = **185 t/s** vs
  pure `nax_int8` = **494 t/s** → **−63%**. So a crossover loses far more on the
  dominant ≥8K chunk than it gains on small ones. Capturing the small-chunk ANE
  win would require a per-chunk scratch tile (engine change), not a profile
  change. (M3U can afford crossovers because dual-cluster ANE wins even under the
  2048 tile.)
- **M5 Max 64 GB (streaming):** ANE-enabled streaming config (routed→ANE, single
  cluster). Streaming uses the `DS4_FLASH_MOE_*` path and the slot-bank — NOT the
  resident scratch-cap/sync-bridge — so ANE here has no global-tile penalty and
  the streaming wins above are real. _Entry to be finalized from the slots=99
  streaming sweep (in progress)._
