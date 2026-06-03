# Profiling a new Mac and baking it into ds4_profile.json

This is the reusable workflow for measuring routed-MoE **prefill** on a Mac and
writing the winning config into `ds4_profile.json`, for **both** deployment
regimes. Worked examples already in-tree:

- `moe-batch-bench/PREFILL_PROFILE_M5MAX.md` — M5 Max (single ANE cluster)
- `moe-batch-bench/PREFILL_PROFILE_M1MAX.md` — M1 Max (single ANE cluster, no NAX; ANE loses to GPU/ALU → streaming stays GPU)
- `moe-batch-bench/SHORT_PREFILL_OPTIMIZATION.md` — M3 Ultra (dual ANE cluster)

## 0. The two regimes (this is the whole point)

A machine runs the model one of two ways, and they behave **oppositely**, so a
correct profile entry carries config for both:

| regime | when | the 81 G model is | profile key |
|---|---|---|---|
| **resident** | RAM ≥ 96 GB | fully wired (`--moe-mode off`) | `env` + `prefill_by_tokens` |
| **streaming** | any (required if RAM < 96 GB) | dense resident + experts streamed from SSD sidecar (`--moe-mode slot-bank`) | `sidecar_env` |

**RAM threshold rule: if RAM < 96 GB, make a STREAMING-ONLY entry.** The 81 G
IQ2_XXS model needs ~96 GB to be resident, so a `< 96 GB` machine can only stream.
Give it an entry with **only `sidecar_env`** (no `env` / `prefill_by_tokens`) —
those resident keys would never apply anyway. Machines with RAM ≥ 96 GB get a
both-modes entry (resident `env`+`prefill_by_tokens` **and** streaming `sidecar_env`).
Order entries most-specific first so the `≥128`, `≥96`, then `<96` (e.g. `48`)
`min_ram_gib` buckets each catch the right machine.

The loader picks `sidecar_env` vs `env`+`prefill_by_tokens` automatically from
`opt->moe_mode == SLOT_BANK && opt->moe_sidecar_path` (`ds4.c`,
`ds4_profile_set_sidecar_mode`). One JSON entry per `{chip, min_ram_gib}` can hold
both — see any M5 Max / M3 Ultra entry.

Rule of thumb that keeps recurring:
- **streaming favors ANE** — routed compute overlaps the SSD pread, so shedding
  it to the ANE frees the GPU. Win is biggest mid-chunk, fades as chunk grows.
- **resident favors GPU/NAX-int8** at the chunk sizes that matter (≥8 K) — no
  I/O bubble to hide ANE behind, and NAX-int8 W8A8 matmul2d is very fast at large
  batch. ANE only wins tiny chunks, which you usually **cannot** exploit (see §6).

## 1. Know your silicon

| chip | NAX (matmul2d)? | ANE clusters | GPU baseline to compare ANE against |
|---|---|---|---|
| M5 / M5 Pro / M5 Max | yes (M5+) | 1 | **nax_int8** (W8A8) |
| M3 Ultra | no (falls back to mul_mm_id) | 2 (UltraFused) | **mulmm** (ALU) |
| M4 / M3 / M2 | no | 1 | **mulmm** (ALU) |

- NAX backends (`nax_int8`, `nax_half`, `nax_half_alu`) are **M5+ only**; on
  older chips they fall back to `mul_mm_id`, so the GPU baseline there is `mulmm`.
- Dual-cluster (M3 Ultra) ANE wins are much larger (+30–40%) than single-cluster
  (M5, +5–15%) because aggregate ANE throughput ~doubles. Set
  `DS4_FLASH_MOE_ANE_DUAL=1 DS4_FLASH_MOE_ANE_THREADS=2` on dual; `=0`/`=1` on
  single, and turn shared-expert + O-proj ANE **off** on single (they lose).

## 2. Make per-route profiler scripts for your chip

Copy the closest pair and adjust the head constants + ANE cluster knobs:

```bash
cp run_ane_prefill_profile_m5max.sh run_ane_prefill_profile_<chip>.sh
cp run_gpu_prefill_profile_m5max.sh run_gpu_prefill_profile_<chip>.sh
```

Edit in each:
- `DS4_SSD_MODEL_DEFAULT` (dense GGUF), `DS4_RESIDENT_MODEL` (full GGUF),
  `DS4_SIDECAR` (expert sidecar dir) — your local paths.
- `DS4_LOCK_FILE` — a unique lock so it never runs alongside another ds4.
- run-name prefix, `DS4_BIN` (`./ds4`).
- **ANE cluster knobs** (ANE script): `DS4_FLASH_MOE_ANE_DUAL`,
  `_THREADS`, `_SHARED_EXPERT`, `_OUTPUT_PROJ` per §1.
- **GPU route** (GPU script): on M5+ default `nax_int8`; on M3/M4 use `--alu`
  (`mulmm`) as the only real GPU path.

Both scripts set `DS4_PROFILE=none` so the profile can't confound the sweep, emit
rich per-layer stats, and write a `.summary.txt`.

## 3. Run the sweep (the harness does GPU baseline + ANE min_refs inner sweep)

```bash
cp moe-batch-bench/run_prefill_chunk_minrefs_sweep_m5max.sh \
   moe-batch-bench/run_prefill_chunk_minrefs_sweep_<chip>.sh
# point it at your run_{ane,gpu}_prefill_profile_<chip>.sh and PROMPT_DIR
```

Two segments per regime (same as the M3U/M5 docs):

```bash
# Segment A — fixed chunk 16K, prompt sweep. Include 10k/12k/14k to resolve the
# crossover in the 8K-16K band; SKIP 32K (slow, and the >=8K regime is settled).
PROMPTS="128 256 1k 4k 8k 10k 12k 14k 16k" CHUNKS="16384" MIN_REFS="32 64 128 256 512" \
  DS4_SLOTS=96 ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_<chip>.sh

# Segment B — 8K prompt, chunk sweep
PROMPTS="8k" CHUNKS="1024 2048 4096 8192 16384" MIN_REFS="32 64 128 256 512" \
  DS4_SLOTS=96 ./moe-batch-bench/run_prefill_chunk_minrefs_sweep_<chip>.sh

# resident: prefix either with DS4_RUN_MODE=resident
```

Prompt grid note: `10k`/`12k`/`14k` fill the 8K-16K gap where the resident
crossover lives (ANE→GPU flip). `coding_14k.txt` may need creating (truncate
`coding_16k.txt`). Do **not** add `32k` — it's slow and adds nothing once the
≥8K trend is established.

Run each **alone on the box** (it auto-aborts on GPU contention). Hot single runs
are fine for the trend; add `COOLDOWN=45` between runs and `REPEATS=3` if you need
tight numbers. Results land in `moe-batch-bench/profile_runs/<id>/results.csv`.

## 4. Read the results

The harness prints best-per-cell. Per `{prompt or chunk}` you want: GPU t/s,
best-ANE t/s, and the `min_refs` that produced it.

- **ANE win %** = `(bestANE − GPU)/GPU`. Positive → ANE; negative/zero → GPU.
- **best `min_refs`**: streaming usually wants **low (32–64)**; resident wants
  **high (256–512)**. "384 too conservative" reproduces on every single-cluster
  chip we've measured — do not default streaming to 384.
- Find the **crossover** chunk/prompt where the winner flips.

## 5. Validate the resident crossover trap BEFORE multi-segment

If resident shows ANE winning small chunks, you'll be tempted by a
`prefill_by_tokens` crossover (`<N → ane_gpu, ≥N → nax_int8`). **Test it first** —
any `ane*` segment makes the loader call `apply_resident_ane_prefill_defaults()`,
which sets `ANE_HYBRID=1` globally → `compact_scratch_requested()` caps the
resident tile to 2048 → sync bridge on the **large nax_int8 chunk too**.

```bash
cp moe-batch-bench/test_resident_crossover_trap_m5max.sh \
   moe-batch-bench/test_resident_crossover_trap_<chip>.sh   # fix MODEL/PROMPT
./moe-batch-bench/test_resident_crossover_trap_<chip>.sh
```

- **crossover ≈ control** → trap doesn't bite → use the multi-segment table.
- **crossover ≪ control** → keep single-segment `nax_int8`. (M5 Max measured
  185 vs 494 t/s = **−63%** → single-segment.) Dual-cluster M3U survives the
  tile and does use crossovers.

## 6. Bake into ds4_profile.json

Add/replace the entry for your `{chip, min_ram_gib}` (most-specific first; the
first match wins). Two halves:

```jsonc
{
  "_comment": "what/when/why + measured t/s + cooldown notes",
  "match": { "chip": "Apple <Chip>", "min_ram_gib": <N> },

  // RESIDENT
  "env": { /* DS4_RESIDENT_MOE_MPP_* + DS4_METAL_PREFILL_CHUNK + DS4_GPU_DENSE_NAX */ },
  "prefill_by_tokens": [ { "min_tokens": 0, "max_tokens": 99999, "backend": "nax_int8" } ],

  // STREAMING (omit if this machine can't fit resident / is streaming-only)
  "_sidecar_comment": "...",
  "sidecar_env": { /* DS4_FLASH_MOE_ANE_* (or MPP int8 if GPU won) + I/O knobs */ }
}
```

- Resident `env` for the **nax_int8** path: `DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1
  MPP_FORCE=1 MPP_COMPACT_BRIDGE=1 MPP_FUSED_DEQUANT=1 MPP_MIN_TOKENS=64
  MPP_COMPACT_MIN_TOKENS=64`, `DS4_METAL_PREFILL_CHUNK=16384`, `DS4_GPU_DENSE_NAX=1`.
  (`MPP_FORCE` only opens the tile-util gate; it does NOT force the backend.)
- `prefill_by_tokens`: per-chunk backend by token count. Valid: `mulmm`,
  `nax_half`, `nax_half_alu`, `nax_int8`, `ane_gpu` (see `_backends_reference`).
  `nax_half` needs `n_tokens % 64 == 0` or it silently demotes to int8.
- `sidecar_env` for the **ANE streaming** path: mirror the validated
  `run_ane_ssd_agent_<chip>.sh` — `DS4_FLASH_MOE_ANE_PREFILL=1` + `_PIPELINE` +
  `_I8I8` + `_I8I8_TILED_FUSED` + `_OVERLAP_*`, `MPP_INT8_PREFILL=0`, cluster
  knobs, `HYBRID_ANE_MIN_REFS=<your best>`, I/O (`PREFETCH=3 ASYNC_PREAD=1
  ASYNC_PREAD_AFTER_STAGE=1`). If GPU won streaming instead, set
  `MPP_INT8_PREFILL=1` and ANE off — mirror `run_nax_ssd_agent_<chip>.sh`.
- Never set `DS4_RESIDENT_MOE_BACKEND` in a profile — it has higher precedence
  and bypasses the per-chunk table.

Validate JSON after editing:
```bash
python3 -c "import json; json.load(open('ds4_profile.json'))" && echo OK
```

## 7. End-to-end validation (profile applied, not the equivalent flags)

Confirm the profile *path* engages the right backend with NO env flags set:

```bash
# streaming: expect "applied sidecar tuning profile" + "routed experts = ANE i8i8"
./ds4-bench -m <dense.gguf> --moe-sidecar <dir> --moe-mode slot-bank \
  --moe-slot-bank <N> --metal --prompt-file <16k+ prompt> \
  --ctx-start 16384 --ctx-max 16384 --gen-tokens 1

# resident: expect "applied tuning profile" + "int8 matmul2d (compact-bridge fused-dequant)"
./ds4-bench -m <full.gguf> --moe-mode off --metal --prompt-file <16k prompt> \
  --ctx-start 16384 --ctx-max 16384 --gen-tokens 1
```
Look for `applied [sidecar ]tuning profile [<chip>]` and the expected
`prefill compute: routed experts = …`. If streaming shows GPU instead of ANE, the
profile table is overriding — clear it or check the `sidecar_env` keys applied
(the "N env defaults set" count).

## Traps that cost real time (all hit during the M5 work)

- **Global 2048 scratch tile (§5)** — any resident `ane*` segment forces
  `ANE_HYBRID` globally and tanks the large nax_int8 chunk. Test before trusting.
- **`tune_profile.sh` auto-cap** — a bare `./ds4` with no
  `DS4_METAL_PREFILL_CHUNK` auto-caps the chunk to 4096 when prompt>4096; the
  resident-NAX tile then takes the slow sync bridge (`map_wait`). Always set the
  chunk explicitly; verify the `resident-NAX tile … sync-bridge=off` startup line.
- **`nax_half` %64 demotion** — non-multiple-of-64 chunks silently run int8.
- **GPU baseline mismatch** — on M5+ compare ANE against **nax_int8**, not ALU
  (`mulmm`); ALU is a much weaker baseline and inflates ANE's apparent win.
- **Streaming I/O flags missing** — a GPU streaming baseline without
  `PREFETCH`/`ASYNC_PREAD` stalls on SSD and isn't a fair comparison.
- **Thermals + contention** — M-series prefill is sensitive; one concurrent GPU
  user or no cooldown skews single runs. Run alone; add `COOLDOWN`/`REPEATS` for
  publishable numbers.

## Optional: let tune_profile.sh bake the winner

`tune_profile.sh` sweeps and writes the winning env into `ds4_profile.json` for
the detected machine. Use the manual flow above when you need per-chunk crossover
tables or both-regime (`sidecar_env`) entries it doesn't yet generate.
