# Short-prefill (Dedup MoE) optimization — M3 Ultra

Goal: maximize prefill t/s for <500 / 1K / 2K token prompts (target >100 t/s).
Hardware: M3 Ultra, 96 GiB unified, expert sidecar on a PCIe NVMe ("optane",
APFS) at `/Volumes/optane/dsv4-iq2xxs-expert-major` (43 MoE layers × ~1.69 GiB).

## Root cause: prefill is expert-weight-streaming bound

The dedup MoE prefill touches **nearly every expert in every layer regardless of
token count** — at 260 tokens already ~156/256 experts/layer, at 1K ~222/256.
So the bytes streamed from SSD per prefill are roughly constant (45 GiB @256,
64 GiB @1K, ~69 GiB @2K) while the token count varies 8×. That makes short
prefill throughput = bytes / SSD_bandwidth / tokens, i.e. SSD-bandwidth bound.

Measured (`gpu graph prefill total`): GPU *execute* is ~3–6 ms; the multi-second
wall is CPU-side staging — dominated by the expert `pread`s.

optane ceiling (probed): ~7 GB/s cold (sequential OR random, 1 or 8 threads);
~12 GB/s warm scattered pread, ~60 GB/s warm mmap-memcpy. **But the warm page
cache does not survive ds4 startup** (8.4 GiB dense-model mmap + Metal pin the
working set, and the ~73 GiB expert set does not fit warm in 96 GiB), so every
prefill streams cold.

## What worked: parallel pread reader pool (shipped)

The async prefill reader was a **single thread** serializing all expert reads at
~5–6 GB/s. Changes in `ds4.c`:
- `ds4_flash_prefill_async_reader` now spawns a **pool** of reader threads
  (`DS4_FLASH_MOE_PREAD_THREADS`, default 6, max 8). Each atomically claims a
  QUEUED slot (→READING) under the mutex — no double-read.
- Slot count 8 → **24**, plus a decoupled read-ahead depth
  (`DS4_FLASH_MOE_ASYNC_READAHEAD`, default 12) so reads run ahead of the
  in-order 4-bank GPU staging and keep all threads fed.

When the reader is busy it sustains ~7.7 GB/s; the residual gap to the ceiling
is reader-idle waiting on the consumer (encode) and the per-layer router
boundary (layer-major prefill can't read layer L+1 until L's router runs).

## What did NOT work (reverted)

- **mmap + memcpy reads**: cold mmap faults page-by-page (no readahead), far
  slower than pread's large sequential reads. 10–32 t/s.
- **Full PRELOAD / cross-layer `madvise(WILLNEED)`** (even bounded with
  `DONTNEED`): on a 96 GiB box the ~73 GiB expert set can't stay resident, so
  WILLNEED floods the SSD and thrashes the page cache. 9–68 t/s, 45–55 s walls.
  These would only help on a machine with RAM ≥ expert set + model.

## Per-length config (the "bundling / GPU-ANE assignment" knob)

Short prefill wants experts **batched onto ANE** instead of run as many tiny
individual GPU dispatches. Optimal `HYBRID_ANE_MIN_REFS` is inverse to length:

| length | PREAD_THREADS | ASYNC_READAHEAD | HYBRID_ANE_MIN_REFS | ANE_BATCHES |
|--------|---------------|-----------------|---------------------|-------------|
| 256    | 6             | 12              | (read-bound; n/a)   | 256         |
| 1K     | 6             | 12–14           | **8–16**            | **128**     |
| 2K     | 6             | 12              | 8–16                | 128–256     |
| 8K     | 6             | 12              | **384** (default)   | 256         |

## Results (ANE prefill profile, M3 Ultra)

| length | tokens | before (1 reader) | reader pool | + short ANE-tune | >100? |
|--------|--------|-------------------|-------------|------------------|-------|
| 256    | 260    | 24.9              | 32          | 34               | no¹   |
| 1K     | 1098   | 71.8              | 90          | **97**           | ~²    |
| 2K     | 2194   | 122               | 150         | **155**          | yes   |
| 8K     | 8423   | 272               | **297**     | —                | yes   |

¹ 256 tokens streams ~45 GiB; at the 7 GB/s cold SSD ceiling that is ~6.4 s →
  hard cap ~40 t/s. >100 t/s would require ~17 GB/s sustained, i.e. RAM-resident
  experts (needs RAM ≥ expert set + model; infeasible on 96 GiB).
² 1K is at the architectural ceiling (~120 t/s if reads never idled); practical
  ~95–98 t/s, limited by per-layer router-boundary reader idle.

The reader pool is a **universal** win (helps 256→8K) and is now the default in
`run_ane_prefill_profile_m3u.sh`. Revert with `DS4_FLASH_MOE_PREAD_THREADS=1`.
