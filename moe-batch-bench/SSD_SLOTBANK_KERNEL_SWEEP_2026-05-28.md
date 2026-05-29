# SSD slot-bank forced-kernel PREFILL sweep — 32 GB M5

**Hardware:** Apple M5, 32 GB unified memory (single-cluster). Model:
`dsv4-iq2xxs-expert-major` dense (8.2 GB) + expert sidecar, SSD-streamed.
**Raw data:** [`SSD_SLOTBANK_KERNEL_SWEEP_2026-05-28.csv`](SSD_SLOTBANK_KERNEL_SWEEP_2026-05-28.csv)

> **Rerun 2026-05-29 (corrected):** the original run (2026-05-28) requested
> `DS4_METAL_PREFILL_CHUNK=16000` but the auto raw-KV cap was hard-capped at 8192,
> so the *effective* chunk was silently ~8K (the chunk/raw-cap mismatch trap, since
> fixed). These numbers use a **real 16K chunk** (raw_kv_rows=16128) and a **45s
> thermal cooldown** between runs. The 16K/32K numbers changed materially — see
> findings. Sub-16K contexts are unaffected (prompt < one chunk).

Forced-backend prefill comparison of the GPU routed-MoE kernels in **SSD streaming
mode** (`--moe-mode slot-bank`, slot-bank=8) on the 32 GB M5. (Resident kernels need
the 81 GB model and are M5-Max-only; here everything streams expert records from SSD.)

## Backends (all share `DS4_FLASH_MOE_MPP_INT8_PREFILL=1`)

| name | path | extra env |
|---|---|---|
| `gpu_int8` | base MPP h×i8 | (none) |
| `nax_int8` | i8×i8 fused-dequant matmul2d | `MPP_I8I8_PREFILL=1 MPP_I8I8_FUSED_PREFILL=1` |
| `nax_half` | NAX-half, **separate** gate/up + swiglu | `RESIDENT_MOE_NAX_HALF=1` |
| `nax_alu` | **Plan A** fused gate+up+swiglu (NAX∥ALU) | `+ NAX_FUSED_GATE_UP=1 NAX_FUSED_MIN_REFS=0` |

## Methodology

- Cache-isolated cold prefill (throwaway `$HOME`, KV cache wiped per run) so all
  backends prefill the identical token count (system 1274 + user prompt).
- `total_tps` = full cold prefill; `resume_tps` = user-prompt phase.
- `DS4_METAL_PREFILL_CHUNK=16000` (now honored: raw_cap auto-sizes to 16128),
  io-split=4, xlayer prefetch topk=4 (auto), async-pread on, ANE off, 45s cooldown.
- 1K ctx omitted (system prompt = 1274 tok > 1024).

## Results — `total_tps / resume_tps` (t/s), real 16K chunk

| ctx | gpu_int8 | nax_int8 | nax_half | nax_alu (Plan A) |
|---|---|---|---|---|
| 2K  | 61.3 / 52.4 | 63.5 / 53.4 | 59.4 / 51.5 | **64.8 / 54.8** |
| 4K  | 78.0 / 82.8 | **85.0 / 91.7** | 75.6 / 79.3 | 81.1 / 86.7 |
| 6K  | 85.6 / 92.2 | **91.5 / 100.8** | 78.4 / 83.8 | 84.3 / 91.2 |
| 8K  | 85.3 / 90.8 | **95.3 / 104.0** | 82.1 / 86.9 | 91.1 / 98.6 |
| 16K | 67.0 / 67.3 | **97.1 / 101.6** | 72.0 / 72.7 | 91.9 / 95.4 |
| 32K | 62.8 / 62.7 | **79.8 / 80.2** | 64.9 / 65.0 | 72.7 / 73.1 |

## Findings

1. **`nax_int8` is fastest at every ctx ≥ 4K** (Plan A edges it only at 2K). It is
   also the most robust to the larger chunk — held ~97 t/s at 16K while the
   half-activation paths collapsed.

2. **Plan A beats its separate baseline (`nax_half`) by a margin that grows with
   ctx, peaking at 16K:**

   | ctx | Plan A vs nax_half (total) |
   |---|---|
   | 2K | +9.1% |
   | 4K | +7.3% |
   | 6K | +7.5% |
   | 8K | +11.0% |
   | 16K | **+27.6%** |
   | 32K | +12.0% |

   At a real 16K chunk the separate half path (`nax_half`) collapses to 72 while the
   **fused** Plan A holds at 92 — fewer device round-trips (gate/up stay in
   cooperative tiles) make it far more robust to the chunk's bandwidth pressure.
   This is the fusion payoff, and it was hidden when the chunk was silently ~8K.

3. **A real 16K chunk is SLOWER than ~8K at large ctx.** Comparing to the original
   (accidentally-~8K) run, every backend is slower at 16K/32K with the honest 16K
   chunk — gpu_int8 16K: 84→67, nax_half 16K: 83→72, all backends 32K: ~15-25%
   lower. The bigger chunk forces a bigger SWA raw-KV window (16128 rows), so
   attention costs more and outweighs MoE-batching gains. **For SSD prefill at large
   ctx, a smaller chunk (~8K) is the better operating point** — the original numbers
   only looked better because they used a smaller chunk by accident.

## Recommendation

- **Fastest routed kernel: `nax_int8`** across all ctx; also the most chunk-robust.
- **Plan A is the best half-precision path** and the recommended NAX option; its
  fusion advantage is largest exactly where it matters (large chunks).
- **Chunk size: prefer ~8K over 16K at large ctx** — the SWA-window cost of a 16K
  chunk outweighs MoE batching here. Worth a dedicated chunk-size sweep
  (4K/6K/8K/12K/16K) to pin the crossover.
- ALU / Path-C remains deferred (counted-indirect; needs id-map/counts machinery the
  per-expert banked path lacks).

## Correctness caveat

This SSD prefill path is non-deterministic run-to-run for every backend (f16
accumulation under concurrent GPU work — verified on the untouched `nax_int8`), so
Plan A is coherent and probe-validated but not bit-verifiable e2e. A true numeric
check needs a logit/intermediate-buffer harness.
