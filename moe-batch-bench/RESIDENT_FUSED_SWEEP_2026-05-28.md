# Resident-MPP fused sweep — 2026-05-28
_M5 Max · DeepSeek-V4-Flash IQ2XXS · 60s cooldowns between runs · single agent_

## Prefill t/s vs ctx

| ctx   | NAX-int8 baseline | Path C (i8-fused) | Plan A (h_h_f-fused, MIN_REFS=0) | Plan A vs baseline |
|------:|------------------:|------------------:|---------------------------------:|-------------------:|
|  2048 |             286.0 |             204.2 |                        **349.6** | **+22.2%** |
|  4096 |             383.7 |             304.6 |                        **454.6** | **+18.5%** |
|  8192 |             527.9 |             520.4 |                        **569.4** | **+7.9%** |
| 16384 |             547.0 |             542.2 |                        **573.1** | **+4.8%** |
| 32768 |                 — |                 — |                                — | (all failed — investigate) |

## Findings

- **Plan A wins at every ctx**. Biggest relative gain at small ctx (per-expert M is
  smallest there, so saved-dispatch dominates). Smallest relative gain at 16K
  (matmul throughput dominates there).
- **Path C is a regression at small/medium ctx** (−28% at 2K, −20% at 4K). Only
  catches up to baseline at 16K (still −1% behind). Verdict reaffirmed: Path C
  should stay default-off; not useful in production on M5.
- **Plan A absolute peak so far**: 573.1 t/s @ 16K, +4.8% over the published
  ~547 ceiling — meaningfully past the 532 nominal.

## Open

- 32K runs all failed — likely prompt-length issue (149K chars ≈ 37K tokens but
  ds4-bench wants enough headroom past --ctx-max). Re-run with longer prompt
  in next iteration.
- The Path C regression at small ctx is interesting: the 4-SG int8 fused
  kernel pays per-tile waste cost at small M (64×32 tile = 64 rows, but
  per-expert M=64 → 100% of M-rows useful, 0 wasted; M=10 → 84% wasted).
  At 2K (per-expert M≈64), the 4-SG matmul amortization is bad even with
  ideal tile fit — possibly the NR1=64 M-tile is too big for small per-expert
  workloads. Future fix: an NR1=32 i8 fused variant.

## Reproduce

```bash
# Plan A peak recipe:
sleep 60   # cooldown if other work just ran
env DS4_LOCK_FILE=/tmp/ds4-bench.lock \
    DS4_METAL_PREFILL_CHUNK=16384 \
    DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1 \
    DS4_RESIDENT_MOE_MPP_FORCE=1 \
    DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64 \
    DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS=64 \
    DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=1 \
    DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT=1 \
    DS4_RESIDENT_MOE_NAX_HALF=1 \
    DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1 \
    DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0 \
    ./ds4-bench \
      -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
      --metal --moe-mode off --warm-weights \
      --ctx-start 16384 --ctx-max 16384 --gen-tokens 1 \
      --prompt-file <prompt-with-≥16K-tokens>
```
