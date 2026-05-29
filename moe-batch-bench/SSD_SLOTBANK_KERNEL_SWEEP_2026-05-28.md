# SSD slot-bank forced-kernel PREFILL sweep — 32 GB M5 (2026-05-28)

**Hardware:** Apple M5, 32 GB unified memory (single-cluster). Model:
`dsv4-iq2xxs-expert-major` dense (8.2 GB) + expert sidecar, SSD-streamed.
**Raw data:** [`SSD_SLOTBANK_KERNEL_SWEEP_2026-05-28.csv`](SSD_SLOTBANK_KERNEL_SWEEP_2026-05-28.csv)

Forced-backend prefill comparison of the GPU routed-MoE kernels in **SSD streaming
mode** (`--moe-mode slot-bank`, slot-bank=8) on the 32 GB M5. This is the regime
that matters for the 32 GB box: the resident kernels (ALU Path-C / NAX-half /
NAX-int8) need the 81 GB model and are M5-Max-only; here everything streams expert
records from SSD.

## What this validates

The Plan A **NAX+ALU fused gate+up+swiglu** kernel was previously resident-only.
This sweep is the first run after wiring it into the slot-bank prefill path
(`ds4_gpu_routed_moe_expert_banked_batch_mpp_int8_tensor`, `use_h_h` branch) behind
`DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1` + `DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS`, with
a new `ds4_mpp_mul_rows_weight_f32` kernel reapplying the per-token routing weight
(the fused kernel omits it).

## Backends (all share `DS4_FLASH_MOE_MPP_INT8_PREFILL=1`)

| name | path | extra env |
|---|---|---|
| `gpu_int8` | base MPP h×i8 | (none) |
| `nax_int8` | i8×i8 fused-dequant matmul2d | `MPP_I8I8_PREFILL=1 MPP_I8I8_FUSED_PREFILL=1` |
| `nax_half` | NAX-half, **separate** gate/up + swiglu | `RESIDENT_MOE_NAX_HALF=1` |
| `nax_alu` | **Plan A** fused gate+up+swiglu (NAX∥ALU) | `+ NAX_FUSED_GATE_UP=1 NAX_FUSED_MIN_REFS=0` |

## Methodology

- Cache-isolated cold prefill: runs under a throwaway `$HOME`, KV cache wiped before
  every run, so all backends prefill the **identical** token count (system 1274 +
  user prompt). Without this, the first backend per ctx cold-prefills while the rest
  reuse the cached system KV → biased, non-comparable t/s.
- `total_tps` = full cold prefill (system+prompt); `resume_tps` = the user-prompt
  phase (larger batches, the more kernel-discriminating signal).
- io-split=4 (default), xlayer prefetch topk=4 (auto), async-pread on, ANE off.
- 1K ctx omitted: the agent's system prompt alone is 1274 tokens (> 1024).

## Results (total_tps / resume_tps, t/s)

| ctx | gpu_int8 | nax_int8 | nax_half | **nax_alu (Plan A)** |
|---|---|---|---|---|
| 2K  | 62.7 / 53.7 | **65.3 / 55.0** | 60.8 / 51.3 | 63.2 / 53.5 |
| 4K  | 75.9 / 80.2 | **82.0 / 88.0** | 71.8 / 75.1 | 77.5 / 82.3 |
| 6K  | 83.9 / 89.7 | **92.3 / 101.1** | 80.6 / 85.7 | 88.0 / 95.6 |
| 8K  | 89.0 / 94.8 | **99.0 / 107.9** | 85.4 / 90.9 | 93.8 / 101.7 |
| 16K | 83.6 / 85.7 | **98.1 / 101.9** | 83.2 / 85.4 | 93.8 / 97.2 |
| 32K | 82.7 / 83.5 | **95.4 / 97.0** | 81.2 / 82.2 | 90.9 / 92.4 |

## Findings

1. **`nax_int8` is the fastest SSD-streaming routed kernel at every context** (~95–99
   t/s at large ctx), ~4–5% ahead of Plan A.
2. **Plan A (NAX+ALU fused) is a solid 2nd and beats its own separate baseline
   (`nax_half`) by a margin that grows with context** — the fusion is a real win in
   SSD mode, confirming the synthetic-probe prediction:

   | ctx | Plan A vs nax_half (total) |
   |---|---|
   | 2K | +3.9% |
   | 4K | +7.9% |
   | 6K | +9.2% |
   | 8K | +9.8% |
   | 16K | +12.7% |
   | 32K | +11.9% |

3. At large ctx the half-separate and base-int8 paths drop into an I/O-bound plateau
   (~81–84 t/s) while `nax_int8` and Plan A hold up (~91–98) — the fused/int8 paths'
   fewer device round-trips help under slot-bank memory pressure.

## Recommendation

For SSD-streaming prefill on the 32 GB M5: **use `nax_int8`** (fastest). **Plan A is
the best half-precision path** and the recommended NAX option; it now works in
slot-bank mode. ALU / Path-C remains deferred (counted-indirect kernel, needs the
id-map/counts machinery the per-expert banked path lacks).

## Correctness caveat

This SSD prefill path is **non-deterministic run-to-run for every backend** (f16
accumulation order under concurrent GPU work — verified on the untouched `nax_int8`
too), so Plan A could not be bit-verified e2e. It is coherent, first-token-consistent
with the other backends, and the kernel is probe-validated in isolation. A true
numeric check would need a logit/intermediate-buffer comparison harness.
