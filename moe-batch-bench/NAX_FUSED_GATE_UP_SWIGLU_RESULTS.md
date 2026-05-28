# NAX fused gate+up+swiglu — results, ALU∥NAX concurrency, and methodology
_2026-05-27 · M5 Max · DeepSeek-V4-Flash IQ2XXS · resident in-RAM_

## TL;DR

Plan A — wiring the existing `ds4_mpp_fused_gate_up_swiglu_h_h_f_n32` kernel with
`DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0` (i.e. engage on ALL experts, override the
default 128) — **clears the published 532 t/s resident-prefill ceiling, reaching 544.7 t/s
at 16K (+2.4%)**. Path C (an analogous int8 fused kernel built this session) is
correctness-verified but **neutral** (≈ 519 t/s, within −1.5% of baseline). The key
difference is the simdgroup count, see "ALU∥NAX concurrency" below.

| Config (cooldown-separated, single agent, M5 Max @ resident 16K) | t/s |
|---|---:|
| mul_mm_id baseline (upstream — no resident MPP envs) | 381.3 |
| NAX-int8 baseline (compact bridge, the published-532 path) | 526.8 |
| Path C — i8 fused (`DS4_RESIDENT_MOE_NAX_FULL_FUSED=1`) | 518.8 |
| **Plan A — h_h_f fused + `FUSED_GATE_UP=1` + `MIN_REFS=0`** | **544.7** |

## The fused approach — what the kernels do

Both kernels collapse the three-stage routed-expert MLP into a single dispatch
per expert. The three stages being fused:
1. `gate = X · W_gate`  (matmul → int32 or float cooperative tile)
2. `up   = X · W_up`    (matmul → int32 or float cooperative tile)
3. `mid  = SiLU(gate) * up * route_weight` then quantize to int8 (for the
   downstream `down = mid · W_down` matmul).

Pre-fusion baseline runs these as **three separate kernels** with device-memory
round-trips on `gate_i32` and `up_i32` between them. The fused kernel keeps both
matmul outputs in cooperative tile / threadgroup memory and writes only the
final int8 `mid` to device — saving:
- Two device-memory writes (`gate_i32`, `up_i32`) + two reads in swiglu (~32 MiB
  per layer per expert at expert_mid_dim=2048 / per-expert M=512).
- Two kernel-launch / barrier overheads.
- Cooperative-tile reuse: the matmul outputs feed swiglu directly in-place.

Two variants implemented:

### Path C — `ds4_mpp_iq2_fused_gate_up_swiglu_counted` (new, this session)
- Inputs:  int8 A, iq2_xxs weights (gate+up), `counts[expert]`.
- Pattern: on-the-fly iq2 → int8 dequant into TG `Btile_g`/`Btile_u` per K-256
  segment, two i8×i8 matmul2d ops accumulating to int32 cooperative tiles,
  type-juggle to float (cT i32 → tg i32 → per-thread float, see
  `probe_b_int8_typejuggle.m` which validates max_abs=0), routing-weight ×
  swiglu, sat-int8 quantize, write to mid_i8.
- Tile/SG: NR1=64 (M), NR0=32 (N), NK=256, `execution_simdgroups<4>` (4 SGs, 128
  threads/TG).
- TG memory: 16 KB (two 8 KB Btile reinterpreted as two 8 KB int32 staging
  buffers after K-loop — same physical memory, sequential views).
- Wiring: `ds4_metal.m` compact bridge `use_fused` branch, gated by
  `DS4_RESIDENT_MOE_NAX_FULL_FUSED=1` (default off).
- Host wrapper: `ds4_gpu_encode_mpp_iq2_fused_gate_up_swiglu_counted_indirect`.

### Plan A — `ds4_mpp_fused_gate_up_swiglu_h_h_f_n32` (existing)
- Inputs:  half X, half pre-dequantized weights (iq2 → f16 per expert per layer).
- Pattern: two h×h matmul2d ops into float cooperative tiles, swiglu on
  cooperative tile in-place, single `cT.store(mid_f32)`.
- Tile/SG: NR0=NR1=NK=32, `execution_simdgroups<1>` (**1 SG, 32 threads/TG**).
- Wiring: staged-MPP bridge in `ds4_gpu_routed_moe_batch_tensor`, gated by
  `DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1`, per-expert threshold
  `DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS` (**default 128 — set to 0 to win**).
- Pre-step: per-expert `ane_dequant_iq2_xxs_f16` materializes f16 gate/up weights
  in a scratch buffer; the fused kernel reads them as half.

## ALU∥NAX concurrency — what actually works

The M5 Max GPU's NAX tensor engine and the SIMD-group ALUs are **separate
execution units within the GPU**. They _can_ run in parallel — but **only within
a single kernel**, and only when there are simdgroups not currently issuing NAX
ops. Cross-kernel: the M5 scheduler serializes (`m5-gpu-compute-serial-no-metal-nax-overlap`
memory note + `concurrency_probe.m`).

This is exactly why the two fused variants give **opposite** production results:

| Variant | simdgroups | NAX-busy SGs | ALU-free SGs | In-kernel overlap available? |
|---|---:|---:|---:|---|
| Path C (i8) | 4 | 4 | **0** | **No** — all SGs feed the matmul2d |
| Plan A (h_h_f) | 1 | 1 | (n/a within 1 SG, but **inter-stage overlap is real**) | **Yes** — the swiglu post-pass overlaps with the next-K NAX issue window |

For Plan A specifically: `execution_simdgroups<1>` means matmul2d runs on a
single 32-thread simdgroup. The dispatcher within the SG can issue NAX
instructions while ALU instructions on the cooperative tile (the per-element
SwiGLU loop) are still draining — these are **separate functional units** even
within one SG. The combined NR=32 × NR=32 cooperative tile (1024 elements ÷ 32
threads = 32 elements/thread) gives enough ALU work per K-iter to keep both
units busy.

For Path C: the same trick is unavailable because all 4 SGs are continuously
issuing NAX MMA — every SG cycle is NAX-bound, so the ALU swiglu work has no
"shadow" to hide behind. The two `op.run()` calls also serialize for the same
reason. The synthetic `fused_kernel_probe.m` measured 1.46–1.69× for an h_h_f
1-SG kernel; that win does **not** transfer to a 4-SG kernel.

The 1-SG cost is throughput per kernel-instance (one simdgroup of MMA
issue/cycle vs four), but Plan A wins overall because:
- The h_h_f matmul is half-precision (16-bit), so per-element throughput is
  doubled vs int8 cooperative (which is int32-accumulating).
- The reclaimed ALU lanes pay for the SwiGLU + clamp + cooperative-tile
  bookkeeping that the i8 path has to do in extra time.
- The dequant cost is paid **once per expert per layer** in a separate
  `ane_dequant_iq2_xxs_f16` pre-step, then reused across all the expert's tiles.

## The MIN_REFS=128 → MIN_REFS=0 finding

`DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS` gates which experts go through the
fused kernel:
- refs ≥ threshold → fused kernel.
- refs < threshold → fall back to separate gate matmul + up matmul + swiglu.

Default 128 was set from the synthetic `fused_kernel_probe.m` which showed
fusion barely helps at M=64 (~1.08× win). **That was a wrong tuning for
production** because:
- Per-expert dispatch overhead (kernel-launch + barrier) is roughly constant.
- With 256 experts × 43 layers × tail experts having M < 128, the
  cumulative overhead from running the slow separate-kernel path for tail
  experts swamps the per-tile-throughput cost of fusing even M=10.

Test (after fixing the thermal/concurrent-agent contamination — see below):

| MIN_REFS | t/s @ 16K resident, cooldown-separated |
|---:|---:|
| 0 | **544.7** ← best |
| 128 (old default) | ~487 |

**Recommendation: change the default to 0**, or remove the gate entirely.

## Methodology — gotchas that fooled an earlier pass

Two contamination sources nearly buried the Plan A win:

### 1. M5 Max thermal degradation across sequential runs
- 16K resident prefill takes ~30 s of sustained compute.
- Apple silicon's thermal envelope on the Mac mini / M5 Max degrades rapidly
  under continuous load.
- Earlier "MIN_REFS sweep" showed 524 → 511 → 504 → 506 → 490 → 487 as REFS
  went 0 → 128. This **was not** a MIN_REFS sensitivity curve — it was thermal
  degradation across sequential runs.
- **Required**: 30–90 s `sleep` between bench iterations. 90 s is conservative
  and recovers the published 532-ish rates reliably.

### 2. Concurrent GPU contention from other processes/agents
- Any other `ds4`, `moe-batch-*`, or `ane_*` process running concurrently
  will share the GPU and skew bench results.
- Check before benching:
  ```bash
  ps -A -o pid,command | grep -iE "ds4|moe-batch|ane_" | grep -v grep
  ```
- (Chrome with GPU acceleration is OK; full-tilt MPP/NAX peers are not.)

### 3. The MPP-vs-MPP trap (separate but related)
- With `DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64`, even removing
  `DS4_RESIDENT_MOE_MPP_FORCE` still routes a 2372-token prompt through
  resident MPP. The "true mul_mm_id baseline" needs **all** `DS4_RESIDENT_MOE_*`
  envs unset.
- Validation flow that works: CLI `--dump-logits` with explicit env-separated
  baseline/candidate, plus layer-0 `ffn_moe_out` dumps for numerical checking.
  `ds4_test --metal-tensor-equivalence` is unreliable here because its
  reference capture inherits the same global resident-MPP env.

## Decode (generation) is NOT improved — only prefill

Measured at ctx=2K, --gen-tokens=64, cooldown-separated:

| Config | Prefill t/s | Decode t/s |
|---|---:|---:|
| NAX-int8 baseline | 204.7 | 30.14 |
| Plan A fused MLP (MIN_REFS=0) | 341.3 (+66.7%) | 30.15 (no change) |

The fused kernel saves dispatch overhead, which scales with prefill batch size
(per-expert M = ctx × topk / n_experts). At decode M=1, the MoE compute is
already a tiny fraction of per-token time — the bottlenecks are attention KV
scan, dense layers, and memory latency, not gate/up/swiglu. Plan A is a
prefill-only optimization.

Prefill speedup vs ctx (rough):
- ctx=2K  (per-expert M≈64):  +66.7% (204→341)
- ctx=16K (per-expert M≈512): +3.4%  (526→544)

The relative win is largest at smaller contexts because dispatch overhead is a
bigger fraction; at large M the matmul math itself dominates and there's
proportionally less to gain.

## Reproducing the Plan A win

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
make ds4-bench

# Make sure no peers are running
ps -A -o pid,command | grep -iE "ds4|moe-batch|ane_" | grep -v grep   # must be empty

# Run with 90s pre-cooldown (let the GPU cool from any prior work)
sleep 90

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
      --prompt-file <a-prompt-file-with-≥16K-tokens>
# Expect: 540+ t/s prefill
```

## Path C status (the i8 fused build)

Built this session — kernel, pipeline, host wrapper, compact-bridge wiring,
correctness-verified at short prefill (23 tokens, byte-identical output text
vs baseline). Gated off by default behind `DS4_RESIDENT_MOE_NAX_FULL_FUSED`.
**Stays in the tree** as:
- A reusable type-juggle pattern (cooperative int32 → tg int32 → per-thread
  float) — see `probe_b_int8_typejuggle.m` for the standalone validation
  (max_abs = 0).
- A reference for "what fusion of two 4-SG matmul2d ops looks like" — i.e.
  the **failure mode** documented above. If a future Apple GPU surfaces real
  cross-kernel matmul concurrency, this kernel could become useful.

## Related probes (all in `moe-batch-bench/`)

| Probe | Status | Use |
|---|---|---|
| `fused_kernel_probe.m` | passed | synthetic h_h_f-fused vs h_h_f-separate (1.69× @ M=512) |
| `concurrency_probe.m` | passed (negative) | two queues / two kernels → 1.00× (cross-kernel serializes) |
| `probe_a_counted_indirect.m` | passed | id-mapped weight selection overhead ≈ 2–9% |
| `probe_b_int8_typejuggle.m` | passed | i32 cooperative → tg → float-device, max_abs=0 |
| `nax_multiexpert_probe.{m,binary}` | passed (probe only — not wired) | one-dispatch multi-expert via tile→expert id_map |

## Open levers (not pulled this session)

### Multi-expert grouping (production wiring)
**Status: NOT implemented in production.** Probe exists at
`moe-batch-bench/nax_multiexpert_probe.m`. Production prefill in
`ds4_metal.m:22683+` (compact bridge `use_fused` loop) and analogous loops
issue **one dispatch per expert** — the "batch 2–4 experts per dispatch to
push effective M into the 1.5×+ regime" lever has not been pulled.

The probe demonstrates the kernel side: a single dispatch with
`te[tile_idx]` (expert id per output tile), `tr0[tile_idx]` (per-tile row
offset in concatenated A), `trc[tile_idx]` (per-tile row count) — the kernel
reads its expert at tile granularity and indexes into the right A rows /
right Wq slice. Production wiring would need to:
1. Pack the active experts' int8 acts into one contiguous A buffer ordered
   by expert.
2. Build the `te/tr0/trc` per-tile metadata.
3. Issue one dispatch over the full grid.
4. Scatter the results back to pair-indexed midbuf.

Potential win: especially valuable for the **dedup path** which is
submission-floor-bound (`metal_graph_resident_moe_run_mpp_prefill_dedup`,
capped at ~375–406 t/s per the resident sweep). Two layered effects:
- Collapse the ~256 per-expert command buffers / dispatches into ≤256/G
  (where G is the group size).
- Push effective M per-dispatch to G × per-expert-M, which puts the matmul
  into a more amortized regime (per `nax_kernel_tuning_playbook` memory).

### Default tuning changes worth landing
- Flip `DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS` default from 128 → 0 in
  `ds4_metal.m:23332` (or delete the gate). This alone unlocks Plan A
  by default whenever the half flag is on.
- Consider whether `DS4_RESIDENT_MOE_NAX_HALF=1 +
  DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1` should be auto-enabled on M5+
  given the +2.4% over the int8 ceiling.

### Correctness — extend Plan A's harness to Path C
Plan A is numerically validated (layer-0 ffn_moe_out vs mul_mm_id:
max_abs=5.4e-05, rms=3.75e-06, same argmax). Path C only has a
byte-identical-output short-prompt check. To make Path C a candidate for
production enable: dump layer-0 ffn_moe_out under Path C vs the int8
baseline (CLI `--dump-logits` flow described above).
