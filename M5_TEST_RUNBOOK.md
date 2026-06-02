# M5 / M5 Max test runbook — ANE, ALU, NAX, NAX-int8

Branch: `agent-clean`. Build: `make ds4-agent ds4-bench`.

Two distinct regimes — pick by machine **by RAM**:

| Regime | Mode | Model | RAM needed | Machine |
|---|---|---|---|---|
| **SSD slot-bank** | experts paged from sidecar | `dense/model-dense.gguf` (8.8 GB) + `--moe-sidecar` | ~10–16 GB | **base M5 (32 GB)**, M4 Pro, M3U |
| **Resident** | all experts in RAM | full `...chat-v2.gguf` (81 GB) `--moe-mode off` | **~81 GB** | M5 Max only |

> **32 GB M5 cannot run resident** (the 81 GB model won't fit). On the 32 GB M5 use
> **(A) SSD-mode e2e** + **(C) synthetic calibration** below. Resident sweep (B) is
> M5 Max only. `DS4_MODEL` / `DS4_SIDECAR` are env-overridable in every launcher.

---

## C. Synthetic kernel calibration  (ANY machine — NO model load, runs on 32 GB)

The fastest, cleanest way to set the kernel gates for a given GPU: microbenchmark
the routed-MoE matmul kernels per per-expert batch size M. No 81 GB model, no SSD
I/O — just the kernels. **This is how the 32 GB M5 calibrates its gates.**

```bash
./run_synthetic_calibrate.sh                 # builds probes, sweeps M=32..1024
# knobs: MS="32 64 128 256 512 1024"  N=2048  K=4096  ITERS=30
```
Reads out, per M: NAX-half fused gate+up+swiglu vs separate (speedup + best tile/SG)
and the NAX-int8 kernel. **Gate rule:** fused speedup > 1 at all M ⇒
`DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0`; if it only wins above M*, set MIN_REFS=M*.
On a slow GPU the absolute ms grow but the speedup ratio (which sets the gate) holds.

M5 Max reference (per M, NAX-half fused speedup vs separate): 32→1.20, 64→1.26,
128→1.31, 256→1.17, 512→1.48, 1024→1.42 — fused wins everywhere ⇒ MIN_REFS=0.

---

## A. SSD-mode ANE-vs-GPU A/B  (any machine, esp. SLOW GPU)

The key test: **routed-on-ANE vs routed-on-GPU**, overlapped with GPU-dense.
On M5 Max (fast GPU) the gap was small; on a **slow GPU** ANE should win wider.

```bash
# one-shot, non-interactive (pipe a >=2K-token prompt, append /quit)
PROMPT=tests/test-vectors/prompts/long_code_audit.txt   # or any large prompt

# routed -> ANE (i8i8 tiled-fused, overlapped with GPU-dense)
{ cat "$PROMPT"; printf '\n/quit\n'; } | ./run_ane_ssd_agent_m5.sh --ctx 16384 --non-interactive -n 8

# routed -> GPU (MPP-int8), ANE off  -- the comparison
{ cat "$PROMPT"; printf '\n/quit\n'; } | ./run_nax_ssd_agent_m5.sh --ctx 16384 --non-interactive -n 8
```
Read the `prefill ... avg=NNN t/s` line. ANE win = `ane avg` > `nax avg`.

`_m5.sh` = 8 slots (base M5), `_m5max.sh` = 48 slots. Override: `DS4_SLOTS=N`.

### Force MORE experts onto ANE (try on slow GPU — the M5-Max 384 floor is too high there)
```bash
DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=0 \   # main gate: refs floor for ANE eligibility (default 384 in launcher)
DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL=0 \    # batch-utilization floor (default 0)
  ./run_ane_ssd_agent_m5.sh --ctx 16384 --non-interactive -n 8 < "$PROMPT"
```
Sweep `HYBRID_ANE_MIN_REFS` ∈ {0, 32, 64, 128, 256, 384} on the slow GPU to find
its crossover (lower than M5 Max's because the GPU side of the overlap is slower).

---

## B. Resident backend sweep — ALU / NAX / NAX-int8  (M5 Max, 81 GB)

Re-establishes the correct gates (MIN_TOKENS, Plan A MIN_REFS, per-ctx winner).

```bash
# 1. make a >=32K-token prompt once
for i in $(seq 8); do cat tests/test-vectors/prompts/long_code_audit.txt; done > /tmp/big_prompt.txt

# 2. close other GPU apps; run the cooldown-separated sweep (2K..32K)
./run_resident_variant_sweep.sh            # -> resident_variant_sweep_<ts>.csv
```
Knobs: `CTXS="2048 8192 16384 32768"`, `VARIANTS="mulmm int8 nax alu"`,
`COOLDOWN=90`, `DS4_PROMPT=`, `OUT=`.

Variant availability by branch:
- `agent-clean`: `mulmm`, `int8` (and `nax` once Plan A wiring is finished).
- `bad-stat-NAX_ALU-ANE`: all four (full session work) — use this branch to sweep ALU/NAX now.

### Reading the sweep → gates
- **`DS4_RESIDENT_MOE_MPP_MIN_TOKENS`**: first ctx where `int8 > mulmm` is the crossover
  (~6–8K historically). Set MIN_TOKENS just below that ctx so resident MPP only
  engages where it wins.
- **Plan A (`nax`)**: where `nax > int8`, NAX-half wins. `DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0`
  was best after proper cooldown (earlier "MIN_REFS sweep" was thermal noise).
- **`alu` (Path C)**: expected ≈ or below `int8` — M5 has no cross-kernel NAX∥ALU
  overlap, so the 4-SG i8 fused kernel can't beat int8. Confirm it's not a regression.

### Resident single-config quick checks
```bash
# NAX-int8 baseline
DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1 DS4_RESIDENT_MOE_MPP_FORCE=1 \
DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64 DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS=64 \
DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=1 DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT=1 \
DS4_METAL_PREFILL_CHUNK=16384 \
  ./ds4-bench -m <full.gguf> --metal --moe-mode off --warm-weights \
     --ctx-start 16384 --ctx-max 16384 --gen-tokens 1 --prompt-file /tmp/big_prompt.txt

# Plan A (NAX-half) = add:
#   DS4_RESIDENT_MOE_NAX_HALF=1 DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1 DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0
# Path C (ALU)      = add:
#   DS4_RESIDENT_MOE_NAX_FULL_FUSED=1
```
Or use `./run_nax_resident_agent_m5max.sh` (interactive Plan A) once it's on the branch.

---

## Methodology (don't skip — these invalidate results)
1. **Cooldown** ≥30–90 s between runs. M5 thermals decay throughput across back-to-back
   runs; a cold first run reads high then falls. The sweep script enforces this.
2. **No other GPU users.** `ps -A | grep -iE 'ds4|moe-batch|ane_'` must be empty.
   The sweep script aborts otherwise.
3. **Resident needs `--warm-weights`** (pages in 81 GB before the prefill timer).
4. Prefer 3 cooldown-separated runs and take the median, not a single shot.

## Known reference numbers (M5 Max, resident 16K, cooldown-clean)
| backend | t/s |
|---|---:|
| mul_mm_id (upstream) | ~381 |
| NAX-int8 | ~532 |
| **NAX-half / Plan A** | **~573** |
| Path C (ALU i8 fused) | ~495 (neutral) |
| ANE-SSD (flash, 16K, M5 Max) | ~330 (different regime) |
