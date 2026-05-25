# M5 Max ANE prefill optimization report

> **2026-05-24 — int8 (W8A8) DENSE: +15–20% end-to-end prefill (the loop's one real win).**
> After exhausting every float NAX knob (indexer relaxed / Morton / attn_out tile — all end-to-end
> neutral) and int8 indexer (neutral, small slice), the win is **int8×int8 dense projections**:
> per-row-scale int8 weight (load-time Q8_0→int8 repack, cached) × per-token-scale int8 activation,
> int8→int32 with fused rescale (kernels `ds4_repack_q8_to_i8_rowscale` / `ds4_quant_act_pertoken_i8`
> / `ds4_dense_i8_fused`, NR1=128/NR0=32/NK=128). Microbench +30–48% vs relaxed float×half (0.7% rel);
> **end-to-end +19.8%@8k, +19.5%@16k, +17.1%@24k, +15.1%@32k** (adjacent-pair A/B, both pairs within
> 2–3%), generation **token-identical** to baseline. Recipe that mattered: NK≥128, weight pre-quantized
> offline (no per-dispatch dequant), fused rescale (NR0=32 to fit the int32 tile in threadgroup).
> Env-gated `DS4_GPU_DENSE_I8` (default off; cost = a parallel int8 weight copy in GPU mem). Likely
> puts the fork **ahead of antirez** (relaxed float×half dense), vs prior ~−5% parity. See memory/*.md.


> **2026-05-24 — full 5-kernel NAX autotune (microbench, no model).** Built per-kernel autotuners
> (`nax_dense_autotune.m`, `nax_attnout_autotune.m`; MoE via `nax_fused_probe.m`; indexer via
> `nax_autotune.m`) and swept every NAX kernel over NR1/NR0/TM/NK/relaxed/walk(/dtype).
> | kernel | verdict | note |
> |---|---|---|
> | dense_q8 | **optimal, no change** | 128/64/32/relaxed/reg = 37.6 TF/s; relaxed_precision is a ~3× lever (already on); Morton hurts |
> | attn_out | **optimal, no change** | 128/64/32/relaxed/reg = 33.5 TF/s; microbench marginally likes 64/64/64 but end-to-end NR1=128 won |
> | indexer | **+3–8%, bit-exact** | ships relaxed=**false**; flip to **true** (stable +2.9/+5.2/+14.6% over 3 runs). int8 = ~2.5× but 5.7% drift → needs coherence A/B |
> | MoE iq2/q2k | **optimal, no change** | bit-exact, 4.2× the dequant-to-global baseline; int8→int32 so relaxed is a no-op, tile baked; lever is larger M (batching) |
> Method lessons: (1) microbench has run-to-run variance — repeat the candidate A/B, don't trust one sweep.
> (2) **Walk order (Morton/regular) cannot be measured in a microbench** — it changes no FLOPs, only L2
> locality, which needs real cache pressure; the isolated working set is L2-resident so the knob is invisible
> there. Morton must be A/B'd end-to-end. (3) Tile confirmation is per-kernel: NR1=128/NK=32 was confirmed by
> microbench only for **dense**; attn_out's microbench best was 64/64/64 (kept 128 on the prior end-to-end
> win); indexer holds TM=16/NK=32; MoE uses 64/32/256. The one actionable kernel change is **indexer
> relaxed=true** (free, bit-exact).

> **2026-05-24 RESULT — NAX parity with antirez.** Ported antirez's three NAX matmul2d kernels
> (dense `direct_rhs`, indexer scores, grouped `attn_out` O-proj) into `metal/nax_fused.metal` with the
> key tuning (NR1=128 token tile, NK=32, direct-from-device activation). vs antirez current build on M5 Max:
> **step-2048 (his measurement): parity/faster** (mean −1.8% over 8–65k, +1–3% at 8–22k); **production
> M=4096: ~−5%** (16k −3%), down from −15% baseline / −11% before attn_out. Generation stays coherent
> (benign f16 drift). Flags (default off): `DS4_GPU_DENSE_NAX` (dense + attn_out), `DS4_GPU_INDEXER_NAX`
> (n_comp≥3072 gate). Lesson: every NAX win hinged on **NR1=128** (NR1=64 *hurt*). drift-patch flags are
> numerical (output-match), not speed. Remaining ~5% = large-M amortization + drift-patch. See memory/*.md.


> **2026-05-24 strategy correction (upstream antirez verdict).** Upstream ds4 commits
> `d4fba7b` (disable routed-MoE TensorOps), `3d14d1c` (gate TensorOps flips top-k routing),
> and `18c2d4b` (NAX speedups + cleanup) establish that **NAX/TensorOps `matmul2d` for the
> routed-MoE gate/up/down is a dead end**: it is numerically unstable (drift vs the legacy
> simdgroup MMA flips top-k router selection → repetition/bad continuations) *and* slower
> than the legacy 32-token expert-major simdgroup kernel. Upstream removed it. The real NAX
> prefill win (README ~463 t/s long-prompt q2) comes from the **attention path**: an
> indexed-attention **F16 read-side shadow KV cache** (F32 canonical, F16 shadow → ~½ the
> 512-row indexer bandwidth) plus direct-RHS tensor layout — MoE stays legacy simdgroup.
>
> Caveats on applicability to this fork (corrected 2026-05-24): the partial-tile <64-row NAX
> correctness bug is already mitigated here (legacy-Metal fallback for tail tiles; `DS4_FLASH_MOE_
> MPP_ALLOW_PARTIAL_TILES`), and the 2026-05-24 fused-dequant kernel handles M-tails via a reduced
> runtime tensor extent (validated bit-exact for M=33/100/130), so it does not hit the garbage-tail
> path. Upstream's residual instability (`3d14d1c`) was **fp16 matmul2d drift** flipping top-k
> routing — fp16-specific, less applicable to our **exact int8** matmul (int8×int8→int32). So the
> int8 fused path is not blanket-unsafe; it just (a) needs a semantic 100-token divergence check vs
> the legacy simdgroup path before trusting for generation, and (b) is **slower than the plain GPU
> `mul_mm_id` path at small ds4-bench chunks** per our own data, so it is not the route to antirez's
> curve. **Next lever = the F16 shadow KV cache for indexed attention** (the validated upstream win).
> See `memory/nax-fused-moe-findings.md`.
>
> **2026-05-24 FINAL: naive NAX/matmul2d does not beat ds4's tuned simdgroup kernels on M5 Max.**
> Four drop-in attempts, all validated-correct but default-off: MoE int8 (~tie), F16 KV shadow (neutral),
> dense Q8_0 float×half (−4%), indexer-score half×half (−2..−7%, worsens with ctx). The clean same-binary
> gap to antirez's current build is ~10–17% (NOT 2x — the 510–710 screenshot was a pre-rollback build),
> growing with context (dominated by the indexer score matmul, per ds4's own profiling note). antirez's
> edge (tensor_matmul=on) is therefore not a droppable single NAX kernel; it's a deeply tuned tensor impl
> and/or his drift-patch stack. Only change that stuck: **4096 prefill-chunk default** (matches antirez,
> helps real long-prompt prefill). Flags (all default off): DS4_GPU_INDEXED_ATTN_COMP_F16,
> DS4_GPU_DENSE_NAX, DS4_GPU_INDEXER_NAX, DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT. See memory/*-findings.md.
>
> **2026-05-24 F16 shadow KV cache — IMPLEMENTED + MEASURED (result: neutral on M5 Max).** Ported
> antirez 18c2d4b: lossless F16 read-side shadow of the compressed KV (the indexed-attn dot already
> casts comp to f16, so 100-token temp-0 output is byte-identical). Gated `DS4_GPU_INDEXED_ATTN_COMP_F16`
> (default off); all in `ds4_metal.m` + `metal/dsv4_misc.metal` (no ds4.c changes). Same-binary
> ds4-bench (2k–65k): Δprefill −2..−4% small-ctx → break-even ~30k → flat ±1% at 40–65k; Δgen ~−1%.
> No win on M5 Max — comp-KV read isn't bandwidth-bound here (~400+ GB/s). Kept default-off (harmless,
> may help lower-bandwidth devices). So neither lever we could port (MoE NAX, F16 comp shadow) explains
> antirez's ~463 t/s on M5 Max; the remaining suspects are 18c2d4b's direct-RHS tensor layout +
> attention-output Metal4 opts (not yet ported) and/or the aggressive pre-rollback build.

Date: 2026-05-23

Host/model context used for the measurements below:

- Host: Apple M5 Max, 128 GB RAM.
- Model: `/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major`.
- Dense GGUF: `/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf`.
- Prompt: `coding_8k.txt`, 8423 input tokens.
- Common run shape unless noted: `--ctx 98192`, `DS4_METAL_PREFILL_CHUNK=32000`, `DS4_METAL_GRAPH_RAW_CAP=8704`, `--moe-mode slot-bank`, 32 slots.
- Timing note: all `prewarm` numbers are paid before the prefill timer and are not included in `ds4: prefill`.

## Executive summary

We currently have three independent ANE optimization surfaces:

| Optimization | What it replaces | Best isolated M5M result seen | Current status |
|---|---|---:|---|
| Streaming routed experts | DeDup routed expert MLP compute | `296.24 t/s` in current M5M ANE wrapper, 100-token run, 48 slots | Best standalone speed path so far, but scheduler/options are sensitive. |
| Shared expert | Always-active shared MLP | `292.80 t/s` with new `i8w-i8x` shared path; `305.99 t/s` fp16 constexpr repeat but noisy | Promising, but not bit-exact against GPU. Use as experimental. |
| O-proj | Attention output projection | `270.60 t/s` in GPU-routed + ANE O-proj i8w+i8x B512; only `+0.4%` vs local `269.40 t/s` baseline and below the best `276.56 t/s` GPU-only reference | Benchmark-only for now. The code now disables ANE O-proj when routed-ANE prefill is active unless `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1` is set. |
| All three combined | Routed + shared + O-proj on ANE | `244.31 t/s` | Not additive. O-proj currently makes the combined stack slower. |

The most important comparison is:

| Category | Routed expert MLP compute | Shared expert | O-proj | Prefill | Interpretation |
|---|---|---|---|---:|---|
| Best GPU-only reference | GPU | GPU | GPU | `276.56 t/s` | Current GPU-only reference. |
| Best routed-ANE standalone path | ANE | GPU | GPU | `296.24 t/s` | Best production candidate so far. |
| GPU-routed hybrid datapoint | GPU | ANE fp16 | ANE i8w/fp16x | `280.07 t/s` | Useful isolation run, but it does not use routed-ANE streaming and is not faster than `296.24 t/s`. |
| All-three ANE stack | ANE | ANE i8i8 | ANE i8w+i8x | `244.31 t/s` | Slower because the current O-proj scheduling introduces a large serial wait. |

So `280.07 t/s` is only "bigger" than the local `269.40 t/s` GPU-routed shared-test baseline. It is not bigger than the `296.24 t/s` routed-ANE path. The reason to keep it in the report is diagnostic: it shows that shared/O-proj can sometimes help when routed experts remain on GPU, but it does not prove that adding shared/O-proj to routed-ANE will help.

Implementation guard added after this measurement: if `DS4_FLASH_MOE_ANE_PREFILL=1` and `DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1`, O-proj falls back to the GPU path by default. To reproduce the old all-three experiment, explicitly add:

```bash
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1
```

Baseline references:

| Run | Config | Prefill |
|---|---|---:|
| `m5max_gpuonly_async_prefetch1_afterstage` | GPU-only routed i8i8, async pread | `276.56 t/s` |
| `m5m_shared_gpu_20260523_073906` | GPU routed/shared baseline in shared-expert test set | `269.40 t/s` |
| `m5max_ane_b256_tok100_s48` | Current M5M routed ANE wrapper, 100 tokens, 48 slots | `296.24 t/s` |

## Resident MPP/NAX int8 prefill

This is separate from sidecar/SSD DeDup. It targets the fully resident
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` path via:

```bash
--resident-mpp-int8-prefill
```

Current implementation status:

- Correctness smoke: first token matches the resident GPU run (`We`) on 1K, 4K, 6K, and 8K prompts. A 100-token 1K smoke produced coherent text.
- Fixed bug: once resident MPP has completed the routed MoE block, the normal Metal `up` matmul is now skipped. Before this fix the MPP path paid duplicate work.
- Auto-gate added: `--resident-mpp-int8-prefill` now keeps small resident chunks on the existing fused GPU route unless the staged MPP/NAX bridge is explicitly forced. The default threshold is `DS4_RESIDENT_MOE_MPP_MIN_TOKENS=16384`; set it to another value to tune, or use `DS4_RESIDENT_MOE_MPP_FORCE=1` for diagnostics. The threshold was raised from `8192` because 8K is a noisy tie on M5M, while 16K is a clear MPP win.
- Default bridge selector: sync-count bridge is used for chunks up to `7168` tokens; above that, the resident MPP path now auto-selects the compact scatter-add bridge when `n_tokens >= DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS` (default: same as `DS4_RESIDENT_MOE_MPP_MIN_TOKENS`, currently `16384`). Override with `DS4_RESIDENT_MOE_MPP_SYNC_BRIDGE=0/1`, `DS4_RESIDENT_MOE_MPP_PAIRROW_BRIDGE=1`, or `DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=0/1`.
- The compact bridge is the current large-chunk resident default on M5M. It fuses gate/up/down dequant and keeps the post-SwiGLU path in int8 before scatter-add to the routed output.
- Experimental dual gate/up MPP dispatch is available with `DS4_RESIDENT_MOE_MPP_DUAL_GATE_UP=1`, but it is not the default. It was noisy: one warm-cache run reached `323.58 t/s`, but another cold/early run was worse than separate gate/up dispatch.

Current resident smoke numbers on M5M:

| Prompt | Bridge | Prefill |
|---|---|---:|
| 1K | auto-gated to fused resident GPU | `254.24 t/s` |
| 1K | forced staged MPP/NAX | `124.92 t/s` |
| 4K | resident GPU reference | `299.45 t/s` |
| 4K | forced staged MPP/NAX | `263.52 t/s` |
| 6K | sync-count | `240.44 t/s` |
| 8K | auto, pair-row indirect, earlier run | `310.06 t/s` |
| 8K | auto, compact bridge, `ctx=100000` | `305.35 t/s` |
| 8K | resident GPU reference, `ctx=100000` | `297.91 t/s` |
| 8K | warm-cache compact bridge sweep | `322.64 t/s` |
| 8K | warm-cache resident GPU sweep | `318.06 t/s` |
| 16K | auto, compact bridge, `ctx=100000` | `334.43 t/s` |
| 16K | resident GPU reference, `ctx=100000` | `281.79 t/s` |
| 16K | auto, compact bridge after 16K gate/prewarm change | `362.93 t/s` |
| 16K | resident GPU reference after 16K gate/prewarm change | `310.11 t/s` |

Stage timing showed why the staged MPP bridge is still fragile: normal resident GPU
does roughly `15 ms gate + 16 ms up + 15 ms down` per 1K layer, while the staged
MPP bridge spends about `150 ms` in gate/up and `65 ms` in down. So MPP only wins
when the bridge overhead amortizes well. The default auto-gate protects ds4-agent
and short/default resident chunks from the slow bridge; large resident chunks now use
the compact MPP bridge because it is the only resident int8 path that has repeatedly
matched or exceeded the fused resident GPU reference on M5M. The long-term speed path
is still a true fused indexed tile kernel, not staged dequant/gather/matmul/scatter.

Latest 16K resident command shape:

```bash
DS4_METAL_PREFILL_CHUNK=18000 \
DS4_METAL_GRAPH_RAW_CAP=18000 \
./ds4 --resident-mpp-int8-prefill \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --metal --ctx 100000 --tokens 1 --temp 0 \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_16k.txt
```

`ds4-agent` smoke also works; a short `--non-interactive --nothink --resident-mpp-int8-prefill` run on a small prompt auto-gated back to fused GPU because the prompt was only `1313` resident tokens, then produced coherent output (`Hi! How can I help you today`).

## Terminology correction

Some profile names use `GPU_` to mean the routed MoE experts stayed on the GPU. They do not mean the whole run is pure GPU.

- `m5m_GPU_OPROJ_*` means GPU-routed experts plus ANE O-proj.
- `m5m_GPU_SHARED_OPROJ_*` means GPU-routed experts plus ANE shared expert plus ANE O-proj.
- A pure GPU baseline has no ANE routed expert path, no ANE shared expert, and no ANE O-proj.

This matters for attribution. The `280.07 t/s` run is not an O-proj-only win; it stacks fp16 shared-expert ANE and O-proj ANE, and it also changes the routed-expert layout:

| Run | Prefill | Routed unique experts | Sidecar bytes staged |
|---|---:|---:|---:|
| `m5m_shared_gpu_20260523_073906` | `269.40 t/s` | `10514` | `70969.50 MiB` |
| `m5m_GPU_OPROJ_I8X_B512_005348` | `270.60 t/s` | `9809` | `66210.75 MiB` |
| `m5m_GPU_SHARED_OPROJ_B512_004013` | `280.07 t/s` | `9790` | `66082.50 MiB` |

So the apparent `280.07 t/s` gain is partly a different ANE stack and partly a changed router/dedup/precision layout, not a clean measurement of O-proj alone.

Why that run does not use streaming routed-ANE:

- It was a component-isolation run. Routed experts were intentionally left on GPU so shared expert and O-proj could be tested without also changing the largest MoE path.
- Combining all three currently loses: `244.31 t/s`, mainly because O-proj has a large main-thread join wait in the combined schedule.
- Therefore the current best production direction is still routed expert streaming on ANE by itself, with shared/O-proj treated as separate experimental branches until their scheduling/correctness issues are fixed.

## 1. Streaming routed experts

This is the DeDup routed expert path. It streams sidecar expert records, dedups token/expert references, quantizes activations, and evaluates the fused `i8w-i8x` tiled MLP on ANE.

Primary flags:

```bash
DS4_FLASH_MOE_ANE_PREFILL=1
DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=1
DS4_FLASH_MOE_OVERLAP_PREFILL=1
DS4_FLASH_MOE_OVERLAP_SCHEDULER=1
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1
DS4_FLASH_MOE_MPP_INT8_PREFILL=0
DS4_FLASH_MOE_MPP_I8I8_PREFILL=0
DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0
```

Important suboptions:

| Suboption | Best/current M5M direction |
|---|---|
| `DS4_FLASH_MOE_ANE_BATCHES` | Current M5M wrapper defaults to `256`; older broad set `64,128,256,512` was slower. |
| `DS4_FLASH_MOE_ANE_MAX_REFS` | Current wrapper uses `256`. |
| `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS` | Current wrapper uses `384` to avoid low-util tiny ANE chunks. |
| `DS4_FLASH_MOE_ANE_OUTPUT_QUEUE` | Current wrapper uses `4`. |
| `DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK` | Current wrapper enables this on M5M. |
| `DS4_FLASH_MOE_ASYNC_PREAD` | Enabled. Helps more on M4 Pro, still kept on M5M. |
| `DS4_FLASH_MOE_PREFETCH` | M5M wrapper currently uses `3`; GPU-only wrapper uses `1`. |

Measured points:

| Log | Notes | Prefill | Useful details |
|---|---|---:|---|
| `m5max_ane_b256_tok100_s48` | Current M5M ANE-only wrapper, 100-token generation, 48 slots | `296.24 t/s` | B256 only, `pad_util=88.15%`, `ane_call_avg=4.86 ms`. |
| `m5m_ROUTED_ANE_BASE_002952` | Older stack-basis routed ANE run, 32 slots | `258.04 t/s` | `ane_call_avg=4.74 ms`, `pad_util=88.44%`. |
| `m5max_aneonly_async_pread_prefetch1_afterstage` | Older broad batches `64,128,256,512` | `214.88 t/s` | `pad_util=73.63%`, output pack/post pipeline cost was much higher. |
| `m5max_aneonly_baseline_no_async` | Same broad batches, no async pread | `204.41 t/s` | Confirms async/pipeline changes matter but do not fix bad batching alone. |

Interpretation:

- Streaming routed experts are the only ANE path that has shown a clean isolated win over the current GPU-only reference.
- The win is mostly from using larger, cleaner ANE chunks and reducing output conversion cost.
- The path is very sensitive to batch/ref policy. Bad padding or too many tiny calls erase the ANE advantage quickly.

Open correctness note:

- This is quantized i8 execution, so exact bit match to GPU fp/q8 is not expected. It still needs a repeatable divergence harness for 100-token and longer generation, not only human-readable output checks.
- M5 NAX/MPP partial-tile risk: the Metal `matmul2d` routed-expert path uses a 64-row `M` tile, while DeDup `refs` can be any value. That matches the reported Metal4/NAX correctness failure mode for non-divisible matrix dimensions. The code now routes only full 64-row chunks through MPP/NAX and computes the tail rows with the legacy GPU path. `DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES=1` or `DS4_FLASH_MOE_NAX_ALLOW_PARTIAL_TILES=1` restores the old behavior for benchmark comparison only.
- The padded ANE path is less exposed to this specific issue because calls are made at fixed batch sizes and only valid rows are copied/scattered back. The workaround is primarily for GPU MPP/NAX int8 routed expert compute, not for the private ANE/CoreML-style padded calls.

## 2. Shared expert

This replaces the always-active shared expert MLP after `ffn_norm`. It starts asynchronously after the norm GPU command buffer and joins before the residual add.

Primary flag:

```bash
DS4_FLASH_MOE_ANE_SHARED_EXPERT=1
```

Suboptions:

| Mode | Flags | Notes |
|---|---|---|
| fp16 split matmul | no extra flag | Mode 1, fp16 weights uploaded per call. |
| fp16 constexpr conv | `DS4_SHARED_EXPERT_ANE_CONV=1` | Mode 9, weights baked per layer, one eval call. |
| i8w+i8x tiled fused | `DS4_SHARED_EXPERT_ANE_I8I8=1` or `DS4_FLASH_MOE_ANE_SHARED_I8I8=1` | New mode 6 shared path; weights cached as int8, activations quantized to int8. |

Scale flags for shared i8i8:

```bash
DS4_SHARED_EXPERT_ANE_W_QSCALE=512      # falls back to DS4_FLASH_MOE_MPP_INT8_QSCALE
DS4_SHARED_EXPERT_ANE_X_QSCALE=32       # falls back to DS4_FLASH_MOE_MPP_INT8_X_QSCALE
DS4_SHARED_EXPERT_ANE_MID_QSCALE=32     # falls back to DS4_FLASH_MOE_MPP_INT8_MID_QSCALE
```

Measured points:

| Log | Shared mode | Prefill | Notes |
|---|---|---:|---|
| `m5m_shared_gpu_20260523_073906` | GPU shared baseline | `269.40 t/s` | `metal total=30240 ms`. |
| `m5m_shared_ane_split_20260523_073906` | fp16 split | `226.64 t/s` | Slow; avoid. |
| `m5m_shared_ane_constexpr_fp16_20260523_073906` | fp16 constexpr conv | `238.61 t/s` | First run slow/noisy. |
| `m5m_shared_ane_constexpr_fp16_repeat_20260523_074219` | fp16 constexpr conv repeat | `305.99 t/s` | Fastest shared-only point, but noisy and not bit-exact. |
| `m5m_shared_ane_i8i8_w512_x32_mid32_20260523_075059` | i8w+i8x, `512/32/32` | `292.80 t/s` | Best current i8 shared point. |
| `m5m_shared_ane_i8i8_w512_x64_mid64_20260523_075156` | i8w+i8x, `512/64/64` | `229.03 t/s` | Cleaner synthetic precision, worse real prefill. |

Synthetic precision smoke for shared/routed i8i8:

```text
H=256 I=128 B=8 wq=512 xq=32 midq=32
ane_vs_i8_cpu rel_rms=0.062654
i8_cpu_vs_fp16_cpu rel_rms=0.107307
ane_vs_fp16_cpu rel_rms=0.100938
```

Correctness status:

- Temp-0 100-token comparison diverges immediately from GPU for shared i8i8.
- The fp16 constexpr shared path also diverges immediately from GPU, so this is not only an int8 scaling issue.
- Shared ANE should currently be treated as approximate/experimental unless we accept output divergence or add a calibrated tolerance harness.

## 3. O-proj

This moves attention output projection to ANE. The implemented fast path is constexpr conv2d 1x1 with int8 per-channel weights; current code also supports int8 activations.

Primary flags:

```bash
DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_INT8=1
```

Suboptions:

| Suboption | Meaning |
|---|---|
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_I8X=1` | Use int8 activations as well as int8 weights. |
| `DS4_FLASH_MOE_ANE_OPROJ_BATCH=512` | Best tested production batch on M5M so far. |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=1..4` | Multiple ANE contexts/workers. B512 single-context is best among current production logs. |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_GPU_CVT=1` | GPU converts input to the bound IOSurface; worker calls `eval_at_chunk`. |
| `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1` | Allows ANE O-proj even when routed-ANE prefill is active. Without this, O-proj falls back to GPU to avoid the known serial join boundary. |

Measured production points:

These are GPU-routed runs with only O-proj moved to ANE. Use `269.40 t/s` as the closest local baseline for deltas, and `276.56 t/s` as the best current GPU-only reference.

| Log | Mode | Prefill | Delta vs local baseline | Notes |
|---|---|---:|---:|---|
| `m5m_GPU_OPROJ_B512_003750` | i8 weights, fp16 activations, B512 | `266.46 t/s` | `-1.1%` | Slower than local GPU-routed baseline. |
| `m5m_GPU_OPROJ_B1024_003900` | i8 weights, fp16 activations, B1024 | `269.10 t/s` | `-0.1%` | Essentially flat. |
| `m5m_GPU_OPROJ_I8X_B512_005348` | i8 weights, i8 activations, B512 | `270.60 t/s` | `+0.4%` | Best O-proj-only point, but this is not a meaningful standalone win yet. |
| `m5m_GPU_OPROJ_I8X_B1024_005500` | i8 weights, i8 activations, B1024 | `221.50 t/s` | `-17.8%` | Bad; avoid B1024 for i8x. |

O-proj smoke result from `m5m_i8x_vs_mlp_20260523_005310`:

- O-proj i8w+fp16x B256 T1: about `9.00 TF/s`.
- O-proj i8w+i8x B256 T1: about `11.28 TF/s`.
- Routed MLP i8w+i8x B256 reference: about `7.3 TF/s` solo, `11.4-11.6 TF/s` aggregate with multi-context.

Interpretation:

- The i8x O-proj kernel itself is better than fp16x.
- O-proj alone does not explain the `280.07 t/s` run. The cleanest O-proj-only production point is `270.60 t/s`, which is basically flat against the local `269.40 t/s` GPU-routed baseline.
- The larger `280.07 t/s` result came from a stacked `GPU-routed + fp16 shared ANE + O-proj ANE` run and a changed routed-expert layout. It should not be attributed to O-proj alone.
- End-to-end prefill still does not clearly win because O-proj has poor overlap with surrounding work.
- In combined mode, O-proj reports `eval_avg=123.888 ms/call` but `join_avg=366.555 ms/call`, so the serial wait is much larger than raw eval.
- Prewarm is also expensive: `~38-40 s` for 43 per-layer O-proj contexts.

Correctness status:

- O-proj i8x is not yet validated as bit-exact or tolerance-safe in generation. Treat it as experimental until we add value probes against the GPU O-proj path.

## 4. Combined stack

Current all-three profile:

```bash
DS4_FLASH_MOE_ANE_SHARED_EXPERT=1
DS4_SHARED_EXPERT_ANE_I8I8=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_CONV=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_INT8=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_I8X=1
DS4_FLASH_MOE_ANE_OPROJ_BATCH=512
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_THREADS=1
DS4_FLASH_MOE_ANE_OUTPUT_PROJ_GPU_CVT=1
./run_ane_prefill_profile_m5max.sh
```

Measured result:

| Log | Config | Prefill |
|---|---|---:|
| `m5m_report_all3_stream_sharedi8_oproji8x_b512_20260523_080639` | Routed streaming ANE + shared i8i8 + O-proj i8x B512 | `244.31 t/s` |
| `m5m_STACK_SHARED_OPROJ_003419` | Older routed ANE + fp16 shared + O-proj i8w/fp16x B256 | `256.29 t/s` |
| `m5m_GPU_SHARED_OPROJ_B512_004013` | GPU-routed experts + fp16 shared ANE + O-proj i8w/fp16x B512 | `280.07 t/s` |

Fresh all-three timing breakdown:

```text
routed ANE eval:        35435.644 ms total, 5500 eval calls, 6.443 ms/call
shared i8i8 eval:        2573.636 ms total, 43 calls, 59.852 ms/call
shared join wait:           0.441 ms total, 0.010 ms/call
O-proj i8x eval:         5327.205 ms total, 43 calls, 123.888 ms/call
O-proj join wait:       15761.866 ms total, 366.555 ms/call
shared prewarm:          7384.8 ms
O-proj prewarm:         39852.7 ms
```

Interpretation:

- Shared i8i8 overlaps well. Its main-thread join cost is essentially zero in the combined run.
- O-proj does not overlap well. It serializes the layer pipeline and dominates the combined slowdown.
- This is now guarded in code: O-proj is kept on GPU whenever routed-ANE prefill is active, unless `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1` is explicitly set.
- The `GPU_SHARED_OPROJ` run is useful as a hybrid datapoint, but it is not evidence that O-proj alone is the largest win.
- The routed ANE scheduler assigned only larger groups because `DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=384`; pad utilization is good (`91.97%`) but the stack still loses due to O-proj.
- The three optimizations are not additive today. Best production direction is to keep streaming routed ANE separate from O-proj until O-proj scheduling is fixed.

## Recommended next steps

1. Keep current production speed candidate as routed streaming ANE only:

```bash
./run_ane_prefill_profile_m5max.sh
```

2. Use shared i8i8 only for experiments:

```bash
DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 \
DS4_SHARED_EXPERT_ANE_I8I8=1 \
DS4_FLASH_MOE_MPP_INT8_QSCALE=512 \
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32 \
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32 \
./run_gpu_prefill_profile_m5max.sh
```

3. Do not enable O-proj by default yet. The next O-proj task should be overlap, not kernel TFLOPs:

- Start O-proj earlier, if dependencies allow.
- Avoid waiting at the immediate consumer boundary.
- Consider splitting O-proj into smaller work units only if that creates real overlap.
- Add per-layer timeline labels for O-proj `dep_wait`, `eval`, `output`, and `join` to make the bubble visible.

4. Add correctness harnesses before treating shared/O-proj ANE as production:

- 100-token deterministic divergence comparison.
- A/B divergence run with `DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES=0` versus `1` on the same prompt to catch NAX/MPP tail-tile drift.
- Per-layer shared-expert tensor comparison GPU vs ANE.
- Per-layer O-proj tensor comparison GPU vs ANE.
- Scale sweep on real activation ranges, not only synthetic MLP inputs.

## Log index

Key logs used:

- `moe-batch-bench/profile_runs/m5max_gpuonly_async_prefetch1_afterstage.summary.txt`
- `moe-batch-bench/profile_runs/m5max_ane_b256_tok100_s48.summary.txt`
- `moe-batch-bench/profile_runs/m5m_shared_gpu_20260523_073906.summary.txt`
- `moe-batch-bench/profile_runs/m5m_shared_ane_i8i8_w512_x32_mid32_20260523_075059.summary.txt`
- `moe-batch-bench/profile_runs/m5m_GPU_OPROJ_I8X_B512_005348.summary.txt`
- `moe-batch-bench/profile_runs/m5m_report_all3_stream_sharedi8_oproji8x_b512_20260523_080639.summary.txt`
