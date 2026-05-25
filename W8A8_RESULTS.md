# W8A8 (int8) prefill optimization — final results

M5 Max, DeepSeek-V4-Flash resident model (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`).
Branch `codex/integrate-ds4-agent`. int8 weight × int8 activation (W8A8) for the Q8_0 dense matmuls in prefill.
Commits: `f07d649` (dense), `2c4230c` (hints), `94ac135`+`8ff7586` (attn_out), `3702ce8` (fusion profiler).

---

## 1. Gates (env flags + dispatch policy)

All **default OFF** — shipped behavior unchanged unless enabled.

| env flag | effect |
|---|---|
| `DS4_GPU_DENSE_I8` | master W8A8 on/off (dense projections, shared expert, attn_out O-proj) |
| `DS4_GPU_DENSE_NAX` | float×half NAX path (attn_out fp16-NAX fallback; dense W8A8 takes precedence when DENSE_I8 on) |
| `DS4_GPU_INDEXER_NAX` | indexer-scores NAX (n_comp≥3072) |
| `DS4_GPU_INDEXER_RELAXED/_WALK`, `DS4_GPU_DENSE_WALK`, `DS4_GPU_ATTN_NR1/NK` | tuning knobs (all e2e-neutral; default off) |
| `DS4_METAL_FUSION_PROFILE` | per-token/chunk fusion-boundary counters (debug) |

**Per-call-site precision policy** (`ds4_mm_hint`, baked at the call site — `ds4_gpu_matmul_q8_0_tensor_ex`):
- `PREFER_I8` → MLA projections (q_a/q_b/kv_a), shared-expert (gate/up/down): W8A8 (cutoff lowered 32→16).
- `NO_I8` → **lm_head** (final logits): kept in **half** (most quality-sensitive; ~free, n_tok=1 anyway).
- `AUTO` → everything else (env-gated W8A8 when eligible).

**Dispatch size-ladder** (`ds4_gpu_matmul_q8_0_tensor_ex`), eligibility `in_dim%128==0 && out_dim%32==0`:
| n_tok | path |
|---|---|
| 1 | GEMV (`kernel_mul_mv`) — regular |
| 2–8 | `mul_mv_ext` — regular |
| ≥16/32 | **W8A8** (`ds4_dense_i8_fused`) when DENSE_I8 + eligible |
| else | W16A16 NAX (`ds4_dense_q8_nax`) → simdgroup `kernel_mul_mm_q8_0_f32` fallback |

**Kernels** (`metal/nax_fused.metal`): `ds4_repack_q8_to_i8_rowscale` (Q8_0→int8+per-row scale, load-time, cached
by weight_offset), `ds4_quant_act_pertoken_i8` (f32→int8+per-token scale), `ds4_dense_i8_fused`
(NR1=128/NR0=32/NK=128, int8×int8→int32, fused per-(row,token) rescale), `ds4_attn_out_low_i8_fused` (grouped).
**Crossover** (measured): W8A8 beats W16A16 NAX at every n_tok (1.7× @16 … 1.4× @4096) → W16A16 is superseded.
W8A8 needs NK≥128 + offline weight repack (NK=32 or per-dispatch dequant erase the win).

---

## 2. Benchmark comparison

### W8A8 on vs off (same binary — the isolated win), prefill t/s, adjacent pairs
| ctx | int8 OFF | int8 ON | Δ |
|---|---|---|---|
| 8k | 265/277 | 312/337 | **+19.6%** |
| 16k | 244/269 | 297/315 | **+19.4%** |
(replicated; matches the +15–20% measured across 8k–32k.)

### vs antirez (his fp16-NAX default), all sizes — both built+run from own dirs, interleaved adjacent pairs
Clean run (gen-8, flat — the reliable one):
| ctx | 8k | 16k | 24k | 32k | 40k | 48k | 56k | 64k | mean |
|---|---|---|---|---|---|---|---|---|---|
| prefill Δ | +0.6 | +1.7 | −0.1 | −0.1 | +0.4 | +0.6 | +0.3 | +0.5 | **+0.5%** |

Full-stack run incl attn_out (gen-32, thermally noisy → per-ctx bimodal; means): prefill **+2.5%**, gen **−0.4%**.

### Our version — absolute prefill t/s (clean gen-8 means; the target the cleanroom must hit/beat)
| ctx | 8k | 16k | 24k | 32k | 40k | 48k | 56k | 64k |
|---|---|---|---|---|---|---|---|---|
| **ours t/s** | 359 | 341 | 321 | 307 | 296 | 283 | 271 | 260 |
| antirez t/s | 357 | 336 | 321 | 307 | 295 | 281 | 270 | 259 |

(Peak across all runs is higher — ~415 t/s @8k — but thermally inflated; the table above is the reliable mean.)
**Live readout format** (ds4 / ds4-agent prints this during prefill — the on-screen t/s "as in our version"):
`ctx 51.9k/100k | prefill [▶▶▶▶▶▶▶▶························] 364/1305 27.9% 253.4 t/s`
The cleanroom's **final delivery** must show this readout (long-context run, e.g. `-c 100000`) for clean-main vs
clean-main+W8A8, so the on-screen prefill t/s is directly comparable to our version's.

**Verdict: PARITY with antirez.** int8 moved the fork from ~−5% behind → level, on the same pure-GPU path
(antirez uses **no ANE**). **Generation unchanged** by W8A8 (decode is n_tok=1 GEMV — int8 doesn't fire); the small
gen Δ is noise/possible fork overhead (to be localized by the cleanroom comparison).

### attn_out O-proj (microbench, its shape)
W8A8 grouped **39703 vs 27252 GF/s** float×half = **+1.46×** (small prefill slice → ~+1% e2e, didn't move the standing).

---

## 3. Quality comparison

- **Generation coherence**: with the full int8 stack on (dense + attn_out, lm_head half), greedy temp-0 output is
  **token-identical to baseline** (binary-search prompt → same "...repeatedly dividing the search interval in half…
  O(log n)…"). No NaN/garbage. Confirmed for dense-only and dense+attn_out.
- **Microbench relative error** (vs exact float dequant dot): **dense 0.7%**, **attn_out 0.6%** (per-row weight scale +
  per-token activation scale; synthetic). Below any quality-relevant threshold for mid-network projections.
- **lm_head kept in half** (`NO_I8`): the final logits are the most quality-sensitive matmul; left fp at ~no speed
  cost (lm_head runs at n_tok=1 in decode / final token in prefill).
- **Recommended before default-on**: a wider eval (perplexity / capability) on real prompts to confirm the 0.7%
  weight-quant drift is benign beyond the coherence spot-check — see the cleanroom task's per-kernel quality gate.

---

Full investigation, measured numbers, and gotchas: `memory/nax-kernel-tuning-playbook.md` and
`moe-batch-bench/M5M_ANE_OPTIMIZATION_REPORT.md`. Cleanroom antirez-PR task: `W8A8_CLEANROOM_TASK.md`.
