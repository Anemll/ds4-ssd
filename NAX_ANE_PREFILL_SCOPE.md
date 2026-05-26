# Layer-wise prefill — NAX / ANE scope review

Goal: inventory the prefill computations that process the **full token batch** (multiple
hidden states), and for each decide the implementation ladder **GPU baseline → fp16-NAX →
W8A8-NAX → ANE**. ANE matters because antirez uses **zero ANE** — it is extra compute we have
and he doesn't (the only credible beyond-parity lever; see `memory/nax-kernel-tuning-playbook.md`).

## Implementation order (LOCKED, per candidate)
For each flagged candidate, implement + validate in this order, each against the GPU-baseline number:
1. **fp16-NAX** (matmul2d) — get it correct + measured first.
2. **W8A8-NAX** (int8) — only "if applicable" (needs sizable K and the right shape; e.g. HC's N=24 may
   not benefit, lm_head quality-sensitive → maybe stays fp16/NAX).
3. **ANE** — last in build order (so GPU NAX/W8A8 reference numbers exist), but **implemented
   UNCONDITIONALLY — do NOT drop it just because it loses to NAX on M5 Max.** Hardware-dependent:
   M5 Max has 1 ANE; **M3 Ultra has a DUAL ANE cluster (two UltraFused dies → ~2× ANE throughput)**, so
   an ANE path that's slower than NAX on M5 can be *faster* than NAX on M3U. Decision criterion is the
   **target-hardware spectrum (esp. M3U), not the M5 NAX-vs-ANE delta.** Keep it env-gated + per-host
   (cf. `run_ane_prefill_profile_m3u.sh` / `_m4pro.sh`); the real lever is still overlap/bubble cost.
Bench each rung clean (fans max, battery healthy, matched pairs) and require net-positive e2e, not just
microbench (dilution rule). Bench ANE on BOTH M5 Max and M3 Ultra before judging.

Method: every batched matmul has a CPU impl (`--cpu` backend / reference, e.g. `matmul_q8_0_batch`)
and a metal-graph impl (the `batch_*` path used by `--metal`). The metal graph is the live target.
Per-stage cost is isolated via `DS4_METAL_LAYER_STAGE_PROFILE=1` (syncs at each boundary → absolute
ms inflated but comparable across configs). Config ladder by env flag:
- GPU baseline (simdgroup `kernel_mul_mm_q8_0_f32`): no flags
- fp16-NAX: `DS4_GPU_DENSE_NAX=1`
- W8A8-NAX: `DS4_GPU_DENSE_I8=1` (W8A8 ≥4096 tok, fp16-NAX below — cutoff `DS4_GPU_DENSE_I8_MIN_TOK`)
- ANE: per-component flag (shared expert: `DS4_FLASH_MOE_ANE_SHARED_EXPERT=1`)

---

## Layer-wise prefill stage map — GPU / NAX / W8A8 / ANE

Per-layer stages from the live profiler (`DS4_METAL_LAYER_STAGE_PROFILE`). M = n_tok (≤ chunk, up to 8k).
Status legend: ✅ live (env-gated) · ◇ scaffold/dormant · — none · n/a not a weight-GEMM.

| # | Stage (profiler label) | Type / shape (M=n_tok) | GPU base | fp16-NAX | W8A8-NAX | ANE | Candidate? |
|---|---|---|---|---|---|---|---|
| A | `norm` (attn/ffn RMS) | elementwise | ✅ | n/a | n/a | n/a | no |
| B | `q_path` → `q_a` | GEMM [4096→q_rank] Q8_0 | ✅ simd | ✅ | ✅ | — | **yes** |
| C | `q_path` → `q_b` | GEMM [q_rank→q_heads] Q8_0 | ✅ simd | ✅ | ✅ | — | **yes** |
| D | `q_path` → `rope`/`*_norm` | elementwise/rope | ✅ | n/a | n/a | n/a | no |
| E | `kv_path` (kv_a) | GEMM [4096→kv_cmp] Q8_0 | ✅ simd | ✅ | ✅ | — | **yes** |
| F | `compressor` (comp_kv) | GEMM, small-M tail | ✅ | small | small | — | marginal |
| G | `indexer_setup` | GEMM, n_comp-gated | ✅ | ◇(`INDEXER_NAX`) | — | — | marginal |
| H | `inv_rope` | elementwise | ✅ | n/a | n/a | n/a | no |
| I | `attention` (flash QKᵀ/·V) | O(n²) flash-attn | ✅ TC-internal | — | — | — | in-kernel |
| J | `output_proj` (O-proj) | grouped GEMM [..→4096] Q8_0 | ✅ simd | ✅ | ✅ | ◇ `ane_output_proj` | **yes** |
| K | `hc_pre`/`hc_post` (mix) | GEMM [16384→24] F16 | ✅ simd-f16 | — (measure) | likely no (N=24) | — | **measure** |
| K2| hc combine/split | elementwise (sinkhorn/wsum) | ✅ | n/a | n/a | n/a | no |
| L | `router` | GEMM [4096→N_EXP] F16 small-N | ✅ simd-f16 | small | — | — | marginal |
| M | `shared_gate_up` | GEMM [4096→2048]×2 Q8_0 | ✅ simd | ✅ | ✅ | ◇ `ANE_SHARED_EXPERT` | **yes (flagged)** |
| N | `shared_down` | GEMM [2048→4096] Q8_0 | ✅ simd | ✅ | ✅ | ◇ (part of shared ANE) | **yes (flagged)** |
| O | `routed_moe` (experts) | per-expert GEMM, dedup/paged | ✅ | (dedup kernels) | (int8?) | ◇ `routed..banked_ane` | **yes (largest)** |
| P | output `lm_head` | GEMM [4096→129280] F16 | ✅ simd-f16 | — | — (quality) | — | **yes if all-token** |

Notes:
- GPU baseline for Q8_0 GEMMs = simdgroup `kernel_mul_mm_q8_0_f32`; for F16 = simdgroup f16. fp16-NAX
  (`DS4_GPU_DENSE_NAX`) and W8A8 (`DS4_GPU_DENSE_I8`, ≥4096 tok) live for B/C/E/J/M/N via
  `ds4_gpu_matmul_q8_0_tensor_ex`.
- **Measured rung-1 GPU baseline** (ctx4096, n_tok≈3970): `shared_gate_up ≈ 10 ms/layer`,
  `shared_down ≈ 8 ms/layer`.
- ANE scaffolds present but dormant: M/N (shared expert), J (O-proj), O (routed banked). lm_head (P) and
  HC (K) have no NAX/W8A8/ANE yet.
- `lm_head` (P): all-token only in spec/MTP prefill (`spec_logits`, 13798); last-token-only in normal
  prefill. Biggest single GEMM when all-token (~8.6 TFLOP @8k).

**Coverage gaps to implement (NAX → W8A8 → ANE order):** B/C/E/J/M/N already have GPU NAX+W8A8 (need ANE
on J/M/N — partly scaffolded). **lm_head (P)** and **HC mix (K)** have NO NAX yet → start there per the ladder.
**routed_moe (O)** is the largest slice — review its current NAX/int8 status next.

---

## Inventory (batched-over-tokens prefill ops)

### 1. Shared-expert MLP — **FLAGGED FOR IMPLEMENTATION**
- Shape: gate/up `[4096→2048]`, down `[2048→4096]`, SwiGLU; batched over `n_tok` (full chunk 4096).
- Live metal site: `ds4.c:16483–16524` (`batch_shared_gate/up/out` via `ds4_gpu_matmul_q8_0_tensor_ex`,
  `PREFER_I8`). CPU/reference: `layer_shared_ffn_batch` (`ds4.c:5705`) → `matmul_q8_0_pair_batch` /
  `matmul_q8_0_batch` (marked `TODO(ANE-OPT)`).
- Existing rungs: GPU NAX + W8A8 live (env-flag toggle). **ANE scaffold already present** but dormant:
  `DS4_FLASH_MOE_ANE_SHARED_EXPERT` → `ds4_gpu_shared_expert_ane_async_start/finish_tensor`
  (`ds4_metal.m:6760`) — full-batch, async (overlaps shared→ANE with routed→GPU). ANE microbench:
  `moe-batch-bench/ane_ds4_mlp_int8w.m`.
- **Rung-1 GPU baseline measured** (2026-05-25, simdgroup, ctx4096, n_tok≈3970, stage profiler):
  `shared_gate_up ≈ 10 ms/layer`, `shared_down ≈ 8 ms/layer` (~18 ms/layer total). This is the
  number the NAX / W8A8 / ANE rungs must beat.
- TODO(impl): complete the ladder GPU→NAX→W8A8→ANE on this shape; then evaluate concurrency
  (shared→ANE ∥ routed→GPU vs token-split shared across NAX+ANE). Open question that decides
  token-split value: **is the GPU idle (SSD-stalled) during the routed-expert phase?**

### 2. Attention MLA projections — TO REVIEW
- `batch_qr` (q_a), `batch_kv_raw` (kv_a), `batch_q` (q_b): `ds4.c:14972/14986/15029/15073`,
  batched, live on GPU NAX/W8A8 (`_ex PREFER_I8`). No ANE path yet. CPU ref: `matmul_q8_0_batch`
  (7752/7760/7770, marked).

### 3. Attention output (O-proj, grouped) — TO REVIEW
- `ds4_gpu_attention_output_q8_batch_tensor` (grouped, batched). W8A8 live. Prior ANE work:
  `moe-batch-bench/OPROJ_ANE_INVESTIGATION.md` (conclusion: overlap bubbles are the hard part, not
  kernel TFLOPs).

### 4. Routed experts (MoE, dedup/paged) — TO REVIEW (largest slice)
- `layer_routed_moe_batch` (`ds4.c:6027`), GPU dedup kernels (`metal/moe.metal`
  `kernel_flash_moe_dedup_*`). Per-expert batched over compacted token lists. GPU + SSD-streaming.
  The dominant prefill cost and the consumer the shared-expert ANE overlaps with.

### 5. Output projection / lm_head — **CANDIDATE (biggest matmul when all-token)**
- Two modes: *normal* autoregressive prefill = last token only (`output_logits_one`, n_tok=1, N/A);
  **speculative/MTP prefill computes `spec_logits` over ALL n_tokens** (`ds4.c:13798`):
  `[n_tok × DS4_N_EMBD=4096] @ [4096 × DS4_N_VOCAB=129280]` ≈ **8.6 TFLOP @ 8k tokens** — the single
  largest matmul, dwarfing the shared expert. Sizable M, K, AND N → ideal NAX/ANE target when active.
  No ANE on lm_head yet. (Decide first: does our prefill/bench path use the all-token spec/MTP logits
  or last-token-only? That gates whether this matters.)

### 6. Hyper-connections (HC) — **MEASURE (undersold earlier)**
- `hc_attn_fn`/`hc_ffn_fn` (F16): `[n_tok × hc_dim=16384] @ [16384 × hc_mix_dim=24]`. In layer-wise
  prefill M=n_tok (up to 8k) and K=16384 are BOTH large → ~6 GFLOP/layer of real work — NOT negligible.
  Only weak spot: N=24 caps the GPU tensor-core N-tile (24/32 ≈ 75% fill); on ANE 24 output channels is
  fine and the cost is reading the n_tok×16384 activations. => measure NAX/ANE here, don't dismiss on
  out-width. Currently on the plain f16 path (`ds4_gpu_matmul_f16_tensor`).
- HC combine/split (`hc_split_sinkhorn`, `hc_weighted_sum`): elementwise/reduction across N_HC=4 streams
  — genuinely not matmuls; not NAX/ANE.

### 7. Attention score math (flash-attention) — tensor-core, but inside the fused kernel
- QKᵀ/softmax/·V over the KV: O(n_tok²) — at 8k the scores are ~8k×8k per head, the dominant attention
  cost. Flash-attn (`ds4_gpu_flash_attn_*`) already drives tensor cores internally for QKᵀ/PV. Tensor
  work, but not a separable weight-GEMM to offload to NAX/ANE — optimize within the flash kernel.

### 8. Indexer / router / compressor — TO REVIEW
- `ds4_gpu_matmul_f16_tensor` sites. Re-check with the layer-wise (M=n_tok) lens before judging —
  same mistake to avoid: small per-token out ≠ small matmul.

### Existing ANE batch hooks (process all n_tokens, dormant)
- Shared expert: `DS4_FLASH_MOE_ANE_SHARED_EXPERT` (`ds4_gpu_shared_expert_ane_async_*`).
- Attention O-proj: `ane_output_proj_enabled_for_run` (`ds4_gpu_oproj_ane_async_*`, 16145) — noted
  "large serial join cost" = the overlap-bubble problem (`OPROJ_ANE_INVESTIGATION.md`).

---

## Rule of thumb (CORRECTED)
Judge by **total matmul work in layer-wise prefill (M·K·N with M = token count, up to 8k)**, NOT by
per-token output width. By that measure the candidate ranking, roughly by size:
**all-token lm_head/output-proj (if spec/MTP active) >> routed experts > shared expert ≈ attention
projections > HC mix (real M·K work; N=24 caps GPU tile, fine on ANE).** Only genuinely excluded:
HC combine/split (elementwise) and attention scores (tensor work but inside the flash kernel).

---

## Measured per-stage ladder (ctx4096, ms/layer, clean batt 100% fans max)
| stage | baseline | NAX | W8A8 | note |
|---|---|---|---|---|
| q_path | 14.17 | 6.85 | 6.83 | NAX 2.1× |
| output_proj | 20.58 | 7.52 | 7.58 | NAX 2.7× |
| shared_gate_up | 5.22 | 2.07 | 2.08 | NAX 2.5× |
| shared_down | 3.85 | 2.28 | 2.27 | NAX 1.7× |
| compressor | 5.43 | 5.48 | 5.45 | not NAX'd (small) |
| **routed_moe** | **61.93** | **62.54** | **62.60** | **dominant, NAX=tie (prior finding), partly SSD-bound** |
- **W8A8 ≈ NAX** everywhere (int8 gives ~0 over fp16-NAX) → W8A8 stage not worth much beyond done.
- **routed_moe dominates (~62ms/layer, ~3× rest)** and is NOT NAX-accelerated; lever = concurrent GPU+ANE
  (`ANE_INT8_COMBINED_BENCH_PROCEDURE.md`), not a fresh NAX kernel.

## Correctness — PASS
100-token layer-wise prefill (`tests/layerwise_prefill_100.txt`) through Dedup-MoE + shared expert: baseline /
NAX / W8A8 produce **byte-identical greedy output** (`ds4 --temp 0 -n 48`). Dense NAX/W8A8 numerically correct.

## Prior ANE findings (don't redo — from moe-batch-bench docs)
- **DUAL_ANE_CLUSTER (M3U): GPU-only 220.5 → dual-cluster ANE 270.3 t/s = +22.6%** on DSv4 IQ2_XXS @8.4k.
  THE validated beyond-parity win. ANE shared-expert offload (i8i8 tiled-fused) ~1.97× across 2 clusters.
- M5 Max single-ANE: shared-expert ANE move ~+6 t/s only; the wall is **GPU command-encode (~95% CPU-side,
  caps ~277 t/s)** so ANE only helps by SHEDDING GPU work, and faster ANE (conv mode 9, 2.6×) gives no e2e win.
- `FUSED_MLP_ANE_CONV`: conv variants off by default (don't win this profile).

## Open optimization hypotheses (flagged)
- **Big-chunk shared expert for REGULAR (resident) prefill** (user 2026-05-25): layer-wise prefill currently
  the SSD/Flash-MoE path; the batched NAX shared-expert could also speed the *resident* (non-SSD) case if tuned
  for bigger chunks. `ffn_batch=128` default (ds4.c:8305) and a `resident_moe_mpp_dedup_prefill` path exists →
  investigate: does resident prefill use the layer-wise NAX path, and does raising the chunk/batch improve
  shared-expert tensor-core M utilization enough to beat the regular path? (NAX shared already 2.5× @4096.)

## Status
- Dense projections (q/kv/shared/O-proj): **NAX done + validated (2–2.7×, byte-correct)**; W8A8 marginal.
- routed_moe: NAX=tie; lever is concurrent GPU+ANE (combined path).
- ANE: shared-expert validated on **M3U (+22.6%)**, small on M5; no ANE skill installed (use ane-lowering-plan
  + ds4 ANE procedure docs). ANE ships regardless (M3U dual-cluster).
- Remaining: big-chunk/resident hypothesis; routed-MoE concurrent-ANE integration; M3U re-validation.
