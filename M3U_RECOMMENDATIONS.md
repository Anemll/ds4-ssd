# M3 Ultra validation recommendations — dense layer-wise prefill (NAX / W8A8 / ANE)

Autonomous M5-Max investigation (2026-05-25). M5 Max = 1 ANE; **M3 Ultra = dual ANEx16 cluster**, and prior
work already showed dual-cluster ANE shared-expert = **+22.6% prefill (220→270 t/s)** on DSv4 IQ2_XXS @8.4k
(`moe-batch-bench/DUAL_ANE_CLUSTER_OPTIMIZATION.md`). These are the configs to (re)validate on M3U, with the
M5 evidence behind each. All benches: fans max, battery healthy, interleaved/matched pairs, stage profiler
(`DS4_METAL_LAYER_STAGE_PROFILE`) for per-stage. Correctness via `tests/layerwise_prefill_100.txt` (greedy,
byte-identical to baseline).

## Confirmed on M5 (carry to M3U)
- **Dense projection NAX** (`DS4_GPU_DENSE_NAX=1` or `DS4_GPU_DENSE_I8=1`): per-stage 1.7–2.7× (q 2.1×,
  O-proj 2.7×, shared 2.5×/1.7×). Byte-identical output. **Recommend: validate as a prefill default on M3U.**
- **W8A8 ≈ fp16-NAX** on M5 (int8 ~0 gain over fp16). Re-check on M3U but expect same → prefer fp16-NAX
  default (simpler, no int8 weight cache).
- **W8A8 cutoff fix** (commit `0d77e43`): W8A8 only ≥4096 tok, sub-cutoff → fp16-NAX. Recommend keep.
- **int8 weight-cache release** at prefill→decode (`87f1d48`): memory hygiene, decode-neutral on M5
  (headroom); **may help M3U/constrained more.**

## To finalize this session
- [x] Big-chunk: NAX ms/token drops ~10–30× from small→4096 chunks → keep chunks ≥4096 (see below).
- [x] ANE shared-expert on M5: correct but −24% (M5 single ANE); WIN on M3U → per-host gate.
- [x] NAX default-ON flip (4858878): +20–24% e2e, byte-correct, M3U-safe fallback.
- [x] Final standing vs antirez: NAX −4.8%, W8A8 −1.9% (residual = fork dispatch overhead, finding #1).
- [ ] routed-MoE concurrent-ANE: M5 would lose (single ANE); M3U lever — validate on M3U.
- [ ] Finding #1 (fork dispatch overhead, −4.8%): localize ds4-ssd vs clean-ds4 host/dispatch. IN PROGRESS.

### ANE shared-expert on M5 — CORRECT but −24% (don't enable on M5)
`DS4_FLASH_MOE_ANE_SHARED_EXPERT=1`: functional (per-layer fp16w split-matmul init ~115ms one-time), greedy
output **byte-identical** to baseline. But prefill: ctx4096 339.6→254.9 (**−24.9%**), ctx8192 348.4→268.9
(**−22.8%**). Single M5 ANE can't beat the GPU NAX shared expert (~2ms/layer) and adds a join bubble.
**Confirms the hardware split: ANE shared LOSES on M5 (1 ANE), WINS on M3U (2 ANE, +22.6% prior).**
**Recommendation: ANE shared-expert OFF on M5, ON on M3U — per-host gate is mandatory, not optional.**

### ✅ NAX default-ON flip — DONE (commit 4858878), the biggest win
Flipped `DS4_GPU_DENSE_NAX` to default-ON when the matmul2d lib compiles (M5+; M3U/M4 fall back to simdgroup,
no crash). Validated clean (fans max, batt 100%):
- e2e prefill vs old simdgroup default: **+20.3% @4096, +24.0% @8192** (and small-ctx +5.4/+10.9/+15.9/+17.9%
  @256/512/1024/2048 — wins at EVERY size, no n_tok floor needed).
- Greedy 100-tok output **byte-IDENTICAL**; decode unaffected.
- `DS4_GPU_DENSE_NAX=0` restores old path; W8A8 (`DS4_GPU_DENSE_I8=1`) opt-in for +1–3% more.
**M3U action:** matmul2d is M5-only, so on M3U this auto-falls-back to simdgroup — the M3U prefill win must come
from **ANE (dual cluster)**, not GPU NAX. Verify the fallback path is clean on M3U.

### (historical) NAX was DEFAULT-OFF — likely the biggest latent win
`DS4_GPU_DENSE_NAX` / `DS4_GPU_DENSE_I8` both default OFF (no `setenv` anywhere); antirez's NAX is default-ON.
So our **default** prefill runs the slow simdgroup path on q/kv/shared/O-proj while NAX (2–2.7×/stage) sits
dormant. **Recommend: make `DS4_GPU_DENSE_NAX` default-ON when MPP is available (M5+), W8A8 opt-in** (W8A8≈NAX,
fp16-NAX is cleaner: no int8 cache, no quality risk). Decode unaffected (n_tok=1 → GEMV, NAX never fires).
[pending clean e2e default-vs-NAX confirmation before flipping]

## M3U-specific (the real wins live here)
- **Dual-cluster ANE shared-expert** (+22.6% prior): re-validate at current code; the lever is overlap
  (shared→ANE ∥ routed→GPU), not kernel TFLOPs. Wall = GPU command-encode (~277 t/s cap on M5).
- ANE ships regardless of M5 NAX-vs-ANE (per directive): M3U dual cluster flips the calculus.

## Final M5 standing (new default vs antirez, clean, prefill t/s)
| ctx | antirez | ours default(NAX) | ours +W8A8 | NAX Δ | W8A8 Δ |
|---|---|---|---|---|---|
| 8k | 431.9 | 416.9 | 429.1 | −3.5% | −0.7% |
| 16k | 419.1 | 399.1 | 410.3 | −4.8% | −2.1% |
| 32k | 394.8 | 374.7 | 384.4 | −5.1% | −2.7% |
| 64k | 357.1 | 336.4 | 344.1 | −5.8% | −2.2% |
| mean | | | | **−4.8%** | **−1.9%** |
- The default flip moved us from ~−20–34% (simdgroup default) to **−4.8% (NAX default)** vs antirez.
- **W8A8 opt-in (`DS4_GPU_DENSE_I8=1`) → −1.9%** (near parity). Decode ~−2% (fork-side, both).
- Residual −4.8%/−1.9% = **fork dispatch/host overhead (finding #1, still OPEN)** — not in the kernels
  (cleanroom = antirez with same kernels). Next frontier: diff ds4-ssd host/dispatch vs clean ds4.

## Finding #1 (the −4.8% NAX residual vs antirez) — LOCALIZED
NOT the kernels-are-different story and NOT a dtype bug (checked: both our `ds4_dense_q8_nax` and antirez's
`kernel_mul_mm_q8_0_f32_nax_direct_rhs` are float×half for q8 — the `[[dense-nax-findings]]` "half×half" note is
about his F16 path, not q8). The gap is **NAX kernel TUNING**:
- antirez ships **n128 / n64 / n32 token-tile variants** and picks by `n_tok%128`/`%64` (best tile per chunk).
  Ours is **fixed NR1=128/NR0=64/NK=32** → non-128-aligned (tail/remainder) chunks pad to 128 wastefully.
- Likely also NR0/NK/simdgroup tuning differences vs his internal constants.
**Fix (next session, use nax-kernel-tuning skill + `nax_autotune.m`):** add n64/n32 variants + autotune NR0/NK.
Expected to recover much of the −4.8%. NOT attempted unattended (kernel surgery, risk). **W8A8 sidesteps this
(−1.9%, our well-tuned int8 kernel)** → the pragmatic near-parity path is W8A8, not more NAX tuning.

## Recommendation hierarchy (M5)
1. **Ship: NAX default-ON** (done, 4858878) — safe, byte-correct, +20–24% over old default.
2. **Recommend W8A8 opt-in** for the extra ~3% (−1.9% vs antirez) AFTER a wider perplexity eval confirms the
   0.7% int8 weight-quant drift is benign (byte-identical on the spot-checks so far). Don't make it the
   unattended default until that eval. Cache-release fix (87f1d48) handles its memory.
3. Keep cutoff fix (W8A8≥4096, fp16-NAX below) + cache-release.

## Follow-up task (CONSTRAINED — do NOT start until all NAX / W8A8 / ANE work below is complete)
Review the **`SSD-prefetch-ANE`** branch (M3 Ultra was working on it) and **merge it with these changes**.
Rationale: my changes carry higher signal because the NAX path is validated here (correctness byte-identical,
per-stage 2–2.7×). Order: finish ALL NAX + NAX-int8(W8A8) + ANE validation/recommendations first, THEN
reconcile/merge. Until then: leave SSD-prefetch-ANE untouched. (User directive 2026-05-25.)

## Open results

### Big-chunk hypothesis — CONFIRMED (NAX efficiency scales hard with chunk size)
NAX per-token cost drops ~10–30× from small chunks to 4096-token chunks (M5, NAX config, stage profiler):
- tok=4096 chunk: q_path ~3.2, shared_gate_up ~0.9, shared_down ~1.1, hc_pre ~0.67, hc_post ~0.46 ms/1k-tok.
- small chunk (~100s tok): ~10–30× higher per token (fixed dispatch overhead + poor tensor-core M fill).
**Implications / recommendations:**
- Keep prefill chunks **≥4096** everywhere NAX runs. The GPU metal prefill already chunks at ~4096 (bulk
  efficient; only the remainder/tail chunk is small/inefficient — minor).
- **HC mix DOES benefit from big chunks** (hc_pre 0.67 vs ~13 ms/1k-tok small) — earlier "HC won't benefit"
  was wrong; the per-call overhead amortizes even with N=24 (absolute HC cost still small though).
- **M3U action:** confirm the resident/regular (non-SSD) prefill path also uses ≥4096-token NAX chunks for the
  shared expert + projections; if it currently uses smaller batches (e.g. CPU `ffn_batch=128` /
  `DS4_PREFILL_BATCH`), routing it through the big-chunk layer-wise NAX path is a large win. **This is the core
  of the "faster for regular too" hypothesis — TEST ON M3U with the resident model.**
