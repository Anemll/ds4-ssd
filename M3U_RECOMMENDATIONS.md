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

## To finalize this session (results filled as benches land)
- [ ] Big-chunk: does shared/q NAX ms/token drop with chunk size? → chunk-size recommendation.
- [ ] ANE shared-expert on M5 (`DS4_FLASH_MOE_ANE_SHARED_EXPERT=1`): correctness + perf.
- [ ] routed-MoE concurrent-ANE feasibility (`ANE_INT8_COMBINED_BENCH_PROCEDURE.md`).

### ANE shared-expert on M5 — CORRECT but −24% (don't enable on M5)
`DS4_FLASH_MOE_ANE_SHARED_EXPERT=1`: functional (per-layer fp16w split-matmul init ~115ms one-time), greedy
output **byte-identical** to baseline. But prefill: ctx4096 339.6→254.9 (**−24.9%**), ctx8192 348.4→268.9
(**−22.8%**). Single M5 ANE can't beat the GPU NAX shared expert (~2ms/layer) and adds a join bubble.
**Confirms the hardware split: ANE shared LOSES on M5 (1 ANE), WINS on M3U (2 ANE, +22.6% prior).**
**Recommendation: ANE shared-expert OFF on M5, ON on M3U — per-host gate is mandatory, not optional.**

### ⚠️ NAX is DEFAULT-OFF — likely the biggest latent win
`DS4_GPU_DENSE_NAX` / `DS4_GPU_DENSE_I8` both default OFF (no `setenv` anywhere); antirez's NAX is default-ON.
So our **default** prefill runs the slow simdgroup path on q/kv/shared/O-proj while NAX (2–2.7×/stage) sits
dormant. **Recommend: make `DS4_GPU_DENSE_NAX` default-ON when MPP is available (M5+), W8A8 opt-in** (W8A8≈NAX,
fp16-NAX is cleaner: no int8 cache, no quality risk). Decode unaffected (n_tok=1 → GEMV, NAX never fires).
[pending clean e2e default-vs-NAX confirmation before flipping]

## M3U-specific (the real wins live here)
- **Dual-cluster ANE shared-expert** (+22.6% prior): re-validate at current code; the lever is overlap
  (shared→ANE ∥ routed→GPU), not kernel TFLOPs. Wall = GPU command-encode (~277 t/s cap on M5).
- ANE ships regardless of M5 NAX-vs-ANE (per directive): M3U dual cluster flips the calculus.

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
