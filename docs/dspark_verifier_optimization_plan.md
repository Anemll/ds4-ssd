# DSpark Verifier Optimization Plan

## 2026-07-02 ACTIVE PLAN — Dense (non-MoE) verify stack, SGLang/vLLM port (strict byte-exact)

This is the current active priority. It supersedes the sections below. Goal
unchanged: DSpark strict byte-exact generation >55 t/s (cmp=0 vs no-draft;
relaxed-accept banned). All items here are byte-safe by construction (no change
to any FP32 reduction order); each ships behind an env flag and is promoted only
on an unfenced n=1000 A/B with `cmp=0`.

### Why the dense stack is the target (arithmetic)

Clean baseline this session (Pygame n=1000, frontier5): **43.77 t/s, verify
65 ms, tau 3.41, block 76.9 ms**. Verify decomposes (real, non-fenced) into:

- **Routed MoE ≈ 40 ms** — CONFIRMED FLOOR. `DS4_DSPARK_ROW_ROUTED_SUBPROFILE=1`
  shows ~1.5 ms/MoE-layer (gate+up+SwiGLU 1.2 ms >> down 0.3 ms). Two grouping
  variants (BATCH_CANONICAL down-dedup; GROUPED_EXACT+GROUPED_IQ2 gate/up) are
  both byte-exact but SLOWER. It is genuine IQ2_XXS dequant+bandwidth of the
  expert weights, model-inherent. Do not re-attempt MoE speedups.
- **Dense non-MoE ≈ 25 ms** — THE HEADROOM. Cutting it to ~9 ms gives
  verify 49 ms → block 62 ms → **55 t/s** at the current tau. Cutting it to ~0
  would give ~64 t/s. tau itself is draft-bound and flat across content
  (prose 3.41 / code 3.33 / json 2.87), so tau is not a lever without a retrain;
  the dense stack is the only remaining byte-safe path to 55.

### The unifying defect: dense stages still dispatch PER ROW

In batched (decode_order) verify, attention collapses to 1 dispatch/layer but
the compressor and indexer still dispatch per row (`compressor_dispatches +=
n_tokens`, `indexer_dispatches += n_tokens`, dspark.c:4617/4620). Flash = 60
layers, hidden 4096, compress ratios (ds4_expected_layer_compress_ratio):
layers 0-1 none; layers 2-59 compressor (ratio≠0); even layers 2-58 also
indexer (ratio 4). So per block (5 rows):

- Compressor: 58 layers × 5 = **290 per-row dispatches**
- Indexer: 29 layers × 5 = **145 per-row dispatches**

~435 tiny per-row dispatches/block whose launch+occupancy cost the fenced
profiler hides (fenced put compressor 2.2 ms / indexer 0.9 ms — almost certainly
understated). Unlike the MoE gate (per-EXPERT, needs route descriptors, grouping
lost), these are dense SINGLE-matrix ops that batch cleanly into one
5-row GEMV/kernel per layer — standard, high-occupancy, and byte-safe (each
row's latent-compression and top-k selection is independent, no cross-row
reduction).

### Targets (ordered by expected value)

| # | Stage | ~ms (rel) | SGLang/vLLM lever | Byte-safety | Notes |
|---|-------|-----------|-------------------|-------------|-------|
| 1 | HC head-compression pre (RMSNorm + fn GEMM) | ~7 | fuse RMSNorm→GEMM | likely — keep RMS sum-of-squares + GEMM dot order intact | fattest single dense bucket; hc_pre_from_state_one_scratch ds4.c:5292 (rms_norm_no_weight → matvec_f16); on-GPU path is hc_pre_rms + hc_pre_fn dispatches |
| 2 | Compressor (KV latent) | ~2–8* | batch per-row → per-layer 5-row GEMV | yes — per-row independent | 290 dispatches/block; *fenced understates; the per-row→batch win |
| 3 | Indexer (top-k select) | ~1–4* | batch rows + exact top-k fusion | yes — selection + per-row independent | 145 dispatches/block; index_query/score/topk/mask |
| 4 | Output projection + inv-RoPE + HC | ~4.6 | fuse output GEMM + RoPE | feasible | batch_hc_tail already fuses at ≤6 rows; check inv-RoPE + output_low are batched |
| 5 | Q projection | ~4.1 | fused QKV / weight layout | feasible | MLA: q down+up; check if q up-proj is per-row |
| 6 | Attention heads (flash) | ~3.9 | flash-attn kernel tuning | yes | flash_attn_ext_vec_rows5 family; NAX playbook knobs |

### Execution sequence

1. **Measure real per-stage cost first (unfenced).** For each stage add/enable a
   real end-to-end delta: baseline verify vs stage-batched verify (cmp=0 must
   hold). Do NOT rank off the fenced subprofile — use it only for relative hints.
   Fastest trustworthy probe: implement the batch and read the `dspark perf:`
   verify-ms delta at n=1000.
2. **Item 2 (compressor batch) and Item 3 (indexer batch) first** — highest
   dispatch-count reduction (435 → 87/block), cleanest byte-safety, and the
   stages the user flagged. Build a per-layer 5-row batched compressor GEMV +
   batched indexer (query/score/topk over 5 rows in one dispatch). Look for
   existing `metal_graph_encode_layer_attention_batch` per-row remnants.
3. **Item 1 (HC-pre norm→GEMM fusion)** — fattest bucket; fuse the RMSNorm into
   the following fn GEMM (single kernel, per-32 exact accumulation).
4. **Items 4–6** — output/Q projection fusion + flash-attn tuning; smaller,
   do after 1–3 land.

### Guardrails (from prior loop lessons)

- Unfenced runs only; fenced attn subprofile is a ~5× artifact — relative use
  only. Single `ds4` process (pgrep -x gate). 40–60 s cooldown; first-run-after-
  load is thermally penalized. Do not trust cross-run absolute deltas under
  thermal drift — interleave arms.
- Every promotion: `cmp -s` vs bench-results/loop55_phase0_135414/nodraft_n1000.out
  must be identical (byte-exact). Occasional 100-sample quality harness to be
  100% sure, not every run.
- Grouping that adds descriptor/route machinery lost twice on MoE — for dense
  single-matrix stages the batch is a plain wider GEMV, no descriptors, so it
  should win where MoE grouping didn't. Confirm empirically, don't assume.

### Realistic outcome

Items 1–3 target ~13–16 ms of the 25 ms dense budget → verify ~49–52 ms →
~52–55 t/s strict byte-exact. That is the honest shot at the goal without a
draft retrain or contract change. If 1–3 land byte-exact but net-neutral (the
per-row overhead turns out to be already L2/occupancy-hidden like the MoE case),
then the ~44 t/s ceiling is final and the decision reverts to retrain-or-relax.

### 2026-07-02 EXECUTION RESULTS — plan largely PRE-EXISTING; stack is tight

Ran the plan. Key correction: **the dense stack is already fully fused and
batched by default**, and the fenced per-stage estimates that seeded the table
above were wrong about *where* the compute is.

- **All dense fusions are default-ON** (flags DISABLE them, ds4.c:10425-10457):
  HC, KV, QKV-norm, HC-norm, shared-down-HC, attn-out-HC, compressor-pair.
- **Compressor row-batch is default-ON** (spec_row_local_attn_comp_rows,
  ds4.c:14321). The "per-row" I flagged was the pessimistic dispatch *estimator*
  (ds4.c:4614), not the real path. Plan item #2 was already done.
- **Indexer row-batch (item #3): tested, byte-exact but SLOWER** — verify
  65.5→67.8 ms with DS4_DSPARK_HYBRID_INDEXER_ROWS=1 + INDEX_COMP_ROWS=1
  (loop55_indexer_ab_110411). Off-by-default for good reason. 3rd batching
  dead-end (after MoE dedup + MoE gate/up): dispatch reduction is neutral —
  workload is compute/bandwidth-bound + serial, not dispatch-bound.
- **Real compute map** (fusion-disable Δblock, loop55_fusionmap_110859):
  attn-out-HC = **+14.6 ms** (dominant dense sink, ALREADY fused+batched+q8_0),
  compressor-pair +2.5, HC-norm +2.0, QKV-norm ~0. So the fenced table's
  "hc_pre ~7 fattest" was wrong; the attention-output→HC path is the fat and it
  is already optimal (batched inv-RoPE + out_low + out_b/hc_expand, ds4.c:17131).

**Conclusion:** every byte-safe structural lever (dispatch batching ×3, all
fusions) is either already on or neutral. Dense compute is concentrated in an
already-fused/batched/q8_0 stage. The only remaining paths to cut the ~12-16 ms
needed for 55 are NOT dispatch-structural:
  (A) **precision** — NAX/int8 for the attn-out-HC and/or MoE compute, applied
      to BOTH no-draft and verifier identically (so draft ≡ no-draft in-build;
      regenerate the reference; gate on 100-sample NLL preservation, not frozen
      bytes). Sidesteps Case-E (which NAX'd only the verifier → divergence).
      Risk: quality drift; memory says dense NAX was −4% at wrong dtype.
  (B) **draft retrain** — higher tau (user=no as of 2026-07-01).
  (C) **relaxed accept contract** — bounded (user=banned).
Absent (A)/(B)/(C), strict byte-exact ceiling is firmly ~44-46 t/s.

---

## 2026-06-30 Priority Reset After Pro Feedback And Fresh n=1000 Sweep

This section supersedes the older attention-first opening below. Keep the
historical measurements for context, but do not use them as the active priority
order.

2026-07-01 continuation check:

- 2026-07-01 late continuation: current acceptance tuning cannot honestly reach
  the >60 t/s target on the standard Pygame prompt. Fresh fixed-cap sweep
  `bench-results/dspark_fastrelaxed_verify_sweep_061252.tsv` measured
  verify=2 `39.41 t/s` / tau `1.90`, verify=3 `46.69 t/s` / tau `2.74`,
  verify=4 `49.16 t/s` / tau `3.49`, and verify=5 `54.39 t/s` / tau `4.58`,
  all simple-canary clean. Keep verify=5 as the best current fast-relaxed cap.
  A clean no-stats recheck
  `bench-results/fastrelaxed_nostats_n1000_072727` measured `54.39 t/s`,
  acceptance `93.6%`, full-accept `87.6%`, first-miss `1.4%`, and canary
  `suspect=0`. This confirms the remaining gap is structural, not logging or
  backend-stat overhead.
- Fresh no-profiler ceiling recheck
  `bench-results/dspark_fast_ceiling_noperf_074738.tsv` confirms the same
  boundary on the Pygame prompt: current `--draft-fast-relaxed` is clean at
  `54.52 t/s`; disabling loop/token guards reaches `57.38 t/s` but repeats
  `self.enemy_direction = 1`; unbounded relaxed accept reaches only `58.99 t/s`
  and collapses into repeated `# Background`; trust-confidence threshold `0.70`
  crosses the numeric target at `72.19 t/s` but corrupts state into `1 = 1`
  garbage. Do not count any acceptance-only or approximate-state shortcut as
  satisfying the >60 goal.
- Fresh ANE/shared-output shortcut check: existing ANE O-proj is explicitly
  disabled for the MXFP4 plane-split Flash sidecar, and the existing ANE
  shared-expert path is a reject. `bench-results/dspark_ane_shared_probe_080227.tsv`
  measured the current fast-relaxed control at `54.37 t/s`, acceptance
  `93.3%`, canary `suspect=0`; enabling
  `DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 DS4_SHARED_EXPERT_ANE_I8I8=1` fell to
  `14.03 t/s`, acceptance `15.7%`, canary `suspect=1`, with repeated `游戏`
  output. This is not the desired independent-engine DSpark draft overlap; it
  is the old synchronous shared-expert path and should stay off for this stack.
- Fresh resident slot-bank probe rejects smaller banks for this workload.
  `bench-results/dspark_slotbank_probe_080449.tsv` measured `--moe-slot-bank
  128` at `17.71 t/s` and `--moe-slot-bank 64` at `18.84 t/s` with the same
  simple canary clean. The reduced resident memory does not help enough to
  offset SSD miss/read overhead, so keep `--resident` with the full 256-slot
  bank for DSpark verifier speed work.
- Fresh route-overlap recheck still supports a new routed-MoE compute-reduction
  kernel. `bench-results/route_overlap_current_080716` was simple-canary clean;
  parsing the full stderr gives 1419 layer-block rows, average reuse `1.69x`,
  median `1.67x`, average unique experts `18.55/30`, and 1052/1419 rows at
  reuse >= `1.5x`. The next MoE attempt should be a new low-register
  row-exact route-dedup kernel for gate/up+SwiGLU, not another threshold tweak
  to the existing grouped-IQ2 wrapper.
- A direct pair2-split attempt on the existing grouped-IQ2 wrapper was tested
  and removed. `bench-results/dspark_grouped_iq2_pair2split_081204.tsv`:
  control `54.19 t/s`, acceptance `93.3%`, tau `4.48`, canary clean; pair2
  split `42.98 t/s`, acceptance `90.9%`, tau `4.32`, canary clean. Splitting
  multiplicity-2 groups out of the current descriptor/wrapper family is still
  slower than default. Do not spend more time on this wrapper; if MoE is next,
  build a different fused route-dedup kernel that avoids the singleton+grouped
  multi-dispatch structure entirely.
- Guarded unbounded acceptance is also rejected. A fresh run
  `bench-results/fastrelaxed_unbounded_guarded_075554` kept the default loop
  guards active and still measured only `41.43 t/s`, acceptance `71.4%`, and
  repeated `player = pygame.image.load('player.png')`. Exact target-state
  update alone does not make force-accept coherent; the output distribution
  still drifts.
- Fresh relaxed-accept ceiling sweep
  `bench-results/dspark_relaxed_ceiling_061634.tsv` shows accept-rate tuning
  alone is spent. Disabling loop/token guards measured `57.44 t/s`, unbounded
  accept measured `58.88 t/s`, and top512/delta20 measured `57.58 t/s`; all
  were canary-suspect. Even the fully unsafe path stays below 60 on this prompt
  while damaging output, so do not pursue wider top-k/logit-delta as the main
  lever.
- Fresh dead-end checks after the current best point:
  `bench-results/dspark_grouped_iq2_pair2_ab_071410.tsv` rejected the
  lower-register grouped-IQ2 pair2 specialization (`43.14 t/s` control,
  `42.48 t/s` pair2, same acceptance), and the pair2 code was removed.
  `bench-results/dspark_defer_heads_fast_ab_072849.tsv` rejected deferred
  attention heads under the fast preset (`54.36 t/s` clean control versus
  `50.77/51.13 t/s`, both canary-suspect). Current batch-canonical also remains
  slower: `bench-results/modeb_nofrontier_timing_072519` forced
  `DS4_DSPARK_BATCH_CANONICAL=1` with frontier disabled and reached only
  `47.11 t/s`, with batch verify still around `77-79 ms/block`.
- Added an opt-in diagnostic hybrid,
  `DS4_DSPARK_TRUST_CONFIDENCE=1`, with
  `DS4_DSPARK_TRUST_CONFIDENCE_THRESHOLD` and
  `DS4_DSPARK_TRUST_CONFIDENCE_MIN_PREFIX`. It requests DSpark confidence,
  commits a high-confidence draft prefix without target verification, and falls
  back to the existing verifier otherwise. This proves the speed ceiling but is
  rejected for quality/state correctness: `bench-results/dspark_trust_conf_sweep_062347.tsv`
  reached `67.61 t/s` at threshold `0.70`, but the output collapsed into
  `# 0`/duplicate-token garbage (`suspect=1`). Thresholds `0.80/0.90/0.95`
  stayed suspect and did not reliably improve speed. Short-prefix variants in
  `bench-results/dspark_trust_prefix_sweep_062908.tsv` were also suspect, and
  `DS4_DSPARK_DRAFT_ONLY_APPROX_SOURCE=output` was slower plus suspect. Keep
  this only as a diagnostic proving that unverified/approximate state is not a
  usable quality path.
- Follow-up trust-confidence loop/token guard wiring was tested after adding the
  same relaxed n-gram/frequency guard to the trusted-prefix path. It prevents
  some immediate spirals but does not produce a usable fast point:
  `bench-results/dspark_trust_guard_0.70_n1000_063818` measured `51.11 t/s`
  and was still suspect; `0.90/0.95` were canary-clean but only
  `47.15/46.00 t/s`. Shorter trusted-prefix sweep
  `bench-results/dspark_trust_guard_t*_p*_n1000_064043` found no good middle:
  `0.95:min3` reached `56.97 t/s` but repeated `player = 0` 185 times, while
  the cleanest `0.90:min4` was only `43.31 t/s` and still visibly malformed.
  Conclusion: confidence-gated approximate target-state commit is rejected as a
  quality/speed path; it is useful only to size the verifier-state contract.
  A later full-resync diagnostic was removed after
  `bench-results/dspark_trust_resync_unguarded_073725.tsv`: resync every 4
  blocks slowed to `38.84 t/s` and still stayed canary-suspect, while less
  frequent/no resync remained suspect.
- Draft-only remains only a sizing bound. `bench-results/dspark_draftonly_n1000_062016`
  measured `413.78 t/s` with verifier cost removed, but canary `suspect=1`.
  Disabling approximate KV refresh stopped after 117 tokens. The DSpark draft
  model is fast enough; the target-state contract is the hard part.
- `DS4_DSPARK_DRAFT_SKIP_BASE_HEAD=1` remains rejected. The recheck
  `bench-results/dspark_skip_head_n1000_063214` improved draft speed to
  `525.6 tok/s`, but acceptance collapsed to `1.7%` and generation fell to
  `20.89 t/s`; the unbounded variant was identical. Do not skip the base head
  for DSpark-5 draft quality.
- Practical conclusion from these measurements: the remaining credible >60 path
  is not another relaxed-accept schedule. It is either true draft overlap on an
  independent engine (the existing draft-prefetch hit rate is good but still on
  the GPU queue), a real batch-canonical committable target-forward path, or a
  verifier compute reduction that keeps target state trustworthy.
- `--draft-fast-relaxed` now enables `DS4_DSPARK_DRAFT_PREFETCH=1` by default.
  Recheck `bench-results/dspark_fastrelaxed_preset_prefetch_builtin_n1000_064652`
  measured `54.18 t/s`, block `82.67 ms`, draft `10.17 ms`, verify `70.33 ms`,
  tau `4.58`, acceptance `93.6%`, and simple canary `suspect=0` without setting
  the prefetch env manually. This is a small queue/readback win, not the true
  overlap needed to cross 60.
- The agent fast preset now matches the CLI fast preset. `ds4-agent
  --draft-fast-relaxed` defaults `DS4_DSPARK_DRAFT_PREFETCH=1`,
  `DS4_DSPARK_RELAXED_TOPK=256`, and `DS4_DSPARK_RELAXED_LOGIT_DELTA=10`
  unless the caller already supplied those env vars. This is not a new speed
  lever, but it removes a stale agent-vs-CLI mismatch from demo benchmarking.

- Fresh draft-overlap upper-bound instrumentation was added to `ds4`,
  `ds4-agent`, and `ds4-server`. It reports the hypothetical block decode rate
  if DSpark draft time were fully hidden behind target verification. Current
  fast preset on the Pygame prompt at n=1000:
  `bench-results/dspark_overlap_ub_fast_n1000_041630` measured real generation
  `51.75 t/s`, block `81.22 ms`, draft `10.75 ms`, verify `68.35 ms`,
  tau `4.32`, acceptance `91.7%`, canary `suspect=0`, and
  `draft-overlap upper-bound: 61.3 tok/s`. This is the first current-tree
  measurement showing a plausible >60 path without changing the verifier
  contract: hide nearly all draft work.
- Recheck `bench-results/dspark_fast_guard_sweep_043848` reproduces the same
  ceiling and rejects simple relaxed-accept tuning as the missing lever. The
  current `--draft-fast-relaxed` preset measured `51.82 t/s`, tau `4.32`,
  acceptance `91.7%`, and `draft-overlap upper-bound: 61.4 tok/s`. Disabling
  relaxed loop/token guards improved the favorable Pygame prompt to
  `54.52 t/s`, tau `4.52`, acceptance `95.4%`, and upper bound `63.2 t/s`, but
  still does not reach 60 without hiding draft time and is not a safe default
  for code/HTML prompts. Raising logit delta to 10 measured only `53.58 t/s`;
  forcing suffix full logits regressed to `47.99 t/s`; cooldown-disable stayed
  near baseline at `52.39 t/s`. Conclusion: guard tuning can provide an
  opt-in speed ceiling on easy prompts, but the real >60 requirement still
  needs draft overlap or a verifier compute reduction.
- Blindly increasing relaxed acceptance is rejected. The unbounded relaxed
  diagnostic `bench-results/dspark_relaxed_unbounded_n1000_041810` measured
  only `40.63 t/s`, tau `4.18`, acceptance `71.4%`, and canary `suspect=1`
  with heavy repeated declarations. Bad non-target accepts poison future draft
  quality and reduce acceptance; do not pursue `FORCE_ACCEPT`/unbounded accept
  as the >60 lever.
- Draft profiling says the hidden work is the full DSpark draft graph, not just
  Markov. `bench-results/dspark_draft_profile_fast_n160_042001` shows warm
  draft blocks around `11.2-11.9 ms`: three DSpark layers at about
  `2.7-2.9 ms` each, base head about `1.4-1.5 ms`, and Markov about
  `1.55-1.7 ms`. Existing ANE helpers cover shared-expert/output-projection
  and prefill paths, not a drop-in DSpark draft runner. A real overlap track
  therefore needs an ANE/independent-engine implementation of the three-layer
  DSpark draft forward, or another way to run that graph concurrently with GPU
  verification.
- Fresh route-overlap sizing keeps routed-MoE on the serious track, but not via
  the existing grouped wrappers. `bench-results/dspark_route_overlap_fast_n160_042232`
  measured current fast n=5 route reuse at about `1.62x` average by the end of
  the run (`1290` route slots/block over 43 layers, roughly `797-946` unique
  expert uses/block late in the sample, many layers still `reuse>=1.50`). This
  is enough reuse to justify a new compact route-dedup kernel.
- Fresh row-routed subprofile
  `bench-results/dspark_row_routed_subprofile_fast_n160_044315` shows where a
  route-dedup kernel must attack. For n=5 verifier calls, aggregate
  row-routed time split was gate/up+SwiGLU `2195.820 ms`, down `549.699 ms`,
  ordered sum `1.002 ms`, total `2746.521 ms` across the profiled calls. Mean
  n=5 call was gate/up `1.547 ms`, down `0.387 ms`, sum `0.001 ms`, total
  `1.936 ms`. Therefore down-only fusion cannot get to 60; the useful target is
  a lower-register exact route-dedup for IQ2 gate/up+SwiGLU, preserving row and
  slot semantics.
- Current grouped-IQ2 route reuse remains rejected. The recheck
  `bench-results/dspark_grouped_iq2_fast_recheck_044354` measured control
  `55.23 t/s`, verify `67.75 ms`, tau `4.48`, acceptance `95.8%`; enabling
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1` and
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_IQ2=1` kept identical acceptance but
  slowed to `45.73 t/s` and verify `84.72 ms`. The existing grouped descriptor
  and IQ2 grouped kernel are not the route-dedup path; do not promote or tune
  them for the fast preset.
- Pair-only grouped-IQ2 was prototyped and rejected. The temporary
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED2_IQ2=1` path used a two-row accumulator
  for exactly duplicated experts and row-exact fallback for all non-pair groups.
  `bench-results/dspark_grouped2_iq2_ab_044902` measured control `55.53 t/s`,
  verify `67.41 ms`, tau `4.48`, acceptance `95.8%`; grouped2 measured only
  `50.71 t/s`, verify `75.17 ms`, with identical tau and acceptance. This
  confirms the current route-grouping overhead is too high even with a smaller
  accumulator footprint; future MoE work should not be another descriptor-based
  grouped wrapper.
- Frontier-vs-standard DSpark scheduling was rechecked as the "tau-6" idea.
  `bench-results/dspark_frontier_vs_nonfrontier_050625` compared the current
  frontier fast preset against the same relaxed/MMA/fast-Q2 flags without
  `DS4_DSPARK_FRONTIER_DRAFT`. Frontier measured `52.40 t/s`, block
  `80.22 ms`, tau `4.32`, acceptance `91.7%`, full-accept `82.1%`, canary
  `suspect=0`. Non-frontier measured only `48.60 t/s`, block `82.15 ms`,
  tau `4.29`, acceptance `85.7%`, full-accept `73.5%`, canary `suspect=0`.
  Conclusion: the normal-first-token path does not buy useful extra tau on the
  current stack; it lowers acceptance and loses. A simple pending "bonus token"
  state would mostly move latency around unless DSpark can draft five *after*
  the pending token, which needs a different two-stage/tree drafter. Keep
  frontier as the fast preset and do not pursue tau-6 plumbing as a standalone
  >60 fix.
- Fresh row-routed-free timing bound:
  `bench-results/dspark_skip_row_routed_ub_n300_042346` set
  `DS4_DSPARK_SKIP_ROW_ROUTED=1` and is intentionally invalid for output. It
  destroyed acceptance (`14.0%`, tau `0.41`) but dropped verifier wall to
  `18.40 ms` from the normal `~68 ms` fast-path range. Therefore routed MoE is
  large enough to be a >60 lever if a correct kernel preserves tau. Do **not**
  confuse this with a usable mode; it is only a sizing bound.
- Fresh current-pass verification of the speed ceiling:
  `bench-results/dspark_relaxed_temp_sweep2_034515` reran the actual
  `--draft-fast-relaxed` CLI preset at n=1000 after an earlier manual-env sweep
  accidentally omitted `DS4_DSPARK_RELAXED_ALLOW_TARGET_TOP_REPEAT=1`. The real
  preset measured `51.45 t/s`, tau `4.32`, acceptance `91.7%`, and canary
  `suspect=0`. Temperature-gated relaxed variants were cleaner but slower:
  `temp_loose` `50.22 t/s` / tau `3.84` / acceptance `88.4%`, and
  `temp_strictish` `48.31 t/s` / tau `3.41` / acceptance `83.5%`. Conclusion:
  temperature/probability gating is useful as a safety filter, not as the >60
  speed lever.
- Fresh slot-bank check rejects the "smaller resident bank fixes decode"
  theory for this DSpark fast path. `bench-results/dspark_slotbank_fast_sweep_034825`
  measured bank256 `51.53 t/s`, tau `4.32`, verify `68.63 ms`; bank128
  collapsed to `18.39 t/s`, verify `164.81 ms`; bank64 collapsed to
  `17.94 t/s`, verify `169.53 ms`. Keep the resident 256-slot bank for DSpark
  verifier work; smaller banks introduce miss/streaming cost that overwhelms any
  memory-pressure benefit.
- Fresh HTML stress rerun
  `bench-results/si_html_010_current_fast_033512` measured current
  fast-relaxed at `37.29 t/s`, tau `3.74`, acceptance `78.8%` on the exact
  4000-token Space Invaders single-file HTML prompt. It diverged from the saved
  strict/no-draft controls (`strict_vs_nodraft_cmp=0`, `fast_vs_strict_cmp=1`)
  and added duplicate JS declarations/selectors plus malformed CSS/JS counts.
  This mode remains non-demo-safe for exacting HTML/code generation.
- Fresh stage-profile sanity
  `bench-results/dspark_fast_stage_profile_035233` is diagnostic only because
  the profiler fences and inflates wall time, but it confirms the current fast
  stack is dominated by target-forward compute, not readback/top-k plumbing:
  attention/output-HC and routed/shared FFN remain the large buckets. No current
  CLI flag or existing grouped-Q2/slot-bank/relaxed-accept setting has credible
  8-10 ms/block savings left.
- Frontier batch-canonical was prototyped behind an opt-in env and then removed
  after A/B. The experiment inserted `metal_graph_verify_suffix_tops` into the
  frontier path before the normal target decode, committed with the batch HC
  helper, and reused compact relaxed top-k suffix checks. It built, but
  `bench-results/dspark_frontier_batch_canonical_ab_035849` measured control
  `54.89 t/s`, tau `4.48`, verify `68.21 ms` versus frontier-batch
  `52.39 t/s`, tau `4.50`, verify `72.61 ms`, both canary clean. Conclusion:
  the existing `verify_suffix_tops` backend is not the missing real Mode-B path;
  do not re-add this shim. A faster Mode B still requires a different unified
  target-forward realization, not simply moving the current batch verifier into
  the frontier branch.
- Fresh `--draft-fast-relaxed` cap check
  `bench-results/dspark_cap3_current_005946` set
  `DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=3` on the current fast preset.
  It measured `51.44 t/s` at n=1000 with the same acceptance shape as the
  current fast base (`91.7%`, position counts `208/204/197/187/173`, full
  accept `82.1%`) and simple canary `suspect=0`. This does **not** improve over
  `bench-results/dspark_fastrelaxed_patched_n1000_003102`; off-argmax capping
  is not the missing >60 lever on the Pygame prompt.
- Fresh verify-cap sweep
  `bench-results/dspark_fast_verify_sweep_040335` reran the current
  `--draft-fast-relaxed` preset at n=500 with fixed verify caps 3/4/5. All
  three outputs were simple-canary clean, but throughput still preferred the
  full DSpark-5 block: verify=3 `47.03 t/s` / tau `2.69`, verify=4
  `50.46 t/s` / tau `3.47`, verify=5 `51.93 t/s` / tau `4.35`. The smaller
  caps reduce verifier wall time but lose more accepted tokens than they save.
  Keep verify=5 as the fast-preset default unless a different verifier backend
  changes this curve.
- The 4000-token HTML Space Invaders stress prompt remains a reject for current
  fast-relaxed. `bench-results/si_html_010_fastrelaxed_003636` measured only
  `37.15 t/s` and ended mid-statement at `let alive =`. The strict comparator
  `bench-results/si_html_010_strict_003940` was also poor (`36.68 t/s`) and
  repeated a sprite row 73 times, so this prompt is a useful stress case but not
  a clean relaxed-vs-strict discriminator. Fresh retest
  `bench-results/si_html_010_fastrelaxed_retest_073924` measured `36.82 t/s`
  and looped on repeated `I'll use a 16x16 array` comments; the no-draft
  baseline `bench-results/si_html_010_nodraft_retest_074406` matched the strict
  failure shape, `32.42 t/s` with repeated `[1,1,...]` sprite rows. Do not
  claim quality from the simple Pygame canary alone.
- Rechecked draft-only artifacts confirm the speed ceiling is unusable:
  confidence thresholds `0.4/0.6/0.8` produced `191/160/136 t/s`, but all
  outputs collapsed immediately into mixed code/math fragments. Keep
  `DS4_DSPARK_DRAFT_ONLY=1` as a diagnostic ceiling only.
- Thresholding the existing grouped-IQ2 routed-MoE consumer by route
  multiplicity also fails. `bench-results/dspark_grouped_iq2_mincount_010345`
  compared the current fast-relaxed baseline (`54.75 t/s`, tau `4.48`,
  acceptance `95.8%`) against grouped-IQ2 min-count `2/3/4`. The grouped rows
  measured only `44.86/46.27/47.89 t/s` with identical acceptance. This rejects
  wrapper-threshold tuning of the current grouped-IQ2 path; any MoE speed win
  needs a different lower-register route-dedup kernel, not another grouped
  descriptor threshold.
- Temporary DSpark draft-layer capping is rejected and was not kept in code.
  `bench-results/dspark_draft_layers_sweep_011747` tested 3/2/1 draft layers
  with the current fast-relaxed stack on the Pygame n=300 canary. The normal
  3-layer row reproduced `54.68 t/s`, draft `11.15 ms`, tau `4.48`,
  acceptance `95.8%`. Two layers reduced draft to `8.62 ms` but collapsed
  acceptance to `27.1%`, tau `0.74`, and `26.04 t/s`. One layer reduced draft
  to `6.13 ms` but collapsed further to `20.5%`, tau `0.59`, and `23.59 t/s`.
  All three DSpark draft layers are quality-critical; draft optimization must
  preserve the full 3-layer graph and Markov/head contract.
- Newest HTML stress rerun:
  `bench-results/si_html_010_fastrelaxed_argmax_012655` used the exact
  4000-token Space Invaders HTML prompt with current fast-relaxed and the
  Markov argmax cleanup. It measured `37.33 t/s`, draft `11.15 ms`, verify
  `80.03 ms`, tau `3.74`, acceptance `78.8%`, and still ended broken at
  `let alive =` with duplicate declarations. Current fast-relaxed is therefore
  not aligned enough for exacting long HTML/code generation.
- Markov argmax and top-k width are closed as cheap levers. Dedicated Markov
  argmax is output-identical to old Markov `topk(1)` but only saves about
  `0.1-0.2 ms` in Markov timing (`54.78 -> 54.62 t/s` in
  `bench-results/dspark_markov_argmax_ab_012922`). A global `topk(1)` argmax
  redirect was removed after it failed to improve strict mode and fast top1 lost
  too much tau. `bench-results/dspark_fast_topk_width_sweep_013708` found
  top4..top512 flat around `54.8-55.0 t/s`; do not spend the next loop on
  top-k-width or top1-argmax tuning.
- Relaxed temperature is also closed as the next speed lever. Ollama's current
  documented speculative knob is a fixed `draft_num_predict` maximum, not a
  published adaptive scheduler. Local temperature-gate matrix
  `bench-results/dspark_temp_gate_matrix_014716` showed the existing
  `DS4_DSPARK_RELAXED_TEMPERATURE` gate only makes the current fast path more
  conservative on the Pygame n=500 canary: base `51.39 t/s`, loose top128/delta6
  `51.76 t/s`, temp `0.8` / min-ratio `0.10` `49.68 t/s`, and temp `1.0` /
  min-ratio `0.05` `48.64 t/s`. Quality canary stayed clean, but tau and
  acceptance dropped. The active >60 work should stay on structural wall-time
  reduction: ANE draft overlap, a different low-register routed-MoE dedup
  kernel, or real Mode B / unified batch-canonical verification.

Latest Pro-agent adjustment: accept the Case E handoff result, but do not let it
recenter the strict verifier plan. The active goal is **>60 generation t/s with
coherent target-faithful output**; byte identity is preferred but not required.
For strict `cmp=0`, the next large levers are MoE route-dedup,
tau/full-accept recovery, and a separate draft-overlap feasibility audit. Case E
top-k sparse is the right next patch only for the long-context
non-byte-identical fast-mode question. A true >60 or ~2x result still requires
Mode B / unified batch-canonical verification, not strict-v1 micro-optimizations.

Active interpretation:

- If the requested product contract is **old-greedy byte identity**, keep
  strict-v1 default and work exact routed-MoE reuse plus tau/full-accept
  recovery. Case E is not the main strict path unless a future selected-top-k
  variant unexpectedly proves `cmp=0`.
- If the requested product contract is **maximum speed with target-faithful but
  not old-byte-identical output**, prioritize real Mode B / unified
  batch-canonical verification. That is the only current path with credible
  >60 headroom.
- In both contracts, rank changes by generation t/s and tau first. Verify-ms
  reductions that lower acceptance are rejects.

2026-06-30 late continuation measurement:

- Current-tree scheduler sweep
  `bench-results/dspark_fast_scheduler_sweep_164454` shows no hidden budget win
  in the existing MMA+fast-Q2 family. Best row was `--draft-verify 5`
  confidence `0.4`: `45.10 t/s`, draft `406.0 tok/s`, verify `72.8`
  proposed tok/s, verify-accepted `59.4 tok/s`, block `79.70 ms`, tau `3.92`,
  acceptance `81.6%`. Verify-4 lowered block wall (`72.13 ms`) but tau fell to
  `3.53`, so generation stayed lower (`44.50 t/s`).
- Current target-gated frontier relaxed sweep
  `bench-results/dspark_frontier_gate_current_164936` also flattens top-k
  tuning. `top8/delta1` and `top128/delta6` both measured about `45 t/s`
  (`45.02` and `44.86`) because the target-margin plus confidence gates dominate
  the looser top-k. Enabling `DS4_DSPARK_RELAXED_GUARD_TARGET_TOP=1` protected
  even target-top repeats but collapsed speed to `30.16 t/s`; keep it as a
  diagnostic, not a default.
- Boundary sweep `bench-results/dspark_frontier_relaxed_boundary_165222` proves
  the older near-50 numbers came from weakening the relaxed contract. With
  `DS4_DSPARK_RELAXED_MARGIN_DISABLE=1`, frontier `top128/delta6` reached
  `49.68 t/s`, tau `4.42`, acceptance `92.9%`. Disabling both margin and
  confidence gates reached `49.89 t/s`, tau `4.35`, acceptance `93.7%`. The
  simple CLI canary did not flag these 500-token outputs, but this is the same
  relaxed family that produced broken `ds4-agent` HTML/game loops, so it remains
  a speed/coherence probe rather than a demo default.
- Numeric stop rule from the current best boundary point: with tau around `4.35`,
  `60 t/s` needs block wall near `72 ms`; current margin-off block wall is
  `86.49 ms`. That means the next real implementation target is roughly
  `14 ms/block` of verifier/draft-overlap savings. More top-k or scheduler
  tuning is not enough; pursue real batch-canonical/unified verification,
  draft/verify overlap, or a new exact route-dedup/shared-weight kernel.
- All-token confidence gating for state-only was tested and removed as a dead
  diagnostic. `bench-results/dspark_stateonly_conf_all_170554` swept thresholds
  `0.4`, `0.65`, `0.8`, and `0.9`. Best generation was only `19.36 t/s`
  (`tau=1.24`, acceptance `46.2%`) and every output was canary-suspect
  (`("("") ## ...`, doubled identifiers, malformed assignments). Do not revive
  `DS4_DSPARK_RELAXED_CONFIDENCE_GATE_ALL`; it reduces tau without restoring
  coherent target state.
- Partial routed-MoE skip by block cadence also fails the speed/coherence
  tradeoff. Fresh sweep `bench-results/dspark_partial_skip_row_routed_171444`
  used the current frontier `top128/delta6`, margin-off, forced-MMA,
  fast-Q2 stack at n=500. Control was clean at `49.70 t/s`, block `88.38 ms`,
  verify `73.98 ms`, tau `4.42`, acceptance `92.9%`. Skipping verifier
  routed-MoE every 4th/3rd/2nd block cut verify to `44.77/45.45/34.47 ms`,
  but tau collapsed to `2.08/1.96/1.32`, generation fell to
  `30.32/28.98/26.42 t/s`, and all three outputs were canary-suspect. This
  rejects approximate routed-MoE omission as a usable >60 path.
- Current-tree sanity after removing the dead all-token confidence gate:
  `bench-results/dspark_current_fast_sanity_173000` reran the same frontier
  `top128/delta6`, margin-off, forced-MMA, fast-Q2 stack at n=500 and measured
  `50.16 t/s`, block `87.58 ms`, draft `11.96 ms`, verify `73.32 ms`, tau
  `4.42`, acceptance `92.9%`, full-accept `83.2%`, with canary `suspect=0`.
  The current fast ceiling is therefore still about `50 t/s`; the >60 gap is
  roughly `14 ms/block` at this tau, or a larger accepted-token contract.
- Fused MoE stop-rule: the useful default is already fused row-exact Q2 routed
  down plus ordered six-slot FP32 sum. The `safe grouped Q2` experiment slowed
  because it added route-group descriptor plumbing while preserving row-exact
  work. The unsafe grouped shared-weight path changes slot-down FP realization.
  Future MoE work must be a new exact shared-weight/dedup kernel, not another
  toggle of the existing grouped wrapper.
- Fast-mode plumbing update: corrected mode sweep
  `bench-results/dspark_mode_compare_fixedenv_002328` shows the current
  `--draft-mode batch|unified` scaffold is not the >60 path even when paired
  with the fast relaxed knobs. At n=300, no-frontier fast measured `49.97 t/s`,
  frontier fast `54.79 t/s`, batch fast `46.03 t/s`, and unified fast
  `46.05 t/s`. A prior lower sweep was invalid because zsh did not split
  grouped env strings, leaving relaxed accept at the fallback `top8/delta1`
  gate. `--draft-fast-relaxed` now defaults `DS4_DSPARK_FRONTIER_DRAFT=1`;
  post-patch smoke `bench-results/dspark_fastrelaxed_frontier_default_002752`
  measured `53.91 t/s`, `tau=4.48`, acceptance `95.8%`, simple canary clean.
  This improves the advertised fast preset, but does not change the main
  optimization priority: true >60 still needs work reduction/overlap or a real
  batch-canonical verifier, not more option plumbing.

2026-06-30 continuation result:

- Fresh current-state speed base:
  `bench-results/dspark_static_budget_233127` rechecked the current loose
  fast stack with frontier relaxed accept, forced-MMA verifier attention,
  fast-Q2 routed down, and full 256-slot resident bank. Static DSpark-5 is
  now the best clean Pygame canary point: budget 3 measured `46.10 t/s`,
  budget 4 `49.36 t/s`, and budget 5 `51.62 t/s`, all simple-canary clean.
  The budget-5 counters were `draft=11.21 ms`, `verify=68.12 ms`,
  `block=81.43 ms`, `tau=4.32`, and acceptance `91.7%`. This supersedes
  older confidence-scheduler numbers as the local fast base. The arithmetic is
  important: hiding the draft wall would put the same point near the >60 target,
  while scheduler tuning alone has no remaining headroom.
- Full-accept "bonus token" was re-audited and rejected as a throughput lever.
  In the frontier verifier, a full accept leaves target logits at the state after
  the last accepted draft row, so the next greedy token is known. However that
  next token's KV/HC state has not been evaluated. Emitting it early would force
  the following frontend cycle to evaluate and suppress the same token before
  any further DSpark block can be valid. With DSpark max block 5, this only
  shifts one token between blocks; it does not reduce verifier or draft work and
  therefore cannot close the >60 gap.
- Slot-bank sweep rejected smaller resident banks as a speed lever.
  `bench-results/dspark_slotbank_fast_232301` with the same fast stack measured
  slot 32 `15.13 t/s`, slot 64 `17.84 t/s`, slot 128 `18.54 t/s`, and slot
  256 `48.28 t/s`. Smaller banks reduce memory pressure but verifier
  expert-miss/I/O cost dominates. Keep full `--resident`/256 slots for DSpark
  speed runs unless a separate SSD-cache experiment proves otherwise.
- Short profiling run `bench-results/dspark_profile_fast_232830` confirms the
  next structural levers. Ignoring first-use warmup, DSpark draft is about
  `11.1 ms/block` (`~9 ms` three-layer graph + `~2.1 ms` Markov/confidence).
  Routed verifier overlap is real but not yet exploited profitably:
  average active-5 route reuse was about `1.6x` (`1290` slots over about
  `806` unique expert uses per block). Existing grouped routed-MoE flags remain
  slower; a useful MoE win must be a new low-overhead exact dedup/shared-weight
  kernel, not the current grouped wrappers.
- Fresh row-routed verifier subprofile after enabling the profiler inside the
  verifier command batch: `bench-results/dspark_row_routed_subprofile_000013_patched`
  emitted 1118 layer rows for the current static-5 fast stack. The fenced
  diagnostic is not a headline speed run, but the split is clear:
  gate/up+SwiGLU mean `1.488 ms/layer`, Q2 down mean `0.382 ms/layer`, ordered
  sum mean `0.001 ms/layer`. The existing grouped-IQ2 path in
  `bench-results/dspark_row_routed_subprofile_grouped_iq2_000105` made the
  dominant stage worse: grouped gate/up mean rose to `1.925 ms/layer` while down
  stayed `0.389 ms/layer`. This rules out the current grouped descriptor/IQ2
  wrapper as the >60 lever. A temporary all-grouped variant that removed the
  singleton kernel also failed (`bench-results/dspark_row_routed_subprofile_grouped_iq2_all_000500`,
  gate/up `1.940 ms/layer`) and was reverted. A specialized multiplicity-2
  grouped IQ2 prototype also failed
  (`bench-results/dspark_row_routed_subprofile_grouped_iq2_pair2_001009`,
  gate/up `2.048 ms/layer`) and was reverted. The next MoE kernel, if pursued,
  must be a different algorithm for sharing/reducing IQ2 gate/up+SwiGLU while
  preserving per-row/slot math; more Q2 down/sum fusion or the current grouped
  descriptor family cannot save enough.
- Removing measurement overhead does not reveal hidden speed. The same static
  DSpark-5 fast stack without `DS4_DSPARK_PERF` or backend stats measured
  `51.50 t/s` in `bench-results/dspark_static5_noperf_233632`, essentially
  identical to the perf-enabled `51.62 t/s` run.
- Wider relaxed gates can improve short-run tau but still do not solve the
  target. `bench-results/dspark_static5_gatewide_233907` measured
  top128/delta6 `51.39 t/s`, top256/delta8 `49.71 t/s`, top512/delta10
  `53.24 t/s`, and unbounded `40.57 t/s` with a canary failure. The best
  short row, top512/delta10, validated on the longer Pygame cap in
  `bench-results/dspark_static5_top512_delta10_n4000_234212`: it stayed
  simple-canary clean but fell to `48.91 t/s` (`2239` tokens, `tau=4.46`,
  acceptance `90.9%`). Treat top512/delta10 as a reduced-quality short-run
  candidate, not a sustained >60 path.
- Added opt-in `DS4_DSPARK_DRAFT_ONLY=1` as a speed ceiling diagnostic. It skips
  the target verifier entirely and can optionally refresh DSpark's main KV from
  draft hidden rows (`DS4_DSPARK_DRAFT_ONLY_APPROX_SOURCE=hc|output`, default
  HC; disable with `DS4_DSPARK_DRAFT_ONLY_APPROX_KV_DISABLE=1`). It clears the
  raw speed target easily (`329-365 t/s` on n=300), but the output becomes
  incoherent almost immediately. Confidence scheduling (`0.4/0.6/0.8`) still
  repeated and produced math/code fragments, so draft-only is not a usable
  product path.
- Added opt-in `DS4_DSPARK_STATE_ONLY_VERIFY=1` /
  `DS4_DSPARK_VERIFY_STATE_ONLY=1`. This runs the target verifier through the
  layers to refresh target KV, HC, and DSpark target-hidden/main-KV state, but
  skips the target LM head/top-k and force-reports the draft suffix as accepted,
  using the DSpark draft logits for the next token. Normal post-token mode cut
  verifier wall to `41.75 ms/block`, but tau fell to `3.48` and generation was
  only `40.15 t/s` with duplicate words. Frontier force-accept mode kept tau at
  `6.00` and reached `53.67 t/s` on n=300 with perf enabled and `52.85 t/s` on
  n=1000 without perf (`bench-results/dspark_goal_continue_115156_frontier_state_only_noperf_n1000`),
  but output still degraded (`pygame.display = pygame.display.set_mode = ...`).
  This is the best current semi-coherent speed diagnostic, not a promotion
  candidate.
- Current conclusion: skipping target logits alone is not enough for >60 in the
  present verifier. The remaining usable path must either make state refresh
  cheaper (true Mode B/unified batch or exact grouped MoE/state kernels), hide
  draft/verification overlap on another engine, or improve the DSpark output
  contract enough that force/frontier modes do not repeat.
- Follow-up routed-MoE skip probes show why the fast state-only family is still
  diagnostic-only. Full routed skip inside state-only frontier mode reached
  `70.57 t/s` (`bench-results/dspark_goal_continue_115602_frontier_state_only_skiprouted_n300`),
  but output was junk. Layer-range/stride skip probes crossed 60 only for
  visibly broken text: `first21` `60.37 t/s`, `last21` `60.80 t/s`
  (`bench-results/dspark_goal_continue_115852_partial_skiprouted_n300`).
  Block-periodic skip also failed the coherence check:
  `block_every2` `60.72 t/s` and `block_every2_offset1` `60.20 t/s`, both
  with immediate repetition / malformed prose, while every-3 variants fell to
  `~58.3-58.4 t/s`
  (`bench-results/dspark_goal_continue_120228_block_skiprouted_n300`).
  Do not promote routed-skip state-only modes; use them only to size the MoE
  cost ceiling.
- Confidence gating does not rescue the verifier-skipping modes. Draft-only
  with confidence thresholds `0.85/0.90/0.95` still generated nonsense
  immediately despite `154-167 t/s`. State-only frontier with thresholds
  `0.65/0.75/0.85` continued to schedule all 5 draft tokens, stayed near
  `53 t/s`, and still emitted degraded code/text
  (`bench-results/dspark_goal_continue_120537_confidence_diagnostic_n300`).
  The confidence head is not currently a quality gate for force/state-only
  acceptance.
- Follow-up `DS4_DSPARK_STATE_ONLY_TARGET_HEAD=1` tested whether the
  state-only failure was merely the DSpark-logit handoff. It was not:
  target-head state-only stayed at `~52.1-52.2 t/s`, was slower than plain
  state-only, and produced the same malformed `pygame.sect` /
  `display.set_mode` text
  (`bench-results/dspark_goal_continue_121216_state_target_head_n300`).
  The failure is forced acceptance of wrong intra-block draft tokens, not just
  the final next-token logits.
- Guarding routed-MoE skip with target top-8 relaxed acceptance also failed.
  The gate rejects many bad blocks, but the verifier signal is already too
  corrupted: full routed skip every 2 blocks dropped to `21.91 t/s` with
  bizarre output, every 3 blocks reached only `26.43 t/s`, first-11-layer skip
  every 2 blocks reached `22.66 t/s`, and last-11-layer skip every 2 blocks
  reached `34.90 t/s`
  (`bench-results/dspark_goal_continue_121857_frontier_skip_gate_n300`).
  Do not combine routed-skip probes with relaxed acceptance as a product path.
- Current coherent knob sweep on this machine:
  `bench-results/dspark_goal_continue_121545_coherent_knob_sweep_n300`.
  Best inspected coherent candidate remains frontier relaxed `top8/delta1`
  with MMA+fast-Q2 at `45.78 t/s`, tau `5.17`, acceptance `89.2%`. Strict b5
  MMA+fast-Q2 was `42.78 t/s`; strict b4 was slower (`41.69 t/s`), normal
  guarded relaxed was `41.59 t/s`, and batch-canonical guarded was
  `40.50 t/s`. This reaffirms that acceptance/scheduler knobs alone cannot
  reach >60; the next useful implementation work must reduce real verifier
  compute (Mode B/unified batch or exact MoE/dense reuse) rather than force or
  skip acceptance.
- Existing grouped/exact routed-MoE flags were re-swept against the current
  coherent frontier top8 baseline in
  `bench-results/dspark_goal_continue_122323_grouped_moe_sweep_n300`.
  None are a hidden win. Reference frontier+MMA+fast-Q2 was `45.72 t/s`.
  Direct ordered-Q2 was `42.85 t/s`, grouped descriptor-only was
  `42.15 t/s`, grouped IQ2 gate/up fell to `36.04 t/s`, grouped Q2 safe was
  `41.52 t/s`, and `DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1` regressed to
  `34.67 t/s` with `~21 ms` commit/overhead. Conclusion: the current grouped
  exact MoE implementation is either overhead-only or slower math; a future MoE
  win needs a new exact route-dedup kernel, not these flags.
- Existing unified/backend flags were also rechecked:
  `bench-results/dspark_goal_continue_122831_unified_backend_sweep_n300`.
  `DS4_DSPARK_UNIFIED_GREEDY=1` with backend `batch` measured `42.35 t/s`,
  `shared_prefix` measured `43.49 t/s`, and `strict_v1` measured `43.23 t/s`.
  The current "shared_prefix" backend is still a strict-v1 wrapper with varmap
  rows5 attention; it is not the missing exact resident shared-prefix N<=5
  kernel and does not beat the frontier fast path.
- Batch-canonical relaxed-gate loosening was not useful:
  `bench-results/dspark_goal_continue_123023_batch_gate_sweep_n300`.
  The best batch-canonical loose gate was only `42.43 t/s`
  (`top256/delta8`). `top16/top32/top128` raised tau/acceptance but also raised
  verify wall to `~80-83 ms`, so generation stayed around `40-42 t/s`.
- Updated current best coherent non-byte fast mode: frontier relaxed with
  forced MMA, fast-Q2, confidence scheduler, and a looser target gate around
  `top96..192 / delta5..6`. The short sweep
  `bench-results/dspark_goal_continue_123349_frontier_gate_sweep_n300` found
  `top128/delta6` at `49.22 t/s`, tau `5.48`, acceptance `95.8%`; the n=1000
  promotion `bench-results/dspark_goal_continue_123705_frontier_top128_n1000`
  held at `48.83 t/s`, tau `5.52`, acceptance `95.6%`, with coherent inspected
  output and no simple repetition markers. Fine-tune
  `bench-results/dspark_goal_continue_123814_frontier_gate_finetune_n300`
  showed the same plateau for `top96/delta5`, `top96/delta6`,
  `top160/delta6`, and `top192/delta6` (`~49.1-49.2 t/s`). This is the best
  current usable fast mode, but still below the >60 objective; reaching >60
  still requires reducing verifier compute rather than more accept-gate tuning.
  A no-perf-counter recheck
  `bench-results/dspark_goal_continue_125904_current_best_noperf_n1000`
  measured the same `48.83 t/s`, so `DS4_DSPARK_PERF=1` accounting is not
  masking a higher user-facing speed.
- Fresh Case E and after-full combinations do not improve that plateau. Case E
  selected-NAX attention under the same frontier `top128/delta6` relaxed gate
  measured only `44.85 t/s` (simple) and `45.57 t/s` (fast-AV) in
  `bench-results/dspark_goal_continue_125014_casee_current_gate_n300`; the
  same run's forced-MMA control measured `48.98 t/s`, tau `5.48`, acceptance
  `95.8%`. `DS4_DSPARK_FRONTIER_AFTER_FULL=1` with the current gate measured
  `47.24 t/s` at budget 5 and `45.02 t/s` at budget 4 in
  `bench-results/dspark_goal_continue_125442_afterfull_top128_n300`. Stop
  spending cycles on Case E / after-full as a >60 path unless a new kernel
  actually cuts verifier compute; the current knobs only reshuffle tau and
  block wall below the existing best.
- Fresh HTML stress test `bench-results/si_html_010_fastrelaxed_003636` shows
  fast relaxed is still quality-fragile on exacting long code prompts: `37.15
  t/s`, tau `3.74`, acceptance `78.8%`, canary suspect, incomplete output. The
  strict comparator `bench-results/si_html_010_strict_003940` was also suspect
  on the same 4000-token cap because it fell into a repeated sprite-row matrix,
  so this prompt is not a clean relaxed-vs-strict discriminator; it is still a
  useful "do not trust speed-only" smoke.
- Fresh current-tree route overlap diagnostic
  `bench-results/dspark_route_overlap_fast_n160_004604` confirms real duplicate
  expert reuse in the verifier: average route reuse `1.63x`, duplicate slots
  `38.1%`, and pair overlap `1.97/6` over 33 profiled blocks. The paired
  row-routed subprofile reports average `gate=0.545 ms`, `down=0.384 ms`,
  `sum=0.001 ms`, `total=0.930 ms` per routed layer call. Therefore an exact
  MoE speed patch must share/reduce IQ2 gate/up+SwiGLU work as well as down;
  down-only ordered-Q2 work cannot recover enough wall time to reach >60.
- Partial state-only target-prefix verification was rechecked in the current
  fast stack (`bench-results/dspark_stateonly_prefix_current_004910`). Prefix
  rows 1, 2, and 3 were canary-suspect and visibly corrupt despite lower block
  walls (`28.69`, `39.31`, and `36.83 t/s`). Prefix 4 was canary-clean at
  `53.94 t/s` on n=300, but the n=1000 promotion
  `bench-results/dspark_stateonly_prefix4_n1000_005140` measured only
  `50.51 t/s`, block `83.16 ms`, tau `4.32`, acceptance `91.6%`. This is not
  better than the current `--draft-fast-relaxed` baseline, so state-only
  prefix gating remains diagnostic-only.
- Added a small relaxed-mode cleanup: when `DS4_DSPARK_RELAXED_LOGIT_DELTA < 0`
  the verifier now passes no top-k-logit-value scratch, so top-k-only relaxed
  acceptance reads token ids only. This removes pointless value readback for
  diagnostics but does **not** produce a usable speed mode. Short sweep
  `bench-results/dspark_goal_continue_130444_topk_only_n300` measured control
  `top128/delta6` at `49.29 t/s`; top-k-only hit `52.08 t/s` for top16,
  `49.92 t/s` for top32, `50.42 t/s` for top128, and `50.08 t/s` for top256.
  The faster cases are visibly broken (`PLAYER_WIDTH` repetition,
  `SCREEN_HEIGHT 600`, `pygame.display = 0`, invalid color tuples). Keep
  top-k-only as a diagnostic for acceptance ceilings, not as a product/demo
  path.
- Draft base-head skip is also rejected. `DS4_DSPARK_DRAFT_SKIP_BASE_HEAD=1`
  was added only as a diagnostic to test whether Markov-only draft logits could
  remove the DSpark base-LM-head cost. The A/B in
  `bench-results/dspark_goal_continue_131924_draft_skip_head_n300` shows the
  draft graph did get cheaper (`12.18 ms` steady median total for control versus
  `10.98 ms` with skip-base-head), but acceptance collapsed to `1.8%`
  (`18/997` draft tokens), tau fell to `1.03`, verifier wall became trivial only
  because almost nothing was accepted, and generation dropped to `18.49 t/s`.
  Keep the base head in the draft path; it is draft-quality-critical, not just
  overhead.
- Fresh current-stack budget check:
  `bench-results/dspark_goal_continue_132210_current_budget_b4b5_n300` measured
  b4 at `47.38 t/s`, tau `4.67`, acceptance `98.0%`, verifier `64.08 ms`,
  while b5 measured `49.16 t/s`, tau `5.48`, acceptance `95.8%`, verifier
  `75.82 ms`. Budget 4 lowers verifier wall but loses more accepted work than it
  saves; keep b5 as the preferred fast-mode benchmark/default unless a new
  kernel changes the block-wall balance.
- Fixed DSpark tau reporting in `ds4`, `ds4-agent`, and `ds4-server`.
  The old status line used `1 + committed_tokens / blocks`, inherited from
  verifier-theory notation, but DSpark's committed counter already includes the
  first emitted token in each block. That is why tau could exceed
  `--draft-verify 4`. New tau is actual emitted/committed tokens per DSpark
  block. Post-fix smoke
  `bench-results/dspark_goal_continue_133108_taufix_fastpath_n300` measured the
  same current fast-path speed (`49.25 t/s`) but reports physically meaningful
  tau `4.48` instead of `5.48` (`draft=12.42 ms`, `verify=75.63 ms`,
  `block=90.35 ms`, acceptance `95.8%`). With tau `4.48`, reaching `60 t/s`
  requires block wall around `74.7 ms`, or a combined acceptance/tau increase
  plus block-wall reduction; a pure metrics/tau artifact cannot claim the goal.
- Added a relaxed-accept loop guard after a `ds4-agent` Space Invaders HTML run
  with the loose frontier `top128/delta6` fast mode started repeating
  `let invader = ...`. The guard is per-token/per-block, not a permanent
  disable: strict target-argmax tokens are still accepted, but relaxed
  non-argmax tokens are rejected if they would extend a recent repeated token
  n-gram or overuse one token in a short recent window. Default knobs are
  `DS4_DSPARK_RELAXED_LOOP_NGRAM=4`,
  `DS4_DSPARK_RELAXED_LOOP_WINDOW=256`,
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_WINDOW=192`, and
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=24`; disable only for speed ceilings with
  `DS4_DSPARK_RELAXED_LOOP_GUARD_DISABLE=1` or
  `DS4_DSPARK_RELAXED_TOKEN_GUARD_DISABLE=1`. Evidence:
  `bench-results/dspark_goal_continue_135310_loop_guard_default_html_n600`
  measured `44.17 t/s`, tau `4.12`, acceptance `88.7%`, and logged 10 guard
  rejections with no simple `# Background`/`PLAYER_WIDTH`/`let invader = []`
  collapse markers. The stricter `ngram=3` canary
  `bench-results/dspark_goal_continue_134832_loop_guard_html_n600` measured
  `42.22 t/s`, tau `4.00`, acceptance `83.3%`. A partial guard-off canary
  reached `49.00 t/s` on the same prompt but should be treated as unsafe for
  long agent runs because the real `ds4-agent` workload already exposed a loop.
  The first token-frequency canary
  `bench-results/dspark_goal_continue_140144_loop_token_guard_html_n400`
  used `DS4_DSPARK_RELAXED_LOOP_NGRAM=3` and
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=12`; it measured `45.37 t/s`, tau `4.26`,
  acceptance `89.7%`, logged both n-gram and token-count guard rejections, and
  showed no simple collapse markers in a 400-token CLI smoke. This is not yet a
  real long `ds4-agent` quality pass.
- Follow-up guarded HTML canaries on the current tree show that tightening the
  target gate too much throws away tau, while removing the logit-delta but
  keeping a small top-k is the best current fast-demo shape. On the same
  500-token Space Invaders HTML prompt:
  `bench-results/dspark_goal_continue_140636_relaxed_guard_grid_n500/top128_delta6_tok12_ng3`
  measured `41.26 t/s`, tau `3.81`, acceptance `82.6%`;
  `bench-results/dspark_goal_continue_141959_relaxed_conf_gate_n500` measured
  `42.31 t/s`, tau `3.90`, acceptance `86.0%`;
  `bench-results/dspark_goal_continue_142130_unbounded_conf_loop_n500` regressed
  to `33.50 t/s`; and
  `bench-results/dspark_goal_continue_142250_top16_nodelta_conf_loop_n500`
  was the best of this mini-sweep at `45.14 t/s`, tau `4.40`, acceptance
  `88.8%`, with no simple loop markers. A real `ds4-agent` HTML/game retest
  later produced duplicate declarations and repeated constants, so this is not a
  recommended fast-agent demo gate. Treat it as a failed diagnostic.
- Recoverable cooldown is now attached to relaxed loop-guard hits:
  `DS4_DSPARK_RELAXED_COOLDOWN_BLOCKS=N` defaults to `4` and
  `DS4_DSPARK_RELAXED_COOLDOWN_DISABLE=1` restores the older behavior. During
  cooldown, target-argmax accepts still pass, but off-argmax relaxed accepts are
  rejected. Current HTML canary
  `bench-results/dspark_relaxed_cooldown_html_174213` on loose
  `top128/delta6` + forced-MMA + fast-Q2 measured `46.20 t/s`, tau `3.87`,
  acceptance `86.6%`, and logged loop-guard rejections. It avoided the visible
  duplicate-declaration loop but still had `suspect=1`, so this is a recoverable
  safety brake, not a path to the >60 objective.
- Existing non-frontier 1+5 mode is not enough. Run
  `bench-results/dspark_nonfrontier_1plus5_fast_174742` used the target first
  token plus five DSpark suffix drafts with the same loose relaxed/MMA/fast-Q2
  stack and measured `44.74 t/s`, tau `3.91`, acceptance `83.1%`, canary
  `suspect=0`. The nominal sixth emitted token is paid for by the normal
  first-token decode, so the real path remains a committable N=6/batch-canonical
  verifier or a block-wall reduction.
- No-perf check does not rescue the current fast stack:
  `bench-results/dspark_current_fast_noperf_175020` removed `DS4_DSPARK_PERF=1`
  and still measured only `50.20 t/s` at n=1000, with acceptance `94.7%`,
  full-accept `85.6%`, first-miss `0.0%`, and canary `suspect=0`. The gap to
  >60 is not caused by performance-counter instrumentation.
- State-only target-prefix is also not the missing middle ground. Sweep
  `bench-results/dspark_stateonly_prefix_sweep_175409` with prefix rows
  `{1,2,3,4}` produced: prefix1 `52.38 t/s` suspect, prefix2 `49.34 t/s`
  suspect, prefix3 `44.30 t/s` clean, prefix4 `49.23 t/s` clean. Adding
  periodic routed-MoE skip on top in
  `bench-results/dspark_stateonly_prefix4_skiprouted_175658` was worse:
  every-2 `18.02 t/s`, every-3 `24.61 t/s`, both canary-suspect. Do not spend
  more loop time on state-only/routed-skip combinations unless the draft quality
  contract changes.
- Frontier relaxed mode now treats `--draft-scheduler confidence` as a
  non-argmax quality gate rather than a raw block-length scheduler. Low-confidence
  target-argmax accepts are still allowed, while low-confidence non-argmax
  accepts are rejected unless
  `DS4_DSPARK_RELAXED_CONFIDENCE_GATE_DISABLE=1` is set. A raw scheduler test
  (`bench-results/dspark_goal_continue_141525_frontier_conf_sched_guard_n500`)
  confirmed why this matters: shortening whole blocks dropped tau to `2.97` and
  speed to `37.16 t/s`.

Current relaxed-accept status:

- `DS4_DSPARK_RELAXED_ACCEPT=1` has been prototyped as a deliberately
  non-target-greedy diagnostic. It still runs the target verifier over the
  emitted draft sequence and commits target state for those emitted draft
  tokens, but it can accept high-confidence non-argmax target tokens. With
  `DS4_DSPARK_FRONTIER_DRAFT=1`, it can also override the current target token.
- Relaxed accept is now target-gated by default: `DS4_DSPARK_RELAXED_ACCEPT=1`
  requires the draft token to pass the target top-k/logit gate and, for
  non-argmax accepts, the target-margin gate:
  `top1-top2 <= DS4_DSPARK_RELAXED_TARGET_MARGIN` and
  `top1-draft <= DS4_DSPARK_RELAXED_DRAFT_MARGIN` (defaults `0.35` / `0.35`).
  Tune with `DS4_DSPARK_RELAXED_TOPK=N`,
  `DS4_DSPARK_RELAXED_LOGIT_DELTA=F`,
  `DS4_DSPARK_RELAXED_TARGET_MARGIN=F`, and
  `DS4_DSPARK_RELAXED_DRAFT_MARGIN=F`. The old top-k/delta-only path is
  explicit-only: `DS4_DSPARK_RELAXED_MARGIN_DISABLE=1`. The fully ungated
  speed-ceiling behavior is explicit-only: `DS4_DSPARK_FORCE_ACCEPT=1` or
  `DS4_DSPARK_RELAXED_UNBOUNDED=1`.
- Fresh margin-gate evidence says this is a safety/coherence guard, not the
  >60 path. `bench-results/dspark_margin_gate_html_n500_144215` with
  `top16`, no logit-delta, confidence `0.4`, and margins `0.35/0.35` measured
  `34.36 t/s`, tau `2.52`, acceptance `68.4%`. Sweep
  `bench-results/dspark_margin_gate_sweep_144358` measured:
  `0.75/0.75` `33.82 t/s`, tau `2.36`;
  `1.25/1.25` `38.35 t/s`, tau `3.24`;
  `2.00/2.00` `41.86 t/s`, tau `4.04`;
  `4.00/4.00` `42.61 t/s`, tau `4.16`.
  The n=400 CLI canaries did not show the simple duplicate-declaration markers,
  but they are still far below the speed target. Do not spend more main-track
  time on top-k/margin tuning without a stronger quality harness.
- Rechecking state-only target-head after the top-k-logit readback fix did not
  change the conclusion: `bench-results/dspark_state_only_target_head_margin_n400_144704`
  with margins `2.0/2.0` measured `41.82 t/s`, tau `4.04`, acceptance `82.6%`,
  essentially the same as the normal margin run.
- Current Mode-B remeasure does not support more flag-toggling:
  `bench-results/dspark_mode_compare_current_145238`, Pygame Space Invaders,
  n=400, measured strict default `37.57 t/s`, forced-MMA `40.35 t/s`, and
  `DS4_DSPARK_VERIFY_CANONICAL=batch` `37.79 t/s`. The existing batch-canonical
  flag is not the reference-style single-forward win; it needs committed-state
  redesign before it can be a >60 path.
- Exact correction-token scheduling was tried and rejected:
  `bench-results/dspark_correction_ab_150049`. Evaluating the target correction
  immediately on draft miss preserved output (`cmp=0`) but slowed generation
  from `37.51 t/s` to `35.92 t/s` because commit/overhead rose to
  `14.27 ms/block`. The local diagnostic patch was removed.
- Medusa-style typical/probability gating was prototyped and rejected locally.
  The temporary opt-in gate was then removed so it does not become another stale
  diagnostic path. With the current default margin gate, HTML canary
  `bench-results/dspark_typical_accept_152401` was identical with and without
  the typical gate: `37.38 t/s` vs `37.34 t/s`, tau `2.78`, acceptance `74.6%`.
  With the margin gate disabled, `bench-results/dspark_typical_margin_off_152601`
  showed the typical gate was too conservative: plain margin-off measured
  `41.97 t/s`, tau `3.79`, acceptance `84.7%`, while typical margin-off fell to
  `36.82 t/s`, tau `3.01`, acceptance `73.1%`. The short CLI tails had no simple
  duplicate markers, but neither path approaches >60; do not spend more
  main-track time on probability-threshold tuning unless a tree/multi-candidate
  acceptor is being built.
- Current-tree grouped exact Q2 recheck also rejects the existing grouped
  descriptor wrapper as the main MoE speed path. In
  `bench-results/dspark_grouped_exact_q2_current_153050`, strict b5/n220 was
  `38.34 t/s`, `verify=84.63 ms`, tau `4.00`, acceptance `80.0%`; enabling
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN=1` stayed byte-clean
  (`cmp=0`) but slowed to `36.96 t/s`, `verify=87.38 ms`. The route grouping
  plumbing is safe, but the current safe wrapper still runs row-exact work after
  grouping. Any useful MoE dedup must be a new shared-weight, dot-order-preserving
  kernel, not this wrapper.
- Current-tree recheck of the existing pair-count-2 shared-weight Q2 kernel also
  stays rejected: `bench-results/dspark_grouped_pair2_current_153259` measured
  strict `38.39 t/s` versus grouped pair2 `33.19 t/s`, and output diverged
  (`cmp=1`, first diff swapped `RED`/`GREEN` constant order). Do not use
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_UNSAFE=1` as a speed or
  correctness candidate.
- Treat relaxed-mode "acceptance 100%" as a contract artifact, not a draft
  quality metric. It means "accepted because forced", not "target agreed".
- Numeric ceiling so far: normal relaxed b5 reached `52.70 t/s` at n=1000
  (`bench-results/over50_relaxed_b5_n1000_061732`). Frontier relaxed with forced
  MMA reached `57.80 t/s` at n=1000
  (`bench-results/over60_frontier_relaxed_noperf_062523/mma.err`) and `58.68
  t/s` at n=300 (`bench-results/over60_frontier_override_probe_062310`). A
  layer-42 shortcut reached `59.53 t/s` at n=300.
- The generated output from the old ungated relaxed/frontier and layer-limit tests is
  visibly degenerate (`# Background` loops, repeated `PLAYER_WIDTH`, repeated
  coordinate fragments). Therefore this is **not** a valid >60 solution and
  should not be promoted. Keep the code only as an opt-in speed-ceiling
  diagnostic until a coherence gate passes on n=1000/4000.
- Follow-up `DS4_DSPARK_RELAXED_TOPK` gates confirm the diagnosis. Top-k gating
  prevents the obvious force-accept loops, but it gives back the fake speed:
  `bench-results/over60_relaxed_topk_sweep_064849` peaked at `46.28 t/s` for
  top-128, while unconstrained force-accept in the same sweep reached `55.67
  t/s` and looped on `# Background`. Budget/confidence variants in
  `bench-results/over60_relaxed_topk_ops_065433` peaked at `46.59 t/s`. Keep
  `DS4_DSPARK_RELAXED_TOPK` as a diagnostic/coherence guard only; it is not a
  >60 path.
- Current guarded-relaxed retest after adding the default target top-k plus
  logit-delta gate: the immediate repetition bug is fixed in short probes, but
  throughput barely moves. `bench-results/relaxed_gate_n1000_072929` measured
  control `36.98 t/s`, guarded `top32/delta2` `37.78 t/s`, and guarded
  `top128/delta4` `37.42 t/s`. The quick 300-token grid
  (`bench-results/relaxed_gate_sweep_072520`) found no obvious `# Background`
  collapse for normal guarded relaxed; frontier guarded was slower. Looser
  normal gates in `bench-results/relaxed_loose_gate_n300_073240` did not help:
  `top256/delta6` matched the tight gate, while `top512/delta8` and
  `top1024/delta10` slowed down. Conclusion: target-gated relaxed accept is a
  useful coherence diagnostic and may be worth the reduced quality test, but it
  is not the main speed lever.
- If the next product target is "fast but not byte-identical", prefer real
  Mode B / unified batch-canonical or tree scheduling over relaxed force-accept.
  Relaxed force-accept is useful only to size the upper bound when argmax
  rejection is removed.
- 2026-06-30 continuation: confidence-softmax cost tuning and pure trust-draft
  are not the route to >60. `bench-results/dspark_softmax_cost_smoke_205845`
  showed aggressive cost settings (`MIN=4`, `FIXED_COST=20`,
  `TOKEN_COST=0.05`, `MASS_WEIGHT=1.0`) simply match static-5 at roughly
  `43.7-43.8 t/s`. Trust-draft/draft-only hit `109-197 t/s` in
  `bench-results/dspark_trust_draft_smoke_210027` and
  `bench-results/dspark_trust_draft_threshold_smoke_210138`, but collapsed into
  incoherent math/Unicode text even at confidence threshold `0.995`; reject it
  unless a real target-state resync mechanism is added.
- Added relaxed off-argmax caps as quality plumbing, not as the final speed
  lever: `DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=N` and
  `DS4_DSPARK_RELAXED_OFFARGMAX_COOLDOWN_BLOCKS=N`. The best current smoke,
  `bench-results/dspark_relaxed_offargmax_cap_n1000_210641`, used loose
  frontier `top128/delta6`, forced-MMA, fast-Q2, and `MAX_OFFARGMAX=1`; it was
  canary-clean at `48.25 t/s`, `tau=3.54`, acceptance `85.9%`. Keep this as a
  safer reduced-quality diagnostic while pursuing real Mode B or verifier-cost
  reduction for >60.
- The best reduced-quality CLI candidate after loosening the margin gate is now
  `DS4_DSPARK_RELAXED_MARGIN_DISABLE=1` plus
  `DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=2` on the frontier/MMA/fast-Q2
  stack. `bench-results/dspark_relaxed_cap_marginoff_grid_211122` measured
  `52.64-52.81 t/s` at n=600 with simple canaries clean; no-stats n=1000
  `bench-results/dspark_relaxed_cap_no_stats_212046` measured `53.06 t/s`,
  acceptance `94.0%`, but duplicate declarations still appeared in the canary
  table. This is a useful speed candidate for CLI smoke, not a robust agent/demo
  mode and not the >60 solution.
- Negative checks to avoid repeating: unbounded relaxed with the off-argmax cap
  was worse and visibly lower quality (`bench-results/dspark_relaxed_cap_unbounded_211336`,
  `47.72 t/s`); layer-limit with the cap destroyed acceptance/output
  (`bench-results/dspark_relaxed_cap_layerlimit_211438`, `22.90-25.53 t/s`);
  lowering verifier indexer top-k to `1024/512/256` did not affect this
  short-context speed (`bench-results/dspark_relaxed_cap_index_topk_211635`);
  `--draft-verify 4` lost to verify-5 and was canary-suspect
  (`bench-results/dspark_relaxed_cap_v4_no_stats_212151`, `50.78 t/s`).
- Current draft profile says hiding draft is still the cleanest >60 bridge:
  `bench-results/dspark_draft_profile_current_211857` shows warm draft cost
  around `11.1 ms/block` (`~9 ms` graph, `~2.1 ms` Markov). With the current
  best fast stack, a high-acceptance n=220 prompt reached `55.01 t/s` at
  `tau=4.62`, block `83.27 ms`. Hiding the draft block under verifier work would
  push this class over 60 without further relaxed-accept risk.
- Batch-canonical relaxed retest after wiring relaxed commit into the batch
  path did not rescue Mode B. With fixed env in
  `bench-results/modeb_relaxed_probe_fixedenv_074328`, fast control was
  `42.93 t/s`, batch relaxed `top32/delta2` was `39.73 t/s`, and batch
  force-accept was only `43.41 t/s` while repeating `display.set_mode`. The
  current batch verifier is cost-bound, not merely acceptance-bound. Follow-up
  smoke `bench-results/batch_busy_smoke_075026` fixed batch/unified GPU-busy
  accounting and reports `70.56 ms/block` active of `77.65 ms` verify, so the
  batch path is also GPU-compute-bound rather than a hidden CPU/dispatch idle
  problem.
- Frontier fixed-env retest (`bench-results/frontier_force_fixedenv_074541`)
  also rejects the current frontier route as the >60 path: exact frontier
  `42.76 t/s`, guarded `top32/delta2` `45.53 t/s`, and force frontier
  `43.93 t/s`. The guarded version avoids the immediate collapse but does not
  recover the historical fake speed.
- New frontier collapse-canary sweep after the target top-k plus
  logit-delta gate (`bench-results/frontier_relaxed_collapse_smoke_075847`)
  reproduces the important decision boundary. The old failure shape was
  `DS4_DSPARK_FRONTIER_DRAFT=1` plus relaxed accept; ungated/default-loose
  variants can produce `# Background`/`display.set_mode` loops. With the current
  gated implementation, `top8/delta1`, `top16/delta1`, and `top32/delta2` all
  survived the n=160 collapse canary at about `45 t/s`.
- Promotion run `bench-results/frontier_relaxed_promote_n300_080222` kept
  those same frontier relaxed gates coherent at n=300. `top8/delta1` measured
  `45.52 t/s`, `tau=5.04`, `87.3%` acceptance, no collapse markers, and
  max repeated segment count 1. `top16/delta1` and `top32/delta2` were nearly
  identical. `top4/delta0.5` is safer but slower.
- Longer check `bench-results/frontier_relaxed_top8_n1000_080452` measured
  frontier relaxed `top8/delta1` at `43.00 t/s`, `tau=4.79`, `85.2%`
  acceptance, verify `72.25 ms/block`, with coherent inspected output. Earlier
  `return pygame` marker counts in this run are normal `return pygame.Rect(...)`
  methods, not repetition collapse. `bench-results/frontier_relaxed_top4_n1000_080558`
  measured `top4/delta0.5` at `41.61 t/s`. Conclusion: gated frontier relaxed
  is now a useful coherence canary and a possible reduced-quality-test target.
  The default relaxed gate has been tightened to this surviving `top8/delta1`
  setting, but it is not a >60 path and should not displace Mode B / MoE-dedup
  work. Post-tighten smoke
  `bench-results/frontier_relaxed_default_after_tighten_081040` confirms the
  default now logs `target top-8 + logit-delta<=1.00`, reaches `45.00 t/s` at
  n=160, and has zero collapse markers.
- Layer-42 shortcut retest (`bench-results/layer42_guard_probe_074733`) is a
  hard reject: guarded layer-42 was `24.01 t/s` and visibly degenerate
  (`import sys pygame...` loops); force layer-42 was `25.53 t/s`. Do not revive
  layer-limit verification for the fast non-byte path.
- 2026-06-30 compact suffix top-k/logit-gate implementation:
  `row_topk` now has optional compact `row_topk_logits`, gathered by
  `ds4_gpu_indexer_topk_logits_tensor`, so guarded relaxed suffix acceptance can
  enforce both top-k membership and logit-delta without reading full vocab rows.
  This fixes the previous "compact top-k ignores delta" plumbing gap, but it
  does **not** change the speed conclusion. Fresh artifacts under
  `bench-results/dspark_fast_guarded_compact`:
  - normal guarded relaxed n=300: `41.29 t/s`, tau `4.67`, acceptance `77.0%`,
    verify `67.02 ms/block`, no quick repetition marker.
  - normal guarded relaxed n=160 sweep: `top8/delta1` `42.47 t/s`,
    `top16/delta2` `40.01`, `top32/delta3` `40.22`, `top64/delta5` `40.42`.
  - frontier guarded n=1000 `top8/delta1`: `43.51 t/s`, tau `4.70`,
    acceptance `85.2%`, verify `69.37 ms/block`, coherent inspected output.
  - batch-canonical guarded n=300: `42.33 t/s`, tau `4.26`, acceptance `83.5%`,
    verify `61.46 ms/block`.
  - grouped exact routed-MoE opt-in n=160 did not help: control `40.59 t/s`,
    grouped exact `40.03 t/s`.
  Conclusion: compact logits make the relaxed guard honest, but acceptance/gate
  tuning is no longer the route to >60. The current blocker is verifier compute
  wall time, not suffix gate readback or relaxed top-k policy.
- Current-build regression note: rechecking the older "best stack" command
  (forced MMA + confidence scheduler + fast-Q2, no relaxed/frontier) in
  `bench-results/dspark_fast_guarded_compact/current_best_recheck` measured
  `42.50 t/s`, verify `71.25 ms/block`, acceptance `83.0%`. The older artifact
  `bench-results/over50_fast_conf_fastq2_n1000_054201` remains `48.00 t/s`,
  verify `63.56 ms/block`, same acceptance. Compact suffix logits are inactive
  in this no-relaxed recheck, so treat the slowdown as current-tree/run-state
  drift until isolated; do not spend more time on relaxed gates to explain it.

Decision queue from the latest Pro feedback:

0. **Measure tau shape on every serious run.** The runtime now prints enough
   data; do not accept a benchmark without pos-wise acceptance, conditional
   acceptance, full-accept rate, tau, draft ms, verify ms, and generation t/s.
   `gen_tps = tau / block_wall`, so a tau move from `4.2 -> 4.8` is worth about
   the same as a substantial verifier kernel patch. Rank by generation t/s and
   tau, not by verify ms alone.
1. **Strict `cmp=0`: MoE route-dedup is the next kernel project.** Keep
   row-exact router logits/top-k/weights, preserve per-row/per-slot down outputs
   or a proven exact equivalent, and keep ordered FP32 slot0..slot5 accumulation.
   Do not revive rejected grouped shared-weight Q2, unordered `sum6`, batched
   router, or batched shared-expert shortcuts. Gate with slot-down max-delta `0`,
   MoE-boundary max-delta `0`, then `cmp=0` at n=160/1000/4000.
2. **Case E: only top-k sparse long-context work remains interesting.** Current
   Case E attends dense over compressed keys while strict uses selected top-k.
   The next selected-key stream must be validated at `-c 4096`, `16384`,
   `65536`, and `100096`, budgets 4 and 5, past the sparse-indexer crossover.
   Keep it opt-in unless it proves `cmp=0`; its purpose is long-context fast
   mode, not replacing strict-v1.
3. **ANE draft overlap is a separate feasibility audit.** Do not count the full
   draft block as hidden until the dependency is proven. Size expected gain as
   `hidden_draft_ms = correct_path_reuse_rate * draft_ms - wasted_work_cost`.
   Measure full-accept rate, optimistic full-accept reuse rate, ANE draft
   latency, sync/transfer overhead, and wasted draft work on partial rejects.
4. **Mode B / unified batch-canonical is the true >60 track.** It changes the
   contract: one target batched forward, argmax accept, commit batched state,
   and compare against the unified no-draft baseline instead of old row-decode
   `cmp=0`. Current `--draft-mode batch|unified` is diagnostic and slower; do
   not report it as implemented Mode B.
5. **Stop rules:** no more attention fusion or dense/NAX work unless a
   non-fenced run shows a real generation-tps win. Case E fast-AV already showed
   lower verify ms can lose when acceptance drops.

Fresh sidecar/resident sweep on branch `dspark-attn`, prompt
`Make a game of Space Invader in Pygame`, `-n 1000`, `-c 4096`:

| mode | gen t/s | vs no-draft | verify ms/block | tau | full accept | notes |
|---|---:|---:|---:|---:|---:|---|
| no draft | 35.95 | 1.00x | n/a | n/a | n/a | paired local baseline |
| strict b4 | 39.64 | 1.10x | 64.71 | 4.22 | 70.9% | cmp-clean path |
| strict b5 | 39.85 | 1.11x | 77.09 | 4.85 | 62.1% | best strict in this sweep |
| Case E NAX b4 | 42.12 | 1.17x | 61.16 | 4.33 | 74.9% | opt-in, non-byte-identical |
| forced MMA b5 | 44.58 | 1.24x | 62.86 | 4.78 | 62.7% | best existing fast mode, non-byte-identical; rechecked after stats fix |
| forced MMA b5 + draft grouped-strided FP8 | 46.79 | 1.30x | 62.43 | 4.78 | 62.7% | output-identical to old draft path |
| forced MMA b5 + confidence fast-Markov | 47.49 | 1.32x | 61.77 | 4.84 | 69.9% | default confidence path now preserves fast Markov |
| forced MMA b5 + confidence fast-Markov + fast-Q2 | 48.00 | 1.34x | 63.56 | 4.97 | 71.6% | current best observed; non-byte-identical fast stack |
| batch-canonical b5 | 36.65 | 1.02x | 83.31 | 4.69 | 54.0% | current implementation is not the fast Mode B path |
| unified-canonical b5 | 36.76 | 1.02x | 83.26 | 4.69 | 54.0% | same as batch in this sweep |

New metrics added to `ds4`, `ds4-server`, and `ds4-agent`: conditional
per-position acceptance and full-accept / first-miss block rates. Example strict
b5 result:

- unconditional: `1=181/206 2=175/206 3=161/206 4=149/206 5=128/206`
- conditional: `1=87.9% 2|1=96.7% 3|1-2=92.0% 4|1-3=92.5% 5|1-4=85.9%`
- full-accept: `62.1%`, first-miss: `12.1%`

Fresh forced-MMA b5 post-stats check (`bench-results/over50_post_stats_005658`):

- generation `44.58 t/s`, `1.24x` vs paired no-draft `35.95 t/s`
- draft `15.76 ms`, verify `62.86 ms`, overhead `1.83 ms`, commit `1.41 ms`
- tau `4.78`, acceptance `75.6% (790/1045)`
- conditional acceptance:
  `1=86.1% 2|1=96.1% 3|1-2=93.1% 4|1-3=90.1% 5|1-4=90.3%`

Draft `attn_output_a` grouped-strided rows5 improvement:

- Implemented a grouped-strided FP8 rows5 matvec for the DSpark draft
  `attn_output_a` group projection. It replaces the old token-by-group loop with
  one rows5 dispatch across all output groups while preserving the same
  per-token FP8 reduction order. Disable with
  `DS4_DSPARK_DRAFT_ATTN_OUT_A_STRIDED_DISABLE=1`.
- Intermediate per-group strided A/B
  (`bench-results/over50_draft_strided_n1000_021947`, b5/n1000,
  forced-MMA verifier): output matched the old draft path (`cmp=0`), draft
  improved from `15.62` to `12.71 ms/block`, generation moved from `44.74` to
  `45.90 t/s`, tau stayed `4.78`, and acceptance stayed `75.6%`.
- Final grouped-strided A/B
  (`bench-results/over50_draft_grouped_strided_n1000_022916`, b5/n1000):
  output matched the old draft path (`cmp=0`), draft improved from `15.68` to
  `11.17 ms/block`, generation moved from `44.73` to `46.79 t/s`, verify stayed
  essentially unchanged (`62.59` vs `62.43 ms`), tau stayed `4.78`, and
  acceptance stayed `75.6%`.
- Verify-budget sweep with the grouped-strided draft path
  (`bench-results/over50_grouped_strided_budget_sweep_023120`, n=1000):
  b3 `42.45 t/s` (tau `3.41`), b4 `45.38 t/s` (tau `4.26`), b5 `46.31 t/s`
  (tau `4.78`). Budget 5 remains the best operating point.
- Remaining gap: this is a real stackable +4.6% headline win versus the paired
  old path and about +40% draft-speed win (`319 -> 448 draft tok/s`), but it
  still leaves the best measured run below 50 t/s. Further draft-only work has
  less obvious single-stage leverage; the post-patch blocking profile
  (`bench-results/over50_draft_strided_block_profile_022202`) showed
  `attn_out` reduced to about `1.6 ms/layer` after the per-group version, and
  the final grouped version reduces the steady-state draft block to about
  `11.1 ms`.

Post-feedback probes already run on this branch:

- Explicit batch verifier gate fix:
  `DS4_DSPARK_BATCH_VERIFY=1` was previously swallowed by the strict-v1 branch
  unless batch-canonical was also enabled. The gate now skips strict-v1 whenever
  explicit batch verification is requested, so the batch path is measurable.
  Fresh n=300 A/B (`bench-results/over50_batch_approx_fixed_041115`) shows the
  actual path is not a speed candidate yet: plain batch verify measured
  `24.03 t/s` with `108.34 ms/block` overhead/commit, and approximate batch
  modes measured only `37.22-37.92 t/s` with about `23 ms/block`
  overhead/commit. Keep this as a diagnostic wiring fix; do not promote batch
  verify as fast mode until commit-state cost is redesigned.
- Batch approximate prefix-commit repair:
  explicit `DS4_DSPARK_BATCH_VERIFY=1 DS4_DSPARK_BATCH_APPROX_STATE=1` now
  captures verifier prefix states by default and commits accepted partial
  prefixes instead of replaying them exactly. Disable with
  `DS4_DSPARK_BATCH_APPROX_PREFIX_COMMIT_DISABLE=1` to restore the old replay
  diagnostic. This fixes the measured commit/replay pathology but is still not
  the >50 path: `bench-results/over50_batch_prefix_042805` improved n=300 batch
  approximate from `37.90 t/s`, block `100.99 ms`, overhead `22.78 ms`, to
  `45.71 t/s`, block `82.39 ms`, overhead `1.95 ms`; normal forced-MMA was
  still faster at `46.62 t/s`. n=1000 confirmation
  (`bench-results/over50_batch_prefix_n1000_043501`) measured normal forced-MMA
  `46.58 t/s` versus repaired batch approximate `45.02 t/s`. Batch variants
  (`bench-results/over50_batch_prefix_variants_044412`) topped out at
  `45.98 t/s` with `DS4_DSPARK_BATCH_DECODE_ORDER=1`. Keep the repair as useful
  diagnostic hygiene, not as the active speed track.
- Scheduler/operating-point sweep after the draft grouped-strided fix:
  confidence scheduling (`bench-results/over50_conf_sched_040244`) peaked at
  threshold `0.4`, `47.79 t/s` on n=300, but higher thresholds mostly disabled
  useful speculation (`37 t/s`). Combining confidence `0.35-0.45` with
  frontier-after-full (`bench-results/over50_frontier_conf_040521`) stayed in
  the `47.16-47.71 t/s` range. Context-size sweep at `-c 4096/8192/16384`
  (`bench-results/over50_ctx_sweep_041344`) stayed at `45.94-46.97 t/s`; the
  raw-cap warning is not the missing verifier speed. PrefixN disable
  (`bench-results/over50_prefix_disable_041729`) regressed to `39.34 t/s`
  because partial-accept replay adds about `22 ms/block`, so prefix capture
  remains required.
- Slot-bank operating-point sweep:
  `--resident --moe-slot-bank 64/128` failed allocation in this sidecar mode,
  while 256 stayed at `46.76 t/s`
  (`bench-results/over50_slotbank_sweep_041618`). Non-resident direct-mmap
  slot-bank runs (`bench-results/over50_stream_slotbank_041913`) were much
  slower (`14.20-21.34 t/s`), so DSpark verifier speed still requires the
  resident 256-slot target bank on this setup.
- Current-turn operating-point checks:
  real confidence scheduling requires `--draft-scheduler confidence` as well as
  `--draft-conf-threshold`. The n=1000 `0.4` check
  (`bench-results/over50_conf_scheduler_n1000_044255`) measured only
  `46.89 t/s`, avg scheduled `4.70`, tau `4.84`; it did not cross 50. Lowering
  routed expert top-k with `DS4_FLASH_MOE_EXPERT_TOPK=5` is not a cheap fast
  mode: the run failed at DSpark draft warmup because the draft/runtime assumes
  the model's full six active experts
  (`bench-results/over50_topk_sweep_043303`). Route-overlap sizing on current
  forced-MMA b5 (`bench-results/over50_route_overlap_043125`) confirms the real
  MoE opportunity: active-5 blocks showed average reuse around `1.7-1.8x`, with
  examples from `1.47x` to `2.01x` and many layers at `reuse>=1.50`.
- Correction-token audit:
  DSpark/DS4 currently returns only tokens whose target state has already been
  committed into the session. On a verifier mismatch the target correction token
  is known from `s->logits` or `row_tops`, but its KV/logits state is not
  committed. Appending that token without evaluating it would leave the next
  sampling step on stale state and corrupt generation. Evaluating the correction
  exactly inside `ds4_session_eval_speculative_argmax` is equivalent to the next
  outer decode step and does not create a free tau win. Do not pursue a
  "bonus/correction token" patch unless the session API is changed to support a
  pending unevaluated token or a true Mode-B committed batched state.
- Cheap-stack recheck after the correction audit:
  `bench-results/over50_combo_probe_050258` measured forced-MMA b5 at
  `46.87 t/s` on n=300. Adding the real confidence scheduler
  (`--draft-scheduler confidence --draft-conf-threshold 0.4`) improved the short
  run to `48.42 t/s`, tau `5.17`, full accept `74.1%`; frontier-after-full did
  not improve it (`48.10 t/s`), and unordered Q2 lost enough tau to fall to
  `47.75 t/s`. The n=1000 confirmation
  (`bench-results/over50_conf_mma_n1000_confirm_050540`) measured only
  `46.88 t/s`, tau `4.84`, full accept `69.9%`. Treat confidence scheduling as
  a useful operating point, not the missing >50 lever.
- Confidence fast-Markov patch:
  confidence scheduling no longer forces the slow per-row Markov path. The fast
  Markov chain can now save each row's Markov embedding into `batch_low_tmp` and
  run the same CPU confidence dot afterward. Disable with
  `DS4_DSPARK_CONF_FAST_MARKOV_DISABLE=1` to restore the older path. A/B
  `bench-results/over50_conf_fast_markov_ab_053746` stayed output-identical
  (`cmp=0`) and improved n=300 from `47.82` to `48.90 t/s`, with draft
  `12.13 -> 11.18 ms`. n=1000 confirmation
  `bench-results/over50_conf_fast_markov_n1000_053914` measured `47.49 t/s`,
  draft `11.24 ms`, verify `61.77 ms`, tau `4.84`, and full accept `69.9%`.
  Stacking the existing non-byte-identical fast-Q2 routed path with this patch
  reached the current best observed n=1000 run:
  `bench-results/over50_fast_conf_fastq2_n1000_054201`, `48.00 t/s`, tau
  `4.97`, acceptance `83.0%`, full accept `71.6%`. A narrow fast-Q2 threshold
  sweep (`bench-results/over50_fastq2_conf_threshold_sweep_054318`) did not
  cross 50. Final current-binary recheck after bulk confidence cleanup
  (`bench-results/over50_current_best_n1000_055324`) measured `47.97 t/s` with
  the same tau/acceptance. The cheap flag/scheduler stack is now exhausted.
- Corrected batch-approx + confidence + MMA check:
  `bench-results/over50_batch_conf_mma_probe_051941` reran the repaired
  batch-approx verifier with both confidence scheduling and
  `DS4_DSPARK_ATTN_FORCE_MMA=1`. It is still slower than strict confidence+MMA:
  base confidence+MMA `48.48 t/s` on n=300, batch-approx confidence+MMA
  `44.83 t/s`, and batch-approx decode-order confidence+MMA `44.87 t/s`. The
  batch path had similar verify wall (`~65.6 ms`) but lower tau (`4.76` vs
  `5.17`). Do not spend more time on current batch-approx variants as the >50
  path.
- Pair2 grouped-Q2 down diagnostic:
  added opt-in `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_PAIR2=1` for the
  grouped shared-weight Q2 down path. It specializes the common multiplicity-2
  duplicate-expert case and falls back to the exact descriptor wrapper for
  larger groups. Compare smoke
  `bench-results/over50_pair2_q2_smoke_051427` showed active-5 and active-2
  slot-down compares exact (`max=0`), unlike the older grouped shared-weight
  kernel. It is still not a speed path: no-compare n=300
  `bench-results/over50_pair2_q2_speed_051621` measured base forced-MMA
  `46.72 t/s`, pair2 `44.69 t/s`, and pair2+confidence `46.39 t/s`. Keep this
  diagnostic opt-in; do not promote it.
- Case E top-k sparse first prototype:
  `DS4_DSPARK_ATTN_NAX=1 DS4_DSPARK_ATTN_NAX_TOPK=1
  DS4_DSPARK_INDEXER_TOP_K_OVERRIDE=128`, b5, n=1000
  (`bench-results/nax_topk128_b5_n1000_011632`) measured `39.19 t/s`,
  verify `74.44 ms`, tau `4.69`, acceptance `73.9%`. This is a regression
  versus forced-MMA b5 and not a win at the standard c4096 target. Treat the
  current implementation as a long-context diagnostic only; the remaining useful
  Case E idea is a better bounded selected-key stream, tested where dense
  compressed attention actually crosses strict top-k cost.
- Strict top-k override without Case E:
  `DS4_DSPARK_INDEXER_TOP_K_OVERRIDE=128`, b5, n=1000
  (`bench-results/topk128_strict_b5_n1000_011742`) measured `39.87 t/s`,
  verify `76.44 ms`, tau `4.85`, acceptance `77.2%`. Lowering top-k alone does
  not buy speed in the normal short-context target.
- Frontier/predraft current-frontier diagnostic:
  `DS4_DSPARK_FRONTIER_DRAFT=1 DS4_DSPARK_ATTN_FORCE_MMA=1`, b5, n=1000
  (`bench-results/frontier_mma_b5_n1000_012429`) measured `38.63 t/s` despite
  high pos1 acceptance (`97.4%`) because overhead/commit ballooned to
  `14.61/14.26 ms`. Do not promote this as a speed path without splitting
  full-accept vs partial-replay timing and fixing commit overhead.
- Frontier prefix-commit repair:
  the frontier diagnostic now uses captured prefix state for partial accepts
  when `commit_drafts <= decodeN_capture_prefix_count`, instead of restoring the
  target frontier and replaying the accepted prefix through exact decode. This
  fixed the overhead pathology but did **not** cross the speed target:
  `bench-results/frontier_prefix_mma_b5_n160_013356` measured `44.21 t/s` with
  block `67.84 ms`, and
  `bench-results/frontier_prefix_mma_b5_n1000_013441` measured `44.64 t/s`,
  block `76.25 ms` (draft `15.50`, verify `59.07`, overhead `1.68`, commit
  `1.34`), tau `4.44`, acceptance `81.9%`. Budget 4
  (`bench-results/frontier_prefix_mma_b4_n1000_013548`) was slightly worse at
  `44.40 t/s`. Conclusion: prefix commit is fixed and useful hygiene, but
  frontier drafting loses enough tau versus the normal post-token draft that it
  only ties current forced-MMA b5.
- Frontier-after-full diagnostic:
  `DS4_DSPARK_FRONTIER_AFTER_FULL=1` gates frontier drafting so it only runs
  after the previous DSpark block fully accepted. The goal was to keep the
  high-pos1 frontier path while avoiding wasted first-token-miss drafts. Fresh
  b5/n1000 A/B (`bench-results/over50_frontier_after_full_031758`) measured
  base forced-MMA `46.55 t/s`, plain frontier `46.92 t/s`, and gated frontier
  `46.83 t/s`. The gated mode raised tau versus plain frontier (`4.71` vs
  `4.44`) but raised verify/block wall (`63.57 ms` vs `59.42 ms`) and did not
  cross the >50 target. Keep it as an opt-in diagnostic, not a primary speed
  path.
- Follow-up frontier heuristic sweep:
  `bench-results/over50_frontier_after_hit_034429` rechecked plain frontier,
  after-full, and a temporary after-first-hit gate at b5/n1000. Plain frontier
  measured `47.01 t/s`; after-full measured `47.39 t/s`; after-hit measured
  `47.32 t/s`; after-hit + fast-Q2 measured `46.79 t/s`. The after-hit code was
  removed after the test because it did not beat after-full. Frontier b4/b5
  sweeps (`bench-results/over50_frontier_budget_sweep_032835` and
  `bench-results/over50_frontier_b4_confirm_033045`) also stayed below 50:
  b4 confirmed `46.68-46.72 t/s` at n=1000.
- Row-routed MoE timing-only skip probe:
  `DS4_DSPARK_SKIP_ROW_ROUTED=1 DS4_DSPARK_ATTN_FORCE_MMA=1`, b5, n=300
  (`bench-results/over50_skip_routed_015744`) dropped verify to `19.44 ms/block`
  and block wall to `35.90 ms`, but acceptance collapsed to `8.9%`, tau to
  `1.44`, and generation to `22.95 t/s`. This is not a valid approximate path.
  It does size the prize: routed MoE is a major part of the remaining forced-MMA
  verifier wall, but it must be computed accurately enough to preserve draft
  acceptance/state. Do not use skip/zeroed MoE as a speed mode.
- Existing grouped MoE flag recheck:
  current-branch b5/n300 sweep
  (`bench-results/over50_moe_existing_flags_032204`) reconfirmed that the
  already-available grouped route diagnostics are not the speed path. Base
  forced-MMA measured `46.72 t/s`, grouped-Q2 safe measured `44.79 t/s`,
  grouped IQ2 gate/up measured `40.35 t/s`, and grouped IQ2 + grouped-Q2 safe
  measured `39.21 t/s`, all with identical acceptance. The next MoE task must
  be a new byte-safe route-dedup design that reuses expert loads while
  preserving row/slot outputs and ordered FP32 accumulation, not another toggle
  of the existing grouped descriptors.
- Existing slotwise row-routed MoE recheck:
  `bench-results/over50_slotwise_moe_034048` measured base `47.02 t/s`,
  `DS4_DSPARK_HYBRID_ROW_ROUTED_SLOTWISE=1` at `41.85 t/s`, slotwise+fast-Q2 at
  `42.75 t/s`, and disabling batch-row-exact at `43.67 t/s`. These paths
  preserve acceptance but increase verifier wall time, so they are not the next
  MoE route.
- Dense rows verifier flag recheck:
  `bench-results/over50_dense_rows_flags_033309` measured base `46.79 t/s`.
  `DS4_DSPARK_VERIFY_F16_ROWS5=1` regressed to `44.47 t/s`,
  `DS4_DSPARK_VERIFY_F16_ROWS5_SEQ=1` to `44.52 t/s`,
  `DS4_DSPARK_OUTPUT_LOW_Q8_ROWS5=1` to `46.41 t/s`, and the combined mode to
  `45.79 t/s`. Q8 rows5 remains useful and default; these extra dense rows
  knobs should stay off.
- Draft layer-count diagnostic:
  A temporary `DS4_DSPARK_DRAFT_LAYER_LIMIT` experiment
  (`bench-results/over50_draft_layer_limit_033748`) proved the DSpark-5 draft
  needs all three draft layers. Two layers improved draft time to `8.63 ms` but
  collapsed acceptance to `28.0%` and generation to `29.87 t/s`; one layer
  collapsed acceptance to `15.6%` and generation to `25.34 t/s`. The code knob
  was removed after measurement.
- Partial target-layer verifier sweep:
  `DS4_DSPARK_ATTN_FORCE_MMA=1 DS4_DSPARK_DECODE2_LAYER_LIMIT={43,42,40,36,32,28}`,
  b5, n=300 (`bench-results/over50_layerlimit_020451`) shows the cheap
  partial-layer shortcut is not viable. Full 43 layers measured `44.86 t/s`,
  verify `63.94 ms`, tau `4.84`, acceptance `77.0%`. Limiting to 42 layers
  reduced verify to `46.40 ms` but collapsed tau to `2.63`, acceptance to
  `32.9%`, and generation to `29.18 t/s`. Lower limits stayed in the
  `23-28 t/s` range with `14.9-24.3%` acceptance. Do not use layer-limit
  verification as the non-byte fast mode; the logits need the full target stack.

Discarded predraft diagnostic reject (`DS4_DSPARK_PREDRAFT=1` in a local
throwaway branch, not kept in the runtime,
`bench-results/over50_predraft_005549`):

- generation fell to `33.40 t/s`
- acceptance fell to `62.0%`
- overhead/commit inflated to `~25 ms/block`

Do not pursue predraft as a shortcut in this form. It exposes a state/dependency
problem and pushes the loop into replay/partial paths.

Interpretation:

1. Tau health is not obviously broken at pos1/pos2. The lost opportunity is the
   late suffix/full-block tail, not a simple first-token parity bug.
2. Existing `--draft-mode batch|unified` is diagnostic only; it is slower than
   strict here and should not be treated as implemented Mode B.
3. Existing fast opt-ins top out at ~1.24x no-draft, well short of >50 t/s.
4. Rank by generation t/s and tau. Verify-ms alone is not enough.

Active priority after the latest Pro review:

0. **Measurement gate first:** every sweep must report pos-wise and conditional
   acceptance, full-accept rate, tau, draft ms, verify ms, and generation t/s.
   Rank by generation t/s and tau, not verifier-ms alone. This is now cheap, and
   it is the fastest way to separate a real throughput win from a lower-ms run
   that loses acceptance.
1. **Strict cmp=0 MoE route-dedup:** this is the main byte-clean verifier kernel
   project. Preserve row-exact router/top-k/weights, separate slot semantics or
   an exact equivalent, and ordered FP32 slot0..slot5 sum. Do not revive the
   rejected grouped shared-weight Q2 path or unordered `sum6`.
   The next MoE attempt should target lower-overhead exact expert reuse:
   dequant/load an expert tile once when multiple row/slot routes hit it, write
   the same per-row/per-slot down outputs as the strict path, then run the
   existing ordered FP32 sum or a proven bit-equivalent ordered sum. Gate with
   slot-down max-delta `0`, MoE-boundary max-delta `0`, then `cmp=0` at
   n=160/1000/4000.
2. **Case E top-k sparse, narrowed:** this is the highest-value Case E-specific
   patch, but it is not the main strict path. The first per-row selected-stream
   prototype regressed at c4096/n1000, so do not treat "top-k sparse" as already
   solved. The remaining Case E task is a better bounded selected-key stream for
   long context, validated at `-c 4096`, `16384`, `65536`, and `100096`, with
   budgets 4 and 5 and prompts long enough to cross the sparse-indexer
   threshold. Keep it opt-in unless `cmp=0` is proven. Its goal is to make Case E
   useful past the dense-vs-sparse crossover, not to replace the strict verifier.
3. **ANE draft overlap dependency audit:** separate agent/track only. First
   prove whether draft(k+1) can start before verify(k) commits. The expected win
   depends on full-accept rate and wasted speculative draft work, so do not count
   the full draft ms as hidden until this audit is done.
   Sizing note: current best forced-MMA b5 spends about `15.5 ms/block` in draft.
   If an ANE path hides most of that under GPU verification, the same measured
   run moves from about `44.6 t/s` toward the low `50s` without changing verifier
   math. However, the dependency audit must size the real hidden work as roughly
   `hidden_draft_ms = correct_path_reuse_rate * draft_ms - wasted_work_cost`.
   Optimistic full-accept predraft, branch/tree predraft, and post-verify ANE
   draft are different schedules with different ceilings; do not collapse them
   into a guaranteed 15 ms/block win.
4. **Dense projection row-wide / NAX work is demoted:** only pursue if a
   non-fenced microbench shows a real generation-tps win. Stop if it gives less
   than ~8% verify-ms improvement.
5. **Mode B / unified batch-canonical:** if the product target is true ~2x or
   >50 t/s quickly, this remains the big contract-changing route. It needs one
   committable batched target forward with its own canonical no-draft baseline;
   current `--draft-mode batch|unified` is diagnostic and does not deliver this.

Tau interpretation: pos1 is already near the expected ceiling on current best
forced-MMA b5, so do not chase a first-token parity bug unless a future run
regresses. The remaining tau upside is late suffix/full-block behavior, likely
from tree/spec scheduling or Mode B rather than a simple draft integration tweak.

Immediate next action from the Pro feedback: keep the new conditional acceptance
footer enabled for every serious run, then choose the product track explicitly:

- strict byte-clean work starts with MoE route-dedup and tau/full-accept analysis
- Case E work starts only if the question is long-context non-byte fast mode
- fast non-identical work starts with real Mode B/unified target forward

Do not spend another cycle on verifier attention microkernels unless the run is
specifically the long-context Case E selected-key experiment.

## Historical Attention Execution Plan — branch `dspark-attn` (2026-06-29, paused)

This section is retained as measurement history. It is no longer the active
priority order after the 2026-06-30 Pro feedback and fresh n=1000 sweep above.
Use it only for evidence about why Case E / attention kernels were demoted.

Execute this in order. **Ceiling-first gate is mandatory:** before bit-cleaning
any stage, measure its unsafe (`cmp=1`) ceiling; if it is below **+8% on a
non-fenced n=1000 run**, STOP and report — do not implement the byte-clean
version. No new env flag / probe until the current step's number exists.

### Step 0 — Settle the real attention budget (do FIRST, ~½ day)

The fenced stage profile (`copy=75%`) is inflated; the only unsafe-ceiling data
so far (`dspark_mixed_speed_ceiling`, +3–5%) is from a weak kernel. Get ground
truth with TWO non-fenced measurements:

1. **GPU active-vs-idle % during the verify block.** Use Metal command-buffer
   `GPUStartTime`/`GPUEndTime` (and/or `MTLCounterSampleBuffer`) summed over the
   verify block, compared to verify wall. Idle → latency-bound → fusion wins big.
   Busy → attention is compute-bound and near-tapped (~5% ceiling).
2. **True attention share of verify.** Add `DS4_DSPARK_ATTN_BYPASS=1`: in the
   verify attention path, short-circuit the committed-prefix attention (reuse
   prior `batch_heads`/zero), so verify runs end-to-end but attention is a no-op.
   The verify-ms delta vs normal = attention's real wall cost (output `cmp`
   invalid; timing valid).

Decision: idle GPU or attention ≥ ~25% of verify → proceed to Step 1. Otherwise
report the ceiling and pivot to tokens-per-block (tree spec).

#### Step 0 RESULT — measured 2026-06-29 on `dspark-attn` (GO)

Implemented `DS4_DSPARK_ATTN_BYPASS` (skips the attention/compressor/indexer
batch in the verifier; output invalid, per-block verify ms valid). Measured with
`DS4_DSPARK_PERF` at n=160/active-5:

- **normal verify = 70.26 ms/block**
- **attention bypassed = 14.84 ms/block**
- → **attention/compressor/indexer = ~55 ms = ~79% of verify.** Everything else
  (routed MoE + shared + router + FFN-pre + tail + head) is only ~15 ms.

N-sweep verify ms (n=96): N=2 38.9, N=3 40.1, N=4 59.8, N=5 63.0 → attention is
~10–12 ms **per row** (2→3 nearly free = latency/occupancy signature). At N=5
roughly **40 ms** is per-row replay that a shared-prefix read collapses.

**The "+3–5% mixed-shared ceiling" was a false alarm.** Re-measured the unsafe
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_EXACT=1 DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1`
kernel with PERF: **verify = 70.77 ms — unchanged from normal.** That kernel
never reduced attention; the +3–5% end-to-end deltas were noise. There is no
evidence against the win — it is simply untested. Disregard the
`dspark_mixed_speed_ceiling` "near-tapped" conclusion.

**New success metric (use this, not end-to-end t/s noise):** the Step-1 kernel
works iff **per-block verify ms (DS4_DSPARK_PERF) drops from ~70 toward ~30**
(15 ms non-attention floor + ~one-row batched attention). Target ~1.6–1.7× of
current strict, i.e. well past 2× of no-draft. A/B every change on verify ms vs
the 14.84 ms bypass floor, NOT on ±1 t/s headline runs.

#### Step 0.1 RESULT — GPU active/idle (added `ds4_gpu_busy_seconds`, footer line)

- normal verify = 69.2 ms, **GPU-busy 62.5 ms = 90% active** (only ~10% idle).
- bypass verify = 14.5 ms, GPU-busy 93% active.

**The verify is GPU-compute-bound, NOT dispatch/latency-bound.** This is why K/V
load-sharing (the team's staging flags) gave ~0% on verify ms: bandwidth was
never the limit. **Abandon the staging-flag direction entirely.**

The win is still real, and the diagnosis changes to *kernel efficiency*:
- FFN/MoE does 5 tokens in **14.5 ms** (batched, dense weights amortized).
- Attention does the same 5 tokens in **55 ms** — ~5× a single decode's attention
  (~11 ms), because it runs the **decode-style per-row matvec kernel**
  (`kernel_flash_attn_varmap_rows5`), not a batched GEMM/MMA.
- A single decode's FFN/MoE is ~17 ms, yet verify does 5-token FFN/MoE in 14.5 ms
  → proof that proper batching collapses 5 rows toward 1×. Attention never got it.

**Revised Step 1 (supersedes staging/shared-tile work):** replace the per-row
varmap attention with a **batched flash-attention kernel that runs the N≤5 verify
queries as a GEMM** (simdgroup_matrix / steel-style; the MLX `steel/attn` pattern).
Target: attention 55 → ~13–15 ms, verify → ~29 ms, ~1.7×. The hard part is
byte-identity (MMA accumulation order ≠ decode reduction order) — either replicate
the decode FP reduction order inside the MMA kernel, or move to Mode B (canonical
batched forward, argmax-accept, no row-decode `cmp=0`).

#### Step 0.2 RESULT — copy-vs-compute decomposition (added `DS4_DSPARK_ATTN_NOCOPY`)

Verify decomposed (n=160, active-5):

| component | ms/block |
| --- | ---: |
| attention COMPUTE (per-row matvec Q·K/softmax/A·V) | **47.5** |
| F32→F16 staging copy | 12.8 |
| FFN/MoE + rest (bypass floor) | 14.4 |
| verify total | 74.7 |

So the bottleneck is **attention compute (47 ms), not the copy (13 ms).** Copy is a
secondary byte-safe-ish win (resident F16 KV / direct read — but the team's
`direct` reads were slower due to irregular access). GPU stays 91% active across
all three, confirming compute-bound.

**Efficiency proxy:** model prefill (the efficient MMA kernel `kernel_flash_attn_ext`,
high batch) = **4.64 ms/token for the *whole* forward** (215 t/s). The verify's
**attention alone** is 47.5/5 = **9.5 ms/token** — 2× the prefill's entire-forward
per-token cost. So the per-row matvec attention is genuinely inefficient (runs far
below peak; "busy but slow").

**Realistic win (revised down from 1.7×):** a batched MMA flash-attention kernel
at N=5 is occupancy-limited (5 rows ≪ a prefill chunk), so it won't reach
prefill's per-token efficiency. Expect attention 47 → ~25 ms, verify 74 → ~52 ms
→ **~1.3–1.4×** of current strict (not 1.7×). The copy (13 ms) on top, if also
eliminated byte-safely, could push a bit further. Only building the MMA kernel
settles the exact number.

**The four diagnostics now in the binary (keep them):** `DS4_DSPARK_ATTN_BYPASS`
(attention share), `ds4_gpu_busy_seconds`/footer (GPU active%), `DS4_DSPARK_ATTN_NOCOPY`
(copy vs compute). A/B all future attention work on **verify ms**, decomposed
against these floors — never on ±1 t/s headline runs.

#### Step 1 RESULT — forced MMA path measured (added `DS4_DSPARK_ATTN_FORCE_MMA`) — STOP signal

Routed the verify's N≤5 queries through the existing efficient MMA kernel
(`kernel_flash_attn_ext_f16_dk512_dv512` via `decode_mixed_batch_heads`):

| path | verify ms/block | cmp |
| --- | ---: | ---: |
| varmap (per-row, current strict) | 69.4 | 0 |
| **FORCE_MMA (batched MMA)** | **58.9** | **1 (diverges)** |

**The efficient MMA kernel gives only ~15% (69→59 ms), and diverges.** GPU is 90%
active in both → attention is genuinely **compute-bound**: the 5 distinct queries
inherently need ~5× the dot-product work, and MMA only recovers the K/V-load
redundancy + tiling (~10 ms). ~45 ms of the 55 ms attention is irreducible
compute. So no attention-kernel rewrite (load-share, MMA, fusion) beats ~15%.

**End-to-end:** verify 69→59 ms → block 88→78 ms → ~39→~44 t/s = **+13%** over
current strict (~1.33× no-draft vs current ~1.18×) — and only if the MMA path is
made byte-exact (hard) or shipped as Mode-B (output changes).

**Decision (per the user's step 3): the win is NOT enough.** A ~13% end-to-end
gain does not justify a multi-day byte-exact MMA kernel or the Mode-B contract
change. **Strict-v1 is at its practical ceiling on this compute-bound attention.**
Recommend: stop the attention-kernel track; bank the diagnostics. The only larger
levers left are non-kernel: tree speculation (more accepted tokens/block) or
GPU+ANE overlap (hide the draft) — both separate efforts, both modest.

Speculative note (from review): the flash kernels use simdgroup_matrix MMA, NOT
the NAX/MPP tensor path that dense Q8/MoE use. Whether attention on NAX beats MMA
is unknown, but with the GPU already 90% active and MMA only +15%, it's unlikely
to change the verdict; not worth pursuing before the non-kernel levers.

### Step 1 — One kernel: `dspark_attn_shared_prefix_mixed_exact_n5`

One threadgroup per `(layer, head, split-lane)`; the N≤5 verifier rows are
simdgroups with their own **F32 online-softmax (M,S)+AV** state. Five hard rules
(each fixes a documented failure):

1. **Read resident K/V in place — zero scratch.** No `raw-union` staging buffer
   (that copy is the bucket every prior variant kept).
2. **Preserve the strict 32-key chunk order.** Iterate chunk-by-chunk in decode
   order; load each shared chunk's K/V once for all rows executing it. (Phase-
   split mixed-prefix diverged because raw/compressed were separated.)
3. **Natural-lane ranges + remap fallback.** 96% raw / 94% comp shared keys land
   in the correct split lane in contiguous runs — consume the range descriptors
   from `ds4_gpu_attention_decode_varmap_rows_tensor`; scatter/remap only the ~5%.
4. **Tile K and V within threadgroup-memory limits** (K+V-together blew tg-mem;
   tile the chunk, do not fall back to K-only which re-reads V per row).
5. **Byte-identical: F32 accumulation, same order.** Candidate heads to scratch;
   promote only when candidate-vs-strict head `max=0`.

Bring-up, each gated to head `max=0`: (a) shared raw-intersection + private raw
tail; (b) add compressed common + tail in the same chunk order; (c) add the ≤5
block-tail per row in exact order; (d) wire authoritative.

### Step 2 — Extend the same kernel pattern to compressor + indexer

`heads≈215` is one third of the 525 row-scaled dispatches; `comp≈205` +
`index≈105` are the rest with the same shared-prefix structure. Apply the Step-1
pattern once heads clears its gate.

### Gates (all steps)

- Ceiling-first +8% (above).
- Promotion: head `max=0` → `cmp=0` at n=160/1000/4000 → block-1/501 audits clean
  at `1e-8`.
- Report speed only from non-fenced runs.

---

Current direction, 2026-06-28: keep strict-v1 as the production fallback, keep
Mode B as a diagnostic throughput contract, and add **Plan C: unified target
forward** as the main research path. The default strict DSpark-5 path is still
the golden production baseline: it is commit-safe, byte-clean against current
DS4 no-draft greedy, and already modestly faster than the local no-draft
baseline. The new bet is not to keep forcing verifier family B to match decode
family A forever; it is to make no-draft decode and DSpark verify share one
target-forward kernel family. Delayed Pro feedback turns this into three
explicit tracks rather than a pivot away from existing work: Plan A is strict old
DS4 compatibility, Plan B is batch-canonical, and Plan C is unified-kernel
greedy.

Latest Plan C checkpoint from the attention row-shape profiler:
`raw_same_count=0/43` means a same-shape rows5 attention shortcut is not viable.
The useful result is the scan reuse estimate instead: active-5 shows about
`3.72x` raw-prefix reuse and `3.85x` compressed-prefix reuse on the first-block
row-shape profiler; the fuller n=320 rows-exact aggregate estimates `4.46x`
raw-prefix reuse and `4.79x` compressed-prefix reuse for the remaining
mixed/compressed path. Treat
same-mask/same-span rows5 attention and threadgroup-only rows5 fusion as closed
diagnostics. The raw vec-rows landing point is byte-clean but noise-level for
speed. The rows-exact intersection profile then found `same_counts=720` but
`same_shape=0` at n=320, so a narrow same-shape mixed/compressed rows5 shortcut
also has no coverage. The next speed kernel is still shared-prefix attention,
but not the current plain mixed online fusion. The rejected
`kernel_dsv4_plain_mixed_attention_*` probe drifts by about `1e-6` even when the
row geometry is the easy prefix case (`raw_start=0`, shared raw intersection,
and shared compressed prefix). The strict-compatible version must preserve the
existing FlashAttention vec/reduce realization while sharing committed raw and
compressed K/V scans across N<=5 rows, or Plan C must explicitly switch to a
unified canonical N=1/N<=5 target-forward baseline. The existing raw vec-rows
backend is only the byte-clean landing point; it does not yet share the prefix
scan and covers a minority of calls.

Latest implementation checkpoint:

- Local reference clone: `/tmp/dspark-research/mlx-vlm` at commit `78b96eb`.
  Relevant source:
  `mlx_vlm/speculative/mtp.py`,
  `mlx_vlm/models/deepseek_v4/language.py`,
  `mlx_vlm/speculative/drafters/deepseek_v4_mtp/deepseek_v4_mtp.py`,
  `mlx_vlm/models/deepseek_v4/hisa_kernel.py`, and
  `mlx_vlm/models/deepseek_v4/hyper_connection.py`.
- The MLX source confirms the main Plan C contract: verify the whole block with
  the target forward, accept by argmax walk, then trim/zero rejected cache state.
  It does not prove DS4 will inherit MLX/Gemma speedups, but it does prove the
  right comparison target is a unified target-forward family, not cross-family
  bit-matching forever.
  Concrete files to keep open while implementing Plan C:
  `mlx_vlm/speculative/mtp.py::_mtp_rounds()` and `_mtp_rounds_batch()` build
  `verify_input = [bonus, draft_tokens]`, call `_mtp_verify_target()`, walk
  target-vs-draft tokens via `_mtp_acceptance_walk()`, and call
  `rollback_speculative_cache()` on rejection. DeepSeek V4's
  `mlx_vlm/models/deepseek_v4/language.py::_speculative_verify()` calls the
  same model `__call__()` for the verify block; `rollback_speculative_cache()`
  snapshots/restores/replays when cache state cannot be trimmed cheaply, or
  trims/zeros rejected tails otherwise. The local/compressed/sparse attention
  classes in that file update/fetch cache once for batched `L` rows and then
  call attention over the local/compressed/sparse K/V stream. DS4 cannot copy
  the MLX kernels, but Plan C should match this unified-forward contract.
- DS4 now has an opt-in varstream rows5 attention prototype:
  `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1`. It is byte-clean and locally exact after
  staging Q as `half4`. It is not yet a durable speed win: n=160 smoke measured
  `40.40 t/s`, but n=1000 measured `38.14 t/s`, below the prior strict active-5
  n=1000 best of `39.04 t/s`. This means the next patch must add real shared-prefix
  K/V tile reuse or move to the unified target-forward contract; small host-side
  cleanup alone is not enough.
- DS4 also has an opt-in varmap rows5 attention prototype:
  `DS4_DSPARK_ATTN_VARMAP_ROWS=1`. It is byte-clean and currently the best
  medium-length Plan C result: n=1000 measured `41.07 t/s` with `cmp=0` and
  `77.1%` acceptance, versus paired no-draft `35.36 t/s` and strict active-5
  `39.04 t/s`. It is not durable enough to promote: n=4000 measured
  `35.94 t/s`, below strict active-5 `36.48 t/s`.
- A direct-resident follow-up,
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_ROWS=1`, removed the scratch F16 stream and
  compared exact (`max=0 rms=0`, `cmp=0`), but was slower (`36.61 t/s` at n=160
  and `35.65 t/s` at n=1000). Treat direct-resident as a diagnostic: reading
  resident F32 K/V directly costs more than the scratch F16 staging it removes.
- Dynamic split-count flags were added for the rows5 prototypes:
  `DS4_DSPARK_ATTN_VARSTREAM_DYNAMIC_NWG=1`,
  `DS4_DSPARK_ATTN_VARMAP_DYNAMIC_NWG=1`, and
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_DYNAMIC_NWG=1`. Dynamic varmap is byte-clean:
  n=160 `38.40 t/s`, n=1000 `39.82 t/s`, n=4000 `36.64 t/s`, all `cmp=0`.
  This slightly improves long varmap but does not change the conclusion: the
  next real speed lever is resident shared-prefix raw/compressed K/V tile reuse
  and unified target-forward state, not more scratch/dispatch trimming.
- Varstream precedence is now fixed. Explicit `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1`
  or compare no longer gets shadowed by default varmap rows; smoke
  `bench-results/dspark_varstream_precedence_041228` logged varstream without
  requiring `DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE=1` and stayed `cmp=0` at n=64.
  Fresh n=160 A/B `bench-results/dspark_varstream_recheck_040853` is byte-clean
  but keeps varstream below default varmap: default varmap `40.07 t/s`,
  row-exact fallback `38.83`, varstream `38.95`, varstream dynamic `38.69`.
  Do not spend more time optimizing varstream scratch staging unless it is a
  stepping stone toward resident shared-prefix tile reuse.
- Current row-shape proof for the next kernel:
  `bench-results/dspark_rowshape_probe_041550` shows active-5 raw rows shaped
  like `[21,22,23,24,25]`, `[26,27,28,29,30]`, etc., with `raw_common` equal to
  the first row and `raw_tail=[0,1,2,3,4]`. Ratio-4 compressed rows also have a
  substantial common prefix plus a small tail. This rules out simply reusing the
  generic rows5 K-stage/kvstage against varmap's row-local `ic`: after the common
  region, the same `ic` points at different raw-union offsets per row. The next
  exact speed kernel must be common-prefix plus row-tail online-softmax combine,
  not same-`ic` tile staging.
- `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` now auto-selects varmap
  rows5 attention inside the strict-v1 delegate instead of only the older raw
  vec-rows landing point. Opt out with
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_ROWS_DISABLE=1`. Current paired
  n=1000 tests after the switch are all byte-clean with identical acceptance,
  but not faster: default strict `38.48 t/s`, verifier-only unified
  shared-prefix `38.33 t/s`, and `--draft-mode unified` shared-prefix
  `38.36 t/s`. Treat this as Plan-C scaffolding, not the speed kernel.
- Gate fix after that switch: setting `DS4_DSPARK_ATTN_VARMAP_ROWS=1` now
  activates deferred row-exact attention by itself, so varmap cannot silently
  bypass the intended path when the older defer envs are omitted. The
  rows-exact attention profile now includes host-side encode, finish, and total
  milliseconds for the rows-exact and varmap helper paths.
- Direct-resident gate fix: `DS4_DSPARK_ATTN_VARMAP_DIRECT_ROWS=1` and direct
  compare flags now also activate the deferred attention path instead of
  silently bypassing it. Current n=160: direct-resident is byte-clean but slow
  (`32.73 t/s`, verifier `94.33 ms`); direct+dynamic recovers to `37.52 t/s`,
  verifier `75.81 ms`, but still trails scratch varmap around `38.4 t/s`.
  Keep direct-resident diagnostic-only.
- Post-hook profile:
  `bench-results/dspark_rows_exact_time_profile_varmap_n1000_after_hook` is
  byte-clean (`cmp=0`) at `38.37 t/s`, acceptance `77.1%`, block timing
  `draft=18.02 ms verify=76.25 ms overhead=2.00 ms tau=4.85`. The varmap helper
  consumed only `8.938 ms` host-side total across `7421` calls and `37105` rows.
  A pure rows-exact fallback n=160 profile consumed `25.529 ms` host-side total
  across `1189` calls while producing essentially the same generation speed.
  Treat varmap as useful host encode cleanup but not the missing speed lever.
- Varmap stage profile:
  `DS4_DSPARK_ATTN_VARMAP_STAGE_PROFILE=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_STAGE_PROFILE=1`) forces scratch
  varmap attention through separate copy/stage, vec, and reduce command-buffer
  fences and extends the aggregate footer with `stage_ms` fields. This is a
  microscope only. `bench-results/dspark_varmap_stage_profile_061339` stayed
  `cmp=0` at n=64, but slowed to `30.39 t/s`. The stage split was
  copy/stage `964.213 ms` total over `492` calls (`1.960 ms/call`), vec
  `162.802 ms` (`0.331 ms/call`), and reduce `121.803 ms` (`0.248 ms/call`).
  Since the same run reports raw and compressed reuse around `4.2x`, the next
  Plan C kernel should remove scratch F32->F16 stream staging for the committed
  raw/compressed prefix, or make that shared-prefix stage resident, before
  spending time on reduce-only micro-optimizations.
- Compressed-only varmap shadow probe:
  `DS4_DSPARK_ATTN_VARMAP_COMP_F16_SHADOW=1` and its
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_COMP_F16_SHADOW=1` alias are
  correct but not a speed path. `bench-results/dspark_varmap_comp_shadow_fix_063107`
  stayed `cmp=0` at n=64, with default DSpark `37.52 t/s` and shadow
  `37.42 t/s`. Fenced follow-up
  `bench-results/dspark_varmap_comp_shadow_stage_063245` reported copy/stage
  `961.712 ms` default versus `967.990 ms` with shadow, so compressed-only
  staging does not shrink the dominant copy bucket. Leave it off by default and
  move the next implementation target to resident raw plus compressed
  shared-prefix attention.
- Raw-shadow/direct-F16 probe:
  `DS4_DSPARK_ATTN_VARMAP_RAW_F16_SHADOW=1` and
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_F16_SHADOW=1` are also correctness scaffolds,
  not speed paths. `bench-results/dspark_varmap_raw_shadow_064157` stayed
  `cmp=0`, but default `37.56 t/s` beat raw-shadow `37.28 t/s`; fenced
  `bench-results/dspark_varmap_raw_shadow_stage_064338` showed copy/stage
  worsening from `942.495 ms` to `953.118 ms`. Direct-F16 shadows in
  `bench-results/dspark_varmap_direct_f16_064826` also stayed `cmp=0`, but
  measured only `30.65 t/s` versus default `37.45 t/s` and direct-F32
  `36.93 t/s`. Do not pursue shadow-to-scratch blits or irregular direct-F16
  reads as the main optimization. The remaining attention prize is true
  committed-prefix sharing inside the varmap/FlashAttention kernel.
- Post-fusion-guard Phase-0 refresh:
  `bench-results/dspark_phase0_post_fusion_guard_035057` used n=320 with b4/b5,
  route overlap, dispatch, block timing, shared-prefix, and attention row-shape
  profiles. Both budgets stayed byte-clean. Baseline was `33.12 t/s`; b4 was
  `35.79 t/s`, acceptance `84.0%`, verify `75.01 ms`, tau `4.32`; b5 was best at
  `38.12 t/s`, acceptance `84.6%`, verify `87.94 ms`, tau `5.23`. b5 still shows
  the major row-scaled dispatch buckets (`heads=215`, `comp=205`, `index=105`)
  and strong first-block scan reuse (`raw=3.71x`, `compressed=3.86x`), so active-5
  remains the target for the shared-prefix mixed/compressed kernel.
- Current clean refresh:
  `bench-results/dspark_clean_refresh_045241` measured no-draft `33.25 t/s` and
  active-5 `41.20 t/s` at n=320 (`cmp=0`, acceptance `84.6%`, draft `18.12 ms`,
  verify `77.57 ms`, tau `5.23`). `bench-results/dspark_budget_sweep_refresh_045720`
  reran budgets 2..5 at n=320 and all outputs matched baseline (`cmp=0`):
  b2 `33.98 t/s`, b3 `36.94`, b4 `39.24`, b5 `41.27`. Active-5 is still the
  right headline target; the next speed work should reduce the N=5 verifier
  rather than lowering the draft budget.
- Varmap split-count probe:
  `DS4_DSPARK_ATTN_VARMAP_NWG=<1..32>` /
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_NWG=<1..32>` and the direct-resident
  `*_VARMAP_DIRECT_NWG` variants are diagnostic-only. In
  `bench-results/dspark_varmap_nwg_probe_035519`, forced `nwg=1` kept final n=160
  output byte-clean but slowed default `39.99 t/s` to `38.28 t/s`, dropped
  acceptance `76.4%` to `68.3%`, and all-layer compare showed `max_delta=1.9e-6`
  against rows-exact; default dynamic split compared exact (`max_delta=0`).
  Do not promote split-count forcing.
- Mixed shared geometry proof:
  `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE_GEOM=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_COMPARE_GEOM=1`) logs row geometry
  beside the mixed-shared candidate-vs-strict head delta. Force it with
  `DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE=1` when default varmap rows would
  otherwise take precedence. `bench-results/dspark_mixed_shared_geom_forced_052550`
  stayed `cmp=0` in compare/restore mode and showed the important failure
  shape: at layer 2 `pos=20`, all rows had `raw_start=0`, raw intersection was
  `[0,21)`, compressed common was `5`, and max head delta was
  `9.53674e-07`. Later rows with larger prefix/intersection showed the same
  `~1e-6` band. Conclusion: this is wrong attention fusion, not wrong
  raw-ring visibility or shared-prefix geometry. The no-compare path is now
  guarded; use `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1` for compare/restore, or
  explicitly set `DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1` only to reproduce the
  rejected committed diagnostic.
- Add MTP as a one-off report-card slice for shared verifier work. MTP TP=2 has
  smaller upside than DSpark N=5, but it is a useful proof that Plan-C changes
  are target-verifier improvements rather than DSpark-only effects. Do not
  branch the optimization loop into MTP. For meaningful verifier changes, record:
  no-draft baseline, literal default MTP TP=2 baseline, current MTP TP=2
  sidecar-batch baseline, MTP TP=2 strict-v1, and any
  Plan-C/shared-prefix diagnostic row. Strict rows must keep `cmp=0` against
  no-draft greedy where expected. Columns: draft kind, backend, active N,
  generation t/s, `cmp`, acceptance, accepted tokens per round/tau,
  draft/verify/overhead ms per block, and verifier flags.
  Literal default TP=2 was measured with MTP verifier env flags unset:
  exact-decode2 verifier, n=1000 `30.21 t/s`, `cmp=0`, acceptance `89.7%`,
  versus paired no-draft `32.88 t/s`.
  TP=3 was checked as a sanity row: it really uses `n=3` verifier rows, not a
  padded N=5 verifier, but it is slower than no-draft on this target
  (`28.20-28.43 t/s` for current/unified-batch at n=1000, `23.17 t/s` for
  shared-prefix, fresh no-draft `32.88 t/s`) and does not pass the long
  no-draft compare. Keep TP=3 archived as evidence, not as an optimization lane.

## Current State

Strict-v1 baseline:

- Default DSpark-5, active 5.
- `cmp=0` through n=1000 and n=4000.
- Focused block-1 and block-501 audits are clean at `1e-8`.
- Row-local compressor/indexer mutation is required.
- Ratio-128 attention-compressor rows and ratio-4 indexer Q/weight projections
  are row-local in decode-order mode.
- Row-QKV, row-output, row-router, row-routed, row-shared, exact ordered
  routed-MoE sum, and prefix-N commit are the current strict contract.
- Fine attention subprofiling now exists behind
  `DS4_DSPARK_ATTN_SUBPROFILE=1 DS4_DSPARK_ATTN_FINE_SUBPROFILE=1`. It is
  fenced and diagnostic only. First active-5 smoke:
  `bench-results/dspark_fine_subprofile_055320`, `cmp=0`; normal non-profile
  companion run was 37.41 t/s versus 33.33 t/s baseline, with verify
  66.34 ms/block. The fenced split shows compressor update/capture and
  indexer-compressor update/capture dominate the strict attention state work on
  the short prompt (`comp_update ~=53 ms`, `comp_capture ~=44 ms`,
  `idx_comp_update ~=28 ms`, `idx_comp_capture ~=22 ms` per active-5 block
  under fences). Treat these as ranking signals, not wall-clock totals.
- The direct ordered-Q2 down+sum kernel is now the default strict local Flash
  IQ2/Q2 path. It proved byte-clean in full-output runs and exact at the
  MoE-boundary compare (`max=0`) for active-5 verifier calls. Disable with
  `DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` for the older
  separate Q2 down plus ordered-sum path. The focused proof hook is
  `DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1`. It is now symmetric:
  when separate is primary it runs direct Q2 into scratch, and when direct Q2
  is primary it runs separate down plus ordered-sum into scratch. Fresh guard
  `bench-results/dspark_direct_q2_primary_compare_051916` stayed `cmp=0` and
  logged 1720 exact boundary compares, including active-5
  `mode=primary-direct exact=yes`, with no `exact=no` lines. Extend MoE fusion
  only along this row-precise family: independent slot accumulators plus the
  exact slot0..slot5 ordered FP32 sum. Do not use grouped shared-weight or
  unordered sum6 as the production contract without an equally strong compare.

Headline strict evidence:

- Current clean n=1000 sweep:
  `bench-results/dspark_current_clean_sweep_012802` paired no-draft
  `32.92 t/s`; all DSpark active sizes were `cmp=0`. Active-5 remains the best
  default at `39.31 t/s` with draft `18.14 ms`, verify `73.04 ms`, tau `4.85`,
  and acceptance `77.1% (794/1030)`. Active-4 was close at `38.77 t/s`.
- Current direct ordered-Q2 n=1000:
  `bench-results/dspark_direct_q2_clean_ab_022733` stayed `cmp=0` and reached
  `39.66 t/s`, versus same-session conservative repeat `39.12 t/s`.
- Current clean n=4000:
  `bench-results/dspark_current_clean_n4000_013646` paired no-draft
  `31.62 t/s`; default strict active-5 was `36.47 t/s`, `cmp=0`, verify
  `79.61 ms`, tau `4.80`, acceptance `76.1% (1963/2580)`.
- Current direct ordered-Q2 n=4000:
  `bench-results/dspark_fused_ordered_q2_current_n4000` stayed `cmp=0` against
  the paired baseline/default output and reached `36.73 t/s`, verify `79.54 ms`,
  same acceptance.
- Direct-Q2 MoE-boundary compare:
  `bench-results/dspark_direct_q2_compare_buckets_022458` stayed `cmp=0` and
  printed exact matches (`max=0`) for `tokens=1`, active-5 verifier calls, and
  the active-3 tail. This says the direct down+ordered-sum math is exact for
  those calls; it does not validate grouped shared-weight MoE or deeper
  gate/up/SwiGLU/down fusion.
- Promoted-default validation:
  `bench-results/dspark_direct_q2_promoted_default_023137` stayed clean: n=160
  `cmp=0`; n=1000 matched the paired no-draft baseline output and reached
  `39.71 t/s`, acceptance `77.1% (794/1030)`.
- Clean active-size sweep after promotion:
  `bench-results/dspark_direct_q2_clean_sweep_023454` used n=1000 with heavy
  diagnostics off. Paired baseline was `33.01 t/s`; b2 `33.83`, b3 `36.70`,
  b4 `39.10`, b5 `39.64 t/s`. Active-5 remains the best strict default.
- Dispatch accounting fix:
  `bench-results/dspark_direct_q2_dispatchfix_024232` stayed `cmp=0` and now
  reports active-5 `ordered_sum=0`, `routed_moe=43`, total dispatch estimate
  `893` instead of stale `936`. Remaining row-scaled buckets are attention heads
  `215`, compressor `205`, and indexer `105`; these are the next verifier
  target, not ordered expert-sum.

Performance shape:

- Draft side: about `265 draft tok/s`, or `18.5-18.9 ms` for 5 draft tokens.
- Verifier side: about `58 proposed tok/s`; this is the bottleneck.
- Route-overlap smoke shows real routed-expert reuse on the local code prompt:
  active-5 blocks measured `1.66-1.84x` reuse across all target layers. That
  keeps grouped exact routed-MoE alive, but below Plan C shared-prefix attention
  and ROWS5/unified-forward work.
- The old `~146 ms/block` number is a fenced blocking stage profile, not normal
  verifier wall time. Use it only to rank buckets.
- Normal measurement should use non-fenced block timing and decode-equivalent
  ratios against a paired no-draft baseline:
  `draft_ms / baseline_decode_ms`, `verify_ms / baseline_decode_ms`,
  `overhead_ms / baseline_decode_ms`, and `tau`.

Dominant verifier buckets:

| Stage | Typical Cost |
| --- | ---: |
| attention / compressor / indexer row loop | ~61 ms |
| row-routed MoE | ~31 ms |
| row-shared expert | ~13 ms |
| row-router | ~13 ms |
| row-FFN-pre | ~11 ms |
| batched tail | ~13.5 ms |

## Contracts

### Strict V1

Freeze the current strict verifier as `strict_v1`.

- This is the current default and production-safe path.
- Do not mutate it directly for speculative speed experiments.
- It must keep `cmp=0` against no-draft greedy.
- It must keep clean block-1 and block-501 audits before being considered
  unchanged.

If adding a named implementation switch, use this shape:

```c
typedef enum {
    DSPARK_VERIFY_STRICT_V1 = 0,
    DSPARK_VERIFY_STRICT_V2_ATTN = 1,
    DSPARK_VERIFY_STRICT_V2_MOE = 2,
} dspark_verify_impl;
```

The default command must keep selecting `DSPARK_VERIFY_STRICT_V1` until a new
path passes the full gates.

### Mode B

Mode B remains useful as a future throughput contract and diagnostic upper
bound, but it is not the current priority. Do not use Mode-B self-consistency as
evidence that strict output is correct. Do not pivot into target-batch baseline
work until the strict verifier-cost path below has been pursued.

### Plan C: Unified Target Forward

Plan C is the new lead research track. It tries to preserve greedy identity by
making ordinary no-draft decode and speculative verification use the same
target-forward family:

```text
no-draft greedy: target_forward_rows_unified(N=1)
DSpark verify:   target_forward_rows_unified(N=2..5)
```

This differs from both strict row replay and the current unsafe batch state. The
first milestone is self-consistency, not speed:

- unified N=1 no-draft is deterministic and can run long prompts
- unified N=1 matches old no-draft at n=160/n=1000 if possible
- unified DSpark spec matches unified no-draft at temp 0
- unified verifier can commit its own hidden/KV/frontier state without the
  hidden/KV explosions seen in `BATCH_APPROX_STATE`

If unified N=1 matches old DS4 decode, Plan C can eventually satisfy both old
`cmp=0` and speed. If it does not, strict-v1 remains the old-compatible default
and Plan C becomes an optional fast greedy mode with its own no-draft baseline.

External MLX evidence to keep in mind:

- `README.md` reports Gemma 4 MTP byte-identical greedy speedups, but those numbers
  are not DeepSeek-V4-Flash DSpark numbers. Do not copy the headline as a DS4
  target without local validation.
- `speculative/mtp.py` forms `verify_input = [bonus, draft_tokens]`, runs the
  target verifier once, walks target argmax vs draft, then calls rollback on
  rejection.
- `models/deepseek_v4/language.py` implements DeepSeek-V4
  `speculative_verify_hidden/logits` by calling the same model forward and
  snapshots/restores cache state when needed.
- `models/deepseek_v4/language.py` also has batched HISA/indexer behavior for
  `L > 1`, which supports the shared-prefix attention/cache/indexer direction.

Implementation hygiene for Plan C profiling and row descriptors:

- Profiler paths must be read-only: no cache mutation, no extra synchronization,
  and no changed command-buffer ordering in normal runs.
- Do not add host readback inside the verifier layer loop for production paths.
  Measurement-only readback is allowed only behind explicit diagnostic envs.
- Descriptor logs must be rich enough to design the kernel: layer, active rows,
  `raw_common`, `raw_tail[]`, `comp_common`, `comp_tail[]`, reuse estimates, and
  sameness flags.

## Closed Branches

Keep these as diagnostics only; do not promote them as strict verifier fixes:

- batch-QKV as a strict verifier
- batched router into row-routed MoE
- batched shared expert
- confidence scheduler
- margin-only fallback
- exact-prefix-layer fallback
- DSpark main-KV batch disable
- unordered direct routed-down `sum6` as default
- plain shared-row mixed online attention as a strict verifier
- slotwise routed path as speed lever
- precomputing indexer rows when it is slower

## Measurement Reset

Do this before drawing new speed conclusions:

- Use clean no-stats paired runs for headline t/s.
- Use `DS4_DSPARK_PERF=1` for lightweight non-fenced aggregate DSpark timing.
- Use `DS4_DSPARK_BLOCK_TIMING=1` only when per-block timing lines are worth the
  extra logging noise. `DS4_DSPARK_TIMING=1` remains an older alias.
- Pass the paired baseline through `DS4_DSPARK_BASELINE_TPS=<tps>` or
  `DS4_DSPARK_BASELINE_DECODE_MS=<ms>` so the summary reports
  decode-equivalents.
- Stop using the fenced `~146 ms/block` profile as normal verifier wall time.
  Keep fenced stage profiles for ranking attention/cache and MoE buckets only.

Example:

```bash
DS4_DSPARK_PERF=1 DS4_DSPARK_BASELINE_TPS=35.34 \
./ds4 -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --temp 0 --nothink -n 1000 \
  -p "Make a game of Space Invader in Pygame" \
  --resident -c 4096
```

Expected extra lines:

```text
ds4: dspark perf: ... (draft=... verify=... overhead=... tau=...)
ds4: dspark decode-eq: draft=... verify=... overhead=... block=... baseline_decode=... ms tau=...
```

## External Reference: mlx-vlm Reframe

`Blaizzy/mlx-vlm` was inspected locally under `/tmp/ds4_mlx_vlm_ref`. The
important correction is that its explicit README speedups, `3.94x` on 26B-A4B
and `2.29x` on 31B at batch 4, are Gemma 4 MTP measurements, not DeepSeek-V4
Flash numbers. Still, the architecture is directly relevant:

- `mlx_vlm/speculative/mtp.py` verifies by concatenating bonus + draft tokens and
  running one target forward over the block, then walking target argmax vs draft.
- `mlx_vlm/models/deepseek_v4/language.py` has a DeepSeek-V4
  `speculative_verify_logits` / `speculative_verify_hidden` path that calls the
  same model forward used by decode, snapshots caches when needed, and rolls back
  rejected suffixes.
- `mlx_vlm/speculative/drafters/deepseek_v4_mtp/deepseek_v4_mtp.py` contains a
  DeepSeek-V4 MTP drafter and marks batched acceptance as uniform. It is not
  DSpark, but it is a useful target-verify reference.
- `mlx_vlm/models/deepseek_v4/hisa_kernel.py` is a batched HISA/indexer
  reference for L>=1, and `hyper_connection.py` has a fused HC sinkhorn/collapse
  Metal kernel.

The reframe: a peer MLX/Metal stack avoids the 827-dispatch strict-verifier
shape by using one batched target forward family. In that design, N=1 decode and
N>1 verify use the same kernel family, so greedy consistency is a property of
the chosen runtime contract rather than a cross-family bit-matching chase. That
does not prove DS4 can claim the Gemma speedups, but it gives us a concrete
experiment that can break the current impasse.

Important caveat: same code family does not make identity automatic. Shape-
dependent reduction order, masks, cache layouts, and quant paths can still
diverge. MLX-VLM is evidence that the contract can be made to work for Gemma 4
MTP, not proof that DS4/DeepSeek-V4-Flash gets it for free.

## Unified Target Forward Experiment

Add this as a branch of the current shared-context plan:

1. Build a DS4 target-forward rows API that accepts `N=1..5` and is intended to
   be used by both no-draft decode and DSpark verifier:

   ```c
   metal_graph_target_forward_rows_n5(
       graph, model, weights, tokens, n_rows, start_pos, mode);
   ```

2. First mode: `strict_exact`. It must reproduce the current no-draft decode
   family at N=1 and remains gated by `cmp=0` to the old baseline.
3. Second mode: `unified_greedy`. It deliberately makes the batched target
   forward canonical. No-draft decode in that mode uses the same rows API at
   N=1, and DSpark verification uses it at N<=5. The correctness gate becomes
   "DSpark output matches no-draft output in the same unified mode" plus
   deterministic repeatability, state self-consistency, and acceptance/tau
   stability.
   Current code status: `--draft-mode unified` /
   `DS4_DSPARK_VERIFY_CANONICAL=unified` is wired as a diagnostic label and the
   CLI sets `DS4_TARGET_FORWARD_UNIFIED=1`. The target-forward rows API exists
   for N=1 and wraps the current decode path; DSpark N<=5 now calls the same API
   and delegates through a selectable backend. The current default is
   `DS4_TARGET_FORWARD_UNIFIED_BACKEND=batch`, which calls the existing batch
   verifier. `DS4_TARGET_FORWARD_UNIFIED_BACKEND=strict_v1` routes N<=5 through
   the byte-clean strict verifier from the same unified wrapper, giving Plan C a
   safe comparison backend before shared-prefix rows land.
   `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` is now the named
   insertion point for the planned shared-prefix attention/cache/indexer rows
   backend and currently uses the strict-v1 core plus shared-prefix profiling so
   explicit shared-prefix experiments stay byte-clean. Set
   `DS4_TARGET_FORWARD_UNIFIED_SHARED_PREFIX_BATCH_FALLBACK=1` only to reproduce
   the older diagnostic batch fallback. `DS4_TARGET_FORWARD_UNIFIED_STRICT_BACKEND=1`
   makes the unified wrapper reject the pending backend instead of using the
   strict-v1 core; the higher-level speculative loop may still fall back to an
   older strict path.
   Use
   `DS4_TARGET_FORWARD_UNIFIED_PROFILE=1` for per-call wrapper timing.
   Shared-prefix rows are still pending, so this must not be promoted yet.
   Smoke evidence: `DS4_TARGET_FORWARD_UNIFIED=1` no-draft n=4 matched the old
   no-draft output byte-for-byte on the real Flash resident target, and
   `--draft-mode unified --draft-verify 4` ran a DSpark diagnostic n=16 smoke
   with both the unified-greedy verifier log and
   `target-forward unified N<=5 backend active: batch verifier n=4`. The newer
   backend/profile smokes add: `DS4_TARGET_FORWARD_UNIFIED_PROFILE=1` prints
   per-call N=1 decode timings while preserving no-draft `cmp=0`;
   explicit `DS4_TARGET_FORWARD_UNIFIED_BACKEND=batch` logs
   `target-forward unified N<=5 backend active: batch n=4`; and
   `DS4_TARGET_FORWARD_UNIFIED_BACKEND=strict_v1` routes through the strict
   byte-clean verifier from the unified wrapper, logs `strict_v1 n=4`, and
   matched default strict DSpark output on the short prompt (`cmp=0`);
   budget-2 batch/unified routing now skips the legacy decode2 special path, and
   `bench-results/dspark_phase0_planc_backend_smoke` confirmed
   `strict_v1 n=2`, `summary.tsv` generation, and `cmp=0` against default strict;
   `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` currently logs an active
   shared-prefix backend label backed by strict-v1 core plus profiling, unless
   the explicit batch fallback diagnostic env is set.
4. The first implementation target is attention/cache/indexer, not MoE:
   shared-prefix attention for the committed context plus exact new-block
   triangle for Plan A; batched causal forward for Plan B / unified mode.
5. ROWS5 shared-weight dense kernels follow for comp/index/projections.
6. Grouped routed-MoE remains below Plan C attention and ROWS5. The existing
   grouped IQ2 gate/up experiment was byte-clean but slower, and the MLX
   reference does not point to MoE grouping as the main win.

Execution merge from delayed Pro feedback: MLX-VLM reproduction is the external
validation lane, not a reason to pause DS4 implementation. If a separate agent
can run it, use the Gemma 4 MTP result to confirm the target-forward/cache
contract. The main DS4 patch stream should continue by making the existing
`shared_prefix` unified backend real, then testing unified N=1 and N<=5 through
the same path.

Do not begin Plan C with a giant verifier-only kernel. Begin with the
`target_forward_rows_unified(N=1..5)` contract, then move one subsystem at a
time into that shared family. The first real `shared_prefix` implementation must
serve both roles: ordinary no-draft decode at N=1 and DSpark verification at
N<=5. The point is to make decode and verify converge onto the same
target-forward realization, not to keep making a verifier-only family chase a
different decode family.

Patch E0 landed as a small strict attention cleanup before the full
shared-prefix kernel: raw single-row flash attention now reuses a persistent zero
F16 mask instead of allocating and CPU-zeroing a transient mask each row, and
gathered single-row flash attention skips the zero-fill mask dispatch when
`use_mask == 0`. This removes one tiny row-attention work item in the strict
DSpark rows-exact path without changing math. Evidence: `make -j8 ds4` passed;
`bench-results/dspark_zero_mask_smoke_191348` was `cmp=0`, baseline
`33.49 t/s`, DSpark b4 `36.91 t/s`, verifier `62.48 ms`;
`bench-results/dspark_zero_mask_n160_191514` was `cmp=0`, baseline `33.42 t/s`,
DSpark b4 `38.63 t/s`, draft `14.56 ms`, verifier `62.71 ms`, tau `4.21`,
acceptance `83.0%`; the same directory's DSpark b5 default-shape run was also
`cmp=0`, `38.46 t/s`, draft `17.48 ms`, verifier `73.22 ms`, tau `4.71`,
acceptance `76.4%`; the new no-draft n=160 output matched the older saved
`baseline_after_default_patch_n160.out` byte-for-byte. This is not the
`dspark_attn_shared_prefix_exact_n5` kernel; it is a low-risk cleanup on the way
there.

Patch E1 landed on top of E0: raw/gathered single-row flash attention now uses
the Metal no-mask function-constant variant when the attention mask is
semantically all zero. Disable with `DS4_FLASH_ATTN_DECODE_NO_MASK_DISABLE=1` or
`DS4_DSPARK_ATTENTION_NO_MASK_DISABLE=1`. This removes all-zero mask reads and
the `fma(score, scale, 0)` specialization from strict rows, while retaining a
fallback for A/B. Evidence: `make -j8 ds4` passed;
`bench-results/dspark_nomask_smoke_192357` was `cmp=0`, baseline `31.35 t/s`,
DSpark b4 `37.33 t/s`, verifier `61.41 ms`;
`bench-results/dspark_nomask_n160_192514` was `cmp=0` for b4 and b5, baseline
`33.47 t/s`, b4 `38.62 t/s`, verifier `62.79 ms`, b5 `38.54 t/s`, verifier
`72.98 ms`; default-shape `bench-results/dspark_nomask_n1000_192727` was
`cmp=0`, baseline `32.96 t/s`, DSpark b5 `38.49 t/s`, verifier `75.91 ms`, tau
`4.85`, acceptance `77.1%`.

Post-E1 active sweep:
`bench-results/dspark_nomask_phase0_n1000_193227` ran budgets 2/3/4/5 with
dispatch, route-overlap, block-timing, and shared-prefix profiles enabled; all
outputs were `cmp=0`. Diagnostics add overhead, so use it for shape/profile
columns, not headline speed. Diagnostic results: baseline `33.12 t/s`; b2
`29.17 t/s`, verifier `51.68 ms`, tau `2.70`; b3 `32.22 t/s`, verifier
`60.34 ms`, tau `3.36`; b4 `34.48 t/s`, verifier `75.51 ms`, tau `4.22`; b5
`35.35 t/s`, verifier `87.27 ms`, tau `4.85`. Shared-prefix opportunity grows
with active size: b5 raw reuse estimate `3.71x`, compressed reuse `3.86x`, with
routed overlap reuse `1.68x`. Clean no-extra-profile compare:
`bench-results/dspark_nomask_clean_n1000_compare` was b4 `cmp=0`, baseline
`32.96 t/s`, b4 `38.07 t/s`, verifier `63.81 ms`, tau `4.22`; paired with the
existing clean b5 run above, b5 is current clean n=1000 winner at `38.49 t/s`.

Patch E2 adds a sharper default-path shape diagnostic:
`DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE=1`
(`DS4_DSPARK_VERIFY_ATTN_ROWS_PROFILE=1` alias). It records the row-count
metadata at the graph layer, so it observes the default strict verifier without
forcing the optional rows-exact-head helper. Smoke
`bench-results/dspark_attn_rows_shape_194956` was byte-clean (`cmp=0`), with
baseline `33.61 t/s`, DSpark `36.89 t/s`, draft `17.82 ms`, verifier
`70.89 ms`, tau `4.44`, acceptance `68.9%`. The first-block row-shape line was:

```text
ds4: dspark attn rows shape block=1 calls=43 indexed=0 rows=215 max_n=5 raw_only=21 mixed=22 raw_same_count=0 raw_same_start=43 comp_same_count=21 raw_keys row=4995 shared_est=1343 reuse=3.72x comp_keys row=600 shared_est=156 reuse=3.85x zero_comp_rows=105 pad_rows=214 ring_rows=0 raw_minmax=21/30 comp_minmax=0/7
```

Decision: do not spend time on a trivial same-shape N-row attention helper.
In the default active-5 strict path, all layers shared the same raw start in this
early-context block, but no layer had identical raw counts across the five rows.
The next useful kernel is still `dspark_attn_shared_prefix_exact_n5`: batch the
common committed prefix, keep the new-block triangle exact, and merge
deterministically.

The Phase-0 sweep can now collect this automatically. Use:

```bash
N=32 BUDGETS=5 ROUTE_OVERLAP=0 DISPATCH_PROFILE=0 BLOCK_TIMING=0 \
SHARED_PREFIX_PROFILE=1 ATTN_ROWS_SHAPE_PROFILE=1 \
scripts/dspark_phase0_sweep.sh
```

Smoke `bench-results/dspark_phase0_attn_rows_shape_smoke_195733` passed
`cmp=0`. The `summary.tsv` row captured both the shared-prefix columns
(`sp_raw_reuse=3.71`, `sp_comp_reuse=3.86`) and the new attention row-shape
columns (`ar_raw_same_count=0`, `ar_raw_same_start=43`, `ar_pad_rows=214`).
Use these `ar_*` columns to reject same-shape shortcuts before writing kernels.

Active-size sweep `bench-results/dspark_phase0_attn_rows_shape_sweep_200046`
ran b2/b3/b4/b5 and all outputs matched the no-draft baseline (`cmp=0`). In
that profiled n=64 sweep, b4 was fastest (`37.72 t/s`, verifier `61.83 ms`,
tau `4.00`), while b5 had the largest shared-prefix target
(`ar_raw_reuse=3.72`, `ar_comp_reuse=3.85`, `ar_raw_same_count=0`). Use b4 for
quick speed smoke when needed, but keep b5 as the main DSpark-5 kernel target.

The row-shape code now builds a reusable `ds4_dspark_attn_row_shape` descriptor
inside `ds4.c`: `common_raw`, `raw_tail[5]`, `common_comp`, `comp_tail[5]`,
sameness flags, and reuse counters. The profiler uses it now; the
`dspark_attn_shared_prefix_exact_n5` kernel should consume the same descriptor.
Set `DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE_VERBOSE=1` for per-layer descriptors.
Smoke `bench-results/dspark_attn_rows_shape_verbose_200432` passed `cmp=0` and
showed layer 42 as
`ratio=128 raw_common=21 raw_reuse=3.61x raw_tail=[0,1,2,3,4]
comp_common=5 comp_reuse=4.29x comp_tail=[0,0,0,1,1]`.
Post-refactor smoke `bench-results/dspark_attn_shape_descriptor_smoke_200723`
also passed `cmp=0` and preserved the aggregate `ar_*` values.

## Agent Split

Use these as separate handoff tasks when parallel work is desired:

- Agent 1, MLX reproduction: reproduce Gemma 4 MTP no-draft vs spec on local
  Apple Silicon, confirm byte-identical greedy output, speedup, block size,
  acceptance/tau, and whether no-draft and spec call the same model path. Then
  inspect DeepSeek-V4/HISA batching. Start with the README-backed Gemma 4 MTP
  command before attempting a DeepSeek-V4 checkpoint:

  ```bash
  mlx_vlm.generate \
    --model mlx-community/gemma-4-31B-it-bf16 \
    --draft-model mlx-community/gemma-4-31B-it-assistant-bf16 \
    --draft-kind mtp \
    --draft-block-size 4 \
    --prompt "Explain speculative decoding in 3 sentences." \
    --max-tokens 256 \
    --temp 0
  ```

  Record no-draft/spec output byte equality, no-draft tok/s, spec tok/s,
  acceptance/accepted tokens per block, and target verify time if exposed.
- Agent 2, DS4 unified-forward prototype: add `DS4_TARGET_FORWARD_UNIFIED=1`
  and a `metal_graph_target_forward_rows_unified(N=1..5)` family. Start by
  wrapping current decode for N=1, then move attention/compressor/indexer,
  FFN/router, routed/shared, and output/head into the shared path one subsystem
  at a time. First milestone: unified N=1 matches old no-draft at n=160/n=1000
  if possible. Second milestone: unified DSpark spec matches unified no-draft at
  temp 0.
- Agent 3, shared-prefix attention: build `dspark_attn_shared_prefix_exact_n5`
  as the Plan C attention vehicle. Preserve row-local KV visibility,
  row-local compressor/indexer mutation, per-row mask/frontier snapshots, and
  exact online-softmax key traversal order. Batch only the N queries over the
  committed prefix, shared KV/compressed/index streams, and row-independent
  projections. Goal: N=1 no-draft works through the same family, then N=5
  verifier uses that same family.
- Agent 4, DeepSpec reference: extract DSpark draft architecture, acceptance
  evaluation logic, confidence/scheduler assumptions, and target-cache interface
  expectations from DeepSpec. This is separate from verifier kernel work.

## Near-Term Patch List

### Patch A: Freeze And Name Current Default

Add or confirm a visible strict-v1 verifier name and log line:

```text
ds4: dspark verifier strict_v1 active=5 prefixN=1
```

No behavior change.

Required gates:

- n=160 `cmp=0`
- n=1000 `cmp=0`
- n=4000 `cmp=0`
- block-1 audit clean
- block-501 audit clean

Status:

- Default DSpark verify budget is now 5.
- Loader reports `strict_v1 verifier`.
- First strict verifier block reports `ds4: dspark verifier strict_v1 active=5
  prefixN=1`.
- Short smoke `dspark_strictv1_default_active5_n64_150213` passed `cmp=0`.
- Longer n=160/n=1000/n=4000 gates remain the promotion checks for future
  strict-v2 changes.

### Patch B: Non-Fenced Timing And Dispatch Census

The blocking profiler tells where time goes but distorts headline speed. Add or
use non-fenced timing and counters, aggregated per verifier block or per run:

- draft ms
- verify ms
- overhead ms
- tau
- decode-equivalent ratios against paired baseline decode ms

- attention pre-dispatches
- compressor projection dispatches
- indexer projection dispatches
- frontier mutation dispatches
- attention head dispatches
- attention output dispatches
- router dispatches
- routed-MoE dispatches
- shared dispatches
- tail dispatches
- total dispatches

Use this to separate actual math cost from command/object churn.

Status:

- `DS4_DSPARK_PERF=1` enables lightweight aggregate DSpark timing without
  backend VM stats.
- `DS4_DSPARK_BASELINE_TPS=<tps>` or
  `DS4_DSPARK_BASELINE_DECODE_MS=<ms>` enables decode-equivalent reporting.
- Smoke `dspark_perf_footer_smoke_151213` printed:
  `draft=18.33 ms`, `verify=72.28 ms`, `overhead=1.98 ms`, `tau=4.00`,
  and decode-equivalents `draft=0.65`, `verify=2.55`, `overhead=0.07`,
  `block=3.27` against `baseline_decode=28.30 ms`.

### Patch C: Draft Profiling

Draft is only about `18-19 ms/block`, but that is now large enough that hiding
or halving it matters. Profile:

- Markov head
- three draft layers
- routed MoE inside the draft
- command/launch overhead

Current hooks:

- `DS4_DSPARK_DRAFT_PROFILE=1`: graph, Markov, and total draft timing.
- `DS4_DSPARK_GRAPH_PROFILE=1`: draft block0/block1/block2/head timing.
- `DS4_DSPARK_BLOCK_PROFILE=1`: per-draft-layer stages, including
  `shared_router` and `routed`.

These profiles fence by design and are for splitting the draft bucket, not for
headline generation t/s. Use `DS4_DSPARK_PERF=1` for the non-fenced aggregate
draft cost first, then enable these only to explain that cost.

Smoke `dspark_draft_profile_smoke_151447` confirmed the hooks. Ignoring the
first warmup-heavy block, typical draft blocks were:

- block0: `4.6-5.5 ms`
- block1: `4.3-5.6 ms`
- block2: `4.4-5.6 ms`
- output head: `1.4-1.6 ms`
- Markov chain: `1.7-2.1 ms`
- total draft: `16.7-20.0 ms`

### Patch D: Phase-0 Measurement Before More Kernels

Do not start the next kernel from fenced stage-profile totals. First run the
strict active-size sweep with non-fenced block timing, dispatch census, route
reuse, and paired decode-equivalent reporting.

Use:

```bash
DS4_DSPARK_ROUTE_OVERLAP_LOG=1 \
DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1 \
DS4_DSPARK_BLOCK_TIMING=1 DS4_DSPARK_PERF=1 \
DS4_DSPARK_BASELINE_TPS=<paired-no-draft-tps> \
./ds4 -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --draft-verify 5 --temp 0 --nothink -n 160 \
  -p "Make a game of Space Invader in Pygame" \
  --resident -c 4096
```

Or use the checked-in sweep helper:

```bash
N=160 scripts/dspark_phase0_sweep.sh
```

Add Plan C backend rows to that same diagnostic table with:

```bash
N=160 PLAN_C_BACKENDS="batch strict_v1" scripts/dspark_phase0_sweep.sh
```

The parser reports `unified_backend`, `unified_active_n`,
`tfwd_decode_avg_ms`, `tfwd_verify_avg_ms/min/max`, failed wrapper calls, and
per-N `tfwd_n{1..5}_avg_ms` columns for those optional backend rows.

Perf accounting note: the first Plan C backend sweep under
`bench-results/dspark_phase0_planc_n64_backend_20260628_173006` undercounted
batch/unified blocks because several successful return paths did not call
`ds4_session_dspark_perf_record()`. Use only post-fix sweeps for
`block_ms`/`tau` conclusions. The post-fix smoke
`bench-results/dspark_phase0_planc_backend_smoke_after_perf` confirmed default
strict b2 and unified strict-v1 b2 both report `blocks=3`, matching the
scheduled-block footer, and their outputs compare `cmp=0`.

Post-fix N=64 Plan C comparison:

| Row | Generation | Verify ms | Tau | Blocks | Compare |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 33.41 t/s | - | - | - | reference |
| default strict b4 | 35.04 t/s | 66.62 | 4.00 | 16 | reference strict |
| default strict b5 | 34.76 t/s | 72.14 | 4.27 | 15 | reference strict |
| unified strict-v1 b4 | 36.10 t/s | 62.99 | 4.00 | 16 | `cmp=0` vs default strict b4 |
| unified strict-v1 b5 | 35.95 t/s | 68.08 | 4.27 | 15 | `cmp=0` vs default strict b5 |

Source directory:
`bench-results/dspark_phase0_planc_n64_strictv1_after_perf`.

Run fixed active sizes `--draft-verify 1`, `2`, `3`, `4`, and `5`, collecting:

- draft ms/block
- verify ms/block
- commit/overhead ms/block
- total block ms
- `tau`
- dispatches/block
- attention dispatches
- compressor/indexer dispatches
- routed-MoE dispatches
- route slots/layer
- unique experts/layer
- duplicate expert slots/layer
- reuse factor = route slots / unique experts
- layers with reuse >= `1.25x` and `1.50x`

Interpretation:

- unique `28-30 / 30`: de-dup is weak; maybe only dispatch cleanup remains.
- unique `22-26 / 30`: useful; grouped exact MoE may be a modest win.
- unique `<=20 / 30`: strong; grouped exact MoE becomes a major verifier target.
- If GPU active-vs-idle counters are available, capture one verify block. The
  current hypothesis predicts latency-bound/idling on serial micro-kernels, not
  saturated arithmetic.

Status:

- `DS4_DSPARK_ROUTE_OVERLAP_LOG=1` is the opt-in route-overlap diagnostic. It
  uses host readback and fences, so it is a measurement tool, not a performance
  path.
- Smoke `dspark_route_overlap_smoke_152402` showed strong reuse on
  `"Make a game of Space Invader in Pygame"`:

  | Block | Active | Slots | Unique | Reuse | Duplicate Slots | Layers >=1.25x | Layers >=1.50x |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 5 | 1290 | 700 | 1.84x | 45.7% | 40/43 | 37/43 |
  | 2 | 5 | 1290 | 779 | 1.66x | 39.6% | 39/43 | 33/43 |
  | 3 | 4 | 1032 | 622 | 1.66x | 39.7% | 39/43 | 36/43 |

- This is not uniform-random routing (`~1.06x` expected reuse for 30 draws over
  256 experts). It means local code-generation contexts cluster enough that
  grouped exact MoE remains worth building, but the dispatch census makes
  shared-context attention/cache/indexer batching the higher-priority strict
  speed lever.
- Fresh active-N diagnostic sweep `bench-results/dspark_phase0_n64_162431` used
  `scripts/dspark_phase0_sweep.sh` with baseline 35.77 t/s. Active-5 reported
  tau `4.27`, draft `17.75 ms`, verify `77.49 ms`, verify decode-equivalent
  `2.77`, dispatch total `936`, comp/index/heads `205/105/215`, and route reuse
  `1.61x`. Dispatch total grew by about 109 per extra verified row from active-2
  to active-5, mostly compressor/indexer/heads. That is the concrete go signal
  for Patch E before more MoE work.

### Patch E: Shared-Context Attention / Cache / Indexer Strict-V2

This is now the first speed kernel after Phase 0. The measured verifier is about
`2.45` decode-equivalents for active-5, while a shared-context 5-token causal
forward should be closer to `1.1-1.3` decode-equivalents because the long
committed KV/cache read is shared.

Design:

- Split each attention-like op into a **shared prefix** and a **new-block
  triangle**.
- Shared prefix: batch N<=5 queries against the same committed K/V or
  compressed/index stream. Read each cache tile once. Each query must still walk
  keys in the same order and use the same dot/softmax/AV accumulation order as
  decode.
- New-block triangle: keep the <=4 freshly written draft rows row-exact and
  decode-order sequential. This is where prior raw-KV / `attn-heads-raw`
  divergence came from.
- Merge shared-prefix and triangle state with the same deterministic online
  softmax combine used by the decode kernel.

Proof gates:

- single-layer bit equality before wiring into the 43-layer chain
- n=160, n=1000, and n=4000 `cmp=0`
- state audit at `1e-8` for final HC, raw KV row, attention/compressor/indexer
  frontier state, and DSpark main KV

Negative diagnostic result: the opt-in
`DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_VARLEN_UNSAFE=1` path added an N<=5 plain
mixed-heads kernel with explicit per-row `n_raw`, `raw_start`, and `n_comp`.
The n=64 b4 smoke improved proposed-token verify cost but was not byte-clean:
acceptance fell to 34.4%, generation fell to 22.07 t/s, and output diverged.
Do not promote plain online batch heads into Mode A. The next attention kernel
must either preserve the strict per-row flash-attention realization, or move
fully into the Plan C unified-greedy contract with its own N=1 baseline.

Safe diagnostic result: `DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_ROW_EXACT=1`
defers only plain heads, keeps indexed heads row-exact, and replays the deferred
rows through the existing flash-attention row encoders in one helper call.
It is byte-clean but not faster: n=64 b4 measured 35.85 t/s, verify 65.57 ms;
n=160 b4 measured 38.44 t/s, verify 63.45 ms. This rules out C-call/view
consolidation as the main prize and points back to the true shared-prefix flash
kernel: cache tile reads and flash-attention vec/reduce work must be shared
across N<=5 queries while preserving the strict row realization or defining a
new unified-greedy realization.

Sizing hook: `DS4_DSPARK_SHARED_PREFIX_PROFILE=1` (alias
`DS4_TARGET_FORWARD_SHARED_PREFIX_PROFILE=1`) now logs a first-block summary of
the row-exact raw/compressed attention key scans versus a shared-prefix estimate
using the row metadata already computed by strict-v1. This is deliberately
non-fenced and does not change verifier math. Add
`DS4_DSPARK_SHARED_PREFIX_PROFILE_ALL=1` only for context-growth curves. Use it
to decide how much of the heads/compressor/indexer row loop can be collapsed
before writing `dspark_attn_shared_prefix_exact_n5`.

Latest review-derived scope: begin with raw attention only. The first
`dspark_attn_shared_prefix_exact_n5` prototype should take N<=5 row Q, the raw
KV cache, and the row descriptor, scan the committed raw prefix once per tile
for all rows, keep one online-softmax state per row, append each row's visible
block-tail keys in exact decode order, and write the same `attn-heads-raw`
result as the row path. Do not touch routed MoE, batch router/shared experts, or
replace strict-v1 default behavior in this milestone. Pull compressor/indexer
into the same design only after raw attention passes the one-layer and block
audits.

Scaffold patch: `DS4_DSPARK_ATTN_RAW_VEC_ROWS_EXPERIMENT=1` (alias
`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS=1`) adds a raw-only N<=5
vector-rows probe underneath the existing row-exact deferred-heads mode. It
uses the same vector flash-attention kernel family as the safe row path, with a
per-row mask for tail visibility, and falls back whenever compressed rows or
non-matching raw starts appear. Treat this as a correctness/shape probe before
the true shared-prefix scan-sharing kernel.

Smoke result: `bench-results/dspark_raw_vec_rows_202053` compared row-exact
deferred-heads control against the raw vec-rows experiment. n=32 and n=160 both
matched byte-for-byte (`cmp=0`). The n=160 pair kept acceptance and tau
identical (`76.4%`, tau `4.71`) and moved verifier wall only slightly
(`72.46 ms` -> `71.96 ms`, generation `38.86` -> `39.03 t/s`). This validates
the wrapper as a safe landing point; it does not replace the shared-prefix
scan-sharing kernel.

Promotion root cause and fix: the first automatic raw vec-rows promotion inside
the named `shared_prefix` backend diverged at n=160 because the unified-mode
environment leaked batch-canonical internals into the supposed `strict_v1`
delegate. A scoped `g_ds4_target_forward_strict_v1_depth` guard now disables
those internals while the strict delegate is executing. After that fix, default
strict matched unified `strict_v1` at n=64, explicit raw vec-rows matched
unified `strict_v1` at n=64 and n=160, and the named `shared_prefix` backend
matched default strict at n=160 while using raw vec-rows
(`bench-results/dspark_sharedprefix_vs_default_scopefixed_n160_204236`,
`cmp=0`). The backend is byte-clean as a landing point; it still does not share
the raw-KV scan yet.

Rows5 fused prototype result: `kernel_flash_attn_ext_vec_rows5_f16_dk512_dv512`
behind `DS4_DSPARK_ATTN_RAW_VEC_ROWS_FUSED=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_FUSED=1`) groups N<=5 rows for one
head/split-K group into one threadgroup. It was byte-clean on the n=32 smoke
against the current `shared_prefix` raw vec-rows path
(`bench-results/dspark_raw_rows5_fused_204908`, `cmp=0`) but slower
(`70.18 ms` -> `83.78 ms` verifier). Do not promote. This negative result says
the next raw attention kernel must explicitly stage/share committed-prefix K/V
tiles; threadgroup co-location without tile staging is not the performance
lever.

Resource diagnostic: `DS4_DSPARK_ATTN_RAW_VEC_ROWS_PROFILE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_PROFILE=1`) prints the first raw
vec-rows call's active rows, `max_raw`, `raw_start`, `nsg/nwg`, fused flag,
mask/KV/tmp/shared memory, dispatch shape, and estimated K-only/KV tile-staging
memory. Use `_ALL=1` only for kernel sizing sweeps because it prints every raw
vec-rows call.

K-stage rows5 diagnostic: `DS4_DSPARK_ATTN_RAW_VEC_ROWS_KSTAGE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_KSTAGE=1`) selects
`kernel_flash_attn_ext_vec_rows5_kstage_f16_dk512_dv512` for the raw vec-rows
landing point when `nsg=1`. It stages each committed-prefix K tile once per
threadgroup and shares it across N<=5 row states. It is byte-clean but not a
speed win yet: n=32 matched strict/shared output but ran `31.09 t/s`, and n=160
matched default strict (`cmp=0`) at `38.86 t/s` versus strict `38.99 t/s` with
the same `76.4% (126/165)` acceptance. Keep it diagnostic; the next revision
needs to reduce barrier/occupancy cost or stage a larger useful prefix, not just
share the small raw K tile.

n=320 aggregate profile sharpens that conclusion:
`bench-results/dspark_raw_vec_agg_shared_n320` matched default strict (`cmp=0`)
and measured `40.75 t/s` versus strict `40.68 t/s`. It reported
`calls=380 rows=1900 avg_rows=5.00 avg_max_raw=72.21 max_raw=123
nsg_hist[1..4]=380/0/0/0`, so all raw vec-rows calls were eligible for the
`nsg=1` staged kernel. K-stage also matched at n=320, but measured `40.63 t/s`,
so staging K alone is not the current bottleneck.

K+V-stage diagnostic:
`kernel_flash_attn_ext_vec_rows5_kvstage_f16_dk512_dv512` exists only behind
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_KVSTAGE_UNSAFE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_KVSTAGE_UNSAFE=1`). The first n=32
smoke diverged (`cmp=1`) and collapsed acceptance to `31.7% (19/60)`, so do not
use it for speed comparisons until its V staging is debugged.

Varmap common-prefix K/V staging diagnostics:
`DS4_DSPARK_ATTN_VARMAP_COMMON_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_COMMON_VSTAGE=1` test the same memory-source substitution
inside the current byte-clean varmap rows5 scaffold. Both preserve row-local
online-softmax order and compare exact against rows-exact attention in local
smokes. K-stage evidence:
`bench-results/dspark_varmap_common_kstage_042048` produced 738 compare lines,
all `max=0`, and final `cmp=0`, but no-compare n=96 was `38.26 t/s`.
V-stage evidence: `bench-results/dspark_varmap_common_vstage_043114` had final
`cmp=0`, zero nonzero compare lines in compare-all n=96, and n=160 `39.30 t/s`
versus paired default `39.86 t/s`. Keep both flags diagnostic-only. This is
good exactness evidence for common-prefix memory reuse, but bad performance
evidence for simple threadgroup staging inside the existing per-row stream. The
production Plan C attention kernel still needs a common-prefix online-softmax
split plus exact row-local tail combine.

Varmap split estimator correction:
the varmap profile now prints both strict row-local prefix reuse and raw
interval-intersection reuse. `bench-results/dspark_varmap_intersection_profile_043922`
stayed `cmp=0` at n=320 and reported prefix-only `raw_split_reuse=1.22x`, but
intersection-based `raw_intersection_reuse=4.46x` with
`raw_intersection_full=190240`; compressed stayed strong at
`comp_split_reuse=4.79x`. This resolves the apparent conflict with older
shared-prefix estimates: the shared raw segment is often the intersection of
sliding raw windows, not a prefix for all rows. Do not build the next kernel as
`common prefix + tail` only. Build it as deterministic online-softmax over
private-old, shared-intersection, and private-new/tail segments, preserving each
row's key order.

Raw-union K-stage probe:
`DS4_DSPARK_ATTN_VARMAP_RAW_UNION_KSTAGE=1` is a narrower exactness proof inside
the current varmap scaffold. It stages the shifted raw-union K tile for each
row-local `ic..ic+31`, so each row still consumes the same 32-key tile in the
same online-softmax order. `bench-results/dspark_varmap_raw_union_kstage_044329`
had final `cmp=0` and zero nonzero compare lines, but clean n=160
`bench-results/dspark_varmap_raw_union_kstage_clean_044600` slowed default DSpark
from `39.65 t/s` to `38.47 t/s` with identical acceptance. Treat this as an
address/order proof, not a production Plan C kernel.
`DS4_DSPARK_ATTN_VARMAP_RAW_UNION_VSTAGE=1` adds the matching value-side probe.
`bench-results/dspark_varmap_raw_union_vstage_050432` had final `cmp=0` and zero
nonzero attention compare lines, but clean n=160
`bench-results/dspark_varmap_raw_union_vstage_clean_050547` slowed default DSpark
from `39.42 t/s` to `38.07 t/s`. Conclusion: raw K/V sharing inside the current
row-local stream is exact but not the speed path; the real Plan C kernel should
split private-old / shared-intersection / private-new phases instead of adding
more per-tile staging to the existing loop.

Raw tile-intersection K-stage probe:
`DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_KSTAGE=1` stages only the
overlapping absolute raw K rows within each shifted row-local 32-key tile and
loads private tile edges directly. The host encoder now binds the additional
tile-intersection cap argument at index 15, fixing the intermediate ABI
mismatch. `bench-results/dspark_varmap_tile_intersection_kstage_051154` had
final `cmp=0` and zero nonzero attention compare lines. Clean n=160
`bench-results/dspark_varmap_tile_intersection_kstage_clean_051307` stayed
`cmp=0`, but slowed default DSpark from `39.08 t/s` to `37.53 t/s`. Keep this
diagnostic-only. It proves the address/order contract, but confirms the fusion
shape is wrong for speed: useful Plan C attention must make the shared
intersection a first-class online-softmax phase, not a staged fragment inside
the existing row-local tile loop.

Raw tile-intersection V-stage probe:
`DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_VSTAGE=1` is the matching
value-side proof. It stayed byte-clean in current local checks:
`bench-results/dspark_raw_tile_vstage_split_084328` kept V-only n=64 `cmp=0`,
and `bench-results/dspark_raw_tile_vstage_vonly_084543` kept V-only n=160
`cmp=0` with matching acceptance. It is still not a speed win (`36.38 t/s` in
the n=160 V-only run), and combined raw tile K+V staging is guarded back to
K-only after `bench-results/dspark_raw_tile_vstage_084149` corrupted acceptance
(`1.7%`, `cmp=1`). This closes the "add more row-local tile staging" branch.
The next useful kernel must remove repeated K/V stream materialization by
making the shared raw/compressed intersection its own online-softmax phase.

Compressed-union K/V staging probe:
`DS4_DSPARK_ATTN_VARMAP_COMP_UNION_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_COMP_UNION_VSTAGE=1` repeat the same proof for
compressed compact-index rows. Two traps were found and guarded: tail tiles must
not rely on row 0 staging when row 0 has no key at that `ic`, and K+V together
exceeds the safe threadgroup-memory shape for this scaffold. The host now
downgrades simultaneous K+V requests to K-stage only. Compare-all after the
guards had no head deltas for K-only, V-only, or guarded K+V, and clean
`bench-results/dspark_varmap_comp_union_guarded160_071734` stayed `cmp=0`, but
it was slower than default: default `38.77 t/s`, K-stage `37.42`, V-stage
`37.49`, guarded-both `37.29`. Keep compressed-union staging diagnostic-only.
It closes the "stage shared compressed tiles inside the row-local stream"
branch; the real Plan C kernel still needs a separate shared compressed scan
plus deterministic online-softmax combine.

The profile footer now distinguishes total common compressed keys from full
32-key common tiles, and prints the estimated reuse if only full tiles are
shared. Fresh n=320
`bench-results/dspark_comp_fulltile_profile_072415` stayed `cmp=0` and measured
default varmap active-5 `41.28 t/s`. Its aggregate was
`comp_common=52890`, `comp_common_full=34272`, `comp_tail=2979`,
`comp_split_est=55869`, `comp_split_reuse=4.79x`, and
`comp_full_tile_calls=777/2337`. This sharpens the next kernel contract: shared
compressed scan reuse is large enough to pursue, but a full-tile-only helper
would miss a substantial common tail. The production Plan C attention phase
needs full-tile plus tail handling under the same deterministic online-softmax
combine.
Fresh validation in `bench-results/dspark_comp_tail_profile_073055` stayed
`cmp=0` and measured `41.51 t/s` at n=160. Its new fields printed
`comp_common=14649`, `comp_common_full=6048`,
`comp_common_partial=8601`, `comp_split_reuse=4.60x`, but only
`comp_full_reuse=1.47x`. Therefore a full-tile-only compressed scan is not a
good production starting point; if implemented at all, treat it as a very short
bring-up probe before adding common-tail handling.
Fresh raw/full-tail validation in
`bench-results/dspark_raw_comp_tail_profile_073546` stayed `cmp=0` and measured
`40.72 t/s` at n=96. The raw intersection fields printed
`raw_intersection=48257`, `raw_intersection_full=36736`,
`raw_intersection_partial=11521`, `raw_intersection_reuse=4.24x`, and
`raw_intersection_full_reuse=2.36x`. The compressed fields printed
`comp_common_full=0`, `comp_common_partial=6048`, and `comp_full_reuse=1.00x`.
This makes partial common tiles a first-class requirement for both raw and
compressed shared scans.
The next profiler update measures whether shared keys land in the same
row-local split lane for the current FlashAttention vec/reduce layout. Fresh
all-block validation in `bench-results/dspark_phase_alignment_all_074309`
stayed `cmp=0`, measured `39.82 t/s` at n=160, and printed 27 attention
row-shape summaries. Weighted totals were `raw_lane_aligned=96464` of
`raw_intersection=100517` (`96.0%`) and `comp_lane_aligned=13137` of
`comp_common=14019` (`93.7%`). The first shared-scan implementation should
therefore use a natural split-lane fast path and a remap fallback for the
misaligned minority. Do not design the first kernel as if all shared keys need
arbitrary remap.
Follow-up run-structure validation in
`bench-results/dspark_phase_runs_profile_074806` stayed `cmp=0` and measured
`39.68 t/s` at n=160. The same weighted alignment was distributed across
`2184` raw natural-lane runs (`avg_run=44.2`, `max_run=124`) and `908`
compressed natural-lane runs (`avg_run=14.5`, `max_run=44`). That is friendly
to compact contiguous range descriptors. The Plan C natural-lane phase should
consume ranges, not per-key scatter lists; keep scatter/remap only for the
misaligned minority.
The host descriptor builder for those ranges now exists in
`ds4_gpu_attention_decode_varmap_rows_tensor()`. It builds raw-union and
compact-compressed natural-lane ranges from the exact varmap row descriptors
and tags each range with the row-local split lane. Fresh n=96 validation
`bench-results/dspark_varmap_phase_desc_profile_075259` stayed `cmp=0` and
reported `phase_raw_keys=48257`, `phase_raw_ranges=1845`,
`phase_raw_avg_run=26.2`, `phase_raw_max_run=32`,
`phase_comp_keys=5397`, `phase_comp_ranges=504`,
`phase_comp_avg_run=10.7`, `phase_comp_max_run=23`, and no descriptor
overflow. The descriptor ABI is now exercised on GPU as well:
`DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE=1` uploads the compact ranges to a
transient Metal buffer and dispatches `kernel_dspark_phase_range_probe`, which
recomputes raw/comp key and range counts, lane mask, max run, and checksum. The
host compares those counters against the CPU builder. Default mode probes the
first varmap call; `DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE_ALL=1` checks every
varmap call. Fresh validation:
`bench-results/dspark_varmap_phase_probe_080002` stayed `cmp=0` at n=64, and
`bench-results/dspark_varmap_phase_probe_all_080115` stayed `cmp=0` at n=16
while checking 123 varmap calls with no mismatch. Next kernel step: replace the
probe body with a compare-only shared-scan phase that writes FlashAttention
split state for these natural-lane ranges, leaving current varmap output
authoritative until the phase reducer is proven exact.
`DS4_DSPARK_ATTN_VARMAP_PHASE_KV_PROBE=1` extends that proof from descriptor
counts to actual consumed-format K/V addressing: it reads the descriptor-covered
F16 `raw-union | compressed` stream and validates expected half4 read counts and
checksum shape without changing authoritative output. Fresh validation:
`bench-results/dspark_varmap_phase_kv_probe_080742` stayed `cmp=0` at n=64 and
logged `kv_reads=3328`; `bench-results/dspark_varmap_phase_kv_probe_all_080914`
stayed `cmp=0` at n=16 while checking 123 varmap calls with no mismatch. This is
still not `dspark_attn_shared_intersection_mixed_exact_n5`; it only proves the
descriptor/KV address contract needed by that kernel.
`DS4_DSPARK_ATTN_VARMAP_PHASE_SOFTMAX_PROBE=1` adds the first compare-only math
step: it computes per-row/per-head/per-split-lane `S/M` online-softmax state for
the descriptor-covered shared raw/compressed ranges, without writing candidate
`so4`, heads, or authoritative output. Fresh validation:
`bench-results/dspark_varmap_phase_softmax_probe_081521` stayed `cmp=0` at n=64
and logged `slots=320 active=320 lanes=1`; the all-call smoke
`bench-results/dspark_varmap_phase_softmax_probe_all_081634` stayed `cmp=0` at
n=16 while checking 123 varmap calls.
`DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_PROBE=1` now writes candidate shared-range
`so4` plus `S/M` into a scratch split-state buffer and reduces through the
existing FlashAttention reducer into candidate heads, with current varmap heads
still authoritative. Fresh validation:
`bench-results/dspark_varmap_phase_head_probe_082347` stayed `cmp=0` at n=64 and
logged `heads=320 floats=163840 nonzero=163840 rms=0.349892`; the all-call smoke
`bench-results/dspark_varmap_phase_head_probe_all_082459` stayed `cmp=0` at n=16
while exercising every varmap call. This candidate is shared-range-only, so a
strict-head delta compare is not meaningful yet. Next exact step: add the
private-old, private-new, and block-tail phases, then compare candidate heads
against strict varmap heads before any promotion.
`DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_SHARED_PROBE=1` is a rows-together variant of
that scaffold. It dispatches one head/split-lane threadgroup with N<=5 verifier
rows as simdgroups; row 0 stages each descriptor K/V chunk once and all rows
consume it. Fresh smoke `bench-results/dspark_phase_head_shared_probe_085449`
stayed `cmp=0` and preserved acceptance, but it is slower as an add-on probe:
default n=64 `36.69 t/s`, old per-row head probe `36.66 t/s`, shared head probe
`33.53 t/s`. This is useful only as implementation scaffolding. It does not
remove the hot copy/stage path and still lacks private-old/private-new/block-tail
coverage.

Main-track correction from the 2026-06-29 feedback:
the project is still speed-limited by attention/compressor/indexer row-scaled
verifier work, not draft quality, prefix commit, ordered MoE, host row views,
raw-only attention, same-shape rows5 attention, direct resident reads,
shadow-to-scratch blits, raw-union masks, simple K/V staging, or more MoE
fusion. The current stage refresh
`bench-results/dspark_varmap_stage_refresh_085143` confirms the attention
copy/materialization bucket dominates the varmap helper on this binary:
`copy=1402.552 ms`, `vec=274.759 ms`, `reduce=187.447 ms` over 738 calls.
Therefore the next kernel must be the real mixed/compressed shared-prefix
candidate, not another row-local staging variant.

Next implementation target:
`dspark_attn_shared_prefix_mixed_exact_n5` (same target as the earlier
`dspark_attn_shared_intersection_mixed_exact_n5` wording). It should:

- build full row descriptors for raw private-old, raw shared-intersection,
  raw private-new/block-tail, compressed private-old, compressed shared-common,
  and compressed private-new/tail;
- keep one independent F32 online-softmax/AV state per verifier row;
- process private segments in exact row key order;
- process shared raw/compressed K/V once per tile/range for all active rows;
- write candidate heads to scratch and restore strict heads until
  candidate-vs-strict attention-head `max=0`;
- gate the public speed path on block audits plus n=160/n=1000/n=4000 `cmp=0`.

Latest mixed-prefix descriptor/compare checkpoint:
`DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_DESC_PROBE=1` now exercises the full
mixed descriptor builder. Smoke `bench-results/dspark_mixed_desc_probe_090253`
stayed `cmp=0` against paired default and showed the desired reuse shape on the
first active-5 block: `raw_row=115 raw_desc=31 raw_intersection=21
raw_shared=21 raw_new=10`, plus `comp_row=27 comp_desc=7 comp_common=5
comp_shared=5 comp_tail=2`. `DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_COMPARE=1`
adds a compare-only GPU candidate using `kernel_dspark_mixed_prefix_head_probe`;
strict varmap heads remain authoritative. `bench-results/dspark_mixed_prefix_compare_real_091336`
proved the GPU reads the descriptor buffer correctly (`raw=31/5 comp=7/3
shared=2 private=6 rows=0x1e lanes=0x1`) and final output still matches default
(`cmp=0`). The candidate itself is a correctness reject: attention-head deltas
are large (`max` roughly `2.7..7.6`). Root cause: the candidate splits raw and
compressed phases, but strict varmap processes each row-local 32-key
FlashAttention chunk as one online-softmax update; early rows put raw and
compressed keys in the same strict chunk. The next candidate must be
chunk-boundary aware: preserve each row's strict 32-key chunk realization while
sharing K/V loads inside that chunk. Do not promote or optimize the phase-split
mixed-prefix candidate.

Mixed chunk-boundary staging checkpoint:
`DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_VSTAGE=1` are now wired as guarded
diagnostics in the strict varmap kernel. They stage the exact row-local
raw/compressed 32-key chunk only when all active verifier rows execute that
chunk; without this all-row guard, K-stage diverged at n=96 in
`bench-results/dspark_mixed_chunk_repeat_092250`. After the guard,
`bench-results/dspark_mixed_chunk_guard_092621` had K-only `cmp=0` and V-only
`cmp=0` at n=96, while combined K+V remained invalid and is guarded back to
K-only. The n=160 K-only smoke
`bench-results/dspark_mixed_chunk_kstage_n160_092915` stayed `cmp=0`, but did
not improve speed: default `36.96 t/s`, verifier `76.92 ms`; K-stage
`36.79 t/s`, verifier `79.89 ms`. Treat this as a correctness/diagnostic
checkpoint, not a production optimization. The next production candidate still
needs to remove repeated consumed-format K/V materialization with a
mixed/shared-prefix attention kernel, not just stage row-local chunks.

Stop spending main-track time on raw-only probes, same-shape probes,
direct-resident probes, shadow blits, raw-union mask variants, plain mixed
unsafe variants, and grouped Q2 MoE attempts unless they directly feed that
mixed/compressed candidate.

Speed-first check before more exactness work:
the old plain mixed-shared attention candidate is not enough of a win to justify
making it the main exactness target. `bench-results/dspark_mixed_speed_ceiling_083126`
measured n=320 no-draft `31.57 t/s`, strict/default varmap DSpark `39.01 t/s`,
unsafe plain mixed-shared `40.15 t/s`, and unsafe heads8 mixed-shared
`37.88 t/s`; both unsafe outputs differed from strict (`cmp=1`).
`bench-results/dspark_mixed_speed_ceiling_n1000_083357` measured n=1000
strict/default varmap `36.01 t/s` versus unsafe plain mixed-shared `37.87 t/s`
(`cmp=1`). Treat that as only a `~3-5%` ceiling on this prompt. The priority is
therefore not exactifying `kernel_dsv4_plain_mixed_attention_*`; it is a
vec/reduce-preserving `dspark_attn_shared_intersection_mixed_exact_n5` kernel
that keeps the strict varmap chunk order, online-softmax combine, and reducer
layout while sharing K/V loads across N<=5 verifier rows.
Use a speed gate before more exactness work: a new attention candidate should
show at least an `8%` n=1000 speed ceiling and no tau/acceptance collapse before
we spend time proving bit identity. The current plain mixed-shared family and
row-local tile staging probes do not pass that gate.

MLX reference map for Plan C:
use `/tmp/dspark-research/mlx-vlm` as the live correspondence point, not just as
general inspiration. `mlx_vlm/speculative/mtp.py::_mtp_verify_target()` runs the
target verify block; `_mtp_rounds()` and `_mtp_rounds_batch()` build
`[bonus, draft_tokens]`, walk target-vs-draft tokens, and call rollback after
rejection. In `mlx_vlm/models/deepseek_v4/language.py`,
`_speculative_verify()` calls the DeepSeek V4 model forward for the verify block,
and `rollback_speculative_cache()` snapshot/restores/replays or trims/zeros the
rejected cache tail. Plan C should preserve that one-forward plus owned
cache-commit contract; strict-v1 remains the DS4 byte-clean row-decode baseline.

Dynamic-NWG diagnostic:
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_DYNAMIC_NWG=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_DYNAMIC_NWG=1`) reduces split-K
work for small raw spans and skips the reduce pass at `nwg=1`. It is byte-clean
at n=32 and n=320, but not faster: n=320 cut tmp estimate from `7629.69 MiB` to
`627.44 MiB`, while speed fell from raw vec-rows `40.75 t/s` to `39.98 t/s`
(strict `40.68 t/s`). Treat this as evidence that raw-only tmp/reduce traffic is
not the primary limiter; the next Plan C speed work should move to the larger
compressed/shared-prefix path or a fused attention pipeline, not another raw-only
split-count tweak.

Rows-exact mixed/compressed profile:
`DS4_DSPARK_ATTN_ROWS_EXACT_PROFILE=1`
(`DS4_TARGET_FORWARD_ROWS_EXACT_PROFILE=1`) summarizes the remaining rows-exact
attention path. n=320 shared-prefix profile
`bench-results/dspark_rows_exact_profile_n320` was byte-clean (`cmp=0`) and
reported `calls=1957 rows=9785 avg_rows=5.00 raw_only_rows=80 mixed_rows=9705
raw_keys=1136780 comp_keys=267429 avg_raw_per_row=116.18 avg_comp_per_row=27.33
same_raw_start=440 same_raw_count=1517 same_comp_count=720 max_raw=128
max_comp=84`. This dwarfs the raw vec-rows landing point (`380` calls in the
same run). Prioritize a mixed/compressed shared-prefix rows kernel or fused
attention pipeline before any further raw-only experiments.

The follow-up intersection profile
`bench-results/dspark_rows_exact_intersections_n320` remained byte-clean
(`cmp=0`) and measured `40.64 t/s`, acceptance `84.6%`. It added
`same_counts=720 same_shape=0 same_shape_mixed=0`, proving that the promising
independent counters do not intersect once raw start is included. Do not build a
same-shape mixed rows helper; it would have zero coverage on the current local
Flash resident trace.

The follow-up common-prefix estimator
`bench-results/dspark_rows_exact_reuse_n320` also remained byte-clean (`cmp=0`)
and measured `40.73 t/s`. It reported `raw_common_all=220408`,
`raw_shared_est=255148`, `raw_reuse_est=4.46x`, `comp_common_all=52890`,
`comp_shared_est=55869`, and `comp_reuse_est=4.79x`. This keeps Plan C alive:
the shared-prefix mixed/compressed kernel has real scan reuse even though a
same-shape helper has none.

Mixed vec-rows scaffold status:
`DS4_DSPARK_ATTN_MIXED_VEC_ROWS_EXPERIMENT=1` packs the raw union plus compressed
rows into one vector FlashAttention stream with per-row masks. It is a useful
Plan C probe, not a strict-v1 optimization. With
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_EXPERIMENT=1`, n=64 matched the row-exact deferred
control (`cmp=0`) with identical acceptance. n=160 still matched final output
(`cmp=0`) but changed acceptance/blocking stats (`126/165` control vs `125/174`
scaffold) and slowed from `39.04 t/s` to `37.23 t/s`. Treat this as evidence
that the vector pack+mask family can preserve short greedy bytes, but not as
evidence that it preserves strict verifier logits/state. A strict-compatible
Plan C kernel must preserve the row-exact mixed attention contract more tightly
or explicitly move into the unified/canonical contract with its own baseline.

Shared-row mixed exact probe:
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_EXACT=1` keeps the row-exact attention update
helper and skips invisible keys rather than masking them. It stages one raw or
compressed K/V row per threadgroup and lets up to five verifier rows and two
heads consume it. n=64 matched the deferred row-exact control (`cmp=0`) with
identical acceptance, but n=160 diverged (`cmp=1`) and changed acceptance from
`76.4% (126/165)` to `73.1% (125/171)`. Adding a visible-row physical raw-cache
consistency guard did not fix it. Treat this as a diagnostic: strict Plan C
needs a tighter `attn-heads` stage audit or a variant that preserves the exact
heads8 grouping while only sharing loads proven not to affect verifier logits.
The committed/no-compare path is now guarded: use
`DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1` for compare/restore, or add
`DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1` only to reproduce the rejected committed
diagnostic.
Gate fix: mixed shared and mixed vec envs now auto-activate the deferred
row-exact head path, so the probes no longer silently bypass when the older
defer flags are omitted. `bench-results/dspark_mixed_shared_gatefix_020444`
entered the branch, logged `mixed-shared-attn` deltas, fell back to rows-exact,
and stayed `cmp=0`; no-compare
`bench-results/dspark_mixed_shared_defer_nocompare_020252` still diverged
(`cmp=1`) for both heads2 and heads8.

Heads8 shared-row follow-up:
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_HEADS8=1` preserves the strict path's
eight-head simdgroup layout and loops the N<=5 verifier rows inside that layout.
It did not fix strict exactness. Compare mode remained fallback-clean but still
showed about `1e-6` `attn-heads` deltas; no-compare n=160 diverged (`cmp=1`),
measured `36.82 t/s`, and changed acceptance to `70.5% (124/176)`. This closes
the "just keep heads8 grouping" branch. Do not promote this path; use it only as
evidence that strict shared-prefix attention needs either a lower-level stage
audit/fix or an explicit unified/canonical target-forward contract.

Mixed shared compare/fallback probe:
`DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_COMPARE=1`, plus `_ALL` variants) now
measures that candidate without committing it. The candidate writes to
`batch_heads`, is copied to `batch_heads_raw`, then strict row-exact heads are
restored into `batch_heads` before the graph continues. n=160 remained `cmp=0`
against the row-exact control at `37.47 t/s` with identical acceptance
`76.4% (126/165)`. The first two verifier positions show layer-local max deltas
roughly `7.15e-07` to `2.38e-06`, so the candidate is numerically close but not
strict-identical. Do not spend time debugging raw visibility for this probe; the
next strict-compatible design must either share prefix loads while preserving the
existing per-row FlashAttention vec/reduce realization, or switch Plan C to a
unified canonical target-forward contract with its own N=1 baseline.

Mixed vec compare/fallback probe:
`DS4_DSPARK_ATTN_MIXED_VEC_COMPARE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_VEC_COMPARE=1`, plus `_ALL` variants)
does the same candidate-copy/strict-restore/log flow for the pack+mask mixed
vec-rows path. n=64 remained `cmp=0` against the row-exact control with
identical acceptance, but the local candidate still differed: first 24 logs
peaked at `1.43e-06` max with RMS roughly `1e-08` to `3e-08`
(`bench-results/dspark_mixed_vec_compare_n64`). This rules out the simple
"reuse vec/reduce with raw-union masking" route as a strict-v1 replacement. The
strict-compatible shared-prefix kernel must keep each row's exact gathered key
stream and chunk geometry while removing redundant copies/launches around it.

Paired-copy and prefix-N checkpoint:
The strict frontier snapshot/restore/commit helpers now use paired KV+score blits
for attention and indexer state. This is safe cleanup but not a speed lever:
`bench-results/dspark_paircopy_060216` stayed `cmp=0` and measured 37.25 t/s
against a paired 33.29 t/s baseline, with draft 16.94 ms, verify 66.86 ms,
overhead 1.82 ms, and commit 1.38 ms per active-5 block. The fine attention
profile before/after paired blits was effectively unchanged. The companion
`DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1` A/B also stayed `cmp=0`, but fell to
31.25 t/s because partial-accept commit rose to 25.49 ms/block. Keep prefix-N
capture on; exact replay for partial accepts is a confirmed regression.

Rows-exact offset cleanup:
The strict rows path now calls offset-aware raw/gathered FlashAttention encoders
instead of allocating `ds4_gpu_tensor_view` wrappers for each verifier row. This
preserves the old kernels and row streams exactly. It passed n=64 and n=160
`cmp=0` against the previous row-exact control with identical acceptance, but it
did not improve speed in the short tests (`36.29` vs `36.81 t/s` at n=64,
`38.77` vs `39.04 t/s` at n=160). Treat it as cleanup and a cleaner future
varstream insertion point, not as the speed lever.

Next exact-kernel target:
Build `DS4_DSPARK_ATTN_VARSTREAM_COMPARE=1` before any commit path. The kernel
should keep the strict per-row gathered stream contract:

- Prepare exact per-row F16 streams in one scratch buffer:
  `[row0 raw+comp | row1 raw+comp | ...]`.
- Pass `row_offset[t]` and `row_n_keys[t]` to a rows5 FlashAttention kernel.
- Initial scope may assert `nwg=32`, `nsg=1`, `head_dim=512`, `n_tokens<=5`,
  and `row_n_keys<2048`, which covers the current local c=4096 DSpark traces.
- Each row must loop only to its own `row_n_keys`; no raw-union mask, no padded
  max-length row, and no changed chunk geometry.
- Compare mode must copy candidate heads to scratch, restore row-exact heads,
  and log candidate-vs-reference deltas. Commit mode is forbidden until the
  compare max/rms is zero on short and medium runs.

This first prototype may still duplicate common keys across row streams. That is
acceptable: the first question is whether one rows5 varstream FlashAttention
dispatch can be bit-identical while reducing launch/reduce overhead. Only after
that proof should we try sharing F16 conversion for common raw/compressed
prefixes.

Sweep integration: `scripts/dspark_phase0_sweep.sh` now enables the first-block
profile by default and `scripts/parse_dspark_phase0.py` emits `sp_*` columns.
Fresh n=64 active sweep `bench-results/dspark_sharedprefix_active_sweep_n64_185504`
was byte-clean for budgets 2/3/4/5. On that short run, active-4 was fastest
(`36.23 t/s` vs baseline `33.37 t/s`), while active-5 had the largest estimated
shared-prefix opportunity (`raw 4945 -> 1333`, `3.71x`; compressed
`567 -> 147`, `3.86x`) but slightly lower speed (`36.12 t/s`) due to higher
verifier wall.

Plan-C backend smoke:
`bench-results/dspark_planc_sharedprefix_backend_n32_190227` verified the
`shared_prefix` backend label. It is currently backed by strict-v1 core plus
profiling, matched the no-draft baseline and default strict DSpark (`cmp=0`),
and reported `unified_backend=shared_prefix`, `tfwd_verify_avg_ms=68.712`,
generation `33.03 t/s`, verifier `61.11 ms`, tau `3.56`. This is not the speed
kernel yet; it is the clean replacement point for `dspark_attn_shared_prefix_exact_n5`.

Secondary fusion target: once the shared-prefix split is correct, fuse
KV-append -> scores -> softmax -> AV -> inverse-RoPE -> HC/output inside one
per-layer verifier kernel where it preserves the exact mutation and accumulation
order.

Applicability to both verifier plans:

- **Plan A / strict exact:** shared-prefix batching must reproduce the no-draft
  decode realization byte-for-byte. The new-block triangle and online-softmax
  merge are the danger points.
- **Plan B / batch canonical:** the batched causal forward becomes the canonical
  verifier state. Prefill-style shared-prefix attention/compressor/indexer reuse
  is allowed when the committed state is self-consistent and acceptance/tau do
  not regress.

Name the shared-context APIs around this contract split, e.g. `strict_exact`
versus `batch_canonical`, so one mode does not accidentally inherit the other
mode's proof assumptions.

### Patch F: ROWS5 Shared-Weight Dense Kernels

Use after or alongside Patch E for comp/index/projection matvecs that still
appear in the census. One kernel should load each weight tile once and loop
N<=5 row accumulators internally while preserving the same per-output reduction
order.

Current status: Q8 rows5 has landed as default-on for the strict verifier's
N<=5 exact Q8 row projections. Disable with
`DS4_DSPARK_VERIFY_Q8_ROWS5_DISABLE=1` or
`DS4_DSPARK_VERIFY_NO_Q8_ROWS5=1`; force the diagnostic sequential rows5 kernel
with `DS4_DSPARK_VERIFY_Q8_ROWS5_SEQ=1`. Evidence:
`bench-results/dspark_q8rows5_default_n160_b4` is byte-clean (`cmp=0`), logs
`dspark verifier Q8 rows5 shared-exact kernel enabled`, and measured
`38.03 t/s`, verifier `62.90 ms`, tau `4.21` on the n=160 b4 smoke. The
same-shape default before promotion was `37.40 t/s`, verifier `64.87 ms`.

F16 rows5 remains opt-in and diagnostic; the current shared F16 kernel is
byte-clean but slower on local n=64 b4. FP8 draft rows5 is default-on for DSpark
dense FP8 N<=5 draft blocks; disable with
`DS4_DSPARK_DRAFT_FP8_ROWS5_DISABLE=1` or
`DS4_DSPARK_DRAFT_NO_FP8_ROWS5=1`. It shares the FP8 weight/scale traversal
across draft rows but keeps the original per-token reduction order. Evidence:
`bench-results/dspark_draft_fp8rows5_n64_b4`, `_n160_b4`, `_n1000_b4`, and
`_n4000_b4` all match the paired no-draft output (`cmp=0`). The n=160 b4 run
measured `38.45 t/s`, draft `14.45 ms`, verify `63.39 ms`, tau `4.21`,
acceptance `83.0%`; the n=1000 b4 run measured `37.95 t/s` versus paired
no-draft `32.93 t/s`, draft `14.87 ms`, verify `64.13 ms`, tau `4.22`,
acceptance `80.4%`; the n=4000 b4 run measured `35.74 t/s` versus paired
no-draft `31.54 t/s`, draft `14.93 ms`, verify `67.86 ms`, tau `4.15`,
acceptance `79.0%`. Nearest no-FP8 rows5 n=160 b4 compares had draft around
`16.0 ms`. Next ROWS5 targets are only MXFP4 dense variants if profiling shows
they matter.

Shared-down+HC rows has landed as a strict-mode micro-win. It dispatches the
same Q8 shared-down reduction and HC expansion for N<=5 rows in one kernel,
preserving the per-token reduction and HC add order while removing the old
CPU-side verifier-row loop. Disable with
`DS4_DSPARK_HYBRID_ROW_SHARED_DOWN_HC_ROWS_DISABLE=1`. Evidence:
`bench-results/dspark_shared_down_rows_n64_053950`, `_n160_054138`, and
`_n1000_054326` all stayed `cmp=0` with identical acceptance against the
disabled path. The n=1000 A/B measured rows default `38.29 t/s`, verifier
`77.23 ms`, versus disabled `38.00 t/s`, verifier `78.24 ms`. Keep this
default-on, but treat it as a narrow tail cleanup; the main remaining speed
target is still attention/cache/indexer shared-prefix work.

### Patch G: Grouped Exact Routed-MoE Prototype

Build a GPU-side grouped expert-row verifier path now that Phase 0 found real
reuse. This is still exact execution, not a semantic shortcut, and it follows
the attention work in priority.

Current implementation status: the GPU grouped descriptor foundation has landed
behind `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1`. It builds grouped
route metadata on GPU for N<=5 and then deliberately falls back to the existing
row-exact math. The n=64 opt-in smoke matched descriptor-off output
byte-for-byte (`cmp=0`), with identical acceptance, and printed the grouped
descriptor log line. This is the scaffolding for the real speed work, not the
speed work itself.

First consumer status: `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_IQ2=1` enables a
byte-clean IQ2 gate/up+SwiGLU grouped experiment, but it is slower than the
default pair kernel. In the paired n=64 profile, row-routed mean increased from
34.08 ms to 54.15 ms. Do not promote this kernel; use the result to avoid
high-register grouped gate/up designs and move the next attempt toward grouped
down/ordered-sum or a more compact expert-row kernel.

Direct ordered Q2 status: this is now the default strict local Flash IQ2/Q2 path.
It keeps the normal row-exact IQ2 gate/up+SwiGLU path, then computes Q2 down
projection and the six-slot ordered FP32 sum in one kernel. It writes the summed
output directly rather than materializing `slot_down[row][slot]`, but the
focused boundary compare proved it exact for observed N<=5 verifier calls.
Disable with `DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` for
A/B against separate down plus ordered-sum. Useful evidence: n=1000 direct
ordered-Q2 reached `39.66 t/s` versus same-session conservative `39.12 t/s`;
n=4000 reached `36.73 t/s` versus `36.47 t/s`, with identical acceptance. The
focused compare hook
`DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1` now gives per-call
MoE-boundary evidence without changing production output; first bucketed run
`bench-results/dspark_direct_q2_compare_buckets_022458` reported exact
`max=0` for `tokens=1`, `tokens=5`, and tail `tokens=3`. Promoted-default
validation `bench-results/dspark_direct_q2_promoted_default_023137` reached
`39.71 t/s` at n=1000 and matched the paired no-draft baseline output. After
the dispatch-est fix, `bench-results/dspark_direct_q2_dispatchfix_024232`
reports `ordered_sum=0`, total `893`, so the next strict work should target
attention heads/compressor/indexer dispatch and cache traffic.

Current wrong-fusion confirmation:
`bench-results/dspark_wrong_fusion_current_n160_062305` keeps default DSpark
active-5 byte-clean (`cmp=0`, baseline `33.41 t/s`, DSpark `39.23 t/s`,
acceptance `76.4%`) while the forced unsafe grouped-Q2 shared-weight path
diverges (`cmp=1`, first diff swaps `RED`/`GREEN`). Its in-call slot-down
compare reports active-5 `exact=no`, so this is still the rejected FP realization.
Do not spend the main optimization track on grouped shared-weight Q2 until a
boundary proof gives exact `max=0`; keep Plan C focused on attention/cache
staging and row/slot-exact fusion only.

Post-direct-Q2 measurement update:

- `bench-results/dspark_pairrows_ab_025145`: `DS4_DSPARK_QKV_PAIR_ROWS=1` plus
  `DS4_DSPARK_F16_PAIR_ROWS=1` remained byte-clean (`cmp=0`) but did not improve
  speed; default active-5 was `40.07 t/s`, pairrows was `39.96 t/s`.
- `bench-results/dspark_f16_rows5_ab_025918`: existing F16 rows5 verifier
  kernels are byte-clean but slower. Default active-5 was `40.07 t/s`, shared
  F16 rows5 was `39.14 t/s`, and seq F16 rows5 was `39.30 t/s`. Keep
  `DS4_DSPARK_VERIFY_F16_ROWS5{,_SEQ}` diagnostic-only; HC-pre needs a new
  fused shape rather than this helper.
- `bench-results/dspark_varmap_profile_025345`: varmap rows attention host
  encode is negligible (`2.98 ms` total host time across 2337 varmap calls in
  the n=320 run). DSpark reached `41.79 t/s`, verify `75.95 ms`, acceptance
  `84.6%`. Do not chase CPU command encoding here; the cost is GPU-side work and
  cache traffic.
- `bench-results/dspark_attn_subprofile_after_profilefix_025610`: the profiling
  double-count around row-output HC was removed. The fenced microscope now
  reports output once per layer (`/43`) and shows the approximate priority:
  HC-pre `42-45 ms`, output `25-28 ms`, q `21-24 ms`, heads `18-23 ms`,
  compressor `12-14 ms`, kv `10-12 ms`, indexer `4.8-5.4 ms`. Absolute times
  are fence-inflated; use the ordering, not the total, for kernel planning.
- `bench-results/dspark_hc_output_subprofile_030709`: new
  `DS4_DSPARK_HC_PRE_SUBPROFILE=1` and `DS4_DSPARK_OUTPUT_SUBPROFILE=1` split
  HC/output further. Active-5 fenced detail is roughly HC RMS `40-42 ms/43`,
  HC function `10-11 ms/43`, HC split/norm `10-11 ms/43`, output inverse-RoPE
  `8-10 ms/43`, output-low Q8 `19-23 ms/43`, and output-HC expand
  `17-20 ms/43`. Clean no-profile A/B in the same directory stayed `cmp=0`:
  baseline `33.33 t/s`, no-direct-Q2 DSpark `39.56 t/s`, direct ordered-Q2
  DSpark `39.99 t/s`. This confirms the direct Q2 down+ordered-sum path is a
  narrow byte-clean micro-win, not the large DSpark fusion boundary. Do not
  extend this into unordered `sum6` or grouped shared-weight Q2 math without a
  new one-layer exactness proof. The current grouped-Q2 proof hook is
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_COMPARE=1`; audit
`bench-results/dspark_fusion_audit_032203` shows the unsafe shared-weight
grouped-Q2 kernel differs from the safe wrapper in the slot-down tensor by
ULP-level active-5 deltas (`max` up to about `4.8e-7` in the n=32 compare)
and still drifts to final `cmp=1` at n=160.
Fresh 2026-06-29 recheck `bench-results/dspark_wrong_fusion_recheck_040257`
keeps the diagnosis: direct ordered-Q2 compare is exact (`max=0`), but grouped
shared-weight Q2 is not. The unsafe path happened to preserve final n=96 text,
yet its active-5 slot-down tensor already differed by up to `4.768e-7`; the same
path diverged at n=160 (`cmp=1`, first diff swaps `RED`/`GREEN`). Do not use this
kernel family as the template for Plan A/B/C fusion. The only acceptable MoE
fusion direction is row/slot-exact: independent slot accumulators, identical
per-slot dot-product reduction, and identical ordered FP32 slot add.
- `bench-results/dspark_grouped_q2_clean_042538` reran the MoE variants without
  the compare hook to remove measurement contamination. n=160 results: paired
  baseline `33.37 t/s`, default direct ordered-Q2 DSpark `39.87 t/s` (`cmp=0`),
  grouped-safe descriptor wrapper `38.34 t/s` (`cmp=0`), unsafe shared-weight
  grouped Q2 `33.74 t/s` (`cmp=1`). Conclusion: the safe descriptor wrapper is
  correct but does not beat the default direct ordered-Q2 fusion; the unsafe
  shared-weight dot is both wrong and slower. Route grouping/output layout are
  safe, so a future exact grouped MoE must reuse the same per-slot dot-product
  FP realization before trying to share expert tiles.
- `bench-results/dspark_grouped_q2_compare_044810` compared primary=unsafe
  shared-weight grouped Q2 against the safe wrapper inside the same verifier.
  Token-count-1 calls were exact, but active-5 slot-down tensors differed by
  about `4.47e-8` to `4.77e-7`. This is enough to diverge over the verifier chain
  and confirms the issue is FP reduction/order drift rather than route grouping,
  output layout, or gross addressing.
- `bench-results/dspark_default_after_fusion_guard_034504`: fresh guard recheck
  keeps the default strict path clean at n=160 (`cmp=0`, DSpark `39.94 t/s`,
  paired baseline `33.28 t/s`, acceptance `76.4%`). The forced unsafe grouped-Q2
  path remains rejected (`cmp=1`, `33.83 t/s`, acceptance `68.0%`); the in-call
  compare hook saw 305 calls with worst slot-down delta `7.15e-7`. This confirms
  the bad fusion is FP-realization/reduction order, not route grouping or output
  layout.
- `bench-results/dspark_output_low_rows5_034252`: the layout-aware output-low Q8
  rows5 diagnostic is byte-clean but slower at n=96 (`37.93 t/s` enabled versus
  `38.76 t/s` disabled, both `cmp=0`). It is now opt-in via
  `DS4_DSPARK_OUTPUT_LOW_Q8_ROWS5=1`; keep default off.
- `bench-results/dspark_hc_scaled_f16_031619`: opt-in
  `DS4_DSPARK_HC_PRE_SCALED_F16=1` computes HC RMS scales into a tiny buffer and
  runs an exact scaled F16 rows5 HC projection without writing `batch_flat_hc`.
  It stayed byte-clean at n=64 (`cmp=0` versus both no-draft and default DSpark)
  but slowed default DSpark from `37.91 t/s` to `34.65 t/s`, with verify moving
  from `64.84 ms` to `75.28 ms`. Keep it diagnostic-only; the scale-buffer plus
  rows5 F16 shape is not the HC-pre win.

Grouped Q2 down status: the route-grouping descriptor itself is safe, but the
first shared-weight grouped Q2 down materializer is not. The grouped diagnostic
now defaults to the safe exact wrapper; forcing the rejected shared-weight math
requires `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_UNSAFE=1`. With that
unsafe path, n=160 diverged (`cmp=1`) and slowed to `33.17 t/s`, verifier
`77.89 ms`, acceptance `68.0%`. With the safe wrapper, the same descriptor and
output layout run through the exact Q2 row kernel per pair and match both
default DSpark and no-draft output (`cmp=0`, n=160 `38.15 t/s`). Treat the
shared-weight grouped Q2 math as rejected until a single-layer exactness test
proves its dot-product/reduction schedule; do not promote this path.

Goal: reduce per-row routed-MoE cost without changing verifier semantics.

Allowed:

- group row/slot expert work by expert id
- dequant/load expert tiles once per expert group when reused
- apply the same expert tile to all rows using that expert
- write separate `slot_down[row][slot]`
- run the existing ordered FP32 expert-sum kernel

Forbidden:

- host readback of router ids inside the layer loop
- batched router selection
- batched shared expert
- unordered direct routed-down `sum6` as default
- changed expert add order

Safe kernel contract:

- Router logits/top-k/weights stay row-exact.
- Semantic order remains `route_id[row][slot]` and `route_weight[row][slot]`.
- Execution may group by expert only internally.
- Each row/slot expert matvec writes the same separate down vector as the old
  path.
- Final routed output uses the existing exact ordered FP32 slot sum.
- No batched router shortcut, no batched shared shortcut, and no unordered
  direct `sum6` default.
- The local Flash Q2 path has one proven exception:
  `kernel_mul_mv_id_q2_K_sum6_ordered_f32` keeps independent slot accumulators,
  reduces each slot separately, then applies the exact slot0..slot5 FP32 add
  chain. It is enabled by default only for Q2 down experts and is guarded by
  `DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1`.
- For grouped-Q2 diagnostics,
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_COMPARE=1` compares the safe
  descriptor wrapper against the unsafe shared-weight grouped kernel before
  the ordered expert sum. Treat any nonzero delta as a stop sign for promotion;
  current evidence says the unsafe kernel is FP-realization-wrong, not
  route-layout-wrong.

Target flag:

```text
DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1
```

Do not use CPU readback in this kernel path. The descriptor-builder option now
exists; the next successful patch should consume that descriptor without the
register pressure seen in the IQ2 gate/up experiment, write the same separate
`slot_down[row][slot]`, and keep the existing ordered FP32 sum.

### Patch H: Row-Offset Attention APIs

Refactor the strict attention verifier without changing math:

- Replace hot-loop `metal_graph_tensor_row_view()` / free pairs where possible
  with base tensor plus row offset APIs.
- Avoid scratch-to-batch-row copy-back kernels unless audit mode needs them.
- Keep exact row-local compressor/indexer behavior. The fine subprofile says the
  first useful target is reducing the number of exact compressor/indexer
  update/capture calls and their row-view/copy overhead, not changing attention
  arithmetic.
- Keep exact prefix capture. Any capture optimization must preserve prefix 1..4
  frontiers for partial accept commit.

Target shape:

```c
metal_graph_encode_dspark_attn_exact_rows_n5(
    graph,
    layer,
    batch_cur_hc,
    row_stride,
    n_rows,
    pos0,
    prefix_capture_count,
    workspace);
```

This is a refactor-first patch, not a new batched attention kernel.

### Patch I: Legacy Strict-V2 Attention Microbatch Checklist

This checklist is kept as guardrails for Patch E. Any DSpark-specific attention
microbatch must preserve:

- same row order
- same arithmetic
- same cache/frontier visibility
- exact row-local KV visibility
- exact row-local compressor projection where required
- exact row-local indexer Q/weight projection where required
- exact compressor/indexer frontier mutation
- exact attention-head read using each row's frontier
- exact attention output / HC update
- prefix frontier snapshot if row < N - 1

Success gate:

- n=160, n=1000, n=4000 `cmp=0`
- stage audit clean at `1e-8`
- `final_hc_max=0`
- `decode_hc_max=0`
- `dspark_hidden max=0`
- `dspark_kv max=0`

### Patch J: Exact Routed MoE V2

This is superseded by Patch E's grouped exact routed-MoE plan. Recover more of
the `sum6` sizing win without changing math.

Allowed:

- group row/expert work to reduce dispatch
- use the N<=5 row dimension
- keep separate down outputs
- use exact ordered FP32 add

Forbidden:

- unordered fused down-sum6 as default verifier
- changed expert add order
- changed selected expert order
- changed router weights

Target shape:

```c
metal_graph_encode_dspark_routed_exact_rows_n5(
    graph,
    layer,
    ffn_norm_rows,
    router_ids_rows,
    router_weights_rows,
    routed_down_slots,
    routed_out_rows);
```

## Command Reuse / Metal ICB

Pursue only if the dispatch census and active-vs-idle counters show CPU
encode/launch overhead remains material after Patch E. Record the fixed N<=5
verifier block once into a reusable command stream / Metal ICB and update only
the changing offsets and bindings per block.

## Separate-Agent Side Track: ANE Draft Overlap

ANE draft overlap is a separate assignment for a separate agent. It should not
block or reorder the verifier/MoE main track. Do not treat ANE as a current
main-track optimization. Question for that agent: can DSpark draft(k+1) run on
ANE while GPU verifies block k? Do not change strict verifier semantics, target
commit state, or the `cmp=0` validation gate.

## 2026-07-01 Current-Build Speed Findings

Use these as stop signs for near-term retests unless the implementation changes.

- Current reliable fast-relaxed Pygame baseline is still below the goal:
  `bench-results/dspark_verify_budget_sweep_030147/fast_v5.err` measured
  `52.02 t/s`, `block=81.36 ms`, `draft=10.73 ms`, `verify=68.51 ms`,
  `tau=4.35`, canary-clean. `verify=3` and `verify=4` were slower
  (`46.98` and `49.99 t/s`).
- Draft profile says further Markov-only work is too small to carry the goal:
  `bench-results/dspark_draft_profile_fast_v5_030638` shows steady-state draft
  `~10.6 ms/block`, with the three draft layers at `~9.0 ms` and Markov at
  only `~1.6 ms`.
- Route overlap is real but current grouped flags are not useful:
  `bench-results/dspark_hybrid_stage_route_profile_030809` measured about
  `1.6x` routed-expert reuse, but
  `bench-results/dspark_route_group_flags_fast_ab_030936` showed current
  grouped IQ2 neutral/slightly slower, safe grouped Q2 slower, and the combo
  much slower.
- Case E/NAX attention is not a current-build win for the fast path:
  `bench-results/dspark_nax_fast_ab_032206` measured baseline `51.97 t/s`
  versus NAX/NAX-topk `~48.6 t/s`, canary-clean but slower.
- A speculative bonus-token experiment was implemented and then removed. With
  this row-exact verifier, the next block spends a draft slot consuming the
  pending bonus, so net `tau` and `t/s` stayed flat (`~51.9 t/s`) and output
  diverged in whitespace. Do not re-add bonus tokens unless the next iteration
  can advance the pending token before drafting a full new block.
- Aggressive accept does not solve the goal safely:
  `bench-results/dspark_aggressive_accept_sweep_032527` measured
  `delta12/topk256` slower (`49.26 t/s`), unbounded relaxed slower and
  canary-suspect, and draft-only extremely fast (`405.75 t/s`) but obvious
  garbage. Current `--draft-fast-relaxed` remains the only canary-clean fast
  diagnostic, and it is still below `60 t/s`.
- Opt-in GPU draft prefetch was implemented behind `DS4_DSPARK_DRAFT_PREFETCH=1`
  as a ceiling probe. It submits the next DSpark draft command buffer after a
  successful frontier commit and consumes it on the next frontier block. The
  path works and logs real hits, but it does not make the `~61 t/s` overlap
  upper bound real in `./ds4`: `bench-results/dspark_prefetch_fast_n1000_052102`
  measured `52.44 t/s`; `bench-results/dspark_prefetch_nolog_n1000_052248`
  measured only `52.58 t/s` with prefetch. Draft time fell from `10.64` to
  `10.14 ms/block`, but total block wall stayed about `80 ms` because the CLI
  loop has almost no host/output slack to overlap with GPU draft work. Treat
  this as an opt-in diagnostic, not the main route to the target.
- Fresh fixed-budget sweep with prefetch confirms budget 5 is still the best
  short-context fast setting. `bench-results/dspark_fast_cap*_prefetch_n1000_052655`
  measured cap2 `39.44 t/s`, cap3 `48.05 t/s`, cap4 `50.80 t/s`, and cap5
  `52.55 t/s`. Smaller caps improve per-block wall but lose too much `tau`.
- Fresh Case-E/NAX stack check confirms it is not a short-context speed lever:
  `bench-results/dspark_combo_nax*_n1000_052422` measured `~48.2 t/s` with
  NAX, NAX-topk, and NAX-topk+prefetch, all slower than the current fast preset.
- Fresh row-routed/native batch-canonical probe did not move the current Flash
  IQ2/Q2 sidecar target. `DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_CANONICAL=1`
  bypasses the MXFP4 row-exact native loop when that native plane-split branch
  is selected, but `bench-results/dspark_rowrouted_{control,canonical}_n1000_054116`
  both measured about `52.4 t/s`, identical output, identical `~67.5 ms`
  verifier wall. For the current sidecar, the active target verifier path is
  the generic IQ2/Q2 row-routed helper, not the MXFP4 native branch.
- Fresh slot-bank sweep rejects smaller resident banks for DSpark verifier
  throughput. `bench-results/dspark_fast_bank64_n1000_054322` measured
  `17.42 t/s` and `bench-results/dspark_fast_bank128_n1000_054322` measured
  `19.46 t/s`; both had verifier GPU-busy below `1%`, so the verifier became
  expert-residency/SSD-bound. Keep the full `--resident`/256-slot bank for
  DSpark speed runs even though it is memory-heavy.
- Fresh relaxed-accept sweep found a slightly better demo preset but not the
  goal. `bench-results/dspark_relaxed_top256_delta10_n1000_054620` measured
  `54.20 t/s`, `tau=4.58`, `acceptance=93.6%`, canary-clean; wider deltas
  (`delta12`, `delta14`) dropped to `~49 t/s`. The longer Pygame confirmation
  `bench-results/dspark_relaxed_top256_delta10_n4000_055143` measured
  `49.88 t/s`, `tau=4.46`, `acceptance=90.9%`, canary-clean. The CLI
  `--draft-fast-relaxed` preset now defaults to `TOPK=256` /
  `LOGIT_DELTA=10`, but this is still a non-byte diagnostic below `60 t/s`.
- Pairing that preset with GPU command-buffer draft prefetch is still only a
  tiny gain: `bench-results/dspark_fastrelaxed_prefetch_preset_n1000_055510`
  measured `54.40 t/s`, canary-clean, with a printed draft-overlap upper bound
  of `63.5 t/s`. This is a useful target for a real separate-engine overlap
  path, not something the current GPU-only prefetch reaches.
- Continuation recheck of the exact HTML stress prompt
  (`/tmp/si-html-010`, 4000 tokens) reproduced the same failure mode as the
  earlier fast-relaxed artifact. `bench-results/si_html_010_fastrelaxed_n4000_065005`
  measured `36.57 t/s`, `tau=3.93`, acceptance `77.6%`, and canary
  `suspect=1`; its output is byte-identical to
  `bench-results/dspark_fastrelaxed_html_n4000_full_060318`. Strict DSpark is
  still byte-identical to no-draft for this prompt, and no-draft/strict are
  also canary-suspect, so this prompt remains a useful stress canary rather
  than a simple proof that one relaxed gate is wrong.
- Relaxed gate retest split the failure into two pieces. With the target-top
  loop guard enabled, even the easy Pygame prompt collapsed to about
  `23.5-24.0 t/s` and `tau=2.68-2.86`
  (`bench-results/dspark_relaxed_gate_sweep_065432.tsv`), so checking every
  target-top repeat is a safety brake, not a speed path. With the target-top
  bypass restored, the current permissive settings reproduced the known
  `54.18 t/s`, `tau=4.58`, acceptance `93.6%`, canary-clean Pygame point, but
  stricter off-argmax variants (`top64/delta4/off2`,
  `top32/delta3/off1`, and `top64/delta4/temp0.2/off2`) all fell to
  `~48.7 t/s` on Pygame and still left the HTML n=1000 smoke canary suspect
  (`bench-results/dspark_relaxed_offargmax_sweep_065907.tsv`). Conclusion:
  top-k/delta/temperature/off-argmax tightening can trade speed for safety, but
  it does not expose a usable >60 operating point and does not fix the hard
  HTML stress case.
- Re-audited the "bonus token" idea at the current call contract. DSpark-5
  drafts include the sampled first target token, so a full-accept block already
  consumes the five available draft rows. The verifier's final logits can name
  the next token, but the target/Metal state is not advanced through that token.
  Emitting it early only moves the first row of the next verifier block unless
  the next iteration can both consume the pending token and still draft/verify a
  full five fresh rows. With the current five-row DSpark package, this cannot
  raise net `tau`; do not pursue it without a real six-row/pending-token
  verifier contract.

Net: simple scheduler, accept-gate, Markov, current NAX, and current grouped
route flags are exhausted; GPU command-buffer draft prefetch is also exhausted
for CLI. Relaxed acceptance can move the demo point into the mid-50s, but it
has not crossed `60 t/s`. To cross `60 t/s` on this verifier family, the next
real work is either (1) a new exact-ish verifier kernel that reduces the
`~68-73 ms` verify wall, especially the generic IQ2/Q2 row-routed path or a
real batch-canonical contract, or (2) a true separate-engine path, likely ANE,
that hides the `~10 ms` DSpark draft under target verification instead of under
the tiny host loop.

## Secondary Cleanup

Only after the attention/MoE work:

- persistent verifier workspace
- one packed prefix-capture descriptor kernel
- row-offset output-HC APIs
- row-offset router APIs
- audit-only scratch copies behind audit flags

## Promotion Checklist

- Every new path has a disable gate.
- Strict-v1 remains selectable and unchanged.
- New strict-v2 path passes n=160, n=1000, and n=4000 `cmp=0`.
- Focused block audits pass at `1e-8`.
- Do not use `DS4_AGENT_ALLOW_BACKEND_STATS=1`, `DS4_DSPARK_BLOCK_TIMING=1`
  (`DS4_DSPARK_TIMING=1` alias), or fenced stage profilers for headline t/s.
- Watch VM/compressed-memory pressure when comparing runs.
