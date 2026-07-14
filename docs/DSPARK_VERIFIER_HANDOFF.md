# DSpark Verifier Handoff

Purpose: give a higher-end reviewer/agent enough current context to advise on the
remaining DSpark verifier performance/correctness problem without replaying the
whole thread.

## Current Goal

Implement DSpark draft inference and verification in DS4, then optimize until
DSpark draft+verify reaches **>60 generation t/s** with coherent
target-faithful output on the local Flash resident setup. Byte identity is
preferred for strict mode, but a non-byte-identical fast mode is acceptable if it
stays coherent and target-faithful.

## 2026-07-02 Rows-6 Frontier Verify — correctness solved, speed work open

The rows-6 exact frontier verify (`DS4_DSPARK_VERIFY_ROWS6=1` +
`DS4_DSPARK_FRONTIER_DRAFT=1`, currently also requires
`DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=0`) carries the known-correct
first token as row 0 plus all five Markov drafts, replacing the per-cycle
first-token decode with one verify row. Status:

- **cmp=0 at n=1000** with tau `4.83`, first-miss `0.0%`, blocks −26%
  (`bench-results/loop55_rows6_v1_052049`). The economics are real; the
  v1 speed is not yet (see below).
- Bugs found/fixed to get there (all latent 5-row assumptions):
  three `<=5u` FFN gates (`spec_decode_ffn` ds4.c:18148,
  `spec_fuse_hc_norm`, tiny-batch slotbank) that silently dropped the FFN
  into the prefill realization at 6 rows; the batch output-HC tail fusion
  is 5-row-only (its gate is now documented and kept at `<=5`; the
  per-row tail is the bit-exact path at 6); plus grown buffers/arrays
  (dspark tensors, dsv4_misc attention args ABI both sides, dense.metal
  NT kernels, prefix slots 4→5).
- Debug method worth reusing: env-gated stage dumps
  (`DS4_METAL_GRAPH_DUMP_PREFIX/NAME/LAYER`) + row-aligned numeric diffs
  between a 5-row and 6-row run at the same position
  (`scratchpad loop55_rows6_stagedump.sh` pattern) — 6-minute iterations,
  bit-level verdicts per stage.
- **Speed work remaining:** v1 runs per-row attention (batch attention
  path is 5-row-capped: varmap rows5 `FOR_UNROLL` loops at
  flash_attn.metal:2105/2139/2175/2214 and gates ds4.c:16491/17048) which
  costs ~47 ms/block → 32.5 t/s vs frontier-5's 34.4. Extending the batch
  attention + HC tail to 6 rows should put blocks at ~100 ms → **~48-50
  t/s byte-exact** per the measured tau.

## 2026-07-01 Strict-Contract Loop Findings (supersedes fast-relaxed direction)

User directive: fast-relaxed and all relaxed-accept gating are rejected as
incorrect. The active contract is strict target-argmax acceptance with output
byte-near-identical to no-draft. Findings from the strict-contract loop (all
unfenced, single-process gate, artifacts under `bench-results/loop55_*`):

- Ground truth: no-draft `35.76 t/s`, strict b5 `41.92 t/s` (`cmp=0` vs
  no-draft at n=1000). Agentic (new `scripts/dspark_agent_ab.sh`): no-draft
  `30.77`, strict b5 `33.47 t/s`, trajectory identical after KV-counter
  normalization.
- Scheduling knobs are noise-bound: identical strict configs measured
  `38.95-42.18 t/s` across batches (±2-3 t/s run noise). prefetch/frontier/
  softmax/dynamic and combos all land `41.6-42.3`; frontier+strict is newly
  proven `cmp=0` (removes the per-cycle first-token decode; block `81.3 ms`
  but tau `3.85 -> 3.41`, net tie). Do not rank scheduling variants on single
  n=1000 runs.
- **Exact MoE route-dedup is closed as a dead end.** The force-row0-routes
  sizing probe (`DS4_DSPARK_DIAG_FORCE_ROW0_ROUTES=1`, diagnostic-only) cut
  verify `76.4 -> 63.1 ms/block`, but batch-canonical MoE
  (`DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_CANONICAL=1`), whose kernels are
  bit-identical per row (`mxfp4_native.metal:220` vs `:323`) and `cmp=0`,
  is speed-neutral (`75.9` vs `76.5/77.5` controls). Duplicate expert reads
  are already L2-served; the probe's win was unique-working-set shrinkage
  (6 experts fit SLC, ~18.5 real uniques do not). A dedup kernel cannot
  reduce unique-expert bandwidth.
- **Forced-MMA strict audited and rejected under the current contract.**
  MMA-strict (strict accept + `DS4_DSPARK_ATTN_FORCE_MMA=1`) measured
  `47.38 t/s` canary and `34.49 t/s` on the 100-sample harness (1.09x —
  the only mode faster than no-draft there; strict is 0.89x) with quality
  metrics statistically unchanged (NLL `0.1723` vs `0.1706`, first_match
  `64=64`, LCP `7.08` vs `7.00`) but `byte_equal` only `25/100`. The new
  flip-gap audit (`gguf-tools/quality-testing/flip_gap_audit.c`,
  `make quality-flip-gap`) shows 23/75 first-divergence flips override a
  reference-realization logit gap > `0.2`, five > `0.5` (max `0.963`,
  a 72%/28% preference). That is not "identical-probability variation",
  so MMA stays opt-in unless the product contract is explicitly loosened.
- Draft wall (unfenced): trunk `9.0 ms` + Markov `1.6 ms` per block; the
  trunk+Markov already run as 2 command buffers with one readback. Fenced
  per-layer stage shares: attn_out `~1.05 ms` > routed `~0.93` >
  shared_router `~0.59` ~ q_proj `~0.56`.
- ANE draft-overlap audit: the draft trunk's heavy ops (attention, RoPE,
  router softmax, MXFP4 routed MoE) cannot move to ANE; only ~11 small
  matmuls (~1-2 ms) can, and the ANE shared-expert path is already a
  measured reject in this stack. Full draft hiding is off the table.
- Verify budget >5 closed by arithmetic: marginal row-6 EV ≈ `0.53` tokens
  for ~`13 ms` verify = `24.5 ms/token` vs the current `23.9` average
  (matches the flat b4→b5 gain), and `block_size=5` is baked into the
  trained draft package plus 20+ rows5 kernel specializations.
- Tree drafting closed by measurement+math: a new diagnostic
  (`DS4_DSPARK_TREE_OPP_LOG=1`, kept in tree) shows the drafter's runner-up
  equals the target at `54.7%` of first misses
  (`bench-results/loop55_tree_opp2_*`), but a branch verify-row's EV
  (`P(reach)×(1-p_k)×0.55 ≈ 0.02-0.08` tokens) is strictly dominated by
  chain depth (~`0.5`), which is itself already break-even.
- The frontier "committable correction token" idea is ALREADY implicit:
  `commit_drafts` starts at 1 (`ds4.c:31290`) and `drafts[0]==first_token`
  (`ds4.c:31043`) — the outer loop feeds the argmax of the last committed
  logits back as the next block's first row. Do not rebuild it.
- Landed: `DS4_DSPARK_DRAFT_PREFETCH=1` is now a default
  (cmp=0, ≥control in a 3-pair interleaved repeat,
  `bench-results/loop55_prefetch_repeat_152217`).
- **Converged byte-safe ceiling on this architecture/hardware: ~42-44 t/s.**
  Every structural byte-safe lever has been measured and closed. Reaching
  >55 under the strict contract requires strategic changes: a ground-up
  exact Mode-B verifier, draft-model retraining (larger block + stronger
  first-position acceptance; first-miss is 12-14%), or an explicit
  product decision on realization tolerance (forced-MMA class).
- **Landed later on 2026-07-01 — encode/execute pipelining (+2.4 t/s
  total, all cmp=0 at n=1000 and n=2500):**
  `DS4_DSPARK_VERIFY_SPLIT_LAYERS=4` is now a default (commit the verify
  command buffer every 4 layers; the GPU no longer idles through the
  CPU's 43-layer encode). Verify idle `8.0% -> 1.5%`,
  `42.0 -> 44.2 t/s` (`loop55_split_sweep_161123`,
  `loop55_split_confirm_161903`). The draft trunk got the same treatment
  (flush after each of the 3 draft blocks, `DS4_DSPARK_DRAFT_SPLIT_BLOCKS=0`
  to disable): draft wall `11.09 -> 10.46 ms`
  (`loop55_draftsplit_ab_165553`). `DS4_DSPARK_DRAFT_PREFETCH=1` is also
  a default now. **Current strict operating point: ~44.3 t/s.**
- Markov/base logit blend: the trained balance is optimal; scaling the
  Markov contribution 0.9/1.15/1.3 all lost acceptance
  (`loop55_alpha_verifier_155900`). Knob kept for future packages:
  `DS4_DSPARK_MARKOV_SCALE` (+ new `ds4_gpu_scale_f32_tensor`).
- Verify stage shares (fenced profile deflated ~1.9x, cross-checked
  against unfenced skip bounds): attention ~33 ms (largest; exact
  attention rework remains a rejected direction), routed MoE ~16-30,
  shared ~7, router ~6.5, ffn_pre ~5.5, batch tail ~6.5. The
  `DS4_DSPARK_ATTN_BYPASS` probe (verify `71 -> 14 ms`) overstates
  attention: skipping it collapses routes (L2 dedup worth ~13 ms alone)
  and the stage includes QKV/indexer/compressor/KV plumbing.
- Draft FP8->MXFP4 re-export is the next draft-side lever (~-2 ms draft,
  byte-safe): needs the source checkpoint volume (`/Volumes/TB36/...`)
  mounted, an F8_E4M3->MXFP4 conversion in `scripts/dspark_export.py`
  (precision is source-dtype-driven today), and an MXFP4 case in
  `ds4_dspark_matmul_record` wired to the existing scale-plane matmul.
- Bench-harness hardening: process gates must use `pgrep -x
  ds4|ds4-agent|ds4-server` (exact name), never `ps | grep` — an orphaned
  wrapper whose command line merely contained `./ds4` deadlocked every
  grep-based gate for an hour.

## 2026-06-30 Current Direction

Fresh n=1000 sidecar resident sweep says the current coherent/target-faithful
branch still tops out well below the >60 t/s target:

- no draft: `35.95 t/s`
- strict b5: `39.85 t/s` (`1.11x`, `tau=4.85`, full-accept `62.1%`)
- strict b4: `39.64 t/s` (`1.10x`, `tau=4.22`)
- Case E NAX b4: `42.12 t/s` (`1.17x`, opt-in/non-byte-identical)
- forced MMA b5: `44.58 t/s` (`1.24x`, best existing fast mode, non-byte-identical)
- forced MMA b5 + draft grouped-strided FP8 rows5: `46.79 t/s` (`1.30x`,
  non-byte-identical verifier, output-identical to the previous draft path)
- forced MMA b5 + confidence fast-Markov: `47.49 t/s` (`1.32x`,
  output-identical to the old confidence scheduler path)
- forced MMA b5 + confidence fast-Markov + fast-Q2: `48.00 t/s` (`1.34x`,
  current best observed fast stack, non-byte-identical)
- current `--draft-mode batch|unified`: `36.65-36.76 t/s`, not the real Mode B path

2026-07-01 continuation notes:

- Latest current-tree fast preset: `--draft-fast-relaxed` now enables
  `DS4_DSPARK_DRAFT_PREFETCH=1` internally. Recheck
  `bench-results/dspark_fastrelaxed_preset_prefetch_builtin_n1000_064652`
  measured `54.18 t/s`, block `82.67 ms`, draft `10.17 ms`, verify
  `70.33 ms`, tau `4.58`, acceptance `93.6%`, and simple canary `suspect=0`.
  This is the current reliable fast demo point, still below the >60 target.
  A no-stats recheck, `bench-results/fastrelaxed_nostats_n1000_072727`,
  measured `54.39 t/s` with the same `93.6%` acceptance and canary
  `suspect=0`, so profiler/backend-stat overhead is not hiding a >60 result.
- `ds4-agent --draft-fast-relaxed` has been aligned with the CLI preset:
  `DS4_DSPARK_DRAFT_PREFETCH=1`, `DS4_DSPARK_RELAXED_TOPK=256`, and
  `DS4_DSPARK_RELAXED_LOGIT_DELTA=10` are now the default fast-relaxed agent
  gates unless explicitly overridden in the environment. This avoids testing a
  stale agent preset while comparing CLI and agent demo speed.
- Fresh no-profiler ceiling recheck
  `bench-results/dspark_fast_ceiling_noperf_074738.tsv` makes the line sharper:
  current `--draft-fast-relaxed` is clean at `54.52 t/s`; guard-off is
  `57.38 t/s` but repeats `self.enemy_direction = 1`; unbounded relaxed accept
  is `58.99 t/s` but repeats `# Background`; trust-confidence threshold `0.70`
  hits `72.19 t/s` but corrupts state into `1 = 1` garbage. The only current
  >60 path is therefore not coherent.
- Existing ANE shared-expert decode is rejected for the Flash DSpark fast path.
  `bench-results/dspark_ane_shared_probe_080227.tsv` measured control
  `54.37 t/s`, acceptance `93.3%`, canary `suspect=0`; enabling
  `DS4_FLASH_MOE_ANE_SHARED_EXPERT=1 DS4_SHARED_EXPERT_ANE_I8I8=1` fell to
  `14.03 t/s`, acceptance `15.7%`, canary `suspect=1`, and output collapsed
  into repeated `游戏` tokens. The log shows `ANE_SHARED_EXPERT=1` is the
  synchronous shared-expert path, not true DSpark draft overlap. Treat it as
  incompatible with this verifier stack unless separately reworked.
- Smaller resident slot banks are also rejected as a shortcut around memory
  pressure. `bench-results/dspark_slotbank_probe_080449.tsv` measured
  `--moe-slot-bank 128` at `17.71 t/s` and `--moe-slot-bank 64` at `18.84 t/s`
  on the same fast-relaxed n=500 canary. Both stayed simple-canary clean, but
  the SSD miss/read overhead dominates. Keep the 256-slot resident bank for
  DSpark speed work.
- Fresh current-tree route-overlap smoke keeps routed-MoE dedup on the serious
  track. `bench-results/route_overlap_current_080716` was simple-canary clean
  and, across 1419 layer-block rows, averaged `1.69x` route reuse
  (`p50=1.67x`, `max=3.00x`, average unique experts `18.55/30`, 1052/1419 rows
  at reuse >= `1.5x`). The opportunity is real; the rejected part is the
  existing grouped-IQ2 wrapper family, not the route-reuse premise.
- A low-register pair2 split of the grouped-IQ2 wrapper was tried and reverted.
  `bench-results/dspark_grouped_iq2_pair2split_081204.tsv` measured the
  current fast-relaxed control at `54.19 t/s`, acceptance `93.3%`, tau `4.48`,
  canary `suspect=0`; the opt-in grouped-IQ2 pair2-split path measured only
  `42.98 t/s`, acceptance `90.9%`, tau `4.32`, canary `suspect=0`. The grouped
  wrapper overhead is still too high even after splitting multiplicity-2 groups.
  Do not re-add this variant; the next MoE attempt needs a different design,
  not another split of the current descriptor/wrapper family.
- Guarded unbounded acceptance was also checked:
  `bench-results/fastrelaxed_unbounded_guarded_075554` kept loop guards active
  but fell to `41.43 t/s`, acceptance `71.4%`, and repeated
  `player = pygame.image.load('player.png')`. Force-accept is not salvageable
  just by preserving exact target state and applying the current loop guard.
- Acceptance-only and confidence-trust shortcuts have been rejected as the main
  >60 path. Fully unbounded relaxed accept reached only `58.88 t/s` and was
  canary-suspect. `DS4_DSPARK_TRUST_CONFIDENCE=1` can exceed 60
  (`67.61 t/s` at threshold `0.70`), but corrupts output/state. Adding the
  relaxed n-gram/token guard made conservative points cleaner but slow
  (`47.15/46.00 t/s` at thresholds `0.90/0.95`), while faster short-prefix
  variants still repeated `player = 0` hundreds of times. Treat trust-confidence
  as a diagnostic proving the target-state contract, not a usable mode. A later
  full-resync experiment was removed after
  `bench-results/dspark_trust_resync_unguarded_073725.tsv`: resync every 4
  blocks slowed to `38.84 t/s` and still stayed suspect.
- Current fast-relaxed plus `DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=3` did
  not move the ceiling. `bench-results/dspark_cap3_current_005946` measured
  `51.44 t/s` at n=1000, acceptance `91.7%`, full-accept `82.1%`, and simple
  canary `suspect=0`, effectively matching
  `bench-results/dspark_fastrelaxed_patched_n1000_003102`.
- The HTML Space Invaders 4000-token stress prompt is a hard canary:
  fast-relaxed `bench-results/si_html_010_fastrelaxed_003636` measured
  `37.15 t/s` and ended at an incomplete `let alive =`; strict
  `bench-results/si_html_010_strict_003940` measured `36.68 t/s` but also
  repeated one sprite row 73 times. Fresh retest
  `bench-results/si_html_010_fastrelaxed_retest_073924` measured `36.82 t/s`
  and looped on repeated `I'll use a 16x16 array` comments; no-draft
  `bench-results/si_html_010_nodraft_retest_074406` was also suspect at
  `32.42 t/s` with the same repeated sprite-row shape as strict. Treat simple
  Pygame canary success as insufficient for demo quality, and treat this HTML
  prompt as a base-model loop stress, not a clean relaxed-vs-strict
  discriminator.
- Draft-only remains rejected despite high speed. The old confidence-threshold
  artifacts hit `135-191 t/s`, but collapsed immediately into mixed math/code
  text. It is only a ceiling probe, not a candidate path.
- The current grouped-IQ2 routed-MoE consumer remains rejected even when only
  high-multiplicity route groups use it. In
  `bench-results/dspark_grouped_iq2_mincount_010345`, the current fast-relaxed
  baseline hit `54.75 t/s` at tau `4.48` and acceptance `95.8%`; grouped-IQ2
  min-count `2/3/4` measured `44.86/46.27/47.89 t/s` with the same acceptance.
  Do not pursue more threshold tuning on this wrapper family; the next MoE
  attempt must be a different lower-register route-dedup kernel.
  A fresh pair2-specialized IQ2 attempt was also rejected and removed:
  `bench-results/dspark_grouped_iq2_pair2_ab_071410.tsv` measured control
  `43.14 t/s`, generic grouped `42.63 t/s`, and pair2 `42.48 t/s`, all with
  identical acceptance. The lower-register specialization did not reduce the
  real gate/up wall.
- Current batch-canonical/Mode-B scaffolding is still not the fast path.
  `bench-results/modeb_nofrontier_timing_072519` forced
  `DS4_DSPARK_BATCH_CANONICAL=1` with frontier disabled and measured only
  `47.11 t/s`, acceptance `83.2%`, and batch verify around `77-79 ms/block`.
  Mixed frontier+batch timing (`bench-results/modeb_branch_timing_072421`)
  showed frontier strict-v1 blocks are still faster than the present batch
  backend. A real Mode B still means a new single-forward target-state contract,
  not the current `verify_suffix_tops` wrapper.
- Deferred attention heads remain rejected for this fast preset. Manual-env A/B
  in `bench-results/dspark_defer_heads_fast_ab_072849.tsv` measured control
  `54.36 t/s` canary-clean, row-exact deferred heads `50.77 t/s` canary-suspect,
  and unsafe deferred heads `51.13 t/s` canary-suspect.
- A temporary draft-layer cap probe was tested and removed. In
  `bench-results/dspark_draft_layers_sweep_011747`, the normal 3-layer draft
  measured `54.68 t/s`, draft `11.15 ms`, tau `4.48`, acceptance `95.8%`.
  Capping to 2 layers reduced draft to `8.62 ms` but collapsed to `26.04 t/s`,
  tau `0.74`, acceptance `27.1%`; capping to 1 layer reduced draft to `6.13 ms`
  but fell to `23.59 t/s`, tau `0.59`, acceptance `20.5%`. Do not optimize
  draft speed by dropping DSpark layers; the full 3-layer graph is
  quality-critical.
- Latest rerun of the exact HTML Space Invaders stress prompt with current
  fast-relaxed/Markov-argmax code is still a reject:
  `bench-results/si_html_010_fastrelaxed_argmax_012655` measured `37.33 t/s`,
  draft `11.15 ms`, verify `80.03 ms`, tau `3.74`, acceptance `78.8%`, and the
  output ended broken at `let alive =` with duplicate declarations. This
  confirms the fast-relaxed path remains too far out of alignment for exacting
  4000-token HTML/code generation.
- The new DSpark Markov argmax kernel is a safe cleanup, not a speed lever.
  `bench-results/dspark_markov_argmax_ab_012922` kept stdout byte-identical to
  the old Markov `topk(1)` path and reduced Markov timing only by about
  `0.1-0.2 ms`; generation stayed `54.78 -> 54.62 t/s` on the short Pygame
  canary. Keep it because it removes an oversized top-k call, but do not expect
  it to move the >60 target.
- Global `topk(1)` to argmax redirection was tested and removed. Strict n=300
  was unchanged (`44.10 -> 44.06 t/s`), and fast top1 stayed around `47.3 t/s`
  because tau collapsed to `3.30`. A separate top-k width sweep
  `bench-results/dspark_fast_topk_width_sweep_013708` showed top4 through
  top512 are effectively flat around `54.8-55.0 t/s` on the simple canary.
  Top-k width and top1 argmax are not the missing >60 lever.
- Relaxed temperature gating also does not recover the >60 path. The current
  public Ollama control surface exposes a fixed `draft_num_predict` cap for
  speculative draft tokens; it does not provide a documented adaptive policy we
  can mirror. Local matrix `bench-results/dspark_temp_gate_matrix_014716` tested
  the existing `DS4_DSPARK_RELAXED_TEMPERATURE` gate on the Pygame n=500 canary:
  base `--draft-fast-relaxed` was `51.39 t/s`, loose top128/delta6 was
  `51.76 t/s`, temp `0.8` with min-ratio `0.10` fell to `49.68 t/s`, and temp
  `1.0` with min-ratio `0.05` fell to `48.64 t/s`. All four were simple-canary
  clean, but the temperature gate lowers tau/acceptance instead of closing the
  speed gap. Do not spend the main loop on more relaxed temperature tuning
  unless a separate quality harness demands it.

Fresh continuation update: the current best simple-canary-clean fast base is now
static DSpark-5, not confidence-scheduler DSpark-5. In
`bench-results/dspark_static_budget_233127`, the loose frontier relaxed +
forced-MMA + fast-Q2 stack measured budget 3 `46.10 t/s`, budget 4 `49.36 t/s`,
and budget 5 `51.62 t/s` on the Pygame n=1000 canary. Budget 5 reported
`draft=11.21 ms`, `verify=68.12 ms`, `block=81.43 ms`, `tau=4.32`, and
acceptance `91.7%`. A same-stack slot-bank sweep
(`bench-results/dspark_slotbank_fast_232301`) rejected smaller banks: slot 32
`15.13 t/s`, slot 64 `17.84 t/s`, slot 128 `18.54 t/s`, slot 256 `48.28 t/s`.
The next real >60 path is therefore not slot-bank sizing or scheduler tuning:
hide/reduce the `~11 ms` draft wall, or build a new exact routed-MoE dedup kernel
that exploits the measured `~1.6x` route reuse without the current grouped-wrapper
overhead.

Follow-up checks: no-perf/no-backend-stats static DSpark-5 measured `51.50 t/s`
(`bench-results/dspark_static5_noperf_233632`), so instrumentation is not hiding
the gap. A wider relaxed gate sweep
(`bench-results/dspark_static5_gatewide_233907`) found one better short-run row:
top512/delta10 reached `53.24 t/s`, `tau=4.58`, acceptance `93.6%`, canary clean
at n=1000. The same shape on the longer Pygame cap
(`bench-results/dspark_static5_top512_delta10_n4000_234212`) stayed simple-canary
clean but fell to `48.91 t/s`; unbounded relaxed failed the canary. This confirms
relaxed-gate tuning is not enough for the >60 objective.

Fresh MoE subprofile: `DS4_DSPARK_ROW_ROUTED_SUBPROFILE=1` now fences and logs
gate/up, down, and sum stages even inside the verifier command batch. The direct
fast-Q2 static-5 diagnostic
`bench-results/dspark_row_routed_subprofile_000013_patched` emitted 1118 layer
rows: gate/up+SwiGLU mean `1.488 ms/layer`, Q2 down mean `0.382 ms/layer`,
ordered sum mean `0.001 ms/layer`. Existing grouped-IQ2 was worse in
`bench-results/dspark_row_routed_subprofile_grouped_iq2_000105`: gate/up mean
`1.925 ms/layer`, down `0.389 ms/layer`. A temporary all-grouped variant that
removed the singleton kernel also failed
(`bench-results/dspark_row_routed_subprofile_grouped_iq2_all_000500`, gate/up
`1.940 ms/layer`) and was reverted. A specialized multiplicity-2 grouped IQ2
prototype also failed
(`bench-results/dspark_row_routed_subprofile_grouped_iq2_pair2_001009`, gate/up
`2.048 ms/layer`) and was reverted. The actionable conclusion is that the
current grouped descriptor/IQ2 family is not the >60 path. Any MoE kernel work
should target a different route-dedup/shared-weight IQ2 gate/up+SwiGLU design,
not another Q2 down/sum toggle or small wrapper variation.

2026-07-01 fast-mode plumbing check: a first mode sweep accidentally used zsh
string variables for grouped environment assignments, so only the first variable
in each blob took effect and the runtime fell back to the default relaxed gate
(`top8/delta1`). The corrected sweep
`bench-results/dspark_mode_compare_fixedenv_002328` used literal env
assignments and `--draft-fast-relaxed`. Results at n=300: no-frontier fast
`49.97 t/s`, frontier fast `54.79 t/s`, batch fast `46.03 t/s`, unified fast
`46.05 t/s`; all simple canaries were clean. Because the frontier path is the
current faster demo path, `--draft-fast-relaxed` now defaults
`DS4_DSPARK_FRONTIER_DRAFT=1` in `ds4`, `ds4-agent`, and `ds4-server` while
preserving an explicit user environment override. Post-patch smoke
`bench-results/dspark_fastrelaxed_frontier_default_002752` confirmed plain
`--draft-fast-relaxed` reaches frontier-like behavior: `53.91 t/s`, `tau=4.48`,
acceptance `95.8%`, canary clean. This is useful progress, but it still does
not meet the >60 t/s goal; batch/unified remains diagnostic rather than the true
Mode-B single-forward verifier.

Later speed-only diagnostic work added `DS4_DSPARK_RELAXED_ACCEPT=1`. This is
deliberately non-target-greedy: it still verifies and commits target state for
the emitted draft token sequence, but it can accept target-supported non-argmax
tokens. With `DS4_DSPARK_FRONTIER_DRAFT=1`, it can also replace the current
target token. The current default is target-gated: relaxed accept requires the
draft token to be in the target top-8 and within logit delta `1.0` of the
target argmax. Tune with `DS4_DSPARK_RELAXED_TOPK=N` and
`DS4_DSPARK_RELAXED_LOGIT_DELTA=F`. The old ungated behavior is now explicit
speed-ceiling only: `DS4_DSPARK_FORCE_ACCEPT=1` or
`DS4_DSPARK_RELAXED_UNBOUNDED=1`. Historical ungated relaxed results:

- normal relaxed b5: `52.70 t/s` at n=1000
  (`bench-results/over50_relaxed_b5_n1000_061732`)
- frontier relaxed + forced MMA: `57.80 t/s` at n=1000
  (`bench-results/over60_frontier_relaxed_noperf_062523/mma.err`)
- frontier relaxed + forced MMA: `58.68 t/s` at n=300
  (`bench-results/over60_frontier_override_probe_062310/frontier.err`)
- layer-42 relaxed shortcut: `59.53 t/s` at n=300
  (`bench-results/over60_relaxed_layerlimit_063303/layers42.err`)

Do **not** promote the ungated form as the current best product mode. The output
tails from the ungated relaxed/frontier and layer-limit tests are visibly
degenerate (`# Background` loops, repeated `PLAYER_WIDTH`, repeated coordinate
fragments). Also, the reported `100%` acceptance in force-accept mode is forced
by the mode contract and is not evidence that the target model accepted the
drafts. Treat target-gated relaxed accept as an experimental non-byte mode until
it passes coherence checks on n=1000/4000.

Newest continuation result, `bench-results/dspark_partial_skip_row_routed_171444`:
partial verifier routed-MoE omission is also rejected. The current frontier
`top128/delta6`, margin-off, forced-MMA, fast-Q2 control was clean at
`49.70 t/s`, block `88.38 ms`, verify `73.98 ms`, tau `4.42`, acceptance
`92.9%`. Skipping verifier routed-MoE every 4th/3rd/2nd block reduced verify
to `44.77/45.45/34.47 ms`, but tau collapsed to `2.08/1.96/1.32`, generation
fell to `30.32/28.98/26.42 t/s`, and all outputs were canary-suspect. Do not
rerun approximate routed-MoE skip as a >60 candidate; it saves compute by
destroying the accepted stream.

Fresh current-tree sanity after removing the dead all-token confidence gate:
`bench-results/dspark_current_fast_sanity_173000` reran the same frontier
`top128/delta6`, margin-off, forced-MMA, fast-Q2 stack at n=500. It measured
`50.16 t/s`, block `87.58 ms`, draft `11.96 ms`, verify `73.32 ms`, tau
`4.42`, acceptance `92.9%`, full-accept `83.2%`, and the simple canary stayed
clean. This confirms the current fast ceiling is still around 50 t/s; reaching
60 t/s from this operating point needs roughly `14 ms/block` saved or hidden.

2026-06-30 continuation:

- Confidence-softmax cost-bias smoke
  (`bench-results/dspark_softmax_cost_smoke_205845`) shows the scheduler is not
  the missing >60 lever. With `MIN=4`, `FIXED_COST=20`, `TOKEN_COST=0.05`, and
  `MASS_WEIGHT=1.0`, softmax becomes effectively static-5: static `43.81 t/s`
  vs softmax `43.67 t/s`, both `tau=4.17` and `avg scheduled=4.98`.
- Pure trust-draft / draft-only is rejected as a product fast path. It is very
  fast but immediately incoherent: `bench-results/dspark_trust_draft_smoke_210027`
  measured `197.09 t/s`, but output collapsed into mixed math/Unicode text.
  Thresholds `0.95/0.98/0.995` in
  `bench-results/dspark_trust_draft_threshold_smoke_210138` still produced the
  same collapse despite `109-135 t/s`. The DSpark confidence head alone is not
  enough to safely skip target verification while using approximate main KV.
- Added opt-in relaxed safety knobs:
  `DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=N` and
  `DS4_DSPARK_RELAXED_OFFARGMAX_COOLDOWN_BLOCKS=N`. These cap how many
  non-argmax but target-supported tokens one block may accept, and can force a
  short target-argmax-only cooldown after any off-argmax accept.
  `bench-results/dspark_relaxed_offargmax_cap_n1000_210641` with loose
  frontier `top128/delta6`, forced-MMA, fast-Q2, and `MAX_OFFARGMAX=1`
  measured `48.25 t/s`, block `72.25 ms`, verify `59.06 ms`, tau `3.54`,
  acceptance `85.9%`, full-accept `72.4%`, and the simple canary stayed clean.
  This is useful as a safer relaxed diagnostic, but still below the >60 target.
- Loosening the same capped-relaxed mode by disabling the margin gate recovers
  some speed. `bench-results/dspark_relaxed_cap_marginoff_211017`, with
  `RELAXED_MARGIN_DISABLE=1`, `MAX_OFFARGMAX=1`, loose frontier `top128/delta6`,
  forced-MMA, and fast-Q2, measured `49.51 t/s`, `tau=4.10`, acceptance `88.5%`,
  simple canary clean. The cap sweep
  `bench-results/dspark_relaxed_cap_marginoff_grid_211122` with caps `2/3/4`
  measured `52.64-52.81 t/s` at n=600, all simple canary clean. A no-stats
  n=1000 check, `bench-results/dspark_relaxed_cap_no_stats_212046`, measured
  `53.06 t/s`, acceptance `94.0%`, but the canary table still reported duplicate
  declarations. Treat this as the current best reduced-quality CLI speed
  candidate, not a demo-safe or >60 solution.
- Dead follow-ups: `DS4_DSPARK_RELAXED_UNBOUNDED=1` plus the off-argmax cap was
  slower and visibly lower quality (`bench-results/dspark_relaxed_cap_unbounded_211336`,
  `47.72 t/s`, output contained `pygame.event()`, `KPACE`, wrong `bullet`
  member). Layer-limit with the cap is also rejected:
  `bench-results/dspark_relaxed_cap_layerlimit_211438` collapsed acceptance and
  output (`layer42` `25.53 t/s`, `layer40` `22.90 t/s`). Reducing
  `DS4_DSPARK_INDEXER_TOP_K_OVERRIDE` to `1024/512/256` did not move speed
  (`bench-results/dspark_relaxed_cap_index_topk_211635`, all `52.67-52.84 t/s`).
  Verify-4 no-stats also lost to verify-5 and was canary-suspect
  (`bench-results/dspark_relaxed_cap_v4_no_stats_212151`, `50.78 t/s`).
- Draft profile for the current capped fast stack:
  `bench-results/dspark_draft_profile_current_211857` shows warm draft cost is
  about `11.1 ms/block`, split roughly `9.0 ms` graph plus `2.1 ms` Markov.
  On a high-acceptance short n=220 run, generation reached `55.01 t/s` with
  `tau=4.62`, block `83.27 ms`. Hiding draft under verifier, for example via a
  real ANE draft-overlap track, remains the cleanest path from the current
  `~53-55 t/s` ceiling to >60.

MoE clarification: DSpark already has a fused row-exact Q2 routed-MoE down path
(`kernel_mul_mv_id_q2_K_sum6_ordered_f32`) and it is the useful default when
`DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1`. The rejected grouped-Q2 result
was not evidence that MoE is unfused; it tested route grouping beyond the
current direct ordered-Q2 fusion. The safe grouped descriptor wrapper stayed
`cmp=0` but slowed because it preserves row-exact math without sharing enough
weight work. The unsafe shared-weight grouped Q2 path changes the FP realization
and remains rejected. Any useful MoE win now requires a new shared-weight/dedup
kernel that preserves each row/slot dot order and the ordered slot sum.

Newest state-only confidence-gate result,
`bench-results/dspark_stateonly_conf_all_170554`: applying draft confidence to
all state-only accepted tokens did not salvage output. Thresholds
`0.4/0.65/0.8/0.9` topped out at `19.36 t/s`, all canary-suspect. The temporary
`DS4_DSPARK_RELAXED_CONFIDENCE_GATE_ALL` diagnostic was removed from the tree.

Follow-up top-k gated relaxed mode:

- Added `DS4_DSPARK_RELAXED_TOPK=N` / `DS4_DSPARK_ACCEPT_TOPK=N` as a diagnostic
  guard. It still changes the greedy contract, but a draft token is committed
  only if the target verifier ranks that token in the current target top-N.
- `bench-results/over60_relaxed_topk_sweep_064849`, frontier+MMA, n=300:
  top-1 `42.74 t/s`, top-2 `44.07`, top-4 `43.32`, top-8 `43.64`, top-16
  `45.19`, top-32 `43.75`, top-64 `46.03`, top-128 `46.28`. The unconstrained
  force-accept condition in the same sweep reached `55.67 t/s` but looped on
  `# Background`.
- Latest guarded-relaxed retest: relaxed mode now also applies a
  target logit-delta gate (`DS4_DSPARK_RELAXED_LOGIT_DELTA`; default now `1.0`).
  `bench-results/relaxed_gate_n1000_072929` showed no immediate repetition, but
  only a small speed gain: control `36.98 t/s`, `top32/delta2` `37.78 t/s`,
  `top128/delta4` `37.42 t/s`. `bench-results/relaxed_loose_gate_n300_073240`
  showed looser gates do not recover the old forced speed: `top256/delta6`
  matched the tighter gates, while `top512/delta8` and `top1024/delta10` slowed
  down. Normal guarded relaxed is safe enough for a reduced quality probe;
  frontier guarded remains slower in the quick collapse tests.
- Fresh frontier-specific collapse canary, aimed at the exact failure command
  shape (`DS4_DSPARK_FRONTIER_DRAFT=1` + relaxed accept):
  `bench-results/frontier_relaxed_collapse_smoke_075847` showed no immediate
  `# Background`/`display.set_mode` collapse for the current gated
  implementation. `top8/delta1`, `top16/delta1`, and `top32/delta2` landed near
  `45 t/s` at n=160.
- Promotion results: `bench-results/frontier_relaxed_promote_n300_080222`
  measured frontier `top8/delta1` at `45.52 t/s`, `tau=5.04`, `87.3%`
  acceptance, with no collapse markers. `top16/delta1` and `top32/delta2`
  matched it. The n=1000 check
  `bench-results/frontier_relaxed_top8_n1000_080452` measured `43.00 t/s`,
  `tau=4.79`, `85.2%` acceptance, and coherent inspected output. The repeated
  `return pygame` marker in that run is normal `return pygame.Rect(...)` class
  code, not the old collapse. `top4/delta0.5` is slower (`41.61 t/s` in
  `bench-results/frontier_relaxed_top4_n1000_080558`). The default relaxed gate
  has been tightened to the surviving `top8/delta1` setting. Treat it as the
  current frontier relaxed candidate for reduced quality testing, not as a >60
  solution. Post-tighten smoke
  `bench-results/frontier_relaxed_default_after_tighten_081040` confirms the
  default now logs `target top-8 + logit-delta<=1.00`, reaches `45.00 t/s` at
  n=160, and has zero collapse markers.
- `bench-results/over60_relaxed_topk_ops_065433` checked budget/confidence
  variants. Best was b5/top-64/confidence `0.40` at `46.59 t/s`; b4 variants
  stayed around `45.5-45.9 t/s`.
- 2026-06-30 compact suffix top-k/logit-gate patch:
  `row_topk` now has optional compact `row_topk_logits`, gathered on GPU and
  read back as only `top_rows * k` floats. This lets guarded relaxed suffix
  acceptance enforce both top-k and logit-delta without reading full vocab rows.
  Fresh artifacts: `bench-results/dspark_fast_guarded_compact`. Results:
  normal guarded relaxed n=300 `41.29 t/s`; normal n=160 sweep peaked at
  `42.47 t/s`; frontier guarded n=1000 `top8/delta1` was `43.51 t/s` with
  coherent inspected output; batch-canonical guarded n=300 was `42.33 t/s`;
  grouped exact routed-MoE n=160 was slightly slower than control (`40.03` vs
  `40.59 t/s`). The guard is now honest, but it does not restore >60.
- Current-build recheck of the older fastest coherent stack, forced MMA +
  confidence scheduler + fast-Q2 with no relaxed/frontier, measured only
  `42.50 t/s` in
  `bench-results/dspark_fast_guarded_compact/current_best_recheck`, versus the
  older `48.00 t/s` artifact at
  `bench-results/over50_fast_conf_fastq2_n1000_054201`. Acceptance is unchanged
  (`83.0%`), but verifier time regressed from `63.56` to `71.25 ms/block`.
  Compact suffix logits are inactive in that no-relaxed command, so isolate
  current-tree/run-state drift separately from relaxed-accept work.
- Interpretation: top-k gating restores basic coherence, but it removes the
  fake throughput. Relaxed top-k is useful evidence and a diagnostic guard, not
  an optimization route to >60.
- 2026-06-30 loop-guard update: a long `ds4-agent` Space Invaders HTML run with
  loose frontier `top128/delta6` relaxed accept started repeating
  `let invader = ...`, so that gate is not agent-safe by itself. Added a
  per-block relaxed accept loop guard: strict target-argmax tokens are still
  accepted, but non-argmax relaxed tokens are rejected if they would extend a
  recent repeated token n-gram or overuse one token in a short window. Default
  knobs are `DS4_DSPARK_RELAXED_LOOP_NGRAM=4`,
  `DS4_DSPARK_RELAXED_LOOP_WINDOW=256`,
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_WINDOW=192`, and
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=24`; disable only for speed ceilings with
  `DS4_DSPARK_RELAXED_LOOP_GUARD_DISABLE=1` or
  `DS4_DSPARK_RELAXED_TOKEN_GUARD_DISABLE=1`. Default-guard canary
  `bench-results/dspark_goal_continue_135310_loop_guard_default_html_n600`
  measured `44.17 t/s`, tau `4.12`, acceptance `88.7%`, and no simple collapse
  markers on the Space Invaders HTML prompt. Stricter `ngram=3`
  (`bench-results/dspark_goal_continue_134832_loop_guard_html_n600`) measured
  `42.22 t/s`. Guard-off can still show `~49 t/s` on short canaries, but treat
  it as unsafe for long agent runs until a real quality test clears it. First
  token-frequency canary
  `bench-results/dspark_goal_continue_140144_loop_token_guard_html_n400`, with
  `DS4_DSPARK_RELAXED_LOOP_NGRAM=3` and
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=12`, measured `45.37 t/s`, tau `4.26`,
  acceptance `89.7%`, and logged both n-gram and token-count guard rejections
  without simple collapse markers in a 400-token CLI smoke. This is not a
  substitute for a long `ds4-agent` quality pass.
- 2026-06-30 target-top loop-guard probe:
  `DS4_DSPARK_RELAXED_GUARD_TARGET_TOP=1` makes the relaxed loop guard check
  draft tokens even when they match the verifier's current target argmax. This
  catches the failure mode where an earlier relaxed token drifts the state into
  a repetitive pocket and later repeated tokens become target-top inside that
  drifted state. It is **opt-in only**, not a speed default. On the loose
  frontier `top128/delta6` Space Invaders HTML canary,
  `bench-results/dspark_relaxed_target_guard_155139` avoided the obvious
  duplicate-declaration collapse but fell to `23.64 t/s`, tau `2.22`, and
  `45.2%` acceptance. Use it to diagnose loop source, not as the path to >60.
- 2026-06-30 margin-aware relaxed gate probe:
  added opt-in `DS4_DSPARK_RELAXED_LOGIT_RATIO=F`
  (`DS4_DSPARK_ACCEPT_LOGIT_RATIO=F`) for non-argmax relaxed accepts. This
  requires both `top2/top1` and `draft/top1` to exceed `F`, matching the
  scale-aware direction from margin-aware speculative verification literature.
  To test the ratio as an alternative gate, set
  `DS4_DSPARK_RELAXED_TARGET_MARGIN=-1` and
  `DS4_DSPARK_RELAXED_DRAFT_MARGIN=-1`. Mini-sweep on the loose frontier
  `top128/delta6` Space Invaders HTML canary:
  `bench-results/dspark_relaxed_ratio_sweep_155843` gave `41.53 t/s`
  (`F=0.970`), `38.00 t/s` (`0.985`), `36.51 t/s` (`0.995`), all offline
  canary-clean. A looser `F=0.900` run
  `bench-results/dspark_relaxed_ratio_0900_160119` reached `44.84 t/s`,
  `83.7%` acceptance, tau `3.87`, and canary-clean output. This is useful as
  a quality/coherence knob, but it still does not recover the >60 speed target.
- The former fast-demo gate, `DS4_DSPARK_RELAXED_TOPK=16` with
  `DS4_DSPARK_RELAXED_LOGIT_DELTA=-1`, plus
  `DS4_DSPARK_RELAXED_LOOP_NGRAM=3` and
  `DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=12`, is now rejected as a demo path. On the
  500-token Space Invaders HTML canary,
  `bench-results/dspark_goal_continue_142250_top16_nodelta_conf_loop_n500`
  measured `45.14 t/s`, tau `4.40`, acceptance `88.8%`, and no simple collapse
  markers, but a real `ds4-agent` HTML/game retest produced duplicate
  declarations and repeated constants. Treat this as proof that loop/frequency
  guards are not enough.
- Recoverable relaxed cooldown probe: `DS4_DSPARK_RELAXED_COOLDOWN_BLOCKS=N`
  now defaults to `4` (`DS4_DSPARK_RELAXED_COOLDOWN_DISABLE=1` restores the old
  behavior). When a loop-guard rejection fires, the next `N` speculation blocks
  still allow target-argmax DSpark accepts but reject off-argmax relaxed accepts,
  so fast mode can recover instead of staying permanently strict. HTML canary
  `bench-results/dspark_relaxed_cooldown_html_174213` with loose
  `top128/delta6`, forced-MMA, fast-Q2 measured `46.20 t/s`, tau `3.87`,
  acceptance `86.6%`, and logged three loop-guard rejections. It avoided the
  obvious duplicate-declaration collapse, but the offline canary still reported
  `suspect=1` from malformed CSS-like lines. Treat this as quality plumbing,
  not a >60 speed path.
- Existing non-frontier 1+5 contract check:
  `bench-results/dspark_nonfrontier_1plus5_fast_174742` disabled
  `DS4_DSPARK_FRONTIER_DRAFT` so the target first token is evaluated normally,
  then DSpark drafts/verifies up to five suffix tokens. With the same loose
  relaxed/MMA/fast-Q2 stack it measured `44.74 t/s`, tau `3.91`, acceptance
  `83.1%`, and the offline canary returned `suspect=0`. This confirms that the
  first-token decode cost eats the extra nominal emitted-token budget; a useful
  bonus-token path needs a real committable N=6/batch-canonical verifier, not the
  existing 1+5 fallback shape.
- Current fast no-perf ceiling check:
  `bench-results/dspark_current_fast_noperf_175020` reran the loose
  frontier relaxed/MMA/fast-Q2 stack without `DS4_DSPARK_PERF=1`. It measured
  `50.20 t/s` at n=1000, acceptance `94.7%`, full-accept `85.6%`, first-miss
  `0.0%`, and canary `suspect=0`. This proves the remaining >60 gap is not
  timing/GPU-busy instrumentation overhead; it is block-wall cost.
- State-only target-prefix sweep:
  `bench-results/dspark_stateonly_prefix_sweep_175409` tested
  `DS4_DSPARK_STATE_ONLY_VERIFY=1`,
  `DS4_DSPARK_STATE_ONLY_TARGET_HEAD=1`,
  `DS4_DSPARK_STATE_ONLY_TARGET_HEAD_LAST_ONLY=1`, and
  `DS4_DSPARK_STATE_ONLY_TARGET_PREFIX_ROWS={1,2,3,4}`. Prefix 1/2 stayed
  faster-ish (`52.38` / `49.34 t/s`) but were canary-suspect. Prefix 3/4 were
  canary-clean but only `44.30` / `49.23 t/s`. This gives no >60 middle ground.
- State-only prefix4 plus routed-MoE skip:
  `bench-results/dspark_stateonly_prefix4_skiprouted_175658` tested periodic
  `DS4_DSPARK_SKIP_ROW_ROUTED_BLOCK_EVERY={2,3}` on top of the prefix4 state-only
  target-head mode. Both were hard rejects: every-2 `18.02 t/s`, acceptance
  `29.3%`, canary `suspect=1`; every-3 `24.61 t/s`, acceptance `47.0%`,
  canary `suspect=1`. Skipping routed MoE damages the verifier state enough to
  reduce speed, not improve it.
- Relaxed accept now has a target-margin gate by default: non-argmax accepts
  require `top1-top2 <= DS4_DSPARK_RELAXED_TARGET_MARGIN` and
  `top1-draft <= DS4_DSPARK_RELAXED_DRAFT_MARGIN` in addition to top-k/delta and
  confidence gates. Defaults are `0.35` / `0.35`. Use
  `DS4_DSPARK_RELAXED_MARGIN_DISABLE=1` only to reproduce the older broken
  speed-ceiling behavior. Frontier relaxed mode still uses
  `--draft-scheduler confidence` as a confidence gate for non-argmax accepts;
  set `DS4_DSPARK_RELAXED_CONFIDENCE_GATE_DISABLE=1` only for A/B timing.
- Fresh margin-gate canary after this change:
  `bench-results/dspark_margin_gate_html_n500_144215`, Space Invaders HTML,
  `top16`, no logit-delta, confidence `0.4`, margins `0.35/0.35`, measured only
  `34.36 t/s`, tau `2.52`, acceptance `68.4%`. It looked sane in the first
  500 tokens but is far too conservative for the speed target.
- Short margin sweep:
  `bench-results/dspark_margin_gate_sweep_144358`, same prompt, n=400.
  Margins `0.75/0.75`: `33.82 t/s`, tau `2.36`, acceptance `67.5%`.
  Margins `1.25/1.25`: `38.35 t/s`, tau `3.24`, acceptance `77.3%`.
  Margins `2.00/2.00`: `41.86 t/s`, tau `4.04`, acceptance `82.6%`.
  Margins `4.00/4.00`: `42.61 t/s`, tau `4.16`, acceptance `82.8%`.
  No simple duplicate-declaration markers appeared in these 400-token CLI
  canaries, but the family is still well below the >60 objective. Do not spend
  the main loop on more top-k/margin tuning unless a stronger quality harness
  says it can replace strict mode.
- Fresh state-only target-head check:
  `bench-results/dspark_state_only_target_head_margin_n400_144704`, with
  state-only target-head plus the `2.0/2.0` margin gate, measured `41.82 t/s`,
  tau `4.04`, acceptance `82.6%`, essentially the same as the normal margin
  run. It is not a new speed lever.
- 2026-06-30 target-head last-row diagnostic:
  `DS4_DSPARK_STATE_ONLY_TARGET_HEAD_LAST_ONLY=1` makes state-only target-head
  run the target LM head only for the final row instead of all verifier rows.
  In `bench-results/dspark_state_target_lastrow_161152`, the control measured
  `43.52 t/s`; last-row mode measured `54.08 t/s`, tau `4.99`, and `99.0%`
  acceptance. The offline canary flagged the output and manual inspection showed
  immediate collapse (`pygame.display = pygame.display.set_mode...`). Keep this
  as evidence that all-row target-head cost was real, but do not promote it: the
  forced acceptance of wrong intra-block draft tokens is still fatal.
- Current three-way remeasure:
  `bench-results/dspark_mode_compare_current_145238`, Pygame Space Invaders,
  n=400. Strict default measured `37.57 t/s`, verify `85.51 ms`, tau `3.94`,
  acceptance `78.8%`. Forced-MMA measured `40.35 t/s`, verify `71.00 ms`,
  tau `3.71`, acceptance `74.1%`. Current batch-canonical/Mode-B env
  (`DS4_DSPARK_VERIFY_CANONICAL=batch`) measured only `37.79 t/s`, verify
  `75.52 ms`, tau `3.60`, acceptance `72.6%`. Treat current Mode-B plumbing as
  not yet the >60 route; it needs a different committed-state contract, not just
  enabling the existing flag.
- 2026-06-30 batch-canonical force-accept checks:
  `bench-results/dspark_batch_canonical_force_160609`, with
  `DS4_DSPARK_VERIFY_CANONICAL=batch DS4_DSPARK_FORCE_ACCEPT=1`, measured only
  `41.36 t/s`, tau `4.29`, and produced duplicated/malformed CSS blocks.
  Adding `DS4_DSPARK_FRONTIER_DRAFT=1` in
  `bench-results/dspark_frontier_batch_force_160747` was worse: `35.53 t/s`,
  tau `3.40`, malformed CSS (`#1a1a2`, incomplete `</style`). This confirms the
  current batch-canonical flag is not the reference-style single-forward Mode B
  path and should not be used for demos.
- Exact correction-token scheduling was tested and rejected:
  `bench-results/dspark_correction_ab_150049`. An opt-in local patch evaluated
  the target correction token immediately on draft miss. Output matched strict
  (`cmp=0`), but generation dropped from strict `37.51 t/s` to `35.92 t/s`;
  commit/overhead rose to `14.27 ms/block`. The patch was removed. Do not
  reintroduce exact correction-token emission unless the first-token loop/API is
  redesigned so state advance is amortized.

The runtime now prints conditional acceptance in `ds4`, `ds4-server`, and
`ds4-agent`, e.g. strict b5:
`1=87.9% 2|1=96.7% 3|1-2=92.0% 4|1-3=92.5% 5|1-4=85.9%`.
This makes tau visibly healthy at pos1/pos2; the remaining gap is late suffix,
full-block rate, verifier wall time, or the speculation contract itself.

Latest Pro-agent adjustment: Case E top-k sparse is useful to pursue only as a
long-context non-byte-identical fast-mode experiment. It should not displace the
strict `cmp=0` plan. For strict-v1, the next large levers are exact MoE
route-dedup/grouped routed work, tau/full-accept recovery, and a separate
draft-overlap feasibility audit. For a real ~2x result, the route remains Mode B
/ unified batch-canonical verification, not strict-v1 attention microkernels.

Contract split after this feedback:

- **Strict / byte-identical:** keep strict-v1 as the production-safe path. Do
  not chase more attention fusion or dense/NAX polish unless a non-fenced run
  improves generation t/s without hurting tau. The next kernel work is exact
  routed-MoE dedup/reuse, gated by slot-down and MoE-boundary `max=0`.
- **Fast / non-identical:** if the near-term goal is >60 t/s more than old
  row-decode byte identity, fix the real Mode B / unified batch-canonical path.
  Case E top-k sparse belongs here as a long-context component, not as the main
  short-context verifier lever.
- **Scheduling:** ANE draft overlap is separate from verifier optimization.
  First measure whether optimistic full-accept pre-draft work can be reused
  often enough to beat wasted work on partial rejects.

Immediate queue after the latest Pro feedback:

1. Measure tau shape on every benchmark: pos-wise acceptance, conditional
   acceptance, full-accept rate, tau, draft ms, verify ms, and generation t/s.
   Do not rank by verify ms alone; a lower-ms path that drops acceptance is not
   a win.
2. Strict `cmp=0` work starts with MoE route-dedup. Preserve row-exact router
   logits/top-k/weights, per-row/per-slot down outputs or a proven exact
   equivalent, and ordered FP32 slot0..slot5 accumulation. Gate with slot-down
   max-delta `0`, MoE-boundary max-delta `0`, then `cmp=0` at n=160/1000/4000.
3. Case E continues only as a selected top-k long-context fast-mode experiment.
   Validate at `-c 4096`, `16384`, `65536`, and `100096`, budgets 4 and 5, and
   keep opt-in unless `cmp=0` is proven.
4. ANE draft overlap is a separate scheduling/dependency audit. Count only the
   draft work that can actually be reused after verify commits; partial rejects
   create wasted work.
5. True ~2x requires Mode B / unified batch-canonical verification. The current
   `--draft-mode batch|unified` flags are diagnostic and slower than strict; do
   not treat them as the real Mode B implementation.

Fresh current-best recheck after the stats fix:
`bench-results/over50_post_stats_005658` measured forced-MMA b5 at
`44.58 t/s`, draft `15.76 ms`, verify `62.86 ms`, tau `4.78`, acceptance
`75.6%`, and conditional acceptance
`1=86.1% 2|1=96.1% 3|1-2=93.1% 4|1-3=90.1% 5|1-4=90.3%`.

New current-best after draft hot-spot fix:
the DSpark draft `attn_output_a` group projection now uses a grouped-strided FP8
rows5 kernel, reducing the old token-by-group loop to one rows5 dispatch across
all output groups. Disable with
`DS4_DSPARK_DRAFT_ATTN_OUT_A_STRIDED_DISABLE=1`. A/B evidence:
`bench-results/over50_draft_grouped_strided_ab_022803` (b5/n160) matched the
old draft path output (`cmp=0`) and improved draft from `15.53` to
`11.04 ms/block`. `bench-results/over50_draft_grouped_strided_n1000_022916`
matched output (`cmp=0`) and measured new `46.79 t/s` versus old `44.73 t/s`,
draft `11.17` versus `15.68 ms/block`, same tau `4.78`, same acceptance
`75.6%`, and essentially unchanged verify (`62.43` versus `62.59 ms`). Budget
sweep `bench-results/over50_grouped_strided_budget_sweep_023120` kept b5 as the
best operating point: b3 `42.45 t/s`, b4 `45.38 t/s`, b5 `46.31 t/s`. This is
a real stackable win, but the remaining >50 gap is now verifier/overlap/Mode-B,
not a single obvious draft microkernel.

Post-feedback probes on the same branch:

- Explicit batch verifier gate fix:
  `DS4_DSPARK_BATCH_VERIFY=1` now bypasses the strict-v1 branch and reaches the
  batch verifier. Before this fix, the flag measured strict-v1 by accident.
  Actual n=300 batch measurements
  (`bench-results/over50_batch_approx_fixed_041115`) are slower, not faster:
  plain batch `24.03 t/s` with `108.34 ms/block` overhead/commit, and approximate
  batch variants `37.22-37.92 t/s` with about `23 ms/block` overhead/commit.
  This is a useful diagnostic fix, but it does not move the >50 path.
- Batch approximate prefix-commit repair:
  `DS4_DSPARK_BATCH_VERIFY=1 DS4_DSPARK_BATCH_APPROX_STATE=1` now captures
  prefix states and commits accepted partial prefixes instead of exact replaying
  them. Disable with `DS4_DSPARK_BATCH_APPROX_PREFIX_COMMIT_DISABLE=1` for the
  old replay diagnostic. The repair removes the measured replay/commit
  pathology but still does not beat the normal fast path. n=300
  `bench-results/over50_batch_prefix_042805`: old replay batch approximate
  `37.90 t/s`, block `100.99 ms`, overhead `22.78 ms`; repaired prefix commit
  `45.71 t/s`, block `82.39 ms`, overhead `1.95 ms`; normal forced-MMA
  `46.62 t/s`. n=1000 `bench-results/over50_batch_prefix_n1000_043501`:
  normal forced-MMA `46.58 t/s`, repaired batch approximate `45.02 t/s`.
  Variant sweep `bench-results/over50_batch_prefix_variants_044412` topped at
  `45.98 t/s` with `DS4_DSPARK_BATCH_DECODE_ORDER=1`. Treat this as diagnostic
  hygiene, not the >50 route.
- Additional cheap-lever sweeps:
  confidence scheduling peaked at threshold `0.4`, `47.79 t/s` on n=300
  (`bench-results/over50_conf_sched_040244`), but did not hold a >50 speed.
  Frontier plus confidence stayed `47.16-47.71 t/s`
  (`bench-results/over50_frontier_conf_040521`). Context `-c
  4096/8192/16384` stayed `45.94-46.97 t/s`
  (`bench-results/over50_ctx_sweep_041344`). Disabling prefixN capture regressed
  to `39.34 t/s` because partial replay adds about `22 ms/block`
  (`bench-results/over50_prefix_disable_041729`), so prefix capture remains on.
  Resident slot-bank 64/128 failed allocation, resident 256 stayed `46.76 t/s`,
  and non-resident direct-mmap slot-bank was only `14.20-21.34 t/s`
  (`bench-results/over50_slotbank_sweep_041618`,
  `bench-results/over50_stream_slotbank_041913`).
- Current-turn cheap-lever rechecks:
  confidence scheduling must be enabled with `--draft-scheduler confidence`, not
  just `--draft-conf-threshold`. The real n=1000 threshold `0.4` run
  (`bench-results/over50_conf_scheduler_n1000_044255`) measured `46.89 t/s`,
  avg scheduled `4.70`, tau `4.84`, so it did not cross 50. Lowering target
  expert top-k with `DS4_FLASH_MOE_EXPERT_TOPK=5` failed at DSpark draft warmup
  (`bench-results/over50_topk_sweep_043303`) because this runtime/draft path
  assumes the model's full six active experts. Route-overlap sizing on the
  current forced-MMA b5 path (`bench-results/over50_route_overlap_043125`)
  confirms the strict MoE route-dedup opportunity: active-5 blocks showed
  average reuse around `1.7-1.8x`, ranging roughly `1.47x-2.01x` in the smoke.
- Confidence fast-Markov patch:
  confidence scheduling now keeps the fast Markov chain and saves each row's
  Markov embedding for the existing CPU confidence dot. Disable with
  `DS4_DSPARK_CONF_FAST_MARKOV_DISABLE=1`. A/B
  `bench-results/over50_conf_fast_markov_ab_053746` stayed output-identical
  (`cmp=0`) and improved n=300 from `47.82` to `48.90 t/s`, draft
  `12.13 -> 11.18 ms`. n=1000 confirmation
  `bench-results/over50_conf_fast_markov_n1000_053914` measured `47.49 t/s`,
  draft `11.24 ms`, verify `61.77 ms`, tau `4.84`, full accept `69.9%`.
  Stacking the existing non-byte-identical fast-Q2 routed path gave the current
  best observed n=1000 run:
  `bench-results/over50_fast_conf_fastq2_n1000_054201`, `48.00 t/s`, tau
  `4.97`, acceptance `83.0%`, full accept `71.6%`. The follow-up threshold
  sweep `bench-results/over50_fastq2_conf_threshold_sweep_054318` did not cross
  50. Final current-binary recheck after bulk confidence cleanup
  `bench-results/over50_current_best_n1000_055324` measured `47.97 t/s` with
  the same tau/acceptance, so the cheap scheduler/flag stack is exhausted.
- Correction-token audit:
  the verifier often knows the target correction token at the first mismatch,
  but DS4's generation loop returns only tokens whose target state has already
  been evaluated and committed. Emitting the correction token without evaluating
  it would leave the session on stale KV/logits; evaluating it exactly inside
  `ds4_session_eval_speculative_argmax` is the same work the next outer decode
  step already performs. Treat correction/bonus-token recovery as a Mode-B or
  session-contract project, not a free strict-v1 tau fix.
- Latest confidence/MMA confirmation:
  `bench-results/over50_combo_probe_050258` found a short n=300 peak of
  `48.42 t/s` for forced-MMA b5 plus the real confidence scheduler
  (`--draft-scheduler confidence --draft-conf-threshold 0.4`), with tau `5.17`
  and full accept `74.1%`. Frontier-after-full did not improve it (`48.10 t/s`),
  and unordered Q2 fell to `47.75 t/s`. Longer n=1000 confirmation
  `bench-results/over50_conf_mma_n1000_confirm_050540` measured `46.88 t/s`,
  tau `4.84`, full accept `69.9%`, so the cheap-stack path is still below the
  >50 goal.
- Corrected batch-approx + confidence + MMA check:
  `bench-results/over50_batch_conf_mma_probe_051941` used
  `DS4_DSPARK_BATCH_VERIFY=1 DS4_DSPARK_BATCH_APPROX_STATE=1
  DS4_DSPARK_ATTN_FORCE_MMA=1 --draft-scheduler confidence
  --draft-conf-threshold 0.4`. It did not recover speed: strict confidence+MMA
  was `48.48 t/s` on n=300, while batch-approx confidence+MMA was `44.83 t/s`
  and decode-order batch-approx confidence+MMA was `44.87 t/s`. The batch path
  lost tau (`4.76` vs `5.17`) despite similar verify wall, so current
  batch-approx variants are not the >50 path.
- Pair2 grouped-Q2 route-dedup diagnostic:
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_PAIR2=1` adds a narrow
  multiplicity-2 shared-weight grouped Q2 down kernel and falls back to the exact
  descriptor wrapper for larger duplicate groups. It was useful as a proof:
  `bench-results/over50_pair2_q2_smoke_051427` saw grouped-Q2 compare
  `exact=yes max=0` for active-5 and active-2 calls. It was not useful for speed:
  `bench-results/over50_pair2_q2_speed_051621` measured base forced-MMA
  `46.72 t/s`, pair2 `44.69 t/s`, and pair2+confidence `46.39 t/s`. Keep it
  opt-in diagnostic only; exact shared-weight Q2 for duplicate pairs does not
  beat the default direct ordered-Q2 path on this workload.
- Case E selected top-k NAX prototype, b5/n1000:
  `bench-results/nax_topk128_b5_n1000_011632` measured `39.19 t/s`, verify
  `74.44 ms`, tau `4.69`, acceptance `73.9%`. This first implementation is a
  short-context regression, not a speed path. Keep it diagnostic and only revisit
  the selected-key idea for long-context cases where dense compressed attention
  exceeds strict sparse top-k work.
- Strict `DS4_DSPARK_INDEXER_TOP_K_OVERRIDE=128`, b5/n1000:
  `bench-results/topk128_strict_b5_n1000_011742` measured `39.87 t/s`, verify
  `76.44 ms`, tau `4.85`, acceptance `77.2%`. Top-k reduction alone did not move
  the strict path.
- `DS4_DSPARK_FRONTIER_DRAFT=1 DS4_DSPARK_ATTN_FORCE_MMA=1`, b5/n1000:
  `bench-results/frontier_mma_b5_n1000_012429` measured `38.63 t/s`. Pos1 rose
  to `97.4%`, but overhead/commit rose to `14.61/14.26 ms`, so this is a failed
  diagnostic until full-accept and partial-replay timing are split and commit
  overhead is fixed.
- Frontier prefix commit has now been repaired. Partial frontier accepts use the
  verifier-captured prefix frontier when available, instead of exact target
  replay. Evidence: `bench-results/frontier_prefix_mma_b5_n160_013356` improved
  the n=160 frontier+MMA path to `44.21 t/s` with overhead/commit back to
  `1.49/1.18 ms`; `bench-results/frontier_prefix_mma_b5_n1000_013441` measured
  `44.64 t/s`, block `76.25 ms`, draft `15.50 ms`, verify `59.07 ms`, tau
  `4.44`, acceptance `81.9%`. Budget 4
  (`bench-results/frontier_prefix_mma_b4_n1000_013548`) was `44.40 t/s`. This
  removes the replay bug, but it is still not a >50 path because frontier draft
  loses tau versus the normal post-token draft.
- `DS4_DSPARK_FRONTIER_AFTER_FULL=1` is now an opt-in frontier diagnostic that
  attempts frontier drafting only after the previous DSpark block fully
  accepted. Fresh b5/n1000 A/B
  (`bench-results/over50_frontier_after_full_031758`) measured base forced-MMA
  `46.55 t/s`, plain frontier `46.92 t/s`, and gated frontier `46.83 t/s`.
  This confirms the waste hypothesis but does not create a >50 path; keep it as
  a diagnostic, not as the main speed plan.
- Follow-up frontier sweep (`bench-results/over50_frontier_after_hit_034429`)
  measured plain frontier `47.01 t/s`, after-full `47.39 t/s`, temporary
  after-first-hit `47.32 t/s`, and after-first-hit + fast-Q2 `46.79 t/s`.
  After-first-hit was removed after the test because it did not beat after-full.
  Frontier b4/b5 remained below 50 at n=1000.

Discarded predraft diagnostic:
`DS4_DSPARK_PREDRAFT=1` in a local throwaway branch
(`bench-results/over50_predraft_005549`) regressed to `33.40 t/s`, acceptance
`62.0%`, and `~25 ms/block` overhead/commit. The runtime flag was not kept; do
not use this as a speed path without a deeper state dependency redesign.

Latest MoE sizing probe:
`DS4_DSPARK_SKIP_ROW_ROUTED=1 DS4_DSPARK_ATTN_FORCE_MMA=1`, b5/n300
(`bench-results/over50_skip_routed_015744`) is timing-only and output-invalid.
It dropped verify to `19.44 ms/block` and block wall to `35.90 ms`, but
acceptance collapsed to `8.9%`, tau to `1.44`, and generation to `22.95 t/s`.
Conclusion: routed MoE is a large remaining verifier bucket, but skipping or
zeroing it destroys draft acceptance. The useful strict path is exact route
dedup/reuse: keep row-exact router/top-k/weights, preserve per-row/per-slot down
outputs, and keep the ordered FP32 slot sum contract. Gate any MoE fusion with
slot-down max-delta `0`, MoE-boundary max-delta `0`, then `cmp=0` at
n=160/1000/4000.

Existing grouped MoE flags were rechecked on the current branch after the draft
hot-spot fix (`bench-results/over50_moe_existing_flags_032204`, b5/n300,
forced-MMA). Base measured `46.72 t/s`; grouped-Q2 safe measured `44.79 t/s`;
grouped IQ2 gate/up measured `40.35 t/s`; grouped IQ2 + grouped-Q2 safe
measured `39.21 t/s`. Acceptance was unchanged, so this is pure verifier cost.
Do not spend more time promoting the existing grouped descriptors; the next MoE
attempt needs a new byte-safe route-dedup kernel or lower-overhead expert-row
kernel that preserves strict row/slot outputs and ordered sum.

Additional current-branch negative probes:

- Slotwise row-routed MoE (`bench-results/over50_slotwise_moe_034048`) regressed:
  base `47.02 t/s`, slotwise `41.85 t/s`, slotwise+fast-Q2 `42.75 t/s`, and
  no batch-row-exact `43.67 t/s`.
- Extra dense rows knobs (`bench-results/over50_dense_rows_flags_033309`)
  regressed: base `46.79 t/s`, F16 rows5 `44.47 t/s`, F16 seq rows5
  `44.52 t/s`, output-low Q8 rows5 `46.41 t/s`, combined `45.79 t/s`.
- A temporary draft-layer-limit diagnostic
  (`bench-results/over50_draft_layer_limit_033748`) proved all three DSpark
  draft layers are needed: two layers dropped acceptance to `28.0%` and
  generation to `29.87 t/s`; one layer dropped acceptance to `15.6%` and
  generation to `25.34 t/s`. The runtime knob was removed after measurement.

Partial target-layer sweep:
`DS4_DSPARK_ATTN_FORCE_MMA=1 DS4_DSPARK_DECODE2_LAYER_LIMIT={43,42,40,36,32,28}`,
b5/n300 (`bench-results/over50_layerlimit_020451`) rejects the cheap
partial-layer verifier idea. Full 43 layers measured `44.86 t/s`, verify
`63.94 ms`, tau `4.84`, acceptance `77.0%`. At 42 layers verify fell to
`46.40 ms`, but tau fell to `2.63`, acceptance to `32.9%`, and generation to
`29.18 t/s`; lower limits were `23-28 t/s` with `14.9-24.3%` acceptance.
Conclusion: a usable target verifier still needs the full target stack unless a
new Mode-B/unified contract defines a different canonical path.

Active recommendation after the latest Pro review:

0. Keep measuring pos-wise/conditional acceptance, full-accept rate, tau, draft
   ms, verify ms, and generation t/s. Do not rank by verifier-ms alone; lower
   verify-ms can lose if tau or acceptance drops.
1. Keep strict cmp=0 as the safe default. The next strict speed target is exact
   MoE route-dedup/grouped routed work with ordered FP32 slot sum, not more
   attention fusion. Preserve row-exact router/top-k/weights, per-row/per-slot
   semantics, and ordered slot0..slot5 accumulation; do not revive the rejected
   grouped shared-weight Q2 path or unordered `sum6`.
2. Keep Case E top-k sparse narrowed to the long-context fast-mode question. It
   is the highest-value Case E-specific patch, but it is not the main strict
   verifier track. The first selected-stream prototype regressed at c4096/n1000,
   so the next attempt must be measured at `-c 4096`, `16384`, `65536`, and
   `100096`, with budgets 4 and 5, past the dense-vs-sparse crossover, and
   ranked by generation t/s, tau, and acceptance, not just verify ms. Its purpose
   is bounded long-context opt-in behavior, not a replacement for the strict
   verifier.
3. Keep ANE draft overlap as a separate feasibility track. First audit whether
   draft(k+1) can start before verify(k) commits; expected gain depends on
   full-accept rate and wasted work on partial rejects. Size it as
   `hidden_draft_ms = correct_path_reuse_rate * draft_ms - wasted_work_cost`, not
   as an automatic full-draft hide.
   Current sizing is favorable enough to justify the audit: forced-MMA b5 is
   `44.6 t/s` with about `15.5 ms/block` of draft time. Hiding most draft work
   on ANE under the GPU verifier is the cleanest measured route into the low
   `50s`, assuming transfer/scheduling overhead and wasted speculative draft
   work stay small.
4. Demote dense projection row-wide/NAX work unless a non-fenced microbench
   proves a real generation-tps win.
5. If >50 t/s or true ~2x is the immediate goal, fix/implement real Mode B /
   unified batch-canonical target forward. The current batch/unified flag is
   diagnostic and slower than strict.

Plain next-step split:

- Strict/cmp=0: measure acceptance on every run, then work MoE route-dedup and
  tau/full-accept analysis.
- Case E: only continue selected-key/top-k work as an opt-in long-context path.
- ANE: separate dependency/scheduling audit before counting speedup.
- True 2x: Mode B/unified contract, not strict-v1 micro-optimization.

Current status:

- Latest Plan C checkpoint from reviewer feedback and local profiler: Plan C is
  the lead research track now, not a side curiosity. The row-shape profiler found
  `raw_same_count=0/43` with active-5, so a same-mask/same-span rows5 attention
  shortcut is dead. The useful signal is the prefix reuse estimate instead:
  first-block raw scan reuse around `3.72x` and compressed scan reuse around
  `3.85x`; the fuller n=320 rows-exact aggregate estimates `4.46x` raw reuse
  and `4.79x` compressed reuse on the remaining mixed/compressed path.
  The raw vec-rows landing point is now byte-clean but not materially faster.
  The follow-up rows-exact profile found the real remaining attention work is
  mixed/compressed (`1957` calls, `9705` mixed rows at n=320), and a strict
  same-shape mixed shortcut has zero coverage (`same_shape=0`). The next useful
  kernel is therefore a **true shared-prefix mixed/compressed attention** path:
  scan committed raw/compressed K/V once per tile for N<=5 rows, keep one
  online-softmax state per row, then append the tiny row-visible block tail in
  exact decode order.
- MLX reference is now local at `/tmp/dspark-research/mlx-vlm` (commit
  `78b96eb`). Its useful lesson is contractual, not drop-in code: MTP verification
  concatenates bonus+draft tokens and runs one target forward, then rolls back the
  rejected suffix. Its DeepSeek-V4 model exposes the same speculative verify
  shape (`speculative_verify_hidden/logits`) plus rollback. This reinforces Plan C:
  make DS4 no-draft N=1 and DSpark verifier N<=5 use the same target-forward
  family instead of trying to make unrelated batch kernels bit-match row decode.
  Concrete reference points:
  `mlx_vlm/speculative/mtp.py::_mtp_rounds()` builds
  `verify_input = [bonus, draft_tokens]`,
  `_mtp_verify_target()` calls the model-specific verifier,
  `_mtp_acceptance_walk()` performs the greedy argmax/draft-token walk, and
  `rollback_speculative_cache()` is called on rejection. Batched MTP mirrors the
  same shape in `_mtp_rounds_batch()` with per-row acceptance and tail cleanup.
  DeepSeek V4 implements the verifier in
  `mlx_vlm/models/deepseek_v4/language.py::_speculative_verify()` by calling the
  model `__call__()` on the verify block, and implements rollback in
  `rollback_speculative_cache()` by snapshot/restore+replay when needed or by
  trimming/zeroing rejected cache tails. The attention modules in the same file
  update/fetch cache for the batched `L` rows before calling attention over the
  combined local/compressed/sparse K/V stream. This is the contract DS4 Plan C is
  chasing, while strict-v1 still has to preserve old DS4 row-decode numerics.
- Latest Plan C implementation checkpoint:
  `DS4_DSPARK_ATTN_VARMAP_ROWS=1` is the best current opt-in attention result,
  but only at medium length. It is byte-clean and hit `41.07 t/s` at n=1000
  (`cmp=0`, acceptance `77.1%`) versus paired no-draft `35.36 t/s` and strict
  active-5 `39.04 t/s`. At n=4000 it fell to `35.94 t/s`, below strict active-5
  `36.48 t/s`. A direct-resident variant
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_ROWS=1` was added and compared exact
  (`max=0 rms=0`, `cmp=0`), but it is slower (`36.61 t/s` at n=160,
  `35.65 t/s` at n=1000), so keep it diagnostic. Dynamic split flags
  (`DS4_DSPARK_ATTN_VARMAP_DYNAMIC_NWG=1`, plus varstream/direct equivalents)
  are byte-clean and slightly improve the long varmap run to `36.64 t/s` at
  n=4000, but this is only a small win. Conclusion: scratch cleanup is not the
  main lever; the next real Plan C work must share resident raw/compressed K/V
  tiles and converge the no-draft N=1 and DSpark N<=5 target-forward family.
- Current Plan C wrapper update: `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix`
  now auto-selects the byte-clean varmap rows5 attention scaffold inside the
  strict-v1 delegate, with
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_ROWS_DISABLE=1` as the A/B opt-out.
  Current paired n=1000 tests after this change: default strict `38.48 t/s`,
  verifier-only unified shared-prefix `38.33 t/s`, `--draft-mode unified`
  shared-prefix `38.36 t/s`, and explicit
  `DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS=1 DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_ROW_EXACT=1 DS4_DSPARK_ATTN_VARMAP_ROWS=1`
  `38.40 t/s`; all were `cmp=0` with identical `77.1%` acceptance. This makes
  the named Plan-C backend truthful and byte-clean, but it is not a speed win
  yet. Continue with resident shared-prefix mixed/compressed attention rather
  than more scratch/dispatch cleanup.
- Strict shared-tail micro-win: the DSpark row-shared tail now uses an exact
  N<=5 shared-down+HC rows kernel by default. It keeps the same per-token Q8
  reduction and HC add order as the old row loop while dispatching all active
  verifier rows at once. Disable with
  `DS4_DSPARK_HYBRID_ROW_SHARED_DOWN_HC_ROWS_DISABLE=1`. Evidence:
  `bench-results/dspark_shared_down_rows_n64_053950`,
  `bench-results/dspark_shared_down_rows_n160_054138`, and
  `bench-results/dspark_shared_down_rows_n1000_054326` all stayed `cmp=0`
  against no-draft with identical acceptance versus the disabled path. The
  n=1000 A/B measured rows default `38.29 t/s`, verifier `77.23 ms`, versus
  disabled `38.00 t/s`, verifier `78.24 ms`. Keep it default-on, but treat it
  as tail cleanup rather than the main attention/cache/indexer speed lever.
- Follow-up gate fix: setting `DS4_DSPARK_ATTN_VARMAP_ROWS=1` now enters the
  deferred row-exact attention path by itself. Older explicit defer envs are
  still accepted for A/B logs, but no longer required to activate the varmap
  rows5 scaffold. `DS4_DSPARK_ATTN_ROWS_EXACT_PROFILE=1` now also reports
  host-side encode, finish, and total milliseconds for the remaining rows-exact
  and varmap attention helpers.
- Varstream precedence fix:
  explicit `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1` or
  `DS4_DSPARK_ATTN_VARSTREAM_COMPARE=1` now takes precedence over the default
  varmap rows5 scaffold. Before this fix, varstream A/B needed
  `DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE=1` or it silently ran varmap first. Fresh
  smoke `bench-results/dspark_varstream_precedence_041228` used only
  `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1`, logged the varstream kernel, and stayed
  `cmp=0` at n=64. The speed conclusion did not change:
  `bench-results/dspark_varstream_recheck_040853` at n=160 was byte-clean for
  all rows paths, with default varmap `40.07 t/s`, row-exact fallback `38.83`,
  varstream `38.95`, and varstream dynamic `38.69`. Keep varstream as a
  correctness scaffold, not the speed target.
- Shared-prefix boundary note:
  `bench-results/dspark_rowshape_probe_041550` confirms active-5 rows are not a
  same-shape batch, but they do have a large ordered prefix. Early layers show
  raw counts like `[21,22,23,24,25]` with `raw_common=21` and `raw_tail=[0,1,2,3,4]`;
  ratio-4 compressed rows show `comp_common=5..9` with tiny tails. Therefore a
  generic K-stage/kvstage over the current varmap row-local `ic` is not valid:
  each row's `ic` maps to a different raw-union offset once the tails start.
  The next real kernel must explicitly split common raw/comp prefix from the
  row-local tail and combine them with the same online-softmax order.
- Follow-up direct-resident gate fix:
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_ROWS=1` and direct compare flags now enter the
  same deferred attention branch instead of silently running the default strict
  path. Current n=160 evidence: `bench-results/dspark_varmap_direct_gatefix_n160`
  was byte-clean (`cmp=0`) but slow, `32.73 t/s`, verifier `94.33 ms`;
  `bench-results/dspark_varmap_direct_dynamic_gatefix_n160` was also byte-clean
  and improved to `37.52 t/s`, verifier `75.81 ms`, but still trailed scratch
  varmap around `38.4 t/s`. Keep direct-resident diagnostic-only.
- Varmap dynamic split is now default-on for scratch varmap and direct-resident
  varmap. Disable with `DS4_DSPARK_ATTN_VARMAP_DYNAMIC_NWG_DISABLE=1` or
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_DYNAMIC_NWG_DISABLE=1` for A/B; use
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_DYNAMIC_NWG_DISABLE=1` only for direct-resident
  diagnostics. Current promoted-default evidence:
  `bench-results/dspark_varmap_dynamic_default_n160` was `cmp=0`,
  `39.15 t/s`, verifier `70.27 ms`; `bench-results/dspark_varmap_dynamic_default_n1000`
  was `cmp=0`, `38.79 t/s`, verifier `73.96 ms`, acceptance `77.1%`.
- Promoted-default recheck after the varmap gate/default cleanup:
  `bench-results/dspark_default_varmap_promoted_n160` was `cmp=0`,
  `39.43 t/s`, verifier `69.96 ms`, acceptance `76.4%`; paired n=1000
  `bench-results/dspark_default_varmap_promoted_n1000` was `cmp=0`,
  `38.75 t/s`, verifier `74.07 ms`, acceptance `77.1%`.
- Added varmap stage profiling:
  `DS4_DSPARK_ATTN_VARMAP_STAGE_PROFILE=1` forces the scratch varmap helper
  through copy/stage, vec, and reduce fences and prints aggregate stage totals.
  It is intentionally slow and diagnostic-only. Short smoke
  `bench-results/dspark_varmap_stage_profile_061339` stayed byte-clean
  (`cmp=0`) but slowed to 30.39 t/s. Over 492 varmap calls it reported stage
  time dominated by scratch staging/copy: copy `964.213 ms` total
  (`1.960 ms/call`), vec `162.802 ms` (`0.331 ms/call`), reduce `121.803 ms`
  (`0.248 ms/call`), with raw/comp reuse estimates both about `4.2x`. Treat
  this as evidence that the next real Plan C kernel should remove or share the
  scratch F32->F16 stream staging for committed raw/compressed prefix tiles
  before spending effort on reduce micro-ops.
  Follow-up compressed-only shadow probe:
  `DS4_DSPARK_ATTN_VARMAP_COMP_F16_SHADOW=1` (alias
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_COMP_F16_SHADOW=1`) stayed
  byte-clean but did not improve speed. `bench-results/dspark_varmap_comp_shadow_fix_063107`
  measured n=64 default DSpark `37.52 t/s` versus shadow `37.42 t/s`, both
  `cmp=0`. The fenced microscope
  `bench-results/dspark_varmap_comp_shadow_stage_063245` measured copy/stage
  `961.712 ms` default versus `967.990 ms` with shadow; vec/reduce were
  essentially unchanged. Keep the flag diagnostic-only. This says the copy
  bucket is mostly raw-union scratch materialization, so the next useful Plan C
  attention target is resident raw plus compressed shared-prefix K/V, not a
  compressed-only shadow blit.
  Raw-shadow/direct-F16 follow-up: added
  `DS4_DSPARK_ATTN_VARMAP_RAW_F16_SHADOW=1` and
  `DS4_DSPARK_ATTN_VARMAP_DIRECT_F16_SHADOW=1` as opt-in probes. Both are
  byte-clean, but both are speed rejects. `bench-results/dspark_varmap_raw_shadow_064157`
  measured n=64 default `37.56 t/s` versus raw-shadow `37.28 t/s`; fenced
  `bench-results/dspark_varmap_raw_shadow_stage_064338` measured copy/stage
  `942.495 ms` default versus `953.118 ms` with raw shadow. Direct-F16 shadows
  in `bench-results/dspark_varmap_direct_f16_064826` stayed `cmp=0`, but slowed
  to `30.65 t/s` versus default `37.45 t/s` and direct-F32 `36.93 t/s`. This
  closes the "F16 shadow instead of stream staging" branch for now. Next Plan C
  work should share committed-prefix work inside the attention kernel; do not
  spend more time on shadow-to-scratch blits or irregular direct-F16 reads.
  The companion non-stage smoke
  `bench-results/dspark_varmap_normal_after_stageprof_061550` stayed `cmp=0`
  and measured 37.55 t/s versus paired no-draft 33.29 t/s, so the diagnostic
  patch did not perturb the ordinary varmap path.
- Fresh clean active-size sweep on the current tree:
  `bench-results/dspark_current_clean_sweep_012802` used n=1000, no heavy
  route/dispatch/block diagnostics, and all DSpark outputs were `cmp=0` against
  the paired no-draft baseline (`32.92 t/s`). Active-5 is still the best strict
  default: b2 `33.27 t/s`, b3 `36.35 t/s`, b4 `38.77 t/s`, b5 `39.31 t/s`.
  b5 timing: draft `18.14 ms`, verify `73.04 ms`, overhead `1.98 ms`,
  tau `4.85`, acceptance `77.1% (794/1030)`.
- New MoE precision result: the direct local Flash IQ2/Q2 Q2-down plus
  ordered-sum kernel is now the default strict Q2 verifier path. It keeps six
  independent slot accumulators inside one kernel and uses the same ordered
  FP32 `slot0 + slot1 + ... + slot5` chain, then writes the summed output row
  directly. Disable with
  `DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` to restore the
  conservative separate Q2 down plus ordered-sum path for A/B. Evidence:
  n=1000 clean A/B `bench-results/dspark_direct_q2_clean_ab_022733` stayed
  byte-clean, with conservative `39.12 t/s` and direct-Q2 `39.66 t/s`, same
  acceptance `77.1% (794/1030)`; n=4000 earlier stayed `cmp=0` at direct-Q2
  `36.73 t/s` versus conservative `36.47 t/s`. Added
  `DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1` to prove this boundary
  directly. The hook is now symmetric: `mode=primary-separate` keeps the
  conservative output authoritative and computes direct Q2 into scratch, while
  `mode=primary-direct` keeps the default fused path authoritative and computes
  separate Q2 down plus ordered-sum into scratch. The diagnostic prints
  exact/max/rms deltas in both modes.
  `bench-results/dspark_direct_q2_compare_buckets_022458` stayed `cmp=0` and
  showed exact matches (`max=0`) for `tokens=1`, active-5 verifier calls, and
  the active-3 tail. Fresh default-primary guard
  `bench-results/dspark_direct_q2_primary_compare_051916` stayed `cmp=0` and
  logged 1720 direct-Q2 boundary compares, including active-5
  `mode=primary-direct exact=yes`, with no `exact=no` lines. This does not
  validate grouped shared-weight MoE or deeper gate/up/SwiGLU/down fusion.
  Post-promotion default validation
  `bench-results/dspark_direct_q2_promoted_default_023137` stayed clean: n=160
  `cmp=0`; n=1000 matched the paired no-draft baseline output and reached
  `39.71 t/s`, acceptance `77.1% (794/1030)`.
- Fresh clean active-size sweep after direct-Q2 promotion:
  `bench-results/dspark_direct_q2_clean_sweep_023454` used n=1000 with heavy
  diagnostics off. Paired baseline was `33.01 t/s`; b2 `33.83`, b3 `36.70`,
  b4 `39.10`, b5 `39.64 t/s`. Active-5 remains the best strict default.
- Dispatch accounting fix:
  `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1` now recognizes the direct-Q2 path and
  does not charge the removed separate ordered-sum dispatch. Validation
  `bench-results/dspark_direct_q2_dispatchfix_024232` stayed `cmp=0` and
  reported active-5 `ordered_sum=0`, `routed_moe=43`, total dispatch estimate
  `893` instead of the stale `936`. Remaining row-scaled buckets are attention
  heads `215`, compressor `205`, and indexer `105`.
- Grouped Q2 down diagnostic result:
  the route-grouping descriptor is safe, but the shared-weight grouped Q2 math
  is the wrong fusion. It diverged at n=160 (`cmp=1`, `33.17 t/s`, verifier
  `77.89 ms`, acceptance `68.0%`). `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN=1`
  now defaults to the exact descriptor wrapper, which uses the same GPU route
  grouping/output layout but invokes the known exact Q2 row kernel per pair; it
  was `cmp=0` against both default DSpark and no-draft output at n=160
  (`38.15 t/s`, acceptance `76.4%`). The rejected shared-weight math requires
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_UNSAFE=1`. Conclusion: route
  grouping is safe; the shared-weight grouped Q2 dot-product/reduction is the
  bug. Keep both flags diagnostic only. New focused hook
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_COMPARE=1` compares the safe wrapper
  and unsafe shared-weight slot-down tensors inside the same verifier call.
  Current audit `bench-results/dspark_fusion_audit_032203` reproduced the
  n=160 final `cmp=1` for unsafe grouped-Q2, while the slot-down compare shows
  ULP-level active-5 deltas (`max` up to about `4.8e-7` in the n=32 compare).
  That says the failing fusion is FP-realization/reduction order, not route
  descriptor or output layout.
- Fresh fusion-guard recheck:
  `bench-results/dspark_default_after_fusion_guard_034504` keeps the default
  strict path byte-clean at n=160 (`cmp=0`, DSpark `39.94 t/s`, paired baseline
  `33.28 t/s`, acceptance `76.4%`). Forcing
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN=1`
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_UNSAFE=1` is both wrong and
  slower (`cmp=1`, `33.83 t/s`, acceptance `68.0%`). The in-call compare hook
  saw 305 grouped-Q2 compare calls with worst slot-down delta `7.15e-7`. Keep
  unsafe grouped-Q2 rejected; do not use it as the template for A/B/C fusion.
- 2026-06-29 recheck:
  `bench-results/dspark_wrong_fusion_recheck_040257` confirms the same boundary.
  Direct ordered-Q2 compare is still exact (`cmp=0`, compare `max=0`, DSpark
  `38.62 t/s` at n=96). The grouped exact wrapper stayed byte-clean in clean
  no-compare testing but remains below the default direct ordered-Q2 path. Fresh
  four-way A/B `bench-results/dspark_grouped_q2_clean_042538` at n=160:
  baseline `33.37 t/s`, default direct ordered-Q2 DSpark `39.87 t/s` (`cmp=0`),
  grouped-safe descriptor wrapper `38.34 t/s` (`cmp=0`), and unsafe grouped
  shared-weight Q2 `33.74 t/s` (`cmp=1`). The unsafe grouped shared-weight kernel
  can get lucky
  on short final text (`cmp=0` at n=96) while the slot-down tensor is already
  non-bit-exact (`max` up to `4.768e-7` in active-5 calls); extending to n=160
  reproduces the visible failure (first diff swaps `RED`/`GREEN`). Treat any
  grouped shared-weight Q2 path as rejected until a one-layer slot-down proof
  gives exact `max=0` for active 2..5. The safe takeaway is narrow: route
  grouping/output layout are not the bug, but the current shared-weight grouped
  dot is the wrong FP realization and does not beat direct ordered-Q2.
  `bench-results/dspark_grouped_q2_compare_044810` confirms the failure mode:
  token-count-1 calls are exact, while active-5 calls differ from the safe
  wrapper by roughly `4.47e-8` to `4.77e-7` in slot-down. That is reduction/order
  drift, not a bad grouped descriptor.
- Current-tree confirmation:
  `bench-results/dspark_wrong_fusion_current_n160_062305` repeats the result
  after the varmap profiling work. Default DSpark active-5 stayed byte-clean
  (`cmp=0`, baseline `33.41 t/s`, DSpark `39.23 t/s`, acceptance `76.4%`).
  Forcing unsafe grouped-Q2 diverged (`cmp=1`), with the first output diff again
  swapping `RED`/`GREEN`; its active-5 slot-down compares reported `exact=no`
  before the ordered sum. This closes the current suspicion: the wrong fusion is
  grouped shared-weight Q2, not the default direct ordered-Q2 path.
- Common raw K-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_COMMON_KSTAGE=1` stages full raw-prefix K tiles shared
  by every active verifier row inside the varmap attention scaffold. Compare-all
  run `bench-results/dspark_varmap_common_kstage_042048` produced 738
  `varmap-attn compare` checks, all `max=0`, and final `cmp=0`, including later
  positions where a full common 32-key tile can be staged. It is not a speed
  candidate as written: no-compare n=96 was `38.26 t/s`, and compare-all fell to
  `31.67 t/s`. Keep this opt-in as a proof scaffold for exact shared-prefix
  attention, not as the production Plan C kernel.
- Common raw V-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_COMMON_VSTAGE=1` adds the matching value-tile staging
  option for the same varmap scaffold, with
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_COMMON_VSTAGE=1` as the unified-backend
  alias. It is byte-clean: `bench-results/dspark_varmap_common_vstage_043114`
  had final `cmp=0`, compare-all n=96 had zero nonzero `varmap-attn compare`
  lines, and n=160 stayed `cmp=0`. It is still slower: paired n=160 default
  DSpark was `39.86 t/s`, V-stage was `39.30 t/s`, with identical acceptance.
  This confirms K/V source substitution can be exact, but the current
  threadgroup-staging shape is not the Plan C speed kernel. The next attention
  implementation should split common-prefix online-softmax from the row-local
  tail instead of adding more staging to the existing per-row stream.
- Varmap split estimator correction:
  `DS4_DSPARK_ATTN_VARMAP_PROFILE=1` now reports both row-local prefix reuse and
  raw interval-intersection reuse. Fresh n=320 run
  `bench-results/dspark_varmap_intersection_profile_043922` stayed `cmp=0`,
  reached `41.38 t/s`, and printed `raw_split_reuse=1.22x` but
  `raw_intersection_reuse=4.46x` (`raw_intersection_full=190240`), with
  `comp_split_reuse=4.79x`. This explains why K/V prefix staging was exact but
  slower: the large shared raw opportunity is usually a middle intersection of
  the sliding raw windows, not a prefix in every row. The next kernel shape is
  private-old + shared-intersection + private-new/tail online-softmax combine.
- Raw-union K-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_RAW_UNION_KSTAGE=1` stages the shifted raw-union K
  rows for each row-local 32-key tile, so each verifier row keeps its original
  tile boundary and online-softmax order while sourcing K from shared
  threadgroup memory. `bench-results/dspark_varmap_raw_union_kstage_044329`
  had final `cmp=0` and zero nonzero `varmap-attn compare` lines. Clean A/B
  `bench-results/dspark_varmap_raw_union_kstage_clean_044600` stayed `cmp=0`
  but slowed n=160 from default DSpark `39.65 t/s` to `38.47 t/s` with the same
  acceptance. Keep it as a diagnostic proof only; raw-union K staging alone is
  not the missing speed lever.
- Raw-union V-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_RAW_UNION_VSTAGE=1` stages the shifted raw-union V
  rows for the same row-local 32-key tiles. It is byte-clean:
  `bench-results/dspark_varmap_raw_union_vstage_050432` had final `cmp=0` and
  zero nonzero `varmap-attn compare` lines. Clean A/B
  `bench-results/dspark_varmap_raw_union_vstage_clean_050547` stayed `cmp=0`
  but slowed n=160 from default DSpark `39.42 t/s` to `38.07 t/s`, with identical
  acceptance. Treat both raw-union K/V staging probes as proof scaffolds only;
  the production shared-prefix/intersection kernel needs a different phase
  structure, not more staging inside the current row-local stream.
- Raw tile-intersection K-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_KSTAGE=1` stages only the
  overlapping absolute raw K rows inside each shifted row-local 32-key tile and
  leaves private tile edges as direct loads. This also fixed the kernel/encoder
  ABI for the new tile-intersection cap argument. Compare-all
  `bench-results/dspark_varmap_tile_intersection_kstage_051154` stayed
  byte-clean (`cmp=0`) with zero nonzero `varmap-attn compare` lines. Clean
  A/B `bench-results/dspark_varmap_tile_intersection_kstage_clean_051307` also
  stayed `cmp=0`, but slowed n=160 from default DSpark `39.08 t/s` to
  `37.53 t/s` with identical acceptance. This confirms the current fusion shape
  is wrong for performance even when exact: staging fragments inside the
  row-local tile loop adds barriers and does not amortize the shared scan. The
  next speed kernel should split private-old / shared-intersection /
  private-new as separate online-softmax phases.
- Raw tile-intersection V-stage probe:
  `DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_VSTAGE=1` adds the matching
  value-side diagnostic. Current local evidence: `bench-results/dspark_raw_tile_vstage_split_084328`
  kept V-only n=64 `cmp=0`, and `bench-results/dspark_raw_tile_vstage_vonly_084543`
  kept V-only n=160 `cmp=0` with default-matching acceptance. It is not a clear
  speed path (`36.38 t/s` on that n=160 run), and the simultaneous K+V shape is
  explicitly guarded after `bench-results/dspark_raw_tile_vstage_084149`
  corrupted verifier acceptance (`1.7%`, `cmp=1`). Keep both raw tile-intersection
  K/V probes diagnostic-only. The useful attention optimization is still to
  remove repeated K/V stream materialization with a first-class shared
  raw/compressed intersection phase, not to add more row-local tile staging.
- Compressed-union K/V staging probe:
  `DS4_DSPARK_ATTN_VARMAP_COMP_UNION_KSTAGE=1` and
  `DS4_DSPARK_ATTN_VARMAP_COMP_UNION_VSTAGE=1` stage row-local compressed
  compact-index unions inside the current varmap tile loop. The first attempt
  was wrong when a later row had a compressed tail tile that row 0 did not
  execute; the fixed guard stages only tiles where every active row participates.
  A second issue showed that simultaneous K+V staging overran the safe
  threadgroup-memory shape and corrupted later ratio-4 layers, so the host guard
  now downgrades K+V requests to K-stage only with a diagnostic line. Evidence:
  `bench-results/dspark_varmap_comp_union_guard2_*` had K-only, V-only, and
  guarded K+V compare-all all `bad=0`; clean A/B
  `bench-results/dspark_varmap_comp_union_guarded160_071734` stayed `cmp=0`
  for default, K-stage, V-stage, and guarded-both. It is not a speed path:
  default n=160 was `38.77 t/s`, K-stage `37.42`, V-stage `37.49`,
  guarded-both `37.29`. Keep this diagnostic-only. The lesson is the same as
  raw-union staging: exact staging inside the row-local stream adds barriers and
  cannot replace a real shared compressed-scan online-softmax phase.
- Full-tile compressed reuse measurement:
  the varmap profile now reports `comp_common_full`,
  `comp_common_partial`, `comp_full_est`, `comp_full_reuse`, and
  `comp_full_tile_calls` alongside total `comp_common`. Fresh n=320
  `bench-results/dspark_comp_fulltile_profile_072415` stayed `cmp=0`, measured
  default varmap active-5 `41.28 t/s`, and printed
  `comp_common=52890`, `comp_common_full=34272`, `comp_tail=2979`,
  `comp_split_est=55869`, `comp_split_reuse=4.79x`, and
  `comp_full_tile_calls=777/2337`. The compressed common region is a real
  Plan C target, but a full-32-key-only shortcut is incomplete; the next shared
  compressed-scan phase must also handle the non-full common tail while
  preserving the deterministic online-softmax combine.
  Fresh footer validation `bench-results/dspark_comp_tail_profile_073055`
  stayed byte-clean (`cmp=0`) and measured default active-5 `41.51 t/s` at
  n=160. Its new estimate was `comp_common=14649`,
  `comp_common_full=6048`, `comp_common_partial=8601`,
  `comp_split_reuse=4.60x`, but `comp_full_est=50145` and
  `comp_full_reuse=1.47x`. This rejects a full-tile-only compressed shared
  scan as anything more than a throwaway bring-up probe.
- Raw/full-tail profile validation:
  `bench-results/dspark_raw_comp_tail_profile_073546` stayed byte-clean
  (`cmp=0`) and measured default active-5 `40.72 t/s` at n=96. Its new raw
  estimate was `raw_intersection=48257`, `raw_intersection_full=36736`,
  `raw_intersection_partial=11521`, `raw_intersection_reuse=4.24x`, but
  `raw_intersection_full_reuse=2.36x`. The same run had
  `comp_common_full=0`, `comp_common_partial=6048`, and
  `comp_full_reuse=1.00x`, so full-tile-only compressed sharing would do
  nothing on the early block. Treat partial common tiles as first-class in the
  Plan C shared scan.
- Phase split-lane alignment profiler:
  the C-side attention row-shape profiler now prints `raw_intersection`,
  `raw_lane_aligned`, `raw_lane`, `comp_common`, `comp_lane_aligned`, and
  `comp_lane`; `scripts/parse_dspark_phase0.py` exposes them as
  `ar_raw_intersection`, `ar_raw_lane_aligned`, `ar_raw_lane`,
  `ar_comp_common`, `ar_comp_lane_aligned`, and `ar_comp_lane`. Fresh n=160
  all-block run `bench-results/dspark_phase_alignment_all_074309` stayed
  `cmp=0`, measured default active-5 `39.82 t/s`, and printed 27 block
  summaries. Weighted over those blocks, raw intersection lane alignment was
  `96464/100517 = 96.0%`; compressed common lane alignment was
  `13137/14019 = 93.7%`. Worst compressed block was early block 2 at `41.2%`,
  but most later blocks were `80-100%`. This supports a Plan C attention kernel
  with a natural split-lane fast path plus remap fallback, not an all-remap
  design.
- Phase natural-lane run profiler:
  the row-shape profiler also prints `raw_lane_runs`, `raw_lane_avg_run`,
  `raw_lane_max_run`, `comp_lane_runs`, `comp_lane_avg_run`, and
  `comp_lane_max_run`, and the phase parser exposes the same columns under
  `ar_*`. Fresh n=160 all-block run
  `bench-results/dspark_phase_runs_profile_074806` stayed `cmp=0`, measured
  active-5 `39.68 t/s`, and over 27 block summaries reported
  `raw_lane_aligned=96464/100517 = 96.0%` across `2184` runs
  (`avg_run=44.2`, `max_run=124`) and
  `comp_lane_aligned=13137/14019 = 93.7%` across `908` runs
  (`avg_run=14.5`, `max_run=44`). The first Plan C natural-lane phase can use
  compact contiguous range descriptors; a per-key scatter table should be
  reserved for the remap fallback, if needed.
- Metal-side phase descriptor builder:
  `ds4_gpu_attention_decode_varmap_rows_tensor()` now has a reusable compact
  natural-lane descriptor builder over `row_raw_base`, `row_n_raw`,
  `row_n_comp`, and `nwg`. It emits raw ranges in raw-union coordinates and
  compressed ranges in compact-compressed coordinates, each tagged with the
  row-local split lane. The builder currently feeds the varmap aggregate
  profile and can be reused by the first Plan C shared-scan kernel without
  changing the descriptor contract. Fresh profile
  `bench-results/dspark_varmap_phase_desc_profile_075259` stayed byte-clean
  (`cmp=0`) at n=96, measured active-5 `37.22 t/s` with profile overhead, and
  printed `phase_raw_keys=48257`, `phase_raw_ranges=1845`,
  `phase_raw_avg_run=26.2`, `phase_raw_max_run=32`,
  `phase_comp_keys=5397`, `phase_comp_ranges=504`,
  `phase_comp_avg_run=10.7`, `phase_comp_max_run=23`, and zero descriptor
  overflow. Note the compressed phase keys are the lane-aligned subset of
  `comp_common=6048`; the remap fallback still needs to cover the remainder.
- Metal-side phase descriptor consumer probe:
  `DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE=1` now materializes the compact phase
  descriptors into a transient Metal buffer and dispatches
  `kernel_dspark_phase_range_probe` to verify the GPU-side ABI. The kernel
  re-aggregates raw keys/ranges and compressed keys/ranges, plus a lane mask and
  checksum, and the host compares those counters against the CPU builder before
  returning. Default probe mode checks the first varmap call; add
  `DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE_ALL=1` for every varmap call. Fresh
  validation: `bench-results/dspark_varmap_phase_probe_080002` stayed `cmp=0`
  at n=64 and logged `ranges=2 raw=21/1 comp=5/1`; the full-call smoke
  `bench-results/dspark_varmap_phase_probe_all_080115` stayed `cmp=0` at n=16
  and checked 123 varmap calls with no descriptor mismatch. This proves the
  descriptor buffer layout is ready for a real shared-scan consumer. Next step:
  replace the probe body with a compare-only shared-intersection FlashAttention
  phase, keeping current varmap output authoritative until attn-head deltas are
  exactly zero.
- K/V-address phase probe:
  `DS4_DSPARK_ATTN_VARMAP_PHASE_KV_PROBE=1` extends the same probe to read the
  descriptor-addressed F16 K/V stream (`raw-union | compressed`) and verify the
  expected half4 read count plus K/V checksum shape. This still does not write a
  candidate attention result; it is an address-contract proof for the future
  shared-intersection kernel. Fresh validation:
  `bench-results/dspark_varmap_phase_kv_probe_080742` stayed `cmp=0` at n=64
  and logged `kv_reads=3328`; the all-call smoke
  `bench-results/dspark_varmap_phase_kv_probe_all_080914` stayed `cmp=0` at
  n=16 and checked 123 varmap calls with no K/V probe mismatch. Next step is
  still the real `dspark_attn_shared_intersection_mixed_exact_n5`: consume these
  descriptors to produce compare-only FlashAttention split state / attn heads,
  then gate promotion on candidate-vs-strict `max=0`.
- Shared-range softmax-state probe:
  `DS4_DSPARK_ATTN_VARMAP_PHASE_SOFTMAX_PROBE=1` computes per-row,
  per-head, per-split-lane `S/M` online-softmax state over the descriptor-covered
  shared raw/compressed ranges. It intentionally does not write `so4`, final
  heads, or authoritative output yet. Fresh validation:
  `bench-results/dspark_varmap_phase_softmax_probe_081521` stayed `cmp=0` at
  n=64 and logged `slots=320 active=320 lanes=1`; the all-call smoke
  `bench-results/dspark_varmap_phase_softmax_probe_all_081634` stayed `cmp=0`
  at n=16 and checked 123 varmap calls with no softmax-state mismatch. This
  proves the descriptor path can run the same score/online-softmax math shape.
  Next implementation step: extend the probe to compute candidate shared-range
  `so4` plus `S/M` into a scratch split-state buffer, run the existing reducer
  into candidate heads, then compare candidate heads against strict varmap heads
  before any promotion.
- Shared-range candidate-head probe:
  `DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_PROBE=1` now performs that scaffold step. It
  consumes the descriptor-addressed F16 `raw-union | compressed` K/V stream,
  writes shared-range-only `so4 + S/M` into a scratch split-state buffer, and
  runs the existing FlashAttention reducer into candidate heads. It leaves the
  normal varmap output authoritative. Fresh validation:
  `bench-results/dspark_varmap_phase_head_probe_082347` stayed `cmp=0` at n=64
  and logged `heads=320 floats=163840 nonzero=163840 rms=0.349892`; the all-call
  smoke `bench-results/dspark_varmap_phase_head_probe_all_082459` stayed `cmp=0`
  at n=16 and exercised every varmap call. Do not compare this candidate against
  strict heads yet: private-old, private-new, and block-tail phases are still
  missing. Next implementation step is to add those phases and then gate on
  candidate-vs-strict attention-head max delta `0`.
- Rows-together shared-range candidate-head probe:
  `DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_SHARED_PROBE=1` adds the next scaffold. It
  uses the same compact natural-lane descriptors, but dispatches one
  head/split-lane threadgroup with all N<=5 verifier rows as simdgroups. Row 0
  stages each descriptor K/V chunk once into threadgroup memory and all rows
  consume it. Fresh smoke `bench-results/dspark_phase_head_shared_probe_085449`
  stayed `cmp=0` and preserved acceptance (`70.0%` at n=64), but as an add-on
  probe it is slower: default `36.69 t/s`, old per-row head probe `36.66 t/s`,
  shared head probe `33.53 t/s`. Keep this as implementation scaffolding only.
  The production `dspark_attn_shared_intersection_mixed_exact_n5` must replace
  repeated consumed-format K/V stream materialization and add the missing
  private-old/private-new/block-tail phases; running this probe beside strict
  varmap only confirms the cost shape.
- Full mixed-prefix descriptor/compare checkpoint:
  `DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_DESC_PROBE=1` now validates the CPU
  descriptor split for raw private/shared/new and compressed shared/tail ranges.
  `bench-results/dspark_mixed_desc_probe_090253` stayed `cmp=0` against paired
  default; the first active-5 block compressed `raw_row=115` into `raw_desc=31`
  and `comp_row=27` into `comp_desc=7`. `DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_COMPARE=1`
  adds a compare-only GPU candidate; strict varmap heads remain authoritative.
  `bench-results/dspark_mixed_prefix_compare_real_091336` proves the GPU
  descriptor read is correct (`raw=31/5 comp=7/3 shared=2 private=6 rows=0x1e`)
  and final output still matches default (`cmp=0`), but candidate heads are not
  close (`max` roughly `2.7..7.6`). Diagnosis: splitting raw and compressed
  phases changes strict varmap's online-softmax realization because a row-local
  32-key chunk can contain both raw and compressed keys. Next patch must make the
  candidate chunk-boundary aware: keep each row's strict 32-key FlashAttention
  chunk intact while sharing K/V loads within that chunk. Do not promote the
  current phase-split mixed-prefix candidate.
- Mixed chunk-boundary staging checkpoint:
  `DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_KSTAGE=1` and
  `DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_VSTAGE=1` now stage K or V for the
  exact strict varmap 32-key mixed raw/compressed chunk, but only when every
  active verifier row participates in that row-local chunk. This all-row guard
  fixed the first K-stage corruption (`bench-results/dspark_mixed_chunk_repeat_092250`
  had K-stage `cmp=1`). After the guard,
  `bench-results/dspark_mixed_chunk_guard_092621` had K-only `cmp=0` and V-only
  `cmp=0` at n=96 with matching acceptance; simultaneous K+V still corrupts
  output and is guarded back to K-only. The longer K-only smoke
  `bench-results/dspark_mixed_chunk_kstage_n160_092915` stayed `cmp=0`, but was
  not faster: default `36.96 t/s`, verifier `76.92 ms`; K-stage `36.79 t/s`,
  verifier `79.89 ms`. Conclusion: chunk-aware staging is useful as an exactness
  proof and safety guard, but it does not clear the speed bar. Continue toward a
  real mixed/shared-prefix verifier attention kernel that removes repeated
  consumed-format K/V materialization instead of adding row-local staging.
- Speed-first check for the old plain mixed-shared attention candidate:
  the forced unsafe path was re-measured before spending more exactness work on
  it. At n=320, `bench-results/dspark_mixed_speed_ceiling_083126` measured
  no-draft `31.57 t/s`, strict/default varmap DSpark `39.01 t/s`, unsafe
  plain mixed-shared `40.15 t/s`, and unsafe heads8 mixed-shared `37.88 t/s`.
  Both unsafe mixed variants diverged from strict output (`cmp=1`; first visible
  diff was a harmless-looking color-line reorder, but strict byte identity still
  fails). At n=1000, `bench-results/dspark_mixed_speed_ceiling_n1000_083357`
  measured strict/default varmap DSpark `36.01 t/s` versus unsafe plain
  mixed-shared `37.87 t/s` (`cmp=1`). Conclusion: the old plain
  `kernel_dsv4_plain_mixed_attention_*` path is only a small speed ceiling
  (`~3-5%` over current strict varmap on these runs) and should not be the main
  `mixed_exact_n5` exactness target. Prioritize a vec/reduce-preserving
  `dspark_attn_shared_intersection_mixed_exact_n5` kernel that keeps strict
  FlashAttention chunk/online-softmax/reduce semantics while sharing K/V loads.
  Apply this go/no-go before more exactness work: promote a new attention
  candidate only if it removes repeated consumed-format K/V materialization and
  shows at least an `8%` n=1000 speed ceiling without tau or acceptance collapse.
  The plain mixed-shared candidate and the row-local tile staging probes do not
  clear that bar.
- MLX reference map for Plan C:
  keep the local checkout `/tmp/dspark-research/mlx-vlm` beside DS4 changes.
  `mlx_vlm/speculative/mtp.py::_mtp_verify_target()` runs target verification;
  `_mtp_rounds()` and `_mtp_rounds_batch()` build `[bonus, draft_tokens]`, walk
  target-vs-draft tokens, and call rollback. In
  `mlx_vlm/models/deepseek_v4/language.py`, `_speculative_verify()` calls the
  DeepSeek V4 forward for the verify block, and `rollback_speculative_cache()`
  handles snapshot/restore/replay or tail trim/zero on rejected cache rows. Plan
  C should mirror this one-forward plus owned cache-commit contract, while
  strict-v1 keeps DS4's current byte-clean row-decode contract.
- Output-low Q8 rows5 diagnostic:
  `DS4_DSPARK_OUTPUT_LOW_Q8_ROWS5=1` now explicitly opts into the layout-aware
  output-low rows5 kernel. It is byte-clean, but current A/B
  `bench-results/dspark_output_low_rows5_034252` was slower at n=96
  (`37.93 t/s` enabled versus `38.76 t/s` disabled, both `cmp=0`). Leave the
  default off.
- Fresh profile after that hook:
  `bench-results/dspark_rows_exact_time_profile_varmap_n1000_after_hook`
  remained `cmp=0`, measured `38.37 t/s`, `77.1%` acceptance, and block timing
  `draft=18.02 ms verify=76.25 ms overhead=2.00 ms tau=4.85`. The varmap
  helper aggregate was `7421` calls / `37105` rows with only `8.938 ms`
  host-side total time across the whole n=1000 run. Compare the pure rows-exact
  fallback n=160 profile: `25.529 ms` host-side time across `1189` calls, but
  essentially identical generation speed. Conclusion: the current varmap
  scaffold has already removed most host encode overhead for this attention
  callsite; the speed blocker is still the broader target verify GPU work and
  resident shared-prefix mixed/compressed attention, not more host dispatch
  cleanup here.
- Fresh post-fusion-guard Phase-0 slice:
  `bench-results/dspark_phase0_post_fusion_guard_035057` used n=320 with route
  overlap, dispatch, block timing, shared-prefix, and attention row-shape
  profiles. Both b4 and b5 were byte-clean (`cmp=0`). Baseline was `33.12 t/s`;
  b4 reached `35.79 t/s` with acceptance `84.0%`, verify `75.01 ms`, tau
  `4.32`; b5 remained best at `38.12 t/s`, acceptance `84.6%`, verify
  `87.94 ms`, tau `5.23`. b5 dispatch estimate is still `893` with row-scaled
  buckets `heads=215`, `comp=205`, `index=105`. The first-block shared-prefix
  estimate is still the strongest target: raw reuse `3.71x`, compressed reuse
  `3.86x`.
- Varmap split-count probe:
  `DS4_DSPARK_ATTN_VARMAP_NWG=<1..32>` and
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_NWG=<1..32>` now force the scratch
  varmap split count for diagnostics; direct-resident has matching
  `*_VARMAP_DIRECT_NWG` names. `bench-results/dspark_varmap_nwg_probe_035519`
  rejects forced `nwg=1`: final n=160 output stayed `cmp=0`, but speed dropped
  from default `39.99 t/s` to `38.28 t/s`, acceptance dropped from `76.4%` to
  `68.3%`, and all-layer compare showed `max_delta=1.9e-6` against rows-exact.
  Default dynamic split compared exact (`max_delta=0`). Keep this as a
  diagnostic knob only.
- MTP is now an explicit report-card check, not a second optimization loop.
  MTP TP=2 uses different draft state and acceptance bookkeeping, but it shares
  the target verifier/cache/frontend problem. Add one-off MTP rows whenever a
  verifier backend changes enough to matter:
  no-draft baseline, current MTP TP=2 baseline, MTP TP=2 strict-v1, MTP TP=2
  Plan-C/shared-prefix diagnostic. Strict MTP rows should still use `cmp=0`
  against no-draft greedy where expected. Required columns: backend, draft kind,
  active N, generation t/s, `cmp`, acceptance, tau/accepted tokens per round,
  draft ms/block, verify ms/block, overhead ms/block, and any verifier backend
  flag. Keep this as a report-card/regression check; DSpark N=5 remains the
  main speed loop.
- MTP TP=2 default-verifier baseline on the local Flash resident target:
  with `DS4_MTP_SIDECAR_BATCH_VERIFY`, unified verifier flags, and
  `DS4_TARGET_FORWARD_UNIFIED_BACKEND` explicitly unset, the verifier reports
  `MTP sidecar verifier: exact decode2 (Flash-MoE target, no prefill dedup
  verifier)`. n=160: `28.13 t/s`, acceptance `88.5%` (`100/113`, by position
  `1=54/59 2=46/54`), `cmp=0`. n=1000 with the paired no-draft baseline
  (`32.88 t/s`): `30.21 t/s`, acceptance `89.7%` (`628/700`, by position
  `1=329/371 2=299/329`), `cmp=0`. This is the clean literal default; it is
  slower than no-draft and much slower than the sidecar-batch report-card row.
- MTP TP=2 A/B/C report-card result on the local Flash resident target:
  all MTP verifier calls pass `n=2` rows, not a padded/zeroed 5-row verifier
  (`target-forward unified N<=5 backend active: ... n=2`). n=160 smoke:
  A/current `35.43 t/s`, B/unified-batch `35.27 t/s`, C/unified-shared-prefix
  `30.73 t/s`; all `cmp=0` against the short no-draft artifact. n=1000 with a
  fresh paired no-draft baseline (`32.93 t/s`): A/current `34.63 t/s`,
  acceptance `89.7%`, `cmp=1`; B/unified-batch `35.04 t/s`, same acceptance,
  `cmp=1`, and `cmp_A_B=0`; C/unified-shared-prefix `30.77 t/s`, acceptance
  `90.1%`, `cmp=0`. Interpretation: unified-batch is a behavior-preserving
  wrapper over current MTP and slightly faster in this run, but not strict
  byte-clean at long length; shared-prefix/strict core is byte-clean but too
  slow for TP=2 today. Keep MTP as the TP=2 regression row while optimizing
  DSpark N=5 shared-prefix attention.
- MTP TP=3 A/B/C follow-up uses `n=3` verifier rows, again not a padded/zeroed
  5-row verifier (`target-forward unified N<=5 backend active: ... n=3`).
  n=160: A/current `28.37 t/s`, acceptance `72.8%` (`107/147`, by position
  `1=47/53 2=37/47 3=23/47`), `cmp=1`; B/unified-batch `28.43 t/s`, same
  acceptance and `cmp=1`, `cmp_A_B=0`; C/unified-shared-prefix `24.98 t/s`,
  acceptance `78.4%` (`109/139`, by position `1=45/51 2=37/44 3=27/44`),
  `cmp=0` on the short gate. n=1000 with a fresh paired no-draft baseline
  (`32.88 t/s`): A/current `28.20 t/s`, acceptance `74.7%` (`676/905`),
  `cmp=1`; B/unified-batch `28.17 t/s`, same acceptance and `cmp=1`,
  `cmp_A_B=0`; C/unified-shared-prefix `23.17 t/s`, acceptance `77.1%`
  (`682/884`), `cmp=1`. Interpretation: TP=3 is a bad MTP lane on this setup:
  the verifier/replay cost overwhelms the extra draft token, current and
  unified-batch are behavior-identical but divergent, and shared-prefix is both
  slow and not long-gate clean. Keep MTP at TP=2 for report-card coverage.
- Read this first: the latest reviewer feedback supersedes older "correct but
  below baseline" language in the archive sections below. The default strict
  DSpark-5 greedy path is now the production baseline: `active=5`, `cmp=0`
  through n=1000/n=4000, modestly faster than the paired no-draft baseline, and
  blocked on verifier cost rather than correctness, acceptance, or draft speed.
  Treat older active-4 and broad batch-verifier sections as historical
  diagnostics unless explicitly called out in the current plan.
- DSpark draft package loads and runs.
- DSpark experts are persistent resident.
- DSpark draft inference uses MPP 4.1 FP8/MXFP4 paths.
- Current verifier optimization north star is
  [dspark_verifier_optimization_plan.md](./dspark_verifier_optimization_plan.md).
  The shipping-safe constraint is byte-identical no-draft greedy output
  (`cmp=0`), so the current byte-clean verifier is now treated as `strict_v1`.
  Do not add unsafe batching to that path; strict-compatible speed work belongs
  behind `strict_v2` style gates and must pass n=160/n=1000/n=4000 compares
  before promotion. A separate future batch-canonical mode can be explored, but
  it must compare against a true batched target baseline, not the row-decode
  `cmp=0` oracle. Delayed Pro feedback adds a third track: **Plan C, unified
  target forward**, where no-draft decode uses the same rows API at N=1 that
  DSpark verification uses at N<=5. This is now the main research path; strict-v1
  remains production fallback and Mode B remains diagnostic. The delayed Pro
  split should be treated as additive, not a replacement: Plan A is strict old
  DS4 compatibility, Plan B is batch-canonical, and Plan C is unified-kernel
  greedy.
- DSpark loader/runtime/draft graph/verifier code has been split out of
  `ds4.c` into `dspark.c` with declarations in `dspark.h`. `ds4.c` still owns
  the shared speculative session loop, frontier snapshot/restore/commit helpers,
  and the MTP-shared generated-token plumbing.
- The old exact decodeN fallback is byte-correct but too slow; keep it as an
  oracle/fallback, not the optimization target.
- The old fast batched verifier can approach or beat baseline speed, but
  diverges from the strict greedy baseline and is diagnostic/future Mode-B work,
  not the next default path.
- The default strict hybrid verifier is commit-safe through the current n=4000
  validation and a focused exact-vs-hybrid state audit. The remaining blocker is
  speed, not draft acceptance or state correctness on the default path.
- Exact decodeN now captures prefix frontiers for accepted lengths 1..4. This
  is correct and cuts partial-accept replay cost, but exact decodeN still pays
  too much target work per proposed token.
- The strict decode-order hybrid path now also captures prefix frontiers when
  row-QKV, row-output, row-router, row-routed, and row-shared exactness are all
  enabled. After moving compressor/indexer state mutation back into the row
  attention loop, it is state-clean through N=5 in the short DSpark-KV audit.
  Plain DSpark commands select this strict hybrid path by default unless
  exact/quality mode is set. The current plain command reports
  `block=5 verify=5 active=5`, matching the strict-v1 DSpark-5 baseline.
- The finer attention-stage audit showed the prior unsafe N=4 decode-order hybrid
  first diverged at `attn-heads-raw`; row-local compressor/indexer mutation fixed
  that short-run drift. The remaining blocker is performance, not draft quality.
- The strict row-QKV path now batches the exact F16 HC-pre projection and exact
  Q8 Q/KV projections for N<=5 by default while preserving the single-row
  reduction kernels, the row-output path batches the exact Q8 output-HC tail,
  and the row-shared exact path uses the fused single-row gate/up/SwiGLU helper
  by default. The historical active-5 path enabled the row-exact tiny-batch
  routed helper, exact multirow F16 compressor projections for all compressed
  layers, exact ratio-4 indexer-compressor rows, and an exact ordered expert-sum
  kernel for routed MoE. It measured 39.04 t/s on the n=1000 smoke and 36.48 t/s
  on the long n=4000 run, remained byte-clean against the no-draft output, and
  was state-clean in the focused exact-vs-hybrid audit. This is now the
  strict-v1 baseline to preserve while optimizing verifier cost.
- Retests after the attention-order fix show that batch QKV and batched router
  are still not commit-safe. The row-routed tiny-batch wrapper is now default-safe
  because the backend either uses the conservative separate-down plus ordered-add
  verifier contract, or the newer direct ordered-Q2 down+sum kernel that preserves
  the same independent slot accumulators and slot0..slot5 FP32 add chain. Row-HC
  exact plus batch-QKV is still unsafe: it first diverges at `attn-heads-raw
  layer=0` and then grows raw-KV/final-HC/DSpark-KV deltas.
- A small verifier hot-path cleanup now keeps the N<=5 compressor/indexer row
  count arrays on the stack instead of allocating per layer. It is not a
  conceptual verifier change, but it preserves byte identity and slightly
  improves the short default smokes: n=160 is 37.33 t/s, n=1000 is 36.40 t/s.
  The long n=4000 recheck measured 33.83 t/s, so treat this as neutral cleanup;
  later ordered-sum and indexer-compressor-row work superseded the previous
  34.45 t/s best with 35.09 t/s.
- Fresh post-split active-5 check on n=320 remains the historical byte-clean
  best on the short resident test: no-draft baseline was 34.50 t/s, DSpark
  active-5 was 41.32 t/s with 84.6% acceptance (258/305 draft tokens), 61
  verifier blocks, mean draft time about 18.5 ms, mean verifier time about
  83.9 ms, and mean block total about 104.6 ms. The paired output matched the
  no-draft baseline (`cmp=0`). This is about 1.20x baseline locally, still
  short of the public ~2.29x no-spec reference target.
- Apparent post-microKV slowdowns were mostly benchmark-mode dependent. For
  headline speed, use clean no-stats paired runs; backend-stats and timing runs
  are diagnostic and can depress throughput.
- Fresh blocking stage profile on active-5 n=128 shows the verifier wall is
  still attention/cache first and routed MoE second: attention/compressor/indexer
  row loop mean 61.73 ms, row-routed MoE 31.64 ms, row-shared 13.71 ms,
  batch tail 13.84 ms, row-router 12.83 ms, row-FFN-pre 11.04 ms, target-hidden
  1.49 ms, head 1.72 ms, readback 0.02 ms, total 148.06 ms under profiling
  fences.
- Added `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1` / `DS4_DSPARK_DISPATCH_PROFILE=1`
  as a lightweight verifier dispatch-estimate profiler. It prints one line per
  hybrid decodeN block, e.g. `ds4: dspark verify dispatch-est path=hybrid ...`.
  This is intentionally helper/substage-level rather than a backend-wide Metal
  dispatch tracer; it is meant to rank the DSpark verifier hot spots before
  deeper kernel work.
- Short n=64 smoke after the profiler patch is byte-clean (`cmp=0`). Default
  active-5 reported full-block dispatch-estimate counts around `kv=215`,
  `comp=205`, `index=105`, `heads=215`, `total=1108`, with generation 36.91
  t/s and 70.0% acceptance. The KV FP8/raw row-store microbatch path later
  passed strict active-4 byte compares at n=160, n=1000, and n=4000 and is now
  default-on for decode-order verifier batches. Use
  `DS4_DSPARK_VERIFY_MICROBATCH_DISABLE=1` or
  `DS4_DSPARK_VERIFY_NO_MICROBATCH=1` to A/B it off.
- Do not over-read the microKV default as a standalone speed win. It is
  correctness-clean and part of the current strict path, but future claims still
  need paired no-stats baselines because VM pressure and diagnostic flags can
  move headline t/s by several points.
- Added `DS4_DSPARK_HYBRID_ATTN_SUBPROFILE=1` /
  `DS4_DSPARK_ATTN_SUBPROFILE=1`, a grouped attention subprofile built on the
  existing attention boundary points. It fences inside attention, so it is
  diagnostic-only and should not be used for headline t/s. A byte-clean n=64
  active-5 run averaged: `hc_pre=38.57 ms`, `q=24.93 ms`, `kv=10.50 ms`,
  `compressor=11.15 ms`, `indexer=4.90 ms`, `heads=23.82 ms`,
  `output=26.19 ms`, total `140.06 ms` under fences. This supports starting
  with exact rows-GEMM / projection bandwidth work before attempting the deeper
  fused attention/cache/indexer core.
- Added an opt-in F16 rows5 verifier experiment in `metal/dense.metal` and
  `ds4_gpu_matmul_f16_rows_exact_tensor()`. `DS4_DSPARK_VERIFY_F16_ROWS5_SEQ=1`
  dispatches one kernel per F16 rows projection but internally loops the current
  decode matvec body over rows 0..N, preserving the original per-row reduction
  path; after fixing the gate placement to the F16 helper, an n=64 active-5
  smoke is byte-clean (`cmp=0`) with 70.0% acceptance and 31.46 t/s. The first
  failed probe had accidentally routed this F16 kernel through the Q8 exact rows
  wrapper, interpreting Q8 weights as F16 and collapsing acceptance to 0.3%.
  `DS4_DSPARK_VERIFY_F16_ROWS5=1` enables the shared-weight-load variant. A
  forced `--dump-logprobs` smokes produced identical JSON continuation dumps
  against the no-draft baseline at n=64 (`cmp=0`, 12,843 bytes each) and n=160
  (`cmp=0`, 29,847 bytes each), so the basic dispatch/math shape is promising.
  It still needs n=1000/n=4000 full-output or dump-logprobs compares plus
  focused state audits before it can be considered safe or promoted.
- Q8 rows5 is now default-on for the strict verifier's N<=5 exact Q8 row
  projections. Disable with `DS4_DSPARK_VERIFY_Q8_ROWS5_DISABLE=1` or
  `DS4_DSPARK_VERIFY_NO_Q8_ROWS5=1`; force the diagnostic sequential kernel with
  `DS4_DSPARK_VERIFY_Q8_ROWS5_SEQ=1`. Current evidence:
  `bench-results/dspark_rows5_q8only_n64_retry` was byte-clean at n=64 b4 and
  improved the post-fix default b4 from `35.04 t/s`, verifier `66.62 ms`, to
  `35.39 t/s`, verifier `65.52 ms`; `bench-results/dspark_rows5_q8only_n160_b4`
  was byte-clean at n=160 and improved the same-shape default compare from
  `37.40 t/s`, verifier `64.87 ms`, to `37.89 t/s`, verifier `63.33 ms`;
  after making Q8 rows5 default, `bench-results/dspark_q8rows5_default_n160_b4`
  logged `dspark verifier Q8 rows5 shared-exact kernel enabled`, remained
  `cmp=0`, and measured `38.03 t/s`, verifier `62.90 ms`, tau `4.21`.
  F16 rows5 remains opt-in: `bench-results/dspark_rows5_f16only_n64_20260628_174745`
  was byte-clean but slower (`34.37 t/s`, verifier `68.57 ms`).
- FP8 draft rows5 is now default-on for DSpark draft dense FP8 N<=5 blocks.
  Disable with `DS4_DSPARK_DRAFT_FP8_ROWS5_DISABLE=1` or
  `DS4_DSPARK_DRAFT_NO_FP8_ROWS5=1`. The kernel
  `kernel_mul_mv_fp8_e4m3_f32_rows5` shares the FP8 weight/scale traversal across
  draft rows while preserving each row's reduction order. Evidence:
  `bench-results/dspark_draft_fp8rows5_n64_b4` was byte-clean (`cmp=0`) and
  measured `36.26 t/s` with draft `14.27 ms`;
  `bench-results/dspark_draft_fp8rows5_n160_b4` was byte-clean (`cmp=0`) and
  measured `38.45 t/s`, draft `14.45 ms`, verifier `63.39 ms`, tau `4.21`,
  acceptance `83.0%`; `bench-results/dspark_draft_fp8rows5_n1000_b4` was
  byte-clean (`cmp=0`) and measured `37.95 t/s` versus paired no-draft
  `32.93 t/s`, draft `14.87 ms`, verifier `64.13 ms`, tau `4.22`, acceptance
  `80.4%`; `bench-results/dspark_draft_fp8rows5_n4000_b4` was byte-clean
  (`cmp=0`) and measured `35.74 t/s` versus paired no-draft `31.54 t/s`, draft
  `14.93 ms`, verifier `67.86 ms`, tau `4.15`, acceptance `79.0%`. Nearest
  no-FP8 rows5 n=160 b4 compares had draft around `16.0 ms`.
- Patch E0 zero-mask attention cleanup: `ds4_gpu_encode_flash_attention_raw_heads`
  now reuses a persistent zero F16 mask instead of allocating/CPU-zeroing a
  transient mask every row, and `ds4_gpu_encode_flash_attention_gathered_heads`
  skips the per-row zero-fill mask dispatch when `use_mask == 0` (the strict
  DSpark rows-exact path passes no comp mask). This is byte-safe cleanup, not the
  final shared-prefix kernel. `make -j8 ds4` passed. Smoke
  `bench-results/dspark_zero_mask_smoke_191348` was `cmp=0`, baseline
  `33.49 t/s`, DSpark b4 `36.91 t/s`, verifier `62.48 ms`, tau `4.00`. Longer
  `bench-results/dspark_zero_mask_n160_191514` was `cmp=0`, baseline
  `33.42 t/s`, DSpark b4 `38.63 t/s`, draft `14.56 ms`, verifier `62.71 ms`,
  tau `4.21`, acceptance `83.0%`; the same directory's DSpark b5 default-shape
  run was also `cmp=0`, `38.46 t/s`, draft `17.48 ms`, verifier `73.22 ms`, tau
  `4.71`, acceptance `76.4%`. The new no-draft n=160 output also matched the
  older saved `baseline_after_default_patch_n160.out` byte-for-byte.
- Patch E1 no-mask attention specialization: the same raw/gathered single-row
  flash attention calls now use the Metal no-mask function-constant variant when
  the mask is semantically all zero. Disable with
  `DS4_FLASH_ATTN_DECODE_NO_MASK_DISABLE=1` or
  `DS4_DSPARK_ATTENTION_NO_MASK_DISABLE=1`. This removes all-zero mask reads and
  the `fma(score, scale, 0)` specialization from strict rows, while retaining the
  zero-mask fallback path for guarded A/B. Evidence: `make -j8 ds4` passed;
  `bench-results/dspark_nomask_smoke_192357` was `cmp=0`, baseline `31.35 t/s`,
  DSpark b4 `37.33 t/s`, verifier `61.41 ms`; `bench-results/dspark_nomask_n160_192514`
  was `cmp=0` for b4 and b5, baseline `33.47 t/s`, b4 `38.62 t/s`, verifier
  `62.79 ms`, b5 `38.54 t/s`, verifier `72.98 ms`; default-shape
  `bench-results/dspark_nomask_n1000_192727` was `cmp=0`, baseline `32.96 t/s`,
  DSpark b5 `38.49 t/s`, verifier `75.91 ms`, tau `4.85`, acceptance `77.1%`.
- Post-E1 active sweep:
  `bench-results/dspark_nomask_phase0_n1000_193227` ran budgets 2/3/4/5 with
  dispatch, route-overlap, block-timing, and shared-prefix profiles enabled; all
  outputs were `cmp=0`. Because these diagnostics add overhead, use the table for
  relative shape and profile columns, not headline t/s. Diagnostic results:
  baseline `33.12 t/s`; b2 `29.17 t/s`, verifier `51.68 ms`, tau `2.70`; b3
  `32.22 t/s`, verifier `60.34 ms`, tau `3.36`; b4 `34.48 t/s`, verifier
  `75.51 ms`, tau `4.22`; b5 `35.35 t/s`, verifier `87.27 ms`, tau `4.85`.
	  Shared-prefix opportunity grows with active size: b5 raw reuse estimate
	  `3.71x`, compressed reuse `3.86x`, with routed overlap reuse `1.68x`.
  Clean no-extra-profile compare: `bench-results/dspark_nomask_clean_n1000_compare`
  was b4 `cmp=0`, baseline `32.96 t/s`, b4 `38.07 t/s`, verifier `63.81 ms`,
  tau `4.22`; paired with the existing clean b5 run above, b5 is current clean
	  n=1000 winner at `38.49 t/s`.
- Patch E3 / Plan C varstream prototype: added
  `kernel_flash_attn_varstream_rows5_f16_dk512_dv512` plus
  `ds4_gpu_attention_decode_varstream_rows_tensor()` and the opt-in
  `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1` / `DS4_DSPARK_ATTN_VARSTREAM_COMPARE=1`
  switches. This is the first strict-compatible Plan C attention kernel that
  prepares per-row exact F16 key streams and runs N<=5 rows through one custom
  attention kernel. The key correctness fix was staging Q as `half4`, matching the
  strict single-row template; the first float-Q version produced local deltas
  around `1e-3`. After the Q fix and removing the scratch zero-fill, compare mode
  logged `max=0 rms=0` for the first 24 checked layers at n=64, and commit mode is
  byte-clean (`cmp=0`) against both strict DSpark and no-draft at n=64/n=160/n=1000.
  Speed is not a durable win yet: n=160 smoke reached `40.40 t/s`, but n=1000
  measured `38.14 t/s` versus the prior strict active-5 best `39.04 t/s` and the
  paired no-draft `35.36 t/s`. Treat varstream as a correctness landing point and
  scaffold for shared-prefix tile reuse, not as the final speed kernel.
- Patch E2 attention row-shape diagnostic: added
  `DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE=1`
  (`DS4_DSPARK_VERIFY_ATTN_ROWS_PROFILE=1` alias) at the graph metadata layer,
  so it observes the default strict verifier without forcing the optional
  rows-exact-head helper. A paired n=80 smoke
  `bench-results/dspark_attn_rows_shape_194956` passed `cmp=0`; baseline was
  `33.61 t/s`, DSpark was `36.89 t/s`, verifier `70.89 ms`, draft `17.82 ms`,
  tau `4.44`, acceptance `68.9%`. First-block shape line:
  `calls=43 indexed=0 rows=215 max_n=5 raw_only=21 mixed=22 raw_same_count=0
  raw_same_start=43 comp_same_count=21 raw_keys row=4995 shared_est=1343
  reuse=3.72x comp_keys row=600 shared_est=156 reuse=3.85x zero_comp_rows=105
  pad_rows=214 ring_rows=0 raw_minmax=21/30 comp_minmax=0/7`. Interpretation:
  the rows share the same raw start in this early-context smoke, but no layer has
  identical raw counts across the five verify rows. A trivial same-shape N-row
  attention helper is therefore not the next win; implement the full
  shared-prefix plus new-block-triangle kernel.
- Patch E2 sweep integration: `scripts/dspark_phase0_sweep.sh` now accepts
  `ATTN_ROWS_SHAPE_PROFILE=1` / `ATTN_ROWS_SHAPE_PROFILE_ALL=1`, and
  `scripts/parse_dspark_phase0.py` emits `ar_*` columns next to the existing
  shared-prefix `sp_*` columns. End-to-end smoke
  `bench-results/dspark_phase0_attn_rows_shape_smoke_195733` used
  `N=32 BUDGETS=5 ROUTE_OVERLAP=0 DISPATCH_PROFILE=0 BLOCK_TIMING=0
  SHARED_PREFIX_PROFILE=1 ATTN_ROWS_SHAPE_PROFILE=1 scripts/dspark_phase0_sweep.sh`.
  It passed `cmp=0`; baseline was `33.62 t/s`, DSpark was `33.14 t/s` with
  profiling overhead, draft `17.13 ms`, verifier `71.75 ms`, tau `4.00`,
  decode-equivalent verifier `2.41`, and the TSV captured both `sp_raw_reuse=3.71`
  and `ar_raw_same_count=0`. Use this sweep form for future Plan C shape checks.
- Patch E2 active-size sweep:
  `bench-results/dspark_phase0_attn_rows_shape_sweep_200046` ran budgets 2/3/4/5
  with shared-prefix and row-shape columns enabled; every output matched the
  no-draft baseline (`cmp=0`). In that profiled n=64 sweep, b4 was fastest
  (`37.72 t/s`, verifier `61.83 ms`, tau `4.00`) while b5 had the largest kernel
  target (`ar_raw_reuse=3.72`, `ar_comp_reuse=3.85`, `ar_raw_same_count=0`).
  This supports using active-4 as a shorter speed smoke but active-5 as the
  primary shared-prefix kernel target.
- Patch E2 descriptor scaffold: the attention row-shape logic in `ds4.c` now
  builds a reusable `ds4_dspark_attn_row_shape` descriptor with `common_raw`,
  `raw_tail[5]`, `common_comp`, `comp_tail[5]`, sameness flags, and reuse
  counters. The profiler consumes that descriptor today; the planned
  `dspark_attn_shared_prefix_exact_n5` kernel should consume the same shape.
  `DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE_VERBOSE=1` prints per-layer descriptors,
  including `ratio`, per-layer raw/compressed reuse estimates, sameness flags,
  and raw/compressed tail lengths.
  Smoke `bench-results/dspark_attn_rows_shape_verbose_200432` passed `cmp=0` and
  produced examples such as layer 42:
  `ratio=128 raw_common=21 raw_reuse=3.61x raw_tail=[0,1,2,3,4]
  comp_common=5 comp_reuse=4.29x comp_tail=[0,0,0,1,1]`.
  Post-refactor smoke `bench-results/dspark_attn_shape_descriptor_smoke_200723`
  also passed `cmp=0` and preserved the aggregate `ar_*` values.
- Patch E3 next kernel scope from the latest review feedback: implement a raw
  attention only `dspark_attn_shared_prefix_exact_n5` prototype first. It should
  consume the descriptor above, batch only the committed raw-KV prefix scan for
  N<=5 rows, maintain each row's online-softmax state in decode order, append
  that row's visible block-tail keys in exact order, and write the same
  `attn-heads-raw` result as row decode. Do not touch routed MoE, batch
  router/shared experts, or replace the strict-v1 default for this milestone.
- Patch E3 scaffold: added `DS4_DSPARK_ATTN_RAW_VEC_ROWS_EXPERIMENT=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS=1` alias) as an opt-in
  raw-only probe under the existing row-exact deferred-heads mode. It routes
  raw-only layers with matching `raw_start` rows through a new
  `ds4_gpu_attention_decode_raw_vec_rows_tensor()` wrapper around the safe
  vector flash-attention family, using per-row masks for the block tail. Any
  compressed row or differing raw-start condition falls back to the current
  rows-exact helper. This is a correctness probe and landing point, not the
  final shared-prefix key-scan-sharing kernel.
- Patch E3 scaffold smoke:
  `bench-results/dspark_raw_vec_rows_202053` compared the existing row-exact
  deferred-heads control against the raw vec-rows experiment. Both n=32 and
  n=160 matched byte-for-byte (`cmp=0`). The n=160 pair had identical
  acceptance `76.4% (126/165)`, identical tau `4.71`, and nearly identical
  speed: control `38.86 t/s`, verifier `72.46 ms`; raw vec-rows `39.03 t/s`,
  verifier `71.96 ms`. Treat this as a clean correctness/landing-point result,
  not a meaningful speed win yet.
- Patch E3 root cause and fix: the first attempt to make the named
  `shared_prefix` backend enable raw vec-rows diverged at n=160 because
  `--draft-mode unified` left `DS4_DSPARK_VERIFY_CANONICAL=unified` visible to
  the low-level `strict_v1` delegate, so internal batch-canonical attention
  shortcuts leaked into the supposed strict reference. Added a scoped
  `g_ds4_target_forward_strict_v1_depth` guard so `strict_v1` disables those
  batch-canonical internals. After the guard, default strict and unified
  `strict_v1` matched at n=64 in
  `bench-results/dspark_unified_strict_scope_203752`, explicit raw vec-rows
  matched unified `strict_v1` at n=64 and n=160
  (`bench-results/dspark_unified_strict_scope_n160_203930`), and the named
  `shared_prefix` backend now enables raw vec-rows by default while matching
  unified `strict_v1` at n=160
  (`bench-results/dspark_sharedprefix_rawvec_scopefixed_n160_204130`, `cmp=0`)
  and default strict at n=160
  (`bench-results/dspark_sharedprefix_vs_default_scopefixed_n160_204236`,
  `cmp=0`). This makes `shared_prefix` a byte-clean Plan C landing point again;
  the real shared-prefix scan-sharing kernel is still pending.
- Patch E4 rows5 fused raw-attention prototype: added
  `kernel_flash_attn_ext_vec_rows5_f16_dk512_dv512` and opt-in
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_FUSED=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_FUSED=1` alias). This groups
  N<=5 raw rows for one head/split-K group into one threadgroup while preserving
  one online-softmax state per row. Smoke
  `bench-results/dspark_raw_rows5_fused_204908` matched the existing
  `shared_prefix` raw vec-rows path at n=32 (`cmp=0`), but was slower:
  control `33.65 t/s`, verifier `70.18 ms`; fused rows5 `30.21 t/s`,
  verifier `83.78 ms`. Do not promote this shape. The next real kernel must
  explicitly stage/share K/V tiles for the committed prefix; merely grouping row
  simdgroups into one threadgroup is not enough.
- Patch E4 diagnostic follow-up: added
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_PROFILE=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_PROFILE=1` alias) for the raw
  vec-rows landing point. It prints the first call's active rows, `max_raw`,
  `raw_start`, `nsg/nwg`, fused flag, mask/KV/tmp/shared memory, dispatch shape,
  and estimated K-only/KV tile-staging memory. Use `_ALL=1` only when sizing the
  real shared-prefix kernel across layers/blocks; it is intentionally noisy.
- Patch E5 K-stage rows5 diagnostic: added
  `kernel_flash_attn_ext_vec_rows5_kstage_f16_dk512_dv512` behind
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_KSTAGE=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_KSTAGE=1` alias). It is still
  opt-in only. For `nsg=1`, it stages each committed-prefix K tile once per
  threadgroup and shares it across N<=5 rows while preserving each row's
  online-softmax state and exact tail mask. Smokes:
  `bench-results/dspark_raw_vec_profile_kstage_n32` matched strict/shared output
  (`cmp=0`) but ran `31.09 t/s`; `bench-results/dspark_kstage_shared_n160`
  also matched default strict (`cmp=0`) and measured `38.86 t/s` versus strict
  `38.99 t/s` with identical acceptance `76.4% (126/165)`. The first profile
  line was `n=5 max_raw=25 nsg=1 shared=48.25 KiB k_stage_est=32.00 KiB`.
  Conclusion: byte-clean and useful as a scaffold/resource check, but not a
  promoted speed path yet. Next attempt needs either broader K/V staging, less
  barrier/occupancy cost, or a larger shared prefix than the current raw-only
  early-context test exposes.
- Patch E5 aggregate raw vec-rows profile: the profile env now also prints one
  aggregate summary at process exit. n=320 shared-prefix raw vec-rows smoke
  `bench-results/dspark_raw_vec_agg_shared_n320` matched default strict
  (`cmp=0`) and reported `40.75 t/s` versus strict `40.68 t/s`. Aggregate:
  `calls=380 rows=1900 avg_rows=5.00 avg_max_raw=72.21 max_raw=123
  nsg_hist[1..4]=380/0/0/0`, with `row_keys=137200`, `tmp=7629.69 MiB`,
  `k_stage_est=11.88 MiB`, `kv_stage_est=23.75 MiB`. K-stage on the same n=320
  output also matched (`cmp=0`) but was slightly slower at `40.63 t/s`.
  Interpretation: all current raw-only calls are eligible for the `nsg=1`
  staged kernel, but staging K alone does not move the bottleneck.
- Patch E6 K+V-stage diagnostic: added
  `kernel_flash_attn_ext_vec_rows5_kvstage_f16_dk512_dv512`, but gated it behind
  explicit unsafe envs only:
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_KVSTAGE_UNSAFE=1` /
  `DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_KVSTAGE_UNSAFE=1`.
  n=32 smoke `bench-results/dspark_raw_vec_profile_kvstage_n32` diverged
  (`cmp=1`) and acceptance collapsed to `31.7% (19/60)`, so this is not a valid
  Plan C path yet. Do not use it for speed comparisons except as a focused V
  staging debug repro.
- Patch E7 dynamic-NWG raw vec-rows diagnostic: added
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_DYNAMIC_NWG=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_DYNAMIC_NWG=1` alias). It
  reduces FlashAttention split count for small raw spans and skips the reduce
  pass when `nwg=1`. n=32 and n=320 both matched strict/shared output
  (`cmp=0`). n=320 aggregate changed `nwg_hist` from fixed `32` to
  `1/2/3/4/8/16/32 = 40/120/120/100/0/0/0` and cut tmp estimate from
  `7629.69 MiB` to `627.44 MiB`, but speed dropped from raw vec-rows
  `40.75 t/s` to `39.98 t/s` (strict was `40.68 t/s`). Conclusion: tmp/reduce
  traffic is not the limiting term by itself in the current raw-only landing
  point; keep dynamic-NWG diagnostic-only.
- Patch E8 rows-exact attention aggregate: added
  `DS4_DSPARK_ATTN_ROWS_EXACT_PROFILE=1`
  (`DS4_TARGET_FORWARD_ROWS_EXACT_PROFILE=1` alias). n=320 shared-prefix profile
  `bench-results/dspark_rows_exact_profile_n320` matched strict (`cmp=0`) and
  measured `40.63 t/s`. It reported:
  `calls=1957 rows=9785 avg_rows=5.00 raw_only_rows=80 mixed_rows=9705
  raw_keys=1136780 comp_keys=267429 avg_raw_per_row=116.18 avg_comp_per_row=27.33
  calls_with_comp=1957 same_raw_start=440 same_raw_count=1517
  same_comp_count=720 max_raw=128 max_comp=84`. This is the clearest priority
  signal so far: the raw vec-rows landing point only covers a small minority
  (`380` raw vec calls in the same run), while the strict verifier still spends
  most attention work in per-row mixed/compressed gathered heads. Next Plan C
  kernel should target mixed/compressed shared-prefix rows, not more raw-only
  K staging or split-count tuning.
- Patch E9 rows-exact intersection profile: the same profile now reports
  intersection counters (`same_counts`, `same_shape`, `same_shape_mixed`) rather
  than only independent sameness counters. n=320
  `bench-results/dspark_rows_exact_intersections_n320` matched strict
  (`cmp=0`) and measured `40.64 t/s`, with the same `84.6%` acceptance. It
  reported `same_counts=720` but `same_shape=0` and `same_shape_mixed=0`.
  Follow-up `bench-results/dspark_rows_exact_reuse_n320` also matched strict
  (`cmp=0`) and measured `40.73 t/s`, adding conservative common-to-all prefix
  estimates: `raw_common_all=220408 raw_shared_est=255148 raw_reuse_est=4.46x`
  and `comp_common_all=52890 comp_shared_est=55869 comp_reuse_est=4.79x`.
  Conclusion: a narrow same raw-start/raw-count/comp-count mixed rows5 kernel has
  no coverage on this trace, but true shared-prefix decomposition has real scan
  reuse. Do not build the same-shape shortcut. Either decompose mixed/compressed
  attention into shared committed-prefix plus row-local tail, or continue the
  broader unified N=1/N<=5 kernel-family path.
- Patch E10 mixed vec-rows scaffold: added
  `DS4_DSPARK_ATTN_MIXED_VEC_ROWS_EXPERIMENT=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_VEC_ROWS_EXPERIMENT=1` alias). This
  is a pack+mask Plan C probe: it constructs a logical `[raw-union | compressed]`
  key stream, feeds the existing vector FlashAttention rows backend, and leaves
  default strict-v1 untouched. With
  `DS4_DSPARK_ATTN_RAW_VEC_ROWS_EXPERIMENT=1`, n=64 matched the deferred
  row-exact control (`cmp=0`) with identical acceptance. n=160 also matched final
  output (`cmp=0`) but changed acceptance/blocking stats (`126/165` draft tokens
  in the control vs `125/174` with the scaffold) and was slower
  (`39.04 t/s` control vs `37.23 t/s` scaffold). Conclusion: the vector
  pack+mask mixed path is not strict bit-identical verifier evidence yet; keep it
  diagnostic for Plan C/canonical experiments. The next strict-compatible kernel
  still needs explicit shared-prefix mixed/compressed attention preserving
  row-exact logits, not just final greedy bytes.
- Patch E11 shared-row mixed exact probe: added
  `DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_EXACT=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_ROWS_SHARED_EXACT=1` alias). This
  kernel keeps the row-exact attention update helper and skips invisible keys
  instead of masking them, while sharing staged K/V across up to five verifier
  rows and two heads per threadgroup. n=64 matched the deferred row-exact control
  (`cmp=0`) with identical acceptance. n=160 diverged (`cmp=1`) and changed
  acceptance/blocking from control `76.4% (126/165)` to `73.1% (125/171)`.
  A visible-row physical-cache guard did not change the result, so the remaining
  delta is likely from row/head grouping or another subtle ordering difference,
  not simple raw-ring aliasing. Keep this diagnostic-only; next step is a
  one-layer/stage audit of `attn-heads` logits or a kernel variant that preserves
  the exact heads8 grouping while sharing only safe common-prefix loads. Gate
  fix: mixed shared and mixed vec envs now auto-activate the deferred row-exact
  head path, so setting the probe alone no longer silently bypasses it. The
  committed mixed-shared path now also requires
  `DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1`; otherwise the old
  `MIXED_ROWS_SHARED_EXACT` flag falls back to rows-exact unless compare mode is
  enabled.
  `bench-results/dspark_mixed_shared_gatefix_020444` entered the branch, logged
  `mixed-shared-attn` deltas, fell back to rows-exact, and stayed `cmp=0`.
  `bench-results/dspark_mixed_shared_defer_nocompare_020252` confirmed the
  no-compare path still diverges (`cmp=1`) for both heads2 and heads8.
- Patch E11b heads8 shared-row mixed probe: added
  `DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_HEADS8=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_ROWS_SHARED_HEADS8=1` alias). This
  keeps the strict path's eight-head simdgroup layout and loops the N<=5 rows
  inside that layout. It did not solve exactness: compare mode stayed
  fallback-clean (`cmp=0`) but still logged `~1e-6` candidate-vs-strict head
  deltas; no-compare n=160 diverged (`cmp=1`), measured `36.82 t/s`, and
  changed acceptance to `70.5% (124/176)`. Conclusion: the old two-head grouping
  was not the only issue. Keep heads8 diagnostic-only and move strict work to a
  stage-level attention audit or a genuinely unified/canonical target-forward
  contract.
- Patch E12 mixed shared compare/fallback probe: added
  `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_COMPARE=1` alias, plus `_ALL`
  variants). Compare mode runs the shared-row candidate into scratch, copies it
  to `batch_heads_raw`, restores the strict row-exact heads into `batch_heads`,
  then logs candidate-vs-strict deltas after a blocking flush. n=160 stayed
  `cmp=0` against the row-exact control at `37.47 t/s` with identical acceptance
  `76.4% (126/165)`. The first logged block shows the candidate is locally close
  but not bit-identical: layer maxima are about `7.15e-07` to `2.38e-06`
  (`bench-results/dspark_mixed_shared_compare_flush_n160`). Conclusion: the
  unsafe shared-row path diverges because tiny attention-realization deltas
  amplify through later layers, not because it is reading the wrong visible
  rows. The strict-compatible Plan C kernel must preserve the row-exact
  FlashAttention vec/reduce realization, or Plan C must move to a unified
  canonical N=1/N<=5 contract instead of old-row-decode `cmp=0`.
- Patch E12b mixed shared geometry compare: added
  `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE_GEOM=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_COMPARE_GEOM=1` alias). This logs
  the row geometry beside the candidate-vs-strict head delta. Force this path
  with `DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE=1` because default varmap rows
  otherwise takes precedence. Forced n=96
  `bench-results/dspark_mixed_shared_geom_forced_052550` stayed `cmp=0` in
  compare/restore mode and showed the key fact: the mixed-shared candidate
  differs even when the row shape is the easy prefix case. At layer 2, `pos=20`,
  rows had `raw_start=0`, raw intersection `[0,21)`, `raw_shared_est=31`,
  `comp_common=5`, and max head delta `9.53674e-07`. Later rows with larger
  prefix/intersection showed the same `~1e-6` band. This rules out raw-ring
  visibility and shared-prefix geometry as the cause for this probe. The cause
  is the different attention realization: the probe uses the plain mixed online
  kernel while strict row-exact uses FlashAttention vec/reduce. Do not continue
  optimizing `kernel_dsv4_plain_mixed_attention_*` for strict `cmp=0`; either
  preserve the vec/reduce contract or move Plan C to a unified canonical
  N=1/N<=5 target-forward baseline. The committed/no-compare variant is now
  guarded: use `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1` for compare/restore, or
  add `DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1` only when intentionally
  reproducing the rejected committed diagnostic.
- Patch E13 mixed vec compare/fallback probe: added
  `DS4_DSPARK_ATTN_MIXED_VEC_COMPARE=1`
  (`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_VEC_COMPARE=1` alias, plus `_ALL`
  variants). This makes the pack+mask mixed vec-rows path run as a candidate,
  copies its heads to `batch_heads_raw`, restores row-exact heads, and logs the
  local delta. n=64 stayed `cmp=0` against the row-exact control with identical
  acceptance, but the candidate still differs locally: the first 24 compare logs
  topped out at `1.43e-06` max with RMS around `1e-08` to `3e-08`
  (`bench-results/dspark_mixed_vec_compare_n64`). This is better RMS than the
  direct shared-row probe but still not bit-identical. Conclusion: changing the
  key stream to a raw-union plus per-row mask changes the strict row verifier's
  numeric realization even when the underlying vec/reduce kernel family is reused.
  A strict speed kernel must preserve each row's exact gathered key stream/chunk
  geometry, not only the kernel family.
- Patch E14 strict rows-exact offset cleanup: added offset-aware raw/gathered
  FlashAttention encoder helpers in `ds4_metal.m` and routed
  `ds4_gpu_attention_decode_heads_rows_exact_tensor` through them. This removes
  per-token `ds4_gpu_tensor_view` allocation/free churn while preserving the same
  kernels, gathered key streams, scratch layout, and command order. It is
  byte-clean but not a meaningful speed win by itself: n=64 matched the previous
  row-exact control (`cmp=0`) at `36.29 t/s` vs old `36.81 t/s`; n=160 also
  matched (`cmp=0`) at `38.77 t/s` vs old `39.04 t/s`
  (`bench-results/dspark_offset_rows_exact_n64`,
  `bench-results/dspark_offset_rows_exact_n160`). Keep it because it simplifies
  the insertion point for a future exact per-row varstream/shared-prefix encoder.
- Next Plan C kernel target: **varstream rows5 FlashAttention**, compare-only
  first. The current c=4096 row-shape traces have per-row totals below 2048
  keys, so strict gathered heads use `nwg=32` and `nsg=1` for every DSpark row.
  That makes the first exact prototype simpler: prepare one contiguous F16
  `[row0 stream | row1 stream | ...]` buffer where each row stream is exactly the
  same gathered raw+compressed key sequence the old rows-exact helper would have
  produced, pass per-row `row_offset`/`n_keys`, and run one rows5-style kernel
  that loops only to that row's true `n_keys`. Do **not** use raw-union masks or
  max-length padding as the strict replacement; both compare probes show that
  changes numeric realization. First gate should be
  `DS4_DSPARK_ATTN_VARSTREAM_COMPARE=1`, restoring row-exact heads after logging
  deltas. Promotion requires local compare max/rms zero at n=64/n=160 and then
  full output `cmp=0` at n=160/n=1000/n=4000.

## Adjusted Optimization Plan (2026-06-28)

Latest reviewer feedback tightens the plan: do not pivot back into correctness
hunting or broad Mode-B experimentation yet. The default strict DSpark-5 greedy
verifier is now commit-safe, byte-clean through long runs, and already above the
local no-draft baseline. The remaining blocker is verifier cost.

Current strict-v1 baseline:

- active-5 default path
- `cmp=0` through n=1000 and n=4000
- block-1 and block-501 audits clean at `1e-8`
- row-local compressor/indexer mutation
- row-local ratio-128 compressor rows
- row-local ratio-4 indexer Q/weight projections
- row-QKV, row-output, row-router, row-routed, row-shared
- exact ordered routed-MoE sum
- prefix-N commit

Headline strict evidence:

- default DSpark-5 n=1000: `39.04 t/s`, `cmp=0`
- no-draft n=1000: about `35.34 t/s`
- default DSpark-5 n=4000: `36.48 t/s`, `cmp=0`
- no-draft n=4000: about `33.90 t/s`

Current diagnosis:

- Draft side is not the bottleneck: about `265 draft tok/s`, or
  `18.5-18.9 ms` for 5 draft tokens.
- Verifier side is the bottleneck: about `58 proposed tok/s`.
- The old `~146 ms/block` number is a fenced blocking stage profile, not normal
  verifier wall time. Use it only to rank buckets.
- Normal measurements should report decode-equivalents against the paired
  no-draft decode time: `draft_ms / baseline_decode_ms`,
  `verify_ms / baseline_decode_ms`, `overhead_ms / baseline_decode_ms`, and
  `tau`.
- Dominant buckets: attention/compressor/indexer row loop `~61 ms`,
  row-routed MoE `~31 ms`, row-shared `~13 ms`, row-router `~13 ms`,
  row-FFN-pre `~11 ms`, batched tail `~13.5 ms`.

Current priority order:

1. Freeze current default as `strict_v1`. Keep it selectable and unchanged.
   The runtime now logs `ds4: dspark verifier strict_v1 active=5 prefixN=1`
   on the first strict verifier block.
2. Stop retesting rejected shortcuts as candidate fixes: batch-QKV as a strict
   verifier, batched router, batched shared, confidence scheduler, margin-only
   fallback, exact-prefix-layer fallback, DSpark main-KV batch disable, direct
   routed-down `sum6`, slotwise routed, and slower precomputed indexer rows.
3. Reset measurement: add/use non-fenced per-block timing, keep fenced stage
   profiles only for bucket ranking, and report decode-equivalents plus `tau`.
   Use `DS4_DSPARK_BLOCK_TIMING=1`, `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1`, and
   `DS4_DSPARK_ROUTE_OVERLAP_LOG=1` for the Phase-0 sweep.
4. Add draft profiling. Draft is about `18-19 ms/block`; hiding or halving it is
   now meaningful. Split Markov head, three draft layers, draft routed MoE, and
   command overhead.
5. Add Plan C as the main research path: `target_forward_rows_unified(N=1)` for
   no-draft greedy and `target_forward_rows_unified(N=2..5)` for DSpark verify.
   First gate self-consistency in unified mode; old DS4 `cmp=0` is desirable but
   not required for the first prototype.
   Delayed Pro feedback makes this a third track, not a replacement for Plan A
   or Plan B: strict-v1 stays the production fallback, batch-canonical remains a
   diagnostic/throughput contract, and unified target-forward is where new
   research energy goes.
6. Prototype shared-context attention/cache/indexer batching first. Split the
   committed shared prefix from the <=4-row draft-block triangle, batch only the
   shared prefix, then merge with deterministic online-softmax state. This
   applies to Plan A/B/C, but with different proof gates: `strict_exact` must
   remain no-draft byte-identical, `batch_canonical` makes the batched causal
   forward the verifier contract, and `unified_greedy` compares DSpark spec
   against no-draft inside the same unified N=1/N<=5 target-forward family.
7. Keep the route-overlap N-sweep, but demote grouped exact routed-MoE below
   Plan C attention and ROWS5. Route reuse is real on the local code prompt, so
   MoE remains worth doing after the target-forward path is clean.
8. Prototype grouped exact routed-MoE while preserving row-exact router
   semantics, separate slot down outputs, and exact ordered FP32 slot
   summation. The first grouped IQ2 gate/up consumer was byte-clean but slower,
   so the next MoE attempt should avoid high-register grouped gate/up designs.
9. Use command reuse / Metal ICB only if dispatch census or active-vs-idle
   counters show CPU encode/launch overhead remains material after the
   shared-context attention work.
10. Keep Mode B diagnostic/future only. If Mode B verify is around `75 ms`, it is
   only modestly better than strict and still needs true committed state plus
   better `tau`/overlap to matter.
11. Keep GPU+ANE overlap as a separate-agent side track, not a main-track
   verifier task. It should not block verifier/MoE work and must not change
   verifier semantics.

MLX reference update:

- `Blaizzy/mlx-vlm` was inspected under `/tmp/ds4_mlx_vlm_ref`. Its README
  speedups, `3.94x` on 26B-A4B and `2.29x` on 31B at batch 4, are Gemma 4 MTP
  measurements, not DS4/DeepSeek-V4 Flash measurements. Do not quote those as a
  DSpark target result.
- The useful lesson is architectural: `mlx_vlm/speculative/mtp.py` verifies a
  bonus+draft block with one target forward, then walks target argmax vs draft.
  The DeepSeek-V4 language model in `mlx_vlm/models/deepseek_v4/language.py`
  exposes `speculative_verify_logits` / `speculative_verify_hidden` that call the
  same forward family as decode and then roll back rejected cache suffixes.
- This reframes the DS4 problem: instead of forever forcing a bespoke verifier
  family to match a different row-decode family, add a unified target-forward
  family used by both N=1 no-draft decode and N<=5 DSpark verify.
- The strong claim must be softened: using one code family does not guarantee
  identity by itself. Shape-dependent reductions, masks, cache layouts, and
  quant paths can still diverge. MLX-VLM proves the contract is plausible, not
  automatic for DS4/DeepSeek-V4-Flash.
- Two contracts are allowed:
  - `strict_exact`: unified rows API must match current no-draft decode at N=1
    and still gates on old-baseline `cmp=0`.
  - `unified_greedy`: no-draft decode and DSpark verify both use the new rows
    API, so the gate is DSpark-vs-no-draft within the same unified mode,
    deterministic repeatability, state self-consistency, and acceptance/tau.
- Current code status: `--draft-mode unified` /
  `DS4_DSPARK_VERIFY_CANONICAL=unified` is wired as a diagnostic label and the
  CLI sets `DS4_TARGET_FORWARD_UNIFIED=1`. The target-forward rows API exists
  for N=1 and wraps the current decode path; DSpark N<=5 now calls the same API
  and delegates through a selectable backend. Current backend knobs:
  - `DS4_TARGET_FORWARD_UNIFIED_BACKEND=batch`: default; existing batch verifier
    backend.
  - `DS4_TARGET_FORWARD_UNIFIED_BACKEND=strict_v1`: route N<=5 through the
    current byte-clean strict verifier from the same unified wrapper. This is the
    safe comparison backend for Plan C until shared-prefix rows are real.
  - `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix`: reserved insertion point
    for the planned shared-prefix attention/cache/indexer rows backend; currently
    uses the strict-v1 core plus shared-prefix profiling so explicit
    shared-prefix tests stay byte-clean.
    Set `DS4_TARGET_FORWARD_UNIFIED_SHARED_PREFIX_BATCH_FALLBACK=1` only to
    reproduce the older diagnostic batch fallback.
  - `DS4_TARGET_FORWARD_UNIFIED_STRICT_BACKEND=1`: make the pending
    shared-prefix backend return false from the unified wrapper instead of using
    the strict-v1 core. The higher-level speculative loop may still fall back to
    an older strict path.
  - `DS4_TARGET_FORWARD_UNIFIED_PROFILE=1`: print per-call wrapper timing.
  It is not yet a full MLX-style contract because the N<=5 internals are still
  the existing batch verifier backend rather than the planned shared-prefix rows
  implementation.
- Plan C smoke status:
  - `dspark_plan_c_unified_n1_smoke`: no-draft
    `DS4_TARGET_FORWARD_UNIFIED=1`, n=4, real Flash resident target, logged
    `target-forward unified active`; paired old no-draft output matched
    byte-for-byte (`cmp=0`).
  - `dspark_plan_c_unified_backend_profile_n1`: no-draft
    `DS4_TARGET_FORWARD_UNIFIED=1 DS4_TARGET_FORWARD_UNIFIED_PROFILE=1`, n=4,
    paired old no-draft output matched byte-for-byte (`cmp=0`) and printed four
    `target-forward unified profile n=1 backend=decode` lines around `29-31 ms`.
  - `dspark_plan_c_unified_mode_api2_smoke`: `--draft-mode unified
    --draft-verify 4`, n=16, real Flash resident target, logged both
    `dspark verifier unified_greedy diagnostic active=4` and
    `target-forward unified N<=5 backend active: batch verifier n=4`;
    acceptance was `70.0% (7/10 draft tokens)`. This proves the N<=5 diagnostic
    verifier now enters through `metal_graph_target_forward_rows_unified`.
    This is a plumbing smoke, not a speed claim.
  - `dspark_plan_c_unified_backend_profile_batch_n16`: explicit
    `DS4_TARGET_FORWARD_UNIFIED_BACKEND=batch DS4_TARGET_FORWARD_UNIFIED_PROFILE=1`
    logged `target-forward unified N<=5 backend active: batch n=4`, produced
    `70.0% (7/10 draft tokens)`, and printed N=4 backend timings.
  - `dspark_plan_c_unified_backend_strictv1_n16`: explicit
    `DS4_TARGET_FORWARD_UNIFIED_BACKEND=strict_v1 DS4_TARGET_FORWARD_UNIFIED_PROFILE=1`
    logged `target-forward unified N<=5 backend active: strict_v1 n=4`, produced
    `70.0% (7/10 draft tokens)`, printed N=4 timings around `69-83 ms`, and
    matched the paired default strict DSpark output on the short prompt
    (`cmp=0`). Use this to compare Plan C API overhead against the shipping-safe
    path.
  - Budget-2 routing fix: batch/unified canonical modes now skip the legacy
    decode2 special path so the selected unified backend is honored for N=2.
    Script smoke `bench-results/dspark_phase0_planc_backend_smoke` with
    `BUDGETS=2 PLAN_C_BACKENDS=strict_v1 N=8` logged
    `target-forward unified N<=5 backend active: strict_v1 n=2`, wrote
    `summary.tsv`, and matched default strict budget-2 output (`cmp=0`).
  - Perf accounting fix after the first Plan C backend sweep: batch/unified
    success paths now record all successful blocks, including full-accept,
    replay, prefix1, margin-exact, and audit returns. Do not use
    `bench-results/dspark_phase0_planc_n64_backend_20260628_173006` for
    `block_ms`/`tau` conclusions because full-accept paths were undercounted.
    `bench-results/dspark_phase0_planc_backend_smoke_after_perf` fixed the b2
    case: default strict and unified strict-v1 both reported `blocks=3`,
    matching `dspark avg scheduled ... (3 blocks)`, and output compare was
    `cmp=0`.
  - Post-fix Plan C N=64 comparison:
    `bench-results/dspark_phase0_planc_n64_strictv1_after_perf` used baseline
    `33.41 t/s`. Default strict b4 was `35.04 t/s`, `verify=66.62 ms`,
    `tau=4.00`, `blocks=16`; default strict b5 was `34.76 t/s`,
    `verify=72.14 ms`, `tau=4.27`, `blocks=15`. Unified strict-v1 b4 was
    `36.10 t/s`, `verify=62.99 ms`, `tau=4.00`, `blocks=16`, and unified
    strict-v1 b5 was `35.95 t/s`, `verify=68.08 ms`, `tau=4.27`, `blocks=15`.
    Both unified outputs matched the paired default strict outputs (`cmp=0`).
  - `dspark_plan_c_unified_backend_profile_sharedfallback_n8`:
    `DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` now logs an active
    shared-prefix backend label backed by strict-v1 core plus profiling. Older
    runs fell back to `batch(fallback)`; set
    `DS4_TARGET_FORWARD_UNIFIED_SHARED_PREFIX_BATCH_FALLBACK=1` only to
    reproduce that diagnostic path.
- Shared-prefix sizing hook:
  `DS4_DSPARK_SHARED_PREFIX_PROFILE=1` (alias
  `DS4_TARGET_FORWARD_SHARED_PREFIX_PROFILE=1`) prints a non-fenced first-block
  summary of row-exact raw/compressed attention key scans versus a
  shared-prefix estimate. It uses existing strict-v1 row metadata only; it does
  not change verifier math or timing. This is the current measurement bridge
  into `dspark_attn_shared_prefix_exact_n5`. Use
  `DS4_DSPARK_SHARED_PREFIX_PROFILE_ALL=1` only when you need one line per
  verifier block. The Phase-0 sweep helper enables the first-block profile by
  default and the parser emits `sp_*` columns; set `SHARED_PREFIX_PROFILE=0` to
  skip them.
  Fresh n=64 sweep `bench-results/dspark_sharedprefix_active_sweep_n64_185504`
  was `cmp=0` for budgets 2/3/4/5. Results: b2 `33.00 t/s`, tau `2.74`,
  raw reuse `1.95x`; b3 `34.95 t/s`, tau `3.20`, raw reuse `2.75x`; b4
  `36.23 t/s`, tau `4.00`, raw reuse `3.33x`; b5 `36.12 t/s`, tau `4.27`,
  raw reuse `3.71x`. This keeps active-4 as the short-run speed winner, while
  active-5 remains the bigger shared-prefix kernel target.
- Plan-C shared-prefix backend smoke:
  `bench-results/dspark_planc_sharedprefix_backend_n32_190227` used
  `PLAN_C_BACKENDS=shared_prefix`, n=32, budget 4. Default DSpark and unified
  shared-prefix both matched the baseline output (`cmp=0`), and unified
  shared-prefix matched default strict DSpark (`cmp=0`). The unified row logged
  `unified_backend=shared_prefix`, `tfwd_verify_avg_ms=68.712`, generation
  `33.03 t/s`, verifier `61.11 ms`, tau `3.56`, and the same shared-prefix
  key-scan estimate as strict b4 (`raw 3.33x`, compressed `3.50x`). This proves
  the backend label is now a clean Plan-C landing point while still using
  strict-v1 core.
- First implementation target remains attention/cache/indexer shared-prefix
  batching and ROWS5 shared-weight kernels. Grouped routed-MoE is demoted below
  those; the current grouped IQ2 consumer was byte-clean but slower.
- Execution merge from delayed Pro feedback: treat MLX-VLM reproduction as an
  external validation lane, not a blocker for the DS4 patch stream. If a
  separate agent is available, run the Gemma 4 MTP compare first and use it to
  sharpen the cache/commit contract. In this repo, keep moving on the named
  `shared_prefix` backend because the unified wrapper, N=1 decode route, and
  strict-v1 N<=5 comparison backend are already wired.
- Implementation rule from delayed Pro feedback: do not start with a giant
  verifier-only kernel. Start with the unified target-forward contract, then move
  one subsystem at a time into that shared family. The first real
  `shared_prefix` implementation must serve both roles: ordinary no-draft decode
  at N=1 and DSpark verification at N<=5. The point is to make decode and verify
  converge onto the same target-forward realization, not to keep making a
  verifier-only family chase a different decode family.

Plan C agent split from delayed Pro feedback:

- Agent 1: reproduce MLX-VLM Gemma 4 MTP no-draft vs spec locally, confirm
  byte-identical greedy output, speedup, block size, acceptance/tau, and inspect
  DeepSeek-V4/HISA batching. Start with the documented Gemma 4 MTP path before
  trying DeepSeek-V4, because that is the README-backed byte-identical reference:

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

  Compare no-draft output bytes, spec output bytes, no-draft tok/s, spec tok/s,
  acceptance/accepted tokens per block, and target verify time if exposed.
- Agent 2: add `DS4_TARGET_FORWARD_UNIFIED=1` and a
  `metal_graph_target_forward_rows_unified(N=1..5)` prototype. Start by wrapping
  current N=1 decode, then move subsystems into the shared path. First milestone:
  unified N=1 matches old no-draft at n=160 and n=1000 if possible. Second
  milestone: unified DSpark spec matches unified no-draft at temp 0.
- Agent 3: build `dspark_attn_shared_prefix_exact_n5` as the Plan C
  shared-prefix attention vehicle. Preserve row-local KV visibility,
  row-local compressor/indexer mutation, per-row mask/frontier snapshots, and
  exact online-softmax key traversal order. Batch only the N queries over the
  committed prefix, shared KV/compressed/index streams, and row-independent
  projections. Goal: N=1 no-draft works through the same family, then N=5
  verifier uses that same family. First milestone is raw attention only; pull
  compressor/indexer into the shared-prefix design only after `attn-heads-raw`
  passes the single-layer, block-1, block-501, and n=1000 gates.
- Agent 4: inspect DeepSpec for DSpark architecture, acceptance evaluation,
  confidence/scheduler assumptions, and target-cache expectations.

See the shorter working plan in
[dspark_verifier_optimization_plan.md](./dspark_verifier_optimization_plan.md).

Patch A status:

- Implemented default active-5: CLI/help and loader fallback now use
  `--draft-verify 5` when no explicit budget is provided.
- Loader log now says `strict_v1 verifier`.
- First strict verifier block logs `ds4: dspark verifier strict_v1 active=5
  prefixN=1`.
- Smoke `dspark_strictv1_default_active5_n64_150213`: no-draft vs plain DSpark
  default `cmp=0`; loader reported `block=5 verify=5 active=5`; generation was
  baseline `35.65 t/s`, DSpark `33.84 t/s`; acceptance `70.0% (49/70)`.

Patch B status:

- Added `DS4_DSPARK_PERF=1` as a lightweight non-fenced aggregate timing path.
- Added `DS4_DSPARK_BLOCK_TIMING=1` as the explicit per-block timing flag;
  `DS4_DSPARK_TIMING=1` remains an older alias.
- Final status can now print decode-equivalents when
  `DS4_DSPARK_BASELINE_TPS=<tps>` or `DS4_DSPARK_BASELINE_DECODE_MS=<ms>` is
  provided.
- The summary reports draft, verify, overhead, block, and `tau` without enabling
  backend VM stats or fenced stage profilers.
- Smoke `dspark_perf_footer_smoke_151213` with
  `DS4_DSPARK_PERF=1 DS4_DSPARK_BASELINE_TPS=35.34` printed:
  draft `18.33 ms`, verify `72.28 ms`, overhead `1.98 ms`, `tau=4.00`,
  decode-equivalents draft `0.65`, verify `2.55`, overhead `0.07`, block
  `3.27`, and generation `32.91 t/s` for n=32. This is a footer-format smoke,
  not a headline speed run.

Draft profiling status:

- Existing hooks cover the immediate draft split: `DS4_DSPARK_DRAFT_PROFILE=1`
  prints graph/Markov/total, `DS4_DSPARK_GRAPH_PROFILE=1` prints
  block0/block1/block2/head, and `DS4_DSPARK_BLOCK_PROFILE=1` prints per-layer
  stages including `shared_router` and `routed`.
- These profiles fence and should be used only after `DS4_DSPARK_PERF=1` shows
  the non-fenced aggregate draft bucket worth investigating.
- Smoke `dspark_draft_profile_smoke_151447` confirmed the split. Ignoring the
  first warmup-heavy block, later draft blocks were about block0 `4.6-5.5 ms`,
  block1 `4.3-5.6 ms`, block2 `4.4-5.6 ms`, output head `1.4-1.6 ms`, Markov
  `1.7-2.1 ms`, total `16.7-20.0 ms`.
- Main-track route-overlap census is implemented and shows enough reuse to move
  on to grouped exact routed-MoE. A first GPU-side grouped descriptor foundation
  is present behind `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1`; it builds
  route groups on GPU and then intentionally falls back to the current row-exact
  math, so it is a correctness/structure step and not a speedup yet. ANE draft
  overlap is a separate-agent side track only, not part of the current verifier
  optimization path.

Route-overlap / grouped-MoE correction:

- `DS4_DSPARK_ROUTE_OVERLAP_LOG=1` is the opt-in diagnostic for route reuse. It
  reads router-selected ids for measurement and therefore is not a performance
  path.
- The grouped exact MoE target is not "batch router" and not unordered direct
  `sum6`.
  Router logits/top-k/weights must stay row-exact; execution may group
  row/slot work by expert only internally; each row/slot still writes a separate
  down vector; final routed output must use the existing exact ordered FP32 slot
  sum.
- Expected interpretation: unique `28-30 / 30` means de-dup is weak, unique
  `22-26 / 30` means modest opportunity, and unique `<=20 / 30` means grouped
  exact MoE is a major verifier target.
- First smoke result: `dspark_route_overlap_smoke_152402` on
  `"Make a game of Space Invader in Pygame"` showed active-5 block reuse
  `1.84x` then `1.66x` across 43 target layers, and an active-4 tail block at
  `1.66x`. The active-5 blocks had `39-40/43` layers at reuse >= `1.25x` and
  `33-37/43` layers at reuse >= `1.50x`. This is strong enough to justify
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1` as the next main patch.
- The diagnostic uses readback/fences. The production grouped path must build a
  GPU-side grouped descriptor, expert-major sparse scan, or selector-produced
  grouped table. Do not add CPU readback in the layer loop.
- First implementation step landed: `kernel_dspark_moe_group_routes_n5` builds
  a compact grouped route descriptor on GPU for the N<=5 verifier. Opt-in smoke
  `dspark_groupdesc_optin_n64_154218` matched the descriptor-off run
  byte-for-byte (`cmp=0`) and logged
  `dspark row-routed grouped exact descriptor enabled`. Generation was
  33.19 t/s versus 34.45 t/s for descriptor-off on the same n=64 smoke; no
  speedup is expected until grouped exact gate/up/down kernels consume the
  descriptor.
- First descriptor consumer experiment is available behind
  `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_IQ2=1`. It splits IQ2 gate/up+SwiGLU
  into singleton exact rows plus grouped duplicate-expert rows. It is byte-clean
  (`cmp=0`) on n=64, but it regresses speed: clean smoke
  `dspark_groupiq2_split_n64_155719` was 29.23 t/s versus default
  `dspark_groupiq2_default_n64_155208` at 34.55 t/s, and paired stage profiling
  showed row-routed mean 54.15 ms versus default 34.08 ms. Keep this consumer
  off for speed runs; use it as a correctness reference while designing a lower
  register-pressure grouped down/sum or grouped expert-row kernel.
- `DS4_DSPARK_ROW_ROUTED_SUBPROFILE=1` now skips inside an active DSpark Metal
  command batch instead of forcing synchronization and breaking the following
  decode. Smoke `dspark_subprofile_skip_smoke_161951` completed with the skip
  warning, generation 33.34 t/s for n=32, draft 17.87 ms/block, verify
  72.04 ms/block, and `tau=4.00`. Use the normal block/stage profiles for DSpark
  verifier runs until a safe in-batch row-routed subprofile hook exists.
- Added `scripts/dspark_phase0_sweep.sh` plus
  `scripts/parse_dspark_phase0.py` to run and summarize the Phase-0 active-N
  sweep. Fresh diagnostic sweep `bench-results/dspark_phase0_n64_162431`
  used baseline 35.77 t/s and budgets 1..5 with block timing, dispatch profile,
  and route overlap enabled:

  | Budget | Gen t/s | Acceptance | Tau | Draft ms | Verify ms | Verify decode-eq | Dispatch n | Dispatch total | comp/index/heads | Route reuse |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
  | 1 | 31.64 | 90.9% | 1.91 | 5.98 | 25.39 | 0.91 | - | - | - | - |
  | 2 | 30.74 | 87.0% | 2.74 | 9.12 | 50.15 | 1.79 | 2 | 609 | 82/42/86 | 1.21x |
  | 3 | 32.88 | 75.9% | 3.20 | 12.02 | 55.53 | 1.99 | 3 | 718 | 123/63/129 | 1.38x |
  | 4 | 33.91 | 78.7% | 4.00 | 14.87 | 73.25 | 2.62 | 4 | 827 | 164/84/172 | 1.71x |
  | 5 | 34.03 | 70.0% | 4.27 | 17.75 | 77.49 | 2.77 | 5 | 936 | 205/105/215 | 1.61x |

  Interpretation: every extra verified row adds about 109 dispatch units, with
  compressor, indexer, and heads scaling linearly (`+41/+21/+43`). This confirms
  the shared-context attention/cache/indexer path as the first strict-v2 target.
  Route reuse is real at active-5 (`1.61x`) but grouped MoE is second priority
  after removing the N-times long-cache attention work.

  Optional Plan C backend comparison is now built into the same script without
  changing the default strict sweep. Set `PLAN_C_BACKENDS="batch strict_v1"` to
  add `--draft-mode unified` rows for each listed
  `DS4_TARGET_FORWARD_UNIFIED_BACKEND`; `TARGET_FORWARD_PROFILE=1` is the
  default for those rows, and `TARGET_FORWARD_PROFILE=0` suppresses per-call
  unified wrapper timing. `scripts/parse_dspark_phase0.py` now includes
  `unified_backend`, `unified_active_n`, `tfwd_decode_avg_ms`,
  `tfwd_verify_avg_ms/min/max`, `tfwd_failed_calls`, and per-N
  `tfwd_n{1..5}_avg_ms` columns so backend comparisons do not require
  hand-reading stderr.

## Latest Recovery Evidence (2026-06-28)

The apparent drop from the older 39-41 t/s active-5 band was mostly a benchmark
mode problem. Current clean no-stats runs are byte-clean and back near the old
band; diagnostic runs with `DS4_AGENT_ALLOW_BACKEND_STATS=1` or
`DS4_DSPARK_BLOCK_TIMING=1` are useful for VM/per-block data but should not be
used as headline speed numbers.

Current headline runs after restoring the no-stats final footer:

| Run | Active | cmp | Acceptance | Generation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dspark_footer_nostats_n320_133808` no-draft | - | reference | - | 32.76 t/s | clean no-stats footer |
| `dspark_footer_nostats_n320_133808` active-5 | 5 | 0 | 84.6% (258/305) | 39.26 t/s | full DSpark-5, microKV on |
| `dspark_footer_nostats_n1000_134020` no-draft | - | reference | - | 35.05 t/s | clean no-stats footer |
| `dspark_footer_nostats_n1000_134020` active-5 | 5 | 0 | 77.1% (794/1030) | 38.63 t/s | full DSpark-5, microKV on |

Fresh current-tree clean refresh:

| Run | Active | cmp | Acceptance | Generation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dspark_clean_refresh_045241` no-draft | - | reference | - | 33.25 t/s | n=320 |
| `dspark_clean_refresh_045241` active-5 | 5 | 0 | 84.6% (258/305) | 41.20 t/s | draft `18.12 ms`, verify `77.57 ms`, tau `5.23` |
| `dspark_budget_sweep_refresh_045720` active-2 | 2 | 0 | 86.3% (202/234) | 33.98 t/s | n=320 sweep |
| `dspark_budget_sweep_refresh_045720` active-3 | 3 | 0 | 78.3% (224/286) | 36.94 t/s | n=320 sweep |
| `dspark_budget_sweep_refresh_045720` active-4 | 4 | 0 | 84.0% (246/293) | 39.24 t/s | n=320 sweep |
| `dspark_budget_sweep_refresh_045720` active-5 | 5 | 0 | 84.6% (258/305) | 41.27 t/s | current best in this sweep |

Strict-v1 wrapper smoke after the plan update:

| Run | Active | cmp | Acceptance | Generation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dspark_strictv1_dispatch_n64_135032` no-draft | - | reference | - | 32.82 t/s | dispatch profiler off |
| `dspark_strictv1_dispatch_n64_135032` active-5 | 5 | 0 | 70.0% (49/70) | 34.08 t/s | `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1`, diagnostic only |

The new compact dispatch line for a full N=5 block was:
`total=936 attn_pre=43 mutate=353 heads=215 attn_out=43 ffn_pre=43 router=43 routed=86 shared=43 tail=62`.

Historical Mode B diagnostic evidence, not the current priority:

| Run | Mode | Acceptance | Generation | Notes |
| --- | --- | ---: | ---: | --- |
| `dspark_modeb_batch_n160_140724` | `--draft-mode batch` | 75.9% (126/166) | 30.77 t/s | opt-in batch-canonical path, not `cmp=0` gated |
| `dspark_modeb_prefixcommit_n160_141611` | `--draft-mode batch` | 65.6% (122/186) | 33.73 t/s | first direct prefix-commit path; removed prefix rerun, self-consistent on repeat |
| `dspark_modeb_batchproj_prefixcommit_n160_141931` | `--draft-mode batch` | 82.1% (128/156) | 35.48 t/s | prefix commit plus batched compressor/indexer projections |
| `dspark_modeb_batchproj_prefixcommit_repeat_n160_142008` | `--draft-mode batch` | 82.1% (128/156) | 35.55 t/s | `self_cmp=0` against the previous Mode B output |
| `baseline_nodraft_n160_142042` | no draft | n/a | 33.22 t/s | matched short baseline |
| `dspark_modeb_batchproj_prefixcommit_n1000_142119` | `--draft-mode batch` | 74.0% (787/1064) | 33.64 t/s | longer Mode B gate |
| `baseline_nodraft_n1000_142248` | no draft | n/a | 32.68 t/s | matched n=1000 baseline |
| `dspark_modeb_v4_deferdefault_n400_143647` | `--draft-mode batch --draft-verify 4` | 78.7% (303/385) | 36.19 t/s | deferred batch heads default-on in Mode B |
| `dspark_modeb_v4_deferdefault_repeat_n400_143742` | `--draft-mode batch --draft-verify 4` | 78.7% (303/385) | 36.11 t/s | `self_cmp=0` against previous deferred-head Mode B output |
| `dspark_modeb_deferdefault_v4_n1000_143950` | `--draft-mode batch --draft-verify 4` | 76.7% (754/983) | 35.89 t/s | current Mode B best; about 1.10x matched no-draft baseline |
| `dspark_modea_guard_n64_140821` | default strict | 70.0% (49/70) | 34.48 t/s | guard run, `cmp=0` against no-draft |
| `dspark_strict_guard_n64_142359` | default strict | 70.0% (49/70) | 33.71 t/s | post-Mode-B guard, `cmp=0` against `baseline_guard_n64_142359` |

Mode B budget sweep after prefix commit + batched projection capture, n=400:

| Run | `--draft-verify` | Acceptance | Generation |
| --- | ---: | ---: | ---: |
| `dspark_modeb_batchproj_v2_n400_142708` | 2 | 85.1% (252/296) | 28.28 t/s |
| `dspark_modeb_batchproj_v3_n400_142744` | 3 | 81.8% (284/347) | 33.60 t/s |
| `dspark_modeb_batchproj_v4_n400_142816` | 4 | 81.4% (306/376) | 34.27 t/s |
| `dspark_modeb_batchproj_v5_n400_142850` | 5 | 76.8% (317/413) | 34.21 t/s |
| `baseline_nodraft_n400_142938` | n/a | n/a | 33.00 t/s |

For this prompt and token count, budget 4 is the current best, with budget 5
effectively tied. Budget 2 is too small to amortize verifier cost.

Mode B budget sweep after deferred batch heads became default in Mode B, n=400:

| Run | `--draft-verify` | Acceptance | Generation |
| --- | ---: | ---: | ---: |
| `dspark_modeb_deferdefault_v3_n400_143831` | 3 | 83.1% (285/343) | 34.58 t/s |
| `dspark_modeb_v4_deferdefault_n400_143647` | 4 | 78.7% (303/385) | 36.19 t/s |
| `dspark_modeb_deferdefault_v5_n400_143904` | 5 | 72.6% (313/431) | 35.79 t/s |
| `baseline_nodraft_n400_142938` | n/a | n/a | 33.00 t/s |

For the current Mode B path, budget 4 is the clear local best on this prompt.
Budget 5 proposes more draft rows but loses enough acceptance/overhead that it
does not win.

Mode B timing averages from `dspark_modeb_batch_n160_140724`:

- Full accepts: 19 blocks, draft 19.46 ms, verify 89.09 ms, commit 1.95 ms,
  total 110.98 ms.
- Partial accepts: 11 blocks, mean committed prefix 3.18, draft 19.86 ms,
  initial verify 92.60 ms, prefix rerun 66.81 ms, total 179.81 ms.

Mode B timing averages from `dspark_modeb_batchproj_prefixcommit_n1000_142119`:

- Full accepts: 115 blocks, draft 20.26 ms, verify 96.88 ms, DSpark KV commit
  2.08 ms, total 119.73 ms.
- Partial accepts: 75 blocks, mean committed prefix 2.84, draft 20.28 ms,
  verify 97.33 ms, prefix commit 1.96 ms, total 120.10 ms.

Mode B timing averages from `dspark_modeb_deferdefault_v4_n1000_143950`:

- Full accepts: 165 blocks, draft 16.63 ms, verify 75.18 ms, DSpark-KV commit
  1.65 ms, total 93.99 ms.
- Partial accepts: 45 blocks, mean committed prefix 2.11, draft 16.59 ms,
  verify 75.20 ms, prefix commit 1.63 ms, total 93.96 ms.
- Short post-change verify profile
  `dspark_modeb_deferdefault_v4_verifyprofile_n160_144420`: for N=4 blocks,
  layer encoding averages 73.29 ms, head 1.71 ms, readback 0.001 ms.

Interpretation:

- `--draft-mode batch` / `DS4_DSPARK_VERIFY_CANONICAL=batch` is now wired.
- Full accepts commit the batched verifier state directly.
- Partial accepts now commit prefix lengths 1..4 captured during the initial
  verifier. The old `~67 ms` prefix rerun is gone; prefix commit itself is about
  `1.6-2.0 ms`.
- Deferred batch attention heads are default-on for Mode B and cut the n=1000
  verify bucket from about `97 ms/block` to about `75 ms/block`.
- Mode B is self-consistent on repeat short gates and beats the matched no-draft
  baselines by about `1.07x` at n=160 and `1.10x` at n=1000 on the best
  local budget-4 run.
- Keep this as diagnostic/future evidence. The current plan is strict-v1 freeze
  plus strict-v2 attention, not target-batch-no-spec or broad Mode-B work.

Interpretation:

- Current clean active-5 is back in the same class as the older active-5
  n=1000 result (`39.04 t/s`), with identical acceptance shape and `cmp=0`.
- The n=320 clean result is still below the older `41.32 t/s`, but the paired
  no-draft baseline is also lower. Treat that as run/machine variance unless it
  repeats under clean no-stats runs.
- `DS4_AGENT_ALLOW_BACKEND_STATS=1` should be used when VM/compressed-memory
  evidence is needed; it is not the headline speed mode.
- `DS4_DSPARK_BLOCK_TIMING=1` is diagnostic only. It emits per-block timing and
  can depress headline throughput. `DS4_DSPARK_TIMING=1` remains an older alias.

Recent diagnostic runs that looked like a regression:

Short paired recovery, tag `dspark_recovery_n320_132313`:

| Run | Active | cmp | Acceptance | Generation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| No DSpark baseline | - | reference | - | 32.71 t/s | VM task-compressed 15.08 GiB |
| Plain default | 4 | 0 | 84.0% (246/293) | 36.34 t/s | draft 238.3 tok/s, verify 56.8 proposed tok/s |
| `--draft-verify 5` | 5 | 0 | 84.6% (258/305) | 37.75 t/s | draft 245.6 tok/s, verify 58.7 proposed tok/s |
| `--draft-verify 5`, microKV off | 5 | 0 | 84.6% (258/305) | 37.20 t/s | `DS4_DSPARK_VERIFY_MICROBATCH_DISABLE=1` |

Longer paired recovery, tag `dspark_recovery_n1000_132748`:

| Run | Active | cmp | Acceptance | Generation | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| No DSpark baseline | - | reference | - | 32.42 t/s | VM task-compressed 14.12 GiB |
| Plain default | 4 | 0 | 80.4% (762/948) | 36.06 t/s | draft 238.0 tok/s, verify 59.5 proposed tok/s |
| `--draft-verify 5` | 5 | 0 | 77.1% (794/1030) | 36.16 t/s | draft 245.6 tok/s, verify 61.8 proposed tok/s |

Timing comparison for active-5 n=320:

| Run | Generation | Avg draft | Avg verify | Avg total |
| --- | ---: | ---: | ---: | ---: |
| Historical `dspark_goal_cont_active5_timing_n320` | 41.32 t/s | 18.50 ms | 83.91 ms | 104.60 ms |
| Current `dspark_recovery_timing_active5_n320_132616` | 37.81 t/s | 20.34 ms | 90.93 ms | 113.73 ms |

Diagnostic interpretation:

- Current active-5 is still byte-clean and slightly faster than active-4 at
  n=320, but only tied with active-4 at n=1000.
- The slowdown is inside block work, not acceptance: the n=320 active-5
  acceptance is still 84.6%, matching the historical run.
- MicroKV is not the obvious culprit; disabling it made active-5 slightly slower
  on n=320.
- These numbers are still useful for draft/verifier split, but they should not
  override the clean no-stats headline numbers above.

## Current Reviewer Ask

Please focus advice on a production N<=5 exact microbatch verifier. The draft
model quality is already strong enough; the default verifier is commit-safe, but
too much of it is still row-fenced. The useful question is how to preserve exact
decode-order cache mutation and accumulation while removing dispatch/per-row
overhead.

The most useful state audit order is:

1. Final HC row after each target layer.
2. Raw KV row and compressed/indexer frontiers.
3. Attention/index compressor state.
4. DSpark target-hidden and draft KV state.
5. Post-FFN/MoE accumulation before commit.

Known-safe batching so far:

- Exact multirow F16 HC-pre projection.
- Exact multirow Q8 Q/KV projections while retaining row-local reduction/update.
- Exact multirow attention-compressor projections for compressed layers.
- Exact Q8 output-HC tail batching.
- Fused single-row shared gate/up/SwiGLU helper.
- Exact ordered routed-MoE expert-sum kernel, which preserves the old
  separate-down-plus-ordered-add FP32 sum contract while reducing add dispatches.

Known-unsafe shortcuts:

- Prefill-style batch QKV.
- Batched router feeding row-routed MoE.
- Batched shared expert.
- Existing row-routed tiny-batch wrapper.
- Row-HC exact plus batch-QKV.
- `DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=1` is fast, and DSpark matches a no-draft
  target run using the same flag, but that target run differs from the normal
  greedy baseline. It changes the routed-down accumulation/rounding contract, so
  it is not a default-exact verifier path.

## Local Test Setup

Target:

```bash
BASE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
```

DSpark draft:

```bash
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
```

Prompt:

```bash
PROMPT='Make a game of Space Invader in Pygame'
```

Baseline command:

```bash
./ds4 -m "$BASE" \
  --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096
```

DSpark command shape:

```bash
DS4_DSPARK_PERF=1 DS4_DSPARK_BASELINE_TPS=<paired-no-draft-tps> \
./ds4 -m "$BASE" \
  --draft dspark --draft-path "$DRAFT" \
  --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096
```

The plain DSpark command now defaults to `block=5 verify=5 active=5`. Use
`DS4_AGENT_ALLOW_BACKEND_STATS=1` only when collecting backend/VM diagnostics;
do not use it for headline speed. If the paired baseline is easier to express as
latency, use `DS4_DSPARK_BASELINE_DECODE_MS=<ms>` instead of
`DS4_DSPARK_BASELINE_TPS=<tps>`.

Important correctness check:

```bash
cmp -s bench-results/<tag>_baseline.out bench-results/<tag>_dspark.out
echo $?
```

`cmp=0` means byte-identical greedy output to no-draft baseline.

## Code Map

Main files touched for DSpark:

- `dspark.c`
- `dspark.h`
- `ds4.c`
- `ds4.h`
- `ds4_cli.c`
- `ds4_gpu.h`
- `ds4_metal.m`
- `metal/dense.metal`
- `metal/dsv4_misc.metal`
- `metal/glu.metal`
- `metal/mxfp4_native.metal`
- `scripts/dspark_export.py`
- `scripts/export_dspark_draft.sh`
- `tests/dspark_export_smoke.py`

Key DSpark functions now live in `dspark.c` / `dspark.h`:

- `ds4_dspark_open` / `ds4_dspark_close`: DSpark draft package loader and
  resident expert setup.
- `metal_graph_eval_dspark_draft`: DSpark draft block generation.
- `metal_graph_dspark_three_layer_forward`: three-layer draft model forward.
- `metal_graph_verify_decode2_exact`: exact N=2 decode-order verifier.
- `metal_graph_verify_decodeN_exact`: exact N<=5 decode-order verifier. This is
  correct but slow; it captures prefix frontiers for partial accepts of lengths
  1..4.
- `metal_graph_verify_decodeN_attn_exact_ffn_batch`: current strict hybrid
  verifier with exact decode-order attention/cache mutation and selected batched
  row-independent tails.
- `ds4_dspark_enable_default_fast_verifier` and
  `ds4_dspark_decodeN_policy_make`: DSpark-specific verifier defaults and env
  policy gates, moved out of the session loop.

Shared speculative/session pieces still live in `ds4.c`:

- `metal_graph_verify_suffix_tops`: fast layer-major batched verifier. This is
  fast but unsafe for committed state on long runs.
- `spec_frontier_snapshot`, `spec_frontier_restore`,
  `spec_frontier_commit_prefix1`, and `spec_frontier_commit_prefix`: shared
  state rollback/commit machinery used by MTP and DSpark.
- DSpark speculative control flow is around the `--draft dspark` path in
  `ds4_session_eval_speculative_argmax`.

## Current Optimization Plan

1. Use clean no-stats runs for headline speed. The current headline numbers are
   active-5 n=320 at 39.26 t/s (`cmp=0`) and active-5 n=1000 at 38.63 t/s
   (`cmp=0`). Use `DS4_AGENT_ALLOW_BACKEND_STATS=1` only when VM/compressed
   memory data is needed, and use `DS4_DSPARK_BLOCK_TIMING=1` only for block
   timing.
2. Keep recovery A/Bs paired and boring: no-draft baseline, plain active-5
   default, and active-5 with `DS4_DSPARK_VERIFY_MICROBATCH_DISABLE=1` only when
   isolating microKV. Smaller budgets are useful A/Bs, but they are not the
   current DSpark-5 default. Every run needs a `cmp=0` gate and its own no-draft
   baseline because machine state can move the baseline by several t/s.
3. Treat active-5 as the primary DSpark-5 optimization target again. Active-4
   remains useful for local A/Bs, but active-5 now has clean current evidence and
   matches the checkpoint's production block size.
4. Use the dispatch-estimate profiler only for hot-spot ranking, not headline
   t/s. The useful profiler inputs are
   `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1` plus, when needed,
   `DS4_DSPARK_HYBRID_ATTN_SUBPROFILE=1`; they should rank hot spots, not
   replace the baseline recovery step.
5. Build the production verifier around a dedicated N<=5 verifier path, not the
   old unsafe batch verifier and not a literal single giant kernel. The path
   should preserve exact row-order cache mutation and FP32 accumulation contracts
   in Plan A, while leaving room for a Plan B batch-canonical contract.
6. Attack attention/cache first with shared-context batching. The current profile
   says attention/compressor/indexer row mutation is the largest bucket
   (~62 ms profiled per five-token block), and the dispatch-estimate profiler
   shows hundreds of KV/compressor/indexer/head row units per block. Split the
   verifier into a batched committed-prefix pass plus a tiny decode-order
   new-block triangle. In Plan A this must be byte-identical to no-draft decode;
   in Plan B the batched causal forward can become canonical and is judged by
   self-consistency plus acceptance/tau.
7. Attack routed MoE second. Row-routed MoE remains the next largest bucket
   (~32 ms profiled per block). The safe direction is an exact-preserving
   multirow helper that keeps separate down outputs and ordered FP32 expert-sum
   semantics, rather than enabling routed-down `sum6`, which changes the normal
   greedy baseline contract.
8. Leave row-shared, row-router, FFN-pre, target-hidden, and output-head work as
   secondary cleanup unless profiling changes; each is materially smaller than
   attention/cache and routed MoE.

Recent diagnostic patch:

- Added graph flag `spec_disable_shared_gate_up_swiglu`.
- The hybrid verifier sets it while calling batched FFN/MoE, because global
  `DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION=1` made a short n=220 hybrid
  run byte-match.
- This did not make n=1000 hybrid safe, so it fixed one drift source but not all.
- Added `DS4_DSPARK_HYBRID_STATE_AUDIT` / `DS4_DSPARK_DECODEN_HYBRID_STATE_AUDIT`
  to run hybrid and exact decodeN side-by-side, log logits/state deltas, then
  continue with exact rows.
- Added exact decodeN prefix-N capture for `N > 1`; partial accepts of lengths
  1..4 now commit without replay. Disable with
  `DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1` or
  `DS4_DSPARK_DECODEN_PREFIX1_DISABLE=1` for A/B testing.
- Prefix-N is byte-correct on the n=160 smoke sweep for budgets 2..5, but it
  does not solve the speed issue because exact decodeN verification is still
  nearly one normal target decode per verified token.
- Added per-layer post-HC audit for exact decodeN vs hybrid:
  `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT=1`, with optional
  `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_VERBOSE=1`.
- The same audit can now compare attention HC and selected FFN substages. Use
  `DS4_DSPARK_HYBRID_ATTN_STAGE_AUDIT=1` for Q-rope, KV/cache row, attention
  heads, and attention output; add `DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1` for
  FFN substages and tune the threshold with
  `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS=<float>`. The audit now prints a
  chronological `dspark hybrid first divergence` summary with stage, layer, row,
  element index, and values; `1e-5` is useful for material drift and `1e-8` for
  byte-clean confirmation.
- Added experimental `DS4_DSPARK_HYBRID_EXACT_PREFIX_LAYERS=N`, which runs
  early target layers in exact decode order before switching to the hybrid
  batched FFN/MoE path. This is diagnostic only; it does not currently produce
  a useful speed/correctness point.
- Added experimental `DS4_DSPARK_HYBRID_ROW_FFN_PRE=1`, which computes FFN
  HC-pre/norm row-exact, then resumes the batched FFN tail. This is a useful
  diagnostic and gives some correct points, but it is not a complete production
  verifier.
- Added experimental `DS4_DSPARK_HYBRID_ROW_ROUTER=1`, which also computes
  router logits/selection/weights row-exact before the batched MoE/shared tail.
  This is diagnostic only; the first n=160 smoke diverged.
- Added experimental `DS4_DSPARK_HYBRID_ROW_SHARED=1`, which computes shared
  gate/up/SwiGLU row-exact before the batched tail.
- Added experimental `DS4_DSPARK_HYBRID_ROW_ROUTED=1`, which computes routed
  MoE row-exact. Combined with row-shared, this is the first fully clean stage
  audit, but the n=1000 speed is still well below baseline.
- Added `DS4_DSPARK_TINY_BATCH_ROW_KERNEL=1` / alias
  `DS4_MTP_SIDECAR_BATCH_SLOTBANK_ROW_EXACT=1`, a diagnostic that encodes the
  single-row native MXFP4 kernels in one command buffer for the tiny-batch routed
  path. Current audits show this is not enough to reproduce the row-routed exact
  boundary by itself.
- The Flash resident tiny-batch path now preserves true expert ids directly when
  the resident slot bank is in identity-selected mode (`slot id == expert id`),
  so it matches the selected-id contract used by normal decode in that mode.
- Added `DS4_DSPARK_DECODEN_DISABLE=1` /
  `DS4_DSPARK_DECODE_N_DISABLE=1` so N=3..5 blocks can be deliberately routed
  into the experimental batch verifier for diagnosis. Use it with
  `DS4_DSPARK_DECODE2_DISABLE=1` when forcing all DSpark blocks through batch
  verification.
- Extended `DS4_DSPARK_BATCH_STATE_AUDIT=1` to capture DSpark target-hidden
  rows. Add `DS4_DSPARK_BATCH_STATE_AUDIT_DSPARK_KV=1` to compare
  phase-correct DSpark draft KV cache rows, and
  `DS4_DSPARK_BATCH_STATE_AUDIT_DECODE_HC=1` to compare the batch verifier's
  final hidden row against exact replay's current decode hidden row.
- Added `DS4_DSPARK_HYBRID_STATE_AUDIT_DSPARK_KV=1` for the decodeN hybrid
  verifier. It rebuilds DSpark draft KV rows from the hybrid accepted-prefix
  target-hidden state, restores the verifier frontier, rebuilds the same rows
  from exact decodeN state, then includes `dspark-kv-cache` in the
  `hybrid-vs-exact` state summary.

## Current Benchmarks

All runs use the same prompt and resident Flash target unless noted.

Archive note: tables below this point include older active-4 and budget-4
experiments from before the default was tightened to strict-v1 active-5. Keep
them for forensics, but do not treat those rows as the current implementation
target.

### Baselines

| Case | Tokens | Correctness | Generation |
| --- | ---: | --- | ---: |
| No DSpark baseline, clean no-stats current | 1000 | reference | 35.05 t/s |
| No DSpark baseline, clean no-stats current | 320 | reference | 32.76 t/s |
| No DSpark baseline, latest recovery | 1000 | reference | 32.42 t/s |
| No DSpark baseline, latest recovery | 320 | reference | 32.71 t/s |
| No DSpark baseline, refreshed current binary | 1000 | reference | 35.46 t/s |
| No DSpark baseline, earlier run | 1000 | reference | 35.19 t/s |
| No DSpark baseline | 220 | reference | 35.62 t/s |

### Latest Recovery Runs

These are the freshest current-tree numbers. Treat clean no-stats rows as
headline speed and `DS4_AGENT_ALLOW_BACKEND_STATS` / `DS4_DSPARK_BLOCK_TIMING`
rows as diagnostic speed.

| Case | Active | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean no-stats active-5, tag `dspark_footer_nostats_n320_133808` | 5 | 320 | 0 | 84.6% (258/305) | 39.26 t/s |
| Clean no-stats active-5, tag `dspark_footer_nostats_n1000_134020` | 5 | 1000 | 0 | 77.1% (794/1030) | 38.63 t/s |
| Plain default, tag `dspark_recovery_n320_132313` | 4 | 320 | 0 | 84.0% (246/293) | 36.34 t/s |
| Active-5, tag `dspark_recovery_n320_132313` | 5 | 320 | 0 | 84.6% (258/305) | 37.75 t/s |
| Active-5 microKV off, tag `dspark_recovery_microkvoff_n320_132516` | 5 | 320 | 0 | 84.6% (258/305) | 37.20 t/s |
| Plain default, tag `dspark_recovery_n1000_132748` | 4 | 1000 | 0 | 80.4% (762/948) | 36.06 t/s |
| Active-5, tag `dspark_recovery_n1000_132748` | 5 | 1000 | 0 | 77.1% (794/1030) | 36.16 t/s |

### Correct DSpark Paths

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact decodeN + prefix1 | 1 | 1000 | 0 | 88.1% (468/531) | 31.43 t/s |
| Exact decodeN/decode2 + prefix1 | 2 | 1000 | 0 | 85.1% (630/740) | 30.43 t/s |
| Exact decodeN + prefix1 | 3 | 1000 | 0 | 78.7% (702/892) | 28.87 t/s |
| Exact decodeN + prefix1 | 4 | 1000 | 0 | 80.4% (762/948) | 27.56 t/s |
| Exact/read1 decodeN | 3 | 1000 | 0 | 78.7% (702/892) | 28.49 t/s |
| Exact/read1 decodeN | 4 | 1000 | 0 | 80.4% (762/948) | 27.11 t/s |
| Exact/read1 decodeN | 5 | 1000 | 0 | 77.1% (794/1030) | 25.19 t/s |
| Hybrid row-routed + row-shared | 2 | 1000 | 0 | 85.1% | 30.37 t/s |
| Hybrid row-routed + row-shared | 4 | 1000 | 0 | 80.4% | 26.77 t/s |

Short current-binary prefix-N smoke, n=160:

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| No DSpark baseline | - | 160 | reference | - | 36.51 t/s |
| Exact decode2 + prefix1 | 2 | 160 | 0 | 77.6% (97/125) | 30.27 t/s |
| Exact decodeN + prefix-N | 3 | 160 | 0 | 70.6% (108/153) | 29.46 t/s |
| Exact decodeN + prefix-N | 4 | 160 | 0 | 83.0% (122/147) | 29.51 t/s |
| Exact decodeN + prefix-N | 5 | 160 | 0 | 76.4% (126/165) | 28.04 t/s |
| Hybrid row-QKV + row-routed + batch router/shared | 3 | 160 | 0 | 72.7% (109/150) | 33.14 t/s |
| Hybrid row-QKV + row-routed + batch router + row-shared | 3 | 160 | 0 | 72.7% (109/150) | 32.69 t/s |
| Hybrid row-QKV + row-routed + batch router + row-shared | 4 | 160 | 0 | 83.0% (122/147) | 32.34 t/s |
| Hybrid row-QKV + row-routed + row-router/shared | 3 | 160 | 0 | 70.6% (108/153) | 31.51 t/s |
| Hybrid row-QKV + row-routed + row-router/shared | 4 | 160 | 0 | 83.0% (122/147) | 31.44 t/s |
| Strict row-QKV/output/router/routed/shared + direct-router + prefix-N | 3 | 160 | 0 | 70.6% (108/153) | 30.38 t/s |
| Historical guarded strict path before row-local compressor fix, N=4 exact fallback | 4 | 160 | 0 | 83.0% (122/147) | 29.54 t/s |
| Historical guarded strict path before row-local compressor fix, N=5 exact fallback | 5 | 160 | 0 | 76.4% (126/165) | 28.10 t/s |
| Row-local compressor strict path | 4 | 160 | 0 | 83.0% (122/147) | 30.53 t/s |
| Row-local compressor strict path + exact output-HC tail batch | 4 | 160 | 0 | 83.0% (122/147) | 32.71 t/s |
| Row-local compressor strict path + output-HC tail batch + fused row-shared default | 4 | 160 | 0 | 83.0% (122/147) | 32.73 t/s |
| Current default clean path + position stats | 4 | 160 | 0 | 83.0% (122/147), pos 1=34/38 2=31/37 3=29/36 4=28/36 | 32.77 t/s |
| Current default clean path + exact Q8 Q/KV rows | 4 | 160 | 0 | 83.0% (122/147), pos 1=34/38 2=31/37 3=29/36 4=28/36 | 33.71 t/s |
| Current default clean path + exact F16 HC-pre rows | 4 | 160 | 0 | 83.0% (122/147), pos 1=34/38 2=31/37 3=29/36 4=28/36 | 34.19 t/s |
| Current default clean path + slotwise row-routed | 4 | 160 | 0 | 83.0% (122/147) | 32.07 t/s |
| Row-local compressor strict path, skip stale ratio-4 batch comp projections | 4 | 160 | 0 | 83.0% (122/147) | 30.15 t/s |
| Row-local compressor strict path, cap raised by env | 5 | 160 | 0 | 76.4% (126/165) | 29.34 t/s |
| Current default clean path, cap raised by env | 5 | 160 | 0 | 76.4% (126/165) | 31.82 t/s |

Current default clean n=1000:

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current default clean path | 3 | 1000 | 0 | 78.7% (702/892) | 31.82 t/s |
| Current default clean path | 4 | 1000 | 0 | 79.1% (759/960) | 32.23 t/s |
| Current default clean path + exact Q8 Q/KV rows | 4 | 1000 | 0 | 79.1% (759/960), pos 1=210/240 2=199/240 3=182/240 4=168/240 | 33.07 t/s |
| Current default clean path + exact F16 HC-pre rows | 4 | 1000 | 0 | 79.1% (759/960), pos 1=210/240 2=199/240 3=182/240 4=168/240 | 33.52 t/s |

Current confidence scheduler n=160, budget 4:

| Threshold | cmp | Acceptance | Generation |
| ---: | ---: | ---: | ---: |
| 0.0 | 0 | 83.0% (122/147) | 32.72 t/s |
| 0.2 | 0 | 83.0% (122/147) | 32.66 t/s |
| 0.4 | 0 | 79.9% (119/149) | 32.39 t/s |
| 0.6 | 0 | 81.4% (114/140) | 31.89 t/s |
| 0.8 | 0 | 90.5% (105/116) | 30.09 t/s |

Timed n=160 runs with `DS4_DSPARK_TIMING=1`:

- Budget 2: decode2 verifier is usually 54-59 ms per two-token block; draft is
  about 9 ms; prefix1 commit is about 1.2-1.8 ms.
- Budget 4: decodeN verifier is usually 108-113 ms per four-token block; draft
  is about 15 ms; prefix-N commit is about 1.3-1.7 ms.
- Strict hybrid budget 3 after prefix-N capture: draft is about 12 ms, verifier
  is about 80-85 ms per three-token block, and prefix commit is about 1.3-1.5 ms.
- Row-local compressor strict budget 4 with exact output-HC tail batching is
  clean but still spends about 150-159 ms per four-token block under the blocking
  stage profiler. Attention is the largest bucket at about 61-65 ms; row-routed
  MoE is the next largest at about 34-37 ms.
- Current exact-F16/Q8-rows strict budget 4 profile
  (`bench-results/dspark_exactf16rows_stage_profile_budget4_n96.err`) averages
  145.83 ms per four-token verifier block under the blocking profiler:
  attention 57.64 ms, row-routed 34.46 ms, row-shared 14.31 ms, row-FFN-pre
  13.15 ms, row-router 11.64 ms, batch-tail 11.69 ms, target-hidden 1.37 ms,
  output head 1.53 ms, readback 0.02 ms. The pre-F16-helper exact-Q8-rows profile
  averaged 151.36 ms total with 61.74 ms attention.

Historical budget-4 interpretation: this table predates the strict-v1 active-5
default and is no longer the "best correct full-length" summary. The current
headline correct path is the active-5 strict-v1 run listed above
(`dspark_footer_nostats_n1000_134020`, 38.63 t/s, `cmp=0`), with older budget-4
rows kept here only to explain which strict subpaths were tried.

### Fast But Unsafe Paths

These use batch verification variants and diverge from the baseline output.

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsafe batch decode-order | 3 | 1000 | 1 | 81.8% (710/868) | 34.43 t/s |
| Unsafe batch decode-order | 4 | 1000 | 1 | 80.3% (762/949) | 35.68 t/s |
| Unsafe batch decode-order | 5 | 1000 | 1 | 75.8% (791/1043) | 33.45 t/s |

Best raw speed: unsafe batch budget 4 at 35.68 t/s. It is not shippable because
it diverges.

### Hybrid Path

Hybrid means exact decode-order attention/cache plus batched FFN/MoE.

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scoped hybrid | 4 | 220 | 0 | 83.7% (169/202) | 31.03 t/s |
| Scoped hybrid | 3 | 1000 | 1 | 80.1% (706/881) | 29.70 t/s |
| Scoped hybrid | 4 | 1000 | 1 | 79.0% (759/961) | 28.89 t/s |
| Scoped hybrid | 5 | 1000 | 1 | 76.6% (793/1035) | 27.11 t/s |
| Scoped hybrid + shared-down/HC disabled | 4 | 1000 | 1 | 80.8% (763/944) | 29.51 t/s |
| Scoped hybrid + HC-norm/shared-down/slotbank disabled | 4 | 1000 | 1 | 81.3% (764/940) | 13.11 t/s |

Important: the n=220 scoped hybrid result was a false sense of safety. At n=1000
it still diverges.

Hybrid exact-prefix sweep, budget 4:

| Case | Exact prefix layers | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid exact-prefix | 0 | 160 | 0 | 83.0% (122/147) | 30.99 t/s |
| Hybrid exact-prefix | 4 | 160 | 1 | 84.8% (123/145) | 29.34 t/s |
| Hybrid exact-prefix | 8 | 160 | 0 | 83.0% (122/147) | 30.09 t/s |
| Hybrid exact-prefix | 12 | 160 | 0 | 83.0% (122/147) | 29.82 t/s |
| Hybrid exact-prefix | 16 | 160 | 0 | 84.8% (123/145) | 27.77 t/s |
| Hybrid exact-prefix | 8 | 1000 | 1 | 79.0% (759/961) | 28.76 t/s |
| Hybrid exact-prefix | 12 | 1000 | 1 | 78.1% (757/969) | 27.92 t/s |
| Hybrid exact-prefix | 16 | 1000 | 0 | 78.6% (758/964) | 27.66 t/s |

Interpretation: making the first 16 layers exact can preserve n=1000 output, but
it is slower than the already-slow exact verifier path. Prefix depth is therefore
not the right speed lever; the next target is the batched FFN/MoE row math and
HC accumulation inside the hybrid verifier.

Hybrid row-exact FFN-pre sweep:

| Case | Budget | Tokens | cmp | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid row-FFN-pre | 2 | 1000 | 0 | 85.1% (630/740) | 30.41 t/s |
| Hybrid row-FFN-pre | 3 | 1000 | 1 | 82.2% (711/865) | 29.65 t/s |
| Hybrid row-FFN-pre | 4 | 160 | 0 | 83.0% (122/147) | 31.02 t/s |
| Hybrid row-FFN-pre | 4 | 1000 | 0 | 78.6% (758/964) | 29.04 t/s |
| Hybrid row-FFN-pre | 5 | 1000 | 1 | 76.1% (791/1040) | 27.16 t/s |
| Hybrid row-router | 4 | 160 | 1 | 84.8% (123/145) | 29.12 t/s |

Interpretation: row-exact FFN-pre/norm can make some long runs safe, including
budget 4, and is faster than exact-prefix K=16. It still does not beat the
no-draft baseline, and correctness is not monotonic across budgets: budgets 3
and 5 still diverge. Row-exact router is not a candidate path in its current
form; it diverged in the short n=160 smoke.

## What We Have Learned

1. Acceptance rate is not the bottleneck.

   Acceptance around 78-82% is common, but exact verification is too expensive.
   The unsafe batch path can get close to baseline speed, so draft quality is
   good enough to be useful if verification/commit state can be fixed.

2. The fast batched verifier's top-token decisions are often plausible, but its
   committed target state drifts.

   Prior audits showed batch tops can agree while committed state diverges. The
   output eventually changes even when acceptance looks healthy.

3. Full exact replay after batch verification is correct but too slow.

   It removes most of the benefit because replay pays target decode cost again.

4. Exact decodeN is correct but dominated by target-layer work.

   Reducing decodeN host readback to one vocab row helped only slightly. It did
   not move budget 4 meaningfully.

5. One hybrid drift source was the tiny shared gate/up SwiGLU fusion.

   Disabling `DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION` globally made a
   short hybrid budget 4 n=220 run byte-match. A scoped graph flag now disables
   that fusion inside the hybrid verifier only. However n=1000 still diverges,
   so there are additional state differences.

6. DSpark main-KV range batching is not the long-run divergence source.

   `DS4_DSPARK_MAIN_KV_BATCH_DISABLE=1` with scoped hybrid budget 4 still
   diverged at n=1000, with the same visible style of output change.

7. Broadly disabling FFN fast paths does not recover correctness.

   Scoped hybrid plus `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` still
   diverged at 29.51 t/s. Disabling HC norm fusion, shared-down/HC fusion, and
   tiny-batch slotbank together still diverged and fell to 13.11 t/s.

8. Stage dumps show the hybrid drift is accumulated state error.

   For the first four-token verifier block at layer 0, hybrid batch rows 0..3
   match exact rows through `hc_ffn_post` to about `1e-7`, and router top-k is
   identical. By layer 3, `hc_ffn_pre` already differs before FFN work starts:
   row deltas were about `1.7e-6`, `5.5e-5`, `2.3e-5`, and `2.5e-5`. That means
   earlier tiny row-wise differences accumulate into later attention/HC state.
   The remaining issue is not a simple wrong router/top-k decision.

9. Full-state hybrid audit confirms logits can look safe while state is not.

   A budget-4 n=80 audit logged matching commit-logit top IDs for every audited
   block, but final HC max deltas ranged roughly `0.3..5.4` and 39-41 layers
   mismatched. First stored raw/attention differences often appeared by layer
   2 or 3.

10. Per-layer post-HC audit localizes the first hybrid drift.

    On a budget-4 n=32 audit, the first mismatching post-HC layer was usually
    layer 3 or 5, with the first block reporting `first=3`, `max=0.000299752`,
    and worst layer 33. A verbose n=4 run showed layer 3 at about `3e-4`, layer
    5 around `1e-3`, then much larger growth around layer 16 (`0.276953`) and
    layer 17 (`0.531644`), with a worst layer near 42. This points at small
    batched FFN/HC differences feeding later attention, not an obvious
    accept/reject boundary bug.

11. Exact-prefix hybrid is a diagnostic, not a production answer.

    `DS4_DSPARK_HYBRID_EXACT_PREFIX_LAYERS=16` is byte-correct at n=1000, but
    only reaches 27.66 t/s. Lower values such as 8 and 12 still diverge at
    n=1000. This reinforces that a production verifier needs exact
    cache/accumulation semantics with selectively batched row-independent work,
    not a broad prefix fallback.

12. FFN-stage audit identifies the first measurable seed in layer 0 FFN math.

    With `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS=1e-8` and
    `DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1`, layer 0 attention-HC is still clean,
    but layer 0 post-HC already differs by about `3.57628e-7`. FFN substages at
    layer 0 show `ffn-pre=1.49012e-8`, `ffn-norm=8.9407e-8`,
    `routed-out=2.38419e-7`, and `shared-out=2.38419e-7`. Layer 1 attention-HC
    then differs by about `1.21444e-6`, and the error grows through later
    layers. This points to batch-vs-row FFN/HC numerics seeding drift before
    attention/cache mutation becomes visibly different.

13. Existing broad reference toggles do not fix the useful path.

    `DS4_METAL_DISABLE_HC_NORM_FUSION=1` did not change the layer-0 seed.
    `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` reduced some later worst-stage
    deltas but kept the layer-0 seed. `DS4_METAL_DISABLE_HC_FUSION=1` was not a
    useful DSpark verifier run in this configuration: the short n=4 run produced
    no hybrid audit summaries and acceptance collapsed to 0%.

14. Row-exact FFN-pre moves, but does not eliminate, the hybrid problem.

    With `DS4_DSPARK_HYBRID_ROW_FFN_PRE=1`, layer 0 `ffn-pre` no longer crosses
    `1e-8`, but routed/shared outputs still differ at about `2.38419e-7`, and
    layer 0 post-HC remains around `1.78814e-7`. Budget 4 n=1000 becomes
    byte-correct at 29.04 t/s, but budgets 3 and 5 still diverge. The tail
    routed/shared/down/HC math is therefore also part of commit-safe state.

15. Row-exact router did not improve the candidate path.

    `DS4_DSPARK_HYBRID_ROW_ROUTER=1` reduced the layer-0 routed-output audit
    seed from about `2.38419e-7` to `1.19209e-7`, but the n=160 budget-4 run
    diverged and only reached 29.12 t/s. Treat this as an audit probe, not a
    correctness fix.

16. Row-exact routed plus row-exact shared defines the current exactness
    boundary.

    `DS4_DSPARK_HYBRID_ROW_ROUTED=1` plus
    `DS4_DSPARK_HYBRID_ROW_SHARED=1` made all FFN-stage audit summaries clean
    at `eps=1e-8`: attention-HC, post-HC, FFN-pre, FFN-norm, routed-out, and
    shared-out all reported `first=-1 max=0 rms=0` through 43 layers. It is
    byte-correct at n=160 and n=1000, but too slow: budget 4 n=1000 reached
    26.26 t/s versus 35.46 t/s baseline; budget 2 n=1000 reached 30.36 t/s.
    This proves the corruption is in the batched FFN tail boundary, not in the
    DSpark draft model.

17. Tiny-batch selected-id and row-kernel experiments did not remove the hybrid
    drift.

    The Flash resident tiny-batch slot-bank path now uses true expert ids when
    the slot bank is identity-resident. The diagnostic
    `DS4_DSPARK_TINY_BATCH_ROW_KERNEL=1` also encodes the single-row native
    MXFP4 kernels inside one command buffer. Neither made row-shared-only or
    row-router+row-shared clean: routed/shared and later attention state still
    diverge. Do not spend more time on selected-id remapping as the primary
    explanation; the remaining production path likely needs a dedicated N<=5
    exact microbatch verifier that preserves the row-routed and row-shared
    mutation/accumulation contract while batching only proven row-independent
    work.

18. Fast batch verifier N=3..5 proves logits can agree while commit state is
    unusable.

    With `DS4_DSPARK_BATCH_VERIFY=1`, `DS4_DSPARK_BATCH_APPROX_STATE=1`,
    `DS4_DSPARK_BATCH_DECODE_ORDER=1`, `DS4_DSPARK_DECODEN_DISABLE=1`, and
    `DS4_DSPARK_DECODE2_DISABLE=1`, an n=16 state-audit run exact-replayed for
    correctness (`cmp=0`) but showed the unsafe batch state is not committable:
    the first audited block had matching batch/exact top IDs for accepted rows,
    yet `decode_current_hc row=3 max=148.405 rms=5.6961`. Later blocks showed
    `decode_hc_max` around `64..164`. DSpark target-hidden and draft KV cache
    rows also differed (`dspark_hidden max` up to about `1.20`, `dspark_kv max`
    up to `0.25`). This makes the bug concrete: the batch verifier is useful for
    approximate decisions, not for direct state commit.

19. Row-routed and row-shared are both required for exact hybrid state.

    A fresh n=4 stage sweep showed `ROW_ROUTED=1` alone still has post-HC drift
    (`final_hc_max=0.864929`), and `ROW_SHARED=1` alone also drifts
    (`final_hc_max=0.629105`). Only `ROW_ROUTED=1` plus `ROW_SHARED=1` produced
    all-zero audit summaries for post-HC, FFN-pre, FFN-norm, routed-out,
    shared-out, commit logits, target cache state, and DSpark target-hidden.
    This is the production semantic boundary to optimize.

20. Batched router selection is not safe inside the clean boundary.

    Added `DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1` to test
    FFN-pre row-exact + batched router + row-routed MoE + row-shared expert. The
    n=4 stage audit failed: `final_hc_max=0.772018`, commit-logit max
    `0.209938`, and post-HC/FFN stages diverged again. So the row-exact boundary
    includes router logits/top-k/weights as well as routed MoE and shared expert.
    Do not optimize by batching router selection unless the router kernel itself
    is made row-order exact.

21. First-divergence audit now points at the earliest material culprit.

    The unsafe hybrid n=4 run with `DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1` and
    `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS=1e-5` reports:
    `stage=ffn-norm layer=2 row=2 idx=1443 delta=1.11163e-05`. Later attention
    mismatch is therefore an echo of prior FFN drift, not the first cause. The
    clean row-routed plus row-shared run with `eps=1e-8` reports
    `stage=clean`, `max=0`, and commit-logit `max=0`.

22. Fused row-shared gate/up is byte-clean but not enough by itself.

    Initially added `DS4_DSPARK_HYBRID_ROW_SHARED_FUSED=1`, which uses the
    existing fused single-row shared gate/up/SwiGLU kernel inside the row-shared
    exact boundary. It stayed byte-clean (`stage=clean`, commit-logit `max=0`)
    and matched the baseline output. In the current tree this is default-on, with
    `DS4_DSPARK_HYBRID_ROW_SHARED_FUSED_DISABLE=1` as the A/B escape hatch. The
    current N=4 strict smoke is roughly 32.7 t/s, with a prior same-shape sample
    at 32.85 t/s. The remaining win still needs a fused row-order verifier kernel
    across the exact boundary, not just the shared gate/up subpath.

23. Prefix-N commit is now enabled for the clean hybrid boundary.

    The row-routed plus row-shared hybrid path now captures prefix frontiers for
    accepted lengths 1..4 and commits partial accepts with `decodeN-prefix`
    instead of exact replay. It is gated to the byte-clean boundary:
    `DS4_DSPARK_HYBRID_ROW_ROUTED=1`, `DS4_DSPARK_HYBRID_ROW_SHARED=1`, and no
    `DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1`. On n=160 budget 5, this kept
    output byte-identical to baseline and improved 24.76 t/s -> 27.48 t/s by
    replacing 10 replay commits with prefix commits. It is still below the
    36.51 t/s no-draft baseline because verifier cost rose to about 143 ms per
    block.

24. Current correct static-budget sweep on n=160.

    All rows below matched `bench-results/baseline_after_default_patch_n160.out`
    exactly:

    | Budget | Verifier path | Acceptance | Generation |
    | --- | --- | --- | --- |
    | baseline | no draft | n/a | 36.51 t/s |
    | 1 | sequential DSpark verify | 81.8% (72/88) | 31.63 t/s |
    | 2 | exact decode2 | 77.6% (97/125) | 30.47 t/s |
    | 3 | clean hybrid + prefix-N | 70.6% (108/153) | 29.26 t/s |
    | 4 | clean hybrid + prefix-N | 83.0% (122/147) | 29.04 t/s |
    | 5 | clean hybrid + prefix-N | 76.4% (126/165) | 27.48 t/s |

    Budget 1/2 being best is the evidence that scheduler tuning cannot recover
    the desired DSpark speedup on this implementation. The verifier must become
    materially cheaper than one normal target decode per verified token.

25. Confidence is not reliable enough for adaptive prefix capture.

    A budget-3 n=160 run with `DS4_DSPARK_CONF_LOG=1` showed partial blocks with
    high confidence minima (`0.858`, `0.978`) and full-accept blocks with low
    minima (`0.286`, `0.413`, `0.496`). A threshold on the minimum confidence
    either misses partial accepts or wastes captures on many full accepts. Do not
    expect confidence-gated prefix capture to close the speed gap.

26. Routed-MoE slotwise row diagnostic.

    Added `DS4_DSPARK_HYBRID_ROW_ROUTED_SLOTWISE=1` as an opt-in experiment for
    the clean hybrid boundary. It keeps routed MoE row-exact but uses the
    slotwise resident MoE helper in Flash identity-slot mode, where slot id and
    expert id are the same. This must still be audited before any performance
    interpretation; it is only meant to answer whether the single-row slotwise
    kernel family can reduce row-routed verifier cost without changing state.
    Result so far: the n=4 audit was clean through post-HC, commit logits, final
    HC, raw KV, compressor state, DSpark hidden, and DSpark KV, but
    `decode-current-hc` scratch differed because that optional check compares
    the exact replay transient `cur_hc`, not the committed hybrid frontier. The
    n=8 stage profile was slower than the generic row-routed path:
    `row_routed=45.361 ms` vs the previous `40.974 ms`, so this is not a speed
    lever.

27. Batched attention with row-exact output still diverges.

    Added `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=1` to pair
    `DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1` with row-exact inverse RoPE,
    output projection, and HC expansion. This reduced the commit-logit delta
    compared with the fully batched output path, but it did not make the verifier
    commit-safe. The n=4 audit still reported first state drift at
    `attn-kv layer=2` and final-HC drift (`final_hc_max=0.43399`). That means
    the remaining non-exactness is upstream of output commit: batched
    HC/Q/KV/compressor/index arithmetic or cache update order, not just the
    output projection.

28. Batched attention with row-exact Q/KV is the current best safe boundary.

    Added `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1`. In the hybrid batch-attention
    experiment it computes HC-pre, Q/KV projection, RoPE, FP8 KV quantization,
    and raw-KV store with the same single-row decode kernels, then resumes the
    existing batched attention/helper path. This collapsed the raw-KV drift in
    short audits and reduced commit-logit deltas to near-noise levels. The n=160
    smoke with budget 4 plus row-output byte-matched but was only 28.90 t/s, so
    row-output is not part of the fastest split.

    Best n=160 smoke so far:

    ```bash
    DS4_DSPARK_BATCH_VERIFY=1 \
    DS4_DSPARK_DECODEN_ATTN_FFN_BATCH=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1 \
    DS4_DSPARK_HYBRID_ROW_FFN_PRE=1 \
    DS4_DSPARK_HYBRID_ROW_ROUTED=1 \
    DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1 \
    ./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
      --draft-verify 3 --temp 0 --nothink -n 160 \
      -p "Make a game of Space Invader in Pygame" --resident -c 4096
    ```

    Result: byte-match against `baseline_after_default_patch_n160.out`,
    generation 33.14 t/s, acceptance 72.7% (109/150). This is still below the
    refreshed no-draft baseline (36.51 t/s in the same n=160 artifact), but it is
    materially faster than exact decodeN and the previous clean hybrid boundary.
    The aggressive variant that batched routed MoE reached 33.75 t/s but diverged
    visibly (`ENEMY_SPEED = 2` became `1`), so routed MoE remains inside the exact
    row boundary for now. Budget 5 and forced budget 2 also diverged in this split;
    budget 3 is the current safest performance point.

29. DSpark KV audit confirms target-state drift is the upstream problem.

    Added `DS4_DSPARK_HYBRID_STATE_AUDIT_DSPARK_KV=1` for the decodeN hybrid
    path. With the fastest n=160 split, the short audit now shows DSpark draft KV
    drift, but only after target state has already diverged:
    `stage=attn-HC layer=3` or `first_stage=raw-kv first_layer=3` in the same
    blocks, followed by `dspark_kv max=0.25`. Adding row-exact attention output
    moves the first HC-stage mismatch into `ffn-norm layer=3/4`, but final HC,
    target-hidden, raw rows, and DSpark KV still drift. The stricter reference
    split with row-QKV, row-output, row-router, row-routed, and row-shared is
    clean for commit logits, final HC, DSpark target-hidden, and DSpark KV in the
    n=16 audit; only one block reported a tiny `attn-score` delta
    (`1.71661e-05`) at the `1e-5` threshold. This makes the production target
    explicit: preserve full row-exact target mutation semantics, then remove the
    dispatch overhead with a dedicated N<=5 microbatch verifier.

    The strict reference split also byte-matches n=160 without audit at
    29.49 t/s, acceptance 70.6% (108/153). Its blocking stage profile for n=3 is
    about 139 ms per verifier block: attention 55 ms, row-routed 28 ms,
    row-shared 14 ms, row-router 14 ms, row-FFN-pre 12 ms, batch-tail 12 ms,
    target-hidden 1.4 ms, output head 1.5 ms. Ablations show both row-shared and
    row-router remain required in this diagnostic composition: dropping row-shared
    while keeping row-router exact still drifts, and batching router while keeping
    row-shared exact also drifts. The original opt-in row-shared fused experiment
    did not improve this older strict reference split: 29.44 t/s vs 29.49 t/s.

30. Row-router direct-write removes copy dispatches and is safe.

    `metal_graph_encode_layer_router_exact_rows()` now writes exact single-row
    router logits, probabilities, selected ids, and weights directly into the
    verifier batch rows instead of computing into scratch tensors and copying
    four tensors back per row. The opt-out is
    `DS4_DSPARK_HYBRID_ROW_ROUTER_DIRECT_DISABLE=1`. The strict n=16 DSpark KV
    audit remains state-clean with the same tiny `attn-score layer=40` blip as
    before. The strict n=160 reference improved from 29.49 t/s to 29.98 t/s,
    still byte-matching baseline output with 70.6% acceptance (108/153). Blocking
    stage profile row-router time dropped from about 13.5-13.8 ms to about
    11.6-11.7 ms per n=3 verifier block. A separate ablation that batched routed
    MoE while keeping row-QKV, row-output, row-router, and row-shared exact still
    drifted (`first_stage=raw-kv first_layer=3`, `final_hc_max` up to about
    0.97), so row-routed exactness remains required.

31. Prefix-N capture is now wired into the strict decode-order hybrid path.

    `metal_graph_encode_layer_attention_batch()` now accepts a prefix capture
    count and snapshots attention/index compressor state after each decode-order
    row. The caller only enables this for the strict state-clean composition:
    row-QKV, row-output, row-router, row-routed, and row-shared exactness, with no
    batched-router/row-routed shortcut. Disable it with
    `DS4_DSPARK_HYBRID_BATCH_ATTN_PREFIXN_DISABLE=1` for A/B checks. The n=16
    budget-3 DSpark-KV audit stays clean for final HC, DSpark hidden, DSpark KV,
    and raw rows, with only the known tiny `attn-score layer=40` epsilon blip.
    The n=160 speed check byte-matches baseline at 30.54 t/s in
    `bench-results/dspark_hybrid_guarded_budget3_current_after_attn_audit_n160.err`,
    improving the
    strict direct-router reference from 29.98 t/s but still below the 36.51 t/s
    no-draft baseline. A row-output ablation is not state-clean: it drifts around
    layer 2-3 and corrupts DSpark hidden/KV, so row-output exactness remains
    required.

32. Superseded: decode-order hybrid N=4/5 was guarded off before the row-local
    compressor fix.

    The N=4 DSpark-KV audit found a full-accept block
    (`drafted=4 committed=4`) with first internal drift at `ffn-norm layer=7
    row=2`, followed by raw-KV/frontier drift (`first_stage=raw-kv first_layer=6`)
    and DSpark hidden/KV differences. The N=5 audit shows the same row-2/layer-7
    pattern and the n=160 output visibly swaps `RED` and `GREEN`. This section is
    retained as historical evidence. The later row-local compressor/indexer
    mutation fix supersedes it: N=4 and N=5 are now default-clean, though active
    4 remains faster on the current long-prompt sweep.

33. Tiny-batch row-routed reuse is faster but not state-clean.

    Added `DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1` as an opt-in diagnostic
    in the strict row-routed verifier helper. It calls the existing
    `ds4_gpu_routed_moe_banked_batch_tensor()` path and forces the Metal
    row-exact tiny-batch encoder. The short N=3 DSpark-KV audit is not clean:
    first drift appears at `stage=attn-HC layer=3` / `first_stage=raw-kv
    first_layer=3`, with `final_hc_max` up to about `0.97` and DSpark hidden/KV
    differences. Stage profiling shows why it was tempting: row-routed time drops
    from about 29 ms to about 23 ms per N=3 verifier block, and total blocking
    verifier time drops from about 137 ms to about 132 ms. Do not use it for
    production commits. The real next MoE optimization would need a multirow
    wrapper around the production single-token native route path, not the decode2
    row-exact batch encoder.

34. Attention-substage audit narrows N=4 drift to attention-heads.

    Added `DS4_DSPARK_HYBRID_ATTN_STAGE_AUDIT=1`. When combined with
    `DS4_DSPARK_HYBRID_LAYER_HC_AUDIT=1`, the hybrid-vs-exact audit now records
    Q-rope, KV/cache row, attention heads, attention output, attention HC, and
    optional FFN stages before printing the chronological first divergence.

    Latest diagnostic artifact:

    ```text
    bench-results/dspark_attn_substage_audit_budget4_n24_row_fused_kv_store.err
    ```

    Command shape:

    ```bash
    DS4_AGENT_ALLOW_BACKEND_STATS=1 \
    DS4_DSPARK_BATCH_VERIFY=1 \
    DS4_DSPARK_DECODEN_ATTN_FFN_BATCH=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=1 \
    DS4_DSPARK_HYBRID_ROW_ROUTER=1 \
    DS4_DSPARK_HYBRID_ROW_ROUTED=1 \
    DS4_DSPARK_HYBRID_ROW_SHARED=1 \
    DS4_DSPARK_HYBRID_LAYER_HC_AUDIT=1 \
    DS4_DSPARK_HYBRID_ATTN_STAGE_AUDIT=1 \
    DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1 \
    DS4_DSPARK_HYBRID_STATE_AUDIT=1 \
    DS4_DSPARK_HYBRID_STATE_AUDIT_DSPARK_KV=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ALLOW_UNSAFE_N_GT3=1 \
    ./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
      --draft-verify 4 --temp 0 --nothink -n 24 \
      -p "$PROMPT" --resident -c 4096
    ```

    Result: several blocks are clean, but the first bad full block reports
    `stage=attn-heads layer=5 row=2` while `q-rope` and `kv-cache-row` are clean.
    The companion state summary then reports raw/frontier drift later
    (`first_stage=raw-kv first_layer=6`) and DSpark hidden/KV differences. This
    means the initial correctness break is inside attention-head computation or
    the attention/compressor frontier read by that computation, before FFN/MoE.

    A/B note: deferring speculative raw-KV visibility out of the row-QKV prepass
    is necessary hygiene but not sufficient. The current code keeps FP8 KV
    preparation row-local and uses the fused `metal_graph_decode_kv_store()` at the
    decode-ordered row attention point for row-QKV mode; the first divergence is
    still `attn-heads`. The production fix should therefore be a dedicated N<=5
    exact microbatch attention verifier that preserves cache/compressor mutation
    and attention accumulation order, while batching only operations proven
    row-independent.

35. Row-local compressor/indexer mutation makes strict N=4 audit clean.

    Follow-up patch split `attn-heads` into `attn-heads-raw` and post-inverse-RoPE
    `attn-heads`. That proved the first N=4 drift was already in raw attention
    output. The root cause was not inverse-RoPE or FFN/MoE: in decode-order mode,
    Q/KV was row-local but the compressed-attention and indexer state updates still
    ran across the whole tiny block before row attention consumed the frontier.

    Current code keeps the projection matmuls batched where safe, but moves
    compressor/indexer state mutation and prefix snapshots into the per-row
    attention loop for speculative decode-order mode. Ratio-4 layers recompute the
    compressor rows into singleton scratch because `batch_comp_kv/sc` is reused for
    attention and indexer projections.

    Latest diagnostic artifact:

    ```text
    bench-results/dspark_attn_rowcompress_audit_budget4_n24.err
    ```

    Result: all audited N=4 blocks report `stage=clean`; companion state audits
    report `first_stage=clean`, `mismatch_layers=0`, and max drift around
    `4.76837e-07`. Default
    `DS4_DSPARK_HYBRID_BATCH_ATTN_MAX_CLEAN` is now `5`.

    N=5 is also audited clean
    (`bench-results/dspark_attn_rowcompress_audit_budget5_n24.err`) and now uses
    the strict hybrid path by default. After the row-exact tiny-batch routed MoE
    change, the latest long sweep byte-matches baseline at 39.04 t/s for n=1000
    and 36.48 t/s for n=4000, making active 5 the current default.

36. Negative retests after the row-local compressor fix.

    The attention-order fix did not make the older shortcuts safe:

    - Batch QKV plus row-output still diverges from layer 0 and grows state drift
      by later layers (`bench-results/dspark_batchqkv_rowcompress_audit_budget4_n24.err`).
    - Row-HC exact plus batch-QKV still diverges at `attn-heads-raw layer=0` and
      grows raw-KV/final-HC/DSpark-KV deltas
      (`bench-results/dspark_rowhc_batchqkv_audit_budget4_n24.err`).
    - Batched router feeding row-routed MoE is still unsafe
      (`bench-results/dspark_batch_router_routed_audit_after_attnfix_budget4_n24.err`).
    - `DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1` logs the resident
      tiny-batch path, but remains unsafe
      (`bench-results/dspark_row_routed_batch_row_exact_audit_budget4_n24.err`).
    - Removing now-unused ratio-4 batch compressor projections did not improve
      speed in the n=160 smoke; it stayed byte-correct but measured 30.15 t/s.

37. Exact multirow F16 HC-pre and Q8 Q/KV projections are the current best clean
    point.

    Added `ds4_gpu_matmul_q8_0_rows_exact_tensor()` and
    `ds4_gpu_matmul_f16_rows_exact_tensor()`. Row-QKV uses them by default for
    the HC-pre F16 projection plus the `attn_q_a`, `attn_kv`, and `attn_q_b` Q8
    projections. This keeps the exact single-row reduction kernels but dispatches
    the N<=5 rows as a tiny Y dimension, reducing command overhead without
    changing cache mutation order. Disable them with
    `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_F16_ROWS_DISABLE=1` and
    `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_QKV_ROWS_DISABLE=1` for A/B
    diagnosis.

    Artifacts:

    ```text
    bench-results/dspark_default_exactq8rows_budget4_n160.err
    bench-results/dspark_rowhc_exactq8rows_budget4_n1000.err
    bench-results/dspark_default_exactq8rows_audit_budget4_n24.err
    bench-results/dspark_exactf16rows_budget4_n160.err
    bench-results/dspark_exactf16rows_budget4_n1000.err
    bench-results/dspark_exactf16rows_audit_budget4_n24.err
    ```

    Results: default N=4 n=160 is byte-clean at 34.19 t/s with 83.0% acceptance.
    The n=1000 run is byte-clean at 33.52 t/s with 79.1% acceptance, still below
    the refreshed no-draft baseline of 35.46 t/s. The short audit reports
    `stage=clean`, `mismatch_layers=0`, `final_hc_max=0`, `decode_hc_max=0`,
    `dspark_hidden max=0`, and `dspark_kv max=0`.

38. Latest safe path is nearly baseline; batch-QKV is fast but still unsafe.

    Added exact row wrappers for DSpark row-router logits and selector:
    `DS4_DSPARK_HYBRID_ROW_ROUTER_LOGITS_ROWS_DISABLE=1` and
    `DS4_DSPARK_HYBRID_ROW_ROUTER_SELECT_ROWS_DISABLE=1` restore the older
    per-row calls. The exact F16 logits rows path is a small aggregate win; the
    selector rows wrapper is correctness-neutral and mostly timing noise. Added
    `DS4_DSPARK_HYBRID_MARGIN_GUARD=<float>` and
    `DS4_DSPARK_HYBRID_READ_ALL_LOGITS=1` so the decodeN hybrid path can read
    all N row logits and exact-replay accepted prefixes whose accepted-row
    top-2 margins are below a threshold.

    Current safe command:

    ```bash
    DS4_AGENT_ALLOW_BACKEND_STATS=1 \
    DS4_DSPARK_BATCH_VERIFY=1 \
    DS4_DSPARK_DECODEN_ATTN_FFN_BATCH=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1 \
    DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=1 \
    DS4_DSPARK_HYBRID_ROW_ROUTER=1 \
    DS4_DSPARK_HYBRID_ROW_ROUTED=1 \
    DS4_DSPARK_HYBRID_ROW_SHARED=1 \
    ./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
      --draft-verify 4 --temp 0 --nothink -n 1000 \
      -p "$PROMPT" --resident -c 4096
    ```

    Artifacts:

    ```text
    bench-results/baseline_after_router_rows_n1000.err
    bench-results/dspark_router_rows_budget2_n1000.err
    bench-results/dspark_router_rows_budget3_n1000.err
    bench-results/dspark_router_rows_budget4_n1000.err
    bench-results/dspark_router_rows_budget5_n1000.err
    bench-results/dspark_safe_after_guard_patch_budget4_n1000.err
    bench-results/dspark_batch_qkv_current_budget4_n1000.err
    bench-results/dspark_batch_qkv_guard0.5_budget4_n1000.err
    bench-results/dspark_batch_qkv_guard1.0_budget4_n1000.err
    bench-results/dspark_batch_qkv_guard2.0_budget4_n1000.err
    bench-results/dspark_batch_qkv_guard4.0_budget4_n1000.err
    ```

    Current numbers on the local Flash resident prompt:

    - no-draft baseline: 35.34 t/s, byte-identical to the prior refreshed baseline.
    - safe DSpark budget 2: 30.40 t/s, cmp clean.
    - safe DSpark budget 3: 34.32 t/s, cmp clean.
    - safe DSpark budget 4: 35.28 t/s best observed in the sweep, cmp clean;
      a later validation after the guard patch measured 35.08 t/s, also cmp clean.
    - safe DSpark budget 5: 28.04 t/s, cmp clean.
    - batch-QKV diagnostic: 36.15 t/s, but cmp failed with real semantic code
      differences.
    - batch-QKV plus margin guard 0.5, 1.0, 2.0, 4.0: all still cmp failed;
      higher thresholds also erased the speed win.

    Conclusion: margin-only fallback is not enough for batch-QKV because wrong
    accepts can have comfortable approximate margins. The exact safe path is now
    within run noise of no-draft baseline, but DeepSeek-style gains require a
    dedicated N<=5 verifier attention/compressor path that preserves exact row
    mutation and accumulation order while eliminating dispatch overhead.

## Failure Examples

Unsafe batch budget 4 first visible n=1000 mismatch:

```diff
-RED = (255, 0, 0)
 GREEN = (0, 255, 0)
+RED = (255, 0, 0)
```

Scoped hybrid budget 4 first visible n=1000 mismatch:

```diff
     def shoot(self):
-        # Random shooting
-        if random.random() < 0.02:  # 2% chance per frame
+        if self.alive and random.random() < 0.02:  # 2% chance to shoot
```

These are real semantic generation changes, not harmless formatting-only
differences.

## Active Env Flags

Current default strict-v1 path:

```bash
./ds4 ... --draft dspark --draft-path "$DRAFT"
```

Lower budgets are diagnostics only:

```bash
./ds4 ... --draft dspark --draft-path "$DRAFT" --draft-verify 2
./ds4 ... --draft dspark --draft-path "$DRAFT" --draft-verify 3
./ds4 ... --draft dspark --draft-path "$DRAFT" --draft-verify 4
```

Unsafe batch path, archive/future Mode-B diagnostics only:

```bash
DS4_DSPARK_SEQUENTIAL_VERIFY=1 \
DS4_DSPARK_BATCH_VERIFY=1 \
DS4_DSPARK_BATCH_APPROX_STATE=1 \
DS4_DSPARK_BATCH_DECODE_ORDER=1 \
./ds4 ... --draft dspark --draft-path "$DRAFT" --draft-verify 4
```

Forced hybrid diagnostic path, not needed for normal strict-v1:

```bash
DS4_DSPARK_DECODEN_ATTN_FFN_BATCH=1 \
./ds4 ... --draft dspark --draft-path "$DRAFT"
```

Useful diagnostics:

```bash
DS4_DSPARK_TIMING=1
DS4_DSPARK_BATCH_AUDIT=1
DS4_DSPARK_BATCH_STATE_AUDIT=1
DS4_DSPARK_STATE_AUDIT_EPS=<float>
DS4_DSPARK_BATCH_STATE_AUDIT_DSPARK_KV=1
DS4_DSPARK_BATCH_STATE_AUDIT_DECODE_HC=1
DS4_DSPARK_SPEC_LOG=1
DS4_DSPARK_BATCH_MARGIN_GUARD=<float>
DS4_DSPARK_BATCH_EXACT_EVERY=<N>
DS4_DSPARK_MAIN_KV_BATCH_DISABLE=1
DS4_DSPARK_VERIFY_SPLIT_LAYERS=<N>
DS4_DSPARK_HYBRID_STATE_AUDIT=1
DS4_DSPARK_HYBRID_LAYER_HC_AUDIT=1
DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_VERBOSE=1
DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS=<float>
DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1
DS4_DSPARK_HYBRID_STAGE_PROFILE=1
DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1
DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1
DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=1
DS4_DSPARK_HYBRID_ROW_OUTPUT_BATCH_ROPE_DISABLE=1
DS4_DSPARK_HYBRID_ROW_OUTPUT_LOW_ROWS_DISABLE=1
DS4_DSPARK_HYBRID_ROW_OUTPUT_BATCH_HC_DISABLE=1
DS4_DSPARK_HYBRID_BATCH_ATTN_PREFIXN_DISABLE=1
DS4_DSPARK_HYBRID_BATCH_ATTN_MAX_CLEAN=<2..5>
DS4_DSPARK_HYBRID_BATCH_ATTN_ALLOW_UNSAFE_N_GT3=1
DS4_DSPARK_HYBRID_EXACT_PREFIX_LAYERS=<N>
DS4_DSPARK_HYBRID_ROW_FFN_PRE=1
DS4_DSPARK_HYBRID_ROW_ROUTER=1
DS4_DSPARK_HYBRID_ROW_ROUTER_LOGITS_ROWS_DISABLE=1
DS4_DSPARK_HYBRID_ROW_ROUTER_SELECT_ROWS_DISABLE=1
DS4_DSPARK_HYBRID_ROW_SHARED=1
DS4_DSPARK_HYBRID_ROW_SHARED_FUSED_DISABLE=1
DS4_DSPARK_HYBRID_ROW_ROUTED=1
DS4_DSPARK_HYBRID_ROW_ROUTED_SLOTWISE=1
DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1
DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1
DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_BATCH_QKV=1
DS4_DSPARK_HYBRID_MARGIN_GUARD=<float>
DS4_DSPARK_HYBRID_READ_ALL_LOGITS=1
DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_F16_ROWS_DISABLE=1
DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_QKV_ROWS_DISABLE=1
DS4_DSPARK_HYBRID_INDEX_COMP_ROWS=1
DS4_DSPARK_TINY_BATCH_ROW_KERNEL=1
DS4_MTP_SIDECAR_BATCH_SLOTBANK_ROW_EXACT=1
DS4_DSPARK_DECODEN_DISABLE=1
DS4_DSPARK_DECODE_N_DISABLE=1
DS4_DSPARK_DECODE2_DISABLE=1
DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1
DS4_DSPARK_DECODEN_PREFIX1_DISABLE=1
```

## Advice Wanted

Main question:

How do we reduce the cost of the already commit-safe strict-v1 verifier without
changing its byte-identical greedy contract?

Specific asks:

1. Design `strict_v2` attention around the current clean row-local contract:
   same row order, same arithmetic, same cache/frontier visibility, fewer
   dispatches, fewer row-view tensors, fewer copy-back kernels, and persistent
   verifier workspace.
2. Add a non-blocking dispatch census so we can distinguish command/object churn
   from actual math before writing more kernels.
3. Propose row-offset attention APIs for N<=5 that avoid hot-loop row-view/free
   patterns while preserving exact compressor/indexer mutation and prefix
   capture.
4. After attention, design exact-preserving routed MoE v2 that recovers more of
   the `sum6` sizing benefit while keeping separate down outputs and exact
   ordered FP32 add.
5. Keep Mode B and rejected shortcuts as diagnostics only for now. Do not
   recommend broad batch-canonical work as the next patch.

## Suggested Next Experiments

1. Patch A: freeze and name current default.

   Add or confirm a visible `strict_v1` implementation name and log line, with
   no behavior change. Gate with n=160/n=1000/n=4000 `cmp=0` plus block-1 and
   block-501 clean audits.

2. Patch B: add non-blocking dispatch census.

   Count attention pre-dispatches, compressor projection dispatches, indexer
   projection dispatches, frontier mutation dispatches, attention head
   dispatches, attention output dispatches, router, routed MoE, shared, tail,
   and total dispatches without fenced timing.

3. Patch C: row-offset attention APIs.

   Refactor the strict attention verifier to pass base tensor plus row offsets
   instead of allocating/freeing row views in the hot loop. Keep exact row-local
   compressor/indexer behavior and exact prefix capture.

4. Patch D: strict-v2 attention microbatch.

   Build a DSpark-specific N<=5 attention path that preserves decode-order
   mutation and exact row-local compressor/indexer/indexer-Q behavior. Success
   requires n=160/n=1000/n=4000 `cmp=0` and clean `1e-8` stage audits.

5. Patch E: exact routed MoE v2.

   Reduce dispatch around row/expert work while keeping strict slot order. Do
   not promote unordered direct `sum6`. The Flash Q2 ordered direct path is the
   current exception: it keeps six independent slot accumulators and writes the
   exact slot0..slot5 FP32 ordered sum, with
   `DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1` available to compare it
   against the conservative separate-down path.

Historical/rejected notes:

6. Do not reuse the existing batch attention path for production verification.

   `DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1` routes the hybrid verifier's
   attention stage through `metal_graph_encode_layer_attention_batch()` while
   preserving decode-order cache/compressor updates and keeping FFN/router/MoE
   row-exact. It is useful as a reproducer, but it is not enough by itself.
   After row-local compressor/indexer mutation, the strict row-QKV/output plus
   row-router/routed/shared split is audit-clean through N=5 by default. Batching
   the exact output-HC tail and enabling fused
   row-shared by default improved the clean N=4 smoke to about 32.7 t/s, but the
   problem is still cost: the verifier spends about 150-159 ms per four-token
   block under the blocking profiler. The next attention win needs new code that
   preserves this row order with less dispatch overhead.

7. Dedicated exact N<=5 attention microbatch is the next target.

   The first audit with `DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1` plus
   row-exact routed/shared experts diverged at `stage=attn-HC layer=0 row=0`
   even before FFN, then grew to commit-logit `max=0.738216` and frontier-state
   `first_stage=attn-kv first_layer=2`. The row-output variant reduced the
   commit-logit delta but still reported `first_stage=attn-kv first_layer=2` and
   `final_hc_max=0.43399`. Row-QKV collapsed enough drift to make budget 3/4 n=160
   output byte-match, but the new DSpark KV audit confirms that the faster split
   still mutates state differently. Build a dedicated verifier attention path
   that keeps exact single-row decode arithmetic and cache mutation order, then
   removes overhead by taking row offsets directly and processing N<=5 rows inside
   one scoped command sequence/kernel family.

8. Dedicated N<=5 exact microbatch verifier is the likely production path.

   The production verifier should preserve exact cache mutation and accumulation
   order, but batch operations only after the audit proves they are
   row-independent. The current evidence says exact multirow F16/Q8 projections
   are safe; prefill-style batch QKV, batch router, and tiny-batch routed MoE are
   not. Keep the stage audit close while moving one operation at a time out of
   the row loop.

## Latest Update: 2026-06-28

Correctness status improved: the strict decode-order hybrid verifier is now
byte-clean on the long n=4000 Space Invader prompt against the no-draft greedy
baseline.

Two late-drift bugs were fixed:

1. Ratio-128 attention compressor rows

   The decode-order verifier previously only used row-local exact compressor
   projections for ratio-4 layers. Ratio-128 compressed-cache layers still used
   the batched F16 compressor projection. Tight audit caught tiny `attn-kv`
   mismatches that later amplified. The verifier now computes compressor
   projections row-locally for all compressed layers in decode-order mode.

2. Ratio-4 indexer query/weight projections

   After the ratio-128 fix, n=4000 still had a late block-local divergence. The
   detailed block-501 audit localized it to `attn-heads-raw` at row 3, with
   `attn-norm`, Q, and KV clean. The remaining difference was selected indexed
   compressed rows: hybrid used batched `batch_indexer_q` /
   `batch_indexer_weights`, while exact decode used row-local F16 projections
   from `qr_norm` and `attn_norm`. The verifier now recomputes indexer Q/weights
   row-locally before scoring/top-k in decode-order mode and no longer
   materializes the unused dense top-k mask on that sparse indexed path.

Validation after the fix:

| Run | Budget | cmp vs no-draft | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: |
| n=160 strict hybrid | 4 | 0 | 83.0% (122/147) | 36.29 t/s |
| n=1000 strict hybrid before current defaults | 4 | 0 | 80.4% (762/948) | 35.98 t/s |
| n=4000 strict hybrid before current defaults | 4 | 0 | 79.0% (1883/2385) | 33.90 t/s |
| n=4000 block-501 audit | 4 | 0 | 79.0% (1883/2385) | 33.75 t/s |

The strict hybrid verifier is now the default DSpark greedy verifier when not in
`--quality` or explicit exact mode. The default static verifier budget is now 5,
matching the DSpark-5 checkpoint (`block_size=5`). Static mode uses
`min(block_size, --draft-verify)`, so plain DSpark runs are active-5. Use
`--draft-verify 2`, `3`, or `4` explicitly for fixed smaller-block A/B
diagnostics.
Plain command validation before the microKV default:

| Run | Budget | cmp vs no-draft | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: |
| n=160 default DSpark | 5 | 0 | 76.4% (126/165) | 38.20 t/s |
| n=1000 default DSpark | 5 | 0 | 77.1% (794/1030) | 39.04 t/s |
| n=4000 default DSpark | 5 | 0 | 76.1% (1963/2580) | 36.48 t/s |

Fresh active-size sweep after adding the `active=` load-log field, forcing
active 2 through the strict decodeN hybrid by default, and reusing session
decodeN row-logit scratch:

| Active cap | Command shape | cmp vs no-draft | Acceptance | Generation |
| ---: | --- | ---: | ---: | ---: |
| 2 | `--draft-verify 2` | 0 | 77.6% (97/125) | 34.33 t/s |
| 3 | `--draft-verify 3` | 0 | 70.6% (108/153) | 36.63 t/s |
| 4 | `--draft-verify 4` | 0 | 83.0% (122/147) | 39.70 t/s |
| 5 | `--draft-verify 5` | 0 | 76.4% (126/165) | 38.20 t/s |

Older rechecks after making the KV FP8/raw row-store microbatch path default
showed active 4 as the best local Flash resident default at that point:
active-4 n=4000 byte-clean at 37.67 t/s versus a no-draft baseline at
34.50 t/s, active-4 n=1000 byte-clean at 39.93 t/s, and a post-default n=160
sweep with active-2 35.60 t/s, active-3 38.44 t/s, active-4 40.59 t/s,
active-5 40.13 t/s. Treat those as historical evidence, not the latest
current-tree state. The latest current plain-default checks are lower
(active-4 n=160 36.62 t/s; n=64 33.87 t/s), so there is a regression or
measurement-state gap to isolate before claiming the current default is best.
Active 5 remains the full DSpark-5 A/B setting and is still the best historical
speed band.
The dormant QKV/F16 pair-row switches were rechecked after the microKV default:
`DS4_DSPARK_QKV_PAIR_ROWS=1`, `DS4_DSPARK_F16_PAIR_ROWS=1`, and both together
were byte-clean at n=160, but all measured within noise of the default. The
combined n=1000 run was also byte-clean but only moved 35.91 to 36.02 t/s under
high VM pressure, so these remain gated diagnostics rather than defaults. Fresh
post-direct-Q2 recheck `bench-results/dspark_pairrows_ab_025145` repeated the
same result: baseline/default/pairrows all matched (`cmp=0`), default active-5
was `40.07 t/s`, and pairrows was `39.96 t/s`.
The existing F16 rows5 verifier kernels were also rechecked directly:
`bench-results/dspark_f16_rows5_ab_025918` matched baseline/default output for
both `DS4_DSPARK_VERIFY_F16_ROWS5=1` and
`DS4_DSPARK_VERIFY_F16_ROWS5_SEQ=1` (`cmp=0`), but both were slower than the
default active-5 path: default `40.07 t/s`, shared rows5 `39.14 t/s`, seq rows5
`39.30 t/s`. Keep F16 rows5 diagnostic-only; the HC-pre win needs a new fused
HC-pre/output shape, not this existing kernel.
The latest active-4 toggle sweep under backend-stats logging found no new
default-worthy verifier flag: disabling output-HC batching, output low rows, or
attention compressor rows was byte-clean but slower; disabling output inverse
rope diverged; the QKV batch shortcut was still not byte-clean; and disabling
index compressor rows repeated byte-clean but slightly slower at n=1000 (35.93
t/s versus 36.02 t/s default). Treat the remaining win as kernel work, not
environment tuning.
A corrected `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1` active-4 n=64 run reports
827 estimated verifier work units per block over 43 layers: `heads=172`,
`comp=164`, `index=84`, `kv=43`, plus one-per-layer buckets for attention,
output-HC, FFN-pre, router, routed MoE, ordered sum, shared, and post-HC. The
estimator now matches the default-on microKV runtime path; the next target is
not raw KV, but a DSpark-only exact microbatch attention/compressor/indexer
helper that preserves row-order mutation while reducing per-row/per-stage
command work.
Current active-5 dispatch estimate after direct ordered-Q2 is `893` per block:
`heads=215`, `comp=205`, `index=105`, `kv=43`, `ordered_sum=0`. A focused
varmap profile (`bench-results/dspark_varmap_profile_025345`) showed host encode
is not the limiter: 2337 varmap calls over the n=320 run cost only `2.98 ms`
total host time, with DSpark generation `41.79 t/s`, verify `75.95 ms`, and
`cmp=0` in the paired clean smoke. The remaining varmap/attention cost is GPU
work and cache traffic, not CPU command encoding.

The attention subprofile was corrected so row-output HC no longer gets counted
again as an empty `hc_post` profiling fence. Recheck
`bench-results/dspark_attn_subprofile_after_profilefix_025610` reports active-5
per-block fenced microscope buckets around: HC-pre `42-45 ms`, q path
`21-24 ms`, kv path `10-12 ms`, compressor `12-14 ms`, indexer `4.8-5.4 ms`,
heads `18-23 ms`, and output `25-28 ms`, with output now counted once per layer
(`/43` instead of `/86`). Absolute totals are fence-inflated, but the priority
ordering changed: HC-pre and attention output are at least as important as the
head kernel itself. Next kernel work should target HC-pre/output fusion or a
unified N<=5 attention-half microbatch, not more environment toggles.

The finer HC/output split is now available behind
`DS4_DSPARK_HC_PRE_SUBPROFILE=1 DS4_DSPARK_OUTPUT_SUBPROFILE=1` plus
`DS4_DSPARK_HYBRID_ATTN_SUBPROFILE=1`. Recheck
`bench-results/dspark_hc_output_subprofile_030709` reports active-5 fenced
details around: HC RMS `40-42 ms/43`, HC function `10-11 ms/43`, HC split/norm
`10-11 ms/43`, output inverse-RoPE `8-10 ms/43`, output-low Q8 `19-23 ms/43`,
and output-HC expand `17-20 ms/43`. The same directory's clean A/B stayed
byte-identical: baseline vs DSpark `cmp=0`, baseline `33.33 t/s`, DSpark
direct ordered-Q2 `39.99 t/s`, no-direct-Q2 `39.56 t/s`. Conclusion: the direct
Q2 down+ordered-sum fusion is correct for the local Q2 boundary but is the wrong
large-speed fusion target; it is a narrow micro-win. Do not extend it into
unordered `sum6` or grouped shared-weight MoE without a separate single-layer
exactness proof. The grouped-Q2 compare hook described above now gives that
single-layer proof signal: unsafe shared-weight grouped Q2 is not byte-exact
even before the ordered sum. A 2026-06-29 recheck in
`bench-results/dspark_wrong_fusion_recheck_040257` shows why short smokes are not
enough: unsafe grouped Q2 happened to match final n=96 text, but active-5
slot-down deltas were already present; at n=160 it diverged visibly. This is the
wrong fusion family for A/B/C unless the slot-down compare becomes exact.

Follow-up HC-pre scale-buffer fusion is also diagnostic-only. The opt-in
`DS4_DSPARK_HC_PRE_SCALED_F16=1` path computes RMS scales into a tiny N<=5
buffer and runs an exact scaled F16 rows5 HC projection from `batch_cur_hc`,
skipping materialization of `batch_flat_hc`. Smoke
`bench-results/dspark_hc_scaled_f16_031619` was byte-clean
(`baseline_vs_scaled_cmp=0`, `default_vs_scaled_cmp=0`) and entered the new
kernel, but slowed n=64 from default DSpark `37.91 t/s` to `34.65 t/s`
(`verify=64.84 ms` to `75.28 ms`). Do not promote this helper. It remains useful
only as an exactness/diagnostic scaffold; the next HC-pre attempt needs a more
integrated kernel, or priority should move back to shared-prefix attention and
output-low/output-HC.

The active-2 default change is specifically worth keeping: on the n=160 smoke,
plain `--draft-verify 2` is byte-clean and measures 34.25 t/s, while
`DS4_DSPARK_DECODEN_VERIFY=0 --draft-verify 2` keeps the old exact decode2 path
available and measures 30.82 t/s.

Fusion double-checks for active 2/3/4/5 are clean at n=64 (`cmp=0`). Runtime
logs confirm the DSpark draft path uses `dspark-strided` native MXFP4 routed MoE
for all active sizes. The verifier logs confirm the strict decodeN hybrid helpers
for all active sizes: exact F16 rows FFN-pre, exact F16 router logits rows, exact
selector rows, exact ordered routed expert-sum, row-routed row-exact tiny batch,
shared exact fused rows, batched output head, and batched DSpark target-hidden/KV
maintenance.

The focused block-501 audit is clean at `1e-8` across `attn-norm`, Q-rope,
KV-cache row, raw attention heads, attention output, attention-HC, FFN-pre,
FFN-norm, routed output, shared output, post-HC, commit logits, final HC,
DSpark target hidden, and DSpark KV/frontier state.
The latest block-1 audit with the implicit default path also reports
`stage=clean`, `mismatch_layers=0`, `final_hc_max=0`, `decode_hc_max=0`,
`dspark_hidden max=0`, and `dspark_kv max=0`.

Blocking stage profile on n=96, budget 5, strict hybrid with row-exact tiny-batch
routed helper and all-ratio exact compressor/indexer rows, per five-token block:

| Stage | Typical Cost |
| --- | ---: |
| attention / compressor / indexer row loop | ~61 ms |
| row-routed MoE | ~31 ms |
| row-shared expert | ~13 ms |
| row-router | ~13 ms |
| row-FFN-pre | ~11 ms |
| batched tail | ~13.5 ms |
| target hidden capture | 1.3-1.5 ms |
| output head | 1.6 ms |
| readback | ~0.03 ms |
| total verifier block, fenced profile only | ~146 ms |

Latest budget evidence after the row-exact tiny-batch routed helper:

| Budget | cmp | Acceptance | Generation |
| ---: | ---: | ---: | ---: |
| 4, n=1000 | 0 | 80.4% (762/948) | 38.60 t/s |
| 5, n=1000 | 0 | 77.1% (794/1030) | 39.04 t/s |
| 5, n=4000 | 0 | 76.1% (1963/2580) | 36.48 t/s |

Confidence scheduler sweep after the current strict path, n=1000, `--draft-verify 5`:

| Threshold | cmp | Avg scheduled | Acceptance | Generation |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 5.00 | 77.1% (794/1030) | 36.66 t/s |
| 0.2 | 0 | 4.98 | 76.9% (793/1031) | 36.37 t/s |
| 0.4 | 0 | 4.69 | 80.7% (787/975) | 36.64 t/s |
| 0.6 | 0 | 4.25 | 85.4% (763/893) | 36.13 t/s |
| 0.8 | 0 | 3.53 | 94.2% (711/755) | 35.07 t/s |

Rejected speed variants:

- `DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1` is not byte-clean.
- Batched shared expert is not byte-clean.
- Forcing `DS4_DSPARK_HYBRID_ROW_SHARED=0` is not byte-clean on n=160 and was
  slower in the latest smoke.
- Forcing `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=0` byte-matched n=160 but
  diverged by n=1000 and was slower than the default.
- `DS4_DSPARK_HYBRID_ROW_ROUTED_DECODE2_ROWS=1` is now the default. It is
  byte-clean through n=4000 and slightly improves the long run: 34.15 t/s versus
  the previous implicit default's 33.86 t/s.
- Exact multirow F16 attention-compressor projections are now default for all
  compressed layers, not only ratio-4. They are audit-clean at `1e-8` for the
  focused exact-vs-hybrid state audit and improved the long run to 34.45 t/s
  before the ordered expert-sum kernel.
- The exact ordered routed-MoE expert-sum kernel is now default. It is
  byte-clean through n=4000, moves the long run to 34.70 t/s, and can be
  disabled with `DS4_DSPARK_ORDERED_MOE_SUM_DISABLE=1` for A/B checks.
- Exact ratio-4 indexer-compressor rows are now default. They are byte-clean
  through n=4000 and move the long run to 35.09 t/s; disable with
  `DS4_DSPARK_HYBRID_INDEX_COMP_ROWS_DISABLE=1` for A/B timing.
- `DS4_DSPARK_HYBRID_ROW_ROUTED_SLOTWISE=1` is byte-clean through n=1000, but
  slower than the current default path: 35.04 t/s versus 37.01 t/s on the latest
  n=1000 smoke.
- `DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_BATCH_QKV=1` is faster on n=160 but not
  byte-clean.
- Precomputing exact indexer rows with `DS4_DSPARK_HYBRID_INDEXER_ROWS=1` is
  byte-clean but slower than the default per-row indexer projection.
- Q8 rows5 is now the promoted dense rows optimization. It is default-on for the
  strict verifier's N<=5 exact Q8 projections and can be disabled with
  `DS4_DSPARK_VERIFY_Q8_ROWS5_DISABLE=1` or
  `DS4_DSPARK_VERIFY_NO_Q8_ROWS5=1`.
- `DS4_DSPARK_VERIFY_F16_ROWS5_SEQ=1` and
  `DS4_DSPARK_VERIFY_F16_ROWS5=1` remain diagnostic-only. They prove the
  one-dispatch N<=5 F16 rows shape can preserve bytes, but the current kernels
  are slower than the default path on local smokes.
- In Mode A/strict, `DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS=1` remains
  diagnostic-only. It defers plain attention heads into a batch call after
  row-ordered cache/index mutation; this is not byte-clean enough for the
  `cmp=0` contract. Do not promote it into strict mode.
- In Mode B/batch-canonical, deferred batch heads are now default-on because the
  batched head order is part of the canonical state contract. Disable for A/B
  with `DS4_DSPARK_BATCH_DEFER_HEADS_DISABLE=1`. The current best Mode B n=1000
  budget-4 run uses this default and reaches 35.89 t/s.
- The current follow-up varlen plain-head diagnostic is gated behind
  `DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_VARLEN_UNSAFE=1` (or
  `DS4_TARGET_FORWARD_SHARED_PREFIX_VARLEN_HEADS_UNSAFE=1`). It passes explicit
  per-row `n_raw`, `raw_start`, and `n_comp` into a new N<=5 plain mixed-heads
  kernel, but n=64 still diverged: acceptance collapsed to 34.4%, generation
  dropped to 22.07 t/s, and output immediately changed to `Space Invoker` /
  malformed fences. Conclusion: metadata is necessary but not sufficient; the
  exact microbatch/shared-prefix helper must preserve the strict per-row flash
  attention realization or explicitly make a unified-greedy contract canonical.
- `DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_ROW_EXACT=1` is the safe companion probe.
  It defers only plain heads, leaves indexed heads on the row path, and then
  replays the deferred rows through the existing row-exact flash-attention
  encoders inside one helper. Evidence:
  `bench-results/dspark_rows_exact_heads_n64_b4` was `cmp=0`, 35.85 t/s,
  verify 65.57 ms; `bench-results/dspark_rows_exact_heads_n160_b4` was `cmp=0`,
  38.44 t/s, verify 63.45 ms. This proves the wrapper/metadata consolidation is
  safe but not a speed lever; the real Patch E must share the long prefix/cache
  read inside a flash-attention-compatible kernel.
- Keeping the small N<=5 compressor/indexer count arrays on the stack is
  byte-clean and helps the short smokes a little, but the n=4000 recheck was
  33.83 t/s. This is cleanup, not a long-run speed win.
- `DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1` is now the default safe
  routed-MoE verifier helper. It calls the row-exact tiny-batch resident encoder.
  On local Flash Q2 down experts, it may use direct ordered-Q2 down+sum, which
  is not the older unsafe `sum6`: it preserves independent slot accumulators and
  the exact slot0..slot5 ordered FP32 sum. Disable it with
  `DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` or verify the
  boundary with `DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1`.
- `DS4_METAL_ENABLE_ROUTED_DOWN_SUM6=1` is a useful sizing experiment but not
  default-safe. On n=160, DSpark with the flag matched a no-draft target run
  with the same flag (`cmp=0`) and reached 38.98 t/s, while the no-draft target
  with the flag reached 38.06 t/s. Both differed from the normal no-flag greedy
  baseline (`cmp=1`). Blocking stage profiling shows the row-routed bucket drops
  from mean 34.92 ms to 29.72 ms per four-token verifier block, but the kernel
  fuses the six routed-down expert slots into the matvec and changes
  accumulation/rounding relative to the default separate-down-plus-ordered-add
  path. Treat the ~5 ms row-routed delta as a target for a future
  exact-preserving kernel, not as a verifier default.
- The exact ordered expert-sum kernel is the safe version of that idea: it keeps
  the routed-down matvec outputs separate and replaces only the five ordered
  expert-add dispatches with one kernel that applies the same FP32 add order per
  output element. It is byte-clean against the normal baseline and moves the
  row-routed profile mean from 34.92 ms to 32.78 ms per four-token verifier
  block.
- Added a fine attention subprofile gate for the strict verifier:
  `DS4_DSPARK_ATTN_SUBPROFILE=1 DS4_DSPARK_ATTN_FINE_SUBPROFILE=1`. This is a
  fenced diagnostic, not a headline-speed mode. It splits the attention bucket
  into KV-store, compressor projection/update/quant/capture, and
  indexer-compressor projection/update/capture plus indexer query/score/top-k.
  Smoke evidence: `bench-results/dspark_fine_subprofile_055320` with
  `--draft-verify 5 -n 64` is byte-clean (`cmp=0`). Normal non-profile mode in
  the same directory is also byte-clean and reaches 37.41 t/s versus the paired
  no-draft baseline at 33.33 t/s; DSpark reports draft 16.97 ms/block, verify
  66.34 ms/block, tau 4.27, acceptance 70.0% (49/70). The fine-fenced run slows
  generation to 14.36 t/s, as expected, but shows the dominant strict attention
  sub-buckets are compressor update/capture and indexer-compressor
  update/capture, not indexer scoring/top-k on this short prompt:
  `comp_update ~=53 ms`, `comp_capture ~=44 ms`, `idx_comp_update ~=28 ms`,
  `idx_comp_capture ~=22 ms` per active-5 block under fences.
- Interpretation of the fine split: the wrong-fusion lesson still stands. Do
  not optimize by switching strict mode to the plain mixed shared-row attention
  kernel family. The next useful strict work is an exact N<=5 attention-state
  microbatch/refactor that reduces compressor/indexer state mutation and prefix
  capture overhead while preserving the row-exact flash-attention realization.
- Paired KV/score blits are now used for attention/indexer frontier snapshot,
  restore, and prefix commit. This is byte-clean cleanup, not a speed lever:
  `bench-results/dspark_paircopy_060216` matched the paired baseline (`cmp=0`)
  and measured 37.25 t/s versus the same-tree baseline at 33.29 t/s, with draft
  16.94 ms/block, verify 66.86 ms/block, overhead 1.82 ms/block, and commit
  1.38 ms/block. The paired-copy fine profile remained essentially unchanged
  from the previous fine-profile run, so this only reduces encoder/blit
  bookkeeping risk.
- Prefix-N capture must remain enabled for the strict DSpark-5 path. The direct
  A/B `DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1` in
  `bench-results/dspark_paircopy_060216` was byte-clean (`cmp=0`) but slowed to
  31.25 t/s. It reported block 107.63 ms with draft 17.04 ms, verify 64.66 ms,
  overhead 25.93 ms, and commit 25.49 ms. That confirms exact replay for partial
  accepts is the wrong recovery path; prefix capture is paying for itself.

Current correctness command:

```bash
./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
  --draft-verify 5 \
  --temp 0 --nothink -n 1000 \
  -p "$PROMPT" --resident -c 4096
```

This exercises full DSpark-5 (`block=5 verify=5 active=5`) and should be
expected to byte-match the paired no-draft greedy baseline. The latest clean
n=1000 run was byte-clean at 38.63 t/s against a 35.05 t/s no-draft baseline.

Active-5 recovery command:

```bash
./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
  --draft-verify 5 \
  --temp 0 --nothink -n 320 \
  -p "$PROMPT" --resident -c 4096
```

The saved historical paired run for this shape was byte-clean (`cmp=0`) at
41.32 t/s against a 34.50 t/s no-draft baseline. The latest clean current-tree
rerun is byte-clean at 39.26 t/s against a 32.76 t/s no-draft baseline.

Paired recovery script shape:

```bash
BASE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
PROMPT='Make a game of Space Invader in Pygame'
TAG=dspark_recovery_$(date +%H%M%S)

./ds4 -m "$BASE" --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096 \
  > "bench-results/${TAG}_baseline.out" \
  2> "bench-results/${TAG}_baseline.err"

./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
  --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096 \
  > "bench-results/${TAG}_active4.out" \
  2> "bench-results/${TAG}_active4.err"

./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
  --draft-verify 5 \
  --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096 \
  > "bench-results/${TAG}_active5.out" \
  2> "bench-results/${TAG}_active5.err"

cmp -s "bench-results/${TAG}_baseline.out" "bench-results/${TAG}_active4.out"
echo "active4-cmp=$?"
cmp -s "bench-results/${TAG}_baseline.out" "bench-results/${TAG}_active5.out"
echo "active5-cmp=$?"
rg -n "generation:|dspark perf:|dspark acceptance|avg scheduled|vm pressure" \
  "bench-results/${TAG}_"*.err
```

Optional VM/compressed-memory diagnostic variant:

```bash
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
./ds4 -m "$BASE" --draft dspark --draft-path "$DRAFT" \
  --draft-verify 5 \
  --temp 0 --nothink -n 1000 -p "$PROMPT" \
  --resident -c 4096
```

Use the diagnostic variant for `vm pressure`, task-compressed, GPU-compressed,
and DSpark draft/verify timing buckets. Do not compare its t/s directly against
clean no-stats headline runs.

Use `DS4_DSPARK_FAST_VERIFY_DISABLE=1`, `DS4_DSPARK_EXACT_VERIFY=1`, or
`--quality` to force the older exact verifier path for A/B checks.

The remaining blocker is performance, not draft quality. The verifier is
commit-safe on the strict path, and clean active-5 runs are back near the
historical 39 t/s band. The next production work is not more recovery
benchmarking; it is an exact N<=5 microbatch verifier that preserves row-order
raw/cache/indexer state while reducing attention/cache and routed-MoE
dispatch/per-row overhead.

## One-Line Summary

DSpark draft quality is good enough and the strict decode-order verifier is
byte-clean. Current clean active-5 reaches 39.26 t/s on n=320 and 38.63 t/s on
n=1000 with `cmp=0`; stats/timing runs are diagnostic and lower. Continue N<=5
exact verifier work, preserving per-row raw/cache state order while reducing
attention/cache and routed-MoE dispatch cost.

## 2026-07-01 Fast-Mode Update

The current fastest simple-canary-clean demo point is no longer the old strict
39 t/s recovery run. With `--draft-fast-relaxed --draft-verify 5` and the Flash
sidecar target, the current Pygame n=1000 point is:

```text
bench-results/normal_allow_current_065907
generation 54.18 t/s, tau=4.58, acceptance=93.6%, canary suspect=0
```

This is a non-byte diagnostic mode. It enables frontier DSpark draft, relaxed
target-supported accept, forced MMA attention, fast Q2 routed down, and GPU
draft prefetch. It is useful for speed demos, but still below the >60 goal.

Stop signs from the same continuation:

- Exact HTML stress prompt `/tmp/si-html-010`, n=4000:
  `bench-results/si_html_010_fastrelaxed_n4000_065005` measured `36.57 t/s`,
  `tau=3.93`, acceptance `77.6%`, canary suspect. It is byte-identical to the
  earlier fast-relaxed HTML artifact.
- Strict DSpark remains byte-identical to no-draft on that HTML prompt, but
  no-draft/strict are also canary-suspect; treat this as a stress canary, not a
  single-gate proof.
- Re-enabling target-top loop checks makes the easy Pygame prompt clean but
  collapses tau/speed to about `24 t/s`; it is a safety brake, not a fast path.
- Tightening only off-argmax accept (`top64/delta4/off2`,
  `top32/delta3/off1`, temperature-style gate) drops Pygame to about
  `48.7 t/s` and still leaves the HTML smoke canary suspect.
- The simple "bonus token" idea is not a throughput lever with the current
  five-row DSpark package. Printing the next target argmax early only shifts a
  token into the next block unless the next verifier can consume the pending
  token and still verify five fresh draft rows.

The credible >60 work remains structural: reduce the current `~68-73 ms`
verifier wall, especially the generic IQ2/Q2 row-routed verifier path, build a
real batch-canonical verifier contract, or hide DSpark draft on a truly
separate engine. More top-k/delta/temperature tuning is now a stop-rule unless
the accept/commit contract changes.
