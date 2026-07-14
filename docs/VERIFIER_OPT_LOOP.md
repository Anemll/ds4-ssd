# DSpark Verifier Optimization Loop (sidecar, strict byte-safe)

GOAL: reduce verify ms/block on the sidecar path (currently ~93-103 @16-32K static-sched;
verify(5 rows) = 2.2-2.5x one decode token). Target: verify <= ~70ms @32K without breaking
the strict contract. Speedup math: S = tau / ((draft+verify+oh)/T_decode) — every ms off
verify is the lever; tau is NOT in scope (retrain-only).

## Hard rules (violating these voids a result)
- Strict contract: no relaxed-accept, no precision changes on the verify path (NAX/MMA/f16
  staging = non-byte, rejected). Draft-side and SCHEDULING changes are byte-safe by design.
- Profiling protocol: `--draft-scheduler static` ALWAYS (confidence default trims rows and
  confounds verify-ms); unfenced only (fenced stage profile inflates 2-8x); single ds4
  instance (check pgrep first; user may be benching — abort, retry later); 15-20s cooldowns;
  paired A/B in the same session; never read tau/t/s from skip-probe runs (outputs corrupt).
- Promotion gates: paired verify-ms win at 16K AND 32K + no loss >1% at 1-4K + tau/acceptance
  pinned in non-probe A/B + `cmp` byte-equality vs no-draft on a low-tie prompt (small prefill).
- Diagnostics stay env-gated default-off. Do not touch the INDEXER_ROWS auto-gate
  (pos>=16384, validated) or the confidence-scheduler default.

## Known dead ends — do NOT retry
MoE route-dedup kernels; gate/up grouping; dispatch-merging without launch-count/occupancy
change (3x proven neutral); SG4 candidate-coarsening in scorer (neutral, 2026-07-05);
barrierless scorer (non-strict); rows-6 default (net-negative except code/math); f16 shadow KV.

## Harness
- Models: BASE=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major (sidecar),
  DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft.
- Prompts: bench-results/context-decode-dspark/prompt-{1024..32768}.txt.
- Run shape: `DS4_DSPARK_PERF=1 [probes] ./ds4 -m $BASE --draft dspark --draft-path $DRAFT
  --draft-scheduler static --temp 0 --nothink -n 1000 -c 40000 --resident
  --prompt-file <prompt>`; metric = the `dspark perf:` verify ms/block.
- Existing probes: DS4_DSPARK_PROF_SKIP_INDEXER_SCORE=1, DS4_DSPARK_PROF_INDEXER_CAND_CAP=N
  (keep N>=2048), DS4_DSPARK_PROF_INDEXER_FORCE_FIRSTK=1,
  DS4_DSPARK_PROF_INDEXER_TOPK_THEN_FIRSTK=1, DS4_DSPARK_INDEXER_SCORE_SG4=1
  (neutral, keep as reference).
- Context: memory file `sidecar-dspark-verify-economics` + `strict-dspark-55-loop-state`
  (monolithic history), code sites ds4.c ~16290-16345 (score/topk row chain), ~15945
  (deferred heads), ~14363 (rows gate), ds4_metal.m:8780 (scorer dispatch).

## Work queue (do in order; one item per iteration)
1. LOCALITY-CONTROLLED SCORER PROBE: skip-score's 22ms @32K is suspect (garbage selection
   -> clustered reads -> cheaper attention inflates the delta; dispatch count is
   ctx-constant but "cost" grew 4.8->22ms). Add probe forcing selection=first-2048 on BOTH
   sides; re-measure true scorer term. If ~5ms, deprioritize item 4 to after item 2/3.
2. DRAFT OVERLAP (+12-15% expected): dispatch next block's draft while verify drains; the
   perf log already prints the upper bound ("draft-overlap upper-bound ... hides ~11ms").
   Draft-side = byte-safe by construction. Validate: gen t/s up, verify unchanged, cmp=0.
3. SIDECAR VERIFY PARITY (~9ms): sidecar verify 74.6ms vs monolithic 65.9 at like tau. Diff
   which stages run rows-batched on monolithic but per-row on sidecar (row_* stage names);
   port rows-5 arms + check DS4_DSPARK_VERIFY_SPLIT_LAYERS=4 active on sidecar.
4. LAUNCH-COUNT BATCHING: per-layer rows-batched exact scorer+topk (5 rows/dispatch after
   row-state updates, per-row cur_index visibility mask, per-candidate simd_sum order
   preserved => byte-exact; 105->21 dispatches). Only if item 1 confirms the term.
5. SLOT-BANK SIZING (baseline lift, not verifier): A/B --moe-slot-bank 48/64 vs 256 and
   --ssd-cache auto at 16K/32K; wired-bank vs page-cache pressure (startup warning).
6. TOPK PROBE: iota-selection skip for ds4_gpu_indexer_topk_tensor if residual >5ms remains.

## Loop protocol (each iteration)
1. Re-read this file fresh, especially ## Feedback; never rely on a copy read at iteration start.
2. `pgrep -x ds4 || pgrep -x ds4-agent` — if busy, stop this iteration (user owns the box).
3. Take the top unresolved queue item; implement smallest testable change (env-gated).
4. Measure per protocol (paired, 16K+32K minimum). Promote or record-and-reject.
5. Re-read this file again immediately before write-back. Make writes surgical: append to ## Log,
   replace only ## Status, and remove only consumed ## Feedback lines.
6. Never commit; leave working tree changes + updated Log section for user review.

## Log
(append results here)
- 2026-07-05 ITEM 1 DONE (locality-controlled scorer probe, DS4_DSPARK_PROF_INDEXER_FORCE_FIRSTK=1
  ds4.c:16335 + dsv4_misc.metal:488 + ds4_metal.m:9262): 32K firstK scorer-on 97.86 vs skip-score
  93.00 ms/block, tau/acc/blocks pinned (3.42/68.7%/226) => TRUE scorer cost = 4.86 ms/block.
  Earlier 22ms was stale-score topk changing downstream attention locality (artifact confirmed).
  SG4 re-confirmed neutral. DECISION: item 4 (scorer batching) DEPRIORITIZED — max win ~5ms.
  Next = item 2 (draft overlap, +12-15% bound printed in perf logs), then item 3 (sidecar
  rows-arm parity ~9ms). Remaining long-ctx growth owner is now most likely the topk/masking +
  serialized per-row chain — item 6 (topk iota probe) promoted to after item 3.
- 2026-07-05 ITEM 2 PARTIAL/REJECTED CURRENT HOOK (draft overlap): no Feedback lines to ack.
  Existing DS4_DSPARK_DRAFT_PREFETCH was first A/B'd as-is and was neutral because it never
  armed on the measured strict_v1 branch. Patched ds4.c so the normal strict decodeN/decode2
  successful commit paths also call ds4_session_dspark_draft_prefetch_start(...); smoke @1K
  confirmed repeated "draft prefetch start/hit" lines. Clean 16K paired run on the rebuilt
  binary remained neutral:
  | ctx | mode | gen t/s | draft ms | verify ms | block ms | tau | acceptance |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 16K | off | 29.30 | 10.90 | 92.40 | 105.00 | 3.12 | 77.3% |
  | 16K | DS4_DSPARK_DRAFT_PREFETCH=1 | 29.31 | 10.77 | 92.34 | 104.83 | 3.12 | 77.3% |
  Earlier unpatched 32K off/on was also neutral-to-negative (25.97/98.55/111.03 vs
  25.61/99.99/112.60). Root cause: this hook starts after verify+commit, so it only overlaps
  stdout/loop overhead, not the ~90-100ms verifier drain. True item-2 overlap requires launching
  the next-block draft after verifier work is submitted but before verifier readback/finish, using
  either duplicated draft scratch or an async verifier/readback split; draft and verifier both use
  batch_* scratch, so launching before verifier encoding is not byte-safe. DECISION: keep the
  fixed prefetch as default-off diagnostic/cache plumbing, do not promote. Item 2 remains open
  as "async verifier seam or duplicate draft scratch"; continue to item 3 unless user explicitly
  wants the larger overlap refactor next.
- 2026-07-05 ack: consumed overseer steer from ## Feedback. Item-2 verdict accepted: neutral
  DS4_DSPARK_DRAFT_PREFETCH fix stays default-off, but async-seam/duplicate-scratch draft-overlap
  refactor is OUT OF LOOP SCOPE because it is multi-day and needs explicit user sign-off. Do not
  attempt it inside this loop. Proceed item 3 then item 6. Item-3 methodology: monolithic comparison
  must use the same prompt file and `--draft-scheduler static`; monolithic GGUF
  `/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
  differs in quant layout, so compare stage STRUCTURE / which rows-batched arms engage, not absolute
  milliseconds across models. Absolute performance target remains paired sidecar-vs-sidecar after
  any port. Protocol refined: re-read this file before feedback handling and before write-back;
  write-back must surgically append Log, replace Status, and remove only consumed Feedback lines.
- 2026-07-05 ITEM 3 DONE/NO PORT (sidecar verify parity structural comparison): Feedback empty;
  no ds4/ds4-agent process active. Short structural runs used the same 16K prompt file and
  `--draft-scheduler static` on the rebuilt ds4-ssd binary:
  `item3-sidecar-20260705-225139.log` (sidecar directory model, `--resident`) and
  `item3-mono-gguf-20260705-225325.log` (monolithic antirez GGUF). Do not compare absolute
  ms/tau across these models: the GGUF quant layout differs and the sampled acceptance/tau also
  differed. Stage structure matched for the rows verifier stack: strict_v1 active=5, Q8 rows5,
  microbatch KV rows, FFN-pre/router rows, shared fused rows, shared-down+HC rows,
  indexer-rows auto at 16K, compressor/indexer rows, deferred batch heads, rows-exact flash
  attention, and varmap rows5 all engaged on both paths. `DS4_DSPARK_VERIFY_SPLIT_LAYERS=4`
  is already defaulted in `dspark.c`. The only structural delta is storage-layout inherent:
  sidecar takes the `g->flash_moe` banked row-exact tiny-batch path plus direct Q2 down
  ordered-sum; GGUF takes the `!g->flash_moe` monolithic row-exact tiny-batch path (and logs
  the routed MoE exact ordered expert-sum kernel). DECISION: no safe/small rows-arm port found
  for item 3; close it as structural parity except for the expected row-routed storage split.
  Next = item 6 (locality-controlled topk/iota probe).
- 2026-07-05 ITEM 6 DONE/REJECTED AS OWNER (locality-controlled topk probe): Feedback empty;
  exact-name guard (`pgrep -x ds4 || pgrep -x ds4-agent`) was idle. Added default-off probe
  `DS4_DSPARK_PROF_INDEXER_TOPK_THEN_FIRSTK=1` in `ds4.c`: it runs real topk and then overwrites
  selection with firstK, so paired with `DS4_DSPARK_PROF_INDEXER_FORCE_FIRSTK=1` both sides feed
  identical downstream attention locality. Built clean (`make -j8 ds4 ds4-agent`, pre-existing
  warnings only). Matrix logs: `bench-results/context-decode-dspark/item6-topk-20260705-231518/`.
  | ctx | mode | verify ms | block ms | tau | acceptance | avg scheduled |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | 16K | FORCE_FIRSTK | 76.13 | 88.73 | 2.22 | 61.9% | 4.99 |
  | 16K | TOPK_THEN_FIRSTK | 78.04 | 90.49 | 2.22 | 61.9% | 4.99 |
  | 32K | FORCE_FIRSTK | 98.67 | 111.70 | 2.71 | 69.3% | 4.99 |
  | 32K | TOPK_THEN_FIRSTK | 98.39 | 110.98 | 2.71 | 69.3% | 4.99 |
  DELTA (topk cost with locality held fixed): +1.91 ms/block at 16K, -0.28 ms/block at 32K
  (noise). DECISION: topk/masking is not the long-context growth owner and no topk optimization
  is promoted. Remaining verify tax points back to the storage path / slot-bank / memory-pressure
  investigation. Next = item 5.
- 2026-07-05 ITEM 5 DONE/REJECTED SMALL BANKS (slot-bank sizing): Initial Feedback was empty;
  exact-name guard idle. Ran `{resident256,resident64,resident48,ssdauto} x {16K,32K}` with
  static scheduler and no probes. Logs: `bench-results/context-decode-dspark/item5-slotbank-20260705-232812/`.
  `--resident` ignores `--ssd-cache`, so `ssdauto` was tested as non-resident direct-mmap cache mode.
  | ctx | mode | gpu/app memory | verify ms | block ms | gen t/s | vm pressure |
  | --- | --- | --- | ---: | ---: | ---: | ---: |
  | 16K | resident256 | gpu 86.31 GiB, app 86.39 GiB | 92.35 | 104.80 | 29.31 | 83% |
  | 16K | resident64 | gpu 31.87 GiB, app 31.88 GiB | 166.94 | 179.99 | 14.40 | 46% |
  | 16K | resident48 | gpu 27.34 GiB, app 27.34 GiB | 171.31 | 184.32 | 14.08 | 43% |
  | 16K | ssdauto | gpu 13.73 GiB, app 13.75 GiB | 134.77 | 147.67 | 15.56 | 59% |
  | 32K | resident256 | gpu 86.99 GiB, app 87.13 GiB | 101.46 | 114.05 | 25.26 | 83% |
  | 32K | resident64 | gpu 32.66 GiB, app 32.61 GiB | 150.39 | 163.41 | 14.38 | 46% |
  | 32K | resident48 | gpu 28.13 GiB, app 28.08 GiB | 155.50 | 168.54 | 13.96 | 43% |
  | 32K | ssdauto | gpu 14.53 GiB, app 14.50 GiB | 139.14 | 152.22 | 15.15 | 60% |
  DECISION: do not promote smaller slot banks or `--ssd-cache auto` for strict DSpark verifier
  speed. They reduce wired/gpu pressure, but the storage/miss path roughly halves generation
  throughput versus all-expert resident256. The warning about full bank pressure is real for
  memory health, but for this workload hot expert residency beats page-cache headroom.
- 2026-07-05 ack: consumed overseer Feedback written during item 5. Attribution accepted:
  scorer about 5 ms and topk <=2 ms, so the skip-score artifact's remaining long-context delta
  is sparse-attention read locality over selected rows. Per steer, performed CHEAP INSPECTION
  ONLY, no implementation. Findings: `ds4_gpu_indexer_topk_tensor` uses descending argsort/merge
  (`kernel_argsort_f32_i32_desc`, `kernel_argsort_merge_f32_i32_desc`), so topk emits indices in
  score order. `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` already has
  `kernel_dsv4_sort_i32_rows_asc`, but fast decode skips it when `decode_one_token` or deferred
  batch heads are active; the indexed attention kernels consume `row_topk[i]` in array order and
  update the online softmax state in that order. Therefore sorting selected indices ascending
  would improve locality but is NOT byte-safe against the current fast strict path, because it
  changes accumulation order. It is only a candidate if the contract is changed to sorted/canonical
  order, `--quality` semantics, or a byte-equality gate proves acceptable.
- 2026-07-05 FINAL SYNTHESIS / LOOP STOP: Long-context verifier attribution is now complete.
  | component / hypothesis | measured result | verdict |
  | --- | --- | --- |
  | Indexer scorer | true cost 4.86 ms/block at 32K under FORCE_FIRSTK locality control | real but too small for scorer batching |
  | Topk/masking | +1.91 ms at 16K, ~0 ms at 32K with TOPK_THEN_FIRSTK vs FORCE_FIRSTK | not owner |
  | Rows-batched indexer gate | +3-4% verify win at 16K/32K, short-context loss avoided by auto pos>=16384 | kept |
  | Sidecar rows-arm parity | rows verifier stack structurally matches monolithic GGUF; only storage-layout row-routed split differs | no port |
  | Slot-bank sizing | 48/64/ssdauto save memory but verify 139-171 ms vs 92-101 ms for resident256 | no speed promotion |
  | Residual long-context tax | sparse-attention read locality / storage path over spread selected rows | likely owner |
  Remaining levers: (1) true draft-overlap async verifier seam or duplicate scratch, byte-safe in
  principle but multi-day and requires explicit user sign-off; (2) selected-row locality sort, not
  byte-safe against current fast order unless the canonical order/quality contract changes and is
  validated. Stop autonomous loop here; future work needs a new directive.

## Status (agent OVERWRITES this block each iteration — at-a-glance state)
current-item: stopped | last-result: final synthesis written; slot-bank sizing rejects small banks/ssdauto for speed
next: none - autonomous verifier optimization loop is complete pending new user directive | blocked-on: user sign-off for larger async overlap or contract-changing locality sort work

## Feedback (USER writes directives here; agent MUST read this FIRST each iteration,
## act on it over the queue, then move each consumed line into the Log with an "ack:" note)
(empty)
