# DSpark +10 t/s Loop (Space Invaders workload, strict)

GOAL: DSpark >= no-draft baseline + 10 t/s on the Space Invaders HTML prompt
(bench-results/space-invaders-20260706-093624/summary.txt has the prompt/commands),
same tree (ds4-ssd), monolithic GGUF or resident — paired no-draft vs draft.

## SEED ANALYSIS (2026-07-06, from ds4ssd-gguf-dspark.log — READ THIS FIRST)
- Measured: tau=2.64, block=81.94ms, verify=69.0ms, acceptance 79.3%,
  conditional acceptance 88-92% AT EVERY POSITION, avg scheduled 4.42.
- THE ANOMALY: with those acceptances, expected tau ≈ 3.8 → ~46 t/s. Measured 2.64.
  perf counts blocks=1762 but scheduler footer counts only 1328 → ~434 blocks (25%)
  bypass the scheduler path entirely (frontier-gate fail / draft fail / room caps /
  EOS handling?) and likely commit ~1 token. THE GAP IS DEGENERATE BLOCKS, NOT
  verify cost and NOT acceptance. Find and fix what those 434 blocks are.
- Baselines invalid in prior bench: "upstream GGUF" row had NO draft and is a
  different tree. Required: ds4-ssd no-draft same-prompt row (missing).
- Math: S_target = base+10. If base ≈ 33 → need ~43 = tau/block: tau 3.6 @ 84ms
  or tau 4.8 @ 111ms (rows-6 shape). Both plausible: my text-type A/B measured
  code content at 1.27-1.30x with rows-6/conf04 (45 vs 35 t/s).

## Protocol (same hard rules as docs/VERIFIER_OPT_LOOP.md)
- Paired same-session runs; single instance (pgrep guard); 20s cooldowns; temp 0
  --nothink; -n 5000; SAME prompt file; DS4_DSPARK_PERF=1 always.
- Divergence between configs is EXPECTED at temp0 on this content (tie-noise;
  see memory temp0-large-prefill-nondeterminism) — score by t/s + tau + block ms,
  not byte-match. Byte gate only via small-prefill low-tie canary when promoting
  a default.
- Report per run: gen t/s, tau, block/draft/verify ms, blocks (perf) vs blocks
  (scheduler footer) — the DELTA is the degenerate-block counter.

## Work queue
1. BASELINES: ds4-ssd GGUF no-draft AND resident sidecar no-draft on this prompt
   (-n 5000). Establish base and the +10 target number.
2. DEGENERATE-BLOCK HUNT (the seed anomaly): instrument/log why perf-blocks >>
   scheduler-blocks (ds4.c normal vs frontier paths; check draft-fail returns,
   room/ctx caps, min_commit=1 paths, EOS truncation). Env-gated one-line reason
   counter per degenerate block. Fix or route around; re-measure. Expected +
   several t/s alone (tau 2.64 -> ~3.5+).
3. SCHEDULER A/B on this content: default confidence vs static vs confidence-cost;
   check conf 0.4 over-trimming (sched 4.42) vs static tau/verify tradeoff.
4. FRONTIER + ROWS-6: DS4_DSPARK_FRONTIER_DRAFT=1, then +DS4_DSPARK_VERIFY_ROWS6=1
   (code content: tau ~4.8, first-miss ~0; my A/B: 1.27x). Watch rows-6 verify
   super-linearity — net win on code was real (45.01 vs 44.78 static).
5. KNOWN SMALL WINS if not default in tree: DS4_DSPARK_VERIFY_SPLIT_LAYERS=4
   (+2.3 t/s monolithic), draft prefetch (+0.35).
2b. CONFIDENCE-HEAD CALIBRATION (user ask "is it truthful?"): env-gated counters
   (DS4_DSPARK_CONF_CALIB=1): bucket predicted per-position accept prob p (10 bins)
   vs REALIZED acceptance; print reliability table + ECE at exit. Compare against
   the measured conditional acceptance (88-92% flat) — if head predicts ~0.5-0.7
   where realized is ~0.9, it is UNDER-confident -> conf 0.4 threshold + cost
   scheduler are mis-calibrated; consider per-position recalibration (scale/bias
   on the logit, byte-safe scheduling-only change) before touching kernels.
2c. TARGET-QUANT ACCEPTANCE A/B (user hypothesis: quantized TARGET lowers acceptance):
   same draft package, target = /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash
   (must run in ds4-ssd) vs the IQ2XXS GGUF, same prompt, measure acceptance/tau/t-s.
   Acceptance = P(draft argmax == QUANTIZED-target argmax); a closer-to-bf16 target
   may agree more with the draft. CAUTION from memory: the MXFP4 DRAFT-package swap
   was already tested and REJECTED (acceptance 77.1->73.0) — that was the draft side;
   this item is the TARGET side, a different question. Verify model kind/paths load
   in ds4-ssd first (MXFP4 native sidecar support memory).
2d. SSD/SIDECAR DSPARK WORKFLOW: repeat winning levers (items 2-5 results) on the
   sidecar config (slot-bank resident); it lags GGUF ~1-3 t/s in prior data; measure
   paired no-draft vs draft there too; long-ctx rows gate already defaults on.
6. If still short: stack best-of (scheduler x frontier x rows6 x split) and
   re-measure; document residual gap vs the parked async-overlap refactor
   (+12-15%, needs user sign-off — do NOT attempt in-loop).

## Loop protocol
Same as VERIFIER_OPT_LOOP.md: read ## Feedback first (re-read fresh, surgical
writes), one queue item per iteration, append ## Log, overwrite ## Status,
never commit. Memory refs: sidecar-dspark-verify-economics,
dspark-confidence-head-texttype-ab, strict-dspark-55-loop-state.

## Status
current-item: 2026-07-10 RESUME HARDENING PASS. `BATCH_BYTE_SAFE` is now quarantined:
it no longer changes generic Metal routed-MoE dispatch or target decode, and an explicit
request logs that strict exact-row evaluation is retained. Native sidecar gate (resident,
n=160) is full-file equal for no-draft, strict DSpark, and the quarantined flag; all three
report Apple M5 Max. GGUF parity smoke remains green for static, confidence, and rate.
No default or performance claim was made. The THREE_WAY graph stamp is cleared at block
exit; its dedicated two-turn agent tail gate remains pending because the first canary was
not a parity-stable baseline. Next speed work is still blocked on a user choice: promote
champion+m6 defaults or authorize the ANE-draft path.

previous-item: 4AM 2026-07-10 RESUME SCHEDULED (session cron 084a0b8d). Overnight yolo
round (bench-results/yolo-20260709-round2/: STATUS_4AM.md/CHAMPION.md/OVERNIGHT.md) found
CHAMPION (default-OFF): DS4_DSPARK_FORCE_TARGET_FIRST=1 + CONF_SCALE=0.85 +
CONF_THRESHOLD=0.50 -> BEATS no-draft +0.37 @n4000 (3 reps) and +0.99 @n5000, first-miss
0%, full-accept ~81%. Margin gate (EVAL_MARGIN_GATE_THRESHOLD=4.5, +2.0 standalone) does
NOT stack with force. Resume agenda: (a) champion cross-content + byte-parity promotion
gates, (b) margin-vs-force per content type, (c) sidecar parity break (still open: HEAD
worktree + no-draft ref ready), (d) ANE draft = the residual +10 path (needs user
sign-off). Rejected overnight: lazy main-KV always, force+margin, margin+rate,
MARKOV_SCALE!=1, verify-4, scale<=0.80 long-ctx.

previous-item: BYTE-PARITY ROOT CAUSE (user-directed 2026-07-08: "check changes and fix byte
parity, LMK what caused it"). FACTS: every DSpark scheduler (static/conf/rate) diverges
from no-draft at the same early point (seg char74/prose 250/json 602, pos ~20-150, all
paths deterministic run-to-run); OLD Jul-6 refs also diverge (char 240, prompt.txt) =>
break predates user's 3 fixes AND A1-A4; committed HEAD 3950b4b8 = PARITY MATCH (worktree
build); HEAD + current kernel files (ds4_metal.m/ds4_gpu.h/metal/dsv4_misc.metal/
diagnostics) = MATCH => culprit is in the UNCOMMITTED ds4.c host diff (93 hunks).
Draft-side exonerated by construction (committed bytes = verify-row argmax regardless of
proposals). budget-1 probe moves divergence 74->1134 => multi-row verify arm implicated
plus a weaker residual. NOW: binary-searching the 93 ds4.c hunks in the HEAD worktree
(scratchpad/bisect_test.sh, pins 1,2,32,36 for A1-A3 plumbing).
PARKED until parity fixed: rate-scheduler promotion (seg 37.64 = beats conf0.4 36.95,
-0.5 from no-draft; json 36.72; byte gate blocked on this bug). | blocked-on: bisect
ROOT CAUSE FOUND + FIXED (2026-07-08): the uncommitted "monolithic row-exact tiny batch
path" in metal_graph_encode_layer_routed_exact_rows (ds4.c ~18233) intercepts the verify
row-routed MoE with ONE metal_graph_routed_moe_batch_tiled call over all <=5 rows instead
of the per-row loop; the tiled multi-row reduction groups expert sums differently from the
single-row calls that bit-match plain decode => verify logits drift ULPs, near-tie argmax
flips, DSpark != no-draft (seg char74). Isolated by 93-hunk bisect in a HEAD worktree
(hunk 23 alone breaks it; all-minus-23 = PARITY OK). It rode the defaulted-on
DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT env; entered the tree between HEAD 3950b4b8
and Jul-5 evening (VERIFIER_OPT_LOOP's Jul-5 stage inventory already lists it; not part of
user's 3 fixes, not the schedulers). FIX: gated behind new explicit opt-in
DS4_DSPARK_HYBRID_ROW_ROUTED_MONO_BATCH (default OFF) + comment. Post-fix gate: GGUF
static+rate vs no-draft PARITY OK (n=300 seg). PERF COST OF FIX: ~ZERO — paired n=1500
static A/B: mono-off 34.41 t/s verify 66.42ms | mono-on 34.63 verify 68.73 (within noise;
the inexact path bought no real speed). tests/dspark_parity_smoke.sh added (no-draft vs
static/confidence/rate cmp) — run it before promoting ANY verify-arm change.
SIDECAR = SECOND SEPARATE BREAK (flash_moe banked arm, mono path is !flash_moe-gated):
still diverges post-fix at char 800; sidecar no-draft IS self-deterministic (repeat match)
and its CONTENT moved HEAD->current at char 405 (something in the uncommitted ssd/flash
hunks changed sidecar decode numerics). Missing datum: HEAD sidecar parity (run killed by
box contention; HEAD worktree + its no-draft ref ready in scratchpad/ds4-head).
- 2026-07-08 USER-PAIR ANSWER ("why no win"): sidecar, user prompt, -c 20000 -n 4000:
  | run | w1 | w2 | w3 | w4 | cum |
  | no-draft        | 38.68 | 38.17 | 37.63 | 37.22 | 37.92 |
  | dspark conf0.4  | 41.54 | 37.46 | 35.31 | 35.39 | 37.26 |
  | dspark rate v1  | 40.41 | 36.38 | 35.33 | 35.09 | 36.68 |
  | rate v2 (probe128/off4) | 40.92 | 36.95 | 36.63 | 35.84 | 37.48 |
  | rate v3 (+dormant prep skip) | 40.61 | 36.96 | 36.35 | 36.02 | 37.40 |
  DSpark WINS +2.9 in the first ~1.2K tokens, then inverts: tau decays (~3.3->2.9) as the
  generated code deepens while verify's ctx slope outruns decode's => break-even tau rises
  as actual tau falls. Nothing broken (verify 97.7-98.3% GPU-busy exact math). Scheduling
  caps losses but cannot flip the sign: the dormant floor sits ~1.1 t/s below no-draft =
  draft-residency tax (per-token dspark main-KV mirror update runs whenever a draft is
  loaded, ds4.c 12976; prep-skip for seed+3-layer probe added, measured ~neutral) + 0.5ms
  probe tax. Rate v2/v3 = best DSpark config on this workload (37.4-37.5 vs conf0.4 37.26).
  RATE SCHEDULER TUNING LANDED: DS4_DSPARK_RATE_PROBE_INTERVAL (default 128, was 32),
  DS4_DSPARK_RATE_OFF_STREAK (default 4, was 8), dormant prep-skip (byte-safe, prepares
  the token before each probe). Parity smoke passes static/confidence/rate on final build.
  => To actually WIN at ctx>1.5K: (a) ANE draft (removes 10.6ms/block AND can absorb the
  mirror tax; break-even tau drops ~3.16->~2.75 = whole run wins), (b) draft retrain for
  tau (external), (c) verify cost — no lever found (efficient + exact). ANE-draft is now
  unambiguously the top lever (= DECISION POINT #2 option a).

previous-item: RATE SCHEDULER VALIDATION. Sidecar-gap item CLOSED (full parity, user's -1.1
table was a content-draw; see Log). New headline: on the post-fix baseline (26.2ms decode)
DSpark static is net-negative at short ctx (break-even tau ~3.5); conf0.4 recovers most but
prose still -7.4 vs no-draft and frontier's min_commit>=1 clamp blocks skipping in the
user's default config. IMPLEMENTED --draft-scheduler rate (k=0 skip, online calibration,
dormant hysteresis; byte-safe scheduling-only) - A/B running in
bench-results/rate-sched-20260707 (rate vs conf-cost vs conf0.4 same-build, GGUF n=2500
seg/json/prose). Promotion gates: (a) byte-identical .out vs no-draft per prompt, (b) rate
>= conf0.4 on seg/json AND rate strictly > conf0.4 on prose, (c) no regression vs no-draft
worse than -1 t/s on any content. If pass: propose rate as default scheduler + revisit
frontier skip semantics. | DECISION POINT #2 (post-A4 levers: ANE-draft concurrency / draft
GPU-time reduction / accept current) still open for user. | blocked-on: A/B ~10min

yolo-item 2026-07-09: ROWS6 HOST-BOUND REPAIR RETAINED. The Flash row-exact Metal
wrappers already accept six rows, but both host routing gates stopped at five and the two
varmap metadata builders accepted six while allocating three five-element arrays. Gates
are now `n_tokens <= 6` and both metadata sets are sized to six. Same-prompt n=1000 A/B:
37.38 -> 38.49 t/s, verifier 100.51 -> 97.12 ms/block, GPU-busy 99.10 -> 95.79 ms,
with identical tau 4.25 / acceptance 70.9% and byte-identical output. A separate n=300
rows6-vs-no-draft smoke is also byte-identical. That first patch covered the
static/frontier rows6 path only.

Yolo continuation: all three deferred items now have implementations and measurements.
Margin-gated target eval can omit DSpark prep/main-KV behind
`DS4_DSPARK_TRUE_PLAIN_SKIP=1`; missing mirror rows are repaired in the same command
buffer as the next live draft, avoiding the prototype's extra blocking drain. Verifier
and draft single-drain paths are default-on. Confidence-aware Mode B is available with
`DS4_DSPARK_VERIFY_ROWS6=1 DS4_DSPARK_ROWS6_CONFIDENCE=1`, and now applies the same
confidence scale/bias/calibration remap as the normal scheduler. Best exact-workload
result: 44.71 t/s at n=1000 and 40.89 t/s at n=4000, versus same-build plain-skip
controls 41.74 mean at n=1000 and 39.55 at n=4000. Keep Mode B opt-in: n=300 no-draft
and n=1000 control parity pass, but the n=4000 outputs first diverge at byte 8333.

## Log
- 2026-07-06 items 1/3/4/2c (GGUF, space-invaders prompt, -n 5000 -c 40000, paired):
  | config | gen t/s | tau | block ms | accept |
  | no-draft baseline | 33.58 | - | - | - |
  | dspark default(conf) | ~35.4-35.5 | 2.74-2.83 | 76.7-79.8 | 67.1% (pos1 78.3!) |
  | static5+frontier | 32.89 | 2.64 | 80.3 | REJECT (conf wins) |
  | rows6+frontier+static | 32.53 | 4.27 | 130.7 | REJECT (verify superlinear; -c4096 win didn't transfer) |
  | MXFP4 target (2c, streaming sb64, -n1500) | n/a (SSD-bound) | 2.65 | - | 62.4% (pos1 73.4) => HYPOTHESIS REFUTED |
  NOTE: rows-6 does NOT engage under the confidence scheduler (identical stats to frontier5) — gate interaction, worth a bug note.
  Degenerate blocks: only 3.4% in this pair (42/1223) vs 22-25% in user's -n 7000 runs — reason likely late-generation EOS/array regions; deprioritized.
  MXFP4 sidecar = 137GiB bank > RAM: full residency impossible on this box; acceptance-only comparisons valid.
- VERDICT: acceptance on this content is draft-intrinsic (67% vs canary 79%); tau 2.7-2.8 bound.
  +10 t/s (43.6) needs tau 3.5+@80ms or verify ~50ms — below the proven byte-safe verify floor
  (loop-55: ceiling ~44 t/s = +8.5 over 35.76 baseline, best-ever strict). Remaining in-scope
  levers: 2b calibration (+1-2?), async-overlap refactor (+12-15% = +4-5, needs sign-off).
  Realistic strict stack tops at ~+6-8 on this content. +10 requires: draft retrain (external),
  or relaxed accept (banned), or redefining target vs the sidecar baseline (29.96 → +10 = 39.96,
  reachable: GGUF+dspark already 35.4; async overlap could close it).
- 2026-07-07 USER RERUN DIGEST (segments-rerun-20260707-220042, conf scheduler + frontier +
  overlap on): sidecar 32.02 (block 70.95, draft 7.86, verify 61.82, tau 2.39, acc 76.5,
  overlap 623/623 hits) | GGUF 33.14 (75.21, 8.07, 65.78, tau 2.60, acc 78.9, 487/487) |
  upstream no-draft 32.97. User fixes landed: decode indexer sparse threshold 512->1024
  default (restored GGUF no-draft parity), DS4_PROGRESS_1K decode windows, Flash-MoE
  prefetch/async-pread gated to true partial-streaming banks only (resident now reports
  async-pread=off bank-prefetch=0 slot-cache-topk=0). KEY REVERSAL: sidecar verify no longer
  slower than GGUF — sidecar deficit is all tau. NOTE overlap engaged on only ~40% of blocks
  under conf scheduler (623/1489, 487/1302) leaving draft ~8ms in accounting — engagement gap
  is real but wall-value ~0 per A4 verdict (serial queue), not worth chasing.
- 2026-07-07 sidecar-gap matrix LAUNCHED (bench-results/sidecar-gap-ab-20260707): static
  scheduler isolates verify-cost comparison + unmodulated acceptance; 3 content types probe
  tau-gap stability; no-draft seg pair anchors backend baseline gap post prefetch-fix.
- 2026-07-07 SIDECAR GAP CLOSED + NEW HEADLINE (sidecar-gap-ab-20260707, static sched, n=2500):
  | prompt | side t/s | gguf t/s | side tau | gguf tau | side verify | gguf verify |
  | seg    | 33.15 | 33.49 | 2.94 | 3.06 | 71.71 | 74.09 |
  | json   | 31.91 | 31.45 | 2.51 | 2.26 | 64.04 | 58.76 |
  | prose  | 24.92 | 24.28 | 1.59 | 1.51 | 58.80 | 58.49 |
  | seg NO-DRAFT | 38.16 | 38.17 | - | - | - | - |
  (1) sidecar==gguf no-draft to 0.01; tau-gap direction flips by content; sidecar wins 2/3
  cells => user's -1.1 table was a single content-draw, sidecar at FULL PARITY. Item closed.
  (2) DSpark STATIC is NET-NEGATIVE vs no-draft everywhere at short ctx on the new baseline
  (user's threshold fix sped decode to ~26.2ms/tok; verify didn't inherit: verify(5)/decode
  = 2.74x; break-even tau ~3.5 > actual 1.5-3.1).
  (3) Block accounting EXCLUDES the per-block first-token eval (~26ms, ds4.c:32489 pre-
  dspark_t0) + ~7ms misc: wall/block 119ms vs accounted 84ms. Not waste - structural
  (correction token needs its own decode; explains why A4's hidden 10.5ms bought +0.2 wall).
- 2026-07-07 THRESHOLD SWEEP at new baseline (sched-thresh-20260707, GGUF n=2500):
  | content | static5 | conf0.4 | conf0.9 | no-draft |
  | seg   | 33.49 | 36.95 (tau 2.75) | 36.14 (tau 1.43) | 38.16 |
  | json  | 31.45 | 37.24 (tau 2.22) | 36.68 | 38.08 |
  | prose | 24.28 | 30.68 | 28.58 | 38.08 |
  conf0.4 >> static everywhere now; th0.9 WORSE than 0.4 on prose => trimming floor = the
  10.5ms draft still paid on 0-scheduled blocks (eval 26 + draft 10.5 = ~27 t/s floor).
  NOTE: user's rerun ran FRONTIER_DRAFT=1 whose scheduler CLAMPS scheduled>=1 (min_commit,
  ds4.c ~31998) - frontier can never skip-verify => user's 33.14 vs my 36.95 on seg is
  plausibly mostly the frontier clamp. Recommend default runs WITHOUT frontier on mixed
  content until frontier gets the same skip semantics.
- 2026-07-07 RATE SCHEDULER IMPLEMENTED (--draft-scheduler rate, byte-safe scheduling-only):
  argmax_k (1+E[extra|k])/(T_eval+cost(k)) incl k=0 skip; head calibrated ONLINE vs realized
  commit rates (decile bins >=32 samples, g_conf_calib_* now always accumulate when rate
  active); T_eval = live EMA of the per-block first-token eval (ds4.c:32489 site); cost(k) =
  dynamic recorder per-budget EMAs (rate reuses cfg.confidence_cost record-only contract);
  draft treated as sunk (EMA-subtracted); +1 bonus token credited (conf-cost omits it -
  known flaw). DORMANT hysteresis: 8 consecutive k=0 => draft skipped entirely (plain
  decode), probe block every 32 tokens to re-enter - kills the 10.5ms draft tax on hopeless
  content (the th0.9-worse-than-th0.4 lesson). Frontier path wired with allow_skip=false
  (its row 0 IS the committed token). A/B RUNNING: bench-results/rate-sched-20260707.
- 2026-07-09 CHAMPION PROMOTION GATES (bench-results/champion-gates-20260709): GATE1 byte
  parity (GGUF seg n=300, champion envs) = OK byte-identical (40.02 vs nodraft 38.74).
  GATE2 cross-content (sidecar n=2500): json champ 39.00 vs nodraft 38.17 = +0.83 WIN
  (accept 92.7, full-accept 90.7); prose champ 32.98 vs nodraft 38.23 = -5.25 VETO.
  Prose loss structural: force clamps conf scheduler to >=1 row => every block pays
  eval+draft+verify(>=1) ~87ms for ~2.5 tok at tau 1.54. Champion NOT a global default
  as-is. TESTING: force + rate scheduler (k=0/dormant escape; force+rate untried
  overnight — only force+margin and margin+rate rejected, both arcade-only): expect
  champion wins on json/arcade + ~no-draft dormant floor on prose.
- 2026-07-09 FORCE+RATE = REJECT (v1 AND v2). v1: rate never skipped under force — TWO
  bugs found+fixed: (1) ds4.c ~33440 clamped scheduled>=1 under force even for rate (the
  "free accept" costs verify(1) ~35ms: k=1 forced block = 2 tok/63ms = 32 t/s < plain
  37.7 — now rate-exempt); (2) force fed the head's NATURAL score for the certain row
  into calibration with guaranteed commits => reliability bins inflated => rate never
  priced k=0 (fixed: conf[0] pinned 1.0 for schedulers, sentinel -1 excludes the row
  from calib bins, rate bypasses bin-lookup for >=0.999 certainty). v2 results: prose
  35.98 (+3.4 vs v1, still -2.25 vs nodraft), json 36.96 (WORSE than v1; champion 39.00),
  arc 37.59. DS4_DSPARK_CONF_LOG debug exposed TWO DEATH SPIRALS in dormant mode:
  (a) post-dormancy blocks run cold -> inflated cost EMAs (cost[5] 86->94ms vs champion's
  real 71) -> more dormancy; (b) probe drafts run on STALE trunk/markov state from the
  dormant gap -> probe acceptance craters (calib 0.3-0.5 where champion sees 92.7%) ->
  calibration learns rows don't commit -> stays dormant on WINNING content (json 73%
  dormant). Rate-under-force parked; estimator hardening (probe bursts, contaminated-
  sample exclusion, cold-block cost quarantine) = backlog. ALSO: early t_eval EMA needs
  warmup skip (first samples ~42ms vs true 26.4 — ctx-grow/session init in the tail).
  NOW TESTING: champion + margin PRE-gate (skips before draft, prep stays on => no
  staleness; overnight rejected it on arcade only, never on prose where champion needs
  the escape): {prose,json} n=2500 + arc n=4000.
- 2026-07-09 NEW CHAMPION: force + conf scale/th + MARGIN GATE 6 (champion-gates-20260709).
  The overnight force+margin rejection was a THRESHOLD ARTIFACT (tested at 4.5 where
  arcade loses; at 6 it wins arcade too). Full matrix (sidecar, n=2500/4000):
  | content | old champ | +m4.5 | +m6 | +m8 | nodraft |
  | prose  | 32.98 | 35.11 | 35.85 | 35.99 | 38.23 |
  | json   | 39.00 | 40.77 | 40.94 | 41.08 | 38.17 |
  | arcade | ~38.09 | 37.59 | 38.47/38.14 (2 reps, mean 38.3) | 38.04 | 37.7-37.9 |
  WINNER ENVS: DS4_DSPARK_FORCE_TARGET_FIRST=1 DS4_DSPARK_CONF_SCALE=0.85
  DS4_DSPARK_CONF_THRESHOLD=0.50 DS4_DSPARK_EVAL_MARGIN_GATE_THRESHOLD=6
  (m8 trades arcade -0.3 for json/prose +0.14; m6 = best mean). PARITY GATE: OK
  (GGUF seg n=300 byte-identical vs no-draft). Dominates or ties every draft-loaded
  config on all three contents; json 41.08 = best number ever on this box (+2.9 vs
  nodraft). PROSE STRUCTURAL CEILING ~36 (draft-residency tax: per-token main-KV
  mirror + prep while a draft is loaded + surviving low-tau verifies) — prose with a
  draft loaded CANNOT reach no-draft until prep/mirror move off the per-token path
  (backlog: conditional post-eval prep; ANE absorbs it entirely).
  PROMOTION PROPOSAL (needs user sign-off, defaults currently opt-in): make these the
  defaults whenever --draft dspark is loaded, with _DISABLE escapes.
- 2026-07-09 ROWS6 BOUNDS + FAST-ROUTE FIX: `ds4.c` now permits six rows in both
  byte-exact Flash row-routed helpers, matching the existing Metal wrapper contract;
  `ds4_metal.m` varmap/direct-varmap row metadata arrays are now six elements, removing
  the host out-of-bounds access. The patched run selected `row-exact tiny batch path`.
  Paired n=1000 before/after at 88% VM pressure: 37.38/38.49 t/s, block
  113.18/109.77 ms, verify 100.51/97.12 ms, draft unchanged at 9.95 ms. Outputs match
  the prepatch fast run and the row-exact fallback byte-for-byte; fresh n=300 strict
  parity against no-draft also passes. Retain. This was the pre-continuation conclusion:
  confidence scheduling still excluded rows6 and 1.4% idle made sync work look too small.
  The next entry supersedes both points with confidence-aware rows6 and measured
  single-drain results. The prior ANE audit still rejects full draft migration, and the
  source checkpoint needed for the documented FP8-to-MXFP4 draft re-export is unmounted.
- 2026-07-09 YOLO ITEMS 2/3/4 COMPLETED (same sidecar, exact user arcade prompt):
  - **True plain skip:** the pre-existing default-off prototype initially measured
    41.51 t/s versus 41.90 control because deferred main-KV repair drained separately.
    Fusing the suffix import into the live draft command buffer is byte-clean and
    measured 42.07/42.08 t/s versus 41.90/41.58 controls (means 42.08 vs 41.74).
    Keep `DS4_DSPARK_TRUE_PLAIN_SKIP=1` opt-in pending cross-content promotion.
  - **Drain removal:** default-on verifier+draft single-drain measured 42.08 t/s and
    2.1% verifier idle; disabling both measured 41.27 t/s and 4.3% idle. Output matched.
    Retain the single terminal wait and the existing nonblocking layer splits.
  - **Architecture / confidence-aware Mode B:** rows6-confidence folds the separate
    target token into a six-row verifier batch. Fixing its ignored `CONF_SCALE=0.85`
    improved n=1000 from 43.92 to 44.71 t/s (verify 87.65 -> 83.02 ms) and n=4000
    from 40.29 to 40.89 t/s. Same-build n=4000 plain-skip control is 39.55 t/s;
    rows6 wins every 1K window after the remap (44.50/43.66/39.04/37.27 versus
    42.04/41.68/38.32/36.68). n=300 no-draft parity and n=1000 control parity pass;
    n=4000 first differs at byte 8333/line 244, so Mode B stays opt-in rather than a
    global strict default. Scaled and unscaled Mode-B outputs are identical at n=4000.

- 2026-07-10 4AM RESUME. Overnight: commit 8cd83523 (absorbed most of the tree incl the
  mono-batch parity fix) + ~1.2K new uncommitted lines (Mode B adaptive, rows6
  adaptive/cost-aware, BATCH_BYTE_SAFE row-routed, FFN NAX co-issue audit). GATES ON THE
  REBUILT TREE: GGUF parity smoke OK (static/confidence/rate). SIDECAR PARITY ITEM (c)
  CLOSED: HEAD-3950b sidecar pair = PARITY OK (the missing Jul-8 datum, run in the
  preserved worktree) AND current-tree sidecar = PARITY OK on seg + json => the sidecar
  break was introduced after 3950b4b8 and FIXED by the overnight work; no bisect needed.
  NEXT: bench overnight opt-ins vs champion+margin6 (diff-audit workflow brief pending),
  then the open decisions: champion-default promotion + ANE draft.
- 2026-07-10 4AM SESSION RESULTS (rebuilt tree, sidecar resident, arcade n=4000 /
  prose n=2500): re-anchor champion+m6 38.18 vs nodraft 37.73 (+0.45, tau 3.75 —
  yesterday's result holds on the new build). Overnight opt-ins benched:
  | config | arcade | prose |
  | modeb-adaptive + rows6 | 37.21 | 36.26 (best draft-prose ever, 96% dormant) |
  | rows6-cost-aware       | 35.57 (verify 115ms, no dormancy) | - |
  VERDICT: champion+m6 stays overall champion; Mode-B router correctly limits rows-6
  damage (dormancy) but cannot beat plain+force economics; rows-6 verify superlinearity
  (~110-115ms) remains the binding constraint on every rows-6 shape.
  DIFF-AUDIT BRIEF (workflow, 3 agents): overnight diff has ZERO default-on numeric
  changes (parity smoke agrees). THREE HARDENING ITEMS in opt-in arms before anyone
  benches them broadly: (1) BYTE_SAFE promotion is process-wide/permanent off possibly
  ONE small-block audit -> should be per-shape (track max n_tokens audited); (2)
  THREE_WAY stamps dspark_router_route into the graph and never clears it -> later
  tail-prefill can write KV in batched FP order = byte drift armed by THREE_WAY=1 alone
  (clear the stamp at block end; two-turn agent byte gate required); (3) the FFN
  co-issue AUDIT env reroutes ALL small q8 matmuls process-wide -> never set during
  other benches. Also: BATCH_ROW_EXACT semantics silently changed for existing users
  (no longer implies FUSED_ORDERED_Q2) — re-validate anything byte-pinned to it.
  OPEN DECISIONS FOR USER: (a) promote champion+m6 defaults-when-draft-loaded,
  (b) ANE draft build sign-off, (c) whether to apply the three hardening fixes to the
  overnight opt-in arms.

- 2026-07-10 RESUME HARDENING — BATCH_BYTE_SAFE CONTAINMENT. Source audit found that
   `DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_BYTE_SAFE` was not local to its comparison: it
   changed the generic banked-MoE selector used by ordinary target decode and MTP/prefill,
   while its candidate also wrote live routed scratch. A rejected audit therefore could
   still alter un-audited math. The flag is now an explicit strict-path quarantine in
   `ds4.c`, and `ds4_metal.m` no longer consumes it in either generic selector. The
   dormant audit state remains per graph and exact row count; a future revival requires
   isolated candidate out/gate/up/mid/expert scratch, full-layer failure poisoning, and
   shape-specific promotion. Native resident sidecar gate (`-n 160`, Space Invader): all
   exits 0; no-draft == strict == flag byte-for-byte (427 bytes), with Apple M5 Max
   detected in every run. No-draft leak sentinel with only the flag set also cmp'd clean,
   emitted no candidate log, and saw Apple M5 Max. Normal `tests/dspark_parity_smoke.sh
   300` remains byte-clean for static/confidence/rate (1134 bytes each). This is safety
   containment, not a throughput result.

## Feedback
(empty)
- 2026-07-06 item 2b DONE (confidence-head calibration + threshold sweep):
  DS4_DSPARK_CONF_CALIB=1 counters added (ds4.c: stash at scheduler entry, central record in
  perf_record — first attempt at commit-sites only was success-biased, realized=1.0; fixed).
  UNBIASED reliability (3390 samples): pred 0.98->realized 0.76 | 0.85->0.61 | 0.75->0.51 |
  0.65->0.44 | ECE=0.20 => head is OVER-confident ~20pts; the >0.9 bin (53% of rows) realizes
  only 76% = draft-side tau ceiling made visible. Threshold sweep (n=3000, same prompt):
  0.4: 35.95 t/s (tau 2.68, 73.8ms) | 0.55: 36.50 (2.22, 60.1) | 0.7: 35.84 (1.90, 52.7).
  CANDIDATE: --draft-conf-threshold default 0.4 -> 0.55 (+0.5-0.6, scheduling-only byte-safe);
  needs one repeat pair before promoting. Best now 36.50 vs baseline 33.58 (+2.9).
- NEXT (main line): ASYNC-OVERLAP REFACTOR (authorized). Design constraints from
  VERIFIER_OPT_LOOP item-2: draft trunk + verifier share batch_* scratch; hook must launch
  next-block draft AFTER verifier encode/submit, BEFORE readback/finish. Options:
  (A) duplicate draft scratch (memory cost ~draft trunk activations), (B) split verifier
  submit/readback seam. Expected +12-15% => ~40-41 t/s total = revised target met.
- 2026-07-06 threshold promotion VETOED by cross-content guard: 0.55 repeat held on
  space-invaders (36.51 vs 35.87 = +0.6, stable) but json regressed 39.02->38.10 (-2.4%),
  prose neutral (33.61/33.59). Default stays 0.4; 0.55 documented as a code-workload knob
  (--draft-conf-threshold 0.55). Item 2b CLOSED. All scheduling levers now exhausted;
  remaining path to +6-8 = async-overlap refactor (in progress, recon running).
- 2026-07-07 A1-A3 LANDED (private dspark_draft_* scratch bank + prefetch routing):
  byte gates ALL PASS (prefetch on/off/pre-change reference identical across 3 pairings:
  A==B==REF, C==D, E==F==REF3). Perf NEUTRAL as designed (33.6-33.7 C/D; 34.1-34.2 E/F/REF3;
  draft ~10.2-10.6ms unchanged) — A1-A3 is the byte-safe ENABLER; launch still post-commit.
  Working tree: ds4.c/ds4_metal.m/ds4_gpu.h/ssd allocation. NEXT: stage A4 (pre-readback
  launch, GPU-argmax seed, speculative KV append + finish-guard discard) = the ~10.5ms/block
  prize -> target ~40-41.
- 2026-07-07 A4 LANDED (overlap draft, DS4_DSPARK_OVERLAP_DRAFT=1, default OFF):
  BYTE GATES ALL PASS (GGUF -n 2000 -c 40000 verify5 temp0 nothink; cmp off==on):
  p1 prompt.txt (miss-heavy) MATCH, p2 inline space-invaders (high-accept) MATCH,
  n=3000 perf pair outputs also identical; regression guard: off-run == /tmp/ovlREF.out
  == /tmp/ovlA.out byte-exact (REF3 was the other-config E/F run, not reproducible from
  recorded flags). Frontier seam pair (FRONTIER_DRAFT=1, scheduler static, n=600) MATCH
  with real misspec traffic (111 hits / 42 fallbacks -> snapshot-restore exercised).
  PERF (n=3000, p1, 20s cooldowns, DS4_DSPARK_PERF=1):
  | config | gen t/s | block ms | draft ms | hits/launched |
  | off    | 35.90   | 73.79    | 10.52    | -             |
  | on     | 36.11   | 63.32    | 0.04     | 816/816, 0 fallbacks |
  Draft fully hidden from the block path (-10.47 ms/block as designed) BUT wall gain is
  only +0.2 t/s: the 10.5ms draft is GPU-EXECUTION dominated and the M5 queue is serial
  (verify GPU-busy 97%), so same-queue overlap only recovers the launch/drain overhead
  (~0.6ms/cycle). The +12-15% async-overlap estimate assumed round-trip-dominated draft
  time; measurement disproves that premise. Remaining headroom needs GPU-time reduction
  or ANE concurrency, not scheduling.
  DESIGN DEVIATIONS (vs DSPARK_OVERLAP_DESIGN.md A4):
  (1) NORMAL path (default config) launches at the EVAL seam, not the verifier seam: the
  first-token decode imports DSpark main-KV for its own position (ds4.c decode
  dspark_main_kv stage) BEFORE the live draft reads it; a pre-readback chain cannot see
  that import, drafts/confidences diverge, block splits shift, and near-tie rows flip
  bytes (observed: first B1 attempt failed cmp at char 500). The eval seam encodes the
  mirror draft AFTER the decode's import in the same submit -> chain is the live draft's
  exact twin, hits are deterministic (100%).
  (2) Verifier-seam speculative KV import (frontier paths) is NOT self-healing: ring slot
  p%128 aliases live window history for position p-128, and target_hidden history is
  gone by repair time, so "re-encode the range" cannot heal it. Fix: 41KB pre-import
  snapshot (dspark_draft_kv_backup) + suffix restore on misspec (misspec_repair at
  commit-prefix time, full restore on verify failure).
  (3) Confidence-hard/softmax schedulers (the DEFAULT --draft-verify config) are now
  supported by the overlap chain: per-row Markov embeds saved on the mirror
  (markov_chain_fast_encode save flag) + mirror-bank confidence head scoring at
  consumption; markov-logits scratch also mirrored (dspark_draft_markov_logits) so the
  chain can never clobber g->logits pre-readback.
  Stats footer: "ds4: dspark overlap-draft: launched=N hits=N fallbacks=N" (atexit, only
  when the env is set and at least one launch happened).
