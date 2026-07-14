# Case E (NAX verifier attention) + verifier profiling — handoff

Continuation doc for the next agent. Covers: per-block timing splits, every test
run since HEAD (`193a5287`), the methodology traps that burned cycles, and where
to take it next. **Bottom line: Case E is a real but small (~5%, ~2–3 t/s) win,
only below ~8k context, and it is NOT byte-identical. The first top-k sparse
prototype regressed at the standard c4096 target, so Case E should remain a
long-context opt-in diagnostic. The larger >50 t/s levers are MoE route-dedup,
tau/tree scheduling, real Mode B, or separate ANE draft overlap rather than more
attention fusion.**

---

## 2026-06-30 Pro Feedback Update

Fresh current-gate recheck: Case E should remain demoted. With the newer
frontier relaxed `top128/delta6` gate and the same forced-MMA/fast-Q2 control,
`bench-results/dspark_goal_continue_125014_casee_current_gate_n300` measured
forced-MMA at `48.98 t/s`, while Case E simple was `44.85 t/s` and Case E
fast-AV was `45.57 t/s`. This is a direct A/B against the current best relaxed
frontier stack, not the older strict-only baseline. Do not spend more short
context work on Case E unless it changes real verifier compute or targets
long-context selected-top-k attention.

Fresh paired n=1000, `-c 4096`, sidecar resident sweep after adding conditional
acceptance logging:

| mode | gen t/s | vs no-draft | verify ms | tau | full accept |
|---|---:|---:|---:|---:|---:|
| no draft | 35.95 | 1.00x | n/a | n/a | n/a |
| strict b5 | 39.85 | 1.11x | 77.09 | 4.85 | 62.1% |
| strict b4 | 39.64 | 1.10x | 64.71 | 4.22 | 70.9% |
| Case E NAX b4 | 42.12 | 1.17x | 61.16 | 4.33 | 74.9% |
| forced MMA b5 | 44.58 | 1.24x | 62.86 | 4.78 | 62.7% |
| batch/unified b5 | 36.65-36.76 | 1.02x | ~83.3 | 4.69 | 54.0% |

Takeaways:

- Case E remains an opt-in fast mode, not a >50 t/s solution by itself.
- The Pro-agent recommendation is to keep Case E top-k sparse as the next
  Case-E-specific long-context experiment, but not as the main strict `cmp=0`
  optimization track. Strict work should move to MoE route-dedup,
  tau/full-accept recovery, and a separate draft-overlap audit.
- Current `--draft-mode batch|unified` is not the real Mode B speed path; it is
  slower than strict and should be treated as diagnostic until the target-forward
  contract is repaired.
- Conditional acceptance says pos1/pos2 are healthy. Strict b5 example:
  `1=87.9% 2|1=96.7% 3|1-2=92.0% 4|1-3=92.5% 5|1-4=85.9%`.
  Throughput recovery is therefore more likely from suffix/tree scheduling,
  MoE route-dedup, or true Mode B than from a simple first-token draft parity fix.
- Current-best forced-MMA b5 recheck:
  `bench-results/over50_post_stats_005658`, `44.58 t/s`, verify `62.86 ms`,
  tau `4.78`, conditional acceptance
  `1=86.1% 2|1=96.1% 3|1-2=93.1% 4|1-3=90.1% 5|1-4=90.3%`.
- Case E top-k sparse first prototype:
  `bench-results/nax_topk128_b5_n1000_011632` used selected top-k NAX with
  `DS4_DSPARK_INDEXER_TOP_K_OVERRIDE=128` and regressed to `39.19 t/s`, verify
  `74.44 ms`, tau `4.69`, acceptance `73.9%`. This does not invalidate the
  long-context selected-key idea, but it means the current gathered stream is not
  a short-context speed path. Validate any next attempt at 16k/64k/100k context
  and budgets 4/5, and rank by generation t/s plus tau.

## 1. Verdict / TL;DR

- **Case E = DSpark verifier attention on the Neural Accelerator (matmul2d)**, opt-in
  via `DS4_DSPARK_ATTN_NAX=1` (+ `DS4_DSPARK_ATTN_NAX_FAST_AV=1` for matmul2d P·V).
  Default left as **strict** (byte-identical) after a failed default-on attempt.
- **Win is small and conditional:** +5% gen / −16% verify-ms ONLY when
  `n_comp ≤ indexer top_k (2048)`, i.e. **context < ~8k tokens**. Above that it
  attends ALL compressed keys dense while strict uses indexer **top-k sparse** →
  Case E is **much slower** at large context (e.g. −c 100096 ≈ 12× the attn work).
- **Not byte-identical:** FP16 QK staging + dense-vs-sparse → argmax flips → output
  diverges from no-draft greedy. strict-v1 is the cmp=0 path.
- **The verifier is efficient, not wasteful.** Real verify ≈ 62ms is genuine
  dense+MoE+attention compute; it is already well-batched (N=1..5 scaling is
  sublinear, ≈2.4× a single forward, not 5×). No single dispatch/redundancy lever.

## 2. Tree state (uncommitted, since HEAD `193a5287`)

Committed this session: `193a5287` = the condition-sweep harness
(`scripts/dspark_condition_sweep.sh` + `dspark_condition_parse.py`). Nothing else committed.

MINE, uncommitted (Case E):
- `metal/nax_fused.metal`: Case E kernels — `ds4_dspark_nax_gather`, `_qk_scores`,
  `_softmax` (per-token raw_off/raw_cnt/comp masking), `_av_simple` (FP32, the good
  one), `_pv`/`_vt`/`_otranspose` (matmul2d P·V), `ds4_dspark_nax_flash` (fused,
  DEAD — slower), `ds4_naalu_bench` (overlap microbench), `ds4_dense_m5_bench`.
- `ds4_metal.m`: `ds4_gpu_dspark_nax_attention_tensor` wrapper (gather→QK→softmax→AV,
  simple-AV default, fast-AV opt-in) + pipeline/scratch globals + the two bench hosts.
- `ds4.c` (~16417): opt-in gate + `nax_ok` graceful-fallback restructure (on Case E
  failure/unsupported HW, falls through to strict instead of erroring).
- `ds4_gpu.h`: wrapper decl.
- NOT mine (pre-existing in working tree): `ds4_server.c`, `dspark.c`, `dspark.h`,
  `metal/flash_attn.metal`.

Diagnostic env flags added (all opt-in, default off):
`DS4_DSPARK_ATTN_NAX`, `DS4_DSPARK_ATTN_NAX_FAST_AV`, `DS4_NA_ALU_OVERLAP_BENCH`,
`DS4_DENSE_M5_BENCH`.

## 3. Per-block timing splits (REAL, skip-based, unfenced)

**Methodology warning: the fenced subprofile (`DS4_DSPARK_ATTN_*_SUBPROFILE`) is an
~8× ARTIFACT** — it inserts a blocking flush per stage, exposing per-dispatch launch
latency that is normally hidden behind pipelining. It falsely reported comp/idx=53%,
hc_pre=38ms. **Only skip-based deltas (guard a stage behind `&& !env_flag_enabled(
"DS4_DSPARK_SKIP_X")`, measure verify-ms delta) are trustworthy.**

Real verify ≈ 62ms (n=200, budget 5, −c 4096) breaks down as:

| stage | real ms | % of verify | reducible? |
|---|---:|---:|---|
| dense projections (hc_pre/q/kv/output) | ~25 | ~37% | already batched M=5; bandwidth-bound (NAX≈ALU) |
| routed MoE + FFN | ~20 | ~30% | route reuse ~1.7× → grouped/dedup MoE (Phase 3) |
| fixed (embed, output head, sampling, KV, comp/idx loop) | ~16 | ~24% | comp/idx loop only ~6ms real (dispatch hidden) |
| attention heads | ~6 | ~9% | **Case E target — but smallest bucket at short ctx** |

Caveat: this split is at SHORT context. **Attention grows with context** (per-row key
scan ∝ compressed-cache size); at large ctx attention becomes the majority — which is
why the original `ATTN_BYPASS` measured ~79%. So the table's 9% is a short-ctx value.

Block wall (budget 5, −c 4096, Space Invaders): draft ~15ms + verify ~70–79ms +
overhead ~1.8 + commit ~1.4. Draft is NOT the bottleneck (~310 draft tok/s).

## 4. All tests since HEAD (numbers)

Correctness fix (n=200 Fibonacci, −c 4096, budget 5): Case E coherent at long ctx
after ring-offset + compressed-gather + per-token-window-offset fix.
- Case E fast-AV: 39.42 t/s, verify 61.6ms, acc 64.7%, tau 4.17
- Case E simple-AV: 38.28 t/s, verify 66.0ms, acc 65.5%
- strict: 37.49 t/s, verify 62.3ms, acc 60.3%

Long context (essay, n=2500, −c 4096, budget 5) — Case E wins where attention dominates:
- Case E fast-AV: 29.64 t/s, verify 56.68ms, acc 40.6%, tau 3.03
- strict: 27.32 t/s, verify 67.67ms, acc 42.0% → **−16% verify**

Condition sweep (essay, n=2500, −c 4096, budget 5, 2 reps, thermal cooldown):
- caseE_simpleav 28.37 t/s / caseE_fastav 27.84 / strict 26.25.
- **simple-AV > fast-AV on NET gen** despite higher verify-ms: fast-AV's FP16 P·V
  drops acceptance ~3pts. **Rank by gen t/s, never verify-ms.**
- Case E ahead at every position pos 0→1000, margin −9%→−16% (but ALL <8k context!).

Budget 2×2 (Space Invaders, n=1500, −c 4096) — clean, matched:
| config | gen t/s | verify ms | acc% | tau |
|---|---:|---:|---:|---:|
| Case E, budget 4 | **41.60** | 61.2 | 81.7 | 4.26 |
| Case E, budget 5 | 40.99 | 70.9 | 74.6 | 4.72 |
| strict, budget 5 | 40.67 | 79.0 | 80.2 | 5.00 |
| strict, budget 4 | 39.61 | 66.9 | 82.3 | 4.29 |
→ **strict prefers budget 5; Case E prefers budget 4** (Case E's approximation hurts
the deepest draft token, dropping b5 acceptance to 74.6%). Optimal = verifier×budget×ctx.

Microbenches (forks, synthetic):
- In-kernel **NA∥ALU co-issue is REAL on M5: 1.83× at balanced load** (`ds4_naalu_bench`),
  ~1.0 when imbalanced (smaller side hides free). CROSS-kernel does NOT overlap.
- **NA matmul2d at M=5: 0.32ms, no tall-skinny penalty** — viable, but near-parity with
  production F16 ALU since both bandwidth-bound at M=5.
- Fused attention-only kernel: DEAD (attention is NA-matmul-dominated, too little ALU
  to overlap → no win, was 2× slower with manual P·V).

## 5. Key findings & corrections (the traps)

1. **Fenced profiler = 8× artifact.** Use skip-based unfenced. Cost me the
   "comp/idx is 53%" and "hc_pre 38ms" wild goose chases (both ~6ms real / red herring).
2. **Validate across the indexer-sparsity crossover (~8k ctx), not just −c 4096.**
   Case E flips from winner to loser at n_comp > top_k=2048. My whole sweep was <8k.
3. **Rank by gen t/s, not verify-ms** — acceptance dominates (simple-AV beats fast-AV).
4. **NA∥ALU overlap only pays for balanced, compute-bound work.** Attention is NA-heavy;
   dense projs are bandwidth-bound → column-split won't help. Lever is narrow.
5. **Case E is not byte-identical.** If cmp=0 is required, Case E is out; strict-v1 only.

## 6. Suggestions for the next agent (prioritized)

A. **Case E selected top-k sparse is narrowed to a long-context experiment.**
   The first gathered selected-stream implementation regressed at c4096/n1000, so
   do not repeat that shape as a "quick win." A useful next Case E attempt must
   keep the selected key count bounded where dense all-compressed attention loses
   to strict top-k sparse, and must be judged by generation t/s, tau, acceptance,
   and coherence across 16k/64k/100k contexts.
B. **The strict >50 t/s lever is NOT more attention fusion.**
   - MoE: route reuse ~1.66–1.84× is real → grouped/dedup down-projection + ordered FP32
     sum (Phase 3, byte-safe). ~8ms potential.
   - Dense projections are demoted: they are bandwidth-bound at M=5 and NAX≈ALU.
     Only pursue if a non-fenced microbench shows a real generation-tps win; stop
     below ~8% verify-ms improvement.
   - Or **Mode B** (single batched forward, argmax-accept) for ~2× but non-identical.
3. **Auto-gate Case E** at `(n_comp ≤ indexer_top_k) AND (byte-identity not required)`,
   detected at runtime — only if A doesn't make it universal.
4. **Extend the harness** (`scripts/dspark_condition_sweep.sh`): default `CTX` spans the
   8k crossover (e.g. 4k/16k/64k), sweep budget 4 AND 5, REPEATS≥3 for acceptance noise.
5. Hardware: Neural Accelerator is M5-gen (per-core, all M5 tiers + A19); on M4 matmul2d
   falls back to ALU → Case E neutral-to-worse. Gate on GPU family if shipping broadly.

## 7. Reproduce

Gate: `export DS4_LOCK_FILE=/tmp/ds4-<name>.lock`; check
`ps -axo pid,command | grep -E '[./](ds4|ds4-agent|ds4-server)( |$)'` before resident runs.
```
# Case E vs strict, any budget/context:
DS4_DSPARK_ATTN_NAX=1 [DS4_DSPARK_ATTN_NAX_FAST_AV=1] DS4_DSPARK_BLOCK_TIMING=1 \
  ./ds4 -m <model> --draft dspark --draft-path <draft> --draft-verify <4|5> \
    --temp 0 --nothink -n 1500 -p "Make a game of Space Invader in Pygame" --resident -c 4096
# strict baseline: add DS4_DSPARK_ATTN_NAX=0
# multi-condition sweep:
CTX=16384 N=6000 BUDGET=4 REPEATS=3 scripts/dspark_condition_sweep.sh
# real per-stage cost: add `&& !env_flag_enabled("DS4_DSPARK_SKIP_<MOE|ATTN|FFN>")`
#   guards in metal_graph_encode_layer_batch (ds4.c ~18986) and read verify-ms deltas.
```
