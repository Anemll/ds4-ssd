# TASK: Cleanroom W8A8 integration into main ds4 (for antirez PR)

**Owner:** autonomous agent.  **Reference impl:** `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd`
(our fork). **Target:** a CLEAN main-ds4 checkout at `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-w8a8`.

## Goal
Re-implement, **cleanly**, the validated W8A8 (int8-weight × int8-activation) **prefill** matmul optimization on top
of antirez's main ds4 — not by copying our fork wholesale (it carries unrelated experimental divergences), but by
porting the *recipe* into antirez's existing structure. Two payoffs:
1. A **submittable PR for antirez** (clean, minimal, his code style, his quality + speed bars).
2. **Localize our fork's speed degradation**: bench `clean-main + W8A8` vs `our-fork(ds4-ssd) + W8A8`. We are at
   PARITY with antirez and suspect our fork lost some speed elsewhere. If clean-main+W8A8 is faster than
   ds4-ssd+W8A8, the gap is in our fork's *other* divergences — this isolates it.

## What W8A8 is (the validated win)
+15–20% end-to-end **prefill** on M5 Max (decode unchanged — it's n_tok=1 GEMV, int8 doesn't fire). Token-coherent
(generation identical). Brought our fork from ~−5% behind antirez to **parity** (he uses tuned fp16-NAX dense; we
match it with int8). Recipe:
- **Weight**: Q8_0 dense weight → **int8 [out×in] + per-row scale**, repacked **once at load** (cached). NOT
  dequant-per-dispatch (that overhead erases the win → measured 0.92× vs 1.30× with offline repack).
- **Activation**: f32 → **int8 + per-token scale** (dynamic, per dispatch).
- **Matmul**: `matmul2d` int8×int8→**int32**, **NK (K-tile) ≥ 128** (NK=32 starves int8: 0.55×), `multiply_accumulate`.
- **Rescale**: fused in the store — `C_f32 = C_i32 · a_scale[token] · w_scale[row]`. Store cooperative tensor to a
  threadgroup int32 tile then flat-loop rescale (NR0=32 so the 32×128 i32 tile = 16KB + weight 4KB fits the 32KB
  threadgroup; NR1=128). The cT→tg store stride MUST be `{1, NR0}` (col-major, matching dst) — row-major gave rel~1.5.
- **Applied to**: dense Q8_0 projections (MLA q_a/q_b/kv_a + shared-expert gate/up/down) and the **grouped attn_out
  O-proj**. **lm_head kept in half** (NO_I8) — it's the final logits, most quality-sensitive, ~free to leave fp.
- **Gates**: prefill only (`n_tok ≥ 16–32`), `in_dim%128==0`, `out_dim%32==0`. Decode/GEMV untouched.

## Reference implementation in ds4-ssd (study, don't blind-copy)

**Reference rule (read this first):** You ARE allowed to consult ds4-ssd freely when you hit a perf or
correctness issue — that's why it's here. "Cleanroom" does NOT mean "don't look." It means: **port the recipe,
not the plumbing.**
- **Copy/port:** the metal kernels (`metal/nax_fused.metal`), tuning constants (NK=128, NR1=128/NR0=32, the
  `{1,NR0}` col-major threadgroup store), the dispatch ladder, the hint policy, the microbench validation, and
  `W8A8_RESULTS.md` numbers as the target-to-match. Re-deriving these blind just reintroduces bugs we already
  paid for (rel~1.5 garbage from row-major store; 0.55× from NK=32).
- **Do NOT lift wholesale:** ds4-ssd's `ds4_metal.m` host glue, env-gate scaffolding, or any fork-specific
  code. Re-integrate cleanly into antirez's main structure. That fork plumbing is the exact surface the
  `clean-main+W8A8` vs `ds4-ssd+W8A8` comparison is meant to expose — importing it defeats the
  localize-degradation goal AND produces a PR antirez won't accept.

Commits (on branch `codex/integrate-ds4-agent`, pushed to `anemll` fork):
- `f07d649` int8 dense W8A8 (+15–20%) — 3 kernels + host wiring + env gate `DS4_GPU_DENSE_I8`.
- `2c4230c` dispatch hints `ds4_mm_hint {AUTO,PREFER_I8,NO_I8}` — lm_head→NO_I8, projections→PREFER_I8.
- `94ac135` grouped attn_out W8A8 kernel (validated +1.46× vs fp16-NAX).
- `8ff7586` wire attn_out W8A8 into the graph.
- (`3702ce8` fusion profiler — optional, not part of W8A8.)

Files / symbols:
- `metal/nax_fused.metal` (a SEPARATE `MTLLanguageVersion4_0` library read at runtime): kernels
  `ds4_repack_q8_to_i8_rowscale`, `ds4_quant_act_pertoken_i8`, `ds4_dense_i8_fused`, `ds4_attn_out_low_i8_fused`.
  NOTE: antirez's main ALREADY compiles `matmul2d` (his `metal/dense.metal` `kernel_mul_mm_mpp_direct_rhs` under
  `#ifdef DS4_METAL_HAS_TENSOR`) — prefer integrating the W8A8 kernels into HIS metal setup rather than importing our
  separate-library mechanism, for a cleaner PR.
- `ds4_metal.m`: pipelines + `ds4_gpu_dense_i8_enabled()` + the int8 branch in `ds4_gpu_matmul_q8_0_tensor_ex`
  (weight-repack cache keyed by `weight_offset`, growing act/scale scratch, 3-encoder repack→quant→matmul) + the
  attn_out int8 branch in `ds4_gpu_attention_output_q8_batch_tensor` + the `ds4_mm_hint` plumbing.
- `ds4_gpu.h`: `ds4_mm_hint` enum + `ds4_gpu_matmul_q8_0_tensor_ex(...)`.
- `ds4.c`: hint annotations at call sites (lm_head `g->logits` → NO_I8; `batch_qr/kv_raw/q`, `batch_shared_*` →
  PREFER_I8).
- **Standalone microbench harnesses (no 86GB model, ~1s/run)** — reuse to validate kernels in isolation FIRST:
  `moe-batch-bench/nax_dense_i8_probe.m` (dense W8A8 vs fp-relaxed, rel + GF/s),
  `moe-batch-bench/nax_attnout_i8_test.m` (grouped attn_out W8A8, rel + GF/s).

## Steps
1. **Setup**: `ds4-w8a8` is ALREADY cloned — confirmed clean antirez main (branch `main` @ `f91c12b`,
   `origin = https://github.com/antirez/ds4`, NO ANE, has `DS4_METAL_HAS_TENSOR`/matmul2d in `metal/dense.metal`).
   PR-ready against upstream. Just `cd ds4-w8a8 && git checkout -b anemll-NAX-w8a8`. Build (`make`); confirm it runs
   the resident model `/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
   (no sidecar). **Run its ds4-bench FROM `ds4-w8a8/`** (reads `metal/*.metal` by relative path — running from
   elsewhere loads the wrong shaders and aborts).
2. **Baseline**: record clean-main prefill (ds4-bench, 8k–64k step 8k) + quality (his eval — see below) BEFORE changes.
3. **Kernel 1 — dense W8A8**: port the repack + per-token act-quant + fused int8 matmul into his metal + host. Validate
   in isolation with `nax_dense_i8_probe.m` (rel <1%, GF/s ≥ his fp16-NAX). Wire (env-gated), then:
   - **Quality test** (must be neutral): run his capability/eval + a coherence gen; compare to baseline.
   - **Speed A/B**: ds4-bench prefill, clean-main vs +dense-W8A8, adjacent pairs, 8k–32k. Expect +15–20%.
4. **Kernel 2 — attn_out grouped W8A8** (`nax_attnout_i8_test.m` to validate rel + GF/s; +1.46× vs fp16-NAX). Wire,
   quality + speed check.
5. **lm_head → keep half** (do NOT int8 the output projection).
6. **Iterate** each kernel until quality-neutral AND speed-positive. Then a full 8k–64k A/B (prefill + **gen_tps**)
   clean-main vs clean-main+W8A8, and clean-main+W8A8 vs ds4-ssd+W8A8 (the localize-degradation comparison).
7. **PR**: minimal diff, antirez's style, env-gated default-off (or propose default-on with the quality data),
   commit messages explaining the recipe. Co-author trailer per repo convention.

## Quality + speed bars (per kernel, iterate until met)
- **Quality**: antirez's eval (README "Capability Evaluation" → `ds4-eval`) and/or a logit/perplexity check must be
  within noise of baseline; generation coherent (greedy temp 0, compare tokens). int8 rel error in microbench <1%
  (dense) / <2% (attn_out). If a kernel degrades quality, gate it half (like lm_head) or drop it.
- **Speed**: ds4-bench prefill A/B (same tool, same flags, **adjacent thermal pairs**, ~2% noise floor — replicate).
  Each kernel must be net-positive end-to-end, not just in microbench (dilution: a kernel that's a small prefill
  slice won't move e2e — dense projections + shared expert are the big slices; attn_out is ~+1%).

## Correctness workflow when you see drift (DO NOT tune a min-token gate to hide it)
If W8A8 output drifts from the non-NAX baseline, **localize the cause and fix it at the source**. Cranking
`DS4_GPU_DENSE_I8_MIN_TOK` / `min_tokens` up until the symptom disappears on one test prompt is a band-aid: it
leaves the bug latent (a longer non-aligned prompt still hits it), throws away the W8A8 win on the gated range
(largest at small–mid n_tok), and hides the real divergence. The proven ds4-ssd recipe is token-identical at a
**16/32** cutoff — so any drift here is in YOUR integration, not the recipe.

1. **Reproduce deterministically.** Fixed prompt, temp 0 / greedy, fixed seed. Record the exact token index where
   output diverges and the top-2 logits there (near-tie vs gross-garbage tells you the failure class).
2. **Localize — which matmul, which shape.** Use per-call-site hints, NOT the global gate: enable W8A8 one call
   site at a time (dense vs attn_out vs shared/proj), then bisect on n_tok. Determine whether drift correlates with
   `n_tok % 128 != 0` (partial tail tile) or appears even on aligned n_tok.
3. **Classify and fix:**
   - **(a) Partial-tile bug** — drift only when `n_tok % 128 != 0`. matmul2d's B-slice `tB.slice(lk, r1)` runs past
     N on the last tile; do not trust in-kernel `tok < N` write-guards + matmul2d OOB-read behavior alone.
     **Fix = align-split:** dispatch W8A8 over `floor(n_tok/128)*128` tokens only, route the `n_tok % 128`
     remainder through the existing non-NAX path (`mul_mv_ext`/simdgroup). Tail ≤127 tokens → negligible perf,
     guaranteed correctness, no magic gate.
   - **(b) Quant precision drift** — small uniform rel error (~0.7%) flipping near-tie greedy tokens even on aligned
     n_tok. Expected int8 error; a min-token gate is the wrong lever. Fix via precision policy (keep sensitive
     matmuls higher precision — lm_head already `NO_I8`) and validate with **perplexity parity**, not exact tokens.
4. **Numeric diff, not just token diff.** At the suspect layer dump the W8A8 output and the non-NAX reference for the
   SAME inputs; compute rel error. Target = ds4-ssd microbench (dense 0.7%, attn_out 0.6%). Much higher only at
   partial tiles → class (a).
5. **Validate on an alignment matrix.** Prompt lengths spanning aligned + deliberately non-aligned chunks: 127, 129,
   255, 2017, 4095 tokens, each vs non-NAX baseline. Require token-identical (after a class-(a) fix) across all.
6. **Remove the band-aid.** Once fixed, restore `min_tokens` to the proven 16/32 so the win is retained; re-run
   step 5 + the all-sizes bench.

**Shortcut:** the W8A8 metal kernels are byte-equivalent to ds4-ssd (token-identical at 16/32), so `diff` your
*integration* against ds4-ssd — activation-quant kernel, static scratch-buffer reuse/sizing, per-token scale
computation, and which call sites get routed. The divergence is there, not in the matmul.

## Gotchas (cost real time in ds4-ssd; pre-warn the agent)
- matmul2d operand pointers must be **non-const** ("Input types must match cooperative tensor types").
- threadgroup tile element type `int8_t`/`half`, not `char`. `multiply_accumulate` (not `multiply`) for manual K-loops.
- NK=32 starves int8 → use NK≥128. Offline/load-time weight repack is ESSENTIAL (online dequant erases the win).
- cT→threadgroup store stride `{1,NR0}` (col-major) to match dst; row-major → garbage.
- Check the **call-site n_tok**, not just weight shape: e.g. comp_kv/comp_sc run at n_tok=4 (tail window) → NOT
  W8A8-eligible; the F16 projections (indexer/router/hc) are small-M → not worth it. Only the large-n_tok, large-M
  Q8_0 matmuls (dense projections, shared expert, attn_out) benefit.
- The matmul2d library compile is a one-time ~30–60s cost (pre-warm for startup timing).

## Existing gates to audit — DO NOT contaminate the A/B or the fusion (CRITICAL)
antirez's main has gates/flags that, if mishandled, silently corrupt the W8A8 result:
- **Baseline must be his SHIPPED default**, which uses his **fp16-NAX dense** (`kernel_mul_mm_f16_f32_mpp_direct_rhs`
  _n128/_n64) + `kernel_attn_out_low_q8_0_mpp_direct_rhs`, gated by his `tensor_matmul` flag. The honest A/B is
  **W8A8 vs his fp16-NAX**, NOT vs his simdgroup fallback. If you disable tensor_matmul in the baseline, W8A8 will
  look hugely (and falsely) faster. Confirm tensor_matmul is ON in BOTH arms (it's the default).
- **Hold ALL his flags constant across A and B**: drift-patch `hc_stable / norm_unify / kv_raw_f32 /
  rope_exp2_log2 / math_safe / tensor_matmul` (logged together in ds4_metal.m ~3028), `prefill_chunk`, and any
  `DS4_METAL_*` env. Toggle ONLY the new W8A8 gate between arms. Never change two things at once.
- **Respect his per-batch tile gating** (_n128 vs _n64 selected by batch size). Size-gate W8A8 so it only fires
  where it wins (large n_tok); don't let it pre-empt his tuned path in regimes where his is faster.
- **Fusion must not regress**: his main HAS `g_batch_cb` + shared `g_batch_enc` (ds4_metal.m ~38, 241, 250).
  Wire the W8A8 sub-ops (repack one-time, then per-call act-quant + matmul) via `ds4_gpu_compute_encoder(cb)` so
  they reuse the batch encoder — NO raw encoders, NO commit/flush/`close_batch_encoder` between act-quant and
  matmul (that injects an encoder switch / sync the baseline doesn't have, contaminating both speed and the
  comparison). The load-time repack writes a buffer the matmul reads in the SAME cb — encoder ordering guarantees
  visibility, no flush needed. **Verify**: per-token fusion-boundary count (owned-commit/wait/flush/enc_end) with
  W8A8 on must equal baseline (port ds4-ssd's `DS4_METAL_FUSION_PROFILE` instrumentation, or count manually). If
  W8A8 adds any owned-commit/wait/flush per token, it's a wiring bug, not a kernel cost.
- **Weight-repack cache**: key by weight_offset so the Q8_0→int8 repack runs ONCE/weight at first use, not per
  dispatch (per-dispatch repack erases the win AND adds an encoder every call).

## Per-caller inventory & selective fallback (after implementation — REQUIRED)
A W8A8 kernel is only worth routing where it's a CLEAR per-call-site win. After each kernel works, inventory
**every caller** and gate per-site; where W8A8 is not a clear win, **fall back to antirez's existing fp16-NAX path**
for that caller (the `ds4_mm_hint`-style per-site policy — AUTO/PREFER_I8/NO_I8 — is the mechanism).
1. **Enumerate callers** of the dense matmul + attn_out in antirez's main (grep his `ds4_gpu_matmul_*` /
   `attention_output` calls): functional part (MLA q_a/q_b/kv_a, shared gate/up/down, attn_out O-proj, lm_head,
   and the F16 projections comp_kv/comp_sc/indexer/router/hc), with each site's **(K, M) shape AND typical
   call-site n_tok** (check the ACTUAL n_tok passed, not just weight shape — e.g. comp_kv runs at n_tok=4).
2. **Per-site A/B**: W8A8 vs his fp16-NAX at each site's real shape+n_tok (microbench + the end-to-end gate).
   A site is "clear win" only if W8A8 beats his path there end-to-end beyond the ~2% noise floor AND quality holds.
3. **Route accordingly**: PREFER_I8 only the clear-win sites (expected: large-M, large-n_tok prefill — dense MLA
   projections, shared expert, attn_out). NO_I8 / leave-on-his-NAX the rest: lm_head (quality), and any small-M
   (router/indexer/hc → ~64-256 out) or small-n_tok (comp_kv/comp_sc → n_tok=4 tail window) site where W8A8
   can't fill its tile / dilutes below noise. Don't blanket-enable.
4. Document the per-site decision table (site → W8A8 or fp16-NAX, with the measured Δ) in the PR.

## Acceptance
- `anemll-NAX-w8a8` branch on a clean main-ds4: dense + attn_out W8A8 (lm_head half), env-gated, builds clean.
- Each kernel: quality-neutral + speed-positive, validated.
- Full A/B recorded: clean-main vs +W8A8 (prefill +15–20%, gen unchanged, coherent); and clean-main+W8A8 vs
  ds4-ssd+W8A8 to localize any fork regression.
- PR description with the recipe + measured deltas + quality evidence.
- **Per-caller decision table**: every matmul call site → routed W8A8 or fp16-NAX-fallback, with its measured
  per-site Δ; W8A8 enabled ONLY where it's a clear end-to-end win, fp16-NAX retained everywhere else.
- **Final delivery — live prefill t/s readout** (the on-screen format, "as in our version"): run a long-context
  pass (e.g. `-c 100000`) and capture ds4/ds4-agent's live prefill progress line, baseline vs +W8A8, e.g.
  `ctx 51.9k/100k | prefill [▶▶▶▶▶▶▶▶················] 364/1305 27.9% 253.4 t/s`. Show both so the on-screen prefill
  t/s is directly comparable to our version's reference (`W8A8_RESULTS.md`: ours ~359/341/321/307/296/283/271/260
  t/s at 8k→64k). If clean-main+W8A8 beats our t/s (esp. at high ctx), that localizes a regression in our fork.

Reference details + measured numbers + the full investigation are in ds4-ssd
`memory/nax-kernel-tuning-playbook.md` and `moe-batch-bench/M5M_ANE_OPTIMIZATION_REPORT.md`.
