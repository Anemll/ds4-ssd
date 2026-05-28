# Handoff: ANE+GPU concurrent in *resident* MoE prefill (target > 532 t/s)

**Repo:** `Anemll/ds4-ssd`, branch `wide-tile-moe`. DeepSeek-V4-Flash IQ2XXS, full GGUF (~82 GB), M5 Max (137 GB RAM). Resident mode (`--moe-mode off`).

## Goal
Bring **GPU + ANE concurrent** routed-MoE prefill into the **resident** code path so the resident prefill exceeds the current single-engine ceiling of **~532 t/s** (NAX-int8 @ ctx 16K, per-expert ~512). The flash hybrid already measures **1.74×–1.86×** from ANE∥GPU concurrency (see ds4_metal.m:17345); the only blocker to landing that win in resident is wiring/infrastructure, not feasibility.

## What we already know (don't re-derive)
- **Single-engine NAX ceiling is real.** Resident NAX-int8 plateaus at ~520–532 across per-expert batches 512→768 (chunks up to 24K single-chunk; ≥32K single-chunk OOMs). Bigger chunks do not push past ~530.
- **GPU compute is serial *across kernels* on Apple M5 (scheduler limit, not hardware).** `moe-batch-bench/concurrency_probe.m` ran `matmul2d` and `simdgroup_multiply_accumulate` in two separate command queues, balanced durations (~2.2 s each), at full grid AND low occupancy (16 threadgroups). Speedup **1.000 / 1.001** in both. An independent expert review then tested same-command-buffer + two-encoders (still 1.00×) AND a **mixed kernel** (matmul2d + ALU in the SAME shader): **1.46×** — proof the hardware *does* support NAX+ALU concurrency, but only inside one kernel via fused matmul2d + immediate ALU post-processing on cooperative tensors. **Implication:** cross-kernel splitting (NAX kernel || Metal-MM kernel) is a no-go on M5's scheduler; the in-kernel fused path is real and adds **Path C** below. Cross-processor GPU + ANE remains the other real concurrency.
- **Flash hybrid path is RAM-bound on this model.** Slot-bank/sidecar (`dsv4-iq2xxs-expert-major`) expert set is ~160 GB > 137 GB RAM → 0% hit-rate, SSD-streaming. Fully-cached flash isn't achievable on this model, so the win has to come via **resident**.
- **Validation harness exists.** `DS4_METAL_GRAPH_DUMP_{PREFIX,NAME,LAYER,POS}` dumps a named tensor as raw f32. The team uses **layer-0 `ffn_moe_out`** dumped per-pos and compared (max-abs-diff) against a baseline that has *all* `DS4_RESIDENT_MOE_MPP_*` envs unset (the MPP-vs-MPP gotcha — see [the existing handoff in this session]). NAX-half was validated this way: layer-0 `ffn_moe_out` max-abs **5.4e-05** vs Metal baseline.

## The technical problem (where I stopped)
The flash hybrid's ANE submit (`ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor`, ds4_metal.m:18281) requires:
- `gate_bank`, `up_bank`, `down_bank` — per-expert tensor views of the quantized weights (IQ2_XXS gate, Q2_K down; dims **must** be 4096/2048/4096). **In resident this is easy** — `ds4_gpu_tensor_view` of the model-mapped expert weights at `layer->ffn_*_exps->abs_offset + expert*expert_bytes`.
- `weights` — per-expert **router weights for refs tokens** (`refs` floats, contiguous).
- `x` — per-expert **gathered activations** (`refs × DS4_N_EMBD` floats, contiguous).
- `n_tokens = refs`.

The flash path builds the per-expert CSR token+weight buffers in its dedup pre-pass (`g->flash_dedup_token_list`, `g->flash_dedup_weight_list`); the views passed to ANE are `tensor_view(flash_dedup_*_list, begin*sizeof(...), refs*sizeof(...))`. **The resident path lacks these** — its `hids` (`g_moe_id_map_buffer` at `hids_off`) holds **pair indices** (token×n_experts + expert_slot), not raw token indices, and the router weights are pair-indexed too.

Consumption pattern (mirror `DS4_FINISH_ANE_SLOT` macro at ds4.c:13063):
```c
ds4_gpu_flush_commands();   // flush encoded GPU gather so ANE thread sees valid x
ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(job, g->flash_prefill_out, &mid_is_f16);
ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out, g->flash_prefill_out,
                                    tokens_view, refs, DS4_N_EMBD);
```
`finish` writes `refs × DS4_N_EMBD` to a per-expert output scratch; `scatter_add` accumulates into `batch_routed_out` at token positions.

## Two paths to the wiring
Both need real new code. Pick one.

### Path A — Compact bridge (`ds4_gpu_routed_moe_batch_tensor`, ds4_metal.m ~22192+)
Most upside (chains on top of the ~509 warm baseline), but **needs new CSR infrastructure**.

What to build:
1. **Per-expert token-index + router-weight CSR buffers** populated each chunk: walk `original_counts[]`, scatter `(token, weight)` pairs into per-expert contiguous regions. Mirrors how the dedup function (ds4.c:11337+) builds `ref_tokens[]`/`ref_weights[]` on the host then writes them to `batch_router_selected`/`batch_router_weights` — but in compact bridge those host arrays don't exist.
2. **ANE branch** at the top of the per-expert loop (ds4_metal.m ~23211, before `if (use_h_h)`): for `use_ane_hybrid && refs >= ane_min_refs`, create views into the per-expert CSR buffers, gather x to a per-expert offset of a dedicated scratch (or rotate scratches), call `_ane_start_tensor`, collect the job + `tokens_view` + `refs` in a jobs array, `continue` (skip GPU gate/up).
3. **Post-loop drain** before the swiglu: for each collected job, `flush_commands` → `finish_tensor` → `scatter_add_rows_f32_tensor` → mark `ane_mask[expert]=1`. Then free retained `tokens_view`s.
4. **Down loop skip** (ds4_metal.m ~23351): early-`continue` when `ane_mask[expert]`. Swiglu doesn't need changes — ANE-handled experts' pair rows in `gate`/`up`/`mid` are simply not consumed downstream.
5. **Scratch sizing**: `ane_x_scratch` = `n_tokens × DS4_N_EMBD × 4` (≥ max refs × in_dim). `ane_out_scratch` similar. Multiple in-flight ANE jobs need disjoint x_scratch regions (per-job offsets via CSR `offsets[]`), OR enforce one-active-job-at-a-time (simpler, loses some concurrency).

### Path B — Dedup function (`metal_graph_resident_moe_run_mpp_prefill_dedup`, ds4.c:11337)
Less upside (today capped ~400 t/s by the per-expert command buffer floor), but **CSR already exists** (`ref_tokens[]`/`ref_weights[]` host arrays at 11464+; written to `batch_router_selected`/`batch_router_weights` per expert).

What to build:
1. **Replace** `flash_moe_run_mpp_int8_safe_tensor` (ds4.c:11579) with a branch: if `use_ane_hybrid && refs >= ane_min_refs` → `_ane_start_tensor` with the same args; collect job in `jobs[ui]`; skip the GPU matmul fallback for this expert.
2. **Fix the scratch-reuse race**: today every expert reuses scratch at offset 0 (the per-expert command buffer flush serializes correctness). For ANE concurrent we need either (a) per-expert CSR offsets so multiple ANE jobs read disjoint regions, or (b) enforce a single active ANE job at a time (wait+finish previous before next submit), reusing one scratch — simpler, less concurrent.
3. **`flush_commands` before each `start_tensor`** so the ANE thread sees the just-gathered `x_tmp` (else it reads pre-gather data).
4. **Post-loop drain** at the function end (ds4.c ~11655): wait/finish/scatter_add each collected job in order; free retained `tokens_view`s.
5. **Note**: the dedup path's existing **per-expert begin/end_commands** is already O(n_unique) submissions — the submission floor caps it at ~400 t/s today. Even with ANE, that floor likely dominates unless you also coalesce the per-expert command buffers (see my earlier tail-end analysis: lay scratch at `offsets[]` and emit many experts in one CB).

### Path C — Fused matmul2d + ALU kernel (in-kernel NAX∥ALU concurrency, ~1.46× on the matmul stage)
**New option from the M5 expert review.** Cross-kernel matmul2d∥simdgroup is serialized by the scheduler, but a SINGLE kernel that interleaves `matmul2d` with ALU post-processing (swiglu / scaling / scatter prep) on cooperative tensors hits **1.46× hardware overlap** between NAX (tensor units) and ALU pipelines.

What to build:
1. A fused **matmul2d (gate) + matmul2d (up) + swiglu + matmul2d (down) + scaled-output** kernel that keeps the intermediate tile in cooperative tensors across the matmul stages, applying swiglu/clamp on-cooperative between matmuls. Replaces the existing 3-kernel pattern (matmul → swiglu → matmul) for the routed-MoE per-expert path.
2. Wire it into the compact bridge as a parallel pipeline alternative (e.g., `DS4_RESIDENT_MOE_NAX_FUSED_MM_SWIGLU=1`).
3. Validate via the layer-0 `ffn_moe_out` harness.

This is orthogonal to ANE (combinable with it) and doesn't require any new CSR plumbing. Smaller surface than Path A but requires writing a non-trivial fused tensor kernel.

## Recommendation
**Best long-term combo: Path A + Path C** (ANE in resident + fused matmul/ALU kernel) — additive: 1.86× from ANE+GPU concurrency × 1.46× from in-kernel fusion ≈ ~2.7× theoretical, capped by other overheads. Start with whichever is more tractable to your team.
- **Path A alone**: clean ~1.86× upside, well-defined infrastructure work (CSR plumbing).
- **Path C alone**: clean ~1.46× upside, smaller surface, requires fused-kernel authoring expertise.
- **Path B**: probably skip — dedup function's per-expert submission floor (~256 cb/layer) caps it at ~400 regardless of which engine you use.

## API surface (already in tree, callable as-is)
- `ds4_gpu_ane_prefill_job *ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(...)` — ds4_metal.m:18281. Returns NULL on bail; runs on its own pthread; **only** accepts gate_type=`DS4_METAL_TENSOR_IQ2_XXS`, down_type=`DS4_METAL_TENSOR_Q2_K`, in/mid/out = 4096/2048/4096.
- `int ds4_gpu_routed_moe_expert_banked_batch_ane_wait_predict_tensor(job)` — ds4_metal.m:18910. Joins ANE pthread.
- `int ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(job, out, &mid_is_f16)` — ds4_metal.m:18941. Writes `refs × DS4_N_EMBD` to `out` and frees job.
- ANE input CoreML model is **B=256, w_qscale=64, x_qscale=16** (banner: `ANE int8w precompiled B=256 H=4096 I=2048 w_qscale=64 x_qscale=16 mid_qscale=16`). The submit fn handles batching internally — pass real `refs`.
- ANE CoreML model is **built from MIL at runtime** (no external `.mlpackage` dependency from `coreml_exports/` for the in-engine path).

## Reference: flash hybrid call sites
- Per-iteration submit: ds4.c:13516 (`active_ane_job = ...start_tensor(gate_b, up_b, down_b, ..., weights_for_refs, g->flash_prefill_x, refs)`).
- View construction: ds4.c:13265–13272 (`token_view`/`weight_view` from `g->flash_dedup_token_list`/`flash_dedup_weight_list` at `begin*sizeof(...)` offsets).
- Drain: ds4.c:13063 `DS4_FINISH_ANE_SLOT` macro — flush_commands → finish → scatter_add_rows_f32 → re-begin_commands.
- Concurrency state machine (pipelined variant, optional once basic works): `DS4_WAIT_ACTIVE_ANE_PREDICT` (ds4.c:13153) + `DS4_QUEUE_READY_ANE` (ds4.c:13116). Start with the simpler "concurrent" pattern (single active job at a time, GPU runs in parallel).

## Validation harness (mandatory before trusting any perf number)
Earlier in this thread we learned the hard way that `ds4-bench --dump-frontier-logits` does NOT reflect the prefill MoE backend (proven via int8 qscale=4 → identical logits). **Use layer-0 `ffn_moe_out` tensor dumps** instead:
- Baseline: **all `DS4_RESIDENT_MOE_MPP_*` envs unset** (true `mul_mm_id`). With even `MIN_TOKENS=64` set, MPP still runs and you get MPP-vs-MPP = 0.0 (false pass).
- Candidate: same prompt + ctx with `DS4_RESIDENT_MOE_ANE_HYBRID=1`.
- Diff: `max_abs` should be small (precision delta, ~e-04 territory like NAX-half's 5.4e-05; *not* a real `int8`-quant ~22.8 delta).
- Sanity canary: `DS4_FLASH_MOE_MPP_INT8_QSCALE=4` vs default — must differ by ~40 in `max_abs`. If it doesn't, the dump isn't seeing the prefill path; reconfigure.

## What's already in the tree (do not redo)
- **iter-4 scaffold:** `DS4_RESIDENT_MOE_ANE_HYBRID` flag + `ane_min_refs` (default 129) + active-job/mask state vars declared in compact bridge (ds4_metal.m ~23146). Builds clean. Path A wiring goes here.
- **Results table:** `moe-batch-bench/RESIDENT_MOE_PREFILL_SWEEP.md` (Metal / NAX-int8 / NAX-half / dedup × 2K–256K, with chunk column).
- **Concurrency probe:** `moe-batch-bench/concurrency_probe.m` + Metal compute-serial finding recorded in memory `m5-gpu-compute-serial-no-metal-nax-overlap.md`.
- **Wide-tile fix** (commit `e2f51ba`, validated byte-identical).
- **Resident NAX-half wiring** (compact bridge, validated 5.4e-05). Confirms the pattern of wiring new paths into the compact bridge works.

## Open design choices for the expert to make
1. **Path A vs B** (see recommendation: A).
2. **Per-expert CSR construction**: GPU kernel that walks `hids` and decodes pair→token, or host-side build + upload?
3. **In-flight ANE concurrency**: single-active-job (simpler, ~partial speedup) vs flash's pipelined predict/queue (full ~1.86× but more code).
4. **Hybrid-routing threshold**: copy flash's `hybrid_ane_min_refs` (default 129) or tune for resident? At chunk 16K, refs ≈ 512 per expert — almost every expert qualifies, which is what we want.
5. **Memory budget**: per-expert ANE scratches at worst-case `n_tokens × DS4_N_EMBD × 4` = ~256 MB at 16K chunk. Acceptable in 137 GB.

## Success criterion
Resident prefill at ctx 16K with `DS4_RESIDENT_MOE_ANE_HYBRID=1` **exceeds 532 t/s** (current best non-dedup ceiling), with layer-0 `ffn_moe_out` `max_abs` ≤ ~e-04 vs Metal baseline (correctness gate). Realistic target from flash measurement: **800–990 t/s** (1.5–1.86× of 532).

## Where I stopped and why
I kept underestimating this build as "wire one call" and ran four iterations of half-edits. Two reassessments later it's clearly a multi-day proper build needing CSR infrastructure, async-safe scratch, and the validation harness. The right move was to write this handoff rather than churn another half-edit.
