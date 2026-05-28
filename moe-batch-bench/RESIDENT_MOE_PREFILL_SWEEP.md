# Resident routed-MoE prefill sweep — Metal vs NAX-int8 vs NAX-half (+ dedup)

**Hardware:** M5 Max, 137 GB. **Model:** DeepSeek-V4-Flash IQ2XXS (full GGUF, resident — all experts in RAM). **Tool:** `ds4-bench --moe-mode off`, `--warm-weights`, greedy, 1 gen token. Numbers are **prefill throughput (tokens/s)**.

## Legend
- **ctx** — prompt length prefilled.
- **prefill chunk (= MoE batch)** — `DS4_METAL_PREFILL_CHUNK`. The routed-MoE batch equals the chunk, and per-expert batch ≈ `chunk × topk / n_experts` (topk=8, 256 experts → chunk/32). This is what NAX amortizes over.
  - ctx ≤ 16K: **single-chunk** (chunk = ctx) → per-expert batch grows with ctx (the amortization curve).
  - ctx ≥ 32K: **chunk fixed at 16384** (single-chunk attention OOMs past ~16–32K) → MoE batch pinned at 16K; deeper ctx only adds attention cost. 16K is the bridge point (single-chunk 16K == chunk-16K).
- **Metal** — `kernel_mul_mm_id` (GPU simdgroup MMA, id-map grouped). This is upstream antirez's routed-MoE prefill; no NAX. Default tile n32 (wide n64/n128 are correct but default-off — occupancy-bound).
- **NAX-int8** — `matmul2d` int8 (M5 tensor API), compact-bridge per-expert gather. `DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1 …_FORCE=1 …_MIN_TOKENS=64 …_COMPACT_BRIDGE=1`.
- **NAX-half** — `matmul2d` half×half (`h_h_f`), same compact bridge. `+ DS4_RESIDENT_MOE_NAX_HALF=1` (tile 128). Validated numerically vs Metal (layer-0 ffn_moe_out max-abs 5.4e-05).
- **NAX-int8 +dedup** — `metal_graph_resident_moe_run_mpp_prefill_dedup` (`DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=1`). NAX-int8 (not ANE); per-unique-expert loop with **one command buffer per expert**.

## Results (prefill t/s)

| ctx | chunk (MoE batch) | Metal | NAX-int8 | NAX-half | NAX-int8 +dedup |
|---|---:|---:|---:|---:|---:|
| 2K   | 2048  | **378** | 178 | 133 | 325 |
| 4K   | 4096  | **447** | 361 | 303 | 376 |
| 6K   | 6144  | **448** | 420 | 370 | 406 |
| 8K   | 8192  | 408 | **490** | 465 | 395 |
| 16K  | 16384 | 413 | **532** | 510 | 397 |
| 32K  | 16384 | 401 | **516** | 495 | 381 |
| 64K  | 16384 | 359 | **446** | 437 | 348 |
| 128K | 16384 | 291 | **361** | 360 | 287 |
| 256K | 16384 | 210 | **248** | 246 | — |

(Bold = best non-dedup at that ctx.)

## Findings
- **Crossover Metal → NAX at ~6–8K.** Below it Metal wins (its flat ~448 beats NAX's small-batch overhead); from 8K on, NAX-int8 leads by **+18–25%**, peaking 532 @16K.
- **Metal is flat** (~448 peak, no batch amortization) and decays with attention depth (210 @256K). **NAX climbs** with per-expert batch then decays in parallel, holding its relative lead.
- **NAX-half tracks NAX-int8** within ~15% at small batch, **converging to ~equal by 64K+** (attention dominates → MoE-backend gap washes out). int8 is the better default; half stays opt-in.
- **Dedup is capped ~375–406 t/s** across all ctx — a **per-expert command-buffer submission floor** (~256 submissions/layer that don't shrink with batch). It only beats *plain NAX-int8* at 2K (325 vs 178, where plain NAX is starved) but **never beats the per-ctx champion** (Metal 378 @2K; NAX-int8 532 @16K). Headroom exists if the per-expert submissions are batched.

## Caveats
- NAX/compute story is for the **resident (in-RAM)** regime. The **flash/slot-bank (SSD-paged)** path is I/O-bound — same compute looked like 139 t/s when streaming — so these numbers don't transfer there unless fully cached.
- ctx ≥ 32K rows hold MoE batch at 16K (chunk cap); they measure attention-depth decay, not bigger MoE batches.

## "Can we exceed 532?" — investigated, no
Loop goal: beat the per-ctx best non-dedup. Every lever exhausted:
- **Bigger chunks**: NAX plateaus (16K=532, 20K=520, 24K=522 per-expert 512→768); single-chunk OOMs > 24K. No.
- **NAX-int8 +dedup**: submission-floor capped ~375–406 (one command buffer per expert, ~256/layer); beats plain NAX only at 2K (325 vs 178) but never the per-ctx champion. No.
- **Metal-MM ∥ NAX concurrent**: micro-bench (`concurrency_probe.m`, two queues) → speedup 1.00 at full grid AND 1.001 at 16-threadgroup low occupancy → **Apple GPU runs compute serially; matmul2d & simdgroup-MMA don't overlap.** No.
- **ANE ∥ NAX concurrent**: genuinely separate processors (flash `HYBRID_PREFILL` does it), but flash-only and the sidecar expert set (~160 GB) exceeds 137 GB RAM → SSD-bound (249 t/s). No (can't be made compute-bound on this model).

**Conclusion: ~532 t/s (resident NAX-int8 @16K) is the M5 prefill ceiling for this model.** The only untested upside is wiring ANE into the *resident* (in-RAM) path so GPU+ANE overlap without SSD — a substantial build, not attempted.
