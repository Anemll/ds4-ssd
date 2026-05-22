# Measuring ANE Wall for Cross-Generation Comparison (M3U / M4 / M5)

## What is "ANE wall"

ANE wall is the **aggregate time the ANE evaluator spent inside `predict`
calls** during one prefill pass, attributed to whichever physical cluster the
OS dispatched each call onto. It excludes CPU-side dequant, IOSurface copies,
GPU encode/blit, and post-processing — those are reported as separate timers.

For an N-worker dispatch (`DS4_FLASH_MOE_ANE_THREADS=N`, one pthread per
worker, stride-N over chunks) the prefill code records four per-worker
totals: `ane_eval_a`, `ane_eval_b`, `ane_eval_c`, `ane_eval_d`. ANE wall is
the maximum of those four:

```
ane_wall_est = max(ane_eval_a, ane_eval_b, ane_eval_c, ane_eval_d)
```

This is the right comparison metric across machines because it isolates ANE
compute from the rest of the pipeline (GPU encode, weight pread, etc.).

## How to measure it

1. **Build and arrange a fixed prompt + model.** The shipped m3u profile uses
   DSv4 IQ2_XXS dense and an 8423-token coding prompt — keep both constant
   across machines or the numbers are not comparable.
   ```bash
   make ds4
   export DS4_MODEL=/path/to/dsv4-iq2xxs-expert-major/dense/model-dense.gguf
   export DS4_SIDECAR=/path/to/dsv4-iq2xxs-expert-major
   export DS4_PROMPT_FILE=/path/to/coding_8k.txt
   ```

2. **Pin per-call batch so the comparison is apples-to-apples.** Without
   this, the multi-worker setup auto-shrinks per-call batch to balance the
   N workers — which itself reduces per-call wall and skews the comparison.
   ```bash
   export DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH=1
   ```

3. **Run the profile script with the THREADS sweep.**
   ```bash
   for n in 1 2 3 4; do
     DS4_RUN_NAME=ane_wall_threads${n}_$(date +%H%M%S) \
       DS4_FLASH_MOE_ANE_THREADS=$n \
       DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH=1 \
       ./run_ane_prefill_profile_m3u.sh
   done
   ```

4. **Extract `ane_wall_est` from each log.** The relevant line:
   ```
   ds4: ANE prefill timing dequant=... ane_eval_a=A ane_eval_b=B ane_eval_c=C ane_eval_d=D ane_wall_est=W ...
   ```
   For N=1, A is the only non-zero value and equals W (single-cluster wall).
   Pull it with:
   ```bash
   grep -E "ANE prefill timing dequant" moe-batch-bench/profile_runs/ane_wall_threads*.log \
     | sed -E 's/.*ane_eval_a=([0-9.]+).*ane_eval_b=([0-9.]+).*ane_eval_c=([0-9.]+).*ane_eval_d=([0-9.]+).*ane_wall_est=([0-9.]+).*/A=\1 B=\2 C=\3 D=\4 wall=\5/'
   ```

5. **Repeat each THREADS at least twice.** ANE wall has run-to-run jitter
   on the order of 5-10% from thermal and scheduling noise. Report the
   mean of 2-3 runs and the spread.

6. **Record full call count too.** `ane_calls=N` from the same line.  With
   `THREADS_FIXED_BATCH=1` the count should be near-constant across N — if
   it isn't, the comparison is contaminated.

## What to compare across M3U / M4 / M5

| Metric | Where to find it | Why it matters |
|---|---|---|
| **Single-cluster ANE wall (N=1)** | `ane_wall_est` at THREADS=1 | Raw ANE compute throughput; scales with per-cluster matmul rate and weight bandwidth |
| **Two-cluster ANE wall (N=2)** | `ane_wall_est` at THREADS=2 | Per-chip cluster count + their independence; M-series chips with one physical ANE will not scale past N=1 |
| **Wall speedup N=1 → N=2** | ratio of the two | Cluster parallelism efficiency; ceiling ≈ physical cluster count |
| **Per-call avg (`ane_call_avg`)** | same log line | Constant-time component (write_surface, IOSurface handoff) — important for chips with smaller per-call overhead |
| **Pad utilization** | `ANE prefill chunks pad_util=...%` | Should match across machines if FIXED_BATCH is set; sanity check |

## M3 Ultra reference numbers

Captured 2026-05-21, DSv4 IQ2_XXS, 8423-token coding prompt, FIXED_BATCH=1:

| THREADS | prefill t/s | ANE wall | per-worker (A/B/C/D) ms | speedup vs N=1 |
|---|---|---|---|---|
| 1 | 254.89 | 27.9 s | 27905 / — / — / — | 1.00x |
| 2 | 277.06 | 11.2 s | 11164 / 7263 / — / — | 2.50x |
| 3 | 276.07 | 7.4 s | 7358 / 5668 / 4036 / — | 3.79x |
| 4 | 275.12 | 6.2 s | 6197 / 5160 / 3738 / 2251 | 4.50x |

Key observations:

- **N=2 is 2.50x, slightly above the 2-cluster theoretical 2x** — extra
  ~25% from amortizing per-call setup over two workers.
- **N=3 and N=4 are super-linear vs cluster count** — the extra software
  threads fill gaps the OS scheduler exposes between predict calls
  (`write_surface` memcpy, dequant scratch handoff). Not new compute, just
  better utilization.
- **End-user prefill caps at ~277 t/s** — the additional ANE-wall
  reduction beyond N=2 does not move t/s because the GPU command-encode
  wall (~30 s) is the binding constraint.

## What to look for on M4 / M5

Hypotheses worth checking:

- **M4 / M5 base/Pro have one ANE cluster (no Ultra fabric)** — expect
  N=1 = N=2 wall (no parallelism win). If N=2 < N=1 you have either two
  clusters or hyperthreading inside the ANE library.
- **M4/M5 ANE per-cluster matmul rate vs M3** — read off N=1 wall. A
  lower N=1 wall at the same shape means a faster cluster (or higher
  weight bandwidth).
- **Per-call overhead change** — `ane_call_avg` includes the
  `write_surface` memcpy, which scales with the IOSurface size. If M4/M5
  has a different IOSurface backing path, the overhead component changes
  independently of the matmul rate.
- **Scaling cap (N=3/N=4 vs N=2)** — the super-linear gain on M3U comes
  from gaps the predict API exposes. If a future ANE library tightens
  scheduling, N=3 may not beat N=2.

## When the comparison is invalid

- **Different prompt / model**: refs and chunk count change; ANE wall is
  not directly comparable.
- **`THREADS_FIXED_BATCH` unset**: per-call batch shrinks with N, so wall
  drops for two unrelated reasons.
- **Different `DS4_FLASH_MOE_ANE_BATCHES` / `MAX_REFS`**: these change
  the call count and per-call work; pin them or report them next to the
  number.
- **First run after boot / fresh model load**: includes ctx compile time
  (`compile_attempts` > 0 in the stats line). Discard the first run or
  warm up first.

## Standalone synthetic test (N=1..16, no GPU / no model loading)

`ane_ds4_mlp_int8w_multi_smoke.m` runs the same int8w/i8i8 tiled-fused engine
the production prefill uses, but feeds it **random runtime int8 weights**
held in caller memory (not baked into the compiled ANE model).  This isolates
ANE compute from every other pipeline cost — no SSD pread, no GPU dequant,
no Metal command encoding.

It supports up to 16 concurrent worker threads, each on its own context,
so you can find the cluster-saturation ceiling cleanly:

```bash
make moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke
./moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke \
    -threads 1,2,3,4,6,8,12,16 \
    -batches 128,256 \
    -warmup 5 -iters 80
```

Output columns to compare across machines:

- `[solo] avg` — per-context single-thread baseline (TFLOP/s for this shape)
- `[conc ] aggregate TFLOP/s` — sustained ANE compute with N workers
- `[conc ] speedup vs N*solo` — how close N concurrent workers get to N
  serial baselines; flattens at the physical cluster count
- `[conc ] per-thread ms/iter min/max + contention%` — how much per-worker
  time grows under contention (bandwidth competition between clusters)

**M3 Ultra reference (DSv4 expert shape, H=4096, I=2048, B=256):**

| N  | aggregate TFLOP/s | speedup vs N×solo | per-thread contention |
|----|-------------------|-------------------|----------------------|
| 1  | ~7.0   | 1.00x | — |
| 2  | ~12.2  | 1.74x | +15% |
| 4  | ~22.1  | 3.19x | +24% |
| 8  | ~22.1  | 3.17x | +141% |
| 16 | ~22.4  | 3.16x | +396% |

Aggregate compute caps at ~22 TFLOP/s — that is the two-cluster compute
ceiling for this shape on M3U.  Beyond N=4 each worker just runs longer.
On a single-cluster M4/M5 Pro/base expect the ceiling to land around N=1.5-2
worth of solo TFLOP/s (driven by per-call overhead, not cluster parallelism).

## Related

- [`DUAL_ANE_CLUSTER_OPTIMIZATION.md`](DUAL_ANE_CLUSTER_OPTIMIZATION.md) —
  why ANE wall is the right metric for the M3 Ultra dual-cluster work.
- [`run_ane_prefill_profile_m3u.sh`](../run_ane_prefill_profile_m3u.sh) —
  the canonical profile script; honors all env knobs documented here.
- [`ane_ds4_mlp_int8w_multi_smoke.m`](ane_ds4_mlp_int8w_multi_smoke.m) —
  standalone N=1..16 ANE-only synthetic; runtime int8 weights, no GPU.
- `ds4_metal.m` `ds4_gpu_ane_prefill_threads()` and the dual_a/_b/_c/_d
  worker thunks — the N-worker scheduler implementation.
