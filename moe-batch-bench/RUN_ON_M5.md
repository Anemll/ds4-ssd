# Running the ANE Throughput Tests on a New Chip (M4 / M5 / M5 Max)

This is a step-by-step recipe to reproduce the M3 Ultra reference numbers on
a different Apple Silicon chip so we can characterise ANE per-cluster
throughput, cluster count, and the hybrid ANE/GPU regime.

The tests below assume **macOS 26+ (any chip)** and the private
`ANEServices` framework. The same code path that production uses (i8w/i8x
tiled-fused MIL → `_ANEInMemoryModel`) is what the smokes exercise.

---

## Test A — Standalone ANE-only smoke (5 minutes, no model needed)

This is the **first thing to run** on a new chip. It needs no model
download, no GGUF, no prompt — just clones the repo and builds one tiny
binary.

### 1. Build

```bash
git clone <ds4-ssd repo url>
cd ds4-ssd
make moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke
```

If `make` complains about missing `metal/` files or the main `ds4` target,
ignore them — this smoke does not link them. Only the `int8w` object file
and the smoke source are needed.

### 2. Run the full sweep

```bash
./moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke \
    -threads 1,2,3,4,6,8,12,16 \
    -batches 16,32,64,128,256 \
    -warmup 5 -iters 80 \
    -output fp16 \
    | tee m5_ane_smoke.log
```

Adjust `-iters` down (e.g. `-iters 30`) if the run is too long; the summary
table at the end is what matters.

### 3. What to report

From the SUMMARY block at the bottom of the log, extract:

- **N=1 solo TFLOP/s for each B** — single-cluster baseline. If the chip
  has only one ANE cluster, N=2 will not improve on N=1.
- **Peak aggregate TFLOP/s for each B** — the chip's saturation ceiling.
- **N at which the peak occurs** — tells you the effective number of
  parallel ANE engines. M3 Ultra peaks at N=4 (2 clusters × 2 software
  workers per cluster). A single-cluster M4/M5 should peak at N=2 (worker
  per logical ANE thread).
- **Aggregate speedup N=2 / N=1** — should be ~2x on a 2-cluster chip
  (M3 Ultra), ~1x on a 1-cluster chip (most M4/M5 Pro/base).

### 4. M3 Ultra reference (for comparison)

```
  out    B    N   solo_TF/s   aggregate_TF/s  speedup
  fp16    16   16     0.72         3.74         5.13x
  fp16    32   16     1.44         7.24         5.17x
  fp16    64   16     2.70        12.87         4.76x
  fp16   128   12     4.90        20.00         4.08x
  fp16   256    4     7.05        22.13         3.15x
```

Per-call solo is the single-cluster baseline; peak aggregate is the
two-cluster ceiling.

---

## Test B — Production end-user prefill (requires model + prompt)

This is the apples-to-apples comparison of real prefill throughput.

### 1. Prerequisites

- The DSv4 IQ2_XXS expert-major sidecar (~120 GB):
  - `model-dense.gguf` (the dense base model)
  - The expert sidecar directory with per-expert int8 weights
- A fixed prompt (the M3U numbers used the 8423-token coding prompt at
  `prompts/coding/coding_8k.txt`)
- Enough free RAM (~70 GB) — the slot-bank stats show ~70065 MiB
  installed at the default `DS4_SLOTS=4`

### 2. Build the full binary

```bash
make ds4
```

### 3. Set paths and run the baseline

```bash
export DS4_MODEL=/path/to/dsv4-iq2xxs-expert-major/dense/model-dense.gguf
export DS4_SIDECAR=/path/to/dsv4-iq2xxs-expert-major
export DS4_PROMPT_FILE=/path/to/coding_8k.txt

# Baseline run with all shipped defaults (THREADS=2, MIN_REFS=384, ...)
DS4_RUN_NAME=m5_prod_baseline \
    ./run_ane_prefill_profile_m3u.sh
```

The script writes to `moe-batch-bench/profile_runs/m5_prod_baseline.log`.
Look at the line:
```
ds4: prefill: XXX.XX t/s, generation: ...
```

### 4. THREADS sweep (does oversubscription help on M5?)

```bash
for n in 1 2 3 4; do
  DS4_RUN_NAME=m5_threads${n} \
      DS4_FLASH_MOE_ANE_THREADS=$n \
      DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH=1 \
      ./run_ane_prefill_profile_m3u.sh
done
```

`THREADS_FIXED_BATCH=1` keeps per-call batch constant across N so the
comparison stays apples-to-apples. Extract the `ane_wall_est` and
`ane_eval_a/b/c/d` columns from each log — see
[`ANE_WALL_MEASUREMENT.md`](ANE_WALL_MEASUREMENT.md) for parsing.

Expected behaviour:

- **1-cluster M5/M4 Pro/base**: N=1 ≈ N=2 ≈ N=3 ≈ N=4 ANE wall (no
  cluster parallelism; only one ANE engine). End-user t/s should also
  be roughly flat.
- **2-cluster M5 Ultra (if/when it ships)**: should look like M3U —
  N=2 gives ~2x ANE-wall reduction, N=3/N=4 give further super-linear
  gains from per-call overhead amortization.

### 5. Hybrid ANE/GPU threshold sweep (optional)

The hybrid model routes refs below the threshold to GPU. M3U sweep was
noise-flat across 128-512:

```bash
for t in 64 128 192 256 384 512 768 1024; do
  DS4_RUN_NAME=m5_hyb${t} \
      DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS=$t \
      ./run_ane_prefill_profile_m3u.sh
done
```

If M5 has a different per-call overhead profile, the threshold sweep
will be more interesting there.

### 6. Numbers to report back

| Metric | Where to find it |
|---|---|
| Baseline prefill t/s | `ds4: prefill: XXX t/s` (default config) |
| GPU-only prefill t/s | run `run_gpu_prefill_profile_m3u.sh` |
| Hybrid uplift vs GPU-only | ratio of the two |
| Per-N ANE wall | `ane_wall_est=...` from each THREADS run |
| Per-worker eval ms | `ane_eval_a=... ane_eval_b=...` |
| Production pad_util | `ANE prefill chunks ... pad_util=XX%` |
| Per-N solo + aggregate TF/s | from Test A summary |
| GPU encode wall | `gpu layer-major prefill total ... encode=XXX ms` |

---

## Test C — Minimal "is the chip even alive" check

If you just want a one-liner that confirms the ANE library loads and a
context can be created:

```bash
./moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke \
    -threads 1 -batches 256 -warmup 1 -iters 5
```

This should finish in <3s and print one `solo_TF/s` number. If it errors
(`FAIL: create ctx[0]`), the private ANE API isn't available on this
build of macOS or the framework path differs — see
`ane_ds4_mlp_int8w.m` for the dlopen logic and fix the framework path
there.

---

## What we hope to learn about M5 / M5 Max

1. **Per-cluster matmul rate vs M3.** Read off N=1 solo TF/s at B=256.
   The M3 cluster does ~7 TF/s on this shape. A faster M5 cluster would
   bump this number directly; weight-bandwidth and compute upgrades both
   show here.

2. **Cluster count.** Is M5 base/Pro one ANE? Is M5 Max one or two? Read
   off the N=2 / N=1 speedup at B=256. If ≈ 2x, two clusters; if ≈ 1x,
   one cluster.

3. **Per-call overhead.** `ane_call_avg` at B=256 = matmul time + IO
   surface read/write + scheduling. M3U sits at ~1.85 ms. A leaner ANE
   library on M5 would show here as a smaller `ane_call_avg` while solo
   TF/s stays similar — implies the saturation N might shift lower.

4. **Hybrid sweet-spot.** Is the optimal MIN_REFS still ~128-512? If the
   GPU is much faster on M5 (Apple GPU compute scales every gen), the
   sweet-spot moves up — more work should go to GPU.

5. **End-user prefill cap.** The 277 t/s M3U cap is GPU-encode bound, not
   ANE-bound. If M5's GPU command-graph construction is faster, the cap
   moves; if it's same, more ANE wins are blocked.

---

## Related

- [`ANE_WALL_MEASUREMENT.md`](ANE_WALL_MEASUREMENT.md) — how to extract and
  interpret ane_wall_est from the production log.
- [`DUAL_ANE_CLUSTER_OPTIMIZATION.md`](DUAL_ANE_CLUSTER_OPTIMIZATION.md) —
  the M3 Ultra dual-cluster story; explains why
  `DS4_FLASH_MOE_ANE_DUAL=1` exists and what each knob does.
- [`ane_ds4_mlp_int8w_multi_smoke.m`](ane_ds4_mlp_int8w_multi_smoke.m) —
  the standalone smoke source; CLI options at the top.
- [`run_ane_prefill_profile_m3u.sh`](../run_ane_prefill_profile_m3u.sh) —
  the production profile script (works on any chip, name notwithstanding).
