# Apple M5 (base, 32 GB) routed-MoE prefill profile — ANE vs GPU/NAX-int8

Apple **M5 base** is **single-cluster ANE** (like M5 Max; M3 Ultra is dual). The
base part ships in 24/32 GB configs, so there is only **one** deployment regime:

- **Streaming** (dense GGUF + expert sidecar, `--moe-mode slot-bank`). The 81 G
  IQ2_XXS model cannot be resident in 32 GB, so there is **no resident path** and
  **no `prefill_by_tokens` kernel segments** — per
  `moe-batch-bench/PROFILE_NEW_MACHINE.md`, segments are resident-only; streaming
  is driven entirely by the `sidecar_env` `DS4_FLASH_MOE_*` knobs.

Sibling of `PREFILL_PROFILE_M5MAX.md` (read that for the resident regime, 128/96 GB
Max only).

## Why this doc exists (the profile gap)

`ds4_profile.json` matches on a chip **substring** + `min_ram_gib`. Before this
sweep the lowest M5 entry was `Apple M5 Max` / `min_ram_gib: 48`. A plain
**`Apple M5` / 32 GB** box matches **neither** (`"Apple M5 Max"` is not a
substring of `"Apple M5"`, and 32 < 48), so it fell through to the catch-all `{}`
entry — which has **no `sidecar_env`**. A bare `./ds4 --moe-mode slot-bank` on
base M5 therefore got **zero ANE streaming config** and ran routed experts on the
GPU with default thresholds. The `_m5` profiling scripts hide this because they
hardcode every `DS4_FLASH_MOE_*` var.

Fix: a new `Apple M5` / `min_ram_gib: 24` streaming-only entry, ordered **after**
all `Apple M5 Max` entries (since `"Apple M5"` ⊂ `"Apple M5 Max"`).

## Setup

Apple M5 / 32 GB, `slots=48` (~14 G RSS), SSD slot-bank only. Prompts synthesized
to size by `run_{ane,gpu}_prefill_profile_m5.sh`. Single gen token,
`DS4_PROFILE=none`, route-only ANE (`ANE_SHARED_EXPERT=0 ANE_OUTPUT_PROJ=0`),
single cluster (`ANE_DUAL=0 THREADS=1`). Hot single runs — trends robust, exact %
±a few. Measured 2026-06-02.

**GPU baseline is `nax_int8` (MPP int8 matmul2d, W8A8)** — the real M5 competitor,
per `PROFILE_NEW_MACHINE.md` §1 ("on M5+ compare ANE against nax_int8, not ALU").
The same script with `ANE_PREFILL=0 MPP_INT8_PREFILL=1 MPP_I8I8_PREFILL=1
MPP_I8I8_FUSED_PREFILL=1` produces this baseline on the identical synthesized
prompts/ctx. ALU (`mul_mm_id`) numbers are kept below only as a reference — they
are a *weaker* baseline and overstate ANE's win (e.g. 8k ANE was +31% vs ALU but
only +16% vs nax_int8).

## Segment A — fixed chunk 16K, prompt sweep

| prompt | nax_int8 | best ANE | min_refs | ANE vs nax_int8 | (ref: GPU/ALU, ANE vs ALU) |
|-------:|---------:|---------:|---------:|----------------:|:--|
| 128 | — | 20.09 | 256 | — | 13.70 (noise — tiny prompt) |
| 256 | 31.75 | 29.71 | 32 | −6% (GPU wins) | 31.02 (−4%) |
| 1K | 66.49 | 68.70 | 32 | +3% | 67.50 (+2%) |
| 4K | 100.15 | 118.55 | 64 | **+18%** | 98.50 (+20%) |
| 8K | 119.33 | 138.84 | 64 | **+16%** | 106.14 (+31%) |
| 10K | 122.14 | 142.38 | 64 | **+17%** | 98.90 (+44%) |
| 12K | 121.84 | 138.59 | 64 | **+14%** | 94.88 (+46%) |
| 14K | 106.31 | 133.91 | 64 | **+26%** | 84.23 (+59%) |
| 16K | 98.57 | 130.20 | 64 | **+32%** | 83.27 (+56%) |

## Segment B — 8K prompt, chunk sweep

| chunk | nax_int8 | best ANE | min_refs | ANE vs nax_int8 |
|------:|---------:|---------:|---------:|----------------:|
| 1024 | 61.32 | 64.20 | 64 | +5% |
| 2048 | 80.24 | 90.16 | 64 | +12% |
| 4096 | 98.14 | 115.31 | 64 | **+18%** |
| 8192 | 118.19 | 138.02 | 64 | +17% |
| 16384 | 119.33 | 138.49 | 128 | +16% |

## Verdict

- **No ANE→GPU crossover anywhere ≥1K.** ANE wins the entire production band
  against the correct nax_int8 baseline. Only 256-token prompts favor GPU (−6%);
  1K is a wash (+3%); 4K-up is a clear ANE win.
- **The win is smallest mid-band (+14% at 12K) and largest at 16K (+32%).**
  nax_int8 GPU peaks ~122 t/s at 10–12K then **degrades** (ctx/KV overhead at
  large ctx: 14K→106, 16K→99), while ANE stays flat at 130–142. The base-M5 GPU
  does not scale into large prompts the way M5 Max's does, so ANE pulls ahead
  exactly where the agent's large prefills live.
- **`min_refs` firmly 32–64** across every prompt and chunk; mr128+ loses. Profile
  uses `ANE_MIN_REFS=32` + `HYBRID_ANE_MIN_REFS=64`. The M3 Ultra default of 384
  is wrong for M5.
- **Streaming can't express a per-token crossover** (one static `sidecar_env`, no
  `prefill_by_tokens`). Since ANE wins all production sizes and only loses at 256
  tokens (rare, and by 6%), **ANE-on is the correct static default**.

## vs M5 Max

Same *shape* (ANE wins broadly ≥1K, `min_refs` 32–64) but the chunk/prompt trend
**inverts at the top**: on M5 Max the ANE win shrinks as size grows (NAX-int8
scales with batch → 16K chunk only +3%); on **base M5 the nax_int8 GPU degrades
past 12K** so the ANE win *grows* to +32% at 16K. Base-M5 nax_int8 is weaker
relative to ANE than the Max's.

## Profile decision (ds4_profile.json)

New entry `{ "chip": "Apple M5", "min_ram_gib": 24 }`, streaming-only
`sidecar_env`, cloned from the `M5 Max <96G` entry with `SLOT_BANK_SLOTS=48` for
32 G RAM. Single cluster (`ANE_DUAL=0 THREADS=1`), shared-expert + O-proj ANE off,
`ANE_MIN_REFS=32`, `HYBRID_ANE_MIN_REFS=64`. No resident `env` /
`prefill_by_tokens` — base M5 cannot host the model, and streaming has no kernel
segments.

## Note on the "disk leak" during this sweep

The sweep tripped `ENOSPC` twice on the first attempts. Root cause was **not** a
`ds4` leak: `ds4` writes only KB-sized logs, and a whole-volume scan found no
large file created during runs. The data volume was physically ~95–100 % full;
the few GB of "free" bounced (purgeable space + macOS swap churn), and the
8K/ctx=10000 runs add ~10 GB transient swap that tipped the razor-thin margin
over. With ~100 GB free the full sweep completed with 0 failures, dipping only to
96.3 GB min. Re-running on a near-full disk: free tens of GB first.
