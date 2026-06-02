# Resident ANE Prefill Handoff

Date: 2026-05-30

Target checkout:

```text
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
branch: agent-clean
remote: https://github.com/Anemll/ds4-ssd
```

Do not use `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd` as the target. That adjacent checkout is dirty and was used accidentally for some tests. Treat it only as a reference for possible patches, not as source of truth.

## Goal

Reimplement and stabilize the resident/full-model DeDup ANE prefill path in `ds4-ssd-anemll`.

The intended runtime shape is:

- resident/full GGUF model, not Flash-MoE sidecar mode;
- large prefill chunks, mainly 8K and 16K;
- high-ref routed experts run on ANE;
- low-ref routed tail falls back to GPU;
- compact routed scratch avoids full `prefill_tokens * top_k` allocations;
- ANE accepts DSv4 dense routed expert tensors where `gate/up/down` are `IQ2_XXS`;
- sidecar ANE knobs must be forced off when `--resident-ane-prefill` is used.

The current target checkout already has an uncommitted implementation attempt. Review and either keep, clean up, or reimplement it file-by-file.

## Baseline State

Last committed target baseline:

```text
201013d Add M5 NAX prefill tile controls
```

Current target working tree has source changes in:

```text
ds4.c
ds4_agent.c
ds4_bench.c
ds4_gpu.h
ds4_metal.m
```

Untracked files to review:

```text
ANE_PREFILL_PROGRESS.md
run_agent_m3u.sh
run_ane_prefill_profile_m3u.sh
run_bench_sweep_m3u.sh
run_gpu_prefill_profile_m3u.sh
```

Do not stage generated binaries or profile output:

```text
ds4-agent
moe-batch-bench/profile_runs*
*.log
*.csv
```

## Implementation Map

### `ds4_agent.c` and `ds4_bench.c`

Add resident ANE user controls:

```text
--resident-ane-prefill
--resident-ane-shared-expert
--resident-ane-oproj
--resident-ane-cache-layers N
--no-decode-split
```

`--resident-ane-prefill` must set resident-mode env defaults and prevent accidental sidecar mixing:

```text
DS4_FLASH_MOE_ANE_PREFILL=0
DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
DS4_FLASH_MOE_OVERLAP_PREFILL=0
DS4_FLASH_MOE_OVERLAP_SCHEDULER=0
DS4_RESIDENT_MOE_ANE_HYBRID=1
DS4_RESIDENT_MOE_ANE_HYBRID_OUTER=0
DS4_RESIDENT_MOE_GROUPED_GPU_TAIL=1
DS4_RESIDENT_MOE_COMPACT_SCRATCH=1
DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=1
DS4_RESIDENT_MOE_ANE_MIN_REFS=128
DS4_RESIDENT_MOE_ANE_MAX_REFS=1024
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1
DS4_FLASH_MOE_ANE_DUAL=1
DS4_FLASH_MOE_ANE_THREADS=2
DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=4
DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK=1
DS4_FLASH_MOE_ANE_PREFLUSH_EVERY=4
DS4_FLASH_MOE_ANE_BATCHES=256
DS4_FLASH_MOE_ANE_MAX_REFS=256
DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS=1
```

`ds4-bench --gen-tokens 0` should be allowed for prefill-only measurement.

### `ds4.c`

Resident prefill changes to preserve:

- compact resident routed scratch buffers sized by per-layer routed refs instead of full chunk times top-k;
- resident ANE/GPU hybrid scheduler using `DS4_RESIDENT_MOE_ANE_MIN_REFS` and `DS4_RESIDENT_MOE_ANE_MAX_REFS`;
- async ANE job queue and finish path;
- scatter-add of ANE output with router weights into the routed output buffer;
- grouped GPU tail fallback for sub-threshold refs;
- command-buffer splitting for long prefill chunks:
  - attention and FFN split for 8K+;
  - for 16K, split FFN stages such as `hc_pre`, norm, router, routed MoE, shared expert, and post stages;
- useful stats, but no noisy per-expert fallback spam by default.

Important envs:

```text
DS4_RESIDENT_MOE_ANE_HYBRID
DS4_RESIDENT_MOE_ANE_MIN_REFS
DS4_RESIDENT_MOE_ANE_MAX_REFS
DS4_RESIDENT_MOE_ANE_HYBRID_MAX_EXPERTS
DS4_RESIDENT_MOE_GROUPED_GPU_TAIL
DS4_RESIDENT_MOE_COMPACT_SCRATCH
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE
DS4_FLASH_MOE_ANE_STATS
DS4_METAL_GRAPH_PREFILL_PROFILE
```

### `ds4_metal.m`

ANE routed expert path must accept DS4 dense-model expert tensors with:

```text
gate_type == IQ2_XXS
up_type   == IQ2_XXS
down_type == IQ2_XXS or Q2_K
```

The dense DSv4 resident model uses `IQ2_XXS` down tensors. If this is not accepted, resident ANE will reject and fall back.

Required behavior:

- compute expected row bytes based on actual `down_type`;
- choose IQ2 down dequant for `IQ2_XXS`, Q2 dequant for `Q2_K`;
- keep fused gate/up/down paths only where the tensor type is actually supported;
- emit reject reason only under debug/stat control;
- preserve detailed start rejection diagnostics for shape, row bytes, mode, refs, and missing pipeline cases;
- keep fallback to GPU on compile/load/eval failure.

Shared expert cache support is optional but useful:

```text
DS4_SHARED_EXPERT_ANE_CACHE_LAYERS
DS4_FLASH_MOE_ANE_SHARED_CACHE_LAYERS
```

### `ds4_gpu.h`

Keep declarations in sync for any new ANE helper functions, especially shared expert cache window helpers and resident routed ANE entry points.

## Validation Commands

Build first:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
make ds4-agent ds4-bench
```

Use a lock file so user sessions are not interrupted:

```bash
export DS4_LOCK_FILE=/tmp/ds4-resident-ane-handoff.lock
```

8K resident ANE smoke, prefill only:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
DS4_LOCK_FILE=/tmp/ds4-resident-ane-handoff.lock \
DS4_METAL_PREFILL_CHUNK=8192 \
DS4_METAL_GRAPH_RAW_CAP=8704 \
DS4_RESIDENT_MOE_ANE_MIN_REFS=256 \
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=4 \
DS4_FLASH_MOE_ANE_STATS=1 \
DS4_METAL_GRAPH_PREFILL_PROFILE=1 \
./ds4-bench \
  -m /Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf \
  --metal \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_12k.txt \
  --ctx-start 8192 \
  --ctx-max 8192 \
  --step-incr 8192 \
  --gen-tokens 0 \
  --resident-ane-prefill \
  --no-decode-split
```

16K resident ANE smoke, prefill only:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd-anemll
DS4_LOCK_FILE=/tmp/ds4-resident-ane-handoff.lock \
DS4_METAL_PREFILL_CHUNK=16384 \
DS4_METAL_GRAPH_RAW_CAP=16896 \
DS4_RESIDENT_MOE_ANE_MIN_REFS=384 \
DS4_FLASH_MOE_ANE_OUTPUT_QUEUE=4 \
DS4_FLASH_MOE_ANE_STATS=1 \
DS4_METAL_GRAPH_PREFILL_PROFILE=1 \
./ds4-bench \
  -m /Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf \
  --metal \
  --prompt-file /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_22k.txt \
  --ctx-start 16384 \
  --ctx-max 16384 \
  --step-incr 16384 \
  --gen-tokens 0 \
  --resident-ane-prefill \
  --no-decode-split
```

Sidecar profiling scripts are separate from resident/full-model DeDup. They are useful for comparison, but they do not validate resident DeDup by themselves:

```bash
./run_ane_prefill_profile_m3u.sh
./run_gpu_prefill_profile_m3u.sh
./run_bench_sweep_m3u.sh
./run_agent_m3u.sh
```

## Known Measurements

Recorded in `ANE_PREFILL_PROGRESS.md`:

```text
8K direct GPU-only chunk, PREFILL_CHUNK=8192: OOM
8K GPU-only baseline, PREFILL_CHUNK=2048: 82.48 t/s
8K resident dual-ANE+GPU, PREFILL_CHUNK=8192, ANE_MIN_REFS=256: 98.61 t/s
16K resident dual-ANE+GPU, PREFILL_CHUNK=16384, ANE_MIN_REFS=384: 58.14 t/s
16K GPU-only baseline, PREFILL_CHUNK=2048: stopped after >12 min
```

Recorded ANE health:

```text
8K:  calls=2038 ok=2038 eval_failures=0 pad_util=75.62%
16K: calls=2243 ok=2243 eval_failures=0 pad_util=84.44%
```

Diagnosis from those runs:

- 8K resident ANE is a real win because direct 8K GPU chunk OOMs and 2K chunked GPU is slower.
- 16K resident ANE completes, but total time is dominated by the small classic GPU fallback tail, not ANE eval.
- After aborting long 16K GPU-only runs, macOS may leave `ds4-bench` in kernel exit state; wait for cleanup or reboot the driver/session before continuing sweeps.

## Open Work

1. Re-run clean 8K and 16K resident tests after build.
2. Sweep `DS4_RESIDENT_MOE_ANE_MIN_REFS` at `32, 64, 128, 256, 384, 512`.
3. Verify `IQ2_XXS` down handling in all routed ANE paths: sync, async, i8i8, tiled fused, and fallback.
4. Replace any remaining per-expert low-ref fallback loops with grouped GPU tail where correctness holds.
5. Keep sidecar and resident naming/logging separate:
   - sidecar GPU-only default is Flash-MoE routed DeDup/slot-bank;
   - resident path is full-model DeDup;
   - do not call generic M3U GPU paths "NAX" in logs.
6. Do not default on shared expert or O-proj ANE until timeline traces show they do not create join wait.
7. Commit only source, scripts, and docs. Exclude binaries, logs, CSVs, profile output, and temporary ANE compile artifacts.

## Wrong-Checkout Reference

The adjacent checkout:

```text
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
branch: codex/integrate-ds4-agent
```

contains additional dirty changes and many untracked logs/binaries/probes. If needed, inspect it only with targeted commands such as:

```bash
git -C /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd diff -- ds4.c ds4_metal.m ds4_agent.c ds4_bench.c ds4_gpu.h
```

Do not copy the whole tree. Reimplement only the parts that survive review against `agent-clean`.
