# ANE Prefill Progress

## 2026-05-30 M3 Ultra resident dense prefill

Model:

```text
/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf
```

Prompt files:

```text
8K:  /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_12k.txt
16K: /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_22k.txt
```

### Code changes in this pass

- Resident ANE prefill accepts dense-model routed expert tensors where `ffn_down_exps` is IQ2_XXS, not only Q2_K.
- Resident ANE prefill uses compact routed scratch buffers sized by per-layer max routed refs instead of allocating full `prefill_tokens * top_k` gate/up/mid/down scratch.
- `--resident-ane-prefill` enables compact resident scratch in both `ds4-bench` and `ds4-agent`.
- Long resident prefill now splits Metal command buffers by layer stage:
  - attention and FFN are separate for 8K+ chunks;
  - 16K resident DeDup FFN additionally splits `hc_pre`, `norm`, `router`, routed MoE, shared expert, and post stages.
- `ds4-bench --gen-tokens 0` is now supported for prefill-only measurement.

### Results

| Case | Command shape | Result |
|---|---:|---:|
| 8K direct GPU-only chunk | `PREFILL_CHUNK=8192`, no resident ANE | OOM |
| 8K GPU-only baseline | `PREFILL_CHUNK=2048`, prefill to 8192 | 82.48 t/s |
| 8K resident dual-ANE+GPU | `PREFILL_CHUNK=8192`, `ANE_MIN_REFS=256` | 98.61 t/s |
| 16K resident dual-ANE+GPU | `PREFILL_CHUNK=16384`, `ANE_MIN_REFS=384`, prefill-only | 58.14 t/s |
| 16K GPU-only baseline | `PREFILL_CHUNK=2048`, prefill to 16384 | did not finish after >12 min; stopped |

8K hybrid ANE stats:

```text
calls=2038 ok=2038 eval_failures=0
ane_wall_est=10390.896 ms
pad_util=75.62%
```

16K hybrid ANE stats:

```text
calls=2243 ok=2243 eval_failures=0
ane_wall_est=18642.932 ms
pad_util=84.44%
```

### Current diagnosis

The 8K target is a real win: direct GPU 8K OOMs, GPU-only 2K-chunked 8K is 82.48 t/s, and resident dual-ANE+GPU 8K is 98.61 t/s.

The 16K target now completes prefill, but total throughput is limited by the small classic GPU fallback tail. ANE itself is not the blocker: dual-cluster wall is about 18.6 s inside a 281.8 s total prefill, with zero ANE failures.

After stopping the long 16K GPU-only baseline, macOS left `ds4-bench` processes in kernel exit state. Further Metal runs should wait for those to clear or reboot the GPU driver/session before continuing threshold sweeps.

### Next candidates

- Sweep `DS4_RESIDENT_MOE_ANE_MIN_REFS=256` at 16K after the Metal driver state is clean.
- Replace per-expert classic fallback for sub-threshold refs with a grouped GPU tail path.
- Revisit partial MPP tiles only if correctness is checked; the current workaround intentionally routes partial rows through legacy GPU.
