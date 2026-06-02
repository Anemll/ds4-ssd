# ANE INT8 Scale Resolver And Checks

Date: 2026-06-02

This note records the current routed-MoE INT8 qscale/scale contract, the checks
added for ANE precision work, and which INT8 kernels are production-improved
versus still experimental.

Important: this is not claiming that a new numeric qscale was discovered for
the profiled M3 Ultra sidecar/resident runs. Those runs already exported
`512/32/32` through the MPP env names. The improvement here is resolver/default
normalization: every ANE INT8 path now resolves the same qscale contract, even
when profile envs are missing or when ANE-specific env names are used.

## Current Scale Contract

The routed prefill INT8 helpers now use one shared resolver in `ds4_metal.m`.
The environment values are qscales. The kernel scale values printed in
diagnostics are their reciprocals:

```text
qscale=512 -> scale=1/512 = 0.00195312
qscale=32  -> scale=1/32  = 0.03125
```

So this runtime line is the expected production contract:

```text
ds4: prefill quant scales (ANE env first): w_qscale=512 x_qscale=32 mid_qscale=32 w_scale=0.00195312 x_scale=0.03125 mid_scale=0.03125
```

| Quantity | Preferred ANE env | Fallback env | Default qscale | Scale |
|---|---|---|---:|---:|
| Weight dequant | `DS4_FLASH_MOE_ANE_INT8_QSCALE` | `DS4_FLASH_MOE_MPP_INT8_QSCALE` | 512 | `1/512` |
| Input activation | `DS4_FLASH_MOE_ANE_X_QSCALE` | `DS4_FLASH_MOE_MPP_INT8_X_QSCALE` | 32 | `1/32` |
| Hidden/mid activation | `DS4_FLASH_MOE_ANE_MID_QSCALE` | `DS4_FLASH_MOE_MPP_INT8_MID_QSCALE` | 32 | `1/32` |

This replaces the older ANE-only qscale fallback of `64/16/16`. In runs where
`DS4_FLASH_MOE_MPP_INT8_QSCALE=512`, `DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32`, and
`DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32` were already set, the numeric scales are
unchanged. The fix is for unprofiled/mis-profiled ANE runs and for code paths
that use ANE-specific env names.

Before this resolver cleanup:

```text
ANE env absent + MPP env present  -> 512/32/32
ANE env absent + MPP env absent   -> 64/16/16
```

After this resolver cleanup:

```text
ANE env absent + MPP env present  -> 512/32/32
ANE env absent + MPP env absent   -> 512/32/32
ANE env present                   -> ANE env values, with MPP fallback
```

## Coverage Check

The resolver/default cleanup now covers these runtime paths:

| Path | Coverage | Env precedence |
|---|---|---|
| Routed MoE ANE mode 6/13 | shared resolver | `DS4_FLASH_MOE_ANE_*` first, then `DS4_FLASH_MOE_MPP_INT8_*` |
| Routed MoE ANE compare/reference | shared resolver | MPP/NAX first for reference, ANE fallback |
| Routed MoE MPP/NAX INT8 resident/sidecar | shared resolver | `DS4_FLASH_MOE_MPP_INT8_*` first, then `DS4_FLASH_MOE_ANE_*` |
| Shared expert ANE i8i8 | local wrapper, normalized fallback | `DS4_SHARED_EXPERT_ANE_*`, then generic `DS4_FLASH_MOE_ANE_*`, then MPP |
| Output projection ANE I8X | local X-qscale wrapper, normalized fallback | `DS4_FLASH_MOE_ANE_OUTPUT_PROJ_X_QSCALE`, then generic ANE X, then MPP X |

The MPP/NAX kernels themselves still receive raw qscale values as Metal
arguments. The fix is the host-side resolution of those qscales, not a shader
change.

NAX-half is not an INT8 path, so it is unaffected by this scale resolver.

For route-before experiments only:

```bash
DS4_FLASH_MOE_ANE_ROUTE_MID_QSCALE=<q>
# alias:
DS4_FLASH_MOE_ANE_ROUTE_BEFORE_MID_QSCALE=<q>
```

These only apply when `DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT=1`. There is no
production default override for this because the fixed route-before mid scale is
not proven safe.

## Production Default

Keep route-before quantization off:

```bash
unset DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT
# or
DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT=0
```

The production routed ANE path is still mode 6:

```text
i8w-i8x-tiled-fused
```

The route weight is applied after the ANE tiled-fused expert output. This avoids
throwing away hidden/mid INT8 levels for small route weights.

Mode 13:

```text
i8w-i8x-tiled-fused-routed
```

is a diagnostic route-before graph. It is useful for proving graph wiring and
for scale sweeps, but it should not be enabled by default until it passes
real-model divergence checks.

## Improved INT8 Pieces

### ANE routed MoE tiled-fused mode 6

Status: improved and production path.

What improved:

- Uses the unified `512/32/32` INT8 qscale contract when no explicit override is
  present, which prints as reciprocal kernel scales
  `0.00195312/0.03125/0.03125`.
- Uses ANE-specific env names, with MPP names as fallback, so sidecar and
  resident tests can share the same scale values without duplicated parsing.
- Prevents accidental fallback to old ANE defaults `64/16/16` when profile envs
  are not applied.
- Keeps route-after behavior by default.
- Dual-cluster execution remains controlled by:

```bash
DS4_FLASH_MOE_ANE_DUAL=1
DS4_FLASH_MOE_ANE_THREADS=2
DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1
```

### ANE routed MoE mode 13 route-before

Status: improved as a diagnostic, not production.

What improved:

- Adds a routed MIL graph that accepts route weights inside the ANE graph.
- Adds a parity check against mode 6 when route is exactly `1.0`.
- Adds optional route-before mid-qscale override for sweeps.

Known problem:

- With route weights below `1.0`, route-before quantizes a smaller hidden value
  using the same fixed INT8 step. That can lose effective precision compared to
  route-after, where the unweighted hidden value uses the full mid-qscale range
  and route is applied at the output.

### ANE compare diagnostics

Status: improved.

What improved:

- `DS4_FLASH_MOE_ANE_COMPARE` now keeps the dequant scratch slot alive until the
  compare has consumed it. This avoids stale scratch false positives.
- Compare output now labels the staged-i8 reference limitation:

```text
note=staged_i8_wx_not_exact_mulmm
```

That note matters: the compare is a useful ANE-vs-staged-INT8 check, but it is
not yet a complete replacement for comparing against the default GPU/ALU route.

### MPP/NAX INT8 routed prefill

Status: safer gating, still chip-specific.

What improved:

- MPP/NAX support is gated to M5-class devices by `ds4_gpu_mpp_nax_supported()`.
- On M3 Ultra, `gpu` should mean the default GPU/ALU route, not NAX-int8.
- This prevents M3U profile and benchmark confusion where "GPU" could silently
  mean an MPP/NAX INT8 path that is not the intended baseline.

## Still Needs Work

### Route-before ANE precision

Mode 13 still needs a better quantization strategy before production use. Likely
options:

- route-aware per-row mid scale,
- apply route after the INT8 down projection instead of before mid quantization,
- keep mid activation in fp16 for the routed graph,
- or prove a safe qscale schedule over real routed weights.

Current fixed-scale route-before is not enough.

### Real GPU/ALU vs ANE divergence

The current smoke tests and `DS4_FLASH_MOE_ANE_COMPARE` checks are useful, but
the important production comparison is still:

```text
default GPU/ALU routed expert output vs ANE route-after mode 6 output
```

Do not use MPP/NAX-int8 as the only baseline for M3 Ultra. That answers a
different question.

### Int8-output mode 7

The `i8w-i8x-tiled-fused-i8out` path exists, but it is still experimental. It
needs output-scale correctness and full downstream accumulation checks before
being used for resident or sidecar prefill.

### MPP/NAX INT8 on M5

M5/M5 Max can use the MPP/NAX path, but it still needs chip-specific A/B:

- NAX-int8 vs default GPU/ALU,
- NAX-int8 vs ANE mode 6,
- short context and deep chunk behavior,
- generated-token divergence with `temp=0`.

### Dense/shared INT8 ANE

The routed expert scale fix does not automatically validate dense/shared expert
ANE kernels. Those paths need their own row/logit checks because their
activation distribution and route-weight placement are different.

## Improvement Checks

Build:

```bash
make ds4 ds4-agent ds4-bench moe-batch-bench/ane_ds4_mlp_i8i8_precision_smoke -j8
git diff --check -- ds4_metal.m moe-batch-bench/ane_ds4_mlp_i8i8_precision_smoke.m
```

Route identity check. This proves the routed graph wiring is exact when route is
`1.0`:

```bash
./moe-batch-bench/ane_ds4_mlp_i8i8_precision_smoke \
  -H 4096 -I 2048 -B 256 \
  -wq 512 -xq 32 -midq 32 \
  -route 1 -const-route -ane-parity
```

Observed:

```text
mode13_routed_vs_mode6_scaled max_abs=0 max_rel=0 rms=0 rel_rms=0
```

Production-shape route-before row-0 diagnostic:

```bash
./moe-batch-bench/ane_ds4_mlp_i8i8_precision_smoke \
  -H 4096 -I 2048 -B 37 \
  -wq 512 -xq 32 -midq 32 \
  -route 0.27048698 -const-route -routed -row0-ref
```

Observed on 2026-06-02:

```text
ane_row0_vs_cpu_i8_row0 rel_rms=0.478044
cpu_i8_row0_vs_fp16_row0 rel_rms=0.519614
ane_row0_vs_fp16_row0 rel_rms=0.39468
```

This is a diagnostic, not a pass condition for production. It shows that graph
wiring works, but it also shows route-before quantization error is material at a
real route weight.

Short real-model ANE compare:

```bash
M=/Volumes/optane/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf
P=/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp/tools/flashmoe-sidecar/prompts/coding/coding_8k.txt

DS4_LOCK_FILE=/tmp/ds4-ane-routeafter-compare.lock \
DS4_METAL_NO_RESIDENCY=1 \
DS4_METAL_DECODE_RESIDENCY=0 \
DS4_FLASH_MOE_ANE_COMPARE=5 \
DS4_FLASH_MOE_ANE_STATS=1 \
./ds4-bench -m "$M" --metal --prompt-file "$P" \
  --ctx-start 1024 --ctx-max 1024 --ctx-alloc 4096 \
  --gen-tokens 1 --csv /tmp/ds4_ane_routeafter_compare.csv
```

Route-before should only be checked explicitly:

```bash
DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT=1 \
DS4_FLASH_MOE_ANE_COMPARE=5 \
./ds4-bench ...
```

Do not leave `DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT=1` in profiles or production
agent scripts until it passes the real GPU/ALU comparison and generated-token
checks.

## Recommended M3 Ultra INT8 Prefill Baseline

For sidecar and resident ANE route-after testing on M3 Ultra:

```bash
DS4_FLASH_MOE_ANE_I8I8_PREFILL=1
DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1
DS4_FLASH_MOE_ANE_DUAL=1
DS4_FLASH_MOE_ANE_THREADS=2
DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1
DS4_FLASH_MOE_ANE_BATCHES=256
DS4_FLASH_MOE_ANE_MIN_REFS=32
DS4_FLASH_MOE_ANE_MAX_REFS=256
DS4_FLASH_MOE_SCHED_ANE_REL_SPEED=99
DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL=0.0
DS4_FLASH_MOE_MPP_INT8_QSCALE=512
DS4_FLASH_MOE_MPP_INT8_X_QSCALE=32
DS4_FLASH_MOE_MPP_INT8_MID_QSCALE=32
DS4_FLASH_MOE_ANE_ROUTE_BEFORE_QUANT=0
```

The MPP qscale env names are still acceptable because the ANE scale resolver
falls back to them. Use the ANE-specific names only when a run needs to separate
ANE and MPP/NAX scale experiments.
