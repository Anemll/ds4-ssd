# M5 Max Dense Prefill Backend Gates

_2026-05-28, M5 Max, resident GGUF path (`--moe-mode off`), DeepSeek-V4-Flash IQ2XXS._

## Run Setup

Model:
`/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`

Prompt:
`/tmp/big_prompt.txt`

Tool:
`./ds4-bench --metal --moe-mode off --warm-weights --gen-tokens 1`

Prefill chunk:
`DS4_METAL_PREFILL_CHUNK=min(ctx, 16384)`, so the 32768-token row runs as
two 16384-token prefill chunks.

Raw CSVs:

- `moe-batch-bench/profile_runs/m5m_dense_prefill_backend_sweep_20260528_232238.csv`
- `moe-batch-bench/profile_runs/m5m_dense_prefill_denseproj_overlay_20260528_234332.csv`
- Controlled 16K rerun with 45s pauses:
  `moe-batch-bench/profile_runs/m5m_dense_prefill_16k_pause45_20260529_081750.csv`
- Kernel-log audit, default raw cap:
  `moe-batch-bench/profile_runs/m5m_dense_prefill_16k_kernel_log_20260529_090236.csv`
- Kernel-log audit, raw cap raised to 18000:
  `moe-batch-bench/profile_runs/m5m_dense_prefill_16k_rawcap18k_kernel_log_20260529_090911.csv`
- Raw-cap sync sanity check for a true 16384-token routed-MoE chunk:
  `moe-batch-bench/profile_runs/m5m_dense_prefill_16k_rawcap16640_default_20260529_091758.log`
- Post-fix exact-16K NAX tile sweep:
  `moe-batch-bench/profile_runs/m5m_nax_tile_sweep_16k_20260529_093024.csv`
- Post-fix true NAX-half pairrow tile sweep:
  `moe-batch-bench/profile_runs/m5m_nax_true_half_tile_sweep_16k_20260529_094006.csv`
- Post-fix exact-16K default routed GPU tile sweep:
  `moe-batch-bench/profile_runs/m5m_default_gpu_tile_sweep_16k_20260529_095758.csv`
- Post-fix exact-16K no-NAX routed GPU baseline:
  `moe-batch-bench/profile_runs/m5m_default_gpu_no_nax_16k_20260529_100320.log`
- Corrected exact-16K default routed GPU tile sweep, with real n256:
  `moe-batch-bench/profile_runs/m5m_default_gpu_tile_sweep_corrected_16k_20260529_102723.csv`
- Corrected exact-16K no-NAX routed GPU baseline:
  `moe-batch-bench/profile_runs/m5m_default_gpu_no_nax_corrected_16k_20260529_103435.log`
- Corrected exact-16K true NAX-half pairrow sweep with `_n256`:
  `moe-batch-bench/profile_runs/nax_half_tile256_sweep_raw16640_20260529_104638.csv`
- Corrected exact-16K compact NAX-int8 tile sweep:
  `moe-batch-bench/profile_runs/nax_int8_tile_sweep_raw16640_20260529_111057.csv`

## Routed MoE Backend Sweep

All rows use the same default dense projection behavior unless noted: dense Q8
projection uses fp16-NAX by default on M5. `alu` below means the default routed
GPU path, not forced resident MPP/NAX.

Note: the full-size tables below were collected with only a 5s cooldown between
runs. They are useful for broad screening, but the 16K rows are superseded by the
controlled 45s-pause rerun later in this document.

Important: the `NAX-int8`, `NAX-half`, `ANE K1`, and `ALU+NAX+ANE K1` columns
were backend-forced diagnostic runs. The routed NAX rows used
`DS4_RESIDENT_MOE_MPP_FORCE=1` with `DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64` so they
override the normal resident gate and run even at 2K/4K where the auto gate would
avoid them. These numbers are therefore useful for backend comparison, not as
natural gate-selected results.

| Ctx | Prefill chunk | ALU/default | NAX-int8 | NAX-half | ANE K1 | ALU+NAX+ANE K1 | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 2048 | 358.8 | 163.4 | 199.3 | 159.9 | 194.5 | ALU/default |
| 4096 | 4096 | 374.0 | 241.3 | 284.4 | 237.6 | 277.4 | ALU/default |
| 8192 | 8192 | 536.5 | 448.4 | 454.1 | 438.5 | 443.9 | ALU/default |
| 16384 | 16384 | 501.0 | 467.0 | 469.1 | 463.9 | 462.5 | ALU/default |
| 32768 | 16384 | 482.0 | 453.8 | 453.0 | 449.4 | 448.9 | ALU/default |

Conclusion: do not force the resident routed MoE MPP/NAX paths on M5 Max. The
current default routed GPU path is fastest at every measured size. The resident
ANE hybrid K1 path is still diagnostic and should remain gated off for production
even before correctness/completeness concerns, because it loses throughput here.

## Dense Projection Overlay

This isolates the dense projection gate on top of the routed backend result.
This table also used the quick 5s cooldown, so use it as a screening result only.

| Ctx | Prefill chunk | Pure legacy dense simd | Default dense fp16-NAX | Default + dense W8A8 | NAX-half + dense W8A8 | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 2048 | 296.9 | 358.8 | 353.6 | 199.4 | Default dense fp16-NAX |
| 4096 | 4096 | 306.8 | 374.0 | 368.8 | 283.8 | Default dense fp16-NAX |
| 8192 | 8192 | 412.9 | 536.5 | 549.6 | 461.7 | Default + dense W8A8 |
| 16384 | 16384 | 401.4 | 501.0 | 514.2 | 470.7 | Default + dense W8A8 |
| 32768 | 16384 | 376.4 | 482.0 | 499.4 | 469.1 | Default + dense W8A8 |

Conclusion from the quick screen: dense fp16-NAX clearly beats legacy dense simd.
Dense W8A8 looked better at 8K+, but the 16K row was later shown to be affected
by cooldown/run-order noise.

## Controlled 16K Rerun

This pass reran only 16K, with `DS4_METAL_PREFILL_CHUNK=16384` and 45 seconds
between variants.

| Pass | Backend | Prefill t/s |
|---:|---|---:|
| 1 | pure legacy dense simd | 414.6 |
| 2 | default routed + dense fp16-NAX | 554.3 |
| 3 | default routed + dense W8A8 | 555.3 |
| 4 | forced routed NAX-int8 + dense fp16-NAX | 496.4 |
| 5 | forced routed NAX-half + dense fp16-NAX | 497.4 |
| 6 | forced routed NAX-half + dense W8A8 | 497.7 |
| 7 | default routed + dense fp16-NAX repeat | 555.4 |

Controlled 16K conclusion: default routed GPU path remains the winner in this
older pass. Dense W8A8 is effectively tied with dense fp16-NAX at 16K, not a
proven production gate win. Treat the rows labeled `NAX-half` in this older pass
as historical only: later audit found the compact bridge can ignore
`DS4_RESIDENT_MOE_NAX_HALF`, so compact-bridge rows logged before the fix are
not valid true-half measurements.

## Precise 16K Kernel Audit

`DS4_RESIDENT_MOE_KERNEL_LOG=1` was added to print the resolved resident routed
MoE path. The key finding is that nominal `DS4_METAL_PREFILL_CHUNK=16384` is not
enough to describe the kernel shape. The actual routed-MoE `tokens=` value and
tile selected by `ds4_gpu_moe_mm_tile_n()` decide whether the default GPU path
gets the wide kernels.

The wide-GPU measurements in this audit are historical. A later layer-0
`ffn_moe_out` dump showed the `_n64/_n128` template was only computing the first
32-token subtile while advertising a wider dispatch tile. The corrected wide
kernel loops over 32-token subtiles internally; see the corrected sweep below.

Raw-cap rule: `DS4_METAL_GRAPH_RAW_CAP` should not be set equal to
`DS4_METAL_PREFILL_CHUNK`. The graph chunk is bounded by
`raw_cap - raw_window`, and `raw_window=128` for this model. For desired
effective routed-MoE chunk `C`, use:

`DS4_METAL_GRAPH_RAW_CAP=align_up(C + 128, 256)`

The allocated context must also be at least that raw cap, otherwise the raw cap
is clamped by context allocation. For a true 16384-token routed-MoE chunk:

`DS4_METAL_PREFILL_CHUNK=16384 DS4_METAL_GRAPH_RAW_CAP=16640`

The pre-correction sanity check with `--ctx-alloc 16640` reported
`raw_kv_rows=16640`, `tokens=16384`, selected the `_n128` default routed
kernels, and measured `544.4 t/s`. That speed number is superseded by the
corrected sweep below.

Code change: when `DS4_METAL_PREFILL_CHUNK` is explicitly set and
`DS4_METAL_GRAPH_RAW_CAP` is not, raw cap now auto-follows
`align_up(prefill_chunk + raw_window, 256)` instead of being silently capped at
8192. Explicit `DS4_METAL_GRAPH_RAW_CAP` remains a manual override; if it limits
the effective chunk below the requested prefill chunk, ds4 prints a warning.

Logging change: `DS4_RESIDENT_MOE_KERNEL_LOG=1` now reports both
`mpp_half_requested` and the actually effective `mpp_mode`. The compact bridge
currently runs the int8 fused-dequant path even when
`DS4_RESIDENT_MOE_NAX_HALF=1` is requested, so it now logs as `mpp_mode=int8`.
True NAX-half measurements require the pairrow bridge.

### Default raw cap

Default raw cap reports `raw_kv_rows=8192`; the resident routed-MoE kernel sees
`tokens=8064`, which is divisible by 128 and selects wide tile-128 kernels.
So a nominal 16384-token prefill runs as `8064 + 8064 + 256`; stats lines that
show `tokens=256` are the final tail ubatch, not an MPP/NAX threadgroup size and
not the main compute shape.

| Variant | Effective MoE tokens | Route | Gate/up kernel | Down kernel | Mid | Dense | t/s |
|---|---:|---|---|---|---|---|---:|
| default routed | 8064 | `default-GPU-mul_mm_id` | `kernel_mul_mm_id_iq2_xxs_f32_n128` | `kernel_mul_mm_id_q2_K_f16_n128` | f16 | fp16-NAX | 533.9 |
| forced NAX-int8 | 8064 | `forced/resident-MPP-NAX-int8` | `ds4_mpp_iq2_i8_i32_counted` via compact indirect | `ds4_mpp_q2k_i8_i32_counted` | i8/i32 staged | fp16-NAX | 476.4 |
| forced NAX-half label, pre-fix compact log | 8064 | compact bridge, half label not trusted | `ds4_mpp_fused_gate_up_swiglu_h_h_f_n32` | h×h down | pre-fix log | fp16-NAX | 474.3 |
| default routed repeat | 8064 | `default-GPU-mul_mm_id` | `kernel_mul_mm_id_iq2_xxs_f32_n128` | `kernel_mul_mm_id_q2_K_f16_n128` | f16 | fp16-NAX | 529.3 |

This pre-correction audit explains why the old default looked fast, but the wide
tile speed itself is not trusted after the `ffn_moe_out` dump mismatch. The
forced resident MPP/NAX path still pays map/compact, gather, scatter, and staged
intermediate overhead.

### Raw cap 18000

With `DS4_METAL_GRAPH_RAW_CAP=18000`, the run reports `raw_kv_rows=16386`; the
routed-MoE kernel sees `tokens=16258`, which is not divisible by 64 or 128. The
default GPU route therefore falls back to tile-32 kernels.
In this `ds4-bench` run, `raw_cap=18000` was clamped by `ctx_alloc=16386`, so
the nominal 16384-token prefill became `16258 + 126`; stats lines that show
`tokens=126` are again the tail ubatch.

| Variant | Effective MoE tokens | Route | Gate/up kernel | Down kernel | Mid | Dense | t/s |
|---|---:|---|---|---|---|---|---:|
| default routed | 16258 | `default-GPU-mul_mm_id` | `kernel_mul_mm_id_iq2_xxs_f32` | `kernel_mul_mm_id_q2_K_f16` | f16 | fp16-NAX | 365.9 |
| forced NAX-int8 | 16258 | `forced/resident-MPP-NAX-int8` | `ds4_mpp_iq2_i8_i32_counted` via compact indirect | `ds4_mpp_q2k_i8_i32_counted` | i8/i32 staged | fp16-NAX | 471.3 |
| forced NAX-half label, pre-fix compact log | 16258 | compact bridge, half label not trusted | `ds4_mpp_fused_gate_up_swiglu_h_h_f_n32` | h×h down | pre-fix log | fp16-NAX | 476.0 |
| default routed repeat | 16258 | `default-GPU-mul_mm_id` | `kernel_mul_mm_id_iq2_xxs_f32` | `kernel_mul_mm_id_q2_K_f16` | f16 | fp16-NAX | 371.5 |

After the wide-tile correctness fix, the gate is simpler: default routed GPU
uses the legacy 32-token tile unless a wider tile is explicitly requested for
diagnostics, and compact resident NAX-int8 is the only verified speed win at
exact 16K.

### Post-fix Exact 16K NAX Tile Sweep

After the raw-cap auto-follow fix, `DS4_METAL_PREFILL_CHUNK=16384` with
`--ctx-alloc 16640` gives `raw_kv_rows=16640` and exact routed-MoE
`tokens=16384` without needing a manual `DS4_METAL_GRAPH_RAW_CAP`.

All rows below use dense fp16-NAX, 45s cooldowns between variants, and
`DS4_RESIDENT_MOE_KERNEL_LOG=1`. The non-MPP default routed GPU row in this
older table was captured before the wide-tile correctness fix below and is kept
only as historical context.

| Variant | Bridge | MPP/NAX tile | Threads/tg | Effective MoE tokens | t/s |
|---|---|---|---:|---:|---:|
| default routed GPU | n/a | GPU routed `_n128` | 128 | 16384 | 547.6 |
| NAX-int8 | compact | M64 x N32 x K256 | 128 | 16384 | 562.7 |
| NAX-half requested, invalid as half | compact | actually int8 compact path | 128 | 16384 | 564.5 |
| NAX-half requested, invalid as half | compact | actually int8 compact path | 128 | 16384 | 563.2 |
| NAX-half requested, invalid as half | compact | actually int8 compact path | 128 | 16384 | 562.0 |

The three rows labeled `NAX-half requested` are not valid true-half numbers:
they were taken before the logging fix and the compact bridge still executed the
int8 fused-dequant path. Treat their spread as compact-int8 noise, not as a half
tile sweep. The validated compact int8 result is `562.7 t/s`.

### Corrected Compact NAX-int8 Tile Sweep

This sweep tests the production compact fused-dequant int8 path with
`DS4_RESIDENT_MOE_NAX_INT8_TILE={32,64,128,256}`. All rows use
`DS4_METAL_PREFILL_CHUNK=16384`, `DS4_METAL_GRAPH_RAW_CAP=16640`,
`--ctx-alloc 16640`, dense fp16-NAX, and 45s cooldowns.

Correctness smoke before the sweep: layer-0 `ffn_moe_out` at 1K prompt showed
N64, N128, and N256 all byte-identical to N32 (`max_abs=0.0`). The N256 int8
kernel cannot use one true 256-wide dequant B tile because that would require
64KB threadgroup memory and the runtime reports a 32KB limit. The implemented
N256 variant is therefore a split tile: one N256 dispatch tile internally runs
two N128 NAX ops with 32KB scratch.

| Variant | Bridge | MPP/NAX tile | Threads/tg | Effective MoE tokens | t/s |
|---|---|---|---:|---:|---:|
| NAX-int8 | compact | M64 x N32 x K256 | 128 | 16384 | 565.7 |
| NAX-int8 | compact | M64 x N64 x K256 | 128 | 16384 | 557.4 |
| NAX-int8 | compact | M64 x N128 x K256 | 128 | 16384 | 523.0 |
| NAX-int8 | compact | M64 x N256 x K256 split as 2 x N128 | 128 | 16384 | 488.5 |

N32 remains the best int8 tile. Wider N tiles reduce the number of dispatch
tiles, but the larger dequant scratch and lower parallelism cost more than they
save on this workload.

### Corrected Default Routed GPU Tile Sweep

This sweep holds the exact 16K shape fixed (`raw_kv_rows=16640`,
`tokens=16384`) and varies only `DS4_METAL_MOE_TILE_MAX`. Dense projection is
still the default fp16-NAX path here.

The first wide-tile sweep (`m5m_default_gpu_tile_sweep_16k_20260529_095758.csv`)
is invalid as a correctness result: the wide template selected `_n64/_n128`, but
only computed the first 32-token subtile inside each expert-major tile. A layer-0
`ffn_moe_out` dump exposed large differences. The corrected kernel now loops
over 32-token subtiles internally; with that fix, n32/n128/n256 dumps are
byte-identical on the 2K layer-0 `ffn_moe_out` smoke test.

| Variant | Dense projection | Gate/down tile | Effective MoE tokens | t/s |
|---|---|---:|---:|---:|
| default routed GPU | fp16-NAX | 32 | 16384 | 390.9 |
| default routed GPU | fp16-NAX | 64 | 16384 | 387.8 |
| default routed GPU | fp16-NAX | 128 | 16384 | 387.7 |
| default routed GPU | fp16-NAX | 256 | 16384 | 382.6 |

So real `_n256` works and is selectable, but it is slower than n32/n64/n128 in
the corrected path. Default `DS4_METAL_MOE_TILE_MAX` is now 32; wider tiles are
kept as explicit diagnostics.

### No-NAX Baseline

This is the clean non-NAX baseline the summary table was missing: default
routed Metal GPU experts, `_n32`, dense projection forced to legacy simd with
`DS4_GPU_DENSE_NAX=0 DS4_GPU_DENSE_I8=0`, and resident MPP/NAX unset.

| Variant | Dense projection | Gate/down tile | Effective MoE tokens | t/s |
|---|---|---:|---:|---:|
| default routed GPU, no NAX anywhere | legacy-simd | 32 | 16384 | 322.7 |

### True NAX-half Pairrow Sweep

This rerun disabled the compact bridge and forced the pairrow bridge:
`DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=0`,
`DS4_RESIDENT_MOE_MPP_PAIRROW_BRIDGE=1`, and
`DS4_RESIDENT_MOE_NAX_HALF=1`. The kernel log reports `mpp_mode=half`, so these
are the valid half measurements.

Follow-up: `_n256` support was added to the true NAX-half pairrow kernels
(`ds4_mpp_h_h_f_n256` and `ds4_mpp_f_h_f_n256`) and selected through
`DS4_RESIDENT_MOE_NAX_HALF_TILE=256`. A layer-0 `ffn_moe_out` smoke test showed
N128 vs N256 was byte-identical (`max_abs=0.0`). The exact 16K sweep below used
`DS4_METAL_PREFILL_CHUNK=16384`, `DS4_METAL_GRAPH_RAW_CAP=16640`,
`--ctx-alloc 16640`, and 45s cooldowns.

| Variant | Bridge | MPP/NAX tile | Threads/tg | Effective MoE tokens | t/s |
|---|---|---|---:|---:|---:|
| NAX-half | pairrow | M64 x N32 x Kdyn | 128 | 16384 | 448.4 |
| NAX-half | pairrow | M64 x N64 x Kdyn | 128 | 16384 | 452.2 |
| NAX-half | pairrow | M64 x N128 x Kdyn | 128 | 16384 | 447.4 |
| NAX-half | pairrow | M64 x N256 x Kdyn | 128 | 16384 | 452.5 |

Among true-half pairrow runs, N256 is now the top tile by a small margin and is
effectively tied with N64. The whole true-half pairrow path is still slower than
compact int8. Do not gate production to true half unless a compact-half
implementation lands and is remeasured.

## Gate Recommendation

Production M5 Max resident prefill:

- Raw cap: for explicit large prefill chunks, let raw cap auto-follow the chunk
  unless deliberately testing a smaller raw cap. Exact effective token shape
  matters more than nominal `DS4_METAL_PREFILL_CHUNK`.
- Routed experts at exact 16K: compact resident NAX-int8 N32 is the current
  measured winner (`565.7 t/s`) among verified paths. True NAX-half pairrow is not
  competitive (`452.5 t/s` best measured).
- Default routed GPU tile gate: corrected wide tiles are no longer a speed win.
  Keep default `_n32` (`390.9 t/s`), with `_n64` (`387.8 t/s`), `_n128`
  (`387.7 t/s`), and `_n256` (`382.6 t/s`) opt-in only.
- Routed experts generally: key the gate on actual routed-MoE token count and
  selected tile, not on nominal context length. If default GPU would fall to
  tile-32 on a large unaligned shape, prefer resident MPP/NAX or split into an
  aligned chunk.
- Dense projections: keep dense fp16-NAX as the default. Keep dense W8A8 opt-in
  until controlled 8K/32K reruns confirm a real crossover.
- Manual overrides remain useful:
  - `DS4_GPU_DENSE_I8=1` enables dense W8A8 for experiments.
  - `DS4_GPU_DENSE_NAX=0` forces legacy simd dense projection for diagnostics.
  - `DS4_GPU_DENSE_I8_MIN_TOK=<N>` moves the W8A8 cutoff when W8A8 is enabled.
