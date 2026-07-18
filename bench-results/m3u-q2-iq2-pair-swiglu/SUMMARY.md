# M3 Ultra Q2 IQ2 gate+up+SwiGLU experiment

Date: 2026-07-17

Device: Apple M3 Ultra, 96 GiB

Model: `/Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2`

Branch: `codex/m3u-q2-shortprefill-iq2-fusion`

## Scope

- GPU/ALU only (`DS4_ANE=0`) at 128, 512, 2048, 4000, and the 8192 crossover.
- Production route: current direct-mmap `mul_mv_id` preference.
- Unfused control: force the existing grouped `mul_mm_id` path.
- Fused candidate: the same grouped path with IQ2 gate+up sharing the B tile and a fused SwiGLU/route-weight epilogue.
- Each comparison used A-B-B-A order. Outcome A/B was production/fused; the causal control A/B was grouped-unfused/grouped-fused.

## Prefill throughput

All values are prefill tokens/second. A pair is the two measurements in A-B-B-A order; mean delta is computed from the two-run arithmetic means.

| Tokens | Production pair | Fused pair | Production mean | Fused mean | Outcome delta | Unfused pair | Causal fused pair | Unfused mean | Causal fused mean | Fusion-only delta |
|---:|:---|:---|---:|---:|---:|:---|:---|---:|---:|---:|
| 128 | 17.82, 26.02 | 26.31, 26.93 | 21.920 | 26.620 | +21.44% | 27.10, 25.83 | 26.80, 26.42 | 26.465 | 26.610 | +0.55% |
| 512 | 39.12, 31.79 | 33.97, 34.62 | 35.455 | 34.295 | -3.27% | 35.12, 34.76 | 35.05, 34.62 | 34.940 | 34.835 | -0.30% |
| 2048 | 71.84, 72.37 | 89.51, 90.36 | 72.105 | 89.935 | +24.73% | 90.62, 90.15 | 88.37, 90.70 | 90.385 | 89.535 | -0.94% |
| 4000 | 95.93, 96.44 | 139.73, 140.36 | 96.185 | 140.045 | +45.60% | 139.38, 138.93 | 140.59, 140.85 | 139.155 | 140.720 | +1.12% |
| 8192 | 102.00, 103.11 | 196.41, 195.56 | 102.555 | 195.985 | +91.10% | 194.41, 195.01 | 196.42, 196.90 | 194.710 | 196.660 | +1.00% |

The large outcome gains at 2048/4000/8192 come from changing `mul_mv_id` to grouped `mul_mm_id`, not from gate/up/SwiGLU fusion. Holding grouped-MM constant isolates the fusion at -0.94% to +1.12% in this matrix. The 128 and 512 production pairs are noisy and should not be used to select a short-context policy.

At 8192, the fused GPU result (195.985 t/s) remains about 18.0% below the separately measured no-override production ANE profile (238.91 t/s), so this experiment does not justify changing the production `>=8192` ANE policy.

## Equivalence

- `ds4_test --metal-kernels`: passed (`metal-kernels: OK`).
- Grouped-unfused versus grouped-fused: byte-identical JSON for 16 greedy decode steps with top-20 logits. This proves the fused kernel preserves the grouped-MM algorithm at the tested NR1=32 production tile.
- Production `mul_mv_id` versus grouped-fused: 15/16 selected tokens match. The final token is a near-tie divergence (`"**,"` versus `"**"`). Across common top-20 entries, maximum absolute logit delta is 1.5318718 and mean absolute delta is 0.378533281. This is a route-algorithm difference, because grouped-unfused and grouped-fused are byte-identical.

## Kernel-route evidence

- Production logs show `route=mul_mv_id`.
- Grouped control logs show `route=mul_mm_id` and `kernel_mul_mm_id_iq2_xxs_f32`.
- Fused logs show `route=mul_mm_id_pair_swiglu` and `kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16`.
- Tiny expert groups still fall back to `mul_mv_id`; for example, the 8192 fused run used grouped fusion for 247 refs and `mul_mv_id` for a 24-ref group.

## Integration decision

The fused implementation remains opt-in. Its enable knob no longer changes route selection by itself; it only fuses work after grouped-MM has already been selected. `DS4_METAL_FLASH_MOE_FORCE_MM_ID` is a separate diagnostic knob used by this benchmark. The measured speed/quality tradeoff does not support making the grouped route the M3 Ultra production default without a broader quality campaign.

Raw CSV, route logs, parity JSON, and Metal-test output are in this directory. The harness is `scripts/bench_m3u_q2_iq2_pair_swiglu.sh`.
