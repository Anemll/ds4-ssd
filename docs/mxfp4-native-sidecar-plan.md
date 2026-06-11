# Native MXFP4 sidecar decoding — implementation plan

Branch: `MXFP4` (cut from `codex/stable-slot-replay-experiment`).
Target sidecar: `/Users/anemll/Models/DSv4-Flash-MXFP4-native-flash`
(manifest source: `/Volumes/TB36/Models/DS/DSv4-Flash-MXFP4-native-flash/manifest.json`).

## What the manifest says

- `sidecar_kind: flashmoe_gguf`, schema v1, expert-major repacked `layer_NNN.bin` files — same
  container the existing IQ2_XXS / Q2_K / Q4_K sidecars use. 43 layers × 3 families
  (gate/up/down), all `quant_type: "MXFP4"`, `include_shared: false`, no dense lead block.
- Per expert: gate/up are 4096-wide × 2048 rows, down is 2048-wide × 4096 rows.
  `bytes_per_expert = 4,456,448` per family, `expert_stride = 13,369,344 = 3 × 4,456,448`,
  family offsets 0 / 4456448 / 8912896. All consistent with 17 bytes per 32-element block:
  `4096·2048/32·17 = 4,456,448` ✓. Gate and up share one quant type, so the existing
  gate/up match check passes.

## MXFP4 format (OCP microscaling, same as ggml `GGML_TYPE_MXFP4`)

```
#define QK_MXFP4 32
typedef struct {
    uint8_t e;        // E8M0 shared scale: value = 2^(e - 127)
    uint8_t qs[16];   // 32 × E2M1 nibbles; qs[j] low nibble = elem j, high nibble = elem j+16
} block_mxfp4;        // 17 bytes
```

E2M1 magnitude LUT: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, top bit of the nibble is sign
(i.e. signed LUT `{0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6}`).
Match ggml's nibble order exactly — the source GGUF is produced by the stock converter.

## Phase 0 — host plumbing (ds4.c)

1. **Enum**: add `DS4_TENSOR_MXFP4 = 39` to the tensor-type enum (~ds4.c:889).
   The existing values mirror ggml type ids (Q8_0=8, Q2_K=10, Q4_K=12, IQ2_XXS=16, I32=26);
   39 is ggml's `GGML_TYPE_MXFP4`, keeping us GGUF-compatible.
2. **Block struct** next to the other block structs (~ds4.c:137–161):
   `block_mxfp4` as above + `DS4_STATIC_ASSERT(sizeof == 17)`.
3. **Manifest string map** `flash_moe_quant_type_id()` (ds4.c:2996):
   `if (!strcmp(quant, "MXFP4")) return DS4_TENSOR_MXFP4;`
4. **Routed-type predicate** `tensor_is_routed_expert_type()` (ds4.c:2223): add MXFP4.
5. **Row-bytes helper — the one real refactor.** `routed_expert_block_bytes()` (ds4.c:2229)
   and `routed_expert_row_bytes_for_type()` (ds4.c:2796) assume every quant block covers
   QK_K=256 elements (`width % QK_K` assert, `width/QK_K × block_bytes`). MXFP4 blocks cover
   32. Introduce `routed_expert_block_elems(type)` (256 for the K-quants, 32 for MXFP4) and
   compute `(width / elems) × block_bytes`, keeping the alignment assert per-type.
6. **CPU reference dequant** (validation only): `ds4_dequant_row_mxfp4()` (scale LUT, no
   vec_dot needed unless the CPU fallback path is exercised). Used by the layer-0
   max-abs-diff validation harness.

Bank loading itself needs **no changes**: the loader trusts manifest
`repacked_offset` / `bytes_per_expert` / `expert_stride` once the type id resolves, and
`metal_graph_flash_moe_init_expert_family_views()` slices family views from those offsets.

## Phase 1 — Metal decode, baseline GPU path (metal/moe.metal + ds4_metal.m)

Goal: correct end-to-end inference on the standard mulmm/mulmv pipelines before any
NAX/ANE work.

moe.metal:
1. `struct block_mxfp4 { uchar e; uchar qs[16]; };` + constant E2M1 LUT + E8M0→float
   (`exp2(float(e) - 127.0f)`; note e=0 → 2^-127, flush like ggml does).
2. `template <typename type4x4> void dequantize_mxfp4(device const block_mxfp4 *xb,
   short il, thread type4x4 &reg)` — `nl = 2` (two 4×4 tiles per 32-elem block), mirroring
   `dequantize_q8_0`. il=0 → low nibbles of qs[0..15], il=1 → high nibbles.
3. `mul_mm_id` instantiations (clone the q8_0 lines at moe.metal:2526/2539, which already
   use a 32-elem block with nl=2):
   - `kernel_mul_mm_id_mxfp4_f32` + `_n64/_n128/_n256`
   - `kernel_mul_mm_id_mxfp4_f16` + `_n64/_n128/_n256`
   (wide-n variants stay default-off per the wide-tile occupancy verdict, but instantiate
   for parity with the other types.)
4. `mul_mv_id` decode kernel: `kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>` +
   `kernel_mul_mv_id_mxfp4_f32` instantiation (moe.metal:~829). Port shape from the q8_0
   mv impl (same 32-block geometry). Skip the pair/pair_swiglu/sum6 fused variants for now —
   they're per-type decode optimizations to add after correctness.

ds4_metal.m:
5. `DS4_METAL_TENSOR_MXFP4 = 39` (~line 29) + type-name string case ("MXFP4").
6. Pipeline globals + reset-on-teardown: `g_moe_mul_mm_id_mxfp4_pipeline`,
   `g_moe_mul_mv_id_mxfp4_pipeline`.
7. Switch cases in: `ds4_gpu_routed_mm_pipeline()` (~11603),
   `ds4_gpu_routed_mm_f16_rhs_pipeline()` (~11628), the mv-pipeline selector (~11596),
   the per-type helper at ~11580 (check its semantics — values-per-x unit), the
   tile-n kernel-name resolver and the wide-tile gate.

Exit criteria: model loads the MXFP4 sidecar, decode + prefill produce sane text;
layer-0 expert output max-abs-diff vs CPU reference dequant ≲ 1e-3 (fp16 accum).

## Phase 2 — fast paths (optional, after Phase 1 is validated)

Per the NAX playbook these are where the wins were for Q4_K/IQ2:
1. **NAX int8 prefill**: `kernel_dsv4_mpp_dequant_mxfp4_transpose_i8` (+ `_counted`
   variant), cloned from the q4_k version at moe.metal:192/261; register its pipeline and
   add MXFP4 to the backend dispatch (`g_mpp_dequant_*_i8_pipeline` block, the
   gate/down-type checks around ds4_metal.m:12858+, and the env/backend remap tables).
2. **ANE hybrid**: `kernel_dsv4_ane_dequant_mxfp4_transpose_f16` (clone q4_k at
   moe.metal:465) so the ANE+NAX hybrid backend can consume MXFP4 banks. Watch the dual
   chunk-size engagement constraints (tile_util gate vs ANE 256 batch cap).
3. **nax_half (h_h_f) resident fused**: the validated h_h_f fused dequant recipe is the
   path that cleared the 532 t/s ceiling — a `mxfp4 → half` fused variant follows the same
   recipe and is cheap to add once (1) exists. MXFP4's dequant is LUT+shift, cheaper than
   iq2_xxs's grid lookup, so expect at-least-parity.

## Phase 3 — validation & bench

- `tests/sidecar_smoke.sh` against the local MXFP4 sidecar dir.
- `./validate_ane_nax.sh` before trusting any hybrid numbers.
- Resume-after-decode prefill repro (2-turn piped stdin) — sidecar bank replay regressions
  historically show up there.
- `tune_profile.sh` backend sweep with proper cooldown, no concurrent agents (M5 Max
  thermal trap).
- Perplexity / logprob sanity vs the FP8/FP4 source model on a fixed prompt set.

## MPP 4.1 readiness scaffolding (do at this stage)

Context: macOS 27 / MSL 4.1 adds native FP4+E8M0 scale-plane `matmul2d` (~2.6× faster than
dequant-to-half on M5). Porting guide: `/Users/anemll/Downloads/MXFP4-MPP41-GUIDE.md`
(MetalFP41Probe-validated). The scaffolding below makes the future MPP 4.1 branch a
probe + one kernel + one dispatch, instead of a layout migration.

**The core incompatibility.** The sidecar stores ggml `block_mxfp4`: 17-byte interleaved
blocks (scale byte inside each block) with split-half nibble order (`qs[j]` low nibble =
elem j, high = elem j+16). MPP 4.1's native path requires two separate contiguous planes —
FP4 data `[n][k/2]` with *sequential-pair* nibble order (low nibble = even k) and an E8M0
scale plane `[n][k/32]` — wrapped as a multi-plane `MTLTensor`. Same total bytes, different
arrangement. So native consumption needs a one-time repack; everything else is shared.

1. **Shared decode header** `metal/mxfp4_common.h`, used by every MXFP4 kernel we write:
   - E2M1 LUT (`{0,.5,1,1.5,2,3,4,6}` + sign nibble-bit), and
   - E8M0 decode as the exact hardware bit-op: `as_type<float>(uint(e) << 23)` — NOT
     `exp2()`. Bit-identical to the M5 native decode (incl. e=0 → 0.0), so fallback and
     native paths can never numerically diverge.
   - nibble-extract helpers for BOTH orders (ggml split-half, MPP sequential-pair), named
     explicitly so each kernel states which layout it reads.
   - adopt the guide's feature-detect pattern now:
     `__has_include(<MetalPerformancePrimitives/...>)` + `__HAVE_TENSOR_MULTIPLANE__`
     → `MXFP4_HAS_NATIVE_SCALE_PLANE`. The guarded native section stays empty on
     macOS 26; one source file serves both OS versions.
2. **Plane-split resident bank layout for the Phase 2 fast paths.** Phase 1 baseline
   (mul_mm_id) reads ggml blocks straight from the streamed bank — keep that. But when
   experts are uploaded into *resident* fast-path banks, repack with a one-time GPU kernel
   `kernel_dsv4_mxfp4_repack_planes` (17B interleaved, split-half → data plane with
   sequential nibbles + scale plane). Then:
   - the macOS 26 dequant-to-half kernel (guide's fallback arm → our validated nax_half
     h_h_f recipe) reads these planes, and
   - on macOS 27 the **same two buffers** wrap into the multi-plane MTLTensor with zero
     repacking (`dims {k,n}`, `strides {1,k}`, scale blockFactors `{32,1}`).
   - Alignment checks out: data rows k/2 = 2048 B (gate/up) / 1024 B (down), both meet the
     128-byte stride rule (guide gotcha #1).
3. **Two-pointer weight views.** All *new* `dsv4_*` MXFP4 kernels take
   `(device const uchar *fp4_data, device const uchar *scales)` instead of one block
   pointer — the exact signature of the guide's raw-pointer native kernel
   (`mxfp4_scaled_matmul_n64`), so host-side binding code is shared verbatim. Carry
   (data_off, scales_off) per family in the bank metadata.
4. **Reserve the backend name now.** Add `mxfp4_native` to the backend-name tables
   (env remap, profile strings, stats labels) plus a runtime stub
   `ds4_gpu_has_native_mxfp4()` → false on macOS 26 (later: `@available(macOS 27)` +
   definitive tensor-creation probe per the guide). tune_profile.sh JSON profiles and
   `DS4_RESIDENT_MOE_BACKEND` values stay forward-compatible.
5. **Golden vectors, layout-parameterized.** Python reference in `tests/` emitting, for
   one random block: ggml 17B block, plane-split pair, expected f16 values. Covers both
   nibble orders + E8M0 edges (e=0, 127, 254). Both branches validate against the same
   files.
6. **Anti-scaffolding (measured dead ends, don't build):** manual in-register scaling
   matmul variants — every variant in the probe was slower (1.5–1.6 ms) than plain
   dequant-to-half + h×h matmul2d (0.88 ms). The dequant→half arm IS the right macOS 26
   shape and it's already our validated recipe. W4A8 is 2× faster than even native MXFP4,
   but E2M1 values don't map losslessly onto the int4 lattice (×2 overflows at ±12) —
   would require requantization; parked.

## Notes / risks

- Nibble order and E8M0 zero/flush semantics must match the converter (ggml) exactly —
  validate with a single-block unit check against a Python reference before wiring kernels.
- Verify the sidecar bins really are raw ggml `block_mxfp4` bytes (the manifest's
  `source_offset`/`exact_byte_length` are contiguous GGUF tensor slices, so they should
  be) — one hexdump of block 0 against the Python reference settles it.
- The local copy is complete: 145G at `/Users/anemll/Models/DSv4-Flash-MXFP4-native-flash`
  (43 layer bins + manifest + `dense/model-dense.gguf` + `flashmoe-package.json`).
- The per-type helper at ds4_metal.m:~11580 returns a per-type magic number (2 for Q4_K,
  4 for Q2_K/IQ2_XXS) — understand what it scales before picking MXFP4's value.
- `_counted` transpose-i8 variants exist only for iq2_xxs/q2_k today; decide whether the
  dedup path needs the mxfp4 `_counted` twin in Phase 2.
