# Native MXFP4 sidecar decoding — implementation plan

> **Status (2026-06-11):** Phases 0–2 implemented and validated on branch `MXFP4`
> (commits 33d951a, +ANE follow-up). MXFP4 experts decode correctly on the
> baseline mul_mm/mul_mv path AND the ANE i8i8 (W8A8) prefill backend.
> Measured vs Q4K (slot-bank 48, M5 Max): 16k prefill 313.0 vs 312.9 t/s
> (parity), decode 4.56 vs 4.47 t/s, identical short-prompt outputs.
> The native package's own dense GGUF (block-128 FP8 type-42 = {e8m0 scale;
> e4m3fn qs[128]}, attn_kv_latent schema) is handled by an offline converter:
> `fp4_samples/convert_native_dense_to_ds4.py` rebuilds it into the chat-v2
> schema → `dense/model-dense-ds4.gguf`. Fully-native package measured BEST of
> all configs: 16k prefill 315.9 t/s, decode 4.59 t/s (vs Q4K 312.9/4.47).
> The converted GGUF now sits at the canonical `dense/model-dense.gguf` (the
> original FP8 export is preserved as `dense/model-dense-fp8-native.gguf`),
> so the package runs plainly: `./ds4 -m <pkg> --moe-mode slot-bank ...`.
> ANE i8i8 confirmed optimal vs GPU mul_mm_id A/B (315.9 vs 307.9 t/s 16k).
> Phase 3 validation complete: 16k smoke, ANE-vs-GPU backend A/B, graded QA
> (2/2 = Q4K), and the 2-turn resume-after-decode repro (clean cross-turn
> context, no bank-replay corruption), and a ds4-server OpenAI-API smoke
> (exact instruction following). Bonus: high-slot decode cliff
> root-caused (page-cache squeeze) and fixed via DS4_SSD_CACHE_AUTO_PCT
> (default 20). Remaining: MPP 4.1 scaffolding items below (macOS 27).

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

## Dense precision audit vs the original HF model (2026-06-11)

HF original (`/Volumes/TB36/Models/DS/DeepSeek-V4-Flash`) is natively FP8:
F8_E4M3 weights + F8_E8M0 scales on 128x128 tiles. Three-way comparison
(relRMS / max-abs) on representative tensors:

| tensor | HF -> native GGUF | native -> converted | total |
|---|---|---|---|
| L0 wq_a / wq_b / wkv / wo_b | 0 / 0 (bit-exact) | 5.5e-3 / ~6e-4 | 5.5e-3 |
| L0 shared experts w1 / w2 | 0 / 0 | 5.5e-3 / ~7e-4 | 5.5e-3 |
| L20 wq_b (mid-network) | 0 / 0 | 5.5e-3 / 8.9e-4 | 5.5e-3 |
| embed (BF16 -> F16) | 0 / 0 | 1.3e-9 | 1.3e-9 |
| head (BF16 -> Q8_0) | 0 / 0 | 5.4e-3 / 1.6e-2 | 5.4e-3 |

- The native GGUF dense is a bit-exact repack of the HF FP8 weights (the
  128x128 tile scale broadcasts losslessly onto row-128 segments).
- The only loss in the whole chain is the converter's FP8->Q8_0 re-encode:
  ~0.55% relRMS, matching Q8_0 roundoff theory, and well below the FP8
  grid's own ~2-3% representation step relative to the pre-FP8 master
  weights. Embeddings are exact (BF16->F16).
- A zero-loss alternative exists (store dense as F16, ~16 GB vs 8.8 GB),
  but Q8_0 is what the optimized W8A8 NAX dense path consumes.

### HF dtype census + expert verification

The HF original is NOT uniformly FP8 — four tiers:
1. FP8 (E4M3, E8M0 128x128 scales): attention projections, indexer wq_b,
   shared experts, MTP projections.
2. Native MXFP4 (E2M1 packed 2/byte declared "I8", E8M0 block-32 scales):
   ALL routed experts — the model is FP4-native in the experts, exactly the
   sidecar format. Verified bit-exact: HF expert0 w1 == sidecar layer0 gate
   values (HF packs nibbles sequential-pair; ggml split-half; values equal).
3. BF16: embeddings, LM head, norms, router gate, compressor/indexer mats.
4. F32: hyper-connection params, attn sinks, router bias, APE tables.

End-to-end: every FP4/FP8 byte in the running package is bit-exact with the
HF release; total pipeline loss is only the dense Q8_0 re-encode (0.55%
relRMS) and exact BF16->F16 conversions.

## Native MXFP4 (MPP 4.1) — phase 4.1a IMPLEMENTED (2026-06-11, branch `mxfp4-MPP4.1`)

Scaffolding items 1, 3, 4, 5 above are done and validated on an M5 (32 GiB)
running macOS 27.0 with the 4.1 Metal toolchain (`__HAVE_TENSOR_MULTIPLANE__`
confirmed at `-std=metal4.1`, absent at 3.2). Everything is **gated and
default-off**: with `DS4_MXFP4_NATIVE` unset, no new library is even compiled.

What landed:

- `metal/mxfp4_common.h` — shared decode header: E2M1 LUT, E8M0 bit-decode
  (`as_type<float>(uint(e)<<23)`), explicit extract helpers for BOTH nibble
  orders (`ds4mx_nibble_splithalf` = ggml, `ds4mx_nibble_seqpair` = MPP
  native), feature macro `DS4_MXFP4_HAS_NATIVE_SCALE_PLANE`. The host
  PREPENDS this file to the library source (runtime `newLibraryWithSource:`
  cannot resolve local includes); offline check:
  `cat metal/mxfp4_common.h metal/mxfp4_native.metal | xcrun -sdk macosx metal -std=metal4.1 -x metal -c - -o /dev/null`
- `metal/mxfp4_native.metal` —
  - `kernel_dsv4_mxfp4_repack_planes` (any MSL): one-time permutation of ggml
    17 B split-half blocks -> seq-pair FP4 data plane + E8M0 scale plane.
    Same total bytes. Multi-expert via gid.z + per-expert strides; two-pointer
    plane views per plan item 3.
  - `kernel_dsv4_mxfp4_repack_selected_planes` (any MSL): per-route repack from
    GPU-resident selected slot IDs, so decode can consume slot banks without CPU
    selected-ID readback.
  - `kernel_dsv4_mxfp4_native_matmul_n64` (4.1 only): raw-pointer scale-plane
    matmul2d, NT=64, transpose_right, float dst — exactly the guide's
    validated recipe.
- Host (`ds4_metal.m`): `ds4_gpu_ensure_mxfp4_native_library()` builds a
  separate `MTLLanguageVersion4_1` library (4_0 fallback = repack-only);
  `ds4_gpu_has_native_mxfp4()` definitive pipeline probe;
  `ds4_gpu_mxfp4_native_requested()` = `DS4_MXFP4_NATIVE=1` opt-in; startup
  log line when the gate is set. Exported in `ds4_gpu.h`.
- Tests: `make mxfp4-native-probe` (standalone, no model needed) — GPU repack
  byte-exact vs CPU reference over 8192 blocks incl. e=0/127 edges; selected
  slot repack byte-exact; native matmul relRMS **1.24e-07** vs CPU
  dequant+GEMM; tail probe; bench mode (`MXFP4_PROBE_BENCH=1`).
  `tests/gen_mxfp4_golden.py` emits
  layout-parameterized golden vectors (both nibble orders, E8M0 edges incl.
  the e=254 f32-overflow-to-inf case) to `tests/test-vectors/mxfp4/`.

Measured on this M5 (single dispatch, m=128):

| shape | ms | TOPS |
|---|---:|---:|
| gate/up n=2048 k=4096 | 0.426 | 5.0 |
| down n=4096 k=2048 | 0.193 | 11.1 |

The gate/up shape under-fills the GPU at a single-expert dispatch (only 32
threadgroups in x); per-expert batching (gid.z) or concurrent expert
dispatches recover occupancy — the down shape already hits the ~11 TOPS
unscaled-FP4 class from the guide.

**Empirical findings that simplify integration (probe-verified):**
- MPP matmul2d clamps to tensor extents: m NOT a multiple of 64 produces
  correct results and writes zero bytes past `m` rows. No padding logic
  needed in the dispatch path.
- The smoke `DS4_MXFP4_NATIVE=1 ./ds4 -m <pkg> ...` prints
  `MXFP4 native library (MSL 4.1): repack=ok selected-repack=ok scale-plane matmul=ok`.

## Phase 4.1b — integration plan: SSD-streaming-optimized consumption

Goal: optimize streamed-MXFP4 inference vs the current version. The streamed
bytes are already optimal (17 B per 32 elems, bit-exact manifest reads — do
NOT change the sidecar format); the wins are in what happens after the pread.

Two integration arms, in order:

**Arm A — per-use staging replacement (drop-in, do first).** Today every
prefill chunk runs `kernel_dsv4_mpp_dequant_mxfp4_transpose_i8` (NAX int8) or
`ane_dequant_mxfp4_transpose_f16` staging passes: k*n bytes (i8) or 2*k*n
(f16) of staging writes per expert family per chunk. With the native path:
`repack_planes` stages only k*n*17/32 bytes (~0.53x i8, ~0.27x f16) and
`native_matmul_n64` consumes FP4 directly — or, for already-installed slots,
skip staging entirely (Arm B). Wire as backend name `mxfp4_native` in the
resident/prefill backend tables, selected only when
`ds4_gpu_mxfp4_native_requested() && ds4_gpu_has_native_mxfp4()` and the
routed type is MXFP4. Validate numerics vs mul_mm_id output, then A/B
against ANE i8i8 (315.9 t/s on M5 Max) — note W8A8 has ~2x the MAC rate, so
the native arm may win only where the dequant pass dominates (short chunks,
low expert reuse) or on memory-squeezed machines (see below).

**Arm B — repack-at-install (the streaming optimization end state).** Repack
each expert family region ONCE when a slot is installed (GPU dispatch right
after the pread completes, overlapped with the next slot's IO), keeping the
bank in plane-split layout: same byte count, same family offsets — data plane
then scale plane within each family region. Then:
- prefill matmuls read the bank in place: zero per-chunk staging, zero
  dequant passes;
- the i8/f16 staging buffers (`g_mpp_prefill_{gate,up,down}_{i8,f16}_buffer`)
  shrink or disappear — on RAM-limited machines that RAM goes back to the OS
  file cache, which is exactly the currency of the decode cliff
  (docs/mxfp4-handoff.md): every freed GiB is page-cache for decode-miss
  preads;
- decode needs a plane-reading `mul_mv` twin (same math as
  `kernel_mul_mv_mxfp4_f32_impl`, seq-pair + scale-plane reads instead of
  17 B blocks) since the bank no longer holds ggml blocks when native is on;
  the fused pair/pair_swiglu decode variants (handoff item 4) should be
  written against the plane layout directly.
- CPU-side bank consumers (validation harness, any CPU fallback) must either
  be disabled under the gate or taught the plane layout.

Cost accounting for Arm B: repack is one extra read+write of each expert's
bytes per INSTALL (vs per-use dequant today). At ~600 MB/s sustained SSD
install rate, a 12.75 MiB expert record repacks in ~50 us of GPU time —
invisible next to the ~20 ms pread it follows.

Not doing (measured dead ends, see guide + plan above): manual in-register
scaling variants, W4A8 requantization of E2M1, host multi-plane MTLTensor
binding (raw-pointer binding is identical perf and needs no tensor API).

### Arm A landed + measured (2026-06-11, base M5 32 GiB, macOS 27.0)

Arm A is wired: `ds4_gpu_routed_moe_expert_banked_batch_mpp_int8_tensor` now
accepts MXFP4/MXFP4 when `DS4_MXFP4_NATIVE=1` && probe true, with a
`use_fp4_native` arm (repack into the i8 scratch buffers — data plane at 0,
scales at rows*k/2 — then three native matmuls + the shared swiglu, swiglu
scale 1.0 since FP4 decode is exact). Activation chain for A/B:
`DS4_FLASH_MOE_ANE_PREFILL=0 DS4_FLASH_MOE_MPP_INT8_PREFILL=1
DS4_MXFP4_NATIVE=1`. Partial 64-row tiles are allowed automatically for the
native arm (validated down to m=1, full reference checks + poisoned padding;
the split-with-ALU-tail workaround remains for the legacy i8/h_h kernels,
whose historical partial-tile miscalculation is NOT shared by matmul2d, which
clamps to tensor extents). DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES is no longer
needed and should NOT be set globally (it re-exposes the legacy bug on
	iq2/q2k packages). Engage log:
	`ds4: [mxfp4-native] MPP 4.1 scale-plane prefill arm engaged`.

### Decode per-use native arm landed (2026-06-15)

`DS4_MXFP4_NATIVE=1` now also covers MXFP4/MXFP4 decode paths when the MPP 4.1
probe passes. This is still Arm A style repack-per-use: ggml 17 B blocks remain
the resident/slot-bank format, each selected route repacks gate/up/down into the
shared scratch planes, then runs three native scale-plane matmuls plus the
existing weighted SwiGLU/sum.

Covered entry points:

- `ds4_gpu_routed_moe_one_banked_tensor`
- `ds4_gpu_routed_moe_one_slots6_tensor`
- `ds4_gpu_routed_moe_one_slots6_record_tensor`
- `ds4_gpu_routed_moe_one_slots6_chunked_tensor`
- `ds4_gpu_routed_moe_one_banked_tensor_slotwise_impl` (regular and baked)

`kernel_dsv4_mxfp4_repack_selected_planes` handles GPU-selected slot IDs for the
banked/slotwise paths. Direct slots6/record/chunked paths pass explicit family
buffers and offsets. The argument-buffer record-table decode path is still the
legacy LUT path; native decode is not wired through that indirection.

Engage log:
`ds4: [mxfp4-native] MPP 4.1 scale-plane decode arm engaged (<path>, repack-per-use)`.
Arm B/repack-at-install is still future work: once installed banks are stored in
plane-split layout, decode must read the plane layout directly instead of
running the per-use repack.

Kernel-level 4-arm benchmark (`MXFP4_PROBE_BENCH=1 ./tests/mxfp4_native_probe`,
m=128, 50 iters, base M5 — NOT the guide's M5 Max):

| arm | gate/up n=2048 k=4096 | down n=4096 k=2048 |
|---|---|---|
| native FP4+scale plane (MPP 4.1) | 0.328 ms / 6.5 TOPS | 0.192 ms / 11.2 TOPS |
| raw FP4, no block scaling (4.1 ref) | 0.155 ms / 13.8 TOPS | 0.145 ms / 14.8 TOPS |
| h x h, resident f16 weights (4.0) | 0.228 ms / 9.4 TOPS | 0.208 ms / 10.3 TOPS |
| dequant->half + h x h per-use (4.0) | 0.354 ms / 6.1 TOPS | 0.350 ms / 6.1 TOPS |

Reading on this GPU (base M5, 4.1 toolchain): the scale plane is NOT ~free
here (unlike the guide's M5 Max measurement) — it costs 2.1x at the gate/up
shape and 1.3x at down; occupancy at n=2048 amplifies it. Native still beats
the honest per-use MPP 4.0 arm (dequant+h_h) at both shapes (1.1x / 1.8x) and
beats even resident-f16 h_h at the down shape while using 3.8x less weight
memory (4.25 vs 16 bits/elem staged) — the memory delta is the streaming win
(page cache currency). End-to-end on this 32 GiB box all prefill backends are
SSD-IO-bound (~16 t/s at 280-token prompts), so backend deltas only show on
memory-rich machines (M5 Max/M3U re-bench pending).
