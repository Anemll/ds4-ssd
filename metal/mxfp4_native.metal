// DS4 native-MXFP4 (MPP 4.1 scale-plane) kernels.
//
// Compiled as a separate runtime library (MTLLanguageVersion4_1 on macOS 27,
// falling back to 4_0/no-native on macOS 26) by
// ds4_gpu_ensure_mxfp4_native_library() in ds4_metal.m, which PREPENDS
// metal/mxfp4_common.h to this source (newLibraryWithSource: cannot resolve
// local includes).  Offline syntax check:
//   cat metal/mxfp4_common.h metal/mxfp4_native.metal | \
//     xcrun -sdk macosx metal -std=metal4.1 -x metal -c - -o /dev/null
//
// Contents:
//   kernel_dsv4_mxfp4_repack_planes   - any MSL version.  One-time repack of
//       ggml block_mxfp4 (17 B interleaved, split-half nibbles) into the
//       plane-split layout the MPP 4.1 native path consumes: FP4 data plane
//       [rows][k/2] with sequential-pair nibbles + E8M0 scale plane
//       [rows][k/32].  Same total bytes; pure permutation, bit-exact.
//   kernel_dsv4_mxfp4_native_matmul_n64 - macOS 27 / M5 only
//       (DS4_MXFP4_HAS_NATIVE_SCALE_PLANE).  matmul2d consumes the FP4 data
//       plane and the E8M0 scale plane together, natively; ~2.6x the
//       dequant-to-half h_h_f rate measured on M5 (see MXFP4-MPP41-GUIDE).
//
// Layout/perf decisions baked in (measured in the MetalFP41Probe project, do
// not re-derive): NT=64 output tile (128 was ~2x slower), scaled right
// operand REQUIRES transpose_right=true, scale block sizes (32,1), float
// destination (~3% cost vs half, no inf overflow), no manual-scaling fused
// variants (all measured slower than dequant-to-half).

// ---------------------------------------------------------------------------
// Repack: ggml split-half blocks -> sequential-pair data plane + scale plane.
//
// One thread per 32-element block.  Grid:
//   gid.x = k-block index within the row   (k / 32)
//   gid.y = output row                     (rows)
//   gid.z = expert index                   (n_experts; strides below)
// Buffers may alias regions of one bank buffer via offsets; the host binds
// (data_off, scales_off) per expert family — the two-pointer weight view of
// docs/mxfp4-native-sidecar-plan.md item 3.
// ---------------------------------------------------------------------------
kernel void kernel_dsv4_mxfp4_repack_planes(
        device const uchar *src          [[buffer(0)]],  // ggml blocks, 17 B each
        device uchar       *dst_data     [[buffer(1)]],  // [rows][k/2] seq-pair
        device uchar       *dst_scales   [[buffer(2)]],  // [rows][k/32] E8M0
        constant uint      &rows         [[buffer(3)]],
        constant uint      &depth        [[buffer(4)]],  // k, multiple of 32
        constant uint      &src_estride  [[buffer(5)]],  // bytes between experts in src
        constant uint      &data_estride [[buffer(6)]],  // bytes between experts in dst_data
        constant uint      &scale_estride[[buffer(7)]],  // bytes between experts in dst_scales
        uint3 gid [[thread_position_in_grid]])
{
    const uint kb  = gid.x;
    const uint row = gid.y;
    const uint kBlocks = depth / DS4_MXFP4_QK;
    if (kb >= kBlocks || row >= rows) return;

    device const ds4mx_block *blk = (device const ds4mx_block *)
        (src + gid.z * src_estride) + row * kBlocks + kb;
    device uchar *data = dst_data + gid.z * data_estride
                       + row * (depth / 2u) + kb * 16u;
    device const uchar *qs = blk->qs;

    // split-half -> sequential-pair: out byte i holds elements 2i (lo) and
    // 2i+1 (hi).  i 0..7 reads two low nibbles, i 8..15 two high nibbles.
    for (uint i = 0; i < 8u; i++) {
        data[i] = (qs[2u*i] & 0x0Fu) | ((qs[2u*i + 1u] & 0x0Fu) << 4u);
    }
    for (uint i = 8u; i < 16u; i++) {
        const uint j = 2u*i - 16u;
        data[i] = (qs[j] >> 4u) | (qs[j + 1u] & 0xF0u);
    }

    (dst_scales + gid.z * scale_estride)[row * kBlocks + kb] = blk->e;
}

// ---------------------------------------------------------------------------
// MPP 4.0 fallback arm + benchmark references (compile on macOS 26 too).
//
// kernel_dsv4_mxfp4_dequant_planes_to_half: LUT-dequant the plane-split
// layout to half [rows][k] (k contiguous).  One thread per packed byte.
// Dispatch: grid (k/2, rows).  This is the macOS 26 arm of the same code:
// planes prepared once, then a plain half x half matmul2d.
// ---------------------------------------------------------------------------
kernel void kernel_dsv4_mxfp4_dequant_planes_to_half(
        device const uchar *data    [[buffer(0)]],  // [rows][k/2] seq-pair
        device const uchar *scales  [[buffer(1)]],  // [rows][k/32] E8M0
        device half        *out     [[buffer(2)]],  // [rows][k]
        constant uint      &rows    [[buffer(3)]],
        constant uint      &depth   [[buffer(4)]],  // k
        uint2 gid [[thread_position_in_grid]])
{
    const uint byteIndex = gid.x;
    const uint row = gid.y;
    const uint bytesPerRow = depth / 2u;
    if (row >= rows || byteIndex >= bytesPerRow) return;

    const uint k = byteIndex * 2u;
    const uint kBlocks = depth / DS4_MXFP4_QK;
    const float scale =
        ds4mx_e8m0_to_float(scales[row * kBlocks + k / DS4_MXFP4_QK]);
    const uchar bits = data[row * bytesPerRow + byteIndex];
    out[row * depth + k] = half(float(ds4mx_e2m1_lut_f16[bits & 0x0Fu]) * scale);
    out[row * depth + k + 1u] = half(float(ds4mx_e2m1_lut_f16[bits >> 4u]) * scale);
}

#if defined(DS4_MXFP4_HAS_MPP)
// half x half matmul2d, same n64 tiling as the native kernel — the MPP 4.0
// matmul the dequant-to-half arm feeds, kept here so the A/B benchmark runs
// all arms from one library.
namespace ds4mx40 {

using namespace metal;
using namespace mpp::tensor_ops;

template <int NT>
inline void h_h_matmul_tile(device half *a, device half *w, device float *c,
                            uint m, uint n, uint k, uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT,
                                              static_cast<int>(dynamic_extent),
                                              false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor((BTensor::data_handle_type)w,
                      dextents<int32_t, 2>{int32_t(k), int32_t(n)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});
    auto mA = tA.slice(0, int(tgid.y) * 64);
    auto mB = tB.slice(0, int(tgid.x) * NT);
    auto mC = tC.slice(int(tgid.x) * NT, int(tgid.y) * 64);
    op.run(mA, mB, mC);
}

} // namespace ds4mx40

kernel void kernel_dsv4_mxfp4_matmul_h_h_n64(
        device half   *a [[buffer(0)]],
        device half   *w [[buffer(1)]],
        device float  *c [[buffer(2)]],
        constant uint &m [[buffer(3)]],
        constant uint &n [[buffer(4)]],
        constant uint &k [[buffer(5)]],
        uint2 tgid [[threadgroup_position_in_grid]])
{
    ds4mx40::h_h_matmul_tile<64>(a, w, c, m, n, k, tgid);
}
#endif // DS4_MXFP4_HAS_MPP

// ---------------------------------------------------------------------------
// Native scale-plane matmul (raw-pointer binding; identical perf to the
// bindless MTLTensor-handle variant, and needs no host tensor API).
//
//   a       half  [m][k], k contiguous          buffer(0)
//   weights uchar [n][k/2] FP4 seq-pair         buffer(1)
//   c       float [m][n], n contiguous          buffer(2)
//   m,n,k   uint                                buffer(3,4,5)
//   scales  uchar [n][k/32] E8M0                buffer(6)
//
// Dispatch: threadgroups ((n+63)/64, (m+63)/64, 1),
//           threadsPerThreadgroup (threadExecutionWidth*4, 1, 1).
// k must be a multiple of 32; row stride k/2 must be 128-byte aligned
// (k >= 256 and k % 256 == 0 satisfies both; gate/up k=4096, down k=2048 ok).
// ---------------------------------------------------------------------------
#if defined(DS4_MXFP4_HAS_NATIVE_SCALE_PLANE)

namespace ds4mx {

using namespace metal;
using namespace mpp::tensor_ops;

using scales_tag = tensor_blockwise<tensor_plane_scales,
                                    device metal_fp8_ue8m0_format, 32, 1>;

template <int NT>
inline void scaled_matmul_tile(device half *a,
                               device uchar *weights,
                               device const uchar *scales,
                               device float *c,
                               uint m, uint n, uint k,
                               uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT,
                                              static_cast<int>(dynamic_extent),
                                              /*transpose_left*/  false,
                                              /*transpose_right*/ true,
                                              /*relaxed_precision*/ false);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>,
                           tensor_inline, scales_tag>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    scales_tag plane(reinterpret_cast<scales_tag::data_handle_type>(
        const_cast<device uchar *>(scales)));
    auto tB = BTensor((BTensor::data_handle_type)weights,
                      dextents<int32_t, 2>{int32_t(k), int32_t(n)},
                      array<int32_t, 2>{1, int32_t(k)},
                      plane);
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, int(tgid.y) * 64);
    auto mB = tB.slice(0, int(tgid.x) * NT);
    auto mC = tC.slice(int(tgid.x) * NT, int(tgid.y) * 64);
    op.run(mA, mB, mC);
}

} // namespace ds4mx

kernel void kernel_dsv4_mxfp4_native_matmul_n64(
        device half        *a       [[buffer(0)]],
        device uchar       *weights [[buffer(1)]],
        device float       *c       [[buffer(2)]],
        constant uint      &m       [[buffer(3)]],
        constant uint      &n       [[buffer(4)]],
        constant uint      &k       [[buffer(5)]],
        device const uchar *scales  [[buffer(6)]],
        uint2 tgid [[threadgroup_position_in_grid]])
{
    ds4mx::scaled_matmul_tile<64>(a, weights, scales, c, m, n, k, tgid);
}

// Benchmark reference: raw FP4 matmul WITHOUT the scale plane.  Not
// numerically usable for MXFP4 weights (drops the per-32-block scales) —
// it exists to measure what the scale plane costs on a given GPU.
namespace ds4mx {

template <int NT>
inline void raw_fp4_matmul_tile(device half *a, device uchar *weights,
                                device float *c, uint m, uint n, uint k,
                                uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT,
                                              static_cast<int>(dynamic_extent),
                                              false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>,
                           tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor((BTensor::data_handle_type)weights,
                      dextents<int32_t, 2>{int32_t(k), int32_t(n)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});
    auto mA = tA.slice(0, int(tgid.y) * 64);
    auto mB = tB.slice(0, int(tgid.x) * NT);
    auto mC = tC.slice(int(tgid.x) * NT, int(tgid.y) * 64);
    op.run(mA, mB, mC);
}

} // namespace ds4mx

kernel void kernel_dsv4_mxfp4_native_matmul_noscale_n64(
        device half   *a       [[buffer(0)]],
        device uchar  *weights [[buffer(1)]],
        device float  *c       [[buffer(2)]],
        constant uint &m       [[buffer(3)]],
        constant uint &n       [[buffer(4)]],
        constant uint &k       [[buffer(5)]],
        uint2 tgid [[threadgroup_position_in_grid]])
{
    ds4mx::raw_fp4_matmul_tile<64>(a, weights, c, m, n, k, tgid);
}

#endif // DS4_MXFP4_HAS_NATIVE_SCALE_PLANE
