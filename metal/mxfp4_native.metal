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
//   kernel_dsv4_mxfp4_plane_id_pair_swiglu_f32 /
//   kernel_dsv4_mxfp4_plane_id_sum6_f32 - decode-shaped plane-layout matvec
//       kernels for DSv4 top-6 routed MoE.  They avoid the native MPP matmul2d
//       64-row tile floor that dominates single-token decode.
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

kernel void kernel_dsv4_mxfp4_repack_selected_planes(
        device const uchar *src         [[buffer(0)]],  // slot bank, ggml blocks
        device uchar       *dst_data    [[buffer(1)]],  // [rows][k/2] seq-pair
        device uchar       *dst_scales  [[buffer(2)]],  // [rows][k/32] E8M0
        device const int   *selected    [[buffer(3)]],  // slot id per route
        constant uint      &rows        [[buffer(4)]],
        constant uint      &depth       [[buffer(5)]],  // k, multiple of 32
        constant uint      &slot_stride [[buffer(6)]],  // bytes between slots
        constant uint      &route       [[buffer(7)]],
        uint2 gid [[thread_position_in_grid]])
{
    const uint kb  = gid.x;
    const uint row = gid.y;
    const uint kBlocks = depth / DS4_MXFP4_QK;
    if (kb >= kBlocks || row >= rows) return;

    const int slot_i = selected[route];
    if (slot_i < 0) return;
    const uint slot = (uint)slot_i;
    device const uchar *expert = src + slot * slot_stride;
    device const ds4mx_block *blk = (device const ds4mx_block *)expert
                                  + row * kBlocks + kb;
    device uchar *data = dst_data + row * (depth / 2u) + kb * 16u;
    device const uchar *qs = blk->qs;

    for (uint i = 0; i < 8u; i++) {
        data[i] = (qs[2u*i] & 0x0Fu) | ((qs[2u*i + 1u] & 0x0Fu) << 4u);
    }
    for (uint i = 8u; i < 16u; i++) {
        const uint j = 2u*i - 16u;
        data[i] = (qs[j] >> 4u) | (qs[j + 1u] & 0xF0u);
    }

    dst_scales[row * kBlocks + kb] = blk->e;
}

kernel void kernel_dsv4_mxfp4_copy_selected_planes(
        device const uchar *src_data    [[buffer(0)]],  // slot bank data planes
        device const uchar *src_scales  [[buffer(1)]],  // slot bank scale planes
        device uchar       *dst_data    [[buffer(2)]],  // [rows][k/2] seq-pair
        device uchar       *dst_scales  [[buffer(3)]],  // [rows][k/32] E8M0
        device const int   *selected    [[buffer(4)]],  // slot id per route
        constant uint      &rows        [[buffer(5)]],
        constant uint      &depth       [[buffer(6)]],  // k, multiple of 32
        constant uint      &slot_stride [[buffer(7)]],  // bytes between slots
        constant uint      &route       [[buffer(8)]],
        uint gid [[thread_position_in_grid]])
{
    const uint data_bytes = rows * (depth / 2u);
    const uint scale_bytes = rows * (depth / DS4_MXFP4_QK);
    const uint total_bytes = data_bytes + scale_bytes;
    if (gid >= total_bytes) return;

    const int slot_i = selected[route];
    if (slot_i < 0) return;
    const uint slot = (uint)slot_i;
    device const uchar *expert_data = src_data + slot * slot_stride;
    device const uchar *expert_scales = src_scales + slot * slot_stride;

    if (gid < data_bytes) {
        dst_data[gid] = expert_data[gid];
    } else {
        const uint scale_gid = gid - data_bytes;
        dst_scales[scale_gid] = expert_scales[scale_gid];
    }
}

// ---------------------------------------------------------------------------
// Decode-shaped plane-layout MXFP4 matvecs.
//
// These read the same plane-split layout as the MPP 4.1 native matmul arm, but
// use a classic decode matvec shape: 2 rows per simdgroup, 2 simdgroups per
// threadgroup.  The route dimension is fixed to DSv4's active top-6 experts.
// Host gates these kernels on DS4_MXFP4_NATIVE=1, plane-split storage, and
// n_expert == 6.
// ---------------------------------------------------------------------------

// Dense plane-split MXFP4 matvec for DSpark draft records converted from FP8
// (scripts/dspark_convert_fp8_to_mxfp4.py). Same dispatch shape and reduction
// order as kernel_mul_mv_fp8_e4m3_f32_rows5 (metal/dense.metal): one 32-thread
// threadgroup per output row, up to 5 token columns.
typedef struct {
    uint in_dim;
    uint out_dim;
    uint scale_cols;   // in_dim / 32 (one E8M0 per 32-block per row)
    uint n_tokens;     // 1..5
} ds4mx_plane_matmul_args;

kernel void kernel_dsv4_mxfp4_plane_matmul_rows5_f32(
        constant ds4mx_plane_matmul_args & args [[buffer(0)]],
        device const uchar *data    [[buffer(1)]],
        device const uchar *scales  [[buffer(2)]],
        device const float *x       [[buffer(3)]],
        device       float *dst     [[buffer(4)]],
        uint   row   [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]])
{
    if (row >= args.out_dim) {
        return;
    }
    device const uchar *drow = data + (ulong)row * (ulong)(args.in_dim / 2u);
    device const uchar *srow = scales + (ulong)row * (ulong)args.scale_cols;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    for (uint k = tiisg; k < args.in_dim; k += 32u) {
        const float scale = ds4mx_e8m0_to_float(srow[k >> 5u]);
        const float w = ds4mx_e2m1_lut_f32[ds4mx_nibble_seqpair(drow, k)] * scale;
        sum0 += x[k] * w;
        if (args.n_tokens > 1u) {
            sum1 += x[(ulong)args.in_dim + k] * w;
        }
        if (args.n_tokens > 2u) {
            sum2 += x[2ul * (ulong)args.in_dim + k] * w;
        }
        if (args.n_tokens > 3u) {
            sum3 += x[3ul * (ulong)args.in_dim + k] * w;
        }
        if (args.n_tokens > 4u) {
            sum4 += x[4ul * (ulong)args.in_dim + k] * w;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2);
    sum3 = simd_sum(sum3);
    sum4 = simd_sum(sum4);
    if (tiisg == 0) {
        dst[row] = sum0;
        if (args.n_tokens > 1u) {
            dst[(ulong)args.out_dim + row] = sum1;
        }
        if (args.n_tokens > 2u) {
            dst[2ul * (ulong)args.out_dim + row] = sum2;
        }
        if (args.n_tokens > 3u) {
            dst[3ul * (ulong)args.out_dim + row] = sum3;
        }
        if (args.n_tokens > 4u) {
            dst[4ul * (ulong)args.out_dim + row] = sum4;
        }
    }
}

inline void ds4mx_plane_dot2_accum(
        device const uchar *data,
        device const uchar *scales,
        device const float *x,
        uint rows,
        uint depth,
        uint row0,
        ushort tiisg,
        thread float *sumf)
{
    const uint kBlocks = depth / DS4_MXFP4_QK;
    const uint bytesPerRow = depth / 2u;
    const ushort ix = tiisg / 2u;
    const ushort it = tiisg & 1u;
    const uint elemBase = uint(it) * 16u;
    const uint byteBase = uint(it) * 8u;

    for (uint ib = uint(ix); ib < kBlocks; ib += 16u) {
        device const float4 *x4 =
            (device const float4 *)(x + ib * DS4_MXFP4_QK + elemBase);
        const float4 xv0 = x4[0];
        const float4 xv1 = x4[1];
        const float4 xv2 = x4[2];
        const float4 xv3 = x4[3];

        for (uint r = 0; r < 2u; r++) {
            const uint row = row0 + r;
            if (row >= rows) continue;
            device const uchar *q = data + (ulong)row * bytesPerRow + ib * 16u;
            const float scale = ds4mx_e8m0_to_float(scales[(ulong)row * kBlocks + ib]);
            float4 acc = float4(0.0f);
            const uchar q0 = q[byteBase + 0u];
            const uchar q1 = q[byteBase + 1u];
            const uchar q2 = q[byteBase + 2u];
            const uchar q3 = q[byteBase + 3u];
            const uchar q4 = q[byteBase + 4u];
            const uchar q5 = q[byteBase + 5u];
            const uchar q6 = q[byteBase + 6u];
            const uchar q7 = q[byteBase + 7u];

            acc[0] += xv0[0] * ds4mx_e2m1_lut_f32[q0 & 0x0Fu];
            acc[1] += xv0[1] * ds4mx_e2m1_lut_f32[q0 >> 4u];
            acc[2] += xv0[2] * ds4mx_e2m1_lut_f32[q1 & 0x0Fu];
            acc[3] += xv0[3] * ds4mx_e2m1_lut_f32[q1 >> 4u];

            acc[0] += xv1[0] * ds4mx_e2m1_lut_f32[q2 & 0x0Fu];
            acc[1] += xv1[1] * ds4mx_e2m1_lut_f32[q2 >> 4u];
            acc[2] += xv1[2] * ds4mx_e2m1_lut_f32[q3 & 0x0Fu];
            acc[3] += xv1[3] * ds4mx_e2m1_lut_f32[q3 >> 4u];

            acc[0] += xv2[0] * ds4mx_e2m1_lut_f32[q4 & 0x0Fu];
            acc[1] += xv2[1] * ds4mx_e2m1_lut_f32[q4 >> 4u];
            acc[2] += xv2[2] * ds4mx_e2m1_lut_f32[q5 & 0x0Fu];
            acc[3] += xv2[3] * ds4mx_e2m1_lut_f32[q5 >> 4u];

            acc[0] += xv3[0] * ds4mx_e2m1_lut_f32[q6 & 0x0Fu];
            acc[1] += xv3[1] * ds4mx_e2m1_lut_f32[q6 >> 4u];
            acc[2] += xv3[2] * ds4mx_e2m1_lut_f32[q7 & 0x0Fu];
            acc[3] += xv3[3] * ds4mx_e2m1_lut_f32[q7 >> 4u];

            sumf[r] += scale * (acc[0] + acc[1] + acc[2] + acc[3]);
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_pair_swiglu_f32(
        device const float *x            [[buffer(0)]],  // [in_dim]
        device const uchar *gate_data    [[buffer(1)]],
        device const uchar *gate_scales  [[buffer(2)]],
        device const uchar *up_data      [[buffer(3)]],
        device const uchar *up_scales    [[buffer(4)]],
        device const int   *selected     [[buffer(5)]],  // 6 slot ids
        device const float *weights      [[buffer(6)]],  // 6 route weights
        device float       *gate_out     [[buffer(7)]],  // [6][mid_dim]
        device float       *up_out       [[buffer(8)]],  // [6][mid_dim]
        device float       *mid          [[buffer(9)]],  // [6][mid_dim]
        constant uint      &in_dim       [[buffer(10)]],
        constant uint      &mid_dim      [[buffer(11)]],
        constant uint      &slot_stride  [[buffer(12)]],
        constant float     &clamp_value  [[buffer(13)]],
        constant uint      &write_clamped[[buffer(14)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = tgpig.z;
    if (route >= 6u) return;
    const int slot_i = selected[route];
    if (slot_i < 0) return;
    const uint slot = uint(slot_i);
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;

    device const uchar *gate_data_cur = gate_data + (ulong)slot * slot_stride;
    device const uchar *gate_scales_cur = gate_scales + (ulong)slot * slot_stride;
    device const uchar *up_data_cur = up_data + (ulong)slot * slot_stride;
    device const uchar *up_scales_cur = up_scales + (ulong)slot * slot_stride;
    float sumg[2] = { 0.0f, 0.0f };
    float sumu[2] = { 0.0f, 0.0f };

    ds4mx_plane_dot2_accum(gate_data_cur, gate_scales_cur, x,
                           mid_dim, in_dim, row0, tiisg, sumg);
    ds4mx_plane_dot2_accum(up_data_cur, up_scales_cur, x,
                           mid_dim, in_dim, row0, tiisg, sumu);

    const float route_weight = weights[route];
    device float *gate_route = gate_out + (ulong)route * mid_dim;
    device float *up_route = up_out + (ulong)route * mid_dim;
    device float *mid_route = mid + (ulong)route * mid_dim;

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float gate_sum = simd_sum(sumg[r]);
        const float up_sum = simd_sum(sumu[r]);
        if (tiisg == 0 && row < mid_dim) {
            float g = gate_sum;
            float u = up_sum;
            gate_route[row] = g;
            up_route[row] = u;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            if (write_clamped != 0u) {
                gate_route[row] = g;
                up_route[row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_route[row] = silu * u * route_weight;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_sum6_f32(
        device const uchar *down_data    [[buffer(0)]],
        device const uchar *down_scales  [[buffer(1)]],
        device const int   *selected     [[buffer(2)]],  // 6 slot ids
        device const float *mid          [[buffer(3)]],  // [6][mid_dim]
        device float       *out          [[buffer(4)]],  // [out_dim]
        constant uint      &mid_dim      [[buffer(5)]],
        constant uint      &out_dim      [[buffer(6)]],
        constant uint      &slot_stride  [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;
    float sumf[2] = { 0.0f, 0.0f };

    for (uint route = 0; route < 6u; route++) {
        const int slot_i = selected[route];
        if (slot_i < 0) continue;
        const uint slot = uint(slot_i);
        device const uchar *down_data_cur = down_data + (ulong)slot * slot_stride;
        device const uchar *down_scales_cur = down_scales + (ulong)slot * slot_stride;
        device const float *mid_route = mid + (ulong)route * mid_dim;
        ds4mx_plane_dot2_accum(down_data_cur, down_scales_cur, mid_route,
                               out_dim, mid_dim, row0, tiisg, sumf);
    }

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float sum = simd_sum(sumf[r]);
        if (tiisg == 0 && row < out_dim) {
            out[row] = sum;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_batch_pair_swiglu_f32(
        device const float *x            [[buffer(0)]],  // [n_tokens][in_dim]
        device const uchar *gate_data    [[buffer(1)]],
        device const uchar *gate_scales  [[buffer(2)]],
        device const uchar *up_data      [[buffer(3)]],
        device const uchar *up_scales    [[buffer(4)]],
        device const int   *selected     [[buffer(5)]],  // [n_tokens][6] slot ids
        device const float *weights      [[buffer(6)]],  // [n_tokens][6] route weights
        device float       *gate_out     [[buffer(7)]],  // [n_tokens][6][mid_dim]
        device float       *up_out       [[buffer(8)]],  // [n_tokens][6][mid_dim]
        device float       *mid          [[buffer(9)]],  // [n_tokens][6][mid_dim]
        constant uint      &in_dim       [[buffer(10)]],
        constant uint      &mid_dim      [[buffer(11)]],
        constant uint      &slot_stride  [[buffer(12)]],
        constant float     &clamp_value  [[buffer(13)]],
        constant uint      &write_clamped[[buffer(14)]],
        constant uint      &n_tokens     [[buffer(15)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint token = tgpig.y;
    const uint route = tgpig.z;
    if (token >= n_tokens || route >= 6u) return;
    const uint pair = token * 6u + route;
    const int slot_i = selected[pair];
    if (slot_i < 0) return;
    const uint slot = uint(slot_i);
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;

    device const float *x_row = x + (ulong)token * in_dim;
    device const uchar *gate_data_cur = gate_data + (ulong)slot * slot_stride;
    device const uchar *gate_scales_cur = gate_scales + (ulong)slot * slot_stride;
    device const uchar *up_data_cur = up_data + (ulong)slot * slot_stride;
    device const uchar *up_scales_cur = up_scales + (ulong)slot * slot_stride;
    float sumg[2] = { 0.0f, 0.0f };
    float sumu[2] = { 0.0f, 0.0f };

    ds4mx_plane_dot2_accum(gate_data_cur, gate_scales_cur, x_row,
                           mid_dim, in_dim, row0, tiisg, sumg);
    ds4mx_plane_dot2_accum(up_data_cur, up_scales_cur, x_row,
                           mid_dim, in_dim, row0, tiisg, sumu);

    const float route_weight = weights[pair];
    const ulong route_base = ((ulong)token * 6ul + (ulong)route) * mid_dim;
    device float *gate_route = gate_out + route_base;
    device float *up_route = up_out + route_base;
    device float *mid_route = mid + route_base;

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float gate_sum = simd_sum(sumg[r]);
        const float up_sum = simd_sum(sumu[r]);
        if (tiisg == 0 && row < mid_dim) {
            float g = gate_sum;
            float u = up_sum;
            gate_route[row] = g;
            up_route[row] = u;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            if (write_clamped != 0u) {
                gate_route[row] = g;
                up_route[row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_route[row] = silu * u * route_weight;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_batch_sum6_f32(
        device const uchar *down_data    [[buffer(0)]],
        device const uchar *down_scales  [[buffer(1)]],
        device const int   *selected     [[buffer(2)]],  // [n_tokens][6] slot ids
        device const float *mid          [[buffer(3)]],  // [n_tokens][6][mid_dim]
        device float       *out          [[buffer(4)]],  // [n_tokens][out_dim]
        constant uint      &mid_dim      [[buffer(5)]],
        constant uint      &out_dim      [[buffer(6)]],
        constant uint      &slot_stride  [[buffer(7)]],
        constant uint      &n_tokens     [[buffer(8)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint token = tgpig.y;
    if (token >= n_tokens) return;
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;
    float sumf[2] = { 0.0f, 0.0f };

    for (uint route = 0; route < 6u; route++) {
        const uint pair = token * 6u + route;
        const int slot_i = selected[pair];
        if (slot_i < 0) continue;
        const uint slot = uint(slot_i);
        device const uchar *down_data_cur = down_data + (ulong)slot * slot_stride;
        device const uchar *down_scales_cur = down_scales + (ulong)slot * slot_stride;
        device const float *mid_route = mid + ((ulong)token * 6ul + (ulong)route) * mid_dim;
        ds4mx_plane_dot2_accum(down_data_cur, down_scales_cur, mid_route,
                               out_dim, mid_dim, row0, tiisg, sumf);
    }

    device float *out_row = out + (ulong)token * out_dim;
    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float sum = simd_sum(sumf[r]);
        if (tiisg == 0 && row < out_dim) {
            out_row[row] = sum;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_batch_pair_swiglu_strided_f32(
        device const float *x                [[buffer(0)]],  // [n_tokens][in_dim]
        device const uchar *gate_data        [[buffer(1)]],
        device const uchar *gate_scales      [[buffer(2)]],
        device const uchar *up_data          [[buffer(3)]],
        device const uchar *up_scales        [[buffer(4)]],
        device const int   *selected         [[buffer(5)]],  // [n_tokens][6] slot ids
        device const float *weights          [[buffer(6)]],  // [n_tokens][6] route weights
        device float       *gate_out         [[buffer(7)]],  // [n_tokens][6][mid_dim]
        device float       *up_out           [[buffer(8)]],  // [n_tokens][6][mid_dim]
        device float       *mid              [[buffer(9)]],  // [n_tokens][6][mid_dim]
        constant uint      &in_dim           [[buffer(10)]],
        constant uint      &mid_dim          [[buffer(11)]],
        constant uint      &data_slot_stride [[buffer(12)]],
        constant uint      &scale_slot_stride[[buffer(13)]],
        constant float     &clamp_value      [[buffer(14)]],
        constant uint      &write_clamped    [[buffer(15)]],
        constant uint      &n_tokens         [[buffer(16)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint token = tgpig.y;
    const uint route = tgpig.z;
    if (token >= n_tokens || route >= 6u) return;
    const uint pair = token * 6u + route;
    const int slot_i = selected[pair];
    if (slot_i < 0) return;
    const uint slot = uint(slot_i);
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;

    device const float *x_row = x + (ulong)token * in_dim;
    device const uchar *gate_data_cur = gate_data + (ulong)slot * data_slot_stride;
    device const uchar *gate_scales_cur = gate_scales + (ulong)slot * scale_slot_stride;
    device const uchar *up_data_cur = up_data + (ulong)slot * data_slot_stride;
    device const uchar *up_scales_cur = up_scales + (ulong)slot * scale_slot_stride;
    float sumg[2] = { 0.0f, 0.0f };
    float sumu[2] = { 0.0f, 0.0f };

    ds4mx_plane_dot2_accum(gate_data_cur, gate_scales_cur, x_row,
                           mid_dim, in_dim, row0, tiisg, sumg);
    ds4mx_plane_dot2_accum(up_data_cur, up_scales_cur, x_row,
                           mid_dim, in_dim, row0, tiisg, sumu);

    const float route_weight = weights[pair];
    const ulong route_base = ((ulong)token * 6ul + (ulong)route) * mid_dim;
    device float *gate_route = gate_out + route_base;
    device float *up_route = up_out + route_base;
    device float *mid_route = mid + route_base;

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float gate_sum = simd_sum(sumg[r]);
        const float up_sum = simd_sum(sumu[r]);
        if (tiisg == 0 && row < mid_dim) {
            float g = gate_sum;
            float u = up_sum;
            gate_route[row] = g;
            up_route[row] = u;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            if (write_clamped != 0u) {
                gate_route[row] = g;
                up_route[row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_route[row] = silu * u * route_weight;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_batch_sum6_strided_f32(
        device const uchar *down_data        [[buffer(0)]],
        device const uchar *down_scales      [[buffer(1)]],
        device const int   *selected         [[buffer(2)]],  // [n_tokens][6] slot ids
        device const float *mid              [[buffer(3)]],  // [n_tokens][6][mid_dim]
        device float       *out              [[buffer(4)]],  // [n_tokens][out_dim]
        constant uint      &mid_dim          [[buffer(5)]],
        constant uint      &out_dim          [[buffer(6)]],
        constant uint      &data_slot_stride [[buffer(7)]],
        constant uint      &scale_slot_stride[[buffer(8)]],
        constant uint      &n_tokens         [[buffer(9)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint token = tgpig.y;
    if (token >= n_tokens) return;
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;
    float sumf[2] = { 0.0f, 0.0f };

    for (uint route = 0; route < 6u; route++) {
        const uint pair = token * 6u + route;
        const int slot_i = selected[pair];
        if (slot_i < 0) continue;
        const uint slot = uint(slot_i);
        device const uchar *down_data_cur = down_data + (ulong)slot * data_slot_stride;
        device const uchar *down_scales_cur = down_scales + (ulong)slot * scale_slot_stride;
        device const float *mid_route = mid + ((ulong)token * 6ul + (ulong)route) * mid_dim;
        ds4mx_plane_dot2_accum(down_data_cur, down_scales_cur, mid_route,
                               out_dim, mid_dim, row0, tiisg, sumf);
    }

    device float *out_row = out + (ulong)token * out_dim;
    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float sum = simd_sum(sumf[r]);
        if (tiisg == 0 && row < out_dim) {
            out_row[row] = sum;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_slots6_sum6_f32(
        device const uchar *down_data0   [[buffer(0)]],
        device const uchar *down_data1   [[buffer(1)]],
        device const uchar *down_data2   [[buffer(2)]],
        device const uchar *down_data3   [[buffer(3)]],
        device const uchar *down_data4   [[buffer(4)]],
        device const uchar *down_data5   [[buffer(5)]],
        device const uchar *down_scales0 [[buffer(6)]],
        device const uchar *down_scales1 [[buffer(7)]],
        device const uchar *down_scales2 [[buffer(8)]],
        device const uchar *down_scales3 [[buffer(9)]],
        device const uchar *down_scales4 [[buffer(10)]],
        device const uchar *down_scales5 [[buffer(11)]],
        device const float *mid          [[buffer(12)]],  // [6][mid_dim]
        device float       *out          [[buffer(13)]],  // [out_dim]
        constant uint      &mid_dim      [[buffer(14)]],
        constant uint      &out_dim      [[buffer(15)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;
    float sumf[2] = { 0.0f, 0.0f };

    for (uint route = 0; route < 6u; route++) {
        device const uchar *down_data_cur = down_data0;
        device const uchar *down_scales_cur = down_scales0;
        switch (route) {
        case 1: down_data_cur = down_data1; down_scales_cur = down_scales1; break;
        case 2: down_data_cur = down_data2; down_scales_cur = down_scales2; break;
        case 3: down_data_cur = down_data3; down_scales_cur = down_scales3; break;
        case 4: down_data_cur = down_data4; down_scales_cur = down_scales4; break;
        case 5: down_data_cur = down_data5; down_scales_cur = down_scales5; break;
        default: break;
        }
        device const float *mid_route = mid + (ulong)route * mid_dim;
        ds4mx_plane_dot2_accum(down_data_cur, down_scales_cur, mid_route,
                               out_dim, mid_dim, row0, tiisg, sumf);
    }

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float sum = simd_sum(sumf[r]);
        if (tiisg == 0 && row < out_dim) {
            out[row] = sum;
        }
    }
}

struct ds4_metal_slots6_chunk_map {
    uint32_t chunk[6];
    uint32_t slot[6];
    ulong slot_stride;
};

static inline device const uchar *ds4mx_slots6_chunk_select(
        uint32_t chunk,
        device const uchar *chunk0,
        device const uchar *chunk1,
        device const uchar *chunk2,
        device const uchar *chunk3)
{
    switch (chunk) {
    case 1: return chunk1;
    case 2: return chunk2;
    case 3: return chunk3;
    default: return chunk0;
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_chunked_pair_swiglu_f32(
        device const float *x            [[buffer(0)]],  // [in_dim]
        constant ds4_metal_slots6_chunk_map &map [[buffer(1)]],
        device const float *weights      [[buffer(2)]],  // 6 route weights
        device float       *gate_out     [[buffer(3)]],  // [6][mid_dim]
        device float       *up_out       [[buffer(4)]],  // [6][mid_dim]
        device float       *mid          [[buffer(5)]],  // [6][mid_dim]
        constant uint      &in_dim       [[buffer(6)]],
        constant uint      &mid_dim      [[buffer(7)]],
        constant float     &clamp_value  [[buffer(8)]],
        constant uint      &write_clamped[[buffer(9)]],
        device const uchar *gate_data0   [[buffer(10)]],
        device const uchar *gate_data1   [[buffer(11)]],
        device const uchar *gate_data2   [[buffer(12)]],
        device const uchar *gate_data3   [[buffer(13)]],
        device const uchar *gate_scales0 [[buffer(14)]],
        device const uchar *gate_scales1 [[buffer(15)]],
        device const uchar *gate_scales2 [[buffer(16)]],
        device const uchar *gate_scales3 [[buffer(17)]],
        device const uchar *up_data0     [[buffer(18)]],
        device const uchar *up_data1     [[buffer(19)]],
        device const uchar *up_data2     [[buffer(20)]],
        device const uchar *up_data3     [[buffer(21)]],
        device const uchar *up_scales0   [[buffer(22)]],
        device const uchar *up_scales1   [[buffer(23)]],
        device const uchar *up_scales2   [[buffer(24)]],
        device const uchar *up_scales3   [[buffer(25)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = tgpig.z;
    if (route >= 6u) return;
    const uint chunk = map.chunk[route];
    if (chunk >= 4u) return;
    const uint slot = map.slot[route];
    const ulong slot_off = (ulong)slot * map.slot_stride;
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;

    device const uchar *gate_data_cur =
        ds4mx_slots6_chunk_select(chunk, gate_data0, gate_data1, gate_data2, gate_data3) + slot_off;
    device const uchar *gate_scales_cur =
        ds4mx_slots6_chunk_select(chunk, gate_scales0, gate_scales1, gate_scales2, gate_scales3) + slot_off;
    device const uchar *up_data_cur =
        ds4mx_slots6_chunk_select(chunk, up_data0, up_data1, up_data2, up_data3) + slot_off;
    device const uchar *up_scales_cur =
        ds4mx_slots6_chunk_select(chunk, up_scales0, up_scales1, up_scales2, up_scales3) + slot_off;
    float sumg[2] = { 0.0f, 0.0f };
    float sumu[2] = { 0.0f, 0.0f };

    ds4mx_plane_dot2_accum(gate_data_cur, gate_scales_cur, x,
                           mid_dim, in_dim, row0, tiisg, sumg);
    ds4mx_plane_dot2_accum(up_data_cur, up_scales_cur, x,
                           mid_dim, in_dim, row0, tiisg, sumu);

    const float route_weight = weights[route];
    device float *gate_route = gate_out + (ulong)route * mid_dim;
    device float *up_route = up_out + (ulong)route * mid_dim;
    device float *mid_route = mid + (ulong)route * mid_dim;

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float gate_sum = simd_sum(sumg[r]);
        const float up_sum = simd_sum(sumu[r]);
        if (tiisg == 0 && row < mid_dim) {
            float g = gate_sum;
            float u = up_sum;
            gate_route[row] = g;
            up_route[row] = u;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            if (write_clamped != 0u) {
                gate_route[row] = g;
                up_route[row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_route[row] = silu * u * route_weight;
        }
    }
}

kernel void kernel_dsv4_mxfp4_plane_id_chunked_sum6_f32(
        constant ds4_metal_slots6_chunk_map &map [[buffer(0)]],
        device const float *mid          [[buffer(1)]],  // [6][mid_dim]
        device float       *out          [[buffer(2)]],  // [out_dim]
        constant uint      &mid_dim      [[buffer(3)]],
        constant uint      &out_dim      [[buffer(4)]],
        device const uchar *down_data0   [[buffer(5)]],
        device const uchar *down_data1   [[buffer(6)]],
        device const uchar *down_data2   [[buffer(7)]],
        device const uchar *down_data3   [[buffer(8)]],
        device const uchar *down_scales0 [[buffer(9)]],
        device const uchar *down_scales1 [[buffer(10)]],
        device const uchar *down_scales2 [[buffer(11)]],
        device const uchar *down_scales3 [[buffer(12)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = (tgpig.x * 2u + uint(sgitg)) * 2u;
    float sumf[2] = { 0.0f, 0.0f };

    for (uint route = 0; route < 6u; route++) {
        const uint chunk = map.chunk[route];
        if (chunk >= 4u) continue;
        const uint slot = map.slot[route];
        const ulong slot_off = (ulong)slot * map.slot_stride;
        device const uchar *down_data_cur =
            ds4mx_slots6_chunk_select(chunk, down_data0, down_data1, down_data2, down_data3) + slot_off;
        device const uchar *down_scales_cur =
            ds4mx_slots6_chunk_select(chunk, down_scales0, down_scales1, down_scales2, down_scales3) + slot_off;
        device const float *mid_route = mid + (ulong)route * mid_dim;
        ds4mx_plane_dot2_accum(down_data_cur, down_scales_cur, mid_route,
                               out_dim, mid_dim, row0, tiisg, sumf);
    }

    for (uint r = 0; r < 2u; r++) {
        const uint row = row0 + r;
        const float sum = simd_sum(sumf[r]);
        if (tiisg == 0 && row < out_dim) {
            out[row] = sum;
        }
    }
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

kernel void kernel_dsv4_mxfp4_native_matmul_selected_n64(
        device half        *a           [[buffer(0)]],
        device uchar       *weights     [[buffer(1)]],
        device float       *c           [[buffer(2)]],
        constant uint      &m           [[buffer(3)]],
        constant uint      &n           [[buffer(4)]],
        constant uint      &k           [[buffer(5)]],
        device const uchar *scales      [[buffer(6)]],
        device const int   *selected    [[buffer(7)]],
        constant uint      &slot_stride [[buffer(8)]],
        constant uint      &route       [[buffer(9)]],
        uint2 tgid [[threadgroup_position_in_grid]])
{
    const int slot_i = selected[route];
    if (slot_i < 0) return;
    const uint slot = uint(slot_i);
    ds4mx::scaled_matmul_tile<64>(a,
                                  weights + slot * slot_stride,
                                  scales + slot * slot_stride,
                                  c,
                                  m,
                                  n,
                                  k,
                                  tgid);
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
