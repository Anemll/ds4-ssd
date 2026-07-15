#include <metal_stdlib>
using namespace metal;

struct hy3_kv_args {
    uint pos;
    uint ctx;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint nwg;
    uint n_tokens;
    float scale;
};

struct hy3_block_q8_0 {
    half d;
    char qs[32];
};

static void hy3_quantize_q8_0(device const float *src,
                              device hy3_block_q8_0 &dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f;
    for (uint i = 0; i < 32u; ++i) amax = max(amax, fabs(src[i]));
    const float d = amax / 127.0f;
    const float inv = d != 0.0f ? 1.0f / d : 0.0f;
    dst.d = half(d);
    for (uint i = 0; i < 32u; ++i) dst.qs[i] = char(round(src[i] * inv));
}

kernel void kernel_hy3_store_kv_q8_0(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *key  [[buffer(1)]],
        device const float   *val  [[buffer(2)]],
        device hy3_block_q8_0 *kc  [[buffer(3)]],
        device hy3_block_q8_0 *vc  [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint blocks = args.n_head_kv * (args.head_dim / 32u);
    if (gid >= blocks || args.pos >= args.ctx) return;
    const ulong off = ulong(args.pos) * ulong(blocks) + ulong(gid);
    hy3_quantize_q8_0(key + gid * 32u, kc[off]);
    hy3_quantize_q8_0(val + gid * 32u, vc[off]);
}

kernel void kernel_hy3_store_kv_q8_0_batch(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *key  [[buffer(1)]],
        device const float   *val  [[buffer(2)]],
        device hy3_block_q8_0 *kc  [[buffer(3)]],
        device hy3_block_q8_0 *vc  [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;
    const uint total = args.n_tokens * blocks_per_token;
    if (gid >= total) return;
    const uint token = gid / blocks_per_token;
    const uint token_block = gid - token * blocks_per_token;
    const uint kv_head = token_block / blocks_per_head;
    const uint block = token_block - kv_head * blocks_per_head;
    const uint pos = args.pos + token;
    if (pos >= args.ctx) return;
    const ulong off = ulong(pos) * ulong(blocks_per_token) + ulong(token_block);
    hy3_quantize_q8_0(key + ulong(gid) * 32u, kc[off]);
    hy3_quantize_q8_0(val + ulong(gid) * 32u, vc[off]);
}

/* Persistent head-major F16 KV cache used by the optional M5 NAX-half
 * attention path.  Each new token is written directly to its final cache
 * location; long-context attention binds that cache in place. */
kernel void kernel_hy3_store_kv_f16_headmajor(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *key  [[buffer(1)]],
        device const float   *val  [[buffer(2)]],
        device half          *kc   [[buffer(3)]],
        device half          *vc   [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_head_kv * args.head_dim;
    if (gid >= total || args.pos >= args.ctx) return;
    const uint kv_head = gid / args.head_dim;
    const uint d = gid - kv_head * args.head_dim;
    const uint ctx_pad = (args.ctx + 31u) & ~31u;
    const ulong dst = (ulong(kv_head) * ulong(ctx_pad) + ulong(args.pos)) *
                      ulong(args.head_dim) + ulong(d);
    kc[dst] = half(key[gid]);
    vc[dst] = half(val[gid]);
}

kernel void kernel_hy3_store_kv_f16_headmajor_batch(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *key  [[buffer(1)]],
        device const float   *val  [[buffer(2)]],
        device half          *kc   [[buffer(3)]],
        device half          *vc   [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint per_token = args.n_head_kv * args.head_dim;
    const uint total = args.n_tokens * per_token;
    if (gid >= total) return;
    const uint token = gid / per_token;
    const uint rem = gid - token * per_token;
    const uint kv_head = rem / args.head_dim;
    const uint d = rem - kv_head * args.head_dim;
    const uint pos = args.pos + token;
    if (pos >= args.ctx) return;
    const uint ctx_pad = (args.ctx + 31u) & ~31u;
    const ulong dst = (ulong(kv_head) * ulong(ctx_pad) + ulong(pos)) *
                      ulong(args.head_dim) + ulong(d);
    kc[dst] = half(key[gid]);
    vc[dst] = half(val[gid]);
}

/* Conventional causal GQA attention for HY3 decode.  Each workgroup owns one
 * query head.  Its 128 lanes reduce Q.K and concurrently accumulate the 128-D
 * value using an online softmax, so no context-sized scores buffer is needed. */
kernel void kernel_hy3_gqa_decode_q8_0(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *query [[buffer(1)]],
        device const hy3_block_q8_0 *kc [[buffer(2)]],
        device const hy3_block_q8_0 *vc [[buffer(3)]],
        device float         *out   [[buffer(4)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]]) {
    if (head >= args.n_head || tid >= args.head_dim) return;
    threadgroup float reduce[128];
    threadgroup float state[4]; // max, sum, alpha, beta

    const uint qbase = head * args.head_dim;
    const uint group = args.n_head / args.n_head_kv;
    const uint kv_head = head / group;
    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;
    const uint block = tid / 32u;
    const uint lane = tid % 32u;
    float acc = 0.0f;

    if (tid == 0u) {
        state[0] = -INFINITY;
        state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = 0u; p <= args.pos; ++p) {
        const ulong koff = ulong(p) * ulong(blocks_per_token) +
                           ulong(kv_head * blocks_per_head + block);
        reduce[tid] = query[qbase + tid] *
                      (float(kc[koff].d) * float(kc[koff].qs[lane]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64u; stride != 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            const float score = reduce[0] * args.scale;
            const float next_max = max(state[0], score);
            state[2] = exp(state[0] - next_max);
            state[3] = exp(score - next_max);
            state[1] = state[1] * state[2] + state[3];
            state[0] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong voff = ulong(p) * ulong(blocks_per_token) +
                           ulong(kv_head * blocks_per_head + block);
        const float vv = float(vc[voff].d) * float(vc[voff].qs[lane]);
        acc = acc * state[2] + state[3] * vv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[qbase + tid] = acc / state[1];
}

/* Flash-style decode attention.  Instead of making one workgroup walk the
 * complete context, split each query head across args.nwg SIMD workgroups.
 * A SIMD group consumes 32 contiguous cache rows at a time: its lanes hold
 * contiguous float4 slices of Q, cooperate on all 32 Q.K reductions, and then
 * accumulate the matching float4 slices of V.  This follows llama.cpp's
 * vector-attention work decomposition while retaining DS4's token-major Q8_0
 * cache and partial/reduce ABI. */
kernel void kernel_hy3_gqa_decode_q8_0_partial(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *query [[buffer(1)]],
        device const hy3_block_q8_0 *kc [[buffer(2)]],
        device const hy3_block_q8_0 *vc [[buffer(3)]],
        device float         *scratch [[buffer(4)]],
        uint group_id [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group_id / args.nwg;
    const uint iwg = group_id - head * args.nwg;
    if (head >= args.n_head) return;

    const uint qbase = head * args.head_dim;
    const uint gqa = args.n_head / args.n_head_kv;
    const uint kv_head = head / gqa;
    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;

    device const float4 *query4 =
        reinterpret_cast<device const float4 *>(query + qbase);
    const float4 qv = query4[lane];
    const uint q8_block = uint(lane) >> 3u;
    const uint q8_lane4 = (uint(lane) & 7u) << 2u;
    threadgroup float tile_weight[32];
    float4 acc = float4(0.0f);
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    const uint tile_stride = args.nwg * 32u;
    for (uint tile = iwg * 32u; tile <= args.pos; tile += tile_stride) {
        const uint tile_count = min(32u, args.pos - tile + 1u);
        float lane_score = -INFINITY;

        /* Each Q8_0 block contains 32 adjacent dimensions.  Lane L owns
         * dimensions [4L, 4L+3], so it reads one float4-equivalent slice from
         * exactly one of the four blocks in a 128-D cache row. */
        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &kb = kc[base];
            const float kd = float(kb.d);
            const float4 kv = float4(float(kb.qs[q8_lane4]),
                                     float(kb.qs[q8_lane4 + 1u]),
                                     float(kb.qs[q8_lane4 + 2u]),
                                     float(kb.qs[q8_lane4 + 3u])) * kd;
            const float score = simd_sum(dot(qv, kv)) * args.scale;
            if (uint(lane) == c) lane_score = score;
        }

        const float next_max = max(local_max, simd_max(lane_score));
        const float alpha = exp(local_max - next_max);
        const float weight = exp(lane_score - next_max);
        tile_weight[lane] = weight;
        const float tile_sum = simd_sum(weight);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 tile_acc = float4(0.0f);
        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &vb = vc[base];
            const float vd = float(vb.d);
            const float4 vv = float4(float(vb.qs[q8_lane4]),
                                     float(vb.qs[q8_lane4 + 1u]),
                                     float(vb.qs[q8_lane4 + 2u]),
                                     float(vb.qs[q8_lane4 + 3u])) * vd;
            tile_acc = fma(vv, tile_weight[c], tile_acc);
        }

        local_sum = fma(local_sum, alpha, tile_sum);
        acc = fma(acc, alpha, tile_acc);
        local_max = next_max;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong partial_stride = ulong(args.head_dim + 2u);
    const ulong dst = ulong(head * args.nwg + iwg) * partial_stride;
    if (lane == 0u) {
        scratch[dst] = local_max;
        scratch[dst + 1u] = local_sum;
    }
    const ulong out4 = dst + 2u + ulong(lane) * 4u;
    scratch[out4] = acc.x;
    scratch[out4 + 1u] = acc.y;
    scratch[out4 + 2u] = acc.z;
    scratch[out4 + 3u] = acc.w;
}

/* Long-context decode variant matching llama.cpp's Q8/DK128 work split: one
 * workgroup still owns one (query head, partial), but four SIMDgroups walk four
 * independent 32-row KV streams.  The four online-softmax states are merged in
 * the workgroup before writing the existing partial/reduce ABI.  This keeps the
 * same Q8 cache and numerical contract while exposing 4x more cache-row
 * parallelism once a single-SIMD partial would otherwise walk several tiles. */
kernel void kernel_hy3_gqa_decode_q8_0_partial4(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *query [[buffer(1)]],
        device const hy3_block_q8_0 *kc [[buffer(2)]],
        device const hy3_block_q8_0 *vc [[buffer(3)]],
        device float         *scratch [[buffer(4)]],
        uint group_id [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint head = group_id / args.nwg;
    const uint iwg = group_id - head * args.nwg;
    if (head >= args.n_head) return;

    const uint qbase = head * args.head_dim;
    const uint gqa = args.n_head / args.n_head_kv;
    const uint kv_head = head / gqa;
    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;

    device const float4 *query4 =
        reinterpret_cast<device const float4 *>(query + qbase);
    const float4 qv = query4[lane];
    const uint q8_block = uint(lane) >> 3u;
    const uint q8_lane4 = (uint(lane) & 7u) << 2u;
    threadgroup float tile_weight[4][32];
    threadgroup float sg_max[4];
    threadgroup float sg_sum[4];
    threadgroup float sg_scale[4];
    threadgroup float4 sg_acc[4][32];
    float4 acc = float4(0.0f);
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    const uint tile_stride = args.nwg * 4u * 32u;
    for (uint tile = (iwg * 4u + uint(sgitg)) * 32u;
         tile <= args.pos; tile += tile_stride) {
        const uint tile_count = min(32u, args.pos - tile + 1u);
        float lane_score = -INFINITY;

        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &kb = kc[base];
            const float kd = float(kb.d);
            const float4 kv = float4(float(kb.qs[q8_lane4]),
                                     float(kb.qs[q8_lane4 + 1u]),
                                     float(kb.qs[q8_lane4 + 2u]),
                                     float(kb.qs[q8_lane4 + 3u])) * kd;
            const float score = simd_sum(dot(qv, kv)) * args.scale;
            if (uint(lane) == c) lane_score = score;
        }

        const float next_max = max(local_max, simd_max(lane_score));
        const float alpha = exp(local_max - next_max);
        const float weight = exp(lane_score - next_max);
        tile_weight[sgitg][lane] = weight;
        const float tile_sum = simd_sum(weight);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 tile_acc = float4(0.0f);
        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &vb = vc[base];
            const float vd = float(vb.d);
            const float4 vv = float4(float(vb.qs[q8_lane4]),
                                     float(vb.qs[q8_lane4 + 1u]),
                                     float(vb.qs[q8_lane4 + 2u]),
                                     float(vb.qs[q8_lane4 + 3u])) * vd;
            tile_acc = fma(vv, tile_weight[sgitg][c], tile_acc);
        }

        local_sum = fma(local_sum, alpha, tile_sum);
        acc = fma(acc, alpha, tile_acc);
        local_max = next_max;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        sg_max[sgitg] = local_max;
        sg_sum[sgitg] = local_sum;
    }
    sg_acc[sgitg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0u) {
        const float candidate = lane < 4u ? sg_max[lane] : -INFINITY;
        const float merged_max = simd_max(candidate);
        float scale_part = 0.0f;
        float sum_part = 0.0f;
        if (lane < 4u) {
            if (sg_sum[lane] > 0.0f) {
                scale_part = exp(candidate - merged_max);
                sum_part = sg_sum[lane] * scale_part;
            }
            sg_scale[lane] = scale_part;
        }
        const float merged_sum = simd_sum(sum_part);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 merged_acc = float4(0.0f);
        for (uint sg = 0u; sg < 4u; ++sg) {
            merged_acc = fma(sg_acc[sg][lane], sg_scale[sg], merged_acc);
        }

        const ulong partial_stride = ulong(args.head_dim + 2u);
        const ulong dst = ulong(head * args.nwg + iwg) * partial_stride;
        if (lane == 0u) {
            scratch[dst] = merged_max;
            scratch[dst + 1u] = merged_sum;
        }
        const ulong out4 = dst + 2u + ulong(lane) * 4u;
        scratch[out4] = merged_acc.x;
        scratch[out4 + 1u] = merged_acc.y;
        scratch[out4 + 2u] = merged_acc.z;
        scratch[out4 + 3u] = merged_acc.w;
    }
}

/* GQA8-fused long-context decode.  One 256-thread workgroup owns one
 * (KV head, partial): its eight SIMDgroups preserve the existing scalar Q.K
 * and P.V order for the eight query heads while sharing each dequantized
 * 32x128 K/V tile.  The four SG4 context streams are still accumulated and
 * merged independently, so the output uses the exact same partial/reduce ABI
 * as kernel_hy3_gqa_decode_q8_0_partial4.  The host only dispatches this
 * kernel for head_dim=128 and n_head/n_head_kv=8. */
kernel void kernel_hy3_gqa_decode_q8_0_gqa8_partial4(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *query [[buffer(1)]],
        device const hy3_block_q8_0 *kc [[buffer(2)]],
        device const hy3_block_q8_0 *vc [[buffer(3)]],
        device float         *scratch [[buffer(4)]],
        uint group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint kv_head = group_id / args.nwg;
    const uint iwg = group_id - kv_head * args.nwg;
    const uint head_in_group = uint(sgitg);
    const uint head = kv_head * 8u + head_in_group;

    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;
    const uint qbase = head * args.head_dim;
    device const float4 *query4 =
        reinterpret_cast<device const float4 *>(query + qbase);
    const float4 qv = query4[lane];

    /* K and V use this storage sequentially.  Keeping the staged values in
     * float preserves Q8 dequantization exactly; no extra half rounding is
     * introduced by the fused path. */
    threadgroup float4 kv_tile[32][32];

    float stream_max[4] = {
        -INFINITY, -INFINITY, -INFINITY, -INFINITY
    };
    float stream_sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 stream_acc[4] = {
        float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)
    };

    const uint tile_stride = args.nwg * 4u * 32u;
    for (uint stream = 0u; stream < 4u; ++stream) {
        float local_max = -INFINITY;
        float local_sum = 0.0f;
        float4 acc = float4(0.0f);

        for (uint tile = (iwg * 4u + stream) * 32u;
             tile <= args.pos; tile += tile_stride) {
            const uint tile_count = min(32u, args.pos - tile + 1u);
            const uint tile_slices = tile_count * 32u;

            /* Cooperatively dequantize K once for all eight query heads.
             * A slice is four adjacent dimensions, matching one SIMD lane's
             * float4 in the existing SG4 kernel. */
            for (uint i = tid; i < tile_slices; i += 256u) {
                const uint c = i >> 5u;
                const uint slice = i & 31u;
                const uint q8_block = slice >> 3u;
                const uint q8_lane4 = (slice & 7u) << 2u;
                const ulong base =
                    ulong(tile + c) * ulong(blocks_per_token) +
                    ulong(kv_head * blocks_per_head + q8_block);
                device const hy3_block_q8_0 &kb = kc[base];
                const float kd = float(kb.d);
                kv_tile[c][slice] =
                    float4(float(kb.qs[q8_lane4]),
                           float(kb.qs[q8_lane4 + 1u]),
                           float(kb.qs[q8_lane4 + 2u]),
                           float(kb.qs[q8_lane4 + 3u])) * kd;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float lane_score = -INFINITY;
            for (uint c = 0u; c < tile_count; ++c) {
                const float score =
                    simd_sum(dot(qv, kv_tile[c][lane])) * args.scale;
                if (uint(lane) == c) lane_score = score;
            }

            const float next_max = max(local_max, simd_max(lane_score));
            const float alpha = exp(local_max - next_max);
            const float weight = exp(lane_score - next_max);
            const float tile_sum = simd_sum(weight);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* Reuse the same shared tile for V.  All query-head SIMDgroups
             * consume these values before the next K tile overwrites them. */
            for (uint i = tid; i < tile_slices; i += 256u) {
                const uint c = i >> 5u;
                const uint slice = i & 31u;
                const uint q8_block = slice >> 3u;
                const uint q8_lane4 = (slice & 7u) << 2u;
                const ulong base =
                    ulong(tile + c) * ulong(blocks_per_token) +
                    ulong(kv_head * blocks_per_head + q8_block);
                device const hy3_block_q8_0 &vb = vc[base];
                const float vd = float(vb.d);
                kv_tile[c][slice] =
                    float4(float(vb.qs[q8_lane4]),
                           float(vb.qs[q8_lane4 + 1u]),
                           float(vb.qs[q8_lane4 + 2u]),
                           float(vb.qs[q8_lane4 + 3u])) * vd;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float4 tile_acc = float4(0.0f);
            for (uint c = 0u; c < tile_count; ++c) {
                tile_acc = fma(kv_tile[c][lane],
                               simd_shuffle(weight, c), tile_acc);
            }

            local_sum = fma(local_sum, alpha, tile_sum);
            acc = fma(acc, alpha, tile_acc);
            local_max = next_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        stream_max[stream] = local_max;
        stream_sum[stream] = local_sum;
        stream_acc[stream] = acc;
    }

    /* Match partial4's four-way online-softmax merge, including its stream
     * order and empty-stream handling. */
    const float candidate =
        lane < 4u ? stream_max[uint(lane)] : -INFINITY;
    const float merged_max = simd_max(candidate);
    float scale_part = 0.0f;
    float sum_part = 0.0f;
    if (lane < 4u) {
        const uint stream = uint(lane);
        if (stream_sum[stream] > 0.0f) {
            scale_part = exp(candidate - merged_max);
            sum_part = stream_sum[stream] * scale_part;
        }
    }
    const float merged_sum = simd_sum(sum_part);

    float4 merged_acc = float4(0.0f);
    for (uint stream = 0u; stream < 4u; ++stream) {
        merged_acc = fma(stream_acc[stream],
                         simd_shuffle(scale_part, stream), merged_acc);
    }

    const ulong partial_stride = ulong(args.head_dim + 2u);
    const ulong dst = ulong(head * args.nwg + iwg) * partial_stride;
    if (lane == 0u) {
        scratch[dst] = merged_max;
        scratch[dst + 1u] = merged_sum;
    }
    const ulong out4 = dst + 2u + ulong(lane) * 4u;
    scratch[out4] = merged_acc.x;
    scratch[out4 + 1u] = merged_acc.y;
    scratch[out4 + 2u] = merged_acc.z;
    scratch[out4 + 3u] = merged_acc.w;
}

kernel void kernel_hy3_gqa_decode_q8_0_reduce(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *scratch [[buffer(1)]],
        device float         *out [[buffer(2)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (head >= args.n_head || tid >= args.head_dim) return;
    threadgroup float weights[32];
    threadgroup float normalizer;
    const ulong partial_stride = ulong(args.head_dim + 2u);

    if (tid < 32u) {
        float m = -INFINITY;
        if (tid < args.nwg) {
            const ulong src = ulong(head * args.nwg + tid) * partial_stride;
            m = scratch[src];
        }
        const float global_max = simd_max(m);
        float weighted_sum = 0.0f;
        float weight = 0.0f;
        if (tid < args.nwg) {
            const ulong src = ulong(head * args.nwg + tid) * partial_stride;
            weight = exp(m - global_max);
            weighted_sum = scratch[src + 1u] * weight;
        }
        if (tid < args.nwg) weights[tid] = weight;
        const float total = simd_sum(weighted_sum);
        if (tid == 0u) normalizer = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint iwg = 0u; iwg < args.nwg; ++iwg) {
        const ulong src = ulong(head * args.nwg + iwg) * partial_stride;
        acc += scratch[src + 2u + tid] * weights[iwg];
    }
    out[head * args.head_dim + tid] = acc / normalizer;
}

/* Layer-major causal prefill variant.  Activations and the persistent Q8 cache
 * both retain DS4's normal token-major layout. */
kernel void kernel_hy3_gqa_prefill_q8_0_partial(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *query [[buffer(1)]],
        device const hy3_block_q8_0 *kc [[buffer(2)]],
        device const hy3_block_q8_0 *vc [[buffer(3)]],
        device float         *scratch [[buffer(4)]],
        uint group_id [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint iwg = group_id % args.nwg;
    const uint query_index = group_id / args.nwg;
    const uint head = query_index % args.n_head;
    const uint token = query_index / args.n_head;
    if (token >= args.n_tokens) return;

    const uint qbase = (token * args.n_head + head) * args.head_dim;
    const uint gqa = args.n_head / args.n_head_kv;
    const uint kv_head = head / gqa;
    const uint blocks_per_head = args.head_dim / 32u;
    const uint blocks_per_token = args.n_head_kv * blocks_per_head;
    const uint last_pos = args.pos + token;

    device const float4 *query4 =
        reinterpret_cast<device const float4 *>(query + qbase);
    const float4 qv = query4[lane];
    const uint q8_block = uint(lane) >> 3u;
    const uint q8_lane4 = (uint(lane) & 7u) << 2u;
    threadgroup float tile_weight[32];
    float4 acc = float4(0.0f);
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    const uint tile_stride = args.nwg * 32u;
    for (uint tile = iwg * 32u; tile <= last_pos; tile += tile_stride) {
        const uint tile_count = min(32u, last_pos - tile + 1u);
        float lane_score = -INFINITY;

        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &kb = kc[base];
            const float kd = float(kb.d);
            const float4 kv = float4(float(kb.qs[q8_lane4]),
                                     float(kb.qs[q8_lane4 + 1u]),
                                     float(kb.qs[q8_lane4 + 2u]),
                                     float(kb.qs[q8_lane4 + 3u])) * kd;
            const float score = simd_sum(dot(qv, kv)) * args.scale;
            if (uint(lane) == c) lane_score = score;
        }

        const float next_max = max(local_max, simd_max(lane_score));
        const float alpha = exp(local_max - next_max);
        const float weight = exp(lane_score - next_max);
        tile_weight[lane] = weight;
        const float tile_sum = simd_sum(weight);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 tile_acc = float4(0.0f);
        for (uint c = 0u; c < tile_count; ++c) {
            const ulong base = ulong(tile + c) * ulong(blocks_per_token) +
                               ulong(kv_head * blocks_per_head + q8_block);
            device const hy3_block_q8_0 &vb = vc[base];
            const float vd = float(vb.d);
            const float4 vv = float4(float(vb.qs[q8_lane4]),
                                     float(vb.qs[q8_lane4 + 1u]),
                                     float(vb.qs[q8_lane4 + 2u]),
                                     float(vb.qs[q8_lane4 + 3u])) * vd;
            tile_acc = fma(vv, tile_weight[c], tile_acc);
        }

        local_sum = fma(local_sum, alpha, tile_sum);
        acc = fma(acc, alpha, tile_acc);
        local_max = next_max;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong partial_stride = ulong(args.head_dim + 2u);
    const ulong dst = ulong(group_id) * partial_stride;
    if (lane == 0u) {
        scratch[dst] = local_max;
        scratch[dst + 1u] = local_sum;
    }
    const ulong out4 = dst + 2u + ulong(lane) * 4u;
    scratch[out4] = acc.x;
    scratch[out4 + 1u] = acc.y;
    scratch[out4 + 2u] = acc.z;
    scratch[out4 + 3u] = acc.w;
}

kernel void kernel_hy3_gqa_prefill_q8_0_reduce(
        constant hy3_kv_args &args [[buffer(0)]],
        device const float   *scratch [[buffer(1)]],
        device float         *out [[buffer(2)]],
        uint query_index [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint head = query_index % args.n_head;
    const uint token = query_index / args.n_head;
    if (token >= args.n_tokens || tid >= args.head_dim) return;
    threadgroup float weights[32];
    threadgroup float normalizer;
    const ulong partial_stride = ulong(args.head_dim + 2u);

    if (tid < 32u) {
        float m = -INFINITY;
        const ulong src = ulong(query_index * args.nwg + tid) * partial_stride;
        if (tid < args.nwg) m = scratch[src];
        const float global_max = simd_max(m);
        float weighted_sum = 0.0f;
        float weight = 0.0f;
        if (tid < args.nwg && isfinite(m)) {
            weight = exp(m - global_max);
            weighted_sum = scratch[src + 1u] * weight;
        }
        if (tid < args.nwg) weights[tid] = weight;
        const float total = simd_sum(weighted_sum);
        if (tid == 0u) normalizer = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint iwg = 0u; iwg < args.nwg; ++iwg) {
        const ulong src = ulong(query_index * args.nwg + iwg) * partial_stride;
        acc += scratch[src + 2u + tid] * weights[iwg];
    }
    out[(token * args.n_head + head) * args.head_dim + tid] = acc / normalizer;
}
