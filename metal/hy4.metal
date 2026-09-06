// HY4 sidecar matvec: exact quantized values, F32 activations/accumulation.
// STQ1_0 layout and codebook: llama.cpp @ 34cccef (MIT). IQ codebooks and
// block layouts come from the existing DS4 MoE source concatenated before this.
static constant uchar hy4_stq1_codebook[32] = {
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
};
struct hy4_block_stq1_0 { uchar qs[32]; uchar sign[8]; half d; };
struct ds4_metal_args_hy4_quant {
    uint in_dim, out_dim, type, pad;
    ulong row_bytes;
};
static inline uint hy4_u32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
kernel void kernel_hy4_quant_matvec(
        constant ds4_metal_args_hy4_quant &a [[buffer(0)]],
        device const uchar *weights [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= a.out_dim) return;
    device const uchar *w = weights + ulong(row) * a.row_bytes;
    float sum = 0.f;
    if (a.type == 43) {
        // Preserve the source STQ lane grouping and accumulation order.
        device const hy4_block_stq1_0 *b = (device const hy4_block_stq1_0 *)w;
        for (uint ib = 0; ib < a.in_dim / 256; ++ib) {
            float local = 0.f;
            for (uint g = lane; g < 64; g += 32) {
                const uint code = (b[ib].qs[g/2] >> (4*(g&1))) & 15;
                const uint sign = (b[ib].sign[g/8] >> (g&7)) & 1;
                const uint pack = hy4_stq1_codebook[(sign<<4)|code];
                for (uint p = 0; p < 4; ++p) {
                    const int q = int((pack >> (2*p)) & 3) - 1;
                    local += x[ib*256 + (g/16)*64 + g%16 + p*16] * q;
                }
            }
            sum += float(b[ib].d) * local;
        }
    } else {
        const uint block_bytes = a.type == 16 ? 66 : a.type == 18 ? 98 : 136;
        for (uint i = lane; i < a.in_dim; i += 32) {
            device const uchar *b = w + ulong(i/256) * block_bytes;
            const float d = float(*(device const half *)b);
            const uint sub = (i%256)/32, j = i%32;
            float value;
            if (a.type == 16 || a.type == 18) {
                device const uchar *q = b + 2 + sub*8;
                const uint aux = hy4_u32(a.type == 16 ? q+4 : b+66+sub*4);
                const float ds = d * (0.5f + (aux>>28)) * (a.type == 16 ? 0.25f : 0.5f);
                const uint group = j/8, v = j%8;
                const uint signs = ds4_metal_ksigns_iq2xs[(aux >> (7*group)) & 127];
                const uint grid = a.type == 16 ?
                    uint((ds4_metal_iq2xxs_grid[q[group]] >> (8*v)) & 255) :
                    (ds4_metal_iq3xxs_grid[q[2*group+v/4]] >> (8*(v%4))) & 255;
                value = ds * grid * (signs & (1u<<v) ? -1.f : 1.f);
            } else {
                const uint hi = uint(b[2]) | (uint(b[3])<<8);
                const uint lo = (b[4+sub/2] >> (4*(sub&1))) & 15;
                const int scale = int(lo | (((hi >> (2*sub)) & 3)<<4)) - 32;
                const uint q = (b[8+sub*16+j%16] >> (4*(j/16))) & 15;
                value = (d * scale) * ds4_metal_kvalues_iq4nl_f[q];
            }
            sum += value * x[i];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) out[row] = total;
}

// Source-equivalent full causal attention, bounded to 2048 keys by the HY4
// session loader. Same absorbed dimensions as GLM; HY4 adds softmax sinks.
kernel void kernel_hy4_attention_decode(
        constant ds4_metal_args_glm52_attention_decode & args [[buffer(0)]],
        device const float * q_abs     [[buffer(1)]],
        device const float * q_raw     [[buffer(2)]],
        device const float * kv_cache  [[buffer(3)]],
        device const float * kpe_cache [[buffer(4)]],
        device       float * scores    [[buffer(5)]],
        device       float * out       [[buffer(6)]],
        device const float * sinks [[buffer(7)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.n_past == 0 || args.n_past > args.ctx) {
        return;
    }

    threadgroup float reduce[256];
    device float * head_scores = scores + (uint64_t)head * args.ctx;
    float local_max = sinks[head];
    device const float * qh = q_abs + (uint64_t)head * 512u;
    device const float * qpe = q_raw + (uint64_t)head * 256u + 192u;

    for (uint32_t p = tid; p < args.n_past; p += ntg) {
        device const float * kv = kv_cache + (uint64_t)p * 512u;
        device const float * kp = kpe_cache + (uint64_t)p * 64u;
        float s = 0.0f;
        for (uint32_t i = 0; i < 512u; i++) {
            s += qh[i] * kv[i];
        }
        for (uint32_t i = 0; i < 64u; i++) {
            s += qpe[i] * kp[i];
        }
        s *= args.scale;
        head_scores[p] = s;
        local_max = max(local_max, s);
    }

    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = max(reduce[tid], reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];
    // All SIMD groups must consume the maximum before the shared reduction
    // array is reused for denominator sums (some lanes have no keys).
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The sink contributes probability mass, but has no value vector.
    float local_sum = tid == 0 ? exp(sinks[head] - max_score) : 0.0f;
    for (uint32_t p = tid; p < args.n_past; p += ntg) {
        const float w = exp(head_scores[p] - max_score);
        head_scores[p] = w;
        local_sum += w;
    }
    reduce[tid] = local_sum;
    // Every output lane subsequently reads all weights in device scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;

    for (uint32_t d = tid; d < 512u; d += ntg) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < args.n_past; p++) {
            acc += (head_scores[p] * inv_sum) * kv_cache[(uint64_t)p * 512u + d];
        }
        out[(uint64_t)head * 512u + d] = acc;
    }
}


// Keep the HY4 gate and post-down reduction on the existing command stream.
// These pragmas are local to each kernel: other DS4 kernels keep their math mode.
kernel void kernel_hy4_sigmoid_mul(
        constant uint &n [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const float *gate [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint j [[thread_position_in_grid]]) {
#pragma clang fp reassociate(off) contract(off)
    if (j >= n) return;
    const float probability = precise::divide(1.0f, 1.0f + precise::exp(-gate[j]));
    out[j] = x[j] * probability;
}

kernel void kernel_hy4_weighted_sum8(
        constant uint &n [[buffer(0)]],
        device const float *down [[buffer(1)]],
        device const float *weights [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint j [[thread_position_in_grid]]) {
#pragma clang fp reassociate(off) contract(off)
    if (j >= n) return;
    // Same expert order and separately rounded F32 multiply/add as the CPU
    // oracle. Volatile also prevents loop contraction in a fast-math library.
    volatile float sum = down[j] * weights[0];
    for (uint k = 1; k < 8; ++k) {
        volatile float product = down[ulong(k) * n + j] * weights[k];
        sum = sum + product;
    }
    out[j] = sum;
}
