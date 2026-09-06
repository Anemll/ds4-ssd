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

// F32 absorbed attention via 8x8 SIMD-group matrix operations. The independent
// head/key tiles share the latent KV row; no half activation conversion occurs.
kernel void kernel_hy4_attention_qk_sg_f32(
        constant ds4_metal_args_glm52_attention_decode &a [[buffer(0)]],
        device const float *qa [[buffer(1)]], device const float *qr [[buffer(2)]],
        device const float *kv [[buffer(3)]], device const float *pe [[buffer(4)]],
        device float *scores [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint h0=group.x*8u,p0=group.y*32u;
    threadgroup float qs[8*64],ks[32*64],dot[8*32];
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<576u;base+=64u) {
        for(uint i=tid;i<8u*64u;i+=128u) {
            uint h=h0+i/64u,d=base+i%64u;
            qs[i]=h<a.n_head ? (d<512u ? qa[ulong(h)*512u+d] : qr[ulong(h)*256u+192u+d-512u]) : 0.0f;
        }
        for(uint i=tid;i<32u*64u;i+=128u) {
            uint p=p0+i/64u,d=base+i%64u;
            ks[i]=p<a.n_past ? (d<512u ? kv[ulong(p)*512u+d] : pe[ulong(p)*64u+d-512u]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint db=0;db<8u;db++) {
            simdgroup_float8x8 q,k;
            simdgroup_load(q,qs+db*8u,64,0,false);
            simdgroup_load(k,ks+uint(sg)*8u*64u+db*8u,64,0,true);
            simdgroup_multiply_accumulate(acc,q,k,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,dot+uint(sg)*8u,32,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<8u*32u;i+=128u) {
        uint h=h0+i/32u,p=p0+i%32u;
        if(h<a.n_head && p<a.n_past) scores[ulong(h)*a.ctx+p]=dot[i]*a.scale;
    }
}

kernel void kernel_hy4_attention_softmax_f32(
        constant ds4_metal_args_glm52_attention_decode &a [[buffer(0)]],
        device float *scores [[buffer(5)]],device const float *sinks [[buffer(7)]],
        uint head [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float reduce[256];
    device float *row=scores+ulong(head)*a.ctx;
    float mx=sinks[head];
    for(uint p=tid;p<a.n_past;p+=256u) mx=max(mx,row[p]);
    reduce[tid]=mx;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(tid<stride) reduce[tid]=max(reduce[tid],reduce[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx=reduce[0];threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=tid==0 ? exp(sinks[head]-mx) : 0.0f;
    for(uint p=tid;p<a.n_past;p+=256u) {float w=exp(row[p]-mx);row[p]=w;sum+=w;}
    reduce[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for(uint stride=128;stride;stride>>=1) {
        if(tid<stride) reduce[tid]+=reduce[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv=reduce[0]>0.0f ? 1.0f/reduce[0] : 0.0f;
    for(uint p=tid;p<a.n_past;p+=256u) row[p]*=inv;
}

kernel void kernel_hy4_attention_av_sg_f32(
        constant ds4_metal_args_glm52_attention_decode &a [[buffer(0)]],
        device const float *kv [[buffer(3)]],device const float *scores [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint HC=8,KC=32,DV=32;
    const uint h0=group.x*HC,d0=group.y*DV;
    const uint sh=uint(sg)/(DV/8u),sd=uint(sg)%(DV/8u);
    threadgroup float ws[HC*KC],vs[KC*DV],dot[HC*DV];
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<a.n_past;base+=KC) {
        for(uint i=tid;i<HC*KC;i+=128u) {
            uint h=h0+i/KC,p=base+i%KC;
            ws[i]=h<a.n_head && p<a.n_past ? scores[ulong(h)*a.ctx+p] : 0.0f;
        }
        for(uint i=tid;i<KC*DV;i+=128u) {
            uint p=base+i/DV,d=d0+i%DV;
            vs[i]=p<a.n_past ? kv[ulong(p)*512u+d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint db=0;db<KC/8u;db++) {
            simdgroup_float8x8 w,v;
            simdgroup_load(w,ws+sh*8u*KC+db*8u,KC,0,false);
            simdgroup_load(v,vs+db*8u*DV+sd*8u,DV,0,false);
            simdgroup_multiply_accumulate(acc,w,v,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,dot+sh*8u*DV+sd*8u,DV,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<HC*DV;i+=128u) {
        uint h=h0+i/DV,d=d0+i%DV;
        if(h<a.n_head) out[ulong(h)*512u+d]=dot[i];
    }
}

// Native HY4 one-token sigmoid router. Preserve the generic 256-element
// bitonic network, including its equal-score ordering, and the active-eight
// SIMD reduction used by kernel_sum_rows_f32_f32.
kernel void kernel_hy4_router_one(
        constant uint &has_bias [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        device const float *bias [[buffer(2)]],
        device float *probs [[buffer(3)]],
        device int32_t *selected [[buffer(4)]],
        device float *weights [[buffer(5)]],
        uint tid [[thread_position_in_threadgroup]]) {
#pragma clang fp reassociate(off) contract(off)
    threadgroup float probability[256], score[256];
    threadgroup int32_t order[256];
    const float p = 1.0f / (1.0f + exp(-logits[tid]));
    probability[tid] = p; probs[tid] = p;
    score[tid] = has_bias ? p + bias[tid] : p;
    order[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k=2; k<=256; k*=2) {
        for (uint j=k/2; j>0; j/=2) {
            const uint other=tid^j;
            if (other>tid) {
                const float a=score[order[tid]], b=score[order[other]];
                if (((tid&k)==0 && a<b) || ((tid&k)!=0 && a>b)) {
                    const int32_t tmp=order[tid];
                    order[tid]=order[other]; order[other]=tmp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid<8) {
        const float w=probability[order[tid]];
        const float sum=simd_sum(0.0f+w);
        const float denominator=clamp(sum,6.103515625e-5f,INFINITY);
        const float normalized=w/denominator;
        selected[tid]=order[tid];
        weights[tid]=normalized*2.827f;
    }
}

// Native HY4 DSA uses original (tail-RoPE) GGUF channels and F32 index keys.
// LayerNorm epsilon matches SGLang's HY4 Indexer/LayerNorm default (1e-6).
kernel void kernel_hy4_index_norm(
        device float *out [[buffer(0)]], device const float *x [[buffer(1)]],
        device const float *weight [[buffer(2)]], device const float *bias [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[4];
    float v=x[tid];
    float sum=simd_sum(v);
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=(sums[0]+sums[1]+sums[2]+sums[3])*(1.0f/128.0f);
    float d=v-mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(d*d);
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=(sums[0]+sums[1]+sums[2]+sums[3])*(1.0f/128.0f);
    out[tid]=(d*rsqrt(var+1e-6f))*weight[tid]+bias[tid];
}
struct hy4_gather_args { uint live, count; };
kernel void kernel_hy4_gather_kv(
        constant hy4_gather_args &a [[buffer(0)]],
        device float *kv_out [[buffer(1)]], device float *pe_out [[buffer(2)]],
        device const float *kv [[buffer(3)]], device const float *pe [[buffer(4)]],
        device const uint *selected [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if(i>=a.count*576u) return;
    uint row=i/576u, d=i%576u, src=selected[row];
    // Invalid indices are poisoned, never used to address a cache. The runtime
    // only supplies indices generated by top-k over exactly [0, live).
    if(d<512u) kv_out[ulong(row)*512u+d]=src<a.live ? kv[ulong(src)*512u+d] : NAN;
    else pe_out[ulong(row)*64u+d-512u]=src<a.live ? pe[ulong(src)*64u+d-512u] : NAN;
}
