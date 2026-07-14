// DS4 fused-dequant NAX int8 MoE matmul kernels.
//
// Compiled as a separate MTLLanguageVersion4_0 library (MetalPerformancePrimitives
// + tensor_ops are not available in the default-options main library).
//
// These fuse the iq2_xxs (gate/up) and q2_K (down) weight dequant directly into a
// NAX matmul2d: per K-tile (256 = QK_K) the weight tile is dequantized into
// threadgroup memory and fed to matmul2d, so the dequantized int8 weights never
// round-trip through device memory.  C = A(int8) * dequant_i8(W) accumulated in an
// int32 cooperative_tensor across K-tiles (multiply_accumulate).  Validated bit-exact
// and ~4x faster than dequant-to-global + i8xi8 matmul in moe-batch-bench/nax_fused_probe.m.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define QK_K 256

// Autotune knobs (injected via MTLCompileOptions.preprocessorMacros from env at
// library-build time; see ds4_gpu_ensure_nax_fused_library). Defaults preserve the
// shipped behavior so an unset env is a no-op.
#ifndef DS4_IDX_RELAXED
#define DS4_IDX_RELAXED false   // indexer matmul2d relaxed_precision (env DS4_GPU_INDEXER_RELAXED)
#endif
#ifndef DS4_IDX_WALK
#define DS4_IDX_WALK 0          // indexer Morton/Z-order tile walk (env DS4_GPU_INDEXER_WALK)
#endif
#ifndef DS4_DENSE_WALK
#define DS4_DENSE_WALK 0        // dense Morton/Z-order tile walk (env DS4_GPU_DENSE_WALK)
#endif
#ifndef DS4_ATTN_NR1
#define DS4_ATTN_NR1 128        // attn_out O-proj token tile (env DS4_GPU_ATTN_NR1)
#endif
#ifndef DS4_ATTN_NK
#define DS4_ATTN_NK 32          // attn_out O-proj K-tile (env DS4_GPU_ATTN_NK)
#endif

// Deinterleave a linear threadgroup id into 2D tile coords (Morton/Z-order inverse).
inline void ds4nf_morton2d(uint lin, thread uint &tx, thread uint &ty) {
    tx = 0; ty = 0;
    for (uint b = 0; b < 16u; b++) { tx |= ((lin >> (2u*b)) & 1u) << b; ty |= ((lin >> (2u*b+1u)) & 1u) << b; }
}

constant uchar ds4nf_ksigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant ulong ds4nf_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

struct block_iq2_xxs { half d; ushort qs[QK_K/8]; };
struct block_q2_K { uchar scales[QK_K/16]; uchar qs[QK_K/4]; half d; half dmin; };
struct block_q8_0 { half d; char qs[32]; };

// Dense Q8_0 NAX matmul, ported from antirez kernel_mul_mm_mpp_direct_rhs (Q8_0).
// C[tokens x out] = activation[tokens x in](f32, read DIRECTLY from device, no staging)
// * dequant(W[out x in] Q8_0 -> half, staged in threadgroup). matmul2d float x half ->
// float, NK=32 K-tile, weight-major 64(out) x 128(token) tile. Out is row-major
// [tokens x out] (dst[token*out + o]), matching the simdgroup mul_mm output.
struct ds4_mm_args {
    int32_t ne00, ne02; uint64_t nb01, nb02, nb03; int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13; int32_t ne0, ne1; int16_t r2, r3;
};
inline void ds4_deq_q8(device const block_q8_0 *xb, short il, thread half4x4 &reg) {
    device const char *qs = (device const char *)xb->qs; const float d = (float)xb->d;
    for (int i = 0; i < 16; i++) reg[i/4][i%4] = (half)((float)qs[i + 16*il] * d);
}
kernel void ds4_dense_q8_nax(
        constant ds4_mm_args &args [[buffer(0)]],
        device const char *srcA [[buffer(1)]],   // weight Q8_0 [out x in]
        device const char *srcB [[buffer(2)]],   // activation f32 [tokens x in]
        device       char *dst  [[buffer(3)]],   // out f32 [tokens x out]
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 64, NK = 32, NL = NK/16, NUM_THREADS = 128;
    const int K = args.ne00, M = args.ne0, N = args.ne1;
    const int im = tgpig.z, i12 = im % args.ne12, i13 = im / args.ne12;
#if DS4_DENSE_WALK
    uint _tx, _ty; ds4nf_morton2d(tgpig.x, _tx, _ty);
    const uint _gx = ((uint)N + NR1 - 1u) / NR1;
    const uint _gy = ((uint)M + NR0 - 1u) / NR0;
    if (_tx >= _gx || _ty >= _gy) return;
    const int r1 = (int)_tx * NR1, r0 = (int)_ty * NR0;
#else
    const int r0 = tgpig.y * NR0, r1 = tgpig.x * NR1;
#endif
    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (int lk = 0; lk < K; lk += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work / NL, kc = work % NL, kpos = lk + kc*16; const short kb = kc*16;
            if (r0 + row < M) {
                const int bidx = kpos / (16*2); const short il = (kpos/16) % 2;
                device const block_q8_0 *rp = (device const block_q8_0 *)(srcA + args.nb01*(r0+row) + offset0);
                half4x4 t; ds4_deq_q8(rp + bidx, il, t);
                for (short i = 0; i < 16; i++) sa[row*NK + kb + i] = (kpos+i < K) ? t[i/4][i%4] : (half)0;
            } else {
                for (short i = 0; i < 16; i++) sa[row*NK + kb + i] = (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0); auto mB = tB.slice(lk, r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *db = (device float *)dst + im*N*M;
    auto tD = tensor(db, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

// Case E (dspark-attn) — build the verifier key stream = ring-resolved raw window
// ++ compressed cache, contiguous [n_keys x dim] (n_keys = n_raw + n_comp), so the
// NAX attention sees the same keys the ALU verifier does (raw SWA + compressed).
struct ds4_dspark_gather_args { int32_t n_raw; int32_t n_comp; int32_t dim; int32_t raw_cap; int32_t raw_start; };
kernel void ds4_dspark_nax_gather(
        constant ds4_dspark_gather_args &args [[buffer(0)]],
        device const float *raw  [[buffer(1)]],   // [raw_cap x dim] ring buffer
        device const float *comp [[buffer(2)]],    // [n_comp x dim]
        device       float *out  [[buffer(3)]],     // [(n_raw+n_comp) x dim]
        uint tid [[thread_position_in_grid]]) {
    const uint nk = (uint)(args.n_raw + args.n_comp);
    const uint total = nk * (uint)args.dim;
    if (tid >= total) return;
    const int key = (int)(tid / (uint)args.dim);
    const int d   = (int)(tid - (uint)key * (uint)args.dim);
    if (key < args.n_raw) {
        const int rr = (args.raw_start + key) % args.raw_cap;   // resolve ring
        out[(uint64_t)key * args.dim + d] = raw[(uint64_t)rr * args.dim + d];
    } else {
        const int cc = key - args.n_raw;
        out[(uint64_t)key * args.dim + d] = comp[(uint64_t)cc * args.dim + d];
    }
}

struct ds4_dspark_gather_topk_args {
    int32_t n_raw;
    int32_t top_k;
    int32_t n_tokens;
    int32_t dim;
    int32_t raw_cap;
    int32_t raw_start;
};
kernel void ds4_dspark_nax_gather_topk(
        constant ds4_dspark_gather_topk_args &args [[buffer(0)]],
        device const float   *raw      [[buffer(1)]], // [raw_cap x dim] ring buffer
        device const float   *comp     [[buffer(2)]], // [n_comp x dim]
        device const int32_t *selected [[buffer(3)]], // [n_tokens x top_k]
        device       float   *out      [[buffer(4)]], // [n_raw + n_tokens*top_k, dim]
        uint tid [[thread_position_in_grid]]) {
    const uint n_comp_stream = (uint)args.top_k * (uint)args.n_tokens;
    const uint nk = (uint)args.n_raw + n_comp_stream;
    const uint total = nk * (uint)args.dim;
    if (tid >= total) return;
    const int key = (int)(tid / (uint)args.dim);
    const int d   = (int)(tid - (uint)key * (uint)args.dim);
    if (key < args.n_raw) {
        const int rr = (args.raw_start + key) % args.raw_cap;
        out[(uint64_t)key * args.dim + d] = raw[(uint64_t)rr * args.dim + d];
    } else {
        const int rel = key - args.n_raw;
        const int tok = rel / args.top_k;
        const int k   = rel - tok * args.top_k;
        const int cc  = selected[(uint64_t)tok * args.top_k + k];
        out[(uint64_t)key * args.dim + d] = cc >= 0 ?
            comp[(uint64_t)cc * args.dim + d] : 0.0f;
    }
}

// Case E (dspark-attn) — WIP/UNTESTED. NAX matmul2d Q@K^T scores for the DSpark
// verifier's batched attention. C[m][key] = Q[m] . K[key], contraction over
// head_dim D, where the M dimension flattens (head, token): m = head*n_tokens + tok.
// K is head-independent (MLA latent) and is read directly from device as K^T via
// strides (element[d][key] = K[key*row + d]); Q is staged per M-row to threadgroup
// as half. Directly adapted from ds4_dense_q8_nax (same matmul2d descriptor /
// operand order / transpose flags) — only A becomes a half-staged activation and B
// becomes the shared K^T activation. Scale + softmax + A@V are applied separately;
// this kernel is the E1 throughput microbench for Q@K^T on the tensor units.
struct ds4_dspark_qk_args {
    int32_t  D;                 // head_dim (contraction K)
    int32_t  n_tokens;          // tokens per head
    int32_t  n_keys;            // N
    int32_t  n_head;            // M = n_head * n_tokens
    uint64_t q_token_stride;    // bytes between tokens in Q
    uint64_t q_head_stride;     // bytes between heads in Q
    uint64_t k_row_stride;      // bytes between keys in K
    uint64_t score_row_stride;  // floats between M-rows in scores (>= n_keys)
};
kernel void ds4_dspark_nax_qk_scores(
        constant ds4_dspark_qk_args &args [[buffer(0)]],
        device const char *q       [[buffer(1)]],   // f32, indexed by token+head strides
        device const char *k       [[buffer(2)]],   // f32 [n_keys x D]
        device       char *scores  [[buffer(3)]],   // f32 [M x n_keys] (M = head*n_tokens+tok)
        threadgroup  char *shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 64, NK = 32, NUM_THREADS = 128;
    const int K = args.D;
    const int M = args.n_head * args.n_tokens;
    const int N = args.n_keys;
    const int r0 = tgpig.y * NR0;   // flattened (head,token) tile
    const int r1 = tgpig.x * NR1;   // key tile

    threadgroup half *sa = (threadgroup half *)shmem;   // [NR0 x NK]
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    // K^T as a device tensor: element[d][key] = k[key*row + d]
    // (non-const: matmul2d cooperative-tensor operands cannot be const-qualified)
    device float *ptrK = (device float *)k;
    const int strideK = (int)(args.k_row_stride / sizeof(float));
    auto tB = tensor(ptrK, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideK}));

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    for (int lk = 0; lk < K; lk += NK) {
        for (int work = tiitg; work < NR0*NK; work += NUM_THREADS) {
            const int row = work / NK, kc = work % NK;
            half v = (half)0;
            const int m = r0 + row;
            if (m < M) {
                const int tok  = m / args.n_head;       // M = tok*n_head + head (row-major heads)
                const int head = m - tok * args.n_head;
                device const float *qrow = (device const float *)(q +
                    (uint64_t)tok  * args.q_token_stride +
                    (uint64_t)head * args.q_head_stride);
                if (lk + kc < K) v = (half)qrow[lk + kc];
            }
            sa[row*NK + kc] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(lk, r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *db = (device float *)scores;
    auto tD = tensor(db, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

// Case E (dspark-attn) E2a — row softmax over scores. scores are column-major
// [key*M + m] (as written by ds4_dspark_nax_qk_scores). Applies scale and softmax
// per M-row, writes probs row-major [m*n_keys + key] for the P@V stage. One thread
// per M-row (n_keys small for the verifier). E2a: no sinks/causal mask yet.
struct ds4_dspark_sm_args { int32_t M; int32_t n_keys; float scale; int32_t n_head; int32_t n_raw; int32_t comp_stride; };
kernel void ds4_dspark_nax_softmax(
        constant ds4_dspark_sm_args &args [[buffer(0)]],
        device const float *scores   [[buffer(1)]],   // [key*M + m]
        device       float *probs    [[buffer(2)]],   // [m*n_keys + key]
        device const int   *raw_cnt  [[buffer(3)]],    // per-token valid raw count
        device const int   *comp_cnt [[buffer(4)]],     // per-token valid comp count
        device const int   *raw_off  [[buffer(5)]],      // per-token raw window offset in stream
        uint tid [[thread_position_in_grid]]) {
    const int m = (int)tid;
    if (m >= args.M) return;
    const int N = args.n_keys, M = args.M;
    // Stream:
    //   dense mode: keys [0..n_raw) = raw union, [n_raw..n_raw+n_comp) = compressed.
    //   top-k mode: keys [n_raw + tok*top_k .. +comp_cnt) hold this token's selected comp rows.
    const int tok = m / args.n_head;
    const int vraw  = raw_cnt[tok];
    const int vcomp = comp_cnt[tok];
    const int off   = raw_off[tok];
    const int n_raw = args.n_raw;
    const int cbase = n_raw + (args.comp_stride > 0 ? tok * args.comp_stride : 0);
    float mx = -INFINITY;
    for (int key = 0; key < N; key++) {
        const bool valid = (key >= off && key < off + vraw) || (key >= cbase && key < cbase + vcomp);
        if (valid) mx = max(mx, scores[(uint64_t)key * M + m] * args.scale);
    }
    float sum = 0.0f;
    for (int key = 0; key < N; key++) {
        const bool valid = (key >= off && key < off + vraw) || (key >= cbase && key < cbase + vcomp);
        const float e = valid ? exp(scores[(uint64_t)key * M + m] * args.scale - mx) : 0.0f;
        probs[(uint64_t)m * N + key] = e;
        sum += e;
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int key = 0; key < N; key++) probs[(uint64_t)m * N + key] *= inv;
}

// Case E (dspark-attn) E2a — P@V on the tensor units. C[M x out_dim] =
// probs[M x n_keys] @ V[n_keys x out_dim]. A = probs staged to threadgroup (half),
// B = V read directly from device as [K=n_keys x N=out_dim] (non-const). Mirrors
// ds4_dense_q8_nax. E2a stores out column-major [n*M + m] (layout fixed in E2b).
struct ds4_dspark_pv_args {
    int32_t  n_keys;        // contraction K
    int32_t  M;             // rows
    int32_t  out_dim;       // N
    uint64_t v_row_stride;  // bytes per key in V
    uint64_t p_row_stride;  // floats per M-row in probs (= n_keys)
};
kernel void ds4_dspark_nax_pv(
        constant ds4_dspark_pv_args &args [[buffer(0)]],
        device       float *probs  [[buffer(1)]],   // [m*n_keys + key] (non-const for matmul2d)
        device       char  *v      [[buffer(2)]],    // f32 [n_keys x out_dim]
        device       char  *out    [[buffer(3)]],    // f32 [M x out_dim]
        threadgroup  char  *shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 64, NK = 32, NUM_THREADS = 128;
    const int K = args.n_keys, M = args.M, N = args.out_dim;
    const int r0 = tgpig.y * NR0;   // M tile
    const int r1 = tgpig.x * NR1;   // out_dim tile

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    // V is pre-transposed to vt[dim][key] so the contraction (key) is contiguous:
    // element[k=key][n=dim] = vt[n*n_keys + k] -> strides {1, K}.
    device float *ptrV = (device float *)v;  // vt[dim x n_keys]
    auto tB = tensor(ptrV, dextents<int32_t, 2>(K, N), array<int, 2>({1, K}));

    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    for (int lk = 0; lk < K; lk += NK) {
        for (int work = tiitg; work < NR0*NK; work += NUM_THREADS) {
            const int row = work / NK, kc = work % NK;
            half pv = (half)0;
            if (r0 + row < M && lk + kc < K) {
                pv = (half)probs[(uint64_t)(r0+row) * args.p_row_stride + (lk + kc)];
            }
            sa[row*NK + kc] = pv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(lk, r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *db = (device float *)out;
    auto tD = tensor(db, dextents<int32_t, 2>(M, N), array<int, 2>({1, M})); // column-major out[dim*M+m] (QK/dense store layout)
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

// Case E (dspark-attn) — transpose the column-major P@V output src[dim*M + m] into
// the row-major attention heads dst[m*out_dim + d] (m = tok*n_head+head).
struct ds4_dspark_ot_args { int32_t M; int32_t out_dim; };
kernel void ds4_dspark_nax_otranspose(
        constant ds4_dspark_ot_args &args [[buffer(0)]],
        device const float *src [[buffer(1)]],   // [d*M + m]
        device       float *dst [[buffer(2)]],    // [m*out_dim + d]
        uint tid [[thread_position_in_grid]]) {
    const uint total = (uint)args.M * (uint)args.out_dim;
    if (tid >= total) return;
    const int m = (int)(tid / (uint)args.out_dim);
    const int d = (int)(tid - (uint)m * (uint)args.out_dim);
    dst[(uint64_t)m * args.out_dim + d] = src[(uint64_t)d * args.M + m];
}

// Case E (dspark-attn) — transpose V[key][dim] -> vt[dim][key] so the P@V matmul2d
// contracts over a contiguous key dimension. vt[d*n_keys + key] = v[key*v_stride_f + d].
struct ds4_dspark_vt_args { int32_t n_keys; int32_t dim; int32_t v_stride_f; };
kernel void ds4_dspark_nax_vtranspose(
        constant ds4_dspark_vt_args &args [[buffer(0)]],
        device const float *v  [[buffer(1)]],   // [key*v_stride_f + d]
        device       float *vt [[buffer(2)]],    // [d*n_keys + key]
        uint tid [[thread_position_in_grid]]) {
    const uint total = (uint)args.n_keys * (uint)args.dim;
    if (tid >= total) return;
    const int key = (int)(tid / (uint)args.dim);
    const int d   = (int)(tid - (uint)key * (uint)args.dim);
    vt[(uint64_t)d * args.n_keys + key] = v[(uint64_t)key * args.v_stride_f + d];
}

// Case E (dspark-attn) — simple/correct AV reference (debug; bypasses the matmul2d
// P@V to isolate operand-layout bugs). out[m][d] = sum_key probs[m][key]*V[key][d].
// One thread per (m,d) output element. Row-major out = heads[tok][head][dim].
struct ds4_dspark_av_args { int32_t M; int32_t n_keys; int32_t out_dim; int32_t v_stride_f; int32_t p_stride_f; };
kernel void ds4_dspark_nax_av_simple(
        constant ds4_dspark_av_args &args [[buffer(0)]],
        device const float *probs [[buffer(1)]],   // [m*p_stride_f + key]
        device const float *v     [[buffer(2)]],    // [key*v_stride_f + d]
        device       float *out   [[buffer(3)]],     // [m*out_dim + d]
        uint tid [[thread_position_in_grid]]) {
    const uint total = (uint)args.M * (uint)args.out_dim;
    if (tid >= total) return;
    const int m = (int)(tid / (uint)args.out_dim);
    const int d = (int)(tid - (uint)m * (uint)args.out_dim);
    float acc = 0.0f;
    for (int key = 0; key < args.n_keys; key++) {
        acc += probs[(uint64_t)m * args.p_stride_f + key] * v[(uint64_t)key * args.v_stride_f + d];
    }
    out[(uint64_t)m * args.out_dim + d] = acc;
}

// Case E (dspark-attn) FUSED — QK(matmul2d / Neural Accelerator) + masked softmax(ALU)
// + PV(ALU) in ONE dispatch. The win on M5 is NA∥ALU co-issue: with this as a single
// kernel, while some threadgroups run the QK matmul2d on the Neural Accelerator, others
// run the softmax/PV on the ALU — the two pipelines stay busy concurrently (impossible
// across separate barrier-separated dispatches). Each TG owns a full M-tile (NR0 rows)
// across ALL keys so its softmax is self-contained; key-tiles are looped internally.
// Scores/probs round-trip through device scratch (cheap, n_keys small) but it is still
// one dispatch, so the scheduler overlaps NA and ALU work across in-flight TGs.
struct ds4_dspark_flash_args {
    int32_t  D;               // head_dim contraction (512)
    int32_t  n_tokens;
    int32_t  n_keys;          // N
    int32_t  n_head;
    int32_t  n_raw;           // boundary: keys [n_raw .. n_raw+comp) are compressed
    float    scale;
    uint64_t q_token_stride;  // bytes between tokens in Q
    uint64_t q_head_stride;   // bytes between heads in Q
    uint64_t k_row_stride;    // bytes between keys in stream (= D*4)
};
kernel void ds4_dspark_nax_flash(
        constant ds4_dspark_flash_args &args [[buffer(0)]],
        device const char  *q       [[buffer(1)]],   // f32, token+head strides
        device       char  *stream  [[buffer(2)]],   // f32 [n_keys x D] (non-const for matmul2d)
        device       float *scores  [[buffer(3)]],   // scratch [key*M + m]
        device       float *probs   [[buffer(4)]],   // scratch [m*n_keys + key]
        device       float *heads   [[buffer(5)]],   // out [m*D + d]
        device const int   *raw_cnt [[buffer(6)]],
        device const int   *comp_cnt[[buffer(7)]],
        device const int   *raw_off [[buffer(8)]],
        threadgroup char   *shmem   [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 32, NK = 32, NUM_THREADS = 128;
    const int K = args.D;
    const int M = args.n_head * args.n_tokens;
    const int N = args.n_keys;
    const int r0 = tgpig.y * NR0;   // this TG owns M-rows [r0 .. r0+NR0)

    threadgroup half *sa = (threadgroup half *)shmem;   // [NR0 x NK]
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrK = (device float *)stream;
    const int strideK = (int)(args.k_row_stride / sizeof(float));   // = D
    auto tB = tensor(ptrK, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideK}));
    auto tD = tensor(scores, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));

    // ---- Phase 1 (Neural Accelerator): QK over all key-tiles -> scores[key*M+m] ----
    for (int r1 = 0; r1 < N; r1 += NR1) {
        matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
        auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        for (int lk = 0; lk < K; lk += NK) {
            for (int work = tiitg; work < NR0*NK; work += NUM_THREADS) {
                const int row = work / NK, kc = work % NK;
                half v = (half)0;
                const int m = r0 + row;
                if (m < M) {
                    const int tok  = m / args.n_head;
                    const int head = m - tok * args.n_head;
                    device const float *qrow = (device const float *)(q +
                        (uint64_t)tok  * args.q_token_stride +
                        (uint64_t)head * args.q_head_stride);
                    if (lk + kc < K) v = (half)qrow[lk + kc];
                }
                sa[row*NK + kc] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto mA = tA.slice(0, 0);
            auto mB = tB.slice(lk, r1);
            mm.run(mB, mA, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 2 (ALU): masked softmax for this TG's rows -> probs[m*N+key] ----
    for (int row = tiitg; row < NR0; row += NUM_THREADS) {
        const int m = r0 + row;
        if (m >= M) continue;
        const int tok = m / args.n_head;
        const int vraw = raw_cnt[tok], vcomp = comp_cnt[tok], off = raw_off[tok], nraw = args.n_raw;
        float mx = -INFINITY;
        for (int key = 0; key < N; key++) {
            const bool valid = (key >= off && key < off + vraw) || (key >= nraw && key < nraw + vcomp);
            if (valid) mx = max(mx, scores[(uint64_t)key * M + m] * args.scale);
        }
        float sum = 0.0f;
        for (int key = 0; key < N; key++) {
            const bool valid = (key >= off && key < off + vraw) || (key >= nraw && key < nraw + vcomp);
            const float e = valid ? exp(scores[(uint64_t)key * M + m] * args.scale - mx) : 0.0f;
            probs[(uint64_t)m * N + key] = e;
            sum += e;
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int key = 0; key < N; key++) probs[(uint64_t)m * N + key] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- Phase 3 (ALU): PV -> heads[m*D+d] = sum_key probs[m][key]*stream[key][d] ----
    for (int w = tiitg; w < NR0*K; w += NUM_THREADS) {
        const int row = w / K, d = w - row * K;
        const int m = r0 + row;
        if (m >= M) continue;
        float acc = 0.0f;
        for (int key = 0; key < N; key++) {
            acc += probs[(uint64_t)m * N + key] * ptrK[(uint64_t)key * strideK + d];
        }
        heads[(uint64_t)m * K + d] = acc;
    }
}

// NA∥ALU in-kernel co-issue microbench. One kernel, 1-D grid; threadgroups are
// partitioned into NA (matmul2d / Neural Accelerator) and ALU (scalar f32 matvec)
// roles by mode: 0=all NA, 1=all ALU, 2=interleaved (even tg = NA, odd tg = ALU).
// NA: C[M x N] = A[M x K] @ B[K x N] (B stored [N x K] row-major, read {1,K}).
// ALU: O[rows x ncol] = X[rows x k] . W[ncol x k], pure scalar lanes, no matmul/simd.
// Both repeat inner-iters to inflate time. Measures whether the two pipelines overlap.
struct ds4_naalu_args {
    int32_t M, N, K;            // matmul2d dims
    int32_t na_iters, alu_iters;
    int32_t alu_rows, alu_ncol, alu_k, alu_tgs;
    int32_t mode;              // 0=NA, 1=ALU, 2=interleaved
};
kernel void ds4_naalu_bench(
        constant ds4_naalu_args &a [[buffer(0)]],
        device const float *A  [[buffer(1)]],   // [M x K]
        device const float *B  [[buffer(2)]],   // [N x K] (element[d][n] = B[n*K+d])
        device       float *C  [[buffer(3)]],    // [M x N]
        device const float *W  [[buffer(4)]],    // [ncol x k]
        device const float *X  [[buffer(5)]],    // [rows x k]
        device       float *O  [[buffer(6)]],     // [rows x ncol]
        threadgroup  char  *shmem [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 64, NK = 32, NUM_THREADS = 128;
    bool do_na; uint role_id;
    if (a.mode == 0)      { do_na = true;  role_id = tgpig; }
    else if (a.mode == 1) { do_na = false; role_id = tgpig; }
    else                  { do_na = ((tgpig & 1u) == 0u); role_id = tgpig >> 1; }

    if (do_na) {
        const int M = a.M, N = a.N, K = a.K;
        const int gx = N / NR1;                 // NA tiles across N
        const int gy = (M + NR0 - 1) / NR0;
        if (role_id >= (uint)(gx * gy)) return;
        const int tx = (int)(role_id % (uint)gx), ty = (int)(role_id / (uint)gx);
        const int r1 = tx * NR1, r0 = ty * NR0;
        threadgroup half *sa = (threadgroup half *)shmem;
        auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
        device float *ptrB = (device float *)B;
        auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, K}));
        matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
        auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (int it = 0; it < a.na_iters; it++) {
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
            for (int lk = 0; lk < K; lk += NK) {
                for (int work = tiitg; work < NR0*NK; work += NUM_THREADS) {
                    const int row = work / NK, kc = work % NK;
                    const int m = r0 + row;
                    half v = (half)0;
                    if (m < M && lk + kc < K) v = (half)A[(uint64_t)m * K + lk + kc];
                    sa[row*NK + kc] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mA = tA.slice(0, 0); auto mB = tB.slice(lk, r1);
                mm.run(mB, mA, cT);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        device float *db = (device float *)C;
        auto tD = tensor(db, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    } else {
        const int ncol = a.alu_ncol, kk = a.alu_k;
        if (role_id >= (uint)a.alu_tgs) return;
        const int rows_per = (a.alu_rows + a.alu_tgs - 1) / a.alu_tgs;
        const int base = (int)role_id * rows_per;
        const int total = rows_per * ncol;
        for (int it = 0; it < a.alu_iters; it++) {
            for (int o = tiitg; o < total; o += NUM_THREADS) {
                const int rr = base + o / ncol;
                const int cc = o % ncol;
                if (rr >= a.alu_rows) continue;
                float acc = 0.0f;
                device const float *xp = X + (uint64_t)rr * kk;
                device const float *wp = W + (uint64_t)cc * kk;
                for (int k = 0; k < kk; k++) acc += xp[k] * wp[k];
                O[(uint64_t)rr * ncol + cc] = acc;
            }
        }
    }
}

// Dense M=5 decision microbench: NA matmul2d (mode 0) vs well-occupied ALU
// ncol-split matvec (mode 1) at the verifier's tall-skinny dense shape.
// A=[M x K] activations (f32), B=[N x K] weights (element[d][n]=B[n*K+d]), C=[M x N].
struct ds4_dm5_args { int32_t M, N, K, iters, mode, ncol_tgs; };
kernel void ds4_dense_m5_bench(
        constant ds4_dm5_args &a [[buffer(0)]],
        device const float *A [[buffer(1)]],
        device const float *B [[buffer(2)]],
        device       float *C [[buffer(3)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 64, NK = 32, NUM_THREADS = 128;
    const int M = a.M, N = a.N, K = a.K;
    if (a.mode == 0) {
        // NA matmul2d over full N, M rows (M padded into one NR0 tile).
        const int gx = N / NR1;
        const int tx = (int)((uint)tgpig % (uint)gx), ty = (int)((uint)tgpig / (uint)gx);
        const int r1 = tx * NR1, r0 = ty * NR0;
        threadgroup half *sa = (threadgroup half *)shmem;
        auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
        device float *ptrB = (device float *)B;
        auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, K}));
        matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
        auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
        for (int it = 0; it < a.iters; it++) {
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
            for (int lk = 0; lk < K; lk += NK) {
                for (int work = tiitg; work < NR0*NK; work += NUM_THREADS) {
                    const int row = work / NK, kc = work % NK;
                    const int m = r0 + row;
                    half v = (half)0;
                    if (m < M && lk + kc < K) v = (half)A[(uint64_t)m * K + lk + kc];
                    sa[row*NK + kc] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                auto mA = tA.slice(0, 0); auto mB = tB.slice(lk, r1);
                mm.run(mB, mA, cT);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        device float *db = (device float *)C;
        auto tD = tensor(db, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    } else {
        // ALU ncol-split matvec: tgpig in [0, ncol_tgs); each TG does all M rows for
        // a slice of N columns (realistic full-occupancy ALU matvec, not row-starved).
        const int cols_per = N / a.ncol_tgs;
        const int c0 = (int)tgpig * cols_per;
        const int total = M * cols_per;
        for (int it = 0; it < a.iters; it++) {
            for (int o = tiitg; o < total; o += NUM_THREADS) {
                const int rr = o / cols_per;
                const int cc = c0 + (o % cols_per);
                float acc = 0.0f;
                device const float *xp = A + (uint64_t)rr * K;
                device const float *wp = B + (uint64_t)cc * K;
                for (int k = 0; k < K; k++) acc += xp[k] * wp[k];
                C[(uint64_t)rr * N + cc] = acc;
            }
        }
    }
}

static inline int8_t ds4nf_f2i8(float x, float qscale) {
    return int8_t(int(rint(clamp(x * qscale, -128.0f, 127.0f))));
}

// Dequant one iq2_xxs segment (seg in 0..15 -> 16 values) of block (row n) at k-block
// kb into the transposed threadgroup tile column nn.  Btile layout [localk*stride + nn].
inline void ds4nf_iq2_seg_stride(device const block_iq2_xxs *blk, uint seg, float qscale,
                                 threadgroup int8_t *Btile, uint nn, uint stride) {
    const uint ib32 = seg / 2u, lane = seg & 1u;
    device const ushort *q2 = blk->qs + 4u * ib32;
    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
    const uint col0 = seg * 16u;
    const ulong gv0 = ds4nf_iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
    const uchar s0 = ds4nf_ksigns[(aux32_s >> (14u * lane)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((s0 & (1u << j)) ? -1.0f : 1.0f);
        Btile[(col0 + j) * stride + nn] = ds4nf_f2i8(v, qscale);
    }
    const ulong gv1 = ds4nf_iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
    const uchar s1 = ds4nf_ksigns[(aux32_s >> (14u * lane + 7u)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((s1 & (1u << j)) ? -1.0f : 1.0f);
        Btile[(col0 + 8u + j) * stride + nn] = ds4nf_f2i8(v, qscale);
    }
}

inline void ds4nf_iq2_seg(device const block_iq2_xxs *blk, uint seg, float qscale,
                          threadgroup int8_t *Btile, uint nn) {
    ds4nf_iq2_seg_stride(blk, seg, qscale, Btile, nn, 32u);
}

inline void ds4nf_q2k_seg_stride(device const block_q2_K *blk, uint seg, float qscale,
                                 threadgroup int8_t *Btile, uint nn, uint stride) {
    device const uchar *q = blk->qs + 32u * (seg / 8u) + 16u * (seg & 1u);
    const uchar sc = blk->scales[seg];
    const uint il = (seg / 2u) & 3u;
    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);
    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);
    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;
    const float ml = float(blk->dmin) * float(sc >> 4);
    const uint col0 = seg * 16u;
    for (uint j = 0; j < 16u; j++) {
        Btile[(col0 + j) * stride + nn] = ds4nf_f2i8(dl * float(q[j] & mask) - ml, qscale);
    }
}

inline void ds4nf_q2k_seg(device const block_q2_K *blk, uint seg, float qscale,
                          threadgroup int8_t *Btile, uint nn) {
    ds4nf_q2k_seg_stride(blk, seg, qscale, Btile, nn, 32u);
}

// Grouped attention-output (O-proj low) NAX matmul, ported from antirez
// kernel_attn_out_low_q8_0_mpp_direct_rhs. Per OUT_GROUP: C[tokens x M] += act[tokens x K](f32,
// direct from device) * dequant(W[M x K] Q8_0 -> half). NR1=64 token tile, NK=32, direct-RHS.
struct ds4_mm_id_args {
    int32_t  ne00, ne02; uint64_t nb01, nb02, nb03; int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13; int32_t ne20, ne21, ne0, ne1; int16_t r2, r3;
};
kernel void ds4_attn_out_low_q8_nax(
        constant ds4_mm_id_args &args [[buffer(0)]],
        device const char *srcA [[buffer(1)]],
        device const char *srcB [[buffer(2)]],
        device       char *dst  [[buffer(3)]],
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = DS4_ATTN_NR1, NR0 = 64, NK = DS4_ATTN_NK, NL = NK/16, NUM_THREADS = 128;
    const int K = args.ne00, M = args.ne0, N = args.ne21, G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;
    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12 / sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL, kc = work%NL, kpos = loop_k + kc*16; const short kbase = kc*16;
            if (full_tile || r0 + row < M) {
                const int bidx = kpos/32; const short il = (kpos/16)%2;
                device const block_q8_0 *rp = (device const block_q8_0 *)(srcA + args.nb01*(r0+row) + group*args.nb02);
                half4x4 t; ds4_deq_q8(rp + bidx, il, t);
                for (short i = 0; i < 16; i++) sa[row*NK + kbase + i] = (full_tile || kpos+i < K) ? t[i/4][i%4] : (half)0;
            } else {
                for (short i = 0; i < 16; i++) sa[row*NK + kbase + i] = (half)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0); auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *dst_group = (device float *)dst + group*M;
    if (full_tile) {
        device float *dst_tile = dst_group + r0 + (uint64_t)r1*G*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, G*M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

// NAX indexer score matmul (the long-context prefill-slope dominator). Computes
// scores[token x comp] = sum_head relu(Q_head[token] . index_comp[comp]) * w[token][head] * scale
// over 64 heads (D=128), with a per-token top-k visibility mask. Each head's Q@K^T is a
// matmul2d half x half -> float; K (head-independent) is staged once into Btile transposed,
// Q is staged per head, results accumulate (relu*weight) into a float cooperative_tensor.
// Tile: 64 tokens (m) x 32 comp (n), K=128 (one tile). q/weights/index_comp/scores are float.
struct ds4_idx_args {
    uint n_comp, n_tokens, n_head, head_dim, pos0, ratio;
    ulong q_token_stride, q_head_stride, weights_token_stride, index_row_stride, score_token_stride;
    float scale;
};
// Tuned port of antirez ds4 kernel_dsv4_indexer_scores_nax: 16-token x 32-comp tile
// (his sweeps found 64-row slower), K-tiled at NK=32 with multiply_accumulate, results
// stored to a threadgroup buffer then relu/weighted/summed with a flat indexed loop
// (avoids per-element cooperative-tensor index lookups). transpose_right: C = K @ Q^T.
kernel void ds4_indexer_scores_nax(
        constant ds4_idx_args &args [[buffer(0)]],
        device const char *q [[buffer(1)]],
        device const char *weights [[buffer(2)]],
        device const char *index_comp [[buffer(3)]],
        device       char *scores [[buffer(4)]],
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

#if DS4_IDX_WALK
    uint _tx, _ty; ds4nf_morton2d(tgpig.x, _tx, _ty);
    const uint _gx = (args.n_comp + TN - 1u) / TN;
    const uint _gy = (args.n_tokens + (uint)TM - 1u) / (uint)TM;
    if (_tx >= _gx || _ty >= _gy) return;
    const uint c0 = _tx * TN;
    const uint t0 = _ty * TM;
#else
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;
#endif

    threadgroup half  *qtg = shared;               // [16][32]
    threadgroup half  *ktg = qtg + TM*NK;          // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [16][32], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint work = tid; work < TN*D; work += NUM_THREADS) {
        const uint cc = work / D;
        const uint d = work - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(krow[d]);
        }
        ktg[cc*D + d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq = tensor(qtg, dextents<int32_t, 2>(NK, TM));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TM, TN), array<int, 2>({1, TM}));

    matmul2d<
        matmul2d_descriptor(TN, TM, NK, false, true, DS4_IDX_RELAXED,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    for (uint head = 0; head < args.n_head; head++) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        for (uint loop_k = 0; loop_k < D; loop_k += NK) {
            for (uint work = tid; work < TM*NK; work += NUM_THREADS) {
                const uint r = work / NK;
                const uint k = work - r*NK;
                const uint token = t0 + r;
                half v = half(0.0f);
                if (token < args.n_tokens) {
                    device const float *qrow = (device const float *)(q +
                        (uint64_t)token * args.q_token_stride +
                        (uint64_t)head  * args.q_head_stride);
                    v = half(qrow[loop_k + k]);
                }
                qtg[r*NK + k] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto mq = tq.slice(0, 0);
            auto mk = tk.slice(loop_k, 0);
            mm.run(mk, mq, ct);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TM + r], 0.0f) * (w[head] * args.scale);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}

// Fused gate/up (iq2_xxs).  Indirect-dispatched per expert: grid.x = N tiles (NT),
// grid.y = M tiles (64); M = counts[expert].  A = gathered int8 acts [M x K] row-major,
// Wq = expert weight [N x K] iq2_xxs, C = int32 [M x N] row-major.
#define DS4_DEFINE_MPP_IQ2_COUNTED(NAME, NT) \
kernel void NAME( \
        device int8_t *A [[buffer(0)]], \
        device const block_iq2_xxs *Wq [[buffer(1)]], \
        device int32_t *C [[buffer(2)]], \
        device const uint *counts [[buffer(3)]], \
        constant uint &expert [[buffer(4)]], \
        constant uint &N [[buffer(5)]], \
        constant uint &K [[buffer(6)]], \
        constant float &qscale [[buffer(7)]], \
        uint2 tgid [[threadgroup_position_in_grid]], \
        uint tidx [[thread_index_in_threadgroup]]) { \
    const uint M = counts[expert]; \
    const uint m0 = tgid.y * 64u, n0 = tgid.x * (uint)(NT); \
    if (M == 0u || m0 >= M || n0 >= N) return; \
    const uint rows = min(64u, M - m0), bpr = K / 256u; \
    threadgroup int8_t Btile[256 * (NT)]; \
    threadgroup int8_t *bptr = Btile; \
    constexpr auto desc = matmul2d_descriptor(64, (NT), 256, false, false, false, \
                                              matmul2d_descriptor::mode::multiply_accumulate); \
    matmul2d<desc, execution_simdgroups<4>> op; \
    auto mA0 = tensor(A, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K}); \
    auto tBt0 = tensor(bptr, dextents<int32_t, 2>{(NT), 256}, array<int32_t, 2>{1, (NT)}); \
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>(); \
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; } \
    for (uint kb = 0; kb < bpr; ++kb) { \
        for (uint w = tidx; w < 16u * (uint)(NT); w += 128u) { \
            uint nn = w % (uint)(NT), seg = w / (uint)(NT); \
            ds4nf_iq2_seg_stride(Wq + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn, (uint)(NT)); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        auto mA = tensor(A + m0 * K + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K}); \
        auto mBt = tensor(bptr, dextents<int32_t, 2>{(NT), 256}, array<int32_t, 2>{1, (NT)}); \
        op.run(mA, mBt, cT); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    auto mC = tensor(C + m0 * N + n0, dextents<int32_t, 2>{(NT), (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N}); \
    cT.store(mC); \
}

DS4_DEFINE_MPP_IQ2_COUNTED(ds4_mpp_iq2_i8_i32_counted, 32)
DS4_DEFINE_MPP_IQ2_COUNTED(ds4_mpp_iq2_i8_i32_counted_n64, 64)
DS4_DEFINE_MPP_IQ2_COUNTED(ds4_mpp_iq2_i8_i32_counted_n128, 128)
#undef DS4_DEFINE_MPP_IQ2_COUNTED

#define DS4_DEFINE_MPP_IQ2_COUNTED_N256_SPLIT(NAME) \
kernel void NAME( \
        device int8_t *A [[buffer(0)]], \
        device const block_iq2_xxs *Wq [[buffer(1)]], \
        device int32_t *C [[buffer(2)]], \
        device const uint *counts [[buffer(3)]], \
        constant uint &expert [[buffer(4)]], \
        constant uint &N [[buffer(5)]], \
        constant uint &K [[buffer(6)]], \
        constant float &qscale [[buffer(7)]], \
        uint2 tgid [[threadgroup_position_in_grid]], \
        uint tidx [[thread_index_in_threadgroup]]) { \
    const uint M = counts[expert]; \
    const uint m0 = tgid.y * 64u, nbase = tgid.x * 256u; \
    if (M == 0u || m0 >= M || nbase >= N) return; \
    const uint rows = min(64u, M - m0), bpr = K / 256u; \
    threadgroup int8_t Btile[256 * 128]; \
    threadgroup int8_t *bptr = Btile; \
    constexpr auto desc = matmul2d_descriptor(64, 128, 256, false, false, false, \
                                              matmul2d_descriptor::mode::multiply_accumulate); \
    matmul2d<desc, execution_simdgroups<4>> op; \
    for (uint part = 0; part < 2u; part++) { \
        const uint n0 = nbase + part * 128u; \
        if (n0 >= N) continue; \
        auto mA0 = tensor(A, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K}); \
        auto tBt0 = tensor(bptr, dextents<int32_t, 2>{128, 256}, array<int32_t, 2>{1, 128}); \
        auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>(); \
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; } \
        for (uint kb = 0; kb < bpr; ++kb) { \
            for (uint w = tidx; w < 2048u; w += 128u) { \
                uint nn = w & 127u, seg = w >> 7; \
                ds4nf_iq2_seg_stride(Wq + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn, 128u); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            auto mA = tensor(A + m0 * K + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K}); \
            auto mBt = tensor(bptr, dextents<int32_t, 2>{128, 256}, array<int32_t, 2>{1, 128}); \
            op.run(mA, mBt, cT); \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        auto mC = tensor(C + m0 * N + n0, dextents<int32_t, 2>{128, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N}); \
        cT.store(mC); \
    } \
}

DS4_DEFINE_MPP_IQ2_COUNTED_N256_SPLIT(ds4_mpp_iq2_i8_i32_counted_n256)
#undef DS4_DEFINE_MPP_IQ2_COUNTED_N256_SPLIT

// =============================================================================
// Multi-expert iq2 fused matmul — one dispatch covers G experts.
// Collapses E per-expert dispatches into 1 single grouped dispatch. Each output
// tile reads its expert id from te[tile_y], A row offset from tr0[tile_y], row
// count from trc[tile_y]. A is packed acts concatenated by expert; C is output
// packed in the same order. Wq is laid out as one [N x K] iq2_xxs slab per
// expert, indexed by expert id.
// Synthetic validation: moe-batch-bench/nax_multiexpert_probe.{m,binary}
//   E=64 TOTAL=25108 (rows 256-512/expert) N=2048 K=7168 → ~39 TFLOPS, max_abs=0.
// Dispatch convention (host-driven, NOT counted-indirect):
//   grid = (ceil(N/32), n_tiles, 1) where n_tiles = sum_e ceil(M_e / 64).
//   tg   = (32 * 4, 1, 1) = 128 threads (4 simdgroups).
kernel void ds4_mpp_iq2_i8_i32_multi(
        device int8_t *A                    [[buffer(0)]],
        device const block_iq2_xxs *Wq      [[buffer(1)]],
        device int32_t *C                   [[buffer(2)]],
        device const uint *te               [[buffer(3)]],   // expert id per tile
        device const uint *tr0              [[buffer(4)]],   // A row offset per tile
        device const uint *trc              [[buffer(5)]],   // rows per tile
        constant uint   &N                  [[buffer(6)]],
        constant uint   &K                  [[buffer(7)]],
        constant float  &qscale             [[buffer(8)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint  tidx [[thread_index_in_threadgroup]]) {
    const uint n0     = tgid.x * 32u;
    if (n0 >= N) return;
    const uint expert = te[tgid.y];
    const uint a0     = tr0[tgid.y];
    const uint rows   = trc[tgid.y];
    if (rows == 0u) return;
    const uint bpr = K / 256u;
    device int8_t  *Ae = A + (ulong)a0 * K;
    device int32_t *Ce = C + (ulong)a0 * N;
    device const block_iq2_xxs *We = Wq + (ulong)expert * (ulong)N * (ulong)bpr;

    threadgroup int8_t Btile[256 * 32];
    threadgroup int8_t *bptr = Btile;
    constexpr auto desc = matmul2d_descriptor(64, 32, 256, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA0  = tensor(Ae, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K});
    auto tBt0 = tensor(bptr, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
    for (uint kb = 0; kb < bpr; ++kb) {
        for (uint w = tidx; w < 512u; w += 128u) {
            uint nn = w & 31u, seg = w >> 5;
            ds4nf_iq2_seg(We + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA  = tensor(Ae + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K});
        auto mBt = tensor(bptr, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
        op.run(mA, mBt, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mC = tensor(Ce + n0, dextents<int32_t, 2>{32, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N});
    cT.store(mC);
}

// q2_K multi-expert variant. Same layout: one [N x K] q2_K slab per expert in Wq.
kernel void ds4_mpp_q2k_i8_i32_multi(
        device int8_t *A                    [[buffer(0)]],
        device const block_q2_K *Wq         [[buffer(1)]],
        device int32_t *C                   [[buffer(2)]],
        device const uint *te               [[buffer(3)]],
        device const uint *tr0              [[buffer(4)]],
        device const uint *trc              [[buffer(5)]],
        constant uint   &N                  [[buffer(6)]],
        constant uint   &K                  [[buffer(7)]],
        constant float  &qscale             [[buffer(8)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint  tidx [[thread_index_in_threadgroup]]) {
    const uint n0     = tgid.x * 32u;
    if (n0 >= N) return;
    const uint expert = te[tgid.y];
    const uint a0     = tr0[tgid.y];
    const uint rows   = trc[tgid.y];
    if (rows == 0u) return;
    const uint bpr = K / 256u;
    device int8_t  *Ae = A + (ulong)a0 * K;
    device int32_t *Ce = C + (ulong)a0 * N;
    device const block_q2_K *We = Wq + (ulong)expert * (ulong)N * (ulong)bpr;

    threadgroup int8_t Btile[256 * 32];
    threadgroup int8_t *bptr = Btile;
    constexpr auto desc = matmul2d_descriptor(64, 32, 256, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA0  = tensor(Ae, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K});
    auto tBt0 = tensor(bptr, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
    for (uint kb = 0; kb < bpr; ++kb) {
        for (uint w = tidx; w < 512u; w += 128u) {
            uint nn = w & 31u, seg = w >> 5;
            ds4nf_q2k_seg(We + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA  = tensor(Ae + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K});
        auto mBt = tensor(bptr, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
        op.run(mA, mBt, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mC = tensor(Ce + n0, dextents<int32_t, 2>{32, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N});
    cT.store(mC);
}

// Fused down (q2_K).  Same dispatch convention.
#define DS4_DEFINE_MPP_Q2K_COUNTED(NAME, NT) \
kernel void NAME( \
        device int8_t *A [[buffer(0)]], \
        device const block_q2_K *Wq [[buffer(1)]], \
        device int32_t *C [[buffer(2)]], \
        device const uint *counts [[buffer(3)]], \
        constant uint &expert [[buffer(4)]], \
        constant uint &N [[buffer(5)]], \
        constant uint &K [[buffer(6)]], \
        constant float &qscale [[buffer(7)]], \
        uint2 tgid [[threadgroup_position_in_grid]], \
        uint tidx [[thread_index_in_threadgroup]]) { \
    const uint M = counts[expert]; \
    const uint m0 = tgid.y * 64u, n0 = tgid.x * (uint)(NT); \
    if (M == 0u || m0 >= M || n0 >= N) return; \
    const uint rows = min(64u, M - m0), bpr = K / 256u; \
    threadgroup int8_t Btile[256 * (NT)]; \
    threadgroup int8_t *bptr = Btile; \
    constexpr auto desc = matmul2d_descriptor(64, (NT), 256, false, false, false, \
                                              matmul2d_descriptor::mode::multiply_accumulate); \
    matmul2d<desc, execution_simdgroups<4>> op; \
    auto mA0 = tensor(A, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K}); \
    auto tBt0 = tensor(bptr, dextents<int32_t, 2>{(NT), 256}, array<int32_t, 2>{1, (NT)}); \
    auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>(); \
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; } \
    for (uint kb = 0; kb < bpr; ++kb) { \
        for (uint w = tidx; w < 16u * (uint)(NT); w += 128u) { \
            uint nn = w % (uint)(NT), seg = w / (uint)(NT); \
            ds4nf_q2k_seg_stride(Wq + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn, (uint)(NT)); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        auto mA = tensor(A + m0 * K + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K}); \
        auto mBt = tensor(bptr, dextents<int32_t, 2>{(NT), 256}, array<int32_t, 2>{1, (NT)}); \
        op.run(mA, mBt, cT); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    auto mC = tensor(C + m0 * N + n0, dextents<int32_t, 2>{(NT), (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N}); \
    cT.store(mC); \
}

DS4_DEFINE_MPP_Q2K_COUNTED(ds4_mpp_q2k_i8_i32_counted, 32)
DS4_DEFINE_MPP_Q2K_COUNTED(ds4_mpp_q2k_i8_i32_counted_n64, 64)
DS4_DEFINE_MPP_Q2K_COUNTED(ds4_mpp_q2k_i8_i32_counted_n128, 128)
#undef DS4_DEFINE_MPP_Q2K_COUNTED

#define DS4_DEFINE_MPP_Q2K_COUNTED_N256_SPLIT(NAME) \
kernel void NAME( \
        device int8_t *A [[buffer(0)]], \
        device const block_q2_K *Wq [[buffer(1)]], \
        device int32_t *C [[buffer(2)]], \
        device const uint *counts [[buffer(3)]], \
        constant uint &expert [[buffer(4)]], \
        constant uint &N [[buffer(5)]], \
        constant uint &K [[buffer(6)]], \
        constant float &qscale [[buffer(7)]], \
        uint2 tgid [[threadgroup_position_in_grid]], \
        uint tidx [[thread_index_in_threadgroup]]) { \
    const uint M = counts[expert]; \
    const uint m0 = tgid.y * 64u, nbase = tgid.x * 256u; \
    if (M == 0u || m0 >= M || nbase >= N) return; \
    const uint rows = min(64u, M - m0), bpr = K / 256u; \
    threadgroup int8_t Btile[256 * 128]; \
    threadgroup int8_t *bptr = Btile; \
    constexpr auto desc = matmul2d_descriptor(64, 128, 256, false, false, false, \
                                              matmul2d_descriptor::mode::multiply_accumulate); \
    matmul2d<desc, execution_simdgroups<4>> op; \
    for (uint part = 0; part < 2u; part++) { \
        const uint n0 = nbase + part * 128u; \
        if (n0 >= N) continue; \
        auto mA0 = tensor(A, dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K}); \
        auto tBt0 = tensor(bptr, dextents<int32_t, 2>{128, 256}, array<int32_t, 2>{1, 128}); \
        auto cT = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>(); \
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; } \
        for (uint kb = 0; kb < bpr; ++kb) { \
            for (uint w = tidx; w < 2048u; w += 128u) { \
                uint nn = w & 127u, seg = w >> 7; \
                ds4nf_q2k_seg_stride(Wq + (n0 + nn) * bpr + kb, seg, qscale, bptr, nn, 128u); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            auto mA = tensor(A + m0 * K + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K}); \
            auto mBt = tensor(bptr, dextents<int32_t, 2>{128, 256}, array<int32_t, 2>{1, 128}); \
            op.run(mA, mBt, cT); \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        auto mC = tensor(C + m0 * N + n0, dextents<int32_t, 2>{128, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)N}); \
        cT.store(mC); \
    } \
}

DS4_DEFINE_MPP_Q2K_COUNTED_N256_SPLIT(ds4_mpp_q2k_i8_i32_counted_n256)
#undef DS4_DEFINE_MPP_Q2K_COUNTED_N256_SPLIT

// ============================================================================
// int8 (W8A8) dense Q8_0 path. Validated +1.4-1.5x vs relaxed float x half at
// 0.7% rel (moe-batch-bench/nax_dense_i8_probe.m). Three kernels:
//   1) ds4_repack_q8_to_i8_rowscale : Q8_0 weight [out x in] -> int8 [out x in]
//      + per-row scale (run once at load, cached by weight offset).
//   2) ds4_quant_act_pertoken_i8    : f32 act [tok x in] -> int8 + per-token scale.
//   3) ds4_dense_i8_fused           : int8 x int8 -> int32, fused rescale in store
//      (NR1=128/NR0=32/NK=128; result tile 32x128 i32 = 16KB + weight 4KB fits TG).
// ============================================================================
struct ds4_i8_dense_args { int32_t K, M, N; };

// One threadgroup per output row: row-max over K (dequant), write int8 + per-row scale.
kernel void ds4_repack_q8_to_i8_rowscale(
        device const char  *wq8src [[buffer(0)]],   // Q8_0 weight [M x bpr blocks]
        device int8_t      *wi8    [[buffer(1)]],    // out int8 [M x K] row-major
        device float       *wscale [[buffer(2)]],    // out per-row scale [M]
        constant uint      &K      [[buffer(3)]],
        constant uint      &Mrows  [[buffer(4)]],
        constant uint64_t  &row_bytes [[buffer(5)]], // Q8_0 row stride (bytes)
        uint   row [[threadgroup_position_in_grid]],
        uint   tid [[thread_position_in_threadgroup]],
        uint   ntg [[threads_per_threadgroup]]) {
    if (row >= Mrows) return;
    const uint bpr = K / 32u;
    device const block_q8_0 *rp = (device const block_q8_0 *)(wq8src + (uint64_t)row * row_bytes);
    float m = 0.0f;
    for (uint kb = tid; kb < bpr; kb += ntg) {
        const float d = (float)rp[kb].d;
        for (int j = 0; j < 32; j++) m = max(m, fabs((float)rp[kb].qs[j] * d));
    }
    threadgroup float sh[256];
    sh[tid] = m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg/2u; s > 0u; s >>= 1) { if (tid < s) sh[tid] = max(sh[tid], sh[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    const float sc = sh[0] > 0.0f ? sh[0] / 127.0f : 1e-9f;
    if (tid == 0) wscale[row] = sc;
    const float inv = 1.0f / sc;
    device int8_t *dr = wi8 + (uint64_t)row * K;
    for (uint kb = tid; kb < bpr; kb += ntg) {
        const float d = (float)rp[kb].d;
        for (int j = 0; j < 32; j++) { float v = rint((float)rp[kb].qs[j] * d * inv); dr[kb*32+j] = (int8_t)clamp(v, -127.0f, 127.0f); }
    }
}

// One threadgroup per token row: row-max over width, write int8 + per-token scale.
kernel void ds4_quant_act_pertoken_i8(
        device const float *src   [[buffer(0)]],   // [rows x width] f32
        device int8_t      *dst   [[buffer(1)]],    // [rows x width] int8
        device float       *scale [[buffer(2)]],    // [rows]
        constant uint      &width [[buffer(3)]],
        constant uint      &rows  [[buffer(4)]],
        uint   row [[threadgroup_position_in_grid]],
        uint   tid [[thread_position_in_threadgroup]],
        uint   ntg [[threads_per_threadgroup]]) {
    if (row >= rows) return;
    device const float *sr = src + (uint64_t)row * width;
    float m = 0.0f;
    for (uint i = tid; i < width; i += ntg) m = max(m, fabs(sr[i]));
    threadgroup float sh[256];
    sh[tid] = m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ntg/2u; s > 0u; s >>= 1) { if (tid < s) sh[tid] = max(sh[tid], sh[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    const float sc = sh[0] > 0.0f ? sh[0] / 127.0f : 1e-9f;
    if (tid == 0) scale[row] = sc;
    const float inv = 1.0f / sc;
    device int8_t *dr = dst + (uint64_t)row * width;
    for (uint i = tid; i < width; i += ntg) { float v = rint(sr[i] * inv); dr[i] = (int8_t)clamp(v, -127.0f, 127.0f); }
}

// int8 x int8 -> int32 dense matmul, fused per-(row,token) rescale -> f32.
kernel void ds4_dense_i8_fused(
        constant ds4_i8_dense_args &args [[buffer(0)]],
        device const char  *wi8    [[buffer(1)]],   // int8 weight [M x K] row-major
        device const char  *ai8    [[buffer(2)]],   // int8 act [N x K] row-major
        device       char  *dst    [[buffer(3)]],   // f32 out [N x M] (dst[t*M+o])
        device const float *wscale [[buffer(4)]],    // [M]
        device const float *ascale [[buffer(5)]],    // [N]
        threadgroup char *shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 32, NK = 128, NL = NK/16, NUM_THREADS = 128;
    const int K = args.K, M = args.M, N = args.N;
    const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
    threadgroup int8_t  *sa = (threadgroup int8_t  *)shmem;
    threadgroup int32_t *sc = (threadgroup int32_t *)(shmem + NR0*NK);
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device int8_t *ptrB = (device int8_t *)ai8;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, K}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), int32_t>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    device const int8_t *wA = (device const int8_t *)wi8;
    for (int lk = 0; lk < K; lk += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL, kb = (work%NL)*16;
            if (r0 + row < M) { device const int8_t *wr = wA + (uint64_t)(r0+row)*K + lk + kb;
                for (int i = 0; i < 16; i++) sa[row*NK + kb + i] = (lk+kb+i < K) ? wr[i] : (int8_t)0; }
            else { for (int i = 0; i < 16; i++) sa[row*NK + kb + i] = 0; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0); auto mB = tB.slice(lk, r1); mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor(sc, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, NR0}));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *db = (device float *)dst;
    for (int w = tiitg; w < NR0*NR1; w += NUM_THREADS) {
        const int m = w % NR0, n = w / NR0, o = r0 + m, t = r1 + n;
        if (o < M && t < N) db[(uint64_t)t*M + o] = (float)sc[m + n*NR0] * ascale[t] * wscale[o];
    }
}

// Grouped attn_out O-proj W8A8: per group g (tgpig.z), low[t][g][m] = sum_k a_i8[t][g][k]*w_i8[g][m][k],
// rescaled by ascale[t*G+g]*wscale[g*M+m] -> f32. int8 operands; weight wi8 is [G*M][K] (group-major),
// activation ai8 is [N*G][K] (row t*G+g, so per-group token stride = G*K). Reuses ds4_repack_q8_to_i8_rowscale
// (Mrows=G*M) + ds4_quant_act_pertoken_i8 (rows=N*G) to produce wi8/wscale + ai8/ascale. NR1=128/NR0=32/NK=128.
struct ds4_attn_i8_args { int32_t K, M, N, G; };
kernel void ds4_attn_out_low_i8_fused(
        constant ds4_attn_i8_args &args [[buffer(0)]],
        device const char  *wi8    [[buffer(1)]],   // int8 weight [G*M x K]
        device const char  *ai8    [[buffer(2)]],   // int8 act [N*G x K]
        device       char  *dst    [[buffer(3)]],   // f32 low[t][g*M+m] = dst[t*G*M + g*M + m]
        device const float *wscale [[buffer(4)]],    // [G*M]
        device const float *ascale [[buffer(5)]],    // [N*G]
        threadgroup char *shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR1 = 128, NR0 = 32, NK = 128, NL = NK/16, NUM_THREADS = 128;
    const int K = args.K, M = args.M, N = args.N, G = args.G;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0, r1 = tgpig.x*NR1;
    threadgroup int8_t  *sa = (threadgroup int8_t  *)shmem;
    threadgroup int32_t *sc = (threadgroup int32_t *)(shmem + NR0*NK);
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    device int8_t *ptrB = (device int8_t *)ai8 + (uint64_t)group*K;   // group's columns; token stride = G*K
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, G*K}));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), int32_t>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    device const int8_t *wA = (device const int8_t *)wi8 + (uint64_t)group*M*K;  // group's M rows
    for (int lk = 0; lk < K; lk += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL, kb = (work%NL)*16;
            if (r0 + row < M) { device const int8_t *wr = wA + (uint64_t)(r0+row)*K + lk + kb;
                for (int i = 0; i < 16; i++) sa[row*NK + kb + i] = (lk+kb+i < K) ? wr[i] : (int8_t)0; }
            else { for (int i = 0; i < 16; i++) sa[row*NK + kb + i] = 0; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0); auto mB = tB.slice(lk, r1); mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto tC = tensor(sc, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, NR0}));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *db = (device float *)dst;
    for (int w = tiitg; w < NR0*NR1; w += NUM_THREADS) {
        const int m = w % NR0, n = w / NR0, o = r0 + m, t = r1 + n;
        if (o < M && t < N)
            db[(uint64_t)t*G*M + (uint64_t)group*M + o] = (float)sc[m + n*NR0] * ascale[t*G + group] * wscale[group*M + o];
    }
}

// =============================================================================
// iter-4 Path C: fused gate + up + swiglu (in-kernel NAX∥ALU concurrency)
// Replaces 3 separate launches per expert (gate matmul, up matmul, swiglu) with
// one shader that runs two matmul2d ops on cooperative tensors and applies the
// SiLU(gate)*up activation per-element on the cooperative tile before storing
// mid. Targets the ~1.46x in-kernel NAX∥ALU overlap measured by the expert.
//
// Inputs:
//   X     [M x K]   half  (gathered activations for this expert's tokens)
//   gateW [N x K]   half  (gate weights, dequant'd elsewhere)
//   upW   [N x K]   half  (up weights, dequant'd elsewhere)
// Output:
//   mid   [M x N]   float (SiLU(gate(x)) * up(x))
//
// Per threadgroup: 32 (rows of M) x 32 (cols of N) output tile, 1 simdgroup.
// Grid: ((N + 31)/32, (M + 31)/32).  tg threads = 32.
// Tile/SG selected via synthetic sweep (moe-batch-bench/fused_kernel_probe.m):
// at realistic per-expert M=256..1024 this beats NR0=64 SG=4 by 1.44–1.69x
// against the separate matmul+matmul+swiglu reference (vs ~1.1x for SG=4).
// =============================================================================
kernel void ds4_mpp_fused_gate_up_swiglu_h_h_f_n32(
        device half  *X         [[buffer(0)]],
        device half  *gateW     [[buffer(1)]],
        device half  *upW       [[buffer(2)]],
        device float *mid       [[buffer(3)]],
        constant uint &M        [[buffer(4)]],
        constant uint &N        [[buffer(5)]],
        constant uint &K        [[buffer(6)]],
        constant float &clamp_v [[buffer(7)]],
        uint2 tgid [[threadgroup_position_in_grid]])
{
    // Match the validated non-fused h_h_f matmul (ds4_mpp_run_tile_nt / ds4_mpp_h_h_f_n32):
    // M-tile 64, N-tile 32, K = dynamic_extent (full-K reduction in one run), transpose
    // all-false, 4 simdgroups, arg order run(X, W, C). The previous version used a FIXED
    // K-tile (NK=32) — reducing only 32 of K — plus swapped (W,X) args and (false,true,true)
    // transpose, which diverged. Dispatch updated to grid.y=ceil(M/64), tg=4 simdgroups.
    // Requires M % 64 == 0 (h_h_f tiles M by 64 with no partial clamp; the resident path's
    // %64 floor guarantees it).
    constexpr int NR0 = 64, NR1 = 32;
    constexpr auto desc = matmul2d_descriptor(NR0, NR1, static_cast<int>(dynamic_extent),
                                              false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;

    auto tX  = tensor(X,     dextents<int32_t, 2>{(int32_t)K, (int32_t)M}, array<int32_t, 2>{1, (int32_t)K});
    auto tG  = tensor(gateW, dextents<int32_t, 2>{(int32_t)N, (int32_t)K}, array<int32_t, 2>{1, (int32_t)N});
    auto tU  = tensor(upW,   dextents<int32_t, 2>{(int32_t)N, (int32_t)K}, array<int32_t, 2>{1, (int32_t)N});
    auto tM  = tensor(mid,   dextents<int32_t, 2>{(int32_t)N, (int32_t)M}, array<int32_t, 2>{1, (int32_t)N});

    auto mX  = tX.slice(0, tgid.y * NR0);
    auto mGw = tG.slice(tgid.x * NR1, 0);
    auto mUw = tU.slice(tgid.x * NR1, 0);
    auto mMo = tM.slice(tgid.x * NR1, tgid.y * NR0);

    auto cG = mm.get_destination_cooperative_tensor<decltype(mX), decltype(mGw), float>();
    auto cU = mm.get_destination_cooperative_tensor<decltype(mX), decltype(mUw), float>();
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) if (cG.is_valid_element(i)) cG[i] = 0.0f;
    for (uint16_t i = 0; i < cU.get_capacity(); ++i) if (cU.is_valid_element(i)) cU[i] = 0.0f;

    // Two matmuls — the runtime can overlap NAX (tensor) and ALU dispatch within
    // this single kernel; the goal of the fusion is in-kernel NAX∥ALU pipelining.
    mm.run(mX, mGw, cG);
    mm.run(mX, mUw, cU);

    // Per-element swiglu on cooperative tile: mid = SiLU(gate) * up
    // SiLU(x) = x / (1 + exp(-x)). Clamp gate to avoid exp overflow if clamp_v>0.
    for (uint16_t i = 0; i < cG.get_capacity(); ++i) {
        if (!cG.is_valid_element(i)) continue;
        float g = cG[i];
        if (clamp_v > 0.0f) g = metal::min(metal::max(g, -clamp_v), clamp_v);
        const float silu_g = g / (1.0f + metal::exp(-g));
        cG[i] = silu_g * cU[i];   // reuse cG as the output cooperative tile
    }

    cG.store(mMo);
}

// Per-row routing-weight reapply for the Plan A fused path. The fused kernel
// above writes mid = SiLU(gate)*up WITHOUT the per-token routing weight (it has
// no weights input), whereas the separate-path ds4_gpu_encode_mpp_swiglu_weight
// folds it in. In the slot-bank banked path the per-expert mid is contiguous and
// row-aligned with the per-token weights buffer, so a single per-row scalar
// multiply restores parity: mid[row, :] *= weights[row].
// Layout matches ds4_mpp_fused_gate_up_swiglu_h_h_f_n32's mid tensor (token row m
// occupies [m*width, m*width+width)). Dispatch: grid = rows threadgroups, nth
// threads each striding over width (mirrors ds4_gpu_encode_mpp_swiglu_weight).
kernel void ds4_mpp_mul_rows_weight_f32(
        device float        *mid     [[buffer(0)]],
        device const float  *weights [[buffer(1)]],
        constant uint        &width  [[buffer(2)]],
        constant uint        &rows   [[buffer(3)]],
        uint  row [[threadgroup_position_in_grid]],
        uint  lid [[thread_position_in_threadgroup]],
        uint  nth [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    const float w = weights[row];
    device float *r = mid + (uint64_t)row * (uint64_t)width;
    for (uint c = lid; c < width; c += nth) r[c] *= w;
}

// =============================================================================
// Path C / non-dedup: fused iq2-gate + iq2-up + swiglu + routing-weight +
// int8 quantize, counted-indirect. Drop-in replacement for the chain:
//   ds4_mpp_iq2_i8_i32_counted (gate, → gate_i32)
//   ds4_mpp_iq2_i8_i32_counted (up,   → up_i32)
//   ds4_mpp_swiglu_i32_hids_weight_i8_counted (gate_i32, up_i32, weights, → mid_i8)
// Three wins layered:
//   (1) two matmul2d ops in one kernel → in-kernel NAX∥ALU overlap window
//       (the 1.46–1.69x regime measured by fused_kernel_probe.m at M≥256).
//   (2) eliminates two device-memory round-trips: gate_i32 / up_i32 never
//       materialize in global memory (they stay in cooperative + threadgroup).
//   (3) one dispatch instead of three (smaller front-end / barrier cost).
// Type-juggling pattern (cooperative i32 → tg i32 → per-thread float) validated
// in moe-batch-bench/probe_b_int8_typejuggle.m (max_abs = 0 vs separate ref).
// Per-element math matches ds4_mpp_swiglu_i32_hids_weight_i8_counted exactly:
//   g = float(gate_i32) * input_scale;  u = float(up_i32) * input_scale;
//   if (clamp_value > 1e-6) { g = min(g, clamp_value); u = clamp(u, ±clamp); }
//   v = silu(g) * u * weights[hids[row]] * mid_qscale;
//   mid[row, col] = sat_int8(rint(v));
// Dispatch convention: same as ds4_mpp_iq2_i8_i32_counted —
//   grid.x = ceil(N/32), grid.y = ceil(M/64), M = counts[expert], tg = 128.
kernel void ds4_mpp_iq2_fused_gate_up_swiglu_counted(
        device int8_t *A                    [[buffer(0)]],
        device const block_iq2_xxs *Wq_gate [[buffer(1)]],
        device const block_iq2_xxs *Wq_up   [[buffer(2)]],
        device int8_t *mid                  [[buffer(3)]],
        device const float *weights         [[buffer(4)]],
        device const int32_t *hids          [[buffer(5)]],
        device const uint  *counts          [[buffer(6)]],
        constant uint   &expert             [[buffer(7)]],
        constant uint   &N                  [[buffer(8)]],
        constant uint   &K                  [[buffer(9)]],
        constant float  &qscale             [[buffer(10)]],
        constant float  &input_scale        [[buffer(11)]],
        constant float  &clamp_value        [[buffer(12)]],
        constant float  &mid_qscale         [[buffer(13)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint  tidx [[thread_index_in_threadgroup]]) {
    const uint M = counts[expert];
    const uint m0 = tgid.y * 64u;
    const uint n0 = tgid.x * 32u;
    if (M == 0u || m0 >= M || n0 >= N) return;
    const uint rows = min(64u, M - m0);
    const uint bpr  = K / 256u;

    // Shared TG scratch: two int8 dequant tiles during K-loop (8KB each = 16KB),
    // reinterpreted as two int32 staging tiles after the K-loop (8KB each = 16KB).
    // Same physical memory, different views — total 16KB threadgroup usage.
    threadgroup int8_t  Btile_buf[2 * 256 * 32];
    threadgroup int8_t *Btile_g = Btile_buf;
    threadgroup int8_t *Btile_u = Btile_buf + (256u * 32u);
    // Per-row routing weights cached in TG (1 fetch per row instead of per element).
    threadgroup float   row_weights[64];

    constexpr auto desc = matmul2d_descriptor(64, 32, 256, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA0  = tensor(A,       dextents<int32_t, 2>{256, 64}, array<int32_t, 2>{1, (int32_t)K});
    auto tBg0 = tensor(Btile_g, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
    auto cT_g = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBg0), int32_t>();
    auto cT_u = op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBg0), int32_t>();
    for (uint16_t i = 0; i < cT_g.get_capacity(); ++i) if (cT_g.is_valid_element(i)) cT_g[i] = 0;
    for (uint16_t i = 0; i < cT_u.get_capacity(); ++i) if (cT_u.is_valid_element(i)) cT_u[i] = 0;

    for (uint kb = 0; kb < bpr; ++kb) {
        // 128 threads cooperatively dequant 32 cols × 16 segs = 512 calls per bank.
        for (uint w = tidx; w < 512u; w += 128u) {
            uint nn = w & 31u, seg = w >> 5;
            ds4nf_iq2_seg(Wq_gate + (n0 + nn) * bpr + kb, seg, qscale, Btile_g, nn);
        }
        for (uint w = tidx; w < 512u; w += 128u) {
            uint nn = w & 31u, seg = w >> 5;
            ds4nf_iq2_seg(Wq_up + (n0 + nn) * bpr + kb, seg, qscale, Btile_u, nn);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA  = tensor(A + m0 * K + kb * 256u, dextents<int32_t, 2>{256, (int32_t)rows}, array<int32_t, 2>{1, (int32_t)K});
        auto mBg = tensor(Btile_g, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
        auto mBu = tensor(Btile_u, dextents<int32_t, 2>{32, 256}, array<int32_t, 2>{1, 32});
        // Two matmuls — the runtime can overlap NAX (tensor) and ALU dispatch within
        // this single kernel (in-kernel NAX∥ALU pipelining is the key Path C win).
        op.run(mA, mBg, cT_g);
        op.run(mA, mBu, cT_u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== type-juggle: cooperative i32 → TG i32 → per-thread float swiglu =====
    // Reuse Btile_buf as int32 output staging (same memory; K-loop has finished
    // reading the dequant tiles by the barrier above).
    threadgroup int32_t *tg_g = (threadgroup int32_t *)Btile_buf;             // [64 x 32] i32 = 8KB
    threadgroup int32_t *tg_u = (threadgroup int32_t *)(Btile_buf + 8192u);   // next 8KB
    auto mTg = tensor(tg_g, dextents<int32_t, 2>{32, 64}, array<int32_t, 2>{1, 32});
    auto mTu = tensor(tg_u, dextents<int32_t, 2>{32, 64}, array<int32_t, 2>{1, 32});
    cT_g.store(mTg);
    cT_u.store(mTu);
    // Pre-load per-row routing weights once (1 fetch/row, then everyone reads tg).
    if (tidx < 64u && tidx < rows) {
        const int32_t tok_id = hids[m0 + tidx];
        row_weights[tidx] = weights[tok_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread int32 → float → swiglu → routing-weight → int8 quantize.
    // Output tile is rows × 32 elements; 128 threads × ≤16 elems each.
    const uint total = rows * 32u;
    for (uint p = tidx; p < total; p += 128u) {
        const uint r = p >> 5;       // p / 32
        const uint c = p & 31u;      // p % 32
        if (n0 + c >= N) continue;
        const float route_weight = row_weights[r];
        float g = (float)tg_g[r * 32u + c] * input_scale;
        float u = (float)tg_u[r * 32u + c] * input_scale;
        if (clamp_value > 1.0e-6f) {
            g = metal::min(g, clamp_value);
            u = metal::clamp(u, -clamp_value, clamp_value);
        }
        const float silu_g = g / (1.0f + metal::exp(-g));
        const float v = silu_g * u * route_weight * mid_qscale;
        mid[(m0 + r) * N + (n0 + c)] = int8_t(int(rint(metal::clamp(v, -128.0f, 127.0f))));
    }
}
