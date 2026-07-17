// DS4 Metal routed-MoE matvec kernels.

#define QK_K 256
#define QK_MXFP4 32
#define N_R0_Q2_K 4
#define N_R0_Q4_K 2
#define N_R0_Q5_K 1
#define N_R0_Q6_K 2
#define N_R0_IQ2_XXS 4
#define N_R0_IQ1_M 4
#define N_R0_IQ3_XXS 4
#define N_R0_IQ4_XS 2
#define N_R0_MXFP4 2

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant float ds4_metal_kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
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

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid
#define iq3xxs_grid ds4_metal_iq3xxs_grid
#define iq1s_grid_gpu ds4_metal_iq1s_grid_gpu

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K/8];
    uchar qs[QK_K/2];
};

struct block_q6_K {
    uchar ql[QK_K/2];
    uchar qh[QK_K/4];
    char scales[QK_K/16];
    half d;
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct block_iq3_xxs {
    half d;
    uchar qs[3*QK_K/8];
};

struct block_iq1_m {
    uchar qs[QK_K/8];
    uchar qh[QK_K/16];
    uchar scales[QK_K/32];
};

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[QK_K/64];
    uchar qs[QK_K/2];
};

union iq1m_scale_t {
    half f16;
    ushort u16;
};

/* MXFP4: 32 FP4 E2M1 values per block sharing one E8M0 scale byte.
 * qs[j] low nibble = element j, high nibble = element j + 16 (ggml order). */
struct block_mxfp4 {
    uchar e;
    uchar qs[QK_MXFP4/2];
};

static constant float ds4_kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* E8M0 scale: 2^(e - 127) as one bit shift; e = 0 decodes to 0.0, matching
 * the M5 hardware decode of the native FP4 scale plane. */
static inline float ds4_e8m0_to_float(uchar e) {
    return as_type<float>(uint(e) << 23);
}

static inline char ds4_mpp_float_to_i8_scaled(float x, float qscale) {
    int v = int(rint(clamp(x * qscale, -128.0f, 127.0f)));
    return char(v);
}

static inline uchar2 ds4_q4_k_scale_min(uint j, uint k, device const uchar *q) {
    if (j < 4u) {
        return uchar2(q[j + k] & 63u, q[j + 4u + k] & 63u);
    }
    return uchar2((q[j + 4u + k] & 0x0Fu) | ((q[j - 4u + k] & 0xC0u) >> 2),
                  (q[j + 4u + k] >> 4)    | ((q[j + k]      & 0xC0u) >> 2));
}

kernel void kernel_dsv4_mpp_dequant_iq2_xxs_transpose_i8(
        device const block_iq2_xxs *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;
    const uint ib32 = il0 / 2u;
    const uint lane = il0 & 1u;
    device const ushort *q2 = blk->qs + 4u * ib32;
    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
    const uint col0 = b * QK_K + il0 * 16u;
    const ulong gv0 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
    const uchar sign0 = ksigns_iq2xs[(aux32_s >> (14u * lane)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
    const ulong gv1 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
    const uchar sign1 = ksigns_iq2xs[(aux32_s >> (14u * lane + 7u)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + 8u + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_q2_k_transpose_i8(
        device const block_q2_K *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_q2_K *blk = src + r * blocks_per_row + b;
    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);
    const uchar sc = blk->scales[il0];
    const uint il = (il0 / 2u) & 3u;
    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);
    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);
    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;
    const float ml = float(blk->dmin) * float(sc >> 4);
    const uint col0 = b * QK_K + il0 * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(dl * float(q[j] & mask) - ml, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_q4_k_transpose_i8(
        device const block_q4_K *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_q4_K *blk = src + r * blocks_per_row + b;
    device const uchar *q = blk->qs + (il0 / 4u) * 32u + 16u * (il0 & 1u);
    const uint il = il0 & 3u;
    const uint is = (il0 / 4u) * 2u;
    const uchar2 sc = ds4_q4_k_scale_min(is, il / 2u, blk->scales);
    const float d = il < 2u ? float(blk->d) : float(blk->d) * (1.0f / 16.0f);
    const float dl = d * float(sc.x);
    const float ml = float(blk->dmin) * float(sc.y);
    const ushort mask = il < 2u ? 0x0Fu : 0xF0u;
    const uint col0 = b * QK_K + il0 * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(dl * float(q[j] & mask) - ml, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_mxfp4_transpose_i8(
        device const block_mxfp4 *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_MXFP4;
    const uint segs_per_row = blocks_per_row * 2u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 2u;
    const uint il = seg & 1u;
    device const block_mxfp4 *blk = src + r * blocks_per_row + b;
    const float d = ds4_e8m0_to_float(blk->e);
    const ushort shift = il ? 4u : 0u;
    const uint col0 = b * QK_MXFP4 + il * 16u;
    for (uint j = 0; j < 16u; j++) {
        const float v = d * ds4_kvalues_mxfp4[(blk->qs[j] >> shift) & 0x0Fu];
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_mxfp4_planes_transpose_i8(
        device const uchar *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_MXFP4;
    const uint bytes_per_row = q_cols / 2u;
    const uint data_bytes = q_rows * bytes_per_row;
    device const uchar *data = src;
    device const uchar *scales = src + data_bytes;

    const uint segs_per_row = blocks_per_row * 2u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 2u;
    const uint il = seg & 1u;
    const float d = ds4_e8m0_to_float(scales[r * blocks_per_row + b]);
    const uint col0 = b * QK_MXFP4 + il * 16u;
    const uint row_data = r * bytes_per_row + b * 16u;
    for (uint j = 0; j < 16u; j++) {
        const uint elem = il * 16u + j;
        const uchar packed = data[row_data + (elem >> 1u)];
        const uchar nib = (elem & 1u) ? (packed >> 4u) : (packed & 0x0Fu);
        const float v = d * ds4_kvalues_mxfp4[nib];
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_iq2_xxs_transpose_i8_counted(
        device const block_iq2_xxs *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        device const uint *counts [[buffer(6)]],
        constant uint &expert [[buffer(7)]],
        uint tid [[thread_position_in_grid]]) {
    if (counts[expert] == 0 || tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;
    const uint ib32 = il0 / 2u;
    const uint lane = il0 & 1u;
    device const ushort *q2 = blk->qs + 4u * ib32;
    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
    const uint col0 = b * QK_K + il0 * 16u;
    const ulong gv0 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
    const uchar sign0 = ksigns_iq2xs[(aux32_s >> (14u * lane)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
    const ulong gv1 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
    const uchar sign1 = ksigns_iq2xs[(aux32_s >> (14u * lane + 7u)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + 8u + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
    }
}

kernel void kernel_dsv4_mpp_dequant_q2_k_transpose_i8_counted(
        device const block_q2_K *src [[buffer(0)]],
        device char *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        constant float &qscale [[buffer(5)]],
        device const uint *counts [[buffer(6)]],
        constant uint &expert [[buffer(7)]],
        uint tid [[thread_position_in_grid]]) {
    if (counts[expert] == 0 || tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_q2_K *blk = src + r * blocks_per_row + b;
    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);
    const uchar sc = blk->scales[il0];
    const uint il = (il0 / 2u) & 3u;
    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);
    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);
    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;
    const float ml = float(blk->dmin) * float(sc >> 4);
    const uint col0 = b * QK_K + il0 * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = ds4_mpp_float_to_i8_scaled(dl * float(q[j] & mask) - ml, qscale);
    }
}

/* Fused gate+up+down dequant kernel.  Replaces three separate dispatches
 * (gate iq2_xxs, up iq2_xxs, down q2_k) with one, saving Metal API overhead
 * on the CPU side (setComputePipelineState + dispatchThreadgroups per call).
 *
 * For DSv4 dimensions (in_dim=4096, mid_dim=2048, out_dim=4096) all three
 * dequants share total = 524288 work items (each row's 16 sub-blocks × the
 * row count), so one launch covers all three weights in parallel.  The
 * caller asserts total values match before dispatching the fused variant.
 *
 * Logic for each sub-op is copied verbatim from the standalone kernels
 * above; keep them in sync if the standalone versions change. */
kernel void kernel_dsv4_mpp_dequant_gate_up_down_i8(
        device const block_iq2_xxs *gate_src [[buffer(0)]],
        device const block_iq2_xxs *up_src   [[buffer(1)]],
        device const block_q2_K    *down_src [[buffer(2)]],
        device char *gate_dst [[buffer(3)]],
        device char *up_dst   [[buffer(4)]],
        device char *down_dst [[buffer(5)]],
        constant uint &gu_q_rows [[buffer(6)]],   // mid_dim (gate/up rows)
        constant uint &gu_q_cols [[buffer(7)]],   // in_dim  (gate/up cols)
        constant uint &d_q_rows  [[buffer(8)]],   // out_dim (down rows)
        constant uint &d_q_cols  [[buffer(9)]],   // mid_dim (down cols)
        constant uint &total     [[buffer(10)]],
        constant float &qscale   [[buffer(11)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;

    // --- GATE + UP (iq2_xxs) ---
    {
        const uint blocks_per_row = gu_q_cols / QK_K;
        const uint segs_per_row = blocks_per_row * 16u;
        const uint r = tid / segs_per_row;
        if (r < gu_q_rows) {
            const uint seg = tid - r * segs_per_row;
            const uint b = seg / 16u;
            const uint il0 = seg - b * 16u;
            const uint ib32 = il0 / 2u;
            const uint lane = il0 & 1u;
            const uint col0 = b * QK_K + il0 * 16u;

            // GATE
            {
                device const block_iq2_xxs *blk = gate_src + r * blocks_per_row + b;
                device const ushort *q2 = blk->qs + 4u * ib32;
                const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
                const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
                const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
                const ulong gv0 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
                const uchar sign0 = ksigns_iq2xs[(aux32_s >> (14u * lane)) & 127u];
                for (uint j = 0; j < 8u; j++) {
                    const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);
                    gate_dst[(col0 + j) * gu_q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
                }
                const ulong gv1 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
                const uchar sign1 = ksigns_iq2xs[(aux32_s >> (14u * lane + 7u)) & 127u];
                for (uint j = 0; j < 8u; j++) {
                    const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);
                    gate_dst[(col0 + 8u + j) * gu_q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
                }
            }

            // UP (same shape, same logic, different src/dst)
            {
                device const block_iq2_xxs *blk = up_src + r * blocks_per_row + b;
                device const ushort *q2 = blk->qs + 4u * ib32;
                const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
                const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
                const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
                const ulong gv0 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
                const uchar sign0 = ksigns_iq2xs[(aux32_s >> (14u * lane)) & 127u];
                for (uint j = 0; j < 8u; j++) {
                    const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);
                    up_dst[(col0 + j) * gu_q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
                }
                const ulong gv1 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
                const uchar sign1 = ksigns_iq2xs[(aux32_s >> (14u * lane + 7u)) & 127u];
                for (uint j = 0; j < 8u; j++) {
                    const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);
                    up_dst[(col0 + 8u + j) * gu_q_rows + r] = ds4_mpp_float_to_i8_scaled(v, qscale);
                }
            }
        }
    }

    // --- DOWN (q2_k, different shape but same tid range) ---
    {
        const uint blocks_per_row = d_q_cols / QK_K;
        const uint segs_per_row = blocks_per_row * 16u;
        const uint r = tid / segs_per_row;
        if (r < d_q_rows) {
            const uint seg = tid - r * segs_per_row;
            const uint b = seg / 16u;
            const uint il0 = seg - b * 16u;
            device const block_q2_K *blk = down_src + r * blocks_per_row + b;
            device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);
            const uchar sc = blk->scales[il0];
            const uint il = (il0 / 2u) & 3u;
            const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);
            const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);
            const float dl = float(blk->d) * float(sc & 0x0fu) * coef;
            const float ml = float(blk->dmin) * float(sc >> 4);
            const uint col0 = b * QK_K + il0 * 16u;
            for (uint j = 0; j < 16u; j++) {
                down_dst[(col0 + j) * d_q_rows + r] = ds4_mpp_float_to_i8_scaled(dl * float(q[j] & mask) - ml, qscale);
            }
        }
    }
}

kernel void kernel_dsv4_ane_dequant_iq2_xxs_transpose_f16(
        device const block_iq2_xxs *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;
    const uint ib32 = il0 / 2u;
    const uint lane = il0 & 1u;
    device const ushort *q2 = blk->qs + 4u * ib32;
    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);
    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);
    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;
    const uint col0 = b * QK_K + il0 * 16u;
    const ulong gv0 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];
    const uchar sign0 = ksigns_iq2xs[(aux32_s >> (14u * lane)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + j) * q_rows + r] = half(v);
    }
    const ulong gv1 = iq2xxs_grid[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];
    const uchar sign1 = ksigns_iq2xs[(aux32_s >> (14u * lane + 7u)) & 127u];
    for (uint j = 0; j < 8u; j++) {
        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);
        dst[(col0 + 8u + j) * q_rows + r] = half(v);
    }
}

kernel void kernel_dsv4_ane_dequant_q2_k_transpose_f16(
        device const block_q2_K *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_q2_K *blk = src + r * blocks_per_row + b;
    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);
    const uchar sc = blk->scales[il0];
    const uint il = (il0 / 2u) & 3u;
    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);
    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);
    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;
    const float ml = float(blk->dmin) * float(sc >> 4);
    const uint col0 = b * QK_K + il0 * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = half(dl * float(q[j] & mask) - ml);
    }
}

kernel void kernel_dsv4_ane_dequant_q4_k_transpose_f16(
        device const block_q4_K *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_K;
    const uint segs_per_row = blocks_per_row * 16u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 16u;
    const uint il0 = seg - b * 16u;
    device const block_q4_K *blk = src + r * blocks_per_row + b;
    device const uchar *q = blk->qs + (il0 / 4u) * 32u + 16u * (il0 & 1u);
    const uint il = il0 & 3u;
    const uint is = (il0 / 4u) * 2u;
    const uchar2 sc = ds4_q4_k_scale_min(is, il / 2u, blk->scales);
    const float d = il < 2u ? float(blk->d) : float(blk->d) * (1.0f / 16.0f);
    const float dl = d * float(sc.x);
    const float ml = float(blk->dmin) * float(sc.y);
    const ushort mask = il < 2u ? 0x0Fu : 0xF0u;
    const uint col0 = b * QK_K + il0 * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = half(dl * float(q[j] & mask) - ml);
    }
}

kernel void kernel_dsv4_ane_dequant_mxfp4_transpose_f16(
        device const block_mxfp4 *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_MXFP4;
    const uint segs_per_row = blocks_per_row * 2u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 2u;
    const uint il = seg & 1u;
    device const block_mxfp4 *blk = src + r * blocks_per_row + b;
    const float d = ds4_e8m0_to_float(blk->e);
    const ushort shift = il ? 4u : 0u;
    const uint col0 = b * QK_MXFP4 + il * 16u;
    for (uint j = 0; j < 16u; j++) {
        dst[(col0 + j) * q_rows + r] = half(d * ds4_kvalues_mxfp4[(blk->qs[j] >> shift) & 0x0Fu]);
    }
}

kernel void kernel_dsv4_ane_dequant_mxfp4_planes_transpose_f16(
        device const uchar *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &q_rows [[buffer(2)]],
        constant uint &q_cols [[buffer(3)]],
        constant uint &total [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= total) return;
    const uint blocks_per_row = q_cols / QK_MXFP4;
    const uint bytes_per_row = q_cols / 2u;
    const uint data_bytes = q_rows * bytes_per_row;
    device const uchar *data = src;
    device const uchar *scales = src + data_bytes;

    const uint segs_per_row = blocks_per_row * 2u;
    const uint r = tid / segs_per_row;
    const uint seg = tid - r * segs_per_row;
    const uint b = seg / 2u;
    const uint il = seg & 1u;
    const float d = ds4_e8m0_to_float(scales[r * blocks_per_row + b]);
    const uint col0 = b * QK_MXFP4 + il * 16u;
    const uint row_data = r * bytes_per_row + b * 16u;
    for (uint j = 0; j < 16u; j++) {
        const uint elem = il * 16u + j;
        const uchar packed = data[row_data + (elem >> 1u)];
        const uchar nib = (elem & 1u) ? (packed >> 4u) : (packed & 0x0Fu);
        dst[(col0 + j) * q_rows + r] = half(d * ds4_kvalues_mxfp4[nib]);
    }
}

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

// DSpark strict verifier route grouping. N<=5 and top-6 gives at most 30 route
// slots. The grouped execution descriptor keeps semantic row/slot order in
// group_pairs as original row-major pair indices (row * active + slot).
struct DSparkMoeGroupRoutesArgs {
    uint32_t n_pairs;
    uint32_t active;
    uint32_t n_expert;
    uint32_t max_pairs;
};

struct ds4_metal_slots6_record_offsets {
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
};

struct ds4_metal_slots6_chunk_map {
    uint32_t chunk[6];
    uint32_t slot[6];
    uint64_t slot_stride;
};

#define DS4_MOE_RECORD_TABLE_MAX 256
struct ds4_metal_slot_record_table {
    array<device const char *, DS4_MOE_RECORD_TABLE_MAX> records [[id(0)]];
};

static inline device const char *ds4_slots6_chunk_select(
        uint32_t chunk,
        device const char *chunk0,
        device const char *chunk1,
        device const char *chunk2,
        device const char *chunk3) {
    switch (chunk) {
    case 1: return chunk1;
    case 2: return chunk2;
    case 3: return chunk3;
    default: return chunk0;
    }
}

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

struct hy3_get_row_q4_k_args {
    uint n_cols;
    uint n_tokens;
};

/* HY3 stores the 120832 x 4096 token table as Q4_K.  Decode just the selected
 * row straight into DS4's resident F32 activation buffer.  One thread handles
 * one 16-value dequantizer lane, matching the native Q4_K matmul layout. */
kernel void kernel_hy3_get_row_q4_K_f32(
        constant hy3_get_row_q4_k_args &args [[buffer(0)]],
        device const block_q4_K        *src  [[buffer(1)]],
        device float                   *dst  [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint segments = args.n_cols / 16u;
    if (gid >= segments) return;
    const uint block = gid / 16u;
    const short lane = short(gid % 16u);
    float4x4 values;
    dequantize_q4_K(src + block, lane, values);
    ((device float4x4 *)dst)[gid] = values;
}

kernel void kernel_hy3_get_rows_q4_K_f32(
        constant hy3_get_row_q4_k_args &args [[buffer(0)]],
        device const block_q4_K        *src  [[buffer(1)]],
        constant int                   *tokens [[buffer(2)]],
        device float                   *dst  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint segments_per_row = args.n_cols / 16u;
    const uint total = args.n_tokens * segments_per_row;
    if (gid >= total) return;
    const uint token_row = gid / segments_per_row;
    const uint segment = gid - token_row * segments_per_row;
    const uint blocks_per_row = args.n_cols / QK_K;
    const uint block = uint(tokens[token_row]) * blocks_per_row + segment / 16u;
    const short lane = short(segment % 16u);
    float4x4 values;
    dequantize_q4_K(src + block, lane, values);
    ((device float4x4 *)dst)[gid] = values;
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q  = xb->qs;
    device const uchar *qh = xb->qh;

    short is = (il / 4) * 2;
    q  = q  + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    uchar ul = uchar(1u << (il / 2));
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0.f)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 &reg) {
    const half d_all = xb->d;
    device const ushort *ql = (device const ushort *)xb->ql;
    device const ushort *qh = (device const ushort *)xb->qh;
    device const char *scales = xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;

    const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u)
                               : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const uint kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il > 2 ? 2 : 0;
    const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shr_l = il > 1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint low = (ql[2 * i] | (uint)(ql[2 * i + 1] << 16)) & kmask2;
        const uint high = (qh[2 * i] | (uint)(qh[2 * i + 1] << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * ((half)(q & 0xFFu))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00u))    - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000u))  - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000u)) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il / 2;
    il = il % 2;
    device const uchar *q3 = xb->qs + 8 * ib32;
    device const ushort *gas = (device const ushort *)(xb->qs + QK_K / 4) + 2 * ib32;
    const uint aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uchar *grid1 = (constant uchar *)(iq3xxs_grid + q3[4 * il + 0]);
    constant uchar *grid2 = (constant uchar *)(iq3xxs_grid + q3[4 * il + 1]);
    uchar signs = ksigns_iq2xs[(aux32 >> (14 * il)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i + 0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i + 4] ? -1.f : 1.f);
    }
    grid1 = (constant uchar *)(iq3xxs_grid + q3[4 * il + 2]);
    grid2 = (constant uchar *)(iq3xxs_grid + q3[4 * il + 3]);
    signs = ksigns_iq2xs[(aux32 >> (14 * il + 7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i + 0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i + 4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    const short ib32 = il / 2;
    const short shift = (il & 1) ? 4 : 0;
    device const uchar *q = xb->qs + 16 * ib32;
    const int ls = (((xb->scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0f) |
                    (((xb->scales_h >> (2 * ib32)) & 0x03) << 4)) - 32;
    const float d = float(xb->d) * float(ls);

    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = d * ds4_metal_kvalues_iq4nl_f[(q[i] >> shift) & 0x0f];
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    const int ib32 = il / 2;
    il = il % 2;
    device const ushort *sc = (device const ushort *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) |
                ((sc[1] >> 8) & 0x00f0) |
                ((sc[2] >> 4) & 0x0f00) |
                (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uchar *qs = xb->qs + 4 * ib32 + 2 * il;
    device const uchar *qh = xb->qh + 2 * ib32 + il;

    const float dl  = d * (2 * ((sc[ib32 / 2] >> (6 * (ib32 % 2) + 3 * il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);
    constant uchar *grid1 = (constant uchar *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uchar *grid2 = (constant uchar *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 *xb, short il, thread type4x4 & reg) {
    device const uchar *qs = xb->qs;
    const float d = ds4_e8m0_to_float(xb->e);
    /* il = 0: elements 0..15 (low nibbles); il = 1: elements 16..31 (high). */
    const short shift = il ? 4 : 0;
    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = d * ds4_kvalues_mxfp4[(qs[i] >> shift) & 0x0F];
    }
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_MXFP4;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const char  * xbase = src0 + offset0;
    device const float * y     = (device const float *) (src1 + offset1);

    /* Lane pairs split each 32-value block: even lanes take elements 0..15
     * (low nibbles), odd lanes take 16..31 (high nibbles). 16 blocks per
     * simdgroup pass. */
    const short ix = tiisg/2;
    const short it = tiisg%2;
    const short shift = it ? 4 : 0;

    device const float * yb = y + ix*QK_MXFP4 + it*16;

    float sumf[nr0] = {0.f};

    for (int ib = ix; ib < nb; ib += 16) {
        float4 yl[4];
        device const float4 * y4 = (device const float4 *) yb;
        yl[0] = y4[0]; yl[1] = y4[1]; yl[2] = y4[2]; yl[3] = y4[3];

        for (short row = 0; row < nr0; row++) {
            device const block_mxfp4 * xb =
                (device const block_mxfp4 *)(xbase + row*args.nb01) + ib;
            device const uchar * qs = xb->qs;

            float4 acc = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; i++) {
                acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4*i+0] >> shift) & 0x0F];
                acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4*i+1] >> shift) & 0x0F];
                acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4*i+2] >> shift) & 0x0F];
                acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4*i+3] >> shift) & 0x0F];
            }
            sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
        }

        yb += 16*QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q5_K *x = (device const block_q5_K *)(src0 + offset0);
    device const float *yy = (device const float *)(src1 + offset1);

    float sumf[nr0] = {0.f};
    float yl[16], yh[16];

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;

    device const float *y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uchar *q1 = x[i].qs + q_offset;
        device const uchar *qh = x[i].qh + l0;
        device const half *dh = &x[i].d;
        device const ushort *a = (device const ushort *)x[i].scales + iq;

        device const float *y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uchar *q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                const uchar h = qh[l];
                acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]       + 16.f * acc2[0]) +
                                  sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                                  sc8[4] * (acc1[2]       + 16.f * acc2[2]) +
                                  sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01 / 2;
            a  += args.nb01 / 2;
        }

        y1 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = tot;
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q6_K *x = (device const block_q6_K *)(src0 + offset0);
    device const float *yy = (device const float *)(src1 + offset1);

    float sumf[nr0] = {0.f};
    float yl[16];

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uchar *q1 = x[i].ql + q_offset_l;
        device const uchar *q2 = q1 + 32;
        device const uchar *qh = x[i].qh + q_offset_h;
        device const char *sc = x[i].scales + is;
        device const half *dh = &x[i].d;

        device const float *y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l +  0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((char)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((char)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                  sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq3_xxs *x = (device const block_iq3_xxs *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};
    const int nb32 = nb * (QK_K / 32);

    threadgroup uint *svalues = (threadgroup uint *)(shmem);
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) yl[i] = y4[i];

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq3_xxs *xr = x + ibl;
        device const uchar *q3 = xr->qs + 8 * ib;
        device const ushort *gas = (device const ushort *)(xr->qs + QK_K / 4) + 2 * ib;
        device const half *dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uchar *grid1 = (const threadgroup uchar *)(svalues + q3[2 * l + 0]);
                const threadgroup uchar *grid2 = (const threadgroup uchar *)(svalues + q3[2 * l + 1]);
                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    sum[1] += yl[8 * l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh += args.nb01 / 2;
            q3 += args.nb01;
            gas += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all * 0.5f;
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq1_m *x = (device const block_iq1_m *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};
    const int nb32 = nb * (QK_K / 32);
    const short ix = tiisg;
    device const float *y4 = y + 32 * ix;
    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i +  8]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 16]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 24]; sumy[3] += yl[i + 24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        device const block_iq1_m *xr = x + ibl;
        device const uchar *qs = xr->qs + 4 * ib;
        device const uchar *qh = xr->qh + 2 * ib;
        device const ushort *sc = (device const ushort *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) |
                        ((sc[1] >> 8) & 0x00f0) |
                        ((sc[2] >> 4) & 0x0f00) |
                        (sc[3] & 0xf000);

            constant uchar *grid1 = (constant uchar *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uchar *grid2 = (constant uchar *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uchar *grid3 = (constant uchar *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uchar *grid4 = (constant uchar *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f, 0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j +  0] * (grid1[j] & 0xf) + yl[j +  4] * (grid1[j] >> 4)
                        + yl[j +  8] * (grid2[j] & 0xf) + yl[j + 12] * (grid2[j] >> 4);
                sum[1] += yl[j + 16] * (grid3[j] & 0xf) + yl[j + 20] * (grid3[j] >> 4)
                        + yl[j + 24] * (grid4[j] & 0xf) + yl[j + 28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA) +
                                 sumy[1] * (qh[0] & 0x80 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA) +
                                 sumy[3] * (qh[1] & 0x80 ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 *
                ((sum[0] + delta1) * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1) +
                 (sum[1] + delta2) * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1));

            sc += args.nb01 / 2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    threadgroup float *shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq4_xs *x = (device const block_iq4_xs *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    const int nb = args.ne00 / QK_K;
    const int ns01 = args.nb01 / args.nb00;

    const short ix = tiisg / 16;
    const short it = tiisg % 16;
    const short ib = it / 2;
    const short il = it % 2;

    shmem_f32[tiisg] = ds4_metal_kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.f};
    device const float *yb = y + ix * QK_K + ib * 32 + il * 8;

    uint aux32[2];
    thread const uchar *q8 = (thread const uchar *)aux32;

    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 *y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < nr0; ++row) {
            device const block_iq4_xs &xb = x[row * ns01 + ibl];
            device const uint *q4 = (device const uint *)(xb.qs + 16 * ib + 8 * il);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            float4 qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            float4 qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;
            const int ls = (((xb.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                            (((xb.scales_h >> (2 * ib)) & 3) << 4)) - 32;
            sumf[row] += float(xb.d) * float(ls) *
                (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float *dst_f32 = (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ds4_metal_args_mul_mv &args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0>
void kernel_mul_mv_iq2_xxs_pair_f32_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * xg = (device const block_iq2_xxs *) (src0_gate + offset0);
    device const block_iq2_xxs * xu = (device const block_iq2_xxs *) (src0_up   + offset0);
    device const float         * y  = (device const float         *) (src1      + offset1);

    float yl[32];
    float sumg[nr0]={0.f};
    float sumu[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xgr = xg + ibl;
        device const block_iq2_xxs * xur = xu + ibl;
        device const uint16_t * qg = xgr->qs + 4 * ib;
        device const uint16_t * qu = xur->qs + 4 * ib;
        device const half * dhg = &xgr->d;
        device const half * dhu = &xur->d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t * aux8g = (device const uint8_t *)qg;
            device const uint8_t * aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t * gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7*l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8*l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01/2;
            dhu += args.nb01/2;
            qg  += args.nb01/2;
            qu  += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_gate_f32 = (device float *) dst_gate + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_up_f32   = (device float *) dst_up   + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            dst_gate_f32[first_row + row] = sum_gate * 0.25f;
            dst_up_f32[first_row + row]   = sum_up   * 0.25f;
        }
    }
}

typedef void (kernel_mul_mv2_disp_t)(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>) mul_mv_id_disp_fn_t;

// Decode-time expert matvec. The ids tensor selects the routed expert for each
// slot, then this wrapper invokes the quantized row kernel for Q8_0, Q2_K, or
// IQ2_XXS weights without materializing per-expert dispatches on the CPU.
template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

template<mul_mv_id_disp_fn_t disp_fn, int nr0>
kernel void kernel_mul_mv_id_accum(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_cur = src0s + i02 * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + ((uint64_t)idx * args.ne0 +
                                  (uint64_t)i12 * args.ne1 * args.ne0) * sizeof(float);

    float prior[nr0] = {0.f};
    device float *dst_f32 = (device float *)dst_cur;
    const bool accumulate = args.ne13 != 0;
    if (accumulate && tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            prior[row] = dst_f32[first_row + row];
        }
    }

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);

    if (accumulate && tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] += prior[row];
        }
    }
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>) kernel_mul_mv_id_q_t;
typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>) kernel_mul_mv_id_q8_0_t;
typedef decltype(kernel_mul_mv_id_accum<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>, N_R0_Q2_K>) kernel_mul_mv_id_q2_accum_t;
typedef decltype(kernel_mul_mv_id_accum<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>, N_R0_Q4_K>) kernel_mul_mv_id_q4_accum_t;

// Host-visible decode MoE matvec variants for the DS4 quant formats.
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_q8_0_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS>>>;
template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;
template [[host_name("kernel_mul_mv_id_q2_K_accum_f32")]] kernel kernel_mul_mv_id_q2_accum_t kernel_mul_mv_id_accum<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>, N_R0_Q2_K>;
template [[host_name("kernel_mul_mv_id_q4_K_accum_f32")]] kernel kernel_mul_mv_id_q4_accum_t kernel_mul_mv_id_accum<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>, N_R0_Q4_K>;

// DS4 attention output low projection, specialized for the fixed block
// diagonal mapping used by the model:
//
//     low[token, group, rank] = heads[token, group, :] * Woa[group, rank, :]
//
// The generic GGML-style id matvec supports arbitrary routed expert ids.  Here
// the id is always equal to the group number, so this wrapper keeps the exact
// Q8_0 dot kernel but removes the id-buffer load and the CPU-side id table.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// DSpark verifier output-low rows5 variant.
//
// The regular output-low wrapper maps (token, group) to independent Q8 matvec
// threadgroups. This wrapper keeps the group fixed and loops N<=5 token rows
// with the same rows5 exact Q8 reduction shape, but writes back to the
// token-major output-low layout used by the original per-row wrapper.
kernel void kernel_dsv4_attn_out_low_q8_0_rows5_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint group = tgpig.z;
    if (group >= (uint)args.nei0 || args.nei1 <= 0 || args.nei1 > 6) return;

    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NT = 6;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;

    device const char *src0_cur = src0s + (uint64_t)group * args.nb02;
    device const char *src1_cur = src1  + (uint64_t)group * args.nb11;

    device const float * y[NT];
    FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
        y[tok] = (tok < args.nei1) ? (device const float *)(src1_cur + (uint64_t)tok*args.nb12)
                                   : (device const float *) src1_cur;
    }

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q8_0 *) (src0_cur + (uint64_t)(r0 + row)*args.nb01);
    }

    float sumf[NT][NR0] = {
        { 0.f }, { 0.f }, { 0.f }, { 0.f }, { 0.f }
    };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;

    float yl[NT][NQ];
    device const float * yb[NT];
    FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
        yb[tok] = y[tok] + ib0*QK8_0 + il*NQ;
    }

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
            if (tok < args.nei1) {
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    yl[tok][i] = yb[tok][i];
                }
            }
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            const float d = ax[row][ib].d;

            FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
                if (tok < args.nei1) {
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NQ; ++i) {
                        sumq += qs[i] * yl[tok][i];
                    }
                    sumf[tok][row] += sumq*d;
                }
            }
        }

        FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
            yb[tok] += NSG*NQ*QK8_0;
        }
    }

    FOR_UNROLL (short tok = 0; tok < NT; ++tok) {
        if (tok < args.nei1) {
            device float *dst_f32 = (device float *)dst +
                (uint64_t)tok * (uint64_t)args.ne1 * (uint64_t)args.ne0 +
                (uint64_t)group * (uint64_t)args.ne0;
            helper_mv_reduce_and_write<NR0>(dst_f32, sumf[tok], r0, args.ne01, tiisg, sgitg, shmem);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void kernel_mul_mv_id_iq2_xxs_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_gate_cur = src0_gate + i02*args.nb02;
    device const char * src0_up_cur   = src0_up   + i02*args.nb02;
    device const char * src1_cur      = src1      + i11*args.nb11 + i12*args.nb12;

    device char * dst_gate_cur = dst_gate + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    device char * dst_up_cur   = dst_up   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Decode-only routed expert gate/up projection fused with the DS4 activation:
//
//     mid = silu(clamp(gate)) * clamp(up) * route_weight
//
// The quantized dot products are intentionally the same IQ2_XXS paired path as
// `kernel_mul_mv_id_iq2_xxs_pair_f32`.  The only extra work is done by lane 0
// after each exact reduced row has been produced.  This removes the separate
// routed activation dispatch and avoids rereading the gate/up rows before the
// down projection.  The host uses this only for the normal release path where
// diagnostics do not request clamped gate/up intermediates.
kernel void kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    const uint64_t route_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + route_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + route_row * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                dst_gate_f32[out_row] = g;
                dst_up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_dspark_mul_mv_id_iq2_xxs_grouped_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        device const char * weights,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_offsets,
        device const uint32_t * group_pairs,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const uint group = tgpig.z;
    if (group >= *group_count) return;

    const uint pair_count = group_counts[group];
    if (pair_count < 2u || pair_count > 5u) return;

    const int32_t expert = group_experts[group];
    if (expert < 0 || expert >= args.ne02) return;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);
    const uint group_begin = group_offsets[group];

    float sumg[5][N_R0_IQ2_XXS];
    float sumu[5][N_R0_IQ2_XXS];
    for (uint p = 0; p < 5u; p++) {
        for (uint row = 0; row < N_R0_IQ2_XXS; row++) {
            sumg[p][row] = 0.0f;
            sumu[p][row] = 0.0f;
        }
    }

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const block_iq2_xxs *xg_base =
        (device const block_iq2_xxs *)(src0_gate + (uint64_t)expert * args.nb02 +
                                       (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu_base =
        (device const block_iq2_xxs *)(src0_up + (uint64_t)expert * args.nb02 +
                                       (uint64_t)first_row * args.nb01);

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg_base + ibl;
        device const block_iq2_xxs *xur = xu_base + ibl;
        device const uint16_t *qg0 = xgr->qs + 4 * ib;
        device const uint16_t *qu0 = xur->qs + 4 * ib;
        device const half *dhg0 = &xgr->d;
        device const half *dhu0 = &xur->d;

        float yl[5][32];
        for (uint p = 0; p < pair_count; p++) {
            const uint pair = group_pairs[group_begin + p];
            const int iid1 = pair / uint(args.nei0);
            const int idx = pair - uint(iid1 * args.nei0);
            const int64_t i11 = idx % args.ne11;
            const int64_t i12 = iid1;
            device const float *y =
                (device const float *)(src1 + (uint64_t)i11 * args.nb11 +
                                       (uint64_t)i12 * args.nb12);
            device const float *y4 = y + 32 * ix + (uint64_t)(ib32 - ix) * 32u;
            for (short i = 0; i < 32; ++i) {
                yl[p][i] = y4[i];
            }
        }

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint16_t *qg = qg0;
            device const uint16_t *qu = qu0;
            device const half *dhg = dhg0;
            device const half *dhu = dhu0;
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            for (uint p = 0; p < pair_count; p++) {
                float sg = 0.0f;
                float su = 0.0f;
                for (short l = 0; l < 4; ++l) {
                    const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                    const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                    const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                    const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                    for (short j = 0; j < 8; ++j) {
                        const float v = yl[p][8 * l + j];
                        sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                sumg[p][row] += dg * sg;
                sumu[p][row] += du * su;
            }

            qg0 += args.nb01 / 2;
            qu0 += args.nb01 / 2;
            dhg0 += args.nb01 / 2;
            dhu0 += args.nb01 / 2;
        }
    }

    const float c = act.clamp_value;
    for (uint p = 0; p < pair_count; p++) {
        const uint pair = group_pairs[group_begin + p];
        const uint64_t route_row = (uint64_t)pair;
        device float *dst_mid_f32 =
            (device float *)(dst_mid + route_row * act.mid_row_stride);
        device const float *route_w =
            (device const float *)(weights + route_row * act.weight_stride);
        const float route_weight = route_w[0];
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const float sum_gate = simd_sum(sumg[p][row]);
            const float sum_up   = simd_sum(sumu[p][row]);
            if (tiisg == 0) {
                const uint out_row = first_row + row;
                float g = sum_gate * 0.25f;
                float u = sum_up * 0.25f;
                if (c > 1.0e-6f) {
                    g = min(g, c);
                    u = clamp(u, -c, c);
                }
                const float silu = g / (1.0f + exp(-g));
                dst_mid_f32[out_row] = silu * u * route_weight;
            }
        }
    }

    (void)tiitg;
}

kernel void kernel_dspark_mul_mv_id_iq2_xxs_singleton_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    uint multiplicity = 0u;
    const uint gc = min(*group_count, 30u);
    for (uint g = 0; g < gc; g++) {
        if (group_experts[g] == i02) {
            multiplicity = group_counts[g];
            break;
        }
    }
    if (multiplicity != 1u) return;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0.0f;
            float su = 0.0f;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    const uint64_t route_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + route_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + route_row * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = sum_gate * 0.25f;
            float u = sum_up * 0.25f;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_mxfp4_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + (uint64_t)iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + (uint64_t)i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + (uint64_t)i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur =
        dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur =
        dst_up + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t route_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + route_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + route_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_MXFP4 && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_mxfp4_record_table_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_slots6_record_offsets & rec,
        device const ds4_metal_slot_record_table & table,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t slot_i32 = ((device const int32_t *)(ids + (uint64_t)iid1 * args.nbi1))[idx];
    const uint slot = (uint)clamp(slot_i32, 0, (int)DS4_MOE_RECORD_TABLE_MAX - 1);
    device const char *record = table.records[slot];
    if (record == nullptr) return;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = record + rec.gate_offset;
    device const char *src0_up_cur   = record + rec.up_offset;
    device const char *src1_cur      = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur =
        dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur =
        dst_up + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t route_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + route_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + route_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_MXFP4 && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_MXFP4 && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_record_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_slots6_record_offsets & rec,
        device const char * slot0,
        device const char * slot1,
        device const char * slot2,
        device const char * slot3,
        device const char * slot4,
        device const char * slot5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *slot_cur = slot0;
    switch (idx) {
    case 1: slot_cur = slot1; break;
    case 2: slot_cur = slot2; break;
    case 3: slot_cur = slot3; break;
    case 4: slot_cur = slot4; break;
    case 5: slot_cur = slot5; break;
    default: break;
    }

    device const char *src0_gate_cur = slot_cur + rec.gate_offset;
    device const char *src0_up_cur   = slot_cur + rec.up_offset;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_MXFP4 && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_chunked_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_slots6_chunk_map & map,
        device const char * gate_chunk0,
        device const char * gate_chunk1,
        device const char * gate_chunk2,
        device const char * gate_chunk3,
        device const char * up_chunk0,
        device const char * up_chunk1,
        device const char * up_chunk2,
        device const char * up_chunk3,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;
    const uint32_t chunk = map.chunk[idx];
    const uint32_t slot = map.slot[idx];
    const uint64_t slot_off = (uint64_t)slot * map.slot_stride;

    device const char *src0_gate_cur =
        ds4_slots6_chunk_select(chunk, gate_chunk0, gate_chunk1, gate_chunk2, gate_chunk3) +
        slot_off;
    device const char *src0_up_cur =
        ds4_slots6_chunk_select(chunk, up_chunk0, up_chunk1, up_chunk2, up_chunk3) +
        slot_off;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_MXFP4;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_MXFP4 && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            if (act.write_clamped != 0) {
                gate_f32[out_row] = g;
                up_f32[out_row] = u;
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q4_K_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Same release-path fusion as the IQ2_XXS kernel above for the Q4_K expert
// variant.  The Q4 pair path reuses the existing exact matvec implementation
// for gate and up, then the same lane that wrote each row derives the routed
// SwiGLU input.  This keeps Q4 behavior aligned with the Q2 optimization while
// preserving the old pair projection arithmetic.
kernel void kernel_mul_mv_id_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

/* HY3's shared expert uses Q5_K gate/up tensors.  Keep the two exact Q5_K
 * projections and weighted SwiGLU in one dispatch, just like the resident
 * Q4_K and routed IQ1_M pair paths. */
kernel void kernel_mul_mv_id_q5_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;
    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };
    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K>(
        args0, src0_gate_cur, src1_cur, dst_gate_cur, shmem,
        tgpig, tiisg, sgitg);
    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K>(
        args0, src0_up_cur, src1_cur, dst_up_cur, shmem,
        tgpig, tiisg, sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q5_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q5_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            mid_f32[out_row] = (g / (1.0f + exp(-g))) * u * route_w[0];
        }
    }
    (void)tiitg;
}

/* HY3's routed gate/up tensors are IQ1_M.  Fuse both selected-expert
 * projections and the weighted SwiGLU into one dispatch, mirroring DS4's Q4_K
 * release path while retaining the exact IQ1_M dot-product implementation. */
kernel void kernel_mul_mv_id_iq1_m_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;
    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };
    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M>(
        args0, src0_gate_cur, src1_cur, dst_gate_cur, shmem,
        tgpig, tiisg, sgitg);
    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M>(
        args0, src0_up_cur, src1_cur, dst_up_cur, shmem,
        tgpig, tiisg, sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ1_M;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(dst_mid + (uint64_t)idx * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + (uint64_t)idx * act.weight_stride);
    const float c = act.clamp_value;
    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ1_M && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            mid_f32[out_row] = (g / (1.0f + exp(-g))) * u * route_w[0];
        }
    }
    (void)tiitg;
}

/* HY3 routed down projection: consume the eight selected IQ3_XXS experts and
 * accumulate their already route-weighted rows directly into the model-width
 * output.  This avoids materializing and then summing eight 4096-wide expert
 * outputs.  Quality mode keeps the legacy per-expert projection plus ordered
 * sum path on the host. */
kernel void kernel_mul_mv_id_iq3_xxs_sum8_f32(
        constant ds4_metal_args_mul_mv_id &args,
        device const char *src0s,
        device const char *src1,
        device       char *dst,
        device const char *ids,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_IQ3_XXS;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    threadgroup uint *svalues = (threadgroup uint *)shmem;
    threadgroup uchar *ssigns = (threadgroup uchar *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sumf[N_R0_IQ3_XXS] = {0.f};
    const int nb32 = nb * (QK_K / 32);
    const int ix = tiisg;

    for (int expert_slot = 0; expert_slot < 8; ++expert_slot) {
        const int32_t expert = token_ids[expert_slot];
        device const block_iq3_xxs *x =
            (device const block_iq3_xxs *)(src0s +
                (uint64_t)expert * args.nb02 +
                (uint64_t)first_row * args.nb01);
        device const float *y =
            (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *y4 = y + 32 * ix;

        for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
            float yl[32];
            for (short i = 0; i < 32; ++i) yl[i] = y4[i];

            const int ibl = ib32 / (QK_K / 32);
            const int ib = ib32 % (QK_K / 32);
            device const block_iq3_xxs *xr = x + ibl;
            device const uchar *q3 = xr->qs + 8 * ib;
            device const ushort *gas =
                (device const ushort *)(xr->qs + QK_K / 4) + 2 * ib;
            device const half *dh = &xr->d;

            for (short row = 0; row < nr0; ++row) {
                if (first_row + row < args.ne0) {
                    const float db = dh[0];
                    const uint aux32 = gas[0] | (gas[1] << 16);
                    const float d = db * (0.5f + (aux32 >> 28));
                    float2 sum = {0.f, 0.f};
                    for (short l = 0; l < 4; ++l) {
                        const threadgroup uchar *grid1 =
                            (const threadgroup uchar *)(svalues + q3[2 * l + 0]);
                        const threadgroup uchar *grid2 =
                            (const threadgroup uchar *)(svalues + q3[2 * l + 1]);
                        const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];
                        for (short j = 0; j < 4; ++j) {
                            sum[0] += yl[8 * l + j + 0] * grid1[j] *
                                (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                            sum[1] += yl[8 * l + j + 4] * grid2[j] *
                                (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                        }
                    }
                    sumf[row] += d * (sum[0] + sum[1]);
                }
                dh += args.nb01 / 2;
                q3 += args.nb01;
                gas += args.nb01 / 2;
            }
            y4 += 32 * 32;
        }
    }

    device float *dst_f32 =
        (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all * 0.5f;
    }
    (void)tiitg;
}

kernel void kernel_mul_mv_id_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + expert*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

// DSpark verifier diagnostic: fuse the Q2_K down projection and the six-slot
// expert sum while preserving the strict row path's observable arithmetic.
// Unlike kernel_mul_mv_id_q2_K_sum6_f32, each expert slot keeps an independent
// accumulator and is reduced with simd_sum before the final ordered FP32
// slot0 + slot1 + ... + slot5 chain.
kernel void kernel_mul_mv_id_q2_K_sum6_ordered_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float slot_sumf[6][N_R0_Q2_K];
    for (int slot = 0; slot < 6; slot++) {
        for (int row = 0; row < N_R0_Q2_K; row++) {
            slot_sumf[slot][row] = 0.f;
        }
    }

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + expert*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    slot_sumf[expert_slot][row] +=
                        dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                        dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float slot0 = simd_sum(slot_sumf[0][row]);
        const float slot1 = simd_sum(slot_sumf[1][row]);
        const float slot2 = simd_sum(slot_sumf[2][row]);
        const float slot3 = simd_sum(slot_sumf[3][row]);
        const float slot4 = simd_sum(slot_sumf[4][row]);
        const float slot5 = simd_sum(slot_sumf[5][row]);
        float sum_all = slot0 + slot1;
        sum_all = sum_all + slot2;
        sum_all = sum_all + slot3;
        sum_all = sum_all + slot4;
        sum_all = sum_all + slot5;
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_dspark_mul_mv_id_q2_K_singleton_down_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const uint pair = tgpig.z;
    const uint token = pair / uint(args.nei0);
    const uint idx = pair - token * uint(args.nei0);
    if (token >= uint(args.nei1) || idx >= uint(args.nei0)) return;

    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    const int32_t expert = token_ids[idx];
    if (expert < 0 || expert >= args.ne02) return;

    uint multiplicity = 0u;
    const uint gc = min(*group_count, 30u);
    for (uint g = 0; g < gc; g++) {
        if (group_experts[g] == expert) {
            multiplicity = group_counts[g];
            break;
        }
    }
    if (multiplicity != 1u) return;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    device const block_q2_K * x =
        (device const block_q2_K *)(src0s + (uint64_t)expert * args.nb02 +
                                   (uint64_t)first_row * args.nb01);
    device const float * y =
        (device const float *)(src1 + (uint64_t)token * args.nb12 +
                              (uint64_t)idx * args.nb11);

    float sumf[N_R0_Q2_K] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;
    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[32];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row < args.ne0) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                    acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                    acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                    acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                    acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                    acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                    acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                    acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                }
                float dall = dh[0];
                float dmin = dh[1] * 1.f/16.f;
                sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                     (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                     (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                     (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                             dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                     sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            }

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 =
        (device float *)(dst + (uint64_t)token * (uint64_t)args.ne1 * args.nb1 +
                         (uint64_t)idx * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
}

kernel void kernel_dspark_mul_mv_id_q2_K_grouped_down_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_offsets,
        device const uint32_t * group_pairs,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const uint group = tgpig.z;
    if (group >= *group_count) return;

    const uint pair_count = group_counts[group];
    if (pair_count < 2u || pair_count > 5u) return;

    const int32_t expert = group_experts[group];
    if (expert < 0 || expert >= args.ne02) return;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint group_begin = group_offsets[group];
    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    device const block_q2_K * x =
        (device const block_q2_K *)(src0s + (uint64_t)expert * args.nb02 +
                                   (uint64_t)first_row * args.nb01);

    float sumf[5][N_R0_Q2_K];
    for (uint p = 0; p < 5u; p++) {
        for (uint row = 0; row < N_R0_Q2_K; row++) {
            sumf[p][row] = 0.0f;
        }
    }

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[5][32];
        float4 sumy[5];
        for (uint p = 0; p < 5u; p++) {
            sumy[p] = float4(0.f, 0.f, 0.f, 0.f);
        }
        for (uint p = 0; p < pair_count; p++) {
            const uint pair = group_pairs[group_begin + p];
            const uint token = pair / uint(args.nei0);
            const uint idx = pair - token * uint(args.nei0);
            device const float * y =
                (device const float *)(src1 + (uint64_t)token * args.nb12 +
                                      (uint64_t)idx * args.nb11);
            device const float * y4 = y + (uint64_t)ib * QK_K + 128 * iq + 8 * ir;
            for (short i = 0; i < 8; ++i) {
                yl[p][i+ 0] = y4[i+ 0]; sumy[p][0] += yl[p][i+ 0];
                yl[p][i+ 8] = y4[i+32]; sumy[p][1] += yl[p][i+ 8];
                yl[p][i+16] = y4[i+64]; sumy[p][2] += yl[p][i+16];
                yl[p][i+24] = y4[i+96]; sumy[p][3] += yl[p][i+24];
            }
        }

        device const uint8_t  * sc0 = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs0 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh0 = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t  * sc = sc0;
            device const uint16_t * qs = qs0;
            device const half     * dh = dh0;
            if (first_row + row < args.ne0) {
                for (uint p = 0; p < pair_count; p++) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[p][i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[p][i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[p][i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[p][i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[p][i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[p][i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[p][i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[p][i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[p][row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                            (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                            (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                            (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                    dmin * (sumy[p][0] * (sc[0] & 0xF0) + sumy[p][1] * (sc[2] & 0xF0) +
                                            sumy[p][2] * (sc[4] & 0xF0) + sumy[p][3] * (sc[6] & 0xF0));
                }
            }
            qs0 += args.nb01/2;
            sc0 += args.nb01;
            dh0 += args.nb01/2;
        }
    }

    for (uint p = 0; p < pair_count; p++) {
        const uint pair = group_pairs[group_begin + p];
        const uint token = pair / uint(args.nei0);
        const uint idx = pair - token * uint(args.nei0);
        device float * dst_f32 =
            (device float *)(dst + (uint64_t)token * (uint64_t)args.ne1 * args.nb1 +
                             (uint64_t)idx * args.nb1);
        for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
            const float sum_all = simd_sum(sumf[p][row]);
            if (tiisg == 0) dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
    (void)tiitg;
}

// Experimental low-register grouped Q2 down path for the common multiplicity-2
// case. Groups larger than two fall back to the exact descriptor wrapper inside
// this same kernel so the descriptor remains complete.
kernel void kernel_dspark_mul_mv_id_q2_K_grouped_down_pair2_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_offsets,
        device const uint32_t * group_pairs,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const uint group = tgpig.z;
    if (group >= *group_count) return;

    const uint pair_count = group_counts[group];
    if (pair_count < 2u || pair_count > 5u) return;

    const int32_t expert = group_experts[group];
    if (expert < 0 || expert >= args.ne02) return;

    const uint group_begin = group_offsets[group];
    if (pair_count != 2u) {
        ds4_metal_args_mul_mv args0 = {
            /*.ne00 =*/ args.ne00,
            /*.ne01 =*/ args.ne01,
            /*.ne02 =*/ 1,
            /*.nb00 =*/ args.nb00,
            /*.nb01 =*/ args.nb01,
            /*.nb02 =*/ args.nb02,
            /*.nb03 =*/ args.nb02,
            /*.ne10 =*/ args.ne10,
            /*.ne11 =*/ 1,
            /*.ne12 =*/ 1,
            /*.nb10 =*/ args.nb10,
            /*.nb11 =*/ args.nb11,
            /*.nb12 =*/ args.nb12,
            /*.nb13 =*/ args.nb12,
            /*.ne0  =*/ args.ne0,
            /*.ne1  =*/ 1,
            /*.nr0  =*/ args.nr0,
            /*.r2   =*/ 1,
            /*.r3   =*/ 1,
        };
        uint3 row_tgpig = tgpig;
        row_tgpig.z = 0;
        for (uint p = 0; p < pair_count; p++) {
            const uint pair = group_pairs[group_begin + p];
            const uint token = pair / uint(args.nei0);
            const uint idx = pair - token * uint(args.nei0);
            device const char *src0_cur = src0s + (uint64_t)expert * args.nb02;
            device const char *src1_cur =
                src1 + (uint64_t)idx * args.nb11 + (uint64_t)token * args.nb12;
            device char *dst_cur =
                dst + ((uint64_t)idx * args.ne0 +
                       (uint64_t)token * args.ne1 * args.ne0) * sizeof(float);
            kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>(
                args0,
                src0_cur,
                src1_cur,
                dst_cur,
                shmem,
                row_tgpig,
                tiisg,
                sgitg);
        }
        (void)tiitg;
        return;
    }

    const uint pair0 = group_pairs[group_begin + 0u];
    const uint pair1 = group_pairs[group_begin + 1u];
    const uint token0 = pair0 / uint(args.nei0);
    const uint idx0 = pair0 - token0 * uint(args.nei0);
    const uint token1 = pair1 / uint(args.nei0);
    const uint idx1 = pair1 - token1 * uint(args.nei0);

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    device const block_q2_K * x =
        (device const block_q2_K *)(src0s + (uint64_t)expert * args.nb02 +
                                   (uint64_t)first_row * args.nb01);
    device const float * y0 =
        (device const float *)(src1 + (uint64_t)token0 * args.nb12 +
                              (uint64_t)idx0 * args.nb11);
    device const float * y1 =
        (device const float *)(src1 + (uint64_t)token1 * args.nb12 +
                              (uint64_t)idx1 * args.nb11);

    float sumf0[N_R0_Q2_K] = {0.f};
    float sumf1[N_R0_Q2_K] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float yl0[32];
        float yl1[32];
        float4 sumy0 = {0.f, 0.f, 0.f, 0.f};
        float4 sumy1 = {0.f, 0.f, 0.f, 0.f};
        device const float * y40 = y0 + (uint64_t)ib * QK_K + 128 * iq + 8 * ir;
        device const float * y41 = y1 + (uint64_t)ib * QK_K + 128 * iq + 8 * ir;
        for (short i = 0; i < 8; ++i) {
            yl0[i+ 0] = y40[i+ 0]; sumy0[0] += yl0[i+ 0];
            yl0[i+ 8] = y40[i+32]; sumy0[1] += yl0[i+ 8];
            yl0[i+16] = y40[i+64]; sumy0[2] += yl0[i+16];
            yl0[i+24] = y40[i+96]; sumy0[3] += yl0[i+24];
            yl1[i+ 0] = y41[i+ 0]; sumy1[0] += yl1[i+ 0];
            yl1[i+ 8] = y41[i+32]; sumy1[1] += yl1[i+ 8];
            yl1[i+16] = y41[i+64]; sumy1[2] += yl1[i+16];
            yl1[i+24] = y41[i+96]; sumy1[3] += yl1[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row < args.ne0) {
                float4 acc10 = {0.f, 0.f, 0.f, 0.f};
                float4 acc20 = {0.f, 0.f, 0.f, 0.f};
                float4 acc11 = {0.f, 0.f, 0.f, 0.f};
                float4 acc21 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc10[0] += yl0[i+ 0] * (qs[i/2] & 0x0003);
                    acc20[0] += yl0[i+ 1] * (qs[i/2] & 0x0300);
                    acc10[1] += yl0[i+ 8] * (qs[i/2] & 0x000c);
                    acc20[1] += yl0[i+ 9] * (qs[i/2] & 0x0c00);
                    acc10[2] += yl0[i+16] * (qs[i/2] & 0x0030);
                    acc20[2] += yl0[i+17] * (qs[i/2] & 0x3000);
                    acc10[3] += yl0[i+24] * (qs[i/2] & 0x00c0);
                    acc20[3] += yl0[i+25] * (qs[i/2] & 0xc000);
                    acc11[0] += yl1[i+ 0] * (qs[i/2] & 0x0003);
                    acc21[0] += yl1[i+ 1] * (qs[i/2] & 0x0300);
                    acc11[1] += yl1[i+ 8] * (qs[i/2] & 0x000c);
                    acc21[1] += yl1[i+ 9] * (qs[i/2] & 0x0c00);
                    acc11[2] += yl1[i+16] * (qs[i/2] & 0x0030);
                    acc21[2] += yl1[i+17] * (qs[i/2] & 0x3000);
                    acc11[3] += yl1[i+24] * (qs[i/2] & 0x00c0);
                    acc21[3] += yl1[i+25] * (qs[i/2] & 0xc000);
                }
                float dall = dh[0];
                float dmin = dh[1] * 1.f/16.f;
                sumf0[row] += dall * ((acc10[0] + 1.f/256.f * acc20[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                      (acc10[1] + 1.f/256.f * acc20[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                      (acc10[2] + 1.f/256.f * acc20[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                      (acc10[3] + 1.f/256.f * acc20[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                              dmin * (sumy0[0] * (sc[0] & 0xF0) + sumy0[1] * (sc[2] & 0xF0) +
                                      sumy0[2] * (sc[4] & 0xF0) + sumy0[3] * (sc[6] & 0xF0));
                sumf1[row] += dall * ((acc11[0] + 1.f/256.f * acc21[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                      (acc11[1] + 1.f/256.f * acc21[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                      (acc11[2] + 1.f/256.f * acc21[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                      (acc11[3] + 1.f/256.f * acc21[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                              dmin * (sumy1[0] * (sc[0] & 0xF0) + sumy1[1] * (sc[2] & 0xF0) +
                                      sumy1[2] * (sc[4] & 0xF0) + sumy1[3] * (sc[6] & 0xF0));
            }
            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst0_f32 =
        (device float *)(dst + (uint64_t)token0 * (uint64_t)args.ne1 * args.nb1 +
                         (uint64_t)idx0 * args.nb1);
    device float * dst1_f32 =
        (device float *)(dst + (uint64_t)token1 * (uint64_t)args.ne1 * args.nb1 +
                         (uint64_t)idx1 * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum0 = simd_sum(sumf0[row]);
        const float sum1 = simd_sum(sumf1[row]);
        if (tiisg == 0) {
            dst0_f32[first_row + row] = sum0;
            dst1_f32[first_row + row] = sum1;
        }
    }

    (void)shmem;
    (void)tiitg;
}

// Diagnostic exact grouped wrapper: uses the grouped route descriptor, but runs
// the same Q2_K row kernel as kernel_mul_mv_id for each pair. This proves
// whether grouping/output plumbing is safe before attempting shared-weight math.
kernel void kernel_dspark_mul_mv_id_q2_K_grouped_down_safe_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const uint32_t * group_count,
        device const int32_t  * group_experts,
        device const uint32_t * group_offsets,
        device const uint32_t * group_pairs,
        device const uint32_t * group_counts,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint group = tgpig.z;
    if (group >= *group_count) return;

    const uint pair_count = group_counts[group];
    if (pair_count < 2u || pair_count > 5u) return;

    const int32_t expert = group_experts[group];
    if (expert < 0 || expert >= args.ne02) return;

    const uint group_begin = group_offsets[group];
    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };
    uint3 row_tgpig = tgpig;
    row_tgpig.z = 0;

    for (uint p = 0; p < pair_count; p++) {
        const uint pair = group_pairs[group_begin + p];
        const uint token = pair / uint(args.nei0);
        const uint idx = pair - token * uint(args.nei0);
        device const char *src0_cur =
            src0s + (uint64_t)expert * args.nb02;
        device const char *src1_cur =
            src1 + (uint64_t)idx * args.nb11 + (uint64_t)token * args.nb12;
        device char *dst_cur =
            dst + ((uint64_t)idx * args.ne0 +
                   (uint64_t)token * args.ne1 * args.ne0) * sizeof(float);
        kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>(
            args0,
            src0_cur,
            src1_cur,
            dst_cur,
            shmem,
            row_tgpig,
            tiisg,
            sgitg);
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_mxfp4_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    const short shift = it ? 4 : 0;

    float sumf[nr0] = {0.f};
    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const char *xbase =
            src0s + (uint64_t)expert * args.nb02 + (uint64_t)first_row * args.nb01;
        device const float *y = (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *yb = y + ix * QK_MXFP4 + it * 16;

        for (int ib = ix; ib < nb; ib += 16) {
            float4 yl[4];
            device const float4 *y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[1];
            yl[2] = y4[2];
            yl[3] = y4[3];

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const block_mxfp4 *xb =
                        (device const block_mxfp4 *)(xbase + (uint64_t)row * args.nb01) + ib;
                    device const uchar *qs = xb->qs;

                    float4 acc = {0.f, 0.f, 0.f, 0.f};
                    for (short i = 0; i < 4; i++) {
                        acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4 * i + 0] >> shift) & 0x0F];
                        acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4 * i + 1] >> shift) & 0x0F];
                        acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4 * i + 2] >> shift) & 0x0F];
                        acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4 * i + 3] >> shift) & 0x0F];
                    }
                    sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
                }
            }

            yb += 16 * QK_MXFP4;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_id_mxfp4_record_table_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_slots6_record_offsets & rec,
        device const ds4_metal_slot_record_table & table,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    const short shift = it ? 4 : 0;

    float sumf[nr0] = {0.f};
    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t slot_i32 = token_ids[expert_slot];
        const uint slot = (uint)clamp(slot_i32, 0, (int)DS4_MOE_RECORD_TABLE_MAX - 1);
        device const char *record = table.records[slot];
        if (record == nullptr) continue;
        device const char *xbase =
            record + rec.down_offset + (uint64_t)first_row * args.nb01;
        device const float *y = (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *yb = y + ix * QK_MXFP4 + it * 16;

        for (int ib = ix; ib < nb; ib += 16) {
            float4 yl[4];
            device const float4 *y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[1];
            yl[2] = y4[2];
            yl[3] = y4[3];

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const block_mxfp4 *xb =
                        (device const block_mxfp4 *)(xbase + (uint64_t)row * args.nb01) + ib;
                    device const uchar *qs = xb->qs;

                    float4 acc = {0.f, 0.f, 0.f, 0.f};
                    for (short i = 0; i < 4; i++) {
                        acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4 * i + 0] >> shift) & 0x0F];
                        acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4 * i + 1] >> shift) & 0x0F];
                        acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4 * i + 2] >> shift) & 0x0F];
                        acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4 * i + 3] >> shift) & 0x0F];
                    }
                    sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
                }
            }

            yb += 16 * QK_MXFP4;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_mxfp4_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    const short shift = it ? 4 : 0;

    float sumf[nr0] = {0.f};
    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const char *xbase = src0_cur + (uint64_t)first_row * args.nb01;
        device const float *y = (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *yb = y + ix * QK_MXFP4 + it * 16;

        for (int ib = ix; ib < nb; ib += 16) {
            float4 yl[4];
            device const float4 *y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[1];
            yl[2] = y4[2];
            yl[3] = y4[3];

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const block_mxfp4 *xb =
                        (device const block_mxfp4 *)(xbase + (uint64_t)row * args.nb01) + ib;
                    device const uchar *qs = xb->qs;

                    float4 acc = {0.f, 0.f, 0.f, 0.f};
                    for (short i = 0; i < 4; i++) {
                        acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4 * i + 0] >> shift) & 0x0F];
                        acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4 * i + 1] >> shift) & 0x0F];
                        acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4 * i + 2] >> shift) & 0x0F];
                        acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4 * i + 3] >> shift) & 0x0F];
                    }
                    sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
                }
            }

            yb += 16 * QK_MXFP4;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_mxfp4_record_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_slots6_record_offsets & rec,
        device const char * slot0,
        device const char * slot1,
        device const char * slot2,
        device const char * slot3,
        device const char * slot4,
        device const char * slot5,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    const short shift = it ? 4 : 0;

    float sumf[nr0] = {0.f};
    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *slot_cur = slot0;
        switch (expert_slot) {
        case 1: slot_cur = slot1; break;
        case 2: slot_cur = slot2; break;
        case 3: slot_cur = slot3; break;
        case 4: slot_cur = slot4; break;
        case 5: slot_cur = slot5; break;
        default: break;
        }
        device const char *xbase = slot_cur + rec.down_offset + (uint64_t)first_row * args.nb01;
        device const float *y = (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *yb = y + ix * QK_MXFP4 + it * 16;

        for (int ib = ix; ib < nb; ib += 16) {
            float4 yl[4];
            device const float4 *y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[1];
            yl[2] = y4[2];
            yl[3] = y4[3];

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const block_mxfp4 *xb =
                        (device const block_mxfp4 *)(xbase + (uint64_t)row * args.nb01) + ib;
                    device const uchar *qs = xb->qs;

                    float4 acc = {0.f, 0.f, 0.f, 0.f};
                    for (short i = 0; i < 4; i++) {
                        acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4 * i + 0] >> shift) & 0x0F];
                        acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4 * i + 1] >> shift) & 0x0F];
                        acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4 * i + 2] >> shift) & 0x0F];
                        acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4 * i + 3] >> shift) & 0x0F];
                    }
                    sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
                }
            }

            yb += 16 * QK_MXFP4;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_mxfp4_chunked_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_slots6_chunk_map & map,
        device const char * chunk0,
        device const char * chunk1,
        device const char * chunk2,
        device const char * chunk3,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MXFP4;
    const int nb = args.ne00 / QK_MXFP4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    const short shift = it ? 4 : 0;

    float sumf[nr0] = {0.f};
    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const uint32_t chunk = map.chunk[expert_slot];
        const uint32_t slot = map.slot[expert_slot];
        device const char *chunk_cur =
            ds4_slots6_chunk_select(chunk, chunk0, chunk1, chunk2, chunk3);
        device const char *xbase =
            chunk_cur + (uint64_t)slot * map.slot_stride +
            (uint64_t)first_row * args.nb01;
        device const float *y = (device const float *)(token_src1 + (uint64_t)expert_slot * args.nb11);
        device const float *yb = y + ix * QK_MXFP4 + it * 16;

        for (int ib = ix; ib < nb; ib += 16) {
            float4 yl[4];
            device const float4 *y4 = (device const float4 *)yb;
            yl[0] = y4[0];
            yl[1] = y4[1];
            yl[2] = y4[2];
            yl[3] = y4[3];

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    device const block_mxfp4 *xb =
                        (device const block_mxfp4 *)(xbase + (uint64_t)row * args.nb01) + ib;
                    device const uchar *qs = xb->qs;

                    float4 acc = {0.f, 0.f, 0.f, 0.f};
                    for (short i = 0; i < 4; i++) {
                        acc[0] += yl[i][0] * ds4_kvalues_mxfp4[(qs[4 * i + 0] >> shift) & 0x0F];
                        acc[1] += yl[i][1] * ds4_kvalues_mxfp4[(qs[4 * i + 1] >> shift) & 0x0F];
                        acc[2] += yl[i][2] * ds4_kvalues_mxfp4[(qs[4 * i + 2] >> shift) & 0x0F];
                        acc[3] += yl[i][3] * ds4_kvalues_mxfp4[(qs[4 * i + 3] >> shift) & 0x0F];
                    }
                    sumf[row] += ds4_e8m0_to_float(xb->e) * (acc[0] + acc[1] + acc[2] + acc[3]);
                }
            }

            yb += 16 * QK_MXFP4;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q2_K_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int iid1 = tgpig.z / args.nei0;
    const int idx = tgpig.z % args.nei0;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_cur = src00;
    switch (idx) {
    case 1: src0_cur = src01; break;
    case 2: src0_cur = src02; break;
    case 3: src0_cur = src03; break;
    case 4: src0_cur = src04; break;
    case 5: src0_cur = src05; break;
    default: break;
    }
    device const block_q2_K *x =
        (device const block_q2_K *)(src0_cur + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[32];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }

        device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
        device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            if (first_row + row < args.ne0) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }
                float dall = dh[0];
                float dmin = dh[1] * 1.f / 16.f;
                sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                     (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                     (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                     (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                             dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                     sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            }

            qs += args.nb01 / 2;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 =
        (device float *)dst + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_mxfp4_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_cur = src00;
    switch (idx) {
    case 1: src0_cur = src01; break;
    case 2: src0_cur = src02; break;
    case 3: src0_cur = src03; break;
    case 4: src0_cur = src04; break;
    case 5: src0_cur = src05; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + expert * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

#define QK_NL 16

// Builds the compact per-expert work map used by batched MoE matmul. DS4 routes
// each token to a small fixed top-k list, so this turns token-major ids into
// expert-major slices that the tiled matmul can consume.
template<short ne20>
kernel void kernel_mul_mm_id_map0(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

// Zero the per-expert token counts (the id-map "tokens-per-expert" / htpe array at
// buffer offset 0) for experts the ANE handled in the ANE+GPU hybrid, so the GPU
// mul_mm_id path skips them (it returns early when counts[expert]==0) instead of
// double-computing what the ANE already scattered. One thread per expert.
// buffer(0) = int32 tokens-per-expert (ne02 entries); buffer(1) = 256-byte skip mask.
// (Completes the host call ds4_gpu_encode_zero_skipped_moe_counts added in 1becf06.)
kernel void kernel_dsv4_moe_zero_skipped_counts(
        device       int32_t * counts [[buffer(0)]],
        device const uchar   * skip   [[buffer(1)]],
        uint e [[thread_position_in_grid]]) {
    if (skip[e] != 0) {
        counts[e] = 0;
    }
}

// Host-visible map builders for the routed-expert counts used by DS4 graph
// shapes. Some arities are generic leftovers retained for nearby batch sizes.
template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Batched routed-expert matmul. It reads the expert-major map produced above,
// loads selected expert weights, and writes results back to token-major slots
// so the DS4 FFN can apply SwiGLU, weighting, and the down projection.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    /* NR1 (tokens-per-threadgroup along the expert-batch axis) is a template
     * parameter so the host can select wider 64/128 tiles for aligned prefill
     * batches (ported from antirez/ds4 PR #264 + the "Harden wide MoE tile
     * dispatch" fix). The dispatch grid-X must match this exactly. */

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1_base = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1_base >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;

    const short il0 = (tiitg % NL0);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    const short iy = 8*(tiitg % NL1);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    /*
     * The MMA body computes 64x32 output lanes per threadgroup. Wider NR1
     * variants reduce host/kernel overhead by letting one threadgroup walk
     * several adjacent 32-token expert-major slices serially. Each subtile is
     * scattered before the next one reuses the same 8 KiB scratch.
     */
    for (short r1_off = 0; r1_off < NR1; r1_off += 32) {
        const int r1 = r1_base + r1_off;
        if (r1 >= neh1) {
            break;
        }
        const short nr1 = (neh1 - r1 < 32) ? (neh1 - r1) : 32;
        const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

        short il = il0;

        const int id = ids_i32[im*args.ne21 + r1 + lr1];

        const short i11 = (id % args.ne20) % args.ne11;
        const short i12 = (id / args.ne20);

        device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

        device const T1 * y = (device const T1 *)(src1
            + args.nb13*i13
            + args.nb12*i12
            + args.nb11*i11
            + args.nb10*iy);

        for (short i = 0; i < 8; i++){
            mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }

        for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
            if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tiitg/NL0)/8;

                    const short lx = (tiitg/NL0)%8;
                    const short ly = i%8;

                    const short ib = 8*sx + sy;

                    *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
                }
            } else {
                S0_4x4 temp_a;
                dequantize_func(x, il, temp_a);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                FOR_UNROLL (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tiitg/NL0)/8;

                    const short lx = (tiitg/NL0)%8;
                    const short ly = i%8;

                    const short ib = 8*sx + sy;

                    *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
                }
            }

            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    const short sx = (tiitg%NL1);
                    const short sy = (tiitg/NL1)/8;

                    const short lx = i;
                    const short ly = (tiitg/NL1)%8;

                    const short ib = 4*sx + sy;

                    *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
            } else {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
            }

            il = (il + 2 < nl) ? il + 2 : il % 2;
            x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

            y += NK;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            // M5 compiles this as a tighter simdgroup_matrix load/MMA chain without no-op barriers.
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                if (!FC_mul_mm_m5_sgmatrix) simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                if (!FC_mul_mm_m5_sgmatrix) simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                if (!FC_mul_mm_m5_sgmatrix) simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short j = sgitg; j < nr1; j += 4) {
            const int idj = ids_i32[im*args.ne21 + r1 + j];

            const short ide = idj % args.ne20;
            const short idt = idj / args.ne20;

            device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            int i = tiisg;
            for (; i < nr0/4; i += 32) {
                *(D4 + i) = *(C4 + i);
            }

            i = (4*(nr0/4)) + tiisg;
            for (; i < nr0; i += 32) {
                *(D + i) = *(C + i);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Fused IQ2 gate+up grouped matmul with the established SwiGLU/route-weight
// epilogue.  This is Ivan Fioravanti's shared-B-tile technique from PR #555,
// adapted to DS4's wide NR1 kernels: a threadgroup walks adjacent 32-token
// subtiles while retaining the 64/128/256-token dispatch contract used by the
// M3 Ultra short-prefill path.
template<short NR1, typename block_q, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id_pair_swiglu_f16_impl(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa_gate = (threadgroup half *)(shmem);
    threadgroup half *sa_up   = (threadgroup half *)(shmem + 4096);
    threadgroup half *sb      = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1_base = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *)htpe;
    device const int32_t  * ids_i32 = (device const int32_t  *)hids;
    const int32_t neh1 = tpe_u32[im];
    if (r1_base >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short il0 = tiitg % NL0;
    const short i13 = 0;
    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short offset1 = il0/QK_NL;
    const short iy = 8*(tiitg % NL1);

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short r1_off = 0; r1_off < NR1; r1_off += 32) {
        const int r1 = r1_base + r1_off;
        if (r1 >= neh1) break;

        const short nr1 = (neh1 - r1 < 32) ? (neh1 - r1) : 32;
        const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;
        short il = il0;

        const int id = ids_i32[im*args.ne21 + r1 + lr1];
        const short i11 = (id % args.ne20) % args.ne11;
        const short i12 = id / args.ne20;

        device const block_q *xg =
            (device const block_q *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
        device const block_q *xu =
            (device const block_q *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;
        device const float *y = (device const float *)(src1
            + args.nb13*i13
            + args.nb12*i12
            + args.nb11*i11
            + args.nb10*iy);

        for (short i = 0; i < 8; i++) {
            mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
            mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }

        for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
            half4x4 temp_gate;
            half4x4 temp_up;
            dequantize_func(xg, il, temp_gate);
            dequantize_func(xu, il, temp_up);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short sx_b = tiitg%NL1;
            const short sy_b = (tiitg/NL1)/8;
            const short ly_b = (tiitg/NL1)%8;
            const short ib_b = 4*sx_b + sy_b;
            *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
                (half2x4)(*((device float2x4 *)y));

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa_gate + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
                *(sa_up   + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma_gate = sa_gate + 4*64*(sgitg%2);
            threadgroup const half *lsma_up   = sa_up   + 4*64*(sgitg%2);
            threadgroup const half *lsmb      = sb      + 2*64*(sgitg/2);

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma_g[i], lsma_gate + 64*i, 8, 0, false);
                    simdgroup_load(ma_u[i], lsma_up   + 64*i, 8, 0, false);
                }
                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                FOR_UNROLL (short i = 0; i < 8; i++) {
                    simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma_g[i%4], mc_gate[i]);
                    simdgroup_multiply_accumulate(mc_up[i],   mb[i/4], ma_u[i%4], mc_up[i]);
                }
                lsma_gate += 8*64;
                lsma_up   += 8*64;
                lsmb      += 4*64;
            }

            il = (il + 2 < QK_NL) ? il + 2 : il % 2;
            xg = (il < 2) ? xg + (2 + QK_NL - 1)/QK_NL : xg;
            xu = (il < 2) ? xu + (2 + QK_NL - 1)/QK_NL : xu;
            y += NK;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *temp_gate = (threadgroup float *)shmem;
        threadgroup float *temp_up = temp_gate + NR0*32;
        threadgroup float *temp_gate_str =
            temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        threadgroup float *temp_up_str =
            temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
            simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float c = act.clamp_value;
        for (short j = sgitg; j < nr1; j += 4) {
            const int idj = ids_i32[im*args.ne21 + r1 + j];
            const short ide = idj % args.ne20;
            const short idt = idj / args.ne20;
            device half *D = (device half *)(dst_mid +
                ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
            device const float *w =
                (device const float *)(weights + (uint64_t)idj*act.weight_stride);
            const float route_weight = w[0];
            threadgroup float *Cg = temp_gate + j*NR0;
            threadgroup float *Cu = temp_up + j*NR0;

            for (int i = tiisg; i < nr0; i += 32) {
                float g = Cg[i];
                float u = Cu[i];
                if (c > 1.0e-6f) {
                    g = min(g, c);
                    u = clamp(u, -c, c);
                }
                const float silu = g / (1.0f + exp(-g));
                D[i] = (half)(silu*u*route_weight);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<32, block_iq2_xxs, dequantize_iq2_xxs>) mul_mm_id_pair_swiglu_f16_iq2_n32;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<64, block_iq2_xxs, dequantize_iq2_xxs>) mul_mm_id_pair_swiglu_f16_iq2_n64;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<128, block_iq2_xxs, dequantize_iq2_xxs>) mul_mm_id_pair_swiglu_f16_iq2_n128;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16_impl<256, block_iq2_xxs, dequantize_iq2_xxs>) mul_mm_id_pair_swiglu_f16_iq2_n256;

template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16")]] kernel mul_mm_id_pair_swiglu_f16_iq2_n32 kernel_mul_mm_id_pair_swiglu_f16_impl<32, block_iq2_xxs, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_n64")]] kernel mul_mm_id_pair_swiglu_f16_iq2_n64 kernel_mul_mm_id_pair_swiglu_f16_impl<64, block_iq2_xxs, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_n128")]] kernel mul_mm_id_pair_swiglu_f16_iq2_n128 kernel_mul_mm_id_pair_swiglu_f16_impl<128, block_iq2_xxs, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_n256")]] kernel mul_mm_id_pair_swiglu_f16_iq2_n256 kernel_mul_mm_id_pair_swiglu_f16_impl<256, block_iq2_xxs, dequantize_iq2_xxs>;

typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_f16_rhs;

// Host-visible batched MoE matmul variants for the DS4 quant formats.
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_n64")]]  kernel mul_mm_id kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_n128")]] kernel mul_mm_id kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_n256")]] kernel mul_mm_id kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_n64")]]  kernel mul_mm_id kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_n128")]] kernel mul_mm_id kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_n256")]] kernel mul_mm_id kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_n64")]]  kernel mul_mm_id kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_n128")]] kernel mul_mm_id kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_n256")]] kernel mul_mm_id kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]      kernel mul_mm_id kernel_mul_mm_id<32,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_n64")]]  kernel mul_mm_id kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_n128")]] kernel mul_mm_id kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_n256")]] kernel mul_mm_id kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq1_m,   QK_NL, dequantize_iq1_m,   half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16_n64")]]  kernel mul_mm_id_f16_rhs kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16_n128")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16_n256")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16_n64")]]  kernel mul_mm_id_f16_rhs kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16_n128")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16_n256")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16_n64")]]  kernel mul_mm_id_f16_rhs kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16_n128")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16_n256")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]      kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16_n64")]]  kernel mul_mm_id_f16_rhs kernel_mul_mm_id<64,  half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16_n128")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<128, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16_n256")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<256, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4,   2,     dequantize_mxfp4,   half, half4x4, half, half2x4>;

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef iq3xxs_grid
#undef iq1s_grid_gpu
#undef QK_K
#undef QK_MXFP4
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_Q5_K
#undef N_R0_Q6_K
#undef N_R0_IQ2_XXS
#undef N_R0_IQ1_M
#undef N_R0_IQ3_XXS
#undef N_R0_IQ4_XS
#undef N_R0_MXFP4

// -----------------------------------------------------------------------------
// Flash-MoE GPU dedup kernels (for prefill with paged experts from SSD sidecar)
// Goal: avoid reading the entire router top-k output back to CPU on every chunk.
// -----------------------------------------------------------------------------

struct FlashDedupHistogramArgs {
    uint32_t n_pairs;
    uint32_t n_expert;
};

kernel void kernel_flash_moe_dedup_histogram(
        constant FlashDedupHistogramArgs & args [[buffer(0)]],
        device const int32_t * selected [[buffer(1)]],
        device atomic_uint * counts [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
{
    if (gid >= args.n_pairs) return;

    int32_t e = selected[gid];
    if (e >= 0 && (uint32_t)e < args.n_expert) {
        atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    }
}

// Compact kernel: given router selected/weights, and starting offsets,
// atomically claim slots and write (token_id, weight) into the per-expert sections.
struct FlashDedupCompactArgs {
    uint32_t n_pairs;
    uint32_t expert_used;   // usually 8
    uint32_t n_expert;
};

kernel void kernel_flash_moe_dedup_compact(
        constant FlashDedupCompactArgs & args [[buffer(0)]],
        device const int32_t * selected   [[buffer(1)]],
        device const float   * pair_weights [[buffer(2)]],
        device atomic_uint * offsets     [[buffer(3)]],   // starting write pos per expert, mutated atomically
        device int32_t  * out_tokens  [[buffer(4)]],
        device float    * out_weights [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
{
    if (gid >= args.n_pairs) return;

    int32_t e = selected[gid];
    if (e < 0 || (uint32_t)e >= args.n_expert) return;

    uint32_t slot = atomic_fetch_add_explicit(&offsets[e], 1u, memory_order_relaxed);
    uint32_t token = gid / args.expert_used;

    out_tokens[slot]  = (int32_t)token;
    out_weights[slot] = pair_weights[gid];
}

// Small slice copy helpers (used to feed per-expert working lists from the big compacted dedup buffers)
kernel void kernel_flash_copy_i32_slice(
        device const int32_t * src [[buffer(0)]],
        device int32_t       * dst [[buffer(1)]],
        constant uint32_t & src_offset [[buffer(2)]],
        constant uint32_t & count      [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[gid] = src[src_offset + gid];
}

kernel void kernel_dspark_moe_group_routes_n5(
        constant DSparkMoeGroupRoutesArgs & args [[buffer(0)]],
        device const int32_t * selected [[buffer(1)]],
        device uint32_t * group_count [[buffer(2)]],
        device int32_t  * group_experts [[buffer(3)]],
        device uint32_t * group_offsets [[buffer(4)]],
        device uint32_t * group_pairs [[buffer(5)]],
        device uint32_t * group_counts [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    const uint max_pairs = min(args.max_pairs, 30u);
    const uint n_pairs = min(args.n_pairs, max_pairs);
    *group_count = 0u;
    for (uint i = 0; i < max_pairs; i++) {
        group_experts[i] = -1;
        group_counts[i] = 0u;
        group_pairs[i] = 0xFFFFFFFFu;
    }
    for (uint i = 0; i <= max_pairs; i++) {
        group_offsets[i] = 0u;
    }

    uint gc = 0u;
    for (uint pair = 0; pair < n_pairs; pair++) {
        const int32_t expert_i = selected[pair];
        if (expert_i < 0 || uint(expert_i) >= args.n_expert) continue;

        uint group = gc;
        for (uint g = 0; g < gc; g++) {
            if (group_experts[g] == expert_i) {
                group = g;
                break;
            }
        }
        if (group == gc) {
            if (gc >= max_pairs) continue;
            group_experts[gc] = expert_i;
            gc++;
        }
        group_counts[group]++;
    }

    uint cursor[30];
    uint offset = 0u;
    for (uint g = 0; g < gc; g++) {
        group_offsets[g] = offset;
        cursor[g] = offset;
        offset += group_counts[g];
    }
    group_offsets[gc] = offset;

    for (uint pair = 0; pair < n_pairs; pair++) {
        const int32_t expert_i = selected[pair];
        if (expert_i < 0 || uint(expert_i) >= args.n_expert) continue;
        for (uint g = 0; g < gc; g++) {
            if (group_experts[g] == expert_i) {
                const uint dst = cursor[g]++;
                if (dst < max_pairs) group_pairs[dst] = pair;
                break;
            }
        }
    }

    *group_count = gc;
}

kernel void kernel_flash_copy_f32_slice(
        device const float * src [[buffer(0)]],
        device float       * dst [[buffer(1)]],
        constant uint32_t & src_offset [[buffer(2)]],
        constant uint32_t & count      [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[gid] = src[src_offset + gid];
}
