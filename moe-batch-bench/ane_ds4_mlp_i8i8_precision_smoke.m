// Precision smoke for the shared/routed MLP i8w+i8x tiled-fused ANE path.
// It reports two errors:
//   1. ANE vs CPU reference for the same quantized graph (implementation check).
//   2. Quantized CPU graph vs fp16 CPU graph (scale/precision check).

#import <Foundation/Foundation.h>
#include "ane_ds4_mlp_int8w.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) { m = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

static int8_t quant_i8(float v, float qscale) {
    int q = (int)lrintf(v * qscale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static float val(uint64_t i, int mod, float scale) {
    return (float)(((int)(i % (uint64_t)mod)) - (mod / 2)) * scale;
}

static void fill_inputs(uint16_t *wg, uint16_t *wu, uint16_t *wd, uint16_t *x,
                        int H, int I, int B, float scale) {
    for (uint64_t i = 0; i < (uint64_t)H * I; i++) {
        wg[i] = f32_to_f16_bits(val(i * 17u + 1u, 29, 0.018f * scale));
        wu[i] = f32_to_f16_bits(val(i * 19u + 3u, 31, 0.016f * scale));
    }
    for (uint64_t i = 0; i < (uint64_t)I * H; i++) {
        wd[i] = f32_to_f16_bits(val(i * 23u + 5u, 37, 0.017f * scale));
    }
    for (uint64_t i = 0; i < (uint64_t)B * H; i++) {
        x[i] = f32_to_f16_bits(val(i * 11u + 7u, 41, 0.035f * scale));
    }
}

static void quantize_f16_fixed(const uint16_t *src, int8_t *dst, uint64_t n, float qscale) {
    for (uint64_t i = 0; i < n; i++) dst[i] = quant_i8(f16_bits_to_f32(src[i]), qscale);
}

static void ref_fp16(float *out, const uint16_t *wg, const uint16_t *wu,
                     const uint16_t *wd, const uint16_t *x,
                     int H, int I, int B) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < H; c++) {
            double sum = 0.0;
            for (int j = 0; j < I; j++) {
                double g = 0.0, u = 0.0;
                for (int h = 0; h < H; h++) {
                    const double xv = f16_bits_to_f32(x[(uint64_t)b * H + h]);
                    g += xv * f16_bits_to_f32(wg[(uint64_t)h * I + j]);
                    u += xv * f16_bits_to_f32(wu[(uint64_t)h * I + j]);
                }
                if (g < -10.0) g = -10.0; if (g > 10.0) g = 10.0;
                if (u < -10.0) u = -10.0; if (u > 10.0) u = 10.0;
                const double hidden = (g / (1.0 + exp(-g))) * u;
                sum += hidden * f16_bits_to_f32(wd[(uint64_t)j * H + c]);
            }
            out[(uint64_t)b * H + c] = (float)sum;
        }
    }
}

static void ref_i8(float *out, const int8_t *wg, const int8_t *wu,
                   const int8_t *wd, const int8_t *x,
                   int H, int I, int B,
                   float w_scale, float x_scale, float mid_scale) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < H; c++) {
            double sum = 0.0;
            for (int j = 0; j < I; j++) {
                double g = 0.0, u = 0.0;
                for (int h = 0; h < H; h++) {
                    const double xv = (double)x[(uint64_t)b * H + h] * x_scale;
                    g += xv * ((double)wg[(uint64_t)h * I + j] * w_scale);
                    u += xv * ((double)wu[(uint64_t)h * I + j] * w_scale);
                }
                if (g < -10.0) g = -10.0; if (g > 10.0) g = 10.0;
                if (u < -10.0) u = -10.0; if (u > 10.0) u = 10.0;
                double hidden = (g / (1.0 + exp(-g))) * u;
                double q = nearbyint(hidden / (double)mid_scale);
                if (q < -128.0) q = -128.0;
                if (q > 127.0) q = 127.0;
                hidden = q * (double)mid_scale;
                sum += hidden * ((double)wd[(uint64_t)j * H + c] * w_scale);
            }
            out[(uint64_t)b * H + c] = (float)sum;
        }
    }
}

static void metrics(const char *name, const float *ref, const float *got, uint64_t n) {
    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    uint64_t worst = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float err = fabsf(got[i] - ref[i]);
        const float rel = err / fmaxf(fabsf(ref[i]), 1.0e-6f);
        sq += (double)err * err;
        ref_sq += (double)ref[i] * ref[i];
        if (err > max_abs) { max_abs = err; max_rel = rel; worst = i; }
    }
    const double rms = sqrt(sq / (double)n);
    const double rel_rms = sqrt(sq / fmax(ref_sq, 1.0e-30));
    printf("%s max_abs=%g max_rel=%g rms=%g rel_rms=%g worst=%llu ref=%g got=%g\n",
           name, max_abs, max_rel, rms, rel_rms,
           (unsigned long long)worst, ref[worst], got[worst]);
}

int main(int argc, const char **argv) {
    int H = 256, I = 128, B = 8;
    float scale = 1.0f, wq = 512.0f, xq = 32.0f, midq = 32.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-H") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I") && i + 1 < argc) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-scale") && i + 1 < argc) scale = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-wq") && i + 1 < argc) wq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-xq") && i + 1 < argc) xq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-midq") && i + 1 < argc) midq = strtof(argv[++i], NULL);
    }
    if (H <= 0 || I <= 0 || B <= 0 || !(wq > 0.0f) || !(xq > 0.0f) || !(midq > 0.0f)) return 2;

    const uint64_t hi = (uint64_t)H * I;
    const uint64_t bh = (uint64_t)B * H;
    uint16_t *wg = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *wu = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *wd = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *x = (uint16_t *)calloc((size_t)bh, sizeof(uint16_t));
    uint16_t *y_ane_f16 = (uint16_t *)calloc((size_t)bh, sizeof(uint16_t));
    int8_t *wgq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *wuq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *wdq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *xqv = (int8_t *)calloc((size_t)bh, 1);
    float *ref16 = (float *)calloc((size_t)bh, sizeof(float));
    float *ref8 = (float *)calloc((size_t)bh, sizeof(float));
    float *ane = (float *)calloc((size_t)bh, sizeof(float));
    if (!wg || !wu || !wd || !x || !y_ane_f16 || !wgq || !wuq || !wdq || !xqv || !ref16 || !ref8 || !ane) return 3;

    fill_inputs(wg, wu, wd, x, H, I, B, scale);
    quantize_f16_fixed(wg, wgq, hi, wq);
    quantize_f16_fixed(wu, wuq, hi, wq);
    quantize_f16_fixed(wd, wdq, hi, wq);
    quantize_f16_fixed(x, xqv, bh, xq);

    const float ws = 1.0f / wq, xs = 1.0f / xq, mids = 1.0f / midq;
    ds4_ane_mlp_int8w_ctx *ctx = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, ws, xs, mids);
    if (!ctx) { fprintf(stderr, "create failed H=%d I=%d B=%d\n", H, I, B); return 4; }
    if (!ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wgq, wuq, wdq, xqv, y_ane_f16)) {
        fprintf(stderr, "eval failed\n");
        return 5;
    }

    ref_fp16(ref16, wg, wu, wd, x, H, I, B);
    ref_i8(ref8, wgq, wuq, wdq, xqv, H, I, B, ws, xs, mids);
    for (uint64_t i = 0; i < bh; i++) ane[i] = f16_bits_to_f32(y_ane_f16[i]);

    printf("i8i8 precision H=%d I=%d B=%d scale=%g wq=%g xq=%g midq=%g\n", H, I, B, scale, wq, xq, midq);
    metrics("ane_vs_i8_cpu", ref8, ane, bh);
    metrics("i8_cpu_vs_fp16_cpu", ref16, ref8, bh);
    metrics("ane_vs_fp16_cpu", ref16, ane, bh);

    ds4_ane_mlp_int8w_destroy(ctx);
    free(wg); free(wu); free(wd); free(x); free(y_ane_f16);
    free(wgq); free(wuq); free(wdq); free(xqv);
    free(ref16); free(ref8); free(ane);
    return 0;
}
