// Precision smoke for the shared/routed MLP i8w+i8x tiled-fused ANE path.
// It reports two errors:
//   1. ANE vs CPU reference for the same quantized graph (implementation check).
//   2. Quantized CPU graph vs fp16 CPU graph (scale/precision check).

#import <Foundation/Foundation.h>
#include "ds4_ane_mlp_int8w.h"
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

static void ref_fp16_route_before(float *out, const uint16_t *wg, const uint16_t *wu,
                                  const uint16_t *wd, const uint16_t *x,
                                  const float *route, int H, int I, int B) {
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
                const double hidden = (g / (1.0 + exp(-g))) * u * (double)route[b];
                sum += hidden * f16_bits_to_f32(wd[(uint64_t)j * H + c]);
            }
            out[(uint64_t)b * H + c] = (float)sum;
        }
    }
}

static void ref_i8_route_before(float *out, const int8_t *wg, const int8_t *wu,
                                const int8_t *wd, const int8_t *x,
                                const float *route, int H, int I, int B,
                                float w_scale, float x_scale, float mid_scale,
                                uint64_t *sat_out) {
    uint64_t sat = 0;
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
                double hidden = (g / (1.0 + exp(-g))) * u * (double)route[b];
                double q = nearbyint(hidden / (double)mid_scale);
                if (q < -128.0) { q = -128.0; sat++; }
                if (q > 127.0) { q = 127.0; sat++; }
                hidden = q * (double)mid_scale;
                sum += hidden * ((double)wd[(uint64_t)j * H + c] * w_scale);
            }
            out[(uint64_t)b * H + c] = (float)sum;
        }
    }
    if (sat_out) *sat_out = sat;
}

static void ref_i8_row0(float *out, const int8_t *wg, const int8_t *wu,
                        const int8_t *wd, const int8_t *x,
                        const float *route, int routed,
                        int H, int I,
                        float w_scale, float x_scale, float mid_scale,
                        uint64_t *sat_out) {
    float *hidden = (float *)calloc((size_t)I, sizeof(float));
    uint64_t sat = 0;
    if (!hidden) return;
    const float route0 = route ? route[0] : 1.0f;
    for (int j = 0; j < I; j++) {
        double g = 0.0, u = 0.0;
        for (int h = 0; h < H; h++) {
            const double xv = (double)x[h] * x_scale;
            g += xv * ((double)wg[(uint64_t)h * I + j] * w_scale);
            u += xv * ((double)wu[(uint64_t)h * I + j] * w_scale);
        }
        if (g < -10.0) g = -10.0; if (g > 10.0) g = 10.0;
        if (u < -10.0) u = -10.0; if (u > 10.0) u = 10.0;
        double v = (g / (1.0 + exp(-g))) * u;
        if (routed) v *= (double)route0;
        double q = nearbyint(v / (double)mid_scale);
        if (q < -128.0) { q = -128.0; sat++; }
        if (q > 127.0) { q = 127.0; sat++; }
        hidden[j] = (float)(q * (double)mid_scale);
    }
    for (int c = 0; c < H; c++) {
        double sum = 0.0;
        for (int j = 0; j < I; j++) {
            sum += (double)hidden[j] * ((double)wd[(uint64_t)j * H + c] * w_scale);
        }
        out[c] = (float)sum;
    }
    if (sat_out) *sat_out = sat;
    free(hidden);
}

static void ref_fp16_row0(float *out, const uint16_t *wg, const uint16_t *wu,
                          const uint16_t *wd, const uint16_t *x,
                          const float *route, int routed,
                          int H, int I) {
    float *hidden = (float *)calloc((size_t)I, sizeof(float));
    if (!hidden) return;
    const float route0 = route ? route[0] : 1.0f;
    for (int j = 0; j < I; j++) {
        double g = 0.0, u = 0.0;
        for (int h = 0; h < H; h++) {
            const double xv = f16_bits_to_f32(x[h]);
            g += xv * f16_bits_to_f32(wg[(uint64_t)h * I + j]);
            u += xv * f16_bits_to_f32(wu[(uint64_t)h * I + j]);
        }
        if (g < -10.0) g = -10.0; if (g > 10.0) g = 10.0;
        if (u < -10.0) u = -10.0; if (u > 10.0) u = 10.0;
        double v = (g / (1.0 + exp(-g))) * u;
        if (routed) v *= (double)route0;
        hidden[j] = (float)v;
    }
    for (int c = 0; c < H; c++) {
        double sum = 0.0;
        for (int j = 0; j < I; j++) {
            sum += (double)hidden[j] * f16_bits_to_f32(wd[(uint64_t)j * H + c]);
        }
        out[c] = (float)sum;
    }
    free(hidden);
}

static void mul_route_rows(float *dst, const float *src, const float *route, int H, int B) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < H; c++) {
            dst[(uint64_t)b * H + c] = src[(uint64_t)b * H + c] * route[b];
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

static void metrics_f16_scaled_rows(const char *name,
                                    const uint16_t *ref_f16,
                                    const uint16_t *got_f16,
                                    const float *route,
                                    int H,
                                    int B) {
    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    uint64_t worst = 0;
    const uint64_t n = (uint64_t)B * (uint64_t)H;
    for (int b = 0; b < B; b++) {
        const float rw = route ? route[b] : 1.0f;
        for (int c = 0; c < H; c++) {
            const uint64_t idx = (uint64_t)b * (uint64_t)H + (uint64_t)c;
            const float ref = f16_bits_to_f32(ref_f16[idx]) * rw;
            const float got = f16_bits_to_f32(got_f16[idx]);
            const float err = fabsf(got - ref);
            const float rel = err / fmaxf(fabsf(ref), 1.0e-6f);
            sq += (double)err * err;
            ref_sq += (double)ref * (double)ref;
            if (err > max_abs) { max_abs = err; max_rel = rel; worst = idx; }
        }
    }
    const double rms = sqrt(sq / (double)n);
    const double rel_rms = sqrt(sq / fmax(ref_sq, 1.0e-30));
    const int wb = H > 0 ? (int)(worst / (uint64_t)H) : 0;
    printf("%s max_abs=%g max_rel=%g rms=%g rel_rms=%g worst=%llu ref_scaled=%g got=%g route=%g\n",
           name, max_abs, max_rel, rms, rel_rms,
           (unsigned long long)worst,
           f16_bits_to_f32(ref_f16[worst]) * (route ? route[wb] : 1.0f),
           f16_bits_to_f32(got_f16[worst]),
           route ? route[wb] : 1.0f);
}

int main(int argc, const char **argv) {
    int H = 256, I = 128, B = 8, routed = 0, ane_parity = 0, const_route = 0, row0_ref = 0;
    float scale = 1.0f, wq = 512.0f, xq = 32.0f, midq = 32.0f;
    float refmidq = 0.0f, route_weight = 1.0f / 6.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-H") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I") && i + 1 < argc) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-scale") && i + 1 < argc) scale = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-wq") && i + 1 < argc) wq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-xq") && i + 1 < argc) xq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-midq") && i + 1 < argc) midq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-refmidq") && i + 1 < argc) refmidq = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-route") && i + 1 < argc) route_weight = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-routed")) routed = 1;
        else if (!strcmp(argv[i], "-ane-parity") || !strcmp(argv[i], "-ane-parity-only")) ane_parity = 1;
        else if (!strcmp(argv[i], "-const-route")) const_route = 1;
        else if (!strcmp(argv[i], "-row0-ref")) row0_ref = 1;
    }
    if (!(refmidq > 0.0f)) refmidq = midq;
    if (H <= 0 || I <= 0 || B <= 0 || !(wq > 0.0f) || !(xq > 0.0f) || !(midq > 0.0f) ||
        !(refmidq > 0.0f) || !(route_weight > 0.0f)) return 2;

    const uint64_t hi = (uint64_t)H * I;
    const uint64_t bh = (uint64_t)B * H;
    uint16_t *wg = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *wu = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *wd = (uint16_t *)calloc((size_t)hi, sizeof(uint16_t));
    uint16_t *x = (uint16_t *)calloc((size_t)bh, sizeof(uint16_t));
    uint16_t *y_ane_f16 = (uint16_t *)calloc((size_t)bh, sizeof(uint16_t));
    uint16_t *y_ref_f16 = (uint16_t *)calloc((size_t)bh, sizeof(uint16_t));
    uint16_t *route_f16 = (uint16_t *)calloc((size_t)B * (size_t)I, sizeof(uint16_t));
    int8_t *wgq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *wuq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *wdq = (int8_t *)calloc((size_t)hi, 1);
    int8_t *xqv = (int8_t *)calloc((size_t)bh, 1);
    float *ref16 = (float *)calloc((size_t)bh, sizeof(float));
    float *ref8 = (float *)calloc((size_t)bh, sizeof(float));
    float *ref16_route_before = (float *)calloc((size_t)bh, sizeof(float));
    float *ref8_route_before = (float *)calloc((size_t)bh, sizeof(float));
    float *ref8_route_after = (float *)calloc((size_t)bh, sizeof(float));
    float *ane = (float *)calloc((size_t)bh, sizeof(float));
    float *ane_route_after = (float *)calloc((size_t)bh, sizeof(float));
    float *route = (float *)calloc((size_t)B, sizeof(float));
    if (!wg || !wu || !wd || !x || !y_ane_f16 || !y_ref_f16 || !route_f16 || !wgq || !wuq || !wdq || !xqv ||
        !ref16 || !ref8 || !ref16_route_before || !ref8_route_before ||
        !ref8_route_after || !ane || !ane_route_after || !route) return 3;

    fill_inputs(wg, wu, wd, x, H, I, B, scale);
    quantize_f16_fixed(wg, wgq, hi, wq);
    quantize_f16_fixed(wu, wuq, hi, wq);
    quantize_f16_fixed(wd, wdq, hi, wq);
    quantize_f16_fixed(x, xqv, bh, xq);
    for (int b = 0; b < B; b++) {
        route[b] = const_route ? route_weight : route_weight * (0.85f + 0.05f * (float)(b % 7));
        for (int j = 0; j < I; j++) {
            route_f16[(uint64_t)b * I + j] = f32_to_f16_bits(route[b]);
        }
    }

    const float ws = 1.0f / wq, xs = 1.0f / xq, mids = 1.0f / midq, refmids = 1.0f / refmidq;
    if (ane_parity) {
        ds4_ane_mlp_int8w_ctx *ctx_ref =
            ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, ws, xs, mids);
        ds4_ane_mlp_int8w_ctx *ctx_routed =
            ds4_ane_mlp_i8w_i8x_tiled_fused_routed_create(H, I, B, ws, xs, mids);
        if (!ctx_ref || !ctx_routed) {
            fprintf(stderr, "create failed H=%d I=%d B=%d parity=1 ctx_ref=%p ctx_routed=%p\n",
                    H, I, B, (void *)ctx_ref, (void *)ctx_routed);
            ds4_ane_mlp_int8w_destroy(ctx_ref);
            ds4_ane_mlp_int8w_destroy(ctx_routed);
            return 4;
        }
        int ok_ref = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx_ref, wgq, wuq, wdq, xqv, y_ref_f16);
        int ok_routed = ds4_ane_mlp_i8w_i8x_tiled_fused_routed_eval(ctx_routed, wgq, wuq, wdq, xqv, route_f16, y_ane_f16);
        if (!ok_ref || !ok_routed) {
            fprintf(stderr, "eval failed parity=1 ref=%d routed=%d\n", ok_ref, ok_routed);
            ds4_ane_mlp_int8w_destroy(ctx_ref);
            ds4_ane_mlp_int8w_destroy(ctx_routed);
            return 5;
        }
        printf("i8i8 ANE parity H=%d I=%d B=%d scale=%g wq=%g xq=%g midq=%g route=%g const_route=%d order=%s\n",
               H, I, B, scale, wq, xq, midq, route_weight, const_route,
               getenv("DS4_ANE_ROUTED_ORDER") ? getenv("DS4_ANE_ROUTED_ORDER") : "default");
        metrics_f16_scaled_rows("mode13_routed_vs_mode6_scaled", y_ref_f16, y_ane_f16, route, H, B);
        ds4_ane_mlp_int8w_destroy(ctx_ref);
        ds4_ane_mlp_int8w_destroy(ctx_routed);
        free(wg); free(wu); free(wd); free(x); free(y_ane_f16); free(y_ref_f16); free(route_f16);
        free(wgq); free(wuq); free(wdq); free(xqv);
        free(ref16); free(ref8); free(ref16_route_before); free(ref8_route_before);
        free(ref8_route_after); free(ane); free(ane_route_after); free(route);
        return 0;
    }

    ds4_ane_mlp_int8w_ctx *ctx = routed ?
        ds4_ane_mlp_i8w_i8x_tiled_fused_routed_create(H, I, B, ws, xs, mids) :
        ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, ws, xs, mids);
    if (!ctx) { fprintf(stderr, "create failed H=%d I=%d B=%d\n", H, I, B); return 4; }
    int ok = routed ?
        ds4_ane_mlp_i8w_i8x_tiled_fused_routed_eval(ctx, wgq, wuq, wdq, xqv, route_f16, y_ane_f16) :
        ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wgq, wuq, wdq, xqv, y_ane_f16);
    if (!ok) {
        fprintf(stderr, "eval failed\n");
        return 5;
    }
    if (row0_ref) {
        uint64_t row0_sat = 0;
        ref_i8_row0(ref8, wgq, wuq, wdq, xqv, route, routed,
                    H, I, ws, xs, refmids, &row0_sat);
        ref_fp16_row0(ref16, wg, wu, wd, x, route, routed, H, I);
        for (int c = 0; c < H; c++) ane[c] = f16_bits_to_f32(y_ane_f16[c]);
        printf("i8i8 row0 H=%d I=%d B=%d scale=%g wq=%g xq=%g midq=%g refmidq=%g route=%g routed=%d const_route=%d row0_sat=%llu/%d order=%s\n",
               H, I, B, scale, wq, xq, midq, refmidq, route_weight, routed, const_route,
               (unsigned long long)row0_sat, I,
               getenv("DS4_ANE_ROUTED_ORDER") ? getenv("DS4_ANE_ROUTED_ORDER") : "default");
        metrics("ane_row0_vs_cpu_i8_row0", ref8, ane, (uint64_t)H);
        metrics("cpu_i8_row0_vs_fp16_row0", ref16, ref8, (uint64_t)H);
        metrics("ane_row0_vs_fp16_row0", ref16, ane, (uint64_t)H);
        ds4_ane_mlp_int8w_destroy(ctx);
        free(wg); free(wu); free(wd); free(x); free(y_ane_f16); free(y_ref_f16); free(route_f16);
        free(wgq); free(wuq); free(wdq); free(xqv);
        free(ref16); free(ref8); free(ref16_route_before); free(ref8_route_before);
        free(ref8_route_after); free(ane); free(ane_route_after); free(route);
        return 0;
    }

    ref_fp16(ref16, wg, wu, wd, x, H, I, B);
    ref_i8(ref8, wgq, wuq, wdq, xqv, H, I, B, ws, xs, mids);
    ref_fp16_route_before(ref16_route_before, wg, wu, wd, x, route, H, I, B);
    uint64_t route_before_sat = 0;
    ref_i8_route_before(ref8_route_before, wgq, wuq, wdq, xqv, route, H, I, B,
                        ws, xs, refmids, &route_before_sat);
    for (uint64_t i = 0; i < bh; i++) ane[i] = f16_bits_to_f32(y_ane_f16[i]);
    mul_route_rows(ref8_route_after, ref8, route, H, B);
    if (routed) {
        memcpy(ane_route_after, ane, (size_t)bh * sizeof(float));
    } else {
        mul_route_rows(ane_route_after, ane, route, H, B);
    }

    printf("i8i8 precision H=%d I=%d B=%d scale=%g wq=%g xq=%g midq=%g refmidq=%g route=%g routed=%d route_before_sat=%llu/%llu\n",
           H, I, B, scale, wq, xq, midq, refmidq, route_weight, routed,
           (unsigned long long)route_before_sat,
           (unsigned long long)((uint64_t)B * (uint64_t)H * (uint64_t)I));
    metrics("ane_vs_i8_cpu", ref8, ane, bh);
    metrics("i8_cpu_vs_fp16_cpu", ref16, ref8, bh);
    metrics("ane_vs_fp16_cpu", ref16, ane, bh);
    metrics("ane_route_after_vs_gpu_i8_route_before", ref8_route_before, ane_route_after, bh);
    metrics("cpu_i8_route_after_vs_gpu_i8_route_before", ref8_route_before, ref8_route_after, bh);
    metrics("gpu_i8_route_before_vs_fp16_route_before", ref16_route_before, ref8_route_before, bh);
    metrics("ane_route_after_vs_fp16_route_before", ref16_route_before, ane_route_after, bh);

    ds4_ane_mlp_int8w_destroy(ctx);
    free(wg); free(wu); free(wd); free(x); free(y_ane_f16); free(y_ref_f16); free(route_f16);
    free(wgq); free(wuq); free(wdq); free(xqv);
    free(ref16); free(ref8); free(ref16_route_before); free(ref8_route_before);
    free(ref8_route_after); free(ane); free(ane_route_after); free(route);
    return 0;
}
