// Tiny value probe for the in-memory ANE DS4 MLP helper.

#import "ane_ds4_mlp_int8w.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) {
        m = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
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
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float val(int i, int mod, float scale) {
    return (float)((i % mod) - (mod / 2)) * scale;
}

static int streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static void fill_dense(uint16_t *wg, uint16_t *wu, uint16_t *wd, uint16_t *x,
                       int H, int I, int B, float scale) {
    for (int i = 0; i < H * I; i++) {
        wg[i] = f32_to_f16_bits(val(i * 17 + 1, 29, 0.018f * scale));
        wu[i] = f32_to_f16_bits(val(i * 19 + 3, 31, 0.016f * scale));
    }
    for (int i = 0; i < I * H; i++) {
        wd[i] = f32_to_f16_bits(val(i * 23 + 5, 37, 0.017f * scale));
    }
    for (int i = 0; i < B * H; i++) {
        x[i] = f32_to_f16_bits(val(i * 11 + 7, 41, 0.035f * scale));
    }
}

static void fill_onehot(uint16_t *wg, uint16_t *wu, uint16_t *wd, uint16_t *x,
                        int H, int I, int B, float scale) {
    memset(wg, 0, (size_t)H * I * sizeof(uint16_t));
    memset(wu, 0, (size_t)H * I * sizeof(uint16_t));
    memset(wd, 0, (size_t)I * H * sizeof(uint16_t));
    for (int h = 0; h < H; h++) {
        int j0 = (h * 17 + 7) % I;
        int j1 = (h * 19 + 11) % I;
        wg[h * I + j0] = f32_to_f16_bits(0.75f * scale);
        wu[h * I + j0] = f32_to_f16_bits(0.50f * scale);
        wg[h * I + j1] = f32_to_f16_bits(-0.35f * scale);
        wu[h * I + j1] = f32_to_f16_bits(0.25f * scale);
    }
    for (int j = 0; j < I; j++) {
        int c0 = (j * 13 + 3) % H;
        int c1 = (j * 7 + 5) % H;
        wd[j * H + c0] = f32_to_f16_bits(0.80f * scale);
        wd[j * H + c1] = f32_to_f16_bits(-0.45f * scale);
    }
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            x[b * H + h] = f32_to_f16_bits(val(b * H + h * 5 + 2, 23, 0.08f * scale));
        }
    }
}

static void compute_ref(float *ref,
                        const uint16_t *wg,
                        const uint16_t *wu,
                        const uint16_t *wd,
                        const uint16_t *x,
                        int H,
                        int I,
                        int B,
                        int gu_transposed,
                        int down_transposed,
                        int test_mode) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < H; c++) {
            if (test_mode == 4) {
                double g = 0.0;
                for (int h = 0; h < H; h++) {
                    float xv = f16_bits_to_f32(x[b * H + h]);
                    const int gu_idx = gu_transposed ? (c * H + h) : (h * I + c);
                    g += (double)xv * f16_bits_to_f32(wg[gu_idx]);
                }
                ref[b * H + c] = (float)g;
                continue;
            }
            double sum = 0.0;
            for (int j = 0; j < I; j++) {
                double g = 0.0;
                double u = 0.0;
                for (int h = 0; h < H; h++) {
                    float xv = f16_bits_to_f32(x[b * H + h]);
                    const int gu_idx = gu_transposed ? (j * H + h) : (h * I + j);
                    g += (double)xv * f16_bits_to_f32(wg[gu_idx]);
                    u += (double)xv * f16_bits_to_f32(wu[gu_idx]);
                }
                double hidden;
                if (test_mode == 1) {
                    hidden = g;
                } else if (test_mode == 2) {
                    hidden = g * u;
                } else if (test_mode == 3) {
                    hidden = g / (1.0 + exp(-g));
                } else {
                    if (g < -10.0) g = -10.0;
                    if (g > 10.0) g = 10.0;
                    if (u < -10.0) u = -10.0;
                    if (u > 10.0) u = 10.0;
                    hidden = (g / (1.0 + exp(-g))) * u;
                }
                const int down_idx = down_transposed ? (c * I + j) : (j * H + c);
                sum += hidden * f16_bits_to_f32(wd[down_idx]);
            }
            ref[b * H + c] = (float)sum;
        }
    }
}

static double rel_rms_for_ref(const float *ref, const uint16_t *y, int n) {
    double sq = 0.0;
    double ref_sq = 0.0;
    for (int i = 0; i < n; i++) {
        const float ane = f16_bits_to_f32(y[i]);
        const float err = ane - ref[i];
        sq += (double)err * (double)err;
        ref_sq += (double)ref[i] * (double)ref[i];
    }
    return sqrt(sq / fmax(ref_sq, 1.0e-30));
}

int main(int argc, const char **argv) {
    const int H = argc > 1 ? atoi(argv[1]) : 64;
    const int I = argc > 2 ? atoi(argv[2]) : 64;
    const int B = argc > 3 ? atoi(argv[3]) : 2;
    const char *pattern = argc > 4 ? argv[4] : "dense";
    const float scale = argc > 5 ? strtof(argv[5], NULL) : 1.0f;
    const int test_mode = argc > 6 ? atoi(argv[6]) : 0;
    if (H <= 0 || I <= 0 || B <= 0) return 2;

    uint16_t *wg = (uint16_t *)calloc((size_t)H * I, sizeof(uint16_t));
    uint16_t *wu = (uint16_t *)calloc((size_t)H * I, sizeof(uint16_t));
    uint16_t *wd = (uint16_t *)calloc((size_t)I * H, sizeof(uint16_t));
    uint16_t *x = (uint16_t *)calloc((size_t)B * H, sizeof(uint16_t));
    uint16_t *y = (uint16_t *)calloc((size_t)B * H, sizeof(uint16_t));
    float *ref = (float *)calloc((size_t)B * H, sizeof(float));
    if (!wg || !wu || !wd || !x || !y || !ref) return 3;

    if (streq(pattern, "onehot")) {
        fill_onehot(wg, wu, wd, x, H, I, B, scale);
    } else if (streq(pattern, "dense")) {
        fill_dense(wg, wu, wd, x, H, I, B, scale);
    } else {
        fprintf(stderr, "unknown pattern '%s' (use dense or onehot)\n", pattern);
        return 2;
    }

    ds4_ane_mlp_int8w_ctx *ctx = ds4_ane_mlp_fp16w_create(H, I, B);
    if (!ctx) {
        fprintf(stderr, "create failed H=%d I=%d B=%d\n", H, I, B);
        return 4;
    }
    if (!ds4_ane_mlp_fp16w_eval(ctx, wg, wu, wd, x, y)) {
        fprintf(stderr, "eval failed\n");
        ds4_ane_mlp_int8w_destroy(ctx);
        return 5;
    }

    compute_ref(ref, wg, wu, wd, x, H, I, B, 0, 0, test_mode);

    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    int worst = 0;
    for (int i = 0; i < B * H; i++) {
        float ane = f16_bits_to_f32(y[i]);
        float err = fabsf(ane - ref[i]);
        float rel = err / fmaxf(fabsf(ref[i]), 1.0e-6f);
        sq += (double)err * err;
        ref_sq += (double)ref[i] * ref[i];
        if (err > max_abs) {
            max_abs = err;
            max_rel = rel;
            worst = i;
        }
    }
    const double rms = sqrt(sq / (double)(B * H));
    const double rel_rms = sqrt(sq / fmax(ref_sq, 1.0e-30));
    const int pass = max_abs < 0.025f || rel_rms < 0.08;
    printf("pattern=%s scale=%g test_mode=%d H=%d I=%d B=%d max_abs=%g max_rel=%g rms=%g rel_rms=%g worst=%d ref=%g ane=%g %s\n",
           pattern, scale, test_mode, H, I, B, max_abs, max_rel, rms, rel_rms,
           worst, ref[worst], f16_bits_to_f32(y[worst]), pass ? "PASS" : "FAIL");
    double best_rel = rel_rms;
    const char *best_name = "normal";
    for (int gt = 0; gt <= 1; gt++) {
        for (int dt = 0; dt <= 1; dt++) {
            compute_ref(ref, wg, wu, wd, x, H, I, B, gt, dt, test_mode);
            double rr = rel_rms_for_ref(ref, y, B * H);
            printf("hypothesis gu_%s down_%s rel_rms=%g\n",
                   gt ? "transposed" : "normal",
                   dt ? "transposed" : "normal",
                   rr);
            if (rr < best_rel) {
                best_rel = rr;
                best_name = gt ? (dt ? "guT_downT" : "guT_downN") :
                                 (dt ? "guN_downT" : "normal");
            }
        }
    }
    compute_ref(ref, wg, wu, wd, x, H, I, B, 0, 0, test_mode);
    printf("best_hypothesis=%s best_rel_rms=%g\n", best_name, best_rel);
    for (int i = 0; i < B * H && i < 12; i++) {
        printf("i=%d ref=%g ane=%g\n", i, ref[i], f16_bits_to_f32(y[i]));
    }

    ds4_ane_mlp_int8w_destroy(ctx);
    free(wg);
    free(wu);
    free(wd);
    free(x);
    free(y);
    free(ref);
    return 0;
}
