// Standalone smoke: drive ds4_ane_mlp_fp16w_constexpr_create + _eval with
// synthetic weights at the shared-expert shape and report compile/load/eval
// status.  Used to confirm the constexpr conv path works without needing the
// full ds4 binary or model file.

#import <Foundation/Foundation.h>
#include "ane_ds4_mlp_int8w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ticks_to_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) { m = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    int H = 4096, I = 2048, B = 256;
    int iters = 20;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "-H")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I")) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B")) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-iters")) iters = atoi(argv[++i]);
    }
    setenv("DS4_FLASH_MOE_ANE_DEBUG", "1", 1);  /* full NSError descriptions */

    printf("=== constexpr conv smoke: H=%d I=%d B=%d ===\n", H, I, B);

    @autoreleasepool {
        const uint64_t gate_elems = (uint64_t)I * (uint64_t)H;
        const uint64_t down_elems = (uint64_t)H * (uint64_t)I;
        uint16_t *Wg = (uint16_t *)malloc((size_t)gate_elems * sizeof(uint16_t));
        uint16_t *Wu = (uint16_t *)malloc((size_t)gate_elems * sizeof(uint16_t));
        uint16_t *Wd = (uint16_t *)malloc((size_t)down_elems * sizeof(uint16_t));
        if (!Wg || !Wu || !Wd) { fprintf(stderr, "alloc fail\n"); return 1; }
        /* Tiny fixed weights — actual values don't matter for compile/eval probe. */
        for (uint64_t i = 0; i < gate_elems; i++) {
            Wg[i] = f32_to_f16_bits(0.01f * ((float)((i * 13u + 7u) & 0xffu) / 255.0f - 0.5f));
            Wu[i] = f32_to_f16_bits(0.01f * ((float)((i * 19u + 3u) & 0xffu) / 255.0f - 0.5f));
        }
        for (uint64_t i = 0; i < down_elems; i++) {
            Wd[i] = f32_to_f16_bits(0.01f * ((float)((i * 7u + 11u) & 0xffu) / 255.0f - 0.5f));
        }

        uint64_t t0 = mach_absolute_time();
        ds4_ane_mlp_int8w_ctx *ctx = ds4_ane_mlp_fp16w_constexpr_create(H, I, B, Wg, Wu, Wd);
        double create_ms = ticks_to_ms(mach_absolute_time() - t0);
        free(Wg); free(Wu); free(Wd);
        if (!ctx) {
            fprintf(stderr, "CONSTEXPR CREATE FAILED (create_ms=%.1f)\n", create_ms);
            return 2;
        }
        printf("create OK: %.1f ms\n", create_ms);

        const NSUInteger x_elems = (NSUInteger)B * (NSUInteger)H;
        uint16_t *X = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        uint16_t *Y = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        if (!X || !Y) { fprintf(stderr, "alloc xy fail\n"); return 3; }
        for (NSUInteger i = 0; i < x_elems; i++) {
            X[i] = f32_to_f16_bits(0.1f * sinf((float)i * 0.001f));
        }

        /* Warmup */
        if (!ds4_ane_mlp_fp16w_constexpr_eval(ctx, X, Y)) {
            fprintf(stderr, "eval FAILED on warmup\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            free(X); free(Y);
            return 4;
        }
        printf("warmup eval OK\n");

        uint64_t et0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (!ds4_ane_mlp_fp16w_constexpr_eval(ctx, X, Y)) {
                fprintf(stderr, "eval FAILED on iter %d\n", i);
                ds4_ane_mlp_int8w_destroy(ctx);
                free(X); free(Y);
                return 5;
            }
        }
        double total_ms = ticks_to_ms(mach_absolute_time() - et0);
        const double per_ms = total_ms / iters;
        const double flops_per_call = 2.0 * (double)B * (double)H * (double)I * 2.0 +  /* gate+up */
                                       2.0 * (double)B * (double)I * (double)H;        /* down */
        printf("eval %d iters: %.2f ms total, %.3f ms/iter, %.2f TFLOP/s\n",
               iters, total_ms, per_ms, flops_per_call / 1e9 / per_ms);

        ds4_ane_mlp_int8w_destroy(ctx);
        free(X); free(Y);
    }
    return 0;
}
