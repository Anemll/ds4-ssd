// Standalone smoke for the int8 + per-channel-scale constexpr linear conv path
// (mode 11). Mirror of ane_ds4_oproj_constexpr_smoke.m but with int8 weights.
#import <Foundation/Foundation.h>
#include "ane_ds4_mlp_int8w.h"
#include <math.h>
#include <pthread.h>
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

typedef struct {
    ds4_ane_mlp_int8w_ctx *ctx;
    const uint16_t *X;
    uint16_t *Y;
    int iters;
    int ok;
} worker_arg;

static void *worker_main(void *arg) {
    worker_arg *w = (worker_arg *)arg;
    int eval_ok = 1;
    for (int i = 0; i < w->iters && eval_ok; i++) {
        if (!ds4_ane_mlp_int8w_linear_constexpr_eval(w->ctx, w->X, w->Y)) eval_ok = 0;
    }
    w->ok = eval_ok;
    return NULL;
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    int H = 4096, I = 8192, B = 256;
    int iters = 30;
    const char *threads_str = "1,2,4";
    for (int i = 1; i + 1 < argc; i++) {
        if      (!strcmp(argv[i], "-H")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I")) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B")) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threads")) threads_str = argv[++i];
    }
    setenv("DS4_FLASH_MOE_ANE_DEBUG", "1", 1);
    printf("=== O-proj int8 (per-channel) constexpr conv smoke: H=%d I=%d B=%d iters/thread=%d ===\n",
           H, I, B, iters);

    @autoreleasepool {
        /* Allocate + fill int8 weights and per-channel scales. */
        const uint64_t a_elems = (uint64_t)I * H;
        const uint64_t b_elems = (uint64_t)H * I;
        int8_t   *Wa_q = (int8_t   *)malloc((size_t)a_elems);
        int8_t   *Wb_q = (int8_t   *)malloc((size_t)b_elems);
        int8_t   *Wa_off = (int8_t *)calloc((size_t)I, 1);
        int8_t   *Wb_off = (int8_t *)calloc((size_t)H, 1);
        uint16_t *Wa_scale = (uint16_t *)malloc((size_t)I * sizeof(uint16_t));
        uint16_t *Wb_scale = (uint16_t *)malloc((size_t)H * sizeof(uint16_t));
        if (!Wa_q || !Wb_q || !Wa_off || !Wb_off || !Wa_scale || !Wb_scale) { fprintf(stderr, "alloc fail\n"); return 1; }
        for (uint64_t i = 0; i < a_elems; i++) Wa_q[i] = (int8_t)((i * 13u + 7u) & 0x7f) - 32;
        for (uint64_t i = 0; i < b_elems; i++) Wb_q[i] = (int8_t)((i * 19u + 3u) & 0x7f) - 32;
        for (int c = 0; c < I; c++) Wa_scale[c] = f32_to_f16_bits(0.01f);
        for (int c = 0; c < H; c++) Wb_scale[c] = f32_to_f16_bits(0.01f);

        int max_threads = 0; int thread_counts[8] = {0}; int n_counts = 0;
        const char *p = threads_str; char buf[8];
        while (*p && n_counts < 8) {
            int j = 0; while (*p && *p != ',' && j + 1 < (int)sizeof(buf)) buf[j++] = *p++;
            buf[j] = 0; int v = atoi(buf);
            if (v > 0) thread_counts[n_counts++] = v;
            if (v > max_threads) max_threads = v;
            if (*p == ',') p++;
        }
        if (max_threads <= 0) return 2;

        ds4_ane_mlp_int8w_ctx *ctxs[8] = {0};
        printf("Creating %d ctx(s)...\n", max_threads);
        for (int i = 0; i < max_threads; i++) {
            uint64_t t0 = mach_absolute_time();
            ctxs[i] = ds4_ane_mlp_int8w_linear_constexpr_create(
                H, I, B, Wa_q, Wa_off, Wa_scale, Wb_q, Wb_off, Wb_scale);
            double ms = ticks_to_ms(mach_absolute_time() - t0);
            if (!ctxs[i]) { fprintf(stderr, "  ctx[%d] CREATE FAILED at %.1f ms\n", i, ms); return 3; }
            printf("  ctx[%d] created in %.1f ms\n", i, ms);
        }
        free(Wa_q); free(Wb_q); free(Wa_off); free(Wb_off); free(Wa_scale); free(Wb_scale);

        const NSUInteger x_elems = (NSUInteger)B * H;
        uint16_t *X = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        for (NSUInteger i = 0; i < x_elems; i++) X[i] = f32_to_f16_bits(0.1f * sinf((float)i * 0.001f));
        uint16_t **Ys = (uint16_t **)calloc(max_threads, sizeof(uint16_t *));
        for (int i = 0; i < max_threads; i++) Ys[i] = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        /* Warmup each ctx. */
        for (int i = 0; i < max_threads; i++) (void)ds4_ane_mlp_int8w_linear_constexpr_eval(ctxs[i], X, Ys[i]);

        const double flops_per_call = 2.0 * (double)B * H * I * 2.0;
        printf("\n┌─────────┬────────────┬───────────┬───────────┬─────────────┐\n");
        printf("│ Threads │ Wall (ms)  │ ms/iter   │ TFLOP/s   │ Scaling vs 1│\n");
        printf("├─────────┼────────────┼───────────┼───────────┼─────────────┤\n");
        double base_per_iter = 0.0;
        for (int t = 0; t < n_counts; t++) {
            int n = thread_counts[t];
            pthread_t threads[8] = {0};
            worker_arg args[8] = {0};
            for (int i = 0; i < n; i++) {
                args[i].ctx = ctxs[i]; args[i].X = X; args[i].Y = Ys[i];
                args[i].iters = iters; args[i].ok = -1;
            }
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < n; i++) pthread_create(&threads[i], NULL, worker_main, &args[i]);
            for (int i = 0; i < n; i++) pthread_join(threads[i], NULL);
            double wall_ms = ticks_to_ms(mach_absolute_time() - t0);
            double per_iter_ms = wall_ms / (double)(n * iters);
            double tflops = (flops_per_call * (double)(n * iters)) / 1e9 / wall_ms;
            if (t == 0) base_per_iter = per_iter_ms;
            printf("│ %7d │ %10.2f │ %9.3f │ %9.2f │   %.2fx     │\n",
                   n, wall_ms, per_iter_ms, tflops, base_per_iter / per_iter_ms);
        }
        printf("└─────────┴────────────┴───────────┴───────────┴─────────────┘\n");

        for (int i = 0; i < max_threads; i++) { ds4_ane_mlp_int8w_destroy(ctxs[i]); free(Ys[i]); }
        free(Ys); free(X);
    }
    return 0;
}
