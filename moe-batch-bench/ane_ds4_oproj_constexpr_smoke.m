// Standalone smoke for the DSv4 O-projection ANE path: linear two-stage
// constexpr conv2d-1x1 (Wa · Wb, no activation).  Probes per-call ms and
// scaling with 1 / 2 / 4 parallel ANE workers (= 1 / 2 dual-cluster +
// oversubscription).  No GPU / model file dependencies.
//
//   build: invoked from Makefile
//   usage: ./ane_ds4_oproj_constexpr_smoke [-H H] [-I I] [-B B]
//                                          [-iters N] [-threads "1,2,4"]
// Defaults match DSv4 O-proj: H=4096, I=8192, B=256.

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
    uint64_t elapsed_ticks;
    int ok;
} worker_arg;

static void *worker_main(void *arg) {
    worker_arg *w = (worker_arg *)arg;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < w->iters; i++) {
        if (!ds4_ane_mlp_fp16w_linear_constexpr_eval(w->ctx, w->X, w->Y)) {
            w->ok = 0;
            break;
        }
    }
    w->elapsed_ticks = mach_absolute_time() - t0;
    if (w->ok < 0) w->ok = 1;
    return NULL;
}

static void run_threads(int n_threads, int iters,
                        ds4_ane_mlp_int8w_ctx **ctxs,
                        const uint16_t *X, NSUInteger x_elems,
                        double *out_per_iter_ms,
                        double *out_wall_ms) {
    pthread_t *threads = (pthread_t *)calloc(n_threads, sizeof(pthread_t));
    worker_arg *args = (worker_arg *)calloc(n_threads, sizeof(worker_arg));
    uint16_t **ys = (uint16_t **)calloc(n_threads, sizeof(uint16_t *));
    for (int i = 0; i < n_threads; i++) {
        ys[i] = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        args[i].ctx = ctxs[i];
        args[i].X = X;
        args[i].Y = ys[i];
        args[i].iters = iters;
        args[i].ok = -1;
    }
    /* Warmup each ctx once so first-call setup doesn't pollute the timing. */
    for (int i = 0; i < n_threads; i++) {
        (void)ds4_ane_mlp_fp16w_linear_constexpr_eval(ctxs[i], X, ys[i]);
    }
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_threads; i++) {
        pthread_create(&threads[i], NULL, worker_main, &args[i]);
    }
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    *out_wall_ms = ticks_to_ms(mach_absolute_time() - t0);
    /* Per-iter across all workers — total work done = n_threads * iters. */
    *out_per_iter_ms = *out_wall_ms / (double)(n_threads * iters);
    for (int i = 0; i < n_threads; i++) free(ys[i]);
    free(ys); free(args); free(threads);
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    int H = 4096, I = 8192, B = 256;
    int iters = 50;
    const char *threads_str = "1,2,4";
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "-H")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I")) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B")) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threads")) threads_str = argv[++i];
    }
    setenv("DS4_FLASH_MOE_ANE_DEBUG", "1", 1);

    printf("=== O-proj constexpr conv smoke: H=%d I=%d B=%d iters/thread=%d ===\n",
           H, I, B, iters);

    @autoreleasepool {
        const uint64_t a_elems = (uint64_t)I * (uint64_t)H;
        const uint64_t b_elems = (uint64_t)H * (uint64_t)I;
        uint16_t *Wa = (uint16_t *)malloc((size_t)a_elems * sizeof(uint16_t));
        uint16_t *Wb = (uint16_t *)malloc((size_t)b_elems * sizeof(uint16_t));
        if (!Wa || !Wb) { fprintf(stderr, "alloc fail\n"); return 1; }
        for (uint64_t i = 0; i < a_elems; i++)
            Wa[i] = f32_to_f16_bits(0.01f * ((float)((i * 13u + 7u) & 0xffu) / 255.0f - 0.5f));
        for (uint64_t i = 0; i < b_elems; i++)
            Wb[i] = f32_to_f16_bits(0.01f * ((float)((i * 19u + 3u) & 0xffu) / 255.0f - 0.5f));

        int max_threads = 0;
        int thread_counts[8] = {0};
        int n_counts = 0;
        const char *p = threads_str;
        char buf[8];
        while (*p && n_counts < 8) {
            int j = 0;
            while (*p && *p != ',' && j + 1 < (int)sizeof(buf)) buf[j++] = *p++;
            buf[j] = 0;
            int v = atoi(buf);
            if (v > 0) thread_counts[n_counts++] = v;
            if (v > max_threads) max_threads = v;
            if (*p == ',') p++;
        }
        if (max_threads <= 0) { fprintf(stderr, "bad -threads %s\n", threads_str); return 2; }

        ds4_ane_mlp_int8w_ctx *ctxs[8] = {0};
        printf("Creating %d ctx(s)...\n", max_threads);
        for (int i = 0; i < max_threads; i++) {
            uint64_t t0 = mach_absolute_time();
            ctxs[i] = ds4_ane_mlp_fp16w_linear_constexpr_create(H, I, B, Wa, Wb);
            double ms = ticks_to_ms(mach_absolute_time() - t0);
            if (!ctxs[i]) { fprintf(stderr, "  ctx[%d] CREATE FAILED\n", i); return 3; }
            printf("  ctx[%d] created in %.1f ms\n", i, ms);
        }
        free(Wa); free(Wb);

        const NSUInteger x_elems = (NSUInteger)B * (NSUInteger)H;
        uint16_t *X = (uint16_t *)calloc(x_elems, sizeof(uint16_t));
        if (!X) { fprintf(stderr, "alloc X fail\n"); return 4; }
        for (NSUInteger i = 0; i < x_elems; i++)
            X[i] = f32_to_f16_bits(0.1f * sinf((float)i * 0.001f));

        /* Compute flops per eval: 2 matmuls × 2 * B * H * I */
        const double flops_per_call = 2.0 * (double)B * H * I * 2.0;

        printf("\n┌─────────┬────────────┬───────────┬───────────┬─────────────┐\n");
        printf("│ Threads │ Wall (ms)  │ ms/iter   │ TFLOP/s   │ Scaling vs 1│\n");
        printf("├─────────┼────────────┼───────────┼───────────┼─────────────┤\n");
        double base_per_iter = 0.0;
        for (int i = 0; i < n_counts; i++) {
            int n = thread_counts[i];
            double per_iter_ms = 0.0, wall_ms = 0.0;
            run_threads(n, iters, ctxs, X, x_elems, &per_iter_ms, &wall_ms);
            double tflops = (flops_per_call * (double)(n * iters)) / 1e9 / wall_ms;
            if (i == 0) base_per_iter = per_iter_ms;
            double scaling = base_per_iter / per_iter_ms;
            printf("│ %7d │ %10.2f │ %9.3f │ %9.2f │   %.2fx     │\n",
                   n, wall_ms, per_iter_ms, tflops, scaling);
        }
        printf("└─────────┴────────────┴───────────┴───────────┴─────────────┘\n");
        printf("\nNote: ms/iter = wall / (n_threads * iters), so a perfect scaling shows\n"
               "      identical ms/iter across thread counts.\n");

        for (int i = 0; i < max_threads; i++) ds4_ane_mlp_int8w_destroy(ctxs[i]);
        free(X);
    }
    return 0;
}
