// ane_ds4_mlp_int8w_dual_smoke.m
//
// Dual-cluster smoke for the int8w/i8i8 ANE engine used by the ds4_metal.m
// prefill path (mode=6, i8w_i8x_tiled_fused).  Mirrors
// ane_ds4_mlp_split3_dual_smoke.m but exercises the same library + dims the
// production prefill uses (DSv4 expert: H=4096, I=2048, B=256).
//
// Goal: confirm that two int8w contexts can run concurrently on M3 Ultra and
// approach 2x aggregate throughput.  This is the precondition for wiring a
// second predict thread into ANE prefill.
//
// Build:
//   clang -fobjc-arc -O2 \
//       ane_ds4_mlp_int8w.m ane_ds4_mlp_int8w_dual_smoke.m \
//       -framework Foundation -framework IOSurface -lpthread \
//       -o ane_ds4_mlp_int8w_dual_smoke

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ane_ds4_mlp_int8w.h"

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    int needed;
    int arrived;
} mini_barrier_t;

static void mb_init(mini_barrier_t *b, int needed) {
    pthread_mutex_init(&b->mtx, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->needed = needed;
    b->arrived = 0;
}
static void mb_wait(mini_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    b->arrived++;
    if (b->arrived >= b->needed) pthread_cond_broadcast(&b->cv);
    else while (b->arrived < b->needed) pthread_cond_wait(&b->cv, &b->mtx);
    pthread_mutex_unlock(&b->mtx);
}
static void mb_destroy(mini_barrier_t *b) {
    pthread_mutex_destroy(&b->mtx); pthread_cond_destroy(&b->cv);
}

typedef struct {
    ds4_ane_mlp_int8w_ctx *ctx;
    const int8_t *Wg; const int8_t *Wu; const int8_t *Wd;
    const int8_t *x_i8;
    uint16_t *out_f16;
    int iters; int warmup; int stagger_us;
    mini_barrier_t *start_barrier;
    double thread_ms_per_iter;
    double thread_total_ms;
    int fail;
} thread_arg_t;

static void *worker(void *p) {
    thread_arg_t *a = (thread_arg_t *)p;
    for (int i = 0; i < a->warmup; i++) {
        if (!ds4_ane_mlp_i8w_i8x_tiled_fused_eval(a->ctx, a->Wg, a->Wu, a->Wd, a->x_i8, a->out_f16)) {
            a->fail = 1; mb_wait(a->start_barrier); return NULL;
        }
    }
    mb_wait(a->start_barrier);
    if (a->stagger_us > 0) usleep((useconds_t)a->stagger_us);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < a->iters; i++) {
        if (!ds4_ane_mlp_i8w_i8x_tiled_fused_eval(a->ctx, a->Wg, a->Wu, a->Wd, a->x_i8, a->out_f16)) {
            a->fail = 1; break;
        }
    }
    a->thread_total_ms = ticksToMs(mach_absolute_time() - t0);
    a->thread_ms_per_iter = a->thread_total_ms / a->iters;
    return NULL;
}

static double solo_eval(ds4_ane_mlp_int8w_ctx *ctx,
                        const int8_t *Wg, const int8_t *Wu, const int8_t *Wd,
                        const int8_t *x_i8, uint16_t *out_f16,
                        int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, Wg, Wu, Wd, x_i8, out_f16);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, Wg, Wu, Wd, x_i8, out_f16);
    return ticksToMs(mach_absolute_time() - t0) / iters;
}

static int run_batch(int H, int I, int B, int warmup, int iters) {
    const float w_scale = 1.0f / 512.0f;
    const float x_scale = 1.0f / 32.0f;
    const float mid_scale = 1.0f / 32.0f;

    printf("============================================================\n");
    printf(" int8w/i8i8 tiled_fused  H=%d  I=%d  B=%d  warmup=%d  iters=%d\n", H, I, B, warmup, iters);
    printf("============================================================\n");

    uint64_t t_c0 = mach_absolute_time();
    ds4_ane_mlp_int8w_ctx *c0 = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
    double ms_create0 = ticksToMs(mach_absolute_time() - t_c0);
    if (!c0) { printf("  FAIL: create ctx0\n"); return 1; }

    uint64_t t_c1 = mach_absolute_time();
    ds4_ane_mlp_int8w_ctx *c1 = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
    double ms_create1 = ticksToMs(mach_absolute_time() - t_c1);
    if (!c1) { printf("  FAIL: create ctx1\n"); ds4_ane_mlp_int8w_destroy(c0); return 2; }
    printf("  create: ctx0=%.1f ms  ctx1=%.1f ms\n", ms_create0, ms_create1);

    // Weight buffers per cluster (separate so each thread reads from its
    // own backing memory — mirrors the real prefill where each cluster
    // dequantises a different expert).
    size_t Wg_b = (size_t)H * I;          // [H, I] int8
    size_t Wu_b = (size_t)H * I;          // [H, I] int8
    size_t Wd_b = (size_t)I * H;          // [I, H] int8 (==Wg_b numerically)
    size_t x_b  = (size_t)B * H;          // [B, H] int8
    size_t o_b  = (size_t)B * H * 2;      // [B, H] fp16

    int8_t *Wg0 = (int8_t *)calloc(1, Wg_b); int8_t *Wg1 = (int8_t *)calloc(1, Wg_b);
    int8_t *Wu0 = (int8_t *)calloc(1, Wu_b); int8_t *Wu1 = (int8_t *)calloc(1, Wu_b);
    int8_t *Wd0 = (int8_t *)calloc(1, Wd_b); int8_t *Wd1 = (int8_t *)calloc(1, Wd_b);
    int8_t *x0  = (int8_t *)calloc(1, x_b);  int8_t *x1  = (int8_t *)calloc(1, x_b);
    uint16_t *o0 = (uint16_t *)calloc(1, o_b); uint16_t *o1 = (uint16_t *)calloc(1, o_b);
    if (!Wg0||!Wg1||!Wu0||!Wu1||!Wd0||!Wd1||!x0||!x1||!o0||!o1) { printf("  alloc fail\n"); return 3; }

    const double gf = 3.0 * 2.0 * (double)B * H * I / 1e9;

    // --- 1. solo ctx0 ---
    double s0 = solo_eval(c0, Wg0, Wu0, Wd0, x0, o0, warmup, iters);
    printf("  [1] solo ctx0           : %.3f ms/iter  |  %.2f TFLOP/s\n", s0, gf / s0);

    // --- 2. solo ctx1 ---
    double s1 = solo_eval(c1, Wg1, Wu1, Wd1, x1, o1, warmup, iters);
    printf("  [2] solo ctx1           : %.3f ms/iter  |  %.2f TFLOP/s\n", s1, gf / s1);

    double solo_avg = 0.5 * (s0 + s1);

    // --- 3. concurrent, no stagger ---
    {
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a = { c0, Wg0, Wu0, Wd0, x0, o0, iters, warmup, 0, &bar, 0, 0, 0 };
        thread_arg_t b = { c1, Wg1, Wu1, Wd1, x1, o1, iters, warmup, 0, &bar, 0, 0, 0 };
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a); pthread_create(&t1, NULL, worker, &b);
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a.fail || b.fail) { printf("  [3] FAIL concurrent\n"); }
        else {
            double aggregate_tflops = (2.0 * gf * iters) / wall;
            double aggregate_speedup = (2.0 * solo_avg * iters) / wall;  // vs 2x serial-solo
            printf("  [3] concurrent          : ctx0=%.3f ctx1=%.3f ms/iter  |  wall=%.2f ms  |  aggregate %.2f TFLOP/s  |  speedup vs 2x-solo = %.2fx\n",
                a.thread_ms_per_iter, b.thread_ms_per_iter, wall, aggregate_tflops, aggregate_speedup);
        }
    }

    // --- 4. concurrent, staggered (half-iter offset) ---
    {
        int stagger_us = (int)(solo_avg * 1000.0 * 0.5);
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a = { c0, Wg0, Wu0, Wd0, x0, o0, iters, warmup, 0,          &bar, 0, 0, 0 };
        thread_arg_t b = { c1, Wg1, Wu1, Wd1, x1, o1, iters, warmup, stagger_us, &bar, 0, 0, 0 };
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a); pthread_create(&t1, NULL, worker, &b);
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a.fail || b.fail) { printf("  [4] FAIL staggered\n"); }
        else {
            double aggregate_speedup = (2.0 * solo_avg * iters) / wall;
            printf("  [4] staggered (%d us)  : ctx0=%.3f ctx1=%.3f ms/iter  |  wall=%.2f ms  |  speedup vs 2x-solo = %.2fx\n",
                stagger_us, a.thread_ms_per_iter, b.thread_ms_per_iter, wall, aggregate_speedup);
        }
    }

    free(Wg0); free(Wg1); free(Wu0); free(Wu1); free(Wd0); free(Wd1);
    free(x0); free(x1); free(o0); free(o1);
    ds4_ane_mlp_int8w_destroy(c0);
    ds4_ane_mlp_int8w_destroy(c1);
    printf("\n");
    return 0;
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    int H = 4096, I = 2048;
    int warmup = 5, iters = 100;
    int batches[8] = {64, 128, 256};
    int nb = 3;
    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-shape") == 0 && ai + 2 < argc) {
            H = atoi(argv[++ai]); I = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-iters") == 0 && ai + 1 < argc) {
            iters = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-warmup") == 0 && ai + 1 < argc) {
            warmup = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-batches") == 0 && ai + 1 < argc) {
            const char *s = argv[++ai]; nb = 0;
            const char *p = s;
            while (*p && nb < 8) {
                char *e = NULL; long v = strtol(p, &e, 10);
                if (e == p || v <= 0) { fprintf(stderr, "bad -batches\n"); return 2; }
                batches[nb++] = (int)v;
                if (*e == ',') p = e + 1; else if (*e == '\0') p = e;
                else { fprintf(stderr, "bad -batches\n"); return 2; }
            }
        } else {
            fprintf(stderr, "usage: %s [-shape H I] [-batches B1,B2,...] [-warmup N] [-iters N]\n", argv[0]);
            return 2;
        }
    }
    printf("=== int8w/i8i8 tiled_fused dual-cluster smoke (M3 Ultra, DSv4 expert) ===\n\n");
    for (int i = 0; i < nb; i++) {
        int rc = run_batch(H, I, batches[i], warmup, iters);
        if (rc) return rc;
    }
    return 0;
}
