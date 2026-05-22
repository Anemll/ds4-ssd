// ane_ds4_mlp_split3_dual_smoke.m
//
// Dual-cluster smoke for the M3 Ultra (two ANEx16 clusters).  Creates two
// independent ds4_ane_mlp_split3 contexts and exercises them on two pthreads
// concurrently to verify the OS dispatches the second predict to the second
// cluster.  Measures per-thread ms/iter and wall-clock vs sequential.
//
// Phases for each batch:
//   1. solo ctx0                          (baseline cluster-0 throughput)
//   2. solo ctx1                          (cluster-1 should match cluster-0)
//   3. concurrent ctx0 || ctx1 (no stagger, zero-copy path)
//   4. concurrent ctx0 || ctx1 staggered  (thread1 starts ~iter_ms/2 late)
//   5. concurrent memcpy path             (full caller-buffer staging each
//                                          iter; this is the bw-bound case)
//
// Best case (zero-copy, two clusters truly in parallel) should approach 2x
// vs solo.  If concurrent ms/iter ~ solo ms/iter and wall ~ solo wall, the
// two clusters are running in parallel.  If concurrent ms/iter ~ 2x solo,
// they are serialised (one cluster only, or the predict API is locked).
//
// Build:
//   clang -fobjc-arc -O2 \
//       ane_ds4_mlp_split3.m ane_ds4_mlp_split3_dual_smoke.m \
//       -framework Foundation -framework IOSurface -lpthread \
//       -o ane_ds4_mlp_split3_dual_smoke

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ane_ds4_mlp_split3.h"

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// macOS pthreads lacks pthread_barrier_t; roll a tiny barrier with mutex+cond.
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
    if (b->arrived >= b->needed) {
        pthread_cond_broadcast(&b->cv);
    } else {
        while (b->arrived < b->needed) pthread_cond_wait(&b->cv, &b->mtx);
    }
    pthread_mutex_unlock(&b->mtx);
}
static void mb_destroy(mini_barrier_t *b) {
    pthread_mutex_destroy(&b->mtx);
    pthread_cond_destroy(&b->cv);
}

typedef struct {
    ds4_ane_mlp_split3_ctx *ctx;
    int iters;
    int warmup;
    int stagger_us;   // delay before first iter (0 = none)
    int memcpy_path;  // 0 = eval_packed (zero-copy), 1 = eval (memcpy each call)

    // memcpy-path scratch buffers (NULL for zero-copy)
    void *in_buf;
    void *wg_buf;
    void *wu_buf;
    void *wd_buf;

    void *out_buf;

    // sync
    mini_barrier_t *start_barrier;

    // results
    double thread_ms_per_iter;
    double thread_total_ms;
    uint64_t thread_start_ticks;
    uint64_t thread_end_ticks;
    int fail;
    char tag[16];
} thread_arg_t;

static void *worker(void *p) {
    thread_arg_t *a = (thread_arg_t *)p;

    // warmup (under the barrier so each cluster pays its own jit/load cost first)
    for (int i = 0; i < a->warmup; i++) {
        bool ok;
        if (a->memcpy_path) {
            ok = ds4_ane_mlp_split3_eval(a->ctx, a->in_buf, a->wg_buf, a->wu_buf, a->wd_buf, a->out_buf);
        } else {
            ok = ds4_ane_mlp_split3_eval_packed(a->ctx, a->out_buf);
        }
        if (!ok) { a->fail = 1; mb_wait(a->start_barrier); return NULL; }
    }

    // line up at the gate
    mb_wait(a->start_barrier);

    // optional stagger so the two threads phase-offset within an iteration
    if (a->stagger_us > 0) usleep((useconds_t)a->stagger_us);

    a->thread_start_ticks = mach_absolute_time();
    for (int i = 0; i < a->iters; i++) {
        bool ok;
        if (a->memcpy_path) {
            ok = ds4_ane_mlp_split3_eval(a->ctx, a->in_buf, a->wg_buf, a->wu_buf, a->wd_buf, a->out_buf);
        } else {
            ok = ds4_ane_mlp_split3_eval_packed(a->ctx, a->out_buf);
        }
        if (!ok) { a->fail = 1; break; }
    }
    a->thread_end_ticks = mach_absolute_time();
    a->thread_total_ms = ticksToMs(a->thread_end_ticks - a->thread_start_ticks);
    a->thread_ms_per_iter = a->thread_total_ms / a->iters;
    return NULL;
}

static double solo_eval_packed(ds4_ane_mlp_split3_ctx *ctx, void *out, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) ds4_ane_mlp_split3_eval_packed(ctx, out);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ds4_ane_mlp_split3_eval_packed(ctx, out);
    return ticksToMs(mach_absolute_time() - t0) / iters;
}

static double solo_eval_memcpy(ds4_ane_mlp_split3_ctx *ctx,
                               void *in, void *wg, void *wu, void *wd, void *out,
                               int warmup, int iters) {
    for (int i = 0; i < warmup; i++) ds4_ane_mlp_split3_eval(ctx, in, wg, wu, wd, out);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ds4_ane_mlp_split3_eval(ctx, in, wg, wu, wd, out);
    return ticksToMs(mach_absolute_time() - t0) / iters;
}

static int run_batch(int H, int I, int B, int warmup, int iters) {
    printf("============================================================\n");
    printf(" H=%d  I=%d  B=%d  warmup=%d  iters=%d\n", H, I, B, warmup, iters);
    printf("============================================================\n");

    uint64_t t_c0 = mach_absolute_time();
    ds4_ane_mlp_split3_ctx *c0 = ds4_ane_mlp_split3_create(H, I, B);
    double ms_create0 = ticksToMs(mach_absolute_time() - t_c0);
    if (!c0) { printf("  FAIL: create ctx0\n"); return 1; }

    uint64_t t_c1 = mach_absolute_time();
    ds4_ane_mlp_split3_ctx *c1 = ds4_ane_mlp_split3_create(H, I, B);
    double ms_create1 = ticksToMs(mach_absolute_time() - t_c1);
    if (!c1) { printf("  FAIL: create ctx1\n"); ds4_ane_mlp_split3_destroy(c0); return 2; }
    printf("  create: ctx0=%.1f ms  ctx1=%.1f ms\n", ms_create0, ms_create1);

    // dummy buffers for memcpy-path runs.  Two pairs so each thread has its
    // own caller-side source bytes (closer to real prefill where each cluster
    // feeds its own weight slice).
    size_t in_b  = (size_t)B * H * 2;
    size_t w_b   = (size_t)H * I * 2;
    size_t wd_b  = (size_t)I * H * 2;
    size_t out_b = (size_t)B * H * 2;
    void *in0 = calloc(1, in_b);   void *in1 = calloc(1, in_b);
    void *wg0 = calloc(1, w_b);    void *wg1 = calloc(1, w_b);
    void *wu0 = calloc(1, w_b);    void *wu1 = calloc(1, w_b);
    void *wd0 = calloc(1, wd_b);   void *wd1 = calloc(1, wd_b);
    void *out0 = calloc(1, out_b); void *out1 = calloc(1, out_b);
    if (!in0||!in1||!wg0||!wg1||!wu0||!wu1||!wd0||!wd1||!out0||!out1) {
        printf("  alloc fail\n"); return 3;
    }

    // seed the zero-copy IOSurfaces once (contents irrelevant; we only time).
    {
        void *p0 = ds4_ane_mlp_split3_begin_input(c0);
        if (p0) ds4_ane_mlp_split3_end_input(c0);
        void *p1 = ds4_ane_mlp_split3_begin_input(c1);
        if (p1) ds4_ane_mlp_split3_end_input(c1);
    }

    const double gf = 3.0 * 2.0 * (double)B * H * I / 1e9;

    // --- Phase 1: solo ctx0 (zero-copy) ---
    double solo0_ms = solo_eval_packed(c0, out0, warmup, iters);
    printf("  [1] solo ctx0  zero-copy : %.3f ms/iter  |  %.2f TFLOP/s\n",
        solo0_ms, gf / solo0_ms);

    // --- Phase 2: solo ctx1 (zero-copy) ---
    double solo1_ms = solo_eval_packed(c1, out1, warmup, iters);
    printf("  [2] solo ctx1  zero-copy : %.3f ms/iter  |  %.2f TFLOP/s\n",
        solo1_ms, gf / solo1_ms);

    double solo_ms_avg = 0.5 * (solo0_ms + solo1_ms);

    // --- Phase 3: concurrent (no stagger, zero-copy) ---
    {
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a0 = { .ctx=c0, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=0, .out_buf=out0, .start_barrier=&bar };
        thread_arg_t a1 = { .ctx=c1, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=0, .out_buf=out1, .start_barrier=&bar };
        strcpy(a0.tag, "c0"); strcpy(a1.tag, "c1");
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a0);
        pthread_create(&t1, NULL, worker, &a1);
        // wait for both threads to finish warmup
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall_ms = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a0.fail || a1.fail) { printf("  [3] FAIL during concurrent zero-copy\n"); }
        else {
            double aggregate_tflops = (2.0 * gf * iters) / wall_ms;
            double speedup = (solo_ms_avg * iters) / wall_ms;  // vs running them back-to-back
            printf("  [3] concurrent     zero-copy : ctx0=%.3f ms/iter  ctx1=%.3f ms/iter  |  wall=%.2f ms  |  aggregate %.2f TFLOP/s  |  speedup vs seq = %.2fx\n",
                a0.thread_ms_per_iter, a1.thread_ms_per_iter, wall_ms, aggregate_tflops, speedup);
        }
    }

    // --- Phase 4: concurrent with stagger (zero-copy) ---
    {
        int stagger_us = (int)(solo_ms_avg * 1000.0 * 0.5);  // half an iter
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a0 = { .ctx=c0, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=0, .out_buf=out0, .start_barrier=&bar };
        thread_arg_t a1 = { .ctx=c1, .iters=iters, .warmup=warmup, .stagger_us=stagger_us,
                            .memcpy_path=0, .out_buf=out1, .start_barrier=&bar };
        strcpy(a0.tag, "c0"); strcpy(a1.tag, "c1");
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a0);
        pthread_create(&t1, NULL, worker, &a1);
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall_ms = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a0.fail || a1.fail) { printf("  [4] FAIL during staggered zero-copy\n"); }
        else {
            double speedup = (solo_ms_avg * iters) / wall_ms;
            printf("  [4] staggered (%d us) zc   : ctx0=%.3f ms/iter  ctx1=%.3f ms/iter  |  wall=%.2f ms  |  speedup vs seq = %.2fx\n",
                stagger_us, a0.thread_ms_per_iter, a1.thread_ms_per_iter, wall_ms, speedup);
        }
    }

    // --- Phase 5a: solo ctx0 memcpy path (for reference) ---
    double solo0_mc_ms = solo_eval_memcpy(c0, in0, wg0, wu0, wd0, out0, warmup, iters);
    printf("  [5a] solo ctx0 memcpy    : %.3f ms/iter  |  %.2f TFLOP/s\n",
        solo0_mc_ms, gf / solo0_mc_ms);

    // --- Phase 5b: concurrent memcpy (no stagger) ---
    {
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a0 = { .ctx=c0, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=1, .in_buf=in0, .wg_buf=wg0, .wu_buf=wu0, .wd_buf=wd0,
                            .out_buf=out0, .start_barrier=&bar };
        thread_arg_t a1 = { .ctx=c1, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=1, .in_buf=in1, .wg_buf=wg1, .wu_buf=wu1, .wd_buf=wd1,
                            .out_buf=out1, .start_barrier=&bar };
        strcpy(a0.tag, "c0"); strcpy(a1.tag, "c1");
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a0);
        pthread_create(&t1, NULL, worker, &a1);
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall_ms = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a0.fail || a1.fail) { printf("  [5b] FAIL during concurrent memcpy\n"); }
        else {
            double speedup = (solo0_mc_ms * iters) / (wall_ms * 0.5);  // vs 2x solo memcpy serial
            double aggregate_tflops = (2.0 * gf * iters) / wall_ms;
            printf("  [5b] concurrent memcpy   : ctx0=%.3f ms/iter  ctx1=%.3f ms/iter  |  wall=%.2f ms  |  aggregate %.2f TFLOP/s  |  speedup vs 2x-solo = %.2fx\n",
                a0.thread_ms_per_iter, a1.thread_ms_per_iter, wall_ms, aggregate_tflops, speedup);
        }
    }

    // --- Phase 5c: concurrent memcpy with stagger ---
    {
        int stagger_us = (int)(solo0_mc_ms * 1000.0 * 0.5);
        mini_barrier_t bar; mb_init(&bar, 3);
        thread_arg_t a0 = { .ctx=c0, .iters=iters, .warmup=warmup, .stagger_us=0,
                            .memcpy_path=1, .in_buf=in0, .wg_buf=wg0, .wu_buf=wu0, .wd_buf=wd0,
                            .out_buf=out0, .start_barrier=&bar };
        thread_arg_t a1 = { .ctx=c1, .iters=iters, .warmup=warmup, .stagger_us=stagger_us,
                            .memcpy_path=1, .in_buf=in1, .wg_buf=wg1, .wu_buf=wu1, .wd_buf=wd1,
                            .out_buf=out1, .start_barrier=&bar };
        strcpy(a0.tag, "c0"); strcpy(a1.tag, "c1");
        pthread_t t0, t1;
        pthread_create(&t0, NULL, worker, &a0);
        pthread_create(&t1, NULL, worker, &a1);
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        pthread_join(t0, NULL); pthread_join(t1, NULL);
        double wall_ms = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        if (a0.fail || a1.fail) { printf("  [5c] FAIL during staggered memcpy\n"); }
        else {
            double speedup = (solo0_mc_ms * iters) / (wall_ms * 0.5);
            printf("  [5c] staggered (%d us) mc: ctx0=%.3f ms/iter  ctx1=%.3f ms/iter  |  wall=%.2f ms  |  speedup vs 2x-solo = %.2fx\n",
                stagger_us, a0.thread_ms_per_iter, a1.thread_ms_per_iter, wall_ms, speedup);
        }
    }

    free(in0); free(in1); free(wg0); free(wg1);
    free(wu0); free(wu1); free(wd0); free(wd1);
    free(out0); free(out1);
    ds4_ane_mlp_split3_destroy(c0);
    ds4_ane_mlp_split3_destroy(c1);
    printf("\n");
    return 0;
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);

    int H = 7168, I = 18432;
    int warmup = 5, iters = 50;
    int batches[8] = {32, 64, 128};
    int nb = 3;

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-shape") == 0 && ai + 2 < argc) {
            H = atoi(argv[++ai]); I = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-iters") == 0 && ai + 1 < argc) {
            iters = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-warmup") == 0 && ai + 1 < argc) {
            warmup = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-batches") == 0 && ai + 1 < argc) {
            const char *s = argv[++ai];
            nb = 0;
            const char *p = s;
            while (*p && nb < 8) {
                char *end = NULL; long v = strtol(p, &end, 10);
                if (end == p || v <= 0) { fprintf(stderr, "bad -batches\n"); return 2; }
                batches[nb++] = (int)v;
                if (*end == ',') p = end + 1;
                else if (*end == '\0') p = end;
                else { fprintf(stderr, "bad -batches\n"); return 2; }
            }
        } else {
            fprintf(stderr, "usage: %s [-shape H I] [-batches B1,B2,...] [-warmup N] [-iters N]\n",
                argv[0]);
            return 2;
        }
    }

    printf("=== ane_ds4_mlp_split3 dual-cluster smoke (M3 Ultra) ===\n\n");
    for (int i = 0; i < nb; i++) {
        int rc = run_batch(H, I, batches[i], warmup, iters);
        if (rc) return rc;
    }
    return 0;
}
