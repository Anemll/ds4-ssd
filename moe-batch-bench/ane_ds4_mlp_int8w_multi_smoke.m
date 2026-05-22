// ane_ds4_mlp_int8w_multi_smoke.m
//
// N-way (up to 16) ANE concurrency smoke for the int8w/i8i8 tiled-fused
// engine used by the production prefill path.  Generalizes
// ane_ds4_mlp_int8w_dual_smoke from N=2 to N=1..16 so the M3 Ultra (and
// future M4 / M5) chips can be characterised across oversubscription levels.
//
// Goal: measure ANE compute throughput in isolation.  Weights are random
// int8 buffers held in caller memory (not baked into the ANE model), so each
// eval still exercises the int8 MatMul path that production uses for expert
// MLPs — but with zero GPU-side dequant / encode overhead.
//
// Phases for each (threads, B):
//   1. solo each ctx (sets the per-context baseline; should match across i)
//   2. concurrent N-way, no stagger
//   3. concurrent N-way, staggered (i * solo_ms / N usec phase offset)
//
// Build:
//   make moe-batch-bench/ane_ds4_mlp_int8w_multi_smoke
//
// Run:
//   ./ane_ds4_mlp_int8w_multi_smoke -threads 1,2,3,4,6,8,12,16 \
//       -batches 128,256 -warmup 5 -iters 80
//
// See ANE_WALL_MEASUREMENT.md for how to interpret the aggregate TFLOP/s
// and speedup numbers when comparing across machines (M3U / M4 / M5).

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ane_ds4_mlp_int8w.h"

#define MAX_THREADS 16

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

/* Output dtype selector — int8 cuts the output IOSurface and per-call
 * read_surface memcpy in half (B*H bytes vs B*H*2 bytes).  This is the
 * primary lever for testing whether output bandwidth bounds aggregate
 * TFLOP/s on a given chip. */
typedef enum { OUT_FP16 = 0, OUT_INT8 = 1 } out_dtype_t;

typedef struct {
    int tid;
    ds4_ane_mlp_int8w_ctx *ctx;
    const int8_t *Wg; const int8_t *Wu; const int8_t *Wd;
    const int8_t *x_i8;
    void *out_buf;  /* uint16_t* for fp16 mode, int8_t* for int8 mode */
    out_dtype_t out_dtype;
    int iters; int warmup; int stagger_us;
    mini_barrier_t *start_barrier;
    double thread_ms_per_iter;
    double thread_total_ms;
    int fail;
} thread_arg_t;

static inline int call_eval(ds4_ane_mlp_int8w_ctx *ctx,
                            const int8_t *Wg, const int8_t *Wu, const int8_t *Wd,
                            const int8_t *x_i8, void *out_buf,
                            out_dtype_t out_dtype) {
    if (out_dtype == OUT_INT8) {
        return ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_eval(
            ctx, Wg, Wu, Wd, x_i8, (int8_t *)out_buf) ? 1 : 0;
    }
    return ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
        ctx, Wg, Wu, Wd, x_i8, (uint16_t *)out_buf) ? 1 : 0;
}

static void *worker(void *p) {
    thread_arg_t *a = (thread_arg_t *)p;
    for (int i = 0; i < a->warmup; i++) {
        if (!call_eval(a->ctx, a->Wg, a->Wu, a->Wd, a->x_i8, a->out_buf, a->out_dtype)) {
            a->fail = 1; mb_wait(a->start_barrier); return NULL;
        }
    }
    mb_wait(a->start_barrier);
    if (a->stagger_us > 0) usleep((useconds_t)a->stagger_us);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < a->iters; i++) {
        if (!call_eval(a->ctx, a->Wg, a->Wu, a->Wd, a->x_i8, a->out_buf, a->out_dtype)) {
            a->fail = 1; break;
        }
    }
    a->thread_total_ms = ticksToMs(mach_absolute_time() - t0);
    a->thread_ms_per_iter = a->thread_total_ms / a->iters;
    return NULL;
}

static double solo_eval(ds4_ane_mlp_int8w_ctx *ctx,
                        const int8_t *Wg, const int8_t *Wu, const int8_t *Wd,
                        const int8_t *x_i8, void *out_buf,
                        out_dtype_t out_dtype,
                        int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        call_eval(ctx, Wg, Wu, Wd, x_i8, out_buf, out_dtype);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        call_eval(ctx, Wg, Wu, Wd, x_i8, out_buf, out_dtype);
    return ticksToMs(mach_absolute_time() - t0) / iters;
}

// xorshift64 PRNG; deterministic per-thread seed so weights are reproducible
// but distinct across contexts (mirrors per-expert randomness in production).
static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static void fill_i8_random(int8_t *buf, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed : 0xDEADBEEFCAFEBABEull;
    for (size_t i = 0; i < n; i++) {
        // map to int8 in [-64, 64] — keeps i32 accumulator from saturating
        // and matches the dynamic-range distribution production weights land in
        buf[i] = (int8_t)((int)(xs64(&s) & 0x7F) - 64);
    }
}

typedef struct {
    int H, I, B, n_threads;
    out_dtype_t out_dtype;
    int ok;
    double solo_avg_ms;
    double solo_tflops;
    double conc_wall_ms;
    double conc_per_iter_ms;
    double conc_aggregate_tflops;
    double conc_speedup;
    double conc_per_thread_min_ms;
    double conc_per_thread_max_ms;
    double stag_wall_ms;
    double stag_speedup;
    int stag_valid;
} run_result_t;

static const char *out_dtype_name(out_dtype_t d) { return d == OUT_INT8 ? "int8" : "fp16"; }

static int run_batch(int H, int I, int B, int n_threads,
                     out_dtype_t out_dtype,
                     int warmup, int iters,
                     run_result_t *out) {
    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;

    if (out) {
        memset(out, 0, sizeof(*out));
        out->H = H; out->I = I; out->B = B; out->n_threads = n_threads;
        out->out_dtype = out_dtype;
    }

    const float w_scale   = 1.0f / 512.0f;
    const float x_scale   = 1.0f / 32.0f;
    const float mid_scale = 1.0f / 32.0f;

    printf("============================================================\n");
    printf(" int8w/i8i8 tiled_fused  H=%d  I=%d  B=%d  N=%d  out=%s  warmup=%d  iters=%d\n",
        H, I, B, n_threads, out_dtype_name(out_dtype), warmup, iters);
    printf("============================================================\n");

    ds4_ane_mlp_int8w_ctx *ctxs[MAX_THREADS] = {0};
    int8_t  *Wg[MAX_THREADS] = {0}, *Wu[MAX_THREADS] = {0}, *Wd[MAX_THREADS] = {0};
    int8_t  *xb[MAX_THREADS] = {0};
    void    *ob[MAX_THREADS] = {0};

    const size_t Wg_b = (size_t)H * I;
    const size_t Wu_b = (size_t)H * I;
    const size_t Wd_b = (size_t)I * H;
    const size_t x_b  = (size_t)B * H;
    const size_t o_b  = (size_t)B * H * (out_dtype == OUT_INT8 ? 1u : 2u);

    // Create N contexts and per-context buffers.  Each context gets its own
    // backing weight allocation so cross-thread bw contention mirrors real
    // prefill (one expert per cluster).
    for (int i = 0; i < n_threads; i++) {
        uint64_t t_c0 = mach_absolute_time();
        ctxs[i] = (out_dtype == OUT_INT8)
            ? ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_create(H, I, B, w_scale, x_scale, mid_scale)
            : ds4_ane_mlp_i8w_i8x_tiled_fused_create     (H, I, B, w_scale, x_scale, mid_scale);
        double ms_create = ticksToMs(mach_absolute_time() - t_c0);
        if (!ctxs[i]) {
            fprintf(stderr, "  FAIL: create ctx[%d] (after %d ctxs)\n", i, i);
            for (int j = 0; j < i; j++) ds4_ane_mlp_int8w_destroy(ctxs[j]);
            return 1;
        }
        printf("  create ctx[%2d]: %.1f ms\n", i, ms_create);
        Wg[i] = (int8_t *)malloc(Wg_b);
        Wu[i] = (int8_t *)malloc(Wu_b);
        Wd[i] = (int8_t *)malloc(Wd_b);
        xb[i] = (int8_t *)malloc(x_b);
        ob[i] = malloc(o_b);
        if (!Wg[i] || !Wu[i] || !Wd[i] || !xb[i] || !ob[i]) {
            fprintf(stderr, "  alloc fail at i=%d\n", i);
            return 2;
        }
        // Random int8 weights + activations — runtime values passed into
        // the MatMul path each call (NOT baked into the compiled model).
        const uint64_t seed = 0xC0FFEEull * (uint64_t)(i + 1);
        fill_i8_random(Wg[i], Wg_b, seed ^ 0x1);
        fill_i8_random(Wu[i], Wu_b, seed ^ 0x2);
        fill_i8_random(Wd[i], Wd_b, seed ^ 0x3);
        fill_i8_random(xb[i], x_b,  seed ^ 0x4);
        memset(ob[i], 0, o_b);
    }

    const double gf = 3.0 * 2.0 * (double)B * H * I / 1e9;

    // --- Phase 1: solo per ctx ---
    double solo_ms[MAX_THREADS] = {0};
    double solo_sum = 0.0;
    for (int i = 0; i < n_threads; i++) {
        solo_ms[i] = solo_eval(ctxs[i], Wg[i], Wu[i], Wd[i], xb[i], ob[i], out_dtype, warmup, iters);
        solo_sum += solo_ms[i];
        printf("  [solo] ctx[%2d]  : %.3f ms/iter  |  %.2f TFLOP/s\n",
            i, solo_ms[i], gf / solo_ms[i]);
    }
    const double solo_avg = solo_sum / n_threads;
    printf("  [solo] avg      : %.3f ms/iter  |  %.2f TFLOP/s  (per-context baseline)\n",
        solo_avg, gf / solo_avg);
    if (out) {
        out->solo_avg_ms = solo_avg;
        out->solo_tflops = gf / solo_avg;
    }

    // --- Phase 2: concurrent N-way, no stagger ---
    {
        mini_barrier_t bar; mb_init(&bar, n_threads + 1);
        thread_arg_t args[MAX_THREADS];
        pthread_t threads[MAX_THREADS];
        for (int i = 0; i < n_threads; i++) {
            args[i] = (thread_arg_t){
                .tid = i, .ctx = ctxs[i],
                .Wg = Wg[i], .Wu = Wu[i], .Wd = Wd[i], .x_i8 = xb[i], .out_buf = ob[i],
                .out_dtype = out_dtype,
                .iters = iters, .warmup = warmup, .stagger_us = 0,
                .start_barrier = &bar,
                .thread_ms_per_iter = 0.0, .thread_total_ms = 0.0, .fail = 0,
            };
            pthread_create(&threads[i], NULL, worker, &args[i]);
        }
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        double wall = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        int any_fail = 0;
        for (int i = 0; i < n_threads; i++) if (args[i].fail) any_fail = 1;
        if (any_fail) { printf("  [conc ] FAIL (one or more workers errored)\n"); }
        else {
            const double aggregate_tflops = ((double)n_threads * gf * iters) / wall;
            const double aggregate_speedup = ((double)n_threads * solo_avg * iters) / wall;
            printf("  [conc ] N=%2d   : wall=%.2f ms  |  per-iter wall=%.3f ms  |  aggregate %.2f TFLOP/s  |  speedup vs N*solo = %.2fx\n",
                n_threads, wall, wall / iters, aggregate_tflops, aggregate_speedup);
            // Per-thread contention summary: max/min per-iter ms.
            double tmin = args[0].thread_ms_per_iter, tmax = args[0].thread_ms_per_iter;
            for (int i = 1; i < n_threads; i++) {
                if (args[i].thread_ms_per_iter < tmin) tmin = args[i].thread_ms_per_iter;
                if (args[i].thread_ms_per_iter > tmax) tmax = args[i].thread_ms_per_iter;
            }
            printf("  [conc ] per-thread ms/iter min=%.3f max=%.3f spread=%.1f%%  vs solo_avg=%.3f (contention=%+.1f%%)\n",
                tmin, tmax, 100.0 * (tmax - tmin) / (tmax > 0 ? tmax : 1.0),
                solo_avg, 100.0 * ((tmax + tmin) * 0.5 - solo_avg) / solo_avg);
            if (out) {
                out->ok = 1;
                out->conc_wall_ms = wall;
                out->conc_per_iter_ms = wall / iters;
                out->conc_aggregate_tflops = aggregate_tflops;
                out->conc_speedup = aggregate_speedup;
                out->conc_per_thread_min_ms = tmin;
                out->conc_per_thread_max_ms = tmax;
            }
        }
    }

    // --- Phase 3: concurrent N-way, staggered (phase = i * solo_avg / N usec) ---
    if (n_threads >= 2) {
        const int base_stagger_us = (int)(solo_avg * 1000.0 / (double)n_threads);
        mini_barrier_t bar; mb_init(&bar, n_threads + 1);
        thread_arg_t args[MAX_THREADS];
        pthread_t threads[MAX_THREADS];
        for (int i = 0; i < n_threads; i++) {
            args[i] = (thread_arg_t){
                .tid = i, .ctx = ctxs[i],
                .Wg = Wg[i], .Wu = Wu[i], .Wd = Wd[i], .x_i8 = xb[i], .out_buf = ob[i],
                .out_dtype = out_dtype,
                .iters = iters, .warmup = warmup,
                .stagger_us = i * base_stagger_us,
                .start_barrier = &bar,
                .thread_ms_per_iter = 0.0, .thread_total_ms = 0.0, .fail = 0,
            };
            pthread_create(&threads[i], NULL, worker, &args[i]);
        }
        mb_wait(&bar);
        uint64_t w0 = mach_absolute_time();
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        double wall = ticksToMs(mach_absolute_time() - w0);
        mb_destroy(&bar);

        int any_fail = 0;
        for (int i = 0; i < n_threads; i++) if (args[i].fail) any_fail = 1;
        if (any_fail) { printf("  [stag ] FAIL\n"); }
        else {
            const double aggregate_speedup = ((double)n_threads * solo_avg * iters) / wall;
            printf("  [stag ] N=%2d   : base=%d us  wall=%.2f ms  |  speedup vs N*solo = %.2fx\n",
                n_threads, base_stagger_us, wall, aggregate_speedup);
            if (out) {
                out->stag_valid = 1;
                out->stag_wall_ms = wall;
                out->stag_speedup = aggregate_speedup;
            }
        }
    }

    for (int i = 0; i < n_threads; i++) {
        free(Wg[i]); free(Wu[i]); free(Wd[i]); free(xb[i]); free(ob[i]);
        ds4_ane_mlp_int8w_destroy(ctxs[i]);
    }
    printf("\n");
    return 0;
}

static int parse_csv_ints(const char *s, int *out, int max_n, const char *what) {
    int n = 0;
    const char *p = s;
    while (*p && n < max_n) {
        char *e = NULL; long v = strtol(p, &e, 10);
        if (e == p || v <= 0) { fprintf(stderr, "bad %s: %s\n", what, s); return -1; }
        out[n++] = (int)v;
        if (*e == ',') p = e + 1;
        else if (*e == '\0') p = e;
        else { fprintf(stderr, "bad %s: %s\n", what, s); return -1; }
    }
    return n;
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    int H = 4096, I = 2048;
    int warmup = 5, iters = 50;
    int batches[16] = {128, 256};
    int nb = 2;
    int threads_list[MAX_THREADS] = {1, 2, 3, 4, 6, 8, 12, 16};
    int nt = 8;
    /* Output dtypes to sweep.  Default fp16 only — int8-output was confirmed
     * to be at the noise floor on M3U (~3.7% byte-share saving at B=256 since
     * weights dominate 24 MB/call).  Use -output int8 or -output both to run
     * the comparison again on a different chip. */
    out_dtype_t dtypes[2] = {OUT_FP16, OUT_INT8};
    int nd = 1;

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-shape") == 0 && ai + 2 < argc) {
            H = atoi(argv[++ai]); I = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-iters") == 0 && ai + 1 < argc) {
            iters = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-warmup") == 0 && ai + 1 < argc) {
            warmup = atoi(argv[++ai]);
        } else if (strcmp(argv[ai], "-batches") == 0 && ai + 1 < argc) {
            int r = parse_csv_ints(argv[++ai], batches, 16, "batches");
            if (r < 0) return 2; nb = r;
        } else if (strcmp(argv[ai], "-threads") == 0 && ai + 1 < argc) {
            int r = parse_csv_ints(argv[++ai], threads_list, MAX_THREADS, "threads");
            if (r < 0) return 2; nt = r;
            for (int k = 0; k < nt; k++) {
                if (threads_list[k] < 1 || threads_list[k] > MAX_THREADS) {
                    fprintf(stderr, "threads must be 1..%d (got %d)\n", MAX_THREADS, threads_list[k]);
                    return 2;
                }
            }
        } else if (strcmp(argv[ai], "-output") == 0 && ai + 1 < argc) {
            const char *v = argv[++ai];
            if (!strcmp(v, "fp16")) { dtypes[0] = OUT_FP16; nd = 1; }
            else if (!strcmp(v, "int8")) { dtypes[0] = OUT_INT8; nd = 1; }
            else if (!strcmp(v, "both")) { dtypes[0] = OUT_FP16; dtypes[1] = OUT_INT8; nd = 2; }
            else { fprintf(stderr, "-output must be fp16|int8|both\n"); return 2; }
        } else {
            fprintf(stderr,
                "usage: %s [-shape H I] [-batches B1,B2,...] [-threads N1,N2,...]\n"
                "          [-output fp16|int8|both] [-warmup N] [-iters N]\n"
                "  default shape: H=4096 I=2048  (DSv4 expert)\n"
                "  default batches: 128,256\n"
                "  default threads: 1,2,3,4,6,8,12,16  (max 16)\n"
                "  default output: fp16 only (int8 is at noise floor on M3U;\n"
                "                  use -output both to re-test on a different chip)\n",
                argv[0]);
            return 2;
        }
    }

    printf("=== int8w/i8i8 tiled_fused multi-thread smoke (up to N=16) ===\n");
    printf("    shape H=%d I=%d  weights+inputs: runtime int8 random (not baked)\n", H, I);
    printf("    output dtypes:");
    for (int di = 0; di < nd; di++) printf(" %s", out_dtype_name(dtypes[di]));
    printf("\n\n");

    const int max_runs = 16 * 16 * 2; // nb * nt * nd upper bound
    run_result_t *results = (run_result_t *)calloc((size_t)max_runs, sizeof(run_result_t));
    if (!results) { fprintf(stderr, "alloc results fail\n"); return 3; }
    int n_results = 0;

    for (int di = 0; di < nd; di++) {
        for (int bi = 0; bi < nb; bi++) {
            for (int ti = 0; ti < nt; ti++) {
                int rc = run_batch(H, I, batches[bi], threads_list[ti],
                                   dtypes[di], warmup, iters,
                                   &results[n_results]);
                n_results++;
                if (rc) {
                    free(results);
                    return rc;
                }
            }
        }
    }

    // --- Summary table ---
    printf("============================================================\n");
    printf(" SUMMARY  H=%d  I=%d   (random int8 weights, runtime MatMul)\n", H, I);
    printf("============================================================\n");
    printf("  out    B    N    solo_ms  solo_TF/s   conc_ms/it  conc_wall_ms   aggregate_TF/s  speedup   stagger_TF/s  contention%%\n");
    printf("  ----  ----  ---  -------  ---------  ----------  ------------   --------------  -------   ------------  -----------\n");
    for (int r = 0; r < n_results; r++) {
        const run_result_t *R = &results[r];
        if (!R->ok) {
            printf("  %4s  %4d  %3d  (failed)\n", out_dtype_name(R->out_dtype), R->B, R->n_threads);
            continue;
        }
        const double gf = 3.0 * 2.0 * (double)R->B * H * I / 1e9;
        const double stag_tflops = R->stag_valid && R->stag_wall_ms > 0.0
            ? ((double)R->n_threads * gf * (double)iters) / R->stag_wall_ms
            : 0.0;
        const double avg_pt = 0.5 * (R->conc_per_thread_min_ms + R->conc_per_thread_max_ms);
        const double contention_pct = R->solo_avg_ms > 0.0
            ? 100.0 * (avg_pt - R->solo_avg_ms) / R->solo_avg_ms
            : 0.0;
        printf("  %4s  %4d  %3d  %7.3f  %9.2f  %10.3f  %12.2f   %14.2f  %6.2fx  %12.2f  %+10.1f%%\n",
            out_dtype_name(R->out_dtype),
            R->B, R->n_threads,
            R->solo_avg_ms, R->solo_tflops,
            R->conc_per_iter_ms, R->conc_wall_ms,
            R->conc_aggregate_tflops, R->conc_speedup,
            R->stag_valid ? stag_tflops : 0.0,
            contention_pct);
    }

    // Per-(dtype, batch) peak aggregate TFLOP/s (the ANE compute ceiling).
    printf("\n  Peak aggregate TFLOP/s by output dtype and batch:\n");
    for (int di = 0; di < nd; di++) {
        for (int bi = 0; bi < nb; bi++) {
            double peak = 0.0;
            int peak_n = 0;
            for (int r = 0; r < n_results; r++) {
                if (results[r].out_dtype == dtypes[di] &&
                    results[r].B == batches[bi] && results[r].ok &&
                    results[r].conc_aggregate_tflops > peak) {
                    peak = results[r].conc_aggregate_tflops;
                    peak_n = results[r].n_threads;
                }
            }
            if (peak > 0.0) {
                printf("    out=%s  B=%4d : %.2f TFLOP/s  @ N=%d\n",
                    out_dtype_name(dtypes[di]), batches[bi], peak, peak_n);
            }
        }
    }
    // int8 vs fp16 head-to-head per batch (only if both dtypes were run).
    int has_fp16 = 0, has_int8 = 0;
    for (int di = 0; di < nd; di++) {
        if (dtypes[di] == OUT_FP16) has_fp16 = 1;
        if (dtypes[di] == OUT_INT8) has_int8 = 1;
    }
    if (has_fp16 && has_int8) {
        printf("\n  int8-output vs fp16-output peak (saturation point):\n");
        for (int bi = 0; bi < nb; bi++) {
            double peak_fp16 = 0.0, peak_int8 = 0.0;
            for (int r = 0; r < n_results; r++) {
                if (results[r].B != batches[bi] || !results[r].ok) continue;
                if (results[r].out_dtype == OUT_FP16 &&
                    results[r].conc_aggregate_tflops > peak_fp16)
                    peak_fp16 = results[r].conc_aggregate_tflops;
                if (results[r].out_dtype == OUT_INT8 &&
                    results[r].conc_aggregate_tflops > peak_int8)
                    peak_int8 = results[r].conc_aggregate_tflops;
            }
            if (peak_fp16 > 0.0 && peak_int8 > 0.0) {
                printf("    B=%4d :  fp16=%.2f TF/s   int8=%.2f TF/s   ratio=%.2fx\n",
                    batches[bi], peak_fp16, peak_int8, peak_int8 / peak_fp16);
            }
        }
    }
    printf("\n");

    free(results);
    return 0;
}
