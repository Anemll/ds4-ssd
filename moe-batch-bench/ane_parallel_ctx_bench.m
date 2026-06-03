// ane_parallel_ctx_bench.m
//
// Test hypothesis: Apple's ANE framework parallelizes submissions made through
// separate _ANERequest contexts. If yes, allocating N parallel
// ds4_ane_mlp_int8w_ctx instances and submitting from N pthreads should give
// throughput close to N× a single-ctx baseline (filling the visible inter-call
// "bubbles" we see in ANE-only mode where ane_eval is ~8% of wall).
//
// Method: build N contexts at B=256 tiled-fused i8w-i8x. Each holds its own
// _ANERequest / IOSurfaces. Spin up N pthreads; each runs M iterations of
// ds4_ane_mlp_i8w_i8x_tiled_fused_eval on its own ctx. Time the wall.
// Compare against (1) serial: 1 ctx, N*M iterations on a single pthread,
// and (2) N pthreads but all using the SAME ctx (framework-level serialization).
//
// Build:
//   make moe-batch-bench/ane_parallel_ctx_bench
// Run:
//   moe-batch-bench/ane_parallel_ctx_bench --n-ctx 4 --iters 200

#import <Foundation/Foundation.h>
#import <pthread.h>
#import <stdatomic.h>
#import <stdbool.h>
#import <stdint.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <mach/mach_time.h>

#import "../ds4_ane_mlp_int8w.h"

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static const int H = 4096;
static const int I = 2048;
static const int B = 256;

static const float W_QSCALE   = 512.0f;
static const float X_QSCALE   = 32.0f;
static const float MID_QSCALE = 32.0f;

typedef struct {
    ds4_ane_mlp_int8w_ctx *ctx;
    const int8_t *Wgate;
    const int8_t *Wup;
    const int8_t *Wdown;
    const int8_t *Xin;
    uint16_t     *Yout;
    int           iters;
    int           thread_id;
    int           skip_read;       // call _eval_to_surface (no read_surface) instead of _eval
    int           skip_weight_write; // call _eval_xonly (assumes weights already in IOSurface)
    uint64_t      eval_ticks_acc;
    uint64_t      ok_count;
} worker_arg_t;

static void *worker_thread(void *arg) {
    worker_arg_t *w = (worker_arg_t *)arg;
    uint64_t acc = 0;
    uint64_t ok = 0;
    for (int it = 0; it < w->iters; it++) {
        uint64_t t0 = mach_absolute_time();
        bool r;
        if (w->skip_weight_write) {
            r = ds4_ane_mlp_i8w_i8x_tiled_fused_eval_xonly(
                w->ctx, w->Xin, w->Yout);
        } else if (w->skip_read) {
            r = ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(
                w->ctx, w->Wgate, w->Wup, w->Wdown, w->Xin);
        } else {
            r = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
                w->ctx, w->Wgate, w->Wup, w->Wdown, w->Xin, w->Yout);
        }
        uint64_t t1 = mach_absolute_time();
        acc += (t1 - t0);
        if (r) ok++;
    }
    w->eval_ticks_acc = acc;
    w->ok_count = ok;
    return NULL;
}

static int8_t *alloc_random_i8(size_t n, int seed) {
    int8_t *b = (int8_t *)malloc(n);
    if (!b) return NULL;
    unsigned s = (unsigned)seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        b[i] = (int8_t)((int)((s >> 8) & 0x1F) - 16);
    }
    return b;
}

static int parse_int_arg(int argc, char **argv, const char *key, int def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return def;
}

int main(int argc, char **argv) {
    mach_timebase_info(&g_tb);
    const int n_ctx = parse_int_arg(argc, argv, "--n-ctx", 4);
    const int iters_per = parse_int_arg(argc, argv, "--iters", 200);
    const int do_serial = parse_int_arg(argc, argv, "--serial", 1);
    const int do_concurrent_same_ctx = parse_int_arg(argc, argv, "--same-ctx", 1);
    const int do_concurrent_n_ctx = parse_int_arg(argc, argv, "--n-ctx-test", 1);
    const int warmup_iters = parse_int_arg(argc, argv, "--warmup", 5);
    const int do_iosurface_probe = parse_int_arg(argc, argv, "--iosurface-probe", 1);

    fprintf(stderr, "ANE parallel-ctx bench: H=%d I=%d B=%d n_ctx=%d iters=%d warmup=%d\n",
            H, I, B, n_ctx, iters_per, warmup_iters);

    const float w_scale = 1.0f / W_QSCALE;
    const float x_scale = 1.0f / X_QSCALE;
    const float mid_scale = 1.0f / MID_QSCALE;

    // Build N contexts; each compiles the same tiled-fused model on its own
    // _ANERequest / IOSurface set so submissions are independent.
    fprintf(stderr, "creating %d contexts...\n", n_ctx);
    ds4_ane_mlp_int8w_ctx **ctxs =
        (ds4_ane_mlp_int8w_ctx **)calloc((size_t)n_ctx, sizeof(*ctxs));
    if (!ctxs) { fprintf(stderr, "ctx array alloc failed\n"); return 1; }
    for (int i = 0; i < n_ctx; i++) {
        ctxs[i] = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
        if (!ctxs[i]) {
            fprintf(stderr, "ctx[%d] create failed\n", i);
            return 1;
        }
    }

    // Per-thread input buffers (different seeds so we don't measure pure cache).
    int8_t   **Wgate = (int8_t **)calloc((size_t)n_ctx, sizeof(*Wgate));
    int8_t   **Wup   = (int8_t **)calloc((size_t)n_ctx, sizeof(*Wup));
    int8_t   **Wdown = (int8_t **)calloc((size_t)n_ctx, sizeof(*Wdown));
    int8_t   **Xin   = (int8_t **)calloc((size_t)n_ctx, sizeof(*Xin));
    uint16_t **Yout  = (uint16_t **)calloc((size_t)n_ctx, sizeof(*Yout));
    const size_t gateu_bytes = (size_t)H * I;     // i8 gate or up: H*I
    const size_t down_bytes  = (size_t)I * H;     // i8 down: I*H
    const size_t x_bytes     = (size_t)B * H;     // i8 input: B*H
    const size_t y_elems     = (size_t)B * H;
    for (int i = 0; i < n_ctx; i++) {
        Wgate[i] = alloc_random_i8(gateu_bytes, 11 + i);
        Wup[i]   = alloc_random_i8(gateu_bytes, 31 + i);
        Wdown[i] = alloc_random_i8(down_bytes, 53 + i);
        Xin[i]   = alloc_random_i8(x_bytes, 71 + i);
        Yout[i]  = (uint16_t *)calloc(y_elems, sizeof(uint16_t));
        if (!Wgate[i] || !Wup[i] || !Wdown[i] || !Xin[i] || !Yout[i]) {
            fprintf(stderr, "buf alloc failed for ctx %d\n", i);
            return 1;
        }
    }

    // Warmup each ctx so first-call compile latency doesn't skew measurement.
    fprintf(stderr, "warmup %d iters per ctx...\n", warmup_iters);
    for (int i = 0; i < n_ctx; i++) {
        for (int w = 0; w < warmup_iters; w++) {
            (void)ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
                ctxs[i], Wgate[i], Wup[i], Wdown[i], Xin[i], Yout[i]);
        }
    }

    const double total_calls_d = (double)n_ctx * (double)iters_per;

    // ----- Variant 1: serial, single ctx, total = n_ctx * iters calls -----
    double serial_ms = 0.0, serial_ane_avg_ms = 0.0;
    if (do_serial) {
        worker_arg_t w = {0};
        w.ctx = ctxs[0];
        w.Wgate = Wgate[0]; w.Wup = Wup[0]; w.Wdown = Wdown[0];
        w.Xin = Xin[0]; w.Yout = Yout[0];
        w.iters = n_ctx * iters_per;
        uint64_t t0 = mach_absolute_time();
        worker_thread(&w);
        uint64_t t1 = mach_absolute_time();
        serial_ms = ticksToMs(t1 - t0);
        serial_ane_avg_ms = ticksToMs(w.eval_ticks_acc) / (double)w.iters;
        fprintf(stderr, "serial single-ctx: total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                serial_ms, serial_ms / w.iters,
                (unsigned long long)w.ok_count, w.iters);
    }

    // ----- Variant 2: concurrent N pthreads, SAME ctx (framework serialization probe) -----
    double same_ctx_ms = 0.0;
    if (do_concurrent_same_ctx) {
        worker_arg_t *ws = (worker_arg_t *)calloc((size_t)n_ctx, sizeof(*ws));
        pthread_t *ths = (pthread_t *)calloc((size_t)n_ctx, sizeof(*ths));
        for (int i = 0; i < n_ctx; i++) {
            ws[i].ctx = ctxs[0];   // all share ctx[0]
            ws[i].Wgate = Wgate[i]; ws[i].Wup = Wup[i]; ws[i].Wdown = Wdown[i];
            ws[i].Xin = Xin[i]; ws[i].Yout = Yout[i];
            ws[i].iters = iters_per;
            ws[i].thread_id = i;
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < n_ctx; i++) pthread_create(&ths[i], NULL, worker_thread, &ws[i]);
        for (int i = 0; i < n_ctx; i++) pthread_join(ths[i], NULL);
        uint64_t t1 = mach_absolute_time();
        same_ctx_ms = ticksToMs(t1 - t0);
        uint64_t total_ok = 0;
        for (int i = 0; i < n_ctx; i++) total_ok += ws[i].ok_count;
        fprintf(stderr, "concurrent N pthreads, SAME ctx: total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                same_ctx_ms, same_ctx_ms / total_calls_d,
                (unsigned long long)total_ok, (int)total_calls_d);
        free(ws); free(ths);
    }

    // ----- Variant 3: concurrent N pthreads, N DIFFERENT ctxs -----
    double n_ctx_ms = 0.0;
    if (do_concurrent_n_ctx) {
        worker_arg_t *ws = (worker_arg_t *)calloc((size_t)n_ctx, sizeof(*ws));
        pthread_t *ths = (pthread_t *)calloc((size_t)n_ctx, sizeof(*ths));
        for (int i = 0; i < n_ctx; i++) {
            ws[i].ctx = ctxs[i];   // each thread its own ctx
            ws[i].Wgate = Wgate[i]; ws[i].Wup = Wup[i]; ws[i].Wdown = Wdown[i];
            ws[i].Xin = Xin[i]; ws[i].Yout = Yout[i];
            ws[i].iters = iters_per;
            ws[i].thread_id = i;
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < n_ctx; i++) pthread_create(&ths[i], NULL, worker_thread, &ws[i]);
        for (int i = 0; i < n_ctx; i++) pthread_join(ths[i], NULL);
        uint64_t t1 = mach_absolute_time();
        n_ctx_ms = ticksToMs(t1 - t0);
        uint64_t total_ok = 0;
        for (int i = 0; i < n_ctx; i++) total_ok += ws[i].ok_count;
        fprintf(stderr, "concurrent N pthreads, N DIFFERENT ctxs: total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                n_ctx_ms, n_ctx_ms / total_calls_d,
                (unsigned long long)total_ok, (int)total_calls_d);
        free(ws); free(ths);
    }

    // ----- Test 3: weight upload cost via _eval_xonly (skips write_surface for gate/up/down) -----
    // Pre-write weights via a full _eval, then loop xonly which only writes X.
    // The delta vs the serial baseline = cost of writing 3 weight IOSurfaces per call.
    double xonly_serial_ms = 0.0;
    {
        // Prime: one full eval to populate weight IOSurfaces.
        (void)ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
            ctxs[0], Wgate[0], Wup[0], Wdown[0], Xin[0], Yout[0]);
        worker_arg_t w = {0};
        w.ctx = ctxs[0];
        w.Xin = Xin[0]; w.Yout = Yout[0];
        w.iters = n_ctx * iters_per;
        w.skip_weight_write = 1;
        uint64_t t0 = mach_absolute_time();
        worker_thread(&w);
        uint64_t t1 = mach_absolute_time();
        xonly_serial_ms = ticksToMs(t1 - t0);
        fprintf(stderr, "serial single-ctx (xonly, weights pre-written): total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                xonly_serial_ms, xonly_serial_ms / w.iters,
                (unsigned long long)w.ok_count, w.iters);
    }

    // ----- Test 4: parallel N-ctx + xonly (no weight rewrite per call) — floor estimate -----
    // Prime weights on every ctx first, then concurrent threads each do xonly only.
    // This combines best of Tests 1 + 3.
    double xonly_parallel_ms = 0.0;
    {
        for (int i = 0; i < n_ctx; i++) {
            (void)ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
                ctxs[i], Wgate[i], Wup[i], Wdown[i], Xin[i], Yout[i]);
        }
        worker_arg_t *ws = (worker_arg_t *)calloc((size_t)n_ctx, sizeof(*ws));
        pthread_t *ths = (pthread_t *)calloc((size_t)n_ctx, sizeof(*ths));
        for (int i = 0; i < n_ctx; i++) {
            ws[i].ctx = ctxs[i];
            ws[i].Xin = Xin[i]; ws[i].Yout = Yout[i];
            ws[i].iters = iters_per;
            ws[i].skip_weight_write = 1;
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < n_ctx; i++) pthread_create(&ths[i], NULL, worker_thread, &ws[i]);
        for (int i = 0; i < n_ctx; i++) pthread_join(ths[i], NULL);
        uint64_t t1 = mach_absolute_time();
        xonly_parallel_ms = ticksToMs(t1 - t0);
        uint64_t total_ok = 0;
        for (int i = 0; i < n_ctx; i++) total_ok += ws[i].ok_count;
        fprintf(stderr, "concurrent N-ctx xonly (no weight wr): total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                xonly_parallel_ms, xonly_parallel_ms / total_calls_d,
                (unsigned long long)total_ok, (int)total_calls_d);
        free(ws); free(ths);
    }

    // ----- Test 2: IOSurface read cost via _eval_to_surface (skips read_surface) -----
    double no_read_serial_ms = 0.0;
    if (do_iosurface_probe) {
        worker_arg_t w = {0};
        w.ctx = ctxs[0];
        w.Wgate = Wgate[0]; w.Wup = Wup[0]; w.Wdown = Wdown[0];
        w.Xin = Xin[0]; w.Yout = Yout[0];
        w.iters = n_ctx * iters_per;
        w.skip_read = 1;
        uint64_t t0 = mach_absolute_time();
        worker_thread(&w);
        uint64_t t1 = mach_absolute_time();
        no_read_serial_ms = ticksToMs(t1 - t0);
        fprintf(stderr, "serial single-ctx (no read_surface): total=%.2f ms (%.3f ms/call) ok=%llu/%d\n",
                no_read_serial_ms, no_read_serial_ms / w.iters,
                (unsigned long long)w.ok_count, w.iters);
    }

    // Summary: speedup ratios.
    fprintf(stderr, "\n=== summary (n_ctx=%d, %d total calls) ===\n",
            n_ctx, (int)total_calls_d);
    if (do_serial)
        fprintf(stderr, "  serial single-ctx           : %.2f ms  (1.00x baseline)\n", serial_ms);
    if (do_serial && do_concurrent_same_ctx)
        fprintf(stderr, "  concurrent same-ctx (%d thr) : %.2f ms  (%.2fx vs serial)\n",
                n_ctx, same_ctx_ms, serial_ms / same_ctx_ms);
    if (do_serial && do_concurrent_n_ctx)
        fprintf(stderr, "  concurrent N-ctx (%d thr)    : %.2f ms  (%.2fx vs serial)\n",
                n_ctx, n_ctx_ms, serial_ms / n_ctx_ms);
    if (do_concurrent_n_ctx && do_concurrent_same_ctx)
        fprintf(stderr, "  N-ctx vs same-ctx speedup   : %.2fx (>1 means framework parallelizes)\n",
                same_ctx_ms / n_ctx_ms);
    if (do_iosurface_probe && do_serial) {
        const double diff = serial_ms - no_read_serial_ms;
        const double per_call_diff = diff / (double)(n_ctx * iters_per);
        fprintf(stderr, "  serial no-read              : %.2f ms  (read_surface saves %.2f ms total / %.3f ms/call = %.1f%% of per-call wall)\n",
                no_read_serial_ms, diff, per_call_diff,
                100.0 * per_call_diff / (serial_ms / (n_ctx * iters_per)));
    }
    if (do_serial) {
        const double diff = serial_ms - xonly_serial_ms;
        const double per_call_diff = diff / (double)(n_ctx * iters_per);
        const double per_call_baseline = serial_ms / (n_ctx * iters_per);
        fprintf(stderr, "  serial xonly (no weight wr) : %.2f ms  (3× weight write_surface saves %.2f ms / %.3f ms/call = %.1f%% of per-call wall)\n",
                xonly_serial_ms, diff, per_call_diff,
                100.0 * per_call_diff / per_call_baseline);
        fprintf(stderr, "  →  remaining ANE-resident cost (xonly per call): %.3f ms = evaluate + X-only write_surface + read_surface\n",
                xonly_serial_ms / (n_ctx * iters_per));
    }
    if (do_serial) {
        const double per_call_xonly_par = xonly_parallel_ms / total_calls_d;
        fprintf(stderr, "  N-ctx xonly (floor)         : %.2f ms (%.3f ms/call = %.2fx vs serial baseline)\n",
                xonly_parallel_ms, per_call_xonly_par, serial_ms / xonly_parallel_ms);
    }
    fprintf(stderr, "\n");

    for (int i = 0; i < n_ctx; i++) {
        ds4_ane_mlp_int8w_destroy(ctxs[i]);
        free(Wgate[i]); free(Wup[i]); free(Wdown[i]); free(Xin[i]); free(Yout[i]);
    }
    free(ctxs); free(Wgate); free(Wup); free(Wdown); free(Xin); free(Yout);
    return 0;
}
