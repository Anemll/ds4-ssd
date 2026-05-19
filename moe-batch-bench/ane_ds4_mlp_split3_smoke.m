// ane_ds4_mlp_split3_smoke.m
//
// Linker-level smoke test for ane_ds4_mlp_split3.{h,m}.  Verifies the public
// C API compiles, the create/eval/destroy cycle works, and a few batches
// round-trip on the full DSv4 shape.  Mirrors the perf numbers from
// ds4_mlp_inmem_bench_packed_split3.m as a sanity check that the library
// wrapping didn't regress.
//
// Build:
//   clang -fobjc-arc -O2 \
//       ane_ds4_mlp_split3.m ane_ds4_mlp_split3_smoke.m \
//       -framework Foundation -framework IOSurface \
//       -o ane_ds4_mlp_split3_smoke

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import "ane_ds4_mlp_split3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static int run_shape(int H, int I, int B) {
    printf("--- create H=%d I=%d B=%d ---\n", H, I, B);
    uint64_t t0 = mach_absolute_time();
    ds4_ane_mlp_split3_ctx *ctx = ds4_ane_mlp_split3_create(H, I, B);
    double create_ms = ticksToMs(mach_absolute_time() - t0);
    if (!ctx) { printf("  FAIL: create returned NULL\n"); return 1; }
    printf("  create: %.1f ms\n", create_ms);

    // Allocate dummy buffers (zeros).  Just want to verify the eval cycle
    // completes; numerical correctness is covered by the bench's verify mode.
    size_t in_b   = (size_t)B * H * 2;
    size_t w_b    = (size_t)H * I * 2;
    size_t wd_b   = (size_t)I * H * 2;
    size_t out_b  = (size_t)B * H * 2;
    void *in    = calloc(1, in_b);
    void *wg    = calloc(1, w_b);
    void *wu    = calloc(1, w_b);
    void *wd    = calloc(1, wd_b);
    void *out   = calloc(1, out_b);
    if (!in || !wg || !wu || !wd || !out) { printf("  alloc fail\n"); return 2; }

    // Warmup (memcpy path)
    for (int i = 0; i < 5; i++) {
        if (!ds4_ane_mlp_split3_eval(ctx, in, wg, wu, wd, out)) {
            printf("  FAIL: warmup eval\n");
            free(in); free(wg); free(wu); free(wd); free(out);
            ds4_ane_mlp_split3_destroy(ctx);
            return 3;
        }
    }
    int iters = 50;
    double gf_total = 3.0 * 2.0 * (double)B * H * I / 1e9;

    // 1. memcpy path (_eval): caller buffers -> IOSurface every call.
    uint64_t t1 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        ds4_ane_mlp_split3_eval(ctx, in, wg, wu, wd, out);
    }
    double ms_memcpy = ticksToMs(mach_absolute_time() - t1) / iters;
    printf("  eval (memcpy path):     %.3f ms/iter  |  %.2f TFLOP/s\n",
        ms_memcpy, gf_total / ms_memcpy);

    // 2. Zero-copy path (_begin_input/_end_input + _eval_packed): IOSurface
    //    already contains the data; the loop is pure ANE eval.  Production
    //    integration should target this number if it can write the packed
    //    bytes once or in parallel with the previous eval.
    {
        void *p = ds4_ane_mlp_split3_begin_input(ctx);
        if (!p) { printf("  begin_input FAIL\n"); }
        else {
            ds4_ane_mlp_split3_end_input(ctx);  // assume contents are valid; one-time set up
            // warmup
            for (int i = 0; i < 5; i++) {
                if (!ds4_ane_mlp_split3_eval_packed(ctx, out)) {
                    printf("  FAIL: warmup eval_packed\n");
                    free(in); free(wg); free(wu); free(wd); free(out);
                    ds4_ane_mlp_split3_destroy(ctx);
                    return 4;
                }
            }
            uint64_t t2 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                ds4_ane_mlp_split3_eval_packed(ctx, out);
            }
            double ms_zerocopy = ticksToMs(mach_absolute_time() - t2) / iters;
            printf("  eval (zero-copy path):  %.3f ms/iter  |  %.2f TFLOP/s   (-%.1f ms vs memcpy)\n",
                ms_zerocopy, gf_total / ms_zerocopy, ms_memcpy - ms_zerocopy);
        }
    }

    free(in); free(wg); free(wu); free(wd); free(out);
    ds4_ane_mlp_split3_destroy(ctx);
    return 0;
}

int main(void) {
    mach_timebase_info(&g_tb);
    printf("=== ane_ds4_mlp_split3 smoke (DSv4 shape, library mode) ===\n\n");
    int batches[] = {1, 8, 16, 32, 64};
    for (int i = 0; i < (int)(sizeof(batches)/sizeof(batches[0])); i++) {
        run_shape(7168, 18432, batches[i]);
        printf("\n");
    }
    return 0;
}
