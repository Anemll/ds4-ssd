// Synthetic DS4 W8A8 ANE helper call benchmark.
//
// Times the same helper path used by ds4 prefill:
//   split:        gate ANE eval + up ANE eval + CPU hidden quant + down ANE eval
//   gateup-fused: gate/up dual-output ANE eval + CPU hidden quant + down ANE eval
//   tiled-fused:  one tiled ANE eval with tile-local hidden quant + down accumulation
//
// Build:
//   cc -fobjc-arc -O2 -Wall -Wextra -o /tmp/ane_ds4_mlp_i8i8_call_bench \
//      moe-batch-bench/ane_ds4_mlp_i8i8_call_bench.m \
//      moe-batch-bench/ane_ds4_mlp_int8w.m \
//      -framework Foundation -framework IOSurface

#import "ane_ds4_mlp_int8w.h"

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mach_timebase_info_data_t g_tb;

static double ticks_to_ms(uint64_t ticks) {
    return (double)ticks * (double)g_tb.numer / (double)g_tb.denom / 1.0e6;
}

static uint32_t lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static void fill_i8(int8_t *dst, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t v = lcg(&s);
        dst[i] = (int8_t)((int)(v % 255u) - 127);
    }
}

static int parse_batches(const char *s, int *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p && n < cap) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 4096) return -1;
        out[n++] = (int)v;
        p = end;
        if (*p == ',') p++;
    }
    return n;
}

static const char *mode_name(int mode) {
    return mode == 6 ? "tiled-fused" : (mode == 5 ? "gateup-fused" : "split");
}

static float f16_to_f32(uint16_t h) {
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

typedef struct {
    pthread_t ane_thread;
    pthread_t post_thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    ds4_ane_mlp_int8w_ctx *ctx;
    const int8_t *wg;
    const int8_t *wu;
    const int8_t *wd;
    const int8_t *x_all;
    const float *route;
    int8_t *x_batch;
    uint16_t *y_batch;
    uint16_t *y_all;
    float *y_f32;
    int H;
    int B;
    int chunks;
    int ready;
    int posted;
    int ane_done;
    int fail;
    double ane_ms;
    double copy_ms;
    double post_ms;
} pipeline_job;

typedef struct {
    pthread_t ane_thread;
    pthread_t post_thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    ds4_ane_mlp_int8w_ctx *ctx[2];
    const int8_t *wg;
    const int8_t *wu;
    const int8_t *wd;
    const int8_t *x_all;
    const float *route;
    int8_t *x_batch;
    float *y_f32;
    int H;
    int B;
    int chunks;
    int slot_free[2];
    int slot_ready[2];
    int slot_chunk[2];
    int ane_done;
    int fail;
    double ane_ms;
    double post_ms;
    double wait_slot_ms;
} surface_pipeline_job;

static void post_one_chunk(const uint16_t *y,
                           const float *route,
                           float *out,
                           int H,
                           int B,
                           int chunk) {
    for (int r = 0; r < B; r++) {
        const float rw = route[(size_t)chunk * (size_t)B + (size_t)r];
        const uint16_t *src = y + ((size_t)chunk * (size_t)B + (size_t)r) * (size_t)H;
        float *dst = out + ((size_t)chunk * (size_t)B + (size_t)r) * (size_t)H;
        for (int c = 0; c < H; c++) {
            dst[c] = f16_to_f32(src[c]) * rw;
        }
    }
}

static void *pipeline_ane_main(void *arg) {
    pipeline_job *j = (pipeline_job *)arg;
    for (int ci = 0; ci < j->chunks; ci++) {
        memcpy(j->x_batch,
               j->x_all + (size_t)ci * (size_t)j->B * (size_t)j->H,
               (size_t)j->B * (size_t)j->H);
        memset(j->y_batch, 0, (size_t)j->B * (size_t)j->H * sizeof(uint16_t));
        const uint64_t t0 = mach_absolute_time();
        const bool ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(j->ctx,
                                                             j->wg,
                                                             j->wu,
                                                             j->wd,
                                                             j->x_batch,
                                                             j->y_batch);
        j->ane_ms += ticks_to_ms(mach_absolute_time() - t0);
        if (!ok) {
            pthread_mutex_lock(&j->mu);
            j->fail = 1;
            j->ane_done = 1;
            pthread_cond_broadcast(&j->cv);
            pthread_mutex_unlock(&j->mu);
            return NULL;
        }
        const uint64_t tc0 = mach_absolute_time();
        memcpy(j->y_all + (size_t)ci * (size_t)j->B * (size_t)j->H,
               j->y_batch,
               (size_t)j->B * (size_t)j->H * sizeof(uint16_t));
        j->copy_ms += ticks_to_ms(mach_absolute_time() - tc0);
        pthread_mutex_lock(&j->mu);
        j->ready = ci + 1;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    pthread_mutex_lock(&j->mu);
    j->ane_done = 1;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    return NULL;
}

static void *pipeline_post_main(void *arg) {
    pipeline_job *j = (pipeline_job *)arg;
    for (;;) {
        pthread_mutex_lock(&j->mu);
        while (j->posted >= j->ready && !j->ane_done && !j->fail) {
            pthread_cond_wait(&j->cv, &j->mu);
        }
        if (j->posted >= j->ready) {
            const int done = j->ane_done || j->fail;
            pthread_mutex_unlock(&j->mu);
            if (done) break;
            continue;
        }
        const int ci = j->posted++;
        pthread_mutex_unlock(&j->mu);
        const uint64_t t0 = mach_absolute_time();
        post_one_chunk(j->y_all, j->route, j->y_f32, j->H, j->B, ci);
        j->post_ms += ticks_to_ms(mach_absolute_time() - t0);
    }
    return NULL;
}

static void *surface_pipeline_ane_main(void *arg) {
    surface_pipeline_job *j = (surface_pipeline_job *)arg;
    for (int ci = 0; ci < j->chunks; ci++) {
        const int slot = ci & 1;
        const uint64_t tw0 = mach_absolute_time();
        pthread_mutex_lock(&j->mu);
        while (!j->slot_free[slot] && !j->fail) {
            pthread_cond_wait(&j->cv, &j->mu);
        }
        pthread_mutex_unlock(&j->mu);
        j->wait_slot_ms += ticks_to_ms(mach_absolute_time() - tw0);
        if (j->fail) break;

        memcpy(j->x_batch,
               j->x_all + (size_t)ci * (size_t)j->B * (size_t)j->H,
               (size_t)j->B * (size_t)j->H);
        const uint64_t ta0 = mach_absolute_time();
        const bool ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(j->ctx[slot],
                                                                        j->wg,
                                                                        j->wu,
                                                                        j->wd,
                                                                        j->x_batch);
        j->ane_ms += ticks_to_ms(mach_absolute_time() - ta0);
        pthread_mutex_lock(&j->mu);
        if (!ok) {
            j->fail = 1;
            j->ane_done = 1;
        } else {
            j->slot_free[slot] = 0;
            j->slot_ready[slot] = 1;
            j->slot_chunk[slot] = ci;
        }
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
        if (!ok) return NULL;
    }
    pthread_mutex_lock(&j->mu);
    j->ane_done = 1;
    pthread_cond_broadcast(&j->cv);
    pthread_mutex_unlock(&j->mu);
    return NULL;
}

static void *surface_pipeline_post_main(void *arg) {
    surface_pipeline_job *j = (surface_pipeline_job *)arg;
    for (;;) {
        pthread_mutex_lock(&j->mu);
        while (!j->slot_ready[0] && !j->slot_ready[1] && !j->ane_done && !j->fail) {
            pthread_cond_wait(&j->cv, &j->mu);
        }
        int slot = -1;
        if (j->slot_ready[0]) slot = 0;
        else if (j->slot_ready[1]) slot = 1;
        if (slot < 0) {
            const int done = j->ane_done || j->fail;
            pthread_mutex_unlock(&j->mu);
            if (done) break;
            continue;
        }
        const int ci = j->slot_chunk[slot];
        j->slot_ready[slot] = 0;
        pthread_mutex_unlock(&j->mu);

        uint64_t elems = 0;
        const uint16_t *surf = ds4_ane_mlp_int8w_lock_output_f16(j->ctx[slot], &elems);
        if (!surf || elems < (uint64_t)j->B * (uint64_t)j->H) {
            if (surf) ds4_ane_mlp_int8w_unlock_output(j->ctx[slot]);
            pthread_mutex_lock(&j->mu);
            j->fail = 1;
            pthread_cond_broadcast(&j->cv);
            pthread_mutex_unlock(&j->mu);
            break;
        }
        const uint64_t tp0 = mach_absolute_time();
        for (int r = 0; r < j->B; r++) {
            const float rw = j->route[(size_t)ci * (size_t)j->B + (size_t)r];
            float *dst = j->y_f32 + ((size_t)ci * (size_t)j->B + (size_t)r) * (size_t)j->H;
            const uint16_t *src = surf + (size_t)r * (size_t)j->H;
            for (int c = 0; c < j->H; c++) {
                dst[c] = f16_to_f32(src[c]) * rw;
            }
        }
        j->post_ms += ticks_to_ms(mach_absolute_time() - tp0);
        ds4_ane_mlp_int8w_unlock_output(j->ctx[slot]);

        pthread_mutex_lock(&j->mu);
        j->slot_free[slot] = 1;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

static int run_pipeline(int H,
                        int I,
                        int B,
                        int chunks,
                        int warmup,
                        int iters,
                        float w_scale,
                        float x_scale,
                        float mid_scale,
                        const int8_t *wg,
                        const int8_t *wu,
                        const int8_t *wd) {
    const size_t batch_elems = (size_t)B * (size_t)H;
    const size_t total_elems = (size_t)chunks * batch_elems;
    int8_t *x_all = (int8_t *)malloc(total_elems);
    int8_t *x_batch = (int8_t *)malloc(batch_elems);
    uint16_t *y_batch = (uint16_t *)malloc(batch_elems * sizeof(uint16_t));
    uint16_t *y_all = (uint16_t *)malloc(total_elems * sizeof(uint16_t));
    float *y_f32 = (float *)malloc(total_elems * sizeof(float));
    float *route = (float *)malloc((size_t)chunks * (size_t)B * sizeof(float));
    if (!x_all || !x_batch || !y_batch || !y_all || !y_f32 || !route) {
        fprintf(stderr, "pipeline alloc failed B=%d chunks=%d\n", B, chunks);
        free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
        return 1;
    }
    fill_i8(x_all, total_elems, 0x56780000u + (uint32_t)B);
    for (int i = 0; i < chunks * B; i++) route[i] = 0.5f + 0.5f * (float)((i * 13) & 255) / 255.0f;

    ds4_ane_mlp_int8w_ctx *ctx = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
    if (!ctx) {
        printf("pipeline B=%d chunks=%d status=CREATE_FAIL\n", B, chunks);
        free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
        return 2;
    }
    for (int i = 0; i < warmup; i++) {
        if (!ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wg, wu, wd, x_all, y_batch)) {
            printf("pipeline B=%d chunks=%d status=WARMUP_FAIL\n", B, chunks);
            ds4_ane_mlp_int8w_destroy(ctx);
            free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
            return 3;
        }
    }

    double serial_wall = 0.0, serial_ane = 0.0, serial_post = 0.0;
    for (int it = 0; it < iters; it++) {
        const uint64_t tw0 = mach_absolute_time();
        for (int ci = 0; ci < chunks; ci++) {
            memcpy(x_batch, x_all + (size_t)ci * batch_elems, batch_elems);
            const uint64_t ta0 = mach_absolute_time();
            bool ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wg, wu, wd, x_batch, y_batch);
            serial_ane += ticks_to_ms(mach_absolute_time() - ta0);
            if (!ok) {
                printf("pipeline B=%d chunks=%d status=SERIAL_FAIL\n", B, chunks);
                ds4_ane_mlp_int8w_destroy(ctx);
                free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
                return 4;
            }
            memcpy(y_all + (size_t)ci * batch_elems, y_batch, batch_elems * sizeof(uint16_t));
            const uint64_t tp0 = mach_absolute_time();
            post_one_chunk(y_all, route, y_f32, H, B, ci);
            serial_post += ticks_to_ms(mach_absolute_time() - tp0);
        }
        serial_wall += ticks_to_ms(mach_absolute_time() - tw0);
    }

    double pipe_wall = 0.0, pipe_ane = 0.0, pipe_copy = 0.0, pipe_post = 0.0;
    for (int it = 0; it < iters; it++) {
        pipeline_job j = {0};
        j.ctx = ctx;
        j.wg = wg; j.wu = wu; j.wd = wd;
        j.x_all = x_all;
        j.route = route;
        j.x_batch = x_batch;
        j.y_batch = y_batch;
        j.y_all = y_all;
        j.y_f32 = y_f32;
        j.H = H; j.B = B; j.chunks = chunks;
        pthread_mutex_init(&j.mu, NULL);
        pthread_cond_init(&j.cv, NULL);
        const uint64_t tw0 = mach_absolute_time();
        pthread_create(&j.post_thread, NULL, pipeline_post_main, &j);
        pthread_create(&j.ane_thread, NULL, pipeline_ane_main, &j);
        pthread_join(j.ane_thread, NULL);
        pthread_join(j.post_thread, NULL);
        pipe_wall += ticks_to_ms(mach_absolute_time() - tw0);
        pipe_ane += j.ane_ms;
        pipe_copy += j.copy_ms;
        pipe_post += j.post_ms;
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        if (j.fail) {
            printf("pipeline B=%d chunks=%d status=PIPE_FAIL\n", B, chunks);
            ds4_ane_mlp_int8w_destroy(ctx);
            free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
            return 5;
        }
    }

    const double denom = (double)iters;
    const double refs = (double)B * (double)chunks;
    printf("pipeline mode=tiled-fused B=%d chunks=%d refs=%.0f serial_wall=%.6f serial_ane=%.6f serial_post=%.6f pipe_wall=%.6f pipe_ane=%.6f pipe_copy=%.6f pipe_post=%.6f speedup=%.3f status=OK\n",
           B,
           chunks,
           refs,
           serial_wall / denom,
           serial_ane / denom,
           serial_post / denom,
           pipe_wall / denom,
           pipe_ane / denom,
           pipe_copy / denom,
           pipe_post / denom,
           serial_wall / fmax(pipe_wall, 1.0e-9));

    ds4_ane_mlp_int8w_destroy(ctx);
    free(x_all); free(x_batch); free(y_batch); free(y_all); free(y_f32); free(route);
    return 0;
}

static int run_surface_pipeline(int H,
                                int I,
                                int B,
                                int chunks,
                                int warmup,
                                int iters,
                                float w_scale,
                                float x_scale,
                                float mid_scale,
                                const int8_t *wg,
                                const int8_t *wu,
                                const int8_t *wd) {
    const size_t batch_elems = (size_t)B * (size_t)H;
    const size_t total_elems = (size_t)chunks * batch_elems;
    int8_t *x_all = (int8_t *)malloc(total_elems);
    int8_t *x_batch = (int8_t *)malloc(batch_elems);
    float *y_f32 = (float *)malloc(total_elems * sizeof(float));
    float *route = (float *)malloc((size_t)chunks * (size_t)B * sizeof(float));
    if (!x_all || !x_batch || !y_f32 || !route) {
        fprintf(stderr, "surface pipeline alloc failed B=%d chunks=%d\n", B, chunks);
        free(x_all); free(x_batch); free(y_f32); free(route);
        return 1;
    }
    fill_i8(x_all, total_elems, 0x67890000u + (uint32_t)B);
    for (int i = 0; i < chunks * B; i++) route[i] = 0.5f + 0.5f * (float)((i * 13) & 255) / 255.0f;

    ds4_ane_mlp_int8w_ctx *ctx0 = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
    ds4_ane_mlp_int8w_ctx *ctx1 = ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale);
    if (!ctx0 || !ctx1) {
        printf("surface_pipeline B=%d chunks=%d status=CREATE_FAIL\n", B, chunks);
        ds4_ane_mlp_int8w_destroy(ctx0);
        ds4_ane_mlp_int8w_destroy(ctx1);
        free(x_all); free(x_batch); free(y_f32); free(route);
        return 2;
    }
    for (int i = 0; i < warmup; i++) {
        if (!ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(ctx0, wg, wu, wd, x_all) ||
            !ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(ctx1, wg, wu, wd, x_all)) {
            printf("surface_pipeline B=%d chunks=%d status=WARMUP_FAIL\n", B, chunks);
            ds4_ane_mlp_int8w_destroy(ctx0);
            ds4_ane_mlp_int8w_destroy(ctx1);
            free(x_all); free(x_batch); free(y_f32); free(route);
            return 3;
        }
    }

    double serial_wall = 0.0, serial_ane = 0.0, serial_post = 0.0;
    for (int it = 0; it < iters; it++) {
        const uint64_t tw0 = mach_absolute_time();
        for (int ci = 0; ci < chunks; ci++) {
            memcpy(x_batch, x_all + (size_t)ci * batch_elems, batch_elems);
            const uint64_t ta0 = mach_absolute_time();
            bool ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(ctx0, wg, wu, wd, x_batch);
            serial_ane += ticks_to_ms(mach_absolute_time() - ta0);
            if (!ok) {
                printf("surface_pipeline B=%d chunks=%d status=SERIAL_FAIL\n", B, chunks);
                ds4_ane_mlp_int8w_destroy(ctx0);
                ds4_ane_mlp_int8w_destroy(ctx1);
                free(x_all); free(x_batch); free(y_f32); free(route);
                return 4;
            }
            uint64_t elems = 0;
            const uint16_t *surf = ds4_ane_mlp_int8w_lock_output_f16(ctx0, &elems);
            if (!surf || elems < (uint64_t)B * (uint64_t)H) {
                if (surf) ds4_ane_mlp_int8w_unlock_output(ctx0);
                printf("surface_pipeline B=%d chunks=%d status=LOCK_FAIL\n", B, chunks);
                ds4_ane_mlp_int8w_destroy(ctx0);
                ds4_ane_mlp_int8w_destroy(ctx1);
                free(x_all); free(x_batch); free(y_f32); free(route);
                return 5;
            }
            const uint64_t tp0 = mach_absolute_time();
            for (int r = 0; r < B; r++) {
                const float rw = route[(size_t)ci * (size_t)B + (size_t)r];
                float *dst = y_f32 + ((size_t)ci * (size_t)B + (size_t)r) * (size_t)H;
                const uint16_t *src = surf + (size_t)r * (size_t)H;
                for (int c = 0; c < H; c++) dst[c] = f16_to_f32(src[c]) * rw;
            }
            serial_post += ticks_to_ms(mach_absolute_time() - tp0);
            ds4_ane_mlp_int8w_unlock_output(ctx0);
        }
        serial_wall += ticks_to_ms(mach_absolute_time() - tw0);
    }

    double pipe_wall = 0.0, pipe_ane = 0.0, pipe_post = 0.0, pipe_wait = 0.0;
    for (int it = 0; it < iters; it++) {
        surface_pipeline_job j = {0};
        j.ctx[0] = ctx0; j.ctx[1] = ctx1;
        j.wg = wg; j.wu = wu; j.wd = wd;
        j.x_all = x_all;
        j.route = route;
        j.x_batch = x_batch;
        j.y_f32 = y_f32;
        j.H = H; j.B = B; j.chunks = chunks;
        j.slot_free[0] = 1; j.slot_free[1] = 1;
        pthread_mutex_init(&j.mu, NULL);
        pthread_cond_init(&j.cv, NULL);
        const uint64_t tw0 = mach_absolute_time();
        pthread_create(&j.post_thread, NULL, surface_pipeline_post_main, &j);
        pthread_create(&j.ane_thread, NULL, surface_pipeline_ane_main, &j);
        pthread_join(j.ane_thread, NULL);
        pthread_join(j.post_thread, NULL);
        pipe_wall += ticks_to_ms(mach_absolute_time() - tw0);
        pipe_ane += j.ane_ms;
        pipe_post += j.post_ms;
        pipe_wait += j.wait_slot_ms;
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        if (j.fail) {
            printf("surface_pipeline B=%d chunks=%d status=PIPE_FAIL\n", B, chunks);
            ds4_ane_mlp_int8w_destroy(ctx0);
            ds4_ane_mlp_int8w_destroy(ctx1);
            free(x_all); free(x_batch); free(y_f32); free(route);
            return 6;
        }
    }

    const double denom = (double)iters;
    printf("surface_pipeline mode=tiled-fused B=%d chunks=%d refs=%d serial_wall=%.6f serial_ane=%.6f serial_post=%.6f pipe_wall=%.6f pipe_ane=%.6f pipe_post=%.6f wait_slot=%.6f speedup=%.3f status=OK\n",
           B,
           chunks,
           B * chunks,
           serial_wall / denom,
           serial_ane / denom,
           serial_post / denom,
           pipe_wall / denom,
           pipe_ane / denom,
           pipe_post / denom,
           pipe_wait / denom,
           serial_wall / fmax(pipe_wall, 1.0e-9));

    ds4_ane_mlp_int8w_destroy(ctx0);
    ds4_ane_mlp_int8w_destroy(ctx1);
    free(x_all); free(x_batch); free(y_f32); free(route);
    return 0;
}

static int run_one(int H,
                   int I,
                   int B,
                   int mode,
                   int warmup,
                   int iters,
                   float w_scale,
                   float x_scale,
                   float mid_scale,
                   const int8_t *wg,
                   const int8_t *wu,
                   const int8_t *wd) {
    const size_t x_elems = (size_t)B * (size_t)H;
    const size_t y_elems = (size_t)B * (size_t)H;
    int8_t *x = (int8_t *)malloc(x_elems);
    uint16_t *y = (uint16_t *)malloc(y_elems * sizeof(uint16_t));
    if (!x || !y) {
        fprintf(stderr, "alloc failed B=%d\n", B);
        free(x);
        free(y);
        return 1;
    }
    fill_i8(x, x_elems, 0x45670000u + (uint32_t)B);
    memset(y, 0, y_elems * sizeof(uint16_t));

    const uint64_t t_create0 = mach_absolute_time();
    ds4_ane_mlp_int8w_ctx *ctx = mode == 6 ?
        ds4_ane_mlp_i8w_i8x_tiled_fused_create(H, I, B, w_scale, x_scale, mid_scale) :
        (mode == 5 ?
         ds4_ane_mlp_i8w_i8x_gateup_fused_create(H, I, B, w_scale, x_scale, mid_scale) :
         ds4_ane_mlp_i8w_i8x_create(H, I, B, w_scale, x_scale, mid_scale));
    const double create_ms = ticks_to_ms(mach_absolute_time() - t_create0);
    if (!ctx) {
        printf("mode=%s B=%d create_ms=%.3f status=CREATE_FAIL\n", mode_name(mode), B, create_ms);
        free(x);
        free(y);
        return 2;
    }

    bool ok = true;
    for (int i = 0; i < warmup; i++) {
        ok = mode == 6 ?
            ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wg, wu, wd, x, y) :
            (mode == 5 ?
             ds4_ane_mlp_i8w_i8x_gateup_fused_eval(ctx, wg, wu, wd, x, y) :
             ds4_ane_mlp_i8w_i8x_eval(ctx, wg, wu, wd, x, y));
        if (!ok) break;
    }
    if (!ok) {
        printf("mode=%s B=%d create_ms=%.3f status=WARMUP_FAIL\n", mode_name(mode), B, create_ms);
        ds4_ane_mlp_int8w_destroy(ctx);
        free(x);
        free(y);
        return 3;
    }

    const uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        ok = mode == 6 ?
            ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ctx, wg, wu, wd, x, y) :
            (mode == 5 ?
             ds4_ane_mlp_i8w_i8x_gateup_fused_eval(ctx, wg, wu, wd, x, y) :
             ds4_ane_mlp_i8w_i8x_eval(ctx, wg, wu, wd, x, y));
        if (!ok) break;
    }
    const double ms = ticks_to_ms(mach_absolute_time() - t0) / (double)iters;
    const int ane_calls = mode == 6 ? 1 : (mode == 5 ? 2 : 3);
    const double flops = 6.0 * (double)B * (double)H * (double)I;
    const double tflops = flops / (ms / 1000.0) / 1.0e12;
    printf("mode=%s B=%d create_ms=%.3f helper_ms=%.6f ane_calls=%d implied_call_ms=%.6f padded_TFLOPs=%.6f status=%s\n",
           mode_name(mode),
           B,
           create_ms,
           ms,
           ane_calls,
           ms / (double)ane_calls,
           tflops,
           ok ? "OK" : "EVAL_FAIL");

    ds4_ane_mlp_int8w_destroy(ctx);
    free(x);
    free(y);
    return ok ? 0 : 4;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        int H = 4096;
        int I = 2048;
        int warmup = 2;
        int iters = 20;
        int mode = 5;
        int pipeline = 0;
        int surface_pipeline = 0;
        int pipeline_chunks = 16;
        int batches[64] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        int nbatches = 10;
        float w_qscale = 512.0f;
        float x_qscale = 32.0f;
        float mid_qscale = 32.0f;

        for (int ai = 1; ai < argc; ai++) {
            if (!strcmp(argv[ai], "--mode") && ai + 1 < argc) {
                ai++;
                if (!strcmp(argv[ai], "split")) mode = 3;
                else if (!strcmp(argv[ai], "gateup-fused")) mode = 5;
                else if (!strcmp(argv[ai], "tiled-fused")) mode = 6;
                else {
                    fprintf(stderr, "unknown mode: %s\n", argv[ai]);
                    return 2;
                }
            } else if (!strcmp(argv[ai], "--batches") && ai + 1 < argc) {
                nbatches = parse_batches(argv[++ai], batches, (int)(sizeof(batches) / sizeof(batches[0])));
                if (nbatches <= 0) {
                    fprintf(stderr, "bad batch list\n");
                    return 2;
                }
            } else if (!strcmp(argv[ai], "--iters") && ai + 1 < argc) {
                iters = atoi(argv[++ai]);
            } else if (!strcmp(argv[ai], "--warmup") && ai + 1 < argc) {
                warmup = atoi(argv[++ai]);
            } else if (!strcmp(argv[ai], "--pipeline")) {
                pipeline = 1;
                mode = 6;
            } else if (!strcmp(argv[ai], "--surface-pipeline")) {
                surface_pipeline = 1;
                pipeline = 1;
                mode = 6;
            } else if (!strcmp(argv[ai], "--pipeline-chunks") && ai + 1 < argc) {
                pipeline_chunks = atoi(argv[++ai]);
            } else if (!strcmp(argv[ai], "--shape") && ai + 2 < argc) {
                H = atoi(argv[++ai]);
                I = atoi(argv[++ai]);
            } else if (!strcmp(argv[ai], "--scales") && ai + 2 < argc) {
                w_qscale = strtof(argv[++ai], NULL);
                x_qscale = strtof(argv[++ai], NULL);
                mid_qscale = strtof(argv[++ai], NULL);
            } else {
                fprintf(stderr,
                        "usage: %s [--mode split|gateup-fused|tiled-fused] [--pipeline|--surface-pipeline] [--pipeline-chunks N] [--shape H I] [--batches list] [--warmup N] [--iters N] [--scales wq xq midq]\n",
                        argv[0]);
                return 2;
            }
        }
        if (H <= 0 || I <= 0 || warmup < 0 || iters <= 0 || pipeline_chunks <= 0 ||
            !(w_qscale > 0.0f) || !(x_qscale > 0.0f) || !(mid_qscale > 0.0f)) {
            fprintf(stderr, "bad arguments\n");
            return 2;
        }

        const size_t gate_elems = (size_t)H * (size_t)I;
        const size_t down_elems = (size_t)I * (size_t)H;
        int8_t *wg = (int8_t *)malloc(gate_elems);
        int8_t *wu = (int8_t *)malloc(gate_elems);
        int8_t *wd = (int8_t *)malloc(down_elems);
        if (!wg || !wu || !wd) {
            fprintf(stderr, "weight alloc failed\n");
            free(wg);
            free(wu);
            free(wd);
            return 3;
        }
        fill_i8(wg, gate_elems, 0x1234u);
        fill_i8(wu, gate_elems, 0x2345u);
        fill_i8(wd, down_elems, 0x3456u);

        printf("shape H=%d I=%d mode=%s warmup=%d iters=%d wq=%.6g xq=%.6g midq=%.6g\n",
               H, I, mode_name(mode), warmup, iters, w_qscale, x_qscale, mid_qscale);
        for (int i = 0; i < nbatches; i++) {
            if (surface_pipeline) {
                run_surface_pipeline(H,
                                     I,
                                     batches[i],
                                     pipeline_chunks,
                                     warmup,
                                     iters,
                                     1.0f / w_qscale,
                                     1.0f / x_qscale,
                                     1.0f / mid_qscale,
                                     wg,
                                     wu,
                                     wd);
            } else if (pipeline) {
                run_pipeline(H,
                             I,
                             batches[i],
                             pipeline_chunks,
                             warmup,
                             iters,
                             1.0f / w_qscale,
                             1.0f / x_qscale,
                             1.0f / mid_qscale,
                             wg,
                             wu,
                             wd);
            } else {
                run_one(H,
                        I,
                        batches[i],
                        mode,
                        warmup,
                        iters,
                        1.0f / w_qscale,
                        1.0f / x_qscale,
                        1.0f / mid_qscale,
                        wg,
                        wu,
                        wd);
            }
        }

        free(wg);
        free(wu);
        free(wd);
        return 0;
    }
}
