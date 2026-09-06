/* Model-free native Metal parity for HY4's absorbed, sink-aware attention.
 * Build with ds4_metal.o, ds4_profile.o, ds4_ane_mlp_int8w.o and Metal frameworks.
 * The largest case uses < 6 MiB of shared GPU memory; no model/sidecar is opened.
 */
#include "../ds4_gpu.h"
#include "../hy4/hy4_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static unsigned checks, cases;
static bool sg_mode;
static uint32_t random_state = UINT32_C(0x48593441);
static double worst_error;
static const float guard = -98765.25f;

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

typedef struct test_tensor {
    ds4_gpu_tensor *base, *view;
    float *data;
    size_t n;
} test_tensor;

static float random_float(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((int32_t)(random_state % 2049u) - 1024) / 1024.0f;
}

static test_tensor tensor_new(size_t n) {
    test_tensor t = {.n = n};
    t.base = ds4_gpu_tensor_alloc((n + 8u) * sizeof(float));
    CHECK(t.base);
    float *all = ds4_gpu_tensor_contents(t.base);
    CHECK(all);
    for (size_t i = 0; i < n + 8u; ++i) all[i] = guard;
    t.view = ds4_gpu_tensor_view(t.base, 4u * sizeof(float), n * sizeof(float));
    CHECK(t.view);
    t.data = ds4_gpu_tensor_contents(t.view);
    CHECK(t.data == all + 4);
    return t;
}

static void tensor_ready(test_tensor *t) {
    CHECK(ds4_gpu_tensor_did_modify(t->base, 0, (t->n + 8u) * sizeof(float)));
}

static void tensor_free(test_tensor *t) {
    const float *all = ds4_gpu_tensor_contents(t->base);
    for (size_t i = 0; i < 4u; ++i) {
        CHECK(all[i] == guard);
        CHECK(all[t->n + 4u + i] == guard);
    }
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->base);
}

static void run_case(uint32_t keys, uint32_t heads, bool zero_query) {
    const uint32_t ctx = keys + 2u;
    test_tensor qa = tensor_new((size_t)heads * 512u);
    test_tensor qr = tensor_new((size_t)heads * 256u);
    test_tensor kv = tensor_new((size_t)ctx * 512u);
    test_tensor pe = tensor_new((size_t)ctx * 64u);
    test_tensor sink = tensor_new(heads);
    test_tensor scores = tensor_new((size_t)ctx * heads);
    test_tensor out = tensor_new((size_t)heads * 512u);
    float *reference = malloc(out.n * sizeof(float));
    float *first = malloc(out.n * sizeof(float));
    float *scratch = malloc(keys * sizeof(float));
    CHECK(reference && first && scratch);

    for (size_t i = 0; i < qa.n; ++i) qa.data[i] = zero_query ? 0.0f : random_float() * 0.5f;
    for (size_t i = 0; i < qr.n; ++i) qr.data[i] = zero_query ? 0.0f : random_float() * 0.5f;
    for (size_t i = 0; i < kv.n; ++i) kv.data[i] = random_float();
    for (size_t i = 0; i < pe.n; ++i) pe.data[i] = random_float();
    const float sink_values[] = {-100.0f, 100.0f, 0.0f, -2.0f, 2.0f};
    for (uint32_t h = 0; h < heads; ++h) sink.data[h] = sink_values[h % 5u];
    tensor_ready(&qa); tensor_ready(&qr); tensor_ready(&kv); tensor_ready(&pe);
    tensor_ready(&sink); tensor_ready(&scores); tensor_ready(&out);

    CHECK(hy4_mla_attention(reference, qa.data, qr.data, 256, 192, kv.data, pe.data,
                            sink.data, heads, keys, 512, 64, 0.0625f, NULL, 0, scratch));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
             pe.view, scores.view, sink.view, keys, ctx, heads, 0.0625f));
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_synchronize());
    for (size_t i = 0; i < out.n; ++i) {
        const double error = fabs((double)out.data[i] - reference[i]);
        if (!hy4_f32_isfinite(out.data[i]) || error > 1e-5) {
            fprintf(stderr, "attention mismatch keys=%u heads=%u i=%zu cpu=%.9g gpu=%.9g error=%.3g\n",
                    keys, heads, i, reference[i], out.data[i], error);
            CHECK(false);
        }
        if (error > worst_error) worst_error = error;
        ++checks;
    }
    /* With zero logits, each key has weight 1/(keys + exp(sink)).
     * This independent analytic check detects a dropped/duplicated sink. */
    if (zero_query) for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t d = 0; d < 512u; ++d) {
            double sum = 0;
            for (uint32_t p = 0; p < keys; ++p) sum += kv.data[(size_t)p * 512u + d];
            CHECK(fabs(out.data[(size_t)h * 512u + d] - sum / (keys + exp(sink.data[h]))) < 1e-5);
        }
    }
    memcpy(first, out.data, out.n * sizeof(float));

    /* Future cache contents are allocated but must neither affect output nor
     * cause writes to scratch past the causal prefix. Also exercise owned CB. */
    for (size_t i = (size_t)keys * 512u; i < kv.n; ++i) kv.data[i] = 1e10f;
    for (size_t i = (size_t)keys * 64u; i < pe.n; ++i) pe.data[i] = -1e10f;
    tensor_ready(&kv); tensor_ready(&pe);
    CHECK(ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
             pe.view, scores.view, sink.view, keys, ctx, heads, 0.0625f));
    CHECK(ds4_gpu_synchronize());
    CHECK(memcmp(first, out.data, out.n * sizeof(float)) == 0);
    for (uint32_t h = 0; h < heads; ++h) for (uint32_t p = keys; p < ctx; ++p)
        CHECK(scores.data[(size_t)h * ctx + p] == guard);

    CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
              pe.view, scores.view, sink.view, 0, ctx, heads, 0.0625f));
    CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
              pe.view, scores.view, sink.view, ctx + 1u, ctx, heads, 0.0625f));
    CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
              pe.view, scores.view, NULL, keys, ctx, heads, 0.0625f));
    if (keys == 1u && heads == 3u) {
        ds4_gpu_tensor *tiny = ds4_gpu_tensor_view(out.base, 16, sizeof(float));
        CHECK(tiny);
        CHECK(!ds4_gpu_hy4_attention_decode_tensor(tiny, qa.view, qr.view, kv.view,
                  pe.view, scores.view, sink.view, keys, ctx, heads, 0.0625f));
        CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view, qa.view, qr.view, kv.view,
                  pe.view, scores.view, tiny, keys, ctx, heads, 0.0625f));
        ds4_gpu_tensor_free(tiny);
    }

    if(sg_mode) {
        CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view,qa.view,qr.view,kv.view,pe.view,out.view,sink.view,keys,ctx,heads,0.0625f));
        CHECK(!ds4_gpu_hy4_attention_decode_tensor(out.view,qa.view,qr.view,kv.view,pe.view,kv.view,sink.view,keys,ctx,heads,0.0625f));
    }
    if(keys==741 && heads==64) {
        const double before=ds4_gpu_busy_seconds();
        CHECK(ds4_gpu_begin_commands());
        for(unsigned repeat=0;repeat<32;repeat++) CHECK(ds4_gpu_hy4_attention_decode_tensor(out.view,qa.view,qr.view,kv.view,pe.view,scores.view,sink.view,keys,ctx,heads,0.0625f));
        CHECK(ds4_gpu_end_commands());
        printf("HY4 attention %s keys741 heads64 GPU_us=%.3f (warm32)\n",sg_mode?"F32_SG":"original",(ds4_gpu_busy_seconds()-before)*1e6/32);
    }
    free(scratch); free(first); free(reference);
    tensor_free(&out); tensor_free(&scores); tensor_free(&sink);
    tensor_free(&pe); tensor_free(&kv); tensor_free(&qr); tensor_free(&qa);
    ++cases;
}

int main(void) {
    CHECK(ds4_gpu_init());
    const uint32_t keys[] = {1,3,31,32,33,63,64,65,127,128,129,741,2048};
    const uint32_t heads[] = {1,3,8,64};
    for(unsigned mode=0;mode<2;mode++) {
        sg_mode=mode!=0;
        setenv("DS4_HY4_SG_ATTENTION",sg_mode?"1":"0",1);
        for(unsigned k=0;k<sizeof(keys)/sizeof(keys[0]);k++) for(unsigned h=0;h<4;h++) run_case(keys[k],heads[h],false);
        run_case(3,3,true);run_case(129,8,true);
    }
    sg_mode=false;setenv("DS4_HY4_SG_ATTENTION","0",1);run_case(741,64,false);
    ds4_gpu_cleanup();
    printf("PASS HY4 Metal sink attention: %u cases, %u checks, maxabs %.3g; causal prefix, offset views, bounds, CPU and analytic oracles\n",
           cases, checks, worst_error);
    return 0;
}
