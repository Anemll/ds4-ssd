/* Standalone HY4 reference-math regressions; no model or Metal allocation.
 * Build examples:
 * cc -O3 -ffast-math -std=c99 -Wall -Wextra \
 *    tests/test_hy4_math.c -lm -o tests/test_hy4_math
 * cc -O1 -g -std=c99 -fsanitize=address,undefined \
 *    tests/test_hy4_math.c -lm -o tests/test_hy4_math_sanitize
 */
#include "../hy4/hy4_math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
static uint32_t rng_state = UINT32_C(0x48593431);

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static uint32_t random_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float random_float(void) {
    return ((int32_t)(random_u32() % 2049u) - 1024) / 1024.0f;
}

static uint32_t bits(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}

static float from_bits(uint32_t b) {
    float x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static void near_value(float actual, double expected, double tolerance) {
    if (!(fabs((double)actual - expected) <= tolerance)) {
        fprintf(stderr, "numeric mismatch: actual %.9g expected %.17g tolerance %.3g\n",
                (double)actual, expected, tolerance);
        CHECK(false);
    }
    ++checks;
}

static double sigmoid_reference(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Independent double-precision equation oracle. This validates HC layout,
 * scale/base partitions, epsilon placement, and the head's distinct shape. */
static void hc_reference(double *reduced, double *post,
                         const float *streams, const float *fn,
                         const float *scale, const float *base,
                         unsigned emb, unsigned hc, bool head) {
    const unsigned flat = emb * hc;
    double square_sum = 0.0;
    for (unsigned i = 0; i < flat; ++i) square_sum += (double)streams[i] * streams[i];
    const double inv = 1.0 / sqrt(square_sum / flat + 1e-5);
    double pre[HY4_MATH_MAX_HC];
    for (unsigned h = 0; h < hc; ++h) {
        double mix_pre = 0.0, mix_post = 0.0;
        for (unsigned i = 0; i < flat; ++i) {
            mix_pre += fn[(size_t)h * flat + i] * ((double)streams[i] * inv);
            if (!head) mix_post += fn[(size_t)(hc + h) * flat + i] * ((double)streams[i] * inv);
        }
        pre[h] = sigmoid_reference(mix_pre * scale[0] + base[h]) + 1e-6;
        if (!head) post[h] = 2.0 * sigmoid_reference(mix_post * scale[1] + base[hc + h]) + 1e-6;
    }
    for (unsigned d = 0; d < emb; ++d) {
        double v = 0.0;
        for (unsigned h = 0; h < hc; ++h) v += (double)streams[(size_t)h * emb + d] * pre[h];
        reduced[d] = v;
    }
}

static void test_hc(void) {
    enum { EMB = 31, HC = 4, FLAT = EMB * HC };
    float streams[FLAT], fn[2 * HC * FLAT], base[2 * HC];
    float reduced[EMB], head[EMB], post[HC], zero_streams[FLAT] = {0};
    const float scale[] = {0.37f, -0.21f};
    double expected[EMB], expected_post[HC];
    for (unsigned trial = 0; trial < 40; ++trial) {
        for (unsigned i = 0; i < FLAT; ++i) streams[i] = random_float() * 2.0f;
        for (unsigned i = 0; i < 2 * HC * FLAT; ++i) fn[i] = random_float() * 0.125f;
        for (unsigned i = 0; i < 2 * HC; ++i) base[i] = random_float();
        CHECK(hy4_hc_pre(reduced, post, streams, fn, scale, base,
                         EMB, HC, 1e-5f, 1e-6f, 2.0f));
        hc_reference(expected, expected_post, streams, fn, scale, base, EMB, HC, false);
        for (unsigned i = 0; i < EMB; ++i) near_value(reduced[i], expected[i], 4e-6);
        for (unsigned i = 0; i < HC; ++i) near_value(post[i], expected_post[i], 8e-7);
        CHECK(hy4_hc_head(head, streams, fn, scale, base, EMB, HC, 1e-5f, 1e-6f));
        hc_reference(expected, NULL, streams, fn, scale, base, EMB, HC, true);
        for (unsigned i = 0; i < EMB; ++i) near_value(head[i], expected[i], 4e-6);
    }
    memset(fn, 0, sizeof(fn));
    memset(base, 0, sizeof(base));
    CHECK(hy4_hc_pre(reduced, post, zero_streams, fn, scale, base,
                     EMB, HC, 1e-5f, 1e-6f, 2.0f));
    for (unsigned i = 0; i < EMB; ++i) CHECK(reduced[i] == 0.0f);
    for (unsigned i = 0; i < HC; ++i) near_value(post[i], 1.000001, 1e-7);
    CHECK(!hy4_hc_pre(reduced, post, streams, fn, scale, base,
                      EMB, 0, 1e-5f, 1e-6f, 2.0f));
    CHECK(!hy4_hc_head(head, streams, fn, scale, base,
                       EMB, HY4_MATH_MAX_HC + 1u, 1e-5f, 1e-6f));
}

static void test_post_rounding(void) {
    const float x[] = {0x1.000002p0f};
    const float post[] = {0x1.fffffcp-1f};
    const float residual[] = {-1.0f};
    float result;
    CHECK(hy4_hc_post(&result, x, residual, post, 1, 1));
    /* Rounded product is exactly 1, so the required separate add is zero.
     * A fused multiply-add returns -2^-46 and must fail this regression. */
    CHECK(bits(result) == bits(0.0f));
    volatile float fused = fmaf(x[0], post[0], residual[0]);
    CHECK(fused != result);

    enum { EMB = 17, HC = 4 };
    float xx[EMB], pp[HC], rr[EMB * HC], out[EMB * HC], in_place[EMB * HC];
    for (unsigned trial = 0; trial < 200; ++trial) {
        for (unsigned i = 0; i < EMB; ++i) xx[i] = random_float();
        for (unsigned i = 0; i < HC; ++i) pp[i] = random_float() + 1.1f;
        for (unsigned i = 0; i < EMB * HC; ++i) rr[i] = random_float();
        CHECK(hy4_hc_post(out, xx, rr, pp, EMB, HC));
        memcpy(in_place, rr, sizeof(rr));
        CHECK(hy4_hc_post(in_place, xx, in_place, pp, EMB, HC));
        for (unsigned h = 0; h < HC; ++h) {
            for (unsigned d = 0; d < EMB; ++d) {
                const unsigned i = h * EMB + d;
                volatile float product = xx[d] * pp[h];
                volatile float expected = rr[i] + product;
                CHECK(bits(out[i]) == bits(expected));
                CHECK(bits(in_place[i]) == bits(expected));
            }
        }
    }
}

static void attention_reference(double *out, const float *qa, const float *qr,
                                const float *kv, const float *pe, const float *sinks,
                                unsigned heads, unsigned keys, unsigned kvdim,
                                unsigned rope, unsigned stride, unsigned offset,
                                float scale, const uint32_t *ids, unsigned count) {
    double *probability = malloc(keys * sizeof(*probability));
    CHECK(probability != NULL);
    if (!ids) count = keys;
    for (unsigned h = 0; h < heads; ++h) {
        double denominator = sinks ? exp((double)sinks[h]) : 0.0;
        for (unsigned i = 0; i < count; ++i) {
            const unsigned p = ids ? ids[i] : i;
            double score = 0.0;
            for (unsigned d = 0; d < kvdim; ++d) score += (double)qa[h * kvdim + d] * kv[p * kvdim + d];
            for (unsigned d = 0; d < rope; ++d) score += (double)qr[h * stride + offset + d] * pe[p * rope + d];
            probability[i] = exp(score * scale);
            denominator += probability[i];
        }
        for (unsigned d = 0; d < kvdim; ++d) {
            double result = 0.0;
            for (unsigned i = 0; i < count; ++i) {
                const unsigned p = ids ? ids[i] : i;
                result += probability[i] / denominator * kv[p * kvdim + d];
            }
            out[h * kvdim + d] = result;
        }
    }
    free(probability);
}

static void test_attention(void) {
    float out, scratch[3];
    const float q[] = {0}, raw[] = {0, 0}, kv[] = {2, 4, 9000};
    const float pe[] = {0, 0, 9000}, sink[] = {0};
    CHECK(hy4_mla_attention(&out, q, raw, 2, 1, kv, pe, sink,
                            1, 2, 1, 1, 0.0625f, NULL, 0, scratch));
    near_value(out, 2.0, 1e-6); /* (2 + 4) / (2 real keys + sink). */
    CHECK(hy4_mla_attention(&out, q, raw, 2, 1, kv, pe, NULL,
                            1, 2, 1, 1, 0.0625f, NULL, 0, scratch));
    near_value(out, 3.0, 1e-6);
    const uint32_t only_second[] = {1}, future[] = {2};
    CHECK(hy4_mla_attention(&out, q, raw, 2, 1, kv, pe, sink,
                            1, 2, 1, 1, 0.0625f, only_second, 1, scratch));
    near_value(out, 2.0, 1e-6);
    CHECK(!hy4_mla_attention(&out, q, raw, 2, 1, kv, pe, sink,
                             1, 2, 1, 1, 0.0625f, future, 1, scratch));

    enum { HEADS = 3, KEYS = 17, KVDIM = 512, ROPE = 64, STRIDE = 256, OFFSET = 192 };
    float qa[HEADS * KVDIM], qr[HEADS * STRIDE], vv[KEYS * KVDIM], pp[KEYS * ROPE];
    float ss[HEADS], result[HEADS * KVDIM], all[HEADS * KVDIM], work[KEYS];
    double reference[HEADS * KVDIM];
    for (unsigned i = 0; i < HEADS * KVDIM; ++i) qa[i] = random_float() * 0.125f;
    for (unsigned i = 0; i < HEADS * STRIDE; ++i) qr[i] = random_float() * 0.125f;
    for (unsigned i = 0; i < KEYS * KVDIM; ++i) vv[i] = random_float();
    for (unsigned i = 0; i < KEYS * ROPE; ++i) pp[i] = random_float();
    for (unsigned i = 0; i < HEADS; ++i) ss[i] = (float)i - 0.75f;
    CHECK(hy4_mla_attention(result, qa, qr, STRIDE, OFFSET, vv, pp, ss,
                            HEADS, KEYS, KVDIM, ROPE, 0.0625f, NULL, 0, work));
    attention_reference(reference, qa, qr, vv, pp, ss, HEADS, KEYS, KVDIM,
                        ROPE, STRIDE, OFFSET, 0.0625f, NULL, 0);
    for (unsigned i = 0; i < HEADS * KVDIM; ++i) near_value(result[i], reference[i], 2e-6);
    const uint32_t subset[] = {16, 3, 11, 0, 5};
    CHECK(hy4_mla_attention(result, qa, qr, STRIDE, OFFSET, vv, pp, ss,
                            HEADS, KEYS, KVDIM, ROPE, 0.0625f, subset, 5, work));
    attention_reference(reference, qa, qr, vv, pp, ss, HEADS, KEYS, KVDIM,
                        ROPE, STRIDE, OFFSET, 0.0625f, subset, 5);
    for (unsigned i = 0; i < HEADS * KVDIM; ++i) near_value(result[i], reference[i], 2e-6);

    CHECK(hy4_mla_attention(all, qa, qr, STRIDE, OFFSET, vv, pp, ss,
                            HEADS, KEYS - 1, KVDIM, ROPE, 0.0625f, NULL, 0, work));
    for (unsigned i = 0; i < KVDIM; ++i) vv[(KEYS - 1) * KVDIM + i] = 1e6f;
    for (unsigned i = 0; i < ROPE; ++i) pp[(KEYS - 1) * ROPE + i] = 1e6f;
    CHECK(hy4_mla_attention(result, qa, qr, STRIDE, OFFSET, vv, pp, ss,
                            HEADS, KEYS - 1, KVDIM, ROPE, 0.0625f, NULL, 0, work));
    CHECK(memcmp(all, result, sizeof(all)) == 0);
}

static void test_gate(void) {
    float x[] = {-2.0f, -0.25f, 0.5f, 1.0f, 3.0f};
    const float gate[] = {-15.0f, -1.0f, 0.0f, 2.0f, 15.0f};
    float out[5];
    hy4_sigmoid_mul(out, x, gate, 5);
    for (unsigned i = 0; i < 5; ++i) near_value(out[i], x[i] * sigmoid_reference(gate[i]), 5e-7);
    hy4_sigmoid_mul(x, x, gate, 5);
    CHECK(memcmp(x, out, sizeof(x)) == 0);
}

int main(void) {
    CHECK(!hy4_f32_isfinite(from_bits(UINT32_C(0x7fc00001))));
    CHECK(!hy4_f32_isfinite(from_bits(UINT32_C(0x7f800000))));
    CHECK(!hy4_f32_isfinite(from_bits(UINT32_C(0xff800000))));
    CHECK(hy4_f32_isfinite(FLT_MAX));
    test_hc();
    test_post_rounding();
    test_attention();
    test_gate();
    printf("PASS HY4 math: %u checks; iHC, exact non-FMA post, gated sink MLA, causal/subset attention\n", checks);
    return 0;
}
