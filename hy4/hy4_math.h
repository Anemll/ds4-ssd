/* HY4 scalar F32 reference math.
 *
 * Tensor layouts follow the published HY4 GGUF graph. These small helpers
 * deliberately keep the independent-HC residual operations separate from
 * DS4's Sinkhorn HC. They also provide an attention correctness path
 * around the existing GPU matrix-vector products.
 *
 * Volatile intermediate stores in add/mul preserve F32 rounding and prevent
 * contraction/reassociation even when the including runtime uses -ffast-math.
 */
#ifndef DS4_HY4_MATH_H
#define DS4_HY4_MATH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define HY4_MATH_MAX_HC 16u

/* C's isnan/isfinite can be folded away by an including -ffast-math build.
 * Inspect IEEE-754 bits so invalid compute still fails closed in that build. */
static inline bool hy4_f32_isfinite(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    volatile uint32_t observed = bits;
    return (observed & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline float hy4_f32_add(float a, float b) {
    volatile float out = a + b;
    return out;
}

static inline float hy4_f32_mul(float a, float b) {
    volatile float out = a * b;
    return out;
}

static inline float hy4_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* x and out may alias. logits are the unactivated attention-gate projection. */
static inline void hy4_sigmoid_mul(float *out, const float *x,
                                    const float *logits, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = hy4_f32_mul(x[i], hy4_sigmoid(logits[i]));
    }
}

static inline float hy4_rms_inv(const float *x, uint32_t n, float eps) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum = hy4_f32_add(sum, hy4_f32_mul(x[i], x[i]));
    }
    return 1.0f / sqrtf(hy4_f32_add(sum / (float)n, eps));
}

static inline float hy4_hc_mix_row(const float *row, const float *x,
                                   uint32_t n, float rms_inv) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float normalized = hy4_f32_mul(x[i], rms_inv);
        sum = hy4_f32_add(sum, hy4_f32_mul(row[i], normalized));
    }
    return sum;
}

/* streams: [hc][emb], fn: [2*hc][hc*emb], scale: [2], base: [2*hc].
 * reduced: [emb], post: [hc]. Reduction is stream 0, then 1, then 2, ...
 */
static inline bool hy4_hc_pre(float *reduced, float *post,
                               const float *streams, const float *fn,
                               const float *scale, const float *base,
                               uint32_t emb, uint32_t hc, float rms_eps,
                               float hc_eps, float magnitude) {
    if (!reduced || !post || !streams || !fn || !scale || !base ||
        !emb || !hc || hc > HY4_MATH_MAX_HC || emb > UINT32_MAX / hc) return false;
    const uint32_t flat = emb * hc;
    const float inv = hy4_rms_inv(streams, flat, rms_eps);
    float pre[HY4_MATH_MAX_HC];
    for (uint32_t h = 0; h < hc; ++h) {
        float mix = hy4_hc_mix_row(fn + (size_t)h * flat, streams, flat, inv);
        mix = hy4_f32_add(hy4_f32_mul(mix, scale[0]), base[h]);
        pre[h] = hy4_f32_add(hy4_sigmoid(mix), hc_eps);
        mix = hy4_hc_mix_row(fn + (size_t)(hc + h) * flat, streams, flat, inv);
        mix = hy4_f32_add(hy4_f32_mul(mix, scale[1]), base[hc + h]);
        post[h] = hy4_f32_add(hy4_f32_mul(hy4_sigmoid(mix), magnitude), hc_eps);
    }
    for (uint32_t d = 0; d < emb; ++d) {
        float value = hy4_f32_mul(streams[d], pre[0]);
        for (uint32_t h = 1; h < hc; ++h) {
            value = hy4_f32_add(value, hy4_f32_mul(streams[(size_t)h * emb + d], pre[h]));
        }
        reduced[d] = value;
    }
    return true;
}

/* out and residual may alias. Each multiply rounds BEFORE its residual add. */
static inline bool hy4_hc_post(float *out, const float *x,
                                const float *residual, const float *post,
                                uint32_t emb, uint32_t hc) {
    if (!out || !x || !residual || !post || !emb || !hc ||
        hc > HY4_MATH_MAX_HC || emb > UINT32_MAX / hc) return false;
    for (uint32_t h = 0; h < hc; ++h) {
        for (uint32_t d = 0; d < emb; ++d) {
            const size_t i = (size_t)h * emb + d;
            out[i] = hy4_f32_add(residual[i], hy4_f32_mul(x[d], post[h]));
        }
    }
    return true;
}

/* fn: [hc][hc*emb], scale: [1], base: [hc]. */
static inline bool hy4_hc_head(float *reduced, const float *streams,
                                const float *fn, const float *scale,
                                const float *base, uint32_t emb, uint32_t hc,
                                float rms_eps, float hc_eps) {
    if (!reduced || !streams || !fn || !scale || !base || !emb || !hc ||
        hc > HY4_MATH_MAX_HC || emb > UINT32_MAX / hc) return false;
    const uint32_t flat = emb * hc;
    const float inv = hy4_rms_inv(streams, flat, rms_eps);
    float pre[HY4_MATH_MAX_HC];
    for (uint32_t h = 0; h < hc; ++h) {
        float mix = hy4_hc_mix_row(fn + (size_t)h * flat, streams, flat, inv);
        mix = hy4_f32_add(hy4_f32_mul(mix, scale[0]), base[h]);
        pre[h] = hy4_f32_add(hy4_sigmoid(mix), hc_eps);
    }
    for (uint32_t d = 0; d < emb; ++d) {
        float value = hy4_f32_mul(streams[d], pre[0]);
        for (uint32_t h = 1; h < hc; ++h) {
            value = hy4_f32_add(value, hy4_f32_mul(streams[(size_t)h * emb + d], pre[h]));
        }
        reduced[d] = value;
    }
    return true;
}

/* Absorbed MLA: q_abs [heads][kv_dim], kv_cache [n_keys][kv_dim],
 * kpe_cache [n_keys][rope_dim], q_raw [heads][q_stride] with its rope tail at
 * q_rope_offset. HY4 uses kv_dim=512, rope_dim=64, q_stride=256, offset=192.
 * The sink contributes probability mass with a zero value vector.
 *
 * n_keys is the causal prefix, including this token. A non-NULL selected list
 * must contain n_selected distinct indices within that prefix. NULL uses all
 * n_keys in their natural order. scores_scratch needs at least n_keys floats.
 * No cache data beyond n_keys is read. out is [heads][kv_dim].
 */
static inline bool hy4_mla_attention(float *out, const float *q_abs,
                                      const float *q_raw, uint32_t q_stride,
                                      uint32_t q_rope_offset,
                                      const float *kv_cache, const float *kpe_cache,
                                      const float *sinks, uint32_t n_heads,
                                      uint32_t n_keys, uint32_t kv_dim,
                                      uint32_t rope_dim, float scale,
                                      const uint32_t *selected, uint32_t n_selected,
                                      float *scores_scratch) {
    if (!out || !q_abs || !q_raw || !kv_cache || !kpe_cache || !scores_scratch ||
        !n_heads || !n_keys || !kv_dim || !rope_dim ||
        q_rope_offset > q_stride || rope_dim > q_stride - q_rope_offset) return false;
    const uint32_t count = selected ? n_selected : n_keys;
    if (!count || count > n_keys) return false;
    if (selected) {
        for (uint32_t i = 0; i < count; ++i) if (selected[i] >= n_keys) return false;
    }
    for (uint32_t h = 0; h < n_heads; ++h) {
        float maximum = sinks ? sinks[h] : -FLT_MAX;
        const float *qh = q_abs + (size_t)h * kv_dim;
        const float *qpe = q_raw + (size_t)h * q_stride + q_rope_offset;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t p = selected ? selected[i] : i;
            const float *kv = kv_cache + (size_t)p * kv_dim;
            const float *pe = kpe_cache + (size_t)p * rope_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < kv_dim; ++d) {
                score = hy4_f32_add(score, hy4_f32_mul(qh[d], kv[d]));
            }
            for (uint32_t d = 0; d < rope_dim; ++d) {
                score = hy4_f32_add(score, hy4_f32_mul(qpe[d], pe[d]));
            }
            score = hy4_f32_mul(score, scale);
            if (!hy4_f32_isfinite(score)) return false;
            scores_scratch[i] = score;
            if (score > maximum) maximum = score;
        }
        if (!hy4_f32_isfinite(maximum)) return false;
        float denominator = sinks ? expf(sinks[h] - maximum) : 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            scores_scratch[i] = expf(scores_scratch[i] - maximum);
            denominator = hy4_f32_add(denominator, scores_scratch[i]);
        }
        const float inv = 1.0f / denominator;
        for (uint32_t d = 0; d < kv_dim; ++d) {
            float value = 0.0f;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t p = selected ? selected[i] : i;
                const float probability = hy4_f32_mul(scores_scratch[i], inv);
                value = hy4_f32_add(value,
                    hy4_f32_mul(probability, kv_cache[(size_t)p * kv_dim + d]));
            }
            out[(size_t)h * kv_dim + d] = value;
        }
    }
    return true;
}

#endif /* DS4_HY4_MATH_H */
