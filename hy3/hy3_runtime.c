/* HY3 Metal decode runtime.
 *
 * Architecture-specific attention is intentionally small.  Dense projections,
 * normalization, routing and expert execution all stay on DS4's existing
 * tensor-resident Metal path.  Routed layers consume the full GGUF expert
 * tensors directly and execute the selected top-8 experts in one fused MoE
 * call, without staging individual experts through the CPU.
 */

#define HY3_DENSE_FF 13312u
#define HY3_Q_DIM ((uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM)
#define HY3_KV_DIM ((uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM)
#define HY3_PREFILL_CAP_DEFAULT 256u
#define HY3_MTP_MAX_DRAFT 16u

typedef struct {
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *last;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *q_raw;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k_raw;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_scratch;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *dense_gate;
    ds4_gpu_tensor *dense_up;
    ds4_gpu_tensor *dense_mid;
    ds4_gpu_tensor *dense_out;
    ds4_gpu_tensor *one_selected;
    ds4_gpu_tensor *one_weight;
    ds4_gpu_tensor *logits_gpu;
    ds4_gpu_tensor *target_hidden;
    ds4_gpu_tensor *target_pending;
    ds4_gpu_tensor *mtp_prev;
    ds4_gpu_tensor *mtp_carry;
    ds4_gpu_tensor *mtp_concat;
    ds4_gpu_tensor *mtp_hidden;
    ds4_gpu_tensor *mtp_logits_gpu;
    ds4_gpu_tensor *mtp_tops_gpu;
    ds4_gpu_tensor *spec_logits_gpu;
    ds4_gpu_tensor *spec_tops_gpu;
    ds4_gpu_tensor *zero_hidden;
    ds4_gpu_tensor *mtp_k;
    ds4_gpu_tensor *mtp_v;
    ds4_gpu_tensor *layer_k[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_v[DS4_MAX_LAYER];
    uint32_t n_past;
    uint32_t mtp_n_past;
    uint32_t kv_ctx_pad;
    bool nax_f16_kv;
    bool mtp_cache_valid;
    bool mtp_auto_disabled;
    bool mtp_auto_logged;
    uint64_t mtp_cycles;
    uint64_t mtp_proposed;
    uint64_t mtp_accepted;
    double mtp_draft_seconds;
    double mtp_verify_seconds;
    double mtp_repair_seconds;
} hy3_runtime;

static bool hy3_session_active(const ds4_session *s) {
    return s && s->engine && DS4_MODEL_VARIANT == DS4_VARIANT_HY3;
}

static hy3_runtime *hy3_rt(ds4_session *s) {
    return s ? (hy3_runtime *)s->variant_runtime : NULL;
}

static bool hy3_mtp_runtime_active(const ds4_session *s) {
    const hy3_runtime *rt = s ? (const hy3_runtime *)s->variant_runtime : NULL;
    return s && s->engine && s->engine->mtp_ready && rt &&
           s->engine->mtp_draft_tokens > 1 &&
           getenv("DS4_MTP_SPEC_DISABLE") == NULL &&
           !rt->mtp_auto_disabled;
}

static bool hy3_mtp_auto_fallback_enabled(void) {
    const char *env = getenv("DS4_HY3_MTP_AUTO_FALLBACK");
    return !env || !env[0] || atoi(env) != 0;
}

static bool hy3_mtp_unsafe_batch_requested(const ds4_session *s) {
    return s && s->prefill_cap >= 2u &&
           env_flag_enabled("DS4_HY3_MTP_BATCH_VERIFY") &&
           env_flag_enabled("DS4_HY3_MTP_UNSAFE_BATCH_VERIFY");
}

/* The strict one-token verifier cannot save target rows, so on M5 it is
 * always net-negative.  Select the measured target-only route before prompt
 * sync as well as decode; otherwise the nominal fallback still pays to build
 * block-80 KV for every system-prompt token. */
static bool hy3_mtp_should_start_target_only(const ds4_session *s) {
    return s && s->engine && s->engine->mtp_ready &&
           s->engine->mtp_draft_tokens > 1 &&
           getenv("DS4_MTP_SPEC_DISABLE") == NULL &&
           hy3_mtp_auto_fallback_enabled() &&
           !hy3_mtp_unsafe_batch_requested(s);
}

static bool hy3_mtp_active_state_requested(const ds4_session *s) {
    return s && s->engine && s->engine->mtp_ready &&
           s->engine->mtp_draft_tokens > 1 &&
           getenv("DS4_MTP_SPEC_DISABLE") == NULL &&
           (!hy3_mtp_auto_fallback_enabled() ||
            hy3_mtp_unsafe_batch_requested(s));
}

static uint32_t hy3_decode_split_layers(void) {
    static bool initialized;
    static uint32_t value;
    if (initialized) return value;
    initialized = true;
    value = 16u;
    const char *env = getenv("DS4_HY3_DECODE_SPLIT_LAYERS");
    if (env && env[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed <= DS4_N_LAYER) value = (uint32_t)parsed;
    }
    return value;
}

typedef enum {
    HY3_DECODE_FLUSH_BOUNDED_2 = 0,
    HY3_DECODE_FLUSH_BLOCKING_1,
    HY3_DECODE_FLUSH_UNBOUNDED,
} hy3_decode_flush_mode;

static hy3_decode_flush_mode hy3_decode_flush_mode_requested(void) {
    static bool initialized;
    static hy3_decode_flush_mode value;
    if (initialized) return value;
    initialized = true;
    value = HY3_DECODE_FLUSH_BOUNDED_2;
    const char *env = getenv("DS4_HY3_DECODE_BLOCKING_FLUSH");
    if (env && env[0]) {
        value = atoi(env) != 0
            ? HY3_DECODE_FLUSH_BLOCKING_1
            : HY3_DECODE_FLUSH_UNBOUNDED;
    }
    const char *unbounded = getenv("DS4_HY3_DECODE_UNBOUNDED_FLUSH");
    if (unbounded && unbounded[0] && atoi(unbounded) != 0) {
        value = HY3_DECODE_FLUSH_UNBOUNDED;
    }
    return value;
}

static const char *hy3_decode_flush_mode_name(void) {
    switch (hy3_decode_flush_mode_requested()) {
        case HY3_DECODE_FLUSH_BLOCKING_1: return "blocking depth 1";
        case HY3_DECODE_FLUSH_UNBOUNDED:  return "unbounded diagnostic";
        default:                          return "bounded depth 2";
    }
}

static uint32_t hy3_prefill_cap_requested(void) {
    uint32_t cap = HY3_PREFILL_CAP_DEFAULT;
    const char *env = getenv("DS4_HY3_PREFILL_CHUNK");
    if (env && env[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && parsed != 0ul) {
            cap = ds4_hy3_prefill_cap_normalize(parsed);
        }
    }
    return cap;
}

/* Full-GGUF HY3 already has persistent lazy Metal views, so a very small
 * prompt suffix can be cheaper through the canonical one-row decode kernels
 * than through the layer-major batch setup.  This remains profile-tuned and
 * off by default on unqualified machines. */
static uint32_t hy3_token_major_max_sync_requested(void) {
    const char *env = getenv("DS4_HY3_TOKEN_MAJOR_MAX_SYNC");
    if (!env || !env[0]) return 0u;
    char *end = NULL;
    const unsigned long parsed = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return 0u;
    return parsed > 64ul ? 64u : (uint32_t)parsed;
}

static ds4_mm_hint hy3_q8_matmul_hint(void) {
    /* Keep the precision policy at the call site as well as engine startup:
     * Metal backend selectors are process-global and may already be cached if
     * a different model was opened before HY3 in the same process. */
    const char *audit = getenv("DS4_HY3_ENABLE_DENSE_NAX");
    return audit && audit[0] && atoi(audit) != 0
        ? DS4_MM_AUTO : DS4_MM_STABLE_Q8;
}

static bool hy3_matmul(ds4_gpu_tensor *out, const ds4_model *m,
                       const ds4_tensor *w, uint64_t in_dim,
                       uint64_t out_dim, const ds4_gpu_tensor *x) {
    if (!out || !m || !w || !x) return false;
    if (w->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_tensor_ex(
                   out, m->map, m->size, w->abs_offset,
                   in_dim, out_dim, x, 1, hy3_q8_matmul_hint()) != 0;
    }
    return ds4_gpu_matmul_gguf_tensor(out, m->map, m->size, w->abs_offset,
                                      w->type, in_dim, out_dim, x, 1) != 0;
}

static bool hy3_matmul_batch(ds4_gpu_tensor *out, const ds4_model *m,
                             const ds4_tensor *w, uint64_t in_dim,
                             uint64_t out_dim, const ds4_gpu_tensor *x,
                             uint32_t n_tokens) {
    if (!out || !m || !w || !x || n_tokens == 0) return false;
    if (w->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_tensor_ex(
                   out, m->map, m->size, w->abs_offset,
                   in_dim, out_dim, x, n_tokens, hy3_q8_matmul_hint()) != 0;
    }
    return ds4_gpu_matmul_gguf_tensor(out, m->map, m->size, w->abs_offset,
                                      w->type, in_dim, out_dim, x, n_tokens) != 0;
}

static bool hy3_embed_token(ds4_gpu_tensor *out, const ds4_model *m,
                            const ds4_tensor *embd, int token) {
    if (!out || !m || !embd || token < 0 || (uint64_t)token >= embd->dim[1] ||
        embd->type != DS4_TENSOR_Q4_K || embd->dim[0] != DS4_N_EMBD) return false;
    const uint64_t row_bytes = embd->bytes / embd->dim[1];
    return ds4_gpu_hy3_get_row_q4_k_tensor(out, m->map, m->size,
                                            embd->abs_offset + (uint64_t)token * row_bytes,
                                            DS4_N_EMBD) != 0;
}

static bool hy3_embed_batch(ds4_gpu_tensor *out, const ds4_model *m,
                            const ds4_tensor *embd, const int *tokens,
                            uint32_t n_tokens) {
    if (!out || !m || !embd || !tokens || n_tokens == 0 ||
        embd->type != DS4_TENSOR_Q4_K || embd->dim[0] != DS4_N_EMBD) return false;
    return ds4_gpu_hy3_get_rows_q4_k_tensor(out, m->map, m->size,
                                             embd->abs_offset,
                                             (uint32_t)embd->dim[1], tokens,
                                             n_tokens, DS4_N_EMBD) != 0;
}

static bool hy3_fused_one_expert(
        hy3_runtime          *rt,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const ds4_model      *m,
        const ds4_tensor     *wg,
        const ds4_tensor     *wu,
        const ds4_tensor     *wd,
        uint32_t              in_dim,
        uint32_t              mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x) {
    const uint64_t gate_row = routed_expert_row_bytes_for_type(wg->type, in_dim);
    const uint64_t down_row = routed_expert_row_bytes_for_type(wd->type, mid_dim);
    return ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, NULL,
                                         m->map, m->size,
                                         wg->abs_offset, wu->abs_offset, wd->abs_offset,
                                         wg->type, wd->type,
                                         (uint64_t)mid_dim * gate_row, gate_row,
                                         (uint64_t)out_dim * down_row, down_row,
                                         in_dim, mid_dim, out_dim,
                                         rt->one_selected, rt->one_weight,
                                         1, 1, 0.0f, x) != 0;
}

static bool hy3_fused_routed(hy3_runtime *rt, const ds4_model *m,
                             const ds4_layer_weights *l) {
    const uint64_t gate_row = routed_expert_row_bytes(l->ffn_gate_exps);
    const uint64_t down_row = routed_expert_row_bytes(l->ffn_down_exps);
    return ds4_gpu_routed_moe_one_tensor(
               rt->routed_out, rt->routed_gate, rt->routed_up, rt->routed_mid,
               rt->routed_down, m->map, m->size,
               l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
               l->ffn_down_exps->abs_offset,
               l->ffn_gate_exps->type, l->ffn_down_exps->type,
               (uint64_t)DS4_N_FF_EXP * gate_row, gate_row,
               (uint64_t)DS4_N_EMBD * down_row, down_row,
               DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
               rt->router_selected, rt->router_weights,
               DS4_N_EXPERT, DS4_N_EXPERT_USED, 0.0f, rt->norm) != 0;
}

static bool hy3_fused_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const ds4_model      *m,
        const ds4_tensor     *wg,
        const ds4_tensor     *wu,
        const ds4_tensor     *wd,
        uint32_t              in_dim,
        uint32_t              mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    const uint64_t gate_row = routed_expert_row_bytes_for_type(wg->type, in_dim);
    const uint64_t down_row = routed_expert_row_bytes_for_type(wd->type, mid_dim);
    bool mid_is_f16 = false;
    return ds4_gpu_routed_moe_batch_tensor(
               out, gate, up, mid, experts, m->map, m->size,
               wg->abs_offset, wu->abs_offset, wd->abs_offset,
               wg->type, wd->type,
               (uint64_t)mid_dim * gate_row, gate_row,
               (uint64_t)out_dim * down_row, down_row,
               in_dim, mid_dim, out_dim, selected, weights,
               n_total_expert, n_expert, 0.0f, x, n_tokens, &mid_is_f16) != 0;
}

static void hy3_runtime_free(hy3_runtime *rt) {
    if (!rt) return;
#define HY3_FREE(name) do { ds4_gpu_tensor_free(rt->name); rt->name = NULL; } while (0)
    HY3_FREE(cur); HY3_FREE(last); HY3_FREE(next); HY3_FREE(norm);
    HY3_FREE(q_raw); HY3_FREE(q); HY3_FREE(k_raw); HY3_FREE(k); HY3_FREE(v);
    HY3_FREE(heads); HY3_FREE(attn_scratch); HY3_FREE(attn_out);
    HY3_FREE(router_logits); HY3_FREE(router_probs);
    HY3_FREE(router_selected); HY3_FREE(router_weights);
    HY3_FREE(routed_gate); HY3_FREE(routed_up); HY3_FREE(routed_mid);
    HY3_FREE(routed_down); HY3_FREE(routed_out);
    HY3_FREE(shared_gate); HY3_FREE(shared_up); HY3_FREE(shared_mid); HY3_FREE(shared_out);
    HY3_FREE(ffn_out); HY3_FREE(dense_gate); HY3_FREE(dense_up);
    HY3_FREE(dense_mid); HY3_FREE(dense_out);
    HY3_FREE(one_selected); HY3_FREE(one_weight); HY3_FREE(logits_gpu);
    HY3_FREE(target_hidden); HY3_FREE(target_pending); HY3_FREE(mtp_prev);
    HY3_FREE(mtp_carry);
    HY3_FREE(mtp_concat); HY3_FREE(mtp_hidden); HY3_FREE(mtp_logits_gpu);
    HY3_FREE(mtp_tops_gpu); HY3_FREE(spec_logits_gpu); HY3_FREE(spec_tops_gpu);
    HY3_FREE(zero_hidden); HY3_FREE(mtp_k); HY3_FREE(mtp_v);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(rt->layer_k[il]);
        ds4_gpu_tensor_free(rt->layer_v[il]);
        rt->layer_k[il] = rt->layer_v[il] = NULL;
    }
#undef HY3_FREE
}

static bool hy3_runtime_alloc(ds4_session *s) {
    hy3_runtime *rt = hy3_rt(s);
    const uint64_t e = DS4_N_EMBD;
    const uint64_t active = DS4_N_EXPERT_USED;
    const uint64_t pc = s->prefill_cap;
    const uint64_t attn_pc = pc < DS4_HY3_PREFILL_ATTN_STRIPE
        ? pc : DS4_HY3_PREFILL_ATTN_STRIPE;
    rt->kv_ctx_pad = ((uint32_t)s->ctx_size + 31u) & ~31u;
    rt->nax_f16_kv = false;
    if (hy3_nax_half_requested()) {
        rt->nax_f16_kv = ds4_gpu_hy3_nax_f16_supported() != 0;
        if (!rt->nax_f16_kv) {
            fprintf(stderr,
                    "ds4: HY3 NAX-half attention unavailable; using Q8_0 KV\n");
        }
    }
    const uint64_t cache_bytes = rt->nax_f16_kv
        ? (uint64_t)rt->kv_ctx_pad * DS4_N_HEAD_KV * DS4_N_HEAD_DIM *
              sizeof(uint16_t)
        : (uint64_t)s->ctx_size * DS4_N_HEAD_KV *
              (DS4_N_HEAD_DIM / 32u) * 34u;
#define HY3_ALLOC(name, count, type) \
    (rt->name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(type)))
    HY3_ALLOC(cur, pc * e, float); HY3_ALLOC(last, e, float);
    HY3_ALLOC(next, pc * e, float); HY3_ALLOC(norm, pc * e, float);
    HY3_ALLOC(q_raw, pc * HY3_Q_DIM, float); HY3_ALLOC(q, pc * HY3_Q_DIM, float);
    HY3_ALLOC(k_raw, pc * HY3_KV_DIM, float); HY3_ALLOC(k, pc * HY3_KV_DIM, float);
    HY3_ALLOC(v, pc * HY3_KV_DIM, float); HY3_ALLOC(heads, pc * HY3_Q_DIM, float);
    HY3_ALLOC(attn_scratch,
              attn_pc * DS4_N_HEAD * 32u * (DS4_N_HEAD_DIM + 2u), float);
    HY3_ALLOC(attn_out, pc * e, float);
    HY3_ALLOC(router_logits, pc * DS4_N_EXPERT, float);
    HY3_ALLOC(router_probs, pc * DS4_N_EXPERT, float);
    HY3_ALLOC(router_selected, pc * active, int32_t);
    HY3_ALLOC(router_weights, pc * active, float);
    HY3_ALLOC(routed_gate, pc * active * DS4_N_FF_EXP, float);
    HY3_ALLOC(routed_up, pc * active * DS4_N_FF_EXP, float);
    HY3_ALLOC(routed_mid, pc * active * DS4_N_FF_EXP, float);
    HY3_ALLOC(routed_down, pc * active * e, float);
    HY3_ALLOC(routed_out, pc * e, float);
    HY3_ALLOC(shared_gate, pc * DS4_N_FF_EXP, float);
    HY3_ALLOC(shared_up, pc * DS4_N_FF_EXP, float);
    HY3_ALLOC(shared_mid, pc * DS4_N_FF_EXP, float);
    HY3_ALLOC(shared_out, pc * e, float); HY3_ALLOC(ffn_out, pc * e, float);
    HY3_ALLOC(dense_gate, pc * HY3_DENSE_FF, float);
    HY3_ALLOC(dense_up, pc * HY3_DENSE_FF, float);
    HY3_ALLOC(dense_mid, pc * HY3_DENSE_FF, float);
    HY3_ALLOC(dense_out, pc * e, float);
    HY3_ALLOC(one_selected, pc, int32_t); HY3_ALLOC(one_weight, pc, float);
    HY3_ALLOC(logits_gpu, DS4_N_VOCAB, float);
    if (s->engine->mtp_ready) {
        HY3_ALLOC(target_hidden, pc * e, float);
        HY3_ALLOC(target_pending, e, float);
        HY3_ALLOC(mtp_prev, pc * e, float);
        HY3_ALLOC(mtp_carry, e, float);
        HY3_ALLOC(mtp_concat, pc * 2u * e, float);
        HY3_ALLOC(mtp_hidden, pc * e, float);
        HY3_ALLOC(mtp_logits_gpu, DS4_N_VOCAB, float);
        HY3_ALLOC(mtp_tops_gpu, 1u, int32_t);
        HY3_ALLOC(spec_logits_gpu, HY3_MTP_MAX_DRAFT * DS4_N_VOCAB, float);
        HY3_ALLOC(spec_tops_gpu, HY3_MTP_MAX_DRAFT, int32_t);
        HY3_ALLOC(zero_hidden, e, float);
        rt->mtp_k = ds4_gpu_tensor_alloc(cache_bytes);
        rt->mtp_v = ds4_gpu_tensor_alloc(cache_bytes);
    }
#undef HY3_ALLOC
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        rt->layer_k[il] = ds4_gpu_tensor_alloc(cache_bytes);
        rt->layer_v[il] = ds4_gpu_tensor_alloc(cache_bytes);
    }

    bool ok = rt->cur && rt->last && rt->next && rt->norm && rt->q_raw && rt->q &&
              rt->k_raw && rt->k && rt->v && rt->heads && rt->attn_scratch &&
              rt->attn_out &&
              rt->router_logits && rt->router_probs && rt->router_selected &&
              rt->router_weights && rt->routed_gate && rt->routed_up &&
              rt->routed_mid && rt->routed_down && rt->routed_out &&
              rt->shared_gate && rt->shared_up && rt->shared_mid &&
              rt->shared_out && rt->ffn_out && rt->dense_gate && rt->dense_up &&
              rt->dense_mid && rt->dense_out && rt->one_selected &&
              rt->one_weight && rt->logits_gpu;
    if (ok && s->engine->mtp_ready) {
        ok = rt->target_hidden && rt->target_pending && rt->mtp_prev &&
             rt->mtp_carry &&
             rt->mtp_concat && rt->mtp_hidden && rt->mtp_logits_gpu &&
             rt->mtp_tops_gpu && rt->spec_logits_gpu && rt->spec_tops_gpu &&
             rt->zero_hidden && rt->mtp_k && rt->mtp_v;
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = rt->layer_k[il] && rt->layer_v[il];
    }
    if (ok) {
        int32_t zero[DS4_HY3_PREFILL_CAP_MAX] = {0};
        float one[DS4_HY3_PREFILL_CAP_MAX];
        for (uint32_t i = 0; i < s->prefill_cap; ++i) one[i] = 1.0f;
        ok = ds4_gpu_tensor_write(rt->one_selected, 0, zero,
                                  pc * sizeof(zero[0])) &&
             ds4_gpu_tensor_write(rt->one_weight, 0, one,
                                  pc * sizeof(one[0]));
        if (ok && s->engine->mtp_ready) {
            ok = ds4_gpu_tensor_fill_f32(rt->zero_hidden, 0.0f, e) &&
                 ds4_gpu_tensor_fill_f32(rt->target_pending, 0.0f, e);
        }
    }
    return ok;
}

static void hy3_session_free(ds4_session *s) {
    if (!s) return;
    hy3_runtime *rt = hy3_rt(s);
    hy3_runtime_free(rt);
    free(rt);
    s->variant_runtime = NULL;
}

static int hy3_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
#ifdef DS4_NO_GPU
    (void)ctx_size;
    fprintf(stderr, "ds4: HY3 requires the Metal backend\n");
    return 1;
#else
    if (e->backend != DS4_BACKEND_METAL || !e->metal_ready) {
        fprintf(stderr, "ds4: HY3 currently requires the Metal backend\n");
        return 1;
    }
    ds4_session *s = xcalloc(1, sizeof(*s));
    hy3_runtime *rt = xcalloc(1, sizeof(*rt));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = e->quality ? 1u : hy3_prefill_cap_requested();
    s->variant_runtime = rt;
    rt->mtp_auto_disabled = hy3_mtp_should_start_target_only(s);
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        s->mtp_draft_token = -1;
    }
    if (!hy3_runtime_alloc(s)) {
        hy3_session_free(s);
        free(s->logits);
        free(s->mtp_logits);
        free(s);
        return 1;
    }
    /* HY3 uses its decode-shaped graph for prompt tokens too.  In lazy-view
     * mode, persist those mmap wrappers before the first token; otherwise the
     * per-token synchronize clears and recreates every tensor view. */
    if (!metal_graph_prepare_decode_model_views_engine(e)) {
        fprintf(stderr, "ds4: HY3 failed to prepare persistent Metal model views\n");
        hy3_session_free(s);
        free(s->logits);
        free(s->mtp_logits);
        free(s);
        return 1;
    }
    const uint64_t cache_bytes = rt->nax_f16_kv
        ? (uint64_t)rt->kv_ctx_pad * DS4_N_HEAD_KV * DS4_N_HEAD_DIM *
              sizeof(uint16_t)
        : (uint64_t)ctx_size * DS4_N_HEAD_KV *
              (DS4_N_HEAD_DIM / 32u) * 34u;
    const uint64_t target_cache_bytes =
        (uint64_t)DS4_N_LAYER * 2u * cache_bytes;
    const uint64_t mtp_scratch_bytes = e->mtp_ready
        ? ((uint64_t)5u * s->prefill_cap * DS4_N_EMBD +
           (uint64_t)3u * DS4_N_EMBD +
           (uint64_t)(1u + HY3_MTP_MAX_DRAFT) * DS4_N_VOCAB) * sizeof(float) +
          (uint64_t)(1u + HY3_MTP_MAX_DRAFT) * sizeof(int32_t)
        : 0u;
    const uint64_t mtp_state_bytes = e->mtp_ready
        ? 2u * cache_bytes + mtp_scratch_bytes : 0u;
    char mtp_mem[80] = "";
    if (e->mtp_ready) {
        snprintf(mtp_mem, sizeof(mtp_mem),
                 " + %.2f MiB block80 state",
                 (double)mtp_state_bytes / (1024.0 * 1024.0));
    }
    fprintf(stderr,
            "ds4: HY3 runtime: %s, layer-major prefill cap %u, "
            "fused selected top-%u full-GGUF MoE, target cache %.2f GiB%s, "
            "decode split %u layers (%s)%s\n",
            rt->nax_f16_kv ? "direct head-major F16 NAX GQA8" :
                             "Q8_0 split-KV GQA",
            s->prefill_cap, DS4_N_EXPERT_USED,
            (double)target_cache_bytes / (1024.0 * 1024.0 * 1024.0),
            mtp_mem,
            hy3_decode_split_layers(),
            hy3_decode_flush_mode_name(),
            e->mtp_ready
                ? (hy3_mtp_runtime_active(s)
                    ? ", HY3 MTP block80 enabled"
                    : ", HY3 MTP block80 loaded but inactive")
                : "");
    const uint32_t token_major_max = hy3_token_major_max_sync_requested();
    if (token_major_max != 0u) {
        fprintf(stderr,
                "ds4: HY3 resident small-sync route: token-major at <=%u "
                "new tokens (override: DS4_HY3_TOKEN_MAJOR_MAX_SYNC)\n",
                token_major_max);
    }
    *out = s;
    return 0;
#endif
}

/* Encode one or more rows through HY3's layer-80 NextN predictor.  The caller
 * owns the command batch.  `prev0` is H_(pos0-1); for catch-up batches,
 * `target_rows` supplies H_pos0.. so the MTP hidden input is shifted right
 * without ever crossing the CPU.  Recursive draft rows pass n_tokens=1 and
 * use the previous MTP hidden as prev0 instead. */
static bool hy3_mtp_encode_rows(
        ds4_session         *s,
        const int           *tokens,
        uint32_t             n_tokens,
        uint32_t             pos0,
        const ds4_gpu_tensor *prev0,
        const ds4_gpu_tensor *target_rows,
        bool                 need_logits,
        const char         **stage) {
    hy3_runtime *rt = hy3_rt(s);
    ds4_engine *e = s ? s->engine : NULL;
    if (!s || !rt || !e || !e->mtp_ready || !tokens || n_tokens == 0 ||
        n_tokens > s->prefill_cap || !prev0) {
        return false;
    }
    const ds4_model *base = &e->model;
    const ds4_weights *bw = &e->weights;
    const ds4_model *m = &e->mtp_model;
    const hy3_mtp_weights *mw = &e->hy3_mtp_weights;
    const ds4_layer_weights *l = &mw->block;
    const uint64_t ebytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint32_t rows = n_tokens * DS4_N_EMBD;
    const float attn_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);

#define HY3_MTP_STEP(name, expr) do { \
        if (ok) { if (stage) *stage = (name); ok = (expr); } \
    } while (0)
    bool ok = true;
    HY3_MTP_STEP("mtp.prev0", ds4_gpu_tensor_copy(
        rt->mtp_prev, 0, prev0, 0, ebytes) != 0);
    if (n_tokens > 1u) {
        HY3_MTP_STEP("mtp.prev_shift", target_rows && ds4_gpu_tensor_copy(
            rt->mtp_prev, ebytes, target_rows, 0,
            (uint64_t)(n_tokens - 1u) * ebytes) != 0);
    }
    HY3_MTP_STEP("mtp.embedding", n_tokens == 1u
        ? hy3_embed_token(rt->next, base, bw->token_embd, tokens[0])
        : hy3_embed_batch(rt->next, base, bw->token_embd, tokens, n_tokens));
    /* AngelSlim/merged llama.cpp uses the semantic token position and keeps
     * E[x_0], with only H_-1 zero.  Keep vLLM's internal-slot convention as an
     * audit switch because it explicitly masks that first embedding. */
    if (pos0 == 0u && env_flag_enabled("DS4_HY3_MTP_ZERO_POS0_EMBED")) {
        HY3_MTP_STEP("mtp.embedding.zero_pos0", ds4_gpu_tensor_copy(
            rt->next, 0, rt->zero_hidden, 0, ebytes) != 0);
    }
    HY3_MTP_STEP("mtp.enorm", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->attn_out, rt->next, m->map, m->size, mw->enorm->abs_offset,
        DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.hnorm", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->norm, rt->mtp_prev, m->map, m->size, mw->hnorm->abs_offset,
        DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.concat", ds4_gpu_concat_f32_rows_tensor(
        rt->mtp_concat, rt->attn_out, rt->norm,
        DS4_N_EMBD, DS4_N_EMBD, n_tokens) != 0);
    HY3_MTP_STEP("mtp.eh_proj", hy3_matmul_batch(
        rt->cur, m, mw->eh_proj, 2u * DS4_N_EMBD, DS4_N_EMBD,
        rt->mtp_concat, n_tokens));

    HY3_MTP_STEP("mtp.attn.rms", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->norm, rt->cur, m->map, m->size, l->attn_norm->abs_offset,
        DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.attn.q", hy3_matmul_batch(
        rt->q_raw, m, l->attn_q, DS4_N_EMBD, HY3_Q_DIM, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.attn.k", hy3_matmul_batch(
        rt->k_raw, m, l->attn_k, DS4_N_EMBD, HY3_KV_DIM, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.attn.v", hy3_matmul_batch(
        rt->v, m, l->attn_v, DS4_N_EMBD, HY3_KV_DIM, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.attn.q_norm", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->q, rt->q_raw, m->map, m->size, l->attn_q_norm->abs_offset,
        DS4_N_HEAD_DIM, n_tokens * DS4_N_HEAD, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.attn.k_norm", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->k, rt->k_raw, m->map, m->size, l->attn_k_norm->abs_offset,
        DS4_N_HEAD_DIM, n_tokens * DS4_N_HEAD_KV, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.attn.q_rope", ds4_gpu_rope_neox_tensor(
        rt->q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
        (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
    HY3_MTP_STEP("mtp.attn.k_rope", ds4_gpu_rope_neox_tensor(
        rt->k, n_tokens, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
        (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
    HY3_MTP_STEP("mtp.attn.gqa", (rt->nax_f16_kv
        ? ds4_gpu_hy3_gqa_attention_f16_nax_batch_tensor(
              rt->heads, rt->attn_scratch, rt->mtp_k, rt->mtp_v,
              rt->q, rt->k, rt->v, pos0, n_tokens, (uint32_t)s->ctx_size,
              DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)
        : ds4_gpu_hy3_gqa_attention_batch_tensor(
              rt->heads, rt->attn_scratch, rt->mtp_k, rt->mtp_v,
              rt->q, rt->k, rt->v, pos0, n_tokens, (uint32_t)s->ctx_size,
              DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)) != 0);
    HY3_MTP_STEP("mtp.attn.output", hy3_matmul_batch(
        rt->attn_out, m, l->attn_output, HY3_Q_DIM, DS4_N_EMBD,
        rt->heads, n_tokens));
    HY3_MTP_STEP("mtp.attn.residual", ds4_gpu_add_tensor(
        rt->next, rt->cur, rt->attn_out, rows) != 0);
    HY3_MTP_STEP("mtp.ffn.rms", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->norm, rt->next, m->map, m->size, l->ffn_norm->abs_offset,
        DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.ffn.router", hy3_matmul_batch(
        rt->router_logits, m, l->ffn_gate_inp,
        DS4_N_EMBD, DS4_N_EXPERT, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.ffn.select", ds4_gpu_glm_router_select_tensor(
        rt->router_selected, rt->router_weights, rt->router_probs,
        m->map, m->size, l->ffn_exp_probs_b->abs_offset,
        DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE,
        true, rt->router_logits, n_tokens) != 0);
    HY3_MTP_STEP("mtp.ffn.routed", hy3_fused_batch(
        rt->routed_out, rt->routed_gate, rt->routed_up, rt->routed_mid,
        rt->routed_down, m, l->ffn_gate_exps, l->ffn_up_exps,
        l->ffn_down_exps, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
        rt->router_selected, rt->router_weights,
        DS4_N_EXPERT, DS4_N_EXPERT_USED, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.ffn.shared", hy3_fused_batch(
        rt->shared_out, rt->shared_gate, rt->shared_up, rt->shared_mid,
        NULL, m, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp,
        DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
        rt->one_selected, rt->one_weight, 1u, 1u, rt->norm, n_tokens));
    HY3_MTP_STEP("mtp.ffn.combine", ds4_gpu_add_tensor(
        rt->ffn_out, rt->routed_out, rt->shared_out, rows) != 0);
    HY3_MTP_STEP("mtp.ffn.residual", ds4_gpu_add_tensor(
        rt->cur, rt->next, rt->ffn_out, rows) != 0);
    HY3_MTP_STEP("mtp.final_norm", ds4_gpu_rms_norm_weight_rows_tensor(
        rt->mtp_hidden, rt->cur, m->map, m->size,
        mw->shared_head_norm->abs_offset,
        DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
    HY3_MTP_STEP("mtp.carry", ds4_gpu_tensor_copy(
        rt->mtp_carry, 0, rt->mtp_hidden,
        (uint64_t)(n_tokens - 1u) * ebytes, ebytes) != 0);

    if (need_logits) {
        const ds4_gpu_tensor *head_in = rt->mtp_hidden;
        if (n_tokens > 1u) {
            HY3_MTP_STEP("mtp.output.last", ds4_gpu_copy_f32_slice(
                rt->mtp_hidden, (n_tokens - 1u) * DS4_N_EMBD,
                rt->last, DS4_N_EMBD) != 0);
            head_in = rt->last;
        }
        HY3_MTP_STEP("mtp.output.logits", hy3_matmul(
            rt->mtp_logits_gpu, base, bw->output,
            DS4_N_EMBD, DS4_N_VOCAB, head_in));
        HY3_MTP_STEP("mtp.output.argmax", ds4_gpu_argmax_f32_tensor(
            rt->mtp_tops_gpu, rt->mtp_logits_gpu,
            DS4_N_VOCAB, 1u) != 0);
    }
#undef HY3_MTP_STEP
    return ok;
}

static int hy3_eval_token_ex(ds4_session *s, int token, bool mtp_draft,
                             char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    ds4_engine *e = s ? s->engine : NULL;
    if (!s || !rt || !e) return 1;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos = rt->n_past;
    const bool mtp_active = hy3_mtp_runtime_active(s);
    const char *stage = "embedding";
    if (pos >= (uint32_t)s->ctx_size) {
        snprintf(err, errlen, "HY3 context is full");
        return 1;
    }
    if (mtp_active && (!rt->mtp_cache_valid ? pos != 0u : rt->mtp_n_past != pos)) {
        snprintf(err, errlen,
                 "HY3 MTP cache is not aligned (target=%u mtp=%u valid=%d)",
                 pos, rt->mtp_n_past, rt->mtp_cache_valid ? 1 : 0);
        return 1;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = hy3_embed_token(rt->cur, m, w->token_embd, token);
#define HY3_STEP(name, expr) do { if (ok) { stage = (name); ok = (expr); } } while (0)
    const float attn_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t split_layers = hy3_decode_split_layers();
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        HY3_STEP("attn.rms", ds4_gpu_rms_norm_weight_tensor(
            rt->norm, rt->cur, m->map, m->size, l->attn_norm->abs_offset,
            DS4_N_EMBD, DS4_RMS_EPS) != 0);
        HY3_STEP("attn.q", hy3_matmul(rt->q_raw, m, l->attn_q,
                                       DS4_N_EMBD, HY3_Q_DIM, rt->norm));
        HY3_STEP("attn.k", hy3_matmul(rt->k_raw, m, l->attn_k,
                                       DS4_N_EMBD, HY3_KV_DIM, rt->norm));
        HY3_STEP("attn.v", hy3_matmul(rt->v, m, l->attn_v,
                                       DS4_N_EMBD, HY3_KV_DIM, rt->norm));
        HY3_STEP("attn.q_norm", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->q, rt->q_raw, m->map, m->size, l->attn_q_norm->abs_offset,
            DS4_N_HEAD_DIM, DS4_N_HEAD, DS4_RMS_EPS) != 0);
        HY3_STEP("attn.k_norm", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->k, rt->k_raw, m->map, m->size, l->attn_k_norm->abs_offset,
            DS4_N_HEAD_DIM, DS4_N_HEAD_KV, DS4_RMS_EPS) != 0);
        HY3_STEP("attn.q_rope", ds4_gpu_rope_neox_tensor(
            rt->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
            (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
        HY3_STEP("attn.k_rope", ds4_gpu_rope_neox_tensor(
            rt->k, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
            (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
        HY3_STEP("attn.gqa", (rt->nax_f16_kv
            ? ds4_gpu_hy3_gqa_attention_f16_nax_tensor(
                  rt->heads, rt->attn_scratch, rt->layer_k[il], rt->layer_v[il],
                  rt->q, rt->k, rt->v, pos, (uint32_t)s->ctx_size,
                  DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)
            : ds4_gpu_hy3_gqa_attention_tensor(
                  rt->heads, rt->attn_scratch, rt->layer_k[il], rt->layer_v[il],
                  rt->q, rt->k, rt->v, pos, (uint32_t)s->ctx_size,
                  DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)) != 0);
        HY3_STEP("attn.output", hy3_matmul(rt->attn_out, m, l->attn_output,
                                            HY3_Q_DIM, DS4_N_EMBD, rt->heads));
        HY3_STEP("attn.residual", ds4_gpu_add_tensor(rt->next, rt->cur,
                                                       rt->attn_out, DS4_N_EMBD) != 0);
        HY3_STEP("ffn.rms", ds4_gpu_rms_norm_weight_tensor(
            rt->norm, rt->next, m->map, m->size, l->ffn_norm->abs_offset,
            DS4_N_EMBD, DS4_RMS_EPS) != 0);
        if (il == 0) {
            HY3_STEP("ffn.dense_fused", hy3_fused_one_expert(
                rt, rt->dense_out, rt->dense_gate, rt->dense_up, rt->dense_mid,
                m, l->ffn_gate, l->ffn_up, l->ffn_down,
                DS4_N_EMBD, HY3_DENSE_FF, DS4_N_EMBD, rt->norm));
            HY3_STEP("ffn.dense_residual", ds4_gpu_add_tensor(
                rt->cur, rt->next, rt->dense_out, DS4_N_EMBD) != 0);
        } else {
            HY3_STEP("ffn.router", hy3_matmul(rt->router_logits, m, l->ffn_gate_inp,
                                               DS4_N_EMBD, DS4_N_EXPERT, rt->norm));
            HY3_STEP("ffn.select", ds4_gpu_glm_router_select_tensor(
                rt->router_selected, rt->router_weights, rt->router_probs,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset,
                DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE,
                true, rt->router_logits, 1) != 0);
            HY3_STEP("ffn.routed_fused", hy3_fused_routed(rt, m, l));
            HY3_STEP("ffn.shared_fused", hy3_fused_one_expert(
                rt, rt->shared_out, rt->shared_gate, rt->shared_up,
                rt->shared_mid, m, l->ffn_gate_shexp, l->ffn_up_shexp,
                l->ffn_down_shexp, DS4_N_EMBD, DS4_N_FF_EXP,
                DS4_N_EMBD, rt->norm));
            HY3_STEP("ffn.combine", ds4_gpu_add_tensor(rt->ffn_out,
                                                        rt->routed_out, rt->shared_out,
                                                        DS4_N_EMBD) != 0);
            HY3_STEP("ffn.residual", ds4_gpu_add_tensor(rt->cur, rt->next,
                                                         rt->ffn_out, DS4_N_EMBD) != 0);
        }
        if (ok && split_layers != 0u && il + 1u < DS4_N_LAYER &&
            ((il + 1u) % split_layers) == 0u) {
            stage = "layer.flush";
            switch (hy3_decode_flush_mode_requested()) {
                case HY3_DECODE_FLUSH_BLOCKING_1:
                    ok = ds4_gpu_flush_commands_blocking() != 0;
                    break;
                case HY3_DECODE_FLUSH_UNBOUNDED:
                    ok = ds4_gpu_flush_commands() != 0;
                    break;
                default:
                    ok = ds4_gpu_flush_commands_bounded(2u) != 0;
                    break;
            }
        }
    }
    ds4_gpu_tensor *target_hidden = e->mtp_ready ? rt->target_hidden : rt->norm;
    HY3_STEP("output.rms", ds4_gpu_rms_norm_weight_tensor(
        target_hidden, rt->cur, m->map, m->size, w->output_norm->abs_offset,
        DS4_N_EMBD, DS4_RMS_EPS) != 0);
    HY3_STEP("output.logits", hy3_matmul(rt->logits_gpu, m, w->output,
                                          DS4_N_EMBD, DS4_N_VOCAB, target_hidden));
    if (mtp_active) {
        if (ok) {
            ok = hy3_mtp_encode_rows(s, &token, 1u, pos,
                                     rt->target_pending, NULL,
                                     mtp_draft, &stage);
        }
        HY3_STEP("mtp.target_pending", ds4_gpu_tensor_copy(
            rt->target_pending, 0, target_hidden, 0,
            (uint64_t)DS4_N_EMBD * sizeof(float)) != 0);
    }
#undef HY3_STEP
    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "HY3 eval failed at %s", stage);
        s->checkpoint_valid = false;
        return 1;
    }
    if (!ds4_gpu_tensor_read(rt->logits_gpu, 0, s->logits,
                             (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        snprintf(err, errlen, "HY3 failed to read logits");
        s->checkpoint_valid = false;
        return 1;
    }
    s->mtp_draft_valid = false;
    if (mtp_active && mtp_draft) {
        int32_t top = -1;
        if (!ds4_gpu_tensor_read(rt->mtp_tops_gpu, 0, &top, sizeof(top))) {
            snprintf(err, errlen, "HY3 failed to read MTP argmax");
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = top;
        s->mtp_draft_valid = top >= 0 && top < (int32_t)DS4_N_VOCAB;
        if (s->mtp_logits && (getenv("DS4_MTP_FULL_LOGITS") ||
                             getenv("DS4_MTP_CONF_LOG"))) {
            if (!ds4_gpu_tensor_read(rt->mtp_logits_gpu, 0, s->mtp_logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float))) {
                snprintf(err, errlen, "HY3 failed to read MTP logits");
                s->checkpoint_valid = false;
                return 1;
            }
        }
    }
    rt->n_past++;
    if (mtp_active) {
        rt->mtp_n_past = rt->n_past;
        rt->mtp_cache_valid = true;
    }
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

static int hy3_eval_batch_ex(ds4_session *s, const int *tokens, uint32_t n_tokens,
                             bool need_logits, bool mtp_catchup,
                             bool commit_checkpoint, bool all_logits,
                             int *row_tops, char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    ds4_engine *e = s ? s->engine : NULL;
    if (!s || !rt || !e || !tokens || n_tokens < 2u ||
        n_tokens > s->prefill_cap ||
        n_tokens > DS4_HY3_PREFILL_CAP_MAX) return 1;
    if (all_logits && (n_tokens > HY3_MTP_MAX_DRAFT || !row_tops ||
                       !e->mtp_ready)) return 1;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos0 = rt->n_past;
    const uint32_t rows = n_tokens * DS4_N_EMBD;
    const char *stage = "embedding.batch";
    if (pos0 >= (uint32_t)s->ctx_size || n_tokens > (uint32_t)s->ctx_size - pos0) {
        snprintf(err, errlen, "HY3 context is full");
        return 1;
    }
    if (mtp_catchup && e->mtp_ready &&
        (!rt->mtp_cache_valid ? pos0 != 0u : rt->mtp_n_past != pos0)) {
        snprintf(err, errlen,
                 "HY3 MTP cache is not aligned (target=%u mtp=%u valid=%d)",
                 pos0, rt->mtp_n_past, rt->mtp_cache_valid ? 1 : 0);
        return 1;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = hy3_embed_batch(rt->cur, m, w->token_embd, tokens, n_tokens);
#define HY3_BATCH_STEP(name, expr) do { if (ok) { stage = (name); ok = (expr); } } while (0)
    const float attn_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t split_layers = hy3_decode_split_layers();
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
        const ds4_layer_weights *l = &w->layer[il];
        HY3_BATCH_STEP("attn.rms.batch", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->norm, rt->cur, m->map, m->size, l->attn_norm->abs_offset,
            DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
        HY3_BATCH_STEP("attn.q.batch", hy3_matmul_batch(
            rt->q_raw, m, l->attn_q, DS4_N_EMBD, HY3_Q_DIM, rt->norm, n_tokens));
        HY3_BATCH_STEP("attn.k.batch", hy3_matmul_batch(
            rt->k_raw, m, l->attn_k, DS4_N_EMBD, HY3_KV_DIM, rt->norm, n_tokens));
        HY3_BATCH_STEP("attn.v.batch", hy3_matmul_batch(
            rt->v, m, l->attn_v, DS4_N_EMBD, HY3_KV_DIM, rt->norm, n_tokens));
        HY3_BATCH_STEP("attn.q_norm.batch", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->q, rt->q_raw, m->map, m->size, l->attn_q_norm->abs_offset,
            DS4_N_HEAD_DIM, n_tokens * DS4_N_HEAD, DS4_RMS_EPS) != 0);
        HY3_BATCH_STEP("attn.k_norm.batch", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->k, rt->k_raw, m->map, m->size, l->attn_k_norm->abs_offset,
            DS4_N_HEAD_DIM, n_tokens * DS4_N_HEAD_KV, DS4_RMS_EPS) != 0);
        HY3_BATCH_STEP("attn.q_rope.batch", ds4_gpu_rope_neox_tensor(
            rt->q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
            (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
        HY3_BATCH_STEP("attn.k_rope.batch", ds4_gpu_rope_neox_tensor(
            rt->k, n_tokens, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
            (uint32_t)DS4_ROPE_ORIG_CTX, DS4_ROPE_FREQ_BASE, 1.0f) != 0);
        HY3_BATCH_STEP("attn.gqa.batch", (rt->nax_f16_kv
            ? ds4_gpu_hy3_gqa_attention_f16_nax_batch_tensor(
                  rt->heads, rt->attn_scratch, rt->layer_k[il], rt->layer_v[il],
                  rt->q, rt->k, rt->v, pos0, n_tokens, (uint32_t)s->ctx_size,
                  DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)
            : ds4_gpu_hy3_gqa_attention_batch_tensor(
                  rt->heads, rt->attn_scratch, rt->layer_k[il], rt->layer_v[il],
                  rt->q, rt->k, rt->v, pos0, n_tokens, (uint32_t)s->ctx_size,
                  DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, attn_scale)) != 0);
        HY3_BATCH_STEP("attn.output.batch", hy3_matmul_batch(
            rt->attn_out, m, l->attn_output, HY3_Q_DIM, DS4_N_EMBD,
            rt->heads, n_tokens));
        HY3_BATCH_STEP("attn.residual.batch", ds4_gpu_add_tensor(
            rt->next, rt->cur, rt->attn_out, rows) != 0);
        HY3_BATCH_STEP("ffn.rms.batch", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->norm, rt->next, m->map, m->size, l->ffn_norm->abs_offset,
            DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
        if (il == 0u) {
            HY3_BATCH_STEP("ffn.dense.batch", hy3_fused_batch(
                rt->dense_out, rt->dense_gate, rt->dense_up, rt->dense_mid, NULL,
                m, l->ffn_gate, l->ffn_up, l->ffn_down,
                DS4_N_EMBD, HY3_DENSE_FF, DS4_N_EMBD,
                rt->one_selected, rt->one_weight, 1u, 1u, rt->norm, n_tokens));
            HY3_BATCH_STEP("ffn.dense_residual.batch", ds4_gpu_add_tensor(
                rt->cur, rt->next, rt->dense_out, rows) != 0);
        } else {
            HY3_BATCH_STEP("ffn.router.batch", hy3_matmul_batch(
                rt->router_logits, m, l->ffn_gate_inp,
                DS4_N_EMBD, DS4_N_EXPERT, rt->norm, n_tokens));
            HY3_BATCH_STEP("ffn.select.batch", ds4_gpu_glm_router_select_tensor(
                rt->router_selected, rt->router_weights, rt->router_probs,
                m->map, m->size, l->ffn_exp_probs_b->abs_offset,
                DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE,
                true, rt->router_logits, n_tokens) != 0);
            HY3_BATCH_STEP("ffn.routed.batch", hy3_fused_batch(
                rt->routed_out, rt->routed_gate, rt->routed_up, rt->routed_mid,
                rt->routed_down, m, l->ffn_gate_exps, l->ffn_up_exps,
                l->ffn_down_exps, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
                rt->router_selected, rt->router_weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED, rt->norm, n_tokens));
            HY3_BATCH_STEP("ffn.shared.batch", hy3_fused_batch(
                rt->shared_out, rt->shared_gate, rt->shared_up, rt->shared_mid,
                NULL, m, l->ffn_gate_shexp, l->ffn_up_shexp, l->ffn_down_shexp,
                DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD,
                rt->one_selected, rt->one_weight, 1u, 1u, rt->norm, n_tokens));
            HY3_BATCH_STEP("ffn.combine.batch", ds4_gpu_add_tensor(
                rt->ffn_out, rt->routed_out, rt->shared_out, rows) != 0);
            HY3_BATCH_STEP("ffn.residual.batch", ds4_gpu_add_tensor(
                rt->cur, rt->next, rt->ffn_out, rows) != 0);
        }
        /* Tiny speculative verification has decode-like residency and command
         * depth, not large-prefill behavior.  Bound its command buffers at the
         * same layer cadence as ordinary decode so an 80-layer verifier does
         * not retain every mmap wrapper and dispatch until the final sync. */
        if (ok && all_logits && split_layers != 0u &&
            il + 1u < DS4_N_LAYER && ((il + 1u) % split_layers) == 0u) {
            stage = "layer.flush.batch";
            switch (hy3_decode_flush_mode_requested()) {
                case HY3_DECODE_FLUSH_BLOCKING_1:
                    ok = ds4_gpu_flush_commands_blocking() != 0;
                    break;
                case HY3_DECODE_FLUSH_UNBOUNDED:
                    ok = ds4_gpu_flush_commands() != 0;
                    break;
                default:
                    ok = ds4_gpu_flush_commands_bounded(2u) != 0;
                    break;
            }
        }
    }
    if (e->mtp_ready || all_logits) {
        HY3_BATCH_STEP("output.rms_all.batch", ds4_gpu_rms_norm_weight_rows_tensor(
            rt->target_hidden, rt->cur, m->map, m->size,
            w->output_norm->abs_offset,
            DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0);
        if (all_logits) {
            HY3_BATCH_STEP("output.logits_all.batch", hy3_matmul_batch(
                rt->spec_logits_gpu, m, w->output,
                DS4_N_EMBD, DS4_N_VOCAB, rt->target_hidden, n_tokens));
            HY3_BATCH_STEP("output.argmax_all.batch", ds4_gpu_argmax_f32_tensor(
                rt->spec_tops_gpu, rt->spec_logits_gpu,
                DS4_N_VOCAB, n_tokens) != 0);
        } else if (need_logits) {
            HY3_BATCH_STEP("output.last.batch", ds4_gpu_copy_f32_slice(
                rt->target_hidden, (n_tokens - 1u) * DS4_N_EMBD,
                rt->last, DS4_N_EMBD) != 0);
            HY3_BATCH_STEP("output.logits.batch", hy3_matmul(
                rt->logits_gpu, m, w->output,
                DS4_N_EMBD, DS4_N_VOCAB, rt->last));
        }
        if (mtp_catchup && ok) {
            ok = hy3_mtp_encode_rows(s, tokens, n_tokens, pos0,
                                     rt->target_pending, rt->target_hidden,
                                     need_logits, &stage);
        }
        if (mtp_catchup) {
            HY3_BATCH_STEP("mtp.target_pending.batch", ds4_gpu_tensor_copy(
                rt->target_pending, 0, rt->target_hidden,
                (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * sizeof(float),
                (uint64_t)DS4_N_EMBD * sizeof(float)) != 0);
        }
    } else if (need_logits) {
        HY3_BATCH_STEP("output.last.batch", ds4_gpu_copy_f32_slice(
            rt->cur, (n_tokens - 1u) * DS4_N_EMBD, rt->last, DS4_N_EMBD) != 0);
        HY3_BATCH_STEP("output.rms.batch", ds4_gpu_rms_norm_weight_tensor(
            rt->norm, rt->last, m->map, m->size, w->output_norm->abs_offset,
            DS4_N_EMBD, DS4_RMS_EPS) != 0);
        HY3_BATCH_STEP("output.logits.batch", hy3_matmul(
            rt->logits_gpu, m, w->output, DS4_N_EMBD, DS4_N_VOCAB, rt->norm));
    }
#undef HY3_BATCH_STEP
    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "HY3 batched prefill failed at %s", stage);
        s->checkpoint_valid = false;
        return 1;
    }
    if (need_logits && !ds4_gpu_tensor_read(
            rt->logits_gpu, 0, s->logits,
            (uint64_t)DS4_N_VOCAB * sizeof(float))) {
        snprintf(err, errlen, "HY3 failed to read batched prefill logits");
        s->checkpoint_valid = false;
        return 1;
    }
    s->mtp_draft_valid = false;
    if (mtp_catchup && need_logits && e->mtp_ready) {
        int32_t top = -1;
        if (!ds4_gpu_tensor_read(rt->mtp_tops_gpu, 0, &top, sizeof(top))) {
            snprintf(err, errlen, "HY3 failed to read batched MTP argmax");
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = top;
        s->mtp_draft_valid = top >= 0 && top < (int32_t)DS4_N_VOCAB;
    }
    if (all_logits && !ds4_gpu_tensor_read(
            rt->spec_tops_gpu, 0, row_tops,
            (uint64_t)n_tokens * sizeof(int32_t))) {
        snprintf(err, errlen, "HY3 failed to read verifier row argmax");
        s->checkpoint_valid = false;
        return 1;
    }
    if (commit_checkpoint) {
        rt->n_past += n_tokens;
        if (mtp_catchup && e->mtp_ready) {
            rt->mtp_n_past = rt->n_past;
            rt->mtp_cache_valid = true;
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            token_vec_push(&s->checkpoint, tokens[i]);
        }
        /* Intermediate chunks intentionally skip the vocabulary projection,
         * so they do not yet form a self-contained snapshot. */
        s->checkpoint_valid = need_logits;
    }
    return 0;
}

static int hy3_eval_batch(ds4_session *s, const int *tokens, uint32_t n_tokens,
                          bool need_logits, char *err, size_t errlen) {
    return hy3_eval_batch_ex(s, tokens, n_tokens, need_logits,
                             hy3_mtp_runtime_active(s),
                             true, false, NULL, err, errlen);
}

static void hy3_session_reset(ds4_session *s) {
    hy3_runtime *rt = hy3_rt(s);
    if (!rt) return;
    rt->n_past = 0;
    rt->mtp_n_past = 0;
    rt->mtp_cache_valid = false;
    rt->mtp_auto_disabled = hy3_mtp_should_start_target_only(s);
    rt->mtp_auto_logged = false;
    if (rt->target_pending) {
        (void)ds4_gpu_tensor_fill_f32(rt->target_pending, 0.0f, DS4_N_EMBD);
    }
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
}

static int hy3_session_sync(ds4_session *s, const ds4_tokens *prompt,
                            char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    int start = 0;
    if (s->checkpoint_valid && prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = s->checkpoint.len;
    } else {
        hy3_session_reset(s);
    }
    const bool batch_enabled = s->prefill_cap > 1u &&
                               getenv("DS4_HY3_DISABLE_BATCH_PREFILL") == NULL &&
                               getenv("DS4_HY3_DISABLE_FLASH_ATTN") == NULL;
    const uint32_t sync_tokens = (uint32_t)(prompt->len - start);
    const uint32_t token_major_max = hy3_token_major_max_sync_requested();
    const bool token_major_sync = batch_enabled && token_major_max != 0u &&
                                  sync_tokens <= token_major_max;
    for (int i = start; i < prompt->len;) {
        const int remaining = prompt->len - i;
        uint32_t n = batch_enabled && !token_major_sync && remaining >= 2
            ? ((uint32_t)remaining < s->prefill_cap
                   ? (uint32_t)remaining : s->prefill_cap)
            : 1u;
        /* Preserve the qualified 256-row tail geometry across wider outer
         * batches.  Without this split, e.g. a 3898-token prompt ends in 314
         * rows at cap 512 instead of the baseline 256 + 58, changing row-batch
         * matmul arithmetic even though attention itself is striped. */
        if (n > DS4_HY3_PREFILL_ATTN_STRIPE &&
            (uint32_t)remaining <= s->prefill_cap) {
            const uint32_t tail = n % DS4_HY3_PREFILL_ATTN_STRIPE;
            if (tail != 0u) n -= tail;
        }
        const bool need_logits = i + (int)n == prompt->len;
        const int rc = n == 1u
            ? hy3_eval_token_ex(s, prompt->v[i],
                                need_logits && hy3_mtp_runtime_active(s),
                                err, errlen)
            : hy3_eval_batch(s, prompt->v + i, n, need_logits, err, errlen);
        if (rc != 0) return 1;
        i += (int)n;
        if (s->progress) s->progress(s->progress_ud, "prefill_token", i, prompt->len);
    }
    return 0;
}

static int hy3_session_eval(ds4_session *s, int token, bool probe_mtp,
                            char *err, size_t errlen) {
    return hy3_eval_token_ex(s, token, probe_mtp, err, errlen);
}

static int hy3_mtp_recursive_draft(ds4_session *s, int token, int *top,
                                   char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    if (!s || !rt || !hy3_mtp_runtime_active(s) || !top) return 1;
    const char *stage = "mtp.begin";
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = hy3_mtp_encode_rows(s, &token, 1u, rt->mtp_n_past,
                                 rt->mtp_carry, NULL, true, &stage);
    }
    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "HY3 recursive MTP draft failed at %s", stage);
        return 1;
    }
    int32_t id = -1;
    if (!ds4_gpu_tensor_read(rt->mtp_tops_gpu, 0, &id, sizeof(id)) ||
        id < 0 || id >= (int32_t)DS4_N_VOCAB) {
        snprintf(err, errlen, "HY3 recursive MTP argmax read failed");
        return 1;
    }
    rt->mtp_n_past++;
    *top = id;
    return 0;
}

static int hy3_mtp_verify_batch(
        ds4_session *s, const int *drafts, int draft_n, int eos_token,
        int *accepted, int *n_accept, int accepted_cap, int *verified_out,
        char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    if (!s || !rt || !hy3_mtp_runtime_active(s) || !drafts || draft_n < 2 ||
        draft_n > (int)HY3_MTP_MAX_DRAFT || !accepted || !n_accept) {
        return 1;
    }
    const uint32_t start = rt->n_past;
    int row_tops[HY3_MTP_MAX_DRAFT] = {0};
    const bool profile_owned = env_flag_enabled("DS4_HY3_MTP_PROFILE") &&
                               getenv("DS4_METAL_MOE_STAGE_PROFILE") == NULL;
    if (profile_owned) (void)setenv("DS4_METAL_MOE_STAGE_PROFILE", "1", 1);
    if (hy3_eval_batch_ex(s, drafts, (uint32_t)draft_n,
                          false, false, false, true, row_tops,
                          err, errlen) != 0) {
        if (profile_owned) (void)unsetenv("DS4_METAL_MOE_STAGE_PROFILE");
        return 1;
    }

    int verified = 1; /* draft[0] was checked against the committed logits. */
    for (int i = 1; i < draft_n; ++i) {
        if (row_tops[i - 1] != drafts[i]) break;
        verified++;
        if (drafts[i] == eos_token) break;
    }
    if (verified <= 0) {
        if (profile_owned) (void)unsetenv("DS4_METAL_MOE_STAGE_PROFILE");
        return 1;
    }

    const uint64_t logits_row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    if (!ds4_gpu_tensor_read(rt->spec_logits_gpu,
                             (uint64_t)(verified - 1) * logits_row_bytes,
                             s->logits, logits_row_bytes)) {
        snprintf(err, errlen, "HY3 MTP failed to read committed verifier logits");
        if (profile_owned) (void)unsetenv("DS4_METAL_MOE_STAGE_PROFILE");
        return 1;
    }

    /* Rewrite only the accepted MTP suffix using target hidden carry.  Draft
     * rows beyond it stay physically present but become unreachable when the
     * logical cursor is lowered to the committed boundary. */
    const char *stage = "mtp.repair.begin";
    rt->mtp_n_past = start;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = hy3_mtp_encode_rows(s, drafts, (uint32_t)verified, start,
                                 rt->target_pending, rt->target_hidden,
                                 true, &stage);
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(
            rt->target_pending, 0, rt->target_hidden,
            (uint64_t)(verified - 1) * DS4_N_EMBD * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        if (!ok) stage = "mtp.repair.pending";
    }
    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "HY3 MTP repair failed at %s", stage);
        s->checkpoint_valid = false;
        if (profile_owned) (void)unsetenv("DS4_METAL_MOE_STAGE_PROFILE");
        return 1;
    }
    if (profile_owned) (void)unsetenv("DS4_METAL_MOE_STAGE_PROFILE");

    rt->n_past = start + (uint32_t)verified;
    rt->mtp_n_past = rt->n_past;
    rt->mtp_cache_valid = true;
    int32_t mtp_top = -1;
    if (!ds4_gpu_tensor_read(rt->mtp_tops_gpu, 0, &mtp_top, sizeof(mtp_top))) {
        snprintf(err, errlen, "HY3 MTP repair argmax read failed");
        s->checkpoint_valid = false;
        return 1;
    }
    for (int i = 0; i < verified && *n_accept < accepted_cap; ++i) {
        token_vec_push(&s->checkpoint, drafts[i]);
        accepted[(*n_accept)++] = drafts[i];
        if (drafts[i] == eos_token) break;
    }
    s->checkpoint_valid = true;
    s->mtp_draft_token = mtp_top;
    s->mtp_draft_valid = mtp_top >= 0 && mtp_top < (int32_t)DS4_N_VOCAB;
    if (verified_out) *verified_out = verified;
    return 0;
}

/* Lossless reference verifier.  This path deliberately uses the ordinary
 * one-token target decode for every accepted draft, so temp=0 output is the
 * exact HY3 target stream.  The layer-major verifier below is enabled only
 * after metrology confirms its row reductions are invariant on the host. */
static int hy3_session_eval_speculative_argmax(
        ds4_session *s, int first_token, int max_tokens, int eos_token,
        int *accepted, int accepted_cap, int *drafted, int *draft_accepted,
        char *err, size_t errlen) {
    if (drafted) *drafted = 0;
    if (draft_accepted) *draft_accepted = 0;
    if (!s || !accepted || accepted_cap <= 0 || max_tokens <= 0) return 0;

    hy3_runtime *rt = hy3_rt(s);
    ds4_engine *e = s->engine;
    const bool timing = getenv("DS4_MTP_TIMING") != NULL;
    const double t0 = timing ? now_sec() : 0.0;
    const bool batch_flag =
        env_flag_enabled("DS4_HY3_MTP_BATCH_VERIFY");
    const bool batch_requested = hy3_mtp_unsafe_batch_requested(s);
    if (batch_flag && !batch_requested) {
        static bool warned_batch_rejected;
        if (!warned_batch_rejected) {
            warned_batch_rejected = true;
            fprintf(stderr,
                    "ds4: HY3 MTP carried batch verifier is disabled: "
                    "its target state is not one-token-decode invariant; "
                    "using the safe route (audit only: also set "
                    "DS4_HY3_MTP_UNSAFE_BATCH_VERIFY=1)\n");
        }
    }

    /* The one-token reference verifier evaluates every emitted token with the
     * complete target model, so MTP can only add work.  Default to a one-way
     * target-only route unless the user explicitly forces reference MTP or the
     * layer-major verifier experiment.  This is session-local: no process-wide
     * environment mutation and no stale MTP maintenance after the switch. */
    if (rt->mtp_auto_disabled ||
        (hy3_mtp_auto_fallback_enabled() && !batch_requested)) {
        rt->mtp_auto_disabled = true;
        s->mtp_draft_valid = false;
        if (!rt->mtp_auto_logged) {
            rt->mtp_auto_logged = true;
            fprintf(stderr,
                    "ds4: HY3 MTP auto-fallback -> plain target decode "
                    "(strict sequential verification cannot reduce target rows; "
                    "set DS4_HY3_MTP_AUTO_FALLBACK=0 to force the reference path)\n");
        }
        if (hy3_session_eval(s, first_token, false, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }

    /* Canonical speculative schedule: carry the predictor made at the prior
     * committed frontier, validate it against the already-available target
     * logits, recursively extend it, then verify [first_token + suffix] in one
     * target pass.  The older reference schedule below evaluates first_token
     * before drafting and therefore cannot fold that row into the verifier.
     * Keep it as the exact one-token diagnostic when batch verification is
     * disabled or unavailable. */
    const bool batch_verify = batch_requested &&
        hy3_mtp_runtime_active(s) && e->mtp_draft_tokens > 1;
    if (batch_verify) {
        if (!s->mtp_draft_valid) {
            if (hy3_session_eval(s, first_token, true, err, errlen) != 0) return -1;
            accepted[0] = first_token;
            return 1;
        }

        const int prior_draft = s->mtp_draft_token;
        s->mtp_draft_valid = false;
        if (prior_draft != first_token) {
            if (hy3_session_eval(s, first_token, true, err, errlen) != 0) return -1;
            accepted[0] = first_token;
            if (drafted) *drafted = 1;
            if (draft_accepted) *draft_accepted = 0;
            rt->mtp_cycles++;
            rt->mtp_proposed++;
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: HY3 MTP carried first miss draft=%d target=%d pos=%u\n",
                        prior_draft, first_token, rt->n_past);
            }
            return 1;
        }

        int cap = e->mtp_draft_tokens;
        if (cap > (int)HY3_MTP_MAX_DRAFT) cap = (int)HY3_MTP_MAX_DRAFT;
        if (cap > max_tokens) cap = max_tokens;
        if (cap > accepted_cap) cap = accepted_cap;
        const int room = s->ctx_size - (int)rt->n_past;
        if (cap > room) cap = room;

        int drafts[HY3_MTP_MAX_DRAFT] = {0};
        drafts[0] = first_token;
        int draft_n = 1;
        const double draft_t0 = timing ? now_sec() : 0.0;
        while (draft_n < cap && drafts[draft_n - 1] != eos_token) {
            if (hy3_mtp_recursive_draft(s, drafts[draft_n - 1],
                                        &drafts[draft_n], err, errlen) != 0) {
                break;
            }
            draft_n++;
        }
        const double draft_done = timing ? now_sec() : 0.0;
        if (draft_n >= 2) {
            int n_accept = 0;
            int verified = 0;
            const double verify_t0 = timing ? now_sec() : 0.0;
            if (hy3_mtp_verify_batch(s, drafts, draft_n, eos_token,
                                     accepted, &n_accept, accepted_cap,
                                     &verified, err, errlen) != 0) {
                return -1;
            }
            const double done = timing ? now_sec() : 0.0;
            if (drafted) *drafted = draft_n;
            if (draft_accepted) *draft_accepted = verified;
            rt->mtp_cycles++;
            rt->mtp_proposed += (uint64_t)draft_n;
            rt->mtp_accepted += (uint64_t)verified;
            if (timing) {
                rt->mtp_draft_seconds += draft_done - draft_t0;
                rt->mtp_verify_seconds += done - verify_t0;
                fprintf(stderr,
                        "ds4: HY3 MTP timing carried proposed=%d accepted=%d "
                        "draft=%.3f ms verify+repair=%.3f ms total=%.3f ms\n",
                        draft_n, verified,
                        (draft_done - draft_t0) * 1000.0,
                        (done - verify_t0) * 1000.0,
                        (done - t0) * 1000.0);
            }
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: HY3 MTP carried proposed=%d accepted=%d pos=%u\n",
                        draft_n, verified, rt->n_past);
            }
            return n_accept;
        }

        /* Context/tail cap left no row to batch.  No recursive row was
         * committed, so the target and MTP frontiers are still aligned. */
        if (hy3_session_eval(s, first_token, true, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        if (drafted) *drafted = 1;
        if (draft_accepted) *draft_accepted = 1;
        rt->mtp_cycles++;
        rt->mtp_proposed++;
        rt->mtp_accepted++;
        return 1;
    }

    if (hy3_session_eval(s, first_token, true, err, errlen) != 0) return -1;
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap ||
        !e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1) {
        return n_accept;
    }

    int cap = e->mtp_draft_tokens;
    if (cap > (int)HY3_MTP_MAX_DRAFT) cap = (int)HY3_MTP_MAX_DRAFT;
    if (cap > max_tokens - n_accept) cap = max_tokens - n_accept;
    if (cap > accepted_cap - n_accept) cap = accepted_cap - n_accept;
    const int room = s->ctx_size - (int)rt->n_past;
    if (cap > room) cap = room;
    if (cap <= 0) return n_accept;

    int drafts[HY3_MTP_MAX_DRAFT] = {0};
    int draft_n = 1;
    drafts[0] = s->mtp_draft_token;
    s->mtp_draft_valid = false;
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (drafted) *drafted = 1;
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: HY3 MTP first miss draft=%d target=%d\n",
                    drafts[0], sample_argmax(s->logits, DS4_N_VOCAB));
        }
        return n_accept;
    }

    const double draft_t0 = timing ? now_sec() : 0.0;
    while (draft_n < cap && drafts[draft_n - 1] != eos_token) {
        if (hy3_mtp_recursive_draft(s, drafts[draft_n - 1],
                                    &drafts[draft_n], err, errlen) != 0) {
            break;
        }
        draft_n++;
    }
    const double draft_done = timing ? now_sec() : 0.0;

    if (draft_n >= 2 && batch_requested) {
        int verified = 0;
        const double verify_t0 = timing ? now_sec() : 0.0;
        if (hy3_mtp_verify_batch(s, drafts, draft_n, eos_token,
                                 accepted, &n_accept, accepted_cap, &verified,
                                 err, errlen) != 0) {
            return -1;
        }
        const double done = timing ? now_sec() : 0.0;
        if (drafted) *drafted = draft_n;
        if (draft_accepted) *draft_accepted = verified;
        rt->mtp_cycles++;
        rt->mtp_proposed += (uint64_t)draft_n;
        rt->mtp_accepted += (uint64_t)verified;
        if (timing) {
            rt->mtp_draft_seconds += draft_done - draft_t0;
            rt->mtp_verify_seconds += done - verify_t0;
            fprintf(stderr,
                    "ds4: HY3 MTP timing batch proposed=%d accepted=%d "
                    "draft=%.3f ms verify+repair=%.3f ms total=%.3f ms\n",
                    draft_n, verified,
                    (draft_done - draft_t0) * 1000.0,
                    (done - verify_t0) * 1000.0,
                    (done - t0) * 1000.0);
        }
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: HY3 MTP batch proposed=%d accepted=%d total=%d pos=%u\n",
                    draft_n, verified, n_accept, rt->n_past);
        }
        return n_accept;
    }

    /* Recursive proposal rows used MTP hidden carry.  Rewind its logical
     * frontier to the exact row produced alongside first_token; accepted rows
     * below overwrite future slots using target hidden carry. */
    rt->mtp_n_past = rt->n_past;
    int verified = 0;
    const double verify_t0 = timing ? now_sec() : 0.0;
    for (int i = 0; i < draft_n && n_accept < accepted_cap; ++i) {
        const int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
        if (target_top != drafts[i]) break;
        if (hy3_eval_token_ex(s, drafts[i], false, err, errlen) != 0) return -1;
        accepted[n_accept++] = drafts[i];
        verified++;
        if (drafts[i] == eos_token) break;
    }
    const double done = timing ? now_sec() : 0.0;
    if (drafted) *drafted = draft_n;
    if (draft_accepted) *draft_accepted = verified;
    rt->mtp_cycles++;
    rt->mtp_proposed += (uint64_t)draft_n;
    rt->mtp_accepted += (uint64_t)verified;
    if (timing) {
        rt->mtp_draft_seconds += draft_done - draft_t0;
        rt->mtp_verify_seconds += done - verify_t0;
        fprintf(stderr,
                "ds4: HY3 MTP timing seq proposed=%d accepted=%d "
                "draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                draft_n, verified,
                (draft_done - draft_t0) * 1000.0,
                (done - verify_t0) * 1000.0,
                (done - t0) * 1000.0);
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        fprintf(stderr,
                "ds4: HY3 MTP seq proposed=%d accepted=%d total=%d pos=%u\n",
                draft_n, verified, n_accept, rt->n_past);
    }
    return n_accept;
}

#ifndef DS4_NO_GPU
#define HY3_SESSION_PAYLOAD_MAGIC UINT32_C(0x31335948) /* "HY31" */
#define HY3_SESSION_PAYLOAD_VERSION_Q8 UINT32_C(4)
#define HY3_SESSION_PAYLOAD_VERSION_F16 UINT32_C(5)
#define HY3_SESSION_PAYLOAD_VERSION_MTP UINT32_C(6)
/* Base layout (13 u32) plus four u64 identity fields for the target and four
 * for the optional MTP support model: dev, inode, size, and mtime_ns. */
#define HY3_SESSION_PAYLOAD_U32_FIELDS 29u
#define HY3_SESSION_IO_CHUNK (8u * 1024u * 1024u)
#define HY3_Q8_0_BLOCK_VALUES 32u
#define HY3_Q8_0_BLOCK_BYTES 34u
#define HY3_KV_FORMAT_F16 1u
#define HY3_KV_LAYOUT_HEAD_MAJOR 1u

static void hy3_payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void hy3_payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)v;
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t hy3_payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static uint32_t hy3_payload_u64_lo(uint64_t v) {
    return (uint32_t)v;
}

static uint32_t hy3_payload_u64_hi(uint64_t v) {
    return (uint32_t)(v >> 32);
}

static uint64_t hy3_payload_header_u64(const uint32_t *h, uint32_t i) {
    return (uint64_t)h[i] | ((uint64_t)h[i + 1u] << 32);
}

static bool hy3_payload_model_identity_matches(
        const ds4_model *m, const uint32_t *h, uint32_t i) {
    return m &&
           hy3_payload_header_u64(h, i) == m->file_dev &&
           hy3_payload_header_u64(h, i + 2u) == m->file_ino &&
           hy3_payload_header_u64(h, i + 4u) == m->size &&
           hy3_payload_header_u64(h, i + 6u) == m->file_mtime_ns;
}

static int hy3_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                                   char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            hy3_payload_set_err(err, errlen, "failed to write HY3 session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int hy3_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                                  uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        hy3_payload_set_err(err, errlen, "truncated HY3 session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            hy3_payload_set_err(err, errlen, "failed to read HY3 session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int hy3_payload_write_u32(FILE *fp, uint32_t v,
                                 char *err, size_t errlen) {
    uint8_t b[4];
    hy3_payload_put_u32(b, v);
    return hy3_payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int hy3_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                                char *err, size_t errlen) {
    uint8_t b[4];
    if (hy3_payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) {
        return 1;
    }
    *v = hy3_payload_get_u32(b);
    return 0;
}

static int hy3_payload_write_tensor_range(FILE *fp,
                                          const ds4_gpu_tensor *tensor,
                                          uint64_t tensor_offset,
                                          uint64_t bytes, uint8_t *buf,
                                          char *err, size_t errlen) {
    if (!tensor || tensor_offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - tensor_offset) {
        hy3_payload_set_err(err, errlen, "HY3 session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > HY3_SESSION_IO_CHUNK ?
                         HY3_SESSION_IO_CHUNK : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, tensor_offset + done, buf, n) == 0) {
            hy3_payload_set_err(err, errlen, "failed to read HY3 Metal session tensor");
            return 1;
        }
        if (hy3_payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int hy3_payload_read_tensor_range(FILE *fp, ds4_gpu_tensor *tensor,
                                         uint64_t tensor_offset,
                                         uint64_t bytes, uint8_t *buf,
                                         uint64_t *remaining,
                                         char *err, size_t errlen) {
    if (!tensor || tensor_offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - tensor_offset) {
        hy3_payload_set_err(err, errlen, "HY3 session tensor is smaller than the payload");
        return 1;
    }
    uint8_t *dst = ds4_gpu_tensor_contents(tensor);
    if (!dst && bytes != 0) {
        hy3_payload_set_err(err, errlen, "HY3 session tensor is not CPU-visible");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > HY3_SESSION_IO_CHUNK ?
                         HY3_SESSION_IO_CHUNK : (size_t)(bytes - done);
        if (hy3_payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        /* HY3 is Metal-only and these runtime tensors use Shared storage on
         * Apple silicon.  Write the mapped buffer directly: didModifyRange is
         * a Managed-storage operation, is deprecated/no-op here, and issuing
         * it once per layer made a 154 MiB prompt cache much slower to load. */
        memcpy(dst + tensor_offset + done, buf, n);
        done += n;
    }
    return 0;
}

static uint64_t hy3_payload_kv_row_bytes(bool f16_head_major) {
    return f16_head_major
        ? (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * sizeof(uint16_t)
        : (uint64_t)DS4_N_HEAD_KV *
              (DS4_N_HEAD_DIM / HY3_Q8_0_BLOCK_VALUES) * HY3_Q8_0_BLOCK_BYTES;
}

static uint64_t hy3_payload_bytes_for_tokens(uint32_t n_tokens,
                                             bool f16_head_major,
                                             bool with_mtp) {
    uint64_t bytes = (uint64_t)HY3_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)n_tokens * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * 2u * n_tokens *
             hy3_payload_kv_row_bytes(f16_head_major);
    if (with_mtp) {
        bytes += 2u * (uint64_t)n_tokens *
                 hy3_payload_kv_row_bytes(f16_head_major);
        bytes += (uint64_t)DS4_N_EMBD * sizeof(float);
    }
    return bytes;
}

static uint64_t hy3_session_payload_bytes(ds4_session *s) {
    hy3_runtime *rt = hy3_rt(s);
    if (!s || !rt || !s->checkpoint_valid || s->checkpoint.len < 0 ||
        rt->n_past != (uint32_t)s->checkpoint.len) {
        return 0;
    }
    /* Once the M5 auto-router selects plain target decode, block-80 state is
     * deliberately no longer maintained.  Persist the still-valid target KV
     * as a normal Q8/F16 HY3 payload instead of making /save fail merely
     * because the engine also has an MTP sidecar open. */
    const bool with_mtp = hy3_mtp_runtime_active(s);
    if (with_mtp &&
        (!rt->mtp_cache_valid || rt->mtp_n_past != rt->n_past ||
         !rt->mtp_k || !rt->mtp_v || !rt->target_pending ||
         env_flag_enabled("DS4_HY3_MTP_ZERO_POS0_EMBED"))) {
        return 0;
    }
    return hy3_payload_bytes_for_tokens(rt->n_past, rt->nax_f16_kv,
                                        with_mtp);
}

static int hy3_session_save_payload(ds4_session *s, FILE *fp,
                                    char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    if (!s || !rt || !fp || !s->checkpoint_valid) {
        hy3_payload_set_err(err, errlen, "HY3 session has no valid checkpoint to save");
        return 1;
    }
    if (s->checkpoint.len < 0 || rt->n_past != (uint32_t)s->checkpoint.len ||
        rt->n_past > (uint32_t)s->ctx_size) {
        hy3_payload_set_err(err, errlen, "HY3 KV row count does not match checkpoint");
        return 1;
    }
    const bool with_mtp = hy3_mtp_runtime_active(s);
    if (with_mtp &&
        (!rt->mtp_cache_valid || rt->mtp_n_past != rt->n_past ||
         !rt->mtp_k || !rt->mtp_v || !rt->target_pending ||
         env_flag_enabled("DS4_HY3_MTP_ZERO_POS0_EMBED"))) {
        hy3_payload_set_err(err, errlen,
                            "HY3 MTP checkpoint is not at a persistable frontier");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        hy3_payload_set_err(err, errlen, "failed to synchronize Metal before HY3 snapshot");
        return 1;
    }

    const ds4_model *target_model = &s->engine->model;
    const ds4_model *mtp_model = with_mtp ? &s->engine->mtp_model : NULL;
    const uint32_t header[HY3_SESSION_PAYLOAD_U32_FIELDS] = {
        HY3_SESSION_PAYLOAD_MAGIC,
        with_mtp ? HY3_SESSION_PAYLOAD_VERSION_MTP :
            (rt->nax_f16_kv ? HY3_SESSION_PAYLOAD_VERSION_F16 :
                              HY3_SESSION_PAYLOAD_VERSION_Q8),
        (uint32_t)s->ctx_size,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD,
        DS4_N_HEAD_KV,
        DS4_N_HEAD_DIM,
        DS4_N_VOCAB,
        rt->n_past,
        DS4_N_EMBD,
        rt->nax_f16_kv ? HY3_KV_FORMAT_F16 : HY3_Q8_0_BLOCK_VALUES,
        rt->nax_f16_kv ? HY3_KV_LAYOUT_HEAD_MAJOR : HY3_Q8_0_BLOCK_BYTES,
        hy3_payload_u64_lo(target_model->file_dev),
        hy3_payload_u64_hi(target_model->file_dev),
        hy3_payload_u64_lo(target_model->file_ino),
        hy3_payload_u64_hi(target_model->file_ino),
        hy3_payload_u64_lo(target_model->size),
        hy3_payload_u64_hi(target_model->size),
        hy3_payload_u64_lo(target_model->file_mtime_ns),
        hy3_payload_u64_hi(target_model->file_mtime_ns),
        hy3_payload_u64_lo(mtp_model ? mtp_model->file_dev : 0u),
        hy3_payload_u64_hi(mtp_model ? mtp_model->file_dev : 0u),
        hy3_payload_u64_lo(mtp_model ? mtp_model->file_ino : 0u),
        hy3_payload_u64_hi(mtp_model ? mtp_model->file_ino : 0u),
        hy3_payload_u64_lo(mtp_model ? mtp_model->size : 0u),
        hy3_payload_u64_hi(mtp_model ? mtp_model->size : 0u),
        hy3_payload_u64_lo(mtp_model ? mtp_model->file_mtime_ns : 0u),
        hy3_payload_u64_hi(mtp_model ? mtp_model->file_mtime_ns : 0u),
    };
    for (uint32_t i = 0; i < HY3_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (hy3_payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        const int token = s->checkpoint.v[i];
        if (token < 0 || (uint32_t)token >= DS4_N_VOCAB) {
            hy3_payload_set_err(err, errlen, "HY3 checkpoint contains an invalid token");
            return 1;
        }
        if (hy3_payload_write_u32(fp, (uint32_t)token, err, errlen) != 0) return 1;
    }
    if (hy3_payload_write_bytes(fp, s->logits,
                                (uint64_t)DS4_N_VOCAB * sizeof(float),
                                err, errlen) != 0) {
        return 1;
    }

    uint8_t *buf = xmalloc(HY3_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t q8_live_bytes =
        (uint64_t)rt->n_past * hy3_payload_kv_row_bytes(false);
    const uint64_t f16_head_bytes =
        (uint64_t)rt->n_past * DS4_N_HEAD_DIM * sizeof(uint16_t);
    const uint64_t f16_head_stride =
        (uint64_t)rt->kv_ctx_pad * DS4_N_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *kv[2] = {rt->layer_k[il], rt->layer_v[il]};
        for (uint32_t which = 0; rc == 0 && which < 2u; ++which) {
            if (!rt->nax_f16_kv) {
                rc = hy3_payload_write_tensor_range(
                    fp, kv[which], 0, q8_live_bytes, buf, err, errlen);
            } else {
                for (uint32_t ih = 0; rc == 0 && ih < DS4_N_HEAD_KV; ++ih) {
                    rc = hy3_payload_write_tensor_range(
                        fp, kv[which], (uint64_t)ih * f16_head_stride,
                        f16_head_bytes, buf, err, errlen);
                }
            }
        }
    }
    if (rc == 0 && with_mtp) {
        ds4_gpu_tensor *mtp_kv[2] = {rt->mtp_k, rt->mtp_v};
        for (uint32_t which = 0; rc == 0 && which < 2u; ++which) {
            if (!rt->nax_f16_kv) {
                rc = hy3_payload_write_tensor_range(
                    fp, mtp_kv[which], 0, q8_live_bytes, buf, err, errlen);
            } else {
                for (uint32_t ih = 0; rc == 0 && ih < DS4_N_HEAD_KV; ++ih) {
                    rc = hy3_payload_write_tensor_range(
                        fp, mtp_kv[which], (uint64_t)ih * f16_head_stride,
                        f16_head_bytes, buf, err, errlen);
                }
            }
        }
        if (rc == 0) {
            rc = hy3_payload_write_tensor_range(
                fp, rt->target_pending, 0,
                (uint64_t)DS4_N_EMBD * sizeof(float), buf, err, errlen);
        }
    }
    free(buf);
    return rc;
}

static int hy3_session_load_payload(ds4_session *s, FILE *fp,
                                    uint64_t payload_bytes,
                                    char *err, size_t errlen) {
    hy3_runtime *rt = hy3_rt(s);
    if (!s || !rt || !fp) {
        hy3_payload_set_err(err, errlen, "invalid HY3 session payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[HY3_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < HY3_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (hy3_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != HY3_SESSION_PAYLOAD_MAGIC ||
        (h[1] != HY3_SESSION_PAYLOAD_VERSION_Q8 &&
         h[1] != HY3_SESSION_PAYLOAD_VERSION_F16 &&
         h[1] != HY3_SESSION_PAYLOAD_VERSION_MTP)) {
        hy3_payload_set_err(err, errlen, "unsupported HY3 session payload version");
        return 1;
    }
    const bool saved_mtp = h[1] == HY3_SESSION_PAYLOAD_VERSION_MTP;
    const bool engine_mtp = s->engine && s->engine->mtp_ready;
    if (saved_mtp && !engine_mtp) {
        hy3_payload_set_err(err, errlen,
                            "HY3 MTP checkpoint requires an MTP sidecar");
        return 1;
    }
    if (!saved_mtp && hy3_mtp_active_state_requested(s)) {
        /* A target-only cache has neither block-80 KV nor the target-hidden
         * carry.  Silently accepting it would make AUTO_FALLBACK=0 (or the
         * unsafe verifier audit) run plain target decode forever.  Let the
         * caller invalidate this cache and rebuild the prompt with MTP. */
        hy3_payload_set_err(err, errlen,
                            "HY3 target-only checkpoint lacks requested MTP state");
        return 1;
    }
    if (!hy3_payload_model_identity_matches(&s->engine->model, h, 13u)) {
        hy3_payload_set_err(err, errlen,
                            "HY3 KV checkpoint was written for a different target GGUF");
        return 1;
    }
    if (saved_mtp &&
        !hy3_payload_model_identity_matches(&s->engine->mtp_model, h, 21u)) {
        hy3_payload_set_err(err, errlen,
                            "HY3 KV checkpoint was written for a different MTP GGUF");
        return 1;
    }
    if (saved_mtp && env_flag_enabled("DS4_HY3_MTP_ZERO_POS0_EMBED")) {
        hy3_payload_set_err(err, errlen,
                            "HY3 MTP checkpoint semantic-position mode differs");
        return 1;
    }

    const uint32_t saved_ctx = h[2];
    const uint32_t saved_tokens = h[3];
    const uint32_t saved_layers = h[4];
    const uint32_t saved_n_head = h[5];
    const uint32_t saved_n_head_kv = h[6];
    const uint32_t saved_head_dim = h[7];
    const uint32_t saved_vocab = h[8];
    const uint32_t saved_n_past = h[9];
    const uint32_t saved_embd = h[10];
    const uint32_t saved_kv_field0 = h[11];
    const uint32_t saved_kv_field1 = h[12];
    const bool saved_f16 = saved_mtp
        ? saved_kv_field0 == HY3_KV_FORMAT_F16 &&
          saved_kv_field1 == HY3_KV_LAYOUT_HEAD_MAJOR
        : h[1] == HY3_SESSION_PAYLOAD_VERSION_F16;
    if (saved_ctx == 0 || saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens > (uint32_t)s->ctx_size || saved_n_past != saved_tokens) {
        hy3_payload_set_err(err, errlen, "HY3 KV checkpoint does not fit current context");
        return 1;
    }
    const bool saved_layout_ok = saved_f16
        ? saved_kv_field0 == HY3_KV_FORMAT_F16 &&
          saved_kv_field1 == HY3_KV_LAYOUT_HEAD_MAJOR
        : saved_kv_field0 == HY3_Q8_0_BLOCK_VALUES &&
          saved_kv_field1 == HY3_Q8_0_BLOCK_BYTES;
    if (saved_layers != DS4_N_LAYER || saved_n_head != DS4_N_HEAD ||
        saved_n_head_kv != DS4_N_HEAD_KV || saved_head_dim != DS4_N_HEAD_DIM ||
        saved_vocab != DS4_N_VOCAB || saved_embd != DS4_N_EMBD ||
        !saved_layout_ok) {
        hy3_payload_set_err(err, errlen, "HY3 KV checkpoint was written for a different layout");
        return 1;
    }
    if (saved_f16 != rt->nax_f16_kv) {
        hy3_payload_set_err(err, errlen,
                            "HY3 KV checkpoint cache format does not match this session");
        return 1;
    }
    if (payload_bytes != hy3_payload_bytes_for_tokens(saved_tokens, saved_f16,
                                                      saved_mtp)) {
        hy3_payload_set_err(err, errlen, "HY3 KV checkpoint payload size does not match its header");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (hy3_payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (tok >= DS4_N_VOCAB) {
            token_vec_free(&new_checkpoint);
            hy3_payload_set_err(err, errlen, "HY3 KV checkpoint contains an invalid token");
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }

    float *new_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (hy3_payload_read_bytes(fp, new_logits,
                               (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0) {
        free(new_logits);
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        free(new_logits);
        token_vec_free(&new_checkpoint);
        s->checkpoint_valid = false;
        hy3_payload_set_err(err, errlen, "failed to synchronize Metal before HY3 snapshot restore");
        return 1;
    }

    uint8_t *buf = xmalloc(HY3_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t q8_live_bytes =
        (uint64_t)saved_n_past * hy3_payload_kv_row_bytes(false);
    const uint64_t f16_head_bytes =
        (uint64_t)saved_n_past * DS4_N_HEAD_DIM * sizeof(uint16_t);
    const uint64_t f16_head_stride =
        (uint64_t)rt->kv_ctx_pad * DS4_N_HEAD_DIM * sizeof(uint16_t);
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *kv[2] = {rt->layer_k[il], rt->layer_v[il]};
        for (uint32_t which = 0; rc == 0 && which < 2u; ++which) {
            if (!saved_f16) {
                rc = hy3_payload_read_tensor_range(
                    fp, kv[which], 0, q8_live_bytes, buf, &remaining,
                    err, errlen);
            } else {
                for (uint32_t ih = 0; rc == 0 && ih < DS4_N_HEAD_KV; ++ih) {
                    rc = hy3_payload_read_tensor_range(
                        fp, kv[which], (uint64_t)ih * f16_head_stride,
                        f16_head_bytes, buf, &remaining, err, errlen);
                }
            }
        }
    }
    if (rc == 0 && saved_mtp) {
        ds4_gpu_tensor *mtp_kv[2] = {rt->mtp_k, rt->mtp_v};
        for (uint32_t which = 0; rc == 0 && which < 2u; ++which) {
            if (!saved_f16) {
                rc = hy3_payload_read_tensor_range(
                    fp, mtp_kv[which], 0, q8_live_bytes, buf, &remaining,
                    err, errlen);
            } else {
                for (uint32_t ih = 0; rc == 0 && ih < DS4_N_HEAD_KV; ++ih) {
                    rc = hy3_payload_read_tensor_range(
                        fp, mtp_kv[which], (uint64_t)ih * f16_head_stride,
                        f16_head_bytes, buf, &remaining, err, errlen);
                }
            }
        }
        if (rc == 0) {
            rc = hy3_payload_read_tensor_range(
                fp, rt->target_pending, 0,
                (uint64_t)DS4_N_EMBD * sizeof(float), buf, &remaining,
                err, errlen);
        }
    }
    free(buf);
    if (rc != 0 || remaining != 0) {
        free(new_logits);
        token_vec_free(&new_checkpoint);
        s->checkpoint_valid = false;
        if (rc == 0) {
            hy3_payload_set_err(err, errlen, "HY3 KV checkpoint has trailing payload bytes");
        }
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    memcpy(s->logits, new_logits, (size_t)DS4_N_VOCAB * sizeof(float));
    free(new_logits);
    rt->n_past = saved_n_past;
    if (saved_mtp) {
        rt->mtp_n_past = saved_n_past;
        rt->mtp_cache_valid = true;
        /* Loading a reference-run cache must not undo the measured safe
         * default.  The valid MTP payload may be retained for this load, but
         * it stays dormant and the next save naturally emits target-only KV. */
        rt->mtp_auto_disabled = hy3_mtp_should_start_target_only(s);
        rt->mtp_auto_logged = false;
    } else {
        rt->mtp_n_past = 0;
        rt->mtp_cache_valid = false;
        /* A target-only payload cannot reconstruct block-80 KV or its hidden
         * carry.  If this engine has a sidecar, keep the restored session on
         * the exact target route; /new or a full prompt rebuild starts a fresh
         * session in which MTP may be used again. */
        rt->mtp_auto_disabled = engine_mtp;
        rt->mtp_auto_logged = engine_mtp;
        if (engine_mtp) {
            fprintf(stderr,
                    "ds4: restored target-only HY3 KV; MTP is disabled for "
                    "this resumed session (start a new session to rebuild block80)\n");
        }
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    return 0;
}
#endif
