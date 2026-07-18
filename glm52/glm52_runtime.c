/* =========================================================================
 * glm52_runtime.c - GLM-5.2 session/runtime entry points.
 * =========================================================================
 *
 * Keep GLM-specific execution out of the DeepSeek V4 HC path.  This file is
 * included by ds4.c after ds4_session is defined, so it can allocate the same
 * public session object while avoiding DS4-only graph allocation.
 */

#define GLM52_N_EFFECTIVE_LAYER (DS4_N_LAYER - DS4_N_NEXTN)
#define GLM52_Q_NOPE_DIM        192u
#define GLM52_Q_HEAD_DIM        256u
#define GLM52_KV_LORA_DIM       512u
#define GLM52_K_PE_DIM          64u
#define GLM52_K_HEAD_DIM        576u
#define GLM52_KQ_SCALE_DIM      GLM52_Q_HEAD_DIM
#define GLM52_V_HEAD_DIM        512u
#define GLM52_V_IMPL_DIM        256u
#define GLM52_DENSE_FF_DIM      12288u

typedef struct {
    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *norm;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv_lora_raw;
    ds4_gpu_tensor *k_pe;
    ds4_gpu_tensor *kv_norm;
    ds4_gpu_tensor *q_abs;
    ds4_gpu_tensor *attn_scores;
    ds4_gpu_tensor *attn_lora;
    ds4_gpu_tensor *attn_heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *ffn_down;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *logits_gpu;
    ds4_gpu_tensor *batch_cur;
    ds4_gpu_tensor *batch_next;
    ds4_gpu_tensor *batch_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv_lora;
    ds4_gpu_tensor *batch_k_pe;
    ds4_gpu_tensor *batch_kv_norm;
    ds4_gpu_tensor *batch_q_abs;
    ds4_gpu_tensor *batch_attn_lora;
    ds4_gpu_tensor *batch_attn_heads;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_ffn_gate;
    ds4_gpu_tensor *batch_ffn_up;
    ds4_gpu_tensor *batch_ffn_mid;
    ds4_gpu_tensor *batch_ffn_down;
    ds4_gpu_tensor *layer_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_kpe[DS4_MAX_LAYER];
    float *embed_host;
    float *batch_embed_host;
    uint32_t batch_cap;
    uint32_t n_past;
} glm52_runtime;

static bool glm52_session_active(const ds4_session *s) {
    return s && s->engine && DS4_MODEL_VARIANT == DS4_VARIANT_GLM52;
}

static inline float glm52_attention_scale(void) {
    return 1.0f / sqrtf((float)GLM52_KQ_SCALE_DIM);
}

static glm52_runtime *glm52_rt(ds4_session *s) {
    return s ? (glm52_runtime *)s->variant_runtime : NULL;
}

static inline void glm52_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void glm52_dequantize_q4_K(const block_q4_K *x, float *y, uint64_t n) {
    if ((n % QK_K) != 0) ds4_die("GLM Q4_K embedding row is not QK_K aligned");
    const uint64_t nb = n / QK_K;
    for (uint64_t ib = 0; ib < nb; ib++) {
        const float d = f16_to_f32(x[ib].d);
        const float dmin = f16_to_f32(x[ib].dmin);
        const uint8_t *q = x[ib].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            glm52_get_scale_min_k4(is++, x[ib].scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = dmin * (float)m;
            glm52_get_scale_min_k4(is++, x[ib].scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = dmin * (float)m;
            for (int l = 0; l < 32; l++) y[j + l] = d1 * (float)(q[l] & 0x0fu) - m1;
            for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
        }
        y += QK_K;
    }
}

static bool glm52_embed_token(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    const ds4_tensor *te = w->token_embd;
    if (!m || !te || !out || token < 0 || (uint64_t)token >= te->dim[1]) return false;
    const uint64_t width = te->dim[0];
    const uint8_t *base = tensor_data(m, te);
    switch (te->type) {
    case DS4_TENSOR_Q4_K: {
        const uint64_t row_blocks = width / QK_K;
        const block_q4_K *row = (const block_q4_K *)(const void *)base +
                                (uint64_t)token * row_blocks;
        glm52_dequantize_q4_K(row, out, width);
        return true;
    }
    case DS4_TENSOR_F16: {
        const uint16_t *row = (const uint16_t *)(const void *)base + (uint64_t)token * width;
        for (uint64_t i = 0; i < width; i++) out[i] = f16_to_f32(row[i]);
        return true;
    }
    case DS4_TENSOR_F32: {
        const float *row = (const float *)(const void *)base + (uint64_t)token * width;
        memcpy(out, row, (size_t)width * sizeof(out[0]));
        return true;
    }
    default:
        fprintf(stderr,
                "ds4: unsupported GLM embedding tensor type %u (%s)\n",
                te->type,
                tensor_type_name(te->type));
        return false;
    }
}

static bool glm52_matmul(
        ds4_gpu_tensor       *out,
        const ds4_model      *m,
        const ds4_tensor     *w,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x) {
    return out && m && w && x &&
           ds4_gpu_matmul_gguf_tensor(out,
                                       m->map,
                                       m->size,
                                       w->abs_offset,
                                       w->type,
                                       in_dim,
                                       out_dim,
                                       x,
                                       1) != 0;
}

static bool glm52_matmul_rows(
        ds4_gpu_tensor       *out,
        const ds4_model      *m,
        const ds4_tensor     *w,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    return out && m && w && x && n_tokens != 0 &&
           ds4_gpu_matmul_gguf_tensor(out,
                                       m->map,
                                       m->size,
                                       w->abs_offset,
                                       w->type,
                                       in_dim,
                                       out_dim,
                                       x,
                                       n_tokens) != 0;
}

static void glm52_runtime_free_tensors(glm52_runtime *rt) {
    if (!rt) return;
#define GLM52_FREE_TENSOR(name) do { ds4_gpu_tensor_free(rt->name); rt->name = NULL; } while (0)
    GLM52_FREE_TENSOR(kv_lora_raw);
    GLM52_FREE_TENSOR(k_pe);
    GLM52_FREE_TENSOR(cur);
    GLM52_FREE_TENSOR(next);
    GLM52_FREE_TENSOR(norm);
    GLM52_FREE_TENSOR(qr);
    GLM52_FREE_TENSOR(qr_norm);
    GLM52_FREE_TENSOR(q);
    GLM52_FREE_TENSOR(kv_raw);
    GLM52_FREE_TENSOR(kv_norm);
    GLM52_FREE_TENSOR(q_abs);
    GLM52_FREE_TENSOR(attn_scores);
    GLM52_FREE_TENSOR(attn_lora);
    GLM52_FREE_TENSOR(attn_heads);
    GLM52_FREE_TENSOR(attn_out);
    GLM52_FREE_TENSOR(ffn_gate);
    GLM52_FREE_TENSOR(ffn_up);
    GLM52_FREE_TENSOR(ffn_mid);
    GLM52_FREE_TENSOR(ffn_down);
    GLM52_FREE_TENSOR(shared_gate);
    GLM52_FREE_TENSOR(shared_up);
    GLM52_FREE_TENSOR(shared_mid);
    GLM52_FREE_TENSOR(shared_out);
    GLM52_FREE_TENSOR(router_logits);
    GLM52_FREE_TENSOR(router_probs);
    GLM52_FREE_TENSOR(router_selected);
    GLM52_FREE_TENSOR(router_weights);
    GLM52_FREE_TENSOR(routed_gate);
    GLM52_FREE_TENSOR(routed_up);
    GLM52_FREE_TENSOR(routed_mid);
    GLM52_FREE_TENSOR(routed_down);
    GLM52_FREE_TENSOR(routed_out);
    GLM52_FREE_TENSOR(ffn_out);
    GLM52_FREE_TENSOR(logits_gpu);
    GLM52_FREE_TENSOR(batch_cur);
    GLM52_FREE_TENSOR(batch_next);
    GLM52_FREE_TENSOR(batch_norm);
    GLM52_FREE_TENSOR(batch_qr);
    GLM52_FREE_TENSOR(batch_qr_norm);
    GLM52_FREE_TENSOR(batch_q);
    GLM52_FREE_TENSOR(batch_kv_raw);
    GLM52_FREE_TENSOR(batch_kv_lora);
    GLM52_FREE_TENSOR(batch_k_pe);
    GLM52_FREE_TENSOR(batch_kv_norm);
    GLM52_FREE_TENSOR(batch_q_abs);
    GLM52_FREE_TENSOR(batch_attn_lora);
    GLM52_FREE_TENSOR(batch_attn_heads);
    GLM52_FREE_TENSOR(batch_attn_out);
    GLM52_FREE_TENSOR(batch_ffn_gate);
    GLM52_FREE_TENSOR(batch_ffn_up);
    GLM52_FREE_TENSOR(batch_ffn_mid);
    GLM52_FREE_TENSOR(batch_ffn_down);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(rt->layer_kv[il]);
        ds4_gpu_tensor_free(rt->layer_kpe[il]);
        rt->layer_kv[il] = NULL;
        rt->layer_kpe[il] = NULL;
    }
#undef GLM52_FREE_TENSOR
    free(rt->embed_host);
    rt->embed_host = NULL;
    free(rt->batch_embed_host);
    rt->batch_embed_host = NULL;
    rt->batch_cap = 0;
}

static void glm52_session_free(ds4_session *s) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt) return;
#ifndef DS4_NO_GPU
    s->graph.router_selected = NULL;
    s->graph.router_weights = NULL;
    s->graph.router_probs = NULL;
    s->graph.router_logits = NULL;
    s->graph.ffn_norm = NULL;
    s->graph.routed_gate = NULL;
    s->graph.routed_up = NULL;
    s->graph.routed_mid = NULL;
    s->graph.routed_down = NULL;
    s->graph.routed_out = NULL;
    metal_graph_free(&s->graph);
#endif
    glm52_runtime_free_tensors(rt);
    free(rt);
    s->variant_runtime = NULL;
}

static bool glm52_alloc_decode_tensors(ds4_session *s) {
    glm52_runtime *rt = glm52_rt(s);
    const uint64_t e = DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * GLM52_Q_HEAD_DIM;
    const uint64_t v_lora_dim = (uint64_t)DS4_N_HEAD * GLM52_V_HEAD_DIM;
    const uint64_t v_impl_dim = (uint64_t)DS4_N_HEAD * GLM52_V_IMPL_DIM;
    const uint64_t shared_dim = (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED;
    const uint64_t active = DS4_N_EXPERT_ACTIVE_USED;
    const uint32_t ctx = (uint32_t)s->ctx_size;
    const uint64_t pc = s->prefill_cap ? s->prefill_cap : 1u;

    rt->cur = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->next = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->norm = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->qr = ds4_gpu_tensor_alloc((uint64_t)DS4_N_LORA_Q * sizeof(float));
    rt->qr_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_LORA_Q * sizeof(float));
    rt->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    rt->kv_raw = ds4_gpu_tensor_alloc(GLM52_K_HEAD_DIM * sizeof(float));
    rt->kv_lora_raw = ds4_gpu_tensor_view(rt->kv_raw, 0, GLM52_KV_LORA_DIM * sizeof(float));
    rt->k_pe = ds4_gpu_tensor_view(rt->kv_raw,
                                   GLM52_KV_LORA_DIM * sizeof(float),
                                   GLM52_K_PE_DIM * sizeof(float));
    rt->kv_norm = ds4_gpu_tensor_alloc(GLM52_KV_LORA_DIM * sizeof(float));
    rt->q_abs = ds4_gpu_tensor_alloc(v_lora_dim * sizeof(float));
    rt->attn_scores = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD * ctx * sizeof(float));
    rt->attn_lora = ds4_gpu_tensor_alloc(v_lora_dim * sizeof(float));
    rt->attn_heads = ds4_gpu_tensor_alloc(v_impl_dim * sizeof(float));
    rt->attn_out = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->ffn_gate = ds4_gpu_tensor_alloc(GLM52_DENSE_FF_DIM * sizeof(float));
    rt->ffn_up = ds4_gpu_tensor_alloc(GLM52_DENSE_FF_DIM * sizeof(float));
    rt->ffn_mid = ds4_gpu_tensor_alloc(GLM52_DENSE_FF_DIM * sizeof(float));
    rt->ffn_down = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    rt->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    rt->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    rt->shared_out = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->router_logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT * sizeof(float));
    rt->router_probs = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT * sizeof(float));
    rt->router_selected = ds4_gpu_tensor_alloc(active * sizeof(int32_t));
    rt->router_weights = ds4_gpu_tensor_alloc(active * sizeof(float));
    rt->routed_gate = ds4_gpu_tensor_alloc(active * (uint64_t)DS4_N_FF_EXP * sizeof(float));
    rt->routed_up = ds4_gpu_tensor_alloc(active * (uint64_t)DS4_N_FF_EXP * sizeof(float));
    rt->routed_mid = ds4_gpu_tensor_alloc(active * (uint64_t)DS4_N_FF_EXP * sizeof(float));
    rt->routed_down = ds4_gpu_tensor_alloc(active * e * sizeof(float));
    rt->routed_out = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->ffn_out = ds4_gpu_tensor_alloc(e * sizeof(float));
    rt->logits_gpu = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    rt->batch_cur = ds4_gpu_tensor_alloc(pc * e * sizeof(float));
    rt->batch_next = ds4_gpu_tensor_alloc(pc * e * sizeof(float));
    rt->batch_norm = ds4_gpu_tensor_alloc(pc * e * sizeof(float));
    rt->batch_qr = ds4_gpu_tensor_alloc(pc * (uint64_t)DS4_N_LORA_Q * sizeof(float));
    rt->batch_qr_norm = ds4_gpu_tensor_alloc(pc * (uint64_t)DS4_N_LORA_Q * sizeof(float));
    rt->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    rt->batch_kv_raw = ds4_gpu_tensor_alloc(pc * GLM52_K_HEAD_DIM * sizeof(float));
    rt->batch_kv_lora = ds4_gpu_tensor_alloc(pc * GLM52_KV_LORA_DIM * sizeof(float));
    rt->batch_k_pe = ds4_gpu_tensor_alloc(pc * GLM52_K_PE_DIM * sizeof(float));
    rt->batch_kv_norm = ds4_gpu_tensor_alloc(pc * GLM52_KV_LORA_DIM * sizeof(float));
    rt->batch_q_abs = ds4_gpu_tensor_alloc(pc * v_lora_dim * sizeof(float));
    rt->batch_attn_lora = ds4_gpu_tensor_alloc(pc * v_lora_dim * sizeof(float));
    rt->batch_attn_heads = ds4_gpu_tensor_alloc(pc * v_impl_dim * sizeof(float));
    rt->batch_attn_out = ds4_gpu_tensor_alloc(pc * e * sizeof(float));
    rt->batch_ffn_gate = ds4_gpu_tensor_alloc(pc * GLM52_DENSE_FF_DIM * sizeof(float));
    rt->batch_ffn_up = ds4_gpu_tensor_alloc(pc * GLM52_DENSE_FF_DIM * sizeof(float));
    rt->batch_ffn_mid = ds4_gpu_tensor_alloc(pc * GLM52_DENSE_FF_DIM * sizeof(float));
    rt->batch_ffn_down = ds4_gpu_tensor_alloc(pc * e * sizeof(float));
    for (uint32_t il = 0; il < GLM52_N_EFFECTIVE_LAYER; il++) {
        rt->layer_kv[il] = ds4_gpu_tensor_alloc((uint64_t)ctx * GLM52_KV_LORA_DIM * sizeof(float));
        rt->layer_kpe[il] = ds4_gpu_tensor_alloc((uint64_t)ctx * GLM52_K_PE_DIM * sizeof(float));
    }
    rt->embed_host = xmalloc(e * sizeof(rt->embed_host[0]));
    rt->batch_embed_host = xmalloc((size_t)pc * e * sizeof(rt->batch_embed_host[0]));
    rt->batch_cap = (uint32_t)pc;

    bool ok = rt->cur && rt->next && rt->norm &&
              rt->qr && rt->qr_norm && rt->q &&
              rt->kv_raw && rt->kv_lora_raw && rt->k_pe && rt->kv_norm &&
              rt->q_abs && rt->attn_scores &&
              rt->attn_lora && rt->attn_heads && rt->attn_out &&
              rt->ffn_gate && rt->ffn_up && rt->ffn_mid && rt->ffn_down &&
              rt->shared_gate && rt->shared_up && rt->shared_mid && rt->shared_out &&
              rt->router_logits && rt->router_probs &&
              rt->router_selected && rt->router_weights &&
              rt->routed_gate && rt->routed_up && rt->routed_mid &&
              rt->routed_down && rt->routed_out && rt->ffn_out &&
              rt->logits_gpu &&
              rt->batch_cur && rt->batch_next && rt->batch_norm &&
              rt->batch_qr && rt->batch_qr_norm && rt->batch_q &&
              rt->batch_kv_raw && rt->batch_kv_lora && rt->batch_k_pe &&
              rt->batch_kv_norm && rt->batch_q_abs &&
              rt->batch_attn_lora && rt->batch_attn_heads && rt->batch_attn_out &&
              rt->batch_ffn_gate && rt->batch_ffn_up && rt->batch_ffn_mid &&
              rt->batch_ffn_down &&
              rt->embed_host && rt->batch_embed_host;
    for (uint32_t il = 0; ok && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        ok = rt->layer_kv[il] && rt->layer_kpe[il];
    }
    return ok;
}

static bool glm52_alloc_moe_batch_tensors(ds4_session *s) {
    if (!s || !s->engine) return false;
    ds4_gpu_graph *g = &s->graph;
    const ds4_layer_weights *layer = &s->engine->weights.layer[DS4_N_DENSE_LEAD];
    if (!layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps ||
        !layer->ffn_gate_shexp || !layer->ffn_up_shexp || !layer->ffn_down_shexp) {
        return false;
    }

    const uint64_t pc = s->prefill_cap ? s->prefill_cap : 1u;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    uint64_t moe_pc = g->batch_routed_scratch_cap;
    if (moe_pc == 0) {
        moe_pc = metal_graph_resident_moe_scratch_cap_for_prefill((uint32_t)pc);
        g->batch_routed_scratch_cap = (uint32_t)moe_pc;
    }

    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int32_t));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_compact_rows = (uint32_t)(moe_pc * DS4_N_EXPERT_USED);
    g->batch_routed_gate = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    const bool ok = g->prefill_tokens &&
                    g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out && g->batch_ffn_out;
    if (!ok) fprintf(stderr, "ds4: failed to allocate GLM batch MoE scratch\n");
    return ok;
}

static int glm52_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
#ifdef DS4_NO_GPU
    (void)e;
    (void)ctx_size;
    return 1;
#else
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    glm52_runtime *rt = xcalloc(1, sizeof(*rt));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = ds4_default_prefill_cap_for_prompt(ctx_size);
    s->variant_runtime = rt;
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));

    bool ok = glm52_alloc_decode_tensors(s);
    s->graph.prefill_cap = s->prefill_cap;
    s->graph.dense_mapped_bytes =
        e->model.size > e->model.tensor_data_pos ?
        e->model.size - e->model.tensor_data_pos :
        e->model.size;
    s->graph.quality = e->quality;
    if (ok) ok = glm52_alloc_moe_batch_tensors(s);
    if (ok && e->flash_moe) {
        ok = metal_graph_enable_flash_moe(&s->graph, e->flash_moe,
                                          &e->weights.layer[DS4_N_DENSE_LEAD]);
        if (!ok) fprintf(stderr, "ds4: failed to allocate GLM Flash-MoE slot banks\n");
    }
    if (ok && e->flash_moe) {
        s->graph.router_selected = rt->router_selected;
        s->graph.router_weights = rt->router_weights;
        s->graph.router_probs = rt->router_probs;
        s->graph.router_logits = rt->router_logits;
        s->graph.ffn_norm = rt->norm;
        s->graph.routed_gate = rt->routed_gate;
        s->graph.routed_up = rt->routed_up;
        s->graph.routed_mid = rt->routed_mid;
        s->graph.routed_down = rt->routed_down;
        s->graph.routed_out = rt->routed_out;
    }

    if (!ok) {
        glm52_session_free(s);
        token_vec_free(&s->checkpoint);
        free(s->logits);
        free(s);
        return 1;
    }

    *out = s;
    return 0;
#endif
}

static bool glm52_prepare_moe_slots(ds4_session *s, uint32_t il, uint32_t pos) {
#ifdef DS4_NO_GPU
    (void)s;
    (void)il;
    (void)pos;
    return false;
#else
    if (!s->graph.flash_moe) return true;
    if (!metal_graph_flash_moe_prepare_decode(&s->graph, il, pos)) return false;
    (void)ds4_gpu_begin_commands();
    return true;
#endif
}

static bool glm52_eval_moe(
        ds4_session             *s,
        const ds4_model         *m,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos,
        const char             **stage) {
    glm52_runtime *rt = glm52_rt(s);
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = (uint64_t)DS4_N_FF_EXP * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = (uint64_t)DS4_N_EMBD * down_row_bytes;
    const uint32_t active = DS4_N_EXPERT_ACTIVE_USED;

    *stage = "moe.router_matmul";
    if (!glm52_matmul(rt->router_logits, m, layer->ffn_gate_inp,
                      DS4_N_EMBD, DS4_N_EXPERT, rt->norm)) {
        return false;
    }
    *stage = "moe.glm_router_select";
    if (!ds4_gpu_glm_router_select_tensor(rt->router_selected,
                                          rt->router_weights,
                                          rt->router_probs,
                                          m->map,
                                          m->size,
                                          layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                          DS4_N_EXPERT,
                                          active,
                                          DS4_EXPERT_WEIGHT_SCALE,
                                          layer->ffn_exp_probs_b != NULL,
                                          rt->router_logits,
                                          1)) {
        return false;
    }

    if (s->engine->flash_moe) {
        *stage = "moe.flash_prepare";
        if (!glm52_prepare_moe_slots(s, il, pos)) return false;

        const ds4_flash_moe_layer_sidecar *flash_layer = &s->graph.flash_moe->layer[il];
        const uint64_t gate_slot_stride =
            s->graph.flash_mixed_slot_bank ? flash_layer->expert_stride : gate_expert_bytes;
        const uint64_t down_slot_stride =
            s->graph.flash_mixed_slot_bank ? flash_layer->expert_stride : down_expert_bytes;
        if (s->graph.flash_per_slot_buffers) {
            if (!s->graph.flash_decode_ids_valid[il]) return false;
            ds4_gpu_tensor *gate_slots[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            ds4_gpu_tensor *up_slots[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            ds4_gpu_tensor *down_slots[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            for (uint32_t k = 0; k < active; k++) {
                const int32_t slot = s->graph.flash_decode_slot_ids[il][k];
                gate_slots[k] = metal_graph_flash_moe_family_slot_cached_view(
                        &s->graph, il, DS4_FLASH_FAMILY_GATE, slot);
                up_slots[k] = metal_graph_flash_moe_family_slot_cached_view(
                        &s->graph, il, DS4_FLASH_FAMILY_UP, slot);
                down_slots[k] = metal_graph_flash_moe_family_slot_cached_view(
                        &s->graph, il, DS4_FLASH_FAMILY_DOWN, slot);
                if (!gate_slots[k] || !up_slots[k] || !down_slots[k]) return false;
            }
            *stage = "moe.flash_slots8";
            return ds4_gpu_routed_moe_one_slots8_tensor(
                       rt->routed_out,
                       rt->routed_gate,
                       rt->routed_up,
                       rt->routed_mid,
                       rt->routed_down,
                       gate_slots,
                       up_slots,
                       down_slots,
                       layer->ffn_gate_exps->type,
                       layer->ffn_down_exps->type,
                       gate_row_bytes,
                       down_row_bytes,
                       DS4_N_EMBD,
                       DS4_N_FF_EXP,
                       DS4_N_EMBD,
                       rt->router_weights,
                       active,
                       DS4_SWIGLU_CLAMP_EXP,
                       rt->norm) != 0;
        }
        if (s->graph.flash_per_expert_buffers) {
            if (!s->graph.flash_decode_ids_valid[il]) return false;
            *stage = "moe.flash_per_expert_routes";
            for (uint32_t k = 0; k < active; k++) {
                if (!metal_graph_flash_moe_compute_route_to_down(&s->graph,
                                                                 layer,
                                                                 il,
                                                                 k,
                                                                 s->graph.flash_decode_slot_ids[il][k],
                                                                 gate_expert_bytes,
                                                                 gate_slot_stride,
                                                                 gate_row_bytes,
                                                                 down_expert_bytes,
                                                                 down_slot_stride,
                                                                 down_row_bytes,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_FF_EXP,
                                                                 DS4_N_EMBD)) {
                    return false;
                }
            }
            *stage = "moe.flash_per_slot_sum";
            return active > 1 ?
                   ds4_gpu_moe_sum_experts_tensor(rt->routed_out,
                                                  rt->routed_down,
                                                  DS4_N_EMBD,
                                                  active) != 0 :
                   ds4_gpu_tensor_copy(rt->routed_out,
                                       0,
                                       rt->routed_down,
                                       0,
                                       (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
        *stage = "moe.flash_grouped";
        return ds4_gpu_routed_moe_one_banked_tensor(rt->routed_out,
                                                    rt->routed_gate,
                                                    rt->routed_up,
                                                    rt->routed_mid,
                                                    rt->routed_down,
                                                    s->graph.flash_gate_bank[il],
                                                    s->graph.flash_up_bank[il],
                                                    s->graph.flash_down_bank[il],
                                                    s->graph.flash_slot_bank,
                                                    layer->ffn_gate_exps->type,
                                                    layer->ffn_down_exps->type,
                                                    gate_expert_bytes,
                                                    gate_slot_stride,
                                                    gate_row_bytes,
                                                    down_expert_bytes,
                                                    down_slot_stride,
                                                    down_row_bytes,
                                                    DS4_N_EMBD,
                                                    DS4_N_FF_EXP,
                                                    DS4_N_EMBD,
                                                    s->graph.router_slot_selected,
                                                    rt->router_weights,
                                                    active,
                                                    DS4_SWIGLU_CLAMP_EXP,
                                                    rt->norm) != 0;
    }

    *stage = "moe.resident";
    return ds4_gpu_routed_moe_one_tensor(rt->routed_out,
                                         rt->routed_gate,
                                         rt->routed_up,
                                         rt->routed_mid,
                                         rt->routed_down,
                                         m->map,
                                         m->size,
                                         layer->ffn_gate_exps->abs_offset,
                                         layer->ffn_up_exps->abs_offset,
                                         layer->ffn_down_exps->abs_offset,
                                         layer->ffn_gate_exps->type,
                                         layer->ffn_down_exps->type,
                                         gate_expert_bytes,
                                         gate_row_bytes,
                                         down_expert_bytes,
                                         down_row_bytes,
                                         DS4_N_EMBD,
                                         DS4_N_FF_EXP,
                                         DS4_N_EMBD,
                                         rt->router_selected,
                                         rt->router_weights,
                                         DS4_N_EXPERT,
                                         active,
                                         DS4_SWIGLU_CLAMP_EXP,
                                         rt->norm) != 0;
}

static bool glm52_eval_moe_batch(
        ds4_session             *s,
        const ds4_model         *m,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens,
        const char             **stage) {
    ds4_gpu_graph *g = &s->graph;
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const uint32_t active = DS4_N_EXPERT_ACTIVE_USED;

    if (!g->batch_ffn_norm || !g->batch_router_logits || !g->batch_router_probs ||
        !g->batch_router_selected || !g->batch_router_weights ||
        !g->batch_routed_out || n_tokens == 0 || n_tokens > g->prefill_cap) {
        return false;
    }
    if (env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE") &&
        (!s->engine->flash_moe || !g->flash_moe)) {
        static bool warned_nax_pc_requires_sidecar = false;
        if (!warned_nax_pc_requires_sidecar) {
            fprintf(stderr,
                    "ds4: ERROR: GLM NAX INT8 per-channel REQUIRE needs a v2 "
                    "Flash-MoE sidecar; full-GGUF resident weights have no "
                    "per-expert scale tensor\n");
            warned_nax_pc_requires_sidecar = true;
        }
        return false;
    }

    *stage = "moe.router_matmul_batch";
    if (!glm52_matmul_rows(g->batch_router_logits, m, layer->ffn_gate_inp,
                           DS4_N_EMBD, DS4_N_EXPERT, g->batch_ffn_norm, n_tokens)) {
        return false;
    }

    *stage = "moe.glm_router_select_batch";
    if (!ds4_gpu_glm_router_select_tensor(g->batch_router_selected,
                                          g->batch_router_weights,
                                          g->batch_router_probs,
                                          m->map,
                                          m->size,
                                          layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                          DS4_N_EXPERT,
                                          active,
                                          DS4_EXPERT_WEIGHT_SCALE,
                                          layer->ffn_exp_probs_b != NULL,
                                          g->batch_router_logits,
                                          n_tokens)) {
        return false;
    }

    if (s->engine->flash_moe && g->flash_moe) {
        g->batch_routed_mid_is_f16 = false;
        *stage = "moe.flash_prefill_dedup";
        return metal_graph_flash_moe_run_prefill_dedup(g,
                                                       layer,
                                                       il,
                                                       n_tokens,
                                                       gate_expert_bytes,
                                                       gate_row_bytes,
                                                       down_expert_bytes,
                                                       down_row_bytes,
                                                       (uint32_t)expert_in_dim,
                                                       (uint32_t)down_in_dim,
                                                       (uint32_t)routed_out_dim);
    }

    bool resident_mpp_done = false;
    char rb_tok[24];
    ds4_gpu_resident_backend_for_tokens(n_tokens, rb_tok, sizeof(rb_tok));
    const bool rb_set = rb_tok[0] != '\0';
    const bool rb_ane = strncmp(rb_tok, "ane", 3) == 0;
    const bool take_dedup =
        rb_ane || (!rb_set && resident_moe_mpp_dedup_prefill_enabled());
    if (take_dedup) {
        g->batch_routed_mid_is_f16 = false;
        *stage = "moe.resident_mpp_dedup";
        resident_mpp_done = metal_graph_resident_moe_run_mpp_prefill_dedup(g,
                                                                           m,
                                                                           layer,
                                                                           il,
                                                                           n_tokens,
                                                                           gate_expert_bytes,
                                                                           gate_row_bytes,
                                                                           down_expert_bytes,
                                                                           down_row_bytes,
                                                                           (uint32_t)expert_in_dim,
                                                                           (uint32_t)down_in_dim,
                                                                           (uint32_t)routed_out_dim,
                                                                           rb_ane);
        if (!resident_mpp_done) {
            static bool warned_resident_mpp_fallback = false;
            if (!warned_resident_mpp_fallback) {
                fprintf(stderr,
                        "ds4: GLM resident MPP/NAX DeDup prefill unavailable; "
                        "falling back to resident Metal grouped MoE\n");
                warned_resident_mpp_fallback = true;
            }
            (void)ds4_gpu_synchronize();
            if (ds4_gpu_begin_commands() == 0) return false;
        }
    }

    if (resident_mpp_done) return true;

    g->batch_routed_mid_is_f16 = false;
    *stage = "moe.resident_batch";
    return metal_graph_routed_moe_batch_tiled(g,
                                              g->batch_routed_out,
                                              m,
                                              layer,
                                              n_tokens,
                                              gate_expert_bytes,
                                              gate_row_bytes,
                                              down_expert_bytes,
                                              down_row_bytes,
                                              (uint32_t)expert_in_dim,
                                              (uint32_t)down_in_dim,
                                              (uint32_t)routed_out_dim,
                                              g->batch_router_selected,
                                              g->batch_router_weights,
                                              g->batch_ffn_norm,
                                              &g->batch_routed_mid_is_f16);
}

static void glm52_report_batch_display_progress(
        ds4_session *s,
        const char  *event,
        uint32_t     pos0,
        uint32_t     n_tokens,
        uint32_t     work_done,
        uint32_t     work_total,
        int          total) {
    if (!s || !s->display_progress || n_tokens == 0 || work_total == 0) return;
    if (work_done > work_total) work_done = work_total;

    uint64_t done = 0;
    if (work_done != 0) {
        done = ((uint64_t)n_tokens * work_done + work_total - 1u) / work_total;
        if (done == 0) done = 1;
    }
    if (work_done >= work_total) done = n_tokens;
    if (done > n_tokens) done = n_tokens;

    s->display_progress(s->display_progress_ud,
                        event && event[0] ? event : "prefill_display",
                        (int)(pos0 + (uint32_t)done),
                        total);
}

static int glm52_eval_batch(ds4_session *s, const int *tokens, uint32_t n_tokens,
                            int progress_total, char *err, size_t errlen) {
    if (!s || !s->engine || !tokens || !glm52_rt(s)) return 1;
    glm52_runtime *rt = glm52_rt(s);
    ds4_engine *e = s->engine;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos0 = rt->n_past;
    const uint64_t e_dim = DS4_N_EMBD;
    const char *stage = "begin";

    if (n_tokens == 0 || n_tokens > rt->batch_cap || n_tokens > s->prefill_cap) {
        snprintf(err, errlen, "GLM invalid batch prefill chunk");
        return 1;
    }
    if (pos0 >= (uint32_t)s->ctx_size || n_tokens > (uint32_t)s->ctx_size - pos0) {
        snprintf(err, errlen, "GLM context is full");
        return 1;
    }

    if (s->display_progress) {
        s->display_progress(s->display_progress_ud,
                            "prefill_upload",
                            (int)pos0,
                            progress_total);
    }

    int32_t *token_i32 = xmalloc((size_t)n_tokens * sizeof(token_i32[0]));
    bool embed_ok = true;
    for (uint32_t t = 0; t < n_tokens; t++) {
        token_i32[t] = (int32_t)tokens[t];
        if (!glm52_embed_token(m, w, tokens[t],
                               rt->batch_embed_host + (uint64_t)t * e_dim)) {
            embed_ok = false;
            break;
        }
    }
    if (!embed_ok ||
        ds4_gpu_tensor_write(rt->batch_cur, 0, rt->batch_embed_host,
                             (uint64_t)n_tokens * e_dim * sizeof(rt->batch_embed_host[0])) == 0 ||
        ds4_gpu_tensor_write(s->graph.prefill_tokens, 0, token_i32,
                             (uint64_t)n_tokens * sizeof(token_i32[0])) == 0) {
        free(token_i32);
        snprintf(err, errlen, "GLM failed to load batch token embeddings");
        return 1;
    }
    free(token_i32);

    (void)ds4_gpu_begin_commands();
    bool ok = true;
    const float attn_scale = glm52_attention_scale();
    glm52_report_batch_display_progress(s,
                                        "prefill_compute",
                                        pos0,
                                        n_tokens,
                                        1,
                                        GLM52_N_EFFECTIVE_LAYER,
                                        progress_total);

#define GLM52_BATCH_STEP(name, expr) do { \
        if (ok) {                         \
            stage = (name);               \
            ok = (expr);                  \
        }                                 \
    } while (0)

    for (uint32_t il = 0; ok && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        const ds4_layer_weights *layer = &w->layer[il];

        GLM52_BATCH_STEP("attn.rms_batch",
                   ds4_gpu_rms_norm_weight_rows_tensor(rt->batch_norm,
                                                       rt->batch_cur,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0);
        GLM52_BATCH_STEP("attn.q_a_batch",
                   glm52_matmul_rows(rt->batch_qr, m, layer->attn_q_a,
                                     DS4_N_EMBD, DS4_N_LORA_Q,
                                     rt->batch_norm, n_tokens));
        GLM52_BATCH_STEP("attn.q_a_norm_batch",
                   ds4_gpu_rms_norm_weight_rows_tensor(rt->batch_qr_norm,
                                                       rt->batch_qr,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_q_a_norm->abs_offset,
                                                       DS4_N_LORA_Q,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0);
        GLM52_BATCH_STEP("attn.q_b_batch",
                   glm52_matmul_rows(rt->batch_q, m, layer->attn_q_b,
                                     DS4_N_LORA_Q,
                                     (uint64_t)DS4_N_HEAD * GLM52_Q_HEAD_DIM,
                                     rt->batch_qr_norm,
                                     n_tokens));
        GLM52_BATCH_STEP("attn.q_rope_batch",
                   ds4_gpu_rope_tail_tensor(rt->batch_q, n_tokens, DS4_N_HEAD,
                                            GLM52_Q_HEAD_DIM, GLM52_K_PE_DIM,
                                            pos0, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        GLM52_BATCH_STEP("attn.kv_a_batch",
                   glm52_matmul_rows(rt->batch_kv_raw, m, layer->attn_kv,
                                     DS4_N_EMBD, GLM52_K_HEAD_DIM,
                                     rt->batch_norm, n_tokens));
        GLM52_BATCH_STEP("attn.kv_split_batch",
                   ds4_gpu_glm52_split_kv_batch_tensor(rt->batch_kv_lora,
                                                       rt->batch_k_pe,
                                                       rt->batch_kv_raw,
                                                       n_tokens) != 0);
        GLM52_BATCH_STEP("attn.kv_norm_batch",
                   ds4_gpu_rms_norm_weight_rows_tensor(rt->batch_kv_norm,
                                                       rt->batch_kv_lora,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_kv_a_norm->abs_offset,
                                                       GLM52_KV_LORA_DIM,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0);
        GLM52_BATCH_STEP("attn.k_rope_batch",
                   ds4_gpu_rope_tail_tensor(rt->batch_k_pe, n_tokens, 1,
                                            GLM52_K_PE_DIM, GLM52_K_PE_DIM,
                                            pos0, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        GLM52_BATCH_STEP("attn.cache_store_batch",
                   ds4_gpu_glm52_store_kv_batch_tensor(rt->layer_kv[il],
                                                       rt->layer_kpe[il],
                                                       rt->batch_kv_norm,
                                                       rt->batch_k_pe,
                                                       pos0,
                                                       n_tokens,
                                                       (uint32_t)s->ctx_size) != 0);
        GLM52_BATCH_STEP("attn.q_absorb_batch",
                   ds4_gpu_glm52_q8_head_matvec_batch_tensor(rt->batch_q_abs,
                                                             m->map,
                                                             m->size,
                                                             layer->attn_k_b->abs_offset,
                                                             GLM52_Q_NOPE_DIM,
                                                             GLM52_V_HEAD_DIM,
                                                             DS4_N_HEAD,
                                                             GLM52_Q_HEAD_DIM,
                                                             GLM52_V_HEAD_DIM,
                                                             0,
                                                             rt->batch_q,
                                                             n_tokens) != 0);
        GLM52_BATCH_STEP("attn.prefill_batch",
                   ds4_gpu_glm52_attention_prefill_tensor(rt->batch_attn_lora,
                                                          rt->batch_q_abs,
                                                          rt->batch_q,
                                                          rt->layer_kv[il],
                                                          rt->layer_kpe[il],
                                                          pos0,
                                                          n_tokens,
                                                          (uint32_t)s->ctx_size,
                                                          DS4_N_HEAD,
                                                          attn_scale) != 0);
        GLM52_BATCH_STEP("attn.v_b_batch",
                   ds4_gpu_glm52_q8_head_matvec_batch_tensor(rt->batch_attn_heads,
                                                             m->map,
                                                             m->size,
                                                             layer->attn_v_b->abs_offset,
                                                             GLM52_V_HEAD_DIM,
                                                             GLM52_V_IMPL_DIM,
                                                             DS4_N_HEAD,
                                                             GLM52_V_HEAD_DIM,
                                                             GLM52_V_IMPL_DIM,
                                                             0,
                                                             rt->batch_attn_lora,
                                                             n_tokens) != 0);
        GLM52_BATCH_STEP("attn.o_batch",
                   glm52_matmul_rows(rt->batch_attn_out, m, layer->attn_output_a,
                                     (uint64_t)DS4_N_HEAD * GLM52_V_IMPL_DIM,
                                     DS4_N_EMBD,
                                     rt->batch_attn_heads,
                                     n_tokens));
        GLM52_BATCH_STEP("attn.residual_batch",
                   ds4_gpu_add_tensor(rt->batch_next,
                                      rt->batch_cur,
                                      rt->batch_attn_out,
                                      (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0);

        GLM52_BATCH_STEP("ffn.rms_batch",
                   ds4_gpu_rms_norm_weight_rows_tensor(s->graph.batch_ffn_norm,
                                                       rt->batch_next,
                                                       m->map,
                                                       m->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0);
        if (il < DS4_N_DENSE_LEAD) {
            GLM52_BATCH_STEP("ffn.dense_gate_batch",
                       glm52_matmul_rows(rt->batch_ffn_gate, m, layer->ffn_gate,
                                         DS4_N_EMBD, GLM52_DENSE_FF_DIM,
                                         s->graph.batch_ffn_norm, n_tokens));
            GLM52_BATCH_STEP("ffn.dense_up_batch",
                       glm52_matmul_rows(rt->batch_ffn_up, m, layer->ffn_up,
                                         DS4_N_EMBD, GLM52_DENSE_FF_DIM,
                                         s->graph.batch_ffn_norm, n_tokens));
            GLM52_BATCH_STEP("ffn.dense_swiglu_batch",
                       ds4_gpu_swiglu_tensor(rt->batch_ffn_mid,
                                             rt->batch_ffn_gate,
                                             rt->batch_ffn_up,
                                             (uint32_t)((uint64_t)n_tokens * GLM52_DENSE_FF_DIM),
                                             0.0f,
                                             1.0f) != 0);
            GLM52_BATCH_STEP("ffn.dense_down_batch",
                       glm52_matmul_rows(rt->batch_ffn_down, m, layer->ffn_down,
                                         GLM52_DENSE_FF_DIM, DS4_N_EMBD,
                                         rt->batch_ffn_mid, n_tokens));
            GLM52_BATCH_STEP("ffn.dense_residual_batch",
                       ds4_gpu_add_tensor(rt->batch_cur,
                                          rt->batch_next,
                                          rt->batch_ffn_down,
                                          (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0);
        } else {
            if (ok) ok = glm52_eval_moe_batch(s, m, layer, il, n_tokens, &stage);
            const uint64_t shared_dim = (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED;
            GLM52_BATCH_STEP("ffn.shared_gate_batch",
                       glm52_matmul_rows(s->graph.batch_shared_gate, m,
                                         layer->ffn_gate_shexp,
                                         DS4_N_EMBD,
                                         shared_dim,
                                         s->graph.batch_ffn_norm,
                                         n_tokens));
            GLM52_BATCH_STEP("ffn.shared_up_batch",
                       glm52_matmul_rows(s->graph.batch_shared_up, m,
                                         layer->ffn_up_shexp,
                                         DS4_N_EMBD,
                                         shared_dim,
                                         s->graph.batch_ffn_norm,
                                         n_tokens));
            GLM52_BATCH_STEP("ffn.shared_swiglu_batch",
                       ds4_gpu_swiglu_tensor(s->graph.batch_shared_mid,
                                             s->graph.batch_shared_gate,
                                             s->graph.batch_shared_up,
                                             (uint32_t)((uint64_t)n_tokens * shared_dim),
                                             0.0f,
                                             1.0f) != 0);
            GLM52_BATCH_STEP("ffn.shared_down_batch",
                       glm52_matmul_rows(s->graph.batch_shared_out, m,
                                         layer->ffn_down_shexp,
                                         shared_dim,
                                         DS4_N_EMBD,
                                         s->graph.batch_shared_mid,
                                         n_tokens));
            GLM52_BATCH_STEP("ffn.combine_batch",
                       ds4_gpu_add_tensor(s->graph.batch_ffn_out,
                                          s->graph.batch_shared_out,
                                          s->graph.batch_routed_out,
                                          (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0);
            GLM52_BATCH_STEP("ffn.residual_batch",
                       ds4_gpu_add_tensor(rt->batch_cur,
                                          rt->batch_next,
                                          s->graph.batch_ffn_out,
                                          (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0);
        }
        glm52_report_batch_display_progress(s,
                                            "prefill_compute",
                                            pos0,
                                            n_tokens,
                                            il + 1u,
                                            GLM52_N_EFFECTIVE_LAYER,
                                            progress_total);
    }

    ds4_gpu_tensor *last_cur = NULL;
    if (ok) {
        stage = "output.view_last";
        last_cur = ds4_gpu_tensor_view(rt->batch_cur,
                                       (uint64_t)(n_tokens - 1u) * DS4_N_EMBD * sizeof(float),
                                       (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = last_cur != NULL;
    }
    GLM52_BATCH_STEP("output.rms",
               ds4_gpu_rms_norm_weight_tensor(rt->norm,
                                              last_cur,
                                              m->map,
                                              m->size,
                                              w->output_norm->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_RMS_EPS) != 0);
    GLM52_BATCH_STEP("output.logits",
               glm52_matmul(rt->logits_gpu, m, w->output,
                            DS4_N_EMBD, DS4_N_VOCAB, rt->norm));
    ds4_gpu_tensor_free(last_cur);

#undef GLM52_BATCH_STEP

    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "GLM-5.2 batch prefill failed at %s", stage ? stage : "unknown");
        s->checkpoint_valid = false;
        return 1;
    }
    if (ds4_gpu_tensor_read(rt->logits_gpu, 0, s->logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0) {
        snprintf(err, errlen, "GLM-5.2 failed to read batch logits");
        s->checkpoint_valid = false;
        return 1;
    }

    rt->n_past += n_tokens;
    for (uint32_t t = 0; t < n_tokens; t++) token_vec_push(&s->checkpoint, tokens[t]);
    s->checkpoint_valid = true;
    return 0;
}

static int glm52_eval_token(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s || !s->engine || !glm52_rt(s)) return 1;
    glm52_runtime *rt = glm52_rt(s);
    ds4_engine *e = s->engine;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos = rt->n_past;
    const char *stage = "begin";

    if (pos >= (uint32_t)s->ctx_size) {
        snprintf(err, errlen, "GLM context is full");
        return 1;
    }
    if (!glm52_embed_token(m, w, token, rt->embed_host) ||
        ds4_gpu_tensor_write(rt->cur, 0, rt->embed_host,
                             (uint64_t)DS4_N_EMBD * sizeof(rt->embed_host[0])) == 0) {
        snprintf(err, errlen, "GLM failed to load token embedding");
        return 1;
    }

    (void)ds4_gpu_begin_commands();
    bool ok = true;
    const float attn_scale = glm52_attention_scale();

#define GLM52_STEP(name, expr) do {       \
        if (ok) {                         \
            stage = (name);               \
            ok = (expr);                  \
        }                                 \
    } while (0)

    for (uint32_t il = 0; ok && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        const ds4_layer_weights *layer = &w->layer[il];

        GLM52_STEP("attn.rms",
                   ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->cur,
                                                  m->map, m->size,
                                                  layer->attn_norm->abs_offset,
                                                  DS4_N_EMBD, DS4_RMS_EPS) != 0);
        GLM52_STEP("attn.q_a",
                   glm52_matmul(rt->qr, m, layer->attn_q_a,
                                DS4_N_EMBD, DS4_N_LORA_Q, rt->norm));
        GLM52_STEP("attn.q_a_norm",
                   ds4_gpu_rms_norm_weight_tensor(rt->qr_norm, rt->qr,
                                                  m->map, m->size,
                                                  layer->attn_q_a_norm->abs_offset,
                                                  DS4_N_LORA_Q, DS4_RMS_EPS) != 0);
        GLM52_STEP("attn.q_b",
                   glm52_matmul(rt->q, m, layer->attn_q_b,
                                DS4_N_LORA_Q,
                                (uint64_t)DS4_N_HEAD * GLM52_Q_HEAD_DIM,
                                rt->qr_norm));
        GLM52_STEP("attn.q_rope",
                   ds4_gpu_rope_tail_tensor(rt->q, 1, DS4_N_HEAD, GLM52_Q_HEAD_DIM,
                                            GLM52_K_PE_DIM, pos, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        GLM52_STEP("attn.kv_a",
                   glm52_matmul(rt->kv_raw, m, layer->attn_kv,
                                DS4_N_EMBD, GLM52_K_HEAD_DIM, rt->norm));
        GLM52_STEP("attn.kv_norm",
                   ds4_gpu_rms_norm_weight_tensor(rt->kv_norm, rt->kv_lora_raw,
                                                  m->map, m->size,
                                                  layer->attn_kv_a_norm->abs_offset,
                                                  GLM52_KV_LORA_DIM, DS4_RMS_EPS) != 0);
        GLM52_STEP("attn.k_rope",
                   ds4_gpu_rope_tail_tensor(rt->k_pe, 1, 1, GLM52_K_PE_DIM,
                                            GLM52_K_PE_DIM, pos, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        GLM52_STEP("attn.cache_store",
                   ds4_gpu_glm52_store_kv_tensor(rt->layer_kv[il],
                                                 rt->layer_kpe[il],
                                                 rt->kv_norm,
                                                 rt->k_pe,
                                                 pos,
                                                 (uint32_t)s->ctx_size) != 0);
        GLM52_STEP("attn.q_absorb",
                   ds4_gpu_glm52_q8_head_matvec_tensor(rt->q_abs,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_k_b->abs_offset,
                                                       GLM52_Q_NOPE_DIM,
                                                       GLM52_V_HEAD_DIM,
                                                       DS4_N_HEAD,
                                                       GLM52_Q_HEAD_DIM,
                                                       GLM52_V_HEAD_DIM,
                                                       0,
                                                       rt->q) != 0);
        GLM52_STEP("attn.decode",
                   ds4_gpu_glm52_attention_decode_tensor(rt->attn_lora,
                                                         rt->q_abs,
                                                         rt->q,
                                                         rt->layer_kv[il],
                                                         rt->layer_kpe[il],
                                                         rt->attn_scores,
                                                         pos + 1u,
                                                         (uint32_t)s->ctx_size,
                                                         DS4_N_HEAD,
                                                         attn_scale) != 0);
        GLM52_STEP("attn.v_b",
                   ds4_gpu_glm52_q8_head_matvec_tensor(rt->attn_heads,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_v_b->abs_offset,
                                                       GLM52_V_HEAD_DIM,
                                                       GLM52_V_IMPL_DIM,
                                                       DS4_N_HEAD,
                                                       GLM52_V_HEAD_DIM,
                                                       GLM52_V_IMPL_DIM,
                                                       0,
                                                       rt->attn_lora) != 0);
        GLM52_STEP("attn.o",
                   glm52_matmul(rt->attn_out, m, layer->attn_output_a,
                                (uint64_t)DS4_N_HEAD * GLM52_V_IMPL_DIM,
                                DS4_N_EMBD,
                                rt->attn_heads));
        GLM52_STEP("attn.residual",
                   ds4_gpu_add_tensor(rt->next, rt->cur, rt->attn_out, DS4_N_EMBD) != 0);

        GLM52_STEP("ffn.rms",
                   ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->next,
                                                  m->map, m->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD, DS4_RMS_EPS) != 0);
        if (il < DS4_N_DENSE_LEAD) {
            GLM52_STEP("ffn.dense_gate",
                       glm52_matmul(rt->ffn_gate, m, layer->ffn_gate,
                                    DS4_N_EMBD, GLM52_DENSE_FF_DIM, rt->norm));
            GLM52_STEP("ffn.dense_up",
                       glm52_matmul(rt->ffn_up, m, layer->ffn_up,
                                    DS4_N_EMBD, GLM52_DENSE_FF_DIM, rt->norm));
            GLM52_STEP("ffn.dense_swiglu",
                       ds4_gpu_swiglu_tensor(rt->ffn_mid,
                                             rt->ffn_gate,
                                             rt->ffn_up,
                                             GLM52_DENSE_FF_DIM,
                                             0.0f,
                                             1.0f) != 0);
            GLM52_STEP("ffn.dense_down",
                       glm52_matmul(rt->ffn_down, m, layer->ffn_down,
                                    GLM52_DENSE_FF_DIM, DS4_N_EMBD, rt->ffn_mid));
            GLM52_STEP("ffn.dense_residual",
                       ds4_gpu_add_tensor(rt->cur, rt->next, rt->ffn_down, DS4_N_EMBD) != 0);
        } else {
            if (ok) ok = glm52_eval_moe(s, m, layer, il, pos, &stage);
            GLM52_STEP("ffn.shared_gate",
                       glm52_matmul(rt->shared_gate, m, layer->ffn_gate_shexp,
                                    DS4_N_EMBD,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    rt->norm));
            GLM52_STEP("ffn.shared_up",
                       glm52_matmul(rt->shared_up, m, layer->ffn_up_shexp,
                                    DS4_N_EMBD,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    rt->norm));
            GLM52_STEP("ffn.shared_swiglu",
                       ds4_gpu_swiglu_tensor(rt->shared_mid,
                                             rt->shared_gate,
                                             rt->shared_up,
                                             (uint32_t)(DS4_N_FF_EXP * DS4_N_EXPERT_SHARED),
                                             0.0f,
                                             1.0f) != 0);
            GLM52_STEP("ffn.shared_down",
                       glm52_matmul(rt->shared_out, m, layer->ffn_down_shexp,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    DS4_N_EMBD,
                                    rt->shared_mid));
            GLM52_STEP("ffn.combine",
                       ds4_gpu_add_tensor(rt->ffn_out, rt->shared_out,
                                          rt->routed_out, DS4_N_EMBD) != 0);
            GLM52_STEP("ffn.residual",
                       ds4_gpu_add_tensor(rt->cur, rt->next, rt->ffn_out, DS4_N_EMBD) != 0);
        }
    }

    GLM52_STEP("output.rms",
               ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->cur,
                                              m->map, m->size,
                                              w->output_norm->abs_offset,
                                              DS4_N_EMBD, DS4_RMS_EPS) != 0);
    GLM52_STEP("output.logits",
               glm52_matmul(rt->logits_gpu, m, w->output,
                            DS4_N_EMBD, DS4_N_VOCAB, rt->norm));

#undef GLM52_STEP

    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "GLM-5.2 eval failed at %s", stage ? stage : "unknown");
        s->checkpoint_valid = false;
        return 1;
    }
    if (ds4_gpu_tensor_read(rt->logits_gpu, 0, s->logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0) {
        snprintf(err, errlen, "GLM-5.2 failed to read logits");
        s->checkpoint_valid = false;
        return 1;
    }

    rt->n_past++;
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

static void glm52_session_reset(ds4_session *s) {
    glm52_runtime *rt = glm52_rt(s);
    if (!rt) return;
    rt->n_past = 0;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
}

static int glm52_session_sync(ds4_session *s, const ds4_tokens *prompt,
                              char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    glm52_runtime *rt = glm52_rt(s);
    if (!rt) {
        snprintf(err, errlen, "GLM-5.2 runtime is not initialized");
        return 1;
    }

    int start = 0;
    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = s->checkpoint.len;
    } else {
        glm52_session_reset(s);
    }

    const char *batch_env = getenv("DS4_GLM52_BATCH_PREFILL");
    const bool batch_enabled = !batch_env || !batch_env[0] || atoi(batch_env) != 0;
    if (!batch_enabled) {
        for (int i = start; i < prompt->len; i++) {
            if (glm52_eval_token(s, prompt->v[i], err, errlen) != 0) return 1;
            if (s->progress) s->progress(s->progress_ud, "prefill_token", i + 1, prompt->len);
        }
    } else {
        int i = start;
        while (i < prompt->len) {
            uint32_t chunk = s->prefill_cap ? s->prefill_cap : 1u;
            const uint32_t remaining = (uint32_t)(prompt->len - i);
            if (chunk > remaining) chunk = remaining;
            if (chunk == 0) chunk = 1;
            if (glm52_eval_batch(s, &prompt->v[i], chunk, prompt->len, err, errlen) != 0) return 1;
            i += (int)chunk;
            if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        }
    }
    s->checkpoint_valid = true;
    return 0;
}

static int glm52_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    return glm52_eval_token(s, token, err, errlen);
}

#ifndef DS4_NO_GPU
#define GLM52_SESSION_PAYLOAD_MAGIC UINT32_C(0x324d4c47) /* "GLM2" */
#define GLM52_SESSION_PAYLOAD_VERSION UINT32_C(1)
#define GLM52_SESSION_PAYLOAD_U32_FIELDS 10u
#define GLM52_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void glm52_payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void glm52_payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)v;
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t glm52_payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int glm52_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                                     char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            glm52_payload_set_err(err, errlen, "failed to write GLM session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int glm52_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                                    uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        glm52_payload_set_err(err, errlen, "truncated GLM session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            glm52_payload_set_err(err, errlen, "failed to read GLM session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int glm52_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    glm52_payload_put_u32(b, v);
    return glm52_payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int glm52_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                                  char *err, size_t errlen) {
    uint8_t b[4];
    if (glm52_payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) {
        return 1;
    }
    *v = glm52_payload_get_u32(b);
    return 0;
}

static int glm52_payload_write_tensor(FILE *fp, const ds4_gpu_tensor *tensor,
                                      uint64_t bytes, uint8_t *buf,
                                      char *err, size_t errlen) {
    if (!tensor || ds4_gpu_tensor_bytes(tensor) < bytes) {
        glm52_payload_set_err(err, errlen, "GLM session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > GLM52_SESSION_IO_CHUNK ?
                         GLM52_SESSION_IO_CHUNK : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, done, buf, n) == 0) {
            glm52_payload_set_err(err, errlen, "failed to read GLM Metal session tensor");
            return 1;
        }
        if (glm52_payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int glm52_payload_read_tensor(FILE *fp, ds4_gpu_tensor *tensor,
                                     uint64_t bytes, uint8_t *buf,
                                     uint64_t *remaining,
                                     char *err, size_t errlen) {
    if (!tensor || ds4_gpu_tensor_bytes(tensor) < bytes) {
        glm52_payload_set_err(err, errlen, "GLM session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > GLM52_SESSION_IO_CHUNK ?
                         GLM52_SESSION_IO_CHUNK : (size_t)(bytes - done);
        if (glm52_payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, done, buf, n) == 0) {
            glm52_payload_set_err(err, errlen, "failed to restore GLM Metal session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

static uint64_t glm52_session_payload_bytes(ds4_session *s) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !s->checkpoint_valid) return 0;
    const uint32_t live = rt->n_past;
    uint64_t bytes = (uint64_t)GLM52_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)GLM52_N_EFFECTIVE_LAYER * live *
             (GLM52_KV_LORA_DIM + GLM52_K_PE_DIM) * sizeof(float);
    return bytes;
}

static int glm52_session_save_payload(ds4_session *s, FILE *fp,
                                      char *err, size_t errlen) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !fp || !s->checkpoint_valid) {
        glm52_payload_set_err(err, errlen, "GLM session has no valid checkpoint to save");
        return 1;
    }
    if (rt->n_past != (uint32_t)s->checkpoint.len) {
        glm52_payload_set_err(err, errlen, "GLM KV row count does not match checkpoint");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        glm52_payload_set_err(err, errlen, "failed to synchronize Metal before GLM snapshot");
        return 1;
    }

    const uint32_t header[GLM52_SESSION_PAYLOAD_U32_FIELDS] = {
        GLM52_SESSION_PAYLOAD_MAGIC,
        GLM52_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        (uint32_t)s->checkpoint.len,
        GLM52_N_EFFECTIVE_LAYER,
        GLM52_KV_LORA_DIM,
        GLM52_K_PE_DIM,
        DS4_N_VOCAB,
        rt->n_past,
        DS4_N_EMBD,
    };
    for (uint32_t i = 0; i < GLM52_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (glm52_payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (glm52_payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (glm52_payload_write_bytes(fp, s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(float),
                                  err, errlen) != 0) {
        return 1;
    }

    uint8_t *buf = xmalloc(GLM52_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t kv_bytes = (uint64_t)rt->n_past * GLM52_KV_LORA_DIM * sizeof(float);
    const uint64_t kpe_bytes = (uint64_t)rt->n_past * GLM52_K_PE_DIM * sizeof(float);
    for (uint32_t il = 0; rc == 0 && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        rc = glm52_payload_write_tensor(fp, rt->layer_kv[il], kv_bytes, buf, err, errlen);
        if (rc == 0) {
            rc = glm52_payload_write_tensor(fp, rt->layer_kpe[il], kpe_bytes, buf, err, errlen);
        }
    }
    free(buf);
    return rc;
}

static int glm52_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                                      char *err, size_t errlen) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !fp) {
        glm52_payload_set_err(err, errlen, "invalid GLM session payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[GLM52_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < GLM52_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (glm52_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != GLM52_SESSION_PAYLOAD_MAGIC || h[1] != GLM52_SESSION_PAYLOAD_VERSION) {
        glm52_payload_set_err(err, errlen, "unsupported GLM session payload version");
        return 1;
    }
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_tokens = h[3];
    const uint32_t saved_layers = h[4];
    const uint32_t saved_kv_dim = h[5];
    const uint32_t saved_kpe_dim = h[6];
    const uint32_t saved_vocab = h[7];
    const uint32_t saved_n_past = h[8];
    const uint32_t saved_embd = h[9];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size ||
        saved_n_past != saved_tokens) {
        glm52_payload_set_err(err, errlen, "GLM KV checkpoint does not fit current context");
        return 1;
    }
    if (saved_layers != GLM52_N_EFFECTIVE_LAYER ||
        saved_kv_dim != GLM52_KV_LORA_DIM ||
        saved_kpe_dim != GLM52_K_PE_DIM ||
        saved_vocab != DS4_N_VOCAB ||
        saved_embd != DS4_N_EMBD) {
        glm52_payload_set_err(err, errlen, "GLM KV checkpoint was written for a different layout");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (glm52_payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (glm52_payload_read_bytes(fp, s->logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float),
                                 &remaining, err, errlen) != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }

    uint8_t *buf = xmalloc(GLM52_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t kv_bytes = (uint64_t)saved_n_past * GLM52_KV_LORA_DIM * sizeof(float);
    const uint64_t kpe_bytes = (uint64_t)saved_n_past * GLM52_K_PE_DIM * sizeof(float);
    for (uint32_t il = 0; rc == 0 && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        rc = glm52_payload_read_tensor(fp, rt->layer_kv[il], kv_bytes, buf,
                                       &remaining, err, errlen);
        if (rc == 0) {
            rc = glm52_payload_read_tensor(fp, rt->layer_kpe[il], kpe_bytes, buf,
                                           &remaining, err, errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        glm52_payload_set_err(err, errlen, "GLM KV checkpoint has trailing payload bytes");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    rt->n_past = saved_n_past;
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    if (s->graph.flash_moe &&
        !metal_graph_flash_moe_reset_slot_cache_after_prefill(&s->graph,
                                                              "after GLM KV payload load")) {
        glm52_payload_set_err(err, errlen, "failed to reset Flash-MoE slot cache after GLM KV payload load");
        return 1;
    }
    return 0;
}
#endif
