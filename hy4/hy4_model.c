/* HY4 preview GGUF metadata and tensor binding. The MLA/iHC graph is native
 * HY4; routed tensors can come from the existing Flash-MoE sidecar bank. */
static bool hy4_model_is_hyv4(const ds4_model *m) {
    ds4_str arch = {0};
    return model_get_string(m, "general.architecture", &arch) && ds4_streq(arch, "hyv4");
}

static bool g_hy4_indexer_is_full[DS4_MAX_LAYER];
static float g_hy4_hc_magnitude = 2.0f;

static void hy4_config_validate_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_HY4;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    config_expect_u32("block_count", required_u32(m, "hyv4.block_count"), DS4_N_LAYER);
    config_expect_u32("embedding_length", required_u32(m, "hyv4.embedding_length"), DS4_N_EMBD);
    config_expect_u32("vocab_size", required_u32(m, "hyv4.vocab_size"), DS4_N_VOCAB);
    config_expect_u32("context_length", required_u32(m, "hyv4.context_length"), 1048576u);
    config_expect_u32("attention.head_count", required_u32(m, "hyv4.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", required_u32(m, "hyv4.attention.head_count_kv"), 1u);
    /* GGUF's generic lengths describe absorbed MLA; this runtime shape uses
     * the expanded 256-wide per-head projections and 512-wide latent cache. */
    config_expect_u32("attention.key_length", required_u32(m, "hyv4.attention.key_length"), 576u);
    config_expect_u32("attention.value_length", required_u32(m, "hyv4.attention.value_length"), 512u);
    config_expect_u32("attention.key_length_mla", required_u32(m, "hyv4.attention.key_length_mla"), 256u);
    config_expect_u32("attention.value_length_mla", required_u32(m, "hyv4.attention.value_length_mla"), 256u);
    config_expect_u32("attention.q_lora_rank", required_u32(m, "hyv4.attention.q_lora_rank"), 2048u);
    config_expect_u32("attention.kv_lora_rank", required_u32(m, "hyv4.attention.kv_lora_rank"), 512u);
    config_expect_u32("rope.dimension_count", required_u32(m, "hyv4.rope.dimension_count"), 64u);
    config_expect_u32("feed_forward_length", required_u32(m, "hyv4.feed_forward_length"), 18432u);
    config_expect_u32("leading_dense_block_count", required_u32(m, "hyv4.leading_dense_block_count"), 1u);
    config_expect_u32("expert_count", required_u32(m, "hyv4.expert_count"), DS4_N_EXPERT);
    config_expect_u32("expert_used_count", required_u32(m, "hyv4.expert_used_count"), DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", required_u32(m, "hyv4.expert_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count", required_u32(m, "hyv4.expert_shared_count"), 1u);
    config_expect_u32("expert_gating_func", required_u32(m, "hyv4.expert_gating_func"), 2u);
    config_expect_u32("hyper_connection.count", required_u32(m, "hyv4.hyper_connection.count"), DS4_N_HC);
    config_expect_f32("attention.layer_norm_rms_epsilon", required_f32(m, "hyv4.attention.layer_norm_rms_epsilon"), DS4_RMS_EPS);
    config_expect_f32("rope.freq_base", required_f32(m, "hyv4.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    config_expect_f32("expert_weights_scale", required_f32(m, "hyv4.expert_weights_scale"), DS4_EXPERT_WEIGHT_SCALE);
    config_expect_f32("hyper_connection.epsilon", required_f32(m, "hyv4.hyper_connection.epsilon"), DS4_HC_EPS);
    g_hy4_hc_magnitude = required_f32(m, "hyv4.hyper_connection.magnitude");
    config_expect_f32("hyper_connection.magnitude", g_hy4_hc_magnitude, 2.0f);
    bool normalized = false;
    if (!model_get_bool(m, "hyv4.expert_weights_norm", &normalized))
        ds4_die("HY4 expert_weights_norm is missing");
    config_expect_bool("expert_weights_norm", normalized, true);
    ds4_array_ref arr;
    if (!model_get_array(m, "hyv4.swiglu_clamp_exp", &arr) ||
        arr.type != GGUF_VALUE_FLOAT32 || arr.len != DS4_N_LAYER)
        ds4_die("HY4 requires one SwiGLU clamp per layer");
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        float clamp = 0;
        if (!cursor_read(&c, &clamp, sizeof(clamp))) ds4_die(c.error);
        config_expect_f32("swiglu_clamp_exp", clamp, 10.0f);
    }
    config_expect_u32("attention.indexer.head_count", required_u32(m, "hyv4.attention.indexer.head_count"), 32u);
    config_expect_u32("attention.indexer.key_length", required_u32(m, "hyv4.attention.indexer.key_length"), 128u);
    config_expect_u32("attention.indexer.top_k", required_u32(m, "hyv4.attention.indexer.top_k"), 2048u);
    if (!model_get_array(m, "hyv4.attention.indexer.is_full", &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32) || arr.len != DS4_N_LAYER)
        ds4_die("HY4 indexer layer schedule is missing or invalid");
    c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t full = 0;
        if (!cursor_u32(&c, &full) || full > 1) ds4_die("invalid HY4 indexer layer flag");
        g_hy4_indexer_is_full[il] = full != 0;
    }
    if (!g_hy4_indexer_is_full[0])
        ds4_die("HY4 layer zero must initialize the shared indexer state");
}

static void hy4_expect_matrix(const ds4_tensor *t, uint32_t ndim,
                              uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t || !hy3_dense_matrix_type_supported(t->type))
        ds4_die("HY4 dense tensor type is unsupported");
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void hy4_weights_bind(ds4_weights *w, const ds4_model *m,
                             const ds4_flash_moe_sidecar *flash_moe) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output_hc_base = required_tensor(m, "output_hc_base.weight");
    w->output_hc_fn = required_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale = required_tensor(m, "output_hc_scale.weight");
    tensor_expect_layout(w->token_embd, DS4_TENSOR_Q4_K, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    hy4_expect_matrix(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output_hc_base, DS4_TENSOR_F32, 1, 4, 0, 0);
    tensor_expect_layout(w->output_hc_fn, DS4_TENSOR_F32, 2, 4u*DS4_N_EMBD, 4, 0);
    tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32, 1, 1, 0, 0);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
#define HY4_BIND(field, name) l->field = required_tensorf(m, "blk.%u." name, il)
        HY4_BIND(attn_norm, "attn_norm.weight");
        HY4_BIND(attn_q_a, "attn_q_a.weight");
        HY4_BIND(attn_q_a_norm, "attn_q_a_norm.weight");
        HY4_BIND(attn_q_b, "attn_q_b.weight");
        HY4_BIND(attn_kv, "attn_kv_a_mqa.weight");
        HY4_BIND(attn_kv_a_norm, "attn_kv_a_norm.weight");
        HY4_BIND(attn_k_b, "attn_k_b.weight");
        HY4_BIND(attn_v_b, "attn_v_b.weight");
        HY4_BIND(attn_output, "attn_output.weight");
        HY4_BIND(attn_gate, "attn_gate.weight");
        HY4_BIND(attn_sinks, "attn_sinks.weight");
        HY4_BIND(hc_attn_fn, "hc_attn_fn.weight");
        HY4_BIND(hc_attn_scale, "hc_attn_scale.weight");
        HY4_BIND(hc_attn_base, "hc_attn_base.weight");
        HY4_BIND(hc_ffn_fn, "hc_ffn_fn.weight");
        HY4_BIND(hc_ffn_scale, "hc_ffn_scale.weight");
        HY4_BIND(hc_ffn_base, "hc_ffn_base.weight");
        HY4_BIND(ffn_norm, "ffn_norm.weight");
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->attn_q_a_norm, DS4_TENSOR_F32, 1, 2048, 0, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, 512, 0, 0);
        tensor_expect_layout(l->attn_sinks, DS4_TENSOR_F32, 1, 64, 0, 0);
        hy4_expect_matrix(l->attn_q_a, 2, DS4_N_EMBD, 2048, 0);
        hy4_expect_matrix(l->attn_q_b, 2, 2048, 16384, 0);
        hy4_expect_matrix(l->attn_kv, 2, DS4_N_EMBD, 576, 0);
        tensor_expect_layout(l->attn_k_b, DS4_TENSOR_Q8_0, 3, 192, 512, 64);
        tensor_expect_layout(l->attn_v_b, DS4_TENSOR_Q8_0, 3, 512, 256, 64);
        hy4_expect_matrix(l->attn_output, 2, 16384, DS4_N_EMBD, 0);
        hy4_expect_matrix(l->attn_gate, 2, DS4_N_EMBD, 16384, 0);
        tensor_expect_layout(l->hc_attn_fn, DS4_TENSOR_F32, 2, 4u*DS4_N_EMBD, 8, 0);
        tensor_expect_layout(l->hc_ffn_fn, DS4_TENSOR_F32, 2, 4u*DS4_N_EMBD, 8, 0);
        tensor_expect_layout(l->hc_attn_base, DS4_TENSOR_F32, 1, 8, 0, 0);
        tensor_expect_layout(l->hc_ffn_base, DS4_TENSOR_F32, 1, 8, 0, 0);
        tensor_expect_layout(l->hc_attn_scale, DS4_TENSOR_F32, 1, 2, 0, 0);
        tensor_expect_layout(l->hc_ffn_scale, DS4_TENSOR_F32, 1, 2, 0, 0);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        if (g_hy4_indexer_is_full[il]) {
            HY4_BIND(indexer_attn_q_b, "indexer.attn_q_b.weight");
            HY4_BIND(indexer_attn_k, "indexer.attn_k.weight");
            HY4_BIND(indexer_k_norm, "indexer.k_norm.weight");
            HY4_BIND(indexer_k_norm_b, "indexer.k_norm.bias");
            HY4_BIND(indexer_proj, "indexer.proj.weight");
            hy4_expect_matrix(l->indexer_attn_q_b, 2, 2048, 4096, 0);
            hy4_expect_matrix(l->indexer_attn_k, 2, DS4_N_EMBD, 128, 0);
            tensor_expect_layout(l->indexer_k_norm, DS4_TENSOR_F32, 1, 128, 0, 0);
            tensor_expect_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, 128, 0, 0);
            tensor_expect_layout(l->indexer_proj, DS4_TENSOR_F32, 2, DS4_N_EMBD, 32, 0);
        }
        if (il == 0) {
            HY4_BIND(ffn_gate, "ffn_gate.weight");
            HY4_BIND(ffn_up, "ffn_up.weight");
            HY4_BIND(ffn_down, "ffn_down.weight");
            hy4_expect_matrix(l->ffn_gate, 2, DS4_N_EMBD, 18432, 0);
            hy4_expect_matrix(l->ffn_up, 2, DS4_N_EMBD, 18432, 0);
            hy4_expect_matrix(l->ffn_down, 2, 18432, DS4_N_EMBD, 0);
        } else {
            HY4_BIND(ffn_gate_inp, "ffn_gate_inp.weight");
            HY4_BIND(ffn_exp_probs_b, "exp_probs_b.bias");
            HY4_BIND(ffn_gate_shexp, "ffn_gate_shexp.weight");
            HY4_BIND(ffn_up_shexp, "ffn_up_shexp.weight");
            HY4_BIND(ffn_down_shexp, "ffn_down_shexp.weight");
            l->ffn_gate_exps = routed_tensorf(w,m,flash_moe,il,DS4_FLASH_FAMILY_GATE,"blk.%u.ffn_gate_exps.weight");
            l->ffn_up_exps = routed_tensorf(w,m,flash_moe,il,DS4_FLASH_FAMILY_UP,"blk.%u.ffn_up_exps.weight");
            l->ffn_down_exps = routed_tensorf(w,m,flash_moe,il,DS4_FLASH_FAMILY_DOWN,"blk.%u.ffn_down_exps.weight");
            tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
            tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
            hy4_expect_matrix(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            hy4_expect_matrix(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            hy4_expect_matrix(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
            tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_up_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        }
#undef HY4_BIND
    }
}
