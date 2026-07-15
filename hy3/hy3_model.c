/* Hunyuan HY3 GGUF metadata validation and tensor binding.
 *
 * Kept at the model boundary so HY3 can use DS4's existing mmap quantized
 * matmuls and fused routed-MoE executor without duplicating the compute spine.
 */

static bool hy3_model_is_hy_v3(const ds4_model *m) {
    ds4_str arch = {0};
    return model_get_string(m, "general.architecture", &arch) &&
           ds4_streq(arch, "hy_v3");
}

typedef struct {
    ds4_tensor *eh_proj;
    ds4_tensor *enorm;
    ds4_tensor *hnorm;
    ds4_tensor *shared_head_norm;
    ds4_layer_weights block;
} hy3_mtp_weights;

static bool hy3_dense_matrix_type_supported(uint32_t type) {
    return type == DS4_TENSOR_F32 || type == DS4_TENSOR_F16 ||
           type == DS4_TENSOR_Q8_0 || type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q5_K || type == DS4_TENSOR_Q6_K;
}

static bool hy3_fused_matrix_type_supported(uint32_t type) {
    /* Keep this in lockstep with ds4_gpu_routed_mv_pipeline().  Q3_K has a
     * generic descriptor but no routed Metal pipeline in this backend. */
    return type == DS4_TENSOR_IQ2_XXS || type == DS4_TENSOR_IQ1_M ||
           type == DS4_TENSOR_IQ3_XXS || type == DS4_TENSOR_IQ4_XS ||
           type == DS4_TENSOR_Q2_K || type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q5_K || type == DS4_TENSOR_Q6_K ||
           type == DS4_TENSOR_MXFP4;
}

static void hy3_expect_matrix_type(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2,
        bool              fused) {
    if (!t) ds4_die("internal error: missing HY3 tensor");
    const bool supported = fused
        ? hy3_fused_matrix_type_supported(t->type)
        : hy3_dense_matrix_type_supported(t->type);
    if (!supported) {
        fprintf(stderr,
                "ds4: unsupported HY3 %s tensor type %s for %.*s\n",
                fused ? "fused-MoE" : "dense-matmul",
                tensor_type_name(t->type), (int)t->name.len, t->name.ptr);
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void hy3_expect_matrix(
        const ds4_tensor *t, uint32_t ndim,
        uint64_t d0, uint64_t d1, uint64_t d2) {
    hy3_expect_matrix_type(t, ndim, d0, d1, d2, false);
}

static void hy3_expect_fused_matrix(
        const ds4_tensor *t, uint32_t ndim,
        uint64_t d0, uint64_t d1, uint64_t d2) {
    hy3_expect_matrix_type(t, ndim, d0, d1, d2, true);
}

static void hy3_expect_fused_expert(
        const ds4_tensor *t, uint32_t ndim,
        uint64_t d0, uint64_t d1, uint64_t d2) {
    tensor_expect_routed_expert(t, ndim, d0, d1, d2);
    if (!hy3_fused_matrix_type_supported(t->type)) {
        fprintf(stderr,
                "ds4: unsupported HY3 routed-expert tensor type %s for %.*s\n",
                tensor_type_name(t->type), (int)t->name.len, t->name.ptr);
        exit(1);
    }
}

static void hy3_expect_gate_up_match(
        const ds4_tensor *gate, const ds4_tensor *up) {
    if (gate->type == up->type) return;
    fprintf(stderr,
            "ds4: HY3 fused gate/up tensor types differ for %.*s (%s) and "
            "%.*s (%s)\n",
            (int)gate->name.len, gate->name.ptr, tensor_type_name(gate->type),
            (int)up->name.len, up->name.ptr, tensor_type_name(up->type));
    exit(1);
}

static void hy3_config_validate_common(const ds4_model *m) {
    config_expect_u32("context_length", required_u32(m, "hy_v3.context_length"),
                      (uint32_t)DS4_ROPE_ORIG_CTX);
    config_expect_u32("embedding_length", required_u32(m, "hy_v3.embedding_length"), DS4_N_EMBD);
    config_expect_u32("attention.head_count", required_u32(m, "hy_v3.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", required_u32(m, "hy_v3.attention.head_count_kv"), DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", required_u32(m, "hy_v3.attention.key_length"), DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length", required_u32(m, "hy_v3.attention.value_length"), DS4_N_VALUE_DIM);
    config_expect_u32("feed_forward_length", required_u32(m, "hy_v3.feed_forward_length"), 13312u);
    config_expect_u32("expert_count", required_u32(m, "hy_v3.expert_count"), DS4_N_EXPERT);
    config_expect_u32("expert_used_count", required_u32(m, "hy_v3.expert_used_count"), DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", required_u32(m, "hy_v3.expert_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("expert_shared_feed_forward_length",
                      required_u32(m, "hy_v3.expert_shared_feed_forward_length"), DS4_N_FF_EXP);
    config_expect_u32("expert_gating_func",
                      required_u32(m, "hy_v3.expert_gating_func"), 2u);
    config_expect_f32("attention.layer_norm_rms_epsilon",
                      required_f32(m, "hy_v3.attention.layer_norm_rms_epsilon"), DS4_RMS_EPS);
    config_expect_f32("rope.freq_base", required_f32(m, "hy_v3.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    config_expect_f32("expert_weights_scale", required_f32(m, "hy_v3.expert_weights_scale"), DS4_EXPERT_WEIGHT_SCALE);

    bool normalized = false;
    if (!model_get_bool(m, "hy_v3.expert_weights_norm", &normalized)) {
        ds4_die("required metadata key is missing or not bool: hy_v3.expert_weights_norm");
    }
    config_expect_bool("expert_weights_norm", normalized, true);
}

static void hy3_config_validate_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_HY3;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));

    config_expect_u32("block_count", required_u32(m, "hy_v3.block_count"), DS4_N_LAYER);
    hy3_config_validate_common(m);
}

static void hy3_weights_validate_layout(const ds4_weights *w) {
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;

    /* The HY3 embedding kernels decode Q4_K rows directly; reject other
     * otherwise-supported matrix types during model open, not at first eval. */
    tensor_expect_layout(w->token_embd, DS4_TENSOR_Q4_K, 2,
                         DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    hy3_expect_matrix(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        hy3_expect_matrix(l->attn_q, 2, DS4_N_EMBD, q_dim, 0);
        hy3_expect_matrix(l->attn_k, 2, DS4_N_EMBD, kv_dim, 0);
        hy3_expect_matrix(l->attn_v, 2, DS4_N_EMBD, kv_dim, 0);
        tensor_expect_layout(l->attn_q_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_k_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        hy3_expect_matrix(l->attn_output, 2, q_dim, DS4_N_EMBD, 0);
        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);

        if (il == 0) {
            hy3_expect_fused_matrix(l->ffn_gate, 2, DS4_N_EMBD, 13312u, 0);
            hy3_expect_fused_matrix(l->ffn_up, 2, DS4_N_EMBD, 13312u, 0);
            hy3_expect_fused_matrix(l->ffn_down, 2, 13312u, DS4_N_EMBD, 0);
            hy3_expect_gate_up_match(l->ffn_gate, l->ffn_up);
        } else {
            tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
            tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
            hy3_expect_fused_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            hy3_expect_fused_expert(l->ffn_up_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            hy3_expect_fused_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
            hy3_expect_gate_up_match(l->ffn_gate_exps, l->ffn_up_exps);
            hy3_expect_fused_matrix(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            hy3_expect_fused_matrix(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            hy3_expect_fused_matrix(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
            hy3_expect_gate_up_match(l->ffn_gate_shexp, l->ffn_up_shexp);
        }
    }
}

static void hy3_weights_bind(
        ds4_weights                 *w,
        const ds4_model             *m,
        const ds4_flash_moe_sidecar *flash_moe) {
    (void)flash_moe;
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
        l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
        l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
        l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
        l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
        l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
        l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);

        if (il == 0) {
            l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
            l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
            l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
        } else {
            l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
            l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b", il);
            if (!l->ffn_exp_probs_b) {
                l->ffn_exp_probs_b = required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
            }
            l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
            l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
            l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
            l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
            l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
            l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
        }
    }

    hy3_weights_validate_layout(w);
}

static void hy3_mtp_weights_validate_layout(const hy3_mtp_weights *w) {
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const ds4_layer_weights *l = &w->block;

    tensor_expect_layout(w->enorm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->shared_head_norm, DS4_TENSOR_F32, 1,
                         DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->eh_proj, DS4_TENSOR_Q8_0, 2,
                         2u * DS4_N_EMBD, DS4_N_EMBD, 0);

    tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    hy3_expect_matrix(l->attn_q, 2, DS4_N_EMBD, q_dim, 0);
    hy3_expect_matrix(l->attn_k, 2, DS4_N_EMBD, kv_dim, 0);
    hy3_expect_matrix(l->attn_v, 2, DS4_N_EMBD, kv_dim, 0);
    tensor_expect_layout(l->attn_q_norm, DS4_TENSOR_F32, 1,
                         DS4_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_k_norm, DS4_TENSOR_F32, 1,
                         DS4_N_HEAD_DIM, 0, 0);
    hy3_expect_matrix(l->attn_output, 2, q_dim, DS4_N_EMBD, 0);
    tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2,
                         DS4_N_EMBD, DS4_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1,
                         DS4_N_EXPERT, 0, 0);
    hy3_expect_fused_expert(l->ffn_gate_exps, 3,
                            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    hy3_expect_fused_expert(l->ffn_up_exps, 3,
                            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    hy3_expect_fused_expert(l->ffn_down_exps, 3,
                            DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    hy3_expect_gate_up_match(l->ffn_gate_exps, l->ffn_up_exps);
    hy3_expect_fused_matrix(l->ffn_gate_shexp, 2,
                            DS4_N_EMBD, DS4_N_FF_EXP, 0);
    hy3_expect_fused_matrix(l->ffn_up_shexp, 2,
                            DS4_N_EMBD, DS4_N_FF_EXP, 0);
    hy3_expect_fused_matrix(l->ffn_down_shexp, 2,
                            DS4_N_FF_EXP, DS4_N_EMBD, 0);
    hy3_expect_gate_up_match(l->ffn_gate_shexp, l->ffn_up_shexp);
}

static void hy3_mtp_weights_bind(hy3_mtp_weights *w, const ds4_model *m) {
    if (!hy3_model_is_hy_v3(m)) {
        ds4_die("HY3 MTP GGUF must use general.architecture=hy_v3");
    }
    config_expect_u32("MTP block_count",
                      required_u32(m, "hy_v3.block_count"), DS4_N_LAYER + 1u);
    hy3_config_validate_common(m);
    config_expect_u32("MTP nextn_predict_layers",
                      required_u32(m, "hy_v3.nextn_predict_layers"), 1u);

    memset(w, 0, sizeof(*w));
    w->eh_proj = required_tensor(m, "blk.80.nextn.eh_proj.weight");
    w->enorm = required_tensor(m, "blk.80.nextn.enorm.weight");
    w->hnorm = required_tensor(m, "blk.80.nextn.hnorm.weight");
    w->shared_head_norm = required_tensor(
        m, "blk.80.nextn.shared_head_norm.weight");

    ds4_layer_weights *l = &w->block;
    l->attn_norm = required_tensor(m, "blk.80.attn_norm.weight");
    l->attn_q = required_tensor(m, "blk.80.attn_q.weight");
    l->attn_q_norm = required_tensor(m, "blk.80.attn_q_norm.weight");
    l->attn_k = required_tensor(m, "blk.80.attn_k.weight");
    l->attn_k_norm = required_tensor(m, "blk.80.attn_k_norm.weight");
    l->attn_v = required_tensor(m, "blk.80.attn_v.weight");
    l->attn_output = required_tensor(m, "blk.80.attn_output.weight");
    l->ffn_norm = required_tensor(m, "blk.80.ffn_norm.weight");
    l->ffn_gate_inp = required_tensor(m, "blk.80.ffn_gate_inp.weight");
    l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b", 80u);
    if (!l->ffn_exp_probs_b) {
        l->ffn_exp_probs_b = required_tensor(m, "blk.80.exp_probs_b.bias");
    }
    l->ffn_gate_exps = required_tensor(m, "blk.80.ffn_gate_exps.weight");
    l->ffn_up_exps = required_tensor(m, "blk.80.ffn_up_exps.weight");
    l->ffn_down_exps = required_tensor(m, "blk.80.ffn_down_exps.weight");
    l->ffn_gate_shexp = required_tensor(m, "blk.80.ffn_gate_shexp.weight");
    l->ffn_up_shexp = required_tensor(m, "blk.80.ffn_up_shexp.weight");
    l->ffn_down_shexp = required_tensor(m, "blk.80.ffn_down_shexp.weight");

    hy3_mtp_weights_validate_layout(w);
}
