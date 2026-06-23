/* =========================================================================
 * glm52_model.c - GLM-5.2 GGUF metadata and tensor binding.
 * =========================================================================
 *
 * This file is included by ds4.c so it can reuse the mmap GGUF helpers and
 * fixed tensor structs without exporting a second internal API.
 */

static bool glm52_model_is_glm_dsa(const ds4_model *m) {
    ds4_str arch = {0};
    return model_get_string(m, "general.architecture", &arch) &&
           ds4_streq(arch, "glm-dsa");
}

static void glm52_expect_tensor_type_any(
        const ds4_tensor *t,
        const uint32_t   *types,
        size_t            n_type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing GLM tensor while validating layout");
    bool ok = false;
    for (size_t i = 0; i < n_type; i++) {
        if (t->type == types[i]) {
            ok = true;
            break;
        }
    }
    if (!ok) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected a supported GLM dense quant\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void glm52_expect_dense_matrix(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    static const uint32_t types[] = {
        DS4_TENSOR_Q4_K,
        DS4_TENSOR_Q5_K,
        DS4_TENSOR_Q6_K,
        DS4_TENSOR_Q8_0,
        DS4_TENSOR_F16,
        DS4_TENSOR_F32,
    };
    glm52_expect_tensor_type_any(t, types, sizeof(types) / sizeof(types[0]), ndim, d0, d1, d2);
}

static void glm52_config_validate_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_GLM52;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));

    const uint32_t n_layer = required_u32(m, "glm-dsa.block_count");
    const uint32_t n_embd = required_u32(m, "glm-dsa.embedding_length");
    const uint32_t n_vocab = required_u32(m, "glm-dsa.vocab_size");
    const uint32_t n_head = required_u32(m, "glm-dsa.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "glm-dsa.attention.head_count_kv");
    const uint32_t n_key_dim = required_u32(m, "glm-dsa.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "glm-dsa.attention.value_length");
    const uint32_t n_rot = required_u32(m, "glm-dsa.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "glm-dsa.attention.q_lora_rank");
    const uint32_t n_lora_kv = required_u32(m, "glm-dsa.attention.kv_lora_rank");
    const uint32_t n_expert = required_u32(m, "glm-dsa.expert_count");
    const uint32_t n_expert_used = required_u32(m, "glm-dsa.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "glm-dsa.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "glm-dsa.expert_shared_count");
    const uint32_t n_dense_lead = required_u32(m, "glm-dsa.leading_dense_block_count");
    const uint32_t n_indexer_head = required_u32(m, "glm-dsa.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "glm-dsa.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "glm-dsa.attention.indexer.top_k");
    uint32_t n_nextn = 0;
    model_get_u32(m, "glm-dsa.nextn_predict_layers", &n_nextn);

    config_expect_u32("block_count", n_layer, DS4_N_LAYER);
    config_expect_u32("embedding_length", n_embd, DS4_N_EMBD);
    config_expect_u32("vocab_size", n_vocab, DS4_N_VOCAB);
    config_expect_u32("attention.head_count", n_head, DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", n_head_kv, DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", n_key_dim, DS4_N_LORA_KV + DS4_N_ROT);
    config_expect_u32("attention.value_length", n_value_dim, DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count", n_rot, DS4_N_ROT);
    config_expect_u32("attention.q_lora_rank", n_lora_q, DS4_N_LORA_Q);
    config_expect_u32("attention.kv_lora_rank", n_lora_kv, DS4_N_LORA_KV);
    config_expect_u32("expert_count", n_expert, DS4_N_EXPERT);
    config_expect_u32("expert_used_count", n_expert_used, DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp, DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count", n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("leading_dense_block_count", n_dense_lead, DS4_N_DENSE_LEAD);
    config_expect_u32("nextn_predict_layers", n_nextn, DS4_N_NEXTN);
    config_expect_u32("attention.indexer.head_count", n_indexer_head, DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k", n_indexer_top_k, DS4_N_INDEXER_TOP_K);

    const float rope_freq_base = required_f32(m, "glm-dsa.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "glm-dsa.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "glm-dsa.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);

    bool expert_weight_norm = true;
    if (model_get_bool(m, "glm-dsa.expert_weights_norm", &expert_weight_norm)) {
        config_expect_bool("expert_weights_norm", expert_weight_norm, true);
    }
}

static void glm52_weights_validate_layout(const ds4_weights *w) {
    const uint64_t k_nope_dim = 192u;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * (k_nope_dim + DS4_N_ROT);
    const uint64_t kv_a_dim = (uint64_t)DS4_N_LORA_KV + DS4_N_ROT;
    const uint64_t v_impl_dim = 256u;
    const uint64_t dense_ff = 12288u;
    const uint32_t n_effective_layer = DS4_N_LAYER - DS4_N_NEXTN;

    glm52_expect_dense_matrix(w->token_embd, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
    glm52_expect_dense_matrix(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < n_effective_layer; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        tensor_expect_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        glm52_expect_dense_matrix(l->attn_q_a, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0, 0);
        glm52_expect_dense_matrix(l->attn_q_b, 2, DS4_N_LORA_Q, q_dim, 0);
        glm52_expect_dense_matrix(l->attn_kv, 2, DS4_N_EMBD, kv_a_dim, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_KV, 0, 0);
        glm52_expect_dense_matrix(l->attn_k_b, 3, k_nope_dim, DS4_N_LORA_KV, DS4_N_HEAD);
        glm52_expect_dense_matrix(l->attn_v_b, 3, DS4_N_LORA_KV, v_impl_dim, DS4_N_HEAD);
        glm52_expect_dense_matrix(l->attn_output_a, 2, (uint64_t)DS4_N_HEAD * v_impl_dim, DS4_N_EMBD, 0);

        tensor_expect_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        glm52_expect_dense_matrix(l->indexer_attn_k, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, 0);
        tensor_expect_layout(l->indexer_k_norm, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        glm52_expect_dense_matrix(l->indexer_proj, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
        glm52_expect_dense_matrix(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM, 0);

        if (il < DS4_N_DENSE_LEAD) {
            glm52_expect_dense_matrix(l->ffn_gate, 2, DS4_N_EMBD, dense_ff, 0);
            glm52_expect_dense_matrix(l->ffn_up, 2, DS4_N_EMBD, dense_ff, 0);
            glm52_expect_dense_matrix(l->ffn_down, 2, dense_ff, DS4_N_EMBD, 0);
        } else {
            tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
            tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
            tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_up_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
            glm52_expect_dense_matrix(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP * DS4_N_EXPERT_SHARED, 0);
            glm52_expect_dense_matrix(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP * DS4_N_EXPERT_SHARED, 0);
            glm52_expect_dense_matrix(l->ffn_down_shexp, 2, DS4_N_FF_EXP * DS4_N_EXPERT_SHARED, DS4_N_EMBD, 0);
        }
    }
}

static void glm52_weights_bind(
        ds4_weights                  *w,
        const ds4_model              *m,
        const ds4_flash_moe_sidecar  *flash_moe) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");

    const uint32_t n_effective_layer = DS4_N_LAYER - DS4_N_NEXTN;
    for (uint32_t il = 0; il < n_effective_layer; il++) {
        ds4_layer_weights *l = &w->layer[il];

        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv = required_tensorf(m, "blk.%u.attn_kv_a_mqa.weight", il);
        l->attn_kv_a_norm = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_k_b = required_tensorf(m, "blk.%u.attn_k_b.weight", il);
        l->attn_v_b = required_tensorf(m, "blk.%u.attn_v_b.weight", il);
        l->attn_output_a = required_tensorf(m, "blk.%u.attn_output.weight", il);

        l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->indexer_attn_k = required_tensorf(m, "blk.%u.indexer.attn_k.weight", il);
        l->indexer_k_norm = required_tensorf(m, "blk.%u.indexer.k_norm.weight", il);
        l->indexer_k_norm_b = required_tensorf(m, "blk.%u.indexer.k_norm.bias", il);
        l->indexer_proj = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
        l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);

        if (il < DS4_N_DENSE_LEAD) {
            l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
            l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
            l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
        } else {
            l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
            l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
            l->ffn_gate_exps = routed_tensorf(w, m, flash_moe, il, DS4_FLASH_FAMILY_GATE, "blk.%u.ffn_gate_exps.weight");
            l->ffn_up_exps = routed_tensorf(w, m, flash_moe, il, DS4_FLASH_FAMILY_UP, "blk.%u.ffn_up_exps.weight");
            l->ffn_down_exps = routed_tensorf(w, m, flash_moe, il, DS4_FLASH_FAMILY_DOWN, "blk.%u.ffn_down_exps.weight");
            l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
            l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
            l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
        }
    }

    glm52_weights_validate_layout(w);
}
