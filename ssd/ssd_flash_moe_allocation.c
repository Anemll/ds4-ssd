/* =========================================================================
 * ssd_flash_moe_allocation.c - Flash-MoE Metal graph allocation and sidecar enablement.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool metal_graph_batch_shared_expert_uses_ane(void) {
    const char *e = getenv("DS4_FLASH_MOE_ANE_SHARED_EXPERT");
    const char *skip = getenv("DS4_FLASH_MOE_SKIP_SHARED_EXPERT");
    const bool ane_on = e && e[0] && atoi(e) != 0;
    const bool skip_on = skip && skip[0] && atoi(skip) != 0;
    return ane_on && !skip_on;
}

static bool metal_graph_resident_moe_compact_scratch_requested(void) {
    const char *compact = getenv("DS4_RESIDENT_MOE_COMPACT_SCRATCH");
    const char *ane = getenv("DS4_RESIDENT_MOE_ANE_HYBRID");
    return (compact && compact[0] && atoi(compact) != 0) ||
           (ane && ane[0] && atoi(ane) != 0);
}

static uint32_t metal_graph_resident_moe_scratch_cap_for_prefill(uint32_t prefill_cap) {
    if (prefill_cap == 0) return 1;
    if (!metal_graph_resident_moe_compact_scratch_requested()) return prefill_cap;

    uint32_t cap = prefill_cap < 2048u ? prefill_cap : 2048u;
    const char *env = getenv("DS4_RESIDENT_MOE_SCRATCH_CAP");
    if (!env || !env[0]) env = getenv("DS4_RESIDENT_MOE_COMPACT_SCRATCH_CAP");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed > 0 && parsed <= UINT32_MAX) {
            cap = (uint32_t)parsed;
        }
    }
    if (cap == 0) cap = 1;
    if (cap > prefill_cap) cap = prefill_cap;
    return cap;
}

static bool env_flag_enabled(const char *name);

static bool metal_graph_prefill_memory_flush_requested(uint32_t n_tokens) {
    if (env_flag_enabled("DS4_METAL_PREFILL_STAGE_FLUSH")) return true;
    if (env_flag_enabled("DS4_METAL_NO_PREFILL_STAGE_FLUSH")) return false;
    return metal_graph_resident_moe_compact_scratch_requested() && n_tokens > 2048u;
}

static bool metal_graph_release_prefill_scratch_on_decode_requested(void) {
    return env_flag_enabled("DS4_METAL_RELEASE_PREFILL_SCRATCH_ON_DECODE");
}

static uint64_t metal_graph_release_prefill_scratch(ds4_gpu_graph *g) {
    if (!g) return 0;
    uint64_t released = 0;
#define DS4_RELEASE_PREFILL_TENSOR(name) do {                          \
        if (g->name) {                                                  \
            released += ds4_gpu_tensor_bytes(g->name);                  \
            ds4_gpu_tensor_free(g->name);                               \
            g->name = NULL;                                             \
        }                                                               \
    } while (0)

    DS4_RELEASE_PREFILL_TENSOR(batch_ffn_out);
    DS4_RELEASE_PREFILL_TENSOR(batch_routed_out);
    DS4_RELEASE_PREFILL_TENSOR(batch_routed_down);
    DS4_RELEASE_PREFILL_TENSOR(batch_routed_mid);
    DS4_RELEASE_PREFILL_TENSOR(batch_routed_up);
    DS4_RELEASE_PREFILL_TENSOR(batch_routed_gate);
    DS4_RELEASE_PREFILL_TENSOR(batch_router_weights);
    DS4_RELEASE_PREFILL_TENSOR(batch_router_selected);
    DS4_RELEASE_PREFILL_TENSOR(batch_router_probs);
    DS4_RELEASE_PREFILL_TENSOR(batch_router_logits);
    DS4_RELEASE_PREFILL_TENSOR(batch_shared_out);
    DS4_RELEASE_PREFILL_TENSOR(batch_shared_mid);
    DS4_RELEASE_PREFILL_TENSOR(batch_shared_up);
    DS4_RELEASE_PREFILL_TENSOR(batch_shared_gate);
    DS4_RELEASE_PREFILL_TENSOR(batch_ffn_norm);
    DS4_RELEASE_PREFILL_TENSOR(batch_ffn_cur);
    DS4_RELEASE_PREFILL_TENSOR(batch_after_attn_hc);
    DS4_RELEASE_PREFILL_TENSOR(batch_low_tmp);
    DS4_RELEASE_PREFILL_TENSOR(batch_group_tmp);
    DS4_RELEASE_PREFILL_TENSOR(batch_attn_out);
    DS4_RELEASE_PREFILL_TENSOR(batch_attn_low);
    DS4_RELEASE_PREFILL_TENSOR(batch_heads);
    DS4_RELEASE_PREFILL_TENSOR(batch_indexer_weights);
    DS4_RELEASE_PREFILL_TENSOR(batch_indexer_q);
    DS4_RELEASE_PREFILL_TENSOR(batch_comp_sc);
    DS4_RELEASE_PREFILL_TENSOR(batch_comp_kv);
    DS4_RELEASE_PREFILL_TENSOR(batch_kv);
    DS4_RELEASE_PREFILL_TENSOR(batch_kv_raw);
    DS4_RELEASE_PREFILL_TENSOR(batch_q);
    DS4_RELEASE_PREFILL_TENSOR(batch_qr_norm);
    DS4_RELEASE_PREFILL_TENSOR(batch_qr);
    DS4_RELEASE_PREFILL_TENSOR(batch_attn_norm);
    DS4_RELEASE_PREFILL_TENSOR(batch_attn_cur);
    DS4_RELEASE_PREFILL_TENSOR(batch_hc_split);
    DS4_RELEASE_PREFILL_TENSOR(batch_hc_mix);
    DS4_RELEASE_PREFILL_TENSOR(batch_flat_hc);
    DS4_RELEASE_PREFILL_TENSOR(batch_next_hc);
    DS4_RELEASE_PREFILL_TENSOR(batch_cur_hc);
    DS4_RELEASE_PREFILL_TENSOR(prefill_tokens);

#undef DS4_RELEASE_PREFILL_TENSOR
    g->batch_routed_mid_is_f16 = false;
    g->batch_routed_compact_rows = 0;
    return released;
}

static void metal_graph_release_prefill_scratch_before_decode(
        ds4_gpu_graph *g,
        bool           memory_report) {
    if (!metal_graph_release_prefill_scratch_on_decode_requested()) return;
    const uint64_t released = metal_graph_release_prefill_scratch(g);
    ds4_gpu_release_i8_prefill_cache();
    ds4_gpu_release_prefill_transients();
    if (released && env_flag_enabled("DS4_METAL_RELEASE_PREFILL_SCRATCH_TRACE")) {
        fprintf(stderr,
                "ds4: released %.2f MiB of Metal prefill scratch before decode\n",
                (double)released / 1048576.0);
    }
    if (memory_report) ds4_gpu_print_memory_report("after releasing prefill scratch");
}

static bool metal_graph_ensure_prefill_scratch(
        ds4_gpu_graph           *g,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer) {
    if (!g || !weights || !layer) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = g->prefill_cap ? g->prefill_cap : 1u;
    uint64_t moe_pc = g->batch_routed_scratch_cap;
    if (moe_pc == 0) {
        moe_pc = metal_graph_resident_moe_scratch_cap_for_prefill((uint32_t)pc);
        g->batch_routed_scratch_cap = (uint32_t)moe_pc;
    }
    const bool ane_shared_batch = metal_graph_batch_shared_expert_uses_ane();

#define DS4_ENSURE_PREFILL_TENSOR(name, bytes) do {                    \
        if (!g->name) g->name = ds4_gpu_tensor_alloc((bytes));          \
    } while (0)

    DS4_ENSURE_PREFILL_TENSOR(prefill_tokens, pc * sizeof(int32_t));
    DS4_ENSURE_PREFILL_TENSOR(batch_cur_hc, pc * hc_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_next_hc, pc * hc_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_flat_hc, pc * hc_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_hc_mix, pc * mix_hc * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_hc_split, pc * mix_hc * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_attn_cur, pc * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_attn_norm, pc * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_qr, pc * q_rank * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_qr_norm, pc * q_rank * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_q, pc * q_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_kv_raw, pc * DS4_N_HEAD_DIM * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_kv, pc * DS4_N_HEAD_DIM * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_comp_kv, pc * comp_width_max * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_comp_sc, pc * comp_width_max * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_indexer_q, pc * indexer_q_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_indexer_weights, pc * DS4_N_INDEXER_HEAD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_heads, pc * q_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_attn_low, pc * low_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_attn_out, pc * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_group_tmp, pc * group_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_low_tmp, pc * DS4_N_LORA_O * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_after_attn_hc, pc * hc_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_ffn_cur, pc * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_ffn_norm, pc * DS4_N_EMBD * sizeof(float));
    if (!ane_shared_batch) {
        DS4_ENSURE_PREFILL_TENSOR(batch_shared_gate, pc * shared_dim * sizeof(float));
        DS4_ENSURE_PREFILL_TENSOR(batch_shared_up, pc * shared_dim * sizeof(float));
        DS4_ENSURE_PREFILL_TENSOR(batch_shared_mid, pc * shared_dim * sizeof(float));
    }
    DS4_ENSURE_PREFILL_TENSOR(batch_shared_out, pc * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_router_logits, pc * DS4_N_EXPERT * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_router_probs, pc * DS4_N_EXPERT * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_router_selected, pc * DS4_N_EXPERT_USED * sizeof(int));
    DS4_ENSURE_PREFILL_TENSOR(batch_router_weights, pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_compact_rows = (uint32_t)(moe_pc * DS4_N_EXPERT_USED);
    DS4_ENSURE_PREFILL_TENSOR(batch_routed_gate, moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_routed_up, moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_routed_mid, moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_routed_down, moe_pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    DS4_ENSURE_PREFILL_TENSOR(batch_routed_out, pc * DS4_N_EMBD * sizeof(float));

#undef DS4_ENSURE_PREFILL_TENSOR

    const bool ok =
        g->prefill_tokens &&
        g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
        g->batch_hc_mix && g->batch_hc_split &&
        g->batch_attn_cur && g->batch_attn_norm &&
        g->batch_qr && g->batch_qr_norm && g->batch_q &&
        g->batch_kv_raw && g->batch_kv &&
        g->batch_comp_kv && g->batch_comp_sc &&
        g->batch_indexer_q && g->batch_indexer_weights &&
        g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
        g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
        g->batch_ffn_cur && g->batch_ffn_norm &&
        (ane_shared_batch ||
         (g->batch_shared_gate && g->batch_shared_up && g->batch_shared_mid)) &&
        g->batch_shared_out &&
        g->batch_router_logits && g->batch_router_probs &&
        g->batch_router_selected && g->batch_router_weights &&
        g->batch_routed_gate && g->batch_routed_up &&
        g->batch_routed_mid && g->batch_routed_down &&
        g->batch_routed_out;
    if (!ok) fprintf(stderr, "ds4: failed to allocate Metal prefill scratch\n");
    return ok;
}

static bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp) {
    memset(g, 0, sizeof(*g));
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    const bool ctx_grow = ds4_ctx_grow_enabled();
    const uint32_t grow_block = ds4_ctx_grow_block();
    g->comp_min_ratio = min_ratio;
    g->comp_cap_max = ctx_size / min_ratio + 2u;
    if (g->comp_cap_max < 2u) g->comp_cap_max = 2u;
    if (ctx_grow) {
        /* Shared comp_cap-sized scratch (indexer_scores, comp_mask) is transient
         * and resized without copy as the context fills; start at one block. */
        uint32_t init_shared = grow_block / min_ratio + 2u;
        if (init_shared < 2u) init_shared = 2u;
        if (init_shared > g->comp_cap_max) init_shared = g->comp_cap_max;
        g->comp_cap = init_shared;
    } else {
        g->comp_cap = g->comp_cap_max;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
            g->layer_comp_cap_max[il] = 0;
        } else {
            uint32_t ceiling = ctx_size / ratio + 2u;
            if (ceiling < 2u) ceiling = 2u;
            g->layer_comp_cap_max[il] = ceiling;
            if (ctx_grow) {
                /* Start at one block; grow toward the ceiling as context fills. */
                uint32_t init_cap = grow_block / ratio + 2u;
                if (init_cap < 2u) init_cap = 2u;
                if (init_cap > ceiling) init_cap = ceiling;
                g->layer_comp_cap[il] = init_cap;
            } else {
                g->layer_comp_cap[il] = ceiling;
            }
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    const bool ane_shared_batch = metal_graph_batch_shared_expert_uses_ane();
    const uint64_t moe_pc = metal_graph_resident_moe_scratch_cap_for_prefill(prefill_cap);
    g->batch_routed_scratch_cap = (uint32_t)moe_pc;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        metal_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    g->context_buffer_bytes = context_bytes;
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    g->kv_cache_managed = managed_kv_cache;
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM * sizeof(float));
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_mtp) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_mtp) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
        g->mtp_n_raw = 0;
    }

    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    if (!ane_shared_batch) {
        g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
        g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
        g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    }
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_compact_rows = (uint32_t)(moe_pc * DS4_N_EXPERT_USED);
    g->batch_routed_gate = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(moe_pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL &&
                               g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL &&
                               g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache && g->spec_logits)) &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    (ane_shared_batch ||
                     (g->batch_shared_gate && g->batch_shared_up &&
                      g->batch_shared_mid)) &&
                    g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) metal_graph_free(g);
    return ok;
}

static bool metal_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false);
}

extern int ds4_gpu_use_m5_simdgroup_matrix(void);
static int get_prefill_dedup_prefetch(void);
static int get_prefill_slot_cache_topk(uint32_t slot_bank);
static void metal_graph_log_prefill_compute_once(const ds4_gpu_graph *g,
                                                 uint32_t slot_bank,
                                                 const ds4_layer_weights *layer);
static bool flash_moe_mixed_slot_bank_enabled(void);
static bool flash_moe_layer_slot_slab_enabled(void);
static bool flash_moe_slot_bank_residency_enabled(void);
static bool flash_moe_slot_bank_touch_pages_enabled(void);
static bool flash_moe_baked_slot_decode_enabled(void);
static bool flash_moe_per_expert_buffers_enabled(void);
static bool flash_moe_per_slot_buffers_enabled(void);
static bool flash_moe_auto_per_slot_buffers_enabled(const ds4_flash_moe_sidecar *sidecar);
static bool flash_moe_per_slot_lazy_alloc_enabled(void);
static bool flash_moe_record_table_enabled(void);
static bool flash_moe_untracked_slot_bank_enabled(void);
static bool flash_moe_direct_mmap_bank_enabled(void);
static bool flash_moe_direct_mmap_auto_enabled(const ds4_flash_moe_sidecar *sidecar,
                                               const ds4_layer_weights     *layer);
static bool flash_moe_direct_mmap_slots6_enabled(void);
static bool flash_moe_direct_mmap_record_slots6_enabled(void);
static bool flash_moe_direct_mmap_prewarm_views_enabled(void);
static bool flash_moe_active_staging_enabled(void);
static bool flash_moe_chunked_mixed_enabled(void);
static uint32_t flash_moe_chunked_mixed_slots(uint32_t slot_bank);
static bool flash_moe_decode_bank_shrink_requested(void);
static bool flash_moe_fast_decode_l1_enabled(void);
static bool flash_moe_prefill_decode_l1_enabled(void);
static uint64_t flash_moe_decode_l1_budget_bytes(void);
static uint32_t flash_moe_decode_slot_bank_for_budget(
        const ds4_flash_moe_sidecar *sidecar,
        uint32_t                     min_slots,
        uint32_t                     cur,
        uint64_t                     budget);
static bool flash_moe_decode_l2_enabled(void);
static bool flash_moe_shrink_carry_enabled(void);
static bool flash_moe_mixed_slots6_grouped_enabled(void);
static int flash_moe_cache_io_split(void);
static bool flash_moe_pread_split(int fd, uint64_t offset, uint8_t *dst,
                                  uint64_t bytes, int want);
static uint32_t flash_moe_decode_prefetch_max_loads(void);
static uint32_t flash_moe_decode_slot_bank_target(const ds4_gpu_graph *g);

static bool metal_graph_flash_moe_prepare_slot_bank_owner(
        ds4_gpu_tensor *tensor,
        const char     *label,
        bool            use_residency,
        bool            touch_pages,
        uint64_t       *touched_bytes,
        uint32_t       *touched_buffers) {
    if (!tensor) return false;
    const uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    if (use_residency && !ds4_gpu_flash_slot_bank_residency_add(tensor)) {
        fprintf(stderr, "ds4: failed to add Flash-MoE slot-bank buffer to Metal residency set: %s\n",
                label ? label : "<unnamed>");
        return false;
    }
    if (touch_pages) {
        if (!ds4_gpu_tensor_touch_pages(tensor, 0)) {
            fprintf(stderr, "ds4: failed to touch Flash-MoE slot-bank pages: %s\n",
                    label ? label : "<unnamed>");
            return false;
        }
        if (touched_bytes) *touched_bytes += bytes;
        if (touched_buffers) (*touched_buffers)++;
    }
    return true;
}

static ds4_gpu_tensor *metal_graph_flash_moe_alloc_slot_bank_tensor(uint64_t bytes) {
    return flash_moe_untracked_slot_bank_enabled() ?
           ds4_gpu_tensor_alloc_untracked(bytes) :
           ds4_gpu_tensor_alloc(bytes);
}

static void metal_graph_flash_moe_free_expert_family_views(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       slot) {
    if (!g || il >= DS4_N_LAYER || slot >= DS4_MAX_EXPERT) return;
    ds4_gpu_tensor_free(g->flash_expert_down_view[il][slot]);
    ds4_gpu_tensor_free(g->flash_expert_up_view[il][slot]);
    ds4_gpu_tensor_free(g->flash_expert_gate_view[il][slot]);
    g->flash_expert_down_view[il][slot] = NULL;
    g->flash_expert_up_view[il][slot] = NULL;
    g->flash_expert_gate_view[il][slot] = NULL;
}

static bool metal_graph_flash_moe_init_expert_family_views(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       slot) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER || slot >= DS4_MAX_EXPERT) {
        return false;
    }
    ds4_gpu_tensor *expert = g->flash_expert_bank[il][slot];
    if (!expert) return false;
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    metal_graph_flash_moe_free_expert_family_views(g, il, slot);
    g->flash_expert_gate_view[il][slot] =
        ds4_gpu_tensor_view(expert,
                            layer->family_offset[DS4_FLASH_FAMILY_GATE],
                            layer->family_bytes[DS4_FLASH_FAMILY_GATE]);
    g->flash_expert_up_view[il][slot] =
        ds4_gpu_tensor_view(expert,
                            layer->family_offset[DS4_FLASH_FAMILY_UP],
                            layer->family_bytes[DS4_FLASH_FAMILY_UP]);
    g->flash_expert_down_view[il][slot] =
        ds4_gpu_tensor_view(expert,
                            layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                            layer->family_bytes[DS4_FLASH_FAMILY_DOWN]);
    const bool ok =
        g->flash_expert_gate_view[il][slot] &&
        g->flash_expert_up_view[il][slot] &&
        g->flash_expert_down_view[il][slot];
    if (!ok) metal_graph_flash_moe_free_expert_family_views(g, il, slot);
    return ok;
}

static bool metal_graph_flash_moe_ensure_direct_mmap_family_views(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       expert) {
    if (!g || !g->flash_moe || !g->flash_direct_mmap_bank ||
        il >= DS4_N_LAYER || expert >= DS4_N_EXPERT ||
        expert >= DS4_MAX_EXPERT) {
        return false;
    }
    if (g->flash_expert_gate_view[il][expert] &&
        g->flash_expert_up_view[il][expert] &&
        g->flash_expert_down_view[il][expert]) {
        return true;
    }

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (!layer->map || layer->map_size == 0 ||
        layer->expert_stride > UINT64_MAX / (uint64_t)expert) {
        return false;
    }
    const uint64_t record_base = (uint64_t)expert * layer->expert_stride;
    metal_graph_flash_moe_free_expert_family_views(g, il, expert);
    g->flash_expert_gate_view[il][expert] =
        ds4_gpu_mmap_tensor_view(layer->map,
                                 layer->map_size,
                                 record_base + layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                 layer->family_bytes[DS4_FLASH_FAMILY_GATE]);
    g->flash_expert_up_view[il][expert] =
        ds4_gpu_mmap_tensor_view(layer->map,
                                 layer->map_size,
                                 record_base + layer->family_offset[DS4_FLASH_FAMILY_UP],
                                 layer->family_bytes[DS4_FLASH_FAMILY_UP]);
    g->flash_expert_down_view[il][expert] =
        ds4_gpu_mmap_tensor_view(layer->map,
                                 layer->map_size,
                                 record_base + layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                 layer->family_bytes[DS4_FLASH_FAMILY_DOWN]);
    const bool ok =
        g->flash_expert_gate_view[il][expert] &&
        g->flash_expert_up_view[il][expert] &&
        g->flash_expert_down_view[il][expert];
    if (!ok) metal_graph_flash_moe_free_expert_family_views(g, il, expert);
    return ok;
}

static bool metal_graph_flash_moe_ensure_direct_mmap_record_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       expert) {
    if (!g || !g->flash_moe || !g->flash_direct_mmap_bank ||
        il >= DS4_N_LAYER || expert >= DS4_N_EXPERT ||
        expert >= DS4_MAX_EXPERT) {
        return false;
    }
    if (g->flash_expert_bank[il][expert]) return true;

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (!layer->map || layer->map_size == 0 ||
        layer->expert_stride > UINT64_MAX / (uint64_t)expert) {
        return false;
    }
    const uint64_t record_base = (uint64_t)expert * layer->expert_stride;
    if (record_base > layer->map_size ||
        layer->expert_stride > layer->map_size - record_base) {
        return false;
    }
    g->flash_expert_bank[il][expert] =
        ds4_gpu_mmap_tensor_view(layer->map,
                                 layer->map_size,
                                 record_base,
                                 layer->expert_stride);
    return g->flash_expert_bank[il][expert] != NULL;
}

static bool metal_graph_flash_moe_direct_mmap_slots6_active(const ds4_gpu_graph *g) {
    return g && g->flash_direct_mmap_bank &&
           (g->flash_direct_mmap_auto || flash_moe_direct_mmap_slots6_enabled());
}

static bool metal_graph_flash_moe_direct_mmap_record_slots6_active(const ds4_gpu_graph *g) {
    if (!metal_graph_flash_moe_direct_mmap_slots6_active(g)) return false;
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_RECORD_SLOTS6");
    if (env && env[0]) return atoi(env) != 0;
    if (env_flag_enabled("DS4_FLASH_MOE_DIRECT_MMAP_FAMILY_SLOTS6")) return false;
    return g->flash_direct_mmap_auto || flash_moe_direct_mmap_record_slots6_enabled();
}

static bool metal_graph_flash_moe_direct_mmap_prewarm_views_active(const ds4_gpu_graph *g) {
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_PREWARM_VIEWS");
    if (env && env[0]) return atoi(env) != 0;
    return metal_graph_flash_moe_direct_mmap_record_slots6_active(g) ||
           flash_moe_direct_mmap_prewarm_views_enabled();
}

static bool metal_graph_flash_moe_ensure_per_slot_buffer(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot) {
    if (!g || !g->flash_moe || !g->flash_per_slot_buffers ||
        il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
        slot >= (int32_t)DS4_MAX_EXPERT) {
        return false;
    }
    if (g->flash_expert_bank[il][slot]) return true;

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    ds4_gpu_tensor *buf = metal_graph_flash_moe_alloc_slot_bank_tensor(layer->expert_stride);
    if (!buf) return false;

    char label[80];
    snprintf(label, sizeof(label), "slot-layer-%u-id-%u", il, (uint32_t)slot);
    const bool ok = metal_graph_flash_moe_prepare_slot_bank_owner(
            buf,
            label,
            false,
            flash_moe_slot_bank_touch_pages_enabled(),
            NULL,
            NULL);
    if (!ok) {
        ds4_gpu_tensor_free(buf);
        return false;
    }
    g->flash_expert_bank[il][slot] = buf;
    if (!metal_graph_flash_moe_init_expert_family_views(g, il, (uint32_t)slot)) {
        ds4_gpu_tensor_free(g->flash_expert_bank[il][slot]);
        g->flash_expert_bank[il][slot] = NULL;
        return false;
    }
    if (g->flash_per_expert_bank_bytes > UINT64_MAX - layer->expert_stride) {
        g->flash_per_expert_bank_bytes = UINT64_MAX;
    } else {
        g->flash_per_expert_bank_bytes += layer->expert_stride;
    }
    return true;
}

static void metal_graph_flash_moe_gpu_l2_free(ds4_gpu_graph *g) {
    if (!g) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->flash_gpu_l2_mixed_bank[il]);
        g->flash_gpu_l2_mixed_bank[il] = NULL;
    }
    free(g->flash_gpu_l2_slot_to_expert);
    free(g->flash_gpu_l2_expert_to_slot);
    free(g->flash_gpu_l2_slot_age);
    g->flash_gpu_l2_slot_to_expert = NULL;
    g->flash_gpu_l2_expert_to_slot = NULL;
    g->flash_gpu_l2_slot_age = NULL;
    g->flash_gpu_l2_slot_bank = 0;
    g->flash_gpu_l2_capacity_bytes = 0;
}

static void metal_graph_flash_moe_init_direct_mmap_identity(ds4_gpu_graph *g) {
    if (!g || !g->flash_direct_mmap_bank || g->flash_slot_bank < DS4_N_EXPERT ||
        !g->flash_slot_to_expert || !g->flash_expert_to_slot || !g->flash_slot_age) {
        return;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        int32_t *slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * g->flash_slot_bank;
        int32_t *expert_to_slot =
            g->flash_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
        uint64_t *slot_age =
            g->flash_slot_age + (uint64_t)il * g->flash_slot_bank;
        for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
            slot_to_expert[expert] = (int32_t)expert;
            expert_to_slot[expert] = (int32_t)expert;
            slot_age[expert] = ++g->flash_age;
        }
    }
}

static void metal_graph_flash_moe_free_slot_banks(ds4_gpu_graph *g) {
    if (!g) return;
    if (g->flash_moe) ds4_gpu_flash_slot_bank_residency_clear();
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        for (uint32_t expert = 0; expert < DS4_N_EXPERT && expert < DS4_MAX_EXPERT; expert++) {
            metal_graph_flash_moe_free_expert_family_views(g, il, expert);
            ds4_gpu_tensor_free(g->flash_expert_bank[il][expert]);
            g->flash_expert_bank[il][expert] = NULL;
        }
        ds4_gpu_tensor_free(g->flash_record_table[il]);
        g->flash_record_table[il] = NULL;
        for (uint32_t chunk = 0; chunk < DS4_FLASH_MOE_MAX_CHUNKS; chunk++) {
            ds4_gpu_tensor_free(g->flash_chunk_weights[il][chunk]);
            ds4_gpu_tensor_free(g->flash_chunk_slot_selected[il][chunk]);
            ds4_gpu_tensor_free(g->flash_chunk_down_bank[il][chunk]);
            ds4_gpu_tensor_free(g->flash_chunk_up_bank[il][chunk]);
            ds4_gpu_tensor_free(g->flash_chunk_gate_bank[il][chunk]);
            ds4_gpu_tensor_free(g->flash_chunk_mixed_bank[il][chunk]);
            g->flash_chunk_weights[il][chunk] = NULL;
            g->flash_chunk_slot_selected[il][chunk] = NULL;
            g->flash_chunk_down_bank[il][chunk] = NULL;
            g->flash_chunk_up_bank[il][chunk] = NULL;
            g->flash_chunk_gate_bank[il][chunk] = NULL;
            g->flash_chunk_mixed_bank[il][chunk] = NULL;
        }
        ds4_gpu_tensor_free(g->flash_stage_down_bank[il]);
        ds4_gpu_tensor_free(g->flash_stage_up_bank[il]);
        ds4_gpu_tensor_free(g->flash_stage_gate_bank[il]);
        ds4_gpu_tensor_free(g->flash_stage_mixed_bank[il]);
        g->flash_stage_down_bank[il] = NULL;
        g->flash_stage_up_bank[il] = NULL;
        g->flash_stage_gate_bank[il] = NULL;
        g->flash_stage_mixed_bank[il] = NULL;
        ds4_gpu_tensor_free(g->flash_down_bank[il]);
        ds4_gpu_tensor_free(g->flash_up_bank[il]);
        ds4_gpu_tensor_free(g->flash_gate_bank[il]);
        ds4_gpu_tensor_free(g->flash_mixed_bank[il]);
        g->flash_down_bank[il] = NULL;
        g->flash_up_bank[il] = NULL;
        g->flash_gate_bank[il] = NULL;
        g->flash_mixed_bank[il] = NULL;
    }
    ds4_gpu_tensor_free(g->flash_layer_slot_slab_bank);
    g->flash_layer_slot_slab_bank = NULL;
    g->flash_layer_slot_slab_bytes = 0;
    ds4_gpu_tensor_free(g->flash_stage_slot_selected);
    g->flash_stage_slot_selected = NULL;
    g->flash_stage_bank_bytes = 0;
    ds4_gpu_tensor_free(g->flash_chunk_partial_out);
    g->flash_chunk_partial_out = NULL;
    g->flash_chunked_mixed_bank = false;
    g->flash_chunk_slots = 0;
    g->flash_chunk_count = 0;
    g->flash_chunked_bank_bytes = 0;
    g->flash_per_expert_bank_bytes = 0;
    metal_graph_flash_moe_gpu_l2_free(g);
}

static bool metal_graph_flash_moe_alloc_slot_banks(
        ds4_gpu_graph               *g,
        const ds4_flash_moe_sidecar *sidecar,
        const char                  *action) {
    if (!g || !sidecar) return false;

    bool ok = true;
    uint64_t total_bank_bytes = 0;
    uint64_t planned_bank_bytes = 0;
    uint64_t layer_mixed_bytes[DS4_MAX_LAYER] = { 0 };
    uint64_t layer_slab_offsets[DS4_MAX_LAYER] = { 0 };
    const bool per_expert = g->flash_per_expert_buffers;
    const bool per_slot = g->flash_per_slot_buffers;
    const bool lazy_per_slot = per_slot && g->flash_per_slot_lazy_alloc;
    const bool active_staging = per_slot && flash_moe_active_staging_enabled();
    const bool use_slot_residency =
        flash_moe_slot_bank_residency_enabled() && !lazy_per_slot;
    const bool touch_slot_pages = flash_moe_slot_bank_touch_pages_enabled();
    uint64_t touched_slot_bytes = 0;
    uint32_t touched_slot_buffers = 0;
    uint32_t residency_capacity = 0;
    if (flash_moe_untracked_slot_bank_enabled()) {
        static int logged_untracked_slot_bank = 0;
        if (!logged_untracked_slot_bank) {
            fprintf(stderr,
                    "ds4: Flash-MoE slot-bank buffers using Metal untracked hazard mode\n");
            logged_untracked_slot_bank = 1;
        }
    }

    if (g->flash_direct_mmap_bank) {
        const bool active_mmap_slots6 =
            metal_graph_flash_moe_direct_mmap_slots6_active(g);
        if (!g->flash_mixed_slot_bank || g->flash_per_expert_buffers ||
            g->flash_per_slot_buffers || g->flash_chunked_mixed_bank) {
            ok = false;
        }
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            if (!layer->map || layer->map_size == 0 || layer->file_size == 0 ||
                layer->file_size > layer->map_size) {
                ok = false;
                break;
            }
            total_bank_bytes += layer->file_size;
            if (active_mmap_slots6) continue;

            g->flash_mixed_bank[il] =
                ds4_gpu_mmap_tensor_view(layer->map, layer->map_size, 0, layer->file_size);
            ok = g->flash_mixed_bank[il] != NULL;
            if (!ok) break;

            const uint64_t gate_view_bytes =
                (uint64_t)(DS4_N_EXPERT - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_GATE];
            const uint64_t up_view_bytes =
                (uint64_t)(DS4_N_EXPERT - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_UP];
            const uint64_t down_view_bytes =
                (uint64_t)(DS4_N_EXPERT - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
            g->flash_gate_bank[il] =
                ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                    gate_view_bytes);
            g->flash_up_bank[il] =
                ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_UP],
                                    up_view_bytes);
            g->flash_down_bank[il] =
                ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                    down_view_bytes);
            ok = g->flash_gate_bank[il] &&
                 g->flash_up_bank[il] &&
                 g->flash_down_bank[il];
        }
        if (!ok) {
            metal_graph_flash_moe_free_slot_banks(g);
            return false;
        }
        metal_graph_flash_moe_init_direct_mmap_identity(g);
        uint32_t prewarmed_views = 0;
        const bool prewarm_record_views =
            metal_graph_flash_moe_direct_mmap_record_slots6_active(g);
        if (active_mmap_slots6 &&
            metal_graph_flash_moe_direct_mmap_prewarm_views_active(g)) {
            for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
                for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
                    ok = prewarm_record_views ?
                        metal_graph_flash_moe_ensure_direct_mmap_record_view(g, il, expert) :
                        metal_graph_flash_moe_ensure_direct_mmap_family_views(g, il, expert);
                    if (!ok) break;
                    prewarmed_views += prewarm_record_views ? 1u : DS4_FLASH_FAMILY_COUNT;
                }
            }
            if (!ok) {
                metal_graph_flash_moe_free_slot_banks(g);
                return false;
            }
        }
        const uint64_t dense_bytes = g->dense_mapped_bytes;
        const uint64_t context_bytes = g->context_buffer_bytes;
        fprintf(stderr,
                "ds4: Flash-MoE direct mmap bank ready: layers=%u slots=%u "
                "file-backed=%.1fGB gpu-bank=0.0GB Dense: %.1fGB Context: %.1fGB "
                "Total virtual <<<< %.1fGB >>>>\n",
                (uint32_t)DS4_N_LAYER,
                g->flash_slot_bank,
                (double)total_bank_bytes / 1073741824.0,
                (double)dense_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0,
                (double)(total_bank_bytes + dense_bytes + context_bytes) / 1073741824.0);
        fprintf(stderr,
                "ds4: Flash-MoE slot bank layout: direct read-only sidecar mmap "
                "(true expert id == slot id; %s)\n",
                active_mmap_slots6 ?
                "active slots6 no-copy views, no parent layer MTLBuffers" :
                "full layer MTLBuffers, no slot installs");
        if (g->flash_direct_mmap_auto) {
            fprintf(stderr,
                    "ds4: Flash-MoE direct mmap auto-selected for large MXFP4 slot bank "
                    "(disable with DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0)\n");
        }
        if (prewarmed_views != 0) {
            fprintf(stderr,
                    "ds4: Flash-MoE direct mmap views prewarmed: %u %s views "
                    "(disable with DS4_FLASH_MOE_DIRECT_MMAP_PREWARM_VIEWS=0)\n",
                    prewarmed_views,
                    prewarm_record_views ? "record" : "family");
        }
        return true;
    }

    if (per_slot) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            if (sidecar->slot_bank != 0 &&
                layer->expert_stride > UINT64_MAX / (uint64_t)sidecar->slot_bank) {
                ok = false;
                break;
            }
            const uint64_t layer_bytes =
                (uint64_t)sidecar->slot_bank * layer->expert_stride;
            if (planned_bank_bytes > UINT64_MAX - layer_bytes) {
                ok = false;
                break;
            }
            planned_bank_bytes += layer_bytes;
        }
    }

    if (ok && use_slot_residency) {
        residency_capacity = (per_expert || per_slot) ?
            (uint32_t)(DS4_N_LAYER * g->flash_slot_bank) :
            (g->flash_chunked_mixed_bank ?
             (uint32_t)(DS4_N_LAYER * g->flash_chunk_count) :
             (g->flash_mixed_slot_bank ?
             (g->flash_layer_slot_slab ? 1u : (uint32_t)DS4_N_LAYER) :
             (uint32_t)DS4_N_LAYER * DS4_FLASH_FAMILY_COUNT));
        if (!ds4_gpu_flash_slot_bank_residency_begin(residency_capacity)) {
            ok = false;
        }
    }
    if (ok && per_expert) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            for (uint32_t expert = 0; ok && expert < DS4_N_EXPERT; expert++) {
                if (expert >= DS4_MAX_EXPERT) {
                    ok = false;
                    break;
                }
                ds4_gpu_tensor *buf = metal_graph_flash_moe_alloc_slot_bank_tensor(layer->expert_stride);
                g->flash_expert_bank[il][expert] = buf;
                ok = buf != NULL;
                if (!ok) break;

                char label[80];
                snprintf(label, sizeof(label), "expert-layer-%u-id-%u", il, expert);
                ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                        buf,
                        label,
                        use_slot_residency,
                        touch_slot_pages,
                        &touched_slot_bytes,
                        &touched_slot_buffers);
                if (!ok) break;
                ok = metal_graph_flash_moe_init_expert_family_views(g, il, expert);
                if (!ok) break;

                uint8_t *dst = (uint8_t *)ds4_gpu_tensor_contents(buf);
                if (!dst && layer->expert_stride != 0) {
                    ok = false;
                    break;
                }
                errno = 0;
                const uint64_t record_offset = (uint64_t)expert * layer->expert_stride;
                ok = flash_moe_pread_split(layer->fd,
                                           record_offset,
                                           dst,
                                           layer->expert_stride,
                                           flash_moe_cache_io_split());
                if (ok) {
                    ok = ds4_gpu_tensor_did_modify(buf, 0, layer->expert_stride) != 0;
                }
                if (!ok) {
                    fprintf(stderr,
                            "ds4: Flash-MoE failed to preload per-expert buffer layer %u expert %u: %s\n",
                            il,
                            expert,
                            errno ? strerror(errno) : "short read");
                    break;
                }
                total_bank_bytes += layer->expert_stride;
                g->flash_installed_bytes += layer->expert_stride;
            }
        }
        if (ok) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                int32_t *slot_to_expert =
                    g->flash_slot_to_expert + (uint64_t)il * g->flash_slot_bank;
                int32_t *expert_to_slot =
                    g->flash_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
                uint64_t *slot_age =
                    g->flash_slot_age + (uint64_t)il * g->flash_slot_bank;
                for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
                    slot_to_expert[expert] = (int32_t)expert;
                    expert_to_slot[expert] = (int32_t)expert;
                    slot_age[expert] = ++g->flash_age;
                }
            }
            g->flash_per_expert_bank_bytes = total_bank_bytes;
        }
    }
    if (ok && per_slot && !lazy_per_slot) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            for (uint32_t slot = 0; ok && slot < sidecar->slot_bank; slot++) {
                if (slot >= DS4_MAX_EXPERT) {
                    ok = false;
                    break;
                }
                ds4_gpu_tensor *buf = metal_graph_flash_moe_alloc_slot_bank_tensor(layer->expert_stride);
                g->flash_expert_bank[il][slot] = buf;
                ok = buf != NULL;
                if (!ok) break;

                char label[80];
                snprintf(label, sizeof(label), "slot-layer-%u-id-%u", il, slot);
                ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                        buf,
                        label,
                        use_slot_residency,
                        touch_slot_pages,
                        &touched_slot_bytes,
                        &touched_slot_buffers);
                if (ok) ok = metal_graph_flash_moe_init_expert_family_views(g, il, slot);
                if (ok) total_bank_bytes += layer->expert_stride;
            }
        }
        if (ok) g->flash_per_expert_bank_bytes = total_bank_bytes;
    }
    if (ok && per_slot && !lazy_per_slot && flash_moe_record_table_enabled()) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            g->flash_record_table[il] =
                ds4_gpu_flash_moe_record_table_alloc(g->flash_slot_bank);
            ok = g->flash_record_table[il] &&
                 ds4_gpu_flash_moe_record_table_set(g->flash_record_table[il],
                                                    g->flash_expert_bank[il],
                                                    g->flash_slot_bank) != 0;
        }
        if (!ok) {
            fprintf(stderr,
                    "ds4: Flash-MoE failed to build per-layer record argument tables\n");
        }
    }
    if (ok && !per_expert && !per_slot && g->flash_mixed_slot_bank &&
        !g->flash_chunked_mixed_bank) {
        const uint64_t slab_align = 4096u;
        uint64_t slab_bytes = 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            if (sidecar->slot_bank != 0 &&
                layer->expert_stride > UINT64_MAX / (uint64_t)sidecar->slot_bank) {
                ok = false;
                break;
            }
            const uint64_t mixed_bytes = (uint64_t)sidecar->slot_bank * layer->expert_stride;
            layer_mixed_bytes[il] = mixed_bytes;
            if (!g->flash_layer_slot_slab) continue;
            if (slab_bytes > UINT64_MAX - (slab_align - 1u)) {
                ok = false;
                break;
            }
            slab_bytes = align_up(slab_bytes, slab_align);
            layer_slab_offsets[il] = slab_bytes;
            if (mixed_bytes > UINT64_MAX - slab_bytes) {
                ok = false;
                break;
            }
            slab_bytes += mixed_bytes;
        }
        if (ok && g->flash_layer_slot_slab) {
            g->flash_layer_slot_slab_bank = metal_graph_flash_moe_alloc_slot_bank_tensor(slab_bytes);
            ok = g->flash_layer_slot_slab_bank != NULL;
            if (ok) {
                ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                        g->flash_layer_slot_slab_bank,
                        "layer-slot-slab",
                        use_slot_residency,
                        touch_slot_pages,
                        &touched_slot_bytes,
                        &touched_slot_buffers);
                g->flash_layer_slot_slab_bytes = slab_bytes;
                total_bank_bytes = slab_bytes;
            }
        }
    }

    if (ok && !per_expert && !per_slot && g->flash_chunked_mixed_bank) {
        if (g->flash_chunk_slots == 0 || g->flash_chunk_count == 0 ||
            g->flash_chunk_count > DS4_FLASH_MOE_MAX_CHUNKS) {
            ok = false;
        }
        g->flash_chunk_partial_out =
            ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        ok = ok && g->flash_chunk_partial_out;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            for (uint32_t chunk = 0; ok && chunk < g->flash_chunk_count; chunk++) {
                const uint32_t first_slot = chunk * g->flash_chunk_slots;
                if (first_slot >= sidecar->slot_bank) {
                    ok = false;
                    break;
                }
                uint32_t slots = sidecar->slot_bank - first_slot;
                if (slots > g->flash_chunk_slots) slots = g->flash_chunk_slots;
                if (slots == 0 ||
                    layer->expert_stride > UINT64_MAX / (uint64_t)slots) {
                    ok = false;
                    break;
                }
                const uint64_t mixed_bytes = (uint64_t)slots * layer->expert_stride;
                g->flash_chunk_mixed_bank[il][chunk] =
                    metal_graph_flash_moe_alloc_slot_bank_tensor(mixed_bytes);
                if (g->flash_chunk_mixed_bank[il][chunk]) {
                    char label[96];
                    snprintf(label, sizeof(label), "mixed-chunk-layer-%u-%u", il, chunk);
                    ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                            g->flash_chunk_mixed_bank[il][chunk],
                            label,
                            use_slot_residency,
                            touch_slot_pages,
                            &touched_slot_bytes,
                            &touched_slot_buffers);
                } else {
                    ok = false;
                }
                if (!ok) break;

                const uint64_t gate_view_bytes =
                    (uint64_t)(slots - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_GATE];
                const uint64_t up_view_bytes =
                    (uint64_t)(slots - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_UP];
                const uint64_t down_view_bytes =
                    (uint64_t)(slots - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
                g->flash_chunk_gate_bank[il][chunk] =
                    ds4_gpu_tensor_view(g->flash_chunk_mixed_bank[il][chunk],
                                        layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                        gate_view_bytes);
                g->flash_chunk_up_bank[il][chunk] =
                    ds4_gpu_tensor_view(g->flash_chunk_mixed_bank[il][chunk],
                                        layer->family_offset[DS4_FLASH_FAMILY_UP],
                                        up_view_bytes);
                g->flash_chunk_down_bank[il][chunk] =
                    ds4_gpu_tensor_view(g->flash_chunk_mixed_bank[il][chunk],
                                        layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                        down_view_bytes);
                g->flash_chunk_slot_selected[il][chunk] =
                    ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_ACTIVE_USED * sizeof(int32_t));
                g->flash_chunk_weights[il][chunk] =
                    ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_ACTIVE_USED * sizeof(float));
                ok = g->flash_chunk_gate_bank[il][chunk] &&
                     g->flash_chunk_up_bank[il][chunk] &&
                     g->flash_chunk_down_bank[il][chunk] &&
                     g->flash_chunk_slot_selected[il][chunk] &&
                     g->flash_chunk_weights[il][chunk];
                if (ok) {
                    total_bank_bytes += mixed_bytes;
                    g->flash_chunked_bank_bytes += mixed_bytes;
                }
            }
        }
    }

    for (uint32_t il = 0; ok && !per_expert && !per_slot &&
                         !g->flash_chunked_mixed_bank && il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
        if (g->flash_mixed_slot_bank) {
            const uint64_t mixed_bytes = layer_mixed_bytes[il];
            if (g->flash_layer_slot_slab) {
                g->flash_mixed_bank[il] =
                    ds4_gpu_tensor_view(g->flash_layer_slot_slab_bank,
                                        layer_slab_offsets[il],
                                        mixed_bytes);
            } else {
                g->flash_mixed_bank[il] =
                    metal_graph_flash_moe_alloc_slot_bank_tensor(mixed_bytes);
                if (g->flash_mixed_bank[il]) {
                    char label[64];
                    snprintf(label, sizeof(label), "mixed-layer-%u", il);
                    ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                            g->flash_mixed_bank[il],
                            label,
                            use_slot_residency,
                            touch_slot_pages,
                            &touched_slot_bytes,
                            &touched_slot_buffers);
                }
            }
            if (ok && g->flash_mixed_bank[il]) {
                const uint64_t gate_view_bytes =
                    (uint64_t)(sidecar->slot_bank - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_GATE];
                const uint64_t up_view_bytes =
                    (uint64_t)(sidecar->slot_bank - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_UP];
                const uint64_t down_view_bytes =
                    (uint64_t)(sidecar->slot_bank - 1u) * layer->expert_stride +
                    layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
                g->flash_gate_bank[il] =
                    ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                        layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                        gate_view_bytes);
                g->flash_up_bank[il] =
                    ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                        layer->family_offset[DS4_FLASH_FAMILY_UP],
                                        up_view_bytes);
                g->flash_down_bank[il] =
                    ds4_gpu_tensor_view(g->flash_mixed_bank[il],
                                        layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                        down_view_bytes);
            }
            ok = ok &&
                 g->flash_mixed_bank[il] &&
                 g->flash_gate_bank[il] &&
                 g->flash_up_bank[il] &&
                 g->flash_down_bank[il];
            if (!g->flash_layer_slot_slab) {
                total_bank_bytes += mixed_bytes;
            }
        } else {
            const uint64_t gate_bytes = (uint64_t)sidecar->slot_bank *
                                        layer->family_bytes[DS4_FLASH_FAMILY_GATE];
            const uint64_t up_bytes = (uint64_t)sidecar->slot_bank *
                                      layer->family_bytes[DS4_FLASH_FAMILY_UP];
            const uint64_t down_bytes = (uint64_t)sidecar->slot_bank *
                                        layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
            g->flash_gate_bank[il] = metal_graph_flash_moe_alloc_slot_bank_tensor(gate_bytes);
            g->flash_up_bank[il] = metal_graph_flash_moe_alloc_slot_bank_tensor(up_bytes);
            g->flash_down_bank[il] = metal_graph_flash_moe_alloc_slot_bank_tensor(down_bytes);
            if (g->flash_gate_bank[il] && g->flash_up_bank[il] && g->flash_down_bank[il]) {
                char label[64];
                snprintf(label, sizeof(label), "gate-layer-%u", il);
                ok = ok && metal_graph_flash_moe_prepare_slot_bank_owner(
                        g->flash_gate_bank[il], label, use_slot_residency,
                        touch_slot_pages, &touched_slot_bytes, &touched_slot_buffers);
                snprintf(label, sizeof(label), "up-layer-%u", il);
                ok = ok && metal_graph_flash_moe_prepare_slot_bank_owner(
                        g->flash_up_bank[il], label, use_slot_residency,
                        touch_slot_pages, &touched_slot_bytes, &touched_slot_buffers);
                snprintf(label, sizeof(label), "down-layer-%u", il);
                ok = ok && metal_graph_flash_moe_prepare_slot_bank_owner(
                        g->flash_down_bank[il], label, use_slot_residency,
                        touch_slot_pages, &touched_slot_bytes, &touched_slot_buffers);
            }
            ok = ok && g->flash_gate_bank[il] && g->flash_up_bank[il] && g->flash_down_bank[il];
            total_bank_bytes += gate_bytes + up_bytes + down_bytes;
        }
    }
    if (ok && active_staging) {
        int32_t local_ids[DS4_N_EXPERT_ACTIVE_USED];
        for (uint32_t k = 0; k < DS4_N_EXPERT_ACTIVE_USED; k++) {
            local_ids[k] = (int32_t)k;
        }
        g->flash_stage_slot_selected =
            ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_ACTIVE_USED * sizeof(local_ids[0]));
        ok = g->flash_stage_slot_selected &&
             ds4_gpu_tensor_write(g->flash_stage_slot_selected,
                                  0,
                                  local_ids,
                                  (uint64_t)DS4_N_EXPERT_ACTIVE_USED * sizeof(local_ids[0])) != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
            if (layer->expert_stride > UINT64_MAX / (uint64_t)DS4_N_EXPERT_ACTIVE_USED) {
                ok = false;
                break;
            }
            const uint64_t stage_bytes =
                (uint64_t)DS4_N_EXPERT_ACTIVE_USED * layer->expert_stride;
            g->flash_stage_mixed_bank[il] =
                metal_graph_flash_moe_alloc_slot_bank_tensor(stage_bytes);
            if (g->flash_stage_mixed_bank[il]) {
                char label[80];
                snprintf(label, sizeof(label), "active-stage-layer-%u", il);
                ok = metal_graph_flash_moe_prepare_slot_bank_owner(
                        g->flash_stage_mixed_bank[il],
                        label,
                        false,
                        touch_slot_pages,
                        &touched_slot_bytes,
                        &touched_slot_buffers);
            } else {
                ok = false;
            }
            if (!ok) break;

            const uint64_t gate_view_bytes =
                (uint64_t)(DS4_N_EXPERT_ACTIVE_USED - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_GATE];
            const uint64_t up_view_bytes =
                (uint64_t)(DS4_N_EXPERT_ACTIVE_USED - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_UP];
            const uint64_t down_view_bytes =
                (uint64_t)(DS4_N_EXPERT_ACTIVE_USED - 1u) * layer->expert_stride +
                layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
            g->flash_stage_gate_bank[il] =
                ds4_gpu_tensor_view(g->flash_stage_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                    gate_view_bytes);
            g->flash_stage_up_bank[il] =
                ds4_gpu_tensor_view(g->flash_stage_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_UP],
                                    up_view_bytes);
            g->flash_stage_down_bank[il] =
                ds4_gpu_tensor_view(g->flash_stage_mixed_bank[il],
                                    layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                    down_view_bytes);
            ok = g->flash_stage_gate_bank[il] &&
                 g->flash_stage_up_bank[il] &&
                 g->flash_stage_down_bank[il];
            if (ok) {
                total_bank_bytes += stage_bytes;
                g->flash_stage_bank_bytes += stage_bytes;
            }
        }
    }
    if (!ok) {
        metal_graph_flash_moe_free_slot_banks(g);
        return false;
    }
    if (use_slot_residency && !ds4_gpu_flash_slot_bank_residency_commit()) {
        metal_graph_flash_moe_free_slot_banks(g);
        return false;
    }
    if (touch_slot_pages) {
        fprintf(stderr,
                "ds4: Flash-MoE slot-bank pages touched: buffers=%u bytes=%.2f GiB\n",
                touched_slot_buffers,
                (double)touched_slot_bytes / 1073741824.0);
    }

    const uint64_t dense_bytes = g->dense_mapped_bytes;
    const uint64_t context_bytes = g->context_buffer_bytes;
    const uint64_t total_bytes = total_bank_bytes + dense_bytes + context_bytes;
    const uint64_t planned_total_bytes =
        lazy_per_slot ? (planned_bank_bytes + dense_bytes + context_bytes) : total_bytes;
    if (lazy_per_slot) {
        fprintf(stderr,
                "ds4: Flash-MoE slot banks %s: layers=%u slots=%u "
                "gpu-bank=%.1fGB allocated, planned=%.1fGB Dense: %.1fGB Context: %.1fGB "
                "Total <<<< %.1fGB allocated, %.1fGB planned >>>>\n",
                action && action[0] ? action : "allocated",
                (uint32_t)DS4_N_LAYER,
                g->flash_slot_bank,
                (double)total_bank_bytes / 1073741824.0,
                (double)planned_bank_bytes / 1073741824.0,
                (double)dense_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0,
                (double)planned_total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: Flash-MoE slot banks %s: layers=%u slots=%u "
                "gpu-bank=%.1fGB Dense: %.1fGB Context: %.1fGB Total <<<< %.1fGB >>>>\n",
                action && action[0] ? action : "allocated",
                (uint32_t)DS4_N_LAYER,
                g->flash_slot_bank,
                (double)total_bank_bytes / 1073741824.0,
                (double)dense_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
    fprintf(stderr,
            "ds4: Flash-MoE slot bank layout: %s\n",
            per_expert ? "full-resident semantic-expert buffers" :
            lazy_per_slot ? "lazy per-slot expert buffers" :
            per_slot ? "per-slot expert buffers" :
            g->flash_chunked_mixed_bank ? "chunked mixed expert-major" :
            g->flash_layer_slot_slab ? "layer-major slab mixed expert-major" :
            (g->flash_mixed_slot_bank ? "mixed expert-major" : "separate families"));
    if (g->flash_chunked_mixed_bank) {
        fprintf(stderr,
                "ds4: Flash-MoE chunked mixed bank: chunks=%u chunk-slots=%u "
                "gpu-bank=%.2f GiB; env DS4_FLASH_MOE_CHUNKED_MIXED=1\n",
                g->flash_chunk_count,
                g->flash_chunk_slots,
                (double)g->flash_chunked_bank_bytes / 1073741824.0);
    }
    if (active_staging && g->flash_stage_bank_bytes != 0) {
        fprintf(stderr,
                "ds4: Flash-MoE active staging: six active experts/layer staged into "
                "tiny mixed banks (extra %.2f GiB); env DS4_FLASH_MOE_ACTIVE_STAGING=1\n",
                (double)g->flash_stage_bank_bytes / 1073741824.0);
    }
    if (per_slot && !lazy_per_slot && flash_moe_record_table_enabled()) {
        fprintf(stderr,
                "ds4: Flash-MoE record-table decode enabled: one argument-buffer table "
                "per layer maps slot ids to full expert records\n");
    }
    const uint64_t warn_slot_bank_bytes = 44ull * 1024ull * 1024ull * 1024ull;
    const uint64_t warn_bank_bytes = lazy_per_slot ? planned_bank_bytes : total_bank_bytes;
    if ((g->flash_slot_bank > 128u || warn_bank_bytes >= warn_slot_bank_bytes) &&
        g->flash_per_slot_buffers) {
        fprintf(stderr,
                "ds4: Flash-MoE large-bank decode guard: slots=%u gpu-bank=%.2f GiB "
                "uses split per-slot buffers%s, avoiding per-dispatch binding of the "
                "full mixed layer bank. Set DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1 "
                "to A/B the legacy mixed-bank path.\n",
                g->flash_slot_bank,
                (double)warn_bank_bytes / 1073741824.0,
                g->flash_per_slot_buffers_auto ? " (auto)" : "");
    } else if (g->flash_slot_bank > 128u || warn_bank_bytes >= warn_slot_bank_bytes) {
        if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO) {
            fprintf(stderr,
                    "ds4: warning: very large Pro Flash-MoE slot bank can collapse decode throughput "
                    "(slots=%u, gpu-bank=%.2f GiB, expert-record %.2f MiB). "
                    "For Pro SSD decode, calibrate on this model/storage: very large models may "
                    "prefer larger banks for hit rate, while residency/page-locality cliffs can "
                    "still make smaller banks faster.\n",
                    g->flash_slot_bank,
                    (double)warn_bank_bytes / 1073741824.0,
                    (double)g->flash_moe->max_expert_stride / 1048576.0);
        } else {
            fprintf(stderr,
                    "ds4: warning: very large Flash-MoE slot bank can collapse decode throughput "
                    "(slots=%u, gpu-bank=%.2f GiB). The wired bank competes with the OS file "
                    "cache that serves decode-miss reads of the sidecar; when the sidecar is "
                    "larger than RAM, smaller banks decode faster. Compare --moe-slot-bank "
                    "32/48/64 or use --ssd-cache auto.\n",
                    g->flash_slot_bank,
                    (double)warn_bank_bytes / 1073741824.0);
        }
    }

    {
        /* Announce the planned decode-bank shrink up front so an explicit
         * --ssd-cache size being repurposed as "prefill budget" is never a
         * surprise discovered from memory graphs. */
        const uint32_t decode_slots = flash_moe_decode_slot_bank_target(g);
        if (decode_slots != 0 && decode_slots < g->flash_slot_bank) {
            fprintf(stderr,
                    "ds4: Flash-MoE bank plan: %u slots (%.2f GiB) for prefill; after the "
                    "first prefill the decode bank shrinks to %u slots (%.2f GiB) so the OS "
                    "file cache can serve decode-miss reads. "
                    "DS4_FLASH_MOE_DECODE_SLOT_BANK=0 keeps the full bank (decode collapses "
                    "when the sidecar is larger than RAM); =<slots> or "
                    "DS4_FLASH_MOE_DECODE_SSD_CACHE=<size> picks your own decode bank.\n",
                    g->flash_slot_bank,
                    (double)warn_bank_bytes / 1073741824.0,
                    decode_slots,
                    (double)decode_slots * (double)DS4_N_LAYER *
                        (double)g->flash_moe->max_expert_stride / 1073741824.0);
        }
    }

    {
        const uint64_t wset = ds4_gpu_recommended_working_set_bytes();
        if (wset != 0 && warn_bank_bytes > wset) {
            fprintf(stderr,
                    "ds4: warning: Flash-MoE slot bank %.2f GiB exceeds the GPU working-set "
                    "budget %.2f GiB; Metal residency will thrash. Shrink the bank or raise "
                    "the limit: sudo sysctl iogpu.wired_limit_mb=%llu\n",
                    (double)warn_bank_bytes / 1073741824.0,
                    (double)wset / 1073741824.0,
                    (unsigned long long)((warn_bank_bytes >> 20) + 16384u));
        }
    }

    return true;
}

static bool metal_graph_enable_flash_moe(
        ds4_gpu_graph                *g,
        const ds4_flash_moe_sidecar  *sidecar,
        const ds4_layer_weights      *layer) {
    if (!g || !sidecar) return true;
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    if (sidecar->slot_bank < active_expert_used || sidecar->slot_bank > DS4_N_EXPERT) return false;
    if (DS4_N_EXPERT > DS4_MAX_EXPERT) return false;

    const bool direct_mmap_env = flash_moe_direct_mmap_bank_enabled();
    const bool direct_mmap_auto =
        !direct_mmap_env && flash_moe_direct_mmap_auto_enabled(sidecar, layer);
    const bool direct_mmap = direct_mmap_env || direct_mmap_auto;
    if (direct_mmap && !sidecar->expert_mmap) {
        if (!ds4_flash_moe_sidecar_ensure_mmap((ds4_flash_moe_sidecar *)sidecar)) {
            return false;
        }
        fprintf(stderr,
                "ds4: Flash-MoE expert mmap cache: enabled late for direct mmap "
                "(%.2f GiB mapped)\n",
                (double)sidecar->expert_mmap_bytes / 1073741824.0);
    }
    const bool per_expert = !direct_mmap && flash_moe_per_expert_buffers_enabled();
    const bool per_slot_env = !direct_mmap && flash_moe_per_slot_buffers_enabled();
    const bool decode_shrink_requested = !direct_mmap && flash_moe_decode_bank_shrink_requested();
    const bool chunked_mixed =
        !direct_mmap && !per_expert && !per_slot_env && flash_moe_chunked_mixed_enabled();
    const bool per_slot_auto =
        !direct_mmap && !per_expert && !per_slot_env && !chunked_mixed && !decode_shrink_requested &&
        flash_moe_auto_per_slot_buffers_enabled(sidecar);
    const bool per_slot = !per_expert && (per_slot_env || per_slot_auto);
    const bool per_slot_lazy = per_slot && flash_moe_per_slot_lazy_alloc_enabled();
    const uint32_t requested_slot_bank =
        (per_expert || direct_mmap) ? DS4_N_EXPERT : sidecar->slot_bank;
    uint32_t effective_slot_bank = requested_slot_bank;
    if (!direct_mmap && !per_expert && !per_slot && flash_moe_prefill_decode_l1_enabled()) {
        const uint64_t budget = flash_moe_decode_l1_budget_bytes();
        const uint32_t capped =
            flash_moe_decode_slot_bank_for_budget(sidecar,
                                                  active_expert_used,
                                                  requested_slot_bank,
                                                  budget);
        if (capped != 0 && capped < requested_slot_bank) {
            effective_slot_bank = capped;
            ((ds4_flash_moe_sidecar *)sidecar)->slot_bank = capped;
            fprintf(stderr,
                    "ds4: Flash-MoE prefill/decode L1 cap: requested %u slots, "
                    "using %u slots (%.2f GiB budget) before prefill; "
                    "extra --ssd-cache budget is left for the OS file cache\n",
                    requested_slot_bank,
                    capped,
                    (double)budget / 1073741824.0);
        }
    }
    const uint64_t layer_slots = (uint64_t)DS4_N_LAYER * effective_slot_bank;
    if (layer_slots > SIZE_MAX / sizeof(int32_t) ||
        layer_slots > SIZE_MAX / sizeof(uint64_t) ||
        (uint64_t)DS4_N_LAYER * DS4_N_EXPERT > SIZE_MAX / sizeof(int32_t)) {
        return false;
    }

    g->flash_moe = sidecar;
    g->flash_slot_bank = effective_slot_bank;
    g->flash_per_slot_buffers = per_slot;
    g->flash_per_slot_buffers_auto = per_slot_auto;
    g->flash_per_slot_lazy_alloc = per_slot_lazy;
    g->flash_per_expert_buffers = per_expert;
    g->flash_direct_mmap_bank = direct_mmap;
    g->flash_direct_mmap_auto = direct_mmap_auto;
    g->flash_chunked_mixed_bank = chunked_mixed;
    g->flash_chunk_slots = chunked_mixed ?
                            flash_moe_chunked_mixed_slots(effective_slot_bank) : 0u;
    g->flash_chunk_count = chunked_mixed && g->flash_chunk_slots ?
                           (effective_slot_bank + g->flash_chunk_slots - 1u) /
                               g->flash_chunk_slots : 0u;
    if (g->flash_chunk_count > DS4_FLASH_MOE_MAX_CHUNKS) return false;
    g->flash_mixed_slot_bank = g->flash_direct_mmap_bank ||
                                (!g->flash_per_expert_buffers &&
                                !g->flash_per_slot_buffers &&
                                (g->flash_chunked_mixed_bank ||
                                 flash_moe_mixed_slot_bank_enabled()));
    g->flash_layer_slot_slab =
        !g->flash_direct_mmap_bank &&
        g->flash_mixed_slot_bank && !g->flash_chunked_mixed_bank &&
        flash_moe_layer_slot_slab_enabled();
    g->flash_layer_slot_slab_bytes = 0;
    g->router_slot_selected = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    g->flash_slot_to_expert = xmalloc((size_t)layer_slots * sizeof(g->flash_slot_to_expert[0]));
    g->flash_expert_to_slot = xmalloc((size_t)DS4_N_LAYER * DS4_N_EXPERT * sizeof(g->flash_expert_to_slot[0]));
    g->flash_slot_age = xcalloc((size_t)layer_slots, sizeof(g->flash_slot_age[0]));
    g->flash_replay_slot_expert = xmalloc((size_t)layer_slots * sizeof(g->flash_replay_slot_expert[0]));
    g->flash_replay_slot_valid = xcalloc((size_t)layer_slots, sizeof(g->flash_replay_slot_valid[0]));
    g->flash_install_buf = xmalloc(sidecar->max_expert_stride ? (size_t)sidecar->max_expert_stride : 1u);
    g->flash_decode_prefetch_scratch_stride = sidecar->max_expert_stride;
    g->flash_decode_prefetch_scratch_slots =
        (g->flash_per_expert_buffers || g->flash_direct_mmap_bank) ?
        0u : flash_moe_decode_prefetch_max_loads();
    if (g->flash_decode_prefetch_scratch_slots > DS4_N_EXPERT_ACTIVE_USED) {
        g->flash_decode_prefetch_scratch_slots = DS4_N_EXPERT_ACTIVE_USED;
    }
    if (g->flash_decode_prefetch_scratch_stride > 0 &&
        g->flash_decode_prefetch_scratch_slots > 0) {
        const uint64_t scratch_bytes =
            g->flash_decode_prefetch_scratch_stride *
            (uint64_t)g->flash_decode_prefetch_scratch_slots;
        g->flash_decode_prefetch_scratch = xmalloc((size_t)scratch_bytes);
    }

    for (uint64_t i = 0; i < layer_slots; i++) g->flash_slot_to_expert[i] = -1;
    for (uint64_t i = 0; i < layer_slots; i++) g->flash_replay_slot_expert[i] = -1;
    for (uint64_t i = 0; i < (uint64_t)DS4_N_LAYER * DS4_N_EXPERT; i++) {
        g->flash_expert_to_slot[i] = -1;
    }

    bool ok = g->router_slot_selected &&
              g->flash_slot_to_expert &&
              g->flash_expert_to_slot &&
              g->flash_slot_age &&
              g->flash_replay_slot_expert &&
              g->flash_replay_slot_valid &&
              g->flash_install_buf &&
              (g->flash_decode_prefetch_scratch_slots == 0 ||
               g->flash_decode_prefetch_scratch_stride == 0 ||
               g->flash_decode_prefetch_scratch != NULL);
    uint64_t prefill_gate_bytes = 0;
    uint64_t prefill_up_bytes = 0;
    uint64_t prefill_down_bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
        if (layer->family_bytes[DS4_FLASH_FAMILY_GATE] > prefill_gate_bytes) {
            prefill_gate_bytes = layer->family_bytes[DS4_FLASH_FAMILY_GATE];
        }
        if (layer->family_bytes[DS4_FLASH_FAMILY_UP] > prefill_up_bytes) {
            prefill_up_bytes = layer->family_bytes[DS4_FLASH_FAMILY_UP];
        }
        if (layer->family_bytes[DS4_FLASH_FAMILY_DOWN] > prefill_down_bytes) {
            prefill_down_bytes = layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
        }
    }
    const uint64_t pc = g->prefill_cap ? g->prefill_cap : 1u;
    g->flash_prefill_gate_bank = ds4_gpu_tensor_alloc(prefill_gate_bytes);
    g->flash_prefill_up_bank = ds4_gpu_tensor_alloc(prefill_up_bytes);
    g->flash_prefill_down_bank = ds4_gpu_tensor_alloc(prefill_down_bytes);

    /* Second prefill scratch bank set for overlapping sidecar loads with compute */
    g->flash_prefill_gate_bank2 = ds4_gpu_tensor_alloc(prefill_gate_bytes);
    g->flash_prefill_up_bank2  = ds4_gpu_tensor_alloc(prefill_up_bytes);
    g->flash_prefill_down_bank2 = ds4_gpu_tensor_alloc(prefill_down_bytes);
    g->flash_prefill_gate_bank3 = ds4_gpu_tensor_alloc(prefill_gate_bytes);
    g->flash_prefill_up_bank3  = ds4_gpu_tensor_alloc(prefill_up_bytes);
    g->flash_prefill_down_bank3 = ds4_gpu_tensor_alloc(prefill_down_bytes);
    g->flash_prefill_gate_bank4 = ds4_gpu_tensor_alloc(prefill_gate_bytes);
    g->flash_prefill_up_bank4  = ds4_gpu_tensor_alloc(prefill_up_bytes);
    g->flash_prefill_down_bank4 = ds4_gpu_tensor_alloc(prefill_down_bytes);
    g->flash_prefill_x = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->flash_prefill_gate = ds4_gpu_tensor_alloc(pc * DS4_N_FF_EXP * sizeof(float));
    g->flash_prefill_up = ds4_gpu_tensor_alloc(pc * DS4_N_FF_EXP * sizeof(float));
    g->flash_prefill_mid = ds4_gpu_tensor_alloc(pc * DS4_N_FF_EXP * sizeof(float));
    g->flash_prefill_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->flash_prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->flash_prefill_selected = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->flash_prefill_weights = ds4_gpu_tensor_alloc(pc * sizeof(float));

    /* GPU dedup buffers (one entry per token-expert pair at worst) */
    const uint64_t dedup_pairs = pc * DS4_N_EXPERT_USED;
    g->flash_dedup_token_list = ds4_gpu_tensor_alloc(dedup_pairs * sizeof(int32_t));
    g->flash_dedup_weight_list = ds4_gpu_tensor_alloc(dedup_pairs * sizeof(float));
    g->flash_dedup_offsets = ds4_gpu_tensor_alloc((DS4_N_EXPERT + 1) * sizeof(uint32_t));

    ok = ok &&
         g->flash_prefill_gate_bank &&
         g->flash_prefill_up_bank &&
         g->flash_prefill_down_bank &&
         g->flash_prefill_gate_bank2 &&
         g->flash_prefill_up_bank2 &&
         g->flash_prefill_down_bank2 &&
         g->flash_prefill_gate_bank3 &&
         g->flash_prefill_up_bank3 &&
         g->flash_prefill_down_bank3 &&
         g->flash_prefill_gate_bank4 &&
         g->flash_prefill_up_bank4 &&
         g->flash_prefill_down_bank4 &&
         g->flash_prefill_x &&
         g->flash_prefill_gate &&
         g->flash_prefill_up &&
         g->flash_prefill_mid &&
         g->flash_prefill_out &&
         g->flash_prefill_tokens &&
         g->flash_prefill_selected &&
         g->flash_prefill_weights &&
         g->flash_dedup_token_list &&
         g->flash_dedup_weight_list &&
         g->flash_dedup_offsets;
    if (ok) {
        int32_t *zeros = xcalloc((size_t)pc, sizeof(zeros[0]));
        ok = ds4_gpu_tensor_write(g->flash_prefill_selected,
                                  0,
                                  zeros,
                                  pc * sizeof(zeros[0])) != 0;
        free(zeros);
    }
    if (ok) ok = metal_graph_flash_moe_alloc_slot_banks(g, sidecar, "allocated");
    if (!ok) return false;

    /* Resolved prefill compute path (routed/dense + precision), printed right
     * here alongside the slot-bank line so it shows at startup. */
    metal_graph_log_prefill_compute_once(g, g->flash_slot_bank, layer);

    /* Diagnostic for M5 fast path + current prefetch depth (matches anemll-llama pipeline depth) */
    {
        bool m5_fast = ds4_gpu_use_m5_simdgroup_matrix();
        int prefetch = get_prefill_dedup_prefetch();
        fprintf(stderr,
                "ds4: Flash-MoE banked kernels using %s simdgroup_matrix fast path (prefetch=%d, banks=4)\n",
                m5_fast ? "M5" : "generic",
                prefetch);
    }

    return true;
}
