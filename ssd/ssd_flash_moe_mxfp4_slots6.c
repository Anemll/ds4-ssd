/* =========================================================================
 * ssd_flash_moe_mxfp4_slots6.c - SSD Flash-MoE MXFP4 slots6 dispatch helpers.
 * =========================================================================
 *
 * Included by ssd_flash_moe_decode.c.  The FP4 kernels live in metal/moe.metal;
 * this file only selects the C-side MXFP4 slot-bank/record-table dispatch path.
 */

static bool flash_moe_record_table_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_RECORD_TABLE") ||
           env_flag_enabled("DS4_FLASH_MOE_SLOT_RECORD_TABLE") ||
           env_flag_enabled("DS4_FLASH_MOE_ARGUMENT_TABLE");
}

static bool flash_moe_direct_mmap_bank_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_DIRECT_MMAP_BANK") ||
           env_flag_enabled("DS4_FLASH_MOE_MMAP_BANK") ||
           env_flag_enabled("DS4_FLASH_MOE_FILE_BACKED_BANK") ||
           env_flag_enabled("DS4_FLASH_MOE_DIRECT_MMAP_SLOTS6") ||
           env_flag_enabled("DS4_FLASH_MOE_ACTIVE_MMAP_SLOTS6");
}

static bool flash_moe_layer_is_q2_slots6(const ds4_layer_weights *layer) {
    return layer &&
           layer->ffn_gate_exps &&
           layer->ffn_up_exps &&
           layer->ffn_down_exps &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
}

static bool flash_moe_direct_mmap_auto_enabled(const ds4_flash_moe_sidecar *sidecar,
                                               const ds4_layer_weights     *layer) {
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_AUTO");
    if (env && env[0] && atoi(env) == 0) return false;
    if (env_flag_enabled("DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO") ||
        env_flag_enabled("DS4_FLASH_MOE_FORCE_MIXED_SLOT_BANK")) {
        return false;
    }
    const bool forced = env && env[0] && atoi(env) != 0;
    if (!sidecar) return false;
    if (!layer || !layer->ffn_gate_exps || !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    if (flash_moe_layer_is_q2_slots6(layer)) return true;
    if (!forced && sidecar->slot_bank <= 128u) return false;
    return layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
           layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
           layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
}

static bool flash_moe_direct_mmap_slots6_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_DIRECT_MMAP_SLOTS6") ||
           env_flag_enabled("DS4_FLASH_MOE_ACTIVE_MMAP_SLOTS6");
}

static bool flash_moe_direct_mmap_record_slots6_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_RECORD_SLOTS6");
    if (env && env[0]) return atoi(env) != 0;
    if (env_flag_enabled("DS4_FLASH_MOE_DIRECT_MMAP_FAMILY_SLOTS6")) {
        return false;
    }
    return flash_moe_direct_mmap_slots6_enabled() ||
           env_flag_enabled("DS4_FLASH_MOE_ACTIVE_MMAP_RECORDS6");
}

static bool flash_moe_direct_mmap_prewarm_views_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_PREWARM_VIEWS");
    if (env && env[0]) return atoi(env) != 0;
    if (env_flag_enabled("DS4_FLASH_MOE_PREWARM_MMAP_VIEWS")) return true;
    return flash_moe_direct_mmap_record_slots6_enabled();
}

static bool flash_moe_chunked_split_slotwise_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_CHUNKED_SPLIT_SLOTWISE");
    if (env && env[0]) return atoi(env) != 0;
    return false;
}

static bool flash_moe_chunked_same_slotwise_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_CHUNKED_SAME_SLOTWISE");
    if (env && env[0]) return atoi(env) != 0;
    return false;
}

static bool metal_graph_flash_moe_mxfp4_direct_mmap_record_slots6(
        ds4_gpu_graph           *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 active_expert_used,
        uint64_t                 gate_row_bytes,
        uint64_t                 down_row_bytes,
        uint32_t                 expert_in_dim,
        uint32_t                 expert_mid_dim,
        uint32_t                 out_dim) {
    ds4_gpu_tensor *slot_records[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
    bool ok = true;
    for (uint32_t k = 0; ok && k < active_expert_used; k++) {
        const int32_t expert = g->flash_decode_true_ids[il][k];
        if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) return false;
        ok = metal_graph_flash_moe_ensure_direct_mmap_record_view(g, il, (uint32_t)expert);
        if (ok) {
            slot_records[k] = g->flash_expert_bank[il][expert];
            ok = slot_records[k] != NULL;
        }
    }
    if (!ok) return false;

    static int logged_direct_mmap_record_slots6 = 0;
    if (!logged_direct_mmap_record_slots6) {
        fprintf(stderr,
                "ds4: Flash-MoE using direct mmap record slots6 decode path "
                "(six cached file-backed expert records, no slot copies)\n");
        logged_direct_mmap_record_slots6 = 1;
    }
    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    return ds4_gpu_routed_moe_one_slots6_record_tensor(
               g->routed_out,
               g->routed_gate,
               g->routed_up,
               g->routed_mid,
               slot_records,
               flash_layer->family_offset[DS4_FLASH_FAMILY_GATE],
               flash_layer->family_offset[DS4_FLASH_FAMILY_UP],
               flash_layer->family_offset[DS4_FLASH_FAMILY_DOWN],
               layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type,
               gate_row_bytes,
               down_row_bytes,
               expert_in_dim,
               expert_mid_dim,
               out_dim,
               g->router_weights,
               active_expert_used,
               DS4_SWIGLU_CLAMP_EXP,
               g->ffn_norm) != 0;
}

static bool metal_graph_flash_moe_mxfp4_chunked_slots6(
        ds4_gpu_graph           *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 active_expert_used,
        uint64_t                 gate_row_bytes,
        uint64_t                 down_row_bytes,
        uint32_t                 expert_in_dim,
        uint32_t                 expert_mid_dim,
        uint32_t                 out_dim) {
    if (g->flash_chunk_count == 0 || g->flash_chunk_count > 4) return false;

    ds4_gpu_tensor *gate_chunks[4] = { NULL, NULL, NULL, NULL };
    ds4_gpu_tensor *up_chunks[4] = { NULL, NULL, NULL, NULL };
    ds4_gpu_tensor *down_chunks[4] = { NULL, NULL, NULL, NULL };
    uint32_t compact_chunks[4] = { 0, 0, 0, 0 };
    uint32_t compact_count = 0;
    uint32_t chunk_ids[6] = { 0, 0, 0, 0, 0, 0 };
    uint32_t local_slots[6] = { 0, 0, 0, 0, 0, 0 };
    for (uint32_t k = 0; k < active_expert_used; k++) {
        uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots) ||
            chunk >= g->flash_chunk_count) {
            return false;
        }
        (void)chunk_slots;
        uint32_t compact = UINT32_MAX;
        for (uint32_t i = 0; i < compact_count; i++) {
            if (compact_chunks[i] == chunk) {
                compact = i;
                break;
            }
        }
        if (compact == UINT32_MAX) {
            if (compact_count >= 4) return false;
            compact = compact_count;
            compact_chunks[compact_count++] = chunk;
            gate_chunks[compact] = g->flash_chunk_gate_bank[il][chunk];
            up_chunks[compact] = g->flash_chunk_up_bank[il][chunk];
            down_chunks[compact] = g->flash_chunk_down_bank[il][chunk];
            if (!gate_chunks[compact] || !up_chunks[compact] || !down_chunks[compact]) {
                return false;
            }
        }
        chunk_ids[k] = compact;
        local_slots[k] = local_slot;
    }
    if (compact_count == 0) return false;

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    if (compact_count == 1 && flash_moe_chunked_same_slotwise_enabled()) {
        const uint32_t chunk = compact_chunks[0];
        const uint32_t first_slot = chunk * g->flash_chunk_slots;
        uint32_t chunk_slots = g->flash_slot_bank - first_slot;
        if (chunk_slots > g->flash_chunk_slots) chunk_slots = g->flash_chunk_slots;
        if (chunk_slots == 0) return false;
        int32_t local_slot_ids[6] = { 0, 0, 0, 0, 0, 0 };
        for (uint32_t k = 0; k < active_expert_used; k++) {
            local_slot_ids[k] = (int32_t)local_slots[k];
        }
        static int logged_chunked_single_chunk_slotwise = 0;
        if (!logged_chunked_single_chunk_slotwise) {
            fprintf(stderr,
                    "ds4: Flash-MoE using same-chunk slotwise decode2 path "
                    "(append-grown bank, local baked slots)\n");
            logged_chunked_single_chunk_slotwise = 1;
        }
        return ds4_gpu_routed_moe_one_banked_tensor_slotwise_baked(
                   g->routed_out,
                   g->routed_gate,
                   g->routed_up,
                   g->routed_mid,
                   g->routed_down,
                   gate_chunks[0],
                   up_chunks[0],
                   down_chunks[0],
                   chunk_slots,
                   layer->ffn_gate_exps->type,
                   layer->ffn_down_exps->type,
                   (uint64_t)expert_mid_dim * gate_row_bytes,
                   flash_layer->expert_stride,
                   gate_row_bytes,
                   (uint64_t)out_dim * down_row_bytes,
                   flash_layer->expert_stride,
                   down_row_bytes,
                   expert_in_dim,
                   expert_mid_dim,
                   out_dim,
                   il,
                   local_slot_ids,
                   g->router_weights,
                   active_expert_used,
                   DS4_SWIGLU_CLAMP_EXP,
                   g->ffn_norm) != 0;
    }

    if (!flash_moe_chunked_split_slotwise_enabled()) {
        static int logged_chunked_split_slots6 = 0;
        if (!logged_chunked_split_slots6) {
            fprintf(stderr,
                    "ds4: Flash-MoE using split-chunk slots6 decode path "
                    "(append-grown bank, one command-buffer local slot map)\n");
            logged_chunked_split_slots6 = 1;
        }
        return ds4_gpu_routed_moe_one_slots6_chunked_tensor(
                   g->routed_out,
                   g->routed_gate,
                   g->routed_up,
                   g->routed_mid,
                   gate_chunks,
                   up_chunks,
                   down_chunks,
                   compact_count,
                   chunk_ids,
                   local_slots,
                   flash_layer->expert_stride,
                   layer->ffn_gate_exps->type,
                   layer->ffn_down_exps->type,
                   gate_row_bytes,
                   down_row_bytes,
                   expert_in_dim,
                   expert_mid_dim,
                   out_dim,
                   g->router_weights,
                   active_expert_used,
                   DS4_SWIGLU_CLAMP_EXP,
                   g->ffn_norm) != 0;
    }

    if (g->flash_chunk_partial_out) {
        static int logged_chunked_split_slotwise = 0;
        if (!logged_chunked_split_slotwise) {
            fprintf(stderr,
                    "ds4: Flash-MoE using split-chunk slotwise decode2 path "
                    "(append-grown bank, zero-weight inactive local slots; forced)\n");
            logged_chunked_split_slotwise = 1;
        }
        bool ok = true;
        bool have_output = false;
        for (uint32_t ci = 0; ok && ci < compact_count; ci++) {
            const uint32_t chunk = compact_chunks[ci];
            const uint32_t first_slot = chunk * g->flash_chunk_slots;
            uint32_t chunk_slots = g->flash_slot_bank - first_slot;
            if (chunk_slots > g->flash_chunk_slots) chunk_slots = g->flash_chunk_slots;
            if (chunk_slots == 0 ||
                !g->flash_chunk_slot_selected[il][chunk] ||
                !g->flash_chunk_weights[il][chunk]) {
                return false;
            }

            int32_t selected_ids[6] = { 0, 0, 0, 0, 0, 0 };
            float selected_weights[6] = { 0, 0, 0, 0, 0, 0 };
            for (uint32_t k = 0; k < active_expert_used; k++) {
                if (chunk_ids[k] != ci) continue;
                selected_ids[k] = (int32_t)local_slots[k];
                selected_weights[k] = g->flash_decode_weights[il][k];
            }
            ok = ds4_gpu_tensor_write(g->flash_chunk_slot_selected[il][chunk],
                                      0,
                                      selected_ids,
                                      sizeof(selected_ids)) != 0 &&
                 ds4_gpu_tensor_write(g->flash_chunk_weights[il][chunk],
                                      0,
                                      selected_weights,
                                      sizeof(selected_weights)) != 0;
            if (!ok) break;

            ds4_gpu_tensor *chunk_out =
                have_output ? g->flash_chunk_partial_out : g->routed_out;
            ok = ds4_gpu_routed_moe_one_banked_tensor_slotwise(
                     chunk_out,
                     g->routed_gate,
                     g->routed_up,
                     g->routed_mid,
                     g->routed_down,
                     gate_chunks[ci],
                     up_chunks[ci],
                     down_chunks[ci],
                     chunk_slots,
                     layer->ffn_gate_exps->type,
                     layer->ffn_down_exps->type,
                     (uint64_t)expert_mid_dim * gate_row_bytes,
                     flash_layer->expert_stride,
                     gate_row_bytes,
                     (uint64_t)out_dim * down_row_bytes,
                     flash_layer->expert_stride,
                     down_row_bytes,
                     expert_in_dim,
                     expert_mid_dim,
                     out_dim,
                     g->flash_chunk_slot_selected[il][chunk],
                     g->flash_chunk_weights[il][chunk],
                     active_expert_used,
                     DS4_SWIGLU_CLAMP_EXP,
                     g->ffn_norm) != 0;
            if (ok && have_output) {
                ok = ds4_gpu_add_tensor(g->routed_out,
                                        g->routed_out,
                                        g->flash_chunk_partial_out,
                                        out_dim) != 0;
            }
            if (ok) have_output = true;
        }
        if (ok && have_output) return true;
    }

    static int logged_chunked_bank_slots6 = 0;
    if (!logged_chunked_bank_slots6) {
        fprintf(stderr,
                "ds4: Flash-MoE using chunk-bank slots6 decode path "
                "(full chunk buffers, local slot map in-kernel)\n");
        logged_chunked_bank_slots6 = 1;
    }
    return ds4_gpu_routed_moe_one_slots6_chunked_tensor(
               g->routed_out,
               g->routed_gate,
               g->routed_up,
               g->routed_mid,
               gate_chunks,
               up_chunks,
               down_chunks,
               compact_count,
               chunk_ids,
               local_slots,
               flash_layer->expert_stride,
               layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type,
               gate_row_bytes,
               down_row_bytes,
               expert_in_dim,
               expert_mid_dim,
               out_dim,
               g->router_weights,
               active_expert_used,
               DS4_SWIGLU_CLAMP_EXP,
               g->ffn_norm) != 0;
}

static bool metal_graph_flash_moe_mxfp4_record_table_slots6(
        ds4_gpu_graph           *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 active_expert_used,
        uint64_t                 gate_row_bytes,
        uint64_t                 down_row_bytes,
        uint32_t                 expert_in_dim,
        uint32_t                 expert_mid_dim,
        uint32_t                 out_dim) {
    static int logged_record_table = 0;
    if (!logged_record_table) {
        fprintf(stderr,
                "ds4: Flash-MoE using MXFP4 record-table decode path "
                "(slot ids index one per-layer argument buffer)\n");
        logged_record_table = 1;
    }
    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    if (flash_layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE]) {
        return false;
    }
    return ds4_gpu_routed_moe_one_record_table_tensor(
               g->routed_out,
               g->routed_gate,
               g->routed_up,
               g->routed_mid,
               g->flash_record_table[il],
               flash_layer->family_offset[DS4_FLASH_FAMILY_GATE],
               flash_layer->family_offset[DS4_FLASH_FAMILY_UP],
               flash_layer->family_offset[DS4_FLASH_FAMILY_DOWN],
               layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type,
               gate_row_bytes,
               down_row_bytes,
               expert_in_dim,
               expert_mid_dim,
               out_dim,
               g->router_slot_selected,
               g->router_weights,
               active_expert_used,
               DS4_SWIGLU_CLAMP_EXP,
               g->ffn_norm) != 0;
}

static bool metal_graph_flash_moe_mxfp4_record_buffer_slots6(
        ds4_gpu_graph           *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 active_expert_used,
        uint64_t                 gate_row_bytes,
        uint64_t                 down_row_bytes,
        uint32_t                 expert_in_dim,
        uint32_t                 expert_mid_dim,
        uint32_t                 out_dim) {
    ds4_gpu_tensor *slot_records[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
    const uint32_t slot_limit =
        g->flash_per_expert_buffers ? DS4_N_EXPERT : g->flash_slot_bank;
    for (uint32_t k = 0; k < active_expert_used; k++) {
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        if (slot < 0 || slot >= (int32_t)slot_limit ||
            slot >= (int32_t)DS4_MAX_EXPERT) {
            return false;
        }
        if (g->flash_per_slot_buffers &&
            !metal_graph_flash_moe_ensure_per_slot_buffer(g, il, (uint32_t)slot)) {
            return false;
        }
        slot_records[k] = g->flash_expert_bank[il][slot];
        if (!slot_records[k]) return false;
    }

    static int logged_record_slots6 = 0;
    if (!logged_record_slots6) {
        fprintf(stderr,
                "ds4: Flash-MoE using grouped slots6 record-buffer decode path "
                "(six full expert records, family offsets in-kernel)\n");
        logged_record_slots6 = 1;
    }
    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    return ds4_gpu_routed_moe_one_slots6_record_tensor(
               g->routed_out,
               g->routed_gate,
               g->routed_up,
               g->routed_mid,
               slot_records,
               flash_layer->family_offset[DS4_FLASH_FAMILY_GATE],
               flash_layer->family_offset[DS4_FLASH_FAMILY_UP],
               flash_layer->family_offset[DS4_FLASH_FAMILY_DOWN],
               layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type,
               gate_row_bytes,
               down_row_bytes,
               expert_in_dim,
               expert_mid_dim,
               out_dim,
               g->router_weights,
               active_expert_used,
               DS4_SWIGLU_CLAMP_EXP,
               g->ffn_norm) != 0;
}
