/* =========================================================================
 * ssd_flash_moe_resident_prefill.c - Flash-MoE resident MPP/NAX/ANE prefill execution.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

/* Diagnostics include defines cmp_i32_asc(); keep the declaration here because
 * resident prefill planning sorts experts before the diagnostics block is included. */
static int cmp_i32_asc(const void *a, const void *b);

static int flash_moe_run_mpp_int8_safe_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    const uint32_t mpp_m_tile = 64u;
    if (mid_is_f16) *mid_is_f16 = false;
    if (n_tokens == 0) return 0;

    /* The MPP 4.1 native-MXFP4 arm handles partial 64-row tiles correctly
     * (matmul2d clamps to tensor extents; validated down to m=1 with full
     * reference checks + poisoned padding in tests/mxfp4_native_probe).
     * The split-with-ALU-tail workaround below exists for the legacy
     * i8/h_h MPP kernels only, where partial tiles miscalculated. */
    const bool mxfp4_native_dequant_experiment =
        env_flag_enabled("DS4_MXFP4_NATIVE_DEQUANT_PREFILL_EXPERIMENT") &&
        flash_moe_mpp_int8_prefill_requested();
    const bool mxfp4_native_partial_ok =
        gate_type == DS4_TENSOR_MXFP4 && down_type == DS4_TENSOR_MXFP4 &&
        ds4_gpu_mxfp4_native_requested() && ds4_gpu_has_native_mxfp4() &&
        !mxfp4_native_dequant_experiment;

    if ((n_tokens % mpp_m_tile) == 0 || mxfp4_native_partial_ok ||
        flash_moe_mpp_partial_tiles_allowed()) {
        return ds4_gpu_routed_moe_expert_banked_batch_mpp_int8_tensor(out,
                                                                      gate,
                                                                      up,
                                                                      mid,
                                                                      gate_bank,
                                                                      up_bank,
                                                                      down_bank,
                                                                      gate_type,
                                                                      down_type,
                                                                      gate_expert_bytes,
                                                                      gate_row_bytes,
                                                                      down_expert_bytes,
                                                                      down_row_bytes,
                                                                      expert_in_dim,
                                                                      expert_mid_dim,
                                                                      out_dim,
                                                                      selected,
                                                                      weights,
                                                                      clamp,
                                                                      x,
                                                                      n_tokens,
                                                                      mid_is_f16);
    }

    const uint32_t full_tokens = (n_tokens / mpp_m_tile) * mpp_m_tile;
    const uint32_t tail_tokens = n_tokens - full_tokens;
    if (full_tokens == 0) return 0;

    static bool warned = false;
    if (!warned) {
        if (ds4_no_int8_paths_enabled()) {
            fprintf(stderr,
                    "ds4: MPP/NAX partial-tile workaround active "
                    "(--no-int8 path; 64-row MPP/NAX tiles, tail rows use "
                    "legacy GPU; set DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES=1 "
                    "to benchmark old behavior)\n");
        } else {
            fprintf(stderr,
                    "ds4: MPP/NAX int8 partial-tile workaround active "
                    "(64-row MPP tiles, tail rows use legacy GPU; set "
                    "DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES=1 to benchmark old behavior)\n");
        }
        warned = true;
    }

    bool full_mid_is_f16 = false;
    if (!ds4_gpu_routed_moe_expert_banked_batch_mpp_int8_tensor(out,
                                                               gate,
                                                               up,
                                                               mid,
                                                               gate_bank,
                                                               up_bank,
                                                               down_bank,
                                                               gate_type,
                                                               down_type,
                                                               gate_expert_bytes,
                                                               gate_row_bytes,
                                                               down_expert_bytes,
                                                               down_row_bytes,
                                                               expert_in_dim,
                                                               expert_mid_dim,
                                                               out_dim,
                                                               selected,
                                                               weights,
                                                               clamp,
                                                               x,
                                                               full_tokens,
                                                               &full_mid_is_f16)) {
        if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] FAIL routed MPP-int8 FULL part full_tokens=%u (of n=%u)\n", full_tokens, n_tokens);
        return 0;
    }

    ds4_gpu_tensor *out_tail = ds4_gpu_tensor_view(
        out,
        (uint64_t)full_tokens * out_dim * sizeof(float),
        (uint64_t)tail_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *selected_tail = ds4_gpu_tensor_view(
        selected,
        (uint64_t)full_tokens * sizeof(int32_t),
        (uint64_t)tail_tokens * sizeof(int32_t));
    ds4_gpu_tensor *weights_tail = ds4_gpu_tensor_view(
        weights,
        (uint64_t)full_tokens * sizeof(float),
        (uint64_t)tail_tokens * sizeof(float));
    ds4_gpu_tensor *x_tail = ds4_gpu_tensor_view(
        x,
        (uint64_t)full_tokens * expert_in_dim * sizeof(float),
        (uint64_t)tail_tokens * expert_in_dim * sizeof(float));

    bool tail_mid_is_f16 = false;
    const int tail_ok = out_tail && selected_tail && weights_tail && x_tail &&
        ds4_gpu_routed_moe_expert_banked_batch_tensor(out_tail,
                                                      gate,
                                                      up,
                                                      mid,
                                                      gate_bank,
                                                      up_bank,
                                                      down_bank,
                                                      gate_type,
                                                      down_type,
                                                      gate_expert_bytes,
                                                      gate_row_bytes,
                                                      down_expert_bytes,
                                                      down_row_bytes,
                                                      expert_in_dim,
                                                      expert_mid_dim,
                                                      out_dim,
                                                      selected_tail,
                                                      weights_tail,
                                                      clamp,
                                                      x_tail,
                                                      tail_tokens,
                                                      &tail_mid_is_f16) != 0;
    if (!tail_ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr,
        "ds4: [resume-dbg] FAIL routed TAIL part tail_tokens=%u (full=%u n=%u) views(out=%d sel=%d w=%d x=%d)\n",
        tail_tokens, full_tokens, n_tokens, out_tail!=NULL, selected_tail!=NULL, weights_tail!=NULL, x_tail!=NULL);
    ds4_gpu_tensor_free(x_tail);
    ds4_gpu_tensor_free(weights_tail);
    ds4_gpu_tensor_free(selected_tail);
    ds4_gpu_tensor_free(out_tail);

    if (mid_is_f16) *mid_is_f16 = full_mid_is_f16 || tail_mid_is_f16;
    return tail_ok;
}

/* Run ONE routed expert's FFN through the per-expert MPP/NAX GPU path and
 * scatter-add the result into g->batch_routed_out. This is the GPU "cold expert"
 * half of the ANE+NAX hybrid (DS4_RESIDENT_MOE_ANE_NAX_HYBRID): the experts ANE
 * did not take run here on NAX (matmul2d half x half when DS4_RESIDENT_MOE_NAX_HALF
 * is set) instead of being recomputed by the grouped mul_mm_id tail. It mirrors the
 * GPU body of the non-hybrid dedup loop but contains NO ANE path.
 *
 * Scratch is the same as the non-hybrid loop (routed_*_base = g->batch_routed_*,
 * reused per expert) so it must be called only after the ANE jobs that shared
 * g->batch_routed_down have been drained. Manages *commands_open exactly like the
 * non-hybrid loop (ends the previous expert's buffer, begins a fresh one, leaves it
 * open); the caller ends the final buffer. Requires the [prefill_cap..] region of
 * batch_router_selected to hold a zero "selected" view of length >= refs (the
 * partial-tile classic tail reads it). Returns true on success; ORs mid_is_f16 into
 * *mid_is_f16_any and sets *used_nax = whether the MPP/NAX kernel (not the classic
 * fallback) ran. */
/* Superseded by the grouped skip-mask cold tail — the per-expert NAX/int8 dedup kernel
 * it drives corrupts. Kept (unused) as a reference if that kernel is ever fixed. */
__attribute__((unused))
static bool resident_moe_nax_tail_one_expert(
        ds4_gpu_graph           *g,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        int32_t                  expert,
        uint32_t                 begin,
        uint32_t                 refs,
        const int32_t           *ref_tokens,
        const float             *ref_weights,
        uint32_t                 prefill_cap,
        uint32_t                 routed_tmp_rows,
        ds4_gpu_tensor          *routed_gate_base,
        ds4_gpu_tensor          *routed_up_base,
        ds4_gpu_tensor          *routed_mid_base,
        ds4_gpu_tensor          *routed_xout_base,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        uint32_t                 expert_in_dim,
        uint32_t                 expert_mid_dim,
        uint32_t                 out_dim,
        bool                    *commands_open,
        bool                    *mid_is_f16_any,
        bool                    *used_nax) {
    if (used_nax) *used_nax = false;
    bool ok =
        ds4_gpu_tensor_write(g->batch_router_selected, 0,
                             ref_tokens + begin,
                             (uint64_t)refs * sizeof(ref_tokens[0])) != 0 &&
        ds4_gpu_tensor_write(g->batch_router_weights, 0,
                             ref_weights + begin,
                             (uint64_t)refs * sizeof(ref_weights[0])) != 0;
    if (!ok) return false;

    if (*commands_open) {
        ok = ds4_gpu_end_commands() != 0;
        *commands_open = false;
        if (!ok) return false;
    }
    ok = ds4_gpu_begin_commands() != 0;
    *commands_open = ok;
    if (!ok) return false;

    ds4_gpu_tensor *tokens_view = ds4_gpu_tensor_view(
        g->batch_router_selected, 0, (uint64_t)refs * sizeof(int32_t));
    ds4_gpu_tensor *selected_zero_view = ds4_gpu_tensor_view(
        g->batch_router_selected,
        (uint64_t)prefill_cap * sizeof(int32_t),
        (uint64_t)refs * sizeof(int32_t));
    ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
        g->batch_router_weights, 0, (uint64_t)refs * sizeof(float));
    ds4_gpu_tensor *gate_tmp = ds4_gpu_tensor_view(
        routed_gate_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *up_tmp = ds4_gpu_tensor_view(
        routed_up_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *mid_tmp = ds4_gpu_tensor_view(
        routed_mid_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *x_tmp = ds4_gpu_tensor_view(
        routed_xout_base, 0, (uint64_t)refs * expert_in_dim * sizeof(float));
    ds4_gpu_tensor *out_tmp = ds4_gpu_tensor_view(
        routed_xout_base,
        (uint64_t)routed_tmp_rows * out_dim * sizeof(float),
        (uint64_t)refs * out_dim * sizeof(float));
    ds4_gpu_tensor *gate_model = ds4_gpu_model_tensor_view(
        model->map, model->size,
        layer->ffn_gate_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
        gate_expert_bytes);
    ds4_gpu_tensor *up_model = ds4_gpu_model_tensor_view(
        model->map, model->size,
        layer->ffn_up_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
        gate_expert_bytes);
    ds4_gpu_tensor *down_model = ds4_gpu_model_tensor_view(
        model->map, model->size,
        layer->ffn_down_exps->abs_offset + (uint64_t)expert * down_expert_bytes,
        down_expert_bytes);

    bool mid_is_f16 = false;
    int used_mpp = 0;
    ok = tokens_view && selected_zero_view && weights_view &&
         gate_tmp && up_tmp && mid_tmp && x_tmp && out_tmp &&
         gate_model && up_model && down_model &&
         ds4_gpu_gather_rows_f32_tensor(x_tmp, g->batch_ffn_norm,
                                        tokens_view, refs, DS4_N_EMBD) != 0;
    if (ok) {
        used_mpp = flash_moe_run_mpp_int8_safe_tensor(out_tmp, gate_tmp, up_tmp, mid_tmp,
                                                      gate_model, up_model, down_model,
                                                      layer->ffn_gate_exps->type,
                                                      layer->ffn_down_exps->type,
                                                      gate_expert_bytes, gate_row_bytes,
                                                      down_expert_bytes, down_row_bytes,
                                                      expert_in_dim, expert_mid_dim, out_dim,
                                                      selected_zero_view, weights_view,
                                                      DS4_SWIGLU_CLAMP_EXP, x_tmp, refs,
                                                      &mid_is_f16);
        if (!used_mpp) {
            ok = ds4_gpu_routed_moe_expert_banked_batch_tensor(out_tmp, gate_tmp, up_tmp, mid_tmp,
                                                               gate_model, up_model, down_model,
                                                               layer->ffn_gate_exps->type,
                                                               layer->ffn_down_exps->type,
                                                               gate_expert_bytes, gate_row_bytes,
                                                               down_expert_bytes, down_row_bytes,
                                                               expert_in_dim, expert_mid_dim, out_dim,
                                                               selected_zero_view, weights_view,
                                                               DS4_SWIGLU_CLAMP_EXP, x_tmp, refs,
                                                               &mid_is_f16) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out, out_tmp,
                                                 tokens_view, refs, out_dim) != 0;
    }
    if (ok && used_nax) *used_nax = used_mpp != 0;
    if (mid_is_f16 && mid_is_f16_any) *mid_is_f16_any = true;

    ds4_gpu_tensor_free(down_model);
    ds4_gpu_tensor_free(up_model);
    ds4_gpu_tensor_free(gate_model);
    ds4_gpu_tensor_free(out_tmp);
    ds4_gpu_tensor_free(x_tmp);
    ds4_gpu_tensor_free(mid_tmp);
    ds4_gpu_tensor_free(up_tmp);
    ds4_gpu_tensor_free(gate_tmp);
    ds4_gpu_tensor_free(weights_view);
    ds4_gpu_tensor_free(selected_zero_view);
    ds4_gpu_tensor_free(tokens_view);
    return ok;
}

static bool metal_graph_routed_moe_batch_tiled(
        ds4_gpu_graph       *g,
        ds4_gpu_tensor      *out,
        const ds4_model     *model,
        const ds4_layer_weights *layer,
        uint32_t             n_tokens,
        uint64_t             gate_expert_bytes,
        uint64_t             gate_row_bytes,
        uint64_t             down_expert_bytes,
        uint64_t             down_row_bytes,
        uint32_t             expert_in_dim,
        uint32_t             expert_mid_dim,
        uint32_t             out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x,
        bool                *mid_is_f16) {
    if (!g || !out || !model || !layer || !selected || !weights || !x ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        !g->batch_routed_gate || !g->batch_routed_up ||
        !g->batch_routed_mid || !g->batch_routed_down) {
        return false;
    }

    const uint32_t scratch_cap =
        g->batch_routed_scratch_cap ? g->batch_routed_scratch_cap : g->prefill_cap;
    if (scratch_cap == 0) return false;
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;

    bool ok = true;
    bool any_mid_f16 = false;
    for (uint32_t base = 0; ok && base < n_tokens; base += scratch_cap) {
        const uint32_t nb = (n_tokens - base < scratch_cap)
            ? (n_tokens - base)
            : scratch_cap;
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            out,
            (uint64_t)base * out_dim * sizeof(float),
            (uint64_t)nb * out_dim * sizeof(float));
        ds4_gpu_tensor *selected_view = ds4_gpu_tensor_view(
            selected,
            (uint64_t)base * active_expert_used * sizeof(int32_t),
            (uint64_t)nb * active_expert_used * sizeof(int32_t));
        ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
            weights,
            (uint64_t)base * active_expert_used * sizeof(float),
            (uint64_t)nb * active_expert_used * sizeof(float));
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
            x,
            (uint64_t)base * expert_in_dim * sizeof(float),
            (uint64_t)nb * expert_in_dim * sizeof(float));
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(
            g->batch_routed_gate,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(
            g->batch_routed_up,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(
            g->batch_routed_mid,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *down_view = ds4_gpu_tensor_view(
            g->batch_routed_down,
            0,
            (uint64_t)nb * active_expert_used * out_dim * sizeof(float));

        bool tile_mid_f16 = false;
        ok = out_view && selected_view && weights_view && x_view &&
             gate_view && up_view && mid_view && down_view &&
             ds4_gpu_routed_moe_batch_tensor(out_view,
                                             gate_view,
                                             up_view,
                                             mid_view,
                                             down_view,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_exps->abs_offset,
                                             layer->ffn_up_exps->abs_offset,
                                             layer->ffn_down_exps->abs_offset,
                                             layer->ffn_gate_exps->type,
                                             layer->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             expert_in_dim,
                                             expert_mid_dim,
                                             out_dim,
                                             selected_view,
                                             weights_view,
                                             DS4_N_EXPERT,
                                             active_expert_used,
                                             DS4_SWIGLU_CLAMP_EXP,
                                             x_view,
                                             nb,
                                             &tile_mid_f16) != 0;
        if (tile_mid_f16) any_mid_f16 = true;

        ds4_gpu_tensor_free(down_view);
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(x_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(out_view);
    }
    if (mid_is_f16) *mid_is_f16 = any_mid_f16;
    return ok;
}

/*
 * Full prompt prefill for a sidecar whose mixed bank is already resident and
 * identity-mapped.  Unlike the legacy Flash-MoE executor, this never stages
 * individual experts into transient banks: each tile feeds the six routed
 * expert ids straight to the grouped mul_mm_id kernels.
 *
 * The caller validates residency/layout and the per-tile minimum before
 * entering this helper.  That preflight matters: a later small tail cannot
 * fall back after earlier tiles have been encoded into the live command
 * buffer.
 */
static bool metal_graph_flash_moe_resident_grouped_prefill_tiled(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint32_t                   n_tokens,
        uint32_t                   min_tile_tokens,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        bool                      *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!g || !g->flash_moe || !layer || il >= DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        !g->batch_router_selected || !g->batch_router_weights ||
        !g->batch_ffn_norm || !g->batch_routed_out ||
        !g->batch_routed_gate || !g->batch_routed_up ||
        !g->batch_routed_mid || !g->batch_routed_down ||
        !g->flash_gate_bank[il] || !g->flash_up_bank[il] ||
        !g->flash_down_bank[il] || min_tile_tokens == 0) {
        return false;
    }

    const uint32_t scratch_cap =
        g->batch_routed_scratch_cap ? g->batch_routed_scratch_cap : g->prefill_cap;
    if (scratch_cap < min_tile_tokens) return false;
    for (uint32_t base = 0; base < n_tokens; base += scratch_cap) {
        const uint32_t nb = n_tokens - base < scratch_cap ?
            n_tokens - base : scratch_cap;
        if (nb < min_tile_tokens) return false;
    }

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint64_t gate_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : gate_expert_bytes;
    const uint64_t down_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : down_expert_bytes;

    bool ok = true;
    bool any_mid_f16 = false;
    for (uint32_t base = 0; ok && base < n_tokens; base += scratch_cap) {
        const uint32_t nb = n_tokens - base < scratch_cap ?
            n_tokens - base : scratch_cap;
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            g->batch_routed_out,
            (uint64_t)base * out_dim * sizeof(float),
            (uint64_t)nb * out_dim * sizeof(float));
        ds4_gpu_tensor *selected_view = ds4_gpu_tensor_view(
            g->batch_router_selected,
            (uint64_t)base * active_expert_used * sizeof(int32_t),
            (uint64_t)nb * active_expert_used * sizeof(int32_t));
        ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights,
            (uint64_t)base * active_expert_used * sizeof(float),
            (uint64_t)nb * active_expert_used * sizeof(float));
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
            g->batch_ffn_norm,
            (uint64_t)base * expert_in_dim * sizeof(float),
            (uint64_t)nb * expert_in_dim * sizeof(float));
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(
            g->batch_routed_gate,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(
            g->batch_routed_up,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(
            g->batch_routed_mid,
            0,
            (uint64_t)nb * active_expert_used * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *down_view = ds4_gpu_tensor_view(
            g->batch_routed_down,
            0,
            (uint64_t)nb * active_expert_used * out_dim * sizeof(float));

        bool tile_mid_f16 = false;
        ok = out_view && selected_view && weights_view && x_view &&
             gate_view && up_view && mid_view && down_view &&
             ds4_gpu_routed_moe_banked_prefill_tensor(
                 out_view,
                 gate_view,
                 up_view,
                 mid_view,
                 down_view,
                 g->flash_gate_bank[il],
                 g->flash_up_bank[il],
                 g->flash_down_bank[il],
                 g->flash_slot_bank,
                 layer->ffn_gate_exps->type,
                 layer->ffn_down_exps->type,
                 gate_expert_bytes,
                 gate_slot_stride,
                 gate_row_bytes,
                 down_expert_bytes,
                 down_slot_stride,
                 down_row_bytes,
                 expert_in_dim,
                 expert_mid_dim,
                 out_dim,
                 selected_view,
                 weights_view,
                 active_expert_used,
                 DS4_SWIGLU_CLAMP_EXP,
                 x_view,
                 nb,
                 &tile_mid_f16) != 0;
        if (tile_mid_f16) any_mid_f16 = true;

        ds4_gpu_tensor_free(down_view);
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(x_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(out_view);
    }
    if (mid_is_f16) *mid_is_f16 = any_mid_f16;
    return ok;
}

static bool metal_graph_resident_moe_run_mpp_prefill_dedup(
        ds4_gpu_graph       *g,
        const ds4_model     *model,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             n_tokens,
        uint64_t             gate_expert_bytes,
        uint64_t             gate_row_bytes,
        uint64_t             down_expert_bytes,
        uint64_t             down_row_bytes,
        uint32_t             expert_in_dim,
        uint32_t             expert_mid_dim,
        uint32_t             out_dim,
        bool                 force_ane_hybrid) {
    if (!g || !model || !layer || il >= DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        !g->batch_ffn_norm || !g->batch_routed_gate ||
        !g->batch_routed_up || !g->batch_routed_mid ||
        !g->batch_routed_down || !g->batch_routed_out ||
        !g->batch_router_selected || !g->batch_router_weights) {
        return false;
    }
    const uint32_t scratch_cap =
        g->batch_routed_scratch_cap ? g->batch_routed_scratch_cap : g->prefill_cap;
    if (scratch_cap == 0) return false;

    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint64_t n_pairs = (uint64_t)n_tokens * active_expert_used;
    if (n_pairs > SIZE_MAX / sizeof(int32_t) ||
        n_pairs > SIZE_MAX / sizeof(float) ||
        n_pairs > UINT32_MAX) {
        return false;
    }

    int32_t *true_ids = xmalloc((size_t)n_pairs * sizeof(true_ids[0]));
    float *pair_weights = xmalloc((size_t)n_pairs * sizeof(pair_weights[0]));
    int32_t *ref_tokens = xmalloc((size_t)n_pairs * sizeof(ref_tokens[0]));
    float *ref_weights = xmalloc((size_t)n_pairs * sizeof(ref_weights[0]));
    int32_t *zero_selected = xcalloc((size_t)n_tokens, sizeof(zero_selected[0]));
    int32_t counts[DS4_MAX_EXPERT] = { 0 };
    int32_t cursor[DS4_MAX_EXPERT] = { 0 };
    int32_t expert_to_index[DS4_MAX_EXPERT];
    int32_t unique[DS4_MAX_EXPERT];
    int32_t offsets[DS4_MAX_EXPERT + 1];
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) expert_to_index[i] = -1;

    bool ok = ds4_gpu_synchronize() != 0;
    if (ok) {
        ok = ds4_gpu_tensor_read(g->batch_router_selected,
                                 0,
                                 true_ids,
                                 n_pairs * sizeof(true_ids[0])) != 0 &&
             ds4_gpu_tensor_read(g->batch_router_weights,
                                 0,
                                 pair_weights,
                                 n_pairs * sizeof(pair_weights[0])) != 0;
    }

    uint32_t n_unique = 0;
    for (uint64_t i = 0; ok && i < n_pairs; i++) {
        const int32_t expert = true_ids[i];
        if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
            fprintf(stderr, "ds4: resident MPP prefill selected invalid expert id %d in layer %u\n",
                    expert, il);
            ok = false;
            break;
        }
        if (counts[expert]++ == 0) unique[n_unique++] = expert;
    }

    if (ok) {
        const bool partial_tiles_allowed = flash_moe_mpp_partial_tiles_allowed();
        const uint32_t mpp_m_tile = 64u;
        uint64_t mpp_tile_refs = 0;
        uint32_t mpp_candidate_groups = 0;
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
            const uint32_t refs = (uint32_t)counts[i];
            const uint32_t tile_refs = partial_tiles_allowed ?
                refs : (refs / mpp_m_tile) * mpp_m_tile;
            if (tile_refs > 0) {
                mpp_tile_refs += tile_refs;
                mpp_candidate_groups++;
            }
        }
        const double tile_util = n_pairs > 0 ? (double)mpp_tile_refs / (double)n_pairs : 0.0;
        const double min_tile_util = resident_moe_mpp_min_tile_util();
        if (tile_util < min_tile_util) {
            if (!backend_diagnostic_logs_suppressed() &&
                (env_flag_enabled("DS4_RESIDENT_MOE_MPP_STATS") ||
                 env_flag_enabled("DS4_FLASH_MOE_SCHED_STATS") ||
                 env_flag_enabled("DS4_FLASH_MOE_PROFILE"))) {
                fprintf(stderr,
                        "ds4: resident MPP/NAX prefill skipped layer=%u "
                        "tile_refs=%" PRIu64 " total_refs=%" PRIu64
                        " tile_util=%.1f%% min=%.1f%% candidate_groups=%u\n",
                        il,
                        mpp_tile_refs,
                        n_pairs,
                        100.0 * tile_util,
                        100.0 * min_tile_util,
                        mpp_candidate_groups);
            }
            ok = false;
        }
    }

    if (!ok) {
        free(zero_selected);
        free(ref_weights);
        free(ref_tokens);
        free(pair_weights);
        free(true_ids);
        return false;
    }

    const bool use_ane_hybrid =
        force_ane_hybrid || env_flag_enabled("DS4_RESIDENT_MOE_ANE_HYBRID");

    static bool announced = false;
    if (!announced) {
        if (backend_stats_logs_enabled()) {
            if (use_ane_hybrid) {
                const uint32_t min_refs =
                    (uint32_t)atoi(getenv("DS4_RESIDENT_MOE_ANE_MIN_REFS") ?: "128");
                const bool ane_nax =
                    env_flag_enabled("DS4_RESIDENT_MOE_ANE_NAX_HYBRID");
                const bool ane_nax_half =
                    ane_nax && env_flag_enabled("DS4_RESIDENT_MOE_NAX_HALF");
                const bool ane_alu =
                    !ane_nax && env_flag_enabled("DS4_RESIDENT_MOE_ANE_ALU_HYBRID");
                const bool ane_gpu_gather =
                    !ane_nax &&
                    env_flag_enabled("DS4_RESIDENT_MOE_ANE_GPU_TAIL_GATHER_SCATTER");
                const bool ane_gpu_gather_overlap =
                    ane_gpu_gather &&
                    env_flag_enabled("DS4_RESIDENT_MOE_ANE_GPU_TAIL_OVERLAP");
                const char *cold = ane_nax_half ? "NAX-half" :
                    ane_nax ? "MPP/NAX-int8" :
                    ane_alu ? "int8/ALU (per-expert)" :
                    ane_gpu_gather_overlap ? "classic GPU gather/scatter overlapped" :
                    ane_gpu_gather ? "classic GPU gather/scatter" :
                    "classic GPU (grouped)";
                fprintf(stderr,
                        "ds4: resident routed MoE using ANE/%s DeDup prefill "
                        "(ANE refs >= %u, cold tail: %s)\n",
                        ane_nax ? "NAX" : ane_alu ? "ALU" : "GPU",
                        min_refs,
                        cold);
            } else {
                fprintf(stderr,
                        "ds4: resident routed MoE using MPP/NAX int8 DeDup prefill "
                        "(coverage-gated, no sidecar/SSD staging)\n");
            }
        }
        announced = true;
    }

    if (ok) {
        qsort(unique, n_unique, sizeof(unique[0]), cmp_i32_asc);
        offsets[0] = 0;
        for (uint32_t i = 0; i < n_unique; i++) {
            const int32_t expert = unique[i];
            expert_to_index[expert] = (int32_t)i;
            offsets[i + 1] = offsets[i] + counts[expert];
            cursor[expert] = offsets[i];
        }

        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t k = 0; k < active_expert_used; k++) {
                const uint64_t pair = (uint64_t)t * active_expert_used + k;
                const int32_t expert = true_ids[pair];
                const int32_t idx = expert_to_index[expert];
                if (idx < 0) {
                    ok = false;
                    break;
                }
                const int32_t dst = cursor[expert]++;
                ref_tokens[dst] = (int32_t)t;
                ref_weights[dst] = pair_weights[pair];
            }
            if (!ok) break;
        }
    }

    /* ANE+NAX hybrid: ANE takes the hot experts (as in the ANE+GPU hybrid), but the
     * remaining "cold" experts run on NAX (matmul2d half x half) instead of the
     * grouped mul_mm_id GPU tail. Requires NAX-half (the GPU half is the per-expert
     * NAX kernel) and the ANE hybrid itself. NAX runs on the GPU, so ANE-engine and
     * NAX overlap is the throughput goal; this first cut keeps them sequential
     * (ANE drains, then NAX) for correctness — see resident_moe_nax_tail_one_expert. */
    /* ANE+NAX: ANE hot experts + resident-NAX cold tail. The cold backend (NAX-half vs
     * NAX-int8) is chosen by DS4_RESIDENT_MOE_NAX_HALF inside the grouped path — NOT gated
     * here. (Half + the compact bridge is forced onto the slow pairrow/count-readback path
     * — ds4_metal.m's want_nax_half&&compact_active trap — so NAX_HALF=0 / int8 cold uses
     * the faster route; both are valid ANE+NAX.) */
    const bool use_ane_nax = use_ane_hybrid &&
        env_flag_enabled("DS4_RESIDENT_MOE_ANE_NAX_HYBRID");
    /* ANE+ALU hybrid: same family as ANE+NAX but the cold experts run classic int8
     * (no NAX-half). All three ANE variants now share ONE correct cold-tail mechanism:
     * the grouped skip-mask path (see the cold tail below). ane_nax additionally lifts
     * the skip-mask NAX block so its cold experts use resident NAX-half; ane_gpu/ane_alu
     * run classic mul_mm_id. On M5 Max (single ANE cluster) ANE is GPU-encode-bound so
     * the win is small; the real benefit is M3 Ultra (dual-cluster). */
    const bool use_ane_alu = use_ane_hybrid &&
        env_flag_enabled("DS4_RESIDENT_MOE_ANE_ALU_HYBRID") &&
        !use_ane_nax;
    (void)use_ane_alu;
    /* A/B probe for resident ane_gpu cold-tail execution. Default uses the grouped
     * skip-mask tail. This opt-in routes only the cold GPU tail through the older
     * per-expert gather/compute/scatter path while keeping the same ANE hot-expert
     * partition, so we can compare tail mechanics without changing ANE selection. */
    const bool use_ane_gpu_tail_gather_scatter = use_ane_hybrid &&
        !use_ane_nax &&
        env_flag_enabled("DS4_RESIDENT_MOE_ANE_GPU_TAIL_GATHER_SCATTER");
    const bool use_ane_gpu_tail_overlap =
        use_ane_gpu_tail_gather_scatter &&
        env_flag_enabled("DS4_RESIDENT_MOE_ANE_GPU_TAIL_OVERLAP");
    /* ANE-parallel-NAX overlap (default off): dispatch the NAX cold tail concurrently
     * with the in-flight ANE jobs (encode + flush before draining ANE) instead of after
     * ANE drains. Wall -> max(T_ANE, T_NAX) instead of T_ANE + T_NAX. The ANE-finish
     * output lives in the disjoint [scratch_cap..] half of batch_routed_down, so the
     * grouped cold tail (which uses [0..scratch_cap)) can run at the same time. */
    const bool use_ane_nax_overlap = use_ane_nax &&
        env_flag_enabled("DS4_RESIDENT_MOE_ANE_NAX_OVERLAP");
    const bool use_ane_tail_overlap =
        use_ane_nax_overlap || use_ane_gpu_tail_overlap;
    /* Throughput-balanced ANE/NAX split (default off): DS4_RESIDENT_MOE_ANE_NAX_FRAC =
     * the fraction of routed TOKENS (refs) to put on ANE, the rest on NAX. Tuning this is
     * how you hit max parallelism: the split where T_ANE == T_NAX (equal finish time) is the
     * one that minimizes the wall max(T_ANE, T_NAX). >0 replaces the refs-band ANE gate with
     * a largest-expert-first assignment that fills ANE up to frac*total_refs. */
    double ane_nax_frac = 0.0;
    if (use_ane_nax) {
        const char *r = getenv("DS4_RESIDENT_MOE_ANE_NAX_FRAC");
        if (r && r[0]) { double v = atof(r); if (v > 0.0 && isfinite(v)) ane_nax_frac = v > 1.0 ? 1.0 : v; }
    }
    const bool use_ane_nax_balance = ane_nax_frac > 0.0;
    const uint64_t router_scratch_rows = (uint64_t)g->prefill_cap * active_expert_used;
    if (ok &&
        (router_scratch_rows < (uint64_t)g->prefill_cap + n_tokens ||
         (uint64_t)g->prefill_cap > UINT64_MAX / sizeof(int32_t))) {
        ok = false;
    }
    if (ok) {
        ok = ds4_gpu_tensor_fill_f32(g->batch_routed_out,
                                     0.0f,
                                     (uint64_t)n_tokens * out_dim) != 0;
        /* Only the non-hybrid per-expert loop reads the zeroed "selected" view at
         * [prefill_cap..] (its partial-tile classic tail). Every ANE hybrid uses the
         * grouped skip-mask cold tail, which reads selected[0..n_tokens*USED) — a range
         * that OVERLAPS [prefill_cap..] (USED=8), so writing the zero view there would
         * corrupt the routed selection. Skip the write for all ANE hybrids. */
        if (ok && !use_ane_hybrid) {
            ok = ds4_gpu_tensor_write(g->batch_router_selected,
                                      (uint64_t)g->prefill_cap * sizeof(int32_t),
                                      zero_selected,
                                      (uint64_t)n_tokens * sizeof(zero_selected[0])) != 0;
        }
    }

    uint32_t mpp_groups = 0;
    uint32_t fallback_groups = 0;
    uint64_t mpp_refs = 0;
    uint64_t fallback_refs = 0;
    bool commands_open = false;
    enum { DS4_RESIDENT_ANE_PENDING_MAX = 256 };
    ds4_gpu_ane_prefill_job *pending_ane_jobs[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    ds4_gpu_tensor *pending_ane_tokens[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    ds4_gpu_tensor *pending_ane_x[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    ds4_gpu_tensor *pending_ane_gate[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    ds4_gpu_tensor *pending_ane_up[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    ds4_gpu_tensor *pending_ane_down[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    uint32_t pending_ane_refs[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    uint32_t pending_ane_experts[DS4_RESIDENT_ANE_PENDING_MAX] = { 0 };
    uint32_t pending_ane_count = 0;
    ds4_gpu_tensor *resident_ane_out = NULL;
    uint32_t ane_groups = 0;
    uint64_t ane_refs = 0;
    uint32_t ane_start_failures = 0;
    uint32_t ane_scatter_failures = 0;
    const uint32_t routed_tmp_rows = scratch_cap;
    ds4_gpu_tensor *routed_gate_base = g->batch_routed_gate;
    ds4_gpu_tensor *routed_up_base = g->batch_routed_up;
    ds4_gpu_tensor *routed_mid_base = g->batch_routed_mid;
    ds4_gpu_tensor *routed_xout_base = g->batch_routed_down;

    const uint32_t ane_min_refs_local =
        (uint32_t)atoi(getenv("DS4_RESIDENT_MOE_ANE_MIN_REFS") ?: "128");
    const uint32_t ane_max_refs_local =
        (uint32_t)atoi(getenv("DS4_RESIDENT_MOE_ANE_MAX_REFS") ?: "1024");
    const uint32_t ane_max_groups_local =
        (uint32_t)atoi(getenv("DS4_RESIDENT_MOE_ANE_HYBRID_MAX_EXPERTS") ?: "0");
    uint32_t ane_queue_depth = 4;
    {
        const char *q = getenv("DS4_RESIDENT_MOE_ANE_QUEUE");
        if (!q || !q[0]) q = getenv("DS4_FLASH_MOE_ANE_OUTPUT_QUEUE");
        if (q && q[0]) {
            long v = strtol(q, NULL, 10);
            if (v > 0 && v <= DS4_RESIDENT_ANE_PENDING_MAX) {
                ane_queue_depth = (uint32_t)v;
            }
        }
    }
    if (ane_queue_depth == 0) ane_queue_depth = 1;

#define DS4_RESIDENT_FINISH_ANE_AT(_idx) do { \
        const uint32_t _ai = (_idx); \
        if (pending_ane_jobs[_ai]) { \
            if (commands_open) { \
                ok = ds4_gpu_flush_commands() != 0 && ok; \
            } \
            if (!resident_ane_out) { \
                resident_ane_out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float)); \
            } \
            bool _ane_mid_is_f16 = false; \
            const int _finish_ok = resident_ane_out && \
                ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor( \
                    pending_ane_jobs[_ai], resident_ane_out, &_ane_mid_is_f16); \
            pending_ane_jobs[_ai] = NULL; \
            if (ok && _finish_ok) { \
                ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out, \
                                                         resident_ane_out, \
                                                         pending_ane_tokens[_ai], \
                                                         pending_ane_refs[_ai], \
                                                         out_dim) != 0; \
                if (!ok) ane_scatter_failures++; \
                if (_ane_mid_is_f16) g->batch_routed_mid_is_f16 = true; \
            } else { \
                ok = false; \
                ane_scatter_failures++; \
            } \
        } \
        ds4_gpu_tensor_free(pending_ane_down[_ai]); \
        ds4_gpu_tensor_free(pending_ane_up[_ai]); \
        ds4_gpu_tensor_free(pending_ane_gate[_ai]); \
        ds4_gpu_tensor_free(pending_ane_x[_ai]); \
        ds4_gpu_tensor_free(pending_ane_tokens[_ai]); \
        pending_ane_down[_ai] = NULL; \
        pending_ane_up[_ai] = NULL; \
        pending_ane_gate[_ai] = NULL; \
        pending_ane_x[_ai] = NULL; \
        pending_ane_tokens[_ai] = NULL; \
        pending_ane_refs[_ai] = 0; \
        pending_ane_experts[_ai] = 0; \
    } while (0)

#define DS4_RESIDENT_FINISH_ANE_HEAD() do { \
        if (pending_ane_count > 0) { \
            DS4_RESIDENT_FINISH_ANE_AT(0); \
            for (uint32_t _mi = 1; _mi < pending_ane_count; _mi++) { \
                pending_ane_jobs[_mi - 1u] = pending_ane_jobs[_mi]; \
                pending_ane_tokens[_mi - 1u] = pending_ane_tokens[_mi]; \
                pending_ane_x[_mi - 1u] = pending_ane_x[_mi]; \
                pending_ane_gate[_mi - 1u] = pending_ane_gate[_mi]; \
                pending_ane_up[_mi - 1u] = pending_ane_up[_mi]; \
                pending_ane_down[_mi - 1u] = pending_ane_down[_mi]; \
                pending_ane_refs[_mi - 1u] = pending_ane_refs[_mi]; \
                pending_ane_experts[_mi - 1u] = pending_ane_experts[_mi]; \
            } \
            pending_ane_count--; \
            pending_ane_jobs[pending_ane_count] = NULL; \
            pending_ane_tokens[pending_ane_count] = NULL; \
            pending_ane_x[pending_ane_count] = NULL; \
            pending_ane_gate[pending_ane_count] = NULL; \
            pending_ane_up[pending_ane_count] = NULL; \
            pending_ane_down[pending_ane_count] = NULL; \
            pending_ane_refs[pending_ane_count] = 0; \
            pending_ane_experts[pending_ane_count] = 0; \
        } \
    } while (0)

    if (ok && use_ane_hybrid) {
        uint8_t ane_mask[DS4_MAX_EXPERT] = { 0 };
        typedef struct {
            ds4_gpu_ane_prefill_job *job;
            uint32_t begin;
            uint32_t refs;
            int32_t expert;
        } resident_ane_pending_job;
        uint32_t ane_queue_cap = 8u;
        const char *ane_queue_env = getenv("DS4_RESIDENT_MOE_ANE_QUEUE");
        if (!ane_queue_env || !ane_queue_env[0]) {
            ane_queue_env = getenv("DS4_FLASH_MOE_ANE_OUTPUT_QUEUE");
        }
        if (ane_queue_env && ane_queue_env[0]) {
            char *end = NULL;
            unsigned long parsed = strtoul(ane_queue_env, &end, 10);
            if (end != ane_queue_env && parsed > 0 && parsed <= 12ul) {
                ane_queue_cap = (uint32_t)parsed;
            }
        }
        const bool ane_defer_dequant =
            env_flag_enabled("DS4_FLASH_MOE_ANE_DEFER_DEQUANT_COMMIT");
        resident_ane_pending_job *ane_queue =
            xcalloc((size_t)ane_queue_cap, sizeof(ane_queue[0]));
        uint32_t ane_queue_n = 0;
        ds4_gpu_tensor *dedup_tokens_gpu = ds4_gpu_tensor_alloc(n_pairs * sizeof(ref_tokens[0]));
        ds4_gpu_tensor *dedup_weights_gpu = ds4_gpu_tensor_alloc(n_pairs * sizeof(ref_weights[0]));
        ok = dedup_tokens_gpu && dedup_weights_gpu &&
             ds4_gpu_tensor_write(dedup_tokens_gpu,
                                  0,
                                  ref_tokens,
                                  n_pairs * sizeof(ref_tokens[0])) != 0 &&
             ds4_gpu_tensor_write(dedup_weights_gpu,
                                  0,
                                  ref_weights,
                                  n_pairs * sizeof(ref_weights[0])) != 0;

        /* PREDEQUANT + the ANE‖NAX overlap reorder deadlock (the batched int8-bank
         * dequant leaves the command stream in a state the deferred-drain overlap can't
         * safely flush). Until that interaction is fixed, predequant is disabled whenever
         * overlap is on — overlap is the shipped path; predequant is an orthogonal opt. */
        bool ane_predequant_requested =
            env_flag_enabled("DS4_RESIDENT_MOE_ANE_PREDEQUANT");
        if (ane_predequant_requested && use_ane_tail_overlap) {
            static bool warned_predequant_overlap = false;
            if (!warned_predequant_overlap) {
                warned_predequant_overlap = true;
                fprintf(stderr,
                        "ds4: DS4_RESIDENT_MOE_ANE_PREDEQUANT is ignored while "
                        "resident ANE tail overlap is enabled (known deadlock); "
                        "running per-expert dequant instead\n");
            }
            ane_predequant_requested = false;
        }
        ds4_gpu_tensor *ane_gate_i8_bank = NULL;
        ds4_gpu_tensor *ane_up_i8_bank = NULL;
        ds4_gpu_tensor *ane_down_i8_bank = NULL;
        const uint64_t ane_gate_i8_expert_bytes =
            (uint64_t)expert_in_dim * expert_mid_dim;
        const uint64_t ane_down_i8_expert_bytes =
            (uint64_t)expert_mid_dim * out_dim;
        int32_t ane_predequant_slot[DS4_MAX_EXPERT];
        int32_t ane_predequant_experts[DS4_MAX_EXPERT];
        uint32_t n_ane_predequant = 0;
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) ane_predequant_slot[i] = -1;
        if (ok && ane_predequant_requested) {
            for (uint32_t ui = 0; ui < n_unique; ui++) {
                const int32_t expert = unique[ui];
                const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
                if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
                if (refs == 0 || refs > scratch_cap ||
                    refs < ane_min_refs_local ||
                    (ane_max_refs_local != 0u && refs > ane_max_refs_local)) {
                    continue;
                }
                ane_predequant_slot[(uint32_t)expert] = (int32_t)n_ane_predequant;
                ane_predequant_experts[n_ane_predequant++] = expert;
            }
            if (n_ane_predequant != 0 &&
                !ds4_gpu_moe_predequant_i8_experts(&ane_gate_i8_bank,
                                                   &ane_up_i8_bank,
                                                   &ane_down_i8_bank,
                                                   ane_predequant_experts,
                                                   n_ane_predequant,
                                                   model->map,
                                                   model->size,
                                                   layer->ffn_gate_exps->abs_offset,
                                                   layer->ffn_up_exps->abs_offset,
                                                   layer->ffn_down_exps->abs_offset,
                                                   layer->ffn_gate_exps->type,
                                                   layer->ffn_down_exps->type,
                                                   gate_expert_bytes,
                                                   gate_row_bytes,
                                                   down_expert_bytes,
                                                   down_row_bytes,
                                                   expert_in_dim,
                                                   expert_mid_dim,
                                                   out_dim)) {
                static bool warned_predequant = false;
                if (!warned_predequant && !backend_diagnostic_logs_suppressed()) {
                    fprintf(stderr,
                            "ds4: resident ANE compact predequant i8 bank unavailable; "
                            "falling back to per-expert ANE dequant\n");
                    warned_predequant = true;
                }
                for (uint32_t i = 0; i < DS4_N_EXPERT; i++) ane_predequant_slot[i] = -1;
                ane_gate_i8_bank = NULL;
                ane_up_i8_bank = NULL;
                ane_down_i8_bank = NULL;
            }
        }
        const bool use_ane_predequant =
            ane_gate_i8_bank && ane_up_i8_bank && ane_down_i8_bank;

        #define DS4_RESIDENT_ANE_FINISH_ONE() do { \
            if (ane_queue_n != 0) { \
                if (ane_defer_dequant && commands_open) { \
                    ok = ds4_gpu_flush_commands() != 0 && ok; \
                    commands_open = ok; \
                } \
                resident_ane_pending_job _p = ane_queue[0]; \
                if (ane_queue_n > 1) { \
                    memmove(ane_queue, ane_queue + 1, \
                            (size_t)(ane_queue_n - 1u) * sizeof(ane_queue[0])); \
                } \
                ane_queue_n--; \
                ds4_gpu_tensor *_tokens_view = ds4_gpu_tensor_view( \
                    dedup_tokens_gpu, \
                    (uint64_t)_p.begin * sizeof(int32_t), \
                    (uint64_t)_p.refs * sizeof(int32_t)); \
                ds4_gpu_tensor *_out_tmp = ds4_gpu_tensor_view( \
                    g->batch_routed_down, \
                    (uint64_t)scratch_cap * out_dim * sizeof(float), \
                    (uint64_t)_p.refs * out_dim * sizeof(float)); \
                bool _mid_is_f16 = false; \
                bool _used_ane = _tokens_view && _out_tmp && \
                    ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor( \
                        _p.job, _out_tmp, &_mid_is_f16) != 0; \
                if (!_tokens_view || !_out_tmp) { \
                    (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor( \
                        _p.job, NULL, NULL); \
                    _used_ane = false; \
                } \
                if (_used_ane) { \
                    const bool _scatter_ok = \
                        ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out, \
                                                            _out_tmp, \
                                                            _tokens_view, \
                                                            _p.refs, \
                                                            out_dim) != 0; \
                    ok = _scatter_ok && ok; \
                    if (_scatter_ok) { \
                        ane_mask[(uint32_t)_p.expert] = 1u; \
                        ane_groups++; \
                        ane_refs += _p.refs; \
                    } else { \
                        ane_scatter_failures++; \
                    } \
                } else { \
                    ane_scatter_failures++; \
                } \
                if (_mid_is_f16) g->batch_routed_mid_is_f16 = true; \
                ds4_gpu_tensor_free(_out_tmp); \
                ds4_gpu_tensor_free(_tokens_view); \
            } \
        } while (0)

        /* Throughput-balanced ANE/NAX assignment by TOKEN count. When
         * DS4_RESIDENT_MOE_ANE_NAX_FRAC>0, decide ANE vs NAX per expert here (before the
         * submit loop) instead of by the refs band: count the total ANE-eligible tokens,
         * then assign experts largest-first to ANE until ANE's token total reaches
         * frac*total — so ~frac of the routed tokens run on ANE and (1-frac) on NAX.
         * Sweeping frac finds the split where the two engines finish together (max overlap).
         * Eligibility still honors the ANE batch cap [min,max]; the rest go to the NAX tail. */
        uint8_t ane_assign[DS4_MAX_EXPERT];
        if (use_ane_nax_balance) {
            memset(ane_assign, 0, sizeof(ane_assign));
            uint32_t ord[DS4_MAX_EXPERT];
            uint32_t nord = 0;
            uint64_t total_eligible_refs = 0;
            for (uint32_t ui = 0; ui < n_unique; ui++) {
                const int32_t e = unique[ui];
                const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
                if (e < 0 || e >= (int32_t)DS4_N_EXPERT) continue;
                if (refs == 0 || refs > scratch_cap) continue;
                if (refs < ane_min_refs_local) continue;
                if (ane_max_refs_local != 0u && refs > ane_max_refs_local) continue;
                ord[nord++] = ui;
                total_eligible_refs += refs;
            }
            for (uint32_t i = 1; i < nord; i++) {
                const uint32_t k = ord[i];
                const uint32_t kr = (uint32_t)(offsets[k + 1] - offsets[k]);
                int j = (int)i - 1;
                while (j >= 0 &&
                       (uint32_t)(offsets[ord[j] + 1] - offsets[ord[j]]) < kr) {
                    ord[j + 1] = ord[j];
                    j--;
                }
                ord[j + 1] = k;
            }
            const double target_ane_refs = ane_nax_frac * (double)total_eligible_refs;
            double load_ane = 0.0;
            uint32_t ane_cnt = 0;
            for (uint32_t i = 0; i < nord; i++) {
                const uint32_t ui = ord[i];
                const int32_t e = unique[ui];
                const double refs = (double)(offsets[ui + 1] - offsets[ui]);
                /* stop adding to ANE once we've reached the token target; cap on group count */
                if (load_ane >= target_ane_refs) break;
                if (ane_max_groups_local != 0u && ane_cnt >= ane_max_groups_local) break;
                ane_assign[(uint32_t)e] = 1u;
                load_ane += refs;
                ane_cnt++;
            }
        }

        for (uint32_t ui = 0; ok && ui < n_unique; ui++) {
            const int32_t expert = unique[ui];
            const uint32_t begin = (uint32_t)offsets[ui];
            const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
            if (refs == 0) continue;
            if (refs > n_tokens || expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
                ok = false;
                break;
            }
            if (use_ane_nax_balance) {
                /* balancer decided the split up front; NAX cold tail takes the rest */
                if (!ane_assign[(uint32_t)expert]) continue;
            } else if (refs > scratch_cap ||
                refs < ane_min_refs_local ||
                (ane_max_refs_local != 0u && refs > ane_max_refs_local) ||
                (ane_max_groups_local != 0 && ane_groups + ane_queue_n >= ane_max_groups_local)) {
                continue;
            }
            if (ane_queue_n == ane_queue_cap) {
                DS4_RESIDENT_ANE_FINISH_ONE();
                if (!ok) break;
            }

            ds4_gpu_tensor *tokens_view = ds4_gpu_tensor_view(
                dedup_tokens_gpu,
                (uint64_t)begin * sizeof(int32_t),
                (uint64_t)refs * sizeof(int32_t));
            ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
                dedup_weights_gpu,
                (uint64_t)begin * sizeof(float),
                (uint64_t)refs * sizeof(float));
            ds4_gpu_tensor *x_tmp = ds4_gpu_tensor_view(
                g->batch_routed_down, 0, (uint64_t)refs * expert_in_dim * sizeof(float));
            const int32_t ane_slot = (use_ane_predequant &&
                                      expert >= 0 &&
                                      expert < (int32_t)DS4_N_EXPERT) ?
                ane_predequant_slot[(uint32_t)expert] : -1;
            const bool use_ane_predequant_for_expert = ane_slot >= 0;
            ds4_gpu_tensor *gate_model = use_ane_predequant_for_expert ?
                ds4_gpu_tensor_view(ane_gate_i8_bank,
                                    (uint64_t)ane_slot * ane_gate_i8_expert_bytes,
                                    ane_gate_i8_expert_bytes) :
                ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_gate_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
            ds4_gpu_tensor *up_model = use_ane_predequant_for_expert ?
                ds4_gpu_tensor_view(ane_up_i8_bank,
                                    (uint64_t)ane_slot * ane_gate_i8_expert_bytes,
                                    ane_gate_i8_expert_bytes) :
                ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_up_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
            ds4_gpu_tensor *down_model = use_ane_predequant_for_expert ?
                ds4_gpu_tensor_view(ane_down_i8_bank,
                                    (uint64_t)ane_slot * ane_down_i8_expert_bytes,
                                    ane_down_i8_expert_bytes) :
                ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_down_exps->abs_offset + (uint64_t)expert * down_expert_bytes,
                    down_expert_bytes);

            ok = tokens_view && weights_view && x_tmp &&
                 gate_model && up_model && down_model;
            if (ok && !commands_open) {
                ok = ds4_gpu_begin_commands() != 0;
                commands_open = ok;
            }
            if (ok) {
                ok = ds4_gpu_gather_rows_f32_tensor(x_tmp,
                                                    g->batch_ffn_norm,
                                                    tokens_view,
                                                    refs,
                                                    DS4_N_EMBD) != 0;
            }
            if (ok) {
                ds4_gpu_ane_prefill_job *ane_job =
                    ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(
                        gate_model,
                        up_model,
                        down_model,
                        use_ane_predequant_for_expert ? DS4_GPU_TENSOR_I8_BANK : layer->ffn_gate_exps->type,
                        use_ane_predequant_for_expert ? DS4_GPU_TENSOR_I8_BANK : layer->ffn_down_exps->type,
                        use_ane_predequant_for_expert ? ane_gate_i8_expert_bytes : gate_expert_bytes,
                        use_ane_predequant_for_expert ? expert_in_dim : gate_row_bytes,
                        use_ane_predequant_for_expert ? ane_down_i8_expert_bytes : down_expert_bytes,
                        use_ane_predequant_for_expert ? expert_mid_dim : down_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        out_dim,
                        weights_view,
                        x_tmp,
                        refs);
                if (ane_job) {
                    ane_queue[ane_queue_n++] = (resident_ane_pending_job){
                        .job = ane_job,
                        .begin = begin,
                        .refs = refs,
                        .expert = expert,
                    };
                    /* Overlap needs the ANE partition known BEFORE draining so the NAX cold
                     * tail can be dispatched concurrently. Mark the expert ANE-handled at
                     * queue time; FINISH_ONE re-marks it (idempotent) and does the stats. A
                     * job that later fails to finish forces ok=false (layer aborts), so a
                     * pre-marked-but-uncomputed expert never yields silently-wrong output. */
                    if (use_ane_tail_overlap) ane_mask[(uint32_t)expert] = 1u;
                } else {
                    ane_start_failures++;
                }
            }

            ds4_gpu_tensor_free(down_model);
            ds4_gpu_tensor_free(up_model);
            ds4_gpu_tensor_free(gate_model);
            ds4_gpu_tensor_free(x_tmp);
            ds4_gpu_tensor_free(weights_view);
            ds4_gpu_tensor_free(tokens_view);
        }
        /* Non-overlap (sequential): drain ANE fully here, then run the cold tail after.
         * Overlap: DEFER the drain until after the cold tail is committed, so GPU work
         * and the ANE engine run concurrently. ane_mask was set at QUEUE time in
         * overlap mode, so the cold partition is already known. */
        if (!use_ane_tail_overlap) {
            while (ok && ane_queue_n != 0) {
                DS4_RESIDENT_ANE_FINISH_ONE();
            }
            if (commands_open) {
                ok = ds4_gpu_end_commands() != 0 && ok;
                commands_open = false;
            }
            if (!ok) {
                while (ane_queue_n != 0) {
                    resident_ane_pending_job p = ane_queue[--ane_queue_n];
                    (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
                        p.job, NULL, NULL);
                }
            }
        }

        uint32_t cold_groups = 0;
        uint64_t cold_refs = 0;
        for (uint32_t ui = 0; ok && ui < n_unique; ui++) {
            const int32_t expert = unique[ui];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
                ok = false;
                break;
            }
            if (ane_mask[(uint32_t)expert]) continue;
            const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
            if (refs == 0) continue;
            cold_groups++;
            cold_refs += refs;
        }
        /* ANE hybrid cold-expert tail. ALL ANE variants (ane_gpu, ane_nax, ane_alu)
         * share the SAME mechanism: keep the full routed selection/weights, set the ANE
         * skip mask, and run the cold experts through the grouped routed-MoE path
         * (metal_graph_routed_moe_batch_tiled -> ds4_gpu_routed_moe_batch_tensor). The
         * skip mask zeroes only the ANE experts' per-expert COUNTS (and the resident
         * per-expert loop checks the mask directly), so the grouped kernel skips them
         * while every token's weights stay correctly normalized — the ANE share was
         * already scatter-added into batch_routed_out. The cold compute backend is
         * chosen INSIDE the grouped path by env flags: ane_nax (ANE_NAX_HYBRID + NAX_HALF)
         * lifts the skip-mask block so the cold tail runs resident NAX-half (odd/partial
         * chunks fall to int8 via the %64 floor); ane_gpu/ane_alu run classic mul_mm_id.
         * NOTE: the per-expert masked-selection tail (writing -1 into the selection) was
         * removed — it broke weight renormalization and produced garbage on odd chunks. */
        if (use_ane_nax) {
            mpp_groups += cold_groups;
            mpp_refs += cold_refs;
        } else {
            fallback_groups += cold_groups;
            fallback_refs += cold_refs;
        }

        bool cold_tail_done = false;
        if (ok && cold_refs != 0 && use_ane_gpu_tail_gather_scatter) {
            ds4_gpu_tensor *cold_selected_zero = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * sizeof(int32_t));
            ok = cold_selected_zero &&
                 ds4_gpu_tensor_write(cold_selected_zero,
                                      0,
                                      zero_selected,
                                      (uint64_t)n_tokens * sizeof(zero_selected[0])) != 0;
            for (uint32_t ui = 0; ok && ui < n_unique; ui++) {
                const int32_t expert = unique[ui];
                if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
                    ok = false;
                    break;
                }
                if (ane_mask[(uint32_t)expert]) continue;
                const uint32_t begin = (uint32_t)offsets[ui];
                const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
                if (refs == 0) continue;
                if (refs > n_tokens) {
                    ok = false;
                    break;
                }

                if (!commands_open) {
                    ok = ds4_gpu_begin_commands() != 0;
                    commands_open = ok;
                    if (!ok) break;
                }

                ds4_gpu_tensor *tokens_view = ds4_gpu_tensor_view(
                    dedup_tokens_gpu,
                    (uint64_t)begin * sizeof(int32_t),
                    (uint64_t)refs * sizeof(int32_t));
                ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
                    dedup_weights_gpu,
                    (uint64_t)begin * sizeof(float),
                    (uint64_t)refs * sizeof(float));
                ds4_gpu_tensor *selected_zero_view = ds4_gpu_tensor_view(
                    cold_selected_zero,
                    0,
                    (uint64_t)refs * sizeof(int32_t));
                ds4_gpu_tensor *gate_tmp = ds4_gpu_tensor_view(
                    routed_gate_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
                ds4_gpu_tensor *up_tmp = ds4_gpu_tensor_view(
                    routed_up_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
                ds4_gpu_tensor *mid_tmp = ds4_gpu_tensor_view(
                    routed_mid_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
                ds4_gpu_tensor *x_tmp = ds4_gpu_tensor_view(
                    routed_xout_base, 0, (uint64_t)refs * expert_in_dim * sizeof(float));
                ds4_gpu_tensor *out_tmp = ds4_gpu_tensor_view(
                    routed_xout_base,
                    (uint64_t)routed_tmp_rows * out_dim * sizeof(float),
                    (uint64_t)refs * out_dim * sizeof(float));
                ds4_gpu_tensor *gate_model = ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_gate_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
                ds4_gpu_tensor *up_model = ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_up_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
                ds4_gpu_tensor *down_model = ds4_gpu_model_tensor_view(
                    model->map,
                    model->size,
                    layer->ffn_down_exps->abs_offset + (uint64_t)expert * down_expert_bytes,
                    down_expert_bytes);

                bool tail_mid_is_f16 = false;
                ok = tokens_view && weights_view && selected_zero_view &&
                     gate_tmp && up_tmp && mid_tmp && x_tmp && out_tmp &&
                     gate_model && up_model && down_model &&
                     ds4_gpu_gather_rows_f32_tensor(x_tmp,
                                                    g->batch_ffn_norm,
                                                    tokens_view,
                                                    refs,
                                                    DS4_N_EMBD) != 0;
                if (ok) {
                    ok = ds4_gpu_routed_moe_expert_banked_batch_tensor(out_tmp,
                                                                       gate_tmp,
                                                                       up_tmp,
                                                                       mid_tmp,
                                                                       gate_model,
                                                                       up_model,
                                                                       down_model,
                                                                       layer->ffn_gate_exps->type,
                                                                       layer->ffn_down_exps->type,
                                                                       gate_expert_bytes,
                                                                       gate_row_bytes,
                                                                       down_expert_bytes,
                                                                       down_row_bytes,
                                                                       expert_in_dim,
                                                                       expert_mid_dim,
                                                                       out_dim,
                                                                       selected_zero_view,
                                                                       weights_view,
                                                                       DS4_SWIGLU_CLAMP_EXP,
                                                                       x_tmp,
                                                                       refs,
                                                                       &tail_mid_is_f16) != 0;
                }
                if (ok) {
                    ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out,
                                                             out_tmp,
                                                             tokens_view,
                                                             refs,
                                                             out_dim) != 0;
                }
                if (tail_mid_is_f16) g->batch_routed_mid_is_f16 = true;

                ds4_gpu_tensor_free(down_model);
                ds4_gpu_tensor_free(up_model);
                ds4_gpu_tensor_free(gate_model);
                ds4_gpu_tensor_free(out_tmp);
                ds4_gpu_tensor_free(x_tmp);
                ds4_gpu_tensor_free(mid_tmp);
                ds4_gpu_tensor_free(up_tmp);
                ds4_gpu_tensor_free(gate_tmp);
                ds4_gpu_tensor_free(selected_zero_view);
                ds4_gpu_tensor_free(weights_view);
                ds4_gpu_tensor_free(tokens_view);
            }
            if (use_ane_gpu_tail_overlap) {
                if (ok && commands_open) {
                    ok = ds4_gpu_flush_commands() != 0 && ok;
                    commands_open = ok;
                }
                while (ok && ane_queue_n != 0) {
                    DS4_RESIDENT_ANE_FINISH_ONE();
                }
                if (!ok) {
                    while (ane_queue_n != 0) {
                        resident_ane_pending_job p = ane_queue[--ane_queue_n];
                        (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
                            p.job, NULL, NULL);
                    }
                }
            }
            if (commands_open) {
                ok = ds4_gpu_end_commands() != 0 && ok;
                commands_open = false;
            }
            ds4_gpu_tensor_free(cold_selected_zero);
            cold_tail_done = true;
        }

        if (ok && cold_refs != 0 && !cold_tail_done) {
            bool gpu_tail_mid_is_f16 = false;
            /* In overlap mode ANE has not been drained yet, so ane_refs is still 0; the
             * pending-queue count tells us ANE will contribute. Stage the NAX cold output
             * into batch_ffn_out whenever ANE contributes (ANE scatters into batch_routed_out
             * concurrently; the two disjoint results are summed after both finish). */
            const bool ane_contributes =
                use_ane_nax_overlap ? (ane_queue_n != 0) : (ane_refs != 0);
            ds4_gpu_tensor *gpu_tail_out = g->batch_routed_out;
            if (ane_contributes) {
                ok = metal_graph_ensure_batch_ffn_out(g);
                gpu_tail_out = g->batch_ffn_out;
            }
            if (ane_contributes) {
                ds4_gpu_set_ane_skip_mask(ane_mask, DS4_N_EXPERT);
            } else {
                ds4_gpu_clear_ane_skip_mask();
            }
            /* Sequential: ANE was drained + commands ended, so open a fresh buffer.
             * Overlap: the submit loop left a command buffer OPEN (begin_commands would
             * fail), so encode the cold tail straight into it. */
            if (ok && !commands_open) {
                ok = ds4_gpu_begin_commands() != 0;
                commands_open = ok;
            }
            if (ok) {
                ok = metal_graph_routed_moe_batch_tiled(g,
                                                        gpu_tail_out,
                                                        model,
                                                        layer,
                                                        n_tokens,
                                                        gate_expert_bytes,
                                                        gate_row_bytes,
                                                        down_expert_bytes,
                                                        down_row_bytes,
                                                        expert_in_dim,
                                                        expert_mid_dim,
                                                        out_dim,
                                                        g->batch_router_selected,
                                                        g->batch_router_weights,
                                                        g->batch_ffn_norm,
                                                        &gpu_tail_mid_is_f16);
            }
            if (use_ane_nax_overlap) {
                /* Commit the NAX cold tail so the GPU starts executing it NOW, then drain
                 * ANE. The ANE jobs have been running on the ANE engine since the submit
                 * loop; flushing here makes GPU(NAX) and ANE(experts) overlap until we join.
                 * ANE finishes scatter into batch_routed_out (disjoint from batch_ffn_out
                 * and from the [scratch_cap..] half of batch_routed_down NAX leaves alone). */
                const bool _ovl_dbg = getenv("DS4_OVL_DBG") != NULL;
                if (_ovl_dbg) fprintf(stderr, "OVL[L%u]: pre-flush cold tail, ane_q=%u\n", il, ane_queue_n);
                if (ok && commands_open) {
                    ok = ds4_gpu_flush_commands() != 0 && ok;
                }
                if (_ovl_dbg) fprintf(stderr, "OVL[L%u]: flushed(ok=%d), draining ane_q=%u\n", il, ok, ane_queue_n);
                while (ok && ane_queue_n != 0) {
                    DS4_RESIDENT_ANE_FINISH_ONE();
                }
                if (_ovl_dbg) fprintf(stderr, "OVL[L%u]: drained(ok=%d), ending\n", il, ok);
                if (commands_open) {
                    ok = ds4_gpu_end_commands() != 0 && ok;
                    commands_open = false;
                }
                if (_ovl_dbg) fprintf(stderr, "OVL[L%u]: ended(ok=%d)\n", il, ok);
                if (!ok) {
                    while (ane_queue_n != 0) {
                        resident_ane_pending_job p = ane_queue[--ane_queue_n];
                        (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
                            p.job, NULL, NULL);
                    }
                }
            }
            if (ok && ane_contributes) {
                ok = ds4_gpu_add_tensor(g->batch_routed_out,
                                        g->batch_routed_out,
                                        gpu_tail_out,
                                        (uint32_t)((uint64_t)n_tokens * out_dim)) != 0;
            }
            ds4_gpu_clear_ane_skip_mask();
            if (gpu_tail_mid_is_f16) g->batch_routed_mid_is_f16 = true;
        } else if (use_ane_tail_overlap && ane_queue_n != 0) {
            /* Overlap mode with no cold experts (everything routed to ANE): the deferred
             * drain still has to run to finish the ANE jobs and scatter their output. */
            while (ok && ane_queue_n != 0) {
                DS4_RESIDENT_ANE_FINISH_ONE();
            }
            if (commands_open) {
                ok = ds4_gpu_end_commands() != 0 && ok;
                commands_open = false;
            }
            if (!ok) {
                while (ane_queue_n != 0) {
                    resident_ane_pending_job p = ane_queue[--ane_queue_n];
                    (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
                        p.job, NULL, NULL);
                }
            }
        }
        #undef DS4_RESIDENT_ANE_FINISH_ONE

        ds4_gpu_tensor_free(ane_down_i8_bank);
        ds4_gpu_tensor_free(ane_up_i8_bank);
        ds4_gpu_tensor_free(ane_gate_i8_bank);
        free(ane_queue);
        ds4_gpu_tensor_free(dedup_weights_gpu);
        ds4_gpu_tensor_free(dedup_tokens_gpu);
    } else for (uint32_t ui = 0; ok && ui < n_unique; ui++) {
        const int32_t expert = unique[ui];
        const uint32_t begin = (uint32_t)offsets[ui];
        const uint32_t refs = (uint32_t)(offsets[ui + 1] - offsets[ui]);
        if (refs == 0) continue;
        if (refs > n_tokens) {
            ok = false;
            break;
        }

        ok = ds4_gpu_tensor_write(g->batch_router_selected,
                                  0,
                                  ref_tokens + begin,
                                  (uint64_t)refs * sizeof(ref_tokens[0])) != 0 &&
             ds4_gpu_tensor_write(g->batch_router_weights,
                                  0,
                                  ref_weights + begin,
                                  (uint64_t)refs * sizeof(ref_weights[0])) != 0;
        if (!ok) break;

        if (commands_open) {
            ok = ds4_gpu_end_commands() != 0;
            commands_open = false;
            if (!ok) break;
        }
        ok = ds4_gpu_begin_commands() != 0;
        commands_open = ok;
        if (!ok) break;

        ds4_gpu_tensor *tokens_view = ds4_gpu_tensor_view(
            g->batch_router_selected, 0, (uint64_t)refs * sizeof(int32_t));
        ds4_gpu_tensor *selected_zero_view = ds4_gpu_tensor_view(
            g->batch_router_selected,
            (uint64_t)g->prefill_cap * sizeof(int32_t),
            (uint64_t)refs * sizeof(int32_t));
        ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights, 0, (uint64_t)refs * sizeof(float));
        ds4_gpu_tensor *gate_tmp = ds4_gpu_tensor_view(
            routed_gate_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *up_tmp = ds4_gpu_tensor_view(
            routed_up_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *mid_tmp = ds4_gpu_tensor_view(
            routed_mid_base, 0, (uint64_t)refs * expert_mid_dim * sizeof(float));
        ds4_gpu_tensor *x_tmp = ds4_gpu_tensor_view(
            routed_xout_base, 0, (uint64_t)refs * expert_in_dim * sizeof(float));
        ds4_gpu_tensor *out_tmp = ds4_gpu_tensor_view(
            routed_xout_base,
            (uint64_t)routed_tmp_rows * out_dim * sizeof(float),
            (uint64_t)refs * out_dim * sizeof(float));
        ds4_gpu_tensor *gate_model = ds4_gpu_model_tensor_view(
            model->map,
            model->size,
            layer->ffn_gate_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
            gate_expert_bytes);
        ds4_gpu_tensor *up_model = ds4_gpu_model_tensor_view(
            model->map,
            model->size,
            layer->ffn_up_exps->abs_offset + (uint64_t)expert * gate_expert_bytes,
            gate_expert_bytes);
        ds4_gpu_tensor *down_model = ds4_gpu_model_tensor_view(
            model->map,
            model->size,
            layer->ffn_down_exps->abs_offset + (uint64_t)expert * down_expert_bytes,
            down_expert_bytes);

        bool mid_is_f16 = false;
        int used_mpp = 0;
        bool deferred_ane = false;
        const bool can_try_ane =
            use_ane_hybrid &&
            refs >= ane_min_refs_local &&
            (ane_max_refs_local == 0 || refs <= ane_max_refs_local) &&
            (ane_max_groups_local == 0 || ane_groups < ane_max_groups_local);
        if (can_try_ane && tokens_view && weights_view &&
            gate_model && up_model && down_model) {
            while (ok && pending_ane_count >= ane_queue_depth) {
                DS4_RESIDENT_FINISH_ANE_HEAD();
            }
            ds4_gpu_tensor *ane_tokens = NULL;
            ds4_gpu_tensor *ane_weights = NULL;
            ds4_gpu_tensor *ane_x = NULL;
            if (ok) {
                ane_tokens = ds4_gpu_tensor_alloc((uint64_t)refs * sizeof(int32_t));
                ane_weights = ds4_gpu_tensor_alloc((uint64_t)refs * sizeof(float));
                ane_x = ds4_gpu_tensor_alloc((uint64_t)refs * expert_in_dim * sizeof(float));
            }
            const bool ane_inputs_ok =
                ok && ane_tokens && ane_weights && ane_x &&
                ds4_gpu_tensor_write(ane_tokens,
                                     0,
                                     ref_tokens + begin,
                                     (uint64_t)refs * sizeof(ref_tokens[0])) != 0 &&
                ds4_gpu_tensor_write(ane_weights,
                                     0,
                                     ref_weights + begin,
                                     (uint64_t)refs * sizeof(ref_weights[0])) != 0 &&
                ds4_gpu_gather_rows_f32_tensor(ane_x,
                                               g->batch_ffn_norm,
                                               ane_tokens,
                                               refs,
                                               DS4_N_EMBD) != 0;
            ds4_gpu_ane_prefill_job *ane_job = NULL;
            if (ane_inputs_ok) {
                ane_job = ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(
                    gate_model,
                    up_model,
                    down_model,
                    layer->ffn_gate_exps->type,
                    layer->ffn_down_exps->type,
                    gate_expert_bytes,
                    gate_row_bytes,
                    down_expert_bytes,
                    down_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    out_dim,
                    ane_weights,
                    ane_x,
                    refs);
            }
            if (ane_job && pending_ane_count < DS4_RESIDENT_ANE_PENDING_MAX) {
                const uint32_t pi = pending_ane_count++;
                pending_ane_jobs[pi] = ane_job;
                pending_ane_tokens[pi] = ane_tokens;
                pending_ane_x[pi] = ane_x;
                pending_ane_gate[pi] = gate_model;
                pending_ane_up[pi] = up_model;
                pending_ane_down[pi] = down_model;
                pending_ane_refs[pi] = refs;
                pending_ane_experts[pi] = (uint32_t)expert;
                gate_model = NULL;
                up_model = NULL;
                down_model = NULL;
                ane_tokens = NULL;
                ane_x = NULL;
                deferred_ane = true;
                ane_groups++;
                ane_refs += refs;
            } else {
                if (ane_job) {
                    bool ignored_mid_is_f16 = false;
                    (void)ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
                        ane_job,
                        resident_ane_out ? resident_ane_out : out_tmp,
                        &ignored_mid_is_f16);
                }
                ane_start_failures++;
            }
            ds4_gpu_tensor_free(ane_weights);
            ds4_gpu_tensor_free(ane_x);
            ds4_gpu_tensor_free(ane_tokens);
        }
        if (!deferred_ane) {
            ok = tokens_view && selected_zero_view && weights_view &&
                 gate_tmp && up_tmp && mid_tmp && x_tmp && out_tmp &&
                 gate_model && up_model && down_model &&
                 ds4_gpu_gather_rows_f32_tensor(x_tmp,
                                                g->batch_ffn_norm,
                                                tokens_view,
                                                refs,
                                                DS4_N_EMBD) != 0;
        }
        if (ok && !deferred_ane) {
            const bool force_classic_small_gpu =
                use_ane_hybrid && refs < ane_min_refs_local;
            if (!force_classic_small_gpu) {
                used_mpp = flash_moe_run_mpp_int8_safe_tensor(out_tmp,
                                                              gate_tmp,
                                                              up_tmp,
                                                              mid_tmp,
                                                              gate_model,
                                                              up_model,
                                                              down_model,
                                                              layer->ffn_gate_exps->type,
                                                              layer->ffn_down_exps->type,
                                                              gate_expert_bytes,
                                                              gate_row_bytes,
                                                              down_expert_bytes,
                                                              down_row_bytes,
                                                              expert_in_dim,
                                                              expert_mid_dim,
                                                              out_dim,
                                                              selected_zero_view,
                                                              weights_view,
                                                              DS4_SWIGLU_CLAMP_EXP,
                                                              x_tmp,
                                                              refs,
                                                              &mid_is_f16);
            }
            if (!used_mpp) {
                ok = ds4_gpu_routed_moe_expert_banked_batch_tensor(out_tmp,
                                                                   gate_tmp,
                                                                   up_tmp,
                                                                   mid_tmp,
                                                                   gate_model,
                                                                   up_model,
                                                                   down_model,
                                                                   layer->ffn_gate_exps->type,
                                                                   layer->ffn_down_exps->type,
                                                                   gate_expert_bytes,
                                                                   gate_row_bytes,
                                                                   down_expert_bytes,
                                                                   down_row_bytes,
                                                                   expert_in_dim,
                                                                   expert_mid_dim,
                                                                   out_dim,
                                                                   selected_zero_view,
                                                                   weights_view,
                                                                   DS4_SWIGLU_CLAMP_EXP,
                                                                   x_tmp,
                                                                   refs,
                                                                   &mid_is_f16) != 0;
            }
        }
        if (ok && !deferred_ane) {
            ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out,
                                                     out_tmp,
                                                     tokens_view,
                                                     refs,
                                                     out_dim) != 0;
        }
        if (ok && !deferred_ane) {
            if (used_mpp) {
                mpp_groups++;
                mpp_refs += refs;
            } else {
                fallback_groups++;
                fallback_refs += refs;
            }
        }
        if (mid_is_f16) g->batch_routed_mid_is_f16 = true;

        ds4_gpu_tensor_free(down_model);
        ds4_gpu_tensor_free(up_model);
        ds4_gpu_tensor_free(gate_model);
        ds4_gpu_tensor_free(out_tmp);
        ds4_gpu_tensor_free(x_tmp);
        ds4_gpu_tensor_free(mid_tmp);
        ds4_gpu_tensor_free(up_tmp);
        ds4_gpu_tensor_free(gate_tmp);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_zero_view);
        ds4_gpu_tensor_free(tokens_view);
    }
    while (pending_ane_count > 0) {
        DS4_RESIDENT_FINISH_ANE_HEAD();
    }
    if (commands_open) {
        ok = ds4_gpu_end_commands() != 0 && ok;
        commands_open = false;
    }

    if (ok && !backend_diagnostic_logs_suppressed() &&
        (env_flag_enabled("DS4_RESIDENT_MOE_MPP_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_ANE_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_SCHED_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_PROFILE"))) {
        const char *cold_tail_mode = use_ane_hybrid ?
            (use_ane_gpu_tail_overlap ? "gather_scatter_gpu_overlap" :
             use_ane_gpu_tail_gather_scatter ? "gather_scatter_gpu" :
             use_ane_nax ? "masked_nax" : "masked_gpu") :
            "per_expert";
        fprintf(stderr,
                "ds4: resident MPP/NAX prefill layer=%u mpp_groups=%u mpp_refs=%" PRIu64
                " fallback_groups=%u fallback_refs=%" PRIu64
                " ane_groups=%u ane_refs=%" PRIu64
                " ane_min_refs=%u ane_max_refs=%u ane_start_failures=%u ane_scatter_failures=%u"
                " cold_tail=%s\n",
                il,
                mpp_groups,
                mpp_refs,
                fallback_groups,
                fallback_refs,
                ane_groups,
                ane_refs,
                ane_min_refs_local,
                ane_max_refs_local,
                ane_start_failures,
                ane_scatter_failures,
                cold_tail_mode);
    }
    if (ok) {
        ok = ds4_gpu_synchronize() != 0;
    }
    if (ok) {
        ok = ds4_gpu_begin_commands() != 0;
        if (!ok && !backend_diagnostic_logs_suppressed() &&
            env_flag_enabled("DS4_RESIDENT_MOE_MPP_STATS")) {
            fprintf(stderr, "ds4: resident MPP/NAX prefill final begin_commands failed layer=%u\n", il);
        }
    }

    ds4_gpu_tensor_free(resident_ane_out);
#undef DS4_RESIDENT_FINISH_ANE_HEAD
#undef DS4_RESIDENT_FINISH_ANE_AT

    free(zero_selected);
    free(ref_weights);
    free(ref_tokens);
    free(pair_weights);
    free(true_ids);
    return ok;
}
