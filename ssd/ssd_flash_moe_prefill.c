/* =========================================================================
 * ssd_flash_moe_prefill.c - Flash-MoE layer-major prefill planning and execution.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

static uint32_t metal_graph_flash_moe_prefill_chunk_cap(const ds4_gpu_graph *g) {
    if (!g || !g->flash_moe || g->flash_slot_bank == 0) return UINT32_MAX;
    /*
     * Layer-major Flash-MoE prefill streams unique experts through transient
     * scratch banks, so the decode slot-bank size must not shrink prompt
     * chunks. This mirrors the anemll llama.cpp dedup executor: slot-bank
     * controls decode residency, not the prefill dedup window.
     */
    return UINT32_MAX;
}

static uint32_t metal_graph_effective_prefill_cap(const ds4_gpu_graph *g) {
    if (!g || g->prefill_cap == 0) return 0;
    uint32_t cap = g->prefill_cap;
    const char *clamp_src = NULL;
    const uint32_t flash_cap = metal_graph_flash_moe_prefill_chunk_cap(g);
    if (flash_cap < cap) { cap = flash_cap; clamp_src = "flash-slot-bank"; }
    const uint32_t raw_window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t raw_chunk_cap = g->raw_cap > raw_window ? g->raw_cap - raw_window : 1u;
    if (raw_chunk_cap < cap) {
        /* The raw-KV cap is too small to hold the requested chunk, so it gets
         * clamped. With the auto raw_cap this no longer happens, but an explicit
         * DS4_METAL_GRAPH_RAW_CAP that is too small still can — make it loud
         * (once) and name the value needed, so we don't silently step on it. */
        static int warned_raw_cap_clamp = 0;
        if (!warned_raw_cap_clamp) {
            warned_raw_cap_clamp = 1;
            fprintf(stderr,
                "ds4: WARNING: prefill chunk %u clamped to %u by raw-KV cap "
                "(raw_cap=%u, window=%u). Set DS4_METAL_GRAPH_RAW_CAP >= %u "
                "(window + chunk) — or unset it to auto-size — to honor the chunk.\n",
                cap, raw_chunk_cap, g->raw_cap, raw_window, raw_window + cap);
        }
        cap = raw_chunk_cap;
        clamp_src = "raw-KV-cap (ctx-grow)";
    }
    if (cap == 0) cap = 1u;
    /* One-time EFFECTIVE prefill-chunk diagnostic. This is useful while tuning, but
     * it is runtime output and corrupts the interactive ds4-agent prompt area, so only
     * emit it under explicit backend stats/profile knobs. */
    if (backend_stats_logs_enabled()) {
        static int diag_done = 0;
        if (!diag_done) {
            diag_done = 1;
            const char *cenv = getenv("DS4_METAL_PREFILL_CHUNK");
            fprintf(stderr,
                "ds4: prefill chunk cap EFFECTIVE=%u tokens (DS4_METAL_PREFILL_CHUNK=%s)%s%s "
                "(resident-NAX tiles this by scratch_cap; that tile gates the sync bridge)\n",
                cap,
                (cenv && cenv[0]) ? cenv : "unset(auto<=4096)",
                clamp_src ? ", CLAMPED by " : "",
                clamp_src ? clamp_src : "");
        }
    }
    return cap ? cap : 1u;
}

/* The direct grouped path is a prefill-speed experiment.  Keep the legacy
 * expert-ordered executor as the default until its reduction is promoted by
 * greedy-output parity; set this explicitly for speed A/Bs. */
static bool flash_moe_resident_grouped_prefill_enabled(void) {
    const char *value = getenv("DS4_FLASH_MOE_RESIDENT_GROUPED_PREFILL");
    return value && value[0] && atoi(value) != 0;
}

static bool flash_moe_f16_scales_positive_finite(
        const uint16_t *values,
        uint64_t        count) {
    if (!values && count != 0) return false;
    uint64_t i = 0;
#if defined(__ARM_NEON)
    const uint16x8_t sign_mask = vdupq_n_u16(0x8000u);
    const uint16x8_t abs_mask = vdupq_n_u16(0x7fffu);
    const uint16x8_t exp_mask = vdupq_n_u16(0x7c00u);
    const uint16x8_t zero = vdupq_n_u16(0);
    for (; i + 8u <= count; i += 8u) {
        const uint16x8_t v = vld1q_u16(values + i);
        const uint16x8_t sign_bad = vcgtq_u16(vandq_u16(v, sign_mask), zero);
        const uint16x8_t zero_bad = vceqq_u16(vandq_u16(v, abs_mask), zero);
        const uint16x8_t finite_bad =
            vceqq_u16(vandq_u16(v, exp_mask), exp_mask);
        const uint16x8_t bad = vorrq_u16(sign_bad,
                                         vorrq_u16(zero_bad, finite_bad));
        const uint64x2_t bad64 = vreinterpretq_u64_u16(bad);
        if ((vgetq_lane_u64(bad64, 0) | vgetq_lane_u64(bad64, 1)) != 0) {
            return false;
        }
    }
#endif
    for (; i < count; i++) {
        const uint16_t bits = values[i];
        if ((bits & 0x8000u) != 0 ||
            (bits & 0x7fffu) == 0 ||
            (bits & 0x7c00u) == 0x7c00u) {
            return false;
        }
    }
    return true;
}

static bool flash_moe_per_channel_scales_for_expert(
        const ds4_flash_moe_layer_sidecar *flash_layer,
        int32_t                            expert,
        uint32_t                           expert_mid_dim,
        uint32_t                           out_dim,
        const uint8_t                     *resident_record,
        uint16_t                          *streaming_scratch_f16,
        uint64_t                           streaming_scratch_count,
        const uint16_t                   **scales_f16_out,
        uint32_t                          *scale_count_out,
        const char                       **reason_out) {
    if (scales_f16_out) *scales_f16_out = NULL;
    if (scale_count_out) *scale_count_out = 0;
    if (reason_out) *reason_out = "unknown";
    if (!flash_layer || !scales_f16_out || !scale_count_out ||
        expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
        if (reason_out) *reason_out = "bad_args";
        return false;
    }
    if (flash_layer->family_major || flash_layer->expert_stride == 0) {
        if (reason_out) *reason_out = "invalid_expert_stride";
        return false;
    }

    const uint32_t expected_count[DS4_FLASH_FAMILY_COUNT] = {
        [DS4_FLASH_FAMILY_GATE] = expert_mid_dim,
        [DS4_FLASH_FAMILY_UP]   = expert_mid_dim,
        [DS4_FLASH_FAMILY_DOWN] = out_dim,
    };
    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
        const uint64_t expected_bytes =
            (uint64_t)expected_count[fam] * sizeof(uint16_t);
        if (!flash_layer->family_ane_i8_scale_available[fam]) {
            if (reason_out) *reason_out = "missing_metadata";
            return false;
        }
        if (flash_layer->family_ane_i8_scale_count[fam] != expected_count[fam]) {
            if (reason_out) *reason_out = "unexpected_count";
            return false;
        }
        if (flash_layer->family_ane_i8_scale_bytes[fam] != expected_bytes) {
            if (reason_out) *reason_out = "unexpected_bytes";
            return false;
        }
        const uint64_t offset = flash_layer->family_ane_i8_scale_offset[fam];
        const uint64_t bytes = flash_layer->family_ane_i8_scale_bytes[fam];
        if (offset > flash_layer->expert_stride ||
            bytes > flash_layer->expert_stride - offset) {
            if (reason_out) *reason_out = "scale_outside_expert_stride";
            return false;
        }
    }

    const uint64_t gate_offset =
        flash_layer->family_ane_i8_scale_offset[DS4_FLASH_FAMILY_GATE];
    const uint64_t gate_bytes =
        flash_layer->family_ane_i8_scale_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_offset =
        flash_layer->family_ane_i8_scale_offset[DS4_FLASH_FAMILY_UP];
    const uint64_t up_bytes =
        flash_layer->family_ane_i8_scale_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_offset =
        flash_layer->family_ane_i8_scale_offset[DS4_FLASH_FAMILY_DOWN];
    const uint64_t down_bytes =
        flash_layer->family_ane_i8_scale_bytes[DS4_FLASH_FAMILY_DOWN];
    if (gate_offset > UINT64_MAX - gate_bytes ||
        gate_offset + gate_bytes != up_offset ||
        up_offset > UINT64_MAX - up_bytes ||
        up_offset + up_bytes != down_offset ||
        down_offset > UINT64_MAX - down_bytes) {
        if (reason_out) *reason_out = "noncontiguous_scale_regions";
        return false;
    }

    const uint64_t scale_count64 =
        (uint64_t)expert_mid_dim * 2u + out_dim;
    if (scale_count64 == 0 || scale_count64 > UINT32_MAX ||
        scale_count64 > UINT64_MAX / sizeof(uint16_t)) {
        if (reason_out) *reason_out = "scale_count_overflow";
        return false;
    }
    const uint64_t scale_bytes64 = scale_count64 * sizeof(uint16_t);
    const uint64_t down_end = down_offset + down_bytes;
    if (down_end < gate_offset || down_end - gate_offset != scale_bytes64) {
        if (reason_out) *reason_out = "unexpected_packed_scale_bytes";
        return false;
    }
    const uint64_t expert_u64 = (uint64_t)expert;
    if (expert_u64 != 0 &&
        flash_layer->expert_stride > UINT64_MAX / expert_u64) {
        if (reason_out) *reason_out = "expert_offset_overflow";
        return false;
    }
    const uint64_t record_offset = expert_u64 * flash_layer->expert_stride;
    if (record_offset > UINT64_MAX - gate_offset) {
        if (reason_out) *reason_out = "scale_offset_overflow";
        return false;
    }
    const uint64_t scale_offset = record_offset + gate_offset;
    if (scale_offset > UINT64_MAX - scale_bytes64) {
        if (reason_out) *reason_out = "scale_range_overflow";
        return false;
    }
    const uint64_t scale_end = scale_offset + scale_bytes64;
    if (scale_end > flash_layer->file_size ||
        (scale_offset & (sizeof(uint16_t) - 1u)) != 0) {
        if (reason_out) *reason_out = "scale_outside_sidecar_file";
        return false;
    }

    const uint8_t *scale_bytes = NULL;
    if (resident_record) {
        /* A fully preloaded mixed bank contains the complete v2 expert record,
         * including its packed [gate I | up I | down H] F16 tail.  The bank is
         * shared-storage and graph-owned, so this pointer remains valid through
         * ANE or NAX submission.  The backend still copies the 16-KiB vector
         * into its per-dispatch FP16 scale tensor; no scale value is baked into
         * the graph or kernel. */
        if (gate_offset > (uint64_t)SIZE_MAX) {
            if (reason_out) *reason_out = "resident_scale_offset_overflow";
            return false;
        }
        scale_bytes = resident_record + (size_t)gate_offset;
        if (((uintptr_t)scale_bytes & (sizeof(uint16_t) - 1u)) != 0) {
            if (reason_out) *reason_out = "resident_scale_misaligned";
            return false;
        }
    } else if (flash_layer->map && flash_layer->map_size != 0) {
        if (scale_end > flash_layer->map_size ||
            scale_offset > (uint64_t)SIZE_MAX) {
            if (reason_out) *reason_out = "scale_outside_mapped_file";
            return false;
        }
        /* Preserve the direct-mmap fast path: the immutable sidecar mapping is
         * already suitably aligned and remains live for the graph lifetime. */
        scale_bytes = flash_layer->map + (size_t)scale_offset;
    } else {
        /* Streaming mode has already read the complete expert record, but the
         * async slot that owns that record is released as soon as the weights
         * are uploaded.  Do not retain a pointer into that reusable slot.  The
         * packed v2 tail is small, so read only [gate I | up I | down H] into a
         * caller-owned buffer whose lifetime covers both ANE and NAX dispatch. */
        if (!streaming_scratch_f16 ||
            streaming_scratch_count < scale_count64) {
            if (reason_out) *reason_out = "streaming_scale_scratch_too_small";
            return false;
        }
        if (flash_layer->fd < 0 || scale_offset > (uint64_t)INT64_MAX ||
            scale_bytes64 > (uint64_t)SIZE_MAX) {
            if (reason_out) *reason_out = "streaming_scale_pread_unavailable";
            return false;
        }
        if (!flash_moe_pread_full(flash_layer->fd,
                                  scale_offset,
                                  (uint8_t *)streaming_scratch_f16,
                                  scale_bytes64)) {
            if (reason_out) *reason_out = "streaming_scale_pread_failed";
            return false;
        }
        scale_bytes = (const uint8_t *)streaming_scratch_f16;
    }
    if (!flash_moe_f16_scales_positive_finite(
            (const uint16_t *)scale_bytes,
            scale_count64)) {
        if (reason_out) *reason_out = "nonpositive_or_nonfinite_scale";
        return false;
    }

    *scales_f16_out = (const uint16_t *)scale_bytes;
    *scale_count_out = (uint32_t)scale_count64;
    if (reason_out) *reason_out = NULL;
    return true;
}

/* A v2 expert-major sidecar advertises one packed F16 dequant scale vector
 * per expert: [gate I | up I | down H].  Use only the immutable manifest
 * geometry here; the expert accessor above still validates the mapped range
 * and every scale value before a group is submitted to ANE or NAX. */
static bool flash_moe_layer_has_per_channel_scale_contract(
        const ds4_flash_moe_layer_sidecar *flash_layer,
        uint32_t                           expert_mid_dim,
        uint32_t                           out_dim) {
    if (!flash_layer || flash_layer->family_major ||
        flash_layer->expert_stride == 0) {
        return false;
    }
    const uint32_t expected_count[DS4_FLASH_FAMILY_COUNT] = {
        [DS4_FLASH_FAMILY_GATE] = expert_mid_dim,
        [DS4_FLASH_FAMILY_UP]   = expert_mid_dim,
        [DS4_FLASH_FAMILY_DOWN] = out_dim,
    };
    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
        if (!flash_layer->family_ane_i8_scale_available[fam] ||
            flash_layer->family_ane_i8_scale_count[fam] != expected_count[fam] ||
            flash_layer->family_ane_i8_scale_bytes[fam] !=
                (uint64_t)expected_count[fam] * sizeof(uint16_t)) {
            return false;
        }
    }
    return true;
}

/* A fully preloaded identity-mapped mixed bank is a storage contract, not a
 * quantization-mode contract.  Once all 256 records are resident, every
 * per-expert gather/scatter backend (scalar h-i8, NAX-half, half+ALU, and PC)
 * must consume zero-copy slot views instead of rereading the same records into
 * transient banks.  Direct mmap, chunked/per-slot/per-expert layouts, streaming
 * sidecars, and full-GGUF execution remain on their existing paths. */
static bool flash_moe_full_resident_identity_bank_ready(
        ds4_gpu_graph                     *g,
        const ds4_flash_moe_layer_sidecar *flash_layer,
        uint32_t                           il) {
    if (!g || !flash_layer || il >= DS4_N_LAYER ||
        !flash_moe_preload_slot_bank_enabled() ||
        !g->flash_mixed_slot_bank || g->flash_chunked_mixed_bank ||
        g->flash_direct_mmap_bank || g->flash_per_expert_buffers ||
        g->flash_per_slot_buffers || g->flash_slot_bank != DS4_N_EXPERT ||
        !g->flash_gate_bank[il] || !g->flash_up_bank[il] ||
        !g->flash_down_bank[il] ||
        !metal_graph_flash_moe_sidecar_layer_has_records(flash_layer)) {
        return false;
    }
    for (int32_t expert = 0; expert < (int32_t)DS4_N_EXPERT; expert++) {
        int32_t slot = -1;
        if (!metal_graph_flash_moe_find_resident_slot(g, il, expert, &slot) ||
            slot != expert ||
            !metal_graph_flash_moe_mixed_slot_ptr(g, il, slot)) {
            return false;
        }
    }
    return true;
}

static bool flash_moe_full_resident_scale_bank_ready(
        bool                               resident_identity_bank_ready,
        const ds4_flash_moe_layer_sidecar *flash_layer,
        uint32_t                           expert_mid_dim,
        uint32_t                           out_dim) {
    return resident_identity_bank_ready &&
        flash_moe_layer_has_per_channel_scale_contract(flash_layer,
                                                       expert_mid_dim,
                                                       out_dim);
}

static const char *flash_moe_nax_pc_mode_name(void) {
    const char *i8_i8 = getenv("DS4_FLASH_MOE_MPP_I8I8_PREFILL");
    const char *i8_act = getenv("DS4_FLASH_MOE_MPP_INT8_ACT");
    return ((i8_i8 && i8_i8[0] && atoi(i8_i8) != 0) ||
            (i8_act && i8_act[0] && atoi(i8_act) != 0)) ?
        "i8i8-pc" : "h-i8-pc";
}

static const char *flash_moe_resident_nax_mode_name(void) {
    if (!flash_moe_mpp_int8_prefill_requested()) return "gpu/no-mpp";
    if (env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL") ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE")) {
        return flash_moe_nax_pc_mode_name();
    }
    if (env_flag_enabled("DS4_RESIDENT_MOE_NAX_HALF") ||
        ds4_no_int8_paths_enabled()) {
        return env_flag_enabled("DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP") ?
            "nax-half+alu-configured" : "nax-half";
    }
    const char *i8_i8 = getenv("DS4_FLASH_MOE_MPP_I8I8_PREFILL");
    const char *i8_act = getenv("DS4_FLASH_MOE_MPP_INT8_ACT");
    return ((i8_i8 && i8_i8[0] && atoi(i8_i8) != 0) ||
            (i8_act && i8_act[0] && atoi(i8_act) != 0)) ?
        "i8i8" : "h-i8";
}

/* Strict NAX quality validation must exercise every logical expert row, even
 * when a routed group is smaller than the MPP/NAX 64-row M tile.  The graph's
 * streaming scratch tensors are allocated to prefill_cap rows, so zero-pad the
 * disjoint tail in-place and, for GPU-compacted routing weights, copy the real
 * prefix into the full-sized streaming weight scratch.  The caller still
 * scatters only logical_refs rows; padded rows carry zero x and zero route
 * weight and are never exposed as model output. */
static bool flash_moe_prepare_strict_nax_pc_padding(
        ds4_gpu_graph         *g,
        const ds4_gpu_tensor *logical_weights,
        uint32_t              logical_refs,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor **dispatch_weights_out,
        uint32_t             *dispatch_refs_out,
        const char          **reason_out) {
    const uint32_t tile = 64u;
    if (dispatch_weights_out) *dispatch_weights_out = logical_weights;
    if (dispatch_refs_out) *dispatch_refs_out = logical_refs;
    if (reason_out) *reason_out = "unknown";
    if (!g || !logical_weights || !dispatch_weights_out ||
        !dispatch_refs_out || logical_refs == 0) {
        if (reason_out) *reason_out = "bad_args";
        return false;
    }
    if ((logical_refs % tile) == 0) {
        if (reason_out) *reason_out = NULL;
        return true;
    }
    if (logical_refs > UINT32_MAX - (tile - 1u)) {
        if (reason_out) *reason_out = "padded_ref_overflow";
        return false;
    }
    const uint32_t padded_refs =
        (logical_refs + (tile - 1u)) & ~(tile - 1u);
    if (padded_refs > g->prefill_cap ||
        !g->flash_prefill_x || !g->flash_prefill_gate ||
        !g->flash_prefill_up || !g->flash_prefill_mid ||
        !g->flash_prefill_out || !g->flash_prefill_weights ||
        ds4_gpu_tensor_bytes(g->flash_prefill_x) <
            (uint64_t)padded_refs * expert_in_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->flash_prefill_gate) <
            (uint64_t)padded_refs * expert_mid_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->flash_prefill_up) <
            (uint64_t)padded_refs * expert_mid_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->flash_prefill_mid) <
            (uint64_t)padded_refs * expert_mid_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->flash_prefill_out) <
            (uint64_t)padded_refs * out_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->flash_prefill_weights) <
            (uint64_t)padded_refs * sizeof(float)) {
        if (reason_out) *reason_out = "insufficient_streaming_scratch";
        return false;
    }

    const uint32_t tail_refs = padded_refs - logical_refs;
    ds4_gpu_tensor *x_tail = ds4_gpu_tensor_view(
        g->flash_prefill_x,
        (uint64_t)logical_refs * expert_in_dim * sizeof(float),
        (uint64_t)tail_refs * expert_in_dim * sizeof(float));
    ds4_gpu_tensor *weight_tail = ds4_gpu_tensor_view(
        g->flash_prefill_weights,
        (uint64_t)logical_refs * sizeof(float),
        (uint64_t)tail_refs * sizeof(float));
    bool ok = x_tail && weight_tail &&
        ds4_gpu_tensor_fill_f32(x_tail,
                                0.0f,
                                (uint64_t)tail_refs * expert_in_dim) != 0 &&
        ds4_gpu_tensor_fill_f32(weight_tail, 0.0f, tail_refs) != 0;
    if (ok && logical_weights != g->flash_prefill_weights) {
        ok = ds4_gpu_tensor_copy(g->flash_prefill_weights,
                                 0,
                                 logical_weights,
                                 0,
                                 (uint64_t)logical_refs * sizeof(float)) != 0;
    }
    ds4_gpu_tensor_free(weight_tail);
    ds4_gpu_tensor_free(x_tail);
    if (!ok) {
        if (reason_out) *reason_out = "padding_or_weight_copy_failed";
        return false;
    }

    *dispatch_weights_out = g->flash_prefill_weights;
    *dispatch_refs_out = padded_refs;
    if (reason_out) *reason_out = NULL;
    return true;
}

/* IQ2_XXS/Q2_K grouped mul_mm_id defaults to 32 rows.  Mirror an explicit
 * override here so the tiled caller never encodes a large prefix and then
 * discovers that its final tile is too small for the GPU kernel. */
static uint32_t flash_moe_resident_grouped_prefill_min_tokens(void) {
    uint32_t minimum = 32u;
    const char *value = getenv("DS4_METAL_MOE_MM_MIN_REFS");
    if (value && value[0]) {
        char *end = NULL;
        const unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && parsed != 0 && parsed <= UINT32_MAX) {
            minimum = (uint32_t)parsed;
        }
    }
    return minimum;
}

/* Direct-mmap sidecars historically prefer per-expert mul_mv regardless of
 * the total prefill chunk.  Allow an M3U-scoped profile to select grouped ALU
 * only in the short-prefill window where it wins, while hard-limiting this
 * knob below the production ANE boundary.  Absent/invalid env keeps the
 * legacy direct-mmap route. */
static bool flash_moe_direct_mmap_prefers_mul_mv(
        const ds4_gpu_graph *g,
        uint32_t             n_tokens) {
    if (!g || !g->flash_direct_mmap_bank) return false;

    const char *min_value =
        getenv("DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MIN_TOKENS");
    if (!min_value || !min_value[0]) return true;

    char *end = NULL;
    const unsigned long parsed_min = strtoul(min_value, &end, 10);
    if (end == min_value || *end != '\0' ||
        parsed_min == 0ul || parsed_min > 8191ul) {
        return true;
    }

    uint32_t max_tokens = 8191u;
    const char *max_value =
        getenv("DS4_FLASH_MOE_DIRECT_MMAP_GROUPED_MAX_TOKENS");
    if (max_value && max_value[0]) {
        end = NULL;
        const unsigned long parsed_max = strtoul(max_value, &end, 10);
        if (end == max_value || *end != '\0' || parsed_max == 0ul) {
            return true;
        }
        max_tokens = parsed_max < 8191ul ? (uint32_t)parsed_max : 8191u;
    }

    const uint32_t min_tokens = (uint32_t)parsed_min;
    if (max_tokens < min_tokens) return true;

    if (!backend_diagnostic_logs_suppressed()) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr,
                    "ds4: direct-mmap grouped ALU window: %u..%u total prefill tokens "
                    "(>=8192 route unchanged)\n",
                    min_tokens,
                    max_tokens);
        }
    }
    return n_tokens < min_tokens || n_tokens > max_tokens;
}

static bool metal_graph_flash_moe_resident_grouped_prefill_eligible(
        const ds4_gpu_graph       *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    if (!flash_moe_resident_grouped_prefill_enabled() ||
        !g || !g->flash_moe || !layer || il >= DS4_N_LAYER ||
        g->quality ||
        n_tokens < flash_moe_resident_grouped_prefill_min_tokens() ||
        n_tokens > g->prefill_cap ||
        !metal_graph_flash_moe_identity_gpu_selected_active(g, il) ||
        g->flash_slot_bank != DS4_N_EXPERT ||
        !g->batch_router_selected || !g->batch_router_weights ||
        !g->batch_ffn_norm || !g->batch_routed_out ||
        !g->batch_routed_gate || !g->batch_routed_up ||
        !g->batch_routed_mid || !g->batch_routed_down ||
        !g->flash_gate_bank[il] || !g->flash_up_bank[il] ||
        !g->flash_down_bank[il]) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    if (!flash_moe_layer_no_mxfp4_plane_split(flash_layer) ||
        !layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps ||
        (active_expert_used != 1u && active_expert_used != 2u &&
         active_expert_used != 4u && active_expert_used != 5u &&
         active_expert_used != 6u) ||
        flash_layer->family_type[DS4_FLASH_FAMILY_GATE] != layer->ffn_gate_exps->type ||
        flash_layer->family_type[DS4_FLASH_FAMILY_UP] != layer->ffn_up_exps->type ||
        flash_layer->family_type[DS4_FLASH_FAMILY_DOWN] != layer->ffn_down_exps->type) {
        return false;
    }

    /* The legacy sidecar executor owns ANE and MPP split/merge modes.  This
     * grouped path owns the complete output, so preserve those explicit
     * experiments rather than silently dropping their work split. */
    return !env_flag_enabled("DS4_FLASH_MOE_ANE_PREFILL") &&
           !flash_moe_ane_force_all_groups_enabled() &&
           !flash_moe_ane_require_enabled() &&
           !env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL") &&
           !env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE") &&
           !env_flag_enabled("DS4_FLASH_MOE_HYBRID_PREFILL") &&
           !env_flag_enabled("DS4_FLASH_MOE_CONCURRENT_PREFILL") &&
           !env_flag_enabled("DS4_FLASH_MOE_ANE_PIPELINE_PREFILL") &&
           !env_flag_enabled("DS4_FLASH_MOE_MPP_INT8_PREFILL") &&
           !env_flag_enabled("DS4_FLASH_MOE_MPP_I8I8_PREFILL") &&
           !env_flag_enabled("DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL");
}

static bool metal_graph_flash_moe_run_tiny_batch_slotbank(
        ds4_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             n_tokens,
        uint64_t             gate_expert_bytes,
        uint64_t             gate_row_bytes,
        uint64_t             down_expert_bytes,
        uint64_t             down_row_bytes,
        uint32_t             expert_in_dim,
        uint32_t             expert_mid_dim,
        uint32_t             out_dim) {
    if (flash_moe_ane_force_all_groups_enabled() ||
        flash_moe_ane_require_enabled() ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL") ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE") ||
        !g || !g->flash_moe || !layer || il >= DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > 5u ||
        !g->flash_prefill_selected ||
        !g->batch_router_selected ||
        !g->batch_router_weights ||
        !g->batch_ffn_norm ||
        !g->batch_routed_out ||
        !g->batch_routed_gate ||
        !g->batch_routed_up ||
        !g->batch_routed_mid ||
        !g->batch_routed_down ||
        !g->flash_gate_bank[il] ||
        !g->flash_up_bank[il] ||
        !g->flash_down_bank[il]) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    if (!g->flash_mixed_slot_bank ||
        g->flash_chunked_mixed_bank ||
        g->flash_per_slot_buffers ||
        g->flash_per_expert_buffers ||
        !flash_moe_layer_all_mxfp4_plane_split(flash_layer)) {
        return false;
    }

    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint64_t n_pairs = (uint64_t)n_tokens * active_expert_used;
    if (n_pairs == 0 || n_pairs > 5u * DS4_N_EXPERT_ACTIVE_USED) return false;

    int32_t true_ids[5u * DS4_N_EXPERT_ACTIVE_USED];
    int32_t slot_ids[5u * DS4_N_EXPERT_ACTIVE_USED];
    bool protected_experts[DS4_MAX_EXPERT];
    memset(protected_experts, 0, sizeof(protected_experts));
    const bool identity_selected =
        metal_graph_flash_moe_identity_gpu_selected_active(g, il);

    bool ok = ds4_gpu_end_commands() != 0;
    if (!ok) return false;
    ok = ds4_gpu_tensor_read(g->batch_router_selected,
                             0,
                             true_ids,
                             n_pairs * sizeof(true_ids[0])) != 0;
    const uint64_t miss_before = g->flash_misses;
    for (uint64_t pair = 0; ok && pair < n_pairs; pair++) {
        const int32_t true_expert = true_ids[pair];
        if (true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
            ok = false;
            break;
        }
        if (identity_selected) {
            slot_ids[pair] = true_expert;
        } else {
            ok = metal_graph_flash_moe_install(g,
                                               il,
                                               true_expert,
                                               protected_experts,
                                               &slot_ids[pair]);
            if (ok) protected_experts[true_expert] = true;
        }
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(g->flash_prefill_selected,
                                  0,
                                  slot_ids,
                                  n_pairs * sizeof(slot_ids[0])) != 0;
    }
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (!ok) return false;

    const uint64_t gate_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : gate_expert_bytes;
    const uint64_t down_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : down_expert_bytes;
    bool mid_is_f16 = false;
    ok = ds4_gpu_routed_moe_banked_batch_tensor(g->batch_routed_out,
                                                g->batch_routed_gate,
                                                g->batch_routed_up,
                                                g->batch_routed_mid,
                                                g->batch_routed_down,
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
                                                g->flash_prefill_selected,
                                                g->batch_router_weights,
                                                active_expert_used,
                                                DS4_SWIGLU_CLAMP_EXP,
                                                g->batch_ffn_norm,
                                                n_tokens) != 0;
    g->batch_routed_mid_is_f16 = mid_is_f16;
    if (ok) {
        static bool logged = false;
        if (!logged && !backend_diagnostic_logs_suppressed()) {
            logged = true;
            fprintf(stderr,
                    "ds4: speculative sidecar verifier: MXFP4 slot-bank tiny batch path "
                    "engaged (n_tokens<=5, misses=%" PRIu64 ")\n",
                    (uint64_t)(g->flash_misses - miss_before));
        }
    }
    return ok;
}

static bool metal_graph_flash_moe_run_resident_batch_slotbank(
        ds4_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             n_tokens,
        uint64_t             gate_expert_bytes,
        uint64_t             gate_row_bytes,
        uint64_t             down_expert_bytes,
        uint64_t             down_row_bytes,
        uint32_t             expert_in_dim,
        uint32_t             expert_mid_dim,
        uint32_t             out_dim) {
    if (flash_moe_ane_force_all_groups_enabled() ||
        flash_moe_ane_require_enabled() ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL") ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE") ||
        !g || !g->flash_moe || !layer || il >= DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > 5u ||
        !g->batch_router_selected ||
        !g->batch_router_weights ||
        !g->batch_ffn_norm ||
        !g->batch_routed_out ||
        !g->batch_routed_gate ||
        !g->batch_routed_up ||
        !g->batch_routed_mid ||
        !g->batch_routed_down ||
        !g->flash_gate_bank[il] ||
        !g->flash_up_bank[il] ||
        !g->flash_down_bank[il]) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    const bool hybrid_batch_experiment =
        env_flag_enabled("DS4_MTP_HYBRID_BATCH_VERIFY_EXPERIMENT");
    if (!g->flash_mixed_slot_bank ||
        g->flash_chunked_mixed_bank ||
        g->flash_per_slot_buffers ||
        g->flash_per_expert_buffers ||
        g->flash_direct_mmap_bank ||
        g->flash_slot_bank < DS4_N_EXPERT ||
        !env_flag_enabled("DS4_FLASH_MOE_PRELOAD_SLOT_BANK") ||
        (flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(flash_layer) &&
         !hybrid_batch_experiment) ||
        flash_moe_layer_all_mxfp4_plane_split(flash_layer)) {
        return false;
    }

    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint64_t gate_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : gate_expert_bytes;
    const uint64_t down_slot_stride = flash_layer->expert_stride ?
        flash_layer->expert_stride : down_expert_bytes;

    const bool ok =
        ds4_gpu_routed_moe_banked_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
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
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               active_expert_used,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               n_tokens) != 0;
    g->batch_routed_mid_is_f16 = false;
    if (ok) {
        static bool logged = false;
        if (!logged && !backend_diagnostic_logs_suppressed()) {
            logged = true;
            fprintf(stderr,
                    "ds4: speculative sidecar verifier: resident slot-bank tiny batch path "
                    "engaged (n_tokens<=5, slot id == expert id)\n");
        }
    }
    return ok;
}

static bool metal_graph_flash_moe_run_prefill_dedup(
        ds4_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             n_tokens,
        uint64_t             gate_expert_bytes,
        uint64_t             gate_row_bytes,
        uint64_t             down_expert_bytes,
        uint64_t             down_row_bytes,
        uint32_t             expert_in_dim,
        uint32_t             expert_mid_dim,
        uint32_t             out_dim) {
    if (!g || !g->flash_moe || !layer || il >= DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > g->prefill_cap) {
        return false;
    }
    const ds4_flash_moe_layer_sidecar *flash_layer =
        &g->flash_moe->layer[il];

    if (metal_graph_flash_moe_resident_grouped_prefill_eligible(g,
                                                                 layer,
                                                                 il,
                                                                 n_tokens)) {
        bool mid_is_f16 = false;
        const bool ok = metal_graph_flash_moe_resident_grouped_prefill_tiled(
            g,
            layer,
            il,
            n_tokens,
            flash_moe_resident_grouped_prefill_min_tokens(),
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            out_dim,
            &mid_is_f16);
        if (!ok) {
            fprintf(stderr,
                    "ds4: resident grouped Flash-MoE prefill failed at layer %u "
                    "(%u tokens); refusing unsafe partial fallback\n",
                    il,
                    n_tokens);
            return false;
        }
        g->batch_routed_mid_is_f16 = mid_is_f16;
        return true;
    }

    if (!g->flash_prefill_x || !g->flash_prefill_gate ||
        !g->flash_prefill_up || !g->flash_prefill_mid ||
        !g->flash_prefill_out || !g->flash_prefill_tokens ||
        !g->flash_prefill_selected || !g->flash_prefill_weights) {
        return false;
    }

    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint64_t n_pairs = (uint64_t)n_tokens * active_expert_used;
    if (n_pairs > SIZE_MAX / sizeof(int32_t) || n_pairs > UINT32_MAX) return false;
    int32_t *true_ids = xmalloc((size_t)n_pairs * sizeof(true_ids[0]));
    float *pair_weights = xmalloc((size_t)n_pairs * sizeof(pair_weights[0]));
    int32_t *ref_tokens = xmalloc((size_t)n_pairs * sizeof(ref_tokens[0]));
    float *ref_weights = xmalloc((size_t)n_pairs * sizeof(ref_weights[0]));
    int32_t counts[DS4_MAX_EXPERT] = { 0 };
    int32_t cursor[DS4_MAX_EXPERT] = { 0 };
    int32_t expert_to_index[DS4_MAX_EXPERT];
    int32_t unique[DS4_MAX_EXPERT];
    int32_t offsets[DS4_MAX_EXPERT + 1];
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) expert_to_index[i] = -1;

    const bool backend_logs = !backend_diagnostic_logs_suppressed();
    const bool resident_identity_bank_ready =
        flash_moe_full_resident_identity_bank_ready(g, flash_layer, il);
    const bool resident_scale_bank_ready =
        flash_moe_full_resident_scale_bank_ready(resident_identity_bank_ready,
                                                 flash_layer,
                                                 expert_mid_dim,
                                                 out_dim);
    const bool prefill_slot_bank_cache_enabled =
        !g->flash_per_expert_buffers &&
        !g->flash_direct_mmap_bank &&
        (g->flash_slot_bank < DS4_N_EXPERT || resident_identity_bank_ready);
    const bool profile = backend_logs && env_flag_enabled("DS4_FLASH_MOE_PROFILE");
    const bool use_gpu_dedup = getenv("DS4_FLASH_MOE_GPU_DEDUP") == NULL || atoi(getenv("DS4_FLASH_MOE_GPU_DEDUP")) != 0;
    if (resident_identity_bank_ready && backend_logs) {
        static bool logged_resident_identity_prefill = false;
        if (!logged_resident_identity_prefill) {
            fprintf(stderr,
                    "ds4: resident sidecar identity prefill active: "
                    "slots=%u weights=zero-copy-slot-views "
                    "transient_expert_stage=off\n",
                    g->flash_slot_bank);
            logged_resident_identity_prefill = true;
        }
    }

    const double t0 = profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_end_commands() != 0;
    if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL @entry end_commands il=%u n_tokens=%u\n", il, n_tokens);
    const double t_sync = profile ? now_sec() : 0.0;

    uint32_t n_unique = 0;
    bool gpu_compacted = false;
    int32_t staged_prefill_expert[4] = { -1, -1, -1, -1 };
    bool slot_cache_expert[DS4_MAX_EXPERT] = { false };
    const size_t dedup_count_bytes = (size_t)DS4_N_EXPERT * sizeof(uint32_t);

    if (use_gpu_dedup && ok) {
        uint32_t zeros[DS4_MAX_EXPERT] = {0};
        ok = ds4_gpu_tensor_write(g->flash_dedup_offsets, 0, zeros, dedup_count_bytes) != 0;

        if (ok) {
            ok = ds4_gpu_flash_moe_dedup_histogram(g->batch_router_selected,
                                                   g->flash_dedup_offsets,
                                                   DS4_N_EXPERT,
                                                   (uint32_t)n_pairs) != 0;
        }

        uint32_t gpu_counts[DS4_MAX_EXPERT] = {0};
        if (ok) {
            ok = ds4_gpu_tensor_read(g->flash_dedup_offsets, 0, gpu_counts, dedup_count_bytes) != 0;
        }

        uint64_t counted_pairs = 0;
        for (uint32_t e = 0; ok && e < DS4_N_EXPERT; e++) {
            counted_pairs += gpu_counts[e];
            counts[e] = (int32_t)gpu_counts[e];
            if (counts[e] > 0) {
                unique[n_unique++] = (int32_t)e;
            }
        }
        if (ok && counted_pairs != n_pairs) {
            fprintf(stderr,
                    "ds4: Flash-MoE GPU dedup counted %" PRIu64
                    " routed refs, expected %" PRIu64 " in layer %u\n",
                    counted_pairs,
                    n_pairs,
                    il);
            ok = false;
        }
    } else {
        /* Legacy CPU path */
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

        for (uint64_t i = 0; ok && i < n_pairs; i++) {
            const int32_t expert = true_ids[i];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
                fprintf(stderr, "ds4: Flash-MoE prefill selected invalid expert id %d in layer %u\n",
                        expert, il);
                ok = false;
                break;
            }
            if (counts[expert]++ == 0) unique[n_unique++] = expert;
        }
    }
    if (ok) {
        g->flash_prefill_refs += n_pairs;
        g->flash_prefill_unique += n_unique;
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
        flash_moe_log_prefill_hist(il, n_tokens, n_pairs, n_unique, unique, counts);

        /* Cross-layer prefetch gate: this layer's router has now run, so its real
         * routed set (counts[]) is known. Cancel the speculative reads queued for
         * this layer from the previous layer's prediction that turned out wrong,
         * before they steal bandwidth from the experts this layer actually needs.
         * Reads that predicted correctly stay and are consumed by the loop below. */
        if (flash_moe_xlayer_prefetch_enabled(n_tokens) && g->flash_prefill_xreader) {
            ds4_flash_prefill_async_cancel_layer_mispredicted(
                (ds4_flash_prefill_async_reader *)g->flash_prefill_xreader,
                il, counts);
        }

        const int slot_cache_topk =
            prefill_slot_bank_cache_enabled ?
            get_prefill_slot_cache_target(g->flash_slot_bank) : 0;
        for (int rank = 0; rank < slot_cache_topk && rank < (int)n_unique; rank++) {
            int32_t best = -1;
            int32_t best_refs = -1;
            for (uint32_t i = 0; i < n_unique; i++) {
                const int32_t expert = unique[i];
                if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT || slot_cache_expert[expert]) continue;
                if (counts[expert] > best_refs) {
                    best_refs = counts[expert];
                    best = expert;
                }
            }
            if (best < 0) break;
            slot_cache_expert[best] = true;
        }

        if (use_gpu_dedup) {
            uint32_t start_offsets[DS4_MAX_EXPERT] = {0};
            for (uint32_t i = 0; i < n_unique; i++) {
                const int32_t expert = unique[i];
                start_offsets[expert] = (uint32_t)offsets[i];
            }

            ok = ds4_gpu_tensor_write(g->flash_dedup_offsets, 0,
                                      start_offsets, dedup_count_bytes) != 0;
            if (ok) {
                ok = ds4_gpu_flash_moe_dedup_compact(g->batch_router_selected,
                                                     g->batch_router_weights,
                                                     g->flash_dedup_offsets,
                                                     g->flash_dedup_token_list,
                                                     g->flash_dedup_weight_list,
                                                     DS4_N_EXPERT,
                                                     (uint32_t)n_pairs,
                                                     active_expert_used) != 0;
            }
            gpu_compacted = ok;
            if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL dedup_compact il=%u n_pairs=%llu\n", il, (unsigned long long)n_pairs);

        } else {
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
            }
        }
        if (ok) {
            flash_moe_log_prefill_dedup(il, n_tokens, n_pairs, n_unique, gpu_compacted);
        }
    }

    const uint64_t miss_before = g->flash_misses;
    const double t_plan = profile ? now_sec() : 0.0;
    if (ok) {
        ok = ds4_gpu_tensor_fill_f32(g->batch_routed_out,
                                     0.0f,
                                     (uint64_t)n_tokens * DS4_N_EMBD) != 0;
        if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL fill batch_routed_out il=%u n_tokens=%u\n", il, n_tokens);
    }

    const bool plane_split_prefill =
        g && g->flash_moe && il < DS4_N_LAYER &&
        g->flash_moe->layer[il].family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
        g->flash_moe->layer[il].family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
        g->flash_moe->layer[il].family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
    const bool native_plane_prefill =
        plane_split_prefill &&
        ds4_gpu_mxfp4_native_requested() &&
        ds4_gpu_has_native_mxfp4();
    if (plane_split_prefill && !native_plane_prefill) {
        static bool warned_native_plane_prefill = false;
        if (!warned_native_plane_prefill) {
            fprintf(stderr,
                    "ds4: MXFP4_NATIVE plane-split prefill requires native MXFP4; "
                    "legacy block prefill paths are disabled for this sidecar\n");
            warned_native_plane_prefill = true;
        }
        ok = false;
    }

    const bool ane_prefill_requested = env_flag_enabled("DS4_FLASH_MOE_ANE_PREFILL");
    const bool ane_prefill_supported =
        flash_moe_ane_prefill_tensor_types_supported(layer);
    const bool nax_int8_per_channel_require =
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE");
    const bool nax_int8_per_channel_requested =
        nax_int8_per_channel_require ||
        env_flag_enabled("DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL");
    const bool nax_int8_per_channel_profile =
        env_flag_enabled("DS4_FLASH_MOE_NAX_PC_PROFILE");
    const bool try_ane_prefill =
        !nax_int8_per_channel_require &&
        !plane_split_prefill && ane_prefill_requested && ane_prefill_supported;
    if (nax_int8_per_channel_require && ane_prefill_requested) {
        static bool logged_nax_require_suppresses_ane = false;
        if (!logged_nax_require_suppresses_ane) {
            fprintf(stderr,
                    "ds4: NAX INT8 per-channel REQUIRE suppresses routed ANE "
                    "prefill so every routed expert group is verified on NAX\n");
            logged_nax_require_suppresses_ane = true;
        }
    }
    const char *ane_per_channel_env =
        getenv("DS4_FLASH_MOE_ANE_PER_CHANNEL");
    const bool ane_per_channel_explicit = ane_per_channel_env != NULL;
    const bool sidecar_has_per_channel_scales =
        flash_moe_layer_has_per_channel_scale_contract(
            flash_layer,
            expert_mid_dim,
            out_dim);
    const bool ane_per_channel_requested =
        try_ane_prefill &&
        (ane_per_channel_explicit ? atoi(ane_per_channel_env) != 0 :
                                    sidecar_has_per_channel_scales);
    uint16_t *streaming_weight_scales_f16 = NULL;
    uint64_t streaming_weight_scale_capacity = 0;
    if ((ane_per_channel_requested || nax_int8_per_channel_requested) &&
        !resident_scale_bank_ready &&
        (!flash_layer->map || flash_layer->map_size == 0)) {
        const uint64_t expected_scale_count =
            (uint64_t)expert_mid_dim * 2u + out_dim;
        if (expected_scale_count != 0 &&
            expected_scale_count <= UINT32_MAX &&
            expected_scale_count <= SIZE_MAX / sizeof(uint16_t)) {
            streaming_weight_scales_f16 =
                xmalloc((size_t)expected_scale_count * sizeof(uint16_t));
            streaming_weight_scale_capacity = expected_scale_count;
        }
    }
    const bool ane_test_force_all_groups =
        flash_moe_ane_force_all_groups_enabled();
    const uint32_t ane_all_groups_min_tokens =
        flash_moe_ane_all_groups_min_tokens();
    const bool ane_long_force_all_groups =
        ane_all_groups_min_tokens != 0 && n_tokens >= ane_all_groups_min_tokens;
    const bool ane_force_all_groups =
        ane_test_force_all_groups || ane_long_force_all_groups;
    const bool ane_require = flash_moe_ane_require_enabled();
    if (ane_test_force_all_groups || ane_require) {
        static bool logged_ane_test_policy = false;
        if (!logged_ane_test_policy) {
            fprintf(stderr,
                    "ds4: Flash-MoE ANE test policy force_all_groups=%u require=%u "
                    "(normal ANE model/type/backend eligibility remains active)\n",
                    ane_test_force_all_groups ? 1u : 0u,
                    ane_require ? 1u : 0u);
            logged_ane_test_policy = true;
        }
    }
    if (ane_long_force_all_groups) {
        static bool logged_ane_long_policy = false;
        if (!logged_ane_long_policy) {
            fprintf(stderr,
                    "ds4: Flash-MoE ANE long-prefill all-groups active "
                    "n_tokens=%u min_tokens=%u\n",
                    n_tokens,
                    ane_all_groups_min_tokens);
            logged_ane_long_policy = true;
        }
    }
    if (plane_split_prefill && ane_prefill_requested && backend_logs) {
        static bool warned_plane_split_ane_prefill = false;
        if (!warned_plane_split_ane_prefill) {
            fprintf(stderr,
                    "ds4: Flash-MoE ANE prefill disabled for MXFP4_NATIVE "
                    "plane-split sidecar; %s "
                    "(set DS4_MXFP4_NATIVE_DEQUANT_PREFILL_EXPERIMENT=1 to "
                    "try the diagnostic MPP/NAX dequant route; it may change "
                    "routing/quality)\n",
                    ane_require ? "strict ANE REQUIRE will fail" :
                                  "using native MXFP4 MPP prefill");
            warned_plane_split_ane_prefill = true;
        }
    }
    if (ane_prefill_requested && !ane_prefill_supported && backend_logs) {
        static bool warned_unsupported_flash_ane = false;
        if (!warned_unsupported_flash_ane) {
            char reason[256];
            fprintf(stderr,
                    "ds4: Flash-MoE ANE prefill requested but disabled: %s; "
                    "actual Flash-MoE ANE eval calls=0, %s\n",
                    flash_moe_ane_prefill_unsupported_reason(layer,
                                                             reason,
                                                             sizeof(reason)),
                    ane_require ? "strict ANE REQUIRE will fail" :
                                  "using GPU/MPP fallback");
            warned_unsupported_flash_ane = true;
        }
    }
    if (ane_require && !try_ane_prefill) {
        char unsupported_reason[256];
        const char *reason = "routed ANE prefill is unavailable";
        if (!ane_prefill_requested) {
            reason = "DS4_FLASH_MOE_ANE_PREFILL is not enabled";
        } else if (plane_split_prefill) {
            reason = "MXFP4 plane-split routed prefill is not ANE-eligible";
        } else if (!ane_prefill_supported) {
            reason = flash_moe_ane_prefill_unsupported_reason(layer,
                                                               unsupported_reason,
                                                               sizeof(unsupported_reason));
        }
        fprintf(stderr,
                "ds4: ERROR: DS4_FLASH_MOE_ANE_REQUIRE=1 cannot be satisfied "
                "at layer %u: %s\n",
                il,
                reason);
        ok = false;
    } else if (ane_force_all_groups && !try_ane_prefill) {
        static bool warned_force_unavailable = false;
        if (!warned_force_unavailable) {
            fprintf(stderr,
                    "ds4: DS4_FLASH_MOE_ANE_FORCE_ALL_GROUPS ignored because "
                    "routed ANE prefill is not eligible\n");
            warned_force_unavailable = true;
        }
    }
    const bool try_mpp_int8_prefill =
        native_plane_prefill || flash_moe_mpp_int8_prefill_enabled();
    if (nax_int8_per_channel_require && !try_mpp_int8_prefill) {
        fprintf(stderr,
                "ds4: ERROR: required NAX INT8 per-channel prefill is not "
                "available at layer %u (MPP/NAX device gate rejected it)\n",
                il);
        ok = false;
    }
    const bool hybrid_prefill =
        try_ane_prefill && try_mpp_int8_prefill && env_flag_enabled("DS4_FLASH_MOE_HYBRID_PREFILL");
    /* Relaxed for ANE-only async exploration. */
    const bool concurrent_prefill =
        try_ane_prefill &&
        (env_flag_enabled("DS4_FLASH_MOE_CONCURRENT_PREFILL") ||
         env_flag_enabled("DS4_FLASH_MOE_HYBRID_CONCURRENT_PREFILL") ||
         env_flag_enabled("DS4_FLASH_MOE_OVERLAP_PREFILL"));
    const bool ane_pipeline_prefill =
        try_ane_prefill && env_flag_enabled("DS4_FLASH_MOE_ANE_PIPELINE_PREFILL");
    const bool split_prefill = hybrid_prefill || concurrent_prefill;
    const uint32_t hybrid_ane_min_refs = get_prefill_hybrid_ane_min_refs();
    const uint32_t concurrent_min_gpu_groups = get_prefill_concurrent_min_gpu_groups();
    const bool overlap_scheduler =
        concurrent_prefill && ane_pipeline_prefill && gpu_compacted &&
        env_flag_enabled("DS4_FLASH_MOE_OVERLAP_SCHEDULER");
    const bool scheduler_stats =
        backend_logs &&
        (profile || env_flag_enabled("DS4_FLASH_MOE_SCHED_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_SCHED_DEBUG"));
    uint64_t hybrid_ane_refs = 0;
    uint64_t hybrid_gpu_refs = 0;
    uint64_t required_ane_expected_refs =
        (ane_require && ane_force_all_groups) ? n_pairs : 0;
    uint64_t required_ane_completed_refs = 0;
    uint64_t planned_ane_refs = 0;
    uint64_t planned_gpu_refs = 0;
    uint32_t planned_ane_groups = 0;
    uint32_t planned_gpu_groups = 0;
    uint32_t planned_ssd_groups = 0;
    uint32_t planned_tail_gpu_groups = 0;
    uint64_t planned_tail_gpu_refs = 0;
    uint32_t hybrid_ane_groups = 0;
    uint32_t hybrid_gpu_groups = 0;
    uint32_t hybrid_fp32_groups = 0;
    uint32_t resident_mpp_groups = 0;
    uint64_t resident_mpp_refs = 0;
    uint64_t resident_mpp_dispatch_refs = 0;
    uint32_t resident_mpp_padded_groups = 0;
    uint64_t resident_mpp_tail_zero_bytes = 0;
    uint32_t required_ane_expected_groups =
        (ane_require && ane_force_all_groups) ? n_unique : 0;
    uint32_t required_ane_completed_groups = 0;
    uint32_t concurrent_ane_groups = 0;
    uint32_t concurrent_gpu_overlap_groups = 0;
    uint32_t concurrent_finishes = 0;
    uint32_t concurrent_waits = 0;
    uint32_t concurrent_gpu_groups_since_ane = 0;
    double planned_gpu_cost = 0.0;
    double planned_ane_cost = 0.0;
    ds4_flash_prefill_plan_item *plan = NULL;
    uint32_t plan_n = 0;
    if (overlap_scheduler && ok) {
        plan = (ds4_flash_prefill_plan_item *)xcalloc((size_t)n_unique * 2u, sizeof(plan[0]));
        plan_n = build_flash_prefill_overlap_plan(plan,
                                                  n_unique,
                                                  unique,
                                                  offsets,
                                                  slot_cache_expert,
                                                  hybrid_ane_min_refs,
                                                  ane_force_all_groups,
                                                  &planned_gpu_cost,
                                                  &planned_ane_cost,
                                                  &planned_ane_groups,
                                                  &planned_gpu_groups,
                                                  &planned_ane_refs,
                                                  &planned_gpu_refs,
                                                  &planned_ssd_groups,
                                                  &planned_tail_gpu_groups,
                                                  &planned_tail_gpu_refs);
        if (plan_n == 0) {
            free(plan);
            plan = NULL;
        }
    }
    const uint32_t exec_n = plan ? plan_n : n_unique;
    const double t_execute0 = scheduler_stats ? now_sec() : 0.0;
    const bool async_pread_enabled =
        prefill_slot_bank_cache_enabled && !resident_identity_bank_ready &&
        env_flag_enabled("DS4_FLASH_MOE_ASYNC_PREAD");
    const bool async_pread_after_stage =
        async_pread_enabled && env_flag_enabled("DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE");
    /* The reader is normally a per-layer local (created/destroyed each call).
     * With cross-layer prefetch it must outlive the call so the next layer's
     * experts queued at the end of THIS call survive to be consumed next call;
     * use the persistent reader held on the graph. The async_reader macro lets
     * the rest of this function reference whichever one is active unchanged. */
    ds4_flash_prefill_async_reader local_async_reader;
    const bool xlayer_prefetch = flash_moe_xlayer_prefetch_enabled(n_tokens);
    const bool direct_mmap_prefer_mul_mv =
        flash_moe_direct_mmap_prefers_mul_mv(g, n_tokens);
    ds4_flash_prefill_async_reader *p_async_reader = &local_async_reader;
    bool xlayer_reader_ready = false;
    if (xlayer_prefetch) {
        if (!g->flash_prefill_xreader) {
            g->flash_prefill_xreader =
                xmalloc(sizeof(ds4_flash_prefill_async_reader));
            memset(g->flash_prefill_xreader, 0,
                   sizeof(ds4_flash_prefill_async_reader));
        }
        p_async_reader = (ds4_flash_prefill_async_reader *)g->flash_prefill_xreader;
        xlayer_reader_ready = p_async_reader->initialized != 0;
    } else {
        memset(&local_async_reader, 0, sizeof(local_async_reader));
    }
#define async_reader (*p_async_reader)
    bool async_reader_ok = false;
    bool *async_submitted = NULL;
    if (async_pread_enabled && ok && g->flash_moe && g->flash_moe->max_expert_stride != 0) {
        async_reader_ok = xlayer_reader_ready ? true :
            ds4_flash_prefill_async_init(&async_reader, g->flash_moe->max_expert_stride);
        if (async_reader_ok) {
            async_submitted = (bool *)xcalloc((size_t)exec_n ? (size_t)exec_n : 1u,
                                              sizeof(async_submitted[0]));
            if (backend_logs &&
                (env_flag_enabled("DS4_FLASH_MOE_SCHED_DEBUG") ||
                 env_flag_enabled("DS4_FLASH_MOE_ASYNC_PREAD_DEBUG"))) {
                fprintf(stderr,
                        "ds4: Flash-MoE async pread enabled slots=%d prefetch=%d after_stage=%u\n",
                        DS4_FLASH_PREFILL_ASYNC_SLOTS,
                        get_prefill_dedup_prefetch(),
                        async_pread_after_stage ? 1u : 0u);
            }
        }
    }
#define DS4_SUBMIT_ASYNC_PREAD(exec_idx_) do { \
        const uint32_t _exec_idx = (uint32_t)(exec_idx_); \
        if (async_reader_ok && async_submitted && _exec_idx < exec_n && !async_submitted[_exec_idx]) { \
            const uint32_t _ui = plan ? plan[_exec_idx].ui : _exec_idx; \
            if (_ui < n_unique) { \
                const int32_t _expert = unique[_ui]; \
                const bool _slot_resident = prefill_slot_bank_cache_enabled && \
                                            _expert >= 0 && _expert < (int32_t)DS4_N_EXPERT && \
                                            metal_graph_flash_moe_find_resident_slot(g, il, _expert, NULL); \
                if (_slot_resident) { \
                    ds4_flash_prefill_async_cancel_expert(&async_reader, il, _expert); \
                } else { \
                    const uint32_t _refs = plan ? plan[_exec_idx].refs : \
                        (uint32_t)(offsets[_ui + 1] - offsets[_ui]); \
                    const int _bank = (int)(_exec_idx & 3u); \
                    const ds4_flash_moe_layer_sidecar *_side_layer = &g->flash_moe->layer[il]; \
                    const uint64_t _src = (uint64_t)_expert * _side_layer->expert_stride; \
                    ok = ds4_flash_prefill_async_submit(&async_reader, \
                                                        il, \
                                                        _expert, \
                                                        _bank, \
                                                        _refs, \
                                                        _side_layer->fd, \
                                                        _src, \
                                                        _side_layer->expert_stride, \
                                                        _side_layer, \
                                                        false /* on-demand within-layer */); \
                    if (!ok) break; \
                } \
                async_submitted[_exec_idx] = true; \
            } \
        } \
    } while (0)
    if (ok && async_reader_ok) {
        /* Pre-queue the full read-ahead window so all reader threads have work
         * immediately; async_submitted[] dedups against later resubmits. */
        const int initial_prefetch = ds4_flash_prefill_readahead();
        for (int i = 0; i <= initial_prefetch && i < (int)exec_n; i++) {
            DS4_SUBMIT_ASYNC_PREAD((uint32_t)i);
            if (!ok) break;
        }
    } else if (ok) {
        int prefetch =
            prefill_slot_bank_cache_enabled ? get_prefill_dedup_prefetch() : 0;
        for (int i = 0; i < prefetch && i < (int)exec_n; i++) {
            const uint32_t future = plan ? plan[i].ui : (uint32_t)i;
            if (future >= n_unique) {
                if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL prefetch future=%u >= n_unique=%u il=%u plan=%d exec_n=%u\n", future, n_unique, il, plan!=NULL, exec_n);
                ok = false;
                break;
            }
            const int future_expert = unique[future];
            if (prefill_slot_bank_cache_enabled &&
                future_expert >= 0 && future_expert < (int)DS4_N_EXPERT &&
                metal_graph_flash_moe_find_resident_slot(g, il, future_expert, NULL)) {
                continue;
            }
            const int bank = i & 3;
            const uint32_t future_refs =
                plan ? plan[i].refs : (uint32_t)(offsets[future + 1] - offsets[future]);
            ok = metal_graph_flash_moe_stage_prefill_expert(g, il, future_expert, bank, future_refs);
            if (!ok) { if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL stage_prefill_expert il=%u expert=%d bank=%d refs=%u\n", il, future_expert, bank, future_refs); break; }
            staged_prefill_expert[bank] = future_expert;
        }
    }
    ds4_gpu_ane_prefill_job *active_ane_job = NULL;
    ds4_gpu_tensor *active_ane_tokens = NULL;
    uint32_t active_ane_refs = 0;
    enum { DS4_ANE_READY_QUEUE_MAX = 8 };
    struct ds4_ane_ready_slot {
        ds4_gpu_ane_prefill_job *job;
        ds4_gpu_tensor *tokens;
        uint32_t refs;
    };
    struct ds4_ane_ready_slot ready_ane[DS4_ANE_READY_QUEUE_MAX];
    memset(ready_ane, 0, sizeof(ready_ane));
    uint32_t ready_ane_count = 0;
    const uint32_t ready_ane_cap =
        ane_pipeline_prefill ? get_prefill_ane_output_queue_depth() : 1u;

    bool commands_open = false;
/* finish_tensor now encodes the writeback as a GPU blit on the live CB; no
 * CPU/GPU race on flash_prefill_out. We only need flush (commit async) so any
 * previously encoded scatter_add gets committed ahead of our new work — the
 * GPU queue serializes the rest. */
#define DS4_FINISH_ANE_SLOT(job_var, tokens_var, refs_var) do { \
        if (job_var) { \
            concurrent_waits++; \
            if (ok && commands_open) { \
                ok = ds4_gpu_flush_commands() != 0; \
                /* flush leaves a fresh CB open; commands_open stays true */ \
            } \
            bool pending_mid_is_f16 = false; \
            int pending_ok = ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor( \
                job_var, g->flash_prefill_out, &pending_mid_is_f16); \
            job_var = NULL; \
            if (ok && pending_ok) { \
                concurrent_finishes++; \
                if (ane_require) { \
                    required_ane_completed_groups++; \
                    required_ane_completed_refs += refs_var; \
                } \
                if (!commands_open) { \
                    ok = ds4_gpu_begin_commands() != 0; \
                    commands_open = ok; \
                } \
                if (ok) { \
                    double scatter_t0 = now_sec(); \
                    ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out, \
                                                             g->flash_prefill_out, \
                                                             tokens_var, \
                                                             refs_var, \
                                                             DS4_N_EMBD) != 0; \
                    double scatter_t1 = now_sec(); \
                    g->flash_prefill_scatter_ms += (scatter_t1 - scatter_t0) * 1000.0; \
                    ds4_prefill_trace_interval_ms("scatter_encode", il, -1, scatter_t0 * 1000.0, scatter_t1 * 1000.0); \
                } \
                if (pending_mid_is_f16) g->batch_routed_mid_is_f16 = true; \
            } else { \
                ok = false; \
            } \
            ds4_gpu_tensor_free(tokens_var); \
            tokens_var = NULL; \
            refs_var = 0; \
            concurrent_gpu_groups_since_ane = 0; \
        } \
    } while (0)
#define DS4_FINISH_READY_ANE_HEAD() do { \
        if (ready_ane_count != 0) { \
            ds4_gpu_ane_prefill_job *_ready_job = ready_ane[0].job; \
            ds4_gpu_tensor *_ready_tokens = ready_ane[0].tokens; \
            uint32_t _ready_refs = ready_ane[0].refs; \
            for (uint32_t _ri = 1; _ri < ready_ane_count; _ri++) { \
                ready_ane[_ri - 1u] = ready_ane[_ri]; \
            } \
            ready_ane_count--; \
            ready_ane[ready_ane_count].job = NULL; \
            ready_ane[ready_ane_count].tokens = NULL; \
            ready_ane[ready_ane_count].refs = 0; \
            DS4_FINISH_ANE_SLOT(_ready_job, _ready_tokens, _ready_refs); \
        } \
    } while (0)
#define DS4_QUEUE_READY_ANE(job_var, tokens_var, refs_var) do { \
        if (job_var) { \
            while (ok && ready_ane_count >= ready_ane_cap) { \
                DS4_FINISH_READY_ANE_HEAD(); \
            } \
            if (ok) { \
                ready_ane[ready_ane_count].job = job_var; \
                ready_ane[ready_ane_count].tokens = tokens_var; \
                ready_ane[ready_ane_count].refs = refs_var; \
                ready_ane_count++; \
                job_var = NULL; \
                tokens_var = NULL; \
                refs_var = 0; \
            } \
        } \
    } while (0)
/* Dual-ANE-cluster optimization (M3 Ultra) — scheduler half.
 *
 * The other half (per-job dequant slot pool, two ANE worker threads, two
 * context handles) lives in ds4_metal.m.  Together they let the M3 Ultra's
 * two ANEx16 clusters run ANE prefill jobs concurrently instead of
 * serialising on a single cluster.  See moe-batch-bench/DUAL_ANE_CLUSTER_
 * OPTIMIZATION.md for the full design.
 *
 * DS4_FLASH_MOE_ANE_MULTI_ACTIVE=1 lets us SKIP the synchronous wait_predict
 * here, transferring the active job to the predicted-handle slot *without*
 * joining its ANE thread.  The thread keeps running concurrently with the
 * next active job's eval on the other cluster; the eventual finish_tensor
 * (called from DS4_FINISH_ANE_SLOT) does the join.  Net effect: eval(N)
 * overlaps with eval(N+1), driven by the two ANE clusters in parallel
 * instead of being serialised at the scheduler.
 *
 * Without this flag, the slot pool in ds4_metal.m is correct but useless —
 * only one ANE eval would actually be in flight at a time.
 *
 * When DS4_FLASH_MOE_ANE_MULTI_ACTIVE is off (or env not set), behaviour
 * matches the previous synchronous wait_predict for safety. */
#define DS4_WAIT_ACTIVE_ANE_PREDICT(job_out, tokens_out, refs_out) do { \
        if (active_ane_job) { \
            concurrent_waits++; \
            if (ok && commands_open) { \
                ok = ds4_gpu_flush_commands() != 0; \
            } \
            int multi_active = env_flag_enabled("DS4_FLASH_MOE_ANE_MULTI_ACTIVE"); \
            if (multi_active) { \
                /* Defer the join: transfer ownership without waiting. */ \
                job_out = active_ane_job; \
                tokens_out = active_ane_tokens; \
                refs_out = active_ane_refs; \
                active_ane_job = NULL; \
                active_ane_tokens = NULL; \
                active_ane_refs = 0; \
            } else { \
                int predict_ok = ok ? ds4_gpu_routed_moe_expert_banked_batch_ane_wait_predict_tensor(active_ane_job) : 0; \
                if (!predict_ok) { \
                    ok = false; \
                } else { \
                    job_out = active_ane_job; \
                    tokens_out = active_ane_tokens; \
                    refs_out = active_ane_refs; \
                    active_ane_job = NULL; \
                    active_ane_tokens = NULL; \
                    active_ane_refs = 0; \
                } \
            } \
        } \
    } while (0)
    /* ANE expert eval begins here: pause speculative cross-layer prefetch so it
     * stops competing with eval for memory bandwidth. On-demand within-layer
     * reads are promoted past the pause as the loop consumes them. */
    if (xlayer_prefetch && async_reader_ok && flash_moe_xlayer_attn_only()) {
        ds4_flash_prefill_async_set_paused(p_async_reader, 1);
    }
    for (uint32_t exec_i = 0; ok && exec_i < exec_n; exec_i++) {
        if (async_reader_ok && !async_pread_after_stage) {
            const uint32_t submit_i = exec_i + (uint32_t)ds4_flash_prefill_readahead();
            DS4_SUBMIT_ASYNC_PREAD(submit_i);
            if (!ok) break;
        }
        const uint32_t ui = plan ? plan[exec_i].ui : exec_i;
        if (ui >= n_unique) {
            if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL exec ui=%u >= n_unique=%u il=%u exec_i=%u plan=%d exec_n=%u\n", ui, n_unique, il, exec_i, plan!=NULL, exec_n);
            ok = false;
            break;
        }
        const ds4_prefill_lane planned_lane =
            plan ? plan[exec_i].lane : DS4_PREFILL_LANE_ANE;
        const int32_t expert = unique[ui];
        const uint32_t begin = plan ? plan[exec_i].begin : (uint32_t)offsets[ui];
        const uint32_t refs = plan ? plan[exec_i].refs : (uint32_t)(offsets[ui + 1] - offsets[ui]);
        if (refs == 0) continue;

        const int bank_set = exec_i & 3;   /* rotate across all prefill banks */

        ds4_gpu_tensor *tokens_for_refs = g->flash_prefill_tokens;
        ds4_gpu_tensor *weights_for_refs = g->flash_prefill_weights;
        ds4_gpu_tensor *token_view = NULL;
        ds4_gpu_tensor *weight_view = NULL;
        ds4_gpu_tensor *slot_gate_view = NULL;
        ds4_gpu_tensor *slot_up_view = NULL;
        ds4_gpu_tensor *slot_down_view = NULL;
        int32_t resident_slot = -1;
        const uint8_t *resident_scale_record = NULL;

        ds4_gpu_tensor *gate_b = NULL;
        ds4_gpu_tensor *up_b = NULL;
        ds4_gpu_tensor *down_b = NULL;
        switch (bank_set) {
        case 1:
            gate_b = g->flash_prefill_gate_bank2;
            up_b = g->flash_prefill_up_bank2;
            down_b = g->flash_prefill_down_bank2;
            break;
        case 2:
            gate_b = g->flash_prefill_gate_bank3;
            up_b = g->flash_prefill_up_bank3;
            down_b = g->flash_prefill_down_bank3;
            break;
        case 3:
            gate_b = g->flash_prefill_gate_bank4;
            up_b = g->flash_prefill_up_bank4;
            down_b = g->flash_prefill_down_bank4;
            break;
        default:
            gate_b = g->flash_prefill_gate_bank;
            up_b = g->flash_prefill_up_bank;
            down_b = g->flash_prefill_down_bank;
            break;
        }

        const bool valid_expert = expert >= 0 && expert < (int32_t)DS4_N_EXPERT;
        const bool use_slot_cache =
            prefill_slot_bank_cache_enabled &&
            valid_expert &&
            metal_graph_flash_moe_find_resident_slot(g,
                                                     il,
                                                     expert,
                                                     &resident_slot);
        if (resident_identity_bank_ready && !use_slot_cache) {
            fprintf(stderr,
                    "ds4: ERROR: resident sidecar identity slot "
                    "lost layer=%u expert=%d; refusing transient-stage fallback\n",
                    il,
                    expert);
            ok = false;
            break;
        }
        const bool wants_slot_prefetch =
            prefill_slot_bank_cache_enabled &&
            valid_expert && slot_cache_expert[expert] && !use_slot_cache;
        uint8_t *slot_prefetch_src = NULL;
        bool slot_prefetch_src_owned = false;
        if (use_slot_cache && async_reader_ok) {
            ds4_flash_prefill_async_cancel_expert(&async_reader, il, expert);
        }

        // Stage the expert weights first. Transient banks can be filled while
        // the previous GPU work drains; slot-cache installs synchronize first
        // because they may evict a bank used by the previous command buffer.
        if (use_slot_cache && commands_open) {
            ok = ds4_gpu_end_commands() != 0;
            commands_open = false;
            if (!ok) { if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL pre-stage end_commands il=%u exec_i=%u expert=%d refs=%u\n", il, exec_i, expert, refs); break; }
        }

        if (gpu_compacted) {
            token_view = ds4_gpu_tensor_view(g->flash_dedup_token_list,
                                             (uint64_t)begin * sizeof(int32_t),
                                             (uint64_t)refs * sizeof(int32_t));
            weight_view = ds4_gpu_tensor_view(g->flash_dedup_weight_list,
                                              (uint64_t)begin * sizeof(float),
                                              (uint64_t)refs * sizeof(float));
            tokens_for_refs = token_view;
            weights_for_refs = weight_view;

            if (use_slot_cache) {
                const uint64_t slot_hits_before = g->flash_hits;
                const uint64_t slot_misses_before = g->flash_misses;
                ok = token_view && weight_view &&
                     metal_graph_flash_moe_install(g,
                                                   il,
                                                   expert,
                                                   NULL,
                                                   &resident_slot);
                if (ok) {
                    metal_graph_flash_moe_record_prefill_slot_cache(g,
                                                                     refs,
                                                                     slot_hits_before,
                                                                     slot_misses_before);
                    slot_gate_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_GATE, resident_slot);
                    slot_up_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_UP, resident_slot);
                    slot_down_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_DOWN, resident_slot);
                    ok = slot_gate_view && slot_up_view && slot_down_view;
                    gate_b = slot_gate_view;
                    up_b = slot_up_view;
                    down_b = slot_down_view;
                }
            } else {
                const bool already_staged = staged_prefill_expert[bank_set] == expert;
                if (token_view && weight_view && already_staged) {
                    ok = true;
                } else if (token_view && weight_view && async_reader_ok) {
                    ok = metal_graph_flash_moe_stage_prefill_expert_async(g,
                                                                          &async_reader,
                                                                          il,
                                                                          expert,
                                                                          bank_set,
                                                                          refs,
                                                                          wants_slot_prefetch ?
                                                                          &slot_prefetch_src : NULL);
                    if (!ok) {
                        /* No async prefetch slot for this expert. On a resume over a
                         * decode-populated bank, SUBMIT cancels the pread for any
                         * expert already resident in a slot -- but a non-slot-cached
                         * expert still takes this staging path, so find_locked()
                         * returns -1. The async prefetch is only an optimization;
                         * fall back to a synchronous stage so a canceled/missing
                         * prefetch can never abort the (resume-)prefill. */
                        slot_prefetch_src = NULL;
                        slot_prefetch_src_owned = false;
                        ok = metal_graph_flash_moe_stage_prefill_expert(g, il, expert, bank_set, refs);
                        if (ok && wants_slot_prefetch) slot_prefetch_src = g->flash_install_buf;
                    } else {
                        slot_prefetch_src_owned = slot_prefetch_src != NULL;
                    }
                } else {
                    ok = token_view && weight_view &&
                         metal_graph_flash_moe_stage_prefill_expert(g, il, expert, bank_set, refs);
                    if (ok && wants_slot_prefetch) slot_prefetch_src = g->flash_install_buf;
                }
                if (ok && !already_staged) staged_prefill_expert[bank_set] = expert;
            }
        } else {
            if (use_slot_cache) {
                const uint64_t slot_hits_before = g->flash_hits;
                const uint64_t slot_misses_before = g->flash_misses;
                ok = metal_graph_flash_moe_install(g,
                                                   il,
                                                   expert,
                                                   NULL,
                                                   &resident_slot);
                if (ok) {
                    metal_graph_flash_moe_record_prefill_slot_cache(g,
                                                                     refs,
                                                                     slot_hits_before,
                                                                     slot_misses_before);
                    slot_gate_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_GATE, resident_slot);
                    slot_up_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_UP, resident_slot);
                    slot_down_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_DOWN, resident_slot);
                    ok = slot_gate_view && slot_up_view && slot_down_view;
                    gate_b = slot_gate_view;
                    up_b = slot_up_view;
                    down_b = slot_down_view;
                }
            } else {
                const bool already_staged = staged_prefill_expert[bank_set] == expert;
                if (already_staged) {
                    ok = true;
                } else if (async_reader_ok) {
                    ok = metal_graph_flash_moe_stage_prefill_expert_async(g,
                                                                          &async_reader,
                                                                          il,
                                                                          expert,
                                                                          bank_set,
                                                                          refs,
                                                                          wants_slot_prefetch ?
                                                                          &slot_prefetch_src : NULL);
                    if (!ok) {
                        /* No async prefetch slot for this expert. On a resume over a
                         * decode-populated bank, SUBMIT cancels the pread for any
                         * expert already resident in a slot -- but a non-slot-cached
                         * expert still takes this staging path, so find_locked()
                         * returns -1. The async prefetch is only an optimization;
                         * fall back to a synchronous stage so a canceled/missing
                         * prefetch can never abort the (resume-)prefill. */
                        slot_prefetch_src = NULL;
                        slot_prefetch_src_owned = false;
                        ok = metal_graph_flash_moe_stage_prefill_expert(g, il, expert, bank_set, refs);
                        if (ok && wants_slot_prefetch) slot_prefetch_src = g->flash_install_buf;
                    } else {
                        slot_prefetch_src_owned = slot_prefetch_src != NULL;
                    }
                } else {
                    ok = metal_graph_flash_moe_stage_prefill_expert(g, il, expert, bank_set, refs);
                    if (ok && wants_slot_prefetch) slot_prefetch_src = g->flash_install_buf;
                }
                if (ok && !already_staged) staged_prefill_expert[bank_set] = expert;
            }
            /* CPU dedup reuses one shared token/weight scratch for every
             * expert.  The previous expert's SwiGLU and scatter still read
             * those buffers until its command batch completes, so drain it
             * before overwriting the next expert's rows.  GPU dedup uses
             * immutable per-expert views and keeps the overlap fast path. */
            if (ok && commands_open) {
                ok = ds4_gpu_end_commands() != 0;
                commands_open = false;
            }
            ok = ok &&
                 ds4_gpu_tensor_write(g->flash_prefill_tokens, 0,
                                      ref_tokens + begin,
                                      (uint64_t)refs * sizeof(ref_tokens[0])) != 0 &&
                 ds4_gpu_tensor_write(g->flash_prefill_weights, 0,
                                      ref_weights + begin,
                                      (uint64_t)refs * sizeof(ref_weights[0])) != 0;
        }
        if (!ok) {
            if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr,
                "ds4: [resume-dbg] flashmoe FAIL stage/install il=%u exec_i=%u expert=%d refs=%u use_slot_cache=%d gpu_compacted=%d tok_v=%d w_v=%d sg_v=%d su_v=%d sd_v=%d\n",
                il, exec_i, expert, refs, use_slot_cache, gpu_compacted,
                token_view!=NULL, weight_view!=NULL, slot_gate_view!=NULL, slot_up_view!=NULL, slot_down_view!=NULL);
            if (slot_prefetch_src_owned) free(slot_prefetch_src);
            ds4_gpu_tensor_free(slot_down_view);
            ds4_gpu_tensor_free(slot_up_view);
            ds4_gpu_tensor_free(slot_gate_view);
            ds4_gpu_tensor_free(weight_view);
            ds4_gpu_tensor_free(token_view);
            break;
        }
        if (resident_scale_bank_ready && use_slot_cache &&
            (ane_per_channel_requested || nax_int8_per_channel_requested)) {
            resident_scale_record =
                metal_graph_flash_moe_mixed_slot_ptr(g, il, resident_slot);
            if (!resident_scale_record) {
                fprintf(stderr,
                        "ds4: ERROR: resident per-channel expert record "
                        "unavailable layer=%u expert=%d slot=%d\n",
                        il,
                        expert,
                        resident_slot);
                ok = false;
                ds4_gpu_tensor_free(slot_down_view);
                ds4_gpu_tensor_free(slot_up_view);
                ds4_gpu_tensor_free(slot_gate_view);
                ds4_gpu_tensor_free(weight_view);
                ds4_gpu_tensor_free(token_view);
                break;
            }
        }
        if (ok && wants_slot_prefetch && slot_prefetch_src) {
            if (commands_open) {
                ok = ds4_gpu_end_commands() != 0;
                commands_open = false;
            }
            if (ok) {
                ok = metal_graph_flash_moe_prefetch_slot_from_buf(g,
                                                                  il,
                                                                  expert,
                                                                  refs,
                                                                  slot_prefetch_src);
                if (ok && async_reader_ok) {
                    ds4_flash_prefill_async_cancel_expert(&async_reader, il, expert);
                }
            }
            if (slot_prefetch_src_owned) {
                free(slot_prefetch_src);
                slot_prefetch_src = NULL;
                slot_prefetch_src_owned = false;
            }
            if (!ok) {
                ds4_gpu_tensor_free(slot_down_view);
                ds4_gpu_tensor_free(slot_up_view);
                ds4_gpu_tensor_free(slot_gate_view);
                ds4_gpu_tensor_free(weight_view);
                ds4_gpu_tensor_free(token_view);
                break;
            }
        }
        if (async_reader_ok && async_pread_after_stage) {
            const int prefetch = ds4_flash_prefill_readahead();
            if (prefetch > 0) {
                for (int ahead = 1; ahead <= prefetch; ahead++) {
                    DS4_SUBMIT_ASYNC_PREAD(exec_i + (uint32_t)ahead);
                    if (!ok) break;
                }
                if (!ok) {
                    ds4_gpu_tensor_free(slot_down_view);
                    ds4_gpu_tensor_free(slot_up_view);
                    ds4_gpu_tensor_free(slot_gate_view);
                    ds4_gpu_tensor_free(weight_view);
                    ds4_gpu_tensor_free(token_view);
                    break;
                }
            } else {
                DS4_SUBMIT_ASYNC_PREAD(exec_i + 1u);
                if (!ok) {
                    ds4_gpu_tensor_free(slot_down_view);
                    ds4_gpu_tensor_free(slot_up_view);
                    ds4_gpu_tensor_free(slot_gate_view);
                    ds4_gpu_tensor_free(weight_view);
                    ds4_gpu_tensor_free(token_view);
                    break;
                }
            }
        }

        metal_graph_flash_moe_tag_layer_family_storage(
                g,
                il,
                gate_b,
                up_b,
                down_b);

        // Synchronize previous work, then run this expert's compute on its bank set.
        if (commands_open) {
            ok = ds4_gpu_end_commands() != 0;
            commands_open = false;
            if (!ok) { if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL pre-compute end_commands il=%u exec_i=%u expert=%d refs=%u\n", il, exec_i, expert, refs); break; }
        }

        ok = ds4_gpu_begin_commands() != 0;
        if (!ok) { if (getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL pre-compute begin_commands il=%u exec_i=%u expert=%d refs=%u\n", il, exec_i, expert, refs); break; }
        commands_open = true;

        bool mid_is_f16 = false;
        bool force_group_can_chunk = false;
        if (ane_force_all_groups) {
            prefill_ane_chunk_refs_for_group(refs, NULL, &force_group_can_chunk);
        }
        const bool try_ane_for_group = try_ane_prefill &&
            (ane_force_all_groups ?
                force_group_can_chunk :
                ((!split_prefill || refs >= hybrid_ane_min_refs) &&
                 (!overlap_scheduler || planned_lane == DS4_PREFILL_LANE_ANE)));
        const bool ane_weight_scales_needed =
            ane_per_channel_requested && try_ane_for_group;
        const bool nax_weight_scales_needed =
            nax_int8_per_channel_requested && try_mpp_int8_prefill;
        const uint16_t *expert_weight_scales_f16 = NULL;
        uint32_t expert_weight_scale_count = 0;
        if (ane_weight_scales_needed || nax_weight_scales_needed) {
            const char *scale_reason = NULL;
            const bool scale_ok =
                flash_moe_per_channel_scales_for_expert(
                    flash_layer,
                    expert,
                    expert_mid_dim,
                    out_dim,
                    resident_scale_record,
                    streaming_weight_scales_f16,
                    streaming_weight_scale_capacity,
                    &expert_weight_scales_f16,
                    &expert_weight_scale_count,
                    &scale_reason);
            if (!scale_ok) {
                static bool warned_ane_per_channel_fallback = false;
                if (ane_weight_scales_needed &&
                    (ane_require || !warned_ane_per_channel_fallback)) {
                    fprintf(stderr,
                            "ds4: %s: ANE per-channel scales unavailable "
                            "layer=%u expert=%d reason=%s%s\n",
                            ane_require ? "ERROR" : "warning",
                            il,
                            expert,
                            scale_reason ? scale_reason : "unknown",
                            ane_require ? "" : "; using scalar ANE weight scale");
                    warned_ane_per_channel_fallback = true;
                }
                static bool warned_nax_per_channel_fallback = false;
                if (nax_weight_scales_needed &&
                    (nax_int8_per_channel_require ||
                     !warned_nax_per_channel_fallback)) {
                    fprintf(stderr,
                            "ds4: %s: NAX INT8 per-channel scales unavailable "
                            "layer=%u expert=%d reason=%s%s\n",
                            nax_int8_per_channel_require ? "ERROR" : "warning",
                            il,
                            expert,
                            scale_reason ? scale_reason : "unknown",
                            nax_int8_per_channel_require ? "" :
                                "; using scalar NAX INT8 weight scale");
                    warned_nax_per_channel_fallback = true;
                }
                if ((ane_weight_scales_needed && ane_require) ||
                    (nax_weight_scales_needed &&
                     nax_int8_per_channel_require)) {
                    ok = false;
                }
            } else {
                static bool logged_per_channel_scales = false;
                if (ane_weight_scales_needed &&
                    !logged_per_channel_scales && backend_logs) {
                    fprintf(stderr,
                            "ds4: Flash-MoE ANE per-channel scales active: "
                            "packed gate/up/down count=%u (%u/%u/%u)\n",
                            expert_weight_scale_count,
                            expert_mid_dim,
                            expert_mid_dim,
                            out_dim);
                    logged_per_channel_scales = true;
                }
                static bool logged_resident_ane_scale_source = false;
                if (ane_weight_scales_needed && resident_scale_record &&
                    !logged_resident_ane_scale_source && backend_logs) {
                    fprintf(stderr,
                            "ds4: resident sidecar ANE per-channel scale source: "
                            "weights=preloaded-identity-bank "
                            "scales=F16-resident-record scale_count=%u "
                            "transient_expert_stage=off scale_upload=per-expert\n",
                            expert_weight_scale_count);
                    logged_resident_ane_scale_source = true;
                }
            }
        }
        const uint16_t *ane_weight_scales_f16 =
            ane_weight_scales_needed ? expert_weight_scales_f16 : NULL;
        const uint32_t ane_weight_scale_count =
            ane_weight_scales_f16 ? expert_weight_scale_count : 0;
        const uint16_t *nax_weight_scales_f16 =
            nax_weight_scales_needed ? expert_weight_scales_f16 : NULL;
        const uint32_t nax_weight_scale_count =
            nax_weight_scales_f16 ? expert_weight_scale_count : 0;
        const bool ane_required_for_group =
            ane_require && (ane_force_all_groups || try_ane_for_group);
        if (ane_required_for_group && !ane_force_all_groups) {
            required_ane_expected_groups++;
            required_ane_expected_refs += refs;
        }
        if (ane_required_for_group && !try_ane_for_group) {
            fprintf(stderr,
                    "ds4: ERROR: required ANE prefill group is not executable "
                    "layer=%u expert=%d refs=%u chunkable=%u lane=%s\n",
                    il,
                    expert,
                    refs,
                    force_group_can_chunk ? 1u : 0u,
                    planned_lane == DS4_PREFILL_LANE_ANE ? "ane" : "gpu");
            ok = false;
        }
        const bool try_mpp_for_group = try_mpp_int8_prefill &&
            (!overlap_scheduler || planned_lane == DS4_PREFILL_LANE_GPU);
        const bool try_pipelined_ane_for_group =
            ane_pipeline_prefill && gpu_compacted && try_ane_for_group;
        const bool try_concurrent_ane_for_group =
            !try_pipelined_ane_for_group &&
            concurrent_prefill && gpu_compacted && try_ane_for_group;
        if (ok) {
            ok = ds4_gpu_gather_rows_f32_tensor(g->flash_prefill_x,
                                                g->batch_ffn_norm,
                                                tokens_for_refs,
                                                refs,
                                                DS4_N_EMBD) != 0;
        }
        if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL gather_rows il=%u exec_i=%u expert=%d refs=%u tok_for_refs=%d\n", il, exec_i, expert, refs, tokens_for_refs!=NULL);
        bool deferred_ane = false;
        if (ok) {
            bool ane_ok = false;
            bool attempted_ane = false;
            bool used_ane = false;
            bool used_mpp = false;
            if (try_pipelined_ane_for_group) {
                ds4_gpu_ane_prefill_job *predicted_ane_job = NULL;
                ds4_gpu_tensor *predicted_ane_tokens = NULL;
                uint32_t predicted_ane_refs = 0;
                DS4_WAIT_ACTIVE_ANE_PREDICT(predicted_ane_job,
                                            predicted_ane_tokens,
                                            predicted_ane_refs);
                if (ok && !commands_open) {
                    ok = ds4_gpu_begin_commands() != 0;
                    commands_open = ok;
                }
                if (ok) {
                    attempted_ane = true;
                    active_ane_job =
                        ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(gate_b,
                                                                                up_b,
                                                                                down_b,
                                                                                layer->ffn_gate_exps->type,
                                                                                layer->ffn_down_exps->type,
                                                                                gate_expert_bytes,
                                                                                gate_row_bytes,
                                                                                down_expert_bytes,
                                                                                down_row_bytes,
                                                                                expert_in_dim,
                                                                                expert_mid_dim,
                                                                                out_dim,
                                                                                weights_for_refs,
                                                                                g->flash_prefill_x,
                                                                                refs,
                                                                                ane_weight_scales_f16,
                                                                                ane_weight_scale_count);
                    if (active_ane_job) {
                        active_ane_tokens = tokens_for_refs;
                        token_view = NULL;
                        active_ane_refs = refs;
                        deferred_ane = true;
                        ane_ok = true;
                        used_ane = true;
                        concurrent_ane_groups++;
                    }
                }
                if (predicted_ane_job) {
                    DS4_QUEUE_READY_ANE(predicted_ane_job,
                                        predicted_ane_tokens,
                                        predicted_ane_refs);
                }
                if (predicted_ane_job) {
                    ds4_gpu_ane_prefill_job *cleanup_job = predicted_ane_job;
                    ds4_gpu_tensor *cleanup_tokens = predicted_ane_tokens;
                    uint32_t cleanup_refs = predicted_ane_refs;
                    DS4_FINISH_ANE_SLOT(cleanup_job, cleanup_tokens, cleanup_refs);
                    predicted_ane_job = NULL;
                    predicted_ane_tokens = NULL;
                    predicted_ane_refs = 0;
                }
            } else if (try_concurrent_ane_for_group) {
                if (active_ane_job &&
                    (ane_force_all_groups ||
                     concurrent_gpu_groups_since_ane >= concurrent_min_gpu_groups)) {
                    DS4_FINISH_ANE_SLOT(active_ane_job, active_ane_tokens, active_ane_refs);
                }
                if (ok && !active_ane_job) {
                    attempted_ane = true;
                    active_ane_job =
                        ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(gate_b,
                                                                                up_b,
                                                                                down_b,
                                                                                layer->ffn_gate_exps->type,
                                                                                layer->ffn_down_exps->type,
                                                                                gate_expert_bytes,
                                                                                gate_row_bytes,
                                                                                down_expert_bytes,
                                                                                down_row_bytes,
                                                                                expert_in_dim,
                                                                                expert_mid_dim,
                                                                                out_dim,
                                                                                weights_for_refs,
                                                                                g->flash_prefill_x,
                                                                                refs,
                                                                                ane_weight_scales_f16,
                                                                                ane_weight_scale_count);
                    if (active_ane_job) {
                        active_ane_tokens = tokens_for_refs;
                        token_view = NULL;
                        active_ane_refs = refs;
                        deferred_ane = true;
                        ane_ok = true;
                        used_ane = true;
                        concurrent_ane_groups++;
                    }
                }
            }
            const bool allow_sync_ane =
                try_ane_for_group && (!concurrent_prefill || !gpu_compacted);
            if (!deferred_ane && allow_sync_ane) {
                const bool ane_debug = env_flag_enabled("DS4_FLASH_MOE_ANE_DEBUG");
                attempted_ane = true;
                if (ane_debug) {
                    fprintf(stderr,
                            "ds4: ANE prefill try layer=%u unique_idx=%u/%u expert=%d refs=%u bank=%d\n",
                            il, ui + 1u, n_unique, expert, refs, bank_set);
                }
                ane_ok = ds4_gpu_routed_moe_expert_banked_batch_ane_tensor(g->flash_prefill_out,
                                                                           g->flash_prefill_gate,
                                                                           g->flash_prefill_up,
                                                                           g->flash_prefill_mid,
                                                                           gate_b,
                                                                           up_b,
                                                                           down_b,
                                                                           layer->ffn_gate_exps->type,
                                                                           layer->ffn_down_exps->type,
                                                                           gate_expert_bytes,
                                                                           gate_row_bytes,
                                                                           down_expert_bytes,
                                                                           down_row_bytes,
                                                                           expert_in_dim,
                                                                           expert_mid_dim,
                                                                           out_dim,
                                                                           g->flash_prefill_selected,
                                                                           weights_for_refs,
                                                                           DS4_SWIGLU_CLAMP_EXP,
                                                                           g->flash_prefill_x,
                                                                           refs,
                                                                           ane_weight_scales_f16,
                                                                           ane_weight_scale_count,
                                                                           &mid_is_f16) != 0;
                used_ane = ane_ok;
                if (ane_ok && ane_required_for_group) {
                    required_ane_completed_groups++;
                    required_ane_completed_refs += refs;
                }
                if (ane_debug) {
                    fprintf(stderr,
                            "ds4: ANE prefill %s layer=%u expert=%d refs=%u\n",
                            ane_ok ? "ok" : "failed", il, expert, refs);
                }
            }
            if (!deferred_ane && !ane_ok && ane_required_for_group) {
                fprintf(stderr,
                        "ds4: ERROR: required ANE prefill dispatch failed; "
                        "GPU/MPP fallback disabled layer=%u expert=%d refs=%u attempted=%u\n",
                        il,
                        expert,
                        refs,
                        attempted_ane ? 1u : 0u);
                ok = false;
            }
            if (ok && !deferred_ane && !ane_ok &&
                (try_mpp_for_group || try_mpp_int8_prefill)) {
                const ds4_gpu_tensor *nax_dispatch_weights = weights_for_refs;
                uint32_t nax_dispatch_refs = refs;
                if (nax_int8_per_channel_require) {
                    const char *padding_reason = NULL;
                    if (!flash_moe_prepare_strict_nax_pc_padding(
                            g,
                            weights_for_refs,
                            refs,
                            expert_in_dim,
                            expert_mid_dim,
                            out_dim,
                            &nax_dispatch_weights,
                            &nax_dispatch_refs,
                            &padding_reason)) {
                        fprintf(stderr,
                                "ds4: ERROR: strict NAX INT8 per-channel "
                                "padding failed layer=%u expert=%d refs=%u "
                                "reason=%s\n",
                                il,
                                expert,
                                refs,
                                padding_reason ? padding_reason : "unknown");
                        ok = false;
                    } else if (nax_dispatch_refs != refs) {
                        static bool logged_nax_pc_padding = false;
                        if (!logged_nax_pc_padding) {
                            fprintf(stderr,
                                    "ds4: strict NAX INT8 per-channel M-tile "
                                    "padding active: logical_refs=%u "
                                    "dispatch_refs=%u tile=64\n",
                                    refs,
                                    nax_dispatch_refs);
                            logged_nax_pc_padding = true;
                        }
                    }
                }
                if (ok) {
                ane_ok = flash_moe_run_mpp_int8_safe_tensor(g->flash_prefill_out,
                                                            g->flash_prefill_gate,
                                                            g->flash_prefill_up,
                                                            g->flash_prefill_mid,
                                                            gate_b,
                                                            up_b,
                                                            down_b,
                                                            layer->ffn_gate_exps->type,
                                                            layer->ffn_down_exps->type,
                                                            gate_expert_bytes,
                                                            gate_row_bytes,
                                                            down_expert_bytes,
                                                            down_row_bytes,
                                                            expert_in_dim,
                                                            expert_mid_dim,
                                                            out_dim,
                                                            g->flash_prefill_selected,
                                                            nax_dispatch_weights,
                                                            DS4_SWIGLU_CLAMP_EXP,
                                                            g->flash_prefill_x,
                                                            nax_dispatch_refs,
                                                            nax_weight_scales_f16,
                                                            nax_weight_scale_count,
                                                            &mid_is_f16) != 0;
                }
                used_mpp = ane_ok;
                if (used_mpp && resident_identity_bank_ready) {
                    resident_mpp_groups++;
                    resident_mpp_refs += refs;
                    if (resident_scale_bank_ready &&
                        nax_int8_per_channel_requested) {
                        resident_mpp_dispatch_refs += nax_dispatch_refs;
                        if (nax_dispatch_refs > refs) {
                            const uint64_t tail_refs =
                                (uint64_t)nax_dispatch_refs - refs;
                            resident_mpp_padded_groups++;
                            resident_mpp_tail_zero_bytes +=
                                tail_refs *
                                ((uint64_t)expert_in_dim * sizeof(float) +
                                 sizeof(float));
                        }
                    }
                }
                if (used_mpp && resident_scale_bank_ready &&
                    resident_scale_record && nax_weight_scales_f16) {
                    static bool logged_resident_nax_pc = false;
                    if (!logged_resident_nax_pc) {
                        fprintf(stderr,
                                "ds4: resident sidecar NAX INT8 per-channel active: "
                                "mode=%s weights=preloaded-identity-bank "
                                "scales=F16-resident-record scale_count=%u "
                                "transient_expert_stage=off "
                                "scale_upload=per-expert\n",
                                flash_moe_nax_pc_mode_name(),
                                nax_weight_scale_count);
                        logged_resident_nax_pc = true;
                    }
                }
                if (!ane_ok && nax_int8_per_channel_require) {
                    fprintf(stderr,
                            "ds4: ERROR: required NAX INT8 per-channel dispatch "
                            "failed; classic GPU fallback disabled "
                            "layer=%u expert=%d refs=%u\n",
                            il,
                            expert,
                            refs);
                    ok = false;
                }
            }
            if (ok && !deferred_ane && !ane_ok) {
                if (attempted_ane && env_flag_enabled("DS4_FLASH_MOE_ANE_DEBUG")) {
                    fprintf(stderr, "ds4: ANE prefill falling back to fp32 GPU layer=%u expert=%d refs=%u\n",
                            il, expert, refs);
                }
                mid_is_f16 = false;
                ok = ds4_gpu_routed_moe_expert_banked_batch_tensor_ex(g->flash_prefill_out,
                                                                      g->flash_prefill_gate,
                                                                      g->flash_prefill_up,
                                                                      g->flash_prefill_mid,
                                                                      gate_b,
                                                                      up_b,
                                                                      down_b,
                                                                      layer->ffn_gate_exps->type,
                                                                      layer->ffn_down_exps->type,
                                                                      gate_expert_bytes,
                                                                      gate_row_bytes,
                                                                      down_expert_bytes,
                                                                      down_row_bytes,
                                                                      expert_in_dim,
                                                                      expert_mid_dim,
                                                                      out_dim,
                                                                      g->flash_prefill_selected,
                                                                      weights_for_refs,
                                                                      DS4_SWIGLU_CLAMP_EXP,
                                                                      g->flash_prefill_x,
                                                                      refs,
                                                                      &mid_is_f16,
                                                                      direct_mmap_prefer_mul_mv) != 0;
                if (ok && hybrid_prefill) hybrid_fp32_groups++;
            }
            if ((hybrid_prefill || concurrent_prefill || ane_pipeline_prefill) && ane_ok) {
                if (used_ane) {
                    hybrid_ane_groups++;
                    hybrid_ane_refs += refs;
                } else if (used_mpp) {
                    hybrid_gpu_groups++;
                    hybrid_gpu_refs += refs;
                }
            }
        }
        if (!deferred_ane) {
            if (ok) {
                const double scatter_t0 = now_sec();
                ok = ds4_gpu_scatter_add_rows_f32_tensor(g->batch_routed_out,
                                                         g->flash_prefill_out,
                                                         tokens_for_refs,
                                                         refs,
                                                         DS4_N_EMBD) != 0;
                if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL scatter_add il=%u exec_i=%u expert=%d refs=%u used_mpp_or_ane\n", il, exec_i, expert, refs);
                const double scatter_t1 = now_sec();
                g->flash_prefill_scatter_ms += (scatter_t1 - scatter_t0) * 1000.0;
                ds4_prefill_trace_interval_ms("scatter_encode",
                                              il,
                                              expert,
                                              scatter_t0 * 1000.0,
                                              scatter_t1 * 1000.0);
            }
            if (active_ane_job) {
                concurrent_gpu_groups_since_ane++;
                concurrent_gpu_overlap_groups++;
            }
        }
        ds4_gpu_tensor_free(slot_down_view);
        ds4_gpu_tensor_free(slot_up_view);
        ds4_gpu_tensor_free(slot_gate_view);
        ds4_gpu_tensor_free(weight_view);
        ds4_gpu_tensor_free(token_view);
        if (mid_is_f16) g->batch_routed_mid_is_f16 = true;

        /* Prefetch stage the expert N ahead (#2) so sidecar load overlaps with current matmul */
        if (!async_reader_ok) {
            int prefetch =
                prefill_slot_bank_cache_enabled ? get_prefill_dedup_prefetch() : 0;
            if (exec_i + (uint32_t)prefetch < exec_n) {
                uint32_t future_exec = exec_i + (uint32_t)prefetch;
                uint32_t future = plan ? plan[future_exec].ui : future_exec;
                int future_expert = unique[future];
                if (prefill_slot_bank_cache_enabled &&
                    future_expert >= 0 && future_expert < (int)DS4_N_EXPERT &&
                    metal_graph_flash_moe_find_resident_slot(g, il, future_expert, NULL)) {
                    continue;
                }
                int future_bank = future_exec % 4;
                uint32_t future_refs =
                    plan ? plan[future_exec].refs : (uint32_t)(offsets[future + 1] - offsets[future]);
                if (staged_prefill_expert[future_bank] != future_expert &&
                    metal_graph_flash_moe_stage_prefill_expert(g, il, future_expert, future_bank, future_refs)) {
                    staged_prefill_expert[future_bank] = future_expert;
                }
            }
        }
    }

    /* --- Cross-layer prefetch (DS4_FLASH_MOE_XLAYER_PREFETCH) ---
     * This layer's experts are all consumed now (reader slots freed). Queue the
     * next layer's hottest experts, predicted from THIS layer's routed counts
     * (~92% expert-set overlap on DSv4). The persistent reader's threads stream
     * them from SSD during the next layer's attention/dense/router compute, so
     * its MoE finds them already read instead of stalling the SSD at the layer
     * boundary (the ~15s cross_layer_post gap). cancel_all first reclaims any
     * mispredicted reads queued for THIS layer that it didn't end up using. */
    if (xlayer_prefetch && async_reader_ok && ok && g->flash_moe &&
        (uint32_t)(il + 1u) < DS4_N_LAYER) {
        ds4_flash_prefill_async_cancel_all(p_async_reader);
        const uint32_t next_il = il + 1u;
        const ds4_flash_moe_layer_sidecar *next_layer = &g->flash_moe->layer[next_il];
        if (next_layer->fd >= 0 && next_layer->expert_stride != 0 &&
            next_layer->expert_stride <= p_async_reader->buf_bytes) {
            const int xk = flash_moe_xlayer_topk((int)(g->flash_slot_bank / 2u));
            bool chosen[DS4_MAX_EXPERT];
            memset(chosen, 0, sizeof(chosen));
            uint32_t xqueued = 0;
            for (int rank = 0; rank < xk && rank < (int)n_unique; rank++) {
                int32_t best = -1;
                int32_t best_refs = -1;
                for (uint32_t i = 0; i < n_unique; i++) {
                    const int32_t e = unique[i];
                    if (e < 0 || e >= (int32_t)DS4_N_EXPERT || chosen[e]) continue;
                    if (counts[e] > best_refs) { best_refs = counts[e]; best = e; }
                }
                if (best < 0) break;
                chosen[best] = true;
                /* Skip experts already cached in the next layer's slot bank. */
                if (prefill_slot_bank_cache_enabled &&
                    metal_graph_flash_moe_find_resident_slot(g, next_il, best, NULL)) {
                    continue;
                }
                const uint64_t src = (uint64_t)best * next_layer->expert_stride;
                if (!ds4_flash_prefill_async_submit(p_async_reader, next_il, best,
                                                    0, (uint32_t)best_refs,
                                                    next_layer->fd, src,
                                                    next_layer->expert_stride,
                                                    next_layer,
                                                    true /* speculative cross-layer */)) {
                    break;  /* pool full or submit error: stop queuing */
                }
                xqueued++;
            }
            if (scheduler_stats) {
                g->flash_prefill_xlayer_queued += xqueued;
                g->flash_prefill_xlayer_calls++;
            }
        }
    }

    if (ane_pipeline_prefill && active_ane_job) {
        ds4_gpu_ane_prefill_job *predicted_ane_job = NULL;
        ds4_gpu_tensor *predicted_ane_tokens = NULL;
        uint32_t predicted_ane_refs = 0;
        DS4_WAIT_ACTIVE_ANE_PREDICT(predicted_ane_job,
                                    predicted_ane_tokens,
                                    predicted_ane_refs);
        if (predicted_ane_job) {
            DS4_QUEUE_READY_ANE(predicted_ane_job,
                                predicted_ane_tokens,
                                predicted_ane_refs);
        }
        if (predicted_ane_job) {
            ds4_gpu_ane_prefill_job *cleanup_job = predicted_ane_job;
            ds4_gpu_tensor *cleanup_tokens = predicted_ane_tokens;
            uint32_t cleanup_refs = predicted_ane_refs;
            DS4_FINISH_ANE_SLOT(cleanup_job, cleanup_tokens, cleanup_refs);
        }
    }
    if (active_ane_job) {
        DS4_FINISH_ANE_SLOT(active_ane_job, active_ane_tokens, active_ane_refs);
    }
    while (ready_ane_count != 0) {
        DS4_FINISH_READY_ANE_HEAD();
    }
    if (ane_require && try_ane_prefill) {
        const bool require_verified =
            ok &&
            required_ane_expected_groups != 0 &&
            required_ane_expected_refs != 0 &&
            required_ane_completed_groups == required_ane_expected_groups &&
            required_ane_completed_refs == required_ane_expected_refs;
        fprintf(stderr,
                "ds4: %sANE REQUIRE layer=%u expected_groups=%u completed_groups=%u "
                "expected_refs=%" PRIu64 " completed_refs=%" PRIu64 " status=%s\n",
                require_verified ? "" : "ERROR: ",
                il,
                required_ane_expected_groups,
                required_ane_completed_groups,
                required_ane_expected_refs,
                required_ane_completed_refs,
                require_verified ? "verified" : "failed");
        if (!require_verified) ok = false;
    }
#undef DS4_WAIT_ACTIVE_ANE_PREDICT
#undef DS4_QUEUE_READY_ANE
#undef DS4_FINISH_READY_ANE_HEAD
#undef DS4_FINISH_ANE_SLOT
    if (!commands_open && ok) {
        ok = ds4_gpu_begin_commands() != 0;
        commands_open = ok;
        if (!ok && getenv("DS4_DEBUG_RESUME")) fprintf(stderr, "ds4: [resume-dbg] flashmoe FAIL post-loop begin_commands il=%u\n", il);
    }

    const double t_done = (profile || scheduler_stats) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: Flash-MoE prefill layer=%u tokens=%u unique=%u refs=%" PRIu64
                " sync=%.3f ms plan=%.3f ms stage/compute=%.3f ms misses=%u\n",
                il,
                n_tokens,
                n_unique,
                n_pairs,
                (t_sync - t0) * 1000.0,
                (t_plan - t_sync) * 1000.0,
                (t_done - t_plan) * 1000.0,
                (uint32_t)(g->flash_misses - miss_before));
    }
    if (ok && resident_identity_bank_ready && backend_logs &&
        (profile || env_flag_enabled("DS4_FLASH_MOE_SCHED_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_KERNEL_LOG") ||
         env_flag_enabled("DS4_RESIDENT_MOE_KERNEL_LOG"))) {
        fprintf(stderr,
                "ds4: resident sidecar prefill layer=%u mode=%s "
                "resident_groups=%u resident_refs=%" PRIu64 " "
                "mpp_groups=%u mpp_logical_refs=%" PRIu64 " "
                "slot_installs=%" PRIu64 " "
                "transient_stage_calls=0 transient_stage_bytes=0\n",
                il,
                flash_moe_resident_nax_mode_name(),
                n_unique,
                n_pairs,
                resident_mpp_groups,
                resident_mpp_refs,
                g->flash_misses - miss_before);
    }
    if (nax_int8_per_channel_profile &&
        nax_int8_per_channel_requested && resident_identity_bank_ready) {
        const double dispatch_util = resident_mpp_dispatch_refs ?
            100.0 * (double)resident_mpp_refs /
                (double)resident_mpp_dispatch_refs : 0.0;
        fprintf(stderr,
                "ds4: NAX PC resident profile layer=%u logical_refs=%" PRIu64
                " dispatch_refs=%" PRIu64 " dispatch_util=%.2f%% "
                "mpp_groups=%u padded_groups=%u tail_zero_bytes=%" PRIu64
                "\n",
                il,
                resident_mpp_refs,
                resident_mpp_dispatch_refs,
                dispatch_util,
                resident_mpp_groups,
                resident_mpp_padded_groups,
                resident_mpp_tail_zero_bytes);
    }
    if (backend_logs &&
        (hybrid_prefill || concurrent_prefill || ane_pipeline_prefill) &&
        (profile || env_flag_enabled("DS4_FLASH_MOE_HYBRID_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_CONCURRENT_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_ANE_PIPELINE_STATS"))) {
        fprintf(stderr,
                "ds4: Flash-MoE hybrid prefill layer=%u ane_groups=%u ane_refs=%" PRIu64
                " gpu_i8_groups=%u gpu_i8_refs=%" PRIu64 " fp32_groups=%u ane_min_refs=%u"
                " concurrent=%u ane_pipeline=%u output_queue=%u async_ane=%u gpu_overlap=%u finishes=%u waits=%u"
                " force_all=%u require=%u required_groups=%u required_refs=%" PRIu64 "\n",
                il,
                hybrid_ane_groups,
                hybrid_ane_refs,
                hybrid_gpu_groups,
                hybrid_gpu_refs,
                hybrid_fp32_groups,
                hybrid_ane_min_refs,
                concurrent_prefill ? 1u : 0u,
                ane_pipeline_prefill ? 1u : 0u,
                ready_ane_cap,
                concurrent_ane_groups,
                concurrent_gpu_overlap_groups,
                concurrent_finishes,
                concurrent_waits,
                ane_force_all_groups ? 1u : 0u,
                ane_require ? 1u : 0u,
                required_ane_completed_groups,
                required_ane_completed_refs);
    }
    if (scheduler_stats && overlap_scheduler) {
        const double est_makespan =
            planned_gpu_cost > planned_ane_cost ? planned_gpu_cost : planned_ane_cost;
        const double est_idle =
            est_makespan > 0.0 ? fabs(planned_gpu_cost - planned_ane_cost) / est_makespan : 0.0;
        fprintf(stderr,
                "ds4: Flash-MoE overlap plan layer=%u exec_groups=%u"
                " planned_ane_groups=%u planned_ane_refs=%" PRIu64
                " planned_gpu_groups=%u planned_gpu_refs=%" PRIu64
                " tail_gpu_groups=%u tail_gpu_refs=%" PRIu64
                " planned_ssd_groups=%u est_gpu=%.1f est_ane=%.1f est_idle=%.2f%%"
                " wall=%.3f ms actual_ane_groups=%u actual_ane_refs=%" PRIu64
                " actual_gpu_groups=%u actual_gpu_refs=%" PRIu64
                " finishes=%u waits=%u overlap_gpu_groups=%u\n",
                il,
                exec_n,
                planned_ane_groups,
                planned_ane_refs,
                planned_gpu_groups,
                planned_gpu_refs,
                planned_tail_gpu_groups,
                planned_tail_gpu_refs,
                planned_ssd_groups,
                planned_gpu_cost,
                planned_ane_cost,
                100.0 * est_idle,
                (t_done - t_execute0) * 1000.0,
                hybrid_ane_groups,
                hybrid_ane_refs,
                hybrid_gpu_groups,
                hybrid_gpu_refs,
                concurrent_finishes,
                concurrent_waits,
                concurrent_gpu_overlap_groups);
    }

    if (scheduler_stats && async_reader_ok) {
        const uint64_t canceled =
            async_reader.canceled_queued +
            async_reader.canceled_reading +
            async_reader.canceled_ready +
            async_reader.canceled_finished;
        if (canceled) {
            fprintf(stderr,
                    "ds4: Flash-MoE async pread canceled queued=%" PRIu64
                    " reading=%" PRIu64 " ready=%" PRIu64
                    " finished=%" PRIu64 "\n",
                    async_reader.canceled_queued,
                    async_reader.canceled_reading,
                    async_reader.canceled_ready,
                    async_reader.canceled_finished);
        }
    }
    /* Leaving the eval phase: resume speculative reads so the next layer's
     * prefetch (queued just above) streams during the upcoming attention window. */
    if (xlayer_prefetch && async_reader_ok && flash_moe_xlayer_attn_only()) {
        ds4_flash_prefill_async_set_paused(p_async_reader, 0);
    }
    /* The persistent cross-layer reader is kept alive (it still holds the next
     * layer's queued reads, and its threads are reused); it is torn down in
     * metal_graph_free. Only a per-layer local reader is destroyed here. */
    if (!xlayer_prefetch) ds4_flash_prefill_async_destroy(&async_reader);
    free(async_submitted);
#undef async_reader
#undef DS4_SUBMIT_ASYNC_PREAD
    free(plan);
    free(streaming_weight_scales_f16);
    free(ref_weights);
    free(ref_tokens);
    free(pair_weights);
    free(true_ids);
    return ok;
}
