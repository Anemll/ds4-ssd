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
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        !g->flash_prefill_x || !g->flash_prefill_gate ||
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
    const bool prefill_slot_bank_cache_enabled =
        !g->flash_per_expert_buffers && !g->flash_direct_mmap_bank;
    const bool profile = backend_logs && env_flag_enabled("DS4_FLASH_MOE_PROFILE");
    const bool use_gpu_dedup = getenv("DS4_FLASH_MOE_GPU_DEDUP") == NULL || atoi(getenv("DS4_FLASH_MOE_GPU_DEDUP")) != 0;

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

        const int slot_cache_topk = get_prefill_slot_cache_target(g->flash_slot_bank);
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
    const bool try_ane_prefill =
        !plane_split_prefill && ane_prefill_requested && ane_prefill_supported;
    if (plane_split_prefill && ane_prefill_requested && backend_logs) {
        static bool warned_plane_split_ane_prefill = false;
        if (!warned_plane_split_ane_prefill) {
            fprintf(stderr,
                    "ds4: Flash-MoE ANE prefill disabled for MXFP4_NATIVE "
                    "plane-split sidecar; using native MXFP4 MPP prefill "
                    "(set DS4_MXFP4_NATIVE_DEQUANT_PREFILL_EXPERIMENT=1 to "
                    "try the diagnostic MPP/NAX dequant route; it may change "
                    "routing/quality)\n");
            warned_plane_split_ane_prefill = true;
        }
    }
    if (ane_prefill_requested && !ane_prefill_supported && backend_logs) {
        static bool warned_unsupported_flash_ane = false;
        if (!warned_unsupported_flash_ane) {
            char reason[256];
            fprintf(stderr,
                    "ds4: Flash-MoE ANE prefill requested but disabled: %s; "
                    "actual Flash-MoE ANE eval calls=0, using GPU/MPP fallback\n",
                    flash_moe_ane_prefill_unsupported_reason(layer,
                                                             reason,
                                                             sizeof(reason)));
            warned_unsupported_flash_ane = true;
        }
    }
    const bool try_mpp_int8_prefill =
        native_plane_prefill || flash_moe_mpp_int8_prefill_enabled();
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
    const bool async_pread_enabled = env_flag_enabled("DS4_FLASH_MOE_ASYNC_PREAD");
    const bool async_pread_after_stage =
        async_pread_enabled && env_flag_enabled("DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE");
    /* The reader is normally a per-layer local (created/destroyed each call).
     * With cross-layer prefetch it must outlive the call so the next layer's
     * experts queued at the end of THIS call survive to be consumed next call;
     * use the persistent reader held on the graph. The async_reader macro lets
     * the rest of this function reference whichever one is active unchanged. */
    ds4_flash_prefill_async_reader local_async_reader;
    const bool xlayer_prefetch = flash_moe_xlayer_prefetch_enabled(n_tokens);
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
        int prefetch = get_prefill_dedup_prefetch();
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
            valid_expert && metal_graph_flash_moe_find_resident_slot(g, il, expert, NULL);
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
                int32_t slot = -1;
                const uint64_t slot_hits_before = g->flash_hits;
                const uint64_t slot_misses_before = g->flash_misses;
                ok = token_view && weight_view &&
                     metal_graph_flash_moe_install(g, il, expert, NULL, &slot);
                if (ok) {
                    metal_graph_flash_moe_record_prefill_slot_cache(g,
                                                                     refs,
                                                                     slot_hits_before,
                                                                     slot_misses_before);
                    slot_gate_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_GATE, slot);
                    slot_up_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_UP, slot);
                    slot_down_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_DOWN, slot);
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
                int32_t slot = -1;
                const uint64_t slot_hits_before = g->flash_hits;
                const uint64_t slot_misses_before = g->flash_misses;
                ok = metal_graph_flash_moe_install(g, il, expert, NULL, &slot);
                if (ok) {
                    metal_graph_flash_moe_record_prefill_slot_cache(g,
                                                                     refs,
                                                                     slot_hits_before,
                                                                     slot_misses_before);
                    slot_gate_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_GATE, slot);
                    slot_up_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_UP, slot);
                    slot_down_view = metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_DOWN, slot);
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
        const bool try_ane_for_group = try_ane_prefill &&
            (!split_prefill || refs >= hybrid_ane_min_refs) &&
            (!overlap_scheduler || planned_lane == DS4_PREFILL_LANE_ANE);
        const bool try_mpp_for_group = try_mpp_int8_prefill &&
            (!overlap_scheduler || planned_lane == DS4_PREFILL_LANE_GPU);
        const bool try_pipelined_ane_for_group =
            ane_pipeline_prefill && gpu_compacted && try_ane_for_group;
        const bool try_concurrent_ane_for_group =
            !try_pipelined_ane_for_group &&
            concurrent_prefill && gpu_compacted && try_ane_for_group;
        ok = ds4_gpu_gather_rows_f32_tensor(g->flash_prefill_x,
                                            g->batch_ffn_norm,
                                            tokens_for_refs,
                                            refs,
                                            DS4_N_EMBD) != 0;
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
                                                                                refs);
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
                    concurrent_gpu_groups_since_ane >= concurrent_min_gpu_groups) {
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
                                                                                refs);
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
                                                                           &mid_is_f16) != 0;
                used_ane = ane_ok;
                if (ane_debug) {
                    fprintf(stderr,
                            "ds4: ANE prefill %s layer=%u expert=%d refs=%u\n",
                            ane_ok ? "ok" : "failed", il, expert, refs);
                }
            }
            if (!deferred_ane && !ane_ok && (try_mpp_for_group || try_mpp_int8_prefill)) {
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
                                                            weights_for_refs,
                                                            DS4_SWIGLU_CLAMP_EXP,
                                                            g->flash_prefill_x,
                                                            refs,
                                                            &mid_is_f16) != 0;
                used_mpp = ane_ok;
            }
            if (!deferred_ane && !ane_ok) {
                if (attempted_ane && env_flag_enabled("DS4_FLASH_MOE_ANE_DEBUG")) {
                    fprintf(stderr, "ds4: ANE prefill falling back to fp32 GPU layer=%u expert=%d refs=%u\n",
                            il, expert, refs);
                }
                mid_is_f16 = false;
                ok = ds4_gpu_routed_moe_expert_banked_batch_tensor(g->flash_prefill_out,
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
                                                                   &mid_is_f16) != 0;
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
            int prefetch = get_prefill_dedup_prefetch();
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
    if (backend_logs &&
        (hybrid_prefill || concurrent_prefill || ane_pipeline_prefill) &&
        (profile || env_flag_enabled("DS4_FLASH_MOE_HYBRID_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_CONCURRENT_STATS") ||
         env_flag_enabled("DS4_FLASH_MOE_ANE_PIPELINE_STATS"))) {
        fprintf(stderr,
                "ds4: Flash-MoE hybrid prefill layer=%u ane_groups=%u ane_refs=%" PRIu64
                " gpu_i8_groups=%u gpu_i8_refs=%" PRIu64 " fp32_groups=%u ane_min_refs=%u"
                " concurrent=%u ane_pipeline=%u output_queue=%u async_ane=%u gpu_overlap=%u finishes=%u waits=%u\n",
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
                concurrent_waits);
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
    free(ref_weights);
    free(ref_tokens);
    free(pair_weights);
    free(true_ids);
    return ok;
}
