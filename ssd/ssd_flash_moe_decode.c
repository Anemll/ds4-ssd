/* =========================================================================
 * ssd_flash_moe_decode.c - Flash-MoE decode staging, routing, and slots6 dispatch.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

static bool flash_moe_ane_prefill_tensor_types_supported(
        const ds4_layer_weights *layer) {
    if (!layer || !layer->ffn_gate_exps || !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    const bool iq2_path =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        (layer->ffn_down_exps->type == DS4_TENSOR_Q2_K ||
         layer->ffn_down_exps->type == DS4_TENSOR_IQ2_XXS);
    const bool q4_path =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    const bool mxfp4_path =
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    return iq2_path || q4_path || mxfp4_path;
}

static const char *flash_moe_ane_prefill_unsupported_reason(
        const ds4_layer_weights *layer,
        char                    *buf,
        size_t                   buf_sz) {
    if (!buf || buf_sz == 0) return "unknown";
    if (!layer || !layer->ffn_gate_exps || !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        snprintf(buf, buf_sz, "missing routed expert tensors");
        return buf;
    }
    snprintf(buf,
             buf_sz,
             "unsupported expert types gate=%s up=%s down=%s "
             "(Flash ANE prefill supports IQ2_XXS/IQ2_XXS/(Q2_K|IQ2_XXS), "
             "Q4_K/Q4_K/Q4_K and MXFP4/MXFP4/MXFP4)",
             tensor_type_name(layer->ffn_gate_exps->type),
             tensor_type_name(layer->ffn_up_exps->type),
             tensor_type_name(layer->ffn_down_exps->type));
    return buf;
}

#include "ssd_flash_moe_slots.h"

/* Prefetch hints remain soft; every victim-selection pass excludes the hard
 * reservations held by a complete decode or grouped-prefill request. */
static uint32_t metal_graph_flash_moe_pick_slot(
        ds4_gpu_graph  *g,
        const int32_t  *slot_to_expert,
        const uint64_t *slot_age,
        const bool     *protected_experts,
        const bool     *reserved_slots) {
    if (!g || g->flash_slot_bank == 0) return UINT32_MAX;
    return ds4_flash_moe_select_slot(g->flash_slot_bank, DS4_N_EXPERT,
            slot_to_expert, slot_age, protected_experts, reserved_slots);
}

static void metal_graph_flash_moe_invalidate_decode_slot(
        ds4_gpu_graph *g, uint32_t il, int32_t slot);
static void metal_graph_flash_moe_commit_decode_slot(
        ds4_gpu_graph *g, uint32_t il, int32_t true_expert,
        int32_t slot, int32_t evicted);

/* Install only a destination chosen by the completed reservation pass. */
static bool metal_graph_flash_moe_install_reserved(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        slot,
        int32_t        evicted) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) return false;
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    bool wrote = false;
    bool wrote_from_gpu_l2 = false;
    bool direct_metal_write = false;
    metal_graph_flash_moe_store_evicted_l1_slot(g, il, evicted, slot);
    if (g->flash_gpu_l2_slot_bank) {
        wrote = metal_graph_flash_moe_gpu_l2_promote_to_l1(g,
                                                           il,
                                                           true_expert,
                                                           (int32_t)slot);
        wrote_from_gpu_l2 = wrote;
    }
    if (!wrote && g->flash_shared_l2_slot_bank) {
        const uint8_t *l2_src = NULL;
        if (metal_graph_flash_moe_shared_l2_lookup(g, il, true_expert, &l2_src) && l2_src) {
            wrote = metal_graph_flash_moe_write_slot_from_buf(g,
                                                              il,
                                                              (int32_t)slot,
                                                              l2_src);
        }
    }
    if (!wrote && g->flash_l2_slot_bank) {
        const uint8_t *l2_src = NULL;
        if (metal_graph_flash_moe_l2_lookup(g, il, true_expert, &l2_src) && l2_src) {
            wrote = metal_graph_flash_moe_write_slot_from_buf(g,
                                                              il,
                                                              (int32_t)slot,
                                                              l2_src);
        }
    }
    if (!wrote && flash_moe_direct_slot_pread_enabled()) {
        errno = 0;
        if (g->flash_mixed_slot_bank) {
            uint8_t *dst = metal_graph_flash_moe_mixed_slot_ptr(g, il, (int32_t)slot);
            if (dst) {
                wrote = metal_graph_flash_moe_read_record_to_buf(layer,
                                                                  true_expert,
                                                                  dst,
                                                                  flash_moe_cache_io_split());
                direct_metal_write = wrote;
            }
        } else {
            uint8_t *family_dst[DS4_FLASH_FAMILY_COUNT] = { NULL, NULL, NULL };
            wrote = metal_graph_flash_moe_direct_slot_ptrs(g, il, (int32_t)slot, family_dst) &&
                    metal_graph_flash_moe_pread_slot_direct(layer,
                                                            true_expert,
                                                            family_dst,
                                                            flash_moe_cache_io_split());
            direct_metal_write = wrote;
        }
        if (!wrote && errno != 0) {
            fprintf(stderr,
                    "ds4: Flash-MoE failed to direct-read layer %u expert %d: %s\n",
                    il,
                    true_expert,
                    strerror(errno));
            metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
            return false;
        }
    }
    if (!wrote) {
        if (!metal_graph_flash_moe_read_record_to_buf(layer,
                                                      true_expert,
                                                      g->flash_install_buf,
                                                      flash_moe_cache_io_split())) {
            fprintf(stderr,
                    "ds4: Flash-MoE failed to read layer %u expert %d: %s\n",
                    il,
                    true_expert,
                    strerror(errno));
            metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
            return false;
        }
        wrote = metal_graph_flash_moe_write_slot_from_buf(g,
                                                          il,
                                                          (int32_t)slot,
                                                          g->flash_install_buf);
    }
    if (wrote && g->flash_gpu_l2_slot_bank && !wrote_from_gpu_l2) {
        (void)metal_graph_flash_moe_gpu_l2_store_l1_slot(g,
                                                         il,
                                                         true_expert,
                                                         (int32_t)slot);
    }
    if (!wrote) {
        fprintf(stderr, "ds4: Flash-MoE failed to upload layer %u expert %d into slot %d\n",
                il,
                true_expert,
                slot);
        metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
        return false;
    }
    if (direct_metal_write && !metal_graph_flash_moe_mark_slot_modified(g, il, (int32_t)slot)) {
        fprintf(stderr, "ds4: Flash-MoE failed to mark layer %u slot %d modified\n",
                il,
                slot);
        metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
        return false;
    }

    metal_graph_flash_moe_commit_decode_slot(g, il, true_expert, slot, evicted);
    g->flash_installed_bytes += layer->expert_stride;
    return true;
}

static bool metal_graph_flash_moe_reserve_decode_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        const bool    *protected_experts,
        const bool    *reserved_slots,
        int32_t       *slot_out,
        int32_t       *evicted_out,
        bool          *miss_out) {
    if (!g || !g->flash_moe || !slot_out || !evicted_out || !miss_out ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    *slot_out = -1;
    *evicted_out = -1;
    *miss_out = false;

    int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
    int32_t *expert_to_slot = flash_moe_expert_to_slot(g, il);
    uint64_t *slot_age = flash_moe_slot_age(g, il);
    if (g->flash_direct_mmap_bank) {
        const int32_t slot = true_expert;
        if (slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
            return false;
        }
        slot_to_expert[slot] = true_expert;
        expert_to_slot[true_expert] = slot;
        slot_age[slot] = ++g->flash_age;
        g->flash_hits++;
        *slot_out = slot;
        *miss_out = false;
        return true;
    }
    if (g->flash_per_expert_buffers) {
        const int32_t slot = true_expert;
        if (slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
            !g->flash_expert_bank[il][true_expert]) {
            return false;
        }
        slot_to_expert[slot] = true_expert;
        expert_to_slot[true_expert] = slot;
        slot_age[slot] = ++g->flash_age;
        g->flash_hits++;
        *slot_out = slot;
        *miss_out = false;
        return true;
    }
    const int32_t existing = expert_to_slot[true_expert];
    if (existing >= 0 && existing < (int32_t)g->flash_slot_bank &&
        slot_to_expert[existing] == true_expert) {
        slot_age[existing] = ++g->flash_age;
        g->flash_hits++;
        *slot_out = existing;
        *miss_out = false;
        return true;
    }

    g->flash_misses++;
    /* A matching resident hit may already be hard-reserved by the request
     * prepass. Only miss victim selection must skip reserved slots. */
    const uint32_t slot = metal_graph_flash_moe_pick_slot(
            g, slot_to_expert, slot_age, protected_experts, reserved_slots);
    if (slot == UINT32_MAX) return false;

    *slot_out = (int32_t)slot;
    *evicted_out = slot_to_expert[slot];
    *miss_out = true;
    return true;
}

static void metal_graph_flash_moe_protect_routed_experts(
        const int32_t *true_ids,
        uint32_t       n_ids,
        bool          *protected_experts) {
    if (!true_ids || !protected_experts) return;
    for (uint32_t k = 0; k < n_ids; k++) {
        if (true_ids[k] >= 0 && true_ids[k] < (int32_t)DS4_N_EXPERT) {
            protected_experts[true_ids[k]] = true;
        }
    }
}

/* A plan never installs or changes mappings. Validate all inputs and protect
 * all resident hits before selecting the first miss destination. */
static bool metal_graph_flash_moe_resolve_request(
        ds4_gpu_graph *g, uint32_t il, const int32_t *true_ids, uint32_t n_ids,
        const bool *protected_experts, int32_t *slot_ids,
        int32_t *evicted, bool *misses) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER || !true_ids ||
        !slot_ids || !evicted || !misses || n_ids > DS4_MAX_EXPERT ||
        g->flash_slot_bank == 0 || g->flash_slot_bank > DS4_MAX_EXPERT) return false;
    if (g->flash_direct_mmap_bank || g->flash_per_expert_buffers) {
        for (uint32_t k = 0; k < n_ids; k++) {
            const int32_t expert = true_ids[k];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT ||
                expert >= (int32_t)g->flash_slot_bank ||
                (g->flash_per_expert_buffers && !g->flash_expert_bank[il][expert])) {
                return false;
            }
        }
        for (uint32_t k = 0; k < n_ids; k++) {
            slot_ids[k] = true_ids[k];
            evicted[k] = -1;
            misses[k] = false;
        }
        return true;
    }
    bool reserved_slots[DS4_MAX_EXPERT] = { false };
    return ds4_flash_moe_resolve_request_slots(true_ids, n_ids, DS4_N_EXPERT,
            flash_moe_expert_to_slot(g, il), flash_moe_slot_to_expert(g, il),
            flash_moe_slot_age(g, il), g->flash_slot_bank, protected_experts,
            reserved_slots, slot_ids, evicted, misses);
}

static void metal_graph_flash_moe_note_request_slot(
        ds4_gpu_graph *g, uint32_t il, int32_t expert, int32_t slot, bool miss) {
    if (miss) {
        g->flash_misses++;
    } else {
        if (g->flash_direct_mmap_bank || g->flash_per_expert_buffers) {
            flash_moe_slot_to_expert(g, il)[slot] = expert;
            flash_moe_expert_to_slot(g, il)[expert] = slot;
        }
        flash_moe_slot_age(g, il)[slot] = ++g->flash_age;
        g->flash_hits++;
    }
}

static bool metal_graph_flash_moe_install_request(
        ds4_gpu_graph *g, uint32_t il, const int32_t *true_ids, uint32_t n_ids,
        const bool *protected_experts, int32_t *slot_ids) {
    int32_t evicted[DS4_MAX_EXPERT];
    bool misses[DS4_MAX_EXPERT];
    if (!metal_graph_flash_moe_resolve_request(g, il, true_ids, n_ids,
            protected_experts, slot_ids, evicted, misses)) return false;
    for (uint32_t k = 0; k < n_ids; k++) {
        bool duplicate = false;
        for (uint32_t prev = 0; prev < k; prev++) {
            if (true_ids[prev] == true_ids[k]) { duplicate = true; break; }
        }
        if (duplicate) continue;
        metal_graph_flash_moe_note_request_slot(g, il, true_ids[k], slot_ids[k], misses[k]);
        if (misses[k] && !metal_graph_flash_moe_install_reserved(g, il,
                true_ids[k], slot_ids[k], evicted[k])) return false;
    }
    return true;
}

/* Expert-major prefill consumes one expert before reusing the shared bank;
 * its single-expert request intentionally permits soft-hint fallback. */
static bool metal_graph_flash_moe_install(
        ds4_gpu_graph *g, uint32_t il, int32_t true_expert,
        const bool *protected_experts, int32_t *slot_out) {
    return metal_graph_flash_moe_install_request(g, il, &true_expert, 1,
            protected_experts, slot_out);
}

static void metal_graph_flash_moe_commit_decode_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        slot,
        int32_t        evicted) {
    int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
    int32_t *expert_to_slot = flash_moe_expert_to_slot(g, il);
    uint64_t *slot_age = flash_moe_slot_age(g, il);
    metal_graph_flash_moe_replay_invalidate_slot(g, il, slot);
    if (evicted >= 0 && evicted < (int32_t)DS4_N_EXPERT &&
        expert_to_slot[evicted] == slot) {
        expert_to_slot[evicted] = -1;
    }
    const int32_t resident = slot_to_expert[slot];
    if (resident >= 0 && resident < (int32_t)DS4_N_EXPERT &&
        resident != evicted && resident != true_expert &&
        expert_to_slot[resident] == slot) {
        expert_to_slot[resident] = -1;
    }
    slot_to_expert[slot] = true_expert;
    expert_to_slot[true_expert] = slot;
    slot_age[slot] = ++g->flash_age;
    metal_graph_flash_moe_replay_mark_resident_missing(g, il, slot, true_expert);
}

static void metal_graph_flash_moe_invalidate_decode_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return;
    }
    int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
    int32_t *expert_to_slot = flash_moe_expert_to_slot(g, il);
    uint64_t *slot_age = flash_moe_slot_age(g, il);
    const int32_t resident = slot_to_expert[slot];
    metal_graph_flash_moe_replay_invalidate_slot(g, il, slot);
    if (resident >= 0 && resident < (int32_t)DS4_N_EXPERT &&
        expert_to_slot[resident] == slot) {
        expert_to_slot[resident] = -1;
    }
    slot_to_expert[slot] = -1;
    slot_age[slot] = 0;
    /* Routing IDs can be recorded before an async read starts. A failed or
     * abandoned destination invalidates that recorded compute plan as well. */
    g->flash_decode_ids_valid[il] = 0;
}

static bool metal_graph_flash_moe_upload_decode_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        slot,
        int32_t        evicted,
        const uint8_t *src_buf) {
    if (!g || !g->flash_moe || !src_buf ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return false;
    }

    const double t0 = now_sec();
    const bool wrote = metal_graph_flash_moe_write_slot_from_buf(g, il, slot, src_buf);
    const double t1 = now_sec();
    g->flash_decode_prefetch_upload_ms += (t1 - t0) * 1000.0;
    if (!wrote) {
        fprintf(stderr,
                "ds4: Flash-MoE failed to upload decode layer %u expert %d into slot %d\n",
                il,
                true_expert,
                slot);
        metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
        return false;
    }
    if (g->flash_gpu_l2_slot_bank) {
        (void)metal_graph_flash_moe_gpu_l2_store_l1_slot(g,
                                                         il,
                                                         true_expert,
                                                         slot);
    }
    metal_graph_flash_moe_commit_decode_slot(g, il, true_expert, slot, evicted);
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    g->flash_installed_bytes += layer->expert_stride;
    return true;
}

static bool metal_graph_flash_moe_upload_prefill_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        slot,
        int32_t        evicted,
        const uint8_t *src_buf) {
    if (!g || !g->flash_moe || !src_buf ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return false;
    }

    const bool wrote = metal_graph_flash_moe_write_slot_from_buf(g, il, slot, src_buf);
    if (!wrote) {
        fprintf(stderr,
                "ds4: Flash-MoE failed to prefill-upload layer %u expert %d into slot %d\n",
                il,
                true_expert,
                slot);
        metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
        return false;
    }
    metal_graph_flash_moe_commit_decode_slot(g, il, true_expert, slot, evicted);
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    g->flash_installed_bytes += layer->expert_stride;
    return true;
}

static bool metal_graph_flash_moe_prefetch_slot_from_buf(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        uint32_t       refs,
        const uint8_t *src_buf) {
    if (!g || !src_buf) return false;
    int32_t slot = -1;
    int32_t evicted = -1;
    bool miss = false;
    const uint64_t hits_before = g->flash_hits;
    const uint64_t misses_before = g->flash_misses;
    bool ok = metal_graph_flash_moe_reserve_decode_slot(g,
                                                        il,
                                                        true_expert,
                                                        NULL,
                                                        NULL,
                                                        &slot,
                                                        &evicted,
                                                        &miss);
    if (ok && miss) {
        metal_graph_flash_moe_store_evicted_l1_slot(g, il, evicted, slot);
    }
    if (ok && miss) {
        ok = metal_graph_flash_moe_upload_prefill_slot(g,
                                                       il,
                                                       true_expert,
                                                       slot,
                                                       evicted,
                                                       src_buf);
    }
    if (ok) {
        metal_graph_flash_moe_record_prefill_slot_cache(g,
                                                        refs,
                                                        hits_before,
                                                        misses_before);
    }
    return ok;
}

static void metal_graph_flash_moe_decode_prefetch_cleanup(
        ds4_gpu_graph *g,
        uint32_t il,
        ds4_flash_decode_prefetch *pf) {
    if (!pf) return;
    __atomic_store_n(&pf->stop_requested, 1, __ATOMIC_RELEASE);
    for (uint32_t i = 0; i < pf->n_loads; i++) {
        if (pf->thread_started[i]) {
            pthread_join(pf->thread[i], NULL);
            pf->thread_started[i] = false;
        }
        /* Cleanup abandons every uncommitted load. A direct read can have
         * overwritten part or all of the old resident even when it failed. */
        if (pf->job[i].direct_record || pf->job[i].direct_slot) {
            metal_graph_flash_moe_invalidate_decode_slot(g, il, pf->load_slot[i]);
        }
        if (pf->job[i].buf_owned) free(pf->job[i].buf);
        pf->job[i].buf = NULL;
        pf->job[i].buf_owned = false;
    }
    pf->n_loads = 0;
    pf->active = false;
}

static bool metal_graph_flash_moe_decode_prefetch_finish(
        ds4_gpu_graph              *g,
        uint32_t                    il,
        ds4_flash_decode_prefetch  *pf) {
    if (!pf || !pf->active) return true;
    bool ok = true;
    for (uint32_t i = 0; i < pf->n_loads; i++) {
        if (pf->thread_started[i]) {
            pthread_join(pf->thread[i], NULL);
            pf->thread_started[i] = false;
        }
        ds4_flash_decode_read_job *job = &pf->job[i];
        g->flash_decode_prefetch_pread_ms += job->pread_t1_ms - job->pread_t0_ms;
        if (pf->sequential_scratch &&
            (job->canceled || !ds4_flash_decode_read_job_is_complete(job))) {
            pf->needs_sync_prepare = true;
            if (job->direct_record || job->direct_slot) {
                metal_graph_flash_moe_invalidate_decode_slot(g, il, pf->load_slot[i]);
            }
            if (job->buf_owned) free(job->buf);
            job->buf = NULL;
            job->buf_owned = false;
            continue;
        }
        if (job->err) {
            fprintf(stderr,
                    "ds4: Flash-MoE decode prefetch failed layer %u expert %d: %s\n",
                    il,
                    pf->load_expert[i],
                    strerror(job->err));
            ok = false;
        }
        if (ok) {
            if (job->direct_record || job->direct_slot) {
                ok = metal_graph_flash_moe_mark_slot_modified(g, il, pf->load_slot[i]);
                if (ok) {
                    if (g->flash_gpu_l2_slot_bank) {
                        (void)metal_graph_flash_moe_gpu_l2_store_l1_slot(g,
                                                                         il,
                                                                         pf->load_expert[i],
                                                                         pf->load_slot[i]);
                    }
                    metal_graph_flash_moe_commit_decode_slot(g,
                                                             il,
                                                             pf->load_expert[i],
                                                             pf->load_slot[i],
                                                             pf->load_evicted[i]);
                    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
                    g->flash_installed_bytes += layer->expert_stride;
                }
            } else {
                ok = metal_graph_flash_moe_upload_decode_slot(g,
                                                              il,
                                                              pf->load_expert[i],
                                                              pf->load_slot[i],
                                                              pf->load_evicted[i],
                                                              job->buf);
            }
        }
        if (!ok && (job->direct_record || job->direct_slot)) {
            metal_graph_flash_moe_invalidate_decode_slot(g, il, pf->load_slot[i]);
        }
        if (job->buf_owned) free(job->buf);
        job->buf = NULL;
        job->buf_owned = false;
    }
    __atomic_store_n(&pf->stop_requested, 0, __ATOMIC_RELEASE);
    pf->n_loads = 0;
    pf->active = false;
    return ok;
}

static bool metal_graph_flash_moe_prepare_decode_prefetch_ids(
        ds4_gpu_graph             *g,
        uint32_t                   il,
        const int32_t             *true_ids,
        ds4_flash_decode_prefetch *pf) {
    if (!g || !g->flash_moe || !true_ids || !pf) return false;
    if (il >= DS4_N_LAYER || !g->router_slot_selected) return false;
    memset(pf, 0, sizeof(*pf));

    bool protected_experts[DS4_MAX_EXPERT];
    int32_t request_evicted[DS4_MAX_EXPERT_USED];
    bool request_misses[DS4_MAX_EXPERT_USED];
    memset(protected_experts, 0, sizeof(protected_experts));
    bool ok = true;
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint32_t max_async_loads = flash_moe_decode_prefetch_max_loads();
    const bool scratch_only = flash_moe_decode_prefetch_scratch_only_enabled();
    if (scratch_only) pf->sequential_scratch = true;
    if (flash_moe_preprotect_topk_enabled()) {
        metal_graph_flash_moe_protect_routed_experts(true_ids,
                                                     active_expert_used,
                                                     protected_experts);
    }
    ok = metal_graph_flash_moe_resolve_request(g, il, true_ids,
            active_expert_used, protected_experts, pf->slot_ids,
            request_evicted, request_misses);
    for (uint32_t k = 0; ok && k < active_expert_used; k++) {
        bool duplicate = false;
        for (uint32_t prev = 0; prev < k; prev++) {
            if (true_ids[prev] == true_ids[k]) { duplicate = true; break; }
        }
        if (duplicate) continue;
        const int32_t slot = pf->slot_ids[k];
        const int32_t evicted = request_evicted[k];
        const bool miss = request_misses[k];
        if (miss && pf->n_loads >= max_async_loads) {
            pf->needs_sync_prepare = true;
            break;
        }
        metal_graph_flash_moe_note_request_slot(g, il, true_ids[k], slot, miss);
        if (miss) metal_graph_flash_moe_store_evicted_l1_slot(g, il, evicted, slot);
        if (ok && miss) {
            if (pf->n_loads >= active_expert_used) {
                ok = false;
                break;
            }
            const uint32_t li = pf->n_loads++;
            ds4_flash_decode_read_job *job = &pf->job[li];
            pf->load_expert[li] = true_ids[k];
            pf->load_slot[li] = slot;
            pf->load_evicted[li] = evicted;
            ok = ds4_flash_decode_read_job_set_sidecar(job,
                                                       layer,
                                                       true_ids[k],
                                                       flash_moe_cache_io_split());
            if (!ok) break;
            if (g->flash_shared_l2_slot_bank) {
                const uint8_t *l2_src = NULL;
                if (metal_graph_flash_moe_shared_l2_lookup(g, il, true_ids[k], &l2_src) && l2_src) {
                    ok = ds4_flash_decode_read_job_set_memory_src_copy(
                            job, l2_src, layer->expert_stride);
                }
            }
            if (g->flash_l2_slot_bank) {
                const uint8_t *l2_src = NULL;
                if (ok && !job->memory_src &&
                    metal_graph_flash_moe_l2_lookup(g, il, true_ids[k], &l2_src) && l2_src) {
                    ok = ds4_flash_decode_read_job_set_memory_src_copy(
                            job, l2_src, layer->expert_stride);
                }
            }
            if (flash_moe_decode_prefetch_direct_slot_pread_enabled()) {
                if (g->flash_mixed_slot_bank) {
                    uint8_t *dst = metal_graph_flash_moe_mixed_slot_ptr(g, il, slot);
                    if (dst) {
                        job->direct_record = true;
                        job->record_dst = dst;
                    }
                } else {
                    uint8_t *family_dst[DS4_FLASH_FAMILY_COUNT] = { NULL, NULL, NULL };
                    if (metal_graph_flash_moe_direct_slot_ptrs(g, il, slot, family_dst)) {
                        job->direct_slot = true;
                        for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
                            job->family_dst[fam] = family_dst[fam];
                        }
                    }
                }
            }
            if (!job->direct_record && !job->direct_slot && !job->memory_src && scratch_only) {
                if (li >= g->flash_decode_prefetch_scratch_slots ||
                    !g->flash_decode_prefetch_scratch ||
                    layer->expert_stride > g->flash_decode_prefetch_scratch_stride) {
                    ok = false;
                    break;
                }
                job->buf = g->flash_decode_prefetch_scratch +
                           (uint64_t)li * g->flash_decode_prefetch_scratch_stride;
                job->buf_owned = false;
            } else if (!job->direct_record && !job->direct_slot && !job->memory_src) {
                job->buf = xmalloc((size_t)layer->expert_stride);
                job->buf_owned = true;
            }
            g->flash_decode_prefetch_loads++;
            g->flash_decode_prefetch_bytes += layer->expert_stride;
            if (pf->n_loads >= max_async_loads && k + 1u < active_expert_used) {
                pf->needs_sync_prepare = true;
                break;
            }
        }
    }
    if (ok && !pf->needs_sync_prepare) {
        ok = ds4_gpu_tensor_write(g->router_slot_selected,
                                  0,
                                  pf->slot_ids,
                                  (uint64_t)active_expert_used * sizeof(pf->slot_ids[0])) != 0;
    }
    if (ok && !pf->needs_sync_prepare) {
        metal_graph_flash_moe_record_decode_slots(g,
                                                  il,
                                                  true_ids,
                                                  pf->slot_ids,
                                                  active_expert_used);
    }
    if (!ok) {
        metal_graph_flash_moe_decode_prefetch_cleanup(g, il, pf);
        return false;
    }
    /* No worker may overwrite a victim until every request slot and every
     * staged job has been resolved successfully. */
    if (!scratch_only) {
        for (uint32_t li = 0; li < pf->n_loads; li++) {
            if (pthread_create(&pf->thread[li], NULL,
                    ds4_flash_decode_read_thread, &pf->job[li]) == 0) {
                pf->thread_started[li] = true;
            } else {
                ds4_flash_decode_read_thread(&pf->job[li]);
            }
        }
    }
    if (scratch_only && pf->n_loads > 0) {
        if (pthread_create(&pf->thread[0],
                           NULL,
                           ds4_flash_decode_scratch_prefetch_worker,
                           pf) == 0) {
            pf->thread_started[0] = true;
        } else {
            ds4_flash_decode_scratch_prefetch_worker(pf);
        }
    }
    pf->active = true;
    g->flash_decode_prefetch_calls++;
    return true;
}

static bool metal_graph_flash_moe_prepare_decode_prefetch(
        ds4_gpu_graph             *g,
        uint32_t                   il,
        uint32_t                   pos,
        ds4_flash_decode_prefetch *pf) {
    if (!g || !g->flash_moe || !pf) return false;
    if (il >= DS4_N_LAYER || !g->router_selected) return false;

    int32_t true_ids[DS4_MAX_EXPERT_USED];
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    if (ds4_gpu_end_commands() == 0) return false;
    const bool ok_read =
        ds4_gpu_tensor_read(g->router_selected,
                            0,
                            true_ids,
                            (uint64_t)active_expert_used * sizeof(true_ids[0])) != 0;
    if (!ok_read) return false;
    flash_moe_decode_trace_record(pos, il, true_ids);
    return metal_graph_flash_moe_prepare_decode_prefetch_ids(g, il, true_ids, pf);
}

static void metal_graph_flash_moe_record_prefill_pread(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        uint32_t       refs,
        uint64_t       bytes,
        double         pread_t0_ms,
        double         pread_t1_ms,
        double        *issue_gap_out,
        double        *post_gap_out) {
    double issue_gap_ms = 0.0;
    double post_gap_ms = 0.0;
    const bool have_previous_pread = g->flash_prefill_pread_last_end_ms > 0.0;
    const bool same_layer_as_previous =
        have_previous_pread && g->flash_prefill_pread_last_layer == il;
    if (have_previous_pread && pread_t0_ms > g->flash_prefill_pread_last_end_ms) {
        issue_gap_ms = pread_t0_ms - g->flash_prefill_pread_last_end_ms;
        g->flash_prefill_pread_issue_gap_ms += issue_gap_ms;
        g->flash_prefill_pread_issue_gap_count++;
        if (issue_gap_ms > g->flash_prefill_pread_max_issue_gap_ms) {
            g->flash_prefill_pread_max_issue_gap_ms = issue_gap_ms;
        }
        if (issue_gap_ms > 1.0) g->flash_prefill_pread_issue_gap_gt_1ms++;
        if (issue_gap_ms > 5.0) g->flash_prefill_pread_issue_gap_gt_5ms++;
        if (issue_gap_ms > 20.0) g->flash_prefill_pread_issue_gap_gt_20ms++;
        if (same_layer_as_previous) {
            g->flash_prefill_pread_same_layer_issue_gap_ms += issue_gap_ms;
        } else {
            g->flash_prefill_pread_cross_layer_issue_gap_ms += issue_gap_ms;
        }
        ds4_prefill_trace_interval_ms("sidecar_pread_issue_gap",
                                      il,
                                      true_expert,
                                      g->flash_prefill_pread_last_end_ms,
                                      pread_t0_ms);
    }
    if (have_previous_pread && g->flash_prefill_stage_last_end_ms > 0.0 &&
        pread_t0_ms > g->flash_prefill_stage_last_end_ms) {
        post_gap_ms = pread_t0_ms - g->flash_prefill_stage_last_end_ms;
        g->flash_prefill_pread_post_stage_gap_ms += post_gap_ms;
        g->flash_prefill_pread_post_stage_gap_count++;
        if (post_gap_ms > g->flash_prefill_pread_max_post_gap_ms) {
            g->flash_prefill_pread_max_post_gap_ms = post_gap_ms;
        }
        if (post_gap_ms > 1.0) g->flash_prefill_pread_post_gap_gt_1ms++;
        if (post_gap_ms > 5.0) g->flash_prefill_pread_post_gap_gt_5ms++;
        if (post_gap_ms > 20.0) g->flash_prefill_pread_post_gap_gt_20ms++;
        if (same_layer_as_previous) {
            g->flash_prefill_pread_same_layer_post_gap_ms += post_gap_ms;
        } else {
            g->flash_prefill_pread_cross_layer_post_gap_ms += post_gap_ms;
        }
        ds4_prefill_trace_interval_ms("sidecar_pread_post_stage_gap",
                                      il,
                                      true_expert,
                                      g->flash_prefill_stage_last_end_ms,
                                      pread_t0_ms);
    }
    if (g->flash_prefill_pread_first_start_ms == 0.0) {
        g->flash_prefill_pread_first_start_ms = pread_t0_ms;
    }
    g->flash_prefill_pread_last_end_ms = pread_t1_ms;
    g->flash_prefill_pread_last_layer = il;
    g->flash_prefill_stage_pread_ms += pread_t1_ms - pread_t0_ms;
    {
        const uint32_t bucket = ds4_prefill_stage_ref_bucket(refs);
        g->flash_prefill_stage_bucket_calls[bucket]++;
        g->flash_prefill_stage_bucket_refs[bucket] += refs;
        g->flash_prefill_stage_bucket_bytes[bucket] += bytes;
        g->flash_prefill_stage_bucket_pread_ms[bucket] += pread_t1_ms - pread_t0_ms;
        g->flash_prefill_stage_bucket_issue_gap_ms[bucket] += issue_gap_ms;
        g->flash_prefill_stage_bucket_post_gap_ms[bucket] += post_gap_ms;
    }
    if (issue_gap_out) *issue_gap_out = issue_gap_ms;
    if (post_gap_out) *post_gap_out = post_gap_ms;
}

static bool metal_graph_flash_moe_upload_prefill_expert(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int            bank_set,
        uint32_t       refs,
        const uint8_t *src_buf,
        double         pread_t0_ms,
        double         pread_t1_ms) {
    if (!g || !g->flash_moe || !src_buf ||
        il >= DS4_N_LAYER || true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }

    ds4_gpu_tensor *gate_bank, *up_bank, *down_bank;
    switch (bank_set) {
    case 1:  gate_bank = g->flash_prefill_gate_bank2; up_bank = g->flash_prefill_up_bank2; down_bank = g->flash_prefill_down_bank2; break;
    case 2:  gate_bank = g->flash_prefill_gate_bank3; up_bank = g->flash_prefill_up_bank3; down_bank = g->flash_prefill_down_bank3; break;
    case 3:  gate_bank = g->flash_prefill_gate_bank4; up_bank = g->flash_prefill_up_bank4; down_bank = g->flash_prefill_down_bank4; break;
    default: gate_bank = g->flash_prefill_gate_bank;  up_bank = g->flash_prefill_up_bank;  down_bank = g->flash_prefill_down_bank;  break;
    }
    if (!gate_bank || !up_bank || !down_bank) return false;

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    metal_graph_flash_moe_tag_layer_family_storage(
            g,
            il,
            gate_bank,
            up_bank,
            down_bank);
    ds4_prefill_trace_interval_ms("sidecar_pread",
                                  il,
                                  true_expert,
                                  pread_t0_ms,
                                  pread_t1_ms);

    const uint64_t gate_bytes = layer->family_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_bytes   = layer->family_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_bytes = layer->family_bytes[DS4_FLASH_FAMILY_DOWN];

    const double gate_t0 = now_sec();
    const bool gate_ok =
        ds4_gpu_tensor_write(gate_bank, 0,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_GATE],
                             gate_bytes) != 0;
    const double gate_t1 = now_sec();
    const bool up_ok = gate_ok &&
        ds4_gpu_tensor_write(up_bank, 0,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_UP],
                             up_bytes) != 0;
    const double up_t1 = now_sec();
    const bool down_ok = up_ok &&
        ds4_gpu_tensor_write(down_bank, 0,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                             down_bytes) != 0;
    const double down_t1 = now_sec();
    const bool ok = gate_ok && up_ok && down_ok;

    if (!ok) {
        fprintf(stderr, "ds4: Flash-MoE failed to stage prefill layer %u expert %d (bank %d)\n",
                il, true_expert, bank_set);
        return false;
    }

    g->flash_misses++;
    g->flash_installed_bytes += layer->expert_stride;
    g->flash_prefill_stage_calls++;
    g->flash_prefill_stage_bytes += layer->expert_stride;
    metal_graph_flash_moe_record_prefill_pread(g,
                                               il,
                                               true_expert,
                                               refs,
                                               layer->expert_stride,
                                               pread_t0_ms,
                                               pread_t1_ms,
                                               NULL,
                                               NULL);
    g->flash_prefill_stage_upload_gate_ms += (gate_t1 - gate_t0) * 1000.0;
    g->flash_prefill_stage_upload_up_ms += (up_t1 - gate_t1) * 1000.0;
    g->flash_prefill_stage_upload_down_ms += (down_t1 - up_t1) * 1000.0;
    g->flash_prefill_stage_upload_ms += (down_t1 - gate_t0) * 1000.0;
    g->flash_prefill_stage_last_end_ms = down_t1 * 1000.0;
    ds4_prefill_trace_interval_ms("metal_upload",
                                  il,
                                  true_expert,
                                  gate_t0 * 1000.0,
                                  down_t1 * 1000.0);
    return true;
}

static bool metal_graph_flash_moe_stage_prefill_expert_async(
        ds4_gpu_graph                  *g,
        ds4_flash_prefill_async_reader *reader,
        uint32_t                        il,
        int32_t                         true_expert,
        int                             bank_set,
        uint32_t                        refs,
        uint8_t                       **src_copy_out) {
    if (src_copy_out) *src_copy_out = NULL;
    if (!reader || !reader->initialized) return false;
    pthread_mutex_lock(&reader->mu);
    int idx = ds4_flash_prefill_async_find_locked(reader, il, true_expert, bank_set);
    /* This is an on-demand consume: if the matching slot is a paused speculative
     * read, promote it so a reader serves it now -- otherwise the wait below
     * would deadlock against the pause held during ANE eval. */
    if (idx >= 0 && reader->slots[idx].speculative) {
        reader->slots[idx].speculative = false;
        pthread_cond_broadcast(&reader->cv);
    }
    while (idx >= 0 &&
           reader->slots[idx].state != DS4_FLASH_ASYNC_READY &&
           reader->slots[idx].state != DS4_FLASH_ASYNC_ERROR) {
        pthread_cond_wait(&reader->cv, &reader->mu);
        idx = ds4_flash_prefill_async_find_locked(reader, il, true_expert, bank_set);
        if (idx >= 0 && reader->slots[idx].speculative) {
            reader->slots[idx].speculative = false;
            pthread_cond_broadcast(&reader->cv);
        }
    }
    if (idx < 0) {
        pthread_mutex_unlock(&reader->mu);
        return false;
    }
    ds4_flash_prefill_async_slot *slot = &reader->slots[idx];
    const bool read_ok = slot->state == DS4_FLASH_ASYNC_READY;
    const int err = slot->err;
    const double pread_t0_ms = slot->pread_t0_ms;
    const double pread_t1_ms = slot->pread_t1_ms;
    const uint64_t bytes = slot->bytes;
    uint8_t *buf = slot->buf;

    if (!read_ok) {
        slot->state = DS4_FLASH_ASYNC_EMPTY;
        slot->canceled = false;
        pthread_cond_broadcast(&reader->cv);
        pthread_mutex_unlock(&reader->mu);
        fprintf(stderr,
                "ds4: Flash-MoE async pread failed layer %u expert %d: %s\n",
                il,
                true_expert,
                strerror(err ? err : EIO));
        return false;
    }
    slot->state = DS4_FLASH_ASYNC_CONSUMING;
    slot->canceled = false;
    pthread_mutex_unlock(&reader->mu);
    (void)bytes;
    const bool ok = metal_graph_flash_moe_upload_prefill_expert(g,
                                                                il,
                                                                true_expert,
                                                                bank_set,
                                                                refs,
                                                                buf,
                                                                pread_t0_ms,
                                                                pread_t1_ms);
    if (ok && src_copy_out && bytes != 0 && bytes <= SIZE_MAX) {
        uint8_t *copy = xmalloc((size_t)bytes);
        memcpy(copy, buf, (size_t)bytes);
        *src_copy_out = copy;
    }
    pthread_mutex_lock(&reader->mu);
    slot->state = DS4_FLASH_ASYNC_EMPTY;
    slot->canceled = false;
    pthread_cond_broadcast(&reader->cv);
    pthread_mutex_unlock(&reader->mu);
    return ok;
}

static bool metal_graph_flash_moe_stage_prefill_expert(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int            bank_set,          /* 0..3 for prefill bank rotation */
        uint32_t       refs)
{
    if (!g || !g->flash_moe ||
        il >= DS4_N_LAYER || true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }

    ds4_gpu_tensor *gate_bank, *up_bank, *down_bank;
    switch (bank_set) {
    case 1:  gate_bank = g->flash_prefill_gate_bank2; up_bank = g->flash_prefill_up_bank2; down_bank = g->flash_prefill_down_bank2; break;
    case 2:  gate_bank = g->flash_prefill_gate_bank3; up_bank = g->flash_prefill_up_bank3; down_bank = g->flash_prefill_down_bank3; break;
    case 3:  gate_bank = g->flash_prefill_gate_bank4; up_bank = g->flash_prefill_up_bank4; down_bank = g->flash_prefill_down_bank4; break;
    default: gate_bank = g->flash_prefill_gate_bank;  up_bank = g->flash_prefill_up_bank;  down_bank = g->flash_prefill_down_bank;  break;
    }

    if (!gate_bank || !up_bank || !down_bank) return false;

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    metal_graph_flash_moe_tag_layer_family_storage(
            g,
            il,
            gate_bank,
            up_bank,
            down_bank);
    const double pread_t0 = now_sec();
    const double pread_t0_ms = pread_t0 * 1000.0;
    double issue_gap_ms = 0.0;
    double post_gap_ms = 0.0;
    const bool have_previous_pread = g->flash_prefill_pread_last_end_ms > 0.0;
    const bool same_layer_as_previous =
        have_previous_pread && g->flash_prefill_pread_last_layer == il;
    if (have_previous_pread && pread_t0_ms > g->flash_prefill_pread_last_end_ms) {
        issue_gap_ms = pread_t0_ms - g->flash_prefill_pread_last_end_ms;
        g->flash_prefill_pread_issue_gap_ms += issue_gap_ms;
        g->flash_prefill_pread_issue_gap_count++;
        if (issue_gap_ms > g->flash_prefill_pread_max_issue_gap_ms) {
            g->flash_prefill_pread_max_issue_gap_ms = issue_gap_ms;
        }
        if (issue_gap_ms > 1.0) g->flash_prefill_pread_issue_gap_gt_1ms++;
        if (issue_gap_ms > 5.0) g->flash_prefill_pread_issue_gap_gt_5ms++;
        if (issue_gap_ms > 20.0) g->flash_prefill_pread_issue_gap_gt_20ms++;
        if (same_layer_as_previous) {
            g->flash_prefill_pread_same_layer_issue_gap_ms += issue_gap_ms;
        } else {
            g->flash_prefill_pread_cross_layer_issue_gap_ms += issue_gap_ms;
        }
        ds4_prefill_trace_interval_ms("sidecar_pread_issue_gap",
                                      il,
                                      true_expert,
                                      g->flash_prefill_pread_last_end_ms,
                                      pread_t0_ms);
    }
    if (have_previous_pread && g->flash_prefill_stage_last_end_ms > 0.0 &&
        pread_t0_ms > g->flash_prefill_stage_last_end_ms) {
        post_gap_ms = pread_t0_ms - g->flash_prefill_stage_last_end_ms;
        g->flash_prefill_pread_post_stage_gap_ms += post_gap_ms;
        g->flash_prefill_pread_post_stage_gap_count++;
        if (post_gap_ms > g->flash_prefill_pread_max_post_gap_ms) {
            g->flash_prefill_pread_max_post_gap_ms = post_gap_ms;
        }
        if (post_gap_ms > 1.0) g->flash_prefill_pread_post_gap_gt_1ms++;
        if (post_gap_ms > 5.0) g->flash_prefill_pread_post_gap_gt_5ms++;
        if (post_gap_ms > 20.0) g->flash_prefill_pread_post_gap_gt_20ms++;
        if (same_layer_as_previous) {
            g->flash_prefill_pread_same_layer_post_gap_ms += post_gap_ms;
        } else {
            g->flash_prefill_pread_cross_layer_post_gap_ms += post_gap_ms;
        }
        ds4_prefill_trace_interval_ms("sidecar_pread_post_stage_gap",
                                      il,
                                      true_expert,
                                      g->flash_prefill_stage_last_end_ms,
                                      pread_t0_ms);
    }
    if (!metal_graph_flash_moe_read_record_to_buf(layer,
                                                  true_expert,
                                                  g->flash_install_buf,
                                                  flash_moe_prefill_io_split())) {
        fprintf(stderr,
                "ds4: Flash-MoE failed to read prefill layer %u expert %d: %s\n",
                il,
                true_expert,
                strerror(errno));
        return false;
    }
    const double pread_t1 = now_sec();
    const double pread_t1_ms = pread_t1 * 1000.0;
    ds4_prefill_trace_interval_ms("sidecar_pread",
                                  il,
                                  true_expert,
                                  pread_t0_ms,
                                  pread_t1_ms);

    const uint64_t gate_bytes = layer->family_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_bytes   = layer->family_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_bytes = layer->family_bytes[DS4_FLASH_FAMILY_DOWN];

    const double gate_t0 = now_sec();
    const bool gate_ok =
        ds4_gpu_tensor_write(gate_bank, 0,
                             g->flash_install_buf + layer->family_offset[DS4_FLASH_FAMILY_GATE],
                             gate_bytes) != 0;
    const double gate_t1 = now_sec();
    const bool up_ok = gate_ok &&
        ds4_gpu_tensor_write(up_bank, 0,
                             g->flash_install_buf + layer->family_offset[DS4_FLASH_FAMILY_UP],
                             up_bytes) != 0;
    const double up_t1 = now_sec();
    const bool down_ok = up_ok &&
        ds4_gpu_tensor_write(down_bank, 0,
                             g->flash_install_buf + layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                             down_bytes) != 0;
    const double down_t1 = now_sec();
    const bool ok = gate_ok && up_ok && down_ok;

    if (!ok) {
        fprintf(stderr, "ds4: Flash-MoE failed to stage prefill layer %u expert %d (bank %d)\n",
                il, true_expert, bank_set);
        return false;
    }

    g->flash_misses++;
    g->flash_installed_bytes += layer->expert_stride;
    g->flash_prefill_stage_calls++;
    g->flash_prefill_stage_bytes += layer->expert_stride;
    g->flash_prefill_stage_pread_ms += (pread_t1 - pread_t0) * 1000.0;
    g->flash_prefill_stage_upload_gate_ms += (gate_t1 - gate_t0) * 1000.0;
    g->flash_prefill_stage_upload_up_ms += (up_t1 - gate_t1) * 1000.0;
    g->flash_prefill_stage_upload_down_ms += (down_t1 - up_t1) * 1000.0;
    g->flash_prefill_stage_upload_ms += (down_t1 - gate_t0) * 1000.0;
    if (g->flash_prefill_pread_first_start_ms == 0.0) {
        g->flash_prefill_pread_first_start_ms = pread_t0_ms;
    }
    g->flash_prefill_pread_last_end_ms = pread_t1_ms;
    g->flash_prefill_stage_last_end_ms = down_t1 * 1000.0;
    g->flash_prefill_pread_last_layer = il;
    {
        const uint32_t bucket = ds4_prefill_stage_ref_bucket(refs);
        g->flash_prefill_stage_bucket_calls[bucket]++;
        g->flash_prefill_stage_bucket_refs[bucket] += refs;
        g->flash_prefill_stage_bucket_bytes[bucket] += layer->expert_stride;
        g->flash_prefill_stage_bucket_pread_ms[bucket] += (pread_t1 - pread_t0) * 1000.0;
        g->flash_prefill_stage_bucket_issue_gap_ms[bucket] += issue_gap_ms;
        g->flash_prefill_stage_bucket_post_gap_ms[bucket] += post_gap_ms;
    }
    ds4_prefill_trace_interval_ms("metal_upload",
                                  il,
                                  true_expert,
                                  gate_t0 * 1000.0,
                                  down_t1 * 1000.0);
    return true;
}

static bool metal_graph_flash_moe_prepare_decode(ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    if (!g || !g->flash_moe) return true;
    if (il >= DS4_N_LAYER || !g->router_slot_selected) return false;

    const bool profile =
        env_flag_enabled("DS4_FLASH_MOE_PROFILE") &&
        !backend_diagnostic_logs_suppressed();
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;
    const bool gpu_l2_blits = g->flash_gpu_l2_slot_bank != 0;
    if (gpu_l2_blits && ds4_gpu_begin_commands() == 0) return false;

    int32_t true_ids[DS4_MAX_EXPERT_USED];
    int32_t slot_ids[DS4_MAX_EXPERT_USED];
    float weight_vals[DS4_MAX_EXPERT_USED];
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    bool protected_experts[DS4_MAX_EXPERT];
    memset(protected_experts, 0, sizeof(protected_experts));
    bool ok = ds4_gpu_tensor_read(g->router_selected,
                                  0,
                                  true_ids,
                                  (uint64_t)active_expert_used * sizeof(true_ids[0])) != 0;
    if (ok) {
        ok = ds4_gpu_tensor_read(g->router_weights,
                                 0,
                                 weight_vals,
                                 (uint64_t)active_expert_used * sizeof(weight_vals[0])) != 0;
    }
    if (ok) flash_moe_decode_trace_record(pos, il, true_ids);
    if (ok && flash_moe_preprotect_topk_enabled()) {
        metal_graph_flash_moe_protect_routed_experts(true_ids,
                                                     active_expert_used,
                                                     protected_experts);
    }
    const uint64_t miss_before = g->flash_misses;
    const bool six_slot_baseline =
        flash_moe_six_slot_baseline_enabled() &&
        !g->flash_direct_mmap_bank &&
        !g->flash_per_expert_buffers &&
        g->flash_slot_bank >= active_expert_used;
    if (g->flash_direct_mmap_bank) {
        for (uint32_t k = 0; ok && k < active_expert_used; k++) {
            const int32_t true_expert = true_ids[k];
            if (true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
                true_expert >= (int32_t)g->flash_slot_bank) {
                ok = false;
                break;
            }
            slot_ids[k] = true_expert;
        }
        if (ok) g->flash_hits += active_expert_used;
    } else if (six_slot_baseline) {
        int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
        const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
        /* This diagnostic mode reloads every routed expert into a fixed
         * destination; it does not consume cached hits. Reject invalid IDs
         * before writing and invalidate each destination before any read. */
        for (uint32_t k = 0; ok && k < active_expert_used; k++) {
            ok = true_ids[k] >= 0 && true_ids[k] < (int32_t)DS4_N_EXPERT;
        }
        for (uint32_t k = 0; ok && k < active_expert_used; k++) {
            const int32_t true_expert = true_ids[k];
            const int32_t slot = (int32_t)k;
            if (true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
                ok = false;
                break;
            }
            const int32_t evicted = slot_to_expert[slot];
            metal_graph_flash_moe_invalidate_decode_slot(g, il, slot);
            bool wrote = false;
            bool direct_metal_write = false;
            if (flash_moe_direct_slot_pread_enabled()) {
                errno = 0;
                if (g->flash_mixed_slot_bank) {
                    uint8_t *dst = metal_graph_flash_moe_mixed_slot_ptr(g, il, slot);
                    if (dst) {
                        wrote = metal_graph_flash_moe_read_record_to_buf(layer,
                                                                          true_expert,
                                                                          dst,
                                                                          flash_moe_cache_io_split());
                        direct_metal_write = wrote;
                    }
                } else {
                    uint8_t *family_dst[DS4_FLASH_FAMILY_COUNT] = { NULL, NULL, NULL };
                    wrote = metal_graph_flash_moe_direct_slot_ptrs(g, il, slot, family_dst) &&
                            metal_graph_flash_moe_pread_slot_direct(layer,
                                                                    true_expert,
                                                                    family_dst,
                                                                    flash_moe_cache_io_split());
                    direct_metal_write = wrote;
                }
                if (!wrote && errno != 0) {
                    fprintf(stderr,
                            "ds4: Flash-MoE six-slot baseline failed direct-read layer %u expert %d: %s\n",
                            il,
                            true_expert,
                            strerror(errno));
                    ok = false;
                    break;
                }
            }
            if (!wrote) {
                if (!metal_graph_flash_moe_read_record_to_buf(layer,
                                                              true_expert,
                                                              g->flash_install_buf,
                                                              flash_moe_cache_io_split()) ||
                    !metal_graph_flash_moe_write_slot_from_buf(g,
                                                               il,
                                                               slot,
                                                               g->flash_install_buf)) {
                    fprintf(stderr,
                            "ds4: Flash-MoE six-slot baseline failed to load layer %u expert %d into slot %d\n",
                            il,
                            true_expert,
                            slot);
                    ok = false;
                    break;
                }
            }
            if (direct_metal_write &&
                !metal_graph_flash_moe_mark_slot_modified(g, il, slot)) {
                fprintf(stderr,
                        "ds4: Flash-MoE six-slot baseline failed to mark layer %u slot %d modified\n",
                        il,
                        slot);
                ok = false;
                break;
            }
            metal_graph_flash_moe_commit_decode_slot(g, il, true_expert, slot, evicted);
            slot_ids[k] = slot;
            g->flash_misses++;
            g->flash_installed_bytes += layer->expert_stride;
        }
    } else {
        if (ok) ok = metal_graph_flash_moe_install_request(g, il, true_ids,
                active_expert_used, protected_experts, slot_ids);
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(g->router_slot_selected,
                                  0,
                                  slot_ids,
                                  (uint64_t)active_expert_used * sizeof(slot_ids[0])) != 0;
    }
    if (ok) {
        metal_graph_flash_moe_record_decode_slots(g,
                                                  il,
                                                  true_ids,
                                                  slot_ids,
                                                  active_expert_used);
        for (uint32_t k = 0; k < active_expert_used; k++) {
            g->flash_decode_weights[il][k] = weight_vals[k];
        }
    }

    const double t_done = profile ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: Flash-MoE layer=%u sync=%.3f ms remap/install=%.3f ms misses=%u\n",
                il,
                (t_sync - t0) * 1000.0,
                (t_done - t_sync) * 1000.0,
                (uint32_t)(g->flash_misses - miss_before));
    }
    if (!ok) {
        if (gpu_l2_blits) (void)ds4_gpu_end_commands();
        return false;
    }
    return gpu_l2_blits ? true : ds4_gpu_begin_commands() != 0;
}

typedef struct ds4_flash_decode_async_load {
    int32_t expert;
    int32_t slot;
    int32_t evicted;
    bool thread_started;
    bool joined;
    bool uploaded;
    ds4_flash_decode_read_job job;
    pthread_t thread;
} ds4_flash_decode_async_load;

static void metal_graph_flash_moe_async_load_cleanup(
        ds4_gpu_graph *g,
        uint32_t il,
        ds4_flash_decode_async_load *loads,
        uint32_t                     n_loads) {
    if (!loads) return;
    for (uint32_t i = 0; i < n_loads; i++) {
        ds4_flash_decode_async_load *load = &loads[i];
        if (load->thread_started && !load->joined) {
            pthread_join(load->thread, NULL);
            load->joined = true;
        }
        if (!load->uploaded && (load->job.direct_record || load->job.direct_slot)) {
            metal_graph_flash_moe_invalidate_decode_slot(g, il, load->slot);
        }
        if (load->job.buf_owned) free(load->job.buf);
        load->job.buf = NULL;
        load->job.buf_owned = false;
    }
}

static bool metal_graph_flash_moe_async_scratch_ensure(
        ds4_gpu_graph *g,
        uint64_t       stride,
        uint32_t       slots) {
    if (!g) return false;
    if (slots == 0) return true;
    if (stride == 0) return false;
    if (slots > DS4_N_EXPERT_ACTIVE_USED) slots = DS4_N_EXPERT_ACTIVE_USED;
    if (g->flash_async_handout_scratch &&
        g->flash_async_handout_scratch_stride >= stride &&
        g->flash_async_handout_scratch_slots >= slots) {
        return true;
    }
    if (stride > SIZE_MAX / (uint64_t)slots) return false;
    free(g->flash_async_handout_scratch);
    g->flash_async_handout_scratch = NULL;
    g->flash_async_handout_scratch_stride = 0;
    g->flash_async_handout_scratch_slots = 0;
    g->flash_async_handout_scratch = xmalloc((size_t)(stride * (uint64_t)slots));
    g->flash_async_handout_scratch_stride = stride;
    g->flash_async_handout_scratch_slots = slots;
    return g->flash_async_handout_scratch != NULL;
}

static bool metal_graph_flash_moe_async_start_loads(
        ds4_gpu_graph                      *g,
        uint32_t                            il,
        const ds4_flash_moe_layer_sidecar  *sidecar_layer,
        ds4_flash_decode_async_load        *loads,
        uint32_t                            n_loads,
        bool                                direct_slot_reads,
        int                                 io_split) {
    if (!g || !sidecar_layer || !loads) return false;
    if (n_loads == 0) return true;
    const bool allow_direct =
        direct_slot_reads && flash_moe_direct_slot_pread_enabled();
    uint32_t scratch_needed = 0;
    for (uint32_t li = 0; li < n_loads; li++) {
        ds4_flash_decode_async_load *load = &loads[li];
        ds4_flash_decode_read_job *job = &load->job;
        if (allow_direct) {
            if (g->flash_mixed_slot_bank) {
                uint8_t *dst = metal_graph_flash_moe_mixed_slot_ptr(g, il, load->slot);
                if (dst) {
                    job->direct_record = true;
                    job->record_dst = dst;
                }
            } else {
                uint8_t *family_dst[DS4_FLASH_FAMILY_COUNT] = { NULL, NULL, NULL };
                if (metal_graph_flash_moe_direct_slot_ptrs(g, il, load->slot, family_dst)) {
                    job->direct_slot = true;
                    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
                        job->family_dst[fam] = family_dst[fam];
                    }
                }
            }
        }
        if (!job->direct_record && !job->direct_slot && !job->memory_src) scratch_needed++;
    }
    if (scratch_needed > 0 &&
        !metal_graph_flash_moe_async_scratch_ensure(g,
                                                    sidecar_layer->expert_stride,
                                                    n_loads)) {
        return false;
    }
    for (uint32_t li = 0; li < n_loads; li++) {
        ds4_flash_decode_async_load *load = &loads[li];
        ds4_flash_decode_read_job *job = &load->job;
        if (!job->direct_record && !job->direct_slot && !job->memory_src) {
            job->buf = g->flash_async_handout_scratch +
                       (uint64_t)li * g->flash_async_handout_scratch_stride;
            job->buf_owned = false;
        }
        job->io_split = io_split;
        g->flash_decode_prefetch_loads++;
        g->flash_decode_prefetch_bytes += sidecar_layer->expert_stride;
        if (pthread_create(&load->thread,
                           NULL,
                           ds4_flash_decode_read_thread,
                           job) == 0) {
            load->thread_started = true;
        } else {
            ds4_flash_decode_read_thread(job);
            load->joined = true;
        }
    }
    return true;
}

static bool metal_graph_flash_moe_async_load_join_upload(
        ds4_gpu_graph                *g,
        uint32_t                      il,
        ds4_flash_decode_async_load  *load) {
    if (!g || !load) return false;
    if (load->uploaded) return true;
    ds4_flash_decode_read_job *job = &load->job;
    if (load->thread_started && !load->joined) {
        if (ds4_flash_decode_read_job_is_complete(job)) {
            g->flash_async_handout_join_ready++;
        } else {
            g->flash_async_handout_join_wait++;
        }
        pthread_join(load->thread, NULL);
        load->joined = true;
    }
    g->flash_decode_prefetch_pread_ms += job->pread_t1_ms - job->pread_t0_ms;
    if (job->err) {
        fprintf(stderr,
                "ds4: Flash-MoE async handout pread failed layer %u expert %d: %s\n",
                il,
                load->expert,
                strerror(job->err));
        return false;
    }
    if (job->direct_record || job->direct_slot) {
        if (!metal_graph_flash_moe_mark_slot_modified(g, il, load->slot)) {
            return false;
        }
        if (g->flash_gpu_l2_slot_bank) {
            (void)metal_graph_flash_moe_gpu_l2_store_l1_slot(g,
                                                             il,
                                                             load->expert,
                                                             load->slot);
        }
        metal_graph_flash_moe_commit_decode_slot(g,
                                                 il,
                                                 load->expert,
                                                 load->slot,
                                                 load->evicted);
        load->uploaded = true;
        return true;
    }
    if (!job->buf) return false;
    if (!metal_graph_flash_moe_upload_decode_slot(g,
                                                  il,
                                                  load->expert,
                                                  load->slot,
                                                  load->evicted,
                                                  job->buf)) {
        return false;
    }
    load->uploaded = true;
    return true;
}

/* Complete-request asynchronous installation, used by native HY4 while the
 * independent shared FFN runs. The caller must join prior bank users before
 * begin, and finish (including on encode failure) before reusing this state.
 * L2/identity/diagnostic modes retain their existing installation path. */
typedef struct {
    uint32_t n_ids, n_loads;
    int32_t ids[DS4_MAX_EXPERT_USED], slots[DS4_MAX_EXPERT_USED];
    int32_t load_index[DS4_MAX_EXPERT_USED];
    ds4_flash_decode_async_load loads[DS4_MAX_EXPERT_USED];
} ds4_flash_moe_request_loads;

static bool metal_graph_flash_moe_request_loads_supported(const ds4_gpu_graph *g, uint32_t il) {
    return g && g->flash_moe && il < DS4_N_LAYER &&
        g->flash_moe->layer[il].expert_stride > 0 &&
        !g->flash_moe->layer[il].family_major && g->flash_mixed_slot_bank &&
        !g->flash_direct_mmap_bank && !g->flash_per_expert_buffers &&
        !g->flash_per_slot_buffers && !g->flash_chunked_mixed_bank &&
        !g->flash_gpu_l2_slot_bank && !g->flash_shared_l2_slot_bank &&
        !g->flash_l2_slot_bank && !flash_moe_six_slot_baseline_enabled();
}

static bool metal_graph_flash_moe_request_loads_begin(ds4_gpu_graph *g,
        uint32_t il, const int32_t *ids, uint32_t n_ids,
        ds4_flash_moe_request_loads *request) {
    if (!request) return false;
    memset(request, 0, sizeof(*request));
    if (!metal_graph_flash_moe_request_loads_supported(g,il) || il >= DS4_N_LAYER ||
        !ids || !n_ids || n_ids > DS4_N_EXPERT_ACTIVE_USED ||
        n_ids > DS4_MAX_EXPERT_USED) return false;
    int32_t evicted[DS4_MAX_EXPERT_USED];
    bool misses[DS4_MAX_EXPERT_USED], protected_experts[DS4_MAX_EXPERT] = {false};
    if (flash_moe_preprotect_topk_enabled())
        metal_graph_flash_moe_protect_routed_experts(ids,n_ids,protected_experts);
    if (!metal_graph_flash_moe_resolve_request(g,il,ids,n_ids,protected_experts,
            request->slots,evicted,misses)) return false;
    request->n_ids=n_ids;
    memcpy(request->ids,ids,n_ids*sizeof(*ids));
    const ds4_flash_moe_layer_sidecar *layer=&g->flash_moe->layer[il];
    for (uint32_t k=0;k<n_ids;k++) {
        request->load_index[k]=-1;
        bool duplicate=false;
        for(uint32_t prev=0;prev<k;prev++) {
            if(ids[prev]==ids[k]) { duplicate=true; break; }
        }
        if(duplicate || !misses[k]) continue;
        uint32_t li=request->n_loads++;
        request->load_index[k]=(int32_t)li;
        ds4_flash_decode_async_load *load=&request->loads[li];
        load->expert=ids[k]; load->slot=request->slots[k]; load->evicted=evicted[k];
        if(!ds4_flash_decode_read_job_set_sidecar(&load->job,layer,ids[k],
                flash_moe_cache_io_split())) return false;
    }
    /* Resolve ALL hits and misses before the first destination is invalidated
     * or written. No mapping may describe an in-flight direct write. */
    for(uint32_t li=0;li<request->n_loads;li++)
        metal_graph_flash_moe_invalidate_decode_slot(g,il,request->loads[li].slot);
    if(!metal_graph_flash_moe_async_start_loads(g,il,layer,request->loads,
            request->n_loads,true,flash_moe_cache_io_split())) {
        metal_graph_flash_moe_async_load_cleanup(g,il,request->loads,request->n_loads);
        return false;
    }
    return true;
}

static bool metal_graph_flash_moe_request_loads_finish(ds4_gpu_graph *g,
        uint32_t il, ds4_flash_moe_request_loads *request, bool install) {
    if(!g || !request) return false;
    bool ok=install;
    /* Commit and touch in original route order, regardless of read completion
     * order. This preserves the synchronous path's LRU ages and hit counts. */
    for(uint32_t k=0;ok && k<request->n_ids;k++) {
        bool duplicate=false;
        for(uint32_t prev=0;prev<k;prev++)
            if(request->ids[prev]==request->ids[k]) { duplicate=true; break; }
        if(duplicate) continue;
        int32_t li=request->load_index[k];
        metal_graph_flash_moe_note_request_slot(g,il,request->ids[k],request->slots[k],li>=0);
        if(li>=0) {
            ds4_flash_decode_async_load *load=&request->loads[li];
            ok=metal_graph_flash_moe_async_load_join_upload(g,il,load);
            // Scratch upload already accounts bytes; direct upload does not.
            if(ok && (load->job.direct_record || load->job.direct_slot))
                g->flash_installed_bytes+=g->flash_moe->layer[il].expert_stride;
        }
    }
    // Always join every reader, even after an earlier I/O/encode/upload error.
    metal_graph_flash_moe_async_load_cleanup(g,il,request->loads,request->n_loads);
    if(!ok) g->flash_decode_ids_valid[il]=0;
    return ok;
}

static bool metal_graph_flash_moe_async_loads_complete(
        const ds4_flash_decode_async_load *loads,
        uint32_t                           n_loads) {
    if (!loads) return false;
    for (uint32_t i = 0; i < n_loads; i++) {
        if (!ds4_flash_decode_read_job_is_complete(&loads[i].job)) return false;
    }
    return true;
}

static bool metal_graph_flash_moe_identity_gpu_selected_active(
        const ds4_gpu_graph *g,
        uint32_t             il) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER ||
        env_flag_enabled("DS4_FLASH_MOE_DISABLE_IDENTITY_GPU_SELECTED") ||
        !flash_moe_preload_slot_bank_enabled() ||
        g->flash_direct_mmap_bank ||
        g->flash_per_expert_buffers ||
        g->flash_per_slot_buffers ||
        g->flash_chunked_mixed_bank ||
        !g->flash_mixed_slot_bank ||
        g->flash_slot_bank < DS4_N_EXPERT ||
        !g->router_selected ||
        !g->router_weights ||
        !g->flash_gate_bank[il] ||
        !g->flash_up_bank[il] ||
        !g->flash_down_bank[il]) {
        return false;
    }
    return metal_graph_flash_moe_sidecar_layer_has_records(&g->flash_moe->layer[il]);
}

static bool metal_graph_flash_moe_compute_grouped_banked(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_slot_stride,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_slot_stride,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        uint32_t                   active_expert_used) {
    if (!g || !layer || il >= DS4_N_LAYER) return false;
    const ds4_flash_moe_layer_sidecar *flash_layer =
        g->flash_moe ? &g->flash_moe->layer[il] : NULL;
    if (!metal_graph_flash_moe_sidecar_layer_has_records(flash_layer)) return false;
    const ds4_gpu_tensor *selected =
        metal_graph_flash_moe_identity_gpu_selected_active(g, il) ?
        g->router_selected : g->router_slot_selected;
    return ds4_gpu_routed_moe_one_banked_tensor(g->routed_out,
                                                g->routed_gate,
                                                g->routed_up,
                                                g->routed_mid,
                                                g->routed_down,
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
                                                selected,
                                                g->router_weights,
                                                active_expert_used,
                                                DS4_SWIGLU_CLAMP_EXP,
	                                                g->ffn_norm) != 0;
}

static bool metal_graph_flash_moe_compute_active_staged(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        uint32_t                   active_expert_used) {
    if (!g || !layer || !g->flash_moe || il >= DS4_N_LAYER ||
        !g->flash_per_slot_buffers ||
        !flash_moe_active_staging_enabled() ||
        active_expert_used != DS4_N_EXPERT_ACTIVE_USED ||
        active_expert_used == 0 ||
        active_expert_used > DS4_MAX_EXPERT_USED ||
        !g->flash_decode_ids_valid[il] ||
        !g->flash_stage_mixed_bank[il] ||
        !g->flash_stage_gate_bank[il] ||
        !g->flash_stage_up_bank[il] ||
        !g->flash_stage_down_bank[il] ||
        !g->flash_stage_slot_selected) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    if (flash_moe_layer_all_mxfp4_plane_split(flash_layer) ||
        flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(flash_layer)) {
        return false;
    }
    for (uint32_t k = 0; k < active_expert_used; k++) {
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        if (slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
            slot >= (int32_t)DS4_MAX_EXPERT) {
            return false;
        }
        if (!metal_graph_flash_moe_ensure_per_slot_buffer(g, il, slot)) {
            return false;
        }
        if (!g->flash_expert_bank[il][slot]) return false;
        const uint64_t dst_off = (uint64_t)k * flash_layer->expert_stride;
        if (ds4_gpu_tensor_copy(g->flash_stage_mixed_bank[il],
                                dst_off,
                                g->flash_expert_bank[il][slot],
                                0,
                                flash_layer->expert_stride) == 0) {
            return false;
        }
    }

    return ds4_gpu_routed_moe_one_banked_tensor(g->routed_out,
                                                g->routed_gate,
                                                g->routed_up,
                                                g->routed_mid,
                                                g->routed_down,
                                                g->flash_stage_gate_bank[il],
                                                g->flash_stage_up_bank[il],
                                                g->flash_stage_down_bank[il],
                                                active_expert_used,
                                                layer->ffn_gate_exps->type,
                                                layer->ffn_down_exps->type,
                                                gate_expert_bytes,
                                                flash_layer->expert_stride,
                                                gate_row_bytes,
                                                down_expert_bytes,
                                                flash_layer->expert_stride,
                                                down_row_bytes,
                                                expert_in_dim,
                                                expert_mid_dim,
                                                out_dim,
                                                g->flash_stage_slot_selected,
                                                g->router_weights,
                                                active_expert_used,
                                                DS4_SWIGLU_CLAMP_EXP,
                                                g->ffn_norm) != 0;
}

static bool metal_graph_flash_moe_compute_chunked_mixed(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        uint32_t                   active_expert_used) {
    if (!g || !layer || !g->flash_moe || il >= DS4_N_LAYER ||
        !g->flash_chunked_mixed_bank ||
        g->flash_chunk_count == 0 ||
        g->flash_chunk_count > DS4_FLASH_MOE_MAX_CHUNKS ||
        active_expert_used == 0 ||
        active_expert_used > DS4_MAX_EXPERT_USED ||
        !g->flash_decode_ids_valid[il] ||
        !g->flash_chunk_partial_out) {
        return false;
    }

    int32_t local_ids[DS4_FLASH_MOE_MAX_CHUNKS][DS4_MAX_EXPERT_USED];
    float local_weights[DS4_FLASH_MOE_MAX_CHUNKS][DS4_MAX_EXPERT_USED];
    uint32_t counts[DS4_FLASH_MOE_MAX_CHUNKS];
    memset(local_ids, 0, sizeof(local_ids));
    memset(local_weights, 0, sizeof(local_weights));
    memset(counts, 0, sizeof(counts));

    for (uint32_t k = 0; k < active_expert_used; k++) {
        uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
            return false;
        }
        (void)chunk_slots;
        if (counts[chunk] >= DS4_MAX_EXPERT_USED) return false;
        const uint32_t dst = counts[chunk]++;
        local_ids[chunk][dst] = (int32_t)local_slot;
        local_weights[chunk][dst] = g->flash_decode_weights[il][k];
    }

    bool have_output = false;
    bool ok = true;
    for (uint32_t chunk = 0; ok && chunk < g->flash_chunk_count; chunk++) {
        const uint32_t n = counts[chunk];
        if (n == 0) continue;
        uint32_t first_slot = chunk * g->flash_chunk_slots;
        uint32_t chunk_slots = g->flash_slot_bank - first_slot;
        if (chunk_slots > g->flash_chunk_slots) chunk_slots = g->flash_chunk_slots;
        if (chunk_slots == 0 ||
            !g->flash_chunk_gate_bank[il][chunk] ||
            !g->flash_chunk_up_bank[il][chunk] ||
            !g->flash_chunk_down_bank[il][chunk] ||
            !g->flash_chunk_slot_selected[il][chunk] ||
            !g->flash_chunk_weights[il][chunk]) {
            return false;
        }
        ok = ds4_gpu_tensor_write(g->flash_chunk_slot_selected[il][chunk],
                                  0,
                                  local_ids[chunk],
                                  (uint64_t)n * sizeof(local_ids[chunk][0])) != 0 &&
             ds4_gpu_tensor_write(g->flash_chunk_weights[il][chunk],
                                  0,
                                  local_weights[chunk],
                                  (uint64_t)n * sizeof(local_weights[chunk][0])) != 0;
        if (!ok) break;

        ds4_gpu_tensor *chunk_out = have_output ? g->flash_chunk_partial_out : g->routed_out;
        ok = ds4_gpu_routed_moe_one_banked_tensor(chunk_out,
                                                  g->routed_gate,
                                                  g->routed_up,
                                                  g->routed_mid,
                                                  g->routed_down,
                                                  g->flash_chunk_gate_bank[il][chunk],
                                                  g->flash_chunk_up_bank[il][chunk],
                                                  g->flash_chunk_down_bank[il][chunk],
                                                  chunk_slots,
                                                  layer->ffn_gate_exps->type,
                                                  layer->ffn_down_exps->type,
                                                  gate_expert_bytes,
                                                  g->flash_moe->layer[il].expert_stride,
                                                  gate_row_bytes,
                                                  down_expert_bytes,
                                                  g->flash_moe->layer[il].expert_stride,
                                                  down_row_bytes,
                                                  expert_in_dim,
                                                  expert_mid_dim,
                                                  out_dim,
                                                  g->flash_chunk_slot_selected[il][chunk],
                                                  g->flash_chunk_weights[il][chunk],
                                                  n,
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

    return ok && have_output;
}

static bool flash_moe_independent_slots6_grouped_enabled(void) {
    if (env_flag_enabled("DS4_FLASH_MOE_DISABLE_SLOTS6_GROUPED") ||
        env_flag_enabled("DS4_FLASH_MOE_FORCE_PER_ROUTE")) {
        return false;
    }
    const char *explicit_enable = getenv("DS4_FLASH_MOE_ENABLE_SLOTS6_GROUPED");
    if (explicit_enable && explicit_enable[0] && atoi(explicit_enable) == 0) {
        return false;
    }
    return true;
}

/* MXFP4/record-table direct-mmap policy and slots6 decode helpers. */
#include "ssd_flash_moe_mxfp4_slots6.c"

static bool metal_graph_flash_moe_compute_independent_slots6_grouped(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint32_t                   active_expert_used,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim) {
    if (!g || !layer || !g->flash_moe || il >= DS4_N_LAYER ||
        active_expert_used != DS4_N_EXPERT_ACTIVE_USED ||
        active_expert_used != 6 ||
        !flash_moe_independent_slots6_grouped_enabled()) {
        return false;
    }
    const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
    const bool q2_slots6_path =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    const bool hybrid_slots6_path =
        flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(flash_layer) &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    const bool mxfp4_slots6_path =
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    if (!q2_slots6_path && !hybrid_slots6_path && !mxfp4_slots6_path) {
        return false;
    }
    if (!g->flash_decode_ids_valid[il]) return false;

    if (metal_graph_flash_moe_direct_mmap_slots6_active(g)) {
        if (mxfp4_slots6_path &&
            metal_graph_flash_moe_direct_mmap_record_slots6_active(g)) {
            return metal_graph_flash_moe_mxfp4_direct_mmap_record_slots6(
                       g,
                       layer,
                       il,
                       active_expert_used,
                       gate_row_bytes,
                       down_row_bytes,
                       expert_in_dim,
                       expert_mid_dim,
                       out_dim);
        }

        ds4_gpu_tensor *gate_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
        ds4_gpu_tensor *up_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
        ds4_gpu_tensor *down_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
        bool ok = true;
        for (uint32_t k = 0; ok && k < active_expert_used; k++) {
            const int32_t expert = g->flash_decode_true_ids[il][k];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
                ok = false;
                break;
            }
            ok = metal_graph_flash_moe_ensure_direct_mmap_family_views(g,
                                                                       il,
                                                                       (uint32_t)expert);
            if (!ok) break;
            gate_slots[k] = g->flash_expert_gate_view[il][expert];
            up_slots[k] = g->flash_expert_up_view[il][expert];
            down_slots[k] = g->flash_expert_down_view[il][expert];
            ok = gate_slots[k] && up_slots[k] && down_slots[k];
        }
        if (ok) {
            static int logged_direct_mmap_slots6 = 0;
            if (!logged_direct_mmap_slots6) {
                fprintf(stderr,
                        "ds4: Flash-MoE using direct mmap slots6 decode path "
                        "(cached active file-backed expert views, no slot copies)\n");
                logged_direct_mmap_slots6 = 1;
            }
            ok = ds4_gpu_routed_moe_one_slots6_tensor(g->routed_out,
                                                      g->routed_gate,
                                                      g->routed_up,
                                                      g->routed_mid,
                                                      g->routed_down,
                                                      gate_slots,
                                                      up_slots,
                                                      down_slots,
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
        return ok;
    }

    if (mxfp4_slots6_path &&
        g->flash_chunked_mixed_bank &&
        flash_moe_chunked_bank_slots6_enabled()) {
        return metal_graph_flash_moe_mxfp4_chunked_slots6(
                   g,
                   layer,
                   il,
                   active_expert_used,
                   gate_row_bytes,
                   down_row_bytes,
                   expert_in_dim,
                   expert_mid_dim,
                   out_dim);
    }

    if (mxfp4_slots6_path &&
        g->flash_per_slot_buffers &&
        flash_moe_record_table_enabled() &&
        g->flash_record_table[il]) {
        return metal_graph_flash_moe_mxfp4_record_table_slots6(
                   g,
                   layer,
                   il,
                   active_expert_used,
                   gate_row_bytes,
                   down_row_bytes,
                   expert_in_dim,
                   expert_mid_dim,
                   out_dim);
    }

    if (mxfp4_slots6_path &&
        (g->flash_per_expert_buffers || g->flash_per_slot_buffers) &&
        env_flag_enabled("DS4_FLASH_MOE_SLOTS6_RECORD_BUFFERS")) {
        return metal_graph_flash_moe_mxfp4_record_buffer_slots6(
                   g,
                   layer,
                   il,
                   active_expert_used,
                   gate_row_bytes,
                   down_row_bytes,
                   expert_in_dim,
                   expert_mid_dim,
                   out_dim);
    }

    ds4_gpu_tensor *gate_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
    ds4_gpu_tensor *up_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
    ds4_gpu_tensor *down_slots[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
    const bool mixed_cached_slot_views =
        !g->flash_per_expert_buffers &&
        !g->flash_per_slot_buffers &&
        g->flash_mixed_slot_bank &&
        !g->flash_chunked_mixed_bank &&
        g->flash_slot_bank <= DS4_MAX_EXPERT;
    const bool use_fresh_views =
        env_flag_enabled("DS4_FLASH_MOE_SLOTS6_FRESH_VIEWS") ||
        (!g->flash_per_expert_buffers &&
         !g->flash_per_slot_buffers &&
         !mixed_cached_slot_views);
    bool ok = true;
    for (uint32_t k = 0; ok && k < active_expert_used; k++) {
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        gate_slots[k] = use_fresh_views ?
            metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_GATE, slot) :
            metal_graph_flash_moe_family_slot_cached_view(g, il, DS4_FLASH_FAMILY_GATE, slot);
        up_slots[k] = use_fresh_views ?
            metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_UP, slot) :
            metal_graph_flash_moe_family_slot_cached_view(g, il, DS4_FLASH_FAMILY_UP, slot);
        down_slots[k] = use_fresh_views ?
            metal_graph_flash_moe_family_slot_view(g, il, DS4_FLASH_FAMILY_DOWN, slot) :
            metal_graph_flash_moe_family_slot_cached_view(g, il, DS4_FLASH_FAMILY_DOWN, slot);
        ok = gate_slots[k] && up_slots[k] && down_slots[k];
    }

    if (ok) {
        static int logged_slots6 = 0;
        if (!logged_slots6) {
            fprintf(stderr,
                    "ds4: Flash-MoE using grouped slots6 decode path "
                    "(direct 6-buffer gate/up and grouped down)\n");
            logged_slots6 = 1;
        }
        ok = ds4_gpu_routed_moe_one_slots6_tensor(g->routed_out,
                                                  g->routed_gate,
                                                  g->routed_up,
                                                  g->routed_mid,
                                                  g->routed_down,
                                                  gate_slots,
                                                  up_slots,
                                                  down_slots,
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

    if (use_fresh_views) {
        for (uint32_t k = 0; k < active_expert_used; k++) {
            ds4_gpu_tensor_free(down_slots[k]);
            ds4_gpu_tensor_free(up_slots[k]);
            ds4_gpu_tensor_free(gate_slots[k]);
        }
    }

    return ok;
}

static bool metal_graph_flash_moe_compute_route_to_down(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint32_t                   route,
        int32_t                    slot,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_slot_stride,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_slot_stride,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim) {
    if (!g || !layer || route >= DS4_N_EXPERT_ACTIVE_USED || slot < 0) return false;
    const bool independent_slot_buffer =
        g->flash_per_expert_buffers || g->flash_per_slot_buffers;
    const bool per_expert_debug =
        independent_slot_buffer &&
        (env_flag_enabled("DS4_FLASH_MOE_PER_EXPERT_DEBUG") ||
         env_flag_enabled("DS4_DEBUG_RESUME"));
    const uint64_t mid_row_bytes = (uint64_t)expert_mid_dim * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(g->routed_gate,
                                                    (uint64_t)route * mid_row_bytes,
                                                    mid_row_bytes);
    ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(g->routed_up,
                                                  (uint64_t)route * mid_row_bytes,
                                                  mid_row_bytes);
    ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(g->routed_mid,
                                                   (uint64_t)route * mid_row_bytes,
                                                   mid_row_bytes);
    ds4_gpu_tensor *down_view = ds4_gpu_tensor_view(g->routed_down,
                                                    (uint64_t)route * out_row_bytes,
                                                    out_row_bytes);
    ds4_gpu_tensor *weight_view = ds4_gpu_tensor_view(g->router_weights,
                                                      (uint64_t)route * sizeof(float),
                                                      sizeof(float));
    ds4_gpu_tensor *route_gate_bank = g->flash_gate_bank[il];
    ds4_gpu_tensor *route_up_bank = g->flash_up_bank[il];
    ds4_gpu_tensor *route_down_bank = g->flash_down_bank[il];
    uint32_t route_n_slots = g->flash_slot_bank;
    uint32_t route_layer_index = il;
    uint64_t route_gate_slot_stride = gate_slot_stride;
    uint64_t route_down_slot_stride = down_slot_stride;
    int32_t one_slot[1] = { slot };
    ds4_gpu_tensor *expert_gate_view = NULL;
    ds4_gpu_tensor *expert_up_view = NULL;
    ds4_gpu_tensor *expert_down_view = NULL;
    if (independent_slot_buffer) {
        const uint32_t slot_limit =
            g->flash_per_expert_buffers ? DS4_N_EXPERT : g->flash_slot_bank;
        if (slot >= (int32_t)slot_limit || !g->flash_expert_bank[il][slot] ||
            !g->flash_moe) {
            if (per_expert_debug) {
                fprintf(stderr,
                        "ds4: Flash-MoE independent-slot route invalid layer=%u route=%u slot=%d has_graph=%d has_bank=%d\n",
                        il,
                        route,
                        slot,
                        g->flash_moe != NULL,
                        slot >= 0 && slot < (int32_t)slot_limit &&
                        g->flash_expert_bank[il][slot] != NULL);
            }
            ds4_gpu_tensor_free(weight_view);
            ds4_gpu_tensor_free(down_view);
            ds4_gpu_tensor_free(mid_view);
            ds4_gpu_tensor_free(up_view);
            ds4_gpu_tensor_free(gate_view);
            return false;
        }
        const ds4_flash_moe_layer_sidecar *flash_layer = &g->flash_moe->layer[il];
        ds4_gpu_tensor *expert = g->flash_expert_bank[il][slot];
        expert_gate_view = ds4_gpu_tensor_view(
                expert,
                flash_layer->family_offset[DS4_FLASH_FAMILY_GATE],
                flash_layer->family_bytes[DS4_FLASH_FAMILY_GATE]);
        expert_up_view = ds4_gpu_tensor_view(
                expert,
                flash_layer->family_offset[DS4_FLASH_FAMILY_UP],
                flash_layer->family_bytes[DS4_FLASH_FAMILY_UP]);
        expert_down_view = ds4_gpu_tensor_view(
                expert,
                flash_layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                flash_layer->family_bytes[DS4_FLASH_FAMILY_DOWN]);
        metal_graph_flash_moe_tag_layer_family_storage(g,
                                                       il,
                                                       expert_gate_view,
                                                       expert_up_view,
                                                       expert_down_view);
        route_gate_bank = expert_gate_view;
        route_up_bank = expert_up_view;
        route_down_bank = expert_down_view;
        route_n_slots = 1;
        route_layer_index = UINT32_MAX;
        route_gate_slot_stride = gate_expert_bytes;
        route_down_slot_stride = down_expert_bytes;
        one_slot[0] = 0;
    }
    const int ok =
        gate_view && up_view && mid_view && down_view && weight_view &&
        route_gate_bank && route_up_bank && route_down_bank &&
        ds4_gpu_routed_moe_one_banked_tensor_slotwise_baked(down_view,
                                                            gate_view,
                                                            up_view,
                                                            mid_view,
                                                            g->routed_down,
                                                            route_gate_bank,
                                                            route_up_bank,
                                                            route_down_bank,
                                                            route_n_slots,
                                                            layer->ffn_gate_exps->type,
                                                            layer->ffn_down_exps->type,
                                                            gate_expert_bytes,
                                                            route_gate_slot_stride,
                                                            gate_row_bytes,
                                                            down_expert_bytes,
                                                            route_down_slot_stride,
                                                            down_row_bytes,
                                                            expert_in_dim,
                                                            expert_mid_dim,
                                                            out_dim,
                                                            route_layer_index,
                                                            one_slot,
                                                            weight_view,
                                                            1,
                                                            DS4_SWIGLU_CLAMP_EXP,
                                                            g->ffn_norm) != 0;
    if (!ok && per_expert_debug) {
        const uint64_t gate_view_bytes = expert_gate_view ? ds4_gpu_tensor_bytes(expert_gate_view) : 0;
        const uint64_t up_view_bytes = expert_up_view ? ds4_gpu_tensor_bytes(expert_up_view) : 0;
        const uint64_t down_view_bytes = expert_down_view ? ds4_gpu_tensor_bytes(expert_down_view) : 0;
        fprintf(stderr,
                "ds4: Flash-MoE per-expert route compute failed layer=%u route=%u slot=%d "
                "gate_view=%d up_view=%d mid_view=%d down_view=%d weight_view=%d "
                "bank_gate=%d bank_up=%d bank_down=%d gate_bytes=%llu/%llu up_bytes=%llu down_bytes=%llu/%llu "
                "dims in=%u mid=%u out=%u types gate=%u down=%u strides gate=%llu down=%llu\n",
                il,
                route,
                slot,
                gate_view != NULL,
                up_view != NULL,
                mid_view != NULL,
                down_view != NULL,
                weight_view != NULL,
                route_gate_bank != NULL,
                route_up_bank != NULL,
                route_down_bank != NULL,
                (unsigned long long)gate_view_bytes,
                (unsigned long long)gate_expert_bytes,
                (unsigned long long)up_view_bytes,
                (unsigned long long)down_view_bytes,
                (unsigned long long)down_expert_bytes,
                expert_in_dim,
                expert_mid_dim,
                out_dim,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                (unsigned long long)route_gate_slot_stride,
                (unsigned long long)route_down_slot_stride);
    }
    ds4_gpu_tensor_free(expert_down_view);
    ds4_gpu_tensor_free(expert_up_view);
    ds4_gpu_tensor_free(expert_gate_view);
    ds4_gpu_tensor_free(weight_view);
    ds4_gpu_tensor_free(down_view);
    ds4_gpu_tensor_free(mid_view);
    ds4_gpu_tensor_free(up_view);
    ds4_gpu_tensor_free(gate_view);
    return ok != 0;
}

static bool metal_graph_flash_moe_submit_route_batch(
        bool  parallel_submits,
        bool *commands_open,
        bool *commands_dirty) {
    if (!commands_open || !commands_dirty) return false;
    if (!*commands_dirty) return true;
    bool ok = false;
    if (parallel_submits) {
        ok = ds4_gpu_flush_commands() != 0;
        *commands_open = ok;
    } else {
        ok = ds4_gpu_end_commands() != 0;
        *commands_open = false;
    }
    *commands_dirty = false;
    return ok;
}

static bool metal_graph_flash_moe_decode_async_handout(
        ds4_gpu_graph             *g,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        uint32_t                   pos,
        uint64_t                   gate_expert_bytes,
        uint64_t                   gate_slot_stride,
        uint64_t                   gate_row_bytes,
        uint64_t                   down_expert_bytes,
        uint64_t                   down_slot_stride,
        uint64_t                   down_row_bytes,
        uint32_t                   expert_in_dim,
        uint32_t                   expert_mid_dim,
        uint32_t                   out_dim,
        bool                      *done_out) {
    if (done_out) *done_out = false;
    if (!g || !g->flash_moe || !layer ||
        il >= DS4_N_LAYER || !g->router_selected || !g->router_slot_selected) {
        return false;
    }
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    if (active_expert_used == 0 || active_expert_used > DS4_MAX_EXPERT_USED) return false;

    if (ds4_gpu_end_commands() == 0) return false;

    int32_t true_ids[DS4_MAX_EXPERT_USED];
    int32_t slot_ids[DS4_MAX_EXPERT_USED];
    int32_t route_load_idx[DS4_MAX_EXPERT_USED];
    bool resident_route[DS4_MAX_EXPERT_USED];
    for (uint32_t k = 0; k < active_expert_used; k++) {
        true_ids[k] = -1;
        slot_ids[k] = -1;
        route_load_idx[k] = -1;
        resident_route[k] = false;
    }

    bool ok = ds4_gpu_tensor_read(g->router_selected,
                                  0,
                                  true_ids,
                                  (uint64_t)active_expert_used * sizeof(true_ids[0])) != 0;
    if (!ok) return false;
    flash_moe_decode_trace_record(pos, il, true_ids);

    bool protected_experts[DS4_MAX_EXPERT];
    int32_t request_evicted[DS4_MAX_EXPERT_USED];
    bool request_misses[DS4_MAX_EXPERT_USED];
    memset(protected_experts, 0, sizeof(protected_experts));
    if (flash_moe_preprotect_topk_enabled()) {
        metal_graph_flash_moe_protect_routed_experts(true_ids,
                                                     active_expert_used,
                                                     protected_experts);
    }
    ds4_flash_decode_async_load loads[DS4_MAX_EXPERT_USED];
    memset(loads, 0, sizeof(loads));
    uint32_t n_loads = 0;
    const ds4_flash_moe_layer_sidecar *sidecar_layer = &g->flash_moe->layer[il];

    ok = metal_graph_flash_moe_resolve_request(g, il, true_ids,
            active_expert_used, protected_experts, slot_ids,
            request_evicted, request_misses);
    for (uint32_t k = 0; ok && k < active_expert_used; k++) {
        bool duplicate = false;
        for (uint32_t prev = 0; prev < k; prev++) {
            if (true_ids[prev] == true_ids[k]) {
                route_load_idx[k] = route_load_idx[prev];
                resident_route[k] = resident_route[prev];
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        const int32_t slot = slot_ids[k];
        const int32_t evicted = request_evicted[k];
        const bool miss = request_misses[k];
        metal_graph_flash_moe_note_request_slot(g, il, true_ids[k], slot, miss);
        if (!miss) {
            resident_route[k] = true;
            continue;
        }
        if (n_loads >= active_expert_used) {
            ok = false;
            break;
        }
        metal_graph_flash_moe_store_evicted_l1_slot(g, il, evicted, slot);
        const uint32_t li = n_loads++;
        route_load_idx[k] = (int32_t)li;
        loads[li].expert = true_ids[k];
        loads[li].slot = slot;
        loads[li].evicted = evicted;
        ds4_flash_decode_read_job *job = &loads[li].job;
        ok = ds4_flash_decode_read_job_set_sidecar(job,
                                                   sidecar_layer,
                                                   true_ids[k],
                                                   flash_moe_cache_io_split());
        if (!ok) break;
        if (g->flash_shared_l2_slot_bank) {
            const uint8_t *l2_src = NULL;
            if (metal_graph_flash_moe_shared_l2_lookup(g, il, true_ids[k], &l2_src) && l2_src) {
                ok = ds4_flash_decode_read_job_set_memory_src_copy(
                        job, l2_src, sidecar_layer->expert_stride);
            }
        }
        if (ok && !job->memory_src && g->flash_l2_slot_bank) {
            const uint8_t *l2_src = NULL;
            if (metal_graph_flash_moe_l2_lookup(g, il, true_ids[k], &l2_src) && l2_src) {
                ok = ds4_flash_decode_read_job_set_memory_src_copy(
                        job, l2_src, sidecar_layer->expert_stride);
            }
        }
    }

    if (ok) {
        ok = ds4_gpu_tensor_write(g->router_slot_selected,
                                  0,
                                  slot_ids,
                                  (uint64_t)active_expert_used * sizeof(slot_ids[0])) != 0;
    }
    if (ok) {
        metal_graph_flash_moe_record_decode_slots(g,
                                                  il,
                                                  true_ids,
                                                  slot_ids,
                                                  active_expert_used);
    }
    if (ok) {
        g->flash_async_handout_calls++;
        const uint32_t hist_idx = n_loads <= DS4_MAX_EXPERT_USED ?
                                  n_loads : DS4_MAX_EXPERT_USED;
        g->flash_async_handout_miss_hist[hist_idx]++;
        if (n_loads == 0) {
            g->flash_async_handout_grouped_calls++;
        } else {
            g->flash_async_handout_miss_calls++;
            g->flash_async_handout_miss_routes += n_loads;
        }
    }

    if (ok && n_loads == 0) {
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) {
            ok = metal_graph_flash_moe_compute_grouped_banked(g,
                                                              layer,
                                                              il,
                                                              gate_expert_bytes,
                                                              gate_slot_stride,
                                                              gate_row_bytes,
                                                              down_expert_bytes,
                                                              down_slot_stride,
                                                              down_row_bytes,
                                                              expert_in_dim,
                                                              expert_mid_dim,
                                                              out_dim,
                                                              active_expert_used);
        }
        if (ok) {
            metal_graph_flash_moe_note_replay_plan_use(g, il, active_expert_used);
            if (done_out) *done_out = true;
        }
        metal_graph_flash_moe_async_load_cleanup(g, il, loads, n_loads);
        return ok;
    }

    const uint32_t wait_miss_max = flash_moe_async_handout_wait_miss_max();
    const bool wait_grouped_force =
        ok &&
        n_loads > 0 &&
        n_loads <= wait_miss_max &&
        flash_moe_async_handout_wait_miss_force_enabled();
    const bool split_requested =
        ok &&
        n_loads > 0 &&
        !wait_grouped_force &&
        flash_moe_async_handout_overlap_misses_enabled() &&
        n_loads >= flash_moe_async_handout_split_miss_min();
    if (ok && n_loads > 0) {
        ok = metal_graph_flash_moe_async_start_loads(g,
                                                     il,
                                                     sidecar_layer,
                                                     loads,
                                                     n_loads,
                                                     !split_requested,
                                                     flash_moe_async_handout_io_split());
    }

    const bool wait_grouped_ready =
        ok &&
        split_requested &&
        n_loads <= wait_miss_max &&
        metal_graph_flash_moe_async_loads_complete(loads, n_loads);
    if (ok && (wait_grouped_force || wait_grouped_ready)) {
        for (uint32_t li = 0; ok && li < n_loads; li++) {
            ok = metal_graph_flash_moe_async_load_join_upload(g, il, &loads[li]);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) {
            ok = metal_graph_flash_moe_compute_grouped_banked(g,
                                                              layer,
                                                              il,
                                                              gate_expert_bytes,
                                                              gate_slot_stride,
                                                              gate_row_bytes,
                                                              down_expert_bytes,
                                                              down_slot_stride,
                                                              down_row_bytes,
                                                              expert_in_dim,
                                                              expert_mid_dim,
                                                              out_dim,
                                                              active_expert_used);
        }
        if (ok) {
            g->flash_async_handout_wait_grouped_calls++;
            metal_graph_flash_moe_note_replay_plan_use(g, il, active_expert_used);
            if (done_out) *done_out = true;
        }
        metal_graph_flash_moe_async_load_cleanup(g, il, loads, n_loads);
        return ok;
    }

    if (ok && !split_requested) {
        for (uint32_t li = 0; ok && li < n_loads; li++) {
            ok = metal_graph_flash_moe_async_load_join_upload(g, il, &loads[li]);
        }
        if (ok) {
            g->flash_async_handout_sync_miss_calls++;
            ok = ds4_gpu_begin_commands() != 0;
        }
        metal_graph_flash_moe_async_load_cleanup(g, il, loads, n_loads);
        return ok;
    }

    if (ok) g->flash_async_handout_split_calls++;
    bool commands_open = false;
    bool commands_dirty = false;
    if (ok) {
        ok = ds4_gpu_begin_commands() != 0;
        commands_open = ok;
    }
    const bool parallel_submits = flash_moe_async_handout_parallel_submits_enabled();
    const uint32_t chunk_miss_min = flash_moe_async_handout_chunk_miss_min();
    const bool chunk_misses =
        chunk_miss_min > 0 && n_loads > 0 && n_loads >= chunk_miss_min;

    for (uint32_t k = 0; ok && k < active_expert_used; k++) {
        if (!resident_route[k]) continue;
        ok = metal_graph_flash_moe_compute_route_to_down(g,
                                                         layer,
                                                         il,
                                                         k,
                                                         slot_ids[k],
                                                         gate_expert_bytes,
                                                         gate_slot_stride,
                                                         gate_row_bytes,
                                                         down_expert_bytes,
                                                         down_slot_stride,
                                                         down_row_bytes,
                                                         expert_in_dim,
                                                         expert_mid_dim,
                                                         out_dim);
        if (ok) commands_dirty = true;
    }

    if (ok && chunk_misses && n_loads > 0) {
        if (commands_dirty) {
            ok = metal_graph_flash_moe_submit_route_batch(parallel_submits,
                                                          &commands_open,
                                                          &commands_dirty);
        }
        for (uint32_t li = 0; ok && li < n_loads; li++) {
            ok = metal_graph_flash_moe_async_load_join_upload(g, il, &loads[li]);
        }
        if (ok && !commands_open) {
            ok = ds4_gpu_begin_commands() != 0;
            commands_open = ok;
        }
        for (uint32_t k = 0; ok && k < active_expert_used; k++) {
            if (resident_route[k]) continue;
            ok = metal_graph_flash_moe_compute_route_to_down(g,
                                                             layer,
                                                             il,
                                                             k,
                                                             slot_ids[k],
                                                             gate_expert_bytes,
                                                             gate_slot_stride,
                                                             gate_row_bytes,
                                                             down_expert_bytes,
                                                             down_slot_stride,
                                                             down_row_bytes,
                                                             expert_in_dim,
                                                             expert_mid_dim,
                                                             out_dim);
            if (ok) commands_dirty = true;
        }
    }

    for (uint32_t k = 0; ok && !chunk_misses && k < active_expert_used; k++) {
        if (resident_route[k]) continue;
        const int32_t li = route_load_idx[k];
        if (li < 0 || li >= (int32_t)n_loads) {
            ok = false;
            break;
        }
        if (commands_dirty) {
            ok = metal_graph_flash_moe_submit_route_batch(parallel_submits,
                                                          &commands_open,
                                                          &commands_dirty);
            if (!ok) break;
        }
        ok = metal_graph_flash_moe_async_load_join_upload(g, il, &loads[li]);
        if (!ok) break;
        if (!commands_open) {
            ok = ds4_gpu_begin_commands() != 0;
            commands_open = ok;
        }
        if (!ok) break;
        ok = metal_graph_flash_moe_compute_route_to_down(g,
                                                         layer,
                                                         il,
                                                         k,
                                                         slot_ids[k],
                                                         gate_expert_bytes,
                                                         gate_slot_stride,
                                                         gate_row_bytes,
                                                         down_expert_bytes,
                                                         down_slot_stride,
                                                         down_row_bytes,
                                                         expert_in_dim,
                                                         expert_mid_dim,
                                                         out_dim);
        if (ok) commands_dirty = true;
    }

    if (ok) {
        ok = ds4_gpu_moe_sum_experts_tensor(g->routed_out,
                                            g->routed_down,
                                            out_dim,
                                            active_expert_used) != 0;
    }
    if (ok) {
        metal_graph_flash_moe_note_replay_plan_use(g, il, active_expert_used);
        if (done_out) *done_out = true;
    }
    if (!ok && commands_open) {
        ds4_gpu_end_commands();
        commands_open = false;
    }
    metal_graph_flash_moe_async_load_cleanup(g, il, loads, n_loads);
    return ok;
}
