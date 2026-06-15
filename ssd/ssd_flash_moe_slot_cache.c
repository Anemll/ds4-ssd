/* =========================================================================
 * ssd_flash_moe_slot_cache.c - Flash-MoE slot cache, replay state, and post-prefill shrink.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

static int32_t *flash_moe_slot_to_expert(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_slot_to_expert + (uint64_t)il * g->flash_slot_bank;
}

static int32_t *flash_moe_expert_to_slot(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
}

static uint64_t *flash_moe_slot_age(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_slot_age + (uint64_t)il * g->flash_slot_bank;
}

static int32_t *flash_moe_replay_slot_expert(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_replay_slot_expert + (uint64_t)il * g->flash_slot_bank;
}

static uint8_t *flash_moe_replay_slot_valid(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_replay_slot_valid + (uint64_t)il * g->flash_slot_bank;
}

static void metal_graph_flash_moe_replay_invalidate_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot) {
    if (!flash_moe_replay_plan_enabled()) return;
    if (!g || !g->flash_moe || !g->flash_replay_slot_expert ||
        !g->flash_replay_slot_valid ||
        il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return;
    }
    int32_t *replay_expert = flash_moe_replay_slot_expert(g, il);
    uint8_t *replay_valid = flash_moe_replay_slot_valid(g, il);
    if (replay_valid[slot] || replay_expert[slot] >= 0) {
        g->flash_replay_invalidations++;
    }
    replay_valid[slot] = 0;
    replay_expert[slot] = -1;
    ds4_gpu_flash_moe_icb_invalidate_slot(il, slot);
}

static void metal_graph_flash_moe_replay_mark_resident_missing(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot,
        int32_t        true_expert) {
    if (!flash_moe_replay_plan_enabled()) return;
    if (!g || !g->flash_moe || !g->flash_replay_slot_expert ||
        !g->flash_replay_slot_valid ||
        il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return;
    }
    int32_t *replay_expert = flash_moe_replay_slot_expert(g, il);
    uint8_t *replay_valid = flash_moe_replay_slot_valid(g, il);
    replay_expert[slot] = true_expert;
    replay_valid[slot] = 0;
}

static void metal_graph_flash_moe_record_decode_slots(
        ds4_gpu_graph *g,
        uint32_t       il,
        const int32_t *true_ids,
        const int32_t *slot_ids,
        uint32_t       n_ids) {
    const bool needs_decode_ids =
        flash_moe_replay_plan_enabled() ||
        flash_moe_mixed_slots6_grouped_enabled() ||
        (g && (g->flash_chunked_mixed_bank ||
               g->flash_direct_mmap_bank ||
               g->flash_per_expert_buffers ||
               g->flash_per_slot_buffers));
    if (!needs_decode_ids) return;
    if (!g || il >= DS4_N_LAYER || !true_ids || !slot_ids ||
        n_ids > DS4_MAX_EXPERT_USED) {
        return;
    }
    for (uint32_t k = 0; k < n_ids; k++) {
        g->flash_decode_true_ids[il][k] = true_ids[k];
        g->flash_decode_slot_ids[il][k] = slot_ids[k];
    }
    g->flash_decode_ids_valid[il] = 1;
}

static void metal_graph_flash_moe_note_replay_plan_use(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_ids) {
    if (!flash_moe_replay_plan_enabled()) return;
    if (!g || !g->flash_moe || !g->flash_replay_slot_expert ||
        !g->flash_replay_slot_valid ||
        il >= DS4_N_LAYER || !g->flash_decode_ids_valid[il]) {
        return;
    }
    if (n_ids > DS4_MAX_EXPERT_USED) n_ids = DS4_MAX_EXPERT_USED;
    bool seen_slots[DS4_MAX_EXPERT];
    memset(seen_slots, 0, sizeof(seen_slots));
    int32_t *replay_expert = flash_moe_replay_slot_expert(g, il);
    uint8_t *replay_valid = flash_moe_replay_slot_valid(g, il);
    for (uint32_t k = 0; k < n_ids; k++) {
        const int32_t true_expert = g->flash_decode_true_ids[il][k];
        const int32_t slot = g->flash_decode_slot_ids[il][k];
        if (true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
            slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
            continue;
        }
        if (slot < (int32_t)DS4_MAX_EXPERT && seen_slots[slot]) continue;
        if (slot < (int32_t)DS4_MAX_EXPERT) seen_slots[slot] = true;
        if (replay_valid[slot] && replay_expert[slot] == true_expert) {
            g->flash_replay_hits++;
            continue;
        }
        g->flash_replay_misses++;
        g->flash_replay_builds++;
        replay_expert[slot] = true_expert;
        replay_valid[slot] = 1;
    }
}

static bool metal_graph_flash_moe_find_resident_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t       *slot_out) {
    if (slot_out) *slot_out = -1;
    if (!g || !g->flash_moe ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
    int32_t *expert_to_slot = flash_moe_expert_to_slot(g, il);
    const int32_t slot = expert_to_slot[true_expert];
    if (slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
        slot_to_expert[slot] != true_expert) {
        return false;
    }
    if (slot_out) *slot_out = slot;
    return true;
}

static void metal_graph_flash_moe_record_prefill_slot_cache(
        ds4_gpu_graph *g,
        uint32_t       refs,
        uint64_t       hits_before,
        uint64_t       misses_before) {
    if (!g) return;
    g->flash_prefill_slot_refs += refs;
    if (g->flash_misses > misses_before) {
        g->flash_prefill_slot_installs += g->flash_misses - misses_before;
    } else if (g->flash_hits > hits_before) {
        g->flash_prefill_slot_hits += g->flash_hits - hits_before;
    }
}

static bool flash_moe_reset_slot_cache_after_prefill_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_RESET_SLOT_CACHE_AFTER_PREFILL") ||
           env_flag_enabled("DS4_FLASH_MOE_CLEAR_SLOT_CACHE_AFTER_PREFILL");
}

static bool flash_moe_realloc_slot_bank_after_prefill_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_REALLOC_SLOT_BANK_AFTER_PREFILL") ||
           env_flag_enabled("DS4_FLASH_MOE_RECREATE_SLOT_BANK_AFTER_PREFILL");
}

static void metal_graph_flash_moe_reset_slot_cache(
        ds4_gpu_graph *g,
        const char    *reason) {
    if (!g || !g->flash_moe || g->flash_slot_bank == 0 ||
        !g->flash_slot_to_expert || !g->flash_expert_to_slot ||
        !g->flash_slot_age) {
        return;
    }

    const uint64_t layer_slots = (uint64_t)DS4_N_LAYER * g->flash_slot_bank;
    for (uint64_t i = 0; i < layer_slots; i++) {
        g->flash_slot_to_expert[i] = -1;
    }
    memset(g->flash_expert_to_slot,
           0xff,
           (size_t)DS4_N_LAYER * DS4_N_EXPERT * sizeof(g->flash_expert_to_slot[0]));
    memset(g->flash_slot_age, 0, (size_t)layer_slots * sizeof(g->flash_slot_age[0]));
    g->flash_age = 0;

    if (g->flash_replay_slot_valid) {
        memset(g->flash_replay_slot_valid,
               0,
               (size_t)layer_slots * sizeof(g->flash_replay_slot_valid[0]));
    }
    if (g->flash_replay_slot_expert) {
        memset(g->flash_replay_slot_expert,
               0xff,
               (size_t)layer_slots * sizeof(g->flash_replay_slot_expert[0]));
    }
    memset(g->flash_decode_ids_valid, 0, sizeof(g->flash_decode_ids_valid));
    memset(g->flash_decode_true_ids, 0xff, sizeof(g->flash_decode_true_ids));
    memset(g->flash_decode_slot_ids, 0xff, sizeof(g->flash_decode_slot_ids));

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        for (uint32_t slot = 0; slot < g->flash_slot_bank; slot++) {
            ds4_gpu_flash_moe_icb_invalidate_slot(il, (int32_t)slot);
        }
    }
    if (g->flash_direct_mmap_bank) {
        metal_graph_flash_moe_init_direct_mmap_identity(g);
    }

    fprintf(stderr,
            "ds4: Flash-MoE slot cache reset %s: layers=%u slots=%u cleared=%llu\n",
            reason && reason[0] ? reason : "after prefill",
            (unsigned)DS4_N_LAYER,
            g->flash_slot_bank,
            (unsigned long long)layer_slots);
}

static bool metal_graph_flash_moe_realloc_slot_banks_after_prefill(
        ds4_gpu_graph *g,
        const char    *reason) {
    if (!flash_moe_realloc_slot_bank_after_prefill_enabled()) return true;
    if (!g || !g->flash_moe || g->flash_slot_bank == 0) return true;

    fprintf(stderr,
            "ds4: Flash-MoE slot banks tearing down/recreating %s: layers=%u slots=%u\n",
            reason && reason[0] ? reason : "after prefill",
            (unsigned)DS4_N_LAYER,
            g->flash_slot_bank);
    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr,
                "ds4: failed to synchronize before Flash-MoE slot-bank reallocation\n");
        return false;
    }
    metal_graph_flash_moe_free_slot_banks(g);
    if (!metal_graph_flash_moe_alloc_slot_banks(g, g->flash_moe, "reallocated after prefill")) {
        fprintf(stderr,
                "ds4: failed to recreate Flash-MoE slot banks %s\n",
                reason && reason[0] ? reason : "after prefill");
        return false;
    }
    return true;
}

/*
 * Decode-bank shrink target. On a RAM-limited machine a big wired slot bank
 * evicts the OS file cache that serves decode-miss preads at RAM speed, so
 * decode collapses to true SSD reads (the "decode cliff"; see
 * docs/flash-moe-stable-slot-progress.md and docs/mxfp4-handoff.md). A large
 * bank still helps prefill streaming and never hurts it, so the production fix
 * is to keep the requested bank for prefill and shrink it to a small decode
 * bank afterward, letting the file cache repopulate.
 *
 * Returns the desired decode slot count, or 0 for "no shrink". Honors:
 *   DS4_FLASH_MOE_DECODE_SLOT_BANK=<slots>   explicit slot count (0 disables)
 *   DS4_FLASH_MOE_DECODE_SSD_CACHE=<size>    GB/MB budget, e.g. 20GB
 * The slot form wins if both are set. The target is clamped to
 * [DS4_N_EXPERT_ACTIVE_USED, current bank]; a value >= current is a no-op.
 */
static uint32_t flash_moe_decode_slot_bank_for_budget(
        const ds4_flash_moe_sidecar *sidecar,
        uint32_t                     min_slots,
        uint32_t                     cur,
        uint64_t                     budget) {
    if (!sidecar || budget == 0 || min_slots == 0 || cur <= min_slots) return 0;
    uint32_t best = 0;
    for (uint32_t s = min_slots; s <= cur; s++) {
        uint64_t bytes = 0;
        if (!ds4_flash_moe_sidecar_bank_bytes_for_slots(sidecar, s, &bytes)) break;
        if (bytes <= budget) best = s; else break;
    }
    if (best == 0 || best >= cur) return 0;
    return best;
}

static uint32_t flash_moe_decode_slot_bank_target(const ds4_gpu_graph *g) {
    if (!g || !g->flash_moe || g->flash_slot_bank == 0) return 0;
    /* Per-expert/per-slot buffer modes keep every expert resident by design;
     * shrinking them is meaningless and would break their slot==expert map. */
    if (g->flash_per_expert_buffers || g->flash_per_slot_buffers) return 0;
    const uint32_t cur = g->flash_slot_bank;
    const uint32_t min_slots = DS4_N_EXPERT_ACTIVE_USED;

    const char *slots_env = getenv("DS4_FLASH_MOE_DECODE_SLOT_BANK");
    if (slots_env && slots_env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(slots_env, &end, 10);
        if (errno == 0 && end != slots_env && v >= 0) {
            if (v == 0) return 0; /* explicit opt-out */
            uint32_t t = (uint32_t)v;
            if (t < min_slots) t = min_slots;
            if (t >= cur) return 0;
            return t;
        }
    }

    const char *gb_env = getenv("DS4_FLASH_MOE_DECODE_SSD_CACHE");
    if (gb_env && gb_env[0]) {
        uint64_t budget = 0;
        if (ds4_parse_u64_suffix(gb_env, &budget) && budget > 0) {
            return flash_moe_decode_slot_bank_for_budget(g->flash_moe,
                                                         min_slots,
                                                         cur,
                                                         budget);
        }
    }

    if (flash_moe_fast_decode_l1_enabled()) {
        uint64_t budget = flash_moe_decode_l1_budget_bytes();
        return flash_moe_decode_slot_bank_for_budget(g->flash_moe,
                                                     min_slots,
                                                     cur,
                                                     budget);
    }

    /*
     * Automatic decode-bank shrink, RAM-limited regime only. When the sidecar
     * is larger than physical RAM, a big wired bank evicts the OS file cache
     * that serves decode-miss preads and decode collapses (measured ~13x on a
     * 96 GiB M3U at a 48 GiB bank). Shrink the decode bank to the same target
     * --ssd-cache auto would pick: DS4_SSD_CACHE_AUTO_PCT% (default 20) of the
     * RAM left after dense + context. On memory-rich machines (sidecar fits in
     * RAM) the bank coexists with the cache, so leave it alone.
     *
     * ONLY applies when the bank size came from --ssd-cache auto. An explicit
     * --ssd-cache <size> or --moe-slot-bank <N> is honored literally through
     * decode — the user said what they meant; they get the cliff warning, not
     * a silent override. Opt in to a decode shrink with an explicit size via
     * DS4_FLASH_MOE_DECODE_SLOT_BANK / DS4_FLASH_MOE_DECODE_SSD_CACHE above.
     */
    if (!g->flash_moe->slot_bank_auto) return 0;
    const uint64_t ram = ds4_gpu_system_memory_bytes();
    uint64_t sidecar_full = 0;
    if (ram == 0 ||
        !ds4_flash_moe_sidecar_bank_bytes_for_slots(g->flash_moe, DS4_N_EXPERT, &sidecar_full) ||
        sidecar_full <= ram) {
        return 0;
    }
    /* Only auto-rescue banks big enough to actually trigger the cliff (the same
     * 44 GiB threshold as the big-bank warning). Smaller banks that fit the
     * decode budget already are left exactly as configured. */
    const uint64_t warn_bank_bytes = 44ull * 1024ull * 1024ull * 1024ull;
    uint64_t cur_bytes = 0;
    if (!ds4_flash_moe_sidecar_bank_bytes_for_slots(g->flash_moe, cur, &cur_bytes) ||
        cur_bytes < warn_bank_bytes) {
        return 0;
    }
    const uint64_t reserved = g->dense_mapped_bytes + g->context_buffer_bytes;
    if (ram <= reserved) return 0;
    uint32_t pct = 20u;
    const char *pct_env = getenv("DS4_SSD_CACHE_AUTO_PCT");
    if (pct_env && pct_env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(pct_env, &end, 10);
        if (errno == 0 && end != pct_env && v >= 1 && v <= 100) pct = (uint32_t)v;
    }
    const uint64_t remaining = ram - reserved;
    const uint64_t budget =
        (remaining / 100u) * pct + (remaining % 100u) * pct / 100u;
    return flash_moe_decode_slot_bank_for_budget(g->flash_moe,
                                                 min_slots,
                                                 cur,
                                                 budget);
}

typedef struct {
    int32_t expert;
    uint32_t slot;
    uint64_t age;
} ds4_flash_moe_carry_slot;

static uint32_t metal_graph_flash_moe_select_hot_slots(
        const int32_t  *slot_to_expert,
        const uint64_t *slot_age,
        uint32_t        from_slots,
        uint32_t        target_slots,
        ds4_flash_moe_carry_slot *out,
        uint64_t       *resident_out) {
    if (resident_out) *resident_out = 0;
    if (!slot_to_expert || !slot_age || !out || target_slots == 0) return 0;

    uint32_t count = 0;
    for (uint32_t slot = 0; slot < from_slots; slot++) {
        const int32_t expert = slot_to_expert[slot];
        if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
        if (resident_out) (*resident_out)++;
        const uint64_t age = slot_age[slot];
        uint32_t pos = count;
        while (pos > 0 && out[pos - 1u].age < age) pos--;
        if (count < target_slots) {
            for (uint32_t j = count; j > pos; j--) out[j] = out[j - 1u];
            count++;
        } else if (pos < target_slots) {
            for (uint32_t j = target_slots - 1u; j > pos; j--) out[j] = out[j - 1u];
        } else {
            continue;
        }
        out[pos].expert = expert;
        out[pos].slot = slot;
        out[pos].age = age;
    }
    return count;
}

static void metal_graph_flash_moe_clear_replay_state_for_slots(
        ds4_gpu_graph *g,
        uint32_t       slots) {
    if (!g || slots == 0) return;
    const uint64_t layer_slots = (uint64_t)DS4_N_LAYER * slots;
    if (g->flash_replay_slot_valid) {
        memset(g->flash_replay_slot_valid,
               0,
               (size_t)layer_slots * sizeof(g->flash_replay_slot_valid[0]));
    }
    if (g->flash_replay_slot_expert) {
        memset(g->flash_replay_slot_expert,
               0xff,
               (size_t)layer_slots * sizeof(g->flash_replay_slot_expert[0]));
    }
    memset(g->flash_decode_ids_valid, 0, sizeof(g->flash_decode_ids_valid));
    memset(g->flash_decode_true_ids, 0xff, sizeof(g->flash_decode_true_ids));
    memset(g->flash_decode_slot_ids, 0xff, sizeof(g->flash_decode_slot_ids));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        for (uint32_t slot = 0; slot < slots; slot++) {
            ds4_gpu_flash_moe_icb_invalidate_slot(il, (int32_t)slot);
        }
    }
}

static void metal_graph_flash_moe_free_mixed_layer_locals(
        ds4_gpu_tensor *mixed,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *down) {
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(mixed);
}

static bool metal_graph_flash_moe_shrink_slot_banks_carry(
        ds4_gpu_graph *g,
        uint32_t       from_slots,
        uint32_t       target_slots,
        const char    *reason) {
    if (!g || !g->flash_moe || !g->flash_mixed_slot_bank ||
        g->flash_layer_slot_slab || target_slots == 0 ||
        target_slots >= from_slots || target_slots > DS4_MAX_EXPERT) {
        return false;
    }

    fprintf(stderr,
            "ds4: Flash-MoE shrink-carry enabled%s: preserving hottest resident "
            "records into the %u-slot mixed decode bank\n",
            reason && reason[0] ? reason : "",
            target_slots);

    uint64_t total_bank_bytes = 0;
    uint64_t resident_total = 0;
    uint64_t carried_total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
        ds4_gpu_tensor *old_mixed = g->flash_mixed_bank[il];
        if (!old_mixed || layer->expert_stride == 0 ||
            layer->expert_stride > UINT64_MAX / (uint64_t)target_slots) {
            return false;
        }
        const uint64_t mixed_bytes = layer->expert_stride * (uint64_t)target_slots;
        const uint64_t gate_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_GATE];
        const uint64_t up_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_UP];
        const uint64_t down_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_DOWN];

        ds4_flash_moe_carry_slot selected[DS4_MAX_EXPERT];
        memset(selected, 0, sizeof(selected));
        uint64_t resident_layer = 0;
        int32_t *old_slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * from_slots;
        uint64_t *old_slot_age =
            g->flash_slot_age + (uint64_t)il * from_slots;
        const uint32_t n_selected =
            metal_graph_flash_moe_select_hot_slots(old_slot_to_expert,
                                                   old_slot_age,
                                                   from_slots,
                                                   target_slots,
                                                   selected,
                                                   &resident_layer);
        resident_total += resident_layer;

        ds4_gpu_tensor *new_mixed =
            metal_graph_flash_moe_alloc_slot_bank_tensor(mixed_bytes);
        if (!new_mixed) return false;
        char label[64];
        snprintf(label, sizeof(label), "mixed-layer-%u-shrunk", il);
        if (!metal_graph_flash_moe_prepare_slot_bank_owner(
                    new_mixed,
                    label,
                    false,
                    flash_moe_slot_bank_touch_pages_enabled(),
                    NULL,
                    NULL)) {
            ds4_gpu_tensor_free(new_mixed);
            return false;
        }
        ds4_gpu_tensor *new_gate =
            ds4_gpu_tensor_view(new_mixed,
                                layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                gate_view_bytes);
        ds4_gpu_tensor *new_up =
            ds4_gpu_tensor_view(new_mixed,
                                layer->family_offset[DS4_FLASH_FAMILY_UP],
                                up_view_bytes);
        ds4_gpu_tensor *new_down =
            ds4_gpu_tensor_view(new_mixed,
                                layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                down_view_bytes);
        if (!new_gate || !new_up || !new_down) {
            metal_graph_flash_moe_free_mixed_layer_locals(new_mixed,
                                                          new_gate,
                                                          new_up,
                                                          new_down);
            return false;
        }

        uint8_t *old_base = (uint8_t *)ds4_gpu_tensor_contents(old_mixed);
        if (!old_base && n_selected != 0) {
            metal_graph_flash_moe_free_mixed_layer_locals(new_mixed,
                                                          new_gate,
                                                          new_up,
                                                          new_down);
            return false;
        }

        int32_t *new_slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * target_slots;
        uint64_t *new_slot_age =
            g->flash_slot_age + (uint64_t)il * target_slots;
        int32_t *expert_to_slot =
            g->flash_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
        for (uint32_t slot = 0; slot < target_slots; slot++) {
            new_slot_to_expert[slot] = -1;
            new_slot_age[slot] = 0;
        }
        for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
            expert_to_slot[expert] = -1;
        }

        for (uint32_t new_slot = 0; new_slot < n_selected; new_slot++) {
            const ds4_flash_moe_carry_slot *sel = &selected[new_slot];
            const uint8_t *src =
                old_base + (uint64_t)sel->slot * layer->expert_stride;
            if (!ds4_gpu_tensor_write(new_mixed,
                                      (uint64_t)new_slot * layer->expert_stride,
                                      src,
                                      layer->expert_stride)) {
                metal_graph_flash_moe_free_mixed_layer_locals(new_mixed,
                                                              new_gate,
                                                              new_up,
                                                              new_down);
                return false;
            }
            new_slot_to_expert[new_slot] = sel->expert;
            expert_to_slot[sel->expert] = (int32_t)new_slot;
            new_slot_age[new_slot] = sel->age;
        }

        ds4_gpu_tensor_free(g->flash_down_bank[il]);
        ds4_gpu_tensor_free(g->flash_up_bank[il]);
        ds4_gpu_tensor_free(g->flash_gate_bank[il]);
        ds4_gpu_tensor_free(g->flash_mixed_bank[il]);
        g->flash_mixed_bank[il] = new_mixed;
        g->flash_gate_bank[il] = new_gate;
        g->flash_up_bank[il] = new_up;
        g->flash_down_bank[il] = new_down;
        total_bank_bytes += mixed_bytes;
        carried_total += n_selected;
    }

    g->flash_slot_bank = target_slots;
    ((ds4_flash_moe_sidecar *)g->flash_moe)->slot_bank = target_slots;
    g->flash_per_expert_bank_bytes = total_bank_bytes;
    g->flash_shrink_preserved_slots = carried_total != 0;
    metal_graph_flash_moe_clear_replay_state_for_slots(g, target_slots);

    const uint64_t dense_bytes = g->dense_mapped_bytes;
    const uint64_t context_bytes = g->context_buffer_bytes;
    fprintf(stderr,
            "ds4: Flash-MoE slot banks shrink-carried for decode: layers=%u slots=%u "
            "gpu-bank=%.1fGB Dense: %.1fGB Context: %.1fGB Total <<<< %.1fGB >>>> "
            "preserved=%" PRIu64 "/%" PRIu64 "\n",
            (unsigned)DS4_N_LAYER,
            target_slots,
            (double)total_bank_bytes / 1073741824.0,
            (double)dense_bytes / 1073741824.0,
            (double)context_bytes / 1073741824.0,
            (double)(total_bank_bytes + dense_bytes + context_bytes) / 1073741824.0,
            carried_total,
            resident_total);
    fprintf(stderr, "ds4: Flash-MoE slot bank layout: mixed expert-major\n");
    return true;
}

static bool metal_graph_flash_moe_shrink_slot_banks_gpu_l2(
        ds4_gpu_graph *g,
        uint32_t       from_slots,
        uint32_t       target_slots,
        const char    *reason) {
    if (!g || !g->flash_moe || !g->flash_mixed_slot_bank ||
        g->flash_layer_slot_slab || target_slots == 0 ||
        target_slots >= from_slots) {
        return false;
    }

    fprintf(stderr,
            "ds4: Flash-MoE split GPU-L2 shrink%s: converting the %u-slot "
            "prefill bank into a %u-slot mixed L1 plus a %u-slot GPU L2\n",
            reason && reason[0] ? reason : "",
            from_slots,
            target_slots,
            from_slots - target_slots);

    const uint32_t l2_slots = from_slots - target_slots;
    if (!metal_graph_flash_moe_gpu_l2_init_maps(g, l2_slots)) {
        fprintf(stderr, "ds4: failed to initialize Flash-MoE GPU-L2 maps\n");
        return false;
    }

    uint64_t l1_bytes_total = 0;
    uint64_t l2_bytes_total = 0;
    uint64_t resident_total = 0;
    uint64_t carried_l1_total = 0;
    uint64_t carried_l2_total = 0;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
        ds4_gpu_tensor *old_mixed = g->flash_mixed_bank[il];
        if (!old_mixed || layer->expert_stride == 0 ||
            layer->expert_stride > UINT64_MAX / (uint64_t)target_slots ||
            layer->expert_stride > UINT64_MAX / (uint64_t)l2_slots) {
            return false;
        }

        const uint64_t l1_bytes =
            layer->expert_stride * (uint64_t)target_slots;
        const uint64_t l2_bytes =
            layer->expert_stride * (uint64_t)l2_slots;
        const uint64_t gate_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_GATE];
        const uint64_t up_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_UP];
        const uint64_t down_view_bytes =
            (uint64_t)(target_slots - 1u) * layer->expert_stride +
            layer->family_bytes[DS4_FLASH_FAMILY_DOWN];

        ds4_gpu_tensor *new_l1 =
            metal_graph_flash_moe_alloc_slot_bank_tensor(l1_bytes);
        ds4_gpu_tensor *new_l2 =
            metal_graph_flash_moe_alloc_slot_bank_tensor(l2_bytes);
        ds4_gpu_tensor *new_gate = NULL;
        ds4_gpu_tensor *new_up = NULL;
        ds4_gpu_tensor *new_down = NULL;
        if (!new_l1 || !new_l2) {
            ds4_gpu_tensor_free(new_l2);
            ds4_gpu_tensor_free(new_l1);
            return false;
        }

        char label[80];
        snprintf(label, sizeof(label), "split-l1-layer-%u", il);
        if (!metal_graph_flash_moe_prepare_slot_bank_owner(
                    new_l1,
                    label,
                    false,
                    flash_moe_slot_bank_touch_pages_enabled(),
                    NULL,
                    NULL)) {
            ds4_gpu_tensor_free(new_l2);
            ds4_gpu_tensor_free(new_l1);
            return false;
        }
        snprintf(label, sizeof(label), "split-gpu-l2-layer-%u", il);
        if (!metal_graph_flash_moe_prepare_slot_bank_owner(
                    new_l2,
                    label,
                    false,
                    flash_moe_slot_bank_touch_pages_enabled(),
                    NULL,
                    NULL)) {
            ds4_gpu_tensor_free(new_l2);
            ds4_gpu_tensor_free(new_l1);
            return false;
        }

        new_gate = ds4_gpu_tensor_view(new_l1,
                                       layer->family_offset[DS4_FLASH_FAMILY_GATE],
                                       gate_view_bytes);
        new_up = ds4_gpu_tensor_view(new_l1,
                                     layer->family_offset[DS4_FLASH_FAMILY_UP],
                                     up_view_bytes);
        new_down = ds4_gpu_tensor_view(new_l1,
                                       layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                                       down_view_bytes);
        if (!new_gate || !new_up || !new_down) {
            metal_graph_flash_moe_free_mixed_layer_locals(new_l1,
                                                          new_gate,
                                                          new_up,
                                                          new_down);
            ds4_gpu_tensor_free(new_l2);
            return false;
        }

        ds4_flash_moe_carry_slot selected[DS4_MAX_EXPERT];
        memset(selected, 0, sizeof(selected));
        uint64_t resident_layer = 0;
        int32_t *old_slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * from_slots;
        uint64_t *old_slot_age =
            g->flash_slot_age + (uint64_t)il * from_slots;
        const uint32_t n_selected =
            metal_graph_flash_moe_select_hot_slots(old_slot_to_expert,
                                                   old_slot_age,
                                                   from_slots,
                                                   from_slots,
                                                   selected,
                                                   &resident_layer);
        resident_total += resident_layer;

        uint8_t *old_base = (uint8_t *)ds4_gpu_tensor_contents(old_mixed);
        if (!old_base && n_selected != 0) {
            metal_graph_flash_moe_free_mixed_layer_locals(new_l1,
                                                          new_gate,
                                                          new_up,
                                                          new_down);
            ds4_gpu_tensor_free(new_l2);
            return false;
        }

        int32_t *new_slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * target_slots;
        uint64_t *new_slot_age =
            g->flash_slot_age + (uint64_t)il * target_slots;
        int32_t *new_expert_to_slot =
            g->flash_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
        int32_t *l2_slot_to_expert =
            metal_graph_flash_moe_gpu_l2_slot_to_expert(g, il);
        int32_t *l2_expert_to_slot =
            metal_graph_flash_moe_gpu_l2_expert_to_slot(g, il);
        uint64_t *l2_slot_age =
            metal_graph_flash_moe_gpu_l2_slot_age(g, il);

        for (uint32_t slot = 0; slot < target_slots; slot++) {
            new_slot_to_expert[slot] = -1;
            new_slot_age[slot] = 0;
        }
        for (uint32_t expert = 0; expert < DS4_N_EXPERT; expert++) {
            new_expert_to_slot[expert] = -1;
        }

        const uint32_t l1_carry =
            n_selected < target_slots ? n_selected : target_slots;
        for (uint32_t new_slot = 0; new_slot < l1_carry; new_slot++) {
            const ds4_flash_moe_carry_slot *sel = &selected[new_slot];
            const uint8_t *src =
                old_base + (uint64_t)sel->slot * layer->expert_stride;
            if (!ds4_gpu_tensor_write(new_l1,
                                      (uint64_t)new_slot * layer->expert_stride,
                                      src,
                                      layer->expert_stride)) {
                metal_graph_flash_moe_free_mixed_layer_locals(new_l1,
                                                              new_gate,
                                                              new_up,
                                                              new_down);
                ds4_gpu_tensor_free(new_l2);
                return false;
            }
            new_slot_to_expert[new_slot] = sel->expert;
            new_expert_to_slot[sel->expert] = (int32_t)new_slot;
            new_slot_age[new_slot] = sel->age;
        }

        uint32_t l2_carry = 0;
        for (uint32_t i = l1_carry;
             i < n_selected && l2_carry < l2_slots;
             i++, l2_carry++) {
            const ds4_flash_moe_carry_slot *sel = &selected[i];
            const uint8_t *src =
                old_base + (uint64_t)sel->slot * layer->expert_stride;
            if (!ds4_gpu_tensor_write(new_l2,
                                      (uint64_t)l2_carry * layer->expert_stride,
                                      src,
                                      layer->expert_stride)) {
                metal_graph_flash_moe_free_mixed_layer_locals(new_l1,
                                                              new_gate,
                                                              new_up,
                                                              new_down);
                ds4_gpu_tensor_free(new_l2);
                return false;
            }
            l2_slot_to_expert[l2_carry] = sel->expert;
            l2_expert_to_slot[sel->expert] = (int32_t)l2_carry;
            l2_slot_age[l2_carry] = sel->age;
        }

        ds4_gpu_tensor_free(g->flash_down_bank[il]);
        ds4_gpu_tensor_free(g->flash_up_bank[il]);
        ds4_gpu_tensor_free(g->flash_gate_bank[il]);
        ds4_gpu_tensor_free(g->flash_mixed_bank[il]);
        g->flash_mixed_bank[il] = new_l1;
        g->flash_gate_bank[il] = new_gate;
        g->flash_up_bank[il] = new_up;
        g->flash_down_bank[il] = new_down;
        g->flash_gpu_l2_mixed_bank[il] = new_l2;

        l1_bytes_total += l1_bytes;
        l2_bytes_total += l2_bytes;
        carried_l1_total += l1_carry;
        carried_l2_total += l2_carry;
    }

    g->flash_slot_bank = target_slots;
    ((ds4_flash_moe_sidecar *)g->flash_moe)->slot_bank = target_slots;
    g->flash_per_expert_bank_bytes = l1_bytes_total;
    g->flash_gpu_l2_capacity_bytes = l2_bytes_total;
    g->flash_shrink_preserved_slots =
        (carried_l1_total + carried_l2_total) != 0;
    metal_graph_flash_moe_clear_replay_state_for_slots(g, target_slots);

    const uint64_t dense_bytes = g->dense_mapped_bytes;
    const uint64_t context_bytes = g->context_buffer_bytes;
    fprintf(stderr,
            "ds4: Flash-MoE split GPU-L2 decode layout: L1=%u slots %.2f GiB, "
            "L2=%u slots %.2f GiB, total-bank=%.2f GiB Dense=%.1fGB "
            "Context=%.1fGB Total <<<< %.1fGB >>>> preserved L1=%" PRIu64
            " L2=%" PRIu64 "/%" PRIu64 "\n",
            target_slots,
            (double)l1_bytes_total / 1073741824.0,
            l2_slots,
            (double)l2_bytes_total / 1073741824.0,
            (double)(l1_bytes_total + l2_bytes_total) / 1073741824.0,
            (double)dense_bytes / 1073741824.0,
            (double)context_bytes / 1073741824.0,
            (double)(l1_bytes_total + l2_bytes_total + dense_bytes + context_bytes) /
                1073741824.0,
            carried_l1_total,
            carried_l2_total,
            resident_total);
    fprintf(stderr,
            "ds4: Flash-MoE slot bank layout: mixed expert-major L1 + GPU victim L2\n");
    return true;
}

/*
 * Shrink the slot bank to `target` slots in place: synchronize, free the old
 * (large) banks, retarget both the graph slot count and the sidecar slot count
 * (which drives buffer byte sizing in the allocator), then re-allocate the
 * smaller banks. The slot index arrays (slot_to_expert/expert_to_slot/age/
 * replay) were allocated for the original, larger bank, so a smaller stride
 * only under-uses them — never overflows — and the subsequent cache reset
 * clears the now-smaller working set. Freeing the large bank returns its wired
 * pages to the OS so the file cache can repopulate for decode.
 */
static bool metal_graph_flash_moe_shrink_slot_banks_after_prefill(
        ds4_gpu_graph *g,
        uint32_t       target,
        const char    *reason) {
    if (!g || !g->flash_moe || g->flash_slot_bank == 0) return true;
    if (target == 0 || target >= g->flash_slot_bank) return true;

    const uint32_t from = g->flash_slot_bank;
    fprintf(stderr,
            "ds4: Flash-MoE shrinking decode slot bank %s: layers=%u slots %u->%u "
            "(frees wired bank pages so the OS file cache can serve decode-miss reads)\n",
            reason && reason[0] ? reason : "after prefill",
            (unsigned)DS4_N_LAYER, from, target);

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr,
                "ds4: failed to synchronize before Flash-MoE decode-bank shrink\n");
        return false;
    }
    g->flash_shrink_preserved_slots = false;
    if (flash_moe_shrink_carry_enabled()) {
        if (g->flash_mixed_slot_bank && !g->flash_layer_slot_slab) {
            if (!metal_graph_flash_moe_shrink_slot_banks_carry(g, from, target, reason)) {
                fprintf(stderr,
                        "ds4: failed to shrink-carry Flash-MoE slot bank "
                        "(slots %u->%u)\n",
                        from,
                        target);
                return false;
            }
            return true;
        }
        fprintf(stderr,
                "ds4: Flash-MoE shrink-carry requested but current layout is unsupported; "
                "falling back to free/reallocate shrink\n");
    }
    if (flash_moe_decode_gpu_l2_enabled()) {
        if (g->flash_mixed_slot_bank && !g->flash_layer_slot_slab) {
            if (!metal_graph_flash_moe_shrink_slot_banks_gpu_l2(g, from, target, reason)) {
                fprintf(stderr,
                        "ds4: failed to shrink Flash-MoE slot bank with GPU-L2 "
                        "(slots %u->%u)\n",
                        from,
                        target);
                return false;
            }
            return true;
        }
        fprintf(stderr,
                "ds4: Flash-MoE GPU-L2 requested but current layout is unsupported; "
                "falling back to free/reallocate shrink\n");
    }
    if (!metal_graph_flash_moe_l2_capture_before_shrink(g, from, target)) {
        fprintf(stderr,
                "ds4: failed to capture Flash-MoE L2 cache before decode-bank shrink\n");
        return false;
    }
    metal_graph_flash_moe_free_slot_banks(g);
    g->flash_slot_bank = target;
    /* The sidecar's slot_bank drives per-layer buffer byte sizing in the
     * allocator; keep it in lockstep with the graph slot count. The underlying
     * object is non-const (it was mutated during --ssd-cache resolution). */
    ((ds4_flash_moe_sidecar *)g->flash_moe)->slot_bank = target;
    if (!metal_graph_flash_moe_alloc_slot_banks(g, g->flash_moe, "shrunk for decode")) {
        fprintf(stderr,
                "ds4: failed to re-allocate Flash-MoE slot banks after decode shrink "
                "(slots=%u)\n", target);
        return false;
    }
    return true;
}

static bool metal_graph_flash_moe_reset_slot_cache_after_prefill(
        ds4_gpu_graph *g,
        const char    *reason) {
    const uint32_t shrink_target = flash_moe_decode_slot_bank_target(g);
    const bool do_shrink = shrink_target != 0;
    const bool do_reset = flash_moe_reset_slot_cache_after_prefill_enabled();
    const bool do_realloc =
        !do_shrink && flash_moe_realloc_slot_bank_after_prefill_enabled();
    if (!do_shrink && !do_reset && !do_realloc) return true;
    if (do_shrink &&
        !metal_graph_flash_moe_shrink_slot_banks_after_prefill(g, shrink_target, reason)) {
        return false;
    }
    if (do_realloc && !metal_graph_flash_moe_realloc_slot_banks_after_prefill(g, reason)) {
        return false;
    }
    if (g->flash_shrink_preserved_slots) {
        fprintf(stderr,
                "ds4: Flash-MoE slot cache preserved %s; skipping post-shrink reset\n",
                reason && reason[0] ? reason : "after prefill");
        return true;
    }
    metal_graph_flash_moe_reset_slot_cache(g, reason);
    return true;
}

/* Flash-MoE diagnostics, oracle replay, histograms, and verbose prefill tracing. */
#include "ssd_flash_moe_diagnostics.c"
