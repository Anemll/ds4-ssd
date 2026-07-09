/* =========================================================================
 * ssd_flash_moe_runtime.c - Flash-MoE runtime policy, IO, slot views, and async readers.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep graph-private helpers in
 * one translation unit while keeping each Flash-MoE area readable.
 */

static int get_prefill_dedup_prefetch(void)
{
    static int cached = -1;
    if (cached >= 0) return cached;

    const char *env = getenv("DS4_FLASH_MOE_PREFETCH");
    int dist = 3;   // default to match pipeline depth 4 with 4 banks

    if (env && env[0]) {
        int v = atoi(env);
        if (v >= 0) dist = v;
    }

    // Clamp to what we support with current allocation (4 banks)
    if (dist > 3) {
        fprintf(stderr, "ds4: DS4_FLASH_MOE_PREFETCH=%d clamped to 3 (only 4 prefill banks allocated)\n", dist);
        dist = 3;
    }
    if (dist < 0) dist = 0;

    cached = dist;
    return dist;
}

static int get_prefill_slot_cache_topk(uint32_t slot_bank)
{
    static int cached = -2;
    static bool warned_clamp = false;
    if (cached >= -1) {
        if (cached < 0) return 0;
        int safe = (int)(slot_bank / 2u);
        if (safe < 1) safe = 1;
        int ret = cached > (int)slot_bank ? (int)slot_bank : cached;
        if (ret > safe) {
            if (!warned_clamp) {
                fprintf(stderr,
                        "ds4: DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK=%d clamped to %d"
                        " to leave prefill slot-bank eviction headroom\n",
                        ret,
                        safe);
                warned_clamp = true;
            }
            ret = safe;
        }
        return ret;
    }

    const char *env = getenv("DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK");
    int topk = 0;
    if (env && env[0]) topk = atoi(env);
    if (topk < 0) topk = 0;
    cached = topk;
    return get_prefill_slot_cache_topk(slot_bank);
}

static bool flash_moe_prefill_slot_cache_topk_configured(void) {
    const char *env = getenv("DS4_FLASH_MOE_PREFILL_SLOT_CACHE_TOPK");
    return env && env[0];
}

static bool flash_moe_prefill_slot_prefetch_requested(void) {
    const char *prefill_env = getenv("DS4_FLASH_MOE_PREFILL_SLOT_PREFETCH");
    if (prefill_env && prefill_env[0]) return atoi(prefill_env) != 0;
    const char *decode_env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH");
    if (!decode_env || !decode_env[0] || atoi(decode_env) == 0) return false;
    const char *ane_prefill_env = getenv("DS4_FLASH_MOE_ANE_PREFILL");
    return !(ane_prefill_env && ane_prefill_env[0] && atoi(ane_prefill_env) != 0);
}

static int get_prefill_slot_cache_target(uint32_t slot_bank)
{
    if (flash_moe_per_expert_buffers_enabled()) return 0;
    int topk = get_prefill_slot_cache_topk(slot_bank);
    if (topk > 0 || flash_moe_prefill_slot_cache_topk_configured()) return topk;
    return flash_moe_prefill_slot_prefetch_requested() ? (int)(slot_bank / 2u) : 0;
}

static bool env_flag_enabled(const char *name) {
    const char *env = getenv(name);
    return env && env[0] && atoi(env) != 0;
}

static void ds4_setenv_override(const char *name, const char *value) {
    if (setenv(name, value, 1) != 0) {
        fprintf(stderr, "ds4: warning: failed to set %s=%s\n", name, value);
    }
}

static bool ds4_no_int8_paths_enabled(void) {
    return env_flag_enabled("DS4_NO_INT8");
}

static bool backend_diagnostic_logs_suppressed(void);
static bool backend_stats_logs_enabled(void);

static bool flash_moe_graph_uses_mxfp4_plane_split(const ds4_gpu_graph *g) {
    return g && g->flash_moe &&
           flash_moe_layer_all_mxfp4_plane_split(&g->flash_moe->layer[0]);
}

static bool ane_output_proj_enabled_for_run(const ds4_gpu_graph *g) {
    if (ds4_no_int8_paths_enabled()) return false;
    if (!env_flag_enabled("DS4_FLASH_MOE_ANE_OUTPUT_PROJ")) return false;
    const bool routed_ane_prefill_active =
        env_flag_enabled("DS4_FLASH_MOE_ANE_PREFILL") &&
        !flash_moe_graph_uses_mxfp4_plane_split(g);
    if (routed_ane_prefill_active &&
        !env_flag_enabled("DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE") &&
        !env_flag_enabled("DS4_FLASH_MOE_ANE_OUTPUT_PROJ_WITH_ROUTED")) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "ds4: ANE O-proj disabled while routed ANE prefill is active "
                    "(serial dependency boundary); set "
                    "DS4_FLASH_MOE_ANE_OUTPUT_PROJ_FORCE=1 to benchmark it anyway\n");
            warned = true;
        }
        return false;
    }
    return true;
}

static bool flash_moe_mpp_nax_forced_non_m5(void) {
    return env_flag_enabled("DS4_MPP_NAX_FORCE_NON_M5") ||
           env_flag_enabled("DS4_FLASH_MOE_MPP_FORCE_NON_M5") ||
           env_flag_enabled("DS4_FLASH_MOE_NAX_FORCE_NON_M5");
}

static bool flash_moe_mpp_nax_allowed(void) {
#ifdef DS4_NO_GPU
    return false;
#else
    return ds4_gpu_mpp_nax_supported() || flash_moe_mpp_nax_forced_non_m5();
#endif
}

static bool flash_moe_mpp_int8_prefill_requested(void) {
    if (ds4_no_int8_paths_enabled()) {
        return env_flag_enabled("DS4_RESIDENT_MOE_NAX_HALF");
    }
    const char *env = getenv("DS4_FLASH_MOE_MPP_INT8_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

static bool flash_moe_mpp_int8_prefill_enabled(void) {
    const bool requested = flash_moe_mpp_int8_prefill_requested();
    const bool allowed = flash_moe_mpp_nax_allowed();
    if (env_flag_enabled("DS4_FLASH_MOE_KERNEL_LOG")) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr,
                    "ds4: Flash-MoE MPP prefill gate: requested=%d allowed=%d "
                    "flash=%s force=%s force_flash=%s force_nax=%s\n",
                    requested ? 1 : 0,
                    allowed ? 1 : 0,
                    getenv("DS4_FLASH_MOE_MPP_INT8_PREFILL") ?: "",
                    getenv("DS4_MPP_NAX_FORCE_NON_M5") ?: "",
                    getenv("DS4_FLASH_MOE_MPP_FORCE_NON_M5") ?: "",
                    getenv("DS4_FLASH_MOE_NAX_FORCE_NON_M5") ?: "");
        }
    }
    if (!requested) return false;
    if (allowed) return true;
    static bool warned = false;
    if (!warned && !backend_diagnostic_logs_suppressed()) {
        warned = true;
        if (ds4_no_int8_paths_enabled()) {
            fprintf(stderr,
                    "ds4: --no-int8 requested but NAX-half fallback is unavailable "
                    "on this Metal device; GPU sidecar stays on grouped ALU\n");
        } else {
            fprintf(stderr,
                    "ds4: DS4_FLASH_MOE_MPP_INT8_PREFILL ignored on this Metal device; "
                    "GPU sidecar stays on grouped ALU. Set DS4_MPP_NAX_FORCE_NON_M5=1 "
                    "for explicit NAX/MPP experiments.\n");
        }
    }
    return false;
}

static bool flash_moe_mpp_partial_tiles_allowed(void) {
    return env_flag_enabled("DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES") ||
           env_flag_enabled("DS4_FLASH_MOE_NAX_ALLOW_PARTIAL_TILES");
}

static bool resident_moe_mpp_dedup_prefill_enabled(void) {
    const char *env = getenv("DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL");
    if (!env || !env[0]) env = getenv("DS4_RESIDENT_MOE_NAX_DEDUP_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

static bool resident_moe_backend_name_is_ane(const char *backend) {
    return backend && strncmp(backend, "ane", 3) == 0;
}

static bool resident_moe_prefill_backend_is_ane(uint32_t n_tokens) {
    if (ds4_no_int8_paths_enabled()) return false;
    char backend[24];
    ds4_gpu_resident_backend_for_tokens(n_tokens, backend, sizeof(backend));
    return resident_moe_backend_name_is_ane(backend);
}

static bool resident_moe_prefill_config_has_ane_backend(void) {
    if (ds4_no_int8_paths_enabled()) return false;
    const char *forced = getenv("DS4_RESIDENT_MOE_BACKEND");
    if (resident_moe_backend_name_is_ane(forced)) return true;
    const char *tbl = getenv("DS4_RESIDENT_MOE_PREFILL_BY_TOKENS");
    return tbl && strstr(tbl, ":ane") != NULL;
}

static double resident_moe_mpp_min_tile_util(void) {
    if (env_flag_enabled("DS4_RESIDENT_MOE_MPP_FORCE") ||
        env_flag_enabled("DS4_RESIDENT_MOE_NAX_FORCE")) {
        return 0.0;
    }
    const char *env = getenv("DS4_RESIDENT_MOE_MPP_MIN_TILE_UTIL");
    if (!env || !env[0]) env = getenv("DS4_RESIDENT_MOE_NAX_MIN_TILE_UTIL");
    if (!env || !env[0]) env = getenv("DS4_RESIDENT_MPP_INT8_MIN_TILE_UTIL");
    double v = 0.75;
    if (env && env[0]) {
        char *end = NULL;
        double parsed = strtod(env, &end);
        if (end != env && isfinite(parsed)) v = parsed;
    }
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return v;
}

static uint32_t get_prefill_hybrid_ane_min_refs(void) {
    static int cached = -1;
    if (cached >= 0) return (uint32_t)cached;

    uint32_t min_refs = 129u;
    const char *env = getenv("DS4_FLASH_MOE_HYBRID_ANE_MIN_REFS");
    if (!env || !env[0]) env = getenv("DS4_FLASH_MOE_ANE_MIN_REFS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed <= 4096ul) min_refs = (uint32_t)parsed;
    }
    cached = (int)min_refs;
    return min_refs;
}

static uint32_t get_prefill_concurrent_min_gpu_groups(void) {
    static int cached = -1;
    if (cached >= 0) return (uint32_t)cached;

    uint32_t min_groups = 1u;
    const char *env = getenv("DS4_FLASH_MOE_CONCURRENT_MIN_GPU_GROUPS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed <= 32ul) min_groups = (uint32_t)parsed;
    }
    cached = (int)min_groups;
    return min_groups;
}

static uint32_t get_prefill_ane_output_queue_depth(void) {
    static int cached = -1;
    if (cached >= 0) return (uint32_t)cached;

    uint32_t depth = 1u;
    const char *env = getenv("DS4_FLASH_MOE_ANE_OUTPUT_QUEUE");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed >= 1ul && parsed <= 8ul) depth = (uint32_t)parsed;
    }
    cached = (int)depth;
    return depth;
}

typedef enum {
    DS4_PREFILL_LANE_GPU = 0,
    DS4_PREFILL_LANE_ANE = 1,
} ds4_prefill_lane;

typedef struct {
    uint32_t ui;
    uint32_t begin;
    uint32_t refs;
    int32_t expert;
    ds4_prefill_lane lane;
    bool tail_slice;
    double gpu_cost;
    double ane_cost;
    double ssd_cost;
    bool slot_cached;
    bool ane_eligible;
} ds4_flash_prefill_plan_item;

static double get_env_double_clamped(const char *name,
                                     double      fallback,
                                     double      min_value,
                                     double      max_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    double v = strtod(env, &end);
    if (end == env || !isfinite(v)) return fallback;
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    return v;
}

static uint32_t get_env_u32_clamped(const char *name,
                                    uint32_t    fallback,
                                    uint32_t    min_value,
                                    uint32_t    max_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    uint64_t v = 0;
    if (!ds4_parse_u64_suffix(env, &v)) return fallback;
    if (v < (uint64_t)min_value) v = min_value;
    if (v > (uint64_t)max_value) v = max_value;
    return (uint32_t)v;
}

static uint32_t prefill_ane_default_batch(void) {
    return get_env_u32_clamped("DS4_FLASH_MOE_ANE_BATCH", 256u, 1u, 4096u);
}

static uint32_t prefill_ane_select_batch_for_refs(uint32_t refs) {
    uint32_t fallback = prefill_ane_default_batch();
    uint32_t batches[16];
    uint32_t n = 0;
    const char *env = getenv("DS4_FLASH_MOE_ANE_BATCHES");
    if (env && env[0]) {
        const char *p = env;
        while (*p && n < (uint32_t)(sizeof(batches) / sizeof(batches[0]))) {
            while (*p == ',' || *p == ':' || *p == ';' || isspace((unsigned char)*p)) p++;
            if (!*p) break;
            char *end = NULL;
            unsigned long v = strtoul(p, &end, 10);
            if (end == p) break;
            if (v > 0 && v <= 4096ul) batches[n++] = (uint32_t)v;
            p = end;
        }
    }
    if (n == 0) batches[n++] = fallback;
    for (uint32_t i = 1; i < n; i++) {
        uint32_t key = batches[i];
        uint32_t j = i;
        while (j > 0 && batches[j - 1] > key) {
            batches[j] = batches[j - 1];
            j--;
        }
        batches[j] = key;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (refs <= batches[i]) return batches[i];
    }
    return batches[n - 1];
}

static uint32_t prefill_ane_chunk_refs_for_group(uint32_t refs, uint32_t *batch_out, bool *can_chunk_out) {
    uint32_t batch = prefill_ane_select_batch_for_refs(refs);
    uint32_t max_refs = batch;
    const char *max_refs_env = getenv("DS4_FLASH_MOE_ANE_MAX_REFS");
    if (max_refs_env && max_refs_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(max_refs_env, &end, 10);
        if (end != max_refs_env && parsed > 0 && parsed <= 4096ul) max_refs = (uint32_t)parsed;
    }
    uint32_t chunk_refs = max_refs < batch ? max_refs : batch;
    if (chunk_refs == 0) chunk_refs = 1;
    const bool chunk_big_refs = env_flag_enabled("DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS");
    if (batch_out) *batch_out = batch;
    if (can_chunk_out) *can_chunk_out = chunk_big_refs || refs <= max_refs;
    return chunk_refs;
}

static void prefill_plan_sort_by_work(ds4_flash_prefill_plan_item *items, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        ds4_flash_prefill_plan_item key = items[i];
        const double key_cost = key.ane_eligible ?
            (key.gpu_cost > key.ane_cost ? key.gpu_cost : key.ane_cost) :
            key.gpu_cost;
        uint32_t j = i;
        while (j > 0) {
            const double prev_cost = items[j - 1].ane_eligible ?
                (items[j - 1].gpu_cost > items[j - 1].ane_cost ? items[j - 1].gpu_cost : items[j - 1].ane_cost) :
                items[j - 1].gpu_cost;
            if (prev_cost > key_cost ||
                (prev_cost == key_cost && items[j - 1].refs >= key.refs)) {
                break;
            }
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
}

static uint32_t build_flash_prefill_overlap_plan(
        ds4_flash_prefill_plan_item *exec,
        uint32_t                     n_unique,
        const int32_t               *unique,
        const int32_t               *offsets,
        const bool                  *slot_cache_expert,
        uint32_t                     hybrid_ane_min_refs,
        double                     *planned_gpu_cost_out,
        double                     *planned_ane_cost_out,
        uint32_t                   *planned_ane_groups_out,
        uint32_t                   *planned_gpu_groups_out,
        uint64_t                   *planned_ane_refs_out,
        uint64_t                   *planned_gpu_refs_out,
        uint32_t                   *planned_ssd_groups_out,
        uint32_t                   *planned_tail_gpu_groups_out,
        uint64_t                   *planned_tail_gpu_refs_out) {
    if (planned_gpu_cost_out) *planned_gpu_cost_out = 0.0;
    if (planned_ane_cost_out) *planned_ane_cost_out = 0.0;
    if (planned_ane_groups_out) *planned_ane_groups_out = 0;
    if (planned_gpu_groups_out) *planned_gpu_groups_out = 0;
    if (planned_ane_refs_out) *planned_ane_refs_out = 0;
    if (planned_gpu_refs_out) *planned_gpu_refs_out = 0;
    if (planned_ssd_groups_out) *planned_ssd_groups_out = 0;
    if (planned_tail_gpu_groups_out) *planned_tail_gpu_groups_out = 0;
    if (planned_tail_gpu_refs_out) *planned_tail_gpu_refs_out = 0;
    if (!exec || !unique || !offsets || n_unique == 0) return 0;

    ds4_flash_prefill_plan_item *work =
        (ds4_flash_prefill_plan_item *)xcalloc((size_t)n_unique * 2u, sizeof(work[0]));
    ds4_flash_prefill_plan_item *ane =
        (ds4_flash_prefill_plan_item *)xcalloc((size_t)n_unique * 2u, sizeof(ane[0]));
    ds4_flash_prefill_plan_item *gpu =
        (ds4_flash_prefill_plan_item *)xcalloc((size_t)n_unique * 2u, sizeof(gpu[0]));

    /* Production M3U sidecar runs use an ANE-heavy schedule. This knob is a
     * relative speed in the cost model, not a fraction: 0.99 means roughly
     * parity with GPU, while 99 makes the scheduler keep ANE work alive. */
    const double ane_rel_speed =
        get_env_double_clamped("DS4_FLASH_MOE_SCHED_ANE_REL_SPEED", 99.0, 0.05, 1024.0);
    const double ane_call_refs =
        get_env_double_clamped("DS4_FLASH_MOE_SCHED_ANE_CALL_REFS", 96.0, 0.0, 8192.0);
    const double ssd_refs =
        get_env_double_clamped("DS4_FLASH_MOE_SCHED_SSD_REFS", 64.0, 0.0, 8192.0);
    const double ane_min_util =
        get_env_double_clamped("DS4_FLASH_MOE_SCHED_ANE_MIN_UTIL", 0.0, 0.0, 1.0);

    double gpu_load = 0.0;
    double ane_load = 0.0;
    uint32_t work_n = 0;
    for (uint32_t i = 0; i < n_unique; i++) {
        const int32_t expert = unique[i];
        if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
        const uint32_t begin = (uint32_t)offsets[i];
        const uint32_t refs = (uint32_t)(offsets[i + 1] - offsets[i]);
        if (refs == 0) continue;
        uint32_t ane_batch = 0;
        bool can_chunk = false;
        const uint32_t chunk_refs = prefill_ane_chunk_refs_for_group(refs, &ane_batch, &can_chunk);
        const uint32_t chunks = (refs + chunk_refs - 1u) / chunk_refs;
        const uint32_t padded_refs = chunks * ane_batch;
        const double group_util = padded_refs ? (double)refs / (double)padded_refs : 0.0;
        const bool slot_cached =
            slot_cache_expert && expert >= 0 && expert < (int32_t)DS4_N_EXPERT && slot_cache_expert[expert];
        const double stage_cost = slot_cached ? 0.0 : ssd_refs;
        const bool split_low_util_tail =
            can_chunk && refs >= hybrid_ane_min_refs && group_util < ane_min_util &&
            refs > chunk_refs && chunk_refs > 0;
        const uint32_t ane_refs =
            split_low_util_tail ? (refs / chunk_refs) * chunk_refs : refs;
        const uint32_t gpu_tail_refs = split_low_util_tail ? refs - ane_refs : 0;
        if (ane_refs != 0) {
            const uint32_t ane_chunks = (ane_refs + chunk_refs - 1u) / chunk_refs;
            const uint32_t ane_padded_refs = ane_chunks * ane_batch;
            ds4_flash_prefill_plan_item item = {
                .ui = i,
                .begin = begin,
                .refs = ane_refs,
                .expert = expert,
                .lane = DS4_PREFILL_LANE_GPU,
                .tail_slice = false,
                .gpu_cost = (double)ane_refs + stage_cost,
                .ane_cost = ((double)ane_padded_refs / ane_rel_speed) + ane_call_refs + stage_cost,
                .ssd_cost = stage_cost,
                .slot_cached = slot_cached,
                .ane_eligible = can_chunk && ane_refs >= hybrid_ane_min_refs &&
                                ((double)ane_refs / (double)ane_padded_refs) >= ane_min_util,
            };
            work[work_n++] = item;
        }
        if (gpu_tail_refs != 0) {
            ds4_flash_prefill_plan_item tail = {
                .ui = i,
                .begin = begin + ane_refs,
                .refs = gpu_tail_refs,
                .expert = expert,
                .lane = DS4_PREFILL_LANE_GPU,
                .tail_slice = true,
                .gpu_cost = (double)gpu_tail_refs + stage_cost,
                .ane_cost = DBL_MAX / 4.0,
                .ssd_cost = stage_cost,
                .slot_cached = slot_cached,
                .ane_eligible = false,
            };
            work[work_n++] = tail;
            if (planned_tail_gpu_groups_out) (*planned_tail_gpu_groups_out)++;
            if (planned_tail_gpu_refs_out) *planned_tail_gpu_refs_out += tail.refs;
        } else if (ane_refs == 0) {
            ds4_flash_prefill_plan_item item = {
                .ui = i,
                .begin = begin,
                .refs = refs,
                .expert = expert,
                .lane = DS4_PREFILL_LANE_GPU,
                .tail_slice = false,
                .gpu_cost = (double)refs + stage_cost,
                .ane_cost = DBL_MAX / 4.0,
                .ssd_cost = stage_cost,
                .slot_cached = slot_cached,
                .ane_eligible = false,
            };
            work[work_n++] = item;
        }
        if (planned_ssd_groups_out && stage_cost > 0.0) (*planned_ssd_groups_out)++;
    }

    prefill_plan_sort_by_work(work, work_n);

    uint32_t ane_n = 0;
    uint32_t gpu_n = 0;
    for (uint32_t i = 0; i < work_n; i++) {
        ds4_flash_prefill_plan_item item = work[i];
        if (item.ane_eligible) {
            const double if_gpu = gpu_load + item.gpu_cost;
            const double if_ane = ane_load + item.ane_cost;
            const double make_gpu = if_gpu > ane_load ? if_gpu : ane_load;
            const double make_ane = gpu_load > if_ane ? gpu_load : if_ane;
            if (make_ane < make_gpu) {
                item.lane = DS4_PREFILL_LANE_ANE;
                ane_load = if_ane;
                ane[ane_n++] = item;
                if (planned_ane_groups_out) (*planned_ane_groups_out)++;
                if (planned_ane_refs_out) *planned_ane_refs_out += item.refs;
            } else {
                gpu_load = if_gpu;
                gpu[gpu_n++] = item;
                if (planned_gpu_groups_out) (*planned_gpu_groups_out)++;
                if (planned_gpu_refs_out) *planned_gpu_refs_out += item.refs;
            }
        } else {
            gpu_load += item.gpu_cost;
            gpu[gpu_n++] = item;
            if (planned_gpu_groups_out) (*planned_gpu_groups_out)++;
            if (planned_gpu_refs_out) *planned_gpu_refs_out += item.refs;
        }
    }

    uint32_t out_n = 0;
    uint32_t ai = 0;
    uint32_t gi = 0;
    while (ai < ane_n || gi < gpu_n) {
        double cover = 0.0;
        if (ai < ane_n) {
            exec[out_n++] = ane[ai];
            cover = ane[ai].ane_cost * get_env_double_clamped("DS4_FLASH_MOE_SCHED_GPU_COVER", 0.90, 0.0, 8.0);
            ai++;
        }
        double covered = 0.0;
        while (gi < gpu_n && (ai >= ane_n || covered < cover || out_n == 0)) {
            exec[out_n++] = gpu[gi];
            covered += gpu[gi].gpu_cost;
            gi++;
        }
        if (ai >= ane_n && gi < gpu_n) {
            while (gi < gpu_n) exec[out_n++] = gpu[gi++];
        }
    }

    if (planned_gpu_cost_out) *planned_gpu_cost_out = gpu_load;
    if (planned_ane_cost_out) *planned_ane_cost_out = ane_load;
    free(gpu);
    free(ane);
    free(work);
    return out_n;
}

static bool flash_moe_pread_full(int fd, uint64_t offset, uint8_t *dst, uint64_t bytes) {
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t remain = bytes - done;
        const size_t want = remain > (1ull << 30) ? (size_t)(1ull << 30) : (size_t)remain;
        if (offset > (uint64_t)INT64_MAX || done > (uint64_t)INT64_MAX - offset) return false;
        ssize_t n = pread(fd, dst + done, want, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

static bool flash_moe_pread_full_interruptible(
        int           fd,
        uint64_t      offset,
        uint8_t      *dst,
        uint64_t      bytes,
        volatile int *stop_requested) {
    const uint64_t max_chunk = 1ull << 20;
    uint64_t done = 0;
    while (done < bytes) {
        if (stop_requested && *stop_requested) {
            errno = ECANCELED;
            return false;
        }
        const uint64_t remain = bytes - done;
        const size_t want = (size_t)(remain > max_chunk ? max_chunk : remain);
        if (offset > (uint64_t)INT64_MAX || done > (uint64_t)INT64_MAX - offset) return false;
        ssize_t n = pread(fd, dst + done, want, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

/* --- io-split: split one expert's SSD read into N page-aligned ranges issued
 * concurrently, so a single expert read keeps several NVMe requests in flight
 * (deeper queue depth -> closer to the drive's real bandwidth). Page-aligned so
 * each range is a whole number of pages; if the expert size isn't page-aligned
 * we fall back to a single read. Opt-in: default 1 (no split). */
#define DS4_FLASH_MOE_IO_PAGE_BYTES 16384u
#define DS4_FLASH_MOE_MAX_IO_SPLIT  16

/* Decode/slot-bank read split. Default 4 (page-aligned reads fall back to 1):
 * fanning each expert read into 4 concurrent NVMe requests ~2x's read bandwidth
 * with no regression (split=1 == baseline). Prefill inherits this when its own
 * split is unset. Override with DS4_FLASH_MOE_CACHE_IO_SPLIT. */
static int flash_moe_cache_io_split(void) {
    const char *env = getenv("DS4_FLASH_MOE_CACHE_IO_SPLIT");
    int n = (env && env[0]) ? atoi(env) : 4;
    if (n < 1) n = 1;
    if (n > DS4_FLASH_MOE_MAX_IO_SPLIT) n = DS4_FLASH_MOE_MAX_IO_SPLIT;
    return n;
}

/* Prefill read split; falls back to the cache split when unset (mirrors llama). */
static int flash_moe_prefill_io_split(void) {
    const char *env = getenv("DS4_FLASH_MOE_PREFILL_IO_SPLIT");
    if (env && env[0]) {
        int n = atoi(env);
        if (n < 1) n = 1;
        if (n > DS4_FLASH_MOE_MAX_IO_SPLIT) n = DS4_FLASH_MOE_MAX_IO_SPLIT;
        return n;
    }
    return flash_moe_cache_io_split();
}

/* Effective split for a given byte count + desired split: only split when the
 * size is an exact multiple of the page, and never into more ranges than pages. */
static uint32_t flash_moe_active_io_split(uint64_t bytes, int want) {
    if (want <= 1 || bytes == 0 || (bytes % DS4_FLASH_MOE_IO_PAGE_BYTES) != 0) return 1u;
    const uint64_t pages = bytes / DS4_FLASH_MOE_IO_PAGE_BYTES;
    uint32_t n = (uint32_t)want;
    if ((uint64_t)n > pages) n = (uint32_t)pages;
    if (n > DS4_FLASH_MOE_MAX_IO_SPLIT) n = DS4_FLASH_MOE_MAX_IO_SPLIT;
    if (n < 1u) n = 1u;
    return n;
}

/* Byte range [off,off+len) for chunk c of an nsplit-way page-aligned split. */
static void flash_moe_io_split_range(uint64_t bytes, uint32_t nsplit, uint32_t c,
                                     uint64_t *off, uint64_t *len) {
    if (nsplit <= 1u) { *off = 0; *len = bytes; return; }
    const uint64_t pages = bytes / DS4_FLASH_MOE_IO_PAGE_BYTES;
    const uint64_t ppc = pages / nsplit;            /* >=1: active_io_split clamps n<=pages */
    const uint64_t page_start = (uint64_t)c * ppc;
    *off = page_start * DS4_FLASH_MOE_IO_PAGE_BYTES;
    *len = (c == nsplit - 1u) ? (bytes - *off) : (ppc * DS4_FLASH_MOE_IO_PAGE_BYTES);
}

/* Synchronous N-way page-aligned read using a transient thread fan-out. Used by
 * the synchronous staging paths (decode install / sync prefill stage). */
typedef struct {
    int fd;
    uint64_t offset;
    uint8_t *dst;
    uint64_t len;
    bool ok;
} ds4_flash_io_split_job;

static void *ds4_flash_io_split_worker(void *arg) {
    ds4_flash_io_split_job *j = (ds4_flash_io_split_job *)arg;
    j->ok = flash_moe_pread_full(j->fd, j->offset, j->dst, j->len);
    return NULL;
}

static bool flash_moe_pread_split(int fd, uint64_t offset, uint8_t *dst,
                                  uint64_t bytes, int want) {
    const uint32_t nsplit = flash_moe_active_io_split(bytes, want);
    if (nsplit <= 1u) return flash_moe_pread_full(fd, offset, dst, bytes);
    pthread_t th[DS4_FLASH_MOE_MAX_IO_SPLIT];
    ds4_flash_io_split_job jobs[DS4_FLASH_MOE_MAX_IO_SPLIT];
    uint32_t started = 0;
    for (uint32_t c = 0; c < nsplit; c++) {
        uint64_t coff = 0, clen = 0;
        flash_moe_io_split_range(bytes, nsplit, c, &coff, &clen);
        jobs[c].fd = fd; jobs[c].offset = offset + coff;
        jobs[c].dst = dst + coff; jobs[c].len = clen; jobs[c].ok = false;
        if (c == nsplit - 1u ||
            pthread_create(&th[c], NULL, ds4_flash_io_split_worker, &jobs[c]) != 0) {
            /* Read the last (or any un-spawnable) chunk inline on this thread. */
            jobs[c].ok = flash_moe_pread_full(fd, jobs[c].offset, jobs[c].dst, jobs[c].len);
            continue;
        }
        started |= (1u << c);
    }
    bool ok = true;
    for (uint32_t c = 0; c < nsplit; c++) {
        if (started & (1u << c)) pthread_join(th[c], NULL);
        ok = ok && jobs[c].ok;
    }
    return ok;
}

static bool metal_graph_flash_moe_family_file_offset(
        const ds4_flash_moe_layer_sidecar *layer,
        int32_t                            true_expert,
        uint32_t                           fam,
        uint64_t                          *offset_out) {
    if (offset_out) *offset_out = 0;
    if (!layer || !offset_out ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        fam >= DS4_FLASH_FAMILY_COUNT) {
        return false;
    }
    const uint64_t bytes = layer->family_bytes[fam];
    if (bytes == 0) return true;
    if (layer->family_major) {
        if (true_expert != 0 &&
            bytes > UINT64_MAX / (uint64_t)true_expert) return false;
        const uint64_t expert_off = (uint64_t)true_expert * bytes;
        if (layer->family_file_offset[fam] > UINT64_MAX - expert_off) return false;
        *offset_out = layer->family_file_offset[fam] + expert_off;
        return true;
    }
    if (true_expert != 0 &&
        layer->expert_stride > UINT64_MAX / (uint64_t)true_expert) return false;
    const uint64_t record_offset = (uint64_t)true_expert * layer->expert_stride;
    if (layer->family_offset[fam] > UINT64_MAX - record_offset) return false;
    *offset_out = record_offset + layer->family_offset[fam];
    return true;
}

static bool metal_graph_flash_moe_record_file_offset(
        const ds4_flash_moe_layer_sidecar *layer,
        int32_t                            true_expert,
        uint64_t                          *offset_out) {
    if (offset_out) *offset_out = 0;
    if (!layer || !offset_out ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    if (true_expert != 0 &&
        layer->expert_stride > UINT64_MAX / (uint64_t)true_expert) return false;
    *offset_out = (uint64_t)true_expert * layer->expert_stride;
    return true;
}

static bool metal_graph_flash_moe_read_record_to_buf(
        const ds4_flash_moe_layer_sidecar *layer,
        int32_t                            true_expert,
        uint8_t                           *dst,
        int                                io_split) {
    if (!layer || !dst ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    if (!layer->family_major) {
        uint64_t record_offset = 0;
        return metal_graph_flash_moe_record_file_offset(layer, true_expert, &record_offset) &&
               flash_moe_pread_split(layer->fd,
                                     record_offset,
                                     dst,
                                     layer->expert_stride,
                                     io_split);
    }
    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
        const uint64_t bytes = layer->family_bytes[fam];
        if (bytes == 0) continue;
        if (layer->family_offset[fam] > layer->expert_stride ||
            bytes > layer->expert_stride - layer->family_offset[fam]) {
            return false;
        }
        uint64_t file_offset = 0;
        if (!metal_graph_flash_moe_family_file_offset(layer,
                                                      true_expert,
                                                      fam,
                                                      &file_offset)) {
            return false;
        }
        if (!flash_moe_pread_split(layer->fd,
                                   file_offset,
                                   dst + layer->family_offset[fam],
                                   bytes,
                                   io_split)) {
            return false;
        }
    }
    return true;
}

static bool flash_moe_direct_slot_pread_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DIRECT_SLOT_PREAD");
    return env == NULL || env[0] == '\0' || atoi(env) != 0;
}

static bool flash_moe_decode_prefetch_scratch_only_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH_SCRATCH_ONLY");
    return env == NULL || env[0] == '\0' || atoi(env) != 0;
}

static bool flash_moe_decode_prefetch_direct_slot_pread_enabled(void) {
    if (flash_moe_decode_prefetch_scratch_only_enabled()) return false;
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH_DIRECT_SLOT_PREAD");
    if (env && env[0]) return atoi(env) != 0;
    return flash_moe_direct_slot_pread_enabled();
}

static uint32_t flash_moe_decode_prefetch_max_loads(void) {
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS");
    if (!env || !env[0]) return 0u;
    char *end = NULL;
    errno = 0;
    long v = strtol(env, &end, 10);
    if (errno != 0 || end == env || v < 0) return 0u;
    if (v > (long)DS4_N_EXPERT_ACTIVE_USED) return DS4_N_EXPERT_ACTIVE_USED;
    return (uint32_t)v;
}

static uint32_t flash_moe_decode_prefetch_layer_stride(void) {
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH_LAYER_STRIDE");
    if (!env || !env[0]) return 1u;
    char *end = NULL;
    errno = 0;
    long v = strtol(env, &end, 10);
    if (errno != 0 || end == env || v < 1) return 1u;
    if (v > (long)DS4_N_LAYER) return DS4_N_LAYER;
    return (uint32_t)v;
}

static bool flash_moe_mixed_slot_bank_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_MIXED_SLOT_BANK");
    return env == NULL || env[0] == '\0' || atoi(env) != 0;
}

static bool flash_moe_layer_slot_slab_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_LAYER_SLOT_SLAB");
}

static bool flash_moe_per_expert_buffers_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_PER_EXPERT_BUFFERS") ||
           env_flag_enabled("DS4_FLASH_MOE_FULL_RESIDENT_EXPERT_BUFFERS");
}

static bool flash_moe_per_slot_buffers_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_PER_SLOT_BUFFERS") ||
           env_flag_enabled("DS4_FLASH_MOE_SEPARATE_SLOT_BUFFERS");
}

static bool flash_moe_sidecar_all_mxfp4(const ds4_flash_moe_sidecar *sidecar) {
    if (!sidecar) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
        for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
            if (!layer->present[fam] || layer->family_type[fam] != DS4_TENSOR_MXFP4) {
                return false;
            }
        }
    }
    return true;
}

static bool flash_moe_sidecar_all_mxfp4_block(const ds4_flash_moe_sidecar *sidecar) {
    if (!flash_moe_sidecar_all_mxfp4(sidecar)) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
        for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
            if (layer->family_mxfp4_plane_split[fam]) return false;
        }
    }
    return true;
}

static bool flash_moe_auto_per_slot_policy_allows(const ds4_flash_moe_sidecar *sidecar) {
    const char *policy = getenv("DS4_FLASH_MOE_AUTO_PER_SLOT_POLICY");
    if (!policy || !policy[0]) policy = "all";
    if (!strcasecmp(policy, "0") || !strcasecmp(policy, "off") ||
        !strcasecmp(policy, "false") || !strcasecmp(policy, "none")) {
        return false;
    }
    if (!strcasecmp(policy, "1") || !strcasecmp(policy, "on") ||
        !strcasecmp(policy, "true") || !strcasecmp(policy, "all")) {
        return true;
    }
    if (!strcasecmp(policy, "mxfp4") || !strcasecmp(policy, "mxfp4-any")) {
        return flash_moe_sidecar_all_mxfp4(sidecar);
    }
    if (!strcasecmp(policy, "mxfp4-block") ||
        !strcasecmp(policy, "mxfp4_interleaved") ||
        !strcasecmp(policy, "mxfp4-interleaved")) {
        return flash_moe_sidecar_all_mxfp4_block(sidecar);
    }
    static bool warned = false;
    if (!warned) {
        fprintf(stderr,
                "ds4: warning: unknown DS4_FLASH_MOE_AUTO_PER_SLOT_POLICY=%s "
                "(expected all, mxfp4, mxfp4-block, or off); using all\n",
                policy);
        warned = true;
    }
    return true;
}

static uint64_t flash_moe_auto_per_slot_threshold_bytes(void) {
    const uint64_t default_threshold = 44ull * 1024ull * 1024ull * 1024ull;
    const char *env = getenv("DS4_FLASH_MOE_AUTO_PER_SLOT_THRESHOLD");
    if (!env || !env[0]) env = getenv("DS4_FLASH_MOE_AUTO_PER_SLOT_THRESHOLD_BYTES");
    if (env && env[0]) {
        uint64_t parsed = 0;
        if (ds4_parse_u64_suffix(env, &parsed) && parsed != 0) return parsed;
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "ds4: warning: invalid DS4_FLASH_MOE_AUTO_PER_SLOT_THRESHOLD=%s "
                    "(use bytes or K/M/G suffix, e.g. 44GB); using 44GB\n",
                    env);
            warned = true;
        }
    }
    return default_threshold;
}

static bool flash_moe_auto_per_slot_buffers_enabled(const ds4_flash_moe_sidecar *sidecar) {
    if (!sidecar || sidecar->slot_bank == 0) return false;
    if (env_flag_enabled("DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS") ||
        env_flag_enabled("DS4_FLASH_MOE_FORCE_MIXED_SLOT_BANK")) {
        return false;
    }
    const char *auto_env = getenv("DS4_FLASH_MOE_AUTO_PER_SLOT_BUFFERS");
    if (auto_env && auto_env[0] && atoi(auto_env) == 0) return false;
    if (!flash_moe_auto_per_slot_policy_allows(sidecar)) return false;

    uint64_t total = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &sidecar->layer[il];
        if (layer->expert_stride > UINT64_MAX / (uint64_t)sidecar->slot_bank) {
            return false;
        }
        const uint64_t layer_bytes =
            (uint64_t)sidecar->slot_bank * layer->expert_stride;
        if (total > UINT64_MAX - layer_bytes) return false;
        total += layer_bytes;
    }

    const uint64_t split_threshold = flash_moe_auto_per_slot_threshold_bytes();
    return total >= split_threshold;
}

static bool flash_moe_per_slot_lazy_alloc_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_PER_SLOT_LAZY_ALLOC") ||
           env_flag_enabled("DS4_FLASH_MOE_LAZY_SLOT_BUFFERS");
}

/* MXFP4/record-table direct-mmap policy helpers are defined below once slot-view helpers are available. */
static bool flash_moe_untracked_slot_bank_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_UNTRACKED_SLOT_BANK") ||
           env_flag_enabled("DS4_FLASH_MOE_UNTRACKED_BANK_BUFFERS");
}

static bool flash_moe_active_staging_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_ACTIVE_STAGING") ||
           env_flag_enabled("DS4_FLASH_MOE_STAGE_ACTIVE") ||
           env_flag_enabled("DS4_FLASH_MOE_ACTIVE_MIXED_STAGE");
}

static bool flash_moe_chunked_mixed_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_CHUNKED_MIXED") ||
           env_flag_enabled("DS4_FLASH_MOE_CHUNKED_MIXED_BANK") ||
           env_flag_enabled("DS4_FLASH_MOE_MIXED_CHUNKS");
}

static uint32_t flash_moe_chunked_mixed_slots(uint32_t slot_bank) {
    const char *env = getenv("DS4_FLASH_MOE_CHUNK_SLOTS");
    uint32_t slots = 56u;
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && v > 0) slots = (uint32_t)v;
    }
    if (slots == 0) slots = 1u;
    if (slots > slot_bank && slot_bank != 0) slots = slot_bank;
    return slots;
}

static bool flash_moe_fast_decode_l1_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_FAST_DECODE_L1") ||
           env_flag_enabled("DS4_FLASH_MOE_POLICY_DECODE_L1");
}

static bool flash_moe_prefill_decode_l1_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_PREFILL_DECODE_L1") ||
           env_flag_enabled("DS4_FLASH_MOE_PREFILL_FAST_L1") ||
           env_flag_enabled("DS4_FLASH_MOE_L1_FROM_PREFILL");
}

static uint64_t flash_moe_decode_l1_budget_bytes(void) {
    uint64_t budget = 32ull * 1024ull * 1024ull * 1024ull;
    const char *envs[] = {
        "DS4_FLASH_MOE_DECODE_SSD_CACHE",
        "DS4_FLASH_MOE_FAST_DECODE_SSD_CACHE",
        "DS4_FLASH_MOE_DECODE_L1_SSD_CACHE",
    };
    for (size_t i = 0; i < sizeof(envs) / sizeof(envs[0]); i++) {
        const char *env = getenv(envs[i]);
        if (!env || !env[0]) continue;
        uint64_t parsed = 0;
        if (ds4_parse_u64_suffix(env, &parsed) && parsed > 0) {
            return parsed;
        }
    }
    return budget;
}

static bool flash_moe_decode_l2_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_DECODE_L2") ||
           env_flag_enabled("DS4_FLASH_MOE_L2_CACHE");
}

static bool flash_moe_decode_shared_l2_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_SHARED_L2") ||
           env_flag_enabled("DS4_FLASH_MOE_DECODE_SHARED_L2") ||
           env_flag_enabled("DS4_FLASH_MOE_SHARED_L2_CACHE");
}

static bool flash_moe_decode_gpu_l2_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_GPU_L2") ||
           env_flag_enabled("DS4_FLASH_MOE_DECODE_GPU_L2") ||
           env_flag_enabled("DS4_FLASH_MOE_RESIDENT_L2");
}

static bool flash_moe_shrink_carry_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_SHRINK_CARRY") ||
           env_flag_enabled("DS4_FLASH_MOE_CARRY_SHRINK_CACHE");
}

static bool flash_moe_mixed_slots6_grouped_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_MIXED_SLOTS6_GROUPED") ||
           env_flag_enabled("DS4_FLASH_MOE_MIXED_SLOTS6");
}

static bool flash_moe_chunked_slots6_grouped_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_CHUNKED_SLOTS6_GROUPED") ||
           env_flag_enabled("DS4_FLASH_MOE_CHUNKED_SLOTS6");
}

static bool flash_moe_chunked_bank_slots6_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_CHUNKED_BANK_SLOTS6") ||
           env_flag_enabled("DS4_FLASH_MOE_CHUNKED_DIRECT_SLOTS6") ||
           env_flag_enabled("DS4_FLASH_MOE_SLOTWISE_DECODE");
}

static bool flash_moe_six_slot_baseline_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_SIX_SLOT_BASELINE") ||
           env_flag_enabled("DS4_FLASH_MOE_DECODE_ACTIVE_ONLY") ||
           env_flag_enabled("DS4_FLASH_MOE_NO_SLOT_REUSE");
}

static bool flash_moe_decode_bank_shrink_requested(void) {
    const char *slots_env = getenv("DS4_FLASH_MOE_DECODE_SLOT_BANK");
    if (slots_env && slots_env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(slots_env, &end, 10);
        if (errno == 0 && end != slots_env) return v > 0;
        return false;
    }

    const char *gb_env = getenv("DS4_FLASH_MOE_DECODE_SSD_CACHE");
    if (gb_env && gb_env[0]) {
        uint64_t budget = 0;
        return ds4_parse_u64_suffix(gb_env, &budget) && budget > 0;
    }

    return flash_moe_fast_decode_l1_enabled() ||
           flash_moe_prefill_decode_l1_enabled();
}

static bool flash_moe_did_modify_range_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DID_MODIFY_RANGE");
    return env == NULL || env[0] == '\0' || atoi(env) != 0;
}

static bool flash_moe_slot_bank_residency_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_SLOT_BANK_RESIDENCY");
}

static bool flash_moe_slot_bank_touch_pages_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES");
}

static bool flash_moe_slotwise_decode_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_SLOTWISE_DECODE");
    return env && env[0] && atoi(env) != 0;
}

static bool metal_graph_flash_moe_l2_store_l1_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        l1_slot);

static bool flash_moe_baked_slot_decode_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *explicit_baked = getenv("DS4_FLASH_MOE_BAKED_SLOT_DECODE");
        if (explicit_baked && explicit_baked[0]) {
            enabled = atoi(explicit_baked) != 0 ? 1 : 0;
        } else {
            enabled = env_flag_enabled("DS4_FLASH_MOE_STABLE_REPLAY") ? 1 : 0;
        }
    }
    return enabled != 0;
}

static bool flash_moe_replay_plan_enabled(void) {
    return flash_moe_baked_slot_decode_enabled();
}

static bool flash_moe_async_handout_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = env_flag_enabled("DS4_FLASH_MOE_ASYNC_HANDOUT") ? 1 : 0;
    }
    return enabled != 0;
}

static bool flash_moe_async_handout_overlap_misses_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_FLASH_MOE_ASYNC_HANDOUT_OVERLAP_MISSES");
        enabled = (env && env[0] && atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool flash_moe_async_handout_parallel_submits_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_ASYNC_HANDOUT_PARALLEL_SUBMITS");
}

static int flash_moe_async_handout_io_split(void) {
    const char *env = getenv("DS4_FLASH_MOE_ASYNC_HANDOUT_IO_SPLIT");
    int n = (env && env[0]) ? atoi(env) : 1;
    if (n < 1) n = 1;
    if (n > DS4_FLASH_MOE_MAX_IO_SPLIT) n = DS4_FLASH_MOE_MAX_IO_SPLIT;
    return n;
}

static bool flash_moe_async_handout_chunk_misses_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_ASYNC_HANDOUT_CHUNK_MISSES");
}

static uint32_t flash_moe_async_handout_chunk_miss_min(void) {
    const char *env = getenv("DS4_FLASH_MOE_ASYNC_HANDOUT_CHUNK_MISS_MIN");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && v > 0) {
            if (v > (long)DS4_MAX_EXPERT_USED) return DS4_MAX_EXPERT_USED;
            return (uint32_t)v;
        }
    }
    return flash_moe_async_handout_chunk_misses_enabled() ? 1u : 0u;
}

static uint32_t flash_moe_async_handout_split_miss_min(void) {
    const char *env = getenv("DS4_FLASH_MOE_ASYNC_HANDOUT_SPLIT_MISS_MIN");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && v > 0) {
            if (v > (long)DS4_MAX_EXPERT_USED) return DS4_MAX_EXPERT_USED;
            return (uint32_t)v;
        }
    }
    return 4u;
}

static uint32_t flash_moe_async_handout_wait_miss_max(void) {
    const char *env = getenv("DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_MAX");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && v >= 0) {
            if (v > (long)DS4_MAX_EXPERT_USED) return DS4_MAX_EXPERT_USED;
            return (uint32_t)v;
        }
    }
    return 0u;
}

static bool flash_moe_async_handout_wait_miss_force_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_ASYNC_HANDOUT_WAIT_MISS_FORCE");
}

static bool flash_moe_preprotect_topk_enabled(void) {
    return env_flag_enabled("DS4_FLASH_MOE_PREPROTECT_TOPK");
}

static bool metal_graph_flash_moe_family_slot_ptr(
        ds4_gpu_tensor *tensor,
        uint64_t        family_bytes,
        uint64_t        slot_stride,
        int32_t         slot,
        uint32_t        slot_bank,
        uint8_t       **ptr_out) {
    if (ptr_out) *ptr_out = NULL;
    if (!tensor || !ptr_out || slot < 0 || slot >= (int32_t)slot_bank) return false;
    if (slot_stride != 0 && (uint64_t)slot > UINT64_MAX / slot_stride) return false;
    const uint64_t offset = (uint64_t)slot * slot_stride;
    const uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    if (offset > bytes || family_bytes > bytes - offset) return false;
    uint8_t *base = (uint8_t *)ds4_gpu_tensor_contents(tensor);
    if (!base && family_bytes != 0) return false;
    *ptr_out = base + offset;
    return true;
}

static bool metal_graph_flash_moe_chunk_for_slot(
        const ds4_gpu_graph *g,
        int32_t              slot,
        uint32_t            *chunk_out,
        uint32_t            *local_slot_out,
        uint32_t            *chunk_slots_out) {
    if (chunk_out) *chunk_out = 0;
    if (local_slot_out) *local_slot_out = 0;
    if (chunk_slots_out) *chunk_slots_out = 0;
    if (!g || !g->flash_chunked_mixed_bank || g->flash_chunk_slots == 0 ||
        g->flash_chunk_count == 0 ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return false;
    }
    const uint32_t uslot = (uint32_t)slot;
    const uint32_t chunk = uslot / g->flash_chunk_slots;
    if (chunk >= g->flash_chunk_count ||
        chunk >= DS4_FLASH_MOE_MAX_CHUNKS) {
        return false;
    }
    const uint32_t local = uslot - chunk * g->flash_chunk_slots;
    uint32_t slots = g->flash_slot_bank - chunk * g->flash_chunk_slots;
    if (slots > g->flash_chunk_slots) slots = g->flash_chunk_slots;
    if (local >= slots) return false;
    if (chunk_out) *chunk_out = chunk;
    if (local_slot_out) *local_slot_out = local;
    if (chunk_slots_out) *chunk_slots_out = slots;
    return true;
}

static bool metal_graph_flash_moe_direct_slot_ptrs(
        ds4_gpu_graph                    *g,
        uint32_t                          il,
        int32_t                           slot,
        uint8_t                          *dst[DS4_FLASH_FAMILY_COUNT]) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER || slot < 0) return false;
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (g->flash_per_slot_buffers) {
        if (slot >= (int32_t)g->flash_slot_bank ||
            !metal_graph_flash_moe_ensure_per_slot_buffer(g, il, slot)) {
            return false;
        }
        uint8_t *base = (uint8_t *)ds4_gpu_tensor_contents(g->flash_expert_bank[il][slot]);
        if (!base && layer->expert_stride != 0) return false;
        const uint64_t bytes = ds4_gpu_tensor_bytes(g->flash_expert_bank[il][slot]);
        if (layer->expert_stride > bytes) return false;
        for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
            if (layer->family_offset[fam] > bytes ||
                layer->family_bytes[fam] > bytes - layer->family_offset[fam]) {
                return false;
            }
            dst[fam] = base + layer->family_offset[fam];
        }
        return true;
    }
    if (g->flash_chunked_mixed_bank) {
        uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
        if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
            return false;
        }
        return
            metal_graph_flash_moe_family_slot_ptr(
                    g->flash_chunk_gate_bank[il][chunk],
                    layer->family_bytes[DS4_FLASH_FAMILY_GATE],
                    layer->expert_stride,
                    (int32_t)local_slot,
                    chunk_slots,
                    &dst[DS4_FLASH_FAMILY_GATE]) &&
            metal_graph_flash_moe_family_slot_ptr(
                    g->flash_chunk_up_bank[il][chunk],
                    layer->family_bytes[DS4_FLASH_FAMILY_UP],
                    layer->expert_stride,
                    (int32_t)local_slot,
                    chunk_slots,
                    &dst[DS4_FLASH_FAMILY_UP]) &&
            metal_graph_flash_moe_family_slot_ptr(
                    g->flash_chunk_down_bank[il][chunk],
                    layer->family_bytes[DS4_FLASH_FAMILY_DOWN],
                    layer->expert_stride,
                    (int32_t)local_slot,
                    chunk_slots,
                    &dst[DS4_FLASH_FAMILY_DOWN]);
    }
    const uint64_t gate_stride = g->flash_mixed_slot_bank ?
                                 layer->expert_stride :
                                 layer->family_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_stride = g->flash_mixed_slot_bank ?
                               layer->expert_stride :
                               layer->family_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_stride = g->flash_mixed_slot_bank ?
                                 layer->expert_stride :
                                 layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
    return
        metal_graph_flash_moe_family_slot_ptr(
                g->flash_gate_bank[il],
                layer->family_bytes[DS4_FLASH_FAMILY_GATE],
                gate_stride,
                slot,
                g->flash_slot_bank,
                &dst[DS4_FLASH_FAMILY_GATE]) &&
        metal_graph_flash_moe_family_slot_ptr(
                g->flash_up_bank[il],
                layer->family_bytes[DS4_FLASH_FAMILY_UP],
                up_stride,
                slot,
                g->flash_slot_bank,
                &dst[DS4_FLASH_FAMILY_UP]) &&
        metal_graph_flash_moe_family_slot_ptr(
                g->flash_down_bank[il],
                layer->family_bytes[DS4_FLASH_FAMILY_DOWN],
                down_stride,
                slot,
                g->flash_slot_bank,
                &dst[DS4_FLASH_FAMILY_DOWN]);
}

static uint8_t *metal_graph_flash_moe_mixed_slot_ptr(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot) {
    if (!g || !g->flash_moe ||
        il >= DS4_N_LAYER || slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return NULL;
    }
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (g->flash_per_slot_buffers) {
        if (!metal_graph_flash_moe_ensure_per_slot_buffer(g, il, slot)) return NULL;
        ds4_gpu_tensor *buf = g->flash_expert_bank[il][slot];
        if (!buf || layer->expert_stride > ds4_gpu_tensor_bytes(buf)) return NULL;
        return (uint8_t *)ds4_gpu_tensor_contents(buf);
    }
    if (g->flash_chunked_mixed_bank) {
        uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
        if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
            return NULL;
        }
        (void)chunk_slots;
        if (layer->expert_stride != 0 &&
            (uint64_t)local_slot > UINT64_MAX / layer->expert_stride) {
            return NULL;
        }
        const uint64_t offset = (uint64_t)local_slot * layer->expert_stride;
        ds4_gpu_tensor *mixed = g->flash_chunk_mixed_bank[il][chunk];
        if (!mixed) return NULL;
        const uint64_t bytes = ds4_gpu_tensor_bytes(mixed);
        if (offset > bytes || layer->expert_stride > bytes - offset) return NULL;
        uint8_t *base = (uint8_t *)ds4_gpu_tensor_contents(mixed);
        return base ? base + offset : NULL;
    }
    if (!g->flash_mixed_slot_bank) return NULL;
    if (layer->expert_stride != 0 &&
        (uint64_t)slot > UINT64_MAX / layer->expert_stride) {
        return NULL;
    }
    const uint64_t offset = (uint64_t)slot * layer->expert_stride;
    ds4_gpu_tensor *mixed = g->flash_mixed_bank[il];
    if (!mixed) return NULL;
    const uint64_t bytes = ds4_gpu_tensor_bytes(mixed);
    if (offset > bytes || layer->expert_stride > bytes - offset) return NULL;
    uint8_t *base = (uint8_t *)ds4_gpu_tensor_contents(mixed);
    return base ? base + offset : NULL;
}

static ds4_gpu_tensor *metal_graph_flash_moe_family_slot_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       fam,
        int32_t        slot) {
    if (!g || !g->flash_moe ||
        il >= DS4_N_LAYER || fam >= DS4_FLASH_FAMILY_COUNT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return NULL;
    }
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (g->flash_per_slot_buffers) {
        if (slot >= (int32_t)g->flash_slot_bank || !g->flash_expert_bank[il][slot]) return NULL;
        return ds4_gpu_tensor_view(g->flash_expert_bank[il][slot],
                                   layer->family_offset[fam],
                                   layer->family_bytes[fam]);
    }
    if (g->flash_chunked_mixed_bank) {
        uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
        if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
            return NULL;
        }
        ds4_gpu_tensor *base = NULL;
        switch (fam) {
        case DS4_FLASH_FAMILY_GATE: base = g->flash_chunk_gate_bank[il][chunk]; break;
        case DS4_FLASH_FAMILY_UP:   base = g->flash_chunk_up_bank[il][chunk]; break;
        case DS4_FLASH_FAMILY_DOWN: base = g->flash_chunk_down_bank[il][chunk]; break;
        default: return NULL;
        }
        if (!base || local_slot >= chunk_slots) return NULL;
        if (layer->expert_stride != 0 &&
            (uint64_t)local_slot > UINT64_MAX / layer->expert_stride) {
            return NULL;
        }
        return ds4_gpu_tensor_view(base,
                                   (uint64_t)local_slot * layer->expert_stride,
                                   layer->family_bytes[fam]);
    }
    ds4_gpu_tensor *base = NULL;
    switch (fam) {
    case DS4_FLASH_FAMILY_GATE: base = g->flash_gate_bank[il]; break;
    case DS4_FLASH_FAMILY_UP:   base = g->flash_up_bank[il]; break;
    case DS4_FLASH_FAMILY_DOWN: base = g->flash_down_bank[il]; break;
    default: return NULL;
    }
    if (!base) return NULL;
    const uint64_t slot_stride = g->flash_mixed_slot_bank ?
                                 layer->expert_stride :
                                 layer->family_bytes[fam];
    if (slot_stride != 0 && (uint64_t)slot > UINT64_MAX / slot_stride) {
        return NULL;
    }
    return ds4_gpu_tensor_view(base,
                               (uint64_t)slot * slot_stride,
                               layer->family_bytes[fam]);
}

static ds4_gpu_tensor *metal_graph_flash_moe_family_slot_cached_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       fam,
        int32_t        slot) {
    if (!g || !g->flash_moe ||
        il >= DS4_N_LAYER || fam >= DS4_FLASH_FAMILY_COUNT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank ||
        slot >= (int32_t)DS4_MAX_EXPERT) {
        return NULL;
    }
    if (g->flash_per_slot_buffers &&
        !metal_graph_flash_moe_ensure_per_slot_buffer(g, il, slot)) {
        return NULL;
    }
    if (!g->flash_per_expert_buffers &&
        !g->flash_per_slot_buffers &&
        g->flash_mixed_slot_bank &&
        !g->flash_chunked_mixed_bank &&
        (!g->flash_expert_gate_view[il][slot] ||
         !g->flash_expert_up_view[il][slot] ||
         !g->flash_expert_down_view[il][slot]) &&
        !metal_graph_flash_moe_init_mixed_slot_family_views(g, il, (uint32_t)slot)) {
        return NULL;
    }
    switch (fam) {
    case DS4_FLASH_FAMILY_GATE:
        return g->flash_expert_gate_view[il][slot];
    case DS4_FLASH_FAMILY_UP:
        return g->flash_expert_up_view[il][slot];
    case DS4_FLASH_FAMILY_DOWN:
        return g->flash_expert_down_view[il][slot];
    default:
        return NULL;
    }
}

static bool metal_graph_flash_moe_pread_slot_direct(
        const ds4_flash_moe_layer_sidecar *layer,
        int32_t                            true_expert,
        uint8_t                           *dst[DS4_FLASH_FAMILY_COUNT],
        int                                io_split) {
    if (!layer || !dst ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
        const uint64_t bytes = layer->family_bytes[fam];
        if (bytes == 0) continue;
        if (!dst[fam]) return false;
        uint64_t file_offset = 0;
        if (!metal_graph_flash_moe_family_file_offset(layer,
                                                      true_expert,
                                                      fam,
                                                      &file_offset)) {
            return false;
        }
        if (!flash_moe_pread_split(layer->fd,
                                   file_offset,
                                   dst[fam],
                                   bytes,
                                   io_split)) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_flash_moe_write_slot_from_buf(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot,
        const uint8_t *src_buf) {
    if (!g || !g->flash_moe || !src_buf ||
        il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (g->flash_per_slot_buffers) {
        if (!metal_graph_flash_moe_ensure_per_slot_buffer(g, il, slot)) return false;
        return ds4_gpu_tensor_write(g->flash_expert_bank[il][slot],
                                    0,
                                    src_buf,
                                    layer->expert_stride) != 0;
    }
    if (g->flash_mixed_slot_bank) {
        if (g->flash_chunked_mixed_bank) {
            uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
            if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
                return false;
            }
            (void)chunk_slots;
            if (!g->flash_chunk_mixed_bank[il][chunk] ||
                (layer->expert_stride != 0 &&
                 (uint64_t)local_slot > UINT64_MAX / layer->expert_stride)) {
                return false;
            }
            return ds4_gpu_tensor_write(g->flash_chunk_mixed_bank[il][chunk],
                                        (uint64_t)local_slot * layer->expert_stride,
                                        src_buf,
                                        layer->expert_stride) != 0;
        }
        if (!g->flash_mixed_bank[il] ||
            (layer->expert_stride != 0 &&
             (uint64_t)slot > UINT64_MAX / layer->expert_stride)) {
            return false;
        }
        return ds4_gpu_tensor_write(g->flash_mixed_bank[il],
                                    (uint64_t)slot * layer->expert_stride,
                                    src_buf,
                                    layer->expert_stride) != 0;
    }

    const uint64_t gate_bytes = layer->family_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_bytes = layer->family_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_bytes = layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
    return
        ds4_gpu_tensor_write(g->flash_gate_bank[il],
                             (uint64_t)slot * gate_bytes,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_GATE],
                             gate_bytes) != 0 &&
        ds4_gpu_tensor_write(g->flash_up_bank[il],
                             (uint64_t)slot * up_bytes,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_UP],
                             up_bytes) != 0 &&
        ds4_gpu_tensor_write(g->flash_down_bank[il],
                             (uint64_t)slot * down_bytes,
                             src_buf + layer->family_offset[DS4_FLASH_FAMILY_DOWN],
                             down_bytes) != 0;
}

static bool metal_graph_flash_moe_mark_slot_modified(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        slot) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return false;
    }

    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    if (g->flash_per_expert_buffers || g->flash_per_slot_buffers) {
        const uint32_t slot_limit =
            g->flash_per_expert_buffers ? DS4_N_EXPERT : g->flash_slot_bank;
        if (slot >= (int32_t)slot_limit || !g->flash_expert_bank[il][slot]) {
            return false;
        }
        return ds4_gpu_tensor_did_modify(g->flash_expert_bank[il][slot],
                                         0,
                                         layer->expert_stride) != 0;
    }
    if (g->flash_mixed_slot_bank) {
        if (g->flash_chunked_mixed_bank) {
            uint32_t chunk = 0, local_slot = 0, chunk_slots = 0;
            if (!metal_graph_flash_moe_chunk_for_slot(g, slot, &chunk, &local_slot, &chunk_slots)) {
                return false;
            }
            (void)chunk_slots;
            if (!g->flash_chunk_mixed_bank[il][chunk] ||
                (layer->expert_stride != 0 &&
                 (uint64_t)local_slot > UINT64_MAX / layer->expert_stride)) {
                return false;
            }
            return ds4_gpu_tensor_did_modify(g->flash_chunk_mixed_bank[il][chunk],
                                             (uint64_t)local_slot * layer->expert_stride,
                                             layer->expert_stride) != 0;
        }
        if (!g->flash_mixed_bank[il] ||
            (layer->expert_stride != 0 &&
             (uint64_t)slot > UINT64_MAX / layer->expert_stride)) {
            return false;
        }
        return ds4_gpu_tensor_did_modify(g->flash_mixed_bank[il],
                                         (uint64_t)slot * layer->expert_stride,
                                         layer->expert_stride) != 0;
    }

    const uint64_t gate_bytes = layer->family_bytes[DS4_FLASH_FAMILY_GATE];
    const uint64_t up_bytes = layer->family_bytes[DS4_FLASH_FAMILY_UP];
    const uint64_t down_bytes = layer->family_bytes[DS4_FLASH_FAMILY_DOWN];
    return
        ds4_gpu_tensor_did_modify(g->flash_gate_bank[il],
                                  (uint64_t)slot * gate_bytes,
                                  gate_bytes) != 0 &&
        ds4_gpu_tensor_did_modify(g->flash_up_bank[il],
                                  (uint64_t)slot * up_bytes,
                                  up_bytes) != 0 &&
        ds4_gpu_tensor_did_modify(g->flash_down_bank[il],
                                  (uint64_t)slot * down_bytes,
                                  down_bytes) != 0;
}

static uint32_t metal_graph_flash_moe_shared_l2_slots_for_budget(
        const ds4_gpu_graph *g,
        uint64_t             budget,
        uint32_t             fallback) {
    if (!g || !g->flash_moe) return fallback;
    const uint64_t stride = g->flash_moe->max_expert_stride;
    if (budget == 0 || stride == 0) return fallback;
    uint64_t slots = budget / stride;
    if (slots == 0) slots = 1;
    if (slots > UINT32_MAX) slots = UINT32_MAX;
    return (uint32_t)slots;
}

static uint32_t metal_graph_flash_moe_shared_l2_configured_slots(
        const ds4_gpu_graph *g,
        uint32_t             fallback) {
    const char *slots_env = getenv("DS4_FLASH_MOE_SHARED_L2_SLOTS");
    if (slots_env && slots_env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(slots_env, &end, 10);
        if (errno == 0 && end != slots_env && v > 0 && v <= UINT32_MAX) {
            return (uint32_t)v;
        }
    }

    const char *budget_envs[] = {
        "DS4_FLASH_MOE_SHARED_L2_CACHE",
        "DS4_FLASH_MOE_SHARED_L2_BYTES",
        "DS4_FLASH_MOE_DECODE_SHARED_L2_CACHE",
    };
    for (size_t i = 0; i < sizeof(budget_envs) / sizeof(budget_envs[0]); i++) {
        const char *env = getenv(budget_envs[i]);
        if (!env || !env[0]) continue;
        uint64_t budget = 0;
        if (ds4_parse_u64_suffix(env, &budget) && budget > 0) {
            return metal_graph_flash_moe_shared_l2_slots_for_budget(g, budget, fallback);
        }
    }
    return fallback;
}

static uint64_t metal_graph_flash_moe_shared_l2_key_idx(
        uint32_t il,
        int32_t  expert) {
    return (uint64_t)il * DS4_N_EXPERT + (uint32_t)expert;
}

static void metal_graph_flash_moe_shared_l2_free(ds4_gpu_graph *g) {
    if (!g) return;
    if (g->flash_shared_l2_slot_buf) {
        for (uint32_t i = 0; i < g->flash_shared_l2_slot_bank; i++) {
            free(g->flash_shared_l2_slot_buf[i]);
        }
    }
    free(g->flash_shared_l2_slot_buf);
    free(g->flash_shared_l2_slot_bytes);
    free(g->flash_shared_l2_slot_layer);
    free(g->flash_shared_l2_slot_expert);
    free(g->flash_shared_l2_expert_to_slot);
    free(g->flash_shared_l2_slot_age);
    g->flash_shared_l2_slot_bank = 0;
    g->flash_shared_l2_slot_buf = NULL;
    g->flash_shared_l2_slot_bytes = NULL;
    g->flash_shared_l2_slot_layer = NULL;
    g->flash_shared_l2_slot_expert = NULL;
    g->flash_shared_l2_expert_to_slot = NULL;
    g->flash_shared_l2_slot_age = NULL;
    g->flash_shared_l2_allocated_bytes = 0;
    g->flash_shared_l2_capacity_bytes = 0;
}

static bool metal_graph_flash_moe_shared_l2_init(
        ds4_gpu_graph *g,
        uint32_t       slots) {
    if (!g || !g->flash_moe || slots == 0) return false;
    if (g->flash_shared_l2_slot_bank == slots &&
        g->flash_shared_l2_slot_buf &&
        g->flash_shared_l2_slot_layer &&
        g->flash_shared_l2_slot_expert &&
        g->flash_shared_l2_expert_to_slot &&
        g->flash_shared_l2_slot_age) {
        return true;
    }

    metal_graph_flash_moe_shared_l2_free(g);
    if ((uint64_t)slots > SIZE_MAX / sizeof(g->flash_shared_l2_slot_buf[0]) ||
        (uint64_t)slots > SIZE_MAX / sizeof(g->flash_shared_l2_slot_bytes[0]) ||
        (uint64_t)slots > SIZE_MAX / sizeof(g->flash_shared_l2_slot_layer[0]) ||
        (uint64_t)slots > SIZE_MAX / sizeof(g->flash_shared_l2_slot_expert[0]) ||
        (uint64_t)slots > SIZE_MAX / sizeof(g->flash_shared_l2_slot_age[0]) ||
        (uint64_t)DS4_N_LAYER * DS4_N_EXPERT >
            SIZE_MAX / sizeof(g->flash_shared_l2_expert_to_slot[0])) {
        return false;
    }

    g->flash_shared_l2_slot_bank = slots;
    g->flash_shared_l2_slot_buf =
        xcalloc((size_t)slots, sizeof(g->flash_shared_l2_slot_buf[0]));
    g->flash_shared_l2_slot_bytes =
        xcalloc((size_t)slots, sizeof(g->flash_shared_l2_slot_bytes[0]));
    g->flash_shared_l2_slot_layer =
        xmalloc((size_t)slots * sizeof(g->flash_shared_l2_slot_layer[0]));
    g->flash_shared_l2_slot_expert =
        xmalloc((size_t)slots * sizeof(g->flash_shared_l2_slot_expert[0]));
    g->flash_shared_l2_slot_age =
        xcalloc((size_t)slots, sizeof(g->flash_shared_l2_slot_age[0]));
    g->flash_shared_l2_expert_to_slot =
        xmalloc((size_t)DS4_N_LAYER * DS4_N_EXPERT *
                sizeof(g->flash_shared_l2_expert_to_slot[0]));
    if (!g->flash_shared_l2_slot_buf ||
        !g->flash_shared_l2_slot_bytes ||
        !g->flash_shared_l2_slot_layer ||
        !g->flash_shared_l2_slot_expert ||
        !g->flash_shared_l2_slot_age ||
        !g->flash_shared_l2_expert_to_slot) {
        metal_graph_flash_moe_shared_l2_free(g);
        return false;
    }
    for (uint32_t i = 0; i < slots; i++) {
        g->flash_shared_l2_slot_layer[i] = -1;
        g->flash_shared_l2_slot_expert[i] = -1;
    }
    for (uint64_t i = 0; i < (uint64_t)DS4_N_LAYER * DS4_N_EXPERT; i++) {
        g->flash_shared_l2_expert_to_slot[i] = -1;
    }
    if (g->flash_moe->max_expert_stride != 0 &&
        (uint64_t)slots <= UINT64_MAX / g->flash_moe->max_expert_stride) {
        g->flash_shared_l2_capacity_bytes =
            (uint64_t)slots * g->flash_moe->max_expert_stride;
    } else {
        g->flash_shared_l2_capacity_bytes = UINT64_MAX;
    }
    fprintf(stderr,
            "ds4: Flash-MoE shared-L2 cache enabled: slots=%u capacity=%.2f GiB "
            "(CPU-backed global victim pool)\n",
            slots,
            (double)g->flash_shared_l2_capacity_bytes / 1073741824.0);
    return true;
}

static uint32_t metal_graph_flash_moe_shared_l2_pick_slot(ds4_gpu_graph *g) {
    for (uint32_t i = 0; i < g->flash_shared_l2_slot_bank; i++) {
        if (g->flash_shared_l2_slot_layer[i] < 0) return i;
    }
    uint32_t slot = 0;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < g->flash_shared_l2_slot_bank; i++) {
        if (g->flash_shared_l2_slot_age[i] < oldest) {
            oldest = g->flash_shared_l2_slot_age[i];
            slot = i;
        }
    }
    return slot;
}

static bool metal_graph_flash_moe_shared_l2_store_record(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        const uint8_t *src) {
    if (!g || !g->flash_moe || !g->flash_shared_l2_slot_bank || !src ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    const uint64_t key = metal_graph_flash_moe_shared_l2_key_idx(il, true_expert);
    uint32_t slot = UINT32_MAX;
    const int32_t existing = g->flash_shared_l2_expert_to_slot[key];
    if (existing >= 0 && existing < (int32_t)g->flash_shared_l2_slot_bank &&
        g->flash_shared_l2_slot_layer[existing] == (int32_t)il &&
        g->flash_shared_l2_slot_expert[existing] == true_expert) {
        slot = (uint32_t)existing;
    } else {
        slot = metal_graph_flash_moe_shared_l2_pick_slot(g);
    }

    const int32_t old_layer = g->flash_shared_l2_slot_layer[slot];
    const int32_t old_expert = g->flash_shared_l2_slot_expert[slot];
    if (old_layer >= 0 && old_layer < (int32_t)DS4_N_LAYER &&
        old_expert >= 0 && old_expert < (int32_t)DS4_N_EXPERT) {
        const uint64_t old_key =
            metal_graph_flash_moe_shared_l2_key_idx((uint32_t)old_layer, old_expert);
        if (g->flash_shared_l2_expert_to_slot[old_key] == (int32_t)slot) {
            g->flash_shared_l2_expert_to_slot[old_key] = -1;
        }
        if (old_layer != (int32_t)il || old_expert != true_expert) {
            g->flash_shared_l2_evictions++;
        }
    }

    if (g->flash_shared_l2_slot_bytes[slot] < layer->expert_stride) {
        uint8_t *buf =
            xrealloc(g->flash_shared_l2_slot_buf[slot], (size_t)layer->expert_stride);
        g->flash_shared_l2_slot_buf[slot] = buf;
        if (g->flash_shared_l2_allocated_bytes <=
            UINT64_MAX - (layer->expert_stride - g->flash_shared_l2_slot_bytes[slot])) {
            g->flash_shared_l2_allocated_bytes +=
                layer->expert_stride - g->flash_shared_l2_slot_bytes[slot];
        } else {
            g->flash_shared_l2_allocated_bytes = UINT64_MAX;
        }
        g->flash_shared_l2_slot_bytes[slot] = layer->expert_stride;
    }
    memcpy(g->flash_shared_l2_slot_buf[slot], src, (size_t)layer->expert_stride);
    g->flash_shared_l2_slot_layer[slot] = (int32_t)il;
    g->flash_shared_l2_slot_expert[slot] = true_expert;
    g->flash_shared_l2_expert_to_slot[key] = (int32_t)slot;
    g->flash_shared_l2_slot_age[slot] = ++g->flash_age;
    g->flash_shared_l2_installs++;
    return true;
}

static bool metal_graph_flash_moe_shared_l2_lookup(
        ds4_gpu_graph  *g,
        uint32_t        il,
        int32_t         true_expert,
        const uint8_t **src_out) {
    if (src_out) *src_out = NULL;
    if (!g || !g->flash_shared_l2_slot_bank || il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    const uint64_t key = metal_graph_flash_moe_shared_l2_key_idx(il, true_expert);
    const int32_t slot = g->flash_shared_l2_expert_to_slot[key];
    if (slot < 0 || slot >= (int32_t)g->flash_shared_l2_slot_bank ||
        g->flash_shared_l2_slot_layer[slot] != (int32_t)il ||
        g->flash_shared_l2_slot_expert[slot] != true_expert ||
        !g->flash_shared_l2_slot_buf[slot]) {
        g->flash_shared_l2_misses++;
        return false;
    }
    g->flash_shared_l2_slot_age[slot] = ++g->flash_age;
    g->flash_shared_l2_hits++;
    if (src_out) *src_out = g->flash_shared_l2_slot_buf[slot];
    return true;
}

static bool metal_graph_flash_moe_shared_l2_store_l1_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        l1_slot) {
    if (!g || !g->flash_shared_l2_slot_bank || !g->flash_mixed_slot_bank ||
        true_expert < 0 || l1_slot < 0) {
        return false;
    }
    uint8_t *src = metal_graph_flash_moe_mixed_slot_ptr(g, il, l1_slot);
    return src && metal_graph_flash_moe_shared_l2_store_record(g, il, true_expert, src);
}

static void metal_graph_flash_moe_store_evicted_l1_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        evicted,
        int32_t        slot) {
    if (!g || evicted < 0 || evicted >= (int32_t)DS4_N_EXPERT ||
        slot < 0 || slot >= (int32_t)g->flash_slot_bank) {
        return;
    }
    if (g->flash_shared_l2_slot_bank) {
        (void)metal_graph_flash_moe_shared_l2_store_l1_slot(g, il, evicted, slot);
    }
    if (g->flash_l2_slot_bank) {
        (void)metal_graph_flash_moe_l2_store_l1_slot(g, il, evicted, slot);
    }
}

typedef struct {
    uint32_t il;
    uint32_t slot;
    int32_t expert;
    uint64_t age;
} ds4_flash_moe_shared_l2_candidate;

static int ds4_flash_moe_shared_l2_candidate_cmp(const void *a, const void *b) {
    const ds4_flash_moe_shared_l2_candidate *ca =
        (const ds4_flash_moe_shared_l2_candidate *)a;
    const ds4_flash_moe_shared_l2_candidate *cb =
        (const ds4_flash_moe_shared_l2_candidate *)b;
    if (ca->age < cb->age) return 1;
    if (ca->age > cb->age) return -1;
    return 0;
}

static bool metal_graph_flash_moe_shared_l2_capture_before_shrink(
        ds4_gpu_graph *g,
        uint32_t       from_slots,
        uint32_t       target_slots) {
    if (!flash_moe_decode_shared_l2_enabled()) return true;
    if (!g || !g->flash_moe || !g->flash_mixed_slot_bank ||
        from_slots <= target_slots || g->flash_slot_bank != from_slots) {
        return true;
    }

    const uint32_t fallback_slots =
        from_slots > target_slots ? from_slots - target_slots : target_slots;
    const uint32_t shared_slots =
        metal_graph_flash_moe_shared_l2_configured_slots(g, fallback_slots);
    if (!metal_graph_flash_moe_shared_l2_init(g, shared_slots)) return false;

    const uint64_t max_candidates = (uint64_t)DS4_N_LAYER * from_slots;
    if (max_candidates > SIZE_MAX / sizeof(ds4_flash_moe_shared_l2_candidate)) {
        return false;
    }
    ds4_flash_moe_shared_l2_candidate *cand =
        xmalloc((size_t)max_candidates * sizeof(cand[0]));
    uint64_t n = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
        uint64_t *slot_age = flash_moe_slot_age(g, il);
        for (uint32_t slot = 0; slot < from_slots; slot++) {
            const int32_t expert = slot_to_expert[slot];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
            cand[n++] = (ds4_flash_moe_shared_l2_candidate){
                .il = il,
                .slot = slot,
                .expert = expert,
                .age = slot_age[slot],
            };
        }
    }
    qsort(cand,
          (size_t)n,
          sizeof(cand[0]),
          ds4_flash_moe_shared_l2_candidate_cmp);

    uint64_t captured = 0;
    const uint64_t limit =
        n < (uint64_t)g->flash_shared_l2_slot_bank ?
        n : (uint64_t)g->flash_shared_l2_slot_bank;
    for (uint64_t i = 0; i < limit; i++) {
        uint8_t *src =
            metal_graph_flash_moe_mixed_slot_ptr(g, cand[i].il, (int32_t)cand[i].slot);
        if (!src) continue;
        if (!metal_graph_flash_moe_shared_l2_store_record(g,
                                                          cand[i].il,
                                                          cand[i].expert,
                                                          src)) {
            free(cand);
            return false;
        }
        captured++;
    }
    free(cand);
    fprintf(stderr,
            "ds4: Flash-MoE shared-L2 captured %" PRIu64
            "/%" PRIu64 " hot resident prefill slots before decode shrink "
            "(global slots=%u)\n",
            captured,
            n,
            g->flash_shared_l2_slot_bank);
    return true;
}

static uint64_t metal_graph_flash_moe_l2_idx(
        const ds4_gpu_graph *g,
        uint32_t             il,
        uint32_t             slot) {
    return (uint64_t)il * g->flash_l2_slot_bank + slot;
}

static void metal_graph_flash_moe_l2_free(ds4_gpu_graph *g) {
    if (!g) return;
    const uint64_t n = (uint64_t)DS4_N_LAYER * g->flash_l2_slot_bank;
    if (g->flash_l2_slot_buf) {
        for (uint64_t i = 0; i < n; i++) free(g->flash_l2_slot_buf[i]);
    }
    free(g->flash_l2_slot_buf);
    free(g->flash_l2_slot_to_expert);
    free(g->flash_l2_expert_to_slot);
    free(g->flash_l2_slot_age);
    g->flash_l2_slot_bank = 0;
    g->flash_l2_slot_buf = NULL;
    g->flash_l2_slot_to_expert = NULL;
    g->flash_l2_expert_to_slot = NULL;
    g->flash_l2_slot_age = NULL;
    g->flash_l2_allocated_bytes = 0;
    g->flash_l2_capacity_bytes = 0;
}

static bool metal_graph_flash_moe_l2_init(ds4_gpu_graph *g, uint32_t slots) {
    if (!g || !g->flash_moe || slots == 0) return false;
    if (g->flash_l2_slot_bank == slots &&
        g->flash_l2_slot_buf &&
        g->flash_l2_slot_to_expert &&
        g->flash_l2_expert_to_slot &&
        g->flash_l2_slot_age) {
        return true;
    }

    metal_graph_flash_moe_l2_free(g);
    const uint64_t layer_slots = (uint64_t)DS4_N_LAYER * slots;
    if (layer_slots > SIZE_MAX / sizeof(g->flash_l2_slot_buf[0]) ||
        layer_slots > SIZE_MAX / sizeof(g->flash_l2_slot_to_expert[0]) ||
        layer_slots > SIZE_MAX / sizeof(g->flash_l2_slot_age[0]) ||
        (uint64_t)DS4_N_LAYER * DS4_N_EXPERT >
            SIZE_MAX / sizeof(g->flash_l2_expert_to_slot[0])) {
        return false;
    }

    g->flash_l2_slot_bank = slots;
    g->flash_l2_slot_buf =
        xcalloc((size_t)layer_slots, sizeof(g->flash_l2_slot_buf[0]));
    g->flash_l2_slot_to_expert =
        xmalloc((size_t)layer_slots * sizeof(g->flash_l2_slot_to_expert[0]));
    g->flash_l2_expert_to_slot =
        xmalloc((size_t)DS4_N_LAYER * DS4_N_EXPERT *
                sizeof(g->flash_l2_expert_to_slot[0]));
    g->flash_l2_slot_age =
        xcalloc((size_t)layer_slots, sizeof(g->flash_l2_slot_age[0]));
    if (!g->flash_l2_slot_buf || !g->flash_l2_slot_to_expert ||
        !g->flash_l2_expert_to_slot || !g->flash_l2_slot_age) {
        metal_graph_flash_moe_l2_free(g);
        return false;
    }
    for (uint64_t i = 0; i < layer_slots; i++) g->flash_l2_slot_to_expert[i] = -1;
    for (uint64_t i = 0; i < (uint64_t)DS4_N_LAYER * DS4_N_EXPERT; i++) {
        g->flash_l2_expert_to_slot[i] = -1;
    }

    uint64_t capacity = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
        if (layer->expert_stride > UINT64_MAX / (uint64_t)slots ||
            capacity > UINT64_MAX - layer->expert_stride * (uint64_t)slots) {
            capacity = UINT64_MAX;
            break;
        }
        capacity += layer->expert_stride * (uint64_t)slots;
    }
    g->flash_l2_capacity_bytes = capacity;
    fprintf(stderr,
            "ds4: Flash-MoE L2 cache enabled: layers=%u slots=%u capacity=%.2f GiB "
            "(CPU-backed, not bound in routed_moe)\n",
            (unsigned)DS4_N_LAYER,
            slots,
            (double)capacity / 1073741824.0);
    return true;
}

static int32_t *metal_graph_flash_moe_l2_slot_to_expert(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_l2_slot_to_expert + (uint64_t)il * g->flash_l2_slot_bank;
}

static int32_t *metal_graph_flash_moe_l2_expert_to_slot(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_l2_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
}

static uint64_t *metal_graph_flash_moe_l2_slot_age(ds4_gpu_graph *g, uint32_t il) {
    return g->flash_l2_slot_age + (uint64_t)il * g->flash_l2_slot_bank;
}

static uint32_t metal_graph_flash_moe_l2_pick_slot(
        ds4_gpu_graph  *g,
        const int32_t  *slot_to_expert,
        const uint64_t *slot_age) {
    for (uint32_t i = 0; i < g->flash_l2_slot_bank; i++) {
        if (slot_to_expert[i] < 0) return i;
    }
    uint32_t slot = 0;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < g->flash_l2_slot_bank; i++) {
        if (slot_age[i] < oldest) {
            oldest = slot_age[i];
            slot = i;
        }
    }
    return slot;
}

static bool metal_graph_flash_moe_l2_store_record(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        const uint8_t *src) {
    if (!g || !g->flash_moe || !g->flash_l2_slot_bank || !src ||
        il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    int32_t *slot_to_expert = metal_graph_flash_moe_l2_slot_to_expert(g, il);
    int32_t *expert_to_slot = metal_graph_flash_moe_l2_expert_to_slot(g, il);
    uint64_t *slot_age = metal_graph_flash_moe_l2_slot_age(g, il);

    uint32_t slot = UINT32_MAX;
    const int32_t existing = expert_to_slot[true_expert];
    if (existing >= 0 && existing < (int32_t)g->flash_l2_slot_bank &&
        slot_to_expert[existing] == true_expert) {
        slot = (uint32_t)existing;
    } else {
        slot = metal_graph_flash_moe_l2_pick_slot(g, slot_to_expert, slot_age);
    }

    const uint64_t idx = metal_graph_flash_moe_l2_idx(g, il, slot);
    if (!g->flash_l2_slot_buf[idx]) {
        g->flash_l2_slot_buf[idx] = xmalloc((size_t)layer->expert_stride);
        if (!g->flash_l2_slot_buf[idx]) return false;
        if (g->flash_l2_allocated_bytes <= UINT64_MAX - layer->expert_stride) {
            g->flash_l2_allocated_bytes += layer->expert_stride;
        } else {
            g->flash_l2_allocated_bytes = UINT64_MAX;
        }
    }

    const int32_t evicted = slot_to_expert[slot];
    if (evicted >= 0 && evicted < (int32_t)DS4_N_EXPERT && evicted != true_expert &&
        expert_to_slot[evicted] == (int32_t)slot) {
        expert_to_slot[evicted] = -1;
        g->flash_l2_evictions++;
    }
    memcpy(g->flash_l2_slot_buf[idx], src, (size_t)layer->expert_stride);
    slot_to_expert[slot] = true_expert;
    expert_to_slot[true_expert] = (int32_t)slot;
    slot_age[slot] = ++g->flash_age;
    g->flash_l2_installs++;
    return true;
}

static bool metal_graph_flash_moe_l2_lookup(
        ds4_gpu_graph  *g,
        uint32_t        il,
        int32_t         true_expert,
        const uint8_t **src_out) {
    if (src_out) *src_out = NULL;
    if (!g || !g->flash_l2_slot_bank || il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    int32_t *slot_to_expert = metal_graph_flash_moe_l2_slot_to_expert(g, il);
    int32_t *expert_to_slot = metal_graph_flash_moe_l2_expert_to_slot(g, il);
    uint64_t *slot_age = metal_graph_flash_moe_l2_slot_age(g, il);
    const int32_t slot = expert_to_slot[true_expert];
    if (slot < 0 || slot >= (int32_t)g->flash_l2_slot_bank ||
        slot_to_expert[slot] != true_expert) {
        g->flash_l2_misses++;
        return false;
    }
    const uint64_t idx = metal_graph_flash_moe_l2_idx(g, il, (uint32_t)slot);
    if (!g->flash_l2_slot_buf[idx]) {
        g->flash_l2_misses++;
        return false;
    }
    slot_age[slot] = ++g->flash_age;
    g->flash_l2_hits++;
    if (src_out) *src_out = g->flash_l2_slot_buf[idx];
    return true;
}

static bool metal_graph_flash_moe_l2_store_l1_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        l1_slot) {
    if (!g || !g->flash_l2_slot_bank || !g->flash_mixed_slot_bank ||
        true_expert < 0 || l1_slot < 0) {
        return false;
    }
    uint8_t *src = metal_graph_flash_moe_mixed_slot_ptr(g, il, l1_slot);
    return src && metal_graph_flash_moe_l2_store_record(g, il, true_expert, src);
}

static bool metal_graph_flash_moe_l2_capture_before_shrink(
        ds4_gpu_graph *g,
        uint32_t       from_slots,
        uint32_t       target_slots) {
    if (!flash_moe_decode_l2_enabled()) return true;
    if (!g || !g->flash_moe || !g->flash_mixed_slot_bank ||
        from_slots <= target_slots || g->flash_slot_bank != from_slots) {
        return true;
    }

    const uint32_t l2_slots = from_slots - target_slots;
    if (!metal_graph_flash_moe_l2_init(g, l2_slots)) return false;

    uint64_t captured = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        int32_t *slot_to_expert = flash_moe_slot_to_expert(g, il);
        for (uint32_t slot = 0; slot < from_slots; slot++) {
            const int32_t expert = slot_to_expert[slot];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
            uint8_t *src = metal_graph_flash_moe_mixed_slot_ptr(g, il, (int32_t)slot);
            if (!src) continue;
            if (!metal_graph_flash_moe_l2_store_record(g, il, expert, src)) {
                return false;
            }
            captured++;
        }
    }
    fprintf(stderr,
            "ds4: Flash-MoE L2 captured %" PRIu64
            " resident prefill slots before decode shrink\n",
            captured);
    return true;
}

static int32_t *metal_graph_flash_moe_gpu_l2_slot_to_expert(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return g->flash_gpu_l2_slot_to_expert +
           (uint64_t)il * g->flash_gpu_l2_slot_bank;
}

static int32_t *metal_graph_flash_moe_gpu_l2_expert_to_slot(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return g->flash_gpu_l2_expert_to_slot + (uint64_t)il * DS4_N_EXPERT;
}

static uint64_t *metal_graph_flash_moe_gpu_l2_slot_age(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return g->flash_gpu_l2_slot_age +
           (uint64_t)il * g->flash_gpu_l2_slot_bank;
}

static bool metal_graph_flash_moe_gpu_l2_init_maps(
        ds4_gpu_graph *g,
        uint32_t       slots) {
    if (!g || !g->flash_moe || slots == 0) return false;
    metal_graph_flash_moe_gpu_l2_free(g);
    const uint64_t layer_slots = (uint64_t)DS4_N_LAYER * slots;
    if (layer_slots > SIZE_MAX / sizeof(g->flash_gpu_l2_slot_to_expert[0]) ||
        layer_slots > SIZE_MAX / sizeof(g->flash_gpu_l2_slot_age[0]) ||
        (uint64_t)DS4_N_LAYER * DS4_N_EXPERT >
            SIZE_MAX / sizeof(g->flash_gpu_l2_expert_to_slot[0])) {
        return false;
    }

    g->flash_gpu_l2_slot_bank = slots;
    g->flash_gpu_l2_slot_to_expert =
        xmalloc((size_t)layer_slots * sizeof(g->flash_gpu_l2_slot_to_expert[0]));
    g->flash_gpu_l2_expert_to_slot =
        xmalloc((size_t)DS4_N_LAYER * DS4_N_EXPERT *
                sizeof(g->flash_gpu_l2_expert_to_slot[0]));
    g->flash_gpu_l2_slot_age =
        xcalloc((size_t)layer_slots, sizeof(g->flash_gpu_l2_slot_age[0]));
    if (!g->flash_gpu_l2_slot_to_expert ||
        !g->flash_gpu_l2_expert_to_slot ||
        !g->flash_gpu_l2_slot_age) {
        metal_graph_flash_moe_gpu_l2_free(g);
        return false;
    }
    for (uint64_t i = 0; i < layer_slots; i++) {
        g->flash_gpu_l2_slot_to_expert[i] = -1;
    }
    for (uint64_t i = 0; i < (uint64_t)DS4_N_LAYER * DS4_N_EXPERT; i++) {
        g->flash_gpu_l2_expert_to_slot[i] = -1;
    }
    g->flash_gpu_l2_hits = 0;
    g->flash_gpu_l2_misses = 0;
    g->flash_gpu_l2_installs = 0;
    g->flash_gpu_l2_evictions = 0;
    g->flash_gpu_l2_capacity_bytes = 0;
    return true;
}

static bool metal_graph_flash_moe_gpu_l2_capture_active_bank(
        ds4_gpu_graph *g,
        uint32_t       from_slots) {
    if (!g || !g->flash_moe || !g->flash_mixed_slot_bank ||
        g->flash_layer_slot_slab || from_slots == 0) {
        return false;
    }
    if (!metal_graph_flash_moe_gpu_l2_init_maps(g, from_slots)) return false;

    uint64_t capacity = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
        const uint64_t need = (uint64_t)from_slots * layer->expert_stride;
        if (!g->flash_mixed_bank[il] ||
            ds4_gpu_tensor_bytes(g->flash_mixed_bank[il]) < need) {
            metal_graph_flash_moe_gpu_l2_free(g);
            return false;
        }
        if (capacity > UINT64_MAX - need) {
            metal_graph_flash_moe_gpu_l2_free(g);
            return false;
        }
        capacity += need;
    }

    uint64_t captured = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        int32_t *old_slot_to_expert =
            g->flash_slot_to_expert + (uint64_t)il * from_slots;
        uint64_t *old_slot_age =
            g->flash_slot_age + (uint64_t)il * from_slots;
        int32_t *l2_slot_to_expert =
            metal_graph_flash_moe_gpu_l2_slot_to_expert(g, il);
        int32_t *l2_expert_to_slot =
            metal_graph_flash_moe_gpu_l2_expert_to_slot(g, il);
        uint64_t *l2_slot_age =
            metal_graph_flash_moe_gpu_l2_slot_age(g, il);
        for (uint32_t slot = 0; slot < from_slots; slot++) {
            const int32_t expert = old_slot_to_expert[slot];
            if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
            l2_slot_to_expert[slot] = expert;
            l2_expert_to_slot[expert] = (int32_t)slot;
            l2_slot_age[slot] = old_slot_age[slot];
            captured++;
        }

        g->flash_gpu_l2_mixed_bank[il] = g->flash_mixed_bank[il];
        g->flash_mixed_bank[il] = NULL;
        ds4_gpu_tensor_free(g->flash_down_bank[il]);
        ds4_gpu_tensor_free(g->flash_up_bank[il]);
        ds4_gpu_tensor_free(g->flash_gate_bank[il]);
        g->flash_down_bank[il] = NULL;
        g->flash_up_bank[il] = NULL;
        g->flash_gate_bank[il] = NULL;
    }
    g->flash_gpu_l2_capacity_bytes = capacity;
    fprintf(stderr,
            "ds4: Flash-MoE GPU-L2 retained full prefill bank: "
            "layers=%u slots=%u capacity=%.2f GiB captured=%" PRIu64 "\n",
            (unsigned)DS4_N_LAYER,
            from_slots,
            (double)capacity / 1073741824.0,
            captured);
    return true;
}

static uint32_t metal_graph_flash_moe_gpu_l2_pick_slot(
        ds4_gpu_graph  *g,
        const int32_t  *slot_to_expert,
        const uint64_t *slot_age) {
    for (uint32_t i = 0; i < g->flash_gpu_l2_slot_bank; i++) {
        if (slot_to_expert[i] < 0) return i;
    }
    uint32_t slot = 0;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < g->flash_gpu_l2_slot_bank; i++) {
        if (slot_age[i] < oldest) {
            oldest = slot_age[i];
            slot = i;
        }
    }
    return slot;
}

static bool metal_graph_flash_moe_gpu_l2_copy_record(
        ds4_gpu_graph *g,
        uint32_t       il,
        ds4_gpu_tensor *dst,
        uint32_t       dst_slot,
        ds4_gpu_tensor *src,
        uint32_t       src_slot) {
    if (!g || !g->flash_moe || il >= DS4_N_LAYER || !dst || !src) return false;
    const ds4_flash_moe_layer_sidecar *layer = &g->flash_moe->layer[il];
    const uint64_t bytes = layer->expert_stride;
    const uint64_t dst_off = (uint64_t)dst_slot * bytes;
    const uint64_t src_off = (uint64_t)src_slot * bytes;
    if (dst_off > ds4_gpu_tensor_bytes(dst) ||
        bytes > ds4_gpu_tensor_bytes(dst) - dst_off ||
        src_off > ds4_gpu_tensor_bytes(src) ||
        bytes > ds4_gpu_tensor_bytes(src) - src_off) {
        return false;
    }
    return ds4_gpu_tensor_copy(dst, dst_off, src, src_off, bytes) != 0;
}

static bool metal_graph_flash_moe_gpu_l2_store_l1_slot(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        l1_slot) {
    if (!g || !g->flash_gpu_l2_slot_bank || il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        l1_slot < 0 || l1_slot >= (int32_t)g->flash_slot_bank ||
        !g->flash_mixed_bank[il] || !g->flash_gpu_l2_mixed_bank[il]) {
        return false;
    }

    int32_t *slot_to_expert =
        metal_graph_flash_moe_gpu_l2_slot_to_expert(g, il);
    int32_t *expert_to_slot =
        metal_graph_flash_moe_gpu_l2_expert_to_slot(g, il);
    uint64_t *slot_age =
        metal_graph_flash_moe_gpu_l2_slot_age(g, il);
    uint32_t l2_slot = UINT32_MAX;
    const int32_t existing = expert_to_slot[true_expert];
    if (existing >= 0 && existing < (int32_t)g->flash_gpu_l2_slot_bank &&
        slot_to_expert[existing] == true_expert) {
        l2_slot = (uint32_t)existing;
    } else {
        l2_slot = metal_graph_flash_moe_gpu_l2_pick_slot(g,
                                                         slot_to_expert,
                                                         slot_age);
    }

    const int32_t evicted = slot_to_expert[l2_slot];
    if (!metal_graph_flash_moe_gpu_l2_copy_record(g,
                                                  il,
                                                  g->flash_gpu_l2_mixed_bank[il],
                                                  l2_slot,
                                                  g->flash_mixed_bank[il],
                                                  (uint32_t)l1_slot)) {
        return false;
    }
    if (evicted >= 0 && evicted < (int32_t)DS4_N_EXPERT &&
        evicted != true_expert && expert_to_slot[evicted] == (int32_t)l2_slot) {
        expert_to_slot[evicted] = -1;
        g->flash_gpu_l2_evictions++;
    }
    slot_to_expert[l2_slot] = true_expert;
    expert_to_slot[true_expert] = (int32_t)l2_slot;
    slot_age[l2_slot] = ++g->flash_age;
    g->flash_gpu_l2_installs++;
    return true;
}

static bool metal_graph_flash_moe_gpu_l2_promote_to_l1(
        ds4_gpu_graph *g,
        uint32_t       il,
        int32_t        true_expert,
        int32_t        l1_slot) {
    if (!g || !g->flash_gpu_l2_slot_bank || il >= DS4_N_LAYER ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT ||
        l1_slot < 0 || l1_slot >= (int32_t)g->flash_slot_bank ||
        !g->flash_mixed_bank[il] || !g->flash_gpu_l2_mixed_bank[il]) {
        return false;
    }

    int32_t *slot_to_expert =
        metal_graph_flash_moe_gpu_l2_slot_to_expert(g, il);
    int32_t *expert_to_slot =
        metal_graph_flash_moe_gpu_l2_expert_to_slot(g, il);
    uint64_t *slot_age =
        metal_graph_flash_moe_gpu_l2_slot_age(g, il);
    const int32_t l2_slot = expert_to_slot[true_expert];
    if (l2_slot < 0 || l2_slot >= (int32_t)g->flash_gpu_l2_slot_bank ||
        slot_to_expert[l2_slot] != true_expert) {
        g->flash_gpu_l2_misses++;
        return false;
    }

    if (!metal_graph_flash_moe_gpu_l2_copy_record(g,
                                                  il,
                                                  g->flash_mixed_bank[il],
                                                  (uint32_t)l1_slot,
                                                  g->flash_gpu_l2_mixed_bank[il],
                                                  (uint32_t)l2_slot)) {
        return false;
    }
    slot_age[l2_slot] = ++g->flash_age;
    g->flash_gpu_l2_hits++;
    return true;
}

typedef struct {
    int fd;
    uint64_t offset;
    uint64_t bytes;
    const uint8_t *memory_src;
    uint8_t *buf;
    bool buf_owned;
    bool direct_slot;
    bool direct_record;
    uint8_t *record_dst;
    int io_split;
    uint8_t *family_dst[DS4_FLASH_FAMILY_COUNT];
    uint64_t family_offset[DS4_FLASH_FAMILY_COUNT];
    uint64_t family_file_offset[DS4_FLASH_FAMILY_COUNT];
    uint64_t family_bytes[DS4_FLASH_FAMILY_COUNT];
    bool family_gather;
    int err;
    int complete;
    bool canceled;
    double pread_t0_ms;
    double pread_t1_ms;
} ds4_flash_decode_read_job;

static bool ds4_flash_decode_read_job_set_sidecar(
        ds4_flash_decode_read_job          *job,
        const ds4_flash_moe_layer_sidecar  *layer,
        int32_t                             true_expert,
        int                                 io_split) {
    if (!job || !layer ||
        true_expert < 0 || true_expert >= (int32_t)DS4_N_EXPERT) {
        return false;
    }
    uint64_t record_offset = 0;
    if (!metal_graph_flash_moe_record_file_offset(layer,
                                                  true_expert,
                                                  &record_offset)) {
        return false;
    }
    job->fd = layer->fd;
    job->offset = record_offset;
    job->bytes = layer->expert_stride;
    job->io_split = io_split;
    job->family_gather = layer->family_major;
    for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
        job->family_offset[fam] = layer->family_offset[fam];
        job->family_bytes[fam] = layer->family_bytes[fam];
        if (!metal_graph_flash_moe_family_file_offset(layer,
                                                      true_expert,
                                                      fam,
                                                      &job->family_file_offset[fam])) {
            return false;
        }
    }
    return true;
}

static bool ds4_flash_decode_read_job_set_memory_src_copy(
        ds4_flash_decode_read_job *job,
        const uint8_t             *src,
        uint64_t                   bytes) {
    if (!job || !src || bytes > SIZE_MAX) return false;
    if (job->buf_owned) free(job->buf);
    job->buf = xmalloc((size_t)bytes);
    memcpy(job->buf, src, (size_t)bytes);
    job->buf_owned = true;
    job->memory_src = job->buf;
    return true;
}

static void ds4_flash_decode_read_job_mark_complete(ds4_flash_decode_read_job *job) {
    if (!job) return;
    __atomic_store_n(&job->complete, 1, __ATOMIC_RELEASE);
}

static bool ds4_flash_decode_read_job_is_complete(const ds4_flash_decode_read_job *job) {
    if (!job) return false;
    return __atomic_load_n(&job->complete, __ATOMIC_ACQUIRE) != 0;
}

typedef struct {
    bool active;
    bool needs_sync_prepare;
    bool sequential_scratch;
    volatile int stop_requested;
    uint32_t n_loads;
    int32_t slot_ids[DS4_MAX_EXPERT_USED];
    int32_t load_expert[DS4_MAX_EXPERT_USED];
    int32_t load_slot[DS4_MAX_EXPERT_USED];
    int32_t load_evicted[DS4_MAX_EXPERT_USED];
    pthread_t thread[DS4_MAX_EXPERT_USED];
    bool thread_started[DS4_MAX_EXPERT_USED];
    ds4_flash_decode_read_job job[DS4_MAX_EXPERT_USED];
} ds4_flash_decode_prefetch;

static bool flash_moe_decode_prefetch_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH");
    return env == NULL || env[0] == '\0' || atoi(env) != 0;
}

static bool flash_moe_decode_prefetch_shared_down_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN");
    return env && env[0] && atoi(env) != 0;
}

static void *ds4_flash_decode_read_thread(void *arg) {
    ds4_flash_decode_read_job *job = (ds4_flash_decode_read_job *)arg;
    if (!job) return NULL;
    job->pread_t0_ms = now_sec() * 1000.0;
    if (job->memory_src) {
        if (job->direct_record) {
            if (!job->record_dst || job->bytes > SIZE_MAX) {
                job->err = EINVAL;
            } else {
                memcpy(job->record_dst, job->memory_src, (size_t)job->bytes);
            }
        } else if (job->direct_slot) {
            for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT && !job->err; fam++) {
                const uint64_t bytes = job->family_bytes[fam];
                const uint64_t off = job->family_offset[fam];
                if (bytes == 0) continue;
                if (!job->family_dst[fam] ||
                    off > job->bytes ||
                    bytes > job->bytes - off ||
                    bytes > SIZE_MAX) {
                    job->err = EINVAL;
                    break;
                }
                memcpy(job->family_dst[fam],
                       job->memory_src + off,
                       (size_t)bytes);
            }
        } else if (job->buf) {
            if (job->bytes > SIZE_MAX) {
                job->err = EINVAL;
            } else {
                memcpy(job->buf, job->memory_src, (size_t)job->bytes);
            }
        } else {
            job->err = EINVAL;
        }
    } else if (job->direct_record) {
        if (!job->record_dst ||
            !flash_moe_pread_split(job->fd,
                                   job->offset,
                                   job->record_dst,
                                   job->bytes,
                                   job->io_split > 0 ? job->io_split : flash_moe_cache_io_split())) {
            job->err = errno ? errno : EIO;
        }
    } else if (job->direct_slot) {
        bool ok = true;
        const int io_split = job->io_split > 0 ? job->io_split : flash_moe_cache_io_split();
        for (uint32_t fam = 0; ok && fam < DS4_FLASH_FAMILY_COUNT; fam++) {
            if (job->family_bytes[fam] == 0) continue;
            if (!job->family_dst[fam]) {
                ok = false;
                break;
            }
            ok = flash_moe_pread_split(job->fd,
                                       job->family_file_offset[fam],
                                       job->family_dst[fam],
                                       job->family_bytes[fam],
                                       io_split);
        }
        if (!ok) job->err = errno ? errno : EIO;
    } else {
        const int io_split = job->io_split > 0 ? job->io_split : flash_moe_cache_io_split();
        bool ok = true;
        if (job->family_gather) {
            if (!job->buf) {
                ok = false;
            }
            for (uint32_t fam = 0; ok && fam < DS4_FLASH_FAMILY_COUNT; fam++) {
                const uint64_t bytes = job->family_bytes[fam];
                if (bytes == 0) continue;
                if (job->family_offset[fam] > job->bytes ||
                    bytes > job->bytes - job->family_offset[fam]) {
                    ok = false;
                    break;
                }
                ok = flash_moe_pread_split(job->fd,
                                           job->family_file_offset[fam],
                                           job->buf + job->family_offset[fam],
                                           bytes,
                                           io_split);
            }
        } else {
            ok = flash_moe_pread_split(job->fd, job->offset, job->buf, job->bytes,
                                       io_split);
        }
        if (!ok) {
            job->err = errno ? errno : EIO;
        }
    }
    job->pread_t1_ms = now_sec() * 1000.0;
    if (!job->err) ds4_flash_decode_read_job_mark_complete(job);
    return NULL;
}

static void *ds4_flash_decode_scratch_prefetch_worker(void *arg) {
    ds4_flash_decode_prefetch *pf = (ds4_flash_decode_prefetch *)arg;
    if (!pf) return NULL;
    for (uint32_t i = 0; i < pf->n_loads; i++) {
        ds4_flash_decode_read_job *job = &pf->job[i];
        if (pf->stop_requested) {
            job->canceled = true;
            job->err = ECANCELED;
            continue;
        }
        job->pread_t0_ms = now_sec() * 1000.0;
        bool ok = job->buf != NULL;
        if (ok && job->family_gather) {
            for (uint32_t fam = 0; ok && fam < DS4_FLASH_FAMILY_COUNT; fam++) {
                const uint64_t bytes = job->family_bytes[fam];
                if (bytes == 0) continue;
                if (job->family_offset[fam] > job->bytes ||
                    bytes > job->bytes - job->family_offset[fam]) {
                    ok = false;
                    errno = EINVAL;
                    break;
                }
                ok = flash_moe_pread_full_interruptible(job->fd,
                                                        job->family_file_offset[fam],
                                                        job->buf + job->family_offset[fam],
                                                        bytes,
                                                        &pf->stop_requested);
            }
        } else if (ok) {
            ok = flash_moe_pread_full_interruptible(job->fd,
                                                    job->offset,
                                                    job->buf,
                                                    job->bytes,
                                                    &pf->stop_requested);
        }
        job->pread_t1_ms = now_sec() * 1000.0;
        if (!ok) {
            if (errno == ECANCELED || pf->stop_requested) {
                job->canceled = true;
                job->err = ECANCELED;
                continue;
            }
            job->err = errno ? errno : EIO;
            continue;
        }
        ds4_flash_decode_read_job_mark_complete(job);
    }
    return NULL;
}

/* Host-side pread slot pool. Bumped from 24 to 160 to give the cross-layer
 * prefetch (DS4_FLASH_MOE_XLAYER_PREFETCH) room to pre-stage a full next-layer
 * top-K (up to ~150 experts) on top of the within-layer read-ahead. Buffers are
 * plain malloc (lazy pages), so the off path that only touches ~24 slots keeps
 * the same resident footprint; the extra address space is untouched. */
#define DS4_FLASH_PREFILL_ASYNC_SLOTS 160

/* Cross-layer prefill prefetch. run_prefill_dedup keeps one persistent async
 * reader and, at the end of each layer, queues the next layer's hottest experts
 * (predicted from this layer's routed set, ~92% overlap on DSv4) so they stream
 * from SSD during the next layer's attention/dense/router compute instead of
 * stalling its MoE. Auto-policy: enabled by default for SHORT prefill
 * (n_tokens <= 6000), where the reader idles between layers so prefetch fills
 * the gap (+3-8%); off at larger context where the SSD is already saturated and
 * prefetch only adds contention (measured net-negative at 8k). Force with
 * DS4_FLASH_MOE_XLAYER_PREFETCH=0/1. */
static bool flash_moe_xlayer_prefetch_enabled(uint32_t n_tokens) {
    const char *env = getenv("DS4_FLASH_MOE_XLAYER_PREFETCH");
    if (env && env[0]) return atoi(env) != 0;
    return n_tokens > 0 && n_tokens <= 6000;
}

/* How many of the current layer's hottest experts to pre-stage for the next
 * layer. Default `dflt` (the caller passes slot_bank/2); clamped so the pool
 * keeps headroom for the within-layer read-ahead and so a single layer's queue
 * cannot exhaust the slot pool (which would block submit forever between
 * layers). Override with DS4_FLASH_MOE_XLAYER_TOPK. */
static int flash_moe_xlayer_topk(int dflt) {
    int k = dflt > 0 ? dflt : 1;
    const char *env = getenv("DS4_FLASH_MOE_XLAYER_TOPK");
    if (env && env[0]) k = atoi(env);
    if (k < 1) k = 1;
    const int cap = DS4_FLASH_PREFILL_ASYNC_SLOTS - 8;  /* leave readahead headroom */
    if (k > cap) k = cap;
    return k;
}

/* When set (default on with xlayer), pause speculative cross-layer reads during
 * the ANE expert-eval loop so they only stream in the attention/dense/router
 * window and never compete with ANE eval for memory bandwidth. Set
 * DS4_FLASH_MOE_XLAYER_ATTN_ONLY=0 to let prefetch run through eval (A/B). */
static bool flash_moe_xlayer_attn_only(void) {
    const char *env = getenv("DS4_FLASH_MOE_XLAYER_ATTN_ONLY");
    if (env && env[0]) return atoi(env) != 0;
    return true;
}

typedef enum ds4_flash_async_state {
    DS4_FLASH_ASYNC_EMPTY = 0,
    DS4_FLASH_ASYNC_QUEUED = 1,
    DS4_FLASH_ASYNC_READING = 2,
    DS4_FLASH_ASYNC_READY = 3,
    DS4_FLASH_ASYNC_ERROR = 4,
    DS4_FLASH_ASYNC_CONSUMING = 5,
} ds4_flash_async_state;

typedef struct ds4_flash_prefill_async_slot {
    ds4_flash_async_state state;
    uint32_t layer;
    int32_t expert;
    int bank;
    uint32_t refs;
    int fd;
    uint64_t offset;
    uint64_t bytes;
    const ds4_flash_moe_layer_sidecar *sidecar_layer;
    bool family_gather;
    int err;
    bool canceled;
    bool speculative;   /* cross-layer prefetch read; paused during ANE eval */
    bool any_err;       /* any chunk of this slot failed */
    uint32_t nsplit;        /* number of page-aligned io-split chunks (1 = whole) */
    uint32_t chunks_claimed; /* chunks picked up by a worker */
    uint32_t chunks_done;    /* chunks finished reading */
    double pread_t0_ms;
    double pread_t1_ms;
    uint8_t *buf;
} ds4_flash_prefill_async_slot;

#define DS4_FLASH_PREFILL_MAX_READERS 8
typedef struct ds4_flash_prefill_async_reader {
    pthread_t threads[DS4_FLASH_PREFILL_MAX_READERS];
    int n_threads;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int initialized;
    int stop;
    uint64_t buf_bytes;
    uint64_t canceled_queued;
    uint64_t canceled_reading;
    uint64_t canceled_ready;
    uint64_t canceled_finished;
    /* When set, reader threads will not START new *speculative* (cross-layer
     * prefetch) reads -- only on-demand within-layer reads proceed. Toggled on
     * during the ANE expert-eval (exec) loop so the prefetch streams the next
     * layer only in the attention/dense/router window and never competes with
     * ANE eval for unified-memory bandwidth (DS4_FLASH_MOE_XLAYER_ATTN_ONLY). */
    int xlayer_paused;
    int io_split;   /* desired page-aligned read split per expert (1 = none) */
    ds4_flash_prefill_async_slot slots[DS4_FLASH_PREFILL_ASYNC_SLOTS];
} ds4_flash_prefill_async_reader;

/* Number of parallel pread worker threads for prefill expert streaming.
 * One reader serializes the SSD at ~5-7 GB/s; this NVMe scales to ~50 GB/s
 * with several outstanding reads.  Override with DS4_FLASH_MOE_PREAD_THREADS. */
static int ds4_flash_prefill_reader_threads(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int n = 4;
    const char *env = getenv("DS4_FLASH_MOE_PREAD_THREADS");
    if (env && env[0]) n = atoi(env);
    if (n < 1) n = 1;
    if (n > DS4_FLASH_PREFILL_MAX_READERS) n = DS4_FLASH_PREFILL_MAX_READERS;
    cached = n;
    return n;
}

/* How many expert preads to keep outstanding (queued into slot buffers) ahead
 * of the in-order GPU-bank staging.  The GPU staging depth stays at the bank
 * prefetch (4 banks); this only governs read-ahead into the decoupled slot
 * buffers so multiple reader threads stay busy.  Override with
 * DS4_FLASH_MOE_ASYNC_READAHEAD; clamped to the slot count minus headroom. */
static int ds4_flash_prefill_readahead(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int d = 12;
    const char *env = getenv("DS4_FLASH_MOE_ASYNC_READAHEAD");
    if (env && env[0]) d = atoi(env);
    if (d < 1) d = 1;
    if (d > DS4_FLASH_PREFILL_ASYNC_SLOTS - 2) d = DS4_FLASH_PREFILL_ASYNC_SLOTS - 2;
    cached = d;
    return d;
}


static void *ds4_flash_prefill_async_thread(void *arg) {
    ds4_flash_prefill_async_reader *r = (ds4_flash_prefill_async_reader *)arg;
    if (!r) return NULL;
    for (;;) {
        pthread_mutex_lock(&r->mu);
        int idx = -1;
        while (!r->stop) {
            /* Claim the next unread io-split chunk of any slot that still has
             * one. A slot is claimable while it has chunks left to start and is
             * not canceled; the first claim moves QUEUED -> READING. Multiple
             * workers can claim different chunks of the same expert, so one
             * expert's read fans out across the pool (deeper NVMe queue). */
            for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
                ds4_flash_prefill_async_slot *s = &r->slots[i];
                if (s->state != DS4_FLASH_ASYNC_QUEUED &&
                    s->state != DS4_FLASH_ASYNC_READING) continue;
                if (s->canceled) continue;
                if (s->chunks_claimed >= s->nsplit) continue;
                if (r->xlayer_paused && s->speculative) continue;
                idx = i;
                break;
            }
            if (idx >= 0) break;
            pthread_cond_wait(&r->cv, &r->mu);
        }
        if (r->stop) {
            pthread_mutex_unlock(&r->mu);
            break;
        }
        ds4_flash_prefill_async_slot *slot = &r->slots[idx];
        if (slot->state == DS4_FLASH_ASYNC_QUEUED) {
            slot->state = DS4_FLASH_ASYNC_READING;
            slot->pread_t0_ms = now_sec() * 1000.0;
        }
        const uint32_t chunk = slot->chunks_claimed++;
        const int fd = slot->fd;
        const uint64_t base_offset = slot->offset;
        const uint64_t bytes = slot->bytes;
        const uint32_t nsplit = slot->nsplit;
        const bool family_gather = slot->family_gather;
        const ds4_flash_moe_layer_sidecar *sidecar_layer = slot->sidecar_layer;
        const int32_t expert = slot->expert;
        uint8_t *buf = slot->buf;
        pthread_mutex_unlock(&r->mu);

        bool ok = false;
        if (family_gather) {
            ok = chunk == 0 &&
                 metal_graph_flash_moe_read_record_to_buf(sidecar_layer,
                                                          expert,
                                                          buf,
                                                          r->io_split);
        } else {
            uint64_t coff = 0, clen = 0;
            flash_moe_io_split_range(bytes, nsplit, chunk, &coff, &clen);
            ok = flash_moe_pread_full(fd, base_offset + coff, buf + coff, clen);
        }
        const double t1 = now_sec() * 1000.0;

        pthread_mutex_lock(&r->mu);
        slot->chunks_done++;
        if (!ok) { slot->any_err = true; if (!slot->err) slot->err = errno; }
        /* A non-canceled slot finalizes when every chunk has been read; a
         * canceled slot finalizes once its already-claimed chunks drain (no new
         * chunks are claimed for it). */
        const bool all_claimed_done = slot->chunks_done == slot->chunks_claimed;
        if (slot->canceled) {
            if (all_claimed_done) {
                slot->state = DS4_FLASH_ASYNC_EMPTY;
                slot->canceled = false;
                r->canceled_finished++;
            }
        } else if (slot->chunks_claimed >= nsplit && all_claimed_done) {
            slot->pread_t1_ms = t1;
            slot->state = slot->any_err ? DS4_FLASH_ASYNC_ERROR : DS4_FLASH_ASYNC_READY;
        }
        pthread_cond_broadcast(&r->cv);
        pthread_mutex_unlock(&r->mu);
    }
    return NULL;
}

static bool ds4_flash_prefill_async_init(ds4_flash_prefill_async_reader *r,
                                         uint64_t buf_bytes) {
    if (!r || buf_bytes == 0 || buf_bytes > SIZE_MAX) return false;
    memset(r, 0, sizeof(*r));
    r->buf_bytes = buf_bytes;
    if (pthread_mutex_init(&r->mu, NULL) != 0) return false;
    if (pthread_cond_init(&r->cv, NULL) != 0) {
        pthread_mutex_destroy(&r->mu);
        return false;
    }
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        r->slots[i].state = DS4_FLASH_ASYNC_EMPTY;
        r->slots[i].buf = xmalloc((size_t)buf_bytes);
    }
    r->initialized = 1;
    r->io_split = flash_moe_prefill_io_split();  /* prefill reader uses prefill split */
    r->n_threads = ds4_flash_prefill_reader_threads();
    int spawned = 0;
    for (int i = 0; i < r->n_threads; i++) {
        if (pthread_create(&r->threads[i], NULL, ds4_flash_prefill_async_thread, r) != 0) break;
        spawned++;
    }
    if (spawned == 0) {
        r->initialized = 0;
        for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
            free(r->slots[i].buf);
            r->slots[i].buf = NULL;
        }
        pthread_cond_destroy(&r->cv);
        pthread_mutex_destroy(&r->mu);
        return false;
    }
    r->n_threads = spawned;  /* honor however many actually started */
    return true;
}

static void ds4_flash_prefill_async_destroy(ds4_flash_prefill_async_reader *r) {
    if (!r || !r->initialized) return;
    pthread_mutex_lock(&r->mu);
    r->stop = 1;
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
    for (int i = 0; i < r->n_threads; i++) pthread_join(r->threads[i], NULL);
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        free(r->slots[i].buf);
        r->slots[i].buf = NULL;
    }
    pthread_cond_destroy(&r->cv);
    pthread_mutex_destroy(&r->mu);
    r->initialized = 0;
}

static int ds4_flash_prefill_async_find_locked(ds4_flash_prefill_async_reader *r,
                                               uint32_t layer,
                                               int32_t expert,
                                               int bank) {
    /* Match on (layer, expert) only, ignoring bank. Each expert appears at most
     * once in a layer's unique[] list, so within a layer this is identical to
     * matching the bank too. Ignoring bank is what lets a cross-layer prefetch
     * (submitted before the layer runs, with a placeholder bank) be consumed
     * regardless of the bank the layer ultimately assigns; slot->bank is unused
     * for the upload, which takes its bank from the caller's parameter. */
    (void)bank;
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        ds4_flash_prefill_async_slot *slot = &r->slots[i];
        if (slot->state != DS4_FLASH_ASYNC_EMPTY &&
            slot->layer == layer &&
            slot->expert == expert) {
            return i;
        }
    }
    return -1;
}

/* Cancel every non-empty slot (queued/reading/ready/error), reclaiming the pool.
 * Used at a layer boundary to drop mispredicted cross-layer prefetches before
 * queuing the next layer's set, so the speculative reads can't exhaust slots. */
static void ds4_flash_prefill_async_cancel_all(ds4_flash_prefill_async_reader *r) {
    if (!r || !r->initialized) return;
    pthread_mutex_lock(&r->mu);
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        ds4_flash_prefill_async_slot *slot = &r->slots[i];
        switch (slot->state) {
        case DS4_FLASH_ASYNC_QUEUED:
        case DS4_FLASH_ASYNC_ERROR:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_queued++;
            break;
        case DS4_FLASH_ASYNC_READING:
            slot->canceled = true;   /* reader thread discards on completion */
            r->canceled_reading++;
            break;
        case DS4_FLASH_ASYNC_READY:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_ready++;
            break;
        case DS4_FLASH_ASYNC_CONSUMING:
        case DS4_FLASH_ASYNC_EMPTY:
        default:
            break;
        }
    }
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

static void ds4_flash_prefill_async_cancel_expert(
        ds4_flash_prefill_async_reader *r,
        uint32_t                        layer,
        int32_t                         expert) {
    if (!r || !r->initialized ||
        expert < 0 || expert >= (int32_t)DS4_N_EXPERT) {
        return;
    }
    pthread_mutex_lock(&r->mu);
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        ds4_flash_prefill_async_slot *slot = &r->slots[i];
        if (slot->state == DS4_FLASH_ASYNC_EMPTY ||
            slot->layer != layer ||
            slot->expert != expert) {
            continue;
        }
        switch (slot->state) {
        case DS4_FLASH_ASYNC_QUEUED:
        case DS4_FLASH_ASYNC_ERROR:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_queued++;
            break;
        case DS4_FLASH_ASYNC_READING:
            slot->canceled = true;
            r->canceled_reading++;
            break;
        case DS4_FLASH_ASYNC_READY:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_ready++;
            break;
        case DS4_FLASH_ASYNC_CONSUMING:
        case DS4_FLASH_ASYNC_EMPTY:
        default:
            break;
        }
    }
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

/* Gate the cross-layer prefetch the moment the layer's real routing is known:
 * cancel every queued/in-flight speculative read for `layer` whose expert was
 * NOT actually routed (counts[expert]==0). The reads were queued speculatively
 * from the previous layer's routing; once this layer's router has run the
 * mispredictions are dead weight stealing SSD/memory bandwidth from the experts
 * that ARE needed, so drop them immediately. Reads that hit (counts>0) are kept
 * and consumed normally. counts[] is indexed by expert id, length DS4_N_EXPERT. */
static void ds4_flash_prefill_async_cancel_layer_mispredicted(
        ds4_flash_prefill_async_reader *r,
        uint32_t                        layer,
        const int32_t                  *counts) {
    if (!r || !r->initialized || !counts) return;
    pthread_mutex_lock(&r->mu);
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        ds4_flash_prefill_async_slot *slot = &r->slots[i];
        if (slot->state == DS4_FLASH_ASYNC_EMPTY ||
            slot->state == DS4_FLASH_ASYNC_CONSUMING ||
            slot->layer != layer) {
            continue;
        }
        const int32_t e = slot->expert;
        if (e >= 0 && e < (int32_t)DS4_N_EXPERT && counts[e] > 0) {
            continue;  /* expert is actually routed this layer: keep it */
        }
        switch (slot->state) {
        case DS4_FLASH_ASYNC_QUEUED:
        case DS4_FLASH_ASYNC_ERROR:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_queued++;
            break;
        case DS4_FLASH_ASYNC_READING:
            slot->canceled = true;
            r->canceled_reading++;
            break;
        case DS4_FLASH_ASYNC_READY:
            slot->state = DS4_FLASH_ASYNC_EMPTY;
            slot->canceled = false;
            r->canceled_ready++;
            break;
        default:
            break;
        }
    }
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

static bool ds4_flash_prefill_async_submit(ds4_flash_prefill_async_reader *r,
                                           uint32_t layer,
                                           int32_t expert,
                                           int bank,
                                           uint32_t refs,
                                           int fd,
                                           uint64_t offset,
                                           uint64_t bytes,
                                           const ds4_flash_moe_layer_sidecar *sidecar_layer,
                                           bool speculative) {
    if (!r || !r->initialized || bytes == 0 || bytes > r->buf_bytes) return false;
    pthread_mutex_lock(&r->mu);
    const int found = ds4_flash_prefill_async_find_locked(r, layer, expert, bank);
    if (found >= 0) {
        /* Already queued/reading/ready. A non-speculative (on-demand) request
         * must promote a speculative slot so a paused reader serves it now. */
        if (!speculative && r->slots[found].speculative) {
            r->slots[found].speculative = false;
            pthread_cond_broadcast(&r->cv);
        }
        pthread_mutex_unlock(&r->mu);
        return true;
    }
    int idx = -1;
    while (idx < 0) {
        for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
            if (r->slots[i].state == DS4_FLASH_ASYNC_EMPTY) {
                idx = i;
                break;
            }
        }
        if (idx >= 0) break;
        pthread_cond_wait(&r->cv, &r->mu);
    }
    ds4_flash_prefill_async_slot *slot = &r->slots[idx];
    slot->state = DS4_FLASH_ASYNC_QUEUED;
    slot->layer = layer;
    slot->expert = expert;
    slot->bank = bank;
    slot->refs = refs;
    slot->fd = fd;
    slot->offset = offset;
    slot->bytes = bytes;
    slot->sidecar_layer = sidecar_layer;
    slot->family_gather = sidecar_layer && sidecar_layer->family_major;
    slot->err = 0;
    slot->canceled = false;
    slot->speculative = speculative;
    slot->any_err = false;
    slot->nsplit = slot->family_gather ? 1u : flash_moe_active_io_split(bytes, r->io_split);
    slot->chunks_claimed = 0;
    slot->chunks_done = 0;
    slot->pread_t0_ms = 0.0;
    slot->pread_t1_ms = 0.0;
    pthread_cond_broadcast(&r->cv);
    pthread_mutex_unlock(&r->mu);
    return true;
}

/* Toggle the speculative-read pause (held during ANE expert eval). */
static void ds4_flash_prefill_async_set_paused(ds4_flash_prefill_async_reader *r,
                                               int paused) {
    if (!r || !r->initialized) return;
    pthread_mutex_lock(&r->mu);
    r->xlayer_paused = paused;
    if (!paused) pthread_cond_broadcast(&r->cv);  /* wake readers to resume */
    pthread_mutex_unlock(&r->mu);
}
