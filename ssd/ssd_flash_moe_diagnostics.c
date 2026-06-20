/* =========================================================================
 * ssd_flash_moe_diagnostics.c - Flash-MoE diagnostics and tracing helpers.
 * =========================================================================
 *
 * Included by ssd_flash_moe_streaming.c to keep private graph/runtime state in
 * one translation unit while separating debug-only support from streaming logic.
 */

static bool backend_diagnostic_logs_suppressed(void) {
    if (env_flag_enabled("DS4_AGENT_ALLOW_BACKEND_STATS")) return false;
    return env_flag_enabled("DS4_AGENT_SUPPRESS_BACKEND_LOGS");
}

static void ds4_apply_no_int8_paths(void) {
    static bool announced = false;
    ds4_setenv_override("DS4_NO_INT8", "1");

    ds4_setenv_override("DS4_FLASH_MOE_ANE_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_PIPELINE_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_OVERLAP_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_OVERLAP_SCHEDULER", "0");
    ds4_setenv_override("DS4_FLASH_MOE_HYBRID_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_CONCURRENT_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_HYBRID_CONCURRENT_PREFILL", "0");

    ds4_setenv_override("DS4_FLASH_MOE_MPP_INT8_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_MPP_INT8_ACT", "0");
    ds4_setenv_override("DS4_FLASH_MOE_MPP_I8I8_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_MPP_I8I8_TILED_FUSED_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_MPP_I8I8_FULL_FUSED_PREFILL", "0");

    ds4_setenv_override("DS4_FLASH_MOE_ANE_FP16W", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_FP16X_INT8W", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_I8I8_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_I8I8_FUSED_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_I8I8_FULL_FUSED_PREFILL", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_SHARED_EXPERT", "0");
    ds4_setenv_override("DS4_SHARED_EXPERT_ANE_I8I8", "0");
    ds4_setenv_override("DS4_FLASH_MOE_ANE_OUTPUT_PROJ", "0");
    ds4_setenv_override("DS4_GPU_DENSE_I8", "0");

    ds4_setenv_override("DS4_RESIDENT_MOE_MPP_INT8_PREFILL", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_NAX_INT8_PREFILL", "0");
    ds4_setenv_override("DS4_RESIDENT_MPP_INT8_PREFILL", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_MPP_PAIRROW_BRIDGE", "1");
    ds4_setenv_override("DS4_RESIDENT_MOE_NAX_FULL_FUSED", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_ANE_HYBRID", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_ANE_NAX_HYBRID", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_ANE_ALU_HYBRID", "0");
    ds4_setenv_override("DS4_RESIDENT_MOE_NAX_HALF", "1");

    if (!announced && !backend_diagnostic_logs_suppressed()) {
        fprintf(stderr,
                "ds4: --no-int8 active: disabled int8 dense/NAX/Flash-MoE/ANE paths; "
                "resident routing will prefer NAX-half where safe, GPU otherwise\n");
        announced = true;
    }
}

static bool backend_stats_logs_enabled(void) {
    if (backend_diagnostic_logs_suppressed()) return false;
    return env_flag_enabled("DS4_AGENT_ALLOW_BACKEND_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_PROFILE") ||
           env_flag_enabled("DS4_FLASH_MOE_STAGE_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_SCHED_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_HYBRID_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_CONCURRENT_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_ANE_PIPELINE_STATS") ||
           env_flag_enabled("DS4_FLASH_MOE_ANE_STATS") ||
           env_flag_enabled("DS4_RESIDENT_MOE_MPP_STATS") ||
           env_flag_enabled("DS4_METAL_GRAPH_PREFILL_PROFILE") ||
           env_flag_enabled("DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE");
}

typedef struct {
    uint32_t pos;
    uint32_t layer;
    int32_t expert[DS4_MAX_EXPERT_USED];
} ds4_flash_decode_oracle_row;

static int cmp_decode_oracle_row(const void *a, const void *b) {
    const ds4_flash_decode_oracle_row *ra = (const ds4_flash_decode_oracle_row *)a;
    const ds4_flash_decode_oracle_row *rb = (const ds4_flash_decode_oracle_row *)b;
    if (ra->pos != rb->pos) return ra->pos < rb->pos ? -1 : 1;
    if (ra->layer != rb->layer) return ra->layer < rb->layer ? -1 : 1;
    return 0;
}

static void flash_moe_decode_trace_record(
        uint32_t       pos,
        uint32_t       il,
        const int32_t *true_ids) {
    const char *path = getenv("DS4_FLASH_MOE_DECODE_TRACE_OUT");
    if (!path || !path[0] || !true_ids) return;

    static FILE *fp = NULL;
    static bool opened = false;
    if (!opened) {
        fp = fopen(path, "w");
        if (!fp) {
            fprintf(stderr,
                    "ds4: failed to open DS4_FLASH_MOE_DECODE_TRACE_OUT=%s: %s\n",
                    path,
                    strerror(errno));
        }
        opened = true;
    }
    if (!fp) return;

    fprintf(fp, "%u %u", pos, il);
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    for (uint32_t k = 0; k < active_expert_used; k++) {
        fprintf(fp, " %d", true_ids[k]);
    }
    fputc('\n', fp);
    fflush(fp);
}

static bool flash_moe_decode_oracle_lookup(
        uint32_t pos,
        uint32_t il,
        int32_t *true_ids_out) {
    const char *path = getenv("DS4_FLASH_MOE_DECODE_ORACLE_IN");
    if (!path || !path[0] || !true_ids_out) return false;

    static bool loaded = false;
    static ds4_flash_decode_oracle_row *rows = NULL;
    static size_t n_rows = 0;
    if (!loaded) {
        FILE *fp = fopen(path, "r");
        if (!fp) {
            fprintf(stderr,
                    "ds4: failed to open DS4_FLASH_MOE_DECODE_ORACLE_IN=%s: %s\n",
                    path,
                    strerror(errno));
            loaded = true;
            return false;
        }
        size_t cap = 0;
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            ds4_flash_decode_oracle_row row;
            memset(&row, 0, sizeof(row));
            char *p = line;
            errno = 0;
            unsigned long pos_ul = strtoul(p, &p, 10);
            if (errno != 0 || p == line) continue;
            errno = 0;
            unsigned long layer_ul = strtoul(p, &p, 10);
            if (errno != 0) continue;
            row.pos = (uint32_t)pos_ul;
            row.layer = (uint32_t)layer_ul;
            const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
            bool row_ok = true;
            for (uint32_t k = 0; k < active_expert_used; k++) {
                errno = 0;
                char *endp = NULL;
                long expert = strtol(p, &endp, 10);
                if (errno != 0 || endp == p) {
                    row_ok = false;
                    break;
                }
                p = endp;
                row.expert[k] = (int32_t)expert;
            }
            if (!row_ok) continue;
            if (n_rows == cap) {
                cap = cap ? cap * 2u : 1024u;
                rows = (ds4_flash_decode_oracle_row *)xrealloc(rows, cap * sizeof(rows[0]));
            }
            rows[n_rows++] = row;
        }
        fclose(fp);
        if (n_rows) qsort(rows, n_rows, sizeof(rows[0]), cmp_decode_oracle_row);
        fprintf(stderr,
                "ds4: Flash-MoE decode oracle loaded %zu rows from %s\n",
                n_rows,
                path);
        loaded = true;
    }
    if (!rows || n_rows == 0) return false;

    ds4_flash_decode_oracle_row key;
    memset(&key, 0, sizeof(key));
    key.pos = pos;
    key.layer = il;
    ds4_flash_decode_oracle_row *found =
        (ds4_flash_decode_oracle_row *)bsearch(&key,
                                               rows,
                                               n_rows,
                                               sizeof(rows[0]),
                                               cmp_decode_oracle_row);
    if (!found) return false;
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    for (uint32_t k = 0; k < active_expert_used; k++) {
        true_ids_out[k] = found->expert[k];
    }
    return true;
}

static int cmp_i32_asc(const void *a, const void *b) {
    const int32_t ia = *(const int32_t *)a;
    const int32_t ib = *(const int32_t *)b;
    return (ia > ib) - (ia < ib);
}

static FILE *flash_moe_hist_csv(void) {
    static int initialized = 0;
    static FILE *fp = NULL;
    if (initialized) return fp;
    initialized = 1;

    const char *path = getenv("DS4_FLASH_MOE_HIST_CSV");
    if (!path || !path[0]) return NULL;

    fp = fopen(path, "ab");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open DS4_FLASH_MOE_HIST_CSV=%s: %s\n",
                path, strerror(errno));
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) == 0 && ftell(fp) == 0) {
        fprintf(fp, "seq,layer,tokens,refs,unique,expert,expert_refs\n");
    }
    setvbuf(fp, NULL, _IOLBF, 0);
    return fp;
}

static void flash_moe_log_prefill_hist(uint32_t       il,
                                       uint32_t       n_tokens,
                                       uint64_t       n_pairs,
                                       uint32_t       n_unique,
                                       const int32_t *unique,
                                       const int32_t *counts) {
    FILE *fp = flash_moe_hist_csv();
    if (!fp) return;

    static uint64_t seq = 0;
    const uint64_t cur = ++seq;
    for (uint32_t i = 0; i < n_unique; i++) {
        const int32_t expert = unique[i];
        if (expert < 0 || expert >= (int32_t)DS4_N_EXPERT) continue;
        fprintf(fp, "%" PRIu64 ",%u,%u,%" PRIu64 ",%u,%d,%d\n",
                cur, il, n_tokens, n_pairs, n_unique, expert, counts[expert]);
    }
}

static void metal_graph_flash_moe_trace_session_sync(
        const char *path,
        int         checkpoint_len,
        int         prompt_len,
        int         suffix,
        uint32_t    resume_min) {
    if (!flash_moe_reset_slot_cache_after_prefill_enabled() &&
        !flash_moe_realloc_slot_bank_after_prefill_enabled() &&
        !flash_moe_restore_slot_bank_after_prefill_enabled() &&
        !env_flag_enabled("DS4_SESSION_SYNC_TRACE")) {
        return;
    }
    fprintf(stderr,
            "ds4: session sync path=%s checkpoint=%d prompt=%d suffix=%d resume-min=%u reset-after-prefill=%s realloc-after-prefill=%s restore-after-prefill=%s\n",
            path && path[0] ? path : "unknown",
            checkpoint_len,
            prompt_len,
            suffix,
            resume_min,
            flash_moe_reset_slot_cache_after_prefill_enabled() ? "on" : "off",
            flash_moe_realloc_slot_bank_after_prefill_enabled() ? "on" : "off",
            flash_moe_restore_slot_bank_after_prefill_enabled() ? "on" : "off");
}

static bool metal_graph_prefill_verbose_trace_enabled(void) {
    return env_flag_enabled("DS4_SESSION_SYNC_TRACE_VERBOSE") ||
           env_flag_enabled("DS4_PREFILL_VERBOSE_TRACE") ||
           env_flag_enabled("DS4_PREFILL_PHASE_TRACE");
}

static void metal_graph_prefill_trace_emit(const char *line, size_t len) {
    if (!line || len == 0) return;
    (void)fwrite(line, 1, len, stderr);
    fflush(stderr);

    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static int fd = -2;
    pthread_mutex_lock(&mu);
    if (fd == -2) {
        const char *path = getenv("DS4_PREFILL_TRACE_LOG");
        fd = (path && path[0]) ?
            open(path, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
    }
    if (fd >= 0) {
        (void)write(fd, line, len);
    }
    pthread_mutex_unlock(&mu);
}

static void metal_graph_prefill_trace_phase(
        const char *scope,
        const char *phase,
        const char *edge,
        uint32_t    start,
        uint32_t    n_tokens,
        int         prompt_len,
        double      t0) {
    if (!metal_graph_prefill_verbose_trace_enabled()) return;
    const double now = now_sec();
    char line[512];
    const int n = snprintf(
            line,
            sizeof(line),
            "ds4: prefill trace scope=%s phase=%s %s start=%u n_tokens=%u prompt=%d elapsed=%.3f ms\n",
            scope && scope[0] ? scope : "prefill",
            phase && phase[0] ? phase : "unknown",
            edge && edge[0] ? edge : "mark",
            start,
            n_tokens,
            prompt_len,
            t0 > 0.0 ? (now - t0) * 1000.0 : 0.0);
    if (n > 0) metal_graph_prefill_trace_emit(line, (size_t)n < sizeof(line) ? (size_t)n : strlen(line));
}

static void metal_graph_prefill_trace_ane_decision(
        const char *scope,
        const char *edge,
        uint32_t    start,
        uint32_t    n_tokens,
        int         prompt_len,
        bool        flash_ane_env,
        bool        flash_ane_executable,
        bool        resident_ane_env,
        bool        token_backend_ane,
        int         result,
        double      t0) {
    if (!metal_graph_prefill_verbose_trace_enabled()) return;
    const double now = now_sec();
    char line[640];
    const int n = snprintf(
            line,
            sizeof(line),
            "ds4: prefill trace scope=%s phase=ANE-precompile %s start=%u n_tokens=%u prompt=%d flash_ane=%s flash_ane_executable=%s resident_ane=%s token_backend_ane=%s result=%d elapsed=%.3f ms\n",
            scope && scope[0] ? scope : "prefill",
            edge && edge[0] ? edge : "mark",
            start,
            n_tokens,
            prompt_len,
            flash_ane_env ? "on" : "off",
            flash_ane_executable ? "on" : "off",
            resident_ane_env ? "on" : "off",
            token_backend_ane ? "on" : "off",
            result,
            t0 > 0.0 ? (now - t0) * 1000.0 : 0.0);
    if (n > 0) metal_graph_prefill_trace_emit(line, (size_t)n < sizeof(line) ? (size_t)n : strlen(line));
}
