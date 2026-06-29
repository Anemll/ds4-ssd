/* DSpark draft loader, draft runtime, and verifier.
 *
 * This file is intentionally included by ds4.c for now so the DSpark path can
 * use the existing internal Metal graph helpers while keeping ds4.c readable.
 */

/* DSpark package loader. */
typedef enum {
    DS4_DSPARK_REC_NONE = 0,
    DS4_DSPARK_REC_F16,
    DS4_DSPARK_REC_F32,
    DS4_DSPARK_REC_FP8_E4M3,
    DS4_DSPARK_REC_MXFP4_NATIVE,
    DS4_DSPARK_REC_COUNT,
} ds4_dspark_record_kind;

typedef struct {
    char name[64];
    char path[PATH_MAX];
    int fd;
    uint64_t size;
    const uint8_t *map;
} ds4_dspark_file;

typedef struct {
    char name[128];
    char file[64];
    char storage_layout[64];
    int file_index;
    int layer;
    ds4_dspark_record_kind kind;
    bool expert_major;
    uint64_t shape[3];
    int ndim;
    uint64_t scale_shape[3];
    int scale_ndim;
    uint64_t offset;
    uint64_t bytes;
    uint64_t plane_data_offset;
    uint64_t plane_data_bytes;
    uint64_t plane_scale_offset;
    uint64_t plane_scale_bytes;
    uint64_t plane_data_bytes_per_expert;
    uint64_t plane_scale_bytes_per_expert;
    int expert_count;
} ds4_dspark_record;

struct ds4_dspark_draft {
    char *path;
    char *manifest_path;
    int hidden_size;
    int layer_count;
    int expert_count;
    int expert_used_count;
    int vocab_size;
    int block_size;
    int draft_layer_count;
    int draft_layer_ids[8];
    int target_layer_count;
    int target_layer_ids[8];
    int markov_rank;
    int window_size;
    int noise_token_id;
    int verify_budget;
    char scheduler[16];
    float conf_threshold;
    ds4_dspark_file files[8];
    int file_count;
    ds4_dspark_record records[128];
    int record_count;
    int record_kind_count[DS4_DSPARK_REC_COUNT];
    bool loaded;
    bool inference_ready;
};

static bool ds4_dspark_is_loaded(const ds4_dspark_draft *d) {
    return d && d->loaded;
}

static void ds4_dspark_enable_default_fast_verifier(void) {
    ds4_setenv_default("DS4_DSPARK_DECODEN_ATTN_FFN_BATCH", "1");
    ds4_setenv_default("DS4_DSPARK_DECODEN_VERIFY", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_INDEX_COMP_ROWS", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_ROW_ROUTER", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_ROW_ROUTED", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_ROW_ROUTED_DECODE2_ROWS", "1");
    ds4_setenv_default("DS4_DSPARK_HYBRID_ROW_SHARED", "1");
    if (!env_flag_enabled("DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE")) {
        ds4_setenv_default("DS4_DSPARK_ATTN_VARMAP_ROWS", "1");
    }
    if (!env_flag_enabled("DS4_DSPARK_ORDERED_MOE_SUM_DISABLE")) {
        ds4_setenv_default("DS4_DSPARK_ORDERED_MOE_SUM", "1");
    }
}

static bool ds4_read_text_file(const char *path, char **out, size_t *len_out) {
    if (out) *out = NULL;
    if (len_out) *len_out = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return false;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return false;
    }
    rewind(fp);
    char *buf = xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        fclose(fp);
        return false;
    }
    fclose(fp);
    buf[n] = '\0';
    if (out) *out = buf;
    else free(buf);
    if (len_out) *len_out = (size_t)n;
    return true;
}

static const char *dspark_json_find_key(const char *begin, const char *end, const char *key) {
    char pat[96];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const size_t n = strlen(pat);
    for (const char *p = begin; p && p + n <= end; p = strstr(p + 1, pat)) {
        p = strstr(p, pat);
        if (!p || p + n > end) return NULL;
        const char *q = p + n;
        while (q < end && isspace((unsigned char)*q)) q++;
        if (q < end && *q == ':') return q + 1;
    }
    return NULL;
}

static bool dspark_json_i64(const char *begin, const char *end, const char *key, int64_t *out) {
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || (!isdigit((unsigned char)*p) && *p != '-')) return false;
    errno = 0;
    char *stop = NULL;
    long long v = strtoll(p, &stop, 10);
    if (errno != 0 || stop == p || stop > end) return false;
    *out = (int64_t)v;
    return true;
}

static bool dspark_json_string_value(const char *begin, const char *end, const char *key,
                                     char *out, size_t out_cap) {
    if (!out || out_cap == 0) return false;
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '"') return false;
    p++;
    size_t n = 0;
    while (p < end && *p != '"') {
        char c = *p++;
        if (c == '\\') {
            if (p >= end) return false;
            c = *p++;
            switch (c) {
            case '"': break;
            case '\\': break;
            case '/': break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            default: break;
            }
        }
        if (n + 1 < out_cap) out[n++] = c;
    }
    if (p >= end || *p != '"') return false;
    out[n] = '\0';
    return true;
}

static bool dspark_json_int_array(const char *begin, const char *end, const char *key,
                                  int *out, int out_cap, int *count_out) {
    if (count_out) *count_out = 0;
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '[') return false;
    p++;
    int n = 0;
    while (p < end) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == ']') {
            if (count_out) *count_out = n;
            return true;
        }
        if (n >= out_cap) return false;
        errno = 0;
        char *stop = NULL;
        long v = strtol(p, &stop, 10);
        if (errno != 0 || stop == p || stop > end || v < 0 || v > INT_MAX) return false;
        out[n++] = (int)v;
        p = stop;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == ',') {
            p++;
            continue;
        }
        if (p < end && *p == ']') {
            if (count_out) *count_out = n;
            return true;
        }
        return false;
    }
    return false;
}

static bool dspark_json_int_required(const char *json, size_t json_len,
                                     const char *key, int *out) {
    int64_t v = 0;
    if (!dspark_json_i64(json, json + json_len, key, &v) || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4: DSpark manifest missing/invalid integer field: %s\n", key);
        return false;
    }
    *out = (int)v;
    return true;
}

static bool dspark_json_u64(const char *begin, const char *end,
                            const char *key, uint64_t *out) {
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || !isdigit((unsigned char)*p)) return false;
    errno = 0;
    char *stop = NULL;
    unsigned long long v = strtoull(p, &stop, 10);
    if (errno != 0 || stop == p || stop > end) return false;
    *out = (uint64_t)v;
    return true;
}

static bool dspark_json_bool(const char *begin, const char *end,
                             const char *key, bool *out) {
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p + 4 <= end && memcmp(p, "true", 4) == 0) {
        *out = true;
        return true;
    }
    if (p + 5 <= end && memcmp(p, "false", 5) == 0) {
        *out = false;
        return true;
    }
    return false;
}

static bool dspark_json_u64_array(const char *begin, const char *end, const char *key,
                                  uint64_t *out, int out_cap, int *count_out) {
    if (count_out) *count_out = 0;
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '[') return false;
    p++;
    int n = 0;
    while (p < end) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == ']') {
            if (count_out) *count_out = n;
            return true;
        }
        if (n >= out_cap) return false;
        errno = 0;
        char *stop = NULL;
        unsigned long long v = strtoull(p, &stop, 10);
        if (errno != 0 || stop == p || stop > end) return false;
        out[n++] = (uint64_t)v;
        p = stop;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == ',') {
            p++;
            continue;
        }
        if (p < end && *p == ']') {
            if (count_out) *count_out = n;
            return true;
        }
        return false;
    }
    return false;
}

static const char *dspark_json_match_container(const char *p, const char *end,
                                               char open_ch, char close_ch) {
    if (!p || p >= end || *p != open_ch) return NULL;
    int depth = 0;
    bool in_string = false;
    bool esc = false;
    for (const char *q = p; q < end; q++) {
        const char c = *q;
        if (in_string) {
            if (esc) {
                esc = false;
            } else if (c == '\\') {
                esc = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open_ch) {
            depth++;
        } else if (c == close_ch) {
            depth--;
            if (depth == 0) return q + 1;
        }
    }
    return NULL;
}

static bool dspark_json_array_range(const char *begin, const char *end,
                                    const char *key,
                                    const char **arr_begin,
                                    const char **arr_end) {
    const char *p = dspark_json_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '[') return false;
    const char *q = dspark_json_match_container(p, end, '[', ']');
    if (!q) return false;
    *arr_begin = p + 1;
    *arr_end = q - 1;
    return true;
}

static bool dspark_json_next_object(const char **cursor, const char *end,
                                    const char **obj_begin,
                                    const char **obj_end) {
    const char *p = *cursor;
    while (p < end && *p != '{') p++;
    if (p >= end) return false;
    const char *q = dspark_json_match_container(p, end, '{', '}');
    if (!q) return false;
    *obj_begin = p;
    *obj_end = q;
    *cursor = q;
    return true;
}

static const char *ds4_dspark_record_kind_name(ds4_dspark_record_kind k) {
    switch (k) {
    case DS4_DSPARK_REC_F16:          return "F16";
    case DS4_DSPARK_REC_F32:          return "F32";
    case DS4_DSPARK_REC_FP8_E4M3:     return "FP8_E4M3";
    case DS4_DSPARK_REC_MXFP4_NATIVE: return "MXFP4_NATIVE";
    default:                          return "?";
    }
}

static ds4_dspark_record_kind ds4_dspark_record_kind_from_string(const char *s) {
    if (!strcmp(s, "F16")) return DS4_DSPARK_REC_F16;
    if (!strcmp(s, "F32")) return DS4_DSPARK_REC_F32;
    if (!strcmp(s, "FP8_E4M3")) return DS4_DSPARK_REC_FP8_E4M3;
    if (!strcmp(s, "MXFP4_NATIVE")) return DS4_DSPARK_REC_MXFP4_NATIVE;
    return DS4_DSPARK_REC_NONE;
}

static int ds4_dspark_file_index(const ds4_dspark_draft *d, const char *name) {
    for (int i = 0; i < d->file_count; i++) {
        if (!strcmp(d->files[i].name, name)) return i;
    }
    return -1;
}

static bool ds4_dspark_range_in_file(const ds4_dspark_draft *d,
                                     const ds4_dspark_record *r,
                                     uint64_t off,
                                     uint64_t bytes) {
    if (r->file_index < 0 || r->file_index >= d->file_count) return false;
    const uint64_t size = d->files[r->file_index].size;
    return off <= size && bytes <= size - off;
}

static bool ds4_dspark_record_ranges_ok(const ds4_dspark_draft *d,
                                        const ds4_dspark_record *r) {
    if (r->kind == DS4_DSPARK_REC_F16 || r->kind == DS4_DSPARK_REC_F32) {
        return ds4_dspark_range_in_file(d, r, r->offset, r->bytes);
    }
    if (r->kind == DS4_DSPARK_REC_FP8_E4M3 ||
        r->kind == DS4_DSPARK_REC_MXFP4_NATIVE) {
        return ds4_dspark_range_in_file(d, r, r->plane_data_offset, r->plane_data_bytes) &&
               ds4_dspark_range_in_file(d, r, r->plane_scale_offset, r->plane_scale_bytes);
    }
    return false;
}

static bool ds4_dspark_parse_record(ds4_dspark_draft *d,
                                    const char *begin,
                                    const char *end,
                                    ds4_dspark_record *r) {
    memset(r, 0, sizeof(*r));
    r->file_index = -1;
    r->layer = -1;
    char kind[32] = {0};
    if (!dspark_json_string_value(begin, end, "name", r->name, sizeof(r->name)) ||
        !dspark_json_string_value(begin, end, "file", r->file, sizeof(r->file)) ||
        !dspark_json_string_value(begin, end, "storage_layout",
                                  r->storage_layout, sizeof(r->storage_layout))) {
        return false;
    }
    if (!dspark_json_string_value(begin, end, "quant_type", kind, sizeof(kind)) &&
        !dspark_json_string_value(begin, end, "exec_dtype", kind, sizeof(kind))) {
        return false;
    }
    r->kind = ds4_dspark_record_kind_from_string(kind);
    if (r->kind == DS4_DSPARK_REC_NONE) return false;
    int64_t layer = -1;
    if (dspark_json_i64(begin, end, "layer", &layer)) r->layer = (int)layer;
    if (!dspark_json_u64_array(begin, end, "shape",
                               r->shape,
                               (int)(sizeof(r->shape) / sizeof(r->shape[0])),
                               &r->ndim) ||
        r->ndim <= 0) {
        return false;
    }
    (void)dspark_json_u64_array(begin, end, "scale_shape",
                                r->scale_shape,
                                (int)(sizeof(r->scale_shape) / sizeof(r->scale_shape[0])),
                                &r->scale_ndim);
    (void)dspark_json_bool(begin, end, "expert_major", &r->expert_major);
    int64_t expert_count = 0;
    if (dspark_json_i64(begin, end, "expert_count", &expert_count)) {
        r->expert_count = (int)expert_count;
    }
    (void)dspark_json_u64(begin, end, "plane_data_bytes_per_expert",
                          &r->plane_data_bytes_per_expert);
    (void)dspark_json_u64(begin, end, "plane_scale_bytes_per_expert",
                          &r->plane_scale_bytes_per_expert);
    if (r->kind == DS4_DSPARK_REC_F16 || r->kind == DS4_DSPARK_REC_F32) {
        if (!dspark_json_u64(begin, end, "offset", &r->offset) ||
            !dspark_json_u64(begin, end, "bytes", &r->bytes)) {
            return false;
        }
    } else {
        if (!dspark_json_u64(begin, end, "plane_data_offset", &r->plane_data_offset) ||
            !dspark_json_u64(begin, end, "plane_data_bytes", &r->plane_data_bytes) ||
            !dspark_json_u64(begin, end, "plane_scale_offset", &r->plane_scale_offset) ||
            !dspark_json_u64(begin, end, "plane_scale_bytes", &r->plane_scale_bytes) ||
            r->scale_ndim <= 0) {
            return false;
        }
    }
    r->file_index = ds4_dspark_file_index(d, r->file);
    return r->file_index >= 0 && ds4_dspark_record_ranges_ok(d, r);
}

static const ds4_dspark_record *ds4_dspark_find_record(
        const ds4_dspark_draft *d,
        const char *name) {
    for (int i = 0; i < d->record_count; i++) {
        if (!strcmp(d->records[i].name, name)) return &d->records[i];
    }
    return NULL;
}

static bool ds4_dspark_require_record(
        const ds4_dspark_draft *d,
        const char             *name,
        ds4_dspark_record_kind  kind,
        int                     ndim,
        uint64_t                d0,
        uint64_t                d1,
        uint64_t                d2) {
    const ds4_dspark_record *r = ds4_dspark_find_record(d, name);
    if (!r) {
        fprintf(stderr, "ds4: DSpark draft package is missing record %s\n", name);
        return false;
    }
    if (r->kind != kind) {
        fprintf(stderr,
                "ds4: DSpark record %s has kind %s, expected %s\n",
                name,
                ds4_dspark_record_kind_name(r->kind),
                ds4_dspark_record_kind_name(kind));
        return false;
    }
    if (r->ndim != ndim) {
        fprintf(stderr,
                "ds4: DSpark record %s has %d dims, expected %d\n",
                name, r->ndim, ndim);
        return false;
    }
    const uint64_t want[3] = { d0, d1, d2 };
    for (int i = 0; i < ndim; i++) {
        if (r->shape[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: DSpark record %s shape[%d]=%" PRIu64 ", expected %" PRIu64 "\n",
                name, i, r->shape[i], want[i]);
        return false;
    }
    return true;
}

static bool ds4_dspark_validate_records(const ds4_dspark_draft *d) {
    if (d->record_count != 81 ||
        d->record_kind_count[DS4_DSPARK_REC_F32] != 41 ||
        d->record_kind_count[DS4_DSPARK_REC_F16] != 6 ||
        d->record_kind_count[DS4_DSPARK_REC_FP8_E4M3] != 25 ||
        d->record_kind_count[DS4_DSPARK_REC_MXFP4_NATIVE] != 9) {
        fprintf(stderr,
                "ds4: DSpark draft package has unexpected record counts "
                "(entries=%d F32=%d F16=%d FP8=%d MXFP4=%d; expected 81/41/6/25/9)\n",
                d->record_count,
                d->record_kind_count[DS4_DSPARK_REC_F32],
                d->record_kind_count[DS4_DSPARK_REC_F16],
                d->record_kind_count[DS4_DSPARK_REC_FP8_E4M3],
                d->record_kind_count[DS4_DSPARK_REC_MXFP4_NATIVE]);
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t routed_down_dim = (uint64_t)DS4_N_FF_EXP / 2u;
    if ((DS4_N_FF_EXP & 1u) != 0) return false;

    if (!ds4_dspark_require_record(d, "dspark.main_norm.weight",
                                   DS4_DSPARK_REC_F32, 1, DS4_N_EMBD, 0, 0) ||
        !ds4_dspark_require_record(d, "dspark.main_proj",
                                   DS4_DSPARK_REC_FP8_E4M3, 2,
                                   DS4_N_EMBD, 3u * DS4_N_EMBD, 0)) {
        return false;
    }

    for (int il = 0; il < 3; il++) {
        char name[128];
#define DS4_DSPARK_REQ(fmt_, kind_, ndim_, d0_, d1_, d2_) do { \
            snprintf(name, sizeof(name), fmt_, il); \
            if (!ds4_dspark_require_record(d, name, kind_, ndim_, d0_, d1_, d2_)) return false; \
        } while (0)

        DS4_DSPARK_REQ("dspark.blk.%d.attn_sinks.weight", DS4_DSPARK_REC_F32, 1,
                       DS4_N_HEAD, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_kv_a_norm.weight", DS4_DSPARK_REC_F32, 1,
                       DS4_N_HEAD_DIM, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_q_a_norm.weight", DS4_DSPARK_REC_F32, 1,
                       DS4_N_LORA_Q, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_kv", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_HEAD_DIM, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_output_a", DS4_DSPARK_REC_FP8_E4M3, 2,
                       out_low_dim, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_output_b", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_EMBD, out_low_dim, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_q_a", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_LORA_Q, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_q_b", DS4_DSPARK_REC_FP8_E4M3, 2,
                       q_dim, DS4_N_LORA_Q, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.attn_norm_weight", DS4_DSPARK_REC_F32, 1,
                       DS4_N_EMBD, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.exp_probs_b.bias", DS4_DSPARK_REC_F32, 1,
                       DS4_N_EXPERT, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_gate_inp.weight", DS4_DSPARK_REC_F16, 2,
                       DS4_N_EXPERT, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_gate_shexp", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_FF_EXP, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_down_shexp", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_EMBD, DS4_N_FF_EXP, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_up_shexp", DS4_DSPARK_REC_FP8_E4M3, 2,
                       DS4_N_FF_EXP, DS4_N_EMBD, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_norm_weight", DS4_DSPARK_REC_F32, 1,
                       DS4_N_EMBD, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_attn_base", DS4_DSPARK_REC_F32, 1,
                       hc_mix_dim, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_attn_fn", DS4_DSPARK_REC_F32, 2,
                       hc_mix_dim, hc_dim, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_attn_scale", DS4_DSPARK_REC_F32, 1,
                       3, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_ffn_base", DS4_DSPARK_REC_F32, 1,
                       hc_mix_dim, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_ffn_fn", DS4_DSPARK_REC_F32, 2,
                       hc_mix_dim, hc_dim, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.hc_ffn_scale", DS4_DSPARK_REC_F32, 1,
                       3, 0, 0);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_gate_exps", DS4_DSPARK_REC_MXFP4_NATIVE, 3,
                       DS4_N_FF_EXP, DS4_N_FF_EXP, DS4_N_EXPERT);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_up_exps", DS4_DSPARK_REC_MXFP4_NATIVE, 3,
                       DS4_N_FF_EXP, DS4_N_FF_EXP, DS4_N_EXPERT);
        DS4_DSPARK_REQ("dspark.blk.%d.ffn_down_exps", DS4_DSPARK_REC_MXFP4_NATIVE, 3,
                       routed_down_dim, DS4_N_EMBD, DS4_N_EXPERT);
#undef DS4_DSPARK_REQ
    }

    return ds4_dspark_require_record(d, "dspark.confidence_proj.weight",
                                     DS4_DSPARK_REC_F16, 2, 1, DS4_N_EMBD + d->markov_rank, 0) &&
           ds4_dspark_require_record(d, "dspark.blk.2.hc_head_base",
                                     DS4_DSPARK_REC_F32, 1, DS4_N_HC, 0, 0) &&
           ds4_dspark_require_record(d, "dspark.blk.2.hc_head_fn",
                                     DS4_DSPARK_REC_F32, 2, DS4_N_HC, hc_dim, 0) &&
           ds4_dspark_require_record(d, "dspark.blk.2.hc_head_scale",
                                     DS4_DSPARK_REC_F32, 1, 1, 0, 0) &&
           ds4_dspark_require_record(d, "dspark.markov_embd.weight",
                                     DS4_DSPARK_REC_F16, 2, DS4_N_VOCAB, d->markov_rank, 0) &&
           ds4_dspark_require_record(d, "dspark.markov_output.weight",
                                     DS4_DSPARK_REC_F16, 2, DS4_N_VOCAB, d->markov_rank, 0) &&
           ds4_dspark_require_record(d, "dspark.blk.2.output_norm.weight",
                                     DS4_DSPARK_REC_F32, 1, DS4_N_EMBD, 0, 0);
}

static bool ds4_dspark_parse_entries(ds4_dspark_draft *d,
                                     const char *json,
                                     size_t json_len) {
    const char *arr = NULL;
    const char *arr_end = NULL;
    if (!dspark_json_array_range(json, json + json_len, "entries", &arr, &arr_end)) {
        fprintf(stderr, "ds4: DSpark manifest is missing entries[]\n");
        return false;
    }
    const char *cur = arr;
    const char *obj = NULL;
    const char *obj_end = NULL;
    while (dspark_json_next_object(&cur, arr_end, &obj, &obj_end)) {
        if (d->record_count >= (int)(sizeof(d->records) / sizeof(d->records[0]))) {
            fprintf(stderr, "ds4: DSpark manifest has too many entries\n");
            return false;
        }
        ds4_dspark_record *r = &d->records[d->record_count];
        if (!ds4_dspark_parse_record(d, obj, obj_end, r)) {
            fprintf(stderr, "ds4: failed to parse/validate DSpark manifest record %d\n",
                    d->record_count);
            return false;
        }
        d->record_kind_count[r->kind]++;
        d->record_count++;
    }
    return ds4_dspark_validate_records(d);
}

static bool dspark_package_file_readable(const char *dir, const char *name) {
    char *path = ds4_join_path(dir, name);
    const bool ok = ds4_path_readable_file(path);
    free(path);
    return ok;
}

static bool ds4_dspark_map_file(ds4_dspark_file *f, const char *dir, const char *name) {
    memset(f, 0, sizeof(*f));
    f->fd = -1;
    snprintf(f->name, sizeof(f->name), "%s", name);
    char *path = ds4_join_path(dir, name);
    snprintf(f->path, sizeof(f->path), "%s", path);
    free(path);

    f->fd = open(f->path, O_RDONLY);
    if (f->fd < 0) {
        fprintf(stderr, "ds4: failed to open DSpark draft file %s: %s\n",
                f->path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(f->fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "ds4: failed to stat DSpark draft file %s: %s\n",
                f->path, strerror(errno));
        close(f->fd);
        f->fd = -1;
        return false;
    }
    f->size = (uint64_t)st.st_size;
    void *map = mmap(NULL, (size_t)f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4: failed to mmap DSpark draft file %s: %s\n",
                f->path, strerror(errno));
        close(f->fd);
        f->fd = -1;
        f->size = 0;
        return false;
    }
    f->map = (const uint8_t *)map;
    return true;
}

static void ds4_dspark_unmap_file(ds4_dspark_file *f) {
    if (!f) return;
    if (f->map) {
        munmap((void *)f->map, (size_t)f->size);
    }
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

static bool ds4_dspark_map_files(ds4_dspark_draft *d) {
    d->file_count = 0;
    for (int i = 0; i < d->draft_layer_count; i++) {
        char name[64];
        snprintf(name, sizeof(name), "draft_layer_%03d.bin", d->draft_layer_ids[i]);
        if (d->file_count >= (int)(sizeof(d->files) / sizeof(d->files[0]))) return false;
        if (!ds4_dspark_map_file(&d->files[d->file_count], d->path, name)) return false;
        d->file_count++;
    }
    return true;
}

static void ds4_dspark_close(ds4_dspark_draft *d) {
    if (!d) return;
#ifndef DS4_NO_GPU
    if (d->file_count > 0) ds4_gpu_clear_sidecar_mmap_cache();
#endif
    for (int i = 0; i < d->file_count; i++) {
        ds4_dspark_unmap_file(&d->files[i]);
    }
    free(d->path);
    free(d->manifest_path);
    memset(d, 0, sizeof(*d));
}

static const ds4_dspark_file *ds4_dspark_record_file(
        const ds4_dspark_draft *d,
        const ds4_dspark_record *r) {
    if (!d || !r || r->file_index < 0 || r->file_index >= d->file_count) return NULL;
    const ds4_dspark_file *f = &d->files[r->file_index];
    return f->map && f->size ? f : NULL;
}

static bool ds4_dspark_record_matrix_dims(
        const ds4_dspark_record *r,
        uint64_t *out_dim,
        uint64_t *in_dim) {
    if (!r || r->ndim != 2 || r->shape[0] == 0 || r->shape[1] == 0) return false;
    if (out_dim) *out_dim = r->shape[0];
    if (in_dim) *in_dim = r->shape[1];
    return true;
}

#ifndef DS4_NO_GPU
static bool ds4_dspark_matmul_record(
        const ds4_dspark_draft *d,
        const ds4_dspark_record *r,
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    uint64_t out_dim = 0;
    uint64_t in_dim = 0;
    if (!ds4_dspark_record_matrix_dims(r, &out_dim, &in_dim)) return false;
    const ds4_dspark_file *f = ds4_dspark_record_file(d, r);
    if (!f) return false;

    switch (r->kind) {
    case DS4_DSPARK_REC_F16:
        return ds4_gpu_matmul_f16_tensor(out,
                                         f->map,
                                         f->size,
                                         r->offset,
                                         in_dim,
                                         out_dim,
                                         x,
                                         n_tok) != 0;
    case DS4_DSPARK_REC_F32:
        return ds4_gpu_matmul_f32_tensor(out,
                                         f->map,
                                         f->size,
                                         r->offset,
                                         in_dim,
                                         out_dim,
                                         x,
                                         n_tok) != 0;
    case DS4_DSPARK_REC_FP8_E4M3:
        if (r->scale_ndim != 2) return false;
        return ds4_gpu_matmul_fp8_e4m3_tensor(out,
                                              f->map,
                                              f->size,
                                              r->plane_data_offset,
                                              r->plane_scale_offset,
                                              in_dim,
                                              out_dim,
                                              r->scale_shape[0],
                                              r->scale_shape[1],
                                              x,
                                              n_tok) != 0;
    default:
        return false;
    }
}

static bool ds4_dspark_matmul_fp8_record_row_slice(
        const ds4_dspark_draft *d,
        const ds4_dspark_record *r,
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        uint64_t row0,
        uint64_t n_rows,
        uint64_t n_tok) {
    uint64_t out_dim = 0;
    uint64_t in_dim = 0;
    if (!ds4_dspark_record_matrix_dims(r, &out_dim, &in_dim) ||
        r->kind != DS4_DSPARK_REC_FP8_E4M3 ||
        r->scale_ndim != 2 ||
        row0 > out_dim ||
        n_rows == 0 ||
        n_rows > out_dim - row0 ||
        (row0 % 128u) != 0 ||
        (n_rows % 128u) != 0) {
        return false;
    }
    const ds4_dspark_file *f = ds4_dspark_record_file(d, r);
    if (!f) return false;

    const uint64_t scale_cols = r->scale_shape[1];
    if (scale_cols == 0 || r->scale_shape[0] < (row0 + n_rows + 127u) / 128u) {
        return false;
    }

    const uint64_t data_offset = r->plane_data_offset + row0 * in_dim;
    const uint64_t scale_offset = r->plane_scale_offset + (row0 / 128u) * scale_cols;
    return ds4_gpu_matmul_fp8_e4m3_tensor(out,
                                          f->map,
                                          f->size,
                                          data_offset,
                                          scale_offset,
                                          in_dim,
                                          n_rows,
                                          n_rows / 128u,
                                          scale_cols,
                                          x,
                                          n_tok) != 0;
}

static bool ds4_dspark_rms_norm_record(
        const ds4_dspark_draft *d,
        const ds4_dspark_record *r,
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        uint32_t width,
        uint32_t rows,
        float eps) {
    if (!d || !r || !out || !x || width == 0 || rows == 0 ||
        r->kind != DS4_DSPARK_REC_F32 || r->ndim != 1 || r->shape[0] != width) {
        return false;
    }
    const ds4_dspark_file *f = ds4_dspark_record_file(d, r);
    if (!f) return false;
    return ds4_gpu_rms_norm_weight_rows_tensor(out,
                                               x,
                                               f->map,
                                               f->size,
                                               r->offset,
                                               width,
                                               rows,
                                               eps) != 0;
}

static bool ds4_dspark_dense_smoke(const ds4_dspark_draft *d) {
    const char *env = getenv("DS4_DSPARK_DENSE_SMOKE");
    if (!env || !env[0] || atoi(env) == 0) return true;

    const ds4_dspark_record *fp8 = ds4_dspark_find_record(d, "dspark.blk.0.attn_kv");
    const ds4_dspark_record *f16 = ds4_dspark_find_record(d, "dspark.confidence_proj.weight");
    const ds4_dspark_record *main_norm = ds4_dspark_find_record(d, "dspark.main_norm.weight");
    const ds4_dspark_record *main_proj = ds4_dspark_find_record(d, "dspark.main_proj");
    if (!fp8 || !f16 || !main_norm || !main_proj) {
        fprintf(stderr, "ds4: DSpark dense smoke missing required records\n");
        return false;
    }

    uint64_t fp8_out = 0, fp8_in = 0;
    uint64_t f16_out = 0, f16_in = 0;
    uint64_t main_out = 0, main_in = 0;
    if (!ds4_dspark_record_matrix_dims(fp8, &fp8_out, &fp8_in) ||
        !ds4_dspark_record_matrix_dims(f16, &f16_out, &f16_in) ||
        !ds4_dspark_record_matrix_dims(main_proj, &main_out, &main_in) ||
        main_norm->ndim != 1 || main_norm->shape[0] != main_out) {
        fprintf(stderr, "ds4: DSpark dense smoke found non-matrix records\n");
        return false;
    }

    ds4_gpu_tensor *x_fp8 = ds4_gpu_tensor_alloc(fp8_in * sizeof(float));
    ds4_gpu_tensor *y_fp8 = ds4_gpu_tensor_alloc(fp8_out * sizeof(float));
    ds4_gpu_tensor *x_f16 = ds4_gpu_tensor_alloc(f16_in * sizeof(float));
    ds4_gpu_tensor *y_f16 = ds4_gpu_tensor_alloc(f16_out * sizeof(float));
    ds4_gpu_tensor *main_hidden = ds4_gpu_tensor_alloc(main_in * sizeof(float));
    ds4_gpu_tensor *main_proj_out = ds4_gpu_tensor_alloc(main_out * sizeof(float));
    ds4_gpu_tensor *main_x = ds4_gpu_tensor_alloc(main_out * sizeof(float));
    if (!x_fp8 || !y_fp8 || !x_f16 || !y_f16 ||
        !main_hidden || !main_proj_out || !main_x) {
        fprintf(stderr, "ds4: DSpark dense smoke failed to allocate tensors\n");
        ds4_gpu_tensor_free(main_x);
        ds4_gpu_tensor_free(main_proj_out);
        ds4_gpu_tensor_free(main_hidden);
        ds4_gpu_tensor_free(y_f16);
        ds4_gpu_tensor_free(x_f16);
        ds4_gpu_tensor_free(y_fp8);
        ds4_gpu_tensor_free(x_fp8);
        return false;
    }

    bool ok = ds4_gpu_tensor_fill_f32(x_fp8, 0.001f, fp8_in) != 0 &&
              ds4_gpu_tensor_fill_f32(x_f16, 0.001f, f16_in) != 0 &&
              ds4_gpu_tensor_fill_f32(main_hidden, 0.001f, main_in) != 0 &&
              ds4_dspark_matmul_record(d, fp8, y_fp8, x_fp8, 1) &&
              ds4_dspark_matmul_record(d, f16, y_f16, x_f16, 1) &&
              ds4_dspark_matmul_record(d, main_proj, main_proj_out, main_hidden, 1) &&
              ds4_dspark_rms_norm_record(d,
                                          main_norm,
                                          main_x,
                                          main_proj_out,
                                          (uint32_t)main_out,
                                          1,
                                          DS4_DEFAULT_RMS_EPS);

    float fp8_sample[8] = {0};
    float f16_sample[1] = {0};
    float main_sample[8] = {0};
    if (ok) {
        ok = ds4_gpu_tensor_read(y_fp8, 0, fp8_sample, sizeof(fp8_sample)) != 0 &&
             ds4_gpu_tensor_read(y_f16, 0, f16_sample, sizeof(f16_sample)) != 0 &&
             ds4_gpu_tensor_read(main_x, 0, main_sample, sizeof(main_sample)) != 0;
    }
    if (ok) {
        double fp8_rms = 0.0;
        double main_rms = 0.0;
        for (size_t i = 0; i < sizeof(fp8_sample) / sizeof(fp8_sample[0]); i++) {
            fp8_rms += (double)fp8_sample[i] * (double)fp8_sample[i];
            main_rms += (double)main_sample[i] * (double)main_sample[i];
        }
        fp8_rms = sqrt(fp8_rms / (double)(sizeof(fp8_sample) / sizeof(fp8_sample[0])));
        main_rms = sqrt(main_rms / (double)(sizeof(main_sample) / sizeof(main_sample[0])));
        fprintf(stderr,
                "ds4: DSpark dense smoke ok: fp8 attn_kv sample_rms=%.6g confidence=%.6g main_proj_sample_rms=%.6g\n",
                fp8_rms,
                (double)f16_sample[0],
                main_rms);
    } else {
        fprintf(stderr, "ds4: DSpark dense smoke failed\n");
    }

    ds4_gpu_tensor_free(main_x);
    ds4_gpu_tensor_free(main_proj_out);
    ds4_gpu_tensor_free(main_hidden);
    ds4_gpu_tensor_free(y_f16);
    ds4_gpu_tensor_free(x_f16);
    ds4_gpu_tensor_free(y_fp8);
    ds4_gpu_tensor_free(x_fp8);
    return ok;
}

static bool ds4_dspark_resident_experts_enabled(void) {
    const char *env = getenv("DS4_DSPARK_RESIDENT");
    if (env && env[0] && atoi(env) == 0) {
        const char *allow = getenv("DS4_DSPARK_ALLOW_NONRESIDENT");
        if (allow && allow[0] && atoi(allow) != 0) {
            fprintf(stderr,
                    "ds4: DSpark resident expert cache disabled by "
                    "DS4_DSPARK_RESIDENT=0 DS4_DSPARK_ALLOW_NONRESIDENT=1 "
                    "(debug only; performance will not represent DSpark)\n");
            return false;
        }
        fprintf(stderr,
                "ds4: ignoring DS4_DSPARK_RESIDENT=0; DSpark requires persistent "
                "resident draft experts (set DS4_DSPARK_ALLOW_NONRESIDENT=1 only "
                "for debug)\n");
    }
    return true;
}

static bool ds4_dspark_cache_resident_experts(const ds4_dspark_draft *d) {
    if (!d) return false;
    if (!ds4_dspark_resident_experts_enabled()) {
        fprintf(stderr, "ds4: DSpark resident expert cache disabled for debug\n");
        return true;
    }

    uint64_t total_bytes = 0;
    int cached = 0;
    for (int i = 0; i < d->record_count; i++) {
        const ds4_dspark_record *r = &d->records[i];
        if (r->kind != DS4_DSPARK_REC_MXFP4_NATIVE) continue;
        const ds4_dspark_file *f = ds4_dspark_record_file(d, r);
        if (!f) return false;
        if (r->plane_data_offset > UINT64_MAX - r->plane_data_bytes ||
            r->plane_scale_offset > UINT64_MAX - r->plane_scale_bytes) {
            return false;
        }
        const uint64_t data_end = r->plane_data_offset + r->plane_data_bytes;
        const uint64_t scale_end = r->plane_scale_offset + r->plane_scale_bytes;
        const uint64_t span_start =
            r->plane_data_offset < r->plane_scale_offset ? r->plane_data_offset : r->plane_scale_offset;
        const uint64_t span_end = data_end > scale_end ? data_end : scale_end;
        if (span_end < span_start) return false;
        const uint64_t span_bytes = span_end - span_start;
        char label[160];
        snprintf(label, sizeof(label), "%s", r->name);
        if (!ds4_gpu_cache_sidecar_range(f->map, f->size, span_start, span_bytes, label)) {
            fprintf(stderr,
                    "ds4: failed to make DSpark expert record resident: %s (%.2f MiB)\n",
                    r->name,
                    (double)span_bytes / (1024.0 * 1024.0));
            return false;
        }
        total_bytes += span_bytes;
        cached++;
    }
    fprintf(stderr,
            "ds4: DSpark resident expert cache: ranges=%d bytes=%.2f GiB "
            "(persistent; debug opt-out requires DS4_DSPARK_ALLOW_NONRESIDENT=1)\n",
            cached,
            (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
    if (cached != 9) return false;
    return ds4_gpu_commit_sidecar_residency("DSpark") != 0;
}
#endif

static bool ds4_dspark_open(ds4_dspark_draft *d, const ds4_engine_options *opt) {
    if (!d || !opt || opt->draft_kind != DS4_DRAFT_DSPARK) return true;
    if (!opt->draft_path || !opt->draft_path[0]) {
        fprintf(stderr, "ds4: --draft dspark requires --draft-path\n");
        return false;
    }
    char *manifest_path = ds4_join_path(opt->draft_path, "manifest.json");
    char *json = NULL;
    size_t json_len = 0;
    if (!ds4_read_text_file(manifest_path, &json, &json_len)) {
        fprintf(stderr, "ds4: failed to read DSpark draft manifest %s: %s\n",
                manifest_path, strerror(errno));
        free(manifest_path);
        return false;
    }

    const char *begin = json;
    const char *end = json + json_len;
    char sidecar_kind[64] = {0};
    char storage_layout[96] = {0};
    bool ok = dspark_json_string_value(begin, end, "sidecar_kind",
                                       sidecar_kind, sizeof(sidecar_kind));
    const bool storage_ok =
        ds4_file_contains_literal(manifest_path,
                                  "\"storage_layout\": \"ds4_dspark_draft_v1\"");
    if (storage_ok) snprintf(storage_layout, sizeof(storage_layout), "ds4_dspark_draft_v1");
    if (!ok || strcmp(sidecar_kind, "dspark_draft") != 0 || !storage_ok) {
        fprintf(stderr,
                "ds4: %s is not a DS4 DSpark draft package "
                "(sidecar_kind=%s storage_layout=%s)\n",
                manifest_path,
                sidecar_kind[0] ? sidecar_kind : "?",
                storage_layout[0] ? storage_layout : "?");
        free(json);
        free(manifest_path);
        return false;
    }

    d->path = ds4_strdup(opt->draft_path);
    d->manifest_path = manifest_path;
    d->verify_budget = opt->draft_verify > 0 ? opt->draft_verify : 5;
    if (d->verify_budget > 5) d->verify_budget = 5;
    snprintf(d->scheduler, sizeof(d->scheduler), "%s",
             (opt->draft_scheduler && opt->draft_scheduler[0]) ? opt->draft_scheduler : "static");
    d->conf_threshold = opt->draft_conf_threshold;

    ok = dspark_json_int_required(json, json_len, "hidden_size", &d->hidden_size) &&
         dspark_json_int_required(json, json_len, "layer_count", &d->layer_count) &&
         dspark_json_int_required(json, json_len, "expert_count", &d->expert_count) &&
         dspark_json_int_required(json, json_len, "expert_used_count", &d->expert_used_count) &&
         dspark_json_int_required(json, json_len, "vocab_size", &d->vocab_size) &&
         dspark_json_int_required(json, json_len, "block_size", &d->block_size) &&
         dspark_json_int_required(json, json_len, "draft_layer_count", &d->draft_layer_count) &&
         dspark_json_int_required(json, json_len, "markov_rank", &d->markov_rank) &&
         dspark_json_int_required(json, json_len, "window_size", &d->window_size) &&
         dspark_json_int_array(begin, end, "draft_layer_ids",
                               d->draft_layer_ids,
                               (int)(sizeof(d->draft_layer_ids) / sizeof(d->draft_layer_ids[0])),
                               &d->draft_layer_count) &&
         dspark_json_int_array(begin, end, "target_layer_ids",
                               d->target_layer_ids,
                               (int)(sizeof(d->target_layer_ids) / sizeof(d->target_layer_ids[0])),
                               &d->target_layer_count);
    if (ok) {
        int64_t noise_id = 128799;
        if (dspark_json_i64(begin, end, "noise_token_id", &noise_id) &&
            noise_id >= 0 && noise_id <= INT_MAX) {
            d->noise_token_id = (int)noise_id;
        } else {
            d->noise_token_id = 128799;
        }
    }
    if (!ok) {
        fprintf(stderr, "ds4: DSpark draft manifest is missing required model/dspark metadata\n");
        free(json);
        ds4_dspark_close(d);
        return false;
    }
    if (d->block_size != 5 || d->draft_layer_count != 3 || d->target_layer_count != 3) {
        fprintf(stderr,
                "ds4: DSpark draft package has unsupported topology "
                "(block=%d draft_layers=%d target_layers=%d; expected DSpark-5 with 3/3 layers)\n",
                d->block_size, d->draft_layer_count, d->target_layer_count);
        free(json);
        ds4_dspark_close(d);
        return false;
    }
    for (int i = 0; i < d->draft_layer_count; i++) {
        char name[64];
        snprintf(name, sizeof(name), "draft_layer_%03d.bin", d->draft_layer_ids[i]);
        if (!dspark_package_file_readable(d->path, name)) {
            fprintf(stderr, "ds4: DSpark draft package is missing %s\n", name);
            free(json);
            ds4_dspark_close(d);
            return false;
        }
    }
    if (d->hidden_size != (int)DS4_N_EMBD ||
        d->layer_count != (int)DS4_N_LAYER ||
        d->expert_count != (int)DS4_N_EXPERT ||
        d->expert_used_count != (int)DS4_N_EXPERT_USED ||
        d->vocab_size != (int)DS4_N_VOCAB ||
        d->noise_token_id < 0 ||
        d->noise_token_id >= (int)DS4_N_VOCAB) {
        fprintf(stderr,
                "ds4: DSpark draft package is not compatible with target model "
                "(draft hidden=%d layers=%d experts=%d topk=%d vocab=%d noise=%d; "
                "target hidden=%u layers=%u experts=%u topk=%u vocab=%u)\n",
                d->hidden_size, d->layer_count, d->expert_count,
                d->expert_used_count, d->vocab_size, d->noise_token_id,
                (uint32_t)DS4_N_EMBD, (uint32_t)DS4_N_LAYER,
                (uint32_t)DS4_N_EXPERT, (uint32_t)DS4_N_EXPERT_USED,
                (uint32_t)DS4_N_VOCAB);
        free(json);
        ds4_dspark_close(d);
        return false;
    }
    for (int i = 0; i < d->target_layer_count; i++) {
        const int expected = (int)DS4_N_LAYER - d->target_layer_count + i;
        if (d->target_layer_ids[i] != expected) {
            fprintf(stderr,
                    "ds4: DSpark target layer mismatch at %d: manifest=%d expected=%d\n",
                    i, d->target_layer_ids[i], expected);
            free(json);
            ds4_dspark_close(d);
            return false;
        }
    }
    if (!ds4_dspark_map_files(d) ||
        !ds4_dspark_parse_entries(d, json, json_len)) {
        free(json);
        ds4_dspark_close(d);
        return false;
    }
    free(json);

#ifndef DS4_NO_GPU
    if (!ds4_dspark_dense_smoke(d)) {
        ds4_dspark_close(d);
        return false;
    }
    if (!ds4_gpu_has_dspark_mxfp4_routed_batch()) {
        fprintf(stderr,
                "ds4: DSpark draft inference requires fused native MXFP4 routed-MoE "
                "kernels (batch-strided pair-SwiGLU + down-sum6)\n");
        ds4_dspark_close(d);
        return false;
    }
    if (!ds4_dspark_cache_resident_experts(d)) {
        ds4_dspark_close(d);
        return false;
    }
#endif

    uint64_t mapped_bytes = 0;
    for (int i = 0; i < d->file_count; i++) mapped_bytes += d->files[i].size;
    d->loaded = true;
    d->inference_ready = true;
    int active_budget = d->verify_budget > 0 ? d->verify_budget : d->block_size;
    if (active_budget > d->block_size) active_budget = d->block_size;
    if (active_budget > 5) active_budget = 5;
    fprintf(stderr,
            "ds4: DSpark draft package loaded: %s "
            "(block=%d verify=%d active=%d scheduler=%s conf=%.3f noise=%d target_layers=%d,%d,%d markov_rank=%d records=%d mapped=%.2f GiB)\n",
            d->path,
            d->block_size,
            d->verify_budget,
            active_budget,
            d->scheduler,
            d->conf_threshold,
            d->noise_token_id,
            d->target_layer_ids[0],
            d->target_layer_ids[1],
            d->target_layer_ids[2],
            d->markov_rank,
            d->record_count,
            (double)mapped_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr,
            "ds4: DSpark draft records: F32=%d F16=%d FP8_E4M3=%d MXFP4_NATIVE=%d\n",
            d->record_kind_count[DS4_DSPARK_REC_F32],
            d->record_kind_count[DS4_DSPARK_REC_F16],
            d->record_kind_count[DS4_DSPARK_REC_FP8_E4M3],
            d->record_kind_count[DS4_DSPARK_REC_MXFP4_NATIVE]);
    fprintf(stderr,
            "ds4: DSpark draft inference enabled: MPP 4.1 FP8/MXFP4 draft kernels, "
            "fused routed-MoE per draft layer, persistent resident experts, "
            "greedy Markov head, strict_v1 verifier\n");
    return true;
}

/* DSpark draft runtime. */
static bool metal_graph_capture_dspark_target_hidden(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       pos) {
    if (!g || !g->dspark_target_hidden || !g->dspark_hc_mean_weights) return true;
    if (DS4_N_LAYER < 3u || il + 3u < DS4_N_LAYER) return true;
    const uint32_t slot = il - (DS4_N_LAYER - 3u);
    if (slot >= 3u) return true;

    const uint32_t ring = pos % 128u;
    ds4_gpu_tensor *dst = ds4_gpu_tensor_view(g->dspark_target_hidden,
                                              ((uint64_t)ring * 3u + slot) * DS4_N_EMBD * sizeof(float),
                                              (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!dst) return false;
    const bool ok = ds4_gpu_hc_weighted_sum_tensor(dst,
                                                   g->after_ffn_hc,
                                                   g->dspark_hc_mean_weights,
                                                   DS4_N_EMBD,
                                                   DS4_N_HC) != 0;
    ds4_gpu_tensor_free(dst);
    if (ok) g->dspark_capture_count++;
    return ok;
}

static bool metal_graph_capture_dspark_target_hidden_batch(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       pos0,
        uint32_t       n_tokens) {
    if (!g || !g->dspark || !g->dspark->loaded ||
        !g->dspark_target_hidden || !g->dspark_hc_mean_weights) {
        return true;
    }
    if (DS4_N_LAYER < 3u || il + 3u < DS4_N_LAYER || n_tokens == 0) return true;
    const uint32_t slot = il - (DS4_N_LAYER - 3u);
    if (slot >= 3u) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t first = n_tokens > 128u ? n_tokens - 128u : 0u;
    bool ok = true;
    for (uint32_t row = first; ok && row < n_tokens; row++) {
        const uint32_t ring = (pos0 + row) % 128u;
        ds4_gpu_tensor *src =
            ds4_gpu_tensor_view(g->batch_next_hc,
                                (uint64_t)row * hc_dim * sizeof(float),
                                hc_dim * sizeof(float));
        ds4_gpu_tensor *dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                ((uint64_t)ring * 3u + slot) * DS4_N_EMBD * sizeof(float),
                                (uint64_t)DS4_N_EMBD * sizeof(float));
        ok = src && dst &&
             ds4_gpu_hc_weighted_sum_tensor(dst,
                                            src,
                                            g->dspark_hc_mean_weights,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        ds4_gpu_tensor_free(dst);
        ds4_gpu_tensor_free(src);
        if (ok) g->dspark_capture_count++;
    }
    return ok;
}

static bool metal_graph_dspark_project_main(ds4_gpu_graph *g, uint32_t pos) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (!g->dspark_target_hidden || !g->dspark_main_proj || !g->dspark_main_x) {
        return false;
    }
    const ds4_dspark_record *main_proj =
        ds4_dspark_find_record(g->dspark, "dspark.main_proj");
    const ds4_dspark_record *main_norm =
        ds4_dspark_find_record(g->dspark, "dspark.main_norm.weight");
    if (!main_proj || !main_norm) return false;
    const uint32_t ring = pos % 128u;
    ds4_gpu_tensor *src =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)ring * 3u * DS4_N_EMBD * sizeof(float),
                            3ull * DS4_N_EMBD * sizeof(float));
    bool ok = src != NULL;
    if (ok) ok = ds4_dspark_matmul_record(g->dspark,
                                          main_proj,
                                          g->dspark_main_proj,
                                          src,
                                          1) &&
                 ds4_dspark_rms_norm_record(g->dspark,
                                             main_norm,
                                             g->dspark_main_x,
                                             g->dspark_main_proj,
                                             DS4_N_EMBD,
                                             1,
                                             DS4_DEFAULT_RMS_EPS);
    ds4_gpu_tensor_free(src);
    return ok;
}

static bool metal_graph_dspark_project_main_range(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (n_tokens == 0) return true;
    if (n_tokens > 5u) return false;
    if (!g->dspark_target_hidden || !g->dspark_main_proj || !g->dspark_main_x) {
        return false;
    }
    const uint32_t ring = start % 128u;
    if (ring + n_tokens > 128u) return false;

    const ds4_dspark_record *main_proj =
        ds4_dspark_find_record(g->dspark, "dspark.main_proj");
    const ds4_dspark_record *main_norm =
        ds4_dspark_find_record(g->dspark, "dspark.main_norm.weight");
    if (!main_proj || !main_norm) return false;

    ds4_gpu_tensor *src =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)ring * 3u * DS4_N_EMBD * sizeof(float),
                            (uint64_t)n_tokens * 3u * DS4_N_EMBD * sizeof(float));
    bool ok = src != NULL;
    if (ok) {
        ok = ds4_dspark_matmul_record(g->dspark,
                                      main_proj,
                                      g->dspark_main_proj,
                                      src,
                                      n_tokens) &&
             ds4_dspark_rms_norm_record(g->dspark,
                                         main_norm,
                                         g->dspark_main_x,
                                         g->dspark_main_proj,
                                         DS4_N_EMBD,
                                         n_tokens,
                                         DS4_DEFAULT_RMS_EPS);
    }
    ds4_gpu_tensor_free(src);
    return ok;
}

static bool metal_graph_dspark_seed_block(
        ds4_gpu_graph       *g,
        const ds4_model     *model,
        const ds4_weights   *weights,
        int                  token,
        uint32_t             n_tokens) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (!model || !weights || !weights->token_embd ||
        !g->dspark_input_ids || !g->dspark_h) {
        return false;
    }
    if (n_tokens == 0 || n_tokens > 5u) return false;
    if (g->dspark->block_size != 5 || token < 0 ||
        token >= (int)weights->token_embd->dim[1]) {
        return false;
    }

    int32_t ids[5];
    ids[0] = (int32_t)token;
    for (uint32_t i = 1; i < 5u; i++) {
        ids[i] = (int32_t)g->dspark->noise_token_id;
    }

    return ds4_gpu_tensor_write(g->dspark_input_ids,
                                0,
                                ids,
                                (uint64_t)n_tokens * sizeof(ids[0])) != 0 &&
           ds4_gpu_embed_tokens_hc_tensor(g->dspark_h,
                                          g->dspark_input_ids,
                                          model->map,
                                          model->size,
                                          weights->token_embd->abs_offset,
                                          (uint32_t)weights->token_embd->dim[1],
                                          n_tokens,
                                          DS4_N_EMBD,
                                          DS4_N_HC) != 0;
}

static bool metal_graph_dspark_update_main_kv(
        ds4_gpu_graph *g,
        uint32_t       pos) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (!g->dspark_main_x) return false;

    for (uint32_t il = 0; il < 3u; il++) {
        if (!g->dspark_main_kv[il] || !g->dspark_kv_cache[il]) return false;
        char kv_name[96];
        char norm_name[96];
        snprintf(kv_name, sizeof(kv_name), "dspark.blk.%u.attn_kv", il);
        snprintf(norm_name, sizeof(norm_name), "dspark.blk.%u.attn_kv_a_norm.weight", il);
        const ds4_dspark_record *kv_rec = ds4_dspark_find_record(g->dspark, kv_name);
        const ds4_dspark_record *norm_rec = ds4_dspark_find_record(g->dspark, norm_name);
        if (!kv_rec || !norm_rec) return false;

        bool ok = ds4_dspark_matmul_record(g->dspark,
                                           kv_rec,
                                           g->dspark_main_kv[il],
                                           g->dspark_main_x,
                                           1) &&
                  ds4_dspark_rms_norm_record(g->dspark,
                                             norm_rec,
                                             g->dspark_main_kv[il],
                                             g->dspark_main_kv[il],
                                             DS4_N_HEAD_DIM,
                                             1,
                                             DS4_DEFAULT_RMS_EPS);
        if (ok) {
            ok = ds4_gpu_rope_tail_tensor(g->dspark_main_kv[il],
                                          1,
                                          1,
                                          DS4_N_HEAD_DIM,
                                          DS4_N_ROT,
                                          pos,
                                          0,
                                          false,
                                          DS4_ROPE_FREQ_BASE,
                                          1.0f,
                                          0.0f,
                                          1.0f,
                                          DS4_ROPE_YARN_BETA_FAST,
                                          DS4_ROPE_YARN_BETA_SLOW) != 0;
        }
        if (ok) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->dspark_main_kv[il],
                                                     1,
                                                     DS4_N_HEAD_DIM,
                                                     DS4_N_ROT) != 0;
        }
        if (ok) {
            const uint64_t row = (uint64_t)(pos % 128u);
            ok = ds4_gpu_tensor_copy(g->dspark_kv_cache[il],
                                     row * DS4_N_HEAD_DIM * sizeof(float),
                                     g->dspark_main_kv[il],
                                     0,
                                     (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0;
        }
        if (!ok) return false;
    }
    return true;
}

static bool metal_graph_dspark_update_main_kv_batch(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (n_tokens == 0) return true;
    if (n_tokens > 5u) return false;
    if (!g->dspark_main_x) return false;
    const uint32_t ring = start % 128u;
    if (ring + n_tokens > 128u) return false;

    const uint64_t row_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < 3u; il++) {
        if (!g->dspark_main_kv[il] || !g->dspark_kv_cache[il]) return false;
        char kv_name[96];
        char norm_name[96];
        snprintf(kv_name, sizeof(kv_name), "dspark.blk.%u.attn_kv", il);
        snprintf(norm_name, sizeof(norm_name), "dspark.blk.%u.attn_kv_a_norm.weight", il);
        const ds4_dspark_record *kv_rec = ds4_dspark_find_record(g->dspark, kv_name);
        const ds4_dspark_record *norm_rec = ds4_dspark_find_record(g->dspark, norm_name);
        if (!kv_rec || !norm_rec) return false;

        bool ok = ds4_dspark_matmul_record(g->dspark,
                                           kv_rec,
                                           g->dspark_main_kv[il],
                                           g->dspark_main_x,
                                           n_tokens) &&
                  ds4_dspark_rms_norm_record(g->dspark,
                                             norm_rec,
                                             g->dspark_main_kv[il],
                                             g->dspark_main_kv[il],
                                             DS4_N_HEAD_DIM,
                                             n_tokens,
                                             DS4_DEFAULT_RMS_EPS);
        if (ok) {
            ok = ds4_gpu_rope_tail_tensor(g->dspark_main_kv[il],
                                          n_tokens,
                                          1,
                                          DS4_N_HEAD_DIM,
                                          DS4_N_ROT,
                                          start,
                                          0,
                                          false,
                                          DS4_ROPE_FREQ_BASE,
                                          1.0f,
                                          0.0f,
                                          1.0f,
                                          DS4_ROPE_YARN_BETA_FAST,
                                          DS4_ROPE_YARN_BETA_SLOW) != 0;
        }
        if (ok) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->dspark_main_kv[il],
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM,
                                                     DS4_N_ROT) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy(g->dspark_kv_cache[il],
                                     (uint64_t)ring * row_bytes,
                                     g->dspark_main_kv[il],
                                     0,
                                     (uint64_t)n_tokens * row_bytes) != 0;
        }
        if (!ok) return false;
    }
    return true;
}

static bool metal_graph_dspark_rebuild_main_kv_window(
        ds4_gpu_graph *g,
        uint32_t       end_pos) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (end_pos == 0) return true;

    const uint32_t begin = end_pos > 128u ? end_pos - 128u : 0u;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t pos = begin; ok && pos < end_pos; pos++) {
        ok = metal_graph_dspark_project_main(g, pos) &&
             metal_graph_dspark_update_main_kv(g, pos);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && getenv("DS4_DSPARK_SPEC_LOG")) {
        fprintf(stderr,
                "ds4: dspark rebuilt main-kv window positions %u..%u\n",
                begin,
                end_pos - 1u);
    }
    return ok;
}

static bool metal_graph_dspark_update_main_kv_range(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens) {
    if (!g || !g->dspark || !g->dspark->loaded) return true;
    if (n_tokens == 0) return true;
    if (n_tokens <= 5u && !env_flag_enabled("DS4_DSPARK_MAIN_KV_BATCH_DISABLE")) {
        const uint32_t ring = start % 128u;
        if (ring + n_tokens <= 128u) {
            bool batch_ok = ds4_gpu_begin_commands() != 0;
            if (batch_ok) {
                batch_ok = metal_graph_dspark_project_main_range(g, start, n_tokens) &&
                           metal_graph_dspark_update_main_kv_batch(g, start, n_tokens);
            }
            if (batch_ok) batch_ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            if (batch_ok) return true;
        }
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t i = 0; ok && i < n_tokens; i++) {
        const uint32_t pos = start + i;
        ok = metal_graph_dspark_project_main(g, pos) &&
             metal_graph_dspark_update_main_kv(g, pos);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}

static bool metal_graph_dspark_record_pair_file(
        const ds4_dspark_draft   *d,
        const ds4_dspark_record  *a,
        const ds4_dspark_record  *b,
        const ds4_dspark_file   **f_out) {
    const ds4_dspark_file *fa = ds4_dspark_record_file(d, a);
    const ds4_dspark_file *fb = ds4_dspark_record_file(d, b);
    if (!fa || !fb || fa != fb) return false;
    if (f_out) *f_out = fa;
    return true;
}

static bool metal_graph_dspark_block_forward(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       pos,
        uint32_t       n_tokens,
        ds4_gpu_tensor *input_hc,
        ds4_gpu_tensor *output_hc,
        bool           log_probe) {
    if (!g || !g->dspark || !g->dspark->loaded || !input_hc || !output_hc) return false;
    if (il >= 3u) return false;
    if (n_tokens == 0 || n_tokens > 5u) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    char name[128];

    ds4_gpu_tensor *input_hc_view =
        ds4_gpu_tensor_view(input_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    ds4_gpu_tensor *output_hc_view =
        ds4_gpu_tensor_view(output_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    ds4_gpu_tensor *hc_mix_view =
        ds4_gpu_tensor_view(g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view =
        ds4_gpu_tensor_view(g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view =
        ds4_gpu_tensor_view(g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view =
        ds4_gpu_tensor_view(g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = input_hc_view && output_hc_view &&
              hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool block_profile = getenv("DS4_DSPARK_BLOCK_PROFILE") != NULL;
    double block_stage_t = block_profile ? now_sec() : 0.0;
#define DS4_DSPARK_BLOCK_STAGE(label_) do { \
        if (block_profile && ok) { \
            ok = ds4_gpu_flush_commands_blocking() != 0; \
            const double now_ = now_sec(); \
            fprintf(stderr, "ds4: dspark block profile layer=%u %s=%.3f ms\n", \
                    il, (label_), (now_ - block_stage_t) * 1000.0); \
            block_stage_t = now_; \
        } \
    } while (0)

    snprintf(name, sizeof(name), "dspark.blk.%u.hc_attn_fn", il);
    const ds4_dspark_record *hc_fn = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.hc_attn_scale", il);
    const ds4_dspark_record *hc_scale = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.hc_attn_base", il);
    const ds4_dspark_record *hc_base = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_norm_weight", il);
    const ds4_dspark_record *attn_norm = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_q_a", il);
    const ds4_dspark_record *q_a = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_q_a_norm.weight", il);
    const ds4_dspark_record *q_norm = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_q_b", il);
    const ds4_dspark_record *q_b = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_kv", il);
    const ds4_dspark_record *kv = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_kv_a_norm.weight", il);
    const ds4_dspark_record *kv_norm = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_sinks.weight", il);
    const ds4_dspark_record *sinks = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_output_a", il);
    const ds4_dspark_record *out_a = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.attn_output_b", il);
    const ds4_dspark_record *out_b = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.hc_ffn_fn", il);
    const ds4_dspark_record *ffn_hc_fn = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.hc_ffn_scale", il);
    const ds4_dspark_record *ffn_hc_scale = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.hc_ffn_base", il);
    const ds4_dspark_record *ffn_hc_base = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_norm_weight", il);
    const ds4_dspark_record *ffn_norm = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_gate_shexp", il);
    const ds4_dspark_record *shared_gate = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_up_shexp", il);
    const ds4_dspark_record *shared_up = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_down_shexp", il);
    const ds4_dspark_record *shared_down = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_gate_inp.weight", il);
    const ds4_dspark_record *router = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.exp_probs_b.bias", il);
    const ds4_dspark_record *router_bias = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_gate_exps", il);
    const ds4_dspark_record *routed_gate = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_up_exps", il);
    const ds4_dspark_record *routed_up = ds4_dspark_find_record(g->dspark, name);
    snprintf(name, sizeof(name), "dspark.blk.%u.ffn_down_exps", il);
    const ds4_dspark_record *routed_down = ds4_dspark_find_record(g->dspark, name);

    const ds4_dspark_file *hc_param_file = NULL;
    const ds4_dspark_file *ffn_hc_param_file = NULL;
    const ds4_dspark_file *router_bias_file = NULL;
    const ds4_dspark_file *sinks_file = NULL;
    const ds4_dspark_file *routed_gate_file = NULL;
    const ds4_dspark_file *routed_up_file = NULL;
    const ds4_dspark_file *routed_down_file = NULL;
    if (ok) {
        ok = hc_fn && hc_scale && hc_base && attn_norm && q_a && q_norm && q_b &&
             kv && kv_norm && sinks && out_a && out_b &&
             ffn_hc_fn && ffn_hc_scale && ffn_hc_base && ffn_norm &&
             shared_gate && shared_up && shared_down && router && router_bias &&
             routed_gate && routed_up && routed_down &&
             metal_graph_dspark_record_pair_file(g->dspark, hc_scale, hc_base, &hc_param_file) &&
             metal_graph_dspark_record_pair_file(g->dspark, ffn_hc_scale, ffn_hc_base, &ffn_hc_param_file) &&
             (router_bias_file = ds4_dspark_record_file(g->dspark, router_bias)) != NULL &&
             (sinks_file = ds4_dspark_record_file(g->dspark, sinks)) != NULL &&
             (routed_gate_file = ds4_dspark_record_file(g->dspark, routed_gate)) != NULL &&
             (routed_up_file = ds4_dspark_record_file(g->dspark, routed_up)) != NULL &&
             (routed_down_file = ds4_dspark_record_file(g->dspark, routed_down)) != NULL;
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                input_hc_view,
                                                (uint32_t)hc_dim,
                                                n_tokens,
                                                DS4_RMS_EPS) != 0 &&
             ds4_dspark_matmul_record(g->dspark,
                                       hc_fn,
                                       hc_mix_view,
                                       g->batch_flat_hc,
                                       n_tokens) &&
             ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                  hc_split_view,
                                                  hc_mix_view,
                                                  input_hc_view,
                                                  hc_param_file->map,
                                                  hc_param_file->size,
                                                  hc_scale->offset,
                                                  hc_base->offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
             ds4_dspark_rms_norm_record(g->dspark,
                                         attn_norm,
                                         g->batch_attn_norm,
                                         attn_cur_view,
                                         DS4_N_EMBD,
                                         n_tokens,
                                         DS4_DEFAULT_RMS_EPS);
    }
    DS4_DSPARK_BLOCK_STAGE("hc_attn");
    if (ok) {
        ok = ds4_dspark_matmul_record(g->dspark, q_a, g->batch_qr, g->batch_attn_norm, n_tokens) &&
             ds4_dspark_rms_norm_record(g->dspark,
                                         q_norm,
                                         g->batch_qr_norm,
                                         g->batch_qr,
                                         DS4_N_LORA_Q,
                                         n_tokens,
                                         DS4_DEFAULT_RMS_EPS) &&
             ds4_dspark_matmul_record(g->dspark, q_b, g->batch_q, g->batch_qr_norm, n_tokens) &&
             ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                          n_tokens,
                                          DS4_N_HEAD,
                                          DS4_N_HEAD_DIM,
                                          DS4_RMS_EPS) != 0 &&
             ds4_gpu_rope_tail_tensor(g->batch_q,
                                      n_tokens,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos + 1u,
                                      0,
                                      false,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_DSPARK_BLOCK_STAGE("q_proj");
    if (ok) {
        ok = ds4_dspark_matmul_record(g->dspark, kv, g->batch_kv_raw, g->batch_attn_norm, n_tokens) &&
             ds4_dspark_rms_norm_record(g->dspark,
                                         kv_norm,
                                         g->batch_kv,
                                         g->batch_kv_raw,
                                         DS4_N_HEAD_DIM,
                                         n_tokens,
                                         DS4_DEFAULT_RMS_EPS) &&
             ds4_gpu_rope_tail_tensor(g->batch_kv,
                                      n_tokens,
                                      1,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos + 1u,
                                      0,
                                      false,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0 &&
             ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                  n_tokens,
                                                  DS4_N_HEAD_DIM,
                                                  DS4_N_ROT) != 0;
    }
    DS4_DSPARK_BLOCK_STAGE("kv_proj");
    if (ok) {
        const uint32_t n_main = pos + 1u < 128u ? pos + 1u : 128u;
        ok = ds4_gpu_dspark_attention_heads_tensor(g->batch_heads,
                                                   sinks_file->map,
                                                   sinks_file->size,
                                                   sinks->offset,
                                                   g->batch_q,
                                                   g->dspark_kv_cache[il],
                                                   g->batch_kv,
                                                   n_tokens,
                                                   n_main,
                                                   DS4_N_HEAD,
                                                   DS4_N_HEAD_DIM) != 0 &&
             ds4_gpu_rope_tail_tensor(g->batch_heads,
                                      n_tokens,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos + 1u,
                                      0,
                                      true,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_DSPARK_BLOCK_STAGE("attn");
    if (ok) {
        const uint32_t n_groups = DS4_N_OUT_GROUP;
        const uint32_t group_heads = DS4_N_HEAD / n_groups;
        const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
        const uint32_t rank = DS4_N_LORA_O;
        const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
        const uint64_t out_low_dim = (uint64_t)n_groups * rank;
        for (uint32_t t = 0; ok && t < n_tokens; t++) {
            for (uint32_t group = 0; ok && group < n_groups; group++) {
                ds4_gpu_tensor *heads_group =
                    ds4_gpu_tensor_view(g->batch_heads,
                                        ((uint64_t)t * q_dim + (uint64_t)group * group_dim) * sizeof(float),
                                        (uint64_t)group_dim * sizeof(float));
                ds4_gpu_tensor *low_group =
                    ds4_gpu_tensor_view(g->batch_attn_low,
                                        ((uint64_t)t * out_low_dim + (uint64_t)group * rank) * sizeof(float),
                                        (uint64_t)rank * sizeof(float));
                ok = heads_group && low_group &&
                     ds4_dspark_matmul_fp8_record_row_slice(g->dspark,
                                                            out_a,
                                                            low_group,
                                                            heads_group,
                                                            (uint64_t)group * rank,
                                                            rank,
                                                            1);
                ds4_gpu_tensor_free(low_group);
                ds4_gpu_tensor_free(heads_group);
            }
        }
        if (ok) {
            ok = ds4_dspark_matmul_record(g->dspark,
                                          out_b,
                                          g->batch_attn_out,
                                          g->batch_attn_low,
                                          n_tokens) &&
                 ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                g->batch_attn_out,
                                                input_hc_view,
                                                hc_split_view,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
        }
    }
    DS4_DSPARK_BLOCK_STAGE("attn_out");
    ds4_gpu_tensor *ffn_cur_view =
        ds4_gpu_tensor_view(g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    bool ffn_view_ok = ffn_cur_view != NULL;
    if (ok) ok = ffn_view_ok;
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                after_attn_hc_view,
                                                (uint32_t)hc_dim,
                                                n_tokens,
                                                DS4_RMS_EPS) != 0 &&
             ds4_dspark_matmul_record(g->dspark,
                                       ffn_hc_fn,
                                       hc_mix_view,
                                       g->batch_flat_hc,
                                       n_tokens) &&
             ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                  hc_split_view,
                                                  hc_mix_view,
                                                  after_attn_hc_view,
                                                  ffn_hc_param_file->map,
                                                  ffn_hc_param_file->size,
                                                  ffn_hc_scale->offset,
                                                  ffn_hc_base->offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
             ds4_dspark_rms_norm_record(g->dspark,
                                         ffn_norm,
                                         g->batch_ffn_norm,
                                         ffn_cur_view,
                                         DS4_N_EMBD,
                                         n_tokens,
                                         DS4_DEFAULT_RMS_EPS);
    }
    DS4_DSPARK_BLOCK_STAGE("ffn_hc");
    if (ok) {
        ok = ds4_dspark_matmul_record(g->dspark,
                                      shared_gate,
                                      g->batch_shared_gate,
                                      g->batch_ffn_norm,
                                      n_tokens) &&
             ds4_dspark_matmul_record(g->dspark,
                                      shared_up,
                                      g->batch_shared_up,
                                      g->batch_ffn_norm,
                                      n_tokens) &&
             ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                   g->batch_shared_gate,
                                   g->batch_shared_up,
                                   n_tokens * DS4_N_FF_EXP,
                                   DS4_SWIGLU_CLAMP_EXP,
                                   1.0f) != 0 &&
             ds4_dspark_matmul_record(g->dspark,
                                      shared_down,
                                      g->batch_shared_out,
                                      g->batch_shared_mid,
                                      n_tokens) &&
             ds4_dspark_matmul_record(g->dspark,
                                      router,
                                      g->batch_router_logits,
                                      g->batch_ffn_norm,
                                      n_tokens) &&
             ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                g->batch_router_weights,
                                                g->batch_router_probs,
                                                router_bias_file->map,
                                                router_bias_file->size,
                                                router_bias->offset,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                true,
                                                false,
                                                g->batch_router_logits,
                                                g->dspark_input_ids,
                                                DS4_N_EXPERT,
                                                DS4_N_EXPERT_ACTIVE_USED,
                                                DS4_EXPERT_WEIGHT_SCALE,
                                                n_tokens) != 0;
    }
    DS4_DSPARK_BLOCK_STAGE("shared_router");
    if (ok) {
        ok = ds4_gpu_dspark_mxfp4_routed_batch_tensor(
                 g->batch_routed_out,
                 g->batch_routed_gate,
                 g->batch_routed_up,
                 g->batch_routed_mid,
                 routed_gate_file->map,
                 routed_gate_file->size,
                 routed_gate->plane_data_offset,
                 routed_gate->plane_data_bytes,
                 routed_gate->plane_scale_offset,
                 routed_gate->plane_scale_bytes,
                 routed_gate->plane_data_bytes_per_expert,
                 routed_gate->plane_scale_bytes_per_expert,
                 routed_up_file->map,
                 routed_up_file->size,
                 routed_up->plane_data_offset,
                 routed_up->plane_data_bytes,
                 routed_up->plane_scale_offset,
                 routed_up->plane_scale_bytes,
                 routed_up->plane_data_bytes_per_expert,
                 routed_up->plane_scale_bytes_per_expert,
                 routed_down_file->map,
                 routed_down_file->size,
                 routed_down->plane_data_offset,
                 routed_down->plane_data_bytes,
                 routed_down->plane_scale_offset,
                 routed_down->plane_scale_bytes,
                 routed_down->plane_data_bytes_per_expert,
                 routed_down->plane_scale_bytes_per_expert,
                 (uint32_t)g->dspark->expert_count,
                 DS4_N_EMBD,
                 DS4_N_FF_EXP,
                 DS4_N_EMBD,
                 g->batch_router_selected,
                 g->batch_router_weights,
                 DS4_N_EXPERT_ACTIVE_USED,
                 DS4_SWIGLU_CLAMP_EXP,
                 g->batch_ffn_norm,
                 n_tokens) != 0 &&
             ds4_gpu_hc_expand_add_split_tensor(output_hc_view,
                                                g->batch_routed_out,
                                                g->batch_shared_out,
                                                after_attn_hc_view,
                                                hc_split_view,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }
    DS4_DSPARK_BLOCK_STAGE("routed");

    if (log_probe && ok && ds4_gpu_flush_commands_blocking() != 0) {
        float q_sample[16] = {0};
        float kv_sample[16] = {0};
        float heads_sample[16] = {0};
        float attn_out_sample[16] = {0};
        float shared_sample[16] = {0};
        float routed_sample[16] = {0};
        float ffn_hc_sample[16] = {0};
        int32_t selected_sample[6] = {0};
        float router_weight_sample[6] = {0};
        ok = ds4_gpu_tensor_read(g->batch_q, 0, q_sample, sizeof(q_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_kv, 0, kv_sample, sizeof(kv_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_heads, 0, heads_sample, sizeof(heads_sample)) != 0 &&
             ds4_gpu_tensor_read(after_attn_hc_view, 0, attn_out_sample, sizeof(attn_out_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_shared_out, 0, shared_sample, sizeof(shared_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_routed_out, 0, routed_sample, sizeof(routed_sample)) != 0 &&
             ds4_gpu_tensor_read(output_hc_view, 0, ffn_hc_sample, sizeof(ffn_hc_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_router_selected, 0, selected_sample, sizeof(selected_sample)) != 0 &&
             ds4_gpu_tensor_read(g->batch_router_weights, 0, router_weight_sample, sizeof(router_weight_sample)) != 0;
        if (ok) {
            double q_ss = 0.0;
            double kv_ss = 0.0;
            double heads_ss = 0.0;
            double attn_out_ss = 0.0;
            double shared_ss = 0.0;
            double routed_ss = 0.0;
            double ffn_hc_ss = 0.0;
            for (size_t i = 0; i < sizeof(q_sample) / sizeof(q_sample[0]); i++) {
                q_ss += (double)q_sample[i] * (double)q_sample[i];
                kv_ss += (double)kv_sample[i] * (double)kv_sample[i];
                heads_ss += (double)heads_sample[i] * (double)heads_sample[i];
                attn_out_ss += (double)attn_out_sample[i] * (double)attn_out_sample[i];
                shared_ss += (double)shared_sample[i] * (double)shared_sample[i];
                routed_ss += (double)routed_sample[i] * (double)routed_sample[i];
                ffn_hc_ss += (double)ffn_hc_sample[i] * (double)ffn_hc_sample[i];
            }
            fprintf(stderr,
                    "ds4: DSpark layer%u block probe q_rms=%.6g kv_rms=%.6g heads_rms=%.6g attn_hc_rms=%.6g shared_rms=%.6g routed_rms=%.6g ffn_hc_rms=%.6g router0=[%d,%d,%d,%d,%d,%d] w0=[%.4g,%.4g,%.4g,%.4g,%.4g,%.4g]\n",
                    il,
                    sqrt(q_ss / (double)(sizeof(q_sample) / sizeof(q_sample[0]))),
                    sqrt(kv_ss / (double)(sizeof(kv_sample) / sizeof(kv_sample[0]))),
                    sqrt(heads_ss / (double)(sizeof(heads_sample) / sizeof(heads_sample[0]))),
                    sqrt(attn_out_ss / (double)(sizeof(attn_out_sample) / sizeof(attn_out_sample[0]))),
                    sqrt(shared_ss / (double)(sizeof(shared_sample) / sizeof(shared_sample[0]))),
                    sqrt(routed_ss / (double)(sizeof(routed_sample) / sizeof(routed_sample[0]))),
                    sqrt(ffn_hc_ss / (double)(sizeof(ffn_hc_sample) / sizeof(ffn_hc_sample[0]))),
                    selected_sample[0],
                    selected_sample[1],
                    selected_sample[2],
                    selected_sample[3],
                    selected_sample[4],
                    selected_sample[5],
                    router_weight_sample[0],
                    router_weight_sample[1],
                    router_weight_sample[2],
                    router_weight_sample[3],
                    router_weight_sample[4],
                    router_weight_sample[5]);
        }
    }

    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(output_hc_view);
    ds4_gpu_tensor_free(input_hc_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_DSPARK_BLOCK_STAGE
    return ok;
}

static bool metal_graph_dspark_output_head_probe(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        ds4_gpu_tensor    *final_hc,
        uint32_t           n_tokens,
        bool               log_probe) {
    if (!g || !g->dspark || !g->dspark->loaded || !model || !weights ||
        !final_hc || !g->spec_logits) {
        return false;
    }
    if (n_tokens == 0 || n_tokens > 5u) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

    const ds4_dspark_record *hc_head_fn =
        ds4_dspark_find_record(g->dspark, "dspark.blk.2.hc_head_fn");
    const ds4_dspark_record *hc_head_scale =
        ds4_dspark_find_record(g->dspark, "dspark.blk.2.hc_head_scale");
    const ds4_dspark_record *hc_head_base =
        ds4_dspark_find_record(g->dspark, "dspark.blk.2.hc_head_base");
    const ds4_dspark_record *output_norm =
        ds4_dspark_find_record(g->dspark, "dspark.blk.2.output_norm.weight");
    const ds4_dspark_file *hc_head_file = NULL;
    bool ok = hc_head_fn && hc_head_scale && hc_head_base && output_norm &&
              metal_graph_dspark_record_pair_file(g->dspark,
                                                  hc_head_scale,
                                                  hc_head_base,
                                                  &hc_head_file);

    ds4_gpu_tensor *final_hc_view = NULL;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm_view = NULL;
    ds4_gpu_tensor *logits = NULL;
    if (ok) {
        final_hc_view =
            ds4_gpu_tensor_view(final_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
        output_pre =
            ds4_gpu_tensor_view(g->batch_hc_mix, 0,
                                (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
        output_weights =
            ds4_gpu_tensor_view(g->batch_hc_split, 0,
                                (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
        output_embd =
            ds4_gpu_tensor_view(g->batch_ffn_cur, 0,
                                (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
        output_norm_view =
            ds4_gpu_tensor_view(g->batch_ffn_norm, 0,
                                (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
        logits =
            ds4_gpu_tensor_view(g->spec_logits, 0,
                                (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(float));
        ok = final_hc_view && output_pre && output_weights &&
             output_embd && output_norm_view && logits;
    }

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                     final_hc_view,
                                                     (uint32_t)hc_dim,
                                                     n_tokens,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_dspark_matmul_record(g->dspark,
                                           hc_head_fn,
                                           output_pre,
                                           g->batch_flat_hc,
                                           n_tokens);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                   output_pre,
                                                   hc_head_file->map,
                                                   hc_head_file->size,
                                                   hc_head_scale->offset,
                                                   hc_head_base->offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                 final_hc_view,
                                                 output_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_dspark_rms_norm_record(g->dspark,
                                             output_norm,
                                             output_norm_view,
                                             output_embd,
                                             DS4_N_EMBD,
                                             n_tokens,
                                             DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor_ex(logits,
                                                model->map,
                                                model->size,
                                                weights->output->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_N_VOCAB,
                                                output_norm_view,
                                                n_tokens,
                                                DS4_MM_NO_I8) != 0;

    if (log_probe && ok && ds4_gpu_flush_commands_blocking() != 0) {
        float *row = xmalloc((size_t)DS4_N_VOCAB * sizeof(row[0]));
        int top[5] = { -1, -1, -1, -1, -1 };
        for (uint32_t i = 0; ok && i < n_tokens; i++) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     (uint64_t)i * DS4_N_VOCAB * sizeof(float),
                                     row,
                                     (uint64_t)DS4_N_VOCAB * sizeof(row[0])) != 0;
            if (ok) top[i] = sample_argmax(row, DS4_N_VOCAB);
        }
        if (ok) {
            fprintf(stderr,
                    "ds4: DSpark head probe base_top=[%d,%d,%d,%d,%d]\n",
                    top[0], top[1], top[2], top[3], top[4]);
        }
        free(row);
    }

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm_view);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    ds4_gpu_tensor_free(final_hc_view);
    return ok;
}

static bool metal_graph_dspark_three_layer_forward(
        ds4_gpu_graph *g,
        const ds4_model *model,
        const ds4_weights *weights,
        uint32_t       pos,
        uint32_t       n_tokens,
        bool           log_probe) {
    if (!g || !g->dspark || !g->dspark->loaded ||
        !g->dspark_h || !g->batch_next_hc || !g->batch_cur_hc) {
        return false;
    }
    if (n_tokens == 0 || n_tokens > 5u) return false;

    const bool profile = getenv("DS4_DSPARK_GRAPH_PROFILE") != NULL;
    double t_stage = profile ? now_sec() : 0.0;
    bool ok = metal_graph_dspark_block_forward(g, 0, pos, n_tokens,
                                               g->dspark_h,
                                               g->batch_next_hc,
                                               log_probe);
    if (profile && ok) {
        ok = ds4_gpu_flush_commands_blocking() != 0;
        const double now = now_sec();
        fprintf(stderr, "ds4: dspark graph profile block0=%.3f ms\n", (now - t_stage) * 1000.0);
        t_stage = now;
    }
    if (ok) {
        ok = metal_graph_dspark_block_forward(g, 1, pos, n_tokens,
                                              g->batch_next_hc,
                                              g->batch_cur_hc,
                                              log_probe);
    }
    if (profile && ok) {
        ok = ds4_gpu_flush_commands_blocking() != 0;
        const double now = now_sec();
        fprintf(stderr, "ds4: dspark graph profile block1=%.3f ms\n", (now - t_stage) * 1000.0);
        t_stage = now;
    }
    if (ok) {
        ok = metal_graph_dspark_block_forward(g, 2, pos, n_tokens,
                                              g->batch_cur_hc,
                                              g->batch_next_hc,
                                              log_probe);
    }
    if (profile && ok) {
        ok = ds4_gpu_flush_commands_blocking() != 0;
        const double now = now_sec();
        fprintf(stderr, "ds4: dspark graph profile block2=%.3f ms\n", (now - t_stage) * 1000.0);
        t_stage = now;
    }
    if (ok) {
        ok = metal_graph_dspark_output_head_probe(g,
                                                  model,
                                                  weights,
                                                  g->batch_next_hc,
                                                  n_tokens,
                                                  log_probe);
    }
    if (profile && ok) {
        ok = ds4_gpu_flush_commands_blocking() != 0;
        const double now = now_sec();
        fprintf(stderr, "ds4: dspark graph profile head=%.3f ms\n", (now - t_stage) * 1000.0);
    }
    return ok;
}

static bool metal_graph_dspark_three_layer_probe(
        ds4_gpu_graph *g,
        const ds4_model *model,
        const ds4_weights *weights,
        uint32_t       pos) {
    const char *env = getenv("DS4_DSPARK_QKV_LOG");
    if (!env || !env[0] || atoi(env) == 0) return true;
    return metal_graph_dspark_three_layer_forward(g, model, weights, pos, 5, true);
}

static bool metal_graph_dspark_markov_step(
        ds4_gpu_graph *g,
        uint32_t       row,
        int            prev_token,
        int           *next_token) {
    if (!g || !g->dspark || !g->dspark->loaded || !g->spec_logits ||
        !g->logits || !g->comp_selected || !g->dspark_input_ids ||
        !next_token || row >= 5u || prev_token < 0 ||
        prev_token >= (int)DS4_N_VOCAB) {
        return false;
    }
    const ds4_dspark_record *markov_embd =
        ds4_dspark_find_record(g->dspark, "dspark.markov_embd.weight");
    const ds4_dspark_record *markov_output =
        ds4_dspark_find_record(g->dspark, "dspark.markov_output.weight");
    if (!markov_embd || !markov_output) return false;
    const ds4_dspark_file *markov_file =
        ds4_dspark_record_file(g->dspark, markov_embd);
    if (!markov_file ||
        markov_embd->kind != DS4_DSPARK_REC_F16 ||
        markov_embd->ndim != 2 ||
        markov_embd->shape[0] != DS4_N_VOCAB ||
        markov_embd->shape[1] != (uint64_t)g->dspark->markov_rank) {
        return false;
    }

    const uint32_t rank = (uint32_t)g->dspark->markov_rank;
    const uint64_t rank_bytes = (uint64_t)rank * sizeof(float);
    ds4_gpu_tensor *markov_embed = NULL;
    if (g->batch_low_tmp && ds4_gpu_tensor_bytes(g->batch_low_tmp) >= rank_bytes) {
        markov_embed = ds4_gpu_tensor_view(g->batch_low_tmp, 0, rank_bytes);
    } else if (g->batch_attn_low && ds4_gpu_tensor_bytes(g->batch_attn_low) >= rank_bytes) {
        markov_embed = ds4_gpu_tensor_view(g->batch_attn_low, 0, rank_bytes);
    }
    ds4_gpu_tensor *logit_row =
        ds4_gpu_tensor_view(g->spec_logits,
                            (uint64_t)row * DS4_N_VOCAB * sizeof(float),
                            (uint64_t)DS4_N_VOCAB * sizeof(float));
    bool ok = markov_embed && logit_row;
    int32_t token_i32 = (int32_t)prev_token;
    if (ok) ok = ds4_gpu_tensor_write(g->dspark_input_ids,
                                      0,
                                      &token_i32,
                                      sizeof(token_i32)) != 0;
    if (ok) ok = ds4_gpu_gather_rows_f16_mmap_tensor(
                    markov_embed,
                    markov_file->map,
                    markov_file->size,
                    markov_embd->offset,
                    g->dspark_input_ids,
                    DS4_N_VOCAB,
                    1,
                    rank) != 0;
    if (ok) ok = ds4_dspark_matmul_record(g->dspark,
                                          markov_output,
                                          g->logits,
                                          markov_embed,
                                          1);
    if (ok) ok = ds4_gpu_add_tensor(logit_row,
                                    logit_row,
                                    g->logits,
                                    DS4_N_VOCAB) != 0;
    if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                             logit_row,
                                             DS4_N_VOCAB,
                                             1,
                                             1) != 0;
    ds4_gpu_tensor_free(logit_row);
    ds4_gpu_tensor_free(markov_embed);
    (void)next_token;
    return ok;
}

static bool metal_graph_dspark_markov_chain_fast(
        ds4_gpu_graph *g,
        int            last_token,
        int           *drafts,
        int            draft_cap,
        int           *drafted) {
    if (!g || !g->dspark || !g->dspark->loaded || !g->spec_logits ||
        !g->logits || !g->comp_selected || !g->dspark_input_ids ||
        !drafts || draft_cap <= 0 || draft_cap > 5 ||
        last_token < 0 || last_token >= (int)DS4_N_VOCAB) {
        return false;
    }

    const ds4_dspark_record *markov_embd =
        ds4_dspark_find_record(g->dspark, "dspark.markov_embd.weight");
    const ds4_dspark_record *markov_output =
        ds4_dspark_find_record(g->dspark, "dspark.markov_output.weight");
    if (!markov_embd || !markov_output) return false;
    const ds4_dspark_file *markov_file =
        ds4_dspark_record_file(g->dspark, markov_embd);
    if (!markov_file ||
        markov_embd->kind != DS4_DSPARK_REC_F16 ||
        markov_embd->ndim != 2 ||
        markov_embd->shape[0] != DS4_N_VOCAB ||
        markov_embd->shape[1] != (uint64_t)g->dspark->markov_rank ||
        ds4_gpu_tensor_bytes(g->dspark_input_ids) <
            (uint64_t)(draft_cap + 1) * sizeof(int32_t)) {
        return false;
    }

    const uint32_t rank = (uint32_t)g->dspark->markov_rank;
    const uint64_t rank_bytes = (uint64_t)rank * sizeof(float);
    if (!rank || rank > 1024u) return false;

    int32_t prev0 = (int32_t)last_token;
    bool ok = ds4_gpu_tensor_write(g->dspark_input_ids,
                                   0,
                                   &prev0,
                                   sizeof(prev0)) != 0 &&
              ds4_gpu_begin_commands() != 0;
    for (int row = 0; ok && row < draft_cap; row++) {
        ds4_gpu_tensor *prev_id =
            ds4_gpu_tensor_view(g->dspark_input_ids,
                                (uint64_t)row * sizeof(int32_t),
                                sizeof(int32_t));
        ds4_gpu_tensor *markov_embed = NULL;
        if (g->batch_low_tmp && ds4_gpu_tensor_bytes(g->batch_low_tmp) >= rank_bytes) {
            markov_embed = ds4_gpu_tensor_view(g->batch_low_tmp, 0, rank_bytes);
        } else if (g->batch_attn_low && ds4_gpu_tensor_bytes(g->batch_attn_low) >= rank_bytes) {
            markov_embed = ds4_gpu_tensor_view(g->batch_attn_low, 0, rank_bytes);
        }
        ds4_gpu_tensor *logit_row =
            ds4_gpu_tensor_view(g->spec_logits,
                                (uint64_t)row * DS4_N_VOCAB * sizeof(float),
                                (uint64_t)DS4_N_VOCAB * sizeof(float));
        ok = prev_id && markov_embed && logit_row;
        if (ok) {
            ok = ds4_gpu_gather_rows_f16_mmap_tensor(
                    markov_embed,
                    markov_file->map,
                    markov_file->size,
                    markov_embd->offset,
                    prev_id,
                    DS4_N_VOCAB,
                    1,
                    rank) != 0;
        }
        if (ok) {
            ok = ds4_dspark_matmul_record(g->dspark,
                                          markov_output,
                                          g->logits,
                                          markov_embed,
                                          1);
        }
        if (ok) {
            ok = ds4_gpu_add_tensor(logit_row,
                                    logit_row,
                                    g->logits,
                                    DS4_N_VOCAB) != 0;
        }
        if (ok) {
            ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                             logit_row,
                                             DS4_N_VOCAB,
                                             1,
                                             1) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy(g->dspark_input_ids,
                                     (uint64_t)(row + 1) * sizeof(int32_t),
                                     g->comp_selected,
                                     0,
                                     sizeof(int32_t)) != 0;
        }
        ds4_gpu_tensor_free(logit_row);
        ds4_gpu_tensor_free(markov_embed);
        ds4_gpu_tensor_free(prev_id);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (!ok) return false;

    int32_t ids[5] = {0, 0, 0, 0, 0};
    ok = ds4_gpu_tensor_read(g->dspark_input_ids,
                             sizeof(int32_t),
                             ids,
                             (uint64_t)draft_cap * sizeof(ids[0])) != 0;
    for (int i = 0; ok && i < draft_cap; i++) {
        if (ids[i] < 0 || ids[i] >= (int32_t)DS4_N_VOCAB) {
            ok = false;
            break;
        }
        drafts[i] = ids[i];
        if (drafted) *drafted = i + 1;
        if (getenv("DS4_DSPARK_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: dspark draft row=%d prev=%d next=%d%s\n",
                    i,
                    i == 0 ? last_token : drafts[i - 1],
                    drafts[i],
                    " fast-markov");
        }
    }
    return ok && draft_cap > 0;
}

static float ds4_sigmoidf_clamped(float x) {
    if (x >= 40.0f) return 1.0f;
    if (x <= -40.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static bool metal_graph_dspark_confidence_score(
        ds4_gpu_graph *g,
        uint32_t       row,
        float         *logit_out,
        float         *prob_out) {
    if (!g || !g->dspark || !g->dspark->loaded ||
        !g->batch_ffn_cur || row >= 5u ||
        (!logit_out && !prob_out)) {
        return false;
    }
    const uint32_t rank = (uint32_t)g->dspark->markov_rank;
    if (rank == 0 || rank > 1024u) return false;
    const ds4_dspark_record *proj =
        ds4_dspark_find_record(g->dspark, "dspark.confidence_proj.weight");
    if (!proj) return false;
    const ds4_dspark_file *file = ds4_dspark_record_file(g->dspark, proj);
    uint64_t out_dim = 0;
    uint64_t in_dim = 0;
    if (!file ||
        !ds4_dspark_record_matrix_dims(proj, &out_dim, &in_dim) ||
        proj->kind != DS4_DSPARK_REC_F16 ||
        out_dim != 1 ||
        in_dim != (uint64_t)DS4_N_EMBD + rank ||
        proj->offset > file->size ||
        (uint64_t)in_dim * sizeof(uint16_t) > file->size - proj->offset) {
        return false;
    }

    const uint64_t rank_bytes = (uint64_t)rank * sizeof(float);
    ds4_gpu_tensor *markov_embed = NULL;
    if (g->batch_low_tmp && ds4_gpu_tensor_bytes(g->batch_low_tmp) >= rank_bytes) {
        markov_embed = g->batch_low_tmp;
    } else if (g->batch_attn_low && ds4_gpu_tensor_bytes(g->batch_attn_low) >= rank_bytes) {
        markov_embed = g->batch_attn_low;
    }
    if (!markov_embed) return false;

    float *hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(hidden[0]));
    float *markov = xmalloc((size_t)rank * sizeof(markov[0]));
    bool ok = ds4_gpu_tensor_read(g->batch_ffn_cur,
                                  (uint64_t)row * DS4_N_EMBD * sizeof(float),
                                  hidden,
                                  (uint64_t)DS4_N_EMBD * sizeof(hidden[0])) != 0 &&
              ds4_gpu_tensor_read(markov_embed,
                                  0,
                                  markov,
                                  rank_bytes) != 0;
    float score = 0.0f;
    if (ok) {
        const uint16_t *w = (const uint16_t *)(const void *)(file->map + proj->offset);
        score = dot_f16_row(w, hidden, DS4_N_EMBD) +
                dot_f16_row(w + DS4_N_EMBD, markov, rank);
    }
    free(markov);
    free(hidden);
    if (!ok) return false;

    if (logit_out) *logit_out = score;
    if (prob_out) *prob_out = ds4_sigmoidf_clamped(score);
    return true;
}

static int ds4_dspark_confident_prefix_len(
        const float *confidence_probs,
        int          draft_n,
        float        threshold) {
    if (!confidence_probs || draft_n <= 0 || threshold <= 0.0f) return draft_n;
    for (int i = 0; i < draft_n; i++) {
        if (confidence_probs[i] < threshold) return i;
    }
    return draft_n;
}

static bool metal_graph_eval_dspark_draft(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                last_token,
        uint32_t           pos,
        int               *drafts,
        int                draft_cap,
        float             *confidence_logits,
        float             *confidence_probs,
        int               *drafted) {
    if (drafted) *drafted = 0;
    if (!g || !g->dspark || !g->dspark->loaded || !model || !weights ||
        !drafts || draft_cap <= 0 || last_token < 0 ||
        last_token >= (int)DS4_N_VOCAB) {
        return false;
    }
    if (draft_cap > g->dspark->block_size) draft_cap = g->dspark->block_size;
    if (draft_cap > 5) draft_cap = 5;
    if (draft_cap <= 0) return false;

    const bool draft_profile = getenv("DS4_DSPARK_DRAFT_PROFILE") != NULL;
    const double t0 = draft_profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_dspark_seed_block(g, model, weights, last_token, (uint32_t)draft_cap);
    if (ok) ok = metal_graph_dspark_three_layer_forward(g, model, weights, pos, (uint32_t)draft_cap, false);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (!ok) return false;
    const double graph_done = draft_profile ? now_sec() : 0.0;

    int n = 0;
    const bool want_confidence = confidence_logits || confidence_probs;
    const char *markov_disable = getenv("DS4_DSPARK_MARKOV_CHAIN_DISABLE");
    const bool fast_markov =
        !want_confidence &&
        !(markov_disable && markov_disable[0] && atoi(markov_disable) != 0);
    if (fast_markov) {
        ok = metal_graph_dspark_markov_chain_fast(g,
                                                  last_token,
                                                  drafts,
                                                  draft_cap,
                                                  drafted);
        if (!ok) return false;
        n = draft_cap;
    } else {
        int prev = last_token;
        for (; n < draft_cap; n++) {
            int next = -1;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_dspark_markov_step(g, (uint32_t)n, prev, &next);
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            if (ok) {
                ok = ds4_gpu_tensor_read(g->comp_selected,
                                         0,
                                         &next,
                                         sizeof(next)) != 0;
            }
            if (ok && want_confidence) {
                ok = metal_graph_dspark_confidence_score(
                        g,
                        (uint32_t)n,
                        confidence_logits ? &confidence_logits[n] : NULL,
                        confidence_probs ? &confidence_probs[n] : NULL);
            }
            if (!ok || next < 0) return false;
            drafts[n] = next;
            prev = next;
            if (drafted) *drafted = n + 1;
            if (getenv("DS4_DSPARK_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: dspark draft row=%d prev=%d next=%d\n",
                        n,
                        n == 0 ? last_token : drafts[n - 1],
                        next);
            }
        }
    }
    if (draft_profile) {
        const double done = now_sec();
        fprintf(stderr,
                "ds4: dspark draft profile cap=%d rows=%d graph=%.3f ms markov=%.3f ms total=%.3f ms\n",
                draft_cap,
                n,
                (graph_done - t0) * 1000.0,
                (done - graph_done) * 1000.0,
                (done - t0) * 1000.0);
    }
    return n > 0;
}

static void metal_graph_log_dspark_kv(ds4_gpu_graph *g) {
    const char *env = getenv("DS4_DSPARK_KV_LOG");
    if (!g || !g->dspark_main_kv[0] || !env || !env[0] || atoi(env) == 0) return;
    if (ds4_gpu_flush_commands_blocking() == 0) return;
    float sample[16] = {0};
    if (ds4_gpu_tensor_read(g->dspark_main_kv[0],
                            0,
                            sample,
                            sizeof(sample)) != 0) {
        double ss = 0.0;
        for (size_t i = 0; i < sizeof(sample) / sizeof(sample[0]); i++) {
            ss += (double)sample[i] * (double)sample[i];
        }
        fprintf(stderr,
                "ds4: DSpark main_kv[0] sample_rms=%.6g\n",
                sqrt(ss / (double)(sizeof(sample) / sizeof(sample[0]))));
    }
}

static void metal_graph_log_dspark_embed(ds4_gpu_graph *g) {
    const char *env = getenv("DS4_DSPARK_EMBED_LOG");
    if (!g || !g->dspark_h || !g->dspark_input_ids || !env || !env[0] || atoi(env) == 0) return;
    if (ds4_gpu_flush_commands_blocking() == 0) return;
    int32_t ids[5] = {0};
    float sample[16] = {0};
    if (ds4_gpu_tensor_read(g->dspark_input_ids,
                            0,
                            ids,
                            sizeof(ids)) != 0 &&
        ds4_gpu_tensor_read(g->dspark_h,
                            0,
                            sample,
                            sizeof(sample)) != 0) {
        double ss = 0.0;
        for (size_t i = 0; i < sizeof(sample) / sizeof(sample[0]); i++) {
            ss += (double)sample[i] * (double)sample[i];
        }
        fprintf(stderr,
                "ds4: DSpark draft seed ids=[%d,%d,%d,%d,%d] embed_sample_rms=%.6g\n",
                ids[0], ids[1], ids[2], ids[3], ids[4],
                sqrt(ss / (double)(sizeof(sample) / sizeof(sample[0]))));
    }
}

static void metal_graph_log_dspark_main(ds4_gpu_graph *g) {
    const char *env = getenv("DS4_DSPARK_MAIN_LOG");
    if (!g || !g->dspark_main_x || !env || !env[0] || atoi(env) == 0) return;
    if (ds4_gpu_flush_commands_blocking() == 0) return;
    float sample[16] = {0};
    if (ds4_gpu_tensor_read(g->dspark_main_x,
                            0,
                            sample,
                            sizeof(sample)) != 0) {
        double ss = 0.0;
        for (size_t i = 0; i < sizeof(sample) / sizeof(sample[0]); i++) {
            ss += (double)sample[i] * (double)sample[i];
        }
        fprintf(stderr,
                "ds4: DSpark main_x sample_rms=%.6g capture_count=%" PRIu64 "\n",
                sqrt(ss / (double)(sizeof(sample) / sizeof(sample[0]))),
                g->dspark_capture_count);
    }
}

static void metal_graph_log_dspark_capture(ds4_gpu_graph *g) {
    const char *env = getenv("DS4_DSPARK_CAPTURE_LOG");
    if (!g || !g->dspark_target_hidden || !env || !env[0] || atoi(env) == 0) return;
    float *buf = xmalloc(3ull * DS4_N_EMBD * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(g->dspark_target_hidden,
                            0,
                            buf,
                            3ull * DS4_N_EMBD * sizeof(buf[0])) != 0) {
        fprintf(stderr, "ds4: DSpark target capture count=%" PRIu64, g->dspark_capture_count);
        for (uint32_t slot = 0; slot < 3u; slot++) {
            double ss = 0.0;
            const float *row = buf + (uint64_t)slot * DS4_N_EMBD;
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) ss += (double)row[i] * row[i];
            fprintf(stderr,
                    " layer%u_rms=%.6g",
                    (unsigned)(DS4_N_LAYER - 3u + slot),
                    sqrt(ss / (double)DS4_N_EMBD));
        }
        fprintf(stderr, "\n");
    }
    free(buf);
}

/* DSpark verifier. */
struct ds4_verify_layer_hc_audit {
		    uint32_t n_layers;
		    uint32_t n_tokens;
		    uint64_t hc_dim;
		    float *layer_attn_norm[DS4_MAX_LAYER];
		    float *layer_q[DS4_MAX_LAYER];
		    float *layer_kv[DS4_MAX_LAYER];
	    float *layer_heads_raw[DS4_MAX_LAYER];
	    float *layer_heads[DS4_MAX_LAYER];
	    float *layer_attn_out[DS4_MAX_LAYER];
	    float *layer_attn_hc[DS4_MAX_LAYER];
	    float *layer_hc[DS4_MAX_LAYER];
	    float *layer_ffn_pre[DS4_MAX_LAYER];
	    float *layer_ffn_norm[DS4_MAX_LAYER];
    float *layer_routed_out[DS4_MAX_LAYER];
    float *layer_shared_out[DS4_MAX_LAYER];
};

static void ds4_verify_layer_hc_audit_free(ds4_verify_layer_hc_audit *a) {
		    if (!a) return;
		    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
		        free(a->layer_attn_norm[il]);
		        free(a->layer_q[il]);
	        free(a->layer_kv[il]);
	        free(a->layer_heads_raw[il]);
	        free(a->layer_heads[il]);
	        free(a->layer_attn_out[il]);
	        free(a->layer_attn_hc[il]);
	        free(a->layer_hc[il]);
	        free(a->layer_ffn_pre[il]);
	        free(a->layer_ffn_norm[il]);
        free(a->layer_routed_out[il]);
        free(a->layer_shared_out[il]);
    }
    memset(a, 0, sizeof(*a));
}

static bool metal_graph_copy_tensor_to_batch_row(
        ds4_gpu_tensor *dst_batch,
        uint32_t        row,
        uint64_t        row_dim,
        ds4_gpu_tensor *src) {
    if (!dst_batch || !src) return false;
    ds4_gpu_tensor *dst = metal_graph_tensor_row_view(dst_batch, row, row_dim);
    const bool ok = dst &&
        ds4_gpu_tensor_copy(dst, 0, src, 0, row_dim * sizeof(float)) != 0;
	    ds4_gpu_tensor_free(dst);
	    return ok;
	}

static bool metal_graph_copy_attention_debug_tensors_to_batch_row(
        ds4_gpu_graph *g,
        uint32_t       row) {
	    if (!g || row >= g->prefill_cap) return false;
	    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
	    return metal_graph_copy_tensor_to_batch_row(g->batch_attn_norm,
	                                                row,
	                                                DS4_N_EMBD,
	                                                g->attn_norm) &&
	           metal_graph_copy_tensor_to_batch_row(g->batch_q,
	                                                row,
	                                                q_dim,
                                                g->q) &&
           metal_graph_copy_tensor_to_batch_row(g->batch_kv,
                                                row,
                                                DS4_N_HEAD_DIM,
                                                g->kv) &&
           metal_graph_copy_tensor_to_batch_row(g->batch_heads,
                                                row,
                                                q_dim,
                                                g->heads) &&
           metal_graph_copy_tensor_to_batch_row(g->batch_attn_out,
                                                row,
                                                DS4_N_EMBD,
                                                g->attn_out);
}

static bool ds4_verify_layer_hc_audit_capture_tensor_slot_dim(
        ds4_verify_layer_hc_audit *a,
        float                    **slot,
        ds4_gpu_tensor            *src,
        uint32_t                   il,
        uint32_t                   n_tokens,
        uint64_t                   row_dim) {
    if (!a || !slot || !src || il >= DS4_MAX_LAYER || n_tokens == 0) return false;
    const uint64_t bytes = (uint64_t)n_tokens * row_dim * sizeof(float);
    if (!slot[il]) slot[il] = xmalloc((size_t)bytes);
    a->n_layers = a->n_layers < il + 1u ? il + 1u : a->n_layers;
    a->n_tokens = n_tokens;
    if (row_dim == (uint64_t)DS4_N_HC * DS4_N_EMBD) a->hc_dim = row_dim;
    return ds4_gpu_tensor_read(src, 0, slot[il], bytes) != 0;
}

static bool ds4_verify_layer_hc_audit_capture_tensor_slot(
        ds4_verify_layer_hc_audit *a,
        float                    **slot,
        ds4_gpu_tensor            *src,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    return ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             slot,
                                                             src,
                                                             il,
                                                             n_tokens,
                                                             hc_dim);
}

static bool ds4_verify_layer_hc_audit_capture_rows_slot_dim(
        ds4_verify_layer_hc_audit *a,
        float                    **slot,
        ds4_gpu_tensor           **rows,
        uint32_t                   il,
        uint32_t                   n_tokens,
        uint64_t                   row_dim) {
    if (!a || !slot || !rows || il >= DS4_MAX_LAYER || n_tokens == 0) return false;
    const uint64_t row_bytes = row_dim * sizeof(float);
    const uint64_t bytes = (uint64_t)n_tokens * row_bytes;
    if (!slot[il]) slot[il] = xmalloc((size_t)bytes);
    a->n_layers = a->n_layers < il + 1u ? il + 1u : a->n_layers;
    a->n_tokens = n_tokens;
    if (row_dim == (uint64_t)DS4_N_HC * DS4_N_EMBD) a->hc_dim = row_dim;
    for (uint32_t t = 0; t < n_tokens; t++) {
        if (!rows[t]) return false;
        if (ds4_gpu_tensor_read(rows[t],
                                0,
                                slot[il] + (uint64_t)t * row_dim,
                                row_bytes) == 0) {
            return false;
        }
    }
    return true;
}

static bool ds4_verify_layer_hc_audit_capture_rows_slot(
        ds4_verify_layer_hc_audit *a,
        float                    **slot,
        ds4_gpu_tensor           **rows,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    return ds4_verify_layer_hc_audit_capture_rows_slot_dim(a,
                                                           slot,
                                                           rows,
                                                           il,
                                                           n_tokens,
                                                           hc_dim);
}

static bool ds4_verify_layer_hc_audit_capture_tensor(
        ds4_verify_layer_hc_audit *a,
        ds4_gpu_tensor            *src,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    return ds4_verify_layer_hc_audit_capture_tensor_slot(a,
                                                         a->layer_hc,
                                                         src,
	                                                         il,
	                                                         n_tokens);
	}

static bool ds4_verify_layer_hc_audit_capture_attention_tensors(
        ds4_verify_layer_hc_audit *a,
        ds4_gpu_graph             *g,
        uint32_t                   il,
        uint32_t                   n_tokens) {
	    if (!a || !g || n_tokens == 0) return false;
	    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
	    return ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
	                                                             a->layer_attn_norm,
	                                                             g->batch_attn_norm,
	                                                             il,
	                                                             n_tokens,
	                                                             DS4_N_EMBD) &&
	           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
	                                                             a->layer_q,
	                                                             g->batch_q,
                                                             il,
                                                             n_tokens,
                                                             q_dim) &&
           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_kv,
                                                             g->batch_kv,
                                                             il,
	                                                             n_tokens,
	                                                             DS4_N_HEAD_DIM) &&
	           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
	                                                             a->layer_heads_raw,
	                                                             g->batch_heads_raw,
	                                                             il,
	                                                             n_tokens,
	                                                             q_dim) &&
	           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
	                                                             a->layer_heads,
	                                                             g->batch_heads,
                                                             il,
                                                             n_tokens,
                                                             q_dim) &&
           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_attn_out,
                                                             g->batch_attn_out,
                                                             il,
                                                             n_tokens,
                                                             DS4_N_EMBD);
}

static bool ds4_verify_layer_hc_audit_capture_attn_tensor(
        ds4_verify_layer_hc_audit *a,
        ds4_gpu_tensor            *src,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    return ds4_verify_layer_hc_audit_capture_tensor_slot(a,
                                                         a->layer_attn_hc,
                                                         src,
                                                         il,
                                                         n_tokens);
}

static bool ds4_verify_layer_hc_audit_capture_ffn_tensors(
        ds4_verify_layer_hc_audit *a,
        ds4_gpu_graph             *g,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    if (!a || !g || n_tokens == 0) return false;
    const uint64_t embd = DS4_N_EMBD;
    return ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_ffn_pre,
                                                             g->batch_ffn_cur,
                                                             il,
                                                             n_tokens,
                                                             embd) &&
           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_ffn_norm,
                                                             g->batch_ffn_norm,
                                                             il,
                                                             n_tokens,
                                                             embd) &&
           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_routed_out,
                                                             g->batch_routed_out,
                                                             il,
                                                             n_tokens,
                                                             embd) &&
           ds4_verify_layer_hc_audit_capture_tensor_slot_dim(a,
                                                             a->layer_shared_out,
                                                             g->batch_shared_out,
                                                             il,
                                                             n_tokens,
                                                             embd);
}

static bool ds4_verify_layer_hc_audit_capture_rows(
        ds4_verify_layer_hc_audit *a,
        ds4_gpu_tensor           **rows,
        uint32_t                   il,
        uint32_t                   n_tokens) {
    return ds4_verify_layer_hc_audit_capture_rows_slot(a,
                                                       a->layer_hc,
                                                       rows,
                                                       il,
                                                       n_tokens);
}

static float ds4_verify_layer_hc_audit_eps(void) {
    const char *env = getenv("DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS");
    if (!env || !env[0]) return 1.0e-5f;
    char *end = NULL;
    const float v = strtof(env, &end);
    if (end == env || !isfinite(v) || v < 0.0f) return 1.0e-5f;
    return v;
}

typedef struct {
    bool     has_first;
    uint32_t first_layer;
    uint32_t first_row;
    uint64_t first_index;
    float    first_delta;
    float    first_hybrid;
    float    first_exact;
    float    first_layer_max;
    float    first_layer_rms;
    bool     has_worst;
    uint32_t worst_layer;
    uint32_t worst_row;
    uint64_t worst_index;
    float    worst_delta;
    float    worst_hybrid;
    float    worst_exact;
    float    worst_layer_rms;
} ds4_verify_layer_hc_audit_stage_delta;

static float ds4_verify_audit_abs_delta(float a, float b) {
    const float d = fabsf(a - b);
    return isfinite(d) ? d : FLT_MAX;
}

static ds4_verify_layer_hc_audit_stage_delta ds4_verify_layer_hc_audit_compare_slot(
        const ds4_verify_layer_hc_audit *hybrid,
        const ds4_verify_layer_hc_audit *exact,
        float * const                   *hybrid_slot,
        float * const                   *exact_slot,
        uint64_t                         row_dim,
        const char                      *stage,
        const char                      *label) {
    ds4_verify_layer_hc_audit_stage_delta result;
    memset(&result, 0, sizeof(result));
    if (!hybrid || !exact) return result;
    const uint32_t n_layers = hybrid->n_layers < exact->n_layers ?
        hybrid->n_layers : exact->n_layers;
    const uint32_t n_tokens = hybrid->n_tokens < exact->n_tokens ?
        hybrid->n_tokens : exact->n_tokens;
    int first = -1;
    float first_max = 0.0f, first_rms = 0.0f;
    float worst_max = 0.0f, worst_rms = 0.0f;
    int worst = -1;
    const float eps = ds4_verify_layer_hc_audit_eps();
    for (uint32_t il = 0; il < n_layers; il++) {
        if (!hybrid_slot[il] || !exact_slot[il]) continue;
        const uint64_t n = (uint64_t)n_tokens * row_dim;
        float dmax = 0.0f;
        double ss = 0.0;
        uint64_t max_index = 0;
        uint64_t first_index = UINT64_MAX;
        for (uint64_t i = 0; i < n; i++) {
            const float d = ds4_verify_audit_abs_delta(hybrid_slot[il][i],
                                                       exact_slot[il][i]);
            ss += (double)d * (double)d;
            if (d > dmax) {
                dmax = d;
                max_index = i;
            }
            if (first_index == UINT64_MAX && d > eps) {
                first_index = i;
            }
        }
        const float drms = n ? (float)sqrt(ss / (double)n) : 0.0f;
        if (dmax > worst_max) {
            worst_max = dmax;
            worst_rms = drms;
            worst = (int)il;
            result.has_worst = true;
            result.worst_layer = il;
            result.worst_row = row_dim ? (uint32_t)(max_index / row_dim) : 0u;
            result.worst_index = row_dim ? max_index % row_dim : max_index;
            result.worst_delta = dmax;
            result.worst_hybrid = hybrid_slot[il][max_index];
            result.worst_exact = exact_slot[il][max_index];
            result.worst_layer_rms = drms;
        }
        if (first < 0 && dmax > eps) {
            first = (int)il;
            first_max = dmax;
            first_rms = drms;
            if (first_index == UINT64_MAX) first_index = max_index;
            result.has_first = true;
            result.first_layer = il;
            result.first_row = row_dim ? (uint32_t)(first_index / row_dim) : 0u;
            result.first_index = row_dim ? first_index % row_dim : first_index;
            result.first_delta = ds4_verify_audit_abs_delta(hybrid_slot[il][first_index],
                                                            exact_slot[il][first_index]);
            result.first_hybrid = hybrid_slot[il][first_index];
            result.first_exact = exact_slot[il][first_index];
            result.first_layer_max = dmax;
            result.first_layer_rms = drms;
        }
        if (env_flag_enabled("DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_VERBOSE") &&
            dmax > eps) {
            fprintf(stderr,
                    "ds4: dspark hybrid layer %s audit %s layer=%u max=%.6g rms=%.6g rows=%u worst_row=%u worst_idx=%llu worst_delta=%.6g\n",
                    stage ? stage : "hc",
                    label ? label : "hybrid-vs-exact",
                    il,
                    dmax,
                    drms,
                    n_tokens,
                    row_dim ? (uint32_t)(max_index / row_dim) : 0u,
                    (unsigned long long)(row_dim ? max_index % row_dim : max_index),
                    dmax);
        }
    }
    fprintf(stderr,
            "ds4: dspark hybrid layer %s audit %s eps=%.6g first=%d row=%u idx=%llu delta=%.6g h=%.6g e=%.6g max=%.6g rms=%.6g worst=%d row=%u idx=%llu worst_delta=%.6g h=%.6g e=%.6g worst_rms=%.6g layers=%u rows=%u\n",
            stage ? stage : "hc",
            label ? label : "hybrid-vs-exact",
            eps,
            first,
            result.has_first ? result.first_row : 0u,
            (unsigned long long)(result.has_first ? result.first_index : 0u),
            result.has_first ? result.first_delta : 0.0f,
            result.has_first ? result.first_hybrid : 0.0f,
            result.has_first ? result.first_exact : 0.0f,
            first_max,
            first_rms,
            worst,
            result.has_worst ? result.worst_row : 0u,
            (unsigned long long)(result.has_worst ? result.worst_index : 0u),
            worst_max,
            result.has_worst ? result.worst_hybrid : 0.0f,
            result.has_worst ? result.worst_exact : 0.0f,
            worst_rms,
            n_layers,
            n_tokens);
    return result;
}

static void ds4_verify_layer_hc_audit_compare(
        const ds4_verify_layer_hc_audit *hybrid,
        const ds4_verify_layer_hc_audit *exact,
        const char                      *label) {
    const uint64_t hc_dim = hybrid->hc_dim && hybrid->hc_dim == exact->hc_dim ?
        hybrid->hc_dim : (uint64_t)DS4_N_HC * DS4_N_EMBD;
	    const char *first_stage = NULL;
	    int first_stage_order = INT_MAX;
	    ds4_verify_layer_hc_audit_stage_delta first_delta;
	    memset(&first_delta, 0, sizeof(first_delta));
	    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
#define DS4_VERIFY_AUDIT_STAGE(stage_order, stage_name, hybrid_slot, exact_slot, dim) do { \
	        ds4_verify_layer_hc_audit_stage_delta _delta = \
	            ds4_verify_layer_hc_audit_compare_slot(hybrid, \
                                                   exact, \
                                                   (hybrid_slot), \
                                                   (exact_slot), \
                                                   (dim), \
                                                   (stage_name), \
                                                   label); \
        if (_delta.has_first && \
            (!first_stage || \
             _delta.first_layer < first_delta.first_layer || \
             (_delta.first_layer == first_delta.first_layer && \
              (stage_order) < first_stage_order))) { \
            first_stage = (stage_name); \
            first_stage_order = (stage_order); \
	            first_delta = _delta; \
	        } \
	    } while (0)
		    if (metal_graph_dspark_attn_stage_audit_enabled()) {
		        DS4_VERIFY_AUDIT_STAGE(0,
		                               "attn-norm",
		                               hybrid->layer_attn_norm,
		                               exact->layer_attn_norm,
		                               DS4_N_EMBD);
		        DS4_VERIFY_AUDIT_STAGE(1,
		                               "q-rope",
		                               hybrid->layer_q,
		                               exact->layer_q,
		                               q_dim);
		        DS4_VERIFY_AUDIT_STAGE(2,
		                               "kv-cache-row",
		                               hybrid->layer_kv,
		                               exact->layer_kv,
		                               DS4_N_HEAD_DIM);
		        DS4_VERIFY_AUDIT_STAGE(3,
		                               "attn-heads-raw",
		                               hybrid->layer_heads_raw,
		                               exact->layer_heads_raw,
		                               q_dim);
		        DS4_VERIFY_AUDIT_STAGE(4,
		                               "attn-heads",
		                               hybrid->layer_heads,
		                               exact->layer_heads,
		                               q_dim);
		        DS4_VERIFY_AUDIT_STAGE(5,
		                               "attn-out",
		                               hybrid->layer_attn_out,
		                               exact->layer_attn_out,
		                               DS4_N_EMBD);
		    }
		    DS4_VERIFY_AUDIT_STAGE(6,
		                           "attn-HC",
		                           hybrid->layer_attn_hc,
		                           exact->layer_attn_hc,
		                           hc_dim);
		    if (env_flag_enabled("DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT")) {
		        DS4_VERIFY_AUDIT_STAGE(7,
		                               "ffn-pre",
		                               hybrid->layer_ffn_pre,
		                               exact->layer_ffn_pre,
		                               DS4_N_EMBD);
		        DS4_VERIFY_AUDIT_STAGE(8,
		                               "ffn-norm",
		                               hybrid->layer_ffn_norm,
		                               exact->layer_ffn_norm,
		                               DS4_N_EMBD);
		        DS4_VERIFY_AUDIT_STAGE(9,
		                               "routed-out",
		                               hybrid->layer_routed_out,
		                               exact->layer_routed_out,
		                               DS4_N_EMBD);
		        DS4_VERIFY_AUDIT_STAGE(10,
		                               "shared-out",
		                               hybrid->layer_shared_out,
		                               exact->layer_shared_out,
		                               DS4_N_EMBD);
		    }
		    DS4_VERIFY_AUDIT_STAGE(11,
		                           "post-HC",
		                           hybrid->layer_hc,
	                           exact->layer_hc,
                           hc_dim);
#undef DS4_VERIFY_AUDIT_STAGE
    if (first_stage) {
        fprintf(stderr,
                "ds4: dspark hybrid first divergence audit %s stage=%s layer=%u row=%u idx=%llu delta=%.6g h=%.6g e=%.6g layer_max=%.6g layer_rms=%.6g\n",
                label ? label : "hybrid-vs-exact",
                first_stage,
                first_delta.first_layer,
                first_delta.first_row,
                (unsigned long long)first_delta.first_index,
                first_delta.first_delta,
                first_delta.first_hybrid,
                first_delta.first_exact,
                first_delta.first_layer_max,
                first_delta.first_layer_rms);
    } else {
        fprintf(stderr,
                "ds4: dspark hybrid first divergence audit %s stage=clean\n",
                label ? label : "hybrid-vs-exact");
    }
}

/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
static bool metal_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        float                 *logits0,
        float                 *logits1,
        bool                   capture_prefix1,
        bool                   batch_output_head) {
    if (!g || !top0 || !logits1 || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *cur0 = metal_graph_tensor_row_view(g->batch_cur_hc, 0, hc_dim);
    ds4_gpu_tensor *cur1 = metal_graph_tensor_row_view(g->batch_cur_hc, 1, hc_dim);
    ds4_gpu_tensor *next0 = metal_graph_tensor_row_view(g->batch_next_hc, 0, hc_dim);
    ds4_gpu_tensor *next1 = metal_graph_tensor_row_view(g->batch_next_hc, 1, hc_dim);
    bool ok = cur0 && cur1 && next0 && next1;

    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token0,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token1,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = capture_prefix1;
    const uint32_t n_layers = metal_graph_spec_verify_layer_limit();
    const uint32_t split_after_layers = metal_graph_spec_verify_split_layers();
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < n_layers; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;

        g->cur_hc = cur0;
        g->after_ffn_hc = next0;
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        if (capture_prefix1) {
            ok = metal_graph_capture_prefix1_attn_state(g, il) &&
                 metal_graph_capture_prefix1_index_state(g, il);
            if (!ok) break;
        }

        g->cur_hc = cur1;
        g->after_ffn_hc = next1;
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0; cur0 = next0; next0 = tmp;
        tmp = cur1; cur1 = next1; next1 = tmp;
        if (ok && split_after_layers != 0 &&
            il + 1u == split_after_layers &&
            il + 1u < n_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok && batch_output_head && g->spec_logits) {
        ds4_gpu_tensor *saved_batch_cur = g->batch_cur_hc;
        ds4_gpu_tensor *final_batch_hc = metal_graph_spec_final_batch_hc(g);
        g->batch_cur_hc = final_batch_hc;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head_batch(g,
                                                          model,
                                                          weights,
                                                          2,
                                                          weights->output->dim[1]);
        if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->spec_logits,
                                                   DS4_N_VOCAB,
                                                   1,
                                                   1) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->batch_cur_hc = saved_batch_cur;
        if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0)) != 0;
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits1[0]),
                                       logits1,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
        }
    } else if (ok) {
        g->cur_hc = cur0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->logits,
                                                   DS4_N_VOCAB,
                                                   1,
                                                   1) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0)) != 0;
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
        if (ok) {
            g->cur_hc = cur1;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            g->cur_hc = saved_cur;
            if (ok) {
                ok = ds4_gpu_tensor_read(g->logits,
                                           0,
                                           logits1,
                                           (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
            }
        }
    }
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;
    g->spec_capture_prefix1 = saved_capture;

    ds4_gpu_tensor_free(next1);
    ds4_gpu_tensor_free(next0);
    ds4_gpu_tensor_free(cur1);
    ds4_gpu_tensor_free(cur0);
    return ok;
}

/* Experimental exact decode-order verifier for N<=5.
 *
 * Unlike the approximate layer-major suffix verifier, this uses the same
 * single-token decode layer kernels and cache mutation order as greedy decode.
 * The work is still exact per token, but it is grouped layer-major in one
 * command stream and uses one batched output head.  Full-accept blocks can
 * commit directly; partial accepts commit from prefix-N frontier snapshots
 * when the caller requests them. */
static float ds4_env_positive_float(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return 0.0f;
    char *end = NULL;
    const float v = strtof(env, &end);
    return (end != env && v > 0.0f) ? v : 0.0f;
}

static bool metal_graph_verify_decodeN_read_all_logits(uint32_t n_tokens) {
    return n_tokens == 2u ||
           env_flag_enabled("DS4_DSPARK_HYBRID_READ_ALL_LOGITS") ||
           ds4_env_positive_float("DS4_DSPARK_HYBRID_MARGIN_GUARD") > 0.0f;
}

typedef struct {
    uint64_t embed_dispatches;
    uint64_t attention_dispatches;
    uint64_t kv_store_dispatches;
    uint64_t compressor_dispatches;
    uint64_t indexer_dispatches;
    uint64_t attn_head_dispatches;
    uint64_t output_hc_dispatches;
    uint64_t ffn_pre_dispatches;
    uint64_t router_dispatches;
    uint64_t routed_moe_dispatches;
    uint64_t ordered_sum_dispatches;
    uint64_t shared_dispatches;
    uint64_t post_hc_dispatches;
    uint64_t target_hidden_dispatches;
    uint64_t output_head_dispatches;
    uint64_t topk_dispatches;
    uint64_t readbacks;
    uint64_t total_dispatches;
} ds4_dspark_verify_dispatch_stats;

typedef struct {
    uint64_t layers;
    uint64_t total_slots;
    uint64_t unique_slots;
    uint64_t duplicate_slots;
    uint64_t row_pairs;
    uint64_t row_pair_overlap_sum;
    uint64_t layers_reuse125;
    uint64_t layers_reuse150;
    uint32_t max_multiplicity;
    uint32_t min_unique;
    uint32_t max_unique;
} ds4_dspark_routed_overlap_stats;

typedef struct {
    uint64_t total_slots;
    uint64_t row_pairs;
    uint64_t row_pair_overlap_sum;
    uint32_t unique;
    uint32_t duplicate_slots;
    uint32_t max_multiplicity;
    int32_t  top_expert[4];
    uint32_t top_count[4];
} ds4_dspark_routed_overlap_layer_stats;

static uint64_t g_dspark_routed_overlap_blocks;

static bool ds4_dspark_verify_dispatch_profile_enabled(void) {
    return env_flag_enabled("DS4_DSPARK_VERIFY_DISPATCH_PROFILE") ||
           env_flag_enabled("DS4_DSPARK_DISPATCH_PROFILE");
}

static bool ds4_dspark_routed_overlap_profile_enabled(void) {
    return env_flag_enabled("DS4_DSPARK_ROUTE_OVERLAP_LOG") ||
           env_flag_enabled("DS4_DSPARK_ROUTED_OVERLAP_PROFILE") ||
           env_flag_enabled("DS4_DSPARK_ROUTED_EXPERT_OVERLAP") ||
           env_flag_enabled("DS4_DSPARK_MOE_OVERLAP_PROFILE");
}

static void ds4_dspark_routed_overlap_stats_add_layer(
        ds4_dspark_routed_overlap_stats *s,
        ds4_dspark_routed_overlap_layer_stats *layer,
        const int32_t                   *selected,
        uint32_t                         n_tokens,
        uint32_t                         active_expert_used,
        uint32_t                         n_expert) {
    if (!s || !selected || n_tokens == 0 || active_expert_used == 0 ||
        n_expert == 0) {
        return;
    }

    ds4_dspark_routed_overlap_layer_stats local;
    memset(&local, 0, sizeof(local));
    for (uint32_t i = 0; i < 4u; i++) local.top_expert[i] = -1;

    uint16_t *counts = xcalloc(n_expert, sizeof(counts[0]));
    uint8_t *row_seen = xcalloc((size_t)n_tokens * n_expert, sizeof(row_seen[0]));
    uint32_t unique = 0;
    const uint64_t total = (uint64_t)n_tokens * active_expert_used;

    for (uint32_t row = 0; row < n_tokens; row++) {
        uint8_t *row_bits = row_seen + (size_t)row * n_expert;
        for (uint32_t slot = 0; slot < active_expert_used; slot++) {
            const int32_t expert =
                selected[(size_t)row * active_expert_used + slot];
            if (expert < 0 || (uint32_t)expert >= n_expert) continue;
            if (counts[expert] == 0) {
                unique++;
            }
            if (counts[expert] != UINT16_MAX) counts[expert]++;
            row_bits[expert] = 1;
        }
    }

    uint32_t max_multiplicity = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        if (count == 0) continue;
        if (count > max_multiplicity) max_multiplicity = count;
        for (uint32_t j = 0; j < 4u; j++) {
            if (count > local.top_count[j]) {
                for (uint32_t k = 3u; k > j; k--) {
                    local.top_count[k] = local.top_count[k - 1u];
                    local.top_expert[k] = local.top_expert[k - 1u];
                }
                local.top_count[j] = count;
                local.top_expert[j] = (int32_t)e;
                break;
            }
        }
    }

    uint64_t row_pairs = 0;
    uint64_t overlap_sum = 0;
    for (uint32_t a = 0; a < n_tokens; a++) {
        const uint8_t *a_bits = row_seen + (size_t)a * n_expert;
        for (uint32_t b = a + 1u; b < n_tokens; b++) {
            const uint8_t *b_bits = row_seen + (size_t)b * n_expert;
            uint32_t overlap = 0;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (a_bits[e] && b_bits[e]) overlap++;
            }
            row_pairs++;
            overlap_sum += overlap;
        }
    }

    s->layers++;
    s->total_slots += total;
    s->unique_slots += unique;
    s->duplicate_slots += total > unique ? total - unique : 0;
    s->row_pairs += row_pairs;
    s->row_pair_overlap_sum += overlap_sum;
    if (max_multiplicity > s->max_multiplicity) {
        s->max_multiplicity = max_multiplicity;
    }
    if (s->min_unique == 0 || unique < s->min_unique) s->min_unique = unique;
    if (unique > s->max_unique) s->max_unique = unique;
    const double reuse = unique ? (double)total / (double)unique : 0.0;
    if (reuse >= 1.25) s->layers_reuse125++;
    if (reuse >= 1.50) s->layers_reuse150++;

    local.total_slots = total;
    local.unique = unique;
    local.duplicate_slots = total > unique ? (uint32_t)(total - unique) : 0u;
    local.max_multiplicity = max_multiplicity;
    local.row_pairs = row_pairs;
    local.row_pair_overlap_sum = overlap_sum;
    if (layer) *layer = local;

    free(row_seen);
    free(counts);
}

static bool ds4_dspark_routed_overlap_stats_capture(
        ds4_gpu_graph                    *g,
        ds4_dspark_routed_overlap_stats  *stats,
        const char                       *path,
        uint32_t                          il,
        uint32_t                          n_tokens,
        bool                             *commands_open) {
    if (!g || !stats || n_tokens == 0) return false;
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    const uint32_t n_expert = DS4_N_EXPERT;
    if (active_expert_used == 0 || n_expert == 0) return false;

    if (commands_open && *commands_open) {
        if (ds4_gpu_end_commands() == 0) return false;
        *commands_open = false;
    }

    const size_t n_selected = (size_t)n_tokens * active_expert_used;
    int32_t *selected = xmalloc(n_selected * sizeof(selected[0]));
    bool ok = ds4_gpu_tensor_read(g->batch_router_selected,
                                  0,
                                  selected,
                                  (uint64_t)n_selected * sizeof(selected[0])) != 0;
    if (ok) {
        ds4_dspark_routed_overlap_layer_stats layer_stats;
        memset(&layer_stats, 0, sizeof(layer_stats));
        ds4_dspark_routed_overlap_stats_add_layer(stats,
                                                  &layer_stats,
                                                  selected,
                                                  n_tokens,
                                                  active_expert_used,
                                                  n_expert);
        const double reuse = layer_stats.unique ?
            (double)layer_stats.total_slots / (double)layer_stats.unique : 0.0;
        const double pair_overlap = layer_stats.row_pairs ?
            (double)layer_stats.row_pair_overlap_sum / (double)layer_stats.row_pairs : 0.0;
        fprintf(stderr,
                "ds4: dspark route overlap path=%s block=%llu layer=%u active=%u "
                "slots=%llu unique=%u duplicate_slots=%u reuse=%.2fx "
                "max_multiplicity=%u pair_overlap=%.2f/%u top_experts:",
                path ? path : "unknown",
                (unsigned long long)(g_dspark_routed_overlap_blocks + 1u),
                il,
                n_tokens,
                (unsigned long long)layer_stats.total_slots,
                layer_stats.unique,
                layer_stats.duplicate_slots,
                reuse,
                layer_stats.max_multiplicity,
                pair_overlap,
                active_expert_used);
        for (uint32_t i = 0; i < 4u && layer_stats.top_expert[i] >= 0; i++) {
            fprintf(stderr,
                    " %d:%u",
                    layer_stats.top_expert[i],
                    layer_stats.top_count[i]);
        }
        fprintf(stderr, "\n");
    }
    free(selected);

    if (ok && commands_open) {
        ok = ds4_gpu_begin_commands() != 0;
        *commands_open = ok;
    }
    return ok;
}

static void ds4_dspark_routed_overlap_stats_print(
        const char                         *path,
        uint32_t                            n_tokens,
        const ds4_dspark_routed_overlap_stats *stats) {
    if (!stats || stats->layers == 0 || stats->total_slots == 0) return;

    static uint64_t s_total_slots = 0;
    static uint64_t s_unique_slots = 0;
    static uint64_t s_row_pairs = 0;
    static uint64_t s_row_pair_overlap_sum = 0;
    g_dspark_routed_overlap_blocks++;
    s_total_slots += stats->total_slots;
    s_unique_slots += stats->unique_slots;
    s_row_pairs += stats->row_pairs;
    s_row_pair_overlap_sum += stats->row_pair_overlap_sum;

    const double reuse =
        stats->unique_slots ? (double)stats->total_slots / (double)stats->unique_slots : 0.0;
    const double dup_pct =
        stats->total_slots ? 100.0 * (double)stats->duplicate_slots / (double)stats->total_slots : 0.0;
    const double pair_overlap =
        stats->row_pairs ?
            (double)stats->row_pair_overlap_sum / (double)stats->row_pairs : 0.0;
    const double avg_reuse =
        s_unique_slots ? (double)s_total_slots / (double)s_unique_slots : 0.0;
    const double avg_pair_overlap =
        s_row_pairs ?
            (double)s_row_pair_overlap_sum / (double)s_row_pairs : 0.0;

    fprintf(stderr,
            "ds4: dspark routed overlap path=%s block=%llu n=%u layers=%llu "
            "slots=%llu unique=%llu reuse=%.2fx dup=%.1f%% pair_overlap=%.2f/%u "
            "max_multiplicity=%u unique_min=%u unique_max=%u reuse>=1.25=%llu "
            "reuse>=1.50=%llu avg_reuse=%.2fx avg_pair_overlap=%.2f/%u\n",
            path ? path : "unknown",
            (unsigned long long)g_dspark_routed_overlap_blocks,
            n_tokens,
            (unsigned long long)stats->layers,
            (unsigned long long)stats->total_slots,
            (unsigned long long)stats->unique_slots,
            reuse,
            dup_pct,
            pair_overlap,
            (unsigned)DS4_N_EXPERT_ACTIVE_USED,
            stats->max_multiplicity,
            stats->min_unique,
            stats->max_unique,
            (unsigned long long)stats->layers_reuse125,
            (unsigned long long)stats->layers_reuse150,
            avg_reuse,
            avg_pair_overlap,
            (unsigned)DS4_N_EXPERT_ACTIVE_USED);
}

static uint64_t ds4_dspark_verify_dispatch_stats_total(
        const ds4_dspark_verify_dispatch_stats *s) {
    if (!s) return 0;
    return s->embed_dispatches +
           s->attention_dispatches +
           s->kv_store_dispatches +
           s->compressor_dispatches +
           s->indexer_dispatches +
           s->attn_head_dispatches +
           s->output_hc_dispatches +
           s->ffn_pre_dispatches +
           s->router_dispatches +
           s->routed_moe_dispatches +
           s->ordered_sum_dispatches +
           s->shared_dispatches +
           s->post_hc_dispatches +
           s->target_hidden_dispatches +
           s->output_head_dispatches +
           s->topk_dispatches +
           s->readbacks;
}

static void ds4_dspark_verify_dispatch_stats_add(
        ds4_dspark_verify_dispatch_stats       *dst,
        const ds4_dspark_verify_dispatch_stats *src) {
    if (!dst || !src) return;
    dst->embed_dispatches += src->embed_dispatches;
    dst->attention_dispatches += src->attention_dispatches;
    dst->kv_store_dispatches += src->kv_store_dispatches;
    dst->compressor_dispatches += src->compressor_dispatches;
    dst->indexer_dispatches += src->indexer_dispatches;
    dst->attn_head_dispatches += src->attn_head_dispatches;
    dst->output_hc_dispatches += src->output_hc_dispatches;
    dst->ffn_pre_dispatches += src->ffn_pre_dispatches;
    dst->router_dispatches += src->router_dispatches;
    dst->routed_moe_dispatches += src->routed_moe_dispatches;
    dst->ordered_sum_dispatches += src->ordered_sum_dispatches;
    dst->shared_dispatches += src->shared_dispatches;
    dst->post_hc_dispatches += src->post_hc_dispatches;
    dst->target_hidden_dispatches += src->target_hidden_dispatches;
    dst->output_head_dispatches += src->output_head_dispatches;
    dst->topk_dispatches += src->topk_dispatches;
    dst->readbacks += src->readbacks;
}

static void ds4_dspark_verify_dispatch_count_attention(
        ds4_dspark_verify_dispatch_stats *s,
        uint32_t                          il,
        uint32_t                          n_tokens,
        bool                              decode_order,
        bool                              row_output_hc) {
    if (!s || n_tokens == 0) return;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool microbatch_kv =
        decode_order &&
        !env_flag_enabled("DS4_DSPARK_VERIFY_MICROBATCH_DISABLE") &&
        !env_flag_enabled("DS4_DSPARK_VERIFY_NO_MICROBATCH");

    /* Helper-level estimates: most rows still encode multiple kernels inside
     * metal_graph_encode_layer_attention_batch(), but this separates the
     * mutation-heavy pieces we need to collapse next. */
    s->attention_dispatches += decode_order ? 1u : n_tokens;
    s->kv_store_dispatches += microbatch_kv ? 1u : n_tokens;
    if (ratio != 0) {
        s->compressor_dispatches += n_tokens;
    }
    if (ratio == 4u) {
        s->indexer_dispatches += n_tokens;
    }
    s->attn_head_dispatches += n_tokens;
    s->output_hc_dispatches += row_output_hc ? 1u : n_tokens;
}

static void ds4_dspark_verify_dispatch_count_exact_layer(
        ds4_dspark_verify_dispatch_stats *s,
        uint32_t                          il,
        uint32_t                          n_tokens) {
    if (!s || n_tokens == 0) return;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    s->attention_dispatches += n_tokens;
    s->kv_store_dispatches += n_tokens;
    if (ratio != 0) {
        s->compressor_dispatches += n_tokens;
    }
    if (ratio == 4u) {
        s->indexer_dispatches += n_tokens;
    }
    s->attn_head_dispatches += n_tokens;
    s->output_hc_dispatches += n_tokens;
    s->ffn_pre_dispatches += n_tokens;
    s->router_dispatches += n_tokens;
    s->routed_moe_dispatches += n_tokens;
    s->ordered_sum_dispatches += n_tokens;
    s->shared_dispatches += n_tokens;
    s->post_hc_dispatches += n_tokens;
}

static bool ds4_dspark_verify_direct_q2_ordered_sum_enabled(
        const ds4_layer_weights *layer) {
    return layer &&
           layer->ffn_down_exps &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           !env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE") &&
           (env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2") ||
            env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT")) &&
           !env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN");
}

static uint32_t ds4_dspark_target_hidden_dispatch_count(
        uint32_t il,
        uint32_t n_tokens) {
    if (DS4_N_LAYER < 3u || il + 3u < DS4_N_LAYER || n_tokens == 0) return 0;
    if (il - (DS4_N_LAYER - 3u) >= 3u) return 0;
    return n_tokens > 128u ? 128u : n_tokens;
}

static void ds4_dspark_verify_dispatch_stats_print(
        const char                            *path,
        uint32_t                               n_tokens,
        uint32_t                               n_layers,
        const ds4_dspark_verify_dispatch_stats *stats) {
    if (!stats) return;

    static ds4_dspark_verify_dispatch_stats s_total;
    static uint64_t s_blocks = 0;
    ds4_dspark_verify_dispatch_stats current = *stats;
    current.total_dispatches =
        ds4_dspark_verify_dispatch_stats_total(&current);
    ds4_dspark_verify_dispatch_stats_add(&s_total, &current);
    s_total.total_dispatches =
        ds4_dspark_verify_dispatch_stats_total(&s_total);
    s_blocks++;

    const uint64_t attn_pre =
        current.attention_dispatches;
    const uint64_t mutate =
        current.kv_store_dispatches +
        current.compressor_dispatches +
        current.indexer_dispatches;
    const uint64_t routed =
        current.routed_moe_dispatches +
        current.ordered_sum_dispatches;
    const uint64_t tail =
        current.post_hc_dispatches +
        current.target_hidden_dispatches +
        current.output_head_dispatches +
        current.topk_dispatches +
        current.readbacks;

    fprintf(stderr,
            "ds4: dspark verify dispatch path=%s block=%llu n=%u layers=%u "
            "total=%llu attn_pre=%llu mutate=%llu heads=%llu attn_out=%llu "
            "ffn_pre=%llu router=%llu routed=%llu shared=%llu tail=%llu "
            "avg_total=%.1f\n",
            path ? path : "unknown",
            (unsigned long long)s_blocks,
            n_tokens,
            n_layers,
            (unsigned long long)current.total_dispatches,
            (unsigned long long)attn_pre,
            (unsigned long long)mutate,
            (unsigned long long)current.attn_head_dispatches,
            (unsigned long long)current.output_hc_dispatches,
            (unsigned long long)current.ffn_pre_dispatches,
            (unsigned long long)current.router_dispatches,
            (unsigned long long)routed,
            (unsigned long long)current.shared_dispatches,
            (unsigned long long)tail,
            s_blocks ? (double)s_total.total_dispatches / (double)s_blocks : 0.0);

    fprintf(stderr,
            "ds4: dspark verify dispatch-est path=%s block=%llu n=%u layers=%u "
            "embed=%llu attn=%llu kv=%llu comp=%llu index=%llu heads=%llu "
            "output_hc=%llu ffn_pre=%llu router=%llu routed_moe=%llu "
            "ordered_sum=%llu shared=%llu post_hc=%llu target_hidden=%llu "
            "head=%llu topk=%llu read=%llu total=%llu avg_total=%.1f\n",
            path ? path : "unknown",
            (unsigned long long)s_blocks,
            n_tokens,
            n_layers,
            (unsigned long long)current.embed_dispatches,
            (unsigned long long)current.attention_dispatches,
            (unsigned long long)current.kv_store_dispatches,
            (unsigned long long)current.compressor_dispatches,
            (unsigned long long)current.indexer_dispatches,
            (unsigned long long)current.attn_head_dispatches,
            (unsigned long long)current.output_hc_dispatches,
            (unsigned long long)current.ffn_pre_dispatches,
            (unsigned long long)current.router_dispatches,
            (unsigned long long)current.routed_moe_dispatches,
            (unsigned long long)current.ordered_sum_dispatches,
            (unsigned long long)current.shared_dispatches,
            (unsigned long long)current.post_hc_dispatches,
            (unsigned long long)current.target_hidden_dispatches,
            (unsigned long long)current.output_head_dispatches,
            (unsigned long long)current.topk_dispatches,
            (unsigned long long)current.readbacks,
            (unsigned long long)current.total_dispatches,
            s_blocks ? (double)s_total.total_dispatches / (double)s_blocks : 0.0);
}

static ds4_dspark_decodeN_policy ds4_dspark_decodeN_policy_make(
        bool quality,
        int  draft_n) {
    ds4_dspark_decodeN_policy p;
    memset(&p, 0, sizeof(p));

    p.batch_verify =
        !quality &&
        !env_flag_enabled("DS4_DSPARK_EXACT_VERIFY") &&
        env_flag_enabled("DS4_DSPARK_BATCH_VERIFY");
    p.sequential_verify =
        env_flag_enabled("DS4_DSPARK_SEQUENTIAL_VERIFY") ||
        env_flag_enabled("DS4_DSPARK_SEQ_VERIFY");
    p.decodeN_attn_ffn_batch =
        env_flag_enabled("DS4_DSPARK_DECODEN_ATTN_FFN_BATCH") ||
        env_flag_enabled("DS4_DSPARK_DECODE_N_ATTN_FFN_BATCH");

    const bool hybrid_batch_attn_decode_order =
        p.decodeN_attn_ffn_batch &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER");
    const bool hybrid_state_audit_requested =
        p.decodeN_attn_ffn_batch &&
        (env_flag_enabled("DS4_DSPARK_HYBRID_STATE_AUDIT") ||
         env_flag_enabled("DS4_DSPARK_DECODEN_HYBRID_STATE_AUDIT"));
    const bool hybrid_prefix_safe =
        p.decodeN_attn_ffn_batch &&
        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_SHARED") &&
        !hybrid_batch_attn_decode_order &&
        !env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED");
    const bool hybrid_batch_attn_prefix_safe =
        p.decodeN_attn_ffn_batch &&
        hybrid_batch_attn_decode_order &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTER") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_SHARED") &&
        !env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED") &&
        !env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_PREFIXN_DISABLE");
    const uint32_t hybrid_decode_order_max_clean =
        get_env_u32_clamped("DS4_DSPARK_HYBRID_BATCH_ATTN_MAX_CLEAN",
                            5u,
                            2u,
                            5u);

    p.hybrid_allowed =
        p.decodeN_attn_ffn_batch &&
        (!hybrid_batch_attn_decode_order ||
         draft_n <= (int)hybrid_decode_order_max_clean ||
         env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ALLOW_UNSAFE_N_GT3"));
    p.hybrid_margin_guard =
        p.hybrid_allowed ? ds4_env_positive_float("DS4_DSPARK_HYBRID_MARGIN_GUARD") : 0.0f;
    p.hybrid_exact_every =
        p.hybrid_allowed ?
        get_env_u32_clamped("DS4_DSPARK_HYBRID_EXACT_EVERY",
                            0u,
                            0u,
                            UINT32_MAX) : 0u;

    static uint64_t s_hybrid_blocks = 0;
    p.hybrid_block_id = p.hybrid_allowed ? ++s_hybrid_blocks : 0u;
    const uint32_t hybrid_state_audit_from =
        hybrid_state_audit_requested ?
        get_env_u32_clamped("DS4_DSPARK_HYBRID_STATE_AUDIT_FROM",
                            1u,
                            1u,
                            UINT32_MAX) : 1u;
    const uint32_t hybrid_state_audit_to =
        hybrid_state_audit_requested ?
        get_env_u32_clamped("DS4_DSPARK_HYBRID_STATE_AUDIT_TO",
                            0u,
                            0u,
                            UINT32_MAX) : 0u;
    p.hybrid_state_audit =
        hybrid_state_audit_requested &&
        p.hybrid_block_id >= hybrid_state_audit_from &&
        (hybrid_state_audit_to == 0u || p.hybrid_block_id <= hybrid_state_audit_to);

    p.decodeN_reads_all_logits =
        metal_graph_verify_decodeN_read_all_logits((uint32_t)draft_n);
    p.decodeN_row_count = (size_t)(p.decodeN_reads_all_logits ? draft_n : 1);
    p.decodeN_capture_prefixes =
        draft_n > 1 &&
        (!p.hybrid_allowed ||
         p.hybrid_state_audit ||
         hybrid_prefix_safe ||
         hybrid_batch_attn_prefix_safe) &&
        !env_flag_enabled("DS4_DSPARK_DECODEN_PREFIX1_DISABLE") &&
        !env_flag_enabled("DS4_DSPARK_DECODEN_PREFIXN_DISABLE");
    if (p.decodeN_capture_prefixes) {
        p.decodeN_capture_prefix_count = (uint32_t)(draft_n - 1);
        if (p.decodeN_capture_prefix_count > DS4_SPEC_PREFIX_SLOTS) {
            p.decodeN_capture_prefix_count = DS4_SPEC_PREFIX_SLOTS;
        }
    }

    p.hybrid_layer_hc_audit =
        p.hybrid_state_audit &&
        env_flag_enabled("DS4_DSPARK_HYBRID_LAYER_HC_AUDIT");
    p.hybrid_dspark_kv_audit =
        p.hybrid_state_audit &&
        (env_flag_enabled("DS4_DSPARK_HYBRID_STATE_AUDIT_DSPARK_KV") ||
         env_flag_enabled("DS4_DSPARK_BATCH_STATE_AUDIT_DSPARK_KV"));
    return p;
}

static bool metal_graph_verify_decodeN_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const int             *tokens,
        uint32_t               n_tokens,
        uint32_t               start,
        uint32_t               capture_prefix_count,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_layer_hc_audit *hc_audit) {
    if (!g || !model || !weights || !tokens ||
        n_tokens == 0 || n_tokens > 5u ||
        n_tokens > g->prefill_cap || !g->spec_logits || g->raw_cap == 0) {
        return false;
    }
    if (n_tokens > 1u && !row_tops) return false;
    if (capture_prefix_count > DS4_SPEC_PREFIX_SLOTS) {
        capture_prefix_count = DS4_SPEC_PREFIX_SLOTS;
    }
    if (capture_prefix_count >= n_tokens) {
        capture_prefix_count = n_tokens > 0 ? n_tokens - 1u : 0u;
    }
	    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
	    const uint32_t n_layers = metal_graph_spec_verify_layer_limit();
	    const bool exact_stage_profile =
	        env_flag_enabled("DS4_DSPARK_EXACT_STAGE_PROFILE");
	    const double exact_profile_t0 = exact_stage_profile ? now_sec() : 0.0;
	    double exact_profile_embed_s = 0.0;
	    double exact_profile_layers_s = 0.0;
	    double exact_profile_head_s = 0.0;
	    double exact_profile_read_s = 0.0;

	    ds4_gpu_tensor *cur[5] = { NULL, NULL, NULL, NULL, NULL };
	    ds4_gpu_tensor *next[5] = { NULL, NULL, NULL, NULL, NULL };
    bool ok = true;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        cur[t] = metal_graph_tensor_row_view(g->batch_cur_hc, t, hc_dim);
        next[t] = metal_graph_tensor_row_view(g->batch_next_hc, t, hc_dim);
        ok = cur[t] && next[t];
    }
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        ok = ds4_gpu_embed_token_hc_tensor(cur[t],
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           (uint32_t)tokens[t],
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
	    }
	    if (ok) ok = ds4_gpu_end_commands() != 0;
	    else (void)ds4_gpu_synchronize();
	    if (exact_stage_profile) {
	        exact_profile_embed_s = now_sec() - exact_profile_t0;
	    }
	    if (!ok) goto done;

	    ds4_gpu_tensor *saved_cur = g->cur_hc;
	    ds4_gpu_tensor *saved_after = g->after_ffn_hc;

	    const double exact_profile_layers_t0 =
	        exact_stage_profile ? now_sec() : 0.0;
	    bool commands_open = false;
	    ok = ds4_gpu_begin_commands() != 0;
	    commands_open = ok;
	    const uint32_t split_after_layers = metal_graph_spec_verify_split_layers();
    for (uint32_t il = 0; ok && il < n_layers; il++) {
	        for (uint32_t t = 0; ok && t < n_tokens; t++) {
	            const uint32_t pos = start + t;
	            g->cur_hc = cur[t];
	            g->after_ffn_hc = next[t];
	            const bool capture_raw_heads =
	                hc_audit && metal_graph_dspark_attn_stage_audit_enabled();
	            if (capture_raw_heads) {
	                g->spec_capture_attn_heads_raw = true;
	                g->spec_capture_attn_heads_raw_row = t;
	            }
	            ok = metal_graph_encode_decode_layer(g,
	                                                 model,
	                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
		                                                 g->raw_cap,
		                                                 pos % g->raw_cap,
		                                                 metal_graph_raw_span_for_batch(g, pos, 1),
		                                                 tokens[t]);
	            if (capture_raw_heads) {
	                g->spec_capture_attn_heads_raw = false;
	            }
		            if (ok && hc_audit && metal_graph_dspark_attn_stage_audit_enabled()) {
		                ok = metal_graph_copy_attention_debug_tensors_to_batch_row(g, t);
		            }
	            if (ok && hc_audit) {
	                ds4_gpu_tensor *attn_dst = metal_graph_tensor_row_view(g->batch_after_attn_hc,
	                                                                       t,
                                                                       hc_dim);
                ok = attn_dst &&
                     ds4_gpu_tensor_copy(attn_dst,
                                         0,
                                         g->after_attn_hc,
                                         0,
                                         hc_dim * sizeof(float)) != 0;
                ds4_gpu_tensor_free(attn_dst);
            }
            if (ok && hc_audit) {
                ok = metal_graph_copy_tensor_to_batch_row(g->batch_ffn_cur,
                                                          t,
                                                          DS4_N_EMBD,
                                                          g->ffn_cur) &&
                     metal_graph_copy_tensor_to_batch_row(g->batch_ffn_norm,
                                                          t,
                                                          DS4_N_EMBD,
                                                          g->ffn_norm) &&
                     metal_graph_copy_tensor_to_batch_row(g->batch_routed_out,
                                                          t,
                                                          DS4_N_EMBD,
                                                          g->routed_out) &&
                     metal_graph_copy_tensor_to_batch_row(g->batch_shared_out,
                                                          t,
                                                          DS4_N_EMBD,
                                                          g->shared_out);
            }
            const uint32_t prefix_len = t + 1u;
            if (ok && prefix_len <= capture_prefix_count) {
                ok = metal_graph_capture_prefix_state(g, il, prefix_len);
            }
        }
        for (uint32_t t = 0; ok && t < n_tokens; t++) {
            ds4_gpu_tensor *tmp = cur[t];
            cur[t] = next[t];
            next[t] = tmp;
        }
	        if (ok && hc_audit) {
	            ok = ds4_gpu_end_commands() != 0;
	            commands_open = false;
	            if (ok && metal_graph_dspark_attn_stage_audit_enabled()) {
	                ok = ds4_verify_layer_hc_audit_capture_attention_tensors(hc_audit,
	                                                                          g,
	                                                                          il,
	                                                                          n_tokens);
	            }
	            if (ok) ok = ds4_verify_layer_hc_audit_capture_rows(hc_audit,
	                                                                 cur,
	                                                                 il,
                                                                 n_tokens);
            if (ok) ok = ds4_verify_layer_hc_audit_capture_attn_tensor(hc_audit,
                                                                        g->batch_after_attn_hc,
                                                                        il,
                                                                        n_tokens);
            if (ok && env_flag_enabled("DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT")) {
                ok = ds4_verify_layer_hc_audit_capture_ffn_tensors(hc_audit,
                                                                   g,
                                                                   il,
                                                                   n_tokens);
            }
            if (ok && il + 1u < n_layers) {
                ok = ds4_gpu_begin_commands() != 0;
                commands_open = ok;
            }
        }
        if (ok && split_after_layers != 0 &&
            commands_open &&
            il + 1u == split_after_layers &&
            il + 1u < n_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }
	    if (ok && commands_open) ok = ds4_gpu_end_commands() != 0;
	    else (void)ds4_gpu_synchronize();
	    if (exact_stage_profile) {
	        exact_profile_layers_s = now_sec() - exact_profile_layers_t0;
	    }
	    g->cur_hc = saved_cur;
	    g->after_ffn_hc = saved_after;
	    if (!ok) goto done;

	    ds4_gpu_tensor *saved_batch_cur = g->batch_cur_hc;
	    ds4_gpu_tensor *final_batch_hc = metal_graph_spec_final_batch_hc(g);
	    g->batch_cur_hc = final_batch_hc;
	    const double exact_profile_head_t0 =
	        exact_stage_profile ? now_sec() : 0.0;
	    ok = ds4_gpu_begin_commands() != 0;
	    if (ok) ok = metal_graph_encode_output_head_batch(g,
	                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok && n_tokens > 1u) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                         g->spec_logits,
                                         DS4_N_VOCAB,
                                         n_tokens - 1u,
                                         1) != 0;
	    }
	    if (ok) ok = ds4_gpu_end_commands() != 0;
	    else (void)ds4_gpu_synchronize();
	    if (exact_stage_profile) {
	        exact_profile_head_s = now_sec() - exact_profile_head_t0;
	    }
	    g->batch_cur_hc = saved_batch_cur;
	    if (!ok) goto done;

	    const double exact_profile_read_t0 =
	        exact_stage_profile ? now_sec() : 0.0;
	    if (n_tokens > 1u) {
	        ok = ds4_gpu_tensor_read(g->comp_selected,
	                                 0,
                                 row_tops,
                                 (uint64_t)(n_tokens - 1u) * sizeof(row_tops[0])) != 0;
    }
    if (ok && row_logits) {
        const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(row_logits[0]);
        if (metal_graph_verify_decodeN_read_all_logits(n_tokens)) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     0,
                                     row_logits,
                                     (uint64_t)n_tokens * row_bytes) != 0;
        } else {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     (uint64_t)(n_tokens - 1u) * row_bytes,
                                     row_logits,
	                                     row_bytes) != 0;
	        }
	    }
	    if (exact_stage_profile) {
	        exact_profile_read_s = now_sec() - exact_profile_read_t0;
	        fprintf(stderr,
	                "ds4: dspark exact decodeN profile n=%u layers=%u prefix_slots=%u read_all=%d embed=%.3f ms layers=%.3f ms head=%.3f ms read=%.3f ms total=%.3f ms\n",
	                n_tokens,
	                n_layers,
	                capture_prefix_count,
	                metal_graph_verify_decodeN_read_all_logits(n_tokens) ? 1 : 0,
	                exact_profile_embed_s * 1000.0,
	                exact_profile_layers_s * 1000.0,
	                exact_profile_head_s * 1000.0,
	                exact_profile_read_s * 1000.0,
	                (now_sec() - exact_profile_t0) * 1000.0);
	    }

	done:
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor_free(next[t]);
        ds4_gpu_tensor_free(cur[t]);
    }
    return ok;
}

/* Byte-clean strict verifier core: exact decode-order attention/cache updates
 * followed by only the row-exact/batch helpers that passed commit-state audits.
 * Keep unsafe verifier experiments out of this function; strict_v2 work should
 * go behind separate gates. */
static bool metal_graph_verify_decodeN_attn_exact_ffn_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const int             *tokens,
        uint32_t               n_tokens,
        uint32_t               start,
        bool                   capture_prefix1,
        uint32_t               capture_prefix_count,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_layer_hc_audit *hc_audit) {
    if (!g || !model || !weights || !tokens || n_tokens == 0 || n_tokens > 5u ||
        n_tokens > g->prefill_cap || !g->spec_logits || g->raw_cap == 0) {
        return false;
    }
    if (n_tokens > 1u && !row_tops) return false;
    if (capture_prefix_count > DS4_SPEC_PREFIX_SLOTS) {
        capture_prefix_count = DS4_SPEC_PREFIX_SLOTS;
    }
    if (capture_prefix_count >= n_tokens) {
        capture_prefix_count = n_tokens > 0 ? n_tokens - 1u : 0u;
    }
    if (!metal_graph_ensure_prefill_scratch(g, weights, &weights->layer[0])) return false;

    int32_t token_ids[5] = {0, 0, 0, 0, 0};
    for (uint32_t i = 0; i < n_tokens; i++) token_ids[i] = (int32_t)tokens[i];
    if (ds4_gpu_tensor_write(g->prefill_tokens,
                             0,
                             token_ids,
                             (uint64_t)n_tokens * sizeof(token_ids[0])) == 0) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        ds4_gpu_tensor *row_hc = metal_graph_tensor_row_view(g->batch_cur_hc, t, hc_dim);
        ok = row_hc &&
             ds4_gpu_embed_token_hc_tensor(row_hc,
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           (uint32_t)tokens[t],
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
        ds4_gpu_tensor_free(row_hc);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (!ok) return false;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after_attn = g->after_attn_hc;
    ds4_gpu_tensor *saved_after_ffn = g->after_ffn_hc;
    ds4_gpu_tensor *saved_batch_cur = g->batch_cur_hc;
    ds4_gpu_tensor *saved_batch_next = g->batch_next_hc;
    const bool saved_capture = g->spec_capture_prefix1;
    const bool saved_mtp_enabled = g->mtp_enabled;
    const bool saved_disable_shared_gate_up_swiglu =
        g->spec_disable_shared_gate_up_swiglu;
    g->spec_capture_prefix1 = capture_prefix1 && n_tokens == 2u;
    g->mtp_enabled = true;
	    g->spec_disable_shared_gate_up_swiglu = true;

	    const uint32_t n_layers = metal_graph_spec_verify_layer_limit();
	    const uint32_t exact_prefix_layers =
	        get_env_u32_clamped("DS4_DSPARK_HYBRID_EXACT_PREFIX_LAYERS",
	                            0u,
	                            0u,
	                            DS4_N_LAYER);
	    const bool row_exact_router =
	        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTER");
	    const bool row_exact_shared =
	        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_SHARED");
		    const bool row_exact_routed =
		        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_ROUTED");
		    const bool batch_router_row_routed =
		        row_exact_routed &&
		        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED");
		    const bool row_exact_ffn_pre =
		        row_exact_router || row_exact_shared || row_exact_routed ||
		        env_flag_enabled("DS4_DSPARK_HYBRID_ROW_FFN_PRE");
			    const bool row_exact_router_effective =
			        row_exact_router || (row_exact_routed && !batch_router_row_routed);
    const bool hybrid_batch_attn_decode_order =
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER");
    const bool hybrid_batch_attn_prefix_safe =
        hybrid_batch_attn_decode_order &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV") &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT") &&
        row_exact_router &&
        row_exact_routed &&
        row_exact_shared &&
        !batch_router_row_routed &&
        !env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_PREFIXN_DISABLE");
    const bool hybrid_batch_attn_row_output =
        hybrid_batch_attn_decode_order &&
        env_flag_enabled("DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT");
    const uint32_t batch_attn_capture_prefix_count =
        hybrid_batch_attn_prefix_safe ? capture_prefix_count : 0u;
    const bool hybrid_stage_profile =
        env_flag_enabled("DS4_DSPARK_HYBRID_STAGE_PROFILE");
    const bool hybrid_dispatch_profile =
        ds4_dspark_verify_dispatch_profile_enabled();
    const bool routed_overlap_profile =
        ds4_dspark_routed_overlap_profile_enabled();
    const bool hybrid_attn_subprofile =
        ds4_dspark_attn_subprofile_enabled();
    const bool shared_prefix_profile =
        ds4_dspark_shared_prefix_profile_enabled();
    ds4_dspark_verify_dispatch_stats hybrid_dispatch_stats;
    memset(&hybrid_dispatch_stats, 0, sizeof(hybrid_dispatch_stats));
    ds4_dspark_routed_overlap_stats routed_overlap_stats;
    memset(&routed_overlap_stats, 0, sizeof(routed_overlap_stats));
    if (hybrid_dispatch_profile) {
        hybrid_dispatch_stats.embed_dispatches += n_tokens;
    }
    if (hybrid_attn_subprofile) {
        ds4_dspark_attn_subprofile_begin(n_tokens, n_layers);
    }
    if (shared_prefix_profile) {
        ds4_dspark_shared_prefix_profile_begin(n_tokens, n_layers);
    }
	    double hybrid_profile_stage_t0 = hybrid_stage_profile ? now_sec() : 0.0;
	    const double hybrid_profile_t0 = hybrid_profile_stage_t0;
	    double hybrid_profile_exact_prefix_s = 0.0;
	    double hybrid_profile_attention_s = 0.0;
	    double hybrid_profile_row_ffn_pre_s = 0.0;
	    double hybrid_profile_row_router_s = 0.0;
	    double hybrid_profile_batch_router_s = 0.0;
	    double hybrid_profile_row_routed_s = 0.0;
	    double hybrid_profile_row_shared_s = 0.0;
	    double hybrid_profile_batch_tail_s = 0.0;
	    double hybrid_profile_target_hidden_s = 0.0;
	    double hybrid_profile_head_s = 0.0;
	    double hybrid_profile_read_s = 0.0;
		    bool commands_open = false;
		    ok = ds4_gpu_begin_commands() != 0;
	    commands_open = ok;
#define DS4_DSPARK_HYBRID_PROFILE_STAGE(accum_) do { \
        if (ok && hybrid_stage_profile) { \
            if (commands_open) { \
                ok = ds4_gpu_flush_commands_blocking() != 0; \
                commands_open = ok; \
            } \
            const double _hybrid_profile_now = now_sec(); \
            (accum_) += _hybrid_profile_now - hybrid_profile_stage_t0; \
            hybrid_profile_stage_t0 = _hybrid_profile_now; \
        } \
    } while (0)
	    for (uint32_t il = 0; ok && il < n_layers; il++) {
        if (il < exact_prefix_layers) {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                const uint32_t pos = start + t;
                ds4_gpu_tensor *row_cur = metal_graph_tensor_row_view(g->batch_cur_hc, t, hc_dim);
                ds4_gpu_tensor *row_next = metal_graph_tensor_row_view(g->batch_next_hc, t, hc_dim);
	                ok = row_cur && row_next;
	                if (ok) {
	                    g->cur_hc = row_cur;
	                    g->after_ffn_hc = row_next;
	                    const bool capture_raw_heads =
	                        hc_audit && metal_graph_dspark_attn_stage_audit_enabled();
	                    if (capture_raw_heads) {
	                        g->spec_capture_attn_heads_raw = true;
	                        g->spec_capture_attn_heads_raw_row = t;
	                    }
	                    ok = metal_graph_encode_decode_layer(g,
	                                                         model,
	                                                         &weights->layer[il],
                                                         il,
                                                         pos,
                                                         g->layer_raw_cache[il],
                                                         g->raw_cap,
		                                                         pos % g->raw_cap,
		                                                         metal_graph_raw_span_for_batch(g, pos, 1),
		                                                         tokens[t]);
	                    if (capture_raw_heads) {
	                        g->spec_capture_attn_heads_raw = false;
	                    }
			                }
			                if (ok && hc_audit && metal_graph_dspark_attn_stage_audit_enabled()) {
			                    ok = metal_graph_copy_attention_debug_tensors_to_batch_row(g, t);
		                }
		                if (ok && hc_audit) {
		                    ds4_gpu_tensor *attn_dst = metal_graph_tensor_row_view(g->batch_after_attn_hc,
		                                                                           t,
	                                                                           hc_dim);
	                    ok = attn_dst &&
	                         ds4_gpu_tensor_copy(attn_dst,
	                                             0,
	                                             g->after_attn_hc,
	                                             0,
	                                             hc_dim * sizeof(float)) != 0;
	                    ds4_gpu_tensor_free(attn_dst);
	                }
	                if (ok && hc_audit) {
	                    ok = metal_graph_copy_tensor_to_batch_row(g->batch_ffn_cur,
	                                                              t,
	                                                              DS4_N_EMBD,
	                                                              g->ffn_cur) &&
	                         metal_graph_copy_tensor_to_batch_row(g->batch_ffn_norm,
	                                                              t,
	                                                              DS4_N_EMBD,
	                                                              g->ffn_norm) &&
	                         metal_graph_copy_tensor_to_batch_row(g->batch_routed_out,
	                                                              t,
	                                                              DS4_N_EMBD,
	                                                              g->routed_out) &&
	                         metal_graph_copy_tensor_to_batch_row(g->batch_shared_out,
	                                                              t,
	                                                              DS4_N_EMBD,
	                                                              g->shared_out);
	                }
	                g->cur_hc = saved_cur;
	                g->after_ffn_hc = saved_after_ffn;
	                ds4_gpu_tensor_free(row_next);
	                ds4_gpu_tensor_free(row_cur);
                const uint32_t prefix_len = t + 1u;
                if (ok && prefix_len <= capture_prefix_count) {
                    ok = metal_graph_capture_prefix_state(g, il, prefix_len);
                }
	                if (ok && t == 0) {
	                    ok = metal_graph_capture_prefix1_attn_state(g, il) &&
	                         metal_graph_capture_prefix1_index_state(g, il);
	                }
	            }
	            if (ok && hybrid_dispatch_profile) {
	                ds4_dspark_verify_dispatch_count_exact_layer(&hybrid_dispatch_stats,
	                                                             il,
	                                                             n_tokens);
	            }
	            DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_exact_prefix_s);
	        } else {
	            if (hybrid_batch_attn_decode_order) {
                ok = metal_graph_encode_layer_attention_batch(g,
                                                              model,
                                                              &weights->layer[il],
                                                              il,
                                                              start,
                                                              n_tokens,
                                                              batch_attn_capture_prefix_count);
	            } else {
	                for (uint32_t t = 0; ok && t < n_tokens; t++) {
	                    const uint32_t pos = start + t;
	                    ds4_gpu_tensor *row_cur = metal_graph_tensor_row_view(g->batch_cur_hc, t, hc_dim);
	                    ds4_gpu_tensor *row_after_attn = metal_graph_tensor_row_view(g->batch_after_attn_hc, t, hc_dim);
		                    ok = row_cur && row_after_attn;
		                    if (ok) {
		                        g->cur_hc = row_cur;
		                        g->after_attn_hc = row_after_attn;
		                        const bool capture_raw_heads =
		                            hc_audit && metal_graph_dspark_attn_stage_audit_enabled();
		                        if (capture_raw_heads) {
		                            g->spec_capture_attn_heads_raw = true;
		                            g->spec_capture_attn_heads_raw_row = t;
		                        }
		                        ok = metal_graph_encode_decode_layer_ex(g,
		                                                                 model,
		                                                                 &weights->layer[il],
	                                                                 il,
	                                                                 pos,
	                                                                 g->layer_raw_cache[il],
	                                                                 g->raw_cap,
	                                                                 pos % g->raw_cap,
			                                                                 metal_graph_raw_span_for_batch(g, pos, 1),
			                                                                 tokens[t],
			                                                                 true);
		                        if (capture_raw_heads) {
		                            g->spec_capture_attn_heads_raw = false;
		                        }
			                    }
			                    if (ok && hc_audit && metal_graph_dspark_attn_stage_audit_enabled()) {
			                        ok = metal_graph_copy_attention_debug_tensors_to_batch_row(g, t);
		                    }
		                    g->cur_hc = saved_cur;
		                    g->after_attn_hc = saved_after_attn;
		                    ds4_gpu_tensor_free(row_after_attn);
	                    ds4_gpu_tensor_free(row_cur);
	                    const uint32_t prefix_len = t + 1u;
	                    if (ok && prefix_len <= capture_prefix_count) {
	                        ok = metal_graph_capture_prefix_state(g, il, prefix_len);
	                    }
	                    if (ok && t == 0) {
	                        ok = metal_graph_capture_prefix1_attn_state(g, il) &&
	                             metal_graph_capture_prefix1_index_state(g, il);
	                    }
	                }
	            }
	            if (ok && hybrid_dispatch_profile) {
	                ds4_dspark_verify_dispatch_count_attention(&hybrid_dispatch_stats,
	                                                           il,
	                                                           n_tokens,
	                                                           hybrid_batch_attn_decode_order,
	                                                           hybrid_batch_attn_row_output);
	            }
	            DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_attention_s);
		            if (ok && row_exact_ffn_pre) {
		                ok = metal_graph_encode_layer_ffn_pre_exact_rows(g,
		                                                                 model,
		                                                                 &weights->layer[il],
		                                                                 n_tokens);
		                if (ok && hybrid_dispatch_profile) {
		                    hybrid_dispatch_stats.ffn_pre_dispatches += 1u;
		                }
		                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_row_ffn_pre_s);
		            }
			            if (ok && row_exact_router_effective) {
			                ok = metal_graph_encode_layer_router_exact_rows(g,
			                                                               model,
			                                                               &weights->layer[il],
			                                                               tokens,
			                                                               n_tokens);
			                if (ok && hybrid_dispatch_profile) {
			                    hybrid_dispatch_stats.router_dispatches += 1u;
			                }
			                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_row_router_s);
			            }
			            if (ok && batch_router_row_routed) {
			                ok = metal_graph_encode_layer_router_batch_only(g,
			                                                               model,
			                                                               &weights->layer[il],
			                                                               il,
			                                                               start,
			                                                               n_tokens);
			                if (ok && hybrid_dispatch_profile) {
			                    hybrid_dispatch_stats.router_dispatches += 1u;
			                }
			                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_batch_router_s);
			            }
			            if (ok &&
			                routed_overlap_profile &&
			                (row_exact_router_effective || batch_router_row_routed)) {
			                ok = ds4_dspark_routed_overlap_stats_capture(g,
			                                                             &routed_overlap_stats,
			                                                             "hybrid",
			                                                             il,
			                                                             n_tokens,
			                                                             &commands_open);
			            }
			            if (ok && row_exact_routed) {
			                ok = metal_graph_encode_layer_routed_exact_rows(g,
			                                                               model,
		                                                               &weights->layer[il],
		                                                               il,
		                                                               n_tokens);
			                if (ok && hybrid_dispatch_profile) {
			                    hybrid_dispatch_stats.routed_moe_dispatches += 1u;
			                    if (!ds4_dspark_verify_direct_q2_ordered_sum_enabled(&weights->layer[il])) {
			                        hybrid_dispatch_stats.ordered_sum_dispatches += 1u;
			                    }
			                }
			                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_row_routed_s);
		            }
		            if (ok && row_exact_shared) {
		                ok = metal_graph_encode_layer_shared_gate_up_exact_rows(g,
		                                                                       model,
		                                                                       &weights->layer[il],
		                                                                       n_tokens);
		                if (ok && hybrid_dispatch_profile) {
		                    hybrid_dispatch_stats.shared_dispatches += 1u;
		                }
		                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_row_shared_s);
		            }
		            if (ok) {
		                ok = metal_graph_encode_layer_ffn_batch_ex(g,
	                                                           model,
	                                                           &weights->layer[il],
	                                                           il,
	                                                           start,
		                                                           n_tokens,
		                                                           row_exact_ffn_pre,
		                                                           row_exact_router_effective ||
		                                                               batch_router_row_routed,
			                                                           row_exact_shared,
			                                                           row_exact_routed);
		                if (ok && hybrid_dispatch_profile) {
		                    hybrid_dispatch_stats.post_hc_dispatches += 1u;
		                }
		                DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_batch_tail_s);
		            }
		        }
	        if (ok) ok = metal_graph_capture_dspark_target_hidden_batch(g, il, start, n_tokens);
	        if (ok && hybrid_dispatch_profile) {
	            hybrid_dispatch_stats.target_hidden_dispatches +=
	                ds4_dspark_target_hidden_dispatch_count(il, n_tokens);
	        }
	        DS4_DSPARK_HYBRID_PROFILE_STAGE(hybrid_profile_target_hidden_s);
	        if (ok) {
            ds4_gpu_tensor *tmp = g->batch_cur_hc;
            g->batch_cur_hc = g->batch_next_hc;
            g->batch_next_hc = tmp;
        }
	        if (ok && hc_audit) {
	            ok = ds4_gpu_end_commands() != 0;
	            commands_open = false;
		            if (ok && metal_graph_dspark_attn_stage_audit_enabled()) {
		                ok = ds4_verify_layer_hc_audit_capture_attention_tensors(hc_audit,
		                                                                          g,
		                                                                          il,
		                                                                          n_tokens);
		            }
		            if (ok) ok = ds4_verify_layer_hc_audit_capture_tensor(hc_audit,
		                                                                   g->batch_cur_hc,
		                                                                   il,
	                                                                   n_tokens);
	            if (ok) ok = ds4_verify_layer_hc_audit_capture_attn_tensor(hc_audit,
	                                                                        g->batch_after_attn_hc,
	                                                                        il,
	                                                                        n_tokens);
	            if (ok && env_flag_enabled("DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT")) {
	                ok = ds4_verify_layer_hc_audit_capture_ffn_tensors(hc_audit,
	                                                                   g,
	                                                                   il,
	                                                                   n_tokens);
	            }
	            if (ok && il + 1u < n_layers) {
	                ok = ds4_gpu_begin_commands() != 0;
	                commands_open = ok;
	            }
        }
    }
    if (ok && commands_open) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && hybrid_attn_subprofile) {
        ds4_dspark_attn_subprofile_print("hybrid");
    }
    if (ok && shared_prefix_profile) {
        ds4_dspark_shared_prefix_profile_print("hybrid");
    }
    g->cur_hc = saved_cur;
    g->after_attn_hc = saved_after_attn;
    g->after_ffn_hc = saved_after_ffn;
    g->spec_capture_prefix1 = saved_capture;
    g->mtp_enabled = saved_mtp_enabled;
    g->spec_disable_shared_gate_up_swiglu = saved_disable_shared_gate_up_swiglu;
    if (!ok) {
        g->batch_cur_hc = saved_batch_cur;
        g->batch_next_hc = saved_batch_next;
        return false;
    }

	    const uint32_t top_rows = n_tokens > 1u ? n_tokens - 1u : 0u;
	    const double hybrid_profile_head_t0 = hybrid_stage_profile ? now_sec() : 0.0;
	    ok = ds4_gpu_begin_commands() != 0;
	    if (ok) ok = metal_graph_encode_output_head_batch(g,
	                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok && hybrid_dispatch_profile) {
        hybrid_dispatch_stats.output_head_dispatches += 1u;
    }
    if (ok && top_rows) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                         g->spec_logits,
                                         DS4_N_VOCAB,
                                         top_rows,
                                         1) != 0;
        if (ok && hybrid_dispatch_profile) {
            hybrid_dispatch_stats.topk_dispatches += 1u;
        }
	    }
	    if (ok) ok = ds4_gpu_end_commands() != 0;
	    else (void)ds4_gpu_synchronize();
	    if (hybrid_stage_profile) {
	        hybrid_profile_head_s += now_sec() - hybrid_profile_head_t0;
	    }

	    const double hybrid_profile_read_t0 = hybrid_stage_profile ? now_sec() : 0.0;
	    if (ok && top_rows) {
	        ok = ds4_gpu_tensor_read(g->comp_selected,
	                                 0,
                                 row_tops,
                                 (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
        if (ok && hybrid_dispatch_profile) {
            hybrid_dispatch_stats.readbacks += 1u;
        }
    }
    if (ok && row_logits) {
        const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(row_logits[0]);
        if (metal_graph_verify_decodeN_read_all_logits(n_tokens)) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     0,
                                     row_logits,
                                     (uint64_t)n_tokens * row_bytes) != 0;
        } else {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     (uint64_t)(n_tokens - 1u) * row_bytes,
	                                     row_logits,
	                                     row_bytes) != 0;
	        }
        if (ok && hybrid_dispatch_profile) {
            hybrid_dispatch_stats.readbacks += 1u;
        }
	    }
	    if (ok && hybrid_dispatch_profile) {
	        ds4_dspark_verify_dispatch_stats_print("hybrid",
	                                               n_tokens,
	                                               n_layers,
	                                               &hybrid_dispatch_stats);
	    }
	    if (ok && routed_overlap_profile) {
	        ds4_dspark_routed_overlap_stats_print("hybrid",
	                                              n_tokens,
	                                              &routed_overlap_stats);
	    }
	    if (hybrid_stage_profile) {
	        hybrid_profile_read_s += now_sec() - hybrid_profile_read_t0;
	        fprintf(stderr,
	                "ds4: dspark hybrid verifier profile n=%u layers=%u exact_prefix=%.3f ms attention=%.3f ms row_ffn_pre=%.3f ms row_router=%.3f ms batch_router=%.3f ms row_routed=%.3f ms row_shared=%.3f ms batch_tail=%.3f ms target_hidden=%.3f ms head=%.3f ms read=%.3f ms total=%.3f ms\n",
	                n_tokens,
	                n_layers,
	                hybrid_profile_exact_prefix_s * 1000.0,
	                hybrid_profile_attention_s * 1000.0,
	                hybrid_profile_row_ffn_pre_s * 1000.0,
	                hybrid_profile_row_router_s * 1000.0,
	                hybrid_profile_batch_router_s * 1000.0,
	                hybrid_profile_row_routed_s * 1000.0,
	                hybrid_profile_row_shared_s * 1000.0,
	                hybrid_profile_batch_tail_s * 1000.0,
	                hybrid_profile_target_hidden_s * 1000.0,
	                hybrid_profile_head_s * 1000.0,
	                hybrid_profile_read_s * 1000.0,
	                (now_sec() - hybrid_profile_t0) * 1000.0);
	    }

	    g->batch_cur_hc = saved_batch_cur;
	    g->batch_next_hc = saved_batch_next;
#undef DS4_DSPARK_HYBRID_PROFILE_STAGE
	    return ok;
	}

/*
 * strict_v1 is the current byte-clean DSpark verifier. Do not add unsafe
 * batching here. New verifier speed work belongs behind strict_v2/profiling
 * gates and must graduate through cmp=0 runs before it can replace this path.
 */
static bool metal_graph_verify_decodeN_strict_v1(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const int             *tokens,
        uint32_t               n_tokens,
        uint32_t               start,
        bool                   capture_prefix1,
        uint32_t               capture_prefix_count,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_layer_hc_audit *hc_audit) {
    return metal_graph_verify_decodeN_attn_exact_ffn_batch(g,
                                                           model,
                                                           weights,
                                                           tokens,
                                                           n_tokens,
                                                           start,
                                                           capture_prefix1,
                                                           capture_prefix_count,
                                                           row_tops,
                                                           row_logits,
                                                           hc_audit);
}
