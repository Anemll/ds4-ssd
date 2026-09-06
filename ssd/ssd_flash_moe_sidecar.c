/* =========================================================================
 * ssd_flash_moe_sidecar.c - SSD Flash-MoE sidecar manifest and slot sizing.
 * =========================================================================
 *
 * Included early by ds4.c inside the GPU-only block so model-load and
 * weight-binding code can parse sidecar manifests before the graph-private
 * SSD runtime include.
 */

static char *flash_moe_join_path(const char *dir, const char *name) {
    return ds4_join_path(dir, name);
}

static bool flash_moe_read_file(const char *path, char **out, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open Flash-MoE manifest %s: %s\n", path, strerror(errno));
        return false;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return false;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return false;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return false;
    }
    char *buf = xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        fclose(fp);
        return false;
    }
    fclose(fp);
    buf[n] = '\0';
    *out = buf;
    *len_out = (size_t)n;
    return true;
}

static const char *flash_moe_find_key(const char *begin, const char *end, const char *key) {
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

static bool flash_moe_json_u64(const char *begin, const char *end, const char *key, uint64_t *out) {
    const char *p = flash_moe_find_key(begin, end, key);
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

static bool flash_moe_json_bool(const char *begin, const char *end, const char *key, bool *out) {
    const char *p = flash_moe_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p + 4 <= end && !strncmp(p, "true", 4)) {
        *out = true;
        return true;
    }
    if (p + 5 <= end && !strncmp(p, "false", 5)) {
        *out = false;
        return true;
    }
    return false;
}

static bool flash_moe_json_string(const char *begin, const char *end, const char *key, char **out) {
    const char *p = flash_moe_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '"') return false;
    p++;
    char *buf = xmalloc((size_t)(end - p) + 1);
    size_t n = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) {
                free(buf);
                return false;
            }
            switch (*p) {
            case '"':  buf[n++] = '"'; break;
            case '\\': buf[n++] = '\\'; break;
            case '/':  buf[n++] = '/'; break;
            case 'b':  buf[n++] = '\b'; break;
            case 'f':  buf[n++] = '\f'; break;
            case 'n':  buf[n++] = '\n'; break;
            case 'r':  buf[n++] = '\r'; break;
            case 't':  buf[n++] = '\t'; break;
            default:   buf[n++] = *p; break;
            }
            p++;
            continue;
        }
        buf[n++] = *p++;
    }
    if (p >= end || *p != '"') {
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    return true;
}

static bool flash_moe_json_u64_array3(const char *begin, const char *end, const char *key, uint64_t out[3]) {
    const char *p = flash_moe_find_key(begin, end, key);
    if (!p) return false;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '[') return false;
    p++;
    for (uint32_t i = 0; i < 3; i++) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end || !isdigit((unsigned char)*p)) return false;
        errno = 0;
        char *stop = NULL;
        unsigned long long v = strtoull(p, &stop, 10);
        if (errno != 0 || stop == p || stop > end) return false;
        out[i] = (uint64_t)v;
        p = stop;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (i < 2) {
            if (p >= end || *p != ',') return false;
            p++;
        }
    }
    while (p < end && isspace((unsigned char)*p)) p++;
    return p < end && *p == ']';
}

static int flash_moe_family_id(const char *family) {
    if (!strcmp(family, "ffn_gate_exps")) return DS4_FLASH_FAMILY_GATE;
    if (!strcmp(family, "ffn_up_exps")) return DS4_FLASH_FAMILY_UP;
    if (!strcmp(family, "ffn_down_exps")) return DS4_FLASH_FAMILY_DOWN;
    return -1;
}

static uint32_t flash_moe_quant_type_id(const char *quant, bool *mxfp4_plane_split) {
    if (mxfp4_plane_split) *mxfp4_plane_split = false;
    if (!strcmp(quant, "STQ1_0")) return DS4_TENSOR_STQ1_0;
    if (!strcmp(quant, "IQ1_M")) return DS4_TENSOR_IQ1_M;
    if (!strcmp(quant, "IQ2_XXS")) return DS4_TENSOR_IQ2_XXS;
    if (!strcmp(quant, "IQ3_XXS")) return DS4_TENSOR_IQ3_XXS;
    if (!strcmp(quant, "IQ4_XS")) return DS4_TENSOR_IQ4_XS;
    if (!strcmp(quant, "Q2_K")) return DS4_TENSOR_Q2_K;
    if (!strcmp(quant, "Q3_K")) return DS4_TENSOR_Q3_K;
    if (!strcmp(quant, "Q4_K")) return DS4_TENSOR_Q4_K;
    if (!strcmp(quant, "MXFP4")) return DS4_TENSOR_MXFP4;
    if (!strcmp(quant, "MXFP4_NATIVE")) {
        if (mxfp4_plane_split) *mxfp4_plane_split = true;
        return DS4_TENSOR_MXFP4;
    }
    fprintf(stderr, "ds4: Flash-MoE sidecar has unsupported routed quant type: %s\n", quant);
    return UINT32_MAX;
}

static bool flash_moe_expected_shape(uint32_t fam, const uint64_t shape[3]) {
    const uint64_t expected[DS4_FLASH_FAMILY_COUNT][3] = {
        [DS4_FLASH_FAMILY_GATE] = { DS4_N_EMBD,   DS4_N_FF_EXP, DS4_N_EXPERT },
        [DS4_FLASH_FAMILY_UP]   = { DS4_N_EMBD,   DS4_N_FF_EXP, DS4_N_EXPERT },
        [DS4_FLASH_FAMILY_DOWN] = { DS4_N_FF_EXP, DS4_N_EMBD,   DS4_N_EXPERT },
    };
    return fam < DS4_FLASH_FAMILY_COUNT &&
           shape[0] == expected[fam][0] &&
           shape[1] == expected[fam][1] &&
           shape[2] == expected[fam][2];
}

static bool flash_moe_layer_all_mxfp4_plane_split(
        const ds4_flash_moe_layer_sidecar *layer) {
    return layer &&
           layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
           layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
           layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
}

static bool flash_moe_layer_no_mxfp4_plane_split(
        const ds4_flash_moe_layer_sidecar *layer) {
    return layer &&
           !layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
           !layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
           !layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
}

static bool flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(
        const ds4_flash_moe_layer_sidecar *layer) {
    return layer &&
           layer->family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_IQ2_XXS &&
           layer->family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_IQ2_XXS &&
           layer->family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_MXFP4 &&
           !layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
           !layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
           layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
}

static bool flash_moe_for_each_entry(
        const char *json,
        size_t      json_len,
        bool      (*fn)(const char *obj, const char *end, void *ud),
        void       *ud) {
    const char *begin = json;
    const char *end = json + json_len;
    const char *entries = strstr(begin, "\"entries\"");
    if (!entries || entries >= end) return false;
    const char *p = strchr(entries, '[');
    if (!p || p >= end) return false;
    p++;
    while (p < end) {
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end || *p == ']') return true;
        if (*p != '{') return false;
        const char *obj = p;
        int depth = 0;
        bool in_string = false;
        bool escape = false;
        for (; p < end; p++) {
            const char c = *p;
            if (in_string) {
                if (escape) {
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    p++;
                    if (!fn(obj, p, ud)) return false;
                    break;
                }
            }
        }
        if (p >= end) return false;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && *p == ',') p++;
    }
    return false;
}

typedef struct {
    ds4_flash_moe_sidecar *sidecar;
    const char *dir;
    uint32_t seen_entries;
} flash_moe_parse_ctx;

static bool flash_moe_parse_entry(const char *obj, const char *end, void *ud) {
    flash_moe_parse_ctx *ctx = ud;
    uint64_t layer_u64 = 0;
    uint64_t bytes = 0;
    uint64_t exact_bytes = 0;
    uint64_t offset = 0;
    uint64_t stride = 0;
    uint64_t shape[3] = {0, 0, 0};
    char *family = NULL;
    char *file = NULL;
    char *quant = NULL;
    char *storage_layout = NULL;
    bool expert_major = true;
    bool has_expert_major = false;
    if (!flash_moe_json_u64(obj, end, "layer", &layer_u64) ||
        !flash_moe_json_string(obj, end, "tensor_family", &family) ||
        !flash_moe_json_string(obj, end, "quant_type", &quant) ||
        !flash_moe_json_u64_array3(obj, end, "shape", shape) ||
        !flash_moe_json_string(obj, end, "repacked_file", &file) ||
        !flash_moe_json_u64(obj, end, "bytes_per_expert", &bytes) ||
        !flash_moe_json_u64(obj, end, "repacked_offset", &offset)) {
        free(family);
        free(file);
        free(quant);
        free(storage_layout);
        return false;
    }
    (void)flash_moe_json_u64(obj, end, "expert_stride", &stride);
    has_expert_major = flash_moe_json_bool(obj, end, "expert_major", &expert_major);
    (void)flash_moe_json_u64(obj, end, "exact_byte_length", &exact_bytes);
    (void)flash_moe_json_string(obj, end, "storage_layout", &storage_layout);
    const int fam = flash_moe_family_id(family);
    bool mxfp4_plane_split = false;
    const uint32_t type = flash_moe_quant_type_id(quant, &mxfp4_plane_split);
    const bool family_major = (!has_expert_major && exact_bytes > bytes) || !expert_major;
    if (exact_bytes == 0) exact_bytes = bytes;
    if (layer_u64 >= DS4_N_LAYER || fam < 0 || bytes == 0 ||
        type == UINT32_MAX || !flash_moe_expected_shape((uint32_t)fam, shape)) {
        free(family);
        free(file);
        free(quant);
        free(storage_layout);
        return false;
    }
    if (mxfp4_plane_split &&
        (!storage_layout || strcmp(storage_layout, "mxfp4_plane_split_v1") != 0)) {
        fprintf(stderr,
                "ds4: Flash-MoE MXFP4_NATIVE entry requires storage_layout=mxfp4_plane_split_v1\n");
        free(family);
        free(file);
        free(quant);
        free(storage_layout);
        return false;
    }
    uint64_t plane_data_bytes = 0;
    uint64_t plane_scale_bytes = 0;
    if (mxfp4_plane_split) {
        if ((shape[0] % 32u) != 0 ||
            shape[1] > UINT64_MAX / (shape[0] / 2u)) {
            free(family);
            free(file);
            free(quant);
            free(storage_layout);
            return false;
        }
        plane_data_bytes = shape[1] * (shape[0] / 2u);
        plane_scale_bytes = shape[1] * (shape[0] / 32u);
        if (plane_data_bytes > UINT64_MAX - plane_scale_bytes ||
            bytes != plane_data_bytes + plane_scale_bytes) {
            fprintf(stderr,
                    "ds4: Flash-MoE MXFP4_NATIVE entry has invalid plane geometry\n");
            free(family);
            free(file);
            free(quant);
            free(storage_layout);
            return false;
        }
        uint64_t manifest_plane_data = 0;
        uint64_t manifest_plane_scale = 0;
        uint64_t manifest_plane_data_off = 0;
        uint64_t manifest_plane_scale_off = 0;
        if (offset > UINT64_MAX - plane_data_bytes ||
            (flash_moe_json_u64(obj, end, "plane_data_bytes", &manifest_plane_data) &&
             manifest_plane_data != plane_data_bytes) ||
            (flash_moe_json_u64(obj, end, "plane_scale_bytes", &manifest_plane_scale) &&
             manifest_plane_scale != plane_scale_bytes) ||
            (flash_moe_json_u64(obj, end, "plane_data_offset", &manifest_plane_data_off) &&
             manifest_plane_data_off != offset) ||
            (flash_moe_json_u64(obj, end, "plane_scale_offset", &manifest_plane_scale_off) &&
             manifest_plane_scale_off != offset + plane_data_bytes)) {
            fprintf(stderr,
                    "ds4: Flash-MoE MXFP4_NATIVE entry plane fields do not match deterministic layout\n");
            free(family);
            free(file);
            free(quant);
            free(storage_layout);
            return false;
        }
    }

    ds4_flash_moe_layer_sidecar *layer = &ctx->sidecar->layer[layer_u64];
    if (layer->present[fam]) {
        free(family);
        free(file);
        free(quant);
        free(storage_layout);
        return false;
    }
    if (!layer->path) {
        layer->path = flash_moe_join_path(ctx->dir, file);
    } else {
        char *path = flash_moe_join_path(ctx->dir, file);
        const bool same = strcmp(layer->path, path) == 0;
        free(path);
        if (!same) {
            free(family);
            free(file);
            free(quant);
            free(storage_layout);
            return false;
        }
    }
    layer->family_offset[fam] = offset;
    layer->family_file_offset[fam] = offset;
    layer->family_bytes[fam] = bytes;
    layer->family_file_bytes[fam] = family_major ? exact_bytes : bytes;
    layer->family_plane_data_bytes[fam] = plane_data_bytes;
    layer->family_plane_scale_bytes[fam] = plane_scale_bytes;
    layer->family_type[fam] = type;
    layer->family_mxfp4_plane_split[fam] = mxfp4_plane_split;
    layer->family_major = layer->family_major || family_major;
    layer->present[fam] = true;
    if (stride != 0) {
        if (layer->expert_stride != 0 && layer->expert_stride != stride) {
            free(family);
            free(file);
            free(quant);
            free(storage_layout);
            return false;
        }
        layer->expert_stride = stride;
    }
    ctx->seen_entries++;
    free(family);
    free(file);
    free(quant);
    free(storage_layout);
    return true;
}

static void ds4_flash_moe_sidecar_close(ds4_flash_moe_sidecar *s) {
    if (!s) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (s->layer[il].map && s->layer[il].map_size <= (uint64_t)SIZE_MAX) {
            munmap((void *)s->layer[il].map, (size_t)s->layer[il].map_size);
        }
        if (s->layer[il].fd >= 0) close(s->layer[il].fd);
        free(s->layer[il].path);
    }
    free(s->dir);
    free(s);
}

static bool ds4_flash_moe_sidecar_ensure_mmap(ds4_flash_moe_sidecar *s) {
    if (!s) return false;
    uint64_t mapped_bytes = s->expert_mmap_bytes;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_flash_moe_layer_sidecar *layer = &s->layer[il];
        if (!layer->present[DS4_FLASH_FAMILY_GATE] &&
            !layer->present[DS4_FLASH_FAMILY_UP] &&
            !layer->present[DS4_FLASH_FAMILY_DOWN]) {
            continue;
        }
        if (layer->map) continue;
        if (layer->fd < 0 || layer->file_size == 0 ||
            layer->file_size > (uint64_t)SIZE_MAX) {
            return false;
        }
        void *map = mmap(NULL,
                         (size_t)layer->file_size,
                         PROT_READ,
                         MAP_SHARED,
                         layer->fd,
                         0);
        if (map == MAP_FAILED) {
            fprintf(stderr,
                    "ds4: failed to mmap Flash-MoE sidecar layer %u: %s\n",
                    il,
                    strerror(errno));
            return false;
        }
        layer->map = (const uint8_t *)map;
        layer->map_size = layer->file_size;
        if (mapped_bytes <= UINT64_MAX - layer->file_size) {
            mapped_bytes += layer->file_size;
        } else {
            mapped_bytes = UINT64_MAX;
        }
    }
    s->expert_mmap = true;
    s->expert_mmap_bytes = mapped_bytes;
    return true;
}

static bool ds4_flash_moe_sidecar_open(
        ds4_flash_moe_sidecar **out,
        const char             *dir,
        uint32_t                slot_bank) {
    *out = NULL;
    if (!dir || !dir[0]) {
        fprintf(stderr, "ds4: --moe-mode slot-bank requires --moe-sidecar\n");
        return false;
    }
    const uint32_t active_expert_used = DS4_N_EXPERT_ACTIVE_USED;
    if (slot_bank < active_expert_used || slot_bank > DS4_N_EXPERT) {
        fprintf(stderr, "ds4: --moe-slot-bank must be between %u and %u\n",
                active_expert_used,
                (uint32_t)DS4_N_EXPERT);
        return false;
    }

    char *sidecar_dir = ds4_strdup(dir);
    char *manifest_path = flash_moe_join_path(sidecar_dir, "manifest.json");
    if (access(manifest_path, R_OK) != 0) {
        free(manifest_path);
        free(sidecar_dir);
        sidecar_dir = flash_moe_join_path(dir, "sidecar");
        manifest_path = flash_moe_join_path(sidecar_dir, "manifest.json");
    }

    char *json = NULL;
    size_t json_len = 0;
    if (!flash_moe_read_file(manifest_path, &json, &json_len)) {
        free(manifest_path);
        free(sidecar_dir);
        return false;
    }
    free(manifest_path);

    ds4_flash_moe_sidecar *s = xcalloc(1, sizeof(*s));
    s->dir = ds4_strdup(sidecar_dir);
    s->slot_bank = slot_bank;
    s->n_layer = DS4_N_LAYER;
    s->n_expert = DS4_N_EXPERT;
    const char *expert_mmap_env = getenv("DS4_FLASH_MOE_EXPERT_MMAP");
    const char *direct_mmap_env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_BANK");
    const char *mmap_bank_env = getenv("DS4_FLASH_MOE_MMAP_BANK");
    const char *file_backed_env = getenv("DS4_FLASH_MOE_FILE_BACKED_BANK");
    const char *direct_slots6_env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_SLOTS6");
    const char *active_mmap_env = getenv("DS4_FLASH_MOE_ACTIVE_MMAP_SLOTS6");
    s->expert_mmap =
        (expert_mmap_env && expert_mmap_env[0] && atoi(expert_mmap_env) != 0) ||
        (direct_mmap_env && direct_mmap_env[0] && atoi(direct_mmap_env) != 0) ||
        (mmap_bank_env && mmap_bank_env[0] && atoi(mmap_bank_env) != 0) ||
        (file_backed_env && file_backed_env[0] && atoi(file_backed_env) != 0) ||
        (direct_slots6_env && direct_slots6_env[0] && atoi(direct_slots6_env) != 0) ||
        (active_mmap_env && active_mmap_env[0] && atoi(active_mmap_env) != 0);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) s->layer[il].fd = -1;

    flash_moe_parse_ctx parse = {
        .sidecar = s,
        .dir = sidecar_dir,
        .seen_entries = 0,
    };
    const uint32_t first_routed_layer = DS4_N_DENSE_LEAD;
    const uint32_t expected_routed_layers =
        DS4_N_LAYER > first_routed_layer ? DS4_N_LAYER - first_routed_layer : 0;
    if (!flash_moe_for_each_entry(json, json_len, flash_moe_parse_entry, &parse) ||
        parse.seen_entries != expected_routed_layers * DS4_FLASH_FAMILY_COUNT) {
        fprintf(stderr, "ds4: Flash-MoE sidecar manifest does not contain the expected DS4 routed entries\n");
        free(json);
        free(sidecar_dir);
        ds4_flash_moe_sidecar_close(s);
        return false;
    }
    free(json);

    const char *direct_mmap_auto_env = getenv("DS4_FLASH_MOE_DIRECT_MMAP_AUTO");
    const char *disable_direct_mmap_auto_env = getenv("DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO");
    const char *force_mixed_env = getenv("DS4_FLASH_MOE_FORCE_MIXED_SLOT_BANK");
    const bool direct_mmap_auto_disabled =
        (direct_mmap_auto_env && direct_mmap_auto_env[0] && atoi(direct_mmap_auto_env) == 0) ||
        (disable_direct_mmap_auto_env && disable_direct_mmap_auto_env[0] &&
         atoi(disable_direct_mmap_auto_env) != 0) ||
        (force_mixed_env && force_mixed_env[0] && atoi(force_mixed_env) != 0);
    const bool direct_mmap_auto_q2_slots6 =
        !direct_mmap_auto_disabled &&
        first_routed_layer < DS4_N_LAYER &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_IQ2_XXS &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_IQ2_XXS &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_Q2_K;
    const bool direct_mmap_auto_large_mxfp4 =
        !direct_mmap_auto_disabled &&
        slot_bank > 128u &&
        first_routed_layer < DS4_N_LAYER &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_MXFP4 &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_MXFP4 &&
        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_MXFP4;
    if (direct_mmap_auto_q2_slots6 || direct_mmap_auto_large_mxfp4) {
        s->expert_mmap = true;
    }

    bool sidecar_has_mxfp4_plane_split = false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (il < first_routed_layer) continue;
        ds4_flash_moe_layer_sidecar *layer = &s->layer[il];
        const uint64_t expected_width[DS4_FLASH_FAMILY_COUNT] = {
            [DS4_FLASH_FAMILY_GATE] = DS4_N_EMBD,
            [DS4_FLASH_FAMILY_UP]   = DS4_N_EMBD,
            [DS4_FLASH_FAMILY_DOWN] = DS4_N_FF_EXP,
        };
        const uint64_t expected_rows[DS4_FLASH_FAMILY_COUNT] = {
            [DS4_FLASH_FAMILY_GATE] = DS4_N_FF_EXP,
            [DS4_FLASH_FAMILY_UP]   = DS4_N_FF_EXP,
            [DS4_FLASH_FAMILY_DOWN] = DS4_N_EMBD,
        };
        uint64_t min_stride = 0;
        uint64_t record_cursor = 0;
        uint64_t required_file_size = 0;
        for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
            if (!layer->present[fam] ||
                !tensor_is_routed_expert_type(layer->family_type[fam])) {
                fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has unexpected routed expert geometry\n", il);
                free(sidecar_dir);
                ds4_flash_moe_sidecar_close(s);
                return false;
            }
            const uint64_t row_bytes =
                routed_expert_row_bytes_for_type(layer->family_type[fam],
                                                 expected_width[fam]);
            const uint64_t expected = expected_rows[fam] * row_bytes;
            if (layer->family_bytes[fam] != expected ||
                layer->family_file_offset[fam] > UINT64_MAX - layer->family_file_bytes[fam]) {
                fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has unexpected routed expert geometry\n", il);
                free(sidecar_dir);
                ds4_flash_moe_sidecar_close(s);
                return false;
            }
            if (layer->family_mxfp4_plane_split[fam]) {
                const uint64_t plane_data = expected_rows[fam] * (expected_width[fam] / 2u);
                const uint64_t plane_scale = expected_rows[fam] * (expected_width[fam] / 32u);
                if (layer->family_type[fam] != DS4_TENSOR_MXFP4 ||
                    layer->family_plane_data_bytes[fam] != plane_data ||
                    layer->family_plane_scale_bytes[fam] != plane_scale ||
                    plane_data > UINT64_MAX - plane_scale ||
                    layer->family_bytes[fam] != plane_data + plane_scale) {
                    fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has invalid MXFP4_NATIVE plane geometry\n", il);
                    free(sidecar_dir);
                    ds4_flash_moe_sidecar_close(s);
                    return false;
                }
                sidecar_has_mxfp4_plane_split = true;
            }
            if (layer->family_major) {
                layer->family_offset[fam] = record_cursor;
                if (record_cursor > UINT64_MAX - layer->family_bytes[fam]) {
                    fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has invalid family-major record geometry\n", il);
                    free(sidecar_dir);
                    ds4_flash_moe_sidecar_close(s);
                    return false;
                }
                record_cursor += layer->family_bytes[fam];
                min_stride = record_cursor;
            } else {
                const uint64_t end = layer->family_offset[fam] + layer->family_bytes[fam];
                if (end > min_stride) min_stride = end;
            }
            const uint64_t file_end = layer->family_file_offset[fam] + layer->family_file_bytes[fam];
            if (file_end > required_file_size) required_file_size = file_end;
        }
        if (!flash_moe_layer_all_mxfp4_plane_split(layer) &&
            !flash_moe_layer_no_mxfp4_plane_split(layer) &&
            !flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(layer)) {
            fprintf(stderr,
                    "ds4: Flash-MoE sidecar layer %u uses unsupported mixed MXFP4 storage layout\n",
                    il);
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        if (layer->family_type[DS4_FLASH_FAMILY_GATE] !=
            layer->family_type[DS4_FLASH_FAMILY_UP]) {
            fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has mismatched gate/up routed quant types\n", il);
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        if (layer->expert_stride == 0) layer->expert_stride = min_stride;
        if (layer->expert_stride < min_stride) {
            fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has invalid expert stride\n", il);
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        if (layer->expert_stride > s->max_expert_stride) s->max_expert_stride = layer->expert_stride;
        layer->fd = open(layer->path, O_RDONLY);
        if (layer->fd < 0) {
            fprintf(stderr, "ds4: failed to open Flash-MoE sidecar layer %u: %s\n", il, strerror(errno));
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        struct stat st;
        if (fstat(layer->fd, &st) != 0 || st.st_size < 0) {
            fprintf(stderr, "ds4: failed to stat Flash-MoE sidecar layer %u\n", il);
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        layer->file_size = (uint64_t)st.st_size;
        const uint64_t required = layer->family_major ?
            required_file_size :
            (uint64_t)(DS4_N_EXPERT - 1) * layer->expert_stride + min_stride;
        if (layer->file_size < required) {
            fprintf(stderr, "ds4: Flash-MoE sidecar layer %u is truncated\n", il);
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        if (s->expert_mmap) {
            /* Cache-only mapping for experiments: the expert record transfer path
             * still uses pread(); this keeps the layer files mapped so the OS can
             * manage their pages independently of dense-model Metal views. */
            if (layer->file_size == 0 || layer->file_size > (uint64_t)SIZE_MAX) {
                fprintf(stderr, "ds4: Flash-MoE sidecar layer %u is too large to mmap\n", il);
                free(sidecar_dir);
                ds4_flash_moe_sidecar_close(s);
                return false;
            }
            void *map = mmap(NULL,
                             (size_t)layer->file_size,
                             PROT_READ,
                             MAP_SHARED,
                             layer->fd,
                             0);
            if (map == MAP_FAILED) {
                fprintf(stderr,
                        "ds4: failed to mmap Flash-MoE sidecar layer %u: %s\n",
                        il,
                        strerror(errno));
                free(sidecar_dir);
                ds4_flash_moe_sidecar_close(s);
                return false;
            }
            layer->map = (const uint8_t *)map;
            layer->map_size = layer->file_size;
            s->expert_mmap_bytes += layer->file_size;
        }
    }

    if (sidecar_has_mxfp4_plane_split) {
        if (!ds4_gpu_mxfp4_native_requested()) {
            fprintf(stderr,
                    "ds4: Flash-MoE sidecar uses MXFP4_NATIVE plane-split storage; "
                    "set DS4_MXFP4_NATIVE=1\n");
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
        if (!ds4_gpu_init() || !ds4_gpu_has_native_mxfp4()) {
            fprintf(stderr,
                    "ds4: Flash-MoE sidecar uses MXFP4_NATIVE plane-split storage, "
                    "but native MXFP4 MPP 4.1 support is not available on this host\n");
            free(sidecar_dir);
            ds4_flash_moe_sidecar_close(s);
            return false;
        }
    }

    free(sidecar_dir);
    *out = s;
    return true;
}

static bool flash_moe_mixed_slot_bank_enabled(void);
static bool flash_moe_layer_slot_slab_enabled(void);

static void ds4_flash_moe_sidecar_log_loaded(const ds4_flash_moe_sidecar *s) {
    if (!s) return;
    fprintf(stderr,
            "ds4: Flash-MoE sidecar loaded: %s (slot-bank=%u, expert-record %.2f MiB)\n",
            s->dir,
            s->slot_bank,
            (double)s->max_expert_stride / 1048576.0);
    const uint32_t first_routed_layer = DS4_N_DENSE_LEAD;
    uint32_t routed_layers = 0;
    uint32_t all_mxfp4_layers = 0;
    uint32_t iq2_mxfp4_down_layers = 0;
    uint32_t iq2_q2_down_layers = 0;
    for (uint32_t il = first_routed_layer; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &s->layer[il];
        if (!layer->present[DS4_FLASH_FAMILY_GATE] &&
            !layer->present[DS4_FLASH_FAMILY_UP] &&
            !layer->present[DS4_FLASH_FAMILY_DOWN]) {
            continue;
        }
        routed_layers++;
        if (flash_moe_layer_all_mxfp4_plane_split(layer)) {
            all_mxfp4_layers++;
        } else if (flash_moe_layer_iq2_gate_up_mxfp4_down_plane_split(layer)) {
            iq2_mxfp4_down_layers++;
        } else if (flash_moe_layer_no_mxfp4_plane_split(layer) &&
                   layer->family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_IQ2_XXS &&
                   layer->family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_IQ2_XXS &&
                   layer->family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_Q2_K) {
            iq2_q2_down_layers++;
        }
    }
    if (routed_layers != 0 && all_mxfp4_layers == routed_layers) {
        fprintf(stderr,
                "ds4: Flash-MoE MXFP4 storage layout: mxfp4_plane_split_v1 "
                "(native plane sidecar, no runtime repack on direct native paths)\n");
    } else if (routed_layers != 0 &&
               iq2_mxfp4_down_layers == routed_layers) {
        fprintf(stderr,
                "ds4: Flash-MoE routed experts: IQ2_XXS gate/up + "
                "MXFP4_NATIVE down (down mxfp4_plane_split_v1)\n");
    } else if (iq2_mxfp4_down_layers != 0 &&
               iq2_mxfp4_down_layers + iq2_q2_down_layers == routed_layers) {
        fprintf(stderr,
                "ds4: Flash-MoE routed experts: IQ2_XXS gate/up + mixed down "
                "(MXFP4_NATIVE plane-split layers=%u, Q2_K layers=%u)\n",
                iq2_mxfp4_down_layers,
                iq2_q2_down_layers);
    }
    if (s->expert_mmap) {
        fprintf(stderr,
                "ds4: Flash-MoE expert mmap cache: on (%.2f GiB mapped); expert reads: pread\n",
                (double)s->expert_mmap_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: Flash-MoE expert mmap cache: off; expert reads: pread\n");
    }
}

static bool ds4_flash_moe_sidecar_bank_bytes_for_slots(
        const ds4_flash_moe_sidecar *s,
        uint32_t                     slots,
        uint64_t                    *bytes_out) {
    if (!s || !bytes_out || slots == 0) return false;

    bool mixed = flash_moe_mixed_slot_bank_enabled();
    bool layer_slab = mixed && flash_moe_layer_slot_slab_enabled();
    uint64_t total = 0;
    const uint64_t slab_align = 4096u;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_flash_moe_layer_sidecar *layer = &s->layer[il];
        uint64_t layer_bytes = 0;
        if (mixed) {
            if (layer->expert_stride > UINT64_MAX / (uint64_t)slots) return false;
            layer_bytes = (uint64_t)slots * layer->expert_stride;
        } else {
            for (uint32_t fam = 0; fam < DS4_FLASH_FAMILY_COUNT; fam++) {
                if (layer->family_bytes[fam] > UINT64_MAX / (uint64_t)slots) return false;
                const uint64_t fam_bytes = (uint64_t)slots * layer->family_bytes[fam];
                if (layer_bytes > UINT64_MAX - fam_bytes) return false;
                layer_bytes += fam_bytes;
            }
        }
        if (layer_slab) {
            if (total > UINT64_MAX - (slab_align - 1u)) return false;
            total = align_up(total, slab_align);
        }
        if (total > UINT64_MAX - layer_bytes) return false;
        total += layer_bytes;
    }

    *bytes_out = total;
    return true;
}

static uint64_t ds4_system_available_memory_bytes(void) {
#if defined(__APPLE__)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    kern_return_t kr = host_statistics64(mach_host_self(),
                                         HOST_VM_INFO64,
                                         (host_info64_t)&vmstat,
                                         &count);
    vm_size_t page_size = 0;
    if (kr == KERN_SUCCESS && host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS) {
        const uint64_t pages = (uint64_t)vmstat.free_count +
                               (uint64_t)vmstat.inactive_count +
                               (uint64_t)vmstat.speculative_count;
        return pages * (uint64_t)page_size;
    }
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return ((uint64_t)info.freeram + (uint64_t)info.bufferram) *
               (uint64_t)info.mem_unit;
    }
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return (uint64_t)pages * (uint64_t)page_size;
    }
#endif
    return 0;
}

static bool ds4_flash_moe_ssd_cache_budget(
        const ds4_engine_options     *opt,
        const ds4_model              *model,
        const ds4_flash_moe_sidecar  *sidecar,
        uint64_t                     *budget_out,
        bool                         *auto_out) {
    (void)sidecar;
    if (!opt || !opt->ssd_cache || !opt->ssd_cache[0] || !budget_out || !auto_out) return false;

    if (!strcasecmp(opt->ssd_cache, "auto")) {
        const uint64_t available = ds4_system_available_memory_bytes();
        const uint64_t dense_bytes =
            model && model->size > model->tensor_data_pos ?
            model->size - model->tensor_data_pos :
            (model ? model->size : 0);
        const uint64_t kv_bytes =
            opt->ctx_size > 0 ?
            ds4_context_memory_estimate(opt->backend, opt->ctx_size).total_bytes : 0;
        if (available == 0 || available <= dense_bytes + kv_bytes) {
            fprintf(stderr,
                    "ds4: --ssd-cache auto cannot find enough available memory "
                    "(available=%.2f GiB, dense=%.2f GiB, context=%.2f GiB)\n",
                    (double)available / 1073741824.0,
                    (double)dense_bytes / 1073741824.0,
                    (double)kv_bytes / 1073741824.0);
            return false;
        }
        const uint64_t remaining = available - dense_bytes - kv_bytes;
        /* A wired slot bank competes with the OS file cache that serves
         * decode-miss preads of the sidecar at RAM speed. When the sidecar is
         * larger than RAM, oversizing the bank evicts that cache and decode
         * collapses to true SSD reads (measured ~40x slower at 85% on a
         * 128 GiB M5 Max; see docs/flash-moe-stable-slot-progress.md). Keep a
         * conservative default and let DS4_SSD_CACHE_AUTO_PCT override.
         * Sweep on M5 Max 128GB / 145GB sidecar (cold decode t/s):
         * 10%:10.5  20%:10.0  30%:9.2  40%:6.5  85%:0.21 — smaller is better
         * in the cold-decode regime; 20% stays near peak with bank headroom. */
        uint32_t pct = 20u;
        const char *pct_env = getenv("DS4_SSD_CACHE_AUTO_PCT");
        if (pct_env && pct_env[0]) {
            char *end = NULL;
            errno = 0;
            long v = strtol(pct_env, &end, 10);
            if (errno == 0 && end != pct_env && v >= 1 && v <= 100) pct = (uint32_t)v;
        }
        *budget_out = (remaining / 100u) * pct + (remaining % 100u) * pct / 100u;
        *auto_out = true;
        fprintf(stderr,
                "ds4: --ssd-cache auto: available=%.2f GiB dense=%.2f GiB "
                "context=%.2f GiB remaining=%.2f GiB budget=%.2f GiB (%u%%, "
                "DS4_SSD_CACHE_AUTO_PCT to override)\n",
                (double)available / 1073741824.0,
                (double)dense_bytes / 1073741824.0,
                (double)kv_bytes / 1073741824.0,
                (double)remaining / 1073741824.0,
                (double)*budget_out / 1073741824.0,
                pct);
        return true;
    }

    uint64_t budget = 0;
    if (!ds4_parse_u64_suffix(opt->ssd_cache, &budget) || budget == 0) {
        fprintf(stderr,
                "ds4: invalid --ssd-cache value '%s' (use bytes with K/M/G suffix, e.g. 25GB, or auto)\n",
                opt->ssd_cache);
        return false;
    }

    /* Clamp explicit budgets so the prefill bank can never overflow memory:
     * bank + dense + context must stay within DS4_SSD_CACHE_MAX_PCT% (default
     * 85, ~the GPU wired working-set budget) of physical RAM. Without this an
     * oversized --ssd-cache wires past RAM during prefill and the machine
     * swaps before the decode-bank shrink ever gets a chance to run. */
    const uint64_t ram = ds4_gpu_system_memory_bytes();
    if (ram > 0) {
        uint32_t max_pct = 85u;
        const char *max_env = getenv("DS4_SSD_CACHE_MAX_PCT");
        if (max_env && max_env[0]) {
            char *end = NULL;
            errno = 0;
            long v = strtol(max_env, &end, 10);
            if (errno == 0 && end != max_env && v >= 1 && v <= 100) max_pct = (uint32_t)v;
        }
        const uint64_t safe_total =
            (ram / 100u) * max_pct + (ram % 100u) * max_pct / 100u;
        const uint64_t dense_bytes =
            model && model->size > model->tensor_data_pos ?
            model->size - model->tensor_data_pos :
            (model ? model->size : 0);
        const uint64_t kv_bytes =
            opt->ctx_size > 0 ?
            ds4_context_memory_estimate(opt->backend, opt->ctx_size).total_bytes : 0;
        const uint64_t reserved = dense_bytes + kv_bytes;
        const uint64_t cap = safe_total > reserved ? safe_total - reserved : 0;
        if (budget > cap) {
            fprintf(stderr,
                    "ds4: --ssd-cache %s would overflow memory during prefill "
                    "(RAM=%.2f GiB, dense=%.2f GiB, context=%.2f GiB, cap=%u%%); "
                    "clamping bank budget %.2f -> %.2f GiB "
                    "(DS4_SSD_CACHE_MAX_PCT to override)\n",
                    opt->ssd_cache,
                    (double)ram / 1073741824.0,
                    (double)dense_bytes / 1073741824.0,
                    (double)kv_bytes / 1073741824.0,
                    max_pct,
                    (double)budget / 1073741824.0,
                    (double)cap / 1073741824.0);
            budget = cap;
        }
    }
    if (budget == 0) {
        fprintf(stderr,
                "ds4: --ssd-cache %s leaves no memory for a slot bank after dense/context\n",
                opt->ssd_cache);
        return false;
    }
    *budget_out = budget;
    *auto_out = false;
    return true;
}

static bool ds4_flash_moe_resolve_ssd_cache_slots(
        const ds4_engine_options    *opt,
        const ds4_model             *model,
        ds4_flash_moe_sidecar       *sidecar,
        uint32_t                    *slot_bank_io) {
    if (!opt || !opt->ssd_cache || !opt->ssd_cache[0] || !sidecar || !slot_bank_io) return true;

    uint64_t budget = 0;
    bool auto_budget = false;
    if (!ds4_flash_moe_ssd_cache_budget(opt, model, sidecar, &budget, &auto_budget)) {
        return false;
    }

    const uint32_t min_slots = DS4_N_EXPERT_ACTIVE_USED;
    uint64_t min_bytes = 0;
    if (!ds4_flash_moe_sidecar_bank_bytes_for_slots(sidecar, min_slots, &min_bytes)) {
        fprintf(stderr, "ds4: failed to size Flash-MoE slot bank for --ssd-cache\n");
        return false;
    }
    if (budget < min_bytes) {
        fprintf(stderr,
                "ds4: --ssd-cache %s is too small for the minimum slot bank "
                "(budget=%.2f GiB, min-slots=%u needs %.2f GiB)\n",
                opt->ssd_cache,
                (double)budget / 1073741824.0,
                min_slots,
                (double)min_bytes / 1073741824.0);
        return false;
    }

    uint32_t lo = min_slots;
    uint32_t hi = DS4_N_EXPERT;
    uint32_t best = min_slots;
    uint64_t best_bytes = min_bytes;
    while (lo <= hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        uint64_t bytes = 0;
        if (!ds4_flash_moe_sidecar_bank_bytes_for_slots(sidecar, mid, &bytes)) {
            if (mid == 0) break;
            hi = mid - 1u;
            continue;
        }
        if (bytes <= budget) {
            best = mid;
            best_bytes = bytes;
            lo = mid + 1u;
        } else {
            if (mid == 0) break;
            hi = mid - 1u;
        }
    }

    sidecar->slot_bank = best;
    sidecar->slot_bank_auto = auto_budget;
    *slot_bank_io = best;
    fprintf(stderr,
            "ds4: --ssd-cache %s resolved Flash-MoE slot bank: slots=%u "
            "gpu-bank=%.2f GiB budget=%.2f GiB%s\n",
            opt->ssd_cache,
            best,
            (double)best_bytes / 1073741824.0,
            (double)budget / 1073741824.0,
            auto_budget ? " (auto)" : "");
    return true;
}

static ds4_tensor *flash_moe_routed_stub_tensor(
        ds4_weights                  *w,
        const ds4_flash_moe_sidecar  *sidecar,
        uint32_t                      layer,
        uint32_t                      fam) {
    static const char *names[DS4_FLASH_FAMILY_COUNT] = {
        [DS4_FLASH_FAMILY_GATE] = "flash_moe.ffn_gate_exps.weight",
        [DS4_FLASH_FAMILY_UP]   = "flash_moe.ffn_up_exps.weight",
        [DS4_FLASH_FAMILY_DOWN] = "flash_moe.ffn_down_exps.weight",
    };
    if (!w || !sidecar || layer >= DS4_N_LAYER || fam >= DS4_FLASH_FAMILY_COUNT) {
        ds4_die("internal Flash-MoE routed tensor stub request is invalid");
    }
    const ds4_flash_moe_layer_sidecar *sl = &sidecar->layer[layer];
    if (!sl->present[fam]) ds4_die("Flash-MoE sidecar is missing a routed tensor family");

    ds4_tensor *t = &w->routed_stub[layer][fam];
    memset(t, 0, sizeof(*t));
    t->name.ptr = names[fam];
    t->name.len = strlen(names[fam]);
    t->ndim = 3;
    t->dim[0] = fam == DS4_FLASH_FAMILY_DOWN ? DS4_N_FF_EXP : DS4_N_EMBD;
    t->dim[1] = fam == DS4_FLASH_FAMILY_DOWN ? DS4_N_EMBD : DS4_N_FF_EXP;
    t->dim[2] = DS4_N_EXPERT;
    t->type = sl->family_type[fam];
    t->elements = t->dim[0] * t->dim[1] * t->dim[2];
    t->bytes = sl->family_bytes[fam] * DS4_N_EXPERT;
    return t;
}
