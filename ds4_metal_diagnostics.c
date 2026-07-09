/* =========================================================================
 * ds4_metal_diagnostics.c - Metal graph dump and stage-profile helpers.
 * =========================================================================
 *
 * Included by ds4.c so these static helpers can keep using the private graph
 * utilities without leaving the main Metal release path in ds4.c.
 */

static const char *metal_graph_debug_dump_prefix(void) {
    static bool initialized;
    static const char *prefix;
    if (!initialized) {
        initialized = true;
        const char *env = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
        if (env && env[0]) prefix = env;
    }
    return prefix;
}

static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    if (!metal_graph_debug_dump_prefix()) return false;

    static bool initialized;
    static const char *name_filter;
    static int layer_filter;
    static int pos_filter;
    if (!initialized) {
        initialized = true;
        layer_filter = -1;
        pos_filter = -1;

        const char *name_env = getenv("DS4_METAL_GRAPH_DUMP_NAME");
        if (name_env && name_env[0]) name_filter = name_env;

        const char *layer_env = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
        if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0) {
            char *end = NULL;
            unsigned long v = strtoul(layer_env, &end, 10);
            layer_filter = (end != layer_env && *end == '\0' && v <= UINT32_MAX) ? (int)v : -2;
        }

        const char *pos_env = getenv("DS4_METAL_GRAPH_DUMP_POS");
        if (pos_env && pos_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(pos_env, &end, 10);
            pos_filter = (end != pos_env && *end == '\0' && v <= UINT32_MAX) ? (int)v : -2;
        }
    }

    if (name_filter && strstr(name_filter, name) == NULL) return false;
    if (layer_filter >= 0 && (uint32_t)layer_filter != il) return false;
    if (layer_filter == -2) return false;
    if (pos_filter >= 0 && (uint32_t)pos_filter != pos) return false;
    if (pos_filter == -2) return false;
    return true;
}

static void metal_graph_debug_dump_tensor(
        const char     *name,
        ds4_gpu_tensor *t,
        uint64_t        n_f32,
        uint32_t        il,
        uint32_t        pos) {
    const char *prefix = metal_graph_debug_dump_prefix();
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_i32_tensor(
        const char     *name,
        ds4_gpu_tensor *t,
        uint64_t        n_i32,
        uint32_t        il,
        uint32_t        pos) {
    const char *prefix = metal_graph_debug_dump_prefix();
    if (!t || n_i32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_profile_layer_env_match_cached(const char *env_name, uint32_t il) {
    static int decode_layer = -2;
    static int layer_stage_layer = -2;
    int *cache = NULL;
    if (strcmp(env_name, "DS4_METAL_DECODE_STAGE_PROFILE_LAYER") == 0) {
        cache = &decode_layer;
    } else if (strcmp(env_name, "DS4_METAL_LAYER_STAGE_PROFILE_LAYER") == 0) {
        cache = &layer_stage_layer;
    }
    if (!cache) {
        const char *layer_env = getenv(env_name);
        if (!layer_env || !layer_env[0]) return true;
        char *end = NULL;
        const unsigned long layer = strtoul(layer_env, &end, 10);
        return end != layer_env && *end == '\0' && layer <= UINT32_MAX && (uint32_t)layer == il;
    }
    if (*cache == -2) {
        const char *layer_env = getenv(env_name);
        if (!layer_env || !layer_env[0]) {
            *cache = -1;
        } else {
            char *end = NULL;
            const unsigned long layer = strtoul(layer_env, &end, 10);
            *cache = (end != layer_env && *end == '\0' && layer <= UINT32_MAX) ? (int)layer : -3;
        }
    }
    return *cache == -1 || (*cache >= 0 && (uint32_t)*cache == il);
}

static bool metal_graph_decode_stage_profile_enabled(uint32_t il) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_METAL_DECODE_STAGE_PROFILE");
        enabled = env && env[0] && strcmp(env, "0") != 0;
    }
    return enabled != 0 &&
           metal_graph_profile_layer_env_match_cached("DS4_METAL_DECODE_STAGE_PROFILE_LAYER", il);
}

static bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}
