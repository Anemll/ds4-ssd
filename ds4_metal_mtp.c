/* =========================================================================
 * ds4_metal_mtp.c - Metal graph raw-cache span and MTP prefix capture helpers.
 * =========================================================================
 *
 * Included by ds4.c after the SSD Flash-MoE aggregator so later Metal graph
 * code can share raw-cache span and speculative prefix helpers.
 */

static uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy_pair(g->spec_prefix1_attn_state_kv[il], 0,
                                    g->layer_attn_state_kv[il], 0, bytes,
                                    g->spec_prefix1_attn_state_score[il], 0,
                                    g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy_pair(g->spec_prefix1_index_state_kv[il], 0,
                                    g->layer_index_state_kv[il], 0, bytes,
                                    g->spec_prefix1_index_state_score[il], 0,
                                    g->layer_index_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix_attn_state(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       prefix_len) {
    if (!g || prefix_len == 0 || prefix_len > DS4_SPEC_PREFIX_SLOTS) return false;
    const uint32_t slot = prefix_len - 1u;
    if (!g->spec_prefix_attn_state_kv[slot][il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix_n_comp[slot][il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy_pair(g->spec_prefix_attn_state_kv[slot][il], 0,
                                    g->layer_attn_state_kv[il], 0, bytes,
                                    g->spec_prefix_attn_state_score[slot][il], 0,
                                    g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix_index_state(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       prefix_len) {
    if (!g || prefix_len == 0 || prefix_len > DS4_SPEC_PREFIX_SLOTS) return false;
    const uint32_t slot = prefix_len - 1u;
    if (!g->spec_prefix_index_state_kv[slot][il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix_n_index_comp[slot][il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy_pair(g->spec_prefix_index_state_kv[slot][il], 0,
                                    g->layer_index_state_kv[il], 0, bytes,
                                    g->spec_prefix_index_state_score[slot][il], 0,
                                    g->layer_index_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix_state(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       prefix_len) {
    return metal_graph_capture_prefix_attn_state(g, il, prefix_len) &&
           metal_graph_capture_prefix_index_state(g, il, prefix_len);
}
