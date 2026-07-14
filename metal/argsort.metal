struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
};

struct ds4_metal_args_topk_logits {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t top_k;
    uint32_t pad;
    uint64_t score_token_stride;
    uint64_t topk_token_stride;
};

struct ds4_metal_args_argmax {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint64_t score_token_stride;
    uint64_t selected_token_stride;
};

struct ds4_metal_args_add_argmax {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint64_t score_token_stride;
    uint64_t add_token_stride;
    uint64_t selected_token_stride;
};

/* DSpark Markov W2 + base-logit selection.  The ordinary path materializes
 * the complete vocab-sized Markov vector and then launches add+argmax.  This
 * two-stage reduction keeps only one (value,index) pair per 256-vocab tile.
 * It is intentionally DSpark-specific: W2 is row-major F16 [vocab, rank],
 * while the Markov embedding and base logits are F32. */
struct ds4_metal_args_markov_argmax {
    uint32_t n_vocab;
    uint32_t rank;
    uint32_t n_blocks;
    uint32_t pad;
};

kernel void kernel_argmax_f32_i32(
        constant ds4_metal_args_argmax &args [[buffer(0)]],
        device const char    *scores   [[buffer(1)]],
        device       int32_t *selected [[buffer(2)]],
        threadgroup  float   *vals     [[threadgroup(0)]],
        threadgroup  int32_t *idxs     [[threadgroup(1)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    if (row >= args.n_tokens || args.n_comp == 0) return;

    device const float *score_row =
        (device const float *)(scores + (uint64_t)row * args.score_token_stride);

    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (uint i = tid; i < args.n_comp; i += nt) {
        const float v = score_row[i];
        if (v > best || (v == best && (int32_t)i < best_i)) {
            best = v;
            best_i = (int32_t)i;
        }
    }

    vals[tid] = best;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = vals[tid + stride];
            const int32_t other_i = idxs[tid + stride];
            if (other > vals[tid] || (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        selected[(uint64_t)row * args.selected_token_stride] = idxs[0];
    }
}

kernel void kernel_add_argmax_f32_i32(
        constant ds4_metal_args_add_argmax &args [[buffer(0)]],
        device       char    *scores   [[buffer(1)]],
        device const char    *add      [[buffer(2)]],
        device       int32_t *selected [[buffer(3)]],
        threadgroup  float   *vals     [[threadgroup(0)]],
        threadgroup  int32_t *idxs     [[threadgroup(1)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    if (row >= args.n_tokens || args.n_comp == 0) return;

    device float *score_row =
        (device float *)(scores + (uint64_t)row * args.score_token_stride);
    device const float *add_row =
        (device const float *)(add + (uint64_t)row * args.add_token_stride);

    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (uint i = tid; i < args.n_comp; i += nt) {
        const float v = score_row[i] + add_row[i];
        score_row[i] = v;
        if (v > best || (v == best && (int32_t)i < best_i)) {
            best = v;
            best_i = (int32_t)i;
        }
    }

    vals[tid] = best;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = vals[tid + stride];
            const int32_t other_i = idxs[tid + stride];
            if (other > vals[tid] || (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        selected[(uint64_t)row * args.selected_token_stride] = idxs[0];
    }
}

kernel void kernel_markov_f16_argmax_blocks(
        constant ds4_metal_args_markov_argmax &args [[buffer(0)]],
        device const half    *weights          [[buffer(1)]],
        device const float   *embedding        [[buffer(2)]],
        device const float   *base_logits      [[buffer(3)]],
        device       char    *scratch          [[buffer(4)]],
        threadgroup  float   *vals             [[threadgroup(0)]],
        threadgroup  int32_t *idxs             [[threadgroup(1)]],
        uint block [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]],
        uint nt    [[threads_per_threadgroup]]) {
    const uint vocab_i = block * nt + tid;
    float score = -INFINITY;
    int32_t score_i = INT_MAX;
    if (vocab_i < args.n_vocab) {
        device const half *row = weights + (uint64_t)vocab_i * args.rank;
        float sum = 0.0f;
        const uint rank4 = args.rank >> 2;
        device const half4 *row4 = (device const half4 *)row;
        device const float4 *emb4 = (device const float4 *)embedding;
        for (uint r4 = 0; r4 < rank4; r4++) {
            sum += dot(float4(row4[r4]), emb4[r4]);
        }
        for (uint r = rank4 << 2; r < args.rank; r++) {
            sum += float(row[r]) * embedding[r];
        }
        score = base_logits[vocab_i] + sum;
        score_i = (int32_t)vocab_i;
    }

    vals[tid] = score;
    idxs[tid] = score_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = vals[tid + stride];
            const int32_t other_i = idxs[tid + stride];
            if (other > vals[tid] ||
                (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0 && block < args.n_blocks) {
        device float *block_vals = (device float *)scratch;
        device int32_t *block_idxs =
            (device int32_t *)(scratch + (uint64_t)args.n_blocks * sizeof(float));
        block_vals[block] = vals[0];
        block_idxs[block] = idxs[0];
    }
}

kernel void kernel_markov_argmax_reduce(
        constant ds4_metal_args_markov_argmax &args [[buffer(0)]],
        device const char    *scratch          [[buffer(1)]],
        device       int32_t *selected         [[buffer(2)]],
        threadgroup  float   *vals             [[threadgroup(0)]],
        threadgroup  int32_t *idxs             [[threadgroup(1)]],
        uint tid [[thread_index_in_threadgroup]],
        uint nt  [[threads_per_threadgroup]]) {
    device const float *block_vals = (device const float *)scratch;
    device const int32_t *block_idxs =
        (device const int32_t *)(scratch + (uint64_t)args.n_blocks * sizeof(float));
    float best = -INFINITY;
    int32_t best_i = INT_MAX;
    for (uint block = tid; block < args.n_blocks; block += nt) {
        const float v = block_vals[block];
        const int32_t i = block_idxs[block];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = vals[tid + stride];
            const int32_t other_i = idxs[tid + stride];
            if (other > vals[tid] ||
                (other == vals[tid] && other_i < idxs[tid])) {
                vals[tid] = other;
                idxs[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) selected[0] = idxs[0];
}

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;

kernel void kernel_gather_topk_logits_f32(
        constant ds4_metal_args_topk_logits &args [[buffer(0)]],
        device const char    *scores [[buffer(1)]],
        device const int32_t *topk   [[buffer(2)]],
        device       float   *out    [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.top_k;
    if (gid >= total || args.top_k == 0 || args.n_comp == 0) return;
    const uint row = gid / args.top_k;
    const uint k = gid - row * args.top_k;
    const int32_t idx = topk[(uint64_t)row * args.topk_token_stride + k];
    if (idx < 0 || (uint32_t)idx >= args.n_comp) {
        out[gid] = -INFINITY;
        return;
    }
    device const float *score_row =
        (device const float *)(scores + (uint64_t)row * args.score_token_stride);
    out[gid] = score_row[idx];
}
