struct ds4_metal_args_glm52_q8_head_matvec {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_head;
    uint32_t in_head_stride;
    uint32_t out_head_stride;
    uint32_t input_offset;
    uint32_t n_tokens;
    uint32_t _pad1;
    uint64_t weight_head_stride;
    uint64_t weight_row_bytes;
};

struct ds4_metal_args_glm52_split_kv_batch {
    uint32_t n_tokens;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

struct ds4_metal_args_glm52_store_kv {
    uint32_t row;
    uint32_t ctx;
};

struct ds4_metal_args_glm52_store_kv_batch {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t ctx;
    uint32_t _pad0;
};

struct ds4_metal_args_glm52_attention_decode {
    uint32_t n_past;
    uint32_t ctx;
    uint32_t n_head;
    float scale;
};

struct ds4_metal_args_glm52_attention_prefill {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t ctx;
    uint32_t n_head;
    float scale;
    uint32_t score_stride;
    uint32_t _pad1;
    uint32_t _pad2;
};

kernel void kernel_glm52_q8_head_matvec(
        constant ds4_metal_args_glm52_q8_head_matvec & args [[buffer(0)]],
        device const char  * weights [[buffer(1)]],
        device const float * x       [[buffer(2)]],
        device       float * out     [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint32_t row = gid.x;
    const uint32_t head = gid.y;
    const uint32_t token = gid.z;
    if (row >= args.out_dim || head >= args.n_head ||
        token >= args.n_tokens || (args.in_dim % QK8_0) != 0) {
        return;
    }

    device const block_q8_0 * w =
        (device const block_q8_0 *)(weights +
                                    (uint64_t)head * args.weight_head_stride +
                                    (uint64_t)row * args.weight_row_bytes);
    device const float * xv =
        x + ((uint64_t)token * args.n_head + head) * args.in_head_stride +
        args.input_offset;

    const uint32_t nb = args.in_dim / QK8_0;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const float d = (float)w[b].d;
        for (uint32_t i = 0; i < QK8_0; i++) {
            sum += d * (float)w[b].qs[i] * xv[(uint64_t)b * QK8_0 + i];
        }
    }
    out[((uint64_t)token * args.n_head + head) * args.out_head_stride + row] = sum;
}

kernel void kernel_glm52_split_kv_batch(
        constant ds4_metal_args_glm52_split_kv_batch & args [[buffer(0)]],
        device const float * kv_raw  [[buffer(1)]],
        device       float * kv_lora [[buffer(2)]],
        device       float * k_pe    [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint32_t j = gid.x;
    const uint32_t token = gid.y;
    if (token >= args.n_tokens || j >= 576u) return;
    device const float * src = kv_raw + (uint64_t)token * 576u;
    if (j < 512u) {
        kv_lora[(uint64_t)token * 512u + j] = src[j];
    } else {
        k_pe[(uint64_t)token * 64u + (j - 512u)] = src[j];
    }
}

kernel void kernel_glm52_store_kv(
        constant ds4_metal_args_glm52_store_kv & args [[buffer(0)]],
        device const float * kv_lora [[buffer(1)]],
        device const float * k_pe    [[buffer(2)]],
        device       float * kv_cache [[buffer(3)]],
        device       float * kpe_cache [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (args.row >= args.ctx) return;
    if (gid < 512u) {
        kv_cache[(uint64_t)args.row * 512u + gid] = kv_lora[gid];
    } else if (gid < 576u) {
        const uint32_t j = gid - 512u;
        kpe_cache[(uint64_t)args.row * 64u + j] = k_pe[j];
    }
}

kernel void kernel_glm52_store_kv_batch(
        constant ds4_metal_args_glm52_store_kv_batch & args [[buffer(0)]],
        device const float * kv_lora [[buffer(1)]],
        device const float * k_pe    [[buffer(2)]],
        device       float * kv_cache [[buffer(3)]],
        device       float * kpe_cache [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint32_t j = gid.x;
    const uint32_t token = gid.y;
    if (token >= args.n_tokens || j >= 576u) return;
    const uint32_t row = args.pos0 + token;
    if (row >= args.ctx) return;
    if (j < 512u) {
        kv_cache[(uint64_t)row * 512u + j] = kv_lora[(uint64_t)token * 512u + j];
    } else {
        const uint32_t k = j - 512u;
        kpe_cache[(uint64_t)row * 64u + k] = k_pe[(uint64_t)token * 64u + k];
    }
}

kernel void kernel_glm52_attention_decode(
        constant ds4_metal_args_glm52_attention_decode & args [[buffer(0)]],
        device const float * q_abs     [[buffer(1)]],
        device const float * q_raw     [[buffer(2)]],
        device const float * kv_cache  [[buffer(3)]],
        device const float * kpe_cache [[buffer(4)]],
        device       float * scores    [[buffer(5)]],
        device       float * out       [[buffer(6)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (head >= args.n_head || args.n_past == 0 || args.n_past > args.ctx) {
        return;
    }

    threadgroup float reduce[256];
    device float * head_scores = scores + (uint64_t)head * args.ctx;
    float local_max = -INFINITY;
    device const float * qh = q_abs + (uint64_t)head * 512u;
    device const float * qpe = q_raw + (uint64_t)head * 256u + 192u;

    for (uint32_t p = tid; p < args.n_past; p += ntg) {
        device const float * kv = kv_cache + (uint64_t)p * 512u;
        device const float * kp = kpe_cache + (uint64_t)p * 64u;
        float s = 0.0f;
        for (uint32_t i = 0; i < 512u; i++) {
            s += qh[i] * kv[i];
        }
        for (uint32_t i = 0; i < 64u; i++) {
            s += qpe[i] * kp[i];
        }
        s *= args.scale;
        head_scores[p] = s;
        local_max = max(local_max, s);
    }

    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = max(reduce[tid], reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (uint32_t p = tid; p < args.n_past; p += ntg) {
        const float w = exp(head_scores[p] - max_score);
        head_scores[p] = w;
        local_sum += w;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;

    for (uint32_t d = tid; d < 512u; d += ntg) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < args.n_past; p++) {
            acc += (head_scores[p] * inv_sum) * kv_cache[(uint64_t)p * 512u + d];
        }
        out[(uint64_t)head * 512u + d] = acc;
    }
}

kernel void kernel_glm52_attention_prefill(
        constant ds4_metal_args_glm52_attention_prefill & args [[buffer(0)]],
        device const float * q_abs     [[buffer(1)]],
        device const float * q_raw     [[buffer(2)]],
        device const float * kv_cache  [[buffer(3)]],
        device const float * kpe_cache [[buffer(4)]],
        device       float * out       [[buffer(5)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint32_t head = tg.x;
    const uint32_t token = tg.y;
    const uint32_t tid = tid3.x;
    const uint32_t ntg = ntg3.x;
    if (head >= args.n_head || token >= args.n_tokens || args.ctx == 0) {
        return;
    }
    const uint32_t n_past = args.pos0 + token + 1u;
    if (n_past == 0 || n_past > args.ctx) return;

    threadgroup float reduce[256];
    device const float * qh = q_abs + ((uint64_t)token * args.n_head + head) * 512u;
    device const float * qpe = q_raw + ((uint64_t)token * args.n_head + head) * 256u + 192u;

    float local_max = -INFINITY;
    for (uint32_t p = tid; p < n_past; p += ntg) {
        device const float * kv = kv_cache + (uint64_t)p * 512u;
        device const float * kp = kpe_cache + (uint64_t)p * 64u;
        float s = 0.0f;
        for (uint32_t i = 0; i < 512u; i++) {
            s += qh[i] * kv[i];
        }
        for (uint32_t i = 0; i < 64u; i++) {
            s += qpe[i] * kp[i];
        }
        local_max = max(local_max, s * args.scale);
    }

    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = max(reduce[tid], reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (uint32_t p = tid; p < n_past; p += ntg) {
        device const float * kv = kv_cache + (uint64_t)p * 512u;
        device const float * kp = kpe_cache + (uint64_t)p * 64u;
        float s = 0.0f;
        for (uint32_t i = 0; i < 512u; i++) {
            s += qh[i] * kv[i];
        }
        for (uint32_t i = 0; i < 64u; i++) {
            s += qpe[i] * kp[i];
        }
        local_sum += exp(s * args.scale - max_score);
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;

    for (uint32_t d = tid; d < 512u; d += ntg) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < n_past; p++) {
            device const float * kv = kv_cache + (uint64_t)p * 512u;
            device const float * kp = kpe_cache + (uint64_t)p * 64u;
            float s = 0.0f;
            for (uint32_t i = 0; i < 512u; i++) {
                s += qh[i] * kv[i];
            }
            for (uint32_t i = 0; i < 64u; i++) {
                s += qpe[i] * kp[i];
            }
            acc += exp(s * args.scale - max_score) * inv_sum * kv[d];
        }
        out[((uint64_t)token * args.n_head + head) * 512u + d] = acc;
    }
}

kernel void kernel_glm52_attention_prefill_scores(
        constant ds4_metal_args_glm52_attention_prefill & args [[buffer(0)]],
        device const float * q_abs     [[buffer(1)]],
        device const float * q_raw     [[buffer(2)]],
        device const float * kv_cache  [[buffer(3)]],
        device const float * kpe_cache [[buffer(4)]],
        device       float * scores    [[buffer(5)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint32_t head = tg.x;
    const uint32_t token = tg.y;
    const uint32_t tid = tid3.x;
    const uint32_t ntg = ntg3.x;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.ctx == 0 || args.score_stride == 0) {
        return;
    }
    const uint32_t n_past = args.pos0 + token + 1u;
    if (n_past == 0 || n_past > args.ctx || n_past > args.score_stride) return;

    device const float * qh = q_abs + ((uint64_t)token * args.n_head + head) * 512u;
    device const float * qpe = q_raw + ((uint64_t)token * args.n_head + head) * 256u + 192u;
    device float * head_scores =
        scores + ((uint64_t)token * args.n_head + head) * args.score_stride;

    for (uint32_t p = tid; p < n_past; p += ntg) {
        device const float * kv = kv_cache + (uint64_t)p * 512u;
        device const float * kp = kpe_cache + (uint64_t)p * 64u;
        float s = 0.0f;
        for (uint32_t i = 0; i < 512u; i++) {
            s += qh[i] * kv[i];
        }
        for (uint32_t i = 0; i < 64u; i++) {
            s += qpe[i] * kp[i];
        }
        head_scores[p] = s * args.scale;
    }
}

kernel void kernel_glm52_attention_prefill_apply(
        constant ds4_metal_args_glm52_attention_prefill & args [[buffer(0)]],
        device const float * kv_cache [[buffer(1)]],
        device       float * scores   [[buffer(2)]],
        device       float * out      [[buffer(3)]],
        uint3 tg [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 ntg3 [[threads_per_threadgroup]]) {
    const uint32_t head = tg.x;
    const uint32_t token = tg.y;
    const uint32_t tid = tid3.x;
    const uint32_t ntg = ntg3.x;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.ctx == 0 || args.score_stride == 0) {
        return;
    }
    const uint32_t n_past = args.pos0 + token + 1u;
    if (n_past == 0 || n_past > args.ctx || n_past > args.score_stride) return;

    threadgroup float reduce[256];
    device float * head_scores =
        scores + ((uint64_t)token * args.n_head + head) * args.score_stride;

    float local_max = -INFINITY;
    for (uint32_t p = tid; p < n_past; p += ntg) {
        local_max = max(local_max, head_scores[p]);
    }
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = max(reduce[tid], reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (uint32_t p = tid; p < n_past; p += ntg) {
        const float w = exp(head_scores[p] - max_score);
        head_scores[p] = w;
        local_sum += w;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;

    for (uint32_t d = tid; d < 512u; d += ntg) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < n_past; p++) {
            acc += (head_scores[p] * inv_sum) * kv_cache[(uint64_t)p * 512u + d];
        }
        out[((uint64_t)token * args.n_head + head) * 512u + d] = acc;
    }
}
