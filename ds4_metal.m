#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/sysctl.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "ds4.h"
#include "ds4_gpu.h"
#include "moe-batch-bench/ane_ds4_mlp_int8w.h"

/*
 * Objective-C Metal glue for the C engine.
 *
 * The C code owns model semantics and graph scheduling.  This file owns only
 * Metal objects: device/queue/library setup, mmap-backed weight views, command
 * batching, persistent tensors, scratch buffers, and thin wrappers around the
 * kernel files in the metal directory.  Keeping this boundary narrow makes the
 * inference path readable from C while still using Objective-C where Metal
 * requires it.
 */

enum {
    DS4_METAL_TENSOR_Q2_K    = 10,
    DS4_METAL_TENSOR_Q4_K    = 12,
    DS4_METAL_TENSOR_IQ2_XXS = 16,
};

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static NSMutableArray<id<MTLCommandBuffer>> *g_pending_cbs;
static id<MTLComputePipelineState> g_set_rows_f32_i32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f32_pipeline;
static id<MTLComputePipelineState> g_get_rows_f16_pipeline;
static id<MTLComputePipelineState> g_get_rows_i32_pipeline;
static id<MTLComputePipelineState> g_repeat_f32_pipeline;
static id<MTLComputePipelineState> g_concat_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f32_pipeline;
static id<MTLComputePipelineState> g_cpy_f32_f16_pipeline;
static id<MTLComputePipelineState> g_cpy_f16_f32_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_add_pipeline;
static id<MTLComputePipelineState> g_mul_pipeline;
static id<MTLComputePipelineState> g_rms_norm_pipeline;
static id<MTLComputePipelineState> g_rms_norm_plain_pipeline;
static id<MTLComputePipelineState> g_dsv4_qkv_rms_norm_pipeline;
static id<MTLComputePipelineState> g_hc_split_sinkhorn_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_split_weighted_sum_norm_pipeline;
static id<MTLComputePipelineState> g_hc_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_hc_expand_pipeline;
static id<MTLComputePipelineState> g_unary_sigmoid_pipeline;
static id<MTLComputePipelineState> g_unary_silu_pipeline;
static id<MTLComputePipelineState> g_unary_softplus_pipeline;
static id<MTLComputePipelineState> g_unary_sqrt_pipeline;
static id<MTLComputePipelineState> g_unary_clamp_pipeline;
static id<MTLComputePipelineState> g_unary_scale_pipeline;
static id<MTLComputePipelineState> g_unary_fill_pipeline;
static id<MTLComputePipelineState> g_unary_fill_f16_pipeline;
static id<MTLComputePipelineState> g_bin_mul_scalar_pipeline;
static id<MTLComputePipelineState> g_bin_div_row_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q2_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mv_id_q4_k_sum6_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mm_id_iq2_xxs_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mm_id_q2_k_pipeline;
static id<MTLComputePipelineState> g_moe_mul_mm_id_q4_k_pipeline;
static id<MTLComputePipelineState> g_mpp_h_i8_f_pipeline;
static id<MTLComputePipelineState> g_mpp_f_i8_f_pipeline;
static id<MTLComputePipelineState> g_mpp_i8_i8_i32_pipeline;
static id<MTLComputePipelineState> g_mpp_swiglu_weight_f32_pipeline;
static id<MTLComputePipelineState> g_mpp_swiglu_i32_weight_f32_pipeline;
static id<MTLComputePipelineState> g_mpp_swiglu_i32_weight_i8_pipeline;
static id<MTLComputePipelineState> g_mpp_quant_f32_i8_pipeline;
static id<MTLComputePipelineState> g_mpp_quant_mid_f32_i8_pipeline;
static id<MTLComputePipelineState> g_mpp_scale_i32_f32_pipeline;
static id<MTLComputePipelineState> g_mpp_dequant_iq2_xxs_i8_pipeline;
static id<MTLComputePipelineState> g_mpp_dequant_q2_k_i8_pipeline;
/* Fused gate+up+down dequant in one dispatch; see metal/moe.metal. */
static id<MTLComputePipelineState> g_mpp_dequant_gud_i8_pipeline;
static id<MTLComputePipelineState> g_ane_dequant_iq2_xxs_f16_pipeline;
static id<MTLComputePipelineState> g_ane_dequant_q2_k_f16_pipeline;
static id<MTLComputePipelineState> g_ane_output_pack_f16_f32_pipeline;
static id<MTLComputePipelineState> g_rope_tail_batch_pipeline;
static id<MTLComputePipelineState> g_dsv4_fp8_kv_quantize_pipeline;
static id<MTLComputePipelineState> g_dsv4_kv_fp8_store_pipeline;
static id<MTLComputePipelineState> g_dsv4_ratio4_shift_pipeline;
static id<MTLComputePipelineState> g_dsv4_softmax_pool_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_pipeline;
static id<MTLComputePipelineState> g_soft_max_f32_4_pipeline;
static id<MTLComputePipelineState> g_argsort_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_argsort_merge_f32_i32_desc_pipeline;
static id<MTLComputePipelineState> g_sum_rows_f32_f32_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_pipeline;
static id<MTLComputePipelineState> g_dsv4_topk_mask_scatter_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_weighted_sum_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexer_score_one_direct_pipeline;
static id<MTLComputePipelineState> g_dsv4_compressor_store_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_sort_i32_rows_asc_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_pipeline;
static id<MTLComputePipelineState> g_dsv4_indexed_attention_heads8_rb4_pipeline;
static id<MTLComputePipelineState> g_dsv4_softplus_sqrt_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_finalize_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_router_weights_one_pipeline;
static id<MTLComputePipelineState> g_dsv4_hc_expand4_pipeline;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipeline_cache;
static NSMutableDictionary<NSString *, id<MTLBuffer>> *g_model_buffer_cache;
static NSMutableArray<id<MTLBuffer>> *g_transient_buffers;
static id g_model_residency_set;
static id<MTLBuffer> g_flash_attn_mask_buffer;
static id<MTLBuffer> g_flash_attn_pad_buffer;
static id<MTLBuffer> g_flash_attn_tmp_buffer;
static id<MTLBuffer> g_flash_attn_blk_buffer;
static id<MTLBuffer> g_flash_attn_ring_buffer;
static id<MTLBuffer> g_flash_attn_kv_buffer;
static id<MTLBuffer> g_compressor_pool_kv_buffer;
static id<MTLBuffer> g_compressor_pool_score_buffer;
static id<MTLBuffer> g_compressor_pool_score_cont_buffer;
static id<MTLBuffer> g_compressor_pool_softmax_buffer;
static id<MTLBuffer> g_compressor_pool_product_buffer;
static id<MTLBuffer> g_compressor_store_ape_buffer;
static id<MTLBuffer> g_compressor_store_score_buffer;
static id<MTLBuffer> g_embed_rows_buffer;
static id<MTLBuffer> g_router_selection_buffer;
static id<MTLBuffer> g_router_weight_sum_buffer;
static id<MTLBuffer> g_indexer_head_scores_buffer;
static id<MTLBuffer> g_indexer_topk_buffer;
static id<MTLBuffer> g_indexed_topk_buffer;
static id<MTLBuffer> g_f16_round_scratch_buffer;
static id<MTLBuffer> g_raw_store_round_buffer;
static id<MTLBuffer> g_moe_gate_scratch_buffer;
static id<MTLBuffer> g_moe_down_scratch_buffer;
static id<MTLBuffer> g_moe_id_map_buffer;
static id<MTLBuffer> g_mpp_prefill_x_half_buffer;
static id<MTLBuffer> g_mpp_prefill_x_i8_buffer;
static id<MTLBuffer> g_mpp_prefill_gate_i8_buffer;
static id<MTLBuffer> g_mpp_prefill_up_i8_buffer;
static id<MTLBuffer> g_mpp_prefill_down_i8_buffer;
static id<MTLBuffer> g_mpp_prefill_gate_i32_buffer;
static id<MTLBuffer> g_mpp_prefill_up_i32_buffer;
static id<MTLBuffer> g_mpp_prefill_mid_i8_buffer;
static id<MTLBuffer> g_mpp_prefill_out_i32_buffer;
static id<MTLBuffer> g_ane_prefill_gate_i8_buffer;
static id<MTLBuffer> g_ane_prefill_up_i8_buffer;
static id<MTLBuffer> g_ane_prefill_down_i8_buffer;
static id<MTLBuffer> g_ane_prefill_x_i8_buffer;
static id<MTLBuffer> g_ane_prefill_gate_f16_buffer;
static id<MTLBuffer> g_ane_prefill_up_f16_buffer;
static id<MTLBuffer> g_ane_prefill_down_f16_buffer;

/* Ping-pong dequant scratch pool — enables multiple in-flight ANE prefill
 * calls without races on the global scratch.  Each ANE call acquires a free
 * slot, encodes its weight+x dequant into that slot's MTLBuffers, and
 * releases the slot once its ANE worker(s) have fully consumed it (i.e.
 * after the eval's write_surface memcpy in ane_ds4_mlp_int8w.m).  Sized to
 * cover OUTPUT_QUEUE depth + active job; defaults match max OUTPUT_QUEUE=8
 * plus a couple of headroom slots. */
#define DS4_ANE_DEQUANT_SLOTS 12
typedef struct {
    __strong id<MTLBuffer> gate_i8;
    __strong id<MTLBuffer> up_i8;
    __strong id<MTLBuffer> down_i8;
    __strong id<MTLBuffer> x_i8;
    NSUInteger gate_i8_bytes;
    NSUInteger up_i8_bytes;
    NSUInteger down_i8_bytes;
    NSUInteger x_i8_bytes;
    /* Output buffers reused across calls so we don't pay newBufferWithLength:
     * per ANE prefill call (~500ms-2s total at the production workload).  Sized
     * to the largest seen n_tokens for this slot; capacity-grown via
     * ds4_gpu_ensure_scratch_buffer. */
    __strong id<MTLBuffer> out_f16_all;
    __strong id<MTLBuffer> out_f32;
    __strong id<MTLBuffer> route_weights;
    NSUInteger out_f16_all_bytes;
    NSUInteger out_f32_bytes;
    NSUInteger route_weights_bytes;
    int in_use;
} ds4_ane_dequant_slot_t;
static ds4_ane_dequant_slot_t g_ane_dequant_slots[DS4_ANE_DEQUANT_SLOTS];
static pthread_mutex_t g_ane_dequant_slot_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_ane_dequant_slot_cv = PTHREAD_COND_INITIALIZER;
static id<MTLBuffer> g_attn_out_group_ids_buffer;
static const void *g_model_map_ptr;
static uint64_t g_model_map_size;
static uint64_t g_model_mapped_offset;
static uint64_t g_model_mapped_size;
static uint64_t g_tensor_alloc_live_bytes;
static uint64_t g_tensor_alloc_peak_bytes;
static uint64_t g_model_wrap_count;
static uint64_t g_model_wrap_bytes;
static uint64_t g_model_wrap_max_bytes;
static uint64_t g_model_residency_count;
static NSUInteger g_flash_attn_mask_bytes;
static NSUInteger g_flash_attn_pad_bytes;
static NSUInteger g_flash_attn_tmp_bytes;
static NSUInteger g_flash_attn_blk_bytes;
static NSUInteger g_flash_attn_ring_bytes;
static NSUInteger g_flash_attn_kv_bytes;
static NSUInteger g_compressor_pool_kv_bytes;
static NSUInteger g_compressor_pool_score_bytes;
static NSUInteger g_compressor_pool_score_cont_bytes;
static NSUInteger g_compressor_pool_softmax_bytes;
static NSUInteger g_compressor_pool_product_bytes;
static NSUInteger g_compressor_store_ape_bytes;
static NSUInteger g_compressor_store_score_bytes;
static NSUInteger g_embed_rows_bytes;
static NSUInteger g_router_selection_bytes;
static NSUInteger g_router_weight_sum_bytes;
static NSUInteger g_indexer_head_scores_bytes;
static NSUInteger g_indexer_topk_bytes;
static NSUInteger g_indexed_topk_bytes;
static NSUInteger g_f16_round_scratch_bytes;
static NSUInteger g_raw_store_round_bytes;
static NSUInteger g_moe_gate_scratch_bytes;
static NSUInteger g_moe_down_scratch_bytes;
static NSUInteger g_moe_id_map_bytes;
static NSUInteger g_mpp_prefill_x_half_bytes;
static NSUInteger g_mpp_prefill_x_i8_bytes;
static NSUInteger g_mpp_prefill_gate_i8_bytes;
static NSUInteger g_mpp_prefill_up_i8_bytes;
static NSUInteger g_mpp_prefill_down_i8_bytes;
static NSUInteger g_mpp_prefill_gate_i32_bytes;
static NSUInteger g_mpp_prefill_up_i32_bytes;
static NSUInteger g_mpp_prefill_mid_i8_bytes;
static NSUInteger g_mpp_prefill_out_i32_bytes;
static NSUInteger g_ane_prefill_gate_i8_bytes;
static NSUInteger g_ane_prefill_up_i8_bytes;
static NSUInteger g_ane_prefill_down_i8_bytes;
static NSUInteger g_ane_prefill_x_i8_bytes;
static NSUInteger g_ane_prefill_gate_f16_bytes;
static NSUInteger g_ane_prefill_up_f16_bytes;
static NSUInteger g_ane_prefill_down_f16_bytes;
static NSUInteger g_attn_out_group_ids_bytes;
static int g_initialized;
static int g_quality_mode;
static ds4_ane_mlp_int8w_ctx *g_ane_int8w_ctx;
#define DS4_ANE_CTX_CACHE_MAX 12
typedef struct {
    ds4_ane_mlp_int8w_ctx *ctx;
    int mode;
    int H;
    int I;
    int B;
    float w_scale;
    float x_scale;
    float mid_scale;
} ds4_ane_ctx_cache_entry;
static ds4_ane_ctx_cache_entry g_ane_ctx_cache[DS4_ANE_CTX_CACHE_MAX];
/* Parallel cache for the second cluster's contexts.  Same shape as ctx_a but
 * a distinct context handle — required because the primary cache dedups by
 * shape and would hand back ctx_a a second time.  Used only when
 * DS4_FLASH_MOE_ANE_DUAL=1 on the tiled_fused path. */
static ds4_ane_ctx_cache_entry g_ane_ctx_cache_b[DS4_ANE_CTX_CACHE_MAX];
/* Optional 3rd/4th worker context caches.  M3 Ultra physically has two ANE
 * clusters (A,B); a 3rd/4th context will be time-shared onto one of them by
 * the OS scheduler.  Gated behind DS4_FLASH_MOE_ANE_THREADS={3,4}. */
static ds4_ane_ctx_cache_entry g_ane_ctx_cache_c[DS4_ANE_CTX_CACHE_MAX];
static ds4_ane_ctx_cache_entry g_ane_ctx_cache_d[DS4_ANE_CTX_CACHE_MAX];
static uint64_t g_ane_prefill_calls;
static uint64_t g_ane_prefill_successes;
static uint64_t g_ane_prefill_skip_big_refs;
static uint64_t g_ane_prefill_compile_attempts;
static uint64_t g_ane_prefill_compile_failures;
static uint64_t g_ane_prefill_eval_calls;
static uint64_t g_ane_prefill_ane_evaluate_calls;
static uint64_t g_ane_prefill_eval_refs;
static uint64_t g_ane_prefill_padded_refs;
static uint64_t g_ane_prefill_chunks_le16;
static uint64_t g_ane_prefill_chunks_le64;
static uint64_t g_ane_prefill_chunks_le128;
static uint64_t g_ane_prefill_chunks_full;
#define DS4_ANE_BATCH_HIST_N 11
static const uint32_t g_ane_batch_hist_values[DS4_ANE_BATCH_HIST_N] =
    {8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 0u};
static uint64_t g_ane_prefill_batch_eval_calls[DS4_ANE_BATCH_HIST_N];
static uint64_t g_ane_prefill_batch_refs[DS4_ANE_BATCH_HIST_N];
static uint64_t g_ane_prefill_batch_padded_refs[DS4_ANE_BATCH_HIST_N];
static uint64_t g_ane_prefill_eval_failures;
static uint64_t g_ane_prefill_write_failures;
static uint64_t g_ane_prefill_reopen_failures;
static uint64_t g_ane_prefill_x_values;
static uint64_t g_ane_prefill_x_saturated;
static float g_ane_prefill_x_abs_max;
static double g_ane_prefill_dequant_ms;
static double g_ane_prefill_dequant_wait_ms;
static double g_ane_prefill_input_ms;
static double g_ane_prefill_x_zero_ms;
static double g_ane_prefill_x_copy_ms;
static double g_ane_prefill_out_zero_ms;
static double g_ane_prefill_eval_ms;
/* Cluster-B accumulated eval ms — only populated when DS4_FLASH_MOE_ANE_DUAL=1.
 * When dual is active, g_ane_prefill_eval_ms holds the cluster-A total only;
 * read these two together to estimate ANE wall = max(A, B) ≈ A under balanced
 * stride-2 split (so wall ≈ (A+B)/2, ~halving baseline). */
static double g_ane_prefill_eval_ms_b;
/* C/D worker eval ms — populated only when DS4_FLASH_MOE_ANE_THREADS={3,4}.
 * ANE wall ≈ max(A,B,C,D). */
static double g_ane_prefill_eval_ms_c;
static double g_ane_prefill_eval_ms_d;
static double g_ane_prefill_output_ms;
static double g_ane_prefill_output_f16_copy_ms;
static double g_ane_prefill_output_convert_ms;
static double g_ane_prefill_output_gpu_encode_ms;
static double g_ane_prefill_post_wait_ms;
static double g_ane_prefill_writeback_ms;
static double g_ane_prefill_wait_predict_ms;
static double g_ane_prefill_finish_join_ane_ms;
static double g_ane_prefill_finish_join_post_ms;
static uint64_t g_ane_prefill_wait_predict_calls;
static uint64_t g_ane_prefill_finish_calls;
static uint64_t g_ane_prefill_output_elems;
static uint64_t g_ane_prefill_output_chunks;
static int g_model_request_residency = 1;
static int g_model_warm_views = 1;

static void ds4_gpu_ane_ctx_cache_clear(void);

static int ds4_gpu_ane_debug_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_DEBUG");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_stats_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_STATS");
    return (env && env[0] && atoi(env) != 0) || ds4_gpu_ane_debug_enabled();
}

static void ds4_gpu_ane_record_batch_stats(uint32_t batch,
                                           uint64_t eval_calls,
                                           uint64_t refs,
                                           uint64_t padded_refs) {
    uint32_t idx = DS4_ANE_BATCH_HIST_N - 1u;
    for (uint32_t i = 0; i + 1u < DS4_ANE_BATCH_HIST_N; i++) {
        if (g_ane_batch_hist_values[i] == batch) {
            idx = i;
            break;
        }
    }
    g_ane_prefill_batch_eval_calls[idx] += eval_calls;
    g_ane_prefill_batch_refs[idx] += refs;
    g_ane_prefill_batch_padded_refs[idx] += padded_refs;
}

static void ds4_gpu_print_ane_prefill_stats(void) {
    if (!ds4_gpu_ane_stats_enabled() || g_ane_prefill_calls == 0) return;
    fprintf(stderr,
            "ds4: ANE prefill stats calls=%llu ok=%llu skipped_big_refs=%llu compile_attempts=%llu compile_failures=%llu eval_calls=%llu eval_failures=%llu write_failures=%llu reopen_failures=%llu\n",
            (unsigned long long)g_ane_prefill_calls,
            (unsigned long long)g_ane_prefill_successes,
            (unsigned long long)g_ane_prefill_skip_big_refs,
            (unsigned long long)g_ane_prefill_compile_attempts,
            (unsigned long long)g_ane_prefill_compile_failures,
            (unsigned long long)g_ane_prefill_eval_calls,
            (unsigned long long)g_ane_prefill_eval_failures,
            (unsigned long long)g_ane_prefill_write_failures,
            (unsigned long long)g_ane_prefill_reopen_failures);
    if (g_ane_prefill_x_values != 0) {
        fprintf(stderr,
                "ds4: ANE quant stats x_values=%llu x_saturated=%llu x_sat_rate=%.6f%% x_abs_max=%.6g\n",
                (unsigned long long)g_ane_prefill_x_values,
                (unsigned long long)g_ane_prefill_x_saturated,
                100.0 * (double)g_ane_prefill_x_saturated / (double)g_ane_prefill_x_values,
                g_ane_prefill_x_abs_max);
    }
    uint64_t hidden_values = 0;
    uint64_t hidden_saturated = 0;
    float hidden_abs_max = 0.0f;
    ds4_ane_mlp_int8w_quant_stats(&hidden_values, &hidden_saturated, &hidden_abs_max);
    if (hidden_values != 0) {
        fprintf(stderr,
                "ds4: ANE i8i8 hidden quant stats hidden_values=%llu hidden_saturated=%llu hidden_sat_rate=%.6f%% hidden_abs_max=%.6g\n",
                (unsigned long long)hidden_values,
                (unsigned long long)hidden_saturated,
                100.0 * (double)hidden_saturated / (double)hidden_values,
                hidden_abs_max);
    }
    if (g_ane_prefill_eval_calls != 0 || g_ane_prefill_dequant_ms != 0.0) {
        const double calls = g_ane_prefill_eval_calls ? (double)g_ane_prefill_eval_calls : 1.0;
        const double ane_calls = g_ane_prefill_ane_evaluate_calls ? (double)g_ane_prefill_ane_evaluate_calls : calls;
        const double pad_util = g_ane_prefill_padded_refs ?
            100.0 * (double)g_ane_prefill_eval_refs / (double)g_ane_prefill_padded_refs : 0.0;
        const double eval_total = g_ane_prefill_eval_ms + g_ane_prefill_eval_ms_b +
                                   g_ane_prefill_eval_ms_c + g_ane_prefill_eval_ms_d;
        /* When dual/multi is active eval_total = A+B(+C+D); ANE wall ≈ max(A..).
         * When solo, eval_ms_{b,c,d} are 0 and ane_wall_est == eval_ms. */
        double ane_wall_est = g_ane_prefill_eval_ms;
        if (g_ane_prefill_eval_ms_b > ane_wall_est) ane_wall_est = g_ane_prefill_eval_ms_b;
        if (g_ane_prefill_eval_ms_c > ane_wall_est) ane_wall_est = g_ane_prefill_eval_ms_c;
        if (g_ane_prefill_eval_ms_d > ane_wall_est) ane_wall_est = g_ane_prefill_eval_ms_d;
        fprintf(stderr,
                "ds4: ANE prefill timing dequant=%.3f ms input=%.3f ms ane_eval=%.3f ms ane_eval_a=%.3f ms ane_eval_b=%.3f ms ane_eval_c=%.3f ms ane_eval_d=%.3f ms ane_wall_est=%.3f ms writeback=%.3f ms helper_avg=%.6f ms ane_calls=%llu ane_call_avg=%.6f ms\n",
                g_ane_prefill_dequant_ms,
                g_ane_prefill_input_ms,
                eval_total,
                g_ane_prefill_eval_ms,
                g_ane_prefill_eval_ms_b,
                g_ane_prefill_eval_ms_c,
                g_ane_prefill_eval_ms_d,
                ane_wall_est,
                g_ane_prefill_writeback_ms,
                eval_total / calls,
                (unsigned long long)g_ane_prefill_ane_evaluate_calls,
                eval_total / ane_calls);
        fprintf(stderr,
                "ds4: ANE prefill timing2 output_pack=%.3f ms measured_total=%.3f ms\n",
                g_ane_prefill_output_ms,
                g_ane_prefill_dequant_ms + g_ane_prefill_input_ms +
                    ane_wall_est + g_ane_prefill_output_ms +
                    g_ane_prefill_writeback_ms);
        if (g_ane_prefill_output_elems != 0 ||
            g_ane_prefill_dequant_wait_ms != 0.0 ||
            g_ane_prefill_post_wait_ms != 0.0) {
            const double gelem = (double)g_ane_prefill_output_elems / 1000000000.0;
            const double convert_rate = g_ane_prefill_output_convert_ms > 0.0 ?
                gelem / (g_ane_prefill_output_convert_ms / 1000.0) : 0.0;
            fprintf(stderr,
                    "ds4: ANE prefill timing3 dequant_wait=%.3f ms post_wait=%.3f ms x_zero=%.3f ms x_copy=%.3f ms out_zero=%.3f ms out_f16_copy=%.3f ms output_convert=%.3f ms output_gpu_encode=%.3f ms output_chunks=%llu output_elems=%llu output_convert_rate=%.3f Gelem/s\n",
                    g_ane_prefill_dequant_wait_ms,
                    g_ane_prefill_post_wait_ms,
                    g_ane_prefill_x_zero_ms,
                    g_ane_prefill_x_copy_ms,
                    g_ane_prefill_out_zero_ms,
                    g_ane_prefill_output_f16_copy_ms,
                    g_ane_prefill_output_convert_ms,
                    g_ane_prefill_output_gpu_encode_ms,
                    (unsigned long long)g_ane_prefill_output_chunks,
                    (unsigned long long)g_ane_prefill_output_elems,
                    convert_rate);
        }
        if (g_ane_prefill_wait_predict_ms != 0.0 ||
            g_ane_prefill_finish_join_ane_ms != 0.0 ||
            g_ane_prefill_finish_join_post_ms != 0.0) {
            fprintf(stderr,
                    "ds4: ANE prefill timing4 wait_predict=%.3f ms finish_join_ane=%.3f ms finish_join_post=%.3f ms wait_predict_calls=%llu finish_calls=%llu\n",
                    g_ane_prefill_wait_predict_ms,
                    g_ane_prefill_finish_join_ane_ms,
                    g_ane_prefill_finish_join_post_ms,
                    (unsigned long long)g_ane_prefill_wait_predict_calls,
                    (unsigned long long)g_ane_prefill_finish_calls);
        }
        fprintf(stderr,
                "ds4: ANE prefill chunks refs=%llu padded_refs=%llu pad_util=%.2f%% le16=%llu le64=%llu le128=%llu full=%llu\n",
                (unsigned long long)g_ane_prefill_eval_refs,
                (unsigned long long)g_ane_prefill_padded_refs,
                pad_util,
                (unsigned long long)g_ane_prefill_chunks_le16,
                (unsigned long long)g_ane_prefill_chunks_le64,
                (unsigned long long)g_ane_prefill_chunks_le128,
                (unsigned long long)g_ane_prefill_chunks_full);
        fprintf(stderr, "ds4: ANE prefill batch_hist");
        for (uint32_t i = 0; i < DS4_ANE_BATCH_HIST_N; i++) {
            const uint64_t evals = g_ane_prefill_batch_eval_calls[i];
            if (!evals) continue;
            const double util = g_ane_prefill_batch_padded_refs[i] ?
                100.0 * (double)g_ane_prefill_batch_refs[i] /
                    (double)g_ane_prefill_batch_padded_refs[i] : 0.0;
            if (g_ane_batch_hist_values[i]) {
                fprintf(stderr,
                        " B%u=%llu refs=%llu util=%.1f%%",
                        g_ane_batch_hist_values[i],
                        (unsigned long long)evals,
                        (unsigned long long)g_ane_prefill_batch_refs[i],
                        util);
            } else {
                fprintf(stderr,
                        " Bother=%llu refs=%llu util=%.1f%%",
                        (unsigned long long)evals,
                        (unsigned long long)g_ane_prefill_batch_refs[i],
                        util);
            }
        }
        fprintf(stderr, "\n");
    }
}

static uint64_t ds4_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void ds4_gpu_print_device_summary(void) {
    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown Metal device";
    uint64_t mem = ds4_gpu_system_memory_bytes();
    if (mem) {
        double gib = (double)mem / 1024.0 / 1024.0 / 1024.0;
        fprintf(stderr, "ds4: Metal device %s, %.2f GiB RAM\n", name, gib);
    } else {
        fprintf(stderr, "ds4: Metal device %s\n", name);
    }
}

static int ds4_gpu_is_m5_device(void) {
    static int initialized;
    static int is_m5;
    if (!initialized) {
        const char *device_name = g_device.name ? [g_device.name UTF8String] : "";
        is_m5 = strstr(device_name, "M5") != NULL;
        initialized = 1;
    }
    return is_m5;
}

static int ds4_gpu_use_m5_private_scratch(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_DISABLE_M5_PRIVATE_SCRATCH") == NULL && ds4_gpu_is_m5_device();
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_scratch_needs_cpu_access(const char *label) {
    if (!label) return 0;
    return strstr(label, "mask") != NULL ||
           strstr(label, "ane") != NULL ||
           strcmp(label, "ds4_attention_output_group_ids") == 0;
}

#define DS4_METAL_MAX_MODEL_VIEWS 16
#define DS4_METAL_MODEL_MAX_TENSOR_BYTES 704643072ull

typedef struct {
    __strong id<MTLBuffer> buffer;
    const void *model_map;
    uint64_t model_size;
    uint64_t model_offset;
    uint64_t bytes;
} ds4_gpu_model_view;

static ds4_gpu_model_view g_model_views[DS4_METAL_MAX_MODEL_VIEWS];
static uint32_t g_model_view_count;

@interface DS4MetalTensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) uint64_t offset;
@property(nonatomic, assign) uint64_t bytes;
@property(nonatomic, assign) uint8_t owner;
@end

@implementation DS4MetalTensor
@end

static DS4MetalTensor *ds4_gpu_tensor_obj(ds4_gpu_tensor *tensor) {
    return (__bridge DS4MetalTensor *)tensor;
}

static const DS4MetalTensor *ds4_gpu_tensor_const_obj(const ds4_gpu_tensor *tensor) {
    return (__bridge const DS4MetalTensor *)tensor;
}

static id<MTLBuffer> ds4_gpu_tensor_buffer(const ds4_gpu_tensor *tensor) {
    if (!tensor) return nil;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.buffer;
}

static NSUInteger ds4_gpu_tensor_offset(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return (NSUInteger)obj.offset;
}

static id<MTLCommandBuffer> ds4_gpu_command_buffer(int *owned) {
    if (g_batch_cb) {
        *owned = 0;
        return g_batch_cb;
    }
    *owned = 1;
    return [g_queue commandBuffer];
}

static id<MTLComputeCommandEncoder> ds4_gpu_compute_encoder(id<MTLCommandBuffer> cb) {
    if (g_batch_cb && cb == g_batch_cb) {
        if (!g_batch_enc) g_batch_enc = [cb computeCommandEncoder];
        return g_batch_enc;
    }
    return [cb computeCommandEncoder];
}

static void ds4_gpu_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    if (!enc) return;
    if (g_batch_cb && cb == g_batch_cb && enc == g_batch_enc) return;
    [enc endEncoding];
}

static void ds4_gpu_close_batch_encoder(void) {
    if (!g_batch_enc) return;
    [g_batch_enc endEncoding];
    g_batch_enc = nil;
}

static int ds4_gpu_wait_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal %s failed: %s\n",
                label, [[cb.error localizedDescription] UTF8String]);
        return 0;
    }
    return 1;
}

static int ds4_gpu_wait_pending_command_buffers(const char *label) {
    int ok = 1;
    for (id<MTLCommandBuffer> pending in g_pending_cbs) {
        if (!ds4_gpu_wait_command_buffer(pending, label)) ok = 0;
    }
    [g_pending_cbs removeAllObjects];
    return ok;
}

static int ds4_gpu_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label) {
    if (!owned) return 1;

    [cb commit];
    int ok = ds4_gpu_wait_pending_command_buffers(label);
    if (!ds4_gpu_wait_command_buffer(cb, label)) ok = 0;
    [g_transient_buffers removeAllObjects];
    return ok;
}

static int ds4_gpu_ensure_scratch_buffer(
        id<MTLBuffer> __strong *buffer,
        NSUInteger    *capacity,
        NSUInteger     bytes,
        const char    *label) {
    if (*buffer && *capacity >= bytes) return 1;
    if (bytes == 0) bytes = 1;
    if (bytes > NSUIntegerMax) return 0;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (ds4_gpu_use_m5_private_scratch() && !ds4_gpu_scratch_needs_cpu_access(label)) {
        /*
         * Keep Metal's default hazard tracking. These scratch buffers are
         * reused by dependent kernels across many compute encoders, and the
         * graph does not insert explicit fences for untracked resources.
         */
        options = MTLResourceStorageModePrivate;
    }

    *buffer = [g_device newBufferWithLength:bytes options:options];
    if (!*buffer && options != MTLResourceStorageModeShared) {
        *buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    if (!*buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal scratch buffer %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        *capacity = 0;
        return 0;
    }
    (*buffer).label = [NSString stringWithUTF8String:label];
    *capacity = bytes;
    return 1;
}

static uint64_t round_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(const char *function_name);
static int ds4_gpu_warm_model_views(void);

static double ds4_gpu_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void ds4_gpu_prefill_trace_interval(const char *name,
                                           double      t0_ms,
                                           double      t1_ms) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    static int fd = -2;
    if (!name || t1_ms < t0_ms) return;
    pthread_mutex_lock(&mu);
    if (fd == -2) {
        const char *path = getenv("DS4_PREFILL_TRACE_CSV");
        fd = (path && path[0]) ? open(path, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1;
    }
    if (fd >= 0) {
        char line[256];
        int n = snprintf(line, sizeof(line), "%.6f,%.6f,%s,0,-1\n",
                         t0_ms, t1_ms, name);
        if (n > 0 && n < (int)sizeof(line)) {
            (void)write(fd, line, (size_t)n);
        }
    }
    pthread_mutex_unlock(&mu);
}

static int ds4_gpu_progress_enabled(void) {
    return ds4_log_is_tty(stderr);
}

static void ds4_gpu_progress_begin(const char *what) {
    if (!ds4_gpu_progress_enabled()) return;
    fprintf(stderr, "ds4: %s...", what);
    fflush(stderr);
}

static void ds4_gpu_progress_done(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" done\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_progress_failed(void) {
    if (!ds4_gpu_progress_enabled()) return;
    fputs(" failed\n", stderr);
    fflush(stderr);
}

static void ds4_gpu_model_views_clear(void) {
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        g_model_views[i].buffer = nil;
        g_model_views[i].model_map = NULL;
        g_model_views[i].model_size = 0;
        g_model_views[i].model_offset = 0;
        g_model_views[i].bytes = 0;
    }
    g_model_view_count = 0;
}

static void ds4_gpu_model_residency_clear(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (g_model_residency_set) {
            [g_model_residency_set endResidency];
            [g_model_residency_set removeAllAllocations];
            g_model_residency_set = nil;
        }
    }
#endif
    g_model_residency_count = 0;
}

static int ds4_gpu_model_residency_request_views(void) {
    if (g_model_view_count == 0 || getenv("DS4_METAL_NO_RESIDENCY") != NULL) return 1;

#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        /*
         * Register all model views as one residency set before inference. This
         * is a GPU residency/budgeting hint, not a request to fault the whole
         * 80+ GB file into memory. Its purpose is to make the driver see the
         * complete set of large shared allocations during setup instead of
         * discovering them lazily from the first measured graph command, where
         * VM validation and residency accounting would look like model compute.
         */
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_model";
        desc.initialCapacity = g_model_view_count;

        NSError *error = nil;
        g_model_residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!g_model_residency_set) {
            fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            [g_model_residency_set addAllocation:g_model_views[i].buffer];
        }
        [g_model_residency_set commit];
        [g_model_residency_set requestResidency];
        g_model_residency_count = g_model_view_count;
    }
#endif

    return 1;
}

static int ds4_gpu_map_model_views(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    map_offset,
        uint64_t    map_size) {
    const double t0 = ds4_gpu_now_ms();
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t model_addr = (uintptr_t)model_map;

    if ((model_addr & (uintptr_t)(page - 1)) != 0) {
        fprintf(stderr, "ds4: Metal model mmap base is not page aligned\n");
        return 0;
    }
    if (map_offset > model_size || map_size > model_size - map_offset) {
        fprintf(stderr, "ds4: Metal model mapped range is outside the GGUF mapping\n");
        return 0;
    }

    const uint64_t page_model_offset = map_offset & ~(page - 1);
    const uint64_t leading = map_offset - page_model_offset;
    const uint64_t mapped_model_size = round_up_u64(leading + map_size, page);
    uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
    max_buffer &= ~(page - 1);

    /*
     * Wrap only the tensor-data part of the GGUF file. Metadata is parsed by the
     * CPU and is never dereferenced by kernels, so exposing it to Metal only
     * grows the residency set and the VM range the driver must validate.
     *
     * Metal buffers have a device-specific maximum length, and this model is
     * larger than that maximum on the target machines. Creating one no-copy
     * buffer per tensor would avoid the length limit, but it would also move a
     * lot of VM-object creation and residency bookkeeping into graph setup. The
     * stable shape here is a tiny number of page-aligned views created once.
     *
     * Adjacent views intentionally overlap by more than the largest tensor, plus
     * one page for alignment. That invariant guarantees every tensor lies wholly
     * inside at least one view, so hot paths pass one buffer and one inner byte
     * offset. We never split a weight tensor across command encoders.
     */
    const uint64_t overlap = round_up_u64(DS4_METAL_MODEL_MAX_TENSOR_BYTES, page) + page;
    if (max_buffer == 0 || max_buffer <= overlap) {
        fprintf(stderr, "ds4: Metal maxBufferLength is too small for DS4 model views\n");
        return 0;
    }

    const uint64_t step = max_buffer - overlap;
    uint64_t off = 0;
    while (off < mapped_model_size) {
        if (g_model_view_count == DS4_METAL_MAX_MODEL_VIEWS) {
            fprintf(stderr, "ds4: Metal model needs more mapped views than expected\n");
            return 0;
        }

        uint64_t view_bytes = mapped_model_size - off;
        if (view_bytes > max_buffer) view_bytes = max_buffer;

        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)(model_addr + page_model_offset + off)
                                                           length:(NSUInteger)view_bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap mmaped model view at %.2f GiB, size %.2f GiB\n",
                    (double)off / (1024.0 * 1024.0 * 1024.0),
                    (double)view_bytes / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", g_model_view_count];

        g_model_views[g_model_view_count].buffer = buffer;
        g_model_views[g_model_view_count].model_map = model_map;
        g_model_views[g_model_view_count].model_size = model_size;
        g_model_views[g_model_view_count].model_offset = page_model_offset + off;
        g_model_views[g_model_view_count].bytes = view_bytes;
        g_model_view_count++;

        g_model_wrap_count++;
        g_model_wrap_bytes += view_bytes;
        if (view_bytes > g_model_wrap_max_bytes) g_model_wrap_max_bytes = view_bytes;

        if (off + view_bytes >= mapped_model_size) break;
        off += step;
    }

    const double t_mapped = ds4_gpu_now_ms();
    const int request_residency = g_model_request_residency &&
                                  getenv("DS4_METAL_NO_RESIDENCY") == NULL;
    if (request_residency) ds4_gpu_progress_begin("requesting Metal residency (may take tens of seconds)");
    if (request_residency && !ds4_gpu_model_residency_request_views()) {
        if (request_residency) ds4_gpu_progress_failed();
        return 0;
    }
    if (request_residency) ds4_gpu_progress_done();
    const double t_resident = ds4_gpu_now_ms();
    int warmed = 1;
    const double t_warm0 = ds4_gpu_now_ms();
    const int warm_model_views = request_residency &&
                                 g_model_warm_views &&
                                 getenv("DS4_METAL_NO_MODEL_WARMUP") == NULL;
    if (warm_model_views) {
        /*
         * The first GPU command touching no-copy mmap storage can pay command
         * queue setup, page-table validation, and shared-allocation residency
         * costs. Sample each model view here so timed graph execution starts
         * after that one-time work. The stride is intentionally coarse: this is
         * a validation touch over the VM ranges, not a full model prefetch. A
         * dense prefetch would create exactly the kind of memory pressure and
         * startup stalls this path is designed to avoid.
         */
        ds4_gpu_progress_begin("warming Metal model views");
        warmed = ds4_gpu_warm_model_views();
        if (warmed) ds4_gpu_progress_done();
        else ds4_gpu_progress_failed();
    }
    const double t_warm = ds4_gpu_now_ms();
    double map_ms = t_mapped - t0;
    double residency_ms = t_resident - t_mapped;
    double warm_ms = t_warm - t_warm0;
    if (map_ms < 0.0) map_ms = 0.0;
    if (residency_ms < 0.0) residency_ms = 0.0;
    if (warm_ms < 0.0) warm_ms = 0.0;
    fprintf(stderr,
            "ds4: Metal model views created in %.3f ms, residency requested in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)\n",
            map_ms,
            residency_ms,
            warm_ms,
            mapped_model_size / 1024.0 / 1024.0,
            page_model_offset / 1024.0 / 1024.0);
    if (!warmed) return 0;
    return 1;
}

static id<MTLBuffer> ds4_gpu_new_transient_buffer(NSUInteger bytes, const char *label) {
    if (bytes == 0) bytes = 1;

    id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "ds4: failed to allocate Metal transient buffer %s (%llu bytes)\n",
                label ? label : "(unnamed)", (unsigned long long)bytes);
        return nil;
    }
    if (label) buffer.label = [NSString stringWithUTF8String:label];

    /*
     * CPU-filled buffers must survive until their command buffer completes.
     * A local ObjC strong variable is not enough when the encoder function
     * returns before the caller commits the command buffer.
     */
    [g_transient_buffers addObject:buffer];
    return buffer;
}

int ds4_gpu_use_m5_simdgroup_matrix(void);

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_pipeline(
        const char *function_name,
        bool        bc_inp,
        bool        bc_out) {
    bool m5_sgmatrix = ds4_gpu_use_m5_simdgroup_matrix() != 0;
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_bco=%d_m5sg=%d",
                     function_name,
                     bc_inp ? 1 : 0,
                     bc_out ? 1 : 0,
                     m5_sgmatrix ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];
    [constants setConstantValue:&m5_sgmatrix type:MTLDataTypeBool atIndex:702];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mm_id_pipeline(
        const char *function_name,
        bool        bc_inp) {
    bool m5_sgmatrix = ds4_gpu_use_m5_simdgroup_matrix() != 0;
    NSString *key = [NSString stringWithFormat:@"%s_bci=%d_m5sg=%d",
                     function_name,
                     bc_inp ? 1 : 0,
                     m5_sgmatrix ? 1 : 0];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&m5_sgmatrix type:MTLDataTypeBool atIndex:702];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_pipeline(
        const char *function_name) {
    NSString *key = [NSString stringWithFormat:@"%s", function_name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found\n", function_name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static int ds4_gpu_disable_hot_pipeline_statics(void) {
    static int initialized;
    static int disabled;
    if (!initialized) {
        disabled = getenv("DS4_METAL_DISABLE_HOT_PIPELINE_STATICS") != NULL;
        initialized = 1;
    }
    return disabled;
}

static id<MTLComputePipelineState> ds4_gpu_hot_pipeline(
        id<MTLComputePipelineState> pipeline,
        const char *fallback_name) {
    if (!ds4_gpu_disable_hot_pipeline_statics()) return pipeline;
    return ds4_gpu_get_pipeline(fallback_name);
}

static int ds4_gpu_use_compressor_pair_nr4(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_COMPRESSOR_PAIR_NR4") != NULL;
        initialized = 1;
    }
    return enabled;
}

int ds4_gpu_use_m5_simdgroup_matrix(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        const char *disable = getenv("DS4_METAL_DISABLE_M5_SIMDGROUP_MATRIX");
        const char *force = getenv("DS4_METAL_FORCE_M5_SIMDGROUP_MATRIX");
        enabled = disable ? 0 : (force ? 1 : ds4_gpu_is_m5_device());
        initialized = 1;
    }
    return enabled;
}

static int ds4_gpu_warm_model_views(void) {
    if (g_model_view_count == 0) return 1;

    id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    if (!pipeline) return 0;

    uint64_t stride = 1024ull * 1024ull;
    const char *stride_env = getenv("DS4_METAL_MODEL_WARMUP_STRIDE_MB");
    if (stride_env && stride_env[0]) {
        char *end = NULL;
        unsigned long long mb = strtoull(stride_env, &end, 10);
        if (end != stride_env && mb > 0 && mb <= 1024) {
            stride = mb * 1024ull * 1024ull;
        }
    }

    uint64_t total_touches = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        total_touches += (g_model_views[i].bytes + stride - 1) / stride;
    }
    if (total_touches == 0 || total_touches > (uint64_t)NSUIntegerMax) return 0;

    const NSUInteger out_bytes = (NSUInteger)total_touches;
    id<MTLBuffer> out = [g_device newBufferWithLength:out_bytes
                                             options:MTLResourceStorageModeShared];
    if (!out) {
        fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
        return 0;
    }
    out.label = @"ds4_model_warmup";

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) {
        fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    uint64_t dst_offset = 0;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        const uint64_t bytes = g_model_views[i].bytes;
        const uint64_t n = (bytes + stride - 1) / stride;
        [enc setBuffer:g_model_views[i].buffer offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc setBytes:&stride length:sizeof(stride) atIndex:2];
        [enc setBytes:&bytes length:sizeof(bytes) atIndex:3];
        [enc setBytes:&dst_offset length:sizeof(dst_offset) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n + 255) / 256), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        dst_offset += n;
    }
    ds4_gpu_end_compute_encoder(cb, enc);

    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                [[cb.error localizedDescription] UTF8String]);
        return 0;
    }

    return 1;
}

static const char *ds4_gpu_mul_mm_id_map0_name(uint32_t ne20) {
    switch (ne20) {
        case 1:  return "kernel_mul_mm_id_map0_ne20_1";
        case 2:  return "kernel_mul_mm_id_map0_ne20_2";
        case 4:  return "kernel_mul_mm_id_map0_ne20_4";
        case 5:  return "kernel_mul_mm_id_map0_ne20_5";
        case 6:  return "kernel_mul_mm_id_map0_ne20_6";
        case 8:  return "kernel_mul_mm_id_map0_ne20_8";
        case 10: return "kernel_mul_mm_id_map0_ne20_10";
        case 16: return "kernel_mul_mm_id_map0_ne20_16";
        case 22: return "kernel_mul_mm_id_map0_ne20_22";
        default: return NULL;
    }
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_pipeline(
        const char *function_name,
        int16_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d", function_name, (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_mul_mv_ext_pipeline(
        const char *function_name,
        int16_t     nsg,
        int16_t     nxpsg) {
    NSString *key = [NSString stringWithFormat:@"%s_nsg=%d_nxpsg=%d",
                     function_name, (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nsg   type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pad_pipeline(
        bool    has_mask,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_pad_mask=%d_ncpsg=%d",
                     has_mask ? 1 : 0, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:100];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:125];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_pad"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_pad pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_blk_pipeline(
        int32_t nqptg,
        int32_t ncpsg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_blk_nqptg=%d_ncpsg=%d",
                     (int)nqptg, (int)ncpsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&nqptg type:MTLDataTypeInt atIndex:224];
    [constants setConstantValue:&ncpsg type:MTLDataTypeInt atIndex:225];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_blk"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_blk pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        bool        bc_mask,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg) {
    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     bc_mask ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:300];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:301];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:302];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:303];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:304];
    [constants setConstantValue:&bc_mask   type:MTLDataTypeBool atIndex:310];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:320];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:321];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:322];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_vec_pipeline(
        const char *function_name,
        bool        has_mask,
        bool        has_sinks,
        bool        has_bias,
        bool        has_scap,
        bool        has_kvpad,
        int32_t     ns10,
        int32_t     ns20,
        int32_t     nsg,
        int32_t     nwg) {
    NSString *key = [NSString stringWithFormat:@"%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
                     function_name,
                     has_mask ? 1 : 0,
                     has_sinks ? 1 : 0,
                     has_bias ? 1 : 0,
                     has_scap ? 1 : 0,
                     has_kvpad ? 1 : 0,
                     (int)ns10,
                     (int)ns20,
                     (int)nsg,
                     (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&has_mask  type:MTLDataTypeBool atIndex:400];
    [constants setConstantValue:&has_sinks type:MTLDataTypeBool atIndex:401];
    [constants setConstantValue:&has_bias  type:MTLDataTypeBool atIndex:402];
    [constants setConstantValue:&has_scap  type:MTLDataTypeBool atIndex:403];
    [constants setConstantValue:&has_kvpad type:MTLDataTypeBool atIndex:404];
    [constants setConstantValue:&ns10 type:MTLDataTypeInt atIndex:420];
    [constants setConstantValue:&ns20 type:MTLDataTypeInt atIndex:421];
    [constants setConstantValue:&nsg  type:MTLDataTypeInt atIndex:422];
    [constants setConstantValue:&nwg  type:MTLDataTypeInt atIndex:423];

    NSError *error = nil;
    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> fn = [g_library newFunctionWithName:name
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal %s function not found: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal %s pipeline failed: %s\n",
                function_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static id<MTLComputePipelineState> ds4_gpu_get_flash_attn_reduce_pipeline(
        int32_t dv,
        int32_t nwg) {
    NSString *key = [NSString stringWithFormat:@"kernel_flash_attn_ext_vec_reduce_dv=%d_nwg=%d",
                     (int)dv, (int)nwg];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached;

    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&dv  type:MTLDataTypeInt atIndex:500];
    [constants setConstantValue:&nwg type:MTLDataTypeInt atIndex:501];

    NSError *error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"kernel_flash_attn_ext_vec_reduce"
                                         constantValues:constants
                                                  error:&error];
    if (!fn) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce function not found: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: Metal kernel_flash_attn_ext_vec_reduce pipeline failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    [g_pipeline_cache setObject:pipeline forKey:key];
    return pipeline;
}

static uint32_t ds4_gpu_flash_attn_vec_nsg(uint32_t n_keys, uint32_t nwg, uint32_t ncpsg) {
    uint32_t nsg = 1;
    while (2u * nwg * nsg * ncpsg < n_keys && nsg < 4u) {
        nsg *= 2u;
    }
    return nsg;
}

static int ds4_gpu_trace_allocs(void) {
    static int initialized;
    static int enabled;
    if (!initialized) {
        enabled = getenv("DS4_METAL_TRACE_ALLOCS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static double ds4_gpu_mib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

static double ds4_gpu_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

void ds4_gpu_print_memory_report(const char *label) {
    const uint64_t scratch =
        (uint64_t)g_flash_attn_mask_bytes +
        (uint64_t)g_flash_attn_pad_bytes +
        (uint64_t)g_flash_attn_tmp_bytes +
        (uint64_t)g_flash_attn_blk_bytes +
        (uint64_t)g_flash_attn_ring_bytes +
        (uint64_t)g_flash_attn_kv_bytes +
        (uint64_t)g_compressor_pool_kv_bytes +
        (uint64_t)g_compressor_pool_score_bytes +
        (uint64_t)g_compressor_pool_score_cont_bytes +
        (uint64_t)g_compressor_pool_softmax_bytes +
        (uint64_t)g_compressor_pool_product_bytes +
        (uint64_t)g_compressor_store_ape_bytes +
        (uint64_t)g_compressor_store_score_bytes +
        (uint64_t)g_embed_rows_bytes +
        (uint64_t)g_router_selection_bytes +
        (uint64_t)g_router_weight_sum_bytes +
        (uint64_t)g_indexer_head_scores_bytes +
        (uint64_t)g_indexer_topk_bytes +
        (uint64_t)g_indexed_topk_bytes +
        (uint64_t)g_f16_round_scratch_bytes +
        (uint64_t)g_raw_store_round_bytes +
        (uint64_t)g_moe_gate_scratch_bytes +
        (uint64_t)g_moe_down_scratch_bytes +
        (uint64_t)g_moe_id_map_bytes;

    fprintf(stderr, "ds4: Metal memory report%s%s\n",
            label && label[0] ? " " : "",
            label && label[0] ? label : "");
    fprintf(stderr,
            "ds4:   runtime tensors live %.2f MiB peak %.2f MiB\n",
            ds4_gpu_mib(g_tensor_alloc_live_bytes),
            ds4_gpu_mib(g_tensor_alloc_peak_bytes));
    fprintf(stderr,
            "ds4:   mmap model wrapper spans %llu buffers %.2f GiB total, %.2f GiB max (not copied)\n",
            (unsigned long long)g_model_wrap_count,
            ds4_gpu_gib(g_model_wrap_bytes),
            ds4_gpu_gib(g_model_wrap_max_bytes));
    fprintf(stderr,
            "ds4:   model residency requests %llu%s\n",
            (unsigned long long)g_model_residency_count,
            getenv("DS4_METAL_NO_RESIDENCY") != NULL ? " (disabled)" : "");
    fprintf(stderr,
            "ds4:   scratch %.2f MiB (flash mask %.2f, pad %.2f, tmp %.2f, blk %.2f, ring %.2f, kv %.2f, compressor %.2f, router %.2f, indexer %.2f, moe %.2f, f16 %.2f, raw-store %.2f)\n",
            ds4_gpu_mib(scratch),
            ds4_gpu_mib((uint64_t)g_flash_attn_mask_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_pad_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_tmp_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_blk_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_ring_bytes),
            ds4_gpu_mib((uint64_t)g_flash_attn_kv_bytes),
            ds4_gpu_mib((uint64_t)g_compressor_pool_kv_bytes +
                          (uint64_t)g_compressor_pool_score_bytes +
                          (uint64_t)g_compressor_pool_score_cont_bytes +
                          (uint64_t)g_compressor_pool_softmax_bytes +
                          (uint64_t)g_compressor_pool_product_bytes +
                          (uint64_t)g_compressor_store_ape_bytes +
                          (uint64_t)g_compressor_store_score_bytes +
                          (uint64_t)g_embed_rows_bytes),
            ds4_gpu_mib((uint64_t)g_router_selection_bytes +
                          (uint64_t)g_router_weight_sum_bytes),
            ds4_gpu_mib((uint64_t)g_indexer_head_scores_bytes +
                          (uint64_t)g_indexer_topk_bytes +
                          (uint64_t)g_indexed_topk_bytes),
            ds4_gpu_mib((uint64_t)g_moe_gate_scratch_bytes +
                          (uint64_t)g_moe_down_scratch_bytes +
                          (uint64_t)g_moe_id_map_bytes),
            ds4_gpu_mib((uint64_t)g_f16_round_scratch_bytes),
            ds4_gpu_mib((uint64_t)g_raw_store_round_bytes));
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void ds4_gpu_set_model_residency_mode(bool request_residency, bool warm_views) {
    g_model_request_residency = request_residency ? 1 : 0;
    g_model_warm_views = warm_views ? 1 : 0;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset);

static const char *ds4_gpu_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_MUL_MM_M5_SGMATRIX 702\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#define M_PI_F 3.14159265358979323846f\n"
"\n"
"// Reads one byte per stride to warm model-backed pages without copying the\n"
"// model. This is outside inference and exists only to reduce first-use stalls.\n"
"kernel void kernel_touch_u8_stride(\n"
"        device const uchar    *src        [[buffer(0)]],\n"
"        device uchar          *dst        [[buffer(1)]],\n"
"        constant ulong        &stride     [[buffer(2)]],\n"
"        constant ulong        &bytes      [[buffer(3)]],\n"
"        constant ulong        &dst_offset [[buffer(4)]],\n"
"        uint gid [[thread_position_in_grid]]) {\n"
"    ulong off = (ulong)gid * stride;\n"
"    if (off >= bytes) return;\n"
"    dst[dst_offset + (ulong)gid] = src[off];\n"
"}\n"
"\n"
"enum ds4_sort_order {\n"
"    DS4_SORT_ORDER_ASC,\n"
"    DS4_SORT_ORDER_DESC,\n"
"};\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"\n";

static NSString *ds4_gpu_full_source(void) {
    NSString *base = [NSString stringWithUTF8String:ds4_gpu_source];
    NSFileManager *fm = [NSFileManager defaultManager];
    /*
     * Kernels are kept as separate files for review, then concatenated into one
     * Metal library.  Environment overrides are still honored so a diagnostic
     * run can swap one source file without changing the executable.
     */
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"DS4_METAL_FLASH_ATTN_SOURCE", @"metal/flash_attn.metal"],
        @[@"DS4_METAL_DENSE_SOURCE",      @"metal/dense.metal"],
        @[@"DS4_METAL_MOE_SOURCE",        @"metal/moe.metal"],
        @[@"DS4_METAL_DSV4_HC_SOURCE",    @"metal/dsv4_hc.metal"],
        @[@"DS4_METAL_UNARY_SOURCE",      @"metal/unary.metal"],
        @[@"DS4_METAL_DSV4_KV_SOURCE",    @"metal/dsv4_kv.metal"],
        @[@"DS4_METAL_DSV4_ROPE_SOURCE",  @"metal/dsv4_rope.metal"],
        @[@"DS4_METAL_DSV4_MISC_SOURCE",  @"metal/dsv4_misc.metal"],
        @[@"DS4_METAL_ARGSORT_SOURCE",    @"metal/argsort.metal"],
        @[@"DS4_METAL_CPY_SOURCE",        @"metal/cpy.metal"],
        @[@"DS4_METAL_CONCAT_SOURCE",     @"metal/concat.metal"],
        @[@"DS4_METAL_GET_ROWS_SOURCE",   @"metal/get_rows.metal"],
        @[@"DS4_METAL_SUM_ROWS_SOURCE",   @"metal/sum_rows.metal"],
        @[@"DS4_METAL_SOFTMAX_SOURCE",    @"metal/softmax.metal"],
        @[@"DS4_METAL_REPEAT_SOURCE",     @"metal/repeat.metal"],
        @[@"DS4_METAL_GLU_SOURCE",        @"metal/glu.metal"],
        @[@"DS4_METAL_NORM_SOURCE",       @"metal/norm.metal"],
        @[@"DS4_METAL_BIN_SOURCE",        @"metal/bin.metal"],
        @[@"DS4_METAL_SET_ROWS_SOURCE",   @"metal/set_rows.metal"],
    ];

    NSMutableString *source = [NSMutableString stringWithString:base];
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv([spec[0] UTF8String]);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0]) {
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        }
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;

            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "ds4: failed to read Metal source %s: %s\n",
                        [path UTF8String], [[error localizedDescription] UTF8String]);
                return nil;
            }
            loaded_path = path;
            break;
        }

        if (!loaded) {
            fprintf(stderr,
                    "ds4: Metal source %s not found (set %s to override)\n",
                    [spec[1] UTF8String], [spec[0] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

typedef struct {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_get_rows_args;

typedef struct {
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
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_repeat_args;

typedef struct {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_set_rows_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
} ds4_gpu_concat_args;

typedef struct {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_cpy_args;

static ds4_gpu_cpy_args ds4_gpu_make_cpy_1d_args(
        uint32_t n,
        uint64_t src_elem,
        uint64_t dst_elem) {
    return (ds4_gpu_cpy_args) {
        .nk0 = (int64_t)n,
        .ne00 = (int64_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = src_elem,
        .nb01 = (uint64_t)n * src_elem,
        .nb02 = (uint64_t)n * src_elem,
        .nb03 = (uint64_t)n * src_elem,
        .ne0 = (int64_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = dst_elem,
        .nb1 = (uint64_t)n * dst_elem,
        .nb2 = (uint64_t)n * dst_elem,
        .nb3 = (uint64_t)n * dst_elem,
    };
}

static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static float ds4_gpu_negative_infinity(void) {
    union { uint32_t u; float f; } v = { 0xff800000u };
    return v.f;
}

static float ds4_gpu_positive_infinity(void) {
    union { uint32_t u; float f; } v = { 0x7f800000u };
    return v.f;
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride);

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride);

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n);

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value);

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n);

typedef struct {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ds4_gpu_glu_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ds4_gpu_bin_args;

typedef struct {
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
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} ds4_gpu_unary_args;

static ds4_gpu_bin_args ds4_gpu_make_bin_rows_args(uint32_t n, uint32_t rows, uint32_t rhs_n) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint64_t rhs_row_bytes = (uint64_t)rhs_n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)rhs_n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = rhs_row_bytes,
        .nb13 = rhs_row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_unary_args ds4_gpu_make_unary_rows_args(
        uint32_t n,
        uint32_t rows,
        int      c4,
        float    scale,
        float    bias) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    const uint32_t n_kernel = c4 ? n / 4u : n;
    return (ds4_gpu_unary_args) {
        .ne00 = (int32_t)n_kernel,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne0 = (int32_t)n_kernel,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .slope = 0.0f,
        .scale = scale,
        .bias = bias,
        .val = 0.0f,
        .min = 0.0f,
        .max = 0.0f,
    };
}

static ds4_gpu_bin_args ds4_gpu_make_bin_same_rows_args(uint32_t n, uint32_t rows) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)n,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = (uint64_t)rows * row_bytes,
        .nb03 = (uint64_t)rows * row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = (uint64_t)rows * row_bytes,
        .nb13 = (uint64_t)rows * row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = (uint64_t)rows * row_bytes,
        .nb3 = (uint64_t)rows * row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off);

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows);

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_q8_0_matvec_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mv_ext_args;

typedef ds4_gpu_q8_0_matvec_args ds4_gpu_f16_matvec_args;

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_q8_0_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_f16_matvec_args ds4_gpu_make_f16_mv_args(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row_bytes = in_dim * sizeof(uint16_t);
    return (ds4_gpu_f16_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(uint16_t),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * sizeof(float),
        .nb13 = in_dim * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_q8_0_matvec_args ds4_gpu_make_f32_mv_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_vec) {
    const uint64_t row_bytes = in_dim * sizeof(float);
    return (ds4_gpu_q8_0_matvec_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_vec,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_vec * sizeof(float),
        .nb13 = in_dim * n_vec * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_vec,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}

typedef struct {
    const char *function_name;
    int16_t     nsg;
    int32_t     nr0;
    NSUInteger  smem;
} ds4_gpu_mv_dispatch;

static ds4_gpu_mv_dispatch ds4_gpu_make_q8_0_mv_dispatch(void) {
    return (ds4_gpu_mv_dispatch) {
        .function_name = "kernel_mul_mv_q8_0_f32",
        .nsg = 4,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mv_dispatch ds4_gpu_make_plain_mv_dispatch(
        uint64_t in_dim,
        int      f32_weights) {
    if (in_dim < 32) {
        return (ds4_gpu_mv_dispatch) {
            .function_name = f32_weights ? "kernel_mul_mv_f32_f32_short" : "kernel_mul_mv_f16_f32_short",
            .nsg = 1,
            .nr0 = 32,
            .smem = 0,
        };
    }

    const int16_t nsg = (int16_t)((in_dim + 127u) / 128u > 8u ? 8u : (in_dim + 127u) / 128u);
    const int use_4 = (in_dim % 4u) == 0;
    return (ds4_gpu_mv_dispatch) {
        .function_name = f32_weights
            ? (use_4 ? "kernel_mul_mv_f32_f32_4" : "kernel_mul_mv_f32_f32")
            : (use_4 ? "kernel_mul_mv_f16_f32_4" : "kernel_mul_mv_f16_f32"),
        .nsg = nsg,
        .nr0 = 2,
        .smem = 32u * 2u * sizeof(float),
    };
}

static ds4_gpu_mul_mm_args ds4_gpu_make_mm_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mm_args) {
        .ne00 = (int32_t)in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static ds4_gpu_mul_mv_ext_args ds4_gpu_make_mv_ext_args(
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t elem_bytes,
        uint64_t row_bytes) {
    return (ds4_gpu_mul_mv_ext_args) {
        .ne00 = (int32_t)in_dim,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = elem_bytes,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = (int32_t)in_dim,
        .ne11 = (int32_t)n_tok,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = in_dim * sizeof(float),
        .nb12 = in_dim * n_tok * sizeof(float),
        .nb13 = in_dim * n_tok * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tok,
        .r2 = 1,
        .r3 = 1,
    };
}

static int16_t ds4_gpu_mv_ext_nxpsg(uint64_t in_dim, uint64_t n_tok) {
    if ((in_dim % 256u) == 0 && n_tok < 3) return 16;
    if ((in_dim % 128u) == 0) return 8;
    return 4;
}

static int16_t ds4_gpu_mv_ext_r1ptg(uint64_t n_tok) {
    switch (n_tok) {
    case 2: return 2;
    case 3:
    case 6: return 3;
    case 4:
    case 7:
    case 8: return 4;
    case 5: return 5;
    default: return 0;
    }
}

static const char *ds4_gpu_mv_ext_name(int q8, int16_t r1ptg) {
    if (q8) {
        switch (r1ptg) {
        case 2: return "kernel_mul_mv_ext_q8_0_f32_r1_2";
        case 3: return "kernel_mul_mv_ext_q8_0_f32_r1_3";
        case 4: return "kernel_mul_mv_ext_q8_0_f32_r1_4";
        case 5: return "kernel_mul_mv_ext_q8_0_f32_r1_5";
        default: return NULL;
        }
    }

    switch (r1ptg) {
    case 2: return "kernel_mul_mv_ext_f16_f32_r1_2";
    case 3: return "kernel_mul_mv_ext_f16_f32_r1_3";
    case 4: return "kernel_mul_mv_ext_f16_f32_r1_4";
    case 5: return "kernel_mul_mv_ext_f16_f32_r1_5";
    default: return NULL;
    }
}

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ds4_gpu_rms_norm_args;

typedef struct {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
} ds4_gpu_qkv_rms_norm_args;

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_args(uint32_t n, uint32_t rows, float eps) {
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n,
        .ne00_t = (int32_t)(n / 4u),
        .nb1 = row_bytes,
        .nb2 = row_bytes * rows,
        .nb3 = row_bytes * rows,
        .eps = eps,
        .nef1 = { (int32_t)rows, 1, 1 },
        .nef2 = { 1, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { row_bytes * rows, row_bytes, row_bytes },
        .nbf3 = { row_bytes * rows, row_bytes, row_bytes },
    };
}

static ds4_gpu_rms_norm_args ds4_gpu_make_rms_norm_3d_args(
        uint32_t n0,
        uint32_t n1,
        uint32_t n2,
        float    eps) {
    const uint64_t row_bytes = (uint64_t)n0 * sizeof(float);
    const uint64_t plane_bytes = row_bytes * n1;
    return (ds4_gpu_rms_norm_args) {
        .ne00 = (int32_t)n0,
        .ne00_t = (int32_t)(n0 / 4u),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = plane_bytes * n2,
        .eps = eps,
        .nef1 = { (int32_t)n1, 1, 1 },
        .nef2 = { (int32_t)n2, 1, 1 },
        .nef3 = { 1, 1, 1 },
        .nbf1 = { row_bytes, row_bytes, row_bytes },
        .nbf2 = { plane_bytes, row_bytes, row_bytes },
        .nbf3 = { plane_bytes * n2, row_bytes, row_bytes },
    };
}

static NSUInteger ds4_gpu_rms_norm_threads(uint32_t n) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < 1024u) nth *= 2u;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

static NSUInteger ds4_gpu_rms_norm_pipeline_threads(
        uint32_t                  n,
        id<MTLComputePipelineState> pipeline) {
    NSUInteger ne00_t = n / 4u;
    NSUInteger max_threads = pipeline ? [pipeline maxTotalThreadsPerThreadgroup] : 1024u;
    NSUInteger nth = 32u;
    while (nth < ne00_t && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > ne00_t) nth = ne00_t;
    return nth ? nth : 1u;
}

typedef struct {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
} ds4_gpu_hc_split_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_hc_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float eps;
} ds4_gpu_hc_split_weighted_sum_args;

typedef struct {
    int64_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float eps;
    float norm_eps;
} ds4_gpu_hc_split_weighted_sum_norm_args;

typedef struct {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t has_add;
} ds4_gpu_hc_expand_args;

typedef struct {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} ds4_gpu_mul_mv_id_args;

typedef struct {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
} ds4_gpu_mul_mm_id_map_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ds4_gpu_mul_mm_id_args;

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0);

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg);

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens);
static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size);

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off);

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off);

typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_gpu_flash_attn_blk_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ds4_gpu_flash_attn_vec_args;

typedef struct {
    int32_t nrows;
} ds4_gpu_flash_attn_reduce_args;

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool src2;
} ds4_gpu_rope_tail_batch_args;

static ds4_gpu_rope_tail_batch_args ds4_gpu_make_rope_tail_args(
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        bool     inverse,
        float    freq_base,
        float    freq_scale,
        float    ext_factor,
        float    attn_factor,
        float    beta_fast,
        float    beta_slow) {
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t tok_bytes = (uint64_t)n_head * row_bytes;
    return (ds4_gpu_rope_tail_batch_args) {
        .ne00 = head_dim,
        .ne01 = n_head,
        .ne02 = n_tok,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = tok_bytes,
        .nb03 = (uint64_t)n_tok * tok_bytes,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = tok_bytes,
        .nb3 = (uint64_t)n_tok * tok_bytes,
        .n_dims = (int32_t)n_rot,
        .mode = 0,
        .n_ctx_orig = (int32_t)n_ctx_orig,
        .inverse = inverse ? 1 : 0,
        .freq_base = freq_base,
        .freq_scale = freq_scale,
        .ext_factor = ext_factor,
        .attn_factor = attn_factor,
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .src2 = false,
    };
}

static int ds4_gpu_encode_rope_tail_inplace(
        id<MTLCommandBuffer>                 cb,
        id<MTLBuffer>                        xbuf,
        NSUInteger                           xoff,
        const ds4_gpu_rope_tail_batch_args *args,
        uint32_t                             n_tok,
        uint32_t                             n_head,
        uint32_t                             head_dim,
        uint32_t                             pos0,
        uint32_t                             pos_step) {
    int32_t pos_stack[256];
    int32_t *pos = pos_stack;
    if (n_tok > (uint32_t)(sizeof(pos_stack) / sizeof(pos_stack[0]))) {
        pos = malloc((size_t)n_tok * sizeof(*pos));
        if (!pos) {
            fprintf(stderr, "ds4: failed to allocate Metal RoPE position buffer\n");
            return 0;
        }
    }
    for (uint32_t t = 0; t < n_tok; t++) pos[t] = (int32_t)(pos0 + t * pos_step);

    const NSUInteger pos_bytes = (NSUInteger)n_tok * sizeof(*pos);
    id<MTLBuffer> posbuf = nil;
    if (pos_bytes > 4096u) {
        /*
         * Metal inline setBytes data is meant for small constants. Long prefill
         * RoPE calls need thousands of positions; passing that much inline can
         * make the Apple driver abort the process instead of reporting a normal
         * API error.
         */
        posbuf = ds4_gpu_new_transient_buffer(pos_bytes, "ds4_rope_positions");
        if (!posbuf) {
            if (pos != pos_stack) free(pos);
            return 0;
        }
        memcpy([posbuf contents], pos, pos_bytes);
    }

    const NSUInteger nth = (NSUInteger)(head_dim < 256u ? head_dim : 256u);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_rope_tail_batch_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:xbuf offset:xoff atIndex:1];
    if (posbuf) {
        [enc setBuffer:posbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:pos length:pos_bytes atIndex:2];
    }
    [enc setBuffer:xbuf offset:xoff atIndex:3];
    [enc setBuffer:xbuf offset:xoff atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(nth ? nth : 1u, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    if (pos != pos_stack) free(pos);
    return 1;
}

typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_rot;
} ds4_gpu_dsv4_fp8_kv_quantize_args;

typedef struct {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
} ds4_gpu_dsv4_kv_fp8_store_args;

typedef struct {
    uint32_t width;
} ds4_gpu_dsv4_ratio4_shift_args;

typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
} ds4_gpu_dsv4_compressor_store_one_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_softmax_pool_args;

typedef struct {
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
} ds4_gpu_kargs_argsort;

typedef struct {
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
} ds4_gpu_kargs_argsort_merge;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ds4_gpu_softmax_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_topk_mask_args;

typedef struct {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
} ds4_gpu_dsv4_indexer_weighted_sum_args;

typedef struct {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
} ds4_gpu_dsv4_router_select_one_args;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
} ds4_gpu_dsv4_indexed_attention_args;

typedef struct {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
} ds4_gpu_dsv4_indexer_scores_fused_args;

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float    clamp_value;
} ds4_gpu_dsv4_moe_swiglu_weight_args;

/* Compile the single in-repo Metal source and create the pipelines that every
 * session uses. Shape-dependent kernels with function constants are built
 * lazily by the small ds4_gpu_get_* caches, so startup stays predictable
 * while long-context prefill and decode can still pick specialized variants. */
int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "ds4: Metal device not available\n");
            return 0;
        }
        ds4_gpu_print_device_summary();

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "ds4: failed to create Metal command queue\n");
            g_device = nil;
            return 0;
        }
        g_model_buffer_cache = [NSMutableDictionary dictionary];
        g_pipeline_cache = [NSMutableDictionary dictionary];
        g_transient_buffers = [NSMutableArray array];
        g_pending_cbs = [NSMutableArray array];
        if (!g_model_buffer_cache || !g_pipeline_cache || !g_transient_buffers || !g_pending_cbs) {
            fprintf(stderr, "ds4: Metal bookkeeping allocation failed\n");
            g_pending_cbs = nil;
            g_transient_buffers = nil;
            g_pipeline_cache = nil;
            g_model_buffer_cache = nil;
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        NSError *error = nil;
        NSString *source = ds4_gpu_full_source();
        if (!source) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "ds4: Metal shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_library = library;

        id<MTLFunction> fn = [library newFunctionWithName:@"kernel_get_rows_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_get_rows_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_get_rows_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_get_rows_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_get_rows_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_repeat_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_repeat_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_repeat_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_repeat_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_set_rows_f32_i32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_set_rows_f32_i32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_set_rows_f32_i32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_set_rows_f32_i32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_concat"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_concat function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_concat_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_concat_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_concat pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f32_f16"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f32_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f32_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f32_f16 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_cpy_f16_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_cpy_f16_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cpy_f16_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_cpy_f16_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_fp8_kv_quantize_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_fp8_kv_quantize_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_fp8_kv_quantize_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_fp8_kv_quantize_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_kv_fp8_store_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_kv_fp8_store_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_kv_fp8_store_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_kv_fp8_store_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_ratio4_shift_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_ratio4_shift_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_ratio4_shift_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_ratio4_shift_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_swiglu_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_op = 0;
        int16_t bin_f = 1;
        bool bin_rb = false;
        bool bin_cb = false;
        [bin_constants setConstantValue:&bin_op type:MTLDataTypeShort atIndex:1300];
        [bin_constants setConstantValue:&bin_f  type:MTLDataTypeShort atIndex:1301];
        [bin_constants setConstantValue:&bin_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_constants setConstantValue:&bin_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_add_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_add_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_plain_op = 2;
        int16_t bin_mul_plain_f = 1;
        bool bin_mul_plain_rb = false;
        bool bin_mul_plain_cb = false;
        [bin_mul_constants setConstantValue:&bin_mul_plain_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_constants setConstantValue:&bin_mul_plain_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_constants setConstantValue:&bin_mul_plain_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_constants setConstantValue:&bin_mul_plain_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_mul_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_mul_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_mul_scalar_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_mul_op = 2;
        int16_t bin_mul_f = 1;
        bool bin_mul_rb = false;
        bool bin_mul_cb = true;
        [bin_mul_scalar_constants setConstantValue:&bin_mul_op type:MTLDataTypeShort atIndex:1300];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_f  type:MTLDataTypeShort atIndex:1301];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_mul_scalar_constants setConstantValue:&bin_mul_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_mul_scalar_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_mul_scalar_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_mul_scalar_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 mul-scalar pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *bin_div_row_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t bin_div_op = 3;
        int16_t bin_div_f = 1;
        bool bin_div_rb = false;
        bool bin_div_cb = true;
        [bin_div_row_constants setConstantValue:&bin_div_op type:MTLDataTypeShort atIndex:1300];
        [bin_div_row_constants setConstantValue:&bin_div_f  type:MTLDataTypeShort atIndex:1301];
        [bin_div_row_constants setConstantValue:&bin_div_rb type:MTLDataTypeBool  atIndex:1302];
        [bin_div_row_constants setConstantValue:&bin_div_cb type:MTLDataTypeBool  atIndex:1303];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_bin_fuse_f32_f32_f32"
                           constantValues:bin_div_row_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_bin_div_row_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_bin_div_row_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_bin_fuse_f32_f32_f32 div-row pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_mul_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_mul_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rms_norm_plain_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rms_norm_plain_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_qkv_rms_norm_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_qkv_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_qkv_rms_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_qkv_rms_norm_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *moe_mv_id_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t moe_mv_id_nsg = 2;
        [moe_mv_id_constants setConstantValue:&moe_mv_id_nsg type:MTLDataTypeShort atIndex:600];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q2_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q2_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q2_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q2_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_pair_swiglu_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_pair_swiglu_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_mul_mv_id_q4_K_sum6_f32"
                           constantValues:moe_mv_id_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_moe_mul_mv_id_q4_k_sum6_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_moe_mul_mv_id_q4_k_sum6_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_mul_mv_id_q4_K_sum6_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_rope_tail_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_rope_tail_batch_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_rope_tail_batch_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_rope_tail_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_softmax_pool"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_softmax_pool_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_softmax_pool_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_softmax_pool pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_soft_max_f32_4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_soft_max_f32_4_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_soft_max_f32_4_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_soft_max_f32_4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_argsort_merge_f32_i32_desc"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_argsort_merge_f32_i32_desc_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_argsort_merge_f32_i32_desc_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_argsort_merge_f32_i32_desc pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *sum_rows_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t sum_rows_op = 10;
        [sum_rows_constants setConstantValue:&sum_rows_op type:MTLDataTypeShort atIndex:1400];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                           constantValues:sum_rows_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_sum_rows_f32_f32_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_sum_rows_f32_f32_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_sum_rows_f32_f32 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_topk_mask_scatter"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_topk_mask_scatter_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_topk_mask_scatter_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_topk_mask_scatter pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_indexer_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_dsv4_indexer_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_dsv4_indexer_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_indexer_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_sinkhorn"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_sinkhorn_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_sinkhorn_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_sinkhorn pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_split_weighted_sum_norm4"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_split_weighted_sum_norm_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_split_weighted_sum_norm_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_split_weighted_sum_norm4 pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_weighted_sum"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_weighted_sum_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_weighted_sum_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_weighted_sum pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sigmoid_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sigmoid_op = 102;
        bool unary_cnt = false;
        [unary_sigmoid_constants setConstantValue:&unary_sigmoid_op type:MTLDataTypeShort atIndex:1200];
        [unary_sigmoid_constants setConstantValue:&unary_cnt        type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sigmoid_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sigmoid_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sigmoid_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sigmoid pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_silu_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_silu_op = 106;
        [unary_silu_constants setConstantValue:&unary_silu_op type:MTLDataTypeShort atIndex:1200];
        [unary_silu_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_silu_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_silu_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_silu_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 silu pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_softplus_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_softplus_op = 115;
        [unary_softplus_constants setConstantValue:&unary_softplus_op type:MTLDataTypeShort atIndex:1200];
        [unary_softplus_constants setConstantValue:&unary_cnt         type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_softplus_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_softplus_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_softplus_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 softplus pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_sqrt_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_sqrt_op = 14;
        [unary_sqrt_constants setConstantValue:&unary_sqrt_op type:MTLDataTypeShort atIndex:1200];
        [unary_sqrt_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_sqrt_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_sqrt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_sqrt_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 sqrt pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_clamp_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_clamp_op = 12;
        [unary_clamp_constants setConstantValue:&unary_clamp_op type:MTLDataTypeShort atIndex:1200];
        [unary_clamp_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32"
                           constantValues:unary_clamp_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_clamp_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_clamp_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32 clamp pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_scale_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_scale_op = 10;
        [unary_scale_constants setConstantValue:&unary_scale_op type:MTLDataTypeShort atIndex:1200];
        [unary_scale_constants setConstantValue:&unary_cnt      type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_scale_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_scale_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_scale_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 scale pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        MTLFunctionConstantValues *unary_fill_constants = [[MTLFunctionConstantValues alloc] init];
        int16_t unary_fill_op = 11;
        [unary_fill_constants setConstantValue:&unary_fill_op type:MTLDataTypeShort atIndex:1200];
        [unary_fill_constants setConstantValue:&unary_cnt     type:MTLDataTypeBool  atIndex:1201];

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f32_f32_4"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f32_f32_4 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        error = nil;
        fn = [library newFunctionWithName:@"kernel_unary_f16_f16"
                           constantValues:unary_fill_constants
                                    error:&error];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill function not found: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_unary_fill_f16_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_unary_fill_f16_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_unary_f16_f16 fill pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        fn = [library newFunctionWithName:@"kernel_dsv4_hc_expand"];
        if (!fn) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand function not found\n");
            g_queue = nil;
            g_device = nil;
            return 0;
        }
        g_hc_expand_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_hc_expand_pipeline) {
            fprintf(stderr, "ds4: Metal kernel_dsv4_hc_expand pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        g_dsv4_indexer_score_one_direct_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexer_score_one_direct");
        g_dsv4_compressor_store_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_compressor_store_one");
        g_dsv4_sort_i32_rows_asc_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_sort_i32_rows_asc");
        g_dsv4_indexed_attention_heads8_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8");
        g_dsv4_indexed_attention_heads8_rb4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb4");
        g_dsv4_softplus_sqrt_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_softplus_sqrt_f32_4");
        g_dsv4_router_finalize_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_finalize_one");
        g_dsv4_router_weights_one_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_router_weights_one");
        g_dsv4_hc_expand4_pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_hc_expand4");
        if (!g_dsv4_indexer_score_one_direct_pipeline ||
            !g_dsv4_compressor_store_one_pipeline ||
            !g_dsv4_sort_i32_rows_asc_pipeline ||
            !g_dsv4_indexed_attention_heads8_pipeline ||
            !g_dsv4_indexed_attention_heads8_rb4_pipeline ||
            !g_dsv4_softplus_sqrt_pipeline ||
            !g_dsv4_router_finalize_one_pipeline ||
            !g_dsv4_router_weights_one_pipeline ||
            !g_dsv4_hc_expand4_pipeline) {
            g_queue = nil;
            g_device = nil;
            return 0;
        }

        ds4_ane_mlp_int8w_quant_stats_reset();
        g_initialized = 1;
    }

    return 1;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) {
            return NULL;
        }
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        g_tensor_alloc_live_bytes += bytes;
        if (g_tensor_alloc_live_bytes > g_tensor_alloc_peak_bytes) {
            g_tensor_alloc_peak_bytes = g_tensor_alloc_live_bytes;
        }
        if (ds4_gpu_trace_allocs()) {
            fprintf(stderr,
                    "ds4: Metal tensor alloc %.3f MiB live %.3f MiB peak %.3f MiB\n",
                    (double)bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                    (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
        }
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    const DS4MetalTensor *base_obj = ds4_gpu_tensor_const_obj(base);
    if (offset > base_obj.bytes || bytes > base_obj.bytes - offset) return NULL;
    if (base_obj.offset > UINT64_MAX - offset) return NULL;
    const uint64_t absolute_offset = base_obj.offset + offset;
    if (absolute_offset > (uint64_t)NSUIntegerMax) return NULL;

    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = base_obj.buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        if (obj.owner) {
            if (obj.bytes <= g_tensor_alloc_live_bytes) {
                g_tensor_alloc_live_bytes -= obj.bytes;
            } else {
                g_tensor_alloc_live_bytes = 0;
            }
            if (ds4_gpu_trace_allocs()) {
                fprintf(stderr,
                        "ds4: Metal tensor free %.3f MiB live %.3f MiB peak %.3f MiB\n",
                        (double)obj.bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_live_bytes / (1024.0 * 1024.0),
                        (double)g_tensor_alloc_peak_bytes / (1024.0 * 1024.0));
            }
        }
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    if (!tensor) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    return obj.bytes;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    return (uint8_t *)[obj.buffer contents] + obj.offset;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > ds4_gpu_tensor_bytes(tensor) / sizeof(float)) return 0;
    float *p = ds4_gpu_tensor_contents(tensor);
    if (!p && count != 0) return 0;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy((uint8_t *)[obj.buffer contents] + obj.offset + offset, data, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0)) return 0;
    const DS4MetalTensor *obj = ds4_gpu_tensor_const_obj(tensor);
    if (offset > obj.bytes || bytes > obj.bytes - offset) return 0;
    if (bytes != 0) {
        memcpy(data, (const uint8_t *)[obj.buffer contents] + obj.offset + offset, (size_t)bytes);
    }
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4MetalTensor *d = ds4_gpu_tensor_obj(dst);
    const DS4MetalTensor *s = ds4_gpu_tensor_const_obj(src);
    if (dst_offset > d.bytes || bytes > d.bytes - dst_offset) return 0;
    if (src_offset > s.bytes || bytes > s.bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:s.buffer
            sourceOffset:(NSUInteger)(s.offset + src_offset)
                toBuffer:d.buffer
       destinationOffset:(NSUInteger)(d.offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int ds4_gpu_gather_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *src,
        const ds4_gpu_tensor *rows,
        uint32_t                n_rows,
        uint32_t                width) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !src || !rows || n_rows == 0 || width == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
        id<MTLBuffer> rowsbuf = ds4_gpu_tensor_buffer(rows);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)width * sizeof(float);
        if (!srcbuf || !rowsbuf || !outbuf ||
            ds4_gpu_tensor_bytes(rows) < (uint64_t)n_rows * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(out) < (uint64_t)n_rows * row_bytes) {
            return 0;
        }

        ds4_gpu_get_rows_args args = {
            .ne00t = (int32_t)width,
            .ne00 = (int32_t)width,
            .nb01 = row_bytes,
            .nb02 = ds4_gpu_tensor_bytes(src),
            .nb03 = ds4_gpu_tensor_bytes(src),
            .ne10 = (int32_t)n_rows,
            .nb10 = sizeof(int32_t),
            .nb11 = (uint64_t)n_rows * sizeof(int32_t),
            .nb12 = (uint64_t)n_rows * sizeof(int32_t),
            .nb1 = row_bytes,
            .nb2 = (uint64_t)n_rows * row_bytes,
            .nb3 = (uint64_t)n_rows * row_bytes,
        };

        NSUInteger nth = width;
        const NSUInteger max_threads = g_get_rows_f32_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > max_threads) nth = max_threads;
        if (nth == 0) nth = 1;
        const NSUInteger nw0 = ((NSUInteger)width + nth - 1u) / nth;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_get_rows_f32_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
        [enc setBuffer:rowsbuf offset:ds4_gpu_tensor_offset(rows) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "gather rows f32");
    }
}

int ds4_gpu_scatter_add_rows_f32_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        const ds4_gpu_tensor *rows,
        uint32_t                n_rows,
        uint32_t                width) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!dst || !src || !rows || n_rows == 0 || width == 0) return 0;

    typedef struct {
        uint32_t width;
        uint32_t rows;
        uint64_t src_row_stride;
        uint64_t dst_row_stride;
    } ds4_gpu_scatter_add_rows_args;

    @autoreleasepool {
        id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
        id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
        id<MTLBuffer> rowsbuf = ds4_gpu_tensor_buffer(rows);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_scatter_add_rows_f32");
        const uint64_t row_bytes = (uint64_t)width * sizeof(float);
        if (!dstbuf || !srcbuf || !rowsbuf || !pipeline ||
            ds4_gpu_tensor_bytes(src) < (uint64_t)n_rows * row_bytes ||
            ds4_gpu_tensor_bytes(rows) < (uint64_t)n_rows * sizeof(int32_t)) {
            return 0;
        }

        ds4_gpu_scatter_add_rows_args args = {
            .width = width,
            .rows = n_rows,
            .src_row_stride = row_bytes,
            .dst_row_stride = row_bytes,
        };

        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > width) nth = width;
        if (nth == 0) nth = 1u;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
        [enc setBuffer:rowsbuf offset:ds4_gpu_tensor_offset(rows) atIndex:2];
        [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, "scatter-add rows f32");
    }
}

int ds4_gpu_begin_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return 0;
    g_batch_cb = [g_queue commandBuffer];
    return g_batch_cb != nil;
}

int ds4_gpu_flush_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!g_batch_cb) return 0;

    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    [cb commit];
    [g_pending_cbs addObject:cb];

    g_batch_cb = [g_queue commandBuffer];
    if (!g_batch_cb) {
        (void)ds4_gpu_wait_pending_command_buffers("command batch");
        [g_transient_buffers removeAllObjects];
        return 0;
    }
    return 1;
}

int ds4_gpu_end_commands(void) {
    if (!g_batch_cb) return 0;
    ds4_gpu_close_batch_encoder();
    id<MTLCommandBuffer> cb = g_batch_cb;
    g_batch_cb = nil;
    return ds4_gpu_finish_command_buffer(cb, 1, "command batch");
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (g_batch_cb) return ds4_gpu_end_commands();
    if ([g_pending_cbs count] != 0) {
        int ok = ds4_gpu_wait_pending_command_buffers("synchronize");
        [g_transient_buffers removeAllObjects];
        return ok;
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;
    return ds4_gpu_finish_command_buffer(cb, 1, "synchronize");
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        ds4_gpu_print_ane_prefill_stats();
        if (g_batch_cb) {
            ds4_gpu_close_batch_encoder();
            [g_batch_cb commit];
            [g_batch_cb waitUntilCompleted];
            g_batch_cb = nil;
        }
        (void)ds4_gpu_wait_pending_command_buffers("cleanup");
        [g_transient_buffers removeAllObjects];
        g_set_rows_f32_i32_pipeline = nil;
        g_get_rows_f32_pipeline = nil;
        g_get_rows_f16_pipeline = nil;
        g_get_rows_i32_pipeline = nil;
        g_repeat_f32_pipeline = nil;
        g_concat_pipeline = nil;
        g_cpy_f32_f32_pipeline = nil;
        g_cpy_f32_f16_pipeline = nil;
        g_cpy_f16_f32_pipeline = nil;
        g_swiglu_pipeline = nil;
        g_add_pipeline = nil;
        g_mul_pipeline = nil;
        g_bin_mul_scalar_pipeline = nil;
        g_bin_div_row_pipeline = nil;
        g_unary_sigmoid_pipeline = nil;
        g_unary_silu_pipeline = nil;
        g_unary_softplus_pipeline = nil;
        g_unary_sqrt_pipeline = nil;
        g_unary_clamp_pipeline = nil;
        g_unary_scale_pipeline = nil;
        g_unary_fill_pipeline = nil;
        g_unary_fill_f16_pipeline = nil;
        g_rms_norm_pipeline = nil;
        g_rms_norm_plain_pipeline = nil;
        g_dsv4_qkv_rms_norm_pipeline = nil;
        g_hc_split_sinkhorn_pipeline = nil;
        g_hc_split_weighted_sum_pipeline = nil;
        g_hc_split_weighted_sum_norm_pipeline = nil;
        g_hc_weighted_sum_pipeline = nil;
        g_hc_expand_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_pipeline = nil;
        g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q2_k_pipeline = nil;
        g_moe_mul_mv_id_q2_k_sum6_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_pipeline = nil;
        g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline = nil;
        g_moe_mul_mv_id_q4_k_sum6_pipeline = nil;
        g_moe_mul_mm_id_iq2_xxs_pipeline = nil;
        g_moe_mul_mm_id_q2_k_pipeline = nil;
        g_moe_mul_mm_id_q4_k_pipeline = nil;
        g_mpp_h_i8_f_pipeline = nil;
        g_mpp_f_i8_f_pipeline = nil;
        g_mpp_i8_i8_i32_pipeline = nil;
        g_mpp_swiglu_weight_f32_pipeline = nil;
        g_mpp_swiglu_i32_weight_f32_pipeline = nil;
        g_mpp_swiglu_i32_weight_i8_pipeline = nil;
        g_mpp_quant_f32_i8_pipeline = nil;
        g_mpp_quant_mid_f32_i8_pipeline = nil;
        g_mpp_scale_i32_f32_pipeline = nil;
        g_mpp_dequant_iq2_xxs_i8_pipeline = nil;
        g_mpp_dequant_q2_k_i8_pipeline = nil;
        g_ane_dequant_iq2_xxs_f16_pipeline = nil;
        g_ane_dequant_q2_k_f16_pipeline = nil;
        g_ane_output_pack_f16_f32_pipeline = nil;
        g_rope_tail_batch_pipeline = nil;
        g_dsv4_fp8_kv_quantize_pipeline = nil;
        g_dsv4_kv_fp8_store_pipeline = nil;
        g_dsv4_ratio4_shift_pipeline = nil;
        g_dsv4_softmax_pool_pipeline = nil;
        g_soft_max_f32_pipeline = nil;
        g_soft_max_f32_4_pipeline = nil;
        g_argsort_f32_i32_desc_pipeline = nil;
        g_argsort_merge_f32_i32_desc_pipeline = nil;
        g_sum_rows_f32_f32_pipeline = nil;
        g_dsv4_topk_mask_pipeline = nil;
        g_dsv4_topk_mask_scatter_pipeline = nil;
        g_dsv4_indexer_weighted_sum_pipeline = nil;
        g_dsv4_indexer_score_one_direct_pipeline = nil;
        g_dsv4_compressor_store_one_pipeline = nil;
        g_dsv4_sort_i32_rows_asc_pipeline = nil;
        g_dsv4_indexed_attention_heads8_pipeline = nil;
        g_dsv4_indexed_attention_heads8_rb4_pipeline = nil;
        g_dsv4_softplus_sqrt_pipeline = nil;
        g_dsv4_router_finalize_one_pipeline = nil;
        g_dsv4_router_weights_one_pipeline = nil;
        g_dsv4_hc_expand4_pipeline = nil;
        g_flash_attn_mask_buffer = nil;
        g_flash_attn_pad_buffer = nil;
        g_flash_attn_tmp_buffer = nil;
        g_flash_attn_blk_buffer = nil;
        g_flash_attn_ring_buffer = nil;
        g_flash_attn_kv_buffer = nil;
        g_compressor_pool_kv_buffer = nil;
        g_compressor_pool_score_buffer = nil;
        g_compressor_pool_score_cont_buffer = nil;
        g_compressor_pool_softmax_buffer = nil;
        g_compressor_pool_product_buffer = nil;
        g_compressor_store_ape_buffer = nil;
        g_compressor_store_score_buffer = nil;
        g_embed_rows_buffer = nil;
        g_router_selection_buffer = nil;
        g_router_weight_sum_buffer = nil;
        g_indexer_head_scores_buffer = nil;
        g_indexer_topk_buffer = nil;
        g_indexed_topk_buffer = nil;
        g_f16_round_scratch_buffer = nil;
        g_raw_store_round_buffer = nil;
        g_moe_gate_scratch_buffer = nil;
        g_moe_down_scratch_buffer = nil;
        g_moe_id_map_buffer = nil;
        g_mpp_prefill_x_half_buffer = nil;
        g_mpp_prefill_x_i8_buffer = nil;
        g_mpp_prefill_gate_i8_buffer = nil;
        g_mpp_prefill_up_i8_buffer = nil;
        g_mpp_prefill_down_i8_buffer = nil;
        g_mpp_prefill_gate_i32_buffer = nil;
        g_mpp_prefill_up_i32_buffer = nil;
        g_mpp_prefill_mid_i8_buffer = nil;
        g_mpp_prefill_out_i32_buffer = nil;
        g_ane_prefill_gate_i8_buffer = nil;
        g_ane_prefill_up_i8_buffer = nil;
        g_ane_prefill_down_i8_buffer = nil;
        g_ane_prefill_x_i8_buffer = nil;
        g_ane_prefill_gate_f16_buffer = nil;
        g_ane_prefill_up_f16_buffer = nil;
        g_ane_prefill_down_f16_buffer = nil;
        g_attn_out_group_ids_buffer = nil;
        g_model_map_ptr = NULL;
        g_model_map_size = 0;
        g_model_mapped_offset = 0;
        g_model_mapped_size = 0;
        g_tensor_alloc_live_bytes = 0;
        g_tensor_alloc_peak_bytes = 0;
        g_flash_attn_mask_bytes = 0;
        g_flash_attn_pad_bytes = 0;
        g_flash_attn_tmp_bytes = 0;
        g_flash_attn_blk_bytes = 0;
        g_flash_attn_ring_bytes = 0;
        g_flash_attn_kv_bytes = 0;
        g_compressor_pool_kv_bytes = 0;
        g_compressor_pool_score_bytes = 0;
        g_compressor_pool_score_cont_bytes = 0;
        g_compressor_pool_softmax_bytes = 0;
        g_compressor_pool_product_bytes = 0;
        g_compressor_store_ape_bytes = 0;
        g_compressor_store_score_bytes = 0;
        g_embed_rows_bytes = 0;
        g_router_selection_bytes = 0;
        g_router_weight_sum_bytes = 0;
        g_indexer_head_scores_bytes = 0;
        g_indexer_topk_bytes = 0;
        g_indexed_topk_bytes = 0;
        g_f16_round_scratch_bytes = 0;
        g_raw_store_round_bytes = 0;
        g_moe_gate_scratch_bytes = 0;
        g_moe_down_scratch_bytes = 0;
        g_moe_id_map_bytes = 0;
        g_mpp_prefill_x_half_bytes = 0;
        g_mpp_prefill_x_i8_bytes = 0;
        g_mpp_prefill_gate_i8_bytes = 0;
        g_mpp_prefill_up_i8_bytes = 0;
        g_mpp_prefill_down_i8_bytes = 0;
        g_mpp_prefill_gate_i32_bytes = 0;
        g_mpp_prefill_up_i32_bytes = 0;
        g_mpp_prefill_mid_i8_bytes = 0;
        g_mpp_prefill_out_i32_bytes = 0;
        g_ane_prefill_gate_i8_bytes = 0;
        g_ane_prefill_up_i8_bytes = 0;
        g_ane_prefill_down_i8_bytes = 0;
        g_ane_prefill_x_i8_bytes = 0;
        g_ane_prefill_gate_f16_bytes = 0;
        g_ane_prefill_up_f16_bytes = 0;
        g_ane_prefill_down_f16_bytes = 0;
        g_attn_out_group_ids_bytes = 0;
        g_model_wrap_count = 0;
        g_model_wrap_bytes = 0;
        g_model_wrap_max_bytes = 0;
        g_ane_prefill_calls = 0;
        g_ane_prefill_successes = 0;
        g_ane_prefill_skip_big_refs = 0;
        g_ane_prefill_compile_attempts = 0;
        g_ane_prefill_compile_failures = 0;
        g_ane_prefill_eval_calls = 0;
        g_ane_prefill_ane_evaluate_calls = 0;
        g_ane_prefill_eval_refs = 0;
        g_ane_prefill_padded_refs = 0;
        g_ane_prefill_chunks_le16 = 0;
        g_ane_prefill_chunks_le64 = 0;
        g_ane_prefill_chunks_le128 = 0;
        g_ane_prefill_chunks_full = 0;
        memset(g_ane_prefill_batch_eval_calls, 0, sizeof(g_ane_prefill_batch_eval_calls));
        memset(g_ane_prefill_batch_refs, 0, sizeof(g_ane_prefill_batch_refs));
        memset(g_ane_prefill_batch_padded_refs, 0, sizeof(g_ane_prefill_batch_padded_refs));
        g_ane_prefill_eval_failures = 0;
        g_ane_prefill_write_failures = 0;
        g_ane_prefill_reopen_failures = 0;
        g_ane_prefill_x_values = 0;
        g_ane_prefill_x_saturated = 0;
        g_ane_prefill_x_abs_max = 0.0f;
        g_ane_prefill_dequant_ms = 0.0;
        g_ane_prefill_dequant_wait_ms = 0.0;
        g_ane_prefill_input_ms = 0.0;
        g_ane_prefill_x_zero_ms = 0.0;
        g_ane_prefill_x_copy_ms = 0.0;
        g_ane_prefill_out_zero_ms = 0.0;
        g_ane_prefill_eval_ms = 0.0;
        g_ane_prefill_eval_ms_b = 0.0;
        g_ane_prefill_eval_ms_c = 0.0;
        g_ane_prefill_eval_ms_d = 0.0;
        g_ane_prefill_output_ms = 0.0;
        g_ane_prefill_output_f16_copy_ms = 0.0;
        g_ane_prefill_output_convert_ms = 0.0;
        g_ane_prefill_output_gpu_encode_ms = 0.0;
        g_ane_prefill_post_wait_ms = 0.0;
        g_ane_prefill_writeback_ms = 0.0;
        g_ane_prefill_wait_predict_ms = 0.0;
        g_ane_prefill_finish_join_ane_ms = 0.0;
        g_ane_prefill_finish_join_post_ms = 0.0;
        g_ane_prefill_wait_predict_calls = 0;
        g_ane_prefill_finish_calls = 0;
        g_ane_prefill_output_elems = 0;
        g_ane_prefill_output_chunks = 0;
        ds4_ane_mlp_int8w_quant_stats_reset();
        g_model_request_residency = 1;
        g_model_warm_views = 1;
        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();
        [g_pipeline_cache removeAllObjects];
        g_pipeline_cache = nil;
        [g_model_buffer_cache removeAllObjects];
        g_model_buffer_cache = nil;
        g_transient_buffers = nil;
        g_pending_cbs = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
        ds4_ane_mlp_int8w_destroy(g_ane_int8w_ctx);
        g_ane_int8w_ctx = NULL;
        ds4_gpu_ane_ctx_cache_clear();
    }
}

static int ds4_gpu_encode_get_rows_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        weight,
        NSUInteger           weight_offset,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    if (!cb || !weight || !tokens || !out || n_vocab == 0 || n_tokens == 0 || n_embd == 0) {
        return 0;
    }

    const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = (int32_t)n_embd,
        .ne00 = (int32_t)n_embd,
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)n_vocab * src_row_bytes,
        .nb03 = (uint64_t)n_vocab * src_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = dst_row_bytes,
        .nb2 = (uint64_t)n_tokens * dst_row_bytes,
        .nb3 = (uint64_t)n_tokens * dst_row_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;
    const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weight offset:weight_offset atIndex:1];
    [enc setBuffer:tokens offset:tokens_offset atIndex:2];
    [enc setBuffer:out offset:out_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_repeat_hc_embedding(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        rows,
        NSUInteger           rows_offset,
        id<MTLBuffer>        out,
        NSUInteger           out_offset,
        uint32_t             n_tokens,
        uint32_t             n_embd,
        uint32_t             n_hc) {
    if (!cb || !rows || !out || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;

    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    ds4_gpu_repeat_args args = {
        .ne00 = (int32_t)n_embd,
        .ne01 = 1,
        .ne02 = (int32_t)n_tokens,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = embd_bytes,
        .nb02 = embd_bytes,
        .nb03 = (uint64_t)n_tokens * embd_bytes,
        .ne0 = (int32_t)n_embd,
        .ne1 = (int32_t)n_hc,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = embd_bytes,
        .nb2 = (uint64_t)n_hc * embd_bytes,
        .nb3 = (uint64_t)n_tokens * n_hc * embd_bytes,
    };

    NSUInteger nth = (NSUInteger)n_embd;
    const NSUInteger max_threads = g_repeat_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_repeat_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:rows offset:rows_offset atIndex:1];
    [enc setBuffer:out offset:out_offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(n_hc, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t out_bytes = (uint64_t)n_embd * n_hc * sizeof(float);
        if (!outbuf || ds4_gpu_tensor_bytes(out_hc) < out_bytes) {
            fprintf(stderr, "ds4: Metal graph embedding received undersized HC output buffer\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger row_bytes = (NSUInteger)n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             row_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t token_i32 = (int32_t)token;
        const uint64_t src_row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
        const uint64_t dst_row_bytes = (uint64_t)n_embd * sizeof(float);
        ds4_gpu_get_rows_args args = {
            .ne00t = (int32_t)n_embd,
            .ne00 = (int32_t)n_embd,
            .nb01 = src_row_bytes,
            .nb02 = (uint64_t)n_vocab * src_row_bytes,
            .nb03 = (uint64_t)n_vocab * src_row_bytes,
            .ne10 = 1,
            .nb10 = sizeof(int32_t),
            .nb11 = sizeof(int32_t),
            .nb12 = sizeof(int32_t),
            .nb1 = dst_row_bytes,
            .nb2 = dst_row_bytes,
            .nb3 = dst_row_bytes,
        };
        NSUInteger nth = (NSUInteger)n_embd;
        const NSUInteger max_threads = g_get_rows_f16_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > max_threads) nth = max_threads;
        if (nth == 0) nth = 1;
        const NSUInteger nw0 = ((NSUInteger)n_embd + nth - 1u) / nth;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_get_rows_f16_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBytes:&token_i32 length:sizeof(token_i32) atIndex:2];
        [enc setBuffer:g_embed_rows_buffer offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(nw0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed token")) return 0;
    }

    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        id<MTLBuffer> tokbuf = ds4_gpu_tensor_buffer(tokens);
        const uint64_t out_bytes = (uint64_t)n_tokens * n_embd * n_hc * sizeof(float);
        const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
        if (!outbuf || !tokbuf ||
            ds4_gpu_tensor_bytes(out_hc) < out_bytes ||
            ds4_gpu_tensor_bytes(tokens) < token_bytes) {
            fprintf(stderr, "ds4: Metal graph batched embedding received undersized buffers\n");
            return 0;
        }

        const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal graph batched embedding range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        const NSUInteger rows_bytes = (NSUInteger)n_tokens * n_embd * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_embed_rows_buffer,
                                             &g_embed_rows_bytes,
                                             rows_bytes,
                                             "ds4_embed_rows")) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_get_rows_f16(cb,
                                           wbuf,
                                           (NSUInteger)inner_offset,
                                           tokbuf,
                                           ds4_gpu_tensor_offset(tokens),
                                           g_embed_rows_buffer,
                                           0,
                                           n_vocab,
                                           n_tokens,
                                           n_embd) ||
            !ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  g_embed_rows_buffer,
                                                  0,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out_hc),
                                                  n_tokens,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph embed tokens")) return 0;
    }

    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;

    @autoreleasepool {
        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        ds4_gpu_model_residency_clear();
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        if (!ds4_gpu_map_model_views(model_map, model_size, map_offset, map_size)) {
            ds4_gpu_model_residency_clear();
            return 0;
        }
        fprintf(stderr,
                "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n",
                g_model_view_count);
        return 1;
    }
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size);
}

int ds4_gpu_set_model_fd(int fd) {
    (void)fd;
    return 1;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    (void)model_map;
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    return nil;
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer score received undersized buffers\n");
            return 0;
        }

        if (n_head == 64 && head_dim == 128) {
            id<MTLComputePipelineState> direct_pipeline =
                ds4_gpu_hot_pipeline(g_dsv4_indexer_score_one_direct_pipeline,
                                        "kernel_dsv4_indexer_score_one_direct");
            if (!direct_pipeline) return 0;

            ds4_gpu_dsv4_indexer_scores_fused_args args = {
                .n_comp = n_comp,
                .n_tokens = 1,
                .n_head = n_head,
                .head_dim = head_dim,
                .pos0 = 0,
                .ratio = 4,
                .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
                .q_head_stride = (uint64_t)head_dim * sizeof(float),
                .weights_token_stride = (uint64_t)n_head * sizeof(float),
                .index_row_stride = (uint64_t)head_dim * sizeof(float),
                .score_token_stride = (uint64_t)n_comp * sizeof(float),
                .scale = scale,
            };

            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:direct_pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
            [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
            [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
            [enc setThreadgroupMemoryLength:(128u + 4u) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_comp, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer direct score")) return 0;
            return 1;
        }

        const uint64_t head_score_bytes = (uint64_t)n_comp * n_head * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_head_scores_buffer,
                                             &g_indexer_head_scores_bytes,
                                             (NSUInteger)head_score_bytes,
                                             "ds4_indexer_head_scores")) {
            return 0;
        }

        ds4_gpu_q8_0_matvec_args dot_args =
            ds4_gpu_make_f32_mv_args(head_dim, n_comp, n_head);
        ds4_gpu_mv_dispatch dot_dispatch =
            ds4_gpu_make_plain_mv_dispatch(head_dim, 1);
        dot_args.nr0 = dot_dispatch.nr0;
        id<MTLComputePipelineState> dot_pipeline =
            ds4_gpu_get_mul_mv_pipeline(dot_dispatch.function_name, dot_dispatch.nsg);
        if (!dot_pipeline) return 0;
        ds4_gpu_dsv4_indexer_weighted_sum_args sum_args = {
            .ne00 = (int64_t)n_comp,
            .ne01 = 1,
            .ne02 = (int64_t)n_head,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * sizeof(float),
            .ne10 = (int64_t)n_head,
            .ne11 = 1,
            .nb10 = sizeof(float),
            .nb11 = (uint64_t)n_head * sizeof(float),
            .ne0 = (int64_t)n_comp,
            .ne1 = 1,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:dot_pipeline];
        [enc setBytes:&dot_args length:sizeof(dot_args) atIndex:0];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:1];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:2];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:3];
        if (dot_dispatch.smem) {
            [enc setThreadgroupMemoryLength:dot_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + (NSUInteger)dot_dispatch.nr0 - 1u) / (NSUInteger)dot_dispatch.nr0,
                                              n_head,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dot_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_indexer_weighted_sum_pipeline];
        [enc setBytes:&sum_args length:sizeof(sum_args) atIndex:0];
        [enc setBuffer:g_indexer_head_scores_buffer offset:0 atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer score")) return 0;
    }

    return 1;
}

static int ds4_gpu_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t weight_bytes = (uint64_t)n_tokens * n_head * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(index_comp);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        if (!qbuf || !wbuf || !compbuf || !scorebuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(weights) < weight_bytes ||
            ds4_gpu_tensor_bytes(index_comp) < comp_bytes ||
            ds4_gpu_tensor_bytes(scores) < score_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer prefill scores received undersized buffers\n");
            return 0;
        }
        if (head_dim != 128) {
            fprintf(stderr, "ds4: Metal fused DS4 indexer scores expect 128-wide rows\n");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = ds4_gpu_get_pipeline(
            g_quality_mode ? "kernel_dsv4_indexer_scores_tiled_f32"
                           : "kernel_dsv4_indexer_scores_tiled");
        if (!pipeline) return 0;

        ds4_gpu_dsv4_indexer_scores_fused_args args = {
            .n_comp = n_comp,
            .n_tokens = n_tokens,
            .n_head = n_head,
            .head_dim = head_dim,
            .pos0 = pos0,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * head_dim * sizeof(float),
            .q_head_stride = (uint64_t)head_dim * sizeof(float),
            .weights_token_stride = (uint64_t)n_head * sizeof(float),
            .index_row_stride = (uint64_t)head_dim * sizeof(float),
            .score_token_stride = (uint64_t)n_comp * sizeof(float),
            .scale = scale,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(index_comp) atIndex:3];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:4];
        if (g_quality_mode) {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared + dot_shared) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        } else {
            const NSUInteger q_shared = 8u * 128u;
            const NSUInteger k_shared = 32u * 128u;
            const NSUInteger dot_shared = 8u * 32u;
            [enc setThreadgroupMemoryLength:(q_shared + k_shared) * sizeof(uint16_t) +
                                            dot_shared * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_comp + 31u) / 32u,
                                                  ((NSUInteger)n_tokens + 7u) / 8u,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer prefill scores")) return 0;
    }

    return 1;
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return ds4_gpu_indexer_scores_batch_tensor(scores,
                                                 q,
                                                 weights,
                                                 index_comp,
                                                 n_comp,
                                                 n_tokens,
                                                 pos0,
                                                 n_head,
                                                 head_dim,
                                                 ratio,
                                                 scale);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 || top_k > n_comp) return 0;

    @autoreleasepool {
        const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
        id<MTLBuffer> scorebuf = ds4_gpu_tensor_buffer(scores);
        id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
        if (!scorebuf || !selbuf ||
            ds4_gpu_tensor_bytes(scores) < score_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes) {
            fprintf(stderr, "ds4: Metal graph indexer top-k received undersized buffers\n");
            return 0;
        }
        NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
        if (max_threads == 0) max_threads = 256;
        int32_t nth = 1;
        while ((uint32_t)nth < n_comp && (uint64_t)2u * (uint64_t)nth <= (uint64_t)max_threads) {
            nth *= 2;
        }
        const int32_t npr = (int32_t)((n_comp + (uint32_t)nth - 1u) / (uint32_t)nth);
        const int32_t block_top_k = (int32_t)(top_k < (uint32_t)nth ? top_k : (uint32_t)nth);
        int32_t work_width = (int32_t)top_k;
        if (npr > 1) {
            const int32_t last_block = (int32_t)n_comp - (npr - 1) * nth;
            work_width = (npr - 1) * block_top_k + (last_block < block_top_k ? last_block : block_top_k);
        }
        const uint64_t scratch_row_bytes = (uint64_t)work_width * sizeof(uint32_t);
        const bool one_pass = npr <= 1;
        const uint64_t scratch_bytes = one_pass ? scratch_row_bytes * n_tokens :
            2u * scratch_row_bytes * n_tokens;
        if (!ds4_gpu_ensure_scratch_buffer(&g_indexer_topk_buffer,
                                             &g_indexer_topk_bytes,
                                             (NSUInteger)scratch_bytes,
                                             "ds4_indexer_topk")) {
            return 0;
        }

        ds4_gpu_kargs_argsort args = {
            .ne00 = (int32_t)n_comp,
            .ne01 = (int32_t)n_tokens,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)n_comp * sizeof(float),
            .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
            .ne0 = work_width,
            .ne1 = (int32_t)n_tokens,
            .ne2 = 1,
            .ne3 = 1,
            .top_k = block_top_k,
        };
        const NSUInteger smem = (((NSUInteger)nth * sizeof(int32_t)) + 15u) & ~(NSUInteger)15u;

        NSUInteger cur_off = 0;
        NSUInteger next_off = (NSUInteger)scratch_row_bytes * n_tokens;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_argsort_f32_i32_desc_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
        [enc setBuffer:one_pass ? selbuf : g_indexer_topk_buffer
              offset:one_pass ? ds4_gpu_tensor_offset(selected) : cur_off
             atIndex:2];
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)npr * n_tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake((NSUInteger)nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        int32_t len = block_top_k;
        while (len < work_width) {
            const int32_t nm = (work_width + 2 * len - 1) / (2 * len);
            const bool final_merge = nm == 1;
            NSUInteger merge_threads = g_argsort_merge_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
            if (merge_threads == 0 || merge_threads > 512u) merge_threads = 512u;
            if (merge_threads > (NSUInteger)len) merge_threads = (NSUInteger)len;
            if (merge_threads == 0) merge_threads = 1;

            ds4_gpu_kargs_argsort_merge merge_args = {
                .ne00 = (int64_t)n_comp,
                .ne01 = (int64_t)n_tokens,
                .ne02 = 1,
                .ne03 = 1,
                .nb00 = sizeof(float),
                .nb01 = (uint64_t)n_comp * sizeof(float),
                .nb02 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .nb03 = (uint64_t)n_comp * n_tokens * sizeof(float),
                .ne0 = work_width,
                .ne1 = (int32_t)n_tokens,
                .ne2 = 1,
                .ne3 = 1,
                .top_k = nm == 1 ? (int32_t)top_k : work_width,
                .len = len,
            };

            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:g_argsort_merge_f32_i32_desc_pipeline];
            [enc setBytes:&merge_args length:sizeof(merge_args) atIndex:0];
            [enc setBuffer:scorebuf offset:ds4_gpu_tensor_offset(scores) atIndex:1];
            [enc setBuffer:g_indexer_topk_buffer offset:cur_off atIndex:2];
            [enc setBuffer:final_merge ? selbuf : g_indexer_topk_buffer
                  offset:final_merge ? ds4_gpu_tensor_offset(selected) : next_off
                 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nm * n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(merge_threads, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            const NSUInteger tmp = cur_off;
            cur_off = next_off;
            next_off = tmp;
            len <<= 1;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "indexer top-k")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    @autoreleasepool {
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> maskbuf = ds4_gpu_tensor_buffer(mask);
        if (!topkbuf || !maskbuf ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(mask) < mask_bytes) {
            fprintf(stderr, "ds4: Metal dsv4 top-k mask received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)n_comp,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_comp * sizeof(float),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)n_comp * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_topk_mask_scatter_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
        [enc setBuffer:maskbuf offset:ds4_gpu_tensor_offset(mask) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((((NSUInteger)top_k * n_tokens) + 255u) / 256u, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "dsv4 top-k mask")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if ((in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8_0 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
            if (out_dim > 65536u) mv_dispatch.nsg = 8;
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matvec")) {
                return 0;
            }
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(1, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, 34, row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor mul_mv_ext")) {
                return 0;
            }
            return 1;
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8_0 tensor matmul")) {
            return 0;
        }
    }

    return 1;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!gate || !up || !mid || !x || !model_map ||
        (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < out_bytes ||
            ds4_gpu_tensor_bytes(up) < out_bytes ||
            ds4_gpu_tensor_bytes(mid) < out_bytes) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up received undersized activation buffers\n");
            return 0;
        }

        const uint64_t blocks = in_dim / 32;
        const uint64_t row_bytes = blocks * 34;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
            up_offset > model_size || weight_bytes > model_size - up_offset) {
            fprintf(stderr, "ds4: Metal shared expert fused gate/up range is outside the mapped model\n");
            return 0;
        }

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        id<MTLBuffer> gate_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, weight_bytes, &gate_inner);
        id<MTLBuffer> up_wbuf =
            ds4_gpu_wrap_model_range(model_map, model_size, up_offset, weight_bytes, &up_inner);
        if (!gate_wbuf || !up_wbuf) return 0;

        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_shared_gate_up_swiglu_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gate_wbuf offset:(NSUInteger)gate_inner atIndex:1];
        [enc setBuffer:up_wbuf offset:(NSUInteger)up_inner atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:4];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:5];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(mid) atIndex:6];
        [enc setThreadgroupMemoryLength:2u * mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                                  (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared expert fused gate/up")) {
            return 0;
        }
    }

    return 1;
}

/* ---------------- ANE shared-expert path (synchronous) ---------------- */

/* Forward declarations: implementations live further down in this file. */
static uint16_t ds4_gpu_f32_to_f16_bits(float f);
static float    ds4_gpu_f16_bits_to_f32(uint16_t h);

/* Per-layer cache: dequantized Q8_0 weights as fp16 + the ANE fp16-weight
 * context for this layer.  Created lazily on first call and held for the
 * lifetime of the run.  Memory cost: 3 × in*mid * 2 bytes per layer
 * (~48 MB at H=4096, I=2048 → ~2 GB across 43 layers).  Precision is
 * exact relative to Q8_0 — same math the GPU q8_0 matmul does. */
typedef struct {
    int   initialized;
    int   compile_failed;     /* sticky: don't retry creating the context */
    uint16_t *gate_f16;       /* [mid_dim, in_dim]   fp16 bit-rep */
    uint16_t *up_f16;         /* [mid_dim, in_dim]   fp16 bit-rep */
    uint16_t *down_f16;       /* [in_dim, mid_dim]   fp16 bit-rep */
    uint32_t batch;           /* compiled ANE batch (B) for the fp16w ctx */
    uint64_t in_dim;
    uint64_t mid_dim;
    ds4_ane_mlp_int8w_ctx *ctx;
} ds4_shared_expert_layer_cache;

#define DS4_SHARED_EXPERT_MAX_LAYERS 64
static ds4_shared_expert_layer_cache g_shared_expert_cache[DS4_SHARED_EXPERT_MAX_LAYERS];
static pthread_mutex_t g_shared_expert_cache_mu = PTHREAD_MUTEX_INITIALIZER;

/* Stats — accumulate per-prefill so the operator can see ms spent in the
 * sync path vs the routed-expert async path. */
static double g_shared_ane_sync_total_ms;
static double g_shared_ane_sync_init_ms;
static double g_shared_ane_sync_input_ms;
static double g_shared_ane_sync_eval_ms;
static double g_shared_ane_sync_output_ms;
static uint64_t g_shared_ane_sync_calls;

/* Q8_0 block: fp16 scale (2 bytes) + 32 int8 values (32 bytes) = 34 bytes per
 * 32 elements.  ggml/GGUF stores weights as [n_rows, n_cols] (= [out_dim,
 * in_dim] for matmul Y = X · W^T) but the ANE int8w/fp16w MIL expects the
 * transposed layout [in_dim, out_dim] (matmul Y = X · W).  This function
 * dequantizes Q8_0 AND transposes in one pass: the output buffer holds
 * [n_cols, n_rows] = [in_dim, out_dim] fp16 row-major.  Confirmed against
 * the routed expert path which uses `kernel_dsv4_mpp_dequant_iq2_xxs_
 * transpose_i8` for the same purpose. */
static int ds4_dequant_q8_0_to_f16_transposed(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t n_rows, uint64_t n_cols,
        uint16_t *out_f16) {
    if (!model_map || !out_f16 || (n_cols & 31u) != 0) return 0;
    const uint64_t blocks_per_row = n_cols / 32u;
    const uint64_t row_bytes = blocks_per_row * 34u;
    const uint64_t total_bytes = n_rows * row_bytes;
    if (offset > model_size || total_bytes > model_size - offset) return 0;

    const uint8_t *base = (const uint8_t *)model_map + offset;
    for (uint64_t r = 0; r < n_rows; r++) {
        const uint8_t *row = base + r * row_bytes;
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = row + b * 34u;
            uint16_t s_bits = (uint16_t)(blk[0]) | ((uint16_t)(blk[1]) << 8);
            float s = ds4_gpu_f16_bits_to_f32(s_bits);
            const int8_t *src_q = (const int8_t *)(blk + 2);
            for (uint32_t i = 0; i < 32u; i++) {
                float v = (float)src_q[i] * s;
                const uint64_t col_idx = b * 32u + i;       /* original col */
                const uint64_t dst_idx = col_idx * n_rows + r; /* transposed */
                out_f16[dst_idx] = ds4_gpu_f32_to_f16_bits(v);
            }
        }
    }
    return 1;
}

static int ds4_shared_expert_batch_env(void) {
    const char *e = getenv("DS4_FLASH_MOE_ANE_SHARED_BATCH");
    if (!e || !e[0]) return 256;
    long v = atol(e);
    if (v < 1) v = 1; if (v > 8192) v = 8192;
    return (int)v;
}

/* Lazy initialize the cache slot for `layer_idx`.  Returns the cache entry
 * on success (initialized == 1), or NULL if initialization failed. */
static ds4_shared_expert_layer_cache *ds4_shared_expert_ensure(
        int layer_idx,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t in_dim, uint64_t mid_dim) {
    if (layer_idx < 0 || layer_idx >= DS4_SHARED_EXPERT_MAX_LAYERS) return NULL;
    ds4_shared_expert_layer_cache *c = &g_shared_expert_cache[layer_idx];
    if (c->initialized) return c;
    if (c->compile_failed) return NULL;

    pthread_mutex_lock(&g_shared_expert_cache_mu);
    if (c->initialized) { pthread_mutex_unlock(&g_shared_expert_cache_mu); return c; }
    if (c->compile_failed) { pthread_mutex_unlock(&g_shared_expert_cache_mu); return NULL; }

    const double t0 = ds4_gpu_now_ms();
    int ok = 1;
    const uint64_t gate_elems = in_dim * mid_dim;
    const uint64_t down_elems = mid_dim * in_dim;
    c->gate_f16 = (uint16_t *)malloc((size_t)gate_elems * sizeof(uint16_t));
    c->up_f16   = (uint16_t *)malloc((size_t)gate_elems * sizeof(uint16_t));
    c->down_f16 = (uint16_t *)malloc((size_t)down_elems * sizeof(uint16_t));
    if (!c->gate_f16 || !c->up_f16 || !c->down_f16) {
        ok = 0;
    }
    if (ok) ok = ds4_dequant_q8_0_to_f16_transposed(model_map, model_size,
                                          gate_offset, mid_dim, in_dim,
                                          c->gate_f16);
    if (ok) ok = ds4_dequant_q8_0_to_f16_transposed(model_map, model_size,
                                          up_offset, mid_dim, in_dim,
                                          c->up_f16);
    if (ok) ok = ds4_dequant_q8_0_to_f16_transposed(model_map, model_size,
                                          down_offset, in_dim, mid_dim,
                                          c->down_f16);
    c->batch = (uint32_t)ds4_shared_expert_batch_env();
    c->in_dim = in_dim;
    c->mid_dim = mid_dim;
    /* The ANE context is shape-bound but not weight-bound (weights are passed
     * per _eval call).  All 43 layers have the same fp16w shape (4096×2048),
     * so we share ONE compiled ctx across them — saves ~5 s of one-time ANE
     * compile on the first prefill (was 138 ms × 43 layers). */
    static ds4_ane_mlp_int8w_ctx *s_shared_ctx = NULL;
    static int s_shared_ctx_in_dim = 0;
    static int s_shared_ctx_mid_dim = 0;
    static int s_shared_ctx_batch = 0;
    if (ok) {
        if (s_shared_ctx &&
            s_shared_ctx_in_dim == (int)in_dim &&
            s_shared_ctx_mid_dim == (int)mid_dim &&
            s_shared_ctx_batch == (int)c->batch) {
            c->ctx = s_shared_ctx;
        } else {
            c->ctx = ds4_ane_mlp_fp16w_create((int)in_dim, (int)mid_dim, (int)c->batch);
            if (!c->ctx) ok = 0;
            else {
                s_shared_ctx = c->ctx;
                s_shared_ctx_in_dim = (int)in_dim;
                s_shared_ctx_mid_dim = (int)mid_dim;
                s_shared_ctx_batch = (int)c->batch;
            }
        }
    }
    if (ok) {
        c->initialized = 1;
        g_shared_ane_sync_init_ms += ds4_gpu_now_ms() - t0;
        fprintf(stderr,
                "ds4: ANE shared-expert init layer=%d in=%llu mid=%llu B=%u "
                "fp16-weight mode init_ms=%.1f\n",
                layer_idx, (unsigned long long)in_dim, (unsigned long long)mid_dim,
                c->batch, ds4_gpu_now_ms() - t0);
    } else {
        free(c->gate_f16); free(c->up_f16); free(c->down_f16);
        c->gate_f16 = c->up_f16 = c->down_f16 = NULL;
        c->compile_failed = 1;
        fprintf(stderr, "ds4: ANE shared-expert init FAILED for layer=%d\n", layer_idx);
    }
    pthread_mutex_unlock(&g_shared_expert_cache_mu);
    return ok ? c : NULL;
}

/* Async job state: an ANE worker thread runs the shared-expert eval in
 * parallel with the GPU encoding subsequent ops.  The CB's completion
 * handler fires when GPU has actually written `batch_ffn_norm`; the worker
 * waits on that flag before reading the buffer, then does all CPU-side
 * fp32↔fp16 conversion + ANE eval + writeback into the shared output
 * MTLBuffer.contents().  The caller pthread_joins the worker at the
 * residual-add point.  Compared to the sync impl this eliminates the
 * per-layer ds4_gpu_synchronize wait (which was structurally the cost). */
typedef struct ds4_gpu_shared_expert_ane_job {
    ds4_shared_expert_layer_cache *cache;
    int layer_idx;
    uint32_t n_tokens;
    uint64_t in_dim;
    uint64_t mid_dim;

    /* Pointers into MTLBuffer.contents() (CPU/GPU coherent on Apple Silicon
     * shared-storage buffers).  Worker reads input after dep_done; writes
     * output before signaling eval_done. */
    const float *in_f32_ptr;
    float       *out_f32_ptr;

    /* Worker-owned scratch — freed at finish. */
    uint16_t *x_f16_all;
    uint16_t *x_f16_pad;
    uint16_t *y_f16_pad;

    pthread_t thread;
    int thread_started;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int dep_done;       /* GPU has written batch_ffn_norm */
    int eval_done;
    int eval_failed;

    double t_start_ms;
} ds4_gpu_shared_expert_ane_job;

static void *ds4_gpu_shared_expert_ane_worker(void *arg) {
    ds4_gpu_shared_expert_ane_job *job = (ds4_gpu_shared_expert_ane_job *)arg;
    if (!job) return NULL;

    /* Wait for GPU to finish writing batch_ffn_norm. */
    pthread_mutex_lock(&job->mu);
    while (!job->dep_done) {
        pthread_cond_wait(&job->cv, &job->mu);
    }
    pthread_mutex_unlock(&job->mu);

    const double t_in0 = ds4_gpu_now_ms();
    const uint64_t n_elems = (uint64_t)job->n_tokens * job->in_dim;
    for (uint64_t i = 0; i < n_elems; i++) {
        job->x_f16_all[i] = ds4_gpu_f32_to_f16_bits(job->in_f32_ptr[i]);
    }
    g_shared_ane_sync_input_ms += ds4_gpu_now_ms() - t_in0;

    const uint32_t B = job->cache->batch;
    int eval_ok = 1;
    for (uint32_t begin = 0; begin < job->n_tokens && eval_ok; begin += B) {
        const uint32_t chunk = (job->n_tokens - begin) < B ? (job->n_tokens - begin) : B;
        memset(job->x_f16_pad, 0, (size_t)B * (size_t)job->in_dim * sizeof(uint16_t));
        memcpy(job->x_f16_pad,
               job->x_f16_all + (uint64_t)begin * job->in_dim,
               (size_t)chunk * (size_t)job->in_dim * sizeof(uint16_t));
        memset(job->y_f16_pad, 0, (size_t)B * (size_t)job->in_dim * sizeof(uint16_t));

        const double t_eval0 = ds4_gpu_now_ms();
        if (!ds4_ane_mlp_fp16w_eval(job->cache->ctx,
                                     job->cache->gate_f16, job->cache->up_f16,
                                     job->cache->down_f16,
                                     job->x_f16_pad, job->y_f16_pad)) {
            eval_ok = 0;
            break;
        }
        g_shared_ane_sync_eval_ms += ds4_gpu_now_ms() - t_eval0;

        const double t_out0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            const uint16_t *src = job->y_f16_pad + (uint64_t)r * job->in_dim;
            float          *dst = job->out_f32_ptr + (uint64_t)(begin + r) * job->in_dim;
            for (uint64_t i = 0; i < job->in_dim; i++) {
                dst[i] = ds4_gpu_f16_bits_to_f32(src[i]);
            }
        }
        g_shared_ane_sync_output_ms += ds4_gpu_now_ms() - t_out0;
    }

    pthread_mutex_lock(&job->mu);
    job->eval_done = 1;
    job->eval_failed = !eval_ok;
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    return NULL;
}

ds4_gpu_shared_expert_ane_job *ds4_gpu_shared_expert_ane_async_start_tensor(
        const ds4_gpu_tensor *in,
        ds4_gpu_tensor       *out,
        int                     layer_idx,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                in_dim,
        uint64_t                mid_dim,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (!in || !out || n_tokens == 0) return NULL;
    if (in_dim > UINT32_MAX || mid_dim > UINT32_MAX) return NULL;

    ds4_shared_expert_layer_cache *c = ds4_shared_expert_ensure(
        layer_idx, model_map, model_size, gate_offset, up_offset, down_offset,
        in_dim, mid_dim);
    if (!c) return NULL;

    id<MTLBuffer> inbuf  = ds4_gpu_tensor_buffer(in);
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!inbuf || !outbuf) return NULL;
    const uint64_t in_bytes  = (uint64_t)n_tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * in_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(in) < in_bytes || ds4_gpu_tensor_bytes(out) < out_bytes) {
        return NULL;
    }

    ds4_gpu_shared_expert_ane_job *job =
        (ds4_gpu_shared_expert_ane_job *)calloc(1, sizeof(*job));
    if (!job) return NULL;
    job->cache = c;
    job->layer_idx = layer_idx;
    job->n_tokens = n_tokens;
    job->in_dim = in_dim;
    job->mid_dim = mid_dim;
    job->in_f32_ptr  = (const float *)((const uint8_t *)[inbuf  contents] + ds4_gpu_tensor_offset(in));
    job->out_f32_ptr = (float       *)((uint8_t       *)[outbuf contents] + ds4_gpu_tensor_offset(out));
    pthread_mutex_init(&job->mu, NULL);
    pthread_cond_init(&job->cv, NULL);
    job->t_start_ms = ds4_gpu_now_ms();
    g_shared_ane_sync_calls++;

    const uint32_t B = c->batch;
    job->x_f16_all = (uint16_t *)malloc((size_t)n_tokens * (size_t)in_dim * sizeof(uint16_t));
    job->x_f16_pad = (uint16_t *)malloc((size_t)B        * (size_t)in_dim * sizeof(uint16_t));
    job->y_f16_pad = (uint16_t *)malloc((size_t)B        * (size_t)in_dim * sizeof(uint16_t));
    if (!job->x_f16_all || !job->x_f16_pad || !job->y_f16_pad ||
        !job->in_f32_ptr || !job->out_f32_ptr) {
        free(job->x_f16_all); free(job->x_f16_pad); free(job->y_f16_pad);
        pthread_mutex_destroy(&job->mu); pthread_cond_destroy(&job->cv);
        free(job);
        return NULL;
    }

    /* Attach completion handler to current CB so worker fires when GPU
     * finishes producing batch_ffn_norm.  Then flush to commit the CB
     * (the handler can only fire after commit).  This does NOT wait. */
    if (g_batch_cb == nil) {
        /* No current CB — caller is in a weird state.  Bail. */
        free(job->x_f16_all); free(job->x_f16_pad); free(job->y_f16_pad);
        pthread_mutex_destroy(&job->mu); pthread_cond_destroy(&job->cv);
        free(job);
        return NULL;
    }
    ds4_gpu_shared_expert_ane_job *job_for_handler = job;
    [g_batch_cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull cb_arg){
        (void)cb_arg;
        pthread_mutex_lock(&job_for_handler->mu);
        job_for_handler->dep_done = 1;
        pthread_cond_broadcast(&job_for_handler->cv);
        pthread_mutex_unlock(&job_for_handler->mu);
    }];
    if (!ds4_gpu_flush_commands()) {
        /* Mark dep_done anyway so worker doesn't hang. */
        pthread_mutex_lock(&job->mu);
        job->dep_done = 1;
        job->eval_failed = 1;
        pthread_cond_broadcast(&job->cv);
        pthread_mutex_unlock(&job->mu);
        free(job->x_f16_all); free(job->x_f16_pad); free(job->y_f16_pad);
        pthread_mutex_destroy(&job->mu); pthread_cond_destroy(&job->cv);
        free(job);
        return NULL;
    }

    if (pthread_create(&job->thread, NULL,
                        ds4_gpu_shared_expert_ane_worker, job) != 0) {
        free(job->x_f16_all); free(job->x_f16_pad); free(job->y_f16_pad);
        pthread_mutex_destroy(&job->mu); pthread_cond_destroy(&job->cv);
        free(job);
        return NULL;
    }
    job->thread_started = 1;
    return job;
}

int ds4_gpu_shared_expert_ane_async_finish_tensor(
        ds4_gpu_shared_expert_ane_job *job) {
    if (!job) return 0;
    if (job->thread_started) {
        pthread_join(job->thread, NULL);
        job->thread_started = 0;
    }
    int ok = !job->eval_failed && job->eval_done;
    g_shared_ane_sync_total_ms += ds4_gpu_now_ms() - job->t_start_ms;
    free(job->x_f16_all);
    free(job->x_f16_pad);
    free(job->y_f16_pad);
    pthread_mutex_destroy(&job->mu);
    pthread_cond_destroy(&job->cv);
    free(job);
    return ok ? 1 : 0;
}

/* Pre-warm: pre-dequantize Q8_0 weights to fp16 and create the shared ANE
 * context.  Called from metal_graph_warmup_prefill_kernels so the ~4 s of
 * one-time init is paid before the prefill timer starts.  Idempotent. */
int ds4_gpu_shared_expert_ane_prewarm(
        int                     layer_idx,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                in_dim,
        uint64_t                mid_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    ds4_shared_expert_layer_cache *c = ds4_shared_expert_ensure(
        layer_idx, model_map, model_size, gate_offset, up_offset, down_offset,
        in_dim, mid_dim);
    return c ? 1 : 0;
}

/* Synchronous wrapper kept for callers that don't want the async lifecycle.
 * Identical observable behaviour to the previous _sync_tensor impl. */
int ds4_gpu_shared_expert_ane_sync_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *in,
        int                     layer_idx,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                in_dim,
        uint64_t                mid_dim,
        uint32_t                n_tokens) {
    ds4_gpu_shared_expert_ane_job *job =
        ds4_gpu_shared_expert_ane_async_start_tensor(in, out, layer_idx,
                                                       model_map, model_size,
                                                       gate_offset, up_offset, down_offset,
                                                       in_dim, mid_dim, n_tokens);
    if (!job) return 0;
    return ds4_gpu_shared_expert_ane_async_finish_tensor(job);
}

/* Legacy sync impl body kept below for reference but no longer reached. */
static int ds4_gpu_shared_expert_ane_sync_tensor_LEGACY(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *in,
        int                     layer_idx,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint64_t                in_dim,
        uint64_t                mid_dim,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !in || n_tokens == 0) return 0;
    if (in_dim > UINT32_MAX || mid_dim > UINT32_MAX) return 0;

    const double t_total0 = ds4_gpu_now_ms();
    g_shared_ane_sync_calls++;

    ds4_shared_expert_layer_cache *c = ds4_shared_expert_ensure(
        layer_idx, model_map, model_size, gate_offset, up_offset, down_offset,
        in_dim, mid_dim);
    if (!c) return 0;

    /* Flush pending GPU work and wait so `in` (batch_ffn_norm) is fully
     * written before we read its bytes. */
    if (g_batch_cb != nil) {
        if (!ds4_gpu_flush_commands()) return 0;
    }
    if (!ds4_gpu_synchronize()) return 0;

    /* Read input fp32 buffer + scratch i8 input buffer + scratch fp16 output. */
    id<MTLBuffer> inbuf  = ds4_gpu_tensor_buffer(in);
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!inbuf || !outbuf) return 0;
    const uint64_t in_bytes  = (uint64_t)n_tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * in_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(in)  < in_bytes ||
        ds4_gpu_tensor_bytes(out) < out_bytes) return 0;

    const float *x_f32 = (const float *)((const uint8_t *)[inbuf contents] + ds4_gpu_tensor_offset(in));
    float       *y_f32 = (float       *)((uint8_t       *)[outbuf contents] + ds4_gpu_tensor_offset(out));
    if (!x_f32 || !y_f32) return 0;

    const double t_in0 = ds4_gpu_now_ms();
    /* Convert all fp32 input rows to fp16 once.  No quantization needed in
     * fp16-weight ANE mode. */
    uint16_t *x_f16_all = (uint16_t *)malloc((size_t)n_tokens * (size_t)in_dim * sizeof(uint16_t));
    if (!x_f16_all) return 0;
    for (uint64_t i = 0; i < (uint64_t)n_tokens * in_dim; i++) {
        x_f16_all[i] = ds4_gpu_f32_to_f16_bits(x_f32[i]);
    }
    g_shared_ane_sync_input_ms += ds4_gpu_now_ms() - t_in0;

    /* Run ANE in chunks of B (compiled batch).  Need padded input + output
     * scratch sized to B rows. */
    const uint32_t B = c->batch;
    uint16_t *x_f16_pad = (uint16_t *)malloc((size_t)B * (size_t)in_dim * sizeof(uint16_t));
    uint16_t *y_f16_pad = (uint16_t *)malloc((size_t)B * (size_t)in_dim * sizeof(uint16_t));
    if (!x_f16_pad || !y_f16_pad) { free(x_f16_all); free(x_f16_pad); free(y_f16_pad); return 0; }

    int eval_ok = 1;
    for (uint32_t begin = 0; begin < n_tokens && eval_ok; begin += B) {
        const uint32_t chunk = (n_tokens - begin) < B ? (n_tokens - begin) : B;
        memset(x_f16_pad, 0, (size_t)B * (size_t)in_dim * sizeof(uint16_t));
        memcpy(x_f16_pad, x_f16_all + (uint64_t)begin * in_dim,
               (size_t)chunk * (size_t)in_dim * sizeof(uint16_t));
        memset(y_f16_pad, 0, (size_t)B * (size_t)in_dim * sizeof(uint16_t));

        const double t_eval0 = ds4_gpu_now_ms();
        if (!ds4_ane_mlp_fp16w_eval(c->ctx,
                                     c->gate_f16, c->up_f16, c->down_f16,
                                     x_f16_pad, y_f16_pad)) {
            eval_ok = 0;
            break;
        }
        g_shared_ane_sync_eval_ms += ds4_gpu_now_ms() - t_eval0;

        /* Dequantize fp16 → fp32 into out buffer at the chunk offset.  Output
         * dim is `in_dim` (=N_EMBD) — same as input. */
        const double t_out0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            const uint16_t *src = y_f16_pad + (uint64_t)r * in_dim;
            float          *dst = y_f32     + (uint64_t)(begin + r) * in_dim;
            for (uint64_t i = 0; i < in_dim; i++) {
                dst[i] = ds4_gpu_f16_bits_to_f32(src[i]);
            }
        }
        g_shared_ane_sync_output_ms += ds4_gpu_now_ms() - t_out0;
    }

    free(x_f16_all);
    free(x_f16_pad);
    free(y_f16_pad);

    g_shared_ane_sync_total_ms += ds4_gpu_now_ms() - t_total0;

    /* Reopen a GPU command batch for the next encoders. */
    if (g_batch_cb == nil) {
        if (ds4_gpu_begin_commands() == 0) return 0;
    }
    return eval_ok;
}

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F16 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (n_tok == 1) {
            ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
            ds4_gpu_mv_dispatch mv_dispatch =
                ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
            if (!g_quality_mode && (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
                mv_dispatch.nr0 = 4;
                mv_dispatch.smem = 32u * 4u * sizeof(float);
            }
            mv_args.nr0 = mv_dispatch.nr0;
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
            if (!pipeline) return 0;

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            if (mv_dispatch.smem) {
                [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
            }
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                                  1,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matvec")) return 0;
            return 1;
        }

        if (n_tok <= 8 && (in_dim % 128u) == 0) {
            const int16_t nsg = 2;
            const int16_t nxpsg = ds4_gpu_mv_ext_nxpsg(in_dim, n_tok);
            const int16_t r1ptg = ds4_gpu_mv_ext_r1ptg(n_tok);
            const char *fn_name = ds4_gpu_mv_ext_name(0, r1ptg);
            id<MTLComputePipelineState> pipeline =
                fn_name ? ds4_gpu_get_mul_mv_ext_pipeline(fn_name, nsg, nxpsg) : nil;
            if (!pipeline) return 0;

            const int16_t nypsg = 32 / nxpsg;
            const uint64_t r0ptg = (uint64_t)nypsg * (uint64_t)nsg;
            ds4_gpu_mul_mv_ext_args args =
                ds4_gpu_make_mv_ext_args(in_dim, out_dim, n_tok, sizeof(uint16_t), row_bytes);

            id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
            [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
            [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)r0ptg - 1u) / (NSUInteger)r0ptg,
                                                  ((NSUInteger)n_tok + (NSUInteger)r1ptg - 1u) / (NSUInteger)r1ptg,
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);

            if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor mul_mv_ext")) return 0;
            return 1;
        }

        const bool bc_inp = (in_dim % 32u) != 0;
        const bool bc_out = (out_dim % 64u) != 0 || (n_tok % 32u) != 0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mm_pipeline("kernel_mul_mm_f16_f32", bc_inp, bc_out);
        if (!pipeline) return 0;

        ds4_gpu_mul_mm_args args = ds4_gpu_make_mm_args(in_dim, out_dim, n_tok, row_bytes);

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_tok + 31u) / 32u,
                                              ((NSUInteger)out_dim + 63u) / 64u,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 tensor matmul")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok != 1 || (in_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outabuf = ds4_gpu_tensor_buffer(out_a);
        id<MTLBuffer> outbbuf = ds4_gpu_tensor_buffer(out_b);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outabuf || !outbbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out_a) < out_bytes ||
            ds4_gpu_tensor_bytes(out_b) < out_bytes) {
            fprintf(stderr, "ds4: Metal F16 paired matvec received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(uint16_t);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_a_offset > model_size || weight_bytes > model_size - weight_a_offset ||
            weight_b_offset > model_size || weight_bytes > model_size - weight_b_offset) {
            fprintf(stderr, "ds4: Metal F16 paired matvec range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_a = 0;
        uint64_t inner_b = 0;
        id<MTLBuffer> wabuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_a_offset, weight_bytes,
                                                         &inner_a);
        id<MTLBuffer> wbbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                         weight_b_offset, weight_bytes,
                                                         &inner_b);
        if (!wabuf || !wbbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_f16_matvec_args mv_args = ds4_gpu_make_f16_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 0);
        if (ds4_gpu_use_compressor_pair_nr4() &&
            (out_dim == 512u || out_dim == 1024u) && in_dim >= 4096u) {
            mv_dispatch.nr0 = 4;
            mv_dispatch.smem = 32u * 4u * sizeof(float);
        }
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_f16_f32_pair_4", mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wabuf offset:(NSUInteger)inner_a atIndex:1];
        [enc setBuffer:wbbuf offset:(NSUInteger)inner_b atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outabuf offset:ds4_gpu_tensor_offset(out_a) atIndex:4];
        [enc setBuffer:outbbuf offset:ds4_gpu_tensor_offset(out_b) atIndex:5];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F16 paired matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX || n_tok != 1) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t out_bytes = out_dim * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul received undersized activation buffers\n");
            return 0;
        }

        const uint64_t row_bytes = in_dim * sizeof(float);
        const uint64_t weight_bytes = row_bytes * out_dim;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal F32 tensor matmul range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        if (!wbuf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_f32_mv_args(in_dim, out_dim, 1);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_plain_mv_dispatch(in_dim, 1);
        mv_args.nr0 = mv_dispatch.nr0;
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline(mv_dispatch.function_name, mv_dispatch.nsg);
        if (!pipeline) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        if (mv_dispatch.smem) {
            [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        }
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) / (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "F32 tensor matvec")) return 0;
    }

    return 1;
}

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> rowbuf = ds4_gpu_tensor_buffer(row);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_bytes = row_bytes * n_hc;
        if (!rowbuf || !outbuf ||
            ds4_gpu_tensor_bytes(row) < row_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal HC repeat received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        if (!ds4_gpu_encode_repeat_hc_embedding(cb,
                                                  rowbuf,
                                                  ds4_gpu_tensor_offset(row),
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  1,
                                                  n_embd,
                                                  n_hc)) {
            return 0;
        }
        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC repeat")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal plain RMS norm received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "plain RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        const uint64_t bytes = row_bytes * rows;
        if (!xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal weighted RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (weight_offset > model_size || row_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal weighted RMS norm range is outside the mapped model\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, row_bytes, &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_args(n, rows, eps);
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "weighted RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_out || !kv || q_n == 0 || kv_n == 0 || rows == 0 ||
        (q_n & 3u) != 0 || (kv_n & 3u) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> qoutbuf = ds4_gpu_tensor_buffer(q_out);
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> kvoutbuf = ds4_gpu_tensor_buffer(kv_out);

        const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
        const uint64_t kv_row_bytes = (uint64_t)kv_n * sizeof(float);
        if (!qbuf || !qoutbuf || !kvbuf || !kvoutbuf ||
            ds4_gpu_tensor_bytes(q) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(q_out) < q_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv) < kv_row_bytes * rows ||
            ds4_gpu_tensor_bytes(kv_out) < kv_row_bytes * rows) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm received undersized activation buffers\n");
            return 0;
        }
        if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
            kv_weight_offset > model_size || kv_row_bytes > model_size - kv_weight_offset) {
            fprintf(stderr, "ds4: Metal fused q/kv RMS norm weight range is outside the mapped model\n");
            return 0;
        }

        uint64_t q_inner_offset = 0;
        uint64_t kv_inner_offset = 0;
        id<MTLBuffer> q_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                          q_weight_offset, q_row_bytes,
                                                          &q_inner_offset);
        if (!q_wbuf) return 0;
        id<MTLBuffer> kv_wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           kv_weight_offset, kv_row_bytes,
                                                           &kv_inner_offset);
        if (!kv_wbuf) return 0;

        ds4_gpu_qkv_rms_norm_args args = {
            .q_n = (int32_t)q_n,
            .q_n4 = (int32_t)(q_n / 4u),
            .kv_n = (int32_t)kv_n,
            .kv_n4 = (int32_t)(kv_n / 4u),
            .q_row_stride = q_row_bytes,
            .kv_row_stride = kv_row_bytes,
            .eps = eps,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_qkv_rms_norm_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:q_wbuf offset:(NSUInteger)q_inner_offset atIndex:2];
        [enc setBuffer:qoutbuf offset:ds4_gpu_tensor_offset(q_out) atIndex:3];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:4];
        [enc setBuffer:kv_wbuf offset:(NSUInteger)kv_inner_offset atIndex:5];
        [enc setBuffer:kvoutbuf offset:ds4_gpu_tensor_offset(kv_out) atIndex:6];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 2, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_threads(q_n), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "fused q/kv RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal head RMS norm received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rms_norm_args args = ds4_gpu_make_rms_norm_3d_args(head_dim, n_head, n_tok, eps);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_rms_norm_plain_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_head, n_tok, 1)
             threadsPerThreadgroup:MTLSizeMake(ds4_gpu_rms_norm_pipeline_threads(head_dim, g_rms_norm_plain_pipeline), 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "head RMS norm")) return 0;
    }

    return 1;
}

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }
    if (n_rot == 0) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal RoPE received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_rope_tail_batch_args args = ds4_gpu_make_rope_tail_args(
            n_tok, n_head, head_dim, n_rot, n_ctx_orig, inverse,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_rope_tail_inplace(cb,
                                                xbuf,
                                                ds4_gpu_tensor_offset(x),
                                                &args,
                                                n_tok,
                                                n_head,
                                                head_dim,
                                                pos0,
                                                1)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "RoPE tail")) return 0;
    }

    return 1;
}

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (n_rot == head_dim) return 1;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        const uint64_t bytes = (uint64_t)n_tok * head_dim * sizeof(float);
        if (!xbuf || ds4_gpu_tensor_bytes(x) < bytes) {
            fprintf(stderr, "ds4: Metal DSV4 FP8 KV quantize received undersized activation buffer\n");
            return 0;
        }

        ds4_gpu_dsv4_fp8_kv_quantize_args args = {
            .ne00 = head_dim,
            .ne01 = n_tok,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = (uint64_t)head_dim * sizeof(float),
            .nb02 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb03 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)head_dim * sizeof(float),
            .nb2 = (uint64_t)n_tok * head_dim * sizeof(float),
            .nb3 = (uint64_t)n_tok * head_dim * sizeof(float),
            .n_rot = (int32_t)n_rot,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_fp8_kv_quantize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "DSV4 FP8 KV quantize")) return 0;
    }

    return 1;
}

static void ds4_gpu_set_rows_thread_shape(
        uint32_t    width,
        NSUInteger *nth_out,
        NSUInteger *nrptg_out) {
    const NSUInteger nk0 = width ? (NSUInteger)width : 1u;
    const NSUInteger max_threads = g_set_rows_f32_i32_pipeline
        ? (NSUInteger)g_set_rows_f32_i32_pipeline.maxTotalThreadsPerThreadgroup
        : 1024u;

    NSUInteger nth = 32u;
    while (nth < nk0 && nth < max_threads) {
        nth *= 2u;
    }

    NSUInteger nrptg = 1u;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1u) / nk0;
        nth = nk0;
        if (nrptg * nth > max_threads) {
            nrptg--;
        }
    }

    if (nth > nk0) nth = nk0;
    if (nth == 0u) nth = 1u;
    if (nrptg == 0u) nrptg = 1u;

    *nth_out = nth;
    *nrptg_out = nrptg;
}

static int ds4_gpu_encode_f16_round_copy_for_raw_store(
        id<MTLCommandBuffer>   cb,
        const ds4_gpu_tensor *src,
        uint32_t               n) {
    id<MTLBuffer> srcbuf = ds4_gpu_tensor_buffer(src);
    const uint64_t src_bytes = (uint64_t)n * sizeof(float);
    if (!srcbuf || ds4_gpu_tensor_bytes(src) < src_bytes) {
        fprintf(stderr, "ds4: Metal raw KV store received undersized source buffer\n");
        return 0;
    }
    if (!ds4_gpu_ensure_scratch_buffer(&g_f16_round_scratch_buffer,
                                         &g_f16_round_scratch_bytes,
                                         (NSUInteger)n * sizeof(uint16_t),
                                         "ds4_f16_round_scratch") ||
        !ds4_gpu_ensure_scratch_buffer(&g_raw_store_round_buffer,
                                         &g_raw_store_round_bytes,
                                         (NSUInteger)n * sizeof(float),
                                         "ds4_raw_store_round")) {
        return 0;
    }

    ds4_gpu_cpy_args f32_to_f16 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    ds4_gpu_cpy_args f16_to_f32 =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth_f32_f16 = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger nth_f16_f32 = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups_f32_f16 = ((NSUInteger)n + nth_f32_f16 - 1u) / nth_f32_f16;
    const NSUInteger groups_f16_f32 = ((NSUInteger)n + nth_f16_f32 - 1u) / nth_f16_f32;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&f32_to_f16 length:sizeof(f32_to_f16) atIndex:0];
    [enc setBuffer:srcbuf offset:ds4_gpu_tensor_offset(src) atIndex:1];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f32_f16, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f32_f16, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&f16_to_f32 length:sizeof(f16_to_f32) atIndex:0];
    [enc setBuffer:g_f16_round_scratch_buffer offset:0 atIndex:1];
    [enc setBuffer:g_raw_store_round_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups_f16_f32, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth_f16_f32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_set_rows_f32_i32(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *dst,
        id<MTLBuffer>        srcbuf,
        NSUInteger           src_off,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             dst_rows,
        uint32_t             width) {
    id<MTLBuffer> dstbuf = ds4_gpu_tensor_buffer(dst);
    const uint64_t dst_bytes = (uint64_t)dst_rows * width * sizeof(float);
    const uint64_t src_bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!dstbuf || !srcbuf || !rows || n_rows == 0 || width == 0 ||
        ds4_gpu_tensor_bytes(dst) < dst_bytes ||
        src_bytes > NSUIntegerMax - src_off) {
        fprintf(stderr, "ds4: Metal DS4 set_rows received invalid buffers\n");
        return 0;
    }

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rows_bytes = (uint64_t)n_rows * sizeof(int32_t);
    ds4_gpu_set_rows_args args = {
        .nk0 = (int32_t)width,
        .ne01 = (int32_t)n_rows,
        .nb01 = row_bytes,
        .nb02 = (uint64_t)n_rows * row_bytes,
        .nb03 = (uint64_t)n_rows * row_bytes,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(int32_t),
        .nb11 = rows_bytes,
        .nb12 = rows_bytes,
        .nb1 = row_bytes,
        .nb2 = (uint64_t)dst_rows * row_bytes,
        .nb3 = (uint64_t)dst_rows * row_bytes,
    };

    NSUInteger nth;
    NSUInteger nrptg;
    ds4_gpu_set_rows_thread_shape(width, &nth, &nrptg);

    id<MTLBuffer> rowsbuf = nil;
    if (rows_bytes > 4096u) {
        rowsbuf = ds4_gpu_new_transient_buffer((NSUInteger)rows_bytes, "ds4_set_rows_indices");
        if (!rowsbuf) return 0;
        memcpy([rowsbuf contents], rows, (NSUInteger)rows_bytes);
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_set_rows_f32_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:srcbuf offset:src_off atIndex:1];
    if (rowsbuf) {
        [enc setBuffer:rowsbuf offset:0 atIndex:2];
    } else {
        [enc setBytes:rows length:(NSUInteger)rows_bytes atIndex:2];
    }
    [enc setBuffer:dstbuf offset:ds4_gpu_tensor_offset(dst) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n_rows + nrptg - 1u) / nrptg, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_add_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        a,
        NSUInteger           a_off,
        id<MTLBuffer>        b,
        NSUInteger           b_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !a || !b || !out || n == 0) return 0;

    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    ds4_gpu_bin_args args = {
        .ne00 = (int32_t)n,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = row_bytes,
        .nb03 = row_bytes,
        .ne10 = (int32_t)n,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = row_bytes,
        .nb13 = row_bytes,
        .ne0 = (int32_t)n,
        .ne1 = 1,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = row_bytes,
        .nb3 = row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };

    NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)n && nth < nth_max) {
        nth *= 2u;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_add_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV store received undersized destination buffer\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const int32_t row_i32 = (int32_t)row;
        if (!ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, head_dim) ||
            !ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               &row_i32,
                                               1,
                                               raw_cap,
                                               head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV DS4 set_rows store")) return 0;
    }

    return 1;
}

/* Release decode fused KV finalizer.  Reference paths are selected by the C
 * graph driver; this Objective-C entry point always means "use the fused
 * Metal kernel." */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap || head_dim == 0 ||
        n_rot > head_dim || raw_cap > INT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_cache);
        const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (!kvbuf || !rawbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal fused KV FP8/raw-store received undersized buffers\n");
            return 0;
        }

        ds4_gpu_dsv4_kv_fp8_store_args args = {
            .head_dim = (int32_t)head_dim,
            .n_rot = (int32_t)n_rot,
            .raw_row = (int32_t)row,
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_dsv4_kv_fp8_store_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_cache) atIndex:2];
        [enc setThreadgroupMemoryLength:64u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "KV FP8/raw-store fused")) return 0;
    }

    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0 || raw_cap > INT32_MAX) return 0;

    @autoreleasepool {
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(raw_cache) < raw_bytes) {
            fprintf(stderr, "ds4: Metal raw KV batch store received undersized destination buffer\n");
            return 0;
        }

        int32_t rows_stack[512];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate raw KV set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            rows[t] = (int32_t)((pos0 + t) % raw_cap);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        const uint64_t n = (uint64_t)n_tokens * head_dim;
        const int ok = n <= UINT32_MAX &&
            ds4_gpu_encode_f16_round_copy_for_raw_store(cb, kv, (uint32_t)n) &&
            ds4_gpu_encode_set_rows_f32_i32(cb, raw_cache,
                                               g_raw_store_round_buffer,
                                               0,
                                               rows,
                                               n_tokens,
                                               raw_cap,
                                               head_dim);
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "raw KV batch DS4 set_rows store")) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_compressor_score_with_ape(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        score_src,
        NSUInteger           score_src_offset,
        id<MTLBuffer>        score_dst,
        NSUInteger           score_dst_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        uint32_t             n_tokens) {
    if (!cb || !score_src || !score_dst || !apebuf ||
        width == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    const uint64_t total_elems64 = (uint64_t)n_tokens * width;
    if (total_elems64 > UINT32_MAX) {
        fprintf(stderr, "ds4: Metal compressor APE add received too many elements\n");
        return 0;
    }
    const uint32_t total_elems = (uint32_t)total_elems64;
    const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                         &g_compressor_store_ape_bytes,
                                         scratch_bytes,
                                         "ds4_compressor_store_ape")) {
        return 0;
    }

    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    uint32_t copied_rows = 0;
    uint32_t pos_mod = pos0 % ratio;
    while (copied_rows < n_tokens) {
        uint32_t seg_rows = ratio - pos_mod;
        if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
        const uint32_t seg_elems = seg_rows * width;
        const NSUInteger src_off = ape_offset + (NSUInteger)pos_mod * width * elem_ape;
        const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
        int ok;
        if (ape_type == 1u) {
            ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        } else {
            ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                 apebuf,
                                                 src_off,
                                                 g_compressor_store_ape_buffer,
                                                 dst_off,
                                                 seg_elems);
        }
        if (!ok) return 0;
        copied_rows += seg_rows;
        pos_mod = 0;
    }

    return ds4_gpu_encode_add_f32_1d(cb,
                                       score_src,
                                       score_src_offset,
                                       g_compressor_store_ape_buffer,
                                       0,
                                       score_dst,
                                       score_dst_offset,
                                       total_elems);
}

static int ds4_gpu_encode_compressor_set_rows_projected(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        id<MTLBuffer>        apebuf,
        NSUInteger           ape_offset,
        uint32_t             ape_type,
        uint32_t             width,
        uint32_t             ratio,
        uint32_t             pos0,
        const int32_t       *rows,
        uint32_t             n_rows,
        uint32_t             state_rows) {
    if (!cb || !state_kv || !state_score || !kvbuf || !scorebuf ||
        !apebuf || !rows || width == 0 || n_rows == 0 || state_rows == 0) {
        return 0;
    }

    const NSUInteger score_scratch_bytes = (NSUInteger)n_rows * width * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                         &g_compressor_store_score_bytes,
                                         score_scratch_bytes,
                                         "ds4_compressor_store_score")) {
        return 0;
    }

    return ds4_gpu_encode_compressor_score_with_ape(cb,
                                                      scorebuf,
                                                      score_offset,
                                                      g_compressor_store_score_buffer,
                                                      0,
                                                      apebuf,
                                                      ape_offset,
                                                      ape_type,
                                                      width,
                                                      ratio,
                                                      pos0,
                                                      n_rows) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_kv,
                                             kvbuf,
                                             kv_offset,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width) &&
           ds4_gpu_encode_set_rows_f32_i32(cb,
                                             state_score,
                                             g_compressor_store_score_buffer,
                                             0,
                                             rows,
                                             n_rows,
                                             state_rows,
                                             width);
}

static int ds4_gpu_compressor_store_one_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                width,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        width == 0 || ratio == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_hot_pipeline(g_dsv4_compressor_store_one_pipeline,
                                "kernel_dsv4_compressor_store_one");
    if (!pipeline) return 0;

    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * row_bytes;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        ds4_gpu_tensor_bytes(kv) < row_bytes ||
        ds4_gpu_tensor_bytes(sc) < row_bytes ||
        ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    uint64_t ape_inner = 0;
    id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                       ape_offset, ape_bytes,
                                                       &ape_inner);
    id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
    id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!apebuf || !kvbuf || !scbuf || !statekvbuf || !statescbuf) return 0;

    ds4_gpu_dsv4_compressor_store_one_args args = {
        .width = width,
        .ratio = ratio,
        .pos = pos,
        .ape_type = ape_type,
    };

    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;

    const NSUInteger nth = 256u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:ds4_gpu_tensor_offset(kv) atIndex:1];
    [enc setBuffer:scbuf offset:ds4_gpu_tensor_offset(sc) atIndex:2];
    [enc setBuffer:apebuf offset:(NSUInteger)ape_inner atIndex:3];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:4];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)width + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return ds4_gpu_finish_command_buffer(cb, owned, "compressor one-row store");
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor batch APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        if (!kvbuf || !scbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor batch store received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const uint64_t total_elems64 = (uint64_t)n_tokens * width;
        if (total_elems64 > UINT32_MAX || state_rows > INT32_MAX) {
            fprintf(stderr, "ds4: Metal compressor batch store received too many elements\n");
            return 0;
        }
        const uint32_t total_elems = (uint32_t)total_elems64;
        const NSUInteger scratch_bytes = (NSUInteger)total_elems * sizeof(float);
        if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_ape_buffer,
                                             &g_compressor_store_ape_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_ape") ||
            !ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                             &g_compressor_store_score_bytes,
                                             scratch_bytes,
                                             "ds4_compressor_store_score")) {
            return 0;
        }

        int32_t rows_stack[16];
        int32_t *rows = rows_stack;
        if (n_tokens > (uint32_t)(sizeof(rows_stack) / sizeof(rows_stack[0]))) {
            rows = malloc((size_t)n_tokens * sizeof(*rows));
            if (!rows) {
                fprintf(stderr, "ds4: failed to allocate compressor set_rows index list\n");
                return 0;
            }
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t pos_mod = (pos0 + t) % ratio;
            rows[t] = (int32_t)(ratio == 4u ? ratio + pos_mod : pos_mod);
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) {
            if (rows != rows_stack) free(rows);
            return 0;
        }

        int ok = 1;
        uint32_t copied_rows = 0;
        uint32_t pos_mod = pos0 % ratio;
        while (ok && copied_rows < n_tokens) {
            uint32_t seg_rows = ratio - pos_mod;
            if (seg_rows > n_tokens - copied_rows) seg_rows = n_tokens - copied_rows;
            const uint32_t seg_elems = seg_rows * width;
            const NSUInteger src_off = (NSUInteger)ape_inner +
                                       (NSUInteger)pos_mod * width * elem_ape;
            const NSUInteger dst_off = (NSUInteger)copied_rows * width * sizeof(float);
            if (ape_type == 1u) {
                ok = ds4_gpu_encode_cpy_f16_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            } else {
                ok = ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                                     apebuf,
                                                     src_off,
                                                     g_compressor_store_ape_buffer,
                                                     dst_off,
                                                     seg_elems);
            }
            copied_rows += seg_rows;
            pos_mod = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_add_f32_1d(cb,
                                             scbuf,
                                             ds4_gpu_tensor_offset(sc),
                                             g_compressor_store_ape_buffer,
                                             0,
                                             g_compressor_store_score_buffer,
                                             0,
                                             total_elems);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_kv,
                                                   kvbuf,
                                                   ds4_gpu_tensor_offset(kv),
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (ok) {
            ok = ds4_gpu_encode_set_rows_f32_i32(cb,
                                                   state_score,
                                                   g_compressor_store_score_buffer,
                                                   0,
                                                   rows,
                                                   n_tokens,
                                                   state_rows,
                                                   width);
        }
        if (rows != rows_stack) free(rows);
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "compressor batch DS4 store")) return 0;
    }

    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_contiguous_3d_args(
        uint32_t cols,
        uint32_t rows,
        uint32_t planes) {
    const uint64_t row_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)rows,
        .ne12 = (int32_t)planes,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = (int32_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_softmax_f32_contiguous(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows,
        uint32_t             planes) {
    if (!cb || !src || !dst || width == 0 || rows == 0 || planes == 0) return 0;

    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)rows * row_bytes;
    ds4_gpu_softmax_args args = {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = (int32_t)planes,
        .nb01 = row_bytes,
        .nb02 = plane_bytes,
        .nb03 = (uint64_t)planes * plane_bytes,
        .ne11 = (int32_t)width,
        .ne12 = (int32_t)rows,
        .ne13 = (int32_t)planes,
        .nb11 = row_bytes,
        .nb12 = plane_bytes,
        .nb13 = (uint64_t)planes * plane_bytes,
        .nb1 = row_bytes,
        .nb2 = plane_bytes,
        .nb3 = (uint64_t)planes * plane_bytes,
        .scale = 1.0f,
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 1,
    };

    id<MTLComputePipelineState> pipeline =
        (width % 4u) == 0 ? g_soft_max_f32_4_pipeline : g_soft_max_f32_pipeline;
    if (!pipeline) return 0;

    NSUInteger nth = 32u;
    if ((width % 4u) == 0) {
        while (nth < (NSUInteger)(width / 4u) &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    } else {
        while (nth < (NSUInteger)width &&
               nth * (NSUInteger)rows * (NSUInteger)planes < 256u) {
            nth *= 2u;
        }
    }
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:src offset:src_off atIndex:2];
    [enc setBuffer:src offset:src_off atIndex:3];
    [enc setBuffer:dst offset:dst_off atIndex:4];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf || n_rows == 0 || head_dim == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * sizeof(float)) {
        return 0;
    }

    const NSUInteger pack_bytes = (NSUInteger)n_rows * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_product_buffer,
                                         &g_compressor_pool_product_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_product") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_cont_buffer,
                                         &g_compressor_pool_score_cont_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_score_cont") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_softmax_buffer,
                                         &g_compressor_pool_softmax_bytes,
                                         pack_bytes,
                                         "ds4_compressor_pool_softmax")) {
        return 0;
    }

    const uint64_t cont_row_stride = (uint64_t)n_rows * sizeof(float);
    const uint64_t cont_plane_stride = (uint64_t)head_dim * cont_row_stride;

    /*
     * Keep the n_comp == 1 compressor path as the unfused graph sequence:
     *
     *   score = soft_max(contiguous(score))
     *   pooled = sum_rows(contiguous(kv) * score)
     *
     * The fused DS4 pool kernel is mathematically equivalent, but it reduces in
     * a different order. That is enough to create ~1e-6 compressor differences
     * and later FP8/routing flips, so this path intentionally keeps the same
     * operation boundary and memory layout as the graph.
     */
    ds4_gpu_bin_args mul_args =
        ds4_gpu_make_bin_contiguous_3d_args(n_rows, head_dim, 1);

    return
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    kvbuf,
                                                    kv_offset,
                                                    g_compressor_pool_product_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    kv_nb0,
                                                    kv_nb1,
                                                    kv_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_cpy_f32_f32_3d_src_strided(cb,
                                                    scorebuf,
                                                    score_offset,
                                                    g_compressor_pool_score_cont_buffer,
                                                    0,
                                                    n_rows,
                                                    head_dim,
                                                    1,
                                                    score_nb0,
                                                    score_nb1,
                                                    score_nb2,
                                                    cont_row_stride,
                                                    cont_plane_stride) &&
        ds4_gpu_encode_softmax_f32_contiguous(cb,
                                                g_compressor_pool_score_cont_buffer,
                                                0,
                                                g_compressor_pool_softmax_buffer,
                                                0,
                                                n_rows,
                                                head_dim,
                                                1) &&
        ds4_gpu_encode_bin_f32_rows(cb,
                                      g_mul_pipeline,
                                      &mul_args,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      g_compressor_pool_softmax_buffer,
                                      0,
                                      g_compressor_pool_product_buffer,
                                      0) &&
        ds4_gpu_encode_sum_rows_f32(cb,
                                      g_compressor_pool_product_buffer,
                                      0,
                                      outbuf,
                                      ds4_gpu_tensor_offset(out),
                                      n_rows,
                                      head_dim);
}

static int ds4_gpu_encode_dsv4_softmax_pool(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *out,
        id<MTLBuffer>        kvbuf,
        NSUInteger           kv_offset,
        uint64_t             kv_nb0,
        uint64_t             kv_nb1,
        uint64_t             kv_nb2,
        id<MTLBuffer>        scorebuf,
        NSUInteger           score_offset,
        uint64_t             score_nb0,
        uint64_t             score_nb1,
        uint64_t             score_nb2,
        uint32_t             n_rows,
        uint32_t             head_dim,
        uint32_t             n_comp) {
    id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
    if (!cb || !outbuf || !kvbuf || !scorebuf ||
        n_rows == 0 || head_dim == 0 || n_comp == 0 ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)head_dim * n_comp * sizeof(float)) {
        return 0;
    }

    if (n_comp == 1) {
        return ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml(cb,
                                                                out,
                                                                kvbuf,
                                                                kv_offset,
                                                                kv_nb0,
                                                                kv_nb1,
                                                                kv_nb2,
                                                                scorebuf,
                                                                score_offset,
                                                                score_nb0,
                                                                score_nb1,
                                                                score_nb2,
                                                                n_rows,
                                                                head_dim);
    }

    ds4_gpu_dsv4_softmax_pool_args args = {
        .ne00 = (int64_t)n_rows,
        .ne01 = (int64_t)head_dim,
        .ne02 = (int64_t)n_comp,
        .nb00 = kv_nb0,
        .nb01 = kv_nb1,
        .nb02 = kv_nb2,
        .nb10 = score_nb0,
        .nb11 = score_nb1,
        .nb12 = score_nb2,
        .ne0 = (int64_t)head_dim,
        .ne1 = (int64_t)n_comp,
        .nb0 = sizeof(float),
        .nb1 = (uint64_t)head_dim * sizeof(float),
    };
    const uint64_t n = (uint64_t)head_dim * n_comp;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_softmax_pool_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:kvbuf offset:kv_offset atIndex:1];
    [enc setBuffer:scorebuf offset:score_offset atIndex:2];
    [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_concat_f32_dim1(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src0,
        NSUInteger           src0_offset,
        uint32_t             src0_rows,
        uint64_t             src0_row_stride,
        id<MTLBuffer>        src1,
        NSUInteger           src1_offset,
        uint32_t             src1_rows,
        uint64_t             src1_row_stride,
        id<MTLBuffer>        dst,
        NSUInteger           dst_offset,
        uint32_t             cols,
        uint64_t             dst_row_stride) {
    if (!cb || !src0 || !src1 || !dst || cols == 0 || src0_rows == 0 || src1_rows == 0) {
        return 0;
    }

    const uint32_t rows = src0_rows + src1_rows;
    const uint64_t src0_plane = (uint64_t)src0_rows * src0_row_stride;
    const uint64_t src1_plane = (uint64_t)src1_rows * src1_row_stride;
    const uint64_t dst_plane = (uint64_t)rows * dst_row_stride;
    ds4_gpu_concat_args args = {
        .ne00 = (int32_t)cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_row_stride,
        .nb02 = src0_plane,
        .nb03 = src0_plane,
        .ne10 = (int32_t)cols,
        .ne11 = (int32_t)src1_rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_stride,
        .nb12 = src1_plane,
        .nb13 = src1_plane,
        .ne0 = (int32_t)cols,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane,
        .nb3 = dst_plane,
        .dim = 1,
    };

    NSUInteger nth = cols < 1024u ? (NSUInteger)cols : 1024u;
    const NSUInteger max_threads = g_concat_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_concat_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src0 offset:src0_offset atIndex:1];
    [enc setBuffer:src1 offset:src1_offset atIndex:2];
    [enc setBuffer:dst offset:dst_offset atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_compressor_pool(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *out,
        const ds4_gpu_tensor *state_kv,
        const ds4_gpu_tensor *state_score,
        uint32_t               head_dim,
        uint32_t               ratio) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !out || !statekvbuf || !statescbuf || head_dim == 0 || ratio == 0) return 0;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rows = coff * ratio;
    const uint64_t state_bytes = (uint64_t)width * rows * sizeof(float);
    if (ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
        ds4_gpu_tensor_bytes(state_score) < state_bytes) {
        return 0;
    }

    if (ratio != 4u) {
        const uint64_t row_stride = (uint64_t)width * sizeof(float);
        return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                  out,
                                                  statekvbuf,
                                                  ds4_gpu_tensor_offset(state_kv),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  statescbuf,
                                                  ds4_gpu_tensor_offset(state_score),
                                                  row_stride,
                                                  sizeof(float),
                                                  (uint64_t)rows * row_stride,
                                                  ratio,
                                                  head_dim,
                                                  1);
    }

    const NSUInteger packed_bytes = (NSUInteger)8u * head_dim * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                         &g_compressor_pool_kv_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                         &g_compressor_pool_score_bytes,
                                         packed_bytes,
                                         "ds4_compressor_pool_score")) {
        return 0;
    }

    const uint64_t state_row_stride = (uint64_t)width * sizeof(float);
    const uint64_t pool_row_stride = (uint64_t)head_dim * sizeof(float);
    const NSUInteger curr_offset = (NSUInteger)4u * state_row_stride +
                                   (NSUInteger)head_dim * sizeof(float);
    if (!ds4_gpu_encode_concat_f32_dim1(cb,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv),
                                          4,
                                          state_row_stride,
                                          statekvbuf,
                                          ds4_gpu_tensor_offset(state_kv) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_kv_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride) ||
        !ds4_gpu_encode_concat_f32_dim1(cb,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score),
                                          4,
                                          state_row_stride,
                                          statescbuf,
                                          ds4_gpu_tensor_offset(state_score) + curr_offset,
                                          4,
                                          state_row_stride,
                                          g_compressor_pool_score_buffer,
                                          0,
                                          head_dim,
                                          pool_row_stride)) {
        return 0;
    }

    return ds4_gpu_encode_dsv4_softmax_pool(cb,
                                              out,
                                              g_compressor_pool_kv_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              g_compressor_pool_score_buffer,
                                              0,
                                              pool_row_stride,
                                              sizeof(float),
                                              packed_bytes,
                                              8,
                                              head_dim,
                                              1);
}

static int ds4_gpu_encode_compressor_shift_ratio4(
        id<MTLCommandBuffer> cb,
        ds4_gpu_tensor    *state_kv,
        ds4_gpu_tensor    *state_score,
        uint32_t             width) {
    id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
    id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
    if (!cb || !statekvbuf || !statescbuf || !g_dsv4_ratio4_shift_pipeline || width == 0) return 0;

    ds4_gpu_dsv4_ratio4_shift_args args = { .width = width };
    const uint32_t n = 4u * width;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_dsv4_ratio4_shift_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:statekvbuf offset:ds4_gpu_tensor_offset(state_kv) atIndex:1];
    [enc setBuffer:statescbuf offset:ds4_gpu_tensor_offset(state_score) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + 255u) / 256u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t n_comp = n_tokens / ratio;
        const uint32_t cutoff = n_comp * ratio;
        const uint32_t rem = n_tokens - cutoff;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor prefill received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }

        if (ok && ratio == 4u) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const int have_prev = cutoff >= ratio ? 1 : 0;
            const uint32_t prev_start = rem == 0 ? cutoff - ratio : cutoff - ratio;
            if (have_prev) {
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)prev_start * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + prev_start,
                                                                     rows_prev,
                                                                     4,
                                                                     state_rows);
            }
            if (ok && rem != 0) {
                int32_t rows_cur[4];
                for (uint32_t i = 0; i < rem; i++) rows_cur[i] = (int32_t)(ratio + i);
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows_cur,
                                                                     rem,
                                                                     state_rows);
            }
        } else if (ok && rem != 0) {
            int32_t rows[128];
            if (rem > (uint32_t)(sizeof(rows) / sizeof(rows[0]))) {
                fprintf(stderr, "ds4: Metal compressor prefill remainder exceeds local row list\n");
                ok = 0;
            } else {
                for (uint32_t i = 0; i < rem; i++) rows[i] = (int32_t)i;
                ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                     state_kv,
                                                                     state_score,
                                                                     kvbuf,
                                                                     ds4_gpu_tensor_offset(kv) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     scbuf,
                                                                     ds4_gpu_tensor_offset(sc) +
                                                                             (NSUInteger)cutoff * width * sizeof(float),
                                                                     apebuf,
                                                                     (NSUInteger)ape_inner,
                                                                     ape_type,
                                                                     width,
                                                                     ratio,
                                                                     pos0 + cutoff,
                                                                     rows,
                                                                     rem,
                                                                     state_rows);
            }
        }

        if (ok && n_comp != 0) {
            const NSUInteger score_bytes = (NSUInteger)cutoff * width * sizeof(float);
            if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                 &g_compressor_store_score_bytes,
                                                 score_bytes,
                                                 "ds4_compressor_store_score")) {
                ok = 0;
            }
            if (ok) {
                ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc),
                                                                 g_compressor_store_score_buffer,
                                                                 0,
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 cutoff);
            }

            if (ok && ratio == 4u) {
                const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
                if (!ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                     &g_compressor_pool_kv_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_kv") ||
                    !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                     &g_compressor_pool_score_bytes,
                                                     pack_bytes,
                                                     "ds4_compressor_pool_score")) {
                    ok = 0;
                }
                if (ok) {
                    ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_kv_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        0.0f) &&
                         ds4_gpu_encode_fill_f32_rows(cb,
                                                        g_compressor_pool_score_buffer,
                                                        0,
                                                        head_dim,
                                                        8u * n_comp,
                                                        ds4_gpu_negative_infinity());
                }
                if (ok) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv) +
                                                                 (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_kv_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         (NSUInteger)head_dim * sizeof(float),
                                                         g_compressor_pool_score_buffer,
                                                         (NSUInteger)4u * head_dim * sizeof(float),
                                                         head_dim,
                                                         ratio,
                                                         n_comp,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok && n_comp > 1u) {
                    const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
                    const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
                    const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
                    const uint64_t dst_plane_stride = 8ull * dst_row_stride;
                    ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         kvbuf,
                                                         ds4_gpu_tensor_offset(kv),
                                                         g_compressor_pool_kv_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride) &&
                         ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                         g_compressor_store_score_buffer,
                                                         0,
                                                         g_compressor_pool_score_buffer,
                                                         dst_plane_stride,
                                                         head_dim,
                                                         ratio,
                                                         n_comp - 1u,
                                                         src_row_stride,
                                                         src_plane_stride,
                                                         dst_row_stride,
                                                         dst_plane_stride);
                }
                if (ok) {
                    ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                            comp_cache,
                                                            g_compressor_pool_kv_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            g_compressor_pool_score_buffer,
                                                            0,
                                                            (uint64_t)head_dim * sizeof(float),
                                                            sizeof(float),
                                                            8ull * head_dim * sizeof(float),
                                                            8,
                                                            head_dim,
                                                            n_comp);
                }
            } else if (ok) {
                const uint64_t row_stride = (uint64_t)width * sizeof(float);
                ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                        comp_cache,
                                                        kvbuf,
                                                        ds4_gpu_tensor_offset(kv),
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        g_compressor_store_score_buffer,
                                                        0,
                                                        row_stride,
                                                        sizeof(float),
                                                        (uint64_t)ratio * row_stride,
                                                        ratio,
                                                        head_dim,
                                                        n_comp);
            }
        }

        if (ok && n_comp != 0) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_comp != 0 && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && n_comp != 0 && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint32_t n_comp = n_tokens / ratio;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor replay tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !compbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            ds4_gpu_tensor_bytes(comp_cache) < comp_bytes) {
            fprintf(stderr, "ds4: Metal compressor replay received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        const NSUInteger score_bytes = (NSUInteger)n_tokens * width * sizeof(float);
        const NSUInteger pack_bytes = (NSUInteger)n_comp * 8u * head_dim * sizeof(float);
        if (ok && (!ds4_gpu_ensure_scratch_buffer(&g_compressor_store_score_buffer,
                                                    &g_compressor_store_score_bytes,
                                                    score_bytes,
                                                    "ds4_compressor_store_score") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_kv_buffer,
                                                    &g_compressor_pool_kv_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_kv") ||
                   !ds4_gpu_ensure_scratch_buffer(&g_compressor_pool_score_buffer,
                                                    &g_compressor_pool_score_bytes,
                                                    pack_bytes,
                                                    "ds4_compressor_pool_score"))) {
            ok = 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_compressor_score_with_ape(cb,
                                                            scbuf,
                                                            ds4_gpu_tensor_offset(sc),
                                                            g_compressor_store_score_buffer,
                                                            0,
                                                            apebuf,
                                                            (NSUInteger)ape_inner,
                                                            ape_type,
                                                            width,
                                                            ratio,
                                                            pos0,
                                                            n_tokens);
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_kv_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                g_compressor_pool_score_buffer,
                                                0,
                                                head_dim,
                                                8u * n_comp,
                                                ds4_gpu_negative_infinity());
        }

        const uint64_t src_row_stride = (uint64_t)width * sizeof(float);
        const uint64_t src_plane_stride = (uint64_t)ratio * src_row_stride;
        const uint64_t dst_row_stride = (uint64_t)head_dim * sizeof(float);
        const uint64_t dst_plane_stride = 8ull * dst_row_stride;
        const NSUInteger state_off = ds4_gpu_tensor_offset(state_kv);
        const NSUInteger state_score_off = ds4_gpu_tensor_offset(state_score);

        if (ok) {
            /*
             * The aligned nonzero ratio-4 path replays the current ubatch
             * compressor, but seeds the first compressed row with the previous
             * compressor state. Rows 0..3 are the previous half, rows 4..7 are
             * the current half.
             */
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statekvbuf,
                                                 state_off,
                                                 g_compressor_pool_kv_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 statescbuf,
                                                 state_score_off,
                                                 g_compressor_pool_score_buffer,
                                                 0,
                                                 head_dim,
                                                 ratio,
                                                 1,
                                                 src_row_stride,
                                                 (uint64_t)ratio * src_row_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv) +
                                                         (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_kv_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 (NSUInteger)head_dim * sizeof(float),
                                                 g_compressor_pool_score_buffer,
                                                 (NSUInteger)4u * head_dim * sizeof(float),
                                                 head_dim,
                                                 ratio,
                                                 n_comp,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok && n_comp > 1u) {
            ok = ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 kvbuf,
                                                 ds4_gpu_tensor_offset(kv),
                                                 g_compressor_pool_kv_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride) &&
                 ds4_gpu_encode_cpy_f32_f32_3d(cb,
                                                 g_compressor_store_score_buffer,
                                                 0,
                                                 g_compressor_pool_score_buffer,
                                                 dst_plane_stride,
                                                 head_dim,
                                                 ratio,
                                                 n_comp - 1u,
                                                 src_row_stride,
                                                 src_plane_stride,
                                                 dst_row_stride,
                                                 dst_plane_stride);
        }
        if (ok) {
            ok = ds4_gpu_encode_dsv4_softmax_pool(cb,
                                                    comp_cache,
                                                    g_compressor_pool_kv_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    g_compressor_pool_score_buffer,
                                                    0,
                                                    dst_row_stride,
                                                    sizeof(float),
                                                    dst_plane_stride,
                                                    8,
                                                    head_dim,
                                                    n_comp);
        }
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache,
                                                       comp_cache,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       n_comp,
                                                       rms_eps) != 0;
        }
        if (ok && n_rot != 0) {
            ds4_gpu_rope_tail_batch_args rope_args = ds4_gpu_make_rope_tail_args(
                n_comp, 1, head_dim, n_rot, n_ctx_orig, false,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb && !owned &&
                 ds4_gpu_encode_rope_tail_inplace(cb,
                                                    compbuf,
                                                    ds4_gpu_tensor_offset(comp_cache),
                                                    &rope_args,
                                                    n_comp,
                                                    1,
                                                    head_dim,
                                                    pos0,
                                                    ratio);
        }
        if (ok && quantize_fp8) {
            ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot) != 0;
        }

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                state_off,
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                state_score_off,
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows_prev[4] = { 0, 1, 2, 3 };
            const uint32_t prev_start = n_tokens - ratio;
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc) +
                                                                         (NSUInteger)prev_start * width * sizeof(float),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0 + prev_start,
                                                                 rows_prev,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t ratio = 4u;
        const uint32_t width = 2u * head_dim;
        const uint32_t state_rows = 8u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
            fprintf(stderr, "ds4: Metal compressor prefill-state APE range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_tail);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_tail);
        id<MTLBuffer> statekvbuf = ds4_gpu_tensor_buffer(state_kv);
        id<MTLBuffer> statescbuf = ds4_gpu_tensor_buffer(state_score);
        if (!kvbuf || !scbuf || !statekvbuf || !statescbuf ||
            ds4_gpu_tensor_bytes(kv_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(sc_tail) < tail_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes) {
            fprintf(stderr, "ds4: Metal compressor prefill-state received undersized buffers\n");
            return 0;
        }

        uint64_t ape_inner = 0;
        id<MTLBuffer> apebuf = ds4_gpu_wrap_model_range(model_map, model_size, ape_offset, ape_bytes, &ape_inner);
        if (!apebuf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        int ok = 1;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) ok = 0;

        if (ok) {
            ok = ds4_gpu_encode_fill_f32_rows(cb,
                                                statekvbuf,
                                                ds4_gpu_tensor_offset(state_kv),
                                                width,
                                                state_rows,
                                                0.0f) &&
                 ds4_gpu_encode_fill_f32_rows(cb,
                                                statescbuf,
                                                ds4_gpu_tensor_offset(state_score),
                                                width,
                                                state_rows,
                                                ds4_gpu_negative_infinity());
        }
        if (ok) {
            int32_t rows[4] = { 0, 1, 2, 3 };
            ok = ds4_gpu_encode_compressor_set_rows_projected(cb,
                                                                 state_kv,
                                                                 state_score,
                                                                 kvbuf,
                                                                 ds4_gpu_tensor_offset(kv_tail),
                                                                 scbuf,
                                                                 ds4_gpu_tensor_offset(sc_tail),
                                                                 apebuf,
                                                                 (NSUInteger)ape_inner,
                                                                 ape_type,
                                                                 width,
                                                                 ratio,
                                                                 pos0,
                                                                 rows,
                                                                 ratio,
                                                                 state_rows);
        }

        if (!had_batch) {
            const int end_ok = ds4_gpu_end_commands();
            ok = end_ok && ok;
        }
        return ok ? 1 : 0;
    }
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) ||
        norm_type != 0u) {
        return 0;
    }

    @autoreleasepool {
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint32_t width = coff * head_dim;
        const uint32_t state_rows = coff * ratio;
        const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
        const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
        const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
        const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
        const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
        const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

        if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
            norm_offset > model_size || norm_bytes > model_size - norm_offset) {
            fprintf(stderr, "ds4: Metal compressor tensor range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> kvbuf = ds4_gpu_tensor_buffer(kv_cur);
        id<MTLBuffer> scbuf = ds4_gpu_tensor_buffer(sc_cur);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_cache);
        if (!kvbuf || !scbuf || !compbuf ||
            ds4_gpu_tensor_bytes(kv_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(sc_cur) < kv_bytes ||
            ds4_gpu_tensor_bytes(state_kv) < state_bytes ||
            ds4_gpu_tensor_bytes(state_score) < state_bytes ||
            (emit && ds4_gpu_tensor_bytes(comp_cache) < comp_bytes)) {
            fprintf(stderr, "ds4: Metal compressor update received undersized buffers\n");
            return 0;
        }

        const bool use_store_one =
            getenv("DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE") == NULL;
        const int store_ok = use_store_one
            ? ds4_gpu_compressor_store_one_tensor(kv_cur,
                                                    sc_cur,
                                                    state_kv,
                                                    state_score,
                                                    model_map,
                                                    model_size,
                                                    ape_offset,
                                                    ape_type,
                                                    width,
                                                    ratio,
                                                    pos)
            : ds4_gpu_compressor_store_batch_tensor(kv_cur,
                                                      sc_cur,
                                                      state_kv,
                                                      state_score,
                                                      model_map,
                                                      model_size,
                                                      ape_offset,
                                                      ape_type,
                                                      head_dim,
                                                      ratio,
                                                      pos,
                                                      1);
        if (!store_ok) {
            return 0;
        }
        if (!emit) return 1;

        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                comp_cache,
                (uint64_t)comp_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
        if (!comp_row_view) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_compressor_pool(cb,
                                                  comp_row_view,
                                                  state_kv,
                                                  state_score,
                                                  head_dim,
                                                  ratio);
        if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor DS4 softmax pool");
        if (ok) {
            ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view,
                                                       comp_row_view,
                                                       model_map,
                                                       model_size,
                                                       norm_offset,
                                                       head_dim,
                                                       1,
                                                       rms_eps) != 0;
        }
        if (ok) {
            const uint32_t comp_pos = pos + 1u - ratio;
            ok = ds4_gpu_rope_tail_tensor(comp_row_view,
                                            1,
                                            1,
                                            head_dim,
                                            n_rot,
                                            comp_pos,
                                            n_ctx_orig,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            beta_fast,
                                            beta_slow) != 0;
        }
        if (ok && ratio == 4u) {
            cb = ds4_gpu_command_buffer(&owned);
            ok = cb &&
                 ds4_gpu_encode_compressor_shift_ratio4(cb,
                                                          state_kv,
                                                          state_score,
                                                          width);
            if (ok) ok = ds4_gpu_finish_command_buffer(cb, owned, "compressor ratio4 state shift");
        }
        ds4_gpu_tensor_free(comp_row_view);
        if (!ok) return 0;
    }

    return 1;
}

static int ds4_gpu_encode_fill_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             width,
        uint32_t             rows,
        float                value) {
    if (!cb || !buf || width == 0 || rows == 0 || (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth_max = g_unary_fill_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !group_tmp || !low_tmp || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0 ||
        group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || (low_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output batch received invalid q8 dimensions\n");
            return 0;
        }
        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t row_b_bytes = (low_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        const uint64_t out_b_bytes = out_dim * row_b_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
            out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) {
            fprintf(stderr, "ds4: Metal attention output batch weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_tokens * n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = (uint64_t)n_tokens * low_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes) {
            fprintf(stderr, "ds4: Metal attention output batch received undersized buffers\n");
            return 0;
        }
        (void)group_tmp;
        (void)low_tmp;

        const bool use_direct_low =
            n_tokens < 32u && getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == NULL;
        const NSUInteger ids_bytes = (NSUInteger)n_tokens * (NSUInteger)n_groups * sizeof(int32_t);
        id<MTLBuffer> group_ids_buffer = nil;
        if (!use_direct_low) {
            if (getenv("DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE") != NULL) {
                group_ids_buffer =
                    ds4_gpu_new_transient_buffer(ids_bytes, "attention output group ids");
                if (!group_ids_buffer) {
                    return 0;
                }
            } else {
                if (!ds4_gpu_ensure_scratch_buffer(&g_attn_out_group_ids_buffer,
                                                     &g_attn_out_group_ids_bytes,
                                                     ids_bytes,
                                                     "ds4_attention_output_group_ids")) {
                    return 0;
                }
                group_ids_buffer = g_attn_out_group_ids_buffer;
            }
            int32_t *ids = (int32_t *)[group_ids_buffer contents];
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t group = 0; group < n_groups; group++) {
                    ids[(uint64_t)t * n_groups + group] = (int32_t)group;
                }
            }
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }
        const bool attn_out_profile =
            getenv("DS4_METAL_ATTN_OUT_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        double attn_out_t0 = attn_out_profile ? ds4_gpu_now_ms() : 0.0;
#define DS4_METAL_PROFILE_ATTN_OUT_STAGE(name) do { \
            if (ok && attn_out_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = false; \
                } else { \
                    const double now_ms = ds4_gpu_now_ms(); \
                    fprintf(stderr, \
                            "ds4: Metal attention output stage tokens=%u %s=%.3f ms\n", \
                            n_tokens, (name), now_ms - attn_out_t0); \
                    attn_out_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = false; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb || owned) ok = false; \
                    } \
                } \
            } \
        } while (0)

        if (ok) {
            /*
             * Batched attention-output projections switch from the vector
             * kernel to the SIMD matrix kernel once the batch has at least 32
             * tokens.  This preserves the single-token generation path while
             * keeping prefill accumulation stable.
             */
            if (n_tokens >= 32u && ds4_gpu_mul_mm_id_map0_name(n_groups) != NULL) {
                ds4_gpu_mul_mm_id_map_args map_args =
                    ds4_gpu_make_mul_mm_id_map_args((uint32_t)group_dim,
                                                      n_groups,
                                                      n_groups,
                                                      n_groups,
                                                      n_tokens);
                ds4_gpu_mul_mm_id_args mm_args =
                    ds4_gpu_make_mul_mm_id_args((uint32_t)group_dim,
                                                  (uint32_t)rank,
                                                  n_groups,
                                                  row_a_bytes,
                                                  (uint64_t)rank * row_a_bytes,
                                                  n_groups,
                                                  n_groups,
                                                  n_tokens);
                id<MTLComputePipelineState> map_pipeline =
                    ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_groups));
                id<MTLComputePipelineState> mm_pipeline =
                    ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q8_0_f32", false);
                ok = ds4_gpu_encode_mul_mm_id(cb,
                                                map_pipeline,
                                                mm_pipeline,
                                                &map_args,
                                                &mm_args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0) != 0;
            } else if (use_direct_low) {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = 0,
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
                ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                             pipeline,
                                                             &args,
                                                             out_a_buf,
                                                             (NSUInteger)out_a_inner,
                                                             ds4_gpu_tensor_buffer(heads),
                                                             ds4_gpu_tensor_offset(heads),
                                                             ds4_gpu_tensor_buffer(low),
                                                             ds4_gpu_tensor_offset(low),
                                                             32u * 2u * sizeof(float),
                                                             4) != 0;
            } else {
                ds4_gpu_mul_mv_id_args args = {
                    .nei0 = (int32_t)n_groups,
                    .nei1 = (int32_t)n_tokens,
                    .nbi1 = (uint64_t)n_groups * sizeof(int32_t),
                    .ne00 = (int32_t)group_dim,
                    .ne01 = (int32_t)rank,
                    .ne02 = (int32_t)n_groups,
                    .nb00 = 34,
                    .nb01 = row_a_bytes,
                    .nb02 = (uint64_t)rank * row_a_bytes,
                    .ne10 = (int32_t)group_dim,
                    .ne11 = (int32_t)n_groups,
                    .ne12 = (int32_t)n_tokens,
                    .ne13 = 1,
                    .nb10 = sizeof(float),
                    .nb11 = (uint64_t)group_dim * sizeof(float),
                    .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                    .ne0 = (int32_t)rank,
                    .ne1 = (int32_t)n_groups,
                    .nb1 = (uint64_t)rank * sizeof(float),
                    .nr0 = 2,
                };
                id<MTLComputePipelineState> pipeline =
                    ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_id_q8_0_f32", 4);
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                pipeline,
                                                &args,
                                                out_a_buf,
                                                (NSUInteger)out_a_inner,
                                                ds4_gpu_tensor_buffer(heads),
                                                ds4_gpu_tensor_offset(heads),
                                                ds4_gpu_tensor_buffer(low),
                                                ds4_gpu_tensor_offset(low),
                                                group_ids_buffer,
                                                0,
                                                32u * 2u * sizeof(float),
                                                4,
                                                true) != 0;
            }
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("low_proj");

        if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                              out_b_offset,
                                              low_dim, out_dim, low, n_tokens) != 0;
        }
        DS4_METAL_PROFILE_ATTN_OUT_STAGE("out_proj");

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
#undef DS4_METAL_PROFILE_ATTN_OUT_STAGE
        return ok ? 1 : 0;
    }
}

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t low_dim = (uint64_t)n_groups * rank;
        if ((group_dim % 32u) != 0 || low_dim > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal attention output low received invalid q8 dimensions\n");
            return 0;
        }

        const uint64_t row_a_bytes = (group_dim / 32u) * 34u;
        const uint64_t out_a_bytes = (uint64_t)n_groups * rank * row_a_bytes;
        if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) {
            fprintf(stderr, "ds4: Metal attention output low weights are outside the mapped model\n");
            return 0;
        }

        const uint64_t heads_bytes = (uint64_t)n_groups * group_dim * sizeof(float);
        const uint64_t low_bytes = low_dim * sizeof(float);
        if (ds4_gpu_tensor_bytes(heads) < heads_bytes ||
            ds4_gpu_tensor_bytes(low) < low_bytes) {
            fprintf(stderr, "ds4: Metal attention output low received undersized buffers\n");
            return 0;
        }

        uint64_t out_a_inner = 0;
        id<MTLBuffer> out_a_buf =
            ds4_gpu_wrap_model_range(model_map, model_size,
                                       out_a_offset, out_a_bytes,
                                       &out_a_inner);
        if (!out_a_buf) return 0;

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;

        bool ok = true;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb || owned) {
            ok = false;
        }

        if (ok) {
            ds4_gpu_mul_mv_id_args args = {
                .nei0 = (int32_t)n_groups,
                .nei1 = 1,
                .nbi1 = 0,
                .ne00 = (int32_t)group_dim,
                .ne01 = (int32_t)rank,
                .ne02 = (int32_t)n_groups,
                .nb00 = 34,
                .nb01 = row_a_bytes,
                .nb02 = (uint64_t)rank * row_a_bytes,
                .ne10 = (int32_t)group_dim,
                .ne11 = (int32_t)n_groups,
                .ne12 = 1,
                .ne13 = 1,
                .nb10 = sizeof(float),
                .nb11 = (uint64_t)group_dim * sizeof(float),
                .nb12 = (uint64_t)n_groups * group_dim * sizeof(float),
                .ne0 = (int32_t)rank,
                .ne1 = (int32_t)n_groups,
                .nb1 = (uint64_t)rank * sizeof(float),
                .nr0 = 2,
            };
            id<MTLComputePipelineState> pipeline =
                ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_attn_out_low_q8_0_f32", 4);
            ok = ds4_gpu_encode_attn_out_low_q8_direct(cb,
                                                         pipeline,
                                                         &args,
                                                         out_a_buf,
                                                         (NSUInteger)out_a_inner,
                                                         ds4_gpu_tensor_buffer(heads),
                                                         ds4_gpu_tensor_offset(heads),
                                                         ds4_gpu_tensor_buffer(low),
                                                         ds4_gpu_tensor_offset(low),
                                                         32u * 2u * sizeof(float),
                                                         4) != 0;
        }

        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        return ok ? 1 : 0;
    }
}

static NSUInteger ds4_gpu_align_up_ns(NSUInteger value, NSUInteger align) {
    return (value + align - 1u) & ~(align - 1u);
}

static int ds4_gpu_encode_cpy_f32_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f32_3d_src_strided(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint32_t             planes,
        uint64_t             src_col_stride,
        uint64_t             src_row_stride,
        uint64_t             src_plane_stride,
        uint64_t             dst_row_stride,
        uint64_t             dst_plane_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0 || planes == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = (int64_t)planes,
        .ne03 = 1,
        .nb00 = src_col_stride,
        .nb01 = src_row_stride,
        .nb02 = src_plane_stride,
        .nb03 = (uint64_t)planes * src_plane_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = (int64_t)planes,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_row_stride,
        .nb2 = dst_plane_stride,
        .nb3 = (uint64_t)planes * dst_plane_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f32_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, planes, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(float), sizeof(uint16_t));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f32_f16_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f32_f16_2d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             cols,
        uint32_t             rows,
        uint64_t             src_row_stride,
        uint64_t             dst_row_stride) {
    if (!cb || !src || !dst || cols == 0 || rows == 0) return 0;

    ds4_gpu_cpy_args args = {
        .nk0 = (int64_t)cols,
        .ne00 = (int64_t)cols,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_stride,
        .nb02 = (uint64_t)rows * src_row_stride,
        .nb03 = (uint64_t)rows * src_row_stride,
        .ne0 = (int64_t)cols,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(uint16_t),
        .nb1 = dst_row_stride,
        .nb2 = (uint64_t)rows * dst_row_stride,
        .nb3 = (uint64_t)rows * dst_row_stride,
    };
    const NSUInteger nth = ds4_gpu_cpy_threads(cols, g_cpy_f32_f16_pipeline);
    const NSUInteger col_groups = ((NSUInteger)cols + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f32_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(col_groups * rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_cpy_f16_f32_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n) {
    if (!cb || !src || !dst || n == 0) return 0;

    ds4_gpu_cpy_args args =
        ds4_gpu_make_cpy_1d_args(n, sizeof(uint16_t), sizeof(float));
    const NSUInteger nth = ds4_gpu_cpy_threads(n, g_cpy_f16_f32_pipeline);
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_cpy_f16_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}


static int ds4_gpu_encode_fill_f16_1d(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        buf,
        NSUInteger           offset,
        uint32_t             n,
        float                value) {
    if (!cb || !buf || n == 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(n, 1, 0, 0.0f, 0.0f);
    args.val = value;

    NSUInteger nth = (NSUInteger)n;
    const NSUInteger max_threads = g_unary_fill_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    const NSUInteger groups = ((NSUInteger)n + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_fill_f16_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:buf offset:offset atIndex:1];
    [enc setBuffer:buf offset:offset atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_raw_heads(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *heads,
        id<MTLBuffer>         sinks_buf,
        NSUInteger            sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || raw_cap < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t heads_bytes = q_bytes;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < heads_bytes) {
        fprintf(stderr, "ds4: Metal DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_raw, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    memset([mask_buffer contents], 0, mask_bytes);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_raw % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_raw % ncpsg) != 0,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        const uint32_t tail_elems = tail_rows * head_dim;
        const uint32_t head_elems = head_rows * head_dim;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }

        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_elems)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_elems))) {
            return 0;
        }

        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }

    if ((n_raw % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static void ds4_gpu_fill_raw_prefill_mask(uint16_t *mask, uint32_t n_tokens, uint32_t window) {
    const uint16_t neg_inf_half = 0xfc00u;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_tokens;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_raw_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  pos0,
        uint32_t  window) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* The caller has already copied the SWA ring into logical order when it
     * wraps, so key row k represents first_raw_pos + k. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_raw;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_mixed_decode_batch_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_raw,
        uint32_t  n_comp,
        uint32_t  pos0,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    /* Raw keys are laid out by logical position; compressed keys follow them. */
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    for (uint32_t q = 0; q < n_tokens; q++) {
        const uint32_t qpos = pos0 + q;
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_raw; k++) {
            const uint32_t kpos = first_raw_pos + k;
            const bool causal = kpos <= qpos;
            const bool in_window = causal && (window == 0 || qpos - kpos < window);
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }
        const uint32_t n_visible = (qpos + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_raw + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

static void ds4_gpu_fill_static_mixed_prefill_mask(
        uint16_t *mask,
        uint32_t  n_tokens,
        uint32_t  n_comp,
        uint32_t  window,
        uint32_t  ratio) {
    const uint16_t neg_inf_half = 0xfc00u;
    const uint32_t n_keys = n_tokens + n_comp;
    for (uint32_t q = 0; q < n_tokens; q++) {
        uint16_t *row = mask + (uint64_t)q * n_keys;
        for (uint32_t k = 0; k < n_tokens; k++) {
            const bool causal = k <= q;
            const bool in_window = window == 0 || q - k < window;
            row[k] = causal && in_window ? 0u : neg_inf_half;
        }

        const uint32_t n_visible = (q + 1u) / ratio;
        for (uint32_t c = 0; c < n_comp; c++) {
            row[n_tokens + c] = c < n_visible ? 0u : neg_inf_half;
        }
    }
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    if (n_comp &&
        !ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         compbuf,
                                         ds4_gpu_tensor_offset(comp_kv),
                                         g_flash_attn_kv_buffer,
                                         (NSUInteger)n_tokens * row_bytes_f16,
                                         n_comp * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 || ratio == 0) {
        return 0;
    }

    const uint32_t n_keys = n_tokens + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill static mixed DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }
    if (n_comp) {
        if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                             compbuf,
                                             ds4_gpu_tensor_offset(comp_kv),
                                             g_flash_attn_kv_buffer,
                                             (NSUInteger)n_tokens * row_bytes_f16,
                                             n_comp * head_dim)) {
            return 0;
        }
    }

    ds4_gpu_fill_static_mixed_prefill_mask((uint16_t *)[mask_buffer contents],
                                             n_tokens,
                                             n_comp,
                                             window,
                                             ratio);
    if (use_comp_mask && n_comp != 0) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_tokens * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, has_kvpad,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_tokens >= 20) {
        return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec_long(cb,
                                                                                       heads,
                                                                                       sinks_buf,
                                                                                       sinks_offset,
                                                                                       q,
                                                                                       raw_kv,
                                                                                       comp_kv,
                                                                                       comp_mask,
                                                                                       use_comp_mask,
                                                                                       n_tokens,
                                                                                       n_comp,
                                                                                       window,
                                                                                       ratio,
                                                                                       n_head,
                                                                                       head_dim);
    }
    return ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_vec(cb,
                                                                           heads,
                                                                           sinks_buf,
                                                                           sinks_offset,
                                                                           q,
                                                                           raw_kv,
                                                                           comp_kv,
                                                                           comp_mask,
                                                                           use_comp_mask,
                                                                           n_tokens,
                                                                           n_comp,
                                                                           window,
                                                                           ratio,
                                                                           n_head,
                                                                           head_dim);
}

static int ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 non-vector FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_tokens % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_tokens * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_tokens + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }
    ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents], n_tokens, window);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_tokens * head_dim)) {
        return 0;
    }

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_tokens,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_tokens * row_bytes_f16,
            .nb13 = (uint64_t)n_tokens * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_tokens * row_bytes_f16,
            .nb23 = (uint64_t)n_tokens * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_tokens,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_tokens,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_tokens * row_bytes_f16,
        .nb13 = (uint64_t)n_tokens * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_tokens * row_bytes_f16,
        .nb23 = (uint64_t)n_tokens * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_prefill_raw_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0) {
        return 0;
    }
    if (n_tokens >= 20) {
        return ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec(cb,
                                                                         heads,
                                                                         sinks_buf,
                                                                         sinks_offset,
                                                                         q,
                                                                         raw_kv,
                                                                         n_tokens,
                                                                         window,
                                                                         n_head,
                                                                         head_dim);
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal prefill raw DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_tokens, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_tokens * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_f16_offset = 0;
    const NSUInteger kv_f16_bytes = (NSUInteger)n_tokens * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_tokens * n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_f16_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }
    ds4_gpu_fill_raw_prefill_mask((uint16_t *)[mask_buffer contents], n_tokens, window);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_tokens % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, true,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         rawbuf,
                                         ds4_gpu_tensor_offset(raw_kv),
                                         g_flash_attn_kv_buffer,
                                         kv_f16_offset,
                                         n_tokens * head_dim)) {
        return 0;
    }

    if ((n_tokens % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_tokens,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_tokens * row_bytes_f16,
            .nb13 = (uint64_t)n_tokens * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_tokens * row_bytes_f16,
            .nb23 = (uint64_t)n_tokens * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_tokens,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_tokens * row_bytes_f16,
        .nb13 = (uint64_t)n_tokens * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_tokens * row_bytes_f16,
        .nb23 = (uint64_t)n_tokens * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_tokens * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:kv_f16_offset atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_tokens, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_gathered_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t               n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_mask,
        uint32_t               n_head,
        uint32_t               head_dim) {
    const uint32_t n_keys = n_raw + n_comp;
    if (head_dim != 512 || n_head == 0 || n_raw == 0 || n_keys == 0 ||
        raw_cap < n_raw || n_keys < n_raw) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : nil;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : nil;
    const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        (n_comp && !compbuf) ||
        (use_mask && !maskbuf) ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes ||
        (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes)) {
        fprintf(stderr, "ds4: Metal gathered DS4 FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t ncpsg = 32;
    const uint32_t nwg = 32;
    const uint32_t nsg = ds4_gpu_flash_attn_vec_nsg(n_keys, nwg, ncpsg);
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = 2u * (NSUInteger)ncpsg * row_bytes_f16 +
                                 (NSUInteger)ncpsg * sizeof(uint16_t);
    const NSUInteger nrows = (NSUInteger)n_head;
    const NSUInteger tmp_bytes = nrows * (NSUInteger)head_dim * (NSUInteger)nwg * sizeof(float) +
                                 nrows * (2u * (NSUInteger)nwg) * sizeof(float);

    if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_mask_buffer,
                                         &g_flash_attn_mask_bytes,
                                         mask_bytes,
                                         "ds4_flash_attn_mask") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_tmp_buffer,
                                         &g_flash_attn_tmp_bytes,
                                         tmp_bytes,
                                         "ds4_flash_attn_tmp")) {
        return 0;
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if ((n_keys % ncpsg) != 0) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> vec_pipeline =
        ds4_gpu_get_flash_attn_vec_pipeline("kernel_flash_attn_ext_vec_f16_dk512_dv512",
                                              true, true, false, false, (n_keys % ncpsg) != 0,
                                              (int32_t)head_dim,
                                              (int32_t)head_dim,
                                              (int32_t)nsg,
                                              (int32_t)nwg);
    id<MTLComputePipelineState> reduce_pipeline =
        ds4_gpu_get_flash_attn_reduce_pipeline((int32_t)head_dim, (int32_t)nwg);
    if (!vec_pipeline || !reduce_pipeline) return 0;

    id<MTLBuffer> raw_linear_buf = rawbuf;
    NSUInteger raw_linear_offset = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_rows = raw_cap - raw_start < n_raw ? raw_cap - raw_start : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }

        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }

        raw_linear_buf = g_flash_attn_ring_buffer;
        raw_linear_offset = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         raw_linear_buf,
                                         raw_linear_offset,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }
    if (n_comp) {
        if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                             compbuf,
                                             ds4_gpu_tensor_offset(comp_kv),
                                             g_flash_attn_kv_buffer,
                                             (NSUInteger)n_raw * row_bytes_f16,
                                             n_comp * head_dim)) {
            return 0;
        }
    }

    if (!ds4_gpu_encode_fill_f16_1d(cb, g_flash_attn_mask_buffer, 0, n_keys, 0.0f)) {
        return 0;
    }
    if (use_mask && n_comp &&
        !ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         maskbuf,
                                         ds4_gpu_tensor_offset(comp_mask),
                                         g_flash_attn_mask_buffer,
                                         (NSUInteger)n_raw * sizeof(uint16_t),
                                         n_comp)) {
        return 0;
    }

    if ((n_keys % ncpsg) != 0) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = 1,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = mask_bytes,
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:g_flash_attn_mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_vec_args vec_args = {
        .ne01 = 1,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = 1,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = mask_bytes,
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = 1,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger shared_elems = (ds4_gpu_align_up_ns(head_dim, 128u) +
                                     4u * ncpsg +
                                     2u * ds4_gpu_align_up_ns(head_dim, 128u)) * nsg;
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:vec_pipeline];
    [enc setBytes:&vec_args length:sizeof(vec_args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:g_flash_attn_mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:7];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_head, nwg)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_reduce_args reduce_args = {
        .nrows = (int32_t)nrows,
    };
    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:reduce_pipeline];
    [enc setBytes:&reduce_args length:sizeof(reduce_args) atIndex:0];
    [enc setBuffer:g_flash_attn_tmp_buffer offset:0 atIndex:1];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32u * nwg, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_raw_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               window,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    if (!qbuf || !rawbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode raw batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_raw % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_raw * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_raw * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_raw + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }
        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }
        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_raw_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                         n_tokens,
                                         n_raw,
                                         pos0,
                                         window);

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_raw,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_raw * row_bytes_f16,
            .nb13 = (uint64_t)n_raw * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_raw * row_bytes_f16,
            .nb23 = (uint64_t)n_raw * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_raw,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_raw,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_raw * row_bytes_f16,
        .nb13 = (uint64_t)n_raw * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_raw * row_bytes_f16,
        .nb23 = (uint64_t)n_raw * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_raw * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

static int ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(
        id<MTLCommandBuffer>   cb,
        ds4_gpu_tensor      *heads,
        id<MTLBuffer>          sinks_buf,
        NSUInteger             sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t               use_comp_mask,
        uint32_t               n_tokens,
        uint32_t               pos0,
        uint32_t               n_raw,
        uint32_t               raw_cap,
        uint32_t               raw_start,
        uint32_t               n_comp,
        uint32_t               window,
        uint32_t               ratio,
        uint32_t               n_head,
        uint32_t               head_dim) {
    if (n_comp == 0) {
        return ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       sinks_offset,
                                                                       q,
                                                                       raw_kv,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       window,
                                                                       n_head,
                                                                       head_dim);
    }
    if (head_dim != 512 || n_head == 0 || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || !comp_kv || (use_comp_mask && !comp_mask)) {
        return 0;
    }

    const uint32_t n_keys = n_raw + n_comp;
    id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
    id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
    id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
    id<MTLBuffer> maskbuf = use_comp_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
    id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t comp_mask_bytes = use_comp_mask ? (uint64_t)n_comp * n_tokens * sizeof(float) : 0u;
    if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf || !sinks_buf ||
        ds4_gpu_tensor_bytes(q) < q_bytes ||
        ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
        ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
        (use_comp_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
        ds4_gpu_tensor_bytes(heads) < q_bytes) {
        fprintf(stderr, "ds4: Metal decode mixed batch FlashAttention received undersized buffers\n");
        return 0;
    }

    const uint32_t nqptg = 8;
    const uint32_t ncpsg = 64;
    const uint32_t nsg = head_dim >= 512 ? 8u : 4u;
    const bool has_kvpad = (n_keys % ncpsg) != 0;
    const bool bc_mask = (n_tokens % nqptg) != 0;
    const NSUInteger row_bytes = (NSUInteger)head_dim * sizeof(float);
    const NSUInteger row_bytes_f16 = (NSUInteger)head_dim * sizeof(uint16_t);
    const NSUInteger mask_bytes = (NSUInteger)n_keys * (NSUInteger)n_tokens * sizeof(uint16_t);
    const NSUInteger kv_bytes = (NSUInteger)n_keys * row_bytes_f16;
    const NSUInteger pad_bytes = has_kvpad
        ? (NSUInteger)ncpsg * (2u * row_bytes_f16 + (NSUInteger)n_tokens * sizeof(uint16_t))
        : 1u;
    const NSUInteger nblk0 = ((NSUInteger)n_keys + ncpsg - 1u) / ncpsg;
    const NSUInteger nblk1 = ((NSUInteger)n_tokens + nqptg - 1u) / nqptg;
    const NSUInteger blk_bytes = ds4_gpu_align_up_ns(nblk0 * nblk1, 32u);

    id<MTLBuffer> mask_buffer =
        ds4_gpu_new_transient_buffer(mask_bytes, "ds4_flash_attn_mask");
    if (!mask_buffer ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_kv_buffer,
                                         &g_flash_attn_kv_bytes,
                                         kv_bytes,
                                         "ds4_flash_attn_kv_f16") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_pad_buffer,
                                         &g_flash_attn_pad_bytes,
                                         pad_bytes,
                                         "ds4_flash_attn_pad") ||
        !ds4_gpu_ensure_scratch_buffer(&g_flash_attn_blk_buffer,
                                         &g_flash_attn_blk_bytes,
                                         blk_bytes,
                                         "ds4_flash_attn_blk")) {
        return 0;
    }

    id<MTLBuffer> kvbuf = rawbuf;
    NSUInteger kvoff = ds4_gpu_tensor_offset(raw_kv);
    if (raw_start != 0) {
        const NSUInteger ring_bytes = (NSUInteger)n_raw * row_bytes;
        const uint32_t tail_avail = raw_cap - raw_start;
        const uint32_t tail_rows = tail_avail < n_raw ? tail_avail : n_raw;
        const uint32_t head_rows = n_raw - tail_rows;
        if (!ds4_gpu_ensure_scratch_buffer(&g_flash_attn_ring_buffer,
                                             &g_flash_attn_ring_bytes,
                                             ring_bytes,
                                             "ds4_flash_attn_ring")) {
            return 0;
        }
        if ((tail_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv) + (NSUInteger)raw_start * row_bytes,
                                              g_flash_attn_ring_buffer,
                                              0,
                                              tail_rows * head_dim)) ||
            (head_rows &&
             !ds4_gpu_encode_cpy_f32_f32_1d(cb,
                                              rawbuf,
                                              ds4_gpu_tensor_offset(raw_kv),
                                              g_flash_attn_ring_buffer,
                                              (NSUInteger)tail_rows * row_bytes,
                                              head_rows * head_dim))) {
            return 0;
        }
        kvbuf = g_flash_attn_ring_buffer;
        kvoff = 0;
    }

    if (!ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         kvbuf,
                                         kvoff,
                                         g_flash_attn_kv_buffer,
                                         0,
                                         n_raw * head_dim) ||
        !ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                         compbuf,
                                         ds4_gpu_tensor_offset(comp_kv),
                                         g_flash_attn_kv_buffer,
                                         (NSUInteger)n_raw * row_bytes_f16,
                                         n_comp * head_dim)) {
        return 0;
    }

    ds4_gpu_fill_mixed_decode_batch_mask((uint16_t *)[mask_buffer contents],
                                           n_tokens,
                                           n_raw,
                                           n_comp,
                                           pos0,
                                           window,
                                           ratio);
    if (use_comp_mask) {
        if (!ds4_gpu_encode_cpy_f32_f16_2d(cb,
                                             maskbuf,
                                             ds4_gpu_tensor_offset(comp_mask),
                                             mask_buffer,
                                             (NSUInteger)n_raw * sizeof(uint16_t),
                                             n_comp,
                                             n_tokens,
                                             (uint64_t)n_comp * sizeof(float),
                                             (uint64_t)n_keys * sizeof(uint16_t))) {
            return 0;
        }
    }

    id<MTLComputePipelineState> pad_pipeline = nil;
    if (has_kvpad) {
        pad_pipeline = ds4_gpu_get_flash_attn_pad_pipeline(true, (int32_t)ncpsg);
        if (!pad_pipeline) return 0;
    }
    id<MTLComputePipelineState> blk_pipeline =
        ds4_gpu_get_flash_attn_blk_pipeline((int32_t)nqptg, (int32_t)ncpsg);
    id<MTLComputePipelineState> attn_pipeline =
        ds4_gpu_get_flash_attn_pipeline("kernel_flash_attn_ext_f16_dk512_dv512",
                                          true, true, false, false, has_kvpad, bc_mask,
                                          (int32_t)head_dim,
                                          (int32_t)head_dim,
                                          (int32_t)nsg);
    if (!blk_pipeline || !attn_pipeline) return 0;

    if (has_kvpad) {
        ds4_gpu_flash_attn_pad_args pad_args = {
            .ne11 = (int32_t)n_keys,
            .ne_12_2 = 1,
            .ne_12_3 = 1,
            .nb11 = row_bytes_f16,
            .nb12 = (uint64_t)n_keys * row_bytes_f16,
            .nb13 = (uint64_t)n_keys * row_bytes_f16,
            .nb21 = row_bytes_f16,
            .nb22 = (uint64_t)n_keys * row_bytes_f16,
            .nb23 = (uint64_t)n_keys * row_bytes_f16,
            .ne31 = (int32_t)n_tokens,
            .ne32 = 1,
            .ne33 = 1,
            .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
            .nb32 = mask_bytes,
            .nb33 = mask_bytes,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pad_pipeline];
        [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:1];
        [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
        [enc setBuffer:mask_buffer offset:0 atIndex:3];
        [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ncpsg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
    }

    ds4_gpu_flash_attn_blk_args blk_args = {
        .ne01 = (int32_t)n_tokens,
        .ne30 = (int32_t)n_keys,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:blk_pipeline];
    [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
    [enc setBuffer:mask_buffer offset:0 atIndex:1];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    ds4_gpu_flash_attn_vec_args args = {
        .ne01 = (int32_t)n_tokens,
        .ne02 = (int32_t)n_head,
        .ne03 = 1,
        .nb01 = (uint64_t)n_head * row_bytes,
        .nb02 = row_bytes,
        .nb03 = (uint64_t)n_tokens * n_head * row_bytes,
        .ne11 = (int32_t)n_keys,
        .ne_12_2 = 1,
        .ne_12_3 = 1,
        .ns10 = (int32_t)head_dim,
        .nb11 = row_bytes_f16,
        .nb12 = (uint64_t)n_keys * row_bytes_f16,
        .nb13 = (uint64_t)n_keys * row_bytes_f16,
        .ns20 = (int32_t)head_dim,
        .nb21 = row_bytes_f16,
        .nb22 = (uint64_t)n_keys * row_bytes_f16,
        .nb23 = (uint64_t)n_keys * row_bytes_f16,
        .ne31 = (int32_t)n_tokens,
        .ne32 = 1,
        .ne33 = 1,
        .nb31 = (uint64_t)n_keys * sizeof(uint16_t),
        .nb32 = mask_bytes,
        .nb33 = mask_bytes,
        .ne1 = (int32_t)n_head,
        .ne2 = (int32_t)n_tokens,
        .ne3 = 1,
        .scale = 1.0f / sqrtf((float)head_dim),
        .max_bias = 0.0f,
        .m0 = 0.0f,
        .m1 = 0.0f,
        .n_head_log2 = 0,
        .logit_softcap = 0.0f,
    };

    const NSUInteger padded_v = ds4_gpu_align_up_ns(head_dim, 64u);
    const NSUInteger shared_elems = (NSUInteger)nqptg *
        ((NSUInteger)head_dim + 2u * padded_v + 2u * (2u * (NSUInteger)ncpsg));
    const NSUInteger shared_bytes = ds4_gpu_align_up_ns(shared_elems * (sizeof(float) / 2u), 16u);

    enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:attn_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:2];
    [enc setBuffer:g_flash_attn_kv_buffer offset:0 atIndex:3];
    [enc setBuffer:mask_buffer offset:0 atIndex:4];
    [enc setBuffer:sinks_buf offset:sinks_offset atIndex:5];
    [enc setBuffer:g_flash_attn_pad_buffer offset:0 atIndex:6];
    [enc setBuffer:g_flash_attn_blk_buffer offset:0 atIndex:7];
    [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:8];
    [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nblk1, n_head, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0) return 0;

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_raw_heads(cb,
                                                                heads,
                                                                sinks_buf,
                                                                (NSUInteger)sinks_inner,
                                                                q,
                                                                raw_kv,
                                                                n_tokens,
                                                                window,
                                                                n_head,
                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill raw attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_raw_batch_heads(cb,
                                                                     heads,
                                                                     sinks_buf,
                                                                     (NSUInteger)sinks_inner,
                                                                     q,
                                                                     raw_kv,
                                                                     n_tokens,
                                                                     pos0,
                                                                     n_raw,
                                                                     raw_cap,
                                                                     raw_start,
                                                                     window,
                                                                     n_head,
                                                                     head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode raw batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        ratio == 0 || (n_comp != 0 && !comp_kv) ||
        (use_comp_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_decode_mixed_batch_heads(cb,
                                                                       heads,
                                                                       sinks_buf,
                                                                       (NSUInteger)sinks_inner,
                                                                       q,
                                                                       raw_kv,
                                                                       comp_kv,
                                                                       comp_mask,
                                                                       use_comp_mask,
                                                                       n_tokens,
                                                                       pos0,
                                                                       n_raw,
                                                                       raw_cap,
                                                                       raw_start,
                                                                       n_comp,
                                                                       window,
                                                                       ratio,
                                                                       n_head,
                                                                       head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph decode mixed batch attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 || top_k > n_comp || (top_k & (top_k - 1u)) != 0 ||
        ratio == 0 || n_head == 0 || head_dim != 512) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal indexed attention sinks range is outside the mapped model\n");
            return 0;
        }

        const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
        const uint64_t q_bytes = (uint64_t)n_tokens * n_head * row_bytes;
        const uint64_t raw_bytes = (uint64_t)raw_cap * row_bytes;
        const uint64_t comp_bytes = (uint64_t)n_comp * row_bytes;
        const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(int32_t);
        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = ds4_gpu_tensor_buffer(comp_kv);
        id<MTLBuffer> topkbuf = ds4_gpu_tensor_buffer(topk);
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        if (!qbuf || !rawbuf || !compbuf || !topkbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            ds4_gpu_tensor_bytes(comp_kv) < comp_bytes ||
            ds4_gpu_tensor_bytes(topk) < topk_bytes ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal indexed mixed attention received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        id<MTLComputePipelineState> sort_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_sort_i32_rows_asc_pipeline,
                                    "kernel_dsv4_sort_i32_rows_asc");
        const bool decode_one_token = n_tokens == 1u;
        id<MTLComputePipelineState> attn_pipeline =
            decode_one_token ?
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_rb4_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8_rb4") :
            ds4_gpu_hot_pipeline(g_dsv4_indexed_attention_heads8_pipeline,
                                   "kernel_dsv4_indexed_mixed_attention_heads8");
        if (!sort_pipeline || !attn_pipeline) return 0;
        if ((NSUInteger)top_k > sort_pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal indexed attention top-k exceeds sort threadgroup limit\n");
            return 0;
        }
        /*
         * Fast decode attends to the same full top-k compressed rows but keeps
         * them in score order, avoiding a chronological sort dispatch.
         * --quality restores the sorted order for stricter reproducibility.
         */
        const bool skip_decode_sort = !g_quality_mode && decode_one_token;
        if (!skip_decode_sort &&
            !ds4_gpu_ensure_scratch_buffer(&g_indexed_topk_buffer,
                                             &g_indexed_topk_bytes,
                                             (NSUInteger)topk_bytes,
                                             "ds4_indexed_topk_sorted")) {
            return 0;
        }

        ds4_gpu_dsv4_topk_mask_args sort_args = {
            .ne00 = (int64_t)top_k,
            .ne01 = (int64_t)n_tokens,
            .nb00 = sizeof(int32_t),
            .nb01 = (uint64_t)top_k * sizeof(int32_t),
            .ne0 = (int64_t)top_k,
            .ne1 = (int64_t)n_tokens,
            .nb0 = sizeof(int32_t),
            .nb1 = (uint64_t)top_k * sizeof(int32_t),
        };
        ds4_gpu_dsv4_indexed_attention_args attn_args = {
            .n_tokens = n_tokens,
            .n_head = n_head,
            .n_raw = n_raw,
            .raw_cap = raw_cap,
            .raw_start = raw_start,
            .n_comp = n_comp,
            .top_k = top_k,
            .pos0 = pos0,
            .window = window,
            .ratio = ratio,
            .q_token_stride = (uint64_t)n_head * row_bytes,
            .q_head_stride = row_bytes,
            .raw_row_stride = row_bytes,
            .comp_row_stride = row_bytes,
            .topk_token_stride = (uint64_t)top_k * sizeof(int32_t),
            .dst_token_stride = (uint64_t)n_head * row_bytes,
            .dst_head_stride = row_bytes,
            .scale = 1.0f / sqrtf((float)head_dim),
        };

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = nil;
        if (!skip_decode_sort) {
            enc = ds4_gpu_compute_encoder(cb);
            [enc setComputePipelineState:sort_pipeline];
            [enc setBytes:&sort_args length:sizeof(sort_args) atIndex:0];
            [enc setBuffer:topkbuf offset:ds4_gpu_tensor_offset(topk) atIndex:1];
            [enc setBuffer:g_indexed_topk_buffer offset:0 atIndex:2];
            [enc setThreadgroupMemoryLength:(NSUInteger)top_k * sizeof(int32_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(n_tokens, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(top_k, 1, 1)];
            ds4_gpu_end_compute_encoder(cb, enc);
        }

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:attn_pipeline];
        [enc setBytes:&attn_args length:sizeof(attn_args) atIndex:0];
        [enc setBuffer:qbuf offset:ds4_gpu_tensor_offset(q) atIndex:1];
        [enc setBuffer:rawbuf offset:ds4_gpu_tensor_offset(raw_kv) atIndex:2];
        [enc setBuffer:compbuf offset:ds4_gpu_tensor_offset(comp_kv) atIndex:3];
        [enc setBuffer:skip_decode_sort ? topkbuf : g_indexed_topk_buffer
              offset:skip_decode_sort ? ds4_gpu_tensor_offset(topk) : 0
             atIndex:4];
        [enc setBuffer:sinks_buf offset:(NSUInteger)sinks_inner atIndex:5];
        [enc setBuffer:headsbuf offset:ds4_gpu_tensor_offset(heads) atIndex:6];
        [enc setThreadgroupMemoryLength:(decode_one_token ? 4u : 1u) * 128u * 4u * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, ((NSUInteger)n_head + 7u) / 8u, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph indexed mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        ratio == 0 || (n_comp != 0 && !comp_kv)) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                NULL,
                                                                                0,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill static mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !q || !raw_kv || !comp_kv || !comp_mask || !model_map ||
        n_tokens == 0 || n_comp == 0 || ratio == 0) {
        return 0;
    }

    @autoreleasepool {
        if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal attention sinks range is outside the mapped model\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                             sinks_offset,
                                                             (uint64_t)n_head * sizeof(float),
                                                             &sinks_inner);
        if (!sinks_buf) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_prefill_static_mixed_heads_nonvec(cb,
                                                                                heads,
                                                                                sinks_buf,
                                                                                (NSUInteger)sinks_inner,
                                                                                q,
                                                                                raw_kv,
                                                                                comp_kv,
                                                                                comp_mask,
                                                                                1,
                                                                                n_tokens,
                                                                                n_comp,
                                                                                window,
                                                                                ratio,
                                                                                n_head,
                                                                                head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph prefill masked mixed attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        n_raw > UINT32_MAX - n_comp || n_raw + n_comp > 8192u ||
        (n_comp != 0 && !comp_kv) ||
        (use_mask != 0 && !comp_mask)) {
        return 0;
    }

    @autoreleasepool {
        const uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
        const uint64_t raw_bytes = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
        const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
        if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
            fprintf(stderr, "ds4: Metal graph attention heads sink range is outside the mapped model\n");
            return 0;
        }

        id<MTLBuffer> qbuf = ds4_gpu_tensor_buffer(q);
        id<MTLBuffer> rawbuf = ds4_gpu_tensor_buffer(raw_kv);
        id<MTLBuffer> compbuf = n_comp ? ds4_gpu_tensor_buffer(comp_kv) : rawbuf;
        id<MTLBuffer> maskbuf = use_mask ? ds4_gpu_tensor_buffer(comp_mask) : rawbuf;
        id<MTLBuffer> headsbuf = ds4_gpu_tensor_buffer(heads);
        const uint64_t comp_mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
        if (!qbuf || !rawbuf || !compbuf || !maskbuf || !headsbuf ||
            ds4_gpu_tensor_bytes(q) < q_bytes ||
            ds4_gpu_tensor_bytes(raw_kv) < raw_bytes ||
            (n_comp && ds4_gpu_tensor_bytes(comp_kv) < comp_bytes) ||
            (use_mask && ds4_gpu_tensor_bytes(comp_mask) < comp_mask_bytes) ||
            ds4_gpu_tensor_bytes(heads) < q_bytes) {
            fprintf(stderr, "ds4: Metal graph attention heads received undersized buffers\n");
            return 0;
        }

        uint64_t sinks_inner = 0;
        id<MTLBuffer> sinks_buf = ds4_gpu_wrap_model_range(model_map, model_size, sinks_offset, sink_bytes, &sinks_inner);
        if (!sinks_buf) return 0;

        if (n_comp == 0) {
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;

            if (!ds4_gpu_encode_flash_attention_raw_heads(cb,
                                                            heads,
                                                            sinks_buf,
                                                            (NSUInteger)sinks_inner,
                                                            q,
                                                            raw_kv,
                                                            n_raw,
                                                            raw_cap,
                                                            raw_start,
                                                            n_head,
                                                            head_dim)) {
                return 0;
            }

            if (!ds4_gpu_finish_command_buffer(cb, owned, "graph raw attention heads")) return 0;
            return 1;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        if (!ds4_gpu_encode_flash_attention_gathered_heads(cb,
                                                             heads,
                                                             sinks_buf,
                                                             (NSUInteger)sinks_inner,
                                                             q,
                                                             raw_kv,
                                                             n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             comp_kv,
                                                             n_comp,
                                                             comp_mask,
                                                             use_mask,
                                                             n_head,
                                                             head_dim)) {
            return 0;
        }

        if (!ds4_gpu_finish_command_buffer(cb, owned, "graph attention heads")) return 0;
    }

    return 1;
}

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (fabsf(clamp) > 1.0e-12f || fabsf(weight - 1.0f) > 1.0e-12f) {
        fprintf(stderr, "ds4: Metal SwiGLU kernel does not support clamp/weight\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!gatebuf || !upbuf || !outbuf ||
            ds4_gpu_tensor_bytes(gate) < bytes ||
            ds4_gpu_tensor_bytes(up) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal SwiGLU received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        ds4_gpu_glu_args args = {
            .ne00 = (int32_t)n,
            .nb01 = (uint64_t)n * sizeof(float),
            .ne10 = (int32_t)n,
            .nb11 = (uint64_t)n * sizeof(float),
            .ne0 = (int32_t)n,
            .nb1 = (uint64_t)n * sizeof(float),
            .i00 = 0,
            .i10 = 0,
            .alpha = 0.0f,
            .limit = 0.0f,
        };
        NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
        const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
        if (nth > ds4_nth) nth = ds4_nth;
        if (nth == 0) nth = 1;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_swiglu_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:gatebuf offset:ds4_gpu_tensor_offset(gate) atIndex:1];
        [enc setBuffer:upbuf offset:ds4_gpu_tensor_offset(up) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "SwiGLU")) return 0;
    }

    return 1;
}

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> abuf = ds4_gpu_tensor_buffer(a);
        id<MTLBuffer> bbuf = ds4_gpu_tensor_buffer(b);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t bytes = (uint64_t)n * sizeof(float);
        if (!abuf || !bbuf || !outbuf ||
            ds4_gpu_tensor_bytes(a) < bytes ||
            ds4_gpu_tensor_bytes(b) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal tensor add received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint64_t row_bytes = (uint64_t)n * sizeof(float);
        ds4_gpu_bin_args args = {
            .ne00 = (int32_t)n,
            .ne01 = 1,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = sizeof(float),
            .nb01 = row_bytes,
            .nb02 = row_bytes,
            .nb03 = row_bytes,
            .ne10 = (int32_t)n,
            .ne11 = 1,
            .ne12 = 1,
            .ne13 = 1,
            .nb10 = sizeof(float),
            .nb11 = row_bytes,
            .nb12 = row_bytes,
            .nb13 = row_bytes,
            .ne0 = (int32_t)n,
            .ne1 = 1,
            .ne2 = 1,
            .ne3 = 1,
            .nb0 = sizeof(float),
            .nb1 = row_bytes,
            .nb2 = row_bytes,
            .nb3 = row_bytes,
            .offs = 0,
            .o1 = { 0 },
        };
        NSUInteger nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth_max > 256u) nth_max = 256u;
        NSUInteger nth = 1;
        while (2u * nth < (NSUInteger)args.ne0 && nth < nth_max) {
            nth *= 2u;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:abuf offset:ds4_gpu_tensor_offset(a) atIndex:1];
        [enc setBuffer:bbuf offset:ds4_gpu_tensor_offset(b) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "tensor add")) return 0;
    }

    return 1;
}

typedef struct {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
} ds4_gpu_directional_steering_project_args;

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_dsv4_directional_steering_project_f32");
        if (!pipeline) return 0;

        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> dbuf = ds4_gpu_tensor_buffer(directions);
        const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
        const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
        if (!xbuf || !dbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(directions) < dir_bytes) {
            fprintf(stderr, "ds4: Metal directional steering received undersized buffers\n");
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        while (nth > width && nth > 1u) nth >>= 1;
        if (nth == 0) nth = 1;

        ds4_gpu_directional_steering_project_args args = {
            .width = width,
            .rows = rows,
            .layer = layer,
            .n_threads = (uint32_t)nth,
            .scale = scale,
        };

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:1];
        [enc setBuffer:dbuf offset:ds4_gpu_tensor_offset(directions) atIndex:2];
        [enc setThreadgroupMemoryLength:nth * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "directional steering")) return 0;
    }

    return 1;
}

static NSUInteger ds4_gpu_bin_threads(uint32_t width, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = 1u;
    while (2u * nth < (NSUInteger)width && nth < nth_max) nth *= 2u;
    return nth ? nth : 1u;
}

static int ds4_gpu_encode_unary_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer>               src,
        NSUInteger                  src_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        uint32_t                    width,
        uint32_t                    rows,
        int                         c4,
        float                       min,
        float                       max) {
    if (!cb || !pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if (c4 && (width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, c4, 0.0f, 0.0f);
    args.min = min;
    args.max = max;

    NSUInteger nth_max = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_scale_f32_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows,
        float                scale) {
    if (!cb || !g_unary_scale_pipeline || !src || !dst || width == 0 || rows == 0) return 0;
    if ((width & 3u) != 0) return 0;

    ds4_gpu_unary_args args = ds4_gpu_make_unary_rows_args(width, rows, 1, scale, 0.0f);
    NSUInteger nth_max = g_unary_scale_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth_max > 256u) nth_max = 256u;
    NSUInteger nth = (NSUInteger)args.ne00;
    if (nth > nth_max) nth = nth_max;
    if (nth == 0) nth = 1u;
    const NSUInteger nk0 = ((NSUInteger)args.ne00 + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_unary_scale_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nk0 * (NSUInteger)args.ne01,
                                          (NSUInteger)args.ne02,
                                          (NSUInteger)args.ne03)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static uint16_t ds4_gpu_f32_to_f16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) {
        m = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

static float ds4_gpu_f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static int ds4_gpu_ane_scalar_output_pack_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_SCALAR_OUTPUT_PACK");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_gpu_output_pack_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_GPU_OUTPUT_PACK");
    return env && env[0] && atoi(env) != 0;
}

static void ds4_gpu_f16_to_weighted_f32_row(float *dst,
                                            const uint16_t *src,
                                            uint32_t n,
                                            float weight,
                                            int force_scalar) {
#if defined(__ARM_NEON)
    if (!force_scalar) {
        uint32_t i = 0;
        const float32x4_t wv = vdupq_n_f32(weight);
        for (; i + 8u <= n; i += 8u) {
            const uint16x8_t hu = vld1q_u16(src + i);
            const float16x8_t hv = vreinterpretq_f16_u16(hu);
            const float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(hv)), wv);
            const float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(hv)), wv);
            vst1q_f32(dst + i, lo);
            vst1q_f32(dst + i + 4u, hi);
        }
        for (; i < n; i++) {
            dst[i] = ds4_gpu_f16_bits_to_f32(src[i]) * weight;
        }
        return;
    }
#else
    (void)force_scalar;
#endif
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = ds4_gpu_f16_bits_to_f32(src[i]) * weight;
    }
}

static int ds4_gpu_encode_bin_f32_rows(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_bin_args   *args,
        id<MTLBuffer>               a,
        NSUInteger                  a_off,
        id<MTLBuffer>               b,
        NSUInteger                  b_off,
        id<MTLBuffer>               out,
        NSUInteger                  out_off) {
    if (!cb || !pipeline || !args || !a || !b || !out || args->ne0 <= 0 || args->ne1 <= 0) {
        return 0;
    }

    const NSUInteger nth = ds4_gpu_bin_threads((uint32_t)args->ne0, pipeline);
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:a offset:a_off atIndex:1];
    [enc setBuffer:b offset:b_off atIndex:2];
    [enc setBuffer:out offset:out_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)args->ne1,
                                          (NSUInteger)args->ne2,
                                          (NSUInteger)args->ne3)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_bin_rowwise_scalar_args(uint32_t width, uint32_t rows) {
    const uint64_t lhs_row_bytes = (uint64_t)width * sizeof(float);
    const uint64_t rhs_row_bytes = sizeof(float);
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)width,
        .ne01 = (int32_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = lhs_row_bytes,
        .nb02 = (uint64_t)rows * lhs_row_bytes,
        .nb03 = (uint64_t)rows * lhs_row_bytes,
        .ne10 = 1,
        .ne11 = (int32_t)rows,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = rhs_row_bytes,
        .nb12 = (uint64_t)rows * rhs_row_bytes,
        .nb13 = (uint64_t)rows * rhs_row_bytes,
        .ne0 = (int32_t)width,
        .ne1 = (int32_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = lhs_row_bytes,
        .nb2 = (uint64_t)rows * lhs_row_bytes,
        .nb3 = (uint64_t)rows * lhs_row_bytes,
        .offs = 0,
        .o1 = { 0 },
    };
}

static ds4_gpu_mul_mv_id_args ds4_gpu_make_mul_mv_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t nr0) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    const uint64_t src0_blocks = src0_cols / 256u;
    const uint64_t src0_block_bytes = src0_blocks ? src0_row_bytes / src0_blocks : 1u;
    return (ds4_gpu_mul_mv_id_args) {
        .nei0 = (int32_t)selected_experts,
        .nei1 = (int32_t)n_tokens,
        .nbi1 = (uint64_t)selected_experts * sizeof(int32_t),
        .ne00 = (int32_t)src0_cols,
        .ne01 = (int32_t)src0_rows,
        .ne02 = (int32_t)src0_experts,
        .nb00 = src0_block_bytes,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .ne12 = (int32_t)n_tokens,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .nb1 = (uint64_t)src0_rows * sizeof(float),
        .nr0 = (int32_t)nr0,
    };
}

static ds4_gpu_mul_mm_id_map_args ds4_gpu_make_mul_mm_id_map_args(
        uint32_t src0_cols,
        uint32_t src0_experts,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * sizeof(float);
    return (ds4_gpu_mul_mm_id_map_args) {
        .ne02 = (int32_t)src0_experts,
        .ne10 = (int32_t)src0_cols,
        .ne11 = (int32_t)src1_expert_rows,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne21 = (int32_t)n_tokens,
        .ne20 = (int32_t)selected_experts,
        .nb21 = (uint64_t)selected_experts * sizeof(int32_t),
    };
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens) {
    return ds4_gpu_make_mul_mm_id_args_src1_size(src0_cols,
                                                   src0_rows,
                                                   src0_experts,
                                                   src0_row_bytes,
                                                   src0_expert_bytes,
                                                   src1_expert_rows,
                                                   selected_experts,
                                                   n_tokens,
                                                   sizeof(float));
}

static ds4_gpu_mul_mm_id_args ds4_gpu_make_mul_mm_id_args_src1_size(
        uint32_t src0_cols,
        uint32_t src0_rows,
        uint32_t src0_experts,
        uint64_t src0_row_bytes,
        uint64_t src0_expert_bytes,
        uint32_t src1_expert_rows,
        uint32_t selected_experts,
        uint32_t n_tokens,
        uint32_t src1_elem_size) {
    const uint64_t src1_row_bytes = (uint64_t)src0_cols * src1_elem_size;
    return (ds4_gpu_mul_mm_id_args) {
        .ne00 = (int32_t)src0_cols,
        .ne02 = (int32_t)src0_experts,
        .nb01 = src0_row_bytes,
        .nb02 = src0_expert_bytes,
        .nb03 = (uint64_t)src0_experts * src0_expert_bytes,
        .ne11 = (int32_t)src1_expert_rows,
        .nb10 = src1_elem_size,
        .nb11 = src1_row_bytes,
        .nb12 = (uint64_t)src1_expert_rows * src1_row_bytes,
        .nb13 = (uint64_t)n_tokens * (uint64_t)src1_expert_rows * src1_row_bytes,
        .ne20 = (int32_t)selected_experts,
        .ne21 = (int32_t)n_tokens,
        .ne0 = (int32_t)src0_rows,
        .ne1 = (int32_t)selected_experts,
        .r2 = 1,
        .r3 = 1,
    };
}

static uint32_t ds4_gpu_routed_mv_nr0(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_Q4_K:    return 2;
    case DS4_METAL_TENSOR_Q2_K:
    case DS4_METAL_TENSOR_IQ2_XXS: return 4;
    default:                       return 0;
    }
}

static NSUInteger ds4_gpu_routed_mv_smem(uint32_t type) {
    if (type == DS4_METAL_TENSOR_IQ2_XXS) {
        return 256u * sizeof(uint64_t) + 128u * sizeof(uint8_t);
    }
    return 0;
}

static id<MTLComputePipelineState> ds4_gpu_routed_mv_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS: return g_moe_mul_mv_id_iq2_xxs_pipeline;
    case DS4_METAL_TENSOR_Q2_K:    return g_moe_mul_mv_id_q2_k_pipeline;
    case DS4_METAL_TENSOR_Q4_K:    return g_moe_mul_mv_id_q4_k_pipeline;
    default:                       return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        if (!g_moe_mul_mm_id_iq2_xxs_pipeline) {
            g_moe_mul_mm_id_iq2_xxs_pipeline =
                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f32", false);
        }
        return g_moe_mul_mm_id_iq2_xxs_pipeline;
    case DS4_METAL_TENSOR_Q2_K:
        if (!g_moe_mul_mm_id_q2_k_pipeline) {
            g_moe_mul_mm_id_q2_k_pipeline =
                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f32", false);
        }
        return g_moe_mul_mm_id_q2_k_pipeline;
    case DS4_METAL_TENSOR_Q4_K:
        if (!g_moe_mul_mm_id_q4_k_pipeline) {
            g_moe_mul_mm_id_q4_k_pipeline =
                ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f32", false);
        }
        return g_moe_mul_mm_id_q4_k_pipeline;
    default:
        return nil;
    }
}

static id<MTLComputePipelineState> ds4_gpu_routed_mm_f16_rhs_pipeline(uint32_t type) {
    switch (type) {
    case DS4_METAL_TENSOR_IQ2_XXS:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_iq2_xxs_f16", false);
    case DS4_METAL_TENSOR_Q2_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q2_K_f16", false);
    case DS4_METAL_TENSOR_Q4_K:
        return ds4_gpu_get_mul_mm_id_pipeline("kernel_mul_mm_id_q4_K_f16", false);
    default:
        return nil;
    }
}

static int ds4_gpu_encode_mul_mv_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_attn_out_low_q8_direct(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0_a offset:src0_a_off atIndex:1];
    [enc setBuffer:src0_b offset:src0_b_off atIndex:2];
    [enc setBuffer:src1   offset:src1_off   atIndex:3];
    [enc setBuffer:dst_a  offset:dst_a_off  atIndex:4];
    [enc setBuffer:dst_b  offset:dst_b_off  atIndex:5];
    [enc setBuffer:ids    offset:ids_off    atIndex:6];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_pair_swiglu(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        const ds4_gpu_dsv4_moe_swiglu_weight_args *act,
        id<MTLBuffer>               src0_a,
        NSUInteger                  src0_a_off,
        id<MTLBuffer>               src0_b,
        NSUInteger                  src0_b_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst_a,
        NSUInteger                  dst_a_off,
        id<MTLBuffer>               dst_b,
        NSUInteger                  dst_b_off,
        id<MTLBuffer>               dst_mid,
        NSUInteger                  dst_mid_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        id<MTLBuffer>               weights,
        NSUInteger                  weights_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg,
        bool                        rows_per_group_is_nr0) {
    if (!cb || !pipeline || !args || !act ||
        !src0_a || !src0_b || !src1 || !dst_a || !dst_b || !dst_mid || !ids || !weights ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 <= 0 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger nr0 = (NSUInteger)args->nr0;
    const NSUInteger rows_per_group = rows_per_group_is_nr0 ? nr0 : nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;
    const NSUInteger pairs = (NSUInteger)args->nei0 * (NSUInteger)args->nei1;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBytes:act  length:sizeof(*act)  atIndex:1];
    [enc setBuffer:src0_a  offset:src0_a_off  atIndex:2];
    [enc setBuffer:src0_b  offset:src0_b_off  atIndex:3];
    [enc setBuffer:src1    offset:src1_off    atIndex:4];
    [enc setBuffer:dst_a   offset:dst_a_off   atIndex:5];
    [enc setBuffer:dst_b   offset:dst_b_off   atIndex:6];
    [enc setBuffer:dst_mid offset:dst_mid_off atIndex:7];
    [enc setBuffer:ids     offset:ids_off     atIndex:8];
    [enc setBuffer:weights offset:weights_off atIndex:9];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, 1, pairs)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mv_id_sum6(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> pipeline,
        const ds4_gpu_mul_mv_id_args *args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off,
        NSUInteger                  threadgroup_bytes,
        NSUInteger                  nsg) {
    if (!cb || !pipeline || !args || !src0 || !src1 || !dst || !ids ||
        args->ne00 <= 0 || args->ne01 <= 0 || args->nei0 != 6 || args->nei1 <= 0) {
        return 0;
    }

    const NSUInteger rows_per_group = (NSUInteger)args->nr0 * nsg;
    const NSUInteger row_groups = ((NSUInteger)args->ne01 + rows_per_group - 1u) / rows_per_group;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:dst  offset:dst_off  atIndex:3];
    [enc setBuffer:ids  offset:ids_off  atIndex:4];
    if (threadgroup_bytes != 0) {
        [enc setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(row_groups, (NSUInteger)args->nei1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !mm_pipeline || !map_args || !mm_args ||
        !src0 || !src1 || !dst || !ids ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    return ds4_gpu_encode_mul_mm_id_map(cb,
                                          map_pipeline,
                                          map_args,
                                          mm_args,
                                          ids,
                                          ids_off) &&
           ds4_gpu_encode_mul_mm_id_mapped(cb,
                                             mm_pipeline,
                                             mm_args,
                                             src0,
                                             src0_off,
                                             src1,
                                             src1_off,
                                             dst,
                                             dst_off);
}

static int ds4_gpu_encode_mul_mm_id_map(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> map_pipeline,
        const ds4_gpu_mul_mm_id_map_args *map_args,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               ids,
        NSUInteger                  ids_off) {
    if (!cb || !map_pipeline || !map_args || !mm_args || !ids ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes) return 0;
    if (!ds4_gpu_ensure_scratch_buffer(&g_moe_id_map_buffer,
                                         &g_moe_id_map_bytes,
                                         tpe_bytes + hids_bytes,
                                         "ds4_moe_id_map")) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:map_pipeline];
    [enc setBytes:map_args length:sizeof(*map_args) atIndex:0];
    [enc setBuffer:ids offset:ids_off atIndex:1];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:3];
    [enc setThreadgroupMemoryLength:(NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne20 * sizeof(uint16_t) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake((NSUInteger)mm_args->ne02, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mul_mm_id_mapped(
        id<MTLCommandBuffer>        cb,
        id<MTLComputePipelineState> mm_pipeline,
        const ds4_gpu_mul_mm_id_args *mm_args,
        id<MTLBuffer>               src0,
        NSUInteger                  src0_off,
        id<MTLBuffer>               src1,
        NSUInteger                  src1_off,
        id<MTLBuffer>               dst,
        NSUInteger                  dst_off) {
    if (!cb || !mm_pipeline || !mm_args || !src0 || !src1 || !dst ||
        !g_moe_id_map_buffer ||
        mm_args->ne00 <= 0 || mm_args->ne0 <= 0 ||
        mm_args->ne20 <= 0 || mm_args->ne21 <= 0 || mm_args->ne02 <= 0) {
        return 0;
    }

    const NSUInteger tpe_bytes = (NSUInteger)mm_args->ne02 * sizeof(int32_t);
    const NSUInteger hids_bytes = (NSUInteger)mm_args->ne02 * (NSUInteger)mm_args->ne21 * sizeof(int32_t);
    if (tpe_bytes > NSUIntegerMax - hids_bytes ||
        g_moe_id_map_bytes < tpe_bytes + hids_bytes) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:mm_pipeline];
    [enc setBytes:mm_args length:sizeof(*mm_args) atIndex:0];
    [enc setBuffer:src0 offset:src0_off atIndex:1];
    [enc setBuffer:src1 offset:src1_off atIndex:2];
    [enc setBuffer:g_moe_id_map_buffer offset:0 atIndex:3];
    [enc setBuffer:g_moe_id_map_buffer offset:tpe_bytes atIndex:4];
    [enc setBuffer:dst offset:dst_off atIndex:5];
    [enc setThreadgroupMemoryLength:8192u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mm_args->ne21 + 31u) / 32u,
                                          ((NSUInteger)mm_args->ne0 + 63u) / 64u,
                                          (NSUInteger)mm_args->ne02)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_swiglu_flat(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             n) {
    if (!cb || !gate || !up || !out || n == 0) return 0;

    ds4_gpu_glu_args args = {
        .ne00 = (int32_t)n,
        .nb01 = (uint64_t)n * sizeof(float),
        .ne10 = (int32_t)n,
        .nb11 = (uint64_t)n * sizeof(float),
        .ne0 = (int32_t)n,
        .nb1 = (uint64_t)n * sizeof(float),
        .i00 = 0,
        .i10 = 0,
        .alpha = 0.0f,
        .limit = 0.0f,
    };
    NSUInteger nth = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger ds4_nth = n > 1 ? (NSUInteger)n / 2u : 1u;
    if (nth > ds4_nth) nth = ds4_nth;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_swiglu_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate offset:gate_off atIndex:1];
    [enc setBuffer:up   offset:up_off   atIndex:2];
    [enc setBuffer:out  offset:out_off  atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_moe_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate,
        NSUInteger           gate_off,
        id<MTLBuffer>        up,
        NSUInteger           up_off,
        id<MTLBuffer>        mid,
        NSUInteger           mid_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows,
        float                clamp_value,
        bool                 mid_f16) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;

    id<MTLComputePipelineState> pipeline =
        ds4_gpu_get_pipeline(mid_f16 ? "kernel_dsv4_moe_swiglu_weight_f16" :
                                         "kernel_dsv4_moe_swiglu_weight");
    if (!pipeline) return 0;

    ds4_gpu_dsv4_moe_swiglu_weight_args args = {
        .width = width,
        .rows = rows,
        .gate_row_stride = (uint64_t)width * sizeof(float),
        .up_row_stride = (uint64_t)width * sizeof(float),
        .mid_row_stride = (uint64_t)width * (mid_f16 ? sizeof(uint16_t) : sizeof(float)),
        .weight_stride = sizeof(float),
        .write_clamped = getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ? 1u : 0u,
        .clamp_value = clamp_value,
    };

    NSUInteger nth = pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:gate    offset:gate_off    atIndex:1];
    [enc setBuffer:up      offset:up_off      atIndex:2];
    [enc setBuffer:mid     offset:mid_off     atIndex:3];
    [enc setBuffer:weights offset:weights_off atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static ds4_gpu_bin_args ds4_gpu_make_moe_add_args(
        uint32_t out_dim,
        uint32_t n_tokens,
        uint64_t src0_token_stride,
        uint64_t src1_token_stride,
        uint64_t dst_token_stride) {
    return (ds4_gpu_bin_args) {
        .ne00 = (int32_t)out_dim,
        .ne01 = (int32_t)n_tokens,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src0_token_stride,
        .nb02 = (uint64_t)n_tokens * src0_token_stride,
        .nb03 = (uint64_t)n_tokens * src0_token_stride,
        .ne10 = (int32_t)out_dim,
        .ne11 = (int32_t)n_tokens,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = src1_token_stride,
        .nb12 = (uint64_t)n_tokens * src1_token_stride,
        .nb13 = (uint64_t)n_tokens * src1_token_stride,
        .ne0 = (int32_t)out_dim,
        .ne1 = (int32_t)n_tokens,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = dst_token_stride,
        .nb2 = (uint64_t)n_tokens * dst_token_stride,
        .nb3 = (uint64_t)n_tokens * dst_token_stride,
        .offs = 0,
        .o1 = { 0 },
    };
}

static int ds4_gpu_encode_moe_sum_experts(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        experts,
        NSUInteger           experts_off,
        id<MTLBuffer>        out,
        NSUInteger           out_off,
        uint32_t             out_dim,
        uint32_t             n_expert,
        uint32_t             n_tokens) {
    if (!cb || !experts || !out || out_dim == 0 || n_expert < 2 || n_tokens == 0) return 0;

    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_token_stride = (uint64_t)n_expert * out_row_bytes;

    ds4_gpu_bin_args first =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, expert_token_stride, expert_token_stride, out_row_bytes);
    if (!ds4_gpu_encode_bin_f32_rows(cb,
                                       g_add_pipeline,
                                       &first,
                                       experts,
                                       experts_off,
                                       experts,
                                       experts_off + (NSUInteger)out_row_bytes,
                                       out,
                                       out_off)) {
        return 0;
    }

    ds4_gpu_bin_args accum =
        ds4_gpu_make_moe_add_args(out_dim, n_tokens, out_row_bytes, expert_token_stride, out_row_bytes);
    for (uint32_t slot = 2; slot < n_expert; slot++) {
        if (!ds4_gpu_encode_bin_f32_rows(cb,
                                           g_add_pipeline,
                                           &accum,
                                           out,
                                           out_off,
                                           experts,
                                           experts_off + (NSUInteger)((uint64_t)slot * out_row_bytes),
                                           out,
                                           out_off)) {
            return 0;
        }
    }
    return 1;
}

static int ds4_gpu_encode_get_rows_i32_token_rows(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        table,
        NSUInteger           table_off,
        id<MTLBuffer>        tokens,
        NSUInteger           tokens_off,
        const int32_t       *token_inline,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        uint32_t             hash_rows,
        uint32_t             n_tokens) {
    if (!cb || !table || !selected || hash_rows == 0 || n_tokens == 0) return 0;
    if (!tokens && !token_inline) return 0;

    const uint64_t table_row_bytes = 6u * sizeof(int32_t);
    const uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    ds4_gpu_get_rows_args args = {
        .ne00t = 6,
        .ne00 = 6,
        .nb01 = table_row_bytes,
        .nb02 = (uint64_t)hash_rows * table_row_bytes,
        .nb03 = (uint64_t)hash_rows * table_row_bytes,
        .ne10 = (int32_t)n_tokens,
        .nb10 = sizeof(int32_t),
        .nb11 = token_bytes,
        .nb12 = token_bytes,
        .nb1 = table_row_bytes,
        .nb2 = (uint64_t)n_tokens * table_row_bytes,
        .nb3 = (uint64_t)n_tokens * table_row_bytes,
    };

    NSUInteger nth = 6u;
    const NSUInteger max_threads = g_get_rows_i32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > max_threads) nth = max_threads;
    if (nth == 0) nth = 1u;
    const NSUInteger nw0 = (6u + nth - 1u) / nth;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_i32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:table offset:table_off atIndex:1];
    if (tokens) {
        [enc setBuffer:tokens offset:tokens_off atIndex:2];
    } else {
        [enc setBytes:token_inline length:sizeof(*token_inline) atIndex:2];
    }
    [enc setBuffer:selected offset:selected_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(nw0 * n_tokens, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_get_rows_f32_router_weights(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        probs,
        NSUInteger           probs_off,
        id<MTLBuffer>        selected,
        NSUInteger           selected_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             n_tokens) {
    if (!cb || !probs || !selected || !weights || n_tokens == 0) return 0;

    const uint64_t probs_token_bytes = 256u * sizeof(float);
    const uint64_t selected_row_bytes = 6u * sizeof(int32_t);
    const uint64_t weights_row_bytes = 6u * sizeof(float);
    ds4_gpu_get_rows_args args = {
        .ne00t = 1,
        .ne00 = 1,
        .nb01 = sizeof(float),
        .nb02 = probs_token_bytes,
        .nb03 = (uint64_t)n_tokens * probs_token_bytes,
        .ne10 = 6,
        .nb10 = sizeof(int32_t),
        .nb11 = selected_row_bytes,
        .nb12 = (uint64_t)n_tokens * selected_row_bytes,
        .nb1 = sizeof(float),
        .nb2 = weights_row_bytes,
        .nb3 = (uint64_t)n_tokens * weights_row_bytes,
    };

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_get_rows_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:probs offset:probs_off atIndex:1];
    [enc setBuffer:selected offset:selected_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(6u, n_tokens, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_sum_rows_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || width == 0 || rows == 0) return 0;

    const uint64_t src_row_bytes = (uint64_t)width * sizeof(float);
    ds4_gpu_kargs_sum_rows args = {
        .ne00 = (int64_t)width,
        .ne01 = (int64_t)rows,
        .ne02 = 1,
        .ne03 = 1,
        .nb00 = sizeof(float),
        .nb01 = src_row_bytes,
        .nb02 = (uint64_t)rows * src_row_bytes,
        .nb03 = (uint64_t)rows * src_row_bytes,
        .ne0 = 1,
        .ne1 = (int64_t)rows,
        .ne2 = 1,
        .ne3 = 1,
        .nb0 = sizeof(float),
        .nb1 = sizeof(float),
        .nb2 = (uint64_t)rows * sizeof(float),
        .nb3 = (uint64_t)rows * sizeof(float),
    };

    NSUInteger nth = 32u;
    const NSUInteger max_threads = g_sum_rows_f32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)args.ne00 && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)args.ne00) nth = (NSUInteger)args.ne00;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_sum_rows_f32_f32_pipeline];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBuffer:dst offset:dst_off atIndex:2];
    [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_router_select(
        id<MTLCommandBuffer>  cb,
        ds4_gpu_tensor     *selected,
        ds4_gpu_tensor     *weights,
        ds4_gpu_tensor     *probs,
        id<MTLBuffer>         logitsbuf,
        NSUInteger            logits_off,
        id<MTLBuffer>         biasbuf,
        NSUInteger            bias_off,
        id<MTLBuffer>         hashbuf,
        NSUInteger            hash_off,
        id<MTLBuffer>         tokensbuf,
        NSUInteger            tokens_off,
        const int32_t        *single_token,
        uint32_t              hash_rows,
        uint32_t              n_tokens,
        bool                  has_bias,
        bool                  hash_mode) {
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
    id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
    const NSUInteger selected_off = ds4_gpu_tensor_offset(selected);
    const NSUInteger weights_off = ds4_gpu_tensor_offset(weights);
    const NSUInteger probs_off = ds4_gpu_tensor_offset(probs);

    if (!cb || !selectedbuf || !weightsbuf || !probsbuf || !logitsbuf || n_tokens == 0) return 0;

    const NSUInteger probs_bytes = (NSUInteger)n_tokens * 256u * sizeof(float);

    int ok = 0;
    if (!g_quality_mode && n_tokens == 1 &&
        getenv("DS4_METAL_DISABLE_ROUTER_SELECT_FUSION") == NULL) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        id<MTLComputePipelineState> router_finalize_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_finalize_one_pipeline,
                                    "kernel_dsv4_router_finalize_one");
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!softplus_sqrt_pipeline || !router_finalize_pipeline || !router_weights_pipeline) return 0;

        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             256,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
        if (!ok) return 0;

        const bool use_token_buffer = single_token == NULL;
        ds4_gpu_dsv4_router_select_one_args args = {
            .has_bias = has_bias ? 1u : 0u,
            .hash_mode = hash_mode ? 1u : 0u,
            .use_token_buffer = use_token_buffer ? 1u : 0u,
            .token = single_token ? (uint32_t)*single_token : 0u,
            .hash_rows = hash_rows,
        };

        const float zero_f32 = 0.0f;
        const int32_t zero_i32 = 0;
        if ((has_bias && !biasbuf) ||
            (hash_mode && !hashbuf) ||
            (use_token_buffer && !tokensbuf)) {
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_finalize_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:probsbuf offset:probs_off atIndex:1];
        if (has_bias) {
            [enc setBuffer:biasbuf offset:bias_off atIndex:2];
        } else {
            [enc setBytes:&zero_f32 length:sizeof(zero_f32) atIndex:2];
        }
        if (hash_mode) {
            [enc setBuffer:hashbuf offset:hash_off atIndex:3];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:3];
        }
        if (use_token_buffer) {
            [enc setBuffer:tokensbuf offset:tokens_off atIndex:4];
        } else {
            [enc setBytes:&zero_i32 length:sizeof(zero_i32) atIndex:4];
        }
        [enc setBuffer:selectedbuf offset:selected_off atIndex:5];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) + 256u * sizeof(int32_t) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    const NSUInteger sum_bytes = (NSUInteger)n_tokens * sizeof(float);
    if (!ds4_gpu_ensure_scratch_buffer(&g_router_weight_sum_buffer,
                                         &g_router_weight_sum_bytes,
                                         sum_bytes,
                                         "ds4_router_weight_sum")) {
        return 0;
    }

    if (!g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> softplus_sqrt_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_softplus_sqrt_pipeline,
                                    "kernel_dsv4_softplus_sqrt_f32_4");
        ok = softplus_sqrt_pipeline &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             softplus_sqrt_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             256,
                                             1,
                                             1,
                                             0.0f,
                                             0.0f);
    } else {
        ok = ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_softplus_pipeline,
                                             logitsbuf,
                                             logits_off,
                                             probsbuf,
                                             probs_off,
                                             256,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f) &&
             ds4_gpu_encode_unary_f32_rows(cb,
                                             g_unary_sqrt_pipeline,
                                             probsbuf,
                                             probs_off,
                                             probsbuf,
                                             probs_off,
                                             256,
                                             n_tokens,
                                             1,
                                             0.0f,
                                             0.0f);
    }
    if (!ok) return 0;

    if (hash_mode) {
        ok = ds4_gpu_encode_get_rows_i32_token_rows(cb,
                                                      hashbuf,
                                                      hash_off,
                                                      tokensbuf,
                                                      tokens_off,
                                                      single_token,
                                                      selectedbuf,
                                                      selected_off,
                                                      hash_rows,
                                                      n_tokens);
    } else {
        ds4_gpu_tensor *score_tensor = probs;
        DS4MetalTensor *selection_view = nil;

        if (has_bias) {
            if (!biasbuf ||
                !ds4_gpu_ensure_scratch_buffer(&g_router_selection_buffer,
                                                 &g_router_selection_bytes,
                                                 probs_bytes,
                                                 "ds4_router_selection")) {
                return 0;
            }

            ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(256, n_tokens, 256);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_add_pipeline,
                                               &add_args,
                                               probsbuf,
                                               probs_off,
                                               biasbuf,
                                               bias_off,
                                               g_router_selection_buffer,
                                               0);
            if (!ok) return 0;

            selection_view = [DS4MetalTensor new];
            selection_view.buffer = g_router_selection_buffer;
            selection_view.offset = 0;
            selection_view.bytes = probs_bytes;
            selection_view.owner = 0;
            score_tensor = (__bridge ds4_gpu_tensor *)selection_view;
        }

        ok = ds4_gpu_indexer_topk_tensor(selected, score_tensor, 256, n_tokens, 6) != 0;
    }
    if (!ok) return 0;

    if (!g_quality_mode && n_tokens == 1) {
        id<MTLComputePipelineState> router_weights_pipeline =
            ds4_gpu_hot_pipeline(g_dsv4_router_weights_one_pipeline,
                                    "kernel_dsv4_router_weights_one");
        if (!router_weights_pipeline) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:router_weights_pipeline];
        [enc setBuffer:probsbuf offset:probs_off atIndex:0];
        [enc setBuffer:selectedbuf offset:selected_off atIndex:1];
        [enc setBuffer:weightsbuf offset:weights_off atIndex:2];
        [enc dispatchThreads:MTLSizeMake(6, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(6, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        return 1;
    }

    ok = ds4_gpu_encode_get_rows_f32_router_weights(cb,
                                                      probsbuf,
                                                      probs_off,
                                                      selectedbuf,
                                                      selected_off,
                                                      weightsbuf,
                                                      weights_off,
                                                      n_tokens) &&
         ds4_gpu_encode_sum_rows_f32(cb,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       6,
                                       n_tokens) &&
         ds4_gpu_encode_unary_f32_rows(cb,
                                         g_unary_clamp_pipeline,
                                         g_router_weight_sum_buffer,
                                         0,
                                         g_router_weight_sum_buffer,
                                         0,
                                         1,
                                         n_tokens,
                                         0,
                                         6.103515625e-5f,
                                         ds4_gpu_positive_infinity());
    if (!ok) return 0;

    ds4_gpu_bin_args div_args = ds4_gpu_make_bin_rowwise_scalar_args(6, n_tokens);
    const float scale = 1.5f;
    ds4_gpu_bin_args scale_args = ds4_gpu_make_bin_rows_args(6, n_tokens, 1);

    ok = ds4_gpu_encode_bin_f32_rows(cb,
                                       g_bin_div_row_pipeline,
                                       &div_args,
                                       weightsbuf,
                                       weights_off,
                                       g_router_weight_sum_buffer,
                                       0,
                                       weightsbuf,
                                       weights_off);
    if (!ok) return 0;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
    [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc setBuffer:weightsbuf offset:weights_off atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)scale_args.ne1,
                                          (NSUInteger)scale_args.ne2,
                                          (NSUInteger)scale_args.ne3)
         threadsPerThreadgroup:MTLSizeMake(ds4_gpu_bin_threads(6, g_bin_mul_scalar_pipeline), 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);

    return 1;
}

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !model_map) return 0;
    if (hash_mode && token >= hash_rows) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf ||
            ds4_gpu_tensor_bytes(logits) < 256u * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < 6u * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < 6u * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < 256u * sizeof(float)) {
            fprintf(stderr, "ds4: Metal router select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = 256u * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        const int32_t token_i32 = (int32_t)token;
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      nil,
                                                      0,
                                                      &token_i32,
                                                      hash_rows,
                                                      1,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0) return 0;
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: Metal router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> logitsbuf = ds4_gpu_tensor_buffer(logits);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> probsbuf = ds4_gpu_tensor_buffer(probs);
        id<MTLBuffer> tokensbuf = ds4_gpu_tensor_buffer(tokens);
        if (!logitsbuf || !selectedbuf || !weightsbuf || !probsbuf || !tokensbuf ||
            ds4_gpu_tensor_bytes(logits) < (uint64_t)n_tokens * 256u * sizeof(float) ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * 6u * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * 6u * sizeof(float) ||
            ds4_gpu_tensor_bytes(probs) < (uint64_t)n_tokens * 256u * sizeof(float) ||
            ds4_gpu_tensor_bytes(tokens) < (uint64_t)n_tokens * sizeof(int32_t)) {
            fprintf(stderr, "ds4: Metal router batch select received undersized buffers\n");
            return 0;
        }

        uint64_t bias_inner = 0;
        uint64_t hash_inner = 0;
        id<MTLBuffer> biasbuf = nil;
        id<MTLBuffer> hashbuf = nil;
        NSUInteger bias_set_offset = 0;
        NSUInteger hash_set_offset = 0;
        if (has_bias && !hash_mode) {
            const uint64_t bias_bytes = 256u * sizeof(float);
            biasbuf = ds4_gpu_wrap_model_range(model_map, model_size, bias_offset, bias_bytes, &bias_inner);
            if (!biasbuf) return 0;
            bias_set_offset = (NSUInteger)bias_inner;
        }
        if (hash_mode) {
            const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
            hashbuf = ds4_gpu_wrap_model_range(model_map, model_size, hash_offset, hash_bytes, &hash_inner);
            if (!hashbuf) return 0;
            hash_set_offset = (NSUInteger)hash_inner;
        }

        const bool had_batch = g_batch_cb != nil;
        if (!had_batch && ds4_gpu_begin_commands() == 0) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        int ok = cb &&
                 ds4_gpu_encode_router_select(cb,
                                                      selected,
                                                      weights,
                                                      probs,
                                                      logitsbuf,
                                                      ds4_gpu_tensor_offset(logits),
                                                      biasbuf,
                                                      bias_set_offset,
                                                      hashbuf,
                                                      hash_set_offset,
                                                      tokensbuf,
                                                      ds4_gpu_tensor_offset(tokens),
                                                      NULL,
                                                      hash_rows,
                                                      n_tokens,
                                                      has_bias && !hash_mode,
                                                      hash_mode);
        if (!had_batch) {
            ok = ds4_gpu_end_commands() != 0 && ok;
        }
        if (!ok) return 0;
    }

    return 1;
}

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float)) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed tensor MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = 256ull * gate_expert_bytes;
        const uint64_t down_tensor_bytes = 256ull * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
        id<MTLBuffer> up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        id<MTLBuffer> down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        if (!gate_buf || !up_buf || !down_buf) return 0;

        const uint32_t n_tokens = 1;
        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, 256,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, 256,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        int ok = 1;
        const bool write_clamped_moe =
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL;
        id<MTLComputePipelineState> pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        const bool fuse_pair_swiglu =
            !g_quality_mode &&
            !write_clamped_moe &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            pair_swiglu_pipeline != nil;
        if (fuse_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                        pair_swiglu_pipeline,
                                                        &gate_args,
                                                        &act_args,
                                                        gate_buf,
                                                        (NSUInteger)gate_inner,
                                                        up_buf,
                                                        (NSUInteger)up_inner,
                                                        xbuf,
                                                        ds4_gpu_tensor_offset(x),
                                                        gatebuf,
                                                        ds4_gpu_tensor_offset(gate),
                                                        upbuf,
                                                        ds4_gpu_tensor_offset(up),
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        selectedbuf,
                                                        ds4_gpu_tensor_offset(selected),
                                                        weightsbuf,
                                                        ds4_gpu_tensor_offset(weights),
                                                        gate_smem,
                                                        2,
                                                        false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            (NSUInteger)gate_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            (NSUInteger)up_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false);
        }
        if (ok && !fuse_pair_swiglu) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    false);
        }

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            n_expert == 6 &&
            n_tokens == 1 &&
            down_sum6_pipeline != nil;
        if (ok && direct_down_sum) {
            ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                 down_sum6_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2);
        } else if (ok) {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                 down_mv_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 down_dst,
                                                 down_dst_off,
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2,
                                                 false);
        }
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed tensor MoE")) return 0;
    }

    return 1;
}

int ds4_gpu_routed_moe_one_banked_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                n_slots,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !gate_bank || !up_bank ||
        !down_bank || !selected || !weights || n_slots == 0 ||
        n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> gate_buf = ds4_gpu_tensor_buffer(gate_bank);
        id<MTLBuffer> up_buf = ds4_gpu_tensor_buffer(up_bank);
        id<MTLBuffer> down_buf = ds4_gpu_tensor_buffer(down_bank);
        const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf ||
            !selectedbuf || !weightsbuf || !gate_buf || !up_buf || !down_buf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_expert * sizeof(int) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_expert * sizeof(float) ||
            ds4_gpu_tensor_bytes(gate_bank) < (uint64_t)n_slots * gate_expert_bytes ||
            ds4_gpu_tensor_bytes(up_bank) < (uint64_t)n_slots * gate_expert_bytes ||
            ds4_gpu_tensor_bytes(down_bank) < (uint64_t)n_slots * down_expert_bytes) {
            fprintf(stderr, "ds4: Metal banked routed tensor MoE received undersized buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal banked routed tensor MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_inner = ds4_gpu_tensor_offset(gate_bank);
        const uint64_t up_inner = ds4_gpu_tensor_offset(up_bank);
        const uint64_t down_inner = ds4_gpu_tensor_offset(down_bank);

        const uint32_t n_tokens = 1;
        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal banked routed MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, n_slots,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, n_slots,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        int ok = 1;
        const bool write_clamped_moe =
            getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL;
        id<MTLComputePipelineState> pair_swiglu_pipeline = nil;
        if (gate_type == DS4_METAL_TENSOR_IQ2_XXS) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
        } else if (gate_type == DS4_METAL_TENSOR_Q4_K) {
            pair_swiglu_pipeline = g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
        }
        const bool fuse_pair_swiglu =
            !g_quality_mode &&
            !write_clamped_moe &&
            getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
            pair_swiglu_pipeline != nil;
        if (fuse_pair_swiglu) {
            ds4_gpu_dsv4_moe_swiglu_weight_args act_args = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            ok = ds4_gpu_encode_mul_mv_id_pair_swiglu(cb,
                                                        pair_swiglu_pipeline,
                                                        &gate_args,
                                                        &act_args,
                                                        gate_buf,
                                                        (NSUInteger)gate_inner,
                                                        up_buf,
                                                        (NSUInteger)up_inner,
                                                        xbuf,
                                                        ds4_gpu_tensor_offset(x),
                                                        gatebuf,
                                                        ds4_gpu_tensor_offset(gate),
                                                        upbuf,
                                                        ds4_gpu_tensor_offset(up),
                                                        midbuf,
                                                        ds4_gpu_tensor_offset(mid),
                                                        selectedbuf,
                                                        ds4_gpu_tensor_offset(selected),
                                                        weightsbuf,
                                                        ds4_gpu_tensor_offset(weights),
                                                        gate_smem,
                                                        2,
                                                        false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            (NSUInteger)gate_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            (NSUInteger)up_inner,
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false);
        }
        if (ok && !fuse_pair_swiglu) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    false);
        }

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            n_expert == 6 &&
            n_tokens == 1 &&
            down_sum6_pipeline != nil;
        if (ok && direct_down_sum) {
            ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                 down_sum6_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 (NSUInteger)down_inner,
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2);
        } else if (ok) {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                             down_mv_pipeline,
                                             &down_args,
                                             down_buf,
                                             (NSUInteger)down_inner,
                                             midbuf,
                                             ds4_gpu_tensor_offset(mid),
                                             down_dst,
                                             down_dst_off,
                                             selectedbuf,
                                             ds4_gpu_tensor_offset(selected),
                                             down_smem,
                                             2,
                                             false);
        }
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                   down_dst,
                                                   down_dst_off,
                                                   outbuf,
                                                   ds4_gpu_tensor_offset(out),
                                                   out_dim,
                                                   n_expert,
                                                   n_tokens);
        }
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "banked routed tensor MoE")) return 0;
    }

    return 1;
}

int ds4_gpu_routed_moe_expert_banked_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !gate_bank || !up_bank ||
        !down_bank || !selected || !weights || n_tokens == 0) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> gate_buf = ds4_gpu_tensor_buffer(gate_bank);
        id<MTLBuffer> up_buf = ds4_gpu_tensor_buffer(up_bank);
        id<MTLBuffer> down_buf = ds4_gpu_tensor_buffer(down_bank);
        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_tokens * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf ||
            !selectedbuf || !weightsbuf || !gate_buf || !up_buf || !down_buf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < (uint64_t)n_tokens * sizeof(int32_t) ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * sizeof(float) ||
            ds4_gpu_tensor_bytes(gate_bank) < gate_expert_bytes ||
            ds4_gpu_tensor_bytes(up_bank) < gate_expert_bytes ||
            ds4_gpu_tensor_bytes(down_bank) < down_expert_bytes) {
            fprintf(stderr, "ds4: Metal Flash-MoE prefill expert batch received undersized buffers\n");
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        id<MTLComputePipelineState> gate_mm_pipeline = nil;
        id<MTLComputePipelineState> down_mm_pipeline = nil;
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal Flash-MoE prefill quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, 1,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, 1, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, 1,
                                          down_row_bytes, down_expert_bytes,
                                          1, 1, n_tokens, down_nr0);
        const bool use_mm_id = n_tokens >= 32u;
        const bool request_mid_f16 =
            !g_quality_mode && getenv("DS4_METAL_MOE_MID_F32") == NULL;
        ds4_gpu_mul_mm_id_map_args gate_map_args = { 0 };
        ds4_gpu_mul_mm_id_args gate_mm_args = { 0 };
        ds4_gpu_mul_mm_id_args down_mm_args = { 0 };
        id<MTLComputePipelineState> map_pipeline = nil;
        if (use_mm_id) {
            gate_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, 1, 1, 1, n_tokens);
            gate_mm_args =
                ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, 1,
                                              gate_row_bytes, gate_expert_bytes,
                                              1, 1, n_tokens);
            down_mm_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, 1,
                                                        down_row_bytes, down_expert_bytes,
                                                        1, 1, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));
            map_pipeline = ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(1));
            gate_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            down_mm_pipeline = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) :
                ds4_gpu_routed_mm_pipeline(down_type);
            if (!map_pipeline || !gate_mm_pipeline || !down_mm_pipeline) return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        int ok = 0;
        if (use_mm_id) {
            ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                                map_pipeline,
                                                &gate_map_args,
                                                &gate_mm_args,
                                                selectedbuf,
                                                ds4_gpu_tensor_offset(selected)) &&
                 ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                   gate_mm_pipeline,
                                                   &gate_mm_args,
                                                   gate_buf,
                                                   ds4_gpu_tensor_offset(gate_bank),
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   gatebuf,
                                                   ds4_gpu_tensor_offset(gate)) &&
                 ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                   gate_mm_pipeline,
                                                   &gate_mm_args,
                                                   up_buf,
                                                   ds4_gpu_tensor_offset(up_bank),
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up));
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_IQ2_XXS &&
                   g_moe_mul_mv_id_iq2_xxs_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_iq2_xxs_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 ds4_gpu_tensor_offset(gate_bank),
                                                 up_buf,
                                                 ds4_gpu_tensor_offset(up_bank),
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else if (!g_quality_mode &&
                   gate_type == DS4_METAL_TENSOR_Q4_K &&
                   g_moe_mul_mv_id_q4_k_pair_pipeline) {
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 g_moe_mul_mv_id_q4_k_pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 ds4_gpu_tensor_offset(gate_bank),
                                                 up_buf,
                                                 ds4_gpu_tensor_offset(up_bank),
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            gate_buf,
                                            ds4_gpu_tensor_offset(gate_bank),
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            gatebuf,
                                            ds4_gpu_tensor_offset(gate),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                            gate_mv_pipeline,
                                            &gate_args,
                                            up_buf,
                                            ds4_gpu_tensor_offset(up_bank),
                                            xbuf,
                                            ds4_gpu_tensor_offset(x),
                                            upbuf,
                                            ds4_gpu_tensor_offset(up),
                                            selectedbuf,
                                            ds4_gpu_tensor_offset(selected),
                                            gate_smem,
                                            2,
                                            false);
        }

        const bool use_mid_f16 =
            use_mm_id && !g_quality_mode && request_mid_f16;
        if (mid_is_f16) *mid_is_f16 = use_mid_f16;
        if (ok) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    n_tokens,
                                                    clamp,
                                                    use_mid_f16);
        }
        if (ok) {
            if (use_mm_id) {
                ok = ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                       down_mm_pipeline,
                                                       &down_mm_args,
                                                       down_buf,
                                                       ds4_gpu_tensor_offset(down_bank),
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out));
            } else {
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                 down_mv_pipeline,
                                                 &down_args,
                                                 down_buf,
                                                 ds4_gpu_tensor_offset(down_bank),
                                                 midbuf,
                                                 ds4_gpu_tensor_offset(mid),
                                                 outbuf,
                                                 ds4_gpu_tensor_offset(out),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 down_smem,
                                                 2,
                                                 false);
            }
        }
        if (!ok) return 0;
        return ds4_gpu_finish_command_buffer(cb, owned, "Flash-MoE prefill expert batch");
    }
}

static int ds4_gpu_ensure_mpp_int8_prefill_pipelines(void);
static int ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols,
        float                qscale);
static int ds4_gpu_encode_mpp_dequant_q2_k_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols,
        float                qscale);
/* Fused gate+up+down dequant.  All three weights' dispatch totals must match
 * (verified up front for the DSv4 shape).  Returns 0 if the fused pipeline
 * isn't available — caller should fall back to 3 separate dispatches. */
static int ds4_gpu_encode_mpp_dequant_gud_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate_src, NSUInteger gate_src_off, id<MTLBuffer> gate_dst,
        id<MTLBuffer>        up_src,   NSUInteger up_src_off,   id<MTLBuffer> up_dst,
        id<MTLBuffer>        down_src, NSUInteger down_src_off, id<MTLBuffer> down_dst,
        uint32_t             expert_mid_dim,
        uint32_t             expert_in_dim,
        uint32_t             out_dim,
        float                qscale);
static int ds4_gpu_encode_mpp_quant_f32_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             n,
        float                qscale);
static int ds4_gpu_encode_ane_dequant_iq2_xxs_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols);
static int ds4_gpu_encode_ane_dequant_q2_k_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols);
static int ds4_gpu_encode_ane_output_pack_f16_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows);

static void ds4_gpu_ane_prefill_env(float *w_qscale,
                                    float *w_scale,
                                    float *x_qscale,
                                    float *x_scale,
                                    float *mid_qscale,
                                    float *mid_scale,
                                    uint32_t *batch,
                                    uint32_t *compile_limit) {
    float aq = 64.0f;
    const char *qscale_env = getenv("DS4_FLASH_MOE_ANE_INT8_QSCALE");
    if (!qscale_env || !qscale_env[0]) qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_QSCALE");
    if (qscale_env && qscale_env[0]) {
        char *end = NULL;
        float parsed = strtof(qscale_env, &end);
        if (end != qscale_env && parsed > 0.0f && isfinite(parsed)) aq = parsed;
    }

    float xq = 16.0f;
    const char *x_qscale_env = getenv("DS4_FLASH_MOE_ANE_X_QSCALE");
    if (!x_qscale_env || !x_qscale_env[0]) x_qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_X_QSCALE");
    if (x_qscale_env && x_qscale_env[0]) {
        char *end = NULL;
        float parsed = strtof(x_qscale_env, &end);
        if (end != x_qscale_env && parsed > 0.0f && isfinite(parsed)) xq = parsed;
    }

    float midq = 16.0f;
    const char *mid_qscale_env = getenv("DS4_FLASH_MOE_ANE_MID_QSCALE");
    if (!mid_qscale_env || !mid_qscale_env[0]) mid_qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_MID_QSCALE");
    if (mid_qscale_env && mid_qscale_env[0]) {
        char *end = NULL;
        float parsed = strtof(mid_qscale_env, &end);
        if (end != mid_qscale_env && parsed > 0.0f && isfinite(parsed)) midq = parsed;
    }

    uint32_t b = 256u;
    const char *batch_env = getenv("DS4_FLASH_MOE_ANE_BATCH");
    if (batch_env && batch_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(batch_env, &end, 10);
        if (end != batch_env && parsed > 0 && parsed <= 4096ul) b = (uint32_t)parsed;
    }

    uint32_t limit = 8u;
    const char *compile_limit_env = getenv("DS4_FLASH_MOE_ANE_COMPILE_LIMIT");
    if (compile_limit_env && compile_limit_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(compile_limit_env, &end, 10);
        if (end != compile_limit_env && parsed <= 128ul) limit = (uint32_t)parsed;
    }

    if (w_qscale) *w_qscale = aq;
    if (w_scale) *w_scale = 1.0f / aq;
    if (x_qscale) *x_qscale = xq;
    if (x_scale) *x_scale = 1.0f / xq;
    if (mid_qscale) *mid_qscale = midq;
    if (mid_scale) *mid_scale = 1.0f / midq;
    if (batch) *batch = b;
    if (compile_limit) *compile_limit = limit;
}

static const char *ds4_gpu_ane_mode_name(int mode);

/* Acquire a free dequant slot; grow its buffers if needed.  Blocks (cv) when
 * all slots are in use, which serves as backpressure if the scheduler tries
 * to start more concurrent ANE jobs than the pool can hold.  Returns -1 on
 * fatal allocation failure. */
static int ds4_gpu_ane_acquire_dequant_slot(NSUInteger gate_i8_bytes,
                                             NSUInteger up_i8_bytes,
                                             NSUInteger down_i8_bytes,
                                             NSUInteger x_i8_bytes) {
    pthread_mutex_lock(&g_ane_dequant_slot_mu);
    int idx = -1;
    while (idx < 0) {
        for (int i = 0; i < DS4_ANE_DEQUANT_SLOTS; i++) {
            if (!g_ane_dequant_slots[i].in_use) {
                g_ane_dequant_slots[i].in_use = 1;
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            pthread_cond_wait(&g_ane_dequant_slot_cv, &g_ane_dequant_slot_mu);
        }
    }
    pthread_mutex_unlock(&g_ane_dequant_slot_mu);

    ds4_ane_dequant_slot_t *slot = &g_ane_dequant_slots[idx];
    if (!ds4_gpu_ensure_scratch_buffer(&slot->gate_i8, &slot->gate_i8_bytes,
                                       gate_i8_bytes, "ds4_ane_dequant_slot_gate_i8") ||
        !ds4_gpu_ensure_scratch_buffer(&slot->up_i8, &slot->up_i8_bytes,
                                       up_i8_bytes, "ds4_ane_dequant_slot_up_i8") ||
        !ds4_gpu_ensure_scratch_buffer(&slot->down_i8, &slot->down_i8_bytes,
                                       down_i8_bytes, "ds4_ane_dequant_slot_down_i8") ||
        !ds4_gpu_ensure_scratch_buffer(&slot->x_i8, &slot->x_i8_bytes,
                                       x_i8_bytes, "ds4_ane_dequant_slot_x_i8")) {
        pthread_mutex_lock(&g_ane_dequant_slot_mu);
        slot->in_use = 0;
        pthread_cond_broadcast(&g_ane_dequant_slot_cv);
        pthread_mutex_unlock(&g_ane_dequant_slot_mu);
        return -1;
    }
    return idx;
}

static void ds4_gpu_ane_release_dequant_slot(int idx) {
    if (idx < 0 || idx >= DS4_ANE_DEQUANT_SLOTS) return;
    pthread_mutex_lock(&g_ane_dequant_slot_mu);
    g_ane_dequant_slots[idx].in_use = 0;
    pthread_cond_broadcast(&g_ane_dequant_slot_cv);
    pthread_mutex_unlock(&g_ane_dequant_slot_mu);
}

static void ds4_gpu_ane_ctx_cache_clear(void) {
    for (uint32_t i = 0; i < DS4_ANE_CTX_CACHE_MAX; i++) {
        ds4_ane_mlp_int8w_destroy(g_ane_ctx_cache[i].ctx);
        memset(&g_ane_ctx_cache[i], 0, sizeof(g_ane_ctx_cache[i]));
        ds4_ane_mlp_int8w_destroy(g_ane_ctx_cache_b[i].ctx);
        memset(&g_ane_ctx_cache_b[i], 0, sizeof(g_ane_ctx_cache_b[i]));
        ds4_ane_mlp_int8w_destroy(g_ane_ctx_cache_c[i].ctx);
        memset(&g_ane_ctx_cache_c[i], 0, sizeof(g_ane_ctx_cache_c[i]));
        ds4_ane_mlp_int8w_destroy(g_ane_ctx_cache_d[i].ctx);
        memset(&g_ane_ctx_cache_d[i], 0, sizeof(g_ane_ctx_cache_d[i]));
    }
}

static uint32_t ds4_gpu_ane_parse_batch_list(uint32_t fallback,
                                             uint32_t *batches,
                                             uint32_t max_batches) {
    if (!batches || max_batches == 0) return 0;
    uint32_t count = 0;
    const char *env = getenv("DS4_FLASH_MOE_ANE_BATCHES");
    if (env && env[0]) {
        const char *p = env;
        while (*p && count < max_batches) {
            while (*p == ',' || *p == ':' || *p == ';' || isspace((unsigned char)*p)) p++;
            if (!*p) break;
            char *end = NULL;
            unsigned long parsed = strtoul(p, &end, 10);
            if (end == p) break;
            if (parsed > 0 && parsed <= 4096ul) {
                uint32_t b = (uint32_t)parsed;
                int dup = 0;
                for (uint32_t i = 0; i < count; i++) {
                    if (batches[i] == b) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup) batches[count++] = b;
            }
            p = end;
        }
    }
    if (count == 0) batches[count++] = fallback ? fallback : 256u;
    for (uint32_t i = 1; i < count; i++) {
        uint32_t v = batches[i];
        uint32_t j = i;
        while (j > 0 && batches[j - 1] > v) {
            batches[j] = batches[j - 1];
            j--;
        }
        batches[j] = v;
    }
    return count;
}

static uint32_t ds4_gpu_ane_select_batch(uint32_t refs, uint32_t fallback) {
    uint32_t batches[DS4_ANE_CTX_CACHE_MAX];
    uint32_t count = ds4_gpu_ane_parse_batch_list(fallback, batches, DS4_ANE_CTX_CACHE_MAX);
    if (count == 0) return fallback ? fallback : 256u;
    for (uint32_t i = 0; i < count; i++) {
        if (refs <= batches[i]) return batches[i];
    }
    return batches[count - 1];
}

static ds4_ane_mlp_int8w_ctx *ds4_gpu_ane_get_i8i8_tiled_ctx(uint32_t H,
                                                             uint32_t I,
                                                             uint32_t B,
                                                             float w_scale,
                                                             float x_scale,
                                                             float mid_scale,
                                                             uint32_t compile_limit) {
    const int mode = 6;
    for (uint32_t i = 0; i < DS4_ANE_CTX_CACHE_MAX; i++) {
        ds4_ane_ctx_cache_entry *entry = &g_ane_ctx_cache[i];
        if (entry->ctx &&
            entry->mode == mode &&
            entry->H == (int)H &&
            entry->I == (int)I &&
            entry->B == (int)B &&
            fabsf(entry->w_scale - w_scale) <= 1.0e-8f &&
            fabsf(entry->x_scale - x_scale) <= 1.0e-8f &&
            fabsf(entry->mid_scale - mid_scale) <= 1.0e-8f) {
            return entry->ctx;
        }
    }
    if (g_ane_prefill_compile_attempts >= compile_limit) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    uint32_t slot = DS4_ANE_CTX_CACHE_MAX;
    for (uint32_t i = 0; i < DS4_ANE_CTX_CACHE_MAX; i++) {
        if (!g_ane_ctx_cache[i].ctx) {
            slot = i;
            break;
        }
    }
    if (slot == DS4_ANE_CTX_CACHE_MAX) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    g_ane_prefill_compile_attempts++;
    ds4_ane_mlp_int8w_ctx *ctx =
        ds4_ane_mlp_i8w_i8x_tiled_fused_create((int)H,
                                                (int)I,
                                                (int)B,
                                                w_scale,
                                                x_scale,
                                                mid_scale);
    if (!ctx) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    g_ane_ctx_cache[slot].ctx = ctx;
    g_ane_ctx_cache[slot].mode = mode;
    g_ane_ctx_cache[slot].H = (int)H;
    g_ane_ctx_cache[slot].I = (int)I;
    g_ane_ctx_cache[slot].B = (int)B;
    g_ane_ctx_cache[slot].w_scale = w_scale;
    g_ane_ctx_cache[slot].x_scale = x_scale;
    g_ane_ctx_cache[slot].mid_scale = mid_scale;
    fprintf(stderr,
            "ds4: ANE %s cached async prefill B=%u H=%u I=%u w_scale=%.6g x_scale=%.6g mid_scale=%.6g\n",
            ds4_gpu_ane_mode_name(mode), B, H, I, w_scale, x_scale, mid_scale);
    return ctx;
}

/* Shared implementation for the secondary-cluster context getters.  The cache
 * argument selects which parallel context slot table to use (one per worker).
 * Like ds4_gpu_ane_get_i8i8_tiled_ctx but allocates against the given cache so
 * the caller gets a context handle distinct from the primary, for use on a
 * different ANE cluster.  Same shape + scales; same fallback budget tracking. */
static ds4_ane_mlp_int8w_ctx *ds4_gpu_ane_get_i8i8_tiled_ctx_from(
        ds4_ane_ctx_cache_entry *cache,
        uint32_t H,
        uint32_t I,
        uint32_t B,
        float w_scale,
        float x_scale,
        float mid_scale,
        uint32_t compile_limit) {
    const int mode = 6;
    for (uint32_t i = 0; i < DS4_ANE_CTX_CACHE_MAX; i++) {
        ds4_ane_ctx_cache_entry *entry = &cache[i];
        if (entry->ctx &&
            entry->mode == mode &&
            entry->H == (int)H &&
            entry->I == (int)I &&
            entry->B == (int)B &&
            fabsf(entry->w_scale - w_scale) <= 1.0e-8f &&
            fabsf(entry->x_scale - x_scale) <= 1.0e-8f &&
            fabsf(entry->mid_scale - mid_scale) <= 1.0e-8f) {
            return entry->ctx;
        }
    }
    if (g_ane_prefill_compile_attempts >= compile_limit) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    uint32_t slot = DS4_ANE_CTX_CACHE_MAX;
    for (uint32_t i = 0; i < DS4_ANE_CTX_CACHE_MAX; i++) {
        if (!cache[i].ctx) {
            slot = i;
            break;
        }
    }
    if (slot == DS4_ANE_CTX_CACHE_MAX) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    g_ane_prefill_compile_attempts++;
    ds4_ane_mlp_int8w_ctx *ctx_new =
        ds4_ane_mlp_i8w_i8x_tiled_fused_create((int)H, (int)I, (int)B,
                                                w_scale, x_scale, mid_scale);
    if (!ctx_new) {
        g_ane_prefill_compile_failures++;
        return NULL;
    }
    cache[slot].ctx = ctx_new;
    cache[slot].mode = mode;
    cache[slot].H = (int)H;
    cache[slot].I = (int)I;
    cache[slot].B = (int)B;
    cache[slot].w_scale = w_scale;
    cache[slot].x_scale = x_scale;
    cache[slot].mid_scale = mid_scale;
    return ctx_new;
}

static ds4_ane_mlp_int8w_ctx *ds4_gpu_ane_get_i8i8_tiled_ctx_b(uint32_t H,
                                                                uint32_t I,
                                                                uint32_t B,
                                                                float w_scale,
                                                                float x_scale,
                                                                float mid_scale,
                                                                uint32_t compile_limit) {
    return ds4_gpu_ane_get_i8i8_tiled_ctx_from(g_ane_ctx_cache_b, H, I, B,
                                                w_scale, x_scale, mid_scale,
                                                compile_limit);
}

static ds4_ane_mlp_int8w_ctx *ds4_gpu_ane_get_i8i8_tiled_ctx_c(uint32_t H,
                                                                uint32_t I,
                                                                uint32_t B,
                                                                float w_scale,
                                                                float x_scale,
                                                                float mid_scale,
                                                                uint32_t compile_limit) {
    return ds4_gpu_ane_get_i8i8_tiled_ctx_from(g_ane_ctx_cache_c, H, I, B,
                                                w_scale, x_scale, mid_scale,
                                                compile_limit);
}

static ds4_ane_mlp_int8w_ctx *ds4_gpu_ane_get_i8i8_tiled_ctx_d(uint32_t H,
                                                                uint32_t I,
                                                                uint32_t B,
                                                                float w_scale,
                                                                float x_scale,
                                                                float mid_scale,
                                                                uint32_t compile_limit) {
    return ds4_gpu_ane_get_i8i8_tiled_ctx_from(g_ane_ctx_cache_d, H, I, B,
                                                w_scale, x_scale, mid_scale,
                                                compile_limit);
}


static int ds4_gpu_ane_prefill_fp16w_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_FP16W");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_prefill_fp16x_i8w_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_FP16X_INT8W");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_prefill_i8i8_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_I8I8_PREFILL");
    if (!env || !env[0]) env = getenv("DS4_FLASH_MOE_MPP_I8I8_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_prefill_i8i8_fused_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_I8I8_FUSED_PREFILL");
    if (!env || !env[0]) env = getenv("DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_prefill_i8i8_full_fused_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_I8I8_FULL_FUSED_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

static int ds4_gpu_ane_prefill_i8i8_tiled_fused_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL");
    if (!env || !env[0]) env = getenv("DS4_FLASH_MOE_MPP_I8I8_TILED_FUSED_PREFILL");
    return env && env[0] && atoi(env) != 0;
}

/* Dual-cluster ANE prefill: when 1, spawn a second ANE worker against a
 * separate i8i8 tiled_fused context so the M3 Ultra's two ANEx16 clusters
 * run in parallel.  Smoke at the production expert shape (H=4096, I=2048,
 * B=256) measured 1.74x aggregate without stagger and 1.86x with a half-iter
 * stagger; B=128 hits 1.97-2.00x.  Only honoured on the tiled_fused path. */
static int ds4_gpu_ane_prefill_dual_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_DUAL");
    return env && env[0] && atoi(env) != 0;
}
static int ds4_gpu_ane_prefill_dual_stagger_us(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_DUAL_STAGGER_US");
    if (!env || !env[0]) return 0;
    long v = atol(env);
    if (v < 0) return 0;
    if (v > 100000) v = 100000;
    return (int)v;
}

/* Per-call ANE worker count.  M3 Ultra physically has two ANEx16 clusters
 * (A,B); a 3rd/4th worker thread will be time-shared by the OS scheduler onto
 * one of them.  The experiment goal is to see whether oversubscription lets
 * post-eval scratch writeback overlap with the next call's eval more
 * aggressively (or hurts via per-thread bandwidth contention).
 *
 *   DS4_FLASH_MOE_ANE_THREADS=1 .. 4   explicit worker count
 *   (unset)                            fall back to: DUAL=1 → 2, else 1
 */
static int ds4_gpu_ane_prefill_threads(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_THREADS");
    if (env && env[0]) {
        long v = atol(env);
        if (v < 1) v = 1;
        if (v > 4) v = 4;
        return (int)v;
    }
    return ds4_gpu_ane_prefill_dual_enabled() ? 2 : 1;
}

static const char *ds4_gpu_ane_mode_name(int mode) {
    switch (mode) {
        case 1: return "fp16w";
        case 2: return "i8w-fp16x";
        case 3: return "i8w-i8x";
        case 4: return "i8w-i8x-full-fused";
        case 5: return "i8w-i8x-gateup-fused";
        case 6: return "i8w-i8x-tiled-fused";
        default: return "int8w";
    }
}

static uint32_t ds4_gpu_ane_mode_evaluate_calls(int mode) {
    switch (mode) {
        case 1:
        case 2:
        case 3:
            return 3u;
        case 5:
            return 2u;
        default:
            return 1u;
    }
}

static uint32_t ds4_gpu_ane_compare_limit(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_COMPARE");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long parsed = strtoul(env, &end, 10);
    if (end != env && parsed > 0 && parsed <= 1000ul) return (uint32_t)parsed;
    return 1;
}

static uint32_t g_ane_prefill_compare_count = 0;

static inline float ds4_gpu_round_f32_to_f16(float v) {
    return ds4_gpu_f16_bits_to_f32(ds4_gpu_f32_to_f16_bits(v));
}

static void ds4_gpu_ane_compare_one_row(const char *mode,
                                        const uint16_t *wg_f16,
                                        const uint16_t *wu_f16,
                                        const uint16_t *wd_f16,
                                        const int8_t *wg_i8,
                                        const int8_t *wu_i8,
                                        const int8_t *wd_i8,
                                        const uint16_t *x_f16,
                                        const int8_t *x_i8,
                                        const uint16_t *ane_out_f16,
                                        uint32_t H,
                                        uint32_t I,
                                        uint64_t call_idx,
                                        uint32_t refs,
                                        uint32_t chunk_begin,
                                        uint32_t chunk,
                                        float w_scale,
                                        float x_scale,
                                        float mid_scale,
                                        int quant_hidden) {
    if (!ane_out_f16 || H == 0 || I == 0) return;
    float *gate = (float *)malloc((size_t)I * sizeof(float));
    float *up = (float *)malloc((size_t)I * sizeof(float));
    float *hidden = (float *)malloc((size_t)I * sizeof(float));
    float *hidden_f16acc = (float *)malloc((size_t)I * sizeof(float));
    if (!gate || !up || !hidden || !hidden_f16acc) {
        free(gate);
        free(up);
        free(hidden);
        free(hidden_f16acc);
        return;
    }
    double gate_sum = 0.0;
    double up_sum = 0.0;
    float gate_abs_max = 0.0f;
    float up_abs_max = 0.0f;
    for (uint32_t j = 0; j < I; j++) {
        double gs = 0.0;
        double us = 0.0;
        float gs_h = 0.0f;
        float us_h = 0.0f;
        for (uint32_t h = 0; h < H; h++) {
            float xv = x_f16 ?
                ds4_gpu_f16_bits_to_f32(x_f16[h]) :
                ((float)x_i8[h] * x_scale);
            const uint64_t wi = (uint64_t)h * I + j;
            const float gw = wg_f16 ?
                ds4_gpu_f16_bits_to_f32(wg_f16[wi]) :
                ((float)wg_i8[wi] * w_scale);
            const float uw = wu_f16 ?
                ds4_gpu_f16_bits_to_f32(wu_f16[wi]) :
                ((float)wu_i8[wi] * w_scale);
            gs += (double)xv * (double)gw;
            us += (double)xv * (double)uw;
            gs_h = ds4_gpu_round_f32_to_f16(gs_h + ds4_gpu_round_f32_to_f16(xv * gw));
            us_h = ds4_gpu_round_f32_to_f16(us_h + ds4_gpu_round_f32_to_f16(xv * uw));
        }
        float g = (float)gs;
        float u = (float)us;
        float gh = gs_h;
        float uh = us_h;
        if (g < -10.0f) g = -10.0f;
        if (g > 10.0f) g = 10.0f;
        if (u < -10.0f) u = -10.0f;
        if (u > 10.0f) u = 10.0f;
        if (gh < -10.0f) gh = -10.0f;
        if (gh > 10.0f) gh = 10.0f;
        if (uh < -10.0f) uh = -10.0f;
        if (uh > 10.0f) uh = 10.0f;
        const float act = g / (1.0f + expf(-g));
        const float act_h = ds4_gpu_round_f32_to_f16(gh / (1.0f + expf(-gh)));
        gate[j] = g;
        up[j] = u;
        hidden[j] = act * u;
        hidden_f16acc[j] = ds4_gpu_round_f32_to_f16(act_h * uh);
        if (quant_hidden && mid_scale > 0.0f) {
            const float mid_qscale = 1.0f / mid_scale;
            float q = nearbyintf(hidden[j] * mid_qscale);
            if (q < -128.0f) q = -128.0f;
            if (q > 127.0f) q = 127.0f;
            hidden[j] = q * mid_scale;
            float qh = nearbyintf(hidden_f16acc[j] * mid_qscale);
            if (qh < -128.0f) qh = -128.0f;
            if (qh > 127.0f) qh = 127.0f;
            hidden_f16acc[j] = qh * mid_scale;
        }
        gate_sum += g;
        up_sum += u;
        const float ga = fabsf(g);
        const float ua = fabsf(u);
        if (ga > gate_abs_max) gate_abs_max = ga;
        if (ua > up_abs_max) up_abs_max = ua;
    }

    double sq = 0.0;
    double ref_sq = 0.0;
    double sq_h = 0.0;
    double ref_h_sq = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_abs_h = 0.0f;
    float max_rel_h = 0.0f;
    uint32_t worst = 0;
    uint32_t worst_h = 0;
    float worst_ref = 0.0f;
    float worst_ane = 0.0f;
    float worst_ref_h = 0.0f;
    float worst_ane_h = 0.0f;
    for (uint32_t c = 0; c < H; c++) {
        double ys = 0.0;
        float ys_h = 0.0f;
        for (uint32_t j = 0; j < I; j++) {
            const uint64_t wi = (uint64_t)j * H + c;
            const float dw = wd_f16 ?
                ds4_gpu_f16_bits_to_f32(wd_f16[wi]) :
                ((float)wd_i8[wi] * w_scale);
            ys += (double)hidden[j] * (double)dw;
            ys_h = ds4_gpu_round_f32_to_f16(ys_h + ds4_gpu_round_f32_to_f16(hidden_f16acc[j] * dw));
        }
        const float ref = (float)ys;
        const float ref_h = ys_h;
        const float ane = ds4_gpu_f16_bits_to_f32(ane_out_f16[c]);
        const float err = fabsf(ane - ref);
        const float rel = err / fmaxf(fabsf(ref), 1.0e-6f);
        const float err_h = fabsf(ane - ref_h);
        const float rel_h = err_h / fmaxf(fabsf(ref_h), 1.0e-6f);
        sq += (double)err * (double)err;
        ref_sq += (double)ref * (double)ref;
        sq_h += (double)err_h * (double)err_h;
        ref_h_sq += (double)ref_h * (double)ref_h;
        if (err > max_abs) {
            max_abs = err;
            max_rel = rel;
            worst = c;
            worst_ref = ref;
            worst_ane = ane;
        }
        if (err_h > max_abs_h) {
            max_abs_h = err_h;
            max_rel_h = rel_h;
            worst_h = c;
            worst_ref_h = ref_h;
            worst_ane_h = ane;
        }
    }
    fprintf(stderr,
            "ds4: ANE compare %s call=%llu refs=%u chunk_begin=%u chunk=%u row0 H=%u I=%u max_abs=%.8g max_rel=%.8g rms=%.8g rel_rms=%.8g worst_c=%u ref=%.8g ane=%.8g gate_mean=%.8g up_mean=%.8g gate_abs_max=%.8g up_abs_max=%.8g\n",
            mode ? mode : "unknown",
            (unsigned long long)call_idx,
            refs,
            chunk_begin,
            chunk,
            H,
            I,
            max_abs,
            max_rel,
            sqrt(sq / (double)H),
            sqrt(sq / fmax(ref_sq, 1.0e-30)),
            worst,
            worst_ref,
            worst_ane,
            (float)(gate_sum / (double)I),
            (float)(up_sum / (double)I),
            gate_abs_max,
            up_abs_max);
    fprintf(stderr,
            "ds4: ANE compare %s fp16acc row0 max_abs=%.8g max_rel=%.8g rms=%.8g rel_rms=%.8g worst_c=%u ref=%.8g ane=%.8g\n",
            mode ? mode : "unknown",
            max_abs_h,
            max_rel_h,
            sqrt(sq_h / (double)H),
            sqrt(sq_h / fmax(ref_h_sq, 1.0e-30)),
            worst_h,
            worst_ref_h,
            worst_ane_h);
    fprintf(stderr, "ds4: ANE compare samples");
    for (uint32_t c = 0; c < 8 && c < H; c++) {
        double ys = 0.0;
        for (uint32_t j = 0; j < I; j++) {
            const uint64_t wi = (uint64_t)j * H + c;
            const float dw = wd_f16 ?
                ds4_gpu_f16_bits_to_f32(wd_f16[wi]) :
                ((float)wd_i8[wi] * w_scale);
            ys += (double)hidden[j] * (double)dw;
        }
        fprintf(stderr, " c%u ref=%.5g ane=%.5g",
                c, (float)ys, ds4_gpu_f16_bits_to_f32(ane_out_f16[c]));
    }
    fprintf(stderr, "\n");
    free(gate);
    free(up);
    free(hidden);
    free(hidden_f16acc);
}

int ds4_gpu_ane_prefill_precompile_from_env(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    float w_qscale = 64.0f, w_scale = 1.0f / 64.0f;
    float x_qscale = 16.0f, x_scale = 1.0f / 16.0f;
    float mid_qscale = 16.0f, mid_scale = 1.0f / 16.0f;
    uint32_t batch = 256u, compile_limit = 8u;
    ds4_gpu_ane_prefill_env(&w_qscale, &w_scale, &x_qscale, &x_scale, &mid_qscale, &mid_scale, &batch, &compile_limit);

    const int fp16w = ds4_gpu_ane_prefill_fp16w_enabled();
    const int fp16x_i8w = !fp16w && ds4_gpu_ane_prefill_fp16x_i8w_enabled();
    const int i8i8 = !fp16w && !fp16x_i8w && ds4_gpu_ane_prefill_i8i8_enabled();
    const int i8i8_fused = i8i8 && ds4_gpu_ane_prefill_i8i8_fused_enabled();
    const int i8i8_full_fused = i8i8 && ds4_gpu_ane_prefill_i8i8_full_fused_enabled();
    const int i8i8_tiled_fused = i8i8 && ds4_gpu_ane_prefill_i8i8_tiled_fused_enabled();
    const int target_mode = fp16w ? 1 : (fp16x_i8w ? 2 : (i8i8 ? (i8i8_tiled_fused ? 6 : (i8i8_full_fused ? 4 : (i8i8_fused ? 5 : 3))) : 0));
    if (i8i8_tiled_fused) {
        uint32_t batches[DS4_ANE_CTX_CACHE_MAX];
        uint32_t count = ds4_gpu_ane_parse_batch_list(batch, batches, DS4_ANE_CTX_CACHE_MAX);
        int ok = count > 0;
        for (uint32_t i = 0; ok && i < count; i++) {
            ok = ds4_gpu_ane_get_i8i8_tiled_ctx(4096,
                                                2048,
                                                batches[i],
                                                w_scale,
                                                x_scale,
                                                mid_scale,
                                                compile_limit) != NULL;
        }
        return ok;
    }
    if (g_ane_int8w_ctx &&
        ds4_ane_mlp_int8w_mode(g_ane_int8w_ctx) == target_mode &&
        ds4_ane_mlp_int8w_H(g_ane_int8w_ctx) == 4096 &&
        ds4_ane_mlp_int8w_I(g_ane_int8w_ctx) == 2048 &&
        ds4_ane_mlp_int8w_B(g_ane_int8w_ctx) == (int)batch &&
        (fp16w ||
         (fp16x_i8w &&
          fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - w_scale) <= 1.0e-8f) ||
         (i8i8 &&
          fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - w_scale) <= 1.0e-8f &&
          fabsf(ds4_ane_mlp_int8w_x_scale(g_ane_int8w_ctx) - x_scale) <= 1.0e-8f &&
          fabsf(ds4_ane_mlp_int8w_mid_scale(g_ane_int8w_ctx) - mid_scale) <= 1.0e-8f) ||
         (!i8i8 && !fp16x_i8w &&
          fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - w_scale) <= 1.0e-8f &&
          fabsf(ds4_ane_mlp_int8w_x_scale(g_ane_int8w_ctx) - x_scale) <= 1.0e-8f))) {
        return 1;
    }

    if (g_ane_prefill_compile_attempts >= compile_limit) {
        g_ane_prefill_compile_failures++;
        fprintf(stderr,
                "ds4: ANE int8w precompile budget exhausted attempts=%llu limit=%u; using GPU fallback\n",
                (unsigned long long)g_ane_prefill_compile_attempts,
                compile_limit);
        return 0;
    }

    ds4_ane_mlp_int8w_destroy(g_ane_int8w_ctx);
    g_ane_int8w_ctx = NULL;
    g_ane_prefill_compile_attempts++;
    g_ane_int8w_ctx = fp16w ?
        ds4_ane_mlp_fp16w_create(4096, 2048, (int)batch) :
        (fp16x_i8w ?
         ds4_ane_mlp_i8w_fp16x_create(4096, 2048, (int)batch, w_scale) :
         (i8i8 ?
          (i8i8_tiled_fused ?
           ds4_ane_mlp_i8w_i8x_tiled_fused_create(4096, 2048, (int)batch, w_scale, x_scale, mid_scale) :
           (i8i8_full_fused ?
           ds4_ane_mlp_i8w_i8x_fused_create(4096, 2048, (int)batch, w_scale, x_scale, mid_scale) :
           (i8i8_fused ?
            ds4_ane_mlp_i8w_i8x_gateup_fused_create(4096, 2048, (int)batch, w_scale, x_scale, mid_scale) :
            ds4_ane_mlp_i8w_i8x_create(4096, 2048, (int)batch, w_scale, x_scale, mid_scale)))) :
          ds4_ane_mlp_int8w_create(4096, 2048, (int)batch, w_scale, x_scale)));
    if (!g_ane_int8w_ctx) {
        g_ane_prefill_compile_failures++;
        fprintf(stderr,
                "ds4: ANE int8w precompile failed B=%u H=4096 I=2048 w_qscale=%.6g x_qscale=%.6g\n",
                batch, w_qscale, x_qscale);
        return 0;
    }
    fprintf(stderr,
            "ds4: ANE %s precompiled B=%u H=4096 I=2048 w_qscale=%.6g x_qscale=%.6g mid_qscale=%.6g\n",
            ds4_gpu_ane_mode_name(target_mode),
            batch, w_qscale, x_qscale, mid_qscale);
    return 1;
}

struct ds4_gpu_ane_prefill_job {
    pthread_t ane_thread;
    pthread_t post_thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int sync_initialized;
    int ane_thread_started;
    int post_thread_started;
    int ane_done;
    int post_done;
    int ane_failed;
    int post_failed;
    int dequant_done_flag;
    double dequant_submit_t0;
    double dequant_wait_ms;
    uint32_t n_tokens;
    uint32_t expert_in_dim;
    uint32_t expert_mid_dim;
    uint32_t out_dim;
    uint32_t ane_batch;
    uint32_t ane_chunk_refs;
    uint32_t chunk_count;
    uint32_t chunks_ready;
    uint32_t chunks_posted;
    int scalar_output_pack;
    int gpu_output_pack;
    float ane_w_scale;
    float ane_x_scale;
    float ane_mid_scale;
    ds4_ane_mlp_int8w_ctx *ane_ctx;
    const int8_t *gate_i8;
    const int8_t *up_i8;
    const int8_t *down_i8;
    const int8_t *x_i8_all;
    int8_t *x_i8_batch;
    uint16_t *out_f16_batch;
    uint16_t *out_f16_all;
    float *out_f32;
    __strong id<MTLBuffer> out_f16_all_mtl;
    /* MTLBuffer backing out_f32 so the eventual writeback into flash_prefill_out
     * can be a GPU blit serialized on the queue (rather than a CPU memcpy that
     * races with the previous scatter_add still in flight). */
    __strong id<MTLBuffer> out_f32_mtl;
    float *route_weights;
    __strong id<MTLBuffer> route_weights_mtl;
    uint64_t out_elems64;
    uint64_t out_bucket_elems64;
    double x_zero_ms;
    double x_copy_ms;
    double out_zero_ms;
    double eval_ms;
    double output_ms;
    double output_f16_copy_ms;
    double output_convert_ms;
    double post_wait_ms;
    uint64_t eval_calls;
    uint64_t eval_refs;
    uint64_t padded_refs;
    uint64_t output_elems;
    uint64_t output_chunks;
    uint64_t chunks_le16;
    uint64_t chunks_le64;
    uint64_t chunks_le128;
    uint64_t chunks_full;

    /* Dequant scratch slot index in g_ane_dequant_slots.  -1 if the job is
     * using the legacy global scratch (e.g. fallback path).  Released when
     * the last ANE worker exits eval (slot data no longer being read). */
    int dequant_slot;

    /* Dual-cluster ANE prefill state.  When dual!=0 we spawn two ANE workers
     * (cluster A: even chunks, cluster B: odd) instead of the single worker.
     * Each cluster has its own context and its own per-chunk scratch (x/out)
     * to avoid memcpy contention on the existing buffers.
     *
     * chunks_done_bitmap is a per-chunk completion flag.  Each worker, after
     * its chunk completes, takes the mutex, sets its bit, and advances
     * chunks_ready as far as the bitmap is contiguous from 0.  This lets the
     * post-thread keep its in-order processing semantics even though chunks
     * complete out of order. */
    int dual;
    int dual_stagger_us;
    /* Per-call worker count (1..4).  workers >= 2 splits chunks stride-N
     * across pthreads on independent contexts.  Used as the stride in the
     * shared run_chunks_strided body so 2/3/4 workers all share one body. */
    int n_workers;
    pthread_t ane_thread_b;
    int ane_thread_b_started;
    int ane_done_b;
    int ane_failed_b;
    int ane_threads_remaining;
    ds4_ane_mlp_int8w_ctx *ane_ctx_b;
    int8_t *x_i8_batch_b;
    uint16_t *out_f16_batch_b;
    uint8_t *chunks_done_bitmap;
    double eval_ms_b;
    uint64_t eval_calls_b;
    uint64_t eval_refs_b;
    uint64_t padded_refs_b;
    /* 3rd/4th worker state (only used when n_workers >= 3 / >= 4). */
    pthread_t ane_thread_c;
    int ane_thread_c_started;
    ds4_ane_mlp_int8w_ctx *ane_ctx_c;
    int8_t *x_i8_batch_c;
    uint16_t *out_f16_batch_c;
    double eval_ms_c;
    uint64_t eval_calls_c;
    uint64_t eval_refs_c;
    uint64_t padded_refs_c;
    pthread_t ane_thread_d;
    int ane_thread_d_started;
    ds4_ane_mlp_int8w_ctx *ane_ctx_d;
    int8_t *x_i8_batch_d;
    uint16_t *out_f16_batch_d;
    double eval_ms_d;
    uint64_t eval_calls_d;
    uint64_t eval_refs_d;
    uint64_t padded_refs_d;
};

static void ds4_gpu_ane_prefill_job_free(ds4_gpu_ane_prefill_job *job) {
    if (!job) return;
    /* Defensive: if a worker errored before releasing the slot, release it
     * here so the pool doesn't leak.  Workers null out dequant_slot under
     * the mutex when they release. */
    if (job->dequant_slot >= 0) {
        ds4_gpu_ane_release_dequant_slot(job->dequant_slot);
        job->dequant_slot = -1;
    }
    if (job->sync_initialized) {
        pthread_cond_destroy(&job->cv);
        pthread_mutex_destroy(&job->mu);
    }
    free(job->x_i8_batch);
    free(job->out_f16_batch);
    free(job->x_i8_batch_b);
    free(job->out_f16_batch_b);
    free(job->x_i8_batch_c);
    free(job->out_f16_batch_c);
    free(job->x_i8_batch_d);
    free(job->out_f16_batch_d);
    free(job->chunks_done_bitmap);
    job->x_i8_batch_b = NULL;
    job->out_f16_batch_b = NULL;
    job->x_i8_batch_c = NULL;
    job->out_f16_batch_c = NULL;
    job->x_i8_batch_d = NULL;
    job->out_f16_batch_d = NULL;
    job->chunks_done_bitmap = NULL;
    if (!job->out_f16_all_mtl) free(job->out_f16_all);
    job->out_f16_all = NULL;
    job->out_f16_all_mtl = nil;
    /* out_f32 aliases out_f32_mtl.contents; ARC releases the MTLBuffer when
     * we drop the __strong reference. */
    job->out_f32 = NULL;
    job->out_f32_mtl = nil;
    if (!job->route_weights_mtl) free(job->route_weights);
    job->route_weights = NULL;
    job->route_weights_mtl = nil;
    free(job);
}

static void *ds4_gpu_ane_prefill_tiled_fused_thread(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    /* Wait for GPU dequant of weights+inputs to complete before reading the
     * shared scratch buffers via job->gate_i8 / up_i8 / down_i8 / x_i8_all. */
    if (job->sync_initialized) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (!job->dequant_done_flag && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        const int failed = job->ane_failed;
        pthread_mutex_unlock(&job->mu);
        job->dequant_wait_ms += ds4_gpu_now_ms() - wait_t0;
        if (failed) return NULL;
    }
    for (uint32_t chunk_idx = 0; chunk_idx < job->chunk_count; chunk_idx++) {
        const uint32_t begin = chunk_idx * job->ane_chunk_refs;
        const uint32_t chunk = (job->n_tokens - begin) < job->ane_chunk_refs ?
            (job->n_tokens - begin) : job->ane_chunk_refs;
        const double x_zero_t0 = ds4_gpu_now_ms();
        memset(job->x_i8_batch, 0, (size_t)job->ane_batch * job->expert_in_dim);
        job->x_zero_ms += ds4_gpu_now_ms() - x_zero_t0;
        const double out_zero_t0 = ds4_gpu_now_ms();
        memset(job->out_f16_batch, 0, (size_t)job->out_bucket_elems64 * sizeof(uint16_t));
        job->out_zero_ms += ds4_gpu_now_ms() - out_zero_t0;
        const double x_copy_t0 = ds4_gpu_now_ms();
        memcpy(job->x_i8_batch,
               job->x_i8_all + (uint64_t)begin * job->expert_in_dim,
               (size_t)((uint64_t)chunk * job->expert_in_dim));
        job->x_copy_ms += ds4_gpu_now_ms() - x_copy_t0;

        job->eval_calls++;
        job->eval_refs += chunk;
        job->padded_refs += job->ane_batch;
        if (chunk <= 16u) {
            job->chunks_le16++;
        } else if (chunk <= 64u) {
            job->chunks_le64++;
        } else if (chunk <= 128u) {
            job->chunks_le128++;
        } else if (chunk == job->ane_batch) {
            job->chunks_full++;
        }

        const double eval_t0 = ds4_gpu_now_ms();
        const int ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(job->ane_ctx,
                                                             job->gate_i8,
                                                             job->up_i8,
                                                             job->down_i8,
                                                             job->x_i8_batch,
                                                             job->out_f16_batch) ? 1 : 0;
        const double eval_t1 = ds4_gpu_now_ms();
        job->eval_ms += eval_t1 - eval_t0;
        ds4_gpu_prefill_trace_interval("ane_eval", eval_t0, eval_t1);
        if (!ok) {
            pthread_mutex_lock(&job->mu);
            job->ane_failed = 1;
            job->ane_done = 1;
            pthread_cond_broadcast(&job->cv);
            pthread_mutex_unlock(&job->mu);
            return NULL;
        }

        const double f16_copy_t0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            memcpy(job->out_f16_all + (uint64_t)(begin + r) * job->out_dim,
                   job->out_f16_batch + (uint64_t)r * job->out_dim,
                   (size_t)job->out_dim * sizeof(uint16_t));
        }
        const double f16_copy_t1 = ds4_gpu_now_ms();
        job->output_f16_copy_ms += f16_copy_t1 - f16_copy_t0;
        ds4_gpu_prefill_trace_interval("out_f16_copy", f16_copy_t0, f16_copy_t1);

        pthread_mutex_lock(&job->mu);
        job->chunks_ready = chunk_idx + 1u;
        pthread_cond_broadcast(&job->cv);
        pthread_mutex_unlock(&job->mu);
    }
    int release_slot_idx = -1;
    pthread_mutex_lock(&job->mu);
    job->ane_done = 1;
    release_slot_idx = job->dequant_slot;
    job->dequant_slot = -1;
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    if (release_slot_idx >= 0) {
        ds4_gpu_ane_release_dequant_slot(release_slot_idx);
    }
    return NULL;
}

/* Shared worker body for dual-cluster ANE prefill.  Stride-iterates over
 * chunks (start_idx, start_idx+stride, ...) using the given context and
 * per-cluster scratch buffers, then atomically advances chunks_ready via the
 * bitmap so the post thread keeps its in-order semantics.  Per-cluster eval
 * ms accumulates into eval_ms_out; other per-chunk stats accumulate into
 * shared fields (small write races on stats are acceptable telemetry-only). */
static void ds4_gpu_ane_prefill_run_chunks_strided(ds4_gpu_ane_prefill_job *job,
                                                    ds4_ane_mlp_int8w_ctx *ane_ctx,
                                                    int8_t *x_i8_batch,
                                                    uint16_t *out_f16_batch,
                                                    uint32_t start_idx,
                                                    uint32_t stride,
                                                    double *eval_ms_out,
                                                    uint64_t *eval_calls_out,
                                                    uint64_t *eval_refs_out,
                                                    uint64_t *padded_refs_out) {
    for (uint32_t chunk_idx = start_idx; chunk_idx < job->chunk_count; chunk_idx += stride) {
        const uint32_t begin = chunk_idx * job->ane_chunk_refs;
        const uint32_t chunk = (job->n_tokens - begin) < job->ane_chunk_refs ?
            (job->n_tokens - begin) : job->ane_chunk_refs;

        const double x_zero_t0 = ds4_gpu_now_ms();
        memset(x_i8_batch, 0, (size_t)job->ane_batch * job->expert_in_dim);
        job->x_zero_ms += ds4_gpu_now_ms() - x_zero_t0;
        const double out_zero_t0 = ds4_gpu_now_ms();
        memset(out_f16_batch, 0, (size_t)job->out_bucket_elems64 * sizeof(uint16_t));
        job->out_zero_ms += ds4_gpu_now_ms() - out_zero_t0;
        const double x_copy_t0 = ds4_gpu_now_ms();
        memcpy(x_i8_batch,
               job->x_i8_all + (uint64_t)begin * job->expert_in_dim,
               (size_t)((uint64_t)chunk * job->expert_in_dim));
        job->x_copy_ms += ds4_gpu_now_ms() - x_copy_t0;

        if (eval_calls_out) (*eval_calls_out)++;
        if (eval_refs_out)  (*eval_refs_out) += chunk;
        if (padded_refs_out) (*padded_refs_out) += job->ane_batch;
        if (chunk <= 16u) { job->chunks_le16++; }
        else if (chunk <= 64u) { job->chunks_le64++; }
        else if (chunk <= 128u) { job->chunks_le128++; }
        else if (chunk == job->ane_batch) { job->chunks_full++; }

        const double eval_t0 = ds4_gpu_now_ms();
        const int ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(ane_ctx,
                                                             job->gate_i8,
                                                             job->up_i8,
                                                             job->down_i8,
                                                             x_i8_batch,
                                                             out_f16_batch) ? 1 : 0;
        const double eval_t1 = ds4_gpu_now_ms();
        if (eval_ms_out) *eval_ms_out += eval_t1 - eval_t0;
        ds4_gpu_prefill_trace_interval("ane_eval", eval_t0, eval_t1);
        if (!ok) {
            pthread_mutex_lock(&job->mu);
            job->ane_failed = 1;
            job->ane_done = 1;
            pthread_cond_broadcast(&job->cv);
            pthread_mutex_unlock(&job->mu);
            return;
        }

        const double f16_copy_t0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            memcpy(job->out_f16_all + (uint64_t)(begin + r) * job->out_dim,
                   out_f16_batch + (uint64_t)r * job->out_dim,
                   (size_t)job->out_dim * sizeof(uint16_t));
        }
        const double f16_copy_t1 = ds4_gpu_now_ms();
        job->output_f16_copy_ms += f16_copy_t1 - f16_copy_t0;
        ds4_gpu_prefill_trace_interval("out_f16_copy", f16_copy_t0, f16_copy_t1);

        pthread_mutex_lock(&job->mu);
        if (job->chunks_done_bitmap) {
            job->chunks_done_bitmap[chunk_idx] = 1;
            while (job->chunks_ready < job->chunk_count &&
                   job->chunks_done_bitmap[job->chunks_ready]) {
                job->chunks_ready++;
            }
        } else {
            /* Solo path fallback; shouldn't be hit when this helper is in use. */
            job->chunks_ready = chunk_idx + 1u;
        }
        pthread_cond_broadcast(&job->cv);
        pthread_mutex_unlock(&job->mu);
    }
}

static void *ds4_gpu_ane_prefill_tiled_fused_thread_dual_a(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    if (job->sync_initialized) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (!job->dequant_done_flag && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        const int failed = job->ane_failed;
        pthread_mutex_unlock(&job->mu);
        job->dequant_wait_ms += ds4_gpu_now_ms() - wait_t0;
        if (failed) goto signal_done;
    }
    {
        const uint32_t stride = job->n_workers >= 2 ? (uint32_t)job->n_workers : 2u;
        ds4_gpu_ane_prefill_run_chunks_strided(job, job->ane_ctx,
                                                job->x_i8_batch, job->out_f16_batch,
                                                0, stride,
                                                &job->eval_ms,
                                                &job->eval_calls,
                                                &job->eval_refs,
                                                &job->padded_refs);
    }
signal_done: {
    int release_slot_idx = -1;
    pthread_mutex_lock(&job->mu);
    if (--job->ane_threads_remaining <= 0) {
        job->ane_done = 1;
        release_slot_idx = job->dequant_slot;
        job->dequant_slot = -1;
    }
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    if (release_slot_idx >= 0) {
        ds4_gpu_ane_release_dequant_slot(release_slot_idx);
    }
    return NULL;
}
}

static void *ds4_gpu_ane_prefill_tiled_fused_thread_dual_b(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    if (job->sync_initialized) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (!job->dequant_done_flag && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        const int failed = job->ane_failed;
        pthread_mutex_unlock(&job->mu);
        job->dequant_wait_ms += ds4_gpu_now_ms() - wait_t0;
        if (failed) goto signal_done;
    }
    if (job->dual_stagger_us > 0) {
        usleep((useconds_t)job->dual_stagger_us);
    }
    {
        const uint32_t stride = job->n_workers >= 2 ? (uint32_t)job->n_workers : 2u;
        ds4_gpu_ane_prefill_run_chunks_strided(job, job->ane_ctx_b,
                                                job->x_i8_batch_b, job->out_f16_batch_b,
                                                1, stride,
                                                &job->eval_ms_b,
                                                &job->eval_calls_b,
                                                &job->eval_refs_b,
                                                &job->padded_refs_b);
    }
signal_done: {
    int release_slot_idx = -1;
    pthread_mutex_lock(&job->mu);
    if (--job->ane_threads_remaining <= 0) {
        job->ane_done = 1;
        release_slot_idx = job->dequant_slot;
        job->dequant_slot = -1;
    }
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    if (release_slot_idx >= 0) {
        ds4_gpu_ane_release_dequant_slot(release_slot_idx);
    }
    return NULL;
}
}

/* 3rd/4th worker bodies (only spawned when n_workers >= 3 / >= 4).  Identical
 * to dual_b except for the worker index used for the chunk stride and the
 * context + scratch buffer set.  Each will be time-shared by the OS scheduler
 * onto one of the two physical ANE clusters. */
static void *ds4_gpu_ane_prefill_tiled_fused_thread_dual_c(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    if (job->sync_initialized) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (!job->dequant_done_flag && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        const int failed = job->ane_failed;
        pthread_mutex_unlock(&job->mu);
        job->dequant_wait_ms += ds4_gpu_now_ms() - wait_t0;
        if (failed) goto signal_done;
    }
    if (job->dual_stagger_us > 0) {
        /* Stagger worker C by 2x the configured offset so all N threads land
         * with even phase separation. */
        usleep((useconds_t)job->dual_stagger_us * 2u);
    }
    {
        const uint32_t stride = job->n_workers >= 2 ? (uint32_t)job->n_workers : 3u;
        ds4_gpu_ane_prefill_run_chunks_strided(job, job->ane_ctx_c,
                                                job->x_i8_batch_c, job->out_f16_batch_c,
                                                2, stride,
                                                &job->eval_ms_c,
                                                &job->eval_calls_c,
                                                &job->eval_refs_c,
                                                &job->padded_refs_c);
    }
signal_done: {
    int release_slot_idx = -1;
    pthread_mutex_lock(&job->mu);
    if (--job->ane_threads_remaining <= 0) {
        job->ane_done = 1;
        release_slot_idx = job->dequant_slot;
        job->dequant_slot = -1;
    }
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    if (release_slot_idx >= 0) {
        ds4_gpu_ane_release_dequant_slot(release_slot_idx);
    }
    return NULL;
}
}

static void *ds4_gpu_ane_prefill_tiled_fused_thread_dual_d(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    if (job->sync_initialized) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (!job->dequant_done_flag && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        const int failed = job->ane_failed;
        pthread_mutex_unlock(&job->mu);
        job->dequant_wait_ms += ds4_gpu_now_ms() - wait_t0;
        if (failed) goto signal_done;
    }
    if (job->dual_stagger_us > 0) {
        usleep((useconds_t)job->dual_stagger_us * 3u);
    }
    {
        const uint32_t stride = job->n_workers >= 2 ? (uint32_t)job->n_workers : 4u;
        ds4_gpu_ane_prefill_run_chunks_strided(job, job->ane_ctx_d,
                                                job->x_i8_batch_d, job->out_f16_batch_d,
                                                3, stride,
                                                &job->eval_ms_d,
                                                &job->eval_calls_d,
                                                &job->eval_refs_d,
                                                &job->padded_refs_d);
    }
signal_done: {
    int release_slot_idx = -1;
    pthread_mutex_lock(&job->mu);
    if (--job->ane_threads_remaining <= 0) {
        job->ane_done = 1;
        release_slot_idx = job->dequant_slot;
        job->dequant_slot = -1;
    }
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    if (release_slot_idx >= 0) {
        ds4_gpu_ane_release_dequant_slot(release_slot_idx);
    }
    return NULL;
}
}

static void *ds4_gpu_ane_prefill_post_thread(void *arg) {
    ds4_gpu_ane_prefill_job *job = (ds4_gpu_ane_prefill_job *)arg;
    if (!job) return NULL;
    for (;;) {
        const double wait_t0 = ds4_gpu_now_ms();
        pthread_mutex_lock(&job->mu);
        while (job->chunks_posted >= job->chunks_ready &&
               !job->ane_done &&
               !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        if (job->chunks_posted >= job->chunks_ready) {
            const int done = job->ane_done || job->ane_failed;
            pthread_mutex_unlock(&job->mu);
            if (done) break;
            continue;
        }
        const uint32_t chunk_idx = job->chunks_posted++;
        pthread_mutex_unlock(&job->mu);
        job->post_wait_ms += ds4_gpu_now_ms() - wait_t0;

        const uint32_t begin = chunk_idx * job->ane_chunk_refs;
        const uint32_t chunk = (job->n_tokens - begin) < job->ane_chunk_refs ?
            (job->n_tokens - begin) : job->ane_chunk_refs;
        const double output_t0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            const float rw = job->route_weights[begin + r];
            const uint64_t row = (uint64_t)(begin + r) * job->out_dim;
            ds4_gpu_f16_to_weighted_f32_row(job->out_f32 + row,
                                            job->out_f16_all + row,
                                            job->out_dim,
                                            rw,
                                            job->scalar_output_pack);
        }
        const double output_ms = ds4_gpu_now_ms() - output_t0;
        job->output_ms += output_ms;
        job->output_convert_ms += output_ms;
        job->output_elems += (uint64_t)chunk * job->out_dim;
        job->output_chunks++;
        ds4_gpu_prefill_trace_interval("output_convert",
                                       output_t0,
                                       output_t0 + output_ms);
    }
    pthread_mutex_lock(&job->mu);
    job->post_done = 1;
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->mu);
    return NULL;
}

ds4_gpu_ane_prefill_job *ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor(
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (!x || !weights || !gate_bank || !up_bank || !down_bank || n_tokens == 0) return NULL;
    if (gate_type != DS4_METAL_TENSOR_IQ2_XXS ||
        down_type != DS4_METAL_TENSOR_Q2_K ||
        expert_in_dim != 4096u ||
        expert_mid_dim != 2048u ||
        out_dim != 4096u) {
        return NULL;
    }

    const uint64_t iq2_row_bytes = (uint64_t)(expert_in_dim / 256u) * 66u;
    const uint64_t q2_row_bytes = (uint64_t)(expert_mid_dim / 256u) * 84u;
    if (gate_row_bytes != iq2_row_bytes ||
        down_row_bytes != q2_row_bytes ||
        gate_expert_bytes != gate_row_bytes * expert_mid_dim ||
        down_expert_bytes != down_row_bytes * out_dim) {
        return NULL;
    }

    float ane_qscale = 64.0f, ane_w_scale = 1.0f / 64.0f;
    float ane_x_qscale = 16.0f, ane_x_scale = 1.0f / 16.0f;
    float ane_mid_qscale = 16.0f, ane_mid_scale = 1.0f / 16.0f;
    uint32_t ane_batch = 256u, ane_compile_limit = 8u;
    ds4_gpu_ane_prefill_env(&ane_qscale,
                            &ane_w_scale,
                            &ane_x_qscale,
                            &ane_x_scale,
                            &ane_mid_qscale,
                            &ane_mid_scale,
                            &ane_batch,
                            &ane_compile_limit);
    ane_batch = ds4_gpu_ane_select_batch(n_tokens, ane_batch);
    if (ds4_gpu_ane_prefill_fp16w_enabled() ||
        ds4_gpu_ane_prefill_fp16x_i8w_enabled() ||
        !ds4_gpu_ane_prefill_i8i8_enabled() ||
        !ds4_gpu_ane_prefill_i8i8_tiled_fused_enabled()) {
        return NULL;
    }

    /* Dual-cluster split: when DUAL is on and n_tokens is large enough that a
     * split into two chunks is meaningful (>= 64 each side), pick a batch sized
     * to ceil(n_tokens/2) so this single call naturally produces 2 chunks that
     * the two cluster threads can run in parallel.  Without this, n_tokens
     * smaller than the selected batch would land in a single chunk and the
     * dual workers never both light up. */
    /* Threshold for "dual-split-eligible" call.  Default 128 has worked well;
     * env-tunable for sweep. */
    uint32_t dual_select_threshold = 128u;
    {
        const char *e = getenv("DS4_FLASH_MOE_ANE_DUAL_SPLIT_THRESH");
        if (e && e[0]) {
            long v = atol(e);
            if (v >= 1 && v <= 4096) dual_select_threshold = (uint32_t)v;
        }
    }
    const int planned_workers_for_split = ds4_gpu_ane_prefill_threads();
    /* When set, suppress both the multi-worker batch shrinkage below and the
     * chunk-refs rebalance further down so every ANE call keeps the same
     * batch regardless of THREADS — used to isolate cluster-parallelism wins
     * from per-call-batch-size effects. */
    int fixed_batch_for_threads = 0;
    {
        const char *e = getenv("DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH");
        if (e && e[0] && atoi(e) != 0) fixed_batch_for_threads = 1;
    }
    const int dual_select_intent = planned_workers_for_split >= 2 &&
                                    n_tokens >= dual_select_threshold &&
                                    !fixed_batch_for_threads;
    if (dual_select_intent) {
        const uint32_t denom = (uint32_t)planned_workers_for_split;
        const uint32_t per_worker = (n_tokens + denom - 1u) / denom;
        const uint32_t multi_batch = ds4_gpu_ane_select_batch(per_worker, ane_batch);
        if (multi_batch < ane_batch) ane_batch = multi_batch;
    }

    uint32_t ane_max_refs = ane_batch;
    const char *max_refs_env = getenv("DS4_FLASH_MOE_ANE_MAX_REFS");
    if (max_refs_env && max_refs_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(max_refs_env, &end, 10);
        if (end != max_refs_env && parsed > 0 && parsed <= 4096ul) ane_max_refs = (uint32_t)parsed;
    }
    uint32_t ane_chunk_refs = ane_max_refs < ane_batch ? ane_max_refs : ane_batch;
    if (ane_chunk_refs == 0) ane_chunk_refs = 1;
    /* Dual/multi chunk balancing: when N workers will fire (n_tokens >
     * ane_chunk_refs) with chunk_count==N, sizing the chunk to ceil(n_tokens/N)
     * gives all N workers equal work instead of N-1 at ane_chunk_refs and the
     * last one at the remainder.  Constrained by ane_chunk_refs ceiling so we
     * stay within ANE batch capability. */
    {
        const int planned_workers = ds4_gpu_ane_prefill_threads();
        if (!fixed_batch_for_threads &&
            planned_workers >= 2 &&
            n_tokens > ane_chunk_refs &&
            n_tokens <= (uint32_t)planned_workers * ane_chunk_refs) {
            const uint32_t balanced =
                (n_tokens + (uint32_t)planned_workers - 1u) / (uint32_t)planned_workers;
            if (balanced > 0 && balanced < ane_chunk_refs) {
                ane_chunk_refs = balanced;
            }
        }
    }
    const char *chunk_all_env = getenv("DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS");
    const int ane_chunk_big_refs = chunk_all_env && chunk_all_env[0] && atoi(chunk_all_env) != 0;

    g_ane_prefill_calls++;
    if (!ane_chunk_big_refs && n_tokens > ane_max_refs) {
        g_ane_prefill_skip_big_refs++;
        return NULL;
    }

    const uint64_t x_all_elems64 = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t x_batch_elems64 = (uint64_t)ane_batch * expert_in_dim;
    const uint64_t out_elems64 = (uint64_t)n_tokens * out_dim;
    const uint64_t out_bucket_elems64 = (uint64_t)ane_batch * out_dim;
    const uint64_t gate_i8_bytes = (uint64_t)expert_in_dim * expert_mid_dim;
    const uint64_t down_i8_bytes = (uint64_t)expert_mid_dim * out_dim;
    if (x_all_elems64 > UINT32_MAX ||
        x_all_elems64 > SIZE_MAX ||
        x_batch_elems64 > SIZE_MAX ||
        out_elems64 > SIZE_MAX / sizeof(float) ||
        out_bucket_elems64 > SIZE_MAX / sizeof(uint16_t)) {
        return NULL;
    }

    int ok = 0;
    int reopened = 0;
    int dequant_committed = 0;
    id<MTLCommandBuffer> cb = nil;
    int owned = 0;
    ds4_gpu_ane_prefill_job *job = NULL;

    /* Commit any in-flight GPU prefill work asynchronously and open a fresh CB
     * for our dequant kernels. Previously this used the synchronous
     * end_commands which stalled the main thread waiting for prefill to drain
     * — that drain was the dominant overhead per ANE submission.
     *
     * Pre-flush batching: every Kth call we skip the pre-flush, letting the
     * dequant kernels pile onto the existing batch_cb alongside prior pending
     * encodes.  When the CB eventually commits (next pre-flush or end of
     * layer), all dequant + GPU work commits together.  This saves K-1 Metal
     * commit syscalls per K calls; each commit is ~100-300 us so over ~5000
     * ANE calls the savings are substantial.  Controlled by
     * DS4_FLASH_MOE_ANE_PREFLUSH_EVERY (default 1 = old behavior). */
    {
        static int preflush_every_cached = -1;
        static int preflush_call_counter = 0;
        if (preflush_every_cached < 0) {
            const char *e = getenv("DS4_FLASH_MOE_ANE_PREFLUSH_EVERY");
            int v = (e && e[0]) ? atoi(e) : 1;
            if (v < 1) v = 1;
            if (v > 32) v = 32;
            preflush_every_cached = v;
        }
        const int do_preflush = (preflush_call_counter++ % preflush_every_cached) == 0;
        if (do_preflush) {
            if (ds4_gpu_flush_commands() == 0) goto done;
        }
    }
    if (!ds4_gpu_ensure_mpp_int8_prefill_pipelines()) goto done;

    /* Acquire a per-job dequant slot so this call's dequant doesn't share the
     * global scratch with any in-flight ANE worker still memcpy-ing from it.
     * This unblocks multi-active scheduler mode (multiple eval threads). */
    const int dequant_slot_idx = ds4_gpu_ane_acquire_dequant_slot((NSUInteger)gate_i8_bytes,
                                                                   (NSUInteger)gate_i8_bytes,
                                                                   (NSUInteger)down_i8_bytes,
                                                                   (NSUInteger)x_all_elems64);
    if (dequant_slot_idx < 0) goto done;
    ds4_ane_dequant_slot_t *slot = &g_ane_dequant_slots[dequant_slot_idx];

    /* Allocate the job + initialize its sync primitives BEFORE we attach a
     * completion handler that signals through them. */
    job = (ds4_gpu_ane_prefill_job *)calloc(1, sizeof(*job));
    if (!job) {
        ds4_gpu_ane_release_dequant_slot(dequant_slot_idx);
        goto done;
    }
    if (pthread_mutex_init(&job->mu, NULL) != 0 ||
        pthread_cond_init(&job->cv, NULL) != 0) {
        ds4_gpu_ane_release_dequant_slot(dequant_slot_idx);
        goto done;
    }
    job->sync_initialized = 1;
    job->dequant_slot = dequant_slot_idx;

    const double dequant_t0 = ds4_gpu_now_ms();
    cb = ds4_gpu_command_buffer(&owned);
    if (!cb) goto done;
    /* Try fused gate+up+down dispatch first when DS4_FLASH_MOE_ANE_FUSED_DEQUANT=1
     * (or default on for the DSv4 shape, which always has matching totals).
     * Falls back to 3 separate dispatches when the fused pipeline is
     * unavailable or when shapes don't satisfy the matching-total constraint. */
    int used_fused = 0;
    {
        static int fused_cached = -1;
        if (fused_cached < 0) {
            const char *e = getenv("DS4_FLASH_MOE_ANE_FUSED_DEQUANT");
            /* Default OFF: measured no perf benefit on DSv4 (Metal already
             * shared the compute encoder across the 3 standalone dispatches,
             * so API savings are small).  Kept as a code path for shapes
             * where the encoder isn't shared. */
            fused_cached = (e && e[0]) ? (atoi(e) != 0) : 0;
        }
        if (fused_cached) {
            used_fused = ds4_gpu_encode_mpp_dequant_gud_i8(cb,
                                                          ds4_gpu_tensor_buffer(gate_bank),
                                                          ds4_gpu_tensor_offset(gate_bank),
                                                          slot->gate_i8,
                                                          ds4_gpu_tensor_buffer(up_bank),
                                                          ds4_gpu_tensor_offset(up_bank),
                                                          slot->up_i8,
                                                          ds4_gpu_tensor_buffer(down_bank),
                                                          ds4_gpu_tensor_offset(down_bank),
                                                          slot->down_i8,
                                                          expert_mid_dim,
                                                          expert_in_dim,
                                                          out_dim,
                                                          ane_qscale);
        }
    }
    if (used_fused) {
        ok = ds4_gpu_encode_mpp_quant_f32_i8(cb,
                                             ds4_gpu_tensor_buffer(x),
                                             ds4_gpu_tensor_offset(x),
                                             slot->x_i8,
                                             (uint32_t)x_all_elems64,
                                             ane_x_qscale);
    } else {
        ok = ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                   ds4_gpu_tensor_buffer(gate_bank),
                                                   ds4_gpu_tensor_offset(gate_bank),
                                                   slot->gate_i8,
                                                   expert_mid_dim,
                                                   expert_in_dim,
                                                   ane_qscale) &&
             ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                   ds4_gpu_tensor_buffer(up_bank),
                                                   ds4_gpu_tensor_offset(up_bank),
                                                   slot->up_i8,
                                                   expert_mid_dim,
                                                   expert_in_dim,
                                                   ane_qscale) &&
             ds4_gpu_encode_mpp_dequant_q2_k_i8(cb,
                                                ds4_gpu_tensor_buffer(down_bank),
                                                ds4_gpu_tensor_offset(down_bank),
                                                slot->down_i8,
                                                out_dim,
                                                expert_mid_dim,
                                                ane_qscale) &&
             ds4_gpu_encode_mpp_quant_f32_i8(cb,
                                             ds4_gpu_tensor_buffer(x),
                                             ds4_gpu_tensor_offset(x),
                                             slot->x_i8,
                                             (uint32_t)x_all_elems64,
                                             ane_x_qscale);
    }
    if (!ok) goto done;

    /* Async drain: when the dequant CB completes on GPU, signal the per-job
     * flag so the CPU ANE pthread can proceed. Then flush (commit + open a
     * fresh CB) so the caller can continue encoding prefill work in parallel
     * with dequant + ANE eval. */
    job->dequant_submit_t0 = dequant_t0;
    ds4_gpu_ane_prefill_job *job_for_handler = job;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull cb_arg){
        (void)cb_arg;
        pthread_mutex_lock(&job_for_handler->mu);
        job_for_handler->dequant_done_flag = 1;
        pthread_cond_broadcast(&job_for_handler->cv);
        pthread_mutex_unlock(&job_for_handler->mu);
    }];
    /* DS4_FLASH_MOE_ANE_DEFER_DEQUANT_COMMIT=1: skip the immediate per-call
     * flush.  The next ANE start_tensor's pre-flush commits this dequant CB,
     * effectively batching N calls' dequant kernels into one commit.  Each
     * job still attaches its own completion handler so all jobs' dequant_done
     * flags fire when the batched CB completes.  ANE workers wait slightly
     * longer for dequant_done (one call interval) but this is hidden by the
     * multi-active scheduler pipeline. */
    {
        static int defer_cached = -1;
        if (defer_cached < 0) {
            const char *e = getenv("DS4_FLASH_MOE_ANE_DEFER_DEQUANT_COMMIT");
            defer_cached = (e && e[0] && atoi(e) != 0) ? 1 : 0;
        }
        if (!defer_cached) {
            if (ds4_gpu_flush_commands() == 0) { ok = 0; goto done; }
            dequant_committed = 1;
        } else {
            /* Mark "committed" so the error-path drain doesn't wait on a
             * dequant_done_flag that will only fire after a future flush. */
            dequant_committed = 1;
        }
    }
    g_ane_prefill_dequant_ms += ds4_gpu_now_ms() - dequant_t0;

    ds4_ane_mlp_int8w_ctx *ane_ctx =
        ds4_gpu_ane_get_i8i8_tiled_ctx(expert_in_dim,
                                       expert_mid_dim,
                                       ane_batch,
                                       ane_w_scale,
                                       ane_x_scale,
                                       ane_mid_scale,
                                       ane_compile_limit);
    if (!ane_ctx) goto done;

    job->n_tokens = n_tokens;
    job->expert_in_dim = expert_in_dim;
    job->expert_mid_dim = expert_mid_dim;
    job->out_dim = out_dim;
    job->ane_batch = ane_batch;
    job->ane_chunk_refs = ane_chunk_refs;
    job->scalar_output_pack = ds4_gpu_ane_scalar_output_pack_enabled();
    job->gpu_output_pack = ds4_gpu_ane_gpu_output_pack_enabled();
    job->ane_w_scale = ane_w_scale;
    job->ane_x_scale = ane_x_scale;
    job->ane_mid_scale = ane_mid_scale;
    job->ane_ctx = ane_ctx;
    job->gate_i8 = (const int8_t *)[slot->gate_i8 contents];
    job->up_i8 = (const int8_t *)[slot->up_i8 contents];
    job->down_i8 = (const int8_t *)[slot->down_i8 contents];
    job->x_i8_all = (const int8_t *)[slot->x_i8 contents];
    job->out_elems64 = out_elems64;
    job->out_bucket_elems64 = out_bucket_elems64;
    job->chunk_count = (n_tokens + ane_chunk_refs - 1u) / ane_chunk_refs;
    job->x_i8_batch = (int8_t *)malloc((size_t)x_batch_elems64);
    job->out_f16_batch = (uint16_t *)malloc((size_t)out_bucket_elems64 * sizeof(uint16_t));

    /* Dual/multi-worker setup: allocate a 2nd/3rd/4th context + scratch +
     * bitmap when DS4_FLASH_MOE_ANE_THREADS>=2 and there is enough work to
     * split (chunks_count >= N).  If any extra context compile fails we cap
     * the worker count at the number that succeeded; if even the 2nd fails we
     * silently fall back to single-cluster. */
    job->ane_threads_remaining = 1;
    job->dual = 0;
    job->n_workers = 1;
    const int requested_workers = ds4_gpu_ane_prefill_threads();
    /* For SOLO (single-chunk) ANE calls with multi enabled, alternate which
     * cluster the call lands on so the workload between A and B clusters is
     * balanced — without this, every single-chunk call runs on A and B is
     * idle.  With multi-active scheduler, this lets two solo calls run in
     * parallel on different clusters.  Stays with A/B since the 3rd/4th
     * context just gets time-shared back onto A or B. */
    if (requested_workers >= 2 && job->chunk_count == 1) {
        static int solo_balance_counter = 0;
        const int use_b = (solo_balance_counter++ & 1) != 0;
        if (use_b) {
            ds4_ane_mlp_int8w_ctx *ctx_b =
                ds4_gpu_ane_get_i8i8_tiled_ctx_b(expert_in_dim,
                                                  expert_mid_dim,
                                                  ane_batch,
                                                  ane_w_scale,
                                                  ane_x_scale,
                                                  ane_mid_scale,
                                                  ane_compile_limit);
            if (ctx_b) job->ane_ctx = ctx_b;  /* land on cluster B */
        }
    }
    if (requested_workers >= 2 && job->chunk_count >= 2) {
        ds4_ane_mlp_int8w_ctx *ane_ctx_b =
            ds4_gpu_ane_get_i8i8_tiled_ctx_b(expert_in_dim,
                                              expert_mid_dim,
                                              ane_batch,
                                              ane_w_scale,
                                              ane_x_scale,
                                              ane_mid_scale,
                                              ane_compile_limit);
        if (ane_ctx_b) {
            job->ane_ctx_b = ane_ctx_b;
            job->x_i8_batch_b = (int8_t *)malloc((size_t)x_batch_elems64);
            job->out_f16_batch_b = (uint16_t *)malloc((size_t)out_bucket_elems64 * sizeof(uint16_t));
            job->chunks_done_bitmap = (uint8_t *)calloc((size_t)job->chunk_count, 1);
            job->dual_stagger_us = ds4_gpu_ane_prefill_dual_stagger_us();
            if (job->x_i8_batch_b && job->out_f16_batch_b && job->chunks_done_bitmap) {
                job->dual = 1;
                job->n_workers = 2;
                job->ane_threads_remaining = 2;
            }
        }
    }
    if (job->n_workers == 2 && requested_workers >= 3 && job->chunk_count >= 3) {
        ds4_ane_mlp_int8w_ctx *ane_ctx_c =
            ds4_gpu_ane_get_i8i8_tiled_ctx_c(expert_in_dim,
                                              expert_mid_dim,
                                              ane_batch,
                                              ane_w_scale,
                                              ane_x_scale,
                                              ane_mid_scale,
                                              ane_compile_limit);
        if (ane_ctx_c) {
            job->ane_ctx_c = ane_ctx_c;
            job->x_i8_batch_c = (int8_t *)malloc((size_t)x_batch_elems64);
            job->out_f16_batch_c = (uint16_t *)malloc((size_t)out_bucket_elems64 * sizeof(uint16_t));
            if (job->x_i8_batch_c && job->out_f16_batch_c) {
                job->n_workers = 3;
                job->ane_threads_remaining = 3;
            }
        }
    }
    if (job->n_workers == 3 && requested_workers >= 4 && job->chunk_count >= 4) {
        ds4_ane_mlp_int8w_ctx *ane_ctx_d =
            ds4_gpu_ane_get_i8i8_tiled_ctx_d(expert_in_dim,
                                              expert_mid_dim,
                                              ane_batch,
                                              ane_w_scale,
                                              ane_x_scale,
                                              ane_mid_scale,
                                              ane_compile_limit);
        if (ane_ctx_d) {
            job->ane_ctx_d = ane_ctx_d;
            job->x_i8_batch_d = (int8_t *)malloc((size_t)x_batch_elems64);
            job->out_f16_batch_d = (uint16_t *)malloc((size_t)out_bucket_elems64 * sizeof(uint16_t));
            if (job->x_i8_batch_d && job->out_f16_batch_d) {
                job->n_workers = 4;
                job->ane_threads_remaining = 4;
            }
        }
    }
    /* Reuse the slot's pooled MTLBuffers instead of newBufferWithLength: per
     * call.  Each ANE prefill call typically needs ~2-4 MB of output buffers;
     * pooling saves ~500ms-2s of buffer creation per prefill at production
     * workload.  ensure_scratch_buffer grows in place if the existing
     * capacity is smaller than requested. */
    const NSUInteger out_f16_all_bytes_req = (NSUInteger)(out_elems64 * sizeof(uint16_t));
    const NSUInteger route_weights_bytes_req = (NSUInteger)n_tokens * sizeof(float);
    const NSUInteger out_f32_bytes_req = (NSUInteger)(out_elems64 * sizeof(float));
    if (job->gpu_output_pack) {
        if (!ds4_gpu_ensure_scratch_buffer(&slot->out_f16_all, &slot->out_f16_all_bytes,
                                           out_f16_all_bytes_req, "ds4_ane_slot_out_f16_all") ||
            !ds4_gpu_ensure_scratch_buffer(&slot->route_weights, &slot->route_weights_bytes,
                                           route_weights_bytes_req, "ds4_ane_slot_route_weights")) {
            goto done;
        }
        job->out_f16_all_mtl = slot->out_f16_all;
        job->out_f16_all = (uint16_t *)[slot->out_f16_all contents];
        job->route_weights_mtl = slot->route_weights;
        job->route_weights = (float *)[slot->route_weights contents];
    } else {
        if (!ds4_gpu_ensure_scratch_buffer(&slot->out_f32, &slot->out_f32_bytes,
                                           out_f32_bytes_req, "ds4_ane_slot_out_f32")) {
            goto done;
        }
        job->out_f32_mtl = slot->out_f32;
        job->out_f32 = (float *)[slot->out_f32 contents];
        /* out_f16_all in non-pack mode is a CPU-side scratch the post thread
         * fills row-by-row.  Keep this on malloc for now — it's freed in
         * job_free; a future pool can swap it in. */
        job->out_f16_all = (uint16_t *)malloc((size_t)out_elems64 * sizeof(uint16_t));
        job->route_weights = (float *)malloc((size_t)n_tokens * sizeof(float));
    }
    if (!job->x_i8_all || !job->x_i8_batch || !job->out_f16_batch ||
        !job->out_f16_all || (!job->gpu_output_pack && (!job->out_f32 || !job->out_f32_mtl)) ||
        (job->gpu_output_pack && (!job->out_f16_all_mtl || !job->route_weights_mtl)) ||
        !job->route_weights ||
        !job->gate_i8 || !job->up_i8 || !job->down_i8) {
        goto done;
    }

    const float *w_ptr = (const float *)((const uint8_t *)[ds4_gpu_tensor_buffer(weights) contents] +
                                        ds4_gpu_tensor_offset(weights));
    if (!w_ptr) goto done;
    const double input_t0 = ds4_gpu_now_ms();
    memcpy(job->route_weights, w_ptr, (size_t)n_tokens * sizeof(float));
    g_ane_prefill_input_ms += ds4_gpu_now_ms() - input_t0;

    if (!job->gpu_output_pack) {
        if (pthread_create(&job->post_thread, NULL, ds4_gpu_ane_prefill_post_thread, job) != 0) {
            goto done;
        }
        job->post_thread_started = 1;
    }
    if (job->dual) {
        if (pthread_create(&job->ane_thread, NULL,
                           ds4_gpu_ane_prefill_tiled_fused_thread_dual_a, job) != 0) {
            goto done;
        }
        job->ane_thread_started = 1;
        if (pthread_create(&job->ane_thread_b, NULL,
                           ds4_gpu_ane_prefill_tiled_fused_thread_dual_b, job) != 0) {
            /* Thread B failed to spawn; we still have thread A running.  Force
             * the unsstarted workers (B/C/D if planned) to be accounted for so
             * threads_remaining decrement reaches zero and ane_done fires. */
            pthread_mutex_lock(&job->mu);
            int phantom = job->n_workers - 1; /* B and any of C,D not yet started */
            while (phantom-- > 0) {
                if (--job->ane_threads_remaining <= 0) job->ane_done = 1;
            }
            pthread_cond_broadcast(&job->cv);
            pthread_mutex_unlock(&job->mu);
            goto done;
        }
        job->ane_thread_b_started = 1;
        if (job->n_workers >= 3) {
            if (pthread_create(&job->ane_thread_c, NULL,
                               ds4_gpu_ane_prefill_tiled_fused_thread_dual_c, job) != 0) {
                pthread_mutex_lock(&job->mu);
                int phantom = job->n_workers - 2; /* C and possibly D */
                while (phantom-- > 0) {
                    if (--job->ane_threads_remaining <= 0) job->ane_done = 1;
                }
                pthread_cond_broadcast(&job->cv);
                pthread_mutex_unlock(&job->mu);
                goto done;
            }
            job->ane_thread_c_started = 1;
        }
        if (job->n_workers >= 4) {
            if (pthread_create(&job->ane_thread_d, NULL,
                               ds4_gpu_ane_prefill_tiled_fused_thread_dual_d, job) != 0) {
                pthread_mutex_lock(&job->mu);
                if (--job->ane_threads_remaining <= 0) job->ane_done = 1;
                pthread_cond_broadcast(&job->cv);
                pthread_mutex_unlock(&job->mu);
                goto done;
            }
            job->ane_thread_d_started = 1;
        }
    } else {
        if (pthread_create(&job->ane_thread, NULL,
                           ds4_gpu_ane_prefill_tiled_fused_thread, job) != 0) {
            goto done;
        }
        job->ane_thread_started = 1;
    }
    ok = 1;

done:
    /* After our flushes the GPU command buffer is already open; only call
     * begin_commands when no CB is currently held. */
    if (g_batch_cb != nil) {
        reopened = 1;
    } else if (ds4_gpu_begin_commands() != 0) {
        reopened = 1;
    }
    if (!ok) {
        if (job) {
            if (job->sync_initialized) {
                pthread_mutex_lock(&job->mu);
                job->ane_failed = 1;
                job->ane_done = 1;
                pthread_cond_broadcast(&job->cv);
                /* If the dequant CB was committed its completion handler
                 * captures `job` and will signal dequant_done_flag from a
                 * Metal-private thread. We must not free the job until that
                 * handler has fired. */
                while (dequant_committed && !job->dequant_done_flag) {
                    pthread_cond_wait(&job->cv, &job->mu);
                }
                pthread_mutex_unlock(&job->mu);
            }
            if (job->ane_thread_started) {
                pthread_join(job->ane_thread, NULL);
                job->ane_thread_started = 0;
            }
            if (job->ane_thread_b_started) {
                pthread_join(job->ane_thread_b, NULL);
                job->ane_thread_b_started = 0;
            }
            if (job->ane_thread_c_started) {
                pthread_join(job->ane_thread_c, NULL);
                job->ane_thread_c_started = 0;
            }
            if (job->ane_thread_d_started) {
                pthread_join(job->ane_thread_d, NULL);
                job->ane_thread_d_started = 0;
            }
            if (job->post_thread_started) {
                pthread_join(job->post_thread, NULL);
                job->post_thread_started = 0;
            }
        }
        ds4_gpu_ane_prefill_job_free(job);
        if (!reopened) g_ane_prefill_reopen_failures++;
        return NULL;
    }
    if (!reopened) {
        if (job) {
            if (job->sync_initialized) {
                pthread_mutex_lock(&job->mu);
                job->ane_failed = 1;
                job->ane_done = 1;
                pthread_cond_broadcast(&job->cv);
                while (dequant_committed && !job->dequant_done_flag) {
                    pthread_cond_wait(&job->cv, &job->mu);
                }
                pthread_mutex_unlock(&job->mu);
            }
            if (job->ane_thread_started) {
                pthread_join(job->ane_thread, NULL);
                job->ane_thread_started = 0;
            }
            if (job->ane_thread_b_started) {
                pthread_join(job->ane_thread_b, NULL);
                job->ane_thread_b_started = 0;
            }
            if (job->ane_thread_c_started) {
                pthread_join(job->ane_thread_c, NULL);
                job->ane_thread_c_started = 0;
            }
            if (job->ane_thread_d_started) {
                pthread_join(job->ane_thread_d, NULL);
                job->ane_thread_d_started = 0;
            }
            if (job->post_thread_started) {
                pthread_join(job->post_thread, NULL);
                job->post_thread_started = 0;
            }
        }
        ds4_gpu_ane_prefill_job_free(job);
        g_ane_prefill_reopen_failures++;
        return NULL;
    }
    return job;
}

int ds4_gpu_routed_moe_expert_banked_batch_ane_wait_predict_tensor(
        ds4_gpu_ane_prefill_job *job) {
    if (!job) return 0;
    g_ane_prefill_wait_predict_calls++;
    const double wait_t0 = ds4_gpu_now_ms();
    if (job->ane_thread_started) {
        pthread_join(job->ane_thread, NULL);
        job->ane_thread_started = 0;
    } else if (job->sync_initialized) {
        pthread_mutex_lock(&job->mu);
        while (!job->ane_done && !job->ane_failed) {
            pthread_cond_wait(&job->cv, &job->mu);
        }
        pthread_mutex_unlock(&job->mu);
    }
    if (job->ane_thread_b_started) {
        pthread_join(job->ane_thread_b, NULL);
        job->ane_thread_b_started = 0;
    }
    if (job->ane_thread_c_started) {
        pthread_join(job->ane_thread_c, NULL);
        job->ane_thread_c_started = 0;
    }
    if (job->ane_thread_d_started) {
        pthread_join(job->ane_thread_d, NULL);
        job->ane_thread_d_started = 0;
    }
    g_ane_prefill_wait_predict_ms += ds4_gpu_now_ms() - wait_t0;
    return !job->ane_failed && job->ane_done;
}

int ds4_gpu_routed_moe_expert_banked_batch_ane_finish_tensor(
        ds4_gpu_ane_prefill_job *job,
        ds4_gpu_tensor          *out,
        bool                    *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!job || !out) {
        ds4_gpu_ane_prefill_job_free(job);
        return 0;
    }
    g_ane_prefill_finish_calls++;
    if (job->ane_thread_started) {
        const double join_t0 = ds4_gpu_now_ms();
        pthread_join(job->ane_thread, NULL);
        g_ane_prefill_finish_join_ane_ms += ds4_gpu_now_ms() - join_t0;
        job->ane_thread_started = 0;
    }
    if (job->ane_thread_b_started) {
        const double join_t0 = ds4_gpu_now_ms();
        pthread_join(job->ane_thread_b, NULL);
        g_ane_prefill_finish_join_ane_ms += ds4_gpu_now_ms() - join_t0;
        job->ane_thread_b_started = 0;
    }
    if (job->ane_thread_c_started) {
        const double join_t0 = ds4_gpu_now_ms();
        pthread_join(job->ane_thread_c, NULL);
        g_ane_prefill_finish_join_ane_ms += ds4_gpu_now_ms() - join_t0;
        job->ane_thread_c_started = 0;
    }
    if (job->ane_thread_d_started) {
        const double join_t0 = ds4_gpu_now_ms();
        pthread_join(job->ane_thread_d, NULL);
        g_ane_prefill_finish_join_ane_ms += ds4_gpu_now_ms() - join_t0;
        job->ane_thread_d_started = 0;
    }
    if (job->post_thread_started) {
        const double join_t0 = ds4_gpu_now_ms();
        pthread_join(job->post_thread, NULL);
        g_ane_prefill_finish_join_post_ms += ds4_gpu_now_ms() - join_t0;
        job->post_thread_started = 0;
    }
    g_ane_prefill_eval_ms += job->eval_ms;
    g_ane_prefill_eval_ms_b += job->eval_ms_b;
    g_ane_prefill_eval_ms_c += job->eval_ms_c;
    g_ane_prefill_eval_ms_d += job->eval_ms_d;
    g_ane_prefill_output_ms += job->output_ms;
    g_ane_prefill_dequant_wait_ms += job->dequant_wait_ms;
    g_ane_prefill_x_zero_ms += job->x_zero_ms;
    g_ane_prefill_x_copy_ms += job->x_copy_ms;
    g_ane_prefill_out_zero_ms += job->out_zero_ms;
    g_ane_prefill_output_f16_copy_ms += job->output_f16_copy_ms;
    g_ane_prefill_output_convert_ms += job->output_convert_ms;
    g_ane_prefill_post_wait_ms += job->post_wait_ms;
    g_ane_prefill_output_elems += job->output_elems;
    g_ane_prefill_output_chunks += job->output_chunks;
    {
        const uint64_t total_eval_calls = job->eval_calls + job->eval_calls_b +
                                          job->eval_calls_c + job->eval_calls_d;
        const uint64_t total_eval_refs  = job->eval_refs  + job->eval_refs_b  +
                                          job->eval_refs_c  + job->eval_refs_d;
        const uint64_t total_padded_refs= job->padded_refs+ job->padded_refs_b+
                                          job->padded_refs_c+ job->padded_refs_d;
        g_ane_prefill_eval_calls   += total_eval_calls;
        g_ane_prefill_eval_refs    += total_eval_refs;
        g_ane_prefill_padded_refs  += total_padded_refs;
        g_ane_prefill_chunks_le16  += job->chunks_le16;
        g_ane_prefill_chunks_le64  += job->chunks_le64;
        g_ane_prefill_chunks_le128 += job->chunks_le128;
        g_ane_prefill_chunks_full  += job->chunks_full;
        ds4_gpu_ane_record_batch_stats(job->ane_batch,
                                       total_eval_calls,
                                       total_eval_refs,
                                       total_padded_refs);
        g_ane_prefill_ane_evaluate_calls +=
            total_eval_calls * ds4_gpu_ane_mode_evaluate_calls(6);
    }
    int ok = !job->ane_failed && !job->post_failed && job->ane_done &&
        (job->gpu_output_pack || job->post_done);
    if (!ok) {
        g_ane_prefill_eval_failures++;
        ds4_gpu_ane_prefill_job_free(job);
        return 0;
    }

    /* Writeback ANE output into the destination tensor as a GPU blit encoded
     * on the live command buffer. Metal serializes this after any previously
     * committed scatter_add reading the same buffer — no CPU/GPU race, no
     * sync drain required. The post_thread has already joined above, so the
     * out_f32_mtl bytes are fully written before we encode. */
    const double writeback_t0 = ds4_gpu_now_ms();
    if (!g_batch_cb) {
        if (ds4_gpu_begin_commands() == 0) {
            g_ane_prefill_write_failures++;
            ds4_gpu_ane_prefill_job_free(job);
            return 0;
        }
    }
    id<MTLBuffer> dst_buf = ds4_gpu_tensor_buffer(out);
    const NSUInteger dst_off = ds4_gpu_tensor_offset(out);
    const NSUInteger bytes = (NSUInteger)(job->out_elems64 * sizeof(float));
    if (job->gpu_output_pack) {
        const NSUInteger f16_bytes = (NSUInteger)(job->out_elems64 * sizeof(uint16_t));
        const NSUInteger weight_bytes = (NSUInteger)job->n_tokens * sizeof(float);
        if (!dst_buf || !job->out_f16_all_mtl || !job->route_weights_mtl ||
            job->out_f16_all_mtl.length < f16_bytes ||
            job->route_weights_mtl.length < weight_bytes) {
            g_ane_prefill_write_failures++;
            ds4_gpu_ane_prefill_job_free(job);
            return 0;
        }
        const double output_t0 = ds4_gpu_now_ms();
        ok = ds4_gpu_encode_ane_output_pack_f16_f32(g_batch_cb,
                                                    job->out_f16_all_mtl,
                                                    0,
                                                    dst_buf,
                                                    dst_off,
                                                    job->route_weights_mtl,
                                                    0,
                                                    job->out_dim,
                                                    job->n_tokens);
        const double output_t1 = ds4_gpu_now_ms();
        if (!ok) {
            g_ane_prefill_write_failures++;
            ds4_gpu_ane_prefill_job_free(job);
            return 0;
        }
        const double output_ms = output_t1 - output_t0;
        g_ane_prefill_output_ms += output_ms;
        g_ane_prefill_output_convert_ms += output_ms;
        g_ane_prefill_output_gpu_encode_ms += output_ms;
        g_ane_prefill_output_elems += job->out_elems64;
        g_ane_prefill_output_chunks += 1;
        ds4_gpu_prefill_trace_interval("output_gpu_encode", output_t0, output_t1);
        g_ane_prefill_writeback_ms += output_t1 - writeback_t0;
        g_ane_prefill_successes++;
        ds4_gpu_ane_prefill_job_free(job);
        return 1;
    }
    if (!dst_buf || job->out_f32_mtl.length < bytes) {
        g_ane_prefill_write_failures++;
        ds4_gpu_ane_prefill_job_free(job);
        return 0;
    }
    ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    if (!blit) {
        g_ane_prefill_write_failures++;
        ds4_gpu_ane_prefill_job_free(job);
        return 0;
    }
    [blit copyFromBuffer:job->out_f32_mtl
            sourceOffset:0
                toBuffer:dst_buf
       destinationOffset:dst_off
                    size:bytes];
    [blit endEncoding];
    ok = 1;
    const double writeback_t1 = ds4_gpu_now_ms();
    g_ane_prefill_writeback_ms += writeback_t1 - writeback_t0;
    ds4_gpu_prefill_trace_interval("writeback_encode", writeback_t0, writeback_t1);
    g_ane_prefill_successes++;
    ds4_gpu_ane_prefill_job_free(job);
    return ok;
}

int ds4_gpu_routed_moe_expert_banked_batch_ane_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    (void)gate;
    (void)up;
    (void)mid;
    (void)selected;
    (void)clamp;
    if (mid_is_f16) *mid_is_f16 = false;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !weights || !gate_bank || !up_bank || !down_bank || n_tokens == 0) return 0;
    if (gate_type != DS4_METAL_TENSOR_IQ2_XXS ||
        down_type != DS4_METAL_TENSOR_Q2_K ||
        expert_in_dim != 4096u ||
        expert_mid_dim != 2048u ||
        out_dim != 4096u) {
        return 0;
    }

    const uint64_t iq2_row_bytes = (uint64_t)(expert_in_dim / 256u) * 66u;
    const uint64_t q2_row_bytes = (uint64_t)(expert_mid_dim / 256u) * 84u;
    if (gate_row_bytes != iq2_row_bytes ||
        down_row_bytes != q2_row_bytes ||
        gate_expert_bytes != gate_row_bytes * expert_mid_dim ||
        down_expert_bytes != down_row_bytes * out_dim) {
        return 0;
    }

    float ane_qscale = 64.0f, ane_w_scale = 1.0f / 64.0f;
    float ane_x_qscale = 16.0f, ane_x_scale = 1.0f / 16.0f;
    float ane_mid_qscale = 16.0f, ane_mid_scale = 1.0f / 16.0f;
    uint32_t ane_batch = 256u, ane_compile_limit = 8u;
    ds4_gpu_ane_prefill_env(&ane_qscale,
                            &ane_w_scale,
                            &ane_x_qscale,
                            &ane_x_scale,
                            &ane_mid_qscale,
                            &ane_mid_scale,
                            &ane_batch,
                            &ane_compile_limit);
    uint32_t ane_max_refs = ane_batch;
    const char *max_refs_env = getenv("DS4_FLASH_MOE_ANE_MAX_REFS");
    if (max_refs_env && max_refs_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(max_refs_env, &end, 10);
        if (end != max_refs_env && parsed > 0 && parsed <= 4096ul) ane_max_refs = (uint32_t)parsed;
    }
    uint32_t ane_chunk_refs = ane_max_refs < ane_batch ? ane_max_refs : ane_batch;
    if (ane_chunk_refs == 0) ane_chunk_refs = 1;
    const char *chunk_all_env = getenv("DS4_FLASH_MOE_ANE_CHUNK_BIG_REFS");
    const int ane_chunk_big_refs = chunk_all_env && chunk_all_env[0] && atoi(chunk_all_env) != 0;
    const int ane_fp16w = ds4_gpu_ane_prefill_fp16w_enabled();
    const int ane_fp16x_i8w = !ane_fp16w && ds4_gpu_ane_prefill_fp16x_i8w_enabled();
    const int ane_i8i8 = !ane_fp16w && !ane_fp16x_i8w && ds4_gpu_ane_prefill_i8i8_enabled();
    const int ane_i8i8_fused = ane_i8i8 && ds4_gpu_ane_prefill_i8i8_fused_enabled();
    const int ane_i8i8_full_fused = ane_i8i8 && ds4_gpu_ane_prefill_i8i8_full_fused_enabled();
    const int ane_i8i8_tiled_fused = ane_i8i8 && ds4_gpu_ane_prefill_i8i8_tiled_fused_enabled();
    const int ane_mode = ane_fp16w ? 1 : (ane_fp16x_i8w ? 2 : (ane_i8i8 ? (ane_i8i8_tiled_fused ? 6 : (ane_i8i8_full_fused ? 4 : (ane_i8i8_fused ? 5 : 3))) : 0));
    g_ane_prefill_calls++;
    if (!ane_chunk_big_refs && n_tokens > ane_max_refs) {
        g_ane_prefill_skip_big_refs++;
        if (ds4_gpu_ane_debug_enabled()) {
            fprintf(stderr, "ds4: ANE tensor skip refs=%u > max_refs=%u\n", n_tokens, ane_max_refs);
        }
        return 0;
    }

    const uint64_t x_elems64 = (uint64_t)ane_batch * expert_in_dim;
    const uint64_t out_elems64 = (uint64_t)n_tokens * out_dim;
    const uint64_t out_bucket_elems64 = (uint64_t)ane_batch * out_dim;
    const uint64_t gate_i8_bytes = (uint64_t)expert_in_dim * expert_mid_dim;
    const uint64_t down_i8_bytes = (uint64_t)expert_mid_dim * out_dim;
    const uint64_t gate_f16_bytes = gate_i8_bytes * sizeof(uint16_t);
    const uint64_t down_f16_bytes = down_i8_bytes * sizeof(uint16_t);
    if (x_elems64 > UINT32_MAX || out_elems64 > UINT32_MAX || out_bucket_elems64 > UINT32_MAX ||
        x_elems64 > SIZE_MAX / sizeof(float) ||
        out_elems64 > SIZE_MAX / sizeof(float) ||
        out_bucket_elems64 > SIZE_MAX / sizeof(uint16_t)) {
        return 0;
    }

    int reopened = 0;
    int ok = 0;
    int8_t *x_i8 = NULL;
    uint16_t *x_f16 = NULL;
    uint16_t *route_f16 = NULL;
    uint16_t *out_f16 = NULL;
    float *out_f32 = NULL;
    id<MTLCommandBuffer> cb = nil;
    int owned = 0;
    const int ane_debug = ds4_gpu_ane_debug_enabled();

    if (ane_debug) {
        fprintf(stderr, "ds4: ANE tensor enter mode=%s refs=%u batch=%u chunk_refs=%u dims H=%u I=%u O=%u qscale=%.6g\n",
                ds4_gpu_ane_mode_name(ane_mode),
                n_tokens, ane_batch, ane_chunk_refs, expert_in_dim, expert_mid_dim, out_dim, ane_qscale);
    }
    if (ds4_gpu_end_commands() == 0) goto done;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor command flush ok\n");

    if (!ds4_gpu_ensure_mpp_int8_prefill_pipelines()) goto done;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor int8 dequant pipelines ok\n");
    const int ane_weight_f16 = ane_fp16w;
    if (!ds4_gpu_ensure_scratch_buffer(ane_weight_f16 ? &g_ane_prefill_gate_f16_buffer : &g_ane_prefill_gate_i8_buffer,
                                       ane_weight_f16 ? &g_ane_prefill_gate_f16_bytes : &g_ane_prefill_gate_i8_bytes,
                                       (NSUInteger)(ane_weight_f16 ? gate_f16_bytes : gate_i8_bytes),
                                       ane_weight_f16 ? "ds4_ane_prefill_gate_f16" : "ds4_ane_prefill_gate_i8") ||
        !ds4_gpu_ensure_scratch_buffer(ane_weight_f16 ? &g_ane_prefill_up_f16_buffer : &g_ane_prefill_up_i8_buffer,
                                       ane_weight_f16 ? &g_ane_prefill_up_f16_bytes : &g_ane_prefill_up_i8_bytes,
                                       (NSUInteger)(ane_weight_f16 ? gate_f16_bytes : gate_i8_bytes),
                                       ane_weight_f16 ? "ds4_ane_prefill_up_f16" : "ds4_ane_prefill_up_i8") ||
        !ds4_gpu_ensure_scratch_buffer(ane_weight_f16 ? &g_ane_prefill_down_f16_buffer : &g_ane_prefill_down_i8_buffer,
                                       ane_weight_f16 ? &g_ane_prefill_down_f16_bytes : &g_ane_prefill_down_i8_bytes,
                                       (NSUInteger)(ane_weight_f16 ? down_f16_bytes : down_i8_bytes),
                                       ane_weight_f16 ? "ds4_ane_prefill_down_f16" : "ds4_ane_prefill_down_i8")) {
        goto done;
    }
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor scratch ok gate=%llu down=%llu\n",
                           (unsigned long long)gate_i8_bytes,
                           (unsigned long long)down_i8_bytes);

    if (ane_debug) fprintf(stderr, "ds4: ANE tensor begin dequant command batch\n");
    const double dequant_t0 = ds4_gpu_now_ms();
    if (ds4_gpu_begin_commands() == 0) goto done;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor command batch opened\n");
    cb = ds4_gpu_command_buffer(&owned);
    if (!cb) goto done;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor encoding dequant\n");
    if (ane_fp16w) {
        ok = ds4_gpu_encode_ane_dequant_iq2_xxs_f16(cb,
                                                    ds4_gpu_tensor_buffer(gate_bank),
                                                    ds4_gpu_tensor_offset(gate_bank),
                                                    g_ane_prefill_gate_f16_buffer,
                                                    expert_mid_dim,
                                                    expert_in_dim) &&
             ds4_gpu_encode_ane_dequant_iq2_xxs_f16(cb,
                                                    ds4_gpu_tensor_buffer(up_bank),
                                                    ds4_gpu_tensor_offset(up_bank),
                                                    g_ane_prefill_up_f16_buffer,
                                                    expert_mid_dim,
                                                    expert_in_dim) &&
             ds4_gpu_encode_ane_dequant_q2_k_f16(cb,
                                                 ds4_gpu_tensor_buffer(down_bank),
                                                 ds4_gpu_tensor_offset(down_bank),
                                                 g_ane_prefill_down_f16_buffer,
                                                 out_dim,
                                                 expert_mid_dim);
    } else {
        ok = ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                   ds4_gpu_tensor_buffer(gate_bank),
                                                   ds4_gpu_tensor_offset(gate_bank),
                                                   g_ane_prefill_gate_i8_buffer,
                                                   expert_mid_dim,
                                                   expert_in_dim,
                                                   ane_qscale) &&
             ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                   ds4_gpu_tensor_buffer(up_bank),
                                                   ds4_gpu_tensor_offset(up_bank),
                                                   g_ane_prefill_up_i8_buffer,
                                                   expert_mid_dim,
                                                   expert_in_dim,
                                                   ane_qscale) &&
             ds4_gpu_encode_mpp_dequant_q2_k_i8(cb,
                                                ds4_gpu_tensor_buffer(down_bank),
                                                ds4_gpu_tensor_offset(down_bank),
                                                g_ane_prefill_down_i8_buffer,
                                                out_dim,
                                                expert_mid_dim,
                                                ane_qscale);
    }
    ok = ds4_gpu_end_commands() != 0 && ok;
    g_ane_prefill_dequant_ms += ds4_gpu_now_ms() - dequant_t0;
    if (!ok) goto done;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor dequant complete\n");

    const int ane_x_f16 = ane_fp16w || ane_fp16x_i8w;
    x_i8 = ane_x_f16 ? NULL : (int8_t *)malloc((size_t)x_elems64);
    x_f16 = ane_x_f16 ? (uint16_t *)malloc((size_t)x_elems64 * sizeof(uint16_t)) : NULL;
    route_f16 = (uint16_t *)malloc((size_t)ane_batch * sizeof(uint16_t));
    out_f16 = (uint16_t *)malloc((size_t)out_bucket_elems64 * sizeof(uint16_t));
    out_f32 = (float *)malloc((size_t)out_elems64 * sizeof(float));
    if ((!ane_x_f16 && !x_i8) || (ane_x_f16 && !x_f16) || !route_f16 || !out_f16 || !out_f32) goto done;

    const float *x_ptr = (const float *)((const uint8_t *)[ds4_gpu_tensor_buffer(x) contents] +
                                        ds4_gpu_tensor_offset(x));
    const float *w_ptr = (const float *)((const uint8_t *)[ds4_gpu_tensor_buffer(weights) contents] +
                                        ds4_gpu_tensor_offset(weights));
    if (!x_ptr || !w_ptr) goto done;
    if (!g_ane_int8w_ctx ||
        ds4_ane_mlp_int8w_mode(g_ane_int8w_ctx) != ane_mode ||
        ds4_ane_mlp_int8w_H(g_ane_int8w_ctx) != (int)expert_in_dim ||
        ds4_ane_mlp_int8w_I(g_ane_int8w_ctx) != (int)expert_mid_dim ||
        ds4_ane_mlp_int8w_B(g_ane_int8w_ctx) != (int)ane_batch ||
        (ane_fp16x_i8w &&
         fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - ane_w_scale) > 1.0e-8f) ||
        (ane_i8i8 &&
         (fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - ane_w_scale) > 1.0e-8f ||
          fabsf(ds4_ane_mlp_int8w_x_scale(g_ane_int8w_ctx) - ane_x_scale) > 1.0e-8f ||
          fabsf(ds4_ane_mlp_int8w_mid_scale(g_ane_int8w_ctx) - ane_mid_scale) > 1.0e-8f)) ||
        (!ane_fp16w && !ane_fp16x_i8w && !ane_i8i8 &&
         (fabsf(ds4_ane_mlp_int8w_scale(g_ane_int8w_ctx) - ane_w_scale) > 1.0e-8f ||
          fabsf(ds4_ane_mlp_int8w_x_scale(g_ane_int8w_ctx) - ane_x_scale) > 1.0e-8f))) {
        if (g_ane_prefill_compile_attempts >= ane_compile_limit) {
            g_ane_prefill_compile_failures++;
            fprintf(stderr,
                    "ds4: ANE %s prefill compile budget exhausted attempts=%llu limit=%u; falling back to GPU\n",
                    ds4_gpu_ane_mode_name(ane_mode),
                    (unsigned long long)g_ane_prefill_compile_attempts,
                    ane_compile_limit);
            goto done;
        }
        ds4_ane_mlp_int8w_destroy(g_ane_int8w_ctx);
        g_ane_prefill_compile_attempts++;
        g_ane_int8w_ctx = ane_fp16w ?
            ds4_ane_mlp_fp16w_create((int)expert_in_dim,
                                     (int)expert_mid_dim,
                                     (int)ane_batch) :
            (ane_fp16x_i8w ?
             ds4_ane_mlp_i8w_fp16x_create((int)expert_in_dim,
                                          (int)expert_mid_dim,
                                          (int)ane_batch,
                                          ane_w_scale) :
             (ane_i8i8 ?
              (ane_i8i8_tiled_fused ?
               ds4_ane_mlp_i8w_i8x_tiled_fused_create((int)expert_in_dim,
                                                       (int)expert_mid_dim,
                                                       (int)ane_batch,
                                                       ane_w_scale,
                                                       ane_x_scale,
                                                       ane_mid_scale) :
               (ane_i8i8_full_fused ?
               ds4_ane_mlp_i8w_i8x_fused_create((int)expert_in_dim,
                                                 (int)expert_mid_dim,
                                                 (int)ane_batch,
                                                 ane_w_scale,
                                                 ane_x_scale,
                                                 ane_mid_scale) :
               (ane_i8i8_fused ?
                ds4_ane_mlp_i8w_i8x_gateup_fused_create((int)expert_in_dim,
                                                         (int)expert_mid_dim,
                                                         (int)ane_batch,
                                                         ane_w_scale,
                                                         ane_x_scale,
                                                         ane_mid_scale) :
                ds4_ane_mlp_i8w_i8x_create((int)expert_in_dim,
                                           (int)expert_mid_dim,
                                           (int)ane_batch,
                                           ane_w_scale,
                                           ane_x_scale,
                                           ane_mid_scale)))) :
              ds4_ane_mlp_int8w_create((int)expert_in_dim,
                                       (int)expert_mid_dim,
                                       (int)ane_batch,
                                       ane_w_scale,
                                       ane_x_scale)));
        if (!g_ane_int8w_ctx) {
            g_ane_prefill_compile_failures++;
            fprintf(stderr,
                    "ds4: ANE %s prefill compile failed B=%u H=%u I=%u w_qscale=%.6g x_qscale=%.6g\n",
                    ds4_gpu_ane_mode_name(ane_mode),
                    ane_batch, expert_in_dim, expert_mid_dim, ane_qscale, ane_x_qscale);
            goto done;
        }
        fprintf(stderr, "ds4: ANE %s prefill compiled B=%u H=%u I=%u w_qscale=%.6g x_qscale=%.6g mid_qscale=%.6g\n",
                ds4_gpu_ane_mode_name(ane_mode),
                ane_batch, expert_in_dim, expert_mid_dim, ane_qscale, ane_x_qscale, ane_mid_qscale);
    }

    const int scalar_output_pack = ds4_gpu_ane_scalar_output_pack_enabled();
    for (uint32_t begin = 0; begin < n_tokens; begin += ane_chunk_refs) {
        const uint32_t chunk = (n_tokens - begin) < ane_chunk_refs ?
            (n_tokens - begin) : ane_chunk_refs;
        if (ane_x_f16) {
            memset(x_f16, 0, (size_t)x_elems64 * sizeof(uint16_t));
        } else {
            memset(x_i8, 0, (size_t)x_elems64);
        }
        memset(route_f16, 0, (size_t)ane_batch * sizeof(uint16_t));
        memset(out_f16, 0, (size_t)out_bucket_elems64 * sizeof(uint16_t));
        const double input_t0 = ds4_gpu_now_ms();
        for (uint64_t i = 0; i < (uint64_t)chunk * expert_in_dim; i++) {
            const float xv = x_ptr[(uint64_t)begin * expert_in_dim + i];
            const float axv = fabsf(xv);
            if (axv > g_ane_prefill_x_abs_max) g_ane_prefill_x_abs_max = axv;
            g_ane_prefill_x_values++;
            if (ane_x_f16) {
                x_f16[i] = ds4_gpu_f32_to_f16_bits(xv);
            } else {
                float v = nearbyintf(xv * ane_x_qscale);
                if (v < -128.0f) {
                    v = -128.0f;
                    g_ane_prefill_x_saturated++;
                }
                if (v > 127.0f) {
                    v = 127.0f;
                    g_ane_prefill_x_saturated++;
                }
                x_i8[i] = (int8_t)v;
            }
        }
        for (uint32_t i = 0; i < chunk; i++) {
            route_f16[i] = ds4_gpu_f32_to_f16_bits(w_ptr[begin + i]);
        }
        g_ane_prefill_input_ms += ds4_gpu_now_ms() - input_t0;
        if (ane_debug) {
            fprintf(stderr, "ds4: ANE tensor eval chunk begin=%u refs=%u\n", begin, chunk);
        }
        g_ane_prefill_eval_calls++;
        g_ane_prefill_eval_refs += chunk;
        g_ane_prefill_padded_refs += ane_batch;
        if (chunk <= 16u) {
            g_ane_prefill_chunks_le16++;
        } else if (chunk <= 64u) {
            g_ane_prefill_chunks_le64++;
        } else if (chunk <= 128u) {
            g_ane_prefill_chunks_le128++;
        } else if (chunk == ane_batch) {
            g_ane_prefill_chunks_full++;
        }
        const double eval_t0 = ds4_gpu_now_ms();
        if (ane_fp16w) {
            ok = ds4_ane_mlp_fp16w_eval(g_ane_int8w_ctx,
                                        (const uint16_t *)[g_ane_prefill_gate_f16_buffer contents],
                                        (const uint16_t *)[g_ane_prefill_up_f16_buffer contents],
                                        (const uint16_t *)[g_ane_prefill_down_f16_buffer contents],
                                        x_f16,
                                        out_f16);
        } else if (ane_fp16x_i8w) {
            ok = ds4_ane_mlp_i8w_fp16x_eval(g_ane_int8w_ctx,
                                            (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                            x_f16,
                                            out_f16);
        } else if (ane_i8i8_tiled_fused) {
            ok = ds4_ane_mlp_i8w_i8x_tiled_fused_eval(g_ane_int8w_ctx,
                                                       (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                                       (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                                       (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                                       x_i8,
                                                       out_f16);
        } else if (ane_i8i8_full_fused) {
            ok = ds4_ane_mlp_i8w_i8x_fused_eval(g_ane_int8w_ctx,
                                                (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                                (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                                (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                                x_i8,
                                                out_f16);
        } else if (ane_i8i8_fused) {
            ok = ds4_ane_mlp_i8w_i8x_gateup_fused_eval(g_ane_int8w_ctx,
                                                       (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                                       (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                                       (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                                       x_i8,
                                                       out_f16);
        } else if (ane_i8i8) {
            ok = ds4_ane_mlp_i8w_i8x_eval(g_ane_int8w_ctx,
                                          (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                          (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                          (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                          x_i8,
                                          out_f16);
        } else {
            ok = ds4_ane_mlp_int8w_eval(g_ane_int8w_ctx,
                                        (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                        (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                        (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                        x_i8,
                                        route_f16,
                                        out_f16);
        }
        g_ane_prefill_eval_ms += ds4_gpu_now_ms() - eval_t0;
        if (!ok) {
            g_ane_prefill_eval_failures++;
            goto done;
        }
        g_ane_prefill_ane_evaluate_calls += ds4_gpu_ane_mode_evaluate_calls(ane_mode);
        const uint32_t compare_limit = ds4_gpu_ane_compare_limit();
        if (compare_limit > 0 && g_ane_prefill_compare_count < compare_limit && chunk > 0) {
            g_ane_prefill_compare_count++;
            const uint64_t compare_call_idx = g_ane_prefill_eval_calls;
            if (ane_fp16w) {
                ds4_gpu_ane_compare_one_row("fp16w",
                                            (const uint16_t *)[g_ane_prefill_gate_f16_buffer contents],
                                            (const uint16_t *)[g_ane_prefill_up_f16_buffer contents],
                                            (const uint16_t *)[g_ane_prefill_down_f16_buffer contents],
                                            NULL,
                                            NULL,
                                            NULL,
                                            x_f16,
                                            NULL,
                                            out_f16,
                                            expert_in_dim,
                                            expert_mid_dim,
                                            compare_call_idx,
                                            n_tokens,
                                            begin,
                                            chunk,
                                            1.0f,
                                            1.0f,
                                            1.0f,
                                            0);
            } else if (ane_fp16x_i8w) {
                ds4_gpu_ane_compare_one_row("i8w-fp16x",
                                            NULL,
                                            NULL,
                                            NULL,
                                            (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                            x_f16,
                                            NULL,
                                            out_f16,
                                            expert_in_dim,
                                            expert_mid_dim,
                                            compare_call_idx,
                                            n_tokens,
                                            begin,
                                            chunk,
                                            ane_w_scale,
                                            1.0f,
                                            1.0f,
                                            0);
            } else if (ane_i8i8) {
                ds4_gpu_ane_compare_one_row(ds4_gpu_ane_mode_name(ane_mode),
                                            NULL,
                                            NULL,
                                            NULL,
                                            (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                            NULL,
                                            x_i8,
                                            out_f16,
                                            expert_in_dim,
                                            expert_mid_dim,
                                            compare_call_idx,
                                            n_tokens,
                                            begin,
                                            chunk,
                                            ane_w_scale,
                                            ane_x_scale,
                                            ane_mid_scale,
                                            1);
            } else {
                ds4_gpu_ane_compare_one_row("int8w",
                                            NULL,
                                            NULL,
                                            NULL,
                                            (const int8_t *)[g_ane_prefill_gate_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_up_i8_buffer contents],
                                            (const int8_t *)[g_ane_prefill_down_i8_buffer contents],
                                            NULL,
                                            x_i8,
                                            out_f16,
                                            expert_in_dim,
                                            expert_mid_dim,
                                            compare_call_idx,
                                            n_tokens,
                                            begin,
                                            chunk,
                                            ane_w_scale,
                                            ane_x_scale,
                                            ane_x_scale,
                                            0);
            }
        }
        const double output_t0 = ds4_gpu_now_ms();
        for (uint32_t r = 0; r < chunk; r++) {
            const float rw = w_ptr[begin + r];
            ds4_gpu_f16_to_weighted_f32_row(out_f32 + (uint64_t)(begin + r) * out_dim,
                                            out_f16 + (uint64_t)r * out_dim,
                                            out_dim,
                                            rw,
                                            scalar_output_pack);
        }
        const double output_ms = ds4_gpu_now_ms() - output_t0;
        g_ane_prefill_output_ms += output_ms;
        g_ane_prefill_output_convert_ms += output_ms;
        g_ane_prefill_output_elems += (uint64_t)chunk * out_dim;
        g_ane_prefill_output_chunks++;
    }
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor eval ok\n");
    const double writeback_t0 = ds4_gpu_now_ms();
    ok = ds4_gpu_tensor_write(out, 0, out_f32, out_elems64 * sizeof(float)) != 0;
    g_ane_prefill_writeback_ms += ds4_gpu_now_ms() - writeback_t0;
    if (!ok) g_ane_prefill_write_failures++;
    if (ane_debug) fprintf(stderr, "ds4: ANE tensor writeback %s\n", ok ? "ok" : "failed");

done:
    if (ane_debug && !ok) fprintf(stderr, "ds4: ANE tensor failed before completion\n");
    free(x_i8);
    free(x_f16);
    free(route_f16);
    free(out_f16);
    free(out_f32);
    if (ds4_gpu_begin_commands() != 0) reopened = 1;
    if (!reopened) {
        g_ane_prefill_reopen_failures++;
        return 0;
    }
    if (ok) g_ane_prefill_successes++;
    return ok;
}

static const char *ds4_gpu_mpp_int8_source(void) {
    return
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "using namespace mpp::tensor_ops;\n"
        "\n"
        "template <typename AT, typename BT, typename CT>\n"
        "inline void ds4_mpp_run_tile(device AT *A, device BT *B, device CT *C,\n"
        "                             constant uint &M, constant uint &N, constant uint &K,\n"
        "                             uint2 tgid) {\n"
        "    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));\n"
        "    matmul2d<desc, execution_simdgroups<4>> op;\n"
        "    auto tA = tensor(A, dextents<int32_t, 2>{(int32_t)K, (int32_t)M}, array<int32_t, 2>{1, (int32_t)K});\n"
        "    auto tB = tensor(B, dextents<int32_t, 2>{(int32_t)N, (int32_t)K}, array<int32_t, 2>{1, (int32_t)N});\n"
        "    auto tC = tensor(C, dextents<int32_t, 2>{(int32_t)N, (int32_t)M}, array<int32_t, 2>{1, (int32_t)N});\n"
        "    auto mA = tA.slice(0, tgid.y * 64);\n"
        "    auto mB = tB.slice(tgid.x * 32, 0);\n"
        "    auto mC = tC.slice(tgid.x * 32, tgid.y * 64);\n"
        "    op.run(mA, mB, mC);\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_h_i8_f(device half *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
        "                          constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                          uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    ds4_mpp_run_tile<half, int8_t, float>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_f_i8_f(device float *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
        "                          constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                          uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    ds4_mpp_run_tile<float, int8_t, float>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_i8_i8_i32(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device int32_t *C [[buffer(2)]],\n"
        "                             constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                             uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    ds4_mpp_run_tile<int8_t, int8_t, int32_t>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_quant_f32_i8(device const float *src [[buffer(0)]], device int8_t *dst [[buffer(1)]],\n"
        "                               constant uint &n [[buffer(2)]], constant float &qscale [[buffer(3)]],\n"
        "                               uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid >= n) return;\n"
        "    int v = int(rint(clamp(src[tid] * qscale, -128.0f, 127.0f)));\n"
        "    dst[tid] = int8_t(v);\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_quant_mid_f32_i8(device const float *src [[buffer(0)]], device int8_t *dst [[buffer(1)]],\n"
        "                                  constant uint &width [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "                                  constant float &qscale [[buffer(4)]],\n"
        "                                  uint row [[threadgroup_position_in_grid]],\n"
        "                                  uint tid [[thread_position_in_threadgroup]],\n"
        "                                  uint ntg [[threads_per_threadgroup]]) {\n"
        "    if (row >= rows) return;\n"
        "    device const float *src_row = src + (ulong)row * width;\n"
        "    device int8_t *dst_row = dst + (ulong)row * width;\n"
        "    for (uint i = tid; i < width; i += ntg) {\n"
        "        int v = int(rint(clamp(src_row[i] * qscale, -128.0f, 127.0f)));\n"
        "        dst_row[i] = int8_t(v);\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_swiglu_weight_f32(device const float *gate [[buffer(0)]],\n"
        "                                     device const float *up [[buffer(1)]],\n"
        "                                     device float *mid [[buffer(2)]],\n"
        "                                     device const float *weights [[buffer(3)]],\n"
        "                                     constant uint &width [[buffer(4)]],\n"
        "                                     constant uint &rows [[buffer(5)]],\n"
        "                                     constant float &clamp_value [[buffer(6)]],\n"
        "                                     constant float &input_scale [[buffer(7)]],\n"
        "                                     uint row [[threadgroup_position_in_grid]],\n"
        "                                     uint tid [[thread_position_in_threadgroup]],\n"
        "                                     uint ntg [[threads_per_threadgroup]]) {\n"
        "    if (row >= rows) return;\n"
        "    const float route_weight = weights[row];\n"
        "    device const float *gate_row = gate + (ulong)row * width;\n"
        "    device const float *up_row = up + (ulong)row * width;\n"
        "    device float *mid_row = mid + (ulong)row * width;\n"
        "    for (uint i = tid; i < width; i += ntg) {\n"
        "        float g = gate_row[i] * input_scale;\n"
        "        float u = up_row[i] * input_scale;\n"
        "        if (clamp_value > 1.0e-6f) {\n"
        "            g = min(g, clamp_value);\n"
        "            u = clamp(u, -clamp_value, clamp_value);\n"
        "        }\n"
        "        const float silu = g / (1.0f + exp(-g));\n"
        "        mid_row[i] = silu * u * route_weight;\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_swiglu_i32_weight_f32(device const int32_t *gate [[buffer(0)]],\n"
        "                                         device const int32_t *up [[buffer(1)]],\n"
        "                                         device float *mid [[buffer(2)]],\n"
        "                                         device const float *weights [[buffer(3)]],\n"
        "                                         constant uint &width [[buffer(4)]],\n"
        "                                         constant uint &rows [[buffer(5)]],\n"
        "                                         constant float &clamp_value [[buffer(6)]],\n"
        "                                         constant float &input_scale [[buffer(7)]],\n"
        "                                         uint row [[threadgroup_position_in_grid]],\n"
        "                                         uint tid [[thread_position_in_threadgroup]],\n"
        "                                         uint ntg [[threads_per_threadgroup]]) {\n"
        "    if (row >= rows) return;\n"
        "    const float route_weight = weights[row];\n"
        "    device const int32_t *gate_row = gate + (ulong)row * width;\n"
        "    device const int32_t *up_row = up + (ulong)row * width;\n"
        "    device float *mid_row = mid + (ulong)row * width;\n"
        "    for (uint i = tid; i < width; i += ntg) {\n"
        "        float g = float(gate_row[i]) * input_scale;\n"
        "        float u = float(up_row[i]) * input_scale;\n"
        "        if (clamp_value > 1.0e-6f) {\n"
        "            g = min(g, clamp_value);\n"
        "            u = clamp(u, -clamp_value, clamp_value);\n"
        "        }\n"
        "        const float silu = g / (1.0f + exp(-g));\n"
        "        mid_row[i] = silu * u * route_weight;\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_swiglu_i32_weight_i8(device const int32_t *gate [[buffer(0)]],\n"
        "                                        device const int32_t *up [[buffer(1)]],\n"
        "                                        device int8_t *mid [[buffer(2)]],\n"
        "                                        device const float *weights [[buffer(3)]],\n"
        "                                        constant uint &width [[buffer(4)]],\n"
        "                                        constant uint &rows [[buffer(5)]],\n"
        "                                        constant float &clamp_value [[buffer(6)]],\n"
        "                                        constant float &input_scale [[buffer(7)]],\n"
        "                                        constant float &mid_qscale [[buffer(8)]],\n"
        "                                        uint row [[threadgroup_position_in_grid]],\n"
        "                                        uint tid [[thread_position_in_threadgroup]],\n"
        "                                        uint ntg [[threads_per_threadgroup]]) {\n"
        "    if (row >= rows) return;\n"
        "    const float route_weight = weights[row];\n"
        "    device const int32_t *gate_row = gate + (ulong)row * width;\n"
        "    device const int32_t *up_row = up + (ulong)row * width;\n"
        "    device int8_t *mid_row = mid + (ulong)row * width;\n"
        "    for (uint i = tid; i < width; i += ntg) {\n"
        "        float g = float(gate_row[i]) * input_scale;\n"
        "        float u = float(up_row[i]) * input_scale;\n"
        "        if (clamp_value > 1.0e-6f) {\n"
        "            g = min(g, clamp_value);\n"
        "            u = clamp(u, -clamp_value, clamp_value);\n"
        "        }\n"
        "        const float silu = g / (1.0f + exp(-g));\n"
        "        const float v = silu * u * route_weight * mid_qscale;\n"
        "        mid_row[i] = int8_t(int(rint(clamp(v, -128.0f, 127.0f))));\n"
        "    }\n"
        "}\n"
        "\n"
        "kernel void ds4_mpp_scale_i32_f32(device const int32_t *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "                                  constant uint &n [[buffer(2)]], constant float &scale [[buffer(3)]],\n"
        "                                  uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid >= n) return;\n"
        "    dst[tid] = float(src[tid]) * scale;\n"
        "}\n"
        "\n"
        "kernel void ds4_ane_output_pack_f16_f32(device const half *src [[buffer(0)]],\n"
        "                                        device float *dst [[buffer(1)]],\n"
        "                                        device const float *weights [[buffer(2)]],\n"
        "                                        constant uint &width [[buffer(3)]],\n"
        "                                        constant uint &rows [[buffer(4)]],\n"
        "                                        uint row [[threadgroup_position_in_grid]],\n"
        "                                        uint tid [[thread_position_in_threadgroup]],\n"
        "                                        uint ntg [[threads_per_threadgroup]]) {\n"
        "    if (row >= rows) return;\n"
        "    const float weight = weights[row];\n"
        "    device const half *src_row = src + (ulong)row * width;\n"
        "    device float *dst_row = dst + (ulong)row * width;\n"
        "    for (uint i = tid; i < width; i += ntg) {\n"
        "        dst_row[i] = float(src_row[i]) * weight;\n"
        "    }\n"
        "}\n";
}

static id<MTLComputePipelineState> ds4_gpu_make_mpp_pipeline(
        id<MTLLibrary> library,
        NSString      *name) {
    if (!library || !name) return nil;
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) return nil;
    id<MTLComputePipelineState> pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        fprintf(stderr, "ds4: MPP int8 pipeline %s failed: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

static int ds4_gpu_ensure_mpp_int8_prefill_pipelines(void) {
    if (g_mpp_h_i8_f_pipeline && g_mpp_f_i8_f_pipeline &&
        g_mpp_i8_i8_i32_pipeline &&
        g_mpp_swiglu_weight_f32_pipeline &&
        g_mpp_swiglu_i32_weight_f32_pipeline &&
        g_mpp_swiglu_i32_weight_i8_pipeline &&
        g_mpp_quant_f32_i8_pipeline &&
        g_mpp_quant_mid_f32_i8_pipeline &&
        g_mpp_scale_i32_f32_pipeline &&
        g_mpp_dequant_iq2_xxs_i8_pipeline &&
        g_mpp_dequant_q2_k_i8_pipeline &&
        g_ane_output_pack_f16_f32_pipeline) {
        return 1;
    }

    g_mpp_dequant_iq2_xxs_i8_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_mpp_dequant_iq2_xxs_transpose_i8");
    g_mpp_dequant_q2_k_i8_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_mpp_dequant_q2_k_transpose_i8");
    g_mpp_dequant_gud_i8_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_mpp_dequant_gate_up_down_i8");
    g_ane_dequant_iq2_xxs_f16_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_ane_dequant_iq2_xxs_transpose_f16");
    g_ane_dequant_q2_k_f16_pipeline =
        ds4_gpu_get_pipeline("kernel_dsv4_ane_dequant_q2_k_transpose_f16");
    if (!g_mpp_dequant_iq2_xxs_i8_pipeline || !g_mpp_dequant_q2_k_i8_pipeline ||
        !g_ane_dequant_iq2_xxs_f16_pipeline || !g_ane_dequant_q2_k_f16_pipeline) {
        return 0;
    }

    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> library =
        [g_device newLibraryWithSource:[NSString stringWithUTF8String:ds4_gpu_mpp_int8_source()]
                               options:options
                                 error:&error];
    if (!library) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "ds4: MPP int8 prefill shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
        }
        return 0;
    }

    g_mpp_h_i8_f_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_h_i8_f");
    g_mpp_f_i8_f_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_f_i8_f");
    g_mpp_i8_i8_i32_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_i8_i8_i32");
    g_mpp_swiglu_weight_f32_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_swiglu_weight_f32");
    g_mpp_swiglu_i32_weight_f32_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_swiglu_i32_weight_f32");
    g_mpp_swiglu_i32_weight_i8_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_swiglu_i32_weight_i8");
    g_mpp_quant_f32_i8_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_quant_f32_i8");
    g_mpp_quant_mid_f32_i8_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_quant_mid_f32_i8");
    g_mpp_scale_i32_f32_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_mpp_scale_i32_f32");
    g_ane_output_pack_f16_f32_pipeline =
        ds4_gpu_make_mpp_pipeline(library, @"ds4_ane_output_pack_f16_f32");

    return g_mpp_h_i8_f_pipeline && g_mpp_f_i8_f_pipeline &&
           g_mpp_i8_i8_i32_pipeline &&
           g_mpp_swiglu_weight_f32_pipeline &&
           g_mpp_swiglu_i32_weight_f32_pipeline &&
           g_mpp_swiglu_i32_weight_i8_pipeline &&
           g_mpp_quant_f32_i8_pipeline &&
           g_mpp_quant_mid_f32_i8_pipeline &&
           g_mpp_scale_i32_f32_pipeline &&
           g_ane_output_pack_f16_f32_pipeline;
}

static int ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols,
        float                qscale) {
    if (!cb || !src || !dst || q_rows == 0 || q_cols == 0 || (q_cols % 256u) != 0) return 0;
    const uint32_t total = q_rows * (q_cols / 256u) * 16u;
    NSUInteger nth = g_mpp_dequant_iq2_xxs_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_dequant_iq2_xxs_i8_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    [enc setBytes:&qscale length:sizeof(qscale) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)total + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_dequant_q2_k_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols,
        float                qscale) {
    if (!cb || !src || !dst || q_rows == 0 || q_cols == 0 || (q_cols % 256u) != 0) return 0;
    const uint32_t total = q_rows * (q_cols / 256u) * 16u;
    NSUInteger nth = g_mpp_dequant_q2_k_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_dequant_q2_k_i8_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    [enc setBytes:&qscale length:sizeof(qscale) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)total + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_dequant_gud_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        gate_src, NSUInteger gate_src_off, id<MTLBuffer> gate_dst,
        id<MTLBuffer>        up_src,   NSUInteger up_src_off,   id<MTLBuffer> up_dst,
        id<MTLBuffer>        down_src, NSUInteger down_src_off, id<MTLBuffer> down_dst,
        uint32_t             expert_mid_dim,
        uint32_t             expert_in_dim,
        uint32_t             out_dim,
        float                qscale) {
    if (!cb || !gate_src || !up_src || !down_src ||
        !gate_dst || !up_dst || !down_dst ||
        expert_mid_dim == 0 || expert_in_dim == 0 || out_dim == 0 ||
        (expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) {
        return 0;
    }
    if (!g_mpp_dequant_gud_i8_pipeline) return 0;
    /* Pre-check: all three dispatches must have matching total work items so
     * one launch covers all of them.  For DSv4 (in=4096, mid=2048, out=4096)
     * this is naturally satisfied (524288 each). */
    const uint32_t gu_total   = expert_mid_dim * (expert_in_dim  / 256u) * 16u;
    const uint32_t down_total = out_dim        * (expert_mid_dim / 256u) * 16u;
    if (gu_total != down_total) return 0;
    const uint32_t total = gu_total;
    NSUInteger nth = g_mpp_dequant_gud_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_dequant_gud_i8_pipeline];
    [enc setBuffer:gate_src offset:gate_src_off atIndex:0];
    [enc setBuffer:up_src   offset:up_src_off   atIndex:1];
    [enc setBuffer:down_src offset:down_src_off atIndex:2];
    [enc setBuffer:gate_dst offset:0 atIndex:3];
    [enc setBuffer:up_dst   offset:0 atIndex:4];
    [enc setBuffer:down_dst offset:0 atIndex:5];
    [enc setBytes:&expert_mid_dim length:sizeof(expert_mid_dim) atIndex:6];
    [enc setBytes:&expert_in_dim  length:sizeof(expert_in_dim)  atIndex:7];
    [enc setBytes:&out_dim        length:sizeof(out_dim)        atIndex:8];
    [enc setBytes:&expert_mid_dim length:sizeof(expert_mid_dim) atIndex:9]; // d_q_cols == mid
    [enc setBytes:&total length:sizeof(total) atIndex:10];
    [enc setBytes:&qscale length:sizeof(qscale) atIndex:11];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)total + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_ane_dequant_iq2_xxs_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols) {
    if (!cb || !src || !dst || q_rows == 0 || q_cols == 0 || (q_cols % 256u) != 0) return 0;
    const uint32_t total = q_rows * (q_cols / 256u) * 16u;
    NSUInteger nth = g_ane_dequant_iq2_xxs_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_ane_dequant_iq2_xxs_f16_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)total + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_ane_dequant_q2_k_f16(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             q_rows,
        uint32_t             q_cols) {
    if (!cb || !src || !dst || q_rows == 0 || q_cols == 0 || (q_cols % 256u) != 0) return 0;
    const uint32_t total = q_rows * (q_cols / 256u) * 16u;
    NSUInteger nth = g_ane_dequant_q2_k_f16_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_ane_dequant_q2_k_f16_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)total + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_matmul(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer> a,
        NSUInteger a_off,
        id<MTLBuffer> b,
        id<MTLBuffer> c,
        NSUInteger c_off,
        uint32_t m,
        uint32_t n,
        uint32_t k) {
    if (!cb || !pipeline || !a || !b || !c || m == 0 || n == 0 || k == 0) return 0;
    MTLSize tg = MTLSizeMake((NSUInteger)pipeline.threadExecutionWidth * 4u, 1, 1);
    MTLSize grid = MTLSizeMake(((NSUInteger)n + 31u) / 32u,
                               ((NSUInteger)m + 63u) / 64u,
                               1);

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:a offset:a_off atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBuffer:c offset:c_off atIndex:2];
    [enc setBytes:&m length:sizeof(m) atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_swiglu_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer> gate,
        NSUInteger gate_off,
        id<MTLBuffer> up,
        NSUInteger up_off,
        id<MTLBuffer> mid,
        NSUInteger mid_off,
        id<MTLBuffer> weights,
        NSUInteger weights_off,
        uint32_t width,
        uint32_t rows,
        float clamp_value,
        float input_scale) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;
    NSUInteger nth = g_mpp_swiglu_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_swiglu_weight_f32_pipeline];
    [enc setBuffer:gate offset:gate_off atIndex:0];
    [enc setBuffer:up offset:up_off atIndex:1];
    [enc setBuffer:mid offset:mid_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc setBytes:&width length:sizeof(width) atIndex:4];
    [enc setBytes:&rows length:sizeof(rows) atIndex:5];
    [enc setBytes:&clamp_value length:sizeof(clamp_value) atIndex:6];
    [enc setBytes:&input_scale length:sizeof(input_scale) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_ane_output_pack_f16_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        id<MTLBuffer>        weights,
        NSUInteger           weights_off,
        uint32_t             width,
        uint32_t             rows) {
    if (!cb || !src || !dst || !weights || width == 0 || rows == 0 ||
        !g_ane_output_pack_f16_f32_pipeline) {
        return 0;
    }
    NSUInteger nth = g_ane_output_pack_f16_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_ane_output_pack_f16_f32_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBuffer:weights offset:weights_off atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&rows length:sizeof(rows) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_quant_f32_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             n,
        float                qscale) {
    if (!cb || !src || !dst || n == 0 || qscale <= 0.0f) return 0;
    NSUInteger nth = g_mpp_quant_f32_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_quant_f32_i8_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&qscale length:sizeof(qscale) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_quant_mid_f32_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        NSUInteger           src_off,
        id<MTLBuffer>        dst,
        uint32_t             width,
        uint32_t             rows,
        float                qscale) {
    if (!cb || !src || !dst || width == 0 || rows == 0 || qscale <= 0.0f) return 0;
    NSUInteger nth = g_mpp_quant_mid_f32_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_quant_mid_f32_i8_pipeline];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&rows length:sizeof(rows) atIndex:3];
    [enc setBytes:&qscale length:sizeof(qscale) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_swiglu_i32_weight(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer> gate,
        id<MTLBuffer> up,
        id<MTLBuffer> mid,
        NSUInteger mid_off,
        id<MTLBuffer> weights,
        NSUInteger weights_off,
        uint32_t width,
        uint32_t rows,
        float clamp_value,
        float input_scale) {
    if (!cb || !gate || !up || !mid || !weights || width == 0 || rows == 0) return 0;
    NSUInteger nth = g_mpp_swiglu_i32_weight_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_swiglu_i32_weight_f32_pipeline];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up offset:0 atIndex:1];
    [enc setBuffer:mid offset:mid_off atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc setBytes:&width length:sizeof(width) atIndex:4];
    [enc setBytes:&rows length:sizeof(rows) atIndex:5];
    [enc setBytes:&clamp_value length:sizeof(clamp_value) atIndex:6];
    [enc setBytes:&input_scale length:sizeof(input_scale) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_swiglu_i32_weight_i8(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer> gate,
        id<MTLBuffer> up,
        id<MTLBuffer> mid,
        id<MTLBuffer> weights,
        NSUInteger weights_off,
        uint32_t width,
        uint32_t rows,
        float clamp_value,
        float input_scale,
        float mid_qscale) {
    if (!cb || !gate || !up || !mid || !weights ||
        width == 0 || rows == 0 || mid_qscale <= 0.0f) {
        return 0;
    }
    NSUInteger nth = g_mpp_swiglu_i32_weight_i8_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth > width) nth = width;
    if (nth == 0) nth = 1u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_swiglu_i32_weight_i8_pipeline];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up offset:0 atIndex:1];
    [enc setBuffer:mid offset:0 atIndex:2];
    [enc setBuffer:weights offset:weights_off atIndex:3];
    [enc setBytes:&width length:sizeof(width) atIndex:4];
    [enc setBytes:&rows length:sizeof(rows) atIndex:5];
    [enc setBytes:&clamp_value length:sizeof(clamp_value) atIndex:6];
    [enc setBytes:&input_scale length:sizeof(input_scale) atIndex:7];
    [enc setBytes:&mid_qscale length:sizeof(mid_qscale) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_encode_mpp_scale_i32_f32(
        id<MTLCommandBuffer> cb,
        id<MTLBuffer>        src,
        id<MTLBuffer>        dst,
        NSUInteger           dst_off,
        uint32_t             n,
        float                scale) {
    if (!cb || !src || !dst || n == 0) return 0;
    NSUInteger nth = g_mpp_scale_i32_f32_pipeline.maxTotalThreadsPerThreadgroup;
    if (nth > 256u) nth = 256u;
    if (nth == 0) nth = 1u;
    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_mpp_scale_i32_f32_pipeline];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&scale length:sizeof(scale) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)n + nth - 1u) / nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

int ds4_gpu_routed_moe_expert_banked_batch_mpp_int8_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *gate_bank,
        ds4_gpu_tensor       *up_bank,
        ds4_gpu_tensor       *down_bank,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    (void)selected;
    if (mid_is_f16) *mid_is_f16 = false;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !gate_bank || !up_bank ||
        !down_bank || !weights || n_tokens == 0) {
        return 0;
    }
    if (gate_type != DS4_METAL_TENSOR_IQ2_XXS ||
        down_type != DS4_METAL_TENSOR_Q2_K ||
        expert_in_dim != 4096u ||
        expert_mid_dim != 2048u ||
        out_dim != 4096u) {
        return 0;
    }

    const uint64_t iq2_row_bytes = (uint64_t)(expert_in_dim / 256u) * 66u;
    const uint64_t q2_row_bytes = (uint64_t)(expert_mid_dim / 256u) * 84u;
    if (gate_row_bytes != iq2_row_bytes ||
        down_row_bytes != q2_row_bytes ||
        gate_expert_bytes != gate_row_bytes * expert_mid_dim ||
        down_expert_bytes != down_row_bytes * out_dim) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> gate_bankbuf = ds4_gpu_tensor_buffer(gate_bank);
        id<MTLBuffer> up_bankbuf = ds4_gpu_tensor_buffer(up_bank);
        id<MTLBuffer> down_bankbuf = ds4_gpu_tensor_buffer(down_bank);

        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t x_half_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(uint16_t);
        const uint64_t x_i8_bytes = (uint64_t)n_tokens * expert_in_dim;
        const uint64_t mid_bytes = (uint64_t)n_tokens * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        const uint64_t gate_i8_bytes = (uint64_t)expert_in_dim * expert_mid_dim;
        const uint64_t down_i8_bytes = (uint64_t)expert_mid_dim * out_dim;
        const uint64_t gate_i32_bytes = (uint64_t)n_tokens * expert_mid_dim * sizeof(int32_t);
        const uint64_t out_i32_bytes = (uint64_t)n_tokens * out_dim * sizeof(int32_t);
        const bool use_i8_i8 = getenv("DS4_FLASH_MOE_MPP_I8I8_PREFILL") != NULL ||
                               getenv("DS4_FLASH_MOE_MPP_INT8_ACT") != NULL;
        const char *i8_i8_fused_env = getenv("DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL");
        const bool use_i8_i8_fused = use_i8_i8 &&
                                     i8_i8_fused_env != NULL &&
                                     atoi(i8_i8_fused_env) != 0;
        float mpp_i8_qscale = 8.0f;
        const char *qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_QSCALE");
        if (!qscale_env || !qscale_env[0]) qscale_env = getenv("DS4_FLASH_MOE_ANE_INT8_QSCALE");
        if (qscale_env && qscale_env[0]) {
            char *end = NULL;
            float parsed = strtof(qscale_env, &end);
            if (end != qscale_env && parsed > 0.0f && isfinite(parsed)) {
                mpp_i8_qscale = parsed;
            }
        }
        const float mpp_i8_scale = 1.0f / mpp_i8_qscale;
        float mpp_x_qscale = 16.0f;
        const char *x_qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_X_QSCALE");
        if (!x_qscale_env || !x_qscale_env[0]) x_qscale_env = getenv("DS4_FLASH_MOE_ANE_X_QSCALE");
        if (x_qscale_env && x_qscale_env[0]) {
            char *end = NULL;
            float parsed = strtof(x_qscale_env, &end);
            if (end != x_qscale_env && parsed > 0.0f && isfinite(parsed)) {
                mpp_x_qscale = parsed;
            }
        }
        float mpp_mid_qscale = 16.0f;
        const char *mid_qscale_env = getenv("DS4_FLASH_MOE_MPP_INT8_MID_QSCALE");
        if (!mid_qscale_env || !mid_qscale_env[0]) mid_qscale_env = getenv("DS4_FLASH_MOE_ANE_MID_QSCALE");
        if (mid_qscale_env && mid_qscale_env[0]) {
            char *end = NULL;
            float parsed = strtof(mid_qscale_env, &end);
            if (end != mid_qscale_env && parsed > 0.0f && isfinite(parsed)) {
                mpp_mid_qscale = parsed;
            }
        }
        const float mpp_gate_scale = 1.0f / (mpp_x_qscale * mpp_i8_qscale);
        const float mpp_down_scale = 1.0f / (mpp_mid_qscale * mpp_i8_qscale);

        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !weightsbuf ||
            !gate_bankbuf || !up_bankbuf || !down_bankbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(weights) < (uint64_t)n_tokens * sizeof(float) ||
            ds4_gpu_tensor_bytes(gate_bank) < gate_expert_bytes ||
            ds4_gpu_tensor_bytes(up_bank) < gate_expert_bytes ||
            ds4_gpu_tensor_bytes(down_bank) < down_expert_bytes) {
            return 0;
        }
        if (x_half_bytes > NSUIntegerMax ||
            x_i8_bytes > NSUIntegerMax ||
            gate_i8_bytes > NSUIntegerMax ||
            down_i8_bytes > NSUIntegerMax ||
            gate_i32_bytes > NSUIntegerMax ||
            out_i32_bytes > NSUIntegerMax ||
            x_bytes / sizeof(float) > UINT32_MAX) {
            return 0;
        }
        if (!ds4_gpu_ensure_mpp_int8_prefill_pipelines()) return 0;
        int scratch_ok = ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_gate_i8_buffer,
                                                       &g_mpp_prefill_gate_i8_bytes,
                                                       (NSUInteger)gate_i8_bytes,
                                                       "ds4_mpp_prefill_gate_i8") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_up_i8_buffer,
                                                       &g_mpp_prefill_up_i8_bytes,
                                                       (NSUInteger)gate_i8_bytes,
                                                       "ds4_mpp_prefill_up_i8") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_down_i8_buffer,
                                                       &g_mpp_prefill_down_i8_bytes,
                                                       (NSUInteger)down_i8_bytes,
                                                       "ds4_mpp_prefill_down_i8");
        if (use_i8_i8) {
            scratch_ok = scratch_ok &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_x_i8_buffer,
                                                       &g_mpp_prefill_x_i8_bytes,
                                                       (NSUInteger)x_i8_bytes,
                                                       "ds4_mpp_prefill_x_i8") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_gate_i32_buffer,
                                                       &g_mpp_prefill_gate_i32_bytes,
                                                       (NSUInteger)gate_i32_bytes,
                                                       "ds4_mpp_prefill_gate_i32") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_up_i32_buffer,
                                                       &g_mpp_prefill_up_i32_bytes,
                                                       (NSUInteger)gate_i32_bytes,
                                                       "ds4_mpp_prefill_up_i32") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_mid_i8_buffer,
                                                       &g_mpp_prefill_mid_i8_bytes,
                                                       (NSUInteger)((uint64_t)n_tokens * expert_mid_dim),
                                                       "ds4_mpp_prefill_mid_i8") &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_out_i32_buffer,
                                                       &g_mpp_prefill_out_i32_bytes,
                                                       (NSUInteger)out_i32_bytes,
                                                       "ds4_mpp_prefill_out_i32");
        } else {
            scratch_ok = scratch_ok &&
                         ds4_gpu_ensure_scratch_buffer(&g_mpp_prefill_x_half_buffer,
                                             &g_mpp_prefill_x_half_bytes,
                                             (NSUInteger)x_half_bytes,
                                             "ds4_mpp_prefill_x_half");
        }
        if (!scratch_ok) {
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        const uint32_t x_elems = n_tokens * expert_in_dim;
        int ok = 0;
        if (use_i8_i8) {
            ok = ds4_gpu_encode_mpp_quant_f32_i8(cb,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 g_mpp_prefill_x_i8_buffer,
                                                 x_elems,
                                                 mpp_x_qscale) &&
                 ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                        gate_bankbuf,
                                                        ds4_gpu_tensor_offset(gate_bank),
                                                        g_mpp_prefill_gate_i8_buffer,
                                                        expert_mid_dim,
                                                        expert_in_dim,
                                                        mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                        up_bankbuf,
                                                        ds4_gpu_tensor_offset(up_bank),
                                                        g_mpp_prefill_up_i8_buffer,
                                                        expert_mid_dim,
                                                        expert_in_dim,
                                                        mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_dequant_q2_k_i8(cb,
                                                     down_bankbuf,
                                                     ds4_gpu_tensor_offset(down_bank),
                                                     g_mpp_prefill_down_i8_buffer,
                                                     out_dim,
                                                     expert_mid_dim,
                                                     mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_i8_i8_i32_pipeline,
                                           g_mpp_prefill_x_i8_buffer,
                                           0,
                                           g_mpp_prefill_gate_i8_buffer,
                                           g_mpp_prefill_gate_i32_buffer,
                                           0,
                                           n_tokens,
                                           expert_mid_dim,
                                           expert_in_dim) &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_i8_i8_i32_pipeline,
                                           g_mpp_prefill_x_i8_buffer,
                                           0,
                                           g_mpp_prefill_up_i8_buffer,
                                           g_mpp_prefill_up_i32_buffer,
                                           0,
                                           n_tokens,
                                           expert_mid_dim,
                                           expert_in_dim);
            if (ok) {
                if (use_i8_i8_fused) {
                    ok = ds4_gpu_encode_mpp_swiglu_i32_weight_i8(cb,
                                                                 g_mpp_prefill_gate_i32_buffer,
                                                                 g_mpp_prefill_up_i32_buffer,
                                                                 g_mpp_prefill_mid_i8_buffer,
                                                                 weightsbuf,
                                                                 ds4_gpu_tensor_offset(weights),
                                                                 expert_mid_dim,
                                                                 n_tokens,
                                                                 clamp,
                                                                 mpp_gate_scale,
                                                                 mpp_mid_qscale);
                } else {
                    ok = ds4_gpu_encode_mpp_swiglu_i32_weight(cb,
                                                              g_mpp_prefill_gate_i32_buffer,
                                                              g_mpp_prefill_up_i32_buffer,
                                                              midbuf,
                                                              ds4_gpu_tensor_offset(mid),
                                                              weightsbuf,
                                                              ds4_gpu_tensor_offset(weights),
                                                              expert_mid_dim,
                                                              n_tokens,
                                                              clamp,
                                                              mpp_gate_scale) &&
                         ds4_gpu_encode_mpp_quant_mid_f32_i8(cb,
                                                             midbuf,
                                                             ds4_gpu_tensor_offset(mid),
                                                             g_mpp_prefill_mid_i8_buffer,
                                                             expert_mid_dim,
                                                             n_tokens,
                                                             mpp_mid_qscale);
                }
            }
            ok = ok &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_i8_i8_i32_pipeline,
                                           g_mpp_prefill_mid_i8_buffer,
                                           0,
                                           g_mpp_prefill_down_i8_buffer,
                                           g_mpp_prefill_out_i32_buffer,
                                           0,
                                           n_tokens,
                                           out_dim,
                                           expert_mid_dim) &&
                 ds4_gpu_encode_mpp_scale_i32_f32(cb,
                                                  g_mpp_prefill_out_i32_buffer,
                                                  outbuf,
                                                  ds4_gpu_tensor_offset(out),
                                                  n_tokens * out_dim,
                                                  mpp_down_scale);
        } else {
            ok = ds4_gpu_encode_cpy_f32_f16_1d(cb,
                                                xbuf,
                                                ds4_gpu_tensor_offset(x),
                                                g_mpp_prefill_x_half_buffer,
                                                0,
                                                x_elems) &&
                 ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                        gate_bankbuf,
                                                        ds4_gpu_tensor_offset(gate_bank),
                                                        g_mpp_prefill_gate_i8_buffer,
                                                        expert_mid_dim,
                                                        expert_in_dim,
                                                        mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_dequant_iq2_xxs_i8(cb,
                                                        up_bankbuf,
                                                        ds4_gpu_tensor_offset(up_bank),
                                                        g_mpp_prefill_up_i8_buffer,
                                                        expert_mid_dim,
                                                        expert_in_dim,
                                                        mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_dequant_q2_k_i8(cb,
                                                     down_bankbuf,
                                                     ds4_gpu_tensor_offset(down_bank),
                                                     g_mpp_prefill_down_i8_buffer,
                                                     out_dim,
                                                     expert_mid_dim,
                                                     mpp_i8_qscale) &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_h_i8_f_pipeline,
                                           g_mpp_prefill_x_half_buffer,
                                           0,
                                           g_mpp_prefill_gate_i8_buffer,
                                           gatebuf,
                                           ds4_gpu_tensor_offset(gate),
                                           n_tokens,
                                           expert_mid_dim,
                                           expert_in_dim) &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_h_i8_f_pipeline,
                                           g_mpp_prefill_x_half_buffer,
                                           0,
                                           g_mpp_prefill_up_i8_buffer,
                                           upbuf,
                                           ds4_gpu_tensor_offset(up),
                                           n_tokens,
                                           expert_mid_dim,
                                           expert_in_dim) &&
                 ds4_gpu_encode_mpp_swiglu_weight(cb,
                                                   gatebuf,
                                                   ds4_gpu_tensor_offset(gate),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   weightsbuf,
                                                   ds4_gpu_tensor_offset(weights),
                                                   expert_mid_dim,
                                                   n_tokens,
                                                   clamp,
                                                   mpp_i8_scale) &&
                 ds4_gpu_encode_mpp_matmul(cb,
                                           g_mpp_f_i8_f_pipeline,
                                           midbuf,
                                           ds4_gpu_tensor_offset(mid),
                                           g_mpp_prefill_down_i8_buffer,
                                           outbuf,
                                           ds4_gpu_tensor_offset(out),
                                           n_tokens,
                                           out_dim,
                                           expert_mid_dim) &&
                 ds4_gpu_encode_scale_f32_rows(cb,
                                               outbuf,
                                               ds4_gpu_tensor_offset(out),
                                               outbuf,
                                               ds4_gpu_tensor_offset(out),
                                               out_dim,
                                               n_tokens,
                                               mpp_i8_scale);
        }
        if (!ok) return 0;
        return ds4_gpu_finish_command_buffer(cb, owned, "Flash-MoE MPP int8 prefill expert batch");
    }
}

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        bool                   *mid_is_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_tokens == 0 || n_expert == 0 || n_expert > 6) {
        return 0;
    }
    if ((expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> gatebuf = ds4_gpu_tensor_buffer(gate);
        id<MTLBuffer> upbuf = ds4_gpu_tensor_buffer(up);
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(mid);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> expertsbuf = ds4_gpu_tensor_buffer(experts);
        id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
        id<MTLBuffer> weightsbuf = ds4_gpu_tensor_buffer(weights);
        const uint64_t x_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(float);
        const uint64_t mid_bytes = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tokens * out_dim * sizeof(float);
        const uint64_t selected_bytes = (uint64_t)n_tokens * n_expert * sizeof(int);
        const uint64_t weights_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
        if (!xbuf || !gatebuf || !upbuf || !midbuf || !outbuf || !selectedbuf || !weightsbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(gate) < mid_bytes ||
            ds4_gpu_tensor_bytes(up) < mid_bytes ||
            ds4_gpu_tensor_bytes(mid) < mid_bytes ||
            ds4_gpu_tensor_bytes(out) < out_bytes ||
            ds4_gpu_tensor_bytes(selected) < selected_bytes ||
            ds4_gpu_tensor_bytes(weights) < weights_bytes) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized activation buffers\n");
            return 0;
        }
        if (n_expert > 1 &&
            (!expertsbuf ||
             ds4_gpu_tensor_bytes(experts) < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float))) {
            fprintf(stderr, "ds4: Metal routed batch MoE received undersized expert output buffer\n");
            return 0;
        }

        const uint64_t gate_tensor_bytes = 256ull * gate_expert_bytes;
        const uint64_t down_tensor_bytes = 256ull * down_expert_bytes;
        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gate_buf = ds4_gpu_wrap_model_range(model_map, model_size, gate_offset, gate_tensor_bytes, &gate_inner);
        id<MTLBuffer> up_buf = ds4_gpu_wrap_model_range(model_map, model_size, up_offset, gate_tensor_bytes, &up_inner);
        id<MTLBuffer> down_buf = ds4_gpu_wrap_model_range(model_map, model_size, down_offset, down_tensor_bytes, &down_inner);
        if (!gate_buf || !up_buf || !down_buf) return 0;

        const uint32_t pair_rows = n_tokens * n_expert;
        const uint64_t down_scratch_bytes = (uint64_t)pair_rows * out_dim * sizeof(float);
        if ((n_expert > 1 && !expertsbuf &&
             !ds4_gpu_ensure_scratch_buffer(&g_moe_down_scratch_buffer,
                                              &g_moe_down_scratch_bytes,
                                              (NSUInteger)down_scratch_bytes,
                                              "ds4_moe_down_scratch"))) {
            return 0;
        }

        const uint32_t gate_nr0 = ds4_gpu_routed_mv_nr0(gate_type);
        const uint32_t down_nr0 = ds4_gpu_routed_mv_nr0(down_type);
        id<MTLComputePipelineState> gate_mv_pipeline = ds4_gpu_routed_mv_pipeline(gate_type);
        id<MTLComputePipelineState> down_mv_pipeline = ds4_gpu_routed_mv_pipeline(down_type);
        id<MTLComputePipelineState> gate_mm_pipeline = nil;
        id<MTLComputePipelineState> down_mm_pipeline = nil;
        if (gate_nr0 == 0 || down_nr0 == 0 || !gate_mv_pipeline || !down_mv_pipeline) {
            fprintf(stderr, "ds4: unsupported Metal routed batch MoE quant types gate=%u down=%u\n",
                    gate_type, down_type);
            return 0;
        }

        ds4_gpu_mul_mv_id_args gate_args =
            ds4_gpu_make_mul_mv_id_args(expert_in_dim, expert_mid_dim, 256,
                                          gate_row_bytes, gate_expert_bytes,
                                          1, n_expert, n_tokens, gate_nr0);
        ds4_gpu_mul_mv_id_args down_args =
            ds4_gpu_make_mul_mv_id_args(expert_mid_dim, out_dim, 256,
                                          down_row_bytes, down_expert_bytes,
                                          n_expert, n_expert, n_tokens, down_nr0);
        const bool use_mm_id = n_tokens >= 32u && ds4_gpu_mul_mm_id_map0_name(n_expert) != NULL;
        /*
         * MTP verification is neither normal decode nor large prefill: the
         * target model must verify a tiny suffix (usually 2 tokens) in one
         * layer-major pass.  For that shape the prefill expert-major GEMM path
         * is too large, but the decode pair kernels are exactly the right
         * primitive: they read the same activation once and compute routed
         * gate/up together for every selected expert row.  Keep this limited to
         * tiny batches so ordinary prefill keeps using the higher-throughput
         * grouped matmul path.
         */
        const bool use_tiny_pair_mv =
            !g_quality_mode &&
            n_tokens <= 4u &&
            !use_mm_id &&
            ((gate_type == DS4_METAL_TENSOR_IQ2_XXS && g_moe_mul_mv_id_iq2_xxs_pair_pipeline) ||
             (gate_type == DS4_METAL_TENSOR_Q4_K && g_moe_mul_mv_id_q4_k_pair_pipeline));
        ds4_gpu_mul_mm_id_map_args gate_map_args = { 0 };
        ds4_gpu_mul_mm_id_args gate_mm_args = { 0 };
        ds4_gpu_mul_mm_id_args down_mm_args = { 0 };
        id<MTLComputePipelineState> map_pipeline = nil;
        /*
         * The grouped routed-MoE matmul loads activation tiles as half before
         * using SIMD-group MMA.  Store the SwiGLU/route-weight intermediate in
         * that same precision so the down projection avoids a large F32 mid
         * write/read.  --quality or DS4_METAL_MOE_MID_F32 keeps the older F32
         * intermediate for isolated diagnostics.
         */
        const bool request_mid_f16 =
            !g_quality_mode && getenv("DS4_METAL_MOE_MID_F32") == NULL;
        if (use_mm_id) {
            gate_map_args =
                ds4_gpu_make_mul_mm_id_map_args(expert_in_dim, 256, 1, n_expert, n_tokens);
            gate_mm_args =
                ds4_gpu_make_mul_mm_id_args(expert_in_dim, expert_mid_dim, 256,
                                              gate_row_bytes, gate_expert_bytes,
                                              1, n_expert, n_tokens);
            down_mm_args =
                ds4_gpu_make_mul_mm_id_args_src1_size(expert_mid_dim, out_dim, 256,
                                                        down_row_bytes, down_expert_bytes,
                                                        n_expert, n_expert, n_tokens,
                                                        request_mid_f16 ? sizeof(uint16_t) : sizeof(float));

            map_pipeline = ds4_gpu_get_pipeline(ds4_gpu_mul_mm_id_map0_name(n_expert));
            gate_mm_pipeline = ds4_gpu_routed_mm_pipeline(gate_type);
            down_mm_pipeline = request_mid_f16 ?
                ds4_gpu_routed_mm_f16_rhs_pipeline(down_type) :
                ds4_gpu_routed_mm_pipeline(down_type);
            if (!map_pipeline || !gate_mm_pipeline || !down_mm_pipeline) {
                return 0;
            }
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        const bool moe_stage_profile =
            getenv("DS4_METAL_MOE_STAGE_PROFILE") != NULL && g_batch_cb != nil;
        double moe_stage_t0 = moe_stage_profile ? ds4_gpu_now_ms() : 0.0;
        if (moe_stage_profile) {
            if (ds4_gpu_end_commands() == 0 || ds4_gpu_begin_commands() == 0) {
                return 0;
            }
            cb = ds4_gpu_command_buffer(&owned);
            if (!cb) return 0;
            moe_stage_t0 = ds4_gpu_now_ms();
        }
#define DS4_METAL_PROFILE_MOE_STAGE(name) do { \
            if (ok && moe_stage_profile) { \
                if (ds4_gpu_end_commands() == 0) { \
                    ok = 0; \
                } else { \
                    const double now_ms = ds4_gpu_now_ms(); \
                    fprintf(stderr, \
                            "ds4: Metal routed MoE stage tokens=%u pairs=%u %s=%.3f ms\n", \
                            n_tokens, pair_rows, (name), now_ms - moe_stage_t0); \
                    moe_stage_t0 = now_ms; \
                    if (ds4_gpu_begin_commands() == 0) { \
                        ok = 0; \
                    } else { \
                        cb = ds4_gpu_command_buffer(&owned); \
                        if (!cb) ok = 0; \
                    } \
                } \
            } \
        } while (0)

        const NSUInteger gate_smem = ds4_gpu_routed_mv_smem(gate_type);
        const NSUInteger down_smem = ds4_gpu_routed_mv_smem(down_type);
        id<MTLComputePipelineState> down_sum6_pipeline = nil;
        if (down_type == DS4_METAL_TENSOR_Q2_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q2_k_sum6_pipeline;
        } else if (down_type == DS4_METAL_TENSOR_Q4_K) {
            down_sum6_pipeline = g_moe_mul_mv_id_q4_k_sum6_pipeline;
        }
        const bool direct_down_sum =
            !g_quality_mode &&
            !use_mm_id &&
            n_expert == 6 &&
            n_tokens <= 4u &&
            down_sum6_pipeline != nil;
        int ok = 0;
        if (use_mm_id) {
            /*
             * The routed pair ids are the same for gate, up, and down. Build
             * the expert-major work map once, then reuse it for all three
             * batched expert matmuls.
             */
            ok = ds4_gpu_encode_mul_mm_id_map(cb,
                                                map_pipeline,
                                                &gate_map_args,
                                                &gate_mm_args,
                                                selectedbuf,
                                                ds4_gpu_tensor_offset(selected));
            DS4_METAL_PROFILE_MOE_STAGE("map");
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                   gate_mm_pipeline,
                                                   &gate_mm_args,
                                                   gate_buf,
                                                   (NSUInteger)gate_inner,
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   gatebuf,
                                                   ds4_gpu_tensor_offset(gate));
                DS4_METAL_PROFILE_MOE_STAGE("gate");
            }
            if (ok) {
                ok = ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                   gate_mm_pipeline,
                                                   &gate_mm_args,
                                                   up_buf,
                                                   (NSUInteger)up_inner,
                                                   xbuf,
                                                   ds4_gpu_tensor_offset(x),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up));
                DS4_METAL_PROFILE_MOE_STAGE("up");
            }
        } else if (use_tiny_pair_mv) {
            id<MTLComputePipelineState> pair_pipeline =
                gate_type == DS4_METAL_TENSOR_IQ2_XXS ?
                    g_moe_mul_mv_id_iq2_xxs_pair_pipeline :
                    g_moe_mul_mv_id_q4_k_pair_pipeline;
            ok = ds4_gpu_encode_mul_mv_id_pair(cb,
                                                 pair_pipeline,
                                                 &gate_args,
                                                 gate_buf,
                                                 (NSUInteger)gate_inner,
                                                 up_buf,
                                                 (NSUInteger)up_inner,
                                                 xbuf,
                                                 ds4_gpu_tensor_offset(x),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 selectedbuf,
                                                 ds4_gpu_tensor_offset(selected),
                                                 gate_smem,
                                                 2,
                                                 false);
        } else {
            ok = ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  gate_buf,
                                                  (NSUInteger)gate_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  gatebuf,
                                                  ds4_gpu_tensor_offset(gate),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false) &&
                 ds4_gpu_encode_mul_mv_id(cb,
                                                  gate_mv_pipeline,
                                                  &gate_args,
                                                  up_buf,
                                                  (NSUInteger)up_inner,
                                                  xbuf,
                                                  ds4_gpu_tensor_offset(x),
                                                  upbuf,
                                                  ds4_gpu_tensor_offset(up),
                                                  selectedbuf,
                                                  ds4_gpu_tensor_offset(selected),
                                                  gate_smem,
                                                  2,
                                                  false);
        }
        DS4_METAL_PROFILE_MOE_STAGE("gate_up");
        const bool use_fused_activation = !g_quality_mode;
        const bool use_mid_f16 =
            use_mm_id &&
            use_fused_activation &&
            request_mid_f16;
        if (mid_is_f16) *mid_is_f16 = use_mid_f16;
        if (ok && use_fused_activation) {
            ok = ds4_gpu_encode_moe_swiglu_weight(cb,
                                                    gatebuf,
                                                    ds4_gpu_tensor_offset(gate),
                                                    upbuf,
                                                    ds4_gpu_tensor_offset(up),
                                                    midbuf,
                                                    ds4_gpu_tensor_offset(mid),
                                                    weightsbuf,
                                                    ds4_gpu_tensor_offset(weights),
                                                    expert_mid_dim,
                                                    pair_rows,
                                                    clamp,
                                                    use_mid_f16);
        } else if (ok && clamp > 1.0e-6f) {
            ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 gatebuf,
                                                 ds4_gpu_tensor_offset(gate),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -FLT_MAX,
                                                 clamp);
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                     g_unary_silu_pipeline,
                                                     gatebuf,
                                                     ds4_gpu_tensor_offset(gate),
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     expert_mid_dim,
                                                     pair_rows,
                                                     1,
                                                     0.0f,
                                                     0.0f);
            }
            if (ok) {
                ok = ds4_gpu_encode_unary_f32_rows(cb,
                                                 g_unary_clamp_pipeline,
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 upbuf,
                                                 ds4_gpu_tensor_offset(up),
                                                 expert_mid_dim,
                                                 pair_rows,
                                                 0,
                                                 -clamp,
                                                 clamp);
            }
            if (ok) {
                ds4_gpu_bin_args mul_args =
                    ds4_gpu_make_bin_same_rows_args(expert_mid_dim, pair_rows);
                ok = ds4_gpu_encode_bin_f32_rows(cb,
                                                   g_mul_pipeline,
                                                   &mul_args,
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid),
                                                   upbuf,
                                                   ds4_gpu_tensor_offset(up),
                                                   midbuf,
                                                   ds4_gpu_tensor_offset(mid));
            }
        } else if (ok) {
            ok = ds4_gpu_encode_swiglu_flat(cb,
                                              gatebuf,
                                              ds4_gpu_tensor_offset(gate),
                                              upbuf,
                                              ds4_gpu_tensor_offset(up),
                                              midbuf,
                                              ds4_gpu_tensor_offset(mid),
                                              (uint32_t)((uint64_t)pair_rows * expert_mid_dim));
        }
        if (ok && !use_fused_activation) {
            ds4_gpu_bin_args weight_args =
                ds4_gpu_make_bin_rowwise_scalar_args(expert_mid_dim, pair_rows);
            ok = ds4_gpu_encode_bin_f32_rows(cb,
                                               g_bin_mul_scalar_pipeline,
                                               &weight_args,
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid),
                                               weightsbuf,
                                               ds4_gpu_tensor_offset(weights),
                                               midbuf,
                                               ds4_gpu_tensor_offset(mid));
        }
        DS4_METAL_PROFILE_MOE_STAGE("activation_weight");

        id<MTLBuffer> down_dst = n_expert == 1 ? outbuf : (expertsbuf ? expertsbuf : g_moe_down_scratch_buffer);
        NSUInteger down_dst_off = n_expert == 1 ? ds4_gpu_tensor_offset(out) :
            (expertsbuf ? ds4_gpu_tensor_offset(experts) : 0);
        if (ok) {
            if (direct_down_sum) {
                ok = ds4_gpu_encode_mul_mv_id_sum6(cb,
                                                     down_sum6_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     outbuf,
                                                     ds4_gpu_tensor_offset(out),
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2);
            } else if (use_mm_id) {
                ok = ds4_gpu_encode_mul_mm_id_mapped(cb,
                                                       down_mm_pipeline,
                                                       &down_mm_args,
                                                       down_buf,
                                                       (NSUInteger)down_inner,
                                                       midbuf,
                                                       ds4_gpu_tensor_offset(mid),
                                                       down_dst,
                                                       down_dst_off);
            } else {
                ok = ds4_gpu_encode_mul_mv_id(cb,
                                                     down_mv_pipeline,
                                                     &down_args,
                                                     down_buf,
                                                     (NSUInteger)down_inner,
                                                     midbuf,
                                                     ds4_gpu_tensor_offset(mid),
                                                     down_dst,
                                                     down_dst_off,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     down_smem,
                                                     2,
                                                     false);
            }
        }
        DS4_METAL_PROFILE_MOE_STAGE("down");
        if (ok && n_expert > 1 && !direct_down_sum) {
            ok = ds4_gpu_encode_moe_sum_experts(cb,
                                                       down_dst,
                                                       down_dst_off,
                                                       outbuf,
                                                       ds4_gpu_tensor_offset(out),
                                                       out_dim,
                                                       n_expert,
                                                       n_tokens);
        }
        DS4_METAL_PROFILE_MOE_STAGE("sum");
        if (!ok) return 0;

        if (!ds4_gpu_finish_command_buffer(cb, owned, "routed batch MoE")) return 0;
#undef DS4_METAL_PROFILE_MOE_STAGE
    }

    return 1;
}

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_hc == 0 || n_hc > 16) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t mix_tensor_bytes = ds4_gpu_tensor_bytes(mix);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (!mixbuf || !outbuf ||
            mix_tensor_bytes < mix_bytes ||
            out_tensor_bytes < mix_bytes) {
            fprintf(stderr, "ds4: Metal HC split received undersized activation buffers\n");
            return 0;
        }
        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal HC split parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        uint64_t n_rows64 = mix_tensor_bytes / mix_bytes;
        const uint64_t out_rows64 = out_tensor_bytes / mix_bytes;
        if (out_rows64 < n_rows64) n_rows64 = out_rows64;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC split row count is outside supported range\n");
            return 0;
        }

        ds4_gpu_hc_split_args args = {
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb01 = mix_bytes,
            .nb1 = mix_bytes,
            .eps = eps,
        };
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_rows64));
        const NSUInteger n_tg = ((NSUInteger)n_rows64 + nth - 1u) / nth;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_sinkhorn_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sinkhorn")) return 0;
    }

    return 1;
}

static int ds4_gpu_hc_weighted_sum_strided(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint64_t                weight_offset,
        uint64_t                weight_row_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0 ||
        weight_row_stride < (uint64_t)n_hc * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> wbuf = ds4_gpu_tensor_buffer(weights);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes || out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC weighted sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / out_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC weighted sum token count is outside supported range\n");
            return 0;
        }

        const uint64_t x_row_values = (uint64_t)n_hc * n_embd;
        if (x_row_values == 0 ||
            x_row_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (x_row_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / ((uint64_t)n_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC weighted sum activation size overflow\n");
            return 0;
        }

        const uint64_t x_bytes = n_tokens64 * x_row_values * sizeof(float);
        const uint64_t w_last = weight_offset +
                                (n_tokens64 - 1u) * weight_row_stride +
                                (uint64_t)n_hc * sizeof(float);
        if (!xbuf || !wbuf || !outbuf ||
            ds4_gpu_tensor_bytes(residual_hc) < x_bytes ||
            ds4_gpu_tensor_bytes(weights) < w_last) {
            fprintf(stderr, "ds4: Metal HC weighted sum received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_weighted_sum_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_w0 = sizeof(float),
            .nb_w1 = weight_row_stride,
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
        };
        const uint64_t n_elem = (uint64_t)n_embd * n_tokens64;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:1];
        [enc setBuffer:wbuf offset:ds4_gpu_tensor_offset(weights) + (NSUInteger)weight_offset atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, label)) return 0;
    }

    return 1;
}

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             weights,
                                             0,
                                             (uint64_t)n_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum");
}

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return ds4_gpu_hc_weighted_sum_strided(out,
                                             residual_hc,
                                             split,
                                             0,
                                             mix_hc * sizeof(float),
                                             n_embd,
                                             n_hc,
                                             "HC weighted sum split");
}

/* Release decode fused HC pre-sublayer operation.  The graph driver owns the
 * optional reference fallback so this function stays a direct fused dispatch. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc == 0) {
        return 0;
    }
    if (n_hc != 4) {
        fprintf(stderr, "ds4: Metal fused HC split/sum is specialized for HC=4\n");
        return 0;
    }

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        if (!scalebuf || !basebuf) return 0;

        ds4_gpu_hc_split_weighted_sum_args args = {
            .n_embd = (int64_t)n_embd,
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb_mix1 = mix_bytes,
            .nb_split1 = mix_bytes,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = residual_row_bytes,
            .nb0 = sizeof(float),
            .nb1 = out_row_bytes,
            .eps = eps,
        };

        NSUInteger nth = g_hc_split_weighted_sum_pipeline.maxTotalThreadsPerThreadgroup;
        if (nth > 256u) nth = 256u;
        if (nth > (NSUInteger)n_embd) nth = (NSUInteger)n_embd;
        if (nth == 0) nth = 1u;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:g_hc_split_weighted_sum_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_hc * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum fused")) return 0;
    }

    return 1;
}

/* Decode-only HC-pre plus the immediately following weighted RMSNorm.  This is
 * intentionally specialized for DS4's fixed HC=4, embd=4096 shape; larger
 * batched prefill keeps using the existing two-stage path. */
int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd != 4096 || n_hc != 4) {
        return 0;
    }

    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t scale_bytes = 3ull * sizeof(float);

    @autoreleasepool {
        id<MTLBuffer> mixbuf = ds4_gpu_tensor_buffer(mix);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        id<MTLBuffer> normbuf = ds4_gpu_tensor_buffer(norm_out);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
            out_tensor_bytes % out_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm output size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_rows64 = out_tensor_bytes / out_row_bytes;
        if (n_rows64 == 0 || n_rows64 > UINT32_MAX ||
            n_rows64 > UINT64_MAX / mix_bytes ||
            n_rows64 > UINT64_MAX / residual_row_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm row count is outside supported range\n");
            return 0;
        }

        const uint64_t mix_total_bytes = n_rows64 * mix_bytes;
        const uint64_t residual_total_bytes = n_rows64 * residual_row_bytes;
        const uint64_t out_total_bytes = n_rows64 * out_row_bytes;
        if (!mixbuf || !splitbuf || !xbuf || !outbuf || !normbuf ||
            ds4_gpu_tensor_bytes(mix) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(split) < mix_total_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < residual_total_bytes ||
            ds4_gpu_tensor_bytes(norm_out) < out_total_bytes) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm received undersized activation buffers\n");
            return 0;
        }

        if (scale_offset > model_size || scale_bytes > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size || out_row_bytes > model_size - norm_weight_offset) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm parameter range is outside the mapped model\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        uint64_t norm_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size, scale_offset, scale_bytes, &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size, base_offset, mix_bytes, &base_inner);
        id<MTLBuffer> normwbuf = ds4_gpu_wrap_model_range(model_map, model_size, norm_weight_offset, out_row_bytes, &norm_inner);
        if (!scalebuf || !basebuf || !normwbuf) return 0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_hot_pipeline(g_hc_split_weighted_sum_norm_pipeline,
                                   "kernel_dsv4_hc_split_weighted_sum_norm4");
        if (!pipeline) return 0;

        ds4_gpu_hc_split_weighted_sum_norm_args args = {
            .n_embd = (int64_t)n_embd,
            .n_hc = (int32_t)n_hc,
            .sinkhorn_iters = (int32_t)sinkhorn_iters,
            .n_rows = (int64_t)n_rows64,
            .mix_hc = (int64_t)mix_hc,
            .nb_mix1 = mix_bytes,
            .nb_split1 = mix_bytes,
            .nb_x0 = sizeof(float),
            .nb_x1 = (uint64_t)n_embd * sizeof(float),
            .nb_x2 = residual_row_bytes,
            .nb0 = sizeof(float),
            .nb1 = out_row_bytes,
            .nb_norm1 = out_row_bytes,
            .eps = eps,
            .norm_eps = norm_eps,
        };

        NSUInteger nth = ds4_gpu_rms_norm_threads(n_embd);
        if (nth > pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: Metal fused HC split/sum/norm requires %lu threads but pipeline supports %lu\n",
                    (unsigned long)nth,
                    (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
            return 0;
        }

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:mixbuf offset:ds4_gpu_tensor_offset(mix) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:3];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:4];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:6];
        [enc setBuffer:normwbuf offset:(NSUInteger)norm_inner atIndex:7];
        [enc setBuffer:normbuf offset:ds4_gpu_tensor_offset(norm_out) atIndex:8];
        [enc setThreadgroupMemoryLength:((NSUInteger)n_embd + 4u + 32u) * sizeof(float)
                                atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_rows64, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC split/sum/norm fused")) return 0;
    }

    return 1;
}

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !pre || !model_map || n_hc == 0) return 0;

    @autoreleasepool {
        if ((n_hc % 4u) != 0) {
            fprintf(stderr, "ds4: Metal output HC weights requires a multiple-of-4 HC width\n");
            return 0;
        }

        id<MTLBuffer> prebuf = ds4_gpu_tensor_buffer(pre);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
        if (row_bytes == 0 || out_tensor_bytes < row_bytes || out_tensor_bytes % row_bytes != 0) {
            fprintf(stderr, "ds4: Metal output HC weights size is not a whole token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX ||
            n_tokens64 > UINT64_MAX / row_bytes) {
            fprintf(stderr, "ds4: Metal output HC weights token count is outside supported range\n");
            return 0;
        }

        const uint64_t bytes = n_tokens64 * row_bytes;
        if (!prebuf || !outbuf ||
            ds4_gpu_tensor_bytes(pre) < bytes ||
            ds4_gpu_tensor_bytes(out) < bytes) {
            fprintf(stderr, "ds4: Metal output HC weights received undersized buffers\n");
            return 0;
        }

        uint64_t scale_inner = 0;
        uint64_t base_inner = 0;
        id<MTLBuffer> scalebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                            scale_offset, sizeof(float),
                                                            &scale_inner);
        id<MTLBuffer> basebuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                           base_offset, row_bytes,
                                                           &base_inner);
        if (!scalebuf || !basebuf) return 0;

        const uint32_t n_tokens = (uint32_t)n_tokens64;
        ds4_gpu_bin_args mul_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, 1);
        ds4_gpu_bin_args add_args = ds4_gpu_make_bin_rows_args(n_hc, n_tokens, n_hc);
        ds4_gpu_unary_args sigmoid_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 0.0f, 0.0f);
        ds4_gpu_unary_args scale_args = ds4_gpu_make_unary_rows_args(n_hc, n_tokens, 1, 1.0f, eps);

        NSUInteger mul_nth_max = g_bin_mul_scalar_pipeline.maxTotalThreadsPerThreadgroup;
        if (mul_nth_max > 256u) mul_nth_max = 256u;
        NSUInteger mul_nth = 1u;
        while (2u * mul_nth < (NSUInteger)mul_args.ne0 && mul_nth < mul_nth_max) {
            mul_nth *= 2u;
        }

        NSUInteger add_nth_max = g_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (add_nth_max > 256u) add_nth_max = 256u;
        NSUInteger add_nth = 1u;
        while (2u * add_nth < (NSUInteger)add_args.ne0 && add_nth < add_nth_max) {
            add_nth *= 2u;
        }

        NSUInteger unary_nth_max = g_unary_sigmoid_pipeline.maxTotalThreadsPerThreadgroup;
        if (unary_nth_max > 256u) unary_nth_max = 256u;
        NSUInteger unary_nth = (NSUInteger)sigmoid_args.ne00;
        if (unary_nth > unary_nth_max) unary_nth = unary_nth_max;
        if (unary_nth == 0) unary_nth = 1u;
        const NSUInteger unary_nk0 = ((NSUInteger)sigmoid_args.ne00 + unary_nth - 1u) / unary_nth;
        const NSUInteger out_offset = ds4_gpu_tensor_offset(out);

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);

        [enc setComputePipelineState:g_bin_mul_scalar_pipeline];
        [enc setBytes:&mul_args length:sizeof(mul_args) atIndex:0];
        [enc setBuffer:prebuf offset:ds4_gpu_tensor_offset(pre) atIndex:1];
        [enc setBuffer:scalebuf offset:(NSUInteger)scale_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)mul_args.ne01,
                                              (NSUInteger)mul_args.ne02,
                                              (NSUInteger)mul_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(mul_nth, 1, 1)];

        [enc setComputePipelineState:g_add_pipeline];
        [enc setBytes:&add_args length:sizeof(add_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:basebuf offset:(NSUInteger)base_inner atIndex:2];
        [enc setBuffer:outbuf offset:out_offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)add_args.ne01,
                                              (NSUInteger)add_args.ne02,
                                              (NSUInteger)add_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(add_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_sigmoid_pipeline];
        [enc setBytes:&sigmoid_args length:sizeof(sigmoid_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)sigmoid_args.ne01,
                                              (NSUInteger)sigmoid_args.ne02,
                                              (NSUInteger)sigmoid_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];

        [enc setComputePipelineState:g_unary_scale_pipeline];
        [enc setBytes:&scale_args length:sizeof(scale_args) atIndex:0];
        [enc setBuffer:outbuf offset:out_offset atIndex:1];
        [enc setBuffer:outbuf offset:out_offset atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(unary_nk0 * (NSUInteger)scale_args.ne01,
                                              (NSUInteger)scale_args.ne02,
                                              (NSUInteger)scale_args.ne03)
             threadsPerThreadgroup:MTLSizeMake(unary_nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "output HC weights")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> postbuf = ds4_gpu_tensor_buffer(post);
        id<MTLBuffer> combbuf = ds4_gpu_tensor_buffer(comb);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t comb_values = (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            comb_values > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (comb_values * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t post_bytes = n_tokens64 * (uint64_t)n_hc * sizeof(float);
        const uint64_t comb_bytes = n_tokens64 * comb_values * sizeof(float);
        if (!blockbuf || !resbuf || !postbuf || !combbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(post) < post_bytes ||
            ds4_gpu_tensor_bytes(comb) < comb_bytes) {
            fprintf(stderr, "ds4: Metal HC expand received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = (uint64_t)n_hc * n_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:postbuf offset:ds4_gpu_tensor_offset(post) atIndex:3];
        [enc setBuffer:combbuf offset:ds4_gpu_tensor_offset(comb) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand split received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand split")) return 0;
    }

    return 1;
}

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;

    @autoreleasepool {
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> addbuf = ds4_gpu_tensor_buffer(block_add);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);
        const uint64_t hc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
        if (hc_row_bytes == 0 || out_tensor_bytes < hc_row_bytes || out_tensor_bytes % hc_row_bytes != 0) {
            fprintf(stderr, "ds4: Metal HC expand add split output size is not a whole HC token row\n");
            return 0;
        }

        const uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
        if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
            fprintf(stderr, "ds4: Metal HC expand add split token count is outside supported range\n");
            return 0;
        }

        const uint64_t block_values = (uint64_t)n_embd;
        const uint64_t hc_values = (uint64_t)n_hc * n_embd;
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        if (hc_values == 0 ||
            hc_values > UINT64_MAX / sizeof(float) ||
            mix_hc > UINT64_MAX / sizeof(float) ||
            n_tokens64 > UINT64_MAX / (block_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (hc_values * sizeof(float)) ||
            n_tokens64 > UINT64_MAX / (mix_hc * sizeof(float))) {
            fprintf(stderr, "ds4: Metal HC expand add split activation size overflow\n");
            return 0;
        }

        const uint64_t block_bytes = n_tokens64 * block_values * sizeof(float);
        const uint64_t hc_bytes = n_tokens64 * hc_values * sizeof(float);
        const uint64_t split_bytes = n_tokens64 * mix_hc * sizeof(float);
        if (!blockbuf || !addbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(block_out) < block_bytes ||
            ds4_gpu_tensor_bytes(block_add) < block_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes) {
            fprintf(stderr, "ds4: Metal HC expand add split received undersized activation buffers\n");
            return 0;
        }

        ds4_gpu_hc_expand_args args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = (int64_t)n_tokens64,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
        };
        id<MTLComputePipelineState> expand_pipeline = g_hc_expand_pipeline;
        uint64_t n_elem = (uint64_t)n_embd * n_hc * n_tokens64;
        if (n_hc == 4) {
            expand_pipeline = ds4_gpu_hot_pipeline(g_dsv4_hc_expand4_pipeline,
                                                      "kernel_dsv4_hc_expand4");
            n_elem = (uint64_t)n_embd * n_tokens64;
        }
        if (!expand_pipeline) return 0;
        const NSUInteger nth = MIN((NSUInteger)256, MAX((NSUInteger)1, (NSUInteger)n_elem));
        const NSUInteger n_tg = ((NSUInteger)n_elem + nth - 1u) / nth;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:expand_pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:1];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:2];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:3];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:4];
        [enc setBuffer:addbuf offset:ds4_gpu_tensor_offset(block_add) atIndex:5];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "HC expand add split")) return 0;
    }

    return 1;
}

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0 ||
        n_hc != 4 || out_dim != n_embd || (in_dim & 31u) != 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> midbuf = ds4_gpu_tensor_buffer(shared_mid);
        id<MTLBuffer> sharedbuf = ds4_gpu_tensor_buffer(shared_out);
        id<MTLBuffer> routedbuf = ds4_gpu_tensor_buffer(routed_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t shared_mid_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!midbuf || !sharedbuf || !routedbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(shared_mid) < shared_mid_bytes ||
            ds4_gpu_tensor_bytes(shared_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(routed_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal shared-down HC fusion received undersized buffers\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 1,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_shared_down_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:midbuf offset:ds4_gpu_tensor_offset(shared_mid) atIndex:3];
        [enc setBuffer:sharedbuf offset:ds4_gpu_tensor_offset(shared_out) atIndex:4];
        [enc setBuffer:routedbuf offset:ds4_gpu_tensor_offset(routed_out) atIndex:5];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:7];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:8];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:9];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "shared-down HC expand fused")) return 0;
    }

    return 1;
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0 || n_hc != 4 || out_dim != n_embd ||
        (in_dim & 31u) != 0 || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> blockbuf = ds4_gpu_tensor_buffer(block_out);
        id<MTLBuffer> resbuf = ds4_gpu_tensor_buffer(residual_hc);
        id<MTLBuffer> splitbuf = ds4_gpu_tensor_buffer(split);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out_hc);

        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t weight_bytes = out_dim * row_bytes;
        const uint64_t x_bytes = in_dim * sizeof(float);
        const uint64_t embd_bytes = out_dim * sizeof(float);
        const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t split_bytes = mix_hc * sizeof(float);

        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion weight range is outside the mapped model\n");
            return 0;
        }
        if (!xbuf || !blockbuf || !resbuf || !splitbuf || !outbuf ||
            ds4_gpu_tensor_bytes(x) < x_bytes ||
            ds4_gpu_tensor_bytes(block_out) < embd_bytes ||
            ds4_gpu_tensor_bytes(residual_hc) < hc_bytes ||
            ds4_gpu_tensor_bytes(split) < split_bytes ||
            ds4_gpu_tensor_bytes(out_hc) < hc_bytes) {
            fprintf(stderr, "ds4: Metal Q8 HC fusion received undersized buffers\n");
            return 0;
        }

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size,
                                                        weight_offset, weight_bytes,
                                                        &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_q8_0_matvec_args mv_args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        ds4_gpu_mv_dispatch mv_dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        mv_args.nr0 = mv_dispatch.nr0;

        ds4_gpu_hc_expand_args hc_args = {
            .n_embd = n_embd,
            .n_hc = n_hc,
            .n_tokens = 1,
            .nb_block0 = sizeof(float),
            .nb_block1 = (uint64_t)n_embd * sizeof(float),
            .nb_add0 = sizeof(float),
            .nb_add1 = (uint64_t)n_embd * sizeof(float),
            .nb_res0 = sizeof(float),
            .nb_res1 = (uint64_t)n_embd * sizeof(float),
            .nb_res2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .nb_post0 = sizeof(float),
            .nb_post1 = mix_hc * sizeof(float),
            .nb_comb0 = sizeof(float),
            .nb_comb1 = (uint64_t)n_hc * sizeof(float),
            .nb_comb2 = mix_hc * sizeof(float),
            .nb0 = sizeof(float),
            .nb1 = (uint64_t)n_embd * sizeof(float),
            .nb2 = (uint64_t)n_hc * n_embd * sizeof(float),
            .has_add = 0,
        };

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_dsv4_q8_hc_expand4_q8_0",
                                          mv_dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&mv_args length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&hc_args length:sizeof(hc_args) atIndex:1];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:2];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:3];
        [enc setBuffer:blockbuf offset:ds4_gpu_tensor_offset(block_out) atIndex:4];
        [enc setBuffer:resbuf offset:ds4_gpu_tensor_offset(residual_hc) atIndex:5];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)n_hc * sizeof(float) atIndex:6];
        [enc setBuffer:splitbuf offset:ds4_gpu_tensor_offset(split) + (NSUInteger)(2u * n_hc) * sizeof(float) atIndex:7];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out_hc) atIndex:8];
        [enc setThreadgroupMemoryLength:mv_dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)mv_dispatch.nr0 - 1u) /
                                              (NSUInteger)mv_dispatch.nr0,
                                              1,
                                              1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)mv_dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Q8 HC expand fused")) return 0;
    }

    return 1;
}

int ds4_gpu_copy_i32_slice(const ds4_gpu_tensor *src, uint32_t src_offset,
                           ds4_gpu_tensor *dst, uint32_t count)
{
    if (!src || !dst || count == 0) return 0;

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-i32");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_flash_copy_i32_slice");
        if (!pipeline) {
            ds4_gpu_end_compute_encoder(cb, enc);
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-i32");
            return 0;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:ds4_gpu_tensor_buffer(src) offset:ds4_gpu_tensor_offset(src) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(dst) offset:ds4_gpu_tensor_offset(dst) atIndex:1];

        uint32_t off = src_offset;
        uint32_t cnt = count;
        [enc setBytes:&off length:sizeof(off) atIndex:2];
        [enc setBytes:&cnt length:sizeof(cnt) atIndex:3];

        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize tg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];

        ds4_gpu_end_compute_encoder(cb, enc);
        if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-i32");
        return 1;
    }
}

int ds4_gpu_copy_f32_slice(const ds4_gpu_tensor *src, uint32_t src_offset,
                           ds4_gpu_tensor *dst, uint32_t count)
{
    if (!src || !dst || count == 0) return 0;

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-f32");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_flash_copy_f32_slice");
        if (!pipeline) {
            ds4_gpu_end_compute_encoder(cb, enc);
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-f32");
            return 0;
        }

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:ds4_gpu_tensor_buffer(src) offset:ds4_gpu_tensor_offset(src) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(dst) offset:ds4_gpu_tensor_offset(dst) atIndex:1];

        uint32_t off = src_offset;
        uint32_t cnt = count;
        [enc setBytes:&off length:sizeof(off) atIndex:2];
        [enc setBytes:&cnt length:sizeof(cnt) atIndex:3];

        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize tg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];

        ds4_gpu_end_compute_encoder(cb, enc);
        if (owned) ds4_gpu_finish_command_buffer(cb, 1, "copy-f32");
        return 1;
    }
}

/* ==========================================================================
 * Flash-MoE GPU dedup support (histogram for prefill when experts are paged
 * from SSD sidecar via --moe-sidecar + slot-bank)
 * ========================================================================== */

struct FlashDedupHistogramArgs {
    uint32_t n_pairs;
    uint32_t _pad;
};

struct FlashDedupCompactArgs {
    uint32_t n_pairs;
    uint32_t expert_used;
};

int ds4_gpu_flash_moe_dedup_histogram(const ds4_gpu_tensor *selected,
                                      ds4_gpu_tensor       *counts256,
                                      uint32_t              n_pairs)
{
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !counts256 || n_pairs == 0) return 0;

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-hist");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_flash_moe_dedup_histogram");
        if (!pipeline) {
            ds4_gpu_end_compute_encoder(cb, enc);
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-hist");
            return 0;
        }

        [enc setComputePipelineState:pipeline];

        struct FlashDedupHistogramArgs args = { .n_pairs = n_pairs, ._pad = 0 };
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(selected) offset:ds4_gpu_tensor_offset(selected) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(counts256) offset:ds4_gpu_tensor_offset(counts256) atIndex:2];

        MTLSize grid = MTLSizeMake(n_pairs, 1, 1);
        MTLSize tg = MTLSizeMake(32, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];

        ds4_gpu_end_compute_encoder(cb, enc);
        if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-hist");
        return 1;
    }
}

int ds4_gpu_flash_moe_dedup_compact(const ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *pair_weights,
                                    ds4_gpu_tensor       *offsets,        // will be advanced atomically
                                    ds4_gpu_tensor       *out_tokens,
                                    ds4_gpu_tensor       *out_weights,
                                    uint32_t              n_pairs,
                                    uint32_t              expert_used)
{
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !pair_weights || !offsets || !out_tokens || !out_weights || n_pairs == 0) return 0;

    @autoreleasepool {
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;

        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) {
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-compact");
            return 0;
        }

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_flash_moe_dedup_compact");
        if (!pipeline) {
            ds4_gpu_end_compute_encoder(cb, enc);
            if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-compact");
            return 0;
        }

        [enc setComputePipelineState:pipeline];

        struct FlashDedupCompactArgs args = { .n_pairs = n_pairs, .expert_used = expert_used };
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(selected)     offset:ds4_gpu_tensor_offset(selected)     atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(pair_weights) offset:ds4_gpu_tensor_offset(pair_weights) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(offsets)      offset:ds4_gpu_tensor_offset(offsets)      atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(out_tokens)   offset:ds4_gpu_tensor_offset(out_tokens)   atIndex:4];
        [enc setBuffer:ds4_gpu_tensor_buffer(out_weights)  offset:ds4_gpu_tensor_offset(out_weights)  atIndex:5];

        MTLSize grid = MTLSizeMake(n_pairs, 1, 1);
        MTLSize tg = MTLSizeMake(32, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];

        ds4_gpu_end_compute_encoder(cb, enc);
        if (owned) ds4_gpu_finish_command_buffer(cb, 1, "dedup-compact");
        return 1;
    }
}
