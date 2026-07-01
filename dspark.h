#ifndef DS4_DSPARK_H
#define DS4_DSPARK_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct ds4_model ds4_model;
typedef struct ds4_weights ds4_weights;
typedef struct ds4_gpu_graph ds4_gpu_graph;
typedef struct ds4_dspark_draft ds4_dspark_draft;
typedef struct ds4_verify_layer_hc_audit ds4_verify_layer_hc_audit;

typedef struct {
    bool batch_verify;
    bool sequential_verify;
    bool decodeN_attn_ffn_batch;
    bool hybrid_allowed;
    bool hybrid_state_audit;
    bool hybrid_layer_hc_audit;
    bool hybrid_dspark_kv_audit;
    bool decodeN_reads_all_logits;
    bool decodeN_capture_prefixes;
    uint32_t decodeN_capture_prefix_count;
    size_t decodeN_row_count;
    float hybrid_margin_guard;
    uint32_t hybrid_exact_every;
    uint64_t hybrid_block_id;
} ds4_dspark_decodeN_policy;

static bool ds4_dspark_is_loaded(const ds4_dspark_draft *d);
static bool ds4_dspark_cased_batch_attn_byte_enabled(void);
static bool ds4_dspark_cased_attn_compare_enabled(void);
static bool ds4_dspark_cased_state_audit_enabled(void);
static bool ds4_dspark_cased_fail_closed_enabled(void);
static void ds4_dspark_enable_default_fast_verifier(void);
static ds4_dspark_decodeN_policy ds4_dspark_decodeN_policy_make(
        bool quality,
        int  draft_n);

static bool metal_graph_capture_dspark_target_hidden(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       pos);
static bool metal_graph_capture_dspark_target_hidden_batch(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       pos0,
        uint32_t       n_tokens);
static bool metal_graph_dspark_project_main(ds4_gpu_graph *g, uint32_t pos);
static bool metal_graph_dspark_seed_block(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        uint32_t           n_tokens);
static bool metal_graph_dspark_update_main_kv(ds4_gpu_graph *g, uint32_t pos);
static bool metal_graph_dspark_update_main_kv_range(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens);
static bool metal_graph_dspark_rebuild_main_kv_window(
        ds4_gpu_graph *g,
        uint32_t       end_pos);
static bool metal_graph_dspark_three_layer_probe(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           pos);
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
        int               *drafted);
static bool metal_graph_eval_dspark_draft_prefetch_start(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                last_token,
        uint32_t           pos,
        int                draft_cap);
static bool metal_graph_eval_dspark_draft_prefetch_finish(
        ds4_gpu_graph *g,
        int           *drafts,
        int            draft_cap,
        int           *drafted);
static int ds4_dspark_confident_prefix_len(
        const float *confidence_probs,
        int          draft_n,
        float        threshold);
static void metal_graph_log_dspark_kv(ds4_gpu_graph *g);
static void metal_graph_log_dspark_embed(ds4_gpu_graph *g);
static void metal_graph_log_dspark_main(ds4_gpu_graph *g);
static void metal_graph_log_dspark_capture(ds4_gpu_graph *g);

static bool metal_graph_verify_decode2_exact(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token0,
        int                token1,
        uint32_t           pos0,
        int               *top0,
        float             *logits0,
        float             *logits1,
        bool               capture_prefix1,
        bool               batch_output_head);
static bool metal_graph_verify_decodeN_exact(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        uint32_t           n_tokens,
        uint32_t           pos0,
        uint32_t           capture_prefix_count,
        int               *row_tops,
        float             *row_logits,
        ds4_verify_layer_hc_audit *hc_audit);
static bool metal_graph_verify_decodeN_attn_exact_ffn_batch(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        uint32_t           n_tokens,
        uint32_t           pos0,
        bool               capture_prefix1,
        uint32_t           capture_prefix_count,
        int               *row_tops,
        int               *row_topk,
        uint32_t           row_topk_k,
        float             *row_topk_logits,
        float             *row_logits,
        ds4_verify_layer_hc_audit *hc_audit);
static bool metal_graph_verify_decodeN_strict_v1(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        uint32_t           n_tokens,
        uint32_t           pos0,
        bool               capture_prefix1,
        uint32_t           capture_prefix_count,
        int               *row_tops,
        int               *row_topk,
        uint32_t           row_topk_k,
        float             *row_topk_logits,
        float             *row_logits,
        ds4_verify_layer_hc_audit *hc_audit);
static void ds4_verify_layer_hc_audit_free(ds4_verify_layer_hc_audit *a);
static void ds4_verify_layer_hc_audit_compare(
        const ds4_verify_layer_hc_audit *hybrid,
        const ds4_verify_layer_hc_audit *exact,
        const char                      *label);

#endif
