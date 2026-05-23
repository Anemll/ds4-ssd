// Experimental ANE DS4 MLP with fp16 activations and int8 dynamic weights.

#ifndef ANE_DS4_MLP_INT8W_H
#define ANE_DS4_MLP_INT8W_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_ane_mlp_int8w_ctx ds4_ane_mlp_int8w_ctx;

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_create(int H, int I, int B, float w_scale, float x_scale);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_create(int H, int I, int B);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_fp16x_create(int H, int I, int B, float w_scale);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_gateup_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
/* Tiled-fused with int8 output (B*H bytes instead of B*H*2). */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale);
/* Single-call fused fp16-weight MLP lowered as conv2d-1x1 (ANE-native pattern). */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_fused_conv_create(int H, int I, int B);

/* Constexpr-weight version: weights are baked into the compiled MIL via a
 * side-loaded fp16 blob file.  Caller passes weights in conv-native [O, I]
 * layout (Wg/Wu: [I, H], Wd: [H, I] — both ggml's native ordering for these
 * tensors).  Only X is uploaded per eval, only Y is read. */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_constexpr_create(int H, int I, int B,
                                                          const uint16_t *Wgate_OI,
                                                          const uint16_t *Wup_OI,
                                                          const uint16_t *Wdown_OI);

bool ds4_ane_mlp_int8w_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    const uint16_t *route_f16,
    uint16_t *output_f16);

bool ds4_ane_mlp_fp16w_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Single-call fused fp16w eval using the conv2d-1x1 MIL.  Same I/O contract as
 * ds4_ane_mlp_fp16w_eval (Wg/Wu [H,I], Wd [I,H], input [B,H], output [B,H]). */
bool ds4_ane_mlp_fp16w_fused_conv_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Constexpr-weight conv eval.  No weight arguments — they're already in the
 * compiled model.  Input [B, H], output [B, H], both fp16. */
bool ds4_ane_mlp_fp16w_constexpr_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Linear two-stage constexpr conv2d-1x1 (no activation) — for LoRA-style
 * projections like DSv4 attention output.  Wa [I, H], Wb [H, I] (ggml-native
 * [O, I] orientation matches conv weight layout).  Input [B, H], output [B, H].
 * Mode 10 ctx with weights baked into MIL via BLOBFILE constexpr. */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_linear_constexpr_create(
    int H, int I, int B,
    const uint16_t *Wa_OI,
    const uint16_t *Wb_OI);

bool ds4_ane_mlp_fp16w_linear_constexpr_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* Linear two-matmul fp16w eval: input · Wa → mid · Wb → output, no
 * activation between.  Reuses a mode-1 (fp16w split) ctx created with the
 * matching shape.  Wa is fed via the gate model, Wb via the down model.
 * Shapes: input [B, H], Wa [H, I], Wb [I, H], output [B, H]. */
bool ds4_ane_mlp_fp16w_linear_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wa_f16,
    const uint16_t *Wb_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16);

/* int8-weight version of the above.  Uses a mode-2 (i8w-fp16x) ctx — the per-
 * tensor w_scale was baked in at ctx create time.  Weight upload bytes are
 * half the fp16 path (Wa+Wb together = H*I + I*H bytes int8 instead of fp16). */
bool ds4_ane_mlp_i8w_fp16x_linear_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t   *Wa_i8,
    const int8_t   *Wb_i8,
    const uint16_t *input_f16,
    uint16_t       *output_f16);

/* Dedicated create for the linear-eval int8 path that lets gate and down have
 * DIFFERENT w_scales (mode-2 create_common uses one scale for both). */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_fp16x_linear_create(int H, int I, int B,
                                                           float a_scale,
                                                           float b_scale);

/* int8 + per-output-channel scale variant of the linear constexpr conv path.
 * Weights baked into MIL via constexpr_blockwise_shift_scale (variant A in
 * INT4_MATMUL_ANE_WORKFLOW.md).  Wa_q/Wb_q [O, I] int8 row-major; Wa_off/Wb_off
 * [O] int8 (typically all zero for symmetric); Wa_scale/Wb_scale [O] fp16.
 * Mode 11.  X [B, H] in, Y [B, H] out. */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_linear_constexpr_create(
    int H, int I, int B,
    const int8_t   *Wa_q_OI,
    const int8_t   *Wa_off_O,
    const uint16_t *Wa_scale_f16_O,
    const int8_t   *Wb_q_OI,
    const int8_t   *Wb_off_O,
    const uint16_t *Wb_scale_f16_O);

bool ds4_ane_mlp_int8w_linear_constexpr_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *input_f16,
    uint16_t       *output_f16);

/* Attach N external input IOSurfaces to an existing mode-11 ctx — creates
 * one ANE request per IOSurface, all sharing the ctx's existing io_out.
 * After this, eval_at_chunk dispatches via the request for the requested
 * chunk index, and ANE reads the matching external IOSurface (no per-call
 * write_surface needed).  Caller owns the IOSurfaces' lifetime. */
#ifdef __OBJC__
#import <IOSurface/IOSurface.h>
bool ds4_ane_mlp_int8w_linear_constexpr_attach_chunks(
    ds4_ane_mlp_int8w_ctx *ctx,
    const IOSurfaceRef    *chunk_input_iosurfaces,
    int                    n_chunks);
#endif

bool ds4_ane_mlp_int8w_linear_constexpr_eval_at_chunk(
    ds4_ane_mlp_int8w_ctx *ctx,
    int                    chunk_idx,
    uint16_t              *output_f16);

bool ds4_ane_mlp_i8w_fp16x_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const uint16_t *input_f16,
    uint16_t *output_f16);

bool ds4_ane_mlp_i8w_i8x_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool ds4_ane_mlp_i8w_i8x_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool ds4_ane_mlp_i8w_i8x_gateup_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16);

/* Mode 7: int8 output. Same i/o as tiled_fused_eval except output is int8. */
bool ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    int8_t       *output_i8);

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8);

/* Variant that skips re-writing the weight IOSurfaces: assumes a prior call
 * has already populated them with the desired expert's gate/up/down weights.
 * Only the input is written and the output is read. Used to measure how much
 * of per-call wall time is spent on weight upload vs. ANE evaluate itself. */
bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval_xonly(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *input_i8,
    uint16_t *output_f16);

const uint16_t *ds4_ane_mlp_int8w_lock_output_f16(ds4_ane_mlp_int8w_ctx *ctx, uint64_t *elems);
void ds4_ane_mlp_int8w_unlock_output(ds4_ane_mlp_int8w_ctx *ctx);

void ds4_ane_mlp_int8w_quant_stats_reset(void);
void ds4_ane_mlp_int8w_quant_stats(uint64_t *hidden_values,
                                   uint64_t *hidden_saturated,
                                   float *hidden_abs_max);

void ds4_ane_mlp_int8w_destroy(ds4_ane_mlp_int8w_ctx *ctx);

int ds4_ane_mlp_int8w_H(const ds4_ane_mlp_int8w_ctx *ctx);
int ds4_ane_mlp_int8w_I(const ds4_ane_mlp_int8w_ctx *ctx);
int ds4_ane_mlp_int8w_B(const ds4_ane_mlp_int8w_ctx *ctx);
float ds4_ane_mlp_int8w_scale(const ds4_ane_mlp_int8w_ctx *ctx);
float ds4_ane_mlp_int8w_x_scale(const ds4_ane_mlp_int8w_ctx *ctx);
float ds4_ane_mlp_int8w_mid_scale(const ds4_ane_mlp_int8w_ctx *ctx);
int ds4_ane_mlp_int8w_mode(const ds4_ane_mlp_int8w_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
