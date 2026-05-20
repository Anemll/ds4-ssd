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

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8);

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
