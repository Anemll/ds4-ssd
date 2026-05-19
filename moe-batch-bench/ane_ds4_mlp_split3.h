// ane_ds4_mlp_split3.h
//
// Pure-C ABI for the DSv4 MLP on ANE.  Built on the proven-on-m3u
// `ds4_mlp_inmem_bench_packed_split3.m` graph:
//
//   Single 4D function input  packed: tensor<fp16, [1, 1, 1, N]>
//   N = B*H + 9 * H * (I/T),  T = 3
//
//   For t in 0..2:
//       gate_t   = matmul(tx=true,  x_3d[1,H,B], W_gate_t[H, I/T])  -> [1,B,I/T]
//       up_t     = matmul(tx=true,  x_3d[1,H,B], W_up_t  [H, I/T])  -> [1,B,I/T]
//       hidden_t = silu(gate_t) * up_t                              -> [1,B,I/T]
//       d_t      = matmul(tx=false, hidden_t,    W_down_t[I/T, H])  -> [1,B,H]
//   output = d_0 + d_1 + d_2                                        -> [1,B,H]
//
// Why split-3:  on M3 Ultra (m3u) any tensor with `I = 18432` in a
// matmul-relevant axis fails ANEC compilation.  Splitting everything along the
// I axis into 3 tiles ensures the maximum I-axis dim anywhere is `I/3 = 6144`
// (well under m3u's empirical `I = 16384` ceiling).  Verified on M5 and m3u
// for the full DSv4 shape (H=7168, I=18432).
//
// Caller-side data layout (per evaluation):
//   input      : fp16 [B, H]                       row-major
//   W_gate_t3  : fp16 [3 tiles][H, I/3]            row-major per tile
//                (i.e. 3 contiguous [H, I/3] row-major blocks, one after
//                 another, totaling H*I fp16 elements.  This is a re-tile of
//                 the original [H, I] row-major weight along the I axis.)
//   W_up_t3    : fp16 same layout as W_gate_t3      H*I*2 bytes
//   W_down_t3  : fp16 [3 tiles][I/3, H]            row-major per tile
//                (natural contiguous split of [I, H] row-major along axis 0)
//                I*H*2 bytes total.
//   output     : fp16 [B, H]                       row-major; caller-allocated
//
// All fp16 buffers are little-endian IEEE 754 binary16.
//
// Threading:  a single context is not thread-safe.  Use one context per
// concurrent caller (the underlying `_ANEInMemoryModel` is a process-shared
// resource; per-context state is just IOSurfaces and the model handle).

#ifndef ANE_DS4_MLP_SPLIT3_H
#define ANE_DS4_MLP_SPLIT3_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_ane_mlp_split3_ctx ds4_ane_mlp_split3_ctx;

// Create + compile + load an ANE model for a specific (H, I, B) shape.
// Returns NULL on failure (private ANE classes missing, ANEC compile fail,
// load fail, OOM).  Caller should treat NULL as "use GPU fallback".
//
// The ANE compiler bakes shapes into the model, so a separate context is
// needed for each (H, I, B) tuple the caller intends to evaluate at.
//
// Constraints:
//   - I must be divisible by 3 (the split factor).
//   - H >= 1024 recommended (smaller toy shapes fail at inference time
//     due to an ANE tile-scheduler quirk).
//   - Verified shapes: H=7168, I=18432, B in {1, 8, 16, 32, 64, 96, 128}.
ds4_ane_mlp_split3_ctx *ds4_ane_mlp_split3_create(int H, int I, int B);

// Single MLP evaluation.  All input/output buffers are fp16.
// Buffer sizes (in bytes):
//   input    : B*H*2
//   Wgate_t3 : H*I*2  (3 tiles of [H, I/3] back-to-back)
//   Wup_t3   : H*I*2
//   Wdown_t3 : I*H*2
//   output   : B*H*2
//
// Returns true on success, false on any ANE error (caller should fall back
// to GPU).  On success, `output` contains the MLP result in fp16 row-major.
bool ds4_ane_mlp_split3_eval(
    ds4_ane_mlp_split3_ctx *ctx,
    const void *input_fp16,
    const void *Wgate_t3_fp16,
    const void *Wup_t3_fp16,
    const void *Wdown_t3_fp16,
    void *output_fp16);

// --- Zero-copy variant for hot paths -------------------------------------
//
// `_eval()` above memcpys ~3*H*I*2 bytes from caller buffers into the
// packed IOSurface on every call.  For DSv4 that's 756 MB, ~12 ms at host
// memory bandwidth, dominating the wall-clock vs the ~10 ms ANE compute.
// If the caller is materialising packed bytes from GPU memory or from a
// quantization decoder, they should write into the IOSurface directly via
// the begin/end pair below.
//
// Memory layout inside the surface (fp16, packed contiguous):
//   [0 .. B*H)                            input        [B, H]
//   [B*H        .. B*H +   H*I)           W_gate_t0..2 [3 tiles][H, I/3]
//   [B*H +   H*I .. B*H + 2*H*I)          W_up_t0..2   [3 tiles][H, I/3]
//   [B*H + 2*H*I .. B*H + 3*H*I)          W_down_t0..2 [3 tiles][I/3, H]
// Total: (B*H + 3*H*I) * 2 bytes.
//
// `_begin_input()` locks the IOSurface for write and returns the base pointer
// (NULL on error).  Call `_end_input()` before `_eval_packed()`.
void *ds4_ane_mlp_split3_begin_input(ds4_ane_mlp_split3_ctx *ctx);
void  ds4_ane_mlp_split3_end_input  (ds4_ane_mlp_split3_ctx *ctx);

// Run the MLP on whatever is currently in the IOSurface.  `output_fp16` is
// caller-allocated, B*H*2 bytes.  Returns true on success.
bool ds4_ane_mlp_split3_eval_packed(
    ds4_ane_mlp_split3_ctx *ctx,
    void *output_fp16);

// Free the context.  Safe with NULL.
void ds4_ane_mlp_split3_destroy(ds4_ane_mlp_split3_ctx *ctx);

// Convenience: shape this context was compiled for.
int  ds4_ane_mlp_split3_H(const ds4_ane_mlp_split3_ctx *ctx);
int  ds4_ane_mlp_split3_I(const ds4_ane_mlp_split3_ctx *ctx);
int  ds4_ane_mlp_split3_B(const ds4_ane_mlp_split3_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif // ANE_DS4_MLP_SPLIT3_H
