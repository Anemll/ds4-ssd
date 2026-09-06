#ifndef HY4_QUANTS_H
#define HY4_QUANTS_H
#include <stddef.h>
#include <stdint.h>
/* GGUF types 16/18/23/43. Returns zero for unsupported or unaligned rows. */
size_t hy4_quant_row_bytes(uint32_t type, size_t n);
int hy4_dequantize_row(uint32_t type, const void *src, float *dst, size_t n);
int hy4_quant_matvec_cpu(uint32_t type, const void *weights, const float *x,
                        float *out, size_t in_dim, size_t out_dim, size_t row_bytes);
#endif
