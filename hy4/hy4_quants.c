/* HY4 routed quantization ABI, ported from llama.cpp @ 34cccef (MIT).
 * Byte loads keep the sidecar ABI explicit and permit unaligned I/O buffers. */
#include "hy4_quants.h"
#include "hy4_quant_tables.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint16_t hy4_load_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}
static uint32_t hy4_load_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static float hy4_half(uint16_t h) {
    const unsigned e = (h >> 10) & 31u, m = h & 1023u;
    const float a = e == 0 ? ldexpf((float)m, -24) :
                    e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(1024u + m), (int)e - 25);
    return h & 0x8000u ? -a : a;
}
size_t hy4_quant_row_bytes(uint32_t type, size_t n) {
    size_t bytes = type == 43 ? 42 : type == 16 ? 66 : type == 18 ? 98 : type == 23 ? 136 : 0;
    if (!bytes || n == 0 || n % 256 || n / 256 > SIZE_MAX / bytes) return 0;
    return n / 256 * bytes;
}
int hy4_dequantize_row(uint32_t type, const void *src, float *dst, size_t n) {
    const size_t row_bytes = hy4_quant_row_bytes(type, n);
    if (!src || !dst || !row_bytes) return 0;
    const size_t block_bytes = hy4_quant_row_bytes(type, 256);
    const uint8_t *b = src;
    for (size_t ib = 0; ib < n / 256; ++ib, b += block_bytes, dst += 256) {
        const float d = hy4_half(hy4_load_u16(b + (type == 43 ? 40 : 0)));
        if (type == 43) {
            for (unsigned g = 0; g < 64; ++g) {
                const unsigned code = (b[g / 2] >> (4 * (g & 1))) & 15;
                const unsigned sign = (b[32 + g / 8] >> (g & 7)) & 1;
                const unsigned pack = hy4_stq1_0_codebook[(sign << 4) | code];
                for (unsigned p = 0; p < 4; ++p)
                    dst[(g / 16) * 64 + g % 16 + p * 16] = d * (float)((int)((pack >> (2*p)) & 3) - 1);
            }
        } else if (type == 16 || type == 18) {
            for (unsigned sub = 0; sub < 8; ++sub) {
                const uint8_t *q = b + 2 + sub * 8;
                const uint32_t aux = hy4_load_u32(type == 16 ? q + 4 : b + 66 + sub * 4);
                const float ds = d * (0.5f + (aux >> 28)) * (type == 16 ? 0.25f : 0.5f);
                for (unsigned group = 0; group < 4; ++group) {
                    const unsigned signs = hy4_ksigns_iq2xs[(aux >> (7 * group)) & 127];
                    for (unsigned j = 0; j < 8; ++j) {
                        const unsigned grid = type == 16 ?
                            (unsigned)((hy4_iq2xxs_grid[q[group]] >> (j * 8)) & 255) :
                            (hy4_iq3xxs_grid[q[2 * group + j / 4]] >> ((j % 4) * 8)) & 255;
                        dst[sub * 32 + group * 8 + j] = ds * grid * (signs & (1u << j) ? -1.f : 1.f);
                    }
                }
            }
        } else {
            const unsigned hi = hy4_load_u16(b + 2);
            for (unsigned sub = 0; sub < 8; ++sub) {
                const unsigned lo = (b[4 + sub / 2] >> (4 * (sub & 1))) & 15;
                const int scale = (int)(lo | (((hi >> (2 * sub)) & 3) << 4)) - 32;
                const float ds = d * scale;
                for (unsigned j = 0; j < 16; ++j) {
                    const unsigned q = b[8 + sub * 16 + j];
                    dst[sub * 32 + j] = ds * hy4_kvalues_iq4nl[q & 15];
                    dst[sub * 32 + j + 16] = ds * hy4_kvalues_iq4nl[q >> 4];
                }
            }
        }
    }
    return 1;
}
int hy4_quant_matvec_cpu(uint32_t type, const void *weights, const float *x,
                        float *out, size_t in_dim, size_t out_dim, size_t row_bytes) {
    const size_t packed = hy4_quant_row_bytes(type, in_dim);
    if (!weights || !x || !out || !packed || row_bytes < packed || !out_dim ||
        in_dim > SIZE_MAX / sizeof(float) || out_dim > SIZE_MAX / row_bytes) return 0;
    float *row = malloc(in_dim * sizeof(float));
    if (!row) return 0;
    for (size_t r = 0; r < out_dim; ++r) {
        hy4_dequantize_row(type, (const uint8_t *)weights + r * row_bytes, row, in_dim);
        double sum = 0;
        for (size_t i = 0; i < in_dim; ++i) sum += (double)row[i] * x[i];
        out[r] = (float)sum;
    }
    free(row);
    return 1;
}
