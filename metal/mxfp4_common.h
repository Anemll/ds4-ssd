// Shared MXFP4 (OCP microscaling FP4) decode helpers for DS4 Metal kernels.
//
// MXFP4 = 32 FP4 E2M1 values sharing one E8M0 scale byte (2^(e-127)).
// Two on-disk/nibble layouts exist in this codebase — every kernel must say
// which one it reads, so both extract helpers are named explicitly:
//
//   split-half (ggml block_mxfp4, the sidecar/streamed format):
//       17-byte interleaved block {uchar e; uchar qs[16];}
//       qs[j] low nibble = element j, high nibble = element j+16
//
//   sequential-pair (MPP 4.1 native scale-plane format, HF safetensors):
//       two separate contiguous planes —
//       data  [n][k/2] bytes, byte b: low nibble = element 2b, high = 2b+1
//       scale [n][k/32] E8M0 bytes
//
// This header is NOT #include-able from runtime-compiled sources (the library
// is built with newLibraryWithSource:, which cannot resolve local includes).
// The host prepends this file to metal/mxfp4_native.metal before compiling —
// see ds4_gpu_ensure_mxfp4_native_library() in ds4_metal.m.  For offline
// syntax checks:  cat metal/mxfp4_common.h metal/mxfp4_native.metal | \
//                 xcrun -sdk macosx metal -std=metal4.1 -x metal -c - -o /dev/null
//
// Feature detection (per docs/mxfp4-native-sidecar-plan.md / MPP 4.1 guide):
// the macOS 27 toolchain defines __HAVE_TENSOR_MULTIPLANE__ at -std=metal4.1;
// DS4_MXFP4_HAS_NATIVE_SCALE_PLANE marks the native scale-plane matmul2d path.
// On macOS 26 (MSL <= 4.0) only the layout helpers + repack kernel compile.

#ifndef DS4_MXFP4_COMMON_H
#define DS4_MXFP4_COMMON_H

#include <metal_stdlib>

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define DS4_MXFP4_HAS_MPP 1
#endif
#endif

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(DS4_MXFP4_HAS_MPP)
#define DS4_MXFP4_HAS_NATIVE_SCALE_PLANE 1
#endif

using namespace metal;

#define DS4_MXFP4_QK 32

// ggml block_mxfp4 (split-half nibble order). 17 bytes.
struct ds4mx_block {
    uchar e;                      // E8M0 shared scale
    uchar qs[DS4_MXFP4_QK / 2];   // 32 x E2M1, split-half order
};

// FP4 E2M1 values by nibble; bit-identical to the M5 hardware decode.
constant float ds4mx_e2m1_lut_f32[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
constant half ds4mx_e2m1_lut_f16[16] = {
     0.0h,  0.5h,  1.0h,  1.5h,  2.0h,  3.0h,  4.0h,  6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

// E8M0 decode: value = 2^(bits - 127), e = 0 decodes to 0.0.
// Exact hardware bit-op (NOT exp2()) so fallback and native paths can never
// numerically diverge.
inline float ds4mx_e8m0_to_float(uchar bits)
{
    return as_type<float>(uint(bits) << 23);
}

// Element j (0..31) of a split-half (ggml) 16-byte nibble array.
inline uchar ds4mx_nibble_splithalf(const device uchar *qs, uint j)
{
    return (qs[j & 15u] >> ((j >> 4u) * 4u)) & 0x0Fu;
}

// Element j (0..31) of a sequential-pair (MPP-native plane) 16-byte run.
inline uchar ds4mx_nibble_seqpair(const device uchar *data, uint j)
{
    return (data[j >> 1u] >> ((j & 1u) * 4u)) & 0x0Fu;
}

#endif // DS4_MXFP4_COMMON_H
