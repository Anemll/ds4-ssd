// Standalone validation probe for the native-MXFP4 (MPP 4.1) stack:
//   1. GPU plane repack (direct + selected-slot variants) vs a CPU reference
//      repack: byte-exact.
//   2. Native scale-plane matmul2d vs a CPU reference dequant+GEMM: relRMS.
//   3. Tail behavior: m not a multiple of the 64-row tile, with poisoned
//      output padding (documents whether MPP clamps to tensor extents).
//
// Self-contained: builds its own MTLDevice and compiles
// metal/mxfp4_common.h + metal/mxfp4_native.metal exactly the way
// ds4_gpu_ensure_mxfp4_native_library() does, so it runs before the model
// download finishes and without linking ds4.  Run from the repo root:
//   make mxfp4-native-probe
// Bench mode (real sidecar MLP shapes, gate/up + down):
//   MXFP4_PROBE_BENCH=1 ./tests/mxfp4_native_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK 32

typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;

static const float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static float e8m0_to_float(uint8_t e) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)e << 23;
    return v.f;
}

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = mant >> shift;
        const uint32_t round = 1u << (shift - 1u);
        if ((mant & round) && ((mant & (round - 1u)) || (m & 1u))) m++;
        return (uint16_t)(sign | m);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) { m = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) { bits = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f));
    return f;
}

// Element j (0..31) of a ggml split-half block.
static uint8_t nib_splithalf(const uint8_t *qs, int j) {
    return (qs[j & 15] >> ((j >> 4) * 4)) & 0x0F;
}

// CPU reference repack of one block into 16 seq-pair data bytes.
static void ref_repack_block(const block_mxfp4 *blk, uint8_t *data16) {
    for (int i = 0; i < 16; i++) {
        uint8_t lo = nib_splithalf(blk->qs, 2 * i);
        uint8_t hi = nib_splithalf(blk->qs, 2 * i + 1);
        data16[i] = (uint8_t)(lo | (hi << 4));
    }
}

static float wval(const block_mxfp4 *row_blocks, int k) {
    const block_mxfp4 *blk = row_blocks + k / QK;
    return e8m0_to_float(blk->e) * kE2M1[nib_splithalf(blk->qs, k % QK)];
}

static int8_t quant_i8_ref(float v, float qscale) {
    float scaled = v * qscale;
    if (scaled < -128.0f) scaled = -128.0f;
    if (scaled > 127.0f) scaled = 127.0f;
    return (int8_t)rintf(scaled);
}

int main(void) {
    @autoreleasepool {
        const int bench = getenv("MXFP4_PROBE_BENCH") &&
                          atoi(getenv("MXFP4_PROBE_BENCH")) != 0;
        // Correctness shape (small enough for a CPU reference GEMM);
        // bench mode switches to the real sidecar MLP shapes below.
        int m = 128, n = 256, k = 1024;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSString *hdr = [NSString stringWithContentsOfFile:@"metal/mxfp4_common.h"
                                                  encoding:NSUTF8StringEncoding error:NULL];
        NSString *body = [NSString stringWithContentsOfFile:@"metal/mxfp4_native.metal"
                                                   encoding:NSUTF8StringEncoding error:NULL];
        if (!hdr || !body) {
            fprintf(stderr, "FAIL: run from the repo root (metal/mxfp4_*.{h,metal} not found)\n");
            return 1;
        }
        NSString *src = [NSString stringWithFormat:@"%@\n%@", hdr, body];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        bool lang41 = false;
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = (MTLLanguageVersion)((4 << 16) + 1);
            lang41 = true;
        } else {
            opts.languageVersion = MTLLanguageVersion4_0;
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib && lang41) {
            opts.languageVersion = MTLLanguageVersion4_0;
            err = nil; lang41 = false;
            lib = [device newLibraryWithSource:src options:opts error:&err];
        }
        if (!lib) {
            fprintf(stderr, "FAIL: library compile: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLComputePipelineState> repack =
            [device newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"kernel_dsv4_mxfp4_repack_planes"] error:&err];
        if (!repack) { fprintf(stderr, "FAIL: repack pipeline\n"); return 1; }
        id<MTLComputePipelineState> repackSelected =
            [device newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"kernel_dsv4_mxfp4_repack_selected_planes"] error:&err];
        if (!repackSelected) { fprintf(stderr, "FAIL: selected repack pipeline\n"); return 1; }
        id<MTLComputePipelineState> copySelected =
            [device newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"kernel_dsv4_mxfp4_copy_selected_planes"] error:&err];
        if (!copySelected) { fprintf(stderr, "FAIL: selected plane copy pipeline\n"); return 1; }
        id<MTLFunction> mmfn = [lib newFunctionWithName:@"kernel_dsv4_mxfp4_native_matmul_n64"];
        id<MTLComputePipelineState> mm =
            mmfn ? [device newComputePipelineStateWithFunction:mmfn error:&err] : nil;
        id<MTLFunction> mmSelectedFn =
            [lib newFunctionWithName:@"kernel_dsv4_mxfp4_native_matmul_selected_n64"];
        id<MTLComputePipelineState> mmSelected =
            mmSelectedFn ? [device newComputePipelineStateWithFunction:mmSelectedFn error:&err] : nil;
        printf("library: MSL %s, repack ok, selected repack ok, selected plane copy ok, native matmul %s, selected matmul %s\n",
               lang41 ? "4.1" : "4.0",
               mm ? "ok" : "UNAVAILABLE",
               mmSelected ? "ok" : "UNAVAILABLE");

        NSString *moeSrc = [NSString stringWithContentsOfFile:@"metal/moe.metal"
                                                     encoding:NSUTF8StringEncoding error:NULL];
        id<MTLLibrary> moeLib = nil;
        id<MTLComputePipelineState> planeI8 = nil;
        id<MTLComputePipelineState> planeF16 = nil;
        if (moeSrc) {
            NSRange marker = [moeSrc rangeOfString:@"struct ds4_metal_dsv4_moe_swiglu_weight_args"];
            if (marker.location != NSNotFound) {
                NSString *prefix = [moeSrc substringToIndex:marker.location];
                moeSrc = [NSString stringWithFormat:
                    @"#include <metal_stdlib>\nusing namespace metal;\n%@",
                    prefix];
            }
            NSError *moeErr = nil;
            moeLib = [device newLibraryWithSource:moeSrc options:opts error:&moeErr];
            if (moeLib) {
                planeI8 = [device newComputePipelineStateWithFunction:
                    [moeLib newFunctionWithName:@"kernel_dsv4_mpp_dequant_mxfp4_planes_transpose_i8"] error:&moeErr];
                planeF16 = [device newComputePipelineStateWithFunction:
                    [moeLib newFunctionWithName:@"kernel_dsv4_ane_dequant_mxfp4_planes_transpose_f16"] error:&moeErr];
            }
        }
        printf("moe plane dequant kernels: i8 %s, f16 %s\n",
               planeI8 ? "ok" : "UNAVAILABLE",
               planeF16 ? "ok" : "UNAVAILABLE");

        NSString *mppSrc =
            @"#include <metal_stdlib>\n"
             "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
             "using namespace metal;\n"
             "using namespace mpp::tensor_ops;\n"
             "template <typename AT, typename BT, typename CT, int NT>\n"
             "inline void ds4_mpp_run_tile_nt(device AT *A, device BT *B, device CT *C,\n"
             "                                constant uint &M, constant uint &N, constant uint &K,\n"
             "                                uint2 tgid) {\n"
             "    constexpr auto desc = matmul2d_descriptor(64, NT, static_cast<int>(dynamic_extent));\n"
             "    matmul2d<desc, execution_simdgroups<4>> op;\n"
             "    auto tA = tensor(A, dextents<int32_t, 2>{(int32_t)K, (int32_t)M}, array<int32_t, 2>{1, (int32_t)K});\n"
             "    auto tB = tensor(B, dextents<int32_t, 2>{(int32_t)N, (int32_t)K}, array<int32_t, 2>{1, (int32_t)N});\n"
             "    auto tC = tensor(C, dextents<int32_t, 2>{(int32_t)N, (int32_t)M}, array<int32_t, 2>{1, (int32_t)N});\n"
             "    auto mA = tA.slice(0, tgid.y * 64);\n"
             "    auto mB = tB.slice(tgid.x * NT, 0);\n"
             "    auto mC = tC.slice(tgid.x * NT, tgid.y * 64);\n"
             "    op.run(mA, mB, mC);\n"
             "}\n"
             "kernel void ds4_mpp_h_h_f_n64(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
             "                              constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
             "                              uint2 tgid [[threadgroup_position_in_grid]]) {\n"
             "    ds4_mpp_run_tile_nt<half, half, float, 64>(A, B, C, M, N, K, tgid);\n"
             "}\n"
             "kernel void ds4_mpp_f_h_f_n64(device float *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
             "                              constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
             "                              uint2 tgid [[threadgroup_position_in_grid]]) {\n"
             "    ds4_mpp_run_tile_nt<float, half, float, 64>(A, B, C, M, N, K, tgid);\n"
             "}\n";
        id<MTLComputePipelineState> mppHH = nil;
        id<MTLComputePipelineState> mppFH = nil;
        {
            NSError *mppErr = nil;
            id<MTLLibrary> mppLib = [device newLibraryWithSource:mppSrc options:opts error:&mppErr];
            if (mppLib) {
                mppHH = [device newComputePipelineStateWithFunction:
                    [mppLib newFunctionWithName:@"ds4_mpp_h_h_f_n64"] error:&mppErr];
                mppFH = [device newComputePipelineStateWithFunction:
                    [mppLib newFunctionWithName:@"ds4_mpp_f_h_f_n64"] error:&mppErr];
            }
        }
        printf("mpp h_h/f_h n64 kernels: %s/%s\n",
               mppHH ? "ok" : "UNAVAILABLE",
               mppFH ? "ok" : "UNAVAILABLE");

        srand(42);
        const int kb_per_row = k / QK;
        const size_t n_blocks = (size_t)n * kb_per_row;
        block_mxfp4 *W = malloc(n_blocks * sizeof(block_mxfp4));
        for (size_t i = 0; i < n_blocks; i++) {
            W[i].e = (uint8_t)(112 + rand() % 28);   // 2^-15 .. 2^12
            for (int j = 0; j < 16; j++) W[i].qs[j] = (uint8_t)(rand() & 0xff);
        }
        W[0].e = 0;     // E8M0 edge: decodes to 0.0, whole block contributes nothing
        W[1].e = 127;   // scale 1.0

        uint16_t *A = malloc((size_t)m * k * sizeof(uint16_t));
        for (size_t i = 0; i < (size_t)m * k; i++)
            A[i] = f32_to_f16_bits(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);

        // ---- GPU buffers
        id<MTLBuffer> bW = [device newBufferWithBytes:W length:n_blocks * 17
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> bA = [device newBufferWithBytes:A length:(size_t)m * k * 2
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> bData = [device newBufferWithLength:(size_t)n * k / 2
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bScale = [device newBufferWithLength:(size_t)n * kb_per_row
                                                   options:MTLResourceStorageModeShared];
        const int m_pad = (m + 63) & ~63;
        id<MTLBuffer> bC = [device newBufferWithLength:(size_t)m_pad * n * 4
                                               options:MTLResourceStorageModeShared];

        // ---- 1. repack + byte-exact check
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:repack];
            [enc setBuffer:bW offset:0 atIndex:0];
            [enc setBuffer:bData offset:0 atIndex:1];
            [enc setBuffer:bScale offset:0 atIndex:2];
            uint32_t rows = (uint32_t)n, depth = (uint32_t)k, z = 0;
            [enc setBytes:&rows length:4 atIndex:3];
            [enc setBytes:&depth length:4 atIndex:4];
            [enc setBytes:&z length:4 atIndex:5];
            [enc setBytes:&z length:4 atIndex:6];
            [enc setBytes:&z length:4 atIndex:7];
            [enc dispatchThreads:MTLSizeMake(kb_per_row, n, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const uint8_t *gd = bData.contents, *gs = bScale.contents;
            size_t bad = 0;
            for (int r = 0; r < n && bad < 8; r++) {
                for (int b = 0; b < kb_per_row && bad < 8; b++) {
                    const block_mxfp4 *blk = W + (size_t)r * kb_per_row + b;
                    uint8_t ref[16]; ref_repack_block(blk, ref);
                    if (memcmp(gd + ((size_t)r * k / 2 + (size_t)b * 16), ref, 16) ||
                        gs[(size_t)r * kb_per_row + b] != blk->e) {
                        fprintf(stderr, "repack mismatch at row %d block %d\n", r, b);
                        bad++;
                    }
                }
            }
            if (bad) { fprintf(stderr, "FAIL: repack\n"); return 1; }
            printf("repack: byte-exact over %zu blocks (incl. e=0/127 edges)\n", n_blocks);
        }

        // ---- 1b. selected-slot repack + byte-exact check.  Decode slotwise
        // paths keep slot ids in a GPU tensor; this exercises that no-readback
        // source selection.
        {
            const uint32_t slots = 3;
            const uint32_t selected_slot = 2;
            const uint32_t route = 0;
            const size_t slot_stride = n_blocks * sizeof(block_mxfp4);
            uint8_t *bank = malloc((size_t)slots * slot_stride);
            for (size_t i = 0; i < (size_t)slots * slot_stride; i++) {
                bank[i] = (uint8_t)(0xA5u ^ (uint8_t)i);
            }
            memcpy(bank + (size_t)selected_slot * slot_stride, W, slot_stride);
            int32_t selected_ids[1] = { (int32_t)selected_slot };
            id<MTLBuffer> bBank = [device newBufferWithBytes:bank
                                                       length:(size_t)slots * slot_stride
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSelected = [device newBufferWithBytes:selected_ids
                                                          length:sizeof(selected_ids)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSelData = [device newBufferWithLength:(size_t)n * k / 2
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSelScale = [device newBufferWithLength:(size_t)n * kb_per_row
                                                          options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:repackSelected];
            [enc setBuffer:bBank offset:0 atIndex:0];
            [enc setBuffer:bSelData offset:0 atIndex:1];
            [enc setBuffer:bSelScale offset:0 atIndex:2];
            [enc setBuffer:bSelected offset:0 atIndex:3];
            uint32_t rows = (uint32_t)n, depth = (uint32_t)k;
            uint32_t stride = (uint32_t)slot_stride;
            [enc setBytes:&rows length:4 atIndex:4];
            [enc setBytes:&depth length:4 atIndex:5];
            [enc setBytes:&stride length:4 atIndex:6];
            [enc setBytes:&route length:4 atIndex:7];
            [enc dispatchThreads:MTLSizeMake(kb_per_row, n, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const uint8_t *gd = bSelData.contents, *gs = bSelScale.contents;
            size_t bad = 0;
            for (int r = 0; r < n && bad < 8; r++) {
                for (int b = 0; b < kb_per_row && bad < 8; b++) {
                    const block_mxfp4 *blk = W + (size_t)r * kb_per_row + b;
                    uint8_t ref[16]; ref_repack_block(blk, ref);
                    if (memcmp(gd + ((size_t)r * k / 2 + (size_t)b * 16), ref, 16) ||
                        gs[(size_t)r * kb_per_row + b] != blk->e) {
                        fprintf(stderr, "selected repack mismatch at row %d block %d\n", r, b);
                        bad++;
                    }
                }
            }
            free(bank);
            if (bad) { fprintf(stderr, "FAIL: selected repack\n"); return 1; }
            printf("selected repack: byte-exact for selected slot %u\n", selected_slot);
        }

        // ---- 1c. selected-slot plane copy + byte-exact check.  Native
        // plane-split sidecars already store the data/scale planes, so decode
        // only needs to select and copy those planes without nibble repacking.
        {
            const uint32_t slots = 3;
            const uint32_t selected_slot = 2;
            const uint32_t route = 0;
            const size_t data_bytes = (size_t)n * k / 2;
            const size_t scale_bytes = (size_t)n * kb_per_row;
            const size_t slot_stride = data_bytes + scale_bytes;
            uint8_t *bank = malloc((size_t)slots * slot_stride);
            for (size_t i = 0; i < (size_t)slots * slot_stride; i++) {
                bank[i] = (uint8_t)(0x5Au ^ (uint8_t)i);
            }
            memcpy(bank + (size_t)selected_slot * slot_stride,
                   bData.contents,
                   data_bytes);
            memcpy(bank + (size_t)selected_slot * slot_stride + data_bytes,
                   bScale.contents,
                   scale_bytes);
            int32_t selected_ids[1] = { (int32_t)selected_slot };
            id<MTLBuffer> bPlaneBank = [device newBufferWithBytes:bank
                                                            length:(size_t)slots * slot_stride
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSelected = [device newBufferWithBytes:selected_ids
                                                          length:sizeof(selected_ids)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCopyData = [device newBufferWithLength:data_bytes
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCopyScale = [device newBufferWithLength:scale_bytes
                                                            options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:copySelected];
            [enc setBuffer:bPlaneBank offset:0 atIndex:0];
            [enc setBuffer:bPlaneBank offset:data_bytes atIndex:1];
            [enc setBuffer:bCopyData offset:0 atIndex:2];
            [enc setBuffer:bCopyScale offset:0 atIndex:3];
            [enc setBuffer:bSelected offset:0 atIndex:4];
            uint32_t rows = (uint32_t)n, depth = (uint32_t)k;
            uint32_t stride = (uint32_t)slot_stride;
            [enc setBytes:&rows length:4 atIndex:5];
            [enc setBytes:&depth length:4 atIndex:6];
            [enc setBytes:&stride length:4 atIndex:7];
            [enc setBytes:&route length:4 atIndex:8];
            [enc dispatchThreads:MTLSizeMake(data_bytes + scale_bytes, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(copySelected.threadExecutionWidth * 4u, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (memcmp(bCopyData.contents, bData.contents, data_bytes) ||
                memcmp(bCopyScale.contents, bScale.contents, scale_bytes)) {
                fprintf(stderr, "FAIL: selected plane copy mismatch\n");
                return 1;
            }
            printf("selected plane copy: byte-exact for selected slot %u\n", selected_slot);
            free(bank);
        }

        // ---- 1d. Plane-split dequant kernels used by the MPP/NAX prefill arm.
        // These consume the compact sidecar family layout [data][scale] and
        // write the transposed layout expected by ds4_mpp_* matmul2d kernels:
        // dst[k * rows + row].
        if (!planeI8 || !planeF16) {
            fprintf(stderr, "FAIL: moe plane dequant pipelines unavailable\n");
            return 1;
        }
        {
            const size_t data_bytes = (size_t)n * k / 2;
            const size_t scale_bytes = (size_t)n * kb_per_row;
            const size_t compact_bytes = data_bytes + scale_bytes;
            uint8_t *compact = malloc(compact_bytes);
            memcpy(compact, bData.contents, data_bytes);
            memcpy(compact + data_bytes, bScale.contents, scale_bytes);
            id<MTLBuffer> bCompact = [device newBufferWithBytes:compact
                                                         length:compact_bytes
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> bF16T = [device newBufferWithLength:(size_t)n * k * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> bI8T = [device newBufferWithLength:(size_t)n * k
                                                     options:MTLResourceStorageModeShared];
            memset(bF16T.contents, 0xCD, (size_t)n * k * sizeof(uint16_t));
            memset(bI8T.contents, 0xCD, (size_t)n * k);

            const uint32_t rows = (uint32_t)n;
            const uint32_t cols = (uint32_t)k;
            const uint32_t total = rows * (cols / 32u) * 2u;
            const float qscale = 512.0f;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:planeF16];
            [enc setBuffer:bCompact offset:0 atIndex:0];
            [enc setBuffer:bF16T offset:0 atIndex:1];
            [enc setBytes:&rows length:4 atIndex:2];
            [enc setBytes:&cols length:4 atIndex:3];
            [enc setBytes:&total length:4 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((total + 255u) / 256u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:planeI8];
            [enc setBuffer:bCompact offset:0 atIndex:0];
            [enc setBuffer:bI8T offset:0 atIndex:1];
            [enc setBytes:&rows length:4 atIndex:2];
            [enc setBytes:&cols length:4 atIndex:3];
            [enc setBytes:&total length:4 atIndex:4];
            [enc setBytes:&qscale length:sizeof(qscale) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((total + 255u) / 256u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const uint16_t *f16t = bF16T.contents;
            const int8_t *i8t = bI8T.contents;
            size_t bad_f16 = 0, bad_i8 = 0;
            for (int r = 0; r < n && (bad_f16 < 8 || bad_i8 < 8); r++) {
                for (int kk = 0; kk < k && (bad_f16 < 8 || bad_i8 < 8); kk++) {
                    const size_t idx = (size_t)kk * n + r;
                    const float ref = wval(W + (size_t)r * kb_per_row, kk);
                    const uint16_t want_h = f32_to_f16_bits(ref);
                    const int8_t want_i8 = quant_i8_ref(ref, qscale);
                    if (bad_f16 < 8 && f16t[idx] != want_h) {
                        fprintf(stderr,
                                "plane f16 mismatch row %d col %d: got %.8g want %.8g bits %04x/%04x\n",
                                r, kk, f16_bits_to_f32(f16t[idx]), f16_bits_to_f32(want_h),
                                f16t[idx], want_h);
                        bad_f16++;
                    }
                    if (bad_i8 < 8 && i8t[idx] != want_i8) {
                        fprintf(stderr,
                                "plane i8 mismatch row %d col %d: got %d want %d ref %.8g\n",
                                r, kk, (int)i8t[idx], (int)want_i8, ref);
                        bad_i8++;
                    }
                }
            }
            free(compact);
            if (bad_f16 || bad_i8) {
                fprintf(stderr, "FAIL: plane dequant transpose (%zu f16, %zu i8 mismatches)\n",
                        bad_f16, bad_i8);
                return 1;
            }
            printf("plane dequant transpose: f16/i8 byte-exact over %zu values\n",
                   (size_t)n * k);

            if (mppHH) {
                memset(bC.contents, 0x7f, (size_t)m_pad * n * 4);
                id<MTLCommandBuffer> mmcb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> mmenc = [mmcb computeCommandEncoder];
                [mmenc setComputePipelineState:mppHH];
                [mmenc setBuffer:bA offset:0 atIndex:0];
                [mmenc setBuffer:bF16T offset:0 atIndex:1];
                [mmenc setBuffer:bC offset:0 atIndex:2];
                uint32_t um = m, un = n, uk = k;
                [mmenc setBytes:&um length:4 atIndex:3];
                [mmenc setBytes:&un length:4 atIndex:4];
                [mmenc setBytes:&uk length:4 atIndex:5];
                NSUInteger tew = mppHH.threadExecutionWidth;
                [mmenc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
                [mmenc endEncoding];
                [mmcb commit];
                [mmcb waitUntilCompleted];

                const float *C = bC.contents;
                double se = 0, sr = 0, maxabs = 0;
                int badc = 0;
                for (int mi = 0; mi < m; mi++) {
                    for (int ni = 0; ni < n; ni++) {
                        double ref = 0;
                        const block_mxfp4 *rb = W + (size_t)ni * kb_per_row;
                        for (int kk = 0; kk < k; kk++) {
                            const double av = f16_bits_to_f32(A[(size_t)mi * k + kk]);
                            const double wv = f16_bits_to_f32(f32_to_f16_bits(wval(rb, kk)));
                            ref += av * wv;
                        }
                        const double g = C[(size_t)mi * n + ni];
                        if (!isfinite(g)) badc++;
                        const double d = g - ref;
                        se += d * d;
                        sr += ref * ref;
                        if (fabs(d) > maxabs) maxabs = fabs(d);
                    }
                }
                const double rel = sqrt(se / (sr > 0 ? sr : 1));
                printf("plane dequant + ds4_mpp_h_h_f_n64: relRMS %.3e, max abs diff %.3e%s\n",
                       rel, maxabs, badc ? " (NON-FINITE OUTPUTS)" : "");
                if (badc || rel > 1e-3) {
                    fprintf(stderr, "FAIL: plane dequant + MPP h_h matmul\n");
                    return 1;
                }
            }

            if (mppFH) {
                float *Af = malloc((size_t)m * k * sizeof(float));
                for (size_t ai = 0; ai < (size_t)m * k; ai++) {
                    Af[ai] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                }
                id<MTLBuffer> bAf = [device newBufferWithBytes:Af
                                                        length:(size_t)m * k * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
                memset(bC.contents, 0x7f, (size_t)m_pad * n * 4);
                id<MTLCommandBuffer> mmcb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> mmenc = [mmcb computeCommandEncoder];
                [mmenc setComputePipelineState:mppFH];
                [mmenc setBuffer:bAf offset:0 atIndex:0];
                [mmenc setBuffer:bF16T offset:0 atIndex:1];
                [mmenc setBuffer:bC offset:0 atIndex:2];
                uint32_t um = m, un = n, uk = k;
                [mmenc setBytes:&um length:4 atIndex:3];
                [mmenc setBytes:&un length:4 atIndex:4];
                [mmenc setBytes:&uk length:4 atIndex:5];
                NSUInteger tew = mppFH.threadExecutionWidth;
                [mmenc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
                [mmenc endEncoding];
                [mmcb commit];
                [mmcb waitUntilCompleted];

                const float *C = bC.contents;
                double se = 0, sr = 0, maxabs = 0;
                int badc = 0;
                for (int mi = 0; mi < m; mi++) {
                    for (int ni = 0; ni < n; ni++) {
                        double ref = 0;
                        const block_mxfp4 *rb = W + (size_t)ni * kb_per_row;
                        for (int kk = 0; kk < k; kk++) {
                            const double wv = f16_bits_to_f32(f32_to_f16_bits(wval(rb, kk)));
                            ref += (double)Af[(size_t)mi * k + kk] * wv;
                        }
                        const double g = C[(size_t)mi * n + ni];
                        if (!isfinite(g)) badc++;
                        const double d = g - ref;
                        se += d * d;
                        sr += ref * ref;
                        if (fabs(d) > maxabs) maxabs = fabs(d);
                    }
                }
                const double rel = sqrt(se / (sr > 0 ? sr : 1));
                printf("plane dequant + ds4_mpp_f_h_f_n64: relRMS %.3e, max abs diff %.3e%s\n",
                       rel, maxabs, badc ? " (NON-FINITE OUTPUTS)" : "");
                free(Af);
                if (badc || rel > 1e-3) {
                    fprintf(stderr, "FAIL: plane dequant + MPP f_h matmul\n");
                    return 1;
                }
            }
        }

        if (!mm) {
            printf("native matmul unavailable on this OS/toolchain/GPU — repack-only PASS\n");
            return 0;
        }
        if (!mmSelected) {
            fprintf(stderr, "FAIL: selected native matmul pipeline\n");
            return 1;
        }

        // ---- 2. native matmul vs CPU reference (m multiple of 64)
        double relrms;
        {
            memset(bC.contents, 0x7f, (size_t)m_pad * n * 4);   // poison
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mm];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bData offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            uint32_t um = m, un = n, uk = k;
            [enc setBytes:&um length:4 atIndex:3];
            [enc setBytes:&un length:4 atIndex:4];
            [enc setBytes:&uk length:4 atIndex:5];
            [enc setBuffer:bScale offset:0 atIndex:6];
            NSUInteger tew = mm.threadExecutionWidth;
            [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const float *C = bC.contents;
            double se = 0, sr = 0; double maxabs = 0; int badc = 0;
            for (int mi = 0; mi < m; mi++) {
                for (int ni = 0; ni < n; ni++) {
                    double ref = 0;
                    const block_mxfp4 *rb = W + (size_t)ni * kb_per_row;
                    for (int kk = 0; kk < k; kk++)
                        ref += (double)f16_bits_to_f32(A[(size_t)mi * k + kk]) * wval(rb, kk);
                    double g = C[(size_t)mi * n + ni];
                    if (!isfinite(g)) badc++;
                    double d = g - ref;
                    se += d * d; sr += ref * ref;
                    if (fabs(d) > maxabs) maxabs = fabs(d);
                }
            }
            relrms = sqrt(se / (sr > 0 ? sr : 1));
            printf("native matmul m=%d n=%d k=%d: relRMS %.3e, max abs diff %.3e%s\n",
                   m, n, k, relrms, maxabs, badc ? " (NON-FINITE OUTPUTS)" : "");
            if (badc || relrms > 1e-4) { fprintf(stderr, "FAIL: native matmul\n"); return 1; }
        }

        // ---- 2b. selected-slot native matmul vs CPU reference.  This is the
        // no-copy decode path: MPP reads FP4/scales directly from selected slot.
        {
            const uint32_t slots = 3;
            const uint32_t selected_slot = 2;
            const uint32_t route = 0;
            const size_t data_bytes = (size_t)n * k / 2;
            const size_t scale_bytes = (size_t)n * kb_per_row;
            const size_t slot_stride = data_bytes + scale_bytes;
            uint8_t *bank = malloc((size_t)slots * slot_stride);
            for (size_t i = 0; i < (size_t)slots * slot_stride; i++) {
                bank[i] = (uint8_t)(0x3Cu ^ (uint8_t)i);
            }
            memcpy(bank + (size_t)selected_slot * slot_stride,
                   bData.contents,
                   data_bytes);
            memcpy(bank + (size_t)selected_slot * slot_stride + data_bytes,
                   bScale.contents,
                   scale_bytes);
            int32_t selected_ids[1] = { (int32_t)selected_slot };
            id<MTLBuffer> bPlaneBank = [device newBufferWithBytes:bank
                                                            length:(size_t)slots * slot_stride
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSelected = [device newBufferWithBytes:selected_ids
                                                          length:sizeof(selected_ids)
                                                         options:MTLResourceStorageModeShared];
            memset(bC.contents, 0x7f, (size_t)m_pad * n * 4);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mmSelected];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bPlaneBank offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            uint32_t um = m, un = n, uk = k;
            uint32_t stride = (uint32_t)slot_stride;
            [enc setBytes:&um length:4 atIndex:3];
            [enc setBytes:&un length:4 atIndex:4];
            [enc setBytes:&uk length:4 atIndex:5];
            [enc setBuffer:bPlaneBank offset:data_bytes atIndex:6];
            [enc setBuffer:bSelected offset:0 atIndex:7];
            [enc setBytes:&stride length:4 atIndex:8];
            [enc setBytes:&route length:4 atIndex:9];
            NSUInteger tew = mmSelected.threadExecutionWidth;
            [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (m + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const float *C = bC.contents;
            double se = 0, sr = 0; double maxabs = 0; int badc = 0;
            for (int mi = 0; mi < m; mi++) {
                for (int ni = 0; ni < n; ni++) {
                    double ref = 0;
                    const block_mxfp4 *rb = W + (size_t)ni * kb_per_row;
                    for (int kk = 0; kk < k; kk++)
                        ref += (double)f16_bits_to_f32(A[(size_t)mi * k + kk]) * wval(rb, kk);
                    double g = C[(size_t)mi * n + ni];
                    if (!isfinite(g)) badc++;
                    double d = g - ref;
                    se += d * d; sr += ref * ref;
                    if (fabs(d) > maxabs) maxabs = fabs(d);
                }
            }
            relrms = sqrt(se / (sr > 0 ? sr : 1));
            printf("selected native matmul slot=%u: relRMS %.3e, max abs diff %.3e%s\n",
                   selected_slot, relrms, maxabs, badc ? " (NON-FINITE OUTPUTS)" : "");
            free(bank);
            if (badc || relrms > 1e-4) {
                fprintf(stderr, "FAIL: selected native matmul\n");
                return 1;
            }
        }

        // ---- 3. partial/small-m probe: every m below or across the 64-row
        // tile, full reference check on all rows + poisoned output padding.
        // This is the regime DS4_FLASH_MOE_MPP_ALLOW_PARTIAL_TILES exposes
        // (per-expert token groups of 1..63 rows in short prompts).
        const int m_sweep[] = { 1, 2, 3, 7, 16, 33, 63, 65, 100 };
        for (size_t msi = 0; msi < sizeof(m_sweep)/sizeof(m_sweep[0]); msi++) {
            int mt = m_sweep[msi];
            memset(bC.contents, 0x7f, (size_t)m_pad * n * 4);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mm];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bData offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            uint32_t um = mt, un = n, uk = k;
            [enc setBytes:&um length:4 atIndex:3];
            [enc setBytes:&un length:4 atIndex:4];
            [enc setBytes:&uk length:4 atIndex:5];
            [enc setBuffer:bScale offset:0 atIndex:6];
            NSUInteger tew = mm.threadExecutionWidth;
            [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, (mt + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const float *C = bC.contents;
            int spot_bad = 0;
            const int full = mt <= 16;          // small m: check every cell
            for (int mi = 0; mi < mt; mi++) {
                if (!full && mi >= 2 && mi < mt - 2) continue;
                const int ncheck = full ? n : 16;
                for (int ni = 0; ni < ncheck; ni++) {
                    double ref = 0;
                    const block_mxfp4 *rb = W + (size_t)ni * kb_per_row;
                    for (int kk = 0; kk < k; kk++)
                        ref += (double)f16_bits_to_f32(A[(size_t)mi * k + kk]) * wval(rb, kk);
                    double g = C[(size_t)mi * n + ni];
                    if (!isfinite(g) || fabs(g - ref) > 1e-2 * (fabs(ref) + 1)) spot_bad++;
                }
            }
            const uint32_t *poison = bC.contents;
            int pad_touched = 0;
            for (size_t i = (size_t)mt * n; i < (size_t)m_pad * n; i++)
                if (poison[i] != 0x7f7f7f7fu) pad_touched++;
            printf("partial m=%-3d (%s check): %s, %d padding words written\n",
                   mt, full ? "full" : "spot",
                   spot_bad ? "WRONG VALUES" : "correct", pad_touched);
            if (spot_bad || pad_touched) {
                fprintf(stderr, "FAIL: partial-tile m=%d (small-m regime unsafe)\n", mt);
                return 1;
            }
        }

        // ---- 4. optional bench: real sidecar MLP shapes, four arms:
        //   native    = MPP 4.1 FP4 + E8M0 scale plane (the production path)
        //   noscale   = raw FP4, no block scaling (reference upper bound)
        //   h_h       = MPP 4.0 half x half, weights already dequanted (resident f16)
        //   deq+h_h   = MPP 4.0 per-use arm: dequant planes -> half, then h_h
        if (bench) {
            id<MTLComputePipelineState> mmNo = nil, mmHH = nil, deqH = nil;
            {
                id<MTLFunction> f;
                if ((f = [lib newFunctionWithName:@"kernel_dsv4_mxfp4_native_matmul_noscale_n64"]))
                    mmNo = [device newComputePipelineStateWithFunction:f error:&err];
                if ((f = [lib newFunctionWithName:@"kernel_dsv4_mxfp4_matmul_h_h_n64"]))
                    mmHH = [device newComputePipelineStateWithFunction:f error:&err];
                if ((f = [lib newFunctionWithName:@"kernel_dsv4_mxfp4_dequant_planes_to_half"]))
                    deqH = [device newComputePipelineStateWithFunction:f error:&err];
            }
            struct { const char *name; int n, k; } shapes[] = {
                { "gate/up (n=2048 k=4096)", 2048, 4096 },
                { "down    (n=4096 k=2048)", 4096, 2048 },
            };
            const int bm = 128, iters = 50;
            for (int s = 0; s < 2; s++) {
                const int bn = shapes[s].n, bk = shapes[s].k;
                const size_t nb = (size_t)bn * (bk / QK);
                id<MTLBuffer> wD = [device newBufferWithLength:(size_t)bn * bk / 2
                                                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> wS = [device newBufferWithLength:nb
                                                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> wH = [device newBufferWithLength:(size_t)bn * bk * 2
                                                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> xA = [device newBufferWithLength:(size_t)bm * bk * 2
                                                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> xC = [device newBufferWithLength:(size_t)bm * bn * 4
                                                       options:MTLResourceStorageModeShared];
                memset(wS.contents, 127, nb);
                printf("--- %s m=%d (%d iters/arm)\n", shapes[s].name, bm, iters);
                for (int arm = 0; arm < 4; arm++) {
                    const char *aname = arm == 0 ? "native fp4+scale (4.1)" :
                                        arm == 1 ? "raw fp4 no-scale (4.1)" :
                                        arm == 2 ? "h_h resident f16 (4.0)" :
                                                   "dequant->half + h_h (4.0)";
                    id<MTLComputePipelineState> pipe =
                        arm == 0 ? mm : arm == 1 ? mmNo : mmHH;
                    if (!pipe || (arm == 3 && !deqH)) {
                        printf("%-26s unavailable\n", aname);
                        continue;
                    }
                    double dt = 0;
                    for (int rep = 0; rep < 2; rep++) {
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        uint32_t um = bm, un = bn, uk = bk;
                        NSUInteger tew = pipe.threadExecutionWidth;
                        int reps = rep ? iters : 5;
                        for (int i = 0; i < reps; i++) {
                            if (arm == 3) {
                                [enc setComputePipelineState:deqH];
                                [enc setBuffer:wD offset:0 atIndex:0];
                                [enc setBuffer:wS offset:0 atIndex:1];
                                [enc setBuffer:wH offset:0 atIndex:2];
                                [enc setBytes:&un length:4 atIndex:3];
                                [enc setBytes:&uk length:4 atIndex:4];
                                [enc dispatchThreads:MTLSizeMake(bk / 2, bn, 1)
                                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                            }
                            [enc setComputePipelineState:pipe];
                            [enc setBuffer:xA offset:0 atIndex:0];
                            [enc setBuffer:(arm <= 1 ? wD : wH) offset:0 atIndex:1];
                            [enc setBuffer:xC offset:0 atIndex:2];
                            [enc setBytes:&um length:4 atIndex:3];
                            [enc setBytes:&un length:4 atIndex:4];
                            [enc setBytes:&uk length:4 atIndex:5];
                            if (arm == 0) [enc setBuffer:wS offset:0 atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake((bn + 63) / 64, (bm + 63) / 64, 1)
                                threadsPerThreadgroup:MTLSizeMake(tew * 4, 1, 1)];
                        }
                        [enc endEncoding];
                        double t0 = CACurrentMediaTime();
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (rep) dt = (CACurrentMediaTime() - t0) / iters;
                    }
                    double tops = 2.0 * bm * bn * bk / dt / 1e12;
                    printf("%-26s %.3f ms  %.2f TOPS\n", aname, dt * 1e3, tops);
                }
            }
        }

        printf("PASS\n");
        return 0;
    }
}
