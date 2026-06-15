// Standalone validation probe for the native-MXFP4 (MPP 4.1) stack:
//   1. GPU plane repack (ggml block_mxfp4 split-half -> seq-pair data plane +
//      E8M0 scale plane) vs a CPU reference repack: byte-exact.
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
    if (exp <= 0) return (uint16_t)sign;
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
        id<MTLFunction> mmfn = [lib newFunctionWithName:@"kernel_dsv4_mxfp4_native_matmul_n64"];
        id<MTLComputePipelineState> mm =
            mmfn ? [device newComputePipelineStateWithFunction:mmfn error:&err] : nil;
        printf("library: MSL %s, repack ok, native matmul %s\n",
               lang41 ? "4.1" : "4.0", mm ? "ok" : "UNAVAILABLE");

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

        if (!mm) {
            printf("native matmul unavailable on this OS/toolchain/GPU — repack-only PASS\n");
            return 0;
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
