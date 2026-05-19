#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef _Float16 fp16_t;

typedef struct {
    int M;
    int N;
    int K;
    int warmup;
    int iters;
    int mlp;
    int verify;
} cfg_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *mpp_source(void) {
    return
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "using namespace mpp::tensor_ops;\n"
        "\n"
        "template <typename AT, typename BT, typename CT>\n"
        "inline void run_tile(device AT *A, device BT *B, device CT *C,\n"
        "                     constant uint &M, constant uint &N, constant uint &K,\n"
        "                     uint2 tgid) {\n"
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
        "kernel void mpp_h_i8_h(device half *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device half *C [[buffer(2)]],\n"
        "                       constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                       uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<half, int8_t, half>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void mpp_h_i8_f(device half *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
        "                       constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                       uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<half, int8_t, float>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void mpp_i8_h_h(device int8_t *A [[buffer(0)]], device half *B [[buffer(1)]], device half *C [[buffer(2)]],\n"
        "                       constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                       uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<int8_t, half, half>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void mpp_i8_h_f(device int8_t *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
        "                       constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                       uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<int8_t, half, float>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void mpp_i8_i8_i32(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device int32_t *C [[buffer(2)]],\n"
        "                          constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                          uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<int8_t, int8_t, int32_t>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void mpp_f_i8_f(device float *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
        "                       constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
        "                       uint2 tgid [[threadgroup_position_in_grid]]) {\n"
        "    run_tile<float, int8_t, float>(A, B, C, M, N, K, tgid);\n"
        "}\n"
        "kernel void swiglu_f32(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],\n"
        "                       device float *mid [[buffer(2)]], constant uint &n [[buffer(3)]],\n"
        "                       uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid >= n) return;\n"
        "    float g = gate[tid];\n"
        "    float s = g / (1.0f + exp(-g));\n"
        "    mid[tid] = s * up[tid];\n"
        "}\n";
}

static id<MTLComputePipelineState> make_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "missing function %s\n", name.UTF8String);
        exit(2);
    }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) {
        fprintf(stderr, "pipeline %s failed: %s\n", name.UTF8String, err.localizedDescription.UTF8String);
        exit(2);
    }
    return ps;
}

static void fill_i8(id<MTLBuffer> b) {
    int8_t *p = b.contents;
    for (NSUInteger i = 0; i < b.length; i++) p[i] = (int8_t)((int)(i * 13u + 7u) % 127 - 63);
}

static void fill_h(id<MTLBuffer> b) {
    fp16_t *p = b.contents;
    NSUInteger n = b.length / sizeof(fp16_t);
    for (NSUInteger i = 0; i < n; i++) p[i] = (fp16_t)(((int)(i * 17u + 3u) % 97 - 48) * 0.01f);
}

static void fill_h_verify(id<MTLBuffer> b) {
    fp16_t *p = b.contents;
    NSUInteger n = b.length / sizeof(fp16_t);
    for (NSUInteger i = 0; i < n; i++) p[i] = (fp16_t)(((int)(i * 17u + 3u) % 97 - 48) * 0.0001f);
}

static double run_one(
        id<MTLCommandQueue> q,
        id<MTLComputePipelineState> ps,
        id<MTLBuffer> A,
        id<MTLBuffer> B,
        id<MTLBuffer> C,
        const cfg_t *cfg,
        const char *name,
        size_t c_elem_size) {
    (void)c_elem_size;
    const uint32_t M = (uint32_t)cfg->M;
    const uint32_t N = (uint32_t)cfg->N;
    const uint32_t K = (uint32_t)cfg->K;
    MTLSize tg = MTLSizeMake((NSUInteger)ps.threadExecutionWidth * 4, 1, 1);
    MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 63) / 64, 1);

    for (int phase = 0; phase < 2; phase++) {
        int iters = phase == 0 ? cfg->warmup : cfg->iters;
        double t0 = now_s();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int i = 0; i < iters; i++) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&K length:sizeof(K) atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "%s command buffer failed: %s\n", name, cb.error.localizedDescription.UTF8String);
            exit(2);
        }
        if (phase == 1) {
            double ms = (now_s() - t0) * 1000.0 / (double)iters;
            double gflops = (2.0 * (double)cfg->M * (double)cfg->N * (double)cfg->K) / (ms * 1e6);
            printf("%s,%d,%d,%d,%d,%.6f,%.3f\n", name, cfg->M, cfg->N, cfg->K, cfg->iters, ms, gflops);
            return ms;
        }
    }
    return NAN;
}

static double run_mlp(
        id<MTLCommandQueue> q,
        id<MTLComputePipelineState> gate_ps,
        id<MTLComputePipelineState> down_ps,
        id<MTLComputePipelineState> swiglu_ps,
        id<MTLBuffer> X_h,
        id<MTLBuffer> W_gate_i8,
        id<MTLBuffer> W_up_i8,
        id<MTLBuffer> W_down_i8,
        id<MTLBuffer> gate_f,
        id<MTLBuffer> up_f,
        id<MTLBuffer> mid_f,
        id<MTLBuffer> out_f,
        const cfg_t *cfg) {
    const uint32_t M = (uint32_t)cfg->M;
    const uint32_t H = (uint32_t)cfg->K;
    const uint32_t I = (uint32_t)cfg->N;
    const uint32_t mid_elems = M * I;
    MTLSize gate_tg = MTLSizeMake((NSUInteger)gate_ps.threadExecutionWidth * 4, 1, 1);
    MTLSize down_tg = MTLSizeMake((NSUInteger)down_ps.threadExecutionWidth * 4, 1, 1);
    MTLSize gate_grid = MTLSizeMake((I + 31) / 32, (M + 63) / 64, 1);
    MTLSize down_grid = MTLSizeMake((H + 31) / 32, (M + 63) / 64, 1);
    NSUInteger sw_nth = swiglu_ps.maxTotalThreadsPerThreadgroup;
    if (sw_nth > 256) sw_nth = 256;
    if (sw_nth == 0) sw_nth = 1;

    for (int phase = 0; phase < 2; phase++) {
        int iters = phase == 0 ? cfg->warmup : cfg->iters;
        double t0 = now_s();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int i = 0; i < iters; i++) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:gate_ps];
            [enc setBuffer:X_h offset:0 atIndex:0];
            [enc setBuffer:W_gate_i8 offset:0 atIndex:1];
            [enc setBuffer:gate_f offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&I length:sizeof(I) atIndex:4];
            [enc setBytes:&H length:sizeof(H) atIndex:5];
            [enc dispatchThreadgroups:gate_grid threadsPerThreadgroup:gate_tg];

            [enc setComputePipelineState:gate_ps];
            [enc setBuffer:X_h offset:0 atIndex:0];
            [enc setBuffer:W_up_i8 offset:0 atIndex:1];
            [enc setBuffer:up_f offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&I length:sizeof(I) atIndex:4];
            [enc setBytes:&H length:sizeof(H) atIndex:5];
            [enc dispatchThreadgroups:gate_grid threadsPerThreadgroup:gate_tg];

            [enc setComputePipelineState:swiglu_ps];
            [enc setBuffer:gate_f offset:0 atIndex:0];
            [enc setBuffer:up_f offset:0 atIndex:1];
            [enc setBuffer:mid_f offset:0 atIndex:2];
            [enc setBytes:&mid_elems length:sizeof(mid_elems) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)mid_elems + sw_nth - 1u) / sw_nth, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(sw_nth, 1, 1)];

            [enc setComputePipelineState:down_ps];
            [enc setBuffer:mid_f offset:0 atIndex:0];
            [enc setBuffer:W_down_i8 offset:0 atIndex:1];
            [enc setBuffer:out_f offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&H length:sizeof(H) atIndex:4];
            [enc setBytes:&I length:sizeof(I) atIndex:5];
            [enc dispatchThreadgroups:down_grid threadsPerThreadgroup:down_tg];
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "mpp_mlp command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
            exit(2);
        }
        if (phase == 1) {
            double ms = (now_s() - t0) * 1000.0 / (double)iters;
            double gflops = (6.0 * (double)cfg->M * (double)cfg->K * (double)cfg->N) / (ms * 1e6);
            printf("mpp_mlp_h_i8_f_f_i8_f,%d,%d,%d,%d,%.6f,%.3f\n",
                   cfg->M, cfg->N, cfg->K, cfg->iters, ms, gflops);
            return ms;
        }
    }
    return NAN;
}

static int verify_mlp_cpu(
        const fp16_t *x,
        const int8_t *w_gate,
        const int8_t *w_up,
        const int8_t *w_down,
        const float *gpu_out,
        const cfg_t *cfg) {
    const int M = cfg->M;
    const int H = cfg->K;
    const int I = cfg->N;
    float *gate = calloc((size_t)M * (size_t)I, sizeof(float));
    float *up = calloc((size_t)M * (size_t)I, sizeof(float));
    float *mid = calloc((size_t)M * (size_t)I, sizeof(float));
    float *ref = calloc((size_t)M * (size_t)H, sizeof(float));
    if (!gate || !up || !mid || !ref) {
        fprintf(stderr, "verify allocation failed\n");
        free(gate); free(up); free(mid); free(ref);
        return 0;
    }
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < I; i++) {
            double gs = 0.0;
            double us = 0.0;
            for (int h = 0; h < H; h++) {
                const float xv = (float)x[(size_t)m * (size_t)H + (size_t)h];
                gs += (double)xv * (double)w_gate[(size_t)h * (size_t)I + (size_t)i];
                us += (double)xv * (double)w_up[(size_t)h * (size_t)I + (size_t)i];
            }
            float gf = (float)gs;
            float uf = (float)us;
            gate[(size_t)m * (size_t)I + (size_t)i] = gf;
            up[(size_t)m * (size_t)I + (size_t)i] = uf;
            mid[(size_t)m * (size_t)I + (size_t)i] = (gf / (1.0f + expf(-gf))) * uf;
        }
    }
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H; h++) {
            double s = 0.0;
            for (int i = 0; i < I; i++) {
                s += (double)mid[(size_t)m * (size_t)I + (size_t)i] *
                     (double)w_down[(size_t)i * (size_t)H + (size_t)h];
            }
            ref[(size_t)m * (size_t)H + (size_t)h] = (float)s;
        }
    }

    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;
    const size_t n = (size_t)M * (size_t)H;
    for (size_t idx = 0; idx < n; idx++) {
        const double a = fabs((double)gpu_out[idx] - (double)ref[idx]);
        const double denom = fmax(fabs((double)ref[idx]), 1e-6);
        const double r = a / denom;
        if (a > max_abs) max_abs = a;
        if (r > max_rel) max_rel = r;
        mean_abs += a;
    }
    mean_abs /= (double)n;
    printf("verify_mlp,M=%d,N=%d,K=%d,max_abs=%.9g,mean_abs=%.9g,max_rel=%.9g\n",
           M, I, H, max_abs, mean_abs, max_rel);
    free(gate); free(up); free(mid); free(ref);
    return max_abs < 5e-3 && max_rel < 5e-2;
}

int main(int argc, char **argv) {
    cfg_t cfg = {.M = 128, .N = 2048, .K = 4096, .warmup = 10, .iters = 200, .mlp = 0, .verify = 0};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shape") && i + 3 < argc) {
            cfg.M = atoi(argv[++i]);
            cfg.K = atoi(argv[++i]);
            cfg.N = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            cfg.warmup = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mlp")) {
            cfg.mlp = 1;
        } else if (!strcmp(argv[i], "--verify")) {
            cfg.verify = 1;
            cfg.mlp = 1;
            if (cfg.iters > 1) cfg.iters = 1;
            cfg.warmup = 0;
        } else {
            fprintf(stderr, "usage: %s [--shape M K N] [--warmup N] [--iters N] [--mlp] [--verify]\n", argv[0]);
            return 2;
        }
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError *err = nil;
        id<MTLLibrary> lib =
            [dev newLibraryWithSource:[NSString stringWithUTF8String:mpp_source()]
                               options:opts
                                 error:&err];
        if (!lib) {
            fprintf(stderr, "library compile failed: %s\n", err.localizedDescription.UTF8String);
            return 2;
        }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLComputePipelineState> h_i8_h = make_pipe(dev, lib, @"mpp_h_i8_h");
        id<MTLComputePipelineState> h_i8_f = make_pipe(dev, lib, @"mpp_h_i8_f");
        id<MTLComputePipelineState> i8_h_h = make_pipe(dev, lib, @"mpp_i8_h_h");
        id<MTLComputePipelineState> i8_h_f = make_pipe(dev, lib, @"mpp_i8_h_f");
        id<MTLComputePipelineState> i8_i8_i32 = make_pipe(dev, lib, @"mpp_i8_i8_i32");
        id<MTLComputePipelineState> f_i8_f = make_pipe(dev, lib, @"mpp_f_i8_f");
        id<MTLComputePipelineState> swiglu = make_pipe(dev, lib, @"swiglu_f32");

        NSUInteger a_h_bytes = (NSUInteger)cfg.M * (NSUInteger)cfg.K * sizeof(fp16_t);
        NSUInteger a_i8_bytes = (NSUInteger)cfg.M * (NSUInteger)cfg.K;
        NSUInteger b_h_bytes = (NSUInteger)cfg.K * (NSUInteger)cfg.N * sizeof(fp16_t);
        NSUInteger b_i8_bytes = (NSUInteger)cfg.K * (NSUInteger)cfg.N;
        NSUInteger c_h_bytes = (NSUInteger)cfg.M * (NSUInteger)cfg.N * sizeof(fp16_t);
        NSUInteger c_f_bytes = (NSUInteger)cfg.M * (NSUInteger)cfg.N * sizeof(float);
        NSUInteger c_i32_bytes = (NSUInteger)cfg.M * (NSUInteger)cfg.N * sizeof(int32_t);

        id<MTLBuffer> A_h = [dev newBufferWithLength:a_h_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> A_i8 = [dev newBufferWithLength:a_i8_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> B_h = [dev newBufferWithLength:b_h_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> B_i8 = [dev newBufferWithLength:b_i8_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> C_h = [dev newBufferWithLength:c_h_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> C_f = [dev newBufferWithLength:c_f_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> C_i32 = [dev newBufferWithLength:c_i32_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> W_up_i8 = [dev newBufferWithLength:b_i8_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> W_down_i8 = [dev newBufferWithLength:(NSUInteger)cfg.N * (NSUInteger)cfg.K options:MTLResourceStorageModeShared];
        id<MTLBuffer> gate_f = [dev newBufferWithLength:c_f_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> up_f = [dev newBufferWithLength:c_f_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> mid_f = [dev newBufferWithLength:c_f_bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out_f =
            [dev newBufferWithLength:(NSUInteger)cfg.M * (NSUInteger)cfg.K * sizeof(float)
                              options:(cfg.verify ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate)];
        if (!A_h || !A_i8 || !B_h || !B_i8 || !C_h || !C_f || !C_i32 ||
            !W_up_i8 || !W_down_i8 || !gate_f || !up_f || !mid_f || !out_f) {
            fprintf(stderr, "buffer allocation failed\n");
            return 2;
        }
        if (cfg.verify) fill_h_verify(A_h);
        else fill_h(A_h);
        fill_i8(A_i8);
        fill_h(B_h);
        fill_i8(B_i8);
        fill_i8(W_up_i8);
        fill_i8(W_down_i8);

        printf("backend,M,N,K,iters,ms,gflops\n");
        if (cfg.mlp) {
            run_mlp(q, h_i8_f, f_i8_f, swiglu, A_h, B_i8, W_up_i8, W_down_i8,
                    gate_f, up_f, mid_f, out_f, &cfg);
            if (cfg.verify) {
                int ok = verify_mlp_cpu((const fp16_t *)A_h.contents,
                                        (const int8_t *)B_i8.contents,
                                        (const int8_t *)W_up_i8.contents,
                                        (const int8_t *)W_down_i8.contents,
                                        (const float *)out_f.contents,
                                        &cfg);
                return ok ? 0 : 1;
            }
            return 0;
        }
        run_one(q, h_i8_h, A_h, B_i8, C_h, &cfg, "mpp_half_int8_half", sizeof(fp16_t));
        run_one(q, h_i8_f, A_h, B_i8, C_f, &cfg, "mpp_half_int8_float", sizeof(float));
        run_one(q, i8_h_h, A_i8, B_h, C_h, &cfg, "mpp_int8_half_half", sizeof(fp16_t));
        run_one(q, i8_h_f, A_i8, B_h, C_f, &cfg, "mpp_int8_half_float", sizeof(float));
        run_one(q, i8_i8_i32, A_i8, B_i8, C_i32, &cfg, "mpp_int8_int8_int32", sizeof(int32_t));
    }
    return 0;
}
