// Synthetic test of the fused gate+up+swiglu kernel (iter-4 Path C).
// Correctness: vs separate-kernel reference (matmul gate + matmul up + swiglu).
// Performance: t_fused vs t_separate over the routed-MoE per-expert workload.
// Decoupled from ds4-bench env-gating so we can validate the kernel directly.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdlib.h>
#include <math.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"\n"
"// Templated fused body — instantiated for several (NR0, NR1, SG) combos so we\n"
"// can sweep small-M tile choices without writing each kernel by hand.\n"
"template <int NR0, int NR1, int NK, int SG>\n"
"inline void fused_gu_swiglu_body(\n"
"        device half *X, device half *gateW, device half *upW, device float *mid,\n"
"        uint M, uint N, uint K, float cv, uint2 tgid) {\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<SG>> mm;\n"
"    auto tX  = tensor(X,     dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tG  = tensor(gateW, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tU  = tensor(upW,   dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tM  = tensor(mid,   dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mX  = tX.slice(0, tgid.y*NR0);\n"
"    auto mGw = tG.slice(tgid.x*NR1, 0);\n"
"    auto mUw = tU.slice(tgid.x*NR1, 0);\n"
"    auto mMo = tM.slice(tgid.x*NR1, tgid.y*NR0);\n"
"    auto cG = mm.template get_destination_cooperative_tensor<decltype(mGw), decltype(mX), float>();\n"
"    auto cU = mm.template get_destination_cooperative_tensor<decltype(mUw), decltype(mX), float>();\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) if (cG.is_valid_element(i)) cG[i]=0.f;\n"
"    for (uint16_t i=0;i<cU.get_capacity();++i) if (cU.is_valid_element(i)) cU[i]=0.f;\n"
"    mm.run(mGw, mX, cG);\n"
"    mm.run(mUw, mX, cU);\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) {\n"
"        if (!cG.is_valid_element(i)) continue;\n"
"        float g = cG[i];\n"
"        if (cv > 0.f) g = clamp(g, -cv, cv);\n"
"        cG[i] = (g / (1.f + exp(-g))) * cU[i];\n"
"    }\n"
"    cG.store(mMo);\n"
"}\n"
"// Variants (NR0 / NR1 / NK / SG):\n"
"kernel void fused_v64_32_4(device half *X [[buffer(0)]], device half *G [[buffer(1)]], device half *U [[buffer(2)]], device float *M_ [[buffer(3)]], constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]], constant float &cv [[buffer(7)]], uint2 t [[threadgroup_position_in_grid]]) { fused_gu_swiglu_body<64,32,32,4>(X,G,U,M_,M,N,K,cv,t); }\n"
"kernel void fused_v32_32_2(device half *X [[buffer(0)]], device half *G [[buffer(1)]], device half *U [[buffer(2)]], device float *M_ [[buffer(3)]], constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]], constant float &cv [[buffer(7)]], uint2 t [[threadgroup_position_in_grid]]) { fused_gu_swiglu_body<32,32,32,2>(X,G,U,M_,M,N,K,cv,t); }\n"
"kernel void fused_v32_16_2(device half *X [[buffer(0)]], device half *G [[buffer(1)]], device half *U [[buffer(2)]], device float *M_ [[buffer(3)]], constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]], constant float &cv [[buffer(7)]], uint2 t [[threadgroup_position_in_grid]]) { fused_gu_swiglu_body<32,16,32,2>(X,G,U,M_,M,N,K,cv,t); }\n"
"kernel void fused_v32_32_1(device half *X [[buffer(0)]], device half *G [[buffer(1)]], device half *U [[buffer(2)]], device float *M_ [[buffer(3)]], constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]], constant float &cv [[buffer(7)]], uint2 t [[threadgroup_position_in_grid]]) { fused_gu_swiglu_body<32,32,32,1>(X,G,U,M_,M,N,K,cv,t); }\n"
"// (legacy single-kernel name kept for the existing main() path)\n"
"kernel void fused_gu_swiglu(\n"
"        device half  *X     [[buffer(0)]],\n"
"        device half  *gateW [[buffer(1)]],\n"
"        device half  *upW   [[buffer(2)]],\n"
"        device float *mid   [[buffer(3)]],\n"
"        constant uint &M    [[buffer(4)]],\n"
"        constant uint &N    [[buffer(5)]],\n"
"        constant uint &K    [[buffer(6)]],\n"
"        constant float &cv  [[buffer(7)]],\n"
"        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
"    constexpr int NR0=64, NR1=32, NK=32;\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<4>> mm;\n"
"    auto tX  = tensor(X,     dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tG  = tensor(gateW, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tU  = tensor(upW,   dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tM  = tensor(mid,   dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mX  = tX.slice(0, tgid.y*NR0);\n"
"    auto mGw = tG.slice(tgid.x*NR1, 0);\n"
"    auto mUw = tU.slice(tgid.x*NR1, 0);\n"
"    auto mMo = tM.slice(tgid.x*NR1, tgid.y*NR0);\n"
"    auto cG = mm.get_destination_cooperative_tensor<decltype(mGw), decltype(mX), float>();\n"
"    auto cU = mm.get_destination_cooperative_tensor<decltype(mUw), decltype(mX), float>();\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) if (cG.is_valid_element(i)) cG[i]=0.f;\n"
"    for (uint16_t i=0;i<cU.get_capacity();++i) if (cU.is_valid_element(i)) cU[i]=0.f;\n"
"    mm.run(mGw, mX, cG);\n"
"    mm.run(mUw, mX, cU);\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) {\n"
"        if (!cG.is_valid_element(i)) continue;\n"
"        float g = cG[i];\n"
"        if (cv > 0.f) g = clamp(g, -cv, cv);\n"
"        cG[i] = (g / (1.f + exp(-g))) * cU[i];\n"
"    }\n"
"    cG.store(mMo);\n"
"}\n"
"\n"
"// REFERENCE: standalone matmul2d (writes a single matrix to device).\n"
"kernel void ref_matmul(\n"
"        device half  *X     [[buffer(0)]],\n"
"        device half  *W     [[buffer(1)]],\n"
"        device float *out   [[buffer(2)]],\n"
"        constant uint &M    [[buffer(3)]],\n"
"        constant uint &N    [[buffer(4)]],\n"
"        constant uint &K    [[buffer(5)]],\n"
"        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
"    constexpr int NR0=64, NR1=32, NK=32;\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<4>> mm;\n"
"    auto tX = tensor(X, dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tW = tensor(W, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tO = tensor(out, dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mX = tX.slice(0, tgid.y*NR0);\n"
"    auto mW = tW.slice(tgid.x*NR1, 0);\n"
"    auto mO = tO.slice(tgid.x*NR1, tgid.y*NR0);\n"
"    auto cO = mm.get_destination_cooperative_tensor<decltype(mW), decltype(mX), float>();\n"
"    for (uint16_t i=0;i<cO.get_capacity();++i) if (cO.is_valid_element(i)) cO[i]=0.f;\n"
"    mm.run(mW, mX, cO);\n"
"    cO.store(mO);\n"
"}\n"
"\n"
"// REFERENCE swiglu (per-element, separate dispatch).\n"
"kernel void ref_swiglu(\n"
"        device float *gate [[buffer(0)]],\n"
"        device float *up   [[buffer(1)]],\n"
"        device float *mid  [[buffer(2)]],\n"
"        constant uint &len [[buffer(3)]],\n"
"        constant float &cv [[buffer(4)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= len) return;\n"
"    float g = gate[tid];\n"
"    if (cv > 0.f) g = clamp(g, -cv, cv);\n"
"    float u = up[tid];\n"
"    mid[tid] = (g / (1.f + exp(-g))) * u;\n"
"}\n";

static double now_ms(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }

static void fill_half(__fp16 *p, size_t n, uint32_t seed){
    uint32_t s = seed;
    for (size_t i=0;i<n;i++){
        s = s*1664525u + 1013904223u;
        float v = ((s>>8) & 0xffff) / 65536.0f - 0.5f;   // [-0.5, 0.5]
        p[i] = (__fp16)(v * 0.1f);  // small magnitudes to avoid SiLU saturation
    }
}

int main(int argc, char **argv){
@autoreleasepool{
    // DS4-Flash-ish per-expert dims; tune via argv.
    uint32_t M = argc>1?atoi(argv[1]):512;    // refs (per-expert tokens)
    uint32_t N = argc>2?atoi(argv[2]):2048;   // mid_dim
    uint32_t K = argc>3?atoi(argv[3]):4096;   // in_dim
    int iters = argc>4?atoi(argv[4]):20;
    printf("dims: M=%u N=%u K=%u  iters=%d\n", M, N, K, iters);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    NSError *err=nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if(!lib){ NSLog(@"compile failed: %@", err); return 1; }
    id<MTLComputePipelineState> pFused = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_gu_swiglu"] error:&err];
    id<MTLComputePipelineState> pMM    = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ref_matmul"] error:&err];
    id<MTLComputePipelineState> pSwi   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ref_swiglu"] error:&err];
    if(!pFused||!pMM||!pSwi){ NSLog(@"pipeline failed: %@", err); return 1; }

    id<MTLBuffer> X     = [dev newBufferWithLength:(NSUInteger)M*K*sizeof(__fp16) options:0];
    id<MTLBuffer> gW    = [dev newBufferWithLength:(NSUInteger)N*K*sizeof(__fp16) options:0];
    id<MTLBuffer> uW    = [dev newBufferWithLength:(NSUInteger)N*K*sizeof(__fp16) options:0];
    id<MTLBuffer> midF  = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    id<MTLBuffer> gateR = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    id<MTLBuffer> upR   = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    id<MTLBuffer> midR  = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    fill_half((__fp16*)X.contents,  (size_t)M*K, 1);
    fill_half((__fp16*)gW.contents, (size_t)N*K, 2);
    fill_half((__fp16*)uW.contents, (size_t)N*K, 3);

    MTLSize tgMM   = MTLSizeMake(128,1,1);
    MTLSize gridMM = MTLSizeMake((N+31)/32, (M+63)/64, 1);
    uint32_t swiLen = M*N; float cv = 7.0f;
    MTLSize tgSwi  = MTLSizeMake(256,1,1);
    MTLSize gridSwi = MTLSizeMake((swiLen+255)/256, 1, 1);

    // === run REFERENCE once (matmul gate, matmul up, swiglu) to fill midR ===
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pMM]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:gW offset:0 atIndex:1]; [e setBuffer:gateR offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e setComputePipelineState:pMM]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:uW offset:0 atIndex:1]; [e setBuffer:upR offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e setComputePipelineState:pSwi]; [e setBuffer:gateR offset:0 atIndex:0]; [e setBuffer:upR offset:0 atIndex:1]; [e setBuffer:midR offset:0 atIndex:2];
        [e setBytes:&swiLen length:4 atIndex:3]; [e setBytes:&cv length:4 atIndex:4];
        [e dispatchThreadgroups:gridSwi threadsPerThreadgroup:tgSwi];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    // === run FUSED once to fill midF ===
    {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pFused]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:gW offset:0 atIndex:1]; [e setBuffer:uW offset:0 atIndex:2]; [e setBuffer:midF offset:0 atIndex:3];
        [e setBytes:&M length:4 atIndex:4];[e setBytes:&N length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&cv length:4 atIndex:7];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    // === correctness: max-abs-diff, mean-abs-diff ===
    float *r=(float*)midR.contents, *f=(float*)midF.contents;
    double max_abs=0.0, sum_abs=0.0, max_rel=0.0; size_t n=(size_t)M*N;
    for (size_t i=0;i<n;i++){
        double d = fabs((double)r[i]-(double)f[i]);
        if (d>max_abs) max_abs=d;
        sum_abs += d;
        double rel = d / (fabs((double)r[i]) + 1e-6);
        if (rel>max_rel) max_rel=rel;
    }
    printf("correctness  max_abs=%.6g  mean_abs=%.6g  max_rel=%.6g  (M*N=%zu)\n",
           max_abs, sum_abs/n, max_rel, n);

    // === timing: separate vs fused ===
    double tSep=1e9, tFused=1e9;
    for (int r=0;r<iters;r++){
        double t0=now_ms();
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pMM]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:gW offset:0 atIndex:1]; [e setBuffer:gateR offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e setComputePipelineState:pMM]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:uW offset:0 atIndex:1]; [e setBuffer:upR offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e setComputePipelineState:pSwi]; [e setBuffer:gateR offset:0 atIndex:0]; [e setBuffer:upR offset:0 atIndex:1]; [e setBuffer:midR offset:0 atIndex:2];
        [e setBytes:&swiLen length:4 atIndex:3]; [e setBytes:&cv length:4 atIndex:4];
        [e dispatchThreadgroups:gridSwi threadsPerThreadgroup:tgSwi];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        tSep = fmin(tSep, now_ms()-t0);
    }
    for (int r=0;r<iters;r++){
        double t0=now_ms();
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pFused]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:gW offset:0 atIndex:1]; [e setBuffer:uW offset:0 atIndex:2]; [e setBuffer:midF offset:0 atIndex:3];
        [e setBytes:&M length:4 atIndex:4];[e setBytes:&N length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&cv length:4 atIndex:7];
        [e dispatchThreadgroups:gridMM threadsPerThreadgroup:tgMM];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        tFused = fmin(tFused, now_ms()-t0);
    }
    printf("timing       separate=%.3fms  fused=%.3fms  speedup=%.3f  (baseline kernel, NR0=64 NR1=32 SG=4)\n",
           tSep, tFused, tSep/tFused);

    // === tile/SG sweep at the given (M, N, K) ===
    typedef struct { const char *name; int NR0; int NR1; int SG; } variant_t;
    variant_t variants[] = {
        { "fused_v64_32_4", 64, 32, 4 },
        { "fused_v32_32_2", 32, 32, 2 },
        { "fused_v32_16_2", 32, 16, 2 },
        { "fused_v32_32_1", 32, 32, 1 },
    };
    printf("--- tile/SG sweep (compare to separate=%.3fms) ---\n", tSep);
    double bestT = 1e9; const char *bestName = "";
    for (size_t vi = 0; vi < sizeof(variants)/sizeof(variants[0]); vi++) {
        variant_t v = variants[vi];
        NSError *verr = nil;
        id<MTLComputePipelineState> pv = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:[NSString stringWithUTF8String:v.name]] error:&verr];
        if (!pv) { printf("  %-18s SKIP (pipeline failed: %s)\n", v.name, verr.localizedDescription.UTF8String ?: "?"); continue; }
        MTLSize tgV   = MTLSizeMake((NSUInteger)v.SG * 32u, 1, 1);
        MTLSize gridV = MTLSizeMake(((NSUInteger)N + (NSUInteger)v.NR1 - 1u) / (NSUInteger)v.NR1,
                                    ((NSUInteger)M + (NSUInteger)v.NR0 - 1u) / (NSUInteger)v.NR0, 1);
        double tBest = 1e9;
        for (int r = 0; r < iters; r++) {
            double t0 = now_ms();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pv];
            [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:gW offset:0 atIndex:1];
            [e setBuffer:uW offset:0 atIndex:2]; [e setBuffer:midF offset:0 atIndex:3];
            [e setBytes:&M length:4 atIndex:4]; [e setBytes:&N length:4 atIndex:5];
            [e setBytes:&K length:4 atIndex:6]; [e setBytes:&cv length:4 atIndex:7];
            [e dispatchThreadgroups:gridV threadsPerThreadgroup:tgV];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            tBest = fmin(tBest, now_ms() - t0);
        }
        printf("  %-18s tg=%lu grid=(%lu,%lu)  %.3fms  speedup=%.3f\n",
               v.name, (unsigned long)tgV.width, (unsigned long)gridV.width, (unsigned long)gridV.height,
               tBest, tSep/tBest);
        if (tBest < bestT) { bestT = tBest; bestName = v.name; }
    }
    printf("--- best: %s at %.3fms (speedup vs separate = %.3f) ---\n", bestName, bestT, tSep/bestT);
}
return 0;}
