// Probe B: int8 cooperative-int32 → float type-juggling synthetic.
//
// Goal: isolate the specific risk in the full iq2+iq2+swiglu fused kernel —
// matmul2d's cooperative tensor for i8×i8 is int32, but SwiGLU needs float.
// There's no "cooperative-int32 → cooperative-float" primitive in MPP that I've
// found. The workaround: cT.store(threadgroup int32 buffer); barrier; per-thread
// read int32 → qscale*float → (eventually SwiGLU) → write float to device.
//
// This probe tests just that pattern (single matmul, no SwiGLU) so we know
// whether the type-juggling works + how much it costs, BEFORE writing the
// full ~200-line fused gate+up+swiglu kernel that would also need it.
//
// Compares two kernels of identical math:
//   FUSED   : i8×i8 matmul2d → cooperative int32 → tg int32 → per-thread
//             convert to float (×qscale) → device float (single dispatch)
//   SEPARATE: i8×i8 matmul2d → device int32 (cT.store-direct) ;
//             then a second kernel reads int32, multiplies by qscale, writes float
//
// Correctness gate: max-abs-diff(FUSED, SEPARATE) ≈ 0 (same math).
// Timing: t_fused vs (t_mm + t_convert). Fused should win on dispatch + I/O.
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
"// ===== FUSED: i8 matmul + in-kernel int32→float convert =====\n"
"kernel void mm_i8_to_float_fused(\n"
"        device int8_t *A   [[buffer(0)]],\n"
"        device int8_t *B   [[buffer(1)]],\n"
"        device float  *C   [[buffer(2)]],\n"
"        constant uint  &M  [[buffer(3)]],\n"
"        constant uint  &N  [[buffer(4)]],\n"
"        constant uint  &K  [[buffer(5)]],\n"
"        constant float &qs [[buffer(6)]],\n"
"        threadgroup char *tg [[threadgroup(0)]],\n"
"        uint2 tgid [[threadgroup_position_in_grid]],\n"
"        uint  tidx [[thread_index_in_threadgroup]]) {\n"
"    constexpr int NR0=32, NR1=64, NK=256;\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<4>> mm;\n"
"\n"
"    const uint m0 = tgid.y * NR0;\n"
"    const uint n0 = tgid.x * NR1;\n"
"    if (m0 >= M || n0 >= N) return;\n"
"    const uint rows = min((uint)NR0, M - m0);\n"
"    const uint cols = min((uint)NR1, N - n0);\n"
"\n"
"    auto tA = tensor(A, dextents<int32_t,2>{(int32_t)K, (int32_t)M}, array<int32_t,2>{1, (int32_t)K});\n"
"    auto tB = tensor(B, dextents<int32_t,2>{(int32_t)N, (int32_t)K}, array<int32_t,2>{1, (int32_t)N});\n"
"\n"
"    auto mA = tA.slice(0, m0);\n"
"    auto mB = tB.slice(n0, 0);\n"
"\n"
"    auto cT = mm.get_destination_cooperative_tensor<decltype(mB), decltype(mA), int32_t>();\n"
"    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;\n"
"    mm.run(mB, mA, cT);\n"
"\n"
"    // ====== the type-juggle: cooperative int32 → tg int32 → float device ======\n"
"    threadgroup int32_t *tg_i32 = (threadgroup int32_t *)tg;   // [NR0 x NR1] int32\n"
"    auto tT = tensor(tg_i32, dextents<int32_t,2>{NR1, NR0}, array<int32_t,2>{1, NR1});\n"
"    auto mT = tT.slice(0, 0);\n"
"    cT.store(mT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Per-thread read tg int32, apply qscale, write float to device.\n"
"    // 128 threads / NR0*NR1 = 128/(32*64=2048) → 16 elems/thread.\n"
"    const uint total = NR0 * NR1;\n"
"    for (uint p = tidx; p < total; p += 128u) {\n"
"        const uint r = p / NR1;\n"
"        const uint c = p % NR1;\n"
"        if (r < rows && c < cols) {\n"
"            C[(m0 + r) * N + (n0 + c)] = (float)tg_i32[r * NR1 + c] * qs;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// ===== REFERENCE: i8 matmul → device int32 (single kernel) =====\n"
"kernel void mm_i8_to_int32(\n"
"        device int8_t *A   [[buffer(0)]],\n"
"        device int8_t *B   [[buffer(1)]],\n"
"        device int32_t *C  [[buffer(2)]],\n"
"        constant uint  &M  [[buffer(3)]],\n"
"        constant uint  &N  [[buffer(4)]],\n"
"        constant uint  &K  [[buffer(5)]],\n"
"        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
"    constexpr int NR0=32, NR1=64, NK=256;\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<4>> mm;\n"
"    const uint m0 = tgid.y * NR0, n0 = tgid.x * NR1;\n"
"    if (m0 >= M || n0 >= N) return;\n"
"    const uint rows = min((uint)NR0, M - m0);\n"
"    auto tA = tensor(A, dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tB = tensor(B, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tC = tensor(C, dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mA = tA.slice(0, m0), mB = tB.slice(n0, 0);\n"
"    auto mC = tC.slice(n0, m0);\n"
"    auto cT = mm.get_destination_cooperative_tensor<decltype(mB), decltype(mA), int32_t>();\n"
"    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;\n"
"    mm.run(mB, mA, cT);\n"
"    (void)rows;\n"
"    cT.store(mC);\n"
"}\n"
"\n"
"// ===== REFERENCE convert: read int32 device → write float device =====\n"
"kernel void convert_i32_to_float(\n"
"        device int32_t *src [[buffer(0)]],\n"
"        device float   *dst [[buffer(1)]],\n"
"        constant uint   &n  [[buffer(2)]],\n"
"        constant float  &qs [[buffer(3)]],\n"
"        uint tid [[thread_position_in_grid]]) {\n"
"    if (tid >= n) return;\n"
"    dst[tid] = (float)src[tid] * qs;\n"
"}\n";

static double now_ms(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }

static void fill_i8(int8_t *p, size_t n, uint32_t seed){
    uint32_t s = seed;
    for (size_t i=0;i<n;i++){
        s = s*1664525u + 1013904223u;
        p[i] = (int8_t)(((s>>8) & 0xff) - 128);   // [-128, 127]
    }
}

int main(int argc, char **argv){
@autoreleasepool{
    uint32_t M = argc>1?atoi(argv[1]):512;
    uint32_t N = argc>2?atoi(argv[2]):2048;
    uint32_t K = argc>3?atoi(argv[3]):4096;
    int iters = argc>4?atoi(argv[4]):30;
    printf("Probe B (minimal) — int8 cooperative-int32 → float type-juggle\n");
    printf("dims: M=%u N=%u K=%u  iters=%d\n\n", M, N, K, iters);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    NSError *err=nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if(!lib){ NSLog(@"compile failed: %@", err); return 1; }
    id<MTLComputePipelineState> pFused = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_i8_to_float_fused"] error:&err];
    id<MTLComputePipelineState> pMM    = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_i8_to_int32"] error:&err];
    id<MTLComputePipelineState> pConv  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"convert_i32_to_float"] error:&err];
    if(!pFused||!pMM||!pConv){ NSLog(@"pipeline failed: %@", err); return 1; }

    id<MTLBuffer> A     = [dev newBufferWithLength:(NSUInteger)M*K*sizeof(int8_t)  options:0];
    id<MTLBuffer> B     = [dev newBufferWithLength:(NSUInteger)N*K*sizeof(int8_t)  options:0];
    id<MTLBuffer> outF1 = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float)   options:0];
    id<MTLBuffer> outI  = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(int32_t) options:0];
    id<MTLBuffer> outF2 = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float)   options:0];
    fill_i8((int8_t*)A.contents, (size_t)M*K, 11);
    fill_i8((int8_t*)B.contents, (size_t)N*K, 22);

    const float qs = 1.0f / (32.0f * 64.0f);   // mock x_qscale * w_qscale typical
    MTLSize tgMM = MTLSizeMake(128, 1, 1);
    MTLSize gMM  = MTLSizeMake((N+63)/64, (M+31)/32, 1);
    const uint32_t total = M * N;
    MTLSize tgC  = MTLSizeMake(256, 1, 1);
    MTLSize gC   = MTLSizeMake((total+255)/256, 1, 1);
    const NSUInteger tg_bytes = (NSUInteger)32 * 64 * sizeof(int32_t);   // NR0*NR1 int32

    void(^run_fused)() = ^{
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pFused];
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:outF1 offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3]; [e setBytes:&N length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
        [e setBytes:&qs length:4 atIndex:6];
        [e setThreadgroupMemoryLength:tg_bytes atIndex:0];
        [e dispatchThreadgroups:gMM threadsPerThreadgroup:tgMM];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    };
    void(^run_separate)() = ^{
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pMM];
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:outI offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3]; [e setBytes:&N length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
        [e dispatchThreadgroups:gMM threadsPerThreadgroup:tgMM];
        [e setComputePipelineState:pConv];
        [e setBuffer:outI offset:0 atIndex:0]; [e setBuffer:outF2 offset:0 atIndex:1];
        [e setBytes:&total length:4 atIndex:2]; [e setBytes:&qs length:4 atIndex:3];
        [e dispatchThreadgroups:gC threadsPerThreadgroup:tgC];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    };

    // warmup + correctness
    run_fused(); run_separate();
    float *f = (float*)outF1.contents, *r = (float*)outF2.contents;
    double max_abs = 0.0; size_t n = (size_t)M*N;
    for (size_t i=0;i<n;i++){ double d=fabs((double)f[i]-(double)r[i]); if (d>max_abs) max_abs=d; }
    printf("correctness  max_abs(fused vs separate) = %.6g  (must be ≈0 — same math)\n", max_abs);

    double tF = 1e9, tS = 1e9;
    for (int i=0;i<iters;i++){ double t0=now_ms(); run_fused();    tF = fmin(tF, now_ms()-t0); }
    for (int i=0;i<iters;i++){ double t0=now_ms(); run_separate(); tS = fmin(tS, now_ms()-t0); }
    printf("timing       fused=%.3fms  separate(mm+convert)=%.3fms  speedup=%.3f\n", tF, tS, tS/tF);
    printf("verdict: %s\n",
           max_abs < 1e-3 ? "type-juggle WORKS — full fused iq2+iq2+swiglu is buildable on this pattern"
                          : "type-juggle BROKE — fused iq2+iq2+swiglu needs a different mechanism");
}
return 0;}
