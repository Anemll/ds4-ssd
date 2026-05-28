// Probe A: counted-indirect synthetic for h_h_f fused gate+up+swiglu.
// Goal: isolate the per-tile id-map indirection cost (the machinery you'd add
// when wiring the fused kernel into the non-dedup counted-indirect path).
//
// Method:
//   - One kernel that ALWAYS takes an id_map and uses id_map[tile_idx] to pick
//     which expert's weights (out of n_experts banks) to multiply against.
//   - "indirection-off" baseline: id_map filled with all-zeros → every tile reads
//     expert 0 → same memory pattern as the existing fused kernel. Isolates the
//     id_map read + offset arithmetic overhead.
//   - "indirection-on": id_map filled with varying expert indices (round-robin)
//     → tiles read different banks → exercises the real id-mapped path.
//   - Correctness: with id_map all-zeros, output must equal the original fused
//     kernel's output (sanity).
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
"// Fused gate+up+swiglu with per-tile id-mapped weight selection.\n"
"// Layout: gateW/upW are [E*N x K] — E experts concatenated along the N axis.\n"
"// Each output tile picks its expert via id_map[tile_index] and slices into\n"
"// the (expert*N) row range of gateW/upW. Output mid is [M x N] (single expert\n"
"// per output region — the test simulates 'this output tile belongs to expert e').\n"
"kernel void fused_idmap(\n"
"        device half  *X       [[buffer(0)]],\n"
"        device half  *gateW   [[buffer(1)]],\n"
"        device half  *upW     [[buffer(2)]],\n"
"        device float *mid     [[buffer(3)]],\n"
"        device uint  *id_map  [[buffer(4)]],\n"
"        constant uint &M      [[buffer(5)]],\n"
"        constant uint &N      [[buffer(6)]],\n"
"        constant uint &K      [[buffer(7)]],\n"
"        constant uint &nGridX [[buffer(8)]],\n"
"        constant float &cv    [[buffer(9)]],\n"
"        uint2 tgid [[threadgroup_position_in_grid]]) {\n"
"    constexpr int NR0=32, NR1=32, NK=32;\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<1>> mm;\n"
"\n"
"    const uint tile_idx = tgid.y * nGridX + tgid.x;\n"
"    const uint expert_id = id_map[tile_idx];\n"
"    const uint base_n = expert_id * N;\n"
"\n"
"    // Weights spans E*N rows along the N axis; gate/up indexed by base_n + col.\n"
"    auto tX  = tensor(X,     dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    // Pretend gateW has only N cols visible to this tile (slice starts at base_n + tgid.x*NR1).\n"
"    // We can't change tensor extents at runtime, so describe gateW with N=E*N and slice by base_n.\n"
"    auto tG  = tensor(gateW, dextents<int32_t,2>{(int32_t)(N * 64u),(int32_t)K}, array<int32_t,2>{1,(int32_t)(N * 64u)});\n"
"    auto tU  = tensor(upW,   dextents<int32_t,2>{(int32_t)(N * 64u),(int32_t)K}, array<int32_t,2>{1,(int32_t)(N * 64u)});\n"
"    auto tM  = tensor(mid,   dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"\n"
"    auto mX   = tX.slice(0, tgid.y*NR0);\n"
"    auto mGw  = tG.slice(base_n + tgid.x*NR1, 0);\n"
"    auto mUw  = tU.slice(base_n + tgid.x*NR1, 0);\n"
"    auto mMo  = tM.slice(tgid.x*NR1, tgid.y*NR0);\n"
"\n"
"    auto cG = mm.get_destination_cooperative_tensor<decltype(mGw), decltype(mX), float>();\n"
"    auto cU = mm.get_destination_cooperative_tensor<decltype(mUw), decltype(mX), float>();\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) if (cG.is_valid_element(i)) cG[i]=0.f;\n"
"    for (uint16_t i=0;i<cU.get_capacity();++i) if (cU.is_valid_element(i)) cU[i]=0.f;\n"
"\n"
"    mm.run(mGw, mX, cG);\n"
"    mm.run(mUw, mX, cU);\n"
"\n"
"    for (uint16_t i=0;i<cG.get_capacity();++i) {\n"
"        if (!cG.is_valid_element(i)) continue;\n"
"        float g = cG[i];\n"
"        if (cv > 0.f) g = clamp(g, -cv, cv);\n"
"        cG[i] = (g / (1.f + exp(-g))) * cU[i];\n"
"    }\n"
"    cG.store(mMo);\n"
"}\n";

static double now_ms(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }

static void fill_half(__fp16 *p, size_t n, uint32_t seed){
    uint32_t s = seed;
    for (size_t i=0;i<n;i++){
        s = s*1664525u + 1013904223u;
        float v = ((s>>8) & 0xffff) / 65536.0f - 0.5f;
        p[i] = (__fp16)(v * 0.1f);
    }
}

int main(int argc, char **argv){
@autoreleasepool{
    uint32_t M = argc>1?atoi(argv[1]):512;
    uint32_t N = argc>2?atoi(argv[2]):2048;
    uint32_t K = argc>3?atoi(argv[3]):4096;
    int iters = argc>4?atoi(argv[4]):30;
    const uint32_t E = 64;   // expert banks in the weights buffer (test fixture)
    printf("Probe A — counted-indirect synthetic (id-map per-tile overhead)\n");
    printf("dims: M=%u N=%u K=%u  E=%u  iters=%d\n\n", M, N, K, E, iters);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    NSError *err=nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if(!lib){ NSLog(@"compile failed: %@", err); return 1; }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_idmap"] error:&err];
    if(!p){ NSLog(@"pipeline failed: %@", err); return 1; }

    // gateW/upW carry E expert banks (each [N x K]). Total N axis = E*N. The %_64 in
    // the kernel's tensor extent is a hack so the descriptor compiles for E=64 — the
    // kernel still slices via base_n which lands inside the buffer.
    id<MTLBuffer> X    = [dev newBufferWithLength:(NSUInteger)M*K*sizeof(__fp16) options:0];
    id<MTLBuffer> gW   = [dev newBufferWithLength:(NSUInteger)E*N*K*sizeof(__fp16) options:0];
    id<MTLBuffer> uW   = [dev newBufferWithLength:(NSUInteger)E*N*K*sizeof(__fp16) options:0];
    id<MTLBuffer> midA = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    id<MTLBuffer> midB = [dev newBufferWithLength:(NSUInteger)M*N*sizeof(float) options:0];
    fill_half((__fp16*)X.contents,  (size_t)M*K, 1);
    fill_half((__fp16*)gW.contents, (size_t)E*N*K, 2);
    fill_half((__fp16*)uW.contents, (size_t)E*N*K, 3);

    const uint32_t nGridX = (N + 31u) / 32u;
    const uint32_t nGridY = (M + 31u) / 32u;
    MTLSize tg   = MTLSizeMake(32, 1, 1);
    MTLSize grid = MTLSizeMake(nGridX, nGridY, 1);
    const NSUInteger nTiles = (NSUInteger)nGridX * nGridY;

    // Two id-map buffers: all-zeros (no real indirection — every tile reads expert 0)
    // and round-robin (every tile reads a different expert in [0, E)).
    id<MTLBuffer> idmap0 = [dev newBufferWithLength:nTiles*sizeof(uint) options:0];
    id<MTLBuffer> idmapR = [dev newBufferWithLength:nTiles*sizeof(uint) options:0];
    memset(idmap0.contents, 0, nTiles*sizeof(uint));
    uint *rrptr = (uint*)idmapR.contents;
    for (NSUInteger i = 0; i < nTiles; i++) rrptr[i] = (uint)(i % E);

    float cv = 7.0f;

    void(^run)(id<MTLBuffer>, id<MTLBuffer>) = ^(id<MTLBuffer> midBuf, id<MTLBuffer> idmap){
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:X offset:0 atIndex:0];
        [e setBuffer:gW offset:0 atIndex:1];
        [e setBuffer:uW offset:0 atIndex:2];
        [e setBuffer:midBuf offset:0 atIndex:3];
        [e setBuffer:idmap offset:0 atIndex:4];
        [e setBytes:&M length:4 atIndex:5];
        [e setBytes:&N length:4 atIndex:6];
        [e setBytes:&K length:4 atIndex:7];
        [e setBytes:&nGridX length:4 atIndex:8];
        [e setBytes:&cv length:4 atIndex:9];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    };

    // Warmup
    run(midA, idmap0); run(midB, idmapR);

    double t0_all = 1e9, tR_all = 1e9;
    for (int r = 0; r < iters; r++) { double t0=now_ms(); run(midA, idmap0); t0_all = fmin(t0_all, now_ms()-t0); }
    for (int r = 0; r < iters; r++) { double t0=now_ms(); run(midB, idmapR); tR_all = fmin(tR_all, now_ms()-t0); }

    printf("id_map all-zeros (no per-tile indirection):  %.3f ms\n", t0_all);
    printf("id_map round-robin (per-tile expert select): %.3f ms\n", tR_all);
    printf("id-map overhead:                             %.3f x  (>1 means indirection adds cost)\n", tR_all / t0_all);
    printf("(if ~1.0, id-mapped weight selection is essentially free → counted-indirect production wiring is safe)\n");

    // Correctness: id_map all-zeros must produce the same output as the original
    // fused kernel (we don't have the original kernel here, so this is just a
    // sanity check that the id-mapped output isn't NaN/inf).
    float *out = (float*)midA.contents;
    int nan_count = 0; float max_abs = 0.f;
    for (size_t i = 0; i < (size_t)M*N; i++) {
        if (!isfinite(out[i])) nan_count++;
        if (fabsf(out[i]) > max_abs) max_abs = fabsf(out[i]);
    }
    printf("sanity:  finite_outputs=%s  max_abs=%.3g  nan_count=%d\n",
           nan_count==0 ? "yes" : "NO", max_abs, nan_count);
}
return 0;}
