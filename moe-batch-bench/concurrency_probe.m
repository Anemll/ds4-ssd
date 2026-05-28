// Metal-MM (simdgroup/ALU) vs NAX (matmul2d/tensor units) concurrency probe.
// Question: on M5, do matmul2d and simdgroup_multiply_accumulate run on separate
// issue ports that OVERLAP, or contend for the same matrix hardware?
// Method: two long busy kernels; compare wall time when submitted concurrently
// (two command buffers, no wait between) vs the sum of their solo times.
//   speedup = (t_nax + t_alu) / t_concurrent ;  ~1.0 = serialize, ~2.0 = full overlap.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"// NAX: repeated 64x32 matmul2d tile (tensor units)\n"
"kernel void k_nax(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
"                  constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
"                  constant uint &iters [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]]) {\n"
"    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));\n"
"    matmul2d<desc, execution_simdgroups<4>> op;\n"
"    auto tA = tensor(A, dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tB = tensor(B, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tC = tensor(C, dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mA = tA.slice(0, 0);\n"
"    auto mB = tB.slice(0, 0);\n"
"    auto mC = tC.slice(0, 0);\n"
"    for (uint i = 0; i < iters; i++) { op.run(mA, mB, mC); }\n"
"}\n"
"// ALU: repeated simdgroup 8x8 MMA (shader cores)\n"
"kernel void k_alu(device half *A [[buffer(0)]], device half *B [[buffer(1)]], device float *C [[buffer(2)]],\n"
"                  constant uint &iters [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]],\n"
"                  ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    simdgroup_half8x8 a, b;\n"
"    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float,8>(0.0f);\n"
"    simdgroup_load(a, A, 8);\n"
"    simdgroup_load(b, B, 8);\n"
"    for (uint i = 0; i < iters; i++) { simdgroup_multiply_accumulate(c, a, b, c); }\n"
"    simdgroup_store(c, C + (tgid.x + tgid.y*1024)*64, 8);\n"
"}\n";

static double now_ms(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }

int main(int argc, char **argv){
@autoreleasepool{
    uint32_t iters_nax = argc>1?atoi(argv[1]):2000;
    uint32_t iters_alu = argc>2?atoi(argv[2]):200000;
    uint32_t gx = argc>3?atoi(argv[3]):64, gy = argc>4?atoi(argv[4]):64;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandQueue> q2 = [dev newCommandQueue];   // separate queue => true concurrency
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    NSError *err=nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if(!lib){ NSLog(@"compile failed: %@", err); return 1; }
    id<MTLComputePipelineState> pNax = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_nax"] error:&err];
    id<MTLComputePipelineState> pAlu = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_alu"] error:&err];
    if(!pNax||!pAlu){ NSLog(@"pipeline failed: %@", err); return 1; }
    uint32_t M=64,N=32,K=2048;
    id<MTLBuffer> A=[dev newBufferWithLength:(NSUInteger)M*K*2 options:0];
    id<MTLBuffer> B=[dev newBufferWithLength:(NSUInteger)N*K*2 options:0];
    id<MTLBuffer> C=[dev newBufferWithLength:(NSUInteger)gx*gy*64*4 options:0];
    MTLSize tg=MTLSizeMake(128,1,1), grid=MTLSizeMake(gx,gy,1);
    void(^encNax)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb){
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pNax];
        [e setBuffer:A offset:0 atIndex:0];[e setBuffer:B offset:0 atIndex:1];[e setBuffer:C offset:0 atIndex:2];
        [e setBytes:&M length:4 atIndex:3];[e setBytes:&N length:4 atIndex:4];[e setBytes:&K length:4 atIndex:5];
        [e setBytes:&iters_nax length:4 atIndex:6];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];[e endEncoding]; };
    void(^encAlu)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb){
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pAlu];
        [e setBuffer:A offset:0 atIndex:0];[e setBuffer:B offset:0 atIndex:1];[e setBuffer:C offset:0 atIndex:2];
        [e setBytes:&iters_alu length:4 atIndex:6];
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];[e endEncoding]; };
    double tn=1e9,ta=1e9,tc=1e9;
    for(int r=0;r<5;r++){
        double t0=now_ms(); id<MTLCommandBuffer> cb=[q commandBuffer]; encNax(cb); [cb commit]; [cb waitUntilCompleted];
        tn=fmin(tn, now_ms()-t0);
        t0=now_ms(); cb=[q commandBuffer]; encAlu(cb); [cb commit]; [cb waitUntilCompleted];
        ta=fmin(ta, now_ms()-t0);
        t0=now_ms(); id<MTLCommandBuffer> c1=[q commandBuffer]; encNax(c1); id<MTLCommandBuffer> c2=[q2 commandBuffer]; encAlu(c2);
        [c1 commit]; [c2 commit]; [c1 waitUntilCompleted]; [c2 waitUntilCompleted];
        tc=fmin(tc, now_ms()-t0);
    }
    printf("iters_nax=%u iters_alu=%u  t_nax=%.2fms  t_alu=%.2fms  t_concurrent=%.2fms\n", iters_nax,iters_alu,tn,ta,tc);
    printf("sum=%.2fms  speedup=(sum/concurrent)=%.3f   [1.0=serialize, ~2.0=full overlap]\n", tn+ta, (tn+ta)/tc);
    printf("verdict: %s\n", (tn+ta)/tc > 1.25 ? "OVERLAP (separate units) -> 3-engine split viable"
                                               : "SERIALIZE (same units) -> no Metal||NAX win");
}
return 0;}
