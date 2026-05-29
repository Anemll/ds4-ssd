// tg_sweep_probe.m
// Threadgroup-size (simdgroup-count) sweep for the two routed-MoE matmul cores,
// at 16K-chunk-representative per-expert dims. No model load.
//
//   NAX-half : half x half -> f32 matmul2d (the h_h_f gate/up core).
//              tile NR0(M)=64, NR1(N) in {32,64,128}, NK=32.
//   NAX-int8 : i8 x i8 -> i32 matmul2d (the iq2_i8_i32 core, dequant excluded —
//              dequant is a fixed SG-independent cost).
//              tile NR0(M)=64, NR1(N)=32, NK=256.
//
// For each kernel we instantiate the SAME tile with SG in {1,2,4,8} simdgroups
// and DISPATCH tg = SG*32 threads (the actual threadgroup size). Production ships
// SG=4 (tg=128). Grid = ceil(N/NR1) x ceil(M/NR0), matching the per-expert dispatch
// (M = refs). Correctness: each variant vs the kernel's SG=4 baseline (max_abs).
//
// Build:  cc -O2 -fobjc-arc tg_sweep_probe.m -o /tmp/tg_sweep \
//             -framework Foundation -framework Metal
// Run:    /tmp/tg_sweep <M=refs> <N=2048> <K_half=4096> <iters=30>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"\n"
"// ---- half x half -> f32 ----\n"
"template <int NR0,int NR1,int NK,int SG>\n"
"void mm_h_body(device half *X, device half *W, device float *O,\n"
"               uint M, uint N, uint K, uint2 tgid) {\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<SG>> mm;\n"
"    auto tX = tensor(X, dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tW = tensor(W, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tO = tensor(O, dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mX = tX.slice(0, tgid.y*NR0);\n"
"    auto mW = tW.slice(tgid.x*NR1, 0);\n"
"    auto mO = tO.slice(tgid.x*NR1, tgid.y*NR0);\n"
"    auto cO = mm.get_destination_cooperative_tensor<decltype(mW), decltype(mX), float>();\n"
"    for (uint16_t i=0;i<cO.get_capacity();++i) if (cO.is_valid_element(i)) cO[i]=0.f;\n"
"    mm.run(mW, mX, cO);\n"
"    cO.store(mO);\n"
"}\n"
"#define HV(nr1,sg) kernel void h_64_##nr1##_##sg(device half *X [[buffer(0)]], device half *W [[buffer(1)]], device float *O [[buffer(2)]], constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]], uint2 t [[threadgroup_position_in_grid]]) { mm_h_body<64,nr1,32,sg>(X,W,O,M,N,K,t); }\n"
"HV(32,1) HV(32,2) HV(32,4) HV(32,8)\n"
"HV(64,1) HV(64,2) HV(64,4) HV(64,8)\n"
"HV(128,1) HV(128,2) HV(128,4) HV(128,8)\n"
"\n"
"// ---- i8 x i8 -> i32 ----\n"
"template <int NR0,int NR1,int NK,int SG>\n"
"void mm_i8_body(device char *X, device char *W, device int *O,\n"
"                uint M, uint N, uint K, uint2 tgid) {\n"
"    constexpr auto desc = matmul2d_descriptor(NR1, NR0, NK, false, true, true,\n"
"                                              matmul2d_descriptor::mode::multiply_accumulate);\n"
"    matmul2d<desc, execution_simdgroups<SG>> mm;\n"
"    auto tX = tensor(X, dextents<int32_t,2>{(int32_t)K,(int32_t)M}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto tW = tensor(W, dextents<int32_t,2>{(int32_t)N,(int32_t)K}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto tO = tensor(O, dextents<int32_t,2>{(int32_t)N,(int32_t)M}, array<int32_t,2>{1,(int32_t)N});\n"
"    auto mX = tX.slice(0, tgid.y*NR0);\n"
"    auto mW = tW.slice(tgid.x*NR1, 0);\n"
"    auto mO = tO.slice(tgid.x*NR1, tgid.y*NR0);\n"
"    auto cO = mm.get_destination_cooperative_tensor<decltype(mW), decltype(mX), int>();\n"
"    for (uint16_t i=0;i<cO.get_capacity();++i) if (cO.is_valid_element(i)) cO[i]=0;\n"
"    mm.run(mW, mX, cO);\n"
"    cO.store(mO);\n"
"}\n"
"#define IV(sg) kernel void i8_64_32_##sg(device char *X [[buffer(0)]], device char *W [[buffer(1)]], device int *O [[buffer(2)]], constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]], uint2 t [[threadgroup_position_in_grid]]) { mm_i8_body<64,32,256,sg>(X,W,O,M,N,K,t); }\n"
"IV(1) IV(2) IV(4) IV(8)\n";

static double now_ms(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
    return (double)mach_absolute_time()*tb.numer/tb.denom/1e6; }

typedef struct { const char *name; int nr1; int sg; bool i8; } variant;

int main(int argc, char **argv){
@autoreleasepool{
    uint32_t M = argc>1?atoi(argv[1]):512;     // refs (per-expert tokens in a 16K chunk)
    uint32_t N = argc>2?atoi(argv[2]):2048;    // mid_dim (gate/up out)
    uint32_t K = argc>3?atoi(argv[3]):4096;    // in_dim (mult of 256 for i8)
    int iters = argc>4?atoi(argv[4]):30;
    printf("dims: M(refs)=%u N=%u K=%u iters=%d   (tg = SG*32; production SG=4)\n", M, N, K, iters);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opt.languageVersion = MTLLanguageVersion4_0;
    NSError *err=nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if(!lib){ NSLog(@"compile failed: %@", err); return 1; }

    id<MTLBuffer> X  = [dev newBufferWithLength:(NSUInteger)M*K*2 options:0];
    id<MTLBuffer> W  = [dev newBufferWithLength:(NSUInteger)N*K*2 options:0];
    id<MTLBuffer> O  = [dev newBufferWithLength:(NSUInteger)M*N*4 options:0];
    id<MTLBuffer> Xi = [dev newBufferWithLength:(NSUInteger)M*K options:0];
    id<MTLBuffer> Wi = [dev newBufferWithLength:(NSUInteger)N*K options:0];
    // deterministic small fills
    { uint32_t s=1; __fp16 *p=(__fp16*)X.contents; for(size_t i=0;i<(size_t)M*K;i++){s=s*1664525u+1013904223u; p[i]=(__fp16)((((s>>8)&0xffff)/65536.0f-0.5f)*0.1f);} }
    { uint32_t s=2; __fp16 *p=(__fp16*)W.contents; for(size_t i=0;i<(size_t)N*K;i++){s=s*1664525u+1013904223u; p[i]=(__fp16)((((s>>8)&0xffff)/65536.0f-0.5f)*0.1f);} }
    { uint32_t s=3; char *p=(char*)Xi.contents; for(size_t i=0;i<(size_t)M*K;i++){s=s*1664525u+1013904223u; p[i]=(char)((int)((s>>8)&0x1f)-16);} }
    { uint32_t s=4; char *p=(char*)Wi.contents; for(size_t i=0;i<(size_t)N*K;i++){s=s*1664525u+1013904223u; p[i]=(char)((int)((s>>8)&0x1f)-16);} }

    variant vs[] = {
        {"h_64_32_1",32,1,false},{"h_64_32_2",32,2,false},{"h_64_32_4",32,4,false},{"h_64_32_8",32,8,false},
        {"h_64_64_1",64,1,false},{"h_64_64_2",64,2,false},{"h_64_64_4",64,4,false},{"h_64_64_8",64,8,false},
        {"h_64_128_1",128,1,false},{"h_64_128_2",128,2,false},{"h_64_128_4",128,4,false},{"h_64_128_8",128,8,false},
        {"i8_64_32_1",32,1,true},{"i8_64_32_2",32,2,true},{"i8_64_32_4",32,4,true},{"i8_64_32_8",32,8,true},
    };
    int nv = sizeof(vs)/sizeof(vs[0]);

    // run a variant -> fill O, return ms (best of iters); writes nothing if pipeline invalid
    double base_h_ms=0, base_i8_ms=0; double *base_h=NULL,*base_i8=NULL;
    float  *refH=NULL; int *refI=NULL;

    const char *fastest_h=NULL, *fastest_i8=NULL; double best_h=1e9, best_i8=1e9;

    printf("\nvariant       tg   ms       GF/s      max_abs_vs_SG4\n");
    for (int vi=0; vi<nv; vi++){
        variant v = vs[vi];
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:v.name]];
        if(!fn){ printf("%-12s  --   (no function)\n", v.name); continue; }
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
        if(!p){ printf("%-12s  --   (invalid SG/tile: %s)\n", v.name, err?[[err localizedDescription] UTF8String]:"?"); err=nil; continue; }
        NSUInteger tg = (NSUInteger)v.sg*32u;
        MTLSize tgs = MTLSizeMake(tg,1,1);
        MTLSize grid = MTLSizeMake((N+v.nr1-1)/v.nr1, (M+63)/64, 1);
        memset(O.contents, 0, (size_t)M*N*4);
        double best=1e9;
        for (int it=0; it<iters+2; it++){
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:(v.i8?Xi:X) offset:0 atIndex:0];
            [e setBuffer:(v.i8?Wi:W) offset:0 atIndex:1];
            [e setBuffer:O offset:0 atIndex:2];
            [e setBytes:&M length:4 atIndex:3];
            [e setBytes:&N length:4 atIndex:4];
            [e setBytes:&K length:4 atIndex:5];
            [e dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
            [e endEncoding];
            double t0=now_ms(); [cb commit]; [cb waitUntilCompleted]; double dt=now_ms()-t0;
            if(it>=2 && dt<best) best=dt;   // drop 2 warmups
        }
        // GFLOPs: 2*M*N*K
        double gfs = (2.0*M*N*K)/(best/1e3)/1e9;
        // correctness vs SG4 baseline of same type
        double maxabs=-1;
        if (v.sg==4){
            if(v.i8){ base_i8=NULL; refI=malloc((size_t)M*N*4); memcpy(refI,O.contents,(size_t)M*N*4); }
            else if(v.nr1==32){ refH=malloc((size_t)M*N*4); memcpy(refH,O.contents,(size_t)M*N*4); }
            maxabs=0;
        } else {
            if(v.i8 && refI){ int *o=(int*)O.contents; long md=0; for(size_t i=0;i<(size_t)M*N;i++){long d=labs((long)o[i]-(long)refI[i]); if(d>md)md=d;} maxabs=(double)md; }
            else if(!v.i8 && refH){ float *o=(float*)O.contents; float md=0; for(size_t i=0;i<(size_t)M*N;i++){float d=fabsf(o[i]-refH[i]); if(d>md)md=d;} maxabs=md; }
        }
        printf("%-12s  %-3lu  %-8.3f %-9.1f %s%.4g\n", v.name, (unsigned long)tg, best, gfs,
               (v.sg==4?"(baseline) ":""), maxabs<0?0:maxabs);
        if(!v.i8 && best<best_h){ best_h=best; fastest_h=v.name; }
        if(v.i8 && best<best_i8){ best_i8=best; fastest_i8=v.name; }
    }
    printf("\n--- fastest NAX-half: %s (%.3f ms) ---\n", fastest_h?fastest_h:"?", best_h);
    printf("--- fastest NAX-int8: %s (%.3f ms) ---\n", fastest_i8?fastest_i8:"?", best_i8);
    printf("(name = kernel_NR0_NR1_SG; tg=SG*32; compare vs the *_*_4 = production tg=128)\n");
    free(refH); free(refI);
}
return 0;
}
