// int8xint8 dense Q8_0 probe: fold Q8_0 per-32-block fp16 scale into int8 weight via a global
// w_qscale (dequant Q8_0 -> real -> requant int8), quantize activations with global a_qscale, do
// int8xint8->int32 matmul2d (single global rescale C_i32/(a_qscale*w_qscale)). Validates vs the
// exact float dequant dot + measures GF/s vs the float(act)xhalf(weight) dense kernel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef _Float16 fp16_t;
typedef struct { fp16_t d; int8_t qs[32]; } block_q8_0;
typedef struct { int32_t ne00,ne02; uint64_t nb01,nb02,nb03; int32_t ne12;
    uint64_t nb10,nb11,nb12,nb13; int32_t ne0,ne1; int16_t r2,r3; } mm_args;
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

// int8 dense: weight Q8_0 staged to threadgroup as int8 (real*w_qscale), activation int8 read direct,
// int8xint8->int32 accumulate over K, store int32 to dst (host rescales by 1/(a_qscale*w_qscale)).
static const char *KSRC_I8 =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct block_q8_0 { half d; char qs[32]; };\n"
"struct A { int ne00,ne02; ulong nb01,nb02,nb03; int ne12; ulong nb10,nb11,nb12,nb13; int ne0,ne1; short r2,r3; };\n"
"kernel void k(constant A &args [[buffer(0)]], device const char *srcA [[buffer(1)]],\n"  // srcA = Q8_0 weight [out x in]
"   device const char *srcB [[buffer(2)]], device char *dst [[buffer(3)]],\n"             // srcB = int8 act [tok x in], dst = int32 [tok x out]
"   constant float &wq [[buffer(4)]],\n"
"   threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],\n"
"   ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"  constexpr int NR1=I8_NR1, NR0=I8_NR0, NK=I8_NK, NL=NK/16, NUM_THREADS=128;\n"
"  const int K=args.ne00, M=args.ne0, N=args.ne1;\n"
"  const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"  threadgroup int8_t *sa=(threadgroup int8_t*)shmem;\n"
"  auto tA=tensor(sa,dextents<int32_t,2>(NK,NR0));\n"
"  device int8_t *ptrB=(device int8_t*)(srcB);\n"
"  const int strideB=args.nb11;\n"  // bytes == elements for int8
"  auto tB=tensor(ptrB,dextents<int32_t,2>(K,N),array<int,2>({1,strideB}));\n"
"  matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  auto cT=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0;\n"
"  for(int lk=0;lk<K;lk+=NK){\n"
"    for(int work=tiitg;work<NR0*NL;work+=NUM_THREADS){ const int row=work/NL, kc=work%NL, kbase=kc*16, kpos=lk+kbase;\n"  // 16 values/work-item
"      if(r0+row<M){ const int bidx=kpos/32; const int j0=kpos%32; device const block_q8_0*rp=(device const block_q8_0*)(srcA+args.nb01*(r0+row)); const float d=(float)rp[bidx].d*wq;\n"
"        for(int i=0;i<16;i++){ float real=(float)rp[bidx].qs[j0+i]*d; sa[row*NK+kbase+i]=(kpos+i<K)?(int8_t)clamp(rint(real),-127.0f,127.0f):(int8_t)0; } }\n"
"      else { for(int i=0;i<16;i++) sa[row*NK+kbase+i]=0; } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tA.slice(0,0); auto mB=tB.slice(lk,r1); mm.run(mB,mA,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  device int32_t *db=(device int32_t*)dst;\n"
"  auto tD=tensor(db,dextents<int32_t,2>(M,N),array<int,2>({1,M}));\n"
"  auto mD=tD.slice(r0,r1); cT.store(mD);\n"
"}\n";

// int8 dense with PRE-quantized weight (simulates offline Q8_0->int8+rowscale repack: no per-dispatch
// dequant, kernel just loads int8 weight directly into the threadgroup tile -- the true int8 ceiling).
static const char *KSRC_I8_PRE =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct A { int ne00,ne02; ulong nb01,nb02,nb03; int ne12; ulong nb10,nb11,nb12,nb13; int ne0,ne1; short r2,r3; };\n"
"kernel void kp(constant A &args [[buffer(0)]], device const char *srcA [[buffer(1)]],\n"  // srcA = int8 weight [out x in], row-major
"   device const char *srcB [[buffer(2)]], device char *dst [[buffer(3)]], constant float &wq [[buffer(4)]],\n"
"   threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],\n"
"   ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"  constexpr int NR1=I8_NR1, NR0=I8_NR0, NK=I8_NK, NUM_THREADS=128;\n"
"  const int K=args.ne00, M=args.ne0, N=args.ne1;\n"
"  const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"  threadgroup int8_t *sa=(threadgroup int8_t*)shmem; auto tA=tensor(sa,dextents<int32_t,2>(NK,NR0));\n"
"  device int8_t *ptrB=(device int8_t*)(srcB); const int strideB=args.nb11;\n"
"  auto tB=tensor(ptrB,dextents<int32_t,2>(K,N),array<int,2>({1,strideB}));\n"
"  matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  auto cT=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0;\n"
"  device const int8_t *wA=(device const int8_t*)srcA; constexpr int NL=NK/16;\n"
"  for(int lk=0;lk<K;lk+=NK){\n"
"    for(int work=tiitg;work<NR0*NL;work+=NUM_THREADS){ const int row=work/NL,kb=(work%NL)*16; if(r0+row<M){ device const int8_t*wr=wA+(size_t)(r0+row)*K+lk+kb; for(int i=0;i<16;i++) sa[row*NK+kb+i]=(lk+kb+i<K)?wr[i]:(int8_t)0; } else { for(int i=0;i<16;i++) sa[row*NK+kb+i]=0; } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tA.slice(0,0); auto mB=tB.slice(lk,r1); mm.run(mB,mA,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  device int32_t *db=(device int32_t*)dst; auto tD=tensor(db,dextents<int32_t,2>(M,N),array<int,2>({1,M}));\n"
"  auto mD=tD.slice(r0,r1); cT.store(mD);\n"
"}\n";

// int8 dense FUSED rescale: pre-int8 weight + per-token/per-row scales, rescale in the store (no int32
// round-trip). NR1=64 so the 64x64 int32 result tile (16KB) + weight tile (8KB) fit threadgroup. Outputs f32.
static const char *KSRC_I8_FUSED =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct A { int ne00,ne02; ulong nb01,nb02,nb03; int ne12; ulong nb10,nb11,nb12,nb13; int ne0,ne1; short r2,r3; };\n"
"kernel void kfu(constant A &args [[buffer(0)]], device const char *srcA [[buffer(1)]],\n"   // srcA=int8 weight[out x in]
"   device const char *srcB [[buffer(2)]], device char *dst [[buffer(3)]],\n"                // srcB=int8 act[tok x in], dst=f32[tok x out]
"   device const float *wscale [[buffer(4)]], device const float *ascale [[buffer(5)]],\n"
"   threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],\n"
"   ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"  constexpr int NR1=128, NR0=32, NK=128, NL=NK/16, NUM_THREADS=128;\n"
"  const int K=args.ne00, M=args.ne0, N=args.ne1;\n"
"  const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"  threadgroup int8_t *sa=(threadgroup int8_t*)shmem;\n"
"  threadgroup int32_t *sc=(threadgroup int32_t*)(shmem + NR0*NK);\n"  // NR0xNR1 int32 result tile
"  auto tA=tensor(sa,dextents<int32_t,2>(NK,NR0));\n"
"  device int8_t *ptrB=(device int8_t*)(srcB); const int strideB=args.nb11;\n"
"  auto tB=tensor(ptrB,dextents<int32_t,2>(K,N),array<int,2>({1,strideB}));\n"
"  matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  auto cT=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0;\n"
"  device const int8_t *wA=(device const int8_t*)srcA;\n"
"  for(int lk=0;lk<K;lk+=NK){\n"
"    for(int work=tiitg;work<NR0*NL;work+=NUM_THREADS){ const int row=work/NL,kb=(work%NL)*16; if(r0+row<M){ device const int8_t*wr=wA+(size_t)(r0+row)*K+lk+kb; for(int i=0;i<16;i++) sa[row*NK+kb+i]=(lk+kb+i<K)?wr[i]:(int8_t)0; } else { for(int i=0;i<16;i++) sa[row*NK+kb+i]=0; } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tA.slice(0,0); auto mB=tB.slice(lk,r1); mm.run(mB,mA,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  auto tC=tensor(sc,dextents<int32_t,2>(NR0,NR1),array<int,2>({1,NR0})); cT.store(tC);\n"   // [out_tile x tok_tile], col-major like dst
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  device float *db=(device float*)dst;\n"
"  for(int w=tiitg;w<NR0*NR1;w+=NUM_THREADS){ int m=w%NR0,n=w/NR0,o=r0+m,t=r1+n; if(o<M&&t<N){ db[(size_t)t*M+o]=(float)sc[m+n*NR0]*ascale[t]*wscale[o]; } }\n"
"}\n";

// float(act)xhalf(weight) reference dense kernel (the shipped one) for the GF/s comparison.
static const char *KSRC_F =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct block_q8_0 { half d; char qs[32]; };\n"
"struct A { int ne00,ne02; ulong nb01,nb02,nb03; int ne12; ulong nb10,nb11,nb12,nb13; int ne0,ne1; short r2,r3; };\n"
"inline void deq(device const block_q8_0 *xb, short il, thread half4x4 &reg){ device const char*qs=(device const char*)xb->qs; const float d=(float)xb->d; for(int i=0;i<16;i++) reg[i/4][i%4]=(half)((float)qs[i+16*il]*d);}\n"
"kernel void kf(constant A &args [[buffer(0)]], device const char *srcA [[buffer(1)]],\n"
"   device const char *srcB [[buffer(2)]], device char *dst [[buffer(3)]],\n"
"   threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],\n"
"   ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"  constexpr int NR1=128, NR0=64, NK=32, NL=NK/16, NUM_THREADS=128;\n"
"  const int K=args.ne00, M=args.ne0, N=args.ne1;\n"
"  const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n"
"  threadgroup half *sa=(threadgroup half*)shmem; auto tA=tensor(sa,dextents<int32_t,2>(NK,NR0));\n"
"  device float *ptrB=(device float*)(srcB); const int strideB=args.nb11/sizeof(float);\n"
"  auto tB=tensor(ptrB,dextents<int32_t,2>(K,N),array<int,2>({1,strideB}));\n"
"  matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,true,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  auto cT=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),float>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0.0f;\n"
"  for(int lk=0;lk<K;lk+=NK){\n"
"    for(int work=tiitg;work<NR0*NL;work+=NUM_THREADS){ const int row=work/NL,kc=work%NL,kpos=lk+kc*16; const short kb=kc*16;\n"
"      if(r0+row<M){ const int bidx=kpos/32; const short il=(kpos/16)%2; device const block_q8_0*rp=(device const block_q8_0*)(srcA+args.nb01*(r0+row)); half4x4 t; deq(rp+bidx,il,t); for(short i=0;i<16;i++) sa[row*NK+kb+i]=(kpos+i<K)?t[i/4][i%4]:(half)0; }\n"
"      else { for(short i=0;i<16;i++) sa[row*NK+kb+i]=(half)0; } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tA.slice(0,0); auto mB=tB.slice(lk,r1); mm.run(mB,mA,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  device float *db=(device float*)dst; auto tD=tensor(db,dextents<int32_t,2>(M,N),array<int,2>({1,M}));\n"
"  auto mD=tD.slice(r0,r1); cT.store(mD);\n"
"}\n";

static id<MTLComputePipelineState> mkpipe(id<MTLDevice>d,const char*src,NSString*fn){
    NSError*e=0; MTLCompileOptions*o=[MTLCompileOptions new]; o.languageVersion=MTLLanguageVersion4_0;
    id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if(!lib){fprintf(stderr,"compile %s:\n%s\n",fn.UTF8String,e.localizedDescription.UTF8String);exit(2);}
    id<MTLFunction>f=[lib newFunctionWithName:fn]; id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];
    if(!p){fprintf(stderr,"pipe %s: %s\n",fn.UTF8String,e.localizedDescription.UTF8String);exit(2);} return p;
}

int main(int argc,char**argv){
    int T=4096, M=2048, K=2048, iters=8;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--T")&&i+1<argc)T=atoi(argv[++i]); else if(!strcmp(argv[i],"--M")&&i+1<argc)M=atoi(argv[++i]); else if(!strcmp(argv[i],"--K")&&i+1<argc)K=atoi(argv[++i]); else if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atoi(argv[++i]); }
    int bpr=K/32;
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue>q=[dev newCommandQueue];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)M*bpr*sizeof(block_q8_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer>Bf=[dev newBufferWithLength:(NSUInteger)T*K*sizeof(float) options:MTLResourceStorageModeShared];   // f32 acts
        id<MTLBuffer>Bi=[dev newBufferWithLength:(NSUInteger)T*K options:MTLResourceStorageModeShared];                 // int8 acts
        id<MTLBuffer>Df=[dev newBufferWithLength:(NSUInteger)T*M*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>Di=[dev newBufferWithLength:(NSUInteger)T*M*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer>Wi=[dev newBufferWithLength:(NSUInteger)M*K options:MTLResourceStorageModeShared]; // pre-int8 weight
        id<MTLBuffer>Ws=[dev newBufferWithLength:(NSUInteger)M*sizeof(float) options:MTLResourceStorageModeShared];   // per-row w scale
        id<MTLBuffer>As=[dev newBufferWithLength:(NSUInteger)T*sizeof(float) options:MTLResourceStorageModeShared];   // per-token a scale
        id<MTLBuffer>Dfu=[dev newBufferWithLength:(NSUInteger)T*M*sizeof(float) options:MTLResourceStorageModeShared];// fused f32 out
        block_q8_0*Wp=W.contents; uint32_t rng=7u; float wmax=0;
        for(NSUInteger b=0;b<(NSUInteger)M*bpr;b++){rng=rng*1664525u+1013904223u;Wp[b].d=(fp16_t)(0.005f+0.0002f*(rng%37));for(int j=0;j<32;j++){rng=rng*1664525u+1013904223u;Wp[b].qs[j]=(int8_t)((int)(rng%127)-63); float r=fabsf((float)Wp[b].qs[j]*(float)Wp[b].d); if(r>wmax)wmax=r;}}
        float*Bp=Bf.contents; float amax=0; for(NSUInteger i=0;i<(NSUInteger)T*K;i++){Bp[i]=(((int)((i*17u+3u)%97)-48))*0.02f; if(fabsf(Bp[i])>amax)amax=fabsf(Bp[i]);}
        float a_qscale=127.0f/amax, w_qscale=127.0f/wmax;   // global (legacy online path)
        int8_t*Bip=Bi.contents; for(NSUInteger i=0;i<(NSUInteger)T*K;i++){ float v=rintf(Bp[i]*a_qscale); Bip[i]=(int8_t)(v>127?127:(v<-127?-127:v)); }
        // PRODUCTION W8A8: per-row weight scale, per-token activation scale.
        float *w_rowscale=malloc((size_t)M*sizeof(float)), *a_tokscale=malloc((size_t)T*sizeof(float));
        int8_t*Wip=Wi.contents;
        for(int m=0;m<M;m++){ float rmax=0; for(int kb=0;kb<bpr;kb++){block_q8_0*blk=&Wp[(size_t)m*bpr+kb];float d=(float)blk->d;for(int j=0;j<32;j++){float r=fabsf((float)blk->qs[j]*d);if(r>rmax)rmax=r;}}
            float sc=rmax>0?rmax/127.0f:1e-9f; w_rowscale[m]=sc; float inv=1.0f/sc;
            for(int kb=0;kb<bpr;kb++){block_q8_0*blk=&Wp[(size_t)m*bpr+kb];float d=(float)blk->d;for(int j=0;j<32;j++){float v=rintf((float)blk->qs[j]*d*inv);Wip[(size_t)m*K+kb*32+j]=(int8_t)(v>127?127:(v<-127?-127:v));}} }
        int8_t*Bi2=Bi.contents;  // reuse Bi for per-token int8 acts (overwrite global ones)
        for(int t=0;t<T;t++){ float rmax=0; for(int k=0;k<K;k++){float r=fabsf(Bp[(size_t)t*K+k]);if(r>rmax)rmax=r;} float sc=rmax>0?rmax/127.0f:1e-9f; a_tokscale[t]=sc; float inv=1.0f/sc; for(int k=0;k<K;k++){float v=rintf(Bp[(size_t)t*K+k]*inv);Bi2[(size_t)t*K+k]=(int8_t)(v>127?127:(v<-127?-127:v));} }
        memcpy(Ws.contents,w_rowscale,(size_t)M*sizeof(float)); memcpy(As.contents,a_tokscale,(size_t)T*sizeof(float));
        // CPU exact float ref (first Tref tokens)
        int Tref=T<64?T:64; double *ref=calloc((size_t)Tref*M,sizeof(double));
        for(int t=0;t<Tref;t++)for(int m=0;m<M;m++){double s=0;for(int kb=0;kb<bpr;kb++){block_q8_0*blk=&Wp[(size_t)m*bpr+kb];double d=(double)(float)blk->d;for(int j=0;j<32;j++){int k=kb*32+j;s+=(double)Bp[(size_t)t*K+k]*((double)blk->qs[j]*d);}}ref[(size_t)t*M+m]=s;}
        mm_args a={.ne00=K,.ne02=1,.nb01=(uint64_t)bpr*sizeof(block_q8_0),.nb02=0,.nb03=0,.ne12=1,.nb10=1,.nb11=(uint64_t)K,.nb12=(uint64_t)T*K,.nb13=0,.ne0=M,.ne1=T,.r2=1,.r3=1};
        mm_args af=a; af.nb11=(uint64_t)K*4; af.nb12=(uint64_t)T*K*4;
        MTLSize grid=MTLSizeMake((T+127)/128,(M+63)/64,1), tg=MTLSizeMake(128,1,1);
        double inv=1.0/((double)a_qscale*(double)w_qscale);
        printf("dense T=%d M=%d K=%d  (NR1=128 NR0=64)\n",T,M,K);
        #define TIME(ps,abuf,argp,sz,tgm,dbuf,extra) ({ double t0=now_s(); id<MTLCommandBuffer>cb=[q commandBuffer]; for(int i=0;i<iters;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:argp length:sz atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:abuf offset:0 atIndex:2];[en setBuffer:dbuf offset:0 atIndex:3];extra [en setThreadgroupMemoryLength:tgm atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[cb commit];[cb waitUntilCompleted]; (now_s()-t0)*1000.0/iters; })
        // ---- float ref (NK=32, relaxed -- its best) ----
        id<MTLComputePipelineState>pf=mkpipe(dev,KSRC_F,@"kf");
        { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:pf];[en setBytes:&af length:sizeof(af) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:Bf offset:0 atIndex:2];[en setBuffer:Df offset:0 atIndex:3];[en setThreadgroupMemoryLength:64*32*2 atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted]; }
        float*Dfp=Df.contents; double fabsmax=0,fden=0; for(int t=0;t<Tref;t++)for(int m=0;m<M;m++){double ab=fabs((double)Dfp[(size_t)t*M+m]-ref[(size_t)t*M+m]); if(ab>fabsmax)fabsmax=ab; if(fabs(ref[(size_t)t*M+m])>fden)fden=fabs(ref[(size_t)t*M+m]);}
        double ms_f=TIME(pf,Bf,&af,sizeof(af),64*32*2,Df,); double gf_f=2.0*T*M*K/(ms_f*1e6);
        printf("  float relaxed NK=32 :  %.3f ms  %8.1f GF/s  rel=%.4f\n",ms_f,gf_f,fabsmax/fmax(fden,1e-6));
        // ---- int8 NK sweep ----
        int NKs[]={32,64,128,256}; double best_i8=0;
        for(int ni=0;ni<4;ni++){ int NK=NKs[ni];
            char def[128]; snprintf(def,sizeof(def),"#define I8_NR1 128\n#define I8_NR0 64\n#define I8_NK %d\n",NK);
            char*src=malloc(strlen(def)+strlen(KSRC_I8)+1); strcpy(src,def); strcat(src,KSRC_I8);
            id<MTLComputePipelineState>pi=mkpipe(dev,src,@"k"); free(src);
            NSUInteger tgm=(NSUInteger)64*NK; // int8 weight tile
            { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:pi];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:Bi offset:0 atIndex:2];[en setBuffer:Di offset:0 atIndex:3];[en setBytes:&w_qscale length:4 atIndex:4];[en setThreadgroupMemoryLength:tgm atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted]; if(cb.error){fprintf(stderr,"i8 NK=%d %s\n",NK,cb.error.localizedDescription.UTF8String);continue;} }
            int32_t*Dip=Di.contents; double maxabs=0,maxden=0; for(int t=0;t<Tref;t++)for(int m=0;m<M;m++){double gv=(double)Dip[(size_t)t*M+m]*inv; double rf=ref[(size_t)t*M+m]; double ab=fabs(gv-rf); if(ab>maxabs)maxabs=ab; if(fabs(rf)>maxden)maxden=fabs(rf);}
            double rel=maxabs/fmax(maxden,1e-6);
            double ms=TIME(pi,Bi,&a,sizeof(a),tgm,Di,[en setBytes:&w_qscale length:4 atIndex:4];); double gf=2.0*T*M*K/(ms*1e6);
            if(gf>best_i8)best_i8=gf;
            printf("  int8 NK=%-3d         :  %.3f ms  %8.1f GF/s  rel=%.4f %s\n",NK,ms,gf,rel, rel<0.05?"OK":"DRIFT");
        }
        printf("  >>> best int8 (online dequant) / float-relaxed = %.2fx\n", best_i8/gf_f);
        // ---- pre-int8 (offline-repacked weight, no per-dispatch dequant): the int8 ceiling ----
        double best_pre=0;
        for(int ni=0;ni<4;ni++){ int NK=NKs[ni];
            char def[128]; snprintf(def,sizeof(def),"#define I8_NR1 128\n#define I8_NR0 64\n#define I8_NK %d\n",NK);
            char*src=malloc(strlen(def)+strlen(KSRC_I8_PRE)+1); strcpy(src,def); strcat(src,KSRC_I8_PRE);
            id<MTLComputePipelineState>pp=mkpipe(dev,src,@"kp"); free(src);
            NSUInteger tgm=(NSUInteger)64*NK;
            double ms=TIME(pp,Bi,&a,sizeof(a),tgm,Di,[en setBytes:&w_qscale length:4 atIndex:4];[en setBuffer:Wi offset:0 atIndex:1];); double gf=2.0*T*M*K/(ms*1e6);
            if(gf>best_pre)best_pre=gf;
            // PRODUCTION accuracy: rescale int32 by per-token a_tokscale[t] * per-row w_rowscale[o]
            int32_t*Dip=Di.contents; double maxabs=0,maxden=0;
            for(int t=0;t<Tref;t++)for(int m=0;m<M;m++){double gv=(double)Dip[(size_t)t*M+m]*(double)a_tokscale[t]*(double)w_rowscale[m]; double rf=ref[(size_t)t*M+m]; double ab=fabs(gv-rf); if(ab>maxabs)maxabs=ab; if(fabs(rf)>maxden)maxden=fabs(rf);}
            double relp=maxabs/fmax(maxden,1e-6);
            printf("  int8-PRE NK=%-3d     :  %.3f ms  %8.1f GF/s  rel(perrow/pertok)=%.4f %s\n",NK,ms,gf,relp, relp<0.02?"OK":"DRIFT");
        }
        printf("  >>> best int8-PRE (offline-repack, no rescale) / float-relaxed = %.2fx\n", best_pre/gf_f);
        // ---- FUSED rescale (NR1=64, outputs f32 directly): the real end-to-end-relevant int8 speed ----
        { id<MTLComputePipelineState>pu=mkpipe(dev,KSRC_I8_FUSED,@"kfu");
          MTLSize fgrid=MTLSizeMake((T+127)/128,(M+31)/32,1); NSUInteger ftgm=(NSUInteger)32*128+(NSUInteger)32*128*4;
          // correctness
          { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:pu];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Wi offset:0 atIndex:1];[en setBuffer:Bi offset:0 atIndex:2];[en setBuffer:Dfu offset:0 atIndex:3];[en setBuffer:Ws offset:0 atIndex:4];[en setBuffer:As offset:0 atIndex:5];[en setThreadgroupMemoryLength:ftgm atIndex:0];[en dispatchThreadgroups:fgrid threadsPerThreadgroup:MTLSizeMake(128,1,1)];[en endEncoding];[cb commit];[cb waitUntilCompleted]; if(cb.error){fprintf(stderr,"fused %s\n",cb.error.localizedDescription.UTF8String);return 2;} }
          float*Dup=Dfu.contents; double maxabs=0,maxden=0; for(int t=0;t<Tref;t++)for(int m=0;m<M;m++){double ab=fabs((double)Dup[(size_t)t*M+m]-ref[(size_t)t*M+m]); if(ab>maxabs)maxabs=ab; if(fabs(ref[(size_t)t*M+m])>maxden)maxden=fabs(ref[(size_t)t*M+m]);}
          double t0=now_s(); id<MTLCommandBuffer>cb=[q commandBuffer]; for(int i=0;i<iters;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:pu];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Wi offset:0 atIndex:1];[en setBuffer:Bi offset:0 atIndex:2];[en setBuffer:Dfu offset:0 atIndex:3];[en setBuffer:Ws offset:0 atIndex:4];[en setBuffer:As offset:0 atIndex:5];[en setThreadgroupMemoryLength:ftgm atIndex:0];[en dispatchThreadgroups:fgrid threadsPerThreadgroup:MTLSizeMake(128,1,1)];[en endEncoding];}[cb commit];[cb waitUntilCompleted];
          double ms=(now_s()-t0)*1000.0/iters; double gf=2.0*T*M*K/(ms*1e6);
          printf("  int8 FUSED NR1=64   :  %.3f ms  %8.1f GF/s  rel=%.4f %s  (%.2fx vs float)\n",ms,gf,maxabs/fmax(maxden,1e-6),maxabs/fmax(maxden,1e-6)<0.02?"OK":"DRIFT",gf/gf_f); }
        free(ref); free(w_rowscale); free(a_tokscale);
    }
    return 0;
}
