// NAX matmul2d autotuner for the ds4 indexer score kernel.
// Templates the kernel over knobs via injected #defines, compiles + validates vs CPU +
// times each config, prints a ranked table and the winning config.
// Knobs: dtype {half,int8}, TM {16,32,64}, NK {32,64,128}, relaxed_precision {0,1},
//        walk {regular, morton}. (post-proc fixed to ct.store(threadgroup)+flat loop, the proven win.)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef _Float16 fp16_t;
typedef struct { uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio,grid_x,grid_y;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale,qscale; } idx_args;
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

// Kernel template (macros TM/NK/RELAXED/WALK/USE_INT8/QSCALE injected as #defines before this).
static const char *KSRC =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct A { uint n_comp,n_tokens,n_head,head_dim,pos0,ratio,grid_x,grid_y; ulong qts,qhs,wts,irs,sts; float scale,qscale; };\n"
"#if USE_INT8\n typedef int8_t ST; typedef int32_t AT;\n#else\n typedef half ST; typedef float AT;\n#endif\n"
"kernel void k(constant A &a [[buffer(0)]], device const char *q [[buffer(1)]],\n"
"   device const char *weights [[buffer(2)]], device const char *index_comp [[buffer(3)]],\n"
"   device char *scores [[buffer(4)]], threadgroup ST *shared [[threadgroup(0)]],\n"
"   uint2 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {\n"
"  constexpr int TN=32, D=128, NUM_THREADS=128;\n"
"#if WALK\n  uint lin=tgpig.x; uint tx=0,ty=0; for(uint b=0;b<16;b++){ tx|=((lin>>(2*b))&1u)<<b; ty|=((lin>>(2*b+1))&1u)<<b; }\n"
"  if(tx>=a.grid_x||ty>=a.grid_y) return; const uint c0=tx*TN, t0=ty*TM;\n"
"#else\n  const uint c0=tgpig.x*TN, t0=tgpig.y*TM;\n#endif\n"
"  threadgroup ST *qtg=shared; threadgroup ST *ktg=qtg+TM*NK; threadgroup AT *dot=(threadgroup AT*)(ktg+TN*D);\n"
"  const uint last=min(t0+(uint)TM,a.n_tokens); const uint maxv=last>t0?min((a.pos0+last)/a.ratio,a.n_comp):0u;\n"
"  if(c0>=maxv){ for(uint i=tid;i<TM*TN;i+=NUM_THREADS){uint r=i/TN,cc=i-r*TN,tok=t0+r,cp=c0+cc; if(tok<a.n_tokens&&cp<a.n_comp){device float*d=(device float*)(scores+(ulong)tok*a.sts)+cp; *d=-INFINITY;}} return; }\n"
"  for(uint w=tid;w<TN*D;w+=NUM_THREADS){ uint cc=w/D,d=w-cc*D,cp=c0+cc; float v=0.0f; if(cp<a.n_comp){device const float*kr=(device const float*)(index_comp+(ulong)cp*a.irs); v=kr[d];}\n"
"#if USE_INT8\n   ktg[cc*D+d]=(int8_t)clamp(rint(v*a.qscale),-127.0f,127.0f);\n#else\n   ktg[cc*D+d]=(half)v;\n#endif\n  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  float acc[(TM*TN+NUM_THREADS-1)/NUM_THREADS]; for(uint j=0;j<(TM*TN+NUM_THREADS-1)/NUM_THREADS;j++) acc[j]=0.0f;\n"
"  auto tq=tensor(qtg,dextents<int32_t,2>(NK,TM)); auto tk=tensor(ktg,dextents<int32_t,2>(D,TN));\n"
"  auto td=tensor(dot,dextents<int32_t,2>(TM,TN),array<int,2>({1,TM}));\n"
"  matmul2d<matmul2d_descriptor(TN,TM,NK,false,true,RELAXED,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  for(uint head=0;head<a.n_head;head++){\n"
"    auto ct=mm.template get_destination_cooperative_tensor<decltype(tk),decltype(tq),AT>();\n"
"    for(uint16_t i=0;i<ct.get_capacity();i++) if(ct.is_valid_element(i)) ct[i]=(AT)0;\n"
"    for(uint lk=0;lk<D;lk+=NK){ for(uint w=tid;w<TM*NK;w+=NUM_THREADS){uint r=w/NK,k=w-r*NK,tok=t0+r; float v=0.0f; if(tok<a.n_tokens){device const float*qr=(device const float*)(q+(ulong)tok*a.qts+(ulong)head*a.qhs); v=qr[lk+k];}\n"
"#if USE_INT8\n      qtg[r*NK+k]=(int8_t)clamp(rint(v*a.qscale),-127.0f,127.0f);\n#else\n      qtg[r*NK+k]=(half)v;\n#endif\n     }\n"
"      threadgroup_barrier(mem_flags::mem_threadgroup); auto mq=tq.slice(0,0); auto mk=tk.slice(lk,0); mm.run(mk,mq,ct); threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    ct.store(td); threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for(uint j=0;j<(TM*TN+NUM_THREADS-1)/NUM_THREADS;j++){ uint lin2=tid+j*NUM_THREADS; if(lin2<TM*TN){uint r=lin2/TN,cc=lin2-r*TN,tok=t0+r; if(tok<a.n_tokens){device const float*wt=(device const float*)(weights+(ulong)tok*a.wts);\n"
"#if USE_INT8\n      float s=(float)dot[cc*TM+r]/(a.qscale*a.qscale);\n#else\n      float s=dot[cc*TM+r];\n#endif\n      acc[j]+=max(s,0.0f)*(wt[head]*a.scale);} } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  for(uint j=0;j<(TM*TN+NUM_THREADS-1)/NUM_THREADS;j++){ uint lin2=tid+j*NUM_THREADS; if(lin2>=TM*TN) continue; uint r=lin2/TN,cc=lin2-r*TN,tok=t0+r,cp=c0+cc; if(tok<a.n_tokens&&cp<a.n_comp){uint vis=min((a.pos0+tok+1u)/a.ratio,a.n_comp); device float*d=(device float*)(scores+(ulong)tok*a.sts)+cp; *d=cp<vis?acc[j]:-INFINITY;} }\n"
"}\n";

typedef struct { int dtype, TM, NK, relaxed, walk; double ms, gflops, maxrel; int ok; } cfg_t;

int main(int argc,char**argv){
    int T=512,C=4096,H=64,D=128,ratio=4,pos0=8192,iters=8; float scale=0.5f, qscale=16.0f;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--T")&&i+1<argc)T=atoi(argv[++i]); else if(!strcmp(argv[i],"--C")&&i+1<argc)C=atoi(argv[++i]); else if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atoi(argv[++i]); }
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue>q=[dev newCommandQueue];
        id<MTLBuffer>Q=[dev newBufferWithLength:(NSUInteger)T*H*D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)T*H*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>K=[dev newBufferWithLength:(NSUInteger)C*D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>S=[dev newBufferWithLength:(NSUInteger)T*C*sizeof(float) options:MTLResourceStorageModeShared];
        float*Qp=Q.contents; for(NSUInteger i=0;i<(NSUInteger)T*H*D;i++) Qp[i]=(((int)((i*17u+3u)%61)-30))*0.03f;
        float*Wp=W.contents; for(NSUInteger i=0;i<(NSUInteger)T*H;i++) Wp[i]=(((int)((i*29u+5u)%23))*0.05f);
        float*Kp=K.contents; for(NSUInteger i=0;i<(NSUInteger)C*D;i++) Kp[i]=(((int)((i*13u+7u)%59)-29))*0.03f;
        // CPU ref (float; used for both, with looser tol for int8)
        double *ref=calloc((size_t)T*C,sizeof(double));
        for(int t=0;t<T;t++){uint vis=(pos0+t+1)/ratio; if(vis>(uint)C)vis=C; for(int c=0;c<C;c++){ double acc=0; for(int h=0;h<H;h++){double dt=0; for(int d=0;d<D;d++) dt+=(double)(fp16_t)Qp[(size_t)t*H*D+h*D+d]*(double)(fp16_t)Kp[(size_t)c*D+d]; acc+=fmax(dt,0.0)*((double)Wp[(size_t)t*H+h]*scale);} ref[(size_t)t*C+c]=(c<(int)vis)?acc:-INFINITY; }}
        int TMs[]={16,32,64}, NKs[]={32,64,128};
        cfg_t results[200]; int nres=0;
        printf("dtype TM  NK  relax walk    ms     GF/s    maxrel  status\n");
        for(int dt=0;dt<2;dt++) for(int ti=0;ti<3;ti++) for(int ki=0;ki<3;ki++) for(int rx=0;rx<2;rx++) for(int wk=0;wk<2;wk++){
            int TM=TMs[ti], NK=NKs[ki];
            char def[256]; snprintf(def,sizeof(def),"#define TM %d\n#define NK %d\n#define RELAXED %s\n#define WALK %d\n#define USE_INT8 %d\n",TM,NK,rx?"true":"false",wk,dt);
            char *src=malloc(strlen(def)+strlen(KSRC)+1); strcpy(src,def); strcat(src,KSRC);
            NSError*e=0; MTLCompileOptions*o=[MTLCompileOptions new]; o.languageVersion=MTLLanguageVersion4_0;
            id<MTLLibrary>lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e]; free(src);
            cfg_t cf={dt,TM,NK,rx,wk,0,0,0,0};
            if(!lib){ printf("%-5s %-3d %-3d %-5d %-4s  COMPILE_FAIL\n", dt?"int8":"half",TM,NK,rx,wk?"mort":"reg"); results[nres++]=cf; continue; }
            id<MTLFunction>fn=[lib newFunctionWithName:@"k"]; id<MTLComputePipelineState>ps=[dev newComputePipelineStateWithFunction:fn error:&e];
            if(!ps){ printf("%-5s %-3d %-3d %-5d %-4s  PIPE_FAIL\n",dt?"int8":"half",TM,NK,rx,wk?"mort":"reg"); results[nres++]=cf; continue; }
            uint gx=(C+31)/32, gy=(T+TM-1)/TM;
            idx_args a={ .n_comp=C,.n_tokens=T,.n_head=H,.head_dim=D,.pos0=pos0,.ratio=ratio,.grid_x=gx,.grid_y=gy,
                .q_token_stride=(uint64_t)H*D*4,.q_head_stride=(uint64_t)D*4,.weights_token_stride=(uint64_t)H*4,
                .index_row_stride=(uint64_t)D*4,.score_token_stride=(uint64_t)C*4,.scale=scale,.qscale=qscale };
            NSUInteger esz = dt?1:2; NSUInteger asz = dt?4:4;
            NSUInteger tgmem = ((NSUInteger)TM*NK + (NSUInteger)32*D)*esz + ((NSUInteger)TM*32)*asz;
            MTLSize tg=MTLSizeMake(128,1,1);
            MTLSize grid = wk ? MTLSizeMake((NSUInteger)1<<(2*(int)ceil(log2(fmax(gx,gy)))),1,1) : MTLSizeMake(gx,gy,1);
            // correctness
            { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Q offset:0 atIndex:1];[en setBuffer:W offset:0 atIndex:2];[en setBuffer:K offset:0 atIndex:3];[en setBuffer:S offset:0 atIndex:4];[en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted]; if(cb.error){printf("%-5s %-3d %-3d %-5d %-4s  RUN_FAIL %s\n",dt?"int8":"half",TM,NK,rx,wk?"mort":"reg",cb.error.localizedDescription.UTF8String);results[nres++]=cf;continue;} }
            float*Sp=S.contents; double maxabs=0,maxden=0; long mm=0; for(int t=0;t<T;t++)for(int c=0;c<C;c++){ double rf=ref[(size_t)t*C+c]; float gv=Sp[(size_t)t*C+c]; if(isinf(rf)!=isinf(gv)){mm++;continue;} if(isinf(rf))continue; double ab=fabs((double)gv-rf); if(ab>maxabs){maxabs=ab;maxden=fabs(rf);} }
            double rel=maxden>1e-3?maxabs/maxden:maxabs; double tol = dt?0.25:3e-2;
            int ok = (mm==0 && rel<tol);
            // time
            double t0=now_s(); id<MTLCommandBuffer>cb=[q commandBuffer]; for(int i=0;i<iters;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Q offset:0 atIndex:1];[en setBuffer:W offset:0 atIndex:2];[en setBuffer:K offset:0 atIndex:3];[en setBuffer:S offset:0 atIndex:4];[en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[cb commit];[cb waitUntilCompleted];
            double ms=(now_s()-t0)*1000.0/iters; double gf=2.0*T*C*H*D/(ms*1e6);
            cf.ms=ms;cf.gflops=gf;cf.maxrel=rel;cf.ok=ok; results[nres++]=cf;
            printf("%-5s %-3d %-3d %-5d %-4s  %6.3f  %7.1f  %7.4f  %s\n",dt?"int8":"half",TM,NK,rx,wk?"mort":"reg",ms,gf,rel, ok?"OK":"BADVAL");
        }
        // best among OK
        int best=-1; for(int i=0;i<nres;i++) if(results[i].ok && (best<0||results[i].gflops>results[best].gflops)) best=i;
        if(best>=0){ cfg_t*b=&results[best]; printf("\nBEST: dtype=%s TM=%d NK=%d relaxed=%d walk=%s  %.1f GF/s (%.3f ms, maxrel %.4f)\n",
            b->dtype?"int8":"half",b->TM,b->NK,b->relaxed,b->walk?"morton":"regular",b->gflops,b->ms,b->maxrel); }
        free(ref);
    }
    return 0;
}
