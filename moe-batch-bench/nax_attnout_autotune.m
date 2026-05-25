// NAX autotuner for the grouped attn_out O-proj kernel (ds4_attn_out_low_q8_nax).
// Templates over NR1{64,128,256}/NR0{32,64}/NK{16,32,64}/relaxed/walk. float(act)xhalf(weight)->float.
// Per group g: low[t][g][m] = sum_k heads[t][g][k] * dequant(Wa[g][m][k] Q8_0). Validate sampled tokens.
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
typedef struct { int32_t ne00,ne02; uint64_t nb01,nb02,nb03; int32_t ne11;
    uint64_t nb10,nb11,nb12,nb13; int32_t ne20,ne21,ne0,ne1; int16_t r2,r3; } mm_id_args;
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static const char *KSRC =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n"
"struct block_q8_0 { half d; char qs[32]; };\n"
"struct A { int ne00,ne02; ulong nb01,nb02,nb03; int ne11; ulong nb10,nb11,nb12,nb13; int ne20,ne21,ne0,ne1; short r2,r3; };\n"
"inline void deq(device const block_q8_0 *xb, short il, thread half4x4 &reg){ device const char*qs=(device const char*)xb->qs; const float d=(float)xb->d; for(int i=0;i<16;i++) reg[i/4][i%4]=(half)((float)qs[i+16*il]*d);}\n"
"kernel void k(constant A &args [[buffer(0)]], device const char *srcA [[buffer(1)]],\n"
"   device const char *srcB [[buffer(2)]], device char *dst [[buffer(3)]],\n"
"   threadgroup char *shmem [[threadgroup(0)]], uint3 tgpig [[threadgroup_position_in_grid]],\n"
"   ushort tiitg [[thread_index_in_threadgroup]]) {\n"
"  constexpr int NL = NK/16, NUM_THREADS = 128;\n"
"  const int K=args.ne00, M=args.ne0, N=args.ne21, G=args.ne1; const int group=tgpig.z;\n"
"#if WALK\n  uint lin=tgpig.x; uint gx=0,gy=0; for(uint b=0;b<16;b++){ gx|=((lin>>(2*b))&1u)<<b; gy|=((lin>>(2*b+1))&1u)<<b; }\n"
"  const int r1=gx*NR1, r0=gy*NR0; if(r1>=N||r0>=M) return;\n"
"#else\n  const int r0=tgpig.y*NR0, r1=tgpig.x*NR1;\n#endif\n"
"  const bool full_tile = r0+NR0<=M && r1+NR1<=N && (K%NK)==0;\n"
"  threadgroup half *sa=(threadgroup half*)shmem; auto tA=tensor(sa,dextents<int32_t,2>(NK,NR0));\n"
"  device float *ptrB=(device float*)(srcB+args.nb11*group); const int strideB=args.nb12/sizeof(float);\n"
"  auto tB=tensor(ptrB,dextents<int32_t,2>(K,N),array<int,2>({1,strideB}));\n"
"  matmul2d<matmul2d_descriptor(NR1,NR0,NK,false,true,RELAXED,matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<4>> mm;\n"
"  auto cT=mm.get_destination_cooperative_tensor<decltype(tB),decltype(tA),float>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i) if(cT.is_valid_element(i)) cT[i]=0.0f;\n"
"  for(int loop_k=0;loop_k<K;loop_k+=NK){\n"
"    for(int work=tiitg;work<NR0*NL;work+=NUM_THREADS){ const int row=work/NL,kc=work%NL,kpos=loop_k+kc*16; const short kbase=kc*16;\n"
"      if(full_tile||r0+row<M){ const int bidx=kpos/32; const short il=(kpos/16)%2; device const block_q8_0*rp=(device const block_q8_0*)(srcA+args.nb01*(r0+row)+group*args.nb02); half4x4 t; deq(rp+bidx,il,t); for(short i=0;i<16;i++) sa[row*NK+kbase+i]=(full_tile||kpos+i<K)?t[i/4][i%4]:(half)0; }\n"
"      else { for(short i=0;i<16;i++) sa[row*NK+kbase+i]=(half)0; } }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tA.slice(0,0); auto mB=tB.slice(loop_k,r1); mm.run(mB,mA,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  device float *dg=(device float*)dst+group*M;\n"
"  if(full_tile){ device float *dt=dg+r0+(uint64_t)r1*G*M; auto tD=tensor(dt,dextents<int32_t,2>(NR0,NR1),array<int,2>({1,G*M})); cT.store(tD); }\n"
"  else { auto tD=tensor(dg,dextents<int32_t,2>(M,N),array<int,2>({1,G*M})); auto mD=tD.slice(r0,r1); cT.store(mD); }\n"
"}\n";

typedef struct { int NR1,NR0,NK,relaxed,walk; double ms,gflops,maxrel; int ok; } cfg_t;

int main(int argc,char**argv){
    int TOK=256, G=8, RANK=1024, GD=512, iters=16;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--T")&&i+1<argc)TOK=atoi(argv[++i]); else if(!strcmp(argv[i],"--G")&&i+1<argc)G=atoi(argv[++i]); else if(!strcmp(argv[i],"--RANK")&&i+1<argc)RANK=atoi(argv[++i]); else if(!strcmp(argv[i],"--GD")&&i+1<argc)GD=atoi(argv[++i]); else if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atoi(argv[++i]); }
    int bpr=GD/32;
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue>q=[dev newCommandQueue];
        id<MTLBuffer>H=[dev newBufferWithLength:(NSUInteger)TOK*G*GD*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)G*RANK*bpr*sizeof(block_q8_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer>L=[dev newBufferWithLength:(NSUInteger)TOK*G*RANK*sizeof(float) options:MTLResourceStorageModeShared];
        float*Hp=H.contents; for(NSUInteger i=0;i<(NSUInteger)TOK*G*GD;i++) Hp[i]=(((int)((i*17u+3u)%97)-48))*0.02f;
        block_q8_0*Wp=W.contents; uint32_t rng=7u;
        for(NSUInteger b=0;b<(NSUInteger)G*RANK*bpr;b++){rng=rng*1664525u+1013904223u;Wp[b].d=(fp16_t)(0.005f+0.0002f*(rng%37));for(int j=0;j<32;j++){rng=rng*1664525u+1013904223u;Wp[b].qs[j]=(int8_t)((int)(rng%127)-63);}}
        int Tref=TOK<64?TOK:64;
        double *ref=calloc((size_t)Tref*G*RANK,sizeof(double));
        for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double s=0;for(int kb=0;kb<bpr;kb++){block_q8_0*blk=&Wp[((size_t)g*RANK+m)*bpr+kb];double d=(double)(float)blk->d;for(int j=0;j<32;j++){int k=kb*32+j;s+=(double)Hp[((size_t)t*G+g)*GD+k]*((double)blk->qs[j]*d);}}ref[((size_t)t*G+g)*RANK+m]=s;}
        int NR1s[]={64,128,256}, NR0s[]={32,64}, NKs[]={16,32,64};
        cfg_t results[300]; int nres=0;
        printf("attn_out TOK=%d G=%d RANK=%d GD=%d\nNR1 NR0 NK  relax walk   ms     GF/s    maxrel  status\n",TOK,G,RANK,GD);
        for(int a=0;a<3;a++)for(int b=0;b<2;b++)for(int c=0;c<3;c++)for(int rx=0;rx<2;rx++)for(int wk=0;wk<2;wk++){
            int NR1=NR1s[a],NR0=NR0s[b],NK=NKs[c];
            char def[256]; snprintf(def,sizeof(def),"#define NR1 %d\n#define NR0 %d\n#define NK %d\n#define RELAXED %s\n#define WALK %d\n",NR1,NR0,NK,rx?"true":"false",wk);
            char *src=malloc(strlen(def)+strlen(KSRC)+1); strcpy(src,def); strcat(src,KSRC);
            NSError*e=0; MTLCompileOptions*o=[MTLCompileOptions new]; o.languageVersion=MTLLanguageVersion4_0;
            id<MTLLibrary>lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e]; free(src);
            cfg_t cf={NR1,NR0,NK,rx,wk,0,0,0,0};
            if(!lib){ printf("%-3d %-3d %-3d %-5d %-4s  COMPILE_FAIL %s\n",NR1,NR0,NK,rx,wk?"mort":"reg",e.localizedDescription.UTF8String); results[nres++]=cf; continue; }
            id<MTLFunction>fn=[lib newFunctionWithName:@"k"]; id<MTLComputePipelineState>ps=[dev newComputePipelineStateWithFunction:fn error:&e];
            if(!ps){ printf("%-3d %-3d %-3d %-5d %-4s  PIPE_FAIL\n",NR1,NR0,NK,rx,wk?"mort":"reg"); results[nres++]=cf; continue; }
            uint gx=(TOK+NR1-1)/NR1, gy=(RANK+NR0-1)/NR0;
            mm_id_args a2={.ne00=GD,.ne02=1,.nb01=(uint64_t)bpr*sizeof(block_q8_0),.nb02=(uint64_t)RANK*bpr*sizeof(block_q8_0),.nb03=0,.ne11=0,.nb10=4,.nb11=(uint64_t)GD*4,.nb12=(uint64_t)G*GD*4,.nb13=0,.ne20=0,.ne21=TOK,.ne0=RANK,.ne1=G,.r2=1,.r3=1};
            NSUInteger tgmem=(NSUInteger)NR0*NK*sizeof(fp16_t);
            MTLSize tg=MTLSizeMake(128,1,1);
            MTLSize grid = wk ? MTLSizeMake((NSUInteger)1<<(2*(int)ceil(log2(fmax(gx,gy)))),1,(NSUInteger)G) : MTLSizeMake(gx,gy,(NSUInteger)G);
            { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a2 length:sizeof(a2) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:H offset:0 atIndex:2];[en setBuffer:L offset:0 atIndex:3];[en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted]; if(cb.error){printf("%-3d %-3d %-3d %-5d %-4s  RUN_FAIL\n",NR1,NR0,NK,rx,wk?"mort":"reg");results[nres++]=cf;continue;} }
            float*Lp=L.contents; double maxabs=0,maxden=0; for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double rf=ref[((size_t)t*G+g)*RANK+m];float gv=Lp[((size_t)t*G+g)*RANK+m];double ab=fabs((double)gv-rf);if(ab>maxabs)maxabs=ab;if(fabs(rf)>maxden)maxden=fabs(rf);}
            double rel=maxden>1e-3?maxabs/maxden:maxabs; int ok=(rel<2e-2);
            double t0=now_s(); id<MTLCommandBuffer>cb=[q commandBuffer]; for(int i=0;i<iters;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a2 length:sizeof(a2) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:H offset:0 atIndex:2];[en setBuffer:L offset:0 atIndex:3];[en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[cb commit];[cb waitUntilCompleted];
            double ms=(now_s()-t0)*1000.0/iters; double gf=2.0*(double)TOK*G*RANK*GD/(ms*1e6);
            cf.ms=ms;cf.gflops=gf;cf.maxrel=rel;cf.ok=ok; results[nres++]=cf;
            printf("%-3d %-3d %-3d %-5d %-4s  %6.3f %8.1f  %7.4f  %s\n",NR1,NR0,NK,rx,wk?"mort":"reg",ms,gf,rel,ok?"OK":"BADVAL");
        }
        int best=-1; for(int i=0;i<nres;i++) if(results[i].ok && (best<0||results[i].gflops>results[best].gflops)) best=i;
        if(best>=0){ cfg_t*b=&results[best]; printf("\nBEST: NR1=%d NR0=%d NK=%d relaxed=%d walk=%s  %.1f GF/s (%.3f ms, maxrel %.4f)\n",b->NR1,b->NR0,b->NK,b->relaxed,b->walk?"morton":"regular",b->gflops,b->ms,b->maxrel); }
        free(ref);
    }
    return 0;
}
