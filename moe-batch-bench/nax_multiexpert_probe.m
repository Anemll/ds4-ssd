// Single-dispatch multi-expert indexed NAX matmul vs per-expert dispatch.
// A = concatenated int8 activations [TOTAL x K] ordered by expert; each output tile
// looks up its expert from tile_expert[] and reads that expert's iq2 weights (fused
// dequant) + contiguous A rows. Goal: keep ~20000 GF/s AND collapse E dispatches->1.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef _Float16 fp16_t;
#define QK_K 256
typedef struct { fp16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
static const uint8_t ks[128]={0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255};
static uint64_t grid[256];
static void load_grid(void){FILE*f=fopen("metal/moe.metal","rb");char*b=0;size_t c=0,l=0;int ch;while((ch=fgetc(f))!=EOF){if(l+1>=c){c=c?c*2:65536;b=realloc(b,c);}b[l++]=ch;}fclose(f);b[l]=0;char*p=strstr(b,"ds4_metal_iq2xxs_grid[256] = {");p+=strlen("ds4_metal_iq2xxs_grid[256] = {");for(int i=0;i<256;i++){char*h=strstr(p,"0x");grid[i]=strtoull(h,&p,16);}free(b);}
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static int8_t f2i8(float x,float q){int v=(int)lrintf(fmaxf(fminf(x*q,127.0f),-128.0f));return(int8_t)v;}
static void deq(const block_iq2_xxs*blk,float q,int8_t*d){float dd=(float)blk->d;for(uint32_t il0=0;il0<16;il0++){uint32_t ib=il0/2,ln=il0&1;const uint16_t*q2=blk->qs+4*ib;uint32_t g=(uint32_t)q2[0]|((uint32_t)q2[1]<<16),s=(uint32_t)q2[2]|((uint32_t)q2[3]<<16);float sc=dd*(0.5f+(float)(s>>28))*0.25f;uint32_t c0=il0*16;uint64_t gv0=grid[(g>>(8*(2*ln+0)))&255];uint8_t s0=ks[(s>>(14*ln))&127];for(uint32_t j=0;j<8;j++)d[c0+j]=f2i8(sc*(float)((gv0>>(8*j))&255ull)*((s0&(1u<<j))?-1.0f:1.0f),q);uint64_t gv1=grid[(g>>(8*(2*ln+1)))&255];uint8_t s1=ks[(s>>(14*ln+7))&127];for(uint32_t j=0;j<8;j++)d[c0+8+j]=f2i8(sc*(float)((gv1>>(8*j))&255ull)*((s1&(1u<<j))?-1.0f:1.0f),q);}}

static const char *tmpl =
"#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal; using namespace mpp::tensor_ops;\n#define QK_K 256\n"
"struct block_iq2_xxs { half d; ushort qs[QK_K/8]; };\n"
"constant uchar KS[128]={%%KS%%};\nconstant ulong GR[256]={%%GR%%};\n"
"static inline int8_t f2i8(float x,float q){return int8_t(int(rint(clamp(x*q,-128.0f,127.0f))));}\n"
"inline void iq2seg(device const block_iq2_xxs*blk,uint seg,float q,threadgroup int8_t*B,uint nn){\n"
"  uint ib=seg/2u,ln=seg&1u; device const ushort*q2=blk->qs+4u*ib; uint g=uint(q2[0])|(uint(q2[1])<<16),s=uint(q2[2])|(uint(q2[3])<<16);\n"
"  float sc=float(blk->d)*(0.5f+float(s>>28))*0.25f; uint c0=seg*16u;\n"
"  ulong gv0=GR[(g>>(8u*(2u*ln+0u)))&255u]; uchar s0=KS[(s>>(14u*ln))&127u];\n"
"  for(uint j=0;j<8u;j++){float v=sc*float((gv0>>(8u*j))&255ul)*((s0&(1u<<j))?-1.0f:1.0f);B[(c0+j)*32u+nn]=f2i8(v,q);}\n"
"  ulong gv1=GR[(g>>(8u*(2u*ln+1u)))&255u]; uchar s1=KS[(s>>(14u*ln+7u))&127u];\n"
"  for(uint j=0;j<8u;j++){float v=sc*float((gv1>>(8u*j))&255ul)*((s1&(1u<<j))?-1.0f:1.0f);B[(c0+8u+j)*32u+nn]=f2i8(v,q);}\n}\n"
// Single dispatch: tgid.y = global tile index; te[]=expert, tr0[]=A row offset, trc[]=rows in tile.
"kernel void multi(device int8_t *A [[buffer(0)]], device const block_iq2_xxs *Wq [[buffer(1)]],\n"
"                  device int32_t *C [[buffer(2)]], device const uint *te [[buffer(3)]],\n"
"                  device const uint *tr0 [[buffer(4)]], device const uint *trc [[buffer(5)]],\n"
"                  constant uint &N [[buffer(6)]], constant uint &K [[buffer(7)]], constant float &wq [[buffer(8)]],\n"
"                  uint2 tgid [[threadgroup_position_in_grid]], uint tidx [[thread_index_in_threadgroup]]){\n"
"  const uint n0=tgid.x*32u; if(n0>=N) return;\n"
"  const uint expert=te[tgid.y], a0=tr0[tgid.y], rows=trc[tgid.y], bpr=K/256u;\n"
"  device int8_t *Ae = A + (ulong)a0*K; device int32_t *Ce = C + (ulong)a0*N;\n"
"  device const block_iq2_xxs *We = Wq + (ulong)expert*N*bpr;\n"
"  threadgroup int8_t Bt[256*32]; threadgroup int8_t *bp=Bt;\n"
"  constexpr auto desc=matmul2d_descriptor(64,32,256,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);\n"
"  matmul2d<desc,execution_simdgroups<4>> op;\n"
"  auto a0t=tensor(Ae,dextents<int32_t,2>{256,64},array<int32_t,2>{1,(int32_t)K});\n"
"  auto b0t=tensor(bp,dextents<int32_t,2>{32,256},array<int32_t,2>{1,32});\n"
"  auto cT=op.get_destination_cooperative_tensor<decltype(a0t),decltype(b0t),int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i){ if(cT.is_valid_element(i)) cT[i]=0; }\n"
"  for(uint kb=0;kb<bpr;++kb){\n"
"    for(uint w=tidx; w<512u; w+=128u){ uint nn=w&31u, seg=w>>5; iq2seg(We+(n0+nn)*bpr+kb,seg,wq,bp,nn);} \n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tensor(Ae + kb*256u, dextents<int32_t,2>{256,(int32_t)rows}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto mB=tensor(bp, dextents<int32_t,2>{32,256}, array<int32_t,2>{1,32});\n"
"    op.run(mA,mB,cT);\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  auto mC=tensor(Ce + n0, dextents<int32_t,2>{32,(int32_t)rows}, array<int32_t,2>{1,(int32_t)N});\n"
"  cT.store(mC);\n}\n";

static id<MTLComputePipelineState> mk(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=0;id<MTLFunction>f=[l newFunctionWithName:n];if(!f){fprintf(stderr,"miss %s\n",n.UTF8String);exit(2);}id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];if(!p){fprintf(stderr,"pipe:%s\n",e.localizedDescription.UTF8String);exit(2);}return p;}

int main(int argc,char**argv){
    int E=64, rmin=32, rmax=64, N=2048, K=7168, iters=50; float wq=512.0f;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--E")&&i+1<argc)E=atoi(argv[++i]);else if(!strcmp(argv[i],"--rows")&&i+2<argc){rmin=atoi(argv[++i]);rmax=atoi(argv[++i]);}else if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atoi(argv[++i]);}
    load_grid();
    char ksb[1024];{char*p=ksb;for(int i=0;i<128;i++)p+=sprintf(p,"%d,",ks[i]);}
    char*grb=malloc(256*24);{char*p=grb;for(int i=0;i<256;i++)p+=sprintf(p,"0x%016llxul,",(unsigned long long)grid[i]);}
    char*src=malloc(strlen(tmpl)+strlen(ksb)+strlen(grb)+16);
    {const char*a=strstr(tmpl,"%%KS%%");size_t pre=a-tmpl;memcpy(src,tmpl,pre);int off=pre;off+=sprintf(src+off,"%s",ksb);const char*r=a+6;const char*b=strstr(r,"%%GR%%");size_t mid=b-r;memcpy(src+off,r,mid);off+=mid;off+=sprintf(src+off,"%s",grb);strcpy(src+off,b+6);}
    int bpr=K/256;
    int *rows=malloc(E*sizeof(int)),*off=malloc((E+1)*sizeof(int)); off[0]=0; srand(7);
    for(int e=0;e<E;e++){rows[e]=rmin+rand()%(rmax-rmin+1);off[e+1]=off[e]+rows[e];}
    int TOTAL=off[E];
    // tile maps
    int ntiles=0; for(int e=0;e<E;e++) ntiles+=(rows[e]+63)/64;
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[dev newCommandQueue];
        MTLCompileOptions*o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;NSError*e=0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
        if(!lib){fprintf(stderr,"compile:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>ps=mk(dev,lib,@"multi");
        id<MTLBuffer>A=[dev newBufferWithLength:(NSUInteger)TOTAL*K options:MTLResourceStorageModeShared];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)E*N*bpr*sizeof(block_iq2_xxs) options:MTLResourceStorageModeShared];
        id<MTLBuffer>C=[dev newBufferWithLength:(NSUInteger)TOTAL*N*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer>teB=[dev newBufferWithLength:ntiles*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer>tr0B=[dev newBufferWithLength:ntiles*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer>trcB=[dev newBufferWithLength:ntiles*4 options:MTLResourceStorageModeShared];
        int8_t*Ap=A.contents;for(NSUInteger i=0;i<(NSUInteger)TOTAL*K;i++)Ap[i]=(int8_t)((int)((i*131u+7u)%127)-63);
        block_iq2_xxs*Wp=W.contents;uint32_t rng=9u;for(NSUInteger i=0;i<(NSUInteger)E*N*bpr;i++){rng=rng*1664525u+1013904223u;Wp[i].d=(fp16_t)(0.01f+0.0001f*(rng%97));for(int j=0;j<QK_K/8;j++){rng=rng*1664525u+1013904223u;Wp[i].qs[j]=rng&0xffff;}}
        uint32_t*te=teB.contents,*tr0=tr0B.contents,*trc=trcB.contents; int ti=0;
        for(int ex=0;ex<E;ex++){for(int r=0;r<rows[ex];r+=64){te[ti]=ex;tr0[ti]=off[ex]+r;trc[ti]=(rows[ex]-r<64)?(rows[ex]-r):64;ti++;}}
        uint32_t uN=N,uK=K; MTLSize grid=MTLSizeMake((uN+31)/32,ntiles,1);MTLSize tg=MTLSizeMake((NSUInteger)ps.threadExecutionWidth*4,1,1);
        // correctness (sample expert 0 and last)
        {id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBuffer:A offset:0 atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:C offset:0 atIndex:2];[en setBuffer:teB offset:0 atIndex:3];[en setBuffer:tr0B offset:0 atIndex:4];[en setBuffer:trcB offset:0 atIndex:5];[en setBytes:&uN length:4 atIndex:6];[en setBytes:&uK length:4 atIndex:7];[en setBytes:&wq length:4 atIndex:8];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.error){fprintf(stderr,"cb %s\n",cb.error.localizedDescription.UTF8String);return 2;}}
        int8_t*Wi8=malloc((size_t)N*K),b256[256]; long mx=0,mm=0; int checks[2]={0,E-1};
        for(int ci=0;ci<2;ci++){int ex=checks[ci];for(int n=0;n<N;n++)for(int kb=0;kb<bpr;kb++){deq(&Wp[((size_t)ex*N+n)*bpr+kb],wq,b256);for(int j=0;j<256;j++)Wi8[(size_t)n*K+kb*256+j]=b256[j];}
          for(int r=0;r<rows[ex];r++){int arow=off[ex]+r;for(int n=0;n<N;n++){long s=0;for(int k=0;k<K;k++)s+=(long)Ap[(size_t)arow*K+k]*(long)Wi8[(size_t)n*K+k];long got=((int32_t*)C.contents)[(size_t)arow*N+n];long d=labs(got-s);if(d>mx)mx=d;if(d)mm++;}}}
        printf("verify_multi E=%d TOTAL=%d ntiles=%d N=%d K=%d max_abs=%ld mism=%ld %s\n",E,TOTAL,ntiles,N,K,mx,mm,mx==0?"OK":"FAIL");
        // timing: single dispatch
        for(int ph=0;ph<2;ph++){int it=ph?iters:5;double t0=now_s();id<MTLCommandBuffer>cb=[q commandBuffer];for(int i=0;i<it;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBuffer:A offset:0 atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:C offset:0 atIndex:2];[en setBuffer:teB offset:0 atIndex:3];[en setBuffer:tr0B offset:0 atIndex:4];[en setBuffer:trcB offset:0 atIndex:5];[en setBytes:&uN length:4 atIndex:6];[en setBytes:&uK length:4 atIndex:7];[en setBytes:&wq length:4 atIndex:8];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[cb commit];[cb waitUntilCompleted];if(ph){double ms=(now_s()-t0)*1000.0/it;double fl=2.0*TOTAL*N*K;printf("single_dispatch,E=%d,TOTAL=%d,%.4f ms,%.1f GF/s\n",E,TOTAL,ms,fl/(ms*1e6));}}
        return mx==0?0:1;
    }
}
