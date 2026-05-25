// Validate metal/nax_fused.metal counted kernels against CPU. Loads the real file.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef _Float16 fp16_t;
#define QK_K 256
typedef struct { fp16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; fp16_t d; fp16_t dmin; } block_q2_K;
static const uint8_t ks[128]={0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255};
static uint64_t grid[256];
static void load_grid(void){FILE*f=fopen("metal/moe.metal","rb");char*b=0;size_t c=0,l=0;int ch;while((ch=fgetc(f))!=EOF){if(l+1>=c){c=c?c*2:65536;b=realloc(b,c);}b[l++]=ch;}fclose(f);b[l]=0;char*p=strstr(b,"ds4_metal_iq2xxs_grid[256] = {");p+=strlen("ds4_metal_iq2xxs_grid[256] = {");for(int i=0;i<256;i++){char*h=strstr(p,"0x");grid[i]=strtoull(h,&p,16);}free(b);}
static int8_t f2i8(float x,float q){int v=(int)lrintf(fmaxf(fminf(x*q,127.0f),-128.0f));return(int8_t)v;}
static void deq_iq2(const block_iq2_xxs*blk,float q,int8_t*d256){float d=(float)blk->d;for(uint32_t il0=0;il0<16;il0++){uint32_t ib=il0/2,ln=il0&1;const uint16_t*q2=blk->qs+4*ib;uint32_t g=(uint32_t)q2[0]|((uint32_t)q2[1]<<16),s=(uint32_t)q2[2]|((uint32_t)q2[3]<<16);float sc=d*(0.5f+(float)(s>>28))*0.25f;uint32_t c0=il0*16;uint64_t gv0=grid[(g>>(8*(2*ln+0)))&255];uint8_t sg0=ks[(s>>(14*ln))&127];for(uint32_t j=0;j<8;j++)d256[c0+j]=f2i8(sc*(float)((gv0>>(8*j))&255ull)*((sg0&(1u<<j))?-1.0f:1.0f),q);uint64_t gv1=grid[(g>>(8*(2*ln+1)))&255];uint8_t sg1=ks[(s>>(14*ln+7))&127];for(uint32_t j=0;j<8;j++)d256[c0+8+j]=f2i8(sc*(float)((gv1>>(8*j))&255ull)*((sg1&(1u<<j))?-1.0f:1.0f),q);}}
static void deq_q2k(const block_q2_K*blk,float q,int8_t*d256){float d=(float)blk->d,dm=(float)blk->dmin;for(uint32_t il0=0;il0<16;il0++){const uint8_t*qq=blk->qs+32*(il0/8)+16*(il0&1);uint8_t sc=blk->scales[il0];uint32_t il=(il0/2)&3;float co=il>1?(il>2?1.0f/64:1.0f/16):(il>0?0.25f:1.0f);uint8_t mk=il>1?(il>2?192:48):(il>0?12:3);float dl=d*(float)(sc&0xf)*co,ml=dm*(float)(sc>>4);uint32_t c0=il0*16;for(uint32_t j=0;j<16;j++)d256[c0+j]=f2i8(dl*(float)(qq[j]&mk)-ml,q);}}

static id<MTLComputePipelineState> mk(id<MTLDevice>dev,id<MTLLibrary>lib,NSString*n){NSError*e=0;id<MTLFunction>f=[lib newFunctionWithName:n];if(!f){fprintf(stderr,"missing %s\n",n.UTF8String);exit(2);}id<MTLComputePipelineState>p=[dev newComputePipelineStateWithFunction:f error:&e];if(!p){fprintf(stderr,"pipe %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(2);}return p;}

static int test(id<MTLDevice>dev,id<MTLCommandQueue>qq,id<MTLComputePipelineState>ps,int isq2k,int M,int N,int K,float qscale){
    int bpr=K/256; int exp=3; int NE=256;
    size_t wbsz=(size_t)N*bpr*(isq2k?sizeof(block_q2_K):sizeof(block_iq2_xxs));
    id<MTLBuffer>A=[dev newBufferWithLength:(NSUInteger)M*K options:MTLResourceStorageModeShared];
    id<MTLBuffer>W=[dev newBufferWithLength:wbsz options:MTLResourceStorageModeShared];
    id<MTLBuffer>C=[dev newBufferWithLength:(NSUInteger)M*N*sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer>counts=[dev newBufferWithLength:NE*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    int8_t*Ap=A.contents;for(NSUInteger i=0;i<(NSUInteger)M*K;i++)Ap[i]=(int8_t)((int)((i*131u+7u)%127)-63);
    uint32_t*cp=counts.contents;memset(cp,0,NE*sizeof(uint32_t));cp[exp]=M;
    uint32_t rng=77u; int8_t*Wi8=malloc((size_t)N*K),b256[256];
    if(isq2k){block_q2_K*Wp=W.contents;for(NSUInteger i=0;i<(NSUInteger)N*bpr;i++){rng=rng*1664525u+1013904223u;Wp[i].d=(fp16_t)(0.02f+0.0001f*(rng%53));Wp[i].dmin=(fp16_t)(0.01f+0.00005f*(rng%29));for(int j=0;j<QK_K/16;j++){rng=rng*1664525u+1013904223u;Wp[i].scales[j]=rng&0xff;}for(int j=0;j<QK_K/4;j++){rng=rng*1664525u+1013904223u;Wp[i].qs[j]=rng&0xff;}}
        for(int n=0;n<N;n++)for(int kb=0;kb<bpr;kb++){deq_q2k(&Wp[n*bpr+kb],qscale,b256);for(int j=0;j<256;j++)Wi8[(size_t)n*K+kb*256+j]=b256[j];}}
    else{block_iq2_xxs*Wp=W.contents;for(NSUInteger i=0;i<(NSUInteger)N*bpr;i++){rng=rng*1664525u+1013904223u;Wp[i].d=(fp16_t)(0.01f+0.0001f*(rng%97));for(int j=0;j<QK_K/8;j++){rng=rng*1664525u+1013904223u;Wp[i].qs[j]=rng&0xffff;}}
        for(int n=0;n<N;n++)for(int kb=0;kb<bpr;kb++){deq_iq2(&Wp[n*bpr+kb],qscale,b256);for(int j=0;j<256;j++)Wi8[(size_t)n*K+kb*256+j]=b256[j];}}
    int32_t*Cref=calloc((size_t)M*N,sizeof(int32_t));
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){long s=0;for(int k=0;k<K;k++)s+=(long)Ap[(size_t)m*K+k]*(long)Wi8[(size_t)n*K+k];Cref[(size_t)m*N+n]=(int32_t)s;}
    uint32_t uexp=exp,uN=N,uK=K;
    MTLSize grid=MTLSizeMake((uN+31)/32,(M+63)/64,1);MTLSize tg=MTLSizeMake((NSUInteger)ps.threadExecutionWidth*4,1,1);
    id<MTLCommandBuffer>cb=[qq commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps];[e setBuffer:A offset:0 atIndex:0];[e setBuffer:W offset:0 atIndex:1];[e setBuffer:C offset:0 atIndex:2];
    [e setBuffer:counts offset:0 atIndex:3];[e setBytes:&uexp length:4 atIndex:4];[e setBytes:&uN length:4 atIndex:5];[e setBytes:&uK length:4 atIndex:6];[e setBytes:&qscale length:4 atIndex:7];
    [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];[e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.error){fprintf(stderr,"cb err %s\n",cb.error.localizedDescription.UTF8String);return 1;}
    int32_t*Cp=C.contents;long mx=0,mm=0;for(NSUInteger i=0;i<(NSUInteger)M*N;i++){long d=labs((long)Cp[i]-(long)Cref[i]);if(d>mx)mx=d;if(d)mm++;}
    printf("%s M=%d N=%d K=%d max_abs=%ld mism=%ld/%lld %s\n",isq2k?"q2k":"iq2",M,N,K,mx,mm,(long long)M*N,mx==0?"OK":"FAIL");
    free(Wi8);free(Cref);return mx==0?0:1;
}

int main(void){
    load_grid();
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[dev newCommandQueue];
        NSError*e=0;NSString*src=[NSString stringWithContentsOfFile:@"metal/nax_fused.metal" encoding:NSUTF8StringEncoding error:&e];
        if(!src){fprintf(stderr,"read fail %s\n",e.localizedDescription.UTF8String);return 2;}
        MTLCompileOptions*o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:src options:o error:&e];
        if(!lib){fprintf(stderr,"compile fail:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>iq2=mk(dev,lib,@"ds4_mpp_iq2_i8_i32_counted");
        id<MTLComputePipelineState>q2k=mk(dev,lib,@"ds4_mpp_q2k_i8_i32_counted");
        int rc=0;
        rc|=test(dev,q,iq2,0,64,2048,7168,512.0f);
        rc|=test(dev,q,iq2,0,100,2048,7168,512.0f);
        rc|=test(dev,q,iq2,0,512,2048,7168,512.0f);
        rc|=test(dev,q,q2k,1,64,7168,2048,512.0f);
        rc|=test(dev,q,q2k,1,130,7168,2048,512.0f);
        return rc;
    }
}
