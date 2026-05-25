// Validate ds4_dense_q8_nax (direct-RHS port) in metal/nax_fused.metal.
// out[tokens x O](f32) = act[tokens x K](f32) * dequant(W[O x K] Q8_0). dst[token*O + o].
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
static id<MTLComputePipelineState> mk(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=0;id<MTLFunction>f=[l newFunctionWithName:n];if(!f){fprintf(stderr,"miss %s\n",n.UTF8String);exit(2);}id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];if(!p){fprintf(stderr,"pipe:%s\n",e.localizedDescription.UTF8String);exit(2);}return p;}

static int test(id<MTLDevice>dev,id<MTLCommandQueue>q,id<MTLComputePipelineState>ps,int TOK,int O,int Kk,int iters){
    int bpr=Kk/32;
    id<MTLBuffer>A=[dev newBufferWithLength:(NSUInteger)TOK*Kk*sizeof(float) options:MTLResourceStorageModeShared]; // activation [tokens x K]
    id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)O*bpr*sizeof(block_q8_0) options:MTLResourceStorageModeShared]; // weight [O x K]
    id<MTLBuffer>C=[dev newBufferWithLength:(NSUInteger)TOK*O*sizeof(float) options:MTLResourceStorageModeShared];
    float*Ap=A.contents; for(NSUInteger i=0;i<(NSUInteger)TOK*Kk;i++) Ap[i]=(((int)((i*17u+3u)%97)-48))*0.02f;
    block_q8_0*Wp=W.contents; uint32_t rng=123u;
    for(NSUInteger b=0;b<(NSUInteger)O*bpr;b++){ rng=rng*1664525u+1013904223u; Wp[b].d=(fp16_t)(0.005f+0.0002f*(rng%37)); for(int j=0;j<32;j++){rng=rng*1664525u+1013904223u; Wp[b].qs[j]=(int8_t)((int)(rng%127)-63);} }
    double *ref=calloc((size_t)TOK*O,sizeof(double));
    for(int t=0;t<TOK;t++)for(int o=0;o<O;o++){ double s=0; for(int kb=0;kb<bpr;kb++){ block_q8_0*blk=&Wp[(size_t)o*bpr+kb]; double d=(double)(float)blk->d; for(int j=0;j<32;j++){int k=kb*32+j; s+=(double)Ap[(size_t)t*Kk+k]*((double)blk->qs[j]*d);} } ref[(size_t)t*O+o]=s; }
    mm_args a={ .ne00=Kk,.ne02=1,.nb01=(uint64_t)bpr*34,.nb02=0,.nb03=0,.ne12=1,.nb10=4,.nb11=(uint64_t)Kk*4,.nb12=0,.nb13=0,.ne0=O,.ne1=TOK,.r2=1,.r3=1 };
    MTLSize grid=MTLSizeMake((TOK+127)/128,(O+63)/64,1); MTLSize tg=MTLSizeMake(128,1,1); NSUInteger tgmem=64*32*sizeof(fp16_t);
    id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps];[e setBytes:&a length:sizeof(a) atIndex:0];[e setBuffer:W offset:0 atIndex:1];[e setBuffer:A offset:0 atIndex:2];[e setBuffer:C offset:0 atIndex:3];
    [e setThreadgroupMemoryLength:tgmem atIndex:0];[e dispatchThreadgroups:grid threadsPerThreadgroup:tg];[e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.error){fprintf(stderr,"cb %s\n",cb.error.localizedDescription.UTF8String);return 1;}
    float*Cp=C.contents; double maxabs=0,maxden=0; for(NSUInteger i=0;i<(NSUInteger)TOK*O;i++){ double ab=fabs((double)Cp[i]-ref[i]); double den=fmax(fabs(ref[i]),0.5); if(ab>maxabs){maxabs=ab;} if(den>maxden)maxden=den; }
    double rel=maxabs/fmax(maxden,1e-3); int ok=rel<2e-2;
    printf("dense_q8_drhs tok=%d O=%d K=%d max_abs=%.4g rel~%.4g %s\n",TOK,O,Kk,maxabs,rel,ok?"OK":"FAIL");
    for(int ph=0;ph<2;ph++){int it=ph?iters:5;double t0=now_s();id<MTLCommandBuffer>c2=[q commandBuffer];for(int i=0;i<it;i++){id<MTLComputeCommandEncoder>en=[c2 computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:A offset:0 atIndex:2];[en setBuffer:C offset:0 atIndex:3];[en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[c2 commit];[c2 waitUntilCompleted];if(ph){double ms=(now_s()-t0)*1000.0/it;printf("  %.4f ms  %.1f GF/s\n",ms,2.0*TOK*O*Kk/(ms*1e6));}}
    free(ref); return ok?0:1;
}
int main(void){
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[dev newCommandQueue];
        NSError*e=0;NSString*src=[NSString stringWithContentsOfFile:@"metal/nax_fused.metal" encoding:NSUTF8StringEncoding error:&e];
        if(!src){fprintf(stderr,"read fail\n");return 2;}
        MTLCompileOptions*o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:src options:o error:&e];
        if(!lib){fprintf(stderr,"compile:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>ps=mk(dev,lib,@"ds4_dense_q8_nax");
        int rc=0;
        rc|=test(dev,q,ps,512,1024,4096,50);   // q_a-like (tokens x out=1024 x in=4096)
        rc|=test(dev,q,ps,512,32768,1024,20);  // q_b-like
        rc|=test(dev,q,ps,100,512,4096,50);    // kv-like, token tail
        return rc;
    }
}
