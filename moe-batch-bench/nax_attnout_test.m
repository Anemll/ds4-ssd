// Validate ds4_attn_out_low_q8_nax (grouped O-proj) in metal/nax_fused.metal.
// Per group g: low[t][g][m] = sum_k heads[t][g][k] * dequant(Wa[g][m][k] Q8_0).
// layouts: heads[t][g][k] (token-major), Wa[g][m] Q8_0 rows, low[t][g*rank+m].
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef _Float16 fp16_t;
typedef struct { fp16_t d; int8_t qs[32]; } block_q8_0;
typedef struct { int32_t ne00,ne02; uint64_t nb01,nb02,nb03; int32_t ne11;
    uint64_t nb10,nb11,nb12,nb13; int32_t ne20,ne21,ne0,ne1; int16_t r2,r3; } mm_id_args;
static id<MTLComputePipelineState> mk(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=0;id<MTLFunction>f=[l newFunctionWithName:n];if(!f){fprintf(stderr,"miss %s\n",n.UTF8String);exit(2);}id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];if(!p){fprintf(stderr,"pipe:%s\n",e.localizedDescription.UTF8String);exit(2);}return p;}
int main(void){
    int TOK=256, G=8, RANK=1024, GD=512;  // tokens, groups, rank(M), group_dim(K)
    int bpr=GD/32;
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[dev newCommandQueue];
        NSError*e=0;NSString*src=[NSString stringWithContentsOfFile:@"metal/nax_fused.metal" encoding:NSUTF8StringEncoding error:&e];
        if(!src){fprintf(stderr,"read fail\n");return 2;}
        MTLCompileOptions*o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:src options:o error:&e];
        if(!lib){fprintf(stderr,"compile:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>ps=mk(dev,lib,@"ds4_attn_out_low_q8_nax");
        // heads[tok][g][k], Wa[g][m] blocks, low[tok][g*RANK+m]
        id<MTLBuffer>H=[dev newBufferWithLength:(NSUInteger)TOK*G*GD*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)G*RANK*bpr*sizeof(block_q8_0) options:MTLResourceStorageModeShared];
        id<MTLBuffer>L=[dev newBufferWithLength:(NSUInteger)TOK*G*RANK*sizeof(float) options:MTLResourceStorageModeShared];
        float*Hp=H.contents; for(NSUInteger i=0;i<(NSUInteger)TOK*G*GD;i++) Hp[i]=(((int)((i*17u+3u)%97)-48))*0.02f;
        block_q8_0*Wp=W.contents; uint32_t rng=7u;
        for(NSUInteger b=0;b<(NSUInteger)G*RANK*bpr;b++){rng=rng*1664525u+1013904223u;Wp[b].d=(fp16_t)(0.005f+0.0002f*(rng%37));for(int j=0;j<32;j++){rng=rng*1664525u+1013904223u;Wp[b].qs[j]=(int8_t)((int)(rng%127)-63);}}
        // CPU ref (sample tokens)
        double *ref=calloc((size_t)TOK*G*RANK,sizeof(double));
        int Tref = TOK<64?TOK:64;
        for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double s=0;for(int kb=0;kb<bpr;kb++){block_q8_0*blk=&Wp[((size_t)g*RANK+m)*bpr+kb];double d=(double)(float)blk->d;for(int j=0;j<32;j++){int k=kb*32+j;s+=(double)Hp[((size_t)t*G+g)*GD+k]*((double)blk->qs[j]*d);}}ref[((size_t)t*G+g)*RANK+m]=s;}
        mm_id_args a={.ne00=GD,.ne02=1,.nb01=(uint64_t)bpr*34,.nb02=(uint64_t)RANK*bpr*34,.nb03=0,.ne11=0,.nb10=4,.nb11=(uint64_t)GD*4,.nb12=(uint64_t)G*GD*4,.nb13=0,.ne20=0,.ne21=TOK,.ne0=RANK,.ne1=G,.r2=1,.r3=1};
        MTLSize grid=MTLSizeMake((TOK+127)/128,(RANK+63)/64,G); MTLSize tg=MTLSizeMake(128,1,1); NSUInteger tgmem=64*32*sizeof(fp16_t);
        id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
        [en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:W offset:0 atIndex:1];[en setBuffer:H offset:0 atIndex:2];[en setBuffer:L offset:0 atIndex:3];
        [en setThreadgroupMemoryLength:tgmem atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.error){fprintf(stderr,"cb %s\n",cb.error.localizedDescription.UTF8String);return 2;}
        float*Lp=L.contents; double maxabs=0,maxden=0;
        for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double rf=ref[((size_t)t*G+g)*RANK+m];float gv=Lp[((size_t)t*G+g)*RANK+m];double ab=fabs((double)gv-rf);if(ab>maxabs)maxabs=ab;if(fabs(rf)>maxden)maxden=fabs(rf);}
        double rel=maxabs/fmax(maxden,1e-3);
        printf("attn_out_low_nax TOK=%d G=%d RANK=%d GD=%d max_abs=%.4g rel~%.4g %s\n",TOK,G,RANK,GD,maxabs,rel,rel<2e-2?"OK":"FAIL");
        return rel<2e-2?0:1;
    }
}
