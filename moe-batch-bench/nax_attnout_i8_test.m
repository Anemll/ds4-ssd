// Validate ds4_attn_out_low_i8_fused (grouped W8A8 O-proj) vs CPU float ref + measure GF/s.
// Per group g: low[t][g][m] = sum_k heads[t][g][k] * dequant(Wa[g][m][k] Q8_0).
// W8A8: per-(g,m) weight scale, per-(t,g) activation scale, int8xint8->int32, fused rescale -> f32.
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
typedef struct { int32_t K, M, N, G; } attn_i8_args;
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static id<MTLComputePipelineState> mk(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=0;id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:n] error:&e];if(!p){fprintf(stderr,"pipe %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(2);}return p;}
int main(int argc,char**argv){
    int TOK=1024, G=8, RANK=1024, GD=512, iters=12;  // N, G, M, K
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--T")&&i+1<argc)TOK=atoi(argv[++i]); else if(!strcmp(argv[i],"--G")&&i+1<argc)G=atoi(argv[++i]); else if(!strcmp(argv[i],"--RANK")&&i+1<argc)RANK=atoi(argv[++i]); else if(!strcmp(argv[i],"--GD")&&i+1<argc)GD=atoi(argv[++i]); }
    int bpr=GD/32;
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue>q=[dev newCommandQueue];
        NSError*e=0; NSString*src=[NSString stringWithContentsOfFile:@"metal/nax_fused.metal" encoding:NSUTF8StringEncoding error:&e];
        MTLCompileOptions*o=[MTLCompileOptions new]; o.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:src options:o error:&e];
        if(!lib){fprintf(stderr,"compile:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>ps=mk(dev,lib,@"ds4_attn_out_low_i8_fused");
        // host buffers
        float *heads=malloc((size_t)TOK*G*GD*sizeof(float));
        for(size_t i=0;i<(size_t)TOK*G*GD;i++) heads[i]=(((int)((i*17u+3u)%97)-48))*0.02f;
        block_q8_0 *W=malloc((size_t)G*RANK*bpr*sizeof(block_q8_0)); uint32_t rng=7u;
        for(size_t b=0;b<(size_t)G*RANK*bpr;b++){rng=rng*1664525u+1013904223u;W[b].d=(fp16_t)(0.005f+0.0002f*(rng%37));for(int j=0;j<32;j++){rng=rng*1664525u+1013904223u;W[b].qs[j]=(int8_t)((int)(rng%127)-63);}}
        // device int8 buffers + scales
        id<MTLBuffer>Ai=[dev newBufferWithLength:(NSUInteger)TOK*G*GD options:MTLResourceStorageModeShared];
        id<MTLBuffer>Wi=[dev newBufferWithLength:(NSUInteger)G*RANK*GD options:MTLResourceStorageModeShared];
        id<MTLBuffer>As=[dev newBufferWithLength:(NSUInteger)TOK*G*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>Ws=[dev newBufferWithLength:(NSUInteger)G*RANK*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>D=[dev newBufferWithLength:(NSUInteger)TOK*G*RANK*sizeof(float) options:MTLResourceStorageModeShared];
        int8_t*Aip=Ai.contents; float*Asp=As.contents;
        for(int t=0;t<TOK;t++)for(int g=0;g<G;g++){ float mx=0; for(int k=0;k<GD;k++){float v=fabsf(heads[((size_t)t*G+g)*GD+k]); if(v>mx)mx=v;} float sc=mx>0?mx/127.f:1e-9f; Asp[t*G+g]=sc; float inv=1.f/sc; for(int k=0;k<GD;k++){float v=rintf(heads[((size_t)t*G+g)*GD+k]*inv); Aip[((size_t)t*G+g)*GD+k]=(int8_t)(v>127?127:(v<-127?-127:v));} }
        int8_t*Wip=Wi.contents; float*Wsp=Ws.contents;
        for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){ block_q8_0*row=&W[((size_t)g*RANK+m)*bpr]; float mx=0; for(int kb=0;kb<bpr;kb++){float d=(float)row[kb].d;for(int j=0;j<32;j++){float r=fabsf((float)row[kb].qs[j]*d);if(r>mx)mx=r;}} float sc=mx>0?mx/127.f:1e-9f; Wsp[g*RANK+m]=sc; float inv=1.f/sc; for(int kb=0;kb<bpr;kb++){float d=(float)row[kb].d;for(int j=0;j<32;j++){float v=rintf((float)row[kb].qs[j]*d*inv); Wip[((size_t)g*RANK+m)*GD+kb*32+j]=(int8_t)(v>127?127:(v<-127?-127:v));}} }
        // CPU ref (sample first Tref tokens)
        int Tref=TOK<48?TOK:48; double*ref=calloc((size_t)Tref*G*RANK,sizeof(double));
        for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double s=0;block_q8_0*row=&W[((size_t)g*RANK+m)*bpr];for(int kb=0;kb<bpr;kb++){double d=(double)(float)row[kb].d;for(int j=0;j<32;j++){int k=kb*32+j;s+=(double)heads[((size_t)t*G+g)*GD+k]*((double)row[kb].qs[j]*d);}}ref[((size_t)t*G+g)*RANK+m]=s;}
        attn_i8_args a={.K=GD,.M=RANK,.N=TOK,.G=G};
        MTLSize grid=MTLSizeMake((TOK+127)/128,(RANK+31)/32,G), tg=MTLSizeMake(128,1,1);
        NSUInteger tgm=(NSUInteger)32*128 + (NSUInteger)32*128*4;
        // correctness
        { id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Wi offset:0 atIndex:1];[en setBuffer:Ai offset:0 atIndex:2];[en setBuffer:D offset:0 atIndex:3];[en setBuffer:Ws offset:0 atIndex:4];[en setBuffer:As offset:0 atIndex:5];[en setThreadgroupMemoryLength:tgm atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted]; if(cb.error){fprintf(stderr,"cb %s\n",cb.error.localizedDescription.UTF8String);return 2;} }
        float*Dp=D.contents; double maxabs=0,maxden=0;
        for(int t=0;t<Tref;t++)for(int g=0;g<G;g++)for(int m=0;m<RANK;m++){double rf=ref[((size_t)t*G+g)*RANK+m]; float gv=Dp[(size_t)t*G*RANK + (size_t)g*RANK + m]; double ab=fabs((double)gv-rf); if(ab>maxabs)maxabs=ab; if(fabs(rf)>maxden)maxden=fabs(rf);}
        double rel=maxabs/fmax(maxden,1e-3);
        // time
        double t0=now_s(); id<MTLCommandBuffer>cb=[q commandBuffer]; for(int i=0;i<iters;i++){id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];[en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Wi offset:0 atIndex:1];[en setBuffer:Ai offset:0 atIndex:2];[en setBuffer:D offset:0 atIndex:3];[en setBuffer:Ws offset:0 atIndex:4];[en setBuffer:As offset:0 atIndex:5];[en setThreadgroupMemoryLength:tgm atIndex:0];[en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];}[cb commit];[cb waitUntilCompleted];
        double ms=(now_s()-t0)*1000.0/iters; double gf=2.0*(double)TOK*G*RANK*GD/(ms*1e6);
        printf("attn_out_i8 TOK=%d G=%d RANK=%d GD=%d : %.3f ms  %.1f GF/s  rel=%.4f %s\n",TOK,G,RANK,GD,ms,gf,rel, rel<2e-2?"OK":"FAIL");
        return rel<2e-2?0:1;
    }
}
