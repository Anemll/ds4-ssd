// Validate ds4_indexer_scores_nax in metal/nax_fused.metal vs CPU.
// scores[t][c] = (c<visible) ? sum_head relu(Q[t][h].K[c]) * w[t][h] * scale : -INF
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef _Float16 fp16_t;
typedef struct { uint32_t n_comp,n_tokens,n_head,head_dim,pos0,ratio;
    uint64_t q_token_stride,q_head_stride,weights_token_stride,index_row_stride,score_token_stride;
    float scale; } idx_args;
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static id<MTLComputePipelineState> mk(id<MTLDevice>d,id<MTLLibrary>l,NSString*n){NSError*e=0;id<MTLFunction>f=[l newFunctionWithName:n];if(!f){fprintf(stderr,"miss %s\n",n.UTF8String);exit(2);}id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];if(!p){fprintf(stderr,"pipe:%s\n",e.localizedDescription.UTF8String);exit(2);}return p;}

int main(int argc,char**argv){
    int T=128, C=256, H=64, D=128, ratio=4, pos0=4096, iters=20; float scale=0.5f;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--T")&&i+1<argc)T=atoi(argv[++i]); else if(!strcmp(argv[i],"--C")&&i+1<argc)C=atoi(argv[++i]); else if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atoi(argv[++i]); }
    @autoreleasepool{
        id<MTLDevice>dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>q=[dev newCommandQueue];
        NSError*e=0;NSString*src=[NSString stringWithContentsOfFile:@"metal/nax_fused.metal" encoding:NSUTF8StringEncoding error:&e];
        if(!src){fprintf(stderr,"read fail\n");return 2;}
        MTLCompileOptions*o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion4_0;
        id<MTLLibrary>lib=[dev newLibraryWithSource:src options:o error:&e];
        if(!lib){fprintf(stderr,"compile:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLComputePipelineState>ps=mk(dev,lib,@"ds4_indexer_scores_nax");
        id<MTLBuffer>Q=[dev newBufferWithLength:(NSUInteger)T*H*D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>W=[dev newBufferWithLength:(NSUInteger)T*H*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>K=[dev newBufferWithLength:(NSUInteger)C*D*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer>S=[dev newBufferWithLength:(NSUInteger)T*C*sizeof(float) options:MTLResourceStorageModeShared];
        float*Qp=Q.contents; for(NSUInteger i=0;i<(NSUInteger)T*H*D;i++) Qp[i]=(((int)((i*17u+3u)%61)-30))*0.03f;
        float*Wp=W.contents; for(NSUInteger i=0;i<(NSUInteger)T*H;i++) Wp[i]=(((int)((i*29u+5u)%23))*0.05f);
        float*Kp=K.contents; for(NSUInteger i=0;i<(NSUInteger)C*D;i++) Kp[i]=(((int)((i*13u+7u)%59)-29))*0.03f;
        idx_args a={ .n_comp=C,.n_tokens=T,.n_head=H,.head_dim=D,.pos0=pos0,.ratio=ratio,
            .q_token_stride=(uint64_t)H*D*4,.q_head_stride=(uint64_t)D*4,.weights_token_stride=(uint64_t)H*4,
            .index_row_stride=(uint64_t)D*4,.score_token_stride=(uint64_t)C*4,.scale=scale };
        MTLSize grid=MTLSizeMake((C+31)/32,(T+15)/16,1); MTLSize tg=MTLSizeMake(128,1,1);
        NSUInteger tgmem=(16*32+32*128)*sizeof(fp16_t)+(16*32)*sizeof(float);
        id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>en=[cb computeCommandEncoder];
        [en setComputePipelineState:ps];[en setBytes:&a length:sizeof(a) atIndex:0];[en setBuffer:Q offset:0 atIndex:1];[en setBuffer:W offset:0 atIndex:2];[en setBuffer:K offset:0 atIndex:3];[en setBuffer:S offset:0 atIndex:4];
        [en setThreadgroupMemoryLength:tgmem atIndex:0];
        [en dispatchThreadgroups:grid threadsPerThreadgroup:tg];[en endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.error){fprintf(stderr,"cb %s\n",cb.error.localizedDescription.UTF8String);return 2;}
        // CPU ref (half-staged dot to match), tolerance. Cap to a token sample so the
        // O(T*C*H*D) reference stays feasible at large T (GPU timing below covers full T).
        float*Sp=S.contents; double maxabs=0,maxden=0; long maskmism=0;
        int Tref = T < 128 ? T : 128;
        for(int t=0;t<Tref;t++){ uint visible=(pos0+t+1)/ratio; if(visible>(uint)C)visible=C;
            for(int c=0;c<C;c++){ double acc=0;
                for(int h=0;h<H;h++){ double dot=0; for(int d=0;d<D;d++){ dot+=(double)(fp16_t)Qp[(size_t)t*H*D+h*D+d]*(double)(fp16_t)Kp[(size_t)c*D+d]; } acc+=fmax(dot,0.0)*((double)Wp[(size_t)t*H+h]*scale); }
                float ref = (c<(int)visible)? (float)acc : -INFINITY;
                float got = Sp[(size_t)t*C+c];
                if(isinf(ref)!=isinf(got)){ maskmism++; continue; }
                if(isinf(ref)) continue;
                double a2=fabs((double)got-(double)ref); if(a2>maxabs){maxabs=a2;maxden=fabs(ref);} }
        }
        double rel = maxden>1e-4?maxabs/maxden:maxabs;
        printf("indexer_nax T=%d C=%d H=%d max_abs=%.4g (ref~%.4g, rel~%.4g) mask_mism=%ld %s\n",T,C,H,maxabs,maxden,rel,maskmism, (rel<3e-2 && maskmism==0)?"OK":"FAIL");
        for(int ph=0;ph<2;ph++){int it=ph?iters:5;double t0=now_s();id<MTLCommandBuffer>c2=[q commandBuffer];for(int i=0;i<it;i++){id<MTLComputeCommandEncoder>e2=[c2 computeCommandEncoder];[e2 setComputePipelineState:ps];[e2 setBytes:&a length:sizeof(a) atIndex:0];[e2 setBuffer:Q offset:0 atIndex:1];[e2 setBuffer:W offset:0 atIndex:2];[e2 setBuffer:K offset:0 atIndex:3];[e2 setBuffer:S offset:0 atIndex:4];[e2 setThreadgroupMemoryLength:tgmem atIndex:0];[e2 dispatchThreadgroups:grid threadsPerThreadgroup:tg];[e2 endEncoding];}[c2 commit];[c2 waitUntilCompleted];if(ph){double ms=(now_s()-t0)*1000.0/it;printf("  %.4f ms  %.1f GF/s\n",ms,2.0*T*C*H*D/(ms*1e6));}}
        return (rel<3e-2 && maskmism==0)?0:1;
    }
}
