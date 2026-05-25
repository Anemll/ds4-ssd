// nax_fused_probe: validate a fused iq2_xxs-dequant + NAX int8 matmul kernel.
//
// Compares three paths for C[MxN] (int32) = A[MxK](int8) * dequant_i8(Wq[NxK] iq2_xxs):
//   1) baseline: separate dequant-to-global (transposed i8) + mpp i8xi8 matmul
//   2) fused:    matmul2d that dequants iq2_xxs weight tiles into threadgroup memory
//
// Goal: prove fused is correct (bit-exact int32) and faster (no global i8 weight
// round-trip), in isolation, with no 86GB model load.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef _Float16 fp16_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---- host-side iq2_xxs replicas (must match metal/moe.metal) ----
#define QK_K 256
typedef struct { fp16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs; // 2 + 32*2 = 66 bytes
typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; fp16_t d; fp16_t dmin; } block_q2_K; // 16+64+2+2=84

static const uint8_t h_ksigns[128] = {
      0,129,130,  3,132,  5,  6,135,136,  9, 10,139, 12,141,142, 15,
    144, 17, 18,147, 20,149,150, 23, 24,153,154, 27,156, 29, 30,159,
    160, 33, 34,163, 36,165,166, 39, 40,169,170, 43,172, 45, 46,175,
     48,177,178, 51,180, 53, 54,183,184, 57, 58,187, 60,189,190, 63,
    192, 65, 66,195, 68,197,198, 71, 72,201,202, 75,204, 77, 78,207,
     80,209,210, 83,212, 85, 86,215,216, 89, 90,219, 92,221,222, 95,
     96,225,226, 99,228,101,102,231,232,105,106,235,108,237,238,111,
    240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255,
};
static uint64_t h_iq2xxs_grid[256];

static void load_grid_from_metal(const char *path) {
    // Parse the 256 hex grid constants out of metal/moe.metal so host and GPU agree.
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    char *buf = NULL; size_t cap = 0, len = 0; int c;
    while ((c = fgetc(f)) != EOF) { if (len+1>=cap){cap=cap?cap*2:65536;buf=realloc(buf,cap);} buf[len++]=(char)c; }
    fclose(f); buf[len]=0;
    const char *marker = "ds4_metal_iq2xxs_grid[256] = {";
    char *p = strstr(buf, marker);
    if (!p) { fprintf(stderr, "grid marker not found\n"); exit(2); }
    p += strlen(marker);
    int n = 0;
    while (n < 256) {
        char *hx = strstr(p, "0x");
        if (!hx) break;
        h_iq2xxs_grid[n++] = strtoull(hx, &p, 16);
    }
    if (n != 256) { fprintf(stderr, "parsed %d grid entries\n", n); exit(2); }
    free(buf);
}

static int8_t f2i8(float x, float qscale) {
    int v = (int)lrintf(fmaxf(fminf(x * qscale, 127.0f), -128.0f));
    return (int8_t)v;
}

// Dequant one iq2_xxs block (256 values) to int8 with qscale, into dst[0..256] (col order).
static void cpu_dequant_block(const block_iq2_xxs *blk, float qscale, int8_t *dst256) {
    const float d = (float)blk->d;
    for (uint32_t il0 = 0; il0 < 16; il0++) {
        const uint32_t ib32 = il0 / 2u, lane = il0 & 1u;
        const uint16_t *q2 = blk->qs + 4u * ib32;
        const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const float scale = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
        const uint32_t col0 = il0 * 16u;
        const uint64_t gv0 = h_iq2xxs_grid[(aux32_g >> (8u * (2u*lane+0u))) & 255u];
        const uint8_t sign0 = h_ksigns[(aux32_s >> (14u*lane)) & 127u];
        for (uint32_t j = 0; j < 8u; j++) {
            float v = scale * (float)((gv0 >> (8u*j)) & 255ull) * ((sign0 & (1u<<j)) ? -1.0f : 1.0f);
            dst256[col0 + j] = f2i8(v, qscale);
        }
        const uint64_t gv1 = h_iq2xxs_grid[(aux32_g >> (8u * (2u*lane+1u))) & 255u];
        const uint8_t sign1 = h_ksigns[(aux32_s >> (14u*lane+7u)) & 127u];
        for (uint32_t j = 0; j < 8u; j++) {
            float v = scale * (float)((gv1 >> (8u*j)) & 255ull) * ((sign1 & (1u<<j)) ? -1.0f : 1.0f);
            dst256[col0 + 8u + j] = f2i8(v, qscale);
        }
    }
}

static void cpu_dequant_q2k_block(const block_q2_K *blk, float qscale, int8_t *dst256) {
    const float d = (float)blk->d, dmin = (float)blk->dmin;
    for (uint32_t il0 = 0; il0 < 16; il0++) {
        const uint8_t *q = blk->qs + 32u*(il0/8u) + 16u*(il0&1u);
        const uint8_t sc = blk->scales[il0];
        const uint32_t il = (il0/2u) & 3u;
        const float coef = il>1u ? (il>2u ? 1.0f/64.0f : 1.0f/16.0f) : (il>0u ? 1.0f/4.0f : 1.0f);
        const uint8_t mask = il>1u ? (il>2u ? 192 : 48) : (il>0u ? 12 : 3);
        const float dl = d * (float)(sc & 0x0fu) * coef;
        const float ml = dmin * (float)(sc >> 4);
        const uint32_t col0 = il0 * 16u;
        for (uint32_t j = 0; j < 16u; j++)
            dst256[col0 + j] = f2i8(dl * (float)(q[j] & mask) - ml, qscale);
    }
}

static const char *metal_src =
"#include <metal_stdlib>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"using namespace metal;\n"
"using namespace mpp::tensor_ops;\n"
"#define QK_K 256\n"
"struct block_iq2_xxs { half d; ushort qs[QK_K/8]; };\n"
"struct block_q2_K { uchar scales[QK_K/16]; uchar qs[QK_K/4]; half d; half dmin; };\n"
"constant uchar ksigns_iq2xs[128] = {\n"
"  0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,\n"
"  160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,\n"
"  192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,\n"
"  96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255};\n"
"constant ulong IQ2XXS_GRID[256] = {\n"
"%%GRID%%\n"
"};\n"
"static inline char f2i8(float x, float qscale){ int v=int(rint(clamp(x*qscale,-128.0f,127.0f))); return char(v);} \n"
// Dequant block (n,kb) segment seg(0..15) -> 16 int8 values into Btile column nn.
"inline void dequant_seg_to_tg(device const block_iq2_xxs *blk, uint seg, float qscale,\n"
"                              threadgroup int8_t *Btile, uint nn){\n"
"  const uint ib32 = seg/2u, lane = seg&1u;\n"
"  device const ushort *q2 = blk->qs + 4u*ib32;\n"
"  const uint aux32_g = uint(q2[0]) | (uint(q2[1])<<16);\n"
"  const uint aux32_s = uint(q2[2]) | (uint(q2[3])<<16);\n"
"  const float scale = float(blk->d) * (0.5f + float(aux32_s>>28)) * 0.25f;\n"
"  const uint col0 = seg*16u;\n"
"  const ulong gv0 = IQ2XXS_GRID[(aux32_g >> (8u*(2u*lane+0u))) & 255u];\n"
"  const uchar s0 = ksigns_iq2xs[(aux32_s >> (14u*lane)) & 127u];\n"
"  for(uint j=0;j<8u;j++){ float v=scale*float((gv0>>(8u*j))&255ul)*((s0&(1u<<j))?-1.0f:1.0f); Btile[(col0+j)*32u+nn]=f2i8(v,qscale);} \n"
"  const ulong gv1 = IQ2XXS_GRID[(aux32_g >> (8u*(2u*lane+1u))) & 255u];\n"
"  const uchar s1 = ksigns_iq2xs[(aux32_s >> (14u*lane+7u)) & 127u];\n"
"  for(uint j=0;j<8u;j++){ float v=scale*float((gv1>>(8u*j))&255ul)*((s1&(1u<<j))?-1.0f:1.0f); Btile[(col0+8u+j)*32u+nn]=f2i8(v,qscale);} \n"
"}\n"
"inline void dequant_q2k_seg_to_tg(device const block_q2_K *blk, uint seg, float qscale,\n"
"                                  threadgroup int8_t *Btile, uint nn){\n"
"  device const uchar *q = blk->qs + 32u*(seg/8u) + 16u*(seg&1u);\n"
"  const uchar sc = blk->scales[seg]; const uint il=(seg/2u)&3u;\n"
"  const float coef = il>1u ? (il>2u ? 1.0f/64.0f : 1.0f/16.0f) : (il>0u ? 1.0f/4.0f : 1.0f);\n"
"  const uchar mask = il>1u ? (il>2u ? 192 : 48) : (il>0u ? 12 : 3);\n"
"  const float dl = float(blk->d)*float(sc & 0x0fu)*coef; const float ml = float(blk->dmin)*float(sc>>4);\n"
"  const uint col0 = seg*16u;\n"
"  for(uint j=0;j<16u;j++){ Btile[(col0+j)*32u+nn]=f2i8(dl*float(q[j]&mask)-ml,qscale);} \n"
"}\n"
"kernel void mpp_i8_q2k_i32(device int8_t *A [[buffer(0)]], device const block_q2_K *Wq [[buffer(1)]],\n"
"                           device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],\n"
"                           constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
"                           constant float &qscale [[buffer(6)]],\n"
"                           uint2 tgid [[threadgroup_position_in_grid]],\n"
"                           uint tidx [[thread_index_in_threadgroup]]){\n"
"  threadgroup int8_t Btile[256*32];\n"
"  const uint m0=tgid.y*64u, n0=tgid.x*32u, bpr=K/256u;\n"
"  if(m0>=M || n0>=N) return; const uint rows=min(64u, M-m0);\n"
"  constexpr auto desc = matmul2d_descriptor(64,32,256,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);\n"
"  matmul2d<desc, execution_simdgroups<4>> op;\n"
"  threadgroup int8_t *bptr=Btile;\n"
"  auto mA0=tensor(A,dextents<int32_t,2>{256,64},array<int32_t,2>{1,(int32_t)K});\n"
"  auto tBt0=tensor(bptr,dextents<int32_t,2>{32,256},array<int32_t,2>{1,32});\n"
"  auto cT=op.get_destination_cooperative_tensor<decltype(mA0),decltype(tBt0),int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i){ if(cT.is_valid_element(i)) cT[i]=0; }\n"
"  for(uint kb=0; kb<bpr; ++kb){\n"
"    for(uint w=tidx; w<512u; w+=128u){ uint nn=w&31u, seg=w>>5; dequant_q2k_seg_to_tg(Wq+(n0+nn)*bpr+kb,seg,qscale,bptr,nn);} \n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tensor(A + m0*K + kb*256u, dextents<int32_t,2>{256,(int32_t)rows}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto mBt=tensor(bptr, dextents<int32_t,2>{32,256}, array<int32_t,2>{1,32});\n"
"    op.run(mA,mBt,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  auto mC=tensor(C + m0*N + n0, dextents<int32_t,2>{32,(int32_t)rows}, array<int32_t,2>{1,(int32_t)N});\n"
"  cT.store(mC);\n"
"}\n"
// ---- baseline: dequant whole weight to global transposed int8 ----
"kernel void dequant_iq2_i8(device const block_iq2_xxs *src [[buffer(0)]], device char *dst [[buffer(1)]],\n"
"                           constant uint &q_rows [[buffer(2)]], constant uint &q_cols [[buffer(3)]],\n"
"                           constant uint &total [[buffer(4)]], constant float &qscale [[buffer(5)]],\n"
"                           uint tid [[thread_position_in_grid]]){\n"
"  if(tid>=total) return;\n"
"  const uint bpr=q_cols/QK_K, spr=bpr*16u; const uint r=tid/spr; const uint seg=tid-r*spr;\n"
"  const uint b=seg/16u, il0=seg-b*16u; device const block_iq2_xxs *blk=src+r*bpr+b;\n"
"  const uint ib32=il0/2u, lane=il0&1u; device const ushort *q2=blk->qs+4u*ib32;\n"
"  const uint aux32_g=uint(q2[0])|(uint(q2[1])<<16); const uint aux32_s=uint(q2[2])|(uint(q2[3])<<16);\n"
"  const float scale=float(blk->d)*(0.5f+float(aux32_s>>28))*0.25f; const uint col0=b*QK_K+il0*16u;\n"
"  const ulong gv0=IQ2XXS_GRID[(aux32_g>>(8u*(2u*lane+0u)))&255u]; const uchar s0=ksigns_iq2xs[(aux32_s>>(14u*lane))&127u];\n"
"  for(uint j=0;j<8u;j++){float v=scale*float((gv0>>(8u*j))&255ul)*((s0&(1u<<j))?-1.0f:1.0f); dst[(col0+j)*q_rows+r]=f2i8(v,qscale);}\n"
"  const ulong gv1=IQ2XXS_GRID[(aux32_g>>(8u*(2u*lane+1u)))&255u]; const uchar s1=ksigns_iq2xs[(aux32_s>>(14u*lane+7u))&127u];\n"
"  for(uint j=0;j<8u;j++){float v=scale*float((gv1>>(8u*j))&255ul)*((s1&(1u<<j))?-1.0f:1.0f); dst[(col0+8u+j)*q_rows+r]=f2i8(v,qscale);}\n"
"}\n"
"kernel void mpp_i8_i8_i32(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device int32_t *C [[buffer(2)]],\n"
"                          constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
"                          uint2 tgid [[threadgroup_position_in_grid]]){\n"
"  constexpr auto desc = matmul2d_descriptor(64,32,static_cast<int>(dynamic_extent));\n"
"  matmul2d<desc, execution_simdgroups<4>> op;\n"
"  auto tA=tensor(A,dextents<int32_t,2>{(int32_t)K,(int32_t)M},array<int32_t,2>{1,(int32_t)K});\n"
"  auto tB=tensor(B,dextents<int32_t,2>{(int32_t)N,(int32_t)K},array<int32_t,2>{1,(int32_t)N});\n"
"  auto tC=tensor(C,dextents<int32_t,2>{(int32_t)N,(int32_t)M},array<int32_t,2>{1,(int32_t)N});\n"
"  auto mA=tA.slice(0,tgid.y*64); auto mB=tB.slice(tgid.x*32,0); auto mC=tC.slice(tgid.x*32,tgid.y*64);\n"
"  op.run(mA,mB,mC);\n"
"}\n"
// ---- fused: dequant iq2_xxs tiles into threadgroup, matmul2d accumulate ----
"kernel void mpp_i8_iq2_i32(device int8_t *A [[buffer(0)]], device const block_iq2_xxs *Wq [[buffer(1)]],\n"
"                           device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],\n"
"                           constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],\n"
"                           constant float &qscale [[buffer(6)]],\n"
"                           uint2 tgid [[threadgroup_position_in_grid]],\n"
"                           uint tidx [[thread_index_in_threadgroup]]){\n"
"  threadgroup int8_t Btile[256*32];\n"
"  const uint m0=tgid.y*64u, n0=tgid.x*32u; const uint bpr=K/256u;\n"
"  if(m0>=M || n0>=N) return; const uint rows=min(64u, M-m0);\n"
"  constexpr auto desc = matmul2d_descriptor(64,32,256,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);\n"
"  matmul2d<desc, execution_simdgroups<4>> op;\n"
"  threadgroup int8_t *bptr = Btile;\n"
"  auto mA0=tensor(A,dextents<int32_t,2>{256,64},array<int32_t,2>{1,(int32_t)K});\n"
"  auto tBt0=tensor(bptr,dextents<int32_t,2>{32,256},array<int32_t,2>{1,32});\n"
"  auto cT=op.get_destination_cooperative_tensor<decltype(mA0), decltype(tBt0), int32_t>();\n"
"  for(uint16_t i=0;i<cT.get_capacity();++i){ if(cT.is_valid_element(i)) cT[i]=0; }\n"
"  for(uint kb=0; kb<bpr; ++kb){\n"
"    for(uint w=tidx; w<512u; w+=128u){ uint nn=w&31u, seg=w>>5; device const block_iq2_xxs *blk=Wq+(n0+nn)*bpr+kb; dequant_seg_to_tg(blk,seg,qscale,bptr,nn);} \n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    auto mA=tensor(A + m0*K + kb*256u, dextents<int32_t,2>{256,(int32_t)rows}, array<int32_t,2>{1,(int32_t)K});\n"
"    auto mBt=tensor(bptr, dextents<int32_t,2>{32,256}, array<int32_t,2>{1,32});\n"
"    op.run(mA,mBt,cT);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  }\n"
"  auto mC=tensor(C + m0*N + n0, dextents<int32_t,2>{32,(int32_t)rows}, array<int32_t,2>{1,(int32_t)N});\n"
"  cT.store(mC);\n"
"}\n";

static id<MTLComputePipelineState> mkpipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *n){
    NSError *e=nil; id<MTLFunction> f=[lib newFunctionWithName:n];
    if(!f){fprintf(stderr,"missing fn %s\n",n.UTF8String);exit(2);}
    id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:f error:&e];
    if(!p){fprintf(stderr,"pipe %s failed: %s\n",n.UTF8String,e.localizedDescription.UTF8String);exit(2);} return p;
}

int main(int argc, char **argv){
    int M=128, N=2048, K=7168, iters=100, warmup=10, q2k=0;
    const char *metal_path="metal/moe.metal";
    float qscale=512.0f;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--shape")&&i+3<argc){M=atoi(argv[++i]);K=atoi(argv[++i]);N=atoi(argv[++i]);}
        else if(!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--warmup")&&i+1<argc) warmup=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--qscale")&&i+1<argc) qscale=atof(argv[++i]);
        else if(!strcmp(argv[i],"--grid")&&i+1<argc) metal_path=argv[++i];
        else if(!strcmp(argv[i],"--q2k")) q2k=1;
    }
    if(N%32){fprintf(stderr,"N must be multiple of 32\n");return 2;}
    if(K%256){fprintf(stderr,"K must be multiple of 256\n");return 2;}
    load_grid_from_metal(metal_path);

    // Build metal source with grid baked in.
    char grid_lines[256*22];
    char *gp=grid_lines;
    for(int i=0;i<256;i++) gp+=sprintf(gp,"0x%016llxul%s",(unsigned long long)h_iq2xxs_grid[i], (i%4==3)?",\n":",");
    char *src=malloc(strlen(metal_src)+sizeof(grid_lines)+16);
    { const char *ph=strstr(metal_src,"%%GRID%%"); size_t pre=ph-metal_src; memcpy(src,metal_src,pre); int off=pre; off+=sprintf(src+off,"%s",grid_lines); strcpy(src+off,ph+strlen("%%GRID%%")); }

    @autoreleasepool{
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        MTLCompileOptions *opt=[MTLCompileOptions new]; opt.languageVersion=MTLLanguageVersion4_0;
        NSError *e=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src] options:opt error:&e];
        if(!lib){fprintf(stderr,"compile failed:\n%s\n",e.localizedDescription.UTF8String);return 2;}
        id<MTLCommandQueue> q=[dev newCommandQueue];
        id<MTLComputePipelineState> p_deq=mkpipe(dev,lib,@"dequant_iq2_i8");
        id<MTLComputePipelineState> p_mm =mkpipe(dev,lib,@"mpp_i8_i8_i32");
        id<MTLComputePipelineState> p_fus=mkpipe(dev,lib,@"mpp_i8_iq2_i32");
        id<MTLComputePipelineState> p_q2k=mkpipe(dev,lib,@"mpp_i8_q2k_i32");

        if(q2k){
            const int bpr=K/256;
            id<MTLBuffer> A=[dev newBufferWithLength:(NSUInteger)M*K options:MTLResourceStorageModeShared];
            id<MTLBuffer> Wq=[dev newBufferWithLength:(NSUInteger)N*bpr*sizeof(block_q2_K) options:MTLResourceStorageModeShared];
            id<MTLBuffer> Cf=[dev newBufferWithLength:(NSUInteger)M*N*sizeof(int32_t) options:MTLResourceStorageModeShared];
            int8_t *Ap=A.contents; for(NSUInteger i=0;i<(NSUInteger)M*K;i++) Ap[i]=(int8_t)((int)((i*131u+7u)%127)-63);
            block_q2_K *Wp=Wq.contents; uint32_t rng=999u;
            for(NSUInteger i=0;i<(NSUInteger)N*bpr;i++){ rng=rng*1664525u+1013904223u; Wp[i].d=(fp16_t)(0.02f+0.0001f*(rng%53)); Wp[i].dmin=(fp16_t)(0.01f+0.00005f*(rng%29));
                for(int j=0;j<QK_K/16;j++){rng=rng*1664525u+1013904223u; Wp[i].scales[j]=(uint8_t)(rng&0xff);}
                for(int j=0;j<QK_K/4;j++){rng=rng*1664525u+1013904223u; Wp[i].qs[j]=(uint8_t)(rng&0xff);} }
            int8_t *Wi8=malloc((size_t)N*K), blk256[256];
            for(int n=0;n<N;n++) for(int kb=0;kb<bpr;kb++){ cpu_dequant_q2k_block(&Wp[n*bpr+kb],qscale,blk256); for(int j=0;j<256;j++) Wi8[(size_t)n*K+kb*256+j]=blk256[j]; }
            int32_t *Cref=calloc((size_t)M*N,sizeof(int32_t));
            for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)Ap[(size_t)m*K+k]*(long)Wi8[(size_t)n*K+k]; Cref[(size_t)m*N+n]=(int32_t)s; }
            uint32_t uM=M,uN=N,uK=K; MTLSize grid=MTLSizeMake((uN+31)/32,(uM+63)/64,1); MTLSize tg=MTLSizeMake((NSUInteger)p_q2k.threadExecutionWidth*4,1,1);
            id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:p_q2k]; [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Wq offset:0 atIndex:1]; [enc setBuffer:Cf offset:0 atIndex:2];
            [enc setBytes:&uM length:4 atIndex:3]; [enc setBytes:&uN length:4 atIndex:4]; [enc setBytes:&uK length:4 atIndex:5]; [enc setBytes:&qscale length:4 atIndex:6];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if(cb.error){fprintf(stderr,"q2k cb error %s\n",cb.error.localizedDescription.UTF8String);return 2;}
            int32_t *Cp=Cf.contents; long maxabs=0,mism=0; for(NSUInteger i=0;i<(NSUInteger)M*N;i++){long d=labs((long)Cp[i]-(long)Cref[i]); if(d>maxabs)maxabs=d; if(d)mism++;}
            printf("verify_q2k M=%d N=%d K=%d max_abs_int=%ld mismatches=%ld/%lld %s\n",M,N,K,maxabs,mism,(long long)M*N,maxabs==0?"OK":"FAIL");
            free(Wi8); free(Cref); return maxabs==0?0:1;
        }

        const int bpr=K/256;
        NSUInteger Abytes=(NSUInteger)M*K, Wbytes=(NSUInteger)N*bpr*sizeof(block_iq2_xxs);
        NSUInteger Bbytes=(NSUInteger)K*N, Cbytes=(NSUInteger)M*N*sizeof(int32_t);
        id<MTLBuffer> A=[dev newBufferWithLength:Abytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Wq=[dev newBufferWithLength:Wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Bi8=[dev newBufferWithLength:Bbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Cbase=[dev newBufferWithLength:Cbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Cfus=[dev newBufferWithLength:Cbytes options:MTLResourceStorageModeShared];

        // random A int8 and random iq2_xxs weights
        int8_t *Ap=A.contents; for(NSUInteger i=0;i<Abytes;i++) Ap[i]=(int8_t)((int)((i*131u+7u)%127)-63);
        block_iq2_xxs *Wp=Wq.contents; uint32_t rng=12345u;
        for(NSUInteger i=0;i<(NSUInteger)N*bpr;i++){ rng=rng*1664525u+1013904223u; Wp[i].d=(fp16_t)(0.01f+0.0001f*(rng%97)); for(int j=0;j<QK_K/8;j++){rng=rng*1664525u+1013904223u; Wp[i].qs[j]=(uint16_t)(rng&0xffff);} }

        // CPU reference: dequant weights, int32 matmul
        int8_t *Wi8=malloc((size_t)N*K); int8_t blk256[256];
        for(int n=0;n<N;n++) for(int kb=0;kb<bpr;kb++){ cpu_dequant_block(&Wp[n*bpr+kb],qscale,blk256); for(int j=0;j<256;j++) Wi8[(size_t)n*K + kb*256 + j]=blk256[j]; }
        int32_t *Cref=calloc((size_t)M*N,sizeof(int32_t));
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(long)Ap[(size_t)m*K+k]*(long)Wi8[(size_t)n*K+k]; Cref[(size_t)m*N+n]=(int32_t)s; }

        uint32_t uM=M,uN=N,uK=K;
        MTLSize grid=MTLSizeMake((uN+31)/32,(uM+63)/64,1);
        MTLSize tg=MTLSizeMake((NSUInteger)p_mm.threadExecutionWidth*4,1,1);

        // ---- run fused once for correctness ----
        { id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
          [enc setComputePipelineState:p_fus]; [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Wq offset:0 atIndex:1];
          [enc setBuffer:Cfus offset:0 atIndex:2]; [enc setBytes:&uM length:4 atIndex:3]; [enc setBytes:&uN length:4 atIndex:4];
          [enc setBytes:&uK length:4 atIndex:5]; [enc setBytes:&qscale length:4 atIndex:6];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
          if(cb.error){fprintf(stderr,"fused cb error: %s\n",cb.error.localizedDescription.UTF8String);return 2;} }
        // verify
        { int32_t *Cf=Cfus.contents; long maxabs=0; long mism=0; for(NSUInteger i=0;i<(NSUInteger)M*N;i++){ long d=labs((long)Cf[i]-(long)Cref[i]); if(d>maxabs)maxabs=d; if(d!=0)mism++; }
          printf("verify_fused M=%d N=%d K=%d qscale=%.1f max_abs_int=%ld mismatches=%ld/%lld %s\n",M,N,K,qscale,maxabs,mism,(long long)M*N, maxabs==0?"OK":"FAIL"); }

        // ---- timing helper ----
        #define TIME_BLOCK(label, body) do{ for(int ph=0;ph<2;ph++){ int it=ph?iters:warmup; double t0=now_s(); id<MTLCommandBuffer> cb=[q commandBuffer]; for(int i=0;i<it;i++){ body } [cb commit]; [cb waitUntilCompleted]; if(cb.error){fprintf(stderr,"%s err %s\n",label,cb.error.localizedDescription.UTF8String);return 2;} if(ph){ double ms=(now_s()-t0)*1000.0/it; double gf=2.0*M*N*K/(ms*1e6); printf("%s,%d,%d,%d,%.4f ms,%.1f GF/s\n",label,M,N,K,ms,gf);} } }while(0)

        // baseline: dequant gate weight then matmul (per call, like the real per-expert path)
        uint32_t deq_total=(uint32_t)N*bpr*16u; // q_rows=N, segments
        MTLSize deqgrid=MTLSizeMake((deq_total+255)/256,1,1), deqtg=MTLSizeMake(256,1,1);
        TIME_BLOCK("baseline_dequant+matmul", {
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:p_deq]; [enc setBuffer:Wq offset:0 atIndex:0]; [enc setBuffer:Bi8 offset:0 atIndex:1];
            [enc setBytes:&uN length:4 atIndex:2]; [enc setBytes:&uK length:4 atIndex:3]; [enc setBytes:&deq_total length:4 atIndex:4]; [enc setBytes:&qscale length:4 atIndex:5];
            [enc dispatchThreadgroups:deqgrid threadsPerThreadgroup:deqtg];
            [enc setComputePipelineState:p_mm]; [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Bi8 offset:0 atIndex:1]; [enc setBuffer:Cbase offset:0 atIndex:2];
            [enc setBytes:&uM length:4 atIndex:3]; [enc setBytes:&uN length:4 atIndex:4]; [enc setBytes:&uK length:4 atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [enc endEncoding];
        });
        TIME_BLOCK("fused_dequant_matmul", {
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            [enc setComputePipelineState:p_fus]; [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:Wq offset:0 atIndex:1]; [enc setBuffer:Cfus offset:0 atIndex:2];
            [enc setBytes:&uM length:4 atIndex:3]; [enc setBytes:&uN length:4 atIndex:4]; [enc setBytes:&uK length:4 atIndex:5]; [enc setBytes:&qscale length:4 atIndex:6];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg]; [enc endEncoding];
        });
        free(Wi8); free(Cref);
    }
    return 0;
}
