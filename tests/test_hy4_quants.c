/* Deterministic HY4 ABI tests; --reference loads the independent source GGML
 * CPU dequantizers. --metal adds tiny resident matvec/view/batch checks.
 * No model loading, inference, expert-bank allocation, or sidecar warming. */
#include "../hy4/hy4_quants.h"
#ifndef HY4_TEST_CPU_ONLY
#include "../ds4_gpu.h"
#endif
#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
/* Standalone GPU tests do not link the model/session engine. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

typedef void (*dequant_fn)(const void *, float *, int64_t);
static uint32_t state = 0x43a18275;
static uint32_t random_u32(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static const char *quant_name(uint32_t type) {
    return type == 43 ? "stq1_0" : type == 16 ? "iq2_xxs" : type == 18 ? "iq3_xxs" : "iq4_xs";
}
static void check_case(uint32_t type, size_t n, size_t rows, int padded, int metal, void *ref, const char *sample, uint64_t sample_offset) {
    const size_t packed = hy4_quant_row_bytes(type, n), stride = packed + (padded ? 16 : 0);
    uint8_t *raw = malloc(rows * stride + 1), *w = raw + 1;
    float *x = malloc(n * sizeof(float)), *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float));
    float *cpu = malloc(rows * sizeof(float));
    assert(raw && x && a && b && cpu);
    for (size_t i = 0; i < rows * stride; ++i) w[i] = random_u32() & 255;
    for (size_t i = 0; i < n; ++i) x[i] = ((int)(random_u32() % 2001) - 1000) / 1000.f;
    const size_t block_bytes = hy4_quant_row_bytes(type, 256);
    for (size_t row = 0; row < rows; ++row) for (size_t k = 0; k < n/256; ++k) {
        const uint16_t scales[] = {0, 0x8000, 1, 0x8001, 0x2000, 0xb000, 0x3401};
        const uint16_t h = scales[(row + k) % (sizeof(scales)/sizeof(scales[0]))];
        const size_t off = row*stride + k*block_bytes + (type == 43 ? 40 : 0);
        w[off] = h & 255; w[off+1] = h >> 8;
    }
    if (sample) {
        FILE *fp = fopen(sample, "rb"); assert(fp);
        assert(fseeko(fp, (off_t)sample_offset, SEEK_SET) == 0);
        for (size_t row = 0; row < rows; ++row) assert(fread(w + row*stride, 1, packed, fp) == packed);
        assert(fclose(fp) == 0);
    }
    dequant_fn dequant = NULL;
    if (ref) {
        char name[96]; snprintf(name, sizeof(name), "dequantize_row_%s", quant_name(type));
        dequant = (dequant_fn)dlsym(ref, name); assert(dequant);
    }
    for (size_t row = 0; row < rows; ++row) {
        assert(hy4_dequantize_row(type, w + row*stride, a, n));
        if (dequant) {
            /* Reference structs require alignment; our byte reader is also checked at +1. */
            void *aligned = malloc(packed); assert(aligned);
            memcpy(aligned, w + row*stride, packed); dequant(aligned, b, (int64_t)n); free(aligned);
            for (size_t j = 0; j < n; ++j) if (a[j] != b[j]) {
                fprintf(stderr, "dequant mismatch type=%u row=%zu j=%zu ours=%.9g source=%.9g\n", type,row,j,a[j],b[j]); abort();
            }
        }
    }
    assert(hy4_quant_matvec_cpu(type,w,x,cpu,n,rows,stride));
#ifndef HY4_TEST_CPU_ONLY
    if (metal) {
        // A nonzero aligned tensor view offset and padded row stride detect rebasing bugs.
        ds4_gpu_tensor *wb = ds4_gpu_tensor_alloc(rows*stride + 32);
        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wb, 16, rows*stride);
        ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(n*sizeof(float));
        ds4_gpu_tensor *ob = ds4_gpu_tensor_alloc((rows+8)*sizeof(float));
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(ob,16,rows*sizeof(float));
        assert(wb && wv && xt && ob && ov);
        assert(ds4_gpu_tensor_write(wv,0,w,rows*stride)); assert(ds4_gpu_tensor_write(xt,0,x,n*sizeof(float)));
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_hy4_quant_matvec_tensor(ov,wv,xt,type,(uint32_t)n,(uint32_t)rows,stride));
        assert(ds4_gpu_end_commands()); assert(ds4_gpu_synchronize());
        float *gpu = malloc(rows*sizeof(float)); assert(gpu);
        assert(ds4_gpu_tensor_read(ov,0,gpu,rows*sizeof(float)));
        for (size_t row=0; row<rows; ++row) {
            const float tol = 5e-5f * fmaxf(1.f,fabsf(cpu[row]));
            if (!isfinite(gpu[row]) || fabsf(gpu[row]-cpu[row])>tol) {
                fprintf(stderr,"matvec mismatch type=%u n=%zu row=%zu cpu=%.9g gpu=%.9g tol=%.9g\n",type,n,row,cpu[row],gpu[row],tol); abort();
            }
        }
        assert(!ds4_gpu_hy4_quant_matvec_tensor(ov,wv,xt,42,(uint32_t)n,(uint32_t)rows,stride));
        assert(!ds4_gpu_hy4_quant_matvec_tensor(ov,wv,xt,type,(uint32_t)n-1,(uint32_t)rows,stride));
        assert(!ds4_gpu_hy4_quant_matvec_tensor(ov,wv,xt,type,(uint32_t)n,(uint32_t)rows,stride-1));
        free(gpu); ds4_gpu_tensor_free(ov); ds4_gpu_tensor_free(ob); ds4_gpu_tensor_free(xt);
        ds4_gpu_tensor_free(wv); ds4_gpu_tensor_free(wb);
    }
#else
    (void)metal;
#endif
    free(cpu);free(b);free(a);free(x);free(raw);
}
static void test_stq_codebook(void) {
    uint8_t block[42] = {0}; float out[256]; block[41] = 0x3c; // d=1
    for (unsigned g=0;g<64;++g) {
        const unsigned code = g%16, sign = (g/16)%2;
        block[g/2] |= code << (4*(g&1)); block[32+g/8] |= sign << (g&7);
    }
    assert(hy4_dequantize_row(43,block,out,256));
    for(unsigned g=0;g<64;++g) {
        int zeros=0;
        for(unsigned p=0;p<4;++p) {
            float v=out[(g/16)*64+g%16+p*16]; assert(v==-1 || v==0 || v==1); zeros+=v==0;
            // Sign halves are exact opposite vectors, including the one zero.
            if(g<16) assert(v == -out[64+g+p*16]);
        }
        assert(zeros==1);
    }
}
int main(int argc,char **argv) {
    int metal=0; void *ref=NULL; const char *sample=NULL; uint32_t sample_type=0; size_t sample_width=0; uint64_t sample_offset=0;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--metal")) metal=1;
        else if(!strcmp(argv[i],"--reference") && i+1<argc) {
            ref=dlopen(argv[++i],RTLD_NOW|RTLD_LOCAL); if(!ref) {fprintf(stderr,"%s\n",dlerror());return 1;}
        } else if(!strcmp(argv[i],"--sample") && i+4<argc) {
            sample_type=(uint32_t)strtoul(argv[++i],NULL,10); sample_width=(size_t)strtoull(argv[++i],NULL,10);
            sample=argv[++i]; sample_offset=strtoull(argv[++i],NULL,10);
            if(!hy4_quant_row_bytes(sample_type,sample_width) || sample_width>6144) return 1;
        } else {fprintf(stderr,"usage: %s [--metal] [--reference libggml-base.dylib] [--sample TYPE WIDTH FILE OFFSET]\n",argv[0]);return 1;}
    }
    test_stq_codebook();
    assert(hy4_quant_row_bytes(43,256)==42); assert(!hy4_quant_row_bytes(43,255));
    assert(!hy4_quant_row_bytes(42,256)); assert(!hy4_quant_row_bytes(43,0));
#ifndef HY4_TEST_CPU_ONLY
    if(metal) assert(ds4_gpu_init());
#else
    assert(!metal);
#endif
    const uint32_t types[]={43,16,18,23}; const size_t widths[]={256,512,2048,6144};
    unsigned cases=0;
    if(sample) {
        check_case(sample_type,sample_width,3,0,metal,ref,sample,sample_offset); ++cases;
        printf("PASS: actual sidecar sample type=%u width=%zu rows=3 offset=%llu file=%s\n", sample_type,sample_width,(unsigned long long)sample_offset,sample);
    } else for(unsigned t=0;t<4;++t) for(unsigned n=0;n<4;++n) for(int pad=0;pad<2;++pad) {
        check_case(types[t],widths[n],17,pad,metal,ref,NULL,0);++cases;
    }
#ifndef HY4_TEST_CPU_ONLY
    if(metal) ds4_gpu_cleanup();
#endif
    if(ref) dlclose(ref);
    printf("PASS: HY4 four-type dequant/matvec %u cases, source-reference=%s Metal=%s; max weights < 64 KiB\n",cases,ref?"yes":"no",metal?"yes":"no");
    return 0;
}
