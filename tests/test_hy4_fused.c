/* Native HY4 fused top-8 regression. All quant triplets, padded interleaved
 * slot banks, permuted/repeated IDs, clamp witnesses, real model widths,
 * queued dependencies and invalid bounds/aliases. No model is loaded. */
#include "../ds4_gpu.h"
#include "../hy4/hy4_quants.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static uint32_t state=0x48594646;
static uint32_t rng(void) { state^=state<<13; state^=state>>17; state^=state<<5; return state; }
static unsigned checks, cases, clamp_gate, clamp_up;
static double max_error, max_normalized_error;
static void compare(const char *stage,const float *a,const float *b,size_t n, double budget) {
    // Cancellation can make an individual output arbitrarily close to zero.
    // Bound infinity-norm error against the vector scale, plus a fixed floor.
    // The independent model test separately requires identical tokens/top IDs.
    double scale=1;
    for(size_t j=0;j<n;j++) scale=fmax(scale,fabs(b[j]));
    for(size_t j=0;j<n;j++) {
        double err=fabs((double)a[j]-b[j]), tol=budget*scale;
        if(err/scale>max_normalized_error) max_normalized_error=err/scale;
        if(!isfinite(a[j]) || !isfinite(b[j]) || err>tol) {
            fprintf(stderr,"FAIL %s j=%zu fused=%.9g ref=%.9g error=%g tol=%g\n",stage,j,a[j],b[j],err,tol); exit(1);
        }
        if(err>max_error) max_error=err;
        checks++;
    }
}
static void run(uint32_t gt,uint32_t dt,uint32_t n,uint32_t ff,int duplicate) {
    const uint32_t types[]={gt,gt,dt};
    uint64_t rb[3],off[3],stride=0;
    for(unsigned f=0;f<3;f++) {
        rb[f]=hy4_quant_row_bytes(types[f],f==2 ? ff:n)+16;
        off[f]=stride; stride+=(rb[f]*(f==2 ? n:ff)+63)&~UINT64_C(63);
    }
    const size_t total=11*stride;
    unsigned char *raw=malloc(total); assert(raw);
    for(size_t i=0;i<total;i++) raw[i]=(unsigned char)rng();
    for(unsigned slot=0;slot<11;slot++) for(unsigned f=0;f<3;f++) {
        unsigned rows=f==2?n:ff, width=f==2?ff:n;
        size_t block=hy4_quant_row_bytes(types[f],256);
        for(unsigned r=0;r<rows;r++) for(unsigned b=0;b<width/256;b++) {
            size_t at=slot*stride+off[f]+r*rb[f]+b*block+(types[f]==43?40:0);
            uint16_t scale=f==2 ? 0x1000 : gt==43 ? 0x3400:0x2000;
            raw[at]=scale&255; raw[at+1]=scale>>8;
        }
    }
    ds4_gpu_tensor *slab=ds4_gpu_tensor_alloc(total+32), *bank[3];
    assert(slab && ds4_gpu_tensor_write(slab,16,raw,total));
    for(unsigned f=0;f<3;f++) { bank[f]=ds4_gpu_tensor_view(slab,16+off[f],total-off[f]); assert(bank[f]); }
    ds4_gpu_tensor *xb=ds4_gpu_tensor_alloc(n*4+32), *x=ds4_gpu_tensor_view(xb,16,n*4);
    ds4_gpu_tensor *h=ds4_gpu_tensor_alloc(ff*8*4),*out=ds4_gpu_tensor_alloc(n*4);
    ds4_gpu_tensor *rh=ds4_gpu_tensor_alloc(ff*8*4),*rd=ds4_gpu_tensor_alloc(n*8*4),*ro=ds4_gpu_tensor_alloc(n*4);
    ds4_gpu_tensor *g=ds4_gpu_tensor_alloc(ff*4),*u=ds4_gpu_tensor_alloc(ff*4),*weights=ds4_gpu_tensor_alloc(32);
    assert(xb&&x&&h&&out&&rh&&rd&&ro&&g&&u&&weights);
    float *input=malloc(n*4); assert(input);
    for(unsigned j=0;j<n;j++) input[j]=((int)(rng()%2001)-1000)*0.006f;
    float ws[8]={0.45f,0.23f,0.57f,0.15f,0.41f,0.67f,0.22f,0.127f};
    int32_t slots[8]={9,2,7,0,10,4,8,1}; if(duplicate) slots[7]=slots[0];
    uint64_t strides[3]={stride,stride,stride};
    assert(ds4_gpu_tensor_write(x,0,input,n*4) && ds4_gpu_tensor_write(weights,0,ws,32));
    assert(ds4_gpu_begin_commands());
    // Reference consists of the existing separate Metal operators, including
    // clamp10 and separately rounded post-down multiplication/reduction.
    for(unsigned k=0;k<8;k++) {
        ds4_gpu_tensor *wv[3];
        for(unsigned f=0;f<3;f++) { wv[f]=ds4_gpu_tensor_view(bank[f],slots[k]*stride,rb[f]*(f==2?n:ff)); assert(wv[f]); }
        ds4_gpu_tensor *hv=ds4_gpu_tensor_view(rh,k*ff*4,ff*4),*dv=ds4_gpu_tensor_view(rd,k*n*4,n*4);
        assert(hv&&dv);
        assert(ds4_gpu_hy4_quant_matvec_tensor(g,wv[0],x,gt,n,ff,rb[0]));
        assert(ds4_gpu_hy4_quant_matvec_tensor(u,wv[1],x,gt,n,ff,rb[1]));
        assert(ds4_gpu_swiglu_tensor(hv,g,u,ff,10,1));
        assert(ds4_gpu_hy4_quant_matvec_tensor(dv,wv[2],hv,dt,ff,n,rb[2]));
        for(unsigned f=0;f<3;f++) ds4_gpu_tensor_free(wv[f]);
        ds4_gpu_tensor_free(hv);ds4_gpu_tensor_free(dv);
    }
    assert(ds4_gpu_hy4_weighted_sum8_tensor(ro,rd,weights,n));
    assert(ds4_gpu_hy4_fused_ffn_tensor(out,h,x,bank[0],bank[1],bank[2],weights,slots,11,gt,dt,n,ff,rb,strides));
    assert(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    // Two independently rounded GPU dot layouts get twice the CPU-oracle
    // tolerance; the fused result must separately pass the tighter CPU gate.
    compare("gate/up/clamp",ds4_gpu_tensor_contents(h),ds4_gpu_tensor_contents(rh),ff*8,1e-4);
    compare("down/weighted",ds4_gpu_tensor_contents(out),ds4_gpu_tensor_contents(ro),n,1e-4);
    { // Independent double-accumulating CPU oracle, including real widths.
        float *cg=malloc(ff*4),*cu=malloc(ff*4),*ch=malloc(ff*8*4),*cd=malloc(n*4),*co=calloc(n,4);
        assert(cg&&cu&&ch&&cd&&co);
        for(unsigned k=0;k<8;k++) {
            assert(hy4_quant_matvec_cpu(gt,raw+slots[k]*stride+off[0],input,cg,n,ff,rb[0]));
            assert(hy4_quant_matvec_cpu(gt,raw+slots[k]*stride+off[1],input,cu,n,ff,rb[1]));
            for(unsigned j=0;j<ff;j++) {
                clamp_gate+=cg[j]>10;clamp_up+=fabsf(cu[j])>10;
                float a=fminf(cg[j],10),b=fminf(10,fmaxf(-10,cu[j]));
                ch[k*ff+j]=a/(1+expf(-a))*b;
            }
            assert(hy4_quant_matvec_cpu(dt,raw+slots[k]*stride+off[2],ch+k*ff,cd,ff,n,rb[2]));
            for(unsigned j=0;j<n;j++) { volatile float product=ws[k]*cd[j];co[j]=k?co[j]+product:product; }
        }
        compare("CPU clamped h",ds4_gpu_tensor_contents(h),ch,ff*8,5e-5);
        compare("CPU weighted out",ds4_gpu_tensor_contents(out),co,n,5e-5);
        free(cg);free(cu);free(ch);free(cd);free(co);
    }
    // Every rejection occurs before encoding; malformed slots cannot read a
    // bank and out/h cannot overwrite pending input or each other.
#define FUSED(O,H,S,C,G,D,N,F,R,T) ds4_gpu_hy4_fused_ffn_tensor(O,H,x,bank[0],bank[1],bank[2],weights,S,C,G,D,N,F,R,T)
    assert(!FUSED(out,h,slots,10,gt,dt,n,ff,rb,strides));
    int32_t bad[8];memcpy(bad,slots,32);bad[0]=-1;
    assert(!FUSED(out,h,bad,11,gt,dt,n,ff,rb,strides));
    assert(!FUSED(out,h,slots,11,42,dt,n,ff,rb,strides));
    assert(!FUSED(out,h,slots,11,gt,43,n,ff,rb,strides));
    assert(!FUSED(out,h,slots,11,gt,dt,n-1,ff,rb,strides));
    assert(!FUSED(out,h,slots,11,gt,dt,n,ff,NULL,strides));
    assert(!FUSED(out,h,slots,11,gt,dt,n,ff,rb,NULL));
    assert(!FUSED(x,h,slots,11,gt,dt,n,ff,rb,strides));
    assert(!FUSED(out,out,slots,11,gt,dt,n,ff,rb,strides));
    uint64_t huge[3]={UINT64_MAX-15,stride,stride};
    assert(!FUSED(out,h,slots,11,gt,dt,n,ff,rb,huge));
#undef FUSED
    printf("PASS fused top8 gate=%u down=%u n=%u ff=%u slots=11 repeated=%d\n",gt,dt,n,ff,duplicate);
    ds4_gpu_tensor *all[]={slab,bank[0],bank[1],bank[2],xb,x,h,out,rh,rd,ro,g,u,weights};
    for(unsigned i=0;i<sizeof(all)/sizeof(*all);i++) ds4_gpu_tensor_free(all[i]);
    free(input);free(raw);cases++;
}
int main(void) {
    assert(ds4_gpu_init());
    for(unsigned g=0;g<2;g++) for(unsigned d=0;d<2;d++) {
        run(g?16:43,d?23:18,256,256,g==d);
        run(g?16:43,d?23:18,6144,2048,0);
    }
    assert(clamp_gate>100 && clamp_up>100);
    printf("PASS HY4 fused FFN %u cases, %u comparisons, max_abs=%g max_normalized=%g, clamp_gate=%u clamp_up=%u\n",cases,checks,max_error,max_normalized_error,clamp_gate,clamp_up);
    ds4_gpu_cleanup(); return 0;
}
