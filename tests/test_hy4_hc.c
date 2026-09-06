/* Independent-HC GPU checks against scalar F32 graph math. */
#include "../ds4_gpu.h"
#include "../hy4/hy4_math.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static unsigned checks,cases;
static float worst;
static uint32_t seed=0x48434948;
static float random_float(void) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5;return ((int)(seed%2001)-1000)*0.001f; }
typedef struct { ds4_gpu_tensor *owner,*t; float *data; size_t n; } tensor;
static tensor alloc(size_t n) {
    tensor a={.owner=ds4_gpu_tensor_alloc((n+8)*4),.n=n};assert(a.owner);
    a.t=ds4_gpu_tensor_view(a.owner,16,n*4);assert(a.t);
    a.data=ds4_gpu_tensor_contents(a.t);
    float *p=ds4_gpu_tensor_contents(a.owner);
    for(size_t j=0;j<n+8;j++) p[j]=-98765.25f;
    return a;
}
static void release(tensor *a) {
    const float *p=ds4_gpu_tensor_contents(a->owner);
    for(unsigned j=0;j<4;j++) {assert(p[j]==-98765.25f && p[a->n+4+j]==-98765.25f); checks+=2;}
    ds4_gpu_tensor_free(a->t);ds4_gpu_tensor_free(a->owner);
}
static void close_enough(const float *actual,const float *ref,size_t n) {
    for(size_t j=0;j<n;j++) {
        float scaled=fabsf(actual[j]-ref[j])/fmaxf(1,fabsf(ref[j]));
        if(!isfinite(actual[j]) || scaled>4e-5f) {
            fprintf(stderr,"FAIL HC j=%zu got=%.9g ref=%.9g scaled=%g\n",j,actual[j],ref[j],scaled);exit(1);
        }
        if(scaled>worst) worst=scaled;
        checks++;
    }
}
static void run(unsigned n,int head,int batch) {
    tensor streams=alloc(4*n),fn=alloc((head?4:8)*4*n),scale=alloc(head?1:2),base=alloc(head?4:8);
    tensor mix=alloc(head?4:8),out=alloc(n),post=alloc(4);
    for(unsigned j=0;j<4*n;j++) streams.data[j]=random_float()*3;
    for(size_t j=0;j<fn.n;j++) fn.data[j]=random_float()/sqrtf(4*n);
    for(size_t j=0;j<base.n;j++) base.data[j]=random_float()*6;
    scale.data[0]=3; if(!head) scale.data[1]=-2;
    float *ref=malloc(n*4),rp[4]; assert(ref);
    assert(head ? hy4_hc_head(ref,streams.data,fn.data,scale.data,base.data,n,4,1e-5f,1e-6f) :
        hy4_hc_pre(ref,rp,streams.data,fn.data,scale.data,base.data,n,4,1e-5f,1e-6f,2));
    if(batch) assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_hy4_hc_pre_tensor(out.t,post.t,mix.t,streams.t,fn.t,scale.t,base.t,n,head));
    if(batch) assert(ds4_gpu_end_commands());
    assert(ds4_gpu_synchronize());
    close_enough(out.data,ref,n); if(!head) close_enough(post.data,rp,4);
    // Use identical gate values to test the post broadcast's rounding exactly.
    const float w[]={0x1.000002p0f,0.375f,-0.25f,0.5f};
    memcpy(post.data,w,16);out.data[0]=0x1.000002p0f;streams.data[0]=-0x1.000004p0f;
    assert(fmaf(out.data[0],post.data[0],streams.data[0])!=0);
    float *res=malloc(4*n*4);assert(res);
    assert(hy4_hc_post(res,out.data,streams.data,post.data,n,4));
    if(batch) assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_hy4_hc_post_tensor(streams.t,out.t,post.t,n));
    if(batch) assert(ds4_gpu_end_commands());
    assert(ds4_gpu_synchronize());
    for(unsigned j=0;j<4*n;j++) {assert(memcmp(res+j,streams.data+j,4)==0);checks++;}
    assert(streams.data[0]==0);
    assert(!ds4_gpu_hy4_hc_pre_tensor(out.t,post.t,mix.t,streams.t,fn.t,scale.t,base.t,0,head));
    assert(!ds4_gpu_hy4_hc_pre_tensor(out.t,post.t,mix.t,streams.t,fn.t,scale.t,base.t,n,2));
    assert(!ds4_gpu_hy4_hc_pre_tensor(out.t,post.t,mix.t,streams.t,fn.t,scale.t,base.t,n+1,head));
    assert(!ds4_gpu_hy4_hc_pre_tensor(streams.t,post.t,mix.t,streams.t,fn.t,scale.t,base.t,n,head));
    assert(!ds4_gpu_hy4_hc_pre_tensor(out.t,mix.t,mix.t,streams.t,fn.t,scale.t,base.t,n,head));
    assert(!ds4_gpu_hy4_hc_post_tensor(streams.t,streams.t,post.t,n));
    assert(!ds4_gpu_hy4_hc_post_tensor(streams.t,out.t,post.t,n+1));
    free(res);free(ref);
    tensor *all[]={&streams,&fn,&scale,&base,&mix,&out,&post};
    for(unsigned i=0;i<7;i++) release(all[i]);cases++;
}
int main(void) {
    assert(ds4_gpu_init());
    const unsigned ns[]={1,31,128,129,6144};
    for(unsigned i=0;i<5;i++) for(int head=0;head<2;head++) for(int batch=0;batch<2;batch++) run(ns[i],head,batch);
    printf("PASS HY4 iHC: %u cases %u checks; pre/head maxscaled=%g, post bit exact incl FMA witness\n",cases,checks,worst);
    ds4_gpu_cleanup();return 0;
}
