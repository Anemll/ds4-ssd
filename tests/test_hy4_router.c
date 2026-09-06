/* Native HY4 fused router versus the existing Metal graph. */
#include "../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)) {fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
typedef struct {ds4_gpu_tensor *base,*view;float *data;size_t n;} tensor;
static tensor make(size_t n) {
    tensor t={.n=n};t.base=ds4_gpu_tensor_alloc((n+8)*4);CHECK(t.base);
    float *p=ds4_gpu_tensor_contents(t.base);CHECK(p);
    for(size_t i=0;i<n+8;i++)p[i]=-98765.25f;
    t.view=ds4_gpu_tensor_view(t.base,16,n*4);CHECK(t.view);
    t.data=ds4_gpu_tensor_contents(t.view);CHECK(t.data==p+4);return t;
}
static void release(tensor t) {
    float *p=ds4_gpu_tensor_contents(t.base);
    for(unsigned i=0;i<4;i++)CHECK(p[i]==-98765.25f && p[t.n+4+i]==-98765.25f);
    ds4_gpu_tensor_free(t.view);ds4_gpu_tensor_free(t.base);
}
static uint32_t state=0x48595254;
static float random_float(void) {state^=state<<13;state^=state>>17;state^=state<<5;return ((int)(state%8193)-4096)/512.f;}
static int route(tensor selected,tensor weights,tensor probs,tensor logits,void *map,size_t size,int bias) {
    return ds4_gpu_glm_router_select_tensor(selected.view,weights.view,probs.view,map,size,64,256,8,2.827f,bias,logits.view,1);
}
int main(void) {
    CHECK(ds4_gpu_init());
    const size_t size=(size_t)sysconf(_SC_PAGESIZE);
    void *map=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);CHECK(map!=MAP_FAILED);
    float *bias=(float *)((char *)map+64);
    tensor x=make(256),p0=make(256),p1=make(256),i0=make(8),i1=make(8),w0=make(8),w1=make(8);
    for(unsigned c=0;c<128;c++) {
        const int use_bias=c%2;
        for(unsigned i=0;i<256;i++) {
            float v=random_float();
            switch(c/2) {
              case 0:v=0;break;
              case 1:v=-100;break;
              case 2:v=100;break;
              case 3:v=(int)(i%8)-4;break;
              case 4:v=i%2 ? INFINITY : -INFINITY;break;
              case 5:v=-16.f+(i%8)*.001f;break;
              case 6:v=-12.f+(i%8)*.01f;break;
            }
            x.data[i]=v;bias[i]=c<8 ? 0.f : random_float()*.05f;
        }
        CHECK(ds4_gpu_tensor_did_modify(x.base,0,(x.n+8)*4));
        // Both paths run in the same batch; no CPU fence between producer and consumers.
        CHECK(ds4_gpu_begin_commands());setenv("DS4_HY4_FUSED_ROUTER","0",1);
        CHECK(route(i0,w0,p0,x,map,size,use_bias));setenv("DS4_HY4_FUSED_ROUTER","1",1);
        CHECK(route(i1,w1,p1,x,map,size,use_bias));CHECK(ds4_gpu_end_commands());
        for(unsigned i=0;i<256;i++) {
            if(memcmp(p0.data+i,p1.data+i,4))fprintf(stderr,"prob c=%u i=%u %.9g %.9g\n",c,i,p0.data[i],p1.data[i]);
            CHECK(!memcmp(p0.data+i,p1.data+i,4));
        }
        for(unsigned i=0;i<8;i++) {
            if(memcmp(i0.data+i,i1.data+i,4))fprintf(stderr,"ID c=%u i=%u %d %d\n",c,i,((int32_t*)i0.data)[i],((int32_t*)i1.data)[i]);
            CHECK(!memcmp(i0.data+i,i1.data+i,4));
            if(memcmp(w0.data+i,w1.data+i,4))fprintf(stderr,"weight c=%u i=%u %.9g %.9g\n",c,i,w0.data[i],w1.data[i]);
            CHECK(!memcmp(w0.data+i,w1.data+i,4));
        }
    }
    for(unsigned mode=0;mode<3;mode++) {
        setenv("DS4_HY4_FUSED_ROUTER",mode==1 ? "1":"0",1);
        CHECK(route(i1,w1,p1,x,map,size,1));CHECK(ds4_gpu_synchronize());
        const double t=ds4_gpu_busy_seconds();
        CHECK(ds4_gpu_begin_commands());
        for(unsigned k=0;k<256;k++)CHECK(route(i1,w1,p1,x,map,size,1));
        CHECK(ds4_gpu_end_commands());
        printf("HY4 router %s GPU_us=%.3f warm256\n",mode==1?"fused":"original",(ds4_gpu_busy_seconds()-t)*1e6/256);
    }
    setenv("DS4_HY4_FUSED_ROUTER","1",1);
    CHECK(!route(i1,w1,p1,x,map,64,1));
    CHECK(!route(i1,i1,p1,x,map,size,0));
    CHECK(!route(i1,w1,x,x,map,size,0));
    tensor short_x=x;short_x.view=ds4_gpu_tensor_view(x.base,16,255*4);CHECK(short_x.view);
    CHECK(!route(i1,w1,p1,short_x,map,size,0));ds4_gpu_tensor_free(short_x.view);
    tensor unaligned=x;unaligned.view=ds4_gpu_tensor_view(x.base,17,256*4);CHECK(unaligned.view);
    CHECK(!route(i1,w1,p1,unaligned,map,size,0));ds4_gpu_tensor_free(unaligned.view);
    release(x);release(p0);release(p1);release(i0);release(i1);release(w0);release(w1);
    ds4_gpu_cleanup();CHECK(!munmap(map,size));
    printf("PASS HY4 router: 128 cases, %u checks; probabilities, top8 IDs and weights bit-exact; ties, bias, clamp, offsets, bounds, queued batches\n",checks);
    return 0;
}
