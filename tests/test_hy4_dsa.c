/* Model-free HY4 DSA oracle: real top-2048 over causal F32 index keys,
 * tail NeoX RoPE, LayerNorm, noncontiguous MLA gathers and attention sinks.
 * Peak memory < 150 MiB, including the 50,480-row case. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4_gpu.h"
#include "../hy4/hy4_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
bool ds4_log_is_tty(FILE *f) { (void)f; return false; }
#define CHECK(x) do { checks++; if(!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1); } } while(0)
static unsigned checks;
static uint32_t rng=12345;
static float rnd(void) { rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng%2049)-1024)/1024.0f; }
static ds4_gpu_tensor *alloc(size_t n) { ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(n*4);CHECK(t);return t; }
static void ready(ds4_gpu_tensor *t) { CHECK(ds4_gpu_tensor_did_modify(t,0,ds4_gpu_tensor_bytes(t))); }
static void norm_rope(void) {
    ds4_gpu_tensor *base=alloc(136),*x=ds4_gpu_tensor_view(base,16,512),*w=alloc(128),*b=alloc(128);
    CHECK(x);float *all=ds4_gpu_tensor_contents(base),*v=ds4_gpu_tensor_contents(x),*ww=ds4_gpu_tensor_contents(w),*bb=ds4_gpu_tensor_contents(b);
    for(int c=0;c<6;c++) {
        for(int i=0;i<136;i++)all[i]=-999.f;
        double mean=0,var=0;float orig[128],ref[128];
        for(int i=0;i<128;i++) { orig[i]=v[i]=c==0 ? 2.f : c==1 ? 1.f+rnd()*1e-4f : rnd()*3;ww[i]=rnd();bb[i]=rnd();mean+=v[i]; }
        mean/=128;for(int i=0;i<128;i++) var+=(orig[i]-mean)*(orig[i]-mean);var/=128;
        for(int i=0;i<128;i++)ref[i]=(orig[i]-mean)/sqrt(var+1e-6)*ww[i]+bb[i];
        ready(base);ready(w);ready(b);CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hy4_index_norm_tensor(x,x,w,b));CHECK(ds4_gpu_synchronize());
        for(int i=0;i<128;i++) CHECK(fabs(v[i]-ref[i])<1e-4);
        memcpy(ref,v,sizeof(ref)); uint32_t pos=(uint32_t[]){0,1,2047,2048,2049,50479}[c];
        CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_rope_neox_tensor(x,1,1,128,64,pos,1048576,1e7f,1.f));CHECK(ds4_gpu_synchronize());
        for(int i=0;i<64;i++)CHECK(v[i]==ref[i]);
        for(int i=0;i<32;i++) {
            // Use the float exponent used by the shared Metal RoPE API.
            float angle=pos*powf(1e7f,-(float)i/32.f),co=cosf(angle),si=sinf(angle);
            CHECK(fabsf(v[64+i]-(ref[64+i]*co-ref[96+i]*si))<.02f);
            CHECK(fabsf(v[96+i]-(ref[64+i]*si+ref[96+i]*co))<.02f);
        }
        for(int i=0;i<4;i++) {CHECK(all[i]==-999.f);CHECK(all[132+i]==-999.f);}
    }
    CHECK(!ds4_gpu_hy4_index_norm_tensor(w,x,w,b));
    ds4_gpu_tensor *small=ds4_gpu_tensor_view(w,0,508);CHECK(small);
    CHECK(!ds4_gpu_hy4_index_norm_tensor(x,x,small,b));ds4_gpu_tensor_free(small);
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(base);ds4_gpu_tensor_free(w);ds4_gpu_tensor_free(b);
}
static float *sort_scores;
static int cmp(const void *a,const void *b) { unsigned i=*(const unsigned*)a,j=*(const unsigned*)b;return sort_scores[i]>sort_scores[j] ? -1 : sort_scores[i]<sort_scores[j] ? 1 : (i>j)-(i<j); }
static void run(unsigned live) {
    unsigned count=live<2048?live:2048,ctx=live+2;
    ds4_gpu_tensor *q=alloc(4096),*w=alloc(32),*k=alloc((size_t)ctx*128),*scores=alloc(ctx),*sel=alloc(count);
    float *qq=ds4_gpu_tensor_contents(q),*ww=ds4_gpu_tensor_contents(w),*kk=ds4_gpu_tensor_contents(k),*ss=ds4_gpu_tensor_contents(scores);
    unsigned *ids=ds4_gpu_tensor_contents(sel),*order=malloc(live*4);unsigned char *seen=calloc(live,1);float *ref=malloc(live*4);
    CHECK(order&&seen&&ref);
    for(int i=0;i<4096;i++)qq[i]=rnd();for(int i=0;i<32;i++)ww[i]=rnd();
    for(size_t i=0;i<(size_t)ctx*128;i++)kk[i]=i<(size_t)live*128 ? rnd() : NAN;
    for(unsigned p=0;p<live;p++) { double sum=0;order[p]=p;for(int h=0;h<32;h++) {double dot=0;for(int d=0;d<128;d++)dot+=(double)qq[h*128+d]*kk[(size_t)p*128+d];sum+=fmax(dot,0)*ww[h]/64.;}ref[p]=sum; }
    ready(q);ready(w);ready(k);CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_indexer_score_one_tensor(scores,q,w,k,live,32,128,1.f/64.f));
    CHECK(ds4_gpu_indexer_topk_tensor(sel,scores,live,1,count));CHECK(ds4_gpu_synchronize());
    double worst=0;for(unsigned p=0;p<live;p++) {double d=fabs(ss[p]-ref[p]);if(d>worst)worst=d;CHECK(isfinite(ss[p])&&d<2e-5);}
    sort_scores=ss;qsort(order,live,4,cmp);
    for(unsigned i=0;i<count;i++) {CHECK(ids[i]<live&&!seen[ids[i]]);seen[ids[i]]=1;CHECK(ss[ids[i]]==ss[order[i]]);}
    // Poisoned future rows are excluded from scoring. Gather arbitrary selected
    // rows and compare both attention implementations against the scalar oracle.
    ds4_gpu_tensor *kv=alloc((size_t)ctx*512),*pe=alloc((size_t)ctx*64),*ck=alloc((size_t)count*512),*cp=alloc((size_t)count*64);
    ds4_gpu_tensor *qa=alloc(4*512),*qr=alloc(4*256),*sink=alloc(4),*out=alloc(4*512),*att=alloc((size_t)count*4);
    float *kvv=ds4_gpu_tensor_contents(kv),*pev=ds4_gpu_tensor_contents(pe),*qav=ds4_gpu_tensor_contents(qa),*qrv=ds4_gpu_tensor_contents(qr),*sv=ds4_gpu_tensor_contents(sink);
    for(size_t i=0;i<(size_t)ctx*512;i++)kvv[i]=i<(size_t)live*512?rnd():NAN;
    for(size_t i=0;i<(size_t)ctx*64;i++)pev[i]=i<(size_t)live*64?rnd():NAN;
    for(int i=0;i<4*512;i++)qav[i]=rnd();for(int i=0;i<4*256;i++)qrv[i]=rnd();for(int i=0;i<4;i++)sv[i]=(i-2)*10;
    ready(kv);ready(pe);ready(qa);ready(qr);ready(sink);
    float oracle[4*512];float *scratch=malloc(count*4);CHECK(scratch);
    CHECK(hy4_mla_attention(oracle,qav,qrv,256,192,kvv,pev,sv,4,live,512,64,.0625f,ids,count,scratch));
    for(int sg=0;sg<2;sg++) {
        setenv("DS4_HY4_SG_ATTENTION",sg?"1":"0",1);CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_hy4_gather_kv_tensor(ck,cp,kv,pe,sel,live,count));
        CHECK(ds4_gpu_hy4_attention_decode_tensor(out,qa,qr,ck,cp,att,sink,count,count,4,.0625f));CHECK(ds4_gpu_synchronize());
        const float *cv=ds4_gpu_tensor_contents(ck),*pv=ds4_gpu_tensor_contents(cp),*ov=ds4_gpu_tensor_contents(out);
        for(unsigned i=0;i<count;i++) {CHECK(!memcmp(cv+(size_t)i*512,kvv+(size_t)ids[i]*512,512*4));CHECK(!memcmp(pv+(size_t)i*64,pev+(size_t)ids[i]*64,64*4));}
        for(int i=0;i<4*512;i++)CHECK(isfinite(ov[i])&&fabsf(ov[i]-oracle[i])<1e-5);
    }
    CHECK(!ds4_gpu_hy4_gather_kv_tensor(kv,cp,kv,pe,sel,live,count));
    CHECK(!ds4_gpu_hy4_gather_kv_tensor(ck,cp,kv,pe,sel,0,count));
    // Invalid indices must never address a future/OOB cache row.
    ids[0]=UINT32_MAX;ready(sel);CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_hy4_gather_kv_tensor(ck,cp,kv,pe,sel,live,count));CHECK(ds4_gpu_synchronize());CHECK(isnan(*(float*)ds4_gpu_tensor_contents(ck)));
    ds4_gpu_tensor *all[]={q,w,k,scores,sel,kv,pe,ck,cp,qa,qr,sink,out,att};for(unsigned i=0;i<sizeof(all)/sizeof(*all);i++)ds4_gpu_tensor_free(all[i]);
    free(order);free(seen);free(ref);free(scratch);
    printf("PASS HY4 DSA live=%u selected=%u score_max_error=%.3g original+SG attention\n",live,count,worst);fflush(stdout);
}
int main(void) {CHECK(ds4_gpu_init());norm_rope();unsigned rows[]={1,2047,2048,2049,4097,50480};for(unsigned i=0;i<sizeof(rows)/sizeof(*rows);i++)run(rows[i]);printf("PASS HY4 DSA %u checks\n",checks);return 0;}
