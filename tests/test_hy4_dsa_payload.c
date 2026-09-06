/* Model-free regression for the actual HY4 payload readers/writers. Small
 * synthetic Metal caches exercise every full/shared layer, independently of
 * expensive model inference. Does not claim model numerical validation. */
#include "../ds4.c"
#include <assert.h>
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,err);exit(1);}}while(0)
static char err[512];
static ds4_gpu_tensor *filled(unsigned rows,unsigned dim,unsigned tag) {
    ds4_gpu_tensor *t=ds4_gpu_tensor_alloc((uint64_t)rows*dim*4);CHECK(t);
    float *p=ds4_gpu_tensor_contents(t);for(unsigned i=0;i<rows*dim;i++)p[i]=(float)(tag*1000+i)/16;
    CHECK(ds4_gpu_tensor_did_modify(t,0,(uint64_t)rows*dim*4));return t;
}
int main(void) {
    g_ds4_shape=DS4_SHAPE_HY4;CHECK(ds4_gpu_init());
    ds4_engine e={0};ds4_session s={.engine=&e,.ctx_size=50480};hy4_runtime h={0};s.variant_runtime=&h;h.dsa=true;
    s.logits=xmalloc(DS4_N_VOCAB*4);for(unsigned i=0;i<DS4_N_VOCAB;i++)s.logits[i]=(float)i/256;
    unsigned full=0;for(unsigned il=0;il<DS4_N_LAYER;il++) {
        g_hy4_indexer_is_full[il]=il==0||il%4==1;full+=g_hy4_indexer_is_full[il];
        h.mla.layer_kv[il]=filled(3,512,il);h.mla.layer_kpe[il]=filled(3,64,il+100);
        if(g_hy4_indexer_is_full[il])h.index_keys[il]=filled(3,128,il+200);
    }
    CHECK(full==21);for(int i=0;i<3;i++)token_vec_push(&s.checkpoint,i+100);s.checkpoint_valid=true;h.mla.n_past=3;
    ds4_session_snapshot a={0},b={0};CHECK(ds4_session_save_snapshot(&s,&a,err,sizeof(err))==0);
    CHECK(a.len==40+12+DS4_N_VOCAB*4+(uint64_t)78*3*576*4+(uint64_t)21*3*128*4);
    // Clear every cache and all logits. A successful byte-identical roundtrip
    // therefore requires the v2 reader to restore every indexer key too.
    for(unsigned il=0;il<78;il++) {
        memset(ds4_gpu_tensor_contents(h.mla.layer_kv[il]),0,3*512*4);
        memset(ds4_gpu_tensor_contents(h.mla.layer_kpe[il]),0,3*64*4);
        if(h.index_keys[il])memset(ds4_gpu_tensor_contents(h.index_keys[il]),0,3*128*4);
    }
    memset(s.logits,0,DS4_N_VOCAB*4);CHECK(ds4_session_load_snapshot(&s,&a,err,sizeof(err))==0);
    CHECK(ds4_session_save_snapshot(&s,&b,err,sizeof(err))==0);CHECK(a.len==b.len&&!memcmp(a.ptr,b.ptr,a.len));
    const uint64_t sizes[]={1,39,40,44,52,100,0};
    uint64_t original=a.len;
    for(unsigned i=0;i<sizeof(sizes)/sizeof(*sizes);i++) {
        a.len=sizes[i]?sizes[i]:original-1;
        CHECK(ds4_session_load_snapshot(&s,&a,err,sizeof(err))!=0);CHECK(!s.checkpoint_valid&&h.mla.n_past==0);
    }
    a.len=original;
    const unsigned fields[]={1,2,3,4,5,6,7,8,9,10};
    for(unsigned i=0;i<sizeof(fields)/sizeof(*fields);i++) {
        uint32_t *header=(uint32_t*)a.ptr;unsigned f=fields[i];uint32_t old=header[f];header[f]=f==1?1:UINT32_MAX;
        CHECK(ds4_session_load_snapshot(&s,&a,err,sizeof(err))!=0);CHECK(!s.checkpoint_valid);header[f]=old;
    }
    CHECK(ds4_session_load_snapshot(&s,&a,err,sizeof(err))==0);
    // Short sessions retain v1, and a v1 payload cannot seed a long session.
    s.ctx_size=128;h.dsa=false;CHECK(ds4_session_save_snapshot(&s,&b,err,sizeof(err))==0);CHECK(((uint32_t*)b.ptr)[1]==1);
    s.ctx_size=50480;h.dsa=true;CHECK(ds4_session_load_snapshot(&s,&b,err,sizeof(err))!=0);CHECK(!s.checkpoint_valid);
    s.ctx_size=128;h.dsa=false;CHECK(ds4_session_load_snapshot(&s,&b,err,sizeof(err))==0);CHECK(s.checkpoint_valid);
    for(unsigned il=0;il<78;il++){ds4_gpu_tensor_free(h.mla.layer_kv[il]);ds4_gpu_tensor_free(h.mla.layer_kpe[il]);ds4_gpu_tensor_free(h.index_keys[il]);}
    token_vec_free(&s.checkpoint);free(s.logits);ds4_session_snapshot_free(&a);ds4_session_snapshot_free(&b);
    printf("PASS HY4 DSA payload: %u checks, 21 full/57 shared layers, v2 exact roundtrip, malformed/v1 rejection, v1 short compatibility\n",checks);return 0;
}
