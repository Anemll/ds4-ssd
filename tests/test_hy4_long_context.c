/* Actual-model DSA boundary/lifecycle test. Run the eight-slot short harness
 * first; this longer test accepts an explicit 8..48 slot bank, default 8.
 * It allocates ctx50480 and evaluates a real 2050-token prefix. This is not
 * a claim that 50,480 tokens of model prefill have been tested. */
#define main hy4_short_harness_main
#include "test_hy4_session.c"
#undef main
static void progress_long(void *ud,const char *event,int current,int total) {
    (void)ud;check(!strcmp(event,"prefill_token"),"long prefill progress",event);
    if(current%64==0 || current>=2047) {fprintf(stderr,"HY4 LONG prefill %d/%d\n",current,total);fflush(stderr);}
}
static void summary(ds4_session *s,const char *stage,ds4_token_score *top) {
    check(ds4_session_top_logprobs(s,top,HY4_TOP)==HY4_TOP,"long top16",stage);
    printf("{\"stage\":\"%s\",\"pos\":%d,\"argmax\":%d,\"top16\":[",stage,ds4_session_pos(s),ds4_session_argmax(s));
    for(int i=0;i<HY4_TOP;i++) {check(isfinite(top[i].logit),"finite long-context logits",stage);printf("%s[%d,%.9g]",i?",":"",top[i].id,top[i].logit);}puts("]}");fflush(stdout);
}
int main(int argc,char **argv) {
    if(argc<3 || argc>4) {fprintf(stderr,"usage: %s HY4_DENSE HY4_SIDECAR [SLOTS=8, maximum 48]\n",argv[0]);return 2;}
    int slots=argc==4?atoi(argv[3]):8;check(slots>=8&&slots<=48,"conservative explicit slots",NULL);
    conservative_environment();
    ds4_engine_options opt={.model_path=argv[1],.moe_sidecar_path=argv[2],.backend=DS4_BACKEND_METAL,
        .moe_mode=DS4_MOE_MODE_SLOT_BANK,.moe_slot_bank=slots,.moe_slot_bank_explicit=true,.ctx_size=50480,.no_int8=true};
    char err[512]={0};ds4_engine *e=NULL;ds4_session *s=NULL;ds4_tokens text={0},prompt={0};
    ds4_session_snapshot boundary={0},end={0},roundtrip={0};ds4_token_score dense[HY4_TOP],sparse[HY4_TOP],final[HY4_TOP],actual[HY4_TOP];
    check(ds4_engine_open(&e,&opt)==0,"long engine",err);check(ds4_engine_uses_hy4_tokenizer(e),"actual HY4 tokenizer",NULL);
    check(ds4_session_create(&s,e,50480)==0,"ctx50480 session",err);
    ds4_runtime_status st={0};check(ds4_session_runtime_status(s,&st)==1&&st.moe_slot_bank==(unsigned)slots,"long bank size",NULL);
    fprintf(stderr,"HY4 LONG allocated ctx=50480 slots=%d; real prefill to 2050 tokens\n",slots);
    ds4_tokenize_text(e,"This is a deterministic test of a long conversation. The attention indexer must retain the full causal history and choose the most relevant tokens.\n",&text);
    check(text.len>0,"test text tokenization",NULL);for(int i=0;i<2050;i++)ds4_tokens_push(&prompt,text.v[i%text.len]);
    ds4_tokens prefix=prompt;prefix.len=2047;ds4_session_set_progress(s,progress_long,NULL);
    check(ds4_session_sync(s,&prefix,err,sizeof(err))==0,"real 2047-token prefill",err);
    check(ds4_session_save_snapshot(s,&boundary,err,sizeof(err))==0,"v2 boundary snapshot",err);
    uint32_t version=0;memcpy(&version,(char*)boundary.ptr+4,4);check(version==2,"long snapshot v2",NULL);
    cancel_probe cancel={.completed=2047,.expected_total=2050,.cancel_after=2048};
    ds4_session_set_progress(s,cancellation_progress,&cancel);ds4_session_set_cancel(s,cancellation_requested,&cancel);
    check(ds4_session_sync(s,&prompt,err,sizeof(err))!=0&&strstr(err,"interrupted"),"cancel at DSA boundary",err);
    check(ds4_session_pos(s)==2048&&cancel.completed==2048,"boundary cancel commits one token",NULL);
    summary(s,"dense_2048",dense);ds4_session_set_cancel(s,NULL,NULL);ds4_session_set_progress(s,NULL,NULL);
    check(ds4_session_eval(s,prompt.v[2048],err,sizeof(err))==0,"first sparse token",err);summary(s,"sparse_2049",sparse);
    check(ds4_session_eval(s,prompt.v[2049],err,sizeof(err))==0,"second sparse token",err);summary(s,"sparse_2050",final);
    check(ds4_session_save_snapshot(s,&end,err,sizeof(err))==0,"save complete sparse history",err);
    ds4_session_rewind(s,2048);check(ds4_session_pos(s)==2048,"rewind across boundary position",NULL);summary(s,"rewind_2048",actual);same_top(dense,actual,"rewind across DSA boundary");
    check(ds4_session_eval(s,prompt.v[2048],err,sizeof(err))==0,"replay first sparse token",err);summary(s,"replay_2049",actual);same_top(sparse,actual,"sparse replay exact");
    check(ds4_session_load_snapshot(s,&boundary,err,sizeof(err))==0,"load pre-boundary index keys",err);
    ds4_session_set_progress(s,progress_long,NULL);check(ds4_session_sync(s,&prompt,err,sizeof(err))==0,"resumed sparse prefill",err);
    summary(s,"resumed_prefill_2050",actual);same_top(final,actual,"sparse prefill vs decode");
    check(ds4_session_save_snapshot(s,&roundtrip,err,sizeof(err))==0,"save resumed v2",err);
    check(end.len==roundtrip.len&&!memcmp(end.ptr,roundtrip.ptr,end.len),"all MLA/index keys and logits bit-exact after resume",NULL);
    // A failed or legacy load must invalidate the checkpoint, not expose stale
    // index keys from the previous successful run.
    uint64_t n=end.len;end.len--;check(ds4_session_load_snapshot(s,&end,err,sizeof(err))!=0,"reject truncated v2",err);end.len=n;
    check(ds4_session_eval(s,prompt.v[2049],err,sizeof(err))!=0,"failed load leaves checkpoint invalid",err);
    uint32_t old=1;memcpy((char*)end.ptr+4,&old,4);check(ds4_session_load_snapshot(s,&end,err,sizeof(err))!=0,"reject MLA-only v1 in long session",err);old=2;memcpy((char*)end.ptr+4,&old,4);
    check(ds4_session_load_snapshot(s,&end,err,sizeof(err))==0,"restore after rejected payloads",err);summary(s,"v2_restored",actual);same_top(final,actual,"v2 restored logits");
    // Preserve an optional artifact for repeat tests without relying on an old
    // agent cache. Caller supplies a permanent directory/path.
    const char *path=getenv("HY4_LONG_PAYLOAD_OUT");if(path&&*path) {FILE *f=fopen(path,"wb");check(f!=NULL,"open payload artifact",path);check(ds4_session_save_payload(s,f,err,sizeof(err))==0,"save payload artifact",err);check(fclose(f)==0,"close payload artifact",path);}
    ds4_session_invalidate(s);prefix.len=1;check(ds4_session_sync(s,&prefix,err,sizeof(err))==0,"reset long-context session",err);
    printf("{\"result\":\"PASS\",\"test\":\"hy4_dsa_boundary_lifecycle\",\"slots\":%d,\"allocated_context\":50480,\"evaluated_tokens\":2050,\"snapshot_bytes\":%llu}\n",slots,(unsigned long long)end.len);
    ds4_session_snapshot_free(&boundary);ds4_session_snapshot_free(&end);ds4_session_snapshot_free(&roundtrip);
    ds4_tokens_free(&text);ds4_tokens_free(&prompt);ds4_session_free(s);ds4_engine_close(e);return 0;
}
