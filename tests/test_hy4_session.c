/* Opt-in native HY4 lifecycle/oracle harness. Explicit dense+sidecar paths,
 * native top-8 bank, ctx128, no INT8, no model downloads or default path.
 * --first-only is the first bring-up gate; build alone does not run inference. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HY4_TEST_CTX 128
#define HY4_TEST_SLOTS 8
#define HY4_TOP 16

static void check(int ok, const char *what, const char *err) {
    if (!ok) {
        fprintf(stderr,"FAIL: HY4 %s: %s\n",what,err ? err : "");
        exit(1);
    }
}
static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p=(const unsigned char *)s; *p; ++p) {
        if (*p=='"' || *p=='\\') { putchar('\\'); putchar(*p); }
        else if (*p<32) printf("\\u%04x",*p);
        else putchar(*p);
    }
    putchar('"');
}
static void json_tokens(const ds4_tokens *tokens) {
    putchar('[');
    for(int i=0;i<tokens->len;++i) printf("%s%d",i ? "," : "",tokens->v[i]);
    putchar(']');
}
static void capture(ds4_session *session,const char *stage,ds4_token_score *top) {
    check(ds4_session_top_logprobs(session,top,HY4_TOP)==HY4_TOP,"top16",stage);
    printf("{\"stage\":");json_string(stage);
    printf(",\"position\":%d,\"token_ids\":",ds4_session_pos(session));
    json_tokens(ds4_session_tokens(session));
    printf(",\"argmax\":%d,\"top16\":[",ds4_session_argmax(session));
    for(int i=0;i<HY4_TOP;++i) {
        check(top[i].id>=0 && isfinite(top[i].logit) && isfinite(top[i].logprob),"finite logits",stage);
        printf("%s{\"id\":%d,\"logit\":%.9g,\"logprob\":%.9g}",i ? "," : "",top[i].id,top[i].logit,top[i].logprob);
    }
    puts("]}");fflush(stdout);
}
static void same_top(const ds4_token_score *a,const ds4_token_score *b,const char *stage) {
    float max_delta=0;
    for(int i=0;i<HY4_TOP;++i) {
        if(a[i].id!=b[i].id || fabsf(a[i].logit-b[i].logit)>1e-5f) {
            fprintf(stderr,"FAIL: HY4 %s rank%d reference=(%d,%.9g) actual=(%d,%.9g)\n",
                    stage,i,a[i].id,a[i].logit,b[i].id,b[i].logit);
            exit(1);
        }
        if(fabsf(a[i].logit-b[i].logit)>max_delta) max_delta=fabsf(a[i].logit-b[i].logit);
    }
    printf("{\"check\":");json_string(stage);printf(",\"pass\":true,\"top16_max_abs_delta\":%.9g}\n",max_delta);
    fflush(stdout);
}
typedef struct {
    int completed;
    int expected_total;
    int cancel_after;
} cancel_probe;
static bool cancellation_requested(void *ud) {
    const cancel_probe *probe = ud;
    return probe->completed >= probe->cancel_after;
}
static void cancellation_progress(void *ud, const char *event, int current, int total) {
    cancel_probe *probe = ud;
    check(!strcmp(event,"prefill_token"),"cancel progress event",event);
    check(total==probe->expected_total && current==probe->completed+1,
          "one progress callback per completed token",event);
    ++probe->completed;
}
static void check_cancellation(ds4_session *session, const ds4_tokens *prompt, int suffix) {
    char err[512]={0};
    ds4_tokens three={0};
    check(prompt->len==1,"single Hello token for cancellation",err);
    ds4_tokens_push(&three,prompt->v[0]);ds4_tokens_push(&three,suffix);ds4_tokens_push(&three,suffix);
    cancel_probe probe={.completed=0,.expected_total=3,.cancel_after=1};
    ds4_token_score resumed[HY4_TOP],cold[HY4_TOP],untouched[HY4_TOP];
    ds4_session_invalidate(session);
    ds4_session_set_progress(session,cancellation_progress,&probe);
    ds4_session_set_cancel(session,cancellation_requested,&probe);
    check(ds4_session_sync(session,&three,err,sizeof(err))!=0 && strstr(err,"interrupted"),
          "cancel prefill at completed token boundary",err);
    check(probe.completed==1 && ds4_session_pos(session)==1 &&
          ds4_session_tokens(session)->len==1 && ds4_session_tokens(session)->v[0]==three.v[0],
          "cancel preserves only completed prefix",err);
    capture(session,"cancelled_prefill_prefix",untouched);
    ds4_session_set_cancel(session,NULL,NULL);
    check(ds4_session_sync(session,&three,err,sizeof(err))==0,"resume cancelled prefill",err);
    check(probe.completed==3 && ds4_session_pos(session)==3,"resume callback count and position",err);
    capture(session,"cancel_resumed_prefill",resumed);
    /* Immediate decode cancellation must neither evaluate nor publish a token. */
    probe.cancel_after=probe.completed;
    ds4_session_set_cancel(session,cancellation_requested,&probe);
    check(ds4_session_eval(session,ds4_session_argmax(session),err,sizeof(err))!=0 && strstr(err,"interrupted"),
          "immediate decode cancellation",err);
    check(ds4_session_pos(session)==3 && probe.completed==3,"cancel decode leaves position unchanged",err);
    capture(session,"cancelled_decode_unchanged",untouched);same_top(resumed,untouched,"cancel decode top16 unchanged");
    ds4_session_set_cancel(session,NULL,NULL);ds4_session_set_progress(session,NULL,NULL);
    ds4_session_invalidate(session);
    check(ds4_session_sync(session,&three,err,sizeof(err))==0,"cold cancellation oracle prefill",err);
    capture(session,"cancel_cold_prefill",cold);same_top(resumed,cold,"cancel resume vs cold top16 parity");
    puts("{\"check\":\"cancellation callbacks and committed prefix\",\"pass\":true,\"completed_callbacks\":3}");
    fflush(stdout);ds4_tokens_free(&three);
}


/* Called only after the ctx128 session has been freed: two live Flash-MoE
 * banks would exceed this harness's deliberately conservative allocation. */
static void check_full_context_snapshot(ds4_engine *engine, const ds4_tokens *prompt, int suffix) {
    char err[512]={0};
    ds4_session *session=NULL;
    ds4_tokens three={0};
    ds4_session_snapshot full={0};
    ds4_token_score before[HY4_TOP],after[HY4_TOP];
    ds4_runtime_status status={0};
    check(ds4_session_create(&session,engine,2049)!=0 && session==NULL,
          "reject context beyond HY4 supported range before allocation",err);
    check(ds4_session_create(&session,engine,4)==0,"create four-token context",err);
    check(ds4_session_runtime_status(session,&status)==1 && status.available &&
          status.moe_slot_bank==HY4_TEST_SLOTS && status.moe_slot_bank_capacity==HY4_TEST_SLOTS,
          "exact eight-slot allocation for four-token context BEFORE inference",err);
    check(prompt->len==1,"single Hello token for full-context snapshot",err);
    ds4_tokens_push(&three,prompt->v[0]);ds4_tokens_push(&three,suffix);ds4_tokens_push(&three,suffix);
    check(ds4_session_sync(session,&three,err,sizeof(err))==0,"prefill three of four context tokens",err);
    check(ds4_session_eval(session,ds4_session_argmax(session),err,sizeof(err))==0,
          "decode final available context token",err);
    check(ds4_session_pos(session)==4 && ds4_session_ctx(session)==4,
          "context exactly full before snapshot",err);
    capture(session,"full_context_before_snapshot",before);
    check(ds4_session_save_snapshot(session,&full,err,sizeof(err))==0,"save full-context snapshot",err);
    ds4_session_invalidate(session);
    check(ds4_session_pos(session)==0,"invalidate full-context snapshot state",err);
    check(ds4_session_load_snapshot(session,&full,err,sizeof(err))==0,"reload snapshot into same full context",err);
    check(ds4_session_pos(session)==4 && ds4_session_tokens(session)->len==4,
          "full-context snapshot restores all four tokens",err);
    capture(session,"full_context_snapshot_loaded",after);
    same_top(before,after,"full context snapshot top16 parity");
    check(ds4_session_runtime_status(session,&status)==1 && status.moe_slot_bank==HY4_TEST_SLOTS &&
          status.moe_slot_bank_capacity==HY4_TEST_SLOTS,"eight slots retained after full-context snapshot",err);
    puts("{\"check\":\"full capacity snapshot save and load\",\"pass\":true,\"ctx\":4,\"position\":4,\"slots\":8}");
    fflush(stdout);
    ds4_session_snapshot_free(&full);ds4_tokens_free(&three);ds4_session_free(session);
}

static void conservative_environment(void) {
    const char *pairs[][2]={
        {"DS4_PROFILE","none"},
        {"DS4_FLASH_MOE_DIRECT_MMAP_AUTO","0"},
        {"DS4_FLASH_MOE_DIRECT_MMAP_BANK","0"},
        {"DS4_FLASH_MOE_PREPROTECT_TOPK","0"},
        {"DS4_FLASH_MOE_ANE_PREFILL","0"},
        {"DS4_FLASH_MOE_DECODE_PREFETCH","0"},
        {"DS4_FLASH_MOE_XLAYER_PREFETCH","0"},
        {"DS4_METAL_PREFILL_CHUNK","1"},
    };
    for(size_t i=0;i<sizeof(pairs)/sizeof(pairs[0]);++i)
        check(setenv(pairs[i][0],pairs[i][1],1)==0,"set conservative environment",pairs[i][0]);
}
int main(int argc,char **argv) {
    int first_only=0,greedy_only=0;
    if(argc==4 && !strcmp(argv[3],"--first-only")) first_only=1;
    else if(argc==4 && !strcmp(argv[3],"--greedy-only")) greedy_only=1;
    else if(argc!=3) {
        fprintf(stderr,"usage: %s /path/to/model-dense-f16head.gguf /path/to/sidecar [--first-only|--greedy-only]\n",argv[0]);
        return 2;
    }
    conservative_environment();
    ds4_engine_options opt={
        .model_path=argv[1],.moe_sidecar_path=argv[2],.backend=DS4_BACKEND_METAL,
        .moe_mode=DS4_MOE_MODE_SLOT_BANK,.moe_slot_bank=HY4_TEST_SLOTS,
        .moe_slot_bank_explicit=true,.ctx_size=HY4_TEST_CTX,.no_int8=true,
    };
    char err[512]={0}; ds4_engine *engine=NULL;ds4_session *session=NULL;
    ds4_tokens prompt={0},suffix={0},greedy_prefix={0},resumed={0};
    ds4_session_snapshot initial={0};
    ds4_token_score first[HY4_TOP],step1[HY4_TOP],before_last[HY4_TOP],last[HY4_TOP],actual[HY4_TOP];
    int greedy[4]={0};
    fprintf(stderr,"HY4 TEST: dense=%s sidecar=%s slots=8 ctx=128 no_int8=1 mode=%s\n",
            argv[1],argv[2],first_only ? "first-only" : greedy_only ? "greedy-only" : "lifecycle");
    check(ds4_engine_open(&engine,&opt)==0,"engine open",err);
    check(ds4_engine_uses_hy4_tokenizer(engine),"requires actual HY4 architecture/tokenizer",argv[1]);
    check(ds4_session_create(&session,engine,HY4_TEST_CTX)==0,"session create",err);
    ds4_runtime_status status={0};
    check(ds4_session_runtime_status(session,&status)==1 && status.available &&
          status.moe_slot_bank==HY4_TEST_SLOTS && status.moe_slot_bank_capacity==HY4_TEST_SLOTS,
          "exact eight-slot allocation BEFORE inference",err);
    printf("{\"stage\":\"allocation\",\"slots\":%u,\"capacity\":%u,\"ctx\":%d,\"resident_bytes\":%" PRIu64 ",\"gpu_footprint_bytes\":%" PRIu64 "}\n",
           status.moe_slot_bank,status.moe_slot_bank_capacity,ds4_session_ctx(session),
           status.resident_bytes,status.gpu_footprint_bytes);fflush(stdout);
    ds4_tokenize_text(engine,"Hello",&prompt);
    check(prompt.len>0 && prompt.len<=4,"short raw Hello tokenization",err);
    printf("{\"stage\":\"raw_prompt\",\"text\":\"Hello\",\"token_ids\":");json_tokens(&prompt);puts("}");fflush(stdout);
    check(ds4_session_sync(session,&prompt,err,sizeof(err))==0,"initial prefill",err);
    capture(session,"prefill",first);
    if(first_only) goto done;
    if(!greedy_only) check(ds4_session_save_snapshot(session,&initial,err,sizeof(err))==0,"save initial snapshot",err);
    for(int i=0;i<4;++i) {
        greedy[i]=ds4_session_argmax(session);
        check(ds4_session_eval(session,greedy[i],err,sizeof(err))==0,"greedy decode",err);
        char stage[32];snprintf(stage,sizeof(stage),"decode_%d",i+1);
        capture(session,stage,actual);
        if(i==0) memcpy(step1,actual,sizeof(step1));
        if(i==2) memcpy(before_last,actual,sizeof(before_last));
    }
    memcpy(last,actual,sizeof(last));
    printf("{\"stage\":\"greedy_tokens\",\"token_ids\":[%d,%d,%d,%d]}\n",greedy[0],greedy[1],greedy[2],greedy[3]);fflush(stdout);
    if(greedy_only) goto done;
    ds4_tokens_copy(&greedy_prefix,ds4_session_tokens(session));
    ds4_tokens_copy(&resumed,&greedy_prefix);
    ds4_tokenize_text(engine,"!",&suffix);
    check(suffix.len==1,"one-token resumed suffix",err);
    ds4_tokens_push(&resumed,suffix.v[0]);
    check(ds4_session_sync(session,&resumed,err,sizeof(err))==0,"one-token resumed prefill",err);
    capture(session,"resumed_prefill",actual);
    check(ds4_session_pos(session)==resumed.len,"resumed position",err);
    /* Rewind before the last greedy token. Syncing this exact shorter prefix
     * must expose its original logits, including when sync evaluates no suffix.
     * Then replay the last token to check subsequent KV/slot reuse as well. */
    ds4_tokens shortened=greedy_prefix;
    shortened.len--;
    ds4_session_rewind(session,shortened.len);
    check(ds4_session_pos(session)==shortened.len,"rewind position",err);
    check(ds4_session_sync(session,&shortened,err,sizeof(err))==0,"sync exact shortened prefix after rewind",err);
    capture(session,"rewind_exact_prefix",actual);same_top(before_last,actual,"rewind exact prefix top16 parity");
    check(ds4_session_eval(session,greedy_prefix.v[greedy_prefix.len-1],err,sizeof(err))==0,"replay after rewind",err);
    capture(session,"rewind_replay",actual);same_top(last,actual,"rewind top16 parity");
    check(ds4_session_load_snapshot(session,&initial,err,sizeof(err))==0,"load initial snapshot",err);
    capture(session,"snapshot_restored",actual);same_top(first,actual,"snapshot top16 parity");
    check(ds4_session_eval(session,greedy[0],err,sizeof(err))==0,"decode restored snapshot",err);
    capture(session,"snapshot_decode",actual);same_top(step1,actual,"snapshot decode top16 parity");
    ds4_session_invalidate(session);
    check(ds4_session_sync(session,&prompt,err,sizeof(err))==0,"invalidate and repeat prefill",err);
    capture(session,"repeat_prefill",actual);same_top(first,actual,"repeat top16 parity");
    check_cancellation(session,&prompt,suffix.v[0]);
    check(ds4_session_runtime_status(session,&status)==1 && status.moe_slot_bank==8 && status.moe_slot_bank_capacity==8,
          "eight slots retained after lifecycle",err);
    ds4_session_free(session);
    session=NULL;
    check_full_context_snapshot(engine,&prompt,suffix.v[0]);
done:
    printf("{\"result\":\"PASS\",\"test\":\"native_hy4_%s\",\"slots\":8,\"ctx\":128,\"prompt_tokens\":%d}\n",
           first_only ? "first_token" : greedy_only ? "greedy4" : "lifecycle",prompt.len);fflush(stdout);
    ds4_session_snapshot_free(&initial);ds4_tokens_free(&resumed);ds4_tokens_free(&greedy_prefix);
    ds4_tokens_free(&suffix);ds4_tokens_free(&prompt);ds4_session_free(session);ds4_engine_close(engine);
    return 0;
}
