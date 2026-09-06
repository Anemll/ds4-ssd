/* Native HY4 bring-up: source math from anemll-flash-llama.cpp 34cccef.
 * Dense/quantized projections, iHC and sink-aware attention use Metal.
 * iHC and attention retain scalar numerical oracles. GLM MLA storage has
 * the same latent shape.
 * Included after glm52_runtime.c and the public session structure. */
#include "hy4_math.h"

typedef struct {
    double pre_cpu, post_cpu, head_cpu, attn_gpu, ffn_gpu, router_gpu;
    double router_install_wall;
    uint32_t quant_dispatches, swiglu_dispatches, reduce_dispatches, fused_dispatches;
} hy4_profile;

typedef struct {
    glm52_runtime mla; /* first member: shared MLA allocation/serialization */
    float *streams; // shared tensor contents, also used by the CPU oracle
    ds4_gpu_tensor *streams_gpu, *hc_post_gpu, *hc_mix_gpu;
    bool cpu_ihc;
    float post[4];
    float *attention_scores;
    ds4_gpu_tensor *attention_gate;
    ds4_gpu_tensor *fused_mid;
    bool profile_enabled;
    hy4_profile profile;
} hy4_runtime;

static bool hy4_session_active(const ds4_session *s) {
    return s && s->engine && DS4_MODEL_VARIANT == DS4_VARIANT_HY4;
}
static hy4_runtime *hy4_rt(ds4_session *s) {
    return s ? (hy4_runtime *)s->variant_runtime : NULL;
}

#if !defined(DS4_NO_GPU) && defined(__APPLE__)
/* Opt-in accounting only: no extra submissions or GPU waits. GPU times
 * come from completed command-buffer timestamps, not enqueue wall times. */
static double hy4_profile_cpu_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&ts)!=0) return 0.0;
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
/* Opt-in parity capture; one token only, in a caller-created directory. */
static void hy4_trace_host(const char *name, int il, const float *x, size_t n) {
    const char *dir=getenv("DS4_HY4_TRACE_DIR");
    if (!dir || !*dir || !x) return;
    char path[4096];
    if (snprintf(path,sizeof(path),"%s/%s-%d.bin",dir,name,il)>=(int)sizeof(path)) return;
    FILE *f=fopen(path,"wb");
    if(f) { (void)fwrite(x,sizeof(float),n,f); fclose(f); }
}
static bool hy4_trace_gpu(const char *name,int il,ds4_gpu_tensor *x,size_t n,uint32_t pos) {
    if (pos || !getenv("DS4_HY4_TRACE_DIR")) return true;
    if (!ds4_gpu_synchronize()) return false;
    hy4_trace_host(name,il,ds4_gpu_tensor_contents(x),n);
    return ds4_gpu_begin_commands()!=0;
}

static bool hy4_host_begin(void) { return ds4_gpu_synchronize() != 0; }
static bool hy4_host_end(void) { return ds4_gpu_begin_commands() != 0; }

static bool hy4_gpu_pre(ds4_session *s, const ds4_tensor *fn,
                         const ds4_tensor *scale, const ds4_tensor *base, bool head) {
    hy4_runtime *h=hy4_rt(s);
    const ds4_model *m=&s->engine->model;
    ds4_gpu_tensor *f=ds4_gpu_model_tensor_view(m->map,m->size,fn->abs_offset,fn->bytes);
    ds4_gpu_tensor *sc=ds4_gpu_model_tensor_view(m->map,m->size,scale->abs_offset,scale->bytes);
    ds4_gpu_tensor *b=ds4_gpu_model_tensor_view(m->map,m->size,base->abs_offset,base->bytes);
    bool ok=f && sc && b && ds4_gpu_hy4_hc_pre_tensor(h->mla.cur,h->hc_post_gpu,
        h->hc_mix_gpu,h->streams_gpu,f,sc,b,DS4_N_EMBD,head);
    ds4_gpu_tensor_free(f);ds4_gpu_tensor_free(sc);ds4_gpu_tensor_free(b);
    return ok;
}

static bool hy4_pre(ds4_session *s, const ds4_tensor *fn,
                     const ds4_tensor *scale, const ds4_tensor *base) {
    hy4_runtime *h = hy4_rt(s);
    const ds4_model *m = &s->engine->model;
    const double cpu0=h->profile_enabled ? hy4_profile_cpu_seconds() : 0.0;
    if (!h->cpu_ihc) {
        bool ok=hy4_gpu_pre(s,fn,scale,base,false);
        if(h->profile_enabled) h->profile.pre_cpu+=hy4_profile_cpu_seconds()-cpu0;
        return ok;
    }
    if (!hy4_host_begin()) return false;
    if (!hy4_hc_pre(h->mla.embed_host, h->post, h->streams,
                tensor_data(m, fn), tensor_data(m, scale), tensor_data(m, base),
                DS4_N_EMBD, 4, 1e-5f, 1e-6f, 2.0f)) return false;
    if(h->profile_enabled) h->profile.pre_cpu+=hy4_profile_cpu_seconds()-cpu0;
    return ds4_gpu_tensor_write(h->mla.cur, 0, h->mla.embed_host,
                                 DS4_N_EMBD * sizeof(float)) && hy4_host_end();
}

static bool hy4_post(ds4_session *s, ds4_gpu_tensor *x) {
    hy4_runtime *h = hy4_rt(s);
    const double cpu0=h->profile_enabled ? hy4_profile_cpu_seconds() : 0.0;
    const double gpu0=h->profile_enabled ? ds4_gpu_busy_seconds() : 0.0;
    if (!h->cpu_ihc && !ds4_gpu_hy4_hc_post_tensor(h->streams_gpu,x,h->hc_post_gpu,DS4_N_EMBD)) return false;
    // Preserve the residual completion boundary while moving arithmetic to
    // Metal. Slot bank reuse and completed command timing retain their joins.
    if (!hy4_host_begin()) return false;
    if(h->profile_enabled) {
        double elapsed=ds4_gpu_busy_seconds()-gpu0;
        if(x==h->mla.attn_out) h->profile.attn_gpu+=elapsed;
        else h->profile.ffn_gpu+=elapsed;
    }
    if (!h->cpu_ihc) {
        if(h->profile_enabled) h->profile.post_cpu+=hy4_profile_cpu_seconds()-cpu0;
        return hy4_host_end();
    }
    if (!ds4_gpu_tensor_read(x, 0, h->mla.embed_host, DS4_N_EMBD * sizeof(float))) return false;
    if (!hy4_hc_post(h->streams, h->mla.embed_host, h->streams, h->post, DS4_N_EMBD, 4)) return false;
    if(h->profile_enabled) h->profile.post_cpu+=hy4_profile_cpu_seconds()-cpu0;
    return hy4_host_end();
}

static bool hy4_head(ds4_session *s) {
    hy4_runtime *h = hy4_rt(s);
    const ds4_model *m = &s->engine->model;
    const ds4_weights *w = &s->engine->weights;
    const double cpu0=h->profile_enabled ? hy4_profile_cpu_seconds() : 0.0;
    if (!h->cpu_ihc) {
        bool ok=hy4_gpu_pre(s,w->output_hc_fn,w->output_hc_scale,w->output_hc_base,true);
        if(h->profile_enabled) h->profile.head_cpu+=hy4_profile_cpu_seconds()-cpu0;
        return ok;
    }
    if (!hy4_host_begin()) return false;
    if (!hy4_hc_head(h->mla.embed_host, h->streams, tensor_data(m,w->output_hc_fn),
                 tensor_data(m,w->output_hc_scale), tensor_data(m,w->output_hc_base),
                 DS4_N_EMBD, 4, 1e-5f, 1e-6f)) return false;
    if(h->profile_enabled) h->profile.head_cpu+=hy4_profile_cpu_seconds()-cpu0;
    return ds4_gpu_tensor_write(h->mla.cur, 0, h->mla.embed_host,
                                 DS4_N_EMBD * sizeof(float)) && hy4_host_end();
}

static bool hy4_attention(ds4_session *s, uint32_t il, uint32_t nkeys) {
    hy4_runtime *h = hy4_rt(s);
    glm52_runtime *r = &h->mla;
    const ds4_model *m = &s->engine->model;
    if (!env_flag_enabled("DS4_HY4_CPU_ATTENTION")) {
        const ds4_tensor *w=s->engine->weights.layer[il].attn_sinks;
        ds4_gpu_tensor *sinks=ds4_gpu_model_tensor_view(m->map,m->size,w->abs_offset,w->bytes);
        bool ok=sinks && ds4_gpu_hy4_attention_decode_tensor(r->attn_lora,
            r->q_abs,r->q,r->layer_kv[il],r->layer_kpe[il],r->attn_scores,sinks,
            nkeys,(uint32_t)s->ctx_size,DS4_N_HEAD,1.0f/16.0f);
        ds4_gpu_tensor_free(sinks);
        return ok;
    }
    if (!hy4_host_begin()) return false;
    /* All these tensors are Metal shared buffers. Synchronization above is
     * mandatory before CPU access, including the just-written KV row. */
    bool ok = hy4_mla_attention(ds4_gpu_tensor_contents(r->attn_lora),
         ds4_gpu_tensor_contents(r->q_abs), ds4_gpu_tensor_contents(r->q), 256, 192,
         ds4_gpu_tensor_contents(r->layer_kv[il]), ds4_gpu_tensor_contents(r->layer_kpe[il]),
         tensor_data(m,s->engine->weights.layer[il].attn_sinks),
         DS4_N_HEAD,nkeys,512,64,1.0f/16.0f,NULL,0,h->attention_scores);
    return ok && ds4_gpu_tensor_did_modify(r->attn_lora,0,DS4_N_HEAD*512*sizeof(float)) && hy4_host_end();
}

static bool hy4_attention_gate(ds4_session *s, const ds4_layer_weights *layer) {
    hy4_runtime *h = hy4_rt(s);
    glm52_runtime *r = &h->mla;
    const uint32_t n = DS4_N_HEAD * 256;
    if (!glm52_matmul(h->attention_gate,&s->engine->model,layer->attn_gate,
                       DS4_N_EMBD,n,r->norm)) return false;
    if (!env_flag_enabled("DS4_HY4_CPU_POINTWISE"))
        return ds4_gpu_hy4_sigmoid_mul_tensor(r->attn_heads,r->attn_heads,h->attention_gate,n)!=0;
    if (!hy4_host_begin()) return false;
    float *out = ds4_gpu_tensor_contents(r->attn_heads);
    const float *gate = ds4_gpu_tensor_contents(h->attention_gate);
    hy4_sigmoid_mul(out,out,gate,n);
    return ds4_gpu_tensor_did_modify(r->attn_heads,0,n*sizeof(float)) && hy4_host_end();
}

static bool hy4_eval_moe(ds4_session *s, const ds4_model *m,
                         const ds4_layer_weights *layer, uint32_t il,
                         uint32_t pos, const char **stage) {
    glm52_runtime *r = glm52_rt(s);
    *stage = "moe.router";
    if (!glm52_matmul(r->router_logits,m,layer->ffn_gate_inp,DS4_N_EMBD,256,r->norm) ||
        !ds4_gpu_glm_router_select_tensor(r->router_selected,r->router_weights,r->router_probs,
            m->map,m->size,layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
            256,8,2.827f,layer->ffn_exp_probs_b != NULL,r->router_logits,1)) return false;
    *stage = "moe.protect_and_install";
    hy4_runtime *h=hy4_rt(s);
    const double install0=h->profile_enabled ? now_sec() : 0.0;
    const double router_gpu0=h->profile_enabled ? ds4_gpu_busy_seconds() : 0.0;
    if (!metal_graph_flash_moe_prepare_decode(&s->graph,il,pos) ||
        !s->graph.flash_decode_ids_valid[il]) return false;
    if(h->profile_enabled) {
        h->profile.router_install_wall+=now_sec()-install0;
        h->profile.router_gpu+=ds4_gpu_busy_seconds()-router_gpu0;
    }
    if (!pos && getenv("DS4_HY4_TRACE_DIR")) {
        if (!hy4_host_begin()) return false;
        float ids[8];
        for (uint32_t k=0;k<8;k++) ids[k]=(float)((const int32_t *)ds4_gpu_tensor_contents(r->router_selected))[k];
        hy4_trace_host("ffn_moe_topk",il,ids,8);
        hy4_trace_host("router_weights",il,ds4_gpu_tensor_contents(r->router_weights),8);
        if (!hy4_host_end()) return false;
    }
    // Install has resolved all hits/misses and drained reads. The next residual
    // join completes these bank readers before any subsequent layer/token reuse.
    if (!env_flag_enabled("DS4_HY4_UNFUSED") && !env_flag_enabled("DS4_HY4_CPU_POINTWISE") &&
        !getenv("DS4_HY4_TRACE_DIR") && !s->graph.flash_per_slot_buffers &&
        !s->graph.flash_chunked_mixed_bank &&
        layer->ffn_gate_exps->type==layer->ffn_up_exps->type) {
        const ds4_flash_moe_layer_sidecar *bank=&s->graph.flash_moe->layer[il];
        const uint64_t rows[]={routed_expert_row_bytes(layer->ffn_gate_exps),
            routed_expert_row_bytes(layer->ffn_up_exps),routed_expert_row_bytes(layer->ffn_down_exps)};
        uint64_t strides[3];
        for(unsigned f=0;f<3;f++) strides[f]=s->graph.flash_mixed_slot_bank ?
            bank->expert_stride : bank->family_bytes[f];
        *stage="moe.fused_top8";
        if (!ds4_gpu_hy4_fused_ffn_tensor(r->routed_out,h->fused_mid,r->norm,
                s->graph.flash_gate_bank[il],s->graph.flash_up_bank[il],s->graph.flash_down_bank[il],
                r->router_weights,s->graph.flash_decode_slot_ids[il],s->graph.flash_slot_bank,
                layer->ffn_gate_exps->type,layer->ffn_down_exps->type,DS4_N_EMBD,DS4_N_FF_EXP,rows,strides)) return false;
        if(h->profile_enabled) h->profile.fused_dispatches+=2;
        return true;
    }
    ds4_gpu_tensor *views[8][3] = {{0}};
    bool ok = true;
    for (uint32_t k=0;ok && k<8;k++) {
        const int32_t slot = s->graph.flash_decode_slot_ids[il][k];
        for (uint32_t f=0;f<3;f++) {
            views[k][f] = metal_graph_flash_moe_family_slot_view(&s->graph,il,f,slot);
            if (!views[k][f]) ok=false;
        }
        if (!ok) break;
        ds4_gpu_tensor *down = ds4_gpu_tensor_view(r->routed_down,
                          k*DS4_N_EMBD*sizeof(float),DS4_N_EMBD*sizeof(float));
        *stage = "moe.quant_gate_up";
        ok = down && ds4_gpu_hy4_quant_matvec_tensor(r->routed_gate,views[k][0],r->norm,
                  layer->ffn_gate_exps->type,DS4_N_EMBD,DS4_N_FF_EXP,
                  routed_expert_row_bytes(layer->ffn_gate_exps)) &&
             ds4_gpu_hy4_quant_matvec_tensor(r->routed_up,views[k][1],r->norm,
                  layer->ffn_up_exps->type,DS4_N_EMBD,DS4_N_FF_EXP,
                  routed_expert_row_bytes(layer->ffn_up_exps)) &&
             ds4_gpu_swiglu_tensor(r->routed_mid,r->routed_gate,r->routed_up,
                  DS4_N_FF_EXP,10.0f,1.0f) &&
             ds4_gpu_hy4_quant_matvec_tensor(down,views[k][2],r->routed_mid,
                  layer->ffn_down_exps->type,DS4_N_FF_EXP,DS4_N_EMBD,
                  routed_expert_row_bytes(layer->ffn_down_exps));
        if(ok && h->profile_enabled) {
            h->profile.quant_dispatches+=3;
            h->profile.swiglu_dispatches++;
        }
        ds4_gpu_tensor_free(down);
    }
    const bool cpu_pointwise=env_flag_enabled("DS4_HY4_CPU_POINTWISE");
    if (cpu_pointwise && !hy4_host_begin()) ok=false;
    // Encoded Metal commands retain the backing buffers. The bank cannot be
    // overwritten until the existing residual/token completion boundary.
    for (uint32_t k=0;k<8;k++) for(uint32_t f=0;f<3;f++) ds4_gpu_tensor_free(views[k][f]);
    if (!ok) return false;
    if (!cpu_pointwise) {
        *stage = "moe.weighted_sum8";
        if(h->profile_enabled) h->profile.reduce_dispatches++;
        return ds4_gpu_hy4_weighted_sum8_tensor(r->routed_out,r->routed_down,
                   r->router_weights,DS4_N_EMBD) &&
               hy4_trace_gpu("routed_out",il,r->routed_out,DS4_N_EMBD,pos) &&
               hy4_trace_gpu("routed_down",il,r->routed_down,8*DS4_N_EMBD,pos);
    }
    /* HY4 applies each selected probability AFTER the expert down projection.
     * Keep this separate from the DS4 fused weighted-SwiGLU path. */
    const float *down = ds4_gpu_tensor_contents(r->routed_down);
    const float *weights = ds4_gpu_tensor_contents(r->router_weights);
    float *out = ds4_gpu_tensor_contents(r->routed_out);
    for (uint32_t j=0;j<DS4_N_EMBD;j++) {
        float sum=hy4_f32_mul(down[j],weights[0]);
        for(uint32_t k=1;k<8;k++) sum = hy4_f32_add(sum,hy4_f32_mul(down[k*DS4_N_EMBD+j],weights[k]));
        out[j]=sum;
    }
    if (!pos && getenv("DS4_HY4_TRACE_DIR")) {
        hy4_trace_host("routed_out",il,out,DS4_N_EMBD);
        hy4_trace_host("routed_down",il,down,8*DS4_N_EMBD);
    }
    return ds4_gpu_tensor_did_modify(r->routed_out,0,DS4_N_EMBD*sizeof(float)) && hy4_host_end();
}

static void hy4_session_free(ds4_session *s) {
    hy4_runtime *h=hy4_rt(s);
    if (!h) return;
    (void)ds4_gpu_synchronize();
    ds4_gpu_tensor_free(h->streams_gpu);
    ds4_gpu_tensor_free(h->hc_post_gpu);
    ds4_gpu_tensor_free(h->hc_mix_gpu);
    free(h->attention_scores);
    ds4_gpu_tensor_free(h->attention_gate);
    ds4_gpu_tensor_free(h->fused_mid);
    glm52_session_free(s);
}

static int hy4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    /* This exact source revision has no DSA. Bound the native port to the
     * range where its full MLA attention is equivalent to HY4 top-2048. */
    if (!out || !e || ctx_size<=0 || ctx_size>2048) {
        fprintf(stderr,"ds4: HY4 at source 34cccef requires --ctx 2048 or smaller; native DSA is not implemented\n");
        return 1;
    }
    if (DS4_N_EXPERT_ACTIVE_USED != 8) {
        fprintf(stderr,"ds4: HY4 requires native top-8; remove the expert-topk override\n");
        return 1;
    }
    if (!e->metal_ready || e->backend!=DS4_BACKEND_METAL || !e->flash_moe) {
        fprintf(stderr,"ds4: HY4 currently requires Metal and an explicit routed sidecar\n");
        return 1;
    }
    ds4_session *s=xcalloc(1,sizeof(*s));
    hy4_runtime *h=xcalloc(1,sizeof(*h));
    s->engine=e; s->ctx_size=ctx_size; s->prefill_cap=1; s->variant_runtime=h;
    h->profile_enabled=env_flag_enabled("DS4_HY4_PROFILE");
    h->cpu_ihc=env_flag_enabled("DS4_HY4_CPU_IHC") || getenv("DS4_HY4_TRACE_DIR");
    s->logits=xmalloc(DS4_N_VOCAB*sizeof(float));
    bool ok=glm52_alloc_decode_tensors(s);
    /* HY4's leading dense FFN is wider than GLM52's. */
    ds4_gpu_tensor_free(h->mla.ffn_gate); ds4_gpu_tensor_free(h->mla.ffn_up); ds4_gpu_tensor_free(h->mla.ffn_mid);
    h->mla.ffn_gate=ds4_gpu_tensor_alloc(18432*sizeof(float));
    h->mla.ffn_up=ds4_gpu_tensor_alloc(18432*sizeof(float));
    h->mla.ffn_mid=ds4_gpu_tensor_alloc(18432*sizeof(float));
    h->attention_gate=ds4_gpu_tensor_alloc(DS4_N_HEAD*256*sizeof(float));
    h->fused_mid=ds4_gpu_tensor_alloc(8*DS4_N_FF_EXP*sizeof(float));
    h->streams_gpu=ds4_gpu_tensor_alloc(4*DS4_N_EMBD*sizeof(float));
    h->hc_post_gpu=ds4_gpu_tensor_alloc(4*sizeof(float));
    h->hc_mix_gpu=ds4_gpu_tensor_alloc(8*sizeof(float));
    h->streams=ds4_gpu_tensor_contents(h->streams_gpu);
    h->attention_scores=xmalloc(ctx_size*sizeof(float));
    ok=ok && h->mla.ffn_gate && h->mla.ffn_up && h->mla.ffn_mid && h->attention_gate && h->fused_mid && h->streams && h->hc_post_gpu && h->hc_mix_gpu;
    s->graph.prefill_cap=1; s->graph.quality=e->quality;
    s->graph.dense_mapped_bytes=e->model.size-e->model.tensor_data_pos;
    if(ok) ok=metal_graph_enable_flash_moe(&s->graph,e->flash_moe,&e->weights.layer[DS4_N_DENSE_LEAD]);
    if(ok) {
        s->graph.router_selected=h->mla.router_selected;
        s->graph.router_weights=h->mla.router_weights;
        s->graph.router_probs=h->mla.router_probs;
        s->graph.router_logits=h->mla.router_logits;
    }
    if(!ok) { hy4_session_free(s); free(s->logits); free(s); return 1; }
    fprintf(stderr,"ds4: HY4 native runtime: top-8, %u slots/layer, token-wise prefill, ctx=%d, gate/reduce=%s, iHC=%s\n",
            s->graph.flash_slot_bank,ctx_size,env_flag_enabled("DS4_HY4_CPU_POINTWISE") ? "CPU reference" : "Metal",h->cpu_ihc ? "CPU reference" : "Metal");
    *out=s; return 0;
}
static int hy4_eval_token(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s || !s->engine || !glm52_rt(s)) return 1;
    glm52_runtime *rt = glm52_rt(s);
    ds4_engine *e = s->engine;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t pos = rt->n_past;
    const char *stage = "begin";
    hy4_runtime *h=hy4_rt(s);
    const double wall0=h->profile_enabled ? now_sec() : 0.0;
    const double cpu0=h->profile_enabled ? hy4_profile_cpu_seconds() : 0.0;
    const double gpu0=h->profile_enabled ? ds4_gpu_busy_seconds() : 0.0;
    const uint64_t hits0=s->graph.flash_hits, misses0=s->graph.flash_misses;
    const uint64_t bytes0=s->graph.flash_installed_bytes;
    if(h->profile_enabled) memset(&h->profile,0,sizeof(h->profile));

    if (pos >= (uint32_t)s->ctx_size) {
        snprintf(err, errlen, "HY4 context is full");
        return 1;
    }
    /* Reuse the source-compatible Metal Q4_K embedding decoder. Its half
     * scale rounding is observable in the original model's high nibbles. */
    if (!hy3_embed_token(rt->cur,m,w->token_embd,token) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(rt->cur,0,rt->embed_host,DS4_N_EMBD*sizeof(float))) {
        snprintf(err, errlen, "HY4 failed to load token embedding");
        return 1;
    }

    for (uint32_t h=0;h<4;h++) memcpy(hy4_rt(s)->streams+h*DS4_N_EMBD,rt->embed_host,DS4_N_EMBD*sizeof(float));
    if (!ds4_gpu_tensor_did_modify(h->streams_gpu,0,4*DS4_N_EMBD*sizeof(float))) {
        snprintf(err,errlen,"HY4 failed to publish embedding streams"); return 1;
    }
    (void)ds4_gpu_begin_commands();
    bool ok = true;

#define HY4_STEP(name, expr) do {       \
        if (ok) {                         \
            stage = (name);               \
            ok = (expr);                  \
        }                                 \
    } while (0)

    for (uint32_t il = 0; ok && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        const ds4_layer_weights *layer = &w->layer[il];

        HY4_STEP("attn.ihc_pre",hy4_pre(s,layer->hc_attn_fn,layer->hc_attn_scale,layer->hc_attn_base));
        HY4_STEP("trace.hc_pre",hy4_trace_gpu("hc_attn_pre",il,rt->cur,DS4_N_EMBD,pos));
        HY4_STEP("attn.rms",
                   ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->cur,
                                                  m->map, m->size,
                                                  layer->attn_norm->abs_offset,
                                                  DS4_N_EMBD, DS4_RMS_EPS) != 0);
        HY4_STEP("trace.attn_norm",hy4_trace_gpu("attn_norm",il,rt->norm,DS4_N_EMBD,pos));
        HY4_STEP("attn.q_a",
                   glm52_matmul(rt->qr, m, layer->attn_q_a,
                                DS4_N_EMBD, DS4_N_LORA_Q, rt->norm));
        HY4_STEP("attn.q_a_norm",
                   ds4_gpu_rms_norm_weight_tensor(rt->qr_norm, rt->qr,
                                                  m->map, m->size,
                                                  layer->attn_q_a_norm->abs_offset,
                                                  DS4_N_LORA_Q, DS4_RMS_EPS) != 0);
        HY4_STEP("attn.q_b",
                   glm52_matmul(rt->q, m, layer->attn_q_b,
                                DS4_N_LORA_Q,
                                (uint64_t)DS4_N_HEAD * GLM52_Q_HEAD_DIM,
                                rt->qr_norm));
        HY4_STEP("attn.q_rope",
                   ds4_gpu_rope_tail_tensor(rt->q, 1, DS4_N_HEAD, GLM52_Q_HEAD_DIM,
                                            GLM52_K_PE_DIM, pos, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        HY4_STEP("attn.kv_a",
                   glm52_matmul(rt->kv_raw, m, layer->attn_kv,
                                DS4_N_EMBD, GLM52_K_HEAD_DIM, rt->norm));
        HY4_STEP("attn.kv_norm",
                   ds4_gpu_rms_norm_weight_tensor(rt->kv_norm, rt->kv_lora_raw,
                                                  m->map, m->size,
                                                  layer->attn_kv_a_norm->abs_offset,
                                                  GLM52_KV_LORA_DIM, DS4_RMS_EPS) != 0);
        HY4_STEP("attn.k_rope",
                   ds4_gpu_rope_tail_tensor(rt->k_pe, 1, 1, GLM52_K_PE_DIM,
                                            GLM52_K_PE_DIM, pos, DS4_ROPE_ORIG_CTX,
                                            false, DS4_ROPE_FREQ_BASE,
                                            DS4_ROPE_SCALE_FACTOR, 0.0f, 1.0f,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0);
        HY4_STEP("attn.cache_store",
                   ds4_gpu_glm52_store_kv_tensor(rt->layer_kv[il],
                                                 rt->layer_kpe[il],
                                                 rt->kv_norm,
                                                 rt->k_pe,
                                                 pos,
                                                 (uint32_t)s->ctx_size) != 0);
        HY4_STEP("attn.q_absorb",
                   ds4_gpu_glm52_q8_head_matvec_tensor(rt->q_abs,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_k_b->abs_offset,
                                                       GLM52_Q_NOPE_DIM,
                                                       GLM52_V_HEAD_DIM,
                                                       DS4_N_HEAD,
                                                       GLM52_Q_HEAD_DIM,
                                                       GLM52_V_HEAD_DIM,
                                                       0,
                                                       rt->q) != 0);
        HY4_STEP("attn.decode",
                   hy4_attention(s,il,pos+1u));
        HY4_STEP("attn.v_b",
                   ds4_gpu_glm52_q8_head_matvec_tensor(rt->attn_heads,
                                                       m->map,
                                                       m->size,
                                                       layer->attn_v_b->abs_offset,
                                                       GLM52_V_HEAD_DIM,
                                                       GLM52_V_IMPL_DIM,
                                                       DS4_N_HEAD,
                                                       GLM52_V_HEAD_DIM,
                                                       GLM52_V_IMPL_DIM,
                                                       0,
                                                       rt->attn_lora) != 0);
        HY4_STEP("trace.kqv_out",hy4_trace_gpu("kqv_out",il,rt->attn_heads,DS4_N_HEAD*256,pos));
        HY4_STEP("attn.gate",hy4_attention_gate(s,layer));
        HY4_STEP("attn.o",
                   glm52_matmul(rt->attn_out, m, layer->attn_output,
                                (uint64_t)DS4_N_HEAD * GLM52_V_IMPL_DIM,
                                DS4_N_EMBD,
                                rt->attn_heads));
        HY4_STEP("trace.attn_out",hy4_trace_gpu("attn_out",il,rt->attn_out,DS4_N_EMBD,pos));
        HY4_STEP("attn.residual",
                   hy4_post(s,rt->attn_out));

        HY4_STEP("ffn.ihc_pre",hy4_pre(s,layer->hc_ffn_fn,layer->hc_ffn_scale,layer->hc_ffn_base));
        HY4_STEP("ffn.rms",
                   ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->cur,
                                                  m->map, m->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD, DS4_RMS_EPS) != 0);
        HY4_STEP("trace.ffn_norm",hy4_trace_gpu("ffn_norm",il,rt->norm,DS4_N_EMBD,pos));
        if (il < DS4_N_DENSE_LEAD) {
            HY4_STEP("ffn.dense_gate",
                       glm52_matmul(rt->ffn_gate, m, layer->ffn_gate,
                                    DS4_N_EMBD, 18432u, rt->norm));
            HY4_STEP("ffn.dense_up",
                       glm52_matmul(rt->ffn_up, m, layer->ffn_up,
                                    DS4_N_EMBD, 18432u, rt->norm));
            HY4_STEP("ffn.dense_swiglu",
                       ds4_gpu_swiglu_tensor(rt->ffn_mid,
                                             rt->ffn_gate,
                                             rt->ffn_up,
                                             18432u,
                                             0.0f,
                                             1.0f) != 0);
            HY4_STEP("ffn.dense_down",
                       glm52_matmul(rt->ffn_down, m, layer->ffn_down,
                                    18432u, DS4_N_EMBD, rt->ffn_mid));
            HY4_STEP("ffn.dense_residual",
                       hy4_post(s,rt->ffn_down));
        } else {
            if (ok) ok = hy4_eval_moe(s, m, layer, il, pos, &stage);
            HY4_STEP("ffn.shared_gate",
                       glm52_matmul(rt->shared_gate, m, layer->ffn_gate_shexp,
                                    DS4_N_EMBD,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    rt->norm));
            HY4_STEP("ffn.shared_up",
                       glm52_matmul(rt->shared_up, m, layer->ffn_up_shexp,
                                    DS4_N_EMBD,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    rt->norm));
            HY4_STEP("ffn.shared_swiglu",
                       ds4_gpu_swiglu_tensor(rt->shared_mid,
                                             rt->shared_gate,
                                             rt->shared_up,
                                             (uint32_t)(DS4_N_FF_EXP * DS4_N_EXPERT_SHARED),
                                             0.0f,
                                             1.0f) != 0);
            HY4_STEP("ffn.shared_down",
                       glm52_matmul(rt->shared_out, m, layer->ffn_down_shexp,
                                    (uint64_t)DS4_N_FF_EXP * DS4_N_EXPERT_SHARED,
                                    DS4_N_EMBD,
                                    rt->shared_mid));
            HY4_STEP("ffn.combine",
                       ds4_gpu_add_tensor(rt->ffn_out, rt->shared_out,
                                          rt->routed_out, DS4_N_EMBD) != 0);
            HY4_STEP("trace.ffn_out",hy4_trace_gpu("ffn_out",il,rt->ffn_out,DS4_N_EMBD,pos));
            HY4_STEP("trace.shared_out",hy4_trace_gpu("shared_out",il,rt->shared_out,DS4_N_EMBD,pos));
            HY4_STEP("ffn.residual",
                       hy4_post(s,rt->ffn_out));
        }
        if (!pos && getenv("DS4_HY4_TRACE_DIR")) hy4_trace_host("l_out",il,hy4_rt(s)->streams,DS4_N_EMBD*4);
    }

    HY4_STEP("output.ihc_head",hy4_head(s));
    HY4_STEP("output.rms",
               ds4_gpu_rms_norm_weight_tensor(rt->norm, rt->cur,
                                              m->map, m->size,
                                              w->output_norm->abs_offset,
                                              DS4_N_EMBD, DS4_RMS_EPS) != 0);
    HY4_STEP("trace.result_norm",hy4_trace_gpu("result_norm",-1,rt->norm,DS4_N_EMBD,pos));
    HY4_STEP("output.logits",
               glm52_matmul(rt->logits_gpu, m, w->output,
                            DS4_N_EMBD, DS4_N_VOCAB, rt->norm));

#undef HY4_STEP

    if (!ds4_gpu_synchronize()) ok = false;
    if (!ok) {
        snprintf(err, errlen, "HY4 eval failed at %s", stage ? stage : "unknown");
        s->checkpoint_valid = false;
        return 1;
    }
    if (ds4_gpu_tensor_read(rt->logits_gpu, 0, s->logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0) {
        snprintf(err, errlen, "HY4 failed to read logits");
        s->checkpoint_valid = false;
        return 1;
    }

    if(h->profile_enabled) {
        const hy4_profile *p=&h->profile;
        fprintf(stderr,"HY4_PROFILE {\"pos\":%u,\"slots\":%u,\"topk\":%u,"
                "\"wall_ms\":%.6f,\"worker_cpu_ms\":%.6f,\"gpu_ms\":%.6f,"
                "\"attn_gpu_ms\":%.6f,\"ffn_gpu_ms\":%.6f,\"router_gpu_ms\":%.6f,"
                "\"ihc_pre_cpu_ms\":%.6f,\"ihc_post_cpu_ms\":%.6f,\"ihc_head_cpu_ms\":%.6f,"
                "\"router_install_wall_ms\":%.6f,\"hits\":%llu,\"misses\":%llu,\"installed_bytes\":%llu,"
                "\"routed_quant_dispatches\":%u,\"routed_swiglu_dispatches\":%u,"
                "\"routed_reduce_dispatches\":%u,\"routed_fused_dispatches\":%u,\"routed_path\":\"%s\",\"ihc_path\":\"%s\"}\n",
                pos,s->graph.flash_slot_bank,DS4_N_EXPERT_ACTIVE_USED,
                (now_sec()-wall0)*1000,(hy4_profile_cpu_seconds()-cpu0)*1000,(ds4_gpu_busy_seconds()-gpu0)*1000,
                p->attn_gpu*1000,p->ffn_gpu*1000,p->router_gpu*1000,
                p->pre_cpu*1000,p->post_cpu*1000,p->head_cpu*1000,p->router_install_wall*1000,
                (unsigned long long)(s->graph.flash_hits-hits0),(unsigned long long)(s->graph.flash_misses-misses0),
                (unsigned long long)(s->graph.flash_installed_bytes-bytes0),
                p->quant_dispatches,p->swiglu_dispatches,p->reduce_dispatches,p->fused_dispatches,
                p->fused_dispatches ? (p->quant_dispatches ? "mixed" : "fused_top8") : "per_expert",h->cpu_ihc ? "cpu" : "metal");
    }
    rt->n_past++;
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}


static void hy4_session_reset(ds4_session *s) {
    if (!s || !hy4_rt(s)) return;
    (void)ds4_gpu_synchronize();
    metal_graph_flash_moe_drain_prefill_reads(&s->graph);
    glm52_session_reset(s);
}

static void hy4_session_rewind(ds4_session *s,int pos) {
    hy4_runtime *h=hy4_rt(s);
    if (!h || pos==s->checkpoint.len) return;
    if (pos<=0 || !s->checkpoint_valid) { hy4_session_reset(s); return; }
    const int last=s->checkpoint.v[pos-1];
    h->mla.n_past=(uint32_t)(pos-1);
    s->checkpoint.len=pos-1;
    char err[256]={0};
    if (hy4_eval_token(s,last,err,sizeof(err))) {
        fprintf(stderr,"ds4: HY4 rewind failed: %s\n",err);
        hy4_session_reset(s);
    }
}

static int hy4_session_sync(ds4_session *s, const ds4_tokens *prompt,
                             char *err, size_t errlen) {
    if (!s || !prompt || prompt->len<=0 || prompt->len>=s->ctx_size) {
        snprintf(err,errlen,"HY4 prompt exceeds context"); return 1;
    }
    int start=0;
    if (s->checkpoint_valid && prompt->len>=s->checkpoint.len &&
        ds4_tokens_starts_with(prompt,&s->checkpoint)) start=s->checkpoint.len;
    else hy4_session_reset(s);
    /* Decode and initial/resumed prefill use the same complete top-8 reserve
     * pass. Each token finishes all SSD reads before publishing its checkpoint. */
    for (int i=start;i<prompt->len;i++) {
        if (s->cancel && s->cancel(s->cancel_ud)) {
            snprintf(err,errlen,"HY4 prefill interrupted"); return 1;
        }
        if (hy4_eval_token(s,prompt->v[i],err,errlen)) return 1;
        if (s->progress) s->progress(s->progress_ud,"prefill_token",i+1,prompt->len);
    }
    return 0;
}
static int hy4_session_eval(ds4_session *s,int token,char *err,size_t errlen) {
    if (s->cancel && s->cancel(s->cancel_ud)) {
        snprintf(err,errlen,"HY4 decode interrupted"); return 1;
    }
    if (!s->checkpoint_valid) {
        snprintf(err,errlen,"HY4 requires a valid prefix before decode"); return 1;
    }
    return hy4_eval_token(s,token,err,errlen);
}
static uint64_t hy4_session_payload_bytes(ds4_session *s) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !s->checkpoint_valid) return 0;
    const uint32_t live = rt->n_past;
    uint64_t bytes = (uint64_t)GLM52_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)GLM52_N_EFFECTIVE_LAYER * live *
             (GLM52_KV_LORA_DIM + GLM52_K_PE_DIM) * sizeof(float);
    return bytes;
}

static int hy4_session_save_payload(ds4_session *s, FILE *fp,
                                      char *err, size_t errlen) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !fp || !s->checkpoint_valid) {
        glm52_payload_set_err(err, errlen, "HY4 session has no valid checkpoint to save");
        return 1;
    }
    if (rt->n_past != (uint32_t)s->checkpoint.len) {
        glm52_payload_set_err(err, errlen, "HY4 KV row count does not match checkpoint");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        glm52_payload_set_err(err, errlen, "failed to synchronize Metal before HY4 snapshot");
        return 1;
    }

    const uint32_t header[GLM52_SESSION_PAYLOAD_U32_FIELDS] = {
        UINT32_C(0x34565948),
        GLM52_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        (uint32_t)s->checkpoint.len,
        GLM52_N_EFFECTIVE_LAYER,
        GLM52_KV_LORA_DIM,
        GLM52_K_PE_DIM,
        DS4_N_VOCAB,
        rt->n_past,
        DS4_N_EMBD,
    };
    for (uint32_t i = 0; i < GLM52_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (glm52_payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (glm52_payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (glm52_payload_write_bytes(fp, s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(float),
                                  err, errlen) != 0) {
        return 1;
    }

    uint8_t *buf = xmalloc(GLM52_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t kv_bytes = (uint64_t)rt->n_past * GLM52_KV_LORA_DIM * sizeof(float);
    const uint64_t kpe_bytes = (uint64_t)rt->n_past * GLM52_K_PE_DIM * sizeof(float);
    for (uint32_t il = 0; rc == 0 && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        rc = glm52_payload_write_tensor(fp, rt->layer_kv[il], kv_bytes, buf, err, errlen);
        if (rc == 0) {
            rc = glm52_payload_write_tensor(fp, rt->layer_kpe[il], kpe_bytes, buf, err, errlen);
        }
    }
    free(buf);
    return rc;
}

static int hy4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                                      char *err, size_t errlen) {
    glm52_runtime *rt = glm52_rt(s);
    if (!s || !rt || !fp) {
        glm52_payload_set_err(err, errlen, "invalid HY4 session payload load");
        return 1;
    }
    hy4_session_reset(s);
    uint64_t remaining = payload_bytes;
    uint32_t h[GLM52_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < GLM52_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (glm52_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != UINT32_C(0x34565948) || h[1] != GLM52_SESSION_PAYLOAD_VERSION) {
        glm52_payload_set_err(err, errlen, "unsupported HY4 session payload version");
        return 1;
    }
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_tokens = h[3];
    const uint32_t saved_layers = h[4];
    const uint32_t saved_kv_dim = h[5];
    const uint32_t saved_kpe_dim = h[6];
    const uint32_t saved_vocab = h[7];
    const uint32_t saved_n_past = h[8];
    const uint32_t saved_embd = h[9];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens > (uint32_t)s->ctx_size ||
        saved_n_past != saved_tokens) {
        glm52_payload_set_err(err, errlen, "HY4 KV checkpoint does not fit current context");
        return 1;
    }
    if (saved_layers != GLM52_N_EFFECTIVE_LAYER ||
        saved_kv_dim != GLM52_KV_LORA_DIM ||
        saved_kpe_dim != GLM52_K_PE_DIM ||
        saved_vocab != DS4_N_VOCAB ||
        saved_embd != DS4_N_EMBD) {
        glm52_payload_set_err(err, errlen, "HY4 KV checkpoint was written for a different layout");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (glm52_payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (glm52_payload_read_bytes(fp, s->logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float),
                                 &remaining, err, errlen) != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }

    uint8_t *buf = xmalloc(GLM52_SESSION_IO_CHUNK);
    int rc = 0;
    const uint64_t kv_bytes = (uint64_t)saved_n_past * GLM52_KV_LORA_DIM * sizeof(float);
    const uint64_t kpe_bytes = (uint64_t)saved_n_past * GLM52_K_PE_DIM * sizeof(float);
    for (uint32_t il = 0; rc == 0 && il < GLM52_N_EFFECTIVE_LAYER; il++) {
        rc = glm52_payload_read_tensor(fp, rt->layer_kv[il], kv_bytes, buf,
                                       &remaining, err, errlen);
        if (rc == 0) {
            rc = glm52_payload_read_tensor(fp, rt->layer_kpe[il], kpe_bytes, buf,
                                           &remaining, err, errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        glm52_payload_set_err(err, errlen, "HY4 KV checkpoint has trailing payload bytes");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    rt->n_past = saved_n_past;
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    if (s->graph.flash_moe &&
        !metal_graph_flash_moe_reset_slot_cache_after_prefill(&s->graph,
                                                              "after HY4 KV payload load")) {
        glm52_payload_set_err(err, errlen, "failed to reset Flash-MoE slot cache after HY4 KV payload load");
        return 1;
    }
    s->checkpoint_valid = true;
    return 0;
}

#else
static int hy4_session_create(ds4_session **out,ds4_engine *e,int ctx) {
    (void)out;(void)e;(void)ctx;
    fprintf(stderr,"ds4: HY4 requires this build's native Metal backend\n"); return 1;
}
static void hy4_session_free(ds4_session *s) {(void)s;}
static void hy4_session_reset(ds4_session *s) {(void)s;}
static void hy4_session_rewind(ds4_session *s,int pos) {(void)s;(void)pos;}
static int hy4_session_sync(ds4_session *s,const ds4_tokens *p,char *err,size_t n) {
    (void)s;(void)p;snprintf(err,n,"HY4 Metal backend unavailable"); return 1;
}
static int hy4_session_eval(ds4_session *s,int t,char *err,size_t n) {
    (void)s;(void)t;snprintf(err,n,"HY4 Metal backend unavailable"); return 1;
}
#ifndef DS4_NO_GPU
static uint64_t hy4_session_payload_bytes(ds4_session *s) {(void)s;return 0;}
static int hy4_session_save_payload(ds4_session *s,FILE *f,char *e,size_t n) {
    (void)s;(void)f;snprintf(e,n,"HY4 Metal backend unavailable");return 1;
}
static int hy4_session_load_payload(ds4_session *s,FILE *f,uint64_t b,char *e,size_t n) {
    (void)s;(void)f;(void)b;snprintf(e,n,"HY4 Metal backend unavailable");return 1;
}
#endif
#endif
