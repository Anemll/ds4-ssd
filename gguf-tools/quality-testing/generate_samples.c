#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char *id;
    char *prompt;
} prompt_case;

typedef struct {
    prompt_case *v;
    int len;
    int cap;
} prompt_list;

typedef struct {
    FILE *fp;
    ds4_engine *engine;
} emit_ctx;

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static char *json_get_string_field(const char *line, const char *field) {
    char key[64];
    snprintf(key, sizeof(key), "\"%s\"", field);
    const char *p = strstr(line, key);
    if (!p) return NULL;
    p += strlen(key);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++;

    size_t cap = strlen(p) + 1;
    char *out = xmalloc(cap);
    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': out[n++] = '\n'; break;
            case 'r': out[n++] = '\r'; break;
            case 't': out[n++] = '\t'; break;
            case '"': out[n++] = '"'; break;
            case '\\': out[n++] = '\\'; break;
            case '/': out[n++] = '/'; break;
            default:
                out[n++] = *p ? *p : '\\';
                break;
            }
            if (*p) p++;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
    return out;
}

static void prompt_list_push(prompt_list *list, const char *id, const char *prompt) {
    if (list->len == list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 128;
        prompt_case *nv = realloc(list->v, (size_t)new_cap * sizeof(nv[0]));
        if (!nv) die("out of memory");
        list->v = nv;
        list->cap = new_cap;
    }
    list->v[list->len].id = xstrdup(id);
    list->v[list->len].prompt = xstrdup(prompt);
    list->len++;
}

static prompt_list load_prompts_jsonl(const char *path, int limit) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }

    prompt_list list = {0};
    char line[65536];
    int fallback_id = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *prompt = json_get_string_field(line, "prompt");
        if (!prompt) continue;
        char *id = json_get_string_field(line, "id");
        char idbuf[32];
        if (!id) {
            snprintf(idbuf, sizeof(idbuf), "case_%03d", fallback_id);
            id = xstrdup(idbuf);
        }
        prompt_list_push(&list, id, prompt);
        free(id);
        free(prompt);
        fallback_id++;
        if (limit > 0 && list.len >= limit) break;
    }
    fclose(fp);
    return list;
}

static void prompt_list_free(prompt_list *list) {
    if (!list) return;
    for (int i = 0; i < list->len; i++) {
        free(list->v[i].id);
        free(list->v[i].prompt);
    }
    free(list->v);
    memset(list, 0, sizeof(*list));
}

static void emit_token(void *ud, int token) {
    emit_ctx *ctx = ud;
    size_t len = 0;
    char *piece = ds4_token_text(ctx->engine, token, &len);
    if (piece && len) (void)fwrite(piece, 1, len, ctx->fp);
    free(piece);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void mkdir_p(const char *path) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    if (system(cmd) != 0) die("mkdir failed");
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [OPTIONS] MODEL prompts.jsonl OUT_DIR\n"
        "\n"
        "Options:\n"
        "  --mode NAME             Label written to summary (default: run)\n"
        "  --ctx N                 Context size (default 4096)\n"
        "  --tokens N              Max generated tokens per prompt (default 128)\n"
        "  --limit N               Prompt limit (default all)\n"
        "  --resident              Use resident sidecar mode\n"
        "  --draft-path PATH       Enable DSpark draft package\n"
        "  --draft-verify N        DSpark verify budget (default 5)\n"
        "  --draft-verify-dynamic  Adapt active DSpark verify budget up to --draft-verify\n"
        "  --draft-scheduler NAME  static|confidence|confidence-softmax|confidence-softmax-long (default static)\n"
        "  --draft-conf-threshold F\n"
        "  --quality               Exact engine quality mode\n"
        "  --no-int8               Disable int8 accelerator paths\n",
        prog);
}

int main(int argc, char **argv) {
    const char *mode = "run";
    const char *model_path = NULL;
    const char *prompts_path = NULL;
    const char *out_dir = NULL;
    const char *draft_path = NULL;
    const char *draft_scheduler = "static";
    int draft_verify = 5;
    float draft_conf = 0.0f;
    int ctx_size = 4096;
    int n_predict = 128;
    int limit = 0;
    bool resident = false;
    bool quality = false;
    bool no_int8 = false;
    bool draft_verify_dynamic = false;

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(a, "--mode")) {
            if (++i >= argc) die("--mode needs value");
            mode = argv[i];
        } else if (!strcmp(a, "--ctx")) {
            if (++i >= argc) die("--ctx needs value");
            ctx_size = atoi(argv[i]);
        } else if (!strcmp(a, "--tokens") || !strcmp(a, "-n")) {
            if (++i >= argc) die("--tokens needs value");
            n_predict = atoi(argv[i]);
        } else if (!strcmp(a, "--limit")) {
            if (++i >= argc) die("--limit needs value");
            limit = atoi(argv[i]);
        } else if (!strcmp(a, "--resident")) {
            resident = true;
        } else if (!strcmp(a, "--draft-path")) {
            if (++i >= argc) die("--draft-path needs value");
            draft_path = argv[i];
        } else if (!strcmp(a, "--draft-verify")) {
            if (++i >= argc) die("--draft-verify needs value");
            draft_verify = atoi(argv[i]);
        } else if (!strcmp(a, "--draft-verify-dynamic")) {
            draft_verify_dynamic = true;
        } else if (!strcmp(a, "--draft-scheduler")) {
            if (++i >= argc) die("--draft-scheduler needs value");
            draft_scheduler = argv[i];
        } else if (!strcmp(a, "--draft-conf-threshold")) {
            if (++i >= argc) die("--draft-conf-threshold needs value");
            draft_conf = strtof(argv[i], NULL);
        } else if (!strcmp(a, "--quality")) {
            quality = true;
            no_int8 = true;
        } else if (!strcmp(a, "--no-int8")) {
            no_int8 = true;
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        } else {
            switch (pos++) {
            case 0: model_path = a; break;
            case 1: prompts_path = a; break;
            case 2: out_dir = a; break;
            default: die("too many positional arguments");
            }
        }
    }
    if (!model_path || !prompts_path || !out_dir) {
        usage(argv[0]);
        return 2;
    }
    if (ctx_size < 1024) ctx_size = 1024;
    if (n_predict < 1) n_predict = 1;

    mkdir_p(out_dir);
    char text_dir[4096];
    snprintf(text_dir, sizeof(text_dir), "%s/outputs", out_dir);
    mkdir_p(text_dir);

    prompt_list prompts = load_prompts_jsonl(prompts_path, limit);
    if (prompts.len == 0) die("no prompts loaded");

    ds4_engine_options opt = {
        .model_path = model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 0,
        .ctx_size = ctx_size,
        .quality = quality,
        .no_int8 = no_int8,
        .resident = resident,
    };
    if (draft_path && draft_path[0]) {
        opt.draft_kind = DS4_DRAFT_DSPARK;
        opt.draft_path = draft_path;
        opt.draft_verify = draft_verify;
        opt.draft_scheduler = draft_scheduler;
        opt.draft_conf_threshold = draft_conf;
        if (draft_verify_dynamic && setenv("DS4_DSPARK_VERIFY_DYNAMIC", "1", 1) != 0) {
            die("setenv DS4_DSPARK_VERIFY_DYNAMIC failed");
        }
    }
    ds4_engine_options_autodetect_sidecar_package(&opt, "generate_samples");
    if (resident) ds4_engine_options_apply_resident_preset(&opt, "generate_samples");

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        ds4_engine_close(engine);
        die("session create failed");
    }

    char summary_path[4096];
    snprintf(summary_path, sizeof(summary_path), "%s/summary.tsv", out_dir);
    FILE *summary = fopen(summary_path, "wb");
    if (!summary) die("open summary failed");
    fprintf(summary,
            "mode\tid\tprompt_tokens\tgenerated_tokens\tdecode_s\tgen_tps\t"
            "draft_slots\tdraft_accepted\tdraft_blocks\tacceptance_pct\toutput_file\n");

    char err[256];
    for (int ci = 0; ci < prompts.len; ci++) {
        const prompt_case *pc = &prompts.v[ci];
        char out_path[4096];
        snprintf(out_path, sizeof(out_path), "%s/%s.txt", text_dir, pc->id);
        FILE *out = fopen(out_path, "wb");
        if (!out) die("open output failed");

        ds4_tokens prompt = {0};
        ds4_encode_chat_prompt(engine, NULL, pc->prompt, DS4_THINK_NONE, &prompt);

        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s sync failed: %s\n", pc->id, err);
            return 1;
        }

        emit_ctx emit = { .fp = out, .engine = engine };
        int generated = 0;
        uint64_t draft_slots = 0;
        uint64_t draft_accepted = 0;
        uint64_t draft_blocks = 0;
        const bool use_speculative =
            draft_path && draft_path[0] && ds4_engine_dspark_draft_tokens(engine) > 0;
        const double t0 = now_sec();
        while (generated < n_predict) {
            int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, NULL);
            if (token == ds4_token_eos(engine) ||
                token == ds4_token_user(engine) ||
                token == ds4_token_assistant(engine)) {
                break;
            }

            int toks[17];
            int ntok = 1;
            toks[0] = token;
            if (use_speculative) {
                int drafted = 0;
                int accepted_draft_count = -1;
                ntok = ds4_session_eval_speculative_argmax(session,
                                                           token,
                                                           n_predict - generated,
                                                           ds4_token_eos(engine),
                                                           toks,
                                                           (int)(sizeof(toks) / sizeof(toks[0])),
                                                           &drafted,
                                                           &accepted_draft_count,
                                                           err,
                                                           sizeof(err));
                if (ntok < 0) {
                    fprintf(stderr, "%s speculative eval failed: %s\n", pc->id, err);
                    return 1;
                }
                int accepted_drafts = accepted_draft_count >= 0 ?
                    accepted_draft_count : (ntok > 1 ? ntok - 1 : 0);
                if (accepted_drafts < 0) accepted_drafts = 0;
                if (drafted > 0) {
                    draft_slots += (uint64_t)drafted;
                    draft_blocks++;
                    if (accepted_drafts > drafted) accepted_drafts = drafted;
                    draft_accepted += (uint64_t)accepted_drafts;
                }
            } else {
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "%s eval failed: %s\n", pc->id, err);
                    return 1;
                }
            }

            for (int i = 0; i < ntok && generated < n_predict; i++) {
                if (toks[i] == ds4_token_eos(engine) ||
                    toks[i] == ds4_token_user(engine) ||
                    toks[i] == ds4_token_assistant(engine)) {
                    goto done_case;
                }
                emit_token(&emit, toks[i]);
                generated++;
            }
        }

done_case:
        ;
        const double decode_s = now_sec() - t0;
        const double gen_tps = decode_s > 0.0 ? (double)generated / decode_s : 0.0;
        const double acc = draft_slots ? 100.0 * (double)draft_accepted / (double)draft_slots : 0.0;
        fprintf(summary,
                "%s\t%s\t%d\t%d\t%.6f\t%.6f\t%llu\t%llu\t%llu\t%.3f\t%s\n",
                mode,
                pc->id,
                prompt.len,
                generated,
                decode_s,
                gen_tps,
                (unsigned long long)draft_slots,
                (unsigned long long)draft_accepted,
                (unsigned long long)draft_blocks,
                acc,
                out_path);
        fflush(summary);
        fprintf(stderr,
                "%s %d/%d generated=%d tps=%.2f acc=%.1f%% out=%s\n",
                pc->id,
                ci + 1,
                prompts.len,
                generated,
                gen_tps,
                acc,
                out_path);

        fclose(out);
        ds4_tokens_free(&prompt);
    }

    fclose(summary);
    ds4_session_free(session);
    ds4_engine_close(engine);
    prompt_list_free(&prompts);
    fprintf(stderr, "wrote %s\n", summary_path);
    return 0;
}
