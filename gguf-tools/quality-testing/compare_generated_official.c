#include "ds4.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *id;
    char *prompt_path;
    char *cont_path;
} manifest_row;

typedef struct {
    manifest_row *v;
    int len;
    int cap;
} manifest_list;

typedef struct {
    char *id;
    int generated_tokens;
    double decode_s;
    double gen_tps;
    uint64_t draft_slots;
    uint64_t draft_accepted;
    uint64_t draft_blocks;
    double acceptance_pct;
    char *output_file;
} summary_row;

typedef struct {
    summary_row *v;
    int len;
    int cap;
} summary_list;

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

static char *read_file_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) die("fseek failed");
    long len = ftell(fp);
    if (len < 0) die("ftell failed");
    rewind(fp);
    char *buf = xmalloc((size_t)len + 1);
    if (len && fread(buf, 1, (size_t)len, fp) != (size_t)len) die("read failed");
    fclose(fp);
    buf[len] = '\0';
    return buf;
}

static void strip_newline(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
}

static void manifest_push(manifest_list *list, manifest_row row) {
    if (list->len == list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 128;
        manifest_row *nv = realloc(list->v, (size_t)new_cap * sizeof(nv[0]));
        if (!nv) die("out of memory");
        list->v = nv;
        list->cap = new_cap;
    }
    list->v[list->len++] = row;
}

static manifest_list load_manifest(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    manifest_list list = {0};
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        strip_newline(line);
        if (!line[0] || line[0] == '#') continue;
        char *id = strtok(line, "\t");
        char *prompt = strtok(NULL, "\t");
        char *cont = strtok(NULL, "\t");
        if (!id || !prompt || !cont) die("bad manifest row");
        manifest_row row = {
            .id = xstrdup(id),
            .prompt_path = xstrdup(prompt),
            .cont_path = xstrdup(cont),
        };
        manifest_push(&list, row);
    }
    fclose(fp);
    return list;
}

static void manifest_free(manifest_list *list) {
    for (int i = 0; i < list->len; i++) {
        free(list->v[i].id);
        free(list->v[i].prompt_path);
        free(list->v[i].cont_path);
    }
    free(list->v);
}

static char *next_tsv_field(char **p) {
    char *s = *p;
    if (!s) return NULL;
    char *tab = strchr(s, '\t');
    if (tab) {
        *tab = '\0';
        *p = tab + 1;
    } else {
        char *nl = strpbrk(s, "\r\n");
        if (nl) *nl = '\0';
        *p = NULL;
    }
    return s;
}

static void summary_push(summary_list *list, summary_row row) {
    if (list->len == list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 128;
        summary_row *nv = realloc(list->v, (size_t)new_cap * sizeof(nv[0]));
        if (!nv) die("out of memory");
        list->v = nv;
        list->cap = new_cap;
    }
    list->v[list->len++] = row;
}

static summary_list load_summary(const char *run_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/summary.tsv", run_dir);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    summary_list list = {0};
    char line[8192];
    if (!fgets(line, sizeof(line), fp)) die("empty summary");
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        char *fields[11];
        int nf = 0;
        while (nf < 11 && (fields[nf] = next_tsv_field(&p)) != NULL) nf++;
        if (nf < 11) continue;
        summary_row row = {
            .id = xstrdup(fields[1]),
            .generated_tokens = atoi(fields[3]),
            .decode_s = strtod(fields[4], NULL),
            .gen_tps = strtod(fields[5], NULL),
            .draft_slots = strtoull(fields[6], NULL, 10),
            .draft_accepted = strtoull(fields[7], NULL, 10),
            .draft_blocks = strtoull(fields[8], NULL, 10),
            .acceptance_pct = strtod(fields[9], NULL),
            .output_file = xstrdup(fields[10]),
        };
        summary_push(&list, row);
    }
    fclose(fp);
    return list;
}

static const summary_row *summary_find(const summary_list *list, const char *id) {
    for (int i = 0; i < list->len; i++) {
        if (!strcmp(list->v[i].id, id)) return &list->v[i];
    }
    return NULL;
}

static void summary_free(summary_list *list) {
    for (int i = 0; i < list->len; i++) {
        free(list->v[i].id);
        free(list->v[i].output_file);
    }
    free(list->v);
}

static int token_lcp(const ds4_tokens *a, const ds4_tokens *b) {
    int n = a->len < b->len ? a->len : b->len;
    int i = 0;
    while (i < n && a->v[i] == b->v[i]) i++;
    return i;
}

static char *load_output_text(const char *run_dir, const summary_row *row) {
    FILE *fp = fopen(row->output_file, "rb");
    if (fp) {
        fclose(fp);
        return read_file_text(row->output_file);
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/outputs/%s.txt", run_dir, row->id);
    return read_file_text(path);
}

static int score_run(ds4_engine *engine,
                     ds4_session *session,
                     const manifest_list *manifest,
                     const char *run_dir,
                     const char *out_path) {
    summary_list summary = load_summary(run_dir);
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "open %s: %s\n", out_path, strerror(errno));
        return 1;
    }
    fprintf(out, "id\tofficial_tokens\tgenerated_model_tokens\tscored_tokens\tgenerated_runtime_tokens\tnll\tavg_nll\tfirst_match\tlcp\tgen_tps\tacceptance_pct\toutput_file\n");

    int cases = 0;
    long official_tokens = 0;
    long generated_model_tokens = 0;
    long generated_runtime_tokens = 0;
    long first_match = 0;
    long lcp_total = 0;
    double nll_total = 0.0;
    long nll_tokens = 0;
    double decode_s = 0.0;
    uint64_t draft_slots = 0;
    uint64_t draft_accepted = 0;
    char err[256];

    for (int i = 0; i < manifest->len; i++) {
        const manifest_row *mr = &manifest->v[i];
        const summary_row *sr = summary_find(&summary, mr->id);
        if (!sr) continue;
        char *prompt_text = read_file_text(mr->prompt_path);
        char *official_text = read_file_text(mr->cont_path);
        char *generated_text = load_output_text(run_dir, sr);
        ds4_tokens prompt = {0};
        ds4_tokens official = {0};
        ds4_tokens generated = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);
        ds4_tokenize_text(engine, official_text, &official);
        ds4_tokenize_text(engine, generated_text, &generated);

        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s sync failed: %s\n", mr->id, err);
            return 1;
        }
        const int score_tokens =
            generated.len < official.len ? generated.len : official.len;
        double nll = 0.0;
        for (int t = 0; t < score_tokens; t++) {
            ds4_token_score score = {0};
            if (!ds4_session_token_logprob(session, generated.v[t], &score) ||
                !isfinite(score.logprob)) {
                fprintf(stderr, "%s logprob failed at generated token %d\n", mr->id, t);
                return 1;
            }
            nll += -(double)score.logprob;
            if (ds4_session_eval(session, generated.v[t], err, sizeof(err)) != 0) {
                fprintf(stderr, "%s eval failed at generated token %d: %s\n",
                        mr->id, t, err);
                return 1;
            }
        }

        const bool fm = official.len > 0 && generated.len > 0 &&
            official.v[0] == generated.v[0];
        const int lcp = token_lcp(&official, &generated);
        const double avg_nll = score_tokens > 0 ? nll / (double)score_tokens : 0.0;
        fprintf(out, "%s\t%d\t%d\t%d\t%d\t%.9f\t%.9f\t%d\t%d\t%.6f\t%.3f\t%s\n",
                mr->id,
                official.len,
                generated.len,
                score_tokens,
                sr->generated_tokens,
                nll,
                avg_nll,
                fm ? 1 : 0,
                lcp,
                sr->gen_tps,
                sr->acceptance_pct,
                sr->output_file);
        fflush(out);

        cases++;
        official_tokens += official.len;
        generated_model_tokens += generated.len;
        generated_runtime_tokens += sr->generated_tokens;
        first_match += fm ? 1 : 0;
        lcp_total += lcp;
        nll_total += nll;
        nll_tokens += score_tokens;
        decode_s += sr->decode_s;
        draft_slots += sr->draft_slots;
        draft_accepted += sr->draft_accepted;

        ds4_tokens_free(&prompt);
        ds4_tokens_free(&official);
        ds4_tokens_free(&generated);
        free(prompt_text);
        free(official_text);
        free(generated_text);
        if ((cases % 10) == 0) {
            fprintf(stderr,
                    "progress %s cases=%d avg_nll=%.6f first_match=%ld avg_lcp=%.3f\n",
                    run_dir,
                    cases,
                    nll_tokens ? nll_total / (double)nll_tokens : 0.0,
                    first_match,
                    cases ? (double)lcp_total / (double)cases : 0.0);
        }
    }
    fclose(out);

    const double gen_tps = decode_s > 0.0 ? (double)generated_runtime_tokens / decode_s : 0.0;
    const double acceptance = draft_slots ? 100.0 * (double)draft_accepted / (double)draft_slots : 0.0;
    fprintf(stderr,
            "summary cases=%d official_tokens=%ld generated_model_tokens=%ld "
            "generated_runtime_tokens=%ld avg_nll=%.9f first_match=%ld avg_lcp=%.3f "
            "generation_tps=%.6f acceptance=%.6f\n",
            cases,
            official_tokens,
            generated_model_tokens,
            generated_runtime_tokens,
            nll_tokens ? nll_total / (double)nll_tokens : 0.0,
            first_match,
            cases ? (double)lcp_total / (double)cases : 0.0,
            gen_tps,
            acceptance);
    summary_free(&summary);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [OPTIONS] MODEL manifest.tsv RUN_DIR OUT_TSV [RUN_DIR OUT_TSV ...]\n"
        "\n"
        "Options:\n"
        "  --ctx N       Context size (default 4096)\n"
        "  --resident    Use resident sidecar mode for scoring\n"
        "  --no-int8     Disable int8 accelerator paths\n"
        "  --quality     Exact engine quality mode; implies --no-int8\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *manifest_path = NULL;
    const char *pairs[32];
    int npairs = 0;
    int ctx_size = 4096;
    bool resident = false;
    bool no_int8 = false;
    bool quality = false;

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(a, "--ctx")) {
            if (++i >= argc) die("--ctx needs value");
            ctx_size = atoi(argv[i]);
        } else if (!strcmp(a, "--resident")) {
            resident = true;
        } else if (!strcmp(a, "--no-int8")) {
            no_int8 = true;
        } else if (!strcmp(a, "--quality")) {
            quality = true;
            no_int8 = true;
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        } else {
            if (pos == 0) model_path = a;
            else if (pos == 1) manifest_path = a;
            else {
                if (npairs >= (int)(sizeof(pairs) / sizeof(pairs[0]))) die("too many run/output pairs");
                pairs[npairs++] = a;
            }
            pos++;
        }
    }

    if (!model_path || !manifest_path || npairs == 0 || (npairs % 2) != 0) {
        usage(argv[0]);
        return 2;
    }
    if (ctx_size < 1024) ctx_size = 1024;

    ds4_engine_options opt = {
        .model_path = model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 0,
        .ctx_size = ctx_size,
        .resident = resident,
        .no_int8 = no_int8,
        .quality = quality,
    };
    ds4_engine_options_autodetect_sidecar_package(&opt, "compare_generated_official");
    if (resident) ds4_engine_options_apply_resident_preset(&opt, "compare_generated_official");
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) die("failed to create session");

    manifest_list manifest = load_manifest(manifest_path);
    int rc = 0;
    for (int i = 0; i < npairs; i += 2) {
        if (score_run(engine, session, &manifest, pairs[i], pairs[i + 1]) != 0) {
            rc = 1;
            break;
        }
    }

    ds4_session_free(session);
    ds4_engine_close(engine);
    manifest_free(&manifest);
    return rc;
}
