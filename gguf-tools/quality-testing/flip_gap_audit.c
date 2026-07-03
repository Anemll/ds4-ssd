/* flip_gap_audit — measure the logit gap at first-divergence positions
 * between two generated-output runs (e.g. no-draft reference vs a
 * realization-changed verifier mode such as forced-MMA strict).
 *
 * For each case present in both runs: tokenize both outputs, find the first
 * divergent token index d, replay prompt + common prefix through the scoring
 * session (reference realization), then read the model logit of BOTH
 * candidate tokens at that state. A tiny |gap| across all cases means the
 * divergences happen only at effectively tied probabilities; a large gap
 * means the alternate realization is making genuinely different choices.
 *
 * Output TSV: id, a_tokens, b_tokens, lcp, div_index, token_a, token_b,
 * logit_a, logit_b, gap, prob_a, prob_b, note.
 *
 * usage: flip_gap_audit [--ctx N] [--resident] MODEL manifest.tsv RUN_A RUN_B OUT_TSV
 */

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

static char *load_case_output(const char *run_dir, const char *id) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/outputs/%s.txt", run_dir, id);
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fclose(fp);
    return read_file_text(path);
}

static int token_lcp(const ds4_tokens *a, const ds4_tokens *b) {
    int n = a->len < b->len ? a->len : b->len;
    int i = 0;
    while (i < n && a->v[i] == b->v[i]) i++;
    return i;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [OPTIONS] MODEL manifest.tsv RUN_A RUN_B OUT_TSV\n"
        "\n"
        "Options:\n"
        "  --ctx N       Context size (default 4096)\n"
        "  --resident    Use resident sidecar mode for scoring\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *manifest_path = NULL;
    const char *run_a = NULL;
    const char *run_b = NULL;
    const char *out_path = NULL;
    int ctx_size = 4096;
    bool resident = false;

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
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        } else {
            if (pos == 0) model_path = a;
            else if (pos == 1) manifest_path = a;
            else if (pos == 2) run_a = a;
            else if (pos == 3) run_b = a;
            else if (pos == 4) out_path = a;
            pos++;
        }
    }
    if (!model_path || !manifest_path || !run_a || !run_b || !out_path) {
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
    };
    ds4_engine_options_autodetect_sidecar_package(&opt, "flip_gap_audit");
    if (resident) ds4_engine_options_apply_resident_preset(&opt, "flip_gap_audit");
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) die("failed to create session");

    manifest_list manifest = load_manifest(manifest_path);
    FILE *out = fopen(out_path, "wb");
    if (!out) die("cannot open output tsv");
    fprintf(out, "id\ta_tokens\tb_tokens\tlcp\tdiv_index\ttoken_a\ttoken_b\t"
                 "logit_a\tlogit_b\tgap\tprob_a\tprob_b\tnote\n");

    int equal_cases = 0, flip_cases = 0, trunc_cases = 0, skipped = 0;
    double max_abs_gap = 0.0, sum_abs_gap = 0.0;

    for (int i = 0; i < manifest.len; i++) {
        const manifest_row *mr = &manifest.v[i];
        char *a_text = load_case_output(run_a, mr->id);
        char *b_text = load_case_output(run_b, mr->id);
        if (!a_text || !b_text) {
            free(a_text);
            free(b_text);
            skipped++;
            continue;
        }
        char *prompt_text = read_file_text(mr->prompt_path);
        ds4_tokens prompt = {0}, ta = {0}, tb = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);
        ds4_tokenize_text(engine, a_text, &ta);
        ds4_tokenize_text(engine, b_text, &tb);

        const int lcp = token_lcp(&ta, &tb);
        const int min_len = ta.len < tb.len ? ta.len : tb.len;
        if (lcp == min_len && ta.len == tb.len) {
            fprintf(out, "%s\t%d\t%d\t%d\t-1\t-1\t-1\t0\t0\t0\t0\t0\tequal\n",
                    mr->id, ta.len, tb.len, lcp);
            equal_cases++;
        } else if (lcp == min_len) {
            fprintf(out, "%s\t%d\t%d\t%d\t-1\t-1\t-1\t0\t0\t0\t0\t0\ttruncation_only\n",
                    mr->id, ta.len, tb.len, lcp);
            trunc_cases++;
        } else {
            char err[256];
            if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
                fprintf(stderr, "%s sync failed: %s\n", mr->id, err);
                skipped++;
                goto next_case;
            }
            bool eval_ok = true;
            for (int t = 0; t < lcp; t++) {
                if (ds4_session_eval(session, ta.v[t], err, sizeof(err)) != 0) {
                    fprintf(stderr, "%s eval failed at %d: %s\n", mr->id, t, err);
                    eval_ok = false;
                    break;
                }
            }
            if (!eval_ok) {
                skipped++;
                goto next_case;
            }
            ds4_token_score sa = {0}, sb = {0};
            if (!ds4_session_token_logprob(session, ta.v[lcp], &sa) ||
                !ds4_session_token_logprob(session, tb.v[lcp], &sb) ||
                !isfinite(sa.logit) || !isfinite(sb.logit)) {
                fprintf(stderr, "%s logprob failed at div %d\n", mr->id, lcp);
                skipped++;
                goto next_case;
            }
            const double gap = (double)sa.logit - (double)sb.logit;
            fprintf(out, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.9f\t%.9f\tflip\n",
                    mr->id, ta.len, tb.len, lcp, lcp,
                    ta.v[lcp], tb.v[lcp],
                    (double)sa.logit, (double)sb.logit, gap,
                    exp((double)sa.logprob), exp((double)sb.logprob));
            fflush(out);
            flip_cases++;
            const double ag = fabs(gap);
            sum_abs_gap += ag;
            if (ag > max_abs_gap) max_abs_gap = ag;
            fprintf(stderr, "flip %s div=%d gap=%.4f prob_a=%.4f prob_b=%.4f\n",
                    mr->id, lcp, gap, exp((double)sa.logprob), exp((double)sb.logprob));
        }
    next_case:
        free(prompt_text);
        free(a_text);
        free(b_text);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&ta);
        ds4_tokens_free(&tb);
    }

    fprintf(stderr,
            "summary equal=%d truncation_only=%d flips=%d skipped=%d "
            "avg_abs_gap=%.6f max_abs_gap=%.6f\n",
            equal_cases, trunc_cases, flip_cases, skipped,
            flip_cases ? sum_abs_gap / flip_cases : 0.0, max_abs_gap);
    fclose(out);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
