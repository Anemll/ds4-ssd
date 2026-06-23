#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) die("fseek failed");
    long n = ftell(fp);
    if (n < 0) die("ftell failed");
    if (fseek(fp, 0, SEEK_SET) != 0) die("fseek failed");
    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory");
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die("read failed");
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

static void strip_newline(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [OPTIONS] MODEL manifest.tsv OUT.tsv\n"
        "\n"
        "Options:\n"
        "  --ctx N          Context size (default 4096, min 1024)\n"
        "  --no-int8        Disable int8 accelerator paths (FP8/fallback)\n"
        "  --quality        Exact kernels; implies --no-int8\n"
        "  --ssd-cache S    SSD cache budget (e.g. 25GB or auto)\n"
        "  --resident       Load a sidecar package as an all-expert resident bank\n"
        "  --limit N        Score at most N cases (0 = all, default: all)\n"
        "  --first-token-only  Score only the first target token\n"
        "  --moe-slot-bank N  Streaming slots per layer (default 32; resident defaults to 256)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *manifest_path = NULL;
    const char *out_path = NULL;
    int ctx_size = 4096;
    bool no_int8 = false;
    bool quality = false;
    bool resident = false;
    const char *ssd_cache = NULL;
    int moe_slot_bank = 32;
    bool moe_slot_bank_explicit = false;
    int limit = 0;
    bool first_token_only = false;

    /* Parse flags, then the three positional args. */
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(a, "--no-int8")) {
            no_int8 = true;
        } else if (!strcmp(a, "--quality")) {
            quality = true;
            no_int8 = true;
        } else if (!strcmp(a, "--first-token-only")) {
            first_token_only = true;
        } else if (!strcmp(a, "--resident")) {
            resident = true;
        } else if (!strcmp(a, "--ssd-cache")) {
            if (i + 1 >= argc) die("--ssd-cache needs a value");
            ssd_cache = argv[++i];
        } else if (!strcmp(a, "--limit")) {
            if (i + 1 >= argc) die("--limit needs a value");
            limit = atoi(argv[++i]);
            if (limit < 0) limit = 0;
        } else if (!strcmp(a, "--ctx")) {
            if (i + 1 >= argc) die("--ctx needs a value");
            ctx_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--moe-slot-bank")) {
            if (i + 1 >= argc) die("--moe-slot-bank needs a value");
            moe_slot_bank = atoi(argv[++i]);
            moe_slot_bank_explicit = true;
        } else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 2;
        } else {
            switch (pos) {
            case 0: model_path = a; break;
            case 1: manifest_path = a; break;
            case 2: out_path = a; break;
            default:
                fprintf(stderr, "too many positional args\n");
                usage(argv[0]);
                return 2;
            }
            pos++;
        }
    }

    if (!model_path || !manifest_path || !out_path) {
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
        .warm_weights = false,
        .quality = quality,
        .no_int8 = no_int8,
        .ssd_cache = ssd_cache,
        .moe_slot_bank = moe_slot_bank,
        .moe_slot_bank_explicit = moe_slot_bank_explicit,
        .ctx_size = ctx_size,
        .resident = resident,
    };

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) die("failed to create session");

    FILE *mf = fopen(manifest_path, "rb");
    if (!mf) {
        fprintf(stderr, "open %s: %s\n", manifest_path, strerror(errno));
        return 1;
    }
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "open %s: %s\n", out_path, strerror(errno));
        return 1;
    }
    fprintf(out, "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\n");

    char line[8192];
    int case_n = 0;
    double total_nll = 0.0;
    long total_tokens = 0;
    long total_lcp = 0;
    long first_matches = 0;
    char err[256];

    while (fgets(line, sizeof(line), mf)) {
        strip_newline(line);
        if (!line[0] || line[0] == '#') continue;

        char *id = strtok(line, "\t");
        char *prompt_path = strtok(NULL, "\t");
        char *cont_path = strtok(NULL, "\t");
        if (!id || !prompt_path || !cont_path) die("bad manifest row");

        char *prompt_text = read_file(prompt_path);
        char *cont_text = read_file(cont_path);

        ds4_tokens prompt = {0};
        ds4_tokens target = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);
        ds4_tokenize_text(engine, cont_text, &target);

        if (prompt.len + target.len + 1 >= ctx_size) {
            fprintf(stderr, "%s exceeds ctx=%d\n", id, ctx_size);
            return 1;
        }
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s sync failed: %s\n", id, err);
            return 1;
        }

        double nll = 0.0;
        int lcp = 0;
        bool still_matching = true;
        bool first_match = false;
        const int score_tokens =
            first_token_only && target.len > 0 ? 1 : target.len;
        for (int i = 0; i < score_tokens; i++) {
            const int greedy = ds4_session_argmax(session);
            if (i == 0) first_match = (greedy == target.v[i]);
            if (still_matching && greedy == target.v[i]) lcp++;
            else still_matching = false;

            ds4_token_score score;
            if (!ds4_session_token_logprob(session, target.v[i], &score)) {
                fprintf(stderr, "%s logprob failed at target token %d\n", id, i);
                return 1;
            }
            nll += -(double)score.logprob;

            if (ds4_session_eval(session, target.v[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "%s eval failed at target token %d: %s\n", id, i, err);
                return 1;
            }
        }

        const double avg = score_tokens ? nll / (double)score_tokens : 0.0;
        fprintf(out, "%s\t%d\t%d\t%.9f\t%.9f\t%d\t%d\n",
                id, prompt.len, score_tokens, nll, avg, first_match ? 1 : 0, lcp);
        fflush(out);

        case_n++;
        total_nll += nll;
        total_tokens += score_tokens;
        total_lcp += lcp;
        first_matches += first_match ? 1 : 0;
        fprintf(stderr,
                "%s cases=%d prompt=%d target=%d avg_nll=%.6f lcp=%d\n",
                id, case_n, prompt.len, score_tokens, avg, lcp);

        ds4_tokens_free(&prompt);
        ds4_tokens_free(&target);
        free(prompt_text);
        free(cont_text);

        if (limit > 0 && case_n >= limit) break;
    }

    fprintf(stderr,
            "summary cases=%d tokens=%ld avg_nll=%.9f first_match=%ld avg_lcp=%.3f\n",
            case_n,
            total_tokens,
            total_tokens ? total_nll / (double)total_tokens : 0.0,
            first_matches,
            case_n ? (double)total_lcp / (double)case_n : 0.0);

    fclose(out);
    fclose(mf);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
