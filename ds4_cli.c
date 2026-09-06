#include "ds4.h"
#include "ds4_profile.h"
#include "linenoise.h"

/* ds4 CLI.
 *
 * One-shot mode builds a single DeepSeek chat prompt and exits.  Interactive
 * mode keeps a rendered token transcript plus one ds4_session, so follow-up
 * turns reuse the live Metal KV checkpoint just like the server does.  The CLI
 * deliberately keeps policy here and leaves graph/cache mechanics inside the
 * engine API. */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *prompt;
    const char *system;
    int n_predict;
    int ctx_size;
    bool ctx_explicit;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
    bool hy3_q8;
} cli_config;

static volatile sig_atomic_t cli_interrupted;
static double cli_process_start_t;
static double cli_engine_open_start_t;
static double cli_engine_open_end_t;
static double cli_prompt_build_seconds;
static double cli_first_emit_t;
static bool cli_first_emit_recorded;

static double cli_now_sec(void);

static void cli_sigint_handler(int sig) {
    (void)sig;
    cli_interrupted = 1;
}

static bool cli_interrupt_requested(void) {
    return cli_interrupted != 0;
}

static void cli_interrupt_clear(void) {
    cli_interrupted = 0;
}

static void cli_note_first_emit(void) {
    if (!cli_first_emit_recorded) {
        cli_first_emit_t = cli_now_sec();
        cli_first_emit_recorded = true;
    }
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n"
        "\n"
        "Invocation modes:\n"
        "  ds4\n"
        "      Start the interactive chat prompt with a session backend: ds4>\n"
        "  ds4 -p TEXT\n"
        "      Run one prompt and exit.\n"
        "  ds4 --prompt-file FILE\n"
        "      Run one prompt read from FILE and exit. Useful for long prompts.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "      A sidecar package directory is also accepted when it contains\n"
        "      manifest.json and dense/model-dense.gguf.\n"
        "  --hy3-q8\n"
        "      HY3 only: use Q8_0 KV instead of the default F16 NAX-half cache.\n"
        "  --mtp FILE\n"
        "      Optional MTP support GGUF used for draft-token probes.\n"
        "  --mtp-draft N\n"
        "      Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F\n"
        "      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n"
        "  --draft dspark\n"
        "      Use a DSpark draft package for speculative decoding. Greedy only.\n"
        "  --draft-path PATH\n"
        "      DS4 DSpark draft package directory, e.g. /Users/anemll/Models/DSv4-Flash-DSpark-draft.\n"
        "  --draft-verify N\n"
        "      Fixed DSpark verification budget, 1..5. Default: 5\n"
        "  --draft-verify-dynamic\n"
        "      Adapt active DSpark verify budget from 2 up to --draft-verify based on recent accepted tokens.\n"
        "  --draft-mode strict|batch|unified\n"
        "      DSpark verifier contract. strict keeps no-draft greedy compatibility; batch/unified are experimental.\n"
        "  --draft-scheduler static|confidence|confidence-cost|rate|confidence-softmax|confidence-softmax-long\n"
        "      DSpark verification scheduler. confidence truncates the draft prefix at the\n"
        "      learned confidence-head threshold; confidence-cost scores the confidence\n"
        "      head against measured per-budget verify-cost EMAs. Default: confidence\n"
        "  --draft-conf-threshold F\n"
        "      DSpark confidence threshold for the confidence scheduler. Default: 0.4\n"
        "  --dspark-attn-force-mma\n"
        "      Diagnostic: force DSpark verifier MMA attention path.\n"
        "  --draft-fast-relaxed\n"
        "      Non-byte DSpark speed preset: frontier draft, relaxed suffix accept, MMA attention,\n"
        "      and fast routed-down Q2. Intended for diagnostics/demo A/B only.\n"
        "  --moe-sidecar PATH\n"
        "      Flash-MoE sidecar directory containing manifest.json and expert records.\n"
        "  --moe-mode NAME\n"
        "      Routed expert weight source: off or slot-bank. Default: off\n"
        "  --moe-slot-bank N\n"
        "      Streaming slots per layer; main RAM/cache knob. Default: 32\n"
        "      Higher caches more experts; lower uses less RAM.\n"
        "  --resident\n"
        "      Flash-MoE sidecar resident mode: load every expert into a mixed\n"
        "      slot bank, request Metal residency, touch pages, and preload all\n"
        "      routed layers. Implies --moe-mode slot-bank and defaults slots\n"
        "      per layer to the model expert count.\n"
        "  --ssd-cache BYTES|auto\n"
        "      Size the Flash-MoE slot bank from a cache budget such as 25GB.\n"
        "      auto uses available memory minus dense weights and context buffers,\n"
        "      then assigns DS4_SSD_CACHE_AUTO_PCT%% (default 20) of the remainder\n"
        "      to the slot bank, leaving headroom for the OS file cache that\n"
        "      serves decode-miss reads of the sidecar.\n"
        "  --moe-expert-topk N\n"
        "      Experimental: route/stream only N experts per token instead of\n"
        "      the model default. Applies to both prefill and decode.\n"
        "  -c, --ctx N\n"
        "      Context size allocated for the session. Default: 32768\n"
        "  --metal\n"
        "      Use the Metal graph backend. This is the normal fast path on macOS.\n"
        "  --cuda\n"
        "      Use the CUDA graph backend. This is the normal fast path on CUDA builds.\n"
        "  --cpu\n"
        "      Use the CPU reference/debug backend. Not recommended for normal inference.\n"
        "  --backend NAME\n"
        "      Select backend explicitly: metal, cuda, or cpu.\n"
        "  -t, --threads N\n"
        "      CPU helper threads for host-side or reference work.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; implies --no-int8.\n"
        "  --no-int8\n"
        "      Disable int8 accelerator paths; use NAX-half/GPU fallbacks for quality-preserving runs.\n"
        "  --ane\n"
        "      Enable ANE prefill profile defaults (off by default: the async ANE i8 arm\n"
        "      is lower precision and non-reproducible run to run).\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n"
        "\n"
        "Prompt and generation:\n"
        "  -p, --prompt TEXT\n"
        "      Prompt to generate from.\n"
        "  --prompt-file FILE\n"
        "      Read the prompt text from FILE.\n"
        "  -sys, --system TEXT\n"
        "      System prompt. Empty string disables the default. Default: You are a helpful assistant\n"
        "  -n, --tokens N\n"
        "      Maximum tokens to generate. Default: 50000\n"
        "  --temp F\n"
        "      Sampling temperature. 0 is greedy/deterministic. Default: 1\n"
        "  --top-p F\n"
        "      Nucleus sampling probability. Default: 1\n"
        "  --min-p F\n"
        "      Keep tokens scoring at least F times the top token. Default: 0.05\n"
        "  --seed N\n"
        "      Sampling seed for reproducible non-greedy runs. Default: time-based\n"
        "  --think\n"
        "      Use normal thinking mode. This is the default.\n"
        "  --think-max\n"
        "      Use Think Max when --ctx is at least 393216 tokens; otherwise normal thinking.\n"
        "  --nothink\n"
        "      Start assistant turns with </think> for direct non-thinking replies.\n"
        "\n"
        "Interactive commands:\n"
        "  /help\n"
        "      Show interactive commands.\n"
        "  /think, /think-max, /nothink\n"
        "      Select normal thinking, context-gated Think Max, or non-thinking mode.\n"
        "  /ctx N\n"
        "      Recreate the interactive session with a new context size.\n"
        "  /read FILE\n"
        "      Read a prompt from FILE and run it as the next user message.\n"
        "  /quit, /exit\n"
        "      Leave the interactive prompt.\n"
        "  Ctrl+C\n"
        "      Stop the current generation and return to ds4> without exiting.\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect\n"
        "      Load the model and print a summary only.\n"
        "  --dump-tokens\n"
        "      Print the rendered prompt tokens, then exit without inference.\n"
        "  --dump-logprobs FILE\n"
        "      Write greedy continuation top-logprobs as JSON without printing text.\n"
        "  --logprobs-top-k N\n"
        "      Number of local alternatives stored by --dump-logprobs. Default: 20\n"
        "  --imatrix-dataset FILE\n"
        "      Rendered DS4 prompt dataset produced by misc/imatrix_dataset.\n"
        "  --imatrix-out FILE\n"
        "      Collect a routed-MoE activation imatrix and write llama-compatible .dat.\n"
        "  --imatrix-max-prompts N\n"
        "      Stop imatrix collection after N prompts. Default: no prompt limit\n"
        "  --imatrix-max-tokens N\n"
        "      Stop imatrix collection after N prompt tokens. Default: no token limit\n"
        "  --head-test\n"
        "      Run the output HC/logits head after the native slice.\n"
        "  --first-token-test\n"
        "      Run an exact CPU whole-model pass for the first prompt token.\n"
        "  --metal-graph-test\n"
        "      Compare first GPU-resident graph stages with CPU.\n"
        "  --metal-graph-full-test\n"
        "      Run the GPU-resident self-token graph across all layers.\n"
        "  --metal-graph-prompt-test\n"
        "      Compare CPU and GPU graph logits for the full prompt.\n"
        "\n"
        "Normal CLI commands:\n"
        "  ./ds4\n"
        "  ./ds4 -p \"Scrivi una storia su una papera scansafatiche\"\n"
        "  ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n"
        "\n"
        "Notes:\n"
        "  The CLI keeps KV cache state across interactive turns on session backends.\n"
        "  CPU mode supports interactive chat too, but it is a slow reference/debug path.\n"
        "  Long added input is processed with batched prefill; short continuations use decode.\n"
        "  Startup prints the extra context-buffer memory for the selected context size.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_int_range(const char *s, const char *opt, int min, int max) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4: invalid backend: %s\n", s);
    fprintf(stderr, "ds4: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static ds4_moe_mode parse_moe_mode(const char *s) {
    if (!strcmp(s, "off")) return DS4_MOE_MODE_OFF;
    if (!strcmp(s, "slot-bank")) return DS4_MOE_MODE_SLOT_BANK;
    fprintf(stderr, "ds4: invalid MoE mode: %s\n", s);
    fprintf(stderr, "ds4: valid MoE modes are: off, slot-bank\n");
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    char grow[96] = "";
    if (m.ctx_grow && m.comp_cap_max > m.comp_cap) {
        snprintf(grow, sizeof(grow), "/%u max, grow_block=%u",
                 m.comp_cap_max, m.ctx_grow_block);
    }
    fprintf(stderr,
            "ds4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u%s)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap,
            grow);
}

static ds4_think_mode cli_effective_think_mode(const cli_generation_options *gen) {
    return ds4_think_mode_for_context(gen->think_mode, gen->ctx_size);
}

static bool cli_think_max_downgraded(const cli_generation_options *gen) {
    return gen->think_mode == DS4_THINK_MAX &&
           cli_effective_think_mode(gen) != DS4_THINK_MAX;
}

static void cli_warn_think_max_downgraded(const cli_generation_options *gen, const char *name) {
    if (!cli_think_max_downgraded(gen)) return;
    ds4_log(stderr,
        DS4_LOG_WARNING,
        "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
        name,
        ds4_think_max_min_context(),
        gen->ctx_size);
}

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static double cli_gib(uint64_t bytes) {
    return (double)bytes / 1073741824.0;
}

static double cli_env_positive_double(const char *name) {
    const char *s = getenv(name);
    if (!s || !*s) return 0.0;
    char *end = NULL;
    const double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 0.0;
    return v;
}

static void cli_log_runtime_line(bool styled, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (styled) {
        char buf[1024];
        vsnprintf(buf, sizeof(buf), fmt, ap);
        ds4_log(stderr, DS4_LOG_TIMING, "%s", buf);
    } else {
        vfprintf(stderr, fmt, ap);
    }
    va_end(ap);
}

static void cli_print_runtime_status(ds4_session *session, bool styled) {
    ds4_runtime_status rt;
    memset(&rt, 0, sizeof(rt));
    if (ds4_session_runtime_status(session, &rt) == 0 || !rt.available) return;

    if (rt.dspark_perf_drafted_tokens != 0 && rt.dspark_perf_blocks != 0) {
        const double draft_tps = rt.dspark_perf_draft_seconds > 0.0 ?
            (double)rt.dspark_perf_drafted_tokens / rt.dspark_perf_draft_seconds : 0.0;
        const double verify_prop_tps = rt.dspark_perf_verify_seconds > 0.0 ?
            (double)rt.dspark_perf_drafted_tokens / rt.dspark_perf_verify_seconds : 0.0;
        const double verify_accept_tps = rt.dspark_perf_verify_seconds > 0.0 ?
            (double)rt.dspark_perf_committed_tokens / rt.dspark_perf_verify_seconds : 0.0;
        const double avg_block_ms = rt.dspark_perf_total_seconds > 0.0 ?
            1000.0 * rt.dspark_perf_total_seconds / (double)rt.dspark_perf_blocks : 0.0;
        const double avg_draft_ms =
            1000.0 * rt.dspark_perf_draft_seconds / (double)rt.dspark_perf_blocks;
        const double avg_verify_ms =
            1000.0 * rt.dspark_perf_verify_seconds / (double)rt.dspark_perf_blocks;
        const double avg_commit_ms =
            1000.0 * rt.dspark_perf_commit_seconds / (double)rt.dspark_perf_blocks;
        double avg_overhead_ms =
            avg_block_ms - avg_draft_ms - avg_verify_ms;
        if (avg_overhead_ms < 0.0) avg_overhead_ms = 0.0;
        const double tau =
            (double)rt.dspark_perf_committed_tokens / (double)rt.dspark_perf_blocks;

        cli_log_runtime_line(styled,
                "ds4: dspark perf: draft=%.1f tok/s, verify=%.1f proposed tok/s, "
                "verify-accepted=%.1f tok/s, block=%.2f ms "
                "(draft=%.2f verify=%.2f overhead=%.2f commit=%.2f, "
                "tau=%.2f, blocks=%llu, skip-pre=%llu, skip-verify=%llu)\n",
                draft_tps,
                verify_prop_tps,
                verify_accept_tps,
                avg_block_ms,
                avg_draft_ms,
                avg_verify_ms,
                avg_overhead_ms,
                avg_commit_ms,
                tau,
                (unsigned long long)rt.dspark_perf_blocks,
                (unsigned long long)rt.dspark_perf_skip_pre_draft,
                (unsigned long long)rt.dspark_perf_skip_verify);

        const double avg_verify_gpu_ms =
            1000.0 * rt.dspark_perf_verify_gpu_seconds / (double)rt.dspark_perf_blocks;
        const double verify_gpu_pct = avg_verify_ms > 0.0 ?
            100.0 * avg_verify_gpu_ms / avg_verify_ms : 0.0;
        cli_log_runtime_line(styled,
                "ds4: dspark verify GPU-busy: %.2f ms/block of %.2f ms verify "
                "(%.1f%% active, %.1f%% idle)\n",
                avg_verify_gpu_ms,
                avg_verify_ms,
                verify_gpu_pct,
                100.0 - verify_gpu_pct);

        const double block_tps = rt.dspark_perf_total_seconds > 0.0 ?
            (double)rt.dspark_perf_committed_tokens / rt.dspark_perf_total_seconds : 0.0;
        const double draft_hidden_seconds =
            rt.dspark_perf_total_seconds - rt.dspark_perf_draft_seconds;
        if (draft_hidden_seconds > 0.0 && block_tps > 0.0) {
            const double draft_hidden_tps =
                (double)rt.dspark_perf_committed_tokens / draft_hidden_seconds;
            cli_log_runtime_line(styled,
                    "ds4: dspark draft-overlap upper-bound: %.1f tok/s "
                    "(%.2fx block decode, hides %.2f ms/block)\n",
                    draft_hidden_tps,
                    draft_hidden_tps / block_tps,
                    avg_draft_ms);
        }

        double baseline_decode_ms = cli_env_positive_double("DS4_DSPARK_BASELINE_DECODE_MS");
        const double baseline_tps = cli_env_positive_double("DS4_DSPARK_BASELINE_TPS");
        if (baseline_decode_ms <= 0.0 && baseline_tps > 0.0) {
            baseline_decode_ms = 1000.0 / baseline_tps;
        }
        if (baseline_decode_ms > 0.0) {
            cli_log_runtime_line(styled,
                    "ds4: dspark decode-eq: draft=%.2f verify=%.2f "
                    "overhead=%.2f block=%.2f baseline_decode=%.2f ms tau=%.2f\n",
                    avg_draft_ms / baseline_decode_ms,
                    avg_verify_ms / baseline_decode_ms,
                    avg_overhead_ms / baseline_decode_ms,
                    avg_block_ms / baseline_decode_ms,
                    baseline_decode_ms,
                    tau);
        }
    }

    if (rt.system_memory_total_bytes != 0 ||
        rt.system_compressor_bytes != 0 ||
        rt.swap_total_bytes != 0 ||
        rt.gpu_compressed_bytes != 0) {
        cli_log_runtime_line(styled,
                "ds4: vm pressure: app=%.2f GiB resident=%.2f GiB wired=%.2f GiB "
                "compressed=%.2f GiB logical=%.2f GiB swap=%.2f/%.2f GiB "
                "pressure=%u%% free=%.2f GiB gpu=%.2f GiB "
                "gpu-compressed=%.2f GiB task-compressed=%.2f GiB "
                "decompressions=%llu\n",
                cli_gib(rt.phys_footprint_bytes),
                cli_gib(rt.resident_bytes),
                cli_gib(rt.system_memory_wired_bytes),
                cli_gib(rt.system_compressor_bytes),
                cli_gib(rt.system_compressed_bytes),
                cli_gib(rt.swap_used_bytes),
                cli_gib(rt.swap_total_bytes),
                rt.system_memory_pressure_pct,
                cli_gib(rt.system_memory_free_bytes),
                cli_gib(rt.gpu_footprint_bytes),
                cli_gib(rt.gpu_compressed_bytes),
                cli_gib(rt.task_compressed_bytes),
                (unsigned long long)rt.decompressions);
    }
}

static char *read_prompt_file(const char *path, bool fatal);

typedef struct {
    int base_tokens;
    int input_tokens;
    int last_processed;
    double start_t;
    double last_t;
    bool use_color;
} cli_prefill_progress;

static void cli_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    cli_prefill_progress *p = ud;
    if (!p || !event || p->input_tokens <= 0) return;
    if (strcmp(event, "prefill_chunk") &&
        strcmp(event, "prefill_display") &&
        strcmp(event, "prefill_display_ane")) return;

    int processed = current - p->base_tokens;
    if (processed < 0) processed = 0;
    if (processed > p->input_tokens) processed = p->input_tokens;
    double pct = 100.0 * (double)processed / (double)p->input_tokens;
    if (pct > 100.0) pct = 100.0;

    const double now = cli_now_sec();
    const double total_dt = now - p->start_t;
    const double avg_tps = total_dt > 0.0 ? (double)processed / total_dt : 0.0;
    int batch_tokens = processed - p->last_processed;
    if (batch_tokens < 0) batch_tokens = 0;
    const double batch_dt = now - p->last_t;
    const double batch_tps = batch_dt > 0.0 ? (double)batch_tokens / batch_dt : 0.0;
    p->last_processed = processed;
    p->last_t = now;

    if (p->use_color) {
        fputc('\r', stderr);
        ds4_log(stderr,
                DS4_LOG_PREFILL,
                "processing %d input tokens: %d/%d (%.1f%%) batch=%.2f t/s avg=%.2f t/s",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct,
                batch_tps,
                avg_tps);
        fputs("\x1b[K", stderr);
        if (processed >= p->input_tokens) fputc('\n', stderr);
    } else {
        fprintf(stderr,
                "processing %d input tokens: %d/%d (%.1f%%) batch=%.2f t/s avg=%.2f t/s\n",
                p->input_tokens,
                processed,
                p->input_tokens,
                pct,
                batch_tps,
                avg_tps);
    }
    fflush(stderr);
}

static bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->in_think) token_printer_set_grey(p);
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        const size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            p->in_think = false;
            token_printer_reset_color(p);
            if (!p->last_output_newline) {
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_finish(token_printer *p) {
    if (p->format_thinking) {
        token_printer_process(p, NULL, 0, true);
        token_printer_reset_color(p);
    }
    fflush(p->fp);
}

static void generation_done(void *ud) {
    token_printer *p = ud;
    token_printer_finish(p);
    if (!p->last_output_newline) {
        fputc('\n', p->fp);
        p->last_output_newline = true;
    }
    fflush(p->fp);
}

static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    if (p->format_thinking) {
        token_printer_process(p, text, len, false);
    } else if (len) {
        fwrite(text, 1, len, p->fp);
        p->last_output_newline = text[len - 1] == '\n';
    }
}

typedef struct {
    token_printer *printer;
    char pending[128];
    size_t pending_len;
} cli_glm_stop_filter;

static const char *cli_glm_stop_text(size_t i) {
    static const char *stops[] = {
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|observation|>",
        "<sop>",
        "[gMASK]",
    };
    return i < sizeof(stops) / sizeof(stops[0]) ? stops[i] : NULL;
}

static bool cli_glm_stop_find(const char *text, size_t len, size_t *off_out) {
    for (size_t off = 0; off < len; off++) {
        for (size_t i = 0; ; i++) {
            const char *s = cli_glm_stop_text(i);
            if (!s) break;
            const size_t n = strlen(s);
            if (off + n <= len && memcmp(text + off, s, n) == 0) {
                if (off_out) *off_out = off;
                return true;
            }
        }
    }
    return false;
}

static size_t cli_glm_stop_suffix_prefix_len(const char *text, size_t len) {
    size_t best = 0;
    for (size_t off = 0; off < len; off++) {
        const size_t suffix_len = len - off;
        for (size_t i = 0; ; i++) {
            const char *s = cli_glm_stop_text(i);
            if (!s) break;
            const size_t n = strlen(s);
            if (suffix_len < n && suffix_len > best &&
                memcmp(text + off, s, suffix_len) == 0) {
                best = suffix_len;
            }
        }
    }
    return best;
}

static void cli_glm_stop_filter_init(cli_glm_stop_filter *f, token_printer *printer) {
    memset(f, 0, sizeof(*f));
    f->printer = printer;
}

static void cli_glm_stop_filter_flush(cli_glm_stop_filter *f) {
    if (f->pending_len) {
        token_printer_write_text(f->printer, f->pending, f->pending_len);
        f->pending_len = 0;
    }
}

static bool cli_glm_stop_filter_write(cli_glm_stop_filter *f, const char *text, size_t len) {
    if (!len) return false;
    if (len >= sizeof(f->pending) || f->pending_len + len >= sizeof(f->pending)) {
        cli_glm_stop_filter_flush(f);
    }
    if (len < sizeof(f->pending) && f->pending_len + len < sizeof(f->pending)) {
        memcpy(f->pending + f->pending_len, text, len);
        f->pending_len += len;
    } else {
        token_printer_write_text(f->printer, text, len);
        return false;
    }

    while (f->pending_len) {
        size_t stop_off = 0;
        if (cli_glm_stop_find(f->pending, f->pending_len, &stop_off)) {
            if (stop_off) token_printer_write_text(f->printer, f->pending, stop_off);
            f->pending_len = 0;
            return true;
        }
        const size_t keep = cli_glm_stop_suffix_prefix_len(f->pending, f->pending_len);
        const size_t safe = f->pending_len - keep;
        if (safe == 0) return false;
        token_printer_write_text(f->printer, f->pending, safe);
        if (keep) memmove(f->pending, f->pending + safe, keep);
        f->pending_len = keep;
        if (keep) return false;
    }
    return false;
}

static void print_generated_token(void *ud, int token) {
    token_printer *p = ud;
    size_t len = 0;
    char *text = ds4_token_text(p->engine, token, &len);
    cli_note_first_emit();
    token_printer_write_text(p, text, len);
    fflush(p->fp);
    free(text);
}

static void build_prompt(ds4_engine *engine, const cli_generation_options *gen, ds4_tokens *out) {
    if (is_rendered_chat_prompt(gen->prompt)) {
        ds4_tokenize_rendered_chat(engine, gen->prompt, out);
    } else {
        ds4_encode_chat_prompt(engine, gen->system, gen->prompt,
                               cli_effective_think_mode(gen), out);
    }
}

static int run_sampled_generation(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    const double t_session0 = cli_now_sec();
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: sampled CLI generation requires a session backend\n");
        return 1;
    }
    const double t_session1 = cli_now_sec();

    char err[160];
    ds4_think_mode think_mode = cli_effective_think_mode(&cfg->gen);
    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };
    cli_glm_stop_filter stop_filter;
    cli_glm_stop_filter_init(&stop_filter, &printer);
    const double t_prefill0 = cli_now_sec();
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .last_processed = 0,
        .start_t = t_prefill0,
        .last_t = t_prefill0,
        .use_color = ds4_log_is_tty(stderr),
    };

    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    uint64_t draft_slots = 0;
    uint64_t draft_accepted = 0;
    uint64_t draft_blocks = 0;
    uint64_t draft_pos_slots[16] = {0};
    uint64_t draft_pos_cond_slots[16] = {0};
    uint64_t draft_pos_accepted[16] = {0};
    uint64_t draft_full_accept_blocks = 0;
    uint64_t draft_first_miss_blocks = 0;
    const bool dspark_draft_enabled = ds4_engine_dspark_draft_tokens(engine) > 0;
    const bool mtp_draft_enabled = ds4_engine_mtp_draft_tokens(engine) > 1;
    const char *draft_label = dspark_draft_enabled ? "dspark" : "mtp";
    const bool draft_spec_available =
        (dspark_draft_enabled && getenv("DS4_DSPARK_SPEC_DISABLE") == NULL) ||
        (mtp_draft_enabled && getenv("DS4_MTP_SPEC_DISABLE") == NULL);
    if (draft_spec_available && cfg->gen.temperature > 0.0f) {
        fprintf(stderr,
                "ds4: %s draft loaded but speculative decode is disabled "
                "because --temp %.6g > 0; use --temp 0 for DSpark/MTP draft\n",
                draft_label,
                (double)cfg->gen.temperature);
    }
    bool stopped = false;
    const double t_decode0 = cli_now_sec();
    const bool progress_1k = getenv("DS4_PROGRESS_1K") != NULL;
    int progress_next = 1000;
    int progress_last_tokens = 0;
    double progress_last_t = t_decode0;
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(session, cfg->gen.temperature, 0,
                                       cfg->gen.top_p, cfg->gen.min_p, &rng);
        if (token == ds4_token_eos(engine) ||
            token == ds4_token_user(engine) ||
            token == ds4_token_assistant(engine)) {
            stopped = true;
            break;
        }

        int toks[17];
        int ntok = 0;
        const bool use_speculative =
            cfg->gen.temperature <= 0.0f &&
            ((dspark_draft_enabled && getenv("DS4_DSPARK_SPEC_DISABLE") == NULL) ||
             (mtp_draft_enabled && getenv("DS4_MTP_SPEC_DISABLE") == NULL));
        if (use_speculative) {
            const int accepted_cap = (int)(sizeof(toks) / sizeof(toks[0]));
            int drafted = 0;
            int accepted_draft_count = -1;
            ntok = ds4_session_eval_speculative_argmax(session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       accepted_cap,
                                                       &drafted,
                                                       &accepted_draft_count,
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            int accepted_drafts = accepted_draft_count >= 0 ?
                accepted_draft_count : (ntok > 1 ? ntok - 1 : 0);
            if (accepted_drafts < 0) accepted_drafts = 0;
            draft_slots += (uint64_t)drafted;
            if (drafted > 0) {
                draft_blocks++;
                const int pos_cap = drafted < 16 ? drafted : 16;
                for (int i = 0; i < pos_cap; i++) draft_pos_slots[i]++;
                if (accepted_drafts > drafted) accepted_drafts = drafted;
                const int accepted_cap_pos =
                    accepted_drafts < pos_cap ? accepted_drafts : pos_cap;
                for (int i = 0; i < pos_cap; i++) {
                    if (i == 0 || accepted_drafts >= i) draft_pos_cond_slots[i]++;
                }
                for (int i = 0; i < accepted_cap_pos; i++) draft_pos_accepted[i]++;
                if (accepted_drafts == drafted) draft_full_accept_blocks++;
                if (accepted_drafts == 0) draft_first_miss_blocks++;
            }
            if (accepted_drafts > 0) draft_accepted += (uint64_t)accepted_drafts;
        } else {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                ds4_session_free(session);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine) ||
                toks[j] == ds4_token_user(engine) ||
                toks[j] == ds4_token_assistant(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            cli_note_first_emit();
            stop = cli_glm_stop_filter_write(&stop_filter, piece, piece_len);
            free(piece);
            if (stop) break;
            fflush(stdout);
            generated++;
            if (progress_1k) {
                while (generated >= progress_next) {
                    const double now = cli_now_sec();
                    const int window_tokens = progress_next - progress_last_tokens;
                    const double window_s = now - progress_last_t;
                    const double total_s = now - t_decode0;
                    fprintf(stderr,
                            "ds4: decode-progress: tokens=%d window=%.2f t/s "
                            "(%d tokens in %.3fs) cumulative=%.2f t/s elapsed=%.3fs\n",
                            progress_next,
                            window_s > 0.0 ? (double)window_tokens / window_s : 0.0,
                            window_tokens,
                            window_s,
                            total_s > 0.0 ? (double)progress_next / total_s : 0.0,
                            total_s);
                    progress_last_tokens = progress_next;
                    progress_last_t = now;
                    progress_next += 1000;
                }
            }
            if (generated >= max_tokens) break;
        }
        if (stop) {
            stopped = true;
            break;
        }
    }
    if (!stopped) cli_glm_stop_filter_flush(&stop_filter);
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);
    if (cli_interrupt_requested()) cli_interrupt_clear();

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    fprintf(stderr, "ds4: ----------------------------------------\n");
    fprintf(stderr,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s (%d tokens in %.3fs)\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0,
            generated,
            decode_s);
    if (cli_process_start_t > 0.0 && cli_engine_open_end_t >= cli_engine_open_start_t) {
        const double total_s = t_decode1 - cli_process_start_t;
        const double engine_s = cli_engine_open_end_t - cli_engine_open_start_t;
        const double session_s = t_session1 - t_session0;
        const double first_s =
            cli_first_emit_recorded ? (cli_first_emit_t - cli_process_start_t) : 0.0;
        fprintf(stderr,
                "ds4: ttf: first-output=%.3fs total=%.3fs "
                "(engine=%.3fs session=%.3fs prompt=%.3fs prefill=%.3fs decode=%.3fs)\n",
                first_s,
                total_s,
                engine_s,
                session_s,
                cli_prompt_build_seconds,
                prefill_s,
                decode_s);
    }
    if (getenv("DS4_AGENT_ALLOW_BACKEND_STATS") ||
        getenv("DS4_DSPARK_PERF") ||
        getenv("DS4_DSPARK_BLOCK_TIMING") ||
        getenv("DS4_DSPARK_TIMING")) {
        cli_print_runtime_status(session, false);
    }
    if (draft_slots > 0) {
        const double draft_acceptance =
            100.0 * (double)draft_accepted / (double)draft_slots;
        fprintf(stderr,
                "ds4: %s acceptance: %.1f%% (%llu/%llu draft tokens)\n",
                draft_label,
                draft_acceptance,
                (unsigned long long)draft_accepted,
                (unsigned long long)draft_slots);
        int pos_limit = 0;
        for (int i = 0; i < 16; i++) {
            if (draft_pos_slots[i] != 0) pos_limit = i + 1;
        }
        if (pos_limit > 0) {
            fprintf(stderr, "ds4: %s acceptance by position:", draft_label);
            for (int i = 0; i < pos_limit; i++) {
                fprintf(stderr,
                        " %d=%llu/%llu",
                        i + 1,
                        (unsigned long long)draft_pos_accepted[i],
                        (unsigned long long)draft_pos_slots[i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "ds4: %s conditional acceptance:", draft_label);
            for (int i = 0; i < pos_limit; i++) {
                const uint64_t denom = draft_pos_cond_slots[i];
                const uint64_t numer = draft_pos_accepted[i];
                const double pct = denom != 0 ?
                    100.0 * (double)numer / (double)denom : 0.0;
                if (i == 0) {
                    fprintf(stderr,
                            " 1=%.1f%% (%llu/%llu)",
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                } else if (i == 1) {
                    fprintf(stderr,
                            " 2|1=%.1f%% (%llu/%llu)",
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                } else {
                    fprintf(stderr,
                            " %d|1-%d=%.1f%% (%llu/%llu)",
                            i + 1,
                            i,
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                }
            }
            fprintf(stderr, "\n");
            fprintf(stderr,
                    "ds4: %s full-accept: %.1f%% (%llu/%llu blocks), first-miss: %.1f%% (%llu/%llu blocks)\n",
                    draft_label,
                    draft_blocks > 0 ?
                        100.0 * (double)draft_full_accept_blocks / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_full_accept_blocks,
                    (unsigned long long)draft_blocks,
                    draft_blocks > 0 ?
                        100.0 * (double)draft_first_miss_blocks / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_first_miss_blocks,
                    (unsigned long long)draft_blocks);
            fprintf(stderr,
                    "ds4: %s avg scheduled: %.2f draft tokens/block (%llu blocks)\n",
                    draft_label,
                    draft_blocks > 0 ? (double)draft_slots / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_blocks);
        }
    }

    ds4_session_free(session);
    return 0;
}

static bool json_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void json_write_string(FILE *fp, const char *s, size_t n) {
    bool valid_utf8 = json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            /* Tokenizer pieces can be arbitrary byte fragments.  The bytes
             * array is authoritative; this escape keeps the JSON valid. */
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    json_write_string(fp, text, n);
    fputs(",\"bytes\":[", fp);
    for (size_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int run_logprob_dump(ds4_engine *engine, const cli_config *cfg, const ds4_tokens *prompt) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4: --dump-logprobs requires a graph session backend\n");
        return 1;
    }

    char err[160];
    const double progress_t0 = cli_now_sec();
    cli_prefill_progress progress = {
        .base_tokens = 0,
        .input_tokens = prompt->len,
        .last_processed = 0,
        .start_t = progress_t0,
        .last_t = progress_t0,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(session);
        return 1;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);

    FILE *fp = fopen(cfg->gen.dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        ds4_session_free(session);
        return 1;
    }

    int k = cfg->gen.dump_logprobs_top_k > 0 ? cfg->gen.dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_token_score *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        ds4_session_free(session);
        return 1;
    }

    fprintf(fp, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n",
            prompt->len, cfg->gen.ctx_size, k);
    int generated = 0;
    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(session) - ds4_session_pos(session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    for (; generated < max_tokens; generated++) {
        int n = ds4_session_top_logprobs(session, scores, k);
        int token = ds4_session_argmax(session);
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        json_write_token(fp, engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n && scores[i].id >= 0; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            json_write_token(fp, engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed while dumping logprobs: %s\n", err);
            free(scores);
            fclose(fp);
            ds4_session_free(session);
            return 1;
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close --dump-logprobs file: %s\n", cfg->gen.dump_logprobs_path);
        free(scores);
        ds4_session_free(session);
        return 1;
    }
    free(scores);
    ds4_session_free(session);
    return 0;
}

static int run_generation(ds4_engine *engine, const cli_config *cfg) {
    ds4_tokens prompt = {0};
    const double t_prompt0 = cli_now_sec();
    build_prompt(engine, &cfg->gen, &prompt);
    cli_prompt_build_seconds = cli_now_sec() - t_prompt0;

    int rc = 0;
    if (cfg->gen.metal_graph_test) {
        rc = ds4_engine_metal_graph_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_full_test) {
        rc = ds4_engine_metal_graph_full_test(engine, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.metal_graph_prompt_test) {
        rc = ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg->gen.ctx_size);
        ds4_tokens_free(&prompt);
        return rc;
    }
    if (cfg->gen.dump_logprobs_path) {
        rc = run_logprob_dump(engine, cfg, &prompt);
        ds4_tokens_free(&prompt);
        return rc;
    }

    const bool diagnostic = cfg->gen.dump_tokens ||
                            cfg->gen.head_test ||
                            cfg->gen.first_token_test;
    if (cfg->gen.head_test) {
        rc = ds4_engine_head_test(engine, &prompt);
    }
    if (rc == 0 && cfg->gen.first_token_test) {
        rc = ds4_engine_first_token_test(engine, &prompt);
    }
    if (cfg->gen.dump_tokens) {
        ds4_engine_dump_tokens(engine, &prompt);
    }

    if (diagnostic) {
        if (rc == 0) {
            fprintf(stderr, "ds4: diagnostic run completed on the native %s path.\n",
                    ds4_backend_name(cfg->engine.backend));
        }
    } else if (cfg->engine.moe_mode == DS4_MOE_MODE_SLOT_BANK ||
               cfg->gen.temperature > 0.0f ||
               ds4_engine_dspark_draft_tokens(engine) > 0 ||
               ds4_engine_mtp_draft_tokens(engine) > 1) {
        rc = run_sampled_generation(engine, cfg, &prompt);
    } else {
        token_printer printer = {
            .engine = engine,
            .fp = stdout,
            .format_thinking = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .in_think = ds4_think_mode_enabled(cli_effective_think_mode(&cfg->gen)),
            .use_color = isatty(fileno(stdout)) != 0,
            .last_output_newline = true,
        };
        const double progress_t0 = cli_now_sec();
        cli_prefill_progress progress = {
            .base_tokens = 0,
            .input_tokens = prompt.len,
            .last_processed = 0,
            .start_t = progress_t0,
            .last_t = progress_t0,
            .use_color = ds4_log_is_tty(stderr),
        };
        rc = ds4_engine_generate_argmax(engine, &prompt, cfg->gen.n_predict,
                                        cfg->gen.ctx_size,
                                        print_generated_token,
                                        generation_done,
                                        &printer,
                                        cli_prefill_progress_cb,
                                        &progress);
    }

    ds4_tokens_free(&prompt);
    return rc;
}

static char *trim_inplace(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void print_repl_help(void) {
    puts("Commands:");
    puts("  /help          Show this help.");
    puts("  /think         Use normal thinking mode.");
    puts("  /think-max     Use Think Max only when context is at least 393216 tokens.");
    puts("  /nothink       Disable thinking mode.");
    puts("  /ctx N         Set context size for following prompts.");
    puts("  /read FILE     Read a prompt from FILE and run it.");
    puts("  /quit, /exit   Leave the prompt.");
    puts("  Ctrl+C         Stop generation and return to the prompt.");
}

static void history_file_path(char *buf, size_t len) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(buf, len, "%s/.ds4_history", home);
}

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    int max_prefix_tokens;
} repl_chat;

static void tokens_insert(ds4_tokens *dst, int pos, const ds4_tokens *src) {
    if (!src || src->len <= 0) return;
    if (pos < 0) pos = 0;
    if (pos > dst->len) pos = dst->len;
    while (dst->len + src->len > dst->cap) {
        dst->cap = dst->cap ? dst->cap * 2 : 64;
        int *next = realloc(dst->v, (size_t)dst->cap * sizeof(dst->v[0]));
        if (!next) {
            perror("ds4: realloc");
            exit(1);
        }
        dst->v = next;
    }
    memmove(dst->v + pos + src->len, dst->v + pos,
            (size_t)(dst->len - pos) * sizeof(dst->v[0]));
    memcpy(dst->v + pos, src->v, (size_t)src->len * sizeof(src->v[0]));
    dst->len += src->len;
}

static void tokens_remove(ds4_tokens *dst, int pos, int n) {
    if (n <= 0 || pos < 0 || pos >= dst->len) return;
    if (pos + n > dst->len) n = dst->len - pos;
    memmove(dst->v + pos, dst->v + pos + n,
            (size_t)(dst->len - pos - n) * sizeof(dst->v[0]));
    dst->len -= n;
}

/* Insert/remove the Think Max prefix inside the existing transcript.  The
 * prefix lives after BOS, before any system/developer text, which mirrors the
 * API rendering path.  Changing it invalidates the session because every later
 * token position would otherwise refer to the wrong prefix. */
static void repl_chat_apply_max_prefix(ds4_engine *engine, repl_chat *chat, bool enable) {
    if (enable && chat->max_prefix_tokens == 0) {
        ds4_tokens prefix = {0};
        ds4_chat_append_max_effort_prefix(engine, &prefix);
        tokens_insert(&chat->transcript, 1, &prefix);
        chat->max_prefix_tokens = prefix.len;
        ds4_tokens_free(&prefix);
        if (chat->session) ds4_session_invalidate(chat->session);
    } else if (!enable && chat->max_prefix_tokens > 0) {
        tokens_remove(&chat->transcript, 1, chat->max_prefix_tokens);
        chat->max_prefix_tokens = 0;
        if (chat->session) ds4_session_invalidate(chat->session);
    }
}

static int repl_chat_create_session(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "ds4: interactive chat KV cache requires a session backend\n");
        return 1;
    }
    if (chat->session) ds4_session_free(chat->session);
    chat->session = session;
    chat->ctx_size = ctx_size;
    return 0;
}

static int repl_chat_init(ds4_engine *engine, repl_chat *chat, const cli_config *cfg) {
    memset(chat, 0, sizeof(*chat));
    ds4_chat_begin(engine, &chat->transcript);
    repl_chat_apply_max_prefix(engine, chat,
                               cli_effective_think_mode(&cfg->gen) == DS4_THINK_MAX);
    if (cfg->gen.system && cfg->gen.system[0]) {
        ds4_chat_append_message(engine, &chat->transcript, "system", cfg->gen.system);
    }
    return repl_chat_create_session(engine, chat, cfg->gen.ctx_size);
}

static void repl_chat_free(repl_chat *chat) {
    if (!chat) return;
    ds4_session_free(chat->session);
    ds4_tokens_free(&chat->transcript);
    memset(chat, 0, sizeof(*chat));
}

static int repl_chat_set_ctx(ds4_engine *engine, repl_chat *chat, int ctx_size) {
    ds4_session_free(chat->session);
    chat->session = NULL;
    chat->ctx_size = 0;
    return repl_chat_create_session(engine, chat, ctx_size);
}

/* Run one interactive turn.  The transcript is tentatively extended with user
 * and assistant markers, then ds4_session_sync() decides whether this is a KV
 * continuation.  If prompt processing fails, the transcript rolls back before
 * returning to the prompt. */
static int run_chat_turn(ds4_engine *engine, cli_config *cfg, repl_chat *chat, const char *user_text) {
    if (!chat->session) {
        fprintf(stderr, "ds4: no active interactive KV cache\n");
        return 1;
    }

    ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                           chat->ctx_size);
    repl_chat_apply_max_prefix(engine, chat, think_mode == DS4_THINK_MAX);
    const int rollback_len = chat->transcript.len;
    ds4_chat_append_message(engine, &chat->transcript, "user", user_text);
    ds4_chat_append_assistant_prefix(engine, &chat->transcript, think_mode);

    const int old_pos = ds4_session_pos(chat->session);
    const int common = ds4_session_common_prefix(chat->session, &chat->transcript);
    const int cached = common == old_pos && chat->transcript.len >= old_pos ? common : 0;
    const int suffix = chat->transcript.len - cached;

    char err[160];
    const double t_prefill0 = cli_now_sec();
    cli_prefill_progress progress = {
        .base_tokens = cached,
        .input_tokens = suffix,
        .last_processed = 0,
        .start_t = t_prefill0,
        .last_t = t_prefill0,
        .use_color = ds4_log_is_tty(stderr),
    };
    ds4_session_set_progress(chat->session, cli_prefill_progress_cb, &progress);
    ds4_session_set_display_progress(chat->session, cli_prefill_progress_cb, &progress);
    if (ds4_session_sync(chat->session, &chat->transcript, err, sizeof(err)) != 0) {
        ds4_session_set_progress(chat->session, NULL, NULL);
        ds4_session_set_display_progress(chat->session, NULL, NULL);
        chat->transcript.len = rollback_len;
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        return 1;
    }
    ds4_session_set_progress(chat->session, NULL, NULL);
    ds4_session_set_display_progress(chat->session, NULL, NULL);
    const double t_prefill1 = cli_now_sec();

    token_printer printer = {
        .engine = engine,
        .fp = stdout,
        .format_thinking = ds4_think_mode_enabled(think_mode),
        .in_think = ds4_think_mode_enabled(think_mode),
        .use_color = isatty(fileno(stdout)) != 0,
        .last_output_newline = true,
    };

    int max_tokens = cfg->gen.n_predict;
    int room = ds4_session_ctx(chat->session) - ds4_session_pos(chat->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    int generated = 0;
    uint64_t draft_slots = 0;
    uint64_t draft_accepted = 0;
    uint64_t draft_blocks = 0;
    uint64_t draft_pos_slots[16] = {0};
    uint64_t draft_pos_cond_slots[16] = {0};
    uint64_t draft_pos_accepted[16] = {0};
    uint64_t draft_full_accept_blocks = 0;
    uint64_t draft_first_miss_blocks = 0;
    const bool dspark_draft_enabled = ds4_engine_dspark_draft_tokens(engine) > 0;
    const bool mtp_draft_enabled = ds4_engine_mtp_draft_tokens(engine) > 1;
    const char *draft_label = dspark_draft_enabled ? "dspark" : "mtp";
    const bool draft_spec_available =
        (dspark_draft_enabled && getenv("DS4_DSPARK_SPEC_DISABLE") == NULL) ||
        (mtp_draft_enabled && getenv("DS4_MTP_SPEC_DISABLE") == NULL);
    if (draft_spec_available && cfg->gen.temperature > 0.0f) {
        fprintf(stderr,
                "ds4: %s draft loaded but speculative decode is disabled "
                "because --temp %.6g > 0; use --temp 0 for DSpark/MTP draft\n",
                draft_label,
                (double)cfg->gen.temperature);
    }
    const double t_decode0 = cli_now_sec();
    while (generated < max_tokens && !cli_interrupt_requested()) {
        int token = ds4_session_sample(chat->session,
                                       cfg->gen.temperature,
                                       0,
                                       cfg->gen.top_p,
                                       cfg->gen.min_p,
                                       &rng);
        if (token == ds4_token_eos(engine)) break;

        int toks[17];
        int ntok = 0;
        const bool use_speculative =
            cfg->gen.temperature <= 0.0f &&
            ((dspark_draft_enabled && getenv("DS4_DSPARK_SPEC_DISABLE") == NULL) ||
             (mtp_draft_enabled && getenv("DS4_MTP_SPEC_DISABLE") == NULL));
        if (use_speculative) {
            const int accepted_cap = (int)(sizeof(toks) / sizeof(toks[0]));
            int drafted = 0;
            int accepted_draft_count = -1;
            ntok = ds4_session_eval_speculative_argmax(chat->session,
                                                       token,
                                                       max_tokens - generated,
                                                       ds4_token_eos(engine),
                                                       toks,
                                                       accepted_cap,
                                                       &drafted,
                                                       &accepted_draft_count,
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
            int accepted_drafts = accepted_draft_count >= 0 ?
                accepted_draft_count : (ntok > 1 ? ntok - 1 : 0);
            if (accepted_drafts < 0) accepted_drafts = 0;
            draft_slots += (uint64_t)drafted;
            if (drafted > 0) {
                draft_blocks++;
                const int pos_cap = drafted < 16 ? drafted : 16;
                for (int i = 0; i < pos_cap; i++) draft_pos_slots[i]++;
                if (accepted_drafts > drafted) accepted_drafts = drafted;
                const int accepted_cap_pos =
                    accepted_drafts < pos_cap ? accepted_drafts : pos_cap;
                for (int i = 0; i < pos_cap; i++) {
                    if (i == 0 || accepted_drafts >= i) draft_pos_cond_slots[i]++;
                }
                for (int i = 0; i < accepted_cap_pos; i++) draft_pos_accepted[i]++;
                if (accepted_drafts == drafted) draft_full_accept_blocks++;
                if (accepted_drafts == 0) draft_first_miss_blocks++;
            }
            if (accepted_drafts > 0) draft_accepted += (uint64_t)accepted_drafts;
        } else {
            if (ds4_session_eval(chat->session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: decode failed: %s\n", err);
                return 1;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop = false;
        for (int j = 0; j < ntok; j++) {
            if (toks[j] == ds4_token_eos(engine)) {
                stop = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, toks[j], &piece_len);
            ds4_tokens_push(&chat->transcript, toks[j]);
            token_printer_write_text(&printer, piece, piece_len);
            fflush(stdout);
            free(piece);
            generated++;
            if (generated >= max_tokens) break;
        }
        if (stop) break;
    }
    const double t_decode1 = cli_now_sec();
    generation_done(&printer);

    const bool interrupted = cli_interrupt_requested();
    if (interrupted && generated == 0) {
        chat->transcript.len = rollback_len;
        ds4_session_invalidate(chat->session);
    } else {
        ds4_tokens_push(&chat->transcript, ds4_token_eos(engine));
    }

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    if (interrupted) cli_interrupt_clear();
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: ----------------------------------------\n");
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)suffix / prefill_s : 0.0,
            decode_s > 0.0 ? (double)generated / decode_s : 0.0);
    cli_print_runtime_status(chat->session, true);
    if (draft_slots > 0) {
        const double draft_acceptance =
            100.0 * (double)draft_accepted / (double)draft_slots;
        ds4_log(stderr,
                DS4_LOG_TIMING,
                "ds4: %s acceptance: %.1f%% (%llu/%llu draft tokens)\n",
                draft_label,
                draft_acceptance,
                (unsigned long long)draft_accepted,
                (unsigned long long)draft_slots);
        int pos_limit = 0;
        for (int i = 0; i < 16; i++) {
            if (draft_pos_slots[i] != 0) pos_limit = i + 1;
        }
        if (pos_limit > 0) {
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: %s acceptance by position:",
                    draft_label);
            for (int i = 0; i < pos_limit; i++) {
                ds4_log(stderr,
                        DS4_LOG_TIMING,
                        " %d=%llu/%llu",
                        i + 1,
                        (unsigned long long)draft_pos_accepted[i],
                        (unsigned long long)draft_pos_slots[i]);
            }
            ds4_log(stderr, DS4_LOG_TIMING, "\n");
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: %s conditional acceptance:",
                    draft_label);
            for (int i = 0; i < pos_limit; i++) {
                const uint64_t denom = draft_pos_cond_slots[i];
                const uint64_t numer = draft_pos_accepted[i];
                const double pct = denom != 0 ?
                    100.0 * (double)numer / (double)denom : 0.0;
                if (i == 0) {
                    ds4_log(stderr,
                            DS4_LOG_TIMING,
                            " 1=%.1f%% (%llu/%llu)",
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                } else if (i == 1) {
                    ds4_log(stderr,
                            DS4_LOG_TIMING,
                            " 2|1=%.1f%% (%llu/%llu)",
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                } else {
                    ds4_log(stderr,
                            DS4_LOG_TIMING,
                            " %d|1-%d=%.1f%% (%llu/%llu)",
                            i + 1,
                            i,
                            pct,
                            (unsigned long long)numer,
                            (unsigned long long)denom);
                }
            }
            ds4_log(stderr, DS4_LOG_TIMING, "\n");
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: %s full-accept: %.1f%% (%llu/%llu blocks), first-miss: %.1f%% (%llu/%llu blocks)\n",
                    draft_label,
                    draft_blocks > 0 ?
                        100.0 * (double)draft_full_accept_blocks / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_full_accept_blocks,
                    (unsigned long long)draft_blocks,
                    draft_blocks > 0 ?
                        100.0 * (double)draft_first_miss_blocks / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_first_miss_blocks,
                    (unsigned long long)draft_blocks);
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: %s avg scheduled: %.2f draft tokens/block (%llu blocks)\n",
                    draft_label,
                    draft_blocks > 0 ? (double)draft_slots / (double)draft_blocks : 0.0,
                    (unsigned long long)draft_blocks);
        }
    }
    return 0;
}

static int run_repl(ds4_engine *engine, cli_config *cfg) {
    repl_chat chat;
    if (repl_chat_init(engine, &chat, cfg) != 0) return 1;

    struct sigaction old_int;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = cli_sigint_handler;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    cli_interrupt_clear();

    char hist[PATH_MAX];
    history_file_path(hist, sizeof(hist));
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    linenoiseHistoryLoad(hist);
    print_repl_help();

    int rc = 0;
    for (;;) {
        errno = 0;
        char *line = linenoise("ds4> ");
        if (!line) {
            if (errno == EAGAIN || cli_interrupt_requested()) {
                cli_interrupt_clear();
                continue;
            }
            break;
        }
        char *cmd = trim_inplace(line);
        if (!cmd[0]) {
            linenoiseFree(line);
            continue;
        }
        linenoiseHistoryAdd(cmd);
        linenoiseHistorySave(hist);

        if (!strcmp(cmd, "/help")) {
            print_repl_help();
        } else if (!strcmp(cmd, "/think")) {
            cfg->gen.think_mode = DS4_THINK_HIGH;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: high.");
        } else if (!strcmp(cmd, "/think-max")) {
            cfg->gen.think_mode = DS4_THINK_MAX;
            bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                     chat.ctx_size) == DS4_THINK_MAX;
            repl_chat_apply_max_prefix(engine, &chat, active);
            cli_warn_think_max_downgraded(&cfg->gen, "/think-max");
            printf("Thinking mode: %s.\n", active ? "max" : "high (ctx below 393216)");
        } else if (!strcmp(cmd, "/nothink")) {
            cfg->gen.think_mode = DS4_THINK_NONE;
            repl_chat_apply_max_prefix(engine, &chat, false);
            puts("Thinking mode: none.");
        } else if (!strncmp(cmd, "/ctx", 4) && (cmd[4] == '\0' || isspace((unsigned char)cmd[4]))) {
            char *arg = trim_inplace(cmd + 4);
            if (!arg[0]) {
                fprintf(stderr, "ds4: /ctx needs a positive integer\n");
            } else {
                cfg->gen.ctx_size = parse_int(arg, "/ctx");
                log_context_memory(cfg->engine.backend, cfg->gen.ctx_size);
                rc = repl_chat_set_ctx(engine, &chat, cfg->gen.ctx_size);
                if (rc != 0) {
                    linenoiseFree(line);
                    break;
                }
                bool active = ds4_think_mode_for_context(cfg->gen.think_mode,
                                                         chat.ctx_size) == DS4_THINK_MAX;
                repl_chat_apply_max_prefix(engine, &chat, active);
                cli_warn_think_max_downgraded(&cfg->gen, "/ctx");
            }
        } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            linenoiseFree(line);
            break;
        } else if (!strncmp(cmd, "/read", 5) && (cmd[5] == '\0' || isspace((unsigned char)cmd[5]))) {
            char *path = trim_inplace(cmd + 5);
            if (!path[0]) {
                fprintf(stderr, "ds4: /read needs a file path\n");
            } else {
                char *prompt = read_prompt_file(path, false);
                if (prompt) {
                    rc = run_chat_turn(engine, cfg, &chat, prompt);
                    free(prompt);
                }
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "ds4: unknown command: %s\n", cmd);
            fprintf(stderr, "ds4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, cfg, &chat, cmd);
        }
        linenoiseFree(line);
    }
    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    repl_chat_free(&chat);
    return rc;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void cli_setenv_or_die(const char *name, const char *value, int overwrite) {
    if (setenv(name, value, overwrite) != 0) {
        fprintf(stderr, "ds4: setenv %s: %s\n", name, strerror(errno));
        exit(2);
    }
}

static void cli_unsetenv_or_die(const char *name) {
    if (unsetenv(name) != 0) {
        fprintf(stderr, "ds4: unsetenv %s: %s\n", name, strerror(errno));
        exit(2);
    }
}

static void cli_enable_dspark_fast_relaxed(void) {
    cli_unsetenv_or_die("DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS");
    cli_setenv_or_die("DS4_DSPARK_FRONTIER_DRAFT", "1", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_ACCEPT", "1", 1);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_MARGIN_DISABLE", "1", 1);
    cli_setenv_or_die("DS4_DSPARK_ATTN_FORCE_MMA", "1", 1);
    cli_setenv_or_die("DS4_DSPARK_HYBRID_ROW_ROUTED_FAST_Q2_SUM6", "1", 1);
    cli_setenv_or_die("DS4_METAL_ENABLE_ROUTED_DOWN_SUM6", "1", 1);
    cli_setenv_or_die("DS4_DSPARK_DRAFT_PREFETCH", "1", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_ALLOW_TARGET_TOP_REPEAT", "1", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_TOPK", "256", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_LOGIT_DELTA", "10", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK", "5", 0);
    cli_setenv_or_die("DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX", "24", 0);
}

static ds4_draft_kind parse_draft_kind(const char *s) {
    if (!strcmp(s, "none") || !strcmp(s, "off")) return DS4_DRAFT_NONE;
    if (!strcmp(s, "dspark")) return DS4_DRAFT_DSPARK;
    fprintf(stderr, "ds4: invalid --draft value: %s (expected none or dspark)\n", s);
    exit(2);
}

static const char *parse_draft_scheduler(const char *s) {
    if (!strcmp(s, "static") ||
        !strcmp(s, "confidence") ||
        !strcmp(s, "confidence-cost") ||
        !strcmp(s, "cost") ||
        !strcmp(s, "rate") ||
        !strcmp(s, "confidence-rate") ||
        !strcmp(s, "confidence-softmax") ||
        !strcmp(s, "softmax") ||
        !strcmp(s, "confidence-softmax-long") ||
        !strcmp(s, "softmax-long")) return s;
    fprintf(stderr,
            "ds4: invalid --draft-scheduler value: %s "
            "(expected static, confidence, confidence-cost, rate, confidence-softmax, or confidence-softmax-long)\n",
            s);
    exit(2);
}

static const char *parse_draft_mode(const char *s) {
    if (!strcmp(s, "strict") || !strcmp(s, "batch") ||
        !strcmp(s, "unified") || !strcmp(s, "unified_greedy")) return s;
    fprintf(stderr, "ds4: invalid --draft-mode value: %s (expected strict, batch, or unified)\n", s);
    exit(2);
}

static char *read_prompt_file(const char *path, bool fatal) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open prompt file: %s\n", path);
        if (fatal) exit(2);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4: failed to seek prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fprintf(stderr, "ds4: failed to size prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    rewind(fp);

    char *buf = malloc((size_t)len + 1);
    if (!buf) {
        fprintf(stderr, "ds4: out of memory reading prompt file: %s\n", path);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    size_t nread = fread(buf, 1, (size_t)len, fp);
    if (nread != (size_t)len) {
        fprintf(stderr, "ds4: failed to read prompt file: %s\n", path);
        free(buf);
        fclose(fp);
        if (fatal) exit(2);
        return NULL;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close prompt file: %s\n", path);
        free(buf);
        if (fatal) exit(2);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
            .draft_kind = DS4_DRAFT_NONE,
            .draft_verify = 5,
            .draft_scheduler = "confidence",
            .draft_conf_threshold = 0.0f,
            .moe_mode = DS4_MOE_MODE_OFF,
            .moe_slot_bank = 32,
        },
        .gen = {
            .prompt = NULL,
            .system = "You are a helpful assistant",
            .n_predict = 50000,
            .ctx_size = 32768,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .dump_logprobs_top_k = 20,
            .think_mode = DS4_THINK_HIGH,
        },
    };

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.gen.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.gen.prompt) {
                fprintf(stderr, "ds4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_prompt_file(need_arg(&i, argc, argv, arg), true);
            c.gen.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--hy3-q8")) {
            c.hy3_q8 = true;
            cli_setenv_or_die("DS4_HY3_NAX_HALF_ATTN", "0", 1);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--draft")) {
            c.engine.draft_kind = parse_draft_kind(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--draft-path")) {
            c.engine.draft_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--draft-verify")) {
            c.engine.draft_verify = parse_int_range(need_arg(&i, argc, argv, arg), arg, 1, 5);
        } else if (!strcmp(arg, "--draft-verify-dynamic")) {
            if (setenv("DS4_DSPARK_VERIFY_DYNAMIC", "1", 1) != 0) {
                fprintf(stderr, "ds4: setenv DS4_DSPARK_VERIFY_DYNAMIC: %s\n", strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "--draft-mode")) {
            const char *mode = parse_draft_mode(need_arg(&i, argc, argv, arg));
            if (!strcmp(mode, "batch")) {
                if (setenv("DS4_DSPARK_VERIFY_CANONICAL", "batch", 1) != 0) {
                    fprintf(stderr, "ds4: setenv DS4_DSPARK_VERIFY_CANONICAL: %s\n", strerror(errno));
                    exit(2);
                }
            } else if (!strcmp(mode, "unified") || !strcmp(mode, "unified_greedy")) {
                if (setenv("DS4_DSPARK_VERIFY_CANONICAL", "unified", 1) != 0) {
                    fprintf(stderr, "ds4: setenv DS4_DSPARK_VERIFY_CANONICAL: %s\n", strerror(errno));
                    exit(2);
                }
                if (setenv("DS4_TARGET_FORWARD_UNIFIED", "1", 1) != 0) {
                    fprintf(stderr, "ds4: setenv DS4_TARGET_FORWARD_UNIFIED: %s\n", strerror(errno));
                    exit(2);
                }
            } else {
                if (setenv("DS4_DSPARK_VERIFY_CANONICAL", "strict", 1) != 0) {
                    fprintf(stderr, "ds4: setenv DS4_DSPARK_VERIFY_CANONICAL: %s\n", strerror(errno));
                    exit(2);
                }
            }
        } else if (!strcmp(arg, "--draft-scheduler")) {
            c.engine.draft_scheduler = parse_draft_scheduler(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--draft-conf-threshold")) {
            c.engine.draft_conf_threshold = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--dspark-attn-force-mma")) {
            cli_unsetenv_or_die("DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS");
            cli_setenv_or_die("DS4_DSPARK_ATTN_FORCE_MMA", "1", 1);
        } else if (!strcmp(arg, "--draft-fast-relaxed")) {
            cli_enable_dspark_fast_relaxed();
        } else if (!strcmp(arg, "--moe-sidecar")) {
            c.engine.moe_sidecar_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--moe-mode")) {
            c.engine.moe_mode = parse_moe_mode(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--moe-slot-bank")) {
            c.engine.moe_slot_bank = parse_int(need_arg(&i, argc, argv, arg), arg);
            c.engine.moe_slot_bank_explicit = true;
        } else if (!strcmp(arg, "--resident")) {
            c.engine.resident = true;
        } else if (!strcmp(arg, "--ssd-cache")) {
            c.engine.ssd_cache = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--moe-expert-topk")) {
            int topk = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (topk < 1) topk = 1;
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", topk);
            if (setenv("DS4_MOE_EXPERT_TOPK", buf, 1) != 0) {
                fprintf(stderr, "ds4: setenv DS4_MOE_EXPERT_TOPK: %s\n", strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
            c.gen.ctx_explicit = true;
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
            c.engine.no_int8 = true;
            if (setenv("DS4_NO_INT8", "1", 1) != 0) {
                fprintf(stderr, "ds4: setenv DS4_NO_INT8: %s\n", strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "--no-int8")) {
            c.engine.no_int8 = true;
            if (setenv("DS4_NO_INT8", "1", 1) != 0) {
                fprintf(stderr, "ds4: setenv DS4_NO_INT8: %s\n", strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "--ane")) {
            if (setenv("DS4_ANE", "1", 1) != 0) {
                fprintf(stderr, "ds4: setenv DS4_ANE: %s\n", strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.gen.dump_tokens = true;
        } else if (!strcmp(arg, "--dump-logprobs")) {
            c.gen.dump_logprobs_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--logprobs-top-k")) {
            c.gen.dump_logprobs_top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-dataset")) {
            c.gen.imatrix_dataset_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--imatrix-out")) {
            c.gen.imatrix_output_path = need_arg(&i, argc, argv, arg);
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--imatrix-max-prompts")) {
            c.gen.imatrix_max_prompts = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--imatrix-max-tokens")) {
            c.gen.imatrix_max_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--head-test")) {
            c.gen.head_test = true;
        } else if (!strcmp(arg, "--first-token-test")) {
            c.gen.first_token_test = true;
        } else if (!strcmp(arg, "--metal-graph-test")) {
            c.gen.metal_graph_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-full-test")) {
            c.gen.metal_graph_full_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-prompt-test")) {
            c.gen.metal_graph_prompt_test = true;
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--metal-graph-generate")) {
            fprintf(stderr, "ds4: --metal-graph-generate was removed; --metal is the graph path\n");
            exit(2);
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--server")) {
            fprintf(stderr, "ds4: use ds4-server for the HTTP server\n");
            exit(2);
        } else {
            fprintf(stderr, "ds4: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    if (c.engine.draft_kind == DS4_DRAFT_DSPARK && (!c.engine.draft_path || !c.engine.draft_path[0])) {
        fprintf(stderr, "ds4: --draft dspark requires --draft-path\n");
        exit(2);
    }
    if (c.engine.draft_kind == DS4_DRAFT_NONE && c.engine.draft_path && c.engine.draft_path[0]) {
        fprintf(stderr, "ds4: --draft-path requires --draft dspark\n");
        exit(2);
    }
    c.engine.ctx_size = c.gen.ctx_size;
    ds4_engine_options_autodetect_sidecar_package(&c.engine, "ds4");
    ds4_engine_options_apply_resident_preset(&c.engine, "ds4");
    if (c.engine.resident &&
        (!c.engine.moe_sidecar_path || !c.engine.moe_sidecar_path[0])) {
        fprintf(stderr,
                "ds4: --resident requires a sidecar package directory passed to -m "
                "or an explicit --moe-sidecar\n");
        exit(2);
    }
    if (c.engine.moe_sidecar_path && c.engine.moe_mode == DS4_MOE_MODE_OFF) {
        fprintf(stderr, "ds4: --moe-sidecar requires --moe-mode slot-bank\n");
        exit(2);
    }
    if (c.gen.imatrix_output_path && !c.gen.imatrix_dataset_path) {
        fprintf(stderr, "ds4: --imatrix-out requires --imatrix-dataset\n");
        exit(2);
    }
    if (c.gen.imatrix_dataset_path && !c.gen.imatrix_output_path) {
        fprintf(stderr, "ds4: --imatrix-dataset requires --imatrix-out\n");
        exit(2);
    }

    return c;
}

int main(int argc, char **argv) {
    cli_process_start_t = cli_now_sec();
    cli_config cfg = parse_options(argc, argv);
    if (cfg.engine.mtp_path && cfg.engine.mtp_path[0] &&
        cfg.gen.temperature > 0.0f && getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
        if (setenv("DS4_MTP_SPEC_DISABLE", "1", 0) != 0) {
            fprintf(stderr, "ds4: failed to disable MTP for non-greedy decode\n");
            free(cfg.prompt_owned);
            return 1;
        }
        fprintf(stderr,
                "ds4: MTP runtime disabled before prefill because "
                "--temp %.6g > 0; use --temp 0 for MTP drafting\n",
                (double)cfg.gen.temperature);
    }
    ds4_engine_options_autodetect_sidecar_package(&cfg.engine, "ds4");
    ds4_engine_options_apply_resident_preset(&cfg.engine, "ds4");
    ds4_profile_set_sidecar_mode(cfg.engine.moe_mode == DS4_MOE_MODE_SLOT_BANK && cfg.engine.moe_sidecar_path);
    ds4_profile_load_and_apply();
    if (cfg.gen.dump_tokens && cfg.gen.prompt == NULL) {
        fprintf(stderr, "ds4: --dump-tokens requires -p or --prompt-file\n");
        free(cfg.prompt_owned);
        return 2;
    }
    if (!cfg.inspect) {
        ds4_model_shape_select_for_path(cfg.engine.model_path);
        if (!cfg.gen.ctx_explicit && ds4_model_shape_is_hy4()) {
            cfg.gen.ctx_size = cfg.engine.ctx_size = 2048;
        }
        log_context_memory(cfg.engine.backend, cfg.gen.ctx_size);
        cli_warn_think_max_downgraded(&cfg.gen, "--think-max");
    }
    ds4_engine *engine = NULL;
    cli_engine_open_start_t = cli_now_sec();
    if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        free(cfg.prompt_owned);
        return 1;
    }
    cli_engine_open_end_t = cli_now_sec();
    if (cfg.hy3_q8 && !ds4_engine_uses_hy3_tokenizer(engine)) {
        fprintf(stderr, "ds4: --hy3-q8 requires a HY3 model\n");
        ds4_engine_close(engine);
        free(cfg.prompt_owned);
        return 2;
    }
    const char *dspark_partial = getenv("DS4_DSPARK_ALLOW_PARTIAL");
    const bool dspark_partial_allowed =
        dspark_partial && dspark_partial[0] && atoi(dspark_partial) != 0;
    if (!cfg.inspect &&
        ds4_engine_has_dspark(engine) &&
        !ds4_engine_dspark_inference_ready(engine) &&
        !dspark_partial_allowed) {
        fprintf(stderr,
                "ds4: DSpark draft package validated, but draft inference kernels are not enabled yet\n");
        ds4_engine_close(engine);
        free(cfg.prompt_owned);
        return 1;
    }
    int rc = 0;
    if (cfg.inspect) {
        ds4_engine_summary(engine);
    } else if (cfg.gen.imatrix_output_path) {
        rc = ds4_engine_collect_imatrix(engine,
                                        cfg.gen.imatrix_dataset_path,
                                        cfg.gen.imatrix_output_path,
                                        cfg.gen.ctx_size,
                                        cfg.gen.imatrix_max_prompts,
                                        cfg.gen.imatrix_max_tokens);
    } else if (cfg.gen.prompt == NULL) {
        rc = run_repl(engine, &cfg);
    } else {
        rc = run_generation(engine, &cfg);
    }
    ds4_engine_close(engine);
    free(cfg.prompt_owned);
    return rc;
}
