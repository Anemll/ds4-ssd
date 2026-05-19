#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../ds4_gpu.h"

#define QK_K 256
#define DS4_METAL_TENSOR_Q2_K 10u
#define DS4_METAL_TENSOR_IQ2_XXS 16u
#define DS4_SWIGLU_CLAMP_EXP 10.0f

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} bench_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} bench_block_iq2_xxs;

#define STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
STATIC_ASSERT(bench_block_q2_k_size, sizeof(bench_block_q2_K) == 84);
STATIC_ASSERT(bench_block_iq2_xxs_size, sizeof(bench_block_iq2_xxs) == 66);

static const uint64_t bench_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

enum {
    DEFAULT_IN_DIM = 4096,
    DEFAULT_MID_DIM = 2048,
    MAX_BATCHES = 64,
};

typedef _Float16 fp16_t;

typedef enum {
    DENSE_DTYPE_FP16 = 1u << 0,
    DENSE_DTYPE_BF16 = 1u << 1,
    DENSE_DTYPE_FP32 = 1u << 2,
} dense_dtype_mask;

typedef enum {
    DENSE_FP16,
    DENSE_BF16,
    DENSE_FP32,
} dense_dtype;

typedef struct {
    int in_dim;
    int mid_dim;
    int batches[MAX_BATCHES];
    int n_batches;
    int iters;
    int warmup;
    bool run_gpu;
    bool run_amx;
    bool run_ds4;
    bool run_split;
    bool run_qhybrid;
    bool run_amx_quant;
    bool run_gpu_synth_quant;
    bool run_ane_pack;
    bool run_ane_pack_gpu;
    bool run_dequant_parts;
    bool run_dequant_batch;
    bool amxq_cached;
    bool amxq_dequant;
    bool amxq_gpu_dequant;
    bool qhybrid_cached;
    bool qhybrid_bgdequant;
    int split_amx_batch;
    uint32_t dtype_mask;
    const char *csv_path;
} bench_config;

typedef struct {
    const char *backend;
    const char *dtype;
    int batch;
    int in_dim;
    int mid_dim;
    int iters;
    double ms;
    double gflops;
} bench_result;

typedef struct {
    const char *backend;
    const char *tensor;
    const char *quant;
    const char *out_dtype;
    int in_dim;
    int mid_dim;
    int iters;
    double quant_mb;
    double dense_mb;
    double compression;
    double ms;
    double dense_gbps;
} dequant_part_result;

typedef struct {
    const char *backend;
    int experts;
    int in_dim;
    int mid_dim;
    int iters;
    double total_ms;
    double ms_per_expert;
    double output_mb;
    double output_gbps;
} dequant_batch_result;

typedef struct {
    id<MTLBuffer> buffer;
    MPSMatrix *matrix;
    NSUInteger rows;
    NSUInteger cols;
    NSUInteger row_bytes;
    NSUInteger elem_bytes;
} gpu_matrix;

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> iq2_pipe;
    id<MTLComputePipelineState> q2_pipe;
    id<MTLComputePipelineState> iq2_i8_pipe;
    id<MTLComputePipelineState> q2_i8_pipe;
    id<MTLComputePipelineState> iq2_i8_batched_pipe;
    id<MTLComputePipelineState> q2_i8_batched_pipe;
    id<MTLBuffer> grid;
    id<MTLBuffer> gate_q;
    id<MTLBuffer> up_q;
    id<MTLBuffer> down_q;
    id<MTLBuffer> w_gate;
    id<MTLBuffer> w_up;
    id<MTLBuffer> w_down;
    id<MTLBuffer> w_gate_i8;
    id<MTLBuffer> w_up_i8;
    id<MTLBuffer> w_down_i8;
    int in_dim;
    int mid_dim;
} gpu_dequant_moe;

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> x_pipe;
    id<MTLComputePipelineState> iq2_pipe;
    id<MTLComputePipelineState> q2_pipe;
    id<MTLBuffer> grid;
    id<MTLBuffer> x;
    id<MTLBuffer> gate_q;
    id<MTLBuffer> up_q;
    id<MTLBuffer> down_q;
    id<MTLBuffer> packed;
    int batch;
    int in_dim;
    int mid_dim;
    int split;
} ane_pack_gpu;

typedef struct {
    BNNSNDArrayDescriptor x;
    BNNSNDArrayDescriptor w_gate;
    BNNSNDArrayDescriptor w_up;
    BNNSNDArrayDescriptor w_down;
    BNNSNDArrayDescriptor gate;
    BNNSNDArrayDescriptor up;
    BNNSNDArrayDescriptor mid;
    BNNSNDArrayDescriptor out;
    void *gate_workspace;
    void *down_workspace;
    dense_dtype dtype;
} bnns_mlp;

typedef struct {
    const char *backend;
    const char *dtype;
    int gpu_batch;
    int amx_batch;
    int combined_batch;
    int in_dim;
    int mid_dim;
    int iters;
    double ms;
    double tokens_per_s;
    double gflops;
    double gpu_seq_ms;
    double gpu_seq_tokens_per_s;
    double speedup_vs_gpu_seq;
} split_result;

typedef struct {
    const char *backend;
    const char *dtype;
    int gpu_batch;
    int amx_batch;
    int combined_batch;
    int in_dim;
    int mid_dim;
    int iters;
    double gpu_conc_ms;
    double amx_conc_ms;
    double bg_dequant_ms;
    double ms;
    double tokens_per_s;
    double gflops;
    double gpu_seq_ms;
    double gpu_seq_tokens_per_s;
    double speedup_vs_gpu_seq;
} qhybrid_result;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void fill_iq2_bank(bench_block_iq2_xxs *blocks, size_t n_blocks, uint32_t seed);
static void fill_q2_bank(bench_block_q2_K *blocks, size_t n_blocks, uint32_t seed);

bool ds4_log_is_tty(FILE *fp) {
    return fp && isatty(fileno(fp));
}

static double mlp_gflop(int batch, int in_dim, int mid_dim) {
    /* gate, up, and down GEMMs. Activation is intentionally not counted. */
    return (6.0 * (double)batch * (double)in_dim * (double)mid_dim) / 1.0e9;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: moe-batch-bench [options]\n"
            "\n"
            "Benchmarks fixed per-expert MoE MLP batch sizes. Dense proxy modes\n"
            "use FP16/BF16/FP32 MPS/BNNS matmuls. The ds4 backend calls the repo's actual\n"
            "Metal IQ2_XXS/Q2_K routed expert block as one timed operation.\n"
            "\n"
            "Options:\n"
            "  --backend both|gpu|amx|amxq|ds4|split|qhybrid|gpuqsynthetic|anepack|anepackgpu|dequantparts|dequantbatch|all\n"
            "                          Backend set to run. Default: both\n"
            "  --dtype fp16|bf16|fp32|both|all\n"
            "                          Dense proxy dtype. Default: fp16\n"
            "  --batches LIST          Comma-separated batch sizes. Default: 1,2,4,8,16,32,64\n"
            "                          For --backend split these are GPU batch sizes.\n"
            "                          For --backend amxq these are AMX batch sizes.\n"
            "  --amxq-mode cached|dequant|gpudequant|both\n"
            "                          Quantized AMX mode for --backend amxq. Default: both\n"
            "  --qhybrid-mode cached|bgdequant|both\n"
            "                          Quantized DS4-GPU + AMX split mode. cached excludes\n"
            "                          materialization from the timed loop; bgdequant runs\n"
            "                          GPU dequant for the next AMX expert concurrently.\n"
            "  --backend gpuqsynthetic  Synthetic Metal packed-weight MLP: FP4 LUT and int8\n"
            "                          fixed-point weights expanded inside simple kernels.\n"
            "  --backend anepack        CPU materialization benchmark for split-3 ANE packed input.\n"
            "  --backend anepackgpu     Metal materialization benchmark for split-3 ANE packed input.\n"
            "  --backend dequantparts   Per-tensor CPU and Metal dequant timing.\n"
            "  --backend dequantbatch   Batched multi-expert Metal int8 dequant timing.\n"
            "  --split-amx-batch N     AMX batch for --backend split. Default: 16\n"
            "  --in N                  Input/output dimension. Default: 4096\n"
            "  --mid N                 Expert hidden dimension. Default: 2048\n"
            "  --iters N               Timed iterations per batch. Default: 20\n"
            "  --warmup N              Warmup iterations per batch. Default: 5\n"
            "  --csv FILE              Also write CSV results to FILE\n"
            "  -h, --help              Show this help\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "moe-batch-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "moe-batch-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static void parse_batches(bench_config *cfg, const char *s) {
    cfg->n_batches = 0;
    const char *p = s;
    while (*p) {
        if (cfg->n_batches >= MAX_BATCHES) {
            fprintf(stderr, "moe-batch-bench: too many batch sizes\n");
            exit(2);
        }

        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > INT32_MAX) {
            fprintf(stderr, "moe-batch-bench: invalid batch list: %s\n", s);
            exit(2);
        }
        cfg->batches[cfg->n_batches++] = (int)v;
        if (*end == '\0') break;
        if (*end != ',') {
            fprintf(stderr, "moe-batch-bench: invalid batch list: %s\n", s);
            exit(2);
        }
        p = end + 1;
    }
    if (cfg->n_batches == 0) {
        fprintf(stderr, "moe-batch-bench: empty batch list\n");
        exit(2);
    }
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .in_dim = DEFAULT_IN_DIM,
        .mid_dim = DEFAULT_MID_DIM,
        .iters = 20,
        .warmup = 5,
        .run_gpu = true,
        .run_amx = true,
        .amxq_cached = true,
        .amxq_dequant = true,
        .qhybrid_cached = true,
        .qhybrid_bgdequant = true,
        .split_amx_batch = 16,
        .dtype_mask = DENSE_DTYPE_FP16,
    };
    parse_batches(&cfg, "1,2,4,8,16,32,64");

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--backend")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "both")) {
                cfg.run_gpu = true;
                cfg.run_amx = true;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
            } else if (!strcmp(v, "gpu")) {
                cfg.run_gpu = true;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
            } else if (!strcmp(v, "amx")) {
                cfg.run_gpu = false;
                cfg.run_amx = true;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
            } else if (!strcmp(v, "amxq")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = true;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.dtype_mask = DENSE_DTYPE_BF16;
            } else if (!strcmp(v, "ds4")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = true;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
            } else if (!strcmp(v, "split")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = true;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.dtype_mask = DENSE_DTYPE_BF16;
            } else if (!strcmp(v, "qhybrid") || !strcmp(v, "hybrid")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = true;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = false;
                cfg.dtype_mask = DENSE_DTYPE_BF16;
            } else if (!strcmp(v, "gpuqsynthetic")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = true;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = false;
            } else if (!strcmp(v, "anepack")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = true;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = false;
            } else if (!strcmp(v, "anepackgpu")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = true;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = false;
            } else if (!strcmp(v, "dequantparts")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = true;
                cfg.run_dequant_batch = false;
            } else if (!strcmp(v, "dequantbatch")) {
                cfg.run_gpu = false;
                cfg.run_amx = false;
                cfg.run_ds4 = false;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = false;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = true;
            } else if (!strcmp(v, "all")) {
                cfg.run_gpu = true;
                cfg.run_amx = true;
                cfg.run_ds4 = true;
                cfg.run_split = false;
                cfg.run_qhybrid = false;
                cfg.run_amx_quant = false;
                cfg.run_gpu_synth_quant = true;
                cfg.run_ane_pack = false;
                cfg.run_ane_pack_gpu = false;
                cfg.run_dequant_parts = false;
                cfg.run_dequant_batch = false;
            } else {
                fprintf(stderr, "moe-batch-bench: invalid backend: %s\n", v);
                exit(2);
            }
        } else if (!strcmp(arg, "--dtype")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "fp16")) {
                cfg.dtype_mask = DENSE_DTYPE_FP16;
            } else if (!strcmp(v, "bf16")) {
                cfg.dtype_mask = DENSE_DTYPE_BF16;
            } else if (!strcmp(v, "fp32")) {
                cfg.dtype_mask = DENSE_DTYPE_FP32;
            } else if (!strcmp(v, "both")) {
                cfg.dtype_mask = DENSE_DTYPE_FP16 | DENSE_DTYPE_BF16;
            } else if (!strcmp(v, "all")) {
                cfg.dtype_mask = DENSE_DTYPE_FP16 | DENSE_DTYPE_BF16 | DENSE_DTYPE_FP32;
            } else {
                fprintf(stderr, "moe-batch-bench: invalid dtype: %s\n", v);
                exit(2);
            }
        } else if (!strcmp(arg, "--batches")) {
            parse_batches(&cfg, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--amxq-mode")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "cached")) {
                cfg.amxq_cached = true;
                cfg.amxq_dequant = false;
                cfg.amxq_gpu_dequant = false;
            } else if (!strcmp(v, "dequant")) {
                cfg.amxq_cached = false;
                cfg.amxq_dequant = true;
                cfg.amxq_gpu_dequant = false;
            } else if (!strcmp(v, "gpudequant")) {
                cfg.amxq_cached = false;
                cfg.amxq_dequant = false;
                cfg.amxq_gpu_dequant = true;
            } else if (!strcmp(v, "both")) {
                cfg.amxq_cached = true;
                cfg.amxq_dequant = true;
                cfg.amxq_gpu_dequant = true;
            } else {
                fprintf(stderr, "moe-batch-bench: invalid amxq mode: %s\n", v);
                exit(2);
            }
        } else if (!strcmp(arg, "--qhybrid-mode")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "cached")) {
                cfg.qhybrid_cached = true;
                cfg.qhybrid_bgdequant = false;
            } else if (!strcmp(v, "bgdequant")) {
                cfg.qhybrid_cached = false;
                cfg.qhybrid_bgdequant = true;
            } else if (!strcmp(v, "both")) {
                cfg.qhybrid_cached = true;
                cfg.qhybrid_bgdequant = true;
            } else {
                fprintf(stderr, "moe-batch-bench: invalid qhybrid mode: %s\n", v);
                exit(2);
            }
        } else if (!strcmp(arg, "--split-amx-batch")) {
            cfg.split_amx_batch = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--in")) {
            cfg.in_dim = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mid")) {
            cfg.mid_dim = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--iters")) {
            cfg.iters = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            cfg.csv_path = need_arg(&i, argc, argv, arg);
        } else {
            fprintf(stderr, "moe-batch-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    return cfg;
}

static const char *dense_dtype_name(dense_dtype dtype) {
    switch (dtype) {
    case DENSE_BF16: return "bf16";
    case DENSE_FP32: return "fp32";
    case DENSE_FP16:
    default: return "fp16";
    }
}

static BNNSDataType dense_bnns_dtype(dense_dtype dtype) {
    switch (dtype) {
    case DENSE_BF16: return BNNSDataTypeBFloat16;
    case DENSE_FP32: return BNNSDataTypeFloat32;
    case DENSE_FP16:
    default: return BNNSDataTypeFloat16;
    }
}

static MPSDataType dense_mps_dtype(dense_dtype dtype) {
    switch (dtype) {
    case DENSE_BF16: return MPSDataTypeBFloat16;
    case DENSE_FP32: return MPSDataTypeFloat32;
    case DENSE_FP16:
    default: return MPSDataTypeFloat16;
    }
}

static size_t dense_elem_bytes(dense_dtype dtype) {
    return dtype == DENSE_FP32 ? sizeof(float) : sizeof(uint16_t);
}

static uint16_t f32_to_fp16_bits(float v) {
    fp16_t h = (fp16_t)v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static float fp16_bits_to_f32(uint16_t bits) {
    fp16_t h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

static uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

static float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = (uint32_t)bits << 16;
    float v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

static uint16_t f32_to_dense16_bits(float v, dense_dtype dtype) {
    return dtype == DENSE_BF16 ? f32_to_bf16_bits(v) : f32_to_fp16_bits(v);
}

static float dense16_bits_to_f32(uint16_t bits, dense_dtype dtype) {
    return dtype == DENSE_BF16 ? bf16_bits_to_f32(bits) : fp16_bits_to_f32(bits);
}

static void *alloc_dense(size_t n, dense_dtype dtype) {
    void *p = NULL;
    const int rc = posix_memalign(&p, 64, n * dense_elem_bytes(dtype));
    if (rc != 0 || !p) {
        fprintf(stderr, "moe-batch-bench: allocation failed: %s\n", strerror(rc ? rc : ENOMEM));
        exit(1);
    }
    return p;
}

static void fill_dense(void *x, size_t n, uint32_t seed, dense_dtype dtype) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        s = 1664525u * s + 1013904223u;
        uint32_t bits = (s >> 9) | 0x3f800000u;
        float v;
        memcpy(&v, &bits, sizeof(v));
        v = (v - 1.5f) * 0.05f;
        if (dtype == DENSE_FP32) {
            ((float *)x)[i] = v;
        } else {
            ((uint16_t *)x)[i] = f32_to_dense16_bits(v, dtype);
        }
    }
}

static BNNSNDArrayDescriptor bnns_matrix(void *data, size_t rows, size_t cols, dense_dtype dtype) {
    BNNSNDArrayDescriptor d = {0};
    d.layout = BNNSDataLayoutRowMajorMatrix;
    d.size[0] = cols;
    d.size[1] = rows;
    d.stride[0] = 1;
    d.stride[1] = cols;
    d.data = data;
    d.data_type = dense_bnns_dtype(dtype);
    return d;
}

static void *bnns_workspace(
        const BNNSNDArrayDescriptor *a,
        const BNNSNDArrayDescriptor *b,
        const BNNSNDArrayDescriptor *c) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    ssize_t n = BNNSMatMulWorkspaceSize(false, false, 1.0f, a, b, c, NULL);
#pragma clang diagnostic pop
    if (n < 0) {
        fprintf(stderr, "moe-batch-bench: BNNSMatMulWorkspaceSize rejected descriptors\n");
        exit(1);
    }
    if (n == 0) return NULL;

    void *p = NULL;
    const int rc = posix_memalign(&p, 64, (size_t)n);
    if (rc != 0 || !p) {
        fprintf(stderr, "moe-batch-bench: BNNS workspace allocation failed: %s\n",
                strerror(rc ? rc : ENOMEM));
        exit(1);
    }
    return p;
}

static bnns_mlp make_bnns_mlp(
        int batch,
        int in_dim,
        int mid_dim,
        void *x,
        void *w_gate,
        void *w_up,
        void *w_down,
        void *gate,
        void *up,
        void *mid,
        void *out,
        dense_dtype dtype) {
    bnns_mlp m = {
        .x = bnns_matrix(x, (size_t)batch, (size_t)in_dim, dtype),
        .w_gate = bnns_matrix(w_gate, (size_t)in_dim, (size_t)mid_dim, dtype),
        .w_up = bnns_matrix(w_up, (size_t)in_dim, (size_t)mid_dim, dtype),
        .w_down = bnns_matrix(w_down, (size_t)mid_dim, (size_t)in_dim, dtype),
        .gate = bnns_matrix(gate, (size_t)batch, (size_t)mid_dim, dtype),
        .up = bnns_matrix(up, (size_t)batch, (size_t)mid_dim, dtype),
        .mid = bnns_matrix(mid, (size_t)batch, (size_t)mid_dim, dtype),
        .out = bnns_matrix(out, (size_t)batch, (size_t)in_dim, dtype),
        .dtype = dtype,
    };
    m.gate_workspace = bnns_workspace(&m.x, &m.w_gate, &m.gate);
    m.down_workspace = bnns_workspace(&m.mid, &m.w_down, &m.out);
    return m;
}

static void bnns_mlp_free(bnns_mlp *m) {
    free(m->gate_workspace);
    free(m->down_workspace);
    memset(m, 0, sizeof(*m));
}

static void cpu_activation_dense16(
        void *mid,
        const void *gate,
        const void *up,
        size_t n,
        dense_dtype dtype) {
    for (size_t i = 0; i < n; i++) {
        float g;
        float u;
        if (dtype == DENSE_FP32) {
            g = ((const float *)gate)[i];
            u = ((const float *)up)[i];
            ((float *)mid)[i] = (g / (1.0f + expf(-g))) * u;
        } else {
            g = dense16_bits_to_f32(((const uint16_t *)gate)[i], dtype);
            u = dense16_bits_to_f32(((const uint16_t *)up)[i], dtype);
            ((uint16_t *)mid)[i] = f32_to_dense16_bits((g / (1.0f + expf(-g))) * u, dtype);
        }
    }
}

static void bnns_mlp_once(bnns_mlp *m, int batch, int mid_dim) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	    if (BNNSMatMul(false, false, 1.0f, &m->x, &m->w_gate, &m->gate,
	                  m->gate_workspace, NULL) != 0 ||
	        BNNSMatMul(false, false, 1.0f, &m->x, &m->w_up, &m->up,
	                  m->gate_workspace, NULL) != 0) {
	        fprintf(stderr, "moe-batch-bench: BNNS %s gate/up matmul failed\n",
                    dense_dtype_name(m->dtype));
	        exit(1);
	    }

	    cpu_activation_dense16(m->mid.data,
	                           m->gate.data,
	                           m->up.data,
	                           (size_t)batch * (size_t)mid_dim,
                               m->dtype);

	    if (BNNSMatMul(false, false, 1.0f, &m->mid, &m->w_down, &m->out,
	                  m->down_workspace, NULL) != 0) {
	        fprintf(stderr, "moe-batch-bench: BNNS %s down matmul failed\n",
                    dense_dtype_name(m->dtype));
	        exit(1);
	    }
#pragma clang diagnostic pop
}

static bench_result bench_amx(const bench_config *cfg, int batch, dense_dtype dtype) {
    const int in_dim = cfg->in_dim;
    const int mid_dim = cfg->mid_dim;
    void *x = alloc_dense((size_t)batch * (size_t)in_dim, dtype);
    void *w_gate = alloc_dense((size_t)in_dim * (size_t)mid_dim, dtype);
    void *w_up = alloc_dense((size_t)in_dim * (size_t)mid_dim, dtype);
    void *w_down = alloc_dense((size_t)mid_dim * (size_t)in_dim, dtype);
    void *gate = alloc_dense((size_t)batch * (size_t)mid_dim, dtype);
    void *up = alloc_dense((size_t)batch * (size_t)mid_dim, dtype);
    void *mid = alloc_dense((size_t)batch * (size_t)mid_dim, dtype);
    void *out = alloc_dense((size_t)batch * (size_t)in_dim, dtype);

    fill_dense(x, (size_t)batch * (size_t)in_dim, 1u, dtype);
    fill_dense(w_gate, (size_t)in_dim * (size_t)mid_dim, 2u, dtype);
    fill_dense(w_up, (size_t)in_dim * (size_t)mid_dim, 3u, dtype);
    fill_dense(w_down, (size_t)mid_dim * (size_t)in_dim, 4u, dtype);

    bnns_mlp mlp = make_bnns_mlp(batch, in_dim, mid_dim,
                                 x, w_gate, w_up, w_down,
                                 gate, up, mid, out, dtype);

    for (int i = 0; i < cfg->warmup; i++) {
        bnns_mlp_once(&mlp, batch, mid_dim);
    }

    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        bnns_mlp_once(&mlp, batch, mid_dim);
    }
    const double sec = now_sec() - t0;

    bnns_mlp_free(&mlp);
    free(out);
    free(mid);
    free(up);
    free(gate);
    free(w_down);
    free(w_up);
    free(w_gate);
    free(x);

    const double ms = sec * 1000.0 / (double)cfg->iters;
    bench_result r = {
        .backend = "amx_bnns_matmul",
        .dtype = dense_dtype_name(dtype),
        .batch = batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
        .iters = cfg->iters,
        .ms = ms,
        .gflops = mlp_gflop(batch, in_dim, mid_dim) / (ms / 1000.0),
    };
    return r;
}

static NSString *activation_metal_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline float bf16_to_float(ushort x) {\n"
            "    return as_type<float>((uint)x << 16);\n"
            "}\n"
            "static inline ushort float_to_bf16(float x) {\n"
            "    uint u = as_type<uint>(x);\n"
            "    uint lsb = (u >> 16) & 1u;\n"
            "    return ushort((u + 0x7fffu + lsb) >> 16);\n"
            "}\n"
            "kernel void silu_mul_f16(device const half *gate [[buffer(0)]],\n"
            "                         device const half *up [[buffer(1)]],\n"
            "                         device half *mid [[buffer(2)]],\n"
            "                         constant uint &cols [[buffer(3)]],\n"
            "                         constant uint &row_stride [[buffer(4)]],\n"
            "                         constant uint &n [[buffer(5)]],\n"
            "                         uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= n) return;\n"
            "    uint row = gid / cols;\n"
            "    uint col = gid - row * cols;\n"
            "    uint off = row * row_stride + col;\n"
            "    half g = gate[off];\n"
            "    mid[off] = half((float(g) / (1.0f + fast::exp(-float(g)))) * float(up[off]));\n"
            "}\n"
            "kernel void silu_mul_f32(device const float *gate [[buffer(0)]],\n"
            "                         device const float *up [[buffer(1)]],\n"
            "                         device float *mid [[buffer(2)]],\n"
            "                         constant uint &cols [[buffer(3)]],\n"
            "                         constant uint &row_stride [[buffer(4)]],\n"
            "                         constant uint &n [[buffer(5)]],\n"
            "                         uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= n) return;\n"
            "    uint row = gid / cols;\n"
            "    uint col = gid - row * cols;\n"
            "    uint off = row * row_stride + col;\n"
            "    float g = gate[off];\n"
            "    mid[off] = (g / (1.0f + fast::exp(-g))) * up[off];\n"
            "}\n"
            "kernel void silu_mul_bf16(device const ushort *gate [[buffer(0)]],\n"
            "                          device const ushort *up [[buffer(1)]],\n"
            "                          device ushort *mid [[buffer(2)]],\n"
            "                          constant uint &cols [[buffer(3)]],\n"
            "                          constant uint &row_stride [[buffer(4)]],\n"
            "                          constant uint &n [[buffer(5)]],\n"
            "                          uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= n) return;\n"
            "    uint row = gid / cols;\n"
            "    uint col = gid - row * cols;\n"
            "    uint off = row * row_stride + col;\n"
            "    float g = bf16_to_float(gate[off]);\n"
            "    float u = bf16_to_float(up[off]);\n"
            "    mid[off] = float_to_bf16((g / (1.0f + fast::exp(-g))) * u);\n"
            "}\n";
}

static NSString *gpu_dequant_metal_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define QK_K 256\n"
            "struct block_q2_K {\n"
            "    uchar scales[QK_K/16];\n"
            "    uchar qs[QK_K/4];\n"
            "    half d;\n"
            "    half dmin;\n"
            "};\n"
            "struct block_iq2_xxs {\n"
            "    half d;\n"
            "    ushort qs[QK_K/8];\n"
            "};\n"
            "static inline ushort float_to_bf16(float x) {\n"
            "    uint u = as_type<uint>(x);\n"
            "    uint lsb = (u >> 16) & 1u;\n"
            "    return ushort((u + 0x7fffu + lsb) >> 16);\n"
            "}\n"
            "static inline uchar iq2_sign_mask(uint s) {\n"
            "    s &= 127u;\n"
            "    return uchar(s | ((popcount(s) & 1u) << 7));\n"
            "}\n"
            "static inline char float_to_i8_scaled(float x) {\n"
            "    int v = int(rint(clamp(x * 8.0f, -128.0f, 127.0f)));\n"
            "    return char(v);\n"
            "}\n"
            "kernel void dequant_iq2_xxs_transpose_bf16(\n"
            "        device const block_iq2_xxs *src [[buffer(0)]],\n"
            "        device const ulong *grid_table [[buffer(1)]],\n"
            "        device ushort *dst [[buffer(2)]],\n"
            "        constant uint &q_rows [[buffer(3)]],\n"
            "        constant uint &q_cols [[buffer(4)]],\n"
            "        constant uint &total [[buffer(5)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    const uint il0 = seg - b * 16u;\n"
            "    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;\n"
            "    const uint ib32 = il0 / 2u;\n"
            "    const uint lane = il0 & 1u;\n"
            "    device const ushort *q2 = blk->qs + 4u * ib32;\n"
            "    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);\n"
            "    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);\n"
            "    const uint gidx0 = (aux32_g >> (8u * (2u * lane + 0u))) & 255u;\n"
            "    const uint gidx1 = (aux32_g >> (8u * (2u * lane + 1u))) & 255u;\n"
            "    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    const ulong gv0 = grid_table[gidx0];\n"
            "    const uchar sign0 = iq2_sign_mask((aux32_s >> (14u * lane)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        dst[(col0 + j) * q_rows + r] = float_to_bf16(v);\n"
            "    }\n"
            "    const ulong gv1 = grid_table[gidx1];\n"
            "    const uchar sign1 = iq2_sign_mask((aux32_s >> (14u * lane + 7u)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        dst[(col0 + 8u + j) * q_rows + r] = float_to_bf16(v);\n"
            "    }\n"
            "}\n"
            "kernel void dequant_q2_k_transpose_bf16(\n"
            "        device const block_q2_K *src [[buffer(0)]],\n"
            "        device ushort *dst [[buffer(1)]],\n"
            "        constant uint &q_rows [[buffer(2)]],\n"
            "        constant uint &q_cols [[buffer(3)]],\n"
            "        constant uint &total [[buffer(4)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    uint il0 = seg - b * 16u;\n"
            "    device const block_q2_K *blk = src + r * blocks_per_row + b;\n"
            "    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);\n"
            "    const uchar sc = blk->scales[il0];\n"
            "    const uint il = (il0 / 2u) & 3u;\n"
            "    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);\n"
            "    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);\n"
            "    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;\n"
            "    const float ml = float(blk->dmin) * float(sc >> 4);\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    for (uint j = 0; j < 16u; j++) {\n"
            "        dst[(col0 + j) * q_rows + r] = float_to_bf16(dl * float(q[j] & mask) - ml);\n"
            "    }\n"
            "}\n"
            "kernel void dequant_iq2_xxs_transpose_i8(\n"
            "        device const block_iq2_xxs *src [[buffer(0)]],\n"
            "        device const ulong *grid_table [[buffer(1)]],\n"
            "        device char *dst [[buffer(2)]],\n"
            "        constant uint &q_rows [[buffer(3)]],\n"
            "        constant uint &q_cols [[buffer(4)]],\n"
            "        constant uint &total [[buffer(5)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    const uint il0 = seg - b * 16u;\n"
            "    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;\n"
            "    const uint ib32 = il0 / 2u;\n"
            "    const uint lane = il0 & 1u;\n"
            "    device const ushort *q2 = blk->qs + 4u * ib32;\n"
            "    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);\n"
            "    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);\n"
            "    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    const ulong gv0 = grid_table[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];\n"
            "    const uchar sign0 = iq2_sign_mask((aux32_s >> (14u * lane)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        dst[(col0 + j) * q_rows + r] = float_to_i8_scaled(v);\n"
            "    }\n"
            "    const ulong gv1 = grid_table[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];\n"
            "    const uchar sign1 = iq2_sign_mask((aux32_s >> (14u * lane + 7u)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        dst[(col0 + 8u + j) * q_rows + r] = float_to_i8_scaled(v);\n"
            "    }\n"
            "}\n"
            "kernel void dequant_q2_k_transpose_i8(\n"
            "        device const block_q2_K *src [[buffer(0)]],\n"
            "        device char *dst [[buffer(1)]],\n"
            "        constant uint &q_rows [[buffer(2)]],\n"
            "        constant uint &q_cols [[buffer(3)]],\n"
            "        constant uint &total [[buffer(4)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    uint il0 = seg - b * 16u;\n"
            "    device const block_q2_K *blk = src + r * blocks_per_row + b;\n"
            "    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);\n"
            "    const uchar sc = blk->scales[il0];\n"
            "    const uint il = (il0 / 2u) & 3u;\n"
            "    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);\n"
            "    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);\n"
            "    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;\n"
            "    const float ml = float(blk->dmin) * float(sc >> 4);\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    for (uint j = 0; j < 16u; j++) {\n"
            "        dst[(col0 + j) * q_rows + r] = float_to_i8_scaled(dl * float(q[j] & mask) - ml);\n"
            "    }\n"
            "}\n"
            "kernel void dequant_iq2_xxs_transpose_i8_batched(\n"
            "        device const block_iq2_xxs *src [[buffer(0)]],\n"
            "        device const ulong *grid_table [[buffer(1)]],\n"
            "        device char *dst [[buffer(2)]],\n"
            "        constant uint &q_rows [[buffer(3)]],\n"
            "        constant uint &q_cols [[buffer(4)]],\n"
            "        constant uint &total [[buffer(5)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    const uint per_expert = q_rows * (q_cols / QK_K) * 16u;\n"
            "    if (tid >= total) return;\n"
            "    const uint e = tid / per_expert;\n"
            "    const uint local = tid - e * per_expert;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = local / segs_per_row;\n"
            "    const uint seg = local - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    const uint il0 = seg - b * 16u;\n"
            "    device const block_iq2_xxs *blk = src + e * q_rows * blocks_per_row + r * blocks_per_row + b;\n"
            "    device char *expert_dst = dst + e * q_rows * q_cols;\n"
            "    const uint ib32 = il0 / 2u;\n"
            "    const uint lane = il0 & 1u;\n"
            "    device const ushort *q2 = blk->qs + 4u * ib32;\n"
            "    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);\n"
            "    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);\n"
            "    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    const ulong gv0 = grid_table[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];\n"
            "    const uchar sign0 = iq2_sign_mask((aux32_s >> (14u * lane)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        expert_dst[(col0 + j) * q_rows + r] = float_to_i8_scaled(v);\n"
            "    }\n"
            "    const ulong gv1 = grid_table[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];\n"
            "    const uchar sign1 = iq2_sign_mask((aux32_s >> (14u * lane + 7u)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        expert_dst[(col0 + 8u + j) * q_rows + r] = float_to_i8_scaled(v);\n"
            "    }\n"
            "}\n"
            "kernel void dequant_q2_k_transpose_i8_batched(\n"
            "        device const block_q2_K *src [[buffer(0)]],\n"
            "        device char *dst [[buffer(1)]],\n"
            "        constant uint &q_rows [[buffer(2)]],\n"
            "        constant uint &q_cols [[buffer(3)]],\n"
            "        constant uint &total [[buffer(4)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    const uint per_expert = q_rows * (q_cols / QK_K) * 16u;\n"
            "    if (tid >= total) return;\n"
            "    const uint e = tid / per_expert;\n"
            "    const uint local = tid - e * per_expert;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = local / segs_per_row;\n"
            "    const uint seg = local - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    uint il0 = seg - b * 16u;\n"
            "    device const block_q2_K *blk = src + e * q_rows * blocks_per_row + r * blocks_per_row + b;\n"
            "    device char *expert_dst = dst + e * q_rows * q_cols;\n"
            "    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);\n"
            "    const uchar sc = blk->scales[il0];\n"
            "    const uint il = (il0 / 2u) & 3u;\n"
            "    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);\n"
            "    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);\n"
            "    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;\n"
            "    const float ml = float(blk->dmin) * float(sc >> 4);\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    for (uint j = 0; j < 16u; j++) {\n"
            "        expert_dst[(col0 + j) * q_rows + r] = float_to_i8_scaled(dl * float(q[j] & mask) - ml);\n"
            "    }\n"
            "}\n"
            "kernel void pack_x_f32_to_f16(\n"
            "        device const float *src [[buffer(0)]],\n"
            "        device half *dst [[buffer(1)]],\n"
            "        constant uint &total [[buffer(2)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    dst[tid] = half(src[tid]);\n"
            "}\n"
            "kernel void pack_iq2_xxs_split3_f16(\n"
            "        device const block_iq2_xxs *src [[buffer(0)]],\n"
            "        device const ulong *grid_table [[buffer(1)]],\n"
            "        device half *dst [[buffer(2)]],\n"
            "        constant uint &q_rows [[buffer(3)]],\n"
            "        constant uint &q_cols [[buffer(4)]],\n"
            "        constant uint &split [[buffer(5)]],\n"
            "        constant uint &total [[buffer(6)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    const uint il0 = seg - b * 16u;\n"
            "    device const block_iq2_xxs *blk = src + r * blocks_per_row + b;\n"
            "    const uint ib32 = il0 / 2u;\n"
            "    const uint lane = il0 & 1u;\n"
            "    device const ushort *q2 = blk->qs + 4u * ib32;\n"
            "    const uint aux32_g = uint(q2[0]) | (uint(q2[1]) << 16);\n"
            "    const uint aux32_s = uint(q2[2]) | (uint(q2[3]) << 16);\n"
            "    const float scale = float(blk->d) * (0.5f + float(aux32_s >> 28)) * 0.25f;\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    const uint tile_cols = q_rows / split;\n"
            "    const uint tile = r / tile_cols;\n"
            "    const uint tile_col = r - tile * tile_cols;\n"
            "    device half *tile_dst = dst + tile * q_cols * tile_cols;\n"
            "    const ulong gv0 = grid_table[(aux32_g >> (8u * (2u * lane + 0u))) & 255u];\n"
            "    const uchar sign0 = iq2_sign_mask((aux32_s >> (14u * lane)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv0 >> (8u * j)) & 255ul) * ((sign0 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        tile_dst[(col0 + j) * tile_cols + tile_col] = half(v);\n"
            "    }\n"
            "    const ulong gv1 = grid_table[(aux32_g >> (8u * (2u * lane + 1u))) & 255u];\n"
            "    const uchar sign1 = iq2_sign_mask((aux32_s >> (14u * lane + 7u)) & 127u);\n"
            "    for (uint j = 0; j < 8u; j++) {\n"
            "        const float v = scale * float((gv1 >> (8u * j)) & 255ul) * ((sign1 & (1u << j)) ? -1.0f : 1.0f);\n"
            "        tile_dst[(col0 + 8u + j) * tile_cols + tile_col] = half(v);\n"
            "    }\n"
            "}\n"
            "kernel void pack_q2_k_split3_f16(\n"
            "        device const block_q2_K *src [[buffer(0)]],\n"
            "        device half *dst [[buffer(1)]],\n"
            "        constant uint &q_rows [[buffer(2)]],\n"
            "        constant uint &q_cols [[buffer(3)]],\n"
            "        constant uint &split [[buffer(4)]],\n"
            "        constant uint &total [[buffer(5)]],\n"
            "        uint tid [[thread_position_in_grid]]) {\n"
            "    if (tid >= total) return;\n"
            "    const uint blocks_per_row = q_cols / QK_K;\n"
            "    const uint segs_per_row = blocks_per_row * 16u;\n"
            "    const uint r = tid / segs_per_row;\n"
            "    const uint seg = tid - r * segs_per_row;\n"
            "    const uint b = seg / 16u;\n"
            "    uint il0 = seg - b * 16u;\n"
            "    device const block_q2_K *blk = src + r * blocks_per_row + b;\n"
            "    device const uchar *q = blk->qs + 32u * (il0 / 8u) + 16u * (il0 & 1u);\n"
            "    const uchar sc = blk->scales[il0];\n"
            "    const uint il = (il0 / 2u) & 3u;\n"
            "    const float coef = il > 1u ? (il > 2u ? 1.0f / 64.0f : 1.0f / 16.0f) : (il > 0u ? 1.0f / 4.0f : 1.0f);\n"
            "    const uchar mask = il > 1u ? (il > 2u ? 192 : 48) : (il > 0u ? 12 : 3);\n"
            "    const float dl = float(blk->d) * float(sc & 0x0fu) * coef;\n"
            "    const float ml = float(blk->dmin) * float(sc >> 4);\n"
            "    const uint col0 = b * QK_K + il0 * 16u;\n"
            "    const uint tile_rows = q_cols / split;\n"
            "    for (uint j = 0; j < 16u; j++) {\n"
            "        const uint col = col0 + j;\n"
            "        const uint tile = col / tile_rows;\n"
            "        const uint tile_row = col - tile * tile_rows;\n"
            "        device half *tile_dst = dst + tile * tile_rows * q_rows;\n"
            "        tile_dst[tile_row * q_rows + r] = half(dl * float(q[j] & mask) - ml);\n"
            "    }\n"
            "}\n";
}

static void dispatch_dequant_iq2(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> grid,
        id<MTLBuffer> dst,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:grid offset:0 atIndex:1];
    [enc setBuffer:dst offset:0 atIndex:2];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:3];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:4];
    [enc setBytes:&total length:sizeof(total) atIndex:5];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_dequant_iq2_offset(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        NSUInteger src_offset,
        id<MTLBuffer> grid,
        id<MTLBuffer> dst,
        NSUInteger dst_offset,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:src_offset atIndex:0];
    [enc setBuffer:grid offset:0 atIndex:1];
    [enc setBuffer:dst offset:dst_offset atIndex:2];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:3];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:4];
    [enc setBytes:&total length:sizeof(total) atIndex:5];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_dequant_q2(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_dequant_q2_offset(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        NSUInteger src_offset,
        id<MTLBuffer> dst,
        NSUInteger dst_offset,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:src_offset atIndex:0];
    [enc setBuffer:dst offset:dst_offset atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_dequant_iq2_batched(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> grid,
        id<MTLBuffer> dst,
        uint32_t experts,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = experts * q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:grid offset:0 atIndex:1];
    [enc setBuffer:dst offset:0 atIndex:2];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:3];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:4];
    [enc setBytes:&total length:sizeof(total) atIndex:5];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_dequant_q2_batched(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        uint32_t experts,
        uint32_t q_rows,
        uint32_t q_cols) {
    const uint32_t total = experts * q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&total length:sizeof(total) atIndex:4];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_pack_x_f16(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        NSUInteger dst_offset,
        uint32_t total) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:dst_offset atIndex:1];
    [enc setBytes:&total length:sizeof(total) atIndex:2];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_pack_iq2_split3(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> grid,
        id<MTLBuffer> dst,
        NSUInteger dst_offset,
        uint32_t q_rows,
        uint32_t q_cols,
        uint32_t split) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:grid offset:0 atIndex:1];
    [enc setBuffer:dst offset:dst_offset atIndex:2];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:3];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:4];
    [enc setBytes:&split length:sizeof(split) atIndex:5];
    [enc setBytes:&total length:sizeof(total) atIndex:6];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void dispatch_pack_q2_split3(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        NSUInteger dst_offset,
        uint32_t q_rows,
        uint32_t q_cols,
        uint32_t split) {
    const uint32_t total = q_rows * (q_cols / QK_K) * 16u;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:dst_offset atIndex:1];
    [enc setBytes:&q_rows length:sizeof(q_rows) atIndex:2];
    [enc setBytes:&q_cols length:sizeof(q_cols) atIndex:3];
    [enc setBytes:&split length:sizeof(split) atIndex:4];
    [enc setBytes:&total length:sizeof(total) atIndex:5];
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void gpu_dequant_moe_once(gpu_dequant_moe *m) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_dequant_iq2(cb, m->iq2_pipe, m->gate_q, m->grid, m->w_gate,
                         (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
    dispatch_dequant_iq2(cb, m->iq2_pipe, m->up_q, m->grid, m->w_up,
                         (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
    dispatch_dequant_q2(cb, m->q2_pipe, m->down_q, m->w_down,
                        (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: GPU dequant command buffer failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static gpu_dequant_moe make_gpu_dequant_moe(int in_dim, int mid_dim) {
    @autoreleasepool {
        gpu_dequant_moe m = {
            .in_dim = in_dim,
            .mid_dim = mid_dim,
        };
        m.dev = MTLCreateSystemDefaultDevice();
        if (!m.dev) {
            fprintf(stderr, "moe-batch-bench: Metal is unavailable\n");
            exit(1);
        }
        m.queue = [m.dev newCommandQueue];
        if (!m.queue) {
            fprintf(stderr, "moe-batch-bench: failed to create Metal command queue\n");
            exit(1);
        }

        NSError *err = nil;
        id<MTLLibrary> lib = [m.dev newLibraryWithSource:gpu_dequant_metal_source()
                                                 options:nil
                                                   error:&err];
        if (!lib) {
            fprintf(stderr, "moe-batch-bench: failed to compile GPU dequant kernels: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        id<MTLFunction> iq2_fn = [lib newFunctionWithName:@"dequant_iq2_xxs_transpose_bf16"];
        id<MTLFunction> q2_fn = [lib newFunctionWithName:@"dequant_q2_k_transpose_bf16"];
        id<MTLFunction> iq2_i8_fn = [lib newFunctionWithName:@"dequant_iq2_xxs_transpose_i8"];
        id<MTLFunction> q2_i8_fn = [lib newFunctionWithName:@"dequant_q2_k_transpose_i8"];
        id<MTLFunction> iq2_i8_batched_fn = [lib newFunctionWithName:@"dequant_iq2_xxs_transpose_i8_batched"];
        id<MTLFunction> q2_i8_batched_fn = [lib newFunctionWithName:@"dequant_q2_k_transpose_i8_batched"];
        m.iq2_pipe = [m.dev newComputePipelineStateWithFunction:iq2_fn error:&err];
        if (!m.iq2_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create IQ2 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.q2_pipe = [m.dev newComputePipelineStateWithFunction:q2_fn error:&err];
        if (!m.q2_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create Q2 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.iq2_i8_pipe = [m.dev newComputePipelineStateWithFunction:iq2_i8_fn error:&err];
        if (!m.iq2_i8_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create IQ2 i8 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.q2_i8_pipe = [m.dev newComputePipelineStateWithFunction:q2_i8_fn error:&err];
        if (!m.q2_i8_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create Q2 i8 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.iq2_i8_batched_pipe = [m.dev newComputePipelineStateWithFunction:iq2_i8_batched_fn error:&err];
        if (!m.iq2_i8_batched_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create IQ2 batched i8 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.q2_i8_batched_pipe = [m.dev newComputePipelineStateWithFunction:q2_i8_batched_fn error:&err];
        if (!m.q2_i8_batched_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create Q2 batched i8 GPU dequant pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }

        const size_t gate_blocks = (size_t)mid_dim * (size_t)(in_dim / QK_K);
        const size_t down_blocks = (size_t)in_dim * (size_t)(mid_dim / QK_K);
        m.gate_q = [m.dev newBufferWithLength:gate_blocks * sizeof(bench_block_iq2_xxs)
                                      options:MTLResourceStorageModeShared];
        m.up_q = [m.dev newBufferWithLength:gate_blocks * sizeof(bench_block_iq2_xxs)
                                    options:MTLResourceStorageModeShared];
        m.down_q = [m.dev newBufferWithLength:down_blocks * sizeof(bench_block_q2_K)
                                      options:MTLResourceStorageModeShared];
        m.w_gate = [m.dev newBufferWithLength:(NSUInteger)in_dim * (NSUInteger)mid_dim * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
        m.w_up = [m.dev newBufferWithLength:(NSUInteger)in_dim * (NSUInteger)mid_dim * sizeof(uint16_t)
                                    options:MTLResourceStorageModeShared];
        m.w_down = [m.dev newBufferWithLength:(NSUInteger)mid_dim * (NSUInteger)in_dim * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
        m.w_gate_i8 = [m.dev newBufferWithLength:(NSUInteger)in_dim * (NSUInteger)mid_dim
                                         options:MTLResourceStorageModeShared];
        m.w_up_i8 = [m.dev newBufferWithLength:(NSUInteger)in_dim * (NSUInteger)mid_dim
                                       options:MTLResourceStorageModeShared];
        m.w_down_i8 = [m.dev newBufferWithLength:(NSUInteger)mid_dim * (NSUInteger)in_dim
                                         options:MTLResourceStorageModeShared];
        m.grid = [m.dev newBufferWithBytes:bench_iq2xxs_grid
                                    length:sizeof(bench_iq2xxs_grid)
                                   options:MTLResourceStorageModeShared];
        if (!m.gate_q || !m.up_q || !m.down_q || !m.w_gate || !m.w_up || !m.w_down ||
            !m.w_gate_i8 || !m.w_up_i8 || !m.w_down_i8 || !m.grid) {
            fprintf(stderr, "moe-batch-bench: failed to allocate GPU dequant buffers\n");
            exit(1);
        }

        fill_iq2_bank((bench_block_iq2_xxs *)[m.gate_q contents], gate_blocks, 101u);
        fill_iq2_bank((bench_block_iq2_xxs *)[m.up_q contents], gate_blocks, 201u);
        fill_q2_bank((bench_block_q2_K *)[m.down_q contents], down_blocks, 301u);

        return m;
    }
}

static void gpu_dequant_iq2_tensor_once(
        gpu_dequant_moe *m,
        id<MTLBuffer> src,
        id<MTLBuffer> dst) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_dequant_iq2(cb, m->iq2_pipe, src, m->grid, dst,
                         (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: GPU IQ2 tensor dequant failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static void gpu_dequant_q2_tensor_once(gpu_dequant_moe *m) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_dequant_q2(cb, m->q2_pipe, m->down_q, m->w_down,
                        (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: GPU Q2 tensor dequant failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static void gpu_dequant_iq2_tensor_i8_once(
        gpu_dequant_moe *m,
        id<MTLBuffer> src,
        id<MTLBuffer> dst) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_dequant_iq2(cb, m->iq2_i8_pipe, src, m->grid, dst,
                         (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: GPU IQ2 tensor i8 dequant failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static void gpu_dequant_q2_tensor_i8_once(gpu_dequant_moe *m) {
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_dequant_q2(cb, m->q2_i8_pipe, m->down_q, m->w_down_i8,
                        (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: GPU Q2 tensor i8 dequant failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static void ane_pack_gpu_once(ane_pack_gpu *m) {
    const NSUInteger x_bytes = (NSUInteger)m->batch * (NSUInteger)m->in_dim * sizeof(uint16_t);
    const NSUInteger w_bytes = (NSUInteger)m->in_dim * (NSUInteger)m->mid_dim * sizeof(uint16_t);
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    dispatch_pack_x_f16(cb, m->x_pipe, m->x, m->packed, 0,
                        (uint32_t)((NSUInteger)m->batch * (NSUInteger)m->in_dim));
    dispatch_pack_iq2_split3(cb, m->iq2_pipe, m->gate_q, m->grid, m->packed,
                             x_bytes,
                             (uint32_t)m->mid_dim, (uint32_t)m->in_dim, (uint32_t)m->split);
    dispatch_pack_iq2_split3(cb, m->iq2_pipe, m->up_q, m->grid, m->packed,
                             x_bytes + w_bytes,
                             (uint32_t)m->mid_dim, (uint32_t)m->in_dim, (uint32_t)m->split);
    dispatch_pack_q2_split3(cb, m->q2_pipe, m->down_q, m->packed,
                            x_bytes + 2u * w_bytes,
                            (uint32_t)m->in_dim, (uint32_t)m->mid_dim, (uint32_t)m->split);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: ANE GPU pack command buffer failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static id<MTLComputePipelineState> make_pipeline_or_die(
        id<MTLDevice> dev,
        id<MTLLibrary> lib,
        NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "moe-batch-bench: Metal function missing: %s\n", [name UTF8String]);
        exit(1);
    }
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) {
        fprintf(stderr, "moe-batch-bench: failed to create Metal pipeline %s: %s\n",
                [name UTF8String], [[err description] UTF8String]);
        exit(1);
    }
    return pipe;
}

static ane_pack_gpu make_ane_pack_gpu(int batch, int in_dim, int mid_dim, int split) {
    @autoreleasepool {
        ane_pack_gpu m = {
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .split = split,
        };
        m.dev = MTLCreateSystemDefaultDevice();
        if (!m.dev) {
            fprintf(stderr, "moe-batch-bench: Metal is unavailable\n");
            exit(1);
        }
        m.queue = [m.dev newCommandQueue];
        if (!m.queue) {
            fprintf(stderr, "moe-batch-bench: failed to create Metal command queue\n");
            exit(1);
        }

        NSError *err = nil;
        id<MTLLibrary> lib = [m.dev newLibraryWithSource:gpu_dequant_metal_source()
                                                 options:nil
                                                   error:&err];
        if (!lib) {
            fprintf(stderr, "moe-batch-bench: failed to compile ANE GPU pack kernels: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        m.x_pipe = make_pipeline_or_die(m.dev, lib, @"pack_x_f32_to_f16");
        m.iq2_pipe = make_pipeline_or_die(m.dev, lib, @"pack_iq2_xxs_split3_f16");
        m.q2_pipe = make_pipeline_or_die(m.dev, lib, @"pack_q2_k_split3_f16");

        const size_t gate_blocks = (size_t)mid_dim * (size_t)(in_dim / QK_K);
        const size_t down_blocks = (size_t)in_dim * (size_t)(mid_dim / QK_K);
        const size_t x_elems = (size_t)batch * (size_t)in_dim;
        const size_t packed_elems = x_elems + 3u * (size_t)in_dim * (size_t)mid_dim;
        m.x = [m.dev newBufferWithLength:x_elems * sizeof(float)
                                 options:MTLResourceStorageModeShared];
        m.gate_q = [m.dev newBufferWithLength:gate_blocks * sizeof(bench_block_iq2_xxs)
                                      options:MTLResourceStorageModeShared];
        m.up_q = [m.dev newBufferWithLength:gate_blocks * sizeof(bench_block_iq2_xxs)
                                    options:MTLResourceStorageModeShared];
        m.down_q = [m.dev newBufferWithLength:down_blocks * sizeof(bench_block_q2_K)
                                      options:MTLResourceStorageModeShared];
        m.packed = [m.dev newBufferWithLength:packed_elems * sizeof(uint16_t)
                                      options:MTLResourceStorageModeShared];
        m.grid = [m.dev newBufferWithBytes:bench_iq2xxs_grid
                                    length:sizeof(bench_iq2xxs_grid)
                                   options:MTLResourceStorageModeShared];
        if (!m.x || !m.gate_q || !m.up_q || !m.down_q || !m.packed || !m.grid) {
            fprintf(stderr, "moe-batch-bench: failed to allocate ANE GPU pack buffers\n");
            exit(1);
        }

        float *x = (float *)[m.x contents];
        for (size_t i = 0; i < x_elems; i++) x[i] = sinf((float)i * 0.001f);
        fill_iq2_bank((bench_block_iq2_xxs *)[m.gate_q contents], gate_blocks, 101u);
        fill_iq2_bank((bench_block_iq2_xxs *)[m.up_q contents], gate_blocks, 201u);
        fill_q2_bank((bench_block_q2_K *)[m.down_q contents], down_blocks, 301u);

        return m;
    }
}

static gpu_matrix make_gpu_matrix(id<MTLDevice> dev, NSUInteger rows, NSUInteger cols, dense_dtype dtype) {
    gpu_matrix m = {0};
    m.rows = rows;
    m.cols = cols;
    m.elem_bytes = dense_elem_bytes(dtype);
    m.row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:cols dataType:dense_mps_dtype(dtype)];
    const NSUInteger bytes = rows * m.row_bytes;
    m.buffer = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!m.buffer) {
        fprintf(stderr, "moe-batch-bench: failed to allocate Metal buffer\n");
        exit(1);
    }
    MPSMatrixDescriptor *desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:cols
                                             rowBytes:m.row_bytes
                                             dataType:dense_mps_dtype(dtype)];
    m.matrix = [[MPSMatrix alloc] initWithBuffer:m.buffer descriptor:desc];
    if (!m.matrix) {
        fprintf(stderr, "moe-batch-bench: failed to create MPSMatrix\n");
        exit(1);
    }
    return m;
}

static void fill_gpu_matrix(gpu_matrix *m, uint32_t seed, dense_dtype dtype) {
    uint8_t *base = (uint8_t *)[m->buffer contents];
    uint32_t s = seed ? seed : 1u;
    for (NSUInteger r = 0; r < m->rows; r++) {
        uint8_t *row_bytes = base + r * m->row_bytes;
        for (NSUInteger c = 0; c < m->cols; c++) {
            s = 1664525u * s + 1013904223u;
            uint32_t bits = (s >> 9) | 0x3f800000u;
            float v;
            memcpy(&v, &bits, sizeof(v));
            v = (v - 1.5f) * 0.05f;
            if (dtype == DENSE_FP32) {
                ((float *)row_bytes)[c] = v;
            } else {
                ((uint16_t *)row_bytes)[c] = f32_to_dense16_bits(v, dtype);
            }
        }
    }
}

static void encode_activation(
        id<MTLCommandBuffer> cb,
        id<MTLComputePipelineState> pipe,
        const gpu_matrix *gate,
        const gpu_matrix *up,
        const gpu_matrix *mid) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:gate->buffer offset:0 atIndex:0];
    [enc setBuffer:up->buffer offset:0 atIndex:1];
    [enc setBuffer:mid->buffer offset:0 atIndex:2];
    uint32_t cols = (uint32_t)mid->cols;
    uint32_t row_stride = (uint32_t)(mid->row_bytes / mid->elem_bytes);
    uint32_t n = (uint32_t)(mid->rows * mid->cols);
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&row_stride length:sizeof(row_stride) atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];

    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void gpu_mlp_once(
        id<MTLCommandQueue> queue,
        MPSMatrixMultiplication *mm_gate,
        MPSMatrixMultiplication *mm_down,
        id<MTLComputePipelineState> activation_pipe,
        const gpu_matrix *x,
        const gpu_matrix *w_gate,
        const gpu_matrix *w_up,
        const gpu_matrix *w_down,
        const gpu_matrix *gate,
        const gpu_matrix *up,
        const gpu_matrix *mid,
        const gpu_matrix *out) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    [mm_gate encodeToCommandBuffer:cb leftMatrix:x->matrix rightMatrix:w_gate->matrix resultMatrix:gate->matrix];
    [mm_gate encodeToCommandBuffer:cb leftMatrix:x->matrix rightMatrix:w_up->matrix resultMatrix:up->matrix];
    encode_activation(cb, activation_pipe, gate, up, mid);
    [mm_down encodeToCommandBuffer:cb leftMatrix:mid->matrix rightMatrix:w_down->matrix resultMatrix:out->matrix];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: Metal command buffer failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static bench_result bench_gpu_mps(const bench_config *cfg, int batch, dense_dtype dtype) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "moe-batch-bench: Metal is unavailable\n");
            exit(1);
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            fprintf(stderr, "moe-batch-bench: failed to create Metal command queue\n");
            exit(1);
        }

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:activation_metal_source()
                                               options:nil
                                                 error:&err];
        if (!lib) {
            fprintf(stderr, "moe-batch-bench: failed to compile activation kernel: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        id<MTLFunction> fn = [lib newFunctionWithName:
            dtype == DENSE_BF16 ? @"silu_mul_bf16" :
            dtype == DENSE_FP32 ? @"silu_mul_f32" : @"silu_mul_f16"];
        id<MTLComputePipelineState> activation_pipe =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!activation_pipe) {
            fprintf(stderr, "moe-batch-bench: failed to create activation pipeline: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }

        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        gpu_matrix x = make_gpu_matrix(dev, (NSUInteger)batch, (NSUInteger)in_dim, dtype);
        gpu_matrix w_gate = make_gpu_matrix(dev, (NSUInteger)in_dim, (NSUInteger)mid_dim, dtype);
        gpu_matrix w_up = make_gpu_matrix(dev, (NSUInteger)in_dim, (NSUInteger)mid_dim, dtype);
        gpu_matrix w_down = make_gpu_matrix(dev, (NSUInteger)mid_dim, (NSUInteger)in_dim, dtype);
        gpu_matrix gate = make_gpu_matrix(dev, (NSUInteger)batch, (NSUInteger)mid_dim, dtype);
        gpu_matrix up = make_gpu_matrix(dev, (NSUInteger)batch, (NSUInteger)mid_dim, dtype);
        gpu_matrix mid = make_gpu_matrix(dev, (NSUInteger)batch, (NSUInteger)mid_dim, dtype);
        gpu_matrix out = make_gpu_matrix(dev, (NSUInteger)batch, (NSUInteger)in_dim, dtype);

        fill_gpu_matrix(&x, 1u, dtype);
        fill_gpu_matrix(&w_gate, 2u, dtype);
        fill_gpu_matrix(&w_up, 3u, dtype);
        fill_gpu_matrix(&w_down, 4u, dtype);

        MPSMatrixMultiplication *mm_gate =
            [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                              transposeLeft:false
                                             transposeRight:false
                                                 resultRows:(NSUInteger)batch
                                              resultColumns:(NSUInteger)mid_dim
                                            interiorColumns:(NSUInteger)in_dim
                                                      alpha:1.0
                                                       beta:0.0];
        MPSMatrixMultiplication *mm_down =
            [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                              transposeLeft:false
                                             transposeRight:false
                                                 resultRows:(NSUInteger)batch
                                              resultColumns:(NSUInteger)in_dim
                                            interiorColumns:(NSUInteger)mid_dim
                                                      alpha:1.0
                                                       beta:0.0];

        for (int i = 0; i < cfg->warmup; i++) {
            gpu_mlp_once(queue, mm_gate, mm_down, activation_pipe,
                         &x, &w_gate, &w_up, &w_down, &gate, &up, &mid, &out);
        }

        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            gpu_mlp_once(queue, mm_gate, mm_down, activation_pipe,
                         &x, &w_gate, &w_up, &w_down, &gate, &up, &mid, &out);
        }
        const double sec = now_sec() - t0;
        const double ms = sec * 1000.0 / (double)cfg->iters;

        bench_result r = {
            .backend = "gpu_mps_matmul",
            .dtype = dense_dtype_name(dtype),
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .iters = cfg->iters,
            .ms = ms,
            .gflops = mlp_gflop(batch, in_dim, mid_dim) / (ms / 1000.0),
        };
        return r;
    }
}

typedef struct {
    id<MTLBuffer> buffer;
    NSUInteger rows;
    NSUInteger cols;
} graph_matrix;

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    graph_matrix x;
    graph_matrix w_gate;
    graph_matrix w_up;
    graph_matrix w_down;
    graph_matrix out_buf;
    MPSGraph *graph;
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds;
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results;
    int batch;
    int in_dim;
    int mid_dim;
} gpu_graph_bf16_mlp;

static graph_matrix make_graph_matrix(id<MTLDevice> dev, NSUInteger rows, NSUInteger cols) {
    graph_matrix m = {0};
    m.rows = rows;
    m.cols = cols;
    const NSUInteger bytes = rows * cols * sizeof(uint16_t);
    m.buffer = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!m.buffer) {
        fprintf(stderr, "moe-batch-bench: failed to allocate MPSGraph buffer\n");
        exit(1);
    }
    return m;
}

static void fill_graph_matrix(graph_matrix *m, uint32_t seed, dense_dtype dtype) {
    uint16_t *base = (uint16_t *)[m->buffer contents];
    fill_dense(base, (size_t)m->rows * (size_t)m->cols, seed, dtype);
}

static MPSShape *graph_shape2(NSUInteger rows, NSUInteger cols) {
    return @[ @(rows), @(cols) ];
}

static MPSGraphTensorData *graph_tensor_data(const graph_matrix *m, dense_dtype dtype) {
    MPSGraphTensorData *data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:m->buffer
                                                shape:graph_shape2(m->rows, m->cols)
                                             dataType:dense_mps_dtype(dtype)];
    if (!data) {
        fprintf(stderr, "moe-batch-bench: failed to wrap MPSGraph tensor data\n");
        exit(1);
    }
    return data;
}

static void graph_mlp_once(
        MPSGraph *graph,
        id<MTLCommandQueue> queue,
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results) {
    [graph runWithMTLCommandQueue:queue
                            feeds:feeds
                 targetOperations:nil
                resultsDictionary:results];
}

static gpu_graph_bf16_mlp make_gpu_graph_bf16_mlp(
        int batch,
        int in_dim,
        int mid_dim,
        uint32_t seed_base) {
    gpu_graph_bf16_mlp m = {
        .batch = batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
    };
    m.dev = MTLCreateSystemDefaultDevice();
    if (!m.dev) {
        fprintf(stderr, "moe-batch-bench: Metal is unavailable\n");
        exit(1);
    }
    m.queue = [m.dev newCommandQueue];
    if (!m.queue) {
        fprintf(stderr, "moe-batch-bench: failed to create Metal command queue\n");
        exit(1);
    }

    m.x = make_graph_matrix(m.dev, (NSUInteger)batch, (NSUInteger)in_dim);
    m.w_gate = make_graph_matrix(m.dev, (NSUInteger)in_dim, (NSUInteger)mid_dim);
    m.w_up = make_graph_matrix(m.dev, (NSUInteger)in_dim, (NSUInteger)mid_dim);
    m.w_down = make_graph_matrix(m.dev, (NSUInteger)mid_dim, (NSUInteger)in_dim);
    m.out_buf = make_graph_matrix(m.dev, (NSUInteger)batch, (NSUInteger)in_dim);

    fill_graph_matrix(&m.x, seed_base + 0u, DENSE_BF16);
    fill_graph_matrix(&m.w_gate, seed_base + 1u, DENSE_BF16);
    fill_graph_matrix(&m.w_up, seed_base + 2u, DENSE_BF16);
    fill_graph_matrix(&m.w_down, seed_base + 3u, DENSE_BF16);

    m.graph = [MPSGraph new];
    MPSGraphTensor *x_t =
        [m.graph placeholderWithShape:graph_shape2((NSUInteger)batch, (NSUInteger)in_dim)
                             dataType:MPSDataTypeBFloat16
                                 name:@"x"];
    MPSGraphTensor *w_gate_t =
        [m.graph placeholderWithShape:graph_shape2((NSUInteger)in_dim, (NSUInteger)mid_dim)
                             dataType:MPSDataTypeBFloat16
                                 name:@"w_gate"];
    MPSGraphTensor *w_up_t =
        [m.graph placeholderWithShape:graph_shape2((NSUInteger)in_dim, (NSUInteger)mid_dim)
                             dataType:MPSDataTypeBFloat16
                                 name:@"w_up"];
    MPSGraphTensor *w_down_t =
        [m.graph placeholderWithShape:graph_shape2((NSUInteger)mid_dim, (NSUInteger)in_dim)
                             dataType:MPSDataTypeBFloat16
                                 name:@"w_down"];

    MPSGraphTensor *gate_t =
        [m.graph matrixMultiplicationWithPrimaryTensor:x_t
                                       secondaryTensor:w_gate_t
                                                  name:@"gate"];
    MPSGraphTensor *up_t =
        [m.graph matrixMultiplicationWithPrimaryTensor:x_t
                                       secondaryTensor:w_up_t
                                                  name:@"up"];
    MPSGraphTensor *sigmoid_t = [m.graph sigmoidWithTensor:gate_t name:@"sigmoid"];
    MPSGraphTensor *silu_t =
        [m.graph multiplicationWithPrimaryTensor:gate_t
                                 secondaryTensor:sigmoid_t
                                            name:@"silu"];
    MPSGraphTensor *mid_t =
        [m.graph multiplicationWithPrimaryTensor:silu_t
                                 secondaryTensor:up_t
                                            name:@"mid"];
    MPSGraphTensor *out_t =
        [m.graph matrixMultiplicationWithPrimaryTensor:mid_t
                                       secondaryTensor:w_down_t
                                                  name:@"out"];
    out_t = [m.graph castTensor:out_t toType:MPSDataTypeBFloat16 name:@"out_bf16"];

    MPSGraphTensorData *x_data = graph_tensor_data(&m.x, DENSE_BF16);
    MPSGraphTensorData *w_gate_data = graph_tensor_data(&m.w_gate, DENSE_BF16);
    MPSGraphTensorData *w_up_data = graph_tensor_data(&m.w_up, DENSE_BF16);
    MPSGraphTensorData *w_down_data = graph_tensor_data(&m.w_down, DENSE_BF16);
    MPSGraphTensorData *out_data = graph_tensor_data(&m.out_buf, DENSE_BF16);

    m.feeds = @{
        x_t: x_data,
        w_gate_t: w_gate_data,
        w_up_t: w_up_data,
        w_down_t: w_down_data,
    };
    m.results = @{
        out_t: out_data,
    };
    return m;
}

static void gpu_graph_bf16_mlp_once(gpu_graph_bf16_mlp *m) {
    graph_mlp_once(m->graph, m->queue, m->feeds, m->results);
}

static bench_result bench_gpu_graph_bf16(const bench_config *cfg, int batch) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        gpu_graph_bf16_mlp mlp = make_gpu_graph_bf16_mlp(batch, in_dim, mid_dim, 1u);

        for (int i = 0; i < cfg->warmup; i++) {
            gpu_graph_bf16_mlp_once(&mlp);
        }

        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            gpu_graph_bf16_mlp_once(&mlp);
        }
        const double sec = now_sec() - t0;
        const double ms = sec * 1000.0 / (double)cfg->iters;

        bench_result r = {
            .backend = "gpu_mpsgraph_mlp",
            .dtype = "bf16",
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .iters = cfg->iters,
            .ms = ms,
            .gflops = mlp_gflop(batch, in_dim, mid_dim) / (ms / 1000.0),
        };
        return r;
    }
}

static bench_result bench_gpu(const bench_config *cfg, int batch, dense_dtype dtype) {
    if (dtype == DENSE_BF16) {
        return bench_gpu_graph_bf16(cfg, batch);
    }
    return bench_gpu_mps(cfg, batch, dtype);
}

typedef struct split_runner split_runner;

typedef struct {
    split_runner *runner;
    bool is_gpu;
    int seen_generation;
} split_worker;

struct split_runner {
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    int generation;
    int done_count;
    bool stop;
    gpu_graph_bf16_mlp *gpu;
    bnns_mlp *amx;
    int amx_batch;
    int mid_dim;
    split_worker gpu_worker;
    split_worker amx_worker;
    pthread_t gpu_thread;
    pthread_t amx_thread;
};

static void *split_worker_main(void *arg) {
    split_worker *w = (split_worker *)arg;
    split_runner *r = w->runner;
    for (;;) {
        pthread_mutex_lock(&r->mutex);
        while (!r->stop && w->seen_generation == r->generation) {
            pthread_cond_wait(&r->start_cond, &r->mutex);
        }
        if (r->stop) {
            pthread_mutex_unlock(&r->mutex);
            return NULL;
        }
        w->seen_generation = r->generation;
        pthread_mutex_unlock(&r->mutex);

        @autoreleasepool {
            if (w->is_gpu) {
                gpu_graph_bf16_mlp_once(r->gpu);
            } else {
                bnns_mlp_once(r->amx, r->amx_batch, r->mid_dim);
            }
        }

        pthread_mutex_lock(&r->mutex);
        r->done_count++;
        if (r->done_count == 2) {
            pthread_cond_signal(&r->done_cond);
        }
        pthread_mutex_unlock(&r->mutex);
    }
}

static void split_runner_start(split_runner *r, gpu_graph_bf16_mlp *gpu, bnns_mlp *amx,
                               int amx_batch, int mid_dim) {
    memset(r, 0, sizeof(*r));
    pthread_mutex_init(&r->mutex, NULL);
    pthread_cond_init(&r->start_cond, NULL);
    pthread_cond_init(&r->done_cond, NULL);
    r->gpu = gpu;
    r->amx = amx;
    r->amx_batch = amx_batch;
    r->mid_dim = mid_dim;
    r->gpu_worker = (split_worker){ .runner = r, .is_gpu = true };
    r->amx_worker = (split_worker){ .runner = r, .is_gpu = false };
    if (pthread_create(&r->gpu_thread, NULL, split_worker_main, &r->gpu_worker) != 0 ||
        pthread_create(&r->amx_thread, NULL, split_worker_main, &r->amx_worker) != 0) {
        fprintf(stderr, "moe-batch-bench: failed to create split worker threads\n");
        exit(1);
    }
}

static void split_runner_once(split_runner *r) {
    pthread_mutex_lock(&r->mutex);
    r->done_count = 0;
    r->generation++;
    pthread_cond_broadcast(&r->start_cond);
    while (r->done_count < 2) {
        pthread_cond_wait(&r->done_cond, &r->mutex);
    }
    pthread_mutex_unlock(&r->mutex);
}

static void split_runner_stop(split_runner *r) {
    pthread_mutex_lock(&r->mutex);
    r->stop = true;
    pthread_cond_broadcast(&r->start_cond);
    pthread_mutex_unlock(&r->mutex);
    pthread_join(r->gpu_thread, NULL);
    pthread_join(r->amx_thread, NULL);
    pthread_cond_destroy(&r->done_cond);
    pthread_cond_destroy(&r->start_cond);
    pthread_mutex_destroy(&r->mutex);
}

static split_result bench_split_bf16(const bench_config *cfg, int gpu_batch, int amx_batch) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        gpu_graph_bf16_mlp gpu = make_gpu_graph_bf16_mlp(gpu_batch, in_dim, mid_dim, 1u);
        gpu_graph_bf16_mlp gpu_other = make_gpu_graph_bf16_mlp(amx_batch, in_dim, mid_dim, 11u);

        void *x = alloc_dense((size_t)amx_batch * (size_t)in_dim, DENSE_BF16);
        void *w_gate = alloc_dense((size_t)in_dim * (size_t)mid_dim, DENSE_BF16);
        void *w_up = alloc_dense((size_t)in_dim * (size_t)mid_dim, DENSE_BF16);
        void *w_down = alloc_dense((size_t)mid_dim * (size_t)in_dim, DENSE_BF16);
        void *gate = alloc_dense((size_t)amx_batch * (size_t)mid_dim, DENSE_BF16);
        void *up = alloc_dense((size_t)amx_batch * (size_t)mid_dim, DENSE_BF16);
        void *mid = alloc_dense((size_t)amx_batch * (size_t)mid_dim, DENSE_BF16);
        void *out = alloc_dense((size_t)amx_batch * (size_t)in_dim, DENSE_BF16);

        fill_dense(x, (size_t)amx_batch * (size_t)in_dim, 11u, DENSE_BF16);
        fill_dense(w_gate, (size_t)in_dim * (size_t)mid_dim, 12u, DENSE_BF16);
        fill_dense(w_up, (size_t)in_dim * (size_t)mid_dim, 13u, DENSE_BF16);
        fill_dense(w_down, (size_t)mid_dim * (size_t)in_dim, 14u, DENSE_BF16);

        bnns_mlp amx = make_bnns_mlp(amx_batch, in_dim, mid_dim,
                                     x, w_gate, w_up, w_down,
                                     gate, up, mid, out, DENSE_BF16);

        for (int i = 0; i < cfg->warmup; i++) {
            gpu_graph_bf16_mlp_once(&gpu);
            gpu_graph_bf16_mlp_once(&gpu_other);
        }

        const double gpu_seq_t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            gpu_graph_bf16_mlp_once(&gpu);
            gpu_graph_bf16_mlp_once(&gpu_other);
        }
        const double gpu_seq_sec = now_sec() - gpu_seq_t0;

        split_runner runner;
        split_runner_start(&runner, &gpu, &amx, amx_batch, mid_dim);

        for (int i = 0; i < cfg->warmup; i++) {
            split_runner_once(&runner);
        }

        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            split_runner_once(&runner);
        }
        const double sec = now_sec() - t0;

        split_runner_stop(&runner);
        bnns_mlp_free(&amx);
        free(out);
        free(mid);
        free(up);
        free(gate);
        free(w_down);
        free(w_up);
        free(w_gate);
        free(x);

        const int combined_batch = gpu_batch + amx_batch;
        const double ms = sec * 1000.0 / (double)cfg->iters;
        const double gpu_seq_ms = gpu_seq_sec * 1000.0 / (double)cfg->iters;
        split_result r = {
            .backend = "split_gpu_mpsgraph_amx_bnns",
            .dtype = "bf16",
            .gpu_batch = gpu_batch,
            .amx_batch = amx_batch,
            .combined_batch = combined_batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .iters = cfg->iters,
            .ms = ms,
            .tokens_per_s = (double)combined_batch * 1000.0 / ms,
            .gflops = (mlp_gflop(gpu_batch, in_dim, mid_dim) +
                       mlp_gflop(amx_batch, in_dim, mid_dim)) / (ms / 1000.0),
            .gpu_seq_ms = gpu_seq_ms,
            .gpu_seq_tokens_per_s = (double)combined_batch * 1000.0 / gpu_seq_ms,
            .speedup_vs_gpu_seq = gpu_seq_ms / ms,
        };
        return r;
    }
}

static uint32_t rng_next(uint32_t *s) {
    *s = 1664525u * (*s) + 1013904223u;
    return *s;
}

static uint16_t fp16_bits(float v) {
    fp16_t h = (fp16_t)v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static uint64_t checked_bytes(uint64_t a, uint64_t b, const char *label) {
    if (a != 0 && b > UINT64_MAX / a) {
        fprintf(stderr, "moe-batch-bench: byte size overflow for %s\n", label);
        exit(1);
    }
    return a * b;
}

static ds4_gpu_tensor *alloc_ds4_tensor(uint64_t bytes, const char *label) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) {
        fprintf(stderr, "moe-batch-bench: failed to allocate DS4 GPU tensor %s (%llu bytes)\n",
                label, (unsigned long long)bytes);
        exit(1);
    }
    return t;
}

static void *ds4_tensor_contents_checked(ds4_gpu_tensor *t, const char *label) {
    void *p = ds4_gpu_tensor_contents(t);
    if (!p) {
        fprintf(stderr, "moe-batch-bench: failed to map DS4 GPU tensor %s\n", label);
        exit(1);
    }
    return p;
}

static void fill_f32(float *x, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (rng_next(&s) >> 9) | 0x3f800000u;
        float v;
        memcpy(&v, &bits, sizeof(v));
        x[i] = (v - 1.5f) * 0.05f;
    }
}

static void fill_i32_zero(int32_t *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = 0;
}

static void fill_f32_one(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = 1.0f;
}

static void fill_iq2_bank(bench_block_iq2_xxs *blocks, size_t n_blocks, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    const uint16_t d = fp16_bits(0.03125f);
    for (size_t i = 0; i < n_blocks; i++) {
        blocks[i].d = d;
        for (size_t j = 0; j < QK_K / 8; j++) {
            blocks[i].qs[j] = (uint16_t)rng_next(&s);
        }
    }
}

static void fill_q2_bank(bench_block_q2_K *blocks, size_t n_blocks, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    const uint16_t d = fp16_bits(0.03125f);
    const uint16_t dmin = fp16_bits(0.0f);
    for (size_t i = 0; i < n_blocks; i++) {
        for (size_t j = 0; j < QK_K / 16; j++) {
            blocks[i].scales[j] = (uint8_t)(0x11u + (rng_next(&s) & 0x0eu));
        }
        for (size_t j = 0; j < QK_K / 4; j++) {
            blocks[i].qs[j] = (uint8_t)rng_next(&s);
        }
        blocks[i].d = d;
        blocks[i].dmin = dmin;
    }
}

static inline uint8_t iq2_sign_mask(uint32_t s) {
    s &= 127u;
    return (uint8_t)(s | ((__builtin_popcount(s) & 1u) ? 128u : 0u));
}

static void dequant_iq2_xxs_transpose_bf16(
        const bench_block_iq2_xxs *src,
        int q_rows,
        int q_cols,
        uint16_t *dst) {
    const int blocks_per_row = q_cols / QK_K;
    for (int r = 0; r < q_rows; r++) {
        const bench_block_iq2_xxs *row = src + (size_t)r * (size_t)blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const bench_block_iq2_xxs *blk = row + b;
            const float d = fp16_bits_to_f32(blk->d);
            for (int il = 0; il < 16; il++) {
                const int ib32 = il / 2;
                const int lane = il & 1;
                const uint16_t *q2 = blk->qs + 4 * ib32;
                const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
                const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
                const uint8_t *aux8 = (const uint8_t *)&aux32_g;
                const float scale = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
                const int col0 = b * QK_K + il * 16;

                const uint8_t *grid0 =
                    (const uint8_t *)(bench_iq2xxs_grid + aux8[2 * lane + 0]);
                const uint8_t sign0 = iq2_sign_mask((aux32_s >> (14 * lane)) & 127u);
                for (int j = 0; j < 8; j++) {
                    const float sgn = (sign0 & (1u << j)) ? -1.0f : 1.0f;
                    dst[(size_t)(col0 + j) * (size_t)q_rows + (size_t)r] =
                        f32_to_bf16_bits(scale * (float)grid0[j] * sgn);
                }

                const uint8_t *grid1 =
                    (const uint8_t *)(bench_iq2xxs_grid + aux8[2 * lane + 1]);
                const uint8_t sign1 = iq2_sign_mask((aux32_s >> (14 * lane + 7)) & 127u);
                for (int j = 0; j < 8; j++) {
                    const float sgn = (sign1 & (1u << j)) ? -1.0f : 1.0f;
                    dst[(size_t)(col0 + 8 + j) * (size_t)q_rows + (size_t)r] =
                        f32_to_bf16_bits(scale * (float)grid1[j] * sgn);
                }
            }
        }
    }
}

static void dequant_q2_k_transpose_bf16(
        const bench_block_q2_K *src,
        int q_rows,
        int q_cols,
        uint16_t *dst) {
    const int blocks_per_row = q_cols / QK_K;
    for (int r = 0; r < q_rows; r++) {
        const bench_block_q2_K *row = src + (size_t)r * (size_t)blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const bench_block_q2_K *blk = row + b;
            const float d = fp16_bits_to_f32(blk->d);
            const float dmin = fp16_bits_to_f32(blk->dmin);
            for (int il0 = 0; il0 < 16; il0++) {
                const uint8_t sc = blk->scales[il0];
                const uint8_t *q = blk->qs + 32 * (il0 / 8) + 16 * (il0 & 1);
                const int il = (il0 / 2) % 4;
                const float coef = il > 1 ? (il > 2 ? 1.0f / 64.0f : 1.0f / 16.0f) :
                                   (il > 0 ? 1.0f / 4.0f : 1.0f);
                const uint8_t mask = il > 1 ? (il > 2 ? 192u : 48u) :
                                     (il > 0 ? 12u : 3u);
                const float dl = d * (float)(sc & 0x0fu) * coef;
                const float ml = dmin * (float)(sc >> 4);
                const int col0 = b * QK_K + il0 * 16;
                for (int j = 0; j < 16; j++) {
                    const float v = dl * (float)(q[j] & mask) - ml;
                    dst[(size_t)(col0 + j) * (size_t)q_rows + (size_t)r] =
                        f32_to_bf16_bits(v);
                }
            }
        }
    }
}

static void dequant_moe_expert_to_bf16(
        const bench_block_iq2_xxs *gate_bank,
        const bench_block_iq2_xxs *up_bank,
        const bench_block_q2_K *down_bank,
        int in_dim,
        int mid_dim,
        uint16_t *w_gate,
        uint16_t *w_up,
        uint16_t *w_down) {
    dequant_iq2_xxs_transpose_bf16(gate_bank, mid_dim, in_dim, w_gate);
    dequant_iq2_xxs_transpose_bf16(up_bank, mid_dim, in_dim, w_up);
    dequant_q2_k_transpose_bf16(down_bank, in_dim, mid_dim, w_down);
}

static void pack_iq2_xxs_split3_fp16(
        const bench_block_iq2_xxs *src,
        int q_rows,
        int q_cols,
        int split,
        uint16_t *dst) {
    const int blocks_per_row = q_cols / QK_K;
    const int tile_cols = q_rows / split;
    for (int r = 0; r < q_rows; r++) {
        const int tile = r / tile_cols;
        const int tile_col = r - tile * tile_cols;
        uint16_t *tile_dst = dst + (size_t)tile * (size_t)q_cols * (size_t)tile_cols;
        const bench_block_iq2_xxs *row = src + (size_t)r * (size_t)blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const bench_block_iq2_xxs *blk = row + b;
            const float d = fp16_bits_to_f32(blk->d);
            for (int il = 0; il < 16; il++) {
                const int ib32 = il / 2;
                const int lane = il & 1;
                const uint16_t *q2 = blk->qs + 4 * ib32;
                const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
                const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
                const uint8_t *aux8 = (const uint8_t *)&aux32_g;
                const float scale = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
                const int col0 = b * QK_K + il * 16;

                const uint8_t *grid0 =
                    (const uint8_t *)(bench_iq2xxs_grid + aux8[2 * lane + 0]);
                const uint8_t sign0 = iq2_sign_mask((aux32_s >> (14 * lane)) & 127u);
                for (int j = 0; j < 8; j++) {
                    const float sgn = (sign0 & (1u << j)) ? -1.0f : 1.0f;
                    tile_dst[(size_t)(col0 + j) * (size_t)tile_cols + (size_t)tile_col] =
                        f32_to_fp16_bits(scale * (float)grid0[j] * sgn);
                }

                const uint8_t *grid1 =
                    (const uint8_t *)(bench_iq2xxs_grid + aux8[2 * lane + 1]);
                const uint8_t sign1 = iq2_sign_mask((aux32_s >> (14 * lane + 7)) & 127u);
                for (int j = 0; j < 8; j++) {
                    const float sgn = (sign1 & (1u << j)) ? -1.0f : 1.0f;
                    tile_dst[(size_t)(col0 + 8 + j) * (size_t)tile_cols + (size_t)tile_col] =
                        f32_to_fp16_bits(scale * (float)grid1[j] * sgn);
                }
            }
        }
    }
}

static void pack_q2_k_split3_fp16(
        const bench_block_q2_K *src,
        int q_rows,
        int q_cols,
        int split,
        uint16_t *dst) {
    const int blocks_per_row = q_cols / QK_K;
    const int tile_rows = q_cols / split;
    for (int r = 0; r < q_rows; r++) {
        const bench_block_q2_K *row = src + (size_t)r * (size_t)blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            const bench_block_q2_K *blk = row + b;
            const float d = fp16_bits_to_f32(blk->d);
            const float dmin = fp16_bits_to_f32(blk->dmin);
            for (int il0 = 0; il0 < 16; il0++) {
                const uint8_t sc = blk->scales[il0];
                const uint8_t *q = blk->qs + 32 * (il0 / 8) + 16 * (il0 & 1);
                const int il = (il0 / 2) % 4;
                const float coef = il > 1 ? (il > 2 ? 1.0f / 64.0f : 1.0f / 16.0f) :
                                   (il > 0 ? 1.0f / 4.0f : 1.0f);
                const uint8_t mask = il > 1 ? (il > 2 ? 192u : 48u) :
                                     (il > 0 ? 12u : 3u);
                const float dl = d * (float)(sc & 0x0fu) * coef;
                const float ml = dmin * (float)(sc >> 4);
                const int col0 = b * QK_K + il0 * 16;
                for (int j = 0; j < 16; j++) {
                    const int col = col0 + j;
                    const int tile = col / tile_rows;
                    const int tile_row = col - tile * tile_rows;
                    uint16_t *tile_dst = dst + (size_t)tile * (size_t)tile_rows * (size_t)q_rows;
                    const float v = dl * (float)(q[j] & mask) - ml;
                    tile_dst[(size_t)tile_row * (size_t)q_rows + (size_t)r] =
                        f32_to_fp16_bits(v);
                }
            }
        }
    }
}

static void materialize_ane_split3_pack_fp16(
        const float *x,
        const bench_block_iq2_xxs *gate_bank,
        const bench_block_iq2_xxs *up_bank,
        const bench_block_q2_K *down_bank,
        int batch,
        int in_dim,
        int mid_dim,
        int split,
        uint16_t *packed) {
    uint16_t *p = packed;
    for (int i = 0; i < batch * in_dim; i++) {
        p[i] = f32_to_fp16_bits(x[i]);
    }
    p += (size_t)batch * (size_t)in_dim;
    pack_iq2_xxs_split3_fp16(gate_bank, mid_dim, in_dim, split, p);
    p += (size_t)in_dim * (size_t)mid_dim;
    pack_iq2_xxs_split3_fp16(up_bank, mid_dim, in_dim, split, p);
    p += (size_t)in_dim * (size_t)mid_dim;
    pack_q2_k_split3_fp16(down_bank, in_dim, mid_dim, split, p);
}

static bench_result bench_ane_pack_materialize(const bench_config *cfg, int batch) {
    const int in_dim = cfg->in_dim;
    const int mid_dim = cfg->mid_dim;
    const int split = 3;
    if ((in_dim % QK_K) != 0 || (mid_dim % QK_K) != 0 || (mid_dim % split) != 0) {
        fprintf(stderr, "moe-batch-bench: anepack requires --in/--mid multiples of 256 and --mid divisible by 3\n");
        exit(2);
    }

    const size_t gate_blocks = (size_t)mid_dim * (size_t)(in_dim / QK_K);
    const size_t down_blocks = (size_t)in_dim * (size_t)(mid_dim / QK_K);
    const size_t x_elems = (size_t)batch * (size_t)in_dim;
    const size_t packed_elems = x_elems + 3u * (size_t)in_dim * (size_t)mid_dim;

    bench_block_iq2_xxs *gate_bank = NULL;
    bench_block_iq2_xxs *up_bank = NULL;
    bench_block_q2_K *down_bank = NULL;
    float *x = NULL;
    uint16_t *packed = NULL;
    if (posix_memalign((void **)&gate_bank, 64, gate_blocks * sizeof(*gate_bank)) != 0 ||
        posix_memalign((void **)&up_bank, 64, gate_blocks * sizeof(*up_bank)) != 0 ||
        posix_memalign((void **)&down_bank, 64, down_blocks * sizeof(*down_bank)) != 0 ||
        posix_memalign((void **)&x, 64, x_elems * sizeof(*x)) != 0 ||
        posix_memalign((void **)&packed, 64, packed_elems * sizeof(*packed)) != 0) {
        fprintf(stderr, "moe-batch-bench: anepack allocation failed\n");
        exit(1);
    }

    fill_iq2_bank(gate_bank, gate_blocks, 101u);
    fill_iq2_bank(up_bank, gate_blocks, 201u);
    fill_q2_bank(down_bank, down_blocks, 301u);
    for (size_t i = 0; i < x_elems; i++) x[i] = sinf((float)i * 0.001f);

    for (int i = 0; i < cfg->warmup; i++) {
        materialize_ane_split3_pack_fp16(x, gate_bank, up_bank, down_bank,
                                         batch, in_dim, mid_dim, split, packed);
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        materialize_ane_split3_pack_fp16(x, gate_bank, up_bank, down_bank,
                                         batch, in_dim, mid_dim, split, packed);
    }
    const double sec = now_sec() - t0;
    volatile uint16_t sink = packed[0] ^ packed[packed_elems - 1];
    (void)sink;

    free(packed);
    free(x);
    free(down_bank);
    free(up_bank);
    free(gate_bank);

    const double ms = sec * 1000.0 / (double)cfg->iters;
    const double packed_gb = (double)(packed_elems * sizeof(uint16_t)) / 1.0e9;
    return (bench_result){
        .backend = "ane_split3_pack_cpu",
        .dtype = "iq2xxs_q2k_to_fp16",
        .batch = batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
        .iters = cfg->iters,
        .ms = ms,
        .gflops = packed_gb / (ms / 1000.0),
    };
}

static bench_result bench_ane_pack_materialize_gpu(const bench_config *cfg, int batch) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        const int split = 3;
        const char *pace_env = getenv("DS4_ANE_PACK_PACE_US");
        const useconds_t pace_us = pace_env ? (useconds_t)strtoul(pace_env, NULL, 10) : 0;
        if ((in_dim % QK_K) != 0 || (mid_dim % QK_K) != 0 || (mid_dim % split) != 0) {
            fprintf(stderr, "moe-batch-bench: anepackgpu requires --in/--mid multiples of 256 and --mid divisible by 3\n");
            exit(2);
        }

        ane_pack_gpu m = make_ane_pack_gpu(batch, in_dim, mid_dim, split);
        for (int i = 0; i < cfg->warmup; i++) {
            ane_pack_gpu_once(&m);
            if (pace_us) usleep(pace_us);
        }

        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            ane_pack_gpu_once(&m);
            if (pace_us) usleep(pace_us);
        }
        const double sec = now_sec() - t0;

        const size_t packed_elems =
            (size_t)batch * (size_t)in_dim + 3u * (size_t)in_dim * (size_t)mid_dim;
        volatile uint16_t sink =
            ((uint16_t *)[m.packed contents])[0] ^ ((uint16_t *)[m.packed contents])[packed_elems - 1];
        (void)sink;

        const double ms = sec * 1000.0 / (double)cfg->iters;
        const double packed_gb = (double)(packed_elems * sizeof(uint16_t)) / 1.0e9;
        return (bench_result){
            .backend = "ane_split3_pack_gpu",
            .dtype = "iq2xxs_q2k_to_fp16",
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .iters = cfg->iters,
            .ms = ms,
            .gflops = packed_gb / (ms / 1000.0),
        };
    }
}

static dequant_part_result make_dequant_part_result(
        const char *backend,
        const char *tensor,
        const char *quant,
        const char *out_dtype,
        int in_dim,
        int mid_dim,
        int iters,
        size_t quant_bytes,
        size_t dense_bytes,
        double ms) {
    return (dequant_part_result){
        .backend = backend,
        .tensor = tensor,
        .quant = quant,
        .out_dtype = out_dtype,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
        .iters = iters,
        .quant_mb = (double)quant_bytes / 1.0e6,
        .dense_mb = (double)dense_bytes / 1.0e6,
        .compression = (double)dense_bytes / (double)quant_bytes,
        .ms = ms,
        .dense_gbps = ((double)dense_bytes / 1.0e9) / (ms / 1000.0),
    };
}

static dequant_part_result bench_dequant_part_cpu(
        const bench_config *cfg,
        const char *tensor,
        bool is_iq2,
        const void *src,
        uint16_t *dst,
        size_t quant_bytes,
        size_t dense_bytes) {
    const int in_dim = cfg->in_dim;
    const int mid_dim = cfg->mid_dim;
    for (int i = 0; i < cfg->warmup; i++) {
        if (is_iq2) {
            dequant_iq2_xxs_transpose_bf16((const bench_block_iq2_xxs *)src, mid_dim, in_dim, dst);
        } else {
            dequant_q2_k_transpose_bf16((const bench_block_q2_K *)src, in_dim, mid_dim, dst);
        }
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        if (is_iq2) {
            dequant_iq2_xxs_transpose_bf16((const bench_block_iq2_xxs *)src, mid_dim, in_dim, dst);
        } else {
            dequant_q2_k_transpose_bf16((const bench_block_q2_K *)src, in_dim, mid_dim, dst);
        }
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)cfg->iters;
    volatile uint16_t sink = dst[0] ^ dst[(dense_bytes / sizeof(uint16_t)) - 1];
    (void)sink;
    return make_dequant_part_result("cpu_dequant", tensor,
                                    is_iq2 ? "IQ2_XXS" : "Q2_K",
                                    "bf16_transposed", in_dim, mid_dim,
                                    cfg->iters, quant_bytes, dense_bytes, ms);
}

static dequant_part_result bench_dequant_part_gpu(
        const bench_config *cfg,
        gpu_dequant_moe *m,
        const char *tensor,
        bool is_iq2,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        size_t quant_bytes,
        size_t dense_bytes) {
    for (int i = 0; i < cfg->warmup; i++) {
        if (is_iq2) {
            gpu_dequant_iq2_tensor_once(m, src, dst);
        } else {
            gpu_dequant_q2_tensor_once(m);
        }
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        if (is_iq2) {
            gpu_dequant_iq2_tensor_once(m, src, dst);
        } else {
            gpu_dequant_q2_tensor_once(m);
        }
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)cfg->iters;
    volatile uint16_t sink =
        ((uint16_t *)[dst contents])[0] ^ ((uint16_t *)[dst contents])[(dense_bytes / sizeof(uint16_t)) - 1];
    (void)sink;
    return make_dequant_part_result("metal_dequant", tensor,
                                    is_iq2 ? "IQ2_XXS" : "Q2_K",
                                    "bf16_transposed", cfg->in_dim, cfg->mid_dim,
                                    cfg->iters, quant_bytes, dense_bytes, ms);
}

static dequant_part_result bench_dequant_part_gpu_i8(
        const bench_config *cfg,
        gpu_dequant_moe *m,
        const char *tensor,
        bool is_iq2,
        id<MTLBuffer> src,
        id<MTLBuffer> dst,
        size_t quant_bytes,
        size_t dense_bytes) {
    for (int i = 0; i < cfg->warmup; i++) {
        if (is_iq2) {
            gpu_dequant_iq2_tensor_i8_once(m, src, dst);
        } else {
            gpu_dequant_q2_tensor_i8_once(m);
        }
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        if (is_iq2) {
            gpu_dequant_iq2_tensor_i8_once(m, src, dst);
        } else {
            gpu_dequant_q2_tensor_i8_once(m);
        }
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)cfg->iters;
    volatile int8_t sink =
        ((int8_t *)[dst contents])[0] ^ ((int8_t *)[dst contents])[dense_bytes - 1];
    (void)sink;
    return make_dequant_part_result("metal_dequant", tensor,
                                    is_iq2 ? "IQ2_XXS" : "Q2_K",
                                    "int8_transposed_s0.125", cfg->in_dim, cfg->mid_dim,
                                    cfg->iters, quant_bytes, dense_bytes, ms);
}

static void print_dequant_part_header(FILE *out) {
    fprintf(out, "backend,tensor,quant,out_dtype,in_dim,mid_dim,iters,quant_mb,dense_mb,compression,ms,dense_gbps\n");
}

static void print_dequant_part_result(FILE *out, dequant_part_result r) {
    fprintf(out, "%s,%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.3f,%.6f,%.3f\n",
            r.backend, r.tensor, r.quant, r.out_dtype, r.in_dim, r.mid_dim,
            r.iters, r.quant_mb, r.dense_mb, r.compression, r.ms, r.dense_gbps);
    fflush(out);
}

static void bench_dequant_parts(const bench_config *cfg, FILE *out, FILE *csv) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        if ((in_dim % QK_K) != 0 || (mid_dim % QK_K) != 0) {
            fprintf(stderr, "moe-batch-bench: dequantparts requires --in and --mid multiples of %d\n", QK_K);
            exit(2);
        }

        const size_t gate_blocks = (size_t)mid_dim * (size_t)(in_dim / QK_K);
        const size_t down_blocks = (size_t)in_dim * (size_t)(mid_dim / QK_K);
        const size_t iq2_bytes = gate_blocks * sizeof(bench_block_iq2_xxs);
        const size_t q2_bytes = down_blocks * sizeof(bench_block_q2_K);
        const size_t dense_bytes = (size_t)in_dim * (size_t)mid_dim * sizeof(uint16_t);

        bench_block_iq2_xxs *gate_bank = NULL;
        bench_block_iq2_xxs *up_bank = NULL;
        bench_block_q2_K *down_bank = NULL;
        uint16_t *dst = NULL;
        if (posix_memalign((void **)&gate_bank, 64, iq2_bytes) != 0 ||
            posix_memalign((void **)&up_bank, 64, iq2_bytes) != 0 ||
            posix_memalign((void **)&down_bank, 64, q2_bytes) != 0 ||
            posix_memalign((void **)&dst, 64, dense_bytes) != 0) {
            fprintf(stderr, "moe-batch-bench: dequantparts allocation failed\n");
            exit(1);
        }
        fill_iq2_bank(gate_bank, gate_blocks, 101u);
        fill_iq2_bank(up_bank, gate_blocks, 201u);
        fill_q2_bank(down_bank, down_blocks, 301u);

        gpu_dequant_moe gpu = make_gpu_dequant_moe(in_dim, mid_dim);

        print_dequant_part_header(out);
        if (csv) print_dequant_part_header(csv);

        dequant_part_result rows[] = {
            bench_dequant_part_cpu(cfg, "W_gate", true, gate_bank, dst, iq2_bytes, dense_bytes),
            bench_dequant_part_cpu(cfg, "W_up", true, up_bank, dst, iq2_bytes, dense_bytes),
            bench_dequant_part_cpu(cfg, "W_down", false, down_bank, dst, q2_bytes, dense_bytes),
            bench_dequant_part_gpu(cfg, &gpu, "W_gate", true, gpu.gate_q, gpu.w_gate, iq2_bytes, dense_bytes),
            bench_dequant_part_gpu(cfg, &gpu, "W_up", true, gpu.up_q, gpu.w_up, iq2_bytes, dense_bytes),
            bench_dequant_part_gpu(cfg, &gpu, "W_down", false, gpu.down_q, gpu.w_down, q2_bytes, dense_bytes),
            bench_dequant_part_gpu_i8(cfg, &gpu, "W_gate", true, gpu.gate_q, gpu.w_gate_i8, iq2_bytes, dense_bytes / 2u),
            bench_dequant_part_gpu_i8(cfg, &gpu, "W_up", true, gpu.up_q, gpu.w_up_i8, iq2_bytes, dense_bytes / 2u),
            bench_dequant_part_gpu_i8(cfg, &gpu, "W_down", false, gpu.down_q, gpu.w_down_i8, q2_bytes, dense_bytes / 2u),
        };
        const size_t n_rows = sizeof(rows) / sizeof(rows[0]);
        for (size_t i = 0; i < n_rows; i++) {
            print_dequant_part_result(out, rows[i]);
            if (csv) print_dequant_part_result(csv, rows[i]);
        }

        free(dst);
        free(down_bank);
        free(up_bank);
        free(gate_bank);
    }
}

static void print_dequant_batch_header(FILE *out) {
    fprintf(out, "backend,experts,in_dim,mid_dim,iters,total_ms,ms_per_expert,output_mb,output_gbps\n");
}

static void print_dequant_batch_result(FILE *out, dequant_batch_result r) {
    fprintf(out, "%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f\n",
            r.backend, r.experts, r.in_dim, r.mid_dim, r.iters,
            r.total_ms, r.ms_per_expert, r.output_mb, r.output_gbps);
    fflush(out);
}

static void dequant_batch_separate_once(
        gpu_dequant_moe *m,
        id<MTLBuffer> gate_q,
        id<MTLBuffer> up_q,
        id<MTLBuffer> down_q,
        id<MTLBuffer> gate_out,
        id<MTLBuffer> up_out,
        id<MTLBuffer> down_out,
        int experts,
        size_t iq2_bytes,
        size_t q2_bytes,
        size_t out_bytes,
        bool one_command_buffer) {
    id<MTLCommandBuffer> cb = one_command_buffer ? [m->queue commandBuffer] : nil;
    for (int e = 0; e < experts; e++) {
        id<MTLCommandBuffer> local_cb = one_command_buffer ? cb : [m->queue commandBuffer];
        dispatch_dequant_iq2_offset(local_cb, m->iq2_i8_pipe,
                                    gate_q, (NSUInteger)e * iq2_bytes,
                                    m->grid, gate_out, (NSUInteger)e * out_bytes,
                                    (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_iq2_offset(local_cb, m->iq2_i8_pipe,
                                    up_q, (NSUInteger)e * iq2_bytes,
                                    m->grid, up_out, (NSUInteger)e * out_bytes,
                                    (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_q2_offset(local_cb, m->q2_i8_pipe,
                                   down_q, (NSUInteger)e * q2_bytes,
                                   down_out, (NSUInteger)e * out_bytes,
                                   (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
        if (!one_command_buffer) {
            [local_cb commit];
            [local_cb waitUntilCompleted];
            if (local_cb.error) {
                fprintf(stderr, "moe-batch-bench: dequantbatch separate command failed: %s\n",
                        [[local_cb.error description] UTF8String]);
                exit(1);
            }
        }
    }
    if (one_command_buffer) {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "moe-batch-bench: dequantbatch one-CB command failed: %s\n",
                    [[cb.error description] UTF8String]);
            exit(1);
        }
    }
}

static dequant_batch_result bench_dequant_batch_mode(
        const bench_config *cfg,
        gpu_dequant_moe *m,
        const char *backend,
        id<MTLBuffer> gate_q,
        id<MTLBuffer> up_q,
        id<MTLBuffer> down_q,
        id<MTLBuffer> gate_out,
        id<MTLBuffer> up_out,
        id<MTLBuffer> down_out,
        int experts,
        size_t iq2_bytes,
        size_t q2_bytes,
        size_t out_bytes,
        bool one_command_buffer) {
    for (int i = 0; i < cfg->warmup; i++) {
        dequant_batch_separate_once(m, gate_q, up_q, down_q, gate_out, up_out, down_out,
                                    experts, iq2_bytes, q2_bytes, out_bytes, one_command_buffer);
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        dequant_batch_separate_once(m, gate_q, up_q, down_q, gate_out, up_out, down_out,
                                    experts, iq2_bytes, q2_bytes, out_bytes, one_command_buffer);
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)cfg->iters;
    volatile int8_t sink =
        ((int8_t *)[gate_out contents])[0] ^
        ((int8_t *)[up_out contents])[(size_t)experts * out_bytes - 1] ^
        ((int8_t *)[down_out contents])[(size_t)experts * out_bytes - 1];
    (void)sink;
    const double output_mb = (double)((size_t)experts * 3u * out_bytes) / 1.0e6;
    return (dequant_batch_result){
        .backend = backend,
        .experts = experts,
        .in_dim = cfg->in_dim,
        .mid_dim = cfg->mid_dim,
        .iters = cfg->iters,
        .total_ms = ms,
        .ms_per_expert = ms / (double)experts,
        .output_mb = output_mb,
        .output_gbps = (output_mb / 1000.0) / (ms / 1000.0),
    };
}

static dequant_batch_result bench_dequant_batch_fused(
        const bench_config *cfg,
        gpu_dequant_moe *m,
        id<MTLBuffer> gate_q,
        id<MTLBuffer> up_q,
        id<MTLBuffer> down_q,
        id<MTLBuffer> gate_out,
        id<MTLBuffer> up_out,
        id<MTLBuffer> down_out,
        int experts,
        size_t out_bytes) {
    for (int i = 0; i < cfg->warmup; i++) {
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        dispatch_dequant_iq2_batched(cb, m->iq2_i8_batched_pipe, gate_q, m->grid, gate_out,
                                     (uint32_t)experts, (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_iq2_batched(cb, m->iq2_i8_batched_pipe, up_q, m->grid, up_out,
                                     (uint32_t)experts, (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_q2_batched(cb, m->q2_i8_batched_pipe, down_q, down_out,
                                    (uint32_t)experts, (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "moe-batch-bench: dequantbatch fused warmup failed: %s\n",
                    [[cb.error description] UTF8String]);
            exit(1);
        }
    }
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        dispatch_dequant_iq2_batched(cb, m->iq2_i8_batched_pipe, gate_q, m->grid, gate_out,
                                     (uint32_t)experts, (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_iq2_batched(cb, m->iq2_i8_batched_pipe, up_q, m->grid, up_out,
                                     (uint32_t)experts, (uint32_t)m->mid_dim, (uint32_t)m->in_dim);
        dispatch_dequant_q2_batched(cb, m->q2_i8_batched_pipe, down_q, down_out,
                                    (uint32_t)experts, (uint32_t)m->in_dim, (uint32_t)m->mid_dim);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "moe-batch-bench: dequantbatch fused command failed: %s\n",
                    [[cb.error description] UTF8String]);
            exit(1);
        }
    }
    const double ms = (now_sec() - t0) * 1000.0 / (double)cfg->iters;
    volatile int8_t sink =
        ((int8_t *)[gate_out contents])[0] ^
        ((int8_t *)[up_out contents])[(size_t)experts * out_bytes - 1] ^
        ((int8_t *)[down_out contents])[(size_t)experts * out_bytes - 1];
    (void)sink;
    const double output_mb = (double)((size_t)experts * 3u * out_bytes) / 1.0e6;
    return (dequant_batch_result){
        .backend = "metal_dequant_i8_batched_kernel",
        .experts = experts,
        .in_dim = cfg->in_dim,
        .mid_dim = cfg->mid_dim,
        .iters = cfg->iters,
        .total_ms = ms,
        .ms_per_expert = ms / (double)experts,
        .output_mb = output_mb,
        .output_gbps = (output_mb / 1000.0) / (ms / 1000.0),
    };
}

static void bench_dequant_batch(const bench_config *cfg, FILE *out, FILE *csv) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        const bool fused_only = getenv("DS4_DEQUANT_BATCH_FUSED_ONLY") &&
                                atoi(getenv("DS4_DEQUANT_BATCH_FUSED_ONLY")) != 0;
        if ((in_dim % QK_K) != 0 || (mid_dim % QK_K) != 0) {
            fprintf(stderr, "moe-batch-bench: dequantbatch requires --in and --mid multiples of %d\n", QK_K);
            exit(2);
        }

        gpu_dequant_moe gpu = make_gpu_dequant_moe(in_dim, mid_dim);
        print_dequant_batch_header(out);
        if (csv) print_dequant_batch_header(csv);

        for (int bi = 0; bi < cfg->n_batches; bi++) {
            const int experts = cfg->batches[bi];
            if (experts < 1) continue;
            const size_t gate_blocks = (size_t)mid_dim * (size_t)(in_dim / QK_K);
            const size_t down_blocks = (size_t)in_dim * (size_t)(mid_dim / QK_K);
            const size_t iq2_bytes = gate_blocks * sizeof(bench_block_iq2_xxs);
            const size_t q2_bytes = down_blocks * sizeof(bench_block_q2_K);
            const size_t out_bytes = (size_t)in_dim * (size_t)mid_dim;

            id<MTLBuffer> gate_q = [gpu.dev newBufferWithLength:(NSUInteger)experts * iq2_bytes
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> up_q = [gpu.dev newBufferWithLength:(NSUInteger)experts * iq2_bytes
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> down_q = [gpu.dev newBufferWithLength:(NSUInteger)experts * q2_bytes
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> gate_out = [gpu.dev newBufferWithLength:(NSUInteger)experts * out_bytes
                                                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> up_out = [gpu.dev newBufferWithLength:(NSUInteger)experts * out_bytes
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> down_out = [gpu.dev newBufferWithLength:(NSUInteger)experts * out_bytes
                                                          options:MTLResourceStorageModeShared];
            if (!gate_q || !up_q || !down_q || !gate_out || !up_out || !down_out) {
                fprintf(stderr, "moe-batch-bench: dequantbatch allocation failed for experts=%d\n", experts);
                exit(1);
            }
            fill_iq2_bank((bench_block_iq2_xxs *)[gate_q contents], gate_blocks * (size_t)experts, 101u);
            fill_iq2_bank((bench_block_iq2_xxs *)[up_q contents], gate_blocks * (size_t)experts, 201u);
            fill_q2_bank((bench_block_q2_K *)[down_q contents], down_blocks * (size_t)experts, 301u);

            dequant_batch_result fused =
                bench_dequant_batch_fused(cfg, &gpu,
                                          gate_q, up_q, down_q, gate_out, up_out, down_out,
                                          experts, out_bytes);
            if (!fused_only) {
                dequant_batch_result looped =
                    bench_dequant_batch_mode(cfg, &gpu, "metal_dequant_i8_loop_cb",
                                             gate_q, up_q, down_q, gate_out, up_out, down_out,
                                             experts, iq2_bytes, q2_bytes, out_bytes, false);
                dequant_batch_result batched =
                    bench_dequant_batch_mode(cfg, &gpu, "metal_dequant_i8_one_cb",
                                             gate_q, up_q, down_q, gate_out, up_out, down_out,
                                             experts, iq2_bytes, q2_bytes, out_bytes, true);
                print_dequant_batch_result(out, looped);
                print_dequant_batch_result(out, batched);
                if (csv) {
                    print_dequant_batch_result(csv, looped);
                    print_dequant_batch_result(csv, batched);
                }
            }
            print_dequant_batch_result(out, fused);
            if (csv) {
                print_dequant_batch_result(csv, fused);
            }
        }
    }
}

static bench_result bench_amx_quant_bf16(
        const bench_config *cfg,
        int batch,
        bool dequant_each_iter) {
    const int in_dim = cfg->in_dim;
    const int mid_dim = cfg->mid_dim;
    const size_t gate_blocks =
        (size_t)mid_dim * (size_t)(in_dim / QK_K);
    const size_t down_blocks =
        (size_t)in_dim * (size_t)(mid_dim / QK_K);

    bench_block_iq2_xxs *gate_bank = NULL;
    bench_block_iq2_xxs *up_bank = NULL;
    bench_block_q2_K *down_bank = NULL;
    if (posix_memalign((void **)&gate_bank, 64, gate_blocks * sizeof(*gate_bank)) != 0 ||
        posix_memalign((void **)&up_bank, 64, gate_blocks * sizeof(*up_bank)) != 0 ||
        posix_memalign((void **)&down_bank, 64, down_blocks * sizeof(*down_bank)) != 0) {
        fprintf(stderr, "moe-batch-bench: quant AMX weight allocation failed\n");
        exit(1);
    }

    fill_iq2_bank(gate_bank, gate_blocks, 101u);
    fill_iq2_bank(up_bank, gate_blocks, 201u);
    fill_q2_bank(down_bank, down_blocks, 301u);

    void *x = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);
    void *w_gate = alloc_dense((size_t)in_dim * (size_t)mid_dim, DENSE_BF16);
    void *w_up = alloc_dense((size_t)in_dim * (size_t)mid_dim, DENSE_BF16);
    void *w_down = alloc_dense((size_t)mid_dim * (size_t)in_dim, DENSE_BF16);
    void *gate = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    void *up = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    void *mid = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    void *out = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);

    fill_dense(x, (size_t)batch * (size_t)in_dim, 1u, DENSE_BF16);
    dequant_moe_expert_to_bf16(gate_bank, up_bank, down_bank, in_dim, mid_dim,
                               (uint16_t *)w_gate, (uint16_t *)w_up, (uint16_t *)w_down);

    bnns_mlp mlp = make_bnns_mlp(batch, in_dim, mid_dim,
                                 x, w_gate, w_up, w_down,
                                 gate, up, mid, out, DENSE_BF16);

    for (int i = 0; i < cfg->warmup; i++) {
        if (dequant_each_iter) {
            dequant_moe_expert_to_bf16(gate_bank, up_bank, down_bank, in_dim, mid_dim,
                                       (uint16_t *)w_gate, (uint16_t *)w_up, (uint16_t *)w_down);
        }
        bnns_mlp_once(&mlp, batch, mid_dim);
    }

    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        if (dequant_each_iter) {
            dequant_moe_expert_to_bf16(gate_bank, up_bank, down_bank, in_dim, mid_dim,
                                       (uint16_t *)w_gate, (uint16_t *)w_up, (uint16_t *)w_down);
        }
        bnns_mlp_once(&mlp, batch, mid_dim);
    }
    const double sec = now_sec() - t0;

    bnns_mlp_free(&mlp);
    free(out);
    free(mid);
    free(up);
    free(gate);
    free(w_down);
    free(w_up);
    free(w_gate);
    free(x);
    free(down_bank);
    free(up_bank);
    free(gate_bank);

    const double ms = sec * 1000.0 / (double)cfg->iters;
    bench_result r = {
        .backend = "amx_bnns_quant",
        .dtype = dequant_each_iter ? "iq2xxs_q2k_dequant_bf16" : "iq2xxs_q2k_cached_bf16",
        .batch = batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
        .iters = cfg->iters,
        .ms = ms,
        .gflops = mlp_gflop(batch, in_dim, mid_dim) / (ms / 1000.0),
    };
    return r;
}

static bench_result bench_amx_quant_gpu_dequant_bf16(const bench_config *cfg, int batch) {
    @autoreleasepool {
        const int in_dim = cfg->in_dim;
        const int mid_dim = cfg->mid_dim;
        gpu_dequant_moe dq = make_gpu_dequant_moe(in_dim, mid_dim);

        void *x = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);
        void *gate = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
        void *up = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
        void *mid = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
        void *out = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);

        fill_dense(x, (size_t)batch * (size_t)in_dim, 1u, DENSE_BF16);

        bnns_mlp mlp = make_bnns_mlp(batch, in_dim, mid_dim,
                                     x,
                                     [dq.w_gate contents],
                                     [dq.w_up contents],
                                     [dq.w_down contents],
                                     gate, up, mid, out, DENSE_BF16);

        for (int i = 0; i < cfg->warmup; i++) {
            gpu_dequant_moe_once(&dq);
            bnns_mlp_once(&mlp, batch, mid_dim);
        }

        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            gpu_dequant_moe_once(&dq);
            bnns_mlp_once(&mlp, batch, mid_dim);
        }
        const double sec = now_sec() - t0;

        bnns_mlp_free(&mlp);
        free(out);
        free(mid);
        free(up);
        free(gate);
        free(x);

        const double ms = sec * 1000.0 / (double)cfg->iters;
        bench_result r = {
            .backend = "amx_bnns_quant",
            .dtype = "iq2xxs_q2k_gpu_dequant_bf16",
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
            .iters = cfg->iters,
            .ms = ms,
            .gflops = mlp_gflop(batch, in_dim, mid_dim) / (ms / 1000.0),
        };
        return r;
    }
}

typedef struct {
    int batch;
    int in_dim;
    int mid_dim;
    void *x;
    void *w_gate;
    void *w_up;
    void *w_down;
    void *gate;
    void *up;
    void *mid;
    void *out;
    bool owns_weights;
    bnns_mlp mlp;
} amx_cached_moe_ctx;

static amx_cached_moe_ctx make_amx_cached_moe_with_weights(
        int batch,
        int in_dim,
        int mid_dim,
        void *w_gate,
        void *w_up,
        void *w_down,
        uint32_t seed_base) {
    amx_cached_moe_ctx c = {
        .batch = batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
        .w_gate = w_gate,
        .w_up = w_up,
        .w_down = w_down,
        .owns_weights = false,
    };
    c.x = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);
    c.gate = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    c.up = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    c.mid = alloc_dense((size_t)batch * (size_t)mid_dim, DENSE_BF16);
    c.out = alloc_dense((size_t)batch * (size_t)in_dim, DENSE_BF16);
    fill_dense(c.x, (size_t)batch * (size_t)in_dim, seed_base + 1u, DENSE_BF16);
    c.mlp = make_bnns_mlp(batch, in_dim, mid_dim,
                          c.x, c.w_gate, c.w_up, c.w_down,
                          c.gate, c.up, c.mid, c.out, DENSE_BF16);
    return c;
}

static void amx_cached_moe_once(amx_cached_moe_ctx *c) {
    bnns_mlp_once(&c->mlp, c->batch, c->mid_dim);
}

static void free_amx_cached_moe(amx_cached_moe_ctx *c) {
    bnns_mlp_free(&c->mlp);
    free(c->out);
    free(c->mid);
    free(c->up);
    free(c->gate);
    free(c->x);
    if (c->owns_weights) {
        free(c->w_down);
        free(c->w_up);
        free(c->w_gate);
    }
    memset(c, 0, sizeof(*c));
}

static void free_ds4_tensor(ds4_gpu_tensor **t) {
    if (*t) {
        ds4_gpu_tensor_free(*t);
        *t = NULL;
    }
}

typedef struct {
    uint32_t batch;
    uint32_t in_dim;
    uint32_t mid_dim;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *gate_bank;
    ds4_gpu_tensor *up_bank;
    ds4_gpu_tensor *down_bank;
    bool mid_is_f16;
} ds4_moe_block_ctx;

static void ds4_moe_block_once(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *gate_bank,
        ds4_gpu_tensor *up_bank,
        ds4_gpu_tensor *down_bank,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t in_dim,
        uint32_t mid_dim,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x,
        uint32_t batch,
        bool *mid_is_f16) {
    if (!ds4_gpu_routed_moe_expert_banked_batch_tensor(out,
                                                       gate,
                                                       up,
                                                       mid,
                                                       gate_bank,
                                                       up_bank,
                                                       down_bank,
                                                       DS4_METAL_TENSOR_IQ2_XXS,
                                                       DS4_METAL_TENSOR_Q2_K,
                                                       gate_expert_bytes,
                                                       gate_row_bytes,
                                                       down_expert_bytes,
                                                       down_row_bytes,
                                                       in_dim,
                                                       mid_dim,
                                                       in_dim,
                                                       selected,
                                                       weights,
                                                       DS4_SWIGLU_CLAMP_EXP,
                                                       x,
                                                       batch,
                                                       mid_is_f16)) {
        fprintf(stderr, "moe-batch-bench: DS4 Metal MoE block failed for batch %u\n", batch);
        exit(1);
    }
}

static ds4_moe_block_ctx make_ds4_moe_block_ctx(int batch, int in_dim_i, int mid_dim_i, uint32_t seed_base) {
    const uint32_t in_dim = (uint32_t)in_dim_i;
    const uint32_t mid_dim = (uint32_t)mid_dim_i;
    if ((in_dim % QK_K) != 0 || (mid_dim % QK_K) != 0) {
        fprintf(stderr, "moe-batch-bench: ds4 backend requires --in and --mid to be multiples of %d\n",
                QK_K);
        exit(2);
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "moe-batch-bench: failed to initialize DS4 Metal backend\n");
        exit(1);
    }

    ds4_moe_block_ctx c = {
        .batch = (uint32_t)batch,
        .in_dim = in_dim,
        .mid_dim = mid_dim,
    };
    const uint64_t gate_blocks_per_row = in_dim / QK_K;
    const uint64_t down_blocks_per_row = mid_dim / QK_K;
    c.gate_row_bytes = checked_bytes(gate_blocks_per_row, sizeof(bench_block_iq2_xxs), "gate row");
    c.down_row_bytes = checked_bytes(down_blocks_per_row, sizeof(bench_block_q2_K), "down row");
    c.gate_expert_bytes = checked_bytes(mid_dim, c.gate_row_bytes, "gate expert");
    c.down_expert_bytes = checked_bytes(in_dim, c.down_row_bytes, "down expert");

    const uint64_t batch_u64 = (uint64_t)batch;
    const uint64_t x_bytes = checked_bytes(checked_bytes(batch_u64, in_dim, "x rows"), sizeof(float), "x");
    const uint64_t mid_rows = checked_bytes(batch_u64, mid_dim, "mid rows");
    const uint64_t mid_bytes = checked_bytes(mid_rows, sizeof(float), "mid");
    const uint64_t out_bytes = checked_bytes(checked_bytes(batch_u64, in_dim, "out rows"), sizeof(float), "out");
    const uint64_t selected_bytes = checked_bytes(batch_u64, sizeof(int32_t), "selected");
    const uint64_t weight_bytes = checked_bytes(batch_u64, sizeof(float), "weights");

    c.x = alloc_ds4_tensor(x_bytes, "x");
    c.gate = alloc_ds4_tensor(mid_bytes, "gate");
    c.up = alloc_ds4_tensor(mid_bytes, "up");
    c.mid = alloc_ds4_tensor(mid_bytes, "mid");
    c.out = alloc_ds4_tensor(out_bytes, "out");
    c.selected = alloc_ds4_tensor(selected_bytes, "selected");
    c.weights = alloc_ds4_tensor(weight_bytes, "weights");
    c.gate_bank = alloc_ds4_tensor(c.gate_expert_bytes, "gate_bank");
    c.up_bank = alloc_ds4_tensor(c.gate_expert_bytes, "up_bank");
    c.down_bank = alloc_ds4_tensor(c.down_expert_bytes, "down_bank");

    fill_f32((float *)ds4_tensor_contents_checked(c.x, "x"),
             (size_t)batch * (size_t)in_dim,
             seed_base + 11u);
    fill_i32_zero((int32_t *)ds4_tensor_contents_checked(c.selected, "selected"),
                  (size_t)batch);
    fill_f32_one((float *)ds4_tensor_contents_checked(c.weights, "weights"),
                 (size_t)batch);
    fill_iq2_bank((bench_block_iq2_xxs *)ds4_tensor_contents_checked(c.gate_bank, "gate_bank"),
                  (size_t)(c.gate_expert_bytes / sizeof(bench_block_iq2_xxs)),
                  seed_base + 21u);
    fill_iq2_bank((bench_block_iq2_xxs *)ds4_tensor_contents_checked(c.up_bank, "up_bank"),
                  (size_t)(c.gate_expert_bytes / sizeof(bench_block_iq2_xxs)),
                  seed_base + 31u);
    fill_q2_bank((bench_block_q2_K *)ds4_tensor_contents_checked(c.down_bank, "down_bank"),
                 (size_t)(c.down_expert_bytes / sizeof(bench_block_q2_K)),
                 seed_base + 41u);
    return c;
}

static void ds4_moe_block_ctx_once(ds4_moe_block_ctx *c) {
    ds4_moe_block_once(c->out,
                       c->gate,
                       c->up,
                       c->mid,
                       c->gate_bank,
                       c->up_bank,
                       c->down_bank,
                       c->gate_expert_bytes,
                       c->gate_row_bytes,
                       c->down_expert_bytes,
                       c->down_row_bytes,
                       c->in_dim,
                       c->mid_dim,
                       c->selected,
                       c->weights,
                       c->x,
                       c->batch,
                       &c->mid_is_f16);
}

static void free_ds4_moe_block_ctx(ds4_moe_block_ctx *c) {
    free_ds4_tensor(&c->down_bank);
    free_ds4_tensor(&c->up_bank);
    free_ds4_tensor(&c->gate_bank);
    free_ds4_tensor(&c->weights);
    free_ds4_tensor(&c->selected);
    free_ds4_tensor(&c->out);
    free_ds4_tensor(&c->mid);
    free_ds4_tensor(&c->up);
    free_ds4_tensor(&c->gate);
    free_ds4_tensor(&c->x);
    memset(c, 0, sizeof(*c));
}

static bench_result bench_ds4_moe_block(const bench_config *cfg, int batch) {
    ds4_moe_block_ctx c = make_ds4_moe_block_ctx(batch, cfg->in_dim, cfg->mid_dim, 0u);
    for (int i = 0; i < cfg->warmup; i++) {
        ds4_moe_block_ctx_once(&c);
    }

    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) {
        ds4_moe_block_ctx_once(&c);
    }
    const double sec = now_sec() - t0;

    const bool mid_is_f16 = c.mid_is_f16;
    free_ds4_moe_block_ctx(&c);

    const double ms = sec * 1000.0 / (double)cfg->iters;
    bench_result r = {
        .backend = "gpu_ds4_moe_block",
        .dtype = mid_is_f16 ? "iq2xxs_q2k_mid_f16" : "iq2xxs_q2k_mid_f32",
        .batch = batch,
        .in_dim = cfg->in_dim,
        .mid_dim = cfg->mid_dim,
        .iters = cfg->iters,
        .ms = ms,
        .gflops = mlp_gflop(batch, cfg->in_dim, cfg->mid_dim) / (ms / 1000.0),
    };
    return r;
}

typedef enum {
    SYNTH_Q_FP4_LUT,
    SYNTH_Q_INT8_FIXED,
} synth_q_kind;

typedef struct {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> fp4_half_pipe;
    id<MTLComputePipelineState> fp4_float_pipe;
    id<MTLComputePipelineState> int8_half_pipe;
    id<MTLComputePipelineState> int8_float_pipe;
    id<MTLComputePipelineState> silu_pipe;
    id<MTLBuffer> x_half;
    id<MTLBuffer> gate_fp4;
    id<MTLBuffer> up_fp4;
    id<MTLBuffer> down_fp4;
    id<MTLBuffer> gate_i8;
    id<MTLBuffer> up_i8;
    id<MTLBuffer> down_i8;
    id<MTLBuffer> lut;
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> mid;
    id<MTLBuffer> out;
    int batch;
    int in_dim;
    int mid_dim;
} synth_quant_ctx;

static NSString *synth_quant_metal_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "static inline half fp4_value(device const half *lut, device const uchar *w, uint idx) {\n"
            "    uchar p = w[idx >> 1];\n"
            "    uchar q = (idx & 1u) ? (p >> 4) : (p & 15u);\n"
            "    return lut[q];\n"
            "}\n"
            "kernel void synth_fp4_halfx(device const half *x [[buffer(0)]],\n"
            "                            device const uchar *w [[buffer(1)]],\n"
            "                            device const half *lut [[buffer(2)]],\n"
            "                            device float *y [[buffer(3)]],\n"
            "                            constant uint &batch [[buffer(4)]],\n"
            "                            constant uint &kdim [[buffer(5)]],\n"
            "                            constant uint &ndim [[buffer(6)]],\n"
            "                            uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = batch * ndim;\n"
            "    if (gid >= total) return;\n"
            "    uint b = gid / ndim;\n"
            "    uint n = gid - b * ndim;\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < kdim; k++) acc += float(x[b * kdim + k]) * float(fp4_value(lut, w, k * ndim + n));\n"
            "    y[gid] = acc;\n"
            "}\n"
            "kernel void synth_fp4_floatx(device const float *x [[buffer(0)]],\n"
            "                             device const uchar *w [[buffer(1)]],\n"
            "                             device const half *lut [[buffer(2)]],\n"
            "                             device float *y [[buffer(3)]],\n"
            "                             constant uint &batch [[buffer(4)]],\n"
            "                             constant uint &kdim [[buffer(5)]],\n"
            "                             constant uint &ndim [[buffer(6)]],\n"
            "                             uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = batch * ndim;\n"
            "    if (gid >= total) return;\n"
            "    uint b = gid / ndim;\n"
            "    uint n = gid - b * ndim;\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < kdim; k++) acc += x[b * kdim + k] * float(fp4_value(lut, w, k * ndim + n));\n"
            "    y[gid] = acc;\n"
            "}\n"
            "kernel void synth_int8_halfx(device const half *x [[buffer(0)]],\n"
            "                             device const char *w [[buffer(1)]],\n"
            "                             constant float &scale [[buffer(2)]],\n"
            "                             device float *y [[buffer(3)]],\n"
            "                             constant uint &batch [[buffer(4)]],\n"
            "                             constant uint &kdim [[buffer(5)]],\n"
            "                             constant uint &ndim [[buffer(6)]],\n"
            "                             uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = batch * ndim;\n"
            "    if (gid >= total) return;\n"
            "    uint b = gid / ndim;\n"
            "    uint n = gid - b * ndim;\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < kdim; k++) acc += float(x[b * kdim + k]) * float(w[k * ndim + n]) * scale;\n"
            "    y[gid] = acc;\n"
            "}\n"
            "kernel void synth_int8_floatx(device const float *x [[buffer(0)]],\n"
            "                              device const char *w [[buffer(1)]],\n"
            "                              constant float &scale [[buffer(2)]],\n"
            "                              device float *y [[buffer(3)]],\n"
            "                              constant uint &batch [[buffer(4)]],\n"
            "                              constant uint &kdim [[buffer(5)]],\n"
            "                              constant uint &ndim [[buffer(6)]],\n"
            "                              uint gid [[thread_position_in_grid]]) {\n"
            "    uint total = batch * ndim;\n"
            "    if (gid >= total) return;\n"
            "    uint b = gid / ndim;\n"
            "    uint n = gid - b * ndim;\n"
            "    float acc = 0.0f;\n"
            "    for (uint k = 0; k < kdim; k++) acc += x[b * kdim + k] * float(w[k * ndim + n]) * scale;\n"
            "    y[gid] = acc;\n"
            "}\n"
            "kernel void synth_silu_mul(device const float *gate [[buffer(0)]],\n"
            "                           device const float *up [[buffer(1)]],\n"
            "                           device float *mid [[buffer(2)]],\n"
            "                           constant uint &total [[buffer(3)]],\n"
            "                           uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid >= total) return;\n"
            "    float g = gate[gid];\n"
            "    mid[gid] = (g / (1.0f + fast::exp(-g))) * up[gid];\n"
            "}\n";
}

static id<MTLComputePipelineState> synth_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
        fprintf(stderr, "moe-batch-bench: failed to create synthetic pipeline %s: %s\n",
                [name UTF8String], [[err description] UTF8String]);
        exit(1);
    }
    return p;
}

static void fill_packed_fp4(id<MTLBuffer> buf, size_t n_values, uint32_t seed) {
    uint8_t *p = (uint8_t *)[buf contents];
    uint32_t s = seed;
    for (size_t i = 0; i < (n_values + 1u) / 2u; i++) {
        uint8_t lo = (uint8_t)(rng_next(&s) & 15u);
        uint8_t hi = (uint8_t)(rng_next(&s) & 15u);
        p[i] = (uint8_t)(lo | (hi << 4));
    }
}

static void fill_int8_weights(id<MTLBuffer> buf, size_t n_values, uint32_t seed) {
    int8_t *p = (int8_t *)[buf contents];
    uint32_t s = seed;
    for (size_t i = 0; i < n_values; i++) {
        p[i] = (int8_t)((int)(rng_next(&s) % 255u) - 127);
    }
}

static synth_quant_ctx make_synth_quant_ctx(int batch, int in_dim, int mid_dim) {
    @autoreleasepool {
        synth_quant_ctx c = {
            .batch = batch,
            .in_dim = in_dim,
            .mid_dim = mid_dim,
        };
        c.dev = MTLCreateSystemDefaultDevice();
        if (!c.dev) {
            fprintf(stderr, "moe-batch-bench: Metal is unavailable\n");
            exit(1);
        }
        c.queue = [c.dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [c.dev newLibraryWithSource:synth_quant_metal_source()
                                                 options:nil
                                                   error:&err];
        if (!lib) {
            fprintf(stderr, "moe-batch-bench: failed to compile synthetic kernels: %s\n",
                    [[err description] UTF8String]);
            exit(1);
        }
        c.fp4_half_pipe = synth_pipe(c.dev, lib, @"synth_fp4_halfx");
        c.fp4_float_pipe = synth_pipe(c.dev, lib, @"synth_fp4_floatx");
        c.int8_half_pipe = synth_pipe(c.dev, lib, @"synth_int8_halfx");
        c.int8_float_pipe = synth_pipe(c.dev, lib, @"synth_int8_floatx");
        c.silu_pipe = synth_pipe(c.dev, lib, @"synth_silu_mul");

        const size_t gate_values = (size_t)in_dim * (size_t)mid_dim;
        const size_t down_values = (size_t)mid_dim * (size_t)in_dim;
        const size_t x_values = (size_t)batch * (size_t)in_dim;
        const size_t mid_values = (size_t)batch * (size_t)mid_dim;
        const size_t out_values = (size_t)batch * (size_t)in_dim;
        c.x_half = [c.dev newBufferWithLength:x_values * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        c.gate_fp4 = [c.dev newBufferWithLength:(gate_values + 1u) / 2u options:MTLResourceStorageModeShared];
        c.up_fp4 = [c.dev newBufferWithLength:(gate_values + 1u) / 2u options:MTLResourceStorageModeShared];
        c.down_fp4 = [c.dev newBufferWithLength:(down_values + 1u) / 2u options:MTLResourceStorageModeShared];
        c.gate_i8 = [c.dev newBufferWithLength:gate_values options:MTLResourceStorageModeShared];
        c.up_i8 = [c.dev newBufferWithLength:gate_values options:MTLResourceStorageModeShared];
        c.down_i8 = [c.dev newBufferWithLength:down_values options:MTLResourceStorageModeShared];
        c.lut = [c.dev newBufferWithLength:16u * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        c.gate = [c.dev newBufferWithLength:mid_values * sizeof(float) options:MTLResourceStorageModeShared];
        c.up = [c.dev newBufferWithLength:mid_values * sizeof(float) options:MTLResourceStorageModeShared];
        c.mid = [c.dev newBufferWithLength:mid_values * sizeof(float) options:MTLResourceStorageModeShared];
        c.out = [c.dev newBufferWithLength:out_values * sizeof(float) options:MTLResourceStorageModeShared];
        if (!c.queue || !c.x_half || !c.gate_fp4 || !c.up_fp4 || !c.down_fp4 ||
            !c.gate_i8 || !c.up_i8 || !c.down_i8 || !c.lut || !c.gate || !c.up ||
            !c.mid || !c.out) {
            fprintf(stderr, "moe-batch-bench: failed to allocate synthetic buffers\n");
            exit(1);
        }

        fill_dense([c.x_half contents], x_values, 11u, DENSE_FP16);
        static const float lut_f32[16] = {
            -1.0000f, -0.6962f, -0.5251f, -0.3949f,
            -0.2844f, -0.1848f, -0.0911f, -0.0300f,
             0.0300f,  0.0911f,  0.1848f,  0.2844f,
             0.3949f,  0.5251f,  0.6962f,  1.0000f,
        };
        uint16_t *lut = (uint16_t *)[c.lut contents];
        for (int i = 0; i < 16; i++) lut[i] = fp16_bits(lut_f32[i]);
        fill_packed_fp4(c.gate_fp4, gate_values, 101u);
        fill_packed_fp4(c.up_fp4, gate_values, 102u);
        fill_packed_fp4(c.down_fp4, down_values, 103u);
        fill_int8_weights(c.gate_i8, gate_values, 201u);
        fill_int8_weights(c.up_i8, gate_values, 202u);
        fill_int8_weights(c.down_i8, down_values, 203u);
        return c;
    }
}

static void synth_dispatch_matmul(id<MTLCommandBuffer> cb,
                                  id<MTLComputePipelineState> pipe,
                                  id<MTLBuffer> x,
                                  id<MTLBuffer> w,
                                  id<MTLBuffer> aux,
                                  id<MTLBuffer> y,
                                  uint32_t batch,
                                  uint32_t kdim,
                                  uint32_t ndim,
                                  bool aux_is_scale) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    if (aux_is_scale) {
        const float scale = 1.0f / 127.0f;
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    } else {
        [enc setBuffer:aux offset:0 atIndex:2];
    }
    [enc setBuffer:y offset:0 atIndex:3];
    [enc setBytes:&batch length:sizeof(batch) atIndex:4];
    [enc setBytes:&kdim length:sizeof(kdim) atIndex:5];
    [enc setBytes:&ndim length:sizeof(ndim) atIndex:6];
    const NSUInteger total = (NSUInteger)batch * (NSUInteger)ndim;
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 128u) tg = 128u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void synth_dispatch_silu(id<MTLCommandBuffer> cb,
                                synth_quant_ctx *c,
                                uint32_t total) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c->silu_pipe];
    [enc setBuffer:c->gate offset:0 atIndex:0];
    [enc setBuffer:c->up offset:0 atIndex:1];
    [enc setBuffer:c->mid offset:0 atIndex:2];
    [enc setBytes:&total length:sizeof(total) atIndex:3];
    NSUInteger tg = c->silu_pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 256u) tg = 256u;
    if (tg == 0u) tg = 1u;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
}

static void synth_quant_once(synth_quant_ctx *c, synth_q_kind kind) {
    id<MTLCommandBuffer> cb = [c->queue commandBuffer];
    const uint32_t batch = (uint32_t)c->batch;
    const uint32_t in_dim = (uint32_t)c->in_dim;
    const uint32_t mid_dim = (uint32_t)c->mid_dim;
    if (kind == SYNTH_Q_FP4_LUT) {
        synth_dispatch_matmul(cb, c->fp4_half_pipe, c->x_half, c->gate_fp4, c->lut,
                              c->gate, batch, in_dim, mid_dim, false);
        synth_dispatch_matmul(cb, c->fp4_half_pipe, c->x_half, c->up_fp4, c->lut,
                              c->up, batch, in_dim, mid_dim, false);
        synth_dispatch_silu(cb, c, batch * mid_dim);
        synth_dispatch_matmul(cb, c->fp4_float_pipe, c->mid, c->down_fp4, c->lut,
                              c->out, batch, mid_dim, in_dim, false);
    } else {
        synth_dispatch_matmul(cb, c->int8_half_pipe, c->x_half, c->gate_i8, nil,
                              c->gate, batch, in_dim, mid_dim, true);
        synth_dispatch_matmul(cb, c->int8_half_pipe, c->x_half, c->up_i8, nil,
                              c->up, batch, in_dim, mid_dim, true);
        synth_dispatch_silu(cb, c, batch * mid_dim);
        synth_dispatch_matmul(cb, c->int8_float_pipe, c->mid, c->down_i8, nil,
                              c->out, batch, mid_dim, in_dim, true);
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "moe-batch-bench: synthetic quant command failed: %s\n",
                [[cb.error description] UTF8String]);
        exit(1);
    }
}

static bench_result bench_synth_quant(const bench_config *cfg, int batch, synth_q_kind kind) {
    synth_quant_ctx c = make_synth_quant_ctx(batch, cfg->in_dim, cfg->mid_dim);
    for (int i = 0; i < cfg->warmup; i++) synth_quant_once(&c, kind);
    const double t0 = now_sec();
    for (int i = 0; i < cfg->iters; i++) synth_quant_once(&c, kind);
    const double sec = now_sec() - t0;
    const double ms = sec * 1000.0 / (double)cfg->iters;
    bench_result r = {
        .backend = "gpu_synth_packed_mlp",
        .dtype = kind == SYNTH_Q_FP4_LUT ? "fp4_lut_in_kernel" : "int8_fixed_in_kernel",
        .batch = batch,
        .in_dim = cfg->in_dim,
        .mid_dim = cfg->mid_dim,
        .iters = cfg->iters,
        .ms = ms,
        .gflops = mlp_gflop(batch, cfg->in_dim, cfg->mid_dim) / (ms / 1000.0),
    };
    return r;
}

typedef struct qhybrid_runner qhybrid_runner;

typedef enum {
    QHYBRID_WORKER_GPU,
    QHYBRID_WORKER_AMX,
    QHYBRID_WORKER_BG_DEQUANT,
} qhybrid_worker_kind;

typedef struct {
    qhybrid_runner *runner;
    qhybrid_worker_kind kind;
    int seen_generation;
} qhybrid_worker;

struct qhybrid_runner {
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    int generation;
    int done_count;
    int worker_count;
    int active_slot;
    bool stop;
    bool collect;
    bool run_bg_dequant;
    ds4_moe_block_ctx *gpu;
    amx_cached_moe_ctx *amx_slots;
    gpu_dequant_moe *dequant_slots;
    double gpu_worker_ms_sum;
    double amx_worker_ms_sum;
    double bg_dequant_ms_sum;
    qhybrid_worker workers[3];
    pthread_t threads[3];
};

static void *qhybrid_worker_main(void *arg) {
    qhybrid_worker *w = (qhybrid_worker *)arg;
    qhybrid_runner *r = w->runner;
    for (;;) {
        pthread_mutex_lock(&r->mutex);
        while (!r->stop && w->seen_generation == r->generation) {
            pthread_cond_wait(&r->start_cond, &r->mutex);
        }
        if (r->stop) {
            pthread_mutex_unlock(&r->mutex);
            return NULL;
        }
        w->seen_generation = r->generation;
        const int active_slot = r->active_slot;
        const int bg_slot = active_slot ^ 1;
        const bool collect = r->collect;
        pthread_mutex_unlock(&r->mutex);

        const double t0 = now_sec();
        @autoreleasepool {
            switch (w->kind) {
            case QHYBRID_WORKER_GPU:
                ds4_moe_block_ctx_once(r->gpu);
                break;
            case QHYBRID_WORKER_AMX:
                amx_cached_moe_once(&r->amx_slots[active_slot]);
                break;
            case QHYBRID_WORKER_BG_DEQUANT:
                if (r->run_bg_dequant) gpu_dequant_moe_once(&r->dequant_slots[bg_slot]);
                break;
            }
        }
        const double ms = (now_sec() - t0) * 1000.0;

        pthread_mutex_lock(&r->mutex);
        if (collect) {
            switch (w->kind) {
            case QHYBRID_WORKER_GPU: r->gpu_worker_ms_sum += ms; break;
            case QHYBRID_WORKER_AMX: r->amx_worker_ms_sum += ms; break;
            case QHYBRID_WORKER_BG_DEQUANT: r->bg_dequant_ms_sum += ms; break;
            }
        }
        r->done_count++;
        if (r->done_count == r->worker_count) {
            pthread_cond_signal(&r->done_cond);
        }
        pthread_mutex_unlock(&r->mutex);
    }
}

static void qhybrid_runner_start(
        qhybrid_runner *r,
        ds4_moe_block_ctx *gpu,
        amx_cached_moe_ctx *amx_slots,
        gpu_dequant_moe *dequant_slots,
        bool run_bg_dequant) {
    memset(r, 0, sizeof(*r));
    pthread_mutex_init(&r->mutex, NULL);
    pthread_cond_init(&r->start_cond, NULL);
    pthread_cond_init(&r->done_cond, NULL);
    r->gpu = gpu;
    r->amx_slots = amx_slots;
    r->dequant_slots = dequant_slots;
    r->run_bg_dequant = run_bg_dequant;
    r->worker_count = run_bg_dequant ? 3 : 2;
    r->workers[0] = (qhybrid_worker){ .runner = r, .kind = QHYBRID_WORKER_GPU };
    r->workers[1] = (qhybrid_worker){ .runner = r, .kind = QHYBRID_WORKER_AMX };
    r->workers[2] = (qhybrid_worker){ .runner = r, .kind = QHYBRID_WORKER_BG_DEQUANT };
    for (int i = 0; i < r->worker_count; i++) {
        if (pthread_create(&r->threads[i], NULL, qhybrid_worker_main, &r->workers[i]) != 0) {
            fprintf(stderr, "moe-batch-bench: failed to create qhybrid worker threads\n");
            exit(1);
        }
    }
}

static void qhybrid_runner_once(qhybrid_runner *r) {
    pthread_mutex_lock(&r->mutex);
    r->done_count = 0;
    r->generation++;
    pthread_cond_broadcast(&r->start_cond);
    while (r->done_count < r->worker_count) {
        pthread_cond_wait(&r->done_cond, &r->mutex);
    }
    if (r->run_bg_dequant) r->active_slot ^= 1;
    pthread_mutex_unlock(&r->mutex);
}

static void qhybrid_runner_set_collect(qhybrid_runner *r, bool collect) {
    pthread_mutex_lock(&r->mutex);
    r->collect = collect;
    if (collect) {
        r->gpu_worker_ms_sum = 0.0;
        r->amx_worker_ms_sum = 0.0;
        r->bg_dequant_ms_sum = 0.0;
    }
    pthread_mutex_unlock(&r->mutex);
}

static void qhybrid_runner_stop(qhybrid_runner *r) {
    pthread_mutex_lock(&r->mutex);
    r->stop = true;
    pthread_cond_broadcast(&r->start_cond);
    pthread_mutex_unlock(&r->mutex);
    for (int i = 0; i < r->worker_count; i++) {
        pthread_join(r->threads[i], NULL);
    }
    pthread_cond_destroy(&r->done_cond);
    pthread_cond_destroy(&r->start_cond);
    pthread_mutex_destroy(&r->mutex);
}

static qhybrid_result bench_qhybrid(const bench_config *cfg, int gpu_batch, int amx_batch, bool bg_dequant) {
    @autoreleasepool {
        ds4_moe_block_ctx gpu =
            make_ds4_moe_block_ctx(gpu_batch, cfg->in_dim, cfg->mid_dim, 100u);
        ds4_moe_block_ctx gpu_other =
            make_ds4_moe_block_ctx(amx_batch, cfg->in_dim, cfg->mid_dim, 200u);

        for (int i = 0; i < cfg->warmup; i++) {
            ds4_moe_block_ctx_once(&gpu);
            ds4_moe_block_ctx_once(&gpu_other);
        }

        const double gpu_seq_t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            ds4_moe_block_ctx_once(&gpu);
            ds4_moe_block_ctx_once(&gpu_other);
        }
        const double gpu_seq_sec = now_sec() - gpu_seq_t0;

        gpu_dequant_moe dequant_slots[2];
        memset(dequant_slots, 0, sizeof(dequant_slots));
        amx_cached_moe_ctx amx_slots[2];
        memset(amx_slots, 0, sizeof(amx_slots));

        if (bg_dequant) {
            dequant_slots[0] = make_gpu_dequant_moe(cfg->in_dim, cfg->mid_dim);
            dequant_slots[1] = make_gpu_dequant_moe(cfg->in_dim, cfg->mid_dim);
            gpu_dequant_moe_once(&dequant_slots[0]);
            gpu_dequant_moe_once(&dequant_slots[1]);
            amx_slots[0] =
                make_amx_cached_moe_with_weights(amx_batch, cfg->in_dim, cfg->mid_dim,
                                                 [dequant_slots[0].w_gate contents],
                                                 [dequant_slots[0].w_up contents],
                                                 [dequant_slots[0].w_down contents],
                                                 300u);
            amx_slots[1] =
                make_amx_cached_moe_with_weights(amx_batch, cfg->in_dim, cfg->mid_dim,
                                                 [dequant_slots[1].w_gate contents],
                                                 [dequant_slots[1].w_up contents],
                                                 [dequant_slots[1].w_down contents],
                                                 400u);
        } else {
            dequant_slots[0] = make_gpu_dequant_moe(cfg->in_dim, cfg->mid_dim);
            gpu_dequant_moe_once(&dequant_slots[0]);
            amx_slots[0] =
                make_amx_cached_moe_with_weights(amx_batch, cfg->in_dim, cfg->mid_dim,
                                                 [dequant_slots[0].w_gate contents],
                                                 [dequant_slots[0].w_up contents],
                                                 [dequant_slots[0].w_down contents],
                                                 300u);
            amx_slots[1] = amx_slots[0];
        }

        qhybrid_runner runner;
        qhybrid_runner_start(&runner, &gpu, amx_slots, dequant_slots, bg_dequant);
        for (int i = 0; i < cfg->warmup; i++) {
            qhybrid_runner_once(&runner);
        }

        qhybrid_runner_set_collect(&runner, true);
        const double t0 = now_sec();
        for (int i = 0; i < cfg->iters; i++) {
            qhybrid_runner_once(&runner);
        }
        const double sec = now_sec() - t0;
        qhybrid_runner_set_collect(&runner, false);

        const double gpu_worker_ms = runner.gpu_worker_ms_sum / (double)cfg->iters;
        const double amx_worker_ms = runner.amx_worker_ms_sum / (double)cfg->iters;
        const double bg_dequant_ms = runner.bg_dequant_ms_sum / (double)cfg->iters;
        qhybrid_runner_stop(&runner);

        if (bg_dequant) {
            free_amx_cached_moe(&amx_slots[1]);
        }
        free_amx_cached_moe(&amx_slots[0]);
        free_ds4_moe_block_ctx(&gpu_other);
        free_ds4_moe_block_ctx(&gpu);

        const int combined_batch = gpu_batch + amx_batch;
        const double ms = sec * 1000.0 / (double)cfg->iters;
        const double gpu_seq_ms = gpu_seq_sec * 1000.0 / (double)cfg->iters;
        qhybrid_result r = {
            .backend = bg_dequant ? "qhybrid_ds4gpu_amx_bgdequant" : "qhybrid_ds4gpu_amx_cached",
            .dtype = "iq2xxs_q2k_to_bf16",
            .gpu_batch = gpu_batch,
            .amx_batch = amx_batch,
            .combined_batch = combined_batch,
            .in_dim = cfg->in_dim,
            .mid_dim = cfg->mid_dim,
            .iters = cfg->iters,
            .gpu_conc_ms = gpu_worker_ms,
            .amx_conc_ms = amx_worker_ms,
            .bg_dequant_ms = bg_dequant_ms,
            .ms = ms,
            .tokens_per_s = (double)combined_batch * 1000.0 / ms,
            .gflops = (mlp_gflop(gpu_batch, cfg->in_dim, cfg->mid_dim) +
                       mlp_gflop(amx_batch, cfg->in_dim, cfg->mid_dim)) / (ms / 1000.0),
            .gpu_seq_ms = gpu_seq_ms,
            .gpu_seq_tokens_per_s = (double)combined_batch * 1000.0 / gpu_seq_ms,
            .speedup_vs_gpu_seq = gpu_seq_ms / ms,
        };
        return r;
    }
}

static void print_header(FILE *out) {
    fprintf(out, "backend,dtype,batch,in_dim,mid_dim,iters,ms,gflops\n");
}

static void print_result(FILE *out, bench_result r) {
    fprintf(out, "%s,%s,%d,%d,%d,%d,%.6f,%.3f\n",
            r.backend, r.dtype, r.batch, r.in_dim, r.mid_dim, r.iters, r.ms, r.gflops);
    fflush(out);
}

static void print_split_header(FILE *out) {
    fprintf(out, "backend,dtype,gpu_batch,amx_batch,batch,in_dim,mid_dim,iters,ms,tokens_per_s,gflops,gpu_seq_ms,gpu_seq_tokens_per_s,speedup_vs_gpu_seq\n");
}

static void print_split_result(FILE *out, split_result r) {
    fprintf(out, "%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.1f,%.3f,%.6f,%.1f,%.3f\n",
            r.backend, r.dtype, r.gpu_batch, r.amx_batch, r.combined_batch,
            r.in_dim, r.mid_dim, r.iters, r.ms, r.tokens_per_s, r.gflops,
            r.gpu_seq_ms, r.gpu_seq_tokens_per_s, r.speedup_vs_gpu_seq);
    fflush(out);
}

static void print_qhybrid_header(FILE *out) {
    fprintf(out, "backend,dtype,gpu_batch,amx_batch,batch,in_dim,mid_dim,iters,gpu_conc_ms,amx_conc_ms,bg_dequant_ms,ms,tokens_per_s,gflops,gpu_seq_ms,gpu_seq_tokens_per_s,speedup_vs_gpu_seq\n");
}

static void print_qhybrid_result(FILE *out, qhybrid_result r) {
    fprintf(out, "%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.1f,%.3f,%.6f,%.1f,%.3f\n",
            r.backend, r.dtype, r.gpu_batch, r.amx_batch, r.combined_batch,
            r.in_dim, r.mid_dim, r.iters, r.gpu_conc_ms, r.amx_conc_ms,
            r.bg_dequant_ms, r.ms, r.tokens_per_s, r.gflops,
            r.gpu_seq_ms, r.gpu_seq_tokens_per_s, r.speedup_vs_gpu_seq);
    fflush(out);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bench_config cfg = parse_options(argc, argv);

        FILE *csv = NULL;
        if (cfg.csv_path) {
            csv = fopen(cfg.csv_path, "wb");
            if (!csv) {
                fprintf(stderr, "moe-batch-bench: failed to open %s: %s\n",
                        cfg.csv_path, strerror(errno));
                return 1;
            }
        }

        if (cfg.run_split) {
            if (cfg.dtype_mask != DENSE_DTYPE_BF16) {
                fprintf(stderr, "moe-batch-bench: --backend split currently supports --dtype bf16 only\n");
                return 2;
            }
            print_split_header(stdout);
            if (csv) print_split_header(csv);
            for (int i = 0; i < cfg.n_batches; i++) {
                split_result r = bench_split_bf16(&cfg, cfg.batches[i], cfg.split_amx_batch);
                print_split_result(stdout, r);
                if (csv) print_split_result(csv, r);
            }
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_qhybrid) {
            if (cfg.dtype_mask != DENSE_DTYPE_BF16) {
                fprintf(stderr, "moe-batch-bench: --backend qhybrid currently supports --dtype bf16 only\n");
                return 2;
            }
            print_qhybrid_header(stdout);
            if (csv) print_qhybrid_header(csv);
            for (int i = 0; i < cfg.n_batches; i++) {
                const int gpu_batch = cfg.batches[i];
                if (cfg.qhybrid_cached) {
                    qhybrid_result r =
                        bench_qhybrid(&cfg, gpu_batch, cfg.split_amx_batch, false);
                    print_qhybrid_result(stdout, r);
                    if (csv) print_qhybrid_result(csv, r);
                }
                if (cfg.qhybrid_bgdequant) {
                    qhybrid_result r =
                        bench_qhybrid(&cfg, gpu_batch, cfg.split_amx_batch, true);
                    print_qhybrid_result(stdout, r);
                    if (csv) print_qhybrid_result(csv, r);
                }
            }
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_dequant_parts) {
            bench_dequant_parts(&cfg, stdout, csv);
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_dequant_batch) {
            bench_dequant_batch(&cfg, stdout, csv);
            if (csv) fclose(csv);
            return 0;
        }

        print_header(stdout);
        if (csv) print_header(csv);

        if (cfg.run_amx_quant) {
            if (cfg.dtype_mask != DENSE_DTYPE_BF16) {
                fprintf(stderr, "moe-batch-bench: --backend amxq currently supports --dtype bf16 only\n");
                return 2;
            }
            for (int i = 0; i < cfg.n_batches; i++) {
                const int batch = cfg.batches[i];
                if (cfg.amxq_cached) {
                    bench_result cached = bench_amx_quant_bf16(&cfg, batch, false);
                    print_result(stdout, cached);
                    if (csv) print_result(csv, cached);
                }
                if (cfg.amxq_dequant) {
                    bench_result dequant = bench_amx_quant_bf16(&cfg, batch, true);
                    print_result(stdout, dequant);
                    if (csv) print_result(csv, dequant);
                }
                if (cfg.amxq_gpu_dequant) {
                    bench_result gpu_dequant = bench_amx_quant_gpu_dequant_bf16(&cfg, batch);
                    print_result(stdout, gpu_dequant);
                    if (csv) print_result(csv, gpu_dequant);
                }
            }
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_gpu_synth_quant) {
            for (int i = 0; i < cfg.n_batches; i++) {
                const int batch = cfg.batches[i];
                bench_result fp4 = bench_synth_quant(&cfg, batch, SYNTH_Q_FP4_LUT);
                print_result(stdout, fp4);
                if (csv) print_result(csv, fp4);
                bench_result i8 = bench_synth_quant(&cfg, batch, SYNTH_Q_INT8_FIXED);
                print_result(stdout, i8);
                if (csv) print_result(csv, i8);
            }
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_ane_pack) {
            for (int i = 0; i < cfg.n_batches; i++) {
                bench_result r = bench_ane_pack_materialize(&cfg, cfg.batches[i]);
                print_result(stdout, r);
                if (csv) print_result(csv, r);
            }
            if (csv) fclose(csv);
            return 0;
        }

        if (cfg.run_ane_pack_gpu) {
            for (int i = 0; i < cfg.n_batches; i++) {
                bench_result r = bench_ane_pack_materialize_gpu(&cfg, cfg.batches[i]);
                print_result(stdout, r);
                if (csv) print_result(csv, r);
            }
            if (csv) fclose(csv);
            return 0;
        }

        for (int i = 0; i < cfg.n_batches; i++) {
            const int batch = cfg.batches[i];
            const dense_dtype dtypes[] = { DENSE_FP16, DENSE_BF16, DENSE_FP32 };
            for (size_t di = 0; di < sizeof(dtypes) / sizeof(dtypes[0]); di++) {
                dense_dtype dtype = dtypes[di];
                const uint32_t bit = dtype == DENSE_BF16 ? DENSE_DTYPE_BF16 :
                                     dtype == DENSE_FP32 ? DENSE_DTYPE_FP32 :
                                     DENSE_DTYPE_FP16;
                if ((cfg.dtype_mask & bit) == 0) continue;
                if (cfg.run_gpu) {
                    bench_result r = bench_gpu(&cfg, batch, dtype);
                    print_result(stdout, r);
                    if (csv) print_result(csv, r);
                }
                if (cfg.run_amx) {
                    bench_result r = bench_amx(&cfg, batch, dtype);
                    print_result(stdout, r);
                    if (csv) print_result(csv, r);
                }
            }
            if (cfg.run_ds4) {
                bench_result r = bench_ds4_moe_block(&cfg, batch);
                print_result(stdout, r);
                if (csv) print_result(csv, r);
            }
        }

        if (csv) fclose(csv);
    }
    return 0;
}
