# Architecture

`ds4-ssd` is a model-specific DeepSeek V4 Flash runtime. It is not a generic
GGUF runner.

## Core Runtime

- `ds4.c`, `ds4.h`: model loader, graph execution, KV cache, sidecar parser,
  resident/streaming MoE paths, and runtime configuration.
- `ds4_cli.c`: command-line runner.
- `ds4_server.c`: local HTTP server with OpenAI, Anthropic, and Responses-style
  API handling.
- `ds4_agent.c`, `ds4_kvstore.*`: local agent-facing runner and KV store.
- `ds4_profile.*`, `ds4_profile.json`: machine profile loader and public alpha
  tuning surface.
- `linenoise.*`, `rax.*`: required support code for CLI/server builds.

## GPU And Accelerator Paths

- `ds4_metal.m` and `metal/*.metal` are required for the Apple Silicon build.
- NAX is the Apple neural-accelerator backed `matmul2d` path exposed through
  Metal on M5-class hardware. DS4 uses it for dense and routed-MoE matrix work
  where the profile selects it.
- `ds4_ane_mlp_int8w.*` is core runtime code for the ANE MLP path. It is
  top-level runtime code, not a benchmark artifact. See
  [ANE_KERNELS.md](ANE_KERNELS.md) for the implemented ANE kernel families.
- `ds4_cuda.cu` and `ds4_iq2_tables_cuda.inc` are inherited CUDA support. They
  remain in tree, but Apple Silicon sidecar mode is the alpha validation focus.

## Sidecar Streaming

Sidecar mode separates the model into:

- dense GGUF tensors, loaded normally;
- routed expert records, addressed through a sidecar manifest;
- a slot-bank cache that keeps only a selected number of expert slots resident.

The goal is to let fast SSDs carry the cold routed-expert footprint while RAM is
used for dense tensors, active expert slots, and KV state.

## Resident Mode

Resident mode keeps the original full-GGUF flow. It remains useful as a baseline
and for high-memory systems.

## Tests

- `ds4_test` covers server behavior, official logprob vectors, long-context
  checks, tool-call behavior, and isolated Metal kernels.
- `tests/ane_ds4_mlp_i8i8_precision_smoke.m` checks the ANE int8 MLP runtime
  against CPU references.
- `tests/sidecar_smoke.sh` runs a 16K prefill sidecar smoke against a real
  sidecar model.
