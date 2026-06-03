# ds4-ssd

`ds4-ssd` is an alpha fork of antirez's DwarfStar 4 (`ds4`) inference engine
for DeepSeek V4 Flash. The fork keeps the narrow, self-contained DS4 runtime and
adds an SSD-streamed routed-MoE sidecar path for Apple Silicon systems where a
fully resident model is not practical.

The main alpha feature is SSD streaming: dense tensors stay in a normal GGUF,
while routed experts live in a sidecar directory and are paged through a
slot-bank cache. Resident full-GGUF mode is still supported for high-memory
machines. Apple Silicon optimizations include NAX, the Apple neural-accelerator
backed `matmul2d` path used by Metal on M5-class hardware, plus Apple Neural
Engine routed-MLP prefill paths where the measured profile says ANE wins.

This branch is intentionally narrower than the research branch. It keeps the
runtime, Metal shaders, GGUF tools, correctness tests, sidecar smoke, and core
docs, while dropping profiling scripts, handoff notes, session exports, and
bench-only ANE probes from the public alpha tree.

## Status

Alpha means:

- SSD sidecar mode is the release headline.
- Resident full-GGUF mode remains available.
- Correctness vectors and an executable 16K sidecar smoke are the current test
  bar.
- Broader mode coverage, CI, and performance regression automation are planned
  for the next stage.

The in-repo `gguf-tools/deepseek4-quantize` tool builds resident GGUFs. It does
not yet emit the sidecar layout. For this alpha, use the prebuilt sidecar
package at
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major).

## Build

On macOS:

```sh
make
```

This builds:

- `./ds4`: CLI runner.
- `./ds4-server`: OpenAI/Anthropic/Responses-compatible local server.
- `./ds4-bench`: throughput sweeps.
- `./ds4-eval`: evaluation helper.
- `./ds4-agent`: local coding-agent frontend.

`metal/` is a required build input. Do not prune it.

CUDA sources are inherited from upstream DS4 and kept in tree, but the alpha
validation focus is Apple Silicon SSD streaming.

## Run SSD Sidecar Mode

Download the prebuilt sidecar package:

```sh
./download_model.sh sidecar
```

Then set `DS4_SIDECAR_DIR` to the sidecar directory containing `manifest.json`
and `dense/model-dense.gguf`:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
```

Run:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m "$DS4_SIDECAR_DIR/dense/model-dense.gguf" \
  --moe-sidecar "$DS4_SIDECAR_DIR" \
  --moe-mode slot-bank \
  --moe-slot-bank 64 \
  --ctx 32768 \
  -p "Hello"
```

`--ctx 32768` is the KV window in this example. `DS4_METAL_PREFILL_CHUNK=16384`
is the prefill chunk cap used by the alpha smoke path. Leave the Metal raw-KV
cap automatic so it follows the chunk size and server checkpoint frontiers stay
aligned.

Run the committed sidecar smoke:

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

To confirm SSD streaming is active, look for these startup lines:

```text
applied sidecar tuning profile
Flash-MoE sidecar loaded
Flash-MoE slot banks allocated
```

If you pass only `-m /path/to/full-model.gguf`, DS4 is in resident/full-GGUF
mode. SSD streaming requires both the dense sidecar GGUF and `--moe-sidecar`.

See [docs/SIDECAR.md](docs/SIDECAR.md).

Machine-specific defaults for M5, M5 Max, M3 Ultra, and M1 Max are selected
from `ds4_profile.json`. Profiles set defaults only; exported environment
variables still win. See [docs/PROFILES.md](docs/PROFILES.md).

## Run Resident GGUF Mode

Download a resident GGUF:

```sh
./download_model.sh q2-imatrix
```

Then run:

```sh
./ds4 -p "Hello"
```

Resident mode loads the full model file and is meant for high-memory machines.
It is still useful for baseline comparison, server use, and systems with enough
RAM to hold the selected quantization.

See [docs/RESIDENT.md](docs/RESIDENT.md) and
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Validate

The alpha validation gate is:

```sh
make clean
make
./ds4_test --server --metal-kernels
make ane-smoke
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

The sidecar smoke uses `tests/test-vectors/prompts/sidecar_16k.txt` and runs one
deterministic token with a 16K prefill chunk cap.

## Docs

- [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md): model files, downloads, and
  sidecar package expectations.
- [docs/SIDECAR.md](docs/SIDECAR.md): SSD streaming mode and smoke test.
- [docs/RESIDENT.md](docs/RESIDENT.md): full-GGUF resident mode.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): runtime layout and accelerator
  paths.
- [docs/PROFILES.md](docs/PROFILES.md): machine-specific tuning defaults and
  override rules.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): current benchmark stance.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common first-run failures.
- [docs/DWARFSTAR4_REFERENCE.md](docs/DWARFSTAR4_REFERENCE.md): original DS4
  README retained for reference.

## Attribution

`ds4-ssd` is derived from antirez's DwarfStar 4 / `ds4` work and keeps the DS4
model-specific design: GGUF loading, prompt rendering, KV handling, server API,
and DeepSeek V4 Flash validation. The project also depends conceptually on the
GGUF, quantization, and kernel work pioneered by `llama.cpp` and GGML.

The SSD-streaming direction is also indebted to Apple's
[LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://machinelearning.apple.com/research/efficient-large-language)
paper and to the original [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
work by Claude Opus 4.6 and Daniel Woods. Read the
[original Flash-MoE paper](https://github.com/danveloper/flash-moe/blob/main/paper/flash_moe.pdf)
for the full story of how they built that engine in 24 hours.

Keep the repository `LICENSE` with redistributions and preserve attribution to
antirez, llama.cpp, GGML, and their contributors.
