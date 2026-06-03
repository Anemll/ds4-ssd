# SSD Sidecar Mode

Sidecar mode is the alpha release path for running DeepSeek V4 Flash when the
full routed-MoE model does not fit comfortably in RAM.

The dense model remains a GGUF. Routed experts are stored outside the GGUF in a
sidecar directory with a manifest and expert records. At runtime, DS4 keeps a
configurable slot-bank of expert weights resident and streams the rest from SSD.

## Required Files

```text
SIDECAR_DIR/
  manifest.json
  dense/
    model-dense.gguf
  ...
```

Use `SIDECAR_DIR/dense/model-dense.gguf` as the model path and `SIDECAR_DIR` as
the sidecar path.

## Command

```sh
export DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major

DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m "$DS4_SIDECAR_DIR/dense/model-dense.gguf" \
  --moe-sidecar "$DS4_SIDECAR_DIR" \
  --moe-mode slot-bank \
  --moe-slot-bank 64 \
  --ctx 32768 \
  -p "Hello"
```

Important axes:

- `--ctx` is the KV context window.
- `DS4_METAL_PREFILL_CHUNK=16384` caps each prefill chunk at 16K tokens.
- Leave `DS4_METAL_GRAPH_RAW_CAP` unset. The Metal raw-KV cap should auto-follow
  the 16K chunk size so server continued checkpoints stay aligned.
- `--moe-slot-bank` controls how many routed expert slots per layer are kept in
  memory. Increase it for more RAM and fewer SSD fetches; decrease it when RAM
  pressure is high.

## Smoke Test

The committed smoke uses a 16K prompt and one deterministic generated token:

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

Equivalent direct command:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m "$DS4_SIDECAR_DIR/dense/model-dense.gguf" \
  --moe-sidecar "$DS4_SIDECAR_DIR" \
  --moe-mode slot-bank \
  --moe-slot-bank 64 \
  --ctx 32768 \
  -n 1 \
  --temp 0 \
  --prompt-file tests/test-vectors/prompts/sidecar_16k.txt
```

Expected startup logs include:

```text
Flash-MoE sidecar loaded
prefill chunk cap: 16384
Flash-MoE slot banks allocated
```

## Conversion Status

The sidecar format is runtime-defined and parsed from `manifest.json`, but the
public alpha tree does not yet include a sidecar packer. Do not use
`gguf-tools/deepseek4-quantize` expecting a sidecar output; it currently emits
resident GGUFs only.

For alpha, use a prebuilt sidecar distribution. A public converter/packer is a
go/no-go item for a later self-hosted release path.
