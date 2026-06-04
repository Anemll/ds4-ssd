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

Use the package root as the model path. If `-m SIDECAR_DIR` points at a
directory containing `manifest.json` and `dense/model-dense.gguf`, DS4
auto-detects the package, uses the dense GGUF, and enables sidecar slot-bank
mode. No explicit `--moe-sidecar` or `--moe-mode` flag is needed for this
package-root path.

Download the prebuilt alpha package from
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major):

```sh
./download_model.sh sidecar
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
```

Do not point `-m` at a full resident GGUF and expect SSD streaming. A command
with only `-m /path/to/full-model.gguf` runs resident/full-GGUF mode. SSD
streaming auto-detects only the package-root layout shown above. The explicit
long form is still useful when validating an expert-only sidecar:

- `-m "$SIDECAR_DIR/dense/model-dense.gguf"`
- `--moe-sidecar "$SIDECAR_DIR"`
- `--moe-mode slot-bank` or the `ds4-bench` sidecar default

## Command

```sh
export DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major

./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 8 \
  --ctx 8192 \
  -p "Hello"
```

Important axes:

- `--ctx` is the KV context window.
- Start with `--ctx 8192` for a low-RAM first run; raise it after checking
  memory pressure.
- Leave `DS4_METAL_GRAPH_RAW_CAP` unset. The Metal raw-KV cap should auto-follow
  the selected prefill chunk size so server continued checkpoints stay aligned.
- `--moe-slot-bank` controls how many routed expert slots per layer are kept in
  memory. Increase it for more RAM and fewer SSD fetches; decrease it when RAM
  pressure is high.

## Smoke Test

The committed smoke uses a shorter 4K-class prompt and 64 deterministic
generated tokens:

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

Equivalent direct command:

```sh
DS4_METAL_PREFILL_CHUNK=4096 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 8 \
  --ctx 8192 \
  -n 64 \
  --temp 0 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt
```

Expected startup logs include:

```text
applied sidecar tuning profile
Flash-MoE sidecar loaded
prefill chunk cap: 4096
Flash-MoE slot banks allocated
```

Resident/full-GGUF mode instead prints `applied tuning profile` and does not
print `Flash-MoE sidecar loaded`.

## Benchmark Sidecar Mode

`ds4-bench` supports the same sidecar arguments. This benchmarks SSD streaming:

```sh
export DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major

DS4_METAL_PREFILL_CHUNK=16384 ./ds4-bench \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 64 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-sidecar-speed.csv
```

This is a resident benchmark, not SSD streaming:

```sh
./ds4-bench \
  -m /path/to/full-resident-model.gguf \
  --prompt-file speed-bench/promessi_sposi.txt
```

## Conversion Status

The sidecar format is runtime-defined and parsed from `manifest.json`, but the
public alpha tree does not yet include a sidecar packer. Do not use
`gguf-tools/deepseek4-quantize` expecting a sidecar output; it currently emits
resident GGUFs only.

For alpha, use the prebuilt sidecar distribution for the turnkey low-RAM path.
This repo also includes `scripts/export_flash_moe_sidecar.sh`, which calls the
public `anemll/anemll-flash-llama.cpp` converter branch to export expert-major
routed sidecar records. That wrapper does not currently build the full alpha
package layout with a stripped `dense/model-dense.gguf`. See
[SIDECAR_EXPORT.md](SIDECAR_EXPORT.md) for the exact command and caveats, and
[STREAMING_KNOBS.md](STREAMING_KNOBS.md) for slot-bank, prefill, I/O, and ANE
tuning knobs.
