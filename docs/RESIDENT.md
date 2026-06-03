# Resident Mode

Resident mode is the original full-GGUF DS4 path. The full quantized model lives
in one GGUF file and the routed experts are loaded from that file rather than
from an SSD sidecar.

Use resident mode when:

- the selected quantization fits comfortably in RAM;
- you want a baseline against sidecar mode;
- you are using upstream-style DS4 workflows or GGUF tooling;
- you want to avoid sidecar packaging until the public converter lands.

## Download

```sh
./download_model.sh q2-imatrix
```

Alternate resident GGUF:

```sh
./download_model.sh huihui-iq2xxs
```

That target downloads
[Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF/resolve/main/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf)
from
[huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF).
It can run resident on a 96 GB M3 Ultra, but memory headroom is tight; run with
little to nothing else active.

The script creates or updates:

```text
./ds4flash.gguf -> ./gguf/<selected model>
```

Run:

```sh
./ds4 -p "Hello"
```

For a larger context:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m ds4flash.gguf \
  --ctx 32768 \
  -p "Hello"
```

## Tuning Profiles

Resident profiles can choose between classic Metal ALU, NAX-backed matmul2d, and
ANE hybrid prefill paths depending on chip, RAM, and prompt size. The profile
file is intentionally the public control surface for alpha. The many `DS4_*`
environment knobs are implementation details unless a doc explicitly names one.

See [PROFILES.md](PROFILES.md) for profile lookup order, sidecar-vs-resident
defaults, and override rules.

## Memory

Resident mode has the same basic memory constraint as upstream DS4: the full
model file and runtime caches need to fit. For smaller-memory Apple Silicon
systems, prefer SSD sidecar mode.

Reaped/pruned resident models are still under investigation and are not yet a
recommended alpha path.
