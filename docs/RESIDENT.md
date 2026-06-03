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
DS4_METAL_PREFILL_CHUNK=16384 DS4_METAL_GRAPH_RAW_CAP=16640 ./ds4 \
  -m ds4flash.gguf \
  --ctx 32768 \
  -p "Hello"
```

## Tuning Profiles

`ds4_profile.json` applies machine-specific defaults. It is loaded from:

1. `DS4_PROFILE` if set;
2. `./ds4_profile.json`;
3. the executable directory;
4. `~/.config/ds4/ds4_profile.json`.

Set `DS4_PROFILE=none` to disable automatic profile defaults.

Resident profiles can choose between classic Metal ALU, NAX-backed matmul2d, and
ANE hybrid prefill paths depending on chip, RAM, and prompt size. The profile
file is intentionally the public control surface for alpha. The many `DS4_*`
environment knobs are implementation details unless a doc explicitly names one.

## Memory

Resident mode has the same basic memory constraint as upstream DS4: the full
model file and runtime caches need to fit. For smaller-memory Apple Silicon
systems, prefer SSD sidecar mode.
