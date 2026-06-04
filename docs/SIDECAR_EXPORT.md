# Exporting A DS4 Flash-MoE Sidecar

DS4 can load routed MoE expert weights from a Flash-MoE sidecar directory. If
the directory is a full alpha package with `manifest.json` and
`dense/model-dense.gguf`, use the package root directly. DS4 detects the dense
GGUF and sidecar metadata; no explicit `--moe-sidecar` or `--moe-mode` flag is
needed:

```sh
./ds4 \
  -m /path/to/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 64 \
  --ctx 32768
```

For export validation or experiments with an expert-only sidecar, use a
compatible GGUF model plus the sidecar explicitly:

```sh
./ds4 \
  -m /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --moe-sidecar /path/to/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --moe-slot-bank 64
```

This repository does not currently ship an in-tree sidecar packer. The public
export tool below exports the routed expert sidecar records. It does not
currently build the full alpha package layout with a stripped dense GGUF at
`dense/model-dense.gguf`; for a turnkey low-RAM SSD streaming package, use the
prebuilt Hugging Face sidecar documented in `MODEL_SETUP.md`.

The public export tool lives in:

- Repo: `https://github.com/anemll/anemll-flash-llama.cpp`
- Branch: `DeepSeek-V4-SSD`
- Script: `tools/flashmoe-sidecar/flashmoe_sidecar.py`

Use that branch specifically. The current `origin/master` version of that repo
may not expose `extract --layout expert-major`. The same path has also existed
on `origin/T105-investigation-checkpoint`, but `DeepSeek-V4-SSD` is the branch
to use for DS4 handoff/repro.

## Wrapper Script

From this DS4 repo:

```sh
scripts/export_flash_moe_sidecar.sh \
  --model /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --out-dir /path/to/dsv4-iq2xxs-expert-major \
  --force \
  --verify
```

If no converter checkout is provided, the wrapper clones the public repo branch
into `./.external/anemll-flash-llama.cpp`. That directory is ignored by git.

To use an existing checkout:

```sh
scripts/export_flash_moe_sidecar.sh \
  --llama-dir /path/to/anemll-flash-llama.cpp \
  --model /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --out-dir /path/to/dsv4-iq2xxs-expert-major \
  --force \
  --verify
```

The wrapper refuses to run if the converter does not expose `--layout`, because
that means it is not on the expected branch or equivalent commit.

## Direct Command

If you prefer to run the converter directly:

```sh
git clone --branch DeepSeek-V4-SSD --single-branch \
  https://github.com/anemll/anemll-flash-llama.cpp.git \
  /path/to/anemll-flash-llama.cpp

python3 /path/to/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py \
  extract \
  --model /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --out-dir /path/to/dsv4-iq2xxs-expert-major \
  --layout expert-major \
  --force

python3 /path/to/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py \
  verify \
  --model /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --sidecar /path/to/dsv4-iq2xxs-expert-major
```

Expected converter output layout:

```text
dsv4-iq2xxs-expert-major/
  manifest.json
  layer_000.bin
  layer_001.bin
  ...
```

The `expert-major` layout stores each layer as expert-contiguous records. DS4
uses `manifest.json` to find each routed family inside the per-layer files.

To match the public alpha package layout, pair this expert sidecar with a
compatible dense-only GGUF under `dense/model-dense.gguf`. If you run DS4 with a
full resident GGUF as `-m`, the sidecar path can still be useful for export
validation and experiments, but it is not the same low-RAM package as the
prebuilt alpha sidecar.

## Run With DS4

If you paired the exported sidecar with a compatible dense GGUF under
`dense/model-dense.gguf`, run the package root:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m /path/to/dsv4-iq2xxs-expert-major \
  --moe-slot-bank 64 \
  --ctx 32768
```

If you only exported expert records, use a compatible DS4 GGUF as `-m` and the
exported sidecar as `--moe-sidecar`:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m /path/to/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --moe-sidecar /path/to/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --moe-slot-bank 64 \
  --ctx 32768
```

Startup should include:

```text
ds4: applied sidecar tuning profile [...]
ds4: Flash-MoE sidecar loaded: ... (slot-bank=64, expert-record ... MiB)
ds4: Flash-MoE slot banks allocated: layers=... slots=64 gpu-bank=... MiB
```

See [STREAMING_KNOBS.md](STREAMING_KNOBS.md) for the slot-bank, prefill, I/O,
and ANE profile knobs.
