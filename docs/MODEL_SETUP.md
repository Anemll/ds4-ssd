# Model Setup

DS4 has two model layouts in this alpha:

- Resident GGUF: one large GGUF containing dense tensors and routed experts.
- SSD sidecar: a dense GGUF plus a sidecar directory containing routed expert
  records and `manifest.json`.

## Resident GGUF

Use the downloader for resident GGUF files:

```sh
./download_model.sh q2-imatrix
```

The script downloads from the selected Hugging Face repo, places files under
`./gguf` by default, and updates `./ds4flash.gguf` to the selected resident model.
Override the output location with:

```sh
DS4_GGUF_DIR=/path/to/gguf ./download_model.sh q2-imatrix
```

Common resident choices:

- `q2-imatrix`: about 81 GB on disk, preferred resident baseline for 96/128 GB
  class machines.
- `huihui-iq2xxs`: about 80 GB on disk, the
  [Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF/resolve/main/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf)
  resident GGUF
  from
  [huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF).
  It can run resident on a 96 GB M3 Ultra, but leave little to nothing else
  active because memory headroom is tight.
- `q4-imatrix`: about 153 GB on disk, intended for much larger RAM systems.
- `mtp`: optional speculative decoding component.

Reaped/pruned resident models are not yet a recommended alpha path. They are
still under investigation.

Run:

```sh
./ds4 -p "Hello"
```

## SSD Sidecar

The sidecar layout is:

```text
dsv4-iq2xxs-expert-major/
  manifest.json
  dense/
    model-dense.gguf
  ...
```

The runtime parses `manifest.json` and opens the expert records directly from
the sidecar directory. Pass the package root with `-m SIDECAR_DIR`; DS4 detects
`manifest.json` plus `dense/model-dense.gguf`, uses the dense GGUF internally,
and enables sidecar slot-bank mode. This is the preferred public command shape:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"

./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 8 \
  --ctx 8192 \
  -p "Hello"
```

Increase `--moe-slot-bank` and `--ctx` only after checking memory pressure.
`--moe-slot-bank 64 --ctx 32768` is a high-memory setting.

The explicit `-m SIDECAR_DIR/dense/model-dense.gguf --moe-sidecar SIDECAR_DIR
--moe-mode slot-bank` form still works, but it is mainly useful for
expert-only export validation or debugging.

`gguf-tools/deepseek4-quantize` creates resident GGUF outputs and does not pack
sidecar expert shards. For the turnkey low-RAM alpha path, use the prebuilt
sidecar package from
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major):

```sh
./download_model.sh sidecar
```

For package export from a supported GGUF, see
[SIDECAR_EXPORT.md](SIDECAR_EXPORT.md). That path uses an external public
converter branch to export both the routed sidecar records and the
dense/shared-only `dense/model-dense.gguf`.

After extracting or mounting the sidecar:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
make sidecar-smoke
```

## Local Host Note

For local validation on this host, the sidecar model is expected under:

```text
/Volumes/optane/dsv4-iq2xxs-expert-major
```

That path is not a public requirement; it is only the local test fixture.
