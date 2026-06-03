# Model Setup

DS4 has two model layouts in this alpha:

- Resident GGUF: one large GGUF containing dense tensors and routed experts.
- SSD sidecar: a dense GGUF plus a sidecar directory containing routed expert
  records and `manifest.json`.

## Resident GGUF

Use the downloader for the antirez resident GGUF files:

```sh
./download_model.sh q2-imatrix
```

The script downloads from `antirez/deepseek-v4-gguf`, places files under
`./gguf` by default, and updates `./ds4flash.gguf` to the selected resident
model. Override the output location with:

```sh
DS4_GGUF_DIR=/path/to/gguf ./download_model.sh q2-imatrix
```

Common resident choices:

- `q2-imatrix`: about 81 GB on disk, preferred resident baseline for 96/128 GB
  class machines.
- `q4-imatrix`: about 153 GB on disk, intended for much larger RAM systems.
- `mtp`: optional speculative decoding component.

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
the sidecar directory. The dense GGUF is passed with `-m`; the sidecar directory
is passed with `--moe-sidecar`.

The current alpha does not ship a public self-conversion command. In particular,
`gguf-tools/deepseek4-quantize` creates resident GGUF outputs and does not pack
sidecar expert shards. Use the prebuilt sidecar package from
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major):

```sh
./download_model.sh sidecar
```

After extracting or mounting the sidecar:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
DS4_SIDECAR_DIR="$DS4_SIDECAR_DIR" make sidecar-smoke
```

## Local Host Note

For local validation on this host, the sidecar model is expected under:

```text
/Volumes/optane/dsv4-iq2xxs-expert-major
```

That path is not a public requirement; it is only the local test fixture.
