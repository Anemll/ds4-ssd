# Troubleshooting

## `--moe-sidecar requires --moe-mode slot-bank`

Sidecar mode must use slot-bank routing:

```sh
--moe-sidecar "$DS4_SIDECAR_DIR" --moe-mode slot-bank
```

## Missing `manifest.json`

`DS4_SIDECAR_DIR` must point at the sidecar directory, not the dense subdirectory.

Correct:

```text
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major
```

Incorrect:

```text
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major/dense
```

## Missing Dense GGUF

The default sidecar smoke expects:

```text
$DS4_SIDECAR_DIR/dense/model-dense.gguf
```

Override with:

```sh
DS4_DENSE_GGUF=/path/to/model-dense.gguf make sidecar-smoke
```

## Unexpected Chunk Size

The alpha sidecar smoke uses:

```sh
DS4_METAL_PREFILL_CHUNK=16384
```

That is the prefill chunk cap. It is different from `--ctx`, which controls the
KV context window. Leave `DS4_METAL_GRAPH_RAW_CAP` unset in normal sidecar runs;
the runtime auto-selects a raw-KV cap that matches the prefill chunk and keeps
server continued checkpoints aligned.

## Build Cannot Find Metal Shaders

`metal/*.metal` is a required build input for `ds4_metal.o`. Keep the `metal/`
directory in the tree.

## Disabling Profiles

Automatic machine tuning profiles can be disabled with:

```sh
DS4_PROFILE=none ./ds4 ...
```

Use this for A/B testing raw backend behavior. Normal alpha runs should let
`ds4_profile.json` apply defaults.
