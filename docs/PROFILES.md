# Machine Profiles

`ds4_profile.json` is the public alpha tuning surface for machine-specific
defaults. It keeps measured Apple Silicon settings in one file instead of
requiring users to export a long list of `DS4_*` environment variables.

Profiles are defaults only. If you export an environment variable yourself, the
profile loader keeps your value.

## Load Order

At startup, DS4 looks for a profile file in this order:

1. `DS4_PROFILE=/path/to/profile.json`
2. `./ds4_profile.json`
3. `<directory of executable>/ds4_profile.json`
4. `~/.config/ds4/ds4_profile.json`

Set `DS4_PROFILE=none` to disable automatic profile defaults.

## Matching

The first profile whose `match` block fits the host is applied. The current
profiles match by Apple chip name substring and minimum RAM size, so order
matters. More-specific entries, such as M5 Max 128 GB, must appear before
broader entries, such as base M5 streaming-only.

Startup logs identify the selected profile:

```text
ds4: applied tuning profile [Apple M3 Ultra] from ds4_profile.json
```

For sidecar runs, the log says:

```text
ds4: applied sidecar tuning profile [Apple M3 Ultra] from ds4_profile.json
```

## Resident vs Sidecar Defaults

Each profile can have separate resident and sidecar settings:

- `env`: defaults for resident/full-GGUF mode.
- `prefill_by_tokens`: resident per-chunk backend routing by token count.
- `sidecar_env`: defaults for SSD sidecar mode.

Sidecar defaults are used only when DS4 is launched with a sidecar path, for
example `--moe-sidecar "$DS4_SIDECAR_DIR"` and `--moe-mode slot-bank`.

Resident profiles can select classic Metal ALU, NAX-backed `matmul2d`, and ANE
hybrid prefill paths. Sidecar profiles tune the Flash-MoE SSD slot-bank path,
including routed expert execution on GPU or ANE where the measured profile says
it wins.

## Current Apple Silicon Coverage

- M5 and M5 Max profiles cover NAX-backed resident routes on resident-capable
  memory sizes, plus ANE-assisted SSD sidecar streaming.
- M3 Ultra profiles cover resident ANE/GPU overlap and sidecar ANE/GPU overlap.
- M1 Max is streaming-focused and keeps routed ANE off by default because the
  measured GPU path is faster for the alpha profile.

CUDA sources remain in tree, but the alpha profile file is centered on Apple
Silicon validation.

## User Overrides

Examples:

```sh
DS4_PROFILE=none ./ds4 -p "Hello"
```

```sh
DS4_PROFILE=/path/to/custom-profile.json ./ds4 -p "Hello"
```

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m "$DS4_SIDECAR_DIR/dense/model-dense.gguf" \
  --moe-sidecar "$DS4_SIDECAR_DIR" \
  --moe-mode slot-bank
```

For normal sidecar runs, leave `DS4_METAL_GRAPH_RAW_CAP` unset so the Metal
raw-KV cap auto-follows the prefill chunk and server checkpoint frontiers remain
aligned.

The many internal `DS4_*` switches are intentionally not all documented. Treat
`ds4_profile.json` and the explicitly named environment variables in these docs
as the supported alpha control surface.
