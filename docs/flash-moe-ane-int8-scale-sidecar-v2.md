# Flash-MoE ANE INT8 scale sidecar v2

`scripts/export_flash_moe_ane_i8_v2.py` creates a **new** expert-major
sidecar from an existing schema-v1 package. It never edits the source package.
The destination retains every source weight byte and appends fp16 symmetric
INT8 dequantization scales to each expert record.

The exporter requires Python 3 and NumPy; both are already used by the
repository's sidecar conversion tools.

The current source package is expected to contain, per routed layer:

- `ffn_gate_exps`: IQ2_XXS
- `ffn_up_exps`: IQ2_XXS
- `ffn_down_exps`: Q2_K
- one layer file with expert-major records

## Record layout

For `/Volumes/optane/dsv4-iq2xxs-expert-major`, one v2 expert record is:

| Region | Relative offset | Bytes |
|---|---:|---:|
| Existing gate/up/down weight record | 0 | 7,077,888 |
| Gate scales, 2,048 × fp16 | 7,077,888 | 4,096 |
| Up scales, 2,048 × fp16 | 7,081,984 | 4,096 |
| Down scales, 4,096 × fp16 | 7,086,080 | 8,192 |
| Alignment padding | 7,094,272 | 0 |

The new expert stride is 7,094,272 bytes. Scale order is exactly the packed ANE
input order `[gate I | up I | down H]`. The full 43-layer package adds about
172 MiB of scale data to the routed-expert files.

At runtime this packed vector is passed to ANE as a dynamic fp16 IOSurface
tensor on every expert evaluation. The per-channel values are not baked into
the compiled graph. Mode 18 bakes only the shared `1/512` base multiplier into
the weight matmuls, then applies each dynamic `scale / (1/512)` correction to
the smaller gate, up, and down outputs.

Each scale is a dequantization multiplier:

```text
scale[output_channel] = max(abs(exactly_dequantized_weight_row)) / 127
```

An all-zero row uses scale `1.0`. The scale is rounded to fp16 before storage.
IQ2_XXS and Q2_K values are decoded from the existing sidecar bytes; the tool
does not use the original unquantized model.

## Dry run

This validates source metadata, tensor geometry, record offsets, quant block
sizes, and all source file lengths without creating the destination:

```bash
python3 scripts/export_flash_moe_ane_i8_v2.py \
  /Volumes/optane/dsv4-iq2xxs-expert-major \
  /Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2 \
  --dry-run
```

No command in this document should be run against `/Volumes/SN8100` until the
destination name and available capacity have been confirmed.

## Small real-data smoke test

Use one layer and one expert in a temporary destination. Partial packages are
marked `export_scope.partial=true` and `runtime_loadable=false` in the manifest.

```bash
smoke_root="$(mktemp -d /tmp/ds4-ane-i8-v2.XXXXXX)"
python3 scripts/export_flash_moe_ane_i8_v2.py \
  /Volumes/optane/dsv4-iq2xxs-expert-major \
  "$smoke_root/package" \
  --layers 0 \
  --expert-limit 1 \
  --dense-mode none \
  --validate \
  --progress-every 1
```

`--layers` accepts comma-separated values and ranges such as `0,2-4`.
`--expert-limit N` always selects the first `N` experts.

## Full export

After selecting the final destination, create a standalone package with the
dense model copied into it:

```bash
python3 scripts/export_flash_moe_ane_i8_v2.py \
  /Volumes/optane/dsv4-iq2xxs-expert-major \
  /Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2 \
  --dense-mode copy \
  --progress-every 16
```

The destination must be absent or empty. Existing files are never overwritten.
Layer files and the manifest are published through same-directory temporary
files and no-replace atomic links. The manifest is written last. A failed dense
copy can leave a partial destination `dense` directory, but no manifest is
published and existing paths are not replaced.

Dense-directory options are:

- `copy`: make a standalone destination; this copies the current ~8.2 GiB
  dense package.
- `symlink`: avoid duplicating the dense model, but keep the source volume as a
  runtime dependency.
- `none`: omit dense data, normally for smoke tests.

## Validation

`--validate` performs a second pass after generation. It checks:

- every destination weight prefix is byte-identical to the source record;
- every scale vector matches a fresh calculation from the source quant blocks;
- alignment padding is zero;
- layer sizes, per-entry scale metadata, and SHA-256 checksums match the v2
  manifest.

Validation intentionally repeats scale calculation. For a full export it can
be run later as a separate read-only operation:

```bash
python3 scripts/export_flash_moe_ane_i8_v2.py \
  /Volumes/optane/dsv4-iq2xxs-expert-major \
  /Volumes/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2 \
  --validate-only
```

## Manifest contract

The output keeps `sidecar_kind: flashmoe_gguf` and
`layout: layer_major_expert`, and sets:

```json
{
  "schema_version": 2,
  "storage_layout": "expert_major_weights_plus_ane_i8_output_scales_v1",
  "ane_i8_scale_scheme": {
    "storage": "appended_to_expert_record",
    "family_order": ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
    "dtype": "F16",
    "endian": "little",
    "semantics": "dequant_multiplier",
    "axis": 1,
    "axis_name": "output_channel",
    "group_size": 1,
    "record_alignment": 64
  }
}
```

Every routed entry receives these flat optional fields:

```text
ane_i8_scale_offset
ane_i8_scale_bytes
ane_i8_scale_count
ane_i8_scale_dtype = "F16"
ane_i8_scale_semantics = "dequant_multiplier"
ane_i8_scale_axis = 1
ane_i8_scale_group_size = 1
```

`ane_i8_scale_offset` is relative to the beginning of that expert's record.
`bytes_per_expert` continues to mean only the original quantized weight region;
`expert_stride` includes the appended scale region.

`layer_files` records old/new strides, scale-region geometry, byte lengths, and
SHA-256 hashes. `source.sidecar_manifest_sha256` binds the v2 package to the
exact source manifest used during generation.

Top-level `runtime_loadable` means the routed sidecar contains every source
layer and expert. `standalone_model_package` additionally requires a copied or
linked dense package; `dense_mode` records which choice was used.

The runtime validates this locked contract when the package is opened. With
`--ane`, a conforming sidecar automatically selects per-output-channel INT8
weights and the quality-approved fp16-activation, output-factored ANE graph
(mode 18). Legacy sidecars without the scale contract retain the scalar ANE
path and its existing batching behavior.

The automatic selection has explicit A/B escape hatches:

- `DS4_FORCE_ANE=1` is the global diagnostic override. It wins over
  `DS4_NO_INT8`, profile/backend preferences, routed-group scheduler thresholds,
  short-prefill gates, and the agent's default GPU system-prompt bootstrap. It
  sets `DS4_METAL_RESUME_PREFILL_MIN=1`, so even a one-token appended suffix
  takes batched ANE prefill instead of the yellow token-by-token decode path. It
  enables strict `DS4_FLASH_MOE_ANE_REQUIRE=1`; unsupported contracts therefore
  fail closed rather than falling back. This strict implementation currently
  accepts only Metal DeepSeek4 Flash/Pro Flash-MoE sidecars; both streaming and
  `--resident` full-slot-bank layouts use the verified sidecar executor.
  Every routed layer must expose valid gate/up/down v2 FP16 per-output-channel
  scales; force explicitly selects the FP16-X output-factored graph. Full-GGUF
  resident, legacy unscaled sidecars, GLM/HY3, and non-Metal launches are rejected. The agent
  rebuilds the forced system prompt on ANE every launch and disables conversation
  KV save/resume; the server disables disk KV because the legacy cache format
  does not prove model/sidecar/scale-policy provenance. The
  narrower `DS4_AGENT_SYSPROMPT_ANE_PREFILL=1` override still uses
  `sysprompt-ane-v1.kv`.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL=0` selects the legacy scalar-weight path.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL=1` requires the runtime to request the packed
  scale vectors; strict tests should also set `DS4_FLASH_MOE_ANE_REQUIRE=1`.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL_FP16X=0` retains per-channel weights but uses
  the int8-activation mode 15.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL_FP16X_OUTPUT_FACTORED=0` selects mode 16 for
  an exact A/B against the promoted mode 18 path.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL_BATCHES` controls mode 18's compile buckets;
  unset defaults to the measured `128,256` pair.
- `DS4_FLASH_MOE_ANE_PER_CHANNEL_THREADS_FIXED_BATCH` controls mode 18's
  thread rebatching; unset defaults to `1`. The generic
  `DS4_FLASH_MOE_ANE_THREADS_FIXED_BATCH` remains a fallback override, while
  legacy modes continue to use the generic batch-list variable.
- `DS4_FLASH_MOE_ANE_FORCE_ALL_GROUPS=1` ignores the normal minimum-size and
  scheduler lane thresholds, so test prompts of any length exercise ANE.
- `DS4_FLASH_MOE_ANE_REQUIRE=1` fails closed if any group expected on ANE is
  rejected or falls back.

## Acceptance results

On the generated 43-layer SN8100 package, a ten-prompt, 240-token quality gate
against non-ANE prefill produced mean KL `0.02269`, mean logit correlation
`0.99037`, and top-1 agreement `1.0`. Average continuation NLL was `0.32023`
for mode 18 versus `0.33064` for the non-ANE baseline.

A paired full-prefill sweep versus the legacy scalar ANE path measured mode 18
throughput deltas of `+1.27%`, `-0.96%`, `-1.86%`, and `-1.60%` at contexts
512, 2048, 3584, and 4096 respectively. Every run used strict all-group ANE
coverage.

## Tests

```bash
python3 -m unittest tests/test_export_flash_moe_ane_i8_v2.py
```

The tests compare both optimized absmax implementations against independent
full dequantization, verify record packing and source immutability, exercise
dry-run and partial export, and confirm that validation detects corruption.
