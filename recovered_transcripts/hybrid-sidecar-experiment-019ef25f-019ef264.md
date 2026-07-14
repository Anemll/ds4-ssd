# Recovered hybrid sidecar experiment

Recovered from `/Users/anemll/.codex/logs_2.sqlite`.

## Source threads

- `019ef25f-a9e8-7a71-9a8a-12573b412383`: 2026-06-22T19:46:56 to 2026-06-22T19:52:55, 268 rows, 7 response snapshots
- `019ef264-d291-7ac1-b63c-ef6e73437823`: 2026-06-22T20:32:22 to 2026-06-22T20:35:55, 1000 rows, 10 response snapshots

## Carry-forward context

- User goal: prepare specs for a DeepSeek V4 Flash model Frankensteining report using the IQ2 sidecar plus selected MXFP4 expert tensors.
- Input model directories: `/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/` and `/Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/`.
- Corrected tensor facts: IQ2 package uses `gate=IQ2_XXS`, `up=IQ2_XXS`, `down=Q2_K`; MXFP4 package uses `gate/up/down=MXFP4_NATIVE`.
- Recommended hybrid: keep IQ2 dense, keep IQ2 gate/up, replace only down with MXFP4 native plane-split records.
- Built artifact source of truth: `/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar`.
- Handoff file: `/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md`.
- Artifact verification: 43 layer files, `layers=43 bad=0 sum_bytes=96670318592 sum_gib=90.03`.
- Runtime blocker to fix next: `ds4: Flash-MoE sidecar layer 0 mixes MXFP4 storage layouts`.
- Path caveat from recovery: `/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar` is currently not a symlink to TB36; it is a separate ~3.1 GiB directory containing `layer_009.bin`.

## Recovered side prompt

```text

# Files mentioned by the user:

## dsv4-iq2xxs-expert-major: /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/

## DSv4-Flash-MXFP4-unpacked-flash: /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/

## My request for Codex:
I need to prepare specs for model Frankensteining for this report.

So I have a sidecar model for 2-bit quantization, and that was extraction of Guff. After that, I also ported 4-bit quantized version to MXFP4. And quality measurement showed that MXFP4 much better than Q4 quantization because it's close to the native nature of the model. Now, the shared tensors, the shared experts, not shared experts, but the sidecar experts in IQ2, I think at least two out of three tensors weights are Q4. What I want is to create a combined kind of model where the dense weights are as is, like Q2, so we don't gonna modify them yet. But share experts, we keep IQ2 side, IQ2 tensor the same, but Q4 we replace with MXFP4 from Q4 model. Does it make sense?

IQ2: attached and
Q4 tensor. you can analaze.
I expect combinded MoE in IQ2+Q4 should be sizmialr size to IQ2 sidecar, because we replace Q4 tensors only
```

## Recovered final answers

### 1. Spec answer

- Source row: `272678384`
- Time: `2026-06-22T19:52:55`

```markdown
Short answer: conceptually yes, but not exactly as stated.

The provided IQ2 sidecar is not `IQ2 + Q4`. Its expert tensors are:

| Tensor | IQ2 sidecar type | MXFP4 sidecar type |
|---|---:|---:|
| `ffn_gate_exps` | `IQ2_XXS` | `MXFP4_NATIVE` |
| `ffn_up_exps` | `IQ2_XXS` | `MXFP4_NATIVE` |
| `ffn_down_exps` | `Q2_K` | `MXFP4_NATIVE` |

So the practical hybrid spec would be:

**Hybrid-DOWN-MXFP4**
- Dense GGUF: keep IQ2 dense as-is.
- Sidecar `gate`: keep `IQ2_XXS`.
- Sidecar `up`: keep `IQ2_XXS`.
- Sidecar `down`: replace `Q2_K` with `MXFP4_NATIVE`.
- Repack each layer file expert-by-expert; whole-layer file swapping will not work because each `layer_###.bin` packs all three tensors together using one expert stride.

Size estimate:

| Model | Routed sidecar size | With IQ2 dense |
|---|---:|---:|
| Current IQ2 sidecar | ~72.55 GiB | ~80.75 GiB |
| Hybrid down-only MXFP4 | ~90.04 GiB | ~98.24 GiB |
| Full MXFP4 sidecar | ~137.09 GiB | ~145.29 GiB |

So it is same general scale as IQ2, but not “similar size” in a strict sense. Replacing only `down` adds about **17.47 GiB** routed sidecar size, roughly **+24%** over the current IQ2 sidecar.

Main risk: current runtime support probably does **not** fully support `IQ2_XXS / IQ2_XXS / MXFP4_NATIVE` yet. Existing code appears to advertise/support these combinations:

- `IQ2_XXS / IQ2_XXS / Q2_K`
- `IQ2_XXS / IQ2_XXS / IQ2_XXS`
- `MXFP4 / MXFP4 / MXFP4`
- some `Q4_K / Q4_K / Q4_K`

The hybrid would likely need explicit decode/prefill kernel support and compatibility gating before it is runtime-safe.

For the report, I would phrase it as:

> A hybrid sidecar is technically feasible by retaining IQ2 dense weights and retaining IQ2_XXS gate/up expert tensors while replacing the Q2_K down expert tensor with MXFP4_NATIVE. This produces a middle-size model between the current IQ2 sidecar and full MXFP4 sidecar, with expected routed expert storage of ~90.0 GiB versus ~72.6 GiB for IQ2 and ~137.1 GiB for full MXFP4. However, it is not a metadata-only merge: each layer must be repacked with a new expert stride, and runtime kernels must explicitly support the mixed `IQ2_XXS/IQ2_XXS/MXFP4_NATIVE` expert layout.
```

### 2. Export and handoff answer

- Source row: `273490812`
- Time: `2026-06-22T20:35:55`

```markdown
Done. The hybrid sidecar is available at:

`/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar`

That path is a symlink to the real payload on TB36 because `/Users` only had ~64 GiB free and the sidecar is ~90 GiB. You can still use the `~/Models/...` path in ds4 commands.

What was created:

- gate: `IQ2_XXS`
- up: `IQ2_XXS`
- down: `MXFP4_NATIVE`
- routed data: `90.03 GiB`
- dense dir: symlink to the original IQ2 dense package

I fixed the manifest plane offsets for the MXFP4 down records and verified:

`layers=43 bad=0 sum_gib=90.03`

Smoke test reaches the expected code blocker:

```text
ds4: Flash-MoE sidecar layer 0 mixes MXFP4 storage layouts
```

So yes, next step is code support. Current loader rejects mixed MXFP4 family layouts, and runtime likely needs an explicit `IQ2_XXS/IQ2_XXS/MXFP4_NATIVE` path.

Handoff is here:

`/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md`
```

## Timeline

- `2026-06-22T19:51:34` assistant: I’ll inspect the two model directories read-only: file layout, sizes, and any metadata/index files that identify tensor names/types. Then I’ll give you a concrete Frankenstein spec and call out size/compatibility risks.
- `2026-06-22T19:51:42` assistant: Both sidecars have the same basic unpacked layout: per-layer `layer_###.bin`, manifest, and dense GGUF package. The MXFP4 sidecar is much larger overall, so the key question is whether tensor records are swappable at record granularity rather than whole-layer file granularity. I’ll inspect manifests and record format metadata.
- `2026-06-22T19:51:54` assistant: The manifests already expose the key mismatch: the IQ2 sidecar is not IQ2+Q4 in these files. It is `gate=IQ2_XXS`, `up=IQ2_XXS`, `down=Q2_K`. The MXFP4 sidecar is `gate/up/down=MXFP4_NATIVE`. I’ll compute exact replacement sizes for plausible combinations.
- `2026-06-22T19:52:08` assistant: A whole-layer file swap will not work because each `layer_###.bin` contains all three families packed expert-major. A Frankenbuild must repack each layer with selected per-family records and update manifest offsets/strides. I’ll check whether the loader accepts mixed quant types per layer or assumes one stride/layout.
- `2026-06-22T20:32:55` assistant: And yes: we’ll need JSON/runtime changes. The manifest can describe the mixed layout, but current code has an explicit guard rejecting “only down is MXFP4 plane-split,” and the compute path probably assumes all three families are MXFP4 for native MXFP4 routed kernels. I’ll capture that in the handoff after the smoke test.
- `2026-06-22T20:33:28` assistant: Layer 40 done, only two more after the current one. Then I’ll verify sizes, ensure the `~/Models` symlink is present, run the expected startup smoke, and write the handoff.
- `2026-06-22T20:33:58` assistant: The hybrid sidecar finished: `90.03 GiB` routed data in about 14 minutes. I’m doing the consistency checks now and then I’ll run the loader smoke.
- `2026-06-22T20:34:18` assistant: The `~/Models` path is a symlink to TB36 as intended. One verification command didn’t follow the symlink, so I’m rerunning that against the real target to count the layer files.
- `2026-06-22T20:34:23` assistant: Good catch from the verifier: the down entries had the new `repacked_offset` but still carried old MXFP plane offsets from the full-MXFP sidecar. I’m fixing the manifest so `plane_data_offset` and `plane_scale_offset` match the hybrid record layout.
- `2026-06-22T20:34:41` assistant: The manifest is now internally consistent. I’m running a tiny startup/inference smoke through the `~/Models/...` symlink path with `DS4_MXFP4_NATIVE=1`; I expect the first blocker to be the mixed-family MXFP4 guard.
- `2026-06-22T20:35:00` assistant: Smoke hit exactly the expected blocker: `ds4: Flash-MoE sidecar layer 0 mixes MXFP4 storage layouts`. So the JSON is good enough to reach validation, and the next step is code support for mixed `IQ2_XXS/IQ2_XXS/MXFP4_NATIVE`. I’ll write the handoff now with the artifact paths, manifest fix, smoke output, and precise code gates.
- `2026-06-22T20:35:17` assistant: The repo is already dirty from the main thread, so I’m not touching source in this side conversation. I’ll put the handoff inside the new sidecar directory, where it travels with the artifact and is visible through the `~/Models/...` symlink.
- `2026-06-22T20:33:47` export process finished after writing layers 041 and 042; total routed payload was 90.03 GiB.
- `2026-06-22T20:34:29` manifest MXFP4 down entries were patched so `plane_data_offset=repacked_offset` and `plane_scale_offset=repacked_offset+plane_data_bytes`.
- `2026-06-22T20:34:42` smoke test reached the expected loader/runtime blocker instead of a missing-file or corrupt-manifest failure.
- `2026-06-22T20:35:28` `HANDOFF.md` was created under the TB36 payload.

## Important command outputs

### `2026-06-22T19:51:54` output `call_AKcf5cduLWzIN8Uao30BlUcg`

```text
Chunk ID: 74670a
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 35
Output:
IQ2_XXS count=86 bytes_per_expert_set=2162688 total_exact=47613739008
Q2_K count=43 bytes_per_expert_set=2752512 total_exact=30299652096
```

### `2026-06-22T19:51:54` output `call_B9dQ20IrNq7Ymee16b8CNUbG`

```text
Chunk ID: 853c0f
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 20
Output:
MXFP4_NATIVE count=129 bytes_per_expert_set=4456448 total_exact=147169738752
```

### `2026-06-22T19:51:54` output `call_vUdSPCuQNhpmPhKo5oqvTizm`

```text
Chunk ID: caaa10
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 57
Output:
ffn_down_exps quant=Q2_K bytes_per_expert=2752512 exact_total=30299652096
ffn_gate_exps quant=IQ2_XXS bytes_per_expert=2162688 exact_total=23806869504
ffn_up_exps quant=IQ2_XXS bytes_per_expert=2162688 exact_total=23806869504
```

### `2026-06-22T19:51:54` output `call_IVzekCsoBcbsPcI5lsq50x2b`

```text
Chunk ID: 0fc368
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 61
Output:
ffn_down_exps quant=MXFP4_NATIVE bytes_per_expert=4456448 exact_total=49056579584
ffn_gate_exps quant=MXFP4_NATIVE bytes_per_expert=4456448 exact_total=49056579584
ffn_up_exps quant=MXFP4_NATIVE bytes_per_expert=4456448 exact_total=49056579584
```

### `2026-06-22T19:52:08` output `call_rTa91ymVBkoNr6K6UIq1adoP`

```text
Chunk ID: 4a6f34
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 9757
Output:
ssd/ssd_flash_moe_sidecar.c:2: * ssd_flash_moe_sidecar.c - SSD Flash-MoE sidecar manifest and slot sizing.
ssd/ssd_flash_moe_sidecar.c:6: * weight-binding code can parse sidecar manifests before the graph-private
ssd/ssd_flash_moe_sidecar.c:17:        fprintf(stderr, "ds4: failed to open Flash-MoE manifest %s: %s\n", path, strerror(errno));
ssd/ssd_flash_moe_sidecar.c:160:static uint32_t flash_moe_quant_type_id(const char *quant, bool *mxfp4_plane_split) {
ssd/ssd_flash_moe_sidecar.c:163:    if (!strcmp(quant, "IQ2_XXS")) return DS4_TENSOR_IQ2_XXS;
ssd/ssd_flash_moe_sidecar.c:166:    if (!strcmp(quant, "Q2_K")) return DS4_TENSOR_Q2_K;
ssd/ssd_flash_moe_sidecar.c:169:    if (!strcmp(quant, "MXFP4")) return DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:170:    if (!strcmp(quant, "MXFP4_NATIVE")) {
ssd/ssd_flash_moe_sidecar.c:172:        return DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:264:        !flash_moe_json_string(obj, end, "quant_type", &quant) ||
ssd/ssd_flash_moe_sidecar.c:267:        !flash_moe_json_u64(obj, end, "bytes_per_expert", &bytes) ||
ssd/ssd_flash_moe_sidecar.c:275:    (void)flash_moe_json_u64(obj, end, "expert_stride", &stride);
ssd/ssd_flash_moe_sidecar.c:281:    const uint32_t type = flash_moe_quant_type_id(quant, &mxfp4_plane_split);
ssd/ssd_flash_moe_sidecar.c:295:                "ds4: Flash-MoE MXFP4_NATIVE entry requires storage_layout=mxfp4_plane_split_v1\n");
ssd/ssd_flash_moe_sidecar.c:318:                    "ds4: Flash-MoE MXFP4_NATIVE entry has invalid plane geometry\n");
ssd/ssd_flash_moe_sidecar.c:325:        uint64_t manifest_plane_data = 0;
ssd/ssd_flash_moe_sidecar.c:326:        uint64_t manifest_plane_scale = 0;
ssd/ssd_flash_moe_sidecar.c:327:        uint64_t manifest_plane_data_off = 0;
ssd/ssd_flash_moe_sidecar.c:328:        uint64_t manifest_plane_scale_off = 0;
ssd/ssd_flash_moe_sidecar.c:330:            (flash_moe_json_u64(obj, end, "plane_data_bytes", &manifest_plane_data) &&
ssd/ssd_flash_moe_sidecar.c:331:             manifest_plane_data != plane_data_bytes) ||
ssd/ssd_flash_moe_sidecar.c:332:            (flash_moe_json_u64(obj, end, "plane_scale_bytes", &manifest_plane_scale) &&
ssd/ssd_flash_moe_sidecar.c:333:             manifest_plane_scale != plane_scale_bytes) ||
ssd/ssd_flash_moe_sidecar.c:334:            (flash_moe_json_u64(obj, end, "plane_data_offset", &manifest_plane_data_off) &&
ssd/ssd_flash_moe_sidecar.c:335:             manifest_plane_data_off != offset) ||
ssd/ssd_flash_moe_sidecar.c:336:            (flash_moe_json_u64(obj, end, "plane_scale_offset", &manifest_plane_scale_off) &&
ssd/ssd_flash_moe_sidecar.c:337:             manifest_plane_scale_off != offset + plane_data_bytes)) {
ssd/ssd_flash_moe_sidecar.c:339:                    "ds4: Flash-MoE MXFP4_NATIVE entry plane fields do not match deterministic layout\n");
ssd/ssd_flash_moe_sidecar.c:381:        if (layer->expert_stride != 0 && layer->expert_stride != stride) {
ssd/ssd_flash_moe_sidecar.c:388:        layer->expert_stride = stride;
ssd/ssd_flash_moe_sidecar.c:470:    char *manifest_path = flash_moe_join_path(sidecar_dir, "manifest.json");
ssd/ssd_flash_moe_sidecar.c:471:    if (access(manifest_path, R_OK) != 0) {
ssd/ssd_flash_moe_sidecar.c:472:        free(manifest_path);
ssd/ssd_flash_moe_sidecar.c:475:        manifest_path = flash_moe_join_path(sidecar_dir, "manifest.json");
ssd/ssd_flash_moe_sidecar.c:480:    if (!flash_moe_read_file(manifest_path, &json, &json_len)) {
ssd/ssd_flash_moe_sidecar.c:481:        free(manifest_path);
ssd/ssd_flash_moe_sidecar.c:485:    free(manifest_path);
ssd/ssd_flash_moe_sidecar.c:517:        fprintf(stderr, "ds4: Flash-MoE sidecar manifest does not contain the expected DS4 routed entries\n");
ssd/ssd_flash_moe_sidecar.c:536:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_IQ2_XXS &&
ssd/ssd_flash_moe_sidecar.c:537:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_IQ2_XXS &&
ssd/ssd_flash_moe_sidecar.c:538:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_Q2_K;
ssd/ssd_flash_moe_sidecar.c:543:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_MXFP4 &&
ssd/ssd_flash_moe_sidecar.c:544:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_MXFP4 &&
ssd/ssd_flash_moe_sidecar.c:545:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:589:                if (layer->family_type[fam] != DS4_TENSOR_MXFP4 ||
ssd/ssd_flash_moe_sidecar.c:594:                    fprintf(stderr, "ds4: Flash-MoE sidecar layer %u has invalid MXFP4_NATIVE plane geometry\n", il);
ssd/ssd_flash_moe_sidecar.c:622:            fprintf(stderr, "ds4: Flash-MoE sidecar layer %u mixes MXFP4 storage layouts\n", il);
ssd/ssd_flash_moe_sidecar.c:634:

[truncated in Markdown at 5000 chars; see parsed JSONL for full output]
```

### `2026-06-22T20:33:58` output `call_fJlPOBQ7MZGGRjEQBssxZED3`

```text
Chunk ID: 8dbd30
Wall time: 0.3223 seconds
Process exited with code 0
Original token count: 118
Output:
schema=2 storage_layout=mixed_iq2xxs_iq2xxs_mxfp4_down_v1 hybrid=IQ2_XXS/IQ2_XXS/MXFP4_NATIVE
ffn_down_exps quant=MXFP4_NATIVE count=43 bytes_per_expert=4456448 stride=8781824 offset=4325376 exact_gib=45.69 storage=mxfp4_plane_split_v1
ffn_gate_exps quant=IQ2_XXS count=43 bytes_per_expert=2162688 stride=8781824 offset=0 exact_gib=22.17 storage=none
ffn_up_exps quant=IQ2_XXS count=43 bytes_per_expert=2162688 stride=8781824 offset=2162688 exact_gib=22.17 storage=none
```

### `2026-06-22T20:34:19` output `call_9TAGwspVwxjdx8130FPSZxr4`

```text
Chunk ID: 2679b6
Wall time: 0.0355 seconds
Process exited with code 0
Original token count: 90
Output:
layers=43 bad=0 sum_bytes=96670318592 sum_gib=90.03
/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/layer_000.bin
/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/layer_001.bin
/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/layer_002.bin
/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/layer_042.bin
```

### `2026-06-22T20:34:30` output `call_bBDmY7p3ZJROkWBdR1pKBurQ`

```text
Chunk ID: 9dc937
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 27
Output:
fixed /Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/manifest.json
manifest plane fields ok
```

### `2026-06-22T20:34:42` output `call_UxObKFizaYkJdcX2BQPxGvku`

```text
Chunk ID: a40f24
Wall time: 0.0518 seconds
Process exited with code 1
Original token count: 123
Output:
ds4: applied sidecar tuning profile [Apple M5 Max] from ds4_profile.json (51 env defaults set, 0 kept from environment)
ds4: warning: raw KV cap limits Metal prefill chunk (prefill_chunk=256, raw_kv_rows=256, raw_window=128, effective_chunk=128; target raw cap for one chunk is 512, also requires ctx >= that value)
ds4: context buffers 25.17 MiB (ctx=256, backend=metal, prefill_chunk=256, raw_kv_rows=256, compressed_kv_rows=66)
ds4: Flash-MoE sidecar layer 0 mixes MXFP4 storage layouts
```

### `2026-06-22T20:35:01` output `call_PvNeNq1Yi5fVmxOdelNpvCis`

```text
Chunk ID: 1206ad
Wall time: 0.0000 seconds
Process exited with code 0
Original token count: 2030
Output:
ssd/ssd_flash_moe_sidecar.c:166:    if (!strcmp(quant, "Q2_K")) return DS4_TENSOR_Q2_K;
ssd/ssd_flash_moe_sidecar.c:169:    if (!strcmp(quant, "MXFP4")) return DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:172:        return DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:377:    layer->family_mxfp4_plane_split[fam] = mxfp4_plane_split;
ssd/ssd_flash_moe_sidecar.c:538:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_Q2_K;
ssd/ssd_flash_moe_sidecar.c:543:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_GATE] == DS4_TENSOR_MXFP4 &&
ssd/ssd_flash_moe_sidecar.c:544:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_UP] == DS4_TENSOR_MXFP4 &&
ssd/ssd_flash_moe_sidecar.c:545:        s->layer[first_routed_layer].family_type[DS4_FLASH_FAMILY_DOWN] == DS4_TENSOR_MXFP4;
ssd/ssd_flash_moe_sidecar.c:586:            if (layer->family_mxfp4_plane_split[fam]) {
ssd/ssd_flash_moe_sidecar.c:589:                if (layer->family_type[fam] != DS4_TENSOR_MXFP4 ||
ssd/ssd_flash_moe_sidecar.c:618:        if (layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] !=
ssd/ssd_flash_moe_sidecar.c:619:                layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] ||
ssd/ssd_flash_moe_sidecar.c:620:            layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] !=
ssd/ssd_flash_moe_sidecar.c:621:                layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN]) {
ssd/ssd_flash_moe_sidecar.c:622:            fprintf(stderr, "ds4: Flash-MoE sidecar layer %u mixes MXFP4 storage layouts\n", il);
ssd/ssd_flash_moe_sidecar.c:709:                    "but native MXFP4 MPP 4.1 support is not available on this host\n");
ssd/ssd_flash_moe_sidecar.c:733:        s->layer[first_routed_layer].family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE]) {
ds4.c:430: *   - Q2_K routed down experts
ds4.c:1313:    DS4_TENSOR_Q2_K     = 10,
ds4.c:1323:    DS4_TENSOR_MXFP4    = 39,
ds4.c:1884:    if (t->type == DS4_TENSOR_IQ2_XXS || t->type == DS4_TENSOR_Q2_K) return false;
ds4.c:2178: * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
ds4.c:2323:/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
ds4.c:2745:    bool family_mxfp4_plane_split[DS4_FLASH_FAMILY_COUNT];
ds4.c:2949:           type == DS4_TENSOR_Q2_K ||
ds4.c:2954:           type == DS4_TENSOR_MXFP4;
ds4.c:2963:    case DS4_TENSOR_Q2_K:    return 84;
ds4.c:2968:    case DS4_TENSOR_MXFP4:   return 17;
ds4.c:2975:    return type == DS4_TENSOR_MXFP4 ? QK_MXFP4 : QK_K;
ds4.c:4868:/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
ds4.c:4875:    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
ds4.c:4879:    if (in_dim % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");
ds4.c:4920:/* Accumulate all selected experts' Q2_K down projections directly into the
ds4.c:4929:    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
ds4.c:4944:            ds4_die("Q2_K expert tensors do not share a layout");
ds4.c:4947:    if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");
ds4.c:5969: * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
ds4.c:6241: * applies SwiGLU and router weights, then accumulates Q2_K down projections. */
ds4.c:6262:    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");
ds4.c:6359:    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");
ds4.c:6406:    if (down_in_dim % QK_K != 0) ds4_die("Q2_K expert input is not QK_K aligned");
ds4.c:6486:            ds4_die("IQ2_XXS batch expert tensor layout mismatch");
ds4.c:6523:            ds4_die("Q2_K batch expert tensor layout mismatch");
ds4.c:11258:            flash_layer->family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE];
ds4.c:17255:        g->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
ds4.c:17256:        g->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
ds4.c:17257:        g->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
ds4.c:17311:        routed = "MPP 4.1 native MXFP4 (plane-sidecar/no-repack)";
ds4.c:21875:            e->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
ds4.c:21876:            e->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
ds4.c:21877:            e->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_DOWN];
ds4.c:21883:                    "using plane-safe native MXFP4 routed paths\n");
ds4.c:22221:                e->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_GATE] &&
ds4.c:22222:                e->flash_moe->layer[0].family_mxfp4_plane_split[DS4_FLASH_FAMILY_UP] &&
ds4.c:22223:                e->flash_moe->layer[0].family_mxfp4

[truncated in Markdown at 5000 chars; see parsed JSONL for full output]
```

## Tool calls

The parsed JSONL contains full call/output records. This Markdown lists the issued commands at a readable level.

1. `2026-06-22T19:51:34` `exec_command`
```sh
find /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major -maxdepth 2 -type f | sed -n '1,120p'
```

2. `2026-06-22T19:51:34` `exec_command`
```sh
find /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash -maxdepth 2 -type f | sed -n '1,120p'
```

3. `2026-06-22T19:51:34` `exec_command`
```sh
du -sh /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash
```

4. `2026-06-22T19:51:34` `exec_command`
```sh
find /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash -maxdepth 2 \( -name '*.json' -o -name '*.txt' -o -name '*.csv' -o -name '*.idx' -o -name '*.meta' \) -type f -print
```

5. `2026-06-22T19:51:42` `exec_command`
```sh
sed -n '1,220p' /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/manifest.json
```

6. `2026-06-22T19:51:42` `exec_command`
```sh
sed -n '1,220p' /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/manifest.json
```

7. `2026-06-22T19:51:42` `exec_command`
```sh
sed -n '1,220p' /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/flashmoe-package.json
```

8. `2026-06-22T19:51:42` `exec_command`
```sh
sed -n '1,220p' /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/dense/flashmoe-package.json
```

9. `2026-06-22T19:51:54` `exec_command`
```sh
jq -r '.entries | group_by(.quant_type)[] | "\(.[0].quant_type) count=\(length) bytes_per_expert_set=\([.[].bytes_per_expert] | unique | join(",")) total_exact=\(map(.exact_byte_length)|add)"' /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/manifest.json
```

10. `2026-06-22T19:51:54` `exec_command`
```sh
jq -r '.entries | group_by(.quant_type)[] | "\(.[0].quant_type) count=\(length) bytes_per_expert_set=\([.[].bytes_per_expert] | unique | join(",")) total_exact=\(map(.exact_byte_length)|add)"' /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/manifest.json
```

11. `2026-06-22T19:51:54` `exec_command`
```sh
jq -r '.entries | group_by(.tensor_family)[] | "\(.[0].tensor_family) quant=\([.[].quant_type] | unique | join(",")) bytes_per_expert=\([.[].bytes_per_expert] | unique | join(",")) exact_total=\(map(.exact_byte_length)|add)"' /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/manifest.json
```

12. `2026-06-22T19:51:54` `exec_command`
```sh
jq -r '.entries | group_by(.tensor_family)[] | "\(.[0].tensor_family) quant=\([.[].quant_type] | unique | join(",")) bytes_per_expert=\([.[].bytes_per_expert] | unique | join(",")) exact_total=\(map(.exact_byte_length)|add)"' /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/manifest.json
```

13. `2026-06-22T19:52:08` `exec_command`
```sh
rg -n "quant_type|expert_stride|bytes_per_expert|MXFP4|IQ2_XXS|Q2_K|manifest" ssd/ssd_flash_moe_sidecar.c ssd/*.c ds4.c ds4.h
```

14. `2026-06-22T19:52:08` `exec_command`
```sh
find /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major -maxdepth 1 -name 'layer_*.bin' -exec stat -f '%N %z' {} \; | sort | awk '{sum+=$2; if(NR<=3||NR>40) print} END{print "layers",NR,"sum",sum}'
```

15. `2026-06-22T19:52:08` `exec_command`
```sh
find /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash -maxdepth 1 -name 'layer_*.bin' -exec stat -f '%N %z' {} \; | sort | awk '{sum+=$2; if(NR<=3||NR>40) print} END{print "layers",NR,"sum",sum}'
```

16. `2026-06-22T19:52:08` `exec_command`
```sh
stat -f '%N %z' /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense/model-dense.gguf /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash/dense/model-dense.gguf
```

17. `2026-06-22T20:32:22` `write_stdin`
```json
{
  "session_id": 12045,
  "chars": "",
  "yield_time_ms": 30000,
  "max_output_tokens": 20000
}
```

18. `2026-06-22T20:32:55` `write_stdin`
```json
{
  "session_id": 12045,
  "chars": "",
  "yield_time_ms": 30000,
  "max_output_tokens": 20000
}
```

19. `2026-06-22T20:33:29` `write_stdin`
```json
{
  "session_id": 12045,
  "chars": "",
  "yield_time_ms": 30000,
  "max_output_tokens": 20000
}
```

20. `2026-06-22T20:33:58` `exec_command`
```sh
set -e
OUT=/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
ls -ld "$OUT" /Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
readlink "$OUT" || true
du -sh "$OUT" /Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
find "$OUT" -maxdepth 1 -name 'layer_*.bin' -exec stat -f '%z' {} \; | awk '{n++; sum+=$1; if($1!=2248146944){bad++}} END{printf "layers=%d bad=%d sum_bytes=%d sum_gib=%.2f\n", n, bad+0, sum, sum/1024/1024/1024}'
```

21. `2026-06-22T20:33:58` `exec_command`
```sh
jq -r '
  "schema=\(.schema_version) storage_layout=\(.storage_layout) hybrid=\(.model.hybrid_expert_layout)",
  (.entries | group_by(.tensor_family)[] | "\(.[0].tensor_family) quant=\([.[].quant_type] | unique | join(",")) count=\(length) bytes_per_expert=\([.[].bytes_per_expert] | unique | join(",")) stride=\([.[].expert_stride] | unique | join(",")) offset=\([.[].repacked_offset] | unique | join(",")) exact_gib=\(((map(.exact_byte_length)|add)/1024/1024/1024*100|round/100)) storage=\([.[].storage_layout // "none"] | unique | join(","))")
' /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/manifest.json
```

22. `2026-06-22T20:33:58` `exec_command`
```sh
ls -l /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/dense && ls -l /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/dense/
```

23. `2026-06-22T20:34:18` `exec_command`
```sh
OUT=/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
find "$OUT" -maxdepth 1 -name 'layer_*.bin' -exec stat -f '%z' {} \; | awk '{n++; sum+=$1; if($1!=2248146944){bad++}} END{printf "layers=%d bad=%d sum_bytes=%d sum_gib=%.2f\n", n, bad+0, sum, sum/1024/1024/1024}'
find "$OUT" -maxdepth 1 -name 'layer_*.bin' | sort | sed -n '1,3p;$p'
```

24. `2026-06-22T20:34:18` `exec_command`
```sh
OUT=/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
python3 - <<'PY'
import json, os
p='/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/manifest.json'
d=json.load(open(p))
stride=set(e['expert_stride'] for e in d['entries'])
offs={fam:sorted(set(e['repacked_offset'] for e in d['entries'] if e['tensor_family']==fam)) for fam in ['ffn_gate_exps','ffn_up_exps','ffn_down_exps']}
print('entries', len(d['entries']), 'stride', stride, 'offsets', offs)
for e in d['entries']:
    if e['tensor_family']=='ffn_down_exps':
        assert e['quant_type']=='MXFP4_NATIVE'
        assert e['storage_layout']=='mxfp4_plane_split_v1'
        assert e['plane_data_offset']==e['repacked_offset']
        assert e['plane_scale_offset']==e['repacked_offset']+e['plane_data_bytes']
print('down plane fields ok')
PY
```

25. `2026-06-22T20:34:29` `exec_command`
```sh
python3 - <<'PY'
import json, os
from pathlib import Path
p=Path('/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/manifest.json')
d=json.load(open(p))
for e in d['entries']:
    if e['tensor_family']=='ffn_down_exps':
        e['plane_data_offset']=e['repacked_offset']
        e['plane_scale_offset']=e['repacked_offset'] + e['plane_data_bytes']
        e['storage_layout']='mxfp4_plane_split_v1'
        e['source_storage_layout']='ggml_block_mxfp4_split_half_v1'
    else:
        for k in ['storage_layout','source_storage_layout','plane_data_format','plane_scale_format','plane_data_offset','plane_data_bytes','plane_scale_offset','plane_scale_bytes','source_quant_type']:
            e.pop(k, None)
tmp=p.with_suffix('.json.tmp')
with open(tmp,'w') as f:
    json.dump(d,f,indent=2)
    f.write('\n')
os.replace(tmp,p)
print('fixed', p)
PY
python3 - <<'PY'
import json
p='/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/manifest.json'
d=json.load(open(p))
for e in d['entries']:
    if e['tensor_family']=='ffn_down_exps':
        assert e['plane_data_offset']==e['repacked_offset']
        assert e['plane_scale_offset']==e['repacked_offset']+e['plane_data_bytes']
print('manifest plane fields ok')
PY
```

26. `2026-06-22T20:34:42` `exec_command`
```sh
DS4_MXFP4_NATIVE=1 ./ds4 \
  -m /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/dense/model-dense.gguf \
  --moe-sidecar /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar \
  --moe-mode slot-bank \
  --moe-slot-bank 8 \
  -c 256 -n 1 -p 'hi' --temp 0
```

27. `2026-06-22T20:35:00` `exec_command`
```sh
git status --short
```

28. `2026-06-22T20:35:00` `exec_command`
```sh
rg -n "mixes MXFP4 storage layouts|all_mxfp4|family_mxfp4_plane_split|native MXFP4|Q2_K|IQ2_XXS batch|DS4_TENSOR_MXFP4" ssd/ssd_flash_moe_sidecar.c ds4.c ds4_metal.m | sed -n '1,260p'
```

29. `2026-06-22T20:35:28` `apply_patch`
```diff
*** Begin Patch
*** Add File: /Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md
+# Hybrid IQ2 Gate/Up + MXFP4 Down Sidecar Handoff
+
+Created: 2026-06-22
+
+## Artifact
+
+Real payload:
+
+```sh
+/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
+```
+
+Convenience path:
+
+```sh
+/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar
+```
+
+The `~/Models` path is a symlink to TB36 because `/Users` had only about 64 GiB free and the routed sidecar is about 90 GiB.
+
+Dense package is also symlinked to the IQ2 dense package:
+
+```sh
+dense -> /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major/dense
+```
+
+## Layout
+
+Per expert record:
+
+```text
+offset 0        ffn_gate_exps  IQ2_XXS       2,162,688 bytes
+offset 2162688  ffn_up_exps    IQ2_XXS       2,162,688 bytes
+offset 4325376  ffn_down_exps  MXFP4_NATIVE  4,456,448 bytes
+stride          total                       8,781,824 bytes
+```
+
+Per layer:
+
+```text
+8,781,824 bytes/expert * 256 experts = 2,248,146,944 bytes
+```
+
+All 43 routed layers:
+
+```text
+96,670,318,592 bytes = 90.03 GiB
+```
+
+Manifest summary:
+
+```text
+ffn_gate_exps: IQ2_XXS       43 entries, 22.17 GiB
+ffn_up_exps:   IQ2_XXS       43 entries, 22.17 GiB
+ffn_down_exps: MXFP4_NATIVE  43 entries, 45.69 GiB
+```
+
+The MXFP4 down entries use `storage_layout=mxfp4_plane_split_v1` and corrected hybrid-local plane offsets:
+
+```text
+plane_data_offset  = repacked_offset
+plane_scale_offset = repacked_offset + plane_data_bytes
+```
+
+## Verification Done
+
+Layer file sizes:
+
+```text
+layers=43 bad=0 sum_bytes=96670318592 sum_gib=90.03
+```
+
+Manifest plane-offset verification passed after fixing the down entries.
+
+Startup smoke command:
+
+```sh
+DS4_MXFP4_NATIVE=1 ./ds4 \
```

30. `2026-06-22T20:35:43` `exec_command`
```sh
ls -l /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md && sed -n '1,20p' /Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md
```

## Artifacts

- Raw SQLite rows: `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/recovered_transcripts/hybrid-sidecar-experiment-019ef25f-019ef264.raw.jsonl`
- Parsed events: `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/recovered_transcripts/hybrid-sidecar-experiment-019ef25f-019ef264.parsed.jsonl`
- Machine summary: `/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd/recovered_transcripts/hybrid-sidecar-experiment-019ef25f-019ef264.summary.json`
- Existing TB36 handoff: `/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md`
