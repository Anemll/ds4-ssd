# Resume Prompt: Hybrid IQ2 Gate/Up + MXFP4 Down Sidecar

Continue the hybrid sidecar experiment in:

```sh
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd
```

Read these first:

```sh
recovered_transcripts/hybrid-sidecar-experiment-019ef25f-019ef264.md
/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar/HANDOFF.md
```

State to preserve:

- Original goal was a model Frankensteining report and prototype: keep IQ2 dense weights, keep `IQ2_XXS` gate/up experts, replace only `down` with `MXFP4_NATIVE`.
- The built sidecar source of truth is `/Volumes/TB36/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar`.
- The recovered final answer claimed `/Users/anemll/Models/DSv4-Flash-Hybrid-IQ2GU-MXFP4Down-sidecar` was a symlink, but recovery found it as a separate ~3.1 GiB directory with only `layer_009.bin`. Verify before using the convenience path.
- Artifact verification passed: `layers=43 bad=0 sum_bytes=96670318592 sum_gib=90.03`.
- Manifest down entries were fixed to use hybrid-local MXFP4 plane offsets:

```text
plane_data_offset  = repacked_offset
plane_scale_offset = repacked_offset + plane_data_bytes
```

Current code blocker:

```text
ds4: Flash-MoE sidecar layer 0 mixes MXFP4 storage layouts
```

Next engineering task:

Support the exact mixed routed layout `gate=IQ2_XXS`, `up=IQ2_XXS`, `down=MXFP4_NATIVE plane-split`. Start with `ssd/ssd_flash_moe_sidecar.c`, then trace required runtime changes through `ds4.c`, `ds4_metal.m`, and `ssd/ssd_flash_moe_*.c`. Keep the support narrowly gated to this known layout until verified.
