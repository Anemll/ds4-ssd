#!/usr/bin/env python3
"""Export DeepSeek DSpark draft shards into a DS4-owned sidecar package.

The Hugging Face DSpark checkpoints keep draft tensors under `mtp.*` in the
last safetensors shards.  This exporter copies those tensors into three compact
draft-layer files and writes a manifest that the DS4 runtime can validate before
the MPP/Metal draft kernels run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import shutil
import struct
import sys
from collections import Counter, defaultdict
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple


CHUNK = 8 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class TensorMeta:
    name: str
    shard: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


class ExportError(RuntimeError):
    pass


def load_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ExportError(f"{path}: truncated safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
        header = f.read(header_len)
        if len(header) != header_len:
            raise ExportError(f"{path}: truncated safetensors JSON header")
    try:
        parsed = json.loads(header)
    except json.JSONDecodeError as exc:
        raise ExportError(f"{path}: invalid safetensors header: {exc}") from exc
    parsed.pop("__metadata__", None)
    return parsed


def safetensors_data_start(path: pathlib.Path) -> int:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ExportError(f"{path}: truncated safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
    return 8 + header_len


def copy_raw_tensor(src_path: pathlib.Path, meta: TensorMeta, out: BinaryIO) -> int:
    data_start = safetensors_data_start(src_path)
    src_off = data_start + meta.data_offsets[0]
    todo = meta.nbytes
    with src_path.open("rb") as src:
        src.seek(src_off)
        remaining = todo
        while remaining:
            buf = src.read(min(CHUNK, remaining))
            if not buf:
                raise ExportError(f"{src_path}: short read for {meta.name}")
            out.write(buf)
            remaining -= len(buf)
    return todo


def write_converted_tensor(src_path: pathlib.Path, name: str, exec_dtype: str, out: BinaryIO) -> int:
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - exercised only without deps.
        raise ExportError("BF16/F16 conversion requires torch and safetensors") from exc

    with safe_open(str(src_path), framework="pt", device="cpu") as sf:
        tensor = sf.get_tensor(name)
    if exec_dtype == "F16":
        tensor = tensor.to(torch.float16).contiguous()
    elif exec_dtype == "F32":
        tensor = tensor.to(torch.float32).contiguous()
    else:
        raise ExportError(f"unsupported converted exec dtype {exec_dtype} for {name}")
    data = tensor.numpy().tobytes(order="C")
    out.write(data)
    return len(data)


def mtp_layer_and_rest(name: str) -> Tuple[int, str]:
    m = re.match(r"^mtp\.(\d+)\.(.+)$", name)
    if not m:
        raise ExportError(f"not an MTP tensor: {name}")
    return int(m.group(1)), m.group(2)


def scale_companion(name: str) -> Optional[str]:
    if name.endswith(".weight"):
        return name[:-len(".weight")] + ".scale"
    return None


def weight_companion(name: str) -> Optional[str]:
    if name.endswith(".scale"):
        return name[:-len(".scale")] + ".weight"
    return None


def ds4_record_name(name: str) -> str:
    layer, rest = mtp_layer_and_rest(name)
    if rest == "main_norm.weight":
        return "dspark.main_norm.weight"
    if rest.startswith("main_proj."):
        return "dspark.main_proj." + rest.rsplit(".", 1)[-1]
    if rest == "norm.weight":
        return f"dspark.blk.{layer}.output_norm.weight"
    if rest.startswith("markov_head.markov_w1."):
        return "dspark.markov_embd." + rest.rsplit(".", 1)[-1]
    if rest.startswith("markov_head.markov_w2."):
        return "dspark.markov_output." + rest.rsplit(".", 1)[-1]
    if rest.startswith("confidence_head.proj."):
        return "dspark.confidence_proj." + rest.rsplit(".", 1)[-1]
    if rest.startswith("hc_head_linear1."):
        return "dspark.output_hc_linear1." + rest.rsplit(".", 1)[-1]
    if rest.startswith("hc_head_linear2."):
        return "dspark.output_hc_linear2." + rest.rsplit(".", 1)[-1]
    if rest == "attn.attn_sink":
        return f"dspark.blk.{layer}.attn_sinks.weight"
    attn_map = {
        "attn.wq_a.": "attn_q_a",
        "attn.q_norm.": "attn_q_a_norm",
        "attn.wq_b.": "attn_q_b",
        "attn.wkv.": "attn_kv",
        "attn.kv_norm.": "attn_kv_a_norm",
        "attn.wo_a.": "attn_output_a",
        "attn.wo_b.": "attn_output_b",
    }
    for prefix, mapped in attn_map.items():
        if rest.startswith(prefix):
            return f"dspark.blk.{layer}.{mapped}.{rest.rsplit('.', 1)[-1]}"
    if rest == "ffn.gate.weight":
        return f"dspark.blk.{layer}.ffn_gate_inp.weight"
    if rest == "ffn.gate.bias":
        return f"dspark.blk.{layer}.exp_probs_b.bias"
    shared_map = {
        "ffn.shared_experts.w1.": "ffn_gate_shexp",
        "ffn.shared_experts.w3.": "ffn_up_shexp",
        "ffn.shared_experts.w2.": "ffn_down_shexp",
    }
    for prefix, mapped in shared_map.items():
        if rest.startswith(prefix):
            return f"dspark.blk.{layer}.{mapped}.{rest.rsplit('.', 1)[-1]}"
    m = re.match(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$", rest)
    if m:
        expert = int(m.group(1))
        fam = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}[m.group(2)]
        return f"dspark.blk.{layer}.{fam}.expert.{expert}.{m.group(3)}"
    return "dspark.blk.%d.%s" % (layer, rest.replace(".", "_"))


def record_base_name(weight_name: str) -> str:
    mapped = ds4_record_name(weight_name)
    if mapped.endswith(".weight"):
        return mapped[:-len(".weight")]
    return mapped


def expert_family_record_name(layer: int, family: str) -> str:
    fam = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}[family]
    return f"dspark.blk.{layer}.{fam}"


def exec_dtype_for_unpaired(meta: TensorMeta) -> Tuple[str, str]:
    if meta.dtype == "BF16":
        return ("F32", "dense_f32_v1") if len(meta.shape) <= 1 else ("F16", "dense_f16_v1")
    if meta.dtype == "F32":
        return "F32", "dense_f32_v1"
    if meta.dtype == "F16":
        return "F16", "dense_f16_v1"
    if meta.dtype in {"I8", "I32", "U8"}:
        return meta.dtype, f"dense_{meta.dtype.lower()}_raw_v1"
    raise ExportError(f"unsupported unpaired tensor dtype {meta.dtype} for {meta.name}")


def is_expert_weight(name: str) -> Optional[Tuple[int, int, str, str]]:
    layer, rest = mtp_layer_and_rest(name)
    m = re.match(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$", rest)
    if not m:
        return None
    return layer, int(m.group(1)), m.group(2), m.group(3)


def load_source(source_dir: pathlib.Path) -> Tuple[dict, Dict[str, TensorMeta], Dict[str, pathlib.Path], Counter]:
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise ExportError(f"missing config.json in {source_dir}")
    if not index_path.is_file():
        raise ExportError(f"missing model.safetensors.index.json in {source_dir}")
    config = load_json(config_path)
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ExportError("model.safetensors.index.json has no weight_map object")
    mtp_names = sorted(name for name in weight_map if name.startswith("mtp."))
    if not mtp_names:
        raise ExportError("no mtp.* tensors found in index")

    shard_names = sorted({weight_map[name] for name in mtp_names})
    shard_paths: Dict[str, pathlib.Path] = {}
    headers = {}
    for shard in shard_names:
        path = source_dir / shard
        if not path.is_file():
            raise ExportError(f"missing draft shard {path}")
        if path.stat().st_size == 0:
            raise ExportError(f"empty draft shard {path}")
        shard_paths[shard] = path
        headers[shard] = read_safetensors_header(path)

    metas: Dict[str, TensorMeta] = {}
    dtypes = Counter()
    for name in mtp_names:
        shard = weight_map[name]
        info = headers[shard].get(name)
        if not isinstance(info, dict):
            raise ExportError(f"{name} is mapped to {shard} but not present in that shard")
        offsets = info.get("data_offsets")
        shape = info.get("shape")
        dtype = info.get("dtype")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ExportError(f"{name}: invalid safetensors data_offsets")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise ExportError(f"{name}: invalid safetensors dtype/shape")
        meta = TensorMeta(
            name=name,
            shard=shard,
            dtype=dtype,
            shape=tuple(int(x) for x in shape),
            data_offsets=(int(offsets[0]), int(offsets[1])),
        )
        if meta.nbytes < 0:
            raise ExportError(f"{name}: negative tensor byte length")
        metas[name] = meta
        dtypes[dtype] += 1

    non_mtp_in_draft_shards = sum(
        1
        for name, shard in weight_map.items()
        if shard in shard_names and not name.startswith("mtp.")
    )
    if non_mtp_in_draft_shards:
        raise ExportError(
            f"draft shard allowlist contains {non_mtp_in_draft_shards} non-mtp tensors; "
            "download only DSpark draft shards"
        )
    return config, metas, shard_paths, dtypes


def validate_experts(metas: Dict[str, TensorMeta], expert_count: int) -> None:
    seen = defaultdict(set)
    shapes = defaultdict(set)
    for name, meta in metas.items():
        parsed = is_expert_weight(name)
        if not parsed:
            continue
        layer, expert, family, kind = parsed
        seen[(layer, family, kind)].add(expert)
        shapes[(layer, family, kind)].add(meta.shape)

    layers = sorted({mtp_layer_and_rest(name)[0] for name in metas})
    for layer in layers:
        for family in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                key = (layer, family, kind)
                experts = seen.get(key, set())
                if len(experts) != expert_count:
                    raise ExportError(
                        f"layer {layer} {family}.{kind}: expected {expert_count} experts, "
                        f"found {len(experts)}"
                    )
                if experts and (min(experts) != 0 or max(experts) != expert_count - 1):
                    raise ExportError(f"layer {layer} {family}.{kind}: expert ids are not contiguous")
                if len(shapes[key]) != 1:
                    raise ExportError(f"layer {layer} {family}.{kind}: inconsistent shapes {sorted(shapes[key])}")


def model_metadata(config: dict, metas: Dict[str, TensorMeta], variant: Optional[str]) -> Tuple[dict, dict]:
    layers = sorted({mtp_layer_and_rest(name)[0] for name in metas})
    hidden = int(config.get("hidden_size", 0))
    layer_count = int(config.get("num_hidden_layers", 0))
    expert_count = int(config.get("n_routed_experts", 0))
    expert_used = int(config.get("num_experts_per_tok", 0))
    vocab = int(config.get("vocab_size", 0))
    target_layers = (
        config.get("dspark_target_layer_ids")
        or config.get("mtp_target_layer_ids")
        or config.get("nextn_target_layer_ids")
        or []
    )
    target_layers = [int(x) for x in target_layers]
    block_size = int(config.get("dspark_block_size", 5))
    noise_token_id = int(config.get("dspark_noise_token_id", 128799))
    markov_rank = int(config.get("dspark_markov_rank", config.get("markov_rank", 0)))
    window_size = int(
        config.get(
            "dspark_window_size",
            config.get("sliding_window", config.get("swa_window_size", 128)),
        )
    )
    if not variant:
        if hidden == 4096 and layer_count == 43 and expert_count == 256:
            variant = "flash"
        elif hidden == 7168 and layer_count == 61 and expert_count == 384:
            variant = "pro"
        else:
            variant = "unknown"
    model = {
        "variant": variant,
        "hidden_size": hidden,
        "layer_count": layer_count,
        "expert_count": expert_count,
        "expert_used_count": expert_used,
        "vocab_size": vocab,
    }
    dspark = {
        "block_size": block_size,
        "draft_layer_count": len(layers),
        "draft_layer_ids": layers,
        "target_layer_ids": target_layers,
        "markov_rank": markov_rank,
        "window_size": window_size,
        "noise_token_id": noise_token_id,
    }
    return model, dspark


def prepare_out_dir(path: pathlib.Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise ExportError(f"output exists: {path} (pass --force to replace)")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "golden").mkdir()


def export_package(
    source_dir: pathlib.Path,
    out_dir: pathlib.Path,
    config: dict,
    metas: Dict[str, TensorMeta],
    shard_paths: Dict[str, pathlib.Path],
    dtypes: Counter,
    variant: Optional[str],
    repo_id: str,
    force: bool,
) -> dict:
    prepare_out_dir(out_dir, force)
    model, dspark = model_metadata(config, metas, variant)
    validate_experts(metas, model["expert_count"])

    layer_ids = dspark["draft_layer_ids"]
    files = {}
    file_handles: Dict[int, BinaryIO] = {}
    try:
        for layer in layer_ids:
            rel = f"draft_layer_{layer:03d}.bin"
            file_handles[layer] = (out_dir / rel).open("wb")
            files[layer] = {"path": rel, "bytes": 0}

        entries = []
        consumed = set()
        for layer in layer_ids:
            out = file_handles[layer]
            rel_file = files[layer]["path"]
            for family in ("w1", "w3", "w2"):
                weight_metas = []
                scale_metas = []
                for expert in range(model["expert_count"]):
                    wname = f"mtp.{layer}.ffn.experts.{expert}.{family}.weight"
                    sname = f"mtp.{layer}.ffn.experts.{expert}.{family}.scale"
                    if wname not in metas or sname not in metas:
                        raise ExportError(f"missing expert tensor pair {wname}/{sname}")
                    wmeta = metas[wname]
                    smeta = metas[sname]
                    if wmeta.dtype != "I8" or smeta.dtype != "F8_E8M0":
                        raise ExportError(
                            f"{wname}/{sname}: expected I8 + F8_E8M0, got {wmeta.dtype} + {smeta.dtype}"
                        )
                    weight_metas.append(wmeta)
                    scale_metas.append(smeta)

                weight_shape = weight_metas[0].shape
                scale_shape = scale_metas[0].shape
                if any(m.shape != weight_shape for m in weight_metas):
                    raise ExportError(f"layer {layer} {family}: inconsistent expert weight shapes")
                if any(m.shape != scale_shape for m in scale_metas):
                    raise ExportError(f"layer {layer} {family}: inconsistent expert scale shapes")

                data_offset = out.tell()
                data_bytes = 0
                for meta in weight_metas:
                    data_bytes += copy_raw_tensor(shard_paths[meta.shard], meta, out)
                    consumed.add(meta.name)
                scale_offset = out.tell()
                scale_bytes = 0
                for meta in scale_metas:
                    scale_bytes += copy_raw_tensor(shard_paths[meta.shard], meta, out)
                    consumed.add(meta.name)

                entries.append({
                    "name": expert_family_record_name(layer, family),
                    "layer": layer,
                    "expert_major": True,
                    "expert_count": model["expert_count"],
                    "source_tensors": {
                        "pattern": f"mtp.{layer}.ffn.experts.<expert>.{family}.{{weight,scale}}"
                    },
                    "quant_type": "MXFP4_NATIVE",
                    "storage_layout": "mxfp4_plane_split_v1",
                    "shape": [weight_shape[1], weight_shape[0], model["expert_count"]],
                    "source_shape": list(weight_shape),
                    "scale_shape": [scale_shape[1], scale_shape[0], model["expert_count"]],
                    "source_scale_shape": list(scale_shape),
                    "file": rel_file,
                    "plane_data_format": "FP4_E2M1_SEQPAIR",
                    "plane_scale_format": "E8M0",
                    "plane_data_offset": data_offset,
                    "plane_data_bytes": data_bytes,
                    "plane_scale_offset": scale_offset,
                    "plane_scale_bytes": scale_bytes,
                    "bytes_per_expert": weight_metas[0].nbytes + scale_metas[0].nbytes,
                    "plane_data_bytes_per_expert": weight_metas[0].nbytes,
                    "plane_scale_bytes_per_expert": scale_metas[0].nbytes,
                })

        for name in sorted(metas):
            if name in consumed:
                continue
            meta = metas[name]
            weight_name = weight_companion(name)
            if weight_name and weight_name in metas and weight_name not in consumed:
                continue
            layer, _ = mtp_layer_and_rest(name)
            out = file_handles[layer]
            rel_file = files[layer]["path"]
            scale_name = scale_companion(name)
            if scale_name and scale_name in metas:
                scale = metas[scale_name]
                if meta.dtype == "I8" and scale.dtype == "F8_E8M0":
                    quant_type = "MXFP4_NATIVE"
                    layout = "mxfp4_plane_split_v1"
                    data_format = "FP4_E2M1_SEQPAIR"
                elif meta.dtype == "F8_E4M3" and scale.dtype == "F8_E8M0":
                    quant_type = "FP8_E4M3"
                    layout = "fp8_e4m3_e8m0_scale_plane_v1"
                    data_format = "F8_E4M3"
                else:
                    raise ExportError(
                        f"unsupported tensor/scale pair {name} ({meta.dtype}) + {scale_name} ({scale.dtype})"
                    )
                data_offset = out.tell()
                data_bytes = copy_raw_tensor(shard_paths[meta.shard], meta, out)
                scale_offset = out.tell()
                scale_bytes = copy_raw_tensor(shard_paths[scale.shard], scale, out)
                entry = {
                    "name": record_base_name(name),
                    "layer": layer,
                    "source_tensors": {"weight": name, "scale": scale_name},
                    "quant_type": quant_type,
                    "storage_layout": layout,
                    "shape": list(meta.shape),
                    "scale_shape": list(scale.shape),
                    "file": rel_file,
                    "plane_data_format": data_format,
                    "plane_scale_format": "E8M0",
                    "plane_data_offset": data_offset,
                    "plane_data_bytes": data_bytes,
                    "plane_scale_offset": scale_offset,
                    "plane_scale_bytes": scale_bytes,
                }
                entries.append(entry)
                consumed.add(name)
                consumed.add(scale_name)
                continue

            exec_dtype, layout = exec_dtype_for_unpaired(meta)
            offset = out.tell()
            if meta.dtype == exec_dtype and meta.dtype != "BF16":
                nbytes = copy_raw_tensor(shard_paths[meta.shard], meta, out)
            else:
                nbytes = write_converted_tensor(shard_paths[meta.shard], name, exec_dtype, out)
            entries.append({
                "name": ds4_record_name(name),
                "layer": layer,
                "source_tensors": {"tensor": name},
                "source_dtype": meta.dtype,
                "exec_dtype": exec_dtype,
                "storage_layout": layout,
                "shape": list(meta.shape),
                "file": rel_file,
                "offset": offset,
                "bytes": nbytes,
            })
            consumed.add(name)

        missing = sorted(set(metas) - consumed)
        if missing:
            raise ExportError(f"{len(missing)} mtp tensors were not consumed, first={missing[:5]}")
    finally:
        for fh in file_handles.values():
            fh.close()

    for layer in layer_ids:
        path = out_dir / files[layer]["path"]
        files[layer]["bytes"] = path.stat().st_size

    manifest = {
        "schema_version": 1,
        "sidecar_kind": "dspark_draft",
        "storage_layout": "ds4_dspark_draft_v1",
        "source": {
            "repo_id": repo_id,
            "path": str(source_dir),
            "mtp_tensor_count": len(metas),
            "dtype_counts": dict(sorted(dtypes.items())),
        },
        "model": model,
        "dspark": dspark,
        "files": [files[layer] for layer in layer_ids],
        "entries": entries,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def print_summary(manifest: dict) -> None:
    model = manifest["model"]
    dspark = manifest["dspark"]
    print(
        "dspark export: "
        f"variant={model['variant']} hidden={model['hidden_size']} "
        f"layers={model['layer_count']} experts={model['expert_count']} "
        f"draft_layers={dspark['draft_layer_ids']} block={dspark['block_size']} "
        f"target_layers={dspark['target_layer_ids']} markov_rank={dspark['markov_rank']} "
        f"entries={len(manifest['entries'])}"
    )
    for f in manifest.get("files", []):
        print(f"  {f['path']}: {f['bytes']:,} bytes")


def metadata_manifest(
    source_dir: pathlib.Path,
    config: dict,
    metas: Dict[str, TensorMeta],
    dtypes: Counter,
    variant: Optional[str],
    repo_id: str,
) -> dict:
    model, dspark = model_metadata(config, metas, variant)
    validate_experts(metas, model["expert_count"])
    return {
        "schema_version": 1,
        "sidecar_kind": "dspark_draft",
        "storage_layout": "ds4_dspark_draft_v1",
        "source": {
            "repo_id": repo_id,
            "path": str(source_dir),
            "mtp_tensor_count": len(metas),
            "dtype_counts": dict(sorted(dtypes.items())),
        },
        "model": model,
        "dspark": dspark,
        "files": [],
        "entries": [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export DeepSeek DSpark draft tensors for DS4")
    ap.add_argument("--source-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--variant", choices=("flash", "pro", "unknown"), default=None)
    ap.add_argument("--repo-id", default="deepseek-ai/DeepSeek-V4-Flash-DSpark")
    ap.add_argument("--metadata-only", action="store_true", help="validate source and print metadata without writing package")
    ap.add_argument("--force", action="store_true", help="replace existing output directory")
    args = ap.parse_args(argv)

    try:
        config, metas, shard_paths, dtypes = load_source(args.source_dir)
        if args.metadata_only:
            manifest = metadata_manifest(args.source_dir, config, metas, dtypes, args.variant, args.repo_id)
        else:
            manifest = export_package(
                args.source_dir,
                args.out_dir,
                config,
                metas,
                shard_paths,
                dtypes,
                args.variant,
                args.repo_id,
                args.force,
            )
        print_summary(manifest)
    except ExportError as exc:
        print(f"dspark_export.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
