#!/usr/bin/env python3
"""Build a v2 Flash-MoE expert-major sidecar with ANE INT8 scales.

The source sidecar is opened read-only.  Each destination expert record is:

    [unchanged v1 expert record]
    [gate per-output-channel fp16 scales]
    [up per-output-channel fp16 scales]
    [down per-output-channel fp16 scales]
    [zero alignment padding]

Scales are symmetric INT8 dequantization multipliers (absmax / 127) computed
from the exact values represented by the source IQ2_XXS and Q2_K blocks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np


QK_K = 256
RECORD_ALIGNMENT = 64
LAYOUT = "layer_major_expert"
STORAGE_LAYOUT = "expert_major_weights_plus_ane_i8_output_scales_v1"
SCALE_SCHEMA_VERSION = 1
FAMILY_ORDER = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
EXPECTED_QUANT = {
    "ffn_gate_exps": "IQ2_XXS",
    "ffn_up_exps": "IQ2_XXS",
    "ffn_down_exps": "Q2_K",
}
BLOCK_BYTES = {"IQ2_XXS": 66, "Q2_K": 84}

# Maximum unsigned magnitude in each of llama.cpp's 256 IQ2_XXS grid rows.
# Signs do not affect an output row's absmax.  Keeping this compact lookup in
# the exporter avoids expanding 256 represented weights per block.
IQ2_XXS_GRID_ABSMAX = np.asarray(
    [
        8, 43, 25, 43, 43, 25, 25, 43, 43, 43, 43, 25, 25, 25, 43, 43,
        43, 43, 43, 43, 43, 25, 25, 25, 25, 25, 43, 43, 43, 43, 43, 43,
        25, 25, 25, 43, 43, 25, 43, 43, 43, 43, 43, 43, 43, 25, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 25, 25, 25, 43, 25, 43, 43, 43,
        43, 25, 43, 43, 25, 43, 43, 43, 43, 25, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 25, 25, 25, 43, 43, 43, 25, 43, 43, 43, 43, 43,
        43, 25, 43, 43, 43, 43, 43, 43, 43, 43, 43, 25, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 25, 43,
        25, 43, 43, 43, 43, 25, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
        43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    ],
    dtype=np.uint8,
)

IQ2_XXS_DTYPE = np.dtype([("d", "<f2"), ("qs", "<u2", (32,))], align=False)
Q2_K_DTYPE = np.dtype(
    [("scales", "u1", (16,)), ("qs", "u1", (64,)), ("d", "<f2"), ("dmin", "<f2")],
    align=False,
)
assert IQ2_XXS_DTYPE.itemsize == 66
assert Q2_K_DTYPE.itemsize == 84


@dataclass
class FamilyPlan:
    layer: int
    family: str
    quant_type: str
    entry: dict[str, Any]
    weight_offset: int
    weight_bytes: int
    input_channels: int
    output_channels: int
    blocks_per_row: int
    scale_offset: int = 0

    @property
    def scale_count(self) -> int:
        return self.output_channels

    @property
    def scale_bytes(self) -> int:
        return self.scale_count * 2


@dataclass
class LayerPlan:
    layer: int
    file_name: str
    source_path: Path
    source_stride: int
    source_expert_count: int
    export_expert_count: int
    families: list[FamilyPlan]
    new_stride: int
    scale_region_offset: int
    scale_region_bytes: int
    alignment_padding_bytes: int


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def parse_layer_spec(spec: str | None) -> set[int] | None:
    if spec is None or not spec.strip():
        return None
    result: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid layer range: {item!r}")
            result.update(range(start, end + 1))
        else:
            layer = int(item)
            if layer < 0:
                raise ValueError(f"invalid layer: {item!r}")
            result.add(layer)
    if not result:
        raise ValueError("--layers selected no layers")
    return result


def read_exact(handle: BinaryIO, nbytes: int, label: str) -> bytes:
    output = bytearray(nbytes)
    view = memoryview(output)
    offset = 0
    while offset < nbytes:
        count = handle.readinto(view[offset:])
        if not count:
            raise EOFError(f"{label}: expected {nbytes} bytes, read {offset}")
        offset += count
    return bytes(output)


def write_exact(handle: BinaryIO, data: bytes, label: str) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        count = handle.write(view[offset:])
        if not count:
            raise OSError(f"{label}: short write after {offset} of {len(view)} bytes")
        offset += count


def scale_bytes_from_absmax(absmax: np.ndarray) -> bytes:
    absmax = np.asarray(absmax, dtype=np.float32)
    if absmax.ndim != 1:
        raise ValueError("absmax must be one-dimensional")
    if not np.all(np.isfinite(absmax)) or np.any(absmax < 0):
        raise ValueError("dequantized row absmax contains a non-finite or negative value")
    scales = np.ones(absmax.shape, dtype=np.float32)
    nonzero = absmax > np.float32(0.0)
    scales[nonzero] = absmax[nonzero] / np.float32(127.0)
    scales_f16 = scales.astype("<f2")
    if not np.all(np.isfinite(scales_f16)) or np.any(scales_f16 <= np.float16(0.0)):
        raise ValueError("fp16 ANE scale is non-finite or underflowed to zero")
    return scales_f16.tobytes(order="C")


def iq2_xxs_row_absmax(raw: bytes | memoryview, input_channels: int, output_channels: int) -> np.ndarray:
    if input_channels % QK_K:
        raise ValueError(f"IQ2_XXS input dimension {input_channels} is not divisible by {QK_K}")
    blocks_per_row = input_channels // QK_K
    expected = output_channels * blocks_per_row * IQ2_XXS_DTYPE.itemsize
    if len(raw) != expected:
        raise ValueError(f"IQ2_XXS payload has {len(raw)} bytes; expected {expected}")

    blocks = np.frombuffer(raw, dtype=IQ2_XXS_DTYPE).reshape(output_channels, blocks_per_row)
    d = np.abs(blocks["d"].astype(np.float32))
    if not np.all(np.isfinite(d)):
        raise ValueError("IQ2_XXS block scale contains a non-finite value")

    # Each 32-value sub-block is encoded by four grid bytes followed by a
    # 28-bit sign field and a high-nibble multiplier.  Signs can be skipped
    # for absmax, while every represented grid magnitude is still considered.
    qs_bytes = blocks["qs"].view(np.uint8).reshape(output_channels, blocks_per_row, 8, 8)
    grid_ids = qs_bytes[..., :4]
    grid_max = IQ2_XXS_GRID_ABSMAX[grid_ids].max(axis=-1).astype(np.float32)
    multipliers = (qs_bytes[..., 7] >> np.uint8(4)).astype(np.float32)
    db = np.multiply(d[..., None], np.float32(0.5) + multipliers, dtype=np.float32)
    db = np.multiply(db, np.float32(0.25), dtype=np.float32)
    represented_max = np.multiply(db, grid_max, dtype=np.float32)
    return represented_max.max(axis=(1, 2))


def q2_k_row_absmax(raw: bytes | memoryview, input_channels: int, output_channels: int) -> np.ndarray:
    if input_channels % QK_K:
        raise ValueError(f"Q2_K input dimension {input_channels} is not divisible by {QK_K}")
    blocks_per_row = input_channels // QK_K
    expected = output_channels * blocks_per_row * Q2_K_DTYPE.itemsize
    if len(raw) != expected:
        raise ValueError(f"Q2_K payload has {len(raw)} bytes; expected {expected}")

    blocks = np.frombuffer(raw, dtype=Q2_K_DTYPE).reshape(output_channels, blocks_per_row)
    d = blocks["d"].astype(np.float32)
    dmin = blocks["dmin"].astype(np.float32)
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(dmin)):
        raise ValueError("Q2_K block scale contains a non-finite value")
    scales = blocks["scales"]
    qs = blocks["qs"]
    row_max = np.zeros(output_channels, dtype=np.float32)

    # A Q2_K group has 16 represented values.  Because dequantization is
    # affine in the 2-bit code, its maximum absolute value occurs at the
    # smallest or largest code actually present in that group.  Computing
    # those two endpoints is exact and avoids expanding all 256 floats.
    for group in range(16):
        half = group // 8
        parity = group & 1
        shift = 2 * ((group // 2) & 3)
        start = half * 32 + parity * 16
        codes = np.bitwise_and(np.right_shift(qs[..., start : start + 16], shift), np.uint8(3))
        code_min = codes.min(axis=-1).astype(np.float32)
        code_max = codes.max(axis=-1).astype(np.float32)
        packed_scale = scales[..., group]
        dl = np.multiply(d, (packed_scale & np.uint8(0x0F)).astype(np.float32), dtype=np.float32)
        ml = np.multiply(dmin, (packed_scale >> np.uint8(4)).astype(np.float32), dtype=np.float32)
        value_min = np.subtract(np.multiply(dl, code_min, dtype=np.float32), ml, dtype=np.float32)
        value_max = np.subtract(np.multiply(dl, code_max, dtype=np.float32), ml, dtype=np.float32)
        group_max = np.maximum(np.abs(value_min), np.abs(value_max)).max(axis=1)
        row_max = np.maximum(row_max, group_max)
    return row_max


def family_scale_bytes(raw: bytes | memoryview, family: FamilyPlan) -> bytes:
    if family.quant_type == "IQ2_XXS":
        absmax = iq2_xxs_row_absmax(raw, family.input_channels, family.output_channels)
    elif family.quant_type == "Q2_K":
        absmax = q2_k_row_absmax(raw, family.input_channels, family.output_channels)
    else:
        raise ValueError(f"unsupported quant type: {family.quant_type}")
    result = scale_bytes_from_absmax(absmax)
    if len(result) != family.scale_bytes:
        raise AssertionError("generated scale payload has the wrong byte length")
    return result


def load_source_manifest(source_dir: Path) -> tuple[dict[str, Any], bytes, int]:
    manifest_path = source_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"{manifest_path}: expected schema_version 1")
    if manifest.get("layout") != LAYOUT:
        raise ValueError(f"{manifest_path}: expected layout {LAYOUT!r}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path}: missing entries")
    model = manifest.get("model") or {}
    expert_count = int(model.get("expert_count", 0))
    if expert_count <= 0:
        raise ValueError(f"{manifest_path}: invalid model.expert_count")
    return manifest, manifest_bytes, expert_count


def build_layer_plans(
    source_dir: Path,
    manifest: dict[str, Any],
    source_expert_count: int,
    selected_layers: set[int] | None,
    export_expert_count: int,
) -> list[LayerPlan]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["entries"]:
        layer = int(entry.get("layer", -1))
        if selected_layers is not None and layer not in selected_layers:
            continue
        by_layer.setdefault(layer, []).append(entry)

    available_layers = {int(entry.get("layer", -1)) for entry in manifest["entries"]}
    if selected_layers is not None:
        missing = selected_layers - available_layers
        if missing:
            raise ValueError(f"requested layers are absent from source manifest: {sorted(missing)}")
    if not by_layer:
        raise ValueError("no source layers selected")

    plans: list[LayerPlan] = []
    seen_file_names: set[str] = set()
    for layer, entries in sorted(by_layer.items()):
        if len(entries) != len(FAMILY_ORDER):
            raise ValueError(
                f"layer {layer}: expected exactly {len(FAMILY_ORDER)} entries, got {len(entries)}"
            )
        by_family = {str(entry.get("tensor_family")): entry for entry in entries}
        if len(by_family) != len(entries) or set(by_family) != set(FAMILY_ORDER):
            raise ValueError(
                f"layer {layer}: expected exactly {list(FAMILY_ORDER)}, got {sorted(by_family)}"
            )
        file_names = {str(entry.get("repacked_file", "")) for entry in entries}
        strides = {int(entry.get("expert_stride", 0)) for entry in entries}
        if len(file_names) != 1 or len(strides) != 1:
            raise ValueError(f"layer {layer}: entries do not share one file and expert stride")
        file_name = file_names.pop()
        if not file_name or Path(file_name).name != file_name:
            raise ValueError(f"layer {layer}: repacked_file must be a plain file name")
        if file_name in seen_file_names:
            raise ValueError(f"layer {layer}: repacked_file {file_name!r} is reused by another layer")
        seen_file_names.add(file_name)
        source_stride = strides.pop()
        if source_stride <= 0:
            raise ValueError(f"layer {layer}: invalid expert_stride")

        source_path = (source_dir / file_name).resolve()
        if source_dir != source_path.parent:
            raise ValueError(f"layer {layer}: repacked_file escapes the source directory")
        expected_file_size = source_stride * source_expert_count
        actual_file_size = source_path.stat().st_size
        if actual_file_size != expected_file_size:
            raise ValueError(
                f"{source_path}: size {actual_file_size} != {source_expert_count} * {source_stride}"
            )

        families: list[FamilyPlan] = []
        weight_ranges: list[tuple[int, int, str]] = []
        for family_name in FAMILY_ORDER:
            entry = by_family[family_name]
            quant_type = str(entry.get("quant_type"))
            if quant_type != EXPECTED_QUANT[family_name]:
                raise ValueError(
                    f"layer {layer} {family_name}: expected {EXPECTED_QUANT[family_name]}, got {quant_type}"
                )
            if not bool(entry.get("expert_major", False)):
                raise ValueError(f"layer {layer} {family_name}: expert_major must be true")
            shape = entry.get("shape")
            if not isinstance(shape, list) or len(shape) != 3:
                raise ValueError(f"layer {layer} {family_name}: expected a three-dimensional shape")
            input_channels, output_channels, entry_experts = map(int, shape)
            if entry_experts != source_expert_count:
                raise ValueError(
                    f"layer {layer} {family_name}: shape expert count {entry_experts} != {source_expert_count}"
                )
            if input_channels <= 0 or input_channels % QK_K or output_channels <= 0:
                raise ValueError(f"layer {layer} {family_name}: invalid shape {shape}")
            blocks_per_row = input_channels // QK_K
            expected_weight_bytes = output_channels * blocks_per_row * BLOCK_BYTES[quant_type]
            weight_bytes = int(entry.get("bytes_per_expert", 0))
            if weight_bytes != expected_weight_bytes:
                raise ValueError(
                    f"layer {layer} {family_name}: bytes_per_expert {weight_bytes} != {expected_weight_bytes}"
                )
            exact_byte_length = int(entry.get("exact_byte_length", 0))
            if exact_byte_length != weight_bytes * source_expert_count:
                raise ValueError(
                    f"layer {layer} {family_name}: exact_byte_length does not match all experts"
                )
            weight_offset = int(entry.get("repacked_offset", -1))
            if weight_offset < 0 or weight_offset + weight_bytes > source_stride:
                raise ValueError(f"layer {layer} {family_name}: weight region is outside the record")
            weight_ranges.append((weight_offset, weight_offset + weight_bytes, family_name))
            families.append(
                FamilyPlan(
                    layer=layer,
                    family=family_name,
                    quant_type=quant_type,
                    entry=entry,
                    weight_offset=weight_offset,
                    weight_bytes=weight_bytes,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    blocks_per_row=blocks_per_row,
                )
            )

        weight_ranges.sort()
        for previous, current in zip(weight_ranges, weight_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"layer {layer}: overlapping {previous[2]} and {current[2]} regions")

        cursor = source_stride
        for family in families:
            family.scale_offset = cursor
            cursor += family.scale_bytes
        new_stride = align_up(cursor, RECORD_ALIGNMENT)
        plans.append(
            LayerPlan(
                layer=layer,
                file_name=file_name,
                source_path=source_path,
                source_stride=source_stride,
                source_expert_count=source_expert_count,
                export_expert_count=export_expert_count,
                families=families,
                new_stride=new_stride,
                scale_region_offset=source_stride,
                scale_region_bytes=cursor - source_stride,
                alignment_padding_bytes=new_stride - cursor,
            )
        )
    return plans


@contextmanager
def atomic_output(path: Path, mode: str) -> Iterator[Any]:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    try:
        kwargs = {} if "b" in mode else {"encoding": "utf-8"}
        exclusive_mode = "xb" if "b" in mode else "x"
        with tmp.open(exclusive_mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes atomically and, unlike rename,
        # fails if another process creates the target after the initial check.
        os.link(tmp, path)
        tmp.unlink()
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def export_layer(
    plan: LayerPlan,
    destination: Path,
    progress_every: int,
) -> dict[str, Any]:
    output_path = destination / plan.file_name
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    zero_padding = bytes(plan.alignment_padding_bytes)
    started = time.monotonic()
    expected_size = plan.export_expert_count * plan.new_stride

    with plan.source_path.open("rb") as source, atomic_output(output_path, "wb") as output:
        for expert in range(plan.export_expert_count):
            record = read_exact(source, plan.source_stride, f"layer {plan.layer} expert {expert}")
            source_hash.update(record)
            write_exact(output, record, f"layer {plan.layer} expert {expert} weights")
            output_hash.update(record)
            for family in plan.families:
                payload = memoryview(record)[
                    family.weight_offset : family.weight_offset + family.weight_bytes
                ]
                scales = family_scale_bytes(payload, family)
                write_exact(
                    output,
                    scales,
                    f"layer {plan.layer} expert {expert} {family.family} scales",
                )
                output_hash.update(scales)
            if zero_padding:
                write_exact(output, zero_padding, f"layer {plan.layer} expert {expert} padding")
                output_hash.update(zero_padding)
            if progress_every > 0 and (expert + 1) % progress_every == 0:
                print(
                    f"  layer {plan.layer}: experts {expert + 1}/{plan.export_expert_count}",
                    flush=True,
                )
        if output.tell() != expected_size:
            raise OSError(
                f"{output_path}: temporary output has {output.tell()} bytes; expected {expected_size}"
            )

    actual_size = output_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{output_path}: wrote {actual_size} bytes; expected {expected_size}")
    shutil.copymode(plan.source_path, output_path)
    elapsed = time.monotonic() - started
    gib = actual_size / (1024**3)
    print(
        f"  layer {plan.layer}: wrote {actual_size} bytes in {elapsed:.1f}s "
        f"({gib / max(elapsed, 1e-9):.2f} GiB/s)",
        flush=True,
    )
    return {
        "layer": plan.layer,
        "file": plan.file_name,
        "expert_count": plan.export_expert_count,
        "source_expert_stride": plan.source_stride,
        "expert_stride": plan.new_stride,
        "scale_region_offset": plan.scale_region_offset,
        "scale_region_bytes": plan.scale_region_bytes,
        "alignment_padding_bytes": plan.alignment_padding_bytes,
        "source_file_byte_length": plan.source_path.stat().st_size,
        "source_exported_records_sha256": source_hash.hexdigest(),
        "byte_length": actual_size,
        "sha256": output_hash.hexdigest(),
    }


def annotate_manifest(
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    source_manifest_sha256: str,
    plans: list[LayerPlan],
    file_metadata: list[dict[str, Any]],
    dense_mode: str,
) -> dict[str, Any]:
    result = copy.deepcopy(source_manifest)
    plan_by_key = {
        (plan.layer, family.family): (plan, family)
        for plan in plans
        for family in plan.families
    }
    selected_layers = [plan.layer for plan in plans]
    source_expert_count = plans[0].source_expert_count
    export_expert_count = plans[0].export_expert_count
    source_layers = sorted({int(entry["layer"]) for entry in source_manifest["entries"]})
    partial = selected_layers != source_layers or export_expert_count != source_expert_count

    entries: list[dict[str, Any]] = []
    for source_entry in source_manifest["entries"]:
        key = (int(source_entry["layer"]), str(source_entry["tensor_family"]))
        pair = plan_by_key.get(key)
        if pair is None:
            continue
        plan, family = pair
        entry = copy.deepcopy(source_entry)
        entry["source_shape"] = copy.deepcopy(entry["shape"])
        entry["source_exact_byte_length"] = int(entry["exact_byte_length"])
        entry["source_expert_stride"] = plan.source_stride
        entry["shape"][2] = export_expert_count
        entry["exact_byte_length"] = family.weight_bytes * export_expert_count
        entry["expert_stride"] = plan.new_stride
        entry.update(
            {
                "ane_i8_scale_offset": family.scale_offset,
                "ane_i8_scale_bytes": family.scale_bytes,
                "ane_i8_scale_count": family.scale_count,
                "ane_i8_scale_dtype": "F16",
                "ane_i8_scale_semantics": "dequant_multiplier",
                "ane_i8_scale_axis": 1,
                "ane_i8_scale_group_size": 1,
            }
        )
        entries.append(entry)

    source_info = copy.deepcopy(result.get("source") or {})
    source_info.update(
        {
            "sidecar_manifest": str(source_manifest_path),
            "sidecar_manifest_sha256": source_manifest_sha256,
            "sidecar_schema_version": int(source_manifest.get("schema_version", 1)),
        }
    )
    model = copy.deepcopy(result.get("model") or {})
    model["source_expert_count"] = source_expert_count
    model["expert_count"] = export_expert_count
    model["sidecar_layout"] = "expert-major"

    result.update(
        {
            "schema_version": 2,
            "sidecar_kind": "flashmoe_gguf",
            "layout": LAYOUT,
            "storage_layout": STORAGE_LAYOUT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "converter": Path(__file__).name,
            "source": source_info,
            "model": model,
            "filters": {
                "layers": selected_layers if selected_layers != source_layers else None,
                "families": None,
            },
            "export_scope": {
                "partial": partial,
                "layers": selected_layers,
                "source_layer_count": len(source_layers),
                "source_expert_count": source_expert_count,
                "expert_count": export_expert_count,
                "runtime_loadable": not partial,
            },
            # This means the routed sidecar has the full source layer/expert
            # scope. Dense-package independence is reported separately.
            "runtime_loadable": not partial,
            "standalone_model_package": not partial and dense_mode in ("copy", "symlink"),
            "ane_i8_scale_scheme": {
                "schema_version": SCALE_SCHEMA_VERSION,
                "storage": "appended_to_expert_record",
                "family_order": list(FAMILY_ORDER),
                "dtype": "F16",
                "endian": "little",
                "semantics": "dequant_multiplier",
                "axis": 1,
                "axis_name": "output_channel",
                "group_size": 1,
                "quantization": {
                    "method": "symmetric_absmax",
                    "qmin": -127,
                    "qmax": 127,
                    "zero_point": 0,
                    "rounding": "nearest_even",
                    "zero_row_scale": 1.0,
                    "source_values": "exact_dequantized_sidecar_weights",
                },
                "record_alignment": RECORD_ALIGNMENT,
            },
            "dense_mode": dense_mode,
            "layer_files": file_metadata,
            "entries": entries,
        }
    )
    return result


def copy_dense(source_dir: Path, destination: Path, mode: str) -> str:
    source_dense = source_dir / "dense"
    if mode == "none":
        return "none"
    if not source_dense.is_dir():
        raise FileNotFoundError(f"--dense-mode {mode} requested, but {source_dense} is absent")
    destination_dense = destination / "dense"
    if destination_dense.exists() or destination_dense.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination_dense}")
    if mode == "symlink":
        os.symlink(source_dense, destination_dense, target_is_directory=True)
        return "symlink"

    # copytree creates the destination directory exclusively. A failed copy
    # can leave it partial, but cannot replace an existing path; the package
    # manifest is written only after the dense copy succeeds.
    shutil.copytree(source_dense, destination_dense, copy_function=shutil.copy2)
    return "copy"


def validate_v2_package(source_dir: Path, destination: Path) -> None:
    source_manifest, source_manifest_bytes, source_expert_count = load_source_manifest(source_dir)
    output_manifest_path = destination / "manifest.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    if int(output_manifest.get("schema_version", 0)) != 2:
        raise ValueError(f"{output_manifest_path}: expected schema_version 2")
    if output_manifest.get("sidecar_kind") != "flashmoe_gguf":
        raise ValueError(f"{output_manifest_path}: unexpected sidecar_kind")
    if output_manifest.get("layout") != LAYOUT:
        raise ValueError(f"{output_manifest_path}: unexpected layout")
    if output_manifest.get("storage_layout") != STORAGE_LAYOUT:
        raise ValueError(f"{output_manifest_path}: unexpected storage_layout")
    scheme = output_manifest.get("ane_i8_scale_scheme") or {}
    expected_scheme_fields = {
        "schema_version": SCALE_SCHEMA_VERSION,
        "storage": "appended_to_expert_record",
        "family_order": list(FAMILY_ORDER),
        "dtype": "F16",
        "endian": "little",
        "semantics": "dequant_multiplier",
        "axis": 1,
        "axis_name": "output_channel",
        "group_size": 1,
        "record_alignment": RECORD_ALIGNMENT,
    }
    for key, expected in expected_scheme_fields.items():
        if scheme.get(key) != expected:
            raise ValueError(f"{output_manifest_path}: ane_i8_scale_scheme.{key} mismatch")
    source_info = output_manifest.get("source") or {}
    expected_manifest_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    if source_info.get("sidecar_manifest_sha256") != expected_manifest_hash:
        raise ValueError("output manifest does not match the supplied source manifest")
    scope = output_manifest.get("export_scope") or {}
    selected_layers = {int(value) for value in scope.get("layers", [])}
    export_expert_count = int(scope.get("expert_count", 0))
    if int(scope.get("source_expert_count", 0)) != source_expert_count:
        raise ValueError("output source_expert_count does not match source")
    if not selected_layers or export_expert_count <= 0 or export_expert_count > source_expert_count:
        raise ValueError("output export_scope is invalid")
    expected_runtime_loadable = (
        len(selected_layers)
        == len({int(entry["layer"]) for entry in source_manifest["entries"]})
        and export_expert_count == source_expert_count
    )
    if output_manifest.get("runtime_loadable") is not expected_runtime_loadable:
        raise ValueError("output runtime_loadable does not match export scope")
    if scope.get("runtime_loadable") is not expected_runtime_loadable:
        raise ValueError("output export_scope.runtime_loadable does not match export scope")
    dense_mode = output_manifest.get("dense_mode")
    if dense_mode not in ("copy", "symlink", "none"):
        raise ValueError("output dense_mode is invalid")
    expected_standalone = expected_runtime_loadable and dense_mode in ("copy", "symlink")
    if output_manifest.get("standalone_model_package") is not expected_standalone:
        raise ValueError("output standalone_model_package does not match dense_mode")
    if dense_mode != "none" and not (destination / "dense").is_dir():
        raise ValueError("output dense package is missing")
    plans = build_layer_plans(
        source_dir,
        source_manifest,
        source_expert_count,
        selected_layers,
        export_expert_count,
    )
    output_entries = output_manifest.get("entries", [])
    output_layer_files = output_manifest.get("layer_files", [])
    entry_by_key = {
        (int(entry["layer"]), str(entry["tensor_family"])): entry
        for entry in output_entries
    }
    file_by_layer = {
        int(item["layer"]): item for item in output_layer_files
    }
    expected_entry_keys = {
        (plan.layer, family.family) for plan in plans for family in plan.families
    }
    if len(output_entries) != len(expected_entry_keys) or set(entry_by_key) != expected_entry_keys:
        raise ValueError("output manifest entry set does not match export scope")
    if len(output_layer_files) != len(selected_layers) or set(file_by_layer) != selected_layers:
        raise ValueError("output manifest layer_files set does not match export scope")

    for plan in plans:
        file_meta = file_by_layer.get(plan.layer)
        if file_meta is None:
            raise ValueError(f"output manifest has no layer_files record for layer {plan.layer}")
        if int(file_meta.get("expert_stride", 0)) != plan.new_stride:
            raise ValueError(f"layer {plan.layer}: manifest expert_stride mismatch")
        expected_file_fields = {
            "file": plan.file_name,
            "expert_count": plan.export_expert_count,
            "source_expert_stride": plan.source_stride,
            "scale_region_offset": plan.scale_region_offset,
            "scale_region_bytes": plan.scale_region_bytes,
            "alignment_padding_bytes": plan.alignment_padding_bytes,
            "source_file_byte_length": plan.source_path.stat().st_size,
            "byte_length": plan.export_expert_count * plan.new_stride,
        }
        for key, expected in expected_file_fields.items():
            if file_meta.get(key) != expected:
                raise ValueError(f"layer {plan.layer}: layer_files.{key} mismatch")
        for family in plan.families:
            output_entry = entry_by_key.get((plan.layer, family.family))
            if output_entry is None:
                raise ValueError(f"layer {plan.layer}: missing output entry for {family.family}")
            expected_entry_fields = {
                "quant_type": family.quant_type,
                "shape": [
                    family.input_channels,
                    family.output_channels,
                    plan.export_expert_count,
                ],
                "bytes_per_expert": family.weight_bytes,
                "exact_byte_length": family.weight_bytes * plan.export_expert_count,
                "repacked_file": plan.file_name,
                "repacked_offset": family.weight_offset,
                "expert_stride": plan.new_stride,
                "expert_major": True,
            }
            for key, expected in expected_entry_fields.items():
                if output_entry.get(key) != expected:
                    raise ValueError(
                        f"layer {plan.layer} {family.family}: {key} mismatch"
                    )
            expected_fields = {
                "ane_i8_scale_offset": family.scale_offset,
                "ane_i8_scale_bytes": family.scale_bytes,
                "ane_i8_scale_count": family.scale_count,
                "ane_i8_scale_dtype": "F16",
                "ane_i8_scale_semantics": "dequant_multiplier",
                "ane_i8_scale_axis": 1,
                "ane_i8_scale_group_size": 1,
            }
            for key, expected in expected_fields.items():
                if output_entry.get(key) != expected:
                    raise ValueError(
                        f"layer {plan.layer} {family.family}: {key} mismatch"
                    )

        output_path = destination / plan.file_name
        expected_size = plan.export_expert_count * plan.new_stride
        if output_path.stat().st_size != expected_size:
            raise ValueError(f"{output_path}: unexpected byte length")
        source_hash = hashlib.sha256()
        output_hash = hashlib.sha256()
        with plan.source_path.open("rb", buffering=0) as source, output_path.open("rb", buffering=0) as output:
            for expert in range(plan.export_expert_count):
                source_record = read_exact(
                    source, plan.source_stride, f"source layer {plan.layer} expert {expert}"
                )
                output_record = read_exact(
                    output, plan.new_stride, f"output layer {plan.layer} expert {expert}"
                )
                source_hash.update(source_record)
                output_hash.update(output_record)
                if output_record[: plan.source_stride] != source_record:
                    raise ValueError(f"layer {plan.layer} expert {expert}: weight bytes changed")
                for family in plan.families:
                    payload = memoryview(source_record)[
                        family.weight_offset : family.weight_offset + family.weight_bytes
                    ]
                    expected_scales = family_scale_bytes(payload, family)
                    actual_scales = output_record[
                        family.scale_offset : family.scale_offset + family.scale_bytes
                    ]
                    if actual_scales != expected_scales:
                        raise ValueError(
                            f"layer {plan.layer} expert {expert} {family.family}: scale mismatch"
                        )
                if plan.alignment_padding_bytes:
                    if any(output_record[-plan.alignment_padding_bytes :]):
                        raise ValueError(f"layer {plan.layer} expert {expert}: non-zero record padding")
            if output.read(1):
                raise ValueError(f"{output_path}: trailing bytes")
        if file_meta.get("source_exported_records_sha256") != source_hash.hexdigest():
            raise ValueError(f"layer {plan.layer}: source record checksum mismatch")
        if file_meta.get("sha256") != output_hash.hexdigest():
            raise ValueError(f"layer {plan.layer}: output checksum mismatch")
        print(
            f"validated layer {plan.layer}: {plan.export_expert_count} experts, weights unchanged, scales exact",
            flush=True,
        )
    print(f"validated v2 package: {destination}")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ensure_safe_destination(source_dir: Path, destination: Path) -> None:
    if destination == source_dir or source_dir in destination.parents:
        raise ValueError("destination must not be the source directory or a directory inside it")


def print_dry_run(
    source_dir: Path,
    destination: Path,
    plans: list[LayerPlan],
    dense_mode: str,
) -> None:
    source_bytes = sum(plan.export_expert_count * plan.source_stride for plan in plans)
    output_bytes = sum(plan.export_expert_count * plan.new_stride for plan in plans)
    scale_bytes = sum(plan.export_expert_count * plan.scale_region_bytes for plan in plans)
    dense_bytes = directory_size(source_dir / "dense") if dense_mode == "copy" else 0
    print(f"source:      {source_dir}")
    print(f"destination: {destination}")
    print(f"layers:      {','.join(str(plan.layer) for plan in plans)}")
    print(f"experts:     {plans[0].export_expert_count}/{plans[0].source_expert_count}")
    print(f"weight bytes to copy: {source_bytes}")
    print(f"scale bytes to append: {scale_bytes}")
    print(f"expert files output:   {output_bytes}")
    print(f"dense bytes to copy:    {dense_bytes}")
    for plan in plans:
        detail = ", ".join(
            f"{family.family}={family.scale_count}xF16@{family.scale_offset}"
            for family in plan.families
        )
        print(
            f"  layer {plan.layer}: stride {plan.source_stride} -> {plan.new_stride}; {detail}"
        )
    print("dry-run: source validated; no output written")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a new v2 expert-major sidecar with appended ANE INT8 per-output scales."
    )
    parser.add_argument("source", type=Path, help="read-only v1 expert-major sidecar directory")
    parser.add_argument("destination", type=Path, help="new v2 sidecar directory")
    parser.add_argument("--layers", help="optional layer selection such as 0 or 0,2-4")
    parser.add_argument(
        "--expert-limit",
        type=int,
        help="write only the first N experts (creates a partial smoke-test package)",
    )
    parser.add_argument(
        "--dense-mode",
        choices=("copy", "symlink", "none"),
        default="copy",
        help="how to populate the destination dense directory (default: copy)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="validate and print layout without writing")
    modes.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an already-generated destination against the source without writing",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="re-read the generated package and verify weights, scales, padding, and checksums",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=16,
        help="report every N experts; use 0 to suppress intermediate progress",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    source_dir = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    ensure_safe_destination(source_dir, destination)

    if args.validate_only:
        if args.layers is not None or args.expert_limit is not None:
            raise ValueError("--validate-only reads scope from the v2 manifest; omit selection flags")
        if not destination.is_dir():
            raise FileNotFoundError(destination)
        validate_v2_package(source_dir, destination)
        return 0

    source_manifest, source_manifest_bytes, source_expert_count = load_source_manifest(source_dir)
    selected_layers = parse_layer_spec(args.layers)
    export_expert_count = source_expert_count if args.expert_limit is None else args.expert_limit
    if export_expert_count <= 0 or export_expert_count > source_expert_count:
        raise ValueError(
            f"--expert-limit must be between 1 and source expert count {source_expert_count}"
        )
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    plans = build_layer_plans(
        source_dir,
        source_manifest,
        source_expert_count,
        selected_layers,
        export_expert_count,
    )
    if args.dry_run:
        print_dry_run(source_dir, destination, plans, args.dense_mode)
        return 0

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} exists and is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"source:      {source_dir}")
    print(f"destination: {destination}")
    print(f"exporting {len(plans)} layer(s), {export_expert_count} expert(s) per layer")

    file_metadata: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, 1):
        print(f"[{index}/{len(plans)}] {plan.file_name}", flush=True)
        file_metadata.append(export_layer(plan, destination, args.progress_every))

    actual_dense_mode = copy_dense(source_dir, destination, args.dense_mode)
    output_manifest = annotate_manifest(
        source_manifest,
        source_dir / "manifest.json",
        hashlib.sha256(source_manifest_bytes).hexdigest(),
        plans,
        file_metadata,
        actual_dense_mode,
    )
    with atomic_output(destination / "manifest.json", "w") as handle:
        json.dump(output_manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"wrote {destination / 'manifest.json'}")
    if args.validate:
        validate_v2_package(source_dir, destination)
    return 0


def main(argv: list[str]) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (EOFError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
