#!/usr/bin/env python3
"""Small-file coverage for export_flash_moe_ane_i8_v2.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_flash_moe_ane_i8_v2.py"
SPEC = importlib.util.spec_from_file_location("export_flash_moe_ane_i8_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def canonical_iq2_grid() -> np.ndarray:
    """Read the runtime's full IQ2_XXS grid for an independent reference."""
    source = (ROOT / "ds4.c").read_text(encoding="utf-8")
    match = re.search(
        r"static const uint64_t iq2xxs_grid\[256\] = \{(.*?)\n\};",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("could not locate iq2xxs_grid in ds4.c")
    words = [int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    if len(words) != 256:
        raise AssertionError(f"expected 256 IQ2 grid words, found {len(words)}")
    return np.asarray(
        [list(word.to_bytes(8, byteorder="little")) for word in words],
        dtype=np.float32,
    )


IQ2_GRID = canonical_iq2_grid()


def make_iq2_payload(rows: int, blocks_per_row: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    output = bytearray()
    for row in range(rows):
        for block in range(blocks_per_row):
            d = np.float16(((-1.0) ** (row + block)) * (0.0025 + 0.0003 * (row + block + 1)))
            qs = rng.integers(0, 256, size=64, dtype=np.uint8)
            # Exercise all multiplier nibbles deterministically while keeping
            # arbitrary grid and sign bits in the rest of each sub-block.
            for ib32 in range(8):
                qs[ib32 * 8 + 7] = np.uint8(
                    ((row + block + ib32) % 16) << 4 | (int(qs[ib32 * 8 + 7]) & 0x0F)
                )
            output.extend(np.asarray([d], dtype="<f2").tobytes())
            output.extend(qs.tobytes())
    return bytes(output)


def reference_iq2_row_absmax(raw: bytes, input_channels: int, output_channels: int) -> np.ndarray:
    blocks_per_row = input_channels // 256
    result = np.zeros(output_channels, dtype=np.float32)
    cursor = 0
    for row in range(output_channels):
        row_max = np.float32(0.0)
        for _ in range(blocks_per_row):
            d = np.frombuffer(raw, dtype="<f2", count=1, offset=cursor)[0].astype(np.float32)
            qs = np.frombuffer(raw, dtype=np.uint8, count=64, offset=cursor + 2)
            cursor += 66
            for ib32 in range(8):
                aux = qs[ib32 * 8 : (ib32 + 1) * 8]
                multiplier = np.float32(int(aux[7]) >> 4)
                db = np.multiply(d, np.float32(0.5) + multiplier, dtype=np.float32)
                db = np.multiply(db, np.float32(0.25), dtype=np.float32)
                for grid_index in aux[:4]:
                    values = np.multiply(db, IQ2_GRID[int(grid_index)], dtype=np.float32)
                    row_max = np.maximum(row_max, np.max(np.abs(values)))
        result[row] = row_max
    return result


def make_q2_k_payload(rows: int, blocks_per_row: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    output = bytearray()
    for row in range(rows):
        for block in range(blocks_per_row):
            scales = rng.integers(0, 256, size=16, dtype=np.uint8)
            qs = rng.integers(0, 256, size=64, dtype=np.uint8)
            d = np.float16(((-1.0) ** block) * (0.003 + 0.0002 * row))
            dmin = np.float16(((-1.0) ** row) * (0.001 + 0.0001 * block))
            output.extend(scales.tobytes())
            output.extend(qs.tobytes())
            output.extend(np.asarray([d, dmin], dtype="<f2").tobytes())
    return bytes(output)


def reference_q2_k_row_absmax(raw: bytes, input_channels: int, output_channels: int) -> np.ndarray:
    blocks_per_row = input_channels // 256
    result = np.zeros(output_channels, dtype=np.float32)
    cursor = 0
    for row in range(output_channels):
        row_max = np.float32(0.0)
        for _ in range(blocks_per_row):
            scales = np.frombuffer(raw, dtype=np.uint8, count=16, offset=cursor)
            qs = np.frombuffer(raw, dtype=np.uint8, count=64, offset=cursor + 16)
            d = np.frombuffer(raw, dtype="<f2", count=1, offset=cursor + 80)[0].astype(np.float32)
            dmin = np.frombuffer(raw, dtype="<f2", count=1, offset=cursor + 82)[0].astype(np.float32)
            cursor += 84
            for group in range(16):
                half = group // 8
                parity = group & 1
                shift = 2 * ((group // 2) & 3)
                start = half * 32 + parity * 16
                sc = int(scales[group])
                dl = np.multiply(d, np.float32(sc & 0x0F), dtype=np.float32)
                ml = np.multiply(dmin, np.float32(sc >> 4), dtype=np.float32)
                for packed in qs[start : start + 16]:
                    code = np.float32((int(packed) >> shift) & 3)
                    value = np.subtract(
                        np.multiply(dl, code, dtype=np.float32), ml, dtype=np.float32
                    )
                    row_max = np.maximum(row_max, np.abs(value))
        result[row] = row_max
    return result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(source: Path, experts: int = 2) -> tuple[int, dict[str, bytes]]:
    source.mkdir()
    families_by_expert: list[dict[str, bytes]] = []
    for expert in range(experts):
        families_by_expert.append(
            {
                "ffn_gate_exps": make_iq2_payload(2, 1, 100 + expert),
                "ffn_up_exps": make_iq2_payload(2, 1, 200 + expert),
                "ffn_down_exps": make_q2_k_payload(3, 1, 300 + expert),
            }
        )
    family_sizes = {
        family: len(families_by_expert[0][family]) for family in EXPORTER.FAMILY_ORDER
    }
    stride = sum(family_sizes.values())
    offsets: dict[str, int] = {}
    cursor = 0
    for family in EXPORTER.FAMILY_ORDER:
        offsets[family] = cursor
        cursor += family_sizes[family]

    layer = bytearray()
    for expert_payloads in families_by_expert:
        for family in EXPORTER.FAMILY_ORDER:
            layer.extend(expert_payloads[family])
    (source / "layer_000.bin").write_bytes(layer)

    shapes = {
        "ffn_gate_exps": [256, 2, experts],
        "ffn_up_exps": [256, 2, experts],
        "ffn_down_exps": [256, 3, experts],
    }
    entries = []
    for family in EXPORTER.FAMILY_ORDER:
        entries.append(
            {
                "layer": 0,
                "tensor_family": family,
                "tensor_name": f"blk.0.{family}.weight",
                "quant_type": EXPORTER.EXPECTED_QUANT[family],
                "block_size": 256,
                "shape": shapes[family],
                "bytes_per_expert": family_sizes[family],
                "exact_byte_length": family_sizes[family] * experts,
                "repacked_file": "layer_000.bin",
                "repacked_offset": offsets[family],
                "expert_stride": stride,
                "expert_major": True,
            }
        )
    manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": "layer_major_expert",
        "source": {"model_files": ["synthetic.gguf"]},
        "model": {"arch": "deepseek4", "expert_count": experts, "expert_used_count": 1},
        "entries": entries,
    }
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return stride, families_by_expert[0]


class FlashMoeAneI8V2ExportTest(unittest.TestCase):
    def test_vectorized_absmax_matches_independent_full_dequantization(self) -> None:
        np.testing.assert_array_equal(
            EXPORTER.IQ2_XXS_GRID_ABSMAX,
            IQ2_GRID.max(axis=1).astype(np.uint8),
        )
        iq2 = make_iq2_payload(rows=3, blocks_per_row=2, seed=7)
        expected_iq2 = reference_iq2_row_absmax(iq2, 512, 3)
        actual_iq2 = EXPORTER.iq2_xxs_row_absmax(iq2, 512, 3)
        np.testing.assert_array_equal(actual_iq2, expected_iq2)
        self.assertEqual(
            EXPORTER.scale_bytes_from_absmax(actual_iq2),
            expected_iq2.__truediv__(np.float32(127.0)).astype("<f2").tobytes(),
        )

        q2_k = make_q2_k_payload(rows=4, blocks_per_row=2, seed=9)
        expected_q2_k = reference_q2_k_row_absmax(q2_k, 512, 4)
        actual_q2_k = EXPORTER.q2_k_row_absmax(q2_k, 512, 4)
        np.testing.assert_array_equal(actual_q2_k, expected_q2_k)
        self.assertEqual(
            EXPORTER.scale_bytes_from_absmax(actual_q2_k),
            expected_q2_k.__truediv__(np.float32(127.0)).astype("<f2").tobytes(),
        )
        self.assertEqual(
            EXPORTER.scale_bytes_from_absmax(np.zeros(3, dtype=np.float32)),
            np.ones(3, dtype="<f2").tobytes(),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            EXPORTER.scale_bytes_from_absmax(np.asarray([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "underflowed"):
            EXPORTER.scale_bytes_from_absmax(np.asarray([1.0e-8], dtype=np.float32))

    def test_full_export_preserves_weights_appends_scales_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "v2"
            source_stride, _ = write_fixture(source)
            source_manifest_hash = file_sha256(source / "manifest.json")
            source_layer_hash = file_sha256(source / "layer_000.bin")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(destination),
                    "--dense-mode",
                    "none",
                    "--progress-every",
                    "0",
                    "--validate",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("validated v2 package", result.stdout)
            self.assertEqual(file_sha256(source / "manifest.json"), source_manifest_hash)
            self.assertEqual(file_sha256(source / "layer_000.bin"), source_layer_hash)

            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["storage_layout"], EXPORTER.STORAGE_LAYOUT)
            self.assertFalse(manifest["export_scope"]["partial"])
            self.assertTrue(manifest["export_scope"]["runtime_loadable"])
            self.assertTrue(manifest["runtime_loadable"])
            self.assertFalse(manifest["standalone_model_package"])
            self.assertEqual(manifest["dense_mode"], "none")

            new_stride = manifest["layer_files"][0]["expert_stride"]
            self.assertEqual(source_stride, 516)
            self.assertEqual(new_stride, 576)
            source_bytes = (source / "layer_000.bin").read_bytes()
            output_bytes = (destination / "layer_000.bin").read_bytes()
            self.assertEqual(len(output_bytes), 2 * new_stride)
            for expert in range(2):
                source_record = source_bytes[expert * source_stride : (expert + 1) * source_stride]
                output_record = output_bytes[expert * new_stride : (expert + 1) * new_stride]
                self.assertEqual(output_record[:source_stride], source_record)
                self.assertEqual(output_record[530:], bytes(new_stride - 530))

            by_family = {entry["tensor_family"]: entry for entry in manifest["entries"]}
            self.assertEqual(by_family["ffn_gate_exps"]["ane_i8_scale_offset"], 516)
            self.assertEqual(by_family["ffn_up_exps"]["ane_i8_scale_offset"], 520)
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_offset"], 524)
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_count"], 3)
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_bytes"], 6)
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_dtype"], "F16")
            self.assertEqual(
                by_family["ffn_down_exps"]["ane_i8_scale_semantics"],
                "dequant_multiplier",
            )
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_axis"], 1)
            self.assertEqual(by_family["ffn_down_exps"]["ane_i8_scale_group_size"], 1)
            self.assertEqual(manifest["ane_i8_scale_scheme"]["axis"], 1)

    def test_dry_run_and_partial_smoke_write_only_the_requested_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            dry_destination = root / "dry"
            partial_destination = root / "partial"
            write_fixture(source)

            dry = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(dry_destination),
                    "--layers",
                    "0",
                    "--expert-limit",
                    "1",
                    "--dense-mode",
                    "none",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("dry-run: source validated; no output written", dry.stdout)
            self.assertFalse(dry_destination.exists())

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(partial_destination),
                    "--layers",
                    "0",
                    "--expert-limit",
                    "1",
                    "--dense-mode",
                    "none",
                    "--progress-every",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((partial_destination / "manifest.json").read_text())
            self.assertTrue(manifest["export_scope"]["partial"])
            self.assertFalse(manifest["export_scope"]["runtime_loadable"])
            self.assertFalse(manifest["runtime_loadable"])
            self.assertFalse(manifest["standalone_model_package"])
            self.assertEqual(manifest["model"]["expert_count"], 1)
            self.assertTrue(all(entry["shape"][2] == 1 for entry in manifest["entries"]))
            self.assertEqual((partial_destination / "layer_000.bin").stat().st_size, 576)

    def test_validate_only_detects_changed_weight_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "v2"
            write_fixture(source)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(destination),
                    "--dense-mode",
                    "none",
                    "--progress-every",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output = destination / "layer_000.bin"
            with output.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 0x01]))

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(destination),
                    "--validate-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("weight bytes changed", result.stderr)

    def test_validate_only_detects_changed_scale_and_manifest_offset(self) -> None:
        for corruption in ("scale", "manifest"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                destination = root / "v2"
                write_fixture(source)
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        str(source),
                        str(destination),
                        "--dense-mode",
                        "none",
                        "--progress-every",
                        "0",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                manifest_path = destination / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if corruption == "scale":
                    scale_offset = manifest["entries"][0]["ane_i8_scale_offset"]
                    with (destination / "layer_000.bin").open("r+b") as handle:
                        handle.seek(scale_offset)
                        original = handle.read(1)
                        handle.seek(scale_offset)
                        handle.write(bytes([original[0] ^ 0x01]))
                    expected_error = "scale mismatch"
                else:
                    manifest["entries"][0]["repacked_offset"] += 1
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    expected_error = "repacked_offset mismatch"
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), str(source), str(destination), "--validate-only"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_rejects_zero_expert_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "v2"
            write_fixture(source)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(destination),
                    "--expert-limit",
                    "0",
                    "--dense-mode",
                    "none",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--expert-limit must be between 1", result.stderr)
            self.assertFalse(destination.exists())

    def test_atomic_output_does_not_replace_a_racing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "layer.bin"
            with self.assertRaises(FileExistsError):
                with EXPORTER.atomic_output(target, "wb") as handle:
                    handle.write(b"generated")
                    target.write_bytes(b"racing writer")
            self.assertEqual(target.read_bytes(), b"racing writer")

    def test_source_manifest_rejects_duplicate_family_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            write_fixture(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["entries"].append(dict(manifest["entries"][0]))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(Path(temporary) / "v2"),
                    "--dense-mode",
                    "none",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly 3 entries, got 4", result.stderr)


if __name__ == "__main__":
    unittest.main()
