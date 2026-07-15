#!/usr/bin/env python3
"""Small-file coverage for scripts/export_hy3_mtp_sidecar.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_hy3_mtp_sidecar.py"
HAS_GGUF = importlib.util.find_spec("gguf") is not None

# Independent copy of the names consumed by hy3_mtp_weights_bind().  Keeping
# this fixture explicit ensures the exporter test fails if its own manifest
# drifts away from the runtime contract.
LOADER_TENSOR_NAMES = (
    "blk.80.nextn.eh_proj.weight",
    "blk.80.nextn.enorm.weight",
    "blk.80.nextn.hnorm.weight",
    "blk.80.nextn.shared_head_norm.weight",
    "blk.80.attn_norm.weight",
    "blk.80.attn_q.weight",
    "blk.80.attn_q_norm.weight",
    "blk.80.attn_k.weight",
    "blk.80.attn_k_norm.weight",
    "blk.80.attn_v.weight",
    "blk.80.attn_output.weight",
    "blk.80.ffn_norm.weight",
    "blk.80.ffn_gate_inp.weight",
    "blk.80.exp_probs_b.bias",
    "blk.80.ffn_gate_exps.weight",
    "blk.80.ffn_up_exps.weight",
    "blk.80.ffn_down_exps.weight",
    "blk.80.ffn_gate_shexp.weight",
    "blk.80.ffn_up_shexp.weight",
    "blk.80.ffn_down_shexp.weight",
)


@unittest.skipUnless(HAS_GGUF, "the gguf Python package is not installed")
class Hy3MtpSidecarExportTest(unittest.TestCase):
    @staticmethod
    def _write_source(
        path: Path,
        *,
        tensor_names=None,
        architecture: str = "hy_v3",
        embedding_length=4096,
        nextn_predict_layers=1,
        expert_used_count=8,
        embedding_as_int32: bool = False,
    ) -> None:
        import gguf
        import numpy as np

        writer = gguf.GGUFWriter(path, arch=architecture)
        writer.add_name("synthetic HY3 MTP export fixture")
        writer.add_uint32("hy_v3.block_count", 81)
        writer.add_uint32("hy_v3.context_length", 262144)
        if embedding_length is not None:
            if embedding_as_int32:
                writer.add_int32("hy_v3.embedding_length", embedding_length)
            else:
                writer.add_uint32("hy_v3.embedding_length", embedding_length)
        if nextn_predict_layers is not None:
            writer.add_uint32("hy_v3.nextn_predict_layers", nextn_predict_layers)
        writer.add_uint32("hy_v3.feed_forward_length", 13312)
        writer.add_uint32("hy_v3.attention.head_count", 64)
        writer.add_uint32("hy_v3.attention.head_count_kv", 8)
        writer.add_uint32("hy_v3.attention.key_length", 128)
        writer.add_uint32("hy_v3.attention.value_length", 128)
        writer.add_uint32("hy_v3.expert_count", 192)
        writer.add_uint32("hy_v3.expert_used_count", expert_used_count)
        writer.add_uint32("hy_v3.expert_feed_forward_length", 1536)
        writer.add_uint32("hy_v3.expert_shared_feed_forward_length", 1536)
        writer.add_bool("hy_v3.expert_weights_norm", True)
        writer.add_float32("hy_v3.expert_weights_scale", 2.826)
        writer.add_uint32("hy_v3.expert_gating_func", 2)
        writer.add_float32("hy_v3.attention.layer_norm_rms_epsilon", 1.0e-5)
        writer.add_float32("hy_v3.rope.freq_base", 11158840.0)
        writer.add_array("test.string_array", ["alpha", "beta", "gamma"])
        writer.add_tensor("blk.79.not_mtp.weight", np.arange(8, dtype=np.float32))
        for index, name in enumerate(tensor_names or LOADER_TENSOR_NAMES):
            if index == 0:
                # Exercise preservation of a raw quantized descriptor/payload,
                # not only plain NumPy dtypes.
                values = np.arange(68, dtype=np.uint8).reshape(2, 34)
                writer.add_tensor(
                    name,
                    values,
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
            else:
                values = np.arange(6, dtype=np.float32).reshape(2, 3) + index
                writer.add_tensor(name, values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_export_preserves_metadata_descriptors_and_payloads(self) -> None:
        import gguf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full.gguf"
            output = root / "mtp.gguf"
            self._write_source(source)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output), "--chunk-mib", "1"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("wrote and verified", result.stdout)

            source_reader = gguf.GGUFReader(source)
            output_reader = gguf.GGUFReader(output)
            self.assertEqual(len(output_reader.tensors), 20)
            self.assertEqual(
                {tensor.name for tensor in output_reader.tensors},
                set(LOADER_TENSOR_NAMES),
            )
            self.assertEqual(output_reader.fields["general.name"].contents(),
                             source_reader.fields["general.name"].contents())
            self.assertEqual(output_reader.fields["test.string_array"].contents(),
                             source_reader.fields["test.string_array"].contents())

            source_tensors = {
                tensor.name: tensor
                for tensor in source_reader.tensors
                if tensor.name.startswith("blk.80.")
            }
            for tensor in output_reader.tensors:
                original = source_tensors[tensor.name]
                self.assertEqual(tuple(tensor.shape), tuple(original.shape))
                self.assertEqual(tensor.tensor_type, original.tensor_type)
                self.assertEqual(tensor.data.tobytes(), original.data.tobytes())

    def test_dry_run_lists_metadata_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "full.gguf"
            self._write_source(source)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", str(source)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("metadata (preserved byte-for-byte)", result.stdout)
            self.assertIn("test.string_array: ARRAY/STRING[3]", result.stdout)
            self.assertIn("selected tensors (20, exactly blk.80.*)", result.stdout)
            self.assertIn("dry-run: no output written", result.stdout)

    def test_rejects_an_incomplete_block_80(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "incomplete.gguf"
            output = root / "mtp.gguf"
            self._write_source(source, tensor_names=LOADER_TENSOR_NAMES[:-1])
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly 20", result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_non_loader_tensor_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wrong-name.gguf"
            output = root / "mtp.gguf"
            names = list(LOADER_TENSOR_NAMES)
            names[-1] = "blk.80.unexpected.weight"
            self._write_source(source, tensor_names=names)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest is not compatible", result.stderr)
            self.assertIn("blk.80.ffn_down_shexp.weight", result.stderr)
            self.assertIn("blk.80.unexpected.weight", result.stderr)
            self.assertFalse(output.exists())

    def test_accepts_exp_probs_name_without_bias_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "full.gguf"
            names = list(LOADER_TENSOR_NAMES)
            names[names.index("blk.80.exp_probs_b.bias")] = "blk.80.exp_probs_b"
            self._write_source(source, tensor_names=names)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", str(source)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("blk.80.exp_probs_b", result.stdout)
            self.assertIn("dry-run: no output written", result.stdout)

    def test_rejects_legacy_architecture_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "legacy.gguf"
            self._write_source(source, architecture="hy-v3")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--dry-run", str(source)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("general.architecture is 'hy-v3'; expected 'hy_v3'", result.stderr)

    def test_rejects_missing_required_metadata(self) -> None:
        cases = (
            ("hy_v3.embedding_length", {"embedding_length": None}),
            ("hy_v3.nextn_predict_layers", {"nextn_predict_layers": None}),
        )
        for expected_key, kwargs in cases:
            with self.subTest(key=expected_key), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "missing.gguf"
                self._write_source(source, **kwargs)
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--dry-run", str(source)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"metadata key is missing: {expected_key}", result.stderr)

    def test_rejects_wrong_required_metadata_values_and_types(self) -> None:
        cases = (
            ("embedding value", {"embedding_length": 2048}, "is 2048; expected 4096"),
            ("nextn value", {"nextn_predict_layers": 2}, "is 2; expected 1"),
            ("expert top-k", {"expert_used_count": 4}, "is 4; expected 8"),
            ("embedding type", {"embedding_as_int32": True}, "has type INT32; expected UINT32"),
        )
        for label, kwargs, expected_error in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "invalid.gguf"
                self._write_source(source, **kwargs)
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--dry-run", str(source)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_force_replaces_output_path_not_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full.gguf"
            target = root / "unrelated.txt"
            output = root / "mtp.gguf"
            self._write_source(source)
            target.write_text("keep me\n")
            output.symlink_to(target)

            subprocess.run(
                [sys.executable, str(SCRIPT), "--force", str(source), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output.is_symlink())
            self.assertEqual(target.read_text(), "keep me\n")


if __name__ == "__main__":
    unittest.main()
