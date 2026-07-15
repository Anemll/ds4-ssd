#!/usr/bin/env python3
"""Extract the HY3 block-80 MTP tensors into a standalone GGUF sidecar.

The source HY3 GGUF is very large, so this tool deliberately never turns a
tensor into a NumPy array.  ``gguf.GGUFReader`` is used only to mmap and parse
the GGUF directory.  Metadata bytes and the 20 selected tensor payloads are
then copied with bounded ``pread`` chunks into a new GGUF v3 file.

All source metadata is preserved byte-for-byte.  Tensor names, dimensions,
GGML types, and payload bytes are preserved; only tensor-relative offsets and
the header tensor count change.
"""

from __future__ import annotations

import argparse
import os
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


GGUF_HEADER_SIZE = 24
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
MTP_PREFIX = "blk.80."
EXPECTED_TENSOR_COUNT = 20
DEFAULT_CHUNK_MIB = 16
REQUIRED_ARCHITECTURE = "hy_v3"
REQUIRED_BLOCK_COUNT = 81
REQUIRED_CONTEXT_LENGTH = 262144
REQUIRED_EMBEDDING_LENGTH = 4096
REQUIRED_FEED_FORWARD_LENGTH = 13312
REQUIRED_HEAD_COUNT = 64
REQUIRED_HEAD_COUNT_KV = 8
REQUIRED_HEAD_DIM = 128
REQUIRED_EXPERT_COUNT = 192
REQUIRED_EXPERT_USED_COUNT = 8
REQUIRED_EXPERT_FEED_FORWARD_LENGTH = 1536
REQUIRED_EXPERT_SHARED_FEED_FORWARD_LENGTH = 1536
REQUIRED_EXPERT_GATING_FUNC = 2
REQUIRED_RMS_EPSILON = 9.999999747378752e-06
REQUIRED_ROPE_FREQ_BASE = 11158840.0
REQUIRED_EXPERT_WEIGHTS_SCALE = 2.8259999752044678
REQUIRED_NEXTN_PREDICT_LAYERS = 1

# Keep this manifest in lockstep with hy3_mtp_weights_bind().  The two
# exp_probs spellings are the only loader-supported name variation.
REQUIRED_TENSOR_NAMES = frozenset(
    {
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
        "blk.80.ffn_gate_exps.weight",
        "blk.80.ffn_up_exps.weight",
        "blk.80.ffn_down_exps.weight",
        "blk.80.ffn_gate_shexp.weight",
        "blk.80.ffn_up_shexp.weight",
        "blk.80.ffn_down_shexp.weight",
    }
)
EXP_PROBS_TENSOR_NAMES = (
    "blk.80.exp_probs_b",
    "blk.80.exp_probs_b.bias",
)
VALID_TENSOR_MANIFESTS = tuple(
    REQUIRED_TENSOR_NAMES | {exp_probs_name}
    for exp_probs_name in EXP_PROBS_TENSOR_NAMES
)
assert all(len(manifest) == EXPECTED_TENSOR_COUNT for manifest in VALID_TENSOR_MANIFESTS)


class ExportError(RuntimeError):
    """A source validation or export failure with a user-facing message."""


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dimensions: Tuple[int, ...]
    ggml_type: int
    ggml_type_name: str
    n_bytes: int
    source_offset: int
    relative_offset: int


@dataclass(frozen=True)
class ExportPlan:
    source: Path
    identity: FileIdentity
    version: int
    source_tensor_count: int
    metadata_count: int
    metadata_end: int
    source_data_offset: int
    alignment: int
    metadata_lines: Tuple[str, ...]
    tensors: Tuple[TensorSpec, ...]
    tensor_payload_bytes: int
    tensor_storage_bytes: int


def _load_gguf() -> Any:
    try:
        import gguf  # type: ignore
    except ImportError as exc:
        raise ExportError(
            "Python package 'gguf' is required (use the environment that "
            "provides GGUFReader)"
        ) from exc
    return gguf


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _identity(st: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=int(st.st_dev),
        inode=int(st.st_ino),
        size=int(st.st_size),
        mtime_ns=int(st.st_mtime_ns),
        mode=stat.S_IMODE(st.st_mode),
    )


def _same_file_identity(a: FileIdentity, b: FileIdentity) -> bool:
    return (
        a.device == b.device
        and a.inode == b.inode
        and a.size == b.size
        and a.mtime_ns == b.mtime_ns
    )


def _read_header(path: Path) -> Tuple[int, int, int]:
    try:
        with path.open("rb") as source:
            header = source.read(GGUF_HEADER_SIZE)
    except OSError as exc:
        raise ExportError(f"cannot read source GGUF {path}: {exc}") from exc
    if len(header) != GGUF_HEADER_SIZE:
        raise ExportError("source is too small to contain a GGUF v3 header")
    if header[:4] != GGUF_MAGIC:
        raise ExportError("source does not start with GGUF magic")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:])
    if version != GGUF_VERSION:
        raise ExportError(
            f"source uses GGUF v{version}; this exporter requires GGUF v{GGUF_VERSION}"
        )
    return int(version), int(tensor_count), int(metadata_count)


def _short_value(value: Any, limit: int = 160) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _metadata_line(key: str, field: Any) -> str:
    type_names = [getattr(value_type, "name", str(value_type)) for value_type in field.types]
    type_label = "/".join(type_names) if type_names else "UNKNOWN"
    if type_names and type_names[0] == "ARRAY":
        return f"{key}: {type_label}[{len(field.data)}]"
    try:
        value = field.contents()
    except Exception as exc:  # A listing failure must not weaken the export.
        return f"{key}: {type_label} <unprintable: {exc}>"
    return f"{key}: {type_label} {_short_value(value)}"


def _require_metadata(reader: Any, key: str, type_name: str, expected: Any) -> None:
    field = reader.fields.get(key)
    if field is None:
        raise ExportError(f"required HY3 MTP metadata key is missing: {key}")
    actual_types = tuple(
        getattr(value_type, "name", str(value_type)) for value_type in field.types
    )
    if actual_types != (type_name,):
        actual_label = "/".join(actual_types) if actual_types else "UNKNOWN"
        raise ExportError(
            f"required HY3 MTP metadata {key} has type {actual_label}; "
            f"expected {type_name}"
        )
    try:
        actual = field.contents()
    except Exception as exc:
        raise ExportError(f"cannot read required HY3 MTP metadata {key}: {exc}") from exc
    if actual != expected:
        raise ExportError(
            f"required HY3 MTP metadata {key} is {actual!r}; expected {expected!r}"
        )


def _validate_required_metadata(reader: Any) -> None:
    _require_metadata(reader, "general.architecture", "STRING", REQUIRED_ARCHITECTURE)
    _require_metadata(reader, "hy_v3.block_count", "UINT32", REQUIRED_BLOCK_COUNT)
    _require_metadata(
        reader, "hy_v3.context_length", "UINT32", REQUIRED_CONTEXT_LENGTH
    )
    _require_metadata(
        reader,
        "hy_v3.embedding_length",
        "UINT32",
        REQUIRED_EMBEDDING_LENGTH,
    )
    _require_metadata(
        reader,
        "hy_v3.feed_forward_length",
        "UINT32",
        REQUIRED_FEED_FORWARD_LENGTH,
    )
    _require_metadata(reader, "hy_v3.attention.head_count", "UINT32", REQUIRED_HEAD_COUNT)
    _require_metadata(
        reader,
        "hy_v3.attention.head_count_kv",
        "UINT32",
        REQUIRED_HEAD_COUNT_KV,
    )
    _require_metadata(reader, "hy_v3.attention.key_length", "UINT32", REQUIRED_HEAD_DIM)
    _require_metadata(reader, "hy_v3.attention.value_length", "UINT32", REQUIRED_HEAD_DIM)
    _require_metadata(reader, "hy_v3.expert_count", "UINT32", REQUIRED_EXPERT_COUNT)
    _require_metadata(
        reader, "hy_v3.expert_used_count", "UINT32", REQUIRED_EXPERT_USED_COUNT
    )
    _require_metadata(
        reader,
        "hy_v3.expert_feed_forward_length",
        "UINT32",
        REQUIRED_EXPERT_FEED_FORWARD_LENGTH,
    )
    _require_metadata(
        reader,
        "hy_v3.expert_shared_feed_forward_length",
        "UINT32",
        REQUIRED_EXPERT_SHARED_FEED_FORWARD_LENGTH,
    )
    _require_metadata(reader, "hy_v3.expert_weights_norm", "BOOL", True)
    _require_metadata(
        reader,
        "hy_v3.expert_weights_scale",
        "FLOAT32",
        REQUIRED_EXPERT_WEIGHTS_SCALE,
    )
    _require_metadata(
        reader,
        "hy_v3.expert_gating_func",
        "UINT32",
        REQUIRED_EXPERT_GATING_FUNC,
    )
    _require_metadata(
        reader,
        "hy_v3.attention.layer_norm_rms_epsilon",
        "FLOAT32",
        REQUIRED_RMS_EPSILON,
    )
    _require_metadata(
        reader,
        "hy_v3.rope.freq_base",
        "FLOAT32",
        REQUIRED_ROPE_FREQ_BASE,
    )
    _require_metadata(
        reader,
        "hy_v3.nextn_predict_layers",
        "UINT32",
        REQUIRED_NEXTN_PREDICT_LAYERS,
    )


def _validate_tensor_manifest(selected: Sequence[Any]) -> None:
    if len(selected) != EXPECTED_TENSOR_COUNT:
        names = ", ".join(tensor.name for tensor in selected) or "<none>"
        raise ExportError(
            f"expected exactly {EXPECTED_TENSOR_COUNT} tensors named {MTP_PREFIX}*, "
            f"found {len(selected)}: {names}"
        )

    actual = frozenset(tensor.name for tensor in selected)
    if actual in VALID_TENSOR_MANIFESTS:
        return

    closest = min(
        VALID_TENSOR_MANIFESTS,
        key=lambda expected: len(actual.symmetric_difference(expected)),
    )
    missing = sorted(closest - actual)
    unexpected = sorted(actual - closest)
    details = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if unexpected:
        details.append("unexpected=" + ",".join(unexpected))
    raise ExportError(
        "block-80 tensor manifest is not compatible with the HY3 MTP loader"
        + (": " + "; ".join(details) if details else "")
    )


def analyze_source(source_path: Path) -> ExportPlan:
    """Parse only the source directory and build a validated export plan."""
    source = source_path.expanduser().resolve()
    if not source.is_file():
        raise ExportError(f"source GGUF does not exist or is not a file: {source}")

    before = _identity(source.stat())
    version, header_tensor_count, header_metadata_count = _read_header(source)
    gguf = _load_gguf()

    try:
        reader = gguf.GGUFReader(source)
    except Exception as exc:
        raise ExportError(f"GGUFReader rejected {source}: {exc}") from exc

    endian_name = getattr(reader.endianess, "name", str(reader.endianess))
    if endian_name != "LITTLE":
        raise ExportError("big-endian GGUF input is not supported by the DS4 sidecar loader")
    if len(reader.tensors) != header_tensor_count:
        raise ExportError(
            "GGUF tensor-count mismatch between header and parsed directory: "
            f"{header_tensor_count} != {len(reader.tensors)}"
        )

    parsed_metadata_count = int(reader.fields["GGUF.kv_count"].contents())
    if parsed_metadata_count != header_metadata_count:
        raise ExportError(
            "GGUF metadata-count mismatch between header and parsed directory: "
            f"{header_metadata_count} != {parsed_metadata_count}"
        )
    if not reader.tensors:
        raise ExportError("source GGUF has no tensors")

    _validate_required_metadata(reader)

    alignment = int(reader.alignment)
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ExportError(f"invalid GGUF tensor alignment: {alignment}")
    metadata_end = int(reader.tensors[0].field.offset)
    source_data_offset = int(reader.data_offset)
    if metadata_end < GGUF_HEADER_SIZE or metadata_end > source_data_offset:
        raise ExportError("GGUFReader reported an invalid metadata/tensor-directory boundary")

    metadata_lines = tuple(
        _metadata_line(key, field)
        for key, field in reader.fields.items()
        if not key.startswith("GGUF.")
    )

    selected = [tensor for tensor in reader.tensors if tensor.name.startswith(MTP_PREFIX)]
    _validate_tensor_manifest(selected)

    relative_offset = 0
    specs = []
    payload_bytes = 0
    for tensor in selected:
        n_bytes = int(tensor.n_bytes)
        source_offset = int(tensor.data_offset)
        dimensions = tuple(int(value) for value in tensor.shape)
        if not dimensions or any(value <= 0 for value in dimensions):
            raise ExportError(f"tensor {tensor.name} has invalid dimensions {dimensions}")
        if n_bytes <= 0:
            raise ExportError(f"tensor {tensor.name} has an invalid byte size {n_bytes}")
        if source_offset < source_data_offset or source_offset + n_bytes > before.size:
            raise ExportError(
                f"tensor {tensor.name} payload [{source_offset}, {source_offset + n_bytes}) "
                "is outside the source GGUF"
            )
        if source_offset % alignment != 0:
            raise ExportError(
                f"tensor {tensor.name} source offset {source_offset} is not {alignment}-byte aligned"
            )

        tensor_type = tensor.tensor_type
        specs.append(
            TensorSpec(
                name=tensor.name,
                dimensions=dimensions,
                ggml_type=int(tensor_type),
                ggml_type_name=getattr(tensor_type, "name", str(tensor_type)),
                n_bytes=n_bytes,
                source_offset=source_offset,
                relative_offset=relative_offset,
            )
        )
        payload_bytes += n_bytes
        relative_offset = _align_up(relative_offset + n_bytes, alignment)

    after = _identity(source.stat())
    if not _same_file_identity(before, after):
        raise ExportError("source GGUF changed while its directory was being parsed")

    return ExportPlan(
        source=source,
        identity=before,
        version=version,
        source_tensor_count=header_tensor_count,
        metadata_count=header_metadata_count,
        metadata_end=metadata_end,
        source_data_offset=source_data_offset,
        alignment=alignment,
        metadata_lines=metadata_lines,
        tensors=tuple(specs),
        tensor_payload_bytes=payload_bytes,
        tensor_storage_bytes=relative_offset,
    )


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def print_plan(plan: ExportPlan, include_metadata: bool) -> None:
    print(f"source: {plan.source}")
    print(
        f"GGUF v{plan.version}: {plan.metadata_count} metadata keys, "
        f"{plan.source_tensor_count} source tensors, alignment={plan.alignment}"
    )
    if include_metadata:
        print("metadata (preserved byte-for-byte):")
        for line in plan.metadata_lines:
            print(f"  {line}")
    print(f"selected tensors ({len(plan.tensors)}, exactly {MTP_PREFIX}*):")
    for index, tensor in enumerate(plan.tensors, 1):
        dimensions = "x".join(str(value) for value in tensor.dimensions)
        print(
            f"  {index:2d}. {tensor.name}  {tensor.ggml_type_name}({tensor.ggml_type}) "
            f"[{dimensions}]  {_human_bytes(tensor.n_bytes)}"
        )
    print(
        "planned tensor payload: "
        f"{_human_bytes(plan.tensor_payload_bytes)} "
        f"({_human_bytes(plan.tensor_storage_bytes)} including alignment padding)"
    )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise ExportError("short write while creating sidecar")
        view = view[written:]


def _copy_range(source_fd: int, output_fd: int, offset: int, length: int, chunk_size: int) -> None:
    copied = 0
    while copied < length:
        want = min(chunk_size, length - copied)
        data = os.pread(source_fd, want, offset + copied)
        if not data:
            raise ExportError(
                f"unexpected EOF while reading source range at offset {offset + copied}"
            )
        _write_all(output_fd, data)
        copied += len(data)


def _encode_tensor_info(tensor: TensorSpec) -> bytes:
    name = tensor.name.encode("utf-8")
    result = bytearray()
    result += struct.pack("<Q", len(name))
    result += name
    result += struct.pack("<I", len(tensor.dimensions))
    for dimension in tensor.dimensions:
        result += struct.pack("<Q", dimension)
    result += struct.pack("<IQ", tensor.ggml_type, tensor.relative_offset)
    return bytes(result)


def _ranges_equal(
    left: Path,
    left_offset: int,
    right: Path,
    right_offset: int,
    length: int,
    chunk_size: int = 1024 * 1024,
) -> bool:
    left_fd = os.open(str(left), os.O_RDONLY)
    right_fd = os.open(str(right), os.O_RDONLY)
    try:
        compared = 0
        while compared < length:
            want = min(chunk_size, length - compared)
            a = os.pread(left_fd, want, left_offset + compared)
            b = os.pread(right_fd, want, right_offset + compared)
            if a != b or not a:
                return False
            compared += len(a)
        return True
    finally:
        os.close(left_fd)
        os.close(right_fd)


def verify_sidecar(path: Path, plan: ExportPlan) -> None:
    """Perform a directory-level verification without reading tensor payloads."""
    version, tensor_count, metadata_count = _read_header(path)
    if version != plan.version or tensor_count != len(plan.tensors):
        raise ExportError("verification failed: output GGUF header does not match the plan")
    if metadata_count != plan.metadata_count:
        raise ExportError("verification failed: output metadata count changed")

    gguf = _load_gguf()
    try:
        reader = gguf.GGUFReader(path)
    except Exception as exc:
        raise ExportError(f"verification failed: output is not readable GGUF: {exc}") from exc
    if len(reader.tensors) != len(plan.tensors):
        raise ExportError("verification failed: output tensor-directory size changed")
    if int(reader.alignment) != plan.alignment:
        raise ExportError("verification failed: output alignment changed")
    if int(reader.tensors[0].field.offset) != plan.metadata_end:
        raise ExportError("verification failed: output metadata extent changed")

    for actual, expected in zip(reader.tensors, plan.tensors):
        actual_dimensions = tuple(int(value) for value in actual.shape)
        actual_type = int(actual.tensor_type)
        expected_offset = int(reader.data_offset) + expected.relative_offset
        if (
            actual.name != expected.name
            or actual_dimensions != expected.dimensions
            or actual_type != expected.ggml_type
            or int(actual.n_bytes) != expected.n_bytes
            or int(actual.data_offset) != expected_offset
        ):
            raise ExportError(f"verification failed for tensor descriptor {expected.name}")

    expected_size = int(reader.data_offset) + plan.tensor_storage_bytes
    if path.stat().st_size != expected_size:
        raise ExportError(
            f"verification failed: output size is {path.stat().st_size}, expected {expected_size}"
        )
    if not _ranges_equal(
        plan.source,
        GGUF_HEADER_SIZE,
        path,
        GGUF_HEADER_SIZE,
        plan.metadata_end - GGUF_HEADER_SIZE,
    ):
        raise ExportError("verification failed: output metadata bytes differ from source")


def export_sidecar(
    plan: ExportPlan,
    output_path: Path,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_MIB * 1024 * 1024,
) -> Path:
    """Stream ``plan`` to ``output_path`` and atomically install the result."""
    output = Path(os.path.abspath(str(output_path.expanduser())))
    output_exists = os.path.lexists(str(output))
    same_as_source = False
    if output_exists:
        try:
            same_as_source = os.path.samefile(str(output), str(plan.source))
        except OSError:
            same_as_source = False
    else:
        same_as_source = output.resolve(strict=False) == plan.source
    if same_as_source:
        raise ExportError("output path must not be the source GGUF")
    if not output.parent.is_dir():
        raise ExportError(f"output directory does not exist: {output.parent}")
    if output_exists and not force:
        raise ExportError(f"output already exists (pass --force to replace it): {output}")
    if chunk_size <= 0:
        raise ExportError("copy chunk size must be positive")

    source_fd = os.open(str(plan.source), os.O_RDONLY)
    output_fd: Optional[int] = None
    temporary: Optional[Path] = None
    try:
        opened_identity = _identity(os.fstat(source_fd))
        if not _same_file_identity(plan.identity, opened_identity):
            raise ExportError("source GGUF changed after the export plan was built")

        output_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
        )
        temporary = Path(temporary_name)
        output_mode = plan.identity.mode & 0o666
        os.fchmod(output_fd, output_mode if output_mode else 0o644)

        header = GGUF_MAGIC + struct.pack(
            "<IQQ", plan.version, len(plan.tensors), plan.metadata_count
        )
        _write_all(output_fd, header)
        _copy_range(
            source_fd,
            output_fd,
            GGUF_HEADER_SIZE,
            plan.metadata_end - GGUF_HEADER_SIZE,
            chunk_size,
        )
        for tensor in plan.tensors:
            _write_all(output_fd, _encode_tensor_info(tensor))

        tensor_data_offset = _align_up(os.lseek(output_fd, 0, os.SEEK_CUR), plan.alignment)
        current = os.lseek(output_fd, 0, os.SEEK_CUR)
        _write_all(output_fd, bytes(tensor_data_offset - current))

        for index, tensor in enumerate(plan.tensors, 1):
            current = os.lseek(output_fd, 0, os.SEEK_CUR)
            expected = tensor_data_offset + tensor.relative_offset
            if current > expected:
                raise ExportError(f"internal offset overlap before {tensor.name}")
            _write_all(output_fd, bytes(expected - current))
            print(
                f"copying {index:2d}/{len(plan.tensors)} {tensor.name} "
                f"({_human_bytes(tensor.n_bytes)})",
                file=sys.stderr,
            )
            _copy_range(
                source_fd,
                output_fd,
                tensor.source_offset,
                tensor.n_bytes,
                chunk_size,
            )

        final_size = tensor_data_offset + plan.tensor_storage_bytes
        current = os.lseek(output_fd, 0, os.SEEK_CUR)
        if current > final_size:
            raise ExportError("internal output-size accounting error")
        _write_all(output_fd, bytes(final_size - current))

        final_identity = _identity(os.fstat(source_fd))
        if not _same_file_identity(plan.identity, final_identity):
            raise ExportError("source GGUF changed while tensor payloads were copied")

        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
        verify_sidecar(temporary, plan)

        if os.path.lexists(str(output)) and not force:
            raise ExportError(f"output appeared during export; refusing to replace it: {output}")
        os.replace(str(temporary), str(output))
        temporary = None
        return output
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(source_fd)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _chunk_mib(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1 or parsed > 1024:
        raise argparse.ArgumentTypeError("must be between 1 and 1024 MiB")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream exactly the 20 blk.80.* HY3 MTP tensors into a GGUF sidecar "
            "while preserving source metadata byte-for-byte."
        )
    )
    parser.add_argument("source", type=Path, help="full HY3 GGUF containing block 80")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="destination sidecar GGUF (not needed with --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and list metadata/tensors without writing an output file",
    )
    parser.add_argument(
        "--force", action="store_true", help="atomically replace an existing output file"
    )
    parser.add_argument(
        "--chunk-mib",
        type=_chunk_mib,
        default=DEFAULT_CHUNK_MIB,
        metavar="N",
        help=f"bounded streaming copy buffer in MiB (default: {DEFAULT_CHUNK_MIB})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("output is required unless --dry-run is used")

    try:
        plan = analyze_source(args.source)
        print_plan(plan, include_metadata=args.dry_run)
        if args.dry_run:
            print("dry-run: no output written")
            return 0
        assert args.output is not None
        output = export_sidecar(
            plan,
            args.output,
            force=args.force,
            chunk_size=args.chunk_mib * 1024 * 1024,
        )
        print(f"wrote and verified: {output} ({_human_bytes(output.stat().st_size)})")
        return 0
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
