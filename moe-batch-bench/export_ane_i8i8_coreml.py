#!/usr/bin/env python3
"""Export DS4 W8A8 ANE MLP graphs as Core ML packages."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

COREML_TARGET = ct.target.iOS26


def _f16(value: float) -> np.float16:
    return np.float16(value)


def build_gateup_program(h: int, i: int, b: int, w_scale: float, x_scale: float):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(b, h), dtype=types.int8),
        ],
        opset_version=COREML_TARGET,
    )
    def prog(Wgq, Wuq, Xq):  # noqa: N803 - names match model I/O.
        Wg = mb.dequantize(
            input=Wgq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wg"
        )
        Wu = mb.dequantize(
            input=Wuq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wu"
        )
        X = mb.dequantize(
            input=Xq, scale=_f16(x_scale), zero_point=np.int8(0), name="X"
        )
        X3 = mb.expand_dims(x=X, axes=[0], name="X3")
        gate = mb.matmul(
            x=X3, y=Wg, transpose_x=False, transpose_y=False, name="gate"
        )
        up = mb.matmul(x=X3, y=Wu, transpose_x=False, transpose_y=False, name="up")
        return gate, up

    return prog


def build_down_program(h: int, i: int, b: int, w_scale: float, mid_scale: float):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(i, h), dtype=types.int8),
            mb.TensorSpec(shape=(b, i), dtype=types.int8),
        ],
        opset_version=COREML_TARGET,
    )
    def prog(Wdq, hidden_q):  # noqa: N803 - names match model I/O.
        Wd = mb.dequantize(
            input=Wdq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wd"
        )
        hidden = mb.dequantize(
            input=hidden_q,
            scale=_f16(mid_scale),
            zero_point=np.int8(0),
            name="hidden",
        )
        hidden3 = mb.expand_dims(x=hidden, axes=[0], name="hidden3")
        Y = mb.matmul(
            x=hidden3, y=Wd, transpose_x=False, transpose_y=False, name="Y"
        )
        return Y

    return prog


def build_full_program(
    h: int, i: int, b: int, w_scale: float, x_scale: float, mid_scale: float
):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(i, h), dtype=types.int8),
            mb.TensorSpec(shape=(b, h), dtype=types.int8),
        ],
        opset_version=COREML_TARGET,
    )
    def prog(Wgq, Wuq, Wdq, Xq):  # noqa: N803 - names match model I/O.
        Wg = mb.dequantize(
            input=Wgq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wg"
        )
        Wu = mb.dequantize(
            input=Wuq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wu"
        )
        Wd = mb.dequantize(
            input=Wdq, scale=_f16(w_scale), zero_point=np.int8(0), name="Wd"
        )
        X = mb.dequantize(
            input=Xq, scale=_f16(x_scale), zero_point=np.int8(0), name="X"
        )
        X3 = mb.expand_dims(x=X, axes=[0], name="X3")
        gate = mb.matmul(
            x=X3, y=Wg, transpose_x=False, transpose_y=False, name="gate"
        )
        up = mb.matmul(x=X3, y=Wu, transpose_x=False, transpose_y=False, name="up")
        gate_c = mb.clip(
            x=gate,
            alpha=mb.const(val=_f16(-10.0), name="clamp_lo"),
            beta=mb.const(val=_f16(10.0), name="clamp_hi"),
            name="gate_c",
        )
        up_c = mb.clip(
            x=up,
            alpha=mb.const(val=_f16(-10.0), name="up_clamp_lo"),
            beta=mb.const(val=_f16(10.0), name="up_clamp_hi"),
            name="up_c",
        )
        act = mb.silu(x=gate_c, name="act")
        hidden_fp = mb.mul(x=act, y=up_c, name="hidden_fp")
        hidden_q = mb.quantize(
            input=hidden_fp,
            scale=_f16(mid_scale),
            zero_point=np.int8(0),
            output_dtype="int8",
            name="hidden_q",
        )
        hidden = mb.dequantize(
            input=hidden_q,
            scale=_f16(mid_scale),
            zero_point=np.int8(0),
            name="hidden",
        )
        Y = mb.matmul(x=hidden, y=Wd, transpose_x=False, transpose_y=False, name="Y")
        return Y

    return prog


def build_tiled_fused_program(
    h: int,
    i: int,
    b: int,
    tile_i: int,
    w_scale: float,
    x_scale: float,
    mid_scale: float,
):
    if tile_i <= 0:
        raise ValueError("tile_i must be positive")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(h, i), dtype=types.int8),
            mb.TensorSpec(shape=(i, h), dtype=types.int8),
            mb.TensorSpec(shape=(b, h), dtype=types.int8),
        ],
        opset_version=COREML_TARGET,
    )
    def prog(Wgq, Wuq, Wdq, Xq):  # noqa: N803 - names match model I/O.
        X = mb.dequantize(
            input=Xq, scale=_f16(x_scale), zero_point=np.int8(0), name="X"
        )
        X3 = mb.expand_dims(x=X, axes=[0], name="X3")
        y_acc = None
        for start in range(0, i, tile_i):
            end = min(start + tile_i, i)
            tag = f"i{start}_{end}"
            Wgq_i = mb.slice_by_index(
                x=Wgq, begin=[0, start], end=[h, end], name=f"Wgq_{tag}"
            )
            Wuq_i = mb.slice_by_index(
                x=Wuq, begin=[0, start], end=[h, end], name=f"Wuq_{tag}"
            )
            Wdq_i = mb.slice_by_index(
                x=Wdq, begin=[start, 0], end=[end, h], name=f"Wdq_{tag}"
            )
            Wg_i = mb.dequantize(
                input=Wgq_i, scale=_f16(w_scale), zero_point=np.int8(0), name=f"Wg_{tag}"
            )
            Wu_i = mb.dequantize(
                input=Wuq_i, scale=_f16(w_scale), zero_point=np.int8(0), name=f"Wu_{tag}"
            )
            Wd_i = mb.dequantize(
                input=Wdq_i, scale=_f16(w_scale), zero_point=np.int8(0), name=f"Wd_{tag}"
            )
            gate_i = mb.matmul(
                x=X3,
                y=Wg_i,
                transpose_x=False,
                transpose_y=False,
                name=f"gate_{tag}",
            )
            up_i = mb.matmul(
                x=X3,
                y=Wu_i,
                transpose_x=False,
                transpose_y=False,
                name=f"up_{tag}",
            )
            gate_c_i = mb.clip(
                x=gate_i,
                alpha=mb.const(val=_f16(-10.0), name=f"gate_lo_{tag}"),
                beta=mb.const(val=_f16(10.0), name=f"gate_hi_{tag}"),
                name=f"gate_c_{tag}",
            )
            up_c_i = mb.clip(
                x=up_i,
                alpha=mb.const(val=_f16(-10.0), name=f"up_lo_{tag}"),
                beta=mb.const(val=_f16(10.0), name=f"up_hi_{tag}"),
                name=f"up_c_{tag}",
            )
            act_i = mb.silu(x=gate_c_i, name=f"act_{tag}")
            hidden_fp_i = mb.mul(x=act_i, y=up_c_i, name=f"hidden_fp_{tag}")
            hidden_q_i = mb.quantize(
                input=hidden_fp_i,
                scale=_f16(mid_scale),
                zero_point=np.int8(0),
                output_dtype="int8",
                name=f"hidden_q_{tag}",
            )
            hidden_i = mb.dequantize(
                input=hidden_q_i,
                scale=_f16(mid_scale),
                zero_point=np.int8(0),
                name=f"hidden_{tag}",
            )
            y_i = mb.matmul(
                x=hidden_i,
                y=Wd_i,
                transpose_x=False,
                transpose_y=False,
                name=f"Y_{tag}",
            )
            y_acc = y_i if y_acc is None else mb.add(x=y_acc, y=y_i, name=f"Y_acc_{tag}")
        return y_acc

    return prog


def convert_and_save(prog, path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")
        shutil.rmtree(path)
    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=COREML_TARGET,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    mlmodel.save(str(path))


def save_program_text(prog, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.write_text(str(prog), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("moe-batch-bench/coreml_exports"))
    p.add_argument("--shape", nargs=3, type=int, metavar=("H", "I", "B"), default=(4096, 2048, 256))
    p.add_argument("--w-qscale", type=float, default=512.0)
    p.add_argument("--x-qscale", type=float, default=32.0)
    p.add_argument("--mid-qscale", type=float, default=32.0)
    p.add_argument(
        "--kind",
        choices=("fused", "fused-tiled", "all", "gateup", "down", "full"),
        default="fused-tiled",
        help="'fused'/'fused-tiled' export the tiled single expert MLP; 'full' is the materializing diagnostic.",
    )
    p.add_argument("--tile-i", type=int, default=256, help="Intermediate dimension tile for fused-tiled.")
    p.add_argument("--force", action="store_true", help="Overwrite existing packages/text.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h, i, b = args.shape
    args.out_dir.mkdir(parents=True, exist_ok=True)
    w_scale = 1.0 / args.w_qscale
    x_scale = 1.0 / args.x_qscale
    mid_scale = 1.0 / args.mid_qscale

    specs = []
    if args.kind in ("all", "gateup"):
        specs.append(("gateup", build_gateup_program(h, i, b, w_scale, x_scale)))
    if args.kind in ("all", "down"):
        specs.append(("down", build_down_program(h, i, b, w_scale, mid_scale)))
    if args.kind in ("all", "full"):
        specs.append(("fused_expert", build_full_program(h, i, b, w_scale, x_scale, mid_scale)))
    if args.kind in ("all", "fused", "fused-tiled"):
        specs.append((
            f"fused_expert_tiled{args.tile_i}",
            build_tiled_fused_program(h, i, b, args.tile_i, w_scale, x_scale, mid_scale),
        ))

    suffix = f"H{h}_I{i}_B{b}_wq{args.w_qscale:g}_xq{args.x_qscale:g}_midq{args.mid_qscale:g}"
    for name, prog in specs:
        mil_path = args.out_dir / f"ds4_i8i8_{name}_{suffix}.mil.txt"
        model_path = args.out_dir / f"ds4_i8i8_{name}_{suffix}.mlpackage"
        print(f"writing {mil_path}")
        save_program_text(prog, mil_path, args.force)
        print(f"converting {model_path}")
        convert_and_save(prog, model_path, args.force)
        print(f"saved {model_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
