#!/usr/bin/env python3
"""Convert the FP4/FP8-native DSv4 dense GGUF into the ds4-loadable schema.

The native package's dense/model-dense.gguf uses:
  - GGUF type 42: block-128 FP8 = { uint8 e8m0_scale; uint8 e4m3fn qs[128] } (129 B)
  - GGUF type 30: BF16
  - renamed tensors (attn_kv_latent, hc_* without .weight, hc_head_*,
    attn_compress_* / indexer.compress_*)
  - missing ds4-required metadata keys (output_lora_rank, output_group_count,
    compress_ratios, hyper_connection.*, ...)

ds4 expects the chat-v2 dense schema exactly (tensor_expect_layout checks
types). This converter is template-driven: it walks the chat-v2 dense GGUF
(metadata copied verbatim, tensor names/types/dims as the target), pulls each
tensor's data from the native GGUF via a name map, and re-encodes to the
template type (F32/F16/Q8_0/I32). FP8 e4m3fn has a 3-bit mantissa, so
re-encoding to Q8_0 (8-bit) per 32-block is effectively lossless.

Usage:
  python3 convert_native_dense_to_ds4.py \
      --template <chat-v2>/dense/model-dense.gguf \
      --native   <native>/dense/model-dense.gguf \
      --out      <native>/dense/model-dense-ds4.gguf
"""

import argparse
import math
import struct
import sys

import numpy as np

ALIGN = 32

GGUF_VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_header(f):
    """Returns (n_tensors, kv_raw_bytes, tensor_infos, data_start)."""
    assert f.read(4) == b"GGUF", "not a GGUF file"
    version = struct.unpack("<I", f.read(4))[0]
    assert version == 3, f"unsupported GGUF version {version}"
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    kv_start = f.tell()

    def skip_value(t):
        if t in GGUF_VALUE_SIZES:
            f.seek(GGUF_VALUE_SIZES[t], 1)
        elif t == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            f.seek(n, 1)
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if et in GGUF_VALUE_SIZES:
                f.seek(n * GGUF_VALUE_SIZES[et], 1)
            else:
                for _ in range(n):
                    skip_value(et)
        else:
            raise ValueError(f"bad gguf value type {t}")

    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        f.seek(klen, 1)
        t = struct.unpack("<I", f.read(4))[0]
        skip_value(t)
    kv_end = f.tell()
    f.seek(kv_start)
    kv_raw = f.read(kv_end - kv_start)

    infos = []
    for _ in range(n_tensors):
        nlen = struct.unpack("<Q", f.read(8))[0]
        name = f.read(nlen).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, ttype, dims, off))
    data_start = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
    return n_tensors, n_kv, kv_raw, infos, data_start


def tensor_nbytes(ttype, n_elem):
    if ttype == 0:
        return n_elem * 4
    if ttype in (1, 30):
        return n_elem * 2
    if ttype == 8:  # q8_0
        assert n_elem % 32 == 0
        return n_elem // 32 * 34
    if ttype == 26:
        return n_elem * 4
    if ttype == 42:  # block-128 fp8
        assert n_elem % 128 == 0
        return n_elem // 128 * 129
    raise ValueError(f"unhandled tensor type {ttype}")


E4M3_LUT = None


def e4m3fn_lut():
    global E4M3_LUT
    if E4M3_LUT is None:
        raw = np.arange(256, dtype=np.uint8)
        sign = np.where(raw & 0x80, -1.0, 1.0)
        exp = ((raw >> 3) & 0x0F).astype(np.int64)
        man = (raw & 0x07).astype(np.float64)
        val = np.where(exp == 0, man * 2.0**-9, (1 + man / 8) * np.exp2(exp - 7.0)) * sign
        E4M3_LUT = val.astype(np.float32)
    return E4M3_LUT


def decode_native(buf, ttype, n_elem):
    if ttype == 0:
        return np.frombuffer(buf, dtype=np.float32, count=n_elem)
    if ttype == 1:
        return np.frombuffer(buf, dtype=np.float16, count=n_elem).astype(np.float32)
    if ttype == 30:
        u = np.frombuffer(buf, dtype=np.uint16, count=n_elem).astype(np.uint32) << 16
        return u.view(np.float32)
    if ttype == 26:
        return np.frombuffer(buf, dtype=np.int32, count=n_elem)
    if ttype == 42:
        blocks = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 129)
        e = blocks[:, 0].astype(np.int32)
        scale = np.where(e == 0, np.float32(0), np.exp2((e - 127).astype(np.float32)))
        vals = e4m3fn_lut()[blocks[:, 1:]]
        return (vals * scale[:, None]).reshape(-1).astype(np.float32)
    raise ValueError(f"unhandled native type {ttype}")


def encode_target(vals, ttype):
    if ttype == 26:
        return vals.astype("<i4").tobytes()
    if ttype == 0:
        return vals.astype("<f4").tobytes()
    if ttype == 1:
        return vals.astype("<f2").tobytes()
    if ttype == 8:  # q8_0: { f16 d; int8 qs[32] }, d = absmax/127, q = round(x/f16(d))
        x = vals.reshape(-1, 32).astype(np.float32)
        d = (np.abs(x).max(axis=1) / 127.0).astype(np.float16)
        df = d.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(df[:, None] > 0, np.rint(x / df[:, None]), 0.0)
        q = np.clip(q, -127, 127).astype(np.int8)
        out = np.empty((x.shape[0], 34), dtype=np.uint8)
        out[:, :2] = d.view(np.uint8).reshape(-1, 1, 2)[:, 0, :]
        out[:, 2:] = q.view(np.uint8)
        return out.tobytes()
    raise ValueError(f"unhandled target type {ttype}")


def native_name_candidates(chat_name):
    cands = [chat_name]
    if chat_name == "blk_dummy":
        return cands
    # output_hc_* -> hc_head_*
    if chat_name.startswith("output_hc_"):
        cands.append(chat_name.replace("output_hc_", "hc_head_").removesuffix(".weight"))
    # attn_kv -> attn_kv_latent
    cands.append(chat_name.replace(".attn_kv.weight", ".attn_kv_latent.weight"))
    # compressor renames
    c = chat_name.replace("attn_compressor_", "attn_compress_")
    cands.append(c)
    cands.append(c.removesuffix(".weight"))
    c = chat_name.replace("indexer_compressor_", "indexer.compress_")
    cands.append(c)
    cands.append(c.removesuffix(".weight"))
    # plain .weight suffix strip (attn_sinks, hc_*, ffn_gate_tid2eid)
    cands.append(chat_name.removesuffix(".weight"))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tf = open(args.template, "rb")
    nf = open(args.native, "rb")
    t_n_tensors, t_n_kv, t_kv_raw, t_infos, t_data_start = read_header(tf)
    _, _, _, n_infos, n_data_start = read_header(nf)
    native = {name: (ttype, dims, off) for name, ttype, dims, off in n_infos}
    template = {name: (ttype, dims, off) for name, ttype, dims, off in t_infos}

    # Tensors ds4 binds optionally that the native export omits; the gate_inp
    # router weights are bit-identical between the two exports (same base
    # model), so sourcing these from the template keeps the same router.
    TEMPLATE_FALLBACK = ("exp_probs_b.bias",)

    # resolve mapping first so we fail before writing anything
    mapping = {}
    fallback = []
    for name, ttype, dims, _ in t_infos:
        for cand in native_name_candidates(name):
            if cand in native:
                src_type, src_dims, _ = native[cand]
                if src_dims != dims:
                    sys.exit(f"dim mismatch {name}: template {dims} native {src_dims}")
                mapping[name] = cand
                break
        else:
            if name.endswith(TEMPLATE_FALLBACK):
                mapping[name] = None  # source from template
                fallback.append(name)
            else:
                sys.exit(f"no native tensor found for template tensor {name}")
    print(f"mapped {len(mapping)} tensors ({len(fallback)} sourced from template: "
          f"{fallback[:3]}...)")

    out = open(args.out, "wb")
    out.write(b"GGUF")
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<Q", t_n_tensors))
    out.write(struct.pack("<Q", t_n_kv))
    out.write(t_kv_raw)  # template metadata verbatim (known-good for ds4)

    # tensor info section with recomputed offsets
    offset = 0
    new_infos = []
    for name, ttype, dims, _ in t_infos:
        n_elem = math.prod(dims)
        nbytes = tensor_nbytes(ttype, n_elem)
        new_infos.append((name, ttype, dims, offset, nbytes))
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    for name, ttype, dims, off, _ in new_infos:
        nb = name.encode()
        out.write(struct.pack("<Q", len(nb)))
        out.write(nb)
        out.write(struct.pack("<I", len(dims)))
        for d in dims:
            out.write(struct.pack("<Q", d))
        out.write(struct.pack("<I", ttype))
        out.write(struct.pack("<Q", off))
    pad = (-out.tell()) % ALIGN
    out.write(b"\x00" * pad)
    data_base = out.tell()

    for i, (name, ttype, dims, off, nbytes) in enumerate(new_infos):
        n_elem = math.prod(dims)
        if mapping[name] is None:
            src_type, src_dims, src_off = template[name]
            tf.seek(t_data_start + src_off)
            raw = tf.read(tensor_nbytes(src_type, n_elem))
        else:
            src_type, src_dims, src_off = native[mapping[name]]
            nf.seek(n_data_start + src_off)
            raw = nf.read(tensor_nbytes(src_type, n_elem))
        vals = decode_native(raw, src_type, n_elem)
        blob = encode_target(vals, ttype)
        assert len(blob) == nbytes, name
        out.seek(data_base + off)
        out.write(blob)
        if i % 100 == 0:
            print(f"  [{i}/{len(new_infos)}] {name}")
    end = data_base + new_infos[-1][3] + new_infos[-1][4]
    out.seek(0, 2)
    if out.tell() < end:
        out.write(b"\x00" * (end - out.tell()))
    out.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
