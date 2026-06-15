#!/usr/bin/env python3
"""Layout-parameterized MXFP4 golden vectors (mxfp4-native-sidecar-plan item 5).

Emits, for a deterministic set of 32-element blocks covering both nibble
orders and the E8M0 edges (e = 0, 127, 254):

  tests/test-vectors/mxfp4/blocks_ggml.bin     ggml block_mxfp4 (17 B, split-half)
  tests/test-vectors/mxfp4/plane_data.bin      seq-pair FP4 data plane (16 B/block)
  tests/test-vectors/mxfp4/plane_scales.bin    E8M0 scale plane (1 B/block)
  tests/test-vectors/mxfp4/expected_f32.bin    32 float32 dequant values per block
  tests/test-vectors/mxfp4/manifest.json       block count + descriptions

Both the macOS 26 fallback kernels (split-half readers) and the MPP 4.1
native path (plane readers) must reproduce expected_f32 bit-for-bit at the
dequant level; the repack kernel must map blocks_ggml -> plane_*.  Keep this
file in sync with metal/mxfp4_common.h.
"""

import json
import math
import random
import struct
from pathlib import Path

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e8m0(e: int) -> float:
    # value = 2^(e-127); e = 0 decodes to 0.0 (the f32 bit pattern e<<23)
    return struct.unpack('<f', struct.pack('<I', e << 23))[0]


def make_blocks():
    rng = random.Random(0x4D58)  # 'MX'
    blocks = []  # (description, e, elems[32] as nibble values 0..15)
    for e, tag in ((0, 'e=0 zero-decode'), (127, 'e=127 scale 1.0'),
                   (254, 'e=254 max scale'), (1, 'e=1 min nonzero')):
        blocks.append((tag, e, [rng.randrange(16) for _ in range(32)]))
    blocks.append(('all nibbles 0..15 twice', 130, [i % 16 for i in range(32)]))
    blocks.append(('asymmetric order check', 120, [(i * 7 + 3) % 16 for i in range(32)]))
    for i in range(10):
        blocks.append((f'random {i}', rng.randrange(100, 150),
                       [rng.randrange(16) for _ in range(32)]))
    return blocks


def main():
    out = Path(__file__).resolve().parent / 'test-vectors' / 'mxfp4'
    out.mkdir(parents=True, exist_ok=True)
    blocks = make_blocks()

    ggml = bytearray()
    plane_data = bytearray()
    plane_scales = bytearray()
    expected = bytearray()
    for _, e, elems in blocks:
        # ggml split-half: qs[j] low nibble = elem j, high nibble = elem j+16
        ggml.append(e)
        ggml.extend(elems[j] | (elems[j + 16] << 4) for j in range(16))
        # seq-pair plane: byte b holds elem 2b (lo) and 2b+1 (hi)
        plane_data.extend(elems[2 * b] | (elems[2 * b + 1] << 4) for b in range(16))
        plane_scales.append(e)
        scale = e8m0(e)
        # f32 semantics: e=254 x |6.0| overflows float32 -> signed inf,
        # exactly what the GPU dequant produces.
        F32_MAX = 3.4028234663852886e+38
        vals = [v if abs(v) <= F32_MAX else math.copysign(math.inf, v)
                for v in (scale * E2M1[x] for x in elems)]
        expected.extend(struct.pack('<32f', *vals))

    (out / 'blocks_ggml.bin').write_bytes(ggml)
    (out / 'plane_data.bin').write_bytes(plane_data)
    (out / 'plane_scales.bin').write_bytes(plane_scales)
    (out / 'expected_f32.bin').write_bytes(expected)
    (out / 'manifest.json').write_text(json.dumps({
        'block_count': len(blocks),
        'block_elems': 32,
        'ggml_block_bytes': 17,
        'descriptions': [d for d, _, _ in blocks],
        'nibble_orders': {
            'blocks_ggml.bin': 'split-half (qs[j] lo=elem j, hi=elem j+16)',
            'plane_data.bin': 'sequential-pair (byte b: lo=elem 2b, hi=elem 2b+1)',
        },
    }, indent=2) + '\n')
    print(f'wrote {len(blocks)} golden blocks to {out}')


if __name__ == '__main__':
    main()
