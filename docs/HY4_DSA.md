# HY4 DSA and long contexts

Explicit HY4 contexts above 2048 now allocate native DSA state. The runtime
retains F32 MLA keys and F32 indexer keys for the full history, chooses the
highest-scoring 2048 causal positions, then runs sink-aware attention over
those rows. Through 2048 positions every causal key participates. Expert
routing remains native top-8 and is independent of this attention top-k.

## Sources and layout

The llama.cpp source revision `34cccef` has no native DSA implementation.
This follow-up uses [SGLang HYV4 at 55bf3380](https://github.com/sgl-project/sglang/blob/55bf3380e073ea1b3399155763228e827139d532/python/sglang/srt/models/hunyuan_v4.py),
its [DSA Indexer](https://github.com/sgl-project/sglang/blob/55bf3380e073ea1b3399155763228e827139d532/python/sglang/srt/layers/attention/dsa/dsa_indexer.py),
[MLA forward](https://github.com/sgl-project/sglang/blob/55bf3380e073ea1b3399155763228e827139d532/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py),
and [official Tencent config at 705d81ee](https://huggingface.co/tencent/Hy4-preview/blob/705d81ee51566a186d645b74c974d642ef2828fe/config.json).

The actual `Hy4-preview-Flash-STQ1_0/model-dense-f16head.gguf` has 32 indexer
heads, 128 channels, 64 rotary channels, and top-k 2048. Full indexers are
layers 0, 1, 5, 9, ..., 77; the other 57 layers reuse the most recent full
layer's indices for the current token. Each full indexer uses:

- Query: its projection of the attention RMS-normalized 2048-wide Q-LoRA.
- Key: its projection of the attention input, then affine LayerNorm with
  epsilon 1e-6 and NeoX RoPE on the final 64 channels, theta 10,000,000.
- Gates: an unactivated 32-wide projection of the attention input.
- Score: `sum_h(gate[h] * max(dot(query[h], key[position]), 0)) / 64`.

The GGUF's layer-0 normalization weight matches the official BF16 tensor
exactly after conversion to F32. Each of the first 128 query projection rows
and all 128 key projection rows matches the same official row, with minimum
cosine correlation >0.99998 after Q8 dequantization. This verifies the native
tail/NeoX layout; no channel permutation is applied. SGLang moves that tail
to its prefix internally. This implementation retains F32 index keys rather
than SGLang's FP8 storage. Its optional normalized Hadamard rotation is
orthogonal and is also omitted by SGLang's fused indexer; it is not needed
for the unquantized dot products here. Bit parity with FP8 CUDA is not claimed.

## Agent command

Build when no process is using this checkout. Start with eight slots for a
new machine, then use an explicitly sized bank after verifying memory usage.
For the 128 GiB M5 Max, a 48-slot command for the published bank path is:

```sh
make -j4 ds4-agent
HY4_PACKAGE="$HOME/Models/HY4/Hy4-preview-Flash-STQ1_0"
DS4_PROFILE=none \
DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0 \
DS4_HY4_SG_ATTENTION=1 \
DS4_HY4_FUSED_ROUTER=1 \
DS4_AGENT_CACHE_DIR="$PWD/profile_runs/hy4-dsa/agent-cache" \
./ds4-agent \
  -m "$HY4_PACKAGE/model-dense-f16head.gguf" \
  --moe-sidecar "$HY4_PACKAGE/sidecar" \
  --moe-mode slot-bank --moe-slot-bank 48 \
  --ctx 50480 --tokens 128 --temp 0 --seed 1 --nothink --no-int8
```

DSA is automatic for contexts above 2048. No opt-out permits full-attention
fallback beyond that boundary. The default when `--ctx` is omitted remains
2048. The model's 1,048,576-token metadata is the hard input ceiling; usable
capacity is also limited by memory. At ctx50480, MLA keys occupy about 8.45 GiB,
and the 21 full indexers add 517.6 MiB, plus attention/indexer scratch. Dense
weights and the 48-slot bank are additional allocations. Prefill is still
one token at a time; larger context allocation does not make prefill batched.

Long-context HY4 snapshots use payload v2 and serialize every full indexer's
keys. Short sessions retain v1. A v1 cache is rejected by a long session,
causing agent prefix reconstruction. Use a separate cache directory when
comparing runs. Expert sidecar bytes and formats are unchanged.

## Memory sizing

`--moe-slot-bank N` controls cached experts **per MoE layer**, independently of
`--ctx`. For this exact package, one additional slot across its 77 MoE layers
is 0.7522 GiB of expert records, calculated from the sidecar manifest strides.
The dense GGUF is 19.712 GiB. F32 MLA and index keys consume 186 KiB per context
token: `78 * (512 + 64) * 4 + 21 * 128 * 4` bytes.

At ctx50480, context buffers including index/attention scratch are about
8.99 GiB. Capacity estimates for fully populated expert banks are:

| Slots per layer | Expert records | Dense file + expert records + context buffers |
| --- | ---: | ---: |
| 48 | 36.11 GiB | 64.80 GiB |
| 64 | 48.14 GiB | 76.84 GiB |
| 80 | 60.18 GiB | 88.87 GiB |
| 96 | 72.21 GiB | 100.91 GiB |

These sums exclude other runtime/driver allocations and other applications;
they are not measured process footprints or guarantees of residency. On the
128 GiB M5 Max used here, Metal reported a 107.52 GiB recommended working-set
budget. Try 64 slots before 80; 96 leaves little GPU budget for other work.
Changing context changes that headroom: 131072 tokens need 23.25 GiB of keys,
262144 need 46.5 GiB, and the metadata ceiling of 1048576 needs 186 GiB for keys
alone. Those context lengths have not been validated with full model histories.

The local experimental `DS4_HY4_MMAP_SLOTS=1` path references file-backed expert
records instead of allocating the bank as private buffers. A logical slot hit
does not guarantee that its pages remain in RAM: macOS can reclaim them and
fault them back from storage. A sampled mapped agent reported about 9.4 GiB
physical footprint, while a separate ordinary-bank test reported 45.2 GiB;
these process readings do not account for file-backed pages in the same way.
An 8-9 GiB reading therefore does not mean the full dense model and expert
cache cost only that much RAM. Mapped-slot and resident-gate experiments remain
local; the published command above uses the ordinary bank.

## Regression commands and coverage

```sh
make -j4 tests/test_hy4_dsa tests/test_hy4_dsa_payload \
  tests/test_hy4_session tests/test_hy4_long_context
./tests/test_hy4_dsa
./tests/test_hy4_dsa_payload
./tests/test_hy4_session "$HY4_PACKAGE/model-dense-f16head.gguf" "$HY4_PACKAGE/sidecar"
./tests/test_hy4_long_context "$HY4_PACKAGE/model-dense-f16head.gguf" "$HY4_PACKAGE/sidecar" 48
```

- Model-free native Metal: 148,648 checks, including histories of 2047, 2048,
  2049 and 50,480 rows; maximum index-score error 1.19e-7 against an independent
  double-accumulating oracle. Original/SG attention agree with directly indexed
  scalar MLA. Future rows are poisoned; top-k IDs are distinct and causal.
- Payload regression: 402 checks across the actual v1/v2 readers and writers,
  all 21 full/57 shared layers, exact roundtrip, truncation, malformed headers,
  invalid token IDs and old-cache rejection.
- Actual model: short eight-slot lifecycle passes; the checked 80 top-logit
  IDs and values match the earlier native reference exactly.
- Actual long-context model lifecycle: PASS with 48 slots and allocated
  ctx50480. A fresh 2047-token prefill on the local mapped-slot build crossed
  into sparse attention at history lengths 2049 and 2050. Cancellation at 2048,
  rewind/replay, resumed prefill, v2 snapshot restore, malformed/v1 rejection,
  recovery and reset passed. The checked logits were exact after each replay.
- The public ordinary-bank build loaded that same-model 2047-token fixture,
  independently crossed the DSA boundary and passed the same lifecycle checks.
  Its complete 2050-token snapshot was byte-identical to the mapped build:
  390,942,768 bytes, including every serialized MLA/index key and final logit.
  All 112 checked top-logit ID/value pairs across seven stages also matched.
  This public run is fixture replay, not a fresh full public prefill.
- An earlier public fresh-prefill attempt was stopped after GPU submission
  stalls before 64 tokens; it provides no correctness verdict. The known
  ordinary-bank slowdown after mapped experiments remains unresolved.

The long harness defaults to fresh prefill. To compare builds without repeating
that prefix, set `HY4_LONG_PREFIX_OUT` to a permanent file path on the fresh run,
then `HY4_LONG_PREFIX_IN` to that file on a run using the **same model and prompt**.
The harness checks the restored position and exact token IDs. Optional
`HY4_LONG_PAYLOAD_OUT` saves the final 2050-token payload for byte comparison.
These are raw test fixtures, not agent cache files; do not mix model artifacts.
The final JSON reports `result: PASS`, `history_tokens: 2050`, and an explicit
`prefix_source: fresh` or `restored`. Allocating ctx50480 is not 50k-token
prefill validation.

Full 50k-token model prefill, million-token execution, long-document quality,
and an 8 tokens/s long-context performance claim are not validated here.
