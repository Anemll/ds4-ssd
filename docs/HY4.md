# HY4 preview SSD support

This native Metal path ports the HY4 implementation in
`Anemll/anemll-flash-llama.cpp` branch `HY4-1.25-bit` at `34cccef`.
It supports the actual `hyv4` STQ1_0 package: 78 layers, four independent HC
streams, 256 routed experts with native top-8, and mixed STQ1_0/IQ2_XXS gate/up
and IQ3_XXS/IQ4_XS down projections. It runs within DS4 sessions and `ds4-agent`.

## Start conservatively

Use a permanent checkout. The package stores its dense model beside `sidecar/`,
so pass both paths explicitly. Start with **8 slots per layer**. Do not use
`--resident` or auto cache sizing for the initial test.

```sh
make -j4 ds4-agent
HY4_PACKAGE="$HOME/Models/HY4/Hy4-preview-Flash-STQ1_0"
DS4_PROFILE=none DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0 \
./ds4-agent \
  -m "$HY4_PACKAGE/model-dense-f16head.gguf" \
  --moe-sidecar "$HY4_PACKAGE/sidecar" \
  --moe-mode slot-bank --moe-slot-bank 8 \
  --ctx 1024 --tokens 8 --temp 0 --seed 1 --nothink --no-int8 \
  --non-interactive -p 'Reply with exactly OK.'
```

For isolated testing, prefix the agent command with
`DS4_AGENT_CACHE_DIR=/absolute/path/to/validation/agent-cache`. This keeps its
`sysprompt.kv` and saved sessions separate from the default `$HOME/.ds4/kvcache`.
An unset or empty override keeps the existing default.

The supplied F16-head dense file is about 19.7 GiB; eight slots across the 77
MoE layers require about 6.018 GiB before alignment. Dense plus bank is about
25.7 GiB, with additional cache and runtime scratch. Verify startup reports
`slots=8`; slot count and native expert top-8 are different settings. Expert
count overrides below eight are rejected.

HY4 sink-aware attention runs in Metal by default. For numerical debugging,
`DS4_HY4_CPU_ATTENTION=1` selects the scalar F32 attention reference; unset or 0
uses Metal. This does not change slot count, routing, or the 2048-key limit.
The two implementations are compared on deterministic buffers before native
model validation.

Attention sigmoid gating and the ordered eight-expert sum after the down
projection also run in Metal. This removes 78 gate and 77 expert-sum CPU
synchronization points per token. The existing iHC and token completion
boundaries still wait for GPU work before slot reuse or cancellation.
`DS4_HY4_CPU_POINTWISE=1` restores both scalar operations for A/B debugging;
unset or 0 uses Metal. This switch does not change iHC, attention, or I/O.
Startup reports `gate/reduce=Metal` or `gate/reduce=CPU reference`.

## Boundaries

The supplied source revision does not implement native DSA. Full MLA attention
matches top-2048 attention only while causal history fits 2048 keys. The native
runtime therefore rejects `--ctx` above 2048, even though the model advertises
1048576. When `ds4` or `ds4-agent` identifies HY4 and `--ctx` was omitted, it chooses 2048.
HY4 MTP and CUDA execution are unsupported.

Initial and resumed prefill run token by token through the same hard-protected
slot request used by decode. All reads and GPU operations complete before a
token checkpoint is published. An optional session cancellation callback is
polled at token boundaries; cancellation preserves completed tokens and does
not invoke their progress callbacks again on resume. Long prompts are slower
than a future validated batched prefill implementation.

HY4 uses independent iHC, sink-aware gated MLA, and expert weights after the
down projection. Its routed SwiGLU clamps gate inputs above 10 and up inputs
to [-10, 10]; dense/shared FFN remains unclamped. DS4 fused weighted-SwiGLU and
Sinkhorn-HC paths are not substituted.

## Reproduce the checks

```sh
make flash-moe-slot-test flash-moe-slot-test-sanitize flash-moe-io-test
make hy4-math-test hy4-quant-test hy4-attention-test hy4-pointwise-test hy4-sanitize-test
make hy4-metadata-test HY4_MODEL="$HY4_PACKAGE/model-dense-f16head.gguf"
make tests/test_hy4_session
./tests/test_hy4_session \
  "$HY4_PACKAGE/model-dense-f16head.gguf" "$HY4_PACKAGE/sidecar" > hy4-session.jsonl
```

The native harness fixes 8 slots and context 128, followed by a separate ctx=4
full-capacity snapshot case with one live bank at a time. It checks raw prompt IDs,
greedy generation, resumed prefill, rewind, snapshot save/restore, reset/replay,
and prefill/decode cancellation with exact progress-callback counts.

For an independent source comparison, use the supplied source binary:

```sh
/path/to/anemll-flash-llama.cpp/build/bin/llama-cli \
  -m "$HY4_PACKAGE/model-dense-f16head.gguf" \
  --moe-sidecar "$HY4_PACKAGE/sidecar" \
  --moe-mode slot-bank --moe-slot-bank 8 --no-slot8 \
  -c 128 -b 1 -ub 1 -n 4 --temp 0 --seed 1 \
  -ctk f32 -ctv f32 -fa off --no-warmup \
  --moe-trace-harness -p Hello \
  --oracle-dump /path/to/hy4-oracle --oracle-topk 16
```

Source warmup must be disabled so evaluation 0 is the prompt.
Then compare:

```sh
python3 tests/compare_hy4_oracle.py /path/to/hy4-oracle/manifest.json hy4-session.jsonl
```

Initial source comparison matched all four greedy predictions and 64 top-logit
IDs, maximum absolute logit difference 0.000104 (tolerance 0.001). Native rewind,
snapshot replay, reset/replay, and cancelled-prefill resume matched top 16 logits exactly.
A ctx=4 session also restored a snapshot saved at completely full capacity. These are
bounded short-prompt results, not a quality evaluation or performance claim.

The native `ds4-agent` command above passed with the actual package, eight
slots and context 1024: a cold system-prompt prefill returned `OK`. A restarted
process restored its cached 709-token system prompt, then two separately
submitted prompts returned `OK` and `4` (for 2 + 2), with process exit 0.
The first system-prompt prefill took about eight minutes; subsequent starts
loaded the isolated cache. This validates a short two-turn conversation;
long conversations have not been evaluated. A subsequent native tool check
with eight slots and context 2048 restored that cache, generated a `read` call,
consumed the local file result, and returned its exact code (exit 0).

Quant tests also compare exact dequantization against the source GGML library
and sample all four actual sidecar types; see [HY4_QUANT_VALIDATION.md](HY4_QUANT_VALIDATION.md).

`DS4_HY4_TRACE_DIR=/existing/directory` optionally captures first-token F32
intermediates for comparison with the source oracle. It is unset by default.
Capture files are overwritten when another prefix is evaluated from position 0.

## Compare GPU gating and expert reduction

The focused pointwise test requires bit-identical ordered F32 expert sums,
including a multiply/add contraction witness. CPU libm and Metal sigmoid are
compared within four float epsilons for exp/divide/multiply. It covers guarded
offset views, in-place gating, queued producer/consumer operations, and invalid
sizes/overlaps. No model is opened by this test.

For a conservative model A/B, the native harness also accepts `--greedy16`:

```sh
DS4_HY4_CPU_POINTWISE=1 ./tests/test_hy4_session \
  "$HY4_PACKAGE/model-dense-f16head.gguf" "$HY4_PACKAGE/sidecar" \
  --greedy16 > hy4-cpu-pointwise.jsonl
DS4_HY4_CPU_POINTWISE=0 ./tests/test_hy4_session \
  "$HY4_PACKAGE/model-dense-f16head.gguf" "$HY4_PACKAGE/sidecar" \
  --greedy16 > hy4-metal-pointwise.jsonl
python3 tests/compare_hy4_oracle.py --native-reference \
  hy4-cpu-pointwise.jsonl hy4-metal-pointwise.jsonl
```

Both commands fix eight slots, native top-8 and context 128. Each JSONL records
all 16 generated token IDs, top-16 logits at every step, and decode wall time
including top-logit capture. Run serially and account for OS/SSD cache state
before interpreting timing differences. The normal harness mode still runs
four greedy tokens followed by cancellation, rewind and snapshot regressions.

The M5 Max follow-up passed 115,801 focused checks (nine sizes up to the
16384-element gate), with bit-identical expert sums and maximum sigmoid error
`2.22e-7 * max(1, abs(reference))`. Serial CPU/Metal A/B runs matched all 16
generated tokens and all top-16 IDs at 17 positions, maximum logit difference
0.0000066. The complete cancellation/rewind/snapshot harness also passed;
source F32/no-flash comparison remained within 0.000105 (tolerance 0.001).

After an initial warmup sequence, matched warm runs with eight slots measured:

| Run | CPU gate/reduction | Metal gate/reduction |
| --- | ---: | ---: |
| 1 (16 decode tokens) | 1.884 t/s | 1.989 t/s |
| 2 (16 decode tokens) | 1.893 t/s | 1.992 t/s |

Combined rates were 1.889 versus 1.991 t/s, about 5.4% higher in this short
warm-cache sample. The initial CPU run was slower and is excluded from that
comparison. This is not a cold-SSD measurement or a general throughput claim.

For a 48-slot agent CPU/GPU/I/O breakdown and actual dispatch counts, see
[HY4_PROFILING.md](HY4_PROFILING.md).
