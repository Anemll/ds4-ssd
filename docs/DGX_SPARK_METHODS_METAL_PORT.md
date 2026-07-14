# DGX Spark optimization methods adapted to DS4 Metal

Date: 2026-07-11

External reference: [tonyd2wild/DeepSeek-v4-Flash-DSpark-Abliterated-Uncensored-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-Abliterated-Uncensored-2x-DGX-Spark), audited at commit `de79a2f9eea657ad1a942d38de10cebe48525048`.

This comparison ports optimization ideas only. It keeps the local target model
`/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major` and local DSpark draft
`/Users/anemll/Models/DSv4-Flash-DSpark-draft`.

## Important command correction

The originally supplied command does not enable DSpark. `DS4_DSPARK_*`
variables are inert unless the command also contains:

```sh
--draft dspark \
--draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft
```

Without those flags the command measures ordinary one-token target decode.

## Method inventory and Metal disposition

| DGX/CUDA method | Metal/DS4 disposition | Result |
| --- | --- | --- |
| B12X fused W4A16 MoE | The external kernel is CUDA-only and does not match the local IQ2_XXS/MXFP4 layout. Its lifecycle is already represented by the resident identity expert bank, persistent model views, reusable private scratch, fused IQ2 gate/up/SwiGLU, and direct Q2 down paths. | Retain local kernels; do not port B12X code. |
| CUDA graph capture for target and draft | DS4 already keeps command encoders open, uses one terminal drain for draft and verifier, submits verifier work in nonblocking four-layer splits, and keeps tensor/scratch addresses stable. Existing route-local ICB replay was slower and the current resident identity route bypasses it. | Keep single-drain/split path; leave ICB off. |
| Fused Markov projection plus argmax | Added a Metal two-stage F16 Markov projection/reduction that emits tile maxima instead of a vocabulary-sized correction vector. | Implemented behind `DS4_DSPARK_FUSED_MARKOV_ARGMAX=1`; byte-exact in n=320 and n=1200 controls. |
| Hardware-aware variable draft prefix | Added cumulative confidence survival scheduling so later rows are submitted only when their cumulative expected survival clears the threshold. | Implemented behind `DS4_DSPARK_CONF_CUMULATIVE=1`; best measured threshold is `0.30` with confidence scale `0.85`. |
| Sparse MLA / shared K-V work | Added an exact N<=6 direct-resident verifier-attention kernel. It keeps each query row independent while cooperatively loading the K/V rows shared by the verifier prefixes, preserving the original accumulation and online-softmax order. | Implemented behind `DS4_DSPARK_ATTN_VARMAP_DIRECT_SHARED_KV=1`; the tensor audit and final output are byte-exact. |
| GPU-side rejected-prefix/state handling | Packed consecutive non-emitting ratio-4 index-compressor updates and all requested prefix snapshots into one Metal dispatch. Compression-boundary arithmetic remains on the established exact path. | Implemented behind `DS4_DSPARK_PACKED_INDEX_FRONTIER=1`; two short repeats and the 1,200-token run are byte-exact. |
| Larger verifier architecture / relaxed product lane | Rows6 previously rejected relaxed acceptance and fell back to the older five-row path, paying a separate target decode. An opt-in now keeps row 0 as the known target argmax, target-verifies all N<=6 rows, and applies relaxed acceptance only to the suffix. | Implemented behind `DS4_DSPARK_RELAXED_ROWS6=1`; non-byte by design and default off. It reached 51.32 t/s at n=320 and 49.94 t/s at n=1200 without profiling. |
| Local vocab-parallel argmax | Tensor parallelism is absent on the single M5 Max. The transferable part is avoiding full-logit materialization, now covered by fused Markov argmax. | Ported in single-device form. |
| Replicated Markov W1 | Removes TP communication on the two-GPU implementation. There is no corresponding communication on one Apple GPU. | Not applicable. |
| GPU rejected-context mask and stable request slots | DS4 already keeps target/draft state and prefix snapshots on device and repairs a skipped suffix in the next live draft command buffer. The external repository's zero-draft fallback is not true plain decode. | Local `DS4_DSPARK_TRUE_PLAIN_SKIP=1` is the correct implementation. |
| Async scheduling/copies and auxiliary CUDA streams | DS4 draft prefetch and verifier layer splits are already byte-safe. Same-GPU overlap has negligible wall value because verification is 98-99% GPU-active. | Retain current overlap; a second Metal queue is not a substitute for less GPU work. |
| FlashInfer warmup/autotuning | Transferable as context/shape-specific Metal tactic selection. Current attention thresholds and varmap workgroup count already have local heuristics and diagnostics. | Useful future workflow, not a current 10 ms/block win. |
| TP=2, NCCL/RoCE, prefix caching, continuous batching | These improve multi-GPU serving or multi-request aggregate throughput, not this single-prompt CLI benchmark. | Not applicable to the requested measurement. |

The external repository also contains disabled or contradictory paths. Its
checked-in draft still selects with argmax; the advertised probabilistic path
is not active. Its nonuniform fallback returns dummy zero drafts rather than
performing ordinary decode. Current compose also overrides the newer proposer
with an older incompatible interface. Those are not methods to copy.

## CUDA workflow translated to Metal

1. Prepare weights once: retain resident expert records and persistent tensor
   views; do not repack or allocate scratch inside the token loop.
2. Keep addresses and shapes stable: reuse private scratch and fixed N<=6
   verifier buffers, analogous to CUDA graph input buffers.
3. Encode coarse command streams: use the existing one-terminal-drain draft and
   verifier encoders, with nonblocking verifier splits to hide CPU encoding.
4. Fuse bandwidth-bound terminal operations: Markov projection plus argmax is
   the first retained port. Fuse only when a tensor-level and final-output byte
   gate passes.
5. Schedule for measured hardware cost: cumulative confidence uses the actual
   verifier row economics instead of always submitting the maximum prefix.
6. Share invariant verifier inputs inside a threadgroup: the retained N<=6
   direct-attention tactic stages common K/V rows once while keeping separate
   query state and the exact reduction order for every verifier row.
7. Pack small mutable-state work only where operations are independent: the
   ratio-4 index frontier combines non-emitting stores and prefix snapshots,
   but preserves the existing pool/RMS/RoPE sequence at compression boundaries.
8. Warm and tune by context bucket: select attention workgroup/tactic by real
   key count, while preserving a fail-closed exact path.
9. Promote only after repeated A/B plus `cmp`: CUDA timing claims are not
   transferable without the same model, prompt, context, acceptance, and output
   contract.

## Retained measurements on the exact supplied prompt

All speculative rows below add the missing `--draft dspark --draft-path ...`
flags. The exact lane uses greedy target output and passed byte comparison.

| Configuration | n | Generation | Draft/block | Verify/block | Tau | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| User force-first/confidence/margin bundle | 1200 | 35.90 t/s | 11.53 ms | 83.91 ms | 3.72 | exact |
| Rows6 confidence control | 1200 | 38.32 t/s | 11.79 ms | 96.33 ms | 4.26 | exact |
| + fused Markov argmax | 1200 | 39.32 t/s | 10.90 ms | 94.38 ms | 4.26 | exact |
| + cumulative survival, threshold 0.30 | 1200 | **40.08 t/s** | 10.91 ms | 89.57 ms | 4.14 | exact |
| Cumulative stack + shared-KV verifier attention | 1200 | **44.28 t/s** | 10.94 ms | 79.35 ms | 4.14 | byte-exact; first run was 44.44 t/s |
| + packed ratio-4 index frontier | 1200 | **45.42 t/s** | 10.69 ms | 77.38 ms | 4.14 | byte-exact; 90.70 ms/block |
| Same cumulative stack, short smoke | 320 | **43.12 t/s** | 10.92 ms | 90.21 ms | 4.49 | exact |
| Static rows6 + shared-KV verifier attention | 320 | **44.86 t/s** | - | 97.43 ms | - | byte-exact; control was 41.00 t/s / 107.73 ms |
| Cumulative stack + shared-KV verifier attention | 320 | **47.31 t/s** | 10.85 ms | 80.99 ms | 4.49 | byte-exact; 94.65 ms/block |
| + packed ratio-4 index frontier | 320 | **48.10-48.20 t/s** | 10.55 ms | 79.62 ms | 4.49 | byte-exact repeats; 92.95 ms/block |
| Token-pair draft-attention diagnostic | 320 | 47.34 t/s | 10.83 ms | 81.04 ms | 4.49 | byte-exact but timing-neutral; code removed |
| Two-of-three draft-layer early exit | 320 | 35.26 t/s | 8.11 ms | 61.95 ms | 2.54 | byte-exact target; rejected after acceptance fell to 76.9% |
| Fast-relaxed diagnostic | 320 | 46.33 t/s | 10.37 ms | 75.26 ms | 4.36 | non-byte, coherent short smoke |
| Unbounded accept with loop guards | 320 | 48.60 t/s | - | - | 4.52 approx. | malformed CSS values |
| Loop guard disabled | 320 | **53.82 t/s** | 10.28 ms | 79.74 ms | 5.00 | rejected: repeated malformed CSS |

The shared-KV kernel passed 164 tensor comparisons, including 123 rows6
comparisons, with `max=0` and `rms=0`. The final generated output also passed
byte comparison. On the full 1,200-token workload it reduced verifier time
from 89.57 to 79.35 ms/block and raised generation from 40.08 to 44.28 t/s;
an independent unprofiled repeat measured 44.44 t/s. The token-pair
draft-attention kernel was exact but measured 47.34 t/s versus 47.31 t/s, so
the neutral kernel and gate were removed. Packing only the smaller ratio-4
index frontier then produced repeatable 48.10/48.14/48.20 t/s short runs and
raised the full run to 45.42 t/s, all with byte-identical output. Applying the same
packing tactic to the four-times-wider attention state regressed to 46.33 t/s
and was removed.

The approximately 53.8 t/s row is a hardware ceiling, not a valid decoder
result. It crosses 50 only by committing every draft token; the generated CSS
immediately repeats and becomes syntactically invalid. No valid exact run in
this pass reached 50 t/s.

## Optional guarded relaxed rows6 lane

`DS4_DSPARK_RELAXED_ROWS6=1` lets the relaxed product lane use Mode-B/rows6
instead of the older five-row verifier. Row 0 remains the committed target
argmax, all rows still run through the target verifier, and only suffix
acceptance is relaxed. It is safer than unconditional draft acceptance, but it
is intentionally not byte-identical to greedy target decode.

| Configuration | n | Generation | Acceptance | Scheduled rows | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| rows6 relaxed, cumulative threshold 0.10 | 320 | **51.32 t/s** | 95.8% | 5.66 | coherent short HTML/CSS sample; non-byte |
| rows6 relaxed, cumulative threshold 0.20, profiled | 1200 | 49.46 t/s | 96.3% | 5.37 | coherent, no repetition; non-byte |
| same threshold 0.20 without profiler | 1200 | **49.94 t/s** | 96.3% | 5.37 | output identical to profiled run |

The short run clears 50 t/s. The longer run is effectively at the target but
does not establish a sustained >=50 t/s guarantee. Wider top-k and unbounded
variants were not promoted: they did not improve the long result and produced
lower-quality or malformed CSS in at least one tested trajectory.

## Why valid exact 50 t/s still needs another architectural cut

The shared-KV architecture plus packed index frontier raised the exact n=320
result from 43.12 to 48.14 t/s. Its best block is 92.95 ms for 4.49 committed
tokens. At 50 t/s, the same block may cost only about 89.8 ms, so the remaining
valid short-run gap is roughly 3.2 ms per block. On n=1200, 4.14 committed
tokens in 90.70 ms needs about 7.9 ms/block more to sustain 50 t/s.
Verification is still 79.62/77.38 ms and the GPU is effectively saturated.
Therefore:

- removing more waits cannot close the gap;
- skipped blocks are already capable of true ordinary decode;
- ICB/static binding cleanup is too small and previously regressed; and
- same-GPU target/draft overlap cannot create compute capacity.

The first architecture target from the original analysis is now implemented:
the exact N<=6 verifier-attention kernel reads shared K/V cooperatively while
preserving row-specific attention semantics. The next target is a low-register
exact routed-MoE batch kernel or another exact reduction of at least 5 ms per
block. Whole-draft ANE offload requires new ANE implementations for mutable
attention, FP8/MXFP4 routed MoE, router/top-k, and Markov selection; the current
repository has only projection and MLP ANE building blocks, so it is not an
activation-only change.

## Recommended exact command

```sh
DS4_DSPARK_TRUE_PLAIN_SKIP=1 \
DS4_DSPARK_VERIFY_ROWS6=1 \
DS4_DSPARK_ROWS6_CONFIDENCE=1 \
DS4_DSPARK_CONF_SCALE=0.85 \
DS4_DSPARK_CONF_THRESHOLD=0.30 \
DS4_DSPARK_CONF_CUMULATIVE=1 \
DS4_DSPARK_FUSED_MARKOV_ARGMAX=1 \
DS4_DSPARK_ATTN_VARMAP_DIRECT_SHARED_KV=1 \
DS4_DSPARK_PACKED_INDEX_FRONTIER=1 \
./ds4 \
  -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark \
  --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --nothink --temp 0 --resident -c 20000 \
  -p "aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics. Save the result to /tmp/si-cv5.html . test file by opening it in web browser. skip web search"
```

This is the fastest retained exact stack from this pass. The new ports remain
opt-in until broader cross-prompt and long-context parity gates pass. The
token-pair draft-attention experiment is absent because its 47.34 versus 47.31
t/s difference was timing noise rather than a useful gain.

## Optional >50 t/s short-run command

This lane changes greedy output and should remain explicit:

```sh
DS4_DSPARK_EVAL_MARGIN_GATE=0 \
DS4_DSPARK_VERIFY_ROWS6=1 \
DS4_DSPARK_ROWS6_CONFIDENCE=1 \
DS4_DSPARK_CONF_SCALE=0.85 \
DS4_DSPARK_CONF_THRESHOLD=0.10 \
DS4_DSPARK_CONF_CUMULATIVE=1 \
DS4_DSPARK_RELAXED_ROWS6=1 \
DS4_DSPARK_RELAXED_ACCEPT=1 \
DS4_DSPARK_RELAXED_TOPK=256 \
DS4_DSPARK_RELAXED_LOGIT_DELTA=10 \
DS4_DSPARK_RELAXED_MARGIN_DISABLE=1 \
DS4_DSPARK_RELAXED_CONFIDENCE_GATE_DISABLE=1 \
DS4_DSPARK_RELAXED_ALLOW_TARGET_TOP_REPEAT=1 \
DS4_DSPARK_FUSED_MARKOV_ARGMAX=1 \
DS4_DSPARK_ATTN_VARMAP_DIRECT_SHARED_KV=1 \
DS4_DSPARK_PACKED_INDEX_FRONTIER=1 \
./ds4 \
  -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark \
  --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --nothink --temp 0 --resident -c 20000 \
  -p "aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics. Save the result to /tmp/si-cv5.html . test file by opening it in web browser. skip web search"
```

For the 1,200-token workload, `DS4_DSPARK_CONF_THRESHOLD=0.20` was the best
tested sustained setting (49.94 t/s without profiling).
