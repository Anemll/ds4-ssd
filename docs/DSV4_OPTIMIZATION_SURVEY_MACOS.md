# DSv4 Optimization Survey and macOS Mapping

Date: 2026-07-02

This note surveys public DeepSeek V4 / V3-family optimization work from
Huawei/Ascend, SGLang, and vLLM, then maps the useful ideas to the current DS4
macOS Metal/ANE/MPP stack.

The shared ChatGPT link from the old thread did not expose its content through
the text fetcher; it rendered only the login shell. The source base below is
therefore public primary/project documentation plus current local DS4 handoff
docs and code comments.

## Executive Summary

The strongest macOS directions are not broad "make it like vLLM" ports. They
are narrower shape-specific moves:

1. Finish the strict rows-6 verifier fast path: exact 6-row attention plus the
   exact HC tail. This is the clearest strict DSpark path because correctness is
   already solved and the remaining loss is identified as the 6-row attention
   path.
2. Borrow the vLLM/SGLang habit of fusing tiny memory-bound decode work. For
   DS4 this means auditing compressed-attention plumbing, HC/norm/RoPE/cache
   inserts, tiny top-k/index transforms, and verifier tails for avoidable
   command-buffer dispatches and buffer round-trips.
3. Treat MegaMoE/DeepGEMM/DeepEP as design references, not direct ports. The
   local strict route-dedup experiments show duplicate expert reads are already
   often cache-served; a useful macOS MoE attempt must reduce real unique work
   or driver/dispatch overhead while preserving row-exact accumulation order.
4. Use MPP 4.1 specifically for plane-split MXFP4 and other TensorOps matmul2d
   shapes where the data layout already matches. MPP 4.0 is more of a fallback
   for dequant-to-half plus matmul2d; MPP 4.1 is the native scale-plane target.
5. Be careful with "Huawei NPU -> Apple ANE" analogies. Huawei Ascend runs the
   whole inference stack around NPU kernels and HCCL. DS4's private ANE paths are
   currently useful mainly for selected prefill routed MLP shapes. They are not
   a drop-in target for strict DSpark decode unless a separate static, batched,
   quality-preserving ANE draft runner is built and measured.
6. Do not revive rejected speed hacks as optimization "directions": relaxed
   accept, forced-MMA strict, existing ANE shared-expert decode, Case E as a
   default, and smaller resident slot banks are all known bad or non-production
   under the current strict contract.

## Source Snapshot

### DeepSeek V4 Architecture

DeepSeek's V4 model card says the preview series has V4-Pro at 1.6T parameters
with 49B activated and V4-Flash at 284B parameters with 13B activated. Both
support a 1M-token context. The card identifies three architectural upgrades
that matter for runtime: hybrid Compressed Sparse Attention / Heavily
Compressed Attention, manifold-constrained hyper-connections, and mixed
precision with FP4 MoE experts plus FP8 for most other parameters.

For DS4, the important detail is that V4-Flash is fundamentally a sparse MoE
plus compressed attention model. Runtime speed comes from keeping expert
movement and compressed-attention state cheap, not from one large dense GEMM.

### vLLM

vLLM's DeepSeek V4 write-up highlights several implementation ideas:

- It uses `--kv-cache-dtype fp8`, expert parallelism, data parallelism, a fixed
  `--block-size 256`, CUDA graphs, and an FP4 indexer cache in the reference
  V4-Pro/V4-Flash commands.
- DeepSeek V4 attention shares key/value vectors, applies inverse RoPE after
  attention, compresses KV with c4a and c128a layers, keeps a short sliding
  window of 128 local tokens, and uses sparse top-k attention where needed.
- vLLM keeps KV memory packed by using one logical block size, treating
  compressor residual state like sliding-window KV state, and unifying page
  sizes into a few shared block pools.
- vLLM fuses small decode kernels: compressor + RMSNorm + RoPE + cache insert,
  inverse RoPE + FP8 quant, and fused Q norm + KV RoPE + K insert. It also
  overlaps indexer, compression, and sliding-window insertion across streams.
- Its roadmap calls out MegaMoE, paged prefill kernels, FP4 indexer work, fast
  top-k, indexer top-k/page-table fusion, pre-attention low-batch GEMM, norm +
  router fusion, MTP, pipeline parallelism, and KV offload.

macOS mapping: DS4 has no CUDA streams, but it does have command-buffer
boundaries, Metal encoders, pipeline specialization, and unified memory. The
relevant idea is not "add DP/EP"; it is "find every small memory-bound stage in
strict decode and either fuse it or overlap its CPU encode with GPU execution".

### SGLang

SGLang's DeepSeek V3/V3.1/R1 docs list DeepSeek-specific optimizations that
also inform V4:

- MLA weight absorption.
- Optimized MLA backends: FlashAttention3, FlashInfer, FlashMLA, CutlassMLA,
  TRTLLM MLA, and Triton.
- W8A8 FP8 and FP8 KV cache support, including FP8 BMM for MLA.
- CUDA Graph and Torch Compile support for MLA and MoE.
- Chunked prefix cache.
- Data-parallel attention to avoid duplicating KV cache under tensor parallel
  serving; SGLang reports up to 1.9x decoding throughput improvement in the
  right high-throughput setting.
- DeepGEMM for FP8 matrix multiplication.
- MTP/EAGLE speculative decoding, with overlap scheduling as an experimental
  direction.

SGLang's V4 roadmap adds W4A16/MXFP4 support on Hopper, DeepGEMM warmup,
pipeline parallelism, MegaMoE, HiCache, FP4 indexer, FlashMLA sparse prefill,
ragged long-context indexer, DeepEP v2, TileKernels, and more aggressive kernel
fusion.

macOS mapping: the highest-value parts for a single Mac are MTP/DSpark contract
design, exact tiny-batch fusion, FP4/MXFP4 layout work, and long-context cache
state management. DP attention, DeepEP, and multi-node EP are conceptually
useful but not directly applicable to one Apple SoC.

### Huawei / Ascend / MindSpore / vLLM-Ascend

MindSpore 2.6 notes list DeepSeek-V3/R1 BF16 and W8A8 inference support,
operator fusion work such as `RmsNormQuant`, `MatMul+Sigmoid+Add`, and
`Transpose+BatchMatMul+Transpose`, MindIE deployment, vLLM-MindSpore support,
AlltoAllV, and Ascend-only MoE token permute/unpermute inference APIs.

The Huawei Ascend deployment FAQ is more operational, but several points map to
DS4:

- DeepSeek V3/R1 quantized deployment is multi-node and NPU-memory constrained.
- Huawei recommends checking HCCL links, rank table consistency, and identical
  service configs across nodes.
- For long model load stalls it recommends NVMe storage, mmap-style loading,
  parallel/layer-split weight loading, and pre-warming.
- It calls out profiling, AIV enablement, deterministic-compute settings, and
  max sequence / prefill / batch size tuning.

vLLM-Ascend release notes add a newer serving angle: DeepSeek support with DP,
TP, and MTP, graph mode, chunked prefill, automatic prefix caching,
W4A8/W8A8 quantization, dynamic/static EPLB, context parallel work, and async
MTP scheduling.

macOS mapping: Ascend's NPU work is a reminder to make quantized fused ops and
profiling first-class. It is not a direct ANE recipe. Apple's ANE is hard to
schedule beside Metal and has different shape constraints; MPP TensorOps on the
GPU timeline is often the more natural Apple analogue for low-precision matmul.

### Apple MPP / Metal TensorOps

Apple's WWDC26 Metal TensorOps material shows:

- Quantized `MTLTensor` creation for FP8 data.
- Multi-plane tensors with a scales auxiliary plane using UE8M0 scales and
  block factors such as `{32, 1}`.
- Shader-side TensorOps `matmul2d` over tensor slices.
- TensorOps handling dequantization automatically for multi-plane quantized
  matmul.
- Cooperative tensor reductions for QK and softmax-like work.

This matches the local DS4 MXFP4 plan: repack ggml MXFP4 blocks into a
plane-split layout, then use the same data/scales buffers for macOS 26 fallback
and macOS 27 / MSL 4.1 native scale-plane TensorOps.

Important distinction: Apple's "Neural Accelerators in Apple M5 and A19 GPUs"
are GPU-side acceleration exposed through Metal TensorOps. That is not the same
thing as the private Apple Neural Engine interfaces DS4 currently uses for
experimental ANE MLP paths.

## What DS4 Already Has

Current DS4/macOS work already covers more of the external playbook than it may
look like at first glance:

- Metal graph backend for DS4 Flash resident and sidecar modes.
- SSD Flash-MoE sidecar with slot banks, split per-slot buffers, mmap/direct
  sidecar handling, residency instrumentation, and compression/page-cache cliff
  mitigations.
- Native MXFP4 routed expert support and plane-split scaffolding for future
  MPP 4.1 TensorOps.
- Experimental ANE routed MLP / shared expert / projection helpers, currently
  used only where profiles prove they help.
- Strict DSpark speculative decoding with target argmax verification.
- Command-buffer split defaults for the verifier:
  `DS4_DSPARK_VERIFY_SPLIT_LAYERS=4`, which reduced GPU idle time and moved the
  strict path to roughly 44 t/s.
- DSpark rows-6 correctness work: `cmp=0` at n=1000, but slower because the
  exact 6-row attention path falls back to a slow per-row path.
- A large body of rejected approximate paths, which is valuable because it
  prevents chasing attractive but non-byte-safe wins.

## Mapping Table

| External direction | Why it helps externally | DS4/macOS current state | Recommended mapping |
|---|---|---|---|
| V4 c4a/c128a compressed attention and KV packing | Keeps 1M context tractable and avoids cache fragmentation | DS4 has raw/compressed KV and Metal FlashAttention paths, but not necessarily vLLM-style unified page buckets | Audit DS4 compressed-cache allocation and prefix/server state against vLLM's "single logical block" and "compressor state as sliding-window KV" ideas |
| Kernel fusion around compressed attention | Removes HBM round-trips and tiny launches | DS4 has many specialized Metal kernels; strict verifier still spends heavily in attention/tails | First apply to rows-6 exact attention and HC tail; then inspect compressor/norm/RoPE/cache insert boundaries |
| Multi-stream / graph decode | Overlaps independent decode branches and reduces launch overhead | DS4 already gained by splitting command buffers every 4 verifier layers | Continue with Metal command-buffer pipelining and avoid CPU encode idle; do not require CUDA-like streams |
| FP4 / MXFP4 native kernels | V4 ships FP4 experts; Blackwell/Hopper stacks target native low precision | DS4 has native MXFP4 sidecar and plane-split scaffolding | Make MPP 4.1 native MXFP4 an opt-in measured backend; keep macOS 26 dequant-to-half fallback |
| MegaMoE / DeepGEMM / DeepEP | Fuses expert dispatch, grouped GEMM, and communication | Single Mac has no inter-GPU AllToAll, and local route-dedup wrappers were rejected | Study scheduling and wave/fusion structure, but only implement if it reduces exact unique work or dispatch/binding cost |
| DP attention / EP | Avoids KV duplication and improves high-throughput serving | Not directly relevant to one Apple SoC | Useful only for future multi-Mac or multi-GPU CUDA work, not the local DS4 path |
| MTP / EAGLE speculative decoding | Speeds decode by verifying multiple drafted tokens | DS4 has DSpark strict verifier and trained draft package | Continue strict DSpark; consider retraining draft for larger/rows6 block economics rather than loosening acceptance |
| Chunked prefix cache / prefix caching | Higher throughput and lower TTFT on repeated prefixes | DS4 has server and KV/session work, but DSpark focus is decode | Consider for server agent workflows after strict decode stabilizes |
| Huawei W8A8 and fused ops | Fits NPU memory and reduces graph stages | DS4 has Q8/int8/ANE paths plus MXFP4/FP8 conversion work | Use as validation that fused quantized norm/router/MLP ops are high ROI; map to Metal/MPP first, ANE only for static batched shapes |
| Huawei mmap/parallel weight loading | Reduces huge model startup stalls | DS4 sidecar already uses mmap/direct read paths and slot banks | Keep improving cold-start preload and per-layer parallel loading; this is practical and low-risk |
| Ascend graph mode / ACLGraph | Cuts dispatch overhead | DS4 equivalent is Metal command-buffer batching, ICBs, graph-like resident resources | Continue profiling encode vs GPU execution; avoid per-token driver work proportional to bank size |
| Apple ANE | Separate accelerator for selected ML graphs | Existing DS4 ANE shared-expert decode regressed badly in DSpark; production ANE is prefill-oriented | Treat as experimental for draft overlap only if static, int8, batched, and independent of strict target state |

## Recommended Improvement Directions

### P0: Strict Rows-6 Fast Attention

Rows-6 is the cleanest near-term strict win. The handoff already says rows-6 is
correct (`cmp=0`) but slow because it uses a slow exact 6-row attention path.
This directly mirrors the public vLLM/SGLang lesson: tiny memory-bound decode
kernels must be shape-specialized and fused.

Concrete work:

- Extend the existing rows5 batch attention kernels to N=6 without changing
  row-local order or sparse/index behavior.
- Extend or replace the exact HC tail so 6 rows do not drop to the slow tail.
- Keep `DS4_DSPARK_VERIFY_SPLIT_LAYERS=4` and compare against the strict n=1000
  Pygame baseline.
- Rank by end-to-end generation t/s plus `cmp=0`, not by stage-local time.

Expected value: local handoff estimates about 48-50 t/s if the measured rows6
tau survives and attention returns to the fast batch path.

### P1: V4-Style Attention State Packing Audit

vLLM's biggest V4 implementation lesson is not the CUDA code itself; it is that
heterogeneous compressed attention becomes a memory allocator problem. DS4 has
raw/compact KV and sidecar memory machinery, so a review pass should ask:

- Are raw KV, compressed KV, indexer/cache metadata, and compressor residuals
  allocated in a way that causes avoidable fragmentation or per-kind branching?
- Are prefix/server checkpoint boundaries aligned to cache block boundaries?
- Can compressor residual state be represented like a small sliding-window KV
  state instead of a separate special path?
- Are DS4's long-context exact attention paths still sparse where they should
  be, especially when experimenting with NAX/MPP alternatives?

This is mainly a long-context/server direction, but it prevents building local
optimizations that fail at 64k+ context.

### P1: Exact Fusion Around Attention and Verifier Tails

vLLM reports large speedups from fusing elementwise stages around compression,
RoPE, cache insertion, inverse RoPE, and quantization. The DS4 equivalent is a
kernel-boundary audit of the strict verifier:

- Q/K/V projection normalization and RoPE staging.
- Raw/compressed KV insertion and row gather transforms.
- Indexer top-k and page/range transform buffers.
- HC split/weighted-sum/norm tails.
- Per-row output-low and logits path for verify.

Rule: only promote a fusion if it improves end-to-end strict generation and
does not perturb byte identity. Existing Case E and forced-MMA results show
that "close enough" attention math is not acceptable under the current contract.

### P1: MPP 4.1 Native MXFP4 Backend

This is the most concrete MPP item.

Use MPP 4.1 when:

- The weight layout is plane-split FP4 data plus UE8M0/E8M0 scale plane.
- The shape is a matmul2d-friendly resident path, not a one-off tiny scalar
  decode path with more setup than compute.
- The backend can be gated by a real runtime probe and benchmarked against the
  macOS 26 dequant-to-half fallback.

Use MPP 4.0 / macOS 26 style fallback when:

- Native scale-plane TensorOps are unavailable.
- Dequant-to-half plus half matmul2d is faster and simpler than manual
  in-register scaled matmul.
- The path is prefill or batched enough for matmul2d setup to pay.

Avoid:

- Repacking every token. Repack once into plane-split resident buffers.
- Treating MPP 4.1 support as proof of speed. Keep the current `default-off`
  posture until measured on the exact M5/M5 Max target.

### P2: MoE Fusion Inspired by MegaMoE, Not Existing Route Dedup

External systems are investing in MegaMoE, DeepGEMM, DeepEP, and TileKernels
because MoE is memory movement plus scheduling. For DS4 strict mode, the local
lesson is subtler:

- The existing route-dedup/grouped wrappers did not improve strict throughput.
- Duplicate expert reads can already be L2/SLC served.
- Forced row0 route probes overstated possible savings by shrinking the unique
  working set, not by proving a real duplicate-read win.

A worthwhile macOS MoE experiment should therefore target:

- Fewer dispatches and less descriptor/binding overhead.
- Better in-kernel wave scheduling for top-6 experts.
- Exact ordered accumulation.
- A measured reduction in real unique memory traffic or driver overhead.

Do not re-add the rejected grouped-IQ2 wrapper family as-is.

### P2: ANE Only for Independent Draft/Prefill Work

Huawei's NPU stack should not be mapped one-for-one to Apple ANE. On Ascend,
MindIE/MindSpore/vLLM-Ascend own the NPU graph and collective runtime. In DS4,
Metal owns the main decode path, and ANE calls are separate private-interface
work with IOSurface and scheduling overhead.

Use ANE when:

- The shape is static and batched enough.
- The quantization is quality-safe for the target path.
- It can run concurrently with the GPU path without blocking target-state
  verification.
- It keeps data in ANE/IOSurface form without excessive CPU conversion.

Do not use ANE when:

- It replaces strict target math with approximate math.
- It serializes with the GPU verifier.
- It is the existing DSpark shared-expert decode path, which is already a known
  severe regression.

The most plausible ANE research path is still an independent DSpark draft
overlap runner, but local profiling says much of the draft graph is attention,
RoPE, router softmax, and MXFP4 routed MoE, not just ANE-friendly dense MLPs.
So treat this as experimental, not the main strict plan.

### P2: MTP/DSpark Contract Improvements

SGLang and vLLM both keep investing in MTP, acceptance length checks, and
speculative scheduling. DS4 already has the local version of this idea:
DSpark.

The next productive DSpark direction is not relaxed acceptance. It is one of:

- Retrain or re-export the draft package for a true rows6/block6 contract.
- Improve first-position acceptance without approximate target state.
- Build a verifier that has a better exact arithmetic shape than today's Mode A.
- Add better acceptance-length diagnostics and A/B harnesses for agent prompts.

Acceptance-only shortcuts have already failed quality or state coherence.

## Explicit Non-Directions

These should stay closed unless the product contract changes:

- `--draft-fast-relaxed` as production default.
- `DS4_DSPARK_RELAXED_ACCEPT` / trust-confidence / force-accept paths.
- Forced-MMA strict as a byte-identical path.
- Case E NAX attention as default strict attention.
- Smaller sidecar slot banks for DSpark speed.
- Existing synchronous ANE shared-expert decode in the DSpark verifier stack.
- Existing grouped-IQ2 route-dedup wrappers.

## Proposed Work Plan

1. Land or isolate the rows6 exact fast attention/HC-tail path.
   - Gate: `cmp=0`, n=1000 and n=2500, strict Pygame prompt.
   - Expected: move from about 44 t/s toward 48-50 t/s if rows6 economics hold.
2. Add a verifier kernel-boundary profile that reports dispatch count, command
   buffer count, and buffer round-trips per block.
   - Goal: identify the DS4 equivalent of vLLM's "small memory-bound kernels".
3. Build a small MPP 4.1 MXFP4 microbench matrix.
   - Shapes: routed gate/up/down resident shapes, DSpark draft dense shapes,
     rows5/rows6 verifier dense shapes.
   - Compare: existing Metal ALU, dequant-to-half MPP fallback, native
     scale-plane MPP 4.1.
4. Review compressed KV/cache block layout for long-context server use.
   - Compare against vLLM's fixed 256-token logical block and shared page-size
     buckets.
5. Only after the above, revisit MoE fusion with a new design target:
   dispatch/binding reduction and exact in-kernel scheduling, not duplicate
   route dedup by itself.
6. Keep ANE as a separate research branch for draft-overlap sizing. It should
   not block strict rows6 or MPP work.

## Source Links

- DeepSeek V4 model card:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- vLLM DeepSeek V4 implementation blog:
  https://vllm.ai/blog/2026-04-24-deepseek-v4
- vLLM DeepSeek V4 roadmap:
  https://github.com/vllm-project/vllm/issues/40902
- vLLM DeepSeek V3/R1 recipe:
  https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html
- SGLang DeepSeek V3/V3.1/R1 docs:
  https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md
- SGLang DeepSeek V4 roadmap:
  https://github.com/sgl-project/sglang/issues/23602
- MindSpore 2.6 release notes:
  https://www.mindspore.cn/docs/en/r2.6.0/RELEASE.html
- Huawei Ascend DeepSeek deployment FAQ:
  https://www.hiascend.com/developer/techArticles/20250224-2
- vLLM-Ascend release notes:
  https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html
- vLLM-MindSpore DeepSeek R1 parallel inference:
  https://www.mindspore.cn/vllm_mindspore/docs/en/r0.3.0/getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.html
- Apple WWDC26 Metal TensorOps:
  https://developer.apple.com/videos/play/wwdc2026/330/

## Local Context Used

- `docs/DSPARK_VERIFIER_HANDOFF.md`
- `docs/dspark_verifier_optimization_plan.md`
- `docs/CASE_E_HANDOFF.md`
- `docs/flash-moe-compression-handoff.md`
- `docs/mxfp4-native-sidecar-plan.md`
- `docs/mxfp4-handoff.md`
- `docs/ANE_KERNELS.md`
- `docs/PERFORMANCE.md`
- `docs/SIDECAR.md`
- `ds4_gpu.h`, `dspark.c`, `ds4.c`, `ds4_metal.m`, `metal/*.metal` by targeted search
