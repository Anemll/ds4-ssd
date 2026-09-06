# HY4 native SSD implementation and Flash-MoE correctness port

Source: [Anemll/anemll-flash-llama.cpp@34cccef1bc8db4f93048bff4033638024cbd8f34](https://github.com/Anemll/anemll-flash-llama.cpp/commit/34cccef1bc8db4f93048bff4033638024cbd8f34),
branch `HY4-1.25-bit`, including its existing HYV4 implementation.
Target base: `Anemll/ds4-ssd` `hy3` at `6aa38025f16435a29e14d5e2d75ed063bd28ffc1`.

## Mapping

| Source change | DS4 equivalent and port |
| --- | --- |
| `flash_moe_protect_request_slots()` before `reserve_expert_slot()` | `ds4_flash_moe_protect_request_slots()` and `ds4_flash_moe_resolve_request_slots()` in `ssd/ssd_flash_moe_slots.h` validate the whole request and reciprocal ownership before hard-reserving every resident hit. `metal_graph_flash_moe_pick_slot()` delegates to the shared selector, retaining soft prefetch hints but never choosing a hard-reserved slot. |
| Request-local duplicate tracking; install only after all slots are resolved | `metal_graph_flash_moe_prepare_decode_prefetch_ids()`, `metal_graph_flash_moe_decode_async_handout()`, `metal_graph_flash_moe_prepare_decode()`, and `metal_graph_flash_moe_run_tiny_batch_slotbank()` use `metal_graph_flash_moe_resolve_request()` / `metal_graph_flash_moe_install_request()` before writes. Matching hits may already be reserved; repeated IDs reuse their chosen slot. Capacity/validation failure cannot partially install a request. |
| Assert expert/slot ownership; invalidate destinations after a failed direct read | DS4 reserve, commit, and invalidate helpers maintain reciprocal mappings. Failed or discarded direct-to-slot jobs invalidate destinations and replay state after joining, before the bank is reused. |
| `finish_shared_io()` at callback stop, graph exit, and runtime destruction | `metal_graph_flash_moe_decode_prefetch_finish()` / `_cleanup()` and `metal_graph_flash_moe_async_load_cleanup()` join outstanding jobs and invalidate abandoned direct writes. `ds4_flash_prefill_async_cancel_and_drain()` is called at failed DeDup layers, `metal_graph_prefill_layer_major()` / `metal_graph_prefill_chunked_range()` boundaries and `metal_graph_reset_prefill_state()`, with existing destructor joins retained. Scratch-read cancellation uses atomic loads/stores. DS4 uses pthread workers rather than futures. |
| Avoid invoking the downstream evaluation callback twice | Audited: DS4 has no llama scheduler `ask`/evaluation callback wrapper. Its progress/display callbacks report distinct events and return void. No duplicate-evaluation equivalent found. Native HY4 adds an optional session cancellation callback, polled before each token; progress fires once per completed prefill token. |
| Slot reservation regression and HY4 lifecycle tests | Model-free tests exercise production slot/reader helpers, including the HY4 host slot-ID recorder. HY4 math, mixed-quant Metal, tokenizer/chat, and native session lifecycle tests exercise the supplied model package. |

## Prefill and shared-bank semantics

Decode and tiny-batch prefill consume several resident slots together, so their
whole request needs hard protection. Larger expert-major DeDup prefill consumes
one expert at a time and synchronizes queued GPU work before mutating the shared
cache. Its cache-population hints remain soft: pinning the whole prompt's expert
union would incorrectly fail when that union exceeds bank capacity.

Hard request protection is unconditional. `DS4_FLASH_MOE_PREPROTECT_TOPK` remains
an optional soft eviction hint; correctness does not depend on it. Existing DS4 profile defaults and sidecar bytes are unchanged. Native HY4
adds its own model/session path and quantized Metal kernels; its initial
implementation keeps token-wise prefill and requires native top-8.

## Native HY4 implementation from the supplied branch

The task was expanded to include the actual `hyv4` model implementation present
on `HY4-1.25-bit` at `34cccef`, in addition to that commit's slot/drain fixes.
These are native DS4 session operations; no llama.cpp subprocess backend is used.

| Source implementation | Native target |
| --- | --- |
| HYV4 metadata and model tensors | `hy4/hy4_model.c`, HY4 shape/model dispatch in `ds4.c`; exact 78-layer, four-stream, 256-expert/top-8 validation; supplied sidecar layout unchanged. |
| STQ1_0 and mixed IQ2_XXS/IQ3_XXS/IQ4_XS routed matvec | `hy4/hy4_quants.c`, `metal/hy4.metal`, `ds4_gpu_hy4_quant_matvec_tensor()`; tests compare source dequantizers and actual sidecar samples. |
| `src/models/hyv4.cpp` independent HC and output head | `hy4/hy4_math.h` and `hy4/hy4_runtime.c`: flattened RMS mixing, separate pre/post gates, ordered F32 residual multiply/add, head reduction. No DS4 Sinkhorn matrix. |
| Gated MLA and attention sinks | Reuse the matching GLM absorbed-MLA projections/cache storage and source-compatible Q4 embedding decoder; HY4 supplies sink-aware attention and sigmoid gating before output projection. |
| HYV4 router/shared and routed FFN | Native sigmoid/bias selection; selected unbiased weights normalized and scaled by 2.827; routed SwiGLU clamp10; weight experts after down projection; dense/shared FFN unclamped. |
| Attention sigmoid and ordered post-down expert sum | `ds4_gpu_hy4_sigmoid_mul_tensor()` and `ds4_gpu_hy4_weighted_sum8_tensor()` keep both operations in the native command batch; the sum preserves separate F32 multiply/add in expert order. `DS4_HY4_CPU_POINTWISE=1` retains the initial scalar reference. |
| HYV4 chat template and tokenizer | HY4 framing and thinking/tool tokens in `ds4.c` and `ds4_agent.c`; vocabulary-only source/Jinja parity fixtures and streamed tool-parser regressions. |
| Runtime lifecycle | Native session create/sync/eval/reset/rewind/payload save/load; synchronous complete-request SSD installation for both initial/resumed prefill and decode; cancellation at completed-token boundaries. |

## Exclusions and boundaries

- Source `34cccef` has no native HY4 DSA or MTP. This port rejects contexts above
  2048 rather than presenting full attention as million-token HY4 support.
- Shared-FFN/SSD overlap defaults and fused slot8/iHC broadcast performance paths
  are not copied wholesale. Correctness is checked before kernel optimization.
- No ANE-INT8, M5 ALU/fusion tree, antirez catch-up, or other branch merge.
- Existing DS4/HY3/GLM kernels and profiles retain their behavior. HY4 uses the
  current sidecar format and explicit dense/sidecar paths.

## Validation and risk

See [HY4.md](docs/HY4.md) for exact model paths, conservative commands, and
current validation. The earlier DS4 six-slot smoke validates the shared cache
port only; it is not HY4 model evidence.

Slot protection can change eviction choices. Failed/interrupted direct reads
may discard destinations and require an SSD reread. HY4 prefill is initially
token-wise; only bounded warm-cache gate/reduction timing is reported in
`docs/HY4.md`. No cold-SSD or long-context result is claimed.
Native Metal validation targets the actual STQ1_0 package on Apple M5 Max.
Other HY4 artifacts, CUDA execution, and HY4 MTP are not supported by this path.
