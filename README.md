# ds4-ssd

`ds4-ssd` is an alpha fork of antirez's DwarfStar 4 (`ds4`) inference engine
for DeepSeek V4 Flash. The fork keeps the narrow, self-contained DS4 runtime and
adds an SSD-streamed routed-MoE sidecar path for Apple Silicon systems where a
fully resident model is not practical.

The main alpha feature is SSD streaming: dense tensors stay in a normal GGUF,
while routed experts live in a sidecar directory and are paged through a
slot-bank cache. Resident full-GGUF mode is still supported for high-memory
machines. Apple Silicon optimizations include NAX, the Apple neural-accelerator
backed `matmul2d` path used by Metal on M5-class hardware, plus Apple Neural
Engine routed-MLP prefill paths where the measured profile says ANE wins.

This branch is intentionally narrower than the research branch. It keeps the
runtime, Metal shaders, GGUF tools, correctness tests, sidecar smoke, and core
docs, while dropping profiling scripts, handoff notes, session exports, and
bench-only ANE probes from the public alpha tree.

## Status

Alpha means:

- SSD sidecar mode is the release headline.
- Resident full-GGUF mode remains available.
- Correctness vectors and an executable 16K sidecar smoke are the current test
  bar.
- Broader mode coverage, CI, and performance regression automation are planned
  for the next stage.

The in-repo `gguf-tools/deepseek4-quantize` tool builds resident GGUFs. It does
not yet emit the sidecar layout. For this alpha, use the prebuilt sidecar
package at
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major).

## Build

On macOS:

```sh
make
```

This builds:

- `./ds4`: CLI runner.
- `./ds4-server`: OpenAI/Anthropic/Responses-compatible local server.
- `./ds4-bench`: throughput sweeps.
- `./ds4-eval`: evaluation helper.
- `./ds4-agent`: local coding-agent frontend.

`metal/` is a required build input. Do not prune it.

CUDA sources are inherited from upstream DS4 and kept in tree, but the alpha
validation focus is Apple Silicon SSD streaming.

## Run SSD Sidecar Mode

Download the prebuilt sidecar package:

```sh
./download_model.sh sidecar
```

Or the native MXFP4 package (bit-exact MXFP4 routed experts, ~156 GB,
[anemll/DSv4-Flash-MXFP4-native-flash](https://huggingface.co/anemll/DSv4-Flash-MXFP4-native-flash)):

```sh
./download_model.sh mxfp4
./ds4 -m models/DSv4-Flash-MXFP4-native-flash --ssd-cache auto -p "Hello"
```

Sidecar manifests that contain `MXFP4_NATIVE` storage automatically default
`DS4_MXFP4_NATIVE=1` when the variable is unset. Explicitly setting
`DS4_MXFP4_NATIVE=0` keeps the guard enabled and will reject native MXFP4
sidecars.

`--ssd-cache` sizes the resident expert slot bank (`auto`, or an explicit value
like `32GB`). Any size is safe: on RAM-limited machines the bank is clamped so
prefill cannot overflow memory and auto-shrinks after prefill so decode-miss
reads stay served by the OS file cache.

Then set `DS4_SIDECAR_DIR` to the sidecar package root containing
`manifest.json` and `dense/model-dense.gguf`:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
```

Run the package root directly. DS4 detects the dense GGUF and sidecar metadata;
no explicit `--moe-sidecar` or `--moe-mode` flag is needed:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 8 \
  --ctx 8192 \
  -p "Hello"
```

`--ctx 8192` is the KV window in this conservative first-run example. Leave the
Metal raw-KV cap automatic so it follows the prefill chunk size and server
checkpoint frontiers stay aligned.

Start with `--moe-slot-bank 8` and raise it once you confirm there is headroom.
The slot bank is the cap on resident routed-expert slots, so a larger value
trades RAM for fewer SSD reads. `--moe-slot-bank 64 --ctx 32768` is a
high-memory setting, not the safest default.

For cache-budget comparisons, especially against upstream SSD-streaming runs,
use `--ssd-cache` instead of manually choosing a slot count. Explicit sizes set
the target routed-expert slot-bank budget, while `auto` sizes the slot bank from
currently available memory after dense weights and context buffers are
estimated:

```sh
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache 32G --ctx 32768 -p "Hello"
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache 64G --ctx 32768 -p "Hello"
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache auto --ctx 32768 -p "Hello"
```

Run the committed sidecar smoke:

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

To confirm SSD streaming is active, look for these startup lines:

```text
applied sidecar tuning profile
Flash-MoE sidecar loaded
Flash-MoE slot banks allocated
```

If `-m` points at a directory containing `manifest.json` and
`dense/model-dense.gguf`, DS4 auto-detects SSD sidecar mode, rewrites the model
path to the dense GGUF, and enables sidecar slot-bank mode internally. If you
pass only `-m /path/to/full-model.gguf`, DS4 is in resident/full-GGUF mode.

See [docs/SIDECAR.md](docs/SIDECAR.md). For the external expert-sidecar
export wrapper, see [docs/SIDECAR_EXPORT.md](docs/SIDECAR_EXPORT.md); the
prebuilt Hugging Face sidecar remains the turnkey low-RAM package.

Machine-specific defaults for M5, M5 Max, M3 Ultra, and M1 Max are selected
from `ds4_profile.json`. Profiles set defaults only; exported environment
variables still win. ANE compute paths are disabled by default: the async
ANE i8 prefill arm is lower precision than the GPU arms and its ANE/GPU work
split is queue-timing dependent, so ANE-computed prefills are not reproducible
run to run (this also poisoned persisted KV caches such as the agent's
`sysprompt.kv`). Pass `--ane` (or export `DS4_ANE=1`) to re-enable the
profile's ANE defaults until the ANE precision work lands; profiles choose ANE
only for chunk shapes where it has measured faster than GPU or NAX on that
machine. See
[docs/PROFILES.md](docs/PROFILES.md) and
[docs/STREAMING_KNOBS.md](docs/STREAMING_KNOBS.md).

## Run Resident Sidecar Mode

On high-memory Apple Silicon systems, a sidecar package can also be loaded as a
fully resident all-expert slot bank. This keeps the sidecar package layout
(`manifest.json` plus `dense/model-dense.gguf`) but avoids decode-time SSD
expert misses.

Use `--resident` with the sidecar package directory:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --ctx 8192 \
  -p "Hello"
```

`--resident` autodetects the dense GGUF, enables sidecar slot-bank mode,
defaults the slot bank to all experts, preloads and touches the resident bank,
and disables direct-mmap auto selection. If you explicitly pass
`--moe-slot-bank`, that value is honored.

The same simplified startup is supported by the local server:

```sh
./ds4-server \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --ctx 32768 \
  --host 127.0.0.1 \
  --port 8000
```

The OpenAI-compatible server advertises `deepseek-v4-flash` from
`GET /v1/models`:

```sh
curl http://127.0.0.1:8000/v1/models
```

Use that id in API calls unless you intentionally want a compatibility alias:
`deepseek-chat` disables thinking and `deepseek-reasoner` enables thinking.

## Run HY3

HY3 full-GGUF models are detected from `general.architecture=hy_v3` and use a
DS4-native Metal runtime. The implementation keeps all mapped GGUF tensors on
the existing DS4 quantized-matmul path. It executes each token's selected top-8
routes together, fuses gate/up/SwiGLU work, and uses a direct top-8 IQ3_XXS
down-and-sum kernel. Attention defaults to a direct head-major F16 NAX-half KV
cache on supported Metal systems. The lower-memory Q8_0 split-KV path remains
available with `--hy3-q8`.

Recommended interactive agent command on M5 Max:

```sh
./ds4-agent \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --ctx 16000 \
  -sys '' \
  --nothink \
  --temp 0
```

Use the same model with the lower-memory Q8_0 KV cache:

```sh
./ds4-agent \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --hy3-q8 \
  --ctx 16000 \
  -sys '' \
  --nothink \
  --temp 0
```

For one-shot CLI generation:

```sh
./ds4 \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --ctx 5000 \
  -sys '' \
  --nothink \
  --temp 0 \
  -p "Make a game of Space invaders in C++"
```

The current HY3 path is Metal-only. It supports normal CLI and interactive
generation, uses the model's Hunyuan tokenizer/chat framing, and intentionally
does not use llama.cpp-specific CLI flags. `-sys ''` disables DS4's default
`You are a helpful assistant` system message and can be omitted when that
default is wanted. Prompt ingestion uses a 256-token layer-major Metal path and
only computes vocabulary logits for the final prompt row. Set
`DS4_HY3_DISABLE_BATCH_PREFILL=1` to use the diagnostic token path, or use
`--quality` to select the conservative one-token prefill and serial-attention
fallbacks. `--moe-mode stock` is an anemll-flash-llama.cpp option and is neither
needed nor accepted by DS4's full-GGUF HY3 path. `ds4-agent` persists HY3's
active Q8_0 or F16 KV state in its normal `~/.ds4/kvcache/sysprompt.kv`, so
subsequent launches restore a matching system/tool prompt instead of
prefilling it again.

For long interactive sessions, keep a larger physical KV allocation while
compacting the normal transcript earlier:

```sh
./ds4-agent \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --ctx 32000 \
  --working-context 12000 \
  -sys '' --nothink --temp 0
```

`--working-context` must be at least 4096 and smaller than `--ctx`. It starts
normal compaction at that frontier, while the full physical context remains
available for a large tool result and the private summary exchange. The footer
shows both values (for example `ctx 7.5k/32k work:12k`), and compaction rebuilds
strictly below the working frontier so the next boundary cannot immediately
compact the same transcript again.

`ds4-agent` renders and parses HY3's native Hunyuan tool-control tokens
(`tool_calls:opensource`, `tool_call:opensource`, `tool_sep:opensource`, and
the matching argument/response tokens). It does not prompt HY3 to imitate
DSML or GLM XML. This matters for file-writing prompts: mixed protocols can
look like model-quality or KV-precision corruption even when the sampled
content itself is sound.

During live `write`, `edit`, and `bash` arguments, `ds4-agent` also stops an
exact short byte pattern that repeats for at least 512 bytes (for example a
literal `, 50, 50, ...` decode loop). It preserves the incomplete `.part`
preview, leaves the destination untouched, discards the bad assistant turn,
and retries once with compact-payload framing. Set
`DS4_AGENT_PAYLOAD_REPEAT_GUARD=0` only when intentionally generating a long
literal repetition that cannot be represented with a loop or `repeat`/`fill`.

HY3 demand-pages GGUF tensors through per-tensor Metal views and drains decode
every 16 layers by default. Decode keeps at most two Metal command buffers in
flight, waiting and retiring only the oldest split while the next one is
encoded; the final synchronize drains the remaining work and releases its
transients normally. `DS4_HY3_DECODE_SPLIT_LAYERS=N` changes that interval.
For A/B diagnostics, `DS4_HY3_DECODE_BLOCKING_FLUSH=1` forces blocking depth 1,
while `DS4_HY3_DECODE_BLOCKING_FLUSH=0` (or
`DS4_HY3_DECODE_UNBOUNDED_FLUSH=1`) restores unbounded submission.

When `--hy3-q8` is active, decode automatically switches to a four-SIMDgroup
Q8_0 split-KV kernel at 2,048 context tokens. `DS4_HY3_ATTN_SG4_MIN_CTX=N`
changes that threshold, and `DS4_HY3_DISABLE_ATTN_SG4=1` keeps the
one-SIMDgroup diagnostic path. At that threshold, HY3 also defaults to an exact
fused GQA8 kernel: one workgroup dequantizes each shared K/V tile once for all
eight query heads while retaining the existing Q8 cache and reducer layout.
Set `DS4_HY3_Q8_FUSED_GQA8=0` to restore the prior SG4 kernel. On Apple M5 Max,
balanced full-model runs improved from 19.835 to 20.727 t/s at 8k context
(+4.50%) and from 15.589 to 16.732 t/s at 16k (+7.33%), with identical greedy
continuations.

HY3 defaults to direct NAX-half attention on supported Metal systems. This
allocates a persistent head-major F16 K/V cache and writes each new token
directly into it; NAX consumes that cache in place, so there is no context
conversion pass. Use `--hy3-q8` to select the lower-memory Q8_0 KV fallback.
`DS4_HY3_NAX_HALF_ATTN=0` remains a backward-compatible low-level equivalent
for existing benchmark scripts. NAX-half uses about 4.88 GiB of K/V at 16k
context for the 80-layer HY3 model, versus about 2.59 GiB for Q8_0. Within the
NAX-half path, prefill groups two adjacent tokens into one 16-query-row MPP tile
by default so they share the same K/V history walk. Set
`DS4_HY3_NAX_GROUPED_PREFILL=0` to select the one-token diagnostic kernel, or
set it to `1` to select the grouped path when its pipeline is available. If
that variable is unset, the presence of
`DS4_HY3_DISABLE_NAX_GROUPED_PREFILL` also disables grouping. HY3 session
snapshots support both Q8_0 and F16 cache layouts, record the active layout,
and restore compact live rows without serializing F16 padding.

NAX-half decode uses an exact fast-tile specialization by default. Complete
32-row tiles skip a no-op mask pass and synchronization barrier, and V-cache
staging uses aligned F16 vectors; softmax and P×V arithmetic are unchanged.
Set `DS4_HY3_NAX_FAST_TILE=0` for the original diagnostic kernel. On Apple M5
Max, a 256-token greedy run improved from 21.81 to 24.19 t/s at 8k context
(+10.9%) and from 18.21 to 20.96 t/s at 16k (+15.1%), with identical generated
token hashes at both frontiers.

HY3 keeps the generic direct-RHS Q8 NAX matrix path disabled for its Q/K/V
projections. On HY3 that path changes greedy logits at its 32-token dispatch
boundary and creates a slow final prefill chunk; the standard simdgroup Q8
projection is both faster and stable. `DS4_HY3_ENABLE_DENSE_NAX=1` is an
audit-only override. It is independent of `DS4_HY3_NAX_HALF_ATTN`, which
selects the F16 NAX attention/KV path by default.

HY3 prompt prefill is layer-major and uses 256-token chunks by default. Set
`DS4_HY3_PREFILL_CHUNK=32`, `64`, `128`, or `256` to benchmark a fixed chunk
size; values above 256 are clamped because the attention and selected-expert
workspaces grow linearly with the chunk. `--quality`,
`DS4_HY3_DISABLE_BATCH_PREFILL`, or `DS4_HY3_DISABLE_FLASH_ATTN` disables this
batched path. Chunks of at least 128 tokens also use a true batched F32 router
projection; `DS4_HY3_F32_ROUTER_MM_MIN_TOKENS=N` changes that crossover and
`DS4_HY3_DISABLE_F32_ROUTER_MM=1` restores the per-token diagnostic path.

### HY3 NextN/MTP sidecar

HY3 stores its optional one-layer NextN predictor as block 80. A full
MTP-bearing GGUF can be used directly as the MTP support model, or reduced to a
2.16 GiB support sidecar while the normal block-0-through-79 GGUF remains the
target:

```sh
export HY3_FULL_MTP_GGUF="$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M-mtp.gguf"
export HY3_MTP_GGUF="$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M-mtp-only.gguf"

python3 scripts/export_hy3_mtp_sidecar.py --dry-run "$HY3_FULL_MTP_GGUF"
python3 scripts/export_hy3_mtp_sidecar.py \
  "$HY3_FULL_MTP_GGUF" "$HY3_MTP_GGUF"
```

The exporter streams tensor payloads without materializing the model and
fails closed unless the source has the exact 20 block-80 tensors and required
`hy_v3` metadata. Run the target plus the exported support model with greedy
decoding and a two-token draft:

```sh
./ds4-agent \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --mtp "$HY3_MTP_GGUF" \
  --mtp-draft 2 \
  --ctx 5000 \
  -sys '' \
  --nothink \
  --temp 0
```

HY3 only activates block 80 when `--mtp-draft` is greater than one and
speculative greedy decoding is available. The default `--mtp-draft 1`,
`DS4_MTP_SPEC_DISABLE=1`, and nonzero `--temp` keep MTP inactive so prompt and
decode do not maintain an unused predictor cache.

The block-80 implementation keeps its own KV cache, keeps target hidden state
on Metal, and uses the same fused selected top-8 GGUF MoE path as the target.
The forced one-token reference below strictly verifies greedy drafts. The
fingerprinted HY3 payload persists target KV, block-80 KV, and the
target-hidden carry in `ds4-agent`'s system-prompt cache, and rejects reuse
after either GGUF file is replaced. After the default one-way target fallback,
save/resume writes a target-only HY3 payload and keeps MTP disabled for that
restored session. A new session builds block-80 state only when reference MTP
or the explicitly unsafe batch experiment is selected.

On M5 Max, the exact one-token verifier is a correctness reference rather than
a speedup: it still evaluates one complete target row per emitted token and
adds the block-80 work. DS4 therefore defaults to a session-local one-way plain
target route before prompt sync, which preserves target output and throughput
without building an unused predictor cache. Force the exact reference MTP path
for acceptance and timing measurements with:

```sh
DS4_HY3_MTP_AUTO_FALLBACK=0 \
DS4_MTP_TIMING=1 \
DS4_MTP_SPEC_LOG=1 \
./ds4-agent \
  -m "$HOME/Models/Hy3-GGUF_1b/Hy3-IQ1_M.gguf" \
  --mtp "$HY3_MTP_GGUF" \
  --mtp-draft 2 \
  --ctx 5000 \
  -sys '' \
  --nothink \
  --temp 0
```

A paired M5 Max coding-prompt run (`--ctx 2048`, 256 generated tokens) was
byte-identical to target-only while accepting 156/186 draft tokens (83.9%).
Target-only measured 27.65 t/s; forced reference MTP measured 26.43 t/s. The
high acceptance but lower throughput is the reason the measured default is
target-only until an invariant verifier can retire more than one target row
per pass.

`DS4_HY3_MTP_BATCH_VERIFY=1` requests the carried, layer-major verifier
experiment, but fails closed to the safe route. Entering the experiment also
requires `DS4_HY3_MTP_UNSAFE_BATCH_VERIFY=1`. It folds the first predicted
token into the verifier batch and uses tiny fused expert-pair kernels, but is
not a target-greedy mode: the current M5 Max run produced 11.97 t/s versus
27.46 t/s target-only, and the tested carried target state diverged from
one-token decode. Its acceptance counters include the folded first-token
prediction, so they measure predictor-row accuracy rather than extra emitted
suffix yield. Always compare its output byte-for-byte with the target-only run;
never use it for quality or production measurements.
`DS4_HY3_MTP_PROFILE=1` adds scoped MoE-stage diagnostics. HY3 uses the F16 NAX
KV layout by default; add `--hy3-q8` for a Q8_0 comparison. No model conversion
is required.

## Run MTP With A Sidecar

MTP speculative decoding is optional. It uses the normal sidecar or resident
sidecar model as the target, plus a small support GGUF that drafts candidate
tokens. Download the support model first:

```sh
./download_model.sh mtp
export DS4_MTP_GGUF="$PWD/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
```

For a basic sidecar MTP smoke test, keep the draft length at 2 and use greedy
decoding:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --mtp "$DS4_MTP_GGUF" \
  --mtp-draft 2 \
  --mtp-margin 0 \
  --temp 0 \
  --nothink \
  -n 128 \
  -p "Write a short Python function that parses a CSV line with quoted fields."
```

Expected startup logs include:

```text
MTP support model loaded
MTP sidecar verifier
```

Expected summary output includes an acceptance line when MTP ran:

```text
ds4: mtp acceptance: 86.3% (1740/2016 draft tokens)
```

If the acceptance line is missing, MTP did not actually draft or verify tokens.
Check that `--mtp "$DS4_MTP_GGUF"` was passed and that `--mtp-draft` is greater
than 1.

For fully resident sidecar runs, enable the sidecar batch verifier. This is the
path to test MTP with the all-expert resident slot bank and without SSD
decode-miss I/O:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_MTP_SIDECAR_BATCH_VERIFY=1 \
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --mtp "$DS4_MTP_GGUF" \
  --mtp-draft 2 \
  --mtp-margin 0 \
  --temp 0 \
  --nothink \
  -n 256 \
  -p "Hello"
```

For full native MXFP4 sidecars, set `DS4_MTP_SIDECAR_BATCH_VERIFY=1`; otherwise
DS4 skips MTP by default because the exact sidecar verifier is slower than
ordinary banked decode on that layout.

For hybrid resident sidecars with `IQ2_XXS` gate/up and `MXFP4_NATIVE` down,
`DS4_MTP_SIDECAR_BATCH_VERIFY=1` intentionally falls back to exact decode2 by
default. The experimental hybrid batch verifier is available for profiling, but
was measured slower than exact decode2:

```sh
DS4_MTP_HYBRID_BATCH_VERIFY_EXPERIMENT=1 \
DS4_MTP_SIDECAR_BATCH_VERIFY=1 \
./ds4 -m "$DS4_SIDECAR_DIR" --resident --mtp "$DS4_MTP_GGUF" --mtp-draft 2
```

MTP is only expected to help when the target verifier is cheaper than the
accepted target tokens it replaces. A high acceptance rate alone does not
guarantee a speedup; compare the final `generation:` tokens-per-second line
against the same command without `--mtp`.

## Run DSpark With A Sidecar

The primary DSpark path is the Flash sidecar target plus a separate DS4-owned
DSpark draft package. Treat GGUF main-model runs as a compatibility path for
agent demos; the sidecar path is the reference for speed and correctness. The
draft checkpoint must match the target shape, so use the Flash DSpark checkpoint
with Flash sidecars, not the Pro DSpark checkpoint.

Use these paths in the examples below:

```sh
export DS4_SIDECAR_DIR=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
export DS4_DSPARK_DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
export TEST_PROMPT='Make a game of Space Invader in Pygame'
```

Download the pre-exported Flash DSpark draft package:

```sh
DS4_DSPARK_DRAFT_DIR="$DS4_DSPARK_DRAFT" ./download_model.sh dspark
```

This downloads [anemll/DSv4-Flash-DSpark-draft](https://huggingface.co/anemll/DSv4-Flash-DSpark-draft)
directly into the DS4 runtime package layout. To rebuild the package locally
from the original DeepSeek shards instead, download only the Flash DSpark draft
shards:

```sh
mkdir -p /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark \
  --include config.json \
  --include model.safetensors.index.json \
  --include model-00046-of-00048.safetensors \
  --include model-00047-of-00048.safetensors \
  --include model-00048-of-00048.safetensors
```

Export the DS4-owned draft package:

```sh
scripts/export_dspark_draft.sh \
  --source-dir /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark \
  --out-dir "$DS4_DSPARK_DRAFT" \
  --variant flash \
  --force
```

Validate the package against the Flash sidecar target:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 5 \
  --inspect
```

The expected package metadata is DSpark-5: block size 5, target layers
`40,41,42`, three draft layers, 256 experts, and Markov rank 256. DSpark-5
means the checkpoint can draft up to 5 tokens per block; it does not require
every run to verify all 5. For current Flash sidecar runs, pin
`--draft-verify 4`: in static mode the active proposal length is
`min(block_size, --draft-verify)`, so the loader prints `block=5 verify=4
active=4`. This avoids the slowest fifth draft position while still emitting
the normal target token before each DSpark block. Use `--draft-verify 2`, `3`,
or `5` for fixed-budget A/B tests.

Run a paired sidecar baseline first:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --temp 0 \
  --nothink \
  -n 1000 \
  -c 4096 \
  -p "$TEST_PROMPT"
```

Then run DSpark on the same sidecar target:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --draft-scheduler static \
  --temp 0 \
  --nothink \
  -n 1000 \
  -c 4096 \
  -p "$TEST_PROMPT"
```

The sidecar command is the current clean reference path. DS4 selects the strict
commit-safe hybrid verifier by default for DSpark greedy runs unless
`--quality`, `DS4_DSPARK_EXACT_VERIFY=1`, or `DS4_DSPARK_FAST_VERIFY_DISABLE=1`
is set. A successful run prints:

```text
ds4: DSpark draft package loaded: ... (block=5 verify=4 active=4 ...)
ds4: DSpark draft inference enabled: MPP 4.1 FP8/MXFP4 draft kernels ...
ds4: dspark perf: draft=... verify=... block=... tau=...
ds4: dspark acceptance: ...
ds4: dspark acceptance by position: ...
ds4: dspark avg scheduled: ...
```

Here `tau` in the DSpark perf line means accepted draft tokens per DSpark
speculation block. End-to-end generated tokens also include the leading target
token that seeded the block. With `--draft-verify 4`, the DSpark perf `tau` cap
is therefore `4.0`; use `--draft-verify 5` to test the full DSpark-5 block.

For adaptive budget experiments, keep `--draft-verify` as the cap and add
`--draft-verify-dynamic`. The default controller is measured-throughput based:
it starts at the cap, learns conditional acceptance by draft position, samples
the active budgets `2,3,4,5`, and then selects the budget with the best expected
accepted draft tokens per measured block second. This is intentionally closer
to the Ollama/MLX dynamic-depth controller than the older last-window tau
heuristic, so it can recover upward when a deeper block becomes worthwhile
again. The default fast-start setting trusts a position after 4 conditional
samples; set `DS4_DSPARK_VERIFY_DYNAMIC_MIN_SAMPLES=10` for a slower
Ollama-like ramp. Set `DS4_DSPARK_VERIFY_DYNAMIC_LOG=1` to print budget
changes, and use `DS4_DSPARK_VERIFY_DYNAMIC_MIN=N` to constrain the lower bound
for A/B tests. Set `DS4_DSPARK_VERIFY_DYNAMIC_LEGACY=1` to restore the older
tau-window controller.

`--draft-scheduler confidence-softmax` is a strict adaptive-budget experiment:
it uses the DSpark confidence head to choose how many proposed tokens to verify,
but it still accepts only normal target-verified tokens. This is not relaxed
acceptance and should not introduce the repetition failures seen in relaxed
fast mode. Start with logging enabled:

```sh
DS4_CTX_GROW_BLOCK=2048 \
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_DSPARK_PERF=1 \
DS4_DSPARK_VERIFY_DYNAMIC_LOG=1 \
DS4_DSPARK_CONFIDENCE_SOFTMAX_LOG=1 \
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 5 \
  --draft-verify-dynamic \
  --draft-scheduler confidence-softmax \
  --temp 0 \
  --nothink \
  -n 1000 \
  -c 4096 \
  -p "$TEST_PROMPT"
```

By default, `confidence-softmax` uses the verifier-cost-biased preset also
available as `confidence-softmax-long`: `MIN=4`, `FIXED_COST=12`,
`TOKEN_COST=0.25`, and `MASS_WEIGHT=0.5`. That keeps the scheduler close to the
full DSpark-5 block while still allowing it to drop the weak tail. Set
`DS4_DSPARK_CONFIDENCE_SOFTMAX_LEGACY=1` to restore the older conservative
defaults for A/B testing. Useful tuning knobs are
`DS4_DSPARK_CONFIDENCE_SOFTMAX_MIN`, `DS4_DSPARK_CONFIDENCE_SOFTMAX_TEMP`,
`DS4_DSPARK_CONFIDENCE_SOFTMAX_FIXED_COST`,
`DS4_DSPARK_CONFIDENCE_SOFTMAX_TOKEN_COST`,
`DS4_DSPARK_CONFIDENCE_SOFTMAX_MASS_WEIGHT`, and
`DS4_DSPARK_CONFIDENCE_SOFTMAX_FLOOR`. Lower `TOKEN_COST` or higher
`MASS_WEIGHT` pushes toward longer prefixes. `DS4_DSPARK_CONFIDENCE_SOFTMAX_MIN=0`
is an emergency-skip diagnostic: it can skip verification for a low-confidence
draft block and fall back to the already-emitted target token, but if it fires
too often it wastes draft time and behaves like baseline plus overhead.
`DS4_DSPARK_CONFIDENCE_SOFTMAX_SKIP_BIAS` controls how willing the scheduler is
to choose that zero-verify escape hatch.

Relaxed fast-mode experiments can raise `tau` by accepting target-supported
non-argmax draft tokens, but this changes the greedy contract and is not
recommended for demos or agent coding tasks. The default relaxed gate now
requires target uncertainty before a non-argmax token is accepted:
`top1-top2 <= DS4_DSPARK_RELAXED_TARGET_MARGIN` and
`top1-draft <= DS4_DSPARK_RELAXED_DRAFT_MARGIN` in addition to the configured
top-k/logit gate. The simple loop guard is only a secondary brake; it rejects
non-argmax draft tokens that would extend a recent repeated token n-gram or
overuse one token in a short window.

For `ds4-agent`, keep the same sidecar model and draft package:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 DS4_AGENT_TURN_STATS=1 \
./ds4-agent \
  --model "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --temp 0 \
  --nothink \
  --ctx 24096 \
  --debug-status
```

The following relaxed-mode shape is retained only as a diagnostic speed
experiment. It changes the greedy contract, so do not use it for demos without
a separate quality pass. The off-argmax cap is important: older loose relaxed
commands without it produced broken `ds4-agent` HTML/game output with duplicate
declarations and repeated constants.

```sh
DS4_CTX_GROW_BLOCK=2048 \
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_DSPARK_PERF=1 \
DS4_AGENT_TURN_STATS=1 \
DS4_DSPARK_RELAXED_LOOP_LOG=1 \
DS4_DSPARK_RELAXED_COOLDOWN_BLOCKS=4 \
./ds4-agent \
  --model "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 5 \
  --draft-fast-relaxed \
  --draft-scheduler confidence \
  --draft-conf-threshold 0.4 \
  --temp 0 \
  --nothink \
  --ctx 24096 \
  --debug-status
```

`DS4_DSPARK_RELAXED_MARGIN_DISABLE=1` restores the older top-k/delta-only
diagnostic gate, which is useful only for speed-ceiling comparisons.
`DS4_DSPARK_RELAXED_LOOP_GUARD_DISABLE=1` is also only for unsafe speed-ceiling
comparisons. `DS4_DSPARK_RELAXED_COOLDOWN_BLOCKS=N` is a recoverable brake:
after a loop-guard rejection, off-argmax relaxed accepts are suppressed for `N`
speculation blocks while target-argmax DSpark accepts can continue.
The relaxed loop guard defaults to `DS4_DSPARK_RELAXED_LOOP_NGRAM=3`,
`DS4_DSPARK_RELAXED_LOOP_TOKEN_MAX=12`, and checks target-top accepted tokens
too; set `DS4_DSPARK_RELAXED_ALLOW_TARGET_TOP_REPEAT=1` only to reproduce the
older looser behavior.
`DS4_DSPARK_RELAXED_MAX_OFFARGMAX_PER_BLOCK=N` caps how many non-argmax but
target-supported draft tokens one block may accept. Start with `N=1` when
testing loose gates. The current `--draft-fast-relaxed` preset uses
`TOPK=256` / `LOGIT_DELTA=10`; it kept the 1000-token Space Invaders smoke
canary-clean in local testing and was the best point in the local relaxed
sweep, but it still measured below the >60 t/s goal. `DS4_DSPARK_RELAXED_OFFARGMAX_COOLDOWN_BLOCKS=N`
can also force a short target-argmax-only cooldown after any off-argmax accept.

For `ds4-server`, use the same resident sidecar target. DSpark is greedy-only,
so client requests must use `temperature: 0` if you want speculative decoding:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 \
./ds4-server \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --ctx 4096 \
  --tokens 4096 \
  --host 127.0.0.1 \
  --port 8000
```

Example OpenAI-compatible request:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Make a game of Space Invader in Pygame"}
    ],
    "max_tokens": 160,
    "temperature": 0,
    "stream": false
  }'
```

Add `--dspark-attn-force-mma` only for a faster Mode-B/demo run where exact
byte identity with the strict verifier is not the goal.

If a demo must use a GGUF main model, keep DSpark as the same external draft
package and pass `--draft-path` explicitly:

```sh
export DS4_GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 DS4_AGENT_TURN_STATS=1 \
./ds4-agent \
  --model "$DS4_GGUF" \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --temp 0 \
  --nothink \
  --ctx 24096 \
  --debug-status
```

The sidecar path remains preferred for DSpark. GGUF main-model runs are useful
for compatibility demos, but the sidecar target has lower memory pressure and
is the speed reference.

Operational notes:

- DSpark is greedy-only today. Use `--temp 0`; nonzero temperature disables the
  draft verifier and prints a warning.
- The Flash DSpark draft package is kept resident by default. A normal run
  refuses to fall back to disk-backed draft experts, so speed measurements do
  not silently switch paths.
- `--draft-verify 5` is the recommended Flash sidecar fast-preset budget. Use
  `--draft-verify 2`, `3`, or `4` only for explicit A/B sweeps.
- `--dspark-attn-force-mma` is a faster demo/Mode-B diagnostic. It is not the
  strict byte-identical verifier path.
- `--draft-fast-relaxed` enables the current fast demo preset, including the
  frontier draft path, relaxed suffix accept (`TOPK=256`, `LOGIT_DELTA=10`),
  GPU-queue draft prefetch, MMA attention, and fast Q2 down. It is a
  non-byte-identical diagnostic mode, not the strict verifier. The prefetch
  hides only queue/readback slack; it is not true ANE/separate-engine overlap.
- DSpark can run against a streaming/direct-mmap sidecar, but that path is not
  the speed target: verifier work becomes SSD/VM-bound. Use `--resident` for
  DSpark throughput measurements.

Useful DSpark diagnostics:

```sh
DS4_DSPARK_PERF=1                 # print draft/verify/block timing
DS4_DSPARK_BLOCK_TIMING=1         # per-block diagnostic timing
DS4_DSPARK_BASELINE_TPS=<t/s>     # normalize against a paired no-draft run
N=160 scripts/dspark_phase0_sweep.sh
```

The final `generation:` line is the headline speed. DSpark also prints
acceptance, acceptance by draft position, average scheduled draft length, and
`tau`, where `tau = accepted/emitted draft tokens per DSpark block`.

### Experimental Pro Support

DeepSeek V4 Pro sidecar support is experimental. For Pro agent runs, use
`--nothink`, keep the slot bank at or below 32 slots while tuning, and keep
shared-down decode prefetch enabled:

```sh
DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN=1 ./ds4-agent \
  -m ~/Models/DSv4Pro-flash/ \
  --moe-slot-bank 32 \
  --ctx 32768 \
  --nothink
```

Larger Pro slot banks can consume enough memory bandwidth and residency budget
to collapse decode throughput, so only raise `--moe-slot-bank` after measuring
reuse and decode stalls on your machine.

For diagnostic fanout tests, add `--moe-expert-topk 4`. This is different from
`--moe-prefetch-topk`: it changes the actual routed expert count for both
prefill and decode, so quality and logits are expected to change.

`--no-int8` is optional. Normal runs use the fastest measured profile path.
For quality-preserving runs, pass `--no-int8`; it disables current int8 dense,
NAX, Flash-MoE, and ANE accelerator paths, using NAX-half where safe and GPU
fallbacks otherwise. `--quality` implies `--no-int8`.

## Run Resident GGUF Mode

Download a resident GGUF:

```sh
./download_model.sh q2-imatrix
```

An alternate resident GGUF,
[Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF/resolve/main/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf),
is available from
[huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF):

```sh
./download_model.sh huihui-iq2xxs
```

Then run:

```sh
./ds4 -p "Hello"
```

Resident mode loads the full model file and is meant for high-memory machines.
It is still useful for baseline comparison, server use, and systems with enough
RAM to hold the selected quantization. The Huihui IQ2_XXS resident GGUF can be
used on a 96 GB M3 Ultra, but memory headroom is tight; run with little to
nothing else active. Reaped/pruned resident models are still under investigation.

See [docs/RESIDENT.md](docs/RESIDENT.md) and
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Validate

The alpha validation gate is:

```sh
make clean
make
./ds4_test --server --metal-kernels
make ane-smoke
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

The sidecar smoke uses `tests/test-vectors/prompts/long_code_audit.txt`, a
shorter 4K-class prompt, and generates 64 deterministic tokens with a 4K
prefill chunk cap.

## Docs

- [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md): model files, downloads, and
  sidecar package expectations.
- [docs/SIDECAR.md](docs/SIDECAR.md): SSD streaming mode and smoke test.
- [docs/SIDECAR_EXPORT.md](docs/SIDECAR_EXPORT.md): external expert-sidecar
  export wrapper and its dense-GGUF caveats.
- [docs/STREAMING_KNOBS.md](docs/STREAMING_KNOBS.md): SSD sidecar slot-bank,
  prefill, I/O, ANE, and profile knobs.
- [docs/RESIDENT.md](docs/RESIDENT.md): full-GGUF resident mode.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): runtime layout and accelerator
  paths.
- [docs/PROFILES.md](docs/PROFILES.md): machine-specific tuning defaults and
  override rules.
- [docs/ANE_KERNELS.md](docs/ANE_KERNELS.md): experimental Apple Neural Engine
  kernel families and private API notes.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): current benchmark stance.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common first-run failures.
- [docs/DWARFSTAR4_REFERENCE.md](docs/DWARFSTAR4_REFERENCE.md): original DS4
  README retained for reference.

## Attribution

`ds4-ssd` is derived from antirez's DwarfStar 4 / `ds4` work and keeps the DS4
model-specific design: GGUF loading, prompt rendering, KV handling, server API,
and DeepSeek V4 Flash validation. The project also depends conceptually on the
GGUF, quantization, and kernel work pioneered by `llama.cpp` and GGML.

The SSD-streaming direction is also indebted to Apple's
[LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://machinelearning.apple.com/research/efficient-large-language)
paper and to the original [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
work by Claude Opus 4.6 and Daniel Woods. Read the
[original Flash-MoE paper](https://github.com/danveloper/flash-moe/blob/main/paper/flash_moe.pdf)
for the full story of how they built that engine in 24 hours.

The Apple Neural Engine path uses GPU-side int8 dequantization/packing together
with ANE MLP execution through private Apple APIs and additional scheduling
optimizations. GPU int8 dequantization for this class of local inference was
pioneered by Liu Liu (Draw Things, @liuliu), and the private ANE API path was
first documented publicly by @maderix.

Keep the repository `LICENSE` with redistributions and preserve attribution to
antirez, llama.cpp, GGML, and their contributors.
