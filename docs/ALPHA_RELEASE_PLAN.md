# DS4 SSD Alpha Release Plan

This is the agent-facing plan for creating a clean alpha `main` branch from the
current `agent-clean` research branch.

## Current State

- Working branch: `agent-clean`.
- Remote default is still `origin/SSD-initial-port`; there is no real `main`
  branch yet.
- `agent-clean` descends from `origin/SSD-initial-port`.
- `agent-clean` contains the full research history and many experiment files.
- The ANE int8w runtime relocation is already done and pushed:
  `4fc77ce refactor: relocate ANE int8w runtime out of moe-batch-bench to top level`.
- Local untracked paths such as `.claude/scheduled_tasks.lock` and
  `profile_runs/` are not release content.

## Release Goal

Create a clean alpha `main` branch for `ds4-ssd` that presents the project as:

- An alpha fork/derivative of Antirez DwarfStar 4 / `ds4`.
- Focused on DeepSeek V4 Flash.
- Centered on SSD-streamed routed-MoE sidecar support.
- Also supporting resident/full-GGUF operation on larger memory machines.
- Including Apple Silicon optimizations:
  - Metal graph path.
  - NAX / Apple GPU `matmul2d` neural-accelerator paths.
  - Apple Neural Engine prefill paths where they win.

The alpha tree should be usable and technically credible, but not polished into
a full v1-style product.

## Hard Rules

- Preserve `LICENSE`.
- Preserve clear attribution to Antirez DwarfStar 4 / `ds4`.
- Preserve the llama.cpp / GGML acknowledgement in the new README, at least as a
  concise pointer.
- `metal/` is mandatory release content. It is a build input through
  `METAL_SRCS := $(wildcard metal/*.metal)`.
- Do not delete research/dev tooling from `agent-clean`; prune only in the alpha
  worktree/squash.
- Do not document commands that do not exist.
- Do not imply 32K prefill chunks are supported for the sidecar alpha path.
  Sidecar smoke/first-run validation uses explicit 16K prefill:
  `DS4_METAL_PREFILL_CHUNK=16384`.

## Go / No-Go

The alpha can ship when:

1. Sidecar setup is honest:
   - Best: an in-repo sidecar packer exists and is tested.
   - Acceptable alpha fallback: prebuilt sidecar download is turnkey, and docs
     state self-conversion is not public yet.
2. The top-level tree is de-noised.
3. The new README tells the correct story:
   - SSD sidecar enables smaller RAM systems.
   - Resident/full-GGUF mode is separate and needs larger memory.
4. The alpha branch is a clean curated branch, not the full noisy experiment log.
5. Minimal alpha verification passes on the squashed commit.

## Branch Strategy

Recommended flow:

```sh
git worktree add ../ds4-alpha origin/SSD-initial-port
cd ../ds4-alpha
git checkout -b main-alpha
git merge --squash agent-clean
```

Then perform all pruning, moves, and doc edits in the staged squash before the
single alpha commit.

After validation:

```sh
git commit -m "alpha: SSD streaming and Apple Silicon optimized DS4"
git branch -m main
git push origin main
git tag v0.1.0-alpha.1
git push origin v0.1.0-alpha.1
```

GitHub still needs a separate default-branch update:

- Set repo default branch to `main`.
- Keep `agent-clean` as the research/blame branch.
- Decide whether `SSD-initial-port` remains an upstream-sync/base branch.
- Prune stale experiment branches separately, not as part of this alpha cut.

## Tree Policy

Keep top level:

- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `MODEL_CARD.md`
- `Makefile`
- core source files: `ds4*.c`, `ds4*.h`, `ds4_cuda.cu`,
  `ds4_iq2_tables_cuda.inc`
- `ds4_ane_mlp_int8w.{m,h}`
- `linenoise.{c,h}`
- `rax.{c,h}`, `rax_malloc.h`
- `metal/`
- `download_model.sh`
- `ds4_profile.json`
- likely `ds4_backend_env.sh`
- `docs/`
- `gguf-tools/`
- `tests/`

Classify explicitly:

- `dir-steering/`: keep as feature only if `--dir-steering-*` stays public;
  otherwise defer from alpha.
- `speed-bench/`: optional; can stay only if it is presented as reference data.
- `moe-batch-bench/`: mostly dev/research. Do not keep as public alpha surface
  unless a specific file is needed for verification or tool support.
  Preferred alpha resolution: move
  `moe-batch-bench/ane_ds4_mlp_i8i8_precision_smoke.m` to `tests/` as the direct
  ANE int8w runtime smoke, then prune `moe-batch-bench/` wholesale.

Move or drop from alpha:

- root profiling/dev scripts: `run_*.sh`, `tune_*.sh`, `validate_*.sh`.
  Move to `scripts/dev/` or remove from the public alpha branch.
- internal handoff/progress notes:
  `ANE_PREFILL_PROGRESS.md`, `RESIDENT_ANE_PREFILL_HANDOFF.md`,
  `RESIDENT_DECODE_STUTTER_HANDOFF.md`, `ane_implementation_analysis.md`,
  `INT4_MATMUL_ANE_WORKFLOW.md`, `M5_TEST_RUNBOOK.md`, `CTX_GROW.md`,
  `AGENT.md`.
  Either delete from alpha or move to `docs/dev/` after distilling the useful
  facts into `docs/ARCHITECTURE.md`.
- huge session/export artifacts such as `INT4_MATMUL_session_export.txt` and
  `2026-05-20-081559-...txt`.
- generated/local output.
- dangling Makefile smoke/probe targets that reference pruned or missing
  `moe-batch-bench` sources, including the old ANE multi/oproj/per-chunk smoke
  targets.

Check while pruning:

- `CONTRIBUTING.md` must not reference scripts or `moe-batch-bench` paths that
  are removed or moved.
- CUDA files are inherited and should remain buildable on non-Darwin, but the
  alpha README / architecture docs should say the alpha validation focus is
  Apple Silicon; CUDA is not the tested headline path for this release.

`.gitignore` addition for alpha:

```gitignore
/profile_runs/
```

## Documentation Structure

Create a new short alpha README and move the current long README into reference
docs.

Recommended structure:

```text
README.md
docs/
  DWARFSTAR4_REFERENCE.md
  MODEL_SETUP.md
  SIDECAR.md
  RESIDENT.md
  ARCHITECTURE.md
  PERFORMANCE.md
  TROUBLESHOOTING.md
  dev/
    ALPHA_RELEASE_PLAN.md
```

`ALPHA_RELEASE_PLAN.md` is internal agent/process documentation. Keep it on
`agent-clean`, or move it under `docs/dev/` if it needs to survive in the alpha
worktree. Do not present it as public user documentation.

Required for alpha:

- `README.md`: 100-180 lines, alpha-facing.
- `docs/DWARFSTAR4_REFERENCE.md`: current README preserved as upstream engine
  reference/context.
- `docs/MODEL_SETUP.md`: Hugging Face downloads, model choices, memory
  expectations.
- `docs/SIDECAR.md`: SSD sidecar usage. Only document real commands.
- `docs/ARCHITECTURE.md`: short technical explanation of SSD streaming,
  resident mode, NAX, ANE, and machine profiles.

Can be stubs/deferred:

- `docs/PERFORMANCE.md`
- `docs/TROUBLESHOOTING.md`

New README must include:

- What `ds4-ssd` is.
- Why SSD sidecar exists.
- Difference between sidecar and resident mode.
- Build commands.
- Fastest path to run from prebuilt models.
- Link to sidecar conversion/download docs.
- Alpha caveats.
- Antirez / DwarfStar 4 attribution.
- llama.cpp / GGML acknowledgement pointer.

## Sidecar Converter / Download Decision

The runtime already loads a sidecar directory with:

- `manifest.json`
- expert-major sidecar records
- dense model at `dense/model-dense.gguf` or equivalent path

The tracked `gguf-tools/deepseek4-quantize.c` does not currently emit that
sidecar format.

Decision required before final docs:

- Option A: add an in-repo `gguf-tools/ds4-sidecar-pack` or equivalent.
- Option B: publish prebuilt sidecar artifacts on Hugging Face and document
  self-conversion as not included in this alpha.

Do not write `docs/SIDECAR.md` around a nonexistent command.

Once Hugging Face paths are known, update `download_model.sh` with sidecar
targets such as `sidecar-q2` / `sidecar-q4` or whatever names are chosen.

## Alpha Verification

Do not overbuild test infrastructure for alpha. Stage 1 is:

1. Clean build:

```sh
make clean
make
```

2. Existing fast correctness/kernel checks:

```sh
./ds4_test --server --metal-kernels
```

3. Direct ANE runtime smoke after the relocation. In the alpha tree this should
   live under `tests/`, not under `moe-batch-bench/`:

```sh
make tests/ane_ds4_mlp_i8i8_precision_smoke
./tests/ane_ds4_mlp_i8i8_precision_smoke
```

4. One sidecar smoke. Use explicit 16K prefill:

```sh
DS4_METAL_PREFILL_CHUNK=16384 ./ds4 \
  -m /path/to/dsv4-iq2xxs-expert-major/dense/model-dense.gguf \
  --moe-sidecar /path/to/dsv4-iq2xxs-expert-major \
  --moe-mode slot-bank \
  --moe-slot-bank 64 \
  --ctx 32768 \
  -n 1 \
  --temp 0 \
  --prompt-file tests/test-vectors/prompts/coding/coding_16k.txt
```

Expected sidecar smoke signals:

- exit code `0`
- `Flash-MoE sidecar loaded`
- `prefill chunk cap: 16384`
- `Flash-MoE slot banks allocated`
- routed compute banner

Full `make test` can be run when the full resident GGUF is present and the time
cost is acceptable, but it is not the sidecar smoke and should not be confused
with the alpha headline verification.

Release notes should say:

```text
Alpha validation covers the existing fast correctness/kernel checks plus one
SSD sidecar smoke. Broader mode coverage, CI, and performance regression testing
are planned for 0.2.
```

## Known Completed Work

- `ds4_ane_mlp_int8w.{m,h}` moved to top level.
- `Makefile` core and smoke targets updated.
- `ds4_metal.m` include updated.
- dependent `moe-batch-bench` smoke/bench includes updated.
- Verification performed before commit:
  - `make clean && make`
  - `./ds4_test --server --metal-kernels`
  - precision smoke built and ran
  - 16K sidecar smoke exited `0`

## Open Decisions

- Prebuilt sidecar Hugging Face repo/path.
- Whether to ship an in-repo sidecar packer for alpha.
- Whether `dir-steering/` remains in the public alpha tree.
- Whether `speed-bench/` remains in the public alpha tree.
- Which root profiling scripts, if any, become supported user scripts.
- Whether internal notes are deleted or moved to `docs/dev/`.
