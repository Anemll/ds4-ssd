# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_agent.c`: native terminal coding agent, DSML tool loop, session save /
  resume UI, and live footer/status reporting.
- `ds4_kvstore.c`, `ds4_kvstore.h`: shared KV checkpoint file format and helper
  routines used by the server cache and native agent sessions.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Native Agent Notes

- `ds4-agent` is part of this repository, not a separate project. It links the
  same engine objects as `ds4` and `ds4-server`, plus `ds4_agent.c`,
  `ds4_kvstore.c`, and `linenoise.c`.
- Agent session files live under `~/.ds4/kvcache` and use the normal DS4 session
  payload. `/save` writes a SHA-named session, `/switch SHA` loads one inside a
  running agent, and `--resume SHA` loads one at startup.
- On clean exit after a saved session, the agent prints a complete
  `./ds4-agent ... --resume SHA` command including model, backend, context, MTP,
  Flash-MoE sidecar, steering, quality, warm-weights, and think-mode flags.
- The footer's "completed in ..." time measures one full submitted user turn:
  prefill, generation, tool calls, compaction, and any continuation rounds. It
  resets when the next prompt is submitted, when starting a new session, or when
  resuming another session.
- Prefill progress has two events. `prefill_chunk` is the durable KV/checkpoint
  boundary and must stay aligned with real safe persistence points.
  `prefill_display` is UI-only progress, currently interpolated by completed
  layer within a large Metal/Flash-MoE chunk so the footer and t/s estimate move
  even when a whole chunk is processed in one pass.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.
