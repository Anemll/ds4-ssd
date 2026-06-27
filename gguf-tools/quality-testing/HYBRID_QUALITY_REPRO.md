# Hybrid vs IQ2 Quality Reproduction

This note records the local commands used to compare the hybrid sidecar against
the baseline IQ2 GGUF. Keep benchmark checkouts and outputs in `/tmp` unless the
result is intentionally being added to the repo.

## Models

Hybrid sidecar package:

```sh
/Users/anemll/Models/ds4-hybrid-down-layer-sweep-model
```

Baseline IQ2 GGUF:

```sh
/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

Hybrid architecture:

- Dense weights remain IQ2.
- Routed gate/up experts remain IQ2_XXS.
- Routed down experts are upgraded to MXFP4_NATIVE plane-split on layers:
  `0, 9, 10, 11, 12, 15, 18, 21, 25, 27`.

## Official Continuation Scorer

Run from the `ds4-ssd` repo:

```sh
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd

DS4_MXFP4_NATIVE=1 \
gguf-tools/quality-testing/score_official \
  --ctx 4096 \
  --ssd-cache auto \
  /Users/anemll/Models/ds4-hybrid-down-layer-sweep-model \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/hybrid_100.tsv
```

Current 100-case snapshot:

```text
HYBRID resident --no-int8 summary cases=100 tokens=2290 avg_nll=0.379792583 first_match=70 avg_lcp=7.570
HYBRID normal              summary cases=100 tokens=2290 avg_nll=0.382228881 first_match=72 avg_lcp=7.540
IQ2                        summary cases=100 tokens=2290 avg_nll=0.412994818 first_match=66 avg_lcp=6.470
```

Delta:

- `avg_nll`: Hybrid resident `--no-int8` lower than IQ2 by `0.033202235`,
  about `8.04%`.
- `first_match`: Hybrid resident `--no-int8` is `+4` vs IQ2, but `-2` vs
  Hybrid normal.
- `avg_lcp`: Hybrid resident `--no-int8` is `+1.10` vs IQ2 and `+0.03` vs
  Hybrid normal.

## IFEval Through Server API

Install once into `/tmp`:

```sh
/opt/homebrew/bin/python3.11 -m venv /tmp/lm-eval
/tmp/lm-eval/bin/pip install 'lm_eval[api,ifeval]'
```

Start Hybrid server:

```sh
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4-ssd

./ds4-server \
  -m /Users/anemll/Models/ds4-hybrid-down-layer-sweep-model \
  --resident \
  --ctx 32768 \
  --port 8000
```

Run IFEval limit 100:

```sh
/tmp/lm-eval/bin/lm_eval \
  --model local-chat-completions \
  --model_args model=deepseek-chat,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=1,max_retries=1,eos_string='<｜end▁of▁sentence｜>' \
  --tasks ifeval \
  --limit 100 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path /tmp/ifeval-hybrid-limit100 \
  --log_samples
```

Start IQ2 server from the upstream `ds4` repo:

```sh
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4

./ds4-server \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx 32768 \
  --port 8000
```

Run the same IFEval command with output path:

```sh
/tmp/lm-eval/bin/lm_eval \
  --model local-chat-completions \
  --model_args model=deepseek-chat,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=1,max_retries=1,eos_string='<｜end▁of▁sentence｜>' \
  --tasks ifeval \
  --limit 100 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path /tmp/ifeval-iq2-limit100 \
  --log_samples
```

Current 100-sample IFEval snapshot:

| Metric | Hybrid | IQ2 |
|---|---:|---:|
| prompt strict | 0.870 | 0.840 |
| prompt loose | 0.890 | 0.860 |
| instruction strict | 0.9141 | 0.9018 |
| instruction loose | 0.9325 | 0.9141 |

## ToolCall-15

ToolCall-15 is useful because it exercises server-side structured tool calls,
multi-turn mocked tool results, restraint, and error recovery.

Install into `/tmp`:

```sh
rm -rf /tmp/ToolCall-15
git clone --depth 1 https://github.com/stevibe/ToolCall-15.git /tmp/ToolCall-15
cd /tmp/ToolCall-15
npm install
npm run build:cli
```

Run against whichever model server is currently listening on port 8000:

```sh
cd /tmp/ToolCall-15

/usr/bin/time -p sh -c 'LLAMACPP_HOST=http://127.0.0.1:8000 LLM_MODELS=llamacpp:deepseek-chat MODEL_REQUEST_TIMEOUT_SECONDS=180 node dist-cli/cli/run.js --model llamacpp:deepseek-chat --temperature 0 --timeout 180 --json --show-raw > /tmp/toolcall15-MODEL.raw' \
  2> /tmp/toolcall15-MODEL.time
```

Extract clean JSON if the raw file contains progress text before the JSON:

```sh
python3 - /tmp/toolcall15-MODEL.raw /tmp/toolcall15-MODEL.clean.json <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
s = src.read_text()
start = s.find('{\n  "scenarios"')
if start < 0:
    start = s.find('{')
data = json.loads(s[start:])
dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

Current ToolCall-15 snapshot:

| Model | Score | Points | Rating |
|---|---:|---:|---|
| Hybrid | 87 | 26/30 | Good |
| Hybrid `--no-int8` | 87 | 26/30 | Good |
| IQ2 | 93 | 28/30 | Excellent |

Category deltas:

| Category | Hybrid | Hybrid `--no-int8` | IQ2 |
|---|---:|---:|---:|
| Tool Selection | 6/6 | 6/6 | 6/6 |
| Parameter Precision | 4/6 | 4/6 | 4/6 |
| Multi-Step Chains | 4/6 | 6/6 | 6/6 |
| Restraint/Refusal | 6/6 | 4/6 | 6/6 |
| Error Recovery | 6/6 | 6/6 | 6/6 |

Observed scenario differences:

- Both models failed `TC-06` by not splitting one request into two valid
  `translate_text` calls.
- Hybrid failed `TC-07`; it did `search_files -> read_file` but did not complete
  the required `get_contacts -> send_email` continuation.
- Hybrid `--no-int8` passed `TC-07`, but failed `TC-12` by calling an unrelated
  tool before refusing the unsupported email-deletion request.
- IQ2 passed `TC-07`.
- `TC-13` passed for both, but Hybrid asked for clarification after an empty
  result while IQ2 retried search and recovered the file.

Speed notes from the 2026-06-23 run:

- Hybrid resident sidecar decoded tool-call chunks mostly around `30-34 t/s`.
- Hybrid `--no-int8` decoded tool-call chunks around `30.94 t/s` average on the
  2026-06-23 full ToolCall-15 run (`min=29.64`, `max=31.62`, wall `226.34s`).
- IQ2 GGUF decoded tool-call chunks mostly around `35-36 t/s`.
- IQ2 ToolCall-15 wall time was `143.63s`.
- For repeatable speed analysis, start each server with stdout/stderr tee'd to a
  `/tmp/toolcall15-*.server.log` file and parse `prompt done`, `decoding chunk`,
  and `finish` lines.

Implementation note:

- The `--no-int8` resident partial-tile warning should read
  `MPP/NAX partial-tile workaround active (--no-int8 path; ...)`.
  The older `MPP/NAX int8 partial-tile workaround active` text is only valid for
  normal fast-mode int8 paths.
