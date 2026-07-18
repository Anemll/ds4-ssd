#!/usr/bin/env python3
"""Score saved official continuations against a live OpenAI-compatible server.

This is the API equivalent of ``score_official.c``.  It never calls the
hosted DeepSeek API: ``manifest.tsv`` and its continuations are the immutable
reference corpus.  For each reference token it asks vLLM's
``/generative_scoring`` endpoint for

    p(reference_token | chat_prompt + prior_reference_tokens)

and reports corpus-weighted negative log likelihood (NLL), perplexity, first
greedy-token matches, and greedy longest-common-prefix (LCP) length.

The server's ``prompt_logprobs`` response is deliberately not used.  On the
current DSpark deployment it returns invalid scores for echoed prompt tokens,
whereas ``/generative_scoring`` produces valid next-token probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "http://gx10-30c1.local:8888"
PROM_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)$"
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def path_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def request_json(
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float,
) -> Tuple[int, Optional[Dict[str, Any]], str, float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None
            return int(response.status), parsed, raw, (time.perf_counter() - started) * 1000.0
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        return int(error.code), parsed, raw, (time.perf_counter() - started) * 1000.0


def request_text(url: str, *, timeout: float) -> Tuple[int, str, float]:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"}, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace"), (time.perf_counter() - started) * 1000.0
    except urllib.error.HTTPError as error:
        return int(error.code), error.read().decode("utf-8", "replace"), (time.perf_counter() - started) * 1000.0


def parse_prometheus(raw: str) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_SAMPLE_RE.match(line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            samples[name + ("{" + labels + "}" if labels else "")] = value
    return samples


def prometheus_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Dict[str, Optional[float]]]:
    delta: Dict[str, Dict[str, Optional[float]]] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        change = None if before_value is None or after_value is None else after_value - before_value
        if change is None or change != 0.0:
            delta[key] = {"before": before_value, "after": after_value, "delta": change}
    return delta


def metric_change(delta: Dict[str, Dict[str, Optional[float]]], metric_name: str) -> Optional[float]:
    values = [
        value["delta"]
        for key, value in delta.items()
        if (key == metric_name or key.startswith(metric_name + "{")) and value["delta"] is not None
    ]
    return float(sum(values)) if values else None


def mean_ms(total_seconds: Optional[float], count: Optional[float]) -> Optional[float]:
    if total_seconds is None or count is None or count <= 0:
        return None
    return 1000.0 * total_seconds / count


def default_manifest() -> Path:
    local = ROOT / "data" / "flash" / "manifest.tsv"
    if local.exists():
        return local
    # The untracked API reference data is commonly retained in the sibling ds4
    # checkout.  This fallback keeps the harness self-contained without making
    # a duplicate 100-case corpus part of this repository.
    sibling = ROOT.parent.parent.parent / "ds4" / "gguf-tools" / "quality-testing" / "data" / "flash" / "manifest.tsv"
    if sibling.exists():
        return sibling
    return local


def default_corpus_root(manifest: Path) -> Path:
    # .../<repo>/gguf-tools/quality-testing/data/flash/manifest.tsv
    try:
        return manifest.parents[4]
    except IndexError as error:  # pragma: no cover - impossible for normal paths
        raise SystemExit(f"cannot infer corpus root from {manifest}") from error


def resolve_reference_path(path_text: str, corpus_root: Path, manifest: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    from_root = corpus_root / path
    if from_root.exists():
        return from_root
    from_manifest = manifest.parent / path
    if from_manifest.exists():
        return from_manifest
    return from_root


def read_manifest(manifest: Path, corpus_root: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest.open(encoding="utf-8", newline="") as fp:
        for raw in fp:
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise SystemExit(f"bad manifest row: {line!r}")
            case_id, prompt_path, continuation_path = fields[:3]
            row = {
                "id": case_id,
                "prompt_path": resolve_reference_path(prompt_path, corpus_root, manifest),
                "continuation_path": resolve_reference_path(continuation_path, corpus_root, manifest),
                "response_path": resolve_reference_path(fields[3], corpus_root, manifest) if len(fields) >= 4 else None,
            }
            for key in ("prompt_path", "continuation_path"):
                if not row[key].is_file():
                    raise SystemExit(f"{case_id}: missing {key}: {row[key]}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no reference rows found in {manifest}")
    return rows


def require_tokens(response: Optional[Dict[str, Any]], label: str) -> List[int]:
    if not isinstance(response, dict) or not isinstance(response.get("tokens"), list):
        raise RuntimeError(f"{label}: /tokenize response has no tokens")
    tokens = response["tokens"]
    if not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"{label}: /tokenize returned non-integer token IDs")
    return list(tokens)


def tokenize(
    base_url: str,
    payload: Dict[str, Any],
    timeout: float,
    label: str,
) -> Tuple[List[int], Dict[str, Any]]:
    status, response, raw, _ = request_json(base_url + "/tokenize", payload=payload, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"{label}: /tokenize HTTP {status}: {raw[:400]}")
    tokens = require_tokens(response, label)
    return tokens, response if isinstance(response, dict) else {}


def tokenize_case(base_url: str, model: str, prompt: str, continuation: str, timeout: float, case_id: str) -> Dict[str, Any]:
    template = {"chat_template_kwargs": {"enable_thinking": False}}
    prefix, prefix_raw = tokenize(
        base_url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            **template,
        },
        timeout,
        case_id + " prefix",
    )
    full, full_raw = tokenize(
        base_url,
        {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": continuation},
            ],
            "add_generation_prompt": False,
            "continue_final_message": True,
            **template,
        },
        timeout,
        case_id + " full",
    )
    target, target_raw = tokenize(
        base_url,
        {"model": model, "prompt": continuation, "add_special_tokens": False},
        timeout,
        case_id + " continuation",
    )
    if full[: len(prefix)] != prefix:
        raise RuntimeError(f"{case_id}: full chat tokenization does not begin with the generation prefix")
    suffix = full[len(prefix) :]
    if suffix[: len(target)] != target:
        raise RuntimeError(
            f"{case_id}: assistant continuation tokenization differs from standalone continuation tokenization"
        )
    terminal = suffix[len(target) :]
    if len(terminal) > 1:
        raise RuntimeError(f"{case_id}: unexpected chat-template terminal tokens: {terminal}")
    return {
        "prefix_tokens": prefix,
        "target_tokens": target,
        "template_terminal_tokens": terminal,
        "tokenize_raw": {"prefix": prefix_raw, "full": full_raw, "target": target_raw},
    }


def greedy_completion(
    base_url: str,
    model: str,
    prompt: str,
    target_token_count: int,
    timeout: float,
    case_id: str,
) -> Tuple[List[int], Dict[str, Any], float]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max(1, target_token_count),
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, response, raw, elapsed_ms = request_json(base_url + "/v1/chat/completions", payload=payload, timeout=timeout)
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"{case_id}: greedy generation HTTP {status}: {raw[:400]}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"{case_id}: greedy generation had no choice")
    token_ids = choices[0].get("token_ids")
    if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
        raise RuntimeError(f"{case_id}: greedy generation did not return token_ids")
    # The API includes a terminal EOS in token_ids; the C scorer compares only
    # the stored continuation tokens, which never include it.
    while token_ids and token_ids[-1] == 1:
        token_ids.pop()
    return token_ids, response, elapsed_ms


def score_next_token(
    base_url: str,
    model: str,
    context_tokens: Sequence[int],
    target_token: int,
    timeout: float,
    label: str,
) -> Tuple[float, float]:
    # A single item per call is intentional.  It makes the exact input
    # sequence visible in the trace and avoids micro-batch-dependent FP8 score
    # rounding when comparing runs.
    payload = {
        "model": model,
        "query": [],
        "items": [list(context_tokens)],
        "label_token_ids": [target_token],
        "apply_softmax": False,
        "add_special_tokens": False,
    }
    status, response, raw, elapsed_ms = request_json(base_url + "/generative_scoring", payload=payload, timeout=timeout)
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"{label}: /generative_scoring HTTP {status}: {raw[:400]}")
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise RuntimeError(f"{label}: unexpected /generative_scoring response")
    score = data[0].get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0.0 or score > 1.000001:
        raise RuntimeError(f"{label}: invalid target probability: {score!r}")
    return float(score), elapsed_ms


def lcp(left: Sequence[int], right: Sequence[int]) -> int:
    result = 0
    for a, b in zip(left, right):
        if a != b:
            break
        result += 1
    return result


def capture_metrics(base_url: str, timeout: float, path: Path) -> Dict[str, float]:
    status, raw, _ = request_text(base_url + "/metrics", timeout=timeout)
    write_text(path, raw)
    if status != 200:
        raise RuntimeError(f"GET /metrics HTTP {status}")
    return parse_prometheus(raw)


def phase_summary(delta: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
    latency_names = {
        "ttft": "vllm:time_to_first_token_seconds",
        "inter_token": "vllm:inter_token_latency_seconds",
        "e2e": "vllm:e2e_request_latency_seconds",
        "prefill": "vllm:request_prefill_time_seconds",
        "decode": "vllm:request_decode_time_seconds",
    }
    latency: Dict[str, Dict[str, Optional[float]]] = {}
    for label, prefix in latency_names.items():
        total_seconds = metric_change(delta, prefix + "_sum")
        count = metric_change(delta, prefix + "_count")
        latency[label] = {"sum_seconds": total_seconds, "count": count, "mean_ms": mean_ms(total_seconds, count)}
    draft = metric_change(delta, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_change(delta, "vllm:spec_decode_num_accepted_tokens_total")
    return {
        "requests_success": metric_change(delta, "vllm:request_success_total"),
        "requests_error": metric_change(delta, "vllm:request_error_total"),
        "prompt_tokens": metric_change(delta, "vllm:prompt_tokens_total"),
        "generation_tokens": metric_change(delta, "vllm:generation_tokens_total"),
        "prefix_cache_queries": metric_change(delta, "vllm:prefix_cache_queries_total"),
        "prefix_cache_hits": metric_change(delta, "vllm:prefix_cache_hits_total"),
        "latency": latency,
        "dspark": {
            "draft_blocks": metric_change(delta, "vllm:spec_decode_num_drafts_total"),
            "draft_tokens": draft,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / draft if draft else None,
        },
    }


def format_number(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(output_dir: Path, suite: Dict[str, Any]) -> None:
    quality = suite["quality"]
    generation = suite["server_metrics"]["greedy_generation"]
    scoring = suite["server_metrics"]["teacher_forced_scoring"]
    report = [
        "# DSpark API quality report",
        "",
        f"- Target server: `{suite['base_url']}`",
        f"- Target model: `{suite['target_model']}`",
        f"- Reference corpus: saved `{suite['reference_model']}` continuations from `{suite['manifest']}`",
        "- Hosted DeepSeek API calls: **0** (the saved corpus was reused)",
        f"- Started / finished: `{suite['started_at']}` / `{suite['finished_at']}`",
        "",
        "## Quality summary",
        "",
        "```text",
        quality["summary_line"],
        f"ppl={quality['ppl']:.6f} bits_per_token={quality['bits_per_token']:.6f}",
        "```",
        "",
        "`avg_nll` is corpus-weighted teacher-forced negative log likelihood of each saved",
        "Flash continuation token under the running Abliterated DSpark model.  `ppl = exp(avg_nll)`;",
        "lower is better.  `first_match` and `avg_lcp` use a temperature-zero greedy completion",
        "from the same chat prefix, matching the semantics of `score_official.c`.",
        "",
        "## Captured server statistics",
        "",
        "All raw Prometheus snapshots and their deltas are retained in this run directory.",
        "The two phases are separated because teacher-forced scoring is not ordinary decode traffic.",
        "",
        "| Phase | API calls | Server successes | Prompt tokens | Generated tokens | Mean TTFT | Mean E2E | DSpark acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Greedy continuations | %d | %s | %s | %s | %s ms | %s ms | %s |"
        % (
            suite["greedy_requests"],
            format_number(generation["requests_success"], 0),
            format_number(generation["prompt_tokens"], 0),
            format_number(generation["generation_tokens"], 0),
            format_number(generation["latency"]["ttft"]["mean_ms"]),
            format_number(generation["latency"]["e2e"]["mean_ms"]),
            format_number(
                None if generation["dspark"]["acceptance_rate"] is None else 100.0 * generation["dspark"]["acceptance_rate"]
            )
            + "%",
        ),
        "| Teacher-forced NLL | %d | %s | %s | %s | %s ms | %s ms | %s |"
        % (
            suite["scoring_requests"],
            format_number(scoring["requests_success"], 0),
            format_number(scoring["prompt_tokens"], 0),
            format_number(scoring["generation_tokens"], 0),
            format_number(scoring["latency"]["ttft"]["mean_ms"]),
            format_number(scoring["latency"]["e2e"]["mean_ms"]),
            format_number(None if scoring["dspark"]["acceptance_rate"] is None else 100.0 * scoring["dspark"]["acceptance_rate"])
            + "%",
        ),
        "",
        "## Artifacts",
        "",
        "- `scores.tsv`: compatible with `compare_scores.py` and the local C scorer's columns.",
        "- `cases/<id>.json`: prompt/target tokenization, greedy response, and every target probability.",
        "- `metrics.*.prom` and `metrics.*.delta.json`: complete server counter snapshots and deltas.",
        "- `summary.json`: machine-readable complete summary.",
        "",
    ]
    write_text(output_dir / "report.md", "\n".join(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Target OpenAI-compatible API base URL")
    parser.add_argument("--model", help="Target model; defaults to the first /v1/models result")
    parser.add_argument("--manifest", type=Path, default=default_manifest(), help="Saved official continuation manifest")
    parser.add_argument("--corpus-root", type=Path, help="Root used to resolve relative manifest paths")
    parser.add_argument("--output-dir", type=Path, help="Directory for this captured run")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases (validation only)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    base_url = args.base_url.rstrip("/")
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise SystemExit(
            f"reference manifest not found: {manifest}\n"
            "Pass --manifest with the saved DeepSeek Flash corpus; this tool will not recollect it from the API."
        )
    corpus_root = (args.corpus_root.resolve() if args.corpus_root else default_corpus_root(manifest))
    cases = read_manifest(manifest, corpus_root, args.limit)
    output_dir = args.output_dir or ROOT / "runs" / ("dspark-api-quality-" + path_timestamp())
    output_dir.mkdir(parents=True, exist_ok=False)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()

    model_status, models, models_raw, _ = request_json(base_url + "/v1/models", timeout=args.timeout)
    write_text(output_dir / "models.raw.json", models_raw)
    if model_status != 200 or not isinstance(models, dict):
        raise SystemExit(f"GET /v1/models failed with HTTP {model_status}")
    model = args.model
    if not model:
        data = models.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not isinstance(data[0].get("id"), str):
            raise SystemExit("/v1/models did not return a model id")
        model = data[0]["id"]

    version_status, version, version_raw, _ = request_json(base_url + "/version", timeout=args.timeout)
    write_text(output_dir / "version.raw.json", version_raw)
    metrics_before = capture_metrics(base_url, args.timeout, output_dir / "metrics.before.prom")

    suite: Dict[str, Any] = {
        "schema": "ds4-api-quality-v1",
        "started_at": utc_timestamp(),
        "base_url": base_url,
        "target_model": model,
        "manifest": str(manifest),
        "corpus_root": str(corpus_root),
        "reference_model": "deepseek-v4-flash",
        "hosted_reference_api_calls": 0,
        "server": {"models_status": model_status, "version_status": version_status, "version": version},
        "cases": [],
    }

    total_nll = 0.0
    total_tokens = 0
    total_lcp = 0
    first_matches = 0
    scoring_requests = 0

    # Phase 1: greedy generations.  This exactly supplies the C scorer's
    # first-match and LCP metrics, but exposes the full HTTP response too.
    for index, row in enumerate(cases, 1):
        case_id = row["id"]
        prompt = row["prompt_path"].read_text(encoding="utf-8")
        continuation = row["continuation_path"].read_text(encoding="utf-8")
        tokenized = tokenize_case(base_url, model, prompt, continuation, args.timeout, case_id)
        target_tokens = tokenized["target_tokens"]
        generated_tokens, greedy_raw, greedy_wall_ms = greedy_completion(
            base_url, model, prompt, len(target_tokens), args.timeout, case_id
        )
        first_match = bool(target_tokens and generated_tokens and target_tokens[0] == generated_tokens[0])
        greedy_lcp = lcp(target_tokens, generated_tokens)
        case: Dict[str, Any] = {
            "id": case_id,
            "prompt_file": str(row["prompt_path"]),
            "continuation_file": str(row["continuation_path"]),
            "prompt_tokens": len(tokenized["prefix_tokens"]),
            "target_tokens": len(target_tokens),
            "prefix_token_ids": tokenized["prefix_tokens"],
            "target_token_ids": target_tokens,
            "template_terminal_token_ids": tokenized["template_terminal_tokens"],
            "greedy_token_ids": generated_tokens,
            "first_match": int(first_match),
            "greedy_lcp": greedy_lcp,
            "greedy_wall_ms": greedy_wall_ms,
            "greedy_response": greedy_raw,
            "tokenize_responses": tokenized["tokenize_raw"],
            "target_scores": [],
        }
        suite["cases"].append(case)
        first_matches += int(first_match)
        total_lcp += greedy_lcp
        print(
            f"greedy {index}/{len(cases)} {case_id}: target={len(target_tokens)} first={int(first_match)} lcp={greedy_lcp}",
            flush=True,
        )

    metrics_after_greedy = capture_metrics(base_url, args.timeout, output_dir / "metrics.after-greedy.prom")
    greedy_delta = prometheus_delta(metrics_before, metrics_after_greedy)
    write_json(output_dir / "metrics.greedy.delta.json", greedy_delta)

    # Phase 2: teacher-forced probability of every saved reference token.
    for case_index, case in enumerate(suite["cases"], 1):
        prefix_tokens = case["prefix_token_ids"]
        target_tokens = case["target_token_ids"]
        nll = 0.0
        for position, target_token in enumerate(target_tokens):
            probability, wall_ms = score_next_token(
                base_url,
                model,
                prefix_tokens + target_tokens[:position],
                target_token,
                args.timeout,
                f"{case['id']} token {position}",
            )
            token_nll = -math.log(probability)
            case["target_scores"].append(
                {
                    "position": position,
                    "token_id": target_token,
                    "probability": probability,
                    "nll": token_nll,
                    "wall_ms": wall_ms,
                }
            )
            nll += token_nll
            scoring_requests += 1
        case["nll"] = nll
        case["avg_nll"] = nll / len(target_tokens) if target_tokens else 0.0
        total_nll += nll
        total_tokens += len(target_tokens)
        write_json(cases_dir / f"{case['id']}.json", case)
        print(
            f"score {case_index}/{len(cases)} {case['id']}: target={len(target_tokens)} "
            f"avg_nll={case['avg_nll']:.6f} lcp={case['greedy_lcp']}",
            flush=True,
        )

    metrics_after_scoring = capture_metrics(base_url, args.timeout, output_dir / "metrics.after-scoring.prom")
    scoring_delta = prometheus_delta(metrics_after_greedy, metrics_after_scoring)
    total_delta = prometheus_delta(metrics_before, metrics_after_scoring)
    write_json(output_dir / "metrics.scoring.delta.json", scoring_delta)
    write_json(output_dir / "metrics.total.delta.json", total_delta)

    scores_path = output_dir / "scores.tsv"
    with scores_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t", lineterminator="\n")
        writer.writerow(("id", "prompt_tokens", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp"))
        for case in suite["cases"]:
            writer.writerow(
                (
                    case["id"],
                    case["prompt_tokens"],
                    case["target_tokens"],
                    f"{case['nll']:.9f}",
                    f"{case['avg_nll']:.9f}",
                    case["first_match"],
                    case["greedy_lcp"],
                )
            )

    avg_nll = total_nll / total_tokens if total_tokens else 0.0
    summary_line = (
        f"summary cases={len(suite['cases'])} tokens={total_tokens} avg_nll={avg_nll:.9f} "
        f"first_match={first_matches} avg_lcp={total_lcp / len(suite['cases']):.3f}"
    )
    suite["finished_at"] = utc_timestamp()
    suite["greedy_requests"] = len(suite["cases"])
    suite["scoring_requests"] = scoring_requests
    suite["quality"] = {
        "cases": len(suite["cases"]),
        "tokens": total_tokens,
        "nll": total_nll,
        "avg_nll": avg_nll,
        "ppl": math.exp(avg_nll),
        "bits_per_token": avg_nll / math.log(2.0),
        "first_match": first_matches,
        "avg_lcp": total_lcp / len(suite["cases"]),
        "summary_line": summary_line,
    }
    suite["server_metrics"] = {
        "greedy_generation": phase_summary(greedy_delta),
        "teacher_forced_scoring": phase_summary(scoring_delta),
        "whole_run": phase_summary(total_delta),
    }
    write_json(output_dir / "summary.json", suite)
    write_report(output_dir, suite)
    print(summary_line)
    print(f"report {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
