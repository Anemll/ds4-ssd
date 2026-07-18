#!/usr/bin/env python3
"""Replay stored DeepSeek V4 Flash API vectors against a live OpenAI API.

This intentionally never calls the hosted DeepSeek API. It reuses the
checked-in official JSON as the reference, sends the same requests to the
target server, and saves raw responses plus full /metrics snapshots before and
after each request.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
            return int(response.status), (json.loads(raw) if raw else None), raw, (time.perf_counter() - started) * 1000.0
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


def content_from_choice(choice: Dict[str, Any]) -> str:
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def step_bytes(step: Dict[str, Any]) -> bytes:
    token = step.get("token")
    if isinstance(token, dict):
        values = token.get("bytes")
        if isinstance(values, list) and all(isinstance(value, int) for value in values):
            return bytes(values)
        token = token.get("text")
    values = step.get("bytes")
    if isinstance(values, list) and all(isinstance(value, int) for value in values):
        return bytes(values)
    return token.encode("utf-8") if isinstance(token, str) else b""


def byte_lcp(left: bytes, right: bytes) -> int:
    result = 0
    for a, b in zip(left, right):
        if a != b:
            break
        result += 1
    return result


def score(reference: Dict[str, Any], target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    reference_content = reference.get("message", {}).get("content", "")
    if not isinstance(reference_content, str):
        reference_content = ""
    reference_steps = reference.get("steps", [])
    if not isinstance(reference_steps, list):
        reference_steps = []

    choice: Dict[str, Any] = {}
    if isinstance(target, dict) and isinstance(target.get("choices"), list) and target["choices"]:
        if isinstance(target["choices"][0], dict):
            choice = target["choices"][0]
    target_content = content_from_choice(choice)
    target_steps = choice.get("logprobs", {}).get("content", []) if isinstance(choice.get("logprobs"), dict) else []
    if not isinstance(target_steps, list):
        target_steps = []

    comparable = min(len(reference_steps), len(target_steps))
    matches = 0
    for index in range(comparable):
        if isinstance(reference_steps[index], dict) and isinstance(target_steps[index], dict):
            matches += int(step_bytes(reference_steps[index]) == step_bytes(target_steps[index]))

    reference_bytes = reference_content.encode("utf-8")
    target_bytes = target_content.encode("utf-8")
    first_match: Optional[bool] = None
    if reference_steps and target_steps and isinstance(reference_steps[0], dict) and isinstance(target_steps[0], dict):
        first_match = step_bytes(reference_steps[0]) == step_bytes(target_steps[0])
    return {
        "reference_content": reference_content,
        "target_content": target_content,
        "output_exact_match": reference_bytes == target_bytes,
        "output_lcp_bytes": byte_lcp(reference_bytes, target_bytes),
        "reference_output_bytes": len(reference_bytes),
        "target_output_bytes": len(target_bytes),
        "logprobs_returned": bool(target_steps),
        "reference_steps": len(reference_steps),
        "target_logprob_steps": len(target_steps),
        "compared_token_steps": comparable,
        "selected_token_matches": matches,
        "first_token_match": first_match,
        "finish_reason": choice.get("finish_reason"),
    }


def selected_metrics(delta: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    fragments = (
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "request_error_total",
        "time_to_first_token",
        "inter_token_latency",
        "e2e",
        "prefill",
        "decode",
        "num_draft_tokens",
        "num_accepted_tokens",
        "accepted_tokens_per_pos",
        "prefix_cache",
        "kv_cache",
    )
    return {key: value for key, value in delta.items() if any(fragment in key for fragment in fragments)}


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


def usage_total(cases: List[Dict[str, Any]], field: str) -> int:
    total = 0
    for case in cases:
        usage = case.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get(field), int):
            total += usage[field]
    return total


def suite_statistics(cases: List[Dict[str, Any]], delta: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
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
        latency[label] = {
            "sum_seconds": total_seconds,
            "count": count,
            "mean_ms": mean_ms(total_seconds, count),
        }

    draft_tokens = metric_change(delta, "vllm:spec_decode_num_draft_tokens_total")
    accepted_tokens = metric_change(delta, "vllm:spec_decode_num_accepted_tokens_total")
    draft_blocks = metric_change(delta, "vllm:spec_decode_num_drafts_total")
    exact_outputs = sum(1 for case in cases if case["score"]["output_exact_match"])
    first_token_matches = sum(1 for case in cases if case["score"]["first_token_match"] is True)
    selected_matches = sum(case["score"]["selected_token_matches"] for case in cases)
    selected_total = sum(case["score"]["reference_steps"] for case in cases)
    return {
        "requests": {
            "sent": len(cases),
            "http_200": sum(1 for case in cases if case["http_status"] == 200),
            "server_success": metric_change(delta, "vllm:request_success_total"),
            "server_errors": metric_change(delta, "vllm:request_error_total"),
        },
        "client_usage": {
            "prompt_tokens": usage_total(cases, "prompt_tokens"),
            "completion_tokens": usage_total(cases, "completion_tokens"),
            "total_tokens": usage_total(cases, "total_tokens"),
        },
        "quality": {
            "exact_output_cases": exact_outputs,
            "case_count": len(cases),
            "first_token_matches": first_token_matches,
            "selected_token_matches": selected_matches,
            "reference_token_steps": selected_total,
        },
        "server_token_counters": {
            "prompt_tokens": metric_change(delta, "vllm:prompt_tokens_total"),
            "generation_tokens": metric_change(delta, "vllm:generation_tokens_total"),
            "prefix_cache_queries": metric_change(delta, "vllm:prefix_cache_queries_total"),
            "prefix_cache_hits": metric_change(delta, "vllm:prefix_cache_hits_total"),
        },
        "latency": latency,
        "dspark_speculative": {
            "draft_blocks": draft_blocks,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": (accepted_tokens / draft_tokens) if draft_tokens else None,
            "accepted_tokens_per_block": (accepted_tokens / draft_blocks) if draft_blocks else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Target OpenAI-compatible API base URL")
    parser.add_argument("--model", help="Target model; defaults to the first /v1/models result")
    parser.add_argument("--vectors-dir", type=Path, default=ROOT, help="Directory containing manifest.json and official/")
    parser.add_argument("--output-dir", type=Path, help="Where to write the captured run")
    parser.add_argument("--only", action="append", help="Run only this vector ID (may be repeated)")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    parser.add_argument("--continue-on-error", action="store_true", help="Capture later cases after an HTTP error")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    vectors_dir = args.vectors_dir.resolve()
    manifest = json.loads((vectors_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(args.only or [])
    cases = [case for case in manifest.get("prompts", []) if not wanted or case.get("id") in wanted]
    if not cases:
        raise SystemExit("no matching reference cases")
    output_dir = args.output_dir or vectors_dir / "runs" / ("dspark-api-" + path_timestamp())
    output_dir.mkdir(parents=True, exist_ok=False)

    model_status, models, models_raw, _ = request_json(base_url + "/v1/models", timeout=args.timeout)
    write_text(output_dir / "models.raw.json", models_raw)
    if model_status != 200 or not isinstance(models, dict):
        raise SystemExit("GET /v1/models failed with HTTP %d" % model_status)
    model = args.model
    if not model:
        data = models.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not isinstance(data[0].get("id"), str):
            raise SystemExit("/v1/models did not return a model id")
        model = data[0]["id"]

    version_status, version, version_raw, _ = request_json(base_url + "/version", timeout=args.timeout)
    write_text(output_dir / "version.raw.json", version_raw)
    metrics_status, suite_before_raw, _ = request_text(base_url + "/metrics", timeout=args.timeout)
    write_text(output_dir / "metrics.suite.before.prom", suite_before_raw)
    if metrics_status != 200:
        raise SystemExit("GET /metrics failed with HTTP %d" % metrics_status)
    suite_before = parse_prometheus(suite_before_raw)

    suite: Dict[str, Any] = {
        "schema": "ds4-api-vector-evaluation-v1",
        "started_at": utc_timestamp(),
        "base_url": base_url,
        "target_model": model,
        "reference": {key: manifest.get(key) for key in ("source", "model", "endpoint", "top_logprobs", "max_tokens")},
        "server": {"models_status": model_status, "version_status": version_status, "version": version},
        "cases": [],
    }

    for case_spec in cases:
        case_id = str(case_spec["id"])
        case_dir = output_dir / case_id
        case_dir.mkdir()
        reference = json.loads((vectors_dir / case_spec["official_file"]).read_text(encoding="utf-8"))
        request_payload = dict(reference["request"])
        request_payload["model"] = model
        request_payload["stream"] = False
        write_json(case_dir / "request.json", request_payload)

        before_status, before_raw, _ = request_text(base_url + "/metrics", timeout=args.timeout)
        write_text(case_dir / "metrics.before.prom", before_raw)
        before = parse_prometheus(before_raw) if before_status == 200 else {}
        load_before_status, load_before, load_before_raw, _ = request_json(base_url + "/load", timeout=args.timeout)
        write_text(case_dir / "load.before.raw.json", load_before_raw)

        status, response, response_raw, wall_ms = request_json(
            base_url + "/v1/chat/completions", payload=request_payload, timeout=args.timeout
        )
        write_text(case_dir / "response.raw.json", response_raw)

        load_after_status, load_after, load_after_raw, _ = request_json(base_url + "/load", timeout=args.timeout)
        write_text(case_dir / "load.after.raw.json", load_after_raw)
        after_status, after_raw, _ = request_text(base_url + "/metrics", timeout=args.timeout)
        write_text(case_dir / "metrics.after.prom", after_raw)
        after = parse_prometheus(after_raw) if after_status == 200 else {}
        delta = prometheus_delta(before, after)
        write_json(case_dir / "metrics.delta.json", delta)

        result: Dict[str, Any] = {
            "id": case_id,
            "kind": case_spec.get("kind"),
            "prompt_chars": case_spec.get("prompt_chars"),
            "http_status": status,
            "wall_ms": wall_ms,
            "response_id": response.get("id") if isinstance(response, dict) else None,
            "usage": response.get("usage") if isinstance(response, dict) else None,
            "load_before_status": load_before_status,
            "load_before": load_before,
            "load_after_status": load_after_status,
            "load_after": load_after,
            "metrics_before_status": before_status,
            "metrics_after_status": after_status,
            "metrics_changed": len(delta),
            "metrics_delta_selected": selected_metrics(delta),
            "score": score(reference, response if status == 200 else None),
        }
        write_json(case_dir / "summary.json", result)
        suite["cases"].append(result)
        print("%s: HTTP %d in %.1f ms" % (case_id, status, wall_ms), flush=True)
        if status != 200 and not args.continue_on_error:
            break

    final_status, suite_after_raw, _ = request_text(base_url + "/metrics", timeout=args.timeout)
    write_text(output_dir / "metrics.suite.after.prom", suite_after_raw)
    suite_delta = prometheus_delta(suite_before, parse_prometheus(suite_after_raw) if final_status == 200 else {})
    write_json(output_dir / "metrics.suite.delta.json", suite_delta)
    suite["finished_at"] = utc_timestamp()
    suite["metrics_after_status"] = final_status
    suite["suite_metrics_changed"] = len(suite_delta)
    suite["suite_metrics_delta_selected"] = selected_metrics(suite_delta)
    suite["suite_metric_totals"] = {
        "prompt_tokens": metric_change(suite_delta, "vllm:prompt_tokens_total"),
        "generation_tokens": metric_change(suite_delta, "vllm:generation_tokens_total"),
        "draft_tokens": metric_change(suite_delta, "vllm:spec_decode_num_draft_tokens_total"),
        "accepted_tokens": metric_change(suite_delta, "vllm:spec_decode_num_accepted_tokens_total"),
    }
    suite["summary_statistics"] = suite_statistics(suite["cases"], suite_delta)
    write_json(output_dir / "summary.json", suite)

    report = [
        "# DSv4 Flash API reference replay",
        "",
        "- Target: `%s` / `%s`" % (base_url, model),
        "- Reference: `%s` (%s)" % (manifest.get("model"), manifest.get("endpoint")),
        "- Started: `%s`" % suite["started_at"],
        "- Finished: `%s`" % suite["finished_at"],
        "- No hosted API call was made; the checked-in reference JSON was replayed.",
        "- This is a deterministic continuation/token regression, not an NLL comparison: the stored official alternate logprobs are not a usable distribution.",
        "",
        "| Case | HTTP | Wall ms | Prompt tokens | Completion tokens | Exact output | Token matches | Byte LCP |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for case in suite["cases"]:
        usage = case.get("usage") if isinstance(case.get("usage"), dict) else {}
        case_score = case["score"]
        token_match = "-"
        if case_score["logprobs_returned"]:
            token_match = "%d/%d" % (case_score["selected_token_matches"], case_score["reference_steps"])
        report.append(
            "| {id} | {status} | {wall:.1f} | {prompt} | {completion} | {exact} | {tokens} | {lcp} |".format(
                id=case["id"],
                status=case["http_status"],
                wall=case["wall_ms"],
                prompt=usage.get("prompt_tokens", "-"),
                completion=usage.get("completion_tokens", "-"),
                exact="yes" if case_score["output_exact_match"] else "no",
                tokens=token_match,
                lcp=case_score["output_lcp_bytes"],
            )
        )
    report.extend(
        [
            "",
            "## Captured server telemetry",
            "",
            "Every raw Prometheus sample is retained in the suite and per-case `metrics*.prom` files. The corresponding `metrics*.delta.json` files contain every changed sample, including latency histograms, cache/KV state, and DSpark draft/acceptance counters.",
            "",
            "```json",
            json.dumps(suite["summary_statistics"], indent=2),
            "```",
        ]
    )
    write_text(output_dir / "summary.md", "\n".join(report) + "\n")
    print("wrote %s" % output_dir, flush=True)
    return 0 if len(suite["cases"]) == len(cases) and all(case["http_status"] == 200 for case in suite["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
