#!/usr/bin/env python3
"""Dependency-free live dashboard for ds4-server's native /metrics route.

The browser talks only to this same-origin adapter.  The adapter keeps the
inference endpoint private by default and samples it at most twice per second,
regardless of how many dashboard tabs are open.
"""

from __future__ import annotations

import atexit
import hmac
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


APP_DIR = Path(__file__).resolve().parent
METRICS_URL = os.environ.get("DS4_METRICS_URL", "http://127.0.0.1:8000/metrics")
BIND_ADDRESS = os.environ.get("DASHBOARD_BIND", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "11002"))
POLL_CACHE_SECONDS = float(os.environ.get("DASHBOARD_POLL_CACHE_SECONDS", "0.45"))
HARDWARE_CACHE_SECONDS = float(os.environ.get("DASHBOARD_HARDWARE_CACHE_SECONDS", "5"))
CONTROL_TOKEN = os.environ.get("DASHBOARD_CONTROL_TOKEN", "")
SERVER_COMMAND = os.environ.get("DASHBOARD_SERVER_COMMAND", "")
SERVER_WORKDIR = os.environ.get("DASHBOARD_SERVER_WORKDIR", str(APP_DIR.parent))
SERVER_AUTOSTART = os.environ.get("DASHBOARD_SERVER_AUTOSTART", "").lower() in {
    "1",
    "true",
    "yes",
}
SERVER_STOP_TIMEOUT = float(os.environ.get("DASHBOARD_SERVER_STOP_GRACE_SECONDS", "10"))
SERVER_LOG_PATH = os.environ.get("DASHBOARD_SERVER_LOG", "/tmp/ds4-dashboard-server.log")
ENABLE_POWERMETRICS = os.environ.get("DS4_DASHBOARD_POWERMETRICS", "").lower() in {
    "1",
    "true",
    "yes",
}


def resolve_agent_api_base_url(metrics_url: str, configured: Optional[str] = None) -> str:
    """Choose a copyable /v1 endpoint without trusting browser loopback."""

    candidate = (configured or "").strip()
    if not candidate:
        parts = urlsplit(metrics_url)
        candidate = urlunsplit((parts.scheme, parts.netloc, "/v1", "", ""))
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return "http://127.0.0.1:8000/v1"
    path = parts.path.rstrip("/")
    if not path or path == "/metrics":
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


AGENT_API_BASE_URL = resolve_agent_api_base_url(
    METRICS_URL, os.environ.get("DS4_AGENT_API_BASE_URL")
)

METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)$"
)
LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
POWER = re.compile(
    r"^\s*(CPU|GPU|ANE)\s+Power:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b",
    re.IGNORECASE | re.MULTILINE,
)

COUNTER_METRICS = {
    "ds4_generation_tokens_total": "generated_tokens",
    "ds4_prompt_tokens_total": "prompt_tokens",
    "ds4_prefill_chunk_tokens_total": "prefill_chunk_tokens",
    "ds4_prefill_chunk_seconds_total": "prefill_chunk_seconds",
    "ds4_prefill_chunks_total": "prefill_chunks",
    "ds4_requests_total": "requests_total",
    "ds4_request_success_total": "completed_requests",
    "ds4_request_error_total": "errors",
    "ds4_time_to_first_token_seconds_sum": "ttft_sum",
    "ds4_time_to_first_token_seconds_count": "ttft_count",
    "ds4_inter_token_latency_seconds_sum": "itl_sum",
    "ds4_inter_token_latency_seconds_count": "itl_count",
    "ds4_e2e_request_latency_seconds_sum": "e2e_sum",
    "ds4_e2e_request_latency_seconds_count": "e2e_count",
    "ds4_dspark_drafted_tokens_total": "dspark_drafted_tokens",
    "ds4_dspark_accepted_tokens_total": "dspark_accepted_tokens",
    "ds4_dspark_blocks_total": "dspark_blocks",
    "ds4_dspark_skip_pre_draft_total": "dspark_skip_pre_draft",
    "ds4_dspark_skip_verify_total": "dspark_skip_verify",
}
GAUGE_METRICS = {
    "ds4_num_requests_running": "running",
    "ds4_num_requests_waiting": "waiting",
    "ds4_dspark_draft_enabled": "dspark_draft_enabled",
    "ds4_dspark_dynamic_verify": "dspark_dynamic_verify",
    "ds4_dspark_active_verify_budget": "dspark_active_verify_budget",
    "ds4_runtime_status_available": "runtime_status_available",
    "ds4_moe_slot_bank": "moe_slot_bank",
    "ds4_memory_process_resident_bytes": "process_resident_bytes",
    "ds4_memory_process_phys_footprint_bytes": "process_phys_footprint_bytes",
    "ds4_memory_gpu_footprint_bytes": "gpu_footprint_bytes",
    "ds4_memory_gpu_compressed_bytes": "gpu_compressed_bytes",
    "ds4_memory_task_compressed_bytes": "task_compressed_bytes",
    "ds4_memory_system_total_bytes": "system_memory_total_bytes",
    "ds4_memory_system_free_bytes": "system_memory_free_bytes",
    "ds4_memory_system_active_bytes": "system_memory_active_bytes",
    "ds4_memory_system_wired_bytes": "system_memory_wired_bytes",
    "ds4_memory_system_compressed_bytes": "system_compressed_bytes",
    "ds4_memory_system_compressor_bytes": "system_compressor_bytes",
    "ds4_memory_system_pressure_percent": "system_memory_pressure_percent",
    "ds4_memory_swap_total_bytes": "swap_total_bytes",
    "ds4_memory_swap_used_bytes": "swap_used_bytes",
}


def _empty_metrics() -> Dict[str, Any]:
    return {
        "model": "DeepSeek V4 Flash",
        "backend": "DS4",
        "generated_tokens": 0.0,
        "prompt_tokens": 0.0,
        "prefill_chunk_tokens": 0.0,
        "prefill_chunk_seconds": 0.0,
        "prefill_chunks": 0.0,
        "requests_total": 0.0,
        "completed_requests": 0.0,
        "errors": 0.0,
        "ttft_sum": 0.0,
        "ttft_count": 0.0,
        "itl_sum": 0.0,
        "itl_count": 0.0,
        "e2e_sum": 0.0,
        "e2e_count": 0.0,
        "running": 0.0,
        "waiting": 0.0,
        "dspark_metrics_available": False,
        "runtime_metrics_available": False,
        "dspark_drafted_tokens": 0.0,
        "dspark_accepted_tokens": 0.0,
        "dspark_blocks": 0.0,
        "dspark_skip_pre_draft": 0.0,
        "dspark_skip_verify": 0.0,
        "dspark_draft_enabled": 0.0,
        "dspark_dynamic_verify": 0.0,
        "dspark_active_verify_budget": 0.0,
        "runtime_status_available": 0.0,
        "moe_slot_bank": 0.0,
        "process_resident_bytes": 0.0,
        "process_phys_footprint_bytes": 0.0,
        "gpu_footprint_bytes": 0.0,
        "gpu_compressed_bytes": 0.0,
        "task_compressed_bytes": 0.0,
        "system_memory_total_bytes": 0.0,
        "system_memory_free_bytes": 0.0,
        "system_memory_active_bytes": 0.0,
        "system_memory_wired_bytes": 0.0,
        "system_compressed_bytes": 0.0,
        "system_compressor_bytes": 0.0,
        "system_memory_pressure_percent": 0.0,
        "swap_total_bytes": 0.0,
        "swap_used_bytes": 0.0,
    }


def _labels(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    return {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape")
        for match in LABEL.finditer(raw)
    }


def parse_prometheus(payload: str) -> Dict[str, Any]:
    metrics = _empty_metrics()
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = _labels(raw_labels)
        if name == "ds4_server_info":
            metrics["model"] = labels.get("model_name", metrics["model"])
            metrics["backend"] = labels.get("backend", metrics["backend"])
            continue
        if name.startswith("ds4_dspark_"):
            metrics["dspark_metrics_available"] = True
        if name.startswith("ds4_memory_") or name in {
            "ds4_runtime_status_available",
            "ds4_moe_slot_bank",
        }:
            metrics["runtime_metrics_available"] = True
        counter_key = COUNTER_METRICS.get(name)
        if counter_key:
            metrics[counter_key] += value
            continue
        gauge_key = GAUGE_METRICS.get(name)
        if gauge_key:
            metrics[gauge_key] += value
    return metrics


def summarize_dspark(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return the same acceptance/tau view that --debug-status reports."""

    drafted = max(0.0, float(metrics["dspark_drafted_tokens"]))
    accepted = min(drafted, max(0.0, float(metrics["dspark_accepted_tokens"])))
    blocks = max(0.0, float(metrics["dspark_blocks"]))
    return {
        "available": bool(metrics["dspark_metrics_available"]),
        "enabled": bool(metrics["dspark_draft_enabled"]),
        "draftedTokens": drafted,
        "acceptedTokens": accepted,
        "blocks": blocks,
        "skipPreDraft": max(0.0, float(metrics["dspark_skip_pre_draft"])),
        "skipVerify": max(0.0, float(metrics["dspark_skip_verify"])),
        "acceptancePct": None if drafted <= 0 else 100.0 * accepted / drafted,
        "tau": None if blocks <= 0 else accepted / blocks,
        "dynamic": bool(metrics["dspark_dynamic_verify"]),
        "activeVerifyBudget": int(max(0.0, metrics["dspark_active_verify_budget"])),
    }


def summarize_memory(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the last worker-side unified-memory snapshot for the UI."""

    total = max(0.0, float(metrics["system_memory_total_bytes"]))
    free = min(total, max(0.0, float(metrics["system_memory_free_bytes"])))
    return {
        "available": bool(metrics["runtime_metrics_available"])
        and bool(metrics["runtime_status_available"]),
        "moeSlots": max(0.0, float(metrics["moe_slot_bank"])),
        "processResidentBytes": max(0.0, float(metrics["process_resident_bytes"])),
        "processFootprintBytes": max(0.0, float(metrics["process_phys_footprint_bytes"])),
        "gpuFootprintBytes": max(0.0, float(metrics["gpu_footprint_bytes"])),
        "gpuCompressedBytes": max(0.0, float(metrics["gpu_compressed_bytes"])),
        "taskCompressedBytes": max(0.0, float(metrics["task_compressed_bytes"])),
        "systemTotalBytes": total,
        "systemFreeBytes": free,
        "systemUsedBytes": max(0.0, total - free),
        "systemActiveBytes": max(0.0, float(metrics["system_memory_active_bytes"])),
        "systemWiredBytes": max(0.0, float(metrics["system_memory_wired_bytes"])),
        "systemCompressedBytes": max(0.0, float(metrics["system_compressed_bytes"])),
        "systemCompressorBytes": max(0.0, float(metrics["system_compressor_bytes"])),
        "pressurePct": max(0.0, float(metrics["system_memory_pressure_percent"])),
        "swapTotalBytes": max(0.0, float(metrics["swap_total_bytes"])),
        "swapUsedBytes": max(0.0, float(metrics["swap_used_bytes"])),
    }


class HardwareSampler:
    """Static Apple Silicon identity plus explicitly opt-in power estimates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._last_fetch = 0.0

    @staticmethod
    def _platform() -> Tuple[str, Optional[int]]:
        try:
            result = subprocess.run(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "Apple Silicon", None
        chip = re.search(r"^\s*(Apple [^:\n]+):\s*$", result.stdout, re.MULTILINE)
        cores = re.search(r"^\s*Total Number of Cores:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
        return chip.group(1) if chip else "Apple Silicon", int(cores.group(1)) if cores else None

    @staticmethod
    def _power() -> Tuple[Dict[str, Optional[float]], str]:
        if not ENABLE_POWERMETRICS:
            return {"cpuPowerW": None, "gpuPowerW": None, "anePowerW": None}, (
                "Power estimates are off (set DS4_DASHBOARD_POWERMETRICS=1 only with a "
                "non-interactive powermetrics privilege rule)."
            )
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    "/usr/bin/powermetrics",
                    "-n",
                    "1",
                    "-i",
                    "1000",
                    "-s",
                    "cpu_power,gpu_power,ane_power,thermal",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"cpuPowerW": None, "gpuPowerW": None, "anePowerW": None}, (
                "Power telemetry is enabled but powermetrics could not run without a password."
            )
        values: Dict[str, Optional[float]] = {"cpuPowerW": None, "gpuPowerW": None, "anePowerW": None}
        for rail, raw_value, unit in POWER.findall(result.stdout):
            watts = float(raw_value) / 1000.0 if unit.lower() == "mw" else float(raw_value)
            values[f"{rail.lower()}PowerW"] = watts
        reported = [value for value in values.values() if value is not None]
        status = "Estimated CPU, GPU, and ANE rail power from powermetrics." if reported else (
            "powermetrics returned no CPU/GPU/ANE rail estimates on this macOS build."
        )
        return values, status

    def _fetch(self) -> Dict[str, Any]:
        platform, gpu_cores = self._platform()
        power, status = self._power()
        reported_power = [value for value in power.values() if value is not None]
        return {
            "available": True,
            "platform": platform,
            "gpuCores": gpu_cores,
            "powerEnabled": ENABLE_POWERMETRICS,
            **power,
            "combinedPowerW": sum(reported_power) if reported_power else None,
            "status": status,
            "sampledAt": datetime.now(timezone.utc).isoformat(),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._latest and now - self._last_fetch < HARDWARE_CACHE_SECONDS:
                return self._latest
            self._latest = self._fetch()
            self._last_fetch = now
            return self._latest


HARDWARE_SAMPLER = HardwareSampler()


class ServerController:
    """Own one opt-in ds4-server child using a fixed, shell-free argv config."""

    def __init__(self, command: str = SERVER_COMMAND, cwd: str = SERVER_WORKDIR) -> None:
        self._lock = threading.RLock()
        self._command = command.strip()
        self._cwd = cwd.strip() or str(APP_DIR.parent)
        self._process: Optional[Any] = None
        self._log_file: Optional[Any] = None
        self._stopping = False
        self._last_exit_code: Optional[int] = None
        self._last_action = "not configured" if not self._command else "configured"
        self._last_started_at: Optional[str] = None

    @staticmethod
    def _validate(command: str, cwd: str) -> Tuple[list[str], str]:
        if not command:
            raise ValueError("set DASHBOARD_SERVER_COMMAND before starting a server")
        try:
            raw_argv = json.loads(command)
        except json.JSONDecodeError as exc:
            raise ValueError("DASHBOARD_SERVER_COMMAND must be a JSON argv array") from exc
        if not isinstance(raw_argv, list) or not raw_argv or not all(
            isinstance(part, str) and part and "\x00" not in part for part in raw_argv
        ):
            raise ValueError("DASHBOARD_SERVER_COMMAND must be a non-empty JSON string array")
        argv = [os.path.expanduser(part) for part in raw_argv]
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("DASHBOARD_SERVER_COMMAND must begin with an executable absolute path")
        if "\x00" in cwd:
            raise ValueError("DASHBOARD_SERVER_CWD cannot contain NUL bytes")
        workdir = Path(os.path.expanduser(cwd or str(APP_DIR.parent)))
        if not workdir.is_dir():
            raise ValueError(f"DASHBOARD_SERVER_CWD does not exist: {workdir}")
        return argv, str(workdir.resolve())

    def _close_log_locked(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _refresh_locked(self) -> None:
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._last_exit_code = exit_code
        self._last_action = f"server exited with code {exit_code}"
        self._process = None
        self._stopping = False
        self._close_log_locked()

    def _snapshot_locked(self) -> Dict[str, Any]:
        self._refresh_locked()
        process = self._process
        return {
            "controlEnabled": bool(CONTROL_TOKEN),
            "configured": bool(self._command),
            "running": process is not None and process.poll() is None,
            "stopping": self._stopping,
            "pid": None if process is None else process.pid,
            "startedAt": self._last_started_at,
            "lastExitCode": self._last_exit_code,
            "lastAction": self._last_action,
        }

    def public_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def is_authorized(self, token: Optional[str]) -> bool:
        return bool(CONTROL_TOKEN) and hmac.compare_digest(token or "", CONTROL_TOKEN)

    def start(self) -> Tuple[bool, str, Dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            if self._stopping:
                return False, "the managed server is stopping", self._snapshot_locked()
            if self._process is not None:
                return False, "the dashboard-managed server is already running", self._snapshot_locked()
            try:
                argv, workdir = self._validate(self._command, self._cwd)
                log_path = Path(os.path.expanduser(SERVER_LOG_PATH))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                self._log_file = os.fdopen(descriptor, "ab", buffering=0)
                child_env = os.environ.copy()
                for name in (
                    "DASHBOARD_CONTROL_TOKEN",
                    "DASHBOARD_SERVER_COMMAND",
                    "DASHBOARD_SERVER_CWD",
                    "DASHBOARD_SERVER_AUTOSTART",
                ):
                    child_env.pop(name, None)
                self._process = subprocess.Popen(
                    argv,
                    cwd=workdir,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                self._close_log_locked()
                self._last_action = f"server start failed: {exc}"
                return False, self._last_action, self._snapshot_locked()
            self._last_exit_code = None
            self._last_started_at = datetime.now(timezone.utc).isoformat()
            self._last_action = "server started"
            return True, self._last_action, self._snapshot_locked()

    def stop(self) -> Tuple[bool, str, Dict[str, Any]]:
        with self._lock:
            self._refresh_locked()
            if self._stopping:
                return False, "the managed server is already stopping", self._snapshot_locked()
            process = self._process
            if process is None:
                return False, "no dashboard-managed server is running", self._snapshot_locked()
            self._stopping = True
            self._last_action = "stopping server"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        try:
            process.wait(timeout=max(1.0, SERVER_STOP_TIMEOUT))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            process.wait(timeout=2)
        with self._lock:
            if self._process is process:
                self._last_exit_code = process.poll()
                self._process = None
                self._close_log_locked()
            self._stopping = False
            self._last_action = "server stopped"
            return True, self._last_action, self._snapshot_locked()

    def restart(self) -> Tuple[bool, str, Dict[str, Any]]:
        stopped, _, _ = self.stop()
        started, message, snapshot = self.start()
        if not started:
            return False, message, snapshot
        return True, "server restarted" if stopped else "server started", snapshot

    def shutdown(self) -> None:
        self.stop()


SERVER_CONTROLLER = ServerController()


class MetricsSampler:
    """Reduce DS4's counters into a compact snapshot and 60-second history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previous: Optional[Tuple[Dict[str, Any], float]] = None
        self._latest: Optional[Dict[str, Any]] = None
        self._last_fetch = 0.0
        self._history: Deque[Dict[str, Any]] = deque(maxlen=121)
        self._last_prefill_tps: Optional[float] = None
        self._last_prefill_chunk_seconds: Optional[float] = None

    @staticmethod
    def _rate(current: float, previous: float, elapsed: float) -> Tuple[Optional[float], bool]:
        if elapsed <= 0:
            return None, False
        delta = current - previous
        if delta < 0:
            return 0.0, True
        return delta / elapsed, False

    @staticmethod
    def _recent_mean(current_sum: float, current_count: float,
                     previous_sum: float, previous_count: float) -> Optional[float]:
        count_delta = current_count - previous_count
        sum_delta = current_sum - previous_sum
        if count_delta <= 0 or sum_delta < 0:
            return None
        return sum_delta / count_delta

    def _fetch(self) -> Tuple[Dict[str, Any], float]:
        started = time.monotonic()
        request = urllib.request.Request(
            METRICS_URL,
            headers={"Accept": "text/plain", "User-Agent": "ds4-live-dashboard/1.0"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = response.read().decode("utf-8", errors="replace")
        return parse_prometheus(payload), (time.monotonic() - started) * 1000.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._latest and now - self._last_fetch < POLL_CACHE_SECONDS:
                return self._latest

            hardware = HARDWARE_SAMPLER.snapshot()
            try:
                current, scrape_ms = self._fetch()
            except Exception:
                unavailable = dict(self._latest or {})
                unavailable.update(
                    {
                        "healthy": False,
                        "message": "DS4 /metrics is unavailable — rebuild and restart ds4-server.",
                        "sampledAt": datetime.now(timezone.utc).isoformat(),
                        "history": list(self._history),
                        "hardware": hardware,
                        "agentApiBaseUrl": AGENT_API_BASE_URL,
                        "serverControl": SERVER_CONTROLLER.public_snapshot(),
                    }
                )
                self._latest = unavailable
                self._last_fetch = now
                return unavailable

            previous = self._previous
            generation_tps: Optional[float] = None
            prefill_tps: Optional[float] = None
            prefill_chunk_seconds: Optional[float] = None
            ttft_seconds: Optional[float] = None
            itl_seconds: Optional[float] = None
            e2e_seconds: Optional[float] = None
            counter_reset = False
            if previous:
                old, old_at = previous
                elapsed = now - old_at
                generation_tps, reset = self._rate(
                    current["generated_tokens"], old["generated_tokens"], elapsed
                )
                counter_reset = counter_reset or reset
                _, reset = self._rate(current["prompt_tokens"], old["prompt_tokens"], elapsed)
                counter_reset = counter_reset or reset
                chunk_tokens = current["prefill_chunk_tokens"] - old["prefill_chunk_tokens"]
                chunk_seconds = current["prefill_chunk_seconds"] - old["prefill_chunk_seconds"]
                if chunk_tokens < 0 or chunk_seconds < 0:
                    counter_reset = True
                elif chunk_tokens > 0 and chunk_seconds > 0:
                    prefill_tps = chunk_tokens / chunk_seconds
                    prefill_chunk_seconds = chunk_seconds
                    self._last_prefill_tps = prefill_tps
                    self._last_prefill_chunk_seconds = prefill_chunk_seconds
                ttft_seconds = self._recent_mean(
                    current["ttft_sum"], current["ttft_count"], old["ttft_sum"], old["ttft_count"]
                )
                itl_seconds = self._recent_mean(
                    current["itl_sum"], current["itl_count"], old["itl_sum"], old["itl_count"]
                )
                e2e_seconds = self._recent_mean(
                    current["e2e_sum"], current["e2e_count"], old["e2e_sum"], old["e2e_count"]
                )

            if counter_reset:
                self._history.clear()
                self._last_prefill_tps = None
                self._last_prefill_chunk_seconds = None
            sample_time = int(time.time() * 1000)
            self._history.append(
                {
                    "time": sample_time,
                    "generationTps": generation_tps,
                    "prefillTps": prefill_tps,
                    "running": current["running"],
                    "waiting": current["waiting"],
                }
            )
            if prefill_tps is not None and prefill_chunk_seconds is not None:
                chunk_start = sample_time - int(prefill_chunk_seconds * 1000.0)
                for point in self._history:
                    if point["time"] >= chunk_start:
                        point["prefillTps"] = prefill_tps
            snapshot = {
                "healthy": True,
                "message": "live",
                "sampledAt": datetime.now(timezone.utc).isoformat(),
                "scrapeMs": round(scrape_ms, 1),
                "warmup": previous is None,
                "counterReset": counter_reset,
                "model": current["model"],
                "backend": current["backend"],
                "generationTps": generation_tps,
                "prefillTps": self._last_prefill_tps,
                "prefillChunkSeconds": self._last_prefill_chunk_seconds,
                "running": current["running"],
                "waiting": current["waiting"],
                "generatedTokens": current["generated_tokens"],
                "promptTokens": current["prompt_tokens"],
                "requestsTotal": current["requests_total"],
                "completedRequests": current["completed_requests"],
                "errors": current["errors"],
                "ttftMs": None if ttft_seconds is None else ttft_seconds * 1000.0,
                "itlMs": None if itl_seconds is None else itl_seconds * 1000.0,
                "e2eMs": None if e2e_seconds is None else e2e_seconds * 1000.0,
                "dspark": summarize_dspark(current),
                "memory": summarize_memory(current),
                "hardware": hardware,
                "agentApiBaseUrl": AGENT_API_BASE_URL,
                "serverControl": SERVER_CONTROLLER.public_snapshot(),
                "history": list(self._history),
            }
            self._previous = (current, now)
            self._latest = snapshot
            self._last_fetch = now
            return snapshot


SAMPLER = MetricsSampler()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        if not self.path.startswith("/api/"):
            super().log_message(format, *args)

    def _json(self, value: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _require_control(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and SERVER_CONTROLLER.is_authorized(token):
            return True
        self._json(
            {
                "ok": False,
                "message": "Server control is disabled or the control token is invalid.",
            },
            HTTPStatus.UNAUTHORIZED,
        )
        return False

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/snapshot":
            self._json(SAMPLER.snapshot())
            return
        if path == "/api/server/status":
            self._json(SERVER_CONTROLLER.public_snapshot())
            return
        if path == "/health":
            snapshot = SAMPLER.snapshot()
            self._json({"ok": bool(snapshot.get("healthy")), "message": snapshot.get("message")})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {
            "/api/server/start",
            "/api/server/stop",
            "/api/server/restart",
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._require_control():
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json({"ok": False, "message": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        if content_length != 0:
            self._json({"ok": False, "message": "control actions do not accept a request body"}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/server/start":
            ok, message, snapshot = SERVER_CONTROLLER.start()
        elif path == "/api/server/stop":
            ok, message, snapshot = SERVER_CONTROLLER.stop()
        else:
            ok, message, snapshot = SERVER_CONTROLLER.restart()
        self._json(
            {"ok": ok, "message": message, "server": snapshot},
            HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST,
        )


atexit.register(SERVER_CONTROLLER.shutdown)


def main() -> None:
    if SERVER_AUTOSTART:
        _, message, _ = SERVER_CONTROLLER.start()
        print(f"DS4 dashboard server control: {message}", flush=True)
    server = ThreadingHTTPServer((BIND_ADDRESS, PORT), DashboardHandler)
    server.daemon_threads = True
    print(f"DS4 live dashboard listening on http://{BIND_ADDRESS}:{PORT}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        SERVER_CONTROLLER.shutdown()


if __name__ == "__main__":
    main()
