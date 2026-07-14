#!/usr/bin/env python3
"""Run a small, fixed DS4 benchmark queue from a host-owned terminal.

This is intentionally not a command executor.  Requests contain only a named
benchmark profile; commands, working directories, and environment overrides are
hard-coded below.  Start ``watch`` yourself from the native host terminal.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_DIR = Path(f"/tmp/ds4-bench-watcher-{os.getuid()}")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "bench-results" / "host-bench-watcher"
METAL_MARKER = "ds4: Metal device Apple M5 Max"

MODEL = "/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major"
DRAFT = "/Users/anemll/Models/DSv4-Flash-DSpark-draft"
A_PROMPT = "Make a game of Space Invader in Pygame"
B_PROMPT = (
    "Create an Asteroids arcade game in a single self-contained HTML file with inline CSS and JavaScript. "
    "Replicate the original arcade experience with screen wrapping, friction-based movement, asteroid splitting, "
    "green vector-style graphics, particle explosions, score, and lives. Save the result to /tmp/si-cv5.html, "
    "test it by opening it in a web browser, and skip web search."
)
A_READY_MAX_BLOCK_MS = 110.0
A_READY_MAX_VERIFY_MS = 96.0


@dataclass(frozen=True)
class Benchmark:
    name: str
    description: str
    env: Mapping[str, str]
    argv: tuple[str, ...]


BENCHMARKS: dict[str, Benchmark] = {
    "a": Benchmark(
        name="a",
        description="Space Invaders / rows6 confidence",
        env={
            "DS4_DSPARK_TRUE_PLAIN_SKIP": "1",
            "DS4_DSPARK_THREE_WAY_LOG": "1",
            "DS4_DSPARK_PERF": "1",
            "DS4_DSPARK_VERIFY_ROWS6": "1",
            "DS4_DSPARK_ROWS6_CONFIDENCE": "1",
            "DS4_DSPARK_CONF_SCALE": "0.85",
            "DS4_DSPARK_CONF_THRESHOLD": "0.50",
        },
        argv=(
            "./ds4",
            "-m",
            MODEL,
            "--draft",
            "dspark",
            "--draft-path",
            DRAFT,
            "--draft-verify",
            "5",
            "--draft-scheduler",
            "confidence",
            "--nothink",
            "--temp",
            "0",
            "--resident",
            "-c",
            "20000",
            "-p",
            A_PROMPT,
        ),
    ),
    "b": Benchmark(
        name="b",
        description="Asteroids / adaptive Mode-B rows6 confidence",
        env={
            "DS4_DSPARK_THREE_WAY_LOG": "1",
            "DS4_DSPARK_FORCE_TARGET_FIRST": "1",
            "DS4_DSPARK_FRONTIER_DRAFT": "1",
            "DS4_DSPARK_VERIFY_ROWS6": "1",
            "DS4_DSPARK_ROWS6_CONFIDENCE": "1",
            "DS4_DSPARK_MODEB_ADAPTIVE": "1",
            "DS4_DSPARK_ROWS6_COST_AWARE": "1",
            "DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER": "0",
            "DS4_DSPARK_CONF_SCALE": "0.85",
            "DS4_DSPARK_CONF_THRESHOLD": "0.50",
        },
        argv=(
            "./ds4",
            "-m",
            MODEL,
            "--draft",
            "dspark",
            "--draft-path",
            DRAFT,
            "--nothink",
            "--temp",
            "0",
            "--resident",
            "-c",
            "20000",
            "-p",
            B_PROMPT,
        ),
    ),
}


PREFILL_RE = re.compile(
    r"ds4: prefill: (?P<prefill>[0-9.]+) t/s, generation: "
    r"(?P<generation>[0-9.]+) t/s \((?P<tokens>\d+) tokens in "
    r"(?P<decode_s>[0-9.]+)s\)"
)
TOTAL_RE = re.compile(r"ds4: ttf: first-output=(?P<first_output>[0-9.]+)s total=(?P<total_s>[0-9.]+)s")
ACCEPTANCE_RE = re.compile(r"ds4: dspark acceptance: (?P<acceptance>[0-9.]+)%")
FULL_ACCEPT_RE = re.compile(r"ds4: dspark full-accept: (?P<full_accept>[0-9.]+)%")
SCHEDULED_RE = re.compile(r"ds4: dspark avg scheduled: (?P<avg_scheduled>[0-9.]+) draft tokens/block")
PERF_RE = re.compile(
    r"ds4: dspark perf: draft=(?P<draft_tps>[0-9.]+) tok/s, "
    r"verify=(?P<verify_tps>[0-9.]+) proposed tok/s, "
    r"verify-accepted=(?P<verify_accepted_tps>[0-9.]+) tok/s, "
    r"block=(?P<block_ms>[0-9.]+) ms \(draft=(?P<draft_ms>[0-9.]+) "
    r"verify=(?P<verify_ms>[0-9.]+) overhead=(?P<overhead_ms>[0-9.]+) "
    r"commit=(?P<commit_ms>[0-9.]+), tau=(?P<tau>[0-9.]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def queue_paths(queue_dir: Path) -> dict[str, Path]:
    return {
        "root": queue_dir,
        "queue": queue_dir / "queue",
        "running": queue_dir / "running",
        "completed": queue_dir / "completed",
        "failed": queue_dir / "failed",
        "pid": queue_dir / "watcher.pid",
        "latest": queue_dir / "latest.json",
    }


def ensure_queue_dirs(paths: Mapping[str, Path]) -> None:
    for name in ("root", "queue", "running", "completed", "failed"):
        paths[name].mkdir(parents=True, exist_ok=True)


def pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class WatcherLock:
    def __init__(self, paths: Mapping[str, Path]):
        self.path = paths["pid"]

    def __enter__(self) -> "WatcherLock":
        if self.path.exists():
            try:
                existing_pid = int(self.path.read_text().strip())
            except ValueError:
                existing_pid = 0
            if existing_pid and pid_is_live(existing_pid):
                raise RuntimeError(f"watcher is already running (pid {existing_pid})")
            self.path.unlink()

        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("could not acquire watcher lock") from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()}\n")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def parse_request(value: object) -> Benchmark:
    if not isinstance(value, dict) or set(value) != {"benchmark"}:
        raise ValueError("request must contain exactly one key: benchmark")
    benchmark_name = value.get("benchmark")
    if not isinstance(benchmark_name, str) or benchmark_name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark_name!r}")
    return BENCHMARKS[benchmark_name]


def read_request(path: Path) -> Benchmark:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    return parse_request(value)


def extract_stats(output: str) -> tuple[dict[str, str], list[str]]:
    stats: dict[str, str] = {}
    for regex in (PREFILL_RE, TOTAL_RE, ACCEPTANCE_RE, FULL_ACCEPT_RE, SCHEDULED_RE, PERF_RE):
        matches = list(regex.finditer(output))
        if matches:
            stats.update(matches[-1].groupdict())

    footer_start = output.rfind("ds4: ----------------------------------------")
    footer = output[footer_start:] if footer_start >= 0 else ""
    footer_lines = [line.rstrip("\r") for line in footer.splitlines() if line.startswith("ds4: ")]
    return stats, footer_lines


def classify_health(benchmark: Benchmark, stats: Mapping[str, str]) -> dict[str, Any]:
    if benchmark.name != "a":
        return {"state": "unclassified", "eligible_for_comparison": None}

    try:
        block_ms = float(stats["block_ms"])
        verify_ms = float(stats["verify_ms"])
    except (KeyError, ValueError):
        return {
            "state": "timing-missing",
            "eligible_for_comparison": False,
            "thresholds": {"max_block_ms": A_READY_MAX_BLOCK_MS, "max_verify_ms": A_READY_MAX_VERIFY_MS},
        }

    ready = block_ms <= A_READY_MAX_BLOCK_MS and verify_ms <= A_READY_MAX_VERIFY_MS
    return {
        "state": "performance-ready" if ready else "thermally-degraded",
        "eligible_for_comparison": ready,
        "thresholds": {"max_block_ms": A_READY_MAX_BLOCK_MS, "max_verify_ms": A_READY_MAX_VERIFY_MS},
        "measured": {"block_ms": block_ms, "verify_ms": verify_ms},
    }


def write_manifest(result_dir: Path, job_id: str, benchmark: Benchmark) -> None:
    atomic_write_json(
        result_dir / "manifest.json",
        {
            "job_id": job_id,
            "benchmark": benchmark.name,
            "description": benchmark.description,
            "cwd": str(REPO_ROOT),
            "argv": list(benchmark.argv),
            "env": dict(benchmark.env),
            "code_sandbox_marker_removed": True,
            "pty": True,
        },
    )


def execute_benchmark(result_dir: Path, job_id: str, benchmark: Benchmark) -> dict[str, Any]:
    write_manifest(result_dir, job_id, benchmark)
    log_path = result_dir / "combined.log"
    env = os.environ.copy()
    env.pop("CODEX_SANDBOX", None)
    env.update(benchmark.env)
    started_at = utc_now()
    started = time.monotonic()

    with log_path.open("w") as log:
        log.write(f"# started_at={started_at}\n")
        log.write(f"# cwd={REPO_ROOT}\n")
        log.write(f"# command={shlex.join(benchmark.argv)}\n")
        log.flush()
        master_fd, slave_fd = pty.openpty()
        try:
            try:
                process = subprocess.Popen(
                    benchmark.argv,
                    cwd=REPO_ROOT,
                    env=env,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                    shell=False,
                )
            except Exception:
                os.close(master_fd)
                raise
        finally:
            os.close(slave_fd)

        chunks: list[str] = []
        try:
            while True:
                try:
                    data = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not data:
                    break
                text = data.decode(errors="replace")
                chunks.append(text)
                log.write(text)
                log.flush()
                sys.stdout.write(f"[{job_id}] {text}")
                sys.stdout.flush()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        finally:
            os.close(master_fd)
        returncode = process.wait()

    output = "".join(chunks)
    stats, footer_lines = extract_stats(output)
    native_metal = METAL_MARKER in output
    health = classify_health(benchmark, stats)
    if returncode != 0:
        status = "failed-command"
    elif not native_metal:
        status = "failed-native-metal-check"
    elif health["eligible_for_comparison"] is False:
        status = "completed-thermally-degraded"
    else:
        status = "completed"

    return {
        "job_id": job_id,
        "benchmark": benchmark.name,
        "status": status,
        "exit_code": returncode,
        "native_metal": native_metal,
        "metal_marker": METAL_MARKER,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "stats": stats,
        "health": health,
        "final_ds4_statistics": footer_lines,
    }


def dry_run(result_dir: Path, job_id: str, benchmark: Benchmark) -> dict[str, Any]:
    write_manifest(result_dir, job_id, benchmark)
    log_path = result_dir / "combined.log"
    log_path.write_text(
        "# dry run: benchmark was not started\n"
        f"# cwd={REPO_ROOT}\n"
        f"# command={shlex.join(benchmark.argv)}\n"
    )
    return {
        "job_id": job_id,
        "benchmark": benchmark.name,
        "status": "dry-run",
        "exit_code": None,
        "native_metal": None,
        "metal_marker": METAL_MARKER,
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "wall_seconds": 0.0,
        "log": str(log_path),
        "stats": {},
        "health": {"state": "not-run", "eligible_for_comparison": None},
        "final_ds4_statistics": [],
    }


def finish_job(
    paths: Mapping[str, Path],
    result_dir: Path,
    job_path: Path,
    result: Mapping[str, Any],
) -> None:
    destination_group = "completed" if result["status"] in {"completed", "completed-thermally-degraded", "dry-run"} else "failed"
    atomic_write_json(result_dir / "result.json", result)
    os.replace(job_path, paths[destination_group] / job_path.name)
    atomic_write_json(paths["latest"], result)


def handle_job(
    paths: Mapping[str, Path],
    results_root: Path,
    job_path: Path,
    dry_run_mode: bool,
) -> None:
    running_path = paths["running"] / job_path.name
    os.replace(job_path, running_path)
    job_id = running_path.stem
    try:
        benchmark = read_request(running_path)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = results_root / f"{stamp}_{benchmark.name}_{job_id}"
        result_dir.mkdir(parents=True, exist_ok=False)
        result = dry_run(result_dir, job_id, benchmark) if dry_run_mode else execute_benchmark(result_dir, job_id, benchmark)
    except Exception as exc:  # Surface malformed requests and runner failures in result files.
        result_dir = results_root / f"failed_{job_id}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "job_id": job_id,
            "status": "failed-request",
            "error": str(exc),
            "finished_at": utc_now(),
        }
    finish_job(paths, result_dir, running_path, result)
    print(f"{result['status']}: {job_id}", flush=True)


def submit(paths: Mapping[str, Path], benchmark_name: str) -> None:
    ensure_queue_dirs(paths)
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    request_path = paths["queue"] / f"{job_id}.json"
    atomic_write_json(request_path, {"benchmark": benchmark_name})
    print(job_id)


def list_benchmarks() -> None:
    for benchmark in BENCHMARKS.values():
        print(f"{benchmark.name}\t{benchmark.description}")


def status(paths: Mapping[str, Path]) -> int:
    if not paths["latest"].exists():
        print("no completed or failed jobs")
        return 0
    print(paths["latest"].read_text(), end="")
    return 0


def mark_orphaned_jobs(paths: Mapping[str, Path], results_root: Path) -> None:
    for job_path in sorted(paths["running"].glob("*.json")):
        job_id = job_path.stem
        result_dir = results_root / f"abandoned_{job_id}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "job_id": job_id,
            "status": "abandoned-after-watcher-restart",
            "finished_at": utc_now(),
        }
        finish_job(paths, result_dir, job_path, result)


def watch(
    paths: Mapping[str, Path],
    results_root: Path,
    poll_seconds: float,
    cooldown_seconds: float,
    once: bool,
    dry_run_mode: bool,
) -> int:
    ensure_queue_dirs(paths)
    results_root.mkdir(parents=True, exist_ok=True)
    with WatcherLock(paths):
        mark_orphaned_jobs(paths, results_root)
        print(f"watching {paths['queue']} (results: {results_root})", flush=True)
        last_finished_at = 0.0
        while True:
            queued_jobs = sorted(paths["queue"].glob("*.json"))
            if queued_jobs:
                remaining_cooldown = cooldown_seconds - (time.monotonic() - last_finished_at)
                if last_finished_at and remaining_cooldown > 0:
                    print(f"cooldown: waiting {remaining_cooldown:.0f}s before next job", flush=True)
                    time.sleep(remaining_cooldown)
                handle_job(paths, results_root, queued_jobs[0], dry_run_mode)
                last_finished_at = time.monotonic()
                if once:
                    return 0
                continue
            if once:
                return 0
            time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="queue an allowlisted benchmark")
    submit_parser.add_argument("benchmark", choices=sorted(BENCHMARKS))
    subparsers.add_parser("list", help="list allowlisted benchmarks")
    subparsers.add_parser("status", help="show the most recent job result")

    watch_parser = subparsers.add_parser("watch", help="run queued jobs serially")
    watch_parser.add_argument("--poll-seconds", type=float, default=1.0)
    watch_parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="minimum idle time between completed jobs (default: 0)",
    )
    watch_parser.add_argument("--once", action="store_true", help="process at most one queued job")
    watch_parser.add_argument("--dry-run", action="store_true", help="write manifests without starting ds4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = queue_paths(args.queue_dir)
    if args.command == "list":
        list_benchmarks()
        return 0
    if args.command == "submit":
        submit(paths, args.benchmark)
        return 0
    if args.command == "status":
        return status(paths)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must not be negative")
    return watch(paths, args.results_root, args.poll_seconds, args.cooldown_seconds, args.once, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("watcher interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"watcher error: {exc}", file=sys.stderr)
        raise SystemExit(2)
