#!/usr/bin/env python3
"""Non-ML smoke coverage for scripts/ds4_bench_watcher.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ds4_bench_watcher.py"
SPEC = importlib.util.spec_from_file_location("ds4_bench_watcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WATCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WATCHER
SPEC.loader.exec_module(WATCHER)


class WatcherSmokeTest(unittest.TestCase):
    def test_only_named_benchmarks_are_accepted(self) -> None:
        self.assertEqual(WATCHER.parse_request({"benchmark": "a"}).name, "a")
        with self.assertRaises(ValueError):
            WATCHER.parse_request({"benchmark": "a", "command": "sh -c nope"})
        with self.assertRaises(ValueError):
            WATCHER.parse_request({"benchmark": "unknown"})

    def test_b_prompt_is_the_single_line_reference_workload(self) -> None:
        prompt = WATCHER.BENCHMARKS["b"].argv[-1]
        self.assertEqual(prompt, WATCHER.B_PROMPT)
        self.assertNotIn("\n", prompt)
        self.assertNotIn("\r", prompt)

    def test_a_health_gate_classifies_slow_verifier_timing(self) -> None:
        benchmark = WATCHER.BENCHMARKS["a"]
        ready = WATCHER.classify_health(benchmark, {"block_ms": "109.99", "verify_ms": "95.99"})
        degraded = WATCHER.classify_health(benchmark, {"block_ms": "121.46", "verify_ms": "106.87"})
        self.assertEqual(ready["state"], "performance-ready")
        self.assertTrue(ready["eligible_for_comparison"])
        self.assertEqual(degraded["state"], "thermally-degraded")
        self.assertFalse(degraded["eligible_for_comparison"])

    def test_dry_run_completes_without_starting_ds4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir = root / "queue"
            results_root = root / "results"
            submit = subprocess.run(
                [sys.executable, str(SCRIPT), "--queue-dir", str(queue_dir), "submit", "a"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(submit.stdout.strip())
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--queue-dir",
                    str(queue_dir),
                    "--results-root",
                    str(results_root),
                    "watch",
                    "--once",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            results = list(results_root.glob("*/result.json"))
            self.assertEqual(len(results), 1)
            result = json.loads(results[0].read_text())
            self.assertEqual(result["benchmark"], "a")
            self.assertEqual(result["status"], "dry-run")
            self.assertIsNone(result["native_metal"])

    def test_host_execution_uses_a_pty_and_unsets_the_marker(self) -> None:
        benchmark = WATCHER.Benchmark(
            name="test",
            description="pty smoke",
            env={},
            argv=(
                "/bin/sh",
                "-c",
                "test -t 1 && test -z \"${CODEX_SANDBOX+x}\" && "
                "printf 'ds4: Metal device Apple M5 Max\\n'",
            ),
        )
        prior_marker = os.environ.get("CODEX_SANDBOX")
        os.environ["CODEX_SANDBOX"] = "restricted"
        try:
            with tempfile.TemporaryDirectory() as temporary:
                result = WATCHER.execute_benchmark(Path(temporary), "pty-test", benchmark)
        finally:
            if prior_marker is None:
                os.environ.pop("CODEX_SANDBOX", None)
            else:
                os.environ["CODEX_SANDBOX"] = prior_marker
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["native_metal"])

    def test_orphaned_running_job_is_not_retried_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = WATCHER.queue_paths(root / "queue")
            WATCHER.ensure_queue_dirs(paths)
            orphan = paths["running"] / "orphan.json"
            orphan.write_text('{"benchmark": "a"}\n')
            self.assertEqual(WATCHER.watch(paths, root / "results", 0.01, 0.0, True, True), 0)
            self.assertTrue((paths["failed"] / orphan.name).exists())
            result = json.loads(paths["latest"].read_text())
            self.assertEqual(result["status"], "abandoned-after-watcher-restart")


if __name__ == "__main__":
    unittest.main()
