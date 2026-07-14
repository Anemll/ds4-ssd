#!/usr/bin/env python3
"""Focused regression checks for the dependency-free dashboard adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("server.py")
SPEC = importlib.util.spec_from_file_location("dashboard_server", MODULE_PATH)
assert SPEC and SPEC.loader
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


METRICS = """
# TYPE ds4_server_info gauge
ds4_server_info{model_name="deepseek-v4-flash",backend="metal"} 1
ds4_generation_tokens_total 300
ds4_num_requests_running 1
ds4_dspark_draft_enabled 1
ds4_dspark_drafted_tokens_total 120
ds4_dspark_accepted_tokens_total 84
ds4_dspark_blocks_total 40
ds4_dspark_skip_pre_draft_total 3
ds4_dspark_skip_verify_total 2
ds4_dspark_dynamic_verify 1
ds4_dspark_active_verify_budget 4
ds4_runtime_status_available 1
ds4_moe_slot_bank 32
ds4_memory_process_resident_bytes 2147483648
ds4_memory_process_phys_footprint_bytes 3221225472
ds4_memory_gpu_footprint_bytes 4294967296
ds4_memory_gpu_compressed_bytes 536870912
ds4_memory_task_compressed_bytes 268435456
ds4_memory_system_total_bytes 137438953472
ds4_memory_system_free_bytes 68719476736
ds4_memory_system_active_bytes 34359738368
ds4_memory_system_wired_bytes 17179869184
ds4_memory_system_compressed_bytes 1073741824
ds4_memory_system_compressor_bytes 536870912
ds4_memory_system_pressure_percent 11
ds4_memory_swap_total_bytes 4294967296
ds4_memory_swap_used_bytes 1073741824
"""


class DashboardMetricsTest(unittest.TestCase):
    def test_dspark_and_memory_summaries(self) -> None:
        parsed = dashboard.parse_prometheus(METRICS)
        dspark = dashboard.summarize_dspark(parsed)
        memory = dashboard.summarize_memory(parsed)

        self.assertEqual(parsed["model"], "deepseek-v4-flash")
        self.assertTrue(dspark["available"])
        self.assertTrue(dspark["enabled"])
        self.assertEqual(dspark["acceptancePct"], 70.0)
        self.assertEqual(dspark["tau"], 2.1)
        self.assertEqual(dspark["skipPreDraft"], 3.0)
        self.assertEqual(dspark["skipVerify"], 2.0)
        self.assertTrue(dspark["dynamic"])
        self.assertEqual(dspark["activeVerifyBudget"], 4)
        self.assertTrue(memory["available"])
        self.assertEqual(memory["systemUsedBytes"], 68719476736.0)
        self.assertEqual(memory["gpuCompressedBytes"], 536870912.0)
        self.assertEqual(memory["moeSlots"], 32.0)

    def test_agent_api_url_uses_metrics_host_or_safe_fallback(self) -> None:
        self.assertEqual(
            dashboard.resolve_agent_api_base_url("http://m5m.local:8000/metrics"),
            "http://m5m.local:8000/v1",
        )
        self.assertEqual(
            dashboard.resolve_agent_api_base_url("not-a-url"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            dashboard.resolve_agent_api_base_url(
                "http://127.0.0.1:8000/metrics", "https://ds4.example.test/api/v1/"
            ),
            "https://ds4.example.test/api/v1",
        )


class ServerControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_log_path = dashboard.SERVER_LOG_PATH
        dashboard.SERVER_LOG_PATH = str(Path(self.tmp.name) / "server.log")

    def tearDown(self) -> None:
        dashboard.SERVER_LOG_PATH = self.old_log_path
        self.tmp.cleanup()

    def test_rejects_non_json_and_relative_executables(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON argv"):
            dashboard.ServerController._validate("./ds4-server --port 8000", self.tmp.name)
        with self.assertRaisesRegex(ValueError, "absolute path"):
            dashboard.ServerController._validate(
                json.dumps(["./ds4-server", ";", "touch", "sentinel"]), self.tmp.name
            )

    def test_managed_child_can_start_and_stop(self) -> None:
        command = json.dumps([sys.executable, "-c", "import time; time.sleep(60)"])
        controller = dashboard.ServerController(command, self.tmp.name)
        try:
            started, _, snapshot = controller.start()
            self.assertTrue(started)
            self.assertTrue(snapshot["running"])
            self.assertIsNotNone(snapshot["pid"])
            stopped, _, snapshot = controller.stop()
            self.assertTrue(stopped)
            self.assertFalse(snapshot["running"])
        finally:
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
