"""LLM接続設定とスモークチェックの回帰試験。"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_services import FakeServices, fresh_worker_imports  # noqa: E402


class LLMConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.services = FakeServices().__enter__()
        self.previous = {
            key: os.environ.get(key)
            for key in ("LLM_BASE_URL", "LLM_MODEL")
        }
        os.environ["LLM_BASE_URL"] = f"{self.services.base}/v1/"
        os.environ["LLM_MODEL"] = "fake-llm"
        fresh_worker_imports()

    def tearDown(self) -> None:
        self.services.__exit__()
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        fresh_worker_imports()

    def run_check(self) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        return subprocess.run(
            [sys.executable, str(ROOT / "worker" / "check_llm.py")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_workers_and_check_accept_trailing_slash(self) -> None:
        import research_worker
        import theme_writer

        self.assertEqual(theme_writer.LLM_BASE_URL, f"{self.services.base}/v1")
        self.assertEqual(research_worker.LLM_BASE_URL, f"{self.services.base}/v1")

        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        llm_calls = [call for call in self.services.calls if call["endpoint"] == "llm"]
        self.assertEqual(len(llm_calls), 1)
        self.assertEqual(llm_calls[0]["path"], "/v1/chat/completions")

    def test_check_rejects_response_without_choices(self) -> None:
        self.services.set_llm_payload({"model": "fake-llm"})

        result = self.run_check()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("choices が空", result.stdout)
