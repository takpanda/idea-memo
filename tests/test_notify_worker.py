import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

import notify_worker  # noqa: E402


class SimilarNotificationDisplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sent = []
        notify_worker.api = lambda method, **params: self.sent.append(
            {"method": method, **params}
        )

    def test_url_import_memo_is_not_truncated(self) -> None:
        url = "https://example.com/articles/a-very-long-path-that-must-remain-intact"
        notify_worker.notify_similar(
            {"id": 1},
            [{
                "id": 2,
                "captured_at": "2026-08-01T00:00:00Z",
                "score": 0.91,
                "body": f"URL: {url}\nタイトル: 長い記事タイトル",
            }],
        )

        text = self.sent[0]["text"]
        self.assertIn(f"タイトル: 長い記事タイトル", text)
        self.assertIn(f"URL: {url}", text)
        self.assertNotIn("…", text)

    def test_non_url_memo_falls_back_to_snippet(self) -> None:
        body = "URLに見えるだけの本文ではなく、従来形式のメモ本文です。" * 4
        notify_worker.notify_similar(
            {"id": 1},
            [{
                "id": 2,
                "captured_at": "2026-08-01T00:00:00Z",
                "score": 0.91,
                "body": body,
            }],
        )

        text = self.sent[0]["text"]
        self.assertIn(notify_worker.snippet(body), text)
        self.assertNotIn(body, text)


if __name__ == "__main__":
    unittest.main()
