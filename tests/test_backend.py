"""Backend issue BEE-495 の回帰テスト。"""

import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.payload if size < 0 else self.payload[:size]


class ThemeWriterResponseTest(unittest.TestCase):
    def test_content_is_used(self):
        import theme_writer

        payload = {"choices": [{"message": {"content": '{"name":"本文"}'}}]}
        with patch.object(theme_writer.urllib.request, "urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(theme_writer.call_llm(["memo"])["name"], "本文")

    def test_reasoning_is_used_when_content_is_null(self):
        import theme_writer

        payload = {"choices": [{"message": {"content": None, "reasoning": '{"name":"推論"}'}}]}
        with patch.object(theme_writer.urllib.request, "urlopen", return_value=FakeResponse(payload)):
            self.assertEqual(theme_writer.call_llm(["memo"])["name"], "推論")

    def test_empty_response_returns_none_without_attribute_error(self):
        import theme_writer

        payload = {"choices": [{"message": {"content": None, "reasoning": None}}]}
        with patch.object(theme_writer.urllib.request, "urlopen", return_value=FakeResponse(payload)):
            self.assertIsNone(theme_writer.call_llm(["memo"]))

    def test_ollama_uses_native_chat_contract_and_strips_existing_base_path(self):
        import theme_writer

        payload = {"message": {"role": "assistant", "content": '{"name":"Ollama"}'}}
        response = FakeResponse(payload)
        with patch.object(theme_writer, "LLM_BASE_URL", "http://ollama.example:11434/v1/"), \
             patch.object(theme_writer.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(theme_writer.call_llm(["memo"])["name"], "Ollama")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://ollama.example:11434/api/chat")
        body = json.loads(request.data)
        self.assertEqual(body["format"], "json")
        self.assertFalse(body["think"])
        self.assertNotIn("raw", body)
        self.assertNotIn("raw", body["options"])
        self.assertEqual(body["model"], theme_writer.LLM_MODEL)

    def test_ollama_ndjson_uses_message_content_from_all_chunks(self):
        import theme_writer

        response = FakeResponse({})
        response.payload = (
            json.dumps({"message": {"content": '{"name":"'}}) + "\n"
            + json.dumps({"message": {"content": "ND" + '"}'}, "done": True})
            + "\n"
        ).encode("utf-8")
        with patch.object(theme_writer, "LLM_BASE_URL", "http://127.0.0.1:11434"), \
             patch.object(theme_writer.urllib.request, "urlopen", return_value=response):
            self.assertEqual(theme_writer.call_llm(["memo"])["name"], "ND")

    def test_ollama_invalid_or_empty_response_returns_none(self):
        import theme_writer

        for raw in (b"", b"not-json\n{broken"):
            response = FakeResponse({})
            response.payload = raw
            with patch.object(theme_writer, "LLM_BASE_URL", "http://127.0.0.1:11434"), \
                 patch.object(theme_writer.urllib.request, "urlopen", return_value=response):
                self.assertIsNone(theme_writer.call_llm(["memo"]))


class TelegramUrlContentTest(unittest.TestCase):
    @staticmethod
    def _db():
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.executescript(
            """
            CREATE TABLE kv (
                key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
            );
            CREATE TABLE ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL,
                body TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_ref TEXT,
                input_kind TEXT NOT NULL,
                parent_id INTEGER,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'inbox'
            );
            CREATE UNIQUE INDEX idx_test_ideas_source_ref
                ON ideas(source, source_ref) WHERE source_ref IS NOT NULL;
            """
        )
        return db

    @staticmethod
    def _page_response():
        response = FakeResponse({})
        response.payload = (
            "<html><head><title>登録対象</title></head>"
            "<body><p>URLから取得した本文です。</p></body></html>"
        ).encode("utf-8")
        response.headers = SimpleNamespace(
            get_content_charset=lambda: "utf-8"
        )
        return response

    @staticmethod
    def _message(message_id=1):
        return {
            "message_id": message_id,
            "date": 1_760_000_000,
            "chat": {"id": 0},
            "text": "https://example.com/article",
        }

    def test_extract_url_trims_sentence_punctuation(self):
        import telegram_ingest

        self.assertEqual(
            telegram_ingest.extract_url("参考 https://example.com/page。"),
            "https://example.com/page",
        )

    def test_fetch_url_content_extracts_title_and_visible_text(self):
        import telegram_ingest

        response = FakeResponse({})
        response.payload = (
            "<html><head><title>Example &amp; title</title></head>"
            "<body><h1>見出し</h1><p>本文です。</p><script>secret()</script>"
            "</body></html>"
        ).encode("utf-8")
        response.headers = SimpleNamespace(
            get_content_charset=lambda: "utf-8"
        )
        with patch.object(telegram_ingest.urllib.request, "urlopen", return_value=response):
            content = telegram_ingest.fetch_url_content("https://example.com/page")

        self.assertIn("URL: https://example.com/page", content)
        self.assertIn("タイトル: Example & title", content)
        self.assertIn("見出し 本文です。", content)
        self.assertNotIn("secret", content)

    def test_http_error_is_reported_and_offset_advances(self):
        import telegram_ingest

        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        sent = []
        error = urllib.error.HTTPError(
            "https://example.com/missing", 404, "Not Found", {}, None
        )

        with patch.object(telegram_ingest, "dispatch",
                          side_effect=lambda db, update: telegram_ingest.fetch_url_content(
                              "https://example.com/missing"
                          )), patch.object(telegram_ingest.urllib.request,
                                           "urlopen", side_effect=error), \
             patch.object(telegram_ingest, "api",
                          side_effect=lambda method, **params: sent.append(
                              {"method": method, **params})):
            telegram_ingest.handle_updates(db, [{"update_id": 41}])

        self.assertEqual(db.execute("SELECT value FROM kv WHERE key = 'telegram_offset'")
                         .fetchone()[0], "42")
        self.assertEqual(sent[0]["method"], "sendMessage")
        self.assertIn("取り込めませんでした", sent[0]["text"])
        db.close()

    def test_url_message_is_registered_with_fetched_content(self):
        import telegram_ingest

        db = self._db()
        written = []
        with patch.object(telegram_ingest.urllib.request, "urlopen",
                          return_value=self._page_response()), \
             patch.object(telegram_ingest, "write_idea_markdown",
                          side_effect=lambda db, idea_id: written.append(idea_id)):
            telegram_ingest.handle_updates(db, [{
                "update_id": 50,
                "message": self._message(),
            }])

        idea = db.execute(
            "SELECT body, input_kind, source, source_ref FROM ideas"
        ).fetchone()
        self.assertEqual(idea["input_kind"], "text")
        self.assertEqual(idea["source"], "telegram")
        self.assertEqual(idea["source_ref"], "0:1")
        self.assertIn("タイトル: 登録対象", idea["body"])
        self.assertIn("URLから取得した本文です。", idea["body"])
        self.assertEqual(written, [1])
        self.assertEqual(db.execute(
            "SELECT value FROM kv WHERE key = 'telegram_offset'"
        ).fetchone()[0], "51")
        db.close()

    def test_database_registration_failure_is_reported_and_offset_advances(self):
        import telegram_ingest

        db = self._db()
        db.executescript(
            """
            CREATE TRIGGER reject_idea BEFORE INSERT ON ideas
            BEGIN
                SELECT RAISE(ABORT, 'database registration failed');
            END;
            """
        )
        sent = []
        with patch.object(telegram_ingest.urllib.request, "urlopen",
                          return_value=self._page_response()), \
             patch.object(telegram_ingest, "api",
                          side_effect=lambda method, **params: sent.append(
                              {"method": method, **params})):
            telegram_ingest.handle_updates(db, [{
                "update_id": 60,
                "message": self._message(2),
            }])

        self.assertEqual(sent[0]["method"], "sendMessage")
        self.assertIn("取り込めませんでした", sent[0]["text"])
        self.assertEqual(db.execute(
            "SELECT value FROM kv WHERE key = 'telegram_offset'"
        ).fetchone()[0], "61")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM ideas").fetchone()[0], 0)
        db.close()

    def test_markdown_write_failure_is_reported_and_offset_advances(self):
        import telegram_ingest

        db = self._db()
        sent = []
        with patch.object(telegram_ingest.urllib.request, "urlopen",
                          return_value=self._page_response()), \
             patch.object(telegram_ingest, "write_idea_markdown",
                          side_effect=OSError("markdown write failed")), \
             patch.object(telegram_ingest, "api",
                          side_effect=lambda method, **params: sent.append(
                              {"method": method, **params})):
            telegram_ingest.handle_updates(db, [{
                "update_id": 70,
                "message": self._message(3),
            }])

        self.assertEqual(sent[0]["method"], "sendMessage")
        self.assertIn("取り込めませんでした", sent[0]["text"])
        self.assertEqual(db.execute(
            "SELECT value FROM kv WHERE key = 'telegram_offset'"
        ).fetchone()[0], "71")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM ideas").fetchone()[0], 1)
        db.close()


class ExistingDatabaseMigrationTest(unittest.TestCase):
    def test_existing_clusters_table_gets_members_json(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "idea.db"
            db = sqlite3.connect(db_path)
            db.execute("CREATE TABLE clusters (id INTEGER PRIMARY KEY, uid TEXT NOT NULL)")
            db.execute("INSERT INTO clusters (uid) VALUES ('old')")
            db.commit()
            db.close()

            old_db = os.environ.get("IDEA_DB")
            os.environ["IDEA_DB"] = str(db_path)
            try:
                # sqlite-vec は本番コンテナ依存。既存 DB の ALTER TABLE 経路は
                # 拡張 API を使わないため、単体テストでは読み込みだけ代替する。
                with patch.dict(sys.modules, {"sqlite_vec": SimpleNamespace(load=lambda db: None)}):
                    import init_db
                    importlib.reload(init_db)
                    self.assertEqual(init_db.main(), 0)
                    # 移行済みDBへの再実行は二重追加せず、既存行も保持する。
                    self.assertEqual(init_db.main(), 0)
            finally:
                if old_db is None:
                    os.environ.pop("IDEA_DB", None)
                else:
                    os.environ["IDEA_DB"] = old_db

            db = sqlite3.connect(db_path)
            columns = {row[1] for row in db.execute("PRAGMA table_info(clusters)")}
            self.assertIn("members_json", columns)
            self.assertEqual(db.execute("SELECT uid FROM clusters").fetchone()[0], "old")
            db.close()

    @unittest.skipUnless(importlib.util.find_spec("sqlite_vec"), "sqlite-vec not installed")
    def test_new_database_has_members_json_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "idea.db"
            old_db = os.environ.get("IDEA_DB")
            old_schema = os.environ.get("IDEA_SCHEMA")
            os.environ["IDEA_DB"] = str(db_path)
            os.environ["IDEA_SCHEMA"] = str(ROOT / "worker" / "schema.sql")
            try:
                import init_db
                importlib.reload(init_db)
                self.assertEqual(init_db.main(), 0)
                self.assertEqual(init_db.main(), 0)
                check_db = sqlite3.connect(db_path)
                columns = {row[1] for row in check_db.execute("PRAGMA table_info(clusters)")}
                self.assertIn("members_json", columns)
                check_db.close()
            finally:
                if old_db is None:
                    os.environ.pop("IDEA_DB", None)
                else:
                    os.environ["IDEA_DB"] = old_db
                if old_schema is None:
                    os.environ.pop("IDEA_SCHEMA", None)
                else:
                    os.environ["IDEA_SCHEMA"] = old_schema


class PreviousMembersTest(unittest.TestCase):
    def test_previous_members_reads_migrated_column(self):
        fake_sklearn = types.ModuleType("sklearn")
        fake_cluster = types.ModuleType("sklearn.cluster")
        fake_cluster.HDBSCAN = object
        fake_sklearn.cluster = fake_cluster
        with patch.dict(sys.modules, {
            "sqlite_vec": SimpleNamespace(),
            "sklearn": fake_sklearn,
            "sklearn.cluster": fake_cluster,
        }):
            sys.modules.pop("cluster_worker", None)
            import cluster_worker
            db = sqlite3.connect(":memory:")
            db.row_factory = sqlite3.Row
            db.execute("CREATE TABLE clusters (id INTEGER, members_json TEXT, closed_at TEXT)")
            db.executemany(
                "INSERT INTO clusters VALUES (?, ?, ?)",
                [(7, "[1, 2]", None), (8, "[3]", "2026-01-01T00:00:00Z")],
            )
            previous, live = cluster_worker.previous_members(db)
            self.assertEqual(previous, {7: {1, 2}, 8: {3}})
            self.assertEqual(live, {7})
            db.close()


if __name__ == "__main__":
    unittest.main()
