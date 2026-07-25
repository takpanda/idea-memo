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

    def read(self):
        return self.payload


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
