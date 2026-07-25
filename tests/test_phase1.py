"""
Phase 1 の通し試験。

Telegram 取り込み -> 文字起こし -> 埋め込み -> 類似通知 -> 判断の書き戻し
までを実際に動かす。Mac mini の 2 サービスと Telegram の file API は
tests/fake_services.py が本物と同じ HTTP 契約で代役を務める。
Telegram の Bot API 呼び出しだけは外部なので記録するだけに差し替える。

  python -m unittest discover -s tests      # 要 sqlite-vec
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_services import FakeServices, fresh_worker_imports  # noqa: E402

CHAT_ID = 424242

# 近い 2 本 (A と VOICE)、やや近い 1 本 (B)、無関係 1 本 (C)。
# 偽の埋め込みでの cos は A-VOICE 0.91 / A-B 0.57 / A-C 0.03 なので、
# SIM_THRESHOLD=0.82 なら A-VOICE だけが通知に上がる
TEXT_A = "週末に近所の川沿いを走るランニングコースを地図アプリでまとめたい"
TEXT_B = "川沿いのランニングコースを地図アプリでまとめておきたい。週末に走る用"
TEXT_C = "確定申告の医療費控除、領収書をスキャンして保管する仕組みを作る"
VOICE_TEXT = "ランニングコースを地図アプリでまとめたい。週末に近所の川沿いを走る"

tmp: tempfile.TemporaryDirectory
services: FakeServices
sent: list[dict]


def setUpModule() -> None:
    """環境変数はワーカーの import 時に読まれるので、先に確定させる。"""
    global tmp, services, sent
    global common, telegram_ingest, notify_worker, supervisor, embed_worker

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "repo").mkdir()
    (root / "data").mkdir()

    services = FakeServices().__enter__()

    os.environ.update({
        "TELEGRAM_BOT_TOKEN": "test:token",
        "ALLOWED_CHAT_ID": str(CHAT_ID),
        "IDEA_DB": str(root / "data" / "idea.db"),
        "IDEA_REPO": str(root / "repo"),
        "IDEA_SCHEMA": str(ROOT / "worker" / "schema.sql"),
        "EMBED_URL": f"{services.base}/embed",
        "TRANSCRIBE_URL": f"{services.base}/transcribe",
        "SIM_THRESHOLD": "0.82",
    })

    fresh_worker_imports()

    import init_db
    assert init_db.main() == 0, "init_db failed"

    import common
    import embed_worker
    import notify_worker
    import supervisor
    import telegram_ingest

    # Telegram Bot API は外に出さない。送信内容は sent に貯めて検証する
    sent = []

    def fake_api(method: str, timeout: int = 65, **params):
        sent.append({"method": method, **params})
        if method == "getFile":
            name = f"{params['file_id']}.ogg"
            return {"file_id": params["file_id"],
                    "file_path": f"voice/{name}",
                    "file_size": len(services.blob(name))}
        if method == "sendMessage":
            return {"message_id": 9000 + len(sent)}
        return True

    telegram_ingest.api = fake_api
    notify_worker.api = fake_api
    telegram_ingest.FILE_API = f"{services.base}/files"


def tearDownModule() -> None:
    services.__exit__()
    tmp.cleanup()


def message(message_id: int, *, text: str | None = None, voice: dict | None = None,
            date: int = 1_760_000_000, reply_to: int | None = None) -> dict:
    msg: dict = {"message_id": message_id, "date": date,
                 "chat": {"id": CHAT_ID, "type": "private"}}
    if text is not None:
        msg["text"] = text
    if voice is not None:
        msg["voice"] = voice
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return msg


class Phase1Pipeline(unittest.TestCase):
    """1 本のシナリオを順に流す。ステップ間で状態を引き継ぐので順序が要る。"""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = common.connect(with_vec=True)

    def uid_of(self, body_prefix: str) -> str:
        row = self.db.execute(
            "SELECT uid FROM ideas WHERE body LIKE ?", (body_prefix + "%",)
        ).fetchone()
        self.assertIsNotNone(row, f"idea not found: {body_prefix}")
        return row["uid"]

    # --- 1. 取り込み ------------------------------------------------
    def test_01_ingest_text(self) -> None:
        for i, text in enumerate([TEXT_A, TEXT_B, TEXT_C], start=100):
            telegram_ingest.handle_message(self.db, message(i, text=text, date=1_760_000_000 + i))

        rows = self.db.execute("SELECT uid, body, input_kind, file_path FROM ideas").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["input_kind"] for r in rows}, {"text"})

        # Markdown が DB と同時に出ていること
        for row in rows:
            path = common.REPO_ROOT / row["file_path"]
            self.assertTrue(path.exists(), f"markdown missing: {row['file_path']}")
            content = path.read_text(encoding="utf-8")
            self.assertIn(f"uid: {row['uid']}", content)
            self.assertIn(row["body"], content)

    def test_02_ingest_is_idempotent(self) -> None:
        """getUpdates の再配信で二重登録しない。"""
        telegram_ingest.handle_message(self.db, message(100, text=TEXT_A, date=1_760_000_100))
        n = self.db.execute("SELECT COUNT(*) AS n FROM ideas").fetchone()["n"]
        self.assertEqual(n, 3)

    def test_03_ingest_voice(self) -> None:
        services.put_file("v1.ogg", VOICE_TEXT.encode("utf-8"))
        telegram_ingest.handle_message(
            self.db,
            message(200, voice={"file_id": "v1", "mime_type": "audio/ogg", "duration": 7},
                    date=1_760_000_200),
        )

        idea = self.db.execute(
            "SELECT id, uid, body, input_kind FROM ideas WHERE input_kind = 'voice'"
        ).fetchone()
        self.assertIsNotNone(idea)
        self.assertEqual(idea["body"], "", "文字起こし前の body は空のまま")

        att = self.db.execute(
            "SELECT kind, bytes, file_path, transcript, meta_json FROM attachments "
            "WHERE idea_id = ?", (idea["id"],)
        ).fetchone()
        self.assertEqual(att["kind"], "voice")
        self.assertIsNone(att["transcript"])
        self.assertEqual(att["bytes"], len(VOICE_TEXT.encode("utf-8")))
        self.assertTrue((common.REPO_ROOT / att["file_path"]).exists())
        self.assertIn("sha256", json.loads(att["meta_json"]))

    def test_04_command_is_not_stored(self) -> None:
        telegram_ingest.handle_message(self.db, message(300, text="/start", date=1_760_000_300))
        n = self.db.execute("SELECT COUNT(*) AS n FROM ideas").fetchone()["n"]
        self.assertEqual(n, 4)
        self.assertEqual(sent[-1]["method"], "sendMessage")

    def test_05_foreign_chat_is_ignored(self) -> None:
        msg = message(400, text="他人のチャット", date=1_760_000_400)
        msg["chat"]["id"] = 999
        telegram_ingest.handle_message(self.db, msg)
        self.assertEqual(self.db.execute("SELECT COUNT(*) AS n FROM ideas").fetchone()["n"], 4)

    # --- 2. reconciliation 1 パス ------------------------------------
    def test_06_single_pass_reaches_notification(self) -> None:
        """ボイスメモが 文字起こし -> 埋め込み -> 通知 を 1 パスで抜ける。"""
        sent.clear()
        moved = supervisor.run_pass(self.db)
        self.assertGreater(moved, 0)

        # 文字起こし
        att = self.db.execute(
            "SELECT transcript, asr_model FROM attachments"
        ).fetchone()
        self.assertEqual(att["transcript"], VOICE_TEXT)
        self.assertIsNotNone(att["asr_model"])
        voice_body = self.db.execute(
            "SELECT body FROM ideas WHERE input_kind = 'voice'"
        ).fetchone()["body"]
        self.assertEqual(voice_body, VOICE_TEXT)

        # 埋め込み: 4 メモ × 2 プレフィックス
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"], 8
        )
        for table in ("vec_ideas_ruri_v3_310m", "vec_ideas_ruri_v3_310m_topic"):
            n = self.db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            self.assertEqual(n, 4, table)

        # プレフィックスは両方ともサーバーに渡っている
        prefixes = {c["prefix"] for c in services.calls if c["endpoint"] == "embed"}
        self.assertEqual(prefixes, {"semantic", "topic"})

        # 全メモの走査が終わっている (候補ゼロでも印を付ける)
        unnotified = self.db.execute(
            "SELECT COUNT(*) AS n FROM ideas WHERE similar_notified_at IS NULL"
        ).fetchone()["n"]
        self.assertEqual(unnotified, 0)

    def test_07_only_similar_pair_is_notified(self) -> None:
        """閾値を超えた 1 組だけ、往復させずに 1 通だけ届く。"""
        notifications = [s for s in sent if s["method"] == "sendMessage"]
        self.assertEqual(len(notifications), 1, f"想定外の通知: {notifications}")

        text = notifications[0]["text"]
        self.assertIn("似ているメモ", text)
        # 提示されるのは「過去のメモ」= テキスト A の方
        self.assertIn(TEXT_A[:20], " ".join(text.split()))

        buttons = notifications[0]["reply_markup"]["inline_keyboard"]
        self.assertEqual(len(buttons), 1)
        self.assertEqual([b["callback_data"].split(":")[3] for b in buttons[0]], ["c", "x"])
        for button in buttons[0]:
            self.assertLessEqual(len(button["callback_data"].encode()), 64)

    def test_08_second_pass_is_noop(self) -> None:
        """reconciliation は収束する。空回りしない。"""
        self.assertEqual(supervisor.run_pass(self.db), 0)

    # --- 3. 人間の判断 -----------------------------------------------
    def test_09_confirm_writes_back_to_markdown(self) -> None:
        data = [s for s in sent if s["method"] == "sendMessage"][0][
            "reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        telegram_ingest.handle_callback(self.db, {"id": "cb1", "data": data})

        rel = self.db.execute("SELECT * FROM relations").fetchone()
        self.assertEqual(rel["verdict"], "confirmed")
        self.assertEqual(rel["kind"], "similar")
        self.assertLess(rel["src_id"], rel["dst_id"])
        self.assertGreater(rel["score"], 0.82)

        voice_uid = self.db.execute(
            "SELECT uid FROM ideas WHERE input_kind = 'voice'"
        ).fetchone()["uid"]
        text_uid = self.uid_of(TEXT_A[:10])

        # 双方の Markdown に相手が出る
        for own, other in ((voice_uid, text_uid), (text_uid, voice_uid)):
            path = common.REPO_ROOT / self.db.execute(
                "SELECT file_path FROM ideas WHERE uid = ?", (own,)
            ).fetchone()["file_path"]
            content = path.read_text(encoding="utf-8")
            self.assertIn("related:", content)
            self.assertIn(f"uid: {other}", content)

    def test_10_decided_pair_is_not_proposed_again(self) -> None:
        voice_id = self.db.execute(
            "SELECT id FROM ideas WHERE input_kind = 'voice'"
        ).fetchone()["id"]
        self.assertEqual(notify_worker.find_similar(self.db, voice_id), [])

    def test_11_rejected_pair_is_not_proposed_again(self) -> None:
        """却下も「判断済み」として二度と出さない。"""
        ids = {r["body"][:6]: r["id"] for r in self.db.execute("SELECT id, body FROM ideas")}
        src, dst = sorted((ids[TEXT_A[:6]], ids[TEXT_B[:6]]))
        telegram_ingest.handle_callback(
            self.db, {"id": "cb2", "data": f"rel:{src}:{dst}:x:57"}
        )
        rel = self.db.execute(
            "SELECT verdict FROM relations WHERE src_id = ? AND dst_id = ?", (src, dst)
        ).fetchone()
        self.assertEqual(rel["verdict"], "rejected")
        # 却下は Markdown に出さない
        path = common.REPO_ROOT / self.db.execute(
            "SELECT file_path FROM ideas WHERE id = ?", (dst,)
        ).fetchone()["file_path"]
        self.assertNotIn("related:", path.read_text(encoding="utf-8"))
        self.assertNotIn(str(dst), [str(m["id"]) for m in
                                    notify_worker.find_similar(self.db, src)])

    # --- 4. 編集と再埋め込み ------------------------------------------
    def test_12_edit_triggers_reembedding(self) -> None:
        idea = self.db.execute(
            "SELECT id, uid FROM ideas WHERE body LIKE ?", (TEXT_C[:6] + "%",)
        ).fetchone()
        before = self.db.execute(
            "SELECT content_hash FROM embeddings WHERE idea_id = ? AND prefix = ''",
            (idea["id"],),
        ).fetchone()["content_hash"]

        edited = TEXT_C + " 家族の分もまとめて申告する"
        telegram_ingest.handle_edit(self.db, message(102, text=edited, date=1_760_000_102))

        self.assertEqual(
            self.db.execute("SELECT body FROM ideas WHERE id = ?", (idea["id"],)).fetchone()["body"],
            edited,
        )
        self.assertIn(edited, (common.REPO_ROOT / self.db.execute(
            "SELECT file_path FROM ideas WHERE id = ?", (idea["id"],)
        ).fetchone()["file_path"]).read_text(encoding="utf-8"))

        self.assertGreater(supervisor.run_pass(self.db), 0)
        after = self.db.execute(
            "SELECT content_hash FROM embeddings WHERE idea_id = ? AND prefix = ''",
            (idea["id"],),
        ).fetchone()["content_hash"]
        self.assertNotEqual(before, after)
        # 重複行を作らずに更新している
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"], 8
        )
        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) AS n FROM vec_ideas_ruri_v3_310m"
            ).fetchone()["n"], 4
        )

    # --- 5. 返信と全文検索 --------------------------------------------
    def test_13_reply_is_linked_and_not_notified(self) -> None:
        parent_id = self.db.execute(
            "SELECT id FROM ideas WHERE body LIKE ?", (TEXT_A[:6] + "%",)
        ).fetchone()["id"]
        sent.clear()
        telegram_ingest.handle_message(
            self.db,
            message(500, text=TEXT_A + " 追記: 距離と高低差も一緒に持たせたい",
                    date=1_760_000_500, reply_to=100),
        )
        child = self.db.execute(
            "SELECT id, parent_id FROM ideas WHERE source_ref = ?",
            (f"{CHAT_ID}:500",),
        ).fetchone()
        self.assertEqual(child["parent_id"], parent_id)

        supervisor.run_pass(self.db)
        # 親子は似ていて当然なので、親を候補に出さない
        proposed = {m["id"] for m in notify_worker.find_similar(self.db, child["id"])}
        self.assertNotIn(parent_id, proposed)
        for s in sent:
            if s["method"] == "sendMessage":
                self.assertNotIn(str(parent_id), json.dumps(s, ensure_ascii=False))

    def test_14_fts_finds_japanese_without_tokenizer(self) -> None:
        rows = self.db.execute(
            "SELECT rowid FROM fts_ideas WHERE fts_ideas MATCH ? ORDER BY bm25(fts_ideas)",
            ('"ランニングコース"',),
        ).fetchall()
        self.assertGreaterEqual(len(rows), 2)

        # 編集後の本文が索引に反映されている (トリガ経由)
        hit = self.db.execute(
            "SELECT rowid FROM fts_ideas WHERE fts_ideas MATCH ?", ('"家族の分"',)
        ).fetchall()
        self.assertEqual(len(hit), 1)


class UpdateLoop(unittest.TestCase):
    """取り込めない 1 通で入口が止まらないこと。"""

    def setUp(self) -> None:
        self.db = common.connect()
        sent.clear()

    def offset(self) -> int:
        row = self.db.execute("SELECT value FROM kv WHERE key = 'telegram_offset'").fetchone()
        return int(row["value"]) if row else 0

    def test_broken_update_does_not_block_the_queue(self) -> None:
        # file_id が偽サーバーに無い = getFile で落ちるボイスメモ
        broken = {"update_id": 700,
                  "message": message(700, voice={"file_id": "missing", "mime_type": "audio/ogg"},
                                     date=1_760_002_000)}
        good = {"update_id": 701,
                "message": message(701, text="壊れた次のメモも取り込まれる", date=1_760_002_001)}

        telegram_ingest.handle_updates(self.db, [broken, good])

        self.assertEqual(self.offset(), 702)
        self.assertIsNotNone(self.db.execute(
            "SELECT id FROM ideas WHERE source_ref = ?", (f"{CHAT_ID}:701",)
        ).fetchone())
        self.assertTrue(any("取り込めませんでした" in s.get("text", "") for s in sent),
                        "捨てたことを黙らない")

    def test_network_failure_keeps_the_offset(self) -> None:
        import urllib.error

        before = self.offset()
        original = telegram_ingest.FILE_API
        telegram_ingest.FILE_API = "http://127.0.0.1:1/files"   # 必ず届かない
        services.put_file("v800.ogg", b"unreachable")
        try:
            with self.assertRaises(urllib.error.URLError):
                telegram_ingest.handle_updates(self.db, [
                    {"update_id": 800,
                     "message": message(800, voice={"file_id": "v800", "mime_type": "audio/ogg"},
                                        date=1_760_002_100)},
                ])
        finally:
            telegram_ingest.FILE_API = original

        self.assertEqual(self.offset(), before, "一時障害では offset を進めない")
        self.assertIsNone(self.db.execute(
            "SELECT id FROM ideas WHERE source_ref = ?", (f"{CHAT_ID}:800",)
        ).fetchone())


class VoiceFailures(unittest.TestCase):
    """文字起こしが返らない添付で reconciliation が空回りしないこと。"""

    def setUp(self) -> None:
        self.db = common.connect(with_vec=True)

    def _add_voice(self, message_id: int, name: str, blob: bytes) -> int:
        services.put_file(name, blob)
        telegram_ingest.handle_message(
            self.db,
            message(message_id, voice={"file_id": name[:-4], "mime_type": "audio/ogg"},
                    date=1_760_001_000 + message_id),
        )
        return self.db.execute(
            "SELECT id FROM ideas WHERE source_ref = ?", (f"{CHAT_ID}:{message_id}",)
        ).fetchone()["id"]

    def test_silent_audio_stops_retrying(self) -> None:
        idea_id = self._add_voice(600, "v600.ogg", b"   ")
        self.assertGreaterEqual(transcribe_tick(self.db), 0)
        att = self.db.execute(
            "SELECT transcript FROM attachments WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        self.assertEqual(att["transcript"], "", "無音は空文字で確定させて再試行を止める")
        self.assertEqual(transcribe_tick(self.db), 0)

    def test_missing_file_does_not_spin(self) -> None:
        idea_id = self._add_voice(601, "v601.ogg", "消える音声".encode("utf-8"))
        path = common.REPO_ROOT / self.db.execute(
            "SELECT file_path FROM attachments WHERE idea_id = ?", (idea_id,)
        ).fetchone()["file_path"]
        path.unlink()
        # 進捗ゼロを返さないと supervisor が毎周回で回り続ける
        self.assertEqual(transcribe_tick(self.db), 0)


def transcribe_tick(db) -> int:
    import transcribe_worker
    return transcribe_worker.tick(db)


if __name__ == "__main__":
    unittest.main(verbosity=2)
