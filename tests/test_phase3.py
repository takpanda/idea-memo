"""
Phase 3 の通し試験。

テーマ調査 (クエリ設計 -> 検索 -> 選別 -> 要約 -> findings 保存 -> テーマノート
書き戻し) -> 参考情報の通知 -> 人の判断の記録 -> 週次ダイジェスト、までを実際に動かす。

GPU ノードの LLM は tests/fake_services.py が本物と同じ HTTP 契約で代役を
務める。検索バックエンド (Brave/SearXNG) だけは HTTP 契約を持たない外部
サービスなので、research_worker.search をモジュール属性ごと差し替える。

  python -m unittest discover -s tests      # 要 sqlite-vec, numpy, scikit-learn
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_services import FakeServices, fresh_worker_imports  # noqa: E402

CHAT_ID = 626262

# test_phase2.py と同じ 3 グループ + 無関係な 1 件。単独のグループだけを
# HDBSCAN に渡しても密度差が無く全ノイズになる (root がクラスタに選ばれない
# ため) ので、対比になる他のグループが要る。研究対象には PLANT (自動水やり)
# を使う。単語や文面は phase2 のものと合わせて、島の形成が確実に効くようにする
RUN = [
    "週末に近所の川沿いを走るランニングコースを地図アプリでまとめたい",
    "川沿いのランニングコースを地図アプリでまとめておきたい。週末に走る用",
    "ランニングコースの地図アプリに距離と高低差を出したい。川沿いを走る週末",
    "地図アプリでランニングコースを共有したい。近所の川沿いを走る週末の記録",
    "週末のランニングで走った川沿いコースを地図アプリに記録してまとめる",
]
TAX = [
    "確定申告の医療費控除、領収書をスキャンして保管する仕組みを作る",
    "医療費控除の領収書をスキャンして確定申告用に保管したい仕組み",
    "確定申告で使う領収書のスキャン、医療費控除の集計まで自動でやる仕組み",
    "領収書スキャンの仕組みを確定申告に合わせる。医療費控除の保管を楽に",
    "医療費控除のために領収書をスキャンして保管、確定申告の集計を自動化する",
]
PLANT = [
    "観葉植物の水やりを土の湿度センサーで自動化する装置を作りたい",
    "土の湿度センサーで観葉植物の水やりを自動でやる装置がほしい",
    "水やり自動化の装置に土の湿度センサーを付けて観葉植物を枯らさない",
    "観葉植物を枯らさないために土の湿度センサーと水やり装置を連動させる",
    "湿度センサーの値で水やり装置を動かす。観葉植物の自動化を作る",
]
PLANT_LATER = "観葉植物の水やり自動化、土の湿度センサーをもう1つ追加して精度を上げたい"
LONER = "駅前のパン屋に並ばずに済む時間帯をメモしておく"

# ほぼ同一文の組。近似埋め込みでも確実に SIM_THRESHOLD (0.82) を超える
DUP_A = "駅前のパン屋の隣にある本屋の営業時間をメモしておきたい"
DUP_B = "駅前のパン屋の隣にある本屋の営業時間をメモしておく"

tmp: tempfile.TemporaryDirectory
services: FakeServices
sent: list[dict]


class FakeSearch:
    """research_worker.search の差し替え。クエリごとの結果を返し、呼び出しを記録する。"""

    def __init__(self) -> None:
        self.by_query: dict[str, list[dict]] = {}
        self.queries: list[str] = []

    def set(self, query: str, hits: list[dict]) -> None:
        self.by_query[query] = hits

    def __call__(self, query: str) -> list[dict]:
        self.queries.append(query)
        return [dict(h) for h in self.by_query.get(query, [])]


def hit(url: str, title: str, site: str = "example.com", snippet: str = "スニペット") -> dict:
    return {"url": url, "title": title, "site": site, "snippet": snippet}


def setUpModule() -> None:
    """環境変数はワーカーの import 時に読まれるので、先に確定させる。"""
    global tmp, services, sent, fake_search
    global common, cluster_worker, embed_worker, telegram_ingest, theme_writer
    global notify_worker, research_worker, digest_worker

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
        "LLM_BASE_URL": f"{services.base}/v1",
        "LLM_MODEL": "fake-llm",
        "MIN_CLUSTER_SIZE": "3",
        "MIN_SAMPLES": "2",
        "INHERIT_THRESHOLD": "0.4",
        "SIM_THRESHOLD": "0.82",
    })

    fresh_worker_imports()

    import init_db
    assert init_db.main() == 0, "init_db failed"

    import cluster_worker
    import common
    import digest_worker
    import embed_worker
    import notify_worker
    import research_worker
    import telegram_ingest
    import theme_writer

    sent = []

    def fake_api(method: str, timeout: int = 65, **params):
        sent.append({"method": method, **params})
        if method == "sendMessage":
            return {"message_id": 9000 + len(sent)}
        return True

    telegram_ingest.api = fake_api
    notify_worker.api = fake_api
    digest_worker.api = fake_api

    fake_search = FakeSearch()
    research_worker.search = fake_search
    # 検索 API へのマナー用・連投防止用の sleep。テストを待たせるだけなので無効化
    research_worker.time.sleep = lambda s: None
    notify_worker.time.sleep = lambda s: None


def tearDownModule() -> None:
    services.__exit__()
    tmp.cleanup()


def message(message_id: int, text: str) -> dict:
    # digest_worker は実時刻基準の「過去 7 日」で集計するので、
    # Phase 1/2 のような固定エポックではなく現在時刻を基準にする
    return {"message_id": message_id, "date": int(time.time()) + message_id,
            "chat": {"id": CHAT_ID, "type": "private"}, "text": text}


class Phase3Pipeline(unittest.TestCase):
    """1 本のシナリオを順に流す。ステップ間で状態を引き継ぐので順序が要る。"""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = common.connect(with_vec=True)

    # --- ヘルパ ------------------------------------------------------
    def ids_of(self, bodies: list[str]) -> set[int]:
        out = set()
        for body in bodies:
            row = self.db.execute(
                "SELECT id FROM ideas WHERE body = ?", (body,)
            ).fetchone()
            self.assertIsNotNone(row, f"idea not found: {body[:16]}")
            out.add(row["id"])
        return out

    def cluster_for(self, bodies: list[str]):
        wanted = self.ids_of(bodies)
        rows = self.db.execute(
            "SELECT DISTINCT cluster_id FROM idea_clusters WHERE idea_id IN "
            f"({','.join('?' * len(wanted))})", tuple(sorted(wanted))
        ).fetchall()
        self.assertEqual(len(rows), 1, f"1 クラスタにまとまっていない: {rows}")
        return self.db.execute(
            "SELECT * FROM clusters WHERE id = ?", (rows[0]["cluster_id"],)
        ).fetchone()

    def embed_all(self) -> None:
        while embed_worker.tick(self.db) > 0:
            pass

    def theme_note(self, uid: str) -> str:
        return (common.REPO_ROOT / f"themes/{uid}.md").read_text(encoding="utf-8")

    # --- 1. 下準備: テーマを作る (Phase 2 の再演。対比が無いと HDBSCAN が
    #        島を選べないので RUN / TAX もいっしょに入れる) ------------------
    def test_01_build_a_named_theme(self) -> None:
        for i, text in enumerate(RUN + TAX + PLANT + [LONER], start=100):
            telegram_ingest.handle_message(self.db, message(i, text))
        self.embed_all()
        cluster_worker.run(self.db)
        self.assertEqual(theme_writer.tick(self.db), 3)

        cluster = self.cluster_for(PLANT)
        self.assertEqual(cluster["name"], "テーマ5件")
        self.assertIsNone(cluster["researched_at"], "まだ調査していない")

    # --- 2. 調査 1 周: クエリ撃ち分け -> 選別 -> 要約 -> 保存 -----------------
    def test_02_research_collects_and_writes_theme_note(self) -> None:
        pending = research_worker.fetch_pending(self.db)
        # RUN・TAX もテーマにはなっているが、まだ調査するには小さすぎない
        # (どちらも size=5 >= MIN_CLUSTER_SIZE=3 なので、実は 3 件とも対象になる)
        self.assertEqual(len(pending), 3)
        cluster = next(c for c in pending if c["id"] == self.cluster_for(PLANT)["id"])

        fake_search.set("自作 潅水", [hit("https://a.example/diy", "潅水キットを自作した")])
        fake_search.set("土壌水分センサー 精度",
                        [hit("https://b.example/sensor", "静電容量式センサーの精度")])
        fake_search.set("自動潅水 失敗", [hit("https://c.example/fail", "自動潅水をやめた理由")])

        services.script_llm(
            json.dumps({
                "prior_art": ["自作 潅水"],
                "evidence": ["土壌水分センサー 精度"],
                "counter": ["自動潅水 失敗"],
            }, ensure_ascii=False),
            json.dumps({
                "picks": [
                    {"index": 0, "stance": "supports", "reason": "作例がある"},
                    {"index": 2, "stance": "challenges", "reason": "失敗談"},
                ]
            }, ensure_ascii=False),
            json.dumps({"summary": "自作の作例はあるが、根腐れの失敗談もある"}, ensure_ascii=False),
            json.dumps({"summary": "湿らせすぎて枯らした、という報告がある"}, ensure_ascii=False),
        )

        research_worker.research(self.db, cluster)

        self.assertEqual(
            sorted(fake_search.queries),
            sorted(["自作 潅水", "土壌水分センサー 精度", "自動潅水 失敗"]),
        )

        findings = self.db.execute(
            "SELECT query_kind, url, stance, summary FROM findings "
            "WHERE cluster_id = ? ORDER BY id", (cluster["id"],),
        ).fetchall()
        self.assertEqual([f["url"] for f in findings],
                         ["https://a.example/diy", "https://c.example/fail"])
        self.assertEqual([f["query_kind"] for f in findings], ["prior_art", "counter"])
        self.assertEqual([f["stance"] for f in findings], ["supports", "challenges"])
        # センサーの記事は選別で落とされている (picks に index 1 が無い)
        self.assertNotIn("https://b.example/sensor", [f["url"] for f in findings])

        row = self.db.execute(
            "SELECT researched_at, researched_member_hash FROM clusters WHERE id = ?",
            (cluster["id"],),
        ).fetchone()
        self.assertIsNotNone(row["researched_at"])
        self.assertEqual(row["researched_member_hash"], cluster["member_hash"])

        note = self.theme_note(cluster["uid"])
        self.assertIn("## 参考情報", note)
        # challenges (反証) を先頭に出す
        self.assertIn("▲ [自動潅水をやめた理由](https://c.example/fail)", note)
        self.assertIn("◯ [潅水キットを自作した](https://a.example/diy)", note)
        self.assertLess(note.index("▲"), note.index("◯"))

    def test_03_researched_theme_is_not_picked_again_within_the_week(self) -> None:
        cluster_id = self.cluster_for(PLANT)["id"]
        pending = [c["id"] for c in research_worker.fetch_pending(self.db, limit=10)]
        self.assertNotIn(cluster_id, pending)

    # --- 3. 前回調査から 1 週間経ち、メンバーが増えた場合の日付比較 -------------
    def test_04_interval_boundary_uses_matching_timestamp_format(self) -> None:
        """researched_at は 'T'/'Z' 付きの ISO8601。datetime('now') と比べると
        書式が違うために同日でも「まだ新しい」と誤判定していた不具合の回帰テスト。"""
        telegram_ingest.handle_message(self.db, message(199, PLANT_LATER))
        self.embed_all()
        cluster_worker.run(self.db)

        cluster = self.cluster_for(PLANT + [PLANT_LATER])
        self.assertEqual(cluster["size"], 6, "メンバーが増えているはず")

        with self.db:
            self.db.execute(
                "UPDATE clusters SET researched_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now','-7 days','-1 seconds') WHERE id = ?",
                (cluster["id"],),
            )

        pending = [c["id"] for c in research_worker.fetch_pending(self.db, limit=10)]
        self.assertIn(cluster["id"], pending)

    # --- 4. LLM の壊れた選別応答・URL の重複除外 ---------------------------
    def test_05_survives_malformed_picks_and_skips_known_urls(self) -> None:
        cluster = self.cluster_for(PLANT + [PLANT_LATER])

        fake_search.set("自動潅水 自作 2", [
            hit("https://a.example/diy", "潅水キットを自作した"),   # 既知 URL。除外される
            hit("https://d.example/new", "新しい作例"),
        ])

        services.script_llm(
            json.dumps({"prior_art": ["自動潅水 自作 2"]}, ensure_ascii=False),
            json.dumps({
                "picks": [
                    {"index": "0", "stance": "support", "reason": "表記ゆれ"},  # stance ゆれ
                    {"index": 99},                                             # 範囲外
                    "ごみ",                                                     # dict でない
                    {"index": -1},                                             # 負数
                ]
            }, ensure_ascii=False),
            json.dumps({"summary": "新しい作例の要約"}, ensure_ascii=False),
        )

        research_worker.research(self.db, cluster)

        findings = self.db.execute(
            "SELECT url, stance FROM findings WHERE cluster_id = ? ORDER BY id",
            (cluster["id"],),
        ).fetchall()
        urls = [f["url"] for f in findings]
        self.assertEqual(urls.count("https://a.example/diy"), 1, "重複 URL を再登録しない")
        self.assertIn("https://d.example/new", urls)
        self.assertIsNone(
            dict(zip(urls, [f["stance"] for f in findings]))["https://d.example/new"],
            "CHECK 制約を通らない stance は NULL に落とす",
        )

    # --- 5. 参考情報の通知 (反証を先に) -------------------------------------
    def test_06_findings_are_notified_challenges_first(self) -> None:
        sent.clear()
        n = notify_worker.tick_findings(self.db)
        self.assertEqual(n, 3)
        self.assertIn("▲", sent[0]["text"])
        self.assertIn("自動潅水をやめた理由", sent[0]["text"])

    def test_07_findings_are_notified_only_once(self) -> None:
        self.assertEqual(notify_worker.tick_findings(self.db), 0)

    # --- 6. 人の判断: 役に立った / 不要 -------------------------------------
    def test_08_useful_verdict_keeps_entry_with_a_checkmark(self) -> None:
        finding = self.db.execute(
            "SELECT id, cluster_id FROM findings WHERE url = ?",
            ("https://d.example/new",),
        ).fetchone()

        telegram_ingest.handle_callback(
            self.db, {"id": "cb1", "data": f"fnd:{finding['id']}:u"}
        )

        self.assertEqual(
            self.db.execute("SELECT verdict FROM findings WHERE id = ?",
                            (finding["id"],)).fetchone()["verdict"],
            "useful",
        )
        cluster = self.db.execute(
            "SELECT uid FROM clusters WHERE id = ?", (finding["cluster_id"],)
        ).fetchone()
        note = self.theme_note(cluster["uid"])
        self.assertIn("https://d.example/new", note)
        self.assertIn("✓", note)

    def test_09_not_useful_verdict_removes_entry_from_theme_note(self) -> None:
        finding = self.db.execute(
            "SELECT id, cluster_id FROM findings WHERE url = ?",
            ("https://a.example/diy",),
        ).fetchone()

        telegram_ingest.handle_callback(
            self.db, {"id": "cb2", "data": f"fnd:{finding['id']}:n"}
        )

        self.assertEqual(
            self.db.execute("SELECT verdict FROM findings WHERE id = ?",
                            (finding["id"],)).fetchone()["verdict"],
            "not_useful",
        )
        cluster = self.db.execute(
            "SELECT uid FROM clusters WHERE id = ?", (finding["cluster_id"],)
        ).fetchone()
        note = self.theme_note(cluster["uid"])
        self.assertNotIn("https://a.example/diy", note, "不要と判断したものは載せない")

    # --- 7. 週次ダイジェスト -------------------------------------------------
    def test_10_undecided_count_needs_an_actual_candidate(self) -> None:
        """候補ゼロで通知済みフラグだけ立ったメモを「判断待ち」に数えない回帰テスト。"""
        for i, text in enumerate([DUP_A, DUP_B], start=900):
            telegram_ingest.handle_message(self.db, message(i, text))
        self.embed_all()

        before = digest_worker.count_undecided(self.db, digest_worker.since())

        sent.clear()
        # tick_ideas は LIMIT 付きの 1 バッチ分しか処理しない。これまでのテストで
        # 通知が要らないメモも溜まっているので、production と同じく涸れるまで回す
        while notify_worker.tick_ideas(self.db) > 0:
            pass
        self.assertTrue(any("似ているメモ" in s.get("text", "") for s in sent),
                        "ほぼ同一文なので通知されるはず")

        after = digest_worker.count_undecided(self.db, digest_worker.since())
        self.assertEqual(after, before + 1)

        keyboard = [s for s in sent if "似ているメモ" in s.get("text", "")][0][
            "reply_markup"]["inline_keyboard"][0]
        confirm = next(b["callback_data"] for b in keyboard if ":c:" in b["callback_data"])
        telegram_ingest.handle_callback(self.db, {"id": "cb3", "data": confirm})

        self.assertEqual(
            digest_worker.count_undecided(self.db, digest_worker.since()), before,
            "判断が付いたら判断待ちから外れる",
        )

    def test_11_digest_renders_and_sends(self) -> None:
        services.script_llm(json.dumps({"intro": "今週は自動水やりまわりに関心が向いた"},
                                       ensure_ascii=False))
        sent.clear()

        digest_worker.main()

        digests = list((common.REPO_ROOT / "digests").glob("*.md"))
        self.assertEqual(len(digests), 1)
        markdown = digests[0].read_text(encoding="utf-8")

        self.assertIn("今週は自動水やりまわりに関心が向いた", markdown)
        self.assertIn("## 集まった参考情報", markdown)
        self.assertIn("▲ [自動潅水をやめた理由](https://c.example/fail)", markdown)
        # 読み上げ原稿には反証だけを拾う。リンクや記号は含めない
        script = markdown.split("## 読み上げ原稿")[1]
        self.assertIn("気になる指摘があります", script)
        self.assertNotIn("https://", script)

        self.assertTrue(sent, "Telegram に送っていない")
        self.assertIn("週次ダイジェスト", sent[-1]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
