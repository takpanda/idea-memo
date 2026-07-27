"""
Web UI の通し試験。

外出先の iPhone から読む側なので、DB を直接組み立てずに
取り込み -> 埋め込み -> クラスタリング -> テーマ命名 まで実際に流してから
HTTP で叩く。埋め込みサーバーと LLM は tests/fake_services.py が代役。

  python -m unittest discover -s tests
  # 要 sqlite-vec, numpy, scikit-learn, fastapi, httpx
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

CHAT_ID = 626262

# 語彙を共有する 3 グループ。偽埋め込みでもこの粒度なら 3 つの島になる
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

# どのクラスタにも寄らない単発。未分類の絞り込みとページングに使う
LONER = "駅前のパン屋に並ばずに済む時間帯をメモしておく"
LIBRARY = "図書館の返却期限を前日に知らせる仕組みがほしい"
PARENT = "帰省の荷物をまとめる手順を毎回忘れる"
REPLY = "充電器とコンタクトの替えを最初に入れる"

# ヒットが末尾にある長文。抜粋がヒット位置に寄ることを見る
LONG = "冒頭は関係のない話。" * 20 + "結論として型番は ZX9000 で決まり。"

tmp: tempfile.TemporaryDirectory
services: FakeServices


def setUpModule() -> None:
    """環境変数はワーカーの import 時に読まれるので、先に確定させる。"""
    global tmp, services
    global common, cluster_worker, embed_worker, telegram_ingest, theme_writer, client

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
        "SIM_THRESHOLD": "0.99",       # 類似通知はここの担当ではない
    })

    fresh_worker_imports()

    import init_db
    assert init_db.main() == 0, "init_db failed"

    import cluster_worker
    import common
    import embed_worker
    import telegram_ingest
    import theme_writer
    import web
    from fastapi.testclient import TestClient

    telegram_ingest.api = lambda method, timeout=65, **params: True
    client = TestClient(web.app)


def tearDownModule() -> None:
    services.__exit__()
    tmp.cleanup()


def message(message_id: int, text: str, date: int | None = None,
            reply_to: int | None = None) -> dict:
    msg = {
        "message_id": message_id,
        "date": date if date is not None else 1_770_000_000 + message_id,
        "chat": {"id": CHAT_ID, "type": "private"},
        "text": text,
    }
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return msg


class WebUI(unittest.TestCase):
    """1 本のシナリオを順に流す。ステップ間で状態を引き継ぐので順序が要る。"""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = common.connect(with_vec=True)

    # --- ヘルパ ------------------------------------------------------
    def uid_of(self, body: str) -> str:
        row = self.db.execute(
            "SELECT uid FROM ideas WHERE body = ?", (body,)
        ).fetchone()
        self.assertIsNotNone(row, f"idea not found: {body[:16]}")
        return row["uid"]

    def theme_uid(self, body: str) -> str:
        """そのメモを抱えているクラスタの uid。HDBSCAN の結果に依存しない引き方。"""
        row = self.db.execute(
            """
            SELECT c.uid FROM ideas i
                   JOIN idea_clusters ic ON ic.idea_id = i.id
                   JOIN clusters c ON c.id = ic.cluster_id
            WHERE  i.body = ?
            """,
            (body,),
        ).fetchone()
        self.assertIsNotNone(row, f"clustered theme not found for: {body[:16]}")
        return row["uid"]

    def get(self, path: str, expect: int = 200) -> dict:
        res = client.get(path)
        self.assertEqual(res.status_code, expect, res.text)
        return res.json() if expect == 200 else {}

    # --- 1. データを用意する ---------------------------------------------
    def test_01_pipeline_fills_the_db(self) -> None:
        for i, text in enumerate(RUN + TAX + PLANT + [LONER, LONG], start=100):
            telegram_ingest.handle_message(self.db, message(i, text))

        # 同じ秒に 2 件。カーソルが captured_at だけだと境界で取りこぼす
        telegram_ingest.handle_message(self.db, message(200, LIBRARY, date=1_770_009_000))
        telegram_ingest.handle_message(self.db, message(201, PARENT, date=1_770_009_000))
        telegram_ingest.handle_message(
            self.db, message(202, REPLY, date=1_770_009_100, reply_to=201)
        )

        while embed_worker.tick(self.db) > 0:
            pass
        cluster_worker.run(self.db)
        theme_writer.tick(self.db)

        self.assertEqual(
            self.db.execute("SELECT COUNT(*) AS n FROM ideas").fetchone()["n"], 20
        )
        # 3 グループがそれぞれ 1 つのテーマにまとまっている
        self.assertEqual(
            len({self.theme_uid(RUN[0]), self.theme_uid(TAX[0]), self.theme_uid(PLANT[0])}), 3
        )

    # --- 2. 1 枚の HTML を返すこと ----------------------------------------
    def test_02_index_is_self_contained(self) -> None:
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        for marker in ("IDEA MEMO", "#/search", "#/themes", "/api/ideas"):
            self.assertIn(marker, res.text)
        # ビルドステップを持たない = 外部アセットを引かない
        self.assertNotIn("<script src=", res.text)
        self.assertNotIn("<link rel=\"stylesheet\"", res.text)

    # --- 3. メモ一覧 -----------------------------------------------------
    def test_03_list_is_newest_first(self) -> None:
        data = self.get("/api/ideas")
        self.assertEqual(data["total"], 20)
        self.assertIsNone(data["scope"])
        self.assertEqual(len(data["ideas"]), 20)      # 既定 30 件に収まる
        self.assertIsNone(data["next"])

        stamps = [i["captured_at"] for i in data["ideas"]]
        self.assertEqual(stamps, sorted(stamps, reverse=True))
        self.assertEqual(data["ideas"][0]["uid"], self.uid_of(REPLY))

        first = data["ideas"][0]
        self.assertEqual(first["snippet"], REPLY)
        self.assertEqual(first["status"], "inbox")
        self.assertEqual(first["input_kind"], "text")
        self.assertEqual(first["attached"], 0)
        self.assertNotIn("body", first, "一覧に全文は載せない")

        # クラスタに入っているメモにはテーマ名が付く
        run = next(i for i in data["ideas"] if i["uid"] == self.uid_of(RUN[0]))
        self.assertEqual(run["cluster_uid"], self.theme_uid(RUN[0]))
        self.assertTrue(run["cluster_name"])

    def test_04_long_body_is_truncated_in_the_list(self) -> None:
        data = self.get("/api/ideas")
        item = next(i for i in data["ideas"] if i["uid"] == self.uid_of(LONG))
        self.assertLess(len(item["snippet"]), len(LONG))
        self.assertTrue(item["snippet"].endswith("…"))
        self.assertTrue(LONG.startswith(item["snippet"][:-1]))

    def test_05_cursor_walks_every_page_once(self) -> None:
        """同じ秒のメモを跨いでも、飛ばさず重複させないこと。"""
        seen, cursor, pages = [], None, 0
        while True:
            path = "/api/ideas?limit=3" + (f"&before={cursor}" if cursor else "")
            data = self.get(path)
            self.assertEqual(data["total"], 20 if pages == 0 else None,
                             "総数は先頭ページだけ")
            seen += [i["uid"] for i in data["ideas"]]
            pages += 1
            cursor = data["next"]
            if not cursor:
                break
            self.assertLess(pages, 20, "カーソルが進んでいない")

        self.assertEqual(len(seen), 20)
        self.assertEqual(len(set(seen)), 20, "同じメモが 2 ページに出た")
        self.assertEqual(seen, [i["uid"] for i in self.get("/api/ideas")["ideas"]])

    def test_06_broken_cursor_is_rejected(self) -> None:
        self.get("/api/ideas?before=2026-07-01T00:00:00Z", expect=400)
        self.get("/api/ideas?before=2026-07-01T00:00:00Z|abc", expect=400)
        self.get("/api/ideas?status=nonsense", expect=400)

    # --- 4. status の絞り込みと変更 ----------------------------------------
    def test_07_status_change_hides_it_from_the_default_list(self) -> None:
        uid = self.uid_of(LONER)
        res = client.post(f"/api/ideas/{uid}/status", json={"status": "archived"})
        self.assertEqual(res.status_code, 200, res.text)

        active = self.get("/api/ideas")
        self.assertEqual(active["total"], 19)
        self.assertNotIn(uid, [i["uid"] for i in active["ideas"]])

        archived = self.get("/api/ideas?status=archived")
        self.assertEqual([i["uid"] for i in archived["ideas"]], [uid])
        self.assertIn(uid, [i["uid"] for i in self.get("/api/ideas?status=all")["ideas"]])

        # Markdown は DB からの再生成。Obsidian 側にも同じ status が出る
        path = common.REPO_ROOT / self.db.execute(
            "SELECT file_path FROM ideas WHERE uid = ?", (uid,)
        ).fetchone()["file_path"]
        self.assertIn("status: archived", path.read_text(encoding="utf-8"))

        client.post(f"/api/ideas/{uid}/status", json={"status": "inbox"})

    def test_08_status_endpoint_guards(self) -> None:
        uid = self.uid_of(LONER)
        self.assertEqual(
            client.post(f"/api/ideas/{uid}/status", json={"status": "burned"}).status_code, 400
        )
        self.assertEqual(
            client.post("/api/ideas/NOPE/status", json={"status": "kept"}).status_code, 404
        )

    # --- 5. テーマで絞り込む ----------------------------------------------
    def test_09_cluster_filter_and_unclustered(self) -> None:
        theme = self.theme_uid(TAX[0])
        members = {row["uid"] for row in self.db.execute(
            """
            SELECT i.uid FROM idea_clusters ic
                   JOIN ideas i ON i.id = ic.idea_id
                   JOIN clusters c ON c.id = ic.cluster_id
            WHERE  c.uid = ?
            """,
            (theme,),
        )}
        data = self.get(f"/api/ideas?cluster={theme}")
        self.assertEqual(data["scope"]["uid"], theme)
        self.assertTrue(data["scope"]["name"])
        self.assertEqual({i["uid"] for i in data["ideas"]}, members)
        self.assertLessEqual({self.uid_of(b) for b in TAX}, members)

        # どのメモがノイズに落ちるかは HDBSCAN 次第なので DB を正とする
        noise = {row["uid"] for row in self.db.execute(
            """
            SELECT uid FROM ideas i
            WHERE  i.status != 'archived'
                   AND NOT EXISTS (SELECT 1 FROM idea_clusters ic WHERE ic.idea_id = i.id)
            """
        )}
        unclustered = self.get("/api/ideas?cluster=none")
        self.assertEqual(unclustered["scope"], {"uid": "none", "name": None})
        self.assertEqual({i["uid"] for i in unclustered["ideas"]}, noise)
        # テーマ画面に出る未分類の件数と一致する
        self.assertEqual(
            self.get("/api/clusters")["unclustered"], len(unclustered["ideas"])
        )

        self.get("/api/ideas?cluster=missing-uid", expect=404)

    # --- 6. メモ詳細 -----------------------------------------------------
    def test_10_detail_carries_the_whole_body(self) -> None:
        d = self.get(f"/api/ideas/{self.uid_of(LONG)}")
        self.assertEqual(d["idea"]["body"], LONG, "詳細では切らない")
        self.assertEqual(d["idea"]["source"], "telegram")
        self.assertTrue(d["idea"]["file_path"].endswith(".md"))
        self.assertEqual(d["attachments"], [])
        self.assertEqual((d["parent"], d["replies"], d["related"]), (None, [], []))

        self.get("/api/ideas/NOPE", expect=404)

    def test_11_detail_shows_the_reply_thread(self) -> None:
        parent = self.get(f"/api/ideas/{self.uid_of(PARENT)}")
        self.assertEqual([r["uid"] for r in parent["replies"]], [self.uid_of(REPLY)])
        self.assertEqual(parent["replies"][0]["snippet"], REPLY)

        reply = self.get(f"/api/ideas/{self.uid_of(REPLY)}")
        self.assertEqual(reply["parent"]["uid"], self.uid_of(PARENT))

    def test_12_detail_shows_approved_relations_and_attachments(self) -> None:
        src, dst = sorted(
            self.db.execute(
                "SELECT id FROM ideas WHERE body IN (?, ?)", (LIBRARY, LONER)
            ).fetchall()[i]["id"] for i in (0, 1)
        )
        with self.db:
            # 人が承認した関連と、却下した関連。出るのは承認したものだけ
            self.db.execute(
                "INSERT INTO relations(src_id, dst_id, kind, score, verdict) "
                "VALUES (?, ?, 'similar', 0.9, 'confirmed')", (src, dst)
            )
            self.db.execute(
                "INSERT INTO attachments(idea_id, kind, mime, bytes, file_path, meta_json) "
                "VALUES (?, 'voice', 'audio/ogg', 2048, 'blobs/x.ogg', ?)",
                (src, json.dumps({"duration": 12, "sha256": "abc"})),
            )

        src_uid = self.db.execute(
            "SELECT uid FROM ideas WHERE id = ?", (src,)
        ).fetchone()["uid"]
        dst_uid = self.db.execute(
            "SELECT uid FROM ideas WHERE id = ?", (dst,)
        ).fetchone()["uid"]

        d = self.get(f"/api/ideas/{src_uid}")
        self.assertEqual([r["uid"] for r in d["related"]], [dst_uid])
        self.assertEqual(d["related"][0]["kind"], "similar")
        self.assertEqual(d["attachments"], [{"kind": "voice", "bytes": 2048, "duration": 12}])

        # 関連は無向。反対側から開いても同じ相手が出る
        self.assertEqual(
            [r["uid"] for r in self.get(f"/api/ideas/{dst_uid}")["related"]], [src_uid]
        )
        # 添付を持つメモは一覧でも分かる
        listed = next(i for i in self.get("/api/ideas?status=all")["ideas"]
                      if i["uid"] == src_uid)
        self.assertEqual(listed["attached"], 1)

    # --- 7. 検索 --------------------------------------------------------
    def test_13_search_hits_both_lanes(self) -> None:
        data = self.get("/api/search?q=" + "医療費控除")
        self.assertTrue(data["vector"], "埋め込みサーバーが応答していない")
        self.assertTrue(data["keyword"])

        uids = [r["uid"] for r in data["results"]]
        for body in TAX:
            self.assertIn(self.uid_of(body), uids)
        top = data["results"][0]
        self.assertGreaterEqual(top["score"], data["results"][-1]["score"])
        self.assertIn("医療費控除", top["snippet"])
        self.assertTrue(top["cluster_name"])

    def test_14_snippet_follows_the_hit(self) -> None:
        """ヒットが末尾にある長文でも、何に当たったのかが抜粋に出ること。"""
        data = self.get("/api/search?q=ZX9000")
        hit = next(r for r in data["results"] if r["uid"] == self.uid_of(LONG))
        self.assertIn("ZX9000", hit["snippet"])
        self.assertTrue(hit["snippet"].startswith("…"))

    def test_15_search_edges(self) -> None:
        self.assertEqual(self.get("/api/search?q=%20")["results"], [])
        # trigram は 3 文字未満を索引できないので LIKE に落ちる
        two = self.get("/api/search?q=図書")
        self.assertIn(self.uid_of(LIBRARY), [r["uid"] for r in two["results"]])

        # ベクトル検索は距離で足切りしないので、無関係な語でも「一番近い」メモが
        # 並ぶ。ここで確かめられるのは全文側が空になることまで
        nonsense = self.get("/api/search?q=" + "存在しない語句" * 3)
        self.assertFalse(nonsense["keyword"])
        self.assertTrue(nonsense["vector"])

    # --- 8. テーマ一覧と中身 ------------------------------------------------
    def test_16_themes_list(self) -> None:
        data = self.get("/api/clusters")
        self.assertEqual(len(data["clusters"]), 3)
        sizes = [c["size"] for c in data["clusters"]]
        self.assertEqual(sizes, sorted(sizes, reverse=True), "大きいテーマから並べる")
        for c in data["clusters"]:
            self.assertTrue(c["name"] and c["summary"], "命名前のテーマが出ている")
            self.assertEqual(c["findings"], 0)

    def test_17_theme_detail_puts_counters_first(self) -> None:
        theme = self.theme_uid(PLANT[0])
        cluster_id = self.db.execute(
            "SELECT id FROM clusters WHERE uid = ?", (theme,)
        ).fetchone()["id"]
        with self.db:
            for kind, url, stance, title in (
                ("evidence", "https://example.com/a", "supports", "自動潅水の実例"),
                ("counter", "https://example.com/b", "challenges", "枯れた事例"),
            ):
                self.db.execute(
                    "INSERT INTO findings(cluster_id, query_kind, query, url, title, "
                    "site, summary, stance) VALUES (?, ?, '土壌 湿度', ?, ?, "
                    "'example.com', '要約', ?)",
                    (cluster_id, kind, url, title, stance),
                )

        d = self.get(f"/api/clusters/{theme}")
        self.assertEqual(d["cluster"]["uid"], theme)
        self.assertNotIn("id", d["cluster"], "内部 id は出さない")
        self.assertEqual({m["uid"] for m in d["members"]},
                         {self.uid_of(b) for b in PLANT})
        self.assertNotIn("body", d["members"][0])
        self.assertTrue(d["members"][0]["snippet"])

        # 反証を先頭に置く。賛成材料だけ読んで終わりにならないように
        self.assertEqual([f["stance"] for f in d["findings"]], ["challenges", "supports"])
        self.assertEqual(self.get("/api/clusters")["clusters"][0]["findings"] or
                         next(c["findings"] for c in self.get("/api/clusters")["clusters"]
                              if c["uid"] == theme), 2)

        self.get("/api/clusters/missing-uid", expect=404)

    def test_18_theme_list_can_be_reordered(self) -> None:
        recent = self.get("/api/clusters?sort=recent")["clusters"]
        stamps = [c["updated_at"] for c in recent]
        self.assertEqual(stamps, sorted(stamps, reverse=True), "動いた順に並べる")

        names = [c["name"] for c in self.get("/api/clusters?sort=name")["clusters"]]
        self.assertEqual(names, sorted(names, key=str.lower))

        self.get("/api/clusters?sort=bogus", expect=400)

        # 一覧で見たいのは「反証が来ているか」。件数と別に数える
        theme = self.theme_uid(PLANT[0])
        card = next(c for c in recent if c["uid"] == theme)
        self.assertEqual((card["findings"], card["challenges"]), (2, 1))

    def test_19_dropping_a_finding_keeps_it_out_of_the_note(self) -> None:
        theme = self.theme_uid(PLANT[0])
        dropped = next(f for f in self.get(f"/api/clusters/{theme}")["findings"]
                       if f["stance"] == "challenges")

        res = client.post(f"/api/findings/{dropped['id']}/verdict",
                          json={"verdict": "not_useful"})
        self.assertEqual(res.status_code, 200, res.text)

        # 判断を尊重してテーマノートからは落とす。ただし行は消さない
        path = common.REPO_ROOT / self.db.execute(
            "SELECT file_path FROM clusters WHERE uid = ?", (theme,)
        ).fetchone()["file_path"]
        self.assertNotIn(dropped["url"], path.read_text(encoding="utf-8"))

        card = next(c for c in self.get("/api/clusters")["clusters"]
                    if c["uid"] == theme)
        self.assertEqual((card["findings"], card["challenges"]), (1, 0))
        self.assertEqual(
            next(f for f in self.get(f"/api/clusters/{theme}")["findings"]
                 if f["id"] == dropped["id"])["verdict"],
            "not_useful", "俯瞰では外したものも残す")

        # 押し直しで取り消せる (誤爆がノートに残り続けない)
        client.post(f"/api/findings/{dropped['id']}/verdict", json={"verdict": None})
        self.assertIn(dropped["url"], path.read_text(encoding="utf-8"))

        self.assertEqual(client.post(f"/api/findings/{dropped['id']}/verdict",
                                     json={"verdict": "meh"}).status_code, 400)
        self.assertEqual(client.post("/api/findings/999999/verdict",
                                     json={"verdict": "useful"}).status_code, 404)

    def test_20_renaming_a_theme_locks_the_name(self) -> None:
        theme = self.theme_uid(PLANT[0])
        res = client.post(f"/api/clusters/{theme}/name", json={"name": "ベランダの水やり"})
        self.assertEqual(res.status_code, 200, res.text)

        row = self.db.execute(
            "SELECT name, name_locked, file_path FROM clusters WHERE uid = ?", (theme,)
        ).fetchone()
        self.assertEqual(row["name"], "ベランダの水やり")
        self.assertEqual(row["name_locked"], 1, "人が付けた名前を LLM に返させない")

        note = (common.REPO_ROOT / row["file_path"]).read_text(encoding="utf-8")
        self.assertIn("ベランダの水やり", note)
        self.assertIn("name_locked: true", note)

        self.assertEqual(
            client.post(f"/api/clusters/{theme}/name", json={"name": "  "}).status_code, 400
        )
        self.assertEqual(
            client.post("/api/clusters/nope/name", json={"name": "x"}).status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
