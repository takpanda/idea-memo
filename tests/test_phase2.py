"""
Phase 2 の通し試験。

取り込み -> 'トピック: ' 埋め込み -> HDBSCAN -> クラスタ ID 継承 ->
テーマ命名 -> テーマノート出力 -> 人の名前が機械に勝つ、までを実際に動かす。

Mac mini の埋め込みサーバーと GPU ノードの LLM は tests/fake_services.py が
本物と同じ HTTP 契約で代役を務めるので、実機もモデルも要らない。

偽の埋め込みは文字 3-gram のハッシュなので意味は持たないが、
語彙を共有する 3 グループはちゃんと 3 つの島になる。HDBSCAN が毎回
ラベルを振り直すことも本物と同じなので、ID 継承の検証はこれで足りる。

  python -m unittest discover -s tests      # 要 sqlite-vec, numpy, scikit-learn
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

CHAT_ID = 515151

# 語彙を共有する 3 グループ + どこにも寄らない 1 件。
# 偽埋め込みでのグループ内 cos は 0.3-0.6、グループ間は 0.2 未満
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
LONER = "駅前のパン屋に並ばずに済む時間帯をメモしておく"
RUN_LATER = "川沿いのランニングコースを地図アプリで週末ごとに走った距離とあわせて残す"

HUMAN_NAME = "川沿いランニングの地図化"

tmp: tempfile.TemporaryDirectory
services: FakeServices


def setUpModule() -> None:
    """環境変数はワーカーの import 時に読まれるので、先に確定させる。"""
    global tmp, services
    global common, cluster_worker, embed_worker, telegram_ingest, theme_writer

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
        # 類似通知は Phase 1 の担当。ここでは走らせない
        "SIM_THRESHOLD": "0.99",
    })

    fresh_worker_imports()

    import init_db
    assert init_db.main() == 0, "init_db failed"

    import cluster_worker
    import common
    import embed_worker
    import telegram_ingest
    import theme_writer

    telegram_ingest.api = lambda method, timeout=65, **params: True


def tearDownModule() -> None:
    services.__exit__()
    tmp.cleanup()


def message(message_id: int, text: str) -> dict:
    return {"message_id": message_id, "date": 1_770_000_000 + message_id,
            "chat": {"id": CHAT_ID, "type": "private"}, "text": text}


class Phase2Pipeline(unittest.TestCase):
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

    def members_of(self, cluster_id: int) -> set[int]:
        return {r["idea_id"] for r in self.db.execute(
            "SELECT idea_id FROM idea_clusters WHERE cluster_id = ?", (cluster_id,))}

    def cluster_for(self, bodies: list[str]):
        """そのグループのメモを全部同じクラスタが抱えていることを確かめて返す。"""
        wanted = self.ids_of(bodies)
        rows = self.db.execute(
            "SELECT DISTINCT cluster_id FROM idea_clusters WHERE idea_id IN "
            f"({','.join('?' * len(wanted))})", tuple(sorted(wanted))
        ).fetchall()
        self.assertEqual(len(rows), 1, f"1 クラスタにまとまっていない: {rows}")
        cluster_id = rows[0]["cluster_id"]
        self.assertEqual(self.members_of(cluster_id), wanted)
        return self.db.execute(
            "SELECT * FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()

    def embed_all(self) -> None:
        while embed_worker.tick(self.db) > 0:
            pass

    def theme_path(self, cluster) -> Path:
        return common.REPO_ROOT / f"themes/{cluster['uid']}.md"

    # --- 1. 埋め込み --------------------------------------------------
    def test_01_topic_vectors_are_built(self) -> None:
        for i, text in enumerate(RUN + TAX + PLANT + [LONER], start=100):
            telegram_ingest.handle_message(self.db, message(i, text))
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) AS n FROM ideas").fetchone()["n"], 16
        )

        self.embed_all()

        # クラスタリングは 'トピック: ' 版を使う。意味検索用とは別ベクトル
        n = self.db.execute(
            "SELECT COUNT(*) AS n FROM vec_ideas_ruri_v3_310m_topic"
        ).fetchone()["n"]
        self.assertEqual(n, 16)
        self.assertEqual(
            {r["prefix"] for r in self.db.execute("SELECT DISTINCT prefix FROM embeddings")},
            {"", "トピック: "},
        )

        semantic, topic = (self.db.execute(
            f"SELECT vec_to_json(embedding) AS v FROM {table} ORDER BY idea_id LIMIT 1"
        ).fetchone()["v"] for table in
            ("vec_ideas_ruri_v3_310m", "vec_ideas_ruri_v3_310m_topic"))
        self.assertNotEqual(semantic, topic, "プレフィックス違いが同じベクトルになっている")

    # --- 2. 初回クラスタリング ------------------------------------------
    def test_02_first_run_forms_clusters(self) -> None:
        cluster_worker.run(self.db)

        clusters = self.db.execute(
            "SELECT id, uid, size FROM clusters WHERE closed_at IS NULL"
        ).fetchall()
        self.assertEqual(len(clusters), 3)
        for group in (RUN, TAX, PLANT):
            self.assertEqual(self.cluster_for(group)["size"], 5)

        # どこにも寄らないメモは行を作らない = まだどのテーマにも属さない
        loner = self.ids_of([LONER]).pop()
        self.assertIsNone(self.db.execute(
            "SELECT idea_id FROM idea_clusters WHERE idea_id = ?", (loner,)
        ).fetchone())

        run = self.db.execute(
            "SELECT * FROM cluster_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(
            (run["n_ideas"], run["n_clusters"], run["n_noise"], run["n_new"], run["n_closed"]),
            (16, 3, 1, 3, 0),
        )
        self.assertEqual(json.loads(run["params_json"])["min_cluster_size"], 3)
        self.assertIsNotNone(run["finished_at"])

    # --- 3. テーマ命名 -------------------------------------------------
    def test_03_themes_are_named_and_written(self) -> None:
        self.assertEqual(theme_writer.tick(self.db), 3)

        for group in (RUN, TAX, PLANT):
            cluster = self.cluster_for(group)
            self.assertEqual(cluster["name"], "テーマ5件")
            self.assertIn("5件のメモ", cluster["summary"])
            self.assertEqual(cluster["file_path"], f"themes/{cluster['uid']}.md")
            # 命名したときのメンバー集合を控えて、次回の再生成判定に使う
            self.assertEqual(cluster["named_member_hash"], cluster["member_hash"])

            note = self.theme_path(cluster).read_text(encoding="utf-8")
            self.assertIn(f"cluster_uid: {cluster['uid']}", note)
            self.assertIn("name_locked: false", note)
            self.assertIn(f"# {cluster['name']}", note)
            for body in group:
                uid = self.db.execute(
                    "SELECT uid FROM ideas WHERE body = ?", (body,)
                ).fetchone()["uid"]
                self.assertIn(f"  - uid: {uid}", note)
                self.assertIn(body[:20], note)

        # プロンプトにはそのクラスタのメモだけが入る
        prompts = [c["prompt"] for c in services.calls if c["endpoint"] == "llm"]
        self.assertEqual(len(prompts), 3)
        run_prompt = next(p for p in prompts if RUN[0][:12] in p)
        self.assertNotIn(TAX[0][:12], run_prompt)
        self.assertNotIn(LONER[:10], run_prompt)

    def test_04_naming_converges(self) -> None:
        """メンバーが変わっていなければ 2 周目は何もしない。"""
        self.assertEqual(theme_writer.tick(self.db), 0)

    # --- 4. クラスタ ID の継承 -------------------------------------------
    def test_05_ids_survive_relabeling(self) -> None:
        before = {group[0]: self.cluster_for(group)["uid"] for group in (RUN, TAX, PLANT)}
        run_cluster = self.cluster_for(RUN)

        telegram_ingest.handle_message(self.db, message(200, RUN_LATER))
        self.embed_all()
        cluster_worker.run(self.db)

        # HDBSCAN のラベルは振り直されるが、uid とテーマ名は動かない
        self.assertEqual(
            {group[0]: self.cluster_for(group + ([RUN_LATER] if group is RUN else []))["uid"]
             for group in (RUN, TAX, PLANT)},
            before,
        )
        grown = self.cluster_for(RUN + [RUN_LATER])
        self.assertEqual(grown["id"], run_cluster["id"])
        self.assertEqual(grown["size"], 6)
        self.assertEqual(grown["name"], "テーマ5件", "名前は再生成しない")
        self.assertNotEqual(grown["member_hash"], run_cluster["member_hash"])

        run = self.db.execute(
            "SELECT * FROM cluster_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual((run["n_new"], run["n_closed"]), (0, 0))

    def test_06_summary_is_regenerated_but_name_is_not(self) -> None:
        """名前が毎晩ちらつかないこと。中身が変わった要約だけ作り直すこと。"""
        self.assertEqual(theme_writer.tick(self.db), 1)

        grown = self.cluster_for(RUN + [RUN_LATER])
        self.assertEqual(grown["name"], "テーマ5件", "偽 LLM は6件と答えるが名前は据え置き")
        self.assertIn("6件のメモ", grown["summary"])
        self.assertIn(RUN_LATER[:20], self.theme_path(grown).read_text(encoding="utf-8"))

        self.assertEqual(theme_writer.tick(self.db), 0)

    # --- 5. 人の判断が機械に勝つ ------------------------------------------
    def test_07_human_name_is_absorbed_from_markdown(self) -> None:
        cluster = self.cluster_for(RUN + [RUN_LATER])
        path = self.theme_path(cluster)
        # Obsidian で front matter を書き換えた人の手つき (引用符なし)
        edited = [
            f"name: {HUMAN_NAME}" if line.startswith("name:")
            else "name_locked: true" if line.startswith("name_locked:")
            else line
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        path.write_text("\n".join(edited), encoding="utf-8")

        # メンバーが動いていなくても、人の編集は次の周回で吸い上げる
        self.assertEqual(theme_writer.tick(self.db), 1)

        locked = self.cluster_for(RUN + [RUN_LATER])
        self.assertEqual(locked["name"], HUMAN_NAME)
        self.assertEqual(locked["name_locked"], 1)
        self.assertIn(f"# {HUMAN_NAME}", path.read_text(encoding="utf-8"))
        self.assertEqual(theme_writer.tick(self.db), 0, "吸い上げは 1 回で収束する")

    def test_08_locked_name_survives_membership_change(self) -> None:
        telegram_ingest.handle_message(self.db, message(201, RUN[0] + " 高低差も出す"))
        self.embed_all()
        cluster_worker.run(self.db)
        self.assertEqual(theme_writer.tick(self.db), 1)

        cluster = self.db.execute(
            "SELECT * FROM clusters WHERE name_locked = 1"
        ).fetchone()
        self.assertEqual(cluster["name"], HUMAN_NAME, "LLM が人の名前を上書きしている")
        self.assertIn("7件のメモ", cluster["summary"], "要約は作り直す")
        self.assertIn(f"# {HUMAN_NAME}", self.theme_path(cluster).read_text(encoding="utf-8"))

    # --- 6. LLM が壊れた応答を返したとき -----------------------------------
    def test_09_broken_llm_does_not_report_progress(self) -> None:
        """進捗ゼロを返さないと supervisor が毎周回で空回りする。"""
        cluster = self.cluster_for(TAX)
        with self.db:
            self.db.execute(
                "UPDATE clusters SET named_member_hash = NULL WHERE id = ?",
                (cluster["id"],),
            )

        services.script_llm("承知しました。JSON は以下の通りです。")
        self.assertEqual(theme_writer.tick(self.db), 0)
        self.assertEqual(
            self.cluster_for(TAX)["named_member_hash"], None, "壊れた応答は確定させない"
        )

    def test_10_code_fenced_json_is_accepted(self) -> None:
        services.script_llm(
            '```json\n{"name": "確定申告の下ごしらえ", "summary": "領収書の整理"}\n```'
        )
        self.assertEqual(theme_writer.tick(self.db), 1)

        cluster = self.cluster_for(TAX)
        self.assertEqual(cluster["name"], "テーマ5件", "既に名前があるので触らない")
        self.assertEqual(cluster["summary"], "領収書の整理")
        self.assertEqual(cluster["named_member_hash"], cluster["member_hash"])

    # --- 7. 消えたクラスタ -------------------------------------------------
    def test_11_vanished_cluster_is_closed_not_deleted(self) -> None:
        keep = self.cluster_for(PLANT)["uid"]
        vanishing = self.db.execute(
            "SELECT * FROM clusters WHERE name_locked = 1"
        ).fetchone()

        # ランニング系を畳む。残り 2 件では min_cluster_size に届かない
        with self.db:
            self.db.execute(
                "UPDATE ideas SET status = 'archived' WHERE body IN (?, ?, ?, ?, ?)",
                (RUN[1], RUN[2], RUN[3], RUN[4], RUN[0] + " 高低差も出す"),
            )
        cluster_worker.run(self.db)

        closed = self.db.execute(
            "SELECT * FROM clusters WHERE id = ?", (vanishing["id"],)
        ).fetchone()
        self.assertIsNotNone(closed["closed_at"], "消えたクラスタは閉じる")
        self.assertEqual(closed["name"], HUMAN_NAME, "名前は履歴として残す")
        self.assertEqual(closed["size"], 0)
        self.assertEqual(self.members_of(closed["id"]), set())

        # 生き残りは巻き添えにしない
        self.assertEqual(self.cluster_for(PLANT)["uid"], keep)
        run = self.db.execute(
            "SELECT * FROM cluster_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual((run["n_clusters"], run["n_new"], run["n_closed"]), (2, 0, 1))

        # 閉じたクラスタは命名の対象にしない
        self.assertEqual(theme_writer.tick(self.db), 0)

    def test_12_closed_theme_note_says_so(self) -> None:
        closed = self.db.execute(
            "SELECT * FROM clusters WHERE closed_at IS NOT NULL"
        ).fetchone()
        note = self.theme_path(closed).read_text(encoding="utf-8")
        self.assertIn(f"closed_at: {closed['closed_at']}", note,
                      "Obsidian 側から見て現役のテーマと区別が付かない")

    def test_13_reformed_cluster_takes_its_old_id_back(self) -> None:
        """半年後に同じ話題が戻ってきたら、テーマ名も履歴も引き継ぐ。"""
        closed = self.db.execute(
            "SELECT * FROM clusters WHERE closed_at IS NOT NULL"
        ).fetchone()

        with self.db:
            self.db.execute("UPDATE ideas SET status = 'inbox' WHERE status = 'archived'")
        cluster_worker.run(self.db)

        reformed = self.cluster_for(RUN + [RUN_LATER, RUN[0] + " 高低差も出す"])
        self.assertEqual(reformed["id"], closed["id"], "作り直すと名前と調査履歴を失う")
        self.assertEqual(reformed["name"], HUMAN_NAME)
        self.assertIsNone(reformed["closed_at"])
        self.assertEqual(reformed["size"], 7)

        run = self.db.execute(
            "SELECT * FROM cluster_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual((run["n_clusters"], run["n_new"], run["n_closed"]), (3, 0, 0))


class ThemeNoteFrontMatter(unittest.TestCase):
    """テーマノートの front matter は書いた側と読む側で往復すること。"""

    def setUp(self) -> None:
        self.db = common.connect()
        self.cluster_id = self.db.execute(
            "SELECT id FROM clusters WHERE name_locked = 0 LIMIT 1"
        ).fetchone()["id"]
        self.before = self.db.execute(
            "SELECT name, name_locked FROM clusters WHERE id = ?", (self.cluster_id,)
        ).fetchone()

    def tearDown(self) -> None:
        with self.db:
            self.db.execute(
                "UPDATE clusters SET name = ?, name_locked = ? WHERE id = ?",
                (self.before["name"], self.before["name_locked"], self.cluster_id),
            )
        common.write_theme_markdown(self.db, self.cluster_id)

    def test_colon_in_name_does_not_break_the_front_matter(self) -> None:
        name = "メモ: 整理の仕組み #2"
        with self.db:
            self.db.execute(
                "UPDATE clusters SET name = ?, name_locked = 1 WHERE id = ?",
                (name, self.cluster_id),
            )
        path = common.write_theme_markdown(self.db, self.cluster_id)

        self.assertIn(f'name: "{name}"', path.read_text(encoding="utf-8"))
        self.assertEqual(
            common.read_theme_overrides(path), {"name": name, "name_locked": 1}
        )

    def test_unlocking_in_markdown_is_absorbed(self) -> None:
        with self.db:
            self.db.execute(
                "UPDATE clusters SET name_locked = 1 WHERE id = ?", (self.cluster_id,)
            )
        path = common.write_theme_markdown(self.db, self.cluster_id)
        path.write_text(
            path.read_text(encoding="utf-8").replace("name_locked: true",
                                                     "name_locked: false"),
            encoding="utf-8",
        )

        self.assertEqual(theme_writer.absorb_overrides(self.db), 1)
        self.assertEqual(self.db.execute(
            "SELECT name_locked FROM clusters WHERE id = ?", (self.cluster_id,)
        ).fetchone()["name_locked"], 0)
        self.assertEqual(theme_writer.absorb_overrides(self.db), 0)


class SingleTopicCorpus(unittest.TestCase):
    """話題が 1 つしか無いうちはテーマが出ない。仕様として固定しておく。

    HDBSCAN は allow_single_cluster=False が既定で、木の根は
    クラスタとして選ばない。使い始めの数週間は「似たメモばかりなのに
    テーマが 1 つも出ない」ことになるが、不具合ではない。
    有効にすると 3 話題あっても全部 1 個に潰れるので、既定のまま使う。
    """

    def vectors(self, texts: list[str]):
        import numpy as np
        from fake_services import fake_embed
        return np.array([fake_embed(t, "トピック: ") for t in texts], dtype=np.float64)

    def test_one_topic_is_all_noise(self) -> None:
        labels, _ = cluster_worker.cluster(self.vectors(RUN + [LONER]))
        self.assertEqual(set(labels), {-1})

    def test_two_topics_separate(self) -> None:
        labels, _ = cluster_worker.cluster(self.vectors(RUN + TAX + [LONER]))
        self.assertEqual(len({l for l in labels if l >= 0}), 2)


class ClusterMatching(unittest.TestCase):
    """ID 継承のマッチングだけを取り出して見る。"""

    def test_greedy_match_picks_the_best_pair_first(self) -> None:
        prev = {10: {1, 2, 3, 4}, 20: {5, 6, 7}}
        new = {0: {1, 2, 3, 4, 5}, 1: {5, 6, 7}}
        self.assertEqual(cluster_worker.match_clusters(prev, new), {0: 10, 1: 20})

    def test_below_threshold_is_not_inherited(self) -> None:
        prev = {10: {1, 2, 3, 4}}
        new = {0: {4, 5, 6, 7}}          # Jaccard = 1/7
        self.assertEqual(cluster_worker.match_clusters(prev, new), {})

    def test_one_cluster_is_not_claimed_twice(self) -> None:
        prev = {10: {1, 2, 3}}
        new = {0: {1, 2, 3}, 1: {1, 2, 4}}
        self.assertEqual(cluster_worker.match_clusters(prev, new), {0: 10})


if __name__ == "__main__":
    unittest.main(verbosity=2)
