"""
埋め込み reconciliation ワーカー (Raspberry Pi 常駐)

キューを持たず、「embeddings に無い / content_hash がズレている」行を
毎周回で拾い直す。ワーカーが落ちても次の周回で勝手に追いつくので、
リトライ処理もデッドレターキューも要らない。

ruri-v3 のプレフィックスごとに別ベクトルを作る。
  ''          -> 類似メモ検出 (Phase 1)
  'トピック: ' -> クラスタリング (Phase 2)

  uv pip install sqlite-vec
  python embed_worker.py
"""

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request

import sqlite_vec

from common import connect

EMBED_URL = os.environ.get("EMBED_URL", "http://mac-mini.local:7997/embed")

MODEL = "cl-nagoya/ruri-v3-310m"
DIM = 768

# (プレフィックス, サーバー側の prefix 名, ベクトルテーブル)
TARGETS = [
    ("", "semantic", "vec_ideas_ruri_v3_310m"),
    ("トピック: ", "topic", "vec_ideas_ruri_v3_310m_topic"),
]

BATCH = 16
INTERVAL_SEC = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-worker")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_pending(db, prefix: str, limit: int) -> list[tuple[int, str]]:
    """未計算 + 本文が変わったものを拾う。

    SQLite に sha256 が無いのでハッシュ比較は Python 側でやる。
    個人用途の行数なら全走査で問題にならない。
    """
    rows = db.execute(
        """
        SELECT i.id, i.body, e.content_hash
        FROM   ideas i
               LEFT JOIN embeddings e
                      ON e.idea_id = i.id
                     AND e.model   = ?
                     AND e.target  = 'body'
                     AND e.prefix  = ?
        WHERE  i.status != 'archived'
               AND i.body != ''          -- 文字起こし待ちの音声メモは飛ばす
        ORDER BY i.captured_at DESC
        """,
        (MODEL, prefix),
    ).fetchall()

    pending = [
        (row["id"], row["body"])
        for row in rows
        if row["content_hash"] is None or row["content_hash"] != sha256(row["body"])
    ]
    return pending[:limit]


def call_embed(texts: list[str], prefix_name: str) -> list[list[float]]:
    payload = json.dumps({"input": texts, "prefix": prefix_name}).encode("utf-8")
    req = urllib.request.Request(
        EMBED_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        data = json.loads(res.read())

    # 次元が変わっていたら黙って壊れるより落とす。モデル差し替え事故の検知
    if data["dim"] != DIM:
        raise RuntimeError(f"dimension mismatch: expected {DIM}, got {data['dim']}")
    return data["embeddings"]


def store(db, table: str, prefix: str, items, vectors) -> None:
    with db:
        for (idea_id, body), vec in zip(items, vectors):
            db.execute(f"DELETE FROM {table} WHERE idea_id = ?", (idea_id,))
            db.execute(
                f"INSERT INTO {table}(idea_id, embedding) VALUES (?, ?)",
                (idea_id, sqlite_vec.serialize_float32(vec)),
            )
            db.execute(
                """
                INSERT INTO embeddings(idea_id, model, dim, target, prefix, content_hash)
                VALUES (?, ?, ?, 'body', ?, ?)
                ON CONFLICT(idea_id, model, target, prefix)
                DO UPDATE SET content_hash = excluded.content_hash,
                              created_at   = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                """,
                (idea_id, MODEL, DIM, prefix, sha256(body)),
            )


def tick(db) -> int:
    total = 0
    for prefix, prefix_name, table in TARGETS:
        items = fetch_pending(db, prefix, BATCH)
        if not items:
            continue
        vectors = call_embed([body for _, body in items], prefix_name)
        store(db, table, prefix, items, vectors)
        log.info("embedded %d ideas (prefix=%r)", len(items), prefix)
        total += len(items)
    return total


def main() -> None:
    db = connect(with_vec=True)
    while True:
        try:
            while tick(db) > 0:
                pass
        except urllib.error.URLError as exc:
            # Mac mini が寝ていても投稿は成功させたい。次の周回で追いつく
            log.warning("embedding service unreachable: %s", exc)
        except Exception:
            log.exception("tick failed")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
