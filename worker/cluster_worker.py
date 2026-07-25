"""
クラスタ再構成ワーカー (Raspberry Pi、夜間バッチ)

'トピック: ' プレフィックスのベクトルを HDBSCAN にかけ、
前回のクラスタと Jaccard 係数でマッチングして cluster_id を引き継ぐ。

引き継ぎをやらないと HDBSCAN が毎回ラベルを振り直すので、
人が付けたテーマ名が消え、テーマノートが毎晩全書き換えになる。

命名とテーマノート生成はここではやらない (theme_writer.py の担当)。

  uv pip install numpy scikit-learn sqlite-vec
  python cluster_worker.py
"""

import hashlib
import json
import logging
import os
import secrets
import time

import numpy as np
import sqlite_vec
from sklearn.cluster import HDBSCAN

from common import connect, write_theme_markdown

VEC_TABLE = "vec_ideas_ruri_v3_310m_topic"

MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "3"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "2"))
# 前回クラスタと同一とみなす Jaccard の下限
INHERIT_THRESHOLD = float(os.environ.get("INHERIT_THRESHOLD", "0.4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cluster-worker")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def member_hash(idea_ids: set[int]) -> str:
    joined = ",".join(str(i) for i in sorted(idea_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


# ------------------------------------------------------------
# 読み込み
# ------------------------------------------------------------
def load_vectors(db) -> tuple[list[int], np.ndarray]:
    rows = db.execute(
        f"""
        SELECT v.idea_id, vec_to_json(v.embedding) AS vec
        FROM   {VEC_TABLE} v
               JOIN ideas i ON i.id = v.idea_id
        WHERE  i.status != 'archived'
        ORDER BY v.idea_id
        """
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0))

    ids = [row["idea_id"] for row in rows]
    mat = np.array([json.loads(row["vec"]) for row in rows], dtype=np.float64)
    return ids, mat


def cluster(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """正規化済みベクトルなので cosine 距離行列を直接作って渡す。

    768 次元のまま HDBSCAN に入れると次元の呪いで効きにくいが、
    数千件規模なら距離行列を precomputed で渡す方が UMAP を挟むより
    再現性が高く、Pi に numba を入れずに済む。
    件数が 5000 を超えたら UMAP での次元削減を検討する。
    """
    distances = np.clip(1.0 - mat @ mat.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="precomputed",
        cluster_selection_method="eom",
        copy=True,
    )
    # copy=True: 距離行列を破壊させない (1.10 でデフォルトが変わる)
    labels = model.fit_predict(distances)
    return labels, model.probabilities_


# ------------------------------------------------------------
# クラスタ ID の引き継ぎ
# ------------------------------------------------------------
def previous_members(db) -> tuple[dict[int, set[int]], set[int]]:
    """(cluster_id -> 前回のメンバー集合, そのうち開いているもの)

    閉じたクラスタも候補に入れる。話題が数か月止まってから戻ってきたとき、
    新しい id を振ると名前も調査履歴も切れてしまうため。
    idea_clusters は 1 メモ 1 行 (idea_id が PRIMARY KEY) で閉じたクラスタの
    メンバーを残しておけないので、集合そのものを clusters に持たせている。
    """
    prev: dict[int, set[int]] = {}
    live: set[int] = set()
    for row in db.execute(
        "SELECT id, members_json, closed_at FROM clusters WHERE members_json IS NOT NULL"
    ):
        prev[row["id"]] = set(json.loads(row["members_json"]))
        if row["closed_at"] is None:
            live.add(row["id"])
    return prev, live


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def match_clusters(
    prev: dict[int, set[int]], new: dict[int, set[int]]
) -> dict[int, int]:
    """新ラベル -> 既存 cluster_id。Jaccard 降順の貪欲マッチング。"""
    pairs = [
        (jaccard(new_members, prev_members), label, cluster_id)
        for label, new_members in new.items()
        for cluster_id, prev_members in prev.items()
    ]
    pairs.sort(reverse=True)

    mapping: dict[int, int] = {}
    used_labels: set[int] = set()
    used_clusters: set[int] = set()

    for score, label, cluster_id in pairs:
        if score < INHERIT_THRESHOLD:
            break
        if label in used_labels or cluster_id in used_clusters:
            continue
        mapping[label] = cluster_id
        used_labels.add(label)
        used_clusters.add(cluster_id)

    return mapping


# ------------------------------------------------------------
# 書き込み
# ------------------------------------------------------------
def persist(db, new: dict[int, set[int]], probs: dict[int, float]) -> tuple[int, int]:
    prev, was_open = previous_members(db)
    mapping = match_clusters(prev, new)

    n_new = 0
    now = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"

    with db:
        db.execute("DELETE FROM idea_clusters")

        live: set[int] = set()
        for label, members in new.items():
            ids_json = json.dumps(sorted(members))
            cluster_id = mapping.get(label)
            if cluster_id is None:
                cur = db.execute(
                    "INSERT INTO clusters(uid, size, member_hash, members_json) "
                    "VALUES (?, ?, ?, ?)",
                    (ulid(), len(members), member_hash(members), ids_json),
                )
                cluster_id = cur.lastrowid
                n_new += 1
            else:
                db.execute(
                    f"UPDATE clusters SET size = ?, member_hash = ?, members_json = ?, "
                    f"updated_at = {now}, closed_at = NULL WHERE id = ?",
                    (len(members), member_hash(members), ids_json, cluster_id),
                )
            live.add(cluster_id)

            db.executemany(
                "INSERT INTO idea_clusters(idea_id, cluster_id, probability) "
                "VALUES (?, ?, ?)",
                [(i, cluster_id, probs.get(i)) for i in sorted(members)],
            )

        # 消えたクラスタは削除せず閉じる。名前と履歴とメンバー集合は残す
        stale = [cid for cid in was_open if cid not in live]
        if stale:
            db.executemany(
                f"UPDATE clusters SET closed_at = {now}, size = 0 WHERE id = ?",
                [(cid,) for cid in stale],
            )

    # 開閉が変わったテーマノートは書き直す。閉じたものが Obsidian で
    # 現役のまま見えていると、消えた話題を追いかけることになる
    for cluster_id in stale + [c for c in live if c in prev and c not in was_open]:
        write_theme_markdown(db, cluster_id)

    return n_new, len(stale)


def run(db) -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ids, mat = load_vectors(db)

    if len(ids) < MIN_CLUSTER_SIZE:
        log.info("not enough vectors yet (%d)", len(ids))
        return

    labels, probabilities = cluster(mat)

    new: dict[int, set[int]] = {}
    probs: dict[int, float] = {}
    for idea_id, label, prob in zip(ids, labels, probabilities):
        probs[idea_id] = float(prob)
        if label >= 0:                      # -1 はノイズ。どのテーマにも属さない
            new.setdefault(int(label), set()).add(idea_id)

    n_new, n_closed = persist(db, new, probs)
    n_noise = int((labels < 0).sum())

    with db:
        db.execute(
            """
            INSERT INTO cluster_runs(started_at, finished_at, n_ideas, n_clusters,
                                     n_noise, n_new, n_closed, params_json)
            VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?, ?, ?, ?, ?, ?)
            """,
            (
                started,
                len(ids),
                len(new),
                n_noise,
                n_new,
                n_closed,
                json.dumps(
                    {
                        "min_cluster_size": MIN_CLUSTER_SIZE,
                        "min_samples": MIN_SAMPLES,
                        "inherit_threshold": INHERIT_THRESHOLD,
                    }
                ),
            ),
        )

    log.info(
        "ideas=%d clusters=%d noise=%d new=%d closed=%d",
        len(ids), len(new), n_noise, n_new, n_closed,
    )


def main() -> None:
    db = connect(with_vec=True)
    run(db)


if __name__ == "__main__":
    main()
