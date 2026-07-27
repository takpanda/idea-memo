"""
閲覧・検索 UI (Raspberry Pi、Docker)

母艦は Obsidian のままだが、外出先の iPhone からは vault を開けない。
ブラウザさえあれば読める側を用意する:

  - メモ一覧   status / テーマで絞り込み、captured_at のキーセットページング
  - 全文検索   ベクトル + FTS5 を RRF で統合したハイブリッド検索
  - テーマ     一覧とテーマの中身 (メンバー + 参考情報)
  - メモ詳細   本文全文・関連メモ・返信の連なり
  - status の手動変更 (自動 archive はしない方針なので、ここが唯一の導線)

ビルドステップを持ちたくないので、HTML は 1 枚をインラインで返す。
画面遷移はハッシュルーティングにして、iPhone の戻るジェスチャで戻れるようにする。

  uvicorn web:app --host 0.0.0.0 --port 8080
"""

import json
import hashlib
import os
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from common import connect

EMBED_URL = os.environ.get("EMBED_URL", "http://mac-mini.local:7997/embed")
VEC_TABLE = "vec_ideas_ruri_v3_310m"
RRF_K = 60          # Reciprocal Rank Fusion の定数。慣例値
POOL = 30
PAGE = 30
MAX_PAGE = 100
MEMBER_PAGE = 20      # テーマ詳細の初期表示。続きは /api/ideas のページングに任せる
SNIPPET_LEN = 140
STATUSES = ("inbox", "kept", "archived")
VERDICTS = ("useful", "not_useful")

# テーマ一覧の並び。既定はサイズ順 (大きい話題から見る)
CLUSTER_ORDER = {
    "size": "c.size DESC, c.updated_at DESC",
    "recent": "c.updated_at DESC, c.size DESC",
    "name": "c.name COLLATE NOCASE, c.size DESC",
}

# ネットワーク図の色は DB に保存する表示状態ではないため、クラスタ UID から
# 決定的に選ぶ。クラスタの再構成で UID が引き継がれる限り色も変わらない。
NETWORK_COLORS = (
    "#3d5afe", "#00897b", "#f4511e", "#8e24aa", "#6d4c41",
    "#039be5", "#7cb342", "#d81b60",
)

app = FastAPI(title="idea-memo")


def snippet(body: str, query: str = "") -> str:
    """一覧・検索結果に載せる抜粋。

    長いメモでヒットが末尾にあると、頭から切っただけでは何に当たったのか
    分からない。キーワードがあればその周りに窓を寄せる。
    """
    flat = " ".join((body or "").split())
    if len(flat) <= SNIPPET_LEN:
        return flat

    start = 0
    if query:
        hit = flat.lower().find(query.lower())
        if hit > 0:
            start = max(0, hit - SNIPPET_LEN // 3)
    end = start + SNIPPET_LEN
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def embed_query(text: str) -> bytes | None:
    """検索クエリを埋め込む。

    ruri-v3 には 検索クエリ: / 検索文書: の非対称ペアがあるが、
    保存側を '' (semantic) で埋めているのでクエリも '' で揃える。
    非対称ペアに変えるなら、文書側も 検索文書: で埋め直す必要がある。
    """
    import sqlite_vec

    payload = json.dumps({"input": [text], "prefix": "semantic"}).encode("utf-8")
    req = urllib.request.Request(
        EMBED_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read())
        return sqlite_vec.serialize_float32(data["embeddings"][0])
    except Exception:
        return None      # 埋め込みが落ちていても全文検索だけで返す


def fts_match(query: str) -> str:
    """trigram tokenizer 向けにフレーズとしてクォートする。

    素の文字列を MATCH に渡すと FTS5 の構文文字で落ちる。
    """
    return '"' + query.replace('"', '""') + '"'


def keyword_search(db, q: str) -> list[int]:
    """trigram tokenizer は 3 文字未満を索引できない。

    「メモ」「整理」「料理」のような 2 文字語は日本語では日常的なので、
    そこだけ LIKE の全走査に落とす。個人規模の行数ならミリ秒で終わる。
    """
    if len(q) >= 3:
        return [
            row["rowid"]
            for row in db.execute(
                "SELECT rowid FROM fts_ideas WHERE fts_ideas MATCH ? "
                "ORDER BY bm25(fts_ideas) LIMIT ?",
                (fts_match(q), POOL),
            )
        ]

    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return [
        row["id"]
        for row in db.execute(
            "SELECT id FROM ideas WHERE body LIKE ? ESCAPE '\\' "
            "AND status != 'archived' ORDER BY captured_at DESC LIMIT ?",
            (f"%{escaped}%", POOL),
        )
    ]


def rrf(*rankings: list[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idea_id in enumerate(ranking, start=1):
            scores[idea_id] = scores.get(idea_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


@app.get("/api/search")
def search(q: str, limit: int = 20) -> dict:
    empty = {"query": q, "vector": False, "keyword": False, "results": []}
    if not q.strip():
        return empty

    db = connect(with_vec=True)

    vector_rank: list[int] = []
    vec = embed_query(q)
    if vec is not None:
        vector_rank = [
            row["idea_id"]
            for row in db.execute(
                f"SELECT idea_id FROM {VEC_TABLE} WHERE embedding MATCH ? AND k = ?",
                (vec, POOL),
            )
        ]

    keyword_rank = keyword_search(db, q.strip())

    scores = rrf(vector_rank, keyword_rank)
    if not scores:
        return {**empty, "vector": bool(vector_rank), "keyword": bool(keyword_rank)}

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    ids = [i for i, _ in top]
    placeholders = ",".join("?" * len(ids))

    rows = {
        row["id"]: dict(row)
        for row in db.execute(
            f"""
            SELECT i.id, i.uid, i.body, i.captured_at, i.status, i.input_kind,
                   c.name AS cluster_name, c.uid AS cluster_uid
            FROM   ideas i
                   LEFT JOIN idea_clusters ic ON ic.idea_id = i.id
                   LEFT JOIN clusters c ON c.id = ic.cluster_id
            WHERE  i.id IN ({placeholders})
            """,
            ids,
        )
    }

    results = []
    for idea_id, score in top:
        row = rows.get(idea_id)
        if row:
            row["snippet"] = snippet(row.pop("body"), q.strip())
            row.pop("id")
            results.append({**row, "score": round(score, 5)})

    return {
        "query": q,
        "vector": bool(vector_rank),
        "keyword": bool(keyword_rank),
        "results": results,
    }


@app.get("/api/ideas")
def list_ideas(
    status: str = "active",
    cluster: str | None = None,
    before: str | None = None,
    limit: int = PAGE,
) -> dict:
    """新しい順のメモ一覧。

    OFFSET ではなくキーセットページングにする。取り込みは常に先頭に入るので、
    読んでいる最中に新着が来ると OFFSET では境界のメモが飛ぶ / 重複する。
    """
    limit = max(1, min(limit, MAX_PAGE))
    db = connect()

    filters: list[str] = []
    params: list = []

    if status == "active":
        filters.append("i.status != 'archived'")
    elif status in STATUSES:
        filters.append("i.status = ?")
        params.append(status)
    elif status != "all":
        raise HTTPException(400, "invalid status")

    scope = None
    if cluster == "none":
        filters.append("ic.idea_id IS NULL")
        scope = {"uid": "none", "name": None}
    elif cluster:
        row = db.execute(
            "SELECT id, uid, name FROM clusters WHERE uid = ?", (cluster,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "cluster not found")
        filters.append("ic.cluster_id = ?")
        params.append(row["id"])
        scope = {"uid": row["uid"], "name": row["name"]}

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    # 総数は先頭ページでだけ数える。以降のページで数え直しても同じ値
    total = None
    if not before:
        total = db.execute(
            f"""
            SELECT COUNT(*) AS n FROM ideas i
                   LEFT JOIN idea_clusters ic ON ic.idea_id = i.id
            {where}
            """,
            params,
        ).fetchone()["n"]

    page_params = list(params)
    page_where = where
    if before:
        # captured_at は同じ秒が並びうるので id まで見て割る
        at, sep, ident = before.partition("|")
        if not sep or not ident.isdigit():
            raise HTTPException(400, "invalid cursor")
        page_where += (" AND " if filters else "WHERE ") + "(i.captured_at, i.id) < (?, ?)"
        page_params += [at, int(ident)]

    rows = db.execute(
        f"""
        SELECT i.id, i.uid, i.body, i.captured_at, i.status, i.input_kind,
               c.uid AS cluster_uid, c.name AS cluster_name, ic.probability,
               EXISTS (SELECT 1 FROM attachments a WHERE a.idea_id = i.id) AS attached
        FROM   ideas i
               LEFT JOIN idea_clusters ic ON ic.idea_id = i.id
               LEFT JOIN clusters c ON c.id = ic.cluster_id
        {page_where}
        ORDER BY i.captured_at DESC, i.id DESC
        LIMIT ?
        """,
        page_params + [limit],
    ).fetchall()

    ideas = []
    for row in rows:
        item = dict(row)
        item["snippet"] = snippet(item.pop("body"))
        item.pop("id")
        ideas.append(item)

    return {
        "ideas": ideas,
        "total": total,
        "scope": scope,
        # 端数ページなら次は無い。ちょうど埋まったときだけカーソルを返す
        "next": f"{rows[-1]['captured_at']}|{rows[-1]['id']}" if len(rows) == limit else None,
    }


@app.get("/api/ideas/{uid}")
def idea_detail(uid: str) -> dict:
    """メモ 1 件。Obsidian で開くのと同じものをブラウザで読むための画面。"""
    db = connect()
    idea = db.execute(
        """
        SELECT i.id, i.uid, i.body, i.summary, i.captured_at, i.updated_at,
               i.status, i.source, i.input_kind, i.file_path, i.parent_id,
               c.uid AS cluster_uid, c.name AS cluster_name
        FROM   ideas i
               LEFT JOIN idea_clusters ic ON ic.idea_id = i.id
               LEFT JOIN clusters c ON c.id = ic.cluster_id
        WHERE  i.uid = ?
        """,
        (uid,),
    ).fetchone()
    if idea is None:
        raise HTTPException(404, "idea not found")

    idea_id = idea["id"]

    attachments = [
        {
            "kind": row["kind"],
            "bytes": row["bytes"],
            "duration": json.loads(row["meta_json"] or "{}").get("duration"),
        }
        for row in db.execute(
            "SELECT kind, bytes, meta_json FROM attachments WHERE idea_id = ? ORDER BY id",
            (idea_id,),
        )
    ]

    # 提案は保存していないので、ここに出るのは人が承認した関連だけ
    related = db.execute(
        """
        SELECT o.uid, o.body, o.captured_at, r.kind, r.score
        FROM   relations r
               JOIN ideas o
                 ON o.id = CASE WHEN r.src_id = ? THEN r.dst_id ELSE r.src_id END
        WHERE  (r.src_id = ? OR r.dst_id = ?) AND r.verdict = 'confirmed'
        ORDER BY o.captured_at DESC
        """,
        (idea_id, idea_id, idea_id),
    ).fetchall()

    parent = None
    if idea["parent_id"]:
        parent = db.execute(
            "SELECT uid, body, captured_at FROM ideas WHERE id = ?", (idea["parent_id"],)
        ).fetchone()

    replies = db.execute(
        "SELECT uid, body, captured_at FROM ideas WHERE parent_id = ? "
        "ORDER BY captured_at",
        (idea_id,),
    ).fetchall()

    def brief(row) -> dict:
        item = dict(row)
        item["snippet"] = snippet(item.pop("body"))
        return item

    detail = {k: idea[k] for k in idea.keys() if k not in ("id", "parent_id")}
    return {
        "idea": detail,
        "attachments": attachments,
        "related": [brief(r) for r in related],
        "parent": brief(parent) if parent else None,
        "replies": [brief(r) for r in replies],
    }


@app.get("/api/clusters")
def clusters(sort: str = "size") -> dict:
    """テーマ一覧。

    参考情報は件数だけでなく反証 (challenges) の数も返す。一覧で見たいのは
    「賛成材料が何件あるか」ではなく「反証が来ているテーマはどれか」なので。
    外したものは両方の数から除く。
    """
    if sort not in CLUSTER_ORDER:
        raise HTTPException(400, "invalid sort")

    db = connect()
    rows = db.execute(
        f"""
        SELECT c.uid, c.name, c.summary, c.size, c.updated_at,
               (SELECT COUNT(*) FROM findings f
                WHERE  f.cluster_id = c.id
                       AND (f.verdict IS NULL OR f.verdict = 'useful')) AS findings,
               (SELECT COUNT(*) FROM findings f
                WHERE  f.cluster_id = c.id AND f.stance = 'challenges'
                       AND (f.verdict IS NULL OR f.verdict = 'useful')) AS challenges
        FROM   clusters c
        WHERE  c.closed_at IS NULL AND c.size > 0
        ORDER BY {CLUSTER_ORDER[sort]}
        """
    ).fetchall()

    noise = db.execute(
        """
        SELECT COUNT(*) AS n FROM ideas i
        WHERE  i.status != 'archived'
               AND NOT EXISTS (SELECT 1 FROM idea_clusters ic WHERE ic.idea_id = i.id)
        """
    ).fetchone()["n"]

    return {"clusters": [dict(r) for r in rows], "unclustered": noise, "sort": sort}


def _cluster_centroids(db, cluster_ids: list[int]) -> dict[int, list[float]]:
    """クラスタごとの平均ベクトルを単位長に正規化して返す。

    ベクトルはクラスタリングで使う topic 版ではなく、保存側の semantic 版を
    使う。検索 API と同じく、保存済みベクトルがないメモは計算対象から外す。
    平均ベクトルがゼロになる場合も、無効な重心として扱う。
    """
    if not cluster_ids:
        return {}

    placeholders = ",".join("?" * len(cluster_ids))
    rows = db.execute(
        f"""
        SELECT ic.cluster_id, vec_to_json(v.embedding) AS vec
        FROM   idea_clusters ic
               JOIN {VEC_TABLE} v ON v.idea_id = ic.idea_id
        WHERE  ic.cluster_id IN ({placeholders})
        """,
        cluster_ids,
    ).fetchall()

    sums: dict[int, list[float]] = {}
    for row in rows:
        vector = json.loads(row["vec"])
        centroid = sums.setdefault(row["cluster_id"], [0.0] * len(vector))
        for index, value in enumerate(vector):
            centroid[index] += value

    centroids: dict[int, list[float]] = {}
    for cluster_id, vector in sums.items():
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            centroids[cluster_id] = [value / norm for value in vector]
    return centroids


def _network_strength(similarity: float) -> str | None:
    if similarity > 0.7:
        return "strong"
    if similarity > 0.5:
        return "medium"
    if similarity > 0.3:
        return "weak"
    return None


@app.get("/api/clusters/network")
def cluster_network() -> dict:
    """開いているテーマ間の類似度をネットワーク用の形で返す。

    ノードはベクトルの有無によらず返す。片方でも重心を作れないペアは
    エッジを作らず、データ欠損を API 500 にしない。
    """
    db = connect(with_vec=True)
    rows = db.execute(
        """
        SELECT id, uid, name, size
        FROM   clusters
        WHERE  closed_at IS NULL AND size > 0
        ORDER BY size DESC, updated_at DESC
        """
    ).fetchall()

    nodes = []
    for row in rows:
        digest = hashlib.sha256(row["uid"].encode("utf-8")).digest()
        nodes.append({
            "id": row["uid"],
            "name": row["name"],
            "size": row["size"],
            "color": NETWORK_COLORS[digest[0] % len(NETWORK_COLORS)],
        })

    centroids = _cluster_centroids(db, [row["id"] for row in rows])
    edges = []
    for left_index, left in enumerate(rows):
        left_vector = centroids.get(left["id"])
        if left_vector is None:
            continue
        for right in rows[left_index + 1:]:
            right_vector = centroids.get(right["id"])
            if right_vector is None:
                continue
            similarity = sum(a * b for a, b in zip(left_vector, right_vector))
            strength = _network_strength(similarity)
            if strength:
                edges.append({
                    "source": left["uid"],
                    "target": right["uid"],
                    "strength": strength,
                })

    # ノイズに対する「近傍」の既存仕様・保存データは存在しないため、
    # 暫定的に空配列を返す。定義が決まれば別途ベクトル検索を追加する。
    return {"nodes": nodes, "edges": edges, "noiseNear": []}


@app.get("/api/clusters/{uid}")
def cluster_detail(uid: str) -> dict:
    db = connect()
    cluster = db.execute(
        """
        SELECT id, uid, name, summary, size, updated_at, closed_at
        FROM   clusters WHERE uid = ?
        """,
        (uid,),
    ).fetchone()
    if cluster is None:
        raise HTTPException(404, "cluster not found")

    # 大きいテーマで全メンバーを一度に描かない。続きは /api/ideas?cluster=... の
    # キーセットページングに載せるので、並び順をそちらと揃えておく
    members = db.execute(
        """
        SELECT i.id, i.uid, i.body, i.captured_at, i.status, i.input_kind,
               ic.probability
        FROM   idea_clusters ic JOIN ideas i ON i.id = ic.idea_id
        WHERE  ic.cluster_id = ?
        ORDER BY i.captured_at DESC, i.id DESC
        LIMIT ?
        """,
        (cluster["id"], MEMBER_PAGE + 1),
    ).fetchall()

    more = len(members) > MEMBER_PAGE
    members = members[:MEMBER_PAGE]

    findings = db.execute(
        """
        SELECT id, title, url, site, summary, stance, query_kind, verdict
        FROM   findings WHERE cluster_id = ?
        -- 外したものは消さずに末尾へ送る。反証を先頭に置くのはその次
        ORDER BY verdict = 'not_useful', stance = 'challenges' DESC, fetched_at DESC
        """,
        (cluster["id"],),
    ).fetchall()

    def member(row) -> dict:
        item = dict(row)
        item["snippet"] = snippet(item.pop("body"))
        item.pop("id")
        return item

    return {
        "cluster": {k: cluster[k] for k in cluster.keys() if k != "id"},
        "members": [member(r) for r in members],
        "members_next": (
            f"{members[-1]['captured_at']}|{members[-1]['id']}" if more else None
        ),
        "findings": [dict(r) for r in findings],
    }


class NameUpdate(BaseModel):
    name: str


@app.post("/api/clusters/{uid}/name")
def rename_cluster(uid: str, payload: NameUpdate) -> dict:
    """テーマ名を人が付け直す。

    付けた時点で name_locked を立てる。ここで立てないと次の theme_writer が
    LLM の名前で上書きしてしまい、押した意味がなくなる。
    """
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "empty name")

    db = connect()
    row = db.execute("SELECT id FROM clusters WHERE uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(404, "cluster not found")

    with db:
        db.execute(
            "UPDATE clusters SET name = ?, name_locked = 1, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
            (name, row["id"]),
        )

    from common import write_theme_markdown

    write_theme_markdown(db, row["id"])
    return {"uid": uid, "name": name, "name_locked": 1}


class VerdictUpdate(BaseModel):
    verdict: str | None


@app.post("/api/findings/{finding_id}/verdict")
def set_verdict(finding_id: int, payload: VerdictUpdate) -> dict:
    """参考情報の当たり外れを記録する。

    行は消さない。テーマノートからは落ちる (write_theme_markdown が not_useful を
    除く) が、俯瞰では薄く残して「一度見て外した」ことが分かるようにする。
    """
    if payload.verdict is not None and payload.verdict not in VERDICTS:
        raise HTTPException(400, "invalid verdict")

    db = connect()
    row = db.execute(
        "SELECT cluster_id FROM findings WHERE id = ?", (finding_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "finding not found")

    with db:
        db.execute(
            "UPDATE findings SET verdict = ? WHERE id = ?", (payload.verdict, finding_id)
        )

    from common import write_theme_markdown

    write_theme_markdown(db, row["cluster_id"])
    return {"id": finding_id, "verdict": payload.verdict}


class StatusUpdate(BaseModel):
    status: str


@app.post("/api/ideas/{uid}/status")
def set_status(uid: str, payload: StatusUpdate) -> dict:
    if payload.status not in STATUSES:
        raise HTTPException(400, "invalid status")

    db = connect()
    with db:
        cur = db.execute(
            "UPDATE ideas SET status = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE uid = ?",
            (payload.status, uid),
        )
    if cur.rowcount == 0:
        raise HTTPException(404, "idea not found")

    from common import write_idea_markdown

    row = db.execute("SELECT id FROM ideas WHERE uid = ?", (uid,)).fetchone()
    write_idea_markdown(db, row["id"])
    return {"uid": uid, "status": payload.status}


INDEX = """<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#161616" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="apple-mobile-web-app-title" content="idea">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='14'>&#128161;</text></svg>">
<title>idea-memo</title>
<style>
  :root { color-scheme: light dark; --fg:#1a1a1a; --dim:#767676; --line:#e0e0e0; --bg:#fff;
          --accent:#3d5afe; --mark:#ffe9a8; --soft:#f6f6f6; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --dim:#9a9a9a; --line:#333; --bg:#161616;
            --accent:#8c9eff; --mark:#5b4a12; --soft:#1e1e1e; }
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; max-width:52rem; margin-inline:auto;
         padding:0 max(1rem, env(safe-area-inset-left)) calc(3rem + env(safe-area-inset-bottom));
         font:16px/1.7 system-ui,-apple-system,"Hiragino Sans",sans-serif;
         color:var(--fg); background:var(--bg); -webkit-text-size-adjust:100%; }
  header { position:sticky; top:0; z-index:2; background:var(--bg);
           padding:calc(.8rem + env(safe-area-inset-top)) 0 0; }
  h1 { font-size:.8rem; letter-spacing:.1em; color:var(--dim); font-weight:600; margin:0; }
  h1 a { color:inherit; text-decoration:none; }
  nav { display:flex; gap:1.2rem; border-bottom:1px solid var(--line); margin-top:.6rem; }
  nav a { padding:.6rem 0; color:var(--dim); text-decoration:none;
          border-bottom:2px solid transparent; margin-bottom:-1px; }
  nav a[aria-current="page"] { color:var(--fg); border-color:var(--accent); }
  input[type=search] { width:100%; padding:.8rem .9rem; font:inherit; color:var(--fg);
                       background:transparent; border:1px solid var(--line); border-radius:10px;
                       margin-top:1rem; }
  .chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:1rem 0 .3rem; }
  .chips a { font-size:.8rem; text-decoration:none; color:var(--dim); padding:.3rem .8rem;
             border:1px solid var(--line); border-radius:99px; }
  .chips a[aria-current="page"] { color:var(--bg); background:var(--fg); border-color:var(--fg); }
  .card { display:block; border-bottom:1px solid var(--line); padding:.9rem .2rem;
          color:inherit; text-decoration:none; }
  a.card:active { background:var(--soft); }
  .meta { font-size:.8rem; color:var(--dim); display:flex; gap:.7rem; flex-wrap:wrap;
          align-items:center; }
  .body { margin:.35rem 0 0; white-space:pre-wrap; overflow-wrap:anywhere; }
  .full { font-size:1.05rem; margin:1rem 0 1.5rem; }
  .tag { font-size:.75rem; border:1px solid var(--line); border-radius:99px; padding:.05rem .5rem; }
  .name { color:var(--fg); font-weight:600; }
  h2 { font-size:.8rem; letter-spacing:.06em; color:var(--dim); font-weight:600;
       margin:2rem 0 .2rem; }
  mark { background:var(--mark); color:inherit; border-radius:3px; }
  .stance-challenges { color:#e0642f; }
  .stance-supports { color:#2e9e5b; }
  /* 不要と判断した参考情報。テーマノートからは消えるが、俯瞰では残す */
  .dropped { opacity:.45; }
  a { color:var(--accent); }
  .empty { color:var(--dim); padding:2rem 0; }
  .back { display:inline-block; font-size:.85rem; color:var(--dim); text-decoration:none;
          margin:1rem 0 .2rem; }
  button { font:inherit; cursor:pointer; }
  .more { width:100%; margin-top:1.2rem; padding:.8rem; background:none; color:var(--dim);
          border:1px solid var(--line); border-radius:10px; }
  .actions { display:flex; gap:.5rem; margin:1.5rem 0 0; }
  .actions button { flex:1; padding:.6rem; font-size:.85rem; background:none;
                    border:1px solid var(--line); border-radius:8px; color:var(--dim); }
  .actions button[aria-pressed="true"] { color:var(--bg); background:var(--fg); border-color:var(--fg); }
  /* 一覧をスキャンできるよう、要約は 2 行で頭打ちにする */
  .clamp { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
           overflow:hidden; }
  .warn { color:#e0642f; }
  .title { display:flex; align-items:baseline; gap:.6rem; margin:.4rem 0; }
  .title h1 { font-size:1.2rem; color:var(--fg); letter-spacing:0; flex:1; }
  .title input { flex:1; min-width:0; font:inherit; font-size:1.1rem; color:var(--fg);
                 background:transparent; border:1px solid var(--line); border-radius:8px;
                 padding:.4rem .6rem; }
  .mini { font-size:.75rem; padding:.25rem .7rem; background:none; color:var(--dim);
          border:1px solid var(--line); border-radius:99px; white-space:nowrap; }
  .mini[aria-pressed="true"] { color:var(--bg); background:var(--fg); border-color:var(--fg); }
</style></head>
<body>
<header>
  <h1><a href="#/">IDEA MEMO</a></h1>
  <nav>
    <a href="#/" data-tab="ideas">メモ</a>
    <a href="#/search" data-tab="search">検索</a>
    <a href="#/themes" data-tab="themes">テーマ</a>
  </nav>
</header>
<main id="view"></main>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* 検索語だけを強調する。エスケープ後の文字列に手を入れると実体参照を
   跨いで壊れるので、素のテキストを切りながらエスケープする */
function mark(text, q) {
  const src = String(text ?? "");
  if (!q) return esc(src);
  const hay = src.toLowerCase(), needle = q.toLowerCase();
  let out = "", from = 0, hit;
  while ((hit = hay.indexOf(needle, from)) !== -1) {
    out += esc(src.slice(from, hit)) +
           "<mark>" + esc(src.slice(hit, hit + needle.length)) + "</mark>";
    from = hit + needle.length;
  }
  return out + esc(src.slice(from));
}

/* captured_at は UTC。読むのは手元の端末なので現地時間に直して出す */
const fmtDay = iso => {
  const d = new Date(iso);
  return d.toLocaleDateString("ja-JP", d.getFullYear() === new Date().getFullYear()
    ? {month:"numeric", day:"numeric", weekday:"short"}
    : {year:"numeric", month:"numeric", day:"numeric"});
};
const fmtStamp = iso => new Date(iso).toLocaleString("ja-JP",
  {year:"numeric", month:"numeric", day:"numeric", hour:"2-digit", minute:"2-digit"});

/* テーマの「止まっている / 動いている」は日付そのものより経過で掴む */
const fmtAgo = iso => {
  const days = Math.floor((Date.now() - new Date(iso)) / 86400000);
  if (days <= 0) return "今日";
  if (days === 1) return "昨日";
  if (days < 7) return `${days} 日前`;
  if (days < 31) return `${Math.floor(days / 7)} 週間前`;
  return `${Math.floor(days / 30)} ヶ月前`;
};

const STATUS_LABEL = {inbox:"inbox", kept:"残す", archived:"archive"};
const STANCE_MARK = {supports:"◯", challenges:"▲", neutral:"・"};
const KIND_LABEL = {prior_art:"前例", evidence:"裏付け", counter:"反証"};
const SORT_LABEL = {size:"大きい順", recent:"最近動いた", name:"名前"};

/* 圏外や Pi の停止もここに来る。生の例外文言を出しても手当てのしようがないので、
   「何が起きたか」だけを日本語で返す */
async function api(path) {
  let res;
  try {
    res = await fetch(path);
  } catch (e) {
    throw new Error("つながりませんでした（電波か Pi の状態を確認）");
  }
  if (res.status === 404) throw new Error("見つかりませんでした");
  if (!res.ok) throw new Error(`読み込めませんでした (${res.status})`);
  return res.json();
}

function show(html) {
  $("#view").innerHTML = html;
}

// ------------------------------------------------------------
// 部品
// ------------------------------------------------------------
function ideaCard(r, q) {
  const kind = r.input_kind === "voice" ? '<span class="tag">音声</span>' : "";
  // HDBSCAN の所属確率。低いものは「たまたま寄っただけ」を疑いながら読む
  const edge = r.probability != null && r.probability < 0.5
    ? '<span class="tag">周辺</span>' : "";
  return `
    <a class="card" href="#/ideas/${encodeURIComponent(r.uid)}">
      <div class="meta">
        <span>${fmtDay(r.captured_at)}</span>
        ${r.cluster_name ? `<span class="tag">${esc(r.cluster_name)}</span>` : ""}
        ${kind}${edge}
        ${r.status && r.status !== "inbox" ? `<span>${STATUS_LABEL[r.status]}</span>` : ""}
      </div>
      <div class="body">${mark(r.snippet, q)}</div>
    </a>`;
}

const moreButton = (base, next) => next
  ? `<button class="more" data-base="${esc(base)}" data-next="${esc(next)}">もっと読む</button>`
  : "";

function ideaList(items, q) {
  if (!items.length) return '<p class="empty">まだ何もありません</p>';
  return items.map(r => ideaCard(r, q)).join("");
}

// ------------------------------------------------------------
// 画面: メモ一覧
// ------------------------------------------------------------
async function renderIdeas(params) {
  const status = params.get("status") || "active";
  const cluster = params.get("cluster") || "";

  const chip = (value, label) => {
    const qs = new URLSearchParams();
    if (value !== "active") qs.set("status", value);
    if (cluster) qs.set("cluster", cluster);
    const href = "#/" + (qs.toString() ? "?" + qs : "");
    return `<a href="${href}" ${status === value ? 'aria-current="page"' : ""}>${label}</a>`;
  };

  show(`
    <div class="chips">
      ${chip("active", "すべて")}${chip("inbox", "inbox")}
      ${chip("kept", "残す")}${chip("archived", "archive")}
    </div>
    <div id="items"><p class="empty">読み込み中…</p></div>`);

  const base = new URLSearchParams({status});
  if (cluster) base.set("cluster", cluster);
  const data = await api("/api/ideas?" + base);

  const scope = data.scope
    ? (data.scope.uid === "none" ? "未分類" : esc(data.scope.name || "(未命名のテーマ)"))
    : null;
  $("#items").innerHTML =
    `<p class="meta">${scope ? `<span class="tag">${scope}</span>` : ""}${data.total} 件</p>` +
    ideaList(data.ideas) + moreButton(base, data.next);
}

// ------------------------------------------------------------
// 画面: メモ詳細
// ------------------------------------------------------------
async function renderIdea(uid) {
  show('<p class="empty">読み込み中…</p>');
  const d = await api(`/api/ideas/${encodeURIComponent(uid)}`);
  const i = d.idea;

  const brief = (r, label) => `
    <a class="card" href="#/ideas/${encodeURIComponent(r.uid)}">
      <div class="meta"><span>${fmtDay(r.captured_at)}</span>
      ${label ? `<span class="tag">${label}</span>` : ""}</div>
      <div class="body">${esc(r.snippet)}</div>
    </a>`;

  const attachments = d.attachments.map(a =>
    `<span class="tag">${a.kind}${a.duration ? ` ${a.duration}s` : ""}</span>`).join(" ");

  show(`
    <a class="back" href="#/">← 一覧</a>
    <div class="meta">
      <span>${fmtStamp(i.captured_at)}</span>
      <span>${i.source} / ${i.input_kind}</span>
      ${attachments}
    </div>
    ${i.cluster_uid
      ? `<div class="meta" style="margin-top:.4rem">テーマ
           <a href="#/themes/${encodeURIComponent(i.cluster_uid)}">${esc(i.cluster_name || "(未命名)")}</a>
         </div>`
      : ""}
    <div class="body full">${esc(i.body) || '<span class="empty">（文字起こし待ち）</span>'}</div>
    <div class="actions">
      ${["inbox", "kept", "archived"].map(s =>
        `<button data-uid="${esc(i.uid)}" data-status="${s}"
                 aria-pressed="${i.status === s}">${STATUS_LABEL[s]}</button>`).join("")}
    </div>
    ${d.parent ? "<h2>返信元</h2>" + brief(d.parent) : ""}
    ${d.replies.length ? "<h2>追記</h2>" + d.replies.map(r => brief(r)).join("") : ""}
    ${d.related.length
      ? "<h2>関連メモ</h2>" + d.related.map(r => brief(r, r.kind)).join("")
      : ""}
    <p class="meta" style="margin-top:2rem">${esc(i.file_path)}</p>`);
}

// ------------------------------------------------------------
// 画面: 検索
// ------------------------------------------------------------
let timer;

function renderSearch(params) {
  const q = params.get("q") || "";
  show(`
    <input type="search" id="q" enterkeyhint="search" autocomplete="off"
           placeholder="キーワードでも、言い回しが違っても" value="${esc(q)}">
    <div id="results"></div>`);
  // iPhone でタブを踏んだだけでキーボードが出ないよう、空のときだけ焦点を置く
  if (!q) $("#q").focus();
  else runSearch(q);
}

async function runSearch(q) {
  $("#results").innerHTML = '<p class="empty">検索中…</p>';
  let data;
  try {
    data = await api(`/api/search?q=${encodeURIComponent(q)}`);
  } catch (e) {
    $("#results").innerHTML = `<p class="empty">${esc(e.message)}</p>`;
    return;
  }
  if (!data.results.length) {
    $("#results").innerHTML = '<p class="empty">見つかりませんでした</p>';
    return;
  }
  const mode = [data.vector ? "ベクトル" : null, data.keyword ? "全文" : null]
    .filter(Boolean).join(" + ");
  $("#results").innerHTML =
    `<p class="meta" style="margin:1rem 0 0">${data.results.length} 件 · ${mode}</p>` +
    ideaList(data.results, q);
}

// ------------------------------------------------------------
// 画面: テーマ
// ------------------------------------------------------------
async function renderThemes(params) {
  const sort = SORT_LABEL[params.get("sort")] ? params.get("sort") : "size";

  const chip = (value) =>
    `<a href="#/themes?sort=${value}" ${sort === value ? 'aria-current="page"' : ""}
      >${SORT_LABEL[value]}</a>`;

  show(`
    <div class="chips">${chip("size")}${chip("recent")}${chip("name")}</div>
    <div id="items"><p class="empty">読み込み中…</p></div>`);

  const data = await api("/api/clusters?sort=" + sort);

  // 未分類は行き止まりではなく脇道。上に置くとテーマより先に目に入ってしまう
  const unclustered = `<p class="meta" style="margin:1.5rem 0 0">
      <a href="#/?cluster=none">未分類のメモ ${data.unclustered} 件</a></p>`;

  if (!data.clusters.length) {
    $("#items").innerHTML =
      '<p class="empty">まだテーマができていません。話題が 2 つ以上に分かれると夜間に生成されます。</p>'
      + unclustered;
    return;
  }

  $("#items").innerHTML = data.clusters.map(c => `
    <a class="card" href="#/themes/${encodeURIComponent(c.uid)}">
      <div class="meta">
        <span class="name">${esc(c.name || "(未命名)")}</span>
        <span>${c.size} 件</span>
        <span>${fmtAgo(c.updated_at)}</span>
        ${c.challenges
          ? `<span class="warn">▲ 反証 ${c.challenges}</span>`
          : c.findings ? `<span>参考情報 ${c.findings}</span>` : ""}
      </div>
      <div class="body clamp">${esc(c.summary)}</div>
    </a>`).join("") + unclustered;
}

function finding(f) {
  const button = (verdict, label) =>
    `<button class="mini" data-verdict="${verdict}"
             aria-pressed="${f.verdict === verdict}">${label}</button>`;
  return `
    <div class="card ${f.verdict === "not_useful" ? "dropped" : ""}" data-finding="${f.id}">
      <div>
        <span class="stance-${f.stance}">${STANCE_MARK[f.stance] || "・"}</span>
        <a href="${esc(f.url)}" target="_blank" rel="noreferrer">${esc(f.title || f.url)}</a>
      </div>
      <div class="meta">
        ${KIND_LABEL[f.query_kind] ? `<span class="tag">${KIND_LABEL[f.query_kind]}</span>` : ""}
        <span>${esc(f.site || "")}</span>
      </div>
      <div class="body clamp">${esc(f.summary || "")}</div>
      <div class="meta" style="margin-top:.5rem">
        ${button("useful", "役に立った")}${button("not_useful", "外す")}
      </div>
    </div>`;
}

async function renderTheme(uid) {
  show('<p class="empty">読み込み中…</p>');
  const d = await api(`/api/clusters/${encodeURIComponent(uid)}`);
  const c = d.cluster;
  const base = new URLSearchParams({status: "all", cluster: c.uid});

  show(`
    <a class="back" href="#/themes">← テーマ</a>
    <div class="title" id="head">
      <h1>${esc(c.name || "(未命名)")}</h1>
      <button class="mini" id="rename" data-uid="${esc(c.uid)}">名前を変える</button>
    </div>
    <div class="meta">${c.size} 件${c.closed_at ? " · 解散済み" : ""}
      · 更新 ${fmtAgo(c.updated_at)}</div>
    ${c.summary ? `<div class="body full">${esc(c.summary)}</div>` : ""}
    <h2>メモ</h2>
    ${ideaList(d.members)}
    ${moreButton(base, d.members_next)}
    ${d.findings.length ? "<h2>参考情報</h2>" + d.findings.map(finding).join("") : ""}
    <p class="meta" style="margin-top:1.2rem">
      <a href="#/?cluster=${encodeURIComponent(c.uid)}&status=all">このテーマのメモを一覧で見る</a></p>`);

  // 名前は差し替えたら再描画で読み直す。楽観更新するほどの頻度ではない
  $("#head").dataset.name = c.name || "";
}

// ------------------------------------------------------------
// ルーティング (ハッシュ。iPhone の戻るジェスチャで戻れる)
// ------------------------------------------------------------
async function route() {
  const [path, query] = location.hash.replace(/^#\\/?/, "").split("?");
  const params = new URLSearchParams(query || "");
  const [head, arg] = [path.split("/")[0], decodeURIComponent(path.split("/")[1] || "")];

  const tab = head === "search" ? "search" : head === "themes" ? "themes" : "ideas";
  document.querySelectorAll("nav a").forEach(a => {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);

  try {
    if (head === "search") return renderSearch(params);
    if (head === "themes") return arg ? await renderTheme(arg) : await renderThemes(params);
    if (head === "ideas" && arg) return await renderIdea(arg);
    return await renderIdeas(params);
  } catch (e) {
    show(`<p class="empty">${esc(e.message)}</p>`);
  }
}

addEventListener("hashchange", route);
addEventListener("DOMContentLoaded", route);

// 検索は打つたびにハッシュを積むと戻るが効かなくなるので replaceState で差し替える
document.addEventListener("input", e => {
  if (e.target.id !== "q") return;
  clearTimeout(timer);
  const q = e.target.value.trim();
  history.replaceState(null, "", "#/search" + (q ? "?q=" + encodeURIComponent(q) : ""));
  if (!q) { $("#results").innerHTML = ""; return; }
  timer = setTimeout(() => runSearch(q), 300);
});

document.addEventListener("click", async e => {
  const more = e.target.closest("button[data-base]");
  if (more) {
    const label = more.textContent;
    more.textContent = "…";
    try {
      const data = await api(
        `/api/ideas?${more.dataset.base}&before=${encodeURIComponent(more.dataset.next)}`);
      // テーマの中では所属テーマ名のタグは自明なので落とす
      if (new URLSearchParams(more.dataset.base).has("cluster")) {
        data.ideas.forEach(r => { r.cluster_name = null; });
      }
      more.outerHTML = ideaList(data.ideas) + moreButton(more.dataset.base, data.next);
    } catch (err) {
      more.textContent = label + "（失敗。もう一度）";
    }
    return;
  }

  const rename = e.target.closest("#rename");
  if (rename) {
    const head = $("#head");
    head.innerHTML = `
      <input id="tname" value="${esc(head.dataset.name)}" enterkeyhint="done"
             autocomplete="off" placeholder="テーマ名">
      <button class="mini" id="tsave" data-uid="${esc(rename.dataset.uid)}">保存</button>
      <button class="mini" id="tcancel">やめる</button>`;
    $("#tname").focus();
    return;
  }

  if (e.target.closest("#tcancel")) return route();

  const save = e.target.closest("#tsave");
  if (save) {
    const name = $("#tname").value.trim();
    if (!name) return $("#tname").focus();
    save.textContent = "…";
    try {
      const res = await fetch(`/api/clusters/${encodeURIComponent(save.dataset.uid)}/name`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name}),
      });
      if (!res.ok) throw new Error(res.status);
    } catch (err) {
      save.textContent = "失敗";
      return;
    }
    return route();
  }

  /* 参考情報の当たり外れ。押し直しで取り消せるようにしておかないと、
     誤爆した判断がテーマノートに残り続ける */
  const verdict = e.target.closest("button[data-verdict]");
  if (verdict) {
    const card = verdict.closest("[data-finding]");
    const pressed = verdict.getAttribute("aria-pressed") === "true";
    const value = pressed ? null : verdict.dataset.verdict;
    const label = verdict.textContent;
    verdict.textContent = "…";
    try {
      const res = await fetch(`/api/findings/${card.dataset.finding}/verdict`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({verdict: value}),
      });
      if (!res.ok) throw new Error(res.status);
    } catch (err) {
      verdict.textContent = label + "（失敗）";
      return;
    }
    verdict.textContent = label;
    card.querySelectorAll("button[data-verdict]").forEach(b =>
      b.setAttribute("aria-pressed", b === verdict && !pressed));
    card.classList.toggle("dropped", value === "not_useful");
    return;
  }

  const status = e.target.closest("button[data-status]");
  if (status) {
    try {
      const res = await fetch(`/api/ideas/${encodeURIComponent(status.dataset.uid)}/status`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({status: status.dataset.status}),
      });
      if (!res.ok) throw new Error(res.status);
    } catch (err) {
      status.textContent = "失敗";       // 押したのに変わっていない、を見せる
      return;
    }
    status.parentElement.querySelectorAll("button").forEach(b =>
      b.setAttribute("aria-pressed", b === status));
  }
});
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX
