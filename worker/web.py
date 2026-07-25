"""
検索・俯瞰 UI (Raspberry Pi、Docker)

Obsidian で読めるものは作らない。DB を引かないとできないことだけを持つ:
  - ハイブリッド検索 (ベクトル + FTS5 を RRF で統合)
  - クラスタ一覧とテーマの中身
  - status の手動変更 (自動 archive はしない方針なので、ここが唯一の導線)

ビルドステップを持ちたくないので、HTML は 1 枚をインラインで返す。

  uvicorn web:app --host 0.0.0.0 --port 8080
"""

import json
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

app = FastAPI(title="idea-memo")


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
            SELECT i.id, i.uid, i.body, i.captured_at, i.status, i.file_path,
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
            results.append({**row, "score": round(score, 5)})

    return {
        "query": q,
        "vector": bool(vector_rank),
        "keyword": bool(keyword_rank),
        "results": results,
    }


@app.get("/api/clusters")
def clusters() -> dict:
    db = connect()
    rows = db.execute(
        """
        SELECT c.uid, c.name, c.summary, c.size, c.updated_at,
               (SELECT COUNT(*) FROM findings f WHERE f.cluster_id = c.id) AS findings
        FROM   clusters c
        WHERE  c.closed_at IS NULL AND c.size > 0
        ORDER BY c.size DESC
        """
    ).fetchall()

    noise = db.execute(
        """
        SELECT COUNT(*) AS n FROM ideas i
        WHERE  i.status != 'archived'
               AND NOT EXISTS (SELECT 1 FROM idea_clusters ic WHERE ic.idea_id = i.id)
        """
    ).fetchone()["n"]

    return {"clusters": [dict(r) for r in rows], "unclustered": noise}


@app.get("/api/clusters/{uid}")
def cluster_detail(uid: str) -> dict:
    db = connect()
    cluster = db.execute(
        "SELECT id, uid, name, summary, size FROM clusters WHERE uid = ?", (uid,)
    ).fetchone()
    if cluster is None:
        raise HTTPException(404, "cluster not found")

    members = db.execute(
        """
        SELECT i.uid, i.body, i.captured_at, i.status, ic.probability
        FROM   idea_clusters ic JOIN ideas i ON i.id = ic.idea_id
        WHERE  ic.cluster_id = ? ORDER BY i.captured_at DESC
        """,
        (cluster["id"],),
    ).fetchall()

    findings = db.execute(
        """
        SELECT title, url, site, summary, stance, query_kind, verdict
        FROM   findings WHERE cluster_id = ?
        ORDER BY stance = 'challenges' DESC, fetched_at DESC
        """,
        (cluster["id"],),
    ).fetchall()

    return {
        "cluster": dict(cluster),
        "members": [dict(r) for r in members],
        "findings": [dict(r) for r in findings],
    }


class StatusUpdate(BaseModel):
    status: str


@app.post("/api/ideas/{uid}/status")
def set_status(uid: str, payload: StatusUpdate) -> dict:
    if payload.status not in ("inbox", "kept", "archived"):
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
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>idea-memo</title>
<style>
  :root { color-scheme: light dark; --fg:#1a1a1a; --dim:#767676; --line:#e0e0e0; --bg:#fff; --accent:#3d5afe; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e8; --dim:#9a9a9a; --line:#333; --bg:#161616; --accent:#8c9eff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:1.5rem 1rem 4rem; max-width:52rem; margin-inline:auto;
         font:15px/1.7 system-ui,-apple-system,"Hiragino Sans",sans-serif;
         color:var(--fg); background:var(--bg); }
  h1 { font-size:1rem; letter-spacing:.08em; color:var(--dim); font-weight:600; margin:0 0 1.2rem; }
  nav { display:flex; gap:1rem; margin-bottom:1.2rem; }
  nav button { background:none; border:none; padding:.3rem 0; cursor:pointer;
               color:var(--dim); font:inherit; border-bottom:2px solid transparent; }
  nav button[aria-current="true"] { color:var(--fg); border-color:var(--accent); }
  input[type=search] { width:100%; padding:.7rem .9rem; font:inherit; color:var(--fg);
                       background:transparent; border:1px solid var(--line); border-radius:8px; }
  .card { border-bottom:1px solid var(--line); padding:.9rem 0; }
  .meta { font-size:.8rem; color:var(--dim); display:flex; gap:.8rem; flex-wrap:wrap; }
  .body { margin:.35rem 0; white-space:pre-wrap; }
  .tag { font-size:.75rem; border:1px solid var(--line); border-radius:99px; padding:.05rem .5rem; }
  .stance-challenges { color:#e0642f; }
  .stance-supports { color:#2e9e5b; }
  /* 不要と判断した参考情報。テーマノートからは消えるが、俯瞰では残す */
  .dropped { opacity:.45; }
  a { color:var(--accent); }
  .empty { color:var(--dim); padding:2rem 0; }
  .actions button { font-size:.75rem; margin-right:.4rem; cursor:pointer;
                    background:none; border:1px solid var(--line); border-radius:5px;
                    color:var(--dim); padding:.1rem .5rem; }
</style></head>
<body>
<h1>IDEA MEMO</h1>
<nav>
  <button id="tab-search" aria-current="true">検索</button>
  <button id="tab-themes">テーマ</button>
</nav>
<div id="search-pane">
  <input type="search" id="q" placeholder="キーワードでも、言い回しが違っても" autofocus>
  <div id="results"></div>
</div>
<div id="themes-pane" hidden><div id="themes"></div></div>

<script>
const $ = s => document.querySelector(s);
const esc = s => (s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function showTab(name) {
  $("#search-pane").hidden = name !== "search";
  $("#themes-pane").hidden = name !== "themes";
  $("#tab-search").setAttribute("aria-current", name === "search");
  $("#tab-themes").setAttribute("aria-current", name === "themes");
  if (name === "themes") loadThemes();
}
$("#tab-search").onclick = () => showTab("search");
$("#tab-themes").onclick = () => showTab("themes");

let timer;
$("#q").oninput = e => {
  clearTimeout(timer);
  const q = e.target.value.trim();
  if (!q) { $("#results").innerHTML = ""; return; }
  timer = setTimeout(() => runSearch(q), 300);
};

async function runSearch(q) {
  const data = await (await fetch(`/api/search?q=${encodeURIComponent(q)}`)).json();
  if (!data.results.length) { $("#results").innerHTML = '<p class="empty">見つかりませんでした</p>'; return; }
  const mode = [data.vector ? "ベクトル" : null, data.keyword ? "全文" : null].filter(Boolean).join(" + ");
  $("#results").innerHTML =
    `<p class="meta" style="margin:1rem 0 0">${data.results.length} 件 · ${mode}</p>` +
    data.results.map(r => `
      <div class="card">
        <div class="meta">
          <span>${r.captured_at.slice(0,10)}</span>
          ${r.cluster_name ? `<span class="tag">${esc(r.cluster_name)}</span>` : ""}
          <span>${r.status}</span>
        </div>
        <div class="body">${esc(r.body).slice(0,300)}</div>
        <div class="actions">
          <button onclick="setStatus('${r.uid}','kept',this)">残す</button>
          <button onclick="setStatus('${r.uid}','archived',this)">archive</button>
        </div>
      </div>`).join("");
}

async function setStatus(uid, status, btn) {
  await fetch(`/api/ideas/${uid}/status`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({status})
  });
  btn.parentElement.innerHTML = `<span class="meta">${status}</span>`;
}

async function loadThemes() {
  const data = await (await fetch("/api/clusters")).json();
  if (!data.clusters.length) {
    $("#themes").innerHTML = '<p class="empty">まだテーマができていません。メモが溜まると夜間に生成されます。</p>';
    return;
  }
  $("#themes").innerHTML =
    `<p class="meta">未分類 ${data.unclustered} 件</p>` +
    data.clusters.map(c => `
      <div class="card">
        <div class="meta">
          <strong style="color:var(--fg)">${esc(c.name)}</strong>
          <span>${c.size} 件</span>
          ${c.findings ? `<span>参考情報 ${c.findings}</span>` : ""}
        </div>
        <div class="body">${esc(c.summary)}</div>
        <button class="tag" onclick="loadCluster('${c.uid}', this)">開く</button>
        <div class="detail"></div>
      </div>`).join("");
}

async function loadCluster(uid, btn) {
  const box = btn.nextElementSibling;
  if (box.innerHTML) { box.innerHTML = ""; return; }
  const d = await (await fetch(`/api/clusters/${uid}`)).json();
  box.innerHTML =
    d.findings.map(f => `
      <div class="card ${f.verdict === "not_useful" ? "dropped" : ""}" style="border:none;padding:.4rem 0">
        <span class="stance-${f.stance}">${f.stance === "challenges" ? "▲" : f.stance === "supports" ? "◯" : "・"}</span>
        <a href="${f.url}" target="_blank" rel="noreferrer">${esc(f.title)}</a>
        <div class="meta">${esc(f.summary)}</div>
      </div>`).join("") +
    d.members.map(m => `
      <div class="card" style="border:none;padding:.3rem 0">
        <span class="meta">${m.captured_at.slice(0,10)}</span>
        <div class="body">${esc(m.body).slice(0,200)}</div>
      </div>`).join("");
}
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX
