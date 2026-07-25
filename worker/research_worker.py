"""
調査ワーカー (Raspberry Pi、週次バッチ)

テーマ (クラスタ) ごとに検索クエリを LLM に作らせ、Web を引いて
参考情報として findings に溜める。

賛成材料だけ集めても価値が薄いので、クエリを 3 種類に撃ち分ける:
  prior_art  既に誰かがやっていないか (先行事例・既存 OSS)
  evidence   技術的な裏付け
  counter    否定的知見・失敗事例・批判

調査単位を個別メモではなくテーマにしているのは、走り書き 1 件からは
まともな検索クエリが作れず、収集結果がノイズだらけになるため。

  SEARCH_BACKEND=brave BRAVE_API_KEY=... python research_worker.py
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from common import connect

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://gpu-node:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

SEARCH_BACKEND = os.environ.get("SEARCH_BACKEND", "brave")   # 'brave' | 'searxng'
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

MIN_CLUSTER_SIZE = 3
RESEARCH_INTERVAL_DAYS = 7
RESULTS_PER_QUERY = 5
MAX_FINDINGS_PER_RUN = 3        # 1 テーマあたり。増やすとノイズで埋まる
FETCH_LIMIT_BYTES = 400_000
FETCH_TEXT_CHARS = 8_000
USER_AGENT = "idea-memo-research/1.0 (personal knowledge tool)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("research-worker")


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
def call_llm(system: str, user: str, max_tokens: int = 900) -> dict | list | None:
    body = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as res:
        payload = json.loads(res.read())

    text = payload["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON: %s", text[:200])
        return None


QUERY_SYSTEM = (
    "あなたは調査担当者です。与えられたテーマについて Web 検索クエリを設計し、"
    "JSON だけを返してください。前置きやコードフェンスは不要です。"
)

QUERY_PROMPT = """テーマ: {name}
概要: {summary}

含まれるメモ:
{memos}

このテーマについて調べるための検索クエリを、次の 3 観点で作ってください。

{{"prior_art": ["..."], "evidence": ["..."], "counter": ["..."]}}

- prior_art: 既に同じことをやっている事例や既存の OSS を探すクエリ
- evidence: 技術的な裏付けや実装方法を探すクエリ
- counter: 否定的な知見、失敗事例、批判を探すクエリ
- 各観点 1〜2 件。日本語と英語を混ぜる
- メモに書かれていない前提を勝手に足さない
"""

SELECT_SYSTEM = (
    "あなたは調査結果を選別する編集者です。JSON だけを返してください。"
)

SELECT_PROMPT = """テーマ: {name}
概要: {summary}

検索結果の候補:
{candidates}

このテーマを考えるうえで実際に価値のあるものを最大 {limit} 件選び、次の形式で返してください。

{{"picks": [{{"index": 0, "stance": "supports|challenges|neutral", "reason": "選んだ理由を1文で"}}]}}

- 内容の薄いまとめ記事、広告、無関係なものは選ばない
- 価値のあるものが無ければ picks は空配列にする
- challenges (テーマに不利な情報) は積極的に拾う。都合の良い情報だけ集めない
"""

SUMMARIZE_SYSTEM = (
    "あなたは技術記事を要約する編集者です。JSON だけを返してください。"
)

SUMMARIZE_PROMPT = """テーマ: {name}

記事タイトル: {title}
URL: {url}
本文 (抜粋):
{body}

次の形式で返してください。

{{"summary": "このテーマにとって何が言えるかを3文以内で。原文の言い回しをそのまま写さず自分の言葉で書く"}}
"""


# ------------------------------------------------------------
# 検索
# ------------------------------------------------------------
def search_brave(query: str) -> list[dict]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
        {"q": query, "count": RESULTS_PER_QUERY}
    )
    req = urllib.request.Request(
        url, headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        payload = json.loads(res.read())

    return [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": item.get("description", ""),
            "site": (item.get("meta_url") or {}).get("hostname", ""),
        }
        for item in (payload.get("web") or {}).get("results", [])
    ]


def search_searxng(query: str) -> list[dict]:
    url = f"{SEARXNG_URL}/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as res:
        payload = json.loads(res.read())

    return [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "site": urllib.parse.urlparse(item.get("url", "")).netloc,
        }
        for item in payload.get("results", [])[:RESULTS_PER_QUERY]
    ]


def search(query: str) -> list[dict]:
    backend = search_searxng if SEARCH_BACKEND == "searxng" else search_brave
    try:
        return backend(query)
    except Exception as exc:
        log.warning("search failed (%s): %s", query, exc)
        return []


_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)
_WS = re.compile(r"\s+")


def fetch_text(url: str) -> str:
    """本文をざっくり取る。失敗してもスニペットで代替できるので握り潰す。

    ちゃんとしたリーダビリティ抽出はしていない。精度が要るようになったら
    trafilatura 等に差し替える。
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as res:
            ctype = res.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            raw = res.read(FETCH_LIMIT_BYTES).decode("utf-8", errors="ignore")
    except Exception as exc:
        log.info("fetch failed %s: %s", url, exc)
        return ""

    return _WS.sub(" ", _TAG.sub(" ", raw)).strip()[:FETCH_TEXT_CHARS]


# ------------------------------------------------------------
# 本体
# ------------------------------------------------------------
def fetch_pending(db, limit: int = 3):
    """未調査、または前回調査後にメンバーが増えて 1 週間経ったテーマ。"""
    return db.execute(
        """
        SELECT id, uid, name, summary, member_hash
        FROM   clusters
        WHERE  closed_at IS NULL
               AND name IS NOT NULL
               AND size >= ?
               AND (researched_at IS NULL
                    OR (member_hash IS NOT researched_member_hash
                        AND researched_at < datetime('now', ?)))
        ORDER BY researched_at IS NOT NULL, size DESC
        LIMIT ?
        """,
        (MIN_CLUSTER_SIZE, f"-{RESEARCH_INTERVAL_DAYS} days", limit),
    ).fetchall()


def collect_memos(db, cluster_id: int) -> str:
    rows = db.execute(
        """
        SELECT i.body FROM idea_clusters ic JOIN ideas i ON i.id = ic.idea_id
        WHERE  ic.cluster_id = ? ORDER BY ic.probability DESC LIMIT 15
        """,
        (cluster_id,),
    ).fetchall()
    return "\n".join(f"- {' '.join(r['body'].split())[:150]}" for r in rows)


def known_urls(db, cluster_id: int) -> set[str]:
    return {
        r["url"]
        for r in db.execute("SELECT url FROM findings WHERE cluster_id = ?", (cluster_id,))
    }


def research(db, cluster) -> None:
    memos = collect_memos(db, cluster["id"])
    plan = call_llm(
        QUERY_SYSTEM,
        QUERY_PROMPT.format(
            name=cluster["name"], summary=cluster["summary"] or "", memos=memos
        ),
    )
    if not isinstance(plan, dict):
        return

    seen = known_urls(db, cluster["id"])
    candidates: list[dict] = []
    for kind in ("prior_art", "evidence", "counter"):
        for query in (plan.get(kind) or [])[:2]:
            for hit in search(str(query)):
                if hit["url"] and hit["url"] not in seen:
                    seen.add(hit["url"])
                    candidates.append({**hit, "query_kind": kind, "query": str(query)})
            time.sleep(1)          # 検索 API へのマナー

    if not candidates:
        log.info("no new candidates for %s", cluster["name"])
    else:
        listing = "\n".join(
            f"[{i}] {c['title']} ({c['site']})\n    {c['snippet'][:200]}"
            for i, c in enumerate(candidates)
        )
        selection = call_llm(
            SELECT_SYSTEM,
            SELECT_PROMPT.format(
                name=cluster["name"],
                summary=cluster["summary"] or "",
                candidates=listing,
                limit=MAX_FINDINGS_PER_RUN,
            ),
        )
        picks = (selection or {}).get("picks", []) if isinstance(selection, dict) else []

        for pick in picks[:MAX_FINDINGS_PER_RUN]:
            try:
                item = candidates[int(pick["index"])]
            except (KeyError, ValueError, IndexError):
                continue

            body = fetch_text(item["url"]) or item["snippet"]
            written = call_llm(
                SUMMARIZE_SYSTEM,
                SUMMARIZE_PROMPT.format(
                    name=cluster["name"], title=item["title"], url=item["url"], body=body
                ),
                max_tokens=400,
            )
            summary = (written or {}).get("summary") if isinstance(written, dict) else None

            with db:
                db.execute(
                    """
                    INSERT INTO findings(cluster_id, query_kind, query, url, title,
                                         site, summary, stance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cluster_id, url) DO NOTHING
                    """,
                    (
                        cluster["id"],
                        item["query_kind"],
                        item["query"],
                        item["url"],
                        item["title"],
                        item["site"],
                        summary or pick.get("reason", ""),
                        pick.get("stance"),
                    ),
                )
            log.info("finding: %s (%s)", item["title"][:50], pick.get("stance"))

    with db:
        db.execute(
            """
            UPDATE clusters
            SET    researched_at = strftime('%Y-%m-%dT%H:%M:%SZ','now'),
                   researched_member_hash = ?
            WHERE  id = ?
            """,
            (cluster["member_hash"], cluster["id"]),
        )


def main() -> None:
    db = connect()
    for cluster in fetch_pending(db):
        try:
            research(db, cluster)
        except urllib.error.URLError as exc:
            log.warning("network error on %s: %s", cluster["name"], exc)
        except Exception:
            log.exception("research failed for %s", cluster["name"])


if __name__ == "__main__":
    main()
