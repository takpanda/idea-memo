"""
テーマノート生成ワーカー (Raspberry Pi 常駐)

命名されていない / メンバーが変わったクラスタを拾って、
GPU ノードの LLM にテーマ名と要約を書かせ、themes/<uid>.md を出力する。

方針:
  - 名前は初回だけ生成する。毎晩付け直すとテーマ名がちらついて追えない
  - 要約はメンバーが変わるたびに作り直す
  - Markdown の front matter で name_locked: true にすると人の名前が優先される

  LLM_BASE_URL=http://gpu-node:8000/v1 python theme_writer.py
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

from common import REPO_ROOT, connect, read_theme_overrides, write_theme_markdown

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://gpu-node:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
MAX_MEMOS_IN_PROMPT = 25
SNIPPET_LEN = 200
INTERVAL_SEC = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("theme-writer")

SYSTEM = (
    "あなたは個人のアイデアメモを整理する編集者です。"
    "与えられたメモ群に共通する主題を読み取り、JSON だけを返してください。"
    "前置き、説明、コードフェンスは一切不要です。"
)

PROMPT = """以下は同じクラスタに分類されたメモです。

{memos}

次の形式の JSON を返してください。

{{"name": "20文字以内のテーマ名", "summary": "3文以内。何についての集まりで、どんな論点が含まれるか"}}

制約:
- name は具体的に。「アイデア」「メモ」「その他」のような中身のない名前にしない
- メモに書かれていないことを推測で足さない
"""


def call_llm(memos: list[str]) -> dict | None:
    body = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(memos="\n".join(memos))},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        payload = json.loads(res.read())

    text = payload["choices"][0]["message"]["content"].strip()
    # JSON だけを返せと言っても囲ってくる場合があるので保険
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        log.warning("LLM returned non-JSON: %s", text[:200])
        return None

    if not isinstance(result.get("name"), str):
        return None
    return result


def fetch_pending(db, limit: int = 5):
    """名前が無い、または命名時からメンバーが変わったクラスタ。"""
    return db.execute(
        """
        SELECT id, uid, name, name_locked, member_hash, named_member_hash
        FROM   clusters
        WHERE  closed_at IS NULL
               AND (name IS NULL OR named_member_hash IS NOT member_hash)
        ORDER BY size DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def collect_memos(db, cluster_id: int) -> list[str]:
    rows = db.execute(
        """
        SELECT i.body, i.captured_at
        FROM   idea_clusters ic
               JOIN ideas i ON i.id = ic.idea_id
        WHERE  ic.cluster_id = ?
        ORDER BY ic.probability DESC, i.captured_at
        LIMIT ?
        """,
        (cluster_id, MAX_MEMOS_IN_PROMPT),
    ).fetchall()

    memos = []
    for row in rows:
        flat = " ".join(row["body"].split())[:SNIPPET_LEN]
        memos.append(f"- ({row['captured_at'][:10]}) {flat}")
    return memos


def process(db, cluster) -> None:
    cluster_id = cluster["id"]
    file_path = f"themes/{cluster['uid']}.md"

    # 人が Markdown を直していたら、そちらを先に DB へ吸い上げる
    overrides = read_theme_overrides(REPO_ROOT / file_path)
    name_locked = overrides.get("name_locked", cluster["name_locked"])
    if overrides.get("name_locked") and overrides.get("name"):
        with db:
            db.execute(
                "UPDATE clusters SET name = ?, name_locked = 1 WHERE id = ?",
                (overrides["name"], cluster_id),
            )

    memos = collect_memos(db, cluster_id)
    if not memos:
        return

    result = call_llm(memos)
    if result is None:
        return

    # 名前は初回のみ。既にあるものは触らない (ちらつき防止)
    keep_name = bool(cluster["name"]) or bool(name_locked)
    new_name = cluster["name"] if keep_name else result["name"][:40]

    with db:
        db.execute(
            """
            UPDATE clusters
            SET    name = ?, summary = ?, file_path = ?,
                   named_member_hash = member_hash,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE  id = ?
            """,
            (new_name, result.get("summary", ""), file_path, cluster_id),
        )

    write_theme_markdown(db, cluster_id)
    log.info("theme %s -> %s (%d memos)", cluster["uid"], new_name, len(memos))


def tick(db) -> int:
    clusters = fetch_pending(db)
    for cluster in clusters:
        process(db, cluster)
    return len(clusters)


def main() -> None:
    db = connect()
    while True:
        try:
            while tick(db) > 0:
                pass
        except urllib.error.URLError as exc:
            log.warning("LLM unreachable: %s", exc)
        except Exception:
            log.exception("tick failed")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
