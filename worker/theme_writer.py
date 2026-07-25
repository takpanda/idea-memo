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

    msg = payload["choices"][0]["message"]; text = (msg.get("content") or msg.get("reasoning") or "").strip()
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
        SELECT id, uid, name, name_locked, file_path, member_hash, named_member_hash
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


def absorb_overrides(db) -> int:
    """テーマノートの front matter に入った人の判断を DB に吸い上げる。

    メンバーが動くまで待つと、Obsidian に出る名前と Web UI・ダイジェストに
    出る名前が食い違ったままになるので、命名待ちかどうかに関係なく毎周回で見る。
    開いているテーマの数だけの小さなファイル読みなので素直に全部読む。
    """
    changed = 0
    rows = db.execute(
        """
        SELECT id, name, name_locked, file_path
        FROM   clusters
        WHERE  closed_at IS NULL AND file_path IS NOT NULL
        """
    ).fetchall()

    for row in rows:
        overrides = read_theme_overrides(REPO_ROOT / row["file_path"])
        if "name_locked" not in overrides:
            continue

        locked = overrides["name_locked"]
        # 名前を人から取るのはロックされているときだけ。
        # ロックが無ければテーマノートは DB の出力でしかない
        name = (overrides.get("name") or row["name"]) if locked else row["name"]
        if (locked, name) == (row["name_locked"], row["name"]):
            continue

        with db:
            db.execute(
                """
                UPDATE clusters
                SET    name = ?, name_locked = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE  id = ?
                """,
                (name, locked, row["id"]),
            )
        write_theme_markdown(db, row["id"])
        changed += 1
        log.info("absorbed override: %s (locked=%d)", name, locked)

    return changed


def process(db, cluster) -> bool:
    cluster_id = cluster["id"]
    file_path = cluster["file_path"] or f"themes/{cluster['uid']}.md"

    memos = collect_memos(db, cluster_id)
    if not memos:
        return False

    result = call_llm(memos)
    if result is None:
        # 確定させずに戻る。次の周回で拾い直すので、進捗としては数えない
        return False

    # 名前は初回のみ。既にあるもの・人が固定したものは触らない (ちらつき防止)
    keep_name = bool(cluster["name"]) or bool(cluster["name_locked"])
    # 改行入りの名前を返してくることがある。見出しにも front matter にも載るので均す
    new_name = cluster["name"] if keep_name else " ".join(result["name"].split())[:40]

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
    return True


def tick(db) -> int:
    """進んだ件数を返す。拾えなかったクラスタを数に入れると
    supervisor が「まだ仕事がある」と見て毎周回で空回りする。"""
    done = absorb_overrides(db)
    for cluster in fetch_pending(db):
        if process(db, cluster):
            done += 1
    return done


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
