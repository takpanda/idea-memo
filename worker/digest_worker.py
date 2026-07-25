"""
週次ダイジェスト (Raspberry Pi、週次バッチ)

過去 1 週間を振り返る Markdown を digests/ に出し、Telegram に送る。
末尾に読み上げ用の原稿を付けてあるので、Mac mini の TTS に流せば
MyNews Radio 側で音声にできる。

  python digest_worker.py
"""

import datetime as dt
import json
import logging
import os
import time
import urllib.error
import urllib.request

from common import CHAT_ID, REPO_ROOT, api, connect

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://gpu-node:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
DAYS = 7

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("digest-worker")

STANCE_MARK = {"supports": "◯", "challenges": "▲", "neutral": "・"}


def since() -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def gather(db, cutoff: str) -> dict:
    new_ideas = db.execute(
        "SELECT COUNT(*) AS n FROM ideas WHERE captured_at >= ?", (cutoff,)
    ).fetchone()["n"]

    new_clusters = db.execute(
        "SELECT uid, name, size FROM clusters "
        "WHERE closed_at IS NULL AND created_at >= ? ORDER BY size DESC",
        (cutoff,),
    ).fetchall()

    findings = db.execute(
        """
        SELECT c.name AS cluster_name, f.title, f.url, f.site, f.summary,
               f.stance, f.query_kind
        FROM   findings f JOIN clusters c ON c.id = f.cluster_id
        WHERE  f.fetched_at >= ?
        ORDER BY c.name, f.stance
        """,
        (cutoff,),
    ).fetchall()

    # 押されずに流れた関連候補。溜まっていたら閾値が合っていない兆候
    undecided = db.execute(
        """
        SELECT COUNT(*) AS n FROM ideas i
        WHERE  i.similar_notified_at >= ?
               AND NOT EXISTS (SELECT 1 FROM relations r
                               WHERE r.src_id = i.id OR r.dst_id = i.id)
        """,
        (cutoff,),
    ).fetchone()["n"]

    return {
        "new_ideas": new_ideas,
        "new_clusters": new_clusters,
        "findings": findings,
        "undecided": undecided,
    }


def write_intro(data: dict) -> str:
    """LLM が使えないときはテンプレートで済ませる。落とすほどのものではない。"""
    fallback = f"今週は {data['new_ideas']} 件のメモが入りました。"
    if not data["new_ideas"]:
        return "今週は新しいメモがありませんでした。"

    themes = ", ".join(c["name"] for c in data["new_clusters"]) or "なし"
    body = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "週次の振り返りを書く編集者です。JSON だけを返してください。",
                },
                {
                    "role": "user",
                    "content": (
                        f"新規メモ {data['new_ideas']} 件、新しく生まれたテーマ: {themes}、"
                        f"収集した参考情報 {len(data['findings'])} 件。\n"
                        '{"intro": "2文以内。今週の関心がどこに向いていたかを述べる"} '
                        "の形式で返してください。"
                    ),
                },
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{LLM_BASE_URL}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            payload = json.loads(res.read())
        text = payload["choices"][0]["message"]["content"].strip().strip("`")
        return json.loads(text[text.index("{"):]).get("intro", fallback)
    except Exception as exc:
        log.info("intro generation skipped: %s", exc)
        return fallback


def render(data: dict, intro: str, week: str) -> tuple[str, str]:
    lines = [
        "---",
        f"week: {week}",
        f"generated_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "---",
        "",
        f"# 週次ダイジェスト {week}",
        "",
        intro,
        "",
        f"- 新規メモ: {data['new_ideas']} 件",
        f"- 新しいテーマ: {len(data['new_clusters'])} 件",
        f"- 収集した参考情報: {len(data['findings'])} 件",
        f"- 判断待ちの関連候補があるメモ: {data['undecided']} 件",
        "",
    ]

    if data["new_clusters"]:
        lines += ["## 新しく生まれたテーマ", ""]
        lines += [
            f"- **{c['name']}** ({c['size']} 件) — `themes/{c['uid']}.md`"
            for c in data["new_clusters"]
        ]
        lines.append("")

    if data["findings"]:
        lines += ["## 集まった参考情報", ""]
        current = None
        for f in data["findings"]:
            if f["cluster_name"] != current:
                current = f["cluster_name"]
                lines += [f"### {current}", ""]
            mark = STANCE_MARK.get(f["stance"], "・")
            lines.append(f"- {mark} [{f['title']}]({f['url']}) — {f['site']}")
            if f["summary"]:
                lines.append(f"  {f['summary']}")
        lines += ["", "◯ 裏付け / ▲ 反証・注意点 / ・ 参考", ""]

    # TTS に流す用。記号やリンクを含めない素の文章にする
    script = [intro]
    if data["new_clusters"]:
        script.append(
            "今週新しく生まれたテーマは、"
            + "、".join(c["name"] for c in data["new_clusters"])
            + " です。"
        )
    for f in data["findings"]:
        if f["stance"] == "challenges" and f["summary"]:
            script.append(f"{f['cluster_name']} について、気になる指摘があります。{f['summary']}")
    if data["undecided"]:
        script.append(f"判断待ちの関連候補が {data['undecided']} 件たまっています。")

    lines += ["## 読み上げ原稿", "", " ".join(script), ""]

    telegram = "\n".join(
        [
            f"📻 週次ダイジェスト {week}",
            "",
            intro,
            "",
            f"新規メモ {data['new_ideas']} 件 / 新テーマ {len(data['new_clusters'])} 件 "
            f"/ 参考情報 {len(data['findings'])} 件",
        ]
    )
    return "\n".join(lines), telegram


def main() -> None:
    db = connect()
    now = dt.datetime.now(dt.timezone.utc)
    week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"

    data = gather(db, since())
    intro = write_intro(data)
    markdown, message = render(data, intro, week)

    path = REPO_ROOT / "digests" / f"{week}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    log.info("wrote %s", path)

    try:
        api("sendMessage", chat_id=CHAT_ID, text=message)
    except urllib.error.URLError as exc:
        log.warning("telegram unreachable: %s", exc)


if __name__ == "__main__":
    main()
