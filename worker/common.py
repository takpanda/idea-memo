"""
共通処理: DB 接続、Telegram API、Markdown 再生成。

Markdown は DB から再生成する。DB にしか無い情報を作らないのが原則なので、
逆向き (Markdown -> DB) の復旧も同じ項目で成立する。
"""

import json
import os
import sqlite3
import urllib.request
from pathlib import Path

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = int(os.environ.get("ALLOWED_CHAT_ID", "0"))
DB_PATH = os.environ.get("IDEA_DB", "idea.db")
REPO_ROOT = Path(os.environ.get("IDEA_REPO", "."))

API = f"https://api.telegram.org/bot{TOKEN}"


def connect(with_vec: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    if with_vec:
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 5000")
    db.row_factory = sqlite3.Row
    return db


def api(method: str, timeout: int = 65, **params):
    req = urllib.request.Request(
        f"{API}/{method}",
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        payload = json.loads(res.read())
    if not payload.get("ok"):
        raise RuntimeError(f"telegram api error: {payload}")
    return payload["result"]


# ------------------------------------------------------------
# Markdown 再生成
# ------------------------------------------------------------
def write_idea_markdown(db: sqlite3.Connection, idea_id: int) -> Path | None:
    idea = db.execute(
        """
        SELECT uid, body, captured_at, source, input_kind, status, file_path
        FROM   ideas WHERE id = ?
        """,
        (idea_id,),
    ).fetchone()
    if idea is None:
        return None

    attachments = db.execute(
        "SELECT file_path, meta_json FROM attachments WHERE idea_id = ? ORDER BY id",
        (idea_id,),
    ).fetchall()

    # 人が承認した関連だけを書き戻す。提案は DB 側にも残さない
    related = db.execute(
        """
        SELECT o.uid, r.kind, r.score
        FROM   relations r
               JOIN ideas o
                 ON o.id = CASE WHEN r.src_id = ? THEN r.dst_id ELSE r.src_id END
        WHERE  (r.src_id = ? OR r.dst_id = ?)
               AND r.verdict = 'confirmed'
        ORDER BY o.uid
        """,
        (idea_id, idea_id, idea_id),
    ).fetchall()

    lines = [
        "---",
        f"uid: {idea['uid']}",
        f"captured_at: {idea['captured_at']}",
        f"source: {idea['source']}",
        f"input_kind: {idea['input_kind']}",
        f"status: {idea['status']}",
    ]

    if attachments:
        lines.append("attachments:")
        for att in attachments:
            meta = json.loads(att["meta_json"] or "{}")
            lines.append(f"  - path: {att['file_path']}")
            if meta.get("sha256"):
                lines.append(f"    sha256: {meta['sha256']}")

    if related:
        lines.append("related:")
        for rel in related:
            lines.append(f"  - uid: {rel['uid']}")
            lines.append(f"    kind: {rel['kind']}")
            if rel["score"] is not None:
                lines.append(f"    score: {rel['score']:.3f}")

    lines += ["---", "", idea["body"], ""]

    path = REPO_ROOT / idea["file_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------
# テーマノート (Phase 2)
# ------------------------------------------------------------
def yaml_value(text: str) -> str:
    """front matter の 1 行に載せる値。

    テーマ名は LLM も人も自由に書くので、`メモ: 整理` のようにコロンが
    入ると front matter ごと壊れる。常に引用して逃がす。
    """
    flat = " ".join(text.split())
    return '"' + flat.replace("\\", "\\\\").replace('"', '\\"') + '"'


def unquote_yaml(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
        if value and '\\' in value:
            value = value.replace('\\"', '"').replace("\\\\", "\\")
    return value


def read_theme_overrides(path: Path) -> dict:
    """既存のテーマノートから人の編集を読み戻す。

    name_locked: true にしておけば LLM が名前を上書きしない。
    YAML パーサを足したくないので front matter を素朴に走査する。
    """
    if not path.exists():
        return {}

    overrides: dict = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("  "):          # ネストした members: は見ない
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), unquote_yaml(value.strip())
        if key == "name" and value:
            overrides["name"] = value
        elif key == "name_locked":
            overrides["name_locked"] = 1 if value.lower() == "true" else 0

    return overrides


def write_theme_markdown(db: sqlite3.Connection, cluster_id: int) -> Path | None:
    cluster = db.execute(
        """
        SELECT uid, name, name_locked, summary, size, file_path, updated_at,
               closed_at, members_json
        FROM   clusters WHERE id = ?
        """,
        (cluster_id,),
    ).fetchone()
    if cluster is None or not cluster["file_path"]:
        return None

    members = db.execute(
        """
        SELECT i.uid, i.body, i.captured_at, i.file_path, ic.probability
        FROM   idea_clusters ic
               JOIN ideas i ON i.id = ic.idea_id
        WHERE  ic.cluster_id = ?
        ORDER BY i.captured_at
        """,
        (cluster_id,),
    ).fetchall()

    ids = json.loads(cluster["members_json"] or "[]")
    if cluster["closed_at"] and not members and ids:
        # 閉じたクラスタは割当が消えている。最後に見たメンバーを載せて、
        # 「何が集まっていた話題だったのか」を追えるようにしておく
        members = db.execute(
            f"""
            SELECT uid, body, captured_at, file_path, NULL AS probability
            FROM   ideas WHERE id IN ({','.join('?' * len(ids))})
            ORDER BY captured_at
            """,
            tuple(ids),
        ).fetchall()

    lines = [
        "---",
        f"cluster_uid: {cluster['uid']}",
        f"name: {yaml_value(cluster['name'] or '')}",
        f"name_locked: {'true' if cluster['name_locked'] else 'false'}",
        f"size: {len(members)}",
        f"updated_at: {cluster['updated_at']}",
    ]
    if cluster["closed_at"]:
        lines.append(f"closed_at: {cluster['closed_at']}")
    lines.append("members:")
    lines += [f"  - uid: {m['uid']}" for m in members]
    lines += ["---", ""]

    lines.append(f"# {cluster['name'] or '(未命名)'}")
    lines.append("")
    if cluster["closed_at"]:
        lines += [f"> {cluster['closed_at'][:10]} にこのテーマは解散した"
                  "（以下は解散時点のメンバー）。同じ話題が戻ってくれば同じテーマに戻る。",
                  ""]
    if cluster["summary"]:
        lines += [cluster["summary"], ""]

    lines.append("## メモ")
    lines.append("")
    for m in members:
        flat = " ".join(m["body"].split())
        snippet = flat[:80] + ("…" if len(flat) > 80 else "")
        lines.append(f"- `{m['captured_at'][:10]}` [{snippet}]({m['file_path']})")
    lines.append("")

    path = REPO_ROOT / cluster["file_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
