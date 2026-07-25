"""
Telegram 受け口 (Raspberry Pi 常駐)

long polling で Bot 宛の更新を取り込む。公開 Webhook が不要なので
Pi を外に出さずに済む。

  テキスト   -> body に格納
  ボイス     -> blobs/ に保存し body は空のまま (transcribe_worker が埋める)
  返信       -> parent_id で親メモに紐付け
  編集       -> body を更新 (content_hash が変わるので再埋め込みは自動)
  ボタン押下 -> relations に confirmed / rejected を記録

依存は標準ライブラリのみ。

  TELEGRAM_BOT_TOKEN=... ALLOWED_CHAT_ID=... python telegram_ingest.py
"""

import hashlib
import html
import json
import logging
import re
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from common import (
    CHAT_ID,
    REPO_ROOT,
    TOKEN,
    api,
    connect,
    write_idea_markdown,
    write_theme_markdown,
)

IDEA_DIR = REPO_ROOT / "ideas"
BLOB_DIR = REPO_ROOT / "blobs"          # .gitignore 対象

FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"
POLL_TIMEOUT = 50
MAX_FILE_BYTES = 20 * 1024 * 1024       # Bot API のダウンロード上限
MAX_PAGE_BYTES = 2 * 1024 * 1024       # URL本文の取得上限
PAGE_TIMEOUT = 30
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


class _PageTextParser(HTMLParser):
    """HTMLからタイトルと可視テキストだけを取り出す。"""

    _IGNORED_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag == "title" and self._ignored_depth == 0:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = " ".join(data.split())
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        else:
            self.text_parts.append(value)


def extract_url(text: str) -> str | None:
    """メッセージから最初のHTTP(S) URLを取り出す。"""
    match = URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,!?;:)]}、。")


def fetch_url_content(url: str) -> str:
    """URLのタイトルと可視本文を取得し、アイデア本文用の文字列を返す。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "idea-memo/1.0"},
    )
    with urllib.request.urlopen(request, timeout=PAGE_TIMEOUT) as response:
        data = response.read(MAX_PAGE_BYTES + 1)
        if len(data) > MAX_PAGE_BYTES:
            raise RuntimeError("URL content is too large")
        charset = response.headers.get_content_charset() if response.headers else None
    source = data.decode(charset or "utf-8", errors="replace")
    parser = _PageTextParser()
    parser.feed(source)
    title = " ".join(parser.title_parts).strip()
    body = " ".join(parser.text_parts).strip()
    if not title and not body:
        raise RuntimeError("URL content is empty")
    lines = [f"URL: {url}"]
    if title:
        lines.append(f"タイトル: {html.unescape(title)}")
    if body:
        lines.append(body)
    return "\n\n".join(lines)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("telegram-ingest")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    """時刻順にソートできる 26 文字の ID。外部依存を増やしたくないので自前。"""
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def get_offset(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT value FROM kv WHERE key = 'telegram_offset'").fetchone()
    return int(row["value"]) if row else 0


def set_offset(db: sqlite3.Connection, offset: int) -> None:
    db.execute(
        """
        INSERT INTO kv(key, value) VALUES ('telegram_offset', ?)
        ON CONFLICT(key) DO UPDATE SET value      = excluded.value,
                                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
        """,
        (str(offset),),
    )


def download_voice(file_id: str, uid: str) -> tuple[Path, int, str]:
    info = api("getFile", file_id=file_id)
    if int(info.get("file_size", 0)) > MAX_FILE_BYTES:
        raise RuntimeError(f"voice file too large: {info.get('file_size')} bytes")

    remote = info["file_path"]
    BLOB_DIR.mkdir(parents=True, exist_ok=True)
    local = BLOB_DIR / f"{uid}{Path(remote).suffix or '.ogg'}"

    with urllib.request.urlopen(f"{FILE_API}/{remote}", timeout=120) as res:
        blob = res.read()
    local.write_bytes(blob)

    return local, len(blob), hashlib.sha256(blob).hexdigest()


def markdown_path(uid: str, captured: datetime) -> Path:
    stamp = captured.strftime("%Y%m%dT%H%M%SZ")
    return IDEA_DIR / f"{captured:%Y}" / f"{captured:%m}" / f"{stamp}-{uid[:8]}.md"


# ------------------------------------------------------------
# メッセージ
# ------------------------------------------------------------
def resolve_parent(db: sqlite3.Connection, chat_id: int, msg: dict) -> int | None:
    reply = msg.get("reply_to_message")
    if not reply:
        return None
    row = db.execute(
        "SELECT id FROM ideas WHERE source = 'telegram' AND source_ref = ?",
        (f"{chat_id}:{reply['message_id']}",),
    ).fetchone()
    return row["id"] if row else None


def handle_message(db: sqlite3.Connection, msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    if chat_id != CHAT_ID:
        log.warning("ignoring message from unexpected chat %s", chat_id)
        return

    source_ref = f"{chat_id}:{msg['message_id']}"
    if db.execute(
        "SELECT 1 FROM ideas WHERE source = 'telegram' AND source_ref = ?", (source_ref,)
    ).fetchone():
        return  # 再配信。UNIQUE 制約もあるが手前で弾く

    uid = ulid()
    captured = datetime.fromtimestamp(msg["date"], tz=timezone.utc)
    voice = msg.get("voice")

    if voice:
        input_kind = "voice"
        body = msg.get("caption", "")   # 文字起こし待ち。空のまま置く
    elif msg.get("text", "").startswith("/"):
        # コマンドはメモとして保存しない
        api("sendMessage", chat_id=chat_id,
            text="テキストか音声をそのまま送ってください。分類やタグ付けは不要です。")
        return
    elif msg.get("text"):
        input_kind = "text"
        body = msg["text"]
        url = extract_url(body)
        if url:
            body = fetch_url_content(url)
    else:
        log.info("unsupported message type, skipped: %s", source_ref)
        return

    path = markdown_path(uid, captured)

    # ダウンロードはトランザクションの外でやる。中に入れると数十秒の
    # 通信のあいだ書き込みロックを握り続け、supervisor 側が
    # busy_timeout (5s) を超えて database is locked で落ちる
    blob = download_voice(voice["file_id"], uid) if voice else None

    with db:
        cur = db.execute(
            """
            INSERT INTO ideas(uid, body, captured_at, source, source_ref,
                              input_kind, parent_id, file_path)
            VALUES (?, ?, ?, 'telegram', ?, ?, ?, ?)
            -- 部分インデックスなので WHERE 句まで一致させないと解決されない
            ON CONFLICT(source, source_ref) WHERE source_ref IS NOT NULL DO NOTHING
            """,
            (
                uid,
                body,
                captured.strftime("%Y-%m-%dT%H:%M:%SZ"),
                source_ref,
                input_kind,
                resolve_parent(db, chat_id, msg),
                str(path.relative_to(REPO_ROOT)),
            ),
        )
        if cur.rowcount == 0:
            return
        idea_id = cur.lastrowid

        if blob:
            local, size, digest = blob
            db.execute(
                """
                INSERT INTO attachments(idea_id, kind, mime, bytes, file_path, meta_json)
                VALUES (?, 'voice', ?, ?, ?, ?)
                """,
                (
                    idea_id,
                    voice.get("mime_type"),
                    size,
                    str(local.relative_to(REPO_ROOT)),
                    json.dumps(
                        {"duration": voice.get("duration"), "sha256": digest},
                        ensure_ascii=False,
                    ),
                ),
            )

    write_idea_markdown(db, idea_id)
    log.info("ingested %s (%s)", uid, input_kind)


def handle_edit(db: sqlite3.Connection, msg: dict) -> None:
    """編集で body を差し替える。再埋め込みは content_hash 経由で自動。"""
    chat_id = msg["chat"]["id"]
    text = msg.get("text")
    if chat_id != CHAT_ID or not text:
        return

    row = db.execute(
        "SELECT id, uid FROM ideas WHERE source = 'telegram' AND source_ref = ?",
        (f"{chat_id}:{msg['message_id']}",),
    ).fetchone()
    if not row:
        return

    with db:
        db.execute(
            "UPDATE ideas SET body = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
            (text, row["id"]),
        )

    write_idea_markdown(db, row["id"])
    log.info("updated %s", row["uid"])


# ------------------------------------------------------------
# ボタン押下 -> 人間の判断を記録する
# ------------------------------------------------------------
def handle_callback(db: sqlite3.Connection, cq: dict) -> None:
    parts = (cq.get("data") or "").split(":")

    if parts[0] == "fnd" and len(parts) == 3:
        handle_finding_verdict(db, cq, int(parts[1]), parts[2])
        return

    if len(parts) != 5 or parts[0] != "rel":
        api("answerCallbackQuery", callback_query_id=cq["id"])
        return

    src_id, dst_id = int(parts[1]), int(parts[2])
    verdict = "confirmed" if parts[3] == "c" else "rejected"
    score = int(parts[4]) / 100.0

    # relations は CHECK(src_id < dst_id) の無向。ここで正規化する
    low, high = sorted((src_id, dst_id))

    with db:
        db.execute(
            """
            INSERT INTO relations(src_id, dst_id, kind, score, verdict)
            VALUES (?, ?, 'similar', ?, ?)
            ON CONFLICT(src_id, dst_id, kind)
            DO UPDATE SET verdict    = excluded.verdict,
                          score      = excluded.score,
                          decided_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            """,
            (low, high, score, verdict),
        )

    # 承認した関連は両方の Markdown に書き戻す
    if verdict == "confirmed":
        write_idea_markdown(db, low)
        write_idea_markdown(db, high)

    api(
        "answerCallbackQuery",
        callback_query_id=cq["id"],
        text="関連付けました" if verdict == "confirmed" else "却下しました",
    )
    log.info("relation %s-%s -> %s", low, high, verdict)


def handle_finding_verdict(db: sqlite3.Connection, cq: dict, finding_id: int,
                          code: str) -> None:
    """参考情報が役に立ったかを記録する。

    not_useful が溜まった検索クエリの種類は、後で調査プロンプトを
    見直す材料になる。
    """
    verdict = "useful" if code == "u" else "not_useful"
    with db:
        db.execute(
            "UPDATE findings SET verdict = ? WHERE id = ?", (verdict, finding_id)
        )

    # not_useful はテーマノートから消える。判断がそのまま見た目に出る
    row = db.execute(
        "SELECT cluster_id FROM findings WHERE id = ?", (finding_id,)
    ).fetchone()
    if row:
        write_theme_markdown(db, row["cluster_id"])

    api(
        "answerCallbackQuery",
        callback_query_id=cq["id"],
        text="記録しました" if verdict == "useful" else "不要として記録しました",
    )
    log.info("finding %s -> %s", finding_id, verdict)


def dispatch(db: sqlite3.Connection, update: dict) -> None:
    if "message" in update:
        handle_message(db, update["message"])
    elif "edited_message" in update:
        handle_edit(db, update["edited_message"])
    elif "callback_query" in update:
        handle_callback(db, update["callback_query"])


def handle_updates(db: sqlite3.Connection, updates: list[dict]) -> None:
    """1 件ずつ確定させる。途中で落ちても取りこぼさない。

    処理できない 1 通で offset を止めると、以降のメモが全部入らなくなる。
    通信起因なら次の周回に賭けて offset を据え置き、それ以外は
    諦めて先に進む。捨てたことは黙らずにチャットへ返す
    (メッセージ自体は Telegram に残るので送り直せる)。
    """
    for update in updates:
        try:
            dispatch(db, update)
        except urllib.error.URLError:
            raise                       # 一時的な障害。offset を進めない
        except Exception as exc:
            log.exception("update %s failed", update.get("update_id"))
            try:
                api("sendMessage", chat_id=CHAT_ID,
                    text=f"⚠️ 取り込めませんでした ({type(exc).__name__})。"
                         "送り直してください。")
            except Exception:
                log.warning("could not report ingest failure")

        with db:
            set_offset(db, update["update_id"] + 1)


def main() -> None:
    db = connect()
    log.info("polling as chat %s", CHAT_ID)

    while True:
        try:
            handle_updates(db, api(
                "getUpdates",
                offset=get_offset(db),
                timeout=POLL_TIMEOUT,
                allowed_updates=["message", "edited_message", "callback_query"],
            ))
        except urllib.error.URLError as exc:
            log.warning("network error: %s", exc)
            time.sleep(5)
        except Exception:
            log.exception("poll failed")
            time.sleep(5)


if __name__ == "__main__":
    main()
