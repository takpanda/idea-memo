"""
文字起こし reconciliation ワーカー (Raspberry Pi 常駐)

embed_worker と同じ考え方で、「transcript が NULL の音声添付」を毎周回で拾う。
body を埋めると content_hash が変わるので、再埋め込みは embed_worker が
自動的に拾い直す。ワーカー間に依存関係もキューも要らない。

  python transcribe_worker.py
"""

import json
import logging
import mimetypes
import os
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from common import REPO_ROOT, connect

TRANSCRIBE_URL = os.environ.get("TRANSCRIBE_URL", "http://mac-mini.local:7998/transcribe")

ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
INTERVAL_SEC = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcribe-worker")


def fetch_pending(db: sqlite3.Connection, limit: int = 4) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT a.id AS attachment_id, a.idea_id, a.file_path, a.mime
        FROM   attachments a
               JOIN ideas i ON i.id = a.idea_id
        WHERE  a.kind = 'voice'
               AND a.transcript IS NULL
               AND i.status != 'archived'
        ORDER BY a.id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def post_audio(path: Path, language: str = "ja") -> str:
    """multipart/form-data を標準ライブラリだけで組む。"""
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    sep = f"--{boundary}".encode()

    body = b"\r\n".join(
        [
            sep,
            b'Content-Disposition: form-data; name="language"',
            b"",
            language.encode(),
            sep,
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"'.encode(),
            f"Content-Type: {mime}".encode(),
            b"",
            path.read_bytes(),
            f"--{boundary}--".encode(),
            b"",
        ]
    )

    req = urllib.request.Request(
        TRANSCRIBE_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        return json.loads(res.read())["text"]


def store(db: sqlite3.Connection, row: sqlite3.Row, text: str) -> None:
    with db:
        db.execute(
            "UPDATE attachments SET transcript = ?, asr_model = ? WHERE id = ?",
            (text, ASR_MODEL, row["attachment_id"]),
        )
        # caption が既に入っている場合は残して追記する
        db.execute(
            """
            UPDATE ideas
            SET    body = CASE WHEN body = '' THEN ? ELSE body || char(10) || char(10) || ? END,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE  id = ?
            """,
            (text, text, row["idea_id"]),
        )


def tick(db: sqlite3.Connection) -> int:
    rows = fetch_pending(db)
    if not rows:
        return 0
    for row in rows:
        path = REPO_ROOT / row["file_path"]
        if not path.exists():
            log.error("audio file missing: %s", path)
            continue
        text = post_audio(path)
        if not text:
            log.warning("empty transcript for idea %s", row["idea_id"])
            continue
        store(db, row, text)
        log.info("transcribed idea %s (%d chars)", row["idea_id"], len(text))
    return len(rows)


def main() -> None:
    db = connect()
    while True:
        try:
            while tick(db) > 0:
                pass
        except urllib.error.URLError as exc:
            # Mac mini が寝ていても投稿は成功済み。次の周回で追いつく
            log.warning("transcription service unreachable: %s", exc)
        except Exception:
            log.exception("tick failed")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
