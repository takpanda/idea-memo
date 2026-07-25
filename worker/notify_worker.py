"""
通知ワーカー (Raspberry Pi 常駐)

人間の判断が要るものを Telegram に投げ、inline keyboard で受ける。
押された結果は telegram_ingest 側の callback ハンドラが記録する。

  1. 類似メモの候補   -> relations に confirmed / rejected
  2. 集まった参考情報 -> findings.verdict に useful / not_useful

類似検出には FTS を混ぜない。全文検索は「ユーザーが明示的に検索したとき」
のためのもので、意味的な近さの自動検出は純粋なベクトル検索の方が効く。

  python notify_worker.py
"""

import logging
import os
import time
import urllib.error

from common import CHAT_ID, api, connect

MODEL = "cl-nagoya/ruri-v3-310m"
VEC_TABLE = "vec_ideas_ruri_v3_310m"

# ruri-v3 は無関係な文でも 0.7 前後は出る。ここは実データを見て要調整
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.82"))
TOP_K = 3
CANDIDATE_K = 12          # 除外される分を見越して多めに引く
SNIPPET_LEN = 60
INTERVAL_SEC = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("notify-worker")

MARKS = ["①", "②", "③"]
STANCE_MARK = {"supports": "◯", "challenges": "▲", "neutral": "・"}


def snippet(body: str, length: int = SNIPPET_LEN) -> str:
    flat = " ".join(body.split())
    return flat[:length] + ("…" if len(flat) > length else "")


# ------------------------------------------------------------
# 1. 類似メモ
# ------------------------------------------------------------
def pending_ideas(db, limit: int = 5):
    return db.execute(
        """
        SELECT i.id, i.uid, i.body
        FROM   ideas i
               JOIN embeddings e
                 ON e.idea_id = i.id
                AND e.model = ? AND e.target = 'body' AND e.prefix = ''
        WHERE  i.similar_notified_at IS NULL
               AND i.status != 'archived'
        ORDER BY i.captured_at
        LIMIT ?
        """,
        (MODEL, limit),
    ).fetchall()


def excluded_ids(db, idea_id: int) -> set[int]:
    """再提示しないもの: 既に判断済み、親子関係、自分自身。"""
    ids = {idea_id}
    ids |= {
        row["other"]
        for row in db.execute(
            """
            SELECT CASE WHEN src_id = ? THEN dst_id ELSE src_id END AS other
            FROM   relations WHERE src_id = ? OR dst_id = ?
            """,
            (idea_id, idea_id, idea_id),
        )
    }
    # 返信でぶら下げたメモは似ていて当然なので通知しない
    ids |= {
        row["id"]
        for row in db.execute(
            "SELECT id FROM ideas WHERE parent_id = ? "
            "UNION SELECT parent_id AS id FROM ideas WHERE id = ? AND parent_id IS NOT NULL",
            (idea_id, idea_id),
        )
    }
    return {i for i in ids if i is not None}


def find_similar(db, idea_id: int) -> list[dict]:
    own = db.execute(
        f"SELECT embedding FROM {VEC_TABLE} WHERE idea_id = ?", (idea_id,)
    ).fetchone()
    if own is None:
        return []

    # KNN は単独クエリで回す (vec0 の MATCH は JOIN と併用しにくい)
    knn = db.execute(
        f"SELECT idea_id, distance FROM {VEC_TABLE} WHERE embedding MATCH ? AND k = ?",
        (own["embedding"], CANDIDATE_K),
    ).fetchall()

    skip = excluded_ids(db, idea_id)
    results = []
    for row in knn:
        if row["idea_id"] in skip:
            continue
        # distance_metric=cosine なので distance = 1 - cos類似度
        score = 1.0 - row["distance"]
        if score < SIM_THRESHOLD:
            continue
        meta = db.execute(
            "SELECT id, uid, body, captured_at FROM ideas "
            "WHERE id = ? AND status != 'archived'",
            (row["idea_id"],),
        ).fetchone()
        if meta is None or not meta["body"]:
            continue
        results.append({**dict(meta), "score": score})
        if len(results) >= TOP_K:
            break
    return results


def notify_similar(idea, matches: list[dict]) -> None:
    lines = [f"💡 似ているメモが {len(matches)} 件あります", ""]
    keyboard = []
    for mark, match in zip(MARKS, matches):
        lines.append(f"{mark} {match['captured_at'][:10]} · {match['score']:.0%}")
        lines.append(snippet(match["body"]))
        lines.append("")
        base = f"rel:{idea['id']}:{match['id']}"
        pct = round(match["score"] * 100)
        keyboard.append(
            [
                {"text": f"{mark} 関連", "callback_data": f"{base}:c:{pct}"},
                {"text": f"{mark} 違う", "callback_data": f"{base}:x:{pct}"},
            ]
        )

    api(
        "sendMessage",
        chat_id=CHAT_ID,
        text="\n".join(lines).strip(),
        reply_markup={"inline_keyboard": keyboard},
    )


def tick_ideas(db) -> int:
    rows = pending_ideas(db)
    for idea in rows:
        matches = find_similar(db, idea["id"])
        if matches:
            notify_similar(idea, matches)
            log.info("notified %s (%d matches)", idea["uid"], len(matches))
        # 候補ゼロでも必ず埋める。埋めないと毎周回スキャンし続ける
        with db:
            db.execute(
                "UPDATE ideas SET similar_notified_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (idea["id"],),
            )
    return len(rows)


# ------------------------------------------------------------
# 2. 参考情報
# ------------------------------------------------------------
def pending_findings(db, limit: int = 3):
    return db.execute(
        """
        SELECT f.id, f.title, f.url, f.site, f.summary, f.stance,
               c.name AS cluster_name
        FROM   findings f JOIN clusters c ON c.id = f.cluster_id
        WHERE  f.notified_at IS NULL
        ORDER BY f.stance = 'challenges' DESC, f.fetched_at
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def notify_finding(finding) -> None:
    mark = STANCE_MARK.get(finding["stance"], "・")
    text = "\n".join(
        [
            f"{mark} 「{finding['cluster_name']}」の参考情報",
            "",
            finding["title"] or finding["url"],
            f"({finding['site']})",
            "",
            finding["summary"] or "",
            "",
            finding["url"],
        ]
    )
    api(
        "sendMessage",
        chat_id=CHAT_ID,
        text=text,
        disable_web_page_preview=True,
        reply_markup={
            "inline_keyboard": [
                [
                    {"text": "役に立った", "callback_data": f"fnd:{finding['id']}:u"},
                    {"text": "不要", "callback_data": f"fnd:{finding['id']}:n"},
                ]
            ]
        },
    )


def tick_findings(db) -> int:
    rows = pending_findings(db)
    for finding in rows:
        notify_finding(finding)
        with db:
            db.execute(
                "UPDATE findings SET notified_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (finding["id"],),
            )
        log.info("notified finding %s", finding["id"])
        time.sleep(1)          # 一度に連投しない
    return len(rows)


def main() -> None:
    db = connect(with_vec=True)
    while True:
        try:
            while tick_ideas(db) > 0:
                pass
            while tick_findings(db) > 0:
                pass
        except urllib.error.URLError as exc:
            log.warning("telegram unreachable: %s", exc)
        except Exception:
            log.exception("tick failed")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
