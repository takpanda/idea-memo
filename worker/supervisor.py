"""
reconciliation スーパーバイザ (Raspberry Pi 常駐)

文字起こし・埋め込み・類似通知・テーマ生成は、どれも
「欠けている派生データを探して埋める」という同じ形をしている。
依存も DB も障害許容の考え方も同じなので、1 プロセスにまとめる。

分けていたときの実害:
  - Python プロセス 6 本で ~500MB。1 本なら ~120MB
  - ボイスメモが 文字起こし -> 埋め込み -> 通知 を抜けるのに最大 3 分
    (各ワーカーが次のポーリングまで相手の出力に気づかない)
  - ログを 6 箇所見に行くことになる

依存順に 1 パスで流すので、ボイスメモも 1 周で通知まで到達する。
各ワーカーの main() は残してあるので、切り分けたいときは単体でも動く。

long polling の telegram_ingest と HTTP サーバーの web は形が違うので
別プロセスのまま。

  python supervisor.py           # 常駐
  python supervisor.py --once    # 1 パスだけ流して終了 (デバッグ用)
"""

import logging
import os
import sys
import time
import urllib.error

import embed_worker
import notify_worker
import theme_writer
import transcribe_worker
from common import connect

INTERVAL_SEC = int(os.environ.get("SUPERVISOR_INTERVAL", "60"))
# 溜まった分は続けて消化するが、暴走は止める
MAX_PASSES = int(os.environ.get("SUPERVISOR_MAX_PASSES", "20"))

# 依存順。前段の出力が後段の入力になる
STAGES = [
    ("transcribe", transcribe_worker.tick),
    ("embed", embed_worker.tick),
    ("similar", notify_worker.tick_ideas),
    ("findings", notify_worker.tick_findings),
    ("theme", theme_writer.tick),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("supervisor")


def run_pass(db) -> int:
    """1 段が落ちても後続は流す。次の周回で拾い直せる設計なので止めない。"""
    done = 0
    for name, tick in STAGES:
        try:
            n = tick(db)
            if n:
                log.info("%s: %d", name, n)
            done += n
        except urllib.error.URLError as exc:
            # Mac mini や GPU ノードが寝ているだけ。警告にとどめる
            log.warning("%s: upstream unreachable (%s)", name, exc)
        except Exception:
            log.exception("%s: failed", name)
    return done


def main() -> None:
    db = connect(with_vec=True)
    once = "--once" in sys.argv

    log.info("stages: %s", ", ".join(name for name, _ in STAGES))

    while True:
        passes = 0
        while run_pass(db) > 0 and passes < MAX_PASSES:
            passes += 1
        if passes >= MAX_PASSES:
            log.warning("hit MAX_PASSES; backlog may be stuck")
        if once:
            return
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
