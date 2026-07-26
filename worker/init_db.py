"""
DB 初期化 (コンテナ内で一度だけ実行する)

vec0 と fts5 の仮想テーブルは拡張が要るので、ホストの sqlite3 コマンドでは
schema.sql を流せない。ここで前提条件も検証しておく。

  docker compose run --rm ingest python init_db.py
"""

import os
import sqlite3
import sys

import sqlite_vec

DB_PATH = os.environ.get("IDEA_DB", "/data/idea.db")
SCHEMA = os.environ.get("IDEA_SCHEMA", "/app/schema.sql")


def main() -> int:
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    version = db.execute("SELECT sqlite_version()").fetchone()[0]
    options = {r[0] for r in db.execute("PRAGMA compile_options")}
    print(f"sqlite {version}")

    # trigram tokenizer は 3.34 以降。Debian bookworm 以降なら満たす
    if tuple(int(x) for x in version.split(".")[:2]) < (3, 34):
        print("sqlite too old for trigram tokenizer", file=sys.stderr)
        return 1
    if not any("FTS5" in opt for opt in options):
        print("sqlite built without FTS5", file=sys.stderr)
        return 1

    if os.path.exists(DB_PATH) and db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='clusters'"
    ).fetchone():
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(clusters)")
        }
        if "members_json" not in columns:
            # Phase 3 で追加された列だけを明示的に移行する。既存データは
            # NULL のまま保持し、SQLite の ALTER TABLE を同一トランザクション
            # で確定する。新規 DB は下の schema.sql が列を作る。
            db.execute("ALTER TABLE clusters ADD COLUMN members_json TEXT")
            db.commit()
            print(f"migrated clusters.members_json: {DB_PATH}")
        else:
            print(f"already up to date: {DB_PATH}")
        db.close()
        return 0

    db.executescript(open(SCHEMA, encoding="utf-8").read())
    db.commit()

    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    print(f"created {len(tables)} tables: {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
