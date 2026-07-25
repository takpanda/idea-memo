-- ============================================================
-- idea-memo Phase 1 schema (SQLite + sqlite-vec + FTS5)
--
-- 原則:
--   正本はこの DB。repo/ の Markdown は Obsidian で読むためのエクスポートで、
--   書き込みは常に DB -> Markdown の一方向 (common.write_idea_markdown)。
--   例外は clusters.name / name_locked だけで、テーマノートの front matter を
--   人が直したものを読み戻す (common.read_theme_overrides)。
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. アイデア本体
-- ------------------------------------------------------------
CREATE TABLE ideas (
  id          INTEGER PRIMARY KEY,
  uid         TEXT    NOT NULL UNIQUE,          -- ULID。Markdown ファイル名と対応
  body        TEXT    NOT NULL,                 -- 常に検索・埋め込み対象の正規テキスト
  summary     TEXT,                             -- Phase 2 で LLM が埋める
  captured_at TEXT    NOT NULL,                 -- 発想時刻 = 送信時刻 (ISO8601 UTC)
  created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),

  source      TEXT    NOT NULL,                 -- 'telegram' | 'shortcut' | 'web'
  source_ref  TEXT,                             -- 例: telegram の "chat_id:message_id"
  input_kind  TEXT    NOT NULL DEFAULT 'text',  -- 'text' | 'voice' | 'link'

  parent_id   INTEGER REFERENCES ideas(id) ON DELETE SET NULL,  -- 返信による追記
  status      TEXT    NOT NULL DEFAULT 'inbox', -- 'inbox' | 'kept' | 'archived'
  file_path   TEXT    NOT NULL,                 -- リポジトリ内の相対パス

  -- 類似メモの通知は副作用なので、冪等な reconciliation にするための目印。
  -- 候補ゼロでも埋めて、再走査を止める
  similar_notified_at TEXT,

  CHECK (source     IN ('telegram','shortcut','web')),
  CHECK (input_kind IN ('text','voice','link')),
  CHECK (status     IN ('inbox','kept','archived'))
);

-- 冪等性の要。再送・リトライ・getUpdates の重複配信で二重登録しない
CREATE UNIQUE INDEX idx_ideas_source_ref
  ON ideas(source, source_ref) WHERE source_ref IS NOT NULL;

CREATE INDEX idx_ideas_captured ON ideas(captured_at DESC);
CREATE INDEX idx_ideas_status   ON ideas(status, captured_at DESC);
CREATE INDEX idx_ideas_parent   ON ideas(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_ideas_unnotified ON ideas(captured_at) WHERE similar_notified_at IS NULL;

-- ------------------------------------------------------------
-- 2. 添付ファイル (音声メモ、画像)
--    body には文字起こし結果が入るので、ここは原本と再生成用の情報のみ
-- ------------------------------------------------------------
CREATE TABLE attachments (
  id         INTEGER PRIMARY KEY,
  idea_id    INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
  kind       TEXT    NOT NULL,      -- 'voice' | 'image' | 'file'
  mime       TEXT,
  bytes      INTEGER,
  file_path  TEXT    NOT NULL,      -- 実体の保存先 (Git LFS or 別ディレクトリ)
  transcript TEXT,                  -- 文字起こしの生出力 (body は編集される可能性がある)
  asr_model  TEXT,                  -- 再生成の判断用
  meta_json  TEXT,

  CHECK (kind IN ('voice','image','file'))
);

CREATE INDEX idx_attachments_idea ON attachments(idea_id);

-- ------------------------------------------------------------
-- 3. 埋め込みのメタデータ
--    ベクトル本体は vec_ideas。ここは「どのモデルで・どの本文から」の記録
-- ------------------------------------------------------------
CREATE TABLE embeddings (
  idea_id      INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
  model        TEXT    NOT NULL,    -- 例: 'cl-nagoya/ruri-v3-310m'
  dim          INTEGER NOT NULL,
  target       TEXT    NOT NULL DEFAULT 'body',  -- 'body' | 'summary'
  -- ruri-v3 の 1+3 プレフィックス。プレフィックスは mean pooling に含まれるので
  -- 同じ本文でも '' と 'トピック: ' では別ベクトルになる
  prefix       TEXT    NOT NULL DEFAULT '',      -- '' | 'トピック: ' | '検索文書: '
  content_hash TEXT    NOT NULL,    -- 本文の SHA-256。変わったら再計算
  created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),

  PRIMARY KEY (idea_id, model, target, prefix),
  CHECK (target IN ('body','summary')),
  CHECK (prefix IN ('', 'トピック: ', '検索クエリ: ', '検索文書: '))
);

-- 未計算・古いものを拾う reconciliation 用
CREATE INDEX idx_embeddings_model ON embeddings(model, created_at);

-- ------------------------------------------------------------
-- 4. ベクトル索引 (sqlite-vec)
--    次元はモデル固定。モデル移行時は新テーブルを作って切り替える
-- ------------------------------------------------------------
CREATE VIRTUAL TABLE vec_ideas_ruri_v3_310m USING vec0(
  idea_id   INTEGER PRIMARY KEY,
  embedding float[768] distance_metric=cosine
);

-- Phase 2 のクラスタリングは 'トピック: ' プレフィックス版を使う。
-- 同じ本文でも別ベクトルになるので、テーブルを分ける
CREATE VIRTUAL TABLE vec_ideas_ruri_v3_310m_topic USING vec0(
  idea_id   INTEGER PRIMARY KEY,
  embedding float[768] distance_metric=cosine
);

-- ------------------------------------------------------------
-- 5. 全文検索 (FTS5 external content)
--    trigram tokenizer は日本語を分かち書きなしで扱える。
--    ベクトル検索が弱い固有名詞・型番・URL の完全一致を補う
-- ------------------------------------------------------------
CREATE VIRTUAL TABLE fts_ideas USING fts5(
  body,
  content='ideas',
  content_rowid='id',
  tokenize='trigram'
);

CREATE TRIGGER ideas_ai AFTER INSERT ON ideas BEGIN
  INSERT INTO fts_ideas(rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER ideas_ad AFTER DELETE ON ideas BEGIN
  INSERT INTO fts_ideas(fts_ideas, rowid, body) VALUES ('delete', old.id, old.body);
END;

CREATE TRIGGER ideas_au AFTER UPDATE OF body ON ideas BEGIN
  INSERT INTO fts_ideas(fts_ideas, rowid, body) VALUES ('delete', old.id, old.body);
  INSERT INTO fts_ideas(rowid, body) VALUES (new.id, new.body);
END;

-- ------------------------------------------------------------
-- 6. 関連リンク
--    「提案」は保存しない (毎回 ANN で出せる)。
--    人間が承認/却下した判断だけを永続化する
-- ------------------------------------------------------------
CREATE TABLE relations (
  id         INTEGER PRIMARY KEY,
  src_id     INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
  dst_id     INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
  kind       TEXT    NOT NULL,     -- 'similar' | 'duplicate'
  score      REAL,                 -- 提案時のスコア (後の閾値チューニング用)
  verdict    TEXT    NOT NULL,     -- 'confirmed' | 'rejected'
  decided_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),

  UNIQUE (src_id, dst_id, kind),
  CHECK (src_id < dst_id),         -- 無向。アプリ側で min/max に正規化して入れる
  CHECK (kind    IN ('similar','duplicate')),
  CHECK (verdict IN ('confirmed','rejected'))
);

CREATE INDEX idx_relations_dst ON relations(dst_id);

-- ------------------------------------------------------------
-- 8. ランタイム状態 (Telegram の getUpdates offset など)
-- ------------------------------------------------------------
CREATE TABLE kv (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ============================================================
-- Phase 2: クラスタリングとテーマノート
-- ============================================================

-- ------------------------------------------------------------
-- 9. クラスタ
--    HDBSCAN は毎回ラベルを振り直すので、そのまま使うと ID が安定しない。
--    前回との Jaccard 係数でマッチングして、この id を引き継ぐ
-- ------------------------------------------------------------
CREATE TABLE clusters (
  id                 INTEGER PRIMARY KEY,
  uid                TEXT    NOT NULL UNIQUE,
  name               TEXT,                        -- LLM または人が付けたテーマ名
  name_locked        INTEGER NOT NULL DEFAULT 0,  -- 1 なら LLM は上書きしない
  summary            TEXT,
  size               INTEGER NOT NULL DEFAULT 0,
  member_hash        TEXT,                        -- 現在のメンバー集合の SHA-256
  named_member_hash  TEXT,                        -- 命名・要約したときのメンバー集合
  researched_at         TEXT,                     -- 最後に調査した日時 (Phase 3)
  researched_member_hash TEXT,                    -- 調査したときのメンバー集合
  file_path          TEXT,                        -- themes/<uid>.md
  created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  closed_at          TEXT,                        -- 消滅したクラスタも履歴として残す

  CHECK (name_locked IN (0,1))
);

CREATE INDEX idx_clusters_open ON clusters(updated_at) WHERE closed_at IS NULL;

-- ------------------------------------------------------------
-- 10. メモとクラスタの割当
--     HDBSCAN はハード割当なので 1 メモ 1 クラスタ。
--     ノイズ (label = -1) は行を作らない = まだどのテーマにも属さない
-- ------------------------------------------------------------
CREATE TABLE idea_clusters (
  idea_id     INTEGER PRIMARY KEY REFERENCES ideas(id) ON DELETE CASCADE,
  cluster_id  INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
  probability REAL,                               -- HDBSCAN の所属確率
  assigned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX idx_idea_clusters_cluster ON idea_clusters(cluster_id);

-- ------------------------------------------------------------
-- 11. 再構成の記録 (パラメータ調整の材料)
-- ------------------------------------------------------------
CREATE TABLE cluster_runs (
  id          INTEGER PRIMARY KEY,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  n_ideas     INTEGER,
  n_clusters  INTEGER,
  n_noise     INTEGER,
  n_new       INTEGER,
  n_closed    INTEGER,
  params_json TEXT
);

-- ------------------------------------------------------------
-- 7. 参考: ハイブリッド検索 (RRF で統合)
-- ------------------------------------------------------------
--
-- WITH knn AS (
--   SELECT idea_id, ROW_NUMBER() OVER (ORDER BY distance) AS rnk
--   FROM   vec_ideas_ruri_v3_310m
--   WHERE  embedding MATCH :qvec AND k = 20
-- ),
-- kw AS (
--   SELECT rowid AS idea_id,
--          ROW_NUMBER() OVER (ORDER BY bm25(fts_ideas)) AS rnk
--   FROM   fts_ideas
--   WHERE  fts_ideas MATCH :qtext
--   LIMIT  20
-- )
-- SELECT i.uid, i.body, i.captured_at,
--        COALESCE(1.0/(60+knn.rnk), 0) + COALESCE(1.0/(60+kw.rnk), 0) AS score
-- FROM   ideas i
--        LEFT JOIN knn ON knn.idea_id = i.id
--        LEFT JOIN kw  ON kw.idea_id  = i.id
-- WHERE  (knn.idea_id IS NOT NULL OR kw.idea_id IS NOT NULL)
--        AND i.id != :self_id
--        AND NOT EXISTS (
--          SELECT 1 FROM relations r
--          WHERE  r.verdict = 'rejected'
--            AND  r.src_id = MIN(i.id, :self_id)
--            AND  r.dst_id = MAX(i.id, :self_id)
--        )
-- ORDER BY score DESC
-- LIMIT 5;

-- ============================================================
-- Phase 3: 調査エージェント
-- ============================================================

-- ------------------------------------------------------------
-- 12. 収集した参考情報
--     賛成材料だけ集めても価値が薄いので、query_kind で
--     先行事例 / 裏付け / 反証 を明示的に撃ち分ける
-- ------------------------------------------------------------
CREATE TABLE findings (
  id           INTEGER PRIMARY KEY,
  cluster_id   INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
  query_kind   TEXT    NOT NULL,   -- 'prior_art' | 'evidence' | 'counter'
  query        TEXT    NOT NULL,
  url          TEXT    NOT NULL,
  title        TEXT,
  site         TEXT,
  summary      TEXT,
  stance       TEXT,               -- このテーマに対して supports / challenges / neutral
  fetched_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  notified_at  TEXT,
  verdict      TEXT,               -- NULL のうちは未判断。人が押したら記録

  UNIQUE (cluster_id, url),        -- 同じテーマに同じ URL を二度入れない
  CHECK (query_kind IN ('prior_art','evidence','counter')),
  CHECK (stance  IS NULL OR stance  IN ('supports','challenges','neutral')),
  CHECK (verdict IS NULL OR verdict IN ('useful','not_useful'))
);

CREATE INDEX idx_findings_cluster ON findings(cluster_id, fetched_at DESC);
CREATE INDEX idx_findings_unnotified ON findings(fetched_at) WHERE notified_at IS NULL;
