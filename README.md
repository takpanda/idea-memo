# idea-memo

日頃思いついたアイデアを Telegram に投げるだけで、自動的に整理され、関連情報が集まってくる個人用ツール。

## 要件

- 日頃思いついたアイデアを簡単に記録できる（Telegram、テキスト/音声）
- 記録された内容は自動で体系的に整理される（埋め込み + HDBSCAN クラスタリング）
- 関連する情報を自動で収集して、アイデアにリンクされる（テーマ単位の調査エージェント）

## 構成

```
Pi (Docker)   ingest / supervisor / web / scheduler
Mac mini      embed_server:7997 (ruri-v3-310m) / transcribe_server:7998 (Whisper)
GPU ノード     vLLM (テーマ命名・要約・調査クエリ生成、Docker 対象外)
```

Mac mini の 2 サービスは MPS / MLX を使うため Docker 化していない（Docker Desktop for Mac は
Metal が見えないコンテナ内 VM で動くため）。launchd で常駐させる。

閲覧は Obsidian で `repo/` を vault として開く。Web UI は DB を引かないとできない検索と
クラスタ俯瞰だけを持つ。

```
worker/    Pi で Docker 常駐させるワーカー・バッチ・スキーマ
macmini/   Mac mini で launchd 常駐させる embed/transcribe サーバーと plist
```

## セットアップ

### Pi 側

```bash
cp .env.example .env   # トークン・chat_id・Mac mini の IP を埋める
mkdir -p data repo logs

docker compose build
docker compose run --rm ingest python init_db.py
docker compose up -d
docker compose logs -f ingest
```

### Mac mini 側

```bash
uv pip install "transformers>=4.48.0" sentence-transformers sentencepiece torch \
               fastapi "uvicorn[standard]" mlx-whisper python-multipart

cp macmini/local.idea-memo.embed.plist macmini/local.idea-memo.transcribe.plist ~/Library/LaunchAgents/
# 各 plist の USERNAME と WorkingDirectory を置き換える
launchctl load -w ~/Library/LaunchAgents/local.idea-memo.embed.plist
launchctl load -w ~/Library/LaunchAgents/local.idea-memo.transcribe.plist

sudo pmset -a sleep 0 disksleep 0   # スリープで文字起こしが止まらないように
```

### 動作確認

```bash
docker compose run --rm ingest python -c \
  "import urllib.request,json; print(json.load(urllib.request.urlopen('http://mac-mini.local:7997/health')))"
# {"status":"ok","model":"cl-nagoya/ruri-v3-310m","device":"mps","dim":768,...}
```

### テスト

Phase 1 の通し試験。Mac mini の 2 サービスと Telegram の file API は
`tests/fake_services.py` が同じ HTTP 契約で代役を務めるので、実機もモデルも要らない。

```bash
pip install sqlite-vec
python -m unittest discover -s tests
```

## Phase

1. **Phase 1** — Telegram 取り込み・音声文字起こし・埋め込み・類似メモ通知（完了）
2. **Phase 2** — HDBSCAN によるクラスタリング（ID 継承あり）・LLM によるテーマ命名
3. **Phase 3** — テーマ単位の調査エージェント（先行事例・裏付け・反証を撃ち分け）・週次ダイジェスト

## 主な設計判断

- 正本は DB、Markdown は再生成可能なエクスポート。DB が壊れても Markdown から復旧できる
- 埋め込みモデルは日本語特化の `cl-nagoya/ruri-v3-310m`（768次元）。Ollama ではなく
  自前の FastAPI サーバーで持つ（プレフィックス方式を正確に制御するため）
- 類似メモの提示・参考情報の要否は「提案」に留め、人間の承認/却下だけを `relations` /
  `findings.verdict` に永続化する
- status の自動 archive はしない。手動判断のみ
- reconciliation（文字起こし・埋め込み・通知・命名）はキューを持たず、欠損を検出して
  埋める設計。ワーカーが落ちても次の周回で追いつく

## チューニング対象

`.env` の `SIM_THRESHOLD` / `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` / `INHERIT_THRESHOLD` は
実データを見てから調整する前提の初期値。
