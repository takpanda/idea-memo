# idea-memo

日頃思いついたアイデアを Telegram に投げるだけで、自動的に整理され、関連情報が集まってくる個人用ツール。

## 要件

- 日頃思いついたアイデアを簡単に記録できる（Telegram、テキスト/音声）
- 記録された内容は自動で体系的に整理される（埋め込み + HDBSCAN クラスタリング）
- 関連する情報を自動で収集して、アイデアにリンクされる（テーマ単位の調査エージェント）

## 構成

```
Pi (Docker)   ingest / supervisor / web / scheduler
Mac-mini-M4Pro    embed_server:7997 (ruri-v3-310m) / transcribe_server:7998 (Whisper)
GPU ノード     vLLM (テーマ命名・要約・調査クエリ生成、Docker 対象外)
```

Mac-mini-M4Pro の 2 サービスは MPS / MLX を使うため Docker 化していない（Docker Desktop for Mac は
Metal が見えないコンテナ内 VM で動くため）。launchd で常駐させる。

腰を据えて読み書きするときは Obsidian で `repo/` を vault として開く。外出先の iPhone や
手元にブラウザしかない端末からは Web UI（`http://<Pi>:8080`）で読む。

```
worker/    Pi で Docker 常駐させるワーカー・バッチ・スキーマ
macmini/   Mac-mini-M4Pro で launchd 常駐させる embed/transcribe サーバーと plist
tests/     外部 (Telegram / LLM / 検索 / 埋め込み) だけを差し替えた通し試験
```

vault 側 (`repo/`) の中身:

```
ideas/YYYY/MM/  メモ 1 件 1 ファイル
themes/         クラスタごとのテーマノート。集めた参考情報もここに載る
digests/        週次ダイジェスト (YYYY-Www.md)
```

### Web UI

ビルドステップなしの HTML 1 枚。iPhone の画面幅とダークモードに合わせてある。

- **メモ** — 新しい順の一覧。status（inbox / 残す / archive）とテーマで絞り込み、
  下端の「もっと読む」で次の 30 件。タップで本文全文・返信の連なり・承認済みの関連メモ
- **検索** — ベクトル + 全文のハイブリッド。ヒット箇所を強調し、抜粋はヒット位置に寄せる
- **テーマ** — テーマ一覧 → 中身（要約・参考情報・メンバー）。未分類のメモもここから辿れる
- status の変更はここからできる（自動 archive はしない方針なので、Telegram 以外の唯一の導線）

画面は `#/ideas/<uid>` のような URL を持つので、iPhone の戻るジェスチャがそのまま効き、
検索結果の URL をそのまま開き直せる。

**認証は持たない。** LAN の外から使うなら Tailscale などの上に置く（そのぶんの追加インフラは
Pi 側には要らない）。ポートをそのままインターネットに晒さないこと。

## セットアップ

### Pi 側

```bash
cp .env.example .env   # トークン・chat_id・Mac-mini-M4Pro の IP を埋める
mkdir -p data repo logs

docker compose build
docker compose run --rm ingest python init_db.py
docker compose up -d
docker compose logs -f ingest
```

### Mac-mini-M4Pro 側

```bash
uv pip install "transformers>=4.48.0" sentence-transformers sentencepiece torch \
               fastapi "uvicorn[standard]" mlx-whisper python-multipart

cp macmini/local.idea-memo.embed.plist macmini/local.idea-memo.transcribe.plist ~/Library/LaunchAgents/
# 各 plist の USERNAME と WorkingDirectory を置き換える
launchctl load -w ~/Library/LaunchAgents/local.idea-memo.embed.plist
launchctl load -w ~/Library/LaunchAgents/local.idea-memo.transcribe.plist

sudo pmset -a sleep 0 disksleep 0   # スリープで文字起こしが止まらないように
```

### 検索バックエンド (Phase 3)

調査エージェントは Brave Search API か、自前の SearXNG を使う。
`.env` の `SEARCH_BACKEND` で切り替える。

```bash
# Brave (既定)。無料枠でも週 1 回のバッチなら足りる
SEARCH_BACKEND=brave BRAVE_API_KEY=...

# 自前で持つ場合
docker compose --profile searxng up -d
# SEARCH_BACKEND=searxng SEARXNG_URL=http://searxng:8080
```

バッチは `worker/crontab` で回している (クラスタ再構成 毎日 4:00 /
調査 日曜 3:00 / ダイジェスト 日曜 8:00)。手で回すなら:

```bash
docker compose run --rm ingest python research_worker.py
```

### LLM の切り替え (Ollama / vLLM)

テーマ生成と調査ワーカーは、`.env` の `LLM_BASE_URL` と `LLM_MODEL` を共通で参照する。
Mac-mini-M4Pro (192.168.1.102) の Ollama を使う例:

```bash
# Mac-mini-M4Pro 上で Ollama をインストール・常駐させる
ollama serve
ollama pull qwen2.5:7b

# Pi 側の .env
LLM_BASE_URL=http://192.168.1.102:11434/v1
LLM_MODEL=qwen2.5:7b

# Pi から OpenAI 互換 API を確認（本文は出力しない）
docker compose run --rm ingest python check_llm.py
```

Ollama をLANから受け付けるには、Mac-mini-M4Pro上で `OLLAMA_HOST=0.0.0.0:11434`
を設定してOllamaを再起動し、macOSファイアウォールでTCP 11434を許可する。
実際の常駐設定は利用中のOllamaインストール方法に合わせること。

テーマ生成の一次確認は、未命名またはメンバー変更済みのクラスタを用意したうえで、
次の1パス実行を使う。`supervisor.py --once` は常駐せず、処理後に終了する。

```bash
started=$(date +%s)
docker compose run --rm supervisor python supervisor.py --once
test "$(find repo/themes -type f -newermt "@$started" | wc -l)" -ge 1
```

終了条件は、コマンドの終了コードが0で、`repo/themes/` に実行開始後に更新された
Markdownが1件以上あること。調査ワーカーは `docker compose run --rm ingest python
research_worker.py` が終了コード0で完了することを確認する。

DGX Sparkへ戻す場合は `.env` を `LLM_BASE_URL=http://gpu-node:8000/v1` と
`LLM_MODEL=deepseek-v4-flash` に戻し、`docker compose up -d --force-recreate supervisor scheduler`
（または対象ワーカーの再起動）を実行する。Ollama側のモデル削除・停止は不要。

### 動作確認

```bash
docker compose run --rm ingest python -c \
  "import urllib.request,json; print(json.load(urllib.request.urlopen('http://mac-mini.local:7997/health')))"
# {"status":"ok","model":"cl-nagoya/ruri-v3-310m","device":"mps","dim":768,...}
```

### テスト

Phase 1 / Phase 2 / Web UI の通し試験。Mac-mini-M4Pro の 2 サービス、GPU ノードの LLM、Telegram の
file API は `tests/fake_services.py` が同じ HTTP 契約で代役を務めるので、
実機もモデルも要らない (検索バックエンドだけはモジュール関数を直接差し替える)。

```bash
pip install sqlite-vec numpy scikit-learn fastapi httpx
python -m unittest discover -s tests
```

## Phase

1. **Phase 1** — Telegram 取り込み・音声文字起こし・埋め込み・類似メモ通知（完了）
2. **Phase 2** — HDBSCAN によるクラスタリング（ID 継承あり）・LLM によるテーマ命名（完了）
3. **Phase 3** — テーマ単位の調査エージェント（先行事例・裏付け・反証を撃ち分け）・週次ダイジェスト（完了）

テーマは **話題が 2 つ以上に分かれてから** 出る。HDBSCAN は木の根をクラスタとして
選ばないため、似たメモばかりが数十件溜まっていても 1 話題のうちは全部「未分類」になる。

## 主な設計判断

- 正本は DB、Markdown は再生成可能なエクスポート。DB が壊れても Markdown から復旧できる
- 埋め込みモデルは日本語特化の `cl-nagoya/ruri-v3-310m`（768次元）。Ollama ではなく
  自前の FastAPI サーバーで持つ（プレフィックス方式を正確に制御するため）
- 類似メモの提示・参考情報の要否は「提案」に留め、人間の承認/却下だけを `relations` /
  `findings.verdict` に永続化する
- テーマ名は初回だけ LLM に付けさせ、要約だけ作り直す。テーマノートの front matter で
  `name_locked: true` にすれば、その名前が LLM より強い
- HDBSCAN のラベルは毎回振り直されるので、メンバー集合の Jaccard 係数で前回の
  クラスタ ID を引き継ぐ。消えたクラスタは閉じるだけで消さず、同じ話題が戻れば
  同じテーマに戻る
- status の自動 archive はしない。手動判断のみ
- Web UI からの書き込みは status の変更だけ。本文編集を入れると Markdown と DB の
  双方向同期が要り、「正本は DB、Markdown は再生成」が崩れる
- reconciliation（文字起こし・埋め込み・通知・命名）はキューを持たず、欠損を検出して
  埋める設計。ワーカーが落ちても次の周回で追いつく

## チューニング対象

`.env` の `SIM_THRESHOLD` / `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` / `INHERIT_THRESHOLD` は
実データを見てから調整する前提の初期値。
