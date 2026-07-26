# idea-memo 設計ドキュメント

日頃思いついたアイデアを記録すると、自動で体系的に整理され、関連情報が集まってくる個人用ツールの検討記録。

## 要件

1. 日頃思いついたアイデアを簡単に記録できる
2. 記録された内容は自動で体系的に整理される
3. 関連する情報を自動で収集して、アイデアにリンクされる

## 全体構成

```
Pi (Docker)   ingest / supervisor / web / scheduler
Mac mini      embed_server:7997 (ruri-v3-310m) / transcribe_server:7998 (Whisper) — launchd
GPU ノード     vLLM (テーマ命名・要約・調査クエリ生成、Docker 対象外)
LLM 以外       Docker Compose。Mac mini の 2 サービスは適材適所で launchd ネイティブ
```

「適材適所で」という前提のもと、Docker 化は Pi 側のみ。Mac mini は MPS/MLX を活かすため
Docker 化していない。腰を据えた閲覧と編集は Obsidian、判断は Telegram、
外出先からの閲覧・検索・テーマ俯瞰は自前の Web UI。

---

## Phase 1: 記録

### 捕捉チャネル: Telegram

Slack ではなく Telegram を選定。

- long polling (`getUpdates`) で完結し、Pi を外部公開しなくてよい
- ボイスメッセージが素直（`.ogg` → `getFile` → Whisper に直結できる）
- inline keyboard が使え、「このメモを既存テーマに統合しますか？」のような人間の判断を
  ボタン一つで受けられる
- 返信 (reply) で親子関係、編集で `edited_message` イベントが取れる

LINE は公開 HTTPS Webhook が必須で Pi を外に出す必要があるため見送り。

### データモデル

正本は **DB**（SQLite）、Markdown は **再生成可能なエクスポート**。
（当初「正本は Markdown」としていたが、実装が進む中で「DB が先、Markdown は再生成」が
実態に近いと判明し、この表現に修正した。ただし Markdown 側に導出不能な項目
〔原文・投稿時刻・承認済み関連〕は全部載っているので、Markdown からの DB 復旧は
当初どおり成立する。）

- `ideas` — 本体。`source_ref` に部分 UNIQUE インデックスを張り、Telegram の再配信に対して
  冪等な `ON CONFLICT ... WHERE source_ref IS NOT NULL DO NOTHING` で対応
- `attachments` — 音声などの添付。**Git 管理外**（`blobs/` を `.gitignore`）。文字起こし後は
  検索・整理に不要になるため、原本は「保険」として別バックアップに留める
- `embeddings` — モデル名・次元・`content_hash`・プレフィックスを持つ。本文編集や
  モデル差し替えを検出するため
- `relations` — 類似メモの「提案」は保存しない（毎回 ANN で出せるため）。人間が
  承認/却下した判断だけを `confirmed` / `rejected` で永続化する
- `kv` — Telegram の `getUpdates` offset など、ランタイム状態

### status 運用

自動 archive はしない。放置されている＝価値がない、ではないため（半年後に別のメモと
クラスタを作ることに価値がある）。`archived` は人間が明示的に「違った」と判断した
ものだけに使う。

### 埋め込みモデル: ruri-v3-310m

日本語特化モデルを Ollama / LM Studio で動かす方針で検討を始めたが、以下の理由で
**自前の FastAPI サーバー**に変更。

- ModernBERT + SentencePiece の組み合わせが llama.cpp 本体対応の範囲外の可能性が高く、
  Ollama / LM Studio で素直に動く保証がない
- 310M パラメータのエンコーダは量子化・VRAM 管理の恩恵がほぼなく、Mac mini の CPU/MPS で
  十分高速
- **ruri-v3 の 1+3 プレフィックス方式**（空文字列=意味、「トピック: 」=分類、
  「検索文書: 」「検索クエリ: 」=検索）を正確に制御したい。プレフィックスは
  mean pooling に含まれるため、付け方を間違えると別のベクトルになる

768 次元。`embeddings` の主キーに `prefix` 列を含めて、同じ本文でもプレフィックス違いは
別ベクトルとして管理する。

### Mac mini 側の注意点

- ModernBERT は `attn_implementation` を明示しない限り sdpa で動く（flash-attn 不要、
  そもそも Apple Silicon には入らない）
- LaunchAgent はログインしないと起動しないため、ヘッドレス運用では自動ログインが必要
- `ProcessType: Interactive` にしないと macOS のリソースマネージャに絞られて応答が鈍る
- `pmset -a sleep 0 disksleep 0` でスリープを止める（止めなくても投稿は失われないが、
  文字起こしが朝まで止まる）
- コンテナ内から `mac-mini.local` は mDNS が引けないため、Docker Compose の
  `extra_hosts` で IP を固定する

### 文字起こし

Whisper (MLX, `whisper-large-v3-turbo`)。`condition_on_previous_text=False` で、
走り書き音声にありがちな言い淀み・無音による幻覚（同じフレーズの繰り返し）を抑制。

### reconciliation パターン

文字起こし・埋め込み・類似通知・テーマ生成は、すべて「欠けている派生データを探して
埋める」という同じ形。キューを持たず、毎周回で欠損を検出する設計にした理由:

- Mac mini や GPU ノードが落ちていても、投稿自体は成功する
- ワーカーが落ちても次の周回で勝手に追いつく（リトライ処理もデッドレターキューも不要）

---

## Phase 2: 自動整理

### クラスタリング: HDBSCAN + ID 継承

768 次元のまま precomputed cosine 距離行列を HDBSCAN に渡す方式。UMAP は挟まない
（非決定的で結果がぶれる、numba 依存を Pi に持ち込むことになる）。数千件規模までは
この方式で十分。

**クラスタ ID の継承**が最大の設計判断。HDBSCAN は毎回ラベルを振り直すため、素直に
使うとテーマ名が消え、テーマノートが毎晩全書き換えになる。前回クラスタと今回クラスタを
**メンバー集合の Jaccard 係数で貪欲マッチング**し、閾値 (0.4) 以上で ID を引き継ぐ。
マッチしなかった旧クラスタは削除せず `closed_at` で閉じる。

閉じたクラスタも次回以降のマッチング候補に残す。話題が数か月止まってから戻ってきた
ときに新しい ID を振ると、テーマ名も調査履歴 (`researched_at` / `findings`) も
切れてしまうため。ただし `idea_clusters` は 1 メモ 1 行（`idea_id` が主キー）で、
閉じたクラスタのメンバーを残しておけない。そこで `clusters.members_json` に
最後のメンバー集合そのものを持たせ、そちらを照合に使う。

- クラスタリング用の埋め込みは「トピック: 」プレフィックス版を別テーブルで保持
- テーマ名は**初回のみ生成**、要約はメンバーが変わるたびに再生成（毎晩ちらつくのを防ぐ）
- 人間が Markdown の front matter で `name_locked: true` にすると、その名前を
  DB 側が吸い上げて LLM が上書きしなくなる（Phase 1 の「人間の判断が機械の出力に勝つ」
  構造と同じ）。吸い上げは命名の要否とは無関係に毎周回で走らせる。メンバーが動くまで
  待つと、Obsidian に出る名前と Web UI・ダイジェストに出る名前が食い違ったままになる

### テーマは 2 話題目が来るまで出ない

HDBSCAN は `allow_single_cluster=False` が既定で、階層木の根はクラスタとして
選ばない。**話題が 1 つしか無いうちは、似たメモが何件溜まっても全部ノイズになる。**
使い始めの数週間は「テーマが 1 つも出ない」ことになるが、不具合ではない。

有効化すれば 1 話題でもテーマは出るが、擬似データで確かめたところ、明確に分かれた
3 話題があっても全部 1 個に潰れた。片方を取れば片方が壊れる類の設定なので、既定の
まま使い、この挙動をテストで固定した (`SingleTopicCorpus`)。

---

## Phase 3: 関連情報収集

### 調査単位: テーマ（個別メモではない）

走り書き 1 件からはまともな検索クエリが作れないため、クラスタ単位で調査する。
Phase 2 を先に作った意味がここで効く。

### クエリの撃ち分け

賛成材料だけ集めても価値が薄いため、3 種類のクエリを明示的に生成:

- `prior_art` — 既に誰かがやっていないか（先行事例・既存 OSS）
- `evidence` — 技術的な裏付け
- `counter` — 否定的知見・失敗事例・批判

選別プロンプトにも「challenges は積極的に拾え」と明記。

### 収集フロー

検索 → LLM が候補を選別（薄いまとめ記事・広告を除外）→ 本文取得（best-effort、
robots.txt は見ていない）→ 要約 → `findings` に保存。1 テーマ 1 回の調査で最大 3 件まで
（増やすとノイズで埋まる）。検索バックエンドは Brave / SearXNG を差し替え可能。

### 週次ダイジェスト

Markdown 出力 + Telegram 通知 + 読み上げ原稿（TTS 用、リンクや記号を含まない素の文章）。
「判断待ちの関連候補が溜まっている件数」を載せ、`SIM_THRESHOLD` の調整サインにする。

---

## UI: 適材適所

Next.js の 3 ビュー構成を当初想定していたが、過剰と判断し撤回。

| やりたいこと | 手段 | 理由 |
|---|---|---|
| 参考情報の有用/不要の判断 | Telegram | 関連メモの承認と同じパターン。UI 不要 |
| 腰を据えた閲覧・編集 | Obsidian | `repo/` を vault として開くだけ。バックリンク・グラフビューが無料で付く |
| 外出先からの閲覧・横断検索・status 変更 | 自前 Web UI (FastAPI + HTML 1枚) | ブラウザしかない端末でも読める |

検索はベクトル（ruri-v3、semantic プレフィックス）と FTS5（trigram tokenizer）を
RRF (Reciprocal Rank Fusion) で統合。ただし **trigram は 3 文字未満の日本語クエリを
索引できない**という穴があり、「メモ」「整理」のような 2 文字語は `LIKE` の全走査に
フォールバックさせている。埋め込みサービスが落ちていても全文検索だけで結果を返す
（片肺で動く設計）。

### Web UI を「検索と俯瞰だけ」から広げた

当初は「Obsidian で読めるものは作らない」として、Web UI を検索とクラスタ俯瞰に
限っていた。しかし iPhone で外出先から vault は開けず、**思い出したいときに手元に
あるのはブラウザだけ**という状況が実際には多い。そこで一覧・本文・返信の連なりまで
Web UI に持たせた。Obsidian は母艦のまま、Web UI は「読める側」に広げる位置づけ。

書き込みは status の変更だけに留める。本文編集を入れると Markdown と DB の
双方向同期が必要になり、「正本は DB、Markdown は再生成」という前提が崩れる。

- **ハッシュルーティング** — `#/ideas/<uid>` のような URL で画面を表す。iPhone の
  戻るジェスチャがそのまま効き、検索結果の URL を共有もできる。検索語の反映は
  `replaceState` にして、1 文字ごとに履歴を積まない
- **キーセットページング** — `(captured_at, id) < (?, ?)` で次ページを引く。OFFSET だと
  読んでいる最中に新着が入ったとき、境界のメモが飛ぶか二重に出る。
  `captured_at` は同じ秒が並びうるので `id` まで見て割る
- **抜粋はサーバー側** — 一覧・検索結果は全文ではなく抜粋を返す。検索時はヒット位置に
  窓を寄せるので、長いメモの末尾に当たった語でも「何に当たったのか」が分かる
- **ビルドステップなし** — HTML/CSS/JS 1 枚をインラインで返す。npm も CDN も挟まないので、
  Pi の Docker イメージは Python だけで完結する

---

## インフラ: Docker と Mac mini の役割分担

「無理して Docker にすることはありません。適材適所で」という方針を 2 度確認し、
以下の形に収束した。

- **Docker 化した**: Pi 側の常駐ワーカー・スケジューラ・Web UI
- **Docker 化しなかった**: Mac mini の埋め込み/文字起こしサービス
  — Docker Desktop for Mac は Metal が見えないコンテナ内 VM で動くため、
  `mlx-whisper` はそもそも動かず、ruri-v3 も MPS が使えず CPU にフォールバックする

### 常駐プロセスの統合

当初 6 コンテナ（ingest / transcriber / embedder / notifier / themer / scheduler）に
分けていたが、**過剰な分割**と判断して 4 つの reconciliation ワーカー
（transcribe / embed / notify / theme）を `supervisor.py` 1 本に統合。

分けていたときの実害:
- Python プロセス 6 本で ~500MB → 1 本なら ~120MB
- ボイスメモが「文字起こし→埋め込み→通知」を抜けるのに各ワーカーの
  ポーリング間隔ぶん遅延し、最大 3 分かかっていた
- ログが 6 箇所に分散する

統合後は依存順（`transcribe → embed → similar → findings → theme`）に 1 パスで流し、
1 段が落ちても後続は続行する。`telegram_ingest`（long polling）と `web`（HTTP サーバー）
は形が違うため分離を維持。常駐は **ingest / supervisor / web** の 3 コンテナ + scheduler。

---

## 動作検証で見つかった不具合と修正

### Phase 1

実装後、擬似データでパイプライン全体を実行して検証し、以下を修正した。

1. **取り込みが全滅するバグ（致命的）** — `source_ref` の UNIQUE が部分インデックスの
   ため、`ON CONFLICT(source, source_ref) DO NOTHING` だけでは解決されず
   `OperationalError` で落ちていた。`WHERE source_ref IS NOT NULL` を `ON CONFLICT` 側にも
   明記して修正
2. **2文字の日本語クエリが検索できない** — trigram tokenizer の仕様上の制約。
   3文字未満は `LIKE` 全走査にフォールバックする形で対応
3. **検索 API のキー欠落** — ヒットゼロ時に `vector`/`keyword` キーを返しておらず
   フロント側で undefined になっていた

検証できたこと: 冪等な取り込み、embed の2プレフィックス生成、類似通知→承認→
relations記録→Markdown書き戻し、ハイブリッド検索、週次ダイジェスト生成、
supervisorの1パス実行。

### Phase 2

Phase 1 と同じ形で通し試験 (`tests/test_phase2.py`) を書いて流したところ、
**設計に書いた振る舞いのうち 4 つがコードで実現されていなかった**。

1. **人が付けたテーマ名が LLM に上書きされる（Phase 2 の要の破れ）** —
   `name_locked: true` を吸い上げて DB に書いた直後、同じ関数の後半が
   *吸い上げ前に読んだ* 行の名前で上書きし直していた。「人間の判断が機械の出力に勝つ」
   という構造そのものが効いていない状態。吸い上げを `absorb_overrides()` として
   命名処理から分離して修正
2. **人の編集がメンバーの動きを待たないと反映されない** — 吸い上げは「命名待ち」の
   クラスタしか見ていなかった。名前を直しても、そのクラスタにメモが増えるまで
   Web UI とダイジェストは古い名前を出し続ける。毎周回で全テーマノートを見るようにした
   （開いているテーマの数だけの小さなファイル読み）
3. **LLM が JSON 以外を返すと supervisor が空回りする** — `tick()` が「拾ったクラスタ数」を
   返していたため、命名に失敗しても「進捗あり」と報告し、`MAX_PASSES` まで回り続けていた。
   Phase 1 の「音声ファイルが無いときに空回りしない」と同じ穴。**進んだ件数**を返すよう修正
4. **閉じたクラスタが二度と戻らない** — 「マッチしなかった旧クラスタは `closed_at` で
   閉じる（再結成に備える）」と書いていたが、照合元が `idea_clusters` だったため、
   閉じた時点でメンバーが消え、同じ話題が戻ってきても新しい ID が振られていた
   （＝テーマ名も調査履歴も失う）。`clusters.members_json` を足して照合し、
   閉じたクラスタも候補に含めるようにした

ついでに直したもの:

- 閉じたテーマノートが Obsidian 側から現役と区別できなかった。`closed_at` を front matter に
  出し、解散時点のメンバーを載せて書き直す（開閉が変わったときだけ）
- テーマ名に `:` が入ると front matter が壊れる。`name` は常に引用して出し、
  読み戻し側で外す

検証できたこと: トピックプレフィックス版ベクトルでのクラスタリング、ラベルが
振り直されても uid とテーマ名が動かないこと、テーマ名は据え置きで要約だけ再生成される
こと、`name_locked` の吸い上げとロック解除、壊れた LLM 応答・コードフェンス付き JSON の
扱い、クラスタの close と再結成、`cluster_runs` の集計。

検証できていないこと（実機確認が必要）:
- Telegram Bot API の実レスポンス（特にボイスメッセージ経路）
- 実際の ruri-v3 ベクトルでの `SIM_THRESHOLD=0.82` の妥当性
  （擬似ベクトルでは高めに出やすく、実データで分布を見て調整が要る）
- 実ベクトルでの `MIN_CLUSTER_SIZE` / `INHERIT_THRESHOLD` の妥当性。偽の埋め込みは
  3-gram ハッシュなので、島の分かれ方が本物より極端に出ている可能性が高い
- vLLM が確実に JSON だけを返すか
- Pi (arm64) での scikit-learn / sqlite-vec のホイール

なお `clusters.members_json` はスキーマ変更なので、既存 DB があるなら
`ALTER TABLE clusters ADD COLUMN members_json TEXT;` が要る（`init_db.py` は
既存 DB には触らない）。Phase 2 はまだ動かしていないので、通常は作り直しで足りる。

---

## リポジトリ

`takpanda/idea-memo`（private）。GitHub App の権限不足（リポジトリ作成 → Contents
書き込み）が発生したため、最終的にはユーザー自身が zip を展開して push する運用とした。

---

## 残っている論点

- `findings.verdict` の `not_useful` を集計して調査プロンプトへフィードバックする
  ループは未実装（実データが数十件溜まってから設計する方が良い）
- `SIM_THRESHOLD` / `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` / `INHERIT_THRESHOLD` は
  実データを見てからの調整が前提の初期値
- 完全 Docker 案（Mac mini を使わず埋め込み・文字起こしも Pi の CPU で回す）は
  `Dockerfile.embed` として用意済みだが未検証。特に faster-whisper の arm64 対応は
  実機確認が必要
