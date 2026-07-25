"""
テスト用スタブ。

Mac mini の埋め込み / 文字起こしサーバー、GPU ノードの LLM (OpenAI 互換)、
Telegram の file API を、本物と同じ HTTP 契約で喋る 1 プロセスに寄せる。
ワーカー側の JSON 組み立て・multipart 生成・次元チェック・
コードフェンス剥がしまで通したいので、関数を差し替えるのではなく
サーバーを立てる。

埋め込みは文字 3-gram をハッシュして 768 次元に畳んだだけの偽物。
「近い文は近いベクトルになる」ことだけが要件で、意味は持たない。
"""

import hashlib
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 768

# ワーカーはモジュール定数として環境変数を読む。同じプロセスで Phase 1 と
# Phase 2 を discover すると、先に import した側の DB とリポジトリを
# 掴んだままになるので、テストモジュールごとに import し直させる
WORKER_MODULES = (
    "common", "init_db", "telegram_ingest", "transcribe_worker", "embed_worker",
    "notify_worker", "theme_writer", "cluster_worker", "research_worker",
    "digest_worker", "supervisor", "web",
)


def fresh_worker_imports() -> None:
    """環境変数を差し替えた後、次の import で読み直させる。"""
    for name in WORKER_MODULES:
        sys.modules.pop(name, None)

# 本物と同じ 1+3 プレフィックス。prefix が違えば別ベクトルになることも再現する
PREFIXES = {
    "semantic": "",
    "topic": "トピック: ",
    "query": "検索クエリ: ",
    "document": "検索文書: ",
}


def fake_embed(text: str, prefix: str = "") -> list[float]:
    vec = [0.0] * DIM
    body = prefix + text
    for i in range(max(len(body) - 2, 1)):
        gram = body[i : i + 3]
        digest = hashlib.sha1(gram.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[slot] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _parse_multipart(body: bytes, boundary: str) -> dict[str, bytes]:
    """テスト側の検証用に最小限だけ実装する。"""
    parts: dict[str, bytes] = {}
    sep = f"--{boundary}".encode()
    for chunk in body.split(sep):
        if not chunk.strip(b"\r\n-"):
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        for line in head.split(b"\r\n"):
            if b"name=" in line:
                name = line.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
                parts[name] = payload.rstrip(b"\r\n")
                break
    return parts


def default_llm_reply(prompt: str) -> str:
    """メモ件数だけを見てテーマ名と要約を作る偽 LLM。

    件数を名前に混ぜてあるので、メンバーが変わると「LLM が返す名前」も
    変わる。テーマ名が初回のまま据え置かれること (ちらつき防止) と、
    要約だけ作り直されることを、この差で見分けられる。
    """
    memos = [line for line in prompt.splitlines() if line.startswith("- (")]
    first = memos[0].split(") ", 1)[-1] if memos else ""
    return json.dumps(
        {
            "name": f"テーマ{len(memos)}件",
            "summary": f"{len(memos)}件のメモ。代表: {first[:20]}",
        },
        ensure_ascii=False,
    )


class Handler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    seen: list[dict] = []
    # 先頭から取り出して返す生の content。空なら default_llm_reply
    llm_script: list[str] = []
    llm_payload: dict | None = None

    def log_message(self, *args) -> None:      # テスト出力を汚さない
        pass

    def _json(self, code: int, payload: dict) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:
        name = self.path.rsplit("/", 1)[-1]
        blob = self.files.get(name)
        if blob is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/ogg")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if self.path == "/embed":
            req = json.loads(body)
            prefix = PREFIXES[req.get("prefix", "semantic")]
            self.seen.append({"endpoint": "embed", "n": len(req["input"]),
                              "prefix": req.get("prefix")})
            self._json(200, {
                "model": "cl-nagoya/ruri-v3-310m",
                "prefix": req.get("prefix", "semantic"),
                "dim": DIM,
                "embeddings": [fake_embed(t, prefix) for t in req["input"]],
            })
            return

        if self.path == "/transcribe":
            ctype = self.headers.get("Content-Type", "")
            boundary = ctype.split("boundary=", 1)[1]
            parts = _parse_multipart(body, boundary)
            self.seen.append({"endpoint": "transcribe",
                              "language": parts.get("language", b"").decode(),
                              "bytes": len(parts.get("file", b""))})
            # 偽の音声ファイルは中身が文字起こし結果そのもの
            self._json(200, {
                "model": "fake-whisper",
                "language": parts.get("language", b"ja").decode(),
                "text": parts.get("file", b"").decode("utf-8").strip(),
            })
            return

        if self.path.endswith("/chat/completions"):
            req = json.loads(body)
            if self.llm_payload is not None:
                self._json(200, self.llm_payload)
                return
            prompt = req["messages"][-1]["content"]
            content = (
                self.llm_script.pop(0) if self.llm_script else default_llm_reply(prompt)
            )
            self.seen.append({"endpoint": "llm", "path": self.path, "prompt": prompt,
                              "model": req.get("model")})
            self._json(200, {
                "model": req.get("model", "fake-llm"),
                "choices": [{"message": {"role": "assistant", "content": content}}],
            })
            return

        self.send_error(404)


class FakeServices:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        Handler.files = {}
        Handler.seen = []
        Handler.llm_script = []
        Handler.llm_payload = None

    def __enter__(self) -> "FakeServices":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def put_file(self, name: str, blob: bytes) -> None:
        Handler.files[name] = blob

    def blob(self, name: str) -> bytes:
        return Handler.files[name]

    def script_llm(self, *replies: str) -> None:
        """次の LLM 応答を指定する。壊れた応答を投げ込むのに使う。"""
        Handler.llm_script = list(replies)

    def set_llm_payload(self, payload: dict | None) -> None:
        """LLM のHTTP 200応答を差し替える。異常レスポンスのテスト用。"""
        Handler.llm_payload = payload

    @property
    def calls(self) -> list[dict]:
        return Handler.seen
