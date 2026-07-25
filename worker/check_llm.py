"""OpenAI 互換 LLM エンドポイントの最小疎通確認。

実行例:
  LLM_BASE_URL=http://192.168.1.102:11434/v1 \
  LLM_MODEL=qwen2.5:7b python check_llm.py

応答本文は出力せず、秘密情報やプロンプトをログに残さない。
"""

import json
import os
import time
import urllib.request


BASE_URL = os.environ.get("LLM_BASE_URL", "http://gpu-node:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")


def main() -> int:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "「疎通確認」とだけ返してください。"}],
            "temperature": 0,
            "max_tokens": 16,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - CLI は原因を簡潔に表示して終了する
        print(f"LLM疎通失敗: {type(exc).__name__}: {exc}")
        return 1

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        print("LLM疎通失敗: choices が空の応答")
        return 1
    elapsed = time.monotonic() - started
    print(f"LLM疎通成功: endpoint={BASE_URL}/chat/completions model={MODEL} elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
