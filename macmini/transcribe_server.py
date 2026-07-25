"""
文字起こしサービス (Mac mini 常駐)

埋め込みサービスと分けてある。埋め込みは 315M で常時軽量に張り付かせたいが、
Whisper は数 GB 食うので、必要時だけ動かす / 別ポートで止める判断をしたいため。

  uv pip install mlx-whisper fastapi uvicorn python-multipart
  uvicorn transcribe_server:app --host 0.0.0.0 --port 7998
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

# M4 なら Metal で動く。精度優先なら whisper-large-v3
MODEL_REPO = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

app = FastAPI(title="whisper transcription")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_REPO}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ja"),
) -> dict:
    import mlx_whisper

    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            result = mlx_whisper.transcribe(
                tmp.name,
                path_or_hf_repo=MODEL_REPO,
                language=language,
                # 走り書きの音声は言い淀みが多い。幻覚を抑える
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise HTTPException(500, f"transcription failed: {exc}") from exc

    return {
        "model": MODEL_REPO,
        "language": result.get("language", language),
        "text": result["text"].strip(),
    }
