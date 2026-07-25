"""
ruri-v3-310m 埋め込みサービス (Mac mini 常駐用)

  uv pip install "transformers>=4.48.0" sentence-transformers sentencepiece \
                 torch fastapi uvicorn
  uvicorn embed_server:app --host 0.0.0.0 --port 7997

Ollama を使わない理由:
  ruri-v3 の 1+3 プレフィックス方式を用途ごとに正確に出し分けたいため。
  プレフィックスは mean pooling に含まれる (include_prompt=True) ので、
  付け方を間違えるとベクトルが変わる。
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("RURI_MODEL", "cl-nagoya/ruri-v3-310m")

# 短いメモが対象なので 8192 は要らない。KVキャッシュとメモリの無駄を避ける。
# Phase 3 で収集記事を埋め込むなら 4096 程度まで上げる
MAX_SEQ_LEN = int(os.environ.get("RURI_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("RURI_BATCH_SIZE", "16"))

# ruri-v3 の 1+3 プレフィックス方式
PREFIXES: dict[str, str] = {
    "semantic": "",             # 意味のエンコード。メモ同士の類似判定はこれ
    "topic": "トピック: ",       # 分類・クラスタリング。Phase 2 で使う
    "query": "検索クエリ: ",     # 検索する側
    "document": "検索文書: ",    # 検索される側
}

PrefixKind = Literal["semantic", "topic", "query", "document"]

state: dict = {}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # M4 Mac mini はこちら。ModernBERT は transformers 側で
    # FlashAttention2 がデフォルトではなくなったため、sdpa で普通に動く
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = pick_device()
    model = SentenceTransformer(MODEL_ID, device=device)
    model.max_seq_length = MAX_SEQ_LEN

    state["model"] = model
    state["device"] = device
    state["dim"] = model.get_sentence_embedding_dimension()

    # 起動時に一度回して、初回リクエストのレイテンシを吸収する
    model.encode(["ウォームアップ"], normalize_embeddings=True)
    yield
    state.clear()


app = FastAPI(title="ruri-v3 embeddings", lifespan=lifespan)


class EmbedRequest(BaseModel):
    input: list[str] = Field(..., min_length=1, max_length=256)
    prefix: PrefixKind = "semantic"


class EmbedResponse(BaseModel):
    model: str
    prefix: PrefixKind
    dim: int
    embeddings: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "model" in state else "loading",
        "model": MODEL_ID,
        "device": state.get("device"),
        "dim": state.get("dim"),
        "max_seq_length": MAX_SEQ_LEN,
    }


def _encode(texts: list[str], prefix_kind: str):
    prefix = PREFIXES[prefix_kind]
    prefixed = [prefix + t for t in texts]
    with torch.inference_mode():
        return state["model"].encode(
            prefixed,
            batch_size=BATCH_SIZE,
            # cosine 前提で正規化。sqlite-vec 側も distance_metric=cosine
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if "model" not in state:
        raise HTTPException(503, "model is still loading")
    vecs = _encode(req.input, req.prefix)
    return EmbedResponse(
        model=MODEL_ID,
        prefix=req.prefix,
        dim=int(vecs.shape[1]),
        embeddings=vecs.tolist(),
    )


# ------------------------------------------------------------
# OpenAI 互換エンドポイント (OpenWebUI / Dify 等から使う場合)
# プレフィックスは model 名のサフィックスで指定する:
#   "ruri-v3-310m"        -> semantic
#   "ruri-v3-310m/topic"  -> トピック:
# ------------------------------------------------------------
class OpenAIEmbedRequest(BaseModel):
    input: str | list[str]
    model: str = "ruri-v3-310m"


@app.post("/v1/embeddings")
def openai_embeddings(req: OpenAIEmbedRequest) -> dict:
    if "model" not in state:
        raise HTTPException(503, "model is still loading")

    texts = [req.input] if isinstance(req.input, str) else req.input
    kind = req.model.split("/")[-1] if "/" in req.model else "semantic"
    if kind not in PREFIXES:
        raise HTTPException(400, f"unknown prefix: {kind}")

    vecs = _encode(texts, kind)
    return {
        "object": "list",
        "model": req.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vecs)
        ],
    }
