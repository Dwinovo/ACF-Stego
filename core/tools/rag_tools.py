from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / config.INDEX_DIR / "chroma_db"


def _require_langchain() -> tuple[Any, Any]:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        return HuggingFaceEmbeddings, Chroma
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain_community.vectorstores import Chroma  # type: ignore
        return HuggingFaceEmbeddings, Chroma


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        batch_size: int = 32,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for remote embeddings.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            for item in resp.data:
                vectors.append(list(item.embedding))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model_name, input=[text])
        return list(resp.data[0].embedding)


def create_embeddings(*, model_name: str, device: str) -> Any:
    return create_local_embeddings(model_name=model_name, device=device)


def create_local_embeddings(*, model_name: str, device: str) -> Any:
    HuggingFaceEmbeddings, _ = _require_langchain()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_remote_embeddings(*, model_name: str, api_key: str, base_url: str) -> Any:
    return OpenAICompatibleEmbeddings(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def open_starter_vector_store(
    *,
    starter_id: int,
    embeddings: Any,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
) -> Any:
    _, Chroma = _require_langchain()
    return Chroma(
        collection_name=f"starter_{starter_id}",
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def search_starter_docs(
    *,
    vector_store: Any,
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    try:
        pairs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    for doc, score in pairs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        hits.append(
            {
                "content": str(getattr(doc, "page_content", "") or ""),
                "score": float(score),
                "metadata": metadata,
            }
        )
    return hits


def build_rag_context(
    hits: list[dict[str, Any]],
    *,
    max_items: int = 3,
    max_chars: int = 2000,
) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    used = 0
    for idx, hit in enumerate(hits[: max(1, max_items)], start=1):
        content = str(hit.get("content", "")).strip()
        if not content:
            continue
        meta = dict(hit.get("metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id", "unknown"))
        source_file = str(meta.get("source_file", "unknown"))
        score = float(hit.get("score", 0.0))
        block = (
            f"[{idx}] source={source_file} chunk={chunk_id} score={score:.4f}\n"
            f"{content}\n"
        )
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            block = block[:remaining]
            lines.append(block)
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()
