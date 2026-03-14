from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

import config


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

def _extract_first_json_block(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = raw.find(open_char)
        end = raw.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1].strip()
    return None


def _truncate_for_rerank(text: str, limit: int = 1200) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3].rstrip() + "..."


def _score_from_rank(rank: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - (rank / max(total, 1)))


class RemoteAPIReranker:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        endpoint: str = "",
        timeout: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for remote reranking.")
        if not model_name:
            raise ValueError("REMOTE_RERANK_MODEL is required for remote reranking.")
        resolved_endpoint = str(endpoint or "").strip()
        if not resolved_endpoint and base_url:
            resolved_endpoint = str(base_url).rstrip("/") + "/rerank"
        if not resolved_endpoint:
            raise ValueError("REMOTE_RERANK_ENDPOINT or OPENAI_BASE_URL is required for remote reranking.")

        self.api_key = str(api_key).strip()
        self.endpoint = resolved_endpoint
        self.model_name = str(model_name).strip()
        self.timeout = float(timeout)

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        limit = max(0, int(top_k))
        if limit == 0 or not hits:
            return []

        payload = {
            "model": self.model_name,
            "query": str(query or "").strip(),
            "top_n": limit,
            "documents": [_truncate_for_rerank(str(hit.get("content", ""))) for hit in hits],
            "return_documents": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content = response.read().decode("utf-8", errors="replace")
        return self._parse_response(content=content, hits=hits, top_k=limit)

    def _parse_response(self, *, content: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        json_block = _extract_first_json_block(content)
        if not json_block:
            raise ValueError("Remote reranker did not return JSON.")

        parsed = json.loads(json_block)
        if isinstance(parsed, dict):
            ranked_items = (
                parsed.get("results")
                or parsed.get("data")
                or parsed.get("ranked_candidates")
                or parsed.get("candidates")
                or []
            )
        elif isinstance(parsed, list):
            ranked_items = parsed
        else:
            raise ValueError(f"Unexpected remote reranker payload type: {type(parsed)}")

        ranked_hits: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        total = len(hits)

        for order, item in enumerate(ranked_items):
            if isinstance(item, dict):
                raw_index = item.get("index")
                raw_score = item.get("relevance_score", item.get("score"))
            else:
                raw_index = item
                raw_score = None

            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= total or index in seen_indices:
                continue

            seen_indices.add(index)
            base_hit = dict(hits[index])
            embedding_score = float(base_hit.get("score", 0.0))
            try:
                remote_score = float(raw_score)
            except (TypeError, ValueError):
                remote_score = _score_from_rank(order, total=max(total, 1))
            remote_score = max(0.0, min(1.0, remote_score))

            base_hit.update(
                {
                    "embedding_score": embedding_score,
                    "remote_rerank_score": remote_score,
                    "score": remote_score,
                    "rerank_method": "remote_api",
                    "rerank_model": self.model_name,
                }
            )
            ranked_hits.append(base_hit)
            if len(ranked_hits) >= top_k:
                break

        if not ranked_hits:
            raise ValueError("Remote reranker returned no valid candidates.")
        return ranked_hits


@dataclass
class SessionRetrievalIndex:
    session_texts: list[str]
    doc_vectors: list[list[float]]
    embeddings_client: Any
    reranker: RemoteAPIReranker

    @classmethod
    def build(
        cls,
        session_texts: list[str],
    ) -> "SessionRetrievalIndex":
        embeddings_client = OpenAICompatibleEmbeddings(
            model_name=config.REMOTE_EMBED_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        reranker = RemoteAPIReranker(
            model_name=config.REMOTE_RERANK_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            endpoint=config.REMOTE_RERANK_ENDPOINT,
        )
        doc_vectors = embeddings_client.embed_documents(session_texts)
        return cls(
            session_texts=session_texts,
            doc_vectors=doc_vectors,
            embeddings_client=embeddings_client,
            reranker=reranker,
        )

    def search(self, query: str, top_k: int, candidate_top_k: int | None = None) -> list[dict[str, Any]]:
        limit = max(0, int(top_k))
        if limit == 0 or not self.session_texts:
            return []
        candidate_limit = max(limit, int(candidate_top_k)) if candidate_top_k is not None else limit

        query_vector = self.embeddings_client.embed_query(query)
        scored = [
            {
                "index": idx,
                "score": float(_cosine_similarity(query_vector, doc_vector)),
                "content": text,
            }
            for idx, (text, doc_vector) in enumerate(zip(self.session_texts, self.doc_vectors))
        ]

        scored.sort(key=lambda item: item["score"], reverse=True)
        candidates = scored[:candidate_limit]
        return self.reranker.rerank(query=str(query or ""), hits=candidates, top_k=limit)


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str,
        batch_size: int = 32,
        timeout: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding retrieval.")

        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model_name = str(model_name).strip()
        self.batch_size = max(1, int(batch_size))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for idx in range(0, len(texts), self.batch_size):
            batch = texts[idx : idx + self.batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            for item in response.data:
                vectors.append(list(item.embedding))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=[text])
        return list(response.data[0].embedding)
