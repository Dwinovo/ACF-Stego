from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Any, Dict

import config
from mem0 import Memory



def create_memory() -> Memory:
    """Create Mem0 with quality-first defaults (AIHUBMIX/OpenAI-compatible).

    Notes:
    - Uses OpenAI-compatible LLM + embedder by default (higher quality).
    - If you want local embedder, pass a custom config or change this function.
    - Set env `OPENAI_API_KEY` before calling.
    - Memory collection is fixed to `experiment`; isolation relies on `user_id`.
    """

    api_key = config.OPENAI_API_KEY
    llm_model = config.MEM0_AGENT
    base_url = config.OPENAI_BASE_URL
    qdrant_dir: str | Path = config.QDRANT_DIR

    if not api_key or not llm_model:
        raise ValueError("Missing required env vars: OPENAI_API_KEY, OPENAI_LLM_MODEL")
    qdrant_path = str(qdrant_dir)
    os.makedirs(qdrant_path, exist_ok=True)
    llm_config = {
        "api_key": api_key,
        "model": llm_model,
    }
    embedder_config = {
        "api_key": api_key,
    }
    if base_url:
        llm_config["openai_base_url"] = base_url
        embedder_config["openai_base_url"] = base_url
    mem0_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": qdrant_path,
                "on_disk": True,
                "collection_name": "experiment",
            },
        },
        "llm": {
            "provider": "openai",
            "config": llm_config,
        },
        "embedder": {
            "provider": "openai",
            "config": embedder_config,
        },
    }
    return Memory.from_config(mem0_config)


def _safe_delete(memory: Memory, memory_id: str) -> Dict[str, Any]:
    try:
        return memory.delete(memory_id=memory_id)
    except TypeError:
        return memory.delete(memory_id)


def reset_memories_by_userid(user_id: str, memory: Memory) -> Dict[str, Any]:
    """Delete all memories for one user_id in fixed collection `experiment`."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    data = memory.get_all(user_id=user_id)
    items = data.get("results", []) if isinstance(data, dict) else []

    deleted = 0
    failed_ids = []
    for item in items:
        memory_id = item.get("id") if isinstance(item, dict) else None
        if not memory_id:
            continue
        try:
            _safe_delete(memory, memory_id)
            deleted += 1
        except Exception:
            failed_ids.append(memory_id)

    return {
        "ok": True,
        "collection_name": "experiment",
        "user_id": user_id,
        "found": len(items),
        "deleted": deleted,
        "failed": len(failed_ids),
        "failed_memory_ids": failed_ids,
    }


def save_memory(
    memory: Memory,
    content: str,
    metadata: Dict[str, Any] | None = None,
    user_id: str = "default_user",
) -> Any:
    """Save one memory item."""

    # Force sync write so memory is durably persisted before process exits.
    try:
        return memory.add(
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata=metadata,
            infer=True,
            async_mode=False,
        )
    except TypeError:
        # Backward compatibility for mem0 versions without async_mode.
        return memory.add(
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata=metadata,
            infer=True,
        )


def search_memory(
    memory: Memory,
    query: str,
    limit: int = 3,
    threshold: float | None = None,
    user_id: str = "default_user",
) -> Any:
    """Search memories."""

    score_threshold = float(threshold) if threshold is not None else None
    return memory.search(
        query=query,
        user_id=user_id,
        limit=int(limit),
        threshold=score_threshold,
        rerank=True,
    )


def list_memories(memory: Memory, user_id: str = "default_user") -> Any:
    """List all memories for one user."""
    return memory.get_all(user_id=user_id)


def delete_memory(memory: Memory, memory_id: str) -> Any:
    """Delete one memory by id."""
    return _safe_delete(memory, memory_id)


def close_memory(memory: Memory | None) -> None:
    """Best-effort close of underlying vector-store clients to release file locks."""
    if memory is None:
        return

    stores = [
        getattr(memory, "vector_store", None),
        getattr(memory, "_telemetry_vector_store", None),
    ]
    for store in stores:
        client = getattr(store, "client", None)
        if client is None:
            continue
        try:
            client.close()
        except Exception:
            pass

    # Encourage timely destructor execution for libs that release resources on GC.
    gc.collect()
