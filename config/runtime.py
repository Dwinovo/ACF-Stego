from __future__ import annotations

import os

from . import _bootstrap  # noqa: F401


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# Runtime / model params
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
TOP_K = int(os.getenv("TOP_K", "100"))
TOP_P = float(os.getenv("TOP_P", "0.0"))
STEGO_PRECISION = int(os.getenv("STEGO_PRECISION", "52"))

# External services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
REMOTE_EMBED_MODEL = os.getenv("REMOTE_EMBED_MODEL", "text-embedding-3-small").strip()
REMOTE_RERANK_MODEL = os.getenv("REMOTE_RERANK_MODEL", "bce-reranker-base").strip()
REMOTE_RERANK_ENDPOINT = os.getenv("REMOTE_RERANK_ENDPOINT", "").strip()
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "").strip()
LLM_JUDGE_TEMPERATURE = float(os.getenv("LLM_JUDGE_TEMPERATURE", "0"))
LLM_JUDGE_MAX_TOKENS = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "300"))
LLM_JUDGE_TIMEOUT_SECONDS = float(os.getenv("LLM_JUDGE_TIMEOUT_SECONDS", "30"))
LLM_JUDGE_PROMPT_VERSION = os.getenv("LLM_JUDGE_PROMPT_VERSION", "semantic_reason_then_score_v1").strip()
SKIP_EXISTING_OUTPUTS = _env_flag("SKIP_EXISTING_OUTPUTS", default=True)
