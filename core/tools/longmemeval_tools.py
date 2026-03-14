from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import config

REQUIRED_RECORD_FIELDS = (
    "question_id",
    "question",
    "answer",
    "question_type",
    "haystack_sessions",
    "haystack_session_ids",
)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_role(role: Any) -> str:
    raw = str(role or "").strip().lower()
    if not raw:
        return ""

    aliases = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return aliases.get(raw, raw)


def validate_message(message: Any, session_index: int, message_index: int) -> None:
    if not isinstance(message, dict):
        raise ValueError(
            f"LongMemEval session #{session_index} message #{message_index} must be a dict, got {type(message)}"
        )
    if "role" not in message or "content" not in message:
        raise ValueError(
            f"LongMemEval session #{session_index} message #{message_index} missing role/content fields"
        )


def validate_session(session: Any, session_index: int) -> None:
    if not isinstance(session, list):
        raise ValueError(f"LongMemEval session #{session_index} must be a list, got {type(session)}")
    for message_index, message in enumerate(session, start=1):
        validate_message(message, session_index, message_index)


def validate_record(record: dict[str, Any], index: int) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"LongMemEval record #{index} missing required fields: {missing}")

    sessions = record.get("haystack_sessions")
    if not isinstance(sessions, list):
        raise ValueError(f"LongMemEval record #{index} haystack_sessions must be a list, got {type(sessions)}")
    session_ids = record.get("haystack_session_ids")
    if not isinstance(session_ids, list):
        raise ValueError(f"LongMemEval record #{index} haystack_session_ids must be a list, got {type(session_ids)}")
    if len(session_ids) != len(sessions):
        raise ValueError(
            f"LongMemEval record #{index} haystack_session_ids length {len(session_ids)} "
            f"!= haystack_sessions length {len(sessions)}"
        )
    for session_index, session in enumerate(sessions, start=1):
        validate_session(session, session_index)


def describe_longmemeval_source() -> dict[str, str]:
    local_path = str(config.LONGMEMEVAL_LOCAL_PATH or "").strip()
    cache_path = Path(str(config.LONGMEMEVAL_CACHE_PATH or "")).expanduser()
    resolved_path: Path | None = None
    if local_path:
        resolved_path = Path(local_path).expanduser()
    elif str(cache_path) and cache_path.exists():
        resolved_path = cache_path

    data_url = str(config.LONGMEMEVAL_DATA_URL or "").strip()
    if resolved_path is not None:
        filename = resolved_path.name
    elif data_url:
        filename = Path(urlparse(data_url).path).name
    else:
        filename = ""

    sha256 = str(config.LONGMEMEVAL_DATASET_SHA256 or "").strip()
    if not sha256 and resolved_path is not None and resolved_path.exists():
        sha256 = _compute_sha256(resolved_path)

    return {
        "split": str(config.LONGMEMEVAL_SPLIT or "").strip(),
        "dataset_version": str(config.LONGMEMEVAL_DATASET_VERSION or "").strip(),
        "dataset_url": data_url,
        "dataset_filename": filename,
        "dataset_sha256": sha256,
        "dataset_download_date": str(config.LONGMEMEVAL_DATASET_DOWNLOAD_DATE or "").strip(),
    }


def _load_payload_from_local(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"LongMemEval payload must be a list, got {type(data)}")
    return data


def _load_payload_from_url(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    if not isinstance(data, list):
        raise ValueError(f"LongMemEval payload must be a list, got {type(data)}")
    return data


def load_longmemeval_s() -> list[dict[str, Any]]:
    local_path = str(config.LONGMEMEVAL_LOCAL_PATH or "").strip()
    if local_path:
        path = Path(local_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LONGMEMEVAL_LOCAL_PATH does not exist: {path}")
        payload = _load_payload_from_local(path)
        for index, record in enumerate(payload, start=1):
            validate_record(record, index)
        return payload

    cache_path = Path(str(config.LONGMEMEVAL_CACHE_PATH or "")).expanduser()
    if str(cache_path) and cache_path.exists():
        payload = _load_payload_from_local(cache_path)
        for index, record in enumerate(payload, start=1):
            validate_record(record, index)
        return payload

    data_url = str(config.LONGMEMEVAL_DATA_URL or "").strip()
    if not data_url:
        raise FileNotFoundError("LONGMEMEVAL_LOCAL_PATH and LONGMEMEVAL_DATA_URL are both unset.")
    payload = _load_payload_from_url(data_url)
    if str(cache_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, record in enumerate(payload, start=1):
        validate_record(record, index)
    return payload


def sample_longmemeval_s(sample_size: int, seed: int) -> list[dict[str, Any]]:
    records = load_longmemeval_s()
    target_size = min(max(0, int(sample_size)), len(records))
    rng = random.Random(seed)
    return rng.sample(records, k=target_size)


def get_gold_answers(record: dict[str, Any]) -> list[str]:
    answer = record.get("answer", "")
    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]
    value = str(answer).strip()
    return [value] if value else []


def get_record_category(record: dict[str, Any]) -> str:
    return str(record.get("question_type", "")).strip()


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": normalize_role(message["role"]),
        "content": str(message["content"]).strip(),
    }


def session_to_messages(sessions: list[list[dict[str, Any]]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for session in sessions:
        for raw_message in session:
            normalized = normalize_message(raw_message)
            if normalized["role"] and normalized["content"]:
                messages.append(normalized)
    return messages


def get_recent_sessions(record: dict[str, Any], window_sessions: int) -> list[Any]:
    sessions = list(record.get("haystack_sessions", []) or [])
    keep = max(0, int(window_sessions))
    if keep == 0:
        return []
    return sessions[-keep:]


def build_question_message(question: str) -> dict[str, str]:
    return {"role": "user", "content": str(question or "").strip()}


def session_to_text(session: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in session_to_messages([session]):
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines).strip()


def get_session_identifier(record: dict[str, Any], session_index: int) -> str | int:
    session_ids = record.get("haystack_session_ids", [])
    if 0 <= session_index < len(session_ids):
        return session_ids[session_index]
    return session_index
