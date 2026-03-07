from __future__ import annotations

import os
import random
from typing import Any

from datasets import Dataset, DownloadConfig, load_dataset
from pathlib import Path

import config

_ID_CANDIDATES = ("id", "absolute_id", "idx", "index")


def _try_load_from_cached_arrow() -> Any | None:
    dataset_root = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_key = config.CONVERSATION_STARTERS_DATASET.replace("/", "___")
    if config.CONVERSATION_STARTERS_REVISION and config.CONVERSATION_STARTERS_REVISION != "main":
        pattern = f"{dataset_key}/**/{config.CONVERSATION_STARTERS_REVISION}/*-train.arrow"
        candidates = sorted(dataset_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        pattern = f"{dataset_key}/**/*-train.arrow"
        candidates = sorted(dataset_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    for arrow_file in candidates:
        try:
            return Dataset.from_file(str(arrow_file))
        except Exception:
            continue
    return None


def load_starters_train_split() -> Any:
    if config.HF_DATASETS_LOCAL_ONLY:
        # Prefer direct cached Arrow load to avoid any Hub metadata retries.
        cached = _try_load_from_cached_arrow()
        if cached is not None:
            return cached

        # Fallback: force offline mode for datasets/hub.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    dataset_kwargs: dict[str, Any] = {
        "path": config.CONVERSATION_STARTERS_DATASET,
        "split": "train",
        "revision": config.CONVERSATION_STARTERS_REVISION,
    }
    if config.HF_DATASETS_LOCAL_ONLY:
        dataset_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return load_dataset(**dataset_kwargs)


def build_absolute_id_starters(dataset: Any) -> list[tuple[int, str]]:
    column_names = set(getattr(dataset, "column_names", []) or [])
    id_key = next((k for k in _ID_CANDIDATES if k in column_names), None)

    indexed: list[tuple[int, str]] = []
    for row_idx, row in enumerate(dataset):
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue

        if id_key is not None:
            raw_id = row.get(id_key)
            try:
                abs_id = int(raw_id)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid dataset id at row {row_idx}: {raw_id!r}") from None
        else:
            # Dataset-level absolute index fallback (stable under fixed revision).
            abs_id = row_idx

        indexed.append((abs_id, prompt))
    return indexed


def sample_absolute_id_starters(sample_size: int, seed: int) -> list[tuple[int, str]]:
    dataset = load_starters_train_split()
    indexed = build_absolute_id_starters(dataset)
    target_size = min(sample_size, len(indexed))
    rng = random.Random(seed)
    return rng.sample(indexed, k=target_size)
