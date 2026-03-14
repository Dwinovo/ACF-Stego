from __future__ import annotations

import os

from . import _bootstrap  # noqa: F401
from .paths import RAW_DIR

LONGMEMEVAL_SPLIT = os.getenv("LONGMEMEVAL_SPLIT", "longmemeval_s").strip() or "longmemeval_s"
LONGMEMEVAL_DATA_URL = os.getenv(
    "LONGMEMEVAL_DATA_URL",
    "https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json",
)
LONGMEMEVAL_LOCAL_PATH = os.getenv("LONGMEMEVAL_LOCAL_PATH", "")
LONGMEMEVAL_CACHE_PATH = os.getenv("LONGMEMEVAL_CACHE_PATH", f"{RAW_DIR}/longmemeval_s_cleaned.json")
LONGMEMEVAL_DATASET_VERSION = os.getenv("LONGMEMEVAL_DATASET_VERSION", "2025-09 cleaned").strip()
LONGMEMEVAL_DATASET_DOWNLOAD_DATE = os.getenv("LONGMEMEVAL_DATASET_DOWNLOAD_DATE", "").strip()
LONGMEMEVAL_DATASET_SHA256 = os.getenv("LONGMEMEVAL_DATASET_SHA256", "").strip()
