from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from . import _bootstrap  # noqa: F401


class ModelEnum(str, Enum):
    DEEPSEEK_R1_DISTILL_QWEN_7B = "/root/autodl-fs/DeepSeek-R1-Distill-Qwen-7B/"
    META_LLAMA_3_1_8B_INSTRUCT = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct/"
    QWEN2_5_7B_INSTRUCT = "/root/autodl-fs/Qwen2.5-7B-Instruct/"
    UNKNOWN = "unknown"


MODEL_NAME = os.getenv("MODEL_NAME", ModelEnum.QWEN2_5_7B_INSTRUCT.name).strip()


def get_model_path() -> str:
    if MODEL_NAME in ModelEnum.__members__:
        return ModelEnum[MODEL_NAME].value
    return MODEL_NAME or ModelEnum.QWEN2_5_7B_INSTRUCT.value


def get_model_label() -> str:
    model_path = get_model_path()
    return Path(str(model_path).rstrip("/\\")).name or "unknown"
