from __future__ import annotations

import os
from typing import Final

from . import _bootstrap  # noqa: F401

# Workflow constants
LONGMEMEVAL_DEBUG_SAMPLE_SIZE = 20
LONGMEMEVAL_PILOT_SAMPLE_SIZE = 100
LONGMEMEVAL_FORMAL_SAMPLE_SIZE = 500
LONGMEMEVAL_RECOMMENDED_REALISTIC_SAMPLE_SIZE = 120
LONGMEMEVAL_RECOMMENDED_CONTROLLED_SAMPLE_SIZE = 50
LONGMEMEVAL_RECOMMENDED_CONTROLLED_SWEEP_SAMPLE_SIZE = 30
LONGMEMEVAL_RECOMMENDED_CONTROLLED_SUMMARY_SAMPLE_SIZE = 30
LONGMEMEVAL_DEBUG_REPEATS = 1
LONGMEMEVAL_PILOT_REPEATS = 2
LONGMEMEVAL_FORMAL_REPEATS = 3
LONGMEMEVAL_RECOMMENDED_REPEATS = 3

EXPERIMENT_STAGE = os.getenv("EXPERIMENT_STAGE", "recommended").strip().lower() or "recommended"

_STAGE_DEFAULTS: Final[dict[str, dict[str, int]]] = {
    "debug": {
        "realistic_sample_size": LONGMEMEVAL_DEBUG_SAMPLE_SIZE,
        "controlled_sample_size": LONGMEMEVAL_DEBUG_SAMPLE_SIZE,
        "controlled_sweep_sample_size": 10,
        "controlled_summary_sample_size": 10,
        "repeats": LONGMEMEVAL_DEBUG_REPEATS,
    },
    "pilot": {
        "realistic_sample_size": LONGMEMEVAL_PILOT_SAMPLE_SIZE,
        "controlled_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SAMPLE_SIZE,
        "controlled_sweep_sample_size": 20,
        "controlled_summary_sample_size": 20,
        "repeats": LONGMEMEVAL_PILOT_REPEATS,
    },
    "formal": {
        "realistic_sample_size": LONGMEMEVAL_FORMAL_SAMPLE_SIZE,
        "controlled_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SAMPLE_SIZE,
        "controlled_sweep_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SWEEP_SAMPLE_SIZE,
        "controlled_summary_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SUMMARY_SAMPLE_SIZE,
        "repeats": LONGMEMEVAL_FORMAL_REPEATS,
    },
    "recommended": {
        "realistic_sample_size": LONGMEMEVAL_RECOMMENDED_REALISTIC_SAMPLE_SIZE,
        "controlled_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SAMPLE_SIZE,
        "controlled_sweep_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SWEEP_SAMPLE_SIZE,
        "controlled_summary_sample_size": LONGMEMEVAL_RECOMMENDED_CONTROLLED_SUMMARY_SAMPLE_SIZE,
        "repeats": LONGMEMEVAL_RECOMMENDED_REPEATS,
    },
}

if EXPERIMENT_STAGE not in _STAGE_DEFAULTS:
    EXPERIMENT_STAGE = "recommended"

_ACTIVE_STAGE_DEFAULTS = _STAGE_DEFAULTS[EXPERIMENT_STAGE]


def _parse_int_tuple(raw: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in str(raw or "").split(",")]
    values: list[int] = []
    seen: set[int] = set()
    for piece in pieces:
        if not piece:
            continue
        try:
            number = int(piece)
        except ValueError:
            continue
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        values.append(number)
    return tuple(values) if values else default

# Actual experiment knobs
LONGMEMEVAL_REALISTIC_SAMPLE_SIZE = int(
    os.getenv("LONGMEMEVAL_REALISTIC_SAMPLE_SIZE", str(_ACTIVE_STAGE_DEFAULTS["realistic_sample_size"]))
)
LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE = int(
    os.getenv("LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE", str(_ACTIVE_STAGE_DEFAULTS["controlled_sample_size"]))
)
LONGMEMEVAL_CONTROLLED_SWEEP_SAMPLE_SIZE = int(
    os.getenv(
        "LONGMEMEVAL_CONTROLLED_SWEEP_SAMPLE_SIZE",
        str(_ACTIVE_STAGE_DEFAULTS["controlled_sweep_sample_size"]),
    )
)
LONGMEMEVAL_CONTROLLED_SUMMARY_SAMPLE_SIZE = int(
    os.getenv(
        "LONGMEMEVAL_CONTROLLED_SUMMARY_SAMPLE_SIZE",
        str(_ACTIVE_STAGE_DEFAULTS["controlled_summary_sample_size"]),
    )
)
LONGMEMEVAL_WINDOW_SESSIONS = int(os.getenv("LONGMEMEVAL_WINDOW_SESSIONS", "5"))
LONGMEMEVAL_DRIFT_KEEP_SESSIONS = int(os.getenv("LONGMEMEVAL_DRIFT_KEEP_SESSIONS", "3"))
LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS = _parse_int_tuple(
    os.getenv("LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS", "4,3,2,1"),
    default=(4, 3, 2, 1),
)
LONGMEMEVAL_ACF_K_VALUES = _parse_int_tuple(
    os.getenv("LONGMEMEVAL_ACF_K_VALUES", "8,12,16"),
    default=(8, 12, 16),
)
LONGMEMEVAL_CONTROLLED_SUMMARY_NOTE_MAX_TOKENS = int(
    os.getenv("LONGMEMEVAL_CONTROLLED_SUMMARY_NOTE_MAX_TOKENS", "48")
)
LONGMEMEVAL_RETRIEVE_TOP_K = int(os.getenv("LONGMEMEVAL_RETRIEVE_TOP_K", "3"))
LONGMEMEVAL_RETRIEVE_CANDIDATE_TOP_K = int(os.getenv("LONGMEMEVAL_RETRIEVE_CANDIDATE_TOP_K", "12"))
LONGMEMEVAL_RETRIEVE_CHUNK_MESSAGES = int(os.getenv("LONGMEMEVAL_RETRIEVE_CHUNK_MESSAGES", "4"))
LONGMEMEVAL_RETRIEVE_CHUNK_STRIDE = int(os.getenv("LONGMEMEVAL_RETRIEVE_CHUNK_STRIDE", "2"))
LONGMEMEVAL_MAX_PROMPT_TOKENS = int(os.getenv("LONGMEMEVAL_MAX_PROMPT_TOKENS", "8192"))
LONGMEMEVAL_RETRIEVAL_MAX_TOKENS = int(os.getenv("LONGMEMEVAL_RETRIEVAL_MAX_TOKENS", "1536"))
LONGMEMEVAL_REPEATS = int(os.getenv("LONGMEMEVAL_REPEATS", str(_ACTIVE_STAGE_DEFAULTS["repeats"])))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
SECRET_BITS_LENGTH = int(os.getenv("SECRET_BITS_LENGTH", "2000"))
