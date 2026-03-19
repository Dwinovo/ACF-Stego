from __future__ import annotations

import gc
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import stegokit
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.tools import entropy_tools
from core.tools import longmemeval_tools
from core.tools import qa_metrics
from core.tools import session_retrieval

STEGO_STOP_ON_EOS = True
STEGO_SECURE_PARAMETER = 16
STEGO_FUNC_TYPE = 0
EMBEDDING_CAPACITY_SCALE_PER_1K_TOKENS = 1000.0
DRIFT_RECENT_PATTERN = re.compile(r"^drift_recent(\d+)$")


def _patch_stegokit_meteor_decode_index_error() -> None:
    try:
        from stegokit.algo.meteor.meteor import MeteorStrategy
    except Exception:
        return

    patch_marker = "_spl2026_meteor_decode_index_guard"
    if bool(getattr(MeteorStrategy, patch_marker, False)):
        return

    original_decode_token_step = MeteorStrategy._decode_token_step

    def patched_decode_token_step(
        self: Any,
        *,
        prob_table: list[float],
        indices: list[int],
        prev_token_id: int,
        precision: int,
        prg: Any | None,
        cur_interval: list[int] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            return original_decode_token_step(
                self,
                prob_table=prob_table,
                indices=indices,
                prev_token_id=prev_token_id,
                precision=precision,
                prg=prg,
                cur_interval=cur_interval,
                extra=extra,
            )
        except IndexError as exc:
            if "out of bounds" not in str(exc).lower():
                raise
            return {"bits": "", "bits_len": 0, "decode_error": f"{exc.__class__.__name__}: {exc}"}

    MeteorStrategy._decode_token_step = patched_decode_token_step
    setattr(MeteorStrategy, patch_marker, True)
    print("[patch] applied stegokit meteor decode index guard")


@dataclass(frozen=True)
class GroupSpec:
    output_group: str
    algorithm: stegokit.StegoAlgorithm | None
    use_retrieval: bool = False
    asymmetric: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    record_name: str
    output_dir: str
    sample_size: int
    groups: tuple[str, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class PromptBudgetInfo:
    messages: list[dict[str, str]]
    prompt_tokens_before: int
    prompt_tokens_after: int
    prompt_trimmed: bool
    trimmed_history_message_count: int
    retrieval_tokens_before: int
    retrieval_tokens_after: int
    retrieval_trimmed: bool


@dataclass(frozen=True)
class RuntimeContext:
    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    special_tokens: set[str]
    temperature: float
    max_new_tokens: int
    top_k: int | None
    top_p: float | None
    precision: int
    base_seed: int
    window_sessions: int
    repeats: int
    retrieve_top_k: int
    retrieval_candidate_top_k: int
    max_prompt_tokens: int
    retrieval_max_tokens: int
    controlled_summary_note_max_tokens: int
    acf_k_values: tuple[int, ...]


GROUP_SPECS: dict[str, GroupSpec] = {
    "group1": GroupSpec(output_group="G1", algorithm=None),
    "group2": GroupSpec(output_group="G2", algorithm=stegokit.StegoAlgorithm.DISCOP),
    "group3": GroupSpec(output_group="G3", algorithm=stegokit.StegoAlgorithm.METEOR),
    "group4": GroupSpec(output_group="G4", algorithm=stegokit.StegoAlgorithm.ASYMMETRIC, asymmetric=True),
    "group5": GroupSpec(
        output_group="G5",
        algorithm=stegokit.StegoAlgorithm.ASYMMETRIC,
        use_retrieval=True,
        asymmetric=True,
    ),
    "group6": GroupSpec(
        output_group="G6",
        algorithm=stegokit.StegoAlgorithm.DISCOP,
        use_retrieval=True,
    ),
    "group7": GroupSpec(
        output_group="G7",
        algorithm=stegokit.StegoAlgorithm.METEOR,
        use_retrieval=True,
    ),
    "group8": GroupSpec(
        output_group="G8",
        algorithm=None,
        use_retrieval=True,
    ),
}

EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "realistic": ExperimentSpec(
        key="realistic",
        record_name="realistic_cognitive_asymmetry",
        output_dir=config.OUTPUT_V2_REALISTIC_DIR,
        sample_size=config.LONGMEMEVAL_REALISTIC_SAMPLE_SIZE,
        groups=("group1", "group2", "group3", "group4", "group5", "group6", "group7", "group8"),
        conditions=("no_drift",),
    ),
    "controlled": ExperimentSpec(
        key="controlled",
        record_name="controlled_cognitive_asymmetry",
        output_dir=config.OUTPUT_V2_CONTROLLED_DIR,
        sample_size=config.LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE,
        groups=("group2", "group3", "group4"),
        conditions=("no_drift", "drift_recent3"),
    ),
    "controlled_sweep": ExperimentSpec(
        key="controlled_sweep",
        record_name="controlled_drift_severity_sweep",
        output_dir=config.OUTPUT_V2_CONTROLLED_SWEEP_DIR,
        sample_size=config.LONGMEMEVAL_CONTROLLED_SWEEP_SAMPLE_SIZE,
        groups=("group2", "group3", "group4"),
        conditions=("no_drift",)
        + tuple(
            f"drift_recent{keep}"
            for keep in config.LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS
            if int(keep) > 0 and int(keep) < int(config.LONGMEMEVAL_WINDOW_SESSIONS)
        ),
    ),
    "controlled_summary": ExperimentSpec(
        key="controlled_summary",
        record_name="controlled_summary_asymmetry",
        output_dir=config.OUTPUT_V2_CONTROLLED_SUMMARY_DIR,
        sample_size=config.LONGMEMEVAL_CONTROLLED_SUMMARY_SAMPLE_SIZE,
        groups=("group2", "group3", "group4"),
        conditions=("summary_only_enc",),
    ),
}


def sanitize_message_text(text: str, special_tokens: set[str]) -> str:
    cleaned = text or ""
    for token in special_tokens:
        if token:
            cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    return cleaned.strip()


def set_trial_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_filename_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    fragment = fragment.strip("._")
    return fragment or "unknown"


def generate_random_bitstring(length: int, seed: int | str) -> str:
    if length < 0:
        raise ValueError("length must be >= 0")
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(length))


def build_benchmark_question_message(question: str) -> dict[str, str]:
    return {"role": "user", "content": str(question or "").strip()}


def build_base_messages(
    recent_sessions: list[list[dict[str, Any]]],
    question: str,
    *,
    retrieved_context: str = "",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": config.LONGMEMEVAL_QA_SYSTEM_PROMPT}]
    messages.extend(longmemeval_tools.session_to_messages(recent_sessions))
    if retrieved_context:
        messages.append({"role": "tool", "content": retrieved_context.strip()})
    messages.append(build_benchmark_question_message(question))
    return messages


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _chat_template_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(rendered, "get"):
        input_ids = rendered.get("input_ids")
        if input_ids is None:
            raise ValueError("apply_chat_template output has no input_ids")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
    elif isinstance(rendered, torch.Tensor):
        input_ids = rendered
    elif isinstance(rendered, (list, tuple)):
        input_ids = torch.tensor(rendered, dtype=torch.long)
    elif isinstance(rendered, str):
        input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
    else:
        raise TypeError(f"Unsupported apply_chat_template output type: {type(rendered)}")

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def count_prompt_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> int:
    return int(_chat_template_input_ids(tokenizer, messages).shape[-1])


def count_text_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> int:
    encoded = tokenizer(str(text or ""), add_special_tokens=False)
    return len(list(encoded.get("input_ids", []) or []))


def truncate_text_to_token_limit(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
) -> str:
    raw = str(text or "").strip()
    limit = max(0, int(max_tokens))
    if limit <= 0 or not raw:
        return ""

    encoded = tokenizer(raw, add_special_tokens=False)
    token_ids = list(encoded.get("input_ids", []) or [])
    if len(token_ids) <= limit:
        return raw

    truncated_ids = token_ids[:limit]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def apply_prompt_budget(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    max_prompt_tokens: int,
    retrieval_max_tokens: int = 0,
) -> PromptBudgetInfo:
    if not messages:
        return PromptBudgetInfo(
            messages=[],
            prompt_tokens_before=0,
            prompt_tokens_after=0,
            prompt_trimmed=False,
            trimmed_history_message_count=0,
            retrieval_tokens_before=0,
            retrieval_tokens_after=0,
            retrieval_trimmed=False,
        )

    raw_messages = [
        {
            "role": str(message.get("role", "")).strip(),
            "content": str(message.get("content", "")).strip(),
        }
        for message in messages
    ]
    prompt_tokens_before = count_prompt_tokens(tokenizer, raw_messages)

    system_message = dict(raw_messages[0])
    user_message = dict(raw_messages[-1])
    middle_messages = [dict(message) for message in raw_messages[1:-1]]

    tool_message: dict[str, str] | None = None
    history_messages: list[dict[str, str]] = []
    for message in middle_messages:
        if tool_message is None and message.get("role") == "tool":
            tool_message = dict(message)
        else:
            history_messages.append(dict(message))

    retrieval_tokens_before = 0
    retrieval_tokens_after = 0
    retrieval_trimmed = False
    current_tool_content = ""
    if tool_message is not None:
        current_tool_content = str(tool_message.get("content", "")).strip()
        retrieval_tokens_before = count_text_tokens(tokenizer, current_tool_content)
        retrieval_tokens_after = retrieval_tokens_before
        retrieval_limit = max(0, int(retrieval_max_tokens))
        if retrieval_limit > 0 and retrieval_tokens_before > retrieval_limit:
            current_tool_content = truncate_text_to_token_limit(tokenizer, current_tool_content, retrieval_limit)
            retrieval_tokens_after = count_text_tokens(tokenizer, current_tool_content)
            retrieval_trimmed = retrieval_tokens_after < retrieval_tokens_before

    def compose(history_start_index: int, tool_content: str) -> list[dict[str, str]]:
        composed = [dict(system_message)]
        composed.extend(dict(message) for message in history_messages[history_start_index:])
        if tool_message is not None and str(tool_content or "").strip():
            composed.append({"role": "tool", "content": str(tool_content).strip()})
        composed.append(dict(user_message))
        return composed

    max_tokens = max(0, int(max_prompt_tokens))
    trimmed_history_message_count = 0
    budgeted_messages = compose(0, current_tool_content)

    if max_tokens > 0:
        current_prompt_tokens = count_prompt_tokens(tokenizer, budgeted_messages)
        if current_prompt_tokens > max_tokens and history_messages:
            low = 0
            high = len(history_messages)
            while low < high:
                mid = (low + high) // 2
                candidate_messages = compose(mid, current_tool_content)
                if count_prompt_tokens(tokenizer, candidate_messages) <= max_tokens:
                    high = mid
                else:
                    low = mid + 1
            trimmed_history_message_count = low
            budgeted_messages = compose(trimmed_history_message_count, current_tool_content)
            current_prompt_tokens = count_prompt_tokens(tokenizer, budgeted_messages)

        if current_prompt_tokens > max_tokens and tool_message is not None and retrieval_tokens_after > 0:
            remaining_history_start = trimmed_history_message_count
            low = 0
            high = retrieval_tokens_after
            best_tool_content = ""
            while low <= high:
                mid = (low + high) // 2
                candidate_tool_content = truncate_text_to_token_limit(tokenizer, current_tool_content, mid)
                candidate_messages = compose(remaining_history_start, candidate_tool_content)
                if count_prompt_tokens(tokenizer, candidate_messages) <= max_tokens:
                    best_tool_content = candidate_tool_content
                    low = mid + 1
                else:
                    high = mid - 1

            current_tool_content = best_tool_content
            retrieval_tokens_after = count_text_tokens(tokenizer, current_tool_content)
            retrieval_trimmed = retrieval_trimmed or (retrieval_tokens_after < retrieval_tokens_before)
            budgeted_messages = compose(remaining_history_start, current_tool_content)
            current_prompt_tokens = count_prompt_tokens(tokenizer, budgeted_messages)

        if current_prompt_tokens > max_tokens:
            raise ValueError(
                f"Prompt budget exceeded after trimming: tokens={current_prompt_tokens} budget={max_tokens}"
            )

    prompt_tokens_after = count_prompt_tokens(tokenizer, budgeted_messages)
    prompt_trimmed = (
        prompt_tokens_after < prompt_tokens_before
        or trimmed_history_message_count > 0
        or retrieval_trimmed
    )
    return PromptBudgetInfo(
        messages=budgeted_messages,
        prompt_tokens_before=prompt_tokens_before,
        prompt_tokens_after=prompt_tokens_after,
        prompt_trimmed=prompt_trimmed,
        trimmed_history_message_count=trimmed_history_message_count,
        retrieval_tokens_before=retrieval_tokens_before,
        retrieval_tokens_after=retrieval_tokens_after,
        retrieval_trimmed=retrieval_trimmed,
    )


def generate_plain_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    special_tokens: set[str],
) -> dict[str, Any]:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    started_at = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generate_time_seconds = time.perf_counter() - started_at

    prompt_len = inputs["input_ids"].shape[1]
    generated_token_ids = output_ids[0, prompt_len:].tolist()
    text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    text = sanitize_message_text(text, special_tokens)
    average_entropy = entropy_tools.compute_average_entropy_for_generated_ids(
        model,
        tokenizer,
        messages,
        generated_token_ids,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return {
        "text": text,
        "generated_token_ids": generated_token_ids,
        "generated_token_count": len(generated_token_ids),
        "generate_time_seconds": float(generate_time_seconds),
        "average_entropy": average_entropy,
    }


def compute_bit_metrics(expected_bits: str, recovered_bits: str) -> tuple[int, int, float]:
    compared_bits_len = len(expected_bits)
    if compared_bits_len == 0:
        return 0, 0, 0.0

    recovered = str(recovered_bits or "")
    bit_errors = 0
    for idx, bit in enumerate(expected_bits):
        if idx >= len(recovered) or recovered[idx] != bit:
            bit_errors += 1
    ber = float(bit_errors / compared_bits_len)
    return compared_bits_len, bit_errors, ber


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _message_contents_to_text(messages: list[dict[str, Any]]) -> str:
    parts = [str(message.get("content", "")).strip() for message in messages if str(message.get("content", "")).strip()]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _keyword_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
        if len(token) >= 3
    }


def _compact_summary_sentence(text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return ""
    sentences = [piece.strip(" .") for piece in re.split(r"[.!?]+", normalized) if piece.strip(" .")]
    summary = sentences[0] if sentences else normalized
    summary = truncate_text_to_token_limit(tokenizer, summary, max_tokens)
    summary = summary.strip(" .")
    if not summary:
        return ""
    return summary + "."


def _chunk_session_messages(
    session: list[dict[str, Any]],
    *,
    chunk_messages: int,
    chunk_stride: int,
) -> list[dict[str, Any]]:
    normalized_messages = longmemeval_tools.session_to_messages([session])
    if not normalized_messages:
        return []

    size = max(1, int(chunk_messages))
    stride = max(1, int(chunk_stride))
    total = len(normalized_messages)
    if total <= size:
        return [
            {
                "message_start": 0,
                "message_end": total,
                "messages": normalized_messages,
            }
        ]

    starts = list(range(0, total - size + 1, stride))
    last_start = total - size
    if starts[-1] != last_start:
        starts.append(last_start)

    chunks: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start in starts:
        end = min(start + size, total)
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        chunks.append(
            {
                "message_start": start,
                "message_end": end,
                "messages": normalized_messages[start:end],
            }
        )
    return chunks


def _ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def build_controlled_summary_note(
    record: dict[str, Any],
    *,
    question: str,
    window_sessions: int,
    tokenizer: PreTrainedTokenizerBase,
    note_max_tokens: int,
) -> str:
    sessions = list(record.get("haystack_sessions", []) or [])
    keep = max(0, int(window_sessions))
    memory_pool_sessions = sessions[:-keep] if keep > 0 else sessions
    if not memory_pool_sessions:
        return ""

    question_terms = _keyword_terms(question)
    chunk_messages = int(config.LONGMEMEVAL_RETRIEVE_CHUNK_MESSAGES)
    chunk_stride = int(config.LONGMEMEVAL_RETRIEVE_CHUNK_STRIDE)
    candidates: list[tuple[int, int, int, str]] = []

    for session_index, session in enumerate(memory_pool_sessions):
        chunks = _chunk_session_messages(
            session,
            chunk_messages=chunk_messages,
            chunk_stride=chunk_stride,
        )
        for chunk in chunks:
            content = _message_contents_to_text(list(chunk.get("messages", []) or []))
            if not content:
                continue
            overlap = len(question_terms & _keyword_terms(content))
            recency = session_index
            candidates.append((overlap, recency, -len(content), content))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    best_content = candidates[0][3]
    summary = _compact_summary_sentence(best_content, tokenizer, max(8, int(note_max_tokens)))
    if not summary:
        return ""
    return f"{config.LONGMEMEVAL_CONTROLLED_SUMMARY_NOTE_PREFIX} {summary}"


def build_retrieved_reference_text(retrieval_hits: list[dict[str, Any]]) -> str:
    if not retrieval_hits:
        return ""

    blocks = [str(config.LONGMEMEVAL_RETRIEVAL_TOOL_PROMPT).strip()]
    for hit in retrieval_hits:
        blocks.append(
            "\n".join(
                [
                    f"Retrieved note {hit['rank']}",
                    f"session_id: {hit['session_id']}",
                    f"chunk_id: {hit['chunk_id']}",
                    f"message_span: {hit['message_start'] + 1}-{hit['message_end']}",
                    f"relevance_score: {hit['score']:.4f}",
                    f"embedding_score: {hit['embedding_score']:.4f}",
                    f"rerank_method: {hit['rerank_method']}",
                    "content:",
                    str(hit["content"]).strip(),
                ]
            ).strip()
        )
    return "\n\n".join(blocks).strip()


def build_retrieval_context(
    record: dict[str, Any],
    *,
    window_sessions: int,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    if top_k <= 0:
        return "", []

    sessions = list(record.get("haystack_sessions", []) or [])
    keep = max(0, int(window_sessions))
    memory_pool_sessions = sessions[:-keep] if keep > 0 else sessions
    chunk_messages = int(config.LONGMEMEVAL_RETRIEVE_CHUNK_MESSAGES)
    chunk_stride = int(config.LONGMEMEVAL_RETRIEVE_CHUNK_STRIDE)
    candidate_top_k = max(top_k, int(config.LONGMEMEVAL_RETRIEVE_CANDIDATE_TOP_K))

    entries: list[dict[str, Any]] = []
    for session_index, session in enumerate(memory_pool_sessions):
        session_id = longmemeval_tools.get_session_identifier(record, session_index)
        chunks = _chunk_session_messages(
            session,
            chunk_messages=chunk_messages,
            chunk_stride=chunk_stride,
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_text = _messages_to_text(list(chunk["messages"]))
            if not chunk_text:
                continue
            entries.append(
                {
                    "session_id": session_id,
                    "session_index": session_index,
                    "chunk_id": f"{session_id}#chunk{chunk_index}",
                    "message_start": int(chunk["message_start"]),
                    "message_end": int(chunk["message_end"]),
                    "content": chunk_text,
                }
            )

    if not entries:
        return "", []

    retrieval_index = session_retrieval.SessionRetrievalIndex.build([entry["content"] for entry in entries])
    scored_hits = retrieval_index.search(
        str(record.get("question", "")),
        top_k=top_k,
        candidate_top_k=candidate_top_k,
    )

    hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(scored_hits, start=1):
        entry = entries[int(hit["index"])]
        hits.append(
            {
                "rank": rank,
                "score": float(hit["score"]),
                "session_id": entry["session_id"],
                "session_index": entry["session_index"],
                "chunk_id": entry["chunk_id"],
                "message_start": entry["message_start"],
                "message_end": entry["message_end"],
                "content": entry["content"],
                "embedding_score": float(hit.get("embedding_score", hit["score"])),
                "rerank_method": str(hit.get("rerank_method", "remote_api")),
                "rerank_model": str(hit.get("rerank_model", "")),
            }
        )

    return build_retrieved_reference_text(hits), hits


def evaluate_prediction(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    return {
        "task_em": qa_metrics.best_exact_match(prediction, gold_answers),
        "task_f1": qa_metrics.best_token_f1(prediction, gold_answers),
    }


def _condition_keep_sessions(condition: str, recent_session_count: int) -> int:
    if condition in {"no_drift", "summary_only_enc"}:
        return max(0, int(recent_session_count))
    match = DRIFT_RECENT_PATTERN.match(str(condition or "").strip())
    if match:
        return min(max(1, int(match.group(1))), max(0, int(recent_session_count)))
    raise ValueError(f"Unsupported decode condition: {condition!r}")


def build_decode_messages_for_condition(
    question: str,
    recent_sessions: list[list[dict[str, Any]]],
    condition: str,
) -> list[dict[str, str]]:
    keep_sessions = _condition_keep_sessions(condition, len(recent_sessions))
    if condition in {"no_drift", "summary_only_enc"}:
        return build_base_messages(recent_sessions, question)
    trimmed_sessions = recent_sessions[-keep_sessions:] if keep_sessions > 0 else []
    return build_base_messages(trimmed_sessions, question)


def _group_acf_k_values(
    experiment: ExperimentSpec,
    spec: GroupSpec,
    runtime: RuntimeContext,
) -> tuple[int | None, ...]:
    if not spec.asymmetric:
        return (None,)
    values = tuple(int(value) for value in runtime.acf_k_values if int(value) > 0)
    if values:
        return values
    return (STEGO_SECURE_PARAMETER,)


def build_stego_extra(spec: GroupSpec, seed: int, *, acf_k: int | None = None) -> dict[str, Any]:
    if not spec.asymmetric:
        return {}
    secure_parameter = int(acf_k) if acf_k is not None else STEGO_SECURE_PARAMETER
    return {
        "seed": seed,
        "secure_parameter": secure_parameter,
        "func_type": STEGO_FUNC_TYPE,
    }


def build_prg(spec: GroupSpec, seed: int) -> Any | None:
    if spec.asymmetric:
        return None
    return stegokit.PRG.from_int_seed(seed)


def build_run_id(
    experiment: ExperimentSpec,
    group_name: str,
    question_id: str,
    condition: str,
    seed: int,
    *,
    acf_k: int | None = None,
) -> str:
    k_suffix = f".k{int(acf_k)}" if acf_k is not None else ""
    return f"{experiment.key}.{config.LONGMEMEVAL_SPLIT}.{question_id}.{group_name}{k_suffix}.{condition}.{seed}"


def build_output_path(
    output_root: Path,
    *,
    group_name: str,
    model_name: str,
    question_id: str,
    condition: str,
    seed: int,
    acf_k: int | None = None,
) -> Path:
    k_fragment = f"k{int(acf_k)}" if acf_k is not None else "base"
    filename = (
        f"{safe_filename_fragment(group_name)}-"
        f"{safe_filename_fragment(model_name)}-"
        f"{safe_filename_fragment(question_id)}-"
        f"{safe_filename_fragment(k_fragment)}-"
        f"{safe_filename_fragment(condition)}-"
        f"{seed}.json"
    )
    return output_root / group_name / filename


def write_json_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def should_skip_existing_output(
    path: Path,
    *,
    group_name: str,
    question_id: str,
    condition: str,
    seed: int,
    acf_k: int | None = None,
) -> bool:
    if not config.SKIP_EXISTING_OUTPUTS or not path.exists():
        return False
    k_suffix = f" k={int(acf_k)}" if acf_k is not None else ""
    print(
        f"[output] skip_existing group={group_name} "
        f"question_id={question_id} condition={condition} seed={seed}{k_suffix}"
    )
    return True


def prepare_group_output_dir(output_root: Path, group_name: str) -> Path:
    group_dir = output_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    existing_paths = sorted(group_dir.glob("*.json"))
    if config.SKIP_EXISTING_OUTPUTS:
        if existing_paths:
            print(f"[output] keep_existing group={group_name} files={len(existing_paths)}")
        return group_dir

    for path in existing_paths:
        path.unlink()
    if existing_paths:
        print(f"[output] cleared_existing group={group_name} files={len(existing_paths)}")
    return group_dir


def build_common_record_fields(
    *,
    experiment: ExperimentSpec,
    spec: GroupSpec,
    record: dict[str, Any],
    question_id: str,
    ground_truth: str | list[str],
    seed: int,
    runtime: RuntimeContext,
    retrieval_top_k: int,
    retrieval_candidate_top_k: int,
    retrieval_rerank_method: str,
    retrieval_rerank_model: str,
    retrieval_hit_count: int,
    retrieved_session_ids: list[Any],
    retrieved_chunk_ids: list[Any],
    retrieval_chars: int,
    encode_budget: PromptBudgetInfo,
    dataset_source: dict[str, str],
    acf_k: int | None = None,
) -> dict[str, Any]:
    record_fields: dict[str, Any] = {
        "experiment": experiment.record_name,
        "experiment_key": experiment.key,
        "split": config.LONGMEMEVAL_SPLIT,
        "question_id": question_id,
        "category": longmemeval_tools.get_record_category(record),
        "group": spec.output_group,
        "ground_truth": ground_truth,
        "seed": seed,
        "window_sessions": runtime.window_sessions,
        "retrieval_top_k": retrieval_top_k,
        "retrieval_candidate_top_k": retrieval_candidate_top_k,
        "retrieval_rerank_method": retrieval_rerank_method,
        "retrieval_rerank_model": retrieval_rerank_model,
        "retrieval_hit_count": retrieval_hit_count,
        "retrieved_session_ids": retrieved_session_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieval_chars": retrieval_chars,
        "max_prompt_tokens": runtime.max_prompt_tokens,
        "retrieval_max_tokens": runtime.retrieval_max_tokens if spec.use_retrieval else 0,
        "encode_prompt_tokens_before": encode_budget.prompt_tokens_before,
        "encode_prompt_tokens_after": encode_budget.prompt_tokens_after,
        "encode_prompt_trimmed": int(encode_budget.prompt_trimmed),
        "encode_trimmed_history_message_count": encode_budget.trimmed_history_message_count,
        "encode_retrieval_tokens_before": encode_budget.retrieval_tokens_before,
        "encode_retrieval_tokens_after": encode_budget.retrieval_tokens_after,
        "encode_retrieval_trimmed": int(encode_budget.retrieval_trimmed),
        **dataset_source,
    }
    if acf_k is not None:
        record_fields["acf_k"] = int(acf_k)
    return record_fields


def _load_runtime() -> RuntimeContext:
    model_path = config.get_model_path()
    model_name = config.get_model_label()
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
    return RuntimeContext(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        special_tokens=set(getattr(tokenizer, "all_special_tokens", []) or []),
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        top_k=int(config.TOP_K),
        top_p=None if float(config.TOP_P) <= 0 else float(config.TOP_P),
        precision=int(config.STEGO_PRECISION),
        base_seed=int(config.RANDOM_SEED),
        window_sessions=int(config.LONGMEMEVAL_WINDOW_SESSIONS),
        repeats=int(config.LONGMEMEVAL_REPEATS),
        retrieve_top_k=int(config.LONGMEMEVAL_RETRIEVE_TOP_K),
        retrieval_candidate_top_k=max(
            int(config.LONGMEMEVAL_RETRIEVE_TOP_K),
            int(config.LONGMEMEVAL_RETRIEVE_CANDIDATE_TOP_K),
        ),
        max_prompt_tokens=max(0, int(config.LONGMEMEVAL_MAX_PROMPT_TOKENS)),
        retrieval_max_tokens=max(0, int(config.LONGMEMEVAL_RETRIEVAL_MAX_TOKENS)),
        controlled_summary_note_max_tokens=max(0, int(config.LONGMEMEVAL_CONTROLLED_SUMMARY_NOTE_MAX_TOKENS)),
        acf_k_values=tuple(int(value) for value in config.LONGMEMEVAL_ACF_K_VALUES),
    )


def _resolve_groups(experiment: ExperimentSpec, group_names: list[str] | None) -> list[str]:
    allowed = set(experiment.groups)
    if not group_names:
        return list(experiment.groups)
    invalid = [name for name in group_names if name not in allowed]
    if invalid:
        raise ValueError(
            f"Groups {invalid} are not valid for experiment={experiment.key}. Allowed: {sorted(allowed)}"
        )
    return list(group_names)


def _sample_records(experiment: ExperimentSpec) -> list[dict[str, Any]]:
    return longmemeval_tools.sample_longmemeval_s(experiment.sample_size, config.RANDOM_SEED)


def _condition_inputs(
    experiment: ExperimentSpec,
    runtime: RuntimeContext,
    question: str,
    recent_sessions: list[list[dict[str, Any]]],
) -> list[tuple[str, PromptBudgetInfo]]:
    condition_inputs: list[tuple[str, PromptBudgetInfo]] = []
    for condition_name in experiment.conditions:
        decode_messages = build_decode_messages_for_condition(question, recent_sessions, condition_name)
        decode_budget = apply_prompt_budget(
            runtime.tokenizer,
            decode_messages,
            max_prompt_tokens=runtime.max_prompt_tokens,
            retrieval_max_tokens=0,
        )
        if decode_budget.prompt_trimmed:
            print(
                f"[decode] prompt_trim condition={condition_name} "
                f"{decode_budget.prompt_tokens_before}->{decode_budget.prompt_tokens_after} "
                f"history_drop={decode_budget.trimmed_history_message_count}"
            )
        condition_inputs.append((condition_name, decode_budget))
    return condition_inputs


def _run_plain_group(
    *,
    experiment: ExperimentSpec,
    spec: GroupSpec,
    group_dir_name: str,
    runtime: RuntimeContext,
    output_root: Path,
    record: dict[str, Any],
    question_id: str,
    gold_answers: list[str],
    common_fields: dict[str, Any],
    encode_messages: list[dict[str, str]],
    encode_budget: PromptBudgetInfo,
    condition_inputs: list[tuple[str, PromptBudgetInfo]],
    trial_seed: int,
    recent_session_count: int,
) -> None:
    output_path = build_output_path(
        output_root,
        group_name=group_dir_name,
        model_name=runtime.model_name,
        question_id=question_id,
        condition="no_drift",
        seed=trial_seed,
        acf_k=None,
    )
    if should_skip_existing_output(
        output_path,
        group_name=group_dir_name,
        question_id=question_id,
        condition="no_drift",
        seed=trial_seed,
        acf_k=None,
    ):
        return

    set_trial_seed(trial_seed)
    plain_result = generate_plain_reply(
        runtime.model,
        runtime.tokenizer,
        encode_messages,
        max_new_tokens=runtime.max_new_tokens,
        temperature=runtime.temperature,
        top_k=runtime.top_k,
        top_p=runtime.top_p,
        special_tokens=runtime.special_tokens,
    )
    prediction_metrics = evaluate_prediction(plain_result["text"], gold_answers)
    condition = "no_drift"
    no_drift_budget = condition_inputs[0][1] if condition_inputs else encode_budget
    run_record = {
        **common_fields,
        "run_id": build_run_id(
            experiment,
            spec.output_group,
            question_id,
            condition,
            trial_seed,
            acf_k=None,
        ),
        "condition": condition,
        "decode_recent_sessions_kept": _condition_keep_sessions(condition, recent_session_count),
        "assistant_answer": plain_result["text"],
        "generated_token_count": plain_result["generated_token_count"],
        "average_entropy": plain_result["average_entropy"],
        "generate_time_seconds": plain_result["generate_time_seconds"],
        "decode_prompt_tokens_before": no_drift_budget.prompt_tokens_before,
        "decode_prompt_tokens_after": no_drift_budget.prompt_tokens_after,
        "decode_prompt_trimmed": int(no_drift_budget.prompt_trimmed),
        "decode_trimmed_history_message_count": no_drift_budget.trimmed_history_message_count,
        **prediction_metrics,
    }
    write_json_record(output_path, run_record)


def _run_stego_group(
    *,
    experiment: ExperimentSpec,
    spec: GroupSpec,
    group_dir_name: str,
    runtime: RuntimeContext,
    output_root: Path,
    question_id: str,
    gold_answers: list[str],
    common_fields: dict[str, Any],
    encode_messages: list[dict[str, str]],
    condition_inputs: list[tuple[str, PromptBudgetInfo]],
    trial_seed: int,
    recent_session_count: int,
    acf_k: int | None,
) -> None:
    pending_conditions: list[tuple[str, PromptBudgetInfo, Path]] = []
    for condition, decode_budget in condition_inputs:
        output_path = build_output_path(
            output_root,
            group_name=group_dir_name,
            model_name=runtime.model_name,
            question_id=question_id,
            condition=condition,
            seed=trial_seed,
            acf_k=acf_k,
        )
        if should_skip_existing_output(
            output_path,
            group_name=group_dir_name,
            question_id=question_id,
            condition=condition,
            seed=trial_seed,
            acf_k=acf_k,
        ):
            continue
        pending_conditions.append((condition, decode_budget, output_path))
    if not pending_conditions:
        return

    stego_dispatcher = stegokit.StegoDispatcher(verbose=False)
    bit_seed = f"{experiment.key}:{trial_seed}:{question_id}:{spec.output_group}:k={acf_k}"
    bitstream = generate_random_bitstring(config.SECRET_BITS_LENGTH, seed=bit_seed)
    set_trial_seed(trial_seed)
    encode_context = stegokit.StegoEncodeContext(
        algorithm=spec.algorithm,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        secret_bits=bitstream,
        messages=encode_messages,
        max_new_tokens=runtime.max_new_tokens,
        temperature=runtime.temperature,
        top_k=runtime.top_k,
        top_p=runtime.top_p,
        precision=runtime.precision,
        prg=build_prg(spec, trial_seed),
        stop_on_eos=STEGO_STOP_ON_EOS,
        extra=build_stego_extra(spec, trial_seed, acf_k=acf_k),
    )
    stego_result = stego_dispatcher.dispatch_encode(encode_context)
    generated_token_ids = list(stego_result.generated_token_ids)
    generated_text = sanitize_message_text(stego_result.text, runtime.special_tokens)
    consumed_bits = int(stego_result.consumed_bits)
    embedded_bits = bitstream[:consumed_bits]
    average_entropy = entropy_tools.compute_average_entropy_for_generated_ids(
        runtime.model,
        runtime.tokenizer,
        encode_messages,
        generated_token_ids,
        temperature=runtime.temperature,
        top_k=runtime.top_k,
        top_p=runtime.top_p,
    )
    prediction_metrics = evaluate_prediction(generated_text, gold_answers)
    encode_time_seconds = float(getattr(stego_result, "encode_time_seconds", 0.0) or 0.0)
    embedding_capacity = float(getattr(stego_result, "embedding_capacity", 0.0) or 0.0)
    embedding_capacity *= EMBEDDING_CAPACITY_SCALE_PER_1K_TOKENS

    for condition, decode_budget, output_path in pending_conditions:
        decode_context = stegokit.StegoDecodeContext(
            algorithm=spec.algorithm,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            generated_token_ids=generated_token_ids,
            messages=decode_budget.messages,
            temperature=runtime.temperature,
            top_k=runtime.top_k,
            top_p=runtime.top_p,
            precision=runtime.precision,
            prg=build_prg(spec, trial_seed),
            max_bits=consumed_bits,
            extra=build_stego_extra(spec, trial_seed, acf_k=acf_k),
        )
        decode_time_seconds = 0.0
        recovered_bits = ""
        decode_error = ""
        try:
            decode_result = stego_dispatcher.dispatch_decode(decode_context)
            recovered_bits = str(decode_result.bits or "")[:consumed_bits]
            decode_time_seconds = float(getattr(decode_result, "decode_time_seconds", 0.0) or 0.0)
        except Exception as exc:
            decode_error = f"{exc.__class__.__name__}: {exc}"
            k_suffix = f" k={acf_k}" if acf_k is not None else ""
            print(
                f"[decode] error group={group_dir_name} condition={condition} "
                f"question_id={question_id} seed={trial_seed}{k_suffix} -> {decode_error}"
            )
        compared_bits_len, bit_errors, ber = compute_bit_metrics(embedded_bits, recovered_bits)
        decode_success = int(recovered_bits == embedded_bits)
        run_record = {
            **common_fields,
            "run_id": build_run_id(
                experiment,
                spec.output_group,
                question_id,
                condition,
                trial_seed,
                acf_k=acf_k,
            ),
            "condition": condition,
            "decode_recent_sessions_kept": _condition_keep_sessions(condition, recent_session_count),
            "secret_bits_budget": int(config.SECRET_BITS_LENGTH),
            "consumed_bits": consumed_bits,
            "assistant_answer": generated_text,
            "generated_token_count": len(generated_token_ids),
            "average_entropy": average_entropy,
            "embedding_capacity": embedding_capacity,
            "encode_time_seconds": encode_time_seconds,
            "decode_time_seconds": decode_time_seconds,
            "embedded_bits": embedded_bits,
            "recovered_bits": recovered_bits,
            "compared_bits_len": compared_bits_len,
            "bit_errors": bit_errors,
            "ber": ber,
            "decode_success": decode_success,
            "decode_prompt_tokens_before": decode_budget.prompt_tokens_before,
            "decode_prompt_tokens_after": decode_budget.prompt_tokens_after,
            "decode_prompt_trimmed": int(decode_budget.prompt_trimmed),
            "decode_trimmed_history_message_count": decode_budget.trimmed_history_message_count,
            **prediction_metrics,
        }
        if decode_error:
            run_record["decode_error"] = decode_error
        write_json_record(output_path, run_record)


def _run_single_group(
    *,
    experiment: ExperimentSpec,
    group_name: str,
    runtime: RuntimeContext,
    records: list[dict[str, Any]],
    dataset_source: dict[str, str],
) -> None:
    spec = GROUP_SPECS[group_name]
    output_root = PROJECT_ROOT / experiment.output_dir
    prepare_group_output_dir(output_root, group_name)
    group_k_values = _group_acf_k_values(experiment, spec, runtime)

    for record_idx, record in enumerate(records, start=1):
        question_id = str(record.get("question_id", f"sample_{record_idx}")).strip()
        if config.SKIP_EXISTING_OUTPUTS:
            expected_paths = [
                build_output_path(
                    output_root,
                    group_name=group_name,
                    model_name=runtime.model_name,
                    question_id=question_id,
                    condition=condition,
                    seed=runtime.base_seed + repeat_idx,
                    acf_k=acf_k,
                )
                for repeat_idx in range(runtime.repeats)
                for condition in experiment.conditions
                for acf_k in group_k_values
            ]
            if expected_paths and all(path.exists() for path in expected_paths):
                print(
                    f"[output] skip_existing_record group={group_name} "
                    f"question_id={question_id} files={len(expected_paths)}"
                )
                continue
        question = str(record.get("question", "")).strip()
        gold_answers = longmemeval_tools.get_gold_answers(record)
        ground_truth = record.get("answer", "")
        recent_sessions = longmemeval_tools.get_recent_sessions(record, runtime.window_sessions)

        retrieved_context = ""
        retrieval_hits: list[dict[str, Any]] = []
        encoder_note = ""
        if spec.use_retrieval and experiment.key == "realistic":
            retrieved_context, retrieval_hits = build_retrieval_context(
                record,
                window_sessions=runtime.window_sessions,
                top_k=runtime.retrieve_top_k,
            )
        elif experiment.key == "controlled_summary":
            encoder_note = build_controlled_summary_note(
                record,
                question=question,
                window_sessions=runtime.window_sessions,
                tokenizer=runtime.tokenizer,
                note_max_tokens=runtime.controlled_summary_note_max_tokens,
            )

        retrieved_session_ids = _ordered_unique([hit["session_id"] for hit in retrieval_hits])
        retrieved_chunk_ids = [hit["chunk_id"] for hit in retrieval_hits]
        retrieval_chars = len(retrieved_context or encoder_note)
        retrieval_top_k = runtime.retrieve_top_k if (spec.use_retrieval and experiment.key == "realistic") else 0
        retrieval_candidate_top_k = (
            runtime.retrieval_candidate_top_k if (spec.use_retrieval and experiment.key == "realistic") else 0
        )
        retrieval_rerank_method = (
            str(retrieval_hits[0].get("rerank_method", "remote_api"))
            if retrieval_hits
            else ("remote_api" if spec.use_retrieval and experiment.key == "realistic" else "none")
        )
        retrieval_rerank_model = (
            str(retrieval_hits[0].get("rerank_model", "")).strip()
            if retrieval_hits
            else (str(config.REMOTE_RERANK_MODEL or "").strip() if spec.use_retrieval and experiment.key == "realistic" else "")
        )

        raw_encode_messages = build_base_messages(
            recent_sessions,
            question,
            retrieved_context=(
                retrieved_context
                if spec.use_retrieval and experiment.key == "realistic"
                else encoder_note
            ),
        )
        encode_budget = apply_prompt_budget(
            runtime.tokenizer,
            raw_encode_messages,
            max_prompt_tokens=runtime.max_prompt_tokens,
            retrieval_max_tokens=(
                runtime.retrieval_max_tokens
                if (spec.use_retrieval and experiment.key == "realistic") or experiment.key == "controlled_summary"
                else 0
            ),
        )
        encode_messages = encode_budget.messages
        if encode_budget.prompt_trimmed:
            print(
                f"[{experiment.key}:{group_name}] prompt_trim encode question_id={question_id} "
                f"{encode_budget.prompt_tokens_before}->{encode_budget.prompt_tokens_after} "
                f"history_drop={encode_budget.trimmed_history_message_count} "
                f"retrieval={encode_budget.retrieval_tokens_before}->{encode_budget.retrieval_tokens_after}"
            )

        condition_inputs = _condition_inputs(experiment, runtime, question, recent_sessions)

        for repeat_idx in range(runtime.repeats):
            trial_seed = runtime.base_seed + repeat_idx
            print(
                f"[{experiment.key}:{group_name}] sample {record_idx}/{len(records)} "
                f"question_id={question_id} seed={trial_seed}"
            )
            for acf_k in group_k_values:
                if acf_k is not None:
                    print(f"[{experiment.key}:{group_name}] question_id={question_id} seed={trial_seed} acf_k={acf_k}")

                common_fields = build_common_record_fields(
                    experiment=experiment,
                    spec=spec,
                    record=record,
                    question_id=question_id,
                    ground_truth=ground_truth,
                    seed=trial_seed,
                    runtime=runtime,
                    retrieval_top_k=retrieval_top_k,
                    retrieval_candidate_top_k=retrieval_candidate_top_k,
                    retrieval_rerank_method=retrieval_rerank_method,
                    retrieval_rerank_model=retrieval_rerank_model,
                    retrieval_hit_count=len(retrieval_hits),
                    retrieved_session_ids=retrieved_session_ids,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_chars=retrieval_chars,
                    encode_budget=encode_budget,
                    dataset_source=dataset_source,
                    acf_k=acf_k,
                )
                if encoder_note:
                    common_fields["encoder_note_chars"] = len(encoder_note)
                    common_fields["encoder_note_present"] = 1

                try:
                    if spec.algorithm is None:
                        _run_plain_group(
                            experiment=experiment,
                            spec=spec,
                            group_dir_name=group_name,
                            runtime=runtime,
                            output_root=output_root,
                            record=record,
                            question_id=question_id,
                            gold_answers=gold_answers,
                            common_fields=common_fields,
                            encode_messages=encode_messages,
                            encode_budget=encode_budget,
                            condition_inputs=condition_inputs,
                            trial_seed=trial_seed,
                            recent_session_count=len(recent_sessions),
                        )
                    else:
                        _run_stego_group(
                            experiment=experiment,
                            spec=spec,
                            group_dir_name=group_name,
                            runtime=runtime,
                            output_root=output_root,
                            question_id=question_id,
                            gold_answers=gold_answers,
                            common_fields=common_fields,
                            encode_messages=encode_messages,
                            condition_inputs=condition_inputs,
                            trial_seed=trial_seed,
                            recent_session_count=len(recent_sessions),
                            acf_k=acf_k,
                        )
                finally:
                    cleanup_cuda_memory()


def run_v2_experiment(experiment_key: str, group_names: list[str] | None = None) -> None:
    if experiment_key not in EXPERIMENT_SPECS:
        raise ValueError(f"Unsupported experiment: {experiment_key}")

    _patch_stegokit_meteor_decode_index_error()

    experiment = EXPERIMENT_SPECS[experiment_key]
    groups = _resolve_groups(experiment, group_names)
    records = _sample_records(experiment)
    dataset_source = longmemeval_tools.describe_longmemeval_source()
    runtime = _load_runtime()

    try:
        for group_name in groups:
            _run_single_group(
                experiment=experiment,
                group_name=group_name,
                runtime=runtime,
                records=records,
                dataset_source=dataset_source,
            )
    finally:
        cleanup_cuda_memory()


def run_v2_group(group_name: str, experiment_key: str = "realistic") -> None:
    run_v2_experiment(experiment_key, [group_name])


def run_controlled_experiment_suite() -> None:
    for experiment_key in ("controlled", "controlled_summary", "controlled_sweep"):
        print(f"[controlled-suite] start experiment={experiment_key}")
        run_v2_experiment(experiment_key)
