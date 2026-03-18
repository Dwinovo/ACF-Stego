from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.tools import analysis_tools
from core.tools import longmemeval_tools

EXPERIMENT_OUTPUTS = {
    "realistic": PROJECT_ROOT / config.OUTPUT_V2_REALISTIC_DIR,
    "controlled": PROJECT_ROOT / config.OUTPUT_V2_CONTROLLED_DIR,
}

ALLOWED_GROUPS = {
    "realistic": {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"},
    "controlled": {"G2", "G3", "G4"},
}

ALLOWED_CONDITIONS = {
    "realistic": {"no_drift"},
    "controlled": {"no_drift", "drift_recent3"},
}

JUDGE_SYSTEM_PROMPT = """
You are a strict semantic evaluator for question answering.

Your job is to judge whether the assistant's answer is semantically equivalent to the gold answer.

Scoring rubric:
- 2 = semantically correct and equivalent to the gold answer.
- 1 = partially correct, incomplete, ambiguous, or contains the right clue but also extra uncertainty or conflict.
- 0 = incorrect, contradicted, unsupported, or not an answer to the question.

Rules:
1. Judge semantic correctness, not style.
2. Do not penalize brevity or verbosity by itself.
3. If the assistant gives the correct answer inside a longer natural sentence, that can still be scored 2.
4. If the assistant includes a correct answer but also adds a material contradiction, do not score 2.
5. If multiple gold answers are provided, treat any semantically equivalent one as correct.
6. Do not use outside world knowledge. Judge only from the question, gold answer(s), and assistant response.
7. First produce a short reason, then decide the score.
8. Output JSON only.

Required JSON format:
{
  "reason": "short reason",
  "score": 0,
  "correct": 0
}
""".strip()
TIMEOUT_ERROR_CLASS_NAMES = {
    "APITimeoutError",
    "ConnectTimeout",
    "PoolTimeout",
    "ReadTimeout",
    "TimeoutError",
    "TimeoutException",
    "WriteTimeout",
}
TIMEOUT_RETRY_SLEEP_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score V2 experiment outputs with a remote LLM judge.")
    parser.add_argument("--experiment", choices=("realistic", "controlled", "all"), default="realistic")
    parser.add_argument("--only-group", choices=[f"group{i}" for i in range(1, 9)], default=None)
    parser.add_argument("--condition", choices=("no_drift", "drift_recent3"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to score.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing LLM judge fields instead of overwriting them.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_client() -> OpenAI:
    if not str(config.OPENAI_API_KEY or "").strip():
        raise ValueError("OPENAI_API_KEY is required for remote LLM judging.")
    if not str(config.LLM_JUDGE_MODEL or "").strip():
        raise ValueError("LLM_JUDGE_MODEL is required for remote LLM judging.")

    client_kwargs: dict[str, Any] = {
        "api_key": config.OPENAI_API_KEY,
        "timeout": float(config.LLM_JUDGE_TIMEOUT_SECONDS),
    }
    if str(config.OPENAI_BASE_URL or "").strip():
        client_kwargs["base_url"] = config.OPENAI_BASE_URL
    return OpenAI(**client_kwargs)


def extract_first_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("LLM judge returned empty content.")

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1].strip()

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected judge JSON object, got {type(parsed)}")
    return parsed


def normalize_gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip()
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")


def extract_response_text(response: Any) -> str:
    choices = list(getattr(response, "choices", []) or [])
    if not choices:
        raise ValueError("LLM judge response has no choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM judge response has no message.")

    return _message_content_to_text(getattr(message, "content", ""))


def build_question_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    try:
        records = longmemeval_tools.load_longmemeval_s()
    except Exception as exc:
        print(f"Warning: failed to load LongMemEval source records for question lookup: {exc}")
        return lookup

    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        if question_id:
            lookup[question_id] = record
    return lookup


def build_user_prompt(run_record: dict[str, Any], source_record: dict[str, Any] | None) -> str:
    question = ""
    if source_record is not None:
        question = str(source_record.get("question", "")).strip()

    assistant_answer = str(run_record.get("assistant_answer", "")).strip()
    gold_answers = normalize_gold_answers(run_record.get("ground_truth"))
    if not gold_answers and source_record is not None:
        gold_answers = longmemeval_tools.get_gold_answers(source_record)

    payload = {
        "question_id": str(run_record.get("question_id", "")).strip(),
        "experiment": str(run_record.get("experiment", "")).strip(),
        "condition": str(run_record.get("condition", "")).strip(),
        "question": question,
        "gold_answers": gold_answers,
        "assistant_response": assistant_answer,
    }
    return (
        "Evaluate the assistant response using the rubric from the system prompt.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("LLM judge JSON missing non-empty 'reason'.")

    try:
        score = int(payload.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM judge JSON has invalid 'score': {payload.get('score')!r}") from exc
    if score not in {0, 1, 2}:
        raise ValueError(f"LLM judge score must be 0, 1, or 2, got {score}")

    return {
        "llm_judge_reason": reason,
        "llm_judge_score": score,
        "llm_judge_correct": 1 if score == 2 else 0,
    }


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_timeout_error(exc: BaseException) -> bool:
    for err in _iter_exception_chain(exc):
        if err.__class__.__name__ in TIMEOUT_ERROR_CLASS_NAMES:
            return True
        if "timeout" in str(err).strip().lower():
            return True
    return False


def request_judge_completion(
    client: OpenAI,
    *,
    run_id: str,
    messages: list[dict[str, str]],
) -> Any:
    retry_count = 0
    while True:
        try:
            return client.chat.completions.create(
                model=config.LLM_JUDGE_MODEL,
                messages=messages,
                temperature=float(config.LLM_JUDGE_TEMPERATURE),
                max_tokens=int(config.LLM_JUDGE_MAX_TOKENS),
            )
        except Exception as exc:
            if not is_timeout_error(exc):
                raise
            retry_count += 1
            print(f"[retry-timeout] {run_id} retry={retry_count}")
            time.sleep(TIMEOUT_RETRY_SLEEP_SECONDS)


def judge_record(
    client: OpenAI,
    run_record: dict[str, Any],
    source_record: dict[str, Any] | None,
    *,
    run_id: str,
) -> dict[str, Any]:
    response = request_judge_completion(
        client,
        run_id=run_id,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(run_record, source_record)},
        ],
    )
    content = extract_response_text(response)
    payload = extract_first_json_object(content)
    parsed = parse_judge_payload(payload)
    parsed.update(
        {
            "llm_judge_model": str(config.LLM_JUDGE_MODEL).strip(),
            "llm_judge_prompt_version": str(config.LLM_JUDGE_PROMPT_VERSION).strip(),
            "llm_judge_temperature": float(config.LLM_JUDGE_TEMPERATURE),
            "llm_judge_scored_at": utc_now_iso(),
        }
    )
    return parsed


def should_skip_record(run_record: dict[str, Any], *, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    existing_score = run_record.get("llm_judge_score")
    existing_model = str(run_record.get("llm_judge_model", "")).strip()
    existing_prompt_version = str(run_record.get("llm_judge_prompt_version", "")).strip()
    return (
        existing_score is not None
        and existing_model == str(config.LLM_JUDGE_MODEL).strip()
        and existing_prompt_version == str(config.LLM_JUDGE_PROMPT_VERSION).strip()
    )


def selected_experiments(experiment: str) -> list[str]:
    if experiment == "all":
        return ["realistic", "controlled"]
    return [experiment]


def effective_condition(args: argparse.Namespace) -> str | None:
    if args.condition is not None:
        return args.condition
    if args.experiment == "realistic":
        return "no_drift"
    return None


def iter_record_paths(experiments: list[str], only_group: str | None, condition: str | None) -> list[Path]:
    groups = [only_group] if only_group else None
    paths: list[Path] = []
    for experiment in experiments:
        paths.extend(analysis_tools.iter_record_paths(EXPERIMENT_OUTPUTS[experiment], groups=groups))
    filtered: list[Path] = []
    for path in paths:
        record = analysis_tools.load_json_record(path)
        if record is None:
            continue
        record_group = str(record.get("group", "")).strip()
        record_condition = str(record.get("condition", "no_drift")).strip()
        record_experiment = str(record.get("experiment_key", "")).strip() or (
            "realistic" if "realistic" in str(record.get("experiment", "")) else "controlled"
        )
        if record_experiment not in experiments:
            continue
        if record_group not in ALLOWED_GROUPS[record_experiment]:
            continue
        if record_condition not in ALLOWED_CONDITIONS[record_experiment]:
            continue
        if condition is not None and record_condition != condition:
            continue
        filtered.append(path)
    return filtered


def write_json_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(args.experiment)
    condition = effective_condition(args)
    paths = iter_record_paths(experiments, args.only_group, condition)
    if args.limit is not None:
        paths = paths[: max(0, int(args.limit))]

    if not paths:
        print("No V2 records found to score.")
        return

    client = build_client()
    question_lookup = build_question_lookup()
    scored = 0
    skipped = 0
    failed = 0

    for index, path in enumerate(paths, start=1):
        run_record = analysis_tools.load_json_record(path)
        if run_record is None:
            print(f"[skip {index}/{len(paths)}] invalid JSON record: {path}")
            skipped += 1
            continue
        if should_skip_record(run_record, skip_existing=bool(args.skip_existing)):
            print(f"[skip {index}/{len(paths)}] already scored: {run_record.get('run_id', path.name)}")
            skipped += 1
            continue

        question_id = str(run_record.get("question_id", "")).strip()
        source_record = question_lookup.get(question_id)
        run_id = str(run_record.get("run_id", path.name)).strip()

        try:
            judge_fields = judge_record(client, run_record, source_record, run_id=run_id)
        except Exception as exc:
            run_record["llm_judge_error"] = str(exc)
            write_json_record(path, run_record)
            print(f"[fail {index}/{len(paths)}] {run_id}: {exc}")
            failed += 1
            continue

        run_record.update(judge_fields)
        run_record.pop("llm_judge_error", None)
        write_json_record(path, run_record)
        print(
            f"[ok {index}/{len(paths)}] {run_id} "
            f"score={judge_fields['llm_judge_score']} "
            f"correct={judge_fields['llm_judge_correct']}"
        )
        scored += 1

    print(
        f"LLM judge completed: scored={scored} skipped={skipped} failed={failed} "
        f"experiments={experiments} condition={condition or 'all'} "
        f"model={config.LLM_JUDGE_MODEL} prompt_version={config.LLM_JUDGE_PROMPT_VERSION}"
    )


if __name__ == "__main__":
    main()
