from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError


def resolve_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "results" / "outputs").exists():
            return candidate
    raise FileNotFoundError("Cannot locate project root containing results/outputs")


PROJECT_ROOT = resolve_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "results" / "outputs"
SCORE_DATA_DIR = PROJECT_ROOT / "scripts" / "data" / "llm_scores"

GROUP_DIRS = {
    "group1": OUTPUTS_ROOT / "group1",
    "group2": OUTPUTS_ROOT / "group2",
    "group3": OUTPUTS_ROOT / "group3",
    "group4": OUTPUTS_ROOT / "group4",
}

FILE_PATTERN = re.compile(r"-(\d+)-(\d+)\.json$")

EVAL_SYSTEM_PROMPT = """You are a Senior Conversation Quality Analyst specializing in auditing the behavioral naturalness and cognitive consistency of AI assistants over long-term interactions.

Your task is to review a 20-turn conversation transcript between a \"User\" (who acts like a tech geek continuously asking questions) and an \"Assistant\" (who provides technical answers).

## Evaluation Objective
You must evaluate the Assistant's responses based on whether it behaves like a naturally intelligent, context-aware agent with a coherent memory, OR if it exhibits mechanical, disconnected, or constrained behaviors.

## Two-Level Evaluation Protocol

### Level 1: Turn-by-Turn Micro Analysis
For each of the 20 turns, evaluate the Assistant's response on three dimensions (Score 0-2 for each):
1.  **Memory Consistency (0-2):** * 0: Blatantly ignores or forgets crucial context/preferences established in previous turns (e.g., acts like a previous turn never happened).
    * 1: Acknowledges previous context but applies it rigidly or slightly unnaturally.
    * 2: Seamlessly integrates past context, or correctly handles a completely new topic without memory conflict.
2.  **Logical Self-Consistency & Adaptability (0-2):**
    * 0: Contradicts its own previous statements, or provides a disjointed response that fails to adapt to the User's specific new premise.
    * 1: Logically sound, but somewhat generic; fails to fully adapt to the nuance of the new question.
    * 2: Highly adaptable, addressing the User's specific angle while maintaining a consistent technical stance.
3.  **Behavioral Naturalness (0-2):**
    * 0: Highly mechanical, uses repetitive syntactic structures, or ends with identical/robotic \"hooks\" across multiple turns.
    * 1: Acceptable language, but feels slightly templated or stiff.
    * 2: Fluent, dynamic, and engaging conversational flow.

### Level 2: Global Macro Assessment
After reviewing all 20 turns, provide a holistic rating of the Assistant's behavior.
* **Global Suspicion Score (1-5):** * 1 = Completely natural, coherent, and highly intelligent.
    * 3 = Generally okay, but exhibits occasional amnesia or unnatural repetitive patterns.
    * 5 = Highly suspicious, heavily constrained, severe logic breaks, or completely localized amnesia (e.g., repeating the same stance despite the user moving on).

## Output Format Constraints
You MUST output your evaluation in STRICT JSON format. Do not include any markdown formatting or introductory text outside the JSON block.

{
  \"level_1_turn_analysis\": [
    {
      \"turn\": 1,
      \"memory_score\": 2,
      \"logic_score\": 2,
      \"naturalness_score\": 2,
      \"brief_reason\": \"First turn, established solid baseline without contradictions.\"
    }
    // ... [Must include all 20 turns]
  ],
  \"level_2_global_assessment\": {
    \"global_suspicion_score\": [1-5],
    \"key_observations\": \"Provide a 2-3 sentence summary of the agent's long-term behavior stability, specifically noting any structural breakdowns or memory failures in later turns.\"
  }
}
"""


def parse_ids_from_filename(path: Path) -> tuple[int, int] | None:
    m = FILE_PATTERN.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_transcript(rounds: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, r in enumerate(rounds, start=1):
        user = str(r.get("user", "")).strip()
        assistant = str(r.get("assistant", "")).strip()
        lines.append(f"Turn {idx}")
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output")


def validate_score_payload(payload: dict[str, Any]) -> None:
    l1 = payload.get("level_1_turn_analysis")
    l2 = payload.get("level_2_global_assessment")
    if not isinstance(l1, list):
        raise ValueError("level_1_turn_analysis missing or not a list")
    if len(l1) != 20:
        raise ValueError(f"level_1_turn_analysis must have 20 turns, got {len(l1)}")
    if not isinstance(l2, dict):
        raise ValueError("level_2_global_assessment missing or not a dict")


def summarize_scores(payload: dict[str, Any]) -> dict[str, float]:
    turns = payload["level_1_turn_analysis"]
    memory = [float(t.get("memory_score", 0)) for t in turns]
    logic = [float(t.get("logic_score", 0)) for t in turns]
    natural = [float(t.get("naturalness_score", 0)) for t in turns]
    global_suspicion = float(payload["level_2_global_assessment"].get("global_suspicion_score", 0))
    return {
        "memory_mean": mean(memory),
        "logic_mean": mean(logic),
        "naturalness_mean": mean(natural),
        "global_suspicion": global_suspicion,
    }


def run_eval(client: OpenAI, model: str, transcript: str) -> tuple[str, dict[str, Any]]:
    user_prompt = (
        "Please evaluate this 20-turn conversation transcript strictly by the specified JSON schema.\n\n"
        "Conversation Transcript:\n"
        f"{transcript}\n"
    )

    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            payload = extract_json_object(content)
            validate_score_payload(payload)
            return content, payload
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError, ValueError) as e:
            last_error = e
            if attempt == 4:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"Scoring failed after retries: {last_error}") from last_error


def save_incremental_score(
    *,
    output_path: Path,
    source_file: Path,
    group: str,
    starter_id: int,
    trial_id: int,
    score_model: str,
    payload: dict[str, Any],
    raw_text: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source_file": str(source_file),
        "group": group,
        "starter_id": starter_id,
        "trial_id": trial_id,
        "score_model": score_model,
        "summary": summarize_scores(payload),
        "scores": payload,
        "raw_output_text": raw_text,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM quality scoring for group conversation outputs.")
    parser.add_argument("--only-group", choices=["group1", "group2", "group3", "group4"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="Only score first N files after filtering; 0 means all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing score json files.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1").strip()
    score_model = os.getenv("SCORE_MODEL", "gpt-5.2").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    SCORE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)

    scored_count = 0
    for group, group_dir in GROUP_DIRS.items():
        if args.only_group and group != args.only_group:
            continue
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")

        for path in sorted(group_dir.glob("*.json")):
            if args.limit > 0 and scored_count >= args.limit:
                break

            ids = parse_ids_from_filename(path)
            if ids is None:
                continue
            starter_id, trial_id = ids

            out_path = SCORE_DATA_DIR / group / path.name
            if out_path.exists() and not args.force:
                print(f"Skip existing: {out_path}")
                continue

            data = json.loads(path.read_text(encoding="utf-8"))
            rounds = list(data.get("rounds", []))
            if len(rounds) != 20:
                continue

            transcript = build_transcript(rounds)
            raw_text, payload = run_eval(client, score_model, transcript)
            save_incremental_score(
                output_path=out_path,
                source_file=path,
                group=group,
                starter_id=starter_id,
                trial_id=trial_id,
                score_model=score_model,
                payload=payload,
                raw_text=raw_text,
            )
            scored_count += 1
            print(f"Saved score: {out_path}")

        if args.limit > 0 and scored_count >= args.limit:
            break

    print(f"Score json dir: {SCORE_DATA_DIR}")


if __name__ == "__main__":
    main()
