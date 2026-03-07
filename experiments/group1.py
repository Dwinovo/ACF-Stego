from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.agent.remote_agent import RemoteAgent
from core.tools import starter_dataset

def sample_conversation_starters(sample_size: int, seed: int) -> list[tuple[int, str]]:
    return starter_dataset.sample_absolute_id_starters(sample_size, seed)


def trim_recent_rounds(messages: list[dict[str, str]], window_rounds: int) -> None:
    if not messages:
        return
    max_turn_messages = max(window_rounds * 2, 0)
    has_system = messages[0].get("role") == "system"
    prefix = messages[:1] if has_system else []
    body = messages[1:] if has_system else messages[:]
    if len(body) > max_turn_messages:
        body = body[-max_turn_messages:]
    messages[:] = [*prefix, *body]


def contexts_equal_ignoring_system(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    def normalize(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "")).lower()
            if role == "system":
                continue
            normalized.append(
                {
                    "role": role,
                    "content": str(message.get("content", "")),
                }
            )
        return normalized

    return normalize(left) == normalize(right)


def sanitize_message_text(text: str, special_tokens: set[str]) -> str:
    cleaned = text or ""
    for token in special_tokens:
        if token:
            cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    return cleaned.strip()


def generate_local_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    special_tokens: set[str],
) -> tuple[str, int, float, float]:
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
    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_text = sanitize_message_text(generated_text, special_tokens)

    average_entropy = 0.0
    if generated_ids:
        # Entropy is computed post-generation from logits only, so sampling behavior is unchanged.
        with torch.no_grad():
            logits = model(output_ids).logits[0]

        per_step_entropy: list[float] = []
        log2 = torch.log(torch.tensor(2.0, device=logits.device, dtype=logits.dtype))
        full_vocab_size = logits.shape[-1]
        for step_idx in range(len(generated_ids)):
            step_logits = logits[prompt_len - 1 + step_idx]
            probs = torch.softmax(step_logits / max(float(temperature), 1e-8), dim=-1)

            if top_k is not None and top_k > 0 and top_k < full_vocab_size:
                probs, _ = torch.topk(probs, k=top_k)

            if top_p is not None and 0 < top_p < 1:
                sorted_probs, _ = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                keep = cumsum <= top_p
                if keep.numel() > 0:
                    keep[0] = True
                probs = sorted_probs[keep]

            probs = probs / probs.sum()
            entropy = -(probs * (torch.log(probs) / log2)).sum()
            entropy_value = float(entropy.item())
            if entropy_value == entropy_value:
                per_step_entropy.append(entropy_value)

        average_entropy = float(sum(per_step_entropy) / len(per_step_entropy)) if per_step_entropy else 0.0

    return generated_text, len(generated_ids), float(generate_time_seconds), average_entropy


def build_dialog_metrics(dialog_record: dict[str, Any]) -> dict[str, Any]:
    rounds = dialog_record.get("rounds", [])
    total_generated_token_count = int(
        sum(int(round_item.get("generated_token_count", 0) or 0) for round_item in rounds)
    )
    weighted_entropy_sum = float(
        sum(
            float(round_item.get("average_entropy", 0.0) or 0.0)
            * int(round_item.get("generated_token_count", 0) or 0)
            for round_item in rounds
        )
    )
    total_generate_time_seconds = float(
        sum(float(round_item.get("generate_time_seconds", 0.0) or 0.0) for round_item in rounds)
    )

    if total_generated_token_count > 0:
        weighted_average_entropy = weighted_entropy_sum / total_generated_token_count
        average_generate_time_per_token_seconds = total_generate_time_seconds / total_generated_token_count
    else:
        weighted_average_entropy = 0.0
        average_generate_time_per_token_seconds = 0.0

    return {
        "total_generated_token_count": total_generated_token_count,
        "weighted_average_entropy": float(weighted_average_entropy),
        "total_generate_time_seconds": total_generate_time_seconds,
        "average_generate_time_per_token_seconds": float(average_generate_time_per_token_seconds),
    }


def main() -> None:
    model_path = config.ModelEnum.QWEN2_5_7B_INSTRUCT.value
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
    model_name = Path(str(model_path).rstrip("/\\")).name
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

    starters = sample_conversation_starters(config.SAMPLE_SIZE, config.RANDOM_SEED)

    temperature = config.TEMPERATURE
    max_new_tokens = config.MAX_NEW_TOKENS
    top_k = int(config.TOP_K)
    top_p = float(config.TOP_P)
    top_p_value = None if top_p <= 0 else top_p
    seed = config.RANDOM_SEED

    remote_agent = RemoteAgent(
        model_name=config.REMOTE_AGENT,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    data_dir = PROJECT_ROOT / config.OUTPUT_DIR / "group1"
    data_dir.mkdir(parents=True, exist_ok=True)

    for starter_pos, (starter_id, starter) in enumerate(starters, start=1):
        for trial in range(1, config.REPEATS_PER_STARTER + 1):
            dialog_record: dict[str, Any] = {
                "time": int(time.time()),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_token": max_new_tokens,
                "seed": seed,
                "remote_agent": config.REMOTE_AGENT,
                "rounds_per_dialog": config.ROUNDS_PER_DIALOGUE,
                "window_size": config.WINDOW_ROUNDS,
                "model": model_path,
                "starter": starter,
                "rounds": [],
            }
            trial_output_path = data_dir / f"group1-{model_name}-{config.WINDOW_ROUNDS}-{starter_id}-{trial}.json"

            try:
                remote_agent_messages: list[dict[str, str]] = []
                local_agent_messages: list[dict[str, str]] = []
                remote_agent_messages.append({"role": "system", "content": config.USER_AGENT_SYSTEM_PROMPT})
                local_agent_messages.append({"role": "system", "content": config.ASSISTANT_AGENT_SYSTEM_PROMPT})

                starter_message = {"role": "user", "content": starter}
                remote_agent_messages.append(starter_message.copy())
                local_agent_messages.append(starter_message.copy())
                print(
                    f"[starter {starter_pos}/{len(starters)} abs_id={starter_id} | "
                    f"trial {trial}/{config.REPEATS_PER_STARTER}]"
                )

                for round_idx in range(1, config.ROUNDS_PER_DIALOGUE + 1):
                    print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Starting round...")
                    is_same_context = contexts_equal_ignoring_system(remote_agent_messages, local_agent_messages)
                    assistant_text, generated_token_count, generate_time_seconds, average_entropy = generate_local_reply(
                        model,
                        tokenizer,
                        local_agent_messages.copy(),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p_value,
                        special_tokens=special_tokens,
                    )
                    print(
                        f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] "
                        f"Generated assistant text with {generated_token_count} tokens."
                    )

                    carrier_assistant_message = {"role": "assistant", "content": assistant_text}
                    remote_agent_messages.append(carrier_assistant_message.copy())
                    local_agent_messages.append(carrier_assistant_message.copy())

                    remote_reply = remote_agent.invoke(messages=remote_agent_messages, temperature=temperature)
                    remote_reply = sanitize_message_text(remote_reply, special_tokens)
                    remote_reply_message = {"role": "user", "content": remote_reply}
                    remote_agent_messages.append(remote_reply_message.copy())
                    local_agent_messages.append(remote_reply_message.copy())

                    trim_recent_rounds(remote_agent_messages, config.WINDOW_ROUNDS)
                    trim_recent_rounds(local_agent_messages, config.WINDOW_ROUNDS)

                    dialog_record["rounds"].append(
                        {
                            "assistant": assistant_text,
                            "user": remote_reply,
                            "generated_token_count": generated_token_count,
                            "average_entropy": average_entropy,
                            "generate_time_seconds": generate_time_seconds,
                            "is_same_context": is_same_context,
                        }
                    )
                    if len(dialog_record["rounds"]) >= config.ROUNDS_PER_DIALOGUE:
                        dialog_record["metrics"] = build_dialog_metrics(dialog_record)
                    with trial_output_path.open("w", encoding="utf-8") as f:
                        json.dump(dialog_record, f, ensure_ascii=False, indent=2)
            finally:
                if len(dialog_record.get("rounds", [])) >= config.ROUNDS_PER_DIALOGUE:
                    dialog_record["metrics"] = build_dialog_metrics(dialog_record)
                else:
                    dialog_record.pop("metrics", None)
                with trial_output_path.open("w", encoding="utf-8") as f:
                    json.dump(dialog_record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
