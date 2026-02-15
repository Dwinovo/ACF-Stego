from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import stegokit
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.agent.remote_agent import RemoteAgent
from core.tools import mem0_tools

# Stego params (group4-local)
STEGO_ALGORITHM = stegokit.StegoAlgorithm.ASYMMETRIC
STEGO_TOP_K = 50
STEGO_TOP_P = None
STEGO_PRECISION = 52
STEGO_STOP_ON_EOS = True
STEGO_SECURE_PARAMETER = 32
STEGO_FUNC_TYPE = 0
LOCAL_FLUSH_ROUNDS = 5
MEMORY_SEARCH_LIMIT = 3


def sample_conversation_starters(sample_size: int, seed: int) -> list[str]:
    dataset = load_dataset("Langame/conversation-starters", split="train")
    all_starters = [str(x).strip() for x in dataset["prompt"] if str(x).strip()]
    target_size = min(sample_size, len(all_starters))
    rng = random.Random(seed)
    return rng.sample(all_starters, k=target_size)


def generate_random_bitstring(length: int, seed: int | str) -> str:
    if length < 0:
        raise ValueError("length must be >= 0")
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(length))


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


def sanitize_message_text(text: str, special_tokens: set[str]) -> str:
    cleaned = text or ""
    for token in special_tokens:
        if token:
            cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)
    return cleaned.strip()

def persist_local_messages_to_mem0(
    local_messages: list[dict[str, Any]],
    memory: Any,
    *,
    user_id: str,
) -> int:
    saved_count = 0
    for message in local_messages[1:]:
        role = str(message.get("role", "")).lower()
        if role not in {"assistant", "user"}:
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        mem0_tools.save_memory(
            memory=memory,
            content=content,
            metadata={"role": role},
            user_id=user_id,
        )
        saved_count += 1
    return saved_count


def is_countable_assistant_turn(message: dict[str, Any]) -> bool:
    """只把真实 assistant 回复计为一轮；tool call 不计轮次。"""
    if str(message.get("role", "")).lower() != "assistant":
        return False
    if message.get("tool_calls"):
        return False
    return bool(str(message.get("content", "") or "").strip())


def main() -> None:
    model_path = config.ModelEnum.QWEN2_5_7B_INSTRUCT.value
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
    model_name = Path(str(model_path).rstrip("/\\")).name
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])

    starters = sample_conversation_starters(config.SAMPLE_SIZE, config.RANDOM_SEED)
    memory = mem0_tools.create_memory()
    stego_dispatcher = stegokit.StegoDispatcher(verbose=False)

    temperature = config.TEMPERATURE
    max_new_tokens = config.MAX_NEW_TOKENS
    seed = config.RANDOM_SEED
    secret_bits_length = config.SECRET_BITS_LENGTH

    remote_agent = RemoteAgent(
        model_name=config.REMOTE_AGENT,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    data_dir = PROJECT_ROOT / config.OUTPUT_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    algorithm = STEGO_ALGORITHM
    top_k = STEGO_TOP_K
    top_p = STEGO_TOP_P
    precision = STEGO_PRECISION
    stop_on_eos = STEGO_STOP_ON_EOS
    secure_parameter = STEGO_SECURE_PARAMETER
    func_type = STEGO_FUNC_TYPE

    try:
        for starter_id, starter in enumerate(starters, start=1):
            for trial in range(1, config.REPEATS_PER_STARTER + 1):
                dialog_record: dict[str, Any] = {
                    "algo": algorithm.name,
                    "time": int(time.time()),
                    "temperature": temperature,
                    "max_token": max_new_tokens,
                    "seed": seed,
                    "remote_agent": config.REMOTE_AGENT,
                    "rounds_per_dialog": config.ROUNDS_PER_DIALOGUE,
                    "window_size": config.WINDOW_ROUNDS,
                    "model": model_path,
                    "starter": starter,
                    "stego_top_k": top_k,
                    "stego_top_p": top_p,
                    "stego_precision": precision,
                    "stego_secure_parameter": secure_parameter,
                    "stego_func_type": func_type,
                    "rounds": [],
                }
                trial_output_path = data_dir / f"group4-{model_name}-{LOCAL_FLUSH_ROUNDS}-{starter_id}-{trial}.json"

                try:
                    remote_agent_messages: list[dict[str, Any]] = []
                    local_agent_messages: list[dict[str, Any]] = []
                    remote_agent_messages.append({"role": "system", "content": config.USER_AGENT_SYSTEM_PROMPT})
                    local_agent_messages.append({"role": "system", "content": config.ASSISTANT_AGENT_SYSTEM_PROMPT})
                    completed_rounds_since_flush = 0

                    starter_message = {"role": "user", "content": starter}
                    remote_agent_messages.append(starter_message.copy())
                    local_agent_messages.append(starter_message.copy())

                    mem0_tools.reset_memories_by_userid("demo", memory=memory)
                    print(
                        f"[starter {starter_id}/{len(starters)} | "
                        f"trial {trial}/{config.REPEATS_PER_STARTER}]"
                    )

                    for round_idx in range(1, config.ROUNDS_PER_DIALOGUE + 1):
                        print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Starting round...")
                        # 每轮回答前强制检索记忆，并仅注入本地上下文，制造上下文不一致。
                        memory_hits: list[dict[str, Any]] = []
                        search_result = mem0_tools.search_memory(
                            memory=memory,
                            query=str(local_agent_messages[-1].get("content", "") or "").strip(),
                            limit=MEMORY_SEARCH_LIMIT,
                            user_id="demo",
                        )
                        memory_hits = list(search_result["results"] or [])
                        tool_call_id = f"mem_search_{starter_id}_{trial}_{round_idx}"
                        local_agent_messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": "search_memory",
                                            "arguments": json.dumps(
                                                {
                                                    "query": str(local_agent_messages[-1].get("content", "") or "").strip(),
                                                    "limit": MEMORY_SEARCH_LIMIT,
                                                },
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            }
                        )
                        local_agent_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": "search_memory",
                                "content": json.dumps({"hits": memory_hits}, ensure_ascii=False),
                            }
                        )

                        bit_seed = f"{seed}:{starter_id}:{trial}:{round_idx}"
                        bitstream = generate_random_bitstring(secret_bits_length, seed=bit_seed)

                        stegoencode_context = stegokit.StegoEncodeContext(
                            algorithm=algorithm,
                            model=model,
                            tokenizer=tokenizer,
                            secret_bits=bitstream,
                            messages=local_agent_messages.copy(),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            prg=None,
                            precision=precision,
                            stop_on_eos=stop_on_eos,
                            extra={
                                "seed": seed,
                                "secure_parameter": secure_parameter,
                                "func_type": func_type,
                            },
                        )
                        stego_result: stegokit.StegoEncodeResult = stego_dispatcher.dispatch_encode(stegoencode_context)
                        consumed_bits = stego_result.consumed_bits
                        generated_text = sanitize_message_text(stego_result.text, special_tokens)
                        generated_token_ids = stego_result.generated_token_ids
                        embedded_bits = bitstream[:consumed_bits]

                        print(
                            f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] "
                            f"Generated text with {consumed_bits} secret bits."
                        )
                        print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Encode bits: {embedded_bits}")

                        is_same_context = remote_agent_messages == local_agent_messages
                        stegodecode_context = stegokit.StegoDecodeContext(
                            algorithm=algorithm,
                            model=model,
                            tokenizer=tokenizer,
                            generated_token_ids=generated_token_ids,
                            messages=remote_agent_messages.copy(),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            prg=None,
                            precision=precision,
                            extra={
                                "seed": seed,
                                "secure_parameter": secure_parameter,
                                "func_type": func_type,
                            },
                        )
                        stego_decode_result: stegokit.StegoDecodeResult = stego_dispatcher.dispatch_decode(stegodecode_context)
                        recovered_all_bits = stego_decode_result.bits or ""
                        if consumed_bits > 0:
                            recovered_bits = recovered_all_bits[:consumed_bits]
                            is_correct = recovered_bits == embedded_bits
                        else:
                            recovered_bits = recovered_all_bits
                            is_correct = embedded_bits == "" and recovered_bits == ""

                        print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Recovered bits: {recovered_bits}")
                        if is_correct:
                            print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Secret bits recovered correctly.")
                        else:
                            print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Secret bits NOT recovered correctly.")

                        generated_token_count = len(generated_token_ids)
                        encode_time_seconds = float(getattr(stego_result, "encode_time_seconds", 0.0) or 0.0)
                        decode_time_seconds = float(getattr(stego_decode_result, "decode_time_seconds", 0.0) or 0.0)
                        average_entropy = float(getattr(stego_result, "average_entropy", 0.0) or 0.0)
                        embedding_capacity = float(getattr(stego_result, "embedding_capacity", 0.0) or 0.0)

                        carrier_assistant_message = {"role": "assistant", "content": generated_text}
                        remote_agent_messages.append(carrier_assistant_message.copy())
                        local_agent_messages.append(carrier_assistant_message.copy())
                        if is_countable_assistant_turn(carrier_assistant_message):
                            completed_rounds_since_flush += 1

                        remote_reply = remote_agent.invoke(messages=remote_agent_messages, temperature=temperature)
                        remote_reply = sanitize_message_text(remote_reply, special_tokens)
                        remote_reply_message = {"role": "user", "content": remote_reply}
                        remote_agent_messages.append(remote_reply_message.copy())
                        local_agent_messages.append(remote_reply_message.copy())

                        trim_recent_rounds(remote_agent_messages, config.WINDOW_ROUNDS)
                        flushed_memories = 0
                        if completed_rounds_since_flush >= LOCAL_FLUSH_ROUNDS:
                            # 满 5 轮后：按 user->assistant 轮次触发，tool 调用不算轮次。
                            # 把本地对话(仅 assistant/user)存入 mem0，并清空本地上下文(保留 system + 最新 user)。
                            flushed_memories = persist_local_messages_to_mem0(
                                local_agent_messages,
                                memory=memory,
                                user_id="demo",
                            )
                            latest_user_message = dict(remote_reply_message)
                            system_message = dict(local_agent_messages[0]) if local_agent_messages else {
                                "role": "system",
                                "content": config.ASSISTANT_AGENT_SYSTEM_PROMPT,
                            }
                            local_agent_messages = [system_message]
                            local_agent_messages.append(latest_user_message)
                            completed_rounds_since_flush = 0

                        dialog_record["rounds"].append(
                            {
                                "assistant": generated_text,
                                "user": remote_reply,
                                "embedded_bits": embedded_bits,
                                "recover_bits": recovered_bits,
                                "is_correct": is_correct,
                                "consumed_bits": consumed_bits,
                                "generated_token_count": generated_token_count,
                                "average_entropy": average_entropy,
                                "embedding_capacity": embedding_capacity,
                                "encode_time_seconds": encode_time_seconds,
                                "decode_time_seconds": decode_time_seconds,
                                "is_same_context": is_same_context,
                                "memory_query": str(local_agent_messages[-1].get("content", "") or "").strip(),
                                "memory_hit_count": len(memory_hits),
                                "memory_hits": memory_hits,
                                "flushed_memories": flushed_memories,
                            }
                        )
                        # 每轮结束立即落盘，便于实时观察 rounds 增长
                        with trial_output_path.open("w", encoding="utf-8") as f:
                            json.dump(dialog_record, f, ensure_ascii=False, indent=2)
                finally:
                    with trial_output_path.open("w", encoding="utf-8") as f:
                        json.dump(dialog_record, f, ensure_ascii=False, indent=2)
    finally:
        mem0_tools.close_memory(memory)


if __name__ == "__main__":
    main()
