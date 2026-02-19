from __future__ import annotations

import argparse
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

# Stego params (group2-local)
DEFAULT_ALGORITHM = "discop"
STEGO_TOP_K = 50
STEGO_TOP_P = None
# DISCOP / METEOR 建议使用较低 precision（官方示例为 16）
STEGO_PRECISION = 16
STEGO_STOP_ON_EOS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Group2 with selectable stego algorithm.")
    parser.add_argument(
        "--algorithm",
        choices=("discop", "meteor"),
        default=DEFAULT_ALGORITHM,
        help="Stego algorithm for group2. Default: discop",
    )
    return parser.parse_args()


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


def resolve_algorithm(name: str) -> stegokit.StegoAlgorithm:
    normalized = name.strip().lower()
    if normalized == "discop":
        return stegokit.StegoAlgorithm.DISCOP
    if normalized == "meteor":
        return stegokit.StegoAlgorithm.METEOR
    raise ValueError(f"Unsupported algorithm: {name}")


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
    total_encode_time_seconds = float(
        sum(float(round_item.get("encode_time_seconds", 0.0) or 0.0) for round_item in rounds)
    )
    total_decode_time_seconds = float(
        sum(float(round_item.get("decode_time_seconds", 0.0) or 0.0) for round_item in rounds)
    )

    if total_generated_token_count > 0:
        weighted_average_entropy = weighted_entropy_sum / total_generated_token_count
        average_encode_time_per_token_seconds = total_encode_time_seconds / total_generated_token_count
        average_decode_time_per_token_seconds = total_decode_time_seconds / total_generated_token_count
    else:
        weighted_average_entropy = 0.0
        average_encode_time_per_token_seconds = 0.0
        average_decode_time_per_token_seconds = 0.0

    return {
        "total_generated_token_count": total_generated_token_count,
        "weighted_average_entropy": float(weighted_average_entropy),
        "total_encode_time_seconds": total_encode_time_seconds,
        "total_decode_time_seconds": total_decode_time_seconds,
        "average_encode_time_per_token_seconds": float(average_encode_time_per_token_seconds),
        "average_decode_time_per_token_seconds": float(average_decode_time_per_token_seconds),
    }


def main() -> None:
    args = parse_args()
    # 加载模型与分词器
    model_path = config.ModelEnum.QWEN2_5_7B_INSTRUCT.value
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
    # 模型名字，保存记录要用
    model_name = Path(str(model_path).rstrip("/\\")).name
    # 获取模型分词器的特殊 tokens，后续生成文本清洗时用
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    # 采样Start
    starters = sample_conversation_starters(config.SAMPLE_SIZE, config.RANDOM_SEED)
    # 隐写分发器
    stego_dispatcher = stegokit.StegoDispatcher(verbose=False)
    # 参数设置
    temperature = config.TEMPERATURE
    max_new_tokens = config.MAX_NEW_TOKENS
    seed = config.RANDOM_SEED
    secret_bits_length = config.SECRET_BITS_LENGTH
    # 初始化远程Agent
    remote_agent = RemoteAgent(
        model_name=config.REMOTE_AGENT,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )

    data_dir = PROJECT_ROOT / config.OUTPUT_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    # 参数设置
    algorithm = resolve_algorithm(args.algorithm)
    top_k = STEGO_TOP_K
    top_p = STEGO_TOP_P
    precision = STEGO_PRECISION
    stop_on_eos = STEGO_STOP_ON_EOS

    # 外层按 starter、内层按重复次数组织实验
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
                "stego_stop_on_eos": stop_on_eos,
                "rounds": [],
            }
            trial_output_path = (
                data_dir
                / f"group2-{algorithm.value}-{model_name}-{algorithm.name}-{config.WINDOW_ROUNDS}-{starter_id}-{trial}.json"
            )

            try:
                # 初始化双方上下文
                remote_agent_messages: list[dict[str, str]] = []
                local_agent_messages: list[dict[str, str]] = []
                # 把两侧 system prompt 加入上下文
                remote_agent_messages.append({"role": "system", "content": config.USER_AGENT_SYSTEM_PROMPT})
                local_agent_messages.append({"role": "system", "content": config.ASSISTANT_AGENT_SYSTEM_PROMPT})
                # 把 starter 加入上下文
                starter_message = {"role": "user", "content": starter}
                remote_agent_messages.append(starter_message.copy())
                local_agent_messages.append(starter_message.copy())
                print(
                    f"[starter {starter_id}/{len(starters)} | "
                    f"trial {trial}/{config.REPEATS_PER_STARTER}]"
                )
                for round_idx in range(1, config.ROUNDS_PER_DIALOGUE + 1):
                    print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Starting round...")
                    # 为当前轮生成可复现实验 bitstream
                    bit_seed = f"{seed}:{starter_id}:{trial}:{round_idx}"
                    bitstream = generate_random_bitstring(secret_bits_length, seed=bit_seed)
                    # 隐写编码生成 assistant 文本
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
                        prg=stegokit.PRG.from_int_seed(seed),
                        precision=precision,
                        stop_on_eos=stop_on_eos,
                    )
                    stego_result: stegokit.StegoEncodeResult = stego_dispatcher.dispatch_encode(stegoencode_context)
                    # 嵌入的比特数量
                    consumed_bits = stego_result.consumed_bits
                    # 去掉了特殊 tokens 的生成文本
                    generated_text = sanitize_message_text(stego_result.text, special_tokens)
                    # 保存token ids，后续解码用
                    generated_token_ids = stego_result.generated_token_ids
                    compared_bits_len = min(consumed_bits, len(bitstream))
                    embedded_bits = bitstream[:compared_bits_len]

                    print(
                        f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] "
                        f"Generated text with {consumed_bits} secret bits."
                    )
                    print(f"  [Round {round_idx}/{config.ROUNDS_PER_DIALOGUE}] Encode bits: {embedded_bits}")

                    # 记录一下上下文是否一致，不包含system信息
                    is_same_context = remote_agent_messages == local_agent_messages
                    # 解码
                    stegodecode_context = stegokit.StegoDecodeContext(
                        algorithm=algorithm,
                        model=model,
                        tokenizer=tokenizer,
                        generated_token_ids=generated_token_ids,
                        # DISCOP / METEOR 对上下文极敏感，解码必须与编码使用完全一致的消息上下文。
                        messages=local_agent_messages.copy(),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        prg=stegokit.PRG.from_int_seed(seed),
                        precision=precision,
                        max_bits=compared_bits_len,
                    )
                    stego_decode_result: stegokit.StegoDecodeResult = stego_dispatcher.dispatch_decode(stegodecode_context)
                    recovered_all_bits = stego_decode_result.bits or ""
                    if compared_bits_len > 0:
                        recovered_bits = recovered_all_bits[:compared_bits_len]
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

                    # 推进一轮对话：assistant -> remote user
                    carrier_assistant_message = {"role": "assistant", "content": generated_text}
                    remote_agent_messages.append(carrier_assistant_message.copy())
                    local_agent_messages.append(carrier_assistant_message.copy())
                    # 调用远程模型作回复
                    remote_reply = remote_agent.invoke(messages=remote_agent_messages, temperature=temperature)
                    remote_reply = sanitize_message_text(remote_reply, special_tokens)
                    remote_reply_message = {"role": "user", "content": remote_reply}
                    remote_agent_messages.append(remote_reply_message.copy())
                    local_agent_messages.append(remote_reply_message.copy())

                    # 6) 按窗口长度裁剪两侧上下文，控制上下文规模
                    trim_recent_rounds(remote_agent_messages, config.WINDOW_ROUNDS)
                    trim_recent_rounds(local_agent_messages, config.WINDOW_ROUNDS)

                    # 7) 记录本轮指标与文本结果
                    dialog_record["rounds"].append(
                        {
                            "assistant": generated_text,
                            "user": remote_reply,
                            "embedded_bits": embedded_bits,
                            "recover_bits": recovered_bits,
                            "is_correct": is_correct,
                            "consumed_bits": consumed_bits,
                            "compared_bits_len": compared_bits_len,
                            "generated_token_count": generated_token_count,
                            "average_entropy": average_entropy,
                            "embedding_capacity": embedding_capacity,
                            "encode_time_seconds": encode_time_seconds,
                            "decode_time_seconds": decode_time_seconds,
                            "is_same_context": is_same_context,
                        }
                    )
                    if len(dialog_record["rounds"]) >= config.ROUNDS_PER_DIALOGUE:
                        dialog_record["metrics"] = build_dialog_metrics(dialog_record)
                    # 每轮结束立即落盘，便于实时观察 rounds 增长
                    with trial_output_path.open("w", encoding="utf-8") as f:
                        json.dump(dialog_record, f, ensure_ascii=False, indent=2)
            finally:
                # 每个 trial 都落盘，避免中途异常导致数据丢失
                if len(dialog_record.get("rounds", [])) >= config.ROUNDS_PER_DIALOGUE:
                    dialog_record["metrics"] = build_dialog_metrics(dialog_record)
                else:
                    dialog_record.pop("metrics", None)
                with trial_output_path.open("w", encoding="utf-8") as f:
                    json.dump(dialog_record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
