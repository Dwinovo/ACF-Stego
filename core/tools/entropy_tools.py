from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _model_device(model: PreTrainedModel) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_message_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[dict[str, Any]],
    device: torch.device,
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
        enc = tokenizer(rendered, return_tensors="pt")
        input_ids = enc["input_ids"]
    else:
        raise TypeError(f"Unsupported apply_chat_template output type: {type(rendered)}")

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    return input_ids.to(device=device, dtype=torch.long)


def compute_average_entropy_for_generated_ids(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[dict[str, Any]],
    generated_token_ids: Sequence[int],
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> float:
    if not generated_token_ids:
        return 0.0

    device = _model_device(model)
    prefix_ids = _prepare_message_input_ids(tokenizer, messages, device)

    def entropy_from_logits(step_logits: torch.Tensor) -> float:
        logits = step_logits.float()
        scaled_logits = logits / max(float(temperature), 1e-8)
        probs = torch.softmax(scaled_logits, dim=-1)

        if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
            probs, _ = torch.topk(probs, k=top_k)

        if top_p is not None and 0 < top_p < 1:
            sorted_probs, _ = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            keep = cumsum <= top_p
            if keep.numel() > 0:
                keep[0] = True
            probs = sorted_probs[keep]

        denom = probs.sum()
        if torch.isnan(denom) or denom <= 0:
            return 0.0

        probs = probs / denom
        probs = probs.clamp_min(1e-12)
        log2_probs = torch.log2(probs)
        entropy = -(probs * log2_probs).sum()
        entropy_value = float(entropy.item())
        return entropy_value if entropy_value == entropy_value else 0.0

    per_step_entropy: list[float] = []
    generated_ids = list(generated_token_ids)

    with torch.no_grad():
        output = model(prefix_ids, use_cache=True)
        next_token_logits = output.logits[0, -1, :]
        past_key_values = getattr(output, "past_key_values", None)

        for token_id in generated_ids:
            entropy_value = entropy_from_logits(next_token_logits)
            if entropy_value == entropy_value:
                per_step_entropy.append(entropy_value)

            token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
            output = model(token_tensor, past_key_values=past_key_values, use_cache=True)
            next_token_logits = output.logits[0, -1, :]
            past_key_values = getattr(output, "past_key_values", None)

    if not per_step_entropy:
        return 0.0
    return float(sum(per_step_entropy) / len(per_step_entropy))
