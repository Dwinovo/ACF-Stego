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
    generated_ids = torch.tensor([list(generated_token_ids)], device=device, dtype=torch.long)
    full_ids = torch.cat([prefix_ids, generated_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits[0]

    prompt_len = prefix_ids.shape[1]
    full_vocab_size = logits.shape[-1]
    log2 = torch.log(torch.tensor(2.0, device=logits.device, dtype=logits.dtype))
    per_step_entropy: list[float] = []

    for step_idx in range(len(generated_token_ids)):
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

    if not per_step_entropy:
        return 0.0
    return float(sum(per_step_entropy) / len(per_step_entropy))
