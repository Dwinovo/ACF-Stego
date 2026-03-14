from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


def normalize_answer(text: str) -> str:
    lowered = str(text or "").lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def exact_match_score(prediction: str, gold_answer: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold_answer) else 0.0


def token_f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    scores = [exact_match_score(prediction, answer) for answer in gold_answers]
    return max(scores, default=0.0)


def best_token_f1(prediction: str, gold_answers: Iterable[str]) -> float:
    scores = [token_f1_score(prediction, answer) for answer in gold_answers]
    return max(scores, default=0.0)
