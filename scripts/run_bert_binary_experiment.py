from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.tools import analysis_tools

LABEL_TO_ID = {"normal": 0, "stego": 1}
ID_TO_LABEL = {0: "normal", 1: "stego"}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_name: str
    negative_group: str
    positive_group: str
    positive_acf_k: int | None = None


@dataclass(frozen=True)
class PairRecord:
    task_id: str
    question_id: str
    seed: int
    negative_text: str
    positive_text: str
    negative_group: str
    positive_group: str
    positive_acf_k: int | None


@dataclass(frozen=True)
class Sample:
    task_id: str
    pair_id: str
    pair_member: str
    question_id: str
    seed: int
    text: str
    label: int
    label_name: str
    source_group: str
    acf_k: int | None


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("task_a_normal_vs_discop", "Normal vs DISCOP", negative_group="G1", positive_group="G2"),
    TaskSpec("task_b_normal_vs_meteor", "Normal vs METEOR", negative_group="G1", positive_group="G3"),
    TaskSpec("task_c_normal_vs_acf_k8", "Normal vs ACF (k=8)", negative_group="G1", positive_group="G4", positive_acf_k=8),
    TaskSpec("task_d_normal_vs_acf_k12", "Normal vs ACF (k=12)", negative_group="G1", positive_group="G4", positive_acf_k=12),
    TaskSpec("task_e_normal_vs_acf_k16", "Normal vs ACF (k=16)", negative_group="G1", positive_group="G4", positive_acf_k=16),
    TaskSpec("task_f_normal_ret_vs_discop_ret", "Normal+RET vs DISCOP+RET", negative_group="G8", positive_group="G6"),
    TaskSpec("task_g_normal_ret_vs_meteor_ret", "Normal+RET vs METEOR+RET", negative_group="G8", positive_group="G7"),
    TaskSpec("task_h_normal_ret_vs_acf_ret_k8", "Normal+RET vs ACF+RET (k=8)", negative_group="G8", positive_group="G5", positive_acf_k=8),
    TaskSpec("task_i_normal_ret_vs_acf_ret_k12", "Normal+RET vs ACF+RET (k=12)", negative_group="G8", positive_group="G5", positive_acf_k=12),
    TaskSpec("task_j_normal_ret_vs_acf_ret_k16", "Normal+RET vs ACF+RET (k=16)", negative_group="G8", positive_group="G5", positive_acf_k=16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run protocol-specific BERT binary classifiers on realistic experiment outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Project data root containing outputs_v2/realistic/*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs_v2" / "bert_binary",
        help="Directory to save per-task splits, models, and summaries.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="/root/autodl-fs/bert-base-uncased",
        help="Hugging Face model id or local path for sequence classification.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto/cpu/cuda/cuda:0 ...",
    )
    parser.add_argument(
        "--task",
        choices=[spec.task_id for spec in TASK_SPECS],
        default=None,
        help="Run only one binary task instead of all tasks.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build task pairings and train/val/test splits without model training.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_length) <= 0:
        raise ValueError("--max-length must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be > 0")
    if float(args.learning_rate) <= 0:
        raise ValueError("--learning-rate must be > 0")
    if float(args.weight_decay) < 0:
        raise ValueError("--weight-decay must be >= 0")
    if float(args.warmup_ratio) < 0 or float(args.warmup_ratio) > 1:
        raise ValueError("--warmup-ratio must be in [0, 1]")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    device_name = str(raw or "auto").strip().lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_realistic_records(data_root: Path) -> list[dict[str, Any]]:
    realistic_root = data_root / "outputs_v2" / "realistic"
    if not realistic_root.exists():
        raise FileNotFoundError(f"Missing realistic outputs directory: {realistic_root}")

    records: list[dict[str, Any]] = []
    for path in sorted(realistic_root.glob("*/*.json")):
        record = analysis_tools.load_json_record(path)
        if record is None:
            continue
        if str(record.get("experiment_key", "")).strip() != "realistic":
            continue
        if str(record.get("condition", "")).strip() != "no_drift":
            continue
        if "assistant_answer" not in record or "group" not in record:
            continue
        records.append(record)
    return records


def build_record_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, int, int | None], dict[str, Any]]:
    index: dict[tuple[str, str, int, int | None], dict[str, Any]] = {}
    for record in records:
        group = str(record.get("group", "")).strip()
        question_id = str(record.get("question_id", "")).strip()
        seed = safe_int(record.get("seed"))
        acf_k = safe_int(record.get("acf_k"))
        if not group or not question_id or seed is None:
            continue
        index[(group, question_id, seed, acf_k)] = record
    return index


def build_task_pairs(task: TaskSpec, record_index: dict[tuple[str, str, int, int | None], dict[str, Any]]) -> list[PairRecord]:
    positive_pairs: list[PairRecord] = []
    positive_acf_k = task.positive_acf_k

    for (group, question_id, seed, acf_k), positive_record in sorted(record_index.items()):
        if group != task.positive_group:
            continue
        if acf_k != positive_acf_k:
            if not (acf_k is None and positive_acf_k is None):
                continue

        negative_record = record_index.get((task.negative_group, question_id, seed, None))
        if negative_record is None:
            continue

        negative_text = str(negative_record.get("assistant_answer", "")).strip()
        positive_text = str(positive_record.get("assistant_answer", "")).strip()
        if not negative_text or not positive_text:
            continue

        positive_pairs.append(
            PairRecord(
                task_id=task.task_id,
                question_id=question_id,
                seed=seed,
                negative_text=negative_text,
                positive_text=positive_text,
                negative_group=task.negative_group,
                positive_group=task.positive_group,
                positive_acf_k=positive_acf_k,
            )
        )

    if not positive_pairs:
        raise ValueError(f"No paired samples found for task {task.task_id}")
    return positive_pairs


def split_pairs(
    pairs: list[PairRecord],
    *,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord], list[PairRecord]]:
    current = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(current)

    n_total = len(current)
    n_train = int(round(n_total * 0.8))
    n_train = min(max(1, n_train), n_total - 2)
    temp = current[n_train:]
    n_val = len(temp) // 2
    n_test = len(temp) - n_val

    if n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Pair split failed: total={n_total} train={n_train} val={n_val} test={n_test}"
        )

    return current[:n_train], temp[:n_val], temp[n_val:]


def pair_to_samples(pair: PairRecord) -> tuple[Sample, Sample]:
    pair_id = f"{pair.task_id}:{pair.question_id}:{pair.seed}"
    negative_sample = Sample(
        task_id=pair.task_id,
        pair_id=pair_id,
        pair_member="negative",
        question_id=pair.question_id,
        seed=pair.seed,
        text=pair.negative_text,
        label=LABEL_TO_ID["normal"],
        label_name="normal",
        source_group=pair.negative_group,
        acf_k=None,
    )
    positive_sample = Sample(
        task_id=pair.task_id,
        pair_id=pair_id,
        pair_member="positive",
        question_id=pair.question_id,
        seed=pair.seed,
        text=pair.positive_text,
        label=LABEL_TO_ID["stego"],
        label_name="stego",
        source_group=pair.positive_group,
        acf_k=pair.positive_acf_k,
    )
    return negative_sample, positive_sample


def flatten_pairs(pairs: list[PairRecord]) -> list[Sample]:
    samples: list[Sample] = []
    for pair in pairs:
        negative_sample, positive_sample = pair_to_samples(pair)
        samples.extend([negative_sample, positive_sample])
    return samples


def encode_dataset(
    tokenizer: PreTrainedTokenizerBase,
    samples: list[Sample],
    *,
    max_length: int,
) -> EncodedTextDataset:
    texts = [sample.text for sample in samples]
    labels = [sample.label for sample in samples]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    kept = {key: value for key, value in encodings.items() if isinstance(value, torch.Tensor)}
    return EncodedTextDataset(kept, labels)


def build_loader(dataset: EncodedTextDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_binary_metrics(predictions: list[int], labels: list[int]) -> dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels length mismatch")
    total = len(labels)
    if total == 0:
        raise ValueError("Empty labels in metric computation")

    tp = sum(1 for pred, gold in zip(predictions, labels) if pred == 1 and gold == 1)
    tn = sum(1 for pred, gold in zip(predictions, labels) if pred == 0 and gold == 0)
    fp = sum(1 for pred, gold in zip(predictions, labels) if pred == 1 and gold == 0)
    fn = sum(1 for pred, gold in zip(predictions, labels) if pred == 0 and gold == 1)

    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_eval(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_count += batch_size

            preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            gold = labels.detach().cpu().tolist()
            all_predictions.extend(int(item) for item in preds)
            all_labels.extend(int(item) for item in gold)

    metrics = compute_binary_metrics(all_predictions, all_labels)
    metrics["loss"] = float(total_loss / max(1, total_count))
    return metrics


def train_one_epoch(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Model returned no training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size

    return float(total_loss / max(1, total_count))


def train_with_validation(
    *,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    train_samples: list[Sample],
    eval_samples: list[Sample],
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, dict[str, Any]]:
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_dataset = encode_dataset(tokenizer, train_samples, max_length=max_length)
    eval_dataset = encode_dataset(tokenizer, eval_samples, max_length=max_length)
    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = build_loader(eval_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * max(1, epochs))
    warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_eval_accuracy = -1.0
    best_eval_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        eval_metrics = run_eval(model, eval_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_precision": eval_metrics["precision"],
                "eval_recall": eval_metrics["recall"],
                "eval_f1": eval_metrics["f1"],
            }
        )

        if eval_metrics["accuracy"] > best_eval_accuracy:
            best_eval_accuracy = float(eval_metrics["accuracy"])
            best_epoch = epoch
            best_eval_metrics = dict(eval_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, {
        "best_epoch": best_epoch,
        "best_eval_metrics": best_eval_metrics,
        "history": history,
    }


def sample_to_json(sample: Sample, sample_id: int) -> dict[str, Any]:
    text = str(sample.text or "")
    return {
        "id": sample_id,
        "task_id": sample.task_id,
        "pair_id": sample.pair_id,
        "pair_member": sample.pair_member,
        "question_id": sample.question_id,
        "seed": sample.seed,
        "text": text,
        "label": int(sample.label),
        "label_name": sample.label_name,
        "source_group": sample.source_group,
        "acf_k": sample.acf_k,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "task_name",
        "negative_group",
        "positive_group",
        "positive_acf_k",
        "pair_count",
        "train_pairs",
        "val_pairs",
        "test_pairs",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_loss",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single_task(
    *,
    task: TaskSpec,
    record_index: dict[tuple[str, str, int, int | None], dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase | None,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    task_dir = ensure_dir(output_dir / task.task_id)
    pairs = build_task_pairs(task, record_index)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=int(args.seed))

    train_samples = flatten_pairs(train_pairs)
    val_samples = flatten_pairs(val_pairs)
    test_samples = flatten_pairs(test_pairs)

    write_jsonl(task_dir / "split_train.jsonl", [sample_to_json(item, idx) for idx, item in enumerate(train_samples, start=1)])
    write_jsonl(task_dir / "split_val.jsonl", [sample_to_json(item, idx) for idx, item in enumerate(val_samples, start=1)])
    write_jsonl(task_dir / "split_test.jsonl", [sample_to_json(item, idx) for idx, item in enumerate(test_samples, start=1)])

    task_summary: dict[str, Any] = {
        "task_id": task.task_id,
        "task_name": task.task_name,
        "negative_group": task.negative_group,
        "positive_group": task.positive_group,
        "positive_acf_k": task.positive_acf_k,
        "pair_count": len(pairs),
        "split_pairs": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "split_samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "split_label_counts": {
            "train": {"normal": len(train_pairs), "stego": len(train_pairs)},
            "val": {"normal": len(val_pairs), "stego": len(val_pairs)},
            "test": {"normal": len(test_pairs), "stego": len(test_pairs)},
        },
    }

    if args.prepare_only:
        write_json(task_dir / "task_summary.json", {"data": task_summary, "prepare_only": True})
        return task_summary

    if tokenizer is None:
        raise ValueError("Tokenizer must be available when training is enabled.")

    model, final_summary = train_with_validation(
        model_name=args.model_name,
        tokenizer=tokenizer,
        train_samples=train_samples,
        eval_samples=val_samples,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        seed=int(args.seed),
        device=device,
    )

    test_dataset = encode_dataset(tokenizer, test_samples, max_length=int(args.max_length))
    test_loader = build_loader(test_dataset, batch_size=int(args.batch_size), shuffle=False)
    test_metrics = run_eval(model, test_loader, device=device)

    model_dir = ensure_dir(task_dir / "final_model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    payload = {
        "data": task_summary,
        "training_config": {
            "model_name": args.model_name,
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "device": str(device),
        },
        "final": {
            "best_epoch_on_val": int(final_summary["best_epoch"]),
            "val_metrics": final_summary["best_eval_metrics"],
            "test_metrics": test_metrics,
            "positive_label": ID_TO_LABEL[1],
        },
    }
    write_json(task_dir / "task_summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    records = load_realistic_records(args.data_root)
    record_index = build_record_index(records)
    selected_tasks = [spec for spec in TASK_SPECS if args.task in {None, spec.task_id}]

    tokenizer: PreTrainedTokenizerBase | None = None
    if not args.prepare_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    task_payloads: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for offset, task in enumerate(selected_tasks, start=1):
        payload = run_single_task(
            task=task,
            record_index=record_index,
            tokenizer=tokenizer,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        task_payloads.append(payload)

        data_payload = payload["data"] if "data" in payload else payload
        final_payload = payload.get("final", {})
        test_metrics = final_payload.get("test_metrics", {})
        csv_rows.append(
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "negative_group": task.negative_group,
                "positive_group": task.positive_group,
                "positive_acf_k": task.positive_acf_k if task.positive_acf_k is not None else "",
                "pair_count": data_payload["pair_count"],
                "train_pairs": data_payload["split_pairs"]["train"],
                "val_pairs": data_payload["split_pairs"]["val"],
                "test_pairs": data_payload["split_pairs"]["test"],
                "test_accuracy": test_metrics.get("accuracy", ""),
                "test_precision": test_metrics.get("precision", ""),
                "test_recall": test_metrics.get("recall", ""),
                "test_f1": test_metrics.get("f1", ""),
                "test_loss": test_metrics.get("loss", ""),
            }
        )

    overall_payload = {
        "created_at": utc_now_iso(),
        "mode": "prepare_only" if args.prepare_only else "train_and_eval",
        "source_record_count": len(records),
        "task_count": len(selected_tasks),
        "tasks": task_payloads,
    }
    write_json(output_dir / "bert_binary_summary.json", overall_payload)
    write_results_csv(output_dir / "bert_binary_results.csv", csv_rows)
    print(json.dumps({"status": "done", "summary": str(output_dir / "bert_binary_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
