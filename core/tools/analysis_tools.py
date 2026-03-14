from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import config

METRICS = (
    "task_em",
    "task_f1",
    "llm_judge_score",
    "llm_judge_correct",
    "average_entropy",
    "generated_token_count",
    "generate_time_seconds",
    "encode_time_seconds",
    "decode_time_seconds",
    "embedding_capacity",
    "consumed_bits",
    "secret_bits_budget",
    "compared_bits_len",
    "bit_errors",
    "ber",
    "decode_success",
    "decode_recent_sessions_kept",
    "retrieval_hit_count",
    "retrieval_chars",
    "encode_prompt_tokens_before",
    "encode_prompt_tokens_after",
    "encode_prompt_trimmed",
    "encode_trimmed_history_message_count",
    "encode_retrieval_tokens_before",
    "encode_retrieval_tokens_after",
    "encode_retrieval_trimmed",
    "decode_prompt_tokens_before",
    "decode_prompt_tokens_after",
    "decode_prompt_trimmed",
    "decode_trimmed_history_message_count",
)

GROUP_ORDER = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7}
CONTROLLED_GROUPS = ("G2", "G3", "G4")
REALISTIC_TASK_GROUPS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
REALISTIC_STEGO_GROUPS = ("G2", "G3", "G4", "G5", "G6", "G7")
DRIFT_RECENT_PATTERN = re.compile(r"^drift_recent(\d+)$")
_ANY_ACF_K = object()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def normalize_acf_k(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    rounded = int(round(number))
    return rounded if rounded > 0 else None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def ci95_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    std = statistics.stdev(values)
    return 1.96 * std / math.sqrt(len(values))


def format_mean_std(mean: float | None, std: float | None, *, precision: int = 4) -> str | None:
    if mean is None:
        return None
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def compact_metric(records: Iterable[dict[str, Any]], metric: str) -> list[float]:
    values = [safe_float(record.get(metric)) for record in records]
    return [value for value in values if value is not None]


def group_sort_key(group: str) -> tuple[int, str]:
    return GROUP_ORDER.get(group, 99), group


def condition_sort_key(condition: str) -> tuple[int, str]:
    normalized = str(condition or "").strip()
    if normalized == "no_drift":
        return 0, "0"
    match = DRIFT_RECENT_PATTERN.match(normalized)
    if match:
        keep = int(match.group(1))
        return 1, f"{999 - keep:03d}"
    return 99, normalized


def decoder_sessions_kept_from_condition(condition: str, row: dict[str, Any] | None = None) -> int | None:
    normalized = str(condition or "").strip()
    if row is not None:
        explicit = safe_float(row.get("decode_recent_sessions_kept_mean"))
        if explicit is None:
            explicit = safe_float(row.get("decode_recent_sessions_kept"))
        if explicit is not None:
            return int(round(explicit))
    if normalized == "no_drift":
        return None
    match = DRIFT_RECENT_PATTERN.match(normalized)
    if not match:
        return None
    return int(match.group(1))


def load_json_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(record, dict):
        return None
    if "run_id" not in record:
        return None
    return record


def iter_record_paths(output_root: Path, groups: list[str] | None = None) -> list[Path]:
    if not output_root.exists():
        return []

    if groups:
        paths: list[Path] = []
        for group_name in groups:
            group_dir = output_root / group_name
            if not group_dir.exists():
                continue
            paths.extend(sorted(group_dir.glob("*.json")))
        return paths

    return sorted(output_root.glob("*/*.json"))


def load_records(output_root: Path, groups: list[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_record_paths(output_root, groups=groups):
        record = load_json_record(path)
        if record is not None:
            records.append(normalize_record_units(record))
    return records


def normalize_record_units(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    acf_k = normalize_acf_k(normalized.get("acf_k"))
    if acf_k is not None:
        normalized["acf_k"] = acf_k
    elif "acf_k" in normalized:
        normalized.pop("acf_k", None)
    cap = safe_float(normalized.get("embedding_capacity"))
    if cap is not None:
        normalized["embedding_capacity"] = cap
    return normalized


def build_instance_means(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        experiment = str(record.get("experiment", "")).strip()
        group = str(record.get("group", "")).strip()
        condition = str(record.get("condition", "no_drift")).strip() or "no_drift"
        question_id = str(record.get("question_id", "")).strip()
        acf_k = normalize_acf_k(record.get("acf_k"))
        buckets[(experiment, group, condition, question_id, acf_k)].append(record)

    instance_rows: list[dict[str, Any]] = []
    for (experiment, group, condition, question_id, acf_k), bucket in buckets.items():
        base_record = bucket[0]
        row: dict[str, Any] = {
            "experiment": experiment,
            "experiment_key": base_record.get("experiment_key"),
            "split": base_record.get("split"),
            "group": group,
            "condition": condition,
            "question_id": question_id,
            "acf_k": acf_k,
            "run_count": len(bucket),
        }
        for metric in METRICS:
            row[metric] = mean_or_none(compact_metric(bucket, metric))
        instance_rows.append(row)

    instance_rows.sort(
        key=lambda row: (
            str(row.get("experiment", "")),
            group_sort_key(str(row.get("group", ""))),
            condition_sort_key(str(row.get("condition", ""))),
            normalize_acf_k(row.get("acf_k")) or 0,
            str(row.get("question_id", "")),
        )
    )
    return instance_rows


def summarize_instance_means(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        experiment = str(row.get("experiment", "")).strip()
        group = str(row.get("group", "")).strip()
        condition = str(row.get("condition", "no_drift")).strip() or "no_drift"
        acf_k = normalize_acf_k(row.get("acf_k"))
        buckets[(experiment, group, condition, acf_k)].append(row)

    summaries: list[dict[str, Any]] = []
    for (experiment, group, condition, acf_k), bucket in buckets.items():
        base_row = bucket[0]
        summary: dict[str, Any] = {
            "experiment": experiment,
            "experiment_key": base_row.get("experiment_key"),
            "split": base_row.get("split"),
            "group": group,
            "condition": condition,
            "acf_k": acf_k,
            "instance_count": len(bucket),
        }
        for metric in METRICS:
            values = compact_metric(bucket, metric)
            mean_value = mean_or_none(values)
            std_value = stdev_or_none(values)
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
            summary[f"{metric}_ci95"] = ci95_or_none(values)
            summary[metric] = format_mean_std(mean_value, std_value)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            str(row.get("experiment", "")),
            group_sort_key(str(row.get("group", ""))),
            condition_sort_key(str(row.get("condition", ""))),
            normalize_acf_k(row.get("acf_k")) or 0,
        )
    )
    return summaries


def select_summary_row(
    summaries: list[dict[str, Any]],
    *,
    experiment: str,
    group: str,
    condition: str,
    acf_k: int | None | object = _ANY_ACF_K,
) -> dict[str, Any] | None:
    for row in summaries:
        if (
            str(row.get("experiment", "")) == experiment
            and str(row.get("group", "")) == group
            and str(row.get("condition", "")) == condition
            and (
                acf_k is _ANY_ACF_K
                or normalize_acf_k(row.get("acf_k")) == normalize_acf_k(acf_k)
            )
        ):
            return row
    return None


def metric_value(row: dict[str, Any] | None, metric: str, *, precision: int = 4) -> str:
    if row is None:
        return "-"
    formatted = row.get(metric)
    if formatted is None:
        return "-"
    return str(formatted)


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines).strip()


def mean_std_value(
    row: dict[str, Any] | None,
    metric: str,
    *,
    precision: int = 4,
    missing: str = "---",
) -> str:
    if row is None:
        return missing
    mean = safe_float(row.get(f"{metric}_mean"))
    std = safe_float(row.get(f"{metric}_std"))
    if mean is None:
        return missing
    std_value = 0.0 if std is None else std
    return f"{mean:.{precision}f} ± {std_value:.{precision}f}"


def mean_std_percent(
    row: dict[str, Any] | None,
    metric: str,
    *,
    precision: int = 2,
    missing: str = "---",
) -> str:
    if row is None:
        return missing
    mean = safe_float(row.get(f"{metric}_mean"))
    std = safe_float(row.get(f"{metric}_std"))
    if mean is None:
        return missing
    std_value = 0.0 if std is None else std
    scale = 100.0
    return f"{mean * scale:.{precision}f}% ± {std_value * scale:.{precision}f}%"


def effective_capacity_from_nominal_and_dsr(
    row: dict[str, Any] | None,
    *,
    precision: int = 4,
    missing: str = "---",
) -> str:
    if row is None:
        return missing
    nominal_mean = safe_float(row.get("embedding_capacity_mean"))
    nominal_std = safe_float(row.get("embedding_capacity_std"))
    dsr_mean = safe_float(row.get("decode_success_mean"))
    dsr_std = safe_float(row.get("decode_success_std"))
    if nominal_mean is None or dsr_mean is None:
        return missing
    nominal_std_value = 0.0 if nominal_std is None else nominal_std
    dsr_std_value = 0.0 if dsr_std is None else dsr_std

    effective_mean = nominal_mean * dsr_mean
    effective_std = math.sqrt((dsr_mean * nominal_std_value) ** 2 + (nominal_mean * dsr_std_value) ** 2)
    return f"{effective_mean:.{precision}f} ± {effective_std:.{precision}f}"


def nominal_capacity_per_1k_value(
    row: dict[str, Any] | None,
    *,
    precision: int = 4,
    missing: str = "---",
) -> str:
    if row is None:
        return missing
    nominal_mean = safe_float(row.get("embedding_capacity_mean"))
    nominal_std = safe_float(row.get("embedding_capacity_std"))
    if nominal_mean is None:
        return missing
    nominal_std_value = 0.0 if nominal_std is None else nominal_std
    return f"{nominal_mean:.{precision}f} ± {nominal_std_value:.{precision}f}"


CONTROLLED_PROTOCOL_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("DISCOP", "G2", None),
    ("METEOR", "G3", None),
    ("ACF (k=8)", "G4", 8),
    ("ACF (k=12)", "G4", 12),
    ("ACF (k=16)", "G4", 16),
)

REALISTIC_PROTOCOL_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("Normal (No Stego)", "G1", None),
    ("DISCOP", "G2", None),
    ("DISCOP+RET", "G6", None),
    ("METEOR", "G3", None),
    ("METEOR+RET", "G7", None),
    ("ACF (k=8)", "G4", 8),
    ("ACF (k=12)", "G4", 12),
    ("ACF (k=16)", "G4", 16),
    ("ACF+RET (k=8)", "G5", 8),
    ("ACF+RET (k=12)", "G5", 12),
    ("ACF+RET (k=16)", "G5", 16),
)


def build_controlled_cognitive_asymmetry_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    main_experiment = "controlled_cognitive_asymmetry"
    summary_experiment = "controlled_summary_asymmetry"
    rows: list[list[str]] = []
    for protocol, group, acf_k in CONTROLLED_PROTOCOL_SPECS:
        no_drift = select_summary_row(
            summaries,
            experiment=main_experiment,
            group=group,
            condition="no_drift",
            acf_k=acf_k,
        )
        drift = select_summary_row(
            summaries,
            experiment=main_experiment,
            group=group,
            condition="drift_recent3",
            acf_k=acf_k,
        )
        summary_only = select_summary_row(
            summaries,
            experiment=summary_experiment,
            group=group,
            condition="summary_only_enc",
            acf_k=acf_k,
        )
        rows.append(
            [
                protocol,
                mean_std_percent(no_drift, "ber"),
                mean_std_percent(drift, "ber"),
                mean_std_percent(drift, "decode_success"),
                mean_std_percent(summary_only, "ber"),
                mean_std_percent(summary_only, "decode_success"),
            ]
        )
    return rows


def build_realistic_integrated_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    experiment = "realistic_cognitive_asymmetry"
    rows: list[list[str]] = []
    for protocol, group, acf_k in REALISTIC_PROTOCOL_SPECS:
        row = select_summary_row(
            summaries,
            experiment=experiment,
            group=group,
            condition="no_drift",
            acf_k=acf_k,
        )
        if group == "G1":
            rows.append(
                [
                    protocol,
                    mean_std_percent(row, "llm_judge_correct"),
                    mean_std_value(row, "llm_judge_score", precision=2),
                    mean_std_percent(row, "task_f1"),
                    "---",
                    "---",
                    "---",
                    "---",
                ]
            )
            continue

        rows.append(
            [
                protocol,
                mean_std_percent(row, "llm_judge_correct"),
                mean_std_value(row, "llm_judge_score", precision=2),
                mean_std_percent(row, "task_f1"),
                mean_std_percent(row, "ber"),
                mean_std_percent(row, "decode_success"),
                nominal_capacity_per_1k_value(row, precision=4),
                effective_capacity_from_nominal_and_dsr(row, precision=4),
            ]
        )
    return rows


def build_controlled_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    main_experiment = "controlled_cognitive_asymmetry"
    summary_experiment = "controlled_summary_asymmetry"
    rows: list[list[str]] = []
    for group in CONTROLLED_GROUPS:
        no_drift = select_summary_row(summaries, experiment=main_experiment, group=group, condition="no_drift")
        drift = select_summary_row(summaries, experiment=main_experiment, group=group, condition="drift_recent3")
        summary_only = select_summary_row(
            summaries,
            experiment=summary_experiment,
            group=group,
            condition="summary_only_enc",
        )
        rows.append(
            [
                group,
                metric_value(no_drift, "ber"),
                metric_value(drift, "ber"),
                metric_value(summary_only, "ber"),
                metric_value(drift, "decode_success"),
                metric_value(summary_only, "decode_success"),
            ]
        )
    return rows


def build_controlled_summary_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    main_experiment = "controlled_cognitive_asymmetry"
    summary_experiment = "controlled_summary_asymmetry"
    rows: list[list[str]] = []
    for group in CONTROLLED_GROUPS:
        no_drift = select_summary_row(summaries, experiment=main_experiment, group=group, condition="no_drift")
        summary_only = select_summary_row(
            summaries,
            experiment=summary_experiment,
            group=group,
            condition="summary_only_enc",
        )
        rows.append(
            [
                group,
                metric_value(no_drift, "ber"),
                metric_value(summary_only, "ber"),
                metric_value(summary_only, "decode_success"),
            ]
        )
    return rows


def build_realistic_task_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    experiment = "realistic_cognitive_asymmetry"
    rows: list[list[str]] = []
    for group in REALISTIC_TASK_GROUPS:
        row = select_summary_row(summaries, experiment=experiment, group=group, condition="no_drift")
        rows.append(
            [
                group,
                metric_value(row, "llm_judge_correct"),
                metric_value(row, "llm_judge_score"),
                metric_value(row, "task_f1"),
                metric_value(row, "task_em"),
            ]
        )
    return rows


def build_realistic_protocol_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    experiment = "realistic_cognitive_asymmetry"
    rows: list[list[str]] = []
    for group in REALISTIC_STEGO_GROUPS:
        row = select_summary_row(summaries, experiment=experiment, group=group, condition="no_drift")
        rows.append(
            [
                group,
                metric_value(row, "ber"),
                metric_value(row, "decode_success"),
                metric_value(row, "embedding_capacity"),
            ]
        )
    return rows


def build_realistic_llm_table_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    experiment = "realistic_cognitive_asymmetry"
    rows: list[list[str]] = []
    for group in REALISTIC_TASK_GROUPS:
        row = select_summary_row(summaries, experiment=experiment, group=group, condition="no_drift")
        rows.append(
            [
                group,
                metric_value(row, "llm_judge_correct"),
                metric_value(row, "llm_judge_score"),
                metric_value(row, "task_f1"),
                metric_value(row, "task_em"),
            ]
        )
    return rows


def build_ber_vs_condition_plot(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    experiment = "controlled_cognitive_asymmetry"
    for group in CONTROLLED_GROUPS:
        for condition in ("no_drift", "drift_recent3"):
            row = select_summary_row(summaries, experiment=experiment, group=group, condition=condition)
            if row is None:
                continue
            ber = safe_float(row.get("ber_mean"))
            if ber is None:
                continue
            points.append(
                {
                    "experiment": experiment,
                    "group": group,
                    "condition": condition,
                    "ber": ber,
                    "decode_success": safe_float(row.get("decode_success_mean")),
                }
            )
    return points


def build_controlled_drift_severity_sweep_plot(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiment = "controlled_drift_severity_sweep"
    points: list[dict[str, Any]] = []
    for row in summaries:
        if str(row.get("experiment", "")).strip() != experiment:
            continue
        group = str(row.get("group", "")).strip()
        condition = str(row.get("condition", "")).strip()
        ber = safe_float(row.get("ber_mean"))
        if ber is None:
            continue
        decoder_sessions_kept = decoder_sessions_kept_from_condition(condition, row)
        if decoder_sessions_kept is None and condition == "no_drift":
            decoder_sessions_kept = int(round(safe_float(row.get("decode_recent_sessions_kept_mean")) or 0))
        points.append(
            {
                "experiment": experiment,
                "group": group,
                "condition": condition,
                "decoder_sessions_kept": (
                    decoder_sessions_kept
                    if decoder_sessions_kept is not None
                    else int(config.LONGMEMEVAL_WINDOW_SESSIONS)
                ),
                "ber": ber,
                "decode_success": safe_float(row.get("decode_success_mean")),
            }
        )
    points.sort(
        key=lambda point: (
            group_sort_key(str(point.get("group", ""))),
            -(int(point.get("decoder_sessions_kept", 0) or 0)),
            str(point.get("condition", "")),
        )
    )
    return points


def build_controlled_summary_asymmetry_plot(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for group in CONTROLLED_GROUPS:
        no_drift = select_summary_row(
            summaries,
            experiment="controlled_cognitive_asymmetry",
            group=group,
            condition="no_drift",
        )
        summary_only = select_summary_row(
            summaries,
            experiment="controlled_summary_asymmetry",
            group=group,
            condition="summary_only_enc",
        )
        for label, row in (("no_drift", no_drift), ("summary_only_enc", summary_only)):
            ber = safe_float((row or {}).get("ber_mean"))
            if ber is None:
                continue
            points.append(
                {
                    "group": group,
                    "condition": label,
                    "ber": ber,
                    "decode_success": safe_float((row or {}).get("decode_success_mean")),
                }
            )
    return points


def build_task_correctness_vs_reliability_plot(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    experiment = "realistic_cognitive_asymmetry"
    points: list[dict[str, Any]] = []
    for group in REALISTIC_TASK_GROUPS:
        no_drift = select_summary_row(summaries, experiment=experiment, group=group, condition="no_drift")
        llm_judge_correct = safe_float((no_drift or {}).get("llm_judge_correct_mean"))
        llm_judge_score = safe_float((no_drift or {}).get("llm_judge_score_mean"))
        if group == "G1":
            reliability = None
        else:
            ber = safe_float((no_drift or {}).get("ber_mean"))
            reliability = None if ber is None else 1.0 - ber
        points.append(
            {
                "experiment": experiment,
                "group": group,
                "task_correctness": llm_judge_correct,
                "communication_reliability": reliability,
                "llm_judge_score": llm_judge_score,
                "legacy_task_f1": safe_float((no_drift or {}).get("task_f1_mean")),
                "legacy_task_em": safe_float((no_drift or {}).get("task_em_mean")),
                "point_role": "task_upper_bound_reference" if group == "G1" else "joint_task_protocol_point",
            }
        )
    return points


def build_task_vs_reliability_plot(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_task_correctness_vs_reliability_plot(summaries)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
