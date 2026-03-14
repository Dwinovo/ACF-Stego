from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

plt.style.use("seaborn-v0_8-whitegrid")

GROUP_COLORS = {
    "G1": "#6c757d",
    "G2": "#d1495b",
    "G3": "#edae49",
    "G4": "#00798c",
    "G5": "#30638e",
    "G6": "#3b8b5d",
    "G7": "#8f4e99",
}

METHOD_LABELS = {
    "G1": "Normal",
    "G2": "DISCOP",
    "G3": "METEOR",
    "G4": "ASYMMETRIC",
    "G5": "ASYM+RET",
    "G6": "DISCOP+RET",
    "G7": "METEOR+RET",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from V2 aggregated artifacts.")
    parser.add_argument("--suffix", default="", help="Optional suffix matching analyze_v2_outputs artifacts.")
    parser.add_argument(
        "--formats",
        default="pdf",
        help="Comma-separated output formats. Example: pdf,png",
    )
    return parser.parse_args()


def normalized_suffix(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("__"):
        return text
    return "__" + text.replace(" ", "_")


def parse_formats(raw: str) -> list[str]:
    formats: list[str] = []
    for piece in str(raw or "").split(","):
        item = piece.strip().lower()
        if item and item not in formats:
            formats.append(item)
    return formats or ["pdf"]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    ensure_dir(output_dir)
    written: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def summary_lookup(rows: list[dict[str, Any]], experiment: str, group: str, condition: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            str(row.get("experiment", "")).strip() == experiment
            and str(row.get("group", "")).strip() == group
            and str(row.get("condition", "")).strip() == condition
        ):
            return row
    return None


def float_field(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def build_figure1_controlled_asymmetry(controlled: list[dict[str, Any]]) -> plt.Figure | None:
    experiment = "controlled_cognitive_asymmetry"
    groups = ["G2", "G3", "G4"]
    labels = [METHOD_LABELS[group] for group in groups]
    no_drift_vals: list[float] = []
    no_drift_errs: list[float] = []
    drift_vals: list[float] = []
    drift_errs: list[float] = []
    for group in groups:
        no_drift = summary_lookup(controlled, experiment, group, "no_drift")
        drift = summary_lookup(controlled, experiment, group, "drift_recent3")
        no_drift_vals.append(float_field(no_drift, "ber_mean") or 0.0)
        no_drift_errs.append(float_field(no_drift, "ber_std") or 0.0)
        drift_vals.append(float_field(drift, "ber_mean") or 0.0)
        drift_errs.append(float_field(drift, "ber_std") or 0.0)
    if not controlled:
        return None

    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x - width / 2, no_drift_vals, width, yerr=no_drift_errs, label="No Drift", color="#7fb069", capsize=4)
    ax.bar(x + width / 2, drift_vals, width, yerr=drift_errs, label="Drift Recent3", color="#d1495b", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False)
    return fig


def build_figure2_task_vs_communication(realistic: list[dict[str, Any]]) -> plt.Figure | None:
    experiment = "realistic_cognitive_asymmetry"
    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    rows = [summary_lookup(realistic, experiment, group, "no_drift") for group in groups]
    if not any(rows):
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for group, row in zip(groups, rows):
        x_value = float_field(row, "llm_judge_correct_mean")
        if x_value is None:
            continue
        if group == "G1":
            y_value = 1.02
            ax.scatter(
                [x_value],
                [y_value],
                s=110,
                marker="^",
                facecolors="none",
                edgecolors=GROUP_COLORS[group],
                linewidths=1.8,
                label=f"{group} Task Ref",
            )
            ax.annotate("G1", (x_value, y_value), textcoords="offset points", xytext=(6, -2), fontsize=10)
            continue
        ber = float_field(row, "ber_mean")
        if ber is None:
            continue
        y_value = 1.0 - ber
        ax.scatter([x_value], [y_value], s=130, color=GROUP_COLORS[group], label=group)
        ax.annotate(group, (x_value, y_value), textcoords="offset points", xytext=(6, 6), fontsize=10)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Task Correctness (LLMJudgeCorrect)")
    ax.set_ylabel("Communication Reliability (1 - BER)")
    legend_handles = []
    legend_labels = []
    for group in groups:
        if group == "G1":
            continue
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS[group], markersize=9)
        )
        legend_labels.append(f"{group} {METHOD_LABELS[group]}")
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color=GROUP_COLORS["G1"],
            markerfacecolor="none",
            linewidth=0,
            markersize=9,
            markeredgewidth=1.6,
        )
    )
    legend_labels.append("G1 Normal (task reference)")
    ax.legend(legend_handles, legend_labels, frameon=False, loc="lower right")
    return fig


def build_figure3_drift_severity_sweep(sweep_points: list[dict[str, Any]]) -> plt.Figure | None:
    if not sweep_points:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for group in ["G2", "G3", "G4"]:
        group_points = [point for point in sweep_points if str(point.get("group")) == group]
        if not group_points:
            continue
        group_points.sort(key=lambda point: -(int(point.get("decoder_sessions_kept") or 0)))
        x_values = [int(point.get("decoder_sessions_kept") or 0) for point in group_points]
        y_values = [float(point.get("ber") or 0.0) for point in group_points]
        ax.plot(x_values, y_values, marker="o", linewidth=2.2, color=GROUP_COLORS[group], label=METHOD_LABELS[group])
    ax.set_xlabel("Decoder Recent Sessions Kept")
    ax.set_ylabel("BER")
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks(sorted({int(point.get('decoder_sessions_kept') or 0) for point in sweep_points}, reverse=True))
    ax.legend(frameon=False)
    return fig


def sweep_points_from_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("experiment", "")).strip() != "controlled_drift_severity_sweep":
            continue
        try:
            ber = float(row.get("ber_mean"))
        except (TypeError, ValueError):
            continue
        try:
            decoder_sessions_kept = int(round(float(row.get("decode_recent_sessions_kept_mean"))))
        except (TypeError, ValueError):
            condition = str(row.get("condition", "")).strip()
            if condition == "no_drift":
                decoder_sessions_kept = int(config.LONGMEMEVAL_WINDOW_SESSIONS)
            else:
                digits = "".join(char for char in condition if char.isdigit())
                if not digits:
                    continue
                decoder_sessions_kept = int(digits)
        points.append(
            {
                "group": str(row.get("group", "")).strip(),
                "condition": str(row.get("condition", "")).strip(),
                "decoder_sessions_kept": decoder_sessions_kept,
                "ber": ber,
                "decode_success": row.get("decode_success_mean"),
            }
        )
    return points


def build_figure4_capacity_length(realistic: list[dict[str, Any]]) -> plt.Figure | None:
    experiment = "realistic_cognitive_asymmetry"
    task_groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    stego_groups = ["G2", "G3", "G4", "G5", "G6", "G7"]
    task_rows = [summary_lookup(realistic, experiment, group, "no_drift") for group in task_groups]
    stego_rows = [summary_lookup(realistic, experiment, group, "no_drift") for group in stego_groups]
    if not any(task_rows) and not any(stego_rows):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    token_means = [float_field(row, "generated_token_count_mean") or 0.0 for row in task_rows]
    token_errs = [float_field(row, "generated_token_count_std") or 0.0 for row in task_rows]
    axes[0].bar(
        [METHOD_LABELS[group] for group in task_groups],
        token_means,
        yerr=token_errs,
        color=[GROUP_COLORS[group] for group in task_groups],
        capsize=4,
    )
    axes[0].set_ylabel("Generated Tokens")
    axes[0].tick_params(axis="x", rotation=25)

    cap_means = [float_field(row, "embedding_capacity_mean") or 0.0 for row in stego_rows]
    cap_errs = [float_field(row, "embedding_capacity_std") or 0.0 for row in stego_rows]
    axes[1].bar(
        [METHOD_LABELS[group] for group in stego_groups],
        cap_means,
        yerr=cap_errs,
        color=[GROUP_COLORS[group] for group in stego_groups],
        capsize=4,
    )
    axes[1].set_ylabel("Embedding Capacity (bits / 1k tokens)")
    axes[1].tick_params(axis="x", rotation=25)
    return fig


def build_appendix_judge_distribution(raw_records: list[dict[str, Any]]) -> plt.Figure | None:
    relevant = [
        record
        for record in raw_records
        if str(record.get("experiment", "")).strip() == "realistic_cognitive_asymmetry"
        and str(record.get("condition", "")).strip() == "no_drift"
        and record.get("llm_judge_score") is not None
    ]
    if not relevant:
        return None

    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    score_levels = [0, 1, 2]
    score_colors = {0: "#d1495b", 1: "#edae49", 2: "#7fb069"}
    counts = {group: {score: 0 for score in score_levels} for group in groups}
    for record in relevant:
        group = str(record.get("group", "")).strip()
        if group not in counts:
            continue
        try:
            score = int(record.get("llm_judge_score"))
        except (TypeError, ValueError):
            continue
        if score in counts[group]:
            counts[group][score] += 1

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bottoms = np.zeros(len(groups))
    x = np.arange(len(groups))
    for score in score_levels:
        values = [counts[group][score] for group in groups]
        total_values = [sum(counts[group].values()) for group in groups]
        fractions = [value / total if total else 0.0 for value, total in zip(values, total_values)]
        ax.bar(x, fractions, bottom=bottoms, color=score_colors[score], label=f"Score {score}")
        bottoms += np.array(fractions)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[group] for group in groups])
    ax.set_ylabel("Fraction of Judged Outputs")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    return fig


def build_appendix_entropy_stats(realistic: list[dict[str, Any]]) -> plt.Figure | None:
    experiment = "realistic_cognitive_asymmetry"
    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    rows = [summary_lookup(realistic, experiment, group, "no_drift") for group in groups]
    if not any(rows):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    entropy_means = [float_field(row, "average_entropy_mean") or 0.0 for row in rows]
    entropy_errs = [float_field(row, "average_entropy_std") or 0.0 for row in rows]
    axes[0].bar(
        [METHOD_LABELS[group] for group in groups],
        entropy_means,
        yerr=entropy_errs,
        color=[GROUP_COLORS[group] for group in groups],
        capsize=4,
    )
    axes[0].set_ylabel("Average Entropy")
    axes[0].tick_params(axis="x", rotation=25)

    consumed_groups = ["G2", "G3", "G4", "G5", "G6", "G7"]
    consumed_rows = [summary_lookup(realistic, experiment, group, "no_drift") for group in consumed_groups]
    consumed_means = [float_field(row, "consumed_bits_mean") or 0.0 for row in consumed_rows]
    consumed_errs = [float_field(row, "consumed_bits_std") or 0.0 for row in consumed_rows]
    axes[1].bar(
        [METHOD_LABELS[group] for group in consumed_groups],
        consumed_means,
        yerr=consumed_errs,
        color=[GROUP_COLORS[group] for group in consumed_groups],
        capsize=4,
    )
    axes[1].set_ylabel("Consumed Bits")
    axes[1].tick_params(axis="x", rotation=25)
    return fig


def build_appendix_summary_asymmetry(controlled: list[dict[str, Any]], controlled_summary: list[dict[str, Any]]) -> plt.Figure | None:
    groups = ["G2", "G3", "G4"]
    no_drift_vals: list[float] = []
    no_drift_errs: list[float] = []
    summary_vals: list[float] = []
    summary_errs: list[float] = []
    for group in groups:
        no_drift = summary_lookup(controlled, "controlled_cognitive_asymmetry", group, "no_drift")
        summary_only = summary_lookup(controlled_summary, "controlled_summary_asymmetry", group, "summary_only_enc")
        no_drift_vals.append(float_field(no_drift, "ber_mean") or 0.0)
        no_drift_errs.append(float_field(no_drift, "ber_std") or 0.0)
        summary_vals.append(float_field(summary_only, "ber_mean") or 0.0)
        summary_errs.append(float_field(summary_only, "ber_std") or 0.0)
    if not controlled and not controlled_summary:
        return None

    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x - width / 2, no_drift_vals, width, yerr=no_drift_errs, label="No Drift", color="#7fb069", capsize=4)
    ax.bar(
        x + width / 2,
        summary_vals,
        width,
        yerr=summary_errs,
        label="Summary Only Enc",
        color="#30638e",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[group] for group in groups])
    ax.set_ylabel("BER")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False)
    return fig


def main() -> None:
    args = parse_args()
    suffix = normalized_suffix(args.suffix)
    formats = parse_formats(args.formats)
    table_dir = PROJECT_ROOT / config.TABLE_V2_DIR
    figure_dir = PROJECT_ROOT / config.FIGURE_V2_DIR

    controlled_main_summary = load_json_list(table_dir / f"controlled_summary{suffix}.json")
    controlled_summary_asymmetry = load_json_list(table_dir / f"controlled_summary_summary{suffix}.json")
    controlled_sweep_summary = load_json_list(table_dir / f"controlled_sweep_summary{suffix}.json")
    realistic_summary = load_json_list(table_dir / f"realistic_summary{suffix}.json")
    all_raw_records = load_json_list(table_dir / f"all_raw_records{suffix}.json")
    sweep_plot = load_json_list(table_dir / f"plot_controlled_drift_severity_sweep{suffix}.json")
    if not sweep_plot:
        sweep_plot = sweep_points_from_summary(controlled_sweep_summary)

    written: dict[str, list[str]] = {}

    figure_builders = [
        ("figure1_controlled_asymmetry_ber", build_figure1_controlled_asymmetry(controlled_main_summary)),
        ("figure2_task_vs_communication", build_figure2_task_vs_communication(realistic_summary)),
        (
            "figure3_ber_vs_drift_severity",
            build_figure3_drift_severity_sweep(sweep_plot or controlled_sweep_summary),
        ),
        ("figure4_token_length_capacity", build_figure4_capacity_length(realistic_summary)),
        ("appendix_figure_a_judge_score_distribution", build_appendix_judge_distribution(all_raw_records)),
        ("appendix_figure_b_entropy_generation_stats", build_appendix_entropy_stats(realistic_summary)),
        (
            "appendix_figure_c_summary_asymmetry_ber",
            build_appendix_summary_asymmetry(controlled_main_summary, controlled_summary_asymmetry),
        ),
    ]

    for stem, figure in figure_builders:
        if figure is None:
            print(f"[skip] {stem}: missing required data")
            continue
        written[stem] = save_figure(figure, figure_dir, f"{stem}{suffix}", formats)
        print(f"[ok] {stem}: {', '.join(written[stem])}")

    manifest = {
        "suffix": suffix,
        "formats": formats,
        "figure_dir": str(figure_dir),
        "written": written,
    }
    ensure_dir(figure_dir)
    (figure_dir / f"figure_manifest{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
