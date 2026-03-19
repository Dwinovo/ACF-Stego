from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.tools import analysis_tools

EXPERIMENT_OUTPUTS = {
    "realistic": PROJECT_ROOT / config.OUTPUT_V2_REALISTIC_DIR,
    "controlled": PROJECT_ROOT / config.OUTPUT_V2_CONTROLLED_DIR,
    "controlled_summary": PROJECT_ROOT / config.OUTPUT_V2_CONTROLLED_SUMMARY_DIR,
    "controlled_sweep": PROJECT_ROOT / config.OUTPUT_V2_CONTROLLED_SWEEP_DIR,
}

ALLOWED_GROUPS = {
    "controlled": {"G2", "G3", "G4"},
    "controlled_summary": {"G2", "G3", "G4"},
    "controlled_sweep": {"G2", "G3", "G4"},
    "realistic": {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"},
}

ALLOWED_CONDITIONS = {
    "controlled": {"no_drift", "drift_recent3"},
    "controlled_summary": {"summary_only_enc"},
    "controlled_sweep": {"no_drift"}
    | {f"drift_recent{keep}" for keep in config.LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS},
    "realistic": {"no_drift"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate V2 controlled/realistic experiment outputs.")
    parser.add_argument(
        "--experiment",
        choices=("all", "realistic", "controlled", "controlled_sweep", "controlled_summary"),
        default="all",
    )
    parser.add_argument("--only-group", choices=[f"group{i}" for i in range(1, 9)], default=None)
    parser.add_argument("--output-suffix", default="", help="Optional suffix to avoid overwriting full summaries.")
    return parser.parse_args()


def selected_experiments(experiment: str) -> list[str]:
    if experiment == "all":
        return ["controlled", "controlled_summary", "controlled_sweep", "realistic"]
    return [experiment]


def suffix_text(args: argparse.Namespace) -> str:
    suffixes: list[str] = []
    if args.only_group:
        suffixes.append(args.only_group)
    raw = str(args.output_suffix or "").strip()
    if raw:
        suffixes.append(raw.strip().replace(" ", "_"))
    return ("__" + "__".join(suffixes)) if suffixes else ""


def filter_records_for_experiment(experiment_key: str, records: list[dict]) -> list[dict]:
    allowed_groups = ALLOWED_GROUPS[experiment_key]
    allowed_conditions = ALLOWED_CONDITIONS[experiment_key]
    return [
        record
        for record in records
        if str(record.get("group", "")).strip() in allowed_groups
        and str(record.get("condition", "no_drift")).strip() in allowed_conditions
    ]


def write_experiment_artifacts(
    *,
    experiment_key: str,
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    instance_rows = analysis_tools.build_instance_means(records)
    summaries = analysis_tools.summarize_instance_means(instance_rows)
    return instance_rows, summaries


def remove_obsolete_artifacts(output_dir: Path, name_suffix: str) -> None:
    obsolete_names = [
        f"paper_table_controlled{name_suffix}.csv",
        f"paper_table_controlled{name_suffix}.md",
        f"paper_table_controlled_summary{name_suffix}.csv",
        f"paper_table_realistic_task{name_suffix}.csv",
        f"paper_table_realistic_task{name_suffix}.md",
        f"paper_table_realistic_protocol{name_suffix}.csv",
        f"paper_table_realistic_protocol{name_suffix}.md",
        f"paper_table_realistic_llm{name_suffix}.md",
        f"paper_table_realistic_llm{name_suffix}.csv",
        f"paper_table_ber_vs_decoder_sessions{name_suffix}.csv",
        f"plot_task_vs_reliability{name_suffix}.json",
        f"plot_ber_vs_condition{name_suffix}.json",
        f"plot_controlled_summary_asymmetry{name_suffix}.json",
        f"plot_controlled_drift_severity_sweep{name_suffix}.json",
        f"plot_task_correctness_vs_reliability{name_suffix}.json",
        f"controlled_summary{name_suffix}.json",
        f"controlled_summary_summary{name_suffix}.json",
        f"controlled_sweep_summary{name_suffix}.json",
        f"realistic_summary{name_suffix}.json",
        f"all_raw_records{name_suffix}.json",
    ]
    for name in obsolete_names:
        path = output_dir / name
        if path.exists():
            path.unlink()

def write_controlled_drift_ber_pdf(
    summaries: list[dict],
    *,
    output_path: Path,
) -> bool:
    points = analysis_tools.build_controlled_drift_severity_sweep_plot(summaries)
    if not points:
        return False

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as exc:
        print(f"[figure] skip ber-vs-decoder-sessions: {exc}")
        return False

    # (group, label, color, marker, linestyle, acf_k_filter, line_width, marker_size, band_alpha, zorder)
    protocol_specs = (
        ("G2", "DISCOP", "#d62728", "o", (0, (3, 1.5)), None, 1.9, 6.8, 0.16, 4),  # red dashed
        ("G3", "METEOR", "#1f77b4", "o", (0, (3, 1.5)), None, 1.9, 6.8, 0.16, 4),  # blue dashed
        ("G4", "ACF (k=8)", "#7f3fbf", "o", "-", 8, 1.9, 6.8, 0.12, 5),  # purple solid
        ("G4", "ACF (k=16)", "#2ca02c", "o", "-", 16, 1.9, 6.8, 0.12, 6),  # green
    )

    decoder_sessions = sorted({
        int(point.get("decoder_sessions_kept", 0) or 0)
        for point in points if int(point.get("decoder_sessions_kept", 0) or 0) > 0
    })
    if not decoder_sessions:
        return False
        
    window_sessions = int(config.LONGMEMEVAL_WINDOW_SESSIONS)
    truncation_ticks = list(range(0, max(1, window_sessions)))

    # Use Times New Roman as the global figure font.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_y_upper: list[float] = []

    for (
        group,
        label,
        color,
        marker,
        linestyle,
        acf_k_filter,
        line_width,
        marker_size,
        band_alpha,
        zorder,
    ) in protocol_specs:
        group_points = [
            p
            for p in points
            if str(p.get("group", "")).strip() == group
            and (
                acf_k_filter is None
                or int(p.get("acf_k", 0) or 0) == int(acf_k_filter)
            )
        ]
        if not group_points:
            continue

        ber_by_truncation = defaultdict(list)
        ber_std_by_truncation = defaultdict(list)
        for row in group_points:
            session_kept = int(row.get("decoder_sessions_kept", 0) or 0)
            ber_value = max(0.0, float(row.get("ber", 0.0) or 0.0) * 100.0)
            ber_std = max(0.0, float(row.get("ber_std", 0.0) or 0.0) * 100.0)
            if session_kept > 0:
                truncation = max(0, window_sessions - session_kept)
                ber_by_truncation[truncation].append(ber_value)
                ber_std_by_truncation[truncation].append(ber_std)

        x_values = sorted(ber_by_truncation.keys())
        if not x_values:
            continue

        y_values = [statistics.mean(ber_by_truncation[x]) for x in x_values]
        y_stds = [
            statistics.mean(ber_std_by_truncation[x]) if ber_std_by_truncation[x] else 0.0
            for x in x_values
        ]
        y_lower = [max(0.0, y - s) for y, s in zip(y_values, y_stds)]
        y_upper = [min(100.0, y + s) for y, s in zip(y_values, y_stds)]
        x_plot = [float(x) for x in x_values]
        y_plot = [float(y) for y in y_values]
        y_lower_plot = [float(y) for y in y_lower]
        y_upper_plot = [float(y) for y in y_upper]
        all_y_upper.extend(y_upper)

        ax.fill_between(
            x_plot,
            y_lower_plot,
            y_upper_plot,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=zorder - 1,
        )

        ax.plot(
            x_plot, y_plot,
            marker=marker, linestyle=linestyle,
            linewidth=line_width, markersize=marker_size,
            color=color, label=label,
            zorder=zorder,
            markeredgewidth=0.6, markeredgecolor="white"
        )

    ax.set_xlabel(r"Context Truncation ($\Delta$ Sessions)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("BER (%)", fontsize=10.5, fontweight="bold")

    ax.set_xticks(truncation_ticks)
    ax.set_xlim(-0.15, max(truncation_ticks) + 0.20)

    y_max = max(all_y_upper) if all_y_upper else 0.0
    y_limit = max(10.0, float(int((y_max + 9.999) // 10) * 10))
    if y_limit < y_max:
        y_limit += 10.0
    y_limit = min(100.0, y_limit)
    # Leave a tiny negative margin so the BER=0 line is slightly above the plot bottom.
    y_lower_margin = 2.0
    ax.set_ylim(-y_lower_margin, y_limit)
    ax.set_yticks(list(range(0, int(y_limit) + 1, 10)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}%"))

    ax.grid(True, axis="y", linestyle="-", linewidth=0.58, color="#ababab", alpha=0.72)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.46, color="#cdcdcd", alpha=0.58)

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(colors="#000000", labelsize=10.0)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#b6b6b6",
        prop={"family": "serif", "size": 8.0, "weight": "bold"},
        handlelength=3.6,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.margins(x=0.02, y=0.02)
    fig.tight_layout(pad=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return True


def write_controlled_drift_ber_source_table(
    summaries: list[dict],
    *,
    output_path: Path,
) -> bool:
    points = analysis_tools.build_controlled_drift_severity_sweep_plot(summaries)
    if not points:
        return False

    protocol_specs = (
        ("G2", "DISCOP", None),
        ("G3", "METEOR", None),
        ("G4", "ACF (k=8)", 8),
        ("G4", "ACF (k=16)", 16),
    )
    window_sessions = int(config.LONGMEMEVAL_WINDOW_SESSIONS)

    headers = [
        "Protocol",
        "Group",
        "DecoderSessionsKept",
        "ContextTruncationDelta",
        "BER(%, mean±std)",
        "DSR(%, mean±std)",
    ]
    rows: list[list[str]] = []

    for group, label, acf_k_filter in protocol_specs:
        group_points = [
            point
            for point in points
            if str(point.get("group", "")).strip() == group
            and (
                acf_k_filter is None
                or int(point.get("acf_k", 0) or 0) == int(acf_k_filter)
            )
        ]
        if not group_points:
            continue

        ber_by_truncation: dict[int, list[float]] = defaultdict(list)
        ber_std_by_truncation: dict[int, list[float]] = defaultdict(list)
        dsr_by_truncation: dict[int, list[float]] = defaultdict(list)
        dsr_std_by_truncation: dict[int, list[float]] = defaultdict(list)

        for point in group_points:
            session_kept = int(point.get("decoder_sessions_kept", 0) or 0)
            if session_kept <= 0:
                continue
            truncation = max(0, window_sessions - session_kept)
            ber_by_truncation[truncation].append(max(0.0, float(point.get("ber", 0.0) or 0.0) * 100.0))
            ber_std_by_truncation[truncation].append(max(0.0, float(point.get("ber_std", 0.0) or 0.0) * 100.0))
            dsr_value = analysis_tools.safe_float(point.get("decode_success"))
            if dsr_value is not None:
                dsr_by_truncation[truncation].append(max(0.0, min(1.0, dsr_value)) * 100.0)
            dsr_std_value = analysis_tools.safe_float(point.get("decode_success_std"))
            if dsr_std_value is not None:
                dsr_std_by_truncation[truncation].append(max(0.0, dsr_std_value) * 100.0)

        for truncation in sorted(ber_by_truncation.keys()):
            ber_mean = statistics.mean(ber_by_truncation[truncation]) if ber_by_truncation[truncation] else 0.0
            ber_std = statistics.mean(ber_std_by_truncation[truncation]) if ber_std_by_truncation[truncation] else 0.0
            dsr_mean = statistics.mean(dsr_by_truncation[truncation]) if dsr_by_truncation[truncation] else 0.0
            dsr_std = (
                statistics.mean(dsr_std_by_truncation[truncation]) if dsr_std_by_truncation[truncation] else 0.0
            )
            decoder_sessions_kept = max(0, window_sessions - truncation)
            rows.append(
                [
                    label,
                    group,
                    str(int(decoder_sessions_kept)),
                    str(int(truncation)),
                    f"{ber_mean:.4f} ± {ber_std:.4f}",
                    f"{dsr_mean:.4f} ± {dsr_std:.4f}",
                ]
            )

    if not rows:
        return False
    analysis_tools.write_csv(output_path, headers, rows)
    return True


def write_realistic_semantic_vs_reliability_pdf(
    summaries: list[dict],
    *,
    output_path: Path,
    realistic_table_csv: Path | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[figure] skip semantic-vs-reliability scatter (matplotlib unavailable): {exc}")
        return False

    realistic_rows = [
        row
        for row in summaries
        if str(row.get("experiment", "")).strip() == "realistic_cognitive_asymmetry"
        and str(row.get("condition", "")).strip() == "no_drift"
    ]
    if not realistic_rows:
        return False

    def _extract_first_number(text: str) -> float | None:
        raw = str(text or "").strip()
        if not raw or raw == "---":
            return None
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _limits_from_csv(path: Path | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if path is None or not path.exists():
            return None, None

        x_vals: list[float] = []
        y_vals: list[float] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    score = _extract_first_number(str(row.get("Score", "")))
                    if score is not None:
                        x_vals.append(score)
                    ber_percent = _extract_first_number(str(row.get("BER", "")))
                    if ber_percent is not None:
                        y_vals.append(1.0 - (ber_percent / 100.0))
        except Exception:
            return None, None

        def _make_limits(
            values: list[float],
            *,
            min_span: float,
            pad_ratio: float,
            lower: float | None = None,
            upper: float | None = None,
        ) -> tuple[float, float] | None:
            if not values:
                return None
            v_min = min(values)
            v_max = max(values)
            span = max(v_max - v_min, min_span)
            center = (v_min + v_max) / 2.0
            pad = span * pad_ratio
            lo = center - span / 2.0 - pad
            hi = center + span / 2.0 + pad
            if lower is not None:
                lo = max(lower, lo)
            if upper is not None:
                hi = min(upper, hi)
            if hi <= lo:
                hi = lo + min_span
            return lo, hi

        x_lim = _make_limits(x_vals, min_span=0.25, pad_ratio=0.14)
        y_lim = _make_limits(y_vals, min_span=0.18, pad_ratio=0.16, lower=0.0, upper=1.02)
        return x_lim, y_lim

    def _match(group: str, acf_k: int | None = None) -> dict | None:
        for row in realistic_rows:
            if str(row.get("group", "")).strip() != group:
                continue
            row_k = analysis_tools.normalize_acf_k(row.get("acf_k"))
            if row_k == analysis_tools.normalize_acf_k(acf_k):
                return row
        return None

    # Baseline vertical lines
    normal_row = _match("G1", None)
    normal_ret_row = _match("G8", None)
    x_normal = analysis_tools.safe_float((normal_row or {}).get("llm_judge_score_mean"))
    x_normal_ret = analysis_tools.safe_float((normal_ret_row or {}).get("llm_judge_score_mean"))

    # (label, row, family, has_ret)
    points_spec: list[tuple[str, dict | None, str, bool]] = [
        ("DISCOP", _match("G2", None), "discop", False),
        ("METEOR", _match("G3", None), "meteor", False),
        ("ACF", _match("G4", 8), "acf", False),
        ("ACF", _match("G4", 12), "acf", False),
        ("ACF", _match("G4", 16), "acf", False),
        ("DISCOP+RET", _match("G6", None), "discop", True),
        ("METEOR+RET", _match("G7", None), "meteor", True),
        ("ACF+RET", _match("G5", 8), "acf", True),
        ("ACF+RET", _match("G5", 12), "acf", True),
        ("ACF+RET", _match("G5", 16), "acf", True),
    ]

    family_color = {
        "discop": "#4783b5",
        "meteor": "#778899",
        "acf": "#f47e52",
    }
    family_marker = {
        "discop": "o",
        "meteor": "s",
        "acf": "^",
    }

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    valid_points = 0
    xs: list[float] = []
    ys: list[float] = []
    for _, row, family, has_ret in points_spec:
        score = analysis_tools.safe_float((row or {}).get("llm_judge_score_mean"))
        ber = analysis_tools.safe_float((row or {}).get("ber_mean"))
        if score is None or ber is None:
            continue
        x = score
        y = 1.0 - ber
        valid_points += 1
        xs.append(x)
        ys.append(y)
        color = family_color[family]
        marker = family_marker[family]
        facecolor = "none" if has_ret else color
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=72,
            edgecolors=color,
            facecolors=facecolor,
            linewidths=1.5,
            zorder=4,
        )
    if valid_points == 0:
        plt.close(fig)
        return False

    # Reference lines: Normal and Normal+RET
    if x_normal is not None:
        ax.axvline(x_normal, color="#8d8d8d", linestyle="--", linewidth=1.3, zorder=2)
    if x_normal_ret is not None:
        ax.axvline(x_normal_ret, color="#111111", linestyle="-.", linewidth=1.3, zorder=2)

    csv_xlim, csv_ylim = _limits_from_csv(realistic_table_csv)
    if csv_xlim is not None:
        ax.set_xlim(csv_xlim[0], csv_xlim[1])
    else:
        x_min = min(xs + [x_normal] if x_normal is not None else xs)
        x_max = max(xs + [x_normal] if x_normal is not None else xs)
        if x_normal_ret is not None:
            x_min = min(x_min, x_normal_ret)
            x_max = max(x_max, x_normal_ret)
        x_pad = max(0.05, (x_max - x_min) * 0.12)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if csv_ylim is not None:
        ax.set_ylim(csv_ylim[0], csv_ylim[1])
    else:
        y_min = min(ys)
        y_max = max(ys)
        y_pad = max(0.03, (y_max - y_min) * 0.15)
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.02, y_max + y_pad))

    ax.set_xlabel("Semantic Utility (Score)", fontsize=11, color="#000000")
    ax.set_ylabel("Communication Reliability (1 - BER)", fontsize=11, color="#000000")
    ax.tick_params(colors="#000000", labelsize=9.5)
    ax.grid(True, linestyle=":", linewidth=0.75, color="#d0d0d0", alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#111111")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4783b5", markeredgecolor="#4783b5", markersize=7, label="DISCOP"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#778899", markeredgecolor="#778899", markersize=7, label="METEOR"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#f47e52", markeredgecolor="#f47e52", markersize=7, label="ASYMMETRIC"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#333333", markeredgecolor="#333333", linestyle="none", markersize=7, label="No RET (filled)"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="none", markeredgecolor="#333333", linestyle="none", markersize=7, label="+RET (hollow)"),
    ]
    if x_normal is not None:
        legend_handles.append(
            Line2D([0], [0], color="#8d8d8d", linestyle="--", linewidth=1.3, label="Normal (No Stego)")
        )
    if x_normal_ret is not None:
        legend_handles.append(
            Line2D([0], [0], color="#111111", linestyle="-.", linewidth=1.3, label="Normal+RET")
        )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a slightly larger padding to keep extra whitespace above the legend.
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    return True


def _extract_first_number(text: str) -> float | None:
    raw = str(text or "").strip()
    if not raw or raw == "---":
        return None
    match = re.search(r"[-+]?\d*\.?\d+", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def write_realistic_dual_axis_tradeoff_pdf(
    *,
    realistic_table_csv: Path,
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Bbox
    except Exception as exc:
        print(f"[figure] skip grouped-bar tradeoff (matplotlib unavailable): {exc}")
        return False

    if not realistic_table_csv.exists():
        return False

    row_by_protocol: dict[str, dict[str, str]] = {}
    with realistic_table_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protocol = str(row.get("Protocol", "")).strip()
            if protocol:
                row_by_protocol[protocol] = {k: str(v) for k, v in row.items()}

    normal_score = _extract_first_number((row_by_protocol.get("Normal (No Stego)", {}) or {}).get("Score", ""))
    normal_ret_score = _extract_first_number((row_by_protocol.get("Normal+RET", {}) or {}).get("Score", ""))

    # Plot protocols only; baselines are shown as reference lines.
    protocol_order: list[str] = [
        "DISCOP",
        "DISCOP+RET",
        "METEOR",
        "METEOR+RET",
        "ACF (k=16)",
        "ACF+RET (k=16)",
    ]

    rows: list[dict[str, object]] = []
    for protocol in protocol_order:
        row = row_by_protocol.get(protocol)
        if row is None:
            continue
        score = _extract_first_number(row.get("Score", ""))
        ber = _extract_first_number(row.get("BER", ""))
        if score is None:
            continue
        reliability = None if ber is None else max(0.0, min(1.0, 1.0 - (ber / 100.0)))
        rows.append(
            {
                "protocol": protocol,
                "score": max(0.0, min(1.0, score)),
                "ber_percent": ber,
                "reliability": reliability,
                "is_ret": "+RET" in protocol,
            }
        )

    if len(rows) < 3:
        return False

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fig, ax_left = plt.subplots(figsize=(9.8, 4.9))
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("white")
    ax_right = ax_left.twinx()
    ax_right.patch.set_alpha(0.0)

    x_positions = list(range(len(rows)))
    bar_width = 0.34
    score_color = "#f7b89d"
    reliability_color = "#aac7e8"

    def _label(protocol: str) -> str:
        if protocol.startswith("ACF+RET (k="):
            k_text = protocol.replace("ACF+RET ", "")
            return f"ACF {k_text}\nRET"
        if protocol.startswith("ACF (k="):
            return protocol
        text = protocol.replace("Normal (No Stego)", "Normal").replace("+RET", "\nRET")
        if "(k=" in text:
            text = text.replace(" (", "\n(")
        return text

    for idx, row in enumerate(rows):
        score = float(row["score"])
        reliability = row["reliability"]

        if reliability is not None:
            ax_left.bar(
                idx - bar_width / 2.0,
                float(reliability),
                width=bar_width,
                color=reliability_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=3,
            )
        ax_right.bar(
            idx + bar_width / 2.0,
            score,
            width=bar_width,
            color=score_color,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )

    normal_line_color = "#2155A6"
    normal_ret_line_color = "#A35A1F"
    if normal_score is not None:
        ax_right.axhline(normal_score, color=normal_line_color, linestyle="--", linewidth=1.4, zorder=10)
    if normal_ret_score is not None:
        ax_right.axhline(normal_ret_score, color=normal_ret_line_color, linestyle="--", linewidth=1.4, zorder=10)

    # Tight horizontal margins: keep only a slim breathing room around outer bars.
    side_pad = 0.08
    x_min = -bar_width - side_pad
    x_max = (len(rows) - 1) + bar_width + side_pad
    ax_left.set_xlim(x_min, x_max)
    ax_right.set_xlim(x_min, x_max)
    ax_left.set_ylim(0.0, 1.08)
    ax_right.set_ylim(0.0, 1.08)
    common_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax_left.set_yticks(common_ticks)
    ax_right.set_yticks(common_ticks)

    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels([_label(str(item["protocol"])) for item in rows], fontsize=18.0, rotation=0, ha="center")
    for tick in ax_left.get_xticklabels():
        tick.set_multialignment("center")
        tick.set_fontweight("bold")
    ax_left.set_xlabel("")
    ax_left.set_ylabel("Channel Reliability (1−BER)", fontsize=13.5, color="#000000", fontweight="bold")
    ax_right.set_ylabel("Semantic Utility (Score)", fontsize=13.5, color="#000000", fontweight="bold")

    ax_left.tick_params(axis="x", colors="#000000", labelsize=13.0, width=1.1, length=5.0)
    ax_left.tick_params(axis="y", colors="#000000", labelsize=14.5, width=1.3, length=6.0)
    ax_right.tick_params(axis="y", colors="#000000", labelsize=14.5, width=1.3, length=6.0)
    for tick in ax_left.get_yticklabels():
        tick.set_fontweight("bold")
    for tick in ax_right.get_yticklabels():
        tick.set_fontweight("bold")
    ax_left.grid(False)
    frame_color = "#000000"
    side_lw = 0.8
    horizontal_lw = 1.3
    frame_zorder = 50

    # Hide all default spines, then draw the frame on the top axis so it
    # always stays above bars from both y-axes.
    for spine_name in ("bottom", "top", "left", "right"):
        ax_left.spines[spine_name].set_visible(False)
        ax_right.spines[spine_name].set_visible(False)

    ax_right.hlines(
        [0.0, 1.0],
        x_min,
        x_max,
        colors=frame_color,
        linewidth=horizontal_lw,
        zorder=frame_zorder,
        clip_on=False,
    )
    ax_right.vlines(
        [x_min, x_max],
        ymin=0.0,
        ymax=1.0,
        colors=frame_color,
        linewidth=side_lw,
        zorder=frame_zorder,
        clip_on=False,
    )

    legend_handles = [
        Patch(facecolor=reliability_color, edgecolor="none", label="Channel Reliability (1−BER)"),
        Patch(facecolor=score_color, edgecolor="none", label="Semantic Utility (Score)"),
    ]
    if normal_score is not None:
        legend_handles.append(
            Line2D([0], [0], color=normal_line_color, linestyle="--", linewidth=1.4, label="Normal Score")
        )
    if normal_ret_score is not None:
        legend_handles.append(
            Line2D([0], [0], color=normal_ret_line_color, linestyle="--", linewidth=1.4, label="Normal+RET Score")
        )
    ax_left.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        prop={"family": "serif", "size": 15.5, "weight": "bold"},
        handletextpad=0.6,
        columnspacing=1.2,
    )

    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Extend only the top side of the tight bbox to add headroom above the legend.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer).padded(0.02)
        extra_top_inches = 0.1
        extended_bbox = Bbox.from_extents(
            tight_bbox.x0,
            tight_bbox.y0,
            tight_bbox.x1,
            tight_bbox.y1 + extra_top_inches,
        )
        fig.savefig(output_path, bbox_inches=extended_bbox)
    except Exception:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return True

def main() -> None:
    args = parse_args()
    groups = [args.only_group] if args.only_group else None
    chosen_experiments = selected_experiments(args.experiment)
    output_dir = PROJECT_ROOT / config.TABLE_V2_DIR
    name_suffix = suffix_text(args)
    remove_obsolete_artifacts(output_dir, name_suffix)

    combined_summaries: list[dict] = []

    for experiment_key in chosen_experiments:
        output_root = EXPERIMENT_OUTPUTS[experiment_key]
        raw_records = analysis_tools.load_records(output_root, groups=groups)
        records = filter_records_for_experiment(experiment_key, raw_records)
        dropped = len(raw_records) - len(records)
        if not records:
            print(f"[{experiment_key}] no records found under {output_root}")
            continue

        instance_rows, summaries = write_experiment_artifacts(
            experiment_key=experiment_key,
            records=records,
        )
        combined_summaries.extend(summaries)

        print(
            f"[{experiment_key}] records={len(records)} "
            f"instances={len(instance_rows)} summaries={len(summaries)}"
        )
        if dropped > 0:
            print(f"  filtered_out={dropped} stale_or_disallowed_records")
        for row in summaries:
            llm_suffix = ""
            llm_score = row.get("llm_judge_score")
            if llm_score is not None:
                llm_suffix = (
                    f" llm_judge_score={row.get('llm_judge_score')}"
                    f" llm_judge_correct={row.get('llm_judge_correct')}"
                )
            print(
                f"  {row['group']} {row['condition']}: "
                f"task_em={row.get('task_em')} task_f1={row.get('task_f1')} "
                f"ber={row.get('ber')} decode_success={row.get('decode_success')}"
                f"{llm_suffix}"
            )

    if not combined_summaries:
        print("No V2 records found.")
        print(f"No CSV tables written under: {output_dir}")
        return

    controlled_headers = [
        "Protocol",
        "BER @ Ideal",
        "Context Truncation BER",
        "Context Truncation DSR",
        "Context Summary BER",
        "Context Summary DSR",
    ]
    controlled_rows = analysis_tools.build_controlled_cognitive_asymmetry_table_rows(combined_summaries)
    analysis_tools.write_csv(
        output_dir / f"paper_table_controlled_cognitive_asymmetry{name_suffix}.csv",
        controlled_headers,
        controlled_rows,
    )

    realistic_headers = [
        "Protocol",
        "Acc",
        "Score",
        "F1",
        "Entropy(bits/token)",
        "BER",
        "DSR",
        "Nom.(bits/1kTok)",
        "Eff.(bits/1kTok)",
    ]
    realistic_rows = analysis_tools.build_realistic_integrated_table_rows(combined_summaries)
    realistic_csv_path = output_dir / f"paper_table_realistic_integrated{name_suffix}.csv"
    analysis_tools.write_csv(
        realistic_csv_path,
        realistic_headers,
        realistic_rows,
    )

    ber_source_table_path = output_dir / f"paper_table_ber_vs_decoder_sessions{name_suffix}.csv"
    if write_controlled_drift_ber_source_table(combined_summaries, output_path=ber_source_table_path):
        print(f"Wrote BER-vs-decoder-sessions source table to: {ber_source_table_path}")
    else:
        print("Skipped BER-vs-decoder-sessions source table (insufficient controlled_sweep data).")

    figure_path = PROJECT_ROOT / config.FIGURE_V2_DIR / f"figure_ber_vs_decoder_sessions{name_suffix}.pdf"
    if write_controlled_drift_ber_pdf(combined_summaries, output_path=figure_path):
        print(f"Wrote BER-vs-decoder-sessions figure to: {figure_path}")
    else:
        print("Skipped BER-vs-decoder-sessions figure (insufficient controlled_sweep data).")

    # Keep only the two paper figures from this script.
    scatter_path = PROJECT_ROOT / config.FIGURE_V2_DIR / f"figure_semantic_vs_reliability{name_suffix}.pdf"
    if scatter_path.exists():
        scatter_path.unlink()

    legacy_dual_axis_path = PROJECT_ROOT / config.FIGURE_V2_DIR / f"figure_tradeoff_dual_axis{name_suffix}.pdf"
    if legacy_dual_axis_path.exists():
        legacy_dual_axis_path.unlink()

    grouped_bar_path = PROJECT_ROOT / config.FIGURE_V2_DIR / f"figure_tradeoff_grouped_bar{name_suffix}.pdf"
    if write_realistic_dual_axis_tradeoff_pdf(
        realistic_table_csv=realistic_csv_path,
        output_path=grouped_bar_path,
    ):
        print(f"Wrote grouped-bar tradeoff figure to: {grouped_bar_path}")
    else:
        print("Skipped grouped-bar tradeoff figure (insufficient realistic data).")

    print(f"Wrote final CSV tables to: {output_dir}")


if __name__ == "__main__":
    main()
