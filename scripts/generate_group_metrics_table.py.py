from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager

def resolve_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "results" / "outputs").exists():
            return candidate
    raise FileNotFoundError("Cannot locate project root containing results/outputs")


PROJECT_ROOT = resolve_project_root()
OUTPUTS_ROOT = PROJECT_ROOT / "results" / "outputs"
TABLE_DIR = PROJECT_ROOT / "results" / "table"
PDF_PATH = TABLE_DIR / "starter_group_metrics_comparison.pdf"
CN_FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "SourceHanSansSC-Regular.otf"
CN_BOLD_FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "SourceHanSansSC-Bold.otf"

GROUP_DIRS = {
    "group1": OUTPUTS_ROOT / "group1",
    "group2": OUTPUTS_ROOT / "group2",
    "group3": OUTPUTS_ROOT / "group3",
    "group4": OUTPUTS_ROOT / "group4",
}

FILE_PATTERN = re.compile(r"-(\d+)-(\d+)\.json$")


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def parse_ids_from_filename(path: Path) -> tuple[int, int] | None:
    m = FILE_PATTERN.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_experiment_meta(outputs_root: Path) -> dict[str, str]:
    sample_meta: dict[str, Any] = {}
    for group in ("group1", "group2", "group3", "group4"):
        group_dir = outputs_root / group
        first_file = next(iter(sorted(group_dir.glob("*.json"))), None)
        if first_file is None:
            continue
        try:
            data = json.loads(first_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in (
            "temperature",
            "window_size",
            "rounds_per_dialog",
            "max_token",
            "top_k",
            "top_p",
            "stego_top_k",
            "stego_top_p",
            "stego_precision",
            "stego_secure_parameter",
            "stego_func_type",
        ):
            if key in data and key not in sample_meta:
                sample_meta[key] = data[key]

    top_k = sample_meta.get("top_k", sample_meta.get("stego_top_k", "N/A"))
    top_p = sample_meta.get("top_p", sample_meta.get("stego_top_p", "N/A"))
    return {
        "temperature": str(sample_meta.get("temperature", "N/A")),
        "window_size": str(sample_meta.get("window_size", "N/A")),
        "rounds_per_dialog": str(sample_meta.get("rounds_per_dialog", "N/A")),
        "max_token": str(sample_meta.get("max_token", "N/A")),
        "top_k": str(top_k),
        "top_p": str(top_p),
        "stego_precision": str(sample_meta.get("stego_precision", "N/A")),
        "stego_secure_parameter": str(sample_meta.get("stego_secure_parameter", "N/A")),
        "stego_func_type": str(sample_meta.get("stego_func_type", "N/A")),
    }


def collect_group_records(group_name: str, group_dir: Path) -> dict[int, dict[str, Any]]:
    by_starter: dict[int, dict[str, Any]] = {}

    for path in sorted(group_dir.glob("*.json")):
        parsed = parse_ids_from_filename(path)
        if parsed is None:
            continue
        starter_id, _trial_id = parsed

        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        starter_text = str(data.get("starter", "")).strip()

        entropy = _safe_float(metrics.get("weighted_average_entropy"))

        # group1 only has generate_time; other groups have encode/decode.
        encode = _safe_float(metrics.get("average_encode_time_per_token_seconds"))
        if encode is None and group_name == "group1":
            encode = _safe_float(metrics.get("average_generate_time_per_token_seconds"))

        decode = _safe_float(metrics.get("average_decode_time_per_token_seconds"))

        if starter_id not in by_starter:
            by_starter[starter_id] = {
                "starter": starter_text,
                "entropy": [],
                "encode": [],
                "decode": [],
            }
        elif not by_starter[starter_id].get("starter") and starter_text:
            by_starter[starter_id]["starter"] = starter_text

        if entropy is not None:
            by_starter[starter_id]["entropy"].append(entropy)
        if encode is not None:
            by_starter[starter_id]["encode"].append(encode)
        if decode is not None:
            by_starter[starter_id]["decode"].append(decode)

    return by_starter


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def avg_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def std_or_zero(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def format_mean_pm_std(values: list[float], digits: int = 2) -> str:
    if not values:
        return "-"
    m = avg_or_none(values)
    s = std_or_zero(values)
    if m is None:
        return "-"
    return f"{m:.{digits}f}\u00b1{s:.{digits}f}"


def format_mean_pm_std_scaled(values: list[float], scale_factor: float, digits: int = 2) -> str:
    if not values:
        return "-"
    scaled = [v * scale_factor for v in values]
    return format_mean_pm_std(scaled, digits=digits)


def starter_prefix(text: str, n: int | None = None) -> str:
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return "-"
    if n is None:
        return t
    return t[:n]


def save_table_png(
    headers: list[str],
    rows: list[dict[str, Any]],
    output_path: Path,
    experiment_meta: dict[str, str],
    highlight_cells: set[tuple[int, int]] | None = None,
) -> None:
    bold_font_prop = None
    if CN_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CN_FONT_PATH))
        cn_name = font_manager.FontProperties(fname=str(CN_FONT_PATH)).get_name()
        plt.rcParams["font.family"] = cn_name
        if CN_BOLD_FONT_PATH.exists():
            font_manager.fontManager.addfont(str(CN_BOLD_FONT_PATH))
            bold_font_prop = font_manager.FontProperties(fname=str(CN_BOLD_FONT_PATH))
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cell_text = [[str(row.get(h, "")) for h in headers] for row in rows]
    # Column 0 is StarterId (short code), no wrapping needed.
    nrows = max(len(rows), 1)
    ncols = len(headers)
    highlight_cells = highlight_cells or set()

    fig_w = max(14.0, ncols * 2.6)
    fig_h = max(8.0, nrows * 0.27 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
    ax.axis("off")
    ax.set_title(
        "Table 1. Group-wise comparison by Starter (mean±std over 3 trials)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    meta_line = (
        f"Settings: temperature={experiment_meta['temperature']}, window={experiment_meta['window_size']}, "
        f"top-k={experiment_meta['top_k']}, top-p={experiment_meta['top_p']}, max_new_tokens={experiment_meta['max_token']}, "
        f"rounds/dialog={experiment_meta['rounds_per_dialog']}, stego_precision={experiment_meta['stego_precision']}, "
        f"secure_parameter={experiment_meta['stego_secure_parameter']}, func_type={experiment_meta['stego_func_type']}"
    )
    group_line = (
        "组1：正常对话 ｜ 组2：对称隐写 ｜ 组3：非对称隐写+上下文一致 ｜ 组4：非对称隐写+上下文不一致"
    )
    ax.text(0.5, 0.988, meta_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3)
    ax.text(0.5, 0.962, group_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3)

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.93],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)

    # Header top/mid rules + vertical separators.
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = "LR"
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        cell.set_text_props(ha="center", va="center")
        if r == 0:
            cell.set_text_props(weight="bold", ha="center", va="center")
            if bold_font_prop is not None:
                cell.get_text().set_fontproperties(bold_font_prop)
            cell.get_text().set_fontsize(8)
            cell.set_facecolor("white")
            cell.visible_edges = "TLRB"
            cell.set_linewidth(1.2)
        if (r, c) in highlight_cells:
            cell.set_text_props(weight="bold", ha="center", va="center")
            if bold_font_prop is not None:
                cell.get_text().set_fontproperties(bold_font_prop)
            cell.get_text().set_fontsize(8)
        cell.get_text().set_wrap(True)
        # Keep StarterId column compact.
        if c == 0:
            cell.set_width(0.11)
        elif c == 1:
            cell.set_width(0.14)
        else:
            cell.set_width(0.19)

    # Group separator rules: one thin line after every 4 data rows (each ID block).
    for r in range(4, nrows, 4):
        for c in range(ncols):
            cell = table[r, c]
            cell.visible_edges = "LRB"
            cell.set_linewidth(0.6)
            cell.set_edgecolor("black")

    # Bottom rule
    last_row_idx = nrows
    for c in range(ncols):
        cell = table[last_row_idx, c]
        cell.visible_edges = "LRB"
        cell.set_linewidth(1.2)
        cell.set_edgecolor("black")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    experiment_meta = extract_experiment_meta(OUTPUTS_ROOT)
    all_group_data: dict[str, dict[int, dict[str, Any]]] = {}
    for group_name, group_dir in GROUP_DIRS.items():
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")
        all_group_data[group_name] = collect_group_records(group_name, group_dir)

    all_starter_ids = sorted(
        {
            sid
            for group_data in all_group_data.values()
            for sid in group_data.keys()
        }
    )

    headers = [
        "starter_id",
        "g1_entropy",
        "g1_encode_s_per_tok",
        "g1_decode_s_per_tok",
        "g2_entropy",
        "g2_encode_s_per_tok",
        "g2_decode_s_per_tok",
        "g3_entropy",
        "g3_encode_s_per_tok",
        "g3_decode_s_per_tok",
        "g4_entropy",
        "g4_encode_s_per_tok",
        "g4_decode_s_per_tok",
    ]

    rows: list[dict[str, Any]] = []
    for sid in all_starter_ids:
        row: dict[str, Any] = {"starter_id": sid}
        for idx, group_name in enumerate(GROUP_DIRS, start=1):
            record = all_group_data[group_name].get(sid, {})
            entropy_values = list(record.get("entropy", []))
            encode_values = list(record.get("encode", []))
            decode_values = list(record.get("decode", []))

            row[f"g{idx}_entropy"] = format_mean_pm_std(entropy_values)
            row[f"g{idx}_encode_s_per_tok"] = format_mean_pm_std_scaled(encode_values, scale_factor=1000.0)
            row[f"g{idx}_decode_s_per_tok"] = format_mean_pm_std_scaled(decode_values, scale_factor=1000.0)

            row[f"g{idx}_entropy_mean"] = avg_or_none(entropy_values)
            row[f"g{idx}_encode_mean"] = avg_or_none(encode_values)
            row[f"g{idx}_decode_mean"] = avg_or_none(decode_values)

        rows.append(row)

    png_headers = [
        "StarterId",
        "Group",
        "Entropy (bit/token)",
        "Encoding (s/10^3 token)",
        "Decoding (s/10^3 token)",
    ]
    png_rows: list[dict[str, Any]] = []
    highlight_cells: set[tuple[int, int]] = set()
    for row in rows:
        sid = row["starter_id"]
        starter_code = str(sid)
        entropy_vals = [
            row.get("g1_entropy_mean"),
            row.get("g2_entropy_mean"),
            row.get("g3_entropy_mean"),
            row.get("g4_entropy_mean"),
        ]
        encode_vals = [
            row.get("g1_encode_mean"),
            row.get("g2_encode_mean"),
            row.get("g3_encode_mean"),
            row.get("g4_encode_mean"),
        ]
        decode_vals = [
            row.get("g1_decode_mean"),
            row.get("g2_decode_mean"),
            row.get("g3_decode_mean"),
            row.get("g4_decode_mean"),
        ]

        base_row = len(png_rows) + 1  # table data rows start from 1 (0 is header)

        valid_entropy = [v for v in entropy_vals if v is not None]
        if valid_entropy:
            max_entropy = max(valid_entropy)
            for i, v in enumerate(entropy_vals):
                if v is not None and abs(v - max_entropy) < 1e-12:
                    highlight_cells.add((base_row + i, 2))

        valid_encode = [v for v in encode_vals if v is not None]
        if valid_encode:
            min_encode = min(valid_encode)
            for i, v in enumerate(encode_vals):
                if v is not None and abs(v - min_encode) < 1e-12:
                    highlight_cells.add((base_row + i, 3))

        valid_decode = [v for v in decode_vals if v is not None]
        if valid_decode:
            min_decode = min(valid_decode)
            for i, v in enumerate(decode_vals):
                if v is not None and abs(v - min_decode) < 1e-12:
                    highlight_cells.add((base_row + i, 4))

        png_rows.extend(
            [
                {
                    "StarterId": "",
                    "Group": "Group1",
                    "Entropy (bit/token)": row["g1_entropy"],
                    "Encoding (s/10^3 token)": row["g1_encode_s_per_tok"],
                    "Decoding (s/10^3 token)": row["g1_decode_s_per_tok"],
                },
                {
                    "StarterId": starter_code,
                    "Group": "Group2",
                    "Entropy (bit/token)": row["g2_entropy"],
                    "Encoding (s/10^3 token)": row["g2_encode_s_per_tok"],
                    "Decoding (s/10^3 token)": row["g2_decode_s_per_tok"],
                },
                {
                    "StarterId": "",
                    "Group": "Group3",
                    "Entropy (bit/token)": row["g3_entropy"],
                    "Encoding (s/10^3 token)": row["g3_encode_s_per_tok"],
                    "Decoding (s/10^3 token)": row["g3_decode_s_per_tok"],
                },
                {
                    "StarterId": "",
                    "Group": "Group4",
                    "Entropy (bit/token)": row["g4_entropy"],
                    "Encoding (s/10^3 token)": row["g4_encode_s_per_tok"],
                    "Decoding (s/10^3 token)": row["g4_decode_s_per_tok"],
                },
            ]
        )

    save_table_png(
        png_headers,
        png_rows,
        PDF_PATH,
        experiment_meta=experiment_meta,
        highlight_cells=highlight_cells,
    )
    print(f"Wrote table PDF to: {PDF_PATH}")


if __name__ == "__main__":
    main()
