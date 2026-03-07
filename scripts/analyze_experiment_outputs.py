from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

OUTPUTS_ROOT = PROJECT_ROOT / config.OUTPUT_DIR
SCORE_DATA_DIR = PROJECT_ROOT / config.SCORE_DATA_DIR
TABLE_DIR = PROJECT_ROOT / config.TABLE_DIR
SCORE_GROUPS = ("group1", "group2", "group3", "group4", "group5")
GROUP_DIRS = {group: OUTPUTS_ROOT / group for group in SCORE_GROUPS}
FILE_PATTERN = re.compile(r"-(\d+)-(\d+)\.json$")
CANONICAL_MEMORY_SCORE_KEY = "memory_pivot_score"
CANONICAL_BEHAVIORAL_SCORE_KEY = "behavioral_elasticity_score"
CANONICAL_GLOBAL_SCORE_KEY = "global_cognitive_coherence_score"

MEMORY_SCORE_KEYS = (
    CANONICAL_MEMORY_SCORE_KEY,
    "pivot_tracking_score",
    "memory_continuity_score",
    "memory_context_score",
)
BEHAVIORAL_SCORE_KEYS = (
    CANONICAL_BEHAVIORAL_SCORE_KEY,
    "behavioral_consistency_score",
    "behavioral_flexibility_score",
    "behavioral_adaptability_score",
    "behavioral_formatting_score",
)
GLOBAL_SCORE_KEYS = (
    CANONICAL_GLOBAL_SCORE_KEY,
    "global_coherence_score",
    "overall_cognitive_coherence_score",
    "cognitive_coherence_score",
)


# ===== score =====
def parse_ids_from_filename(path: Path) -> tuple[int, int] | None:
    m = FILE_PATTERN.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_transcript(rounds: list[dict[str, Any]], *, starter: str = "") -> str:
    """
    Build a 20-turn transcript aligned with the actual generation order:
    starter -> assistant_1, user_1 -> assistant_2, ..., user_19 -> assistant_20.

    Note: round_20.user is the next prompt for a potential assistant_21, so it is
    intentionally not paired in this 20-turn transcript.
    """
    lines: list[str] = []
    for idx, r in enumerate(rounds, start=1):
        if idx == 1:
            user = str(starter or "").strip()
            if not user:
                # Fallback for legacy records missing starter.
                user = str(r.get("user", "")).strip()
        else:
            prev_round = rounds[idx - 2]
            user = str(prev_round.get("user", "")).strip()
            if not user:
                # Fallback for malformed records with missing previous user turn.
                user = str(r.get("user", "")).strip()
        assistant = str(r.get("assistant", "")).strip()
        lines.append(f"Turn {idx}")
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output")


def _canonicalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _resolve_score_key(item: dict[str, Any], keys: tuple[str, ...], *, kind: str) -> str | None:
    normalized_to_original: dict[str, str] = {}
    for original in item:
        normalized_to_original.setdefault(_canonicalize_key(original), original)

    for key in keys:
        if key in item:
            return key
        normalized = _canonicalize_key(key)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    for original in item:
        normalized = _canonicalize_key(original)
        if not normalized.endswith("_score"):
            continue
        if kind == "memory" and ("memory" in normalized or "pivot" in normalized):
            return original
        if kind == "behavioral" and (
            "behavior" in normalized
            or "elasticity" in normalized
            or "consistency" in normalized
            or "format" in normalized
        ):
            return original
        if kind == "global" and ("coherence" in normalized or "cognitive" in normalized):
            return original
    return None


def normalize_score_payload(payload: dict[str, Any]) -> None:
    turns = payload.get("turn_analysis")
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            memory_key = _resolve_score_key(turn, MEMORY_SCORE_KEYS, kind="memory")
            if memory_key is not None and CANONICAL_MEMORY_SCORE_KEY not in turn:
                turn[CANONICAL_MEMORY_SCORE_KEY] = turn[memory_key]
            behavioral_key = _resolve_score_key(turn, BEHAVIORAL_SCORE_KEYS, kind="behavioral")
            if behavioral_key is not None and CANONICAL_BEHAVIORAL_SCORE_KEY not in turn:
                turn[CANONICAL_BEHAVIORAL_SCORE_KEY] = turn[behavioral_key]

    global_assessment = payload.get("global_assessment")
    if isinstance(global_assessment, dict):
        global_key = _resolve_score_key(global_assessment, GLOBAL_SCORE_KEYS, kind="global")
        if global_key is not None and CANONICAL_GLOBAL_SCORE_KEY not in global_assessment:
            global_assessment[CANONICAL_GLOBAL_SCORE_KEY] = global_assessment[global_key]


def validate_score_payload(payload: dict[str, Any]) -> None:
    turns = payload.get("turn_analysis")
    global_assessment = payload.get("global_assessment")
    if not isinstance(turns, list):
        raise ValueError("turn_analysis missing or not a list")
    if len(turns) != 20:
        raise ValueError(f"turn_analysis must have 20 turns, got {len(turns)}")
    if not isinstance(global_assessment, dict):
        raise ValueError("global_assessment missing or not a dict")

    for idx, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"turn_analysis[{idx}] must be an object")
        _extract_turn_score(turn, MEMORY_SCORE_KEYS, idx, kind="memory")
        _extract_turn_score(turn, BEHAVIORAL_SCORE_KEYS, idx, kind="behavioral")
    _extract_global_score(global_assessment)


def _extract_turn_score(turn: dict[str, Any], keys: tuple[str, ...], turn_idx: int, *, kind: str) -> float:
    key = _resolve_score_key(turn, keys, kind=kind)
    if key is None:
        available = ", ".join(sorted(turn.keys()))
        raise ValueError(f"turn_analysis[{turn_idx}] missing {kind} score key; available keys: [{available}]")
    try:
        return float(turn[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"turn_analysis[{turn_idx}] invalid value for '{key}': {turn[key]!r}") from exc


def _extract_global_score(global_assessment: dict[str, Any]) -> float:
    key = _resolve_score_key(global_assessment, GLOBAL_SCORE_KEYS, kind="global")
    if key is None:
        available = ", ".join(sorted(global_assessment.keys()))
        raise ValueError(f"global_assessment missing coherence score key; available keys: [{available}]")
    try:
        return float(global_assessment[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"global_assessment invalid value for '{key}': {global_assessment[key]!r}"
        ) from exc


def summarize_scores(payload: dict[str, Any]) -> dict[str, float]:
    turns = payload["turn_analysis"]
    memory = [_extract_turn_score(t, MEMORY_SCORE_KEYS, idx, kind="memory") for idx, t in enumerate(turns, start=1)]
    elasticity = [
        _extract_turn_score(t, BEHAVIORAL_SCORE_KEYS, idx, kind="behavioral")
        for idx, t in enumerate(turns, start=1)
    ]
    global_score = _extract_global_score(payload["global_assessment"])
    return {
        "memory_context_mean": mean(memory),
        "behavioral_elasticity_mean": mean(elasticity),
        "global_cognitive_coherence": global_score,
    }


def run_eval(client: OpenAI, model: str, transcript: str) -> tuple[str, dict[str, Any]]:
    user_prompt = (
        "Please evaluate this 20-turn conversation transcript strictly by the specified JSON schema.\n\n"
        "Conversation Transcript:\n"
        f"{transcript}\n"
    )

    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": config.SCORE_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            payload = extract_json_object(content)
            normalize_score_payload(payload)
            validate_score_payload(payload)
            return content, payload
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError, ValueError) as e:
            last_error = e
            if attempt == 4:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"Scoring failed after retries: {last_error}") from last_error


def save_incremental_score(
    *,
    output_path: Path,
    source_file: Path,
    group: str,
    starter_id: int,
    trial_id: int,
    score_model: str,
    payload: dict[str, Any],
    raw_text: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source_file": str(source_file),
        "group": group,
        "starter_id": starter_id,
        "trial_id": trial_id,
        "score_model": score_model,
        "summary": summarize_scores(payload),
        "scores": payload,
        "raw_output_text": raw_text,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def command_score(*, only_group: str | None, limit: int, force: bool) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = config.OPENAI_API_KEY.strip()
    base_url = config.OPENAI_BASE_URL.strip()
    score_model = config.SCORE_AGENT.strip()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    if not score_model:
        raise RuntimeError("SCORE_AGENT is missing in config.py")

    SCORE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 120.0,
        "max_retries": 2,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    scored_count = 0
    failed_count = 0
    for group, group_dir in GROUP_DIRS.items():
        if only_group and group != only_group:
            continue
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")

        for path in sorted(group_dir.glob("*.json")):
            if limit > 0 and scored_count >= limit:
                break

            ids = parse_ids_from_filename(path)
            if ids is None:
                continue
            starter_id, trial_id = ids

            out_path = SCORE_DATA_DIR / group / path.name
            if out_path.exists() and not force:
                print(f"Skip existing: {out_path}")
                continue

            data = json.loads(path.read_text(encoding="utf-8"))
            rounds = list(data.get("rounds", []))
            if len(rounds) != 20:
                continue

            transcript = build_transcript(rounds, starter=str(data.get("starter", "")))
            try:
                raw_text, payload = run_eval(client, score_model, transcript)
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(f"[WARN] failed to score {path.name}: {exc}")
                continue

            save_incremental_score(
                output_path=out_path,
                source_file=path,
                group=group,
                starter_id=starter_id,
                trial_id=trial_id,
                score_model=score_model,
                payload=payload,
                raw_text=raw_text,
            )
            scored_count += 1
            print(f"Saved score: {out_path}")

        if limit > 0 and scored_count >= limit:
            break

    print(f"Score json dir: {SCORE_DATA_DIR}")
    print(f"Scored: {scored_count}, failed: {failed_count}")


# ===== coherence report =====
def extract_score(record: dict) -> float | None:
    score = record.get("scores", {}).get("global_assessment", {}).get("global_cognitive_coherence_score")
    return None if score is None else float(score)


def extract_ids(record: dict) -> tuple[int | None, int | None]:
    starter_id = record.get("starter_id")
    trial_id = record.get("trial_id")
    if starter_id is not None and trial_id is not None:
        return int(starter_id), int(trial_id)
    return None, None


def load_group_scores(group: str) -> tuple[dict[int, list[float]], list[str]]:
    group_dir = SCORE_DATA_DIR / group
    if not group_dir.exists():
        return {}, [f"[{group}] missing directory: {group_dir}"]

    by_starter: dict[int, list[float]] = defaultdict(list)
    warnings: list[str] = []

    for json_file in sorted(group_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"[{group}] failed to parse {json_file.name}: {exc}")
            continue

        starter_id, trial_id = extract_ids(data)
        if starter_id is None or trial_id is None:
            warnings.append(f"[{group}] missing starter/trial id: {json_file.name}")
            continue

        score = extract_score(data)
        if score is None:
            warnings.append(f"[{group}] missing coherence score: {json_file.name}")
            continue

        by_starter[starter_id].append(score)

    return by_starter, warnings


def write_coherence_pdf(rows: list[dict[str, object]], output_path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows_per_page = 34
    total_rows = len(rows)
    total_pages = max((total_rows + rows_per_page - 1) // rows_per_page, 1)

    col_labels = ["Start", *[f"Group{idx}" for idx in range(1, len(SCORE_GROUPS) + 1)]]
    group_col_width = 0.10 if len(SCORE_GROUPS) >= 5 else 0.12
    col_widths = [0.10, *([group_col_width] * len(SCORE_GROUPS))]

    with PdfPages(output_path) as pdf:
        for page_idx in range(total_pages):
            page_start = page_idx * rows_per_page
            page_end = min((page_idx + 1) * rows_per_page, total_rows)
            chunk = rows[page_start:page_end]

            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")

            cell_text: list[list[str]] = []
            for row in chunk:
                line = [str(row["starter"])]
                for group in SCORE_GROUPS:
                    score = row[group]
                    if isinstance(score, str):
                        line.append(score)
                    elif isinstance(score, (float, int)):
                        line.append(format_significant(float(score), 2))
                    else:
                        line.append("N/A")
                cell_text.append(line)

            if not cell_text:
                cell_text = [["-"] * len(col_labels)]

            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                colWidths=col_widths,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.45)

            for (r, _c), cell in table.get_celld().items():
                cell.set_edgecolor("#666666")
                if r == 0:
                    cell.set_facecolor("#F2F2F2")
                    cell.set_text_props(weight="bold")

            # 每个 Starter 行中，按各 Group 的“均值”比较，最高者加粗。
            # 若并列最高，则并列项全部加粗。
            for row_idx, row in enumerate(chunk, start=1):
                group_means = [
                    _safe_float(row.get(f"__{group}_mean"))
                    for group in SCORE_GROUPS
                ]
                valid_means = [m for m in group_means if m is not None]
                if not valid_means:
                    continue
                max_mean = max(valid_means)
                for col_offset, group_mean in enumerate(group_means, start=1):
                    if group_mean is None:
                        continue
                    if abs(group_mean - max_mean) < 1e-12:
                        table[row_idx, col_offset].set_text_props(weight="bold")

            ax.set_title(
                f"Coherence Score by Start (mean±std, Page {page_idx + 1}/{total_pages})",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def command_coherence() -> None:
    print("Coherence view has been merged into the unified metrics table.")
    command_metrics()


# ===== metrics report =====
def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def extract_experiment_meta() -> dict[str, str]:
    sample_meta: dict[str, Any] = {}
    for group in SCORE_GROUPS:
        group_dir = OUTPUTS_ROOT / group
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
        encode = _safe_float(metrics.get("average_encode_time_per_token_seconds"))
        if encode is None and group_name == "group1":
            encode = _safe_float(metrics.get("average_generate_time_per_token_seconds"))
        decode = _safe_float(metrics.get("average_decode_time_per_token_seconds"))

        if starter_id not in by_starter:
            by_starter[starter_id] = {"starter": starter_text, "entropy": [], "encode": [], "decode": []}
        elif not by_starter[starter_id].get("starter") and starter_text:
            by_starter[starter_id]["starter"] = starter_text

        if entropy is not None:
            by_starter[starter_id]["entropy"].append(entropy)
        if encode is not None:
            by_starter[starter_id]["encode"].append(encode)
        if decode is not None:
            by_starter[starter_id]["decode"].append(decode)

    return by_starter


def extract_trial_score_means(score_record: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    summary = score_record.get("summary", {})
    memory_mean = _safe_float(summary.get("memory_context_mean"))
    behavioral_mean = _safe_float(summary.get("behavioral_elasticity_mean"))
    coherence_mean = _safe_float(summary.get("global_cognitive_coherence"))
    if memory_mean is not None and behavioral_mean is not None and coherence_mean is not None:
        return memory_mean, behavioral_mean, coherence_mean

    turns = score_record.get("scores", {}).get("turn_analysis", [])
    if coherence_mean is None:
        global_assessment = score_record.get("scores", {}).get("global_assessment")
        if isinstance(global_assessment, dict):
            try:
                coherence_mean = _extract_global_score(global_assessment)
            except ValueError:
                coherence_mean = None
    if not isinstance(turns, list):
        return memory_mean, behavioral_mean, coherence_mean

    if memory_mean is None:
        memory_values: list[float] = []
        for idx, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            try:
                memory_values.append(_extract_turn_score(turn, MEMORY_SCORE_KEYS, idx, kind="memory"))
            except ValueError:
                continue
        memory_mean = avg_or_none(memory_values)

    if behavioral_mean is None:
        behavioral_values: list[float] = []
        for idx, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            try:
                behavioral_values.append(_extract_turn_score(turn, BEHAVIORAL_SCORE_KEYS, idx, kind="behavioral"))
            except ValueError:
                continue
        behavioral_mean = avg_or_none(behavioral_values)

    return memory_mean, behavioral_mean, coherence_mean


def collect_group_score_records(group_name: str) -> dict[int, dict[str, Any]]:
    group_score_dir = SCORE_DATA_DIR / group_name
    by_starter: dict[int, dict[str, Any]] = {}
    if not group_score_dir.exists():
        return by_starter

    for path in sorted(group_score_dir.glob("*.json")):
        try:
            score_record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        starter_id = score_record.get("starter_id")
        if starter_id is None:
            parsed = parse_ids_from_filename(path)
            if parsed is None:
                continue
            starter_id, _trial_id = parsed
        starter_id = int(starter_id)

        memory_mean, behavioral_mean, coherence_mean = extract_trial_score_means(score_record)
        if starter_id not in by_starter:
            by_starter[starter_id] = {"memory": [], "behavioral": [], "coherence": []}
        if memory_mean is not None:
            by_starter[starter_id]["memory"].append(memory_mean)
        if behavioral_mean is not None:
            by_starter[starter_id]["behavioral"].append(behavioral_mean)
        if coherence_mean is not None:
            by_starter[starter_id]["coherence"].append(coherence_mean)

    return by_starter


def avg_or_none(values: list[float]) -> float | None:
    return None if not values else mean(values)


def std_or_zero(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else float(stdev(values))


def format_significant(value: float, significant_digits: int = 2) -> str:
    if not math.isfinite(value):
        return "-"
    if significant_digits < 1:
        raise ValueError("significant_digits must be >= 1")
    if value == 0:
        return "0" if significant_digits == 1 else f"0.{('0' * (significant_digits - 1))}"

    order = math.floor(math.log10(abs(value)))
    decimals = significant_digits - order - 1
    rounded = round(value, decimals)

    # 处理四舍五入后数量级变化（例如 9.96 -> 10）
    if rounded != 0:
        rounded_order = math.floor(math.log10(abs(rounded)))
        if rounded_order != order:
            decimals = significant_digits - rounded_order - 1
            rounded = round(value, decimals)

    if decimals > 0:
        return f"{rounded:.{decimals}f}"
    return f"{rounded:.0f}"


def format_mean_pm_std(values: list[float], significant_digits: int = 2) -> str:
    if not values:
        return "-"
    m = avg_or_none(values)
    s = std_or_zero(values)
    if m is None:
        return "-"
    return f"{format_significant(m, significant_digits)}±{format_significant(s, significant_digits)}"


def format_mean_pm_std_scaled(values: list[float], scale_factor: float, significant_digits: int = 2) -> str:
    if not values:
        return "-"
    return format_mean_pm_std([v * scale_factor for v in values], significant_digits=significant_digits)


def save_table_pdf(
    headers: list[str],
    rows: list[dict[str, Any]],
    output_path: Path,
    experiment_meta: dict[str, str],
    highlight_cells: set[tuple[int, int]] | None = None,
    *,
    title: str = "Table 1. Group-wise comparison by Starter (mean±std over 3 trials)",
    subgroup_size: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cn_font_path = PROJECT_ROOT / "assets" / "fonts" / "SourceHanSansSC-Regular.otf"
    cn_bold_font_path = PROJECT_ROOT / "assets" / "fonts" / "SourceHanSansSC-Bold.otf"

    bold_font_prop = None
    if cn_font_path.exists():
        font_manager.fontManager.addfont(str(cn_font_path))
        cn_name = font_manager.FontProperties(fname=str(cn_font_path)).get_name()
        plt.rcParams["font.family"] = cn_name
        if cn_bold_font_path.exists():
            font_manager.fontManager.addfont(str(cn_bold_font_path))
            bold_font_prop = font_manager.FontProperties(fname=str(cn_bold_font_path))
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "SimHei",
            "Microsoft YaHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False

    cell_text = [[str(row.get(h, "")) for h in headers] for row in rows]
    nrows = max(len(rows), 1)
    ncols = len(headers)
    highlight_cells = highlight_cells or set()

    fig_w = max(12.0, ncols * 2.0)
    fig_h = max(8.0, nrows * 0.27 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=240)
    ax.axis("off")
    ax.set_title(
        title,
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
        "组1：正常对话 ｜ 组2：对称隐写(DISCOP) ｜ 组3：对称隐写(METEOR) ｜ "
        "组4：非对称隐写+上下文一致 ｜ 组5：非对称隐写+上下文不一致+RAG"
    )
    ax.text(0.5, 0.988, meta_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3)
    ax.text(0.5, 0.962, group_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3)

    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.93])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_text_props(ha="center", va="center")
        if r == 0:
            cell.set_text_props(weight="bold", ha="center", va="center")
            if bold_font_prop is not None:
                cell.get_text().set_fontproperties(bold_font_prop)
            cell.get_text().set_fontsize(8)
            cell.set_facecolor("white")
            cell.visible_edges = "TLRB"
            cell.set_linewidth(1.2)
        else:
            # Add horizontal separators for every data row to improve readability.
            cell.visible_edges = "LRB"
            cell.set_linewidth(0.5)
            # For grouped tables, keep the first column visually merged within each subgroup.
            if c == 0 and subgroup_size is not None and subgroup_size > 0:
                if (r % subgroup_size) != 0 and r != nrows:
                    cell.visible_edges = "LR"

        if (r, c) in highlight_cells:
            cell.set_text_props(weight="bold", ha="center", va="center")
            if bold_font_prop is not None:
                cell.get_text().set_fontproperties(bold_font_prop)
            cell.get_text().set_fontsize(8)
        cell.get_text().set_wrap(True)
        if ncols <= 5:
            if c == 0:
                cell.set_width(0.11)
            elif c == 1:
                cell.set_width(0.14)
            else:
                cell.set_width(0.19)
        else:
            if c == 0:
                cell.set_width(0.06)
            elif c == 1:
                cell.set_width(0.08)
            else:
                cell.set_width(0.11)

    if subgroup_size is not None and subgroup_size > 0:
        for r in range(subgroup_size, nrows, subgroup_size):
            for c in range(ncols):
                cell = table[r, c]
                cell.visible_edges = "LRB"
                cell.set_linewidth(0.6)
                cell.set_edgecolor("black")

    last_row_idx = nrows
    for c in range(ncols):
        cell = table[last_row_idx, c]
        cell.visible_edges = "LRB"
        cell.set_linewidth(1.2)
        cell.set_edgecolor("black")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def command_metrics() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    experiment_meta = extract_experiment_meta()
    significant_digits = 2
    encode_decode_scale_factor = 1000.0  # Display in 10^-3 s/token.

    all_group_data: dict[str, dict[int, dict[str, Any]]] = {}
    all_group_score_data: dict[str, dict[int, dict[str, Any]]] = {}
    for group_name, group_dir in GROUP_DIRS.items():
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")
        all_group_data[group_name] = collect_group_records(group_name, group_dir)
        all_group_score_data[group_name] = collect_group_score_records(group_name)

    all_starter_ids = sorted(
        {sid for group_data in all_group_data.values() for sid in group_data.keys()}
        | {sid for group_score_data in all_group_score_data.values() for sid in group_score_data.keys()}
    )

    rows: list[dict[str, Any]] = []
    for sid in all_starter_ids:
        row: dict[str, Any] = {"starter_id": sid}
        for idx, group_name in enumerate(GROUP_DIRS, start=1):
            record = all_group_data[group_name].get(sid, {})
            score_record = all_group_score_data[group_name].get(sid, {})
            entropy_values = list(record.get("entropy", []))
            encode_values = list(record.get("encode", []))
            decode_values = list(record.get("decode", []))
            memory_values = list(score_record.get("memory", []))
            behavioral_values = list(score_record.get("behavioral", []))
            coherence_values = list(score_record.get("coherence", []))

            row[f"g{idx}_entropy"] = format_mean_pm_std(entropy_values, significant_digits=significant_digits)
            row[f"g{idx}_encode_s_per_tok"] = format_mean_pm_std_scaled(
                encode_values,
                scale_factor=encode_decode_scale_factor,
                significant_digits=significant_digits,
            )
            row[f"g{idx}_decode_s_per_tok"] = format_mean_pm_std_scaled(
                decode_values,
                scale_factor=encode_decode_scale_factor,
                significant_digits=significant_digits,
            )
            row[f"g{idx}_memory_score"] = format_mean_pm_std(memory_values, significant_digits=significant_digits)
            row[f"g{idx}_behavioral_score"] = format_mean_pm_std(behavioral_values, significant_digits=significant_digits)
            row[f"g{idx}_coherence"] = format_mean_pm_std(coherence_values, significant_digits=significant_digits)
            row[f"g{idx}_entropy_mean"] = avg_or_none(entropy_values)
            row[f"g{idx}_encode_mean"] = avg_or_none(encode_values)
            row[f"g{idx}_decode_mean"] = avg_or_none(decode_values)
            row[f"g{idx}_memory_mean"] = avg_or_none(memory_values)
            row[f"g{idx}_behavioral_mean"] = avg_or_none(behavioral_values)
            row[f"g{idx}_coherence_mean"] = avg_or_none(coherence_values)
        rows.append(row)

    headers = [
        "StarterId",
        "Group",
        "Entropy (bit/token)",
        "Encoding (10^-3 s/token)",
        "Decoding (10^-3 s/token)",
        "Memory Pivot Score",
        "Behavioral Elasticity Score",
        "Global Coherence",
    ]

    pdf_rows: list[dict[str, Any]] = []
    highlight_cells: set[tuple[int, int]] = set()

    for row in rows:
        sid = row["starter_id"]
        group_indices = list(range(1, len(GROUP_DIRS) + 1))
        entropy_vals = [row.get(f"g{idx}_entropy_mean") for idx in group_indices]
        encode_vals = [row.get(f"g{idx}_encode_mean") for idx in group_indices]
        decode_vals = [row.get(f"g{idx}_decode_mean") for idx in group_indices]
        memory_vals = [row.get(f"g{idx}_memory_mean") for idx in group_indices]
        behavioral_vals = [row.get(f"g{idx}_behavioral_mean") for idx in group_indices]
        coherence_vals = [row.get(f"g{idx}_coherence_mean") for idx in group_indices]

        base_row = len(pdf_rows) + 1

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

        valid_memory = [v for v in memory_vals if v is not None]
        if valid_memory:
            max_memory = max(valid_memory)
            for i, v in enumerate(memory_vals):
                if v is not None and abs(v - max_memory) < 1e-12:
                    highlight_cells.add((base_row + i, 5))

        valid_behavioral = [v for v in behavioral_vals if v is not None]
        if valid_behavioral:
            max_behavioral = max(valid_behavioral)
            for i, v in enumerate(behavioral_vals):
                if v is not None and abs(v - max_behavioral) < 1e-12:
                    highlight_cells.add((base_row + i, 6))

        valid_coherence = [v for v in coherence_vals if v is not None]
        if valid_coherence:
            max_coherence = max(valid_coherence)
            for i, v in enumerate(coherence_vals):
                if v is not None and abs(v - max_coherence) < 1e-12:
                    highlight_cells.add((base_row + i, 7))

        for idx in group_indices:
            pdf_rows.append(
                {
                    "StarterId": str(sid) if idx == 1 else "",
                    "Group": f"Group{idx}",
                    "Entropy (bit/token)": row[f"g{idx}_entropy"],
                    "Encoding (10^-3 s/token)": row[f"g{idx}_encode_s_per_tok"],
                    "Decoding (10^-3 s/token)": row[f"g{idx}_decode_s_per_tok"],
                    "Memory Pivot Score": row[f"g{idx}_memory_score"],
                    "Behavioral Elasticity Score": row[f"g{idx}_behavioral_score"],
                    "Global Coherence": row[f"g{idx}_coherence"],
                }
            )

    output = TABLE_DIR / "starter_group_metrics_comparison.pdf"
    save_table_pdf(
        headers,
        pdf_rows,
        output,
        experiment_meta=experiment_meta,
        highlight_cells=highlight_cells,
        title="Table 1. Group-wise comparison by Starter (mean±std over 3 trials)",
        subgroup_size=len(GROUP_DIRS),
    )
    print(f"Wrote table PDF to: {output}")

    # Table 2: aggregate Table 1 over all starters, keeping the same mean±std representation.
    starter_count = len(all_starter_ids)
    group_summary_rows: list[dict[str, Any]] = []
    for idx in range(1, len(GROUP_DIRS) + 1):
        entropy_values = [v for v in (row.get(f"g{idx}_entropy_mean") for row in rows) if v is not None]
        encode_values = [v for v in (row.get(f"g{idx}_encode_mean") for row in rows) if v is not None]
        decode_values = [v for v in (row.get(f"g{idx}_decode_mean") for row in rows) if v is not None]
        memory_values = [v for v in (row.get(f"g{idx}_memory_mean") for row in rows) if v is not None]
        behavioral_values = [v for v in (row.get(f"g{idx}_behavioral_mean") for row in rows) if v is not None]
        coherence_values = [v for v in (row.get(f"g{idx}_coherence_mean") for row in rows) if v is not None]

        group_summary_rows.append(
            {
                "group": f"Group{idx}",
                "entropy_values": entropy_values,
                "encode_values": encode_values,
                "decode_values": decode_values,
                "memory_values": memory_values,
                "behavioral_values": behavioral_values,
                "coherence_values": coherence_values,
                "entropy_mean": avg_or_none(entropy_values),
                "encode_mean": avg_or_none(encode_values),
                "decode_mean": avg_or_none(decode_values),
                "memory_mean": avg_or_none(memory_values),
                "behavioral_mean": avg_or_none(behavioral_values),
                "coherence_mean": avg_or_none(coherence_values),
            }
        )

    summary_highlight_cells: set[tuple[int, int]] = set()
    summary_entropy_means = [g["entropy_mean"] for g in group_summary_rows]
    summary_encode_means = [g["encode_mean"] for g in group_summary_rows]
    summary_decode_means = [g["decode_mean"] for g in group_summary_rows]
    summary_memory_means = [g["memory_mean"] for g in group_summary_rows]
    summary_behavioral_means = [g["behavioral_mean"] for g in group_summary_rows]
    summary_coherence_means = [g["coherence_mean"] for g in group_summary_rows]

    valid_summary_entropy = [v for v in summary_entropy_means if v is not None]
    if valid_summary_entropy:
        max_entropy = max(valid_summary_entropy)
        for i, v in enumerate(summary_entropy_means, start=1):
            if v is not None and abs(v - max_entropy) < 1e-12:
                summary_highlight_cells.add((i, 2))

    valid_summary_encode = [v for v in summary_encode_means if v is not None]
    if valid_summary_encode:
        min_encode = min(valid_summary_encode)
        for i, v in enumerate(summary_encode_means, start=1):
            if v is not None and abs(v - min_encode) < 1e-12:
                summary_highlight_cells.add((i, 3))

    valid_summary_decode = [v for v in summary_decode_means if v is not None]
    if valid_summary_decode:
        min_decode = min(valid_summary_decode)
        for i, v in enumerate(summary_decode_means, start=1):
            if v is not None and abs(v - min_decode) < 1e-12:
                summary_highlight_cells.add((i, 4))

    valid_summary_memory = [v for v in summary_memory_means if v is not None]
    if valid_summary_memory:
        max_memory = max(valid_summary_memory)
        for i, v in enumerate(summary_memory_means, start=1):
            if v is not None and abs(v - max_memory) < 1e-12:
                summary_highlight_cells.add((i, 5))

    valid_summary_behavioral = [v for v in summary_behavioral_means if v is not None]
    if valid_summary_behavioral:
        max_behavioral = max(valid_summary_behavioral)
        for i, v in enumerate(summary_behavioral_means, start=1):
            if v is not None and abs(v - max_behavioral) < 1e-12:
                summary_highlight_cells.add((i, 6))

    valid_summary_coherence = [v for v in summary_coherence_means if v is not None]
    if valid_summary_coherence:
        max_coherence = max(valid_summary_coherence)
        for i, v in enumerate(summary_coherence_means, start=1):
            if v is not None and abs(v - max_coherence) < 1e-12:
                summary_highlight_cells.add((i, 7))

    summary_pdf_rows: list[dict[str, Any]] = []
    scope_label = f"All {starter_count} Starters"
    for idx, group_summary in enumerate(group_summary_rows):
        summary_pdf_rows.append(
            {
                "StarterId": scope_label if idx == 0 else "",
                "Group": group_summary["group"],
                "Entropy (bit/token)": format_mean_pm_std(
                    list(group_summary["entropy_values"]),
                    significant_digits=significant_digits,
                ),
                "Encoding (10^-3 s/token)": format_mean_pm_std_scaled(
                    list(group_summary["encode_values"]),
                    scale_factor=encode_decode_scale_factor,
                    significant_digits=significant_digits,
                ),
                "Decoding (10^-3 s/token)": format_mean_pm_std_scaled(
                    list(group_summary["decode_values"]),
                    scale_factor=encode_decode_scale_factor,
                    significant_digits=significant_digits,
                ),
                "Memory Pivot Score": format_mean_pm_std(
                    list(group_summary["memory_values"]),
                    significant_digits=significant_digits,
                ),
                "Behavioral Elasticity Score": format_mean_pm_std(
                    list(group_summary["behavioral_values"]),
                    significant_digits=significant_digits,
                ),
                "Global Coherence": format_mean_pm_std(
                    list(group_summary["coherence_values"]),
                    significant_digits=significant_digits,
                ),
            }
        )

    summary_output = TABLE_DIR / "starter_group_metrics_overall_summary.pdf"
    save_table_pdf(
        headers,
        summary_pdf_rows,
        summary_output,
        experiment_meta=experiment_meta,
        highlight_cells=summary_highlight_cells,
        title=f"Table 2. Group-wise aggregate over {starter_count} Starters (mean±std)",
        subgroup_size=None,
    )
    print(f"Wrote table PDF to: {summary_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze experiment outputs: score + reports.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Run LLM scoring on dialogue outputs.")
    score.add_argument("--only-group", choices=list(SCORE_GROUPS), default=None)
    score.add_argument("--limit", type=int, default=0)
    score.add_argument("--force", action="store_true")

    sub.add_parser("coherence", help="Alias: generate the unified metrics table (coherence merged).")
    sub.add_parser("metrics", help="Generate unified metrics comparison table PDF.")

    all_cmd = sub.add_parser("all", help="Run score + unified metrics table in order.")
    all_cmd.add_argument("--only-group", choices=list(SCORE_GROUPS), default=None)
    all_cmd.add_argument("--limit", type=int, default=0)
    all_cmd.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "score":
        command_score(only_group=args.only_group, limit=args.limit, force=bool(args.force))
        return

    if args.command == "coherence":
        command_coherence()
        return

    if args.command == "metrics":
        command_metrics()
        return

    if args.command == "all":
        command_score(only_group=args.only_group, limit=args.limit, force=bool(args.force))
        command_metrics()
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
