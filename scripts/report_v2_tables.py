from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.tools import analysis_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-friendly V2 tables from aggregated summaries.")
    parser.add_argument("--suffix", default="", help="Optional suffix that matches analyze_v2_outputs artifacts.")
    return parser.parse_args()


def normalized_suffix(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("__"):
        return text
    return "__" + text.replace(" ", "_")


def load_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [item for item in payload if isinstance(item, dict)]


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    print(analysis_tools.render_markdown_table(headers, rows))


def main() -> None:
    args = parse_args()
    suffix = normalized_suffix(args.suffix)
    table_dir = PROJECT_ROOT / config.TABLE_V2_DIR

    controlled = load_summary(table_dir / f"controlled_summary{suffix}.json")
    controlled_summary = load_summary(table_dir / f"controlled_summary_summary{suffix}.json")
    realistic = load_summary(table_dir / f"realistic_summary{suffix}.json")

    combined = (controlled or []) + (controlled_summary or []) + (realistic or [])

    if controlled or controlled_summary:
        print_table(
            "Controlled Cognitive Asymmetry",
            [
                "Protocol",
                "BER @ Ideal",
                "Context Truncation BER",
                "Context Truncation DSR",
                "Context Summary BER",
                "Context Summary DSR",
            ],
            analysis_tools.build_controlled_cognitive_asymmetry_table_rows(combined),
        )

    if realistic:
        print_table(
            "Realistic Integrated Table",
            ["Protocol", "Acc", "Score", "F1", "BER", "DSR", "Nom.(bits/1kTok)", "Eff.(bits/1kTok)"],
            analysis_tools.build_realistic_integrated_table_rows(combined),
        )

    if not controlled and not realistic:
        print(f"No aggregated V2 summaries found in {table_dir}")


if __name__ == "__main__":
    main()
