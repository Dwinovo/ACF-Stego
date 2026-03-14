from __future__ import annotations

import argparse
import sys
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
    "realistic": {"G1", "G2", "G3", "G4", "G5", "G6", "G7"},
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
    parser.add_argument("--only-group", choices=[f"group{i}" for i in range(1, 8)], default=None)
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
        f"plot_task_vs_reliability{name_suffix}.json",
        f"plot_ber_vs_condition{name_suffix}.json",
        f"plot_controlled_summary_asymmetry{name_suffix}.json",
        f"plot_controlled_drift_severity_sweep{name_suffix}.json",
        f"plot_task_correctness_vs_reliability{name_suffix}.json",
    ]
    for name in obsolete_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


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
        "BER",
        "DSR",
        "Nom.(bits/1kTok)",
        "Eff.(bits/1kTok)",
    ]
    realistic_rows = analysis_tools.build_realistic_integrated_table_rows(combined_summaries)
    analysis_tools.write_csv(
        output_dir / f"paper_table_realistic_integrated{name_suffix}.csv",
        realistic_headers,
        realistic_rows,
    )

    print(f"Wrote final CSV tables to: {output_dir}")


if __name__ == "__main__":
    main()
