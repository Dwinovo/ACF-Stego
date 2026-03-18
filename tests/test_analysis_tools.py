from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.tools import analysis_tools


class AnalysisToolsTest(unittest.TestCase):
    def test_build_instance_means_aggregates_repeats(self) -> None:
        records = [
            {
                "run_id": "realistic.longmemeval_s.q1.G2.no_drift.42",
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q1",
                "task_f1": 0.4,
                "ber": 0.0,
            },
            {
                "run_id": "realistic.longmemeval_s.q1.G2.no_drift.43",
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q1",
                "task_f1": 0.6,
                "ber": 0.2,
            },
        ]

        instance_rows = analysis_tools.build_instance_means(records)

        self.assertEqual(len(instance_rows), 1)
        self.assertAlmostEqual(instance_rows[0]["task_f1"], 0.5)
        self.assertAlmostEqual(instance_rows[0]["ber"], 0.1)

    def test_build_instance_means_splits_acf_k_variants(self) -> None:
        records = [
            {
                "run_id": "r1",
                "experiment": "realistic_cognitive_asymmetry",
                "group": "G4",
                "condition": "no_drift",
                "question_id": "q1",
                "acf_k": 8,
                "ber": 0.1,
            },
            {
                "run_id": "r2",
                "experiment": "realistic_cognitive_asymmetry",
                "group": "G4",
                "condition": "no_drift",
                "question_id": "q1",
                "acf_k": 12,
                "ber": 0.2,
            },
        ]

        instance_rows = analysis_tools.build_instance_means(records)
        self.assertEqual(len(instance_rows), 2)
        self.assertEqual(sorted(row.get("acf_k") for row in instance_rows), [8, 12])

    def test_realistic_task_table_rows_include_llm_metrics(self) -> None:
        instance_rows = [
            {
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G1",
                "condition": "no_drift",
                "question_id": "q1",
                "task_em": 1.0,
                "task_f1": 0.9,
                "llm_judge_score": 2.0,
                "llm_judge_correct": 1.0,
            },
            {
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q2",
                "task_em": 0.0,
                "task_f1": 0.4,
                "llm_judge_score": 1.0,
                "llm_judge_correct": 0.0,
            },
        ]

        summaries = analysis_tools.summarize_instance_means(instance_rows)
        rows = analysis_tools.build_realistic_task_table_rows(summaries)

        self.assertEqual(rows[0], ["G1", "1.0000 ± 0.0000", "2.0000 ± 0.0000", "0.9000 ± 0.0000", "1.0000 ± 0.0000"])
        self.assertEqual(rows[1], ["G2", "0.0000 ± 0.0000", "1.0000 ± 0.0000", "0.4000 ± 0.0000", "0.0000 ± 0.0000"])
        self.assertEqual(rows[4], ["G5", "-", "-", "-", "-"])
        self.assertEqual(rows[6], ["G7", "-", "-", "-", "-"])

    def test_controlled_table_uses_no_drift_and_drift_rows(self) -> None:
        instance_rows = [
            {
                "experiment": "controlled_cognitive_asymmetry",
                "experiment_key": "controlled",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q1",
                "ber": 0.0,
                "decode_success": 1.0,
            },
            {
                "experiment": "controlled_cognitive_asymmetry",
                "experiment_key": "controlled",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "drift_recent3",
                "question_id": "q1",
                "ber": 0.5,
                "decode_success": 0.0,
            },
            {
                "experiment": "controlled_summary_asymmetry",
                "experiment_key": "controlled_summary",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "summary_only_enc",
                "question_id": "q1",
                "ber": 0.3,
                "decode_success": 0.4,
            },
        ]

        summaries = analysis_tools.summarize_instance_means(instance_rows)
        rows = analysis_tools.build_controlled_table_rows(summaries)

        self.assertEqual(
            rows[0],
            [
                "G2",
                "0.0000 ± 0.0000",
                "0.5000 ± 0.0000",
                "0.3000 ± 0.0000",
                "0.0000 ± 0.0000",
                "0.4000 ± 0.0000",
            ],
        )

    def test_controlled_cognitive_asymmetry_table_rows_support_acf_k(self) -> None:
        summaries = [
            {
                "experiment": "controlled_cognitive_asymmetry",
                "group": "G4",
                "condition": "no_drift",
                "acf_k": 12,
                "ber_mean": 0.01,
                "ber_std": 0.001,
            },
            {
                "experiment": "controlled_cognitive_asymmetry",
                "group": "G4",
                "condition": "drift_recent3",
                "acf_k": 12,
                "ber_mean": 0.02,
                "ber_std": 0.002,
                "decode_success_mean": 0.98,
                "decode_success_std": 0.01,
            },
            {
                "experiment": "controlled_summary_asymmetry",
                "group": "G4",
                "condition": "summary_only_enc",
                "acf_k": 12,
                "ber_mean": 0.03,
                "ber_std": 0.003,
                "decode_success_mean": 0.97,
                "decode_success_std": 0.02,
            },
        ]

        rows = analysis_tools.build_controlled_cognitive_asymmetry_table_rows(summaries)
        self.assertEqual(
            rows[3],
            [
                "ACF (k=12)",
                "1.00% ± 0.10%",
                "2.00% ± 0.20%",
                "98.00% ± 1.00%",
                "3.00% ± 0.30%",
                "97.00% ± 2.00%",
            ],
        )

    def test_realistic_protocol_table_uses_no_drift_only(self) -> None:
        instance_rows = [
            {
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q1",
                "ber": 0.1,
                "decode_success": 0.9,
                "embedding_capacity": 800.0,
            },
        ]

        summaries = analysis_tools.summarize_instance_means(instance_rows)
        rows = analysis_tools.build_realistic_protocol_table_rows(summaries)

        self.assertEqual(rows[0], ["G2", "0.1000 ± 0.0000", "0.9000 ± 0.0000", "800.0000 ± 0.0000"])

    def test_load_records_keeps_latest_capacity_and_normalizes_acf_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            group_dir = output_root / "group2"
            group_dir.mkdir(parents=True, exist_ok=True)

            record = {"run_id": "r1", "embedding_capacity": 800.0, "acf_k": 12.0}
            (group_dir / "r1.json").write_text(json.dumps(record), encoding="utf-8")

            records = sorted(analysis_tools.load_records(output_root), key=lambda row: str(row.get("run_id", "")))

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], "r1")
            self.assertAlmostEqual(records[0]["embedding_capacity"], 800.0)
            self.assertEqual(records[0]["acf_k"], 12)

    def test_task_correctness_vs_reliability_plot_uses_llm_judge(self) -> None:
        instance_rows = [
            {
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G1",
                "condition": "no_drift",
                "question_id": "q1",
                "task_em": 1.0,
                "task_f1": 0.9,
                "llm_judge_score": 2.0,
                "llm_judge_correct": 1.0,
            },
            {
                "experiment": "realistic_cognitive_asymmetry",
                "experiment_key": "realistic",
                "split": "longmemeval_s",
                "group": "G4",
                "condition": "no_drift",
                "question_id": "q2",
                "task_em": 0.5,
                "task_f1": 0.8,
                "llm_judge_score": 2.0,
                "llm_judge_correct": 1.0,
                "ber": 0.1,
            },
        ]

        summaries = analysis_tools.summarize_instance_means(instance_rows)
        points = analysis_tools.build_task_correctness_vs_reliability_plot(summaries)

        self.assertEqual(points[0]["group"], "G1")
        self.assertEqual(points[0]["task_correctness"], 1.0)
        self.assertIsNone(points[0]["communication_reliability"])
        self.assertEqual(points[0]["point_role"], "task_upper_bound_reference")
        self.assertEqual(points[3]["group"], "G4")
        self.assertAlmostEqual(points[3]["task_correctness"], 1.0)
        self.assertAlmostEqual(points[3]["communication_reliability"], 0.9)
        self.assertAlmostEqual(points[3]["legacy_task_f1"], 0.8)

    def test_realistic_integrated_table_rows_include_nominal_and_effective_capacity(self) -> None:
        summaries = [
            {
                "experiment": "realistic_cognitive_asymmetry",
                "group": "G1",
                "condition": "no_drift",
                "llm_judge_correct_mean": 0.5,
                "llm_judge_correct_std": 0.1,
                "llm_judge_score_mean": 1.2,
                "llm_judge_score_std": 0.2,
                "task_f1_mean": 0.04,
                "task_f1_std": 0.01,
            },
            {
                "experiment": "realistic_cognitive_asymmetry",
                "group": "G5",
                "condition": "no_drift",
                "acf_k": 12,
                "llm_judge_correct_mean": 0.6,
                "llm_judge_correct_std": 0.05,
                "llm_judge_score_mean": 1.5,
                "llm_judge_score_std": 0.1,
                "task_f1_mean": 0.07,
                "task_f1_std": 0.01,
                "ber_mean": 0.01,
                "ber_std": 0.001,
                "decode_success_mean": 0.99,
                "decode_success_std": 0.005,
                "embedding_capacity_mean": 250.0,
                "embedding_capacity_std": 20.0,
            },
        ]

        rows = analysis_tools.build_realistic_integrated_table_rows(summaries)
        self.assertEqual(
            rows[0],
            [
                "Normal (No Stego)",
                "50.00% ± 10.00%",
                "1.20 ± 0.20",
                "4.00% ± 1.00%",
                "---",
                "---",
                "---",
                "---",
            ],
        )
        self.assertEqual(
            rows[1],
            [
                "Normal+RET",
                "---",
                "---",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
        )
        self.assertEqual(
            rows[10],
            [
                "ACF+RET (k=12)",
                "60.00% ± 5.00%",
                "1.50 ± 0.10",
                "7.00% ± 1.00%",
                "1.00% ± 0.10%",
                "99.00% ± 0.50%",
                "250.0000 ± 20.0000",
                "247.5000 ± 19.8394",
            ],
        )

    def test_controlled_drift_severity_sweep_plot_orders_by_kept_sessions(self) -> None:
        instance_rows = [
            {
                "experiment": "controlled_drift_severity_sweep",
                "experiment_key": "controlled_sweep",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "no_drift",
                "question_id": "q1",
                "ber": 0.0,
                "decode_success": 1.0,
                "decode_recent_sessions_kept": 5,
            },
            {
                "experiment": "controlled_drift_severity_sweep",
                "experiment_key": "controlled_sweep",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "drift_recent4",
                "question_id": "q1",
                "ber": 0.2,
                "decode_success": 0.8,
                "decode_recent_sessions_kept": 4,
            },
            {
                "experiment": "controlled_drift_severity_sweep",
                "experiment_key": "controlled_sweep",
                "split": "longmemeval_s",
                "group": "G2",
                "condition": "drift_recent2",
                "question_id": "q1",
                "ber": 0.5,
                "decode_success": 0.0,
                "decode_recent_sessions_kept": 2,
            },
        ]

        summaries = analysis_tools.summarize_instance_means(instance_rows)
        points = analysis_tools.build_controlled_drift_severity_sweep_plot(summaries)

        self.assertEqual([point["condition"] for point in points], ["no_drift", "drift_recent4", "drift_recent2"])
        self.assertEqual([point["decoder_sessions_kept"] for point in points], [5, 4, 2])
        self.assertAlmostEqual(points[1]["ber"], 0.2)


if __name__ == "__main__":
    unittest.main()
