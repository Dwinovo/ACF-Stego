from __future__ import annotations

from scripts import run_bert_binary_experiment


def _build_record(group: str, question_id: str, seed: int, *, acf_k: int | None = None) -> dict:
    record = {
        "run_id": f"realistic.longmemeval_s.{question_id}.{group}.no_drift.{seed}",
        "experiment_key": "realistic",
        "condition": "no_drift",
        "question_id": question_id,
        "seed": seed,
        "group": group,
        "assistant_answer": f"{group}:{question_id}:{seed}:{acf_k}",
    }
    if acf_k is not None:
        record["acf_k"] = acf_k
    return record


class TestBertBinaryExperiment:
    def test_task_specs_cover_expected_protocols(self) -> None:
        task_ids = [spec.task_id for spec in run_bert_binary_experiment.TASK_SPECS]
        assert len(task_ids) == 10
        assert "task_a_normal_vs_discop" in task_ids
        assert "task_j_normal_ret_vs_acf_ret_k16" in task_ids

    def test_build_task_pairs_matches_question_seed_and_k(self) -> None:
        records = [
            _build_record("G1", "q1", 42),
            _build_record("G1", "q1", 43),
            _build_record("G2", "q1", 42),
            _build_record("G2", "q1", 43),
            _build_record("G4", "q1", 42, acf_k=8),
            _build_record("G4", "q1", 42, acf_k=12),
            _build_record("G4", "q1", 42, acf_k=16),
            _build_record("G8", "q1", 42),
            _build_record("G5", "q1", 42, acf_k=12),
        ]
        record_index = run_bert_binary_experiment.build_record_index(records)

        task_discop = next(spec for spec in run_bert_binary_experiment.TASK_SPECS if spec.task_id == "task_a_normal_vs_discop")
        discop_pairs = run_bert_binary_experiment.build_task_pairs(task_discop, record_index)
        assert len(discop_pairs) == 2
        assert {pair.seed for pair in discop_pairs} == {42, 43}

        task_acf_k12 = next(spec for spec in run_bert_binary_experiment.TASK_SPECS if spec.task_id == "task_d_normal_vs_acf_k12")
        acf_pairs = run_bert_binary_experiment.build_task_pairs(task_acf_k12, record_index)
        assert len(acf_pairs) == 1
        assert acf_pairs[0].positive_acf_k == 12
        assert acf_pairs[0].negative_group == "G1"

        task_acf_ret_k12 = next(
            spec for spec in run_bert_binary_experiment.TASK_SPECS if spec.task_id == "task_i_normal_ret_vs_acf_ret_k12"
        )
        acf_ret_pairs = run_bert_binary_experiment.build_task_pairs(task_acf_ret_k12, record_index)
        assert len(acf_ret_pairs) == 1
        assert acf_ret_pairs[0].negative_group == "G8"
        assert acf_ret_pairs[0].positive_group == "G5"

    def test_split_pairs_and_flatten_keep_balance(self) -> None:
        pairs = [
            run_bert_binary_experiment.PairRecord(
                task_id="task_a_normal_vs_discop",
                question_id=f"q{idx}",
                seed=42,
                negative_text=f"normal-{idx}",
                positive_text=f"stego-{idx}",
                negative_group="G1",
                positive_group="G2",
                positive_acf_k=None,
            )
            for idx in range(360)
        ]

        train_pairs, val_pairs, test_pairs = run_bert_binary_experiment.split_pairs(pairs, seed=42)
        assert len(train_pairs) == 288
        assert len(val_pairs) == 36
        assert len(test_pairs) == 36

        train_samples = run_bert_binary_experiment.flatten_pairs(train_pairs)
        val_samples = run_bert_binary_experiment.flatten_pairs(val_pairs)
        test_samples = run_bert_binary_experiment.flatten_pairs(test_pairs)

        assert len(train_samples) == 576
        assert len(val_samples) == 72
        assert len(test_samples) == 72

        assert sum(1 for sample in train_samples if sample.label == 0) == 288
        assert sum(1 for sample in train_samples if sample.label == 1) == 288
        assert sum(1 for sample in val_samples if sample.label == 0) == 36
        assert sum(1 for sample in val_samples if sample.label == 1) == 36
        assert sum(1 for sample in test_samples if sample.label == 0) == 36
        assert sum(1 for sample in test_samples if sample.label == 1) == 36

        train_pair_ids = {sample.pair_id for sample in train_samples}
        val_pair_ids = {sample.pair_id for sample in val_samples}
        test_pair_ids = {sample.pair_id for sample in test_samples}

        assert train_pair_ids.isdisjoint(val_pair_ids)
        assert train_pair_ids.isdisjoint(test_pair_ids)
        assert val_pair_ids.isdisjoint(test_pair_ids)
