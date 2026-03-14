from __future__ import annotations

import unittest

from core.tools import qa_metrics


class QaMetricsTest(unittest.TestCase):
    def test_exact_match_uses_full_prediction(self) -> None:
        prediction = "My sister Emily lives in Denver."
        self.assertEqual(qa_metrics.exact_match_score(prediction, "Denver"), 0.0)

    def test_token_f1_uses_full_prediction(self) -> None:
        prediction = (
            "My sister Emily lives in Denver. She's excited to show you around "
            "and help you find some fun kid-friendly attractions while you're there."
        )
        self.assertAlmostEqual(qa_metrics.token_f1_score(prediction, "Denver"), 0.08333333333333333)

    def test_token_f1_still_rewards_exact_short_span(self) -> None:
        self.assertEqual(qa_metrics.token_f1_score("Denver", "Denver"), 1.0)


if __name__ == "__main__":
    unittest.main()
