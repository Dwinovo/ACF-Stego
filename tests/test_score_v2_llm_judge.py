from __future__ import annotations

import unittest
from unittest import mock

from scripts import score_v2_llm_judge


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **_: object) -> object:
        self.calls += 1
        result = self._responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeChat:
    def __init__(self, responses: list[object]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = _FakeChat(responses)


class ScoreV2JudgeTest(unittest.TestCase):
    def test_is_timeout_error_detects_timeout_class(self) -> None:
        self.assertTrue(score_v2_llm_judge.is_timeout_error(TimeoutError("request timed out")))
        self.assertFalse(score_v2_llm_judge.is_timeout_error(ValueError("bad response payload")))

    def test_request_judge_completion_retries_timeout(self) -> None:
        client = _FakeClient(
            [
                TimeoutError("first timeout"),
                TimeoutError("second timeout"),
                {"status": "ok"},
            ]
        )

        with mock.patch("scripts.score_v2_llm_judge.time.sleep") as sleep_mock, mock.patch(
            "builtins.print"
        ) as print_mock:
            response = score_v2_llm_judge.request_judge_completion(
                client,
                run_id="realistic.longmemeval_s.q1.G2.no_drift.42",
                messages=[{"role": "user", "content": "judge this"}],
            )

        self.assertEqual(response, {"status": "ok"})
        self.assertEqual(client.chat.completions.calls, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        print_mock.assert_any_call("[retry-timeout] realistic.longmemeval_s.q1.G2.no_drift.42 retry=1")
        print_mock.assert_any_call("[retry-timeout] realistic.longmemeval_s.q1.G2.no_drift.42 retry=2")


if __name__ == "__main__":
    unittest.main()
