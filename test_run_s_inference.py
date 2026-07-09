"""
用 mock 掉 requests.post 的方式測試 run_s_inference.py 的邏輯，不需要真的
OPENROUTER_API_KEY、也不需要對外網路。跑法：

  python test_run_s_inference.py

涵蓋：答案抽取/計分、成功回應解析、429 重試後成功、4xx 不重試直接拋錯、
以及 main() 端到端跑一輪 + 續跑時正確跳過已完成的 (qid, sample_idx)。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

import run_s_inference as rsi


def _fake_response(status_code, content=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if content is not None:
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _fake_message_response(status_code, message, finish_reason=None):
    """給需要精確控制 message dict(例如 content=None 但有 reasoning_content)的測試用。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    resp.json.return_value = {
        "choices": [{"message": message, "finish_reason": finish_reason}]
    }
    return resp


class TestScoring(unittest.TestCase):
    def test_extract_choice_letter(self):
        self.assertEqual(rsi.extract_choice_letter("The answer is B."), "B")
        self.assertIsNone(rsi.extract_choice_letter("I'm not sure."))

    def test_score_answer(self):
        self.assertEqual(rsi.score_answer("The answer is C", "C"), 1)
        self.assertEqual(rsi.score_answer("The answer is C", "d"), 0)
        self.assertEqual(rsi.score_answer("no letter here", "A"), 0)

    def test_extract_numeric_answer_boxed(self):
        self.assertEqual(rsi.extract_numeric_answer("blah blah \\boxed{116} done"), "116")

    def test_extract_numeric_answer_answer_is(self):
        self.assertEqual(rsi.extract_numeric_answer("So the answer is 42."), "42")

    def test_extract_numeric_answer_fallback_last_number(self):
        # No boxed/"answer is" pattern -> falls back to last standalone integer
        self.assertEqual(rsi.extract_numeric_answer("We had 3 cases, then 7, so total 10"), "10")

    def test_extract_numeric_answer_none(self):
        self.assertIsNone(rsi.extract_numeric_answer("I have no idea what the answer could be."))

    def test_score_answer_numeric(self):
        self.assertEqual(rsi.score_answer("\\boxed{756}", "756", "numeric"), 1)
        self.assertEqual(rsi.score_answer("\\boxed{756}", "757", "numeric"), 0)
        self.assertEqual(rsi.score_answer("no number at all", "150", "numeric"), 0)
        # boxed takes priority even if a different number appears earlier in reasoning
        self.assertEqual(
            rsi.score_answer("first I tried 99 but the real answer is \\boxed{150}", "150", "numeric"), 1
        )

    def test_none_generated_text_does_not_crash(self):
        # Regression: OpenRouter can return content=None (e.g. truncated reasoning
        # model output) — extraction/scoring must degrade to "no answer", not raise.
        self.assertIsNone(rsi.extract_choice_letter(None))
        self.assertIsNone(rsi.extract_numeric_answer(None))
        self.assertEqual(rsi.score_answer(None, "150", "numeric"), 0)
        self.assertEqual(rsi.score_answer(None, "A", "letter"), 0)

    def test_none_ground_truth_does_not_crash(self):
        self.assertEqual(rsi.score_answer("\\boxed{150}", None, "numeric"), 0)
        self.assertEqual(rsi.score_answer("The answer is A", None, "letter"), 0)


class TestCallOpenRouter(unittest.TestCase):
    def test_success(self):
        with patch("run_s_inference.requests.post", return_value=_fake_response(200, "Answer: A")) as m:
            text, finish_reason = rsi.call_openrouter("key", "qwen/qwen3.5-27b", "2+2=?", 0.7, 512, 30.0, 3)
        self.assertEqual(text, "Answer: A")
        self.assertIsNone(finish_reason)  # _fake_response doesn't set one
        m.assert_called_once()

    def test_retries_then_succeeds(self):
        responses = [_fake_response(429, text="rate limited"), _fake_response(200, "Answer: B")]
        with patch("run_s_inference.requests.post", side_effect=responses):
            with patch("run_s_inference.time.sleep"):  # skip real backoff delay in test
                text, _ = rsi.call_openrouter("key", "qwen/qwen3.5-27b", "q", 0.7, 512, 30.0, 3)
        self.assertEqual(text, "Answer: B")

    def test_null_content_falls_back_to_reasoning_content(self):
        # Regression for the real failure seen against qwen3.5-27b: content is null,
        # actual output ended up in reasoning_content instead.
        resp = _fake_message_response(
            200, {"content": None, "reasoning_content": "I worked it out: \\boxed{116}"},
            finish_reason="length",
        )
        with patch("run_s_inference.requests.post", return_value=resp):
            text, finish_reason = rsi.call_openrouter("key", "m", "q", 0.7, 512, 30.0, 3)
        self.assertEqual(text, "I worked it out: \\boxed{116}")
        self.assertEqual(finish_reason, "length")

    def test_null_content_and_no_reasoning_raises_clear_error(self):
        resp = _fake_message_response(200, {"content": None}, finish_reason="length")
        with patch("run_s_inference.requests.post", return_value=resp):
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                rsi.call_openrouter("key", "m", "q", 0.7, 512, 30.0, 3)

    def test_exhausts_retries(self):
        with patch("run_s_inference.requests.post", return_value=_fake_response(503, text="down")):
            with patch("run_s_inference.time.sleep"):
                with self.assertRaises(RuntimeError):
                    rsi.call_openrouter("key", "qwen/qwen3.5-27b", "q", 0.7, 512, 30.0, 2)

    def test_4xx_does_not_retry(self):
        with patch("run_s_inference.requests.post", return_value=_fake_response(401, text="bad key")) as m:
            with self.assertRaises(RuntimeError):
                rsi.call_openrouter("key", "qwen/qwen3.5-27b", "q", 0.7, 512, 30.0, 3)
        # 401 應該立刻拋錯，不該觸發重試迴圈裡的多次呼叫
        self.assertEqual(m.call_count, 1)


class TestMainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.queries_path = Path(self.tmpdir.name) / "queries.jsonl"
        self.out_path = Path(self.tmpdir.name) / "s_out.parquet"

        with open(self.queries_path, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"qid": f"q{i}", "query": f"Question {i}", "ground_truth": "A"}) + "\n")

    def _run_main(self, n_samples=2, limit=None):
        argv = [
            "run_s_inference.py",
            "--queries", str(self.queries_path),
            "--out", str(self.out_path),
            "--n-samples", str(n_samples),
            "--flush-every", "1",
        ]
        if limit is not None:
            argv += ["--limit", str(limit)]
        with patch.object(sys, "argv", argv):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                rsi.main()

    def test_writes_expected_rows_and_resumes(self):
        with patch("run_s_inference.call_openrouter", return_value=("Answer: A", "stop")):
            self._run_main(n_samples=2)

        df = pd.read_parquet(self.out_path)
        self.assertEqual(len(df), 6)  # 3 qids x 2 samples
        self.assertTrue((df["correct"] == 1).all())
        self.assertTrue((df["finish_reason"] == "stop").all())

        # 續跑：call_openrouter 這次會直接報錯，若續跑邏輯正確，
        # 因為所有 (qid, sample_idx) 都已存在，不該再呼叫它，也不該報錯。
        with patch("run_s_inference.call_openrouter", side_effect=AssertionError("不該被呼叫")):
            self._run_main(n_samples=2)

        df_after = pd.read_parquet(self.out_path)
        self.assertEqual(len(df_after), 6)  # 沒有重複、也沒有新增

    def test_failures_logged_to_errors_jsonl(self):
        # Alternate success/failure so we can check both the main output and
        # the sidecar error log end up with the right counts.
        call_count = {"n": 0}

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("simulated API failure")
            return "Answer: A", "stop"

        with patch("run_s_inference.call_openrouter", side_effect=flaky):
            self._run_main(n_samples=2)  # 3 qids x 2 samples = 6 calls -> 3 ok, 3 fail

        df = pd.read_parquet(self.out_path)
        self.assertEqual(len(df), 3)

        errors_path = self.out_path.with_suffix(self.out_path.suffix + ".errors.jsonl")
        self.assertTrue(errors_path.exists())
        with open(errors_path, encoding="utf-8") as f:
            error_rows = [json.loads(line) for line in f]
        self.assertEqual(len(error_rows), 3)
        self.assertIn("simulated API failure", error_rows[0]["error"])
        self.assertIn("qid", error_rows[0])
        self.assertIn("sample_idx", error_rows[0])

    def test_missing_api_key_exits(self):
        argv = [
            "run_s_inference.py",
            "--queries", str(self.queries_path),
            "--out", str(self.out_path),
        ]
        with patch.object(sys, "argv", argv):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit):
                    rsi.main()


if __name__ == "__main__":
    unittest.main()
