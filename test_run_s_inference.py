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


class TestScoring(unittest.TestCase):
    def test_extract_choice_letter(self):
        self.assertEqual(rsi.extract_choice_letter("The answer is B."), "B")
        self.assertIsNone(rsi.extract_choice_letter("I'm not sure."))

    def test_score_answer(self):
        self.assertEqual(rsi.score_answer("The answer is C", "C"), 1)
        self.assertEqual(rsi.score_answer("The answer is C", "d"), 0)
        self.assertEqual(rsi.score_answer("no letter here", "A"), 0)


class TestCallOpenRouter(unittest.TestCase):
    def test_success(self):
        with patch("run_s_inference.requests.post", return_value=_fake_response(200, "Answer: A")) as m:
            text = rsi.call_openrouter("key", "qwen/qwen3.5-27b", "2+2=?", 0.7, 512, 30.0, 3)
        self.assertEqual(text, "Answer: A")
        m.assert_called_once()

    def test_retries_then_succeeds(self):
        responses = [_fake_response(429, text="rate limited"), _fake_response(200, "Answer: B")]
        with patch("run_s_inference.requests.post", side_effect=responses):
            with patch("run_s_inference.time.sleep"):  # skip real backoff delay in test
                text = rsi.call_openrouter("key", "qwen/qwen3.5-27b", "q", 0.7, 512, 30.0, 3)
        self.assertEqual(text, "Answer: B")

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
        with patch("run_s_inference.call_openrouter", return_value="Answer: A"):
            self._run_main(n_samples=2)

        df = pd.read_parquet(self.out_path)
        self.assertEqual(len(df), 6)  # 3 qids x 2 samples
        self.assertTrue((df["correct"] == 1).all())

        # 續跑：call_openrouter 這次會直接報錯，若續跑邏輯正確，
        # 因為所有 (qid, sample_idx) 都已存在，不該再呼叫它，也不該報錯。
        with patch("run_s_inference.call_openrouter", side_effect=AssertionError("不該被呼叫")):
            self._run_main(n_samples=2)

        df_after = pd.read_parquet(self.out_path)
        self.assertEqual(len(df_after), 6)  # 沒有重複、也沒有新增

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
