"""
MVP: Precedence / Leakage Control (Control #2)

Verify that failure-lock happens BEFORE the working answer emerges.
If failure happens AFTER answer stabilizes, suspect label leakage.

Approach:
  1. Use logit-lens: decode what the model thinks at each step
  2. Identify when the correct answer token appears likely
  3. Identify when trajectory diverges (failure-lock)
  4. Check: failure-lock.step < answer-emergence.step

Data requirement (NOT produced by extract_all_layers.py today):
  Each W trace row must carry:
    - "logits_per_step": array [num_steps, vocab_size]
    - "answer_token_id": int, the vocab id of the ground-truth answer token
  If these columns are absent, this module reports status
  "insufficient_data" instead of silently fabricating a "pass" verdict —
  see run_mvp.py's --skip-precedence flag to bypass this control entirely
  until that data is collected.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

import yaml

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class PrecedenceChecker:
    """
    Verify failure-lock precedes answer emergence.
    """

    REQUIRED_COLUMNS = {"logits_per_step", "answer_token_id"}

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.traces_w = None
        self.results = {}

    def load_data(self) -> None:
        """Load W traces with per-step logits."""
        print("[prec] Loading W traces with logit history...", file=sys.stderr)
        traces_path = self.config["data"]["traces_path"].format(
            model=self.config["models"]["weak"]["name"]
        )
        self.traces_w = pd.read_parquet(traces_path)
        print(f"  → {len(self.traces_w)} traces", file=sys.stderr)

    @staticmethod
    def _parse_logits(value) -> np.ndarray:
        """Normalize stored logits into a float32 [num_steps, vocab_size] array."""
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if isinstance(value, (bytes, bytearray)):
            return np.asarray(pickle.loads(value), dtype=np.float32)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float32)
        raise TypeError(f"Unsupported logits type: {type(value)}")

    def find_answer_emergence(
        self, logits_per_step: np.ndarray, answer_token_id: int
    ) -> Optional[int]:
        """
        Find first step where the answer token becomes competitive with the
        current top logit (within 1.0 nat). Returns None if it never does.
        """
        answer_logits = logits_per_step[:, answer_token_id]
        top_logits = logits_per_step.max(axis=1)

        logit_diffs = top_logits - answer_logits
        threshold = 1.0

        emergence_steps = np.where(logit_diffs < threshold)[0]
        if len(emergence_steps) > 0:
            return int(emergence_steps[0])
        return None

    def find_failure_lock(
        self, logits_per_step: np.ndarray
    ) -> Optional[Tuple[int, float]]:
        """
        Identify when the trajectory becomes persistently indecisive
        (sustained entropy spike), taken as a proxy for "failure-lock".

        Returns (step_index, divergence_score) or None if no spike found.
        """
        probs_per_step = np.exp(logits_per_step) / np.exp(logits_per_step).sum(axis=1, keepdims=True)
        entropies = entropy(probs_per_step.T)  # one entropy value per step

        window_size = max(2, len(entropies) // 4)
        rolling_entropy = pd.Series(entropies).rolling(window=window_size, center=True).mean()

        threshold = rolling_entropy.mean() + 0.5 * rolling_entropy.std()
        high_entropy_steps = np.where(rolling_entropy > threshold)[0]

        if len(high_entropy_steps) > 0:
            lock_step = int(high_entropy_steps[0])
            return lock_step, float(rolling_entropy.iloc[lock_step])

        return None

    def check_precedence_single_trace(
        self, logits_per_step: np.ndarray, answer_token_id: int, is_correct: bool
    ) -> Dict:
        """
        For a single W✗ trace, check whether failure-lock precedes answer
        emergence. Only meaningful for incorrect traces.
        """
        if is_correct:
            return {"is_correct": True, "check_result": "not_applicable"}

        lock_info = self.find_failure_lock(logits_per_step)
        emergence_step = self.find_answer_emergence(logits_per_step, answer_token_id)

        if lock_info is None:
            return {
                "is_correct": False,
                "failure_lock_step": None,
                "answer_emergence_step": emergence_step,
                "check_result": "no_lock",
            }

        lock_step, divergence = lock_info

        if emergence_step is None:
            return {
                "is_correct": False,
                "failure_lock_step": lock_step,
                "answer_emergence_step": None,
                "check_result": "no_emergence",
                "divergence_score": divergence,
            }

        precedence_satisfied = lock_step < emergence_step
        return {
            "is_correct": False,
            "failure_lock_step": lock_step,
            "answer_emergence_step": emergence_step,
            "precedence_satisfied": precedence_satisfied,
            "check_result": "pass" if precedence_satisfied else "suspicious",
            "divergence_score": divergence,
        }

    def run_checks(self, sample_limit: Optional[int] = None) -> None:
        """
        Run precedence check on a sample of W✗ traces. Requires
        REQUIRED_COLUMNS to be present in self.traces_w; if missing, reports
        status="insufficient_data" instead of a misleading "pass".
        """
        print("[prec] Checking precedence on W✗ traces...", file=sys.stderr)

        missing = self.REQUIRED_COLUMNS - set(self.traces_w.columns)
        if missing:
            print(f"  ⚠️  Missing columns: {sorted(missing)}", file=sys.stderr)
            print(
                "     Precedence check needs per-step logits + the ground-truth "
                "answer's token id. extract_all_layers.py does not save these yet — "
                "see DATA_CHECKLIST.md. Use --skip-precedence until collected.",
                file=sys.stderr,
            )
            self.results["precedence"] = {
                "status": "insufficient_data",
                "missing_columns": sorted(missing),
                "verdict": "insufficient_data",
            }
            return

        df_w_wrong = self.traces_w[~self.traces_w["correct"].astype(bool)]
        if sample_limit:
            df_w_wrong = df_w_wrong.sample(min(sample_limit, len(df_w_wrong)))

        results_per_trace = []
        check_results_count: Dict[str, int] = {}
        n_parse_errors = 0

        for idx, row in df_w_wrong.iterrows():
            try:
                logits_per_step = self._parse_logits(row["logits_per_step"])
                answer_token_id = int(row["answer_token_id"])

                result = self.check_precedence_single_trace(
                    logits_per_step, answer_token_id, bool(row["correct"])
                )
                results_per_trace.append(result)
                check_result = result.get("check_result", "unknown")
                check_results_count[check_result] = check_results_count.get(check_result, 0) + 1

            except Exception as e:
                n_parse_errors += 1
                if n_parse_errors <= 5:
                    print(f"  ⚠️  Skipping trace {idx}: {e}", file=sys.stderr)

        n_checked = len(results_per_trace)

        if n_checked == 0:
            print("  ⚠️  No traces could be checked (all failed to parse)", file=sys.stderr)
            self.results["precedence"] = {
                "status": "no_valid_traces",
                "verdict": "insufficient_data",
            }
            return

        print(f"\n  Precedence check results (n={n_checked}):", file=sys.stderr)
        for key, count in check_results_count.items():
            pct = 100.0 * count / n_checked
            print(f"    {key}: {count} ({pct:.1f}%)", file=sys.stderr)

        pass_count = check_results_count.get("pass", 0)
        suspicious_count = check_results_count.get("suspicious", 0)
        suspicious_rate = suspicious_count / n_checked

        if suspicious_rate > 0.1:
            print(
                f"  ⚠️  {suspicious_count}/{n_checked} traces show potential leakage "
                f"(answer emerged before failure-lock)",
                file=sys.stderr,
            )

        self.results["precedence"] = {
            "status": "ok",
            "total_checked": n_checked,
            "results": check_results_count,
            "pass_rate": pass_count / n_checked,
            "suspicious_rate": suspicious_rate,
            "verdict": "pass" if suspicious_rate <= 0.1 else "suspicious",
        }

    def run(self) -> None:
        """Main pipeline."""
        print("\n" + "=" * 70, file=sys.stderr)
        print("CONTROL #2: PRECEDENCE / LEAKAGE CHECK", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        if not self.config["precedence_check"]["enabled"]:
            print("  [disabled in config]", file=sys.stderr)
            self.results["precedence"] = {"status": "disabled", "verdict": "skipped"}
            return

        self.load_data()
        self.run_checks(sample_limit=100)

        out_dir = Path(self.config["output"]["results_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "precedence_check.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        print(f"✅ Results saved to {out_dir / 'precedence_check.json'}", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_experiments.yaml")
    args = ap.parse_args()

    checker = PrecedenceChecker(args.config)
    checker.run()
