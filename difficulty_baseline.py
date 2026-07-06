"""
MVP: Difficulty Baseline (Control #1)

This is THE critical control. We must prove:
  1. W's pass@1 is a naive difficulty baseline
  2. WITHIN same-difficulty bin, W's geometry still separates S✗ vs S✓
     → If yes: geometry carries info beyond difficulty (victory)
     → If no: we're only measuring difficulty (defeat)

This module takes `df_test` produced by CorePredictionPipeline.run()
(core_prediction.py) — it does NOT retrain anything. It only reads the
probe's predictions (column "y_pred_prob") off that dataframe and compares
them, bin by bin, against a pure pass@1 difficulty baseline.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

import yaml

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class DifficultyBaseline:
    """
    Compute W's pass@k, stratify by difficulty, evaluate probe AUC per bin.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.traces_w = None
        self.results = {}

    def load_data(self) -> None:
        """
        Load full W traces (all samples, not just the W✗ test split) —
        pass@1 needs every sample per problem to be a stable estimate.

        Skipped entirely if `run()` was already handed a pre-loaded
        `traces_w` (the normal path via run_mvp.py, which avoids reading the
        same parquet file twice).
        """
        print("[diff] Loading W traces...", file=sys.stderr)
        traces_path = self.config["data"]["traces_path"].format(
            model=self.config["models"]["weak"]["name"]
        )
        self.traces_w = pd.read_parquet(traces_path)

    def compute_pass_at_k(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each problem, compute W's pass@1 (and pass@k for k in config).
        Used as the difficulty signal / naive baseline.

        NOTE: extract_all_layers.py currently runs one greedy generation per
        qid (no sampling), so unless you've added a "sample_idx" column by
        running multiple sampled generations, `total_samples` will be 1 per
        qid and pass@1 collapses to {0, 1} — the difficulty-bin control below
        will then only ever see one difficulty bin. See DATA_CHECKLIST.md.
        """
        print("[diff] Computing pass@k per problem...", file=sys.stderr)

        pass_stats = (
            df.groupby("qid")
            .agg(
                total_samples=("correct", "count"),
                successes=("correct", "sum"),
            )
            .reset_index()
        )

        for k in self.config["difficulty_baseline"]["pass_at_k"]:
            pass_stats[f"pass_at_{k}"] = np.minimum(pass_stats["successes"], k) / k

        pass_stats["difficulty"] = 1.0 - pass_stats["pass_at_1"]

        print(
            f"  pass@1 range: [{pass_stats['pass_at_1'].min():.3f}, {pass_stats['pass_at_1'].max():.3f}]",
            file=sys.stderr,
        )

        return pass_stats

    def stratify_by_difficulty(self, df_test: pd.DataFrame, pass_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the probe's test-set predictions with difficulty scores and
        stratify into quantile bins.
        """
        print("[diff] Stratifying by difficulty...", file=sys.stderr)

        df = df_test.reset_index(drop=True).merge(
            pass_stats[["qid", "pass_at_1", "difficulty"]],
            on="qid",
            how="left",
        )

        n_bins = self.config["difficulty_baseline"]["n_bins"]
        # pass@1 is coarse (e.g. only 9 distinct levels with 8 samples/problem),
        # so plain qcut+duplicates="drop" can collapse to a single bin if one
        # difficulty value dominates the test split. Break ties by row order
        # first so qcut always returns close to n_bins groups of near-equal size.
        try:
            n_bins_actual = min(n_bins, df["difficulty"].nunique())
            if n_bins_actual < 2:
                raise ValueError("fewer than 2 distinct difficulty values")
            ranks = df["difficulty"].rank(method="first")
            df["difficulty_bin"] = pd.qcut(ranks, q=n_bins_actual, labels=False, duplicates="drop")
        except ValueError as e:
            print(f"  ⚠️  Falling back to a single bin: {e}", file=sys.stderr)
            df["difficulty_bin"] = 0

        for bin_id in sorted(df["difficulty_bin"].dropna().unique()):
            bin_mask = df["difficulty_bin"] == bin_id
            bin_difficulty_range = df.loc[bin_mask, "difficulty"]
            print(
                f"  Bin {bin_id}: n={bin_mask.sum()}, "
                f"difficulty ∈ [{bin_difficulty_range.min():.3f}, {bin_difficulty_range.max():.3f}]",
                file=sys.stderr,
            )

        return df

    def evaluate_probe_per_bin(self, df: pd.DataFrame) -> Dict:
        """
        For each difficulty bin, compute:
          - AUC of the probe's predictions (df["y_pred_prob"], geometry-based)
          - AUC of pass@1 (pure difficulty baseline)
          - Aggregate: does probe beat baseline on average (paired t-test)?
        """
        print("[diff] Evaluating probe AUC per difficulty bin...", file=sys.stderr)

        results_per_bin = {}
        probe_aucs = []
        baseline_aucs = []

        for bin_id in sorted(df["difficulty_bin"].dropna().unique()):
            mask = df["difficulty_bin"] == bin_id
            y_true_bin = df.loc[mask, "label"].values
            y_pred_prob_bin = df.loc[mask, "y_pred_prob"].values
            pass_at_1_bin = df.loc[mask, "pass_at_1"].values

            n_bin = int(mask.sum())
            n_shared_basin = int(y_true_bin.sum())

            if len(np.unique(y_true_bin)) < 2:
                probe_auc_bin = None
                baseline_auc_bin = None
                print(f"  Bin {bin_id} (n={n_bin}): skipped (single class in bin)", file=sys.stderr)
            else:
                probe_auc_bin = float(roc_auc_score(y_true_bin, y_pred_prob_bin))
                # Baseline: harder problem (lower pass@1) → more likely S also fails
                baseline_score_bin = 1.0 - pass_at_1_bin
                baseline_auc_bin = float(roc_auc_score(y_true_bin, baseline_score_bin))

                probe_aucs.append(probe_auc_bin)
                baseline_aucs.append(baseline_auc_bin)

                print(
                    f"  Bin {bin_id} (n={n_bin}, shared={n_shared_basin}): "
                    f"probe AUC={probe_auc_bin:.4f}, baseline AUC={baseline_auc_bin:.4f}",
                    file=sys.stderr,
                )

            results_per_bin[f"bin_{bin_id}"] = {
                "n": n_bin,
                "n_shared_basin": n_shared_basin,
                "probe_auc": probe_auc_bin,
                "baseline_auc": baseline_auc_bin,
            }

        if len(probe_aucs) >= 2:
            probe_aucs_arr = np.array(probe_aucs)
            baseline_aucs_arr = np.array(baseline_aucs)

            mean_probe = float(probe_aucs_arr.mean())
            mean_baseline = float(baseline_aucs_arr.mean())
            diff = mean_probe - mean_baseline

            t_stat, p_val = stats.ttest_rel(probe_aucs_arr, baseline_aucs_arr)

            print(f"\n  Probe beats baseline? Δ AUC = {diff:+.4f} (p={p_val:.4f})", file=sys.stderr)

            results_per_bin["summary"] = {
                "mean_probe_auc": mean_probe,
                "mean_baseline_auc": mean_baseline,
                "delta_auc": diff,
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "significant": bool(p_val < 0.05),
            }

            if diff <= 0 or p_val >= 0.05:
                print("  ❌ Probe does NOT beat difficulty baseline", file=sys.stderr)
            else:
                print("  ✅ Probe WINS over difficulty baseline", file=sys.stderr)
        else:
            print(
                "  ⚠️  Fewer than 2 usable bins — cannot compare probe vs. baseline reliably",
                file=sys.stderr,
            )
            results_per_bin["summary"] = {
                "status": "insufficient_bins",
                "usable_bins": len(probe_aucs),
            }

        return results_per_bin

    def run(self, df_test: pd.DataFrame, traces_w: pd.DataFrame = None) -> None:
        """
        Main: get W traces (for stable pass@1), stratify df_test by
        difficulty, evaluate probe vs. baseline AUC per bin.

        Args:
          df_test: test-split dataframe from CorePredictionPipeline, with
                    columns qid, label, y_pred_prob (at minimum).
          traces_w: pass `pipeline.traces_w` from an already-run
                    CorePredictionPipeline to skip reloading the same
                    parquet file from disk. If None, loads it independently
                    (needed for standalone use, see this file's __main__).
        """
        print("\n" + "=" * 70, file=sys.stderr)
        print("CONTROL #1: DIFFICULTY BASELINE (Same-Bin Separation)", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        required_cols = {"qid", "label", "y_pred_prob"}
        missing = required_cols - set(df_test.columns)
        if missing:
            raise ValueError(f"df_test missing required columns: {missing}")

        if traces_w is not None:
            self.traces_w = traces_w
        else:
            self.load_data()
        pass_stats = self.compute_pass_at_k(self.traces_w)
        df_stratified = self.stratify_by_difficulty(df_test, pass_stats)

        results = self.evaluate_probe_per_bin(df_stratified)
        self.results["difficulty_baseline"] = results

        out_dir = Path(self.config["output"]["results_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "difficulty_baseline.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Results saved to {out_dir / 'difficulty_baseline.json'}", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    from core_prediction import CorePredictionPipeline

    ap = argparse.ArgumentParser(
        description="Standalone runner: trains the core probe, then evaluates "
                    "it against the difficulty baseline, bin by bin."
    )
    ap.add_argument("--config", type=str, default="config_experiments.yaml")
    args = ap.parse_args()

    pipeline = CorePredictionPipeline(args.config)
    pipeline.run()

    baseline = DifficultyBaseline(args.config)
    baseline.run(pipeline.df_test, traces_w=pipeline.traces_w)
