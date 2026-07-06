"""
MVP: Core Prediction Task
Predict "S also fails" from W's early hidden-state trajectory.

Core flow:
  1. Load W traces + S correctness labels
  2. Filter to W✗ problems only
  3. For each prefix fraction, extract early-trace features
  4. Train probe: early trace → "S also fails" (label=1)
  5. Compute AUC (must beat difficulty baseline)
  6. Report per-cell stats with CI

After run(), the pipeline exposes:
  - self.df_test:  test-split rows (qid, label, y_pred_prob, ...)
                   used by difficulty_baseline.py for same-bin analysis
  - self.results["core_prediction"]: AUC / CI / probe object
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import yaml

# Windows consoles often default to a legacy codepage (e.g. cp950) that can't
# encode the checkmark/emoji characters used in status messages below.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class CorePredictionPipeline:
    """
    Train & evaluate probe: W's early trace → "S also fails"
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.traces_w = None
        self.s_correctness = None
        self.queries = None
        self.results = {}
        self.df_test = None  # populated by run(); consumed by difficulty_baseline.py

    def load_data(self) -> None:
        """
        Load:
          - W traces (parquet, from extract_all_layers.py):
              qid, correct, hidden_all_layers, ...
          - S correctness (parquet, from prepare_pool_data.py's labels.parquet
            or a single-model equivalent): qid, [model,] correct
          - Queries (jsonl, from prepare_pool_data.py): qid, query, ground_truth
        """
        print("[core] Loading W traces...", file=sys.stderr)
        traces_path = self.config["data"]["traces_path"].format(
            model=self.config["models"]["weak"]["name"]
        )
        self.traces_w = pd.read_parquet(traces_path)
        print(f"  → {len(self.traces_w)} samples", file=sys.stderr)

        print("[core] Loading S correctness labels...", file=sys.stderr)
        s_path = self.config["data"]["s_correctness_path"]
        s_df = pd.read_parquet(s_path)
        if "model" in s_df.columns and s_df["model"].nunique() > 1:
            # Long-format labels.parquet covering the whole benchmark pool —
            # filter down to the configured S model.
            s_model_label = self.config["data"]["s_model_label"]
            s_df = s_df[s_df["model"] == s_model_label]
            if len(s_df) == 0:
                available = sorted(pd.read_parquet(s_path)["model"].unique())
                raise ValueError(
                    f"s_model_label={s_model_label!r} not found in {s_path}. "
                    f"Available: {available}"
                )
        self.s_correctness = s_df
        print(f"  → {len(self.s_correctness)} samples", file=sys.stderr)

        print("[core] Loading queries...", file=sys.stderr)
        self.queries = []
        with open(self.config["data"]["queries_path"], "r", encoding="utf-8") as f:
            for line in f:
                self.queries.append(json.loads(line))
        print(f"  → {len(self.queries)} problems", file=sys.stderr)

    def build_labels(self) -> pd.DataFrame:
        """
        For each W trace, decide:
          - W✓/W✗: W's correctness
          - S✓/S✗: S's correctness (aggregate over samples)
          - label: 1 if W✗S✗ (shared basin), 0 if W✗S✓ (salvageable)

        Only return W✗ rows (core prediction subset).
        """
        print("[core] Building 2×2 labels...", file=sys.stderr)

        # Aggregate S correctness per problem (mean = pass rate; with a
        # single-sample S file this is just 0 or 1, which is fine — pass_rate
        # > 0.5 still reduces to "S got it right")
        s_agg = (
            self.s_correctness.groupby("qid")
            .agg({"correct": ["mean", "sum", "count"]})
            .reset_index()
        )
        s_agg.columns = ["qid", "s_pass_rate", "s_correct_count", "s_total"]

        # Merge W traces with S info
        df = self.traces_w.merge(s_agg, on="qid", how="left")

        # Define S✓ if pass_rate > 0.5 (at least half succeed)
        df["s_correct"] = (df["s_pass_rate"] > 0.5).astype(int)
        df["w_correct"] = df["correct"].astype(bool)

        # Label for probe: 1 = shared basin (W✗S✗), 0 = salvageable (W✗S✓)
        df["label"] = ((~df["w_correct"]) & (~df["s_correct"].astype(bool))).astype(int)

        # Filter to W✗ only, and reset index so positions are contiguous
        # (downstream featurize/train_test_split relies on positional alignment)
        df_w_wrong = df[~df["w_correct"]].reset_index(drop=True).copy()

        # Report cell distribution
        cells = df_w_wrong.groupby("label").size()
        print(f"  W✗S✗ (shared basin): {cells.get(1, 0)}", file=sys.stderr)
        print(f"  W✗S✓ (salvageable):  {cells.get(0, 0)}", file=sys.stderr)

        # Warning if cells too small
        for cell_val, cell_name in [(1, "W✗S✗"), (0, "W✗S✓")]:
            n = cells.get(cell_val, 0)
            if n < self.config["statistics"]["min_cell_size"]:
                print(
                    f"  ⚠️  {cell_name}: n={n} < min {self.config['statistics']['min_cell_size']}",
                    file=sys.stderr,
                )

        return df_w_wrong

    @staticmethod
    def _parse_hidden_state(value) -> np.ndarray:
        """
        extract_all_layers.py stores `hidden_all_layers` as a plain Python
        list[num_layers+1][hidden_dim] (via `.tolist()`), which parquet
        round-trips as a nested list/ndarray-of-objects. Normalize whatever
        comes back into a float32 [num_layers, hidden_dim] array — same
        parsing sweep_layers.py uses (`np.array([np.array(l) for l in v])`).
        """
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.float32)
        if isinstance(value, (bytes, bytearray)):
            return np.asarray(pickle.loads(value), dtype=np.float32)
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.asarray([np.asarray(layer, dtype=np.float32) for layer in value])
        if isinstance(value, str):
            # Fallback for JSON-encoded strings; not the expected path.
            return np.asarray(json.loads(value), dtype=np.float32)
        raise TypeError(f"Unsupported hidden state type: {type(value)}")

    def extract_early_trace_features(
        self, traces: np.ndarray, prefix_fraction: float
    ) -> np.ndarray:
        """
        Extract features from early portion of trace.

        Args:
          traces: shape [num_layers, hidden_dim] of all-layer hidden states
          prefix_fraction: 0.25 → use first 25% of layers

        Returns:
          features: shape [feature_dim] (fixed length regardless of cutoff)
        """
        n_layers = traces.shape[0]
        cutoff = max(1, int(n_layers * prefix_fraction))
        early_trace = traces[:cutoff, :]

        features = [
            early_trace.mean(axis=0),  # [hidden_dim]
            early_trace.std(axis=0),
            early_trace.max(axis=0),
        ]

        # PCA compression over the layer axis (treats each layer as one
        # observation). Needs >= 2 layers to fit; pad to a fixed width of 10
        # so every row produces the same feature length even when cutoff
        # varies near the start of the sequence.
        n_pca_components = 10
        if early_trace.shape[0] >= 2:
            n_components = min(n_pca_components, early_trace.shape[0] - 1)
            pca = PCA(n_components=n_components)
            pca_feat = pca.fit_transform(early_trace).mean(axis=0)
            if n_components < n_pca_components:
                pca_feat = np.pad(pca_feat, (0, n_pca_components - n_components))
        else:
            pca_feat = np.zeros(n_pca_components)
        features.append(pca_feat)

        return np.concatenate(features)

    def featurize_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract early-trace features from all samples.

        Args:
          df: labeled dataset (W✗ rows), must have a contiguous 0..n-1 index

        Returns:
          X: [n_valid, feature_dim]
          y: [n_valid] ∈ {0, 1}
          row_positions: [n_valid] positions into `df` (via .iloc) that
                         survived parsing, for re-aligning results back to df
        """
        print("[core] Featurizing early traces...", file=sys.stderr)

        prefix_fraction = self.config["core_prediction"]["prefix_fraction"]
        features_list = []
        row_positions = []
        n_parse_errors = 0

        for pos in range(len(df)):
            row = df.iloc[pos]
            try:
                hidden = self._parse_hidden_state(row["hidden_all_layers"])
                feat = self.extract_early_trace_features(hidden, prefix_fraction)
                features_list.append(feat)
                row_positions.append(pos)
            except Exception as e:
                n_parse_errors += 1
                if n_parse_errors <= 5:
                    print(f"  ⚠️  Skipping row {pos}: {e}", file=sys.stderr)

        if n_parse_errors > 5:
            print(f"  ⚠️  ... and {n_parse_errors - 5} more parse errors", file=sys.stderr)

        X = np.array(features_list)
        row_positions = np.array(row_positions)
        y = df.iloc[row_positions]["label"].values

        print(f"  → X shape: {X.shape}, y: {y.sum()}/{len(y)} (shared basin)", file=sys.stderr)
        return X, y, row_positions

    def train_and_evaluate_probe(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
        """
        Train logistic regression probe on early traces.
        Report AUC with 95% CI via bootstrap.

        `groups` must hold one qid per row of X/y. Multiple samples of the
        same problem share the same S-derived label (it's a per-problem
        aggregate), so a plain row-level split can put samples from the same
        problem on both sides and leak problem identity into the "test"
        score. We split by qid (GroupShuffleSplit) instead, so every sample
        of a given problem lands entirely in train or entirely in test.

        Also returns `test_positions` (indices into X/y) and the raw
        `y_pred_prob_test` array so callers can re-attach predictions to the
        original dataframe rows for downstream analysis (e.g. difficulty
        stratification).
        """
        print("[core] Training probe...", file=sys.stderr)

        all_positions = np.arange(len(X))
        random_state = self.config["core_prediction"]["random_state"]
        test_fraction = self.config["core_prediction"]["test_fraction"]

        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
            pos_train, pos_test = next(splitter.split(X, y, groups=groups))
        else:
            # Only one problem in the whole dataset — grouping is moot
            pos_train, pos_test = train_test_split(
                all_positions, test_size=test_fraction, random_state=random_state
            )

        X_train, X_test = X[pos_train], X[pos_test]
        y_train, y_test = y[pos_train], y[pos_test]

        probe = LogisticRegression(max_iter=1000, random_state=random_state)
        probe.fit(X_train, y_train)

        y_pred_prob = probe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        ci_low, ci_high = self._bootstrap_ci(y_test, y_pred_prob)

        print(f"  AUC: {auc:.4f} [95% CI: {ci_low:.4f} – {ci_high:.4f}]", file=sys.stderr)

        return {
            "auc": float(auc),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_test": int(len(y_test)),
            "n_shared_basin_test": int(y_test.sum()),
            "probe": probe,
            "test_positions": pos_test,       # positions into X/y (0-indexed)
            "y_pred_prob_test": y_pred_prob,  # aligned with test_positions
        }

    def _bootstrap_ci(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[float, float]:
        """Compute 95% CI for AUC via bootstrap resampling."""
        n_resamples = self.config["statistics"]["ci_resamples"]
        aucs = []

        for _ in range(n_resamples):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue  # skip degenerate resamples (all one class)
            aucs.append(roc_auc_score(y_true[indices], y_pred_prob[indices]))

        if not aucs:
            return float("nan"), float("nan")

        aucs = np.array(aucs)
        return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    def run(self) -> None:
        """
        Main pipeline: load → label → featurize → train probe → report.

        Raises RuntimeError (not sys.exit) on hard failures so callers like
        run_mvp.py can catch it and continue reporting other steps.
        """
        print("\n" + "=" * 70, file=sys.stderr)
        print("MVP: CORE PREDICTION (W✗ → S✗)", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        self.load_data()
        df_labeled = self.build_labels()

        min_w_wrong = self.config["experiment_a"]["min_w_wrong_samples"]
        if len(df_labeled) < min_w_wrong:
            raise RuntimeError(f"Too few W✗ samples: {len(df_labeled)} < {min_w_wrong}")

        X, y, row_positions = self.featurize_dataset(df_labeled)

        if len(np.unique(y)) < 2:
            raise RuntimeError("Only one class in labels, cannot train probe")

        groups = df_labeled.iloc[row_positions]["qid"].values
        results = self.train_and_evaluate_probe(X, y, groups)
        self.results["core_prediction"] = results

        # Re-attach predictions to their original dataframe rows so
        # difficulty_baseline.py can stratify by difficulty without any
        # separate positional bookkeeping.
        test_row_positions = row_positions[results["test_positions"]]
        self.df_test = df_labeled.iloc[test_row_positions].copy()
        self.df_test["y_pred_prob"] = results["y_pred_prob_test"]

        # Save results (exclude non-JSON-serializable objects/arrays)
        out_dir = Path(self.config["output"]["results_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        skip_keys = {"probe", "test_positions", "y_pred_prob_test"}
        with open(out_dir / "core_prediction.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in results.items() if k not in skip_keys},
                f,
                indent=2,
            )

        print(f"✅ Results saved to {out_dir / 'core_prediction.json'}", file=sys.stderr)

        auc = results["auc"]
        min_cell = self.config["experiment_a"]["min_cell_w_wrong_s_wrong"]
        if auc <= 0.5:
            print(f"\n❌ FAIL: AUC = {auc:.4f} ≤ 0.5 (no signal)", file=sys.stderr)
        elif results["n_shared_basin_test"] < min_cell:
            print(
                f"\n⚠️  WARNING: Only {results['n_shared_basin_test']} shared-basin test samples",
                file=sys.stderr,
            )
        else:
            print(f"\n✅ PASS: AUC = {auc:.4f}, sufficient cell sizes", file=sys.stderr)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_experiments.yaml")
    args = ap.parse_args()

    pipeline = CorePredictionPipeline(args.config)
    pipeline.run()
