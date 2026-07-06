"""
MVP Orchestrator: Run all three essential controls for Experiment A.

Flow:
  1. Core Prediction: W's early trace → "S also fails" (AUC)
  2. Difficulty Baseline: same-difficulty-bin separation (AUC per bin)
  3. Precedence Check: failure-lock before answer emergence
  4. Report: Go/No-go decision

Step 2 consumes `pipeline.df_test` produced by Step 1 directly — there is no
separate serialization/reload step, so the probe's predictions and the
difficulty bins are always computed on the exact same rows.
"""

import sys
import json
import argparse
from pathlib import Path

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core_prediction import CorePredictionPipeline
from difficulty_baseline import DifficultyBaseline
from precedence_check import PrecedenceChecker


def main():
    ap = argparse.ArgumentParser(
        description="MVP: Shared Hard Basin - Experiment A (Same Family)"
    )
    ap.add_argument("--config", type=str, default="config_experiments.yaml", help="Config file (YAML)")
    ap.add_argument("--skip-precedence", action="store_true",
                    help="Skip precedence check (if per-step logit data not available)")
    ap.add_argument("--results-dir", type=str, default="results/", help="Output directory for results")
    args = ap.parse_args()

    print("\n" + "=" * 80, file=sys.stderr)
    print(" " * 20 + "MVP: SHARED HARD BASIN", file=sys.stderr)
    print(" " * 15 + "Experiment A (Same-Family Model)", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Core Prediction
    # ========================================================================
    print("\n[STEP 1/3] Core Prediction Task", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    core_auc = core_ci_low = core_ci_high = None
    pipeline = None

    try:
        pipeline = CorePredictionPipeline(args.config)
        pipeline.run()

        core_results = pipeline.results.get("core_prediction", {})
        core_auc = core_results.get("auc")
        core_ci_low = core_results.get("ci_low")
        core_ci_high = core_results.get("ci_high")

        if core_auc is None or core_auc <= 0.5:
            print(f"\n❌ FAIL @ Step 1: Core AUC = {core_auc} ≤ 0.5", file=sys.stderr)
            print("  → No cross-model signal detected.", file=sys.stderr)
            _save_summary(out_dir, {
                "experiment": "A (Same-Family)",
                "step_1_core_prediction": {"status": "fail", "auc": core_auc},
                "step_2_difficulty_baseline": {"status": "not_run"},
                "step_3_precedence": {"status": "not_run"},
                "verdict": "NO-GO",
            })
            return False

        print(
            f"\n✅ PASS @ Step 1: Core AUC = {core_auc:.4f} [{core_ci_low:.4f}–{core_ci_high:.4f}]",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"❌ Step 1 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        _save_summary(out_dir, {
            "experiment": "A (Same-Family)",
            "step_1_core_prediction": {"status": "error", "message": str(e)},
            "verdict": "NO-GO",
        })
        return False

    # ========================================================================
    # STEP 2: Difficulty Baseline (Same-Bin Separation)
    # ========================================================================
    print("\n[STEP 2/3] Difficulty Baseline Control", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    min_auc_over_baseline = pipeline.config["experiment_a"]["min_auc_over_baseline"]

    try:
        baseline = DifficultyBaseline(args.config)
        baseline.run(pipeline.df_test, traces_w=pipeline.traces_w)
        baseline_results = baseline.results.get("difficulty_baseline", {})

        summary = baseline_results.get("summary", {})
        beats_by_enough = summary.get("delta_auc", 0) >= min_auc_over_baseline
        if summary.get("significant") and beats_by_enough:
            print(
                f"\n✅ PASS @ Step 2: Probe beats difficulty baseline "
                f"(Δ={summary['delta_auc']:+.4f} ≥ {min_auc_over_baseline}, p={summary['p_val']:.4f})",
                file=sys.stderr,
            )
        elif summary.get("status") == "insufficient_bins":
            print("\n⚠️  Step 2 inconclusive: not enough usable difficulty bins", file=sys.stderr)
        elif summary.get("significant") and not beats_by_enough:
            print(
                f"\n⚠️  Step 2: significant but small margin "
                f"(Δ={summary['delta_auc']:+.4f} < required {min_auc_over_baseline})",
                file=sys.stderr,
            )
        else:
            print("\n❌ Step 2: Probe does NOT significantly beat difficulty baseline", file=sys.stderr)

    except Exception as e:
        print(f"⚠️  Step 2 failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        baseline_results = {"status": "error", "message": str(e)}

    # ========================================================================
    # STEP 3: Precedence / Leakage Control
    # ========================================================================
    if not args.skip_precedence:
        print("\n[STEP 3/3] Precedence / Leakage Control", file=sys.stderr)
        print("-" * 80, file=sys.stderr)

        try:
            checker = PrecedenceChecker(args.config)
            checker.run()

            prec_results = checker.results.get("precedence", {})
            prec_verdict = prec_results.get("verdict", "unknown")

            if prec_verdict == "suspicious":
                print("\n⚠️  WARNING @ Step 3: Potential leakage detected", file=sys.stderr)
            elif prec_verdict == "insufficient_data":
                print("\n⚠️  Step 3 skipped: required logit columns not present in data", file=sys.stderr)
            else:
                print("\n✅ PASS @ Step 3: Precedence satisfied", file=sys.stderr)

        except Exception as e:
            print(f"⚠️  Step 3 failed: {e}", file=sys.stderr)
            prec_results = {"status": "error", "message": str(e)}
    else:
        print("\n[STEP 3/3] Precedence Check [SKIPPED via --skip-precedence]", file=sys.stderr)
        prec_results = {"status": "skipped", "reason": "--skip-precedence flag"}

    # ========================================================================
    # SUMMARY & GO/NO-GO
    # ========================================================================
    print("\n" + "=" * 80, file=sys.stderr)
    print("MVP SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    print(f"\nCore Prediction AUC: {core_auc:.4f} [{core_ci_low:.4f}–{core_ci_high:.4f}]", file=sys.stderr)

    print("\n" + "-" * 80, file=sys.stderr)
    print("GO/NO-GO DECISION:", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    diff_summary = baseline_results.get("summary", {})
    beats_difficulty = (
        bool(diff_summary.get("significant"))
        and diff_summary.get("delta_auc", 0) >= min_auc_over_baseline
    )
    prec_verdict = prec_results.get("verdict", "skipped")

    if core_auc <= 0.5:
        verdict = "NO-GO"
        print("❌ NO-GO: Core AUC ≤ 0.5 (no signal)", file=sys.stderr)
    elif baseline_results.get("status") == "error":
        verdict = "BLOCKED"
        print("⚠️  BLOCKED: Difficulty baseline failed (see above)", file=sys.stderr)
        print("   → Recheck: sufficient multi-sample data per problem?", file=sys.stderr)
    elif not beats_difficulty:
        verdict = "NO-GO (difficulty only)"
        print("❌ NO-GO: Probe does not beat the pass@1 difficulty baseline", file=sys.stderr)
    elif prec_verdict == "suspicious":
        verdict = "CONDITIONAL"
        print("⚠️  CONDITIONAL: Signal beats difficulty, but leakage is suspected", file=sys.stderr)
        print("   → Recheck: are hidden states captured before generation starts?", file=sys.stderr)
    else:
        verdict = "GO"
        print("✅ GO: Experiment A passed core checks", file=sys.stderr)
        if prec_verdict == "insufficient_data":
            print("   (precedence check not yet run — collect logit data before publishing)", file=sys.stderr)
        print("   → Ready for Phase 2 (Experiment B with heterogeneous S)", file=sys.stderr)

    summary_payload = {
        "experiment": "A (Same-Family)",
        "step_1_core_prediction": {"status": "pass", "auc": core_auc, "ci": [core_ci_low, core_ci_high]},
        "step_2_difficulty_baseline": baseline_results,
        "step_3_precedence": prec_results,
        "verdict": verdict,
    }
    _save_summary(out_dir, summary_payload)

    print(f"\n📄 Full summary → {out_dir / 'mvp_summary.json'}", file=sys.stderr)
    print("\n" + "=" * 80, file=sys.stderr)

    return verdict == "GO"


def _save_summary(out_dir: Path, summary: dict) -> None:
    with open(out_dir / "mvp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
