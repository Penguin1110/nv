"""
Generate synthetic data matching the REAL schema produced by this repo's
existing scripts, so run_mvp.py can be smoke-tested end-to-end without a GPU
or real model runs:

  - queries.jsonl                    ~ prepare_pool_data.py's --out-queries
  - traces_*.parquet                 ~ extract_all_layers.py's output
                                       (qid, correct, hidden_all_layers as a
                                       nested list[num_layers+1][hidden_dim],
                                       matching the `.tolist()` used there)
  - s_correctness_qwen3.5-27b.parquet  ~ run_s_inference.py's output
                                       (qid, sample_idx, model, correct)

Neither extract_all_layers.py (W) nor run_s_inference.py (S) has actually
been run for real yet — see EXPERIMENT.md Stage 2/3. This mock adds a
"sample_idx" column for both W and S so the difficulty-baseline control
(which needs multiple samples per qid to estimate pass@1) can be exercised
against something before real multi-sample data exists.

The synthetic signal is designed so the hypothesis SHOULD hold: a "shared
basin" latent per problem shifts W's early-layer hidden states AND makes
both W and S more likely to fail, so core_prediction.py's probe has a real
(if easy) signal to recover — this validates the pipeline runs correctly,
not that the scientific hypothesis is true on real data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles often default to a legacy codepage (e.g. cp950) that can't
# encode the checkmark/emoji characters used in status messages below.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

np.random.seed(42)

N_PROBLEMS = 80
SAMPLES_PER_PROBLEM = 8
N_LAYERS_INCL_EMBED = 33  # embedding + 32 transformer layers
HIDDEN_DIM = 64  # kept small so PCA/logreg run fast in a smoke test

S_MODEL_LABEL = "qwen/qwen3.5-27b"

Path("data").mkdir(exist_ok=True)

is_shared_basin = np.random.rand(N_PROBLEMS) < 0.4
base_difficulty = np.random.uniform(0.1, 0.9, N_PROBLEMS)

w_rows = []
s_rows = []
queries = []

for p in range(N_PROBLEMS):
    qid = f"prob_{p:03d}"
    shared = is_shared_basin[p]
    difficulty = base_difficulty[p]

    queries.append({
        "qid": qid,
        "query": f"Synthetic problem #{p}",
        "ground_truth": "A",
    })

    # Shared-basin problems: W's hidden state gets a distinct offset signature,
    # visible early (first half of layers) — this is the signal the probe
    # should be able to recover from an early-prefix slice.
    basin_offset = np.random.randn(HIDDEN_DIM) * 2.0 if shared else np.zeros(HIDDEN_DIM)

    for sample_idx in range(SAMPLES_PER_PROBLEM):
        hidden = np.random.randn(N_LAYERS_INCL_EMBED, HIDDEN_DIM).astype(np.float32)
        hidden[: N_LAYERS_INCL_EMBED // 2] += basin_offset

        w_fail_prob = difficulty * (1.5 if shared else 0.8)
        w_correct = int(np.random.rand() > min(w_fail_prob, 0.95))

        w_rows.append({
            "qid": qid,
            "sample_idx": sample_idx,
            "correct": w_correct,
            "hidden_all_layers": hidden.tolist(),  # matches extract_all_layers.py's .tolist()
            "raw_generation": "A" if w_correct else "B",
        })

        s_fail_prob = difficulty * (1.2 if shared else 0.15)
        s_correct = int(np.random.rand() > min(s_fail_prob, 0.95))

        s_rows.append({
            "qid": qid,
            "sample_idx": sample_idx,
            "model": S_MODEL_LABEL,
            "correct": s_correct,
        })

df_w = pd.DataFrame(w_rows)
df_s = pd.DataFrame(s_rows)

df_w.to_parquet("data/traces_Qwen3.5-4B.parquet")
df_s.to_parquet("data/s_correctness_qwen3.5-27b.parquet")

with open("data/queries.jsonl", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")

print(f"W traces: {len(df_w)} rows ({N_PROBLEMS} problems x {SAMPLES_PER_PROBLEM} samples)")
print(f"S correctness: {len(df_s)} rows, model={S_MODEL_LABEL}")
print(f"Queries: {len(queries)}")
print(f"W overall pass rate: {df_w['correct'].mean():.2%}")
print(f"S overall pass rate: {df_s['correct'].mean():.2%}")
print("✅ Mock data written to data/")
