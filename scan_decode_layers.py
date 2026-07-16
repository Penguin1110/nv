# -*- coding: utf-8 -*-
"""
逐層掃描 decode-time hidden state 的可分性：每一層各自算「shared-hard vs
salvageable」的 LOO-CV AUC，找出哪幾層帶最多訊號——回應「最後一層(為預測
下一個 token 服務)未必是任務語意最明顯的層」這個問題。

特徵做法(每層、每題)：把整條軌跡的 2560 維向量做平均(mean-pool)，
另外算一份只用前 25% 生成進度的早期平均——不經過 PCA，避免「主成分只
解釋 19% 變異」造成的資訊瓶頸。n 很小(約20題)、維度很高(2560)，所以
用強正則化的 logistic regression + LOO-CV 拿誠實的估計。

這是探索性掃描：33 層 x 2 種特徵 = 66 個 AUC，一定會有幾層因為運氣浮上來。
所以結果只拿來「挑候選層」，之後要用另一半資料(first45)獨立驗證才算數。

必須在放得下完整 parquet 的機器上跑(跟 fit_pca_basis.py --source decode 同級)。

用法：
  python scan_decode_layers.py \
      --traces data/traces_Qwen3.5-4B_decode_last45.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --out results/scan_decode_layers_last45.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def loo_auc(X, y, C=0.01):
    """LOO-CV 的 AUC。C 預設壓很小(強正則化)，因為 n≈20、d=2560。"""
    scores = np.empty(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), bool)
        mask[i] = False
        sc = StandardScaler().fit(X[mask])
        clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[mask]), y[mask])
        scores[i] = clf.predict_proba(sc.transform(X[i : i + 1]))[0, 1]
    return float(roc_auc_score(y, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True)
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--early-frac", type=float, default=0.25,
                     help="「早期」特徵取生成進度前多少比例(預設 0.25)")
    ap.add_argument("--C", type=float, default=0.01,
                     help="logistic regression 正則化強度(越小越強)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"[scan] 載入 {args.traces} (需要夠大的記憶體)...", file=sys.stderr)
    df = pd.read_parquet(args.traces)
    print(f"  → {len(df)} 題", file=sys.stderr)

    s = pd.read_parquet(args.s_labels)
    sp = s.groupby("qid")["correct"].mean()

    wrong = df[~df["correct"].astype(bool)].copy()
    wrong["label"] = (wrong["qid"].map(sp) <= 0.5).astype(int)  # 1=shared hard
    print(f"[scan] W✗ {len(wrong)} 題：shared_hard={int(wrong['label'].sum())}, "
          f"salvageable={int((1 - wrong['label']).sum())}", file=sys.stderr)

    n_layers = int(wrong.iloc[0]["num_layers_incl_embed"])
    y = wrong["label"].to_numpy()

    # 一次遍歷資料、同時累積每層的 mean-pool 特徵，避免 33 層各掃一遍 DataFrame
    # (巢狀欄位很大，iterrows 一次就好)。
    feats_full = {l: [] for l in range(n_layers)}   # 全程平均
    feats_early = {l: [] for l in range(n_layers)}  # 前 early_frac 平均
    for _, row in wrong.iterrows():
        steps = row["decode_hidden_states_by_step"]  # (n_steps, n_layers, hidden_dim)
        arr = np.array([np.array([np.asarray(v, dtype=np.float32) for v in step]) for step in steps])
        n_early = max(1, int(len(arr) * args.early_frac))
        for l in range(n_layers):
            feats_full[l].append(arr[:, l].mean(axis=0))
            feats_early[l].append(arr[:n_early, l].mean(axis=0))
        print(f"[scan] qid={row['qid']} 特徵完成 (steps={len(arr)})", file=sys.stderr)

    results = []
    for l in range(n_layers):
        auc_full = loo_auc(np.array(feats_full[l]), y, C=args.C)
        auc_early = loo_auc(np.array(feats_early[l]), y, C=args.C)
        results.append({"layer": l, "auc_full_mean": auc_full, "auc_early_mean": auc_early})
        print(f"[scan] layer {l:2d}: AUC(全程平均)={auc_full:.3f}  AUC(早期平均)={auc_early:.3f}",
              file=sys.stderr)

    out = {
        "traces": args.traces,
        "n_questions": int(len(wrong)),
        "n_shared_hard": int(wrong["label"].sum()),
        "early_frac": args.early_frac,
        "C": args.C,
        "qids": wrong["qid"].tolist(),
        "labels": wrong["label"].tolist(),
        "per_layer": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[scan] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
