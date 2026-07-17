# -*- coding: utf-8 -*-
"""
詞袋對照實驗：整輪 decode 分析的最終裁決。

背景：所有內部狀態分析裡唯一通過兩半驗證的訊號是 layer 0-1 的 mean-pool
(AUC ~0.65-0.70)。但 layer 0 是 embedding 層，它的軌跡平均在數學上非常接近
「生成文字用了哪些 token 的加權摘要」。所以要回答的問題是：

    這個訊號需要 hidden state 嗎？還是純粹看模型「寫了哪些字」就有了？

做法：同一批題目、同一套 leave-pair-out 切分，比三種特徵：
  bow      詞袋——每題的 generated_token_ids 統計成 token 出現次數向量
           (log(1+count) 後 L2 歸一)，完全不碰 hidden state
  layer0   embedding 層軌跡的 mean-pool(之前活下來的那個訊號，重算作為
           同場對照，保證兩者用同樣的題目、同樣的配對切分)
  length   生成長度(token 數)——最笨的 baseline

判讀：
  bow ≈ layer0        → 訊號在文字表面，hidden state 沒有加值，本輪可結案
  bow 明顯低於 layer0 → embedding 平均抓到了文字以外的東西，值得繼續挖
  (n=36 下 AUC 差距要 >0.15 才值得當一回事)

用法(在放得下 parquet 的機器上)：
  python compare_bow_baseline.py \
      --traces data/traces_Qwen3.5-4B_decode_first45.parquet \
               data/traces_Qwen3.5-4B_decode_last45.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --out results/bow_vs_layer0.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def lpo_auc(X, y, C=0.01):
    """Leave-pair-out AUC(訓練集恆平衡)。理由見 scan_decode_layers.py。"""
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    wins = 0.0
    for i in idx1:
        for j in idx0:
            mask = np.ones(len(X), bool)
            mask[i] = mask[j] = False
            sc = StandardScaler().fit(X[mask])
            clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[mask]), y[mask])
            si, sj = clf.decision_function(sc.transform(X[[i, j]]))
            wins += 1.0 if si > sj else (0.5 if si == sj else 0.0)
    return float(wins / (len(idx1) * len(idx0)))


def rank_auc(values, y):
    """單變數(如生成長度)不用訓練，直接算排序 AUC。"""
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    wins = sum(1.0 if values[i] > values[j] else (0.5 if values[i] == values[j] else 0.0)
               for i in i1 for j in i0)
    return float(wins / (len(i1) * len(i0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, nargs="+", required=True)
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--min-df", type=int, default=3,
                     help="token 至少要出現在幾題裡才進詞袋(太罕見的 token 只會"
                          "讓分類器記住個別題目)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    s = pd.read_parquet(args.s_labels)
    sp = s.groupby("qid")["correct"].mean()

    qids, y_list = [], []
    token_counts = {}   # qid -> Counter
    layer0_feats = {}   # qid -> 2560 維 mean-pool
    lengths = {}
    for path in args.traces:
        print(f"[bow] 載入 {path} ...", file=sys.stderr)
        df = pd.read_parquet(path)
        wrong = df[~df["correct"].astype(bool)]
        for _, row in wrong.iterrows():
            q = row["qid"]
            qids.append(q)
            y_list.append(1 if sp[q] <= 0.5 else 0)
            token_counts[q] = Counter(int(t) for t in row["generated_token_ids"])
            lengths[q] = int(row["n_new_tokens"])
            steps = row["decode_hidden_states_by_step"]
            layer0_feats[q] = np.mean(
                [np.asarray(step[0], dtype=np.float32) for step in steps], axis=0)
            print(f"[bow] qid={q} (tokens={lengths[q]}) 完成", file=sys.stderr)

    y = np.array(y_list)
    print(f"[bow] {len(qids)} 題, shared_hard={y.sum()}, salvageable={(1 - y).sum()}",
          file=sys.stderr)

    # 詞袋:只留出現在 >= min_df 題裡的 token
    df_count = Counter()
    for q in qids:
        for t in token_counts[q]:
            df_count[t] += 1
    vocab = sorted(t for t, c in df_count.items() if c >= args.min_df)
    print(f"[bow] 詞彙量: {len(vocab)} (min_df={args.min_df})", file=sys.stderr)

    X_bow = np.zeros((len(qids), len(vocab)), dtype=np.float32)
    t2i = {t: i for i, t in enumerate(vocab)}
    for r, q in enumerate(qids):
        for t, c in token_counts[q].items():
            if t in t2i:
                X_bow[r, t2i[t]] = np.log1p(c)
        norm = np.linalg.norm(X_bow[r])
        if norm > 0:
            X_bow[r] /= norm

    X_l0 = np.array([layer0_feats[q] for q in qids])
    lens = np.array([lengths[q] for q in qids], dtype=float)

    print("[bow] 計算三種特徵的 LPO AUC(同樣的題目、同樣的配對切分)...", file=sys.stderr)
    auc_bow = lpo_auc(X_bow, y)
    print(f"[bow] 詞袋(純文字):        AUC = {auc_bow:.4f}", file=sys.stderr)
    auc_l0 = lpo_auc(X_l0, y)
    print(f"[bow] layer 0 mean-pool:  AUC = {auc_l0:.4f}", file=sys.stderr)
    auc_len = rank_auc(lens, y)
    print(f"[bow] 生成長度:           AUC = {auc_len:.4f}", file=sys.stderr)

    out = {
        "traces": args.traces,
        "n_questions": len(qids),
        "n_shared_hard": int(y.sum()),
        "min_df": args.min_df,
        "vocab_size": len(vocab),
        "auc_bow": auc_bow,
        "auc_layer0_meanpool": auc_l0,
        "auc_length": auc_len,
        "qids": qids,
        "labels": y_list,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[bow] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
