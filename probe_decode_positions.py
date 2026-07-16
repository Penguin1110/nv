# -*- coding: utf-8 -*-
"""
逐(層 x 時間位置)探測 decode-time hidden state：在每個相對生成進度的切片上，
用完整 2560 維向量訓練 logistic probe(leave-pair-out CV)、算 AUC，輸出「AUC 隨生成進度變化」
的曲線(每層一條)。

跟 scan_decode_layers.py(mean-pool，抹掉時序)的差別：這裡完全不做時間上的
壓縮——時序資訊體現在「哪個進度位置的 AUC 高」這條曲線本身；維度資訊也不壓
——每個切片都用全部 2560 維。兩者都保留，代價是檢定數量變多
(33 層 x 20 位置 = 660 個 AUC)，n≈20 的情況下一定會有雜訊浮上來，
所以這是探索性掃描：只拿來找「哪一層、哪段進度」是候選，
要用另一半資料(first45)在同一個(層,位置)上獨立驗證才算數。

每個切片的向量取法：nearest neighbor(取最接近該相對進度的實際取樣點)，
不做向量內插——內插出來的是資料裡不存在的假狀態。

必須在放得下完整 parquet 的機器上跑。

用法：
  python probe_decode_positions.py \
      --traces data/traces_Qwen3.5-4B_decode_last45.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --out results/probe_decode_positions_last45.json
"""

import argparse
import json
import sys
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
    """
    Leave-pair-out AUC(每次留兩組各一題、訓練集保持 9v9 平衡)。
    不用 LOO 的原因見 scan_decode_layers.py 的同名函式說明——LOO 在平衡小樣本
    上會讓截距系統性偏向另一組，強正則化下 AUC 被機制性拖到遠低於 0.5。
    """
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    wins = 0.0
    for i in idx1:
        for j in idx0:
            mask = np.ones(len(X), bool)
            mask[i] = mask[j] = False
            sc = StandardScaler().fit(X[mask])
            clf = LogisticRegression(max_iter=1000, C=C).fit(sc.transform(X[mask]), y[mask])
            si, sj = clf.decision_function(sc.transform(X[[i, j]]))
            wins += 1.0 if si > sj else (0.5 if si == sj else 0.0)
    return float(wins / (len(idx1) * len(idx0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True)
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--n-positions", type=int, default=20,
                     help="相對生成進度切成幾個位置(預設 20，即 0%%,5%%,...,100%%)")
    ap.add_argument("--layers", type=str, default="all",
                     help="要掃哪些層：'all' 或逗號分隔(例如 '0,8,16,24,32')。"
                          "先用 all 掃一遍,之後驗證時鎖定候選層就好")
    ap.add_argument("--C", type=float, default=0.01,
                     help="logistic regression 正則化強度(越小越強；n=20、d=2560 要壓很小)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"[probe] 載入 {args.traces} (需要夠大的記憶體)...", file=sys.stderr)
    df = pd.read_parquet(args.traces)
    print(f"  → {len(df)} 題", file=sys.stderr)

    s = pd.read_parquet(args.s_labels)
    sp = s.groupby("qid")["correct"].mean()

    wrong = df[~df["correct"].astype(bool)].copy()
    wrong["label"] = (wrong["qid"].map(sp) <= 0.5).astype(int)  # 1=shared hard
    print(f"[probe] W✗ {len(wrong)} 題：shared_hard={int(wrong['label'].sum())}, "
          f"salvageable={int((1 - wrong['label']).sum())}", file=sys.stderr)

    n_layers = int(wrong.iloc[0]["num_layers_incl_embed"])
    layer_list = (list(range(n_layers)) if args.layers.strip().lower() == "all"
                  else [int(x) % n_layers for x in args.layers.split(",")])
    y = wrong["label"].to_numpy()
    n_pos = args.n_positions
    rel_positions = np.linspace(0, 1, n_pos)

    # slices[l][p] = list of 2560-dim vectors (每題一個)
    slices = {l: [[] for _ in range(n_pos)] for l in layer_list}
    for _, row in wrong.iterrows():
        steps = row["decode_hidden_states_by_step"]  # (n_steps, n_layers, hidden_dim)
        n_steps = len(steps)
        # nearest-neighbor：每個相對位置取最接近的實際取樣點
        idxs = np.rint(rel_positions * (n_steps - 1)).astype(int)
        for p, si in enumerate(idxs):
            step = steps[si]
            for l in layer_list:
                slices[l][p].append(np.asarray(step[l], dtype=np.float32))
        print(f"[probe] qid={row['qid']} 切片完成 (steps={n_steps})", file=sys.stderr)

    auc_matrix = {}
    for l in layer_list:
        row_auc = []
        for p in range(n_pos):
            X = np.array(slices[l][p])
            row_auc.append(lpo_auc(X, y, C=args.C))
        auc_matrix[l] = row_auc
        best_p = int(np.argmax(row_auc))
        print(f"[probe] layer {l:2d}: AUC range [{min(row_auc):.3f}, {max(row_auc):.3f}], "
              f"最高在進度 {rel_positions[best_p]:.0%}", file=sys.stderr)

    out = {
        "traces": args.traces,
        "n_questions": int(len(wrong)),
        "n_shared_hard": int(wrong["label"].sum()),
        "n_positions": n_pos,
        "rel_positions": rel_positions.tolist(),
        "C": args.C,
        "qids": wrong["qid"].tolist(),
        "labels": wrong["label"].tolist(),
        "auc_by_layer": {str(l): auc_matrix[l] for l in layer_list},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
