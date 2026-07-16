# -*- coding: utf-8 -*-
"""
軌跡動力學分析：不訓練分類器、不算 AUC，直接量測每題 decode 軌跡的幾何/
動力學性質，兩組(shared-hard vs salvageable)用 Mann-Whitney U 檢定比較。

動機：「共享難盆地」假設的字面意義就是動力學的——模型困在錯誤的吸引盆地裡
出不來。如果假設成立，訊號應該直接體現在「軌跡繞圈、移動慢、一直回到同一
區域」這類性質上，而不是(只)藏在某個 2560 維線性方向裡。每題壓成一個純量
再做無母數檢定，總共只有 6 個檢定，不會有掃 AUC 那種 660 格多重比較問題。

指標(全部在指定層、對取樣到的狀態序列計算)：
  mean_step_norm    相鄰取樣點的平均距離(除以平均狀態範數做尺度歸一)
  tortuosity        路徑總長 / 頭尾淨位移——越大越繞
  mean_consec_cos   相鄰兩步方向向量的平均 cosine——越高方向越一致
  longrange_selfsim 相隔超過 25% 進度的狀態對的平均 cosine——越高越「困在
                    同一區域」，這是「盆地」最直接的操作化定義
  settle_frac       從哪個相對進度開始，狀態跟最終狀態的 cosine 一直 >0.9
                    ——越小代表越早「定案」
  spread            狀態到軌跡質心的平均距離(除以平均狀態範數)

用法(在放得下 parquet 的機器上)：
  python analyze_decode_dynamics.py \
      --traces data/traces_Qwen3.5-4B_decode_last45.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --layer -1 \
      --out results/dynamics_last45_layer-1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def trajectory_metrics(states: np.ndarray) -> dict:
    """states: (n_steps, hidden_dim) float32，回傳 6 個動力學指標。"""
    n = len(states)
    norms = np.linalg.norm(states, axis=1)
    mean_norm = float(norms.mean())

    out = {}
    if n < 3:
        # 太短的軌跡大部分指標沒有意義，全部標 NaN，下游會剔除
        return {k: float("nan") for k in
                ["mean_step_norm", "tortuosity", "mean_consec_cos",
                 "longrange_selfsim", "settle_frac", "spread"]}

    diffs = states[1:] - states[:-1]
    step_norms = np.linalg.norm(diffs, axis=1)
    out["mean_step_norm"] = float(step_norms.mean() / mean_norm)

    net = float(np.linalg.norm(states[-1] - states[0]))
    out["tortuosity"] = float(step_norms.sum() / net) if net > 0 else float("nan")

    denom = step_norms[:-1] * step_norms[1:]
    valid = denom > 0
    cons = (diffs[:-1] * diffs[1:]).sum(axis=1)[valid] / denom[valid]
    out["mean_consec_cos"] = float(cons.mean()) if len(cons) else float("nan")

    # 長程自相似：相隔 >25% 進度的所有狀態對的平均 cosine
    unit = states / norms[:, None]
    sim = unit @ unit.T
    lag = max(1, int(n * 0.25))
    iu = np.triu_indices(n, k=lag)
    out["longrange_selfsim"] = float(sim[iu].mean()) if len(iu[0]) else float("nan")

    # 收斂時間：與最終狀態 cosine 從某點起持續 >0.9 的最早相對位置
    cos_final = sim[:, -1]
    settle = n - 1
    for i in range(n - 1, -1, -1):
        if cos_final[i] > 0.9:
            settle = i
        else:
            break
    out["settle_frac"] = float(settle / (n - 1))

    centroid = states.mean(axis=0)
    out["spread"] = float(np.linalg.norm(states - centroid, axis=1).mean() / mean_norm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, nargs="+", required=True,
                     help="一個或多個 decode parquet(給多個會合併，例如 first45+last45)")
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    s = pd.read_parquet(args.s_labels)
    sp = s.groupby("qid")["correct"].mean()

    per_qid = {}
    labels = {}
    for path in args.traces:
        print(f"[dyn] 載入 {path} ...", file=sys.stderr)
        df = pd.read_parquet(path)
        wrong = df[~df["correct"].astype(bool)]
        n_layers = int(wrong.iloc[0]["num_layers_incl_embed"])
        layer = args.layer % n_layers
        for _, row in wrong.iterrows():
            steps = row["decode_hidden_states_by_step"]
            states = np.array([np.asarray(step[layer], dtype=np.float32) for step in steps])
            per_qid[row["qid"]] = trajectory_metrics(states)
            labels[row["qid"]] = int(sp[row["qid"]] <= 0.5)  # 1=shared hard
            print(f"[dyn] qid={row['qid']} (steps={len(states)}) 完成", file=sys.stderr)

    metric_names = ["mean_step_norm", "tortuosity", "mean_consec_cos",
                    "longrange_selfsim", "settle_frac", "spread"]
    tests = {}
    print(f"\n[dyn] === 分組比較 (shared_hard={sum(labels.values())}, "
          f"salvageable={len(labels) - sum(labels.values())}) ===", file=sys.stderr)
    for m in metric_names:
        sh = np.array([per_qid[q][m] for q in per_qid if labels[q] == 1])
        sv = np.array([per_qid[q][m] for q in per_qid if labels[q] == 0])
        sh = sh[~np.isnan(sh)]
        sv = sv[~np.isnan(sv)]
        u, p = stats.mannwhitneyu(sh, sv, alternative="two-sided")
        tests[m] = {
            "shared_hard_median": float(np.median(sh)),
            "salvageable_median": float(np.median(sv)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
            "n_shared_hard": int(len(sh)),
            "n_salvageable": int(len(sv)),
        }
        print(f"[dyn] {m:18s} 中位數 共享難={np.median(sh):.4f} 可救回={np.median(sv):.4f}  p={p:.4f}",
              file=sys.stderr)

    out = {
        "traces": args.traces,
        "layer": args.layer,
        "labels": labels,
        "per_qid_metrics": per_qid,
        "group_tests": tests,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[dyn] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
