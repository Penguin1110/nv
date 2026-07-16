"""
把 data/traces_Qwen3.5-4B_full.parquet 的完整時序 hidden state 壓縮成
「每個 token 位置一個代表數值」的軌跡，依 W✗S✗(共享難) vs W✗S✓(可救回) 分組比較。

因為這個檔案很大(單一 row group,17GB+ 解碼需求),必須在記憶體夠大的機器上跑
(不是那個受限的沙箱)。輸出只有壓縮後的小陣列,可以再帶回沙箱做後續統計/畫圖。

用法：
  python analyze_trajectories.py \
      --traces data/traces_Qwen3.5-4B_full.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --queries data/queries.jsonl \
      --layer -1 \
      --n-resample 20 \
      --out results/trajectories.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resample_trajectory(values: np.ndarray, n_points: int) -> np.ndarray:
    """
    把長度不一的軌跡(每題 seq_len 不同)線性內插成固定長度 n_points，
    這樣不同題目的軌跡才能疊在一起算平均/畫圖。x 軸視為 0~100% 的相對位置。
    """
    if len(values) == 1:
        return np.full(n_points, values[0])
    x_original = np.linspace(0, 1, len(values))
    x_target = np.linspace(0, 1, n_points)
    return np.interp(x_target, x_original, values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True)
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--layer", type=int, default=-1,
                     help="要看哪一層(0=embedding,1..32=transformer層,-1=最後一層)")
    ap.add_argument("--n-resample", type=int, default=20,
                     help="把每題軌跡內插成固定幾個點,方便跨題目比較/平均")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print("[traj] 載入 W 完整時序資料(這一步需要夠大的記憶體)...", file=sys.stderr)
    df_w = pd.read_parquet(args.traces)
    print(f"  → {len(df_w)} 題", file=sys.stderr)

    print("[traj] 載入 S 標籤...", file=sys.stderr)
    df_s = pd.read_parquet(args.s_labels)
    s_pass_rate = df_s.groupby("qid")["correct"].mean()

    # 建 2x2 標籤：只看 W 答錯的題目
    df_w["w_correct"] = df_w["correct"].astype(bool)
    df_w_wrong = df_w[~df_w["w_correct"]].copy()
    df_w_wrong["s_pass_rate"] = df_w_wrong["qid"].map(s_pass_rate)
    df_w_wrong["s_correct"] = df_w_wrong["s_pass_rate"] > 0.5
    df_w_wrong["label"] = (~df_w_wrong["s_correct"]).astype(int)  # 1=共享難, 0=可救回

    print(f"  W✗ 共 {len(df_w_wrong)} 題，共享難 {df_w_wrong['label'].sum()}，可救回 {(~df_w_wrong['label'].astype(bool)).sum()}", file=sys.stderr)

    trajectories = {"shared_hard": [], "salvageable": []}
    raw_per_qid = {}

    for _, row in df_w_wrong.iterrows():
        qid = row["qid"]
        hidden_by_pos = row["hidden_states_by_position"]  # list[seq_len][num_layers][hidden_dim]
        n_pos = len(hidden_by_pos)

        # 對每個位置，抽出指定層、算 L2 norm，壓成一個純量
        norms = np.empty(n_pos)
        for i, layer_vecs in enumerate(hidden_by_pos):
            vec = np.array(layer_vecs[args.layer], dtype=np.float32)
            norms[i] = np.linalg.norm(vec)

        resampled = resample_trajectory(norms, args.n_resample)
        raw_per_qid[qid] = norms.tolist()

        if row["label"] == 1:
            trajectories["shared_hard"].append(resampled)
        else:
            trajectories["salvageable"].append(resampled)

        print(f"[traj] qid={qid} 處理完成 (seq_len={n_pos}, label={'共享難' if row['label']==1 else '可救回'})", file=sys.stderr)

    result = {
        "layer": args.layer,
        "n_resample": args.n_resample,
        "shared_hard_mean": np.mean(trajectories["shared_hard"], axis=0).tolist() if trajectories["shared_hard"] else None,
        "shared_hard_std": np.std(trajectories["shared_hard"], axis=0).tolist() if trajectories["shared_hard"] else None,
        "shared_hard_n": len(trajectories["shared_hard"]),
        "salvageable_mean": np.mean(trajectories["salvageable"], axis=0).tolist() if trajectories["salvageable"] else None,
        "salvageable_std": np.std(trajectories["salvageable"], axis=0).tolist() if trajectories["salvageable"] else None,
        "salvageable_n": len(trajectories["salvageable"]),
        "raw_per_qid": raw_per_qid,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[traj] 完成 → {out_path}", file=sys.stderr)
    print(f"[traj] 共享難組: {result['shared_hard_n']} 題, 可救回組: {result['salvageable_n']} 題", file=sys.stderr)


if __name__ == "__main__":
    main()
