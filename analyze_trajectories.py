"""
把 data/traces_Qwen3.5-4B_full.parquet 的完整時序 hidden state 壓縮成
「每個 token 位置一個代表數值」的軌跡，依 W✗S✗(共享難) vs W✗S✓(可救回) 分組比較。

因為這個檔案很大(單一 row group,17GB+ 解碼需求),必須在記憶體夠大的機器上跑
(不是那個受限的沙箱)。輸出只有壓縮後的小陣列,可以再帶回沙箱做後續統計/畫圖。

用法：
  # 純量指標:每個位置的向量長度(L2 norm)——只看「量值」,不看方向
  python analyze_trajectories.py \
      --traces data/traces_Qwen3.5-4B_full.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --metric norm --layer -1 \
      --out results/trajectories_norm.json

  # 投影指標:把每個位置投影到 PCA 主成分方向上——保留方向資訊,比 norm 更有意義。
  # 需要先跑 fit_pca_basis.py 從小的單一位置檔案擬合出 PCA 基底。
  python analyze_trajectories.py \
      --traces data/traces_Qwen3.5-4B_full.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --metric pca_projection --pca-basis data/pca_basis_layer_last.json \
      --n-components 3 \
      --out results/trajectories_pca.json
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


def project_onto_pca(vec: np.ndarray, pca_mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    把一個 hidden_dim 維的向量投影到 PCA 主成分方向上(先減去擬合時的平均值，
    再跟每個主成分方向做內積)，回傳每個主成分上的投影值(shape: n_components)。
    """
    return (vec - pca_mean) @ components.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True)
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--metric", choices=["norm", "pca_projection"], default="norm",
                     help="norm=每個位置的向量長度(只看量值)；"
                          "pca_projection=投影到 PCA 主成分方向(保留方向資訊，"
                          "需要 --pca-basis)")
    ap.add_argument("--pca-basis", type=str, default=None,
                     help="fit_pca_basis.py 產生的 PCA 基底檔(--metric pca_projection 時必填)")
    ap.add_argument("--n-components", type=int, default=1,
                     help="--metric pca_projection 時，要追蹤前幾個主成分(各自存一條軌跡)")
    ap.add_argument("--layer", type=int, default=-1,
                     help="要看哪一層(0=embedding,1..32=transformer層,-1=最後一層)。"
                          "--metric pca_projection 時必須跟 PCA 基底擬合時用的層一致")
    ap.add_argument("--n-resample", type=int, default=20,
                     help="把每題軌跡內插成固定幾個點,方便跨題目比較/平均")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    pca_mean = pca_components = None
    if args.metric == "pca_projection":
        if not args.pca_basis:
            print("[traj] --metric pca_projection 需要 --pca-basis", file=sys.stderr)
            sys.exit(1)
        with open(args.pca_basis, encoding="utf-8") as f:
            basis = json.load(f)
        if basis["layer"] != args.layer:
            print(
                f"[traj] 警告：PCA 基底是用 layer={basis['layer']} 擬合的，"
                f"跟這次指定的 --layer {args.layer} 不一致，投影結果沒有意義",
                file=sys.stderr,
            )
        pca_mean = np.array(basis["mean"], dtype=np.float32)
        pca_components = np.array(basis["components"], dtype=np.float32)[: args.n_components]

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

    # metric="norm" 只有一條軌跡；metric="pca_projection" 每個主成分各一條，
    # 用 component_0/component_1/... 當 key，兩種模式共用同一套分組/內插邏輯。
    n_series = args.n_components if args.metric == "pca_projection" else 1
    series_keys = [f"component_{k}" for k in range(n_series)] if args.metric == "pca_projection" else ["norm"]
    trajectories = {key: {"shared_hard": [], "salvageable": []} for key in series_keys}
    raw_per_qid = {}

    for _, row in df_w_wrong.iterrows():
        qid = row["qid"]
        hidden_by_pos = row["hidden_states_by_position"]  # list[seq_len][num_layers][hidden_dim]
        n_pos = len(hidden_by_pos)

        # 對每個位置，抽出指定層，依 --metric 壓成一個(或多個)純量
        series_values = np.empty((n_series, n_pos))
        for i, layer_vecs in enumerate(hidden_by_pos):
            vec = np.array(layer_vecs[args.layer], dtype=np.float32)
            if args.metric == "pca_projection":
                series_values[:, i] = project_onto_pca(vec, pca_mean, pca_components)
            else:
                series_values[0, i] = np.linalg.norm(vec)

        raw_per_qid[qid] = series_values.tolist()
        group = "shared_hard" if row["label"] == 1 else "salvageable"
        for k, key in enumerate(series_keys):
            resampled = resample_trajectory(series_values[k], args.n_resample)
            trajectories[key][group].append(resampled)

        print(f"[traj] qid={qid} 處理完成 (seq_len={n_pos}, label={'共享難' if row['label']==1 else '可救回'})", file=sys.stderr)

    series_results = {}
    n_shared_hard = n_salvageable = 0
    for key in series_keys:
        sh = trajectories[key]["shared_hard"]
        sv = trajectories[key]["salvageable"]
        n_shared_hard, n_salvageable = len(sh), len(sv)
        series_results[key] = {
            "shared_hard_mean": np.mean(sh, axis=0).tolist() if sh else None,
            "shared_hard_std": np.std(sh, axis=0).tolist() if sh else None,
            "salvageable_mean": np.mean(sv, axis=0).tolist() if sv else None,
            "salvageable_std": np.std(sv, axis=0).tolist() if sv else None,
        }

    result = {
        "metric": args.metric,
        "layer": args.layer,
        "n_resample": args.n_resample,
        "shared_hard_n": n_shared_hard,
        "salvageable_n": n_salvageable,
        "series": series_results,  # {"norm": {...}} 或 {"component_0": {...}, "component_1": {...}, ...}
        "raw_per_qid": raw_per_qid,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[traj] 完成 → {out_path}", file=sys.stderr)
    print(f"[traj] 共享難組: {n_shared_hard} 題, 可救回組: {n_salvageable} 題", file=sys.stderr)


if __name__ == "__main__":
    main()
