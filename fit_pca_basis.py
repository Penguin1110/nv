"""
從單一位置版的 W hidden state 檔案(小,可以在任何機器上跑)擬合 PCA 基底，
給 analyze_trajectories.py 的 --metric pca_projection 用。

為什麼從單一位置檔案擬合、而不是從完整時序檔案：後者太大(見
data/traces_Qwen3.5-4B_full.parquet, 1.49GB, 單一 row group,一般機器記憶體
裝不下)，但 PCA 基底只需要「這一層的 hidden state 大致分布在哪些方向上」，
用 90 題各一個向量(單一位置版本已有)就足夠擬合，不需要動用到完整時序資料。

用法：
  python fit_pca_basis.py \
      --traces data/traces_Qwen3.5-4B.parquet \
      --layer -1 \
      --n-components 5 \
      --out data/pca_basis_layer_last.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True,
                     help="單一位置版的 W traces 檔(欄位 hidden_all_layers)")
    ap.add_argument("--layer", type=int, default=-1,
                     help="要擬合哪一層(0=embedding,1..32=transformer層,-1=最後一層)")
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"[pca] 載入 {args.traces} ...", file=sys.stderr)
    df = pd.read_parquet(args.traces)
    print(f"  → {len(df)} 題", file=sys.stderr)

    vectors = np.array([
        np.array(row["hidden_all_layers"][args.layer], dtype=np.float32)
        for _, row in df.iterrows()
    ])
    print(f"[pca] vectors shape: {vectors.shape}", file=sys.stderr)

    n_components = min(args.n_components, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(vectors)

    print(f"[pca] 解釋變異量比例: {pca.explained_variance_ratio_}", file=sys.stderr)
    print(f"[pca] 累積: {pca.explained_variance_ratio_.cumsum()}", file=sys.stderr)

    basis = {
        "layer": args.layer,
        "source_traces": str(args.traces),
        "n_fit_samples": int(vectors.shape[0]),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),  # shape (n_components, hidden_dim)
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(basis, f)

    print(f"[pca] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
