"""
擬合 PCA 基底，給 analyze_trajectories.py 的 --metric pca_projection 用。

支援兩種來源(--source)，基底必須跟要投影的資料同一種來源才有意義——
prompt 的 prefill 向量(模型「讀題」時的表示)跟生成過程中的 decode 向量
(模型「答題」時的表示)在 residual stream 裡的分布不必然相似，混用等於拿
量錯的尺去量:
  prefill = 從單一位置版的 W hidden state 檔案(小,可以在任何機器上跑)擬合。
            為什麼不從完整時序檔案擬合：後者太大(見
            data/traces_Qwen3.5-4B_full.parquet, 1.49GB, 單一 row group,
            一般機器記憶體裝不下),但 PCA 基底只需要「這一層的 hidden state
            大致分布在哪些方向上」,用 90 題各一個向量就足夠擬合。
  decode  = 從 extract_all_layers.py --capture-decode-hidden 產生的檔案擬合,
            把「每題、每個生成步驟」的向量全部攤平、當成同一個母體來擬合——
            這樣基底才反映的是「模型在生成推理過程中」的向量分布,而不是
            「讀題時」的分布。

用法：
  python fit_pca_basis.py \
      --traces data/traces_Qwen3.5-4B.parquet \
      --source prefill --layer -1 \
      --n-components 5 \
      --out data/pca_basis_layer_last.json

  python fit_pca_basis.py \
      --traces data/traces_Qwen3.5-4B_decode.parquet \
      --source decode --layer -1 \
      --n-components 5 \
      --out data/pca_basis_decode_layer_last.json
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


def resolve_captured_layer_index(captured_layers, num_layers_total, want_layer):
    """
    decode 模式只存了選定的幾層(見 --decode-layers)，不是完整的 0..num_layers_total-1，
    所以「第 want_layer 層」在 decode_hidden_states_by_step[step] 這個 list 裡的實際
    index，要用 decode_layers_captured 這欄反查——不能直接拿 want_layer 當 index 用。
    """
    want_idx = num_layers_total + want_layer if want_layer < 0 else want_layer
    resolved = [num_layers_total + l if l < 0 else l for l in captured_layers]
    if want_idx not in resolved:
        raise ValueError(
            f"要擬合的層 {want_layer}(解析後絕對層數 {want_idx})不在這個檔案實際存的 "
            f"decode_layers_captured={captured_layers} 之中"
        )
    return resolved.index(want_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True,
                     help="單一位置版的 W traces 檔(--source prefill,欄位 hidden_all_layers)"
                          "或 decode-hidden 檔(--source decode,欄位 decode_hidden_states_by_step)")
    ap.add_argument("--source", choices=["prefill", "decode"], default="prefill",
                     help="prefill=用 prompt 最後位置的向量擬合(單一位置版檔案)；"
                          "decode=把生成過程中每一步的向量全部攤平當同一個母體擬合")
    ap.add_argument("--layer", type=int, default=-1,
                     help="要擬合哪一層(0=embedding,1..32=transformer層,-1=最後一層)")
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"[pca] 載入 {args.traces} ...", file=sys.stderr)
    df = pd.read_parquet(args.traces)
    print(f"  → {len(df)} 題", file=sys.stderr)

    if args.source == "decode":
        vectors = []
        for _, row in df.iterrows():
            layer_idx = resolve_captured_layer_index(
                row["decode_layers_captured"], row["num_layers_incl_embed"], args.layer
            )
            for step_layers in row["decode_hidden_states_by_step"]:
                vectors.append(np.array(step_layers[layer_idx], dtype=np.float32))
        vectors = np.array(vectors)
        print(f"[pca] 從 {len(df)} 題攤平出 {vectors.shape[0]} 個 decode 步驟向量", file=sys.stderr)
    else:
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
        "source": args.source,
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
