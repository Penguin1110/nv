"""
逐層掃 AUC：對 extract_all_layers.py 存下的每一層 hidden state，
各訓一個 probe、算 CV AUC，畫出「哪一層最能預測 target 對錯」。

這一步在功能上取代 NVIDIA 的自動選層（Fisher Separability / Effective
Dimensionality）— 他們用幾何指標挑層，你直接暴力掃全層看 AUC 曲線，
對 3 個模型的規模這是合理且更直觀的替代（還能直接檢驗他們
「上半層最有訊號」的宣稱在 Qwen3 上成不成立）。

⚠️ 30 題規模下，每層的 AUC 都會非常吵（±0.2 是正常的）。
   這一輪只看「管線能不能跑出整條曲線」，數字本身不要當真。

用法：
  # target = 閉源模型（Encoder-Target Decoupling 的形狀）
  python sweep_layers.py \
      --encoder-feats data/all_layers_4b.parquet \
      --target-feats data/target_gpt5.parquet \
      --out results/layer_sweep_4b_to_gpt5.json

  # target = encoder 自己（自我預測，最基本的 sanity check）
  python sweep_layers.py \
      --encoder-feats data/all_layers_4b.parquet \
      --target-feats data/all_layers_4b.parquet \
      --out results/layer_sweep_4b_self.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cv_auc(X, y, pca_dim=50, n_splits=5, seed=3407):
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())),
                          shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        d = min(pca_dim, X.shape[1], len(tr) - 1)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=d)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        pipe.fit(X[tr], y[tr])
        if len(np.unique(y[va])) < 2:
            continue
        aucs.append(roc_auc_score(y[va], pipe.predict_proba(X[va])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-feats", type=str, required=True,
                     help="extract_all_layers.py 的輸出（含 hidden_all_layers 欄）")
    ap.add_argument("--target-feats", type=str, required=True,
                     help="qid/correct 標籤來源：extract 輸出檔，或 prepare_pool_data 的 labels.parquet"
                          "（後者需搭配 --target-model）")
    ap.add_argument("--target-model", type=str, default=None,
                     help="target-feats 是長表 labels.parquet 時，指定要當 target 的模型名"
                          "（需與 benchmark 目錄名一致）")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    enc = pd.read_parquet(args.encoder_feats)
    if "hidden_all_layers" not in enc.columns:
        raise ValueError("encoder 檔缺 hidden_all_layers 欄，請用 extract_all_layers.py 產生")
    tgt_df = pd.read_parquet(args.target_feats)
    if "model" in tgt_df.columns and tgt_df["model"].nunique() > 1:
        # 長表 labels.parquet：需要指定 target 模型
        if args.target_model is None:
            raise ValueError(
                f"labels 檔含多個模型，請用 --target-model 指定其一。可選：\n"
                + "\n".join(sorted(tgt_df["model"].unique())))
        tgt_df = tgt_df[tgt_df["model"] == args.target_model]
        if len(tgt_df) == 0:
            raise ValueError(f"labels 檔裡沒有模型 {args.target_model!r}，檢查名稱是否與目錄名一致")
    tgt = tgt_df[["qid", "correct"]].rename(columns={"correct": "correct_target"})

    merged = enc.merge(tgt, on="qid", how="inner")
    if len(merged) == 0:
        raise ValueError("qid 對不上任何一筆，確認兩邊用同一份 queries.jsonl")
    print(f"[sweep] 對齊 {len(merged)} 筆；target 答對率 "
          f"{merged['correct_target'].mean():.2f}")

    # hidden_all_layers: list[num_layers+1][hidden_dim]
    all_layers = merged["hidden_all_layers"].apply(
        lambda v: np.array([np.array(l) for l in v]))
    n_layers = len(all_layers.iloc[0])
    y = merged["correct_target"].values.astype(int)

    results = {}
    for li in range(n_layers):
        X = np.stack([row[li] for row in all_layers])
        auc = cv_auc(X, y, pca_dim=args.pca_dim)
        results[li] = auc
        tag = "embed" if li == 0 else f"L{li}"
        print(f"[sweep] {tag:>6}  auc={auc:.4f}")

    valid = {k: v for k, v in results.items() if not np.isnan(v)}
    best_layer = max(valid, key=valid.get) if valid else None

    out = {
        "encoder_feats": args.encoder_feats,
        "target_feats": args.target_feats,
        "n_samples": int(len(y)),
        "base_rate": float(y.mean()),
        "pca_dim": args.pca_dim,
        "auc_by_layer": results,
        "best_layer": best_layer,
        "best_auc": valid.get(best_layer) if best_layer is not None else None,
        "note": "layer 0 = embedding 輸出；小樣本下 AUC 噪音極大，僅供管線驗證",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[sweep] best layer = {best_layer}（auc={out['best_auc']}）→ {out_path}")


if __name__ == "__main__":
    main()
