"""
從 LLMRouterBench 下載包一次收齊:
  1. 題目(用 records 的 `prompt` 欄——注意不是 origin_query,見下)
  2. 池內「每一個模型」的對錯標籤(records 的 score 欄)

為什麼用 prompt 而不是 origin_query:
  benchmark 收集答案時餵給模型的是加了格式指令的完整 prompt。
  你之後抽的 hidden state 要對應「產生那個被評分答案」的內部狀態,
  就必須餵一模一樣的字串。餵 origin_query 會讓 hidden state 和 score 對不上。

輸出:
  --out-queries : queries.jsonl(qid, prompt, ground_truth)
  --out-labels  : labels.parquet(qid, model, correct)— 長表,每模型每題一列

用法:
  python prepare_pool_data.py \
      --bench-root <LLMRouterBench 根目錄> \
      --dataset mmlu_pro \
      --out-queries data/queries.jsonl \
      --out-labels data/labels.parquet \
      --limit 30
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="mmlu_pro")
    ap.add_argument("--out-queries", type=str, required=True)
    ap.add_argument("--out-labels", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="只取前 N 個 qid(依 index 排序後),pilot 用")
    args = ap.parse_args()

    ds_dir = Path(args.bench_root) / "results" / "bench" / args.dataset
    if not ds_dir.exists():
        raise FileNotFoundError(
            f"{ds_dir} 不存在。確認已解壓 bench-release 且 dataset 名稱正確;"
            f"可先 ls results/bench/ 看實際的資料集目錄名。")

    # 每個模型一個子目錄(目錄名 = 模型名,這就是「規定好的池」的權威名單)
    model_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        # 有些版本可能多一層 split 目錄
        for split in ("test", "validation", "val"):
            alt = ds_dir / split
            if alt.exists():
                model_dirs = sorted([d for d in alt.iterdir() if d.is_dir()])
                break
    if not model_dirs:
        raise FileNotFoundError(f"{ds_dir} 底下找不到模型子目錄,請確認目錄結構")

    print(f"[pool] 找到 {len(model_dirs)} 個模型目錄(這就是規定的池):",
          file=sys.stderr)
    for d in model_dirs:
        print(f"[pool]   {d.name}", file=sys.stderr)

    label_rows = []
    queries = {}  # qid -> {prompt, ground_truth}

    for mdir in model_dirs:
        json_files = sorted(mdir.rglob("*.json"))
        if not json_files:
            print(f"[pool] 警告:{mdir.name} 沒有結果檔,跳過", file=sys.stderr)
            continue
        # 同模型若有多個 timestamp 檔,取最新的一個
        with open(json_files[-1], "r", encoding="utf-8") as f:
            blob = json.load(f)
        for rec in blob.get("records", []):
            qid = rec.get("index")
            prompt = rec.get("prompt") or rec.get("origin_query")
            gt = rec.get("ground_truth")
            score = rec.get("score")
            if qid is None or prompt is None or score is None:
                continue
            label_rows.append({
                "qid": qid,
                "model": mdir.name,
                "correct": int(float(score) > 0.5),
            })
            if qid not in queries:
                queries[qid] = {"prompt": prompt, "ground_truth": gt}

    if not label_rows:
        raise ValueError("沒有讀到任何標籤,確認檔案格式與欄位名(index/prompt/score)")

    qids_sorted = sorted(queries.keys())
    if args.limit is not None:
        keep = set(qids_sorted[: args.limit])
        label_rows = [r for r in label_rows if r["qid"] in keep]
        qids_sorted = [q for q in qids_sorted if q in keep]

    out_q = Path(args.out_queries)
    out_q.parent.mkdir(parents=True, exist_ok=True)
    with open(out_q, "w", encoding="utf-8") as f:
        for qid in qids_sorted:
            f.write(json.dumps({
                "qid": qid,
                "query": queries[qid]["prompt"],   # 注意:這是完整 prompt
                "ground_truth": queries[qid]["ground_truth"],
            }, ensure_ascii=False) + "\n")

    out_l = Path(args.out_labels)
    out_l.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(label_rows)
    df.to_parquet(out_l)

    # 摘要:每個模型的答對率,順便讓你檢查池子有沒有讀對
    summary = df.groupby("model")["correct"].agg(["mean", "count"])
    print(f"\n[pool] 題目 {len(qids_sorted)} 筆 → {out_q}", file=sys.stderr)
    print(f"[pool] 標籤 {len(df)} 列 → {out_l}", file=sys.stderr)
    print("[pool] 各模型答對率:", file=sys.stderr)
    print(summary.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
