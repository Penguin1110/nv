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
      --bench-root <解壓後的 bench-release 目錄> \
      --dataset aime \
      --out-queries data/queries.jsonl \
      --out-labels data/labels.parquet \
      --limit 30

實測下來 LLMRouterBench 實際解壓出來的結構是 <bench-release>/<dataset>/..., 不是官方
README/download.md 寫的 results/bench/<dataset>——下面兩種都會嘗試,不需要自己搬檔案。
資料集底下有時還多一層 split 目錄(test/valid/hybrid/subset_500/test_1000/... 不等,
沒有固定命名),用 --split 指定;不指定的話,只有一種 split 會自動使用,超過一種要你自己選。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _has_json(d: Path) -> bool:
    return next(d.rglob("*.json"), None) is not None


def find_dataset_dir(bench_root: Path, dataset: str) -> Path:
    """
    嘗試兩種可能的目錄佈局:
      1. <bench_root>/results/bench/<dataset>  (README 寫的樣子)
      2. <bench_root>/<dataset>                (實測解壓後的真實樣子)
    """
    candidates = [
        bench_root / "results" / "bench" / dataset,
        bench_root / dataset,
    ]
    for c in candidates:
        if c.exists():
            return c

    available = sorted(
        d.name for base in (bench_root / "results" / "bench", bench_root)
        if base.exists() for d in base.iterdir() if d.is_dir()
    )
    raise FileNotFoundError(
        f"找不到 dataset={dataset!r}。試過的路徑:\n  " +
        "\n  ".join(str(c) for c in candidates) +
        (f"\n\n{bench_root} 底下實際有的目錄:\n  " + "\n  ".join(available) if available else "")
    )


def find_model_dirs(ds_dir: Path, requested_split: Optional[str]) -> List[Path]:
    """
    每個模型一個子目錄(目錄名 = 模型名,這就是「規定好的池」的權威名單)。
    有些資料集在模型目錄外還包一層 split 目錄,而且各資料集叫法不一致
    (test/valid/hybrid/subset_500/test_1000/test_3000/v1/verified/...),
    所以不用寫死白名單——直接偵測「這個子目錄底下有沒有 *.json」來判斷
    它本身是不是模型目錄,不是的話就當作 split 目錄往下一層找。
    """
    direct_children = sorted(d for d in ds_dir.iterdir() if d.is_dir())
    direct_model_dirs = [d for d in direct_children if _has_json(d)]

    # 全部子目錄都直接有 json → 沒有 split 這一層,直接當模型目錄用
    if direct_model_dirs and len(direct_model_dirs) == len(direct_children):
        return direct_model_dirs

    split_candidates = [d for d in direct_children if d not in direct_model_dirs]

    # 混合情況(例如 arenahard 底下同時有模型目錄和一個雜散的 test/ split 目錄):
    # 優先採用直接的模型目錄,雜散的 split 目錄只警告、不採用
    if direct_model_dirs:
        if split_candidates:
            print(
                f"[pool] 警告:{ds_dir.name} 底下混雜非模型子目錄，忽略：" +
                ", ".join(d.name for d in split_candidates),
                file=sys.stderr,
            )
        return direct_model_dirs

    if not split_candidates:
        return []

    if requested_split is not None:
        match = next((d for d in split_candidates if d.name == requested_split), None)
        if match is None:
            raise ValueError(
                f"--split {requested_split!r} 不存在。{ds_dir.name} 底下可用的 split："
                + ", ".join(d.name for d in split_candidates)
            )
        return sorted(d for d in match.iterdir() if d.is_dir())

    if len(split_candidates) == 1:
        return sorted(d for d in split_candidates[0].iterdir() if d.is_dir())

    raise ValueError(
        f"{ds_dir.name} 底下有多個可能的 split，請用 --split 指定其中一個："
        + ", ".join(d.name for d in split_candidates)
    )


import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="aime")
    ap.add_argument("--split", type=str, default=None,
                     help="資料集底下有多個 split 時指定其一(例如 test_1000)；"
                          "只有一種 split 時可省略")
    ap.add_argument("--out-queries", type=str, required=True)
    ap.add_argument("--out-labels", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="只取前 N 個 qid(依 index 排序後),pilot 用")
    args = ap.parse_args()

    ds_dir = find_dataset_dir(Path(args.bench_root), args.dataset)
    model_dirs = find_model_dirs(ds_dir, args.split)
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
