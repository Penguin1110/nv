"""
Stage 1（改用公開 HF dataset 版本）：抓 AIME 題目，產出跟 prepare_pool_data.py
完全相同格式的 queries.jsonl（qid, query, ground_truth），讓後面的
run_s_inference.py / extract_all_layers.py / run_mvp.py 完全不用改。

資料來源：AI-MO/aimo-validation-aime（已用 huggingface_hub 直接確認過 schema：
id, problem, solution, answer, url ——只取 problem 當題目、answer 當標準答案，
不碰 solution，因為裡面完整寫了推導過程跟 \\boxed{答案}，混進 query 會洩漏答案）。

⚠️ 只有 90 題，比原始設計文件建議的「至少 200-500 題」少很多，用來估計難度分佈
   會偏粗——先用來跑通整條管線，數量不夠時再考慮疊加其他年份/來源的 AIME 題目。

⚠️ answer 欄位是 0-999 的整數字串，不是選擇題字母——run_s_inference.py 和
   extract_all_layers.py 都要加 --answer-type numeric 才會正確計分。

用法：
  python prepare_aime_data.py --out-queries data/queries.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "AI-MO/aimo-validation-aime"
DATASET_FILE = "data/train-00000-of-00001.parquet"

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-queries", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="只取前 N 題（依 id 排序後），pilot 用")
    args = ap.parse_args()

    print(f"[aime] 下載 {DATASET_REPO} ...", file=sys.stderr)
    path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE)
    df = pd.read_parquet(path)

    required_cols = {"id", "problem", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"資料集欄位跟預期不符，缺少 {missing}。實際欄位：{list(df.columns)}"
        )

    df = df.sort_values("id")
    if args.limit is not None:
        df = df.head(args.limit)

    out_q = Path(args.out_queries)
    out_q.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_q, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            answer = str(row["answer"]).strip()
            try:
                int(answer)
            except ValueError:
                print(f"[aime] 跳過 id={row['id']}：answer={answer!r} 不是整數", file=sys.stderr)
                continue
            f.write(json.dumps({
                "qid": f"aime_{row['id']}",
                "query": row["problem"],
                "ground_truth": answer,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[aime] 題目 {n_written} 筆 → {out_q}", file=sys.stderr)
    if n_written < 50:
        print(
            f"[aime] 警告：只有 {n_written} 題，低於 config_experiments.yaml 的 "
            f"min_w_wrong_samples 預設門檻(50)，run_mvp.py 可能會直接中止",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
