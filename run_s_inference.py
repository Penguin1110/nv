"""
呼叫 S（強模型，例如 Qwen3.5-27B）取得對錯標籤——透過 OpenRouter 的
OpenAI-相容 chat completions API。

跟 W 不同，這裡完全不需要 S 的 hidden state，只需要它「答對還是答錯」，
所以用 API 呼叫就夠，不需要本地載入 27B 的權重。

讀 prepare_pool_data.py 產生的 queries.jsonl（qid, query, ground_truth），
對每一題呼叫 S `--n-samples` 次（temperature > 0 讓每次採樣不同，才能像 W
一樣估計 pass@1），輸出 qid/sample_idx/model/correct 到一個 parquet 檔——
格式對應 core_prediction.py 讀 S 標籤時接受的「單一模型檔」格式。

可續跑：中斷後重跑，已經有結果的 (qid, sample_idx) 會自動跳過，不重打 API。

用法：
  cp .env.example .env   # 填入你自己的 OPENROUTER_API_KEY（.env 已加進 .gitignore）
  python run_s_inference.py \
      --queries data/queries.jsonl \
      --out data/s_correctness_qwen3.5-27b.parquet \
      --model qwen/qwen3.5-27b \
      --n-samples 8 \
      --limit 30

也可以不建 .env，直接用環境變數：export OPENROUTER_API_KEY=sk-or-...

⚠️ --model 的字串是 OpenRouter 型錄裡的 slug，可能隨時間變動，正式跑之前
   先去 https://openrouter.ai/models 搜尋 "qwen3.5-27b" 確認目前的正確名稱。
⚠️ 這支腳本本身沒有在這個沙箱環境被真的打過 API（沒有金鑰、也可能沒有對外
   網路），邏輯用 test_run_s_inference.py 的假 HTTP 回應測試過，但還沒有經過
   真實 OpenRouter 回應驗證——第一次正式跑之前，建議先用 --limit 2 --n-samples 1
   小規模跑一次，人工檢查 raw_generation 是否合理。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ if present; no-op (and harmless) if absent

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-27b"


def extract_choice_letter(text: str) -> Optional[str]:
    """
    跟 extract_all_layers.py 用同一套「找獨立字母 A-J」規則，但額外排除 "I"——
    因為 "I" 同時也是英文代名詞（"I'm not sure" 會被誤判成選了選項 I）。
    這個誤判在 extract_all_layers.py 裡目前也存在，尚未回報/修正那邊，這裡先
    在新程式碼中修掉，避免 S 的標籤被這個假陽性污染。
    """
    for m in re.finditer(r"\b([A-J])\b", text.strip()):
        if m.group(1) == "I":
            continue
        return m.group(1)
    return None


def score_answer(generated_text: str, ground_truth: str) -> int:
    pred = extract_choice_letter(generated_text)
    if pred is None:
        return 0
    gt = ground_truth.strip().upper()
    return int(pred == (gt[0] if gt else None))


def call_openrouter(
    api_key: str,
    model: str,
    query: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> str:
    """
    打一次 OpenRouter chat completions，回傳生成文字。
    對 429/5xx 做簡單的指數退避重試；其餘錯誤直接拋出。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"OpenRouter 回應沒有 choices：{data}")
            return choices[0]["message"]["content"]

        if resp.status_code in (429, 500, 502, 503, 504):
            last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(min(2 ** attempt, 30))
            continue

        # 其他錯誤（例如 401/404 模型名稱錯）直接拋出，不要浪費重試次數
        raise RuntimeError(f"OpenRouter 呼叫失敗 HTTP {resp.status_code}: {resp.text[:500]}")

    raise RuntimeError(f"重試 {max_retries} 次後仍失敗：{last_error}")


def load_existing_keys(out_path: Path) -> set:
    """讀已存在的輸出檔，回傳已經跑過的 (qid, sample_idx) 集合，供續跑跳過用。"""
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path)
    return set(zip(df["qid"], df["sample_idx"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help=f"OpenRouter 型錄裡的模型 slug（預設 {DEFAULT_MODEL}，正式跑前請去 "
                          f"https://openrouter.ai/models 核對）")
    ap.add_argument("--n-samples", type=int, default=8,
                     help="每題呼叫幾次（多重採樣，用來估計 S 的 pass@1）")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 環境變數 OPENROUTER_API_KEY 未設定", file=sys.stderr)
        sys.exit(1)

    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    if args.limit is not None:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing_keys(out_path)
    if done:
        print(f"[s_infer] 已有 {len(done)} 筆 (qid, sample_idx) 結果，將跳過重跑", file=sys.stderr)

    buffer_rows = []

    def flush(rows):
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        if out_path.exists():
            df_new = pd.concat([pd.read_parquet(out_path), df_new], ignore_index=True)
        df_new.to_parquet(out_path)
        print(f"[s_infer] 累計 {len(df_new)} 筆 → {out_path}", file=sys.stderr)

    total_calls = len(queries) * args.n_samples
    n_done = 0

    for item in queries:
        qid = item["qid"]
        for sample_idx in range(args.n_samples):
            n_done += 1
            if (qid, sample_idx) in done:
                continue
            try:
                gen_text = call_openrouter(
                    api_key, args.model, item["query"],
                    args.temperature, args.max_tokens, args.timeout, args.max_retries,
                )
                correct = score_answer(gen_text, item["ground_truth"])
                buffer_rows.append({
                    "qid": qid,
                    "sample_idx": sample_idx,
                    "model": args.model,
                    "correct": correct,
                    "raw_generation": gen_text,
                })
            except Exception as e:
                print(f"[s_infer] qid={qid} sample={sample_idx} 失敗：{e}", file=sys.stderr)
                continue

            if len(buffer_rows) >= args.flush_every:
                flush(buffer_rows)
                buffer_rows = []

            if n_done % 20 == 0:
                print(f"[s_infer] 進度 {n_done}/{total_calls}", file=sys.stderr)

    flush(buffer_rows)
    print(f"[s_infer] 完成 {len(queries)} 題 x {args.n_samples} 次採樣", file=sys.stderr)


if __name__ == "__main__":
    main()
