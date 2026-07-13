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
from typing import Optional, Tuple

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


def extract_choice_letter(text: Optional[str]) -> Optional[str]:
    """
    跟 extract_all_layers.py 用同一套「找獨立字母 A-J」規則(兩邊都修過)，但額外
    排除 "I"——因為 "I" 同時也是英文代名詞（"I'm not sure" 會被誤判成選了選項 I）。
    """
    if text is None:
        return None
    for m in re.finditer(r"\b([A-J])\b", text.strip()):
        if m.group(1) == "I":
            continue
        return m.group(1)
    return None


def extract_numeric_answer(text: Optional[str]) -> Optional[str]:
    """
    給 AIME 這類整數答案(0-999)用。優先順序：
      1. \\boxed{...} —— 數學競賽標準答案格式，跟資料集本身 solution 欄位寫法一致
      2. "answer is/answer:/final answer:" 後面接的數字
      3. 退而求其次：文字裡最後一個獨立整數(模型通常把最終答案放在推理過程最後)

    text 可能是 None(例如 API 回應被截斷、content 是空的)，直接視為抓不到答案。
    """
    if text is None:
        return None
    text = text.strip()

    m = re.search(r"\\boxed\{(-?\d+)\}", text)
    if m:
        return m.group(1)

    m = re.search(r"(?:answer is|answer:|final answer:?)\s*\$?(-?\d+)\$?", text, re.IGNORECASE)
    if m:
        return m.group(1)

    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None


def score_answer(generated_text: Optional[str], ground_truth: str, answer_type: str = "letter") -> int:
    if ground_truth is None:
        return 0

    if answer_type == "numeric":
        pred = extract_numeric_answer(generated_text)
        if pred is None:
            return 0
        try:
            return int(int(pred) == int(ground_truth.strip()))
        except ValueError:
            return 0

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
) -> Tuple[str, Optional[str]]:
    """
    打一次 OpenRouter chat completions，回傳 (生成文字, finish_reason)。
    finish_reason="length" 代表在 max_tokens 用完前被截斷，還沒真正結束推理——
    存下來讓你之後分析時能篩掉這種「沒有機會寫出答案」的資料列，跟真正推理
    失敗的資料分開看，不然難度/失敗訊號會被截斷雜訊污染。
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
            message = choices[0]["message"]
            finish_reason = choices[0].get("finish_reason")
            content = message.get("content")
            if not content:
                # 推理型模型有時把輸出放在 reasoning/reasoning_content，若還在思考中
                # 就被 max_tokens 截斷，content 會是 null 或空字串。退而求其次抓推理欄位，
                # 兩者都沒有的話明確報錯，不要讓 None 一路傳下去在別處炸出難懂的錯誤。
                content = message.get("reasoning_content") or message.get("reasoning")
            if not content:
                raise ValueError(
                    f"OpenRouter 回應沒有可用文字內容(content/reasoning 皆空，"
                    f"finish_reason={finish_reason!r})——可能是 max_tokens 太小，"
                    f"模型還在思考就被截斷了，試著調大 --max-tokens"
                )
            return content, finish_reason

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
    ap.add_argument("--answer-type", choices=["letter", "numeric"], default="letter",
                     help="letter=A-J 選擇題(預設)；numeric=整數答案(例如 AIME)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048,
                     help="AIME 這類長推理題容易需要較多 token 才能推到最終答案；"
                          "選擇題可以調小一點省成本")
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

    # 失敗記錄寫進獨立的 jsonl，跟主輸出檔同目錄同前綴——這樣長時間背景執行
    # (nohup/斷線)結束後,不用翻終端機捲軸也能知道哪些 (qid, sample_idx) 失敗、
    # 為什麼失敗,方便之後單獨重跑或人工檢查。
    errors_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")
    errors_file = open(errors_path, "a", encoding="utf-8")

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
    n_skipped = 0
    n_succeeded = 0
    n_failed = 0
    n_truncated = 0

    for item in queries:
        qid = item["qid"]
        for sample_idx in range(args.n_samples):
            n_done += 1
            if (qid, sample_idx) in done:
                n_skipped += 1
                continue
            t0 = time.time()
            print(
                f"[s_infer] [{n_done}/{total_calls}] qid={qid} sample={sample_idx} 開始...",
                file=sys.stderr,
            )
            try:
                gen_text, finish_reason = call_openrouter(
                    api_key, args.model, item["query"],
                    args.temperature, args.max_tokens, args.timeout, args.max_retries,
                )
                correct = score_answer(gen_text, item["ground_truth"], args.answer_type)
                buffer_rows.append({
                    "qid": qid,
                    "sample_idx": sample_idx,
                    "model": args.model,
                    "correct": correct,
                    "raw_generation": gen_text,
                    "finish_reason": finish_reason,
                })
                n_succeeded += 1
                if finish_reason == "length":
                    n_truncated += 1
                print(
                    f"[s_infer] [{n_done}/{total_calls}] qid={qid} sample={sample_idx} 完成 "
                    f"(耗時 {time.time()-t0:.1f}s, correct={correct}, finish_reason={finish_reason})",
                    file=sys.stderr,
                )
            except Exception as e:
                n_failed += 1
                print(
                    f"[s_infer] [{n_done}/{total_calls}] qid={qid} sample={sample_idx} 失敗 "
                    f"(耗時 {time.time()-t0:.1f}s)：{e}",
                    file=sys.stderr,
                )
                errors_file.write(json.dumps({
                    "qid": qid, "sample_idx": sample_idx, "error": str(e),
                }, ensure_ascii=False) + "\n")
                errors_file.flush()
                continue

            if len(buffer_rows) >= args.flush_every:
                flush(buffer_rows)
                buffer_rows = []

            if n_done % 20 == 0:
                print(
                    f"[s_infer] 進度 {n_done}/{total_calls} "
                    f"(成功 {n_succeeded}, 失敗 {n_failed}, 跳過 {n_skipped})",
                    file=sys.stderr,
                )

    flush(buffer_rows)
    errors_file.close()

    print(
        f"[s_infer] 完成 {len(queries)} 題 x {args.n_samples} 次採樣 —— "
        f"成功 {n_succeeded}, 失敗 {n_failed}, 跳過(已存在) {n_skipped}",
        file=sys.stderr,
    )
    if n_failed > 0:
        print(f"[s_infer] 失敗細節 → {errors_path}", file=sys.stderr)
    if n_truncated > 0:
        print(
            f"[s_infer] ⚠️  {n_truncated}/{n_succeeded} 筆成功結果的 finish_reason=length"
            f"（在 --max-tokens={args.max_tokens} 用完前被截斷，可能沒推到最終答案）——"
            f"分析前建議用 finish_reason 欄位篩掉這些列，或調大 --max-tokens 重跑",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
