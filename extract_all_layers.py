"""
全層 prefill 抽取：對每題存「所有層」在指定 token 位置的 hidden state。

與 extract_hidden_states.py 的差異：
  - 不只抓上半層，而是 embedding 之後的每一層都存（給之後逐層掃 AUC 用）
  - token 位置可選：
      last  = prompt 最後一個位置（模型讀完整題、要生成第 0 個 token 的那一刻）
              → 這是 NVIDIA 的 last-token prefill，也是預設、也是你該用的
      first = 序列第 0 個位置（只看得到第一個 token，幾乎不含題目資訊）
              → 留著讓你驗證它確實沒用；正式實驗不要用這個
  - 仍然會跑短生成來判斷「本模型自己」的對錯（closed-source target 的對錯
    由 run_closed_targets.py 另外收）

⚠️ 這支腳本沒有拿真實模型跑過完整流程過（開發環境的 GPU VRAM 太小跑不動 4B 模型），
   語法/計分邏輯確認過沒問題，但正式全量跑之前務必先 --limit 5 試跑。

用法：
  python extract_all_layers.py \
      --model Qwen/Qwen3.5-4B \
      --queries data/queries.jsonl \
      --out data/traces_Qwen3.5-4B.parquet \
      --answer-type numeric --max-new-tokens 8192 \
      --token-pos last --limit 30
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_choice_letter(text):
    """
    找獨立字母 A-J，但排除 "I"——它同時也是英文代名詞
    ("I'm not sure" 會被誤判成選了選項 I)。
    """
    if text is None:
        return None
    for m in re.finditer(r"\b([A-J])\b", text.strip()):
        if m.group(1) == "I":
            continue
        return m.group(1)
    return None


def extract_numeric_answer(text):
    """
    給 AIME 這類整數答案(0-999)用。優先順序：
      1. \\boxed{...} —— 數學競賽標準答案格式
      2. "answer is/answer:/final answer:" 後面接的數字
      3. 退而求其次：文字裡最後一個獨立整數
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


def score_answer(generated_text: str, ground_truth: str, answer_type: str = "letter") -> int:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--token-pos", choices=["last", "first"], default="last",
                     help="last=prompt 最後位置(正確選擇)；first=位置0(對照用，預期無訊號)")
    ap.add_argument("--skip-generate", action="store_true",
                     help="不跑生成、不判對錯(標籤改由 benchmark 的 labels.parquet 提供時用這個);"
                          "只做 prefill forward 存 hidden state,速度快非常多")
    ap.add_argument("--answer-type", choices=["letter", "numeric"], default="letter",
                     help="letter=A-J 選擇題(預設)；numeric=整數答案(例如 AIME)")
    ap.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True,
                     help="用 tokenizer 的 chat template 包裝 query 再餵給模型(預設開)。"
                          "instruct 模型沒套用這個通常不會照指令回答，只會亂接龍到"
                          "--max-new-tokens 上限。只有題目本身已經是完整格式化 prompt"
                          "(例如舊的 LLMRouterBench 流程)時才需要用 --no-apply-chat-template 關掉")
    ap.add_argument("--max-new-tokens", type=int, default=8,
                     help="letter 選擇題 8 個 token 就夠；numeric(長推理題)建議調到"
                          "1024-2048，不然模型還沒推到答案就被截斷")
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    if args.answer_type == "numeric" and args.max_new_tokens == 8:
        print(
            "[警告] --answer-type numeric 但 --max-new-tokens 還是預設值 8，"
            "長推理題大機率還沒推到答案就被截斷，建議加 --max-new-tokens 1024 以上",
            file=sys.stderr,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[警告] 無 CUDA，32B 在 CPU 上會極慢", file=sys.stderr)

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"[extract_all] 載入 {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device
    )
    model.eval()

    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    if args.limit is not None:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_rows = []

    def flush(rows):
        if not rows:
            return
        import pandas as pd
        df = pd.DataFrame(rows)
        if out_path.exists():
            df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
        df.to_parquet(out_path)
        print(f"[extract_all] 累計 {len(df)} 筆 → {out_path}", file=sys.stderr)

    n_total = len(queries)
    for i, item in enumerate(queries):
        t0 = time.time()
        print(f"[extract_all] [{i+1}/{n_total}] qid={item['qid']} 開始...", file=sys.stderr)
        try:
            if args.apply_chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["query"]}],
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt_text = item["query"]
            inputs = tokenizer(prompt_text, return_tensors="pt",
                               truncation=True, max_length=4096).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)

            attn_mask = inputs["attention_mask"][0]
            seq_len = int(attn_mask.sum().item())
            pos = (seq_len - 1) if args.token_pos == "last" else 0

            # hidden_states[0] 是 embedding 輸出，1..L 是各 transformer 層
            layer_vecs = [h[0, pos].float().cpu().numpy().tolist()
                          for h in out.hidden_states]

            if args.skip_generate:
                gen_text, correct, hit_max_new_tokens, n_new_tokens = "", None, None, None
            else:
                with torch.no_grad():
                    # 明確傳入 eos_token_id：這個模型的 config.json/generation_config.json
                    # 都沒有設定 eos_token_id，generate() 不知道該在 <|im_end|> 停下來，
                    # 沒這行的話每次都會硬跑到 max_new_tokens 上限，就算已經寫出答案也不停。
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                         eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.eos_token_id)
                n_new_tokens = gen.shape[1] - inputs["input_ids"].shape[1]
                # 沒有在自然的 EOS 前停下來，是被 max_new_tokens 硬切斷的——
                # 跟 run_s_inference.py 的 finish_reason=="length" 是同一件事，
                # 存下來才能分辨「真的推理失敗」還是「只是沒機會寫完」。
                hit_max_new_tokens = (n_new_tokens >= args.max_new_tokens)
                gen_text = tokenizer.decode(
                    gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                correct = score_answer(gen_text, item["ground_truth"], args.answer_type)

            buffer_rows.append({
                "qid": item["qid"],
                "model": args.model,
                "token_pos": args.token_pos,
                "seq_len": seq_len,
                "num_layers_incl_embed": len(layer_vecs),
                "correct": correct,
                "raw_generation": gen_text,
                "n_new_tokens": n_new_tokens,
                "hit_max_new_tokens": hit_max_new_tokens,
                "hidden_all_layers": layer_vecs,  # list[num_layers+1][hidden_dim]
            })
            print(
                f"[extract_all] [{i+1}/{n_total}] qid={item['qid']} 完成 "
                f"(耗時 {time.time()-t0:.1f}s, seq_len={seq_len}, correct={correct}, "
                f"新生成token數={n_new_tokens}, 是否被截斷={hit_max_new_tokens})",
                file=sys.stderr,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[extract_all] [{i+1}/{n_total}] qid={item['qid']} 失敗 "
                f"(耗時 {time.time()-t0:.1f}s)：{e}",
                file=sys.stderr,
            )
            continue

        if (i + 1) % args.flush_every == 0:
            flush(buffer_rows); buffer_rows = []
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    flush(buffer_rows)
    print(f"[extract_all] 完成 {len(queries)} 筆", file=sys.stderr)


if __name__ == "__main__":
    main()
