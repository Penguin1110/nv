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

⚠️ 本沙箱無 GPU/torch，此腳本未實際執行過，先 --limit 5 試跑。

用法：
  python extract_all_layers.py \
      --model Qwen/Qwen3-4B-Instruct \
      --queries data/queries.jsonl \
      --out data/all_layers_4b.parquet \
      --token-pos last --limit 30
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_choice_letter(text: str):
    m = re.search(r"\b([A-J])\b", text.strip())
    return m.group(1) if m else None


def score_answer(generated_text: str, ground_truth: str) -> int:
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
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

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

    for i, item in enumerate(queries):
        try:
            inputs = tokenizer(item["query"], return_tensors="pt",
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
                gen_text, correct = "", None
            else:
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(
                    gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                correct = score_answer(gen_text, item["ground_truth"])

            buffer_rows.append({
                "qid": item["qid"],
                "model": args.model,
                "token_pos": args.token_pos,
                "seq_len": seq_len,
                "num_layers_incl_embed": len(layer_vecs),
                "correct": correct,
                "raw_generation": gen_text,
                "hidden_all_layers": layer_vecs,  # list[num_layers+1][hidden_dim]
            })
        except Exception as e:  # noqa: BLE001
            print(f"[extract_all] 第 {i} 筆失敗：{e}", file=sys.stderr)
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
