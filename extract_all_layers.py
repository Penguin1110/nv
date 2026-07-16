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

可續跑（含 Ctrl+C）：已經在輸出檔裡的 qid 會自動跳過，不重跑；每完成一題就立刻
寫入磁碟（--flush-every 預設 1），單題常要好幾分鐘 GPU 時間，不會因為中斷而丟失。

--capture-decode-hidden 模式下單題可能要跑到上萬 token、好幾分鐘，如果 Ctrl+C
發生在單題生成「中途」，這一題目前為止已生成的 token/已取樣的 hidden state 會
存成 interrupted=True 的一列（不計入 correct），不會整題憑空消失；重跑同一個
指令時這種列會被清掉、那一題會整題重新跑（見 generate_with_decode_hidden 的
Ctrl+C 說明）。

⚠️ 已在真實 GPU 機器上驗證過能跑（Qwen3.5-4B），且需要 --apply-chat-template
   （預設開）才會照指令回答，不然會亂生成到 --max-new-tokens 上限。

用法：
  python extract_all_layers.py \
      --model Qwen/Qwen3.5-4B \
      --queries data/queries.jsonl \
      --out data/traces_Qwen3.5-4B.parquet \
      --answer-type numeric --max-new-tokens 8192 \
      --token-pos last --limit 30

decode-time 軌跡（每生成一個 token 就抓一次 hidden state，不是 prompt 的
prefill 位置）：加 --capture-decode-hidden。預設只存最後一層、只存前 200 筆
（--decode-stride 1、--max-decode-samples 200，等於只看生成開頭一小段）：
  python extract_all_layers.py \
      --model Qwen/Qwen3.5-4B \
      --queries data/queries.jsonl \
      --out data/traces_Qwen3.5-4B_decode.parquet \
      --answer-type numeric --max-new-tokens 8192 \
      --capture-decode-hidden --decode-layers -1 --max-decode-samples 200 \
      --limit 1   # 先跑 1 題驗證，再跑全部

想涵蓋「整個生成過程」（不只開頭）又要存全部層時，用 --decode-stride 拉大
取樣間隔取代硬性只存前 N 筆——例如 20000 token 上限、全部 33 層、每 100 步
取一次，每題約 68MB（33 層 x 200 筆 x hidden_dim x 4 bytes），90 題約 6GB，
比每步都存、只存前 200 步(只看得到開頭)划算得多：
  python extract_all_layers.py \
      --model Qwen/Qwen3.5-4B \
      --queries data/queries.jsonl \
      --out data/traces_Qwen3.5-4B_decode.parquet \
      --answer-type numeric --max-new-tokens 20000 \
      --capture-decode-hidden --decode-layers all \
      --decode-stride 100 --max-decode-samples 0 \
      --limit 1   # 先跑 1 題驗證，再跑全部
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


def generate_with_decode_hidden(model, tokenizer, inputs, max_new_tokens, decode_layer_spec,
                                 decode_stride=1, max_decode_samples=None):
    """
    手動逐步 greedy 解碼，只在「取樣到」的步數才打開 output_hidden_states。

    為什麼不直接用 model.generate(output_hidden_states=True)：transformers
    會把「整個生成過程」每一步的 hidden state 都留在記憶體裡，即使之後只想存
    一部分，它還是得先把全部步數都攢住——AIME 這種長推理題常生成到上萬
    token、又想存全部層時，這樣做記憶體/硬碟都會爆。逐步手動跑可以只在
    取樣到的步數才打開 output_hidden_states，其餘步數退回普通生成。

    decode_stride：每隔幾步取樣一次(1=每步都存)。想涵蓋「整個生成過程」又要
    存全部層時，把這個調大(例如 50)取代硬性只存前 N 步——這樣才能同時看到
    推理早期、中期、晚期的軌跡，而不是只看得到開頭一小段。
    max_decode_samples：取樣後最多存幾筆(不是原始 step 數，是取樣完的筆數)；
    超過後仍會繼續生成、繼續判對錯，只是不再存 hidden state。None/不設代表
    取樣涵蓋整個生成過程，不論生成多長。

    回傳的 decode_step_indices 記錄每一筆取樣「對應到生成的第幾個 token」——
    downstream 分析要算「相對生成進度」時，必須用這個而不是取樣筆數本身
    (取樣筆數如果被 max_decode_samples 提前截斷，筆數跟實際生成進度就不成比例)。

    代價：prompt 部分等於在這裡多跑一次 forward pass(第 0 步用完整
    input_ids、use_cache=True 建立 KV cache)，跟檔案裡另一段專門存 prompt
    position 的 prefill forward(use_cache=False)是各自獨立的兩次計算，
    多花一點 GPU 時間換取兩段邏輯互不耦合、比較不容易出錯。

    Ctrl+C 安全：單題現在可能要生成上萬 token、跑好幾分鐘，如果中斷發生在
    generate() 中間，外層的 --flush-every 只保得住「已經完整跑完的前幾題」，
    這一題已經花掉的 GPU 時間會整個不見。這裡在迴圈內接住 KeyboardInterrupt，
    把目前為止已經生成的 token/已經取樣的 hidden state 都回傳(用
    was_interrupted=True 標記)，呼叫端存成這一題的「不完整」列，而不是整個丟掉。
    """
    eos_id = tokenizer.eos_token_id
    cur_ids = inputs["input_ids"]
    cur_mask = inputs["attention_mask"]
    past_key_values = None

    generated_ids = []
    decode_hidden_by_step = []
    decode_step_indices = []
    layer_idxs = None
    n_samples = 0
    was_interrupted = False

    try:
        for step in range(max_new_tokens):
            is_stride_hit = (step % decode_stride == 0)
            under_cap = (max_decode_samples is None) or (n_samples < max_decode_samples)
            want_hidden = is_stride_hit and under_cap
            with torch.no_grad():
                out = model(
                    input_ids=cur_ids,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=want_hidden,
                )
            past_key_values = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1)

            if want_hidden:
                if layer_idxs is None:
                    n_layers_total = len(out.hidden_states)
                    layer_idxs = (
                        list(range(n_layers_total)) if decode_layer_spec == "all"
                        else [(n_layers_total + l if l < 0 else l) for l in decode_layer_spec]
                    )
                decode_hidden_by_step.append([
                    out.hidden_states[l][0, -1].float().cpu().numpy().tolist() for l in layer_idxs
                ])
                decode_step_indices.append(step)
                n_samples += 1

            token_id = next_id.item()
            generated_ids.append(token_id)
            if token_id == eos_id:
                break

            cur_ids = next_id.unsqueeze(0)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1
            )
    except KeyboardInterrupt:
        was_interrupted = True

    return generated_ids, decode_hidden_by_step, decode_step_indices, (layer_idxs or []), was_interrupted


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
                     help="last=prompt 最後位置(正確選擇)；first=位置0(對照用，預期無訊號)。"
                          "有指定 --token-positions 時這個參數會被忽略")
    ap.add_argument("--token-positions", type=str, default=None,
                     help="逗號分隔的固定絕對 token 位置(例如 '10,20,30,50,75')，或傳 'all' "
                          "存每一個位置(完整時序，之後要切哪個 fraction/位置都行，不用重跑"
                          "GPU；90 題大約多用 ~5GB 硬碟，Qwen3.5-4B 的 hidden_size=2560)。"
                          "指定這個會覆蓋 --token-pos，改成多位置模式。題目太短、某個位置"
                          "超出 seq_len 範圍時會直接跳過(不同題目實際存到的位置數可能不"
                          "一樣，要用 available_token_positions 欄位對齊，不要假設每列"
                          "位置數相同)")
    ap.add_argument("--skip-generate", action="store_true",
                     help="不跑生成、不判對錯(標籤改由 benchmark 的 labels.parquet 提供時用這個);"
                          "只做 prefill forward 存 hidden state,速度快非常多")
    ap.add_argument("--capture-decode-hidden", action="store_true",
                     help="除了 prompt 的 prefill hidden state 之外，額外抓「生成階段每吐"
                          "一個 token」的 hidden state(decode-time 軌跡，跟 prompt 的"
                          "prefill 軌跡是不同的一段序列)。跟 --skip-generate 不相容")
    ap.add_argument("--decode-layers", type=str, default="-1",
                     help="decode 階段要存哪幾層(逗號分隔，例如 '-1' 或 '0,16,-1')，或傳"
                          "'all' 存全部層。預設只存最後一層——長推理題可能生成到上萬"
                          "token，存全部 33 層會讓檔案暴增，搭配 --decode-stride 拉大"
                          "取樣間隔才控制得住檔案大小")
    ap.add_argument("--decode-stride", type=int, default=1,
                     help="每隔幾個生成 token 取樣一次 decode hidden state(預設 1=每個"
                          "都存)。想涵蓋『整個生成過程』又要存全部層時，把這個調大"
                          "(例如 50 或 100)取代只存前 N 步——用稀疏取樣涵蓋全長，"
                          "才看得到推理早期/中期/晚期，而不是只看得到開頭一小段")
    ap.add_argument("--max-decode-samples", type=int, default=200,
                     help="取樣後最多存幾筆(是取樣完的筆數，不是原始 step 數);超過這個"
                          "數量後仍會繼續生成、繼續判對錯，只是不再存 hidden state。傳 0 "
                          "或負數關閉上限，取樣涵蓋整個生成過程不論多長(存全部層時務必"
                          "配大一點的 --decode-stride，不然檔案會爆——例如 20000 token、"
                          "全部 33 層、stride=100 大約每題 68MB；stride=1 會是這個的 100 倍)")
    ap.add_argument("--answer-type", choices=["letter", "numeric"], default="letter",
                     help="letter=A-J 選擇題(預設)；numeric=整數答案(例如 AIME)")
    ap.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True,
                     help="用 tokenizer 的 chat template 包裝 query 再餵給模型(預設開)。"
                          "instruct 模型沒套用這個通常不會照指令回答，只會亂接龍到"
                          "--max-new-tokens 上限。只有題目本身已經是完整格式化 prompt"
                          "(例如舊的 LLMRouterBench 流程)時才需要用 --no-apply-chat-template 關掉")
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
                     help="思考型模型(chat template 會自動接 <think>)是否允許長鏈推理"
                          "(預設開)。實測 AIME 難題常常推理到 max_new_tokens 都收斂不了；"
                          "用 --no-enable-thinking 會強制塞一個已關閉的空 <think></think>，"
                          "逼模型跳過推理直接回答，速度快很多但正確率可能下降。"
                          "只有 --apply-chat-template 開著、且模型的 chat template 支援"
                          "這個參數時才有作用")
    ap.add_argument("--max-new-tokens", type=int, default=8,
                     help="letter 選擇題 8 個 token 就夠；numeric(長推理題)建議調到"
                          "1024-2048，不然模型還沒推到答案就被截斷")
    ap.add_argument("--flush-every", type=int, default=1,
                     help="每完成幾題就寫入一次磁碟(預設每題都寫)。單題常要幾分鐘，"
                          "調大這個值中斷時會多丟掉最多這個數字的結果(而且是已經花"
                          "GPU 時間算出來的)——沒有特別理由不建議調大")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids-file", type=str, default=None,
                     help="只跑這個檔案裡列出的 qid(一行一個)，其餘題目直接略過。"
                          "適合分機器時要跑「特定一批題目」而不是均分(例如把 W✗S✗ 跟 "
                          "W✗S✓ 兩組分給不同機器)；跟 --num-shards/--shard-index 是"
                          "不同的篩選方式，兩者可以疊加(先照 qid 篩，再切 shard)")
    ap.add_argument("--num-shards", type=int, default=1,
                     help="要切成幾份分給幾台機器平行跑(每台各跑各的 GPU，互不衝突)。"
                          "搭配 --shard-index 使用；每台機器用同一個 --queries、"
                          "同樣的 --num-shards，只有 --shard-index 跟 --out 不同，"
                          "事後再把幾個 --out 檔案 concat 起來合併")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="這台機器要跑第幾份(0-indexed，範圍 0..num-shards-1)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    if args.num_shards > 1 and not (0 <= args.shard_index < args.num_shards):
        print(f"[錯誤] --shard-index 必須在 0..{args.num_shards - 1} 範圍內", file=sys.stderr)
        sys.exit(1)

    if args.answer_type == "numeric" and args.max_new_tokens == 8:
        print(
            "[警告] --answer-type numeric 但 --max-new-tokens 還是預設值 8，"
            "長推理題大機率還沒推到答案就被截斷，建議加 --max-new-tokens 1024 以上",
            file=sys.stderr,
        )

    if args.skip_generate and args.capture_decode_hidden:
        print("[錯誤] --capture-decode-hidden 需要實際生成文字，不能跟 --skip-generate 一起用", file=sys.stderr)
        sys.exit(1)

    decode_layer_spec = None
    max_decode_samples = None
    if args.capture_decode_hidden:
        if args.decode_stride < 1:
            print("[錯誤] --decode-stride 至少要是 1", file=sys.stderr)
            sys.exit(1)
        decode_layer_spec = (
            "all" if args.decode_layers.strip().lower() == "all"
            else [int(x) for x in args.decode_layers.split(",")]
        )
        max_decode_samples = args.max_decode_samples if args.max_decode_samples > 0 else None
        print(
            f"[extract_all] decode-time hidden state 模式：層={decode_layer_spec}，"
            f"每 {args.decode_stride} 步取樣一次，最多"
            f"{max_decode_samples if max_decode_samples is not None else '(不限，注意檔案大小)'} 筆",
            file=sys.stderr,
        )

    token_positions = None
    extract_all_positions = False
    if args.token_positions:
        if args.token_positions.strip().lower() == "all":
            extract_all_positions = True
            print("[extract_all] 多位置模式：存每一題完整序列的每一個位置", file=sys.stderr)
        else:
            token_positions = sorted(int(x) for x in args.token_positions.split(","))
            print(f"[extract_all] 多位置模式，固定位置：{token_positions}", file=sys.stderr)

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

    if args.qids_file:
        with open(args.qids_file, "r", encoding="utf-8") as f:
            wanted_qids = {line.strip() for line in f if line.strip()}
        n_before_qids = len(queries)
        queries = [q for q in queries if q["qid"] in wanted_qids]
        print(
            f"[extract_all] --qids-file 篩選：{args.queries} 的 {n_before_qids} 題中"
            f"符合的有 {len(queries)} 題(qids-file 裡列了 {len(wanted_qids)} 個)",
            file=sys.stderr,
        )

    if args.num_shards > 1:
        queries = [q for idx, q in enumerate(queries) if idx % args.num_shards == args.shard_index]
        print(
            f"[extract_all] shard {args.shard_index}/{args.num_shards}：分到 {len(queries)} 題",
            file=sys.stderr,
        )

    if args.limit is not None:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 可續跑：已經在輸出檔裡的 qid 直接跳過，不重跑(單題常要好幾分鐘 GPU 時間，
    # 中斷後重打全部太浪費)。
    if out_path.exists():
        import pandas as pd
        try:
            light = pd.read_parquet(out_path, columns=["qid", "interrupted"])
            has_interrupted_col = True
        except (KeyError, ValueError):
            light = pd.read_parquet(out_path, columns=["qid"])
            has_interrupted_col = False

        if has_interrupted_col and light["interrupted"].fillna(False).any():
            # 上次是 Ctrl+C 中斷時留下的不完整列，要從檔案裡清掉再重新跑這幾題，
            # 不然這題會同時有一列不完整的舊資料、一列完整的新資料，qid 重複。
            n_interrupted = int(light["interrupted"].fillna(False).sum())
            full = pd.read_parquet(out_path)
            full = full[~full["interrupted"].fillna(False)]
            full.to_parquet(out_path)
            print(
                f"[extract_all] 清掉 {n_interrupted} 筆先前中斷時留下的不完整結果，"
                f"這些題目會重新跑",
                file=sys.stderr,
            )
            done_qids = set(full["qid"])
        else:
            done_qids = set(light["qid"])

        n_before = len(queries)
        queries = [q for q in queries if q["qid"] not in done_qids]
        if n_before != len(queries):
            print(
                f"[extract_all] 已有 {n_before - len(queries)} 筆結果，將跳過重跑",
                file=sys.stderr,
            )

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
    try:
        for i, item in enumerate(queries):
            t0 = time.time()
            print(f"[extract_all] [{i+1}/{n_total}] qid={item['qid']} 開始...", file=sys.stderr)
            try:
                if args.apply_chat_template:
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["query"]}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                else:
                    prompt_text = item["query"]
                inputs = tokenizer(prompt_text, return_tensors="pt",
                                   truncation=True, max_length=4096).to(device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, use_cache=False)

                attn_mask = inputs["attention_mask"][0]
                seq_len = int(attn_mask.sum().item())

                # hidden_states[0] 是 embedding 輸出，1..L 是各 transformer 層。
                # forward pass 本身就已經算出每一個 token 位置的 hidden state，
                # 這裡只是決定要「存」哪幾個位置，不需要為了多位置額外多跑一次模型。
                if extract_all_positions:
                    available_positions = list(range(seq_len))
                    hidden_by_position = [
                        [h[0, p].float().cpu().numpy().tolist() for h in out.hidden_states]
                        for p in available_positions
                    ]
                    layer_vecs = None
                elif token_positions is not None:
                    available_positions = [p for p in token_positions if p < seq_len]
                    hidden_by_position = [
                        [h[0, p].float().cpu().numpy().tolist() for h in out.hidden_states]
                        for p in available_positions
                    ]
                    layer_vecs = None  # 多位置模式下不存單一位置版本，避免混淆
                else:
                    pos = (seq_len - 1) if args.token_pos == "last" else 0
                    layer_vecs = [h[0, pos].float().cpu().numpy().tolist()
                                  for h in out.hidden_states]
                    available_positions = None
                    hidden_by_position = None

                decode_hidden_by_step = decode_step_indices = decode_layers_captured = generated_ids = None
                was_interrupted = False
                if args.skip_generate:
                    gen_text, correct, hit_max_new_tokens, n_new_tokens = "", None, None, None
                elif args.capture_decode_hidden:
                    generated_ids, decode_hidden_by_step, decode_step_indices, decode_layers_captured, was_interrupted = (
                        generate_with_decode_hidden(
                            model, tokenizer, inputs, args.max_new_tokens, decode_layer_spec,
                            args.decode_stride, max_decode_samples,
                        )
                    )
                    n_new_tokens = len(generated_ids)
                    hit_max_new_tokens = (n_new_tokens >= args.max_new_tokens) and not was_interrupted
                    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    # 被 Ctrl+C 中斷的話文字/hidden state 都只跑到一半，不能拿去判對錯，
                    # 用 None 標記(跟 --skip-generate 的「還沒判」共用同一個語意)，
                    # row["interrupted"]=True 會讓下次重跑時不把這題當成已完成而跳過。
                    correct = None if was_interrupted else score_answer(gen_text, item["ground_truth"], args.answer_type)
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

                row = {
                    "qid": item["qid"],
                    "model": args.model,
                    "seq_len": seq_len,
                    "num_layers_incl_embed": len(out.hidden_states),
                    "correct": correct,
                    "raw_generation": gen_text,
                    "n_new_tokens": n_new_tokens,
                    "hit_max_new_tokens": hit_max_new_tokens,
                    # True 代表這一列是 Ctrl+C 中斷時的部分結果(只有 decode-hidden 模式
                    # 會發生)，不是真的跑完；重跑同一個指令時這題會被視為未完成、重新跑，
                    # 不會被 resume 邏輯誤判成已經做完而跳過。
                    "interrupted": was_interrupted,
                }
                if extract_all_positions or token_positions is not None:
                    if not available_positions:
                        print(
                            f"[extract_all] 警告：qid={item['qid']} 的 seq_len={seq_len} "
                            f"比所有指定位置都短，這題沒有任何位置的 hidden state 可存",
                            file=sys.stderr,
                        )
                    row["requested_token_positions"] = "all" if extract_all_positions else token_positions
                    row["available_token_positions"] = available_positions
                    # list[num_available_positions][num_layers+1][hidden_dim]；
                    # 不同題目 num_available_positions 可能不同(見上面警告)，
                    # 分析時要用 available_token_positions 對齊，不要假設固定長度。
                    row["hidden_states_by_position"] = hidden_by_position
                else:
                    row["token_pos"] = args.token_pos
                    row["hidden_all_layers"] = layer_vecs  # list[num_layers+1][hidden_dim]
                if args.capture_decode_hidden:
                    row["decode_hidden_states_by_step"] = decode_hidden_by_step
                    # 每一筆對應到生成的第幾個 token(0-indexed)——因為有 --decode-stride
                    # 取樣間隔、又可能被 --max-decode-samples 提前截斷，筆數本身不能
                    # 代表「相對生成進度」，分析時要用這欄位除以 n_new_tokens 換算。
                    row["decode_step_indices"] = decode_step_indices
                    row["decode_layers_captured"] = decode_layers_captured
                    row["decode_stride"] = args.decode_stride
                    row["n_decode_samples_captured"] = len(decode_hidden_by_step)
                    row["generated_token_ids"] = generated_ids
                buffer_rows.append(row)

                if was_interrupted:
                    flush(buffer_rows)
                    print(
                        f"\n[extract_all] 使用者中斷(Ctrl+C)，發生在 qid={item['qid']} 生成到一半"
                        f"(已生成 {n_new_tokens} token、取樣 {len(decode_hidden_by_step or [])} 筆"
                        f"hidden state)——這一題的部分結果已存成 interrupted=True，"
                        f"重跑同一個指令會把這題當成未完成、重新從頭跑這一題",
                        file=sys.stderr,
                    )
                    sys.exit(130)

                decode_note = (
                    f", decode取樣筆數={len(decode_hidden_by_step)}" if args.capture_decode_hidden else ""
                )
                print(
                    f"[extract_all] [{i+1}/{n_total}] qid={item['qid']} 完成 "
                    f"(耗時 {time.time()-t0:.1f}s, seq_len={seq_len}, correct={correct}, "
                    f"新生成token數={n_new_tokens}, 是否被截斷={hit_max_new_tokens}{decode_note})",
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
    except KeyboardInterrupt:
        flush(buffer_rows)
        print(
            f"\n[extract_all] 使用者中斷(Ctrl+C)——已寫入的結果都在 {out_path}，"
            f"重跑同一個指令會自動跳過已完成的 qid、從這裡繼續",
            file=sys.stderr,
        )
        sys.exit(130)

    flush(buffer_rows)
    print(f"[extract_all] 完成 {len(queries)} 筆", file=sys.stderr)


if __name__ == "__main__":
    main()
