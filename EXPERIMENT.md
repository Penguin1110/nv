# Shared Hard Basin 實驗說明

## 老實說：現在的完整度

**分析層的程式碼跟邏輯都就緒且測試過，但整條線還沒有拿真實資料完整跑過一次。**

- 分析層（`core_prediction.py` / `difficulty_baseline.py` / `precedence_check.py` /
  `run_mvp.py`）用 `mock_data.py` 的合成假資料測過，管線邏輯正確、三個控制組判斷邏輯
  也正確；但從沒接過真實模型輸出。
- Stage 1（題目）、Stage 3 的 S 標籤腳本（`run_s_inference.py`）都寫好且測試過（mock
  測試，不是真的打過 API），可以在有 GPU/API key 的機器上直接跑。
- Stage 2（W 抽 hidden state）程式碼已經改好支援數字答案，但**還沒有實際跑過**——你的
  遠端機器有 GPU（GeForce GTX 1650，4GB VRAM），但這張卡對 4B 級模型的半精度權重
  （~8GB）偏緊，很可能需要 4-bit 量化才跑得動，沒有實測過，不能保證一定成功。
- W 的多重採樣、逐步 logits 收集——**完全沒有程式碼**，是目前最大的缺口。

已經把最新的修正 push 上 GitHub（`Penguin1110/nv` master 分支），你在遠端機器上
`git pull` 就能拿到。

---

## 實驗要驗證什麼

**假說**：有些題目對所有模型都是「共享的難」（shared hard basin），不是單一模型的問題。
弱模型（W）早期的 hidden-state 軌跡，能不能預測「強模型（S）也會答錯」，而且這個預測力
**贏過單純的難度**（W 自己的 pass@1）？

如果贏得過，代表「難」這件事有跨模型共享的幾何結構，能拿來做模型升級路由的決策依據
（這題换更大的模型救不救得回，事先就能判斷）。如果贏不過，就只是在講「難題大家都難」，
沒有新資訊。

---

## 完整實驗的五個階段

```
Stage 1        Stage 2              Stage 3                          Stage 4              Stage 5
準備 AIME 題 → W 回答問題+記錄狀態 →  S 標籤(OpenRouter,已寫) +        →  統計分析(3個控制) →  Go/No-Go
                                     W 多重採樣/逐步logits(缺程式)
```

### Stage 1：準備題目

**程式**：`prepare_aime_data.py`（新寫，已用真實資料跑過）
**做什麼**：從 HuggingFace `AI-MO/aimo-validation-aime` 抓題目，只取 `problem`（題目文字）
和 `answer`（0-999 整數答案），不碰 `solution`（裡面有完整解法跟 `\boxed{答案}`，混進去
會洩漏答案）。
**輸出**：`queries.jsonl`（qid, query, ground_truth）—— 跟原本 `prepare_pool_data.py` 的
格式完全相容，後面的腳本不用改。
**現況**：✅ 已實測，90 題全部正確寫出。

```bash
python3 prepare_aime_data.py --out-queries data/queries.jsonl
```

> **為什麼放棄 LLMRouterBench**：原本想用它同時拿題目跟 S 的標籤，但實測解壓後的目錄
> 結構（`bench-release/<dataset>/...`）跟它自己的 README 寫的不一樣，且各資料集的
> split 命名毫無規律（test/valid/hybrid/subset_500/test_1000...），也沒有解決多重採樣
> 缺口。改用乾淨的公開 AIME dataset，題目來源單純、自己完全掌控。
>
> ⚠️ **只有 90 題**，比原始設計建議的「至少 200-500 題」少很多。先用來跑通整條管線；
> 如果 W✗ 的樣本數不夠（見下方 `min_w_wrong_samples` 門檻），需要疊加其他年份的 AIME
> 題目或換更大的資料集。

### Stage 2：弱模型（W）回答問題，同時記錄內部狀態

**程式**：`extract_all_layers.py`（既有腳本，這次修過）
**做什麼**：載入 W，對每一題：
1. 把完整 prompt 餵進去做一次 forward pass，抓「prompt 最後一個 token 位置」在**每一層**
   的 hidden state（模型讀完題目、還沒開始生成任何東西那一刻的內部狀態）
2. 做一次貪婪生成（`do_sample=False`），拿生成結果跟 ground_truth 比對算對錯
**輸出**：`traces_{model}.parquet`（qid, correct, hidden_all_layers 等欄）

**這次修的東西**：
- `extract_choice_letter()` 原本會把 "I'm not sure" 的 "I" 誤判成選了選項 I，已修掉
- 新增 `--answer-type numeric`：AIME 答案是整數，不是選擇題字母，**這次一定要加這個參數**
- 新增 `--max-new-tokens`：原本寫死 8 個 token（只夠選擇題答案），長推理題完全不夠，
  改成可調參數，AIME 建議 1024 以上（沒指定會印警告提醒你）

```bash
python3 extract_all_layers.py \
    --model <W模型的HF路徑> \
    --queries data/queries.jsonl \
    --out data/traces_<model>.parquet \
    --answer-type numeric \
    --max-new-tokens 1024
```

**現況**：邏輯改好、語法確認過，但**還沒有拿真實模型跑過**。你的 GPU（GTX 1650, 4GB）
對 4B 模型的半精度權重偏緊，建議先用 `--limit 5` 小規模試跑，觀察會不會 OOM；OOM 的話
需要加 4-bit 量化（`bitsandbytes`），目前腳本還沒支援，要跑不動再回來加。

### Stage 3：取得 S 的答案 + 目前還缺的部分

**S 的標籤**——`run_s_inference.py`（新寫，8→13 個 mock 測試通過，**沒打過真實 API**）
- 讀 `queries.jsonl`，對每一題呼叫 OpenRouter 的 S 模型 `--n-samples` 次
  （預設 8 次，temperature=0.7），把每次的對錯記下來
- 同樣加了 `--answer-type numeric`，AIME 一定要加
- 因為 S 只需要對錯、不需要 hidden state，用 API 呼叫就夠，不用本地跑大模型
- 可續跑：中斷重跑會跳過已經打過的 (qid, sample_idx)

```bash
cp .env.example .env   # 填入真實的 OPENROUTER_API_KEY
python3 run_s_inference.py \
    --queries data/queries.jsonl \
    --out data/s_correctness_<model>.parquet \
    --model <OpenRouter型錄裡的slug> \
    --answer-type numeric \
    --n-samples 8 \
    --limit 2   # 先小規模試跑，人工檢查 raw_generation 合理再拿掉這個參數跑全量
```

**W 的多重採樣**——**還沒有程式碼**。`extract_all_layers.py` 目前每題只生成一次
（貪婪解碼）。要估計 W 的 pass@1，需要同一題用 temperature > 0 跑很多次。沒有這個，
難度控制組（Stage 4 控制 #1）就沒有意義。→ 需要在生成迴圈外面再包一層採樣迴圈，
仿照 `run_s_inference.py` 的續跑/多樣本設計。

**W 的逐步 logits**——**還沒有程式碼**。要驗證「W 是不是在答案生成前就已經鎖死在錯誤
方向」，需要生成過程中每一步的 logits，`extract_all_layers.py` 目前只存最終生成文字。
→ 需要生成時用 `output_scores=True` 之類的方式，把每步 logits 存下來。

### Stage 4：統計分析（用假資料測試過）

**程式**：`core_prediction.py` → `difficulty_baseline.py` → `precedence_check.py`，由
`run_mvp.py` 串起來執行。完全不會去跑任何模型，純粹讀 parquet/jsonl 做統計。

**① 核心預測**（`core_prediction.py`）
- 只挑「W 答錯」的題目，把每一列 hidden state 取前 `prefix_fraction`（例如 25%）的層，
  算平均值/標準差/最大值/PCA 主成分當特徵
- 標籤：這題 S 是否也答錯（1 = 兩個都錯，是我們要找的「共享難」；0 = S 能救回來）
- 訓一個邏輯迴歸 probe 預測這個標籤，算 AUC
- train/test 按「題目」分組切（同一題的樣本不會同時出現在兩邊）——同一題的多個樣本
  共用同一個 S 標籤，列層級隨機切分會洩漏，AUC 會虛高（實測過：修正前 1.0，修正後
  降到合理的 0.5 附近）

**② 難度控制**（`difficulty_baseline.py`）——**整個實驗的命門**
- 算每題 W 的 pass@1，按難度分桶，在**同一個難度桶內**比較 probe AUC 跟純難度 baseline
  AUC 誰贏——贏了才算「幾何訊號有意義」，不然只是在測難度
- 需要 Stage 3 的 W 多重採樣資料才有意義，目前這塊還缺

**③ 洩漏控制**（`precedence_check.py`）
- 檢查「模型鎖死在錯誤方向」是否發生在「正確答案 token 變得有競爭力」之前
- 需要 Stage 3 的逐步 logits，目前沒有這份資料，程式會誠實回報 `insufficient_data`

### Stage 5：Go / No-Go 判斷

`run_mvp.py` 彙整①②③結果，寫進 `results/mvp_summary.json`：
- ① AUC ≤ 0.5 → NO-GO（沒訊號）
- ② probe 沒有顯著贏過難度 baseline → NO-GO（只是難度）
- ③ 顯示可疑（答案先於鎖死出現）→ CONDITIONAL（訊號可能不乾淨）
- 都過關 → GO，可以進到 Phase 2（跨家族模型）

---

## 在你的遠端機器上，照順序要做的事

```bash
cd ~/nv
git pull origin master        # 拿最新程式碼

pip install -r requirements.txt

# Stage 1
python3 prepare_aime_data.py --out-queries data/queries.jsonl

# Stage 3（S，先小規模試跑）
cp .env.example .env          # 填入真實 OPENROUTER_API_KEY（若還沒設過）
python3 run_s_inference.py --queries data/queries.jsonl \
    --out data/s_correctness_<S模型>.parquet \
    --model <OpenRouter slug> --answer-type numeric --limit 2

# Stage 2（W，先小規模試跑，注意 4GB VRAM 可能 OOM）
python3 extract_all_layers.py --model <W模型HF路徑> \
    --queries data/queries.jsonl \
    --out data/traces_<W模型>.parquet \
    --answer-type numeric --max-new-tokens 1024 --limit 5
```

小規模都正常後，拿掉 `--limit` 跑全量，再更新 `config_experiments.yaml` 裡的
`data.s_model_label` 對應到你實際用的 S slug，最後跑：

```bash
python3 run_mvp.py --config config_experiments.yaml --skip-precedence
```

（`--skip-precedence` 是因為 Stage 3 的逐步 logits 還沒有程式碼收集）

**還沒做、會擋住完整結果的事**：W 的多重採樣（難度 baseline 會不準）、W 的逐步 logits
（precedence check 完全跑不了）。

---

## 資料格式速查

| 檔案 | 欄位 | 來源 |
|---|---|---|
| `data/queries.jsonl` | `qid, query, ground_truth` | `prepare_aime_data.py` |
| `data/traces_{W模型名}.parquet` | `qid, correct, hidden_all_layers` | `extract_all_layers.py` |
| `data/s_correctness_{S模型名}.parquet` | `qid, sample_idx, model, correct` | `run_s_inference.py` |

`hidden_all_layers` 是巢狀 list `list[層數+1][hidden_dim]`（`.tolist()` 存出來的），
不是 numpy array——這點跟 `sweep_layers.py` 的解析方式一致。

`config_experiments.yaml` 裡的 `data.s_model_label` 要跟 `run_s_inference.py --model`
用的 slug 完全一致。`ground_truth` 現在是 AIME 的整數字串（例如 `"116"`），不是選擇題
字母——`--answer-type numeric` 就是為了對應這個。

---

## 檔案一覽

| 檔案 | 屬於哪個 Stage | 狀態 |
|---|---|---|
| `prepare_aime_data.py` | Stage 1 | ✅ 已實測（90 題） |
| `extract_all_layers.py` | Stage 2 | 邏輯修好(numeric 計分/可調 max-new-tokens/I bug)，未實測 |
| `sweep_layers.py` | 輔助分析(逐層掃 AUC) | 既有，未跑過 |
| `run_all_encoders.sh` | Stage 2 批次版 | 既有，未跑過 |
| `run_s_inference.py` | Stage 3（S 標籤，OpenRouter） | 13 個 mock 測試通過，**沒打過真實 API** |
| `test_run_s_inference.py` | 上面的單元測試 | 新增 |
| `.env` / `.env.example` | 放 `OPENROUTER_API_KEY`（`.env` 已 gitignore） | 新增 |
| `prepare_pool_data.py` | 目前不用(LLMRouterBench 已放棄) | 目錄偵測邏輯修過，留著備用 |
| （W 多重採樣/逐步 logits 腳本） | Stage 3（W） | **不存在，需另外寫** |
| `core_prediction.py` | Stage 4-① | 假資料測試過 |
| `difficulty_baseline.py` | Stage 4-② | 假資料測試過 |
| `precedence_check.py` | Stage 4-③ | 假資料測試過（真資料跑不了） |
| `run_mvp.py` | Stage 4+5 協調器 | 假資料測試過 |
| `mock_data.py` | 測試用合成資料產生器 | 新增 |
| `config_experiments.yaml` | 全域參數設定 | 新增 |
