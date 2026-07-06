# Shared Hard Basin 實驗說明

## 老實說：現在的完整度

**還沒有一條從「模型回答問題」到「分析結論」都真的跑過真實資料的完整實驗。**

- 這個開發環境**沒有 GPU**（`torch.cuda.is_available()` = False），所以「載入 Qwen3.5-4B/27B
  真的跑題目」這件事，從頭到尾都無法在這裡執行、也從沒執行過。
- 分析層（`core_prediction.py` / `difficulty_baseline.py` / `precedence_check.py` /
  `run_mvp.py`）我只用 `mock_data.py` 產生的**合成假資料**測試過管線邏輯本身不會崩潰、
  三個控制組的判斷邏輯正確；從沒有接過真實模型輸出。
- 資料收集層（讓模型回答問題、記錄內部狀態）你原本就有 `extract_all_layers.py` /
  `prepare_pool_data.py`，但這兩支目前只能做到「單次貪婪生成 + 存 prefill hidden
  state」，缺兩塊本實驗要用到的東西（下面 Stage 3 詳述），需要另外寫程式碼補上，
  目前**還沒有寫**。

下面用「實驗流程」的角度，把整條線該有的五個階段、每階段現況、每階段程式邏輯講清楚。

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
Stage 1     Stage 2              Stage 3                          Stage 4              Stage 5
準備題目 →  W 回答問題+記錄狀態 →  S 標籤(OpenRouter,已寫) +        →  統計分析(3個控制) →  Go/No-Go
                                  W 多重採樣/逐步logits(缺程式)
```

### Stage 1：準備題目

**程式**：`prepare_pool_data.py`（既有，未修改）
**做什麼**：從 LLMRouterBench 的 benchmark 下載包裡，讀出每一題的完整 prompt 和標準答案。
**輸出**：`queries.jsonl`（qid, query, ground_truth）。
**現況**：程式存在，邏輯沒問題（用 `prompt` 而非 `origin_query`，確保跟後面 hidden state
對應的是同一份輸入）。**這次沒有實際跑過**——需要你手上有 LLMRouterBench 的下載包。

> **這裡有一個架構決定**：LLMRouterBench 的池子裡只有小模型，沒有 Qwen3.5-27B 這種大模型
> 的結果，所以 `prepare_pool_data.py` 原本產出的 `labels.parquet`（池內每個模型的對錯）
> **不會拿來當 S 的標籤**，只取它的 `queries.jsonl` 部分（題目本身）。S 的標籤改由
> Stage 3 的 `run_s_inference.py` 自己產生。

### Stage 2：弱模型（W）回答問題，同時記錄內部狀態

**程式**：`extract_all_layers.py`（既有，未修改）
**做什麼**：載入 W（例如 Qwen3.5-4B），對每一題：
1. 把完整 prompt 餵進去做一次 forward pass，抓「prompt 最後一個 token 位置」在**每一層**
   的 hidden state（這是模型讀完題目、還沒開始生成任何東西那一刻的內部狀態）
2. 接著做一次貪婪生成（`do_sample=False`），拿生成結果跟 ground_truth 比對算對錯
**輸出**：`traces_{model}.parquet`（qid, correct, hidden_all_layers 等欄）
**現況**：程式存在，邏輯沒問題。**這次沒有實際跑過**（沒有 GPU，4B 模型在 CPU 上跑不動）。

> **發現一個既有 bug**：判斷對錯用的 `extract_choice_letter()` 正規表達式
> `\b([A-J])\b` 會把英文代名詞 "I"（例如生成文字裡的 "I'm not sure"）誤判成選了選項
> I，可能讓 W 的對錯標籤被污染。這個 bug 存在於這支既有程式裡，這次沒有動它（不在授權
> 範圍內），只在我自己新寫的 `run_s_inference.py` 裡先修掉了。你之後跑 W 之前，建議在
> `extract_all_layers.py` 裡也套用同樣的排除邏輯。

### Stage 3：取得 S 的答案（新寫的程式）+ 目前還缺的部分

**S 的標籤**——`run_s_inference.py`（新增，已用 mock HTTP 回應測試過 8 個案例，
**但沒有打過真實 OpenRouter API**）
- 讀 `queries.jsonl`，對每一題呼叫 OpenRouter 的 Qwen3.5-27B（`qwen/qwen3.5-27b`）
  `--n-samples` 次（預設 8 次，temperature=0.7），把每次的對錯記下來
- 因為 S 在 Experiment A 只需要對錯、不需要 hidden state，用 API 呼叫就夠，不需要本地
  跑 32B 的權重
- 可續跑：中斷重跑會跳過已經打過的 (qid, sample_idx)
- **正式跑之前**：先去 https://openrouter.ai/models 搜尋 "qwen3.5-27b" 確認 model slug
  沒變，並用 `--limit 2 --n-samples 1` 小規模跑一次、人工檢查 `raw_generation` 合理再
  跑全量

**W 的多重採樣**——**還沒有程式碼**
- `extract_all_layers.py` 目前每題只生成一次（貪婪解碼）。要估計「這題 W 答對的機率」
  （pass@1），需要同一題用 temperature > 0 跑很多次，記錄每次的對錯。
  沒有這個，pass@1 只會是 0 或 1，難度控制組（Stage 4 的控制 #1）就沒有意義。
  → 需要在 `extract_all_layers.py` 的生成迴圈外面再包一層採樣迴圈，仿照
  `run_s_inference.py` 的續跑/多樣本設計。

**W 的逐步 logits**——**還沒有程式碼**
- 要驗證「W 是不是在答案生成前就已經鎖死在錯誤方向」，需要生成過程中**每一步**的
  logits（`[num_steps, vocab_size]`），以及正確答案對應的 token id。
  `extract_all_layers.py` 目前只存最終生成文字，完全沒有記錄這個。
  → 需要生成時用 `output_scores=True` 之類的方式，把每步 logits 存下來。

### Stage 4：統計分析（本次新增，用假資料測試過）

**程式**：`core_prediction.py` → `difficulty_baseline.py` → `precedence_check.py`，由
`run_mvp.py` 串起來執行。

這一步的**輸入**是 Stage 1+2（+3）產出的三個檔案，**完全不會去跑任何模型**，純粹是讀
parquet/jsonl 做統計。三個子步驟：

**① 核心預測**（`core_prediction.py`）
- 只挑「W 答錯」的題目
- 把每一列的 hidden state 取前 `prefix_fraction`（例如 25%）的層，算平均值/標準差/最大值
  /PCA 主成分，串成一個特徵向量
- 標籤：這題 S 是否也答錯（1 = 兩個都錯，是我們要找的「共享難」；0 = S 能救回來）
- 訓一個邏輯迴歸 probe，用早期特徵預測這個標籤，算 AUC
- **重要**：切 train/test 時是按「題目」分組切（同一題的所有樣本只會在 train 或只會在
  test，不會兩邊都有）——因為同一題的多個樣本會共用同一個 S 標籤，如果讓同一題同時出現
  在兩邊，等於讓 test 集偷看到答案，AUC 會虛高。我一開始沒注意到這點，用假資料測出
  AUC=1.0 這種不合理的數字才發現，改成分組切分後降到合理的 0.5 附近。

**② 難度控制**（`difficulty_baseline.py`）——**這是整個實驗的命門**
- 算每題 W 的 pass@1（有 Stage 3 的多重採樣資料才準；現在只有單次生成，pass@1 只能是
  0，這個控制目前形同虛設）
- 把題目按難度分成幾桶
- 在**同一個難度桶內**，比較①的 probe 預測力，跟「單純用難度高低去猜」這個最笨的
  baseline，誰的 AUC 比較高
- 如果 probe 在同一難度桶內還是贏，代表 hidden state 裡有難度數字本身沒有的資訊
  （這才是「幾何訊號有意義」的證據）；贏不了就代表只是在測難度而已

**③ 洩漏控制**（`precedence_check.py`）
- 檢查「模型開始鎖死在錯誤方向」這件事，是不是發生在「正確答案的 token 開始變得有競爭力」
  之前——如果鎖死發生在答案幾乎要生成出來之後，代表訊號可能只是從已經生成的 token 洩漏
  回來的，不是真正早期的訊號
- **現在完全跑不了**，因為需要 Stage 3 的逐步 logits，目前沒有這份資料。程式會誠實回報
  `insufficient_data`，不會假裝檢查通過。

### Stage 5：Go / No-Go 判斷

`run_mvp.py` 把①②③的結果彙整，寫進 `results/mvp_summary.json`：
- ① AUC ≤ 0.5 → 直接 NO-GO（沒訊號）
- ② probe 沒有顯著贏過難度 baseline → NO-GO（只是難度）
- ③ 顯示可疑（答案先於鎖死出現）→ CONDITIONAL（訊號可能不乾淨）
- 都過關 → GO，可以進到 Phase 2（跨家族模型）

---

## 現在能做什麼、不能做什麼

| 能做 | 不能做 |
|---|---|
| 用 `mock_data.py` 造假資料，驗證 Stage 4 的三個模組邏輯跑不跑得通 | 真的載入 W 跑題目（沒 GPU） |
| 呼叫 OpenRouter 取得 S 的真實標籤（`run_s_inference.py`，需要你自己的 API key） | 驗證假說在真實資料上成不成立（W 那邊還沒有資料） |
| 檢查 Stage 4 的程式碼邏輯是否正確（train/test 切分、分桶、AUC 計算） | W 的多重採樣、逐步 logits 收集（程式碼還沒寫） |

### 快速跑一次（只驗證管線邏輯，數字沒有科學意義）

```bash
pip install -r requirements.txt
python mock_data.py
python run_mvp.py --skip-precedence
cat results/mvp_summary.json
```

### 真正要跑實驗，照順序要做的事

1. 有 LLMRouterBench 下載包 → 跑 `prepare_pool_data.py`，只取用它產出的 `queries.jsonl`
2. `cp .env.example .env`，填入真實的 `OPENROUTER_API_KEY` → 跑 `run_s_inference.py`
   取得 S 的標籤（先 `--limit 2 --n-samples 1` 小規模檢查一次再跑全量）
3. 有 GPU → 跑 `extract_all_layers.py` 對 W 模型抽 hidden state（目前只有單次生成）
4. **[待寫程式]** 幫 W 加上多重採樣（給難度 baseline 用，仿照 `run_s_inference.py` 的
   續跑設計）
5. **[待寫程式]** 幫 W 加上逐步 logits 收集（給洩漏控制用）
6. 跑 `python run_mvp.py --config config_experiments.yaml`（不加 `--skip-precedence`，
   因為屆時資料齊全，可以真的檢查洩漏）

---

## 資料格式速查

| 檔案 | 欄位 | 來源 |
|---|---|---|
| `data/queries.jsonl` | `qid, query, ground_truth` | `prepare_pool_data.py` |
| `data/traces_{W模型名}.parquet` | `qid, correct, hidden_all_layers` | `extract_all_layers.py` |
| `data/s_correctness_qwen3.5-27b.parquet` | `qid, sample_idx, model, correct` | `run_s_inference.py` |

`hidden_all_layers` 是巢狀 list `list[層數+1][hidden_dim]`（`.tolist()` 存出來的），
不是 numpy array——這點跟你既有的 `sweep_layers.py` 解析方式一致。

`config_experiments.yaml` 裡的 `data.s_model_label` 要跟 `run_s_inference.py --model`
用的 slug 完全一致（預設都是 `qwen/qwen3.5-27b`）。

---

## 檔案一覽

| 檔案 | 屬於哪個 Stage | 狀態 |
|---|---|---|
| `prepare_pool_data.py` | Stage 1 | 既有，未跑過 |
| `extract_all_layers.py` | Stage 2 | 既有，未跑過；已知 "I" 誤判 bug 未修 |
| `sweep_layers.py` | 輔助分析（逐層掃 AUC，跟本實驗的 probe 是同一類工具） | 既有，未跑過 |
| `run_all_encoders.sh` | Stage 2 批次版 | 既有，未跑過 |
| `run_s_inference.py` | Stage 3（S 標籤，OpenRouter） | 新增，8 個 mock 測試通過，**沒打過真實 API** |
| `test_run_s_inference.py` | `run_s_inference.py` 的單元測試 | 新增 |
| `.env` / `.env.example` | 放 `OPENROUTER_API_KEY`（`.env` 已 gitignore，只是佔位符） | 新增 |
| （W 多重採樣/逐步 logits 腳本） | Stage 3（W） | **不存在，需另外寫** |
| `core_prediction.py` | Stage 4-① | 新增，假資料測試過 |
| `difficulty_baseline.py` | Stage 4-② | 新增，假資料測試過 |
| `precedence_check.py` | Stage 4-③ | 新增，假資料測試過（真資料跑不了） |
| `run_mvp.py` | Stage 4+5 協調器 | 新增，假資料測試過 |
| `mock_data.py` | 測試用合成資料產生器 | 新增 |
| `config_experiments.yaml` | 全域參數設定 | 新增 |
