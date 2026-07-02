#!/usr/bin/env bash
# 批次抽取:對池內每一個模型,序列載入 → prefill forward → 存 hidden state → 卸載。
#
# 用法:
#   1. 跑完 prepare_pool_data.py 後,終端會列出池的目錄名。
#   2. 把目錄名 → HuggingFace repo id 的對應填進下面的 MODELS 陣列
#      (格式: "目錄名:HF_repo_id",目錄名會拿來當輸出檔名)。
#   3. bash run_all_encoders.sh data/queries.jsonl data/hs 30
#      參數:queries 路徑、輸出目錄前綴、limit(留空=全量)
#
# 注意:
#   - 每個模型跑完會自動釋放,GPU 只需容納一個 ~7B 模型
#   - 中斷後重跑:已完成的輸出檔存在就跳過,不會重抽

set -u

QUERIES="${1:?用法: run_all_encoders.sh <queries.jsonl> <輸出前綴> [limit]}"
OUT_PREFIX="${2:?缺輸出前綴,例如 data/hs}"
LIMIT="${3:-}"

# ↓↓↓ 跑完 prepare_pool_data.py 之後,把真實名單填進來 ↓↓↓
# 格式:"目錄名:HuggingFace_repo_id"
MODELS=(
  "Qwen3-8B:Qwen/Qwen3-8B"
  # "GLM-Z1:zai-org/GLM-Z1-9B-0414"          # ← 範例,以實際目錄名與 HF id 為準
  # "Llama-3.1-8B:meta-llama/Llama-3.1-8B-Instruct"
  # ... 其餘池成員
)

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  repo="${entry#*:}"
  out="${OUT_PREFIX}_${name}.parquet"
  if [ -f "$out" ]; then
    echo "[batch] $name 已存在 → 跳過($out)"
    continue
  fi
  echo "[batch] ===== $name ($repo) ====="
  python extract_all_layers.py \
    --model "$repo" \
    --queries "$QUERIES" \
    --out "$out" \
    --skip-generate $LIMIT_ARG
  status=$?
  if [ $status -ne 0 ]; then
    echo "[batch] $name 失敗(exit $status),繼續下一個;之後重跑本腳本會自動補"
  fi
done

echo "[batch] 全部完成。輸出:${OUT_PREFIX}_<模型名>.parquet"
