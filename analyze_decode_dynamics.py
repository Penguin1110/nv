# -*- coding: utf-8 -*-
"""
軌跡動力學分析：不訓練分類器、不算 AUC，直接量測每題 decode 軌跡的幾何/
動力學性質，兩組(shared-hard vs salvageable)用 Mann-Whitney U 檢定比較。

動機：「共享難盆地」假設的字面意義就是動力學的——模型困在錯誤的吸引盆地裡
出不來。如果假設成立，訊號應該直接體現在「軌跡繞圈、移動慢、一直回到同一
區域」這類性質上，而不是(只)藏在某個 2560 維線性方向裡。每題壓成一個純量
再做無母數檢定，總共只有 6 個檢定，不會有掃 AUC 那種 660 格多重比較問題。

線性指標(假設直線距離/夾角有意義)：
  mean_step_norm    相鄰取樣點的平均距離(除以平均狀態範數做尺度歸一)
  tortuosity        路徑總長 / 頭尾淨位移——越大越繞
  mean_consec_cos   相鄰兩步方向向量的平均 cosine——越高方向越一致
  longrange_selfsim 相隔超過 25% 進度的狀態對的平均 cosine——越高越「困在
                    同一區域」，這是「盆地」最直接的操作化定義
  settle_frac       從哪個相對進度開始，狀態跟最終狀態的 cosine 一直 >0.9
                    ——越小代表越早「定案」
  spread            狀態到軌跡質心的平均距離(除以平均狀態範數)

非線性指標(只用「誰跟誰接近」的局部關係，不假設整個空間是平的；皆使用
Theiler 窗排除時間上相鄰的點對，避免被時間自相關淹沒)：
  nearest_revisit   最近回訪距離——每個時刻到「時間上不相鄰的其他時刻狀態」
                    的最近距離(尺度歸一後取平均)。越小代表越常回到以前
                    待過的地方 → 越「盆地」。(曾試過 RQA 式的遞迴率/確定性
                    /回訪連貫性,全被合成資料煙霧測試判死:門檻式定義在
                    2560 維上不是循環定義就是全零,連貫性則被 Theiler 窗
                    邊界效應與多圈平手問題弄反方向,故只保留這個無門檻量)
  intrinsic_dim     內在維度(TwoNN 估計)——軌跡局部實際活動在幾維的流形
                    上，衡量探索的「複雜度」而非範圍

用法(在放得下 parquet 的機器上)：
  python analyze_decode_dynamics.py \
      --traces data/traces_Qwen3.5-4B_decode_last45.parquet \
      --s-labels data/s_correctness_qwen3.5-27b.parquet \
      --layer -1 \
      --out results/dynamics_last45_layer-1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is not None and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def trajectory_metrics(states: np.ndarray) -> dict:
    """states: (n_steps, hidden_dim) float32，回傳 6 個動力學指標。"""
    n = len(states)
    norms = np.linalg.norm(states, axis=1)
    mean_norm = float(norms.mean())

    out = {}
    if n < 3:
        # 太短的軌跡大部分指標沒有意義，全部標 NaN，下游會剔除
        return {k: float("nan") for k in
                ["mean_step_norm", "tortuosity", "mean_consec_cos",
                 "longrange_selfsim", "settle_frac", "spread"]}

    diffs = states[1:] - states[:-1]
    step_norms = np.linalg.norm(diffs, axis=1)
    out["mean_step_norm"] = float(step_norms.mean() / mean_norm)

    net = float(np.linalg.norm(states[-1] - states[0]))
    out["tortuosity"] = float(step_norms.sum() / net) if net > 0 else float("nan")

    denom = step_norms[:-1] * step_norms[1:]
    valid = denom > 0
    cons = (diffs[:-1] * diffs[1:]).sum(axis=1)[valid] / denom[valid]
    out["mean_consec_cos"] = float(cons.mean()) if len(cons) else float("nan")

    # 長程自相似：相隔 >25% 進度的所有狀態對的平均 cosine
    unit = states / norms[:, None]
    sim = unit @ unit.T
    lag = max(1, int(n * 0.25))
    iu = np.triu_indices(n, k=lag)
    out["longrange_selfsim"] = float(sim[iu].mean()) if len(iu[0]) else float("nan")

    # 收斂時間：與最終狀態 cosine 從某點起持續 >0.9 的最早相對位置
    cos_final = sim[:, -1]
    settle = n - 1
    for i in range(n - 1, -1, -1):
        if cos_final[i] > 0.9:
            settle = i
        else:
            break
    out["settle_frac"] = float(settle / (n - 1))

    centroid = states.mean(axis=0)
    out["spread"] = float(np.linalg.norm(states - centroid, axis=1).mean() / mean_norm)

    # ---- 以下三個是非線性指標：不假設「直線距離/夾角」在整個空間有意義 ----
    # 共同準備：全對全距離矩陣 + Theiler 窗(排除時間上太近的點對——相鄰取樣點
    # 本來就相似，不排除的話所有指標都會被「時間自相關」這個 trivial 事實淹沒)
    D = np.sqrt(np.maximum(
        (states ** 2).sum(1)[:, None] + (states ** 2).sum(1)[None, :] - 2.0 * (states @ states.T),
        0.0,
    ))
    W = 5  # Theiler 窗：忽略 |i-j|<=5 的點對(=100 個 token 內)
    far_mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > W

    valid_d = D[far_mask & np.triu(np.ones((n, n), bool), 1)]
    if len(valid_d) == 0:
        out.update({"nearest_revisit": float("nan"), "intrinsic_dim": float("nan")})
        return out
    # 高維空間不用「距離 < 門檻」的計數式定義：門檻取分位數會變循環定義
    # (回訪率恆等於分位數本身)、取固定比例又常整條軌跡全零(2560 維的雜訊
    # 地板往往高於任何合理門檻)——這兩種都實際犯過、被合成資料煙霧測試抓
    # 出來。以下改用兩個無門檻的連續量。
    D_masked = D.copy()
    D_masked[~far_mask] = np.inf

    # 最近回訪距離：每個時刻,找「時間上不相鄰的過去/未來狀態」中最近的那個,
    # 距離除以典型距離(中位數)做尺度歸一,再對全軌跡平均。
    # 越小 = 軌跡越常回到以前待過的地方附近 = 越「盆地」。
    nn_idx = np.argmin(D_masked, axis=1)
    nn_dist = D_masked[np.arange(n), nn_idx]
    finite = np.isfinite(nn_dist)
    med = float(np.median(valid_d))
    out["nearest_revisit"] = float(nn_dist[finite].mean() / med) if finite.any() and med > 0 else float("nan")

    # 內在維度(TwoNN 估計)：軌跡局部實際上活動在幾維的流形上。
    # 用每個點的最近鄰/次近鄰距離比 mu=r2/r1 的分布做最大概似估計；
    # 若兩組探索的流形複雜度不同,這個數字會分開。排除 Theiler 窗內的鄰居。
    part = np.partition(D_masked, 1, axis=1)[:, :2]
    r1, r2 = part[:, 0], part[:, 1]
    ok = (r1 > 0) & np.isfinite(r2)
    mu = r2[ok] / r1[ok]
    mu = mu[mu > 1.0]
    out["intrinsic_dim"] = float(len(mu) / np.log(mu).sum()) if len(mu) > 0 else float("nan")
    return out


METRIC_NAMES = ["mean_step_norm", "tortuosity", "mean_consec_cos",
                "longrange_selfsim", "settle_frac", "spread",
                "nearest_revisit", "intrinsic_dim"]


def group_tests_for(per_qid, labels):
    tests = {}
    for m in METRIC_NAMES:
        sh = np.array([per_qid[q][m] for q in per_qid if labels[q] == 1])
        sv = np.array([per_qid[q][m] for q in per_qid if labels[q] == 0])
        sh = sh[~np.isnan(sh)]
        sv = sv[~np.isnan(sv)]
        if len(sh) < 2 or len(sv) < 2:
            tests[m] = {"p_value": float("nan")}
            continue
        u, p = stats.mannwhitneyu(sh, sv, alternative="two-sided")
        tests[m] = {
            "shared_hard_median": float(np.median(sh)),
            "salvageable_median": float(np.median(sv)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
            "n_shared_hard": int(len(sh)),
            "n_salvageable": int(len(sv)),
        }
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, nargs="+", required=True,
                     help="一個或多個 decode parquet(給多個會合併，例如 first45+last45)")
    ap.add_argument("--s-labels", type=str, required=True)
    ap.add_argument("--layer", type=str, default="-1",
                     help="要分析哪一層(整數，-1=最後一層)，或 'all' 把 33 層每層各算一遍"
                          "(輸出每層 x 每指標的組間檢定——注意 33x8=264 個檢定，p<0.05 純"
                          "運氣期望約 13 個，要看的是成片的層區段，不是單格)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    s = pd.read_parquet(args.s_labels)
    sp = s.groupby("qid")["correct"].mean()

    # results[layer][qid] = metrics
    results = {}
    labels = {}
    n_layers_seen = None
    for path in args.traces:
        print(f"[dyn] 載入 {path} ...", file=sys.stderr)
        df = pd.read_parquet(path)
        wrong = df[~df["correct"].astype(bool)]
        n_layers = int(wrong.iloc[0]["num_layers_incl_embed"])
        n_layers_seen = n_layers
        layer_list = (list(range(n_layers)) if args.layer.strip().lower() == "all"
                      else [int(args.layer) % n_layers])
        for _, row in wrong.iterrows():
            steps = row["decode_hidden_states_by_step"]
            # 一次轉成 (n_steps, n_layers, hidden) 再逐層切，避免 33 層各自重新解析巢狀 list
            arr = np.array([np.array([np.asarray(v, dtype=np.float32) for v in step]) for step in steps])
            for layer in layer_list:
                results.setdefault(layer, {})[row["qid"]] = trajectory_metrics(arr[:, layer])
            labels[row["qid"]] = int(sp[row["qid"]] <= 0.5)  # 1=shared hard
            print(f"[dyn] qid={row['qid']} (steps={len(arr)}, layers={len(layer_list)}) 完成", file=sys.stderr)

    n_sh = sum(labels.values())
    print(f"\n[dyn] === 分組比較 (shared_hard={n_sh}, salvageable={len(labels) - n_sh}) ===", file=sys.stderr)
    all_tests = {}
    n_sig_total = 0
    for layer in sorted(results):
        tests = group_tests_for(results[layer], labels)
        all_tests[str(layer)] = tests
        sig = [(m, t["p_value"]) for m, t in tests.items()
               if not np.isnan(t["p_value"]) and t["p_value"] < 0.05]
        n_sig_total += len(sig)
        if len(results) == 1:
            for m, t in tests.items():
                if "shared_hard_median" in t:
                    print(f"[dyn] {m:18s} 中位數 共享難={t['shared_hard_median']:.4f} "
                          f"可救回={t['salvageable_median']:.4f}  p={t['p_value']:.4f}", file=sys.stderr)
        else:
            note = "  ".join(f"{m}(p={p:.3f})" for m, p in sig) if sig else "-"
            print(f"[dyn] layer {layer:2d}: p<0.05 → {note}", file=sys.stderr)

    if len(results) > 1:
        n_tests = len(results) * len(METRIC_NAMES)
        print(f"[dyn] 顯著格總數: {n_sig_total}/{n_tests} (純運氣期望約 {n_tests * 0.05:.0f})",
              file=sys.stderr)

    out = {
        "traces": args.traces,
        "layer_arg": args.layer,
        "n_layers": n_layers_seen,
        "labels": labels,
        "per_qid_metrics": {str(l): results[l] for l in results},
        "group_tests": all_tests,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[dyn] 完成 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
