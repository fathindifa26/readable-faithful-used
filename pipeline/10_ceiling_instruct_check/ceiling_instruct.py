"""Review A6 / R8.2: ceiling identity-swap di base vs INSTRUCT.

Klaim paper 6.1 ("output nyaris tak berubah saat seluruh identitas
diganti") diukur di Mistral-7B base. Reviewer: bisa jadi properti
interface (base + letter readout), bukan properti model. Notebook 16
mengukur mulut Mistral-7B-v0.1 (prompt mentah) vs
Mistral-7B-Instruct-v0.2 (chat template) pada sel & soal yang sama.

Per model x tipe, atas pasangan sel setipe (A,B) terurut di soal bersama:
- ceiling  : mean_q [ wd(predA, realB) - wd(predB, realB) ]
             (persis definisi 6.1: seberapa jauh prediksi bergerak
             saat SELURUH identitas diganti, diukur relatif ke truth B)
- adv (D2) : mean_q [ wd(predA, realB) - wd(predA, realA) ]
             (>0 = prediksi A lebih dekat ke truth A sendiri);
             median + sign test level PASANGAN (bukan item).
- wd_abs   : akurasi absolut mean wd(pred, real).

Output: notebooks/output/16_ceiling_instruct_kaggle/analisis/ceiling_summary.csv
"""
import ast
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wasserstein_distance

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB16 = results_dir("10_ceiling_instruct_check")
OUT_DIR = NB16

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REAL = {(r.gk, r.qkey): (np.array(r.responses, float), np.array(r.ordinal, float))
        for r in raw.itertuples()}

rows = []
for tag in ["mistral_base", "mistral_instruct"]:
    d = pd.read_csv(f"{NB16}/mouth_preds_{tag}.csv")
    d["p"] = d["pred"].apply(lambda s: np.fromstring(s, sep=","))
    pred_by = {(r.gk, r.qk): r.p for r in d.itertuples()}
    q_by = d.groupby("gk")["qk"].apply(set).to_dict()

    for ty, g in d.groupby("ty"):
        # akurasi absolut
        wd_abs = np.mean([wasserstein_distance(REAL[(r.gk, r.qk)][1],
                                               REAL[(r.gk, r.qk)][1],
                                               u_weights=r.p,
                                               v_weights=REAL[(r.gk, r.qk)][0])
                          for r in g.itertuples()])
        cells = sorted(g.gk.unique())
        ceil_pair, adv_pair = [], []
        for A, B in itertools.permutations(cells, 2):
            shared = q_by[A] & q_by[B]
            cs, asq = [], []
            for qk in shared:
                pA, oA = REAL[(A, qk)]
                pB, oB = REAL[(B, qk)]
                if len(oA) != len(oB):
                    continue
                prA, prB = pred_by[(A, qk)], pred_by[(B, qk)]
                wAB = wasserstein_distance(oB, oB, u_weights=prA, v_weights=pB)
                wBB = wasserstein_distance(oB, oB, u_weights=prB, v_weights=pB)
                wAA = wasserstein_distance(oA, oA, u_weights=prA, v_weights=pA)
                cs.append(wAB - wBB)
                asq.append(wAB - wAA)
            if cs:
                ceil_pair.append(np.mean(cs))
                adv_pair.append(np.mean(asq))
        ceil_pair, adv_pair = np.array(ceil_pair), np.array(adv_pair)
        n_pos = int((adv_pair > 0).sum())
        p_sign = binomtest(n_pos, len(adv_pair)).pvalue
        rows.append(dict(model=tag, tipe=ty, n_pair=len(ceil_pair),
                         wd_abs=round(float(wd_abs), 4),
                         ceiling=round(float(np.mean(ceil_pair)), 4),
                         ceiling_pct=round(float(np.mean(ceil_pair)) / float(wd_abs) * 100, 2),
                         adv_median=round(float(np.median(adv_pair)), 4),
                         frac_adv_pos=round(n_pos / len(adv_pair), 3),
                         p_sign=round(float(p_sign), 4)))
        print(f"[{tag}|{ty}] wd_abs {wd_abs:.4f} | ceiling {np.mean(ceil_pair):+.4f} "
              f"({np.mean(ceil_pair)/wd_abs*100:.1f}%) | adv med {np.median(adv_pair):+.4f} "
              f"| frac+ {n_pos}/{len(adv_pair)} p={p_sign:.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT_DIR}/ceiling_summary.csv", index=False)
print("\n== ringkasan per model (rata-rata antar tipe) ==")
print(res.groupby("model")[["wd_abs", "ceiling", "ceiling_pct", "frac_adv_pos"]].mean().round(4).to_string())
