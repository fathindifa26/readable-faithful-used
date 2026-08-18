"""Review B4: dua cek fairness untuk hasil probe (finding 10 v2).

1. KALIBRASI: "probe 22-30% lebih dekat dari mulut" bisa cuma karena mulut
   miskalibrasi (softmax letter-logits terlalu datar/lancip), sedangkan
   ridge otomatis belajar skala benar. Baseline adil: mulut + TEMPERATURE
   SCALING (1 parameter, di-fit di fold training, dievaluasi leave-one-cell-
   out seperti probe).
2. BOTTLENECK TAK SIMETRIS: L11 penuh direduksi PCA-128, head dipakai utuh
   128 dim. "Single head does the work" bisa artefak bottleneck. Tambah:
   ridge di L11 PENUH 4096-dim tanpa PCA.

Output: notebooks/output/14_.../probe_fairness.csv
"""
import ast
import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]
TEMPS = np.concatenate([np.arange(0.2, 1.0, 0.1), np.arange(1.0, 5.1, 0.25)])
ALPHA_FULL = 100.0   # dim lebih besar -> regularisasi lebih besar (tetap fair)

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REAL = {(r.gk, r.qkey): np.array(r.responses, float) for r in raw.itertuples()}
ORDI = {r.qkey: np.array(r.ordinal, float) for r in raw.itertuples()}
del raw

rows = []
for ty in TYPES:
    t0 = time.time()
    df = pd.read_csv(f"{OUT}/probeA_{ty}.csv")
    Z = np.load(f"{OUT}/probeA_{ty}.npz")
    X11 = Z["vecs_l11"].astype(np.float32)
    n = len(df)
    nopt = df.n_opt.to_numpy()
    qk = df.qk.to_numpy()
    gk = df.gk.to_numpy()
    ords = [ORDI[q] for q in qk]
    Y = np.zeros((n, 6), np.float32)
    mouth = np.zeros((n, 6), np.float32)
    for i, r in enumerate(df.itertuples()):
        p = REAL[(r.gk, r.qk)]
        Y[i, :len(p)] = p / p.sum()
        v = np.fromstring(r.mouth_pred, sep=",")
        mouth[i, :len(v)] = v / v.sum()

    def wd_mean(P, idx):
        out = 0.0
        for i in idx:
            k = nopt[i]
            p = np.clip(P[i, :k], 0, None)
            p = p / p.sum() if p.sum() > 0 else np.full(k, 1 / k)
            o = ords[i]
            out += wasserstein_distance(o, o, u_weights=p, v_weights=Y[i, :k])
        return out / len(idx)

    def temp_apply(P, T):
        Q = np.zeros_like(P)
        for i in range(len(P)):
            k = nopt[i]
            p = np.clip(P[i, :k], 1e-9, None)
            q = p ** (1.0 / T)
            Q[i, :k] = q / q.sum()
        return Q

    cells = sorted(set(gk))
    # pre-hitung WD per baris utk tiap T sekali saja (temp_apply tak tergantung fold)
    wd_by_T = {}
    for T in TEMPS:
        Q = temp_apply(mouth, T)
        w = np.empty(n)
        for i in range(n):
            k = nopt[i]
            w[i] = wasserstein_distance(ords[i], ords[i],
                                        u_weights=Q[i, :k], v_weights=Y[i, :k])
        wd_by_T[T] = (Q, w)

    pred_temp = np.zeros((n, 6), np.float32)
    pred_full = np.zeros((n, 6), np.float32)
    temps_used = []
    for cell in cells:
        te = np.where(gk == cell)[0]
        tr_mask = gk != cell
        tr = np.where(tr_mask)[0]
        # (1) temperature scaling: fit T di training fold (pakai pre-hitung)
        best_T = min(TEMPS, key=lambda T: wd_by_T[T][1][tr_mask].mean())
        temps_used.append(best_T)
        pred_temp[te] = wd_by_T[best_T][0][te]
        # (2) ridge L11 PENUH tanpa PCA
        mu = X11[tr].mean(0)
        pred_full[te] = Ridge(alpha=ALPHA_FULL).fit(X11[tr] - mu, Y[tr]).predict(X11[te] - mu)

    allidx = range(n)
    rows.append(dict(tipe=ty, n=n,
                     wd_mulut=wd_mean(mouth, allidx),
                     wd_mulut_temp=wd_mean(pred_temp, allidx),
                     T_median=float(np.median(temps_used)),
                     wd_L11_fullridge=wd_mean(pred_full, allidx)))
    print(f"[{ty}] {time.time()-t0:.0f}s mulut {rows[-1]['wd_mulut']:.4f} | "
          f"mulut+temp {rows[-1]['wd_mulut_temp']:.4f} (T~{rows[-1]['T_median']:.2f}) | "
          f"L11 full-ridge {rows[-1]['wd_L11_fullridge']:.4f}")

res = pd.DataFrame(rows)
# gabung dgn hasil lama biar sebanding dalam satu tabel
old = pd.read_csv(f"{OUT}/probe_v2_summary.csv")[["tipe", "wd_L11", "wd_L11H16", "wd_qmean"]]
res = res.merge(old, on="tipe")
res.to_csv(f"{OUT}/probe_fairness.csv", index=False)
print()
print(res.round(4).to_string(index=False))
