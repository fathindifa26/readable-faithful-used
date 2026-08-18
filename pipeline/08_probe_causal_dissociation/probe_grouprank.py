"""Uji pembeda-kelompok yang BERSIH dari artefak leave-one-out.

Masalah versi sebelumnya: probe dilatih leave-one-cell-out, jadi prediksi
tiap sel dihitung dari model yang TIDAK melihat sel itu. Akibatnya prediksi
otomatis anti-korelasi dengan sel yang ditinggalkan (baseline rata-rata soal
versi LOO dapat grouprank -0.99 -- murni artefak, bukan sinyal).

Perbaikan: belah sel jadi dua bagian. Latih di bagian A, prediksi SEMUA sel
bagian B dengan model yang sama, lalu hitung korelasi urutan kelompok di
dalam B saja. Tukar A/B, rata-ratakan. Tidak ada struktur LOO sama sekali.

Metrik: per soal, Spearman antara "rata-rata opini" prediksi per sel vs versi
asli per sel (dalam separuh yang diuji), lalu dirata-rata antar soal.
Pembanding "mulut" dihitung di subset sel yang sama persis.
"""
import ast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]
N_PC, ALPHA, SEED, N_SPLIT = 128, 1.0, 42, 10
STAR_HEAD, HEAD_DIM = 16, 128

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REAL = {(r.gk, r.qkey): np.array(r.responses, float) for r in raw.itertuples()}
ORDI = {r.qkey: np.array(r.ordinal, float) for r in raw.itertuples()}
del raw

rows = []
for ty in TYPES:
    df = pd.read_csv(f"{OUT}/probeA_{ty}.csv")
    Z = np.load(f"{OUT}/probeA_{ty}.npz")
    feats = {"L11": Z["vecs_l11"].astype(np.float32), "L1": Z["vecs_l1"].astype(np.float32)}
    feats["L11H16"] = feats["L11"][:, STAR_HEAD * HEAD_DIM:(STAR_HEAD + 1) * HEAD_DIM]
    n = len(df)
    nopt, qk, gk = df.n_opt.to_numpy(), df.qk.to_numpy(), df.gk.to_numpy()
    Y = np.zeros((n, 6), np.float32)
    mouth = np.zeros((n, 6), np.float32)
    for i, r in enumerate(df.itertuples()):
        p = REAL[(r.gk, r.qk)]
        Y[i, :len(p)] = p / p.sum()
        v = np.fromstring(r.mouth_pred, sep=",")
        mouth[i, :len(v)] = v / v.sum()

    def expected(M, idx):
        k = nopt[idx[0]]
        o = ORDI[qk[idx[0]]]
        W = np.clip(M[idx, :k], 0, None)
        W = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)
        return W @ o

    def grouprank(P, eval_idx):
        """rata-rata Spearman per soal, hanya di baris eval_idx."""
        keep = np.zeros(len(P), bool)
        keep[eval_idx] = True
        out = []
        for q in set(qk[eval_idx]):
            idx = np.where((qk == q) & keep)[0]
            if len(idx) < 5:
                continue
            a, b = expected(P, idx), expected(Y, idx)
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            r = spearmanr(a, b).statistic
            if not np.isnan(r):
                out.append(r)
        return out

    cells = np.array(sorted(set(gk)))
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in list(feats) + ["mulut"]}
    for rep in range(N_SPLIT):
        perm = rng.permutation(len(cells))
        halves = [set(cells[perm[:len(cells) // 2]]), set(cells[perm[len(cells) // 2:]])]
        for train_cells in halves:
            tr = np.array([i for i in range(n) if gk[i] in train_cells])
            te = np.array([i for i in range(n) if gk[i] not in train_cells])
            acc["mulut"] += grouprank(mouth, te)
            for name, X in feats.items():
                if X.shape[1] > N_PC:
                    pca = PCA(n_components=N_PC, svd_solver="randomized",
                              random_state=0).fit(X[tr])
                    Xtr, Xte = pca.transform(X[tr]), pca.transform(X[te])
                else:
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
                P = np.zeros((n, 6), np.float32)
                P[te] = Ridge(alpha=ALPHA).fit(Xtr, Y[tr]).predict(Xte)
                acc[name] += grouprank(P, te)
    row = dict(tipe=ty, n_sel=len(cells))
    for k, v in acc.items():
        v = np.array(v)
        row[f"gr_{k}"] = v.mean()
        row[f"gr_{k}_se"] = v.std(ddof=1) / np.sqrt(len(v))
    rows.append(row)
    print(f"[{ty}] mulut {row['gr_mulut']:+.3f}  L11 {row['gr_L11']:+.3f}  "
          f"L1 {row['gr_L1']:+.3f}  H16 {row['gr_L11H16']:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/probe_v2_grouprank.csv", index=False)
print("\n=== PEMBEDA KELOMPOK (belah-dua, bebas artefak LOO) ===")
print(res.round(4).to_string(index=False))
