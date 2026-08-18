"""Finding 10 v2 — probe dari peta internal vs "mulut" model.

Pertanyaan: berapa banyak informasi opini kelompok yang ADA di dalam model
tapi HILANG waktu jadi jawaban?

Desain:
  fitur   : aktivasi di L11 (4096 dim), L1 (4096 dim), dan L11 H16 saja (128 dim)
  target  : distribusi jawaban survei ASLI utk (sel, soal) itu
  CV      : leave-one-cell-out -- probe diuji di kelompok yang BELUM PERNAH dilihat
  metrik  : Wasserstein distance ke distribusi asli (makin kecil makin bagus)
  alpha   : dipilih di dalam fold (inner split per-sel), bukan di data uji
  pembanding:
    - mulut  : softmax huruf jawaban model sendiri
    - rata2 soal : rata-rata distribusi asli soal itu dari sel-sel TRAINING
                   (baseline penting: nggak pakai info kelompok sama sekali)

Output: notebooks/output/14_.../probe_v2_summary.csv + probe_v2_percell.csv
"""
import ast
import time

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]
N_PC = 128
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
STAR_HEAD, HEAD_DIM = 16, 128
SEED = 42

print("baca ground truth ...")
raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REAL = {(r.gk, r.qkey): np.array(r.responses, float) for r in raw.itertuples()}
ORDI = {r.qkey: np.array(r.ordinal, float) for r in raw.itertuples()}
del raw


def wd_vec(pred, Y, nopt, ords):
    """WD per baris antara prediksi dan distribusi asli."""
    out = np.empty(len(pred))
    for i in range(len(pred)):
        k = nopt[i]
        p = np.clip(pred[i, :k], 0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(k, 1.0 / k)
        o = ords[i]
        out[i] = wasserstein_distance(o, o, u_weights=p, v_weights=Y[i, :k])
    return out


rows_sum, rows_cell = [], []
for ty in TYPES:
    t0 = time.time()
    df = pd.read_csv(f"{OUT}/probeA_{ty}.csv")
    Z = np.load(f"{OUT}/probeA_{ty}.npz")
    feats = {"L11": Z["vecs_l11"].astype(np.float32),
             "L1": Z["vecs_l1"].astype(np.float32)}
    feats["L11H16"] = feats["L11"][:, STAR_HEAD * HEAD_DIM:(STAR_HEAD + 1) * HEAD_DIM]

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

    cells = sorted(set(gk))
    rng = np.random.default_rng(SEED)
    pred = {k: np.zeros((n, 6), np.float32) for k in list(feats) + ["qmean"]}
    alpha_used = {k: [] for k in feats}

    for fold, cell in enumerate(cells):
        te = np.where(gk == cell)[0]
        tr = np.where(gk != cell)[0]
        # baseline rata-rata soal (hanya dari sel training)
        dfq = pd.DataFrame(Y[tr]).groupby(qk[tr]).mean()
        glob = Y[tr].mean(0)
        for i in te:
            pred["qmean"][i] = dfq.loc[qk[i]].to_numpy() if qk[i] in dfq.index else glob

        # inner split per-sel buat milih alpha
        tr_cells = [c for c in cells if c != cell]
        inner_val = set(rng.choice(tr_cells, size=max(1, len(tr_cells) // 5), replace=False))
        in_tr = np.array([i for i in tr if gk[i] not in inner_val])
        in_va = np.array([i for i in tr if gk[i] in inner_val])

        for name, X in feats.items():
            if X.shape[1] > N_PC:
                pca = PCA(n_components=N_PC, svd_solver="randomized", random_state=0).fit(X[tr])
                Xtr, Xte, Xin, Xiv = (pca.transform(X[tr]), pca.transform(X[te]),
                                      pca.transform(X[in_tr]), pca.transform(X[in_va]))
            else:
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
                Xin, Xiv = (X[in_tr] - mu) / sd, (X[in_va] - mu) / sd
            best_a, best_wd = None, np.inf
            for a in ALPHAS:
                pv = Ridge(alpha=a).fit(Xin, Y[in_tr]).predict(Xiv)
                w = wd_vec(pv, Y[in_va], nopt[in_va], [ords[i] for i in in_va]).mean()
                if w < best_wd:
                    best_a, best_wd = a, w
            alpha_used[name].append(best_a)
            pred[name][te] = Ridge(alpha=best_a).fit(Xtr, Y[tr]).predict(Xte)

    # skor
    scores = {"mulut": wd_vec(mouth, Y, nopt, ords)}
    for k in pred:
        scores[k] = wd_vec(pred[k], Y, nopt, ords)
    row = dict(tipe=ty, n_baris=n, n_sel=len(cells), n_soal=df.qk.nunique())
    for k, v in scores.items():
        row[f"wd_{k}"] = v.mean()
    row["menang_vs_mulut"] = float((scores["L11"] < scores["mulut"]).mean())
    row["menang_vs_qmean"] = float((scores["L11"] < scores["qmean"]).mean())
    row["alpha_L11_median"] = float(np.median(alpha_used["L11"]))
    rows_sum.append(row)

    np.savez_compressed(f"{OUT}/probe_v2_pred_{ty}.npz",
                        **{f"pred_{k}": v for k, v in pred.items()},
                        mouth=mouth, Y=Y, nopt=nopt, gk=gk.astype(str), qk=qk.astype(str))

    # --- metrik pembeda-kelompok: per soal, urutan kelompok bener nggak? ---
    def group_rank_corr(P):
        """Per soal: korelasi Spearman antara 'rata-rata opini' prediksi
        per sel vs versi asli, lalu dirata-rata antar soal."""
        from scipy.stats import spearmanr as _sp
        out = []
        for q in set(qk):
            idx = np.where(qk == q)[0]
            if len(idx) < 5:
                continue
            k = nopt[idx[0]]
            o = ords[idx[0]]
            def em(M):
                W = np.clip(M[idx, :k], 0, None)
                W = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)
                return W @ o
            a, b = em(P), em(Y)
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            r = _sp(a, b).statistic
            if not np.isnan(r):
                out.append(r)
        return float(np.mean(out)), len(out)

    for k in ["mulut"] + list(pred):
        P = mouth if k == "mulut" else pred[k]
        r, nq = group_rank_corr(P)
        row[f"grouprank_{k}"] = r
    row["grouprank_n_soal"] = nq

    for cell in cells:
        m = gk == cell
        rows_cell.append(dict(tipe=ty, sel=cell, n=int(m.sum()),
                              **{f"wd_{k}": float(v[m].mean()) for k, v in scores.items()}))
    print(f"[{ty}] {time.time()-t0:.0f}s  mulut {row['wd_mulut']:.4f} | "
          f"L11 {row['wd_L11']:.4f} | L1 {row['wd_L1']:.4f} | "
          f"H16 {row['wd_L11H16']:.4f} | rata2soal {row['wd_qmean']:.4f}")

pd.DataFrame(rows_sum).to_csv(f"{OUT}/probe_v2_summary.csv", index=False)
pd.DataFrame(rows_cell).to_csv(f"{OUT}/probe_v2_percell.csv", index=False)
print("\n=== RINGKASAN ===")
print(pd.DataFrame(rows_sum).round(4).to_string(index=False))
