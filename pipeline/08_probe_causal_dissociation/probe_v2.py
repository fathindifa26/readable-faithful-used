"""Finding 10 v2 — probe from the internal map vs the model's "mouth".

Question: how much group-opinion information EXISTS inside the model but is
LOST by the time it turns into an answer?

Design:
  features: activations at L11 (4096 dim), L1 (4096 dim), and L11 H16 only (128 dim)
  target  : the REAL survey answer distribution for that (cell, question)
  CV      : leave-one-cell-out -- the probe is tested on a group it has NEVER seen
  metric  : Wasserstein distance to the real distribution (smaller is better)
  alpha   : picked inside the fold (inner per-cell split), not on the test data
  comparisons:
    - mouth : softmax over the model's own answer letters
    - question mean : mean real distribution of that question over the TRAINING
                      cells (important baseline: uses no group info at all)

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

print("reading ground truth ...")
raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REAL = {(r.gk, r.qkey): np.array(r.responses, float) for r in raw.itertuples()}
ORDI = {r.qkey: np.array(r.ordinal, float) for r in raw.itertuples()}
del raw


def wd_vec(pred, Y, nopt, ords):
    """Per-row WD between the prediction and the real distribution."""
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
        # question-mean baseline (from the training cells only)
        dfq = pd.DataFrame(Y[tr]).groupby(qk[tr]).mean()
        glob = Y[tr].mean(0)
        for i in te:
            pred["qmean"][i] = dfq.loc[qk[i]].to_numpy() if qk[i] in dfq.index else glob

        # inner per-cell split for picking alpha
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

    # scores
    scores = {"mouth": wd_vec(mouth, Y, nopt, ords)}
    for k in pred:
        scores[k] = wd_vec(pred[k], Y, nopt, ords)
    row = dict(type=ty, n_rows=n, n_cells=len(cells), n_questions=df.qk.nunique())
    for k, v in scores.items():
        row[f"wd_{k}"] = v.mean()
    row["win_vs_mouth"] = float((scores["L11"] < scores["mouth"]).mean())
    row["win_vs_qmean"] = float((scores["L11"] < scores["qmean"]).mean())
    row["alpha_L11_median"] = float(np.median(alpha_used["L11"]))
    rows_sum.append(row)

    np.savez_compressed(f"{OUT}/probe_v2_pred_{ty}.npz",
                        **{f"pred_{k}": v for k, v in pred.items()},
                        mouth=mouth, Y=Y, nopt=nopt, gk=gk.astype(str), qk=qk.astype(str))

    # --- group-discrimination metric: per question, is the group ordering right? ---
    def group_rank_corr(P):
        """Per question: Spearman correlation between the predicted per-cell
        'mean opinion' and the real version, then averaged across questions."""
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

    for k in ["mouth"] + list(pred):
        P = mouth if k == "mouth" else pred[k]
        r, nq = group_rank_corr(P)
        row[f"grouprank_{k}"] = r
    row["grouprank_n_questions"] = nq

    for cell in cells:
        m = gk == cell
        rows_cell.append(dict(type=ty, cell=cell, n=int(m.sum()),
                              **{f"wd_{k}": float(v[m].mean()) for k, v in scores.items()}))
    print(f"[{ty}] {time.time()-t0:.0f}s  mouth {row['wd_mouth']:.4f} | "
          f"L11 {row['wd_L11']:.4f} | L1 {row['wd_L1']:.4f} | "
          f"H16 {row['wd_L11H16']:.4f} | qmean {row['wd_qmean']:.4f}")

pd.DataFrame(rows_sum).to_csv(f"{OUT}/probe_v2_summary.csv", index=False)
pd.DataFrame(rows_cell).to_csv(f"{OUT}/probe_v2_percell.csv", index=False)
print("\n=== SUMMARY ===")
print(pd.DataFrame(rows_sum).round(4).to_string(index=False))
