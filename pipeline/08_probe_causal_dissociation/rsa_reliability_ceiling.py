"""Review B3: reliability ceiling untuk klaim RSA utama (rho ~ 0.63).

"0.63 dari langit-langit 0.70" cerita yang beda banget dari "0.63 dari 0.95".
Dua sumber noise yang membatasi:

1. SURVEI: tiap sel cuma n~132 responden -> RDM survei sendiri berisik.
   Ukur: simulasikan 2 sampel survei independen per (sel, soal) sebesar
   n_unweighted asli -> 2 RDM -> Spearman antar keduanya (10 ulangan).
2. MODEL: RDM model berubah antar template. Ukur: split template 2v2
   (3 pasangan kombinasi) di L11H16 & head terbaik per tipe -> Spearman
   antar dua RDM belahan.

Langit-langit gabungan (attenuation) = sqrt(r_survei x r_model) --
batas atas rho yang MUNGKIN diamati kalau model sempurna.

Output: notebooks/output/14_.../rsa_reliability.csv
"""
import ast
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB09 = results_dir("04_fidelity_map")
OUT = results_dir("08_probe_causal_dissociation")
STAR_LAYER, STAR_HEAD = 11, 16
N_REP_SURVEY = 10
N_Q_CAP = 300
SEED = 42

real = np.load(f"{NB09}/group_real_dist.npy")
Z = np.load(f"{NB09}/emb_heads.npz", allow_pickle=True)
keys = list(Z["group_keys"])
emb = Z["emb"].astype(np.float32)          # (4, 169, 32, 32, 128)
types = np.array([k.split(" :: ")[0] for k in keys])

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REC = {(r.gk, r.qkey): (np.array(r.responses, float), np.array(r.ordinal, float),
                        int(max(r.n_unweighted, 5))) for r in raw.itertuples()}
QSET = raw.groupby("gk")["qkey"].apply(set).to_dict()


def clean_subset(idx):
    idx = list(idx)
    while True:
        sub = real[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        if not np.isnan(sub[iu]).any():
            return np.array(idx)
        idx.pop(int(np.argmax(np.isnan(sub).sum(0))))


def rdm_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return 1.0 - S[iu]


rng = np.random.default_rng(SEED)
rows = []
for ty in sorted(set(types)):
    idx = clean_subset(np.where(types == ty)[0])
    m = len(idx)
    cells = [keys[i] for i in idx]
    iu = np.triu_indices(m, 1)

    # ---------- 1. reliabilitas RDM survei (split-half simulasi) ----------
    shared = {}
    for a in range(m):
        for b in range(a + 1, m):
            qs = sorted(QSET[cells[a]] & QSET[cells[b]])
            if len(qs) > N_Q_CAP:
                pick = rng.choice(len(qs), N_Q_CAP, replace=False)
                qs = [qs[i] for i in pick]
            shared[(a, b)] = qs

    r_survey = []
    for rep in range(N_REP_SURVEY):
        D1 = np.zeros(len(iu[0]))
        D2 = np.zeros(len(iu[0]))
        for k, (a, b) in enumerate(zip(*iu)):
            w1, w2 = [], []
            for q in shared[(a, b)]:
                pa, oa, na = REC[(cells[a], q)]
                pb, ob, nb = REC[(cells[b], q)]
                if len(oa) != len(ob):
                    continue
                pa = pa / pa.sum()
                pb = pb / pb.sum()
                s1a = rng.multinomial(na, pa) / na
                s1b = rng.multinomial(nb, pb) / nb
                s2a = rng.multinomial(na, pa) / na
                s2b = rng.multinomial(nb, pb) / nb
                w1.append(wasserstein_distance(oa, oa, u_weights=s1a, v_weights=s1b))
                w2.append(wasserstein_distance(oa, oa, u_weights=s2a, v_weights=s2b))
            D1[k], D2[k] = np.mean(w1), np.mean(w2)
        r_survey.append(spearmanr(D1, D2).statistic)
    r_survey = float(np.mean(r_survey))

    # ---------- 2. reliabilitas RDM model (split template 2v2) ----------
    def model_split_reliability(loc_fn):
        rs = []
        for t_a in combinations(range(4), 2):
            t_b = tuple(t for t in range(4) if t not in t_a)
            Xa = loc_fn(emb[list(t_a)].mean(0))
            Xb = loc_fn(emb[list(t_b)].mean(0))
            rs.append(spearmanr(rdm_cos(Xa), rdm_cos(Xb)).statistic)
        return float(np.mean(rs))

    r_model_star = model_split_reliability(lambda E: E[idx, STAR_LAYER, STAR_HEAD, :])

    # head terbaik tipe ini (Tmean, dari peta lama) -- cari cepat
    Tm = emb.mean(0)
    flat = Tm[idx].reshape(m, 32 * 32, 128).transpose(1, 0, 2)
    flat = flat / (np.linalg.norm(flat, axis=2, keepdims=True) + 1e-8)
    sims = np.einsum("lcd,lkd->lck", flat, flat)
    D_all = 1.0 - sims[:, iu[0], iu[1]]
    y = real[np.ix_(idx, idx)][iu]
    rhos = np.array([spearmanr(dd, y).statistic for dd in D_all])
    best = int(np.nanargmax(rhos))
    bl, bh = divmod(best, 32)
    r_model_best = model_split_reliability(lambda E: E[idx, bl, bh, :])

    ceil_star = float(np.sqrt(max(r_survey, 0) * max(r_model_star, 0)))
    ceil_best = float(np.sqrt(max(r_survey, 0) * max(r_model_best, 0)))
    obs_star = float(spearmanr(rdm_cos(Tm[idx, STAR_LAYER, STAR_HEAD, :]), y).statistic)

    rows.append(dict(tipe=ty, n_sel=m,
                     r_survei=r_survey, r_model_star=r_model_star,
                     r_model_best=r_model_best,
                     ceiling_star=ceil_star, ceiling_best=ceil_best,
                     obs_star=obs_star, obs_best=float(rhos[best]),
                     pct_ceiling_star=obs_star / ceil_star if ceil_star > 0 else np.nan,
                     pct_ceiling_best=float(rhos[best]) / ceil_best if ceil_best > 0 else np.nan))
    print(f"[{ty}] survei {r_survey:+.3f} | model H16 {r_model_star:+.3f} "
          f"-> ceiling {ceil_star:.3f} | obs H16 {obs_star:+.3f} "
          f"({obs_star/ceil_star:.0%} dari ceiling)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/rsa_reliability.csv", index=False)
print("\ndisimpan -> rsa_reliability.csv")
