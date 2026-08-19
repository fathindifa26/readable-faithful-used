"""Review B3: reliability ceiling for the main RSA claim (rho ~ 0.63).

"0.63 out of a ceiling of 0.70" is a very different story from "0.63 out of
0.95". Two sources of noise set the limit:

1. SURVEY: each cell has only n~132 respondents -> the survey RDM is itself
   noisy. Measure: simulate 2 independent survey samples per (cell, question)
   of the original n_unweighted size -> 2 RDMs -> Spearman between them
   (10 repetitions).
2. MODEL: the model RDM changes across templates. Measure: 2v2 template splits
   (3 combination pairs) at L11H16 and at the best head per type -> Spearman
   between the two half RDMs.

Combined ceiling (attenuation) = sqrt(r_survey x r_model) -- the upper bound on
the rho that COULD be observed even if the model were perfect.

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

    # ---------- 1. survey RDM reliability (simulated split-half) ----------
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

    # ---------- 2. model RDM reliability (2v2 template split) ----------
    def model_split_reliability(loc_fn):
        rs = []
        for t_a in combinations(range(4), 2):
            t_b = tuple(t for t in range(4) if t not in t_a)
            Xa = loc_fn(emb[list(t_a)].mean(0))
            Xb = loc_fn(emb[list(t_b)].mean(0))
            rs.append(spearmanr(rdm_cos(Xa), rdm_cos(Xb)).statistic)
        return float(np.mean(rs))

    r_model_star = model_split_reliability(lambda E: E[idx, STAR_LAYER, STAR_HEAD, :])

    # best head for this type (Tmean, from the old map) -- quick search
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

    rows.append(dict(type=ty, n_cells=m,
                     r_survey=r_survey, r_model_star=r_model_star,
                     r_model_best=r_model_best,
                     ceiling_star=ceil_star, ceiling_best=ceil_best,
                     obs_star=obs_star, obs_best=float(rhos[best]),
                     pct_ceiling_star=obs_star / ceil_star if ceil_star > 0 else np.nan,
                     pct_ceiling_best=float(rhos[best]) / ceil_best if ceil_best > 0 else np.nan))
    print(f"[{ty}] survey {r_survey:+.3f} | model H16 {r_model_star:+.3f} "
          f"-> ceiling {ceil_star:.3f} | obs H16 {obs_star:+.3f} "
          f"({obs_star/ceil_star:.0%} of the ceiling)")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/rsa_reliability.csv", index=False)
print("\nsaved -> rsa_reliability.csv")
