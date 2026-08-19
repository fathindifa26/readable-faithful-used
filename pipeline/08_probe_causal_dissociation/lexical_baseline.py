"""Review A2: lexical baseline + partial Spearman — the most dangerous confound.

The worry: prompts differ between cells only in the WORDS of the attribute
values ("Democrat", "Hindu", "30-49", ...). What if the "fidelity map" merely
records the static semantic similarity of those words — which does correlate
with real opinion similarity?

Test: build a lexical RDM from embeddings of small non-opinion sentences
(all-MiniLM-L6-v2, 22M params, without any survey context), then:
  1. rho(lexical, survey) per type       -> how strong the confound is
  2. rho(L11H16, survey) & rho(head-best) -> the old numbers
  3. PARTIAL Spearman: rho(model, survey | lexical)
     -> the fidelity that REMAINS after word similarity is controlled for
  4. extra control: rho(model, lexical) -> how lexical the head itself is

Two variants of the lexical RDM:
  - "value"   : mean embedding of the two value phrases ("18-29", "Democrat")
  - "sentence" : embedding of the whole T0 sentence (exactly the prompt used)

Output: notebooks/output/14_.../lexical_partial.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB09 = results_dir("04_fidelity_map")
OUT = results_dir("08_probe_causal_dissociation")
STAR_LAYER, STAR_HEAD = 11, 16

# ---------- data ----------
real = np.load(f"{NB09}/group_real_dist.npy")
Z = np.load(f"{NB09}/emb_heads.npz", allow_pickle=True)
keys = list(Z["group_keys"])
heads = Z["emb"].astype(np.float32).mean(0)     # Tmean: (169, 32, 32, 128)
types = np.array([k.split(" :: ")[0] for k in keys])

ATTR_LABELS = {
    "RACExRELIG": ("race", "religion"),
    "RACExPOLPARTY": ("race", "political party affiliation"),
    "RACExPOLIDEOLOGY": ("race", "political ideology"),
    "RELIGxPOLPARTY": ("religion", "political party affiliation"),
    "EDUCATIONxINCOME": ("highest level of education", "household income"),
    "AGExPOLPARTY": ("age group", "political party affiliation"),
}

# ---------- lexical embedding ----------
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

v1s, v2s, t0s = [], [], []
for k in keys:
    ty, grp = k.split(" :: ", 1)
    v1, v2 = grp.split(" | ", 1)
    l1, l2 = ATTR_LABELS[ty]
    v1s.append(v1)
    v2s.append(v2)
    t0s.append(f"This survey respondent's {l1} is {v1} and their {l2} is {v2}.")
E1 = enc.encode(v1s, normalize_embeddings=True)
E2 = enc.encode(v2s, normalize_embeddings=True)
E_nilai = (E1 + E2) / 2.0
E_kalimat = enc.encode(t0s, normalize_embeddings=True)


def rdm(X, idx):
    Xs = X[idx]
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    S = Xs @ Xs.T
    iu = np.triu_indices(len(idx), 1)
    return 1.0 - S[iu]


def clean_subset(idx):
    idx = list(idx)
    while True:
        sub = real[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        if not np.isnan(sub[iu]).any():
            return np.array(idx)
        idx.pop(int(np.argmax(np.isnan(sub).sum(0))))


def partial_spearman(x, y, z):
    """rho(x, y | z) on ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


def perm_p_partial(x, y, z, n_perm=2000, seed=0):
    """p for the partial rho: permute cell labels (not pairs)."""
    # x,y,z are already pair vectors; the permutation has to happen at the
    # cell level -> reconstruction
    return None  # p is computed in the caller by permuting the cell matrix


rows = []
rng = np.random.default_rng(0)
for ty in sorted(set(types)):
    idx = clean_subset(np.where(types == ty)[0])
    m = len(idx)
    iu = np.triu_indices(m, 1)
    sub_real = real[np.ix_(idx, idx)]
    y = sub_real[iu]                                  # survey

    # model: L11 H16 and the best head for this type
    x_star = rdm(heads[:, STAR_LAYER, STAR_HEAD, :], idx)
    flat = heads[idx].reshape(m, 32 * 32, 128).transpose(1, 0, 2)
    flat = flat / (np.linalg.norm(flat, axis=2, keepdims=True) + 1e-8)
    sims = np.einsum("lcd,lkd->lck", flat, flat)
    D_all = 1.0 - sims[:, iu[0], iu[1]]
    rhos = np.array([spearmanr(d, y).statistic for d in D_all])
    best = int(np.nanargmax(rhos))
    x_best = D_all[best]

    for lex_name, E in [("value", E_nilai), ("sentence", E_kalimat)]:
        z = rdm(E, idx)                               # lexical
        r_lex_survey = spearmanr(z, y).statistic
        r_star = spearmanr(x_star, y).statistic
        r_best = rhos[best]
        r_star_lex = spearmanr(x_star, z).statistic
        p_star = partial_spearman(x_star, y, z)
        p_best = partial_spearman(x_best, y, z)

        # permutation p for the partial (shuffle cell labels in the survey RDM, 2000x)
        cnt_s = cnt_b = 0
        for _ in range(2000):
            perm = rng.permutation(m)
            yp = sub_real[perm[iu[0]], perm[iu[1]]]
            if partial_spearman(x_star, yp, z) >= p_star - 1e-12:
                cnt_s += 1
            if partial_spearman(x_best, yp, z) >= p_best - 1e-12:
                cnt_b += 1
        rows.append(dict(type=ty, lex=lex_name, n_cells=m,
                         rho_lex_survey=r_lex_survey,
                         rho_star=r_star, rho_star_lex=r_star_lex,
                         partial_star=p_star, p_partial_star=cnt_s / 2000,
                         rho_best=r_best, partial_best=p_best,
                         p_partial_best=cnt_b / 2000))
        print(f"[{ty}|{lex_name}] lex-vs-survey {r_lex_survey:+.3f} | "
              f"H16 {r_star:+.3f} -> partial {p_star:+.3f} (p={cnt_s/2000:.4f}) | "
              f"best {r_best:+.3f} -> partial {p_best:+.3f} (p={cnt_b/2000:.4f})")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/lexical_partial.csv", index=False)
print("\nsaved -> lexical_partial.csv")
