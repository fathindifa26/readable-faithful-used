"""Review B2: sensitivity of the D_real construction (the survey ruler).

Two criticisms:
(a) WD between ordinal distributions depends on SCALE LENGTH (2-option vs
    6-option questions) -> long-scale questions dominate the mean.
    Variant: WD normalised by scale range, WD/(max(ordinal)-min(ordinal)).
(b) Each pair is averaged over a DIFFERENT SET OF QUESTIONS -> RDM entries
    are not comparable across pairs.
    Variant: restrict to the questions answered by ALL cells within a type.

For each ruler variant, recompute fidelity at L11 H16 and at the best head
per type. If the conclusion holds -> it becomes a sensitivity appendix.

Output: notebooks/output/14_.../dreal_sensitivity.csv
"""
import ast

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
N_Q_CAP, SEED = 300, 42

Z = np.load(f"{NB09}/emb_heads.npz", allow_pickle=True)
keys = list(Z["group_keys"])
heads = Z["emb"].astype(np.float32).mean(0)
types = np.array([k.split(" :: ")[0] for k in keys])
real0 = np.load(f"{NB09}/group_real_dist.npy")

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]
REC = {(r.gk, r.qkey): (np.array(r.responses, float), np.array(r.ordinal, float))
       for r in raw.itertuples()}
QSET = raw.groupby("gk")["qkey"].apply(set).to_dict()


def build_dreal(cells, mode, rng):
    """mode: 'raw' | 'norm' (WD/range) | 'shared' (questions all cells)."""
    m = len(cells)
    D = np.full((m, m), np.nan)
    if mode == "shared":
        common = set.intersection(*[QSET[c] for c in cells])
        qs_for = lambda a, b: sorted(common)
    else:
        def qs_for(a, b):
            qs = sorted(QSET[cells[a]] & QSET[cells[b]])
            if len(qs) > N_Q_CAP:
                pick = rng.choice(len(qs), N_Q_CAP, replace=False)
                qs = [qs[i] for i in pick]
            return qs
    for a in range(m):
        D[a, a] = 0.0
        for b in range(a + 1, m):
            ws = []
            for q in qs_for(a, b):
                pa, oa = REC[(cells[a], q)]
                pb, ob = REC[(cells[b], q)]
                if len(oa) != len(ob):
                    continue
                w = wasserstein_distance(oa, oa, u_weights=pa, v_weights=pb)
                if mode in ("norm", "shared"):
                    rng_scale = oa.max() - oa.min()
                    if rng_scale > 0:
                        w = w / rng_scale
                ws.append(w)
            if ws:
                D[a, b] = D[b, a] = float(np.mean(ws))
    return D


def clean_idx(D, m):
    idx = list(range(m))
    while True:
        iu = np.triu_indices(len(idx), 1)
        sub = D[np.ix_(idx, idx)]
        if not np.isnan(sub[iu]).any():
            return idx
        idx.pop(int(np.argmax(np.isnan(sub).sum(0))))


def rdm_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return 1.0 - S[iu]


rng = np.random.default_rng(SEED)
rows = []
for ty in sorted(set(types)):
    gidx = np.where(types == ty)[0]
    cells = [keys[i] for i in gidx]
    for mode in ["raw", "norm", "shared"]:
        D = build_dreal(cells, mode, rng)
        sub_i = clean_idx(D, len(cells))
        if len(sub_i) < 6:
            rows.append(dict(type=ty, variant=mode, n_cells=len(sub_i),
                             n_shared_questions=np.nan, rho_star=np.nan, rho_best=np.nan))
            continue
        gsel = gidx[sub_i]
        iu = np.triu_indices(len(sub_i), 1)
        y = D[np.ix_(sub_i, sub_i)][iu]
        x_star = rdm_cos(heads[gsel, STAR_LAYER, STAR_HEAD, :])
        r_star = spearmanr(x_star, y).statistic
        flat = heads[gsel].reshape(len(gsel), 1024, 128).transpose(1, 0, 2)
        flat = flat / (np.linalg.norm(flat, axis=2, keepdims=True) + 1e-8)
        sims = np.einsum("lcd,lkd->lck", flat, flat)
        Dh = 1.0 - sims[:, iu[0], iu[1]]
        rhos = np.array([spearmanr(dd, y).statistic for dd in Dh])
        n_common = (len(set.intersection(*[QSET[c] for c in cells]))
                    if mode == "shared" else np.nan)
        rows.append(dict(type=ty, variant=mode, n_cells=len(sub_i),
                         n_shared_questions=n_common,
                         rho_star=float(r_star), rho_best=float(np.nanmax(rhos))))
        print(f"[{ty}|{mode}] n_cells={len(sub_i)} "
              f"{'q_shared=%d ' % n_common if mode=='shared' else ''}"
              f"H16 {r_star:+.3f} | best {np.nanmax(rhos):+.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/dreal_sensitivity.csv", index=False)
print("\nsaved -> dreal_sensitivity.csv")
