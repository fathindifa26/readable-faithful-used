"""Review B1: an apple-to-apple comparison of head vs residual.

Two unfairnesses that get corrected:
1. ASYMMETRIC SELECTION: "head best" is picked from 1,024 candidates,
   "residual best" from 33 -- but the table has no HELD-OUT residual column.
   -> compute a template-held-out residual (pick the layer on 3 templates,
      measure on the 4th, average over 4 folds), parallel to the held-out head.
2. DIMENSION & ANISOTROPY: a 128-dim head vs an anisotropic 4,096-dim residual.
   -> control: 1,024 RANDOM 128-dim PROJECTIONS of the residual stream
      (32 projections x 32 layers, mirroring the number of head candidates).
      If the max over the 1024 random projections matches the head max, then
      "heads dominate" is partly a geometry/selection artefact, not content.

Output: notebooks/output/14_.../head_vs_residual_fair.csv
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
N_PROJ_PER_LAYER = 32
PROJ_DIM = 128
SEED = 42

real = np.load(f"{NB09}/group_real_dist.npy")
Zh = np.load(f"{NB09}/emb_heads.npz", allow_pickle=True)
Zr = np.load(f"{NB09}/emb_resid.npz", allow_pickle=True)
keys = list(Zh["group_keys"])
E_h = Zh["emb"].astype(np.float32)          # (4,169,32,32,128)
E_r = Zr["emb"].astype(np.float32)          # (4,169,33,4096)
types = np.array([k.split(" :: ")[0] for k in keys])
L_RES = E_r.shape[2]


def clean_subset(idx):
    idx = list(idx)
    while True:
        sub = real[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        if not np.isnan(sub[iu]).any():
            return np.array(idx)
        idx.pop(int(np.argmax(np.isnan(sub).sum(0))))


def rdm_rows(X):
    """X (n_loc, n_cell, d) -> cosine distances (n_loc, n_pair)."""
    Xn = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-8)
    S = np.einsum("lcd,lkd->lck", Xn, Xn)
    iu = np.triu_indices(X.shape[1], 1)
    return 1.0 - S[:, iu[0], iu[1]]


def spearman_rows(D, y):
    ry = rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    rr = np.apply_along_axis(rankdata, 1, D)
    with np.errstate(invalid="ignore"):
        rr = (rr - rr.mean(1, keepdims=True)) / rr.std(1, keepdims=True)
    return rr @ ry / len(y)


rng = np.random.default_rng(SEED)
# the random projections are FIXED ONCE for all types/folds (fair: same candidates)
PROJ = [rng.standard_normal((4096, PROJ_DIM)).astype(np.float32) / np.sqrt(4096)
        for _ in range(N_PROJ_PER_LAYER)]

rows = []
for ty in sorted(set(types)):
    idx = clean_subset(np.where(types == ty)[0])
    m = len(idx)
    iu = np.triu_indices(m, 1)
    y = real[np.ix_(idx, idx)][iu]

    def all_locs_heads(E4):        # E4 (169,32,32,128) already Tmean/template subset
        X = E4[idx].reshape(m, 1024, 128).transpose(1, 0, 2)
        return rdm_rows(X)

    def all_locs_resid(E4):        # E4 (169,33,4096)
        X = E4[idx].transpose(1, 0, 2)
        return rdm_rows(X)

    def all_locs_proj(E4):         # projected residual: 32 layers x 32 projections
        outs = []
        for L in range(1, L_RES):  # skip layer 0 (degenerate)
            base = E4[idx, L, :]
            for P in PROJ:
                outs.append(base @ P)
        X = np.stack(outs).astype(np.float32)   # (32*32, m, 128)
        return rdm_rows(X)

    # --- Tmean (selected) ---
    Tm_h, Tm_r = E_h.mean(0), E_r.mean(0)
    rho_h = spearman_rows(all_locs_heads(Tm_h), y)
    rho_r = spearman_rows(all_locs_resid(Tm_r), y)
    rho_p = spearman_rows(all_locs_proj(Tm_r), y)

    # --- template held-out for ALL THREE candidate families (identical procedure) ---
    def heldout(all_locs, E):
        ho = []
        for t_out in range(4):
            t_in = [t for t in range(4) if t != t_out]
            D_in = all_locs(E[t_in].mean(0))
            selected = int(np.nanargmax(spearman_rows(D_in, y)))
            D_out = all_locs(E[t_out])
            ho.append(float(spearman_rows(D_out[selected:selected + 1], y)[0]))
        return float(np.mean(ho))

    ho_head = heldout(all_locs_heads, E_h)
    ho_resid = heldout(all_locs_resid, E_r)
    ho_proj = heldout(all_locs_proj, E_r)

    rows.append(dict(type=ty, n_cells=m,
                     head_selected=float(np.nanmax(rho_h)),
                     resid_selected=float(np.nanmax(rho_r)),
                     proj_selected=float(np.nanmax(rho_p)),
                     head_heldout=ho_head, resid_heldout=ho_resid,
                     proj_heldout=ho_proj,
                     head_median=float(np.nanmedian(rho_h)),
                     proj_median=float(np.nanmedian(rho_p))))
    print(f"[{ty}] selected: head {np.nanmax(rho_h):+.3f} resid {np.nanmax(rho_r):+.3f} "
          f"proj {np.nanmax(rho_p):+.3f} | held-out: head {ho_head:+.3f} "
          f"resid {ho_resid:+.3f} proj {ho_proj:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/head_vs_residual_fair.csv", index=False)
print("\nsaved -> head_vs_residual_fair.csv")
