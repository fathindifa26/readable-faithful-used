"""Review B6 / round-2 R8.3: shuffled-donor control for the H16 damage.

Data: notebook 17 (protocol & pairs EXACTLY as in notebook 12; H16 patch
conditions in prompt A: true / othercell_1 / othercell_2 / dimshuffle).

Analysis, everything at the PAIR level (12 clusters, exact sign-flip
2^12):
  1. Replication: `true` vs 0 should be ~t=-4.2 in RACExRELIG (finding 08:
     t_vs_0 = -4.24; -5.21 vs the random head, absent here).
  2. KEY TEST: per-pair difference shift_true - shift_control.
     - difference ~0  -> generic corruption (every donor is equally
       harmful);
     - difference < 0 and significant -> `true` is more harmful -> an
       identity-specific component.
  3. total_movement across conditions: do all conditions shift the
     prediction equally far mechanically (needed for #2 to be fair).

Output: results/11_shuffled_donor_control/analisis/
        shuffled_donor_summary.csv + shuffled_donor_contrasts.csv
"""
import itertools
import os

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB17 = results_dir("11_shuffled_donor_control")
OUT = NB17
NB08 = results_dir("08_probe_causal_dissociation")

CONTROLS = ["othercell_1", "othercell_2", "dimshuffle"]


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def exact_flips(n):
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n)))


def p_twosided_exact(pm):
    F = exact_flips(len(pm))
    M = F * pm[None, :]
    ts = M.mean(1) / (M.std(1, ddof=1) / np.sqrt(M.shape[1]))
    return float((np.abs(ts) >= abs(tstat(pm)) - 1e-12).mean())


d = pd.read_csv(f"{NB17}/shuffled_donor_rows.csv")

# per-pair mean per condition
pm_shift = d.pivot_table(index=["attr_type", "pair"], columns="condition",
                         values="shift_to_realB", aggfunc="mean")
pm_gerak = d.pivot_table(index=["attr_type", "pair"], columns="condition",
                         values="total_movement", aggfunc="mean")

rows, contrasts = [], []
for ty in sorted(d.attr_type.unique()):
    S = pm_shift.loc[ty]
    G = pm_gerak.loc[ty]
    print(f"\n=== {ty} (n_pair={len(S)}) ===")
    for cond in ["true"] + CONTROLS:
        x = S[cond].to_numpy()
        rows.append(dict(type=ty, condition=cond, n_pair=len(x),
                         mean_shift=float(x.mean()), t_vs_0=float(tstat(x)),
                         p_vs_0_2s=p_twosided_exact(x),
                         frac_neg=float((x < 0).mean()),
                         total_movement=float(G[cond].mean())))
        print(f"  {cond:12s} shift={x.mean():+.5f} t={tstat(x):+.2f} "
              f"p2s={p_twosided_exact(x):.4f} (<0: {(x<0).mean():.0%}) "
              f"| movement={G[cond].mean():.4f}")

    print("  -- contrast true vs control (per-pair difference) --")
    for cond in CONTROLS:
        diff = (S["true"] - S[cond]).to_numpy()
        gdiff = (G["true"] - G[cond]).to_numpy()
        contrasts.append(dict(
            type=ty, contrast=f"true - {cond}", n_pair=len(diff),
            mean_diff=float(diff.mean()), t=float(tstat(diff)),
            p_2s=p_twosided_exact(diff),
            frac_true_more_harmful=float((diff < 0).mean()),
            mean_movement_diff=float(gdiff.mean()), t_movement=float(tstat(gdiff)),
            p_movement_2s=p_twosided_exact(gdiff)))
        print(f"  true-{cond:12s} dshift={diff.mean():+.5f} t={tstat(diff):+.2f} "
              f"p2s={p_twosided_exact(diff):.4f} "
              f"| dmovement={gdiff.mean():+.5f} (t={tstat(gdiff):+.2f}, "
              f"p={p_twosided_exact(gdiff):.4f})")

pd.DataFrame(rows).to_csv(f"{OUT}/shuffled_donor_summary.csv", index=False)
pd.DataFrame(contrasts).to_csv(f"{OUT}/shuffled_donor_contrasts.csv", index=False)

# sanity: replication vs finding 08
ref = pd.read_csv(f"{NB08}/causal_pairlevel_patchv2.csv")
ref = ref[(ref.type == "RACExRELIG") & (ref.condition == "patch_L11H16")]
t_new = [r for r in rows if r["type"] == "RACExRELIG" and r["condition"] == "true"][0]["t_vs_0"]
print(f"\n[replication] RACExRELIG H16: finding08 t_vs_0={float(ref.t_vs_0.iloc[0]):.2f} "
      f"| nb17 `true` t={t_new:.2f}")
