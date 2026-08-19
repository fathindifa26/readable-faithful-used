"""Review round 2, R1: does the dissociation conclusion survive the
choice of fidelity ruler?

Three fidelity estimators per attribute type:
  (a) split-half median (the one used by fig_fidelity_vs_causal),
  (b) partial-lexical (controls for word echo, the 'value' variant -- the
      one reported in app:lexical),
  (c) held-out template.
For each estimator: Spearman vs the pair-level causal t at (i) the
a-priori locus L11, (ii) the winning layer of each type.

Plus an extra A3: direct contrast RELIGxPOLPARTY vs RACExPOLIDEOLOGY
(bootstrap CI of the t difference + 2-sample permutation) -- because under
the partial ruler, RELIGxPP rises to the most faithful type (tied with AGE).

Output: notebooks/output/14_.../r1_ruler_sensitivity.csv (+ print).
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
NB04 = results_dir("04_fidelity_map")
NB07 = results_dir("07_sweep_and_probe")
TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]

pl = pd.read_csv(f"{OUT}/causal_pairlevel_sweep.csv").set_index("type")
lex = pd.read_csv(f"{OUT}/lexical_partial.csv")
lex = lex[lex.lex == "value"].set_index("type")

# (a) split-half median -- same source as the figure
sh = pd.read_csv(f"{NB04}/selection_correction/check3_splithalf.csv")
sh = sh.set_index("attr_type")["heldout_median"]

rulers = {
    "split-half": sh,
    "partial-lexical": lex["partial_star"],
    "held-out-template": None,  # filled in below if available
}
try:
    ho = pd.read_csv(f"{NB04}/selection_correction/check1_heldout_template.csv")
    rulers["held-out-template"] = ho.set_index("attr_type")["heldout_mean"]
except FileNotFoundError:
    del rulers["held-out-template"]

rows = []
for name, fid in rulers.items():
    x = fid[TYPES].to_numpy()
    for lab, col in [("t@L11", "t_L11_pair"), ("t@win", "t_win_pair")]:
        y = pl.loc[TYPES, col].to_numpy()
        r = spearmanr(x, y)
        rows.append(dict(ruler=name, causal=lab,
                         rho=round(float(r.statistic), 3),
                         p=round(float(r.pvalue), 3)))
        print(f"[{name:>18} vs {lab}] rho={r.statistic:+.3f} p={r.pvalue:.3f}")
    print("  fidelity ordering:", " > ".join(fid[TYPES].sort_values(ascending=False).index))

pd.DataFrame(rows).to_csv(f"{OUT}/r1_ruler_sensitivity.csv", index=False)

# ---- extra A3: RELIGxPP vs RACExPI -----------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
d = pd.concat([pd.read_csv(f"{NB07}/sweep_rows.csv"),
               pd.read_csv(f"{OUT}/sweep6_rows.csv")])
d["diff"] = d.shift_to_realB - d.shift_ctrl_to_realB


def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def pairmeans_at(ty, layer):
    g = d[d.attr_type == ty]
    pm = g.pivot_table(index="pair", columns="layer", values="diff", aggfunc="mean")
    return pm[layer].to_numpy()


def diff_test(tyA, layA, tyB, layB, n_boot=10000):
    a, b = pairmeans_at(tyA, layA), pairmeans_at(tyB, layB)
    t_a, t_b = tstat(a), tstat(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (tstat(rng.choice(a, len(a), replace=True))
                    - tstat(rng.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pool = np.concatenate([a, b])
    obs = t_a - t_b
    cnt = 0
    for _ in range(10000):
        perm = rng.permutation(len(pool))
        pa, pb = pool[perm[:len(a)]], pool[perm[len(a):]]
        if abs(tstat(pa) - tstat(pb)) >= abs(obs) - 1e-12:
            cnt += 1
    return dict(A=f"{tyA}@L{layA}", B=f"{tyB}@L{layB}", t_A=round(t_a, 3),
                t_B=round(t_b, 3), dt=round(obs, 3), ci_lo=round(lo, 3),
                ci_hi=round(hi, 3), p_perm2s=cnt / 10000)


print("\n=== extra A3: RELIGxPP (most faithful under the partial ruler) vs RACExPI ===")
tests = [("RELIGxPOLPARTY", 8, "RACExPOLIDEOLOGY", 11),   # each winning layer
         ("RELIGxPOLPARTY", 11, "RACExPOLIDEOLOGY", 11)]  # fixed location L11
res = pd.DataFrame([diff_test(*t) for t in tests])
print(res.to_string(index=False))
res.to_csv(f"{OUT}/r1_religpp_vs_racepi.csv", index=False)
