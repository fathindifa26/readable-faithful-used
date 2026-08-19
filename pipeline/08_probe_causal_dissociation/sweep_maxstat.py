"""Max-statistic permutation correction for the per-layer sweep.

Used twice:
  1. notebook 13 data (3 old attribute types) -> must reproduce the finding
     09 thresholds (2.905 / 2.906 / 2.918). This validates the implementation.
  2. notebook 14 Part B data (3 new attribute types) -> finding 11.

Null: paired sign-flip. Every item (pair-question) is multiplied by a random
+1/-1, and the SAME flip is used across all 32 layers so that the correlation
structure between layers is preserved too. Per-layer statistic = paired t
(patch vs random-layer control); winner statistic = max t over 32 layers.
"""
import sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import results_dir

N_PERM = 2000
SEED = 42


def analyse(path, label):
    d = pd.read_csv(path)
    d["diff"] = d["shift_to_realB"] - d["shift_ctrl_to_realB"]
    d["item"] = d["pair"] + " || " + d["qkey"]
    out = []
    for ty, g in d.groupby("attr_type"):
        piv = g.pivot_table(index="item", columns="layer", values="diff")
        piv = piv.dropna()
        M = piv.to_numpy()                      # (n_item, n_layer)
        n, L = M.shape
        t_obs = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n))
        rng = np.random.default_rng(SEED)
        null_max = np.empty(N_PERM)
        for b in range(N_PERM):
            s = rng.choice([-1.0, 1.0], size=(n, 1))
            Mp = M * s
            t = Mp.mean(0) / (Mp.std(0, ddof=1) / np.sqrt(n))
            null_max[b] = t.max()
        win = int(np.argmax(t_obs))
        p = float((null_max >= t_obs.max()).mean())
        out.append(dict(attr_type=ty, n_item=n, n_layer=L, win_layer=piv.columns[win],
                        t_win=t_obs.max(), null_p95=np.percentile(null_max, 95),
                        null_mean=null_max.mean(), p_corrected=p,
                        mean_shift_win=M[:, win].mean(),
                        runner_up=piv.columns[int(np.argsort(t_obs)[-2])],
                        t_runner=np.sort(t_obs)[-2]))
    res = pd.DataFrame(out)
    print(f"\n=== {label} ===")
    print(res.to_string(index=False,
                        formatters={"t_win": "{:.3f}".format, "null_p95": "{:.3f}".format,
                                    "null_mean": "{:.3f}".format, "p_corrected": "{:.4f}".format,
                                    "mean_shift_win": "{:.5f}".format, "t_runner": "{:.3f}".format}))
    return res


if __name__ == "__main__":
    NB07 = results_dir("07_sweep_and_probe")
    NB08 = results_dir("08_probe_causal_dissociation")
    old = analyse(f"{NB07}/sweep_rows.csv",
                  "VALIDATION: notebook 13 data (must match finding 09)")
    new = analyse(f"{NB08}/sweep6_rows.csv",
                  "NEW: notebook 14 Part B data (finding 11)")
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        outdir = NB08
        new.to_csv(f"{outdir}/sweep6_maxstat.csv", index=False)
        print("\nsaved ->", f"{outdir}/sweep6_maxstat.csv")
