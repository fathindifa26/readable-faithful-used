"""Koreksi max-statistic permutation buat sweep per-layer.

Dipakai dua kali:
  1. data notebook 13 (3 tipe lama) -> harus mereproduksi ambang finding 09
     (2.905 / 2.906 / 2.918). Ini validasi implementasi.
  2. data notebook 14 Part B (3 tipe baru) -> finding 11.

Null: sign-flip berpasangan. Tiap item (pasangan-soal) dikali +1/-1 acak,
flip yang SAMA dipakai di semua 32 layer supaya struktur korelasi antar
layer ikut terjaga. Statistik per layer = t berpasangan (patch vs kontrol
layer-acak); statistik juara = max t atas 32 layer.
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
    d["diff"] = d["shift_ke_realB"] - d["shift_ctrl_ke_realB"]
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
                  "VALIDASI: data notebook 13 (harus cocok finding 09)")
    new = analyse(f"{NB08}/sweep6_rows.csv",
                  "BARU: data notebook 14 Part B (finding 11)")
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        outdir = NB08
        new.to_csv(f"{outdir}/sweep6_maxstat.csv", index=False)
        print("\ndisimpan ->", f"{outdir}/sweep6_maxstat.csv")
