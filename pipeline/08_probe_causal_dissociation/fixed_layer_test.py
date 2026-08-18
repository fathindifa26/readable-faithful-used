"""Uji lokasi-TETAP di level layer: L11 (dari Babak 2) & L1 (dari finding 09).

Kenapa perlu: sweep memilih layer juara per tipe -> kena winner's curse.
Dua lokasi ini ditetapkan SEBELUM melihat data 3 tipe baru, jadi mengujinya
di keenam tipe bebas dari seleksi per-tipe (analog cek4 di finding 06).

Sekalian hitung langit-langit output per tipe (geseran kalau SELURUH
identitas di prompt diganti) supaya "% langit-langit" bisa dibaca.

Output: notebooks/output/14_.../fixed_layer_L11_L1.csv
"""
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

NB07 = results_dir("07_sweep_and_probe")
OUT = results_dir("08_probe_causal_dissociation")
N_PERM, SEED = 2000, 7

d = pd.concat([pd.read_csv(f"{NB07}/sweep_rows.csv"),
               pd.read_csv(f"{OUT}/sweep6_rows.csv")])
d["diff"] = d.shift_ke_realB - d.shift_ctrl_ke_realB
d["item"] = d.pair + " || " + d.qkey

rows = []
for ty, g in d.groupby("attr_type"):
    piv = g.pivot_table(index="item", columns="layer", values="diff").dropna()
    M = piv.to_numpy()
    n = len(M)
    t = M.mean(0) / (M.std(0, ddof=1) / np.sqrt(n))
    gg = g.drop_duplicates("item")
    ceil = (gg.wd_A_to_realB - gg.wd_B_to_realB).mean()   # ganti seluruh identitas

    rng = np.random.default_rng(SEED)

    def pval(col):
        obs = t[col]
        hit = 0
        for _ in range(N_PERM):
            s = rng.choice([-1.0, 1.0], size=n)
            mm = M[:, col] * s
            if mm.mean() / (mm.std(ddof=1) / np.sqrt(n)) >= obs:
                hit += 1
        return hit / N_PERM

    cols = list(piv.columns)
    i11, i1 = cols.index(11), cols.index(1)
    rows.append(dict(tipe=ty, n=n, langit=ceil,
                     t_L11=t[i11], p_L11=pval(i11), shift_L11=M[:, i11].mean(),
                     t_L1=t[i1], p_L1=pval(i1), shift_L1=M[:, i1].mean(),
                     t_max=t.max(), layer_max=cols[int(np.argmax(t))]))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/fixed_layer_L11_L1.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
