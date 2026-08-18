"""Langit-langit reliabilitas metrik "urutan kelompok per soal".

Kenapa perlu: probe & mulut sama-sama dapat korelasi urutan ~0.05-0.10.
Kecil -- tapi kecil dibanding APA? Data survei sendiri punya noise sampling
(tiap sel cuma n~132 responden), jadi metrik ini punya batas atas < 1.

Cara: buat DUA sampel survei independen dari distribusi tiap sel (multinomial
sebesar n_unweighted asli), lalu jalankan uji yang sama antar dua sampel itu.
Hasilnya = setinggi apa metrik ini bisa dicapai kalau read-out-nya sempurna.

Output: notebooks/output/14_.../grouprank_noise_ceiling.csv
"""
import ast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _common import data_path, results_dir

OUT = results_dir("08_probe_causal_dissociation")
TYPES = ["AGExPOLPARTY", "EDUCATIONxINCOME", "RELIGxPOLPARTY",
         "RACExPOLPARTY", "RACExPOLIDEOLOGY", "RACExRELIG"]

raw = pd.read_csv(data_path("opinionqa_intersectional.csv"))
raw["ordinal"] = raw["ordinal"].apply(ast.literal_eval)
raw["responses"] = raw["responses"].apply(ast.literal_eval)
raw["gk"] = raw["attribute"] + " :: " + raw["group"]

rng = np.random.default_rng(0)
rows = []
for ty in TYPES:
    df = pd.read_csv(f"{OUT}/probeA_{ty}.csv")
    sub = raw[raw.attribute == ty]
    key = {(r.gk, r.qkey): (np.array(r.responses, float), np.array(r.ordinal, float),
                            r.n_unweighted) for r in sub.itertuples()}
    out = []
    for q, g in df.groupby("qk"):
        cells = g.gk.tolist()
        if len(cells) < 5:
            continue
        a, b = [], []
        for c in cells:
            p, o, n = key[(c, q)]
            p = p / p.sum()
            n = int(max(n, 5))
            a.append((rng.multinomial(n, p) / n) @ o)
            b.append((rng.multinomial(n, p) / n) @ o)
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        r = spearmanr(a, b).statistic
        if not np.isnan(r):
            out.append(r)
    rows.append(dict(tipe=ty, n_soal=len(out), langit_reliabilitas=float(np.mean(out))))
    print(f"[{ty}] langit-langit = {np.mean(out):+.3f} (n soal={len(out)})")

pd.DataFrame(rows).to_csv(f"{OUT}/grouprank_noise_ceiling.csv", index=False)
