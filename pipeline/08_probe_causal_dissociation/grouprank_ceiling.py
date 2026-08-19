"""Reliability ceiling of the "group ordering per question" metric.

Why it is needed: probe & the model's own answers both get an ordering
correlation of ~0.05-0.10. Small -- but small compared to WHAT? The survey
data itself has sampling noise (each cell has only n~132 respondents), so
this metric has an upper bound < 1.

How: build TWO independent survey samples from each cell's distribution
(multinomial of the original n_unweighted size), then run the same test
between those two samples. The result = how high this metric could go if
the read-out were perfect.

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
    rows.append(dict(type=ty, n_questions=len(out), reliability_ceiling=float(np.mean(out))))
    print(f"[{ty}] ceiling = {np.mean(out):+.3f} (n questions={len(out)})")

pd.DataFrame(rows).to_csv(f"{OUT}/grouprank_noise_ceiling.csv", index=False)
