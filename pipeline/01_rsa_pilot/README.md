# 01 — RSA pilot

`notebook.ipynb` computes representational similarity (cosine RDM of
mean-pooled hidden states vs. real survey-response RDM) per layer, on
single-attribute demographic groups.

**Finding:** RSA is positive (~0.17 pooled) only when comparing cells of
the *same* attribute type; it is type-conditional, not a general
demographic signal. See `../../docs/research-log.md` (Act 1) and
`results/01_rsa_pilot/rsa_summary.csv`.
