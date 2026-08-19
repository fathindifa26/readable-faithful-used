# 04 — The fidelity map (Act 2)

- `01_kaggle_build_map.ipynb` (GPU): runs Mistral-7B over 169
  intersectional demographic cells, captures every attention head's
  output and the residual stream at every layer, and computes per-head /
  per-layer cosine RDMs against the real survey-response RDM.
- `02_local_correction.ipynb` (CPU, local): winner's-curse correction —
  max-statistic permutation over 1,024 head candidates, held-out-template
  replication, split-half reliability — needed because "best head out of
  1,024" is a biased estimate on its own.

**Finding:** attention heads beat the residual stream in all six types;
one head, **layer 11 / head 16**, is significantly faithful as a *fixed*
location across all six types (the paper's `\starhead`). Honest
(held-out) fidelity ranges ρ≈0.50–0.63 across types. Outputs:
`results/04_fidelity_map/fidelity_map_full.csv` and
`results/04_fidelity_map/selection_correction/check*.csv`.
