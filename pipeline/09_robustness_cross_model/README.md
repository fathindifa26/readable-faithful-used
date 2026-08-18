# 09 — Cross-model-family robustness

`notebook.ipynb` (GPU) replicates the Act 2 fidelity map and Act 1 mouth
check on three checkpoints of a second model family (Qwen: base →
mid-simulation-training → fully-trained). `robustness_map.py` computes
the map/mouth statistics locally, the same way stage 04/08 do for
Mistral.

**Finding:** the head-over-residual advantage, the type hierarchy
(RACE×RELIGION always weakest), and an analogous fixed-location head
(**L33H9** for this family) all replicate — at a family-specific address,
not a fixed layer/head index. ~10B tokens of simulation-specific training
barely moves either the map or the output. Output:
`results/09_robustness_cross_model/map_summary.csv`,
`mouth_summary.csv`.
