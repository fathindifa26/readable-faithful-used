# Pipeline

Eleven stages, in the order they were run. Each `pipeline/<NN_name>/`
folder holds the notebook(s) that generate raw model activations on
Kaggle (GPU, Mistral-7B unless noted) and the local analysis script(s)
that turn them into the CSVs shipped in `results/<NN_name>/`.

Every `analyze*.py` script imports `pipeline/_common.py`, which resolves
`results_dir("<stage>")` and `data_path(...)` relative to the repo root —
run a script from anywhere and it finds its inputs/outputs the same way.
Large intermediate arrays (`*.npz`/`*.npy`, hundreds of MB to a few GB per
stage) are **not** shipped; regenerate them by re-running the notebook and
dropping its output into `results/<stage>/` next to the shipped CSVs.

| Stage | Notebook(s) | Analysis script(s) | Produces (paper reference) |
|---|---|---|---|
| `01_rsa_pilot` | `notebook.ipynb` | — | RSA pilot, Act 1 (finding 1) |
| `02_rsa_intersectional` | `notebook.ipynb` | — | RSA on true intersectional cells (finding 1-2) |
| `03_lgroup_pilot` | `notebook.ipynb` | — | anisotropy / centering pilot (finding 3-4) |
| `04_fidelity_map` | `01_kaggle_build_map.ipynb` (raw activations), `02_local_correction.ipynb` (winner's-curse correction, selection) | — | the head-level fidelity map, Act 2 (finding 5-6) |
| `05_patching_v1` | `notebook.ipynb` | — | weak-lever patching, reads as null (finding 7) |
| `06_patching_v2` | `notebook.ipynb` | — | matched-strength patching, raw graded effect (finding 8) |
| `07_sweep_and_probe` | `notebook.ipynb` | — | raw 32-layer sweep + probe features, 3 original types |
| `08_probe_causal_dissociation` | `notebook.ipynb` | `causal_cluster_robust.py` (pair-level cluster-robust inference — the paper's central statistic), `sweep_maxstat.py` (winner's-curse-corrected layer sweep), `fixed_layer_test.py` (pre-registered L11/L1 test), `probe_v2.py` + `probe_grouprank.py` + `grouprank_ceiling.py` + `probe_fairness.py` (the probe-vs-mouth dissociation, finding 10), `dreal_sensitivity.py`, `head_vs_residual_fair.py`, `lexical_baseline.py`, `rsa_reliability_ceiling.py` (robustness checks on the fidelity map itself), `r1_ruler_sensitivity.py` (review response: fidelity-ruler sensitivity, §6) | Act 3's core tables (finding 9-11) |
| `09_robustness_cross_model` | `notebook.ipynb` | `robustness_map.py` | cross-model-family replication (finding 12) |
| `10_ceiling_instruct_check` | `notebook.ipynb` | `ceiling_instruct.py` | base-vs-instruct ceiling replication (finding 13) |
| `11_shuffled_donor_control` | `notebook.ipynb` | `shuffled_donor.py` | donor-control for the low-fidelity backfire (finding 14) |

`08_probe_causal_dissociation` is the biggest stage because it's where the
paper's central claim lives: one raw-data notebook, twelve analysis
scripts, almost all of Act 3. That's not clutter to be trimmed — it's an
accurate reflection of how much local statistical correction the raw
patching/sweep data needed before it supported a claim.

For the story these stages tell in order — including the two hypotheses
that got overturned along the way — see `../docs/research-log.md`.
