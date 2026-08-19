# Pipeline

Eleven stages, in the order they were run. Each `pipeline/<NN_name>/`
folder holds the notebook(s) that generate raw model activations on
Kaggle (GPU, Mistral-7B unless noted) and the local analysis script(s)
that turn them into the CSVs shipped in `results/<NN_name>/`.

Every analysis script imports `pipeline/_common.py`, which resolves
`results_dir("<stage>")` and `data_path(...)` relative to the repo root —
run a script from anywhere and it finds its inputs/outputs the same way.

## What runs without a GPU or extra downloads

Five scripts run against nothing but what this repo ships — and they are
the ones behind the paper's central causal claims, so the main results
can be checked immediately:

```bash
python pipeline/08_probe_causal_dissociation/causal_cluster_robust.py   # the primary pair-level inference
python pipeline/08_probe_causal_dissociation/sweep_maxstat.py           # winner's-curse-corrected layer sweep
python pipeline/08_probe_causal_dissociation/fixed_layer_test.py        # pre-registered L11/L1 test
python pipeline/08_probe_causal_dissociation/r1_ruler_sensitivity.py    # fidelity-ruler sensitivity
python pipeline/11_shuffled_donor_control/shuffled_donor.py             # donor control (finding 14)
```

The remaining scripts need one of two things, and say so on stderr with
the exact path they wanted:

| Missing input | Scripts | How to get it |
|---|---|---|
| `data/opinionqa_intersectional.csv` | `probe_v2`, `probe_grouprank`, `probe_fairness`, `grouprank_ceiling`, `ceiling_instruct` | rebuild it — see [`data/README.md`](../data/README.md) |
| `results/04_fidelity_map/emb_heads.npz` (and `emb_resid.npz`) — ~160MB each | `dreal_sensitivity`, `head_vs_residual_fair`, `lexical_baseline`, `rsa_reliability_ceiling`, `qa_context_fidelity`, `robustness_map` | re-run `pipeline/04_fidelity_map/01_kaggle_build_map.ipynb` (GPU) and drop the output into `results/04_fidelity_map/` |

Small arrays **are** shipped where they unlock analysis cheaply: the
survey ground-truth distance matrix (`group_real_dist.npy`, 224KB — the
RSA target everything is scored against) and the per-type probe
predictions (`probe_v2_pred_*.npz`, ~3MB total). The large per-head
activation dumps (11MB–500MB each, several GB total) are not.

| Stage | Notebook(s) | Analysis script(s) | Produces (paper reference) |
|---|---|---|---|
| `01_rsa_pilot` | `notebook.ipynb` | — | RSA pilot, Act 1 (finding 1) |
| `02_rsa_intersectional` | `notebook.ipynb` | — | RSA on true intersectional cells (finding 1-2) |
| `03_lgroup_pilot` | `notebook.ipynb` | — | anisotropy / centering pilot (finding 3-4) |
| `04_fidelity_map` | `01_kaggle_build_map.ipynb` (raw activations), `02_local_correction.ipynb` (winner's-curse correction, selection) | — | the head-level fidelity map, Act 2 (finding 5-6) |
| `05_patching_v1` | `notebook.ipynb` | — | weak-lever patching, reads as null (finding 7) |
| `06_patching_v2` | `notebook.ipynb` | — | matched-strength patching, raw graded effect (finding 8) |
| `07_sweep_and_probe` | `notebook.ipynb` | — | raw 32-layer sweep + probe features, 3 original types |
| `08_probe_causal_dissociation` | `notebook.ipynb` | `causal_cluster_robust.py` (pair-level cluster-robust inference — the paper's central statistic), `sweep_maxstat.py` (winner's-curse-corrected layer sweep), `fixed_layer_test.py` (pre-registered L11/L1 test), `probe_v2.py` + `probe_grouprank.py` + `grouprank_ceiling.py` + `probe_fairness.py` (the probe-vs-mouth dissociation, finding 10), `dreal_sensitivity.py`, `head_vs_residual_fair.py`, `lexical_baseline.py`, `rsa_reliability_ceiling.py` (robustness checks on the fidelity map itself), `r1_ruler_sensitivity.py` (fidelity-ruler sensitivity), `qa_context_fidelity.py` (is the map still faithful in the QA context the causal experiments run in? finding 15) | Act 3's core tables (finding 9-11, 15) |
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
