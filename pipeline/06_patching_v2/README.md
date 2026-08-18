# 06 — Patching v2 (matched-strength lever)

Same patching design, lever strength matched to prior work
(`llm-opinions`). With a real lever, results are graded rather than null:
AGE (high fidelity) shows a small correctly-signed bridge; RACE-related
types (low fidelity) show a single head that pushes output the *wrong*
direction.

**Careful reading:** on its own this stage's data reads as "causal use
tracks fidelity" — that reading does not survive the properly-corrected
sweep in stage 08 (see `causal_typediff_tests.csv` there). Output:
`results/06_patching_v2/patching_v2_summary.csv` (consumed by
`08_probe_causal_dissociation/causal_cluster_robust.py`).
