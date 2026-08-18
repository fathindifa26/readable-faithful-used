# 07 — Raw 32-layer sweep + probe features (3 original types)

Raw-data notebook: sweeps every one of 32 layers for a causal effect
(instead of assuming L11), and extracts linear-probe features, for the
three attribute types studied up to this point.

**Finding:** the sweep is what first located RACE's real causal locus at
**L1**, not L11 — a stronger, better-corrected effect than the L11
result. All statistical correction on this raw data happens in
`08_probe_causal_dissociation/sweep_maxstat.py` and
`fixed_layer_test.py`. Outputs: `results/07_sweep_and_probe/sweep_rows.csv`,
`probe_features_L11.csv`.
