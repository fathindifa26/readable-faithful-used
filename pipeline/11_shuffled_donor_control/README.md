# 11 — Donor control for the low-fidelity backfire

`notebook.ipynb` (GPU) re-runs the RACE×RELIGION backfire patch (and
AGE×POLPARTY as a negative control) under four donor conditions: the true
paired-group donor (replication), two other-identity donors, and a
dimension-shuffled donor (same vector, same norm/location statistics,
structure destroyed). `shuffled_donor.py` computes the comparison
locally, including a sanity check against stage 08's original number.

**Finding:** the backfire replicates (t=-2.4, p=0.005). The
dimension-shuffled donor perturbs the output *twice as hard* mechanically
but produces *zero* systematic drift — ruling out "generic activation
corruption" as the explanation; the drift needs coherent identity
content. Paired-group specificity (vs. any wrong identity) is suggestive
but not certified at n=12 pairs (p=0.073). Output:
`results/11_shuffled_donor_control/shuffled_donor_summary.csv`,
`shuffled_donor_contrasts.csv`.
