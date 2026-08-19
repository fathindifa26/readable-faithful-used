# 08 — Probe/causal dissociation (Act 3's core)

Raw-data notebook (`notebook.ipynb`) extends the stage-07 sweep and probe
extraction to all six attribute types. Every other file here is a local
analysis script consuming that raw data (plus stage 04/06/07 outputs) —
this is where almost all of Act 3's actual statistics are computed:

| Script | Answers |
|---|---|
| `causal_cluster_robust.py` | the paper's central inference: pair-level, cluster-robust (exact sign-flip over 2^12 combinations) causal test, per type |
| `sweep_maxstat.py` | winner's-curse-corrected layer sweep (validates against stage 07's 3-type result, then runs the new 3 types) |
| `fixed_layer_test.py` | the pre-registered L11/L1 test, decided *before* seeing the new types, so it's free of per-type selection |
| `probe_v2.py`, `probe_grouprank.py`, `grouprank_ceiling.py`, `probe_fairness.py` | the probe-vs-mouth dissociation: does a linear probe on the fidelity map beat the model's own answers, and can it recover per-question group ordering? |
| `dreal_sensitivity.py`, `head_vs_residual_fair.py`, `lexical_baseline.py`, `rsa_reliability_ceiling.py` | robustness checks on the Act 2 map itself (does it survive controlling for lexical similarity, random projections, measurement-reliability ceilings?) |
| `r1_ruler_sensitivity.py` | is "fidelity doesn't predict causal use" sensitive to which fidelity estimator you pick? (split-half / partial-lexical / held-out — no) |
| `qa_context_fidelity.py` | the map is scored on identity-only prompts, but the causal experiments run in a QA context — is it still faithful *there*? (finding 15) |

**Findings:** fidelity does not predict causal strength (Spearman never
significantly positive across three rulers); RACE×POLIDEOLOGY (low
fidelity) has the strongest causal locus of all six types; EDUCATION×INCOME
(highest fidelity) has none; a probe on the fidelity map beats the model's
own mouth on accuracy but not on group ordering.

`qa_context_fidelity.py` is worth reading for its own sake: it shows the
map is substantially *weaker* in the QA context (e.g. EDUCATION×INCOME
0.67→0.10) once question composition is held constant across cells — a
control that matters, because the naive version of the same measurement
(each cell averaged over its own question subset) reports the opposite.
The fidelity-vs-causal correlation is unchanged under either ruler
(ρ=+0.03 at the a-priori locus), so the dissociation does not rest on
which context the map is measured in. See `../../docs/research-log.md`
for the full arc.
