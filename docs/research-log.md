# Research log

A condensed, chronological account of how this project's central claim was
found, why two earlier hypotheses were abandoned, and which experiment
produced which number in the paper. Written for anyone who wants to audit
the trail without reading the full paper first. Section numbers below refer
to `paper/main.tex`.

## The question

Where in an LLM's residual stream does a demographic identity ("Black
Protestant", "Democrat aged 18-29") live, and how faithfully does that
representation track the group's *actual* survey-response distribution
(Pew American Trends Panel, via [OpinionQA](https://arxiv.org/abs/2303.17548))?
And separately: does the model's downstream behavior (the "mouth") actually
*use* that representation, in proportion to how faithful it is?

## Act 1 — a naive readout barely works, and picks the wrong thing to trust

*Pipeline: `01_rsa_pilot`, `02_rsa_intersectional`, `03_lgroup_pilot`.*

- Cosine similarity between mean-pooled hidden states of two demographic
  cells correlates with the two cells' survey-answer similarity, but only
  when the cells share an attribute type — RSA is strongly type-conditional
  (~+0.17 pooled, but AGE and EDUCATION alone reach ~0.50 while
  RACE×RELIGION sits at 0.067, not significant).
- An early pilot happened to select RACE×RELIGION, the *weakest* type — a
  false start that took a while to diagnose.
- The representation is severely anisotropic (avg. pairwise cosine ≈ 0.95).
  Mean-centering roughly doubles pooled RSA but does **not** help the
  within-type kernel that a downstream application would actually use — a
  dead end for the `L_group` loss this project originally set out to build
  (parked; see `notes/findings/percobaan_lgroup/` in the private repo).
- Top-k neighbor precision is noisy across every type (~2x chance, only
  ~2/5 neighbors correct) — a naive last-layer, mean-pooled readout is not
  reliable enough to build on.

**Conclusion of Act 1:** stop reading out identity from the wrong place
(final layer, pooled, cosine) and go looking properly.

## Act 2 — a proper fidelity map: which head, which layer

*Pipeline: `04_fidelity_map`.*

- Per-head, per-layer probing (32×32 heads, cosine RDM vs. the real
  survey-response RDM) beats the pooled residual stream in **all six**
  attribute-pair types (e.g. AGE 0.51→0.74).
- After correcting for the winner's-curse of picking the best head out of
  1024 candidates (max-statistic permutation, held-out template
  replication, split-half reliability — `notebook 10`, the local
  correction/selection step) the signal survives: every type is
  significant, with honest (held-out) fidelity ranging ~0.50 (weakest) to
  ~0.63 (strongest).
- One head recurs as informative across all six types: **layer 11, head
  16** ("L11H16" / `\starhead` in the paper) — the paper's fixed a-priori
  probe location for everything downstream.

**Conclusion of Act 2:** a specific, reproducible location carries a
faithful-to-survey-data representation of demographic identity. Now: does
the model's output actually use it?

## Act 3 — the gap: fidelity does not predict causal use

*Pipeline: `05_patching_v1`, `06_patching_v2`, `07_sweep_and_probe`,
`08_probe_causal_dissociation`, `10_ceiling_instruct_check`,
`11_shuffled_donor_control`.*

This is the paper's core arc, and it reversed direction twice.

1. **v1 patching reads as null** (`05_patching_v1`): a weak lever (1-3
   heads, last token only) moves the output by ~0.004 — indistinguishable
   from zero. Diagnosed as a lever problem, not evidence of "no causal
   path": the lever was too weak and mis-positioned relative to prior work
   (`llm-opinions`).
2. **v2, a stronger lever, shows graded effects** (`06_patching_v2`): with
   a lever matched to that prior work's strength, AGE (high fidelity)
   shows a small but correctly-signed, significant bridge (p=0.0027, ~41%
   of the output ceiling recovered); RACE×RELIGION (low fidelity) shows a
   *single star head that pushes the output in the wrong direction*
   (p≈0.0000!). First pass: this reads as "causal use is graded and
   tracks fidelity."
3. **A 32-layer sweep breaks that story** (`07_sweep_and_probe`, extended
   to 6 types in `08_probe_causal_dissociation`): under max-statistic
   permutation correction across all 32 layers, RACE turns out to have its
   *real* causal locus not at L11 but at **L1** — the earliest layer,
   p=0.0020, stronger than AGE's L11 effect. Precision (small variance)
   beats raw magnitude once corrected properly.
4. **The 6-type sweep kills "graded, tracks fidelity" outright**
   (`08_probe_causal_dissociation`, finding 11): RACE×POLIDEOLOGY — one of
   the *least* faithful types on the Act 2 map — has the **strongest**
   causal locus of all six, at L11 (exact pair-level p=0.0005).
   EDUCATION×INCOME — the *most* faithful type — has **no detectable
   single-layer causal locus anywhere**, despite having the largest raw
   ceiling and output movement available to it. (Every patch here is
   applied at one layer, so redundant encoding across layers stays an
   alternative reading the paper states rather than rules out.) Spearman correlation between fidelity
   and causal strength across three independent fidelity rulers
   (split-half, held-out-template, partial-lexical — and later a fourth,
   QA-context ruler, item 7) is never significantly positive —
   **fidelity does not predict causal use.**
5. **A probe/mouth dissociation** (finding 10, `08_probe_causal_dissociation`):
   a linear probe on L11 reads the map correctly (beats the model's own
   output in all six types, 22-30% closer to ground truth; L11H16 alone,
   128 dimensions, matches the whole layer), but a group-identity-blind
   baseline beats the probe too, and a per-question group-*ordering* test
   sits near the noise floor (7-13% of ceiling). The map is readable, but
   what makes the probe win is question calibration, not knowledge of
   which group is which.
6. **The low-fidelity backfire is not a base-model interface artifact**
   (finding 13, `10_ceiling_instruct_check`): replicating the ceiling
   experiment on instruction-tuned Mistral (vs. base) shows the same
   near-invariance to a full identity swap (3.2% vs. 3.9% of total error).
   Instruction tuning changes *accuracy* (error roughly doubles, Wasserstein
   distance 0.37→0.66) and *direction* (becomes reliably correct in all six
   types, p<0.001) — not identity-dependence.
7. **The map is measured on identity-only prompts, and it weakens where
   the causal experiments actually look** (finding 15,
   `08_probe_causal_dissociation/qa_context_fidelity.py`): Act 2's
   fidelity is scored on a prompt containing an identity and no opinion
   question; Act 3 intervenes in a full QA context at the answer
   position. Re-scoring the same head on the activations already
   collected for the probe — holding question composition constant
   across cells, which matters a great deal here — the map is
   substantially *weaker* in five of six types (EDUCATION×INCOME
   0.67→0.10; RACE×POLPARTY 0.59→0.17), the exception being
   RACE×RELIGION, which rises (0.33→0.49). This is not measurement
   noise: the QA-context RDM is more reliable (split-half 0.98–0.99)
   than the identity-only one (0.72–0.94). So Act 2's numbers describe
   an identity-only prompt and are scoped as such. The dissociation
   itself is unaffected — recomputing the fidelity-vs-causal correlation
   on the QA-context ruler gives exactly what the identity-only ruler
   gives (ρ=+0.03 at the a-priori locus) — so "the wrong map was
   measured" does not explain result 4 away.
8. **The backfire is carried by identity content, not generic corruption**
   (finding 14, `11_shuffled_donor_control`): the RACE×RELIGION backfire
   (t=−2.4, p=0.005, replicated independently here) could in principle be
   "overwriting 128 numbers mid-computation just breaks whatever is
   downstream." A donor-control experiment rules this out: a
   dimension-shuffled donor (same vector, same norm/location statistics,
   structure destroyed) perturbs the output **twice as hard** mechanically
   but produces **zero** systematic drift. Generic corruption predicts the
   opposite. Whether the drift additionally requires the *paired* group's
   identity specifically (vs. any coherent wrong identity) is suggestive
   but not certified at this pair-count (p=0.073).

**Conclusion of Act 3:** the model maintains a demographically faithful
representation of a group's opinions at a specific, replicable location,
and *also* has a causal pathway from a different, unrelated set of
locations into its output — the two are dissociated. Where the causal path
exists, it is not reliably a force for accuracy: it moved a low-fidelity
type's output further from the truth in every diagnostic thrown at it.

## Robustness — does this replicate, and across what?

*Pipeline: `09_robustness_cross_model`.*

- Three checkpoints of a Qwen model family (base → mid-simulation-training
  → fully-trained) replicate the head-over-residual-stream advantage in
  every type at every checkpoint, and replicate the same type hierarchy
  (RACE×RELIGION is always the weakest, failing max-stat correction in all
  three).
- The family has its own analog of L11H16: **L33H9**, informative in 5/6
  types (0.35-0.58) and weakest on the same race-intersecting type — the
  qualitative pattern crosses model families at a *family-specific
  address*, not a fixed layer/head index.
- ~10B tokens of the simulation-specific training barely moved either the
  fidelity map or the output distribution.

## What this repo does *not* claim

- The Act 2 fidelity numbers describe an *identity-only* prompt. In the
  QA context where the model actually answers, the same head is
  measurably less faithful (item 7 above); the paper scopes the claim
  accordingly rather than presenting one number as both.
- A GSS/WVS cross-institution replication was scoped and then dropped from
  this paper (see Limitations in `paper/`) — single-institution (Pew) US
  ground truth is an explicit limitation, not a checked box.
- The paired-group specificity of the finding-14 backfire (item 8 above)
  is reported as suggestive, not settled — 12 donor-recipient pairs is a
  real power ceiling for that particular question.

## Mapping findings → code

| Finding | Pipeline stage | Key result file in `results/` |
|---|---|---|
| 1-4 (Act 1) | `01_rsa_pilot`, `02_rsa_intersectional`, `03_lgroup_pilot` | `rsa_summary.csv`, `rsa_intersectional_summary.csv`, `pilot_summary.csv` |
| 5-6 (Act 2, fidelity map) | `04_fidelity_map` | `peta_kesetiaan_full.csv`, `koreksi_seleksi/cek*.csv` |
| 7-8 (patching v1/v2) | `05_patching_v1`, `06_patching_v2` | `patching_summary.csv`, `patching_v2_summary.csv` |
| 9, 11 (sweep, dissociation) | `07_sweep_and_probe`, `08_probe_causal_dissociation` | `sweep_all6_maxstat.csv`, `causal_pairlevel_sweep.csv`, `causal_typediff_tests.csv` |
| 10 (probe vs. mouth) | `08_probe_causal_dissociation` | `probe_v2_summary.csv`, `probe_v2_grouprank.csv` |
| 12 (cross-model robustness) | `09_robustness_cross_model` | `map_summary.csv`, `mouth_summary.csv` |
| 13 (instruct ceiling) | `10_ceiling_instruct_check` | `ceiling_summary.csv` |
| 14 (donor control) | `11_shuffled_donor_control` | `shuffled_donor_summary.csv`, `shuffled_donor_contrasts.csv` |
| 15 (QA-context fidelity) | `08_probe_causal_dissociation` | `qa_context_fidelity.csv` |

For exact numbers, run the corresponding `analyze` script in
`pipeline/<stage>/` against the CSVs in `results/<stage>/` — see the
top-level `README.md` and `pipeline/README.md` for how the two directories
relate.
