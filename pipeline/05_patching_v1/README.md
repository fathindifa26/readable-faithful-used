# 05 — Patching v1 (weak lever, reads as null)

First activation-patching attempt: a weak lever (1-3 heads, last token
only). Result reads as a total null — even replacing the entire
demographic phrase in the prompt moves the output by ~0.004.

**Why it's kept, not deleted:** this initially looked like "the model
doesn't causally use identity at all." Stage 06 shows that conclusion was
an artifact of a lever too weak to move anything, not evidence of no
causal path. Output: `results/05_patching_v1/patching_summary.csv`.
