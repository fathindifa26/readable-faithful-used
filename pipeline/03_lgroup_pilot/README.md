# 03 — Anisotropy / centering pilot

Characterizes the representation geometry itself: pairwise-cosine
anisotropy (~0.95, severe) and whether mean-centering helps a downstream
top-k retrieval kernel.

**Finding:** centering roughly doubles *pooled* RSA but does not rescue
the *within-type* kernel a real application would use — top-k precision
stays noisy (~2/5 correct) in every type. This is what ended the original
`L_group` loss direction and motivated Act 2's move to a proper per-head,
per-layer map. Output: `results/03_lgroup_pilot/pilot_summary.csv`.
