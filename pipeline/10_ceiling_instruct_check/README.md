# 10 — Base vs. instruct ceiling check

`notebook.ipynb` (GPU) repeats the Act 3 "ceiling" experiment — replacing
a cell's *entire* identity phrase and measuring how much the output
moves — on both base Mistral-7B and Mistral-7B-Instruct-v0.2, same cells,
same questions. `ceiling_instruct.py` computes the comparison locally.

**Finding:** near-invariance to a full identity swap replicates on the
instruct model too (3.2% vs. 3.9% of total error) — so it isn't a
base-model interface artifact. Instruction tuning changes accuracy (error
roughly doubles) and direction (becomes reliably correct everywhere), not
identity-dependence. Output:
`results/10_ceiling_instruct_check/ceiling_summary.csv`.
