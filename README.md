# Readable, Faithful, Used

Code and data pipeline for **"Readable, Faithful, Used: Three Dissociable
Properties of Demographic Identity in a Language Model."**

Paper PDF: [`paper/main.pdf`](paper/main.pdf) (arXiv link to be added on
submission). Author: Fathin Difa Robbani, independent researcher
([ORCID 0009-0000-6184-8919](https://orcid.org/0009-0000-6184-8919)).

## What this paper finds

LLMs are widely used to simulate survey respondents, but their answers are
homogeneous and unfaithful to real inter-group differences. Using
representational similarity analysis against Pew American Trends Panel
ground truth (169 intersectional demographic cells, e.g. "Black
Protestant"), we score 1,089 read-out locations in Mistral-7B and then
intervene causally at every layer across six demographic attribute-pair
types. Four headline results:

1. **A single attention head reads out group identity faithfully**
   (selection-corrected fidelity up to ρ=0.63) — better than the standard
   last-token residual read-out, and better than a lexical-similarity
   baseline can explain. It replicates at a family-specific address in an
   independent model family.
2. **Causal use does not follow fidelity.** The strongest causal pathway
   into the model's output sits in one of the *least* faithful types
   (exact p=0.002); the *most* faithful type has no detectable causal
   locus at all.
3. Where a causal pathway exists for a low-fidelity type, it can move the
   output *away* from the truth — and a donor-control experiment shows
   this is carried by identity content, not generic activation corruption.
4. A linear probe on the faithful head lands 22-30% closer to survey
   truth than the model's own answers, yet cannot recover per-question
   group ordering — the map is readable without being usable that way.

The full argument, all six attribute types, and every caveat are in the
paper. `docs/research-log.md` is a shorter, code-linked walkthrough of how
these results were found, in the order they were found (including two
earlier hypotheses that turned out to be wrong).

## Repo map

```
paper/      LaTeX source + compiled PDF
pipeline/   one folder per experiment stage: the Kaggle notebook(s) that
            generate raw model activations, and the local analysis
            script(s) that turn them into the numbers/tables in the paper
results/    the small CSV/PNG outputs those scripts produce (~30MB total) —
            what's actually cited in the paper, so you can check a number
            without a GPU
data/       instructions + script to rebuild the survey ground-truth table
            (not shipped — see data/README.md)
docs/       research-log.md: the findings, in order, with caveats
```

See `pipeline/README.md` for the full stage-by-stage map and the
reproduction convention every script follows.

## Quickstart

```bash
pip install -r requirements.txt

# Check a result that doesn't need a GPU or the raw survey data:
python pipeline/11_shuffled_donor_control/shuffled_donor.py

# Rebuild the causal-fidelity dissociation tables (needs data/, see data/README.md):
python pipeline/08_probe_causal_dissociation/causal_cluster_robust.py
```

Re-running a `*_kaggle.ipynb` notebook (GPU, Mistral-7B) regenerates the
raw activations under `results/<stage>/`; the paired `analyze*.py` script
in the same stage folder then reproduces the paper's statistics from them.
Cross-checking the paper's numbers doesn't require the GPU step — all the
final CSVs already ship in `results/`.

## License

Code: MIT (`LICENSE`). Paper text and figures: CC BY 4.0.

## Citation

```bibtex
@misc{robbani2026readable,
  title  = {Readable, Faithful, Used: Three Dissociable Properties of
            Demographic Identity in a Language Model},
  author = {Robbani, Fathin Difa},
  year   = {2026},
  note   = {Preprint}
}
```
