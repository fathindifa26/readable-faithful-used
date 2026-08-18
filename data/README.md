# Data

This paper's demographic ground truth is derived from the Pew Research
**American Trends Panel** (ATP), repackaged as
[OpinionQA](https://github.com/tatsu-lab/opinions_qa) (Santurkar et al.,
2023). It is **not** included in this repo: ATP microdata redistribution
terms don't clearly permit it, and even the processed derivative is ~60MB.

## Regenerating `opinionqa_intersectional.csv`

1. Obtain the raw ATP wave data from the OpinionQA repo (its `data/human_resp/`
   directory), or directly from Pew's [ATP dataset pages](https://www.pewresearch.org/politics/collection/american-trends-panel-datasets/).
2. Arrange it as:
   ```
   data/opinionqa_original/data/human_resp/American_Trends_Panel_W<N>/
       responses.csv
       info.csv
   ```
   for each wave `<N>` you have (this project used the six waves covering
   RACE, RELIGION, POLPARTY, POLIDEOLOGY, EDUCATION, INCOME, AGE).
3. Run, from the repo root:
   ```bash
   python data/build_intersectional_cells.py
   ```
   This builds two-attribute intersectional cells (e.g. "White |
   Protestant") from the individual-respondent microdata — the same
   demographic granularity the paper's 2D attribute-pair types need — and
   writes `data/opinionqa_intersectional.csv`.

Every `pipeline/*/analyze*.py` script that needs the survey ground truth
reads it from `data/opinionqa_intersectional.csv` via `pipeline/_common.py`'s
`data_path()`.

If you only want to check the paper's numbers without regenerating this
file, you don't need it: `results/<stage>/` already ships the final CSVs
these scripts produce, so most sanity checks in `docs/research-log.md`'s
mapping table can be read directly.
