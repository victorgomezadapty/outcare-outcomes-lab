# OutCaRe Outcomes Lab

A reproducible re-analysis of an open cardiac rehabilitation registry: how walking distance,
training load and patient-reported measures change during cardiac rehabilitation (CR), and
what is associated with return to work (RTW) and health-related quality of life (SF-12)
6 months after CR.

**Status: public.** Repository: https://github.com/victorgomezadapty/outcare-outcomes-lab

## Question

In the German multi-centre OutCaRe registry (1262 working-age patients who answered the
6 month follow-up), how do 6-minute walking distance (6MWD) and other clinical or
patient-reported measures change between CR admission and discharge, and which measures at
admission or discharge are associated with RTW and SF-12 physical (PCS) and mental (MCS)
component scores 6 months later?

## What this project is

* A transparent, offline re-analysis of one open dataset (Zenodo, CC BY 4.0).
* A check of the published descriptive results of Salzwedel et al. (PLOS ONE 2020), recomputed
  from the file, with any differences stated.
* Paired pre/post summaries with confidence intervals and the sample size behind every number.
* Pre-specified regression models for RTW and SF-12, fitted by complete case analysis and by
  multiple imputation, compared with the direction of the paper's findings.

## What this project is not

* Not a replication of the paper's exact model (the paper used backward selection on 20 imputed
  data sets with variables that are not in the open file).
* Not causal. There is no control group; changes during CR and model coefficients are
  associations.
* Not clinical advice, a diagnostic tool or a prediction model for individual patients.
* Not a redistribution of questionnaires. Only the published scores are used.

## Main results

See [RESULTS.md](RESULTS.md) for all numbers. In short:

* n = 1262; mean age 54.2 years; 968 men (76.7%); 864 returned to work (68.5%, 95% CI 65.8 to 71.0).
* 6MWD: paired for 812 patients only; mean change +74.8 m (95% CI 70.7 to 78.8).
  That change already appears in Salzwedel et al. 2020 Table 2 (n = 812, 451.8 to 526.6 m, SES 0.82).
  Added here are the confidence interval, the paired-subset versus rest comparison, and the
  exploratory association of 6MWD gain with 6 month RTW and SF-12 (the paper recorded 6MWD but
  did not put it in the prediction models).
* Endurance training load: paired n = 1204; mean change +21.1 W (95% CI 20.0 to 22.3).
* Pension desire and a negative self-assessed occupational prognosis at discharge show the
  strongest negative associations with RTW, consistent with the paper.

## Project layout

| Path | Content |
|-|-|
| `data/raw/OutCaRe-Datenbank.xlsx` | Unmodified Zenodo file (CC BY 4.0) |
| `src/outcare_lab/` | Loading, hash check, recoding, analysis, figures, report |
| `scripts/run_analysis.py` | Runs everything and regenerates `tables/`, `figures/`, `results/summary.json`, `RESULTS.md` |
| `scripts/run_notebook.py` | Executes the notebook in place |
| `scripts/fetch_data.py` | Re-downloads the file from Zenodo and checks the md5 (only if missing) |
| `notebooks/outcare_outcomes_lab.ipynb` | Walkthrough notebook (offline) |
| `tests/` | Integrity and descriptive tests |
| `tables/`, `figures/` | Aggregate outputs for a later case page |
| `DATA_LICENSE.md`, `PROVENANCE.md` | Source, license, hash and attribution |

## How to run

Python 3.13 was used. From the project root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/run_analysis.py      # about 40 s, 20 imputations
python scripts/run_notebook.py      # executes the notebook
python -m pytest -q                 # tests
```

Everything runs offline on the local workbook. Set `OUTCARE_FAST=1` to use 5 imputations for a
quick check. The multiple imputation uses a fixed seed, so reruns give the same numbers on the
same library versions.

## Methods in brief

* The md5 of the workbook is verified before every load; the analysis stops if it differs.
* Codebook coding: 1 = yes, 2 = no for binary items (recoded to 1/0); sex 1 = male, 2 = female;
  "negative occupational prognosis" 1 = yes means a negative self-assessment.
* Paired change: mean change with t based 95% CI, Wilcoxon signed rank test, standardised effect
  size (mean change divided by SD at admission), McNemar test for binary items.
* Proportions: Wilson 95% CI.
* Models: one pre-specified set of 18 predictors for logistic (RTW) and linear (PCS, MCS)
  regression; complete case and multiple imputation by chained equations (predictive mean
  matching, 20 imputations, Rubin's rules). No variable selection. VIFs reported.

## Limitations

Selection to follow-up (only responders are in the file), German short mostly inpatient CR,
heavy reliance on patient-reported measures, many missing 6MWD values and no centre variable,
observational design. Details in [RESULTS.md](RESULTS.md#5-limitations).

## Data, citation and license

Data: Salzwedel, A., Balzer, K., Koran, I., Langheim, E., Schlitt, A., Nothroff, A., Bongarth, C.,
Wrenger, M., Zoch-Lesniak, B., Wegscheider, K., & Völler, H. (2020). Patient-reported outcomes
predict return to work and health-related quality of life 6 months after cardiac rehabilitation:
Results from a German multi-centre registry (OutCaRe): dataset [Dataset]. Zenodo.
https://doi.org/10.5281/zenodo.3605618. License CC BY 4.0.

Article: Salzwedel A, Koran I, Langheim E, et al. (2020) PLOS ONE 15(5): e0232752.
https://doi.org/10.1371/journal.pone.0232752

This is the only data source. Code: MIT (`LICENSE`). Data and derived outputs: CC BY 4.0
(`DATA_LICENSE.md`).
