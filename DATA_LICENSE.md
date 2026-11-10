# Data license and attribution

## Data

The only data used in this project is the OutCaRe dataset published on Zenodo:

> Salzwedel, A., Balzer, K., Koran, I., Langheim, E., Schlitt, A., Nothroff, A., Bongarth, C.,
> Wrenger, M., Zoch-Lesniak, B., Wegscheider, K., & Völler, H. (2020). Patient-reported outcomes
> predict return to work and health-related quality of life 6 months after cardiac rehabilitation:
> Results from a German multi-centre registry (OutCaRe): dataset [Dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.3605618

* License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ (Zenodo license id `cc-by-4.0`, access right `open`).
* File: `data/raw/OutCaRe-Datenbank.xlsx`, unmodified copy of the Zenodo file.
* The same workbook and a walkthrough notebook for this analysis are also on Kaggle, as a reuse
  copy of this Zenodo file:
  * Dataset: https://www.kaggle.com/datasets/vctorandrsgmezlpez/outcare-cardiac-rehabilitation-outcomes-registry
  * Notebook: https://www.kaggle.com/code/vctorandrsgmezlpez/outcare-outcomes-lab-cr-change-and-return-to-work
* The tables in `tables/`, figures in `figures/`, `RESULTS.md` and the notebook outputs are
  derived from this dataset and are shared under the same CC BY 4.0 terms with the attribution
  above. Changes made: recoding of 1 = yes / 2 = no variables to 0/1 indicators, derived change
  scores, aggregation and statistical modelling. No rows were added, removed or edited in the
  raw file.

The authors and the licensor do not endorse this project or its interpretation.

## Related article

Please also cite the article that describes the study:

> Salzwedel, A., Koran, I., Langheim, E., Schlitt, A., Nothroff, J., Bongarth, C., et al. (2020).
> Patient-reported outcomes predict return to work and health-related quality of life six months
> after cardiac rehabilitation: Results from a German multi-centre registry (OutCaRe).
> PLOS ONE 15(5): e0232752. https://doi.org/10.1371/journal.pone.0232752 (CC BY 4.0)

The article is used only to cite the study and to compare descriptive results. It is not a
second data source.

## Questionnaires

The dataset contains summary scores from PHQ-9, HAF-17, WHO-5, SF-12, IRES-24, ASKU and the
Würzburg screening. This project uses those published scores only. It does not include,
reproduce or redistribute any questionnaire, item wording or scoring manual. Some of these
instruments (for example SF-12) are licensed by their owners, and their use in new data
collection requires the respective license.

## Code

The code in this repository is licensed under the MIT License (see `LICENSE`). The MIT license
does not apply to the data.

## No other data

No other dataset is used, merged or referenced as input. Any addition of another data source
would require a new license review before it is added.
