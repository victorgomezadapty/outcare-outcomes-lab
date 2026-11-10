# Data provenance

| Field | Value |
|-|-|
| Source | Zenodo record 3605618 |
| DOI | 10.5281/zenodo.3605618 |
| Title | Patient-reported outcomes predict return to work and health-related quality of life 6 months after cardiac rehabilitation: Results from a German multi-centre registry (OutCaRe): dataset |
| Published on Zenodo | 2020-01-12 |
| License | CC BY 4.0 (Zenodo license id `cc-by-4.0`, access right `open`) |
| File | `OutCaRe-Datenbank.xlsx` |
| Download URL | https://zenodo.org/api/records/3605618/files/OutCaRe-Datenbank.xlsx/content |
| md5 | `1d739ee05282cb6133ad415fb0290b23` (matches the checksum published by Zenodo) |
| Size | 459424 bytes |
| Sheets | `database` (1262 rows x 76 coded variables), `variable` (codebook, 76 rows) |
| Identifiers | No names, dates, centre codes or other direct identifiers in the file |
| Trial registry | DRKS00011418 states individual participant data are shared in anonymized form via this Zenodo dataset |
| Article | Salzwedel et al., PLOS ONE 2020, https://doi.org/10.1371/journal.pone.0232752 |
| Kaggle dataset | https://www.kaggle.com/datasets/vctorandrsgmezlpez/outcare-cardiac-rehabilitation-outcomes-registry |
| Kaggle notebook | https://www.kaggle.com/code/vctorandrsgmezlpez/outcare-outcomes-lab-cr-change-and-return-to-work |

The Kaggle pages host this same Zenodo file and the analysis notebook for reuse. Loading and the md5 check still use only the Zenodo file above.

## Rules for this project

1. This Zenodo file is the only data source. No other dataset is added, merged or used for
   comparison except the published summary numbers in the article above.
2. The hash is checked every time the data are loaded (`outcare_lab.data.verify_raw`), in the
   analysis script, the notebook and the tests. The analysis stops if it does not match.
3. Reports, tables and figures contain aggregates only. No row level extracts are written.
   Categorical counts below 5 are shown as "<5" in descriptive tables.
4. Attribution as in `DATA_LICENSE.md` must accompany any reuse of the data or derived results.
