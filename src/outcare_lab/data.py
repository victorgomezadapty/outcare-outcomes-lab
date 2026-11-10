"""Loading, integrity checks and recoding for the OutCaRe dataset.

Only data source: Zenodo record 3605618 (DOI 10.5281/zenodo.3605618), CC BY 4.0.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "OutCaRe-Datenbank.xlsx"
EXPECTED_MD5 = "1d739ee05282cb6133ad415fb0290b23"
EXPECTED_BYTES = 459424
EXPECTED_ROWS = 1262
EXPECTED_COLS = 76
ZENODO_DOI = "10.5281/zenodo.3605618"
ZENODO_FILE_URL = "https://zenodo.org/api/records/3605618/files/OutCaRe-Datenbank.xlsx/content"
ARTICLE_DOI = "10.1371/journal.pone.0232752"

# Variables coded 1 = yes, 2 = no in the codebook (sheet "variable").
YES_NO_VARS = [
    "occupation", "sickleave", "arthyp", "hyperlip", "diab", "atrialfib", "pad",
    "depression", "copd", "kidney", "t0smoker", "t1smoker", "t0lifestyle",
    "t1lifestyle", "t0health", "t1health", "t0pension", "t1pension", "t0occupat",
    "t1occupat", "rtw", "t2retired", "t2pension", "t2unemployed", "t2sick",
    "t2hospital", "t2health", "t2capacity",
]

# Column name fragments that would suggest direct identifiers. None should be present.
IDENTIFIER_PATTERNS = [
    "name", "vorname", "nachname", "id", "patient", "birth", "geburt", "dob", "address",
    "adresse", "street", "strasse", "zip", "plz", "postcode", "phone", "telefon",
    "email", "mail", "insurance", "versich", "ssn", "date", "datum", "centre",
    "center", "zentrum", "klinik", "site",
]


def md5sum(path: Path = RAW_PATH) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(path: Path = RAW_PATH) -> None:
    """Raise if the raw file is missing or differs from the Zenodo record."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/fetch_data.py first.")
    size = path.stat().st_size
    digest = md5sum(path)
    if digest != EXPECTED_MD5 or size != EXPECTED_BYTES:
        raise ValueError(
            f"Raw file does not match Zenodo record {ZENODO_DOI}: md5={digest}, bytes={size}"
        )


def load_raw(path: Path = RAW_PATH, verify: bool = True) -> pd.DataFrame:
    if verify:
        verify_raw(path)
    return pd.read_excel(path, sheet_name="database")


def load_codebook(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="variable")


def identifier_like_columns(columns) -> list[str]:
    hits = []
    for c in columns:
        low = str(c).lower()
        # "id" only as a whole token, other patterns as substrings
        for p in IDENTIFIER_PATTERNS:
            if p == "id":
                if low == "id" or low.endswith("_id") or low.startswith("id_"):
                    hits.append(c)
            elif p in low:
                hits.append(c)
    return sorted(set(hits))


def yes_no(series: pd.Series) -> pd.Series:
    """Map 1 = yes / 2 = no coding to 1.0 / 0.0, keep missing as NaN."""
    return series.map({1: 1.0, 2: 0.0})


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis frame with readable 0/1 indicators and change scores.

    Original columns are kept; derived columns get explicit names.
    """
    d = raw.copy()
    for v in YES_NO_VARS:
        d[f"{v}_yes"] = yes_no(d[v])
    d["female"] = (d["sex"] == 2).astype(float)
    d["edu_high"] = np.where(d["education"].isna(), np.nan, (d["education"] == 3).astype(float))
    d["living_alone"] = np.where(d["living"].isna(), np.nan, (d["living"] == 2).astype(float))
    d["acs"] = (d["indication"] == 1).astype(float)
    d["hf_indication"] = (d["indication"] == 3).astype(float)
    d["chronic_admission"] = (d["cradmission"] == 2).astype(float)
    d["outpatient"] = (d["crsetting"] == 2).astype(float)
    pairs = {
        "6mwd": ("t06mwd", "t16mwd"), "training": ("t0training", "t1training"),
        "maxcap": ("t0maxcap", "t1maxcap"), "sys": ("t0sys", "t1sys"),
        "dia": ("t0dia", "t1dia"), "ldl": ("t0ldl", "t1ldl"),
        "phq": ("t0phqsum", "t1phqsum"), "haf": ("t0hafmean", "t1hafmean"),
        "who5": ("t0who5", "t1who5"), "pcs": ("t0pcs12", "t1pcs12"),
        "mcs": ("t0mcs12", "t1mcs12"), "ires_phys": ("t0ires24phys", "t1ires24phys"),
        "ires_subj": ("t0ires24subj", "t1ires24subj"),
        "ires_pain": ("t0ires24pain", "t1ires24pain"), "asku": ("t0askumean", "t1askumean"),
    }
    for k, (a, b) in pairs.items():
        d[f"d_{k}"] = d[b] - d[a]
    return d
