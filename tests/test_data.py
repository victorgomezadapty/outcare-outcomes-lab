"""Basic integrity and descriptive checks for the OutCaRe Outcomes Lab."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outcare_lab import analysis as A  # noqa: E402
from outcare_lab import data as D  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return D.load_raw()


@pytest.fixture(scope="module")
def d(raw):
    return D.prepare(raw)


def test_hash_and_size():
    assert D.RAW_PATH.stat().st_size == D.EXPECTED_BYTES
    assert D.md5sum() == D.EXPECTED_MD5


def test_shape(raw):
    assert raw.shape == (1262, 76)


def test_sheets_present():
    cb = D.load_codebook()
    assert len(cb) == 76
    assert list(cb["varname"]) == list(D.load_raw().columns)


def test_no_identifier_like_columns(raw):
    assert D.identifier_like_columns(raw.columns) == []


def test_identifier_check_catches_examples():
    assert D.identifier_like_columns(["patient_id", "Name", "birthdate", "age"]) == ["Name", "birthdate", "patient_id"]


def test_age_and_sex(raw):
    assert raw["age"].min() == 21 and raw["age"].max() == 65
    assert round(raw["age"].mean()) == 54
    assert (raw["sex"] == 1).sum() == 968
    assert (raw["sex"] == 2).sum() == 294


def test_rtw(d):
    assert int(d["rtw_yes"].sum()) == 864
    assert d["rtw_yes"].notna().all()
    assert round(100 * d["rtw_yes"].mean(), 1) == 68.5


def test_yes_no_coding(raw, d):
    for v in D.YES_NO_VARS:
        assert set(raw[v].dropna().unique()) <= {1, 2}, v
        assert set(d[f"{v}_yes"].dropna().unique()) <= {0.0, 1.0}, v


def test_6mwd_paired_count_and_direction(d):
    cont = A.continuous_change(d).set_index("measure")
    row = cont.loc["6MWD (m)"]
    assert row.n_paired == 812
    assert row.ci_low > 0


def test_training_change_matches_paper(d):
    cont = A.continuous_change(d).set_index("measure")
    assert abs(cont.loc["Endurance training load (W)", "mean_change"] - 21) < 1


def test_followup_sf12(d):
    assert d["t2pcs12"].notna().sum() == 1181
    assert round(d["t2pcs12"].mean(), 1) == 45.5
    assert round(d["t2mcs12"].mean(), 1) == 48.9


def test_small_cells_masked():
    assert A.fmt_n(3) == "<5" and A.fmt_n(0) == "0" and A.fmt_n(12) == "12"


def test_outputs_are_aggregates():
    """Generated tables must not contain one row per patient."""
    tables = ROOT / "tables"
    if not tables.exists():
        pytest.skip("run scripts/run_analysis.py first")
    import pandas as pd
    for f in tables.glob("*.csv"):
        t = pd.read_csv(f)
        assert len(t) < 200, f.name
        assert "age" not in t.columns or len(t) < 20, f.name


def test_no_double_or_long_dashes_in_prose():
    files = list(ROOT.glob("*.md")) + list((ROOT / "notebooks").glob("*.ipynb"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        for bad in ("--", "\u2013", "\u2014"):
            assert bad not in text, f"{f.name} contains {bad!r}"
