"""Download the OutCaRe workbook from Zenodo and verify its md5.

Only needed if data/raw/OutCaRe-Datenbank.xlsx is missing. The analysis itself runs offline.
Source: Zenodo record 3605618, DOI 10.5281/zenodo.3605618, CC BY 4.0.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from outcare_lab import data as D  # noqa: E402

if __name__ == "__main__":
    D.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if D.RAW_PATH.exists():
        D.verify_raw()
        print("Already present and verified:", D.RAW_PATH)
    else:
        urllib.request.urlretrieve(D.ZENODO_FILE_URL, D.RAW_PATH)
        D.verify_raw()
        print("Downloaded and verified:", D.RAW_PATH)
