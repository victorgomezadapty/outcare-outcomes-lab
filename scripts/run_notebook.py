"""Execute notebooks/outcare_outcomes_lab.ipynb in place (offline)."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "outcare_outcomes_lab.ipynb"

if __name__ == "__main__":
    nb = nbformat.read(NB, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbformat.write(nb, NB)
    print("Executed", NB)
