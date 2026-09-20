"""
Downloads the Instacart Market Basket Analysis dataset via the Kaggle API.

Prerequisites (one-time, manual):
1. Create a Kaggle account if you don't have one.
2. Go to kaggle.com/settings -> API -> "Create New Token".
   This downloads kaggle.json.
3. Place it at: C:\\Users\\<you>\\.kaggle\\kaggle.json
4. This is a *competition* dataset, not a plain dataset, so you must also
   visit https://www.kaggle.com/c/instacart-market-basket-analysis/rules
   and click "I Understand and Accept" — the API download will fail with a
   403 until you've accepted the rules on the site at least once.

Usage:
    python scripts/00_download_data.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
COMPETITION = "instacart-market-basket-analysis"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Invoke the kaggle CLI as a module of the *current* interpreter, so this
    # works whether or not the venv's Scripts/ dir is on PATH.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(RAW_DIR),
        ],
        check=True,
    )

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    # The competition zip nests several more per-table zips — flatten those too.
    for nested_zip in RAW_DIR.glob("*.zip"):
        if nested_zip.name != zip_path.name:
            with zipfile.ZipFile(nested_zip) as zf:
                zf.extractall(RAW_DIR)

    print(f"Done. Files extracted to {RAW_DIR}")


if __name__ == "__main__":
    main()
