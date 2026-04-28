"""Download the public M5 Forecasting Accuracy archive from Zenodo."""

from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

from inventory_pricing_rl.data import M5_ZENODO_URL


def main() -> None:
    raw_dir = Path("data/raw/m5")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "m5-forecasting-accuracy.zip"
    if not zip_path.exists():
        print(f"Downloading {M5_ZENODO_URL}")
        urllib.request.urlretrieve(M5_ZENODO_URL, zip_path)
    else:
        print(f"Found existing archive: {zip_path}")
    if not (raw_dir / "calendar.csv").exists():
        print("Extracting archive")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    print(f"M5 data ready in {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
