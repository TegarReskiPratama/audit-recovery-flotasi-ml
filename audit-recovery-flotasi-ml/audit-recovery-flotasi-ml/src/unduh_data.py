"""Ambil mirror Kaggle; CSV mentah tidak didistribusikan dalam repo."""
from pathlib import Path
import shutil

from .data import SUMBER


def main():
    import kagglehub

    asal = Path(kagglehub.dataset_download(SUMBER)) / "gold_recovery_full.csv"
    if not asal.is_file():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {asal}")
    tujuan = Path("data/gold_recovery_full.csv")
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asal, tujuan)
    print(f"Dataset siap: {tujuan} ({tujuan.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
