"""Payload valid menggunakan median train; bukan kondisi operasi nyata."""
import json
from pathlib import Path

import joblib


def main():
    path = Path("artifacts/model.joblib")
    if not path.is_file():
        raise SystemExit("Jalankan python -m src.latih terlebih dahulu")
    paket = joblib.load(path)
    print(json.dumps({"waktu": "2018-01-01T00:00:00", "sensor": paket["median_fitur"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
