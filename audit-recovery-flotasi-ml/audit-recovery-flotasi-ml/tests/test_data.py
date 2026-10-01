"""Uji batas waktu, label hilang, dan larangan fitur pascaproses."""
import numpy as np
import pandas as pd
import pytest

from src.data import FITUR, TARGET, bagi_kronologis, berlabel, baca_csv


def contoh(n=60):
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="h")})
    for f in FITUR:
        d[f] = 1.0
    d[TARGET] = np.linspace(70, 90, n)
    return d


def test_fitur_rougher_tidak_mengandung_output_atau_perhitungan():
    assert len(FITUR) == 22
    assert all(x.startswith(("rougher.input.", "rougher.state.")) for x in FITUR)
    assert TARGET not in FITUR


def test_split_berdasarkan_waktu_sebelum_buang_label_hilang():
    d = contoh()
    d.loc[5, TARGET] = np.nan
    train, val, test = bagi_kronologis(d)
    assert train.date.max() < val.date.min() < test.date.min()
    assert len(train) == 36 and len(val) == 12 and len(test) == 12
    x, y = berlabel(train)
    assert len(x) == 35 and len(y) == 35
    assert list(x.columns) == FITUR


def test_tolak_timestamp_duplikat_dan_target_tidak_fisik(tmp_path):
    d = contoh(30)
    d["date"] = d["date"].dt.strftime("%d/%m/%Y %H:%M")
    path = tmp_path / "data.csv"
    d.loc[1, "date"] = d.loc[0, "date"]
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplikat"):
        baca_csv(path)
    d.loc[1, "date"] = "01/01/2020 01:00"
    d.loc[1, TARGET] = 101
    d.to_csv(path, index=False)
    with pytest.raises(ValueError, match="0–100"):
        baca_csv(path)
