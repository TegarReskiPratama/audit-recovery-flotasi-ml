"""Kontrak fitur dan pemisahan data proses berdasarkan waktu."""
from pathlib import Path

import numpy as np
import pandas as pd


SUMBER = "manuelantoniocherres/gold-recovery-full"
TARGET = "rougher.output.recovery"
FITUR = [
    "rougher.input.feed_ag", "rougher.input.feed_pb", "rougher.input.feed_rate",
    "rougher.input.feed_size", "rougher.input.feed_sol", "rougher.input.feed_au",
    "rougher.input.floatbank10_sulfate", "rougher.input.floatbank10_xanthate",
    "rougher.input.floatbank11_sulfate", "rougher.input.floatbank11_xanthate",
    *[f"rougher.state.floatbank10_{i}_{jenis}" for i in "abcdef" for jenis in ("air", "level")],
]


def baca_csv(path: str | Path) -> pd.DataFrame:
    """Baca hanya 22 fitur yang tersedia di rougher serta target untuk audit."""
    df = pd.read_csv(path, usecols=["date", *FITUR, TARGET], low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce")
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError("Timestamp hilang, tak terbaca, atau duplikat")
    for kolom in [*FITUR, TARGET]:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("date").reset_index(drop=True)
    tidak_fisik = df[TARGET].notna() & ~df[TARGET].between(0, 100)
    if tidak_fisik.any():
        raise ValueError(f"Ada {tidak_fisik.sum()} recovery di luar 0–100%")
    return df


def bagi_kronologis(df: pd.DataFrame, prop_train=0.60, prop_val=0.20):
    if not (0 < prop_train < 1 and 0 < prop_val < 1 and prop_train + prop_val < 1):
        raise ValueError("Proporsi split tidak sah")
    if df["date"].isna().any() or not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("Data harus memiliki timestamp unik dan terurut")
    a, b = int(len(df) * prop_train), int(len(df) * (prop_train + prop_val))
    hasil = (df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy())
    if any(len(x) < 10 or x[TARGET].notna().sum() < 10 for x in hasil):
        raise ValueError("Setiap split membutuhkan sedikitnya 10 label valid")
    return hasil


def berlabel(df: pd.DataFrame):
    return df.loc[df[TARGET].notna(), FITUR], df.loc[df[TARGET].notna(), TARGET].to_numpy(dtype=float)
