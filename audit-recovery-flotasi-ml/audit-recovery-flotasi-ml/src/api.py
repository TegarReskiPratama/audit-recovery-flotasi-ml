"""Alarm demonstrasi recovery rougher; bukan kendali proses otomatis."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .data import FITUR


app = FastAPI(title="Alarm Risiko Recovery Rougher", version="1.0.0",
              description="Klasifikasi pada data publik; bukan alarm operasional AMMAN.")


class Permintaan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    waktu: datetime
    sensor: dict[str, float]


@lru_cache(maxsize=1)
def muat_paket():
    path = Path(os.getenv("MODEL_DIR", "artifacts")) / "model.joblib"
    paket = joblib.load(path)
    if paket["fitur"] != FITUR:
        raise ValueError("Kontrak fitur model tidak cocok dengan kode API")
    return paket


@app.get("/health")
def health():
    ada = (Path(os.getenv("MODEL_DIR", "artifacts")) / "model.joblib").is_file()
    return {"status": "siap" if ada else "model_belum_tersedia"}


@app.get("/fitur")
def fitur():
    return {"jumlah": len(FITUR), "nama": FITUR,
            "keterangan": "Input dan state rougher; perlu pemetaan ulang untuk tiap pabrik."}


@app.post("/prediksi")
def prediksi(body: Permintaan):
    if body.waktu.tzinfo is not None:
        raise HTTPException(status_code=422, detail="Gunakan timestamp lokal tanpa zona waktu sesuai data sumber")
    if set(body.sensor) != set(FITUR):
        kurang = sorted(set(FITUR) - set(body.sensor))
        lebih = sorted(set(body.sensor) - set(FITUR))
        raise HTTPException(status_code=422, detail={"fitur_kurang": kurang, "fitur_tidak_dikenal": lebih})
    try:
        paket = muat_paket()
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Model belum tersedia: {e}") from e
    nilai = np.asarray([body.sensor[k] for k in FITUR], dtype=float)
    if not np.isfinite(nilai).all():
        raise HTTPException(status_code=422, detail="Semua nilai sensor harus finite")
    bawah = np.array([paket["batas_bawah"][k] for k in FITUR])
    atas = np.array([paket["batas_atas"][k] for k in FITUR])
    tidak_didukung = [FITUR[i] for i in np.where((nilai < bawah) | (nilai > atas))[0]]
    if tidak_didukung:
        raise HTTPException(status_code=422, detail={"sensor_di_luar_rentang_train": tidak_didukung})
    x = pd.DataFrame([body.sensor], columns=FITUR)
    skor = float(paket["model"].predict_proba(x)[0, 1])
    return {
        "waktu": body.waktu.isoformat(), "skor_risiko": skor,
        "alarm_tinjau_operator": skor >= paket["ambang_skor"],
        "ambang_skor": paket["ambang_skor"],
        "definisi_kejadian": f"recovery rougher <{paket['batas_recovery']:.0f}% (ilustrasi)",
        "model": paket["model_nama"],
        "catatan": "Skor belum dikalibrasi sebagai probabilitas. Perlukan verifikasi lab, waktu rilis, dan keputusan insinyur proses.",
    }
