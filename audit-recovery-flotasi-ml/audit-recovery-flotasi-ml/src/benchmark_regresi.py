"""Benchmark hemat komputasi untuk recovery rougher, dengan audit leakage."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FITUR, SUMBER, TARGET, baca_csv, bagi_kronologis, berlabel


def metrik(y, pred):
    pred = np.clip(np.asarray(pred), 0, 100)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "p90_galat_absolut": float(np.quantile(np.abs(y - pred), .90)),
    }


def bootstrap_selisih(y, model, acuan, seed=42, panjang_blok=24, ulangan=800):
    """CI deskriptif selisih MAE model−median; blok berurutan pada baris berlabel."""
    n = len(y)
    rng = np.random.default_rng(seed)
    selisih = []
    for _ in range(ulangan):
        awal = rng.integers(0, max(n-panjang_blok+1, 1), size=int(np.ceil(n/panjang_blok)))
        indeks = np.concatenate([np.arange(i, min(i+panjang_blok, n)) for i in awal])[:n]
        selisih.append(np.mean(np.abs(y[indeks]-model[indeks])) - np.mean(np.abs(y[indeks]-acuan[indeks])))
    return [float(x) for x in np.quantile(selisih, [.025, .975])]


def buat_audit(splits):
    hasil = {}
    for nama, d in zip(("train", "validasi", "test"), splits):
        delta = d["date"].diff().dropna()
        hasil[nama] = {
            "awal": str(d["date"].min()), "akhir": str(d["date"].max()),
            "baris": len(d), "label_ada": int(d[TARGET].notna().sum()),
            "label_hilang": int(d[TARGET].isna().sum()),
            "label_nol": int(d[TARGET].eq(0).sum()),
            "jarak_lebih_1_jam": int((delta > pd.Timedelta("1h")).sum()),
            "fitur_hilang_persen": {c: round(float(d[c].isna().mean()*100), 2) for c in FITUR},
        }
    return hasil


def buat_grafik(hasil, pred, out):
    warna = {"median_train": "#9ca3af", "ridge": "#3b82f6", "hist_gradient_boosting": "#e89c3b"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for nama, data in hasil.items():
        ax.bar(nama.replace("hist_gradient_boosting", "HGBR"), data["mae"], color=warna[nama])
    ax.set_ylabel("MAE test (poin persentase)")
    ax.set_title("Recovery rougher: model vs acuan")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "perbandingan_mae.png", dpi=150)
    plt.close(fig)

    segmen = pred.iloc[:168]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(segmen["date"], segmen["aktual"], label="Hasil tercatat", linewidth=1.3)
    ax.plot(segmen["date"], segmen["model_terpilih"], label="Estimasi ML", linewidth=1.2)
    ax.set_ylabel("Recovery rougher (%)")
    ax.set_title("Contoh 168 observasi berlabel pertama pada periode test")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "contoh_prediksi_test.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/gold_recovery_full.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/regresi"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error("CSV tidak ada. Jalankan python -m src.unduh_data terlebih dulu.")
    assert len(FITUR) == 22 and all(c.startswith(("rougher.input.", "rougher.state.")) for c in FITUR)

    semua = baca_csv(args.csv)
    splits = bagi_kronologis(semua)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (berlabel(d) for d in splits)
    median = float(np.median(y_train))
    model = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100)),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, min_samples_leaf=40,
                                          learning_rate=.06, l2_regularization=1.0,
                                          early_stopping=False, random_state=42),
        ),
    }
    for estimator in model.values():
        estimator.fit(x_train, y_train)

    acuan_val = np.full(len(y_val), median)
    val = {"median_train": metrik(y_val, acuan_val)}
    for nama, estimator in model.items():
        val[nama] = metrik(y_val, estimator.predict(x_val))
    terbaik = min(model, key=lambda nama: val[nama]["mae"])
    terpilih = model[terbaik]
    radius_90 = float(np.quantile(np.abs(y_val - np.clip(terpilih.predict(x_val), 0, 100)), .90))

    prediksi = {"median_train": np.full(len(y_test), median)}
    prediksi.update({nama: np.clip(estimator.predict(x_test), 0, 100) for nama, estimator in model.items()})
    test = {nama: metrik(y_test, p) for nama, p in prediksi.items()}
    batas_bawah = np.clip(prediksi[terbaik]-radius_90, 0, 100)
    batas_atas = np.clip(prediksi[terbaik]+radius_90, 0, 100)
    coverage = float(np.mean((y_test >= batas_bawah) & (y_test <= batas_atas)))

    sampel_val = x_val.iloc[np.linspace(0, len(x_val)-1, min(1000, len(x_val)), dtype=int)]
    y_sampel = y_val[np.linspace(0, len(y_val)-1, min(1000, len(y_val)), dtype=int)]
    penting = permutation_importance(terpilih, sampel_val, y_sampel, n_repeats=3,
                                    random_state=42, scoring="neg_mean_absolute_error", n_jobs=1)
    importansi = pd.DataFrame({"fitur": FITUR, "kenaikan_mae_perm": penting.importances_mean,
                               "simpangan": penting.importances_std}).sort_values("kenaikan_mae_perm", ascending=False)

    audit = buat_audit(splits)
    perkiraan_layak = test[terbaik]["mae"] < test["median_train"]["mae"]
    ringkasan = {
        "sumber": SUMBER, "target": TARGET, "fitur": FITUR,
        "desain": "sensor dan input rougher pada timestamp yang sama; rilis lab belum diketahui",
        "split": audit, "model_terpilih_dari_validasi": terbaik,
        "median_target_train": median, "validasi": val, "test": test,
        "interval_residual_val_90": {
            "radius_poin_persentase": radius_90, "cakupan_empiris_test": coverage,
            "catatan": "Bukan jaminan cakupan masa depan atau interval tersertifikasi di bawah drift temporal."
        },
        "ci95_selisih_mae_model_minus_median_test_blok_24": bootstrap_selisih(
            y_test, prediksi[terbaik], prediksi["median_train"]),
        "mengalahkan_median_test": bool(perkiraan_layak),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    kuantil = x_train.quantile([.005, .995]).fillna(x_train.median())
    paket = {
        "model": terpilih, "fitur": FITUR, "target": TARGET, "model_nama": terbaik,
        "radius_90": radius_90, "median_fitur": x_train.median().to_dict(),
        "batas_bawah": kuantil.loc[.005].to_dict(), "batas_atas": kuantil.loc[.995].to_dict(),
    }
    joblib.dump(paket, args.out / "model.joblib", compress=3)
    (args.out / "metrics.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    importansi.to_csv(args.out / "penting_fitur_validasi.csv", index=False)
    waktu_test = splits[2].loc[splits[2][TARGET].notna(), "date"].reset_index(drop=True)
    pred = pd.DataFrame({
        "date": waktu_test, "aktual": y_test, "median_train": prediksi["median_train"],
        "ridge": prediksi["ridge"], "hist_gradient_boosting": prediksi["hist_gradient_boosting"],
        "model_terpilih": prediksi[terbaik], "batas_bawah_90": batas_bawah, "batas_atas_90": batas_atas,
    })
    pred.to_csv(args.out / "prediksi_test.csv", index=False)
    buat_grafik(test, pred, args.out)

    isi = ["# Laporan eksperimen: recovery rougher (dataset Kaggle)", "",
           "Metrik test dalam poin persentase, kecuali R². Ini bukan hasil AMMAN.", "",
           "| Model | MAE | RMSE | R² | P90 galat |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for nama, nilai in test.items():
        isi.append(f"| {nama} | {nilai['mae']:.3f} | {nilai['rmse']:.3f} | {nilai['r2']:.3f} | {nilai['p90_galat_absolut']:.3f} |")
    isi += ["", f"Model dipilih dari validasi: **{terbaik}**.",
            f"Radius interval residual validasi 90%: **±{radius_90:.3f}** poin; cakupan empiris test: **{coverage:.1%}**.",
            f"Selisih MAE model−median test, CI bootstrap blok 24: {ringkasan['ci95_selisih_mae_model_minus_median_test_blok_24']}.",
            "", "Interpretasi: gunakan sebagai studi metode; tidak mengklaim optimasi operasi atau sebab-akibat dari importansi fitur.",
            "Waktu rilis lab dan kesetaraan sensor dengan AMMAN harus diaudit sebelum uji lapangan."]
    (args.out / "laporan_hasil.md").write_text("\n".join(isi)+"\n", encoding="utf-8")
    print("\n".join(isi))


if __name__ == "__main__":
    main()
