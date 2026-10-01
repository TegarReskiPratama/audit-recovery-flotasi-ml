"""Deteksi risiko recovery rougher rendah dengan evaluasi kronologis."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark_regresi import buat_audit
from .data import FITUR, SUMBER, TARGET, baca_csv, bagi_kronologis, berlabel


AMBANG_RENDAH = 75.0  # contoh tujuan analisis; bukan batas operasi AMMAN
PORSI_ALERT_VALIDASI = .02  # kapasitas peninjauan ilustratif, bukan SLA pabrik
SEED = 42


def nilai_metrik(y, skor, batas):
    alarm = np.asarray(skor) >= batas
    prevalensi = float(np.mean(y))
    presisi = float(precision_score(y, alarm, zero_division=0))
    return {
        "jumlah": len(y), "kejadian": int(np.sum(y)), "prevalensi": prevalensi,
        "average_precision": float(average_precision_score(y, skor)),
        "roc_auc": float(roc_auc_score(y, skor)),
        "brier": float(brier_score_loss(y, skor)),
        "ambang_skor": float(batas), "porsi_alarm": float(alarm.mean()),
        "presisi_alarm": presisi, "recall_alarm": float(recall_score(y, alarm, zero_division=0)),
        "lift_vs_prevalensi": float(presisi / prevalensi) if prevalensi else None,
        "matriks_kebingungan_tn_fp_fn_tp": confusion_matrix(y, alarm, labels=[0, 1]).ravel().astype(int).tolist(),
    }


def plot_hasil(y, prediksi, terpilih, hasil, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = ["Prevalensi acak", "Regresi logistik", "Gradient boosting"]
    nilai = [float(y.mean()), hasil["regresi_logistik"]["average_precision"],
             hasil["hist_gradient_boosting"]["average_precision"]]
    ax.bar(n, nilai, color=["#9ca3af", "#3b82f6", "#db8c36"])
    ax.set_ylabel("Average precision test (lebih tinggi lebih baik)")
    ax.set_title("Deteksi risiko recovery rougher <75%")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "perbandingan_ap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for nama, skor in prediksi.items():
        presisi, recall, _ = precision_recall_curve(y, skor)
        ax.plot(recall, presisi, label=f"{nama}: AP={average_precision_score(y,skor):.3f}")
    ax.axhline(float(y.mean()), ls="--", c="gray", label="Prevalensi test")
    ax.set(xlabel="Recall", ylabel="Precision", title="Kurva precision–recall pada test")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "kurva_pr_test.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/gold_recovery_full.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error("CSV tidak ada. Jalankan python -m src.unduh_data lebih dulu.")
    assert len(FITUR) == 22 and all(c.startswith(("rougher.input.", "rougher.state.")) for c in FITUR)

    splits = bagi_kronologis(baca_csv(args.csv))
    (x_train, r_train), (x_val, r_val), (x_test, r_test) = (berlabel(d) for d in splits)
    if x_train.isna().all().any():
        raise ValueError("Fitur train seluruhnya hilang; periksa kualitas data")
    y_train, y_val, y_test = [(r < AMBANG_RENDAH).astype(int) for r in (r_train, r_val, r_test)]
    if any(len(np.unique(y)) != 2 for y in (y_train, y_val, y_test)):
        raise ValueError("Setiap split harus memuat kedua kelas")
    model = {
        "regresi_logistik": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, min_samples_leaf=40,
                                           learning_rate=.06, l2_regularization=1,
                                           early_stopping=False, random_state=SEED)),
    }
    for m in model.values():
        m.fit(x_train, y_train)
    skor_val = {nama: m.predict_proba(x_val)[:, 1] for nama, m in model.items()}
    pilih = max(model, key=lambda nama: average_precision_score(y_val, skor_val[nama]))
    # Ambang diputuskan dengan validasi, lalu dibekukan untuk test dan API.
    ambang = float(np.quantile(skor_val[pilih], 1-PORSI_ALERT_VALIDASI))
    validasi = {nama: nilai_metrik(y_val, skor, ambang) for nama, skor in skor_val.items()}
    skor_test = {nama: m.predict_proba(x_test)[:, 1] for nama, m in model.items()}
    test = {nama: nilai_metrik(y_test, skor, ambang) for nama, skor in skor_test.items()}

    rng = np.random.default_rng(SEED)
    indeks = np.sort(rng.choice(len(x_val), size=min(1000, len(x_val)), replace=False))
    penting = permutation_importance(model[pilih], x_val.iloc[indeks], y_val[indeks],
                                    n_repeats=3, random_state=SEED,
                                    scoring="average_precision", n_jobs=1)
    importansi = pd.DataFrame({"fitur": FITUR, "penurunan_ap_perm": penting.importances_mean,
                               "simpangan": penting.importances_std}).sort_values("penurunan_ap_perm", ascending=False)

    audit = buat_audit(splits)
    rentang = x_train.quantile([.005, .995]).fillna(x_train.median())
    input_ood = ((x_test.lt(rentang.loc[.005]) | x_test.gt(rentang.loc[.995])) & x_test.notna()).any(axis=1)
    data_test = splits[2].loc[splits[2][TARGET].notna()].reset_index(drop=True)
    pred = pd.DataFrame({
        "date": data_test["date"], "recovery_aktual": r_test,
        "kejadian_di_bawah_75": y_test,
        "skor_regresi_logistik": skor_test["regresi_logistik"],
        "skor_hist_gradient_boosting": skor_test["hist_gradient_boosting"],
        "skor_model_terpilih": skor_test[pilih],
        "alarm": skor_test[pilih] >= ambang,
        "input_di_luar_rentang_train": input_ood.to_numpy(),
    })
    ringkasan = {
        "sumber": SUMBER, "target_label": f"{TARGET} < {AMBANG_RENDAH}%",
        "ambang_75_adalah_ilustrasi_bukan_spesifikasi_pabrik": True,
        "fitur": FITUR, "split": audit, "seed": SEED,
        "model_terpilih_dari_average_precision_validasi": pilih,
        "kebijakan_alarm": f"ambang skor persentil 98 dari validasi, target sekitar {PORSI_ALERT_VALIDASI:.0%} alarm validasi",
        "ambang_skor_beku": ambang,
        "validasi": validasi, "test": test,
        "baris_test_dengan_setidaknya_satu_fitur_di_luar_p005_p995_train": float(input_ood.mean()),
        "status_penerapan": "riset offline; perlu validasi definisi recovery, rilis lab, drift, dan persetujuan proses",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model[pilih], "fitur": FITUR, "model_nama": pilih,
        "ambang_skor": ambang, "batas_recovery": AMBANG_RENDAH,
        "median_fitur": x_train.median().to_dict(),
        "batas_bawah": rentang.loc[.005].to_dict(), "batas_atas": rentang.loc[.995].to_dict(),
    }, args.out / "model.joblib", compress=3)
    (args.out / "metrics.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    pred.to_csv(args.out / "prediksi_test.csv", index=False)
    importansi.to_csv(args.out / "penting_fitur_validasi.csv", index=False)
    plot_hasil(y_test, skor_test, pilih, test, args.out)

    hasil = test[pilih]
    laporan = [
        "# Hasil penelitian: alarm risiko recovery rougher rendah", "",
        f"Kejadian didefinisikan secara ilustratif sebagai recovery <{AMBANG_RENDAH:.0f}%; bukan ambang AMMAN.",
        "Model dan ambang alarm dipilih memakai validasi waktu; metrik di bawah pada test berikutnya.", "",
        "| Model | AP validasi | AP test | ROC-AUC test | Brier test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for nama in model:
        laporan.append(f"| {nama} | {validasi[nama]['average_precision']:.3f} | {test[nama]['average_precision']:.3f} | {test[nama]['roc_auc']:.3f} | {test[nama]['brier']:.3f} |")
    laporan += [
        "", f"Prevalensi kejadian test: **{hasil['prevalensi']:.1%}** (AP acak kira-kira sebesar prevalensi).",
        f"Model terpilih: **{pilih}**. Ambang skor dari persentil 98 validasi: **{ambang:.3f}**.",
        f"Pada test: alarm **{hasil['porsi_alarm']:.1%}** dari baris berlabel, presisi **{hasil['presisi_alarm']:.1%}**, recall **{hasil['recall_alarm']:.1%}**; lift presisi **{hasil['lift_vs_prevalensi']:.1f}×** dari prevalensi.",
        f"Fitur di luar rentang persentil 0,5–99,5 train pada **{input_ood.mean():.1%}** baris test. Jika gate input API aktif, banyak baris akan ditahan dan cakupan alert berkurang.",
        "", "Perubahan prevalensi dan distribusi skor antara validasi dan test membuat porsi alarm test berbeda dari anggaran 2% validasi.",
        "Kejadian dengan recovery 0 tetap dihitung; bisa mencakup henti operasi, kesalahan label, atau proses nyata dan perlu audit domain.",
        "Importansi permutasi bersifat prediktif, bukan bukti sebab-akibat ataupun usulan menaikkan/menurunkan setpoint.",
        "Tanpa timestamp rilis assay, klaim alarm dini atau manfaat finansial belum dapat dibuat.",
    ]
    (args.out / "laporan_hasil.md").write_text("\n".join(laporan)+"\n", encoding="utf-8")
    print("\n".join(laporan))


if __name__ == "__main__":
    main()
