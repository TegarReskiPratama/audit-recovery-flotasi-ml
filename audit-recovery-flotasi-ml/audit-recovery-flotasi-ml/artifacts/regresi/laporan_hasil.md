# Laporan eksperimen: recovery rougher (dataset Kaggle)

Metrik test dalam poin persentase, kecuali R². Ini bukan hasil AMMAN.

| Model | MAE | RMSE | R² | P90 galat |
| --- | ---: | ---: | ---: | ---: |
| median_train | 8.308 | 11.427 | -0.282 | 12.281 |
| ridge | 9.616 | 12.746 | -0.596 | 17.827 |
| hist_gradient_boosting | 11.852 | 15.330 | -1.308 | 20.588 |

Model dipilih dari validasi: **ridge**.
Radius interval residual validasi 90%: **±14.073** poin; cakupan empiris test: **78.7%**.
Selisih MAE model−median test, CI bootstrap blok 24: [0.5830720214644622, 2.1497455247978863].

Interpretasi: gunakan sebagai studi metode; tidak mengklaim optimasi operasi atau sebab-akibat dari importansi fitur.
Waktu rilis lab dan kesetaraan sensor dengan AMMAN harus diaudit sebelum uji lapangan.
