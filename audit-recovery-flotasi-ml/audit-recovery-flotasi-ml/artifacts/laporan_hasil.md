# Hasil penelitian: alarm risiko recovery rougher rendah

Kejadian didefinisikan secara ilustratif sebagai recovery <75%; bukan ambang AMMAN.
Model dan ambang alarm dipilih memakai validasi waktu; metrik di bawah pada test berikutnya.

| Model | AP validasi | AP test | ROC-AUC test | Brier test |
| --- | ---: | ---: | ---: | ---: |
| regresi_logistik | 0.249 | 0.284 | 0.839 | 0.341 |
| hist_gradient_boosting | 0.265 | 0.191 | 0.746 | 0.109 |

Prevalensi kejadian test: **3.6%** (AP acak kira-kira sebesar prevalensi).
Model terpilih: **hist_gradient_boosting**. Ambang skor dari persentil 98 validasi: **0.595**.
Pada test: alarm **9.1%** dari baris berlabel, presisi **18.4%**, recall **46.5%**; lift presisi **5.1×** dari prevalensi.
Fitur di luar rentang persentil 0,5–99,5 train pada **40.8%** baris test. Jika gate input API aktif, banyak baris akan ditahan dan cakupan alert berkurang.

Perubahan prevalensi dan distribusi skor antara validasi dan test membuat porsi alarm test berbeda dari anggaran 2% validasi.
Kejadian dengan recovery 0 tetap dihitung; bisa mencakup henti operasi, kesalahan label, atau proses nyata dan perlu audit domain.
Importansi permutasi bersifat prediktif, bukan bukti sebab-akibat ataupun usulan menaikkan/menurunkan setpoint.
Tanpa timestamp rilis assay, klaim alarm dini atau manfaat finansial belum dapat dibuat.
