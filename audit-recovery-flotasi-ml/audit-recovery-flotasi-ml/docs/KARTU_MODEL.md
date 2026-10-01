# Kartu Model: Alarm Recovery Rougher Rendah

## Tujuan dan ruang lingkup

Model memeringkat kemungkinan label `rougher.output.recovery <75%` pada dataset Kaggle publik. Ambang 75% dipilih untuk eksperimen portofolio, bukan batas teknik AMMAN. Input adalah 22 variabel `rougher.input.*` dan `rougher.state.*` pada timestamp yang sama dengan target; output adalah skor dan usulan **review operator**. Tidak boleh dipakai mengontrol proses atau mengklaim peningkatan hasil.

## Data dan asumsi

Mirror Kaggle `manuelantoniocherres/gold-recovery-full` (22.716 baris, 2016–2018). Sumber primer, unit sebagian fitur, waktu rilis lab, cara penanganan downtime, dan asal lokasi operasi tidak terverifikasi dari mirror. Target kosong dibuang **setelah** batas waktu split ditetapkan. Target 0 tetap masuk dan memerlukan validasi domain. Input contemporaneous bisa saja baru tersedia setelah hasil proses; keterlambatan belum dimodelkan.

## Eksperimen dan kinerja

Split kronologis train 60%, validasi 20%, test 20%. Model dipilih menurut average precision validasi. HistGradientBoosting dipilih; regresi logistik dilaporkan. Ambang skor ditetapkan dari persentil 98 validasi untuk anggaran review kira-kira 2% pada validasi.

Pada test, prevalensi label 3,6%, AP model terpilih 0,191, ROC-AUC 0,746; dengan ambang tetap, 9,1% baris mendapat alarm, presisi 18,4%, recall 46,5%. AP regresi logistik pada test 0,284, meski AP validasinya lebih rendah. Drift dan seleksi model antarperiode masih menjadi masalah. Lihat `artifacts/metrics.json` untuk angka tepat dan rentang waktu.

## Pengamanan dan keterbatasan

- API memeriksa nama fitur, angka finite, dan rentang persentil 0,5–99,5 train. Pemeriksaan per fitur tidak cukup mendeteksi kombinasi keadaan asing; pada 40,8% baris test berlabel setidaknya satu fitur melewati batas tersebut.
- Skor bukan probabilitas terkalibrasi; Brier test HGB 0,109. AP dan presisi alert tidak menjamin manfaat operasional.
- Importansi fitur diperoleh melalui permutasi pada validasi; korelasi antar fitur dapat mengubah ranking. Tidak dapat disimpulkan efek perubahan setpoint.
- Prevalensi positif turun dari train ke test. Porsi alarm test menjadi 9,1% meski anggaran validasi sekitar 2%.
- Evaluasi offline tidak memperhitungkan waktu ketersediaan hasil lab, biaya false alarm, downtime, shift, geometalurgi, dan potensi bahaya keputusan manusia.

## Kepemilikan dan perubahan

Artefak hanya untuk demonstrasi, tanpa afiliasi dengan pemilik tambang atau AMMAN. Sebelum uji lapangan, definisikan kontrak data dan pemilik proses, lakukan review privasi serta akses, versi model dan data, logging, monitoring, jalur rollback, dan persetujuan metallurgist. Mengganti komoditas, sensor, atau target mensyaratkan pelatihan dan validasi ulang.
