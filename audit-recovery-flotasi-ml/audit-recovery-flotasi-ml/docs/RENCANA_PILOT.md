# Rencana Adaptasi pada Departemen Proses

Dokumen ini adalah rancangan diskusi untuk tim proses dan data science, bukan proposal yang sudah disetujui AMMAN.

## Tahap 1 — Kesesuaian masalah dan kontrak data

Pemilik proses memilih satu KPI yang nyata, misalnya recovery Cu atau kualitas konsentrat pada sirkuit rougher, dan mendefinisikan satuan, batas aman, serta keputusan apa yang mungkin terbantu oleh alarm. Fitur Kaggle harus dipetakan ulang terhadap historian pabrik dan disetujui engineer instrumentasi. Catat cap waktu pengukuran sensor, waktu ambil sampel, waktu hasil lab diterbitkan, sistem zona waktu, periode henti operasi, pergantian bijih, dan peristiwa pemeliharaan. Jika fitur ternyata tersedia setelah kejadian, keluarkan dari model.

## Tahap 2 — Eksperimen historis yang jujur

Pisahkan data menurut kampanye atau periode berurutan, bukan acak. Bandingkan prevalensi, baseline sederhana, regresi logistik, boosting, dan kemampuan prediksi pada data setelah perubahan proses. Ukur AP, presisi dan recall pada kapasitas review yang disepakati, waktu antisipasi *yang benar-benar tersedia*, serta perubahan beban alarm antar shift. Laporkan secara terpisah baris yang tidak memiliki label atau ditahan oleh gate input. Simpan audit false positive dan false negative untuk diskusi metallurgist.

## Tahap 3 — Shadow mode tanpa kendali

Jalankan skor di samping sistem yang ada tanpa instruksi setpoint. Minta operator mencatat apakah alarm dapat ditindaklanjuti, apakah lab membenarkannya, dan apakah ketepatan waktu cukup. Pantau sensor hilang, out-of-range, drift, proporsi alarm, dan kegagalan API. Tentukan pemilik eskalasi dan hentikan alat jika data tak memenuhi kontrak.

## Tahap 4 — Keputusan pilot

Lanjut hanya jika kinerja prospektif dan kapasitas review sesuai kriteria tertulis yang disetujui tim proses. Hitung potensi manfaat dengan data tonase, grade, dan recovery **aktual** serta biaya alarm; jangan mengalikan hasil Kaggle menjadi manfaat AMMAN. Setiap perubahan setpoint tetap berada dalam prosedur operasi dan keselamatan yang berlaku. Dokumentasikan persetujuan, versi, monitoring, dan rollback.
