# Studi Kasus: Aturan Manual vs Machine Learning + Audit Bias

Berdasarkan kuliah umum **"Introduction to Machine Learning"** oleh **Dr. Taniza Tajudin** (UiTM Cawangan Kedah, Malaysia).

Persetujuan pinjaman diputuskan dua cara pada 6.000 pemohon **sintetis**:
1. **Pemrograman tradisional**: aturan `if-else` (penghasilan, skor kredit, rasio utang) yang ambangnya ditentukan manusia.
2. **Machine Learning**: regresi logistik yang belajar dari keputusan historis.

Lalu keduanya diaudit: dari pemohon yang **sebenarnya layak** membayar, berapa persen yang disetujui di kota vs desa?

## Hubungan dengan materi

| Poin materi | Implementasi |
|---|---|
| Pemrograman tradisional vs ML | Aturan manual (dapat digeser di web) vs model yang belajar dari data |
| Supervised learning: classification | Regresi logistik memprediksi disetujui/ditolak |
| Evaluasi akurasi model | Akurasi pada data uji (1.200 pemohon) |
| Etika: bias | Audit selisih peluang disetujui antarwilayah (*equal opportunity*) |
| Etika: transparansi | Koefisien model ditampilkan; aturan manual dapat dibaca langsung |

## Cara menjalankan

```bash
pip install -r requirements.txt
python aturan_vs_ml.py     
```
link index.html https://krbara.github.io/studi-kasus-aturan-vs-ml/

## Hasil contoh (dengan seed bawaan)

| Metode | Akurasi | Selisih peluang disetujui (kota − desa) |
|---|---|---|
| Aturan manual | 66,0% | 6,4 poin |
| ML penuh (pakai wilayah) | 74,8% | 41,5 poin |
| ML tanpa wilayah | 73,4% | 32,0 poin |
| ML tanpa wilayah & proksi | 71,4% | 10,2 poin |

Angka dapat sedikit berbeda jika seed atau parameter diubah.

## Temuan

- Model ML lebih **akurat** meniru keputusan historis, tetapi ikut **mewarisi bias** yang ada di data historis.
- **Menghapus kolom wilayah tidak cukup**: jarak ke cabang bertindak sebagai proksi, sehingga bias masih tersisa.
- Menghapus proksi menurunkan bias, dengan sedikit pengorbanan akurasi (**trade-off** akurasi vs keadilan).
- Aturan manual lebih transparan dan selisihnya lebih kecil di sini, tetapi akurasinya lebih rendah dan ambangnya tetap dipilih manusia.

## Catatan penting

- Data dan bias **sengaja dibuat** (sintetis) untuk demonstrasi. Tidak merepresentasikan lembaga keuangan nyata.
- Akurasi diukur terhadap keputusan historis (yang sudah bias), bukan terhadap kemampuan bayar sebenarnya.
- Ukuran keadilan yang dipakai hanya satu (*equal opportunity*). Ukuran lain, misalnya *demographic parity*, bisa memberi kesimpulan berbeda.
- Ini model sederhana untuk belajar, bukan sistem penilaian kredit.

## Pengembangan lanjutan

Menambah mitigasi lain (pembobotan ulang data, ambang per kelompok), mencoba model lain (decision tree, random forest), atau membandingkan beberapa ukuran keadilan.
