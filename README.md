# Manual Book Penggunaan Program
_Rekomendasi Program Latihan Gym berbasis ML_

Dokumen ini menjelaskan cara **menjalankan**, **menggunakan**, dan **memelihara** aplikasi web rekomendasi program latihan gym.

---

## 1. Struktur Proyek
```
DEPLOYMENT/
├─ app.py
├─ KNN,_DT,_RF_Gym_Program_Classification_(Skripsi).ipynb
├─ mlp_gym_model.h5
├─ MLP_Gym_Program_Classification.ipynb
├─ scaler.pkl
├─ data/
│├─ gym_datasets.csv
├─ templates/
│├─index.html
│├─result.html
```

**Keterangan singkat:**
- `app.py` — Aplikasi web (Flask) untuk **inference** (prediksi) menggunakan model MLP.
- `mlp_gym_model.h5` — Model MLP (TensorFlow/Keras) untuk klasifikasi program latihan.
- `scaler.pkl` — Scaler (sklearn) untuk normalisasi fitur saat inference.
- `data/gym_datasets.csv` — Dataset sumber (211 baris) untuk pelatihan & validasi.
- `templates/` — Tampilan antarmuka (HTML): form input & halaman hasil.
- `*.ipynb` — Notebook pelatihan model:
  - `MLP_Gym_Program_Classification.ipynb` (model **MLP**)
  - `KNN,_DT,_RF_Gym_Program_Classification_(Skripsi).ipynb` (model **KNN, DT, RF** — revisi dosen)

---

## 2. Prasyarat
- **Python** 3.9–3.11 (disarankan 3.10)
- **pip** terbaru
- Opsional: **virtualenv**/**venv** untuk lingkungan terisolasi

### 2.1. Instalasi Dependensi
Buat dan aktifkan virtual environment (opsional tapi disarankan), lalu instal paket:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install flask tensorflow scikit-learn joblib numpy pandas
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install flask tensorflow scikit-learn joblib numpy pandas
```

> **Catatan:** Jika instalasi `tensorflow` bermasalah di perangkat Anda, gunakan versi CPU saja: `pip install tensorflow-cpu` (khusus beberapa OS/arsitektur).

---

## 3. Menjalankan Aplikasi (Inference)
1. Ekstrak `DEPLOYMENT.zip` ke folder kerja Anda.
2. (Opsional) Aktifkan virtual env dan instal dependensi (lihat Bagian 2.1).
3. Jalankan aplikasi:
   ```bash
   python app.py
   ```
4. Buka browser dan akses: **http://127.0.0.1:5000**

### 3.1. Alur Penggunaan di UI
- Isi **empat input** berikut pada halaman utama:
  1. **Profesi** (contoh: `karyawan`, `mahasiswa`, `sekolah`)
  2. **Frekuensi Latihan per Minggu** (contoh: 1–7)
  3. **Durasi per Sesi (menit)** (contoh: 45, 60, 90, 120)
  4. **Tingkat Kesibukan** (`rendah`, `sedang`, `tinggi`)
- Klik **Submit** untuk melihat hasil.
- Halaman hasil menampilkan:
  - **Program rekomendasi** (`Push Pull Legs`, `Upper Lower`, `Full Body Workout`)
  - **Probabilitas tiap kelas** (ditampilkan dalam grafik/daftar)
  - **Daftar gerakan** sesuai program (dibagi per hari/sesi)
  - **Video YouTube** rujukan (embedded) untuk program terkait

> Aplikasi otomatis melakukan **scaling** input menggunakan `scaler.pkl` sebelum memanggil model `mlp_gym_model.h5` untuk prediksi.

---

## 4. Skema Data & Pra-pemrosesan
Dataset sumber (`data/gym_datasets.csv`) memiliki kolom:
```
['profesi', 'frekuensi latihan', 'durasi latihan', 'tingkat kesibukan', 'program latihan']
```
Kelas target (`program latihan`) mencakup: `full body workout, push, pull, leg, upper & lower`

**Pra-pemrosesan umum (ringkas):**
- Variabel kategorikal (`profesi`, `tingkat kesibukan`) fitur `profesi` di encoding menggunakan label encoding, sementara fitur `tingkat kesibukan` di encoding menggunakan ordinal encoding (karena memiliki tingkatan).
- Variabel numerik (`frekuensi latihan`, `durasi latihan`) dinormalisasi (contoh: `MinMaxScaler`).
- Model inference **mengharuskan** urutan fitur & skema **identik** dengan saat pelatihan.

> **Penting:** Jika Anda mengubah kamus label (mis. menambah kategori profesi), **latih ulang** model dan **perbarui** `scaler.pkl` agar konsisten.

---

## 5. Pelatihan & Pembaruan Model

### 5.1. MLP (`MLP_Gym_Program_Classification.ipynb`)
1. Buka notebook dan jalankan sel **berurutan** (Run All).
2. Periksa metrik (confusion matrix, classification report).
3. Simpan artefak untuk deployment (biasanya sudah ada di notebook):
   - `mlp_gym_model.h5`
   - `scaler.pkl`
4. **Salin** artefak ke folder yang sama dengan `app.py` (replace file lama).
5. Restart aplikasi `python app.py` untuk memakai model terbaru.

### 5.2. KNN / Decision Tree / Random Forest  
Notebook: `KNN,_DT,_RF_Gym_Program_Classification_(Skripsi).ipynb)`

**Langkah umum:**
1. Jalankan notebook untuk **training** & **evaluasi** ketiga model.
2. **Pilih model terbaik** (berdasarkan akurasi/validasi).
3. **Simpan** model terpilih (contoh RF) sebagai `best_model.pkl`:
   ```python
   import joblib
   joblib.dump(best_model, "best_model.pkl")
   joblib.dump(scaler, "scaler.pkl")  # jika perlu scaler yang sama
   ```
4. **Ubah** `app.py` agar memuat model sklearn:
   ```python
   # Ganti import & loader
   from flask import Flask, request, render_template
   import joblib
   import numpy as np

   app = Flask(__name__)

   model = joblib.load("best_model.pkl")
   scaler = joblib.load("scaler.pkl")

   # ... di dalam route:
   X = np.array([[freq, durasi, profesi_encoded, kesibukan_encoded]])  # urutan harus sama
   X_scaled = scaler.transform(X)
   pred = model.predict(X_scaled)[0]           # jika predict -> label langsung
   # atau kalau predict_proba tersedia:
   proba = getattr(model, "predict_proba", None)
   if proba:
       pred_probs = model.predict_proba(X_scaled)[0]
   ```
5. Sesuaikan **mapping label** agar **konsisten** dengan kelas pada model Anda.

> **Catatan:** Model sklearn **umumnya lebih ringan** untuk deployment dibanding TensorFlow, tapi pastikan akurasinya memenuhi target skripsi.

---

## 6. Konfigurasi & Kustomisasi

### 6.1. Mengubah Daftar Gerakan & Video
Di `app.py` terdapat:
- `label_map` — memetakan indeks model ke label string.
- `video_id` — YouTube video ID per label.
- `program_gerakan` — daftar gerakan per program.

Anda bisa mengubah/menambah isi dictionary tersebut (mis. mengganti variasi latihan atau video referensi).

### 6.2. Validasi Input
Tambahkan validasi (mis. rentang 1–7 untuk frekuensi, 30–120 menit untuk durasi) pada sisi **frontend** (`templates/index.html`) maupun **backend** (`app.py`) untuk menghindari input tidak wajar.

---

## 7. Menjalankan di Jaringan Lokal / Server
- Jalankan pada host semua interface:
  ```python
  app.run(host="0.0.0.0", port=5000, debug=False)
  ```
- Reverse proxy (opsional): Nginx/Apache men-forward ke `localhost:5000`.
- **Production server** (opsional, model sklearn): Gunicorn/Waitress/Uvicorn (via WSGI), contoh:
  ```bash
  pip install gunicorn
  gunicorn -w 2 -b 0.0.0.0:5000 app:app
  ```

> Untuk TensorFlow, perhatikan memori; jalankan worker lebih sedikit atau gunakan CPU-only bila GPU tidak tersedia.

---

## 8. Troubleshooting
- **`ModuleNotFoundError: No module named 'sklearn'`**  
  Instal scikit-learn: `pip install scikit-learn`
- **`ValueError: X has n features, but scaler is expecting m features`**  
  Urutan/jumlah fitur tidak cocok. Samakan **pipeline preprocessing** dengan saat training.
- **`OSError: SavedModel file does not exist: mlp_gym_model.h5`**  
  Pastikan file model berada **satu folder** dengan `app.py` dan nama persis sama.
- **Prediksi aneh/out-of-range**  
  Cek encoding kategorikal & rentang numerik; data input mungkin di luar distribusi training.
- **Browser tidak menampilkan grafik/video**  
  Cek koneksi internet (untuk video YouTube) dan pastikan Chart.js/CDN dapat diakses.

---

**Selesai.** Manual ini sudah mencakup: setup, running, penggantian model MLP ↔︎ KNN/DT/RF, skema data, kustomisasi, dan deployment ringan.
