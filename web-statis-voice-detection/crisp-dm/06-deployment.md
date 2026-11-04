# Tahap 6: Deployment

**Metodologi:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*)  
**Disusun Oleh:** Kelompok 4 (Muhammad Ainul Fuady, Irwan Dwi Mukhlisin, Fikri Mutawakkil)

---

## 6.1 Paket Serialisasi Model (`model_voice_rf_5s.pkl`)

Agar model Machine Learning yang telah terlatih dan tervalidasi dapat digunakan di lingkungan produksi (*production*), seluruh komponen inferensi dibungkus ke dalam satu file terpadu menggunakan *library* `joblib`:

```python
paket_model = {
    'model': rf_model,                    # Model RandomForestClassifier utama
    'svm_model': svm_calibrated_model,    # Model SVM RBF alternatif
    'scaler': scaler_48,                  # Objek StandardScaler (48 fitur)
    'kolom_fitur': kolom_fitur,           # Daftar urutan 48 nama fitur audio
    'fitur_terpilih_20': fitur_terpilih,  # Daftar 20 fitur hasil feature selection
    'sr': 16000,                          # Target sampling rate (16 kHz mono)
    'durasi_segmen': 5,                   # Durasi potongan analisis (5 detik)
    'label_map': {
        0: 'Suara Manusia Asli (Human Voice)',
        1: 'Suara Manipulasi AI (AI-Generated Voice)'
    },
    'metrics': {'accuracy': 98.75, 'f1_score': 98.73},
    'mean_asli': dict_mean_asli,          # Nilai referensi rata-rata fitur Asli
    'mean_ai': dict_mean_ai               # Nilai referensi rata-rata fitur AI
}
```

File serialisasi disimpan pada lokasi:
`backend/main/model_voice_rf_5s.pkl` (Ukuran file: $\approx 1{,}48\text{ MB}$).

---

## 6.2 Alur Kerja Inferensi Audio (*Inference Pipeline*)

Saat menerima file rekaman audio baru dari pengguna, sistem menjalankan alur pemrosesan sekuensial berikut:

```
  Input Audio (.wav / .mp3)
              │
              ▼
  1. Standarisasi Sinyal (Resample ke 16.000 Hz, Konversi Mono, Normalisasi RMS)
              │
              ▼
  2. Segmentasi Audio (Pemotongan per segmen 5 detik = 80.000 sampel)
              │
              ▼
  3. Ekstraksi 48 Fitur Librosa per Segmen (Time, Spectral, Pitch F0, MFCC)
              │
              ▼
  4. Transformasi Z-Score menggunakan StandardScaler terlatih
              │
              ▼
  5. Estimasi Probabilitas Random Forest per Segmen: [Prob_Asli, Prob_AI]
              │
              ▼
  6. Agregasi Keputusan (Voting Rata-rata Segmen) & Visualisasi Hasil
```

---

## 6.3 Implementasi 1: Aplikasi Web Interaktif (Flask + Tailwind CSS)

Aplikasi web interaktif dibangun menggunakan framework Python **Flask** pada `frontend/app.py` dan antarmuka web modern pada `frontend/templates/index.html`.

### Fitur Antarmuka Web:
1. **Voice Langsung (Mikrofon):** Perekaman audio langsung melalui Web Audio API browser pada frekuensi sampling 16 kHz.
2. **Upload File Audio (.wav):** Pemeriksaan file audio eksternal yang diunggah pengguna.
3. **Pengujian Sampel Dataset:** Fitur pengujian cepat untuk memutar dan menguji file contoh asli dan manipulasi AI dari dataset.
4. **Visualisasi Terintegrasi:** Menampilkan grafik gelombang waktu (*waveform*), spektrogram Mel, dan skor keyakinan (*confidence score*) per segmen 5 detik.

### Cara Menjalankan Aplikasi Web Flask:
```powershell
cd "D:\SEMESTER 5 TIF 2026\PSD\voice-detection\frontend"
python app.py
```
Aplikasi dapat dibuka melalui browser pada alamat:
`http://127.0.0.1:5000`

---

## 6.4 Implementasi 2: Aplikasi Terminal Real-Time Live Voice Detector (10 Detik)

Untuk kebutuhan pengujian langsung melalui *Command Line Interface* (CLI) tanpa memerlukan GUI/browser, disediakan skrip streaming mikrofon real-time pada `backend/main/tes_suara_terminal.py`.

### Mekanisme Kerja Real-Time 10 Detik:
- Program mendengarkan mikrofon secara pasif hingga terdeteksi adanya energi suara aktif (*voice activity detection*).
- Begitu suara masuk, program mengumpulkan rekaman **10 detik utuh secara kontinu** (terdiri dari 2 segmen @ 5 detik).
- Tepat di detik ke-10, program mengekstrak fitur kedua segmen dan langsung mengeluarkan **Vonis Akhir 10 Detik** beserta skor probabilitas dan 5 fitur akustik utamanya.

### Cara Menjalankan Aplikasi Terminal:
```powershell
cd "D:\SEMESTER 5 TIF 2026\PSD\voice-detection\backend\main"
python tes_suara_terminal.py
```

---

## 6.5 Panduan Deployment ke GitHub Pages

Proyek ini telah dilengkapi dengan alur kerja otomatis **GitHub Actions Workflow** pada `.github/workflows/deploy-book.yml`.

### Langkah-langkah Mempublikasikan Web Statis ke GitHub Pages:
1. Buat repositori baru di GitHub (misalnya: `voice-authenticity-detection`).
2. Masuk ke folder web statis dan inisialisasi Git:
   ```bash
   cd "D:\SEMESTER 5 TIF 2026\PSD\voice-detection\web-statis-voice-detection"
   git init
   git add .
   git commit -m "feat: complete CRISP-DM static book for Kelompok 4"
   git branch -M main
   git remote add origin https://github.com/USERNAME/REPO_NAME.git
   git push -u origin main
   ```
3. Buka menu **Settings** $\rightarrow$ **Pages** pada repositori GitHub Anda.
4. Pada opsi **Build and deployment**, pilih **Deploy from a branch** dan pilih branch `gh-pages` (folder `/root`).
5. Web statis Jupyter Book Kelompok 4 akan langsung aktif dan dapat diakses secara publik oleh dosen dan pengguna di internet!
