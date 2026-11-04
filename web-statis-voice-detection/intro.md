# Proyek Sains Data — Kelompok 4

## Klasifikasi Audio Suara Manusia Asli dan Suara Hasil AI / Voice Changer

Selamat datang di repositori dokumentasi web statis resmi **Proyek Sains Data (PSD)** untuk sistem **Deteksi Keaslian Suara (Human Voice vs. AI-Generated Voice)**. Dokumentasi ini disusun secara komprehensif mengikuti metodologi standar internasional **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)**.

---

### Identitas Tim Pengembang (Kelompok 4)

Proyek ini disusun dan dikembangkan oleh mahasiswa Program Studi Teknik Informatika angkatan 2024:

| No | NIM | Nama Mahasiswa | Peran & Kontribusi Utama |
| :---: | :---: | :--- | :--- |
| **1** | **24-009** | **Muhammad Ainul Fuady** | *Data Pipeline, Feature Engineering Librosa & Deployment* |
| **2** | **24-010** | **Irwan Dwi Mukhlisin** | *Data Understanding, Exploratory Data Analysis & Preprocessing* |
| **3** | **24-105** | **Fikri Mutawakkil** | *Multi-Model Training, Hyperparameter Tuning & Evaluation* |

---

### Ringkasan Eksekutif Proyek

```
                                  ALUR KERJA PROYEK KELOMPOK 4
  ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
  │  200 File Audio WAV  │ ──> │ Segmentasi 5 Detik   │ ──> │ Ekstraksi 48 Fitur   │
  │ (100 Asli vs 100 AI) │     │ (1.200 Baris Segmen) │     │ (Time, Spectral, MFCC│
  └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                        │
  ┌──────────────────────┐     ┌──────────────────────┐                 │
  │ Web Flask & Realtime │ <── │ Evaluasi Multi-Model │ <───────────────┘
  │ Terminal Live Audio  │     │ (RF, SVM, GBDT, KNN) │     (Akurasi Test: 98.75% - 99.58%)
  └──────────────────────┘     └──────────────────────┘
```

Perkembangan teknologi generatif kecerdasan buatan (*Generative AI*) seperti *Text-to-Speech* (TTS) dan *Voice Cloning* memungkinkan pembuatan rekaman suara tiruan yang menyerupai manusia asli. Hal ini memicu risiko serius berupa penipuan berbasis suara (*voice phishing / vishing*), rekayasa sosial, dan pemalsuan identitas biometrik.

Proyek ini bertujuan membedakan **Suara Manusia Asli (`asli_human = 0`)** dan **Suara Manipulasi AI (`ai_manipulasi = 1`)** menggunakan segmentasi sinyal audio berdurasi **5 detik**, ekstraksi **48 fitur akustik Librosa**, serta klasifikasi berbasis Machine Learning.

---

### Struktur Dokumentasi (6 Tahap CRISP-DM)

Dokumentasi web statis ini terbagi ke dalam 6 bab utama sesuai tahapan CRISP-DM:

1. **[Tahap 1: Business Understanding](crisp-dm/01-business-understanding.md)**  
   Latar belakang masalah, rumusan masalah, tujuan bisnis (*business objectives*), tujuan teknis sains data (*data mining goals*), serta kriteria kesuksesan (*success criteria*).
2. **[Tahap 2: Data Understanding](crisp-dm/02-data-understanding.md)**  
   Inventarisasi dataset 200 file WAV (16.000 Hz, 30 detik), validasi integritas audio, serta eksplorasi sinyal audio mendalam (*Exploratory Data Analysis / EDA*) pada domain waktu, frekuensi spektral, dan kontur *pitch* biometrik.
3. **[Tahap 3: Data Preparation](crisp-dm/03-data-preparation.md)**  
   Strategi pemotongan 30s $\rightarrow$ 6 segmen @ 5s (total 1.200 segmen), kamus lengkap 48 fitur akustik, pembersihan data & deteksi *outlier* (metode IQR), analisis pengaruh fitur, seleksi fitur (*Feature Selection*), pembagian data bebas kebocoran (*Group Split by file_induk*), dan normalisasi `StandardScaler`.
4. **[Tahap 4: Modeling](crisp-dm/04-modeling.md)**  
   Pelatihan dan pengujian 4 algoritma klasifikasi (*Random Forest*, *SVM RBF Kernel*, *Gradient Boosting*, dan *K-Nearest Neighbors*) dengan validasi silang *5-Fold Stratified Cross-Validation*.
5. **[Tahap 5: Evaluation](crisp-dm/05-evaluation.md)**  
   Analisis metrik evaluasi komprehensif (*Accuracy, Precision, Recall, F1-Score, ROC-AUC*), visualisasi *Confusion Matrix*, analisis kurva ROC-AUC, inspeksi kesalahan prediksi (*Error Analysis*), serta evaluasi ketercapaian kriteria kesuksesan.
6. **[Tahap 6: Deployment](crisp-dm/06-deployment.md)**  
   Penyimpanan paket model terintegrasi (`model_voice_rf_5s.pkl`), perancangan alur inferensi, implementasi aplikasi web interaktif Flask, serta aplikasi terminal real-time *Live Voice Detector* 10 detik.
