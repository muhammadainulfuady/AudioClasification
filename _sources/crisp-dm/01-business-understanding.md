# Tahap 1: Business Understanding

**Metodologi:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*)  
**Disusun Oleh:** Kelompok 4 (Muhammad Ainul Fuady, Irwan Dwi Mukhlisin, Fikri Mutawakkil)

---

## 1.1 Latar Belakang Masalah

Kemajuan pesat di bidang kecerdasan buatan, terutama model *Deep Learning* generatif seperti *Generative Adversarial Networks* (GAN), *Diffusion Models*, dan arsitektur *Neural Vocoder* (seperti HiFi-GAN, WaveGlow, Tacotron), telah merevolusi sintesis suara (*speech synthesis*). Saat ini, suara manusia dapat ditiru (*voice cloning*) dengan sangat akurat hanya menggunakan cuplikan suara referensi selama beberapa detik.

Kendati teknologi ini membawa dampak positif bagi industri asisten virtual, sulih suara hiburan, dan aksesibilitas penyandang disabilitas, teknologi ini membuka celah ancaman kejahatan digital (*cybercrime*) yang eksponensial:
1. **Penipuan Berbasis Suara (*Voice Phishing / Vishing*):** Pelaku menirukan suara anggota keluarga, rekan kerja, atau pimpinan perusahaan untuk meminta transfer dana darurat.
2. **Disinformasi Politik & Rekayasa Sosial (*Deepfake Audio*):** Pembuatan rekaman suara palsu tokoh publik yang dapat memicu ketidakstabilan sosial dan manipulasi opini publik.
3. **Kerentanan Autentikasi Biometrik Suara:** Bank dan layanan digital yang mengandalkan verifikasi suara (*voice recognition*) rentan ditembus jika tidak dilengkapi lapisan verifikasi keaslian (*Liveness / Anti-Spoofing Detection*).

Oleh karena itu, diperlukan sebuah sistem deteksi keaslian suara otomatis yang cepat, akurat, dan dapat bekerja secara andal hanya dari potongan rekaman suara singkat.

---

## 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah dalam proyek ini adalah:
1. Bagaimana karakteristik akustik suara manusia asli dapat dibedakan secara matematis dari suara hasil sintesis/manipulasi AI?
2. Berapa durasi segmen suara optimal yang cukup singkat untuk deteksi cepat namun tetap mempertahankan informasi karakteristik vokal secara akurat?
3. Fitur audio apa saja dari domain waktu, spektral, pitch pita suara, dan saluran vokal (*MFCC*) yang paling berpengaruh secara signifikan dalam mendeteksi suara AI?
4. Algoritma *Machine Learning* apa yang memberikan kinerja terbaik dengan tingkat akurasi dan kestabilan tertinggi pada data uji?

---

## 1.3 Tujuan Proyek

### 1.3.1 Tujuan Bisnis (*Business Objectives*)
1. Membangun sistem deteksi keaslian suara (*Voice Authenticity Detector*) yang mampu mengklasifikasikan suara ke dalam dua kategori:
   - **`asli_human` (Label `0`):** Suara rekaman manusia asli biologis.
   - **`ai_manipulasi` (Label `1`):** Suara hasil sintesis kecerdasan buatan atau pengubah suara (*voice changer/TTS*).
2. Memungkinkan pengujian secara fleksibel baik melalui unggah file rekaman suara, pengujian sampel dataset, maupun perekaman mikrofon secara *real-time*.

### 1.3.2 Tujuan Teknis Sains Data (*Data Mining Goals*)
1. Mengolah dataset audio standar berjumlah 200 file WAV (masing-masing berdurasi 30 detik pada *sampling rate* `16.000 Hz`).
2. Menerapkan strategi segmentasi **5 detik** (memotong setiap file 30 detik menjadi 6 segmen @ 5 detik) untuk memperbanyak sampel menjadi **1.200 baris segmen**.
3. Mengekstrak **48 fitur akustik Librosa** yang mencakup 4 domain karakteristik suara.
4. Menganalisis pengaruh fitur dan melakukan seleksi fitur (*Feature Selection*) guna menentukan subset fitur optimal.
5. Melatih dan membandingkan 4 model klasifikasi (*Random Forest*, *Support Vector Machine*, *Gradient Boosting*, dan *K-Nearest Neighbors*) menggunakan *5-Fold Stratified Cross-Validation*.

---

## 1.4 Kriteria Kesuksesan Proyek (*Success Criteria*)

Untuk memastikan proyek ini bernilai aplikatif dan memenuhi standar ilmiah, kriteria kesuksesan yang ditetapkan meliputi:

| No | Parameter Evaluasi | Target Metrik | Keterangan |
| :-: | :--- | :-: | :--- |
| **1** | **Akurasi Data Uji (*Test Accuracy*)** | $\ge 95{,}0\%$ | Kemampuan model mengklasifikasikan segmen uji 5 detik yang belum pernah dilihat. |
| **2** | **F1-Score Model** | $\ge 95{,}0\%$ | Keseimbangan antara Presisi dan Recall pada kelas Asli maupun AI. |
| **3** | **Area Under ROC Curve (ROC-AUC)** | $\ge 0{,}98$ | Kemampuan separasi probabilitas model dalam membedakan kedua kelas secara tegas. |
| **4** | **Integritas Pembagian Data (*Zero Leakage*)** | $100\%$ Bebas Leakage | Pembagian *Train/Test* wajib dikelompokkan berdasarkan `file_induk` (bukan mengacak segmen secara bebas). |
