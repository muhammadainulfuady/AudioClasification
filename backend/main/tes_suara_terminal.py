import os
import sys
import time
import queue
import warnings
import joblib
import librosa
import sounddevice as sd
import numpy as np
import scipy.stats as stats

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_voice_rf_5s.pkl')


def ekstrak_fitur_segmen(y_seg, sr, f0_seg):
    """Mengekstrak 48 fitur audio Librosa dari 1 segmen berdurasi 5 detik."""
    rms = librosa.feature.rms(y=y_seg)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_seg)[0]
    silence_ratio = float(np.mean(rms < 0.01) * 100.0)
    skew_val = float(stats.skew(y_seg))
    kurt_val = float(stats.kurtosis(y_seg))

    S = np.abs(librosa.stft(y_seg))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    voiced_mask = rms[:len(f0_seg)] > 0.01
    f0_voiced = f0_seg[voiced_mask] if np.any(voiced_mask) else f0_seg
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_jitter = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0

    mel_S = librosa.feature.melspectrogram(S=S**2, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    fitur = {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'silence_ratio': silence_ratio,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'bandwidth_mean': float(np.mean(bandwidth)),
        'bandwidth_std': float(np.std(bandwidth)),
        'rolloff_mean': float(np.mean(rolloff)),
        'rolloff_std': float(np.std(rolloff)),
        'flatness_mean': float(np.mean(flatness)),
        'flatness_std': float(np.std(flatness)),
        'contrast_mean': float(np.mean(contrast)),
        'contrast_std': float(np.std(contrast)),
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_jitter': f0_jitter,
        'delta_mfcc_mean': float(np.mean(delta_mfcc)),
        'delta_mfcc_std': float(np.std(delta_mfcc)),
    }
    for i in range(13):
        fitur[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        fitur[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
    return fitur


def prediksi_segmen_5s(y_5s, sr, bundle):
    """Mengolah 1 jendela 5 detik secara real-time dan mengembalikan [prob_asli, prob_ai]."""
    y_5s = y_5s.astype(np.float32)
    y_5s = y_5s - np.mean(y_5s)

    # Normalisasi RMS ke rata-rata dataset (~0.023) agar jarak mulut ke mic tidak mengganggu
    rms_curr = float(np.sqrt(np.mean(y_5s**2)))
    if rms_curr > 1e-4:
        y_5s = np.clip(y_5s * (0.023 / rms_curr), -1.0, 1.0)

    f0_5s = librosa.yin(y_5s, fmin=50, fmax=400, sr=sr)
    d_fit = ekstrak_fitur_segmen(y_5s, sr, f0_5s)
    vec = np.array([[d_fit[k] for k in bundle['kolom_fitur']]])
    vec_scaled = bundle['scaler'].transform(vec)
    proba = bundle['model'].predict_proba(vec_scaled)[0]
    return proba, d_fit


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] File model tidak ditemukan: {MODEL_PATH}')
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    sr = bundle.get('sr', 16000)

    # Parameter Real-Time Streaming
    CHUNK_DETIK = 0.5                  # Baca mikrofon setiap 0.5 detik
    CHUNK_SAMPEL = int(CHUNK_DETIK * sr)
    WINDOW_5S_SAMPEL = 5 * sr          # 1 segmen model = 5 detik (80.000 sampel)
    TARGET_10S_SAMPEL = 10 * sr        # Target putusan penuh = 10 detik bicara (160.000 sampel)
    AMBANG_BICARA_RMS = 0.005          # Ambang batas suara aktif (bukan hening)

    q_audio = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        q_audio.put(indata[:, 0].copy())

    dev_info = sd.query_devices(kind='input')
    print('================================================================================')
    print('   REAL-TIME LIVE VOICE DETECTOR (STREAMING LANGSUNG DARI MIKROFON)             ')
    print('================================================================================')
    print(f' Mikrofon Aktif : {dev_info["name"]}')
    print(f' Cara Kerja     : Langsung ngomong saja! Suara diolah real-time saat Anda bicara.')
    print(f'                  - Detik ke-5 bicara  : Hasil Deteksi Segmen 1 langsung keluar')
    print(f'                  - Detik ke-10 bicara : VONIS AKHIR 10 DETIK langsung keluar')
    print(' Tekan [Ctrl + C] kapan saja untuk berhenti.')
    print('================================================================================\n')

    buffer_suara_aktif = np.zeros(0, dtype=np.float32)
    riwayat_proba_10s = []
    sudah_cek_5s_pertama = False
    sesi_ke = 1

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                            blocksize=CHUNK_SAMPEL, callback=audio_callback):
            while True:
                chunk = q_audio.get()
                rms_chunk = float(np.sqrt(np.mean(chunk**2)))

                # Buat visualisasi bar volume suara real-time
                level_bar = min(20, int((rms_chunk / 0.08) * 20))
                meter_str = '#' * level_bar + '-' * (20 - level_bar)

                if rms_chunk >= AMBANG_BICARA_RMS:
                    # Tambahkan potongan suara ke buffer bicara aktif
                    buffer_suara_aktif = np.concatenate([buffer_suara_aktif, chunk])
                    durasi_terkumpul = len(buffer_suara_aktif) / sr

                    print(f' [LIVE MIC] [{meter_str}] Terdeteksi Bicara: {durasi_terkumpul:4.1f} / 10.0 detik   ', end='\r', flush=True)

                    # 1. Begitu mencapai 5.0 detik pertama bicara -> Langsung olah Segmen 1 secara real-time!
                    if durasi_terkumpul >= 5.0 and not sudah_cek_5s_pertama:
                        sudah_cek_5s_pertama = True
                        segmen_1 = buffer_suara_aktif[:WINDOW_5S_SAMPEL]
                        proba_1, _ = prediksi_segmen_5s(segmen_1, sr, bundle)
                        riwayat_proba_10s.append(proba_1)
                        lbl_1 = 'MANUSIA ASLI' if proba_1[0] >= proba_1[1] else 'MANIPULASI AI'
                        print(f'\n   -> [5.0 Detik Pertama] Prediksi Sementara : {lbl_1:14s} (Asli: {proba_1[0]*100:5.1f}% | AI: {proba_1[1]*100:5.1f}%)')

                    # 2. Setiap tambahan 1 detik setelah 5 detik (detik 6, 7, 8, 9) -> Sliding window real-time!
                    elif durasi_terkumpul > 5.0 and durasi_terkumpul < 10.0 and abs(durasi_terkumpul - round(durasi_terkumpul)) < 0.01:
                        window_terbaru = buffer_suara_aktif[-WINDOW_5S_SAMPEL:]
                        proba_w, _ = prediksi_segmen_5s(window_terbaru, sr, bundle)
                        riwayat_proba_10s.append(proba_w)
                        mean_sementara = np.mean(riwayat_proba_10s, axis=0)
                        lbl_w = 'MANUSIA ASLI' if mean_sementara[0] >= mean_sementara[1] else 'MANIPULASI AI'
                        print(f'\n   -> [Update Detik {int(round(durasi_terkumpul)):2d} ] Analisis Real-Time : {lbl_w:14s} (Asli: {mean_sementara[0]*100:5.1f}% | AI: {mean_sementara[1]*100:5.1f}%)')

                    # 3. Begitu genap 10.0 detik bicara -> Keluarkan VONIS AKHIR 10 DETIK!
                    if len(buffer_suara_aktif) >= TARGET_10S_SAMPEL:
                        segmen_2 = buffer_suara_aktif[WINDOW_5S_SAMPEL:TARGET_10S_SAMPEL]
                        proba_2, fit_2 = prediksi_segmen_5s(segmen_2, sr, bundle)
                        riwayat_proba_10s.append(proba_2)

                        mean_final = np.mean(riwayat_proba_10s, axis=0)
                        idx_final = int(np.argmax(mean_final))
                        vonis_teks = 'SUARA MANUSIA ASLI (HUMAN VOICE)' if idx_final == 0 else 'SUARA MANIPULASI AI (AI VOICE)'

                        print('\n================================================================================')
                        print(f' [VONIS 10 DETIK - SESI #{sesi_ke}] KEPUTUSAN REAL-TIME KELUAR:')
                        print(f' >>> HASIL DETEKSI : {vonis_teks}')
                        print(f' >>> CONFIDENCE    : {mean_final[idx_final]*100:.2f}%  (Manusia Asli: {mean_final[0]*100:.2f}% | AI: {mean_final[1]*100:.2f}%)')
                        print(f' >>> Fitur Kunci   : mfcc_8={fit_2["mfcc_8_mean"]:.2f} | mfcc_11={fit_2["mfcc_11_mean"]:.2f} | f0_std={fit_2["f0_std"]:.2f} Hz')
                        print('================================================================================\n')

                        # Reset buffer otomatis agar siap mendengarkan kalimat berikutnya secara real-time
                        buffer_suara_aktif = np.zeros(0, dtype=np.float32)
                        riwayat_proba_10s = []
                        sudah_cek_5s_pertama = False
                        sesi_ke += 1

                else:
                    # Jika sedang diam / jeda napas, tetap tampilkan status mendengarkan tanpa menghapus progres
                    durasi_terkumpul = len(buffer_suara_aktif) / sr
                    print(f' [MENUNGGU] [{meter_str}] Silakan bicara... (Terkumpul: {durasi_terkumpul:4.1f} / 10.0 detik) ', end='\r', flush=True)

    except KeyboardInterrupt:
        print('\n\n[SELESAI] Streaming real-time dihentikan oleh pengguna.')


if __name__ == '__main__':
    main()
 