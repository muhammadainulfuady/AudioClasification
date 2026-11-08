import os
import io
import base64
import joblib
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Impor arsitektur CNN jika paket model menggunakan PyTorch
try:
    from cnn_architecture import AudioCNN1D, CNNVoiceClassifier
except ImportError:
    pass

# Lokasi file model .pkl dan folder dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'main', 'model_voice_cnn_5s.pkl')

DATA_DIR = os.path.join(BASE_DIR, 'data', 'dataset_audio_kaggle')

# Dataset fitur terseleksi yang sudah diekstrak untuk pengujian instan
CSV_SELEKSI = os.path.join(BASE_DIR, 'hasil_ekstraksi', 'file_seleksi_fitur.csv')
if not os.path.exists(CSV_SELEKSI):
    CSV_SELEKSI = os.path.join(BASE_DIR, 'data', 'hasil_ekstraksi', 'file_seleksi_fitur.csv')

DF_FITUR_PRECOMPUTED = None
if os.path.exists(CSV_SELEKSI):
    DF_FITUR_PRECOMPUTED = pd.read_csv(CSV_SELEKSI)

# Muat paket model (.pkl) berbasis fitur terseleksi saat server dijalankan
paket_model = joblib.load(MODEL_PATH)
cnn_model = paket_model['model']
scaler = paket_model['scaler']
kolom_fitur = paket_model['kolom_fitur']
TARGET_SR = paket_model.get('sr', 16000)
DURASI_SEGMEN = paket_model.get('durasi_segmen', 5)
LABEL_MAP = paket_model.get('label_map', {
    0: 'Human Voice',
    1: 'AI Generated Voice'
})
MEAN_ASLI = paket_model.get('mean_asli', {})
MEAN_AI = paket_model.get('mean_ai', {})


def ekstrak_fitur_segmen(y_seg, sr, f0_seg):
    """Mengekstrak fitur audio Librosa dari 1 segmen suara berdurasi 5 detik."""
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


def buat_visualisasi_prediksi(y_audio, sr, nama_sumber):
    """Membuat visualisasi Matplotlib ringkas (Waveform & Mel-Spectrogram) untuk inferensi langsung."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    durasi_total = len(y_audio) / sr
    waktu = np.linspace(0, durasi_total, len(y_audio))

    # Plot 1: Waveform Sinyal Audio
    axes[0].plot(waktu, y_audio, color='#2563eb', alpha=0.85, linewidth=0.7)
    axes[0].set_title(f'Bentuk Gelombang Audio ({nama_sumber})', fontweight='bold', fontsize=10)
    axes[0].set_xlabel('Waktu (Detik)')
    axes[0].set_ylabel('Amplitudo')
    axes[0].set_xlim(0, durasi_total)
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # Plot 2: Mel-Spectrogram
    S_mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S_mel, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
    axes[1].set_title('Mel-Spectrogram Frekuensi', fontweight='bold', fontsize=10)
    axes[1].set_xlabel('Waktu (Detik)')
    axes[1].set_ylabel('Frekuensi (Hz)')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def analisis_audio_array(y_raw, sr_in, nama_sumber='Rekaman Suara', is_mic=False):
    """Menjalankan inferensi langsung model Deep Learning CNN tanpa looping segmen berulang."""
    if y_raw.ndim > 1:
        y_raw = np.mean(y_raw, axis=1)
    y_raw = y_raw.astype(np.float32)

    if sr_in != TARGET_SR:
        y_audio = librosa.resample(y_raw, orig_sr=sr_in, target_sr=TARGET_SR)
    else:
        y_audio = y_raw

    # Jika berasal dari mikrofon langsung, sesuaikan level energi RMS
    if is_mic:
        y_trimmed, _ = librosa.effects.trim(y_audio, top_db=28)
        if len(y_trimmed) > TARGET_SR:
            y_audio = y_trimmed
        rms_curr = float(np.sqrt(np.mean(y_audio**2)))
        if rms_curr > 1e-4:
            y_audio = y_audio * (0.023 / rms_curr)
            y_audio = np.clip(y_audio, -1.0, 1.0)

    sampel_5s = DURASI_SEGMEN * TARGET_SR  # 80.000 sampel per 5 detik
    durasi_asli = round(len(y_audio) / TARGET_SR, 2)

    # Ambil audio representatif 5 detik secara langsung dengan mendeteksi suara aktif
    if len(y_audio) > sampel_5s:
        win = int(0.25 * TARGET_SR)  # 4000 sampel per 0.25 detik
        energi_frame = np.array([np.mean(y_audio[i:i+win]**2) for i in range(0, len(y_audio) - win, win // 2)])
        ambang = max(1e-5, float(np.percentile(energi_frame, 30)))
        idx_aktif = np.where(energi_frame > ambang)[0]
        if len(idx_aktif) > 0:
            start_idx = max(0, int(idx_aktif[0] * (win // 2)))
            if start_idx + sampel_5s <= len(y_audio):
                y_5s = y_audio[start_idx : start_idx + sampel_5s]
            else:
                y_5s = y_audio[-sampel_5s:]
        else:
            y_5s = y_audio[:sampel_5s]
    elif len(y_audio) < sampel_5s:
        ulang = int(np.ceil(sampel_5s / max(1, len(y_audio))))
        y_5s = np.tile(y_audio, ulang)[:sampel_5s]
    else:
        y_5s = y_audio

    # Ekstraksi cepat 20 fitur akustik terpilih dalam 1 kali proses (tanpa segment looping)
    f0_seg = librosa.yin(y_5s, fmin=60, fmax=360, sr=TARGET_SR, hop_length=1024)
    d_fitur = ekstrak_fitur_segmen(y_5s, TARGET_SR, f0_seg)
    fitur_vec = [d_fitur[k] for k in kolom_fitur]

    # Inferensi langsung model Deep Learning CNN
    X_input = scaler.transform(np.array([fitur_vec]))
    proba = cnn_model.predict_proba(X_input)[0]  # shape: (2,)
    pred_label = int(np.argmax(proba))

    prob_asli = round(float(proba[0]) * 100, 2)
    prob_ai = round(float(proba[1]) * 100, 2)

    rincian_segmen = [{
        'segmen_ke': 1,
        'rentang_waktu': f'0 - {min(durasi_asli, 5.0)} detik',
        'prediksi_label': pred_label,
        'prediksi_teks': 'Human Voice' if pred_label == 0 else 'AI Generated Voice',
        'prob_asli': prob_asli,
        'prob_ai': prob_ai
    }]

    # Rata-rata fitur penting untuk ditampilkan di tabel perbandingan
    fitur_utama = [f for f in ['mfcc_8_mean', 'mfcc_11_mean', 'mfcc_7_mean', 'mfcc_8_std', 'mfcc_4_mean', 'skewness', 'f0_std'] if f in kolom_fitur]
    perbandingan_fitur = []
    for nf in fitur_utama:
        perbandingan_fitur.append({
            'nama_fitur': nf,
            'nilai_suara_uji': round(float(d_fitur[nf]), 4),
            'rata2_asli': round(float(MEAN_ASLI.get(nf, 0.0)), 4),
            'rata2_ai': round(float(MEAN_AI.get(nf, 0.0)), 4)
        })

    plot_base64 = buat_visualisasi_prediksi(y_5s, TARGET_SR, nama_sumber)

    return {
        'status': 'success',
        'nama_sumber': nama_sumber,
        'durasi_detik': durasi_asli,
        'jumlah_segmen_5s': 1,
        'prediksi_label': pred_label,
        'prediksi_nama': LABEL_MAP[pred_label],
        'confidence_utama': round(float(proba[pred_label]) * 100, 2),
        'prob_asli': prob_asli,
        'prob_ai': prob_ai,
        'rincian_segmen': rincian_segmen,
        'perbandingan_fitur': perbandingan_fitur,
        'plot_base64': plot_base64
    }


@app.route('/')
def index():
    # Siapkan daftar contoh file dari dataset
    sampel_asli = sorted([f for f in os.listdir(os.path.join(DATA_DIR, 'asli_human')) if f.endswith('.wav')])[:15]
    sampel_ai = sorted([f for f in os.listdir(os.path.join(DATA_DIR, 'ai_manipulasi')) if f.endswith('.wav')])[:15]
    # Sesuai poin 4 PROMT.md: tidak menampilkan akurasi/metrik model pada halaman pengujian
    return render_template(
        'index.html',
        sampel_asli=sampel_asli,
        sampel_ai=sampel_ai,
        jumlah_fitur=len(kolom_fitur),
        nama_model=paket_model.get('model_name', 'Deep Learning 1D-CNN (5s)')
    )


@app.route('/sample_audio/<kelas>/<filename>')
def sample_audio(kelas, filename):
    if kelas not in ['asli_human', 'ai_manipulasi']:
        return jsonify({'status': 'error', 'message': 'Kelas tidak valid'}), 400
    folder = os.path.join(DATA_DIR, kelas)
    return send_from_directory(folder, filename)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Jika memilih sampel langsung dari dataset -> LANGSUNG PREDIKSI MENGGUNAKAN FITUR PRECOMPUTED
        sample_kelas = request.form.get('sample_kelas')
        sample_file = request.form.get('sample_file')
        if sample_kelas and sample_file:
            wav_path = os.path.join(DATA_DIR, sample_kelas, sample_file)
            if not os.path.exists(wav_path):
                return jsonify({'status': 'error', 'message': 'File sampel tidak ditemukan.'}), 404

            # Jika sampel sudah ada dalam dataset seleksi fitur, langsung prediksi instan
            if DF_FITUR_PRECOMPUTED is not None:
                sub = DF_FITUR_PRECOMPUTED[
                    (DF_FITUR_PRECOMPUTED['kelas'] == sample_kelas) &
                    (DF_FITUR_PRECOMPUTED['file_induk'] == sample_file)
                ]
                if len(sub) > 0:
                    fitur_vec = sub[kolom_fitur].mean().values
                    X_input = scaler.transform(np.array([fitur_vec]))
                    proba = cnn_model.predict_proba(X_input)[0]
                    pred_label = int(np.argmax(proba))
                    prob_asli = round(float(proba[0]) * 100, 2)
                    prob_ai = round(float(proba[1]) * 100, 2)

                    y_raw, sr_in = sf.read(wav_path, dtype='float32')
                    durasi_total = round(len(y_raw) / sr_in, 2)
                    plot_base64 = buat_visualisasi_prediksi(y_raw[:sr_in * DURASI_SEGMEN], sr_in, f'{sample_kelas}/{sample_file}')

                    fitur_utama = [f for f in ['mfcc_8_mean', 'mfcc_11_mean', 'mfcc_7_mean', 'mfcc_8_std', 'mfcc_4_mean', 'skewness', 'f0_std'] if f in kolom_fitur]
                    perbandingan_fitur = []
                    for nf in fitur_utama:
                        perbandingan_fitur.append({
                            'nama_fitur': nf,
                            'nilai_suara_uji': round(float(fitur_vec[kolom_fitur.index(nf)]), 4),
                            'rata2_asli': round(float(MEAN_ASLI.get(nf, 0.0)), 4),
                            'rata2_ai': round(float(MEAN_AI.get(nf, 0.0)), 4)
                        })

                    return jsonify({
                        'status': 'success',
                        'nama_sumber': f'{sample_kelas}/{sample_file}',
                        'durasi_detik': durasi_total,
                        'jumlah_segmen_5s': 1,
                        'prediksi_label': pred_label,
                        'prediksi_nama': LABEL_MAP[pred_label],
                        'confidence_utama': round(float(proba[pred_label]) * 100, 2),
                        'prob_asli': prob_asli,
                        'prob_ai': prob_ai,
                        'rincian_segmen': [{
                            'segmen_ke': 1,
                            'rentang_waktu': f'0 - {min(durasi_total, 5.0)} detik',
                            'prediksi_label': pred_label,
                            'prediksi_teks': 'Human Voice' if pred_label == 0 else 'AI Generated Voice',
                            'prob_asli': prob_asli,
                            'prob_ai': prob_ai
                        }],
                        'perbandingan_fitur': perbandingan_fitur,
                        'plot_base64': plot_base64
                    })

            # Jika file fisik lain, proses langsung tanpa segment looping
            y_raw, sr_in = sf.read(wav_path, dtype='float32')
            hasil = analisis_audio_array(
                y_raw, sr_in,
                nama_sumber=f'{sample_kelas}/{sample_file}',
                is_mic=False
            )
            return jsonify(hasil)

        # 2. Jika mengunggah file audio atau hasil rekaman mikrofon langsung
        if 'audio_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file suara yang dikirim.'}), 400

        file_obj = request.files['audio_file']
        is_mic = request.form.get('is_mic', 'false').lower() == 'true'
        nama_file = file_obj.filename or ('Rekaman_Mikrofon.wav' if is_mic else 'Upload_Audio.wav')

        audio_bytes = file_obj.read()
        try:
            y_raw, sr_in = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        except Exception:
            # Fallback menggunakan librosa load jika format butuh backend alternatif
            import tempfile
            ext = os.path.splitext(nama_file)[1] or '.wav'
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                y_raw, sr_in = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        hasil = analisis_audio_array(y_raw, sr_in, nama_sumber=nama_file, is_mic=is_mic)
        return jsonify(hasil)

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal memproses audio: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
