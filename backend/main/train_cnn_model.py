import os
import sys

# Tambahkan direktori saat ini ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from cnn_architecture import AudioCNN1D, CNNVoiceClassifier

csv_path = 'uas/AudioClasification/hasil_ekstraksi/file_seleksi_fitur.csv'
df = pd.read_csv(csv_path)

meta_cols = ['id_sampel', 'file_induk', 'segmen_ke', 'durasi_detik', 'kelas', 'label']
feature_cols = [c for c in df.columns if c not in meta_cols]

unique_files = sorted(df['file_induk'].unique())
train_files, test_files = train_test_split(unique_files, test_size=0.20, random_state=42)

df_train = df[df['file_induk'].isin(train_files)].copy()
df_test = df[df['file_induk'].isin(test_files)].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols].values)
y_train = df_train['label'].values
X_test = scaler.transform(df_test[feature_cols].values)
y_test = df_test['label'].values

if __name__ == '__main__':
    print('Training AudioCNN1D Deep Learning Model...')
    torch.manual_seed(42)
    np.random.seed(42)

    cnn = CNNVoiceClassifier(in_features=len(feature_cols), num_classes=2, epochs=50)
    cnn.fit(X_train, y_train)

    preds = cnn.predict(X_test)
    probs = cnn.predict_proba(X_test)

    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds) * 100
    rec = recall_score(y_test, preds) * 100
    f1 = f1_score(y_test, preds) * 100

    metrics = {
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2)
    }
    print('Metrics:', metrics)
    print('Confusion Matrix:\n', confusion_matrix(y_test, preds))

    mean_asli = df[df['label'] == 0][feature_cols].mean().to_dict()
    mean_ai = df[df['label'] == 1][feature_cols].mean().to_dict()

    paket_cnn = {
        'model_name': 'AudioCNN1D (Deep Learning Convolutional Neural Network)',
        'model': cnn,
        'scaler': scaler,
        'kolom_fitur': feature_cols,
        'metrics': metrics,
        'mean_asli': mean_asli,
        'mean_ai': mean_ai,
        'label_map': {0: 'Human Voice', 1: 'AI Generated Voice'},
        'sr': 16000,
        'durasi_segmen': 5
    }

    BASE_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.makedirs(os.path.join(BASE_PROJ, 'backend', 'main'), exist_ok=True)
    os.makedirs(os.path.join(BASE_PROJ, 'web-statis-voice-detection'), exist_ok=True)

    pkl_targets = [
        os.path.join(BASE_PROJ, 'backend', 'main', 'model_voice_cnn_5s.pkl'),
        os.path.join(BASE_PROJ, 'web-statis-voice-detection', 'model_voice_cnn_5s.pkl')
    ]

    for p in pkl_targets:
        joblib.dump(paket_cnn, p)
        print(f'Exported: {p}')

    torch.save(cnn.model.state_dict(), os.path.join(BASE_PROJ, 'backend', 'main', 'model_voice_cnn_5s.pt'))
    torch.save(cnn.model.state_dict(), os.path.join(BASE_PROJ, 'web-statis-voice-detection', 'model_voice_cnn_5s.pt'))
    print('Saved PyTorch state dicts (.pt)!')
