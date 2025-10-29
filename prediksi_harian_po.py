import joblib
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# Base directory: folder script Python ini
base_dir = os.path.dirname(__file__)

encoder_path = os.path.join(base_dir, "encoder_harian_prediksi_po.joblib")
model_path = os.path.join(base_dir, "model_harian_prediksi_po.joblib")
csv_path = os.path.join(base_dir, "dataset_prediksi_PO_LaTansa_2024.csv")
output_path = os.path.join(base_dir, "hasil_prediksi_harian_PO.csv")

# Load model dan encoder
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

# Load dataset
df = pd.read_csv(csv_path)

# Encoding
for col, le in encoder.items():
    if col in df.columns:
        df[col] = df[col].astype(str)
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])
        df[col] = le.transform(df[col])

# Pastikan semua fitur ada
model_features = [
    'Produk', 'Harga', 'Hari', 'Bulan', 'Event_Hari_Besar', 'Promo',
    'Total_Pendapatan_PO', 'Tahun', 'WeekOfYear', 'DayOfMonth',
    'Lag_1', 'Lag_3', 'Lag_7', 'Lag_14',
    'RollingMean_7', 'RollingStd_7', 'RollingMean_14', 'RollingStd_14'
]
for feature in model_features:
    if feature not in df.columns:
        df[feature] = 0

df_model = df[model_features]
prediksi = model.predict(df_model)

# Optional: jika ada scaler
scaler_y_path = os.path.join(base_dir, "scaler_y_harian_prediksi_po.joblib")
if os.path.exists(scaler_y_path):
    scaler_y = joblib.load(scaler_y_path)
    prediksi = scaler_y.inverse_transform(prediksi.reshape(-1, 1)).ravel()

df["Prediksi_Jumlah_PO"] = prediksi

# Simpan hasil prediksi
df.to_csv(output_path, index=False)
print(f"Hasil prediksi tersimpan di {output_path}")
