from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Path
base_dir = os.path.dirname(__file__)
encoder_path = os.path.join(base_dir, "encoder_harian_prediksi_po.joblib")
model_path = os.path.join(base_dir, "model_harian_prediksi_po.joblib")
scaler_path = os.path.join(base_dir, "scaler_y_harian_prediksi_po.joblib")

# Load model dan encoder
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
scaler_y = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Contoh data input (nanti bisa dari DB atau request)
df = pd.read_csv(os.path.join(base_dir, "dataset_prediksi_PO_LaTansa_2024.csv"))

# Encoding
for col, le in encoder.items():
    if col in df.columns:
        df[col] = df[col].astype(str)
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])
        df[col] = le.transform(df[col])

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

if scaler_y:
    prediksi = scaler_y.inverse_transform(prediksi.reshape(-1, 1)).ravel()

df["Prediksi_Jumlah_PO"] = prediksi

# API endpoint
@app.route("/prediksi", methods=["GET"])
def get_prediksi():
    # Ubah hasil prediksi menjadi list dict
    data = []
    for _, row in df.iterrows():
        data.append({
            "Produk": row["Produk"],
            "Prediksi_Jumlah_PO": float(row["Prediksi_Jumlah_PO"])
        })
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
