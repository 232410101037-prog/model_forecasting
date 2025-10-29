import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

encoder = joblib.load("python/encoder_harian_prediksi_po.joblib")
print(type(encoder))
print(encoder)
data = joblib.load("python/model_harian_prediksi_po.joblib")

print(type(data))
print(data)

# {'Produk': LabelEncoder(), 'Hari': LabelEncoder(), 'Bulan': LabelEncoder(), 'Event_Hari_Besar': LabelEncoder(), 'Promo': LabelEncoder()}