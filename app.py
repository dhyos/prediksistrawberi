import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # Import untuk Imputer
from sklearn.ensemble import RandomForestClassifier # Model Klasifikasi Final
from typing import List, Union # Union untuk model

# --- KONSTANTA (PENTING) ---
N_FEATURES = 235 # Total fitur mentah
FEATURE_COLUMNS = [f'Fitur_Time_{i}' for i in range(1, N_FEATURES + 1)]

# Menggunakan nama file dari pipeline Random Forest (RF) final
MODEL_FILES = {
    'imputer': 'imputer_ig_rf_13features_final.joblib', # Tambahkan Imputer
    'scaler': 'scaler_ig_rf_13features_final.joblib',
    'model': 'rf_model_ig_13features_final.joblib', # Model RF
    'features': 'features_ig_rf_13features_final.joblib' # Daftar 13 fitur terbaik IG
}

# --- 2. FUNGSI MEMUAT PIPELINE ---
@st.cache_resource
def load_pipeline():
    try:
        # Muat semua objek pipeline
        imputer = joblib.load(MODEL_FILES['imputer'])
        scaler = joblib.load(MODEL_FILES['scaler'])
        model = joblib.load(MODEL_FILES['model'])
        best_features = joblib.load(MODEL_FILES['features']) 
        
        # Mengembalikan Imputer, Scaler, Model, dan Daftar Fitur
        return imputer, scaler, model, best_features
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan: {e}. Pastikan keempat file .joblib berada di direktori yang sama.")
        return None, None, None, None

# --- 3. FUNGSI PRA-PEMROSESAN DAN PREDIKSI ---
def make_prediction(raw_data_df: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler, 
                    model: RandomForestClassifier, best_features: List[str]):
    
    # 1. Cek input dimensi mentah (harus 235 kolom)
    if raw_data_df.shape[1] != N_FEATURES:
        st.error(f"Input file harus memiliki {N_FEATURES} kolom data (titik spektral). Ditemukan {raw_data_df.shape[1]} kolom.")
        return None, None
    
    # 2. SELEKSI FITUR: Filter DataFrame hanya untuk fitur terbaik
    try:
        X_selected = raw_data_df[best_features]
    except KeyError as e:
        st.error(f"Error: Nama kolom fitur di file input tidak sesuai. Detail: {e}")
        return None, None
    
    # Konversi ke numpy array float (ini penting karena model dilatih dengan numpy)
    try:
        X_data = X_selected.values.astype(float)
    except ValueError as e:
        st.error(f"Error: Data input mengandung nilai non-numerik. Detail: {e}")
        return None, None

    # --- Eksekusi Pipeline Pra-pemrosesan ---
    
    # a. Imputasi (Mengisi NaN)
    # Ini penting jika data implementasi mengandung nilai yang hilang
    X_imputed = imputer.transform(X_data)
    
    # b. Standarisasi
    # Catatan: Walaupun RF tidak butuh scaling, kita tetap gunakan scaler untuk konsistensi data.
    X_scaled = scaler.transform(X_imputed)
    
    # c. Prediksi
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    
    return prediction[0], proba[0]

# --- 4. TAMPILAN UTAMA STREAMLIT ---
def main():
    st.set_page_config(page_title="Prediksi Adulterasi Stroberi", layout="wide")
    st.title("🍓 Aplikasi Klasifikasi Spektral Stroberi")
    st.markdown(f"""
        Aplikasi ini menggunakan model **Random Forest** yang dilatih pada **{len(best_features)} fitur terbaik** (Information Gain) 
        untuk memprediksi keaslian bubur stroberi.
    """)

    # Muat Pipeline
    imputer, scaler, model, best_features = load_pipeline()
    if not model:
        return

    # --- Input Data dari Pengguna (File Upload) ---
    st.header("Upload Data Spektral Baru")
    st.info(f"Model membutuhkan {len(best_features)} fitur. Harap unggah file yang berisi total {N_FEATURES} kolom data mentah.")
    
    uploaded_file = st.file_uploader(
        "Pilih file TXT/CSV untuk diuji",
        type=["txt", "csv"]
    )

    if uploaded_file is not None:
        try:
            # MEMBACA DATA DENGAN MENGABAIKAN HEADER
            
            data_string = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

            # Menggunakan regex untuk menangani koma atau spasi
            raw_input_df = pd.read_csv(
                StringIO(data_string), 
                sep='[,\s]+', 
                header=None, 
                skiprows=1, 
                engine='python' 
            )
            
            # 1. Pembersihan dan Verifikasi
            raw_input_df.dropna(axis=1, how='all', inplace=True)
            
            # 2. Penanganan Kolom Ekstra
            if raw_input_df.shape[1] > N_FEATURES:
                raw_input_df = raw_input_df.iloc[:, 1:] 
            
            # 3. Penamaan Kolom Penuh (WAJIB untuk filtering IG)
            if raw_input_df.shape[1] != N_FEATURES:
                 raise ValueError(f"Dimensi kolom tidak cocok setelah pembersihan: {raw_input_df.shape[1]}. Seharusnya {N_FEATURES}.")

            raw_input_df.columns = FEATURE_COLUMNS
            single_sample_df = raw_input_df.iloc[0:1].copy() # Hanya ambil 1 sampel pertama

            st.subheader(f"Data Sampel Input yang Akan Diproses ({N_FEATURES} Fitur)")
            st.dataframe(single_sample_df.head(), use_container_width=True)

            # --- Membuat Prediksi ---
            st.subheader("Hasil Prediksi")
            
            # Memanggil fungsi prediksi dengan 4 objek pipeline
            prediction, proba = make_prediction(single_sample_df, imputer, scaler, model, best_features)

            if prediction is not None:
                # Menampilkan Hasil Klasifikasi
                if prediction == 1:
                    result_text = "STROBERI MURNI (Pure Strawberry Purée)"
                    st.success(f"✅ Klasifikasi: {result_text}")
                elif prediction == 2:
                    result_text = "NON-STROBERI / TERADULTERASI"
                    st.warning(f"❌ Klasifikasi: {result_text}")
                else:
                    st.error("Model mengembalikan label yang tidak dikenal.")


                # Menampilkan Probabilitas
                st.info(f"""
                    **Probabilitas:**
                    * Kelas Murni (1): {proba[0]*100:.2f}%
                    * Kelas Non-Murni (2): {proba[1]*100:.2f}%
                """)

        except Exception as e:
            st.error(f"Error saat memproses file. Detail teknis: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()