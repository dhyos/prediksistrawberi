import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from typing import List

# ================= KONSTANTA =================
N_FEATURES = 235
FEATURE_COLUMNS = [f'Fitur_Time_{i}' for i in range(1, N_FEATURES + 1)]

# 4 Fitur Terbaik dari Information Gain
BEST_FEATURES_LIST = ['Fitur_Time_180', 'Fitur_Time_183', 'Fitur_Time_57', 'Fitur_Time_58']

# Peta Fitur ke Rentang Gelombang Spektral (Contoh berdasarkan asumsi konteks spektral)
FEATURE_WAVENUMBER = {
    'Fitur_Time_180': 'Rentang 1800-1820 cm⁻¹ (Ikatan C=O Ester/Aldehida)',
    'Fitur_Time_183': 'Rentang 1700-1750 cm⁻¹ (Asam Lemak/Gula)',
    'Fitur_Time_57': 'Rentang 1000-1050 cm⁻¹ (Gugus C-O Karbohidrat/Gula)',
    'Fitur_Time_58': 'Rentang 1050-1100 cm⁻¹ (Cincin Polysaccharide)'
}

# Menggunakan nama file model yang sudah final
MODEL_NAME = 'svm_ig_4features_final'
MODEL_FILES = {
    'imputer': f'imputer_{MODEL_NAME}.joblib',
    'scaler': f'scaler_{MODEL_NAME}.joblib',
    'model': f'model_{MODEL_NAME}.joblib',
    'features': f'features_{MODEL_NAME}.joblib'
}

# ================= LOAD PIPELINE =================
@st.cache_resource
def load_pipeline():
    try:
        imputer = joblib.load(MODEL_FILES['imputer'])
        scaler = joblib.load(MODEL_FILES['scaler'])
        model = joblib.load(MODEL_FILES['model'])
        best_features = BEST_FEATURES_LIST

        return imputer, scaler, model, best_features
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan: {e}. Pastikan keempat file .joblib ada.")
        return None, None, None, None

# ================= PREDIKSI =================
def make_prediction(
    input_data: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    model: SVC,
    best_features: List[str]
):
    # Data input sudah difilter di main()
    # Pastikan hanya kolom yang diperlukan yang diambil untuk preprocessing
    X = input_data[best_features].values.astype(float)
    
    # Pipeline preprocessing
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # Prediksi
    prediction = model.predict(X_scaled)
    # proba[0] -> Probabilitas Kelas 1 (Murni), proba[1] -> Probabilitas Kelas 2 (Adulterasi)
    proba = model.predict_proba(X_scaled) 

    return prediction[0], proba[0]

# ================= STREAMLIT APP =================
def main():
    st.set_page_config(
        page_title="Prediksi Adulterasi Stroberi",
        layout="wide"
    )

    st.title("🍓 Aplikasi Klasifikasi Spektral Stroberi (SVM)")
    st.markdown("---")

    # Muat Model
    imputer, scaler, model, best_features = load_pipeline()
    if model is None:
        return
        
    st.markdown(f"""
    **Model:** Support Vector Machine (SVM) dengan kernel RBF.  
    **Fitur:** Menggunakan **{len(best_features)} fitur terbaik** yang diseleksi dari total **{N_FEATURES} titik spektral** mentah.
    """)
    
    # PILIH METODE INPUT
    input_method = st.radio(
        "Pilih Metode Input Data:",
        ('Input Manual (4 Fitur)', 'Upload File Spektral Lengkap (235 Fitur)'),
        horizontal=True
    )
    
    st.markdown("---")
    
    input_data = None

    if input_method == 'Input Manual (4 Fitur)':
        # ======== INPUT MANUAL ========
        st.header("📝 Input Data Manual")
        st.info(f"""
        **Konteks Data:** Dataset spektral asli terdiri dari {N_FEATURES} titik absorbansi/reflektansi.
        
        **Representasi Fitur:** Model ini hanya menggunakan **{len(best_features)} fitur** yang dipilih melalui metode **Information Gain (IG)**. Fitur-fitur ini dianggap paling relevan dan representatif karena menunjukkan korelasi tertinggi dengan label kelas (Murni/Adulterasi).
        
        Masukkan nilai mentah absorbansi/reflektansi untuk 4 fitur utama berikut:
        """)
        
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        # Input Fitur 1 dan 2
        with col1:
            input_values[best_features[0]] = st.number_input(
                f"1. {best_features[0]} - {FEATURE_WAVENUMBER[best_features[0]]}", 
                value=0.5, format="%.6f", key='val_180'
            )
            input_values[best_features[1]] = st.number_input(
                f"2. {best_features[1]} - {FEATURE_WAVENUMBER[best_features[1]]}", 
                value=0.5, format="%.6f", key='val_183'
            )
        # Input Fitur 3 dan 4
        with col2:
            input_values[best_features[2]] = st.number_input(
                f"3. {best_features[2]} - {FEATURE_WAVENUMBER[best_features[2]]}", 
                value=0.5, format="%.6f", key='val_57'
            )
            input_values[best_features[3]] = st.number_input(
                f"4. {best_features[3]} - {FEATURE_WAVENUMBER[best_features[3]]}", 
                value=0.5, format="%.6f", key='val_58'
            )
            
        # Bentuk DataFrame input untuk diproses (hanya mengisi 4 kolom dari total 235)
        
        input_data_row = [0.0] * N_FEATURES
        
        # Isi nilai 4 fitur terpilih pada indeks kolom yang sesuai
        for feature, value in input_values.items():
            try:
                index = FEATURE_COLUMNS.index(feature)
                input_data_row[index] = value
            except ValueError:
                 st.warning(f"Fitur {feature} tidak ditemukan dalam daftar {N_FEATURES} fitur spektral.")
                 return

        input_data = pd.DataFrame([input_data_row], columns=FEATURE_COLUMNS)
        
        st.caption(f"Nilai mentah yang akan diproses:")
        st.dataframe(input_data[best_features], use_container_width=True)


    elif input_method == 'Upload File Spektral Lengkap (235 Fitur)':
        # ======== FILE UPLOAD (Kode tidak berubah dari sebelumnya) ========
        st.header("📤 Upload File Spektral Lengkap")
        st.info(f"Unggah file yang berisi {N_FEATURES} kolom data spektral (TXT/CSV). Model hanya akan menggunakan {len(best_features)} fitur terbaik.")

        uploaded_file = st.file_uploader(
            "Pilih file CSV / TXT",
            type=["csv", "txt"]
        )
        
        if uploaded_file is not None:
            try:
                data_string = uploaded_file.getvalue().decode("utf-8")
                raw_df = pd.read_csv(
                    StringIO(data_string), sep='[,\s]+', header=None, engine='python'
                )
    
                if raw_df.iloc[0].astype(str).str.contains("Fitur_Time").any():
                    raw_df = raw_df.iloc[1:].reset_index(drop=True)
                raw_df.dropna(axis=1, how='all', inplace=True)
                
                if raw_df.shape[1] > N_FEATURES:
                    raw_df = raw_df.iloc[:, 1:]
    
                if raw_df.shape[1] != N_FEATURES:
                    st.error(f"Jumlah kolom tidak sesuai: {raw_df.shape[1]}. Harusnya {N_FEATURES}.")
                    return
    
                raw_df.columns = FEATURE_COLUMNS
                input_data = raw_df.iloc[[0]].copy()
    
                st.subheader("Contoh Data Input (1 Sampel Awal)")
                st.dataframe(input_data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error saat memuat file: {type(e).__name__} - {e}")
                return


    # ======== EKSEKUSI PREDIKSI ========
    if input_data is not None:
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        try:
            pred, proba = make_prediction(
                input_data, imputer, scaler, model, best_features
            )
    
            if pred is not None:
                if pred == 1:
                    st.success("✅ KLASIFIKASI: STROBERI MURNI")
                elif pred == 2:
                    st.warning("❌ KLASIFIKASI: STROBERI TERADULTERASI")
                else:
                    st.error("Label tidak dikenali")
                    return
    
                # Tampilkan Probabilitas
                st.info(f"""
                **Probabilitas:** - Murni (1): **{proba[0]*100:.2f}%** - Adulterasi (2): **{proba[1]*100:.2f}%**
                """)
                
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {type(e).__name__} - {e}")


if __name__ == "__main__":
    main()