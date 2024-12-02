import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Karir Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Judul Aplikasi
st.title("🎓 Prediksi Karir Mahasiswa 🎓")
st.markdown("Aplikasi prediksi karir mahasiswa menggunakan metode **Regresi Linier**.")

st.sidebar.header("Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["Dataset", "Visualisasi", "Prediksi Karir"])

# Fungsi untuk menyimpan model
def save_model(model, filename="model_prediksi_karir.sav"):
    pickle.dump(model, open(filename, "wb"))

# Fungsi untuk memuat model
def load_model(filename="model_prediksi_karir.sav"):
    return pickle.load(open(filename, "rb"))

# Load dataset
df = pd.read_csv("cs_students.csv")

# Encoding untuk `Interested Domain`
label_encoder = LabelEncoder()
df["Interested Domain Encoded"] = label_encoder.fit_transform(df["Interested Domain"])

# Preprocessing: Mengubah data kategori menjadi numerik untuk keperluan model
def encode_skills(skill_level):
    return {"Weak": 1, "Average": 2, "Strong": 3}.get(skill_level, 0)

df["Python"] = df["Python"].apply(encode_skills)
df["SQL"] = df["SQL"].apply(encode_skills)
df["Java"] = df["Java"].apply(encode_skills)

# Halaman Dataset
if menu == "Dataset":
    st.header("📊 Dataset Mahasiswa")
    
    # Expander dengan desain yang lebih modern
    with st.expander("🔍 Lihat Data Lengkap", expanded=True):
        st.dataframe(df.drop(columns=["Interested Domain Encoded"]), use_container_width=True)
    
    # Kolom dengan analisis statistik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Statistik Deskriptif")
        # Tampilkan statistik deskriptif
        desc_stats = df.drop(columns=["Interested Domain Encoded"]).describe()
        st.dataframe(desc_stats, use_container_width=True)
    
    with col2:
        st.subheader("🕵️ Analisis Data")
        # Informasi data kosong
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.write("**Jumlah Data Kosong per Kolom:**")
        missing_data = df.isnull().sum()
        for column, missing in missing_data.items():
            st.text(f"{column}: {missing}")
        st.markdown('</div>', unsafe_allow_html=True)

# Halaman Visualisasi
elif menu == "Visualisasi":
    st.header("📈 Visualisasi Data Mahasiswa")

    st.subheader("Distribusi Karir Masa Depan")
    fig, ax = plt.subplots(figsize=(10, 6))
    career_counts = df["Future Career"].value_counts().sort_values(ascending=False)
    sns.barplot(y=career_counts.index, x=career_counts.values, palette="viridis", ax=ax)
    ax.set_title("Distribusi Karir Masa Depan", fontsize=16)
    ax.set_xlabel("Jumlah", fontsize=14)
    ax.set_ylabel("Future Career", fontsize=14)
    st.pyplot(fig)

    st.subheader("Word Cloud Karir Masa Depan")
    all_career = " ".join(df["Future Career"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_career)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)
    
    st.subheader("Hubungan IPK Dengan Karir")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df["GPA"], y=df["Future Career"], color="purple", ax=ax)
    ax.set_title("IPK vs Karir", fontsize=16, fontweight="bold")
    st.pyplot(fig)

# Halaman Prediksi Karir
elif menu == "Prediksi Karir":
    st.header("🔮 Prediksi Karir Mahasiswa")
    
    # Fitur dan target
    X = df[["GPA", "Interested Domain Encoded", "Python", "SQL", "Java"]]
    y = df["Future Career"].astype("category").cat.codes
    
    # Split data untuk pelatihan model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    save_model(model)
    
    # Input Prediksi
    st.write("### Masukkan Data Mahasiswa:")
    gpa = st.number_input(
        "IPK Mahasiswa",
        min_value=0.0,
        max_value=4.0,
        value=3.5)
    
    interested_domain = st.selectbox(
        "Pilih Bidang yang Diminati",
        options=label_encoder.classes_)
    
    python_skill = st.select_slider(
        "Tingkat Keahlian Bahasa Pemrograman Python",
        options=["Weak", "Average", "Strong"], value="Average")
    
    sql_skill = st.select_slider(
        "Tingkat Keahlian Bahasa Pemrograman SQL",
        options=["Weak", "Average", "Strong"], value="Average")
    
    java_skill = st.select_slider(
        "Tingkat Keahlian Bahasa Pemrograman Java",
        options=["Weak", "Average", "Strong"], value="Average")
    
    # Konversi input
    python_skill = encode_skills(python_skill)
    sql_skill = encode_skills(sql_skill)
    java_skill = encode_skills(java_skill)
    interested_domain_encoded = label_encoder.transform([interested_domain])[0]
    
    if st.button("Prediksi Karir"):
        input_data = np.array([[gpa, interested_domain_encoded, python_skill, sql_skill, java_skill]])
        prediction = model.predict(input_data)
        predicted_career = df["Future Career"].astype("category").cat.categories[int(prediction[0])]
        st.success(f"Prediksi Karir Mahasiswa: **{predicted_career}**")

    # Evaluasi Model
    st.write("### Evaluasi Model:")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")