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

# Fungsi untuk menyimpan model
def save_model(model, filename="model_prediksi_karir.sav"):
    pickle.dump(model, open(filename, "wb"))

# Fungsi untuk memuat model
def load_model(filename="model_prediksi_karir.sav"):
    return pickle.load(open(filename, "rb"))

# Load dataset
df = pd.read_csv("cs_students.csv")

# Encoding untuk Interested Domain
label_encoder = LabelEncoder()
df["Interested Domain Encoded"] = label_encoder.fit_transform(df["Interested Domain"])

project_encoder = LabelEncoder()
df["Projects Encoded"] = project_encoder.fit_transform(df["Projects"])

# Judul Halaman
st.markdown(
    """
    <style>
        .main-title {
            font-size: 3.5rem;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(45deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px 0;
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 1.8rem;
            color: #757575;
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background-color: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: scale(1.03);
        }
        .feature-icon {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 10px;
            color: #4CAF50;
        }
    </style>
    <h1 class="main-title">🎓 Prediksi Karir Mahasiswa</h1>
    <p class="sub-title">Jelajahi data, visualisasi, dan prediksi karir berdasarkan keahlian mahasiswa</p>
    """,
    unsafe_allow_html=True,
)

# Tab navigasi di bagian atas
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Home", "📊 Dataset", "📈 Visualisasi", "🔮 Prediksi Karir"]
)

# Tab Home
with tab1:
    # Header utama dengan kartu gaya material
    st.markdown(
        """
        <div class="card">
            <h2 style="text-align: center; color: #4CAF50;">🎉 Selamat Datang di Website Prediksi Karir Mahasiswa! 🎉</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Menampilkan gambar
    st.image("coba.jpg", use_container_width=True)

    # Informasi pengembang dengan kartu
    st.markdown(
    """
    <div class="card">
        <h3 style="color: #2196F3;">👨‍💻 Tentang Web Prediksi Karir Mahasiswa</h3>
        <p>Sebuah platform berbasis web menggunakan Framework Streamlit yang dirancang untuk membantu mahasiswa jurusan IT dalam memahami potensi karir mereka berdasarkan faktor-faktor penting seperti Indeks Prestasi Kumulatif (IPK), Project yang telah dikembangkan, dan minat terhadap bidang tertentu. Pembuatan web ini memanfaatkan teknologi Machine Learning dengan metode regresi linier untuk menganalisis data mahasiswa dan memberikan prediksi yang relevan tentang jalur karir yang sesuai dengan profil mereka. Dengan visualisasi data yang informatif dan interaktif, aplikasi ini memberikan wawasan yang dapat membantu mahasiswa membuat keputusan yang lebih baik terkait rencana masa depan mereka.</p>
        <h3>Latar Belakang Pembuatan Website</h3>
        <p>Alasan di balik pembuatan web prediksi karir mahasiswa adalah untuk membantu mahasiswa, khususnya jurusan IT, mengatasi kebingungan dalam menentukan jalur karir yang sesuai dengan kemampuan, minat, dan pencapaian akademik Mahasiswa. Dengan banyaknya pilihan karir di bidang teknologi yang serba luas, sering kali mahasiswa merasa kesulitan dalam memilih jalur yang paling tepat. Aplikasi ini dirancang untuk memberikan panduan berbasis data dengan memanfaatkan teknologi analitik dan pembelajaran mesin, sehingga mahasiswa dapat memahami potensi mereka secara objektif dan merencanakan masa depan dengan lebih percaya diri.</p>
        <ul>
            <li>🔍 <strong>Pengembang:</strong> Kelompok 3 - Farhan, Favian, Arung</li>
            <li>📦 <strong>Sumber Dataset: <a href="https://www.kaggle.com/datasets/devildyno/computer-science-students-career-prediction" target="_blank">Link ke Dataset</a></strong></li>
            <li>💻 <strong>Bahan Pembuatan:</strong> Framework Streamlit, Scikit-learn, Pandas, Matplotlib, Seaborn, Wordcloud</li>
        </ul>
    </div>
    """, 
    unsafe_allow_html=True
)


    # Fitur utama dengan kartu
    st.markdown(
        """
        <div class="card">
            <h3 style="color: #FF9800;">🚀 Fitur Utama</h3>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div class="feature-icon">📊</div>
                    <p><strong>Dataset</strong><br>Lihat data mahasiswa</p>
                </div>
                <div>
                    <div class="feature-icon">📈</div>
                    <p><strong>Visualisasi</strong><br>Menjelajahi grafik karir</p>
                </div>
                <div>
                    <div class="feature-icon">🤖</div>
                    <p><strong>Prediksi Karir</strong><br>Rekomendasi personal</p>
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Penutup
    st.success("Pilih tab di atas untuk mulai menjelajahi fitur!")

# Tab Dataset
with tab2:
    st.header("📊 Dataset Mahasiswa")
    with st.expander("Lihat Data"):
        st.dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
    with col2:
        st.subheader("Data Kosong")
        st.write(df.isnull().sum())

    # Penjelasan encoding
    st.markdown("### Penjelasan Pengunaan Encoding")
    st.write("""
    Encoding adalah proses yang digunakan untuk mealukan konversi data dari format teks atau kategori menjadi format numerik yang dapat diproses oleh algoritma komputer.Penggunaan encoding pada project yaitu data kategori bidang minat atau Project yang dikembangkan yang bentuk awalnya berupa teks,tipe data yang digunakan pada proses ini yaitu berupa Any artinya tipe data ini dapat menggunakan tipe data apapun.
    - **Label_Encoded**: Digunakan untuk mengonversi bidang minat menjadi numerik.
    - **Project_Encoded**: Digunakan untuk mengonversi Project mahasiswa menjadi numerik.
    """)

with tab3:
    st.header("📈 Visualisasi Data Mahasiswa")

    # Distribusi Karir
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Distribusi Karir Mahasiswa")
        career_counts = df["Future Career"].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        career_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title("Distribusi Karir Mahasiswa", fontsize=16, fontweight='bold')
        ax.set_xlabel("Future Career", fontsize=14)
        ax.set_ylabel("Number of Students", fontsize=14)
        plt.xticks(rotation=45, fontsize=12, ha='right')
        plt.yticks(fontsize=12)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### 📊 Interpretasi Grafik Distribusi Karir
        
        Grafik ini menggambarkan pendistibusian pilihan karir mahasiswa yaitu dengan
        menunjukkan distribusi karir mahasiswa berdasarkan jumlah mahasiswa yang bercita-cita untuk setiap profesi tertentu. Sumbu horizontal menunjukkan berbagai pilihan karir, seperti Web Developer, Mobile App Developer, Machine Learning Engineer, dan lainnya, sementara sumbu vertikal menunjukkan jumlah mahasiswa yang memilih profesi tersebut.
        
        **Insights Utama:**
        - Karir paling diminati
        - Variasi pilihan karir
        - Distribusi kesempatan kerja
        """, unsafe_allow_html=True)

    # Word Cloud
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("WordCloud Karir Mahasiswa")
        all_career = " ".join(df["Future Career"].astype(str))
        wordcloud = WordCloud(
            width=800, 
            height=300, 
            background_color="white", 
            colormap='viridis'
        ).generate(all_career)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### 📝 Interpretasi Word Cloud
        
        Word cloud menvisualisasikan frekuensi karir melalui ukuran teks menggambarkan jumlah mahasiswa yang memilih karir tersebut,Warna gradient menunjukkan variasi pilihan serta memudahkan identifikasi karir populer sekilas
        
        **Cara Membaca:**
        - Kata besar = Karir sangat diminati
        - Kata kecil = Karir kurang populer
        - Warna = Variasi pilihan
        """, unsafe_allow_html=True)
    
    # Seaborn
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Hubungan IPK Dengan Karir")
        fig, ax = plt.subplots(figsize=(12, 8))  # Ukuran grafik diperbesar
        sns.scatterplot(
            x=df["GPA"],
            y=df["Future Career"],
            color="purple",
            s=70,  # Ukuran marker diperbesar
            alpha=0.8,  # Transparansi untuk keterbacaan
            ax=ax,
        )
        
        ax.set_title("IPK vs Karir", fontsize=18, fontweight="bold")
        ax.set_xlabel("GPA", fontsize=14, labelpad=10)
        ax.set_ylabel("Future Career", fontsize=14, labelpad=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)
        plt.tight_layout() 
        st.pyplot(fig)

    with col2:
        st.markdown("""
        ### 📝 Interpretasi Seaborn
        
        Seaborn akan menvisualisasikan grafik mahasiswa yang akan meraih pada karir tersebut
        Berdasarkan IPK (GPA) terdapat di sumbu horizontal, sedangkan sumbu vertikal menunjukkan berbagai pilihan karir yang diinginkan, seperti Machine Learning Researcher, Data Scientist, Software Engineer, dan lainnya. Setiap titik ungu pada grafik merepresentasikan mahasiswa dengan IPK tertentu yang bercita-cita untuk mencapai salah satu karir tersebut.
        
       **Cara Membaca:**
        - Periksa IPK/GPA pada sumbu Horizontal
        - Perthatikan dot kecil pada baris sesuai dengan sumbu vertikal dari IPK/GPA
        - Jika semakin banyak dot kecil pada suatu tempat artinya banyak mahasiswa yang bercita-cita menekuni pada bidang tersebut
        """, unsafe_allow_html=True)

# Tab Prediksi Karir
with tab4:
    st.header("🔮 Prediksi Karir Mahasiswa")
    st.markdown(
        """
        ### Masukkan Data untuk Mendapatkan Prediksi Karir
        Model akan memprediksi karir berdasarkan IPK, Bidang Minat, dan Project yang dikembangkan atau diikuti Mahasiswa menggunakan metode regresi linier.
        metode regresi linier digunakan untuk memprediksi karir mahasiswa berdasarkan tiga fitur utama: IPK (Indeks Prestasi Kumulatif), bidang minat mahasiswa, dan proyek yang telah dikembangkan atau ikuti oleh Mahasiswa.
        """
        """
        Setelah data dimasukkan, model regresi linier memetakan nilai input (seperti IPK, bidang minat, dan proyek) ke dalam kategori karir yang sesuai.
        Dalam penerapan pada kode ini, data dari kolom "GPA", "Interested Domain Encoded", dan "Projects Encoded" digunakan sebagai variabel independen (X), sementara kolom "Future Career" yang telah dikategorikan menjadi variabel dependen (y). Proses pelatihan dilakukan menggunakan metode train_test_split untuk membagi data menjadi set pelatihan dan set pengujian, dan model regresi linier dilatih menggunakan data pelatihan tersebut. Hasilnya adalah model yang dapat digunakan untuk memprediksi karir masa depan mahasiswa berdasarkan input yang diberikan, seperti IPK dan minat. Evaluasi model dilakukan dengan mengukur MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error), yang memberikan gambaran tentang seberapa akurat model dalam memprediksi hasil berdasarkan data uji.
        """
    )

    X = df[["GPA", "Interested Domain Encoded", "Projects Encoded"]]
    y = df["Future Career"].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        model = load_model()
        st.success("Model yang sudah dilatih berhasil dimuat.")
    except:
        model = LinearRegression()
        model.fit(X_train, y_train)
        save_model(model)  # Simpan model setelah dilatih
        st.success("Model baru telah dilatih dan disimpan.")

    # Penjelasan model
    st.markdown("### Penjelasan Model")
    st.write("""
    Model yang digunakan adalah Linear Regression, yang memprediksi karir mahasiswa berdasarkan:
    - IPK (GPA): Nilai indeks prestasi mahasiswa.
    - Bidang Minat: Pilihan bidang yang diminati mahasiswa.
    - Project: Projek yang pernah dikembangkan atau diikuti oleh Mahasiswa
    """)

    # Input Prediksi
    gpa = st.number_input(
        "IPK Mahasiswa", min_value=3.0, max_value=4.0, value=3.5)
    interested_domain = st.selectbox(
        "Pilih Bidang yang Diminati", options=label_encoder.classes_)
    projects = st.selectbox(
        "Projek yang dikembangkan", options=project_encoder.classes_)

    interested_domain_encoded = label_encoder.transform([interested_domain])[0]
    project_encoded = project_encoder.transform([projects])[0]

    if st.button("Prediksi Karir"):
        input_data = np.array([[gpa, interested_domain_encoded, project_encoded]])
        prediction = model.predict(input_data)
        predicted_career = df["Future Career"].astype("category").cat.categories[int(prediction[0])]
        st.success(f"Prediksi Karir Mahasiswa: {predicted_career}")

    # Evaluasi Model
    st.markdown("### Evaluasi Model")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    st.write(
        """
        - MAE (Mean Absolute Error): Rata-rata kesalahan absolut antara prediksi dan data aktual.
        - MSE (Mean Squared Error): Rata-rata kesalahan kuadrat,
        """)
    st.markdown(
        """
        Hasil evaluasi model menunjukkan dua metrik utama untuk mengukur kinerja prediksi: MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error). MAE adalah rata-rata dari perbedaan absolut antara nilai yang diprediksi oleh model dan nilai aktual dalam data uji. Dalam hal ini, nilai MAE sebesar 5.35 menunjukkan bahwa, secara rata-rata, prediksi karir model menyimpang sekitar 5.35 kategori karir dari nilai aktual yang sebenarnya. Metrik RMSE, yang memberikan bobot lebih pada kesalahan besar, adalah 6.10. Ini menunjukkan bahwa kesalahan prediksi yang lebih besar memiliki pengaruh lebih besar terhadap model, sehingga memperlihatkan potensi ketidaktepatan yang lebih besar dalam beberapa kasus tertentu. Semakin rendah nilai MAE dan RMSE, semakin akurat model dalam melakukan prediksi.
        """
    )

# Footer
st.markdown("---")
st.markdown("**Prediksi Karir Mahasiswa** - Dikembangkan oleh tim Kelompok 3")