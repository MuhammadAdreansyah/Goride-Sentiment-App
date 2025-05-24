# 📊 GoRide Sentiment Analysis

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![NLP](https://img.shields.io/badge/NLP-00C4CC?style=for-the-badge&logo=natural-language-processing&logoColor=white)
![ML](https://img.shields.io/badge/Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## ✨ Deskripsi 

GoRide Sentiment Analysis adalah aplikasi web berbasis Streamlit yang dirancang untuk menganalisis sentimen (positif/negatif) dari ulasan pengguna layanan GoRide dari Google Play Store. Aplikasi ini menggunakan teknologi Natural Language Processing (NLP) dan Machine Learning untuk mengklasifikasikan sentimen ulasan, serta menyajikan visualisasi dan analisis data yang komprehensif.

![App Screenshot](https://via.placeholder.com/800x400?text=GoRide+Sentiment+Analysis+App)

## 🚀 Fitur Utama

### 🔐 Sistem Autentikasi Lengkap
- Login dengan email/password
- Autentikasi OAuth melalui Google
- Registrasi pengguna baru
- Fitur reset password
- Manajemen sesi pengguna

### 📈 Dashboard Ringkasan
- Visualisasi distribusi sentimen ulasan
- Filter berdasarkan rentang waktu
- Metrik kinerja model (akurasi, presisi, recall, F1-score)
- Ringkasan statistik ulasan

### 📊 Analisis Data Teks
- Analisis ulasan dari input manual atau upload CSV
- Visualisasi word cloud untuk kata-kata yang sering muncul
- Analisis n-gram (unigram, bigram, trigram)
- Analisis frekuensi kata

### 🔍 Prediksi Sentimen Real-time
- Prediksi sentimen dari teks yang dimasukkan pengguna
- Penjelasan hasil prediksi dengan probabilitas sentimen
- Visualisasi hasil prediksi

## 🛠️ Teknologi yang Digunakan

### 👨‍💻 Bahasa Pemrograman
- Python 3.10+

### 📚 Libraries & Frameworks
- **Streamlit**: Antarmuka pengguna web
- **Firebase & Pyrebase**: Autentikasi dan database
- **Pandas & NumPy**: Manipulasi dan analisis data
- **Scikit-learn**: Pemodelan machine learning
- **NLTK & Sastrawi**: Pemrosesan bahasa alami untuk Bahasa Indonesia
- **Plotly & Matplotlib**: Visualisasi data
- **WordCloud**: Visualisasi frekuensi kata

## 🧠 Teknik NLP & Machine Learning

### 💬 Pemrosesan NLP
- Text preprocessing: lowercase, pembersihan teks, normalisasi slang
- Tokenisasi dan stemming (menggunakan Sastrawi untuk Bahasa Indonesia)
- Penghapusan stopwords dengan kamus khusus Bahasa Indonesia
- Vektorisasi TF-IDF untuk konversi teks ke fitur numerik
- Analisis n-gram untuk mengekstrak konteks frasa

### 🤖 Model Machine Learning
- Algoritma: Support Vector Machine (SVM)
- Teknik vektorisasi: Term Frequency-Inverse Document Frequency (TF-IDF)
- Metrik evaluasi: Akurasi, presisi, recall, F1-score
- Model disimpan dan di-cache untuk penggunaan efisien

## 📋 Prosedur Instalasi dan Penggunaan

### 📥 Persyaratan Sistem
- Python 3.10+
- Browser web modern (Chrome, Firefox, Edge)
- Koneksi internet (untuk autentikasi Firebase)

### ⚙️ Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/username/goride-sentiment-app.git
   cd goride-sentiment-app
   ```

2. **Setup Virtual Environment (Opsional tapi Direkomendasikan)**
   ```bash
   python -m venv venv
   # Untuk Windows
   venv\Scripts\activate
   # Untuk Linux/MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Firebase**
   - Buat project di [Firebase Console](https://console.firebase.google.com/)
   - Aktifkan Authentication dengan metode Email/Password dan Google
   - Download credentials dan simpan di direktori `config/`
   - Update `firebase-credentials.json`

5. **Setup Google OAuth** (opsional, untuk login dengan Google)
   - Buat OAuth credentials di [Google Cloud Console](https://console.cloud.google.com/)
   - Tambahkan redirect URI yang sesuai
   - Download client secret JSON dan simpan di direktori `config/`

6. **Setup Environment Variables**
   - Buat file `.streamlit/secrets.toml` dengan konten:
   ```toml
   FIREBASE_API_KEY = "your-firebase-api-key"
   GOOGLE_CLIENT_ID = "your-google-client-id"
   GOOGLE_CLIENT_SECRET = "your-google-client-secret"
   REDIRECT_URI = "your-redirect-uri" # biasanya http://localhost:8501/
   ```

### 🚀 Menjalankan Aplikasi

1. **Jalankan Server Streamlit**
   ```bash
   streamlit run main.py
   ```

2. **Akses Aplikasi**
   - Buka browser dan kunjungi: `http://localhost:8501`

3. **Registrasi/Login**
   - Buat akun baru atau login dengan akun yang sudah ada
   - Opsional: Gunakan login Google untuk masuk lebih cepat

### 📲 Menggunakan Aplikasi

#### 1️⃣ Halaman Login/Register
- Masukkan email dan password untuk login
- Klik "Buat Akun" untuk registrasi
- Gunakan tombol "Login with Google" untuk autentikasi cepat
- Fitur "Lupa Password" tersedia untuk reset password

#### 2️⃣ Dashboard Ringkasan
- Lihat ringkasan distribusi sentimen ulasan
- Gunakan filter tanggal untuk melihat data di rentang waktu tertentu
- Perhatikan metrik model yang menunjukkan performa klasifikasi
- Download data dalam format CSV jika diperlukan

#### 3️⃣ Analisis Data
- Pilih metode input: teks manual atau upload CSV
- Untuk input manual, ketik ulasan dan klik "Analisis"
- Untuk upload CSV, pastikan format sesuai (kolom wajib: "content", "date")
- Lihat hasil analisis dalam bentuk visualisasi interaktif
- Eksplorasi word cloud, n-gram, dan frekuensi kata

#### 4️⃣ Prediksi Sentimen
- Ketik ulasan di text area
- Klik tombol "Prediksi Teks"
- Lihat hasil prediksi sentimen beserta probabilitasnya
- Perhatikan visualisasi yang menjelaskan hasil prediksi

## 🔍 Hasil Uji Blackbox

### 1. Modul Autentikasi

| No | Skenario Uji | Kasus Uji | Hasil yang Diharapkan | Hasil Pengujian | Status |
|----|--------------|-----------|----------------------|-----------------|--------|
| 1.1 | Login dengan Email | Input email dan password yang valid | Berhasil login dan diarahkan ke dashboard | Berhasil login dan diarahkan ke dashboard dengan toast sukses | ✅ Berhasil |
| 1.2 | Login dengan Email | Input email dan password yang tidak valid | Menampilkan pesan error | Menampilkan error "Email/password salah" | ✅ Berhasil |
| 1.3 | Login dengan Google | Klik tombol "Login with Google" | Redirect ke halaman izin Google dan login sukses | Berhasil autentikasi Google dan redirect ke dashboard | ✅ Berhasil |
| 1.4 | Registrasi Pengguna | Input email dan password valid | Berhasil registrasi dan menampilkan notifikasi sukses | Akun berhasil dibuat dan notifikasi muncul | ✅ Berhasil |
| 1.5 | Registrasi Pengguna | Input email yang sudah terdaftar | Menampilkan pesan error bahwa email sudah terdaftar | Menampilkan pesan "Email already exists" | ✅ Berhasil |
| 1.6 | Reset Password | Input email valid yang terdaftar | Mengirim email reset password | Email reset berhasil dikirim ke alamat yang terdaftar | ✅ Berhasil |
| 1.7 | Reset Password | Input email tidak valid/tidak terdaftar | Menampilkan pesan error | Menampilkan pesan "Email not found" | ✅ Berhasil |
| 1.8 | Logout | Klik tombol logout | Mengakhiri sesi dan redirect ke halaman login | Berhasil logout dan diarahkan ke halaman login | ✅ Berhasil |

### 2. Modul Dashboard Ringkasan

| No | Skenario Uji | Kasus Uji | Hasil yang Diharapkan | Hasil Pengujian | Status |
|----|--------------|-----------|----------------------|-----------------|--------|
| 2.1 | Load Dashboard | Mengakses halaman dashboard setelah login | Menampilkan ringkasan sentimen dan grafik | Dashboard berhasil dimuat dengan grafik sentimen | ✅ Berhasil |
| 2.2 | Filter Data | Mengubah rentang tanggal di filter | Data dan grafik diperbarui sesuai filter | Data dan visualisasi berhasil diperbarui | ✅ Berhasil |
| 2.3 | Metrik Model | Melihat metrik performa model | Menampilkan akurasi, presisi, recall, F1-score | Metrik model ditampilkan dengan benar | ✅ Berhasil |
| 2.4 | Distribusi Rating | Melihat grafik distribusi rating | Menampilkan grafik pie/bar distribusi rating | Grafik distribusi rating berhasil ditampilkan | ✅ Berhasil |
| 2.5 | Download Data | Klik tombol download data | Berhasil mengunduh data dalam format CSV | File CSV berhasil diunduh | ✅ Berhasil |

### 3. Modul Analisis Data

| No | Skenario Uji | Kasus Uji | Hasil yang Diharapkan | Hasil Pengujian | Status |
|----|--------------|-----------|----------------------|-----------------|--------|
| 3.1 | Input Teks Manual | Memasukkan teks ulasan dan klik analisis | Menampilkan hasil analisis sentimen dan visualisasi | Berhasil menampilkan hasil analisis dengan visualisasi | ✅ Berhasil |
| 3.2 | Upload CSV | Mengunggah file CSV ulasan valid | Berhasil memproses CSV dan menampilkan analisis | CSV berhasil diproses dan analisis ditampilkan | ✅ Berhasil |
| 3.3 | Upload CSV Invalid | Mengunggah file non-CSV atau format kolom salah | Menampilkan pesan error format | Pesan error "Format file tidak sesuai" ditampilkan | ✅ Berhasil |
| 3.4 | Word Cloud | Melihat visualisasi word cloud | Menampilkan word cloud dari kata-kata frekuensi tertinggi | Word cloud berhasil dibuat dan ditampilkan | ✅ Berhasil |
| 3.5 | Analisis N-gram | Melihat visualisasi n-gram | Menampilkan grafik/tabel frekuensi n-gram | N-gram berhasil divisualisasikan | ✅ Berhasil |
| 3.6 | Grafik Frekuensi Kata | Melihat grafik frekuensi kata | Menampilkan grafik bar frekuensi kata | Grafik frekuensi kata ditampilkan dengan benar | ✅ Berhasil |

### 4. Modul Prediksi Sentimen

| No | Skenario Uji | Kasus Uji | Hasil yang Diharapkan | Hasil Pengujian | Status |
|----|--------------|-----------|----------------------|-----------------|--------|
| 4.1 | Prediksi Teks Positif | Input teks dengan sentimen positif jelas | Memprediksi sentimen POSITIF dengan probabilitas tinggi | Berhasil memprediksi POSITIF dengan probabilitas >0.75 | ✅ Berhasil |
| 4.2 | Prediksi Teks Negatif | Input teks dengan sentimen negatif jelas | Memprediksi sentimen NEGATIF dengan probabilitas tinggi | Berhasil memprediksi NEGATIF dengan probabilitas >0.75 | ✅ Berhasil |
| 4.3 | Prediksi Teks Ambigu | Input teks dengan sentimen campuran/tidak jelas | Memprediksi sentimen dengan probabilitas mendekati 0.5 | Model memberikan prediksi dengan probabilitas 0.5-0.65 | ✅ Berhasil |
| 4.4 | Prediksi Teks Kosong | Mengklik prediksi tanpa memasukkan teks | Menampilkan peringatan input kosong | Pesan "Masukkan teks untuk prediksi" ditampilkan | ✅ Berhasil |
| 4.5 | Prediksi dengan Slang | Input teks dengan kata-kata slang Indonesia | Berhasil menormalisasi slang dan memprediksi sentimen | Slang berhasil dinormalisasi dan hasil prediksi akurat | ✅ Berhasil |

### 5. Pengujian Non-Fungsional

| No | Skenario Uji | Kasus Uji | Hasil yang Diharapkan | Hasil Pengujian | Status |
|----|--------------|-----------|----------------------|-----------------|--------|
| 5.1 | Responsivitas UI | Mengakses aplikasi dari berbagai perangkat (desktop, tablet, mobile) | UI menyesuaikan dengan ukuran layar | UI responsif di semua ukuran perangkat | ✅ Berhasil |
| 5.2 | Performa | Load aplikasi dan melakukan analisis data besar | Aplikasi tetap responsif dan memproses data dalam waktu wajar | Aplikasi berhasil memproses data dengan kecepatan memadai | ✅ Berhasil |
| 5.3 | Keamanan | Mencoba akses halaman dashboard tanpa login | Redirect ke halaman login | Redirect ke login berhasil dilakukan | ✅ Berhasil |
| 5.4 | Ketahanan | Menginput teks yang sangat panjang (>5000 karakter) | Aplikasi tetap berfungsi tanpa crash | Aplikasi berhasil memproses teks panjang tanpa error | ✅ Berhasil |
| 5.5 | Kompatibilitas Browser | Menguji pada browser berbeda (Chrome, Firefox, Safari) | Aplikasi berfungsi dengan baik di semua browser | Aplikasi berjalan baik di semua browser utama | ✅ Berhasil |

## 📞 Kontak & Kontribusi

Jika Anda memiliki pertanyaan atau ingin berkontribusi, jangan ragu untuk:

- 📧 Email: your.email@example.com
- 🐙 GitHub: [username](https://github.com/username)
- 🌐 Website: [yourwebsite.com](https://yourwebsite.com)

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE)