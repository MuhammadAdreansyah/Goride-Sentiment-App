"""
Halaman Prediksi Sentimen Teks GoRide
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import nltk
import sys
import os
import base64
import plotly.express as px
from ui.auth import auth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_sample_data, display_model_metrics, predict_sentiment, preprocess_text, get_or_train_model
)

def render_sentiment_prediction():
    # Sinkronisasi status login dari cookie ke session_state (penting untuk refresh)
    auth.sync_login_state()

    # Load data (untuk referensi, tidak untuk training ulang model)
    data = load_sample_data()
    if data.empty:
        st.error("❌ Data tidak tersedia untuk analisis!")
        st.stop()
        
    preprocessing_options = {
        'lowercase': True,
        'clean_text': True,
        'normalize_slang': True,
        'remove_repeated': True,
        'remove_punctuation': True,
        'remove_numbers': True,
        'tokenize': True,
        'remove_stopwords': True,
        'stemming': True,
        'rejoin': True
    }
    
    # Model sudah disiapkan sebelumnya, langsung load (tanpa SMOTE)
    pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options, use_tanpa_smote=True)

    st.title("🔍 Prediksi Sentimen Teks")
    st.subheader("Analisis Sentimen Ulasan GoRide secara Real-time")

    st.write("### Masukkan teks ulasan:")
    text_input = st.text_area(
        "Ketik ulasan di sini...",
        height=150,
        placeholder="Contoh: Saya puas dengan pelayanan GoRide. Driver ramah dan cepat sampai."
    )
    predict_button = st.button("🔍 Prediksi Teks", type="primary")

    st.write("""
    > **Tips Penggunaan:**
    > - Masukkan ulasan GoRide dalam Bahasa Indonesia.
    > - Model hanya mengenali sentimen POSITIF dan NEGATIF.
    > - Semakin panjang dan jelas ulasan, prediksi akan lebih akurat.
    > - Model tidak mengenali sarkasme, typo berat, atau bahasa campuran.
    """)

    if text_input and predict_button:
        with st.spinner('Menganalisis teks...'):
            result = predict_sentiment(text_input, pipeline, preprocessing_options)
            prediction = result['sentiment']
            if prediction == "POSITIF":
                confidence = result['probabilities']['POSITIF'] * 100
                emoji = "😊"
                gauge_color = "green"
            else:
                confidence = result['probabilities']['NEGATIF'] * 100
                emoji = "😔"
                gauge_color = "red"
            tabs = st.tabs(["Analisis Sentimen", "Ringkasan"])
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Hasil Prediksi:")
                    if prediction == "POSITIF":
                        st.success(f"Sentimen: {prediction} {emoji}")
                    else:
                        st.error(f"Sentimen: {prediction} {emoji}")
                    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        title = {'text': "Tingkat Kepercayaan"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': gauge_color},
                            'steps': [
                                {'range': [0, 33], 'color': "#f9e8e8"},
                                {'range': [33, 66], 'color': "#f0f0f0"},
                                {'range': [66, 100], 'color': "#e8f9e8"}
                            ]
                        },
                        number = {'suffix': "%", 'valueformat': ".1f"}
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                # Tambahkan tombol download hasil prediksi
                pred_df = pd.DataFrame({
                    'Teks Ulasan': [text_input],
                    'Sentimen': [prediction],
                    'Confidence (%)': [f"{confidence:.2f}"],
                })
                csv = pred_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi_goride.csv">📥 Download Hasil Prediksi (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
            with tabs[1]:
                st.subheader("Ringkasan Analisis")
                summary_data = {
                    "Aspek": ["Sentimen Terdeteksi", "Tingkat Kepercayaan", "Jumlah Kata", "Jumlah Karakter"],
                    "Nilai": [
                        f"{prediction} {emoji}",
                        f"{confidence:.2f}%",
                        len(nltk.word_tokenize(text_input)),
                        len(text_input)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)
                st.subheader("Kata Kunci yang Mempengaruhi Prediksi")
                clean_tokens = nltk.word_tokenize(
                    text_input.lower()
                )
                if clean_tokens:
                    token_df = pd.DataFrame({
                        'Token': clean_tokens,
                        'Present': [1] * len(clean_tokens)
                    })
                    if len(token_df) > 10:
                        token_df = token_df.head(10)
                    fig = px.bar(
                        token_df,
                        x='Present',
                        y='Token',
                        orientation='h',
                        title="Kata Kunci dalam Teks (Top 10)",
                        color='Present',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tidak cukup kata kunci untuk ditampilkan setelah preprocessing.")
    elif predict_button and not text_input:
        st.error("⚠️ Silakan masukkan teks terlebih dahulu untuk diprediksi.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
        <p style="margin: 0; font-size: 0.9rem; color: #666;">
            © 2025 GoRide Sentiment Analysis Dashboard • Developed by Mhd Adreansyah
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #888;">
            🎓 Aplikasi ini merupakan bagian dari Tugas Akhir/Skripsi di bawah perlindungan Hak Cipta
        </p>
    </div>
    """, unsafe_allow_html=True)
    
# Call the function to render the UI
if __name__ == "__main__":
    render_sentiment_prediction()
