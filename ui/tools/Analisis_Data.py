"""
Halaman Analisis Data Teks GoRide (CSV Saja)
"""
import streamlit as st
from ui.auth import auth
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import base64
import nltk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_sample_data, get_or_train_model, display_model_metrics, predict_sentiment,
    preprocess_text, get_word_frequencies, get_ngrams, create_wordcloud
)

def render_data_analysis():
    # Sinkronisasi status login dari cookie ke session_state (penting untuk refresh)
    auth.sync_login_state()

    """
    Function to render the data analysis page
    """
    # Load data dan model (cache)
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
    
    # Model sudah disiapkan sebelumnya, langsung load
    pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)

    st.title("📑 Analisis Teks")

    # ======================
    # ANALISIS TEKS CSV SAJA
    # ======================

    # Inisialisasi session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    def reset_analysis_state():
        st.session_state.analysis_complete = False
        if 'csv_results' in st.session_state:
            del st.session_state.csv_results
        if 'csv_preprocessed' in st.session_state:
            del st.session_state.csv_preprocessed

    # Hanya tampilkan upload CSV
    st.write("### Unggah file CSV dengan ulasan:")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="csv_uploader")
    st.write("### 🛠️ Opsi Preprocessing Teks")
    with st.expander("Pengaturan Preprocessing", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            lowercase = st.checkbox("Konversi ke huruf kecil", value=True, key="csv_lowercase")
            clean_text_opt = st.checkbox("Cleansing teks (emoji, URL, dll)", value=True, key="csv_clean")
            normalize_slang_opt = st.checkbox("Normalisasi kata gaul/slang", value=True, key="csv_normalize")
            remove_repeated_opt = st.checkbox("Hapus karakter berulang", value=True, key="csv_repeated")
            remove_punct = st.checkbox("Hapus tanda baca", value=True, key="csv_punct")
        with col2:
            remove_num = st.checkbox("Hapus angka", value=True, key="csv_num")
            tokenize_opt = st.checkbox("Tokenisasi teks", value=True, key="csv_tokenize")
            remove_stopwords_opt = st.checkbox("Hapus stopwords", value=True, key="csv_stopwords")
            stemming_opt = st.checkbox("Stemming (Sastrawi)", value=True, key="csv_stemming")
            rejoin_opt = st.checkbox("Gabungkan kembali token menjadi teks", value=True, key="csv_rejoin")
    preprocess_options = {
        'lowercase': lowercase,
        'clean_text': clean_text_opt,
        'normalize_slang': normalize_slang_opt,
        'remove_repeated': remove_repeated_opt,
        'remove_punctuation': remove_punct,
        'remove_numbers': remove_num,
        'tokenize': tokenize_opt,
        'remove_stopwords': remove_stopwords_opt,
        'stemming': stemming_opt,
        'rejoin': rejoin_opt
    }
    predict_csv_button = st.button("🔍 Analisis Teks", type="primary", disabled=uploaded_file is None)
    if uploaded_file is not None and predict_csv_button:
        st.session_state.analysis_complete = True
        st.session_state.preprocess_options = preprocess_options
        try:
            my_bar = st.progress(0, text="Memproses file CSV...")
            df = pd.read_csv(uploaded_file)
            my_bar.progress(25, text="File berhasil diunggah...")
            if 'review_text' not in df.columns:
                review_col_name = st.selectbox("Pilih kolom yang berisi teks ulasan:", df.columns)
                if review_col_name:
                    df['review_text'] = df[review_col_name]
            if 'review_text' in df.columns:
                my_bar.progress(50, text="Memprediksi sentimen...")
                df['teks_preprocessing'] = df['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocess_options))
                predicted_results = [predict_sentiment(text, pipeline, preprocess_options) for text in df['teks_preprocessing']]
                df['predicted_sentiment'] = [result['sentiment'] for result in predicted_results]
                confidence_scores = [max(result['probabilities'].values()) for result in predicted_results]
                df['confidence'] = confidence_scores
                st.session_state.csv_results = df
                st.session_state.csv_preprocessed = True
                my_bar.progress(100, text="Analisis selesai!")
                import time
                time.sleep(0.5)
                my_bar.empty()
                st.success("✅ Analisis sentimen selesai!")
            else:
                st.error("❌ Kolom teks ulasan tidak ditemukan dalam file CSV.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {str(e)}")
            st.session_state.analysis_complete = False
    elif predict_csv_button and uploaded_file is None:
        st.error("⚠️ Silakan unggah file CSV terlebih dahulu untuk dianalisis.")

    # ========== HASIL ANALISIS ==========
    if st.session_state.get('analysis_complete', False) and st.session_state.get('csv_results') is not None:
        df = st.session_state.csv_results
        st.write("### Analisis Sentimen")
        col1, col2 = st.columns(2)
        with col1:
            pos_count = len(df[df['predicted_sentiment'] == 'POSITIF'])
            pos_percentage = pos_count / len(df) * 100 if len(df) > 0 else 0
            st.metric(label="Sentimen Positif 🟢", value=f"{pos_count} ulasan", delta=f"{pos_percentage:.2f}%")
        with col2:
            neg_count = len(df[df['predicted_sentiment'] == 'NEGATIF'])
            neg_percentage = neg_count / len(df) * 100 if len(df) > 0 else 0
            st.metric(label="Sentimen Negatif 🔴", value=f"{neg_count} ulasan", delta=f"{neg_percentage:.2f}%")
        st.write("### Visualisasi Hasil Analisis")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', color='Sentiment', color_discrete_map={'POSITIF': 'green', 'NEGATIF': 'red'}, title="Distribusi Sentimen pada Data yang Diunggah")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            positive_pct = pos_percentage
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=positive_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Persentase Sentimen Positif"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if positive_pct >= 50 else "red"},
                    'steps': [
                        {'range': [0, 33], 'color': 'lightgray'},
                        {'range': [33, 66], 'color': 'gray'},
                        {'range': [66, 100], 'color': 'darkgray'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': positive_pct
                    }
                },
                number={'suffix': "%", 'valueformat': ".1f"}
            ))
            st.plotly_chart(fig, use_container_width=True)
        all_text = " ".join(df['review_text'].astype(str).tolist())
        preprocess_options = st.session_state.preprocess_options
        preprocessed_all_text = preprocess_text(all_text, preprocess_options)
        # ========== TAB ANALISIS LANJUTAN UNTUK CSV ==========
        tabs = st.tabs([
            "📋 Tabel Hasil Prediksi",
            "📊 Frekuensi Kata",
            "🔄 Analisis N-Gram",
            "☁️ Word Cloud",
            "📝 Ringkasan Teks"
        ])
        # TAB 0: Tabel Hasil Prediksi
        with tabs[0]:
            st.subheader("Tabel Hasil Prediksi Sentimen")
            # Pilihan filter sentimen
            filter_sentimen = st.selectbox("Filter berdasarkan sentimen:", ["Semua", "POSITIF", "NEGATIF"], key="filter_sentimen")
            if filter_sentimen == "POSITIF":
                filtered_df = df[df['predicted_sentiment'] == 'POSITIF']
            elif filter_sentimen == "NEGATIF":
                filtered_df = df[df['predicted_sentiment'] == 'NEGATIF']
            else:
                filtered_df = df
            st.dataframe(filtered_df[[col for col in filtered_df.columns if col in ['review_text','teks_preprocessing','predicted_sentiment','confidence']]], use_container_width=True)
            # Tombol download CSV
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi_goride.csv">📥 Download Hasil Prediksi (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        # TAB 1: Frekuensi Kata
        with tabs[1]:
            st.subheader("Frekuensi Kata (CSV)")
            top_n = st.slider("Pilih jumlah kata teratas untuk ditampilkan:", 5, 30, 10, key="csv_word_freq")
            word_freq = get_word_frequencies(preprocessed_all_text, top_n=top_n)
            if word_freq:
                word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
                word_freq_df = word_freq_df.sort_values('Frequency', ascending=True)
                fig = px.bar(word_freq_df, x='Frequency', y='Word', orientation='h', title="Frekuensi Kata dalam Data CSV", color='Frequency', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Tabel Data Frekuensi Kata:**")
                word_freq_df = word_freq_df.sort_values('Frequency', ascending=False)
                st.dataframe(word_freq_df)
            else:
                st.info("Tidak cukup kata unik untuk analisis frekuensi setelah preprocessing.")
        # TAB 2: N-Gram
        with tabs[2]:
            st.subheader("Analisis N-Gram (CSV)")
            n_gram_type = st.radio("Pilih tipe N-gram:", ["Bigram (2 kata)", "Trigram (3 kata)"], key="csv_ngram_type")
            top_n_ngrams = st.slider("Pilih jumlah N-gram teratas untuk ditampilkan:", 3, 20, 10, key="csv_ngram_slider")
            if n_gram_type == "Bigram (2 kata)":
                n_gram_data = get_ngrams(preprocessed_all_text, 2, top_n=top_n_ngrams)
            else:
                n_gram_data = get_ngrams(preprocessed_all_text, 3, top_n=top_n_ngrams)
            if n_gram_data:
                n_gram_df = pd.DataFrame(list(n_gram_data.items()), columns=['N-gram', 'Frequency'])
                n_gram_df = n_gram_df.sort_values('Frequency', ascending=True)
                fig = px.bar(n_gram_df, x='Frequency', y='N-gram', orientation='h', title=f"Frekuensi {n_gram_type} dalam Data CSV", color='Frequency', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**Tabel Data {n_gram_type}:**")
                n_gram_df = n_gram_df.sort_values('Frequency', ascending=False)
                st.dataframe(n_gram_df)
            else:
                st.info(f"Tidak cukup {n_gram_type.lower()} untuk dianalisis.")
        # TAB 3: Word Cloud
        with tabs[3]:
            st.subheader("Word Cloud (CSV)")
            max_words = st.slider("Jumlah maksimum kata:", 50, 200, 100, key="csv_wc_max_words")
            colormap = st.selectbox("Pilih skema warna:", ["viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "YlOrRd"], key="csv_wc_colormap")
            if preprocessed_all_text.strip():
                wordcloud = create_wordcloud(preprocessed_all_text, max_words=max_words, background_color='white')
                if wordcloud is not None:
                    st.image(wordcloud.to_array(), use_column_width=True)
                else:
                    st.info("Word cloud tidak dapat dibuat.")
        # TAB 4: Ringkasan
        with tabs[4]:
            st.subheader("Ringkasan Teks (CSV)")
            # Hitung statistik dasar
            sentences = nltk.sent_tokenize(preprocessed_all_text)
            word_count = len(nltk.word_tokenize(preprocessed_all_text))
            char_count = len(preprocessed_all_text)
            sent_count = len(sentences)
            unique_words = len(set(nltk.word_tokenize(preprocessed_all_text)))
            avg_word_len = sum(len(word) for word in nltk.word_tokenize(preprocessed_all_text)) / word_count if word_count > 0 else 0
            avg_sent_len = sum(len(nltk.word_tokenize(sent)) for sent in sentences) / len(sentences) if len(sentences) > 0 else 0
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Jumlah Kata", value=word_count)
            with col2:
                st.metric(label="Jumlah Karakter", value=char_count)
            with col3:
                st.metric(label="Jumlah Kalimat", value=sent_count)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Rata-rata Panjang Kata", value=f"{avg_word_len:.2f} karakter")
            with col2:
                st.metric(label="Rata-rata Panjang Kalimat", value=f"{avg_sent_len:.2f} kata")
            with col3:
                st.metric(label="Keragaman Leksikal", value=f"{lexical_diversity:.2f}", help="Rasio kata unik terhadap total kata (0-1). Nilai lebih tinggi menunjukkan keragaman kata yang lebih besar.")
            if sent_count > 2:
                st.subheader("Ringkasan Ekstraktif Otomatis (CSV)")
                word_freq = nltk.FreqDist(nltk.word_tokenize(preprocessed_all_text))
                sent_scores = {}
                for i, sent in enumerate(sentences):
                    sent_scores[i] = sum(word_freq[word] for word in nltk.word_tokenize(sent) if word in word_freq)
                summary_length = st.slider("Persentase teks untuk ringkasan:", 10, 90, 30, key="csv_summary_length")
                num_sent_for_summary = max(1, int(len(sentences) * summary_length / 100))
                top_sent_indices = sorted(sorted(sent_scores.items(), key=lambda x: -x[1])[:num_sent_for_summary], key=lambda x: x[0])
                summary = ' '.join(sentences[idx] for idx, _ in top_sent_indices)
                st.write("**Ringkasan Teks:**")
                st.info(summary)
                compression = (1 - (len(summary) / len(preprocessed_all_text))) * 100
                st.caption(f"Ringkasan menghasilkan kompresi {compression:.2f}% dari teks asli.")
            else:
                st.info("Teks terlalu pendek untuk membuat ringkasan terkait hasil sentiment.")
    
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
    
if __name__ == "__main__":
    render_data_analysis()
