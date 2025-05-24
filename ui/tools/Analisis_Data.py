"""
Halaman Analisis Data Teks GoRide (Manual & CSV)
"""
import streamlit as st
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
    load_sample_data, get_or_train_model, display_model_metrics, predict_sentiment, preprocess_text,
    get_word_frequencies, get_ngrams, create_wordcloud, get_table_download_link
)

def render_data_analysis():
    """
    Function to render the data analysis page
    """
    # Load data dan model (cache)
    data = load_sample_data()
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
    pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)

    st.title("📑 Analisis Teks")

    # ======================
    # ANALISIS TEKS MANUAL & CSV
    # ======================

    # Inisialisasi session state
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "✏️ Input Teks Manual"
    if 'manual_text_input' not in st.session_state:
        st.session_state.manual_text_input = ""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
        
    def reset_analysis_state():
        st.session_state.analysis_complete = False
        # Reset hasil analisis lainnya
        if 'sentiment_result' in st.session_state:
            del st.session_state.sentiment_result
        if 'preprocessed_text' in st.session_state:
            del st.session_state.preprocessed_text
            
    def input_method_change():
        st.session_state.analysis_complete = False
        # Reset hasil analisis sebelumnya saat mengubah metode input
        if 'sentiment_result' in st.session_state:
            del st.session_state.sentiment_result
        if 'preprocessed_text' in st.session_state:
            del st.session_state.preprocessed_text
        if 'csv_results' in st.session_state:
            del st.session_state.csv_results

    # Pilihan metode input
    input_method = st.radio(
        "Pilih metode input:",
        ["✏️ Input Teks Manual", "📤 Unggah File CSV"],
        key="input_method_radio",
        on_change=input_method_change
    )
    st.session_state.input_method = input_method

    # ========== INPUT TEKS MANUAL ==========
    if input_method == "✏️ Input Teks Manual":
        st.write("### Masukkan Teks untuk Dianalisis:")
        text_input = st.text_area(
            "Ketik atau tempel teks di sini...",
            value=st.session_state.manual_text_input,
            height=200,
            placeholder="Contoh: Saya sangat puas dengan layanan GoRide. Driver sangat ramah dan profesional. Aplikasi mudah digunakan dan tarif terjangkau."
        )
        st.session_state.manual_text_input = text_input
        st.write("### 🛠️ Opsi Preprocessing Teks")
        with st.expander("Pengaturan Preprocessing", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                lowercase = st.checkbox("Konversi ke huruf kecil", value=True)
                clean_text_opt = st.checkbox("Cleansing teks (emoji, URL, dll)", value=True)
                normalize_slang_opt = st.checkbox("Normalisasi kata gaul/slang", value=True)
                remove_repeated_opt = st.checkbox("Hapus karakter berulang", value=True)
                remove_punct = st.checkbox("Hapus tanda baca", value=True)
            with col2:
                remove_num = st.checkbox("Hapus angka", value=True)
                tokenize_opt = st.checkbox("Tokenisasi teks", value=True)
                remove_stopwords_opt = st.checkbox("Hapus stopwords", value=True)
                stemming_opt = st.checkbox("Stemming (Sastrawi)", value=True)
                rejoin_opt = st.checkbox("Gabungkan kembali token menjadi teks", value=True)
        preprocess_options = {
            'lowercase': lowercase,
            'clean_text': clean_text_opt,
            'normalize_slang': normalize_slang_opt,
            'remove_repeated': remove_repeated_opt,
            'remove_punctuation': remove_punct,
            'remove_numbers': remove_num,
            'tokenize': tokenize_opt,
            'remove_stopwords': remove_stopwords_opt,
            'stemming': stemming_opt,            'rejoin': rejoin_opt
        }
        
        if st.button("🔍 Analisis Teks", type="primary"):
            if text_input:
                # Reset hasil analisis sebelumnya
                if 'sentiment_result' in st.session_state:
                    del st.session_state.sentiment_result
                if 'preprocessed_text' in st.session_state:
                    del st.session_state.preprocessed_text
                if 'teks_preprocessing' in st.session_state:
                    del st.session_state.teks_preprocessing
                st.session_state.analysis_complete = True
                st.session_state.text_to_analyze = text_input
                st.session_state.preprocess_options = preprocess_options
                st.session_state.input_source = "manual"
                # Simpan hasil preprocessing ke teks_preprocessing
                st.session_state.teks_preprocessing = preprocess_text(text_input, preprocess_options)
            else:
                st.error("⚠️ Mohon masukkan teks untuk dianalisis terlebih dahulu.")

    # ========== INPUT CSV ==========
    else:
        if st.session_state.get('input_source') == "manual":
            st.session_state.analysis_complete = False
            st.session_state.input_source = None
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
            st.session_state.input_source = "csv"
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
                    # Preprocessing batch ke kolom teks_preprocessing
                    df['teks_preprocessing'] = df['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocess_options))
                    predicted_results = []
                    for text in df['teks_preprocessing']:
                        result = predict_sentiment(text, pipeline, preprocess_options)
                        predicted_results.append(result)
                    df['predicted_sentiment'] = [result['sentiment'] for result in predicted_results]
                    confidence_scores = []
                    for result in predicted_results:
                        if result['sentiment'] == "POSITIF":
                            confidence_scores.append(result['probabilities']['POSITIF'] * 100)
                        else:
                            confidence_scores.append(result['probabilities']['NEGATIF'] * 100)
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
    if st.session_state.get('analysis_complete', False):
        if st.session_state.get('input_source') == "manual" and input_method == "✏️ Input Teks Manual" and st.session_state.get('text_to_analyze'):
            text_input = st.session_state.text_to_analyze
            preprocess_options = st.session_state.preprocess_options
            with st.spinner("Menganalisis teks..."):
                st.write("### Live Preview Preprocessing")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Teks Asli:**")
                    st.text_area("", value=text_input, height=150, disabled=True, key="original_text_area")
                with col2:
                    st.write("**Teks Setelah Preprocessing:**")
                    preprocessed_text = preprocess_text(text_input, preprocess_options)
                    st.text_area("", value=preprocessed_text, height=150, disabled=True, key="preprocessed_text_area")
                
                # Selalu perbarui teks yang telah dipreprocessing
                st.session_state.preprocessed_text = preprocessed_text
                
                st.write("### Analisis Sentimen")
                if 'sentiment_result' not in st.session_state:
                    st.session_state.sentiment_result = predict_sentiment(text_input, pipeline, preprocess_options)
                result = st.session_state.sentiment_result
                prediction = result['sentiment']
                pos_prob = result['probabilities']['POSITIF']
                neg_prob = result['probabilities']['NEGATIF']
                if prediction == "POSITIF":
                    confidence = pos_prob * 100
                    emoji = "😊"
                    color = "green"
                else:
                    confidence = neg_prob * 100
                    emoji = "😔"
                    color = "red"
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == "POSITIF":
                        st.success(f"Sentimen Terdeteksi: {prediction} {emoji}")
                    else:
                        st.error(f"Sentimen Terdeteksi: {prediction} {emoji}")
                    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Tingkat Kepercayaan"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 33], 'color': 'lightgray'},
                                {'range': [33, 66], 'color': 'gray'},
                                {'range': [66, 100], 'color': 'darkgray'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                # ========== TAB ANALISIS LANJUTAN ==========
                tabs = st.tabs(["📊 Frekuensi Kata", "🔄 Analisis N-Gram", "☁️ Word Cloud", "📝 Ringkasan Teks"])
                with tabs[0]:
                    st.subheader("Frekuensi Kata")
                    top_n = st.slider("Pilih jumlah kata teratas untuk ditampilkan:", 5, 30, 10)
                    word_freq = get_word_frequencies(preprocessed_text, top_n=top_n)
                    if word_freq:
                        word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
                        word_freq_df = word_freq_df.sort_values('Frequency', ascending=True)
                        fig = px.bar(word_freq_df, x='Frequency', y='Word', orientation='h', title="Frekuensi Kata dalam Teks", color='Frequency', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("**Tabel Data Frekuensi Kata:**")
                        word_freq_df = word_freq_df.sort_values('Frequency', ascending=False)
                        st.dataframe(word_freq_df)
                    else:
                        st.info("Tidak cukup kata unik untuk analisis frekuensi setelah preprocessing.")
                with tabs[1]:
                    st.subheader("Analisis N-Gram")
                    n_gram_type = st.radio("Pilih tipe N-gram:", ["Bigram (2 kata)", "Trigram (3 kata)"])
                    top_n_ngrams = st.slider("Pilih jumlah N-gram teratas untuk ditampilkan:", 3, 20, 10)
                    if n_gram_type == "Bigram (2 kata)":
                        n_gram_data = get_ngrams(preprocessed_text, 2, top_n=top_n_ngrams)
                    else:
                        n_gram_data = get_ngrams(preprocessed_text, 3, top_n=top_n_ngrams)
                    if n_gram_data:
                        n_gram_df = pd.DataFrame(list(n_gram_data.items()), columns=['N-gram', 'Frequency'])
                        n_gram_df = n_gram_df.sort_values('Frequency', ascending=True)
                        fig = px.bar(n_gram_df, x='Frequency', y='N-gram', orientation='h', title=f"Frekuensi {n_gram_type}", color='Frequency', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(f"**Tabel Data {n_gram_type}:**")
                        n_gram_df = n_gram_df.sort_values('Frequency', ascending=False)
                        st.dataframe(n_gram_df)
                    else:
                        st.info(f"Tidak cukup {n_gram_type.lower()} untuk dianalisis.")
                with tabs[2]:
                    st.subheader("Word Cloud")
                    max_words = st.slider("Jumlah maksimum kata:", 50, 200, 100)
                    colormap = st.selectbox("Pilih skema warna:", ["viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "YlOrRd"])
                    if preprocessed_text.strip():
                        wordcloud = create_wordcloud(preprocessed_text, max_words=max_words, background_color='white')
                        if wordcloud is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                            img_data = io.BytesIO()
                            wordcloud.to_image().save(img_data, format='PNG')
                            img_data.seek(0)
                            st.download_button(label="📥 Download Word Cloud sebagai PNG", data=img_data, file_name="wordcloud.png", mime="image/png")
                        else:
                            st.info("Tidak cukup kata untuk membuat word cloud setelah preprocessing.")
                with tabs[3]:
                    st.subheader("Ringkasan Teks")
                    word_count = len(nltk.word_tokenize(preprocessed_text))
                    char_count = len(preprocessed_text)
                    sent_count = len(nltk.sent_tokenize(preprocessed_text))
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Jumlah Kata", value=word_count)
                    with col2:
                        st.metric(label="Jumlah Karakter", value=char_count)
                    with col3:
                        st.metric(label="Jumlah Kalimat", value=sent_count)
                    avg_word_len = sum(len(word) for word in nltk.word_tokenize(preprocessed_text)) / word_count if word_count > 0 else 0
                    sentences = nltk.sent_tokenize(preprocessed_text)
                    avg_sent_len = sum(len(nltk.word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
                    unique_words = len(set(nltk.word_tokenize(preprocessed_text)))
                    lexical_diversity = unique_words / word_count if word_count > 0 else 0
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Rata-rata Panjang Kata", value=f"{avg_word_len:.2f} karakter")
                    with col2:
                        st.metric(label="Rata-rata Panjang Kalimat", value=f"{avg_sent_len:.2f} kata")
                    with col3:
                        st.metric(label="Keragaman Leksikal", value=f"{lexical_diversity:.2f}", help="Rasio kata unik terhadap total kata (0-1). Nilai lebih tinggi menunjukkan keragaman kata yang lebih besar.")
                    if sent_count > 2:
                        st.subheader("Ringkasan Ekstraktif Otomatis")
                        word_freq = nltk.FreqDist(nltk.word_tokenize(preprocessed_text))
                        sent_scores = {}
                        for i, sent in enumerate(sentences):
                            sent_words = nltk.word_tokenize(sent.lower())
                            sent_scores[i] = sum(word_freq.get(word, 0) for word in sent_words) / len(sent_words) if sent_words else 0
                        summary_length = st.slider("Persentase teks untuk ringkasan:", 10, 90, 30)
                        num_sent_for_summary = max(1, int(len(sentences) * summary_length / 100))
                        top_sent_indices = sorted(sorted(sent_scores.items(), key=lambda x: -x[1])[:num_sent_for_summary], key=lambda x: x[0])
                        summary = ' '.join(sentences[idx] for idx, _ in top_sent_indices)
                        st.write("**Ringkasan Teks:**")
                        st.info(summary)
                        compression = (1 - (len(summary) / len(preprocessed_text))) * 100
                        st.caption(f"Ringkasan menghasilkan kompresi {compression:.2f}% dari teks asli.")
                    else:
                        st.info("Teks terlalu pendek untuk membuat ringkasan terkait hasil sentiment.")
        elif st.session_state.get('input_source') == "csv" and input_method == "📤 Unggah File CSV" and st.session_state.get('csv_results') is not None:
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
            tabs = st.tabs(["📊 Frekuensi Kata", "🔄 Analisis N-Gram", "☁️ Word Cloud", "📝 Ringkasan Teks"])
            with tabs[0]:
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
            with tabs[1]:
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
            with tabs[2]:
                st.subheader("Word Cloud (CSV)")
                max_words = st.slider("Jumlah maksimum kata:", 50, 200, 100, key="csv_wc_max_words")
                colormap = st.selectbox("Pilih skema warna:", ["viridis", "plasma", "inferno", "magma", "cividis", "YlGnBu", "YlOrRd"], key="csv_wc_colormap")
                if preprocessed_all_text.strip():
                    wordcloud = create_wordcloud(preprocessed_all_text, max_words=max_words, background_color='white')
                    if wordcloud is not None:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        img_data = io.BytesIO()
                        wordcloud.to_image().save(img_data, format='PNG')
                        img_data.seek(0)
                        st.download_button(label="📥 Download Word Cloud sebagai PNG", data=img_data, file_name="wordcloud_csv.png", mime="image/png")
                    else:
                        st.info("Tidak cukup kata untuk membuat word cloud setelah preprocessing.")
            with tabs[3]:
                st.subheader("Ringkasan Teks (CSV)")
                word_count = len(nltk.word_tokenize(preprocessed_all_text))
                char_count = len(preprocessed_all_text)
                sent_count = len(nltk.sent_tokenize(preprocessed_all_text))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Jumlah Kata", value=word_count)
                with col2:
                    st.metric(label="Jumlah Karakter", value=char_count)
                with col3:
                    st.metric(label="Jumlah Kalimat", value=sent_count)
                avg_word_len = sum(len(word) for word in nltk.word_tokenize(preprocessed_all_text)) / word_count if word_count > 0 else 0
                sentences = nltk.sent_tokenize(preprocessed_all_text)
                avg_sent_len = sum(len(nltk.word_tokenize(sent)) for sent in sentences) / len(sentences) if sentences else 0
                unique_words = len(set(nltk.word_tokenize(preprocessed_all_text)))
                lexical_diversity = unique_words / word_count if word_count > 0 else 0
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
                        sent_words = nltk.word_tokenize(sent.lower())
                        sent_scores[i] = sum(word_freq.get(word, 0) for word in sent_words) / len(sent_words) if sent_words else 0
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
    st.markdown("---")
    st.caption("© 2025 GoRide Sentiment Analysis App • Develop By Mhd Adreansyah")
    st.caption("Aplikasi ini merupakan Tugas Akhir/Skripsi dibawah perlindungan Hak Cipta")

if __name__ == "__main__":
    render_data_analysis()
