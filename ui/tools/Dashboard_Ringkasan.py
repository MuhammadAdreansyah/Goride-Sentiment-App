import streamlit as st
from ui.auth import auth
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_sample_data, get_or_train_model, display_model_metrics, predict_sentiment,
    preprocess_text, get_word_frequencies, get_ngrams, create_wordcloud, get_table_download_link
)

def render_dashboard():
    # Sinkronisasi status login dari cookie ke session_state (penting untuk refresh)
    auth.sync_login_state()
    # Tampilkan toast jika login baru saja berhasil (untuk fallback jika main.py tidak sempat menampilkan)
    if st.session_state.get('login_success', False):
        st.toast(f"User {st.session_state.get('user_email', '')} login successfully!", icon="✅")
        st.session_state['login_success'] = False
    # Load data dan model (cache)
    data = load_sample_data()
    if 'data_loaded_toast_shown' not in st.session_state:
        if not data.empty:
            st.toast(f"Data berhasil dimuat: {len(data)} ulasan", icon="✅")
        else:
            st.toast("Data gagal dimuat atau kosong!", icon="⚠️")
        st.session_state['data_loaded_toast_shown'] = True
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

    st.title("📊 Dashboard Analisis Sentimen GoRide")
    st.subheader("Analisis Ulasan Pengguna dari Google Play Store")

    # Date filter
    st.subheader("Filter Berdasarkan Rentang Waktu")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Tanggal Mulai", value=pd.to_datetime(data['date']).min())
    with col2:
        end_date = st.date_input("Tanggal Selesai", value=pd.to_datetime(data['date']).max())
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    with st.spinner('Filtering data...'):
        filtered_data = data[(pd.to_datetime(data['date']) >= start_date) & (pd.to_datetime(data['date']) <= end_date)]
    if filtered_data.empty:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
        return
    @st.cache_data(ttl=300)
    def calculate_metrics(df):
        total = len(df)
        pos_count = len(df[df['sentiment'] == 'POSITIF'])
        neg_count = len(df[df['sentiment'] == 'NEGATIF'])
        pos_percentage = (pos_count / total * 100) if total > 0 else 0
        neg_percentage = (neg_count / total * 100) if total > 0 else 0
        pos_pct = pos_count/total*100 if total > 0 else 0
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        today_count = len(df[df['date'] == today])
        return {
            'total': total,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_percentage': pos_percentage,
            'neg_percentage': neg_percentage,
            'today_count': today_count,
            'pos_pct': pos_pct
        }
    metrics = calculate_metrics(filtered_data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Ulasan 📈", value=metrics['total'], delta=f"{metrics['today_count']} hari ini")
    with col2:
        st.metric(label="Sentimen Positif 🟢", value=f"{metrics['pos_percentage']:.2f}%", delta=f"{metrics['pos_percentage'] - 50:.2f}% dari rata-rata")
    with col3:
        st.metric(label="Sentimen Negatif 🔴", value=f"{metrics['neg_percentage']:.2f}%", delta=f"{metrics['neg_percentage'] - 50:.2f}% dari rata-rata", delta_color="inverse")
    with col4:
        st.metric(label="Indeks Kepuasan 👍", value=f"{metrics['pos_pct']:.1f}%", delta=f"{metrics['pos_pct'] - 50:.1f} dari rata-rata%", delta_color="inverse")
    # Pastikan kolom teks_preprocessing tersedia dan konsisten
    if 'teks_preprocessing' not in data.columns:
        st.info("Melakukan preprocessing batch untuk seluruh data...")
        data.loc[:, 'teks_preprocessing'] = data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    if 'teks_preprocessing' not in filtered_data.columns:
        filtered_data = filtered_data.copy()
        filtered_data.loc[:, 'teks_preprocessing'] = filtered_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    # Ambil topik dari top 20 kata paling sering muncul di hasil preprocessing
    all_words = " ".join(filtered_data['teks_preprocessing'])
    word_freq = get_word_frequencies(all_words, top_n=20)
    topics = ["All"] + list(word_freq.keys())
    selected_topic = st.selectbox("Filter berdasarkan topik:", topics)
    if selected_topic != "All":
        topic_data = filtered_data[filtered_data['teks_preprocessing'].str.contains(selected_topic, case=False)].copy()
        st.info(f"Ditemukan {len(topic_data)} data untuk topik '{selected_topic}'.")
    else:
        topic_data = filtered_data.copy()
    if 'teks_preprocessing' not in topic_data.columns:
        topic_data = topic_data.copy()
        topic_data.loc[:, 'teks_preprocessing'] = topic_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    if topic_data.empty:
        st.warning(f"Tidak ada data untuk topik '{selected_topic}'. Coba pilih topik lain atau periksa hasil preprocessing/stemming.")
        topic_data = filtered_data.copy()
    tab1, tab2, tab3 = st.tabs(["Distribusi Sentimen", "Tren Waktu", "Analisis Kata"])
    with tab1:
        st.subheader("📊 Distribusi Sentimen")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = topic_data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            bar_chart = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', color_discrete_map={'POSITIF': 'green', 'NEGATIF': 'red'}, title="Distribusi Sentimen Ulasan")
            st.plotly_chart(bar_chart, use_container_width=True)
        with col2:
            pie_chart = px.pie(sentiment_counts, values='Count', names='Sentiment', color='Sentiment', color_discrete_map={'POSITIF': 'green', 'NEGATIF': 'red'}, title="Persentase Sentimen Ulasan")
            pie_chart.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(pie_chart, use_container_width=True)
    with tab2:
        st.subheader("📈 Tren Sentimen dari Waktu ke Waktu")
        time_granularity = st.radio("Granularitas Waktu:", options=["Harian", "Mingguan", "Bulanan"], horizontal=True)
        visualization_data = topic_data
        if len(topic_data) > 10000:
            sample_size = min(10000, int(len(topic_data) * 0.3))
            st.info(f"Dataset terlalu besar untuk visualisasi ({len(topic_data):,} baris). Menggunakan sampel {sample_size:,} baris untuk memperlancar visualisasi.")
            visualization_data = topic_data.sample(sample_size, random_state=42)
            use_all_data = st.checkbox("Gunakan semua data (bisa memperlambat aplikasi)")
            if use_all_data:
                visualization_data = topic_data
                st.warning("Menggunakan semua data dapat memperlambat visualisasi. Harap bersabar.")
        if time_granularity == "Harian":
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m-%d')
            unique_days = visualization_data['time_group'].nunique()
            if unique_days > 100:
                st.info(f"Data harian terlalu banyak ({unique_days} hari). Menerapkan binning otomatis.")
                visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.to_period('W').astype(str)
        elif time_granularity == "Mingguan":
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%W')
        else:
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m')
        sentiment_trend = topic_data.groupby(['time_group', 'sentiment']).size().reset_index(name='count')
        try:
            sentiment_pivot = sentiment_trend.pivot(index='time_group', columns='sentiment', values='count').reset_index()
            sentiment_pivot.fillna(0, inplace=True)
            if 'POSITIF' not in sentiment_pivot.columns:
                sentiment_pivot['POSITIF'] = 0
            if 'NEGATIF' not in sentiment_pivot.columns:
                sentiment_pivot['NEGATIF'] = 0
            sentiment_pivot['total'] = sentiment_pivot['POSITIF'] + sentiment_pivot['NEGATIF']
            sentiment_pivot['positive_percentage'] = np.where(sentiment_pivot['total'] > 0, (sentiment_pivot['POSITIF'] / sentiment_pivot['total'] * 100).round(2), 0)
            line_chart = px.line(sentiment_pivot, x='time_group', y=['POSITIF', 'NEGATIF'], title=f"Tren Jumlah Ulasan Berdasarkan Sentimen ({time_granularity})", labels={'value': 'Jumlah Ulasan', 'time_group': 'Waktu', 'variable': 'Sentimen'}, color_discrete_map={'POSITIF': 'green', 'NEGATIF': 'red'})
            st.plotly_chart(line_chart, use_container_width=True)
            pct_line_chart = px.line(sentiment_pivot, x='time_group', y='positive_percentage', title=f"Tren Persentase Sentimen Positif ({time_granularity})", labels={'positive_percentage': '% Ulasan Positif', 'time_group': 'Waktu'})
            pct_line_chart.update_traces(line_color='green')
            pct_line_chart.add_shape(type="line", x0=0, y0=50, x1=1, y1=50, xref="paper", line=dict(color="gray", width=1, dash="dash"))
            st.plotly_chart(pct_line_chart, use_container_width=True)
            csv = sentiment_pivot.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_trend.csv">📥 Download Trend Data (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating trend charts: {e}")
            st.info("Try adjusting your date range or filters to include more data points.")
    with tab3:
        st.subheader("📝 Analisis Kata dalam Ulasan")
        col1, col2 = st.columns(2)
        @st.cache_data(ttl=3600)
        def safe_create_wordcloud(text, max_words=100, max_length=10000, timeout_seconds=15):
            import threading
            import signal
            import time
            class TimeoutException(Exception):
                pass
            def timeout_handler(signum, frame):
                raise TimeoutException("Wordcloud generation timed out")
            result = [None]
            if len(text) > max_length:
                st.info(f"Text size reduced from {len(text):,} to {max_length:,} characters for efficient wordcloud generation")
                words = text.split()
                sampled_words = random.sample(words, min(max_length, len(words)))
                text = " ".join(sampled_words)
            reduce_complexity = False
            current_memory = 0
            try:
                import psutil
                process = psutil.Process(os.getpid())
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > 1000:
                    reduce_complexity = True
            except Exception:
                pass
            if reduce_complexity or len(text) > 100000:
                max_words = min(50, max_words)
                st.info(f"Reducing wordcloud complexity due to high memory usage or large text size.")
            try:
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    start_time = time.time()
                    wordcloud = create_wordcloud(text, max_words=max_words)
                    generation_time = time.time() - start_time
                    signal.alarm(0)
                else:
                    import threading
                    result = [None]
                    error = [None]
                    def target_func():
                        try:
                            result[0] = create_wordcloud(text, max_words=max_words)
                        except Exception as e:
                            error[0] = str(e)
                    thread = threading.Thread(target=target_func)
                    start_time = time.time()
                    thread.start()
                    thread.join(timeout_seconds)
                    generation_time = time.time() - start_time
                    if thread.is_alive():
                        st.warning(f"Wordcloud generation timed out after {timeout_seconds} seconds. Trying with smaller sample...")
                        words = text.split()
                        smaller_sample = random.sample(words, min(1000, len(words)))
                        sampled_text = " ".join(smaller_sample)
                        return create_wordcloud(sampled_text, max_words=50)
                    wordcloud = result[0]
                    if error[0] is not None:
                        raise Exception(error[0])
                if generation_time > 3:
                    st.info(f"Wordcloud generated in {generation_time:.1f} seconds")
                return wordcloud
            except TimeoutException:
                st.warning(f"Wordcloud generation timed out after {timeout_seconds} seconds. Trying with smaller sample...")
                words = text.split()
                smaller_sample = random.sample(words, min(1000, len(words)))
                sampled_text = " ".join(smaller_sample)
                return create_wordcloud(sampled_text, max_words=50)
            except Exception as e:
                st.error(f"Error generating wordcloud: {str(e)}")
                return None
        with col1:
            st.write("### 🟢 Wordcloud Ulasan Positif")
            positive_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
            if not positive_reviews.empty:
                positive_text = " ".join(positive_reviews['teks_preprocessing'])
                if positive_text.strip():
                    with st.spinner('Generating positive word cloud...'):
                        pos_wordcloud = safe_create_wordcloud(positive_text)
                        if pos_wordcloud is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(pos_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig, use_container_width=True)
            st.write("#### Top Kata Positif berdasarkan TF-IDF")
            feature_names = tfidf_vectorizer.get_feature_names_out()
            pos_samples = positive_reviews['teks_preprocessing']
            pos_tfidf = tfidf_vectorizer.transform(pos_samples)
            if pos_tfidf.shape[0] > 0:
                pos_importance = np.asarray(pos_tfidf.mean(axis=0)).flatten()
                pos_indices = np.argsort(pos_importance)[-10:]
                pos_words_df = pd.DataFrame({'Word': [feature_names[i] for i in pos_indices], 'Importance': [pos_importance[i] for i in pos_indices]})
                fig = px.bar(pos_words_df, x='Importance', y='Word', orientation='h', title="Kata Kunci dalam Ulasan Positif", color='Importance', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data ulasan positif untuk ditampilkan.")
        with col2:
            st.write("### 🔴 Wordcloud Ulasan Negatif")
            negative_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
            if not negative_reviews.empty:
                negative_text = " ".join(negative_reviews['teks_preprocessing'])
                if negative_text.strip():
                    with st.spinner('Generating negative word cloud...'):
                        neg_wordcloud = safe_create_wordcloud(negative_text)
                        if neg_wordcloud is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(neg_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig, use_container_width=True)
            st.write("#### Top Kata Negatif berdasarkan TF-IDF")
            feature_names = tfidf_vectorizer.get_feature_names_out()
            neg_samples = negative_reviews['teks_preprocessing']
            neg_tfidf = tfidf_vectorizer.transform(neg_samples)
            if neg_tfidf.shape[0] > 0:
                neg_importance = np.asarray(neg_tfidf.mean(axis=0)).flatten()
                neg_indices = np.argsort(neg_importance)[-10:]
                neg_words_df = pd.DataFrame({'Word': [feature_names[i] for i in neg_indices], 'Importance': [neg_importance[i] for i in neg_indices]})
                fig = px.bar(neg_words_df, x='Importance', y='Word', orientation='h', title="Kata Kunci dalam Ulasan Negatif", color='Importance', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data ulasan negatif untuk ditampilkan.")
        st.subheader("🔍 Analisis Topik")
        bigrams = get_ngrams(" ".join(topic_data['teks_preprocessing']), 2, top_n=20)
        bigrams_df = pd.DataFrame(list(bigrams.items()), columns=['Bigram', 'Frequency'])
        fig = px.bar(bigrams_df.sort_values('Frequency', ascending=True).tail(10), x='Frequency', y='Bigram', orientation='h', title="Top 10 Frasa yang Paling Sering Muncul", color='Frequency', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("📋 Tabel Ulasan Interaktif")
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Cari Ulasan:", "")
    with col2:
        sentiment_filter = st.multiselect("Filter berdasarkan Sentimen:", options=["POSITIF", "NEGATIF"], default=["POSITIF", "NEGATIF"])
    filtered_display = topic_data
    if search_term:
        filtered_display = filtered_display[filtered_display['teks_preprocessing'].str.contains(search_term, case=False)]
    if sentiment_filter:
        filtered_display = filtered_display[filtered_display['sentiment'].isin(sentiment_filter)]
    sort_option = st.selectbox("Urutkan berdasarkan:", ["Terbaru", "Terlama", "Sentiment (Positif Dulu)", "Sentiment (Negatif Dulu)"])
    if sort_option == "Terbaru":
        filtered_display = filtered_display.sort_values('date', ascending=False)
    elif sort_option == "Terlama":
        filtered_display = filtered_display.sort_values('date', ascending=True)
    elif sort_option == "Sentiment (Positif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=False)
    elif sort_option == "Sentiment (Negatif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=True)
    if st.checkbox("Tampilkan Skor Confidence Model"):
        filtered_display = filtered_display.copy()
        filtered_display['confidence'] = filtered_display['review_text'].apply(lambda x: predict_sentiment(x, pipeline)['confidence'])
        def highlight_confidence(val):
            color = f'rgba(0, 255, 0, {val})' if val > 0.5 else f'rgba(255, 0, 0, {1-val})'
            return f'background-color: {color}'
        def style_sentiment(val):
            if val == 'POSITIF':
                return 'background-color: #c6efce; color: #006100'
            else:
                return 'background-color: #ffc7ce; color: #9c0006'
        st.dataframe(filtered_display.style.map(style_sentiment, subset=['sentiment']).map(highlight_confidence, subset=['confidence']), height=400)
    else:
        def style_sentiment(val):
            if val == 'POSITIF':
                return 'background-color: #c6efce; color: #006100'
            else:
                return 'background-color: #ffc7ce; color: #9c0006'
    # Urutkan data berdasarkan tanggal terbaru sebelum paginasi
    filtered_display = filtered_display.sort_values('date', ascending=False).reset_index(drop=True)
    rows_per_page = st.slider("Jumlah baris per halaman:", min_value=10, max_value=100, value=25, step=5)
    total_pages = max(1, len(filtered_display) // rows_per_page + (0 if len(filtered_display) % rows_per_page == 0 else 1))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.number_input("Halaman:", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(filtered_display))
    paginated_data = filtered_display.iloc[start_idx:end_idx]
    # Konversi semua kolom ke string agar Arrow tidak error
    paginated_data = paginated_data.copy()
    for col in paginated_data.columns:
        if paginated_data[col].dtype == 'object':
            paginated_data[col] = paginated_data[col].astype(str)
        elif pd.api.types.is_numeric_dtype(paginated_data[col]):
            paginated_data[col] = paginated_data[col].astype(str)
    st.dataframe(paginated_data, height=400)
    st.write(f"Menampilkan {start_idx+1}-{end_idx} dari {len(filtered_display)} ulasan (Halaman {current_page} dari {total_pages})")
    st.markdown(get_table_download_link(filtered_display, "goride_reviews_filtered", "📥 Download Data Terfilter (CSV)"), unsafe_allow_html=True)
    st.subheader("💡 Ringkasan Insights")
    pos_pct = metrics['pos_percentage']
    neg_pct = metrics['neg_percentage']
    pos_terms = get_word_frequencies(" ".join(filtered_display[filtered_display['sentiment'] == 'POSITIF']['teks_preprocessing']), top_n=5)
    neg_terms = get_word_frequencies(" ".join(filtered_display[filtered_display['sentiment'] == 'NEGATIF']['teks_preprocessing']), top_n=5)
    insights = []
    if pos_pct > 80:
        insights.append(f"✅ Sentimen sangat positif ({pos_pct:.1f}%), dengan kata kunci positif: {', '.join(list(pos_terms.keys())[:3])}")
    elif pos_pct > 60:
        insights.append(f"✅ Sentimen cukup positif ({pos_pct:.1f}%), dengan kata kunci positif: {', '.join(list(pos_terms.keys())[:3])}")
    elif pos_pct < 40:
        insights.append(f"❌ Sentimen cenderung negatif ({neg_pct:.1f}%), dengan kata kunci negatif: {', '.join(list(neg_terms.keys())[:3])}")
    else:
        insights.append(f"⚠️ Sentimen campuran ({pos_pct:.1f}% positif, {neg_pct:.1f}% negatif)")
    if 'sentiment_pivot' in locals() and len(sentiment_pivot) > 1:
        first_ratio = sentiment_pivot.iloc[0]['positive_percentage'] if 'positive_percentage' in sentiment_pivot.columns else 0
        last_ratio = sentiment_pivot.iloc[-1]['positive_percentage'] if 'positive_percentage' in sentiment_pivot.columns else 0
        if last_ratio - first_ratio > 5:
            insights.append(f"📈 Tren sentimen positif meningkat ({(last_ratio - first_ratio):.1f}% dalam periode yang dipilih)")
        elif first_ratio - last_ratio > 5:
            insights.append(f"📉 Tren sentimen positif menurun ({(first_ratio - last_ratio):.1f}% dalam periode yang dipilih)")
        else:
            insights.append("📊 Tren sentimen relatif stabil dalam periode yang dipilih")
    for i, insight in enumerate(insights):
        st.info(insight)
    if len(filtered_display[filtered_display['sentiment'] == 'NEGATIF']) > 0:
        st.subheader("🔄 Rekomendasi Tindakan")
        neg_text = " ".join(filtered_display[filtered_display['sentiment'] == 'NEGATIF']['teks_preprocessing'])
        neg_bigrams = get_ngrams(neg_text, 2, top_n=5)
        st.write("Berdasarkan analisis ulasan negatif, berikut beberapa area yang perlu mendapat perhatian:")
        for i, (bigram, freq) in enumerate(neg_bigrams.items(), 1):
            st.write(f"{i}. Tinjau masalah terkait **{bigram}** (disebutkan {freq} kali)")
        st.write("**Rekomendasi Umum:**")
        st.write("• Lakukan analisis mendalam untuk kategori ulasan negatif yang paling umum")
        st.write("• Pantau tren sentimen secara berkala untuk mengevaluasi dampak perubahan layanan")
        st.write("• Identifikasi driver dengan ulasan positif konsisten untuk best practice sharing")

    st.markdown("---")
    st.caption("© 2025 GoRide Sentiment Analysis App • Develop By Mhd Adreansyah")
    st.caption("Aplikasi ini merupakan Tugas Akhir/Skripsi dibawah perlindungan Hak Cipta")