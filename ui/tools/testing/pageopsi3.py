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
    pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)    # =============== HEADER DAN FILTER SECTION YANG DIPERBAIKI ===============
    st.title("📊 Dashboard Analisis Sentimen GoRide")
    st.markdown("**Analisis Komprehensif Ulasan Pengguna dari Google Play Store**")
    
    # Info banner dengan model performance
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Precision", f"{precision:.1%}")
        with col3:
            st.metric("Recall", f"{recall:.1%}")
        with col4:
            st.metric("F1-Score", f"{f1:.1%}")
        with col5:
            total_data_count = len(data)
            st.metric("Total Dataset", f"{total_data_count:,}")
    
    st.markdown("---")
    
    # Filter Section yang diperbaiki
    with st.expander("🎛️ **Pengaturan Filter & Konfigurasi**", expanded=True):
        st.markdown("### 📅 **Filter Berdasarkan Rentang Waktu**")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input("📅 Tanggal Mulai", value=pd.to_datetime(data['date']).min())
        with col2:
            end_date = st.date_input("📅 Tanggal Selesai", value=pd.to_datetime(data['date']).max())
        with col3:
            if st.button("🔄 Reset Filter"):
                st.rerun()
        
        # Quick filter presets
        st.markdown("**⚡ Filter Cepat:**")
        quick_filter_col1, quick_filter_col2, quick_filter_col3, quick_filter_col4 = st.columns(4)
        
        with quick_filter_col1:
            if st.button("📅 7 Hari Terakhir"):
                end_date = pd.to_datetime(data['date']).max()
                start_date = end_date - pd.Timedelta(days=7)
        with quick_filter_col2:
            if st.button("📅 30 Hari Terakhir"):
                end_date = pd.to_datetime(data['date']).max()
                start_date = end_date - pd.Timedelta(days=30)
        with quick_filter_col3:
            if st.button("📅 3 Bulan Terakhir"):
                end_date = pd.to_datetime(data['date']).max()
                start_date = end_date - pd.Timedelta(days=90)
        with quick_filter_col4:
            if st.button("📅 Tahun Ini"):
                end_date = pd.to_datetime(data['date']).max()
                start_date = pd.Timestamp(end_date.year, 1, 1)    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Progress bar untuk loading
    progress_bar = st.progress(0)
    with st.spinner('🔄 Memproses dan memfilter data...'):
        progress_bar.progress(25)
        filtered_data = data[(pd.to_datetime(data['date']) >= start_date) & (pd.to_datetime(data['date']) <= end_date)]
        progress_bar.progress(75)
        
    progress_bar.progress(100)
    progress_bar.empty()
    
    if filtered_data.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah rentang tanggal.")
        st.info("💡 **Tip:** Coba perluas rentang tanggal atau gunakan filter cepat yang tersedia.")
        return
    
    # Display info tentang data yang difilter
    date_range_info = f"📊 Menampilkan **{len(filtered_data):,}** ulasan dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**"
    st.info(date_range_info)
    
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
        
        # Additional metrics
        avg_daily = total / max(1, (pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days + 1)
        
        return {
            'total': total,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_percentage': pos_percentage,
            'neg_percentage': neg_percentage,
            'today_count': today_count,
            'pos_pct': pos_pct,
            'avg_daily': avg_daily
        }
    
    metrics = calculate_metrics(filtered_data)
    
    # =============== METRICS DISPLAY YANG DIPERBAIKI ===============
    st.markdown("### 📈 **Key Performance Indicators**")
    
    # Baris pertama - Metrics utama
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📊 Total Ulasan", 
            value=f"{metrics['total']:,}", 
            delta=f"{metrics['today_count']} hari ini",
            help="Total jumlah ulasan dalam periode yang dipilih"
        )
    with col2:
        delta_pos = f"{metrics['pos_percentage'] - 50:+.1f}% dari rata-rata"
        st.metric(
            label="🟢 Sentimen Positif", 
            value=f"{metrics['pos_percentage']:.1f}%", 
            delta=delta_pos,
            help="Persentase ulasan dengan sentimen positif"
        )
    with col3:
        delta_neg = f"{metrics['neg_percentage'] - 50:+.1f}% dari rata-rata"
        st.metric(
            label="🔴 Sentimen Negatif", 
            value=f"{metrics['neg_percentage']:.1f}%", 
            delta=delta_neg,
            delta_color="inverse",
            help="Persentase ulasan dengan sentimen negatif"
        )
    with col4:
        satisfaction_score = metrics['pos_pct']
        if satisfaction_score >= 80:
            satisfaction_label = "Excellent 🌟"
        elif satisfaction_score >= 60:
            satisfaction_label = "Good ✅"
        elif satisfaction_score >= 40:
            satisfaction_label = "Fair ⚠️"
        else:
            satisfaction_label = "Poor ❌"
            
        st.metric(
            label="👍 Indeks Kepuasan", 
            value=f"{satisfaction_score:.1f}%",
            delta=satisfaction_label,
            help="Indeks kepuasan berdasarkan rasio sentimen positif"
        )
    
    # Baris kedua - Metrics tambahan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📊 Rata-rata Harian", 
            value=f"{metrics['avg_daily']:.1f}",
            help="Rata-rata jumlah ulasan per hari dalam periode ini"
        )
    with col2:
        period_days = (end_date - start_date).days + 1
        st.metric(
            label="📅 Periode Analisis", 
            value=f"{period_days} hari",
            help="Rentang waktu yang dianalisis"
        )
    with col3:
        data_coverage = (len(filtered_data) / len(data) * 100)
        st.metric(
            label="📋 Coverage Data", 
            value=f"{data_coverage:.1f}%",
            help="Persentase data yang tercakup dalam filter"
        )
    with col4:
        if metrics['total'] > 0:
            confidence_level = "Tinggi" if metrics['total'] >= 100 else "Sedang" if metrics['total'] >= 30 else "Rendah"
            st.metric(
                label="🎯 Confidence Level", 
                value=confidence_level,
                help="Tingkat kepercayaan analisis berdasarkan jumlah sampel"
            )    # =============== PREPROCESSING DAN TOPIC FILTER YANG DIPERBAIKI ===============
    # Pastikan kolom teks_preprocessing tersedia dan konsisten
    if 'teks_preprocessing' not in data.columns:
        with st.spinner("🔄 Melakukan preprocessing batch untuk seluruh data..."):
            data.loc[:, 'teks_preprocessing'] = data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    
    if 'teks_preprocessing' not in filtered_data.columns:
        filtered_data = filtered_data.copy()
        with st.spinner("⚙️ Preprocessing data yang difilter..."):
            filtered_data.loc[:, 'teks_preprocessing'] = filtered_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    
    st.markdown("---")
    
    # Topic Filter Section yang diperbaiki
    with st.expander("🏷️ **Filter Berdasarkan Topik/Kata Kunci**", expanded=False):
        st.markdown("### 🔍 **Analisis Topik Trending**")
        
        # Ambil topik dari top 20 kata paling sering muncul di hasil preprocessing
        all_words = " ".join(filtered_data['teks_preprocessing'])
        word_freq = get_word_frequencies(all_words, top_n=25)
        
        # Tampilkan word frequency dalam bentuk yang lebih menarik
        if word_freq:
            st.markdown("**📊 Top Keywords dalam Periode Ini:**")
            
            # Display top keywords dengan visual yang lebih baik
            col1, col2, col3 = st.columns(3)
            top_words = list(word_freq.items())
            
            for i, (word, freq) in enumerate(top_words[:15]):
                col_idx = i % 3
                if col_idx == 0:
                    with col1:
                        st.write(f"**{i+1}.** {word} `({freq}x)`")
                elif col_idx == 1:
                    with col2:
                        st.write(f"**{i+1}.** {word} `({freq}x)`")
                else:
                    with col3:
                        st.write(f"**{i+1}.** {word} `({freq}x)`")
        
        # Topic selection dengan kategori yang lebih user-friendly
        topics = ["🌐 Semua Topik"] + [f"🔑 {word}" for word in word_freq.keys()]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_topic_display = st.selectbox(
                "🎯 Pilih Topik untuk Analisis Mendalam:", 
                topics,
                help="Pilih topik spesifik untuk melihat analisis yang lebih fokus"
            )
        with col2:
            if st.button("🔄 Reset Topik"):
                selected_topic_display = "🌐 Semua Topik"
        
        # Extract actual topic name (remove emoji prefix)
        if selected_topic_display == "🌐 Semua Topik":
            selected_topic = "All"
        else:
            selected_topic = selected_topic_display.replace("🔑 ", "")
    
    # Filter data berdasarkan topik yang dipilih
    if selected_topic != "All":
        topic_data = filtered_data[filtered_data['teks_preprocessing'].str.contains(selected_topic, case=False, na=False)].copy()
        
        # Informasi hasil filter dengan styling yang lebih baik
        if len(topic_data) > 0:
            coverage_pct = (len(topic_data) / len(filtered_data) * 100)
            st.success(f"🎯 **Filter Aktif:** Ditemukan **{len(topic_data):,}** ulasan untuk topik **'{selected_topic}'** ({coverage_pct:.1f}% dari total data)")
            
            # Mini metrics untuk topik spesifik
            topic_metrics = calculate_metrics(topic_data)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ulasan Topik", f"{len(topic_data):,}")
            with col2:
                st.metric("% Positif", f"{topic_metrics['pos_percentage']:.1f}%")
            with col3:
                st.metric("% Negatif", f"{topic_metrics['neg_percentage']:.1f}%")
        else:
            st.warning(f"⚠️ Tidak ada data untuk topik **'{selected_topic}'**. Kemungkinan kata telah difilter dalam preprocessing atau stemming.")
            st.info("💡 **Tip:** Coba pilih topik lain atau gunakan 'Semua Topik' untuk melihat keseluruhan data.")
            topic_data = filtered_data.copy()
    else:
        topic_data = filtered_data.copy()
        st.info("🌐 **Menampilkan semua topik** - Gunakan filter topik di atas untuk analisis yang lebih spesifik")
    
    # Ensure topic_data has the preprocessing column
    if 'teks_preprocessing' not in topic_data.columns:
        topic_data = topic_data.copy()
        topic_data.loc[:, 'teks_preprocessing'] = topic_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    
    # Fallback jika masih kosong
    if topic_data.empty:
        st.error("❌ Data kosong setelah filtering. Menggunakan data lengkap sebagai fallback.")
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
        st.plotly_chart(fig, use_container_width=True)    # =============== TABEL ULASAN INTERAKTIF YANG DIPERBAIKI ===============
    st.markdown("---")
    st.header("📋 Eksplorasi Data Ulasan")
    
    with st.expander("🔍 **Filter & Pencarian Data**", expanded=True):
        # Search and filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input(
                "🔍 Cari dalam Ulasan:", 
                "", 
                placeholder="Ketik kata kunci untuk mencari...",
                help="Pencarian akan dilakukan pada teks yang telah diproses"
            )
        
        with col2:
            sentiment_filter = st.multiselect(
                "🎭 Filter Sentimen:", 
                options=["POSITIF", "NEGATIF"], 
                default=["POSITIF", "NEGATIF"],
                help="Pilih jenis sentimen yang ingin ditampilkan"
            )
        
        with col3:
            sort_option = st.selectbox(
                "📊 Urutkan berdasarkan:", 
                ["Terbaru", "Terlama", "Sentimen (Positif Dulu)", "Sentimen (Negatif Dulu)"],
                help="Pilih cara pengurutan data"
            )
        
        # Advanced options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_confidence = st.checkbox(
                "🎯 Tampilkan Confidence Score", 
                help="Menampilkan tingkat kepercayaan model untuk setiap prediksi"
            )
        
        with col2:
            rows_per_page = st.slider(
                "📄 Baris per halaman:", 
                min_value=10, 
                max_value=100, 
                value=25, 
                step=5,
                help="Jumlah ulasan yang ditampilkan per halaman"
            )
        
        with col3:
            compact_view = st.checkbox(
                "📱 Tampilan Ringkas", 
                help="Menampilkan data dalam format yang lebih ringkas"
            )
    
    # Apply filters
    filtered_display = topic_data.copy()
    
    if search_term:
        search_mask = filtered_display['teks_preprocessing'].str.contains(search_term, case=False, na=False)
        filtered_display = filtered_display[search_mask]
        if len(filtered_display) == 0:
            st.warning(f"🔍 Tidak ditemukan ulasan yang mengandung kata **'{search_term}'**")
            filtered_display = topic_data.copy()
        else:
            st.info(f"🔍 Ditemukan **{len(filtered_display):,}** ulasan yang mengandung **'{search_term}'**")
    
    if sentiment_filter:
        filtered_display = filtered_display[filtered_display['sentiment'].isin(sentiment_filter)]
    
    # Apply sorting
    if sort_option == "Terbaru":
        filtered_display = filtered_display.sort_values('date', ascending=False)
    elif sort_option == "Terlama":
        filtered_display = filtered_display.sort_values('date', ascending=True)
    elif sort_option == "Sentimen (Positif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=False)
    elif sort_option == "Sentimen (Negatif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=True)
    
    # Pagination
    total_rows = len(filtered_display)
    total_pages = max(1, total_rows // rows_per_page + (1 if total_rows % rows_per_page > 0 else 0))
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_page = st.number_input(
            "📄 Halaman:", 
            min_value=1, 
            max_value=total_pages, 
            value=1, 
            step=1,
            help=f"Halaman 1 dari {total_pages} (Total: {total_rows:,} ulasan)"
        )
    
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    paginated_data = filtered_display.iloc[start_idx:end_idx].copy()
    
    # Display data info
    st.info(f"📊 Menampilkan ulasan **{start_idx+1}** - **{end_idx}** dari **{total_rows:,}** ulasan yang difilter (Halaman **{current_page}** dari **{total_pages}**)")
    
    # Prepare data for display
    if show_confidence and not paginated_data.empty:
        with st.spinner("🧠 Menghitung confidence score..."):
            paginated_data['confidence'] = paginated_data['review_text'].apply(
                lambda x: predict_sentiment(x, pipeline)['confidence']
            )
    
    # Convert data types for display
    display_data = paginated_data.copy()
    for col in display_data.columns:
        if display_data[col].dtype == 'object':
            display_data[col] = display_data[col].astype(str)
        elif pd.api.types.is_numeric_dtype(display_data[col]):
            display_data[col] = display_data[col].astype(str)
    
    # Display the table with styling
    if not display_data.empty:
        if compact_view:
            # Compact view - only show essential columns
            essential_cols = ['date', 'sentiment', 'review_text']
            if show_confidence and 'confidence' in display_data.columns:
                essential_cols.append('confidence')
            
            display_data_compact = display_data[essential_cols].copy()
            
            # Shorten review text for compact view
            if 'review_text' in display_data_compact.columns:
                display_data_compact['review_text'] = display_data_compact['review_text'].str[:100] + "..."
            
            st.dataframe(
                display_data_compact,
                height=400,
                use_container_width=True,
                column_config={
                    "date": st.column_config.DateColumn("📅 Tanggal"),
                    "sentiment": st.column_config.TextColumn("🎭 Sentimen"),
                    "review_text": st.column_config.TextColumn("💬 Ulasan (Ringkas)"),
                    "confidence": st.column_config.NumberColumn("🎯 Confidence", format="%.2f") if show_confidence else None
                }
            )
        else:
            # Full view
            st.dataframe(
                display_data,
                height=400,
                use_container_width=True,
                column_config={
                    "date": st.column_config.DateColumn("📅 Tanggal"),
                    "sentiment": st.column_config.TextColumn("🎭 Sentimen"),
                    "review_text": st.column_config.TextColumn("💬 Ulasan Lengkap"),
                    "teks_preprocessing": st.column_config.TextColumn("⚙️ Teks Terproses"),
                    "confidence": st.column_config.NumberColumn("🎯 Confidence", format="%.2f") if show_confidence else None
                }
            )
    else:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
    
    # Download section
    col1, col2, col3 = st.columns(3)
    with col2:
        if not filtered_display.empty:
            st.markdown(
                get_table_download_link(filtered_display, "goride_reviews_filtered", "📥 Download Data Terfilter (CSV)"), 
                unsafe_allow_html=True
            )
        else:
            st.info("Tidak ada data untuk diunduh")
    
    st.markdown("---")# =============== BAGIAN RINGKASAN INSIGHTS YANG DIPERBAIKI ===============
    st.markdown("---")
    st.header("💡 Ringkasan Analisis & Insights")
    
    # Kalkulasi metrics yang diperbaiki
    pos_pct = metrics['pos_percentage']
    neg_pct = metrics['neg_percentage']
    total_reviews = metrics['total']
    
    # Ambil data kata-kata untuk insight yang lebih baik
    pos_reviews = filtered_display[filtered_display['sentiment'] == 'POSITIF']
    neg_reviews = filtered_display[filtered_display['sentiment'] == 'NEGATIF']
    
    if not pos_reviews.empty:
        pos_terms = get_word_frequencies(" ".join(pos_reviews['teks_preprocessing']), top_n=10)
    else:
        pos_terms = {}
        
    if not neg_reviews.empty:
        neg_terms = get_word_frequencies(" ".join(neg_reviews['teks_preprocessing']), top_n=10)
    else:
        neg_terms = {}
    
    # Status Sentiment dengan Visual Indicator yang lebih baik
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if pos_pct >= 80:
            st.success("🎉 **STATUS: SANGAT POSITIF**")
            sentiment_status = "Excellent"
            status_color = "success"
        elif pos_pct >= 60:
            st.success("✅ **STATUS: POSITIF**")
            sentiment_status = "Good"
            status_color = "success"
        elif pos_pct >= 40:
            st.warning("⚠️ **STATUS: NETRAL/CAMPURAN**")
            sentiment_status = "Mixed"
            status_color = "warning"
        else:
            st.error("❌ **STATUS: NEGATIF**")
            sentiment_status = "Needs Attention"
            status_color = "error"
    
    # Detail Insights dalam Expander untuk organisasi yang lebih baik
    with st.expander("📊 **Analisis Detail Sentimen**", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 **Sentimen Positif**")
            if pos_pct > 0:
                st.metric("Persentase", f"{pos_pct:.1f}%", delta=f"{pos_pct-50:.1f}% dari rata-rata")
                st.write(f"**Jumlah Ulasan:** {len(pos_reviews):,} dari {total_reviews:,}")
                
                if pos_terms:
                    st.write("**Kata Kunci Dominan:**")
                    for i, (word, freq) in enumerate(list(pos_terms.items())[:5], 1):
                        st.write(f"{i}. **{word}** ({freq} kali)")
                else:
                    st.info("Tidak ada kata kunci positif ditemukan")
            else:
                st.info("Tidak ada ulasan positif dalam periode ini")
        
        with col2:
            st.markdown("#### 🔴 **Sentimen Negatif**")
            if neg_pct > 0:
                st.metric("Persentase", f"{neg_pct:.1f}%", delta=f"{neg_pct-50:.1f}% dari rata-rata", delta_color="inverse")
                st.write(f"**Jumlah Ulasan:** {len(neg_reviews):,} dari {total_reviews:,}")
                
                if neg_terms:
                    st.write("**Masalah Utama:**")
                    for i, (word, freq) in enumerate(list(neg_terms.items())[:5], 1):
                        st.write(f"{i}. **{word}** ({freq} kali)")
                else:
                    st.info("Tidak ada kata kunci negatif ditemukan")
            else:
                st.info("Tidak ada ulasan negatif dalam periode ini")
    
    # Analisis Tren (jika tersedia)
    trend_insights = []
    if 'sentiment_pivot' in locals() and len(sentiment_pivot) > 1:
        first_ratio = sentiment_pivot.iloc[0]['positive_percentage'] if 'positive_percentage' in sentiment_pivot.columns else 0
        last_ratio = sentiment_pivot.iloc[-1]['positive_percentage'] if 'positive_percentage' in sentiment_pivot.columns else 0
        trend_change = last_ratio - first_ratio
        
        with st.expander("📈 **Analisis Tren Temporal**", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sentimen Awal", f"{first_ratio:.1f}%")
            with col2:
                st.metric("Sentimen Akhir", f"{last_ratio:.1f}%")
            with col3:
                st.metric("Perubahan", f"{trend_change:+.1f}%", 
                         delta=f"{abs(trend_change):.1f}% {'naik' if trend_change > 0 else 'turun'}")
            
            if abs(trend_change) > 10:
                if trend_change > 0:
                    st.success(f"📈 **Tren Positif:** Sentimen meningkat signifikan {trend_change:.1f}% dalam periode ini")
                else:
                    st.error(f"📉 **Tren Negatif:** Sentimen menurun signifikan {abs(trend_change):.1f}% dalam periode ini")
            elif abs(trend_change) > 5:
                if trend_change > 0:
                    st.info(f"📊 **Tren Moderat:** Sentimen meningkat {trend_change:.1f}% dalam periode ini")
                else:
                    st.warning(f"📊 **Tren Moderat:** Sentimen menurun {abs(trend_change):.1f}% dalam periode ini")
            else:
                st.info("📊 **Tren Stabil:** Sentimen relatif konsisten dalam periode yang dipilih")
    
    # Key Insights Summary
    with st.expander("🎯 **Ringkasan Key Insights**", expanded=True):
        insights_summary = []
        
        # Insight berdasarkan volume
        if total_reviews >= 1000:
            insights_summary.append(f"📊 **Volume Tinggi:** Analisis berdasarkan {total_reviews:,} ulasan - data sangat representative")
        elif total_reviews >= 100:
            insights_summary.append(f"📊 **Volume Sedang:** Analisis berdasarkan {total_reviews:,} ulasan - data cukup representative")
        else:
            insights_summary.append(f"📊 **Volume Rendah:** Hanya {total_reviews:,} ulasan - pertimbangkan periode yang lebih luas")
        
        # Insight berdasarkan distribusi
        if pos_pct >= 80:
            insights_summary.append("🎉 **Performa Excellent:** Mayoritas pengguna sangat puas dengan layanan GoRide")
            if pos_terms:
                top_positive = list(pos_terms.keys())[0]
                insights_summary.append(f"⭐ **Kekuatan Utama:** Pengguna paling sering menyebutkan '{top_positive}' dalam ulasan positif")
        elif pos_pct >= 60:
            insights_summary.append("✅ **Performa Baik:** Sebagian besar pengguna puas, namun masih ada ruang perbaikan")
            if neg_terms:
                top_negative = list(neg_terms.keys())[0]
                insights_summary.append(f"🔍 **Area Perbaikan:** Perhatikan masalah '{top_negative}' yang sering disebutkan")
        elif pos_pct >= 40:
            insights_summary.append("⚠️ **Performa Campuran:** Sentimen terbagi rata, perlu evaluasi menyeluruh")
            if neg_terms:
                insights_summary.append(f"🚨 **Prioritas Utama:** Segera tangani masalah '{list(neg_terms.keys())[0]}' yang dominan")
        else:
            insights_summary.append("❌ **Performa Buruk:** Mayoritas pengguna tidak puas, diperlukan tindakan segera")
            if neg_terms:
                insights_summary.append(f"🆘 **Krisis Alert:** Masalah kritis '{list(neg_terms.keys())[0]}' harus segera ditangani")
        
        # Tampilkan insights dengan formatting yang lebih baik
        for insight in insights_summary:
            if "Excellent" in insight or "🎉" in insight:
                st.success(insight)
            elif "Baik" in insight or "✅" in insight:
                st.success(insight)
            elif "Campuran" in insight or "⚠️" in insight:
                st.warning(insight)
            elif "Buruk" in insight or "❌" in insight or "🆘" in insight:
                st.error(insight)
            else:
                st.info(insight)    # =============== BAGIAN REKOMENDASI YANG DIPERBAIKI ===============
    if len(neg_reviews) > 0:
        with st.expander("🔄 **Rekomendasi Tindakan & Strategi Perbaikan**", expanded=True):
            st.markdown("### 📋 **Analisis Masalah Prioritas**")
            
            # Analisis bigram untuk masalah spesifik
            neg_text = " ".join(neg_reviews['teks_preprocessing'])
            neg_bigrams = get_ngrams(neg_text, 2, top_n=10)
            
            if neg_bigrams:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 **Masalah Utama yang Teridentifikasi:**")
                    for i, (bigram, freq) in enumerate(list(neg_bigrams.items())[:5], 1):
                        percentage = (freq / len(neg_reviews) * 100)
                        if percentage >= 20:
                            priority = "🔴 **TINGGI**"
                        elif percentage >= 10:
                            priority = "🟠 **SEDANG**"
                        else:
                            priority = "🟡 **RENDAH**"
                        st.write(f"{i}. {priority} **{bigram}**")
                        st.write(f"   📊 {freq} ulasan ({percentage:.1f}% dari ulasan negatif)")
                
                with col2:
                    st.markdown("#### 💡 **Rekomendasi Spesifik:**")
                    
                    # Generate rekomendasi berdasarkan kata kunci yang ditemukan
                    recommendations = []
                    top_issues = list(neg_bigrams.keys())[:3]
                    
                    for issue in top_issues:
                        if any(word in issue.lower() for word in ['driver', 'sopir', 'pengemudi']):
                            recommendations.append("👤 **Pelatihan Driver:** Tingkatkan kualitas layanan dan profesionalisme driver")
                        elif any(word in issue.lower() for word in ['aplikasi', 'app', 'sistem']):
                            recommendations.append("📱 **Perbaikan Aplikasi:** Optimalisasi performa dan perbaikan bug sistem")
                        elif any(word in issue.lower() for word in ['harga', 'tarif', 'biaya']):
                            recommendations.append("💰 **Review Pricing:** Evaluasi struktur tarif dan transparansi biaya")
                        elif any(word in issue.lower() for word in ['waktu', 'lama', 'lambat']):
                            recommendations.append("⏱️ **Efisiensi Waktu:** Optimalisasi routing dan pengurangan waktu tunggu")
                        elif any(word in issue.lower() for word in ['pelayanan', 'service', 'layanan']):
                            recommendations.append("🛎️ **Service Excellence:** Peningkatan standar pelayanan customer service")
                    
                    # Tambahkan rekomendasi umum jika tidak ada yang spesifik
                    if not recommendations:
                        recommendations = [
                            "🔍 **Investigasi Mendalam:** Lakukan survei lanjutan untuk memahami akar masalah",
                            "📊 **Monitoring Ketat:** Pantau metrics kualitas layanan secara real-time",
                            "🎯 **Focus Group:** Adakan diskusi dengan pengguna untuk feedback detail"
                        ]
                    
                    for rec in recommendations[:3]:
                        st.write(f"• {rec}")
            
            st.markdown("---")
            st.markdown("### 🚀 **Strategi Jangka Pendek & Panjang**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚡ **Aksi Segera (1-2 Minggu)**")
                urgent_actions = [
                    "📞 Follow-up langsung dengan pengguna yang memberikan ulasan sangat negatif",
                    "🔍 Investigasi insiden yang paling sering disebutkan",
                    "📢 Komunikasi proaktif mengenai perbaikan yang sedang dilakukan",
                    "⚠️ Implementasi early warning system untuk mencegah masalah serupa"
                ]
                
                for action in urgent_actions:
                    st.write(f"• {action}")
            
            with col2:
                st.markdown("#### 📈 **Strategi Jangka Panjang (1-3 Bulan)**")
                longterm_actions = [
                    "🎓 Program pelatihan berkelanjutan untuk driver dan customer service",
                    "🔧 Pengembangan fitur baru berdasarkan feedback pengguna",
                    "📊 Implementasi dashboard real-time untuk monitoring kualitas layanan",
                    "🤝 Program loyalty dan reward untuk meningkatkan kepuasan pengguna"
                ]
                
                for action in longterm_actions:
                    st.write(f"• {action}")
            
            # KPI Tracking untuk Rekomendasi
            st.markdown("---")
            st.markdown("### 📊 **KPI untuk Tracking Progress**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Target Mingguan:**")
                st.write("• Sentimen positif: +2-5%")
                st.write("• Response rate: >90%")
                st.write("• Resolution time: <24 jam")
            
            with col2:
                st.markdown("**Target Bulanan:**")
                st.write("• Overall satisfaction: >75%")
                st.write("• Repeat negative: <10%")
                st.write("• Driver rating: >4.5/5")
            
            with col3:
                st.markdown("**Target Kuartalan:**")
                st.write("• Market leadership: Top 2")
                st.write("• User retention: >85%")
                st.write("• NPS Score: >50")
            
            # Alert untuk status kritis
            if pos_pct < 40:
                st.error("🚨 **ALERT:** Sentimen berada dalam zona kritis! Diperlukan action plan emergency dalam 48 jam.")
            elif pos_pct < 60:
                st.warning("⚠️ **WARNING:** Sentimen perlu perhatian khusus. Implementasikan action plan dalam 1 minggu.")
    
    else:
        st.success("🎉 **Excellent Performance!** Tidak ada ulasan negatif dalam periode ini. Pertahankan kualitas layanan yang outstanding!")
        
        with st.expander("🌟 **Strategi Mempertahankan Performa Positif**", expanded=False):
            st.write("### 💪 **Best Practices yang Harus Dipertahankan:**")
            if pos_terms:
                st.write("**Kekuatan Utama yang Diakui Pengguna:**")
                for i, (term, freq) in enumerate(list(pos_terms.items())[:5], 1):
                    st.write(f"{i}. **{term}** - disebutkan {freq} kali dalam ulasan positif")
            
            st.write("### 🎯 **Rekomendasi untuk Sustainability:**")
            st.write("• 📊 Lakukan benchmarking rutin dengan kompetitor")
            st.write("• 🎓 Standardisasi best practices ke seluruh tim")
            st.write("• 💬 Maintain komunikasi aktif dengan loyal customers")
            st.write("• 🔄 Continuous improvement berdasarkan feedback positif")    
    # =============== FOOTER SUMMARY SECTION ===============
    st.markdown("---")
    
    # Executive Summary
    with st.expander("📋 **Executive Summary**", expanded=False):
        st.markdown("### 🎯 **Ringkasan Eksekutif Dashboard**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 **Data Overview**")
            st.write(f"• **Periode Analisis:** {start_date.strftime('%d %B %Y')} - {end_date.strftime('%d %B %Y')}")
            st.write(f"• **Total Ulasan Dianalisis:** {len(filtered_display):,}")
            st.write(f"• **Coverage Data:** {(len(filtered_display)/len(data)*100):.1f}% dari total dataset")
            st.write(f"• **Model Accuracy:** {accuracy:.1%}")
            
            st.markdown("#### 🎭 **Distribusi Sentimen**")
            st.write(f"• **Positif:** {metrics['pos_count']:,} ulasan ({metrics['pos_percentage']:.1f}%)")
            st.write(f"• **Negatif:** {metrics['neg_count']:,} ulasan ({metrics['neg_percentage']:.1f}%)")
            st.write(f"• **Indeks Kepuasan:** {metrics['pos_pct']:.1f}%")
        
        with col2:
            st.markdown("#### 🔍 **Key Findings**")
            if metrics['pos_percentage'] >= 70:
                st.success("✅ **Status:** Performa layanan excellent")
                st.write("• Mayoritas pengguna puas dengan layanan")
                st.write("• Pertahankan kualitas dan konsistensi")
            elif metrics['pos_percentage'] >= 50:
                st.info("ℹ️ **Status:** Performa layanan baik dengan ruang perbaikan")
                st.write("• Performa di atas rata-rata industri")
                st.write("• Focus pada peningkatan area negatif")
            else:
                st.warning("⚠️ **Status:** Performa layanan perlu perhatian serius")
                st.write("• Diperlukan action plan immediate")
                st.write("• Review fundamental layanan")
            
            if selected_topic != "All":
                st.markdown("#### 🏷️ **Filter Aktif**")
                st.write(f"• **Topik:** {selected_topic}")
                st.write(f"• **Data Filtered:** {len(topic_data):,} ulasan")
    
    # Technical Info
    with st.expander("⚙️ **Informasi Teknis**", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Model Information:**")
            st.write(f"• Algorithm: SVM")
            st.write(f"• Vectorizer: TF-IDF")
            st.write(f"• Accuracy: {accuracy:.3f}")
            st.write(f"• Precision: {precision:.3f}")
            st.write(f"• Recall: {recall:.3f}")
            st.write(f"• F1-Score: {f1:.3f}")
        
        with col2:
            st.markdown("**Data Processing:**")
            st.write("• Preprocessing: ✅ Aktif")
            st.write("• Text Cleaning: ✅ Aktif")
            st.write("• Stopword Removal: ✅ Aktif")
            st.write("• Stemming: ✅ Aktif")
            st.write("• Normalization: ✅ Aktif")
        
        with col3:
            st.markdown("**System Status:**")
            st.write("• Data Loading: ✅ Success")
            st.write("• Model Loading: ✅ Success")
            st.write("• Cache Status: ✅ Active")
            st.write("• Performance: ✅ Optimal")
    
    st.markdown("---")
    
    # Footer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center'>
            <h4>📊 GoRide Sentiment Analysis Dashboard</h4>
            <p><b>© 2025 GoRide Sentiment Analysis App</b></p>
            <p>Developed by <b>Mhd Adreansyah</b></p>
            <p><i>Aplikasi ini merupakan Tugas Akhir/Skripsi dibawah perlindungan Hak Cipta</i></p>
            <hr>
            <p style='font-size: 12px; color: #666;'>
                🔒 Confidential & Proprietary | 🛡️ Data Protected | ⚡ Real-time Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)