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
    
    # Welcome header dengan styling yang lebih baik
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f77b4, #2ca02c); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0; text-align: center;">📊 Dashboard Analisis Sentimen GoRide</h1>
        <p style="color: white; margin: 5px 0 0 0; text-align: center; opacity: 0.9;">Analisis Ulasan Pengguna dari Google Play Store</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan toast jika login baru saja berhasil
    if st.session_state.get('login_success', False):
        st.toast(f"User {st.session_state.get('user_email', '')} login successfully!", icon="✅")
        st.session_state['login_success'] = False
          # Load data dan model dengan progress bar
    with st.spinner('🔄 Memuat data dan model...'):
        data = load_sample_data()
        if 'data_loaded_toast_shown' not in st.session_state:
            if not data.empty:
                st.success(f"✅ Data berhasil dimuat: {len(data):,} ulasan")
            else:
                st.error("❌ Data gagal dimuat atau kosong!")
                return
            st.session_state['data_loaded_toast_shown'] = True
    
    # Preprocessing options
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
    
    with st.spinner('🤖 Memuat model analisis sentimen...'):
        pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)

    # Sidebar untuk kontrol filter dengan styling yang lebih baik
    with st.sidebar:
        st.markdown("### 🎛️ Kontrol Filter")
        
        # Model Performance Info
        with st.expander("📈 Performa Model", expanded=False):
            st.metric("Akurasi", f"{accuracy:.2%}")
            st.metric("Precision", f"{precision:.2%}")
            st.metric("Recall", f"{recall:.2%}")
            st.metric("F1-Score", f"{f1:.2%}")
          # Date filter dengan styling
        st.markdown("#### 📅 Filter Waktu")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Mulai", value=pd.to_datetime(data['date']).min())
        with col2:
            end_date = st.date_input("Selesai", value=pd.to_datetime(data['date']).max())
    
    # Convert dates to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
      # Validasi tanggal
    if start_date > end_date:
        st.sidebar.error("❌ Tanggal mulai tidak boleh lebih besar dari tanggal selesai!")
        return
    
    with st.spinner('🔍 Memfilter data...'):
        filtered_data = data[(pd.to_datetime(data['date']) >= start_date) & (pd.to_datetime(data['date']) <= end_date)]
        
    if filtered_data.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.info("💡 Coba perluas rentang tanggal atau periksa kembali filter Anda.")
        return
    
    # Metrics calculation function
    @st.cache_data(ttl=300)
    def calculate_metrics(df):
        total = len(df)
        pos_count = len(df[df['sentiment'] == 'POSITIF'])
        neg_count = len(df[df['sentiment'] == 'NEGATIF'])
        pos_percentage = (pos_count / total * 100) if total > 0 else 0
        neg_percentage = (neg_count / total * 100) if total > 0 else 0
        pos_pct = pos_count/total*100 if total > 0 else 0
        
        # Hitung perubahan dari periode sebelumnya
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        today_count = len(df[df['date'] == today])
        yesterday_count = len(df[df['date'] == yesterday])
        daily_change = today_count - yesterday_count
        
        return {
            'total': total,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_percentage': pos_percentage,
            'neg_percentage': neg_percentage,
            'today_count': today_count,
            'pos_pct': pos_pct,
            'daily_change': daily_change
        }
    
    metrics = calculate_metrics(filtered_data)
    
    # KPI Cards dengan styling yang lebih baik
    st.markdown("### 📊 Ringkasan Kinerja")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if metrics['daily_change'] >= 0 else "inverse"
        st.metric(
            label="📈 Total Ulasan", 
            value=f"{metrics['total']:,}", 
            delta=f"{metrics['daily_change']:+d} hari ini",
            delta_color=delta_color
        )
    
    with col2:
        pos_delta = metrics['pos_percentage'] - 50
        delta_color = "normal" if pos_delta > 0 else "inverse"
        st.metric(
            label="🟢 Sentimen Positif", 
            value=f"{metrics['pos_percentage']:.1f}%", 
            delta=f"{pos_delta:+.1f}% dari netral",
            delta_color=delta_color
        )
    
    with col3:
        neg_delta = metrics['neg_percentage'] - 50
        delta_color = "inverse" if neg_delta > 0 else "normal"
        st.metric(
            label="🔴 Sentimen Negatif", 
            value=f"{metrics['neg_percentage']:.1f}%", 
            delta=f"{neg_delta:+.1f}% dari netral",
            delta_color=delta_color        )
    
    with col4:
        satisfaction_delta = metrics['pos_pct'] - 70  # 70% sebagai target kepuasan
        delta_color = "normal" if satisfaction_delta > 0 else "inverse"
        st.metric(
            label="👍 Indeks Kepuasan", 
            value=f"{metrics['pos_pct']:.1f}%", 
            delta=f"{satisfaction_delta:+.1f}% dari target",
            delta_color=delta_color
        )
    
    # Pastikan kolom teks_preprocessing tersedia dan konsisten
    if 'teks_preprocessing' not in data.columns:
        with st.spinner("🔄 Melakukan preprocessing batch untuk seluruh data..."):
            data.loc[:, 'teks_preprocessing'] = data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
            
    if 'teks_preprocessing' not in filtered_data.columns:
        filtered_data = filtered_data.copy()
        with st.spinner("🔄 Preprocessing data terfilter..."):
            filtered_data.loc[:, 'teks_preprocessing'] = filtered_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))    # Topic filter dengan UI yang lebih baik
    with st.sidebar:
        st.markdown("#### 🏷️ Filter Topik")
        # Ambil topik dari top 20 kata paling sering muncul di hasil preprocessing
        all_words = " ".join(filtered_data['teks_preprocessing'])
        word_freq = get_word_frequencies(all_words, top_n=20)
        topics = ["Semua Topik"] + list(word_freq.keys())
        selected_topic = st.selectbox("Pilih topik:", topics, key="topic_filter")
        
        if selected_topic != "Semua Topik":
            topic_data = filtered_data[filtered_data['teks_preprocessing'].str.contains(selected_topic, case=False)].copy()
            st.info(f"📌 {len(topic_data):,} ulasan untuk topik '{selected_topic}'")
        else:
            topic_data = filtered_data.copy()
        
        # Advanced filters
        with st.expander("🔧 Filter Lanjutan"):
            min_rating = st.slider("Rating minimum:", 1, 5, 1)
            # Apply rating filter if available
            if 'rating' in topic_data.columns:
                topic_data = topic_data[topic_data['rating'] >= min_rating]
                  # Validasi topic data
        if 'teks_preprocessing' not in topic_data.columns:
            topic_data = topic_data.copy()
            topic_data.loc[:, 'teks_preprocessing'] = topic_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
            
        if topic_data.empty:
            st.sidebar.warning(f"⚠️ Tidak ada data untuk topik '{selected_topic}'")
            st.sidebar.info("💡 Coba pilih topik lain atau periksa hasil preprocessing")
            topic_data = filtered_data.copy()

    # Main content area dengan tabs yang lebih terorganisir
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "📈 Tren Waktu", "📝 Analisis Kata", "🔍 Insight Cerdas"])
    
    with tab1:
        st.markdown("### 📊 Distribusi Sentimen")
        
        # Alert untuk kondisi data
        if topic_data.empty:
            st.error("⚠️ Tidak ada data untuk ditampilkan. Silakan sesuaikan filter Anda.")
            return
            
        sentiment_counts = topic_data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Tambahkan persentase
        sentiment_counts['Percentage'] = (sentiment_counts['Count'] / sentiment_counts['Count'].sum() * 100).round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart dengan styling yang lebih baik
            bar_chart = px.bar(
                sentiment_counts, 
                x='Sentiment', 
                y='Count', 
                color='Sentiment',
                color_discrete_map={'POSITIF': '#28a745', 'NEGATIF': '#dc3545'},
                title="📊 Distribusi Jumlah Ulasan",
                text='Count'
            )
            bar_chart.update_traces(texttemplate='%{text}', textposition='outside')
            bar_chart.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(bar_chart, use_container_width=True)
            
        with col2:
            # Pie chart dengan informasi yang lebih detail
            pie_chart = px.pie(
                sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={'POSITIF': '#28a745', 'NEGATIF': '#dc3545'},
                title="📈 Persentase Sentimen"
            )
            pie_chart.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'
            )
            pie_chart.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(pie_chart, use_container_width=True)
            
        # Summary metrics untuk tab ini        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Ulasan", f"{len(topic_data):,}")
        with col2:
            pos_pct = (len(topic_data[topic_data['sentiment'] == 'POSITIF']) / len(topic_data) * 100)
            st.metric("🟢 Positif", f"{pos_pct:.1f}%")
        # with col3:
        #     neg_pct = (len(topic_data[topic_data['sentiment'] == 'NEGATIF']) / len(topic_data) * 100)
        #     st.metric("🔴 Negatif", f"{neg_pct:.1f}%")
    
    with tab2:
        st.markdown("### 📈 Tren Sentimen dari Waktu ke Waktu")
        
        # Time granularity controls dengan styling yang lebih baik
        col1, col2 = st.columns([3, 1])
        with col1:
            time_granularity = st.radio(
                "Pilih periode analisis:", 
                options=["Harian", "Mingguan", "Bulanan"], 
                horizontal=True,
                help="Pilih granularitas waktu untuk analisis tren"
            )
        with col2:
            show_prediction = st.checkbox("🔮 Prediksi Tren", help="Tampilkan prediksi tren untuk periode selanjutnya")
        
        # Data sampling untuk performa yang lebih baik
        visualization_data = topic_data.copy()
        if len(topic_data) > 10000:
            sample_size = min(10000, int(len(topic_data) * 0.3))
            st.info(f"📊 Dataset besar ({len(topic_data):,} baris). Menggunakan sampel representatif {sample_size:,} baris untuk visualisasi optimal.")
            visualization_data = topic_data.sample(sample_size, random_state=42)
            
            use_all_data = st.toggle("Gunakan semua data", help="⚠️ Menggunakan semua data dapat memperlambat aplikasi")
            if use_all_data:
                visualization_data = topic_data
                st.warning("🐌 Menggunakan dataset lengkap. Harap bersabar...")
        
        # Group data berdasarkan granularitas waktu
        if time_granularity == "Harian":
            visualization_data = visualization_data.copy()
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m-%d')
            unique_days = visualization_data['time_group'].nunique()
            if unique_days > 100:
                st.info(f"📅 Data harian terlalu padat ({unique_days} hari). Auto-grouping ke mingguan.")
                visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.to_period('W').astype(str)
        elif time_granularity == "Mingguan":
            visualization_data = visualization_data.copy()
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-W%U')
        else:  # Bulanan
            visualization_data = visualization_data.copy()
            visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m')
        
        # Buat pivot table untuk tren
        try:
            sentiment_trend = visualization_data.groupby(['time_group', 'sentiment']).size().reset_index(name='count')
            sentiment_pivot = sentiment_trend.pivot(index='time_group', columns='sentiment', values='count').fillna(0).reset_index()
            
            # Pastikan kolom sentimen ada
            for sentiment in ['POSITIF', 'NEGATIF']:
                if sentiment not in sentiment_pivot.columns:
                    sentiment_pivot[sentiment] = 0
            
            sentiment_pivot['total'] = sentiment_pivot['POSITIF'] + sentiment_pivot['NEGATIF']
            sentiment_pivot['positive_percentage'] = np.where(
                sentiment_pivot['total'] > 0,
                (sentiment_pivot['POSITIF'] / sentiment_pivot['total'] * 100).round(2),
                0
            )
            
            # Visualisasi tren jumlah
            fig_count = px.line(
                sentiment_pivot, 
                x='time_group', 
                y=['POSITIF', 'NEGATIF'],
                title=f"📊 Tren Jumlah Ulasan ({time_granularity})",
                labels={'value': 'Jumlah Ulasan', 'time_group': 'Periode', 'variable': 'Sentimen'},
                color_discrete_map={'POSITIF': '#28a745', 'NEGATIF': '#dc3545'}
            )
            fig_count.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_count, use_container_width=True)
            
            # Visualisasi persentase sentimen positif
            fig_pct = px.line(
                sentiment_pivot, 
                x='time_group', 
                y='positive_percentage',
                title=f"📈 Tren Persentase Sentimen Positif ({time_granularity})",
                labels={'positive_percentage': '% Sentimen Positif', 'time_group': 'Periode'},
                color_discrete_sequence=['#17a2b8']
            )
            
            # Tambahkan garis referensi 50%
            fig_pct.add_hline(y=50, line_dash="dash", line_color="gray", 
                             annotation_text="Netral (50%)", annotation_position="bottom right")
            fig_pct.add_hline(y=70, line_dash="dot", line_color="green", 
                             annotation_text="Target (70%)", annotation_position="top right")
            
            fig_pct.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pct, use_container_width=True)
            
            # Insight cepat berdasarkan tren
            if len(sentiment_pivot) > 1:
                latest_pct = sentiment_pivot.iloc[-1]['positive_percentage']
                previous_pct = sentiment_pivot.iloc[-2]['positive_percentage'] if len(sentiment_pivot) > 1 else 50
                trend_change = latest_pct - previous_pct
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Periode Terakhir", f"{latest_pct:.1f}%", f"{trend_change:+.1f}%")
                with col2:
                    avg_pct = sentiment_pivot['positive_percentage'].mean()
                    st.metric("📈 Rata-rata", f"{avg_pct:.1f}%")
                with col3:
                    volatility = sentiment_pivot['positive_percentage'].std()
                    st.metric("📉 Volatilitas", f"{volatility:.1f}%")
            
            # Download data tren
            csv = sentiment_pivot.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            download_link = f'<a href="data:file/csv;base64,{b64}" download="sentiment_trend_{time_granularity.lower()}.csv" style="text-decoration: none;">📥 Download Data Tren (CSV)</a>'
            st.markdown(download_link, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error dalam generate chart tren: {str(e)}")
            st.info("💡 Coba sesuaikan rentang tanggal atau filter untuk memperbanyak data point.")    # Fungsi helper untuk word cloud (cache untuk performa)
    @st.cache_data(ttl=3600)
    def safe_create_wordcloud(text, max_words=100, max_length=10000, timeout_seconds=15):
        import threading
        import time
        
        if len(text) > max_length:
            st.info(f"📊 Teks dikurangi dari {len(text):,} ke {max_length:,} karakter untuk optimasi")
            words = text.split()
            sampled_words = random.sample(words, min(max_length, len(words)))
            text = " ".join(sampled_words)
        
        try:
            # Memory check
            try:
                import psutil
                process = psutil.Process(os.getpid())
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > 1000:  # Jika memory > 1GB
                    max_words = min(50, max_words)
                    st.info("🔧 Mengurangi kompleksitas karena penggunaan memory tinggi")
            except ImportError:
                pass
            
            # Generate wordcloud dengan timeout handling
            start_time = time.time()
            wordcloud = create_wordcloud(text, max_words=max_words)
            generation_time = time.time() - start_time
            
            if generation_time > 3:
                st.info(f"⏱️ Word cloud dibuat dalam {generation_time:.1f} detik")
            
            return wordcloud
            
        except Exception as e:
            st.error(f"❌ Error membuat word cloud: {str(e)}")
            return None

    with tab3:
        st.markdown("### 📝 Analisis Kata dalam Ulasan")
        
        # Kontrol untuk analisis kata
        col1, col2 = st.columns([3, 1])
        with col1:            analysis_type = st.radio(
                "Pilih jenis analisis:", 
                ["Word Cloud", "TF-IDF Analysis", "N-gram Analysis"], 
                horizontal=True
            )
        with col2:
            max_words = st.slider("Max kata:", 20, 200, 100, help="Jumlah maksimal kata untuk ditampilkan")
        
        if analysis_type == "Word Cloud":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Word Cloud - Ulasan Positif")
                positive_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
                if not positive_reviews.empty:
                    positive_text = " ".join(positive_reviews['teks_preprocessing'])
                    if positive_text.strip():
                        with st.spinner('🔄 Membuat word cloud positif...'):
                            pos_wordcloud = safe_create_wordcloud(positive_text, max_words=max_words)
                            if pos_wordcloud is not None:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(pos_wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title('Kata-kata dalam Ulasan Positif', fontsize=14, pad=20)
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan positif")
                    else:
                        st.info("📝 Tidak ada teks yang cukup untuk word cloud positif")
                else:
                    st.info("📝 Tidak ada ulasan positif dalam data yang dipilih")
            
            with col2:
                st.markdown("#### 🔴 Word Cloud - Ulasan Negatif")
                negative_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
                if not negative_reviews.empty:
                    negative_text = " ".join(negative_reviews['teks_preprocessing'])
                    if negative_text.strip():
                        with st.spinner('🔄 Membuat word cloud negatif...'):
                            neg_wordcloud = safe_create_wordcloud(negative_text, max_words=max_words)
                            if neg_wordcloud is not None:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(neg_wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title('Kata-kata dalam Ulasan Negatif', fontsize=14, pad=20)
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan negatif")
                    else:
                        st.info("📝 Tidak ada teks yang cukup untuk word cloud negatif")
                else:
                    st.info("📝 Tidak ada ulasan negatif dalam data yang dipilih")
        
        elif analysis_type == "TF-IDF Analysis":
            st.markdown("#### 🔍 Analisis Kata Kunci berdasarkan TF-IDF")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🟢 Kata Kunci Ulasan Positif")
                positive_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
                if not positive_reviews.empty and len(positive_reviews) > 0:
                    try:
                        feature_names = tfidf_vectorizer.get_feature_names_out()
                        pos_samples = positive_reviews['teks_preprocessing']
                        pos_tfidf = tfidf_vectorizer.transform(pos_samples)
                        
                        if pos_tfidf.shape[0] > 0:
                            pos_importance = np.asarray(pos_tfidf.mean(axis=0)).flatten()
                            top_n = min(15, len(feature_names))
                            pos_indices = np.argsort(pos_importance)[-top_n:]
                            
                            pos_words_df = pd.DataFrame({
                                'Kata': [feature_names[i] for i in pos_indices],
                                'Skor TF-IDF': [pos_importance[i] for i in pos_indices]
                            }).sort_values('Skor TF-IDF', ascending=True)
                            
                            fig = px.bar(
                                pos_words_df, 
                                x='Skor TF-IDF', 
                                y='Kata', 
                                orientation='h',
                                title="Top Kata dalam Ulasan Positif",
                                color='Skor TF-IDF',
                                color_continuous_scale='Greens'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📝 Tidak ada data untuk analisis TF-IDF positif")
                    except Exception as e:
                        st.error(f"❌ Error dalam analisis TF-IDF positif: {str(e)}")
                else:
                    st.info("📝 Tidak ada ulasan positif untuk dianalisis")
            
            with col2:
                st.markdown("##### 🔴 Kata Kunci Ulasan Negatif")
                negative_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
                if not negative_reviews.empty and len(negative_reviews) > 0:
                    try:
                        feature_names = tfidf_vectorizer.get_feature_names_out()
                        neg_samples = negative_reviews['teks_preprocessing']
                        neg_tfidf = tfidf_vectorizer.transform(neg_samples)
                        
                        if neg_tfidf.shape[0] > 0:
                            neg_importance = np.asarray(neg_tfidf.mean(axis=0)).flatten()
                            top_n = min(15, len(feature_names))
                            neg_indices = np.argsort(neg_importance)[-top_n:]
                            
                            neg_words_df = pd.DataFrame({
                                'Kata': [feature_names[i] for i in neg_indices],
                                'Skor TF-IDF': [neg_importance[i] for i in neg_indices]
                            }).sort_values('Skor TF-IDF', ascending=True)
                            
                            fig = px.bar(
                                neg_words_df, 
                                x='Skor TF-IDF', 
                                y='Kata', 
                                orientation='h',
                                title="Top Kata dalam Ulasan Negatif",
                                color='Skor TF-IDF',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📝 Tidak ada data untuk analisis TF-IDF negatif")
                    except Exception as e:
                        st.error(f"❌ Error dalam analisis TF-IDF negatif: {str(e)}")
                else:
                    st.info("📝 Tidak ada ulasan negatif untuk dianalisis")
        
        else:  # N-gram Analysis
            st.markdown("#### 🔍 Analisis N-gram (Frasa)")
            ngram_type = st.selectbox("Pilih jenis N-gram:", ["Bigram (2 kata)", "Trigram (3 kata)"])
            n = 2 if ngram_type == "Bigram (2 kata)" else 3
            
            try:
                all_text = " ".join(topic_data['teks_preprocessing'])
                ngrams = get_ngrams(all_text, n, top_n=min(20, max_words//5))
                
                if ngrams:
                    ngrams_df = pd.DataFrame(
                        list(ngrams.items()), 
                        columns=[f'{ngram_type.split()[0]}', 'Frekuensi']
                    ).sort_values('Frekuensi', ascending=True).tail(15)
                    
                    fig = px.bar(
                        ngrams_df, 
                        x='Frekuensi', 
                        y=f'{ngram_type.split()[0]}',                        orientation='h',
                        title=f"Top 15 {ngram_type} yang Paling Sering Muncul",
                        color='Frekuensi',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tampilkan tabel detail
                    with st.expander("📋 Lihat Detail N-gram"):
                        st.dataframe(
                            ngrams_df.sort_values('Frekuensi', ascending=False).reset_index(drop=True),
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Tidak dapat membuat analisis N-gram dari data yang tersedia")
                    
            except Exception as e:
                st.error(f"❌ Error dalam analisis N-gram: {str(e)}")

    # Tab 4: Insight Cerdas yang baru
    with tab4:
        st.markdown("### 🔍 Insight Cerdas & Rekomendasi")
        
        # Hitung metrics untuk insight
        pos_pct = metrics['pos_percentage']
        neg_pct = metrics['neg_percentage']
        
        # Alert berdasarkan sentiment ratio
        if pos_pct > 80:
            st.success(f"🎉 **Sentimen Sangat Positif!** ({pos_pct:.1f}% positif)")
            sentiment_status = "excellent"
        elif pos_pct > 60:
            st.info(f"✅ **Sentimen Positif** ({pos_pct:.1f}% positif)")
            sentiment_status = "good"
        elif pos_pct < 40:
            st.error(f"⚠️ **Perhatian: Sentimen Negatif Tinggi** ({neg_pct:.1f}% negatif)")
            sentiment_status = "poor"
        else:
            st.warning(f"⚖️ **Sentimen Campuran** ({pos_pct:.1f}% positif, {neg_pct:.1f}% negatif)")
            sentiment_status = "mixed"
            
        # Analisis kata kunci untuk insight
        pos_text = " ".join(topic_data[topic_data['sentiment'] == 'POSITIF']['teks_preprocessing'])
        neg_text = " ".join(topic_data[topic_data['sentiment'] == 'NEGATIF']['teks_preprocessing'])
        
        pos_terms = get_word_frequencies(pos_text, top_n=10) if pos_text.strip() else {}
        neg_terms = get_word_frequencies(neg_text, top_n=10) if neg_text.strip() else {}
        
        # Insight berdasarkan analisis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Insight Utama")
            insights = []
            
            # Insight berdasarkan sentiment distribution
            if sentiment_status == "excellent":
                insights.append("🌟 Pengguna sangat puas dengan layanan GoRide")
                insights.append("📈 Pertahankan kualitas layanan yang sudah ada")
            elif sentiment_status == "good":
                insights.append("👍 Mayoritas pengguna puas dengan layanan")
                insights.append("🔧 Ada ruang untuk perbaikan kecil")
            elif sentiment_status == "poor":
                insights.append("🚨 Perlu tindakan segera untuk perbaikan layanan")
                insights.append("📊 Fokus pada area yang paling banyak dikritik")
            else:
                insights.append("⚖️ Sentimen pengguna beragam")
                insights.append("🎯 Perlu analisis lebih detail per kategori")
            
            # Insight dari kata kunci positif
            if pos_terms:
                top_pos_words = list(pos_terms.keys())[:3]
                insights.append(f"✨ Aspek yang dipuji: {', '.join(top_pos_words)}")
            
            # Insight dari kata kunci negatif
            if neg_terms:
                top_neg_words = list(neg_terms.keys())[:3]
                insights.append(f"⚠️ Area perhatian: {', '.join(top_neg_words)}")
            
            # Tampilkan insights
            for insight in insights:
                st.write(f"• {insight}")
        
        with col2:
            st.markdown("#### 🔄 Rekomendasi Tindakan")
            recommendations = []
            
            if sentiment_status == "excellent":
                recommendations.extend([
                    "🏆 Dokumentasikan best practices saat ini",
                    "📢 Leverage ulasan positif untuk marketing",
                    "🔍 Monitor untuk mempertahankan standar"
                ])
            elif sentiment_status == "good":
                recommendations.extend([
                    "📊 Analisis feedback untuk improvement",
                    "🎯 Target 80%+ sentimen positif",
                    "👥 Training tim berdasarkan feedback"
                ])
            elif sentiment_status == "poor":
                recommendations.extend([
                    "🚨 Review segera proses operasional",
                    "📞 Follow up dengan pengguna yang komplain",
                    "🔧 Implementasi quick wins"
                ])
            else:
                recommendations.extend([
                    "📈 Segment analysis berdasarkan kategori",
                    "🎯 Prioritas perbaikan area kritikal",
                    "📝 Buat action plan spesifik"
                ])
            
            # Rekomendasi berdasarkan kata kunci negatif
            if neg_terms:
                neg_bigrams = get_ngrams(neg_text, 2, top_n=5)
                if neg_bigrams:
                    st.markdown("##### 🎯 Area Fokus Perbaikan:")
                    for bigram, freq in list(neg_bigrams.items())[:3]:
                        recommendations.append(f"🔧 Review masalah '{bigram}' ({freq}x disebutkan)")
            
            # Tampilkan rekomendasi
            for rec in recommendations:
                st.write(f"• {rec}")
        
        # Tren analysis untuk insight
        if 'sentiment_pivot' in locals() and len(sentiment_pivot) > 1:
            st.markdown("#### 📈 Analisis Tren")
            
            first_pct = sentiment_pivot.iloc[0]['positive_percentage']
            last_pct = sentiment_pivot.iloc[-1]['positive_percentage']
            trend_change = last_pct - first_pct
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if trend_change > 5:
                    st.success(f"📈 Tren Membaik (+{trend_change:.1f}%)")
                    trend_insight = "Strategi saat ini efektif, pertahankan!"
                elif trend_change < -5:
                    st.error(f"📉 Tren Menurun ({trend_change:.1f}%)")
                    trend_insight = "Perlu investigasi penyebab penurunan"
                else:
                    st.info(f"📊 Tren Stabil ({trend_change:+.1f}%)")
                    trend_insight = "Stabilitas baik, cari peluang peningkatan"
            
            with col2:
                volatility = sentiment_pivot['positive_percentage'].std()
                if volatility > 20:
                    st.warning(f"🌊 Volatilitas Tinggi ({volatility:.1f}%)")
                    vol_insight = "Konsistensi layanan perlu diperbaiki"
                else:
                    st.success(f"🎯 Volatilitas Rendah ({volatility:.1f}%)")
                    vol_insight = "Layanan cukup konsisten"
            
            with col3:
                avg_pct = sentiment_pivot['positive_percentage'].mean()
                if avg_pct > 70:
                    st.success(f"🏆 Rata-rata Baik ({avg_pct:.1f}%)")
                    avg_insight = "Performance di atas target"
                else:
                    st.warning(f"📊 Rata-rata ({avg_pct:.1f}%)")
                    avg_insight = "Masih di bawah target 70%"
            
            st.write(f"**Insight Tren:** {trend_insight}")
            st.write(f"**Insight Volatilitas:** {vol_insight}")
            st.write(f"**Insight Performance:** {avg_insight}")
        
        # Action plan generator
        st.markdown("#### 📋 Action Plan Generator")
        
        if st.button("🎯 Generate Action Plan", type="primary"):
            with st.spinner("🤖 Menganalisis data dan membuat action plan..."):
                action_plan = []
                
                # Priority berdasarkan sentiment
                if sentiment_status == "poor":
                    action_plan.append({
                        'priority': 'HIGH',
                        'action': 'Crisis Management',
                        'description': 'Segera tangani masalah utama yang menyebabkan sentiment negatif tinggi',
                        'timeline': '1-2 minggu'
                    })
                
                # Actions berdasarkan keyword analysis
                if neg_terms:
                    top_issues = list(neg_terms.keys())[:2]
                    for issue in top_issues:
                        action_plan.append({
                            'priority': 'HIGH' if sentiment_status == "poor" else 'MEDIUM',
                            'action': f'Improve {issue}',
                            'description': f'Analisis dan perbaiki masalah terkait "{issue}"',
                            'timeline': '2-4 minggu'
                        })
                
                # Positive reinforcement
                if pos_terms:
                    top_strength = list(pos_terms.keys())[0]
                    action_plan.append({
                        'priority': 'MEDIUM',
                        'action': f'Leverage {top_strength}',
                        'description': f'Maksimalkan kekuatan dalam "{top_strength}" untuk marketing',
                        'timeline': '1-3 minggu'
                    })
                
                # Monitoring action
                action_plan.append({
                    'priority': 'LOW',
                    'action': 'Setup Monitoring',
                    'description': 'Implementasi monitoring rutin sentiment analysis',
                    'timeline': 'Ongoing'
                })
                
                # Display action plan
                for i, action in enumerate(action_plan, 1):
                    priority_color = {
                        'HIGH': '🔴',
                        'MEDIUM': '🟡', 
                        'LOW': '🟢'                    }
                    
                    st.write(f"**{i}. {action['action']}** {priority_color[action['priority']]} {action['priority']}")
                    st.write(f"   📝 {action['description']}")
                    st.write(f"   ⏰ Timeline: {action['timeline']}")
                    st.write("")

    # Bagian Tabel Ulasan yang diperbaiki
    st.markdown("---")
    st.markdown("### 📋 Eksplorasi Data Ulasan")
    
    # Filter controls dengan layout yang lebih baik
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            search_term = st.text_input(
                "🔍 Cari dalam ulasan:", 
                placeholder="Ketik kata kunci...",
                help="Cari kata atau frasa dalam teks ulasan"
            )
        
        with col2:
            sentiment_filter = st.multiselect(
                "Filter sentimen:", 
                options=["POSITIF", "NEGATIF"], 
                default=["POSITIF", "NEGATIF"],
                help="Pilih sentimen yang ingin ditampilkan"
            )
        
        with col3:
            show_confidence = st.toggle(
                "🎯 Skor Model", 
                help="Tampilkan confidence score dari model"
            )
    
    # Sorting options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_option = st.selectbox(
            "📊 Urutkan berdasarkan:", 
            ["Terbaru", "Terlama", "Sentiment (Positif Dulu)", "Sentiment (Negatif Dulu)"],
            help="Pilih cara pengurutan data"
        )
    
    # Filter dan sort data
    filtered_display = topic_data.copy()
    
    if search_term:
        mask = (
            filtered_display['teks_preprocessing'].str.contains(search_term, case=False, na=False) |
            filtered_display['review_text'].str.contains(search_term, case=False, na=False)
        )
        filtered_display = filtered_display[mask]
        
    if sentiment_filter:
        filtered_display = filtered_display[filtered_display['sentiment'].isin(sentiment_filter)]
    
    # Sorting
    if sort_option == "Terbaru":
        filtered_display = filtered_display.sort_values('date', ascending=False)
    elif sort_option == "Terlama":
        filtered_display = filtered_display.sort_values('date', ascending=True)
    elif sort_option == "Sentiment (Positif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=False)
    elif sort_option == "Sentiment (Negatif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=True)
    
    # Add confidence scores if requested
    if show_confidence:
        with st.spinner("🤖 Menghitung confidence score..."):
            filtered_display = filtered_display.copy()
            # Batasi perhitungan untuk performa
            sample_size = min(100, len(filtered_display))
            if len(filtered_display) > sample_size:
                st.info(f"💡 Menampilkan confidence score untuk {sample_size} ulasan teratas")
                display_data = filtered_display.head(sample_size).copy()
            else:
                display_data = filtered_display.copy()
                
            display_data['confidence'] = display_data['review_text'].apply(
                lambda x: predict_sentiment(x, pipeline)['confidence']
            )
            filtered_display = display_data
    
    # Pagination
    if len(filtered_display) > 0:
        with col2:
            rows_per_page = st.selectbox(
                "📄 Baris per halaman:", 
                [10, 25, 50, 100], 
                index=1,
                help="Jumlah ulasan per halaman"
            )
        
        total_pages = max(1, len(filtered_display) // rows_per_page + (0 if len(filtered_display) % rows_per_page == 0 else 1))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.number_input(
                "Halaman:", 
                min_value=1, 
                max_value=total_pages, 
                value=1, 
                step=1
            )
        
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(filtered_display))
        paginated_data = filtered_display.iloc[start_idx:end_idx].copy()
        
        # Prepare data untuk display
        display_columns = ['date', 'review_text', 'sentiment']
        if show_confidence and 'confidence' in paginated_data.columns:
            display_columns.append('confidence')
        
        # Clean data untuk Arrow compatibility
        for col in paginated_data.columns:
            if paginated_data[col].dtype == 'object':
                paginated_data[col] = paginated_data[col].astype(str)
            elif pd.api.types.is_numeric_dtype(paginated_data[col]):
                paginated_data[col] = paginated_data[col].astype(str)
        
        # Display table dengan styling
        if show_confidence and 'confidence' in paginated_data.columns:
            # Style function untuk confidence
            def style_data(df):
                styled = df.style.format({'confidence': '{:.2f}'})
                return styled.map(
                    lambda val: 'background-color: #c6efce; color: #006100' if val == 'POSITIF' 
                    else 'background-color: #ffc7ce; color: #9c0006' if val == 'NEGATIF' 
                    else '', subset=['sentiment']
                )
            
            st.dataframe(
                style_data(paginated_data[display_columns]), 
                height=400,
                use_container_width=True
            )
        else:
            st.dataframe(
                paginated_data[display_columns], 
                height=400,
                use_container_width=True
            )
        
        # Info dan download
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Menampilkan {start_idx+1}-{end_idx} dari {len(filtered_display):,} ulasan (Halaman {current_page} dari {total_pages})")
        
        with col2:
            if len(filtered_display) > 0:
                csv_data = filtered_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv_data,
                    file_name=f"goride_reviews_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",                    mime="text/csv"
                )
    else:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.info("💡 Coba sesuaikan filter atau term pencarian Anda.")

    st.markdown("---")
    st.caption("© 2025 GoRide Sentiment Analysis App • Develop By Mhd Adreansyah")
    st.caption("Aplikasi ini merupakan Tugas Akhir/Skripsi dibawah perlindungan Hak Cipta")
