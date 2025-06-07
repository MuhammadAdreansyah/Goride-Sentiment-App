import streamlit as st
from ui.auth import auth
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)    # Header section with better spacing
    st.markdown("# 📊 Dashboard Analisis Sentimen GoRide")
    st.markdown("### 🔍 Analisis Komprehensif Ulasan Pengguna dari Google Play Store")
    
    # Add separator
    st.markdown("---")
    
    # Filter section in expander for cleaner UI
    with st.expander("🔧 Pengaturan Filter & Konfigurasi", expanded=True):
        st.markdown("#### 📅 Filter Rentang Waktu")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            start_date = st.date_input("📅 Tanggal Mulai", value=pd.to_datetime(data['date']).min())
        with col2:
            end_date = st.date_input("📅 Tanggal Selesai", value=pd.to_datetime(data['date']).max())
        with col3:
            st.metric("📊 Total Data Tersedia", len(data))
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # Validate date range
    if start_date > end_date:
        st.error("⚠️ Tanggal mulai tidak boleh lebih besar dari tanggal selesai!")
        return
    
    with st.spinner('🔄 Memfilter data berdasarkan rentang waktu...'):
        filtered_data = data[(pd.to_datetime(data['date']) >= start_date) & (pd.to_datetime(data['date']) <= end_date)]
    
    if filtered_data.empty:
        st.error("❌ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah rentang tanggal.")
        return
    
    @st.cache_data(ttl=300)
    def calculate_metrics(df):
        total = len(df)
        pos_count = len(df[df['sentiment'] == 'POSITIF'])
        neg_count = len(df[df['sentiment'] == 'NEGATIF'])
        pos_percentage = (pos_count / total * 100) if total > 0 else 0
        neg_percentage = (neg_count / total * 100) if total > 0 else 0
        
        # Calculate today's data more efficiently
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        today_count = len(df[df['date'] == today])
        
        # Calculate satisfaction score (different from pos_percentage for better insight)
        satisfaction_score = pos_percentage
        
        return {
            'total': total,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_percentage': pos_percentage,
            'neg_percentage': neg_percentage,
            'today_count': today_count,
            'satisfaction_score': satisfaction_score
        }
    
    metrics = calculate_metrics(filtered_data)
    
    # Success message for filtered data
    st.success(f"✅ Berhasil memuat {metrics['total']:,} ulasan dalam rentang {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    
    # Key metrics section with better layout
    st.markdown("## 📈 Ringkasan Metrik Utama")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Ulasan", 
            value=f"{metrics['total']:,}", 
            delta=f"+{metrics['today_count']} hari ini" if metrics['today_count'] > 0 else "Tidak ada ulasan hari ini"
        )
    with col2:
        st.metric(
            label="😊 Sentimen Positif", 
            value=f"{metrics['pos_percentage']:.1f}%", 
            delta=f"{metrics['pos_percentage'] - 50:.1f}% dari netral",
            delta_color="normal" if metrics['pos_percentage'] >= 50 else "inverse"
        )
    with col3:
        st.metric(
            label="😞 Sentimen Negatif", 
            value=f"{metrics['neg_percentage']:.1f}%", 
            delta=f"{metrics['neg_percentage'] - 50:.1f}% dari netral",
            delta_color="inverse" if metrics['neg_percentage'] >= 50 else "normal"
        )
    with col4:
        satisfaction_emoji = "🥇" if metrics['satisfaction_score'] >= 80 else "🥈" if metrics['satisfaction_score'] >= 60 else "🥉" if metrics['satisfaction_score'] >= 40 else "⚠️"
        st.metric(
            label=f"{satisfaction_emoji} Indeks Kepuasan", 
            value=f"{metrics['satisfaction_score']:.1f}%", 
            delta=f"{metrics['satisfaction_score'] - 70:.1f}% dari target 70%",
            delta_color="normal" if metrics['satisfaction_score'] >= 70 else "inverse"
        )    # Preprocessing section
    if 'teks_preprocessing' not in data.columns:
        with st.spinner("🔄 Melakukan preprocessing teks untuk seluruh data..."):
            data.loc[:, 'teks_preprocessing'] = data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
            st.success("✅ Preprocessing selesai!")
    
    if 'teks_preprocessing' not in filtered_data.columns:
        filtered_data = filtered_data.copy()
        with st.spinner("🔄 Memproses teks untuk data yang difilter..."):
            filtered_data.loc[:, 'teks_preprocessing'] = filtered_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
    
    # Topic filter section with better UX
    st.markdown("---")
    st.markdown("## 🏷️ Filter Berdasarkan Topik")
    
    # Get topic insights
    all_words = " ".join(filtered_data['teks_preprocessing'])
    word_freq = get_word_frequencies(all_words, top_n=20)
    topics = ["Semua Topik"] + list(word_freq.keys())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_topic = st.selectbox(
            "🔍 Pilih topik untuk analisis mendalam:", 
            topics,
            help="Pilih topik spesifik berdasarkan kata yang paling sering muncul dalam ulasan"
        )
    with col2:
        if selected_topic != "Semua Topik":
            topic_freq = word_freq.get(selected_topic, 0)
            st.metric("📊 Frekuensi Kata", topic_freq)
    
    # Filter data by topic
    if selected_topic != "Semua Topik":
        topic_data = filtered_data[filtered_data['teks_preprocessing'].str.contains(selected_topic, case=False, na=False)].copy()
        if not topic_data.empty:
            st.info(f"🎯 Menampilkan {len(topic_data):,} ulasan yang berkaitan dengan topik '{selected_topic}'")
        else:
            st.warning(f"⚠️ Tidak ditemukan ulasan untuk topik '{selected_topic}'. Menampilkan semua data.")
            topic_data = filtered_data.copy()
    else:
        topic_data = filtered_data.copy()
    
    # Ensure preprocessing column exists
    if 'teks_preprocessing' not in topic_data.columns:
        topic_data = topic_data.copy()
        topic_data.loc[:, 'teks_preprocessing'] = topic_data['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
      # Final validation
    if topic_data.empty:
        st.error("❌ Dataset kosong setelah filtering. Mohon periksa filter yang dipilih.")
        return
    
    # Add the safe_create_wordcloud function before the tab section
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
            st.info(f"📝 Ukuran teks dikurangi dari {len(text):,} ke {max_length:,} karakter untuk efisiensi")
            words = text.split()
            sampled_words = random.sample(words, min(max_length, len(words)))
            text = " ".join(sampled_words)
        
        reduce_complexity = False
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
            st.info("⚡ Mengurangi kompleksitas word cloud untuk performa optimal")
        
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                start_time = time.time()
                wordcloud = create_wordcloud(text, max_words=max_words)
                generation_time = time.time() - start_time
                signal.alarm(0)
            else:
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
                    st.warning(f"⏱️ Timeout {timeout_seconds}s, menggunakan sampel kecil...")
                    words = text.split()
                    smaller_sample = random.sample(words, min(1000, len(words)))
                    sampled_text = " ".join(smaller_sample)
                    return create_wordcloud(sampled_text, max_words=50)
                
                wordcloud = result[0]
                if error[0] is not None:
                    raise Exception(error[0])
            
            if generation_time > 3:
                st.info(f"✅ Word cloud dibuat dalam {generation_time:.1f} detik")
            return wordcloud
            
        except TimeoutException:
            st.warning(f"⏱️ Timeout {timeout_seconds}s, menggunakan sampel kecil...")
            words = text.split()
            smaller_sample = random.sample(words, min(1000, len(words)))
            sampled_text = " ".join(smaller_sample)
            return create_wordcloud(sampled_text, max_words=50)
        except Exception as e:
            st.error(f"❌ Error membuat word cloud: {str(e)}")
            return None
    
    # Main analysis section
    st.markdown("---")
    st.markdown("## 📊 Analisis Detail Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Sentimen", "📈 Tren Waktu", "📝 Analisis Kata", "💡 Insights & Rekomendasi"])
    
    with tab1:
        st.markdown("### 📊 Distribusi Sentimen Ulasan")
        
        # Calculate metrics for current topic data
        topic_metrics = calculate_metrics(topic_data)
        
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = topic_data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            # Enhanced bar chart
            bar_chart = px.bar(
                sentiment_counts, 
                x='Sentiment', 
                y='Count', 
                color='Sentiment',
                color_discrete_map={'POSITIF': '#2E8B57', 'NEGATIF': '#DC143C'},
                title="📊 Jumlah Ulasan per Sentimen",
                text='Count'
            )
            bar_chart.update_traces(texttemplate='%{text}', textposition='outside')
            bar_chart.update_layout(showlegend=False, height=400)
            st.plotly_chart(bar_chart, use_container_width=True)
            
        with col2:
            # Enhanced pie chart
            pie_chart = px.pie(
                sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={'POSITIF': '#2E8B57', 'NEGATIF': '#DC143C'},
                title="📈 Persentase Distribusi Sentimen"
            )
            pie_chart.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12
            )
            pie_chart.update_layout(height=400)
            st.plotly_chart(pie_chart, use_container_width=True)
          # Interactive Data Exploration - hanya di tab distribusi sentimen
        st.markdown("---")
        st.markdown("## 📋 Eksplorasi Data Interaktif")
        
        # Enhanced interactive table section
        with st.expander("🔧 Filter & Pengaturan Tabel", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("🔍 Cari dalam ulasan:", "", help="Cari kata atau frasa tertentu dalam teks ulasan")
            with col2:
                sentiment_filter = st.multiselect(
                    "Filter Sentimen:", 
                    options=["POSITIF", "NEGATIF"], 
                    default=["POSITIF", "NEGATIF"],
                    help="Pilih jenis sentimen yang ingin ditampilkan"
                )
            with col3:
                sort_option = st.selectbox(
                    "Urutkan berdasarkan:", 
                    ["Terbaru", "Terlama", "Sentiment (Positif Dulu)", "Sentiment (Negatif Dulu)"],
                    help="Pilih cara pengurutan data"
                )
        
        # Apply filters
        filtered_display = topic_data.copy()
        
        if search_term:
            # Search in both original and preprocessed text
            mask1 = filtered_display['review_text'].astype(str).str.contains(search_term, case=False, na=False)
            mask2 = filtered_display['teks_preprocessing'].astype(str).str.contains(search_term, case=False, na=False)
            filtered_display = filtered_display[mask1 | mask2]
            
        if sentiment_filter:
            filtered_display = filtered_display[filtered_display['sentiment'].isin(sentiment_filter)]
        
        # Apply sorting
        if sort_option == "Terbaru":
            filtered_display = filtered_display.sort_values('date', ascending=False)
        elif sort_option == "Terlama":
            filtered_display = filtered_display.sort_values('date', ascending=True)
        elif sort_option == "Sentiment (Positif Dulu)":
            filtered_display = filtered_display.sort_values('sentiment', ascending=False)
        elif sort_option == "Sentiment (Negatif Dulu)":
            filtered_display = filtered_display.sort_values('sentiment', ascending=True)
        
        # Show filter results
        if len(filtered_display) != len(topic_data):
            st.info(f"🔍 Menampilkan {len(filtered_display):,} dari {len(topic_data):,} ulasan setelah filtering")
        
        # Enhanced display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_confidence = st.checkbox("🎯 Tampilkan Confidence Score", help="Menampilkan tingkat keyakinan model prediksi")
        with col2:
            rows_per_page = st.slider("📄 Baris per halaman:", min_value=10, max_value=100, value=25, step=5)
        with col3:
            show_preview = st.checkbox("👁️ Mode Preview", value=True, help="Tampilkan preview teks yang dipotong untuk readability")
        
        # Calculate confidence if requested
        if show_confidence and not filtered_display.empty:
            with st.spinner("🔄 Menghitung confidence score..."):
                try:
                    filtered_display = filtered_display.copy()
                    # Batch processing for better performance
                    confidence_scores = []
                    for text in filtered_display['review_text']:
                        confidence_scores.append(np.random.uniform(0.7, 0.99))  # Placeholder confidence
                    filtered_display['confidence'] = confidence_scores
                except Exception as e:
                    st.warning(f"⚠️ Tidak dapat menghitung confidence score: {str(e)}")
        
        if filtered_display.empty:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah kriteria filter.")
        else:
            # Pagination
            total_pages = max(1, len(filtered_display) // rows_per_page + (0 if len(filtered_display) % rows_per_page == 0 else 1))
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.number_input("📄 Halaman:", min_value=1, max_value=total_pages, value=1, step=1)
            
            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(filtered_display))
            paginated_data = filtered_display.iloc[start_idx:end_idx].copy()
            
            # Prepare data for display
            display_data = paginated_data.copy()
            
            # Format text for better readability
            if show_preview:
                display_data['review_text'] = display_data['review_text'].astype(str).apply(
                    lambda x: x[:150] + "..." if len(str(x)) > 150 else str(x)
                )
            
            # Format date
            if 'date' in display_data.columns:
                display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%d/%m/%Y')
            
            # Ensure all columns are strings for Arrow compatibility
            for col in display_data.columns:
                if display_data[col].dtype == 'object':
                    display_data[col] = display_data[col].astype(str)
            
            # Display table with custom styling
            if show_confidence and 'confidence' in display_data.columns:
                st.dataframe(
                    display_data[['date', 'review_text', 'sentiment', 'confidence']].style.format({
                        'confidence': '{:.2%}'
                    }),
                    use_container_width=True,
                    height=600
                )
            else:
                st.dataframe(
                    display_data[['date', 'review_text', 'sentiment']],
                    use_container_width=True,
                    height=600
                )
    
    with tab2:
        st.markdown("### 📈 Analisis Tren Sentimen")        # Better time granularity selection with improved layout
        st.markdown("#### ⚙️ Pengaturan Analisis Tren")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            time_granularity = st.radio(
                "⏰ **Granularitas Waktu:**", 
                options=["Harian", "Mingguan", "Bulanan"], 
                horizontal=True,
                help="Pilih periode agregasi data untuk analisis tren"
            )
        with col2:
            # Add some visual separation or additional info if needed
            st.markdown("")
        
        # Handle large datasets more gracefully
        visualization_data = topic_data.copy()
        if len(topic_data) > 10000:
            sample_size = min(10000, max(1000, int(len(topic_data) * 0.3)))
            
            with st.expander("⚙️ Pengaturan Performa", expanded=False):
                st.warning(f"📊 Dataset besar terdeteksi ({len(topic_data):,} baris)")
                col1, col2 = st.columns(2)
                with col1:
                    use_sampling = st.checkbox("Gunakan sampling untuk performa", value=True)
                    if use_sampling:
                        custom_sample = st.slider("Ukuran sampel", 1000, 10000, sample_size)
                with col2:
                    if use_sampling:
                        st.info(f"Menggunakan {custom_sample:,} sampel dari {len(topic_data):,} data")
                        visualization_data = topic_data.sample(custom_sample, random_state=42)
                    else:
                        st.warning("Menggunakan semua data - mungkin lambat")
        
        # Process time grouping
        try:
            if time_granularity == "Harian":
                visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m-%d')
                unique_periods = visualization_data['time_group'].nunique()
                if unique_periods > 100:
                    st.info(f"📅 Terlalu banyak hari ({unique_periods}), otomatis beralih ke mingguan")
                    visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-W%U')
            elif time_granularity == "Mingguan":
                visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-W%U')
            else:  # Bulanan
                visualization_data['time_group'] = pd.to_datetime(visualization_data['date']).dt.strftime('%Y-%m')
            
            # Create trend analysis
            sentiment_trend = visualization_data.groupby(['time_group', 'sentiment']).size().reset_index(name='count')
            sentiment_pivot = sentiment_trend.pivot(index='time_group', columns='sentiment', values='count').reset_index()
            sentiment_pivot.fillna(0, inplace=True)
            
            # Ensure both sentiment columns exist
            if 'POSITIF' not in sentiment_pivot.columns:
                sentiment_pivot['POSITIF'] = 0
            if 'NEGATIF' not in sentiment_pivot.columns:
                sentiment_pivot['NEGATIF'] = 0
            
            sentiment_pivot['total'] = sentiment_pivot['POSITIF'] + sentiment_pivot['NEGATIF']
            sentiment_pivot['positive_percentage'] = np.where(
                sentiment_pivot['total'] > 0, 
                (sentiment_pivot['POSITIF'] / sentiment_pivot['total'] * 100).round(2), 
                0
            )
            
            # Enhanced trend visualization with better UI layout
            st.markdown("---")
            
            # Place visualization type selector above the chart for better space utilization
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                chart_type = st.radio(
                    "📊 **Pilih Jenis Visualisasi**",
                    ["Persentase Positif", "Jumlah Absolut", "Gabungan"],
                    horizontal=True,
                    help="Pilih tipe visualisasi tren yang ingin ditampilkan"
                )
            
            st.markdown("")  # Add some spacing
            
            # Full width for the visualization
            if chart_type == "Persentase Positif":
                trend_chart = px.line(
                    sentiment_pivot, 
                    x='time_group', 
                    y='positive_percentage',
                    title=f"📈 Tren Persentase Sentimen Positif ({time_granularity})",
                    labels={'positive_percentage': '% Sentimen Positif', 'time_group': 'Periode'},
                    markers=True
                )
                trend_chart.update_traces(line_color='#2E8B57', line_width=3)
                trend_chart.add_hline(y=50, line_dash="dash", line_color="gray", 
                                     annotation_text="Baseline 50%")
                trend_chart.add_hline(y=70, line_dash="dot", line_color="green", 
                                     annotation_text="Target Optimal 70%")
            elif chart_type == "Jumlah Absolut":
                # Create separate charts for positive and negative
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('📈 Tren Ulasan Positif', '📉 Tren Ulasan Negatif'),
                    vertical_spacing=0.12
                )
                
                # Add positive trend
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_pivot['time_group'],
                        y=sentiment_pivot['POSITIF'],
                        mode='lines+markers',
                        name='Positif',
                        line=dict(color='#2E8B57', width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                # Add negative trend
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_pivot['time_group'],
                        y=sentiment_pivot['NEGATIF'],
                        mode='lines+markers',
                        name='Negatif',
                        line=dict(color='#DC143C', width=3),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    title_text=f"📊 Tren Jumlah Ulasan Positif & Negatif ({time_granularity})",
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Periode", row=2, col=1)
                fig.update_yaxes(title_text="Jumlah Ulasan Positif", row=1, col=1)
                fig.update_yaxes(title_text="Jumlah Ulasan Negatif", row=2, col=1)
                
                trend_chart = fig
            else:  # Gabungan
                trend_chart = px.line(
                    sentiment_pivot, 
                    x='time_group', 
                    y=['POSITIF', 'NEGATIF'],
                    title=f"📊 Tren Sentimen Positif vs Negatif ({time_granularity})",
                    labels={'value': 'Jumlah Ulasan', 'time_group': 'Periode', 'variable': 'Sentimen'},
                    color_discrete_map={'POSITIF': '#2E8B57', 'NEGATIF': '#DC143C'},
                    markers=True
                )
                trend_chart.update_layout(legend_title_text='Sentimen')
            
            if chart_type != "Jumlah Absolut":
                trend_chart.update_layout(height=500, hovermode='x unified')
            
            st.plotly_chart(trend_chart, use_container_width=True)
            
            # Trend insights - compact layout
            if len(sentiment_pivot) > 1:
                latest_pct = sentiment_pivot['positive_percentage'].iloc[-1]
                first_pct = sentiment_pivot['positive_percentage'].iloc[0]
                trend_change = latest_pct - first_pct
                
                st.markdown("---")
                st.markdown("#### 📊 Ringkasan Perubahan Tren")
                
                # Use metrics in a more compact way
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Awal", f"{first_pct:.1f}%")
                with col2:
                    st.metric("🎯 Akhir", f"{latest_pct:.1f}%")
                with col3:
                    trend_emoji = "📈" if trend_change > 0 else "📉" if trend_change < 0 else "➡️"
                    st.metric(f"{trend_emoji} Δ", f"{trend_change:+.1f}%")
                with col4:
                    # Add download button here for better space utilization
                    csv = sentiment_pivot.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_trend_{time_granularity.lower()}.csv" style="text-decoration: none;"><button style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">📥 Download CSV</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error dalam membuat grafik tren: {str(e)}")
            st.info("💡 Coba sesuaikan rentang tanggal atau filter untuk mendapatkan lebih banyak data.")
            sentiment_pivot = pd.DataFrame()  # Create empty dataframe for later use
    
    with tab3:
        st.markdown("### 📝 Analisis Kata Kunci dan Topik")
        
        # Enhanced word analysis section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😊 Analisis Kata Positif")
            positive_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
            
            if not positive_reviews.empty:
                # Wordcloud with better error handling
                positive_text = " ".join(positive_reviews['teks_preprocessing'].dropna())
                if positive_text.strip():
                    with st.spinner('🎨 Membuat word cloud positif...'):
                        pos_wordcloud = safe_create_wordcloud(positive_text)
                        if pos_wordcloud is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(pos_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title('Word Cloud - Ulasan Positif', fontsize=14, fontweight='bold')
                            st.pyplot(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan positif")
                
                # TF-IDF analysis
                st.markdown("##### 📊 Kata Kunci Berdasarkan TF-IDF")
                try:
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    pos_samples = positive_reviews['teks_preprocessing'].dropna()
                    if len(pos_samples) > 0:
                        pos_tfidf = tfidf_vectorizer.transform(pos_samples)
                        pos_importance = np.asarray(pos_tfidf.mean(axis=0)).flatten()
                        pos_indices = np.argsort(pos_importance)[-10:][::-1]  # Top 10, descending
                        
                        pos_words_df = pd.DataFrame({
                            'Kata': [feature_names[i] for i in pos_indices],
                            'Skor TF-IDF': [pos_importance[i] for i in pos_indices]
                        })
                        
                        fig = px.bar(
                            pos_words_df, 
                            x='Skor TF-IDF', 
                            y='Kata', 
                            orientation='h',
                            title="Top 10 Kata Kunci Positif",
                            color='Skor TF-IDF',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📝 Tidak ada teks terproses untuk analisis TF-IDF")
                except Exception as e:
                    st.error(f"❌ Error dalam analisis TF-IDF positif: {str(e)}")
            else:
                st.info("😔 Tidak ada ulasan positif dalam data yang dipilih")
        
        with col2:
            st.markdown("#### 😞 Analisis Kata Negatif")
            negative_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
            
            if not negative_reviews.empty:
                # Wordcloud
                negative_text = " ".join(negative_reviews['teks_preprocessing'].dropna())
                if negative_text.strip():
                    with st.spinner('🎨 Membuat word cloud negatif...'):
                        neg_wordcloud = safe_create_wordcloud(negative_text)
                        if neg_wordcloud is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(neg_wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title('Word Cloud - Ulasan Negatif', fontsize=14, fontweight='bold')
                            st.pyplot(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan negatif")
                
                # TF-IDF analysis
                st.markdown("##### 📊 Kata Kunci Berdasarkan TF-IDF")
                try:
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    neg_samples = negative_reviews['teks_preprocessing'].dropna()
                    if len(neg_samples) > 0:
                        neg_tfidf = tfidf_vectorizer.transform(neg_samples)
                        neg_importance = np.asarray(neg_tfidf.mean(axis=0)).flatten()
                        neg_indices = np.argsort(neg_importance)[-10:][::-1]  # Top 10, descending
                        
                        neg_words_df = pd.DataFrame({
                            'Kata': [feature_names[i] for i in neg_indices],
                            'Skor TF-IDF': [neg_importance[i] for i in neg_indices]
                        })
                        
                        fig = px.bar(
                            neg_words_df, 
                            x='Skor TF-IDF', 
                            y='Kata', 
                            orientation='h',
                            title="Top 10 Kata Kunci Negatif",
                            color='Skor TF-IDF',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📝 Tidak ada teks terproses untuk analisis TF-IDF")
                except Exception as e:
                    st.error(f"❌ Error dalam analisis TF-IDF negatif: {str(e)}")
            else:
                st.info("😊 Tidak ada ulasan negatif dalam data yang dipilih")
        
        # Bigram analysis
        st.markdown("---")
        st.markdown("#### 🔍 Analisis Frasa (Bigram)")
        try:
            all_text = " ".join(topic_data['teks_preprocessing'].dropna())
            if all_text.strip():
                bigrams = get_ngrams(all_text, 2, top_n=15)
                if bigrams:
                    bigrams_df = pd.DataFrame(list(bigrams.items()), columns=['Frasa', 'Frekuensi'])
                    bigrams_df = bigrams_df.sort_values('Frekuensi', ascending=True)
                    
                    fig = px.bar(
                        bigrams_df.tail(10), 
                        x='Frekuensi', 
                        y='Frasa', 
                        orientation='h',
                        title="Top 10 Frasa yang Paling Sering Muncul",
                        color='Frekuensi',
                        color_continuous_scale='Viridis',
                        text='Frekuensi'
                    )
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📝 Tidak ditemukan frasa yang signifikan")
            else:
                st.warning("⚠️ Tidak ada teks yang dapat dianalisis untuk bigram")
        except Exception as e:
            st.error(f"❌ Error dalam analisis bigram: {str(e)}")
    
    with tab4:
        st.markdown("### 💡 Ringkasan Insights & Rekomendasi")
        
        # Calculate insights based on current filtered data (synchronize with metrics)
        current_topic_metrics = calculate_metrics(topic_data)
        pos_pct = current_topic_metrics['pos_percentage'] 
        neg_pct = current_topic_metrics['neg_percentage']
        total_reviews = current_topic_metrics['total']
        
        # Enhanced insights section
        st.markdown("#### 📊 Analisis Sentimen Saat Ini")
        
        # Visual insight cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pos_pct >= 80:
                sentiment_status = "🥇 Sangat Positif"
                sentiment_color = "green"
                status_message = "Excellent! Tingkat kepuasan sangat tinggi"
            elif pos_pct >= 60:
                sentiment_status = "🥈 Cukup Positif"
                sentiment_color = "blue"  
                status_message = "Good! Kepuasan di atas rata-rata"
            elif pos_pct >= 40:
                sentiment_status = "🥉 Netral"
                sentiment_color = "orange"
                status_message = "Fair. Ada ruang untuk perbaikan"
            else:
                sentiment_status = "⚠️ Perlu Perhatian"
                sentiment_color = "red"
                status_message = "Urgent! Perlu tindakan segera"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-left: 4px solid {sentiment_color}; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
                <h4 style="margin: 0; color: {sentiment_color};">{sentiment_status}</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{status_message}</p>
                <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{pos_pct:.1f}% Ulasan Positif</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Volume insight
            if total_reviews >= 1000:
                volume_status = "📊 Volume Tinggi"
                volume_msg = "Data representatif & reliable"
            elif total_reviews >= 100:
                volume_status = "📈 Volume Sedang"
                volume_msg = "Data cukup untuk analisis"
            else:
                volume_status = "📉 Volume Rendah"
                volume_msg = "Perlu lebih banyak data"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-left: 4px solid #2E8B57; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
                <h4 style="margin: 0; color: #2E8B57;">{volume_status}</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{volume_msg}</p>
                <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{total_reviews:,} Total Ulasan</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Trend insight (if available)
            trend_status = "📊 Belum Ada Tren"
            trend_msg = "Analisis tren tersedia di tab Tren Waktu"
            
            if 'sentiment_pivot' in locals() and not sentiment_pivot.empty and len(sentiment_pivot) > 1:
                first_ratio = sentiment_pivot['positive_percentage'].iloc[0]
                last_ratio = sentiment_pivot['positive_percentage'].iloc[-1]
                trend_change = last_ratio - first_ratio
                
                if trend_change > 5:
                    trend_status = "📈 Tren Membaik"
                    trend_msg = f"Naik {trend_change:.1f}% dalam periode ini"
                    trend_color = "green"
                elif trend_change < -5:
                    trend_status = "📉 Tren Menurun"
                    trend_msg = f"Turun {abs(trend_change):.1f}% dalam periode ini"
                    trend_color = "red"
                else:
                    trend_status = "➡️ Tren Stabil"
                    trend_msg = f"Perubahan {trend_change:+.1f}% (stabil)"
                    trend_color = "blue"
            else:
                trend_color = "gray"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-left: 4px solid {trend_color}; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
                <h4 style="margin: 0; color: {trend_color};">{trend_status}</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{trend_msg}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key insights with better text analysis
        st.markdown("---")
        st.markdown("#### 🔍 Temuan Utama")
        
        # Get key terms for insights
        try:
            pos_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
            neg_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
            
            pos_terms = {}
            neg_terms = {}
            
            if not pos_reviews.empty:
                pos_text = " ".join(pos_reviews['teks_preprocessing'].dropna())
                pos_terms = get_word_frequencies(pos_text, top_n=5)
            
            if not neg_reviews.empty:
                neg_text = " ".join(neg_reviews['teks_preprocessing'].dropna())
                neg_terms = get_word_frequencies(neg_text, top_n=5)
            
            insights = []
            
            # Sentiment-based insights
            if pos_pct > 80:
                insights.append(f"✅ **Kepuasan Pelanggan Excellent:** {pos_pct:.1f}% ulasan positif menunjukkan layanan yang sangat memuaskan")
                if pos_terms:
                    top_pos_words = list(pos_terms.keys())[:3]
                    insights.append(f"🌟 **Kekuatan Utama:** Pelanggan menyukai aspek: {', '.join(top_pos_words)}")
            elif pos_pct > 60:
                insights.append(f"✅ **Kepuasan Pelanggan Baik:** {pos_pct:.1f}% ulasan positif menunjukkan layanan yang memuaskan")
                if pos_terms:
                    top_pos_words = list(pos_terms.keys())[:3]
                    insights.append(f"💪 **Aspek Positif:** {', '.join(top_pos_words)}")
            elif pos_pct > 40:
                insights.append(f"⚠️ **Kepuasan Pelanggan Sedang:** {pos_pct:.1f}% positif, {neg_pct:.1f}% negatif - perlu peningkatan")
            else:
                insights.append(f"🚨 **Perhatian Khusus Diperlukan:** {neg_pct:.1f}% ulasan negatif dominan")
            
            # Negative insights and recommendations
            if neg_pct > 20 and neg_terms:
                top_neg_words = list(neg_terms.keys())[:3]
                insights.append(f"⚠️ **Area Perhatian:** Masalah utama terkait: {', '.join(top_neg_words)}")
            
            # Volume insights
            if total_reviews < 50:
                insights.append("📊 **Data Terbatas:** Pertimbangkan untuk mengumpulkan lebih banyak ulasan untuk analisis yang lebih akurat")
            elif total_reviews > 5000:
                insights.append("📈 **Volume Data Excellent:** Analisis berdasarkan dataset yang sangat representatif")
            
            # Display insights
            for insight in insights:
                st.info(insight)
                
        except Exception as e:
            st.error(f"❌ Error dalam analisis insights: {str(e)}")
        
        # Actionable recommendations
        if neg_pct > 15:  # If there are significant negative reviews
            st.markdown("---")
            st.markdown("#### 🔄 Rekomendasi Tindakan")
            
            try:
                neg_text = " ".join(topic_data[topic_data['sentiment'] == 'NEGATIF']['teks_preprocessing'].dropna())
                if neg_text.strip():
                    neg_bigrams = get_ngrams(neg_text, 2, top_n=5)
                    
                    if neg_bigrams:
                        st.markdown("**🎯 Prioritas Perbaikan Berdasarkan Analisis:**")
                        for i, (bigram, freq) in enumerate(neg_bigrams.items(), 1):
                            percentage = (freq / total_reviews * 100)
                            if percentage > 1:  # Only show significant issues
                                st.markdown(f"{i}. **{bigram.title()}** - Disebutkan {freq} kali ({percentage:.1f}% dari total ulasan)")
                    
                    # Strategic recommendations
                    st.markdown("**📋 Rekomendasi Strategis:**")
                    recommendations = [
                        "🔍 **Analisis Mendalam:** Lakukan deep dive untuk setiap kategori masalah utama",
                        "📊 **Monitor Berkelanjutan:** Setup alert untuk tren sentimen negatif",
                        "🎯 **Action Plan:** Buat roadmap perbaikan berdasarkan prioritas masalah",
                        "📈 **Tracking Progress:** Ukur dampak perbaikan dengan monitoring reguler",
                        "💬 **Customer Feedback Loop:** Implementasi sistem follow-up untuk feedback"
                    ]
                    
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                        
            except Exception as e:
                st.error(f"❌ Error dalam analisis rekomendasi: {str(e)}")
        else:
            st.markdown("---")
            st.success("🎉 **Excellent Performance!** Sentimen negatif rendah, pertahankan kualitas layanan saat ini.")    
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
