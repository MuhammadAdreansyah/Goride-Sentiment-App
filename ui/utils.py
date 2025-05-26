"""
Fungsi utilitas dan resource untuk aplikasi GoRide Sentiment Analysis.
Berisi: preprocessing, analisis kata, load data, model, dsb.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import time
import signal
from datetime import datetime
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import io
import base64
from wordcloud import WordCloud
import networkx as nx
import os
from pathlib import Path
import traceback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import joblib

# Download NLTK resources jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Definisikan direktori dasar project
BASE_DIR = Path(__file__).parent.parent
DICTIONARY_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "data"

# Inisialisasi stemmer dan stopword untuk bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
stopword_list = set(stop_factory.get_stop_words())

# Load resources
slang_path = DICTIONARY_DIR / "kamus_slang_formal.txt"
stopwords_path = DICTIONARY_DIR / "stopwordsID.txt"

if not os.path.exists(slang_path):
    slang_path = BASE_DIR / "kamus_slang_formal.txt"
try:
    slang_dict = dict(line.strip().split(':') for line in open(slang_path, encoding='utf-8'))
except FileNotFoundError:
    slang_dict = {}

if not os.path.exists(stopwords_path):
    stopwords_path = BASE_DIR / "stopwordsID.txt"
try:
    custom_stopwords = set(open(stopwords_path, encoding='utf-8').read().splitlines())
    stopword_list.update(custom_stopwords)
except FileNotFoundError:
    pass

def preprocess_text(text, options=None):
    import re
    # Validasi input
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    if not text.strip():
        return text
    if options is None:
        options = {
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
    try:
        tokens = text
        # Lowercase
        if options.get('lowercase', False):
            tokens = tokens.lower()
        # Cleansing: hapus emoji, url, mention, hashtag, dsb
        if options.get('clean_text', False):
            # Hapus URL
            tokens = re.sub(r'http\S+|www\S+', '', tokens)
            # Hapus mention dan hashtag
            tokens = re.sub(r'@[\w_]+|#[\w_]+', '', tokens)
            # Hapus emoji dan karakter non-alfabet
            tokens = re.sub(r'[^\w\s.,!?-]', '', tokens)
        # Normalisasi slang
        if options.get('normalize_slang', False) and slang_dict:
            def normalize_word(word):
                return slang_dict.get(word, word)
            tokens = ' '.join([normalize_word(w) for w in tokens.split()])
        # Hapus karakter berulang (contoh: "baguuuusss" -> "bagus")
        if options.get('remove_repeated', False):
            tokens = re.sub(r'(\w)\1{2,}', r'\1', tokens)
        # Hapus tanda baca
        if options.get('remove_punctuation', False):
            tokens = re.sub(r'[\.,!?;:\-\"\'\(\)\[\]\/]', ' ', tokens)
        # Hapus angka
        if options.get('remove_numbers', False):
            tokens = re.sub(r'\d+', '', tokens)
        # Tokenisasi
        if options.get('tokenize', False):
            tokens = word_tokenize(tokens)
        else:
            tokens = [tokens] if isinstance(tokens, str) else tokens
        # Hapus stopwords
        if options.get('remove_stopwords', False) and isinstance(tokens, list):
            tokens = [w for w in tokens if w not in stopword_list and len(w) > 1]
        # Stemming
        if options.get('stemming', False) and isinstance(tokens, list):
            tokens = [stemmer.stem(w) for w in tokens]
        # Gabungkan kembali jika rejoin True
        if options.get('rejoin', True) and isinstance(tokens, list):
            return ' '.join(tokens)
        return tokens if isinstance(tokens, list) else str(text)
    except Exception as e:
        import streamlit as st
        import traceback
        st.error(f"Error dalam preprocessing: {str(e)}")
        st.error(traceback.format_exc())
        return text

def get_word_frequencies(text, top_n=10):
    try:
        words = nltk.word_tokenize(text) if isinstance(text, str) else text
        word_freq = Counter(words)
        return dict(word_freq.most_common(top_n))
    except Exception as e:
        st.error(f"Error in word frequency analysis: {str(e)}")
        return {}

def get_ngrams(text, n, top_n=10):
    try:
        words = nltk.word_tokenize(text) if isinstance(text, str) else text
        n_grams = list(ngrams(words, n))
        n_gram_freq = Counter([' '.join(g) for g in n_grams])
        return dict(n_gram_freq.most_common(top_n))
    except Exception as e:
        st.error(f"Error in n-gram analysis: {str(e)}")
        return {}

def create_wordcloud(text, max_words=100, background_color='white'):
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text if isinstance(text, str) else ' '.join(text))
        return wordcloud
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def prepare_and_load_preprocessed_data(max_rows=None, chunksize=10000, preprocessing_options=None):
    """
    Load data preprocessed jika sudah ada, jika belum lakukan batch preprocessing dan simpan ke file preprocessed.
    """
    preprocessed_path = DATA_DIR / "ulasan_goride_preprocessed.csv"
    raw_path = DATA_DIR / "ulasan_goride.csv"
    if preprocessing_options is None:
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
    # Jika file preprocessed sudah ada, langsung load
    if os.path.exists(preprocessed_path):
        try:
            df = pd.read_csv(preprocessed_path, nrows=max_rows)
            # Pastikan mapping label tetap benar jika file lama
            label_map = {
                'Positive': 'POSITIF', 'POSITIVE': 'POSITIF',
                'Negative': 'NEGATIF', 'NEGATIVE': 'NEGATIF',
                'Netral': 'NETRAL', 'Neutral': 'NETRAL', 'NETRAL': 'NETRAL', 'NEUTRAL': 'NETRAL'
            }
            df['sentiment'] = df['sentiment'].replace(label_map)
            df = df[df['sentiment'].isin(['POSITIF', 'NEGATIF'])]
            # Pastikan kolom wajib tetap ada
            required_columns = ['review_text', 'sentiment', 'date', 'teks_preprocessing']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Kolom {col} tidak ditemukan di file preprocessed!")
                    return pd.DataFrame(columns=required_columns)
            return df
        except Exception as e:
            st.error(f"Gagal membaca file preprocessed: {str(e)}")
            # Jika gagal, hapus file preprocessed agar bisa regenerate
            try:
                os.remove(preprocessed_path)
            except Exception:
                pass
    # Jika belum ada, lakukan batch preprocessing dan simpan
    if not os.path.exists(raw_path):
        st.error("File ulasan_goride.csv tidak ditemukan!")
        return pd.DataFrame(columns=['review_text', 'sentiment', 'date', 'teks_preprocessing'])
    try:
        df = pd.read_csv(raw_path, nrows=max_rows)
        # Validasi kolom
        required_columns = ['review_text', 'sentiment', 'date']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom {col} tidak ditemukan di file CSV!")
                return pd.DataFrame(columns=required_columns+['teks_preprocessing'])
        # Mapping label sebelum preprocessing
        label_map = {
            'Positive': 'POSITIF', 'POSITIVE': 'POSITIF',
            'Negative': 'NEGATIF', 'NEGATIVE': 'NEGATIF',
            'Netral': 'NETRAL', 'Neutral': 'NETRAL', 'NETRAL': 'NETRAL', 'NEUTRAL': 'NETRAL'
        }
        df['sentiment'] = df['sentiment'].replace(label_map)
        df = df[df['sentiment'].isin(['POSITIF', 'NEGATIF'])]
        # Preprocessing batch
        with st.spinner("Melakukan batch preprocessing dan menyimpan hasil..."):
            df['teks_preprocessing'] = df['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
            # Simpan ke file preprocessed
            df.to_csv(preprocessed_path, index=False)
        return df
    except Exception as e:
        st.error(f"Gagal melakukan preprocessing batch: {str(e)}")
        return pd.DataFrame(columns=['review_text', 'sentiment', 'date', 'teks_preprocessing'])

# Ganti load_sample_data agar hanya wrapper ke prepare_and_load_preprocessed_data
@st.cache_data(ttl=1800)
def load_sample_data(max_rows=None, chunksize=10000):
    return prepare_and_load_preprocessed_data(max_rows=max_rows, chunksize=chunksize)

@st.cache_resource(ttl=3600)
def train_model(data, preprocessing_options=None, batch_size=1000):
    if preprocessing_options is None:
        preprocessing_options = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': True,
            'clean_text': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_texts = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch = data.iloc[i:batch_end]
            batch_num = i // batch_size + 1
            status_text.text(f"Preprocessing batch {batch_num}/{total_batches}...")
            progress_bar.progress(i / len(data))
            batch_processed = []
            for text in batch['review_text']:
                try:
                    processed = preprocess_text(text, preprocessing_options)
                    batch_processed.append(processed)
                except Exception as e:
                    batch_processed.append(text)
            processed_texts.extend(batch_processed)
        status_text.text("Vectorizing text data...")
        progress_bar.progress(1.0)
        tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            lowercase=False,
            strip_accents='unicode',
            norm='l2',
            sublinear_tf=True,
        )
        status_text.text("Transforming text to TF-IDF features...")
        X = tfidf.fit_transform(processed_texts)
        y = data['sentiment']
        status_text.empty()
        progress_bar.empty()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        svm = SVC(
            C=10,
            kernel='linear',
            gamma='scale',
            probability=True,
            class_weight='balanced'
        )
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="POSITIF")
        recall = recall_score(y_test, y_pred, pos_label="POSITIF")
        f1 = f1_score(y_test, y_pred, pos_label="POSITIF")
        cm = confusion_matrix(y_test, y_pred)
        pipeline = Pipeline([
            ('vectorizer', tfidf),
            ('classifier', svm)
        ])
        pipeline.fit(processed_texts, y)
        return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, 0, 0, 0, 0, None, None, None, None, None

def save_model_and_vectorizer(pipeline, tfidf, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    joblib.dump(pipeline.named_steps['classifier'], model_path)
    joblib.dump(tfidf, vectorizer_path)
    return model_path, vectorizer_path

def load_saved_model(model_dir="models"):
    model_path = os.path.join(model_dir, "svm.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        svm_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        return svm_model, tfidf_vectorizer
    return None, None

def load_saved_model_tanpa_smote(model_dir="models"):
    model_path = os.path.join(model_dir, "svm_model_tanpa_smote.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer_tanpa_smote.pkl")
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        svm_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        return svm_model, tfidf_vectorizer
    return None, None

def get_or_train_model(data, preprocessing_options=None, batch_size=1000, use_tanpa_smote=False):
    from sklearn.pipeline import Pipeline
    if use_tanpa_smote:
        svm_model, tfidf_vectorizer = load_saved_model_tanpa_smote()
    else:
        svm_model, tfidf_vectorizer = load_saved_model()
    if svm_model is not None and tfidf_vectorizer is not None:
        # Model sudah ada, tidak perlu training ulang
        tfidf = tfidf_vectorizer
        svm = svm_model
        processed_texts = data['review_text'].astype(str).tolist()
        X = tfidf.transform(processed_texts)
        y = data['sentiment']
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="POSITIF")
        recall = recall_score(y_test, y_pred, pos_label="POSITIF")
        f1 = f1_score(y_test, y_pred, pos_label="POSITIF")
        cm = confusion_matrix(y_test, y_pred)
        pipeline = Pipeline([
            ('vectorizer', tfidf),
            ('classifier', svm)
        ])
        return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm
    # Jika model belum ada, lakukan training dan simpan
    pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm = train_model(data, preprocessing_options, batch_size)
    save_model_and_vectorizer(pipeline, tfidf)
    return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm

def predict_sentiment(text, pipeline, preprocessing_options=None):
    if preprocessing_options is None:
        preprocessing_options = {'rejoin': True}
    try:
        processed_text = preprocess_text(text, preprocessing_options)
        if not processed_text.strip():
            return {
                'sentiment': 'NETRAL',
                'confidence': 0,
                'probabilities': {'POSITIF': 0, 'NEGATIF': 0}
            }
        prediction = pipeline.predict([processed_text])[0]
        probabilities = pipeline.predict_proba([processed_text])[0]
        confidence = max(probabilities)
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': {
                'POSITIF': probabilities[list(pipeline.classes_).index('POSITIF')],
                'NEGATIF': probabilities[list(pipeline.classes_).index('NEGATIF')]
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            'sentiment': 'ERROR',
            'confidence': 0,
            'probabilities': {'POSITIF': 0, 'NEGATIF': 0}
        }

def analyze_sentiment_trends(data):
    try:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])
        sentiment_trends = data.groupby([data['date'].dt.strftime('%Y-%m-%d'), 'sentiment']).size().reset_index(name='count')
        pivot_trends = sentiment_trends.pivot(index='date', columns='sentiment', values='count').fillna(0)
        if 'POSITIF' in pivot_trends.columns and 'NEGATIF' in pivot_trends.columns:
            pivot_trends['ratio'] = pivot_trends['POSITIF'] / (pivot_trends['POSITIF'] + pivot_trends['NEGATIF'])
        return pivot_trends
    except Exception as e:
        st.error(f"Error in trend analysis: {str(e)}")
        return pd.DataFrame()

def get_table_download_link(df, filename, text):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    except Exception as e:
        st.error(f"Error generating download link: {str(e)}")
        return ''

def display_model_metrics(accuracy, precision, recall, f1, confusion_mat):
    with st.sidebar.expander("🏆 Model Metrics", expanded=False):
        st.write(f"✅ Accuracy: {accuracy:.4f}")
        st.write(f"✅ Precision: {precision:.4f}")
        st.write(f"✅ Recall: {recall:.4f}")
        st.write(f"✅ F1-Score: {f1:.4f}")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Confusion Matrix")
        im = ax.imshow(confusion_mat, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NEGATIF", "POSITIF"])
        ax.set_yticklabels(["NEGATIF", "POSITIF"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, confusion_mat[i, j], ha="center", va="center", 
                    color="white" if confusion_mat[i, j] > confusion_mat.max()/2 else "black")
        st.pyplot(fig)