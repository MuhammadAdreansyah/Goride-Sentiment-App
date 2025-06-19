import streamlit as st

# 1. Cek status login dari session_state (harus sebelum komponen Streamlit lain)
logged_in = False
try:
    logged_in = st.session_state.get('logged_in', False)
except Exception:
    logged_in = False

# 2. Konfigurasi page dinamis sesuai status login
if logged_in:
    st.set_page_config(
        page_title="GoRide Sentiment Analysis",
        page_icon="🛵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="GoRide Sentiment Analysis",
        page_icon="🔐",
        layout="centered",
        initial_sidebar_state="expanded"
    )

from ui.auth import auth
from ui.tools.Dashboard_Ringkasan import render_dashboard
from ui.tools.Analisis_Data import render_data_analysis
from ui.tools.Prediksi_Sentimen import render_sentiment_prediction
from ui.utils import check_and_prepare_models_with_progress

# 3. Definisi fungsi untuk setiap halaman utama
def login_page():
    """Halaman autentikasi (login/register/lupa password)"""
    st.markdown(
        """
        <style>
        [data-testid='stSidebar'], [data-testid='collapsedControl'] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Jika login sukses, redirect agar page_config baru diterapkan
    if st.session_state.get('login_success', False):
        st.markdown('<meta http-equiv="refresh" content="0;url=/" />', unsafe_allow_html=True)
        return
    auth.main()

def model_preparation_page():
    """Halaman khusus untuk persiapan model setelah login berhasil"""
    st.markdown(
        """
        <style>
        [data-testid='stSidebar'], [data-testid='collapsedControl'] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header halaman
    st.markdown("# 🤖 Persiapan Model Sentiment Analysis")
    st.markdown("---")
    
    # Welcome message
    user_email = st.session_state.get('user_email', 'User')
    st.markdown(f"### Selamat datang, **{user_email}**! 👋")
    
    st.info("""
    🔧 **Sistem sedang mempersiapkan model AI untuk analisis sentimen GoRide**
    
    📝 Proses ini meliputi:
    - ✅ Pemeriksaan model yang tersedia
    - 🤖 Pelatihan model jika diperlukan  
    - 📊 Validasi performa model
    - 🚀 Persiapan tools analisis
    
    *Proses ini hanya dilakukan sekali dan membutuhkan waktu beberapa menit.*
    """)
    
    # Status container
    status_container = st.container()
    
    # Check if models are being prepared
    if not st.session_state.get('model_preparation_started', False):
        with status_container:
            if st.button("🚀 **Mulai Persiapan Model**", type="primary", use_container_width=True):
                st.session_state['model_preparation_started'] = True
                st.rerun()
    else:
        # Model preparation in progress
        with status_container:
            st.markdown("### 🔄 Sedang Mempersiapkan Model...")
            
            try:
                # Show preparation progress
                models_ready = check_and_prepare_models_with_progress()
                
                if models_ready:
                    st.session_state['models_prepared'] = True
                    st.session_state['model_preparation_completed'] = True
                    
                    # Success message with celebration
                    st.balloons()
                    st.success("🎉 **Model berhasil disiapkan!**")
                    
                    st.markdown("---")
                    st.markdown("### ✅ Persiapan Selesai!")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🎯 **Lanjutkan ke Dashboard**", type="primary", use_container_width=True):
                            st.session_state['ready_for_tools'] = True
                            st.rerun()
                            
                    st.markdown("*Anda siap menggunakan semua fitur analisis sentimen!*")
                else:
                    st.error("❌ **Gagal mempersiapkan model**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🔄 Coba Lagi"):
                            st.session_state['model_preparation_started'] = False
                            st.rerun()
                    with col2:
                        if st.button("🚪 Logout"):
                            auth.logout()
                            
            except Exception as e:
                st.error(f"❌ **Error saat mempersiapkan model:** {str(e)}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Coba Lagi"):
                        st.session_state['model_preparation_started'] = False
                        st.rerun()
                with col2:
                    if st.button("🚪 Logout"):
                        auth.logout()

def logout_page():
    """Fungsi logout dan redirect ke halaman utama autentikasi"""
    auth.logout()
    # Redirect ke root dengan query param logout=1
    st.markdown(
        '<meta http-equiv="refresh" content="0;url=/?logout=1" />',
        unsafe_allow_html=True
    )

def dashboard_page():
    """Halaman Dashboard Ringkasan"""
    render_dashboard()

def analisis_data_page():
    """Halaman Analisis Data"""
    render_data_analysis()

def prediksi_sentimen_page():
    """Halaman Prediksi Sentimen"""
    render_sentiment_prediction()

# 4. Definisi Page untuk navigasi multi-workflow
logout_pg = st.Page(logout_page, title="Logout", icon=":material/logout:")
dash_pg = st.Page(dashboard_page, title="Dashboard Ringkasan", icon=":material/dashboard:", default=True)
data_pg = st.Page(analisis_data_page, title="Analisis Data", icon=":material/analytics:")
pred_pg = st.Page(prediksi_sentimen_page, title="Prediksi Sentimen", icon=":material/psychology:")

# 5. Fungsi main() sebagai workflow utama aplikasi
def main():
    """Workflow utama aplikasi: autentikasi, navigasi, dan routing modul."""
    # Sinkronisasi status login dari cookie ke session_state (penting untuk refresh)
    auth.sync_login_state()
    auth.initialize_session_state()
    
    # Tampilkan toast jika login sukses
    if st.session_state.get('login_success', False):
        st.toast(f"User {st.session_state.get('user_email', '')} login successfully!", icon="✅")
        st.session_state['login_success'] = False
    
    # Workflow berdasarkan status user
    if st.session_state.get('logged_in', False):
        # User sudah login, cek status model
        if st.session_state.get('ready_for_tools', False):
            # Model sudah siap, tampilkan tools utama
            pg = st.navigation({
                "Tools": [dash_pg, data_pg, pred_pg],
                "Akun": [logout_pg],
            })
            pg.run()
        else:
            # Model belum siap, tampilkan halaman persiapan
            model_preparation_page()
    else:
        # User belum login, tampilkan halaman login
        login_page()

# 6. Entry point aplikasi
if __name__ == "__main__":
    main()

# =============================================
# GoRide Sentiment Analysis - Main Entry Point
# =============================================
# Struktur:
# 1. Import library & modul
# 2. Konfigurasi halaman Streamlit (dinamis)
# 3. Definisi fungsi halaman (login, logout, dashboard, tools)
# 4. Definisi Page untuk navigasi
# 5. Fungsi main() sebagai workflow utama
# 6. Entry point


