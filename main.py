import streamlit as st

# Sederhanakan: layout ditentukan hanya di awal, berdasarkan status login
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    layout = 'wide'
else:
    layout = 'centered'
st.set_page_config(**{
    'page_title': "GoRide Sentiment Analysis",
    'page_icon': "🛵",
    'layout': layout,
    'initial_sidebar_state': "auto"
})


# 1. Import library & modul eksternal/internal
from ui.auth import auth
from ui.tools.Dashboard_Ringkasan import render_dashboard
from ui.tools.Analisis_Data import render_data_analysis
from ui.tools.Prediksi_Sentimen import render_sentiment_prediction

# 3. Definisi fungsi untuk setiap halaman utama
def login_page():
    """Halaman autentikasi (login/register/lupa password)"""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    auth.main()

def logout_page():
    """Fungsi logout dan redirect ke halaman login"""
    auth.logout()
    # Redirect ke modul autentikasi (root) dengan meta refresh agar transisi smooth
    st.markdown("""
        <meta http-equiv='refresh' content='0.7; url=/' />
    """, unsafe_allow_html=True)
    st.stop()

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
    # Jika sudah login, tampilkan navigasi modul utama & logout
    if st.session_state.get('logged_in', False):
        pg = st.navigation({
            "Tools": [dash_pg, data_pg, pred_pg],
            "Akun": [logout_pg],
        })
        pg.run()
    else:
        # Jika belum login, tampilkan halaman autentikasi
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


