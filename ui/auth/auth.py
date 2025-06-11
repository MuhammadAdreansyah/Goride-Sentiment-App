"""
Streamlit Authentication System with Firebase Integration (Improved Version)

This application provides:
- Email/password authentication
- Google OAuth login
- User registration
- Password reset functionality
- Session management
"""

import warnings
# Suppress the pkg_resources deprecation warning from gcloud/pyrebase
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase
import json
import os
import asyncio
import httpx
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Union, Any
from urllib.parse import urlencode
import uuid
from streamlit_cookies_controller import CookieController

# =============================================
# INITIAL SETUP AND CONFIGURATION
# =============================================

# Initialize logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for storing environment variables and constants"""
    GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
    REDIRECT_URI = st.secrets["REDIRECT_URI"]
    FIREBASE_API_KEY = st.secrets["FIREBASE_API_KEY"]
    SESSION_TIMEOUT = 3600
    MAX_LOGIN_ATTEMPTS = 5
    RATE_LIMIT_WINDOW = 300

# Inisialisasi controller cookies
cookie_controller = CookieController()

# =============================================
# SESSION MANAGEMENT
# =============================================

def initialize_session_state():
    """Ensure all required session state variables are initialized"""
    required_keys = {
        'logged_in': False,
        'login_attempts': 0,
        'firebase_initialized': False,
        'auth_type': '🔒 Login',
        'auth_type_changed': False
    }
    
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def verify_environment():
    """Check all required environment variables are set"""
    required_vars = [
        'GOOGLE_CLIENT_ID',
        'GOOGLE_CLIENT_SECRET',
        'REDIRECT_URI',
        'FIREBASE_API_KEY',
        'firebase',
    ]
    missing_vars = []
    for var in required_vars:
        if var not in st.secrets:
            missing_vars.append(var)
    if missing_vars:
        logger.error(f"Missing secrets: {', '.join(missing_vars)}")
        st.error(f"Kesalahan konfigurasi: Variabel rahasia yang hilang - {', '.join(missing_vars)}")
        return False
    return True

def sync_login_state():
    """Sinkronisasi status login dari cookie ke session_state di awal aplikasi"""
    is_logged_in_cookie = cookie_controller.get('is_logged_in')
    user_email_cookie = cookie_controller.get('user_email')
    remember_me_cookie = cookie_controller.get('remember_me')
    
    if is_logged_in_cookie == 'True':
        st.session_state['logged_in'] = True
        if user_email_cookie:
            st.session_state['user_email'] = user_email_cookie
        if remember_me_cookie == 'True':
            st.session_state['remember_me'] = True
    else:
        st.session_state['logged_in'] = False

def set_remember_me_cookies(email: str, remember: bool = False) -> None:
    """Set cookies for remember me functionality"""
    if remember:
        # Set cookies with longer expiration (30 days)
        cookie_controller.set('is_logged_in', 'True', max_age=30*24*60*60)
        cookie_controller.set('user_email', email, max_age=30*24*60*60)
        cookie_controller.set('remember_me', 'True', max_age=30*24*60*60)
        cookie_controller.set('last_email', email, max_age=90*24*60*60)  # Remember email for 90 days
    else:
        # Set session cookies (expire when browser closes)
        cookie_controller.set('is_logged_in', 'True')
        cookie_controller.set('user_email', email)
        cookie_controller.set('remember_me', 'False')

def get_remembered_email() -> str:
    """Get last remembered email for convenience"""
    return cookie_controller.get('last_email') or ""

def clear_remember_me_cookies() -> None:
    """Clear all remember me related cookies"""
    cookie_controller.remove('is_logged_in')
    cookie_controller.remove('user_email')
    cookie_controller.remove('remember_me')
    # Keep last_email for convenience unless explicitly cleared

# =============================================
# EMAIL VERIFICATION UTILITIES
# =============================================

def check_email_verification_quota() -> Tuple[bool, str]:
    """Check if email verification is likely to succeed based on recent attempts"""
    now = datetime.now()
    quota_key = 'email_verification_attempts'
    attempts = st.session_state.get(quota_key, [])
    
    # Remove attempts older than 1 hour
    valid_attempts = [
        attempt for attempt in attempts 
        if (now - attempt) < timedelta(hours=1)
    ]
    
    # Firebase has a limit of ~100 email actions per hour for free tier
    if len(valid_attempts) >= 50:  # Conservative limit
        return False, "Batas pengiriman email tercapai untuk jam ini. Silakan coba lagi nanti."
    
    valid_attempts.append(now)
    st.session_state[quota_key] = valid_attempts
    return True, ""

def send_email_verification_safe(firebase_auth: Any, id_token: str, email: str) -> Tuple[bool, str]:
    """Safely send email verification with comprehensive error handling"""
    try:
        # Check quota first
        can_send, quota_message = check_email_verification_quota()
        if not can_send:
            return False, quota_message
        
        # Attempt to send verification
        firebase_auth.send_email_verification(id_token)
        logger.info(f"Email verification sent successfully to: {email}")
        return True, "Email verifikasi berhasil dikirim"
        
    except Exception as e:
        error_str = str(e).upper()
        logger.error(f"Failed to send email verification to {email}: {str(e)}")
        
        # Handle specific Firebase errors
        if "QUOTA_EXCEEDED" in error_str or "TOO_MANY_REQUESTS" in error_str:
            return False, "Batas pengiriman email Firebase tercapai. Silakan coba lagi dalam beberapa jam."
        elif "INVALID_ID_TOKEN" in error_str:
            return False, "Token tidak valid. Silakan coba registrasi ulang atau hubungi admin."
        elif "USER_NOT_FOUND" in error_str:
            return False, "User tidak ditemukan. Silakan registrasi ulang."
        elif "EMAIL_NOT_FOUND" in error_str:
            return False, "Email tidak ditemukan dalam sistem."
        elif "OPERATION_NOT_ALLOWED" in error_str:
            return False, "Operasi email verifikasi tidak diizinkan. Hubungi admin."
        else:
            return False, f"Gagal mengirim email verifikasi: {str(e)}"

# =============================================
# ENHANCED VALIDATION SYSTEM
# =============================================

import re

def validate_email_format(email: str) -> Tuple[bool, str]:
    """Validate email format with comprehensive rules"""
    if not email:
        return False, "Email tidak boleh kosong"
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, "Format email tidak valid. Contoh: nama@domain.com"
    
    # Additional checks
    if len(email) > 254:  # RFC 5321 limit
        return False, "Email terlalu panjang (maksimal 254 karakter)"
    
    local_part, domain = email.rsplit('@', 1)
    if len(local_part) > 64:  # RFC 5321 limit
        return False, "Bagian lokal email terlalu panjang"
    
    # Check for consecutive dots
    if '..' in email:
        return False, "Email tidak boleh mengandung titik berturut-turut"
    
    # Check if starts or ends with dot
    if local_part.startswith('.') or local_part.endswith('.'):
        return False, "Email tidak boleh dimulai atau diakhiri dengan titik"
    
    return True, ""

def validate_name_format(name: str, field_name: str) -> Tuple[bool, str]:
    """Validate name format"""
    if not name:
        return False, f"{field_name} tidak boleh kosong"
    
    if len(name) < 2:
        return False, f"{field_name} minimal 2 karakter"
    
    if len(name) > 50:
        return False, f"{field_name} maksimal 50 karakter"
    
    # Allow letters, spaces, and common name characters
    name_pattern = r'^[a-zA-Z\s\'-]+$'
    if not re.match(name_pattern, name):
        return False, f"{field_name} hanya boleh mengandung huruf, spasi, apostrof, dan tanda hubung"
    
    return True, ""

def show_validation_feedback(is_valid: bool, message: str, field_name: str) -> None:
    """Show real-time validation feedback"""
    if not is_valid and message:
        st.error(f"❌ {message}")
    elif is_valid and message != "":
        st.success(f"✅ {field_name} valid")

# =============================================
# SECURITY UTILITIES
# =============================================

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength requirements"""
    if len(password) < 8:
        return False, "Kata sandi harus minimal 8 karakter"
    if not any(c.isupper() for c in password):
        return False, "Kata sandi harus mengandung huruf besar"
    if not any(c.islower() for c in password):
        return False, "Kata sandi harus mengandung huruf kecil"
    if not any(c.isdigit() for c in password):
        return False, "Kata sandi harus mengandung angka"
    return True, ""

def check_rate_limit(user_email: str) -> bool:
    """Check if user has exceeded rate limit for login attempts"""
    now = datetime.now()
    rate_limit_key = f'ratelimit_{user_email}'
    attempts = st.session_state.get(rate_limit_key, [])

    # Remove attempts outside the window
    valid_attempts = [
        attempt for attempt in attempts 
        if (now - attempt) < timedelta(seconds=Config.RATE_LIMIT_WINDOW)
    ]

    if len(valid_attempts) >= Config.MAX_LOGIN_ATTEMPTS:
        return False

    valid_attempts.append(now)
    st.session_state[rate_limit_key] = valid_attempts
    return True

def check_session_timeout() -> bool:
    """Check if user session has timed out"""
    if 'login_time' in st.session_state:
        elapsed = (datetime.now() - st.session_state['login_time']).total_seconds()
        if elapsed > Config.SESSION_TIMEOUT:
            logout()
            st.error("Sesi telah berakhir. Silakan login kembali.")
            return False
    return True

# =============================================
# FIREBASE INITIALIZATION
# =============================================

def initialize_firebase() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize Firebase Admin SDK and Pyrebase with better error handling
    and retry logic. Ambil semua konfigurasi dari st.secrets.
    """
    max_retries = 3
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            if st.session_state.get('firebase_initialized', False):
                return st.session_state['firebase_auth'], st.session_state['firestore']
            if not verify_environment():
                return None, None
            # Ambil konfigurasi kredensial dari st.secrets
            service_account = dict(st.secrets["firebase"])
            cred = credentials.Certificate(service_account)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            config = {
                "apiKey": Config.FIREBASE_API_KEY,
                "authDomain": f"{service_account['project_id']}.firebaseapp.com",
                "projectId": service_account['project_id'],
                "databaseURL": f"https://{service_account['project_id']}-default-rtdb.firebaseio.com",
                "storageBucket": f"{service_account['project_id']}.appspot.com"
            }
            pb = pyrebase.initialize_app(config)
            firebase_auth = pb.auth()
            firestore_client = firestore.client()
            st.session_state['firebase_auth'] = firebase_auth
            st.session_state['firestore'] = firestore_client
            st.session_state['firebase_initialized'] = True
            logger.info("Firebase initialized successfully (from secrets)")
            return firebase_auth, firestore_client
        except Exception as e:
            logger.error(f"Firebase initialization attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            st.error(f"Gagal menginisialisasi Firebase setelah {max_retries} percobaan. Silakan periksa konfigurasi Anda.")
            return None, None
    
    return None, None

# =============================================
# GOOGLE OAUTH INTEGRATION
# =============================================

async def exchange_google_token(code: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Exchange Google authorization code for user info
    """
    async with httpx.AsyncClient() as client:
        token_url = 'https://oauth2.googleapis.com/token'
        payload = {
            'client_id': Config.GOOGLE_CLIENT_ID,
            'client_secret': Config.GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': Config.REDIRECT_URI
        }

        try:
            # Exchange code for tokens
            response = await client.post(token_url, data=payload)
            response.raise_for_status()
            token_data = response.json()

            # Get user info using access token
            user_info_response = await client.get(
                'https://www.googleapis.com/oauth2/v3/userinfo',
                headers={'Authorization': f'Bearer {token_data["access_token"]}'}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()

            logger.info(f"Successfully exchanged Google token for user: {user_info.get('email')}")
            return user_info.get('email'), user_info

        except Exception as e:
            logger.error(f"Google token exchange error: {str(e)}")
            return None, None

def handle_google_login_callback() -> bool:
    """
    Handle Google OAuth callback after user authentication
    Returns True if successful, False otherwise
    """
    try:
        if 'code' not in st.query_params:
            return False
            
        code = st.query_params.get('code')
        if not code or not isinstance(code, str):
            logger.warning("Invalid authorization code received")
            return False

        async def async_token_exchange():
            try:
                return await exchange_google_token(code)
            except Exception as e:
                logger.error(f"Token exchange error: {str(e)}")
                return None, None

        user_email, user_info = asyncio.run(async_token_exchange())
        if not user_email or not user_info:
            return False

        try:
            # Check if user exists in Firebase
            try:
                firebase_user = auth.get_user_by_email(user_email)
                firestore_client = st.session_state.get('firestore')
                
                if not firestore_client:
                    logger.error("Firestore client not available")
                    st.error("Kesalahan autentikasi: Masalah koneksi database")
                    return False

                user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
                if not user_doc.exists:
                    logger.warning(f"Google OAuth: User exists in Auth but not in Firestore: {user_email}")
                    st.session_state['google_auth_error'] = True
                    st.session_state['google_auth_email'] = user_email
                    if 'logged_in' in st.session_state:
                        del st.session_state['logged_in']
                    st.query_params.clear()
                    st.rerun()
                    return False                # Successful login
                st.session_state.update({
                    'user_email': user_email,
                    'user_info': user_info,
                    'firebase_user': {
                        'uid': firebase_user.uid,
                        'email': firebase_user.email
                    },
                    'logged_in': True,
                    'login_time': datetime.now(),
                    'login_success': True,  # New flag for clean state management
                    'pending_config_change': True  # Flag to indicate we need to update layout
                })
                
                # Clear any logout or force_auth flags that might interfere
                for key in ['force_auth_page', 'logout_success']:
                    if key in st.session_state:
                        del st.session_state[key]
                    
                logger.info(f"Successfully logged in user: {user_email}")
                # Set remember me cookies (Google login = auto remember)
                set_remember_me_cookies(user_email, True)
                
                # Show success toast
                username = user_email.split('@')[0].capitalize()
                show_success_toast(f"Login Google berhasil! Selamat datang, {username}!")
                
                st.rerun()
                return True

            except auth.UserNotFoundError:
                logger.warning(f"Google OAuth: User not registered in system: {user_email}")
                st.session_state['google_auth_error'] = True
                st.session_state['google_auth_email'] = user_email
                if 'logged_in' in st.session_state:
                    del st.session_state['logged_in']
                st.query_params.clear()
                st.rerun()
                return False

        except Exception as user_error:
            logger.error(f"User processing error: {str(user_error)}")
            if 'logged_in' in st.session_state:
                del st.session_state['logged_in']
            st.query_params.clear()
            return False

    except Exception as e:
        logger.error(f"Google login callback error: {str(e)}")
        if 'logged_in' in st.session_state:
            del st.session_state['logged_in']
        return False

def verify_user_exists(user_email: str, firestore_client: Any) -> bool:
    """
    Verify that the user exists and has valid data in Firestore
    """
    try:
        firebase_user = auth.get_user_by_email(user_email)
        user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            required_fields = ['email', 'first_name', 'last_name']
            return all(field in user_data for field in required_fields)
        
        logger.warning(f"User {user_email} has no Firestore data")
        return False

    except auth.UserNotFoundError:
        logger.warning(f"User {user_email} not found in Firebase Auth")
        return False
    except Exception as e:
        logger.error(f"Error verifying user {user_email}: {str(e)}")
        return False

def get_google_authorization_url() -> str:
    """Generate Google OAuth authorization URL with required scopes"""
    base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    params = {
        'client_id': Config.GOOGLE_CLIENT_ID,
        'redirect_uri': Config.REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    return f"{base_url}?{urlencode(params)}"

# =============================================
# AUTHENTICATION FUNCTIONS
# =============================================

def logout() -> None:
    """Handle user logout with session cleanup and proper cookie management"""
    try:
        user_email = st.session_state.get('user_email')
        logger.info(f"Logging out user: {user_email}")
        
        # Simpan hanya objek firebase yang diperlukan
        fb_auth = st.session_state.get('firebase_auth', None)
        fs_client = st.session_state.get('firestore', None)
        fb_initialized = st.session_state.get('firebase_initialized', False)
        
        # Bersihkan seluruh session state
        st.session_state.clear()
        
        # Kembalikan objek firebase jika ada
        if fb_auth:
            st.session_state['firebase_auth'] = fb_auth
        if fs_client:
            st.session_state['firestore'] = fs_client
        if fb_initialized:
            st.session_state['firebase_initialized'] = fb_initialized
        
        # Reset status login
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None
        st.session_state["logout_success"] = True
        st.query_params.clear()
        
        # Clear remember me cookies but keep last_email for convenience
        clear_remember_me_cookies()
        
        # Show logout success toast
        # show_success_toast("Berhasil logout. Sampai jumpa lagi!")
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        show_error_toast(f"Logout failed: {str(e)}")

# =============================================
# UI FEEDBACK SYSTEM
# =============================================

def show_toast_notification(message: str, icon: str = "ℹ️") -> None:
    """Display toast notification with consistent styling"""
    st.toast(message, icon=icon)

def show_success_toast(message: str) -> None:
    """Display success toast notification"""
    show_toast_notification(message, "✅")

def show_error_toast(message: str) -> None:
    """Display error toast notification"""
    show_toast_notification(message, "❌")

def show_warning_toast(message: str) -> None:
    """Display warning toast notification"""
    show_toast_notification(message, "⚠️")

def show_info_toast(message: str) -> None:
    """Display info toast notification"""
    show_toast_notification(message, "ℹ️")

# =============================================
# UI COMPONENTS
# =============================================

def display_login_form(firebase_auth: Any) -> None:
    """Display and handle login form with email verification check and enhanced validation"""
    if st.session_state.get('google_auth_error', False):
        email = st.session_state.get('google_auth_email', '')
        show_error_toast(f"Akun Google {email} tidak terdaftar dalam sistem kami. Silakan daftar terlebih dahulu.")
        del st.session_state['google_auth_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']

    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Sign In")

        # Get remembered email for convenience
        remembered_email = get_remembered_email()
        
        email = st.text_input(
            "Email",
            value=remembered_email,
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email terdaftar Anda"
        )

        # Real-time email validation
        if email and email != remembered_email:
            is_valid_email, email_message = validate_email_format(email)
            if not is_valid_email:
                st.error(f"❌ {email_message}")

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Masukkan kata sandi Anda",
            help="Masukkan kata sandi yang aman"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            remember = st.checkbox("Remember me", value=True, help="Simpan login selama 30 hari")
        with col2:
            st.markdown(
                """
                <div class='forgot-password-right'><a href='#' class='reset-link'>Lupa Kata Sandi?</a></div>
                """,
                unsafe_allow_html=True
            )

        if st.form_submit_button("Lanjutkan dengan Email", use_container_width=True):
            if email and password:
                # Validate email format before attempting login
                is_valid_email, email_message = validate_email_format(email)
                if not is_valid_email:
                    show_error_toast(email_message)
                    return
                    
                try:
                    with st.spinner("Mengautentikasi..."):
                        time.sleep(0.5)
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        account_info = firebase_auth.get_account_info(user['idToken'])
                        email_verified = account_info['users'][0]['emailVerified']

                        if not email_verified:
                            # Try to send email verification with better error handling
                            try:
                                firebase_auth.send_email_verification(user['idToken'])
                                show_warning_toast("Email belum diverifikasi! Link verifikasi telah dikirim ulang.")
                                logger.info(f"Email verification resent to: {email}")
                            except Exception as verification_error:
                                logger.error(f"Failed to resend email verification to {email}: {str(verification_error)}")
                                error_str = str(verification_error).upper()
                                if "QUOTA_EXCEEDED" in error_str or "TOO_MANY_REQUESTS" in error_str:
                                    show_warning_toast("Email belum diverifikasi! Batas pengiriman email tercapai, coba lagi nanti.")
                                else:
                                    show_warning_toast("Email belum diverifikasi! Gagal mengirim ulang email verifikasi.")
                            logger.warning(f"Unverified email login attempt: {email}")
                            return
                        
                        # Update session state for successful login
                        st.session_state.update({
                            'user': user,
                            'user_email': email,
                            'logged_in': True,
                            'login_time': datetime.now(),
                            'login_success': True,  # New flag to trigger clean state transitions
                            'pending_config_change': True  # Flag to indicate we need to update layout
                        })
                        
                        # Set remember me cookies
                        set_remember_me_cookies(email, remember)
                        
                        # Clear any logout or force_auth flags that might interfere
                        for key in ['force_auth_page', 'logout_success']:
                            if key in st.session_state:
                                del st.session_state[key]

                        # Show success toast
                        username = email.split('@')[0].capitalize()
                        show_success_toast(f"Selamat datang kembali, {username}!")
                        
                        logger.info(f"Successful login for verified user: {email}")
                        # Return to main app which will handle navigation
                        st.rerun()

                except Exception as e:
                    # Only check and increment rate limit if login failed
                    if not check_rate_limit(email):
                        show_error_toast("Terlalu banyak percobaan login. Silakan coba lagi nanti.")
                        return
                    error_message = str(e)
                    logger.error(f"Login failed for {email}: {error_message}")
                    st.session_state['login_error'] = True

                    if "INVALID_PASSWORD" in error_message:
                        show_error_toast("Kata sandi tidak valid. Silakan coba lagi.")
                    elif "EMAIL_NOT_FOUND" in error_message:
                        show_error_toast("Email tidak ditemukan. Silakan daftar terlebih dahulu.")
                    elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                        show_error_toast("Terlalu banyak percobaan login. Silakan coba lagi nanti.")
                    else:
                        show_error_toast("Autentikasi gagal. Silakan coba lagi.")
            else:
                show_warning_toast("Silakan isi kolom email dan kata sandi.")

        # Divider dengan garis kiri-kanan dan teks "ATAU" di tengah
        st.markdown("""
            <div class='auth-divider-custom'>
                <div class='divider-line-custom'></div>
                <span class='divider-text-custom'>ATAU</span>
                <div class='divider-line-custom'></div>
            </div>
        """, unsafe_allow_html=True)

        google_login_btn = st.form_submit_button("Lanjutkan dengan Google", use_container_width=True)
        if google_login_btn:
            with st.spinner("Mengalihkan ke Google..."):
                try:
                    authorization_url = get_google_authorization_url()
                    show_info_toast("Mengalihkan ke Google untuk autentikasi...")
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={authorization_url}">', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Google login failed: {str(e)}")
                    show_error_toast("Gagal terhubung ke Google. Silakan coba lagi nanti.")

def display_register_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Display and handle user registration form with enhanced validation"""
    google_email = st.session_state.get('google_auth_email', '')
    
    # Initialize form data in session state to preserve values on error
    if 'register_form_data' not in st.session_state:
        st.session_state['register_form_data'] = {
            'first_name': '',
            'last_name': '',
            'email': google_email,
            'terms_accepted': False
        }
    
    # Update email if google_email is set
    if google_email and st.session_state['register_form_data']['email'] != google_email:
        st.session_state['register_form_data']['email'] = google_email

    with st.form("register_form", clear_on_submit=False):
        st.markdown("### Sign Up")

        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input(
                "Username", 
                value=st.session_state['register_form_data']['first_name'],
                placeholder="John"
            )
            # Real-time validation for first name
            if first_name and first_name.strip():
                is_valid_fname, fname_message = validate_name_format(first_name.strip(), "Username")
                if not is_valid_fname:
                    st.error(f"❌ {fname_message}")
                    
        with col2:
            last_name = st.text_input(
                "Lastname", 
                value=st.session_state['register_form_data']['last_name'],
                placeholder="Doe"
            )
            # Real-time validation for last name
            if last_name and last_name.strip():
                is_valid_lname, lname_message = validate_name_format(last_name.strip(), "Lastname")
                if not is_valid_lname:
                    st.error(f"❌ {lname_message}")

        email = st.text_input(
            "Email",
            value=st.session_state['register_form_data']['email'],
            placeholder="email.anda@contoh.com",
            help="Kami akan mengirimkan link verifikasi ke email ini"
        )
        
        # Real-time email validation
        if email and email.strip() and not google_email:
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")
            else:
                st.success("✅ Format email valid")

        if not google_email:
            col3, col4 = st.columns(2)
            with col3:
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Buat kata sandi yang kuat",
                    help="Gunakan 8+ karakter dengan campuran huruf, angka & simbol"
                )
                # Real-time password validation
                if password:
                    is_valid_pass, pass_message = validate_password(password)
                    if not is_valid_pass:
                        st.error(f"❌ {pass_message}")
                    else:
                        st.success("✅ Kata sandi memenuhi kriteria")
                        
            with col4:
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Masukkan ulang kata sandi"
                )
                # Password confirmation validation
                if confirm_password and password:
                    if password != confirm_password:
                        st.error("❌ Kata sandi tidak cocok")
                    else:
                        st.success("✅ Kata sandi cocok")
        else:
            password = st.text_input(
                "Kata Sandi (Dibuat otomatis untuk akun Google)",
                type="password",
                value=f"Google-{uuid.uuid4().hex[:8]}",
                disabled=True
            )
            confirm_password = password
            st.info("Karena Anda mendaftar dengan akun Google, kami akan mengelola kata sandi dengan aman.")

        terms = st.checkbox(
            "Saya setuju dengan Syarat Layanan dan Kebijakan Privasi",
            value=st.session_state['register_form_data']['terms_accepted']
        )
        button_text = "Daftar dengan Google" if google_email else "Buat Akun"

        if st.form_submit_button(button_text, use_container_width=True):
            # Update session state with current form values
            st.session_state['register_form_data'].update({
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'terms_accepted': terms
            })
            
            # Comprehensive validation
            validation_errors = []
            
            if not terms:
                validation_errors.append("Silakan terima Syarat Layanan untuk melanjutkan.")

            if not all([first_name, last_name, email, password]):
                validation_errors.append("Silakan isi semua kolom yang diperlukan.")
            
            # Validate names
            if first_name and first_name.strip():
                is_valid_fname, fname_message = validate_name_format(first_name.strip(), "Username")
                if not is_valid_fname:
                    validation_errors.append(fname_message)
                
            if last_name and last_name.strip():
                is_valid_lname, lname_message = validate_name_format(last_name.strip(), "Lastname")
                if not is_valid_lname:
                    validation_errors.append(lname_message)
            
            # Validate email
            if not google_email and email and email.strip():
                is_valid_email, email_message = validate_email_format(email.strip())
                if not is_valid_email:
                    validation_errors.append(email_message)

            if password != confirm_password:
                validation_errors.append("Kata sandi tidak cocok!")

            if not google_email:
                is_valid_password, password_error = validate_password(password)
                if not is_valid_password:
                    validation_errors.append(password_error)
            
            # Show all validation errors
            if validation_errors:
                for error in validation_errors:
                    show_error_toast(error)
                return

            try:
                try:
                    existing_user = auth.get_user_by_email(email)
                    show_error_toast("Email ini sudah terdaftar. Silakan login.")
                    return
                except auth.UserNotFoundError:
                    pass

                with st.spinner("Membuat akun Anda..."):
                    if google_email:
                        user_record = auth.create_user(
                            email=email,
                            password=password,
                            display_name=f"{first_name} {last_name}",
                            email_verified=True
                        )
                        user = {'localId': user_record.uid}
                        registration_success = True
                        email_verification_sent = False  # Google users don't need verification
                    else:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        registration_success = True
                        
                        # Use safe email verification function
                        email_verification_sent, verification_message = send_email_verification_safe(
                            firebase_auth, user['idToken'], email.strip() if email else ""
                        )
                        
                        if not email_verification_sent:
                            logger.warning(f"Email verification failed for {email}: {verification_message}")
                            show_warning_toast(f"Akun berhasil dibuat, tapi: {verification_message}")
                        else:
                            logger.info(f"Email verification sent successfully to: {email}")

                    user_data = {
                        "first_name": first_name,
                        "last_name": last_name,
                        "email": email,
                        "auth_provider": "google" if google_email else "email",
                        "created_at": datetime.now().isoformat(),
                        "last_login": datetime.now().isoformat()
                    }

                    firestore_client.collection('users').document(user['localId']).set(user_data)
                    logger.info(f"Successfully created account for: {email}")

                    # Clear form data on successful registration
                    if 'register_form_data' in st.session_state:
                        del st.session_state['register_form_data']

                    # Show appropriate success message based on verification status
                    if google_email:
                        show_success_toast("Akun Google berhasil didaftarkan! Anda sekarang dapat login.")
                        if 'google_auth_email' in st.session_state:
                            del st.session_state['google_auth_email']
                    else:
                        if email_verification_sent:
                            show_success_toast("Akun berhasil dibuat! Periksa email untuk verifikasi.")
                            st.success("✅ Akun berhasil dibuat! Email verifikasi telah dikirim ke alamat Anda.")
                        else:
                            show_success_toast("Akun berhasil dibuat! Email verifikasi mungkin tertunda.")
                            st.warning("⚠️ Akun berhasil dibuat, namun email verifikasi belum terkirim. Anda dapat meminta pengiriman ulang saat login.")

                    # Store verification status for potential resend feature
                    st.session_state['last_registration_email'] = email
                    st.session_state['email_verification_sent'] = email_verification_sent

            except Exception as e:
                logger.error(f"Registration failed for {email}: {str(e)}")
                if "EMAIL_EXISTS" in str(e):
                    show_error_toast("Email ini sudah terdaftar. Silakan login.")
                elif "WEAK_PASSWORD" in str(e):
                    show_error_toast("Kata sandi terlalu lemah. Silakan pilih kata sandi yang lebih kuat.")
                else:
                    show_error_toast(f"Pendaftaran gagal: {str(e)}")

def display_reset_password_form(firebase_auth: Any) -> None:
    """Display and handle password reset form with enhanced feedback"""
    with st.form("reset_form", clear_on_submit=True):
        st.markdown("### Reset Password")
        st.info("Masukkan alamat email Anda di bawah ini dan kami akan mengirimkan petunjuk untuk mereset kata sandi Anda.")

        email = st.text_input(
            "Alamat Email",
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email yang terkait dengan akun Anda"
        )

        # Real-time email validation
        if email and email.strip():
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")

        if st.form_submit_button("Kirim Link Reset", use_container_width=True):
            if not email or not email.strip():
                show_warning_toast("Silakan masukkan alamat email Anda.")
                return

            # Validate email format
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                show_error_toast(email_message)
                return

            if not check_rate_limit(f"reset_{email}"):
                show_error_toast("Terlalu banyak percobaan reset. Silakan coba lagi nanti.")
                return

            try:
                try:
                    auth.get_user_by_email(email)
                except auth.UserNotFoundError:
                    show_error_toast("Tidak ada akun yang ditemukan dengan alamat email ini.")
                    return

                firebase_auth.send_password_reset_email(email)
                logger.info(f"Password reset email sent to: {email}")

                show_success_toast("Link reset password telah dikirim ke email Anda!")
                st.success("Petunjuk reset telah dikirim! Silakan periksa email Anda. Link akan kedaluwarsa dalam 1 jam.")
                st.markdown(
                    "<div style='text-align: center; margin-top: 1rem;'>"
                    "Tidak menerima email? Periksa folder spam Anda atau "
                    "<a href='#' class='reset-link'>coba lagi</a></div>",
                    unsafe_allow_html=True
                )

            except Exception as e:
                logger.error(f"Password reset failed for {email}: {str(e)}")
                show_error_toast("Gagal mengirim link reset. Silakan coba lagi nanti.")

# =============================================
# MAIN APPLICATION
# =============================================

def main() -> None:
    """Main application entry point with better error handling"""
    try:
        # Sinkronisasi status login dari cookie ke session_state
        sync_login_state()
        # Initialize session state first
        initialize_session_state()
        
        # Note: Page configuration is now handled by main.py
        # for better coordination between authentication and main views
        
        # Custom CSS styling - but page configuration is handled by main app
        st.markdown("""
            <style>
            .main { padding: 2rem; }
            .stSelectbox { margin-bottom: 2rem; }
            .stButton button {
                width: 100%;
                border-radius: 20px;
                height: 3rem;
                font-weight: bold;
            }
            .success-message {
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            div[data-testid="stForm"] {
                border: 1px solid #f0f2f6;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            /* Divider custom untuk "ATAU" */
            .auth-divider-custom {
                display: flex;
                align-items: center;
                margin: 1.5rem 0 1.5rem 0;
            }
            .divider-line-custom {
                flex: 1;
                height: 1px;
                background: #e0e0e0;
            }
            .divider-text-custom {
                margin: 0 1rem;
                color: #888;
                font-weight: 600;
                letter-spacing: 1px;
                font-size: 0.95rem;
            }
            /* Lupa kata sandi di kanan */
            .forgot-password-right {
                text-align: right;
                width: 100%;
                margin-top: 0.2rem;
            }
            .forgot-password-right a {
                color: #1976d2;
                text-decoration: none;
                font-size: 0.95rem;
            }
            .forgot-password-right a:hover {
                text-decoration: underline;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display loading state while initializing
        with st.spinner("Menginisialisasi aplikasi..."):
            # Initialize Firebase services
            firebase_auth, firestore_client = initialize_firebase()
            
        if not firebase_auth or not firestore_client:
            st.error("Gagal menginisialisasi layanan autentikasi. Silakan hubungi dukungan.")
            logger.critical("Critical initialization failure - Firebase services not available")
            return        # Check if user is already logged in
        if st.session_state.get('logged_in', False) and not st.session_state.get('force_auth_page', False):
            if check_session_timeout():
                user_email = st.session_state.get('user_email')
                if user_email and verify_user_exists(user_email, firestore_client):
                    # Return control to main.py which will handle navigation
                    return
                else:
                    # Force logout if user verification fails
                    logger.warning(f"User {user_email} failed verification, forcing logout")
                    st.error("Masalah autentikasi terdeteksi. Silakan login kembali.")
                    logout()
                    st.session_state['force_auth_page'] = True
                    st.rerun()
            return

        # Display welcome message with enhanced styling
        with st.container():
            st.markdown("""
                <div class="welcome-header">
                    <h1 style='text-align: center;'>Selamat Datang 👋</h1>
                    <p style='text-align: center; color: #6c757d; margin-bottom: 2rem; font-size: 1.1rem;'>
                        Ini aplikasi tugas akhir saya, setelah login anda akan melihat karya yang saya ciptakan dari pagi sampai ke pagi lagi 😂
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Tampilkan pesan logout jika ada query param
        if st.query_params.get("logout") == "1":
            st.toast("Anda telah berhasil logout.", icon="✅")
            st.query_params.clear()

        # Handle Google OAuth callback if present
        if not handle_google_login_callback():
            previous_auth_type = st.session_state.get('auth_type', '')
            auth_type = st.selectbox(
                "Pilih metode autentikasi",  # label yang jelas
                ["🔒 Sign In", "📝 Sign Up", "🔑 Reset Password"],
                index=0,
                help="Pilih metode autentikasi Anda",
                label_visibility="collapsed"  # label tetap ada, tapi disembunyikan
            )

            if previous_auth_type != auth_type:
                st.session_state['auth_type'] = auth_type
                st.session_state['auth_type_changed'] = True
                # Clear form data when switching between auth types
                if 'register_form_data' in st.session_state:
                    del st.session_state['register_form_data']

            with st.container():
                if auth_type == "🔒 Sign In":
                    display_login_form(firebase_auth)
                elif auth_type == "📝 Sign Up":
                    display_register_form(firebase_auth, firestore_client)
                elif auth_type == "🔑 Reset Password":
                    display_reset_password_form(firebase_auth)

    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("Terjadi kesalahan yang tidak terduga. Silakan coba lagi nanti.")
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()