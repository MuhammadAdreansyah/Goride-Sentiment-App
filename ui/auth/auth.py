"""
Streamlit Authentication System with Firebase Integration (Improved Version)

This application provides:
- Email/password authentication
- Google OAuth login
- User registration
- Password reset functionality
- Session management
"""

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
from typing import Optional, Tuple, Dict
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
        st.error(f"Configuration error: Missing required secrets - {', '.join(missing_vars)}")
        return False
    return True

def sync_login_state():
    """Sinkronisasi status login dari cookie ke session_state di awal aplikasi"""
    is_logged_in_cookie = cookie_controller.get('is_logged_in')
    user_email_cookie = cookie_controller.get('user_email')
    if is_logged_in_cookie == 'True':
        st.session_state['logged_in'] = True
        if user_email_cookie:
            st.session_state['user_email'] = user_email_cookie
    else:
        st.session_state['logged_in'] = False

# =============================================
# SECURITY UTILITIES
# =============================================

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain number"
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
            st.error("Session expired. Please login again.")
            return False
    return True

# =============================================
# FIREBASE INITIALIZATION
# =============================================

def initialize_firebase():
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
            st.error(f"Failed to initialize Firebase after {max_retries} attempts. Please check your configuration.")
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
                    st.error("Authentication error: Database connection issue")
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
                # Set cookie login
                cookie_controller.set('is_logged_in', 'True')
                cookie_controller.set('user_email', user_email)
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

def verify_user_exists(user_email: str, firestore_client: object) -> bool:
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
    """Handle user logout with session cleanup"""
    try:
        logger.info(f"Logging out user: {st.session_state.get('user_email')}")
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
        # Hapus cookie login
        cookie_controller.remove('is_logged_in')
        cookie_controller.remove('user_email')
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        st.error(f"Logout failed: {str(e)}")

# =============================================
# UI COMPONENTS
# =============================================

def display_login_form(firebase_auth: object) -> None:
    """Display and handle login form with email verification check"""
    if st.session_state.get('google_auth_error', False):
        email = st.session_state.get('google_auth_email', '')
        st.error(f"The Google account {email} is not registered in our system. Please register first.")
        del st.session_state['google_auth_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']

    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Sign In")

        email = st.text_input(
            "Email",
            placeholder="your.email@example.com",
            help="Enter your registered email address"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Enter your secure password"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            remember = st.checkbox("Remember me", value=True)
        with col2:
            st.markdown(
                "<div style='text-align: right;'><a href='#' style='color: #1E88E5;'>Forgot Password?</a></div>", 
                unsafe_allow_html=True
            )

        if st.form_submit_button("Continue With Email", use_container_width=True):
            if email and password:
                try:
                    with st.spinner("Authenticating..."):
                        time.sleep(0.5)
                        user = firebase_auth.sign_in_with_email_and_password(email, password)
                        account_info = firebase_auth.get_account_info(user['idToken'])
                        email_verified = account_info['users'][0]['emailVerified']

                        if not email_verified:
                            firebase_auth.send_email_verification(user['idToken'])
                            st.error("Email not verified! Please check your inbox.")
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
                        # Set cookie login
                        cookie_controller.set('is_logged_in', 'True')
                        cookie_controller.set('user_email', email)
                        # Clear any logout or force_auth flags that might interfere
                        for key in ['force_auth_page', 'logout_success']:
                            if key in st.session_state:
                                del st.session_state[key]

                        logger.info(f"Successful login for verified user: {email}")
                        # Return to main app which will handle navigation
                        st.rerun()

                except Exception as e:
                    # Only check and increment rate limit if login failed
                    if not check_rate_limit(email):
                        st.error("Too many login attempts. Please try again later.")
                        return
                    error_message = str(e)
                    logger.error(f"Login failed for {email}: {error_message}")
                    st.session_state['login_error'] = True

                    if "INVALID_PASSWORD" in error_message:
                        st.error("Invalid password. Please try again.")
                    elif "EMAIL_NOT_FOUND" in error_message:
                        st.error("Email not found. Please register first.")
                    elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                        st.error("Too many login attempts. Please try again later.")
                    else:
                        st.error("Authentication failed. Please try again.")
            else:
                st.warning("Please fill in both email and password fields 🚨.")

        st.markdown("""
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <div style='flex-grow: 1; height: 1px; background-color: #e0e0e0;'></div>
                <span style='margin: 0 10px; color: #888;'>OR</span>
                <div style='flex-grow: 1; height: 1px; background-color: #e0e0e0;'></div>
            </div>
        """, unsafe_allow_html=True)

        google_login_btn = st.form_submit_button("Continue with Google", use_container_width=True)
        if google_login_btn:
            with st.spinner("Redirecting to Google..."):
                try:
                    authorization_url = get_google_authorization_url()
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={authorization_url}">', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Google login failed: {str(e)}")
                    st.error("Failed to connect to Google. Please try again later.")

def display_register_form(firebase_auth: object, firestore_client: object) -> None:
    """Display and handle user registration form"""
    google_email = st.session_state.get('google_auth_email', '')

    with st.form("register_form", clear_on_submit=True):
        st.markdown("### Create Account")

        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", placeholder="John")
        with col2:
            last_name = st.text_input("Last Name", placeholder="Doe")

        email = st.text_input(
            "Email",
            value=google_email,
            placeholder="your.email@example.com",
            help="We'll send a verification link to this email"
        )

        if not google_email:
            col3, col4 = st.columns(2)
            with col3:
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a strong password",
                    help="Use 8+ characters with mix of letters, numbers & symbols"
                )
            with col4:
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter password"
                )
        else:
            password = st.text_input(
                "Password (Auto-generated for Google accounts)",
                type="password",
                value=f"Google-{uuid.uuid4().hex[:8]}",
                disabled=True
            )
            confirm_password = password
            st.info("Since you're registering with a Google account, we'll handle password management securely.")

        terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        button_text = "Register with Google" if google_email else "Create Account"

        if st.form_submit_button(button_text, use_container_width=True):
            if not terms:
                st.error("Please accept the Terms of Service to continue.")
                return

            if not all([first_name, last_name, email, password]):
                st.error("Please fill in all required fields.")
                return

            if password != confirm_password:
                st.error("Passwords do not match!")
                return

            if not google_email:
                is_valid_password, password_error = validate_password(password)
                if not is_valid_password:
                    st.error(password_error)
                    return

            try:
                try:
                    existing_user = auth.get_user_by_email(email)
                    st.error("This email is already registered. Please login instead.")
                    return
                except auth.UserNotFoundError:
                    pass

                with st.spinner("Creating your account..."):
                    if google_email:
                        user_record = auth.create_user(
                            email=email,
                            password=password,
                            display_name=f"{first_name} {last_name}",
                            email_verified=True
                        )
                        user = {'localId': user_record.uid}
                    else:
                        user = firebase_auth.create_user_with_email_and_password(email, password)
                        firebase_auth.send_email_verification(user['idToken'])

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

                    if google_email:
                        st.success("Google account registered successfully! You can now login using Google authentication.")
                        if 'google_auth_email' in st.session_state:
                            del st.session_state['google_auth_email']
                    else:
                        st.success("Account created successfully! Please check your email to verify your account.")

            except Exception as e:
                logger.error(f"Registration failed for {email}: {str(e)}")
                if "EMAIL_EXISTS" in str(e):
                    st.error("This email is already registered. Please login instead.")
                elif "WEAK_PASSWORD" in str(e):
                    st.error("Password is too weak. Please choose a stronger password.")
                else:
                    st.error(f"Registration failed: {str(e)}")

def display_riset_password_form(firebase_auth: object) -> None:
    """Display and handle password reset form"""
    with st.form("reset_form", clear_on_submit=True):
        st.markdown("### Reset Password")
        st.info("Enter your email address below and we'll send you instructions to reset your password.")

        email = st.text_input(
            "Email Address",
            placeholder="your.email@example.com",
            help="Enter the email address associated with your account"
        )

        if st.form_submit_button("Send Reset Link", use_container_width=True):
            if not email:
                st.error("Please enter your email address.")
                return

            if not check_rate_limit(f"reset_{email}"):
                st.error("Too many reset attempts. Please try again later.")
                return

            try:
                try:
                    auth.get_user_by_email(email)
                except auth.UserNotFoundError:
                    st.error("No account found with this email address.")
                    return

                firebase_auth.send_password_reset_email(email)
                logger.info(f"Password reset email sent to: {email}")

                st.success("Reset instructions sent! Please check your email. The link will expire in 1 hour.")
                st.markdown(
                    "<div style='text-align: center; margin-top: 1rem;'>"
                    "Didn't receive the email? Check your spam folder or "
                    "<a href='#' style='color: #1E88E5;'>try again</a></div>",
                    unsafe_allow_html=True
                )

            except Exception as e:
                logger.error(f"Password reset failed for {email}: {str(e)}")
                st.error("Failed to send reset link. Please try again later.")

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
            </style>
        """, unsafe_allow_html=True)

        # Display loading state while initializing
        with st.spinner("Initializing application..."):
            # Initialize Firebase services
            firebase_auth, firestore_client = initialize_firebase()
            
        if not firebase_auth or not firestore_client:
            st.error("Failed to initialize authentication services. Please contact support.")
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
                    st.error("Authentication issue detected. Please login again.")
                    logout()
                    st.session_state['force_auth_page'] = True
                    st.rerun()
            return

        # Display welcome message
        with st.container():
            st.markdown("""
                <h1 style='text-align: center; animation: fadeIn 1.5s;'>Selamat Datang👋</h1>
                <p style='text-align: center; color: #666; margin-bottom: 2rem;'>
                Ini aplikasi tugas akhir saya, setelah login anda akan melihat karya yang saya ciptakan dari pagi sampai ke pagi lagi 😂 </p>
            """, unsafe_allow_html=True)

        # Handle Google OAuth callback if present
        if not handle_google_login_callback():
            if 'logout_success' in st.session_state:
                st.success("You have been successfully logged out.")
                del st.session_state['logout_success']

            previous_auth_type = st.session_state.get('auth_type', '')
            auth_type = st.selectbox(
                "Pilih metode autentikasi",  # label yang jelas
                ["🔒 Login", "📝 Register", "🔑 Reset Password"],
                index=0,
                help="Choose your authentication method",
                label_visibility="collapsed"  # label tetap ada, tapi disembunyikan
            )

            if previous_auth_type != auth_type:
                st.session_state['auth_type'] = auth_type
                st.session_state['auth_type_changed'] = True

            with st.container():
                if auth_type == "🔒 Login":
                    display_login_form(firebase_auth)
                elif auth_type == "📝 Register":
                    display_register_form(firebase_auth, firestore_client)
                else:
                    display_riset_password_form(firebase_auth)

    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()