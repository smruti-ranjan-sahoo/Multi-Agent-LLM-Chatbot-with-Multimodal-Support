import streamlit as st
from supabase import create_client

# 🔥 Use Streamlit secrets (secure)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# 🔥 Centralized redirect URL (clean + reliable)
REDIRECT_URL = "https://multi-agent-llm-chatbot-with-multimodal-support.streamlit.app"


# ✅ GitHub Login
def login_github():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "github",
        "options": {
            "redirect_to": REDIRECT_URL
        }
    })
    return res.url


# ✅ Google Login
def login_google():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": REDIRECT_URL
        }
    })
    return res.url


# ✅ Get logged-in user
def get_user():
    return supabase.auth.get_user()