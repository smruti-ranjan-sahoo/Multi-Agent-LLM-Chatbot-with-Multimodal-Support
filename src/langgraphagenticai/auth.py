import streamlit as st
from supabase import create_client

# 🔥 Use Streamlit secrets (NOT hardcoded)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# 🔥 Dynamic redirect (works for local + cloud)
def get_redirect_url():
    return st.get_option("server.baseUrlPath") or "/"


def login_github():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "github",
        "options": {
            "redirect_to": st.experimental_get_query_params().get(
                "redirect_to",
                ["https://multi-agent-llm-chatbot-with-multimodal-support.streamlit.app"]
            )[0]
        }
    })
    return res.url


def login_google():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": st.experimental_get_query_params().get(
                "redirect_to",
                ["https://multi-agent-llm-chatbot-with-multimodal-support.streamlit.app"]
            )[0]
        }
    })
    return res.url


def get_user():
    return supabase.auth.get_user()