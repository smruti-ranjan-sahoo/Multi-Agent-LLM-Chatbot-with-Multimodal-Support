from supabase import create_client

SUPABASE_URL = "https://wlbjgmpkquqgxiapnhsg.supabase.co"
SUPABASE_KEY = "sb_publishable_zlimAt4wYvqKNThtHtLkYw_JRhQz2wV"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def login_github():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "github"
    })
    return res.url


def login_google():
    res = supabase.auth.sign_in_with_oauth({
        "provider": "google"
    })
    return res.url


def get_user():
    return supabase.auth.get_user()