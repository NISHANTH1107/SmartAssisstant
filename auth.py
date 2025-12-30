import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from datetime import datetime, timezone
from db import users_col

def build_config_from_db():
    users = users_col.find({})
    credentials = {"usernames": {}}

    for u in users:
        credentials["usernames"][u["username"]] = {
            "email": u["email"],
            "name": u["name"],
            "password": u["password"]
        }

    config = {
        "credentials": credentials,
        "cookie": {
            "expiry_days": 30,
            "key": "studymate_signature_key",
            "name": "studymate_cookie"
        },
        "preauthorized": {
            "emails": []
        }
    }
    return config

def get_authenticator():
    config = build_config_from_db()

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )
    return authenticator, config

def register_new_user(authenticator, config):
    result = authenticator.register_user(location="main")

    if not result:
        return False

    # streamlit_authenticator versions vary:
    # result can be (username, email, name) OR email OR username
    credentials = config["credentials"]["usernames"]

    # Safely detect the newly added user
    new_username = None
    for uname in credentials:
        if users_col.find_one({"username": uname}) is None:
            new_username = uname
            break

    if not new_username:
        st.error("Registration failed: unable to detect new user")
        return False

    user = credentials[new_username]

    users_col.insert_one({
        "username": new_username,
        "name": user.get("name", new_username),
        "email": user.get("email", ""),
        "password": user["password"],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    })

    st.success("✅ User registered successfully! Please login.")
    return True


def reset_password(authenticator, config, username):
    if authenticator.reset_password(username, location="main"):
        hashed = config["credentials"]["usernames"][username]["password"]
        users_col.update_one(
            {"username": username},
            {"$set": {"password": hashed, "updated_at": datetime.utcnow()}}
        )
        st.success("Password updated successfully!")
        return True
    return False

def forgot_password(authenticator, config):
    result = authenticator.forgot_password(location="main")
    if result and len(result) == 3:
        username, email, new_password = result
        hashed = Hasher.hash_list([new_password])[0]

        users_col.update_one(
            {"username": username},
            {"$set": {"password": hashed, "updated_at": datetime.utcnow()}}
        )

        st.success(f"New password: {new_password}")
        st.info("Please login and change it immediately.")
        return True
    return False

def forgot_username(authenticator, config):
    result = authenticator.forgot_username(location="main")
    if result and len(result) == 2:
        username, _ = result
        if username:
            st.success(f"Your username is: {username}")
            return True
    return False