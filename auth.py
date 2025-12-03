import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
from pathlib import Path

# Configuration file path
CONFIG_FILE = Path("./config.yaml")

def init_config():
    if not CONFIG_FILE.exists():
        # Create default config with hashed passwords
        config = {
            'credentials': {
                'usernames': {
                    'demo_user': {
                        'email': 'demo@example.com',
                        'name': 'Demo User',
                        'password': Hasher.hash_list(['demo123'])[0]
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'studymate_signature_key',
                'name': 'studymate_cookie'
            },
            'preauthorized': {
                'emails': []
            }
        }
        
        with open(CONFIG_FILE, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    
    # Load config
    with open(CONFIG_FILE) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def get_authenticator():
    config = init_config()
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    return authenticator, config

def register_new_user(authenticator, config):
    """Fixed registration with proper error handling"""
    try:
        # Try new API first
        result = authenticator.register_user(
            location='main',
            fields={
                'Form name': 'Register User',
                'Email': 'Email',
                'Username': 'Username',
                'Password': 'Password',
                'Repeat password': 'Repeat password',
                'Register': 'Register',
            }
        )
        
        # Handle different return types
        if result is True:
            save_config(config)
            st.success('User registered successfully! Please login.')
            return True
        elif result is False:
            # Registration form shown but not submitted yet
            return False
        else:
            # None or other values - form shown
            return False
            
    except TypeError:
        # Try old API
        try:
            result = authenticator.register_user(location='main')
            if result is True:
                save_config(config)
                st.success('User registered successfully! Please login.')
                return True
            return False
        except Exception as e:
            st.error(f"Registration failed: {str(e)}")
            return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def reset_password(authenticator, config, username):
    try:
        result = authenticator.reset_password(username, location='main')
        if result is True:
            save_config(config)
            st.success('Password modified successfully!')
            return True
    except Exception as e:
        st.error(f'Password reset failed: {str(e)}')
    return False

def forgot_password(authenticator, config):
    try:
        result = authenticator.forgot_password(location='main')
        if result and len(result) == 3:
            username, email, random_password = result
            if username:
                save_config(config)
                st.success(f'New password: {random_password}')
                st.info('Please use this password to login and change it immediately.')
                return True
            elif username is False:
                st.error('Username not found')
        return False
    except Exception as e:
        st.error(f'Error: {str(e)}')
        return False

def forgot_username(authenticator, config):
    try:
        result = authenticator.forgot_username(location='main')
        if result and len(result) == 2:
            username, email = result
            if username:
                st.success(f'Your username is: {username}')
                return True
            elif username is False:
                st.error('Email not found')
        return False
    except Exception as e:
        st.error(f'Error: {str(e)}')
        return False

def update_user_details(authenticator, config, username):
    try:
        result = authenticator.update_user_details(username, location='main')
        if result is True:
            save_config(config)
            st.success('Details updated successfully!')
            return True
    except Exception as e:
        st.error(f'Update failed: {str(e)}')
    return False