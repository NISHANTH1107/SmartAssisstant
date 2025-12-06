import streamlit as st
from pathlib import Path
from utils import *
from auth import *
import sys
from io import StringIO

# Custom error handler to suppress verbose Streamlit errors
class GracefulErrorHandler:
    def __init__(self):
        self.error_occurred = False
    
    def handle_error(self, error_msg: str):
        if not self.error_occurred:
            st.error(f"⚠️ {error_msg}")
            self.error_occurred = True

error_handler = GracefulErrorHandler()

# Configure page
st.set_page_config(
    page_title="StudyMate AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize authenticator with error handling
try:
    authenticator, config = get_authenticator()
except Exception as e:
    st.error("⚠️ Authentication system initialization failed. Please check your configuration.")
    st.stop()

# Check if already authenticated
if st.session_state.get('authentication_status') is True:
    name = st.session_state.get('name')
    username = st.session_state.get('username')
    
    # Initialize session state
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_files' not in st.session_state:
        st.session_state.chat_files = []
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True
        # Clean up any temp data from previous sessions
        cleanup_temp_data(username)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.title(f"🎓 Welcome, {name}!")
        
        authenticator.logout('Logout', 'sidebar')
        
        st.markdown("---")
        
        # Account Settings in sidebar
        with st.expander("⚙️ Account Settings", expanded=False):
            reset_password(authenticator, config, username)
        
        st.markdown("---")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            cleanup_temp_data(username)
            st.session_state.current_chat = None
            st.session_state.messages = []
            st.session_state.chat_files = []
            st.rerun()
        
        st.markdown("---")
        st.subheader("Your Chats")

        chats = list_chats(username)
        current = st.session_state.get("current_chat")

        if chats:
            for chat_name in chats:
                is_selected = (chat_name == current)

                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button(
                        f"💬 {chat_name}",
                        key=f"load_{chat_name}",
                        use_container_width=True,
                        type=("primary" if is_selected else "secondary")
                    ):
                        cleanup_temp_data(username)
                        st.session_state.current_chat = chat_name
                        data = load_chat(chat_name, username)
                        st.session_state.messages = data['messages']
                        st.session_state.chat_files = data['files']
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"del_{chat_name}"):
                        delete_chat(chat_name, username)
                        if current == chat_name:
                            cleanup_temp_data(username)
                            st.session_state.current_chat = None
                            st.session_state.messages = []
                            st.session_state.chat_files = []
                        st.rerun()
        else:
            st.info("No saved chats yet")

        
        st.markdown("---")
        st.caption("📚 Upload files and ask questions!")
        
        # Clear cache button
        if st.button("🔄 Clear Response Cache", use_container_width=True):
            semantic_cache.clear_expired()
            st.success("Cache cleared!")

    # ===== MAIN CHAT AREA =====
    st.title("💬 StudyMate - Your AI Study Assistant")
    
    # Info banner
    if WIKIPEDIA_AVAILABLE:
        st.info("💡 I can answer from your uploaded files, our conversation history, and search Wikipedia for general knowledge!")
    else:
        st.info("💡 I can answer from your uploaded files and our conversation history!")

    # Chat name input for new chats
    if st.session_state.current_chat is None and len(st.session_state.messages) > 0:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                chat_name = st.text_input("💾 Save this chat as:", placeholder="e.g., Biology Chapter 5")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Save Chat", type="primary"):
                    if chat_name.strip():
                        st.session_state.current_chat = chat_name.strip()
                        save_chat(
                            st.session_state.current_chat,
                            st.session_state.messages,
                            st.session_state.chat_files,
                            username
                        )
                        st.success(f"✅ Chat saved as '{chat_name}'!")
                        st.rerun()
                    else:
                        st.error("⚠️ Please enter a chat name")

    # File Upload Area
    with st.expander("📎 Upload Files (PDF, DOCX, PPTX, TXT, XLS, XLSX, Images)", expanded=not st.session_state.messages):
        uploaded_files = st.file_uploader(
            "Drop your study materials here (includes OCR for scanned documents)",
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "txt", "xls", "xlsx", "jpg", "jpeg", "png", "bmp", "tiff"],
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📥 Process Files", type="primary"):
                try:
                    with st.spinner("Processing your files..."):
                        # Save files to temp or current chat
                        chat_context = st.session_state.current_chat or "temp"
                        saved_files = save_uploaded_files(
                            uploaded_files, 
                            chat_context,
                            username
                        )
                        
                        # Only add new files
                        for f in saved_files:
                            if f not in st.session_state.chat_files:
                                st.session_state.chat_files.append(f)
                        
                        # Update FAISS index
                        build_index_for_chat(
                            chat_context,
                            st.session_state.chat_files,
                            username
                        )
                        
                        # Save if chat exists
                        if st.session_state.current_chat:
                            save_chat(
                                st.session_state.current_chat,
                                st.session_state.messages,
                                st.session_state.chat_files,
                                username
                            )
                        
                        st.success(f"✅ Processed {len(saved_files)} file(s)!")
                        st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error processing files. Please try again.")

    # Display uploaded files for current chat
    if st.session_state.chat_files:
        with st.expander(f"📚 Files in this chat ({len(st.session_state.chat_files)})", expanded=False):
            files_to_remove = []
            for idx, file in enumerate(st.session_state.chat_files):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text(f"• {file}")
                with col2:
                    if st.button("x", key=f"remove_{idx}_{file}"):
                        files_to_remove.append(file)
            
            if files_to_remove:
                try:
                    for file in files_to_remove:
                        st.session_state.chat_files.remove(file)
                        remove_file_from_chat(
                            st.session_state.current_chat or "temp",
                            file,
                            username
                        )
                    
                    # Rebuild index
                    if st.session_state.chat_files:
                        build_index_for_chat(
                            st.session_state.current_chat or "temp",
                            st.session_state.chat_files,
                            username
                        )
                    
                    # Save changes
                    if st.session_state.current_chat:
                        save_chat(
                            st.session_state.current_chat,
                            st.session_state.messages,
                            st.session_state.chat_files,
                            username
                        )
                    
                    st.success(f"🗑️ Removed {len(files_to_remove)} file(s)")
                    st.rerun()
                except Exception as e:
                    st.error("⚠️ Error removing files. Please try again.")

    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display quiz if exists
            if message["role"] == "assistant" and "quiz" in message:
                with st.expander("📝 Quiz Questions", expanded=True):
                    for i, q in enumerate(message["quiz"], 1):
                        st.markdown(f"**Q{i}. {q['question']}**")
                        for j, choice in enumerate(q['choices']):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;{chr(65+j)}. {choice}")
                        st.success(f"**Answer: {q['answer']}**")
                        if i < len(message["quiz"]):
                            st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask me anything about your study materials..."):
        # Check if we have files or chat context
        if not st.session_state.chat_files and not st.session_state.messages:
            st.warning("⚠️ Please upload some files first to get started!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        response = get_ai_response(
                            prompt,
                            st.session_state.current_chat or "temp",
                            st.session_state.messages,
                            username
                        )
                        st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save chat if it exists
                    if st.session_state.current_chat:
                        save_chat(
                            st.session_state.current_chat,
                            st.session_state.messages,
                            st.session_state.chat_files,
                            username
                        )
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error("⚠️ I encountered an error. Please try rephrasing your question.")

    # Quiz Generation Button
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Generate Quiz from This Chat", use_container_width=True, type="secondary"):
                try:
                    with st.spinner("Generating quiz questions..."):
                        quiz = generate_quiz_for_chat(
                            st.session_state.current_chat or "temp",
                            st.session_state.messages,
                            username
                        )
                        
                        # Add quiz to messages
                        quiz_message = {
                            "role": "assistant",
                            "content": f"📝 **Quiz Generated!** Here are {len(quiz)} questions based on our conversation:",
                            "quiz": quiz
                        }
                        st.session_state.messages.append(quiz_message)
                        
                        # Save if chat exists
                        if st.session_state.current_chat:
                            save_chat(
                                st.session_state.current_chat,
                                st.session_state.messages,
                                st.session_state.chat_files,
                                username
                            )
                        
                        st.rerun()
                except Exception as e:
                    st.error("⚠️ Unable to generate quiz. Please try again.")

# Not authenticated - LOGIN PAGE
else:
    # Suppress registration success messages when page loads
    if 'registration_attempted' not in st.session_state:
        st.session_state.registration_attempted = False
    
    st.title("🎓 Welcome to StudyMate AI")
    st.markdown("### Your Personal AI Study Assistant")
    
    # Login section
    st.markdown("---")
    st.subheader("🔐 Login")
    
    try:
        # Attempt login
        login_result = authenticator.login('main')
        
        # Handle different return formats
        if login_result is not None and len(login_result) == 3:
            name, authentication_status, username = login_result
        else:
            # Login form was just displayed, not submitted
            name = None
            authentication_status = None
            username = None
        
    except Exception as e:
        # Graceful error handling for login issues
        st.error("⚠️ Login system error. Please refresh the page.")
        st.stop()

    # Handle authentication results
    if authentication_status is False:
        st.error('❌ Username/password is incorrect')

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🔑 Forgot Password?"):
                forgot_password(authenticator, config)
        with col2:
            with st.expander("👤 Forgot Username?"):
                forgot_username(authenticator, config)

    elif authentication_status is None:
        st.info('👆 Please enter your username and password above')
    
    elif authentication_status is True:
        # Store in session state and rerun
        st.session_state['name'] = name
        st.session_state['username'] = username
        st.session_state['authentication_status'] = True
        st.rerun()
    
    # Registration section
    st.markdown("---")
    with st.expander("📝 New User? Register Here", expanded=False):
        if not st.session_state.registration_attempted:
            st.session_state.registration_attempted = True
        register_new_user(authenticator, config)
    
    # Footer
    st.markdown("---")
    st.caption("💡 Demo credentials: username: `demo_user` | password: `demo123`")