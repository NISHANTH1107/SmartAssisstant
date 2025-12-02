import streamlit as st
from pathlib import Path
from utils import *
from auth import *

st.set_page_config(page_title="StudyMate AI", layout="wide", initial_sidebar_state="expanded")

# Get authenticator
authenticator, config = get_authenticator()

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
            st.session_state.current_chat = None
            st.session_state.messages = []
            st.session_state.chat_files = []
            st.rerun()
        
        st.markdown("---")
        st.subheader("Your Chats")
        
        # List existing chats
        chats = list_chats(username)
        if chats:
            for chat_name in chats:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"💬 {chat_name}", key=f"load_{chat_name}", use_container_width=True):
                        st.session_state.current_chat = chat_name
                        data = load_chat(chat_name, username)
                        st.session_state.messages = data['messages']
                        st.session_state.chat_files = data['files']
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{chat_name}"):
                        delete_chat(chat_name, username)
                        if st.session_state.current_chat == chat_name:
                            st.session_state.current_chat = None
                            st.session_state.messages = []
                            st.session_state.chat_files = []
                        st.rerun()
        else:
            st.info("No saved chats yet")
        
        st.markdown("---")
        st.caption("Upload files and ask questions to get started!")

    # ===== MAIN CHAT AREA =====
    st.title("👥 StudyMate - Your AI Study Assistant")

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
                        st.success(f"Chat saved as '{chat_name}'!")
                        st.rerun()
                    else:
                        st.error("Please enter a chat name")

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
                with st.spinner("Processing your files..."):
                    # Save files and build index
                    saved_files = save_uploaded_files(
                        uploaded_files, 
                        st.session_state.current_chat or "temp",
                        username
                    )
                    
                    # Only add new files
                    for f in saved_files:
                        if f not in st.session_state.chat_files:
                            st.session_state.chat_files.append(f)
                    
                    # Update FAISS index
                    build_index_for_chat(
                        st.session_state.current_chat or "temp",
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

    # Display uploaded files for current chat
    if st.session_state.chat_files:
        with st.expander(f"📚 Files in this chat ({len(st.session_state.chat_files)})", expanded=False):
            files_to_remove = []
            for idx, file in enumerate(st.session_state.chat_files):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text(f"• {file}")
                with col2:
                    if st.button("❌", key=f"remove_{idx}_{file}"):
                        files_to_remove.append(file)
            
            if files_to_remove:
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
            
            # Get AI response (user-specific)
            with st.chat_message("assistant"):
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

    # Quiz Generation Button (only show if there are messages)
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Generate Quiz from This Chat", use_container_width=True, type="secondary"):
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

# Not authenticated
else:
    with st.expander("📝 New user? Register here", expanded=False):
        register_new_user(authenticator, config)

    try:
        name, authentication_status, username = authenticator.login('main')
    except Exception as e:
        st.error(f"Login error: {e}")
        st.stop()

    # Handle authentication
    if authentication_status is False:
        st.error('Username/password is incorrect')

        with st.expander("📝 New User? Register Here"):
            register_new_user(authenticator, config)

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🔑 Forgot Password?"):
                forgot_password(authenticator, config)
        with col2:
            with st.expander("👤 Forgot Username?"):
                forgot_username(authenticator, config)

    elif authentication_status is None:
        st.warning('Please enter your username and password')
        with st.expander("📝 New User? Register Here"):
            register_new_user(authenticator, config)
    
    elif authentication_status is True:
        # Store in session state and rerun to show main app
        st.session_state['name'] = name
        st.session_state['username'] = username
        st.session_state['authentication_status'] = True
        st.rerun()