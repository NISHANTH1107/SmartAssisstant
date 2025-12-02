import os
import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# File processing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import pandas as pd
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# LangChain & FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

import google.generativeai as genai

load_dotenv()

DATA_DIR = Path("./data")

# Configure Gemini
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

genai.configure(api_key=GEMINI_KEY)

# Embedding model
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ===== USER-SPECIFIC DIRECTORIES =====
def get_user_dirs(username: str):
    user_dir = DATA_DIR / "users" / username
    return {
        'upload_dir': user_dir / "uploads",
        'index_dir': user_dir / "indexes",
        'chat_dir': user_dir / "chats"
    }

def init_user_dirs(username: str):
    dirs = get_user_dirs(username)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs

# ===== FILE EXTRACTION =====
def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page in reader.pages:
                txt = page.extract_text()
                if txt and txt.strip():
                    text_parts.append(txt)
            
            if not text_parts and OCR_AVAILABLE:
                try:
                    images = pdf2image.convert_from_path(str(file_path))
                    for img in images:
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            text_parts.append(ocr_text)
                except Exception as e:
                    text_parts.append(f"[OCR failed: {str(e)}]")
            
            return "\n\n".join(text_parts) if text_parts else "[No text extracted from PDF]"
        
        elif suffix == ".docx":
            doc = DocxDocument(str(file_path))
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        
        elif suffix == ".pptx":
            prs = Presentation(str(file_path))
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        texts.append(shape.text)
            return "\n\n".join(texts)
        
        elif suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            return df.to_string()
        
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] and OCR_AVAILABLE:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
        
    except Exception as e:
        return f"[Error extracting {file_path.name}: {str(e)}]"
    
    return ""

# ===== FILE MANAGEMENT =====
def save_uploaded_files(uploaded_files, chat_name: str, username: str) -> List[str]:
    saved = []
    dirs = init_user_dirs(username)
    chat_upload_dir = dirs['upload_dir'] / chat_name
    chat_upload_dir.mkdir(exist_ok=True)
    
    for file in uploaded_files:
        file_path = chat_upload_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(file.name)
    
    return saved

# ===== FAISS INDEX =====
def build_index_for_chat(chat_name: str, file_names: List[str], username: str):
    dirs = init_user_dirs(username)
    chat_upload_dir = dirs['upload_dir'] / chat_name
    
    all_texts = []
    for file_name in file_names:
        file_path = chat_upload_dir / file_name
        if file_path.exists():
            text = extract_text_from_file(file_path)
            all_texts.append(Document(
                page_content=text,
                metadata={"source": file_name}
            ))
    
    if not all_texts:
        return
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in all_texts:
        splits = splitter.split_text(doc.page_content)
        for split in splits:
            chunks.append(Document(
                page_content=split,
                metadata=doc.metadata
            ))
    
    # Build FAISS index
    if chunks:
        vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
        index_path = dirs['index_dir'] / chat_name
        vectorstore.save_local(str(index_path))

def load_index(chat_name: str, username: str):
    dirs = get_user_dirs(username)
    index_path = dirs['index_dir'] / chat_name
    if index_path.exists():
        try:
            return FAISS.load_local(str(index_path), EMBEDDINGS, allow_dangerous_deserialization=True)
        except:
            return None
    return None

# ===== GEMINI AI =====
def get_ai_response(query: str, chat_name: str, chat_history: List[Dict], username: str) -> str:  
    # Retrieve context from FAISS
    context_parts = []
    vectorstore = load_index(chat_name, username)
    
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=4)
        if docs:
            context_parts.append("=== RELEVANT CONTENT FROM YOUR FILES ===")
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"\n[Source: {doc.metadata.get('source', 'Unknown')}]")
                context_parts.append(doc.page_content)
    

    if len(chat_history) > 1: #last 6 messages
        context_parts.append("\n\n=== CONVERSATION HISTORY ===")
        recent = chat_history[-6:-1] 
        for msg in recent:
            role = msg['role'].upper()
            content = msg['content']
            context_parts.append(f"\n[{role}]: {content}")
    
    #prompt
    context = "\n".join(context_parts)
    
    prompt = f"""You are StudyMate, a helpful AI study assistant. Use the context below to answer the user's question accurately and clearly.
            {context}   USER QUESTION: {query}
            Provide a clear, well-structured answer. If the context doesn't contain the information, say so and provide a general answer based on your knowledge. Always be helpful and educational."""

    #Gemini
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# ===== QUIZ GENERATION =====
def generate_quiz_for_chat(chat_name: str, chat_history: List[Dict], username: str, num_questions: int = 5) -> List[Dict]:
    
    context_parts = []
    vectorstore = load_index(chat_name, username)
    
    if vectorstore:
        all_docs = vectorstore.similarity_search("", k=10)
        for doc in all_docs[:5]:  #top 5 chunks
            context_parts.append(doc.page_content)
    
    #last 10 messages
    for msg in chat_history[-10:]:
        if msg['role'] == 'user':
            context_parts.append(f"Q: {msg['content']}")
        elif msg['role'] == 'assistant' and 'quiz' not in msg:
            context_parts.append(f"A: {msg['content']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following study material and conversation, generate {num_questions} multiple-choice questions.

            MATERIAL:
            {context}

            Create {num_questions} questions in this EXACT JSON format:
            [
            {{
                "question": "Question text here?",
                "choices": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "B"
            }}
            ]

            Make questions educational and test understanding of key concepts. Ensure answer letters (A/B/C/D) match the choice positions."""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        response_text = response.text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        quiz = json.loads(response_text)
        return quiz[:num_questions] 
        
    except Exception as e:
        return [{
            "question": f"Error generating quiz: {str(e)}",
            "choices": ["Please try again"],
            "answer": "A"
        }]

# ===== CHAT MANAGEMENT =====
def list_chats(username: str) -> List[str]:
    dirs = get_user_dirs(username)
    chat_dir = dirs['chat_dir']
    if chat_dir.exists():
        return sorted([f.stem for f in chat_dir.glob("*.json")])
    return []

def save_chat(chat_name: str, messages: List[Dict], files: List[str], username: str):
    dirs = init_user_dirs(username)
    chat_path = dirs['chat_dir'] / f"{chat_name}.json"
    data = {
        "messages": messages,
        "files": files
    }
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_chat(chat_name: str, username: str) -> Dict:
    dirs = get_user_dirs(username)
    chat_path = dirs['chat_dir'] / f"{chat_name}.json"
    if chat_path.exists():
        with open(chat_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"messages": [], "files": []}

def delete_chat(chat_name: str, username: str):
    dirs = get_user_dirs(username)
    
    # Delete chat file
    chat_path = dirs['chat_dir'] / f"{chat_name}.json"
    if chat_path.exists():
        chat_path.unlink()
    
    # Delete uploaded files
    upload_path = dirs['upload_dir'] / chat_name
    if upload_path.exists():
        import shutil
        shutil.rmtree(upload_path)
    
    # Delete index
    index_path = dirs['index_dir'] / chat_name
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)

def remove_file_from_chat(chat_name: str, file_name: str, username: str):
    dirs = get_user_dirs(username)
    file_path = dirs['upload_dir'] / chat_name / file_name
    if file_path.exists():
        file_path.unlink()