import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta
import shutil

# File processing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import pandas as pd

# Wikipedia search
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# LangChain & FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import google.generativeai as genai

load_dotenv()

DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configure Gemini
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

genai.configure(api_key=GEMINI_KEY)

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ===== SEMANTIC CACHE =====
class OptimizedSemanticCache:
    
    def __init__(self, cache_dir: Path = CACHE_DIR, expiry_hours: int = 24, similarity_threshold: float = 0.85):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "semantic_cache.json"
        self.expiry_hours = expiry_hours
        self.similarity_threshold = similarity_threshold
        self.embeddings = EMBEDDINGS  

        # In-memory caches for speed
        self.cache = self._load_cache()
        self._embedding_cache = {}  
        self._context_groups = {}    
        
        # Build context groups on init
        self._rebuild_context_groups()
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        text_key = text.lower().strip()
        
        # Check memory cache first
        if text_key in self._embedding_cache:
            return self._embedding_cache[text_key]
        
        embedding = np.array(self.embeddings.embed_query(text_key))
        
        
        if len(self._embedding_cache) < 1000:  # Limit to 1000 embeddings
            self._embedding_cache[text_key] = embedding
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _is_expired(self, timestamp: str) -> bool:
        try:
            cached_time = datetime.fromisoformat(timestamp)
            return datetime.now() - cached_time > timedelta(hours=self.expiry_hours)
        except:
            return True
    
    def _create_context_hash(self, context: str) -> str:
        context_key = context[:500].lower().strip()
        return hashlib.md5(context_key.encode()).hexdigest()[:8]  # Use first 8 chars for speed
    
    def _rebuild_context_groups(self):
        self._context_groups.clear()
        
        for cache_key, entry in self.cache.items():
            context_hash = entry.get('context_hash', 'default')
            if context_hash not in self._context_groups:
                self._context_groups[context_hash] = []
            self._context_groups[context_hash].append((cache_key, entry))
    
    def get(self, query: str, context: str = "") -> Optional[Tuple[str, float]]:
        query_embedding = self._get_embedding(query)
        context_hash = self._create_context_hash(context)
        
        # Get relevant entries from context group only
        relevant_entries = self._context_groups.get(context_hash, [])
        
        if not relevant_entries:
            return None
        
        best_match = None
        best_similarity = 0.0
        expired_keys = []
        
        # Batch process all embeddings in the group
        for cache_key, entry in relevant_entries:
            if self._is_expired(entry['timestamp']):
                expired_keys.append(cache_key)
                continue
            
            # Get cached embedding
            cached_embedding = entry.get('embedding')
            if cached_embedding is None:
                continue
            
            cached_embedding = np.array(cached_embedding)
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            # Perfect match
            if similarity > 0.99:
                return (entry['response'], similarity)
            
            # Track best match
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry['response']
        
        # Clean up expired entries
        if expired_keys:
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
            self._save_cache()
            self._rebuild_context_groups()
        
        return (best_match, best_similarity) if best_match else None
    
    def set(self, query: str, response: str, context: str = ""):
        # Generate unique key
        cache_key = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Get embedding
        query_embedding = self._get_embedding(query)
        context_hash = self._create_context_hash(context)
        
        # Store with list format for JSON serialization
        self.cache[cache_key] = {
            'query': query,
            'response': response,
            'embedding': query_embedding.tolist(),  # Convert numpy to list
            'context_hash': context_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update context groups
        if context_hash not in self._context_groups:
            self._context_groups[context_hash] = []
        self._context_groups[context_hash].append((cache_key, self.cache[cache_key]))
        
        # Save to disk
        self._save_cache()
        
        if len(self.cache) > 500:  # Max 500 cached queries
            self._trim_cache()
    
    def _trim_cache(self):
        # Sort by timestamp and keep newest 400
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        self.cache = dict(sorted_entries[:400])
        self._save_cache()
        self._rebuild_context_groups()
    
    def clear_expired(self) -> int:
        expired_keys = [
            key for key, value in self.cache.items()
            if self._is_expired(value['timestamp'])
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._save_cache()
            self._rebuild_context_groups()
            # Clear memory cache to free space
            self._embedding_cache.clear()
        
        return len(expired_keys)

# Global cache instance
semantic_cache = OptimizedSemanticCache()

# ===== WIKIPEDIA SEARCH =====
def search_wikipedia(query: str, sentences: int = 3) -> str:
    if not WIKIPEDIA_AVAILABLE:
        return ""
    
    try:
        # Search for relevant pages
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return ""
        
        # Get summary of the most relevant page
        try:
            summary = wikipedia.summary(search_results[0], sentences=sentences)
            return f"\n\n=== WIKIPEDIA INFORMATION ===\nTopic: {search_results[0]}\n{summary}\n"
        except wikipedia.exceptions.DisambiguationError as e:
            # If disambiguation, try the first option
            try:
                summary = wikipedia.summary(e.options[0], sentences=sentences)
                return f"\n\n=== WIKIPEDIA INFORMATION ===\nTopic: {e.options[0]}\n{summary}\n"
            except:
                return ""
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""

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

def cleanup_temp_data(username: str):
    dirs = get_user_dirs(username)
    
    # Clean temp uploads
    temp_upload = dirs['upload_dir'] / "temp"
    if temp_upload.exists():
        shutil.rmtree(temp_upload)
    
    # Clean temp index
    temp_index = dirs['index_dir'] / "temp"
    if temp_index.exists():
        shutil.rmtree(temp_index)

def move_temp_to_chat(chat_name: str, username: str):
    dirs = get_user_dirs(username)
    
    # Move uploads
    temp_upload = dirs['upload_dir'] / "temp"
    chat_upload = dirs['upload_dir'] / chat_name
    if temp_upload.exists():
        if chat_upload.exists():
            # Merge files
            for file in temp_upload.iterdir():
                if file.is_file():
                    shutil.copy2(file, chat_upload / file.name)
            shutil.rmtree(temp_upload)
        else:
            temp_upload.rename(chat_upload)
    
    # Move index
    temp_index = dirs['index_dir'] / "temp"
    chat_index = dirs['index_dir'] / chat_name
    if temp_index.exists():
        if chat_index.exists():
            shutil.rmtree(chat_index)
        temp_index.rename(chat_index)

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
                except Exception:
                    pass
            
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

# ===== GEMINI AI WITH OPTIMIZED SEMANTIC CACHING =====
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
    
    # Add conversation history
    if len(chat_history) > 1:
        context_parts.append("\n\n=== CONVERSATION HISTORY ===")
        recent = chat_history[-6:-1] 
        for msg in recent:
            role = msg['role'].upper()
            content = msg['content']
            context_parts.append(f"\n[{role}]: {content}")
    
    context = "\n".join(context_parts)
    
    # Check semantic cache first (now truly semantic!)
    cache_result = semantic_cache.get(query, context)
    if cache_result:
        response, similarity = cache_result
        return f"{response}\n\n*[Cached response - {similarity:.1%} match]*"
    
    # Search Wikipedia if no files context
    wiki_info = ""
    if not vectorstore or not context_parts:
        wiki_info = search_wikipedia(query, sentences=3)
        if wiki_info:
            context_parts.append(wiki_info)
            context = "\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are StudyMate, a helpful AI study assistant. Use the context below to answer the user's question accurately and clearly.

            {context}

            USER QUESTION: {query}

            Provide a clear, well-structured answer. If the context doesn't contain the information, say so and provide a general answer based on your knowledge. Always be helpful and educational."""

    # Call Gemini
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        result = response.text
        
        # Cache the response with semantic matching
        semantic_cache.set(query, result, context)
        
        return result
    except Exception as e:
        return f"I encountered an error while processing your request. Please try again."

# ===== QUIZ GENERATION =====
def generate_quiz_for_chat(chat_name: str, chat_history: List[Dict], username: str, num_questions: int = 5) -> List[Dict]:
    context_parts = []
    vectorstore = load_index(chat_name, username)
    
    if vectorstore:
        all_docs = vectorstore.similarity_search("", k=10)
        for doc in all_docs[:5]:
            context_parts.append(doc.page_content)
    
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
        
        response_text = response.text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        quiz = json.loads(response_text)
        return quiz[:num_questions]
        
    except Exception:
        return [{
            "question": "Unable to generate quiz at this time. Please try again.",
            "choices": ["Try again later"],
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
    # Move temp data chat
    move_temp_to_chat(chat_name, username)
    
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
        shutil.rmtree(upload_path)
    
    # Delete index
    index_path = dirs['index_dir'] / chat_name
    if index_path.exists():
        shutil.rmtree(index_path)

def remove_file_from_chat(chat_name: str, file_name: str, username: str):
    dirs = get_user_dirs(username)
    file_path = dirs['upload_dir'] / chat_name / file_name
    if file_path.exists():
        file_path.unlink()