
# StudyMate — AI Study Assistant

A lightweight, local-first study assistant that lets you upload documents (PDF, DOCX, PPTX, TXT, XLS/XLSX) and ask questions about them. It uses a FAISS-based retrieval index and an LLM (Google  via `google-generativeai`) to provide context-aware answers and generate quizzes from your study materials.

Key goals:
- Fast, privacy-respecting document Q&A for study material
- Simple Streamlit UI for uploading files, saving chats, and generating quizzes
- Local FAISS index per chat for efficient retrieval

---

## Highlights

- Upload multiple file types (PDF, DOCX, PPTX, TXT, XLS/XLSX)
- Per-chat file management and local FAISS indices
- Context-aware responses using retrieved document chunks + recent chat history
- One-click quiz generation from conversation + documents
- **Semantic caching** for smart response reuse and optimized performance
- **Summary caching** for instant document overview retrieval

---

## Requirements

- Python 3.10 or newer
- A valid Google  API key (set as `API_KEY` in a `.env` file)

See `requirements.txt` for the Python packages used by the project.

## Tech Stack

| Layer | Tools / Technologies |
| --- | --- |
| Frontend | Streamlit (UI Rendering), Python |
| Backend / App Logic | Python, Streamlit Server, Custom Utility Modules (`utils.py`) |
| LLM Provider | Google Generative AI (`google-generativeai`) |
| RAG Pipeline | LangChain (document loaders, text splitting, regex-based chunking, retrieval pipeline) |
| Embeddings | HuggingFace Sentence Transformers (`sentence-transformers`) |
| Vector Index (Retrieval) | FAISS (Local, per-chat vector store) |
| Caching | Semantic Cache (smart response reuse), Summary Cache (document-level caching) |
| File Processing | PyPDF2, python-docx, python-pptx, pandas, xlrd/openpyxl |
| Storage | Local file storage under `data/` (`uploads/`, `indexes/`, `chats/`, `cache/`) |
| Environment Management | `python-dotenv` |
| Styling / UI | Streamlit Components |
| Language | Python 3.10+ |

---

## Quick start (Windows - cmd.exe)

1. Clone the repository to your local device:

```cmd
git clone https://github.com/NISHANTH1107/SmartAssisstant
```

2. Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```cmd
pip install -r requirements.txt
```

4. Add your  API key to a `.env` file in the project root:

```text
API_KEY = "your__api_key_here"
```

5. Run the app:

```cmd
streamlit run main.py
```

Open the displayed local URL in your browser. Upload files, ask questions, and save chats as needed.

---

## Configuration

- API key: The app expects `API_KEY` to be available via environment variables or a `.env` file (the project uses `python-dotenv`).
- Embeddings: The app uses a Hugging Face sentence-transformer model via `HuggingFaceEmbeddings`.
- Index storage: FAISS indexes and uploaded files are stored under `data/indexes/` and `data/uploads/` respectively.

Do not commit your `.env` file or API keys to version control.

---

## Project structure

- `main.py` — Streamlit app and UI.
- `utils.py` — Core utilities: file extraction, FAISS index build/load,  calls, chat persistence, and quiz generation.
- `requirements.txt` — Python dependencies.
- `data/` — Application data: `uploads/`, `indexes/`, and `chats/`.

---

## How it works (brief)

1. Uploaded documents are saved under `data/uploads/<chat_name>/`.
2. Documents are extracted to text, split into chunks, embedded, and stored in a FAISS index per chat.
3. When you ask a question, the app performs a similarity search to retrieve relevant chunks and includes recent chat history when building the LLM prompt.
4. The LLM generates the answer, and you can optionally generate a short quiz from the same context.

**Demo video:** [Watch the demonstration video](https://drive.google.com/file/d/1CmvM-Qu--0-OVfFT-FnVczgfqARUfgHC/view?usp=sharing)

---

## Semantic Caching (Smart Response Reuse)

This application implements semantic caching to optimize performance and reduce repeated LLM calls.

**What it does:**

When a user asks a question, the system checks whether a semantically similar question has already been asked earlier in the same chat.

If a close match is found, the cached response is reused instantly instead of calling the LLM again.

This significantly:

- Reduces API usage and cost
- Improves response speed
- Avoids generating duplicate answers for similar doubts

**Why it matters:**

Students often rephrase the same question multiple times. Semantic caching ensures the assistant responds efficiently without unnecessary recomputation.

---

## Summary Caching (Document-Level Optimization)

The app also applies summary caching for uploaded documents.

**How it works:**

When a file is uploaded and summarized for the first time, its summary is:

- Stored locally in the cache
- Reused whenever needed again in the future

If the same file is reopened or referenced again:

- The summary is loaded instantly
- The LLM is not called again

**Benefits:**

- Faster document overview
- Lower API usage
- Instant access to previously generated summaries

---

## Recommended Usage Guidelines (Very Important)

To avoid confusion between subjects, topics, and contexts, follow these best practices:

**Always create a separate chat for each subject or topic**

Example:
- One chat for Operating Systems
- One chat for Data Structures
- One chat for Machine Learning

**Use one chat per document set**

Do not mix unrelated files (e.g., Physics + History) into the same chat, for better clarity and avoiding turbid/cloudy responses.

---

## Troubleshooting

-  API key missing: If the app raises `API_KEY not found`, create a `.env` file or set the environment variable before starting Streamlit.
- File extraction errors: Some PDFs or office files may fail to extract; the app returns an extraction error string. Re-check the file or try exporting text manually.
- FAISS load/save issues: Index corruption or incompatible versions may cause failures when loading. Deleting the folder under `data/indexes/<chat_name>/` and rebuilding the index usually fixes the issue.
- Large documents: For very large documents, consider splitting them manually or increase chunk sizes in `utils.py` cautiously.

If you hit other issues, open an issue (PR's will be actively reviewed) or inspect the logs printed by Streamlit for traceback details.


---

## Security & privacy notes

- Uploaded documents and generated indexes are stored locally under `data/`. If you share the repository, remove `data/` or sensitive files first.
- Keep API keys private and never commit `.env` files.

---

