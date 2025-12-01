
# StudyMate — AI Study Assistant

A lightweight, local-first study assistant that lets you upload documents (PDF, DOCX, PPTX, TXT, XLS/XLSX) and ask questions about them. It uses a FAISS-based retrieval index and an LLM (Google Gemini via `google-generativeai`) to provide context-aware answers and generate quizzes from your study materials.

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

---

## Requirements

- Python 3.10 or newer
- A valid Google Gemini API key (set as `GEMINI_API_KEY` in a `.env` file)

See `requirements.txt` for the Python packages used by the project.

---

## Quick start (Windows - cmd.exe)

1. Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```cmd
pip install -r requirements.txt
```

3. Add your Gemini API key to a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the app:

```cmd
streamlit run main.py
```

Open the displayed local URL in your browser. Upload files, ask questions, and save chats as needed.

---

## Configuration

- API key: The app expects `GEMINI_API_KEY` to be available via environment variables or a `.env` file (the project uses `python-dotenv`).
- Embeddings: The app uses a Hugging Face sentence-transformer model via `HuggingFaceEmbeddings`.
- Index storage: FAISS indexes and uploaded files are stored under `data/indexes/` and `data/uploads/` respectively.

Do not commit your `.env` file or API keys to version control.

---

## How it works (brief)

1. Uploaded documents are saved under `data/uploads/<chat_name>/`.
2. Documents are extracted to text, split into chunks, embedded, and stored in a FAISS index per chat.
3. When you ask a question, the app performs a similarity search to retrieve relevant chunks and includes recent chat history when building the LLM prompt.
4. The LLM (Gemini) generates the answer, and you can optionally generate a short quiz from the same context.

---

## Project structure

- `main.py` — Streamlit app and UI.
- `utils.py` — Core utilities: file extraction, FAISS index build/load, Gemini calls, chat persistence, and quiz generation.
- `requirements.txt` — Python dependencies.
- `data/` — Application data: `uploads/`, `indexes/`, and `chats/`.

---

## Troubleshooting

- GEMINI API key missing: If the app raises `GEMINI_API_KEY not found`, create a `.env` file or set the environment variable before starting Streamlit.
- File extraction errors: Some PDFs or office files may fail to extract; the app returns an extraction error string. Re-check the file or try exporting text manually.
- FAISS load/save issues: Index corruption or incompatible versions may cause failures when loading. Deleting the folder under `data/indexes/<chat_name>/` and rebuilding the index usually fixes the issue.
- Large documents: For very large documents, consider splitting them manually or increase chunk sizes in `utils.py` cautiously.

If you hit other issues, open an issue or inspect the logs printed by Streamlit for traceback details.

---

## Security & privacy notes

- Uploaded documents and generated indexes are stored locally under `data/`. If you share the repository, remove `data/` or sensitive files first.
- Keep API keys private and never commit `.env` files.

---


