# AgriScheme AI — FastAPI + Streamlit (Windows)

## Architecture

- `frontend.py`: Streamlit agriculture-themed interface. It uses HTTP requests to FastAPI, never FAISS directly.
- `backend.py`: FastAPI endpoints for upload, delete, PDF access, hybrid FAISS + BM25 retrieval, query translation, and evidence-grounded OpenAI answers.
- `data/`: Automatically created persistent local original PDFs, FAISS index, chunk records, and manifest.
- **No web search or web crawling is implemented.** OpenAI API calls still require internet connectivity and transmit PDF text during indexing / retrieved evidence for answers.

## Install (Windows Command Prompt)

Extract the ZIP to, for example:

`C:\Users\Admin\Documents\Gen AI Program\tn-agri-scheme-ai`

```bat
cd "C:\Users\Admin\Documents\Gen AI Program\tn-agri-scheme-ai"
```

To reuse your existing environment:

```bat
"C:\Users\Admin\Documents\Gen AI Program\venv\Scripts\activate"
python -m pip install -r requirements.txt
```

Or create an isolated environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place your existing `.env` in this project directory:

```env
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AGRISCHEME_API_URL=http://127.0.0.1:8000
```

Do not commit `.env` or `data/` to GitHub.

## Start two terminals

**Terminal 1 — backend:**

```bat
cd "C:\Users\Admin\Documents\Gen AI Program\tn-agri-scheme-ai"
.venv\Scripts\activate
python -m uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend:**

```bat
cd "C:\Users\Admin\Documents\Gen AI Program\tn-agri-scheme-ai"
.venv\Scripts\activate
python -m streamlit run frontend.py
```

If you activated the shared `venv` instead, activate that same environment in both terminals instead of `.venv`.

Open Streamlit at http://localhost:8501 and Swagger docs at http://127.0.0.1:8000/docs.

## First use

1. Upload 2–5 official text-based government PDFs (Tamil or English).
2. Click **Index selected PDFs**. The originals, vectors, chunk text and manifest are stored in `data/`.
3. Ask a question in Tamil or English. Set language to **Auto** to answer in the question's language.
4. Expand evidence diagnostics. Citations include original PDF filename, page and chunk IDs.
5. Restart both servers: saved PDFs and FAISS index remain; re-embedding is not required.
6. Add additional PDFs later; SHA-256 hashes prevent duplicate embeddings.

## REST endpoints

- `GET /health` — status and index counts
- `GET /documents` — list PDFs
- `POST /documents/upload` — multipart `files` upload
- `DELETE /documents/{doc_id}` — remove PDF from saved library and FAISS
- `GET /documents/{doc_id}/pdf` — view saved PDF
- `POST /search` — hybrid retrieval diagnostics
- `POST /chat` — retrieve and answer, with citation evidence

Example `POST /chat` body:

```json
{"question":"What support is available for micro-irrigation?","language":"English","top_k":7}
```

## Notes and limits

- Scanned or image-only PDFs require OCR (not included).
- Tamil retrieval uses FAISS embeddings plus optional English query translation; Tamil keyword tokenization is basic, so benchmark retrieval with Tamil test questions before launch.
- RRF values are not confidence probabilities.
- PDFs are untrusted data; never import downloaded FAISS index folders because deserialization is unsafe.
- This local buildathon backend has no authentication. Bind it only to `127.0.0.1`, **not** public interfaces. Add auth, rate limits, and storage transactions before production deployment.
- The UI conversation history is stored only in the Streamlit session. PDFs, chunks, and FAISS index are persistent.
- Local `data/` persists on your Windows disk; ephemeral cloud hosts need mounted persistent storage.
