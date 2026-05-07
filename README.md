# RAG ADK Project

Document intelligence agent — ask natural language questions about your PDFs.
Built with Google ADK, Gemini, Qdrant, and deployed to Cloud Run.

---

## Stack

| Requirement  | Technology                                      |
|--------------|-------------------------------------------------|
| ADK          | `google-adk` — `root_agent` + `search_docs`    |
| Gen AI       | `google.genai` — Gemini 2.5 Flash + Embedding  |
| Gemini API   | ADC via `google.genai` client                  |
| Cloud Run    | `adk deploy cloud_run --with_ui`               |
| UI           | ADK web UI bundled at Cloud Run URL            |
| Vector store | Qdrant `rag_adk_collection`                    |
| PDF parsing  | `pypdf` — pure Python, no GCP auth             |

---

## Project structure

```
rag_adk_project/
├── rag_agent/               ← ADK agent (deployed to Cloud Run)
│   ├── __init__.py
│   ├── agent.py
│   ├── requirements.txt
│   └── tools/
│       ├── __init__.py
│       └── search_docs.py
│
├── pipeline/                ← standalone ingestion (run locally once)
│   ├── ingest.py
│   └── requirements.txt
│
├── .env                     ← secrets — never commit this
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python 3.11 or higher
- Google Cloud SDK (`gcloud`) installed
- A GCP project with Vertex AI API enabled
- A Qdrant Cloud account and cluster

---

## Setup (Windows — PowerShell)

**Step 1 — Create virtual environment:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If you get a permissions error run this first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Step 2 — Install dependencies:**
```powershell
pip install -r rag_agent\requirements.txt
pip install -r pipeline\requirements.txt
```

**Step 3 — Authenticate with GCP:**
```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project rag-adk-495601
gcloud auth application-default set-quota-project rag-adk-495601
```

**Step 4 — Create your `.env` file** in the project root:
```
GOOGLE_CLOUD_PROJECT=rag-adk-495601
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
GCP_PROJECT_ID=rag-adk-495601
GCP_LOCATION=us-central1
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=rag_adk_collection
```

---

## Step 1 — Ingest your documents (run once)

Point the ingestion script at a folder containing your PDF files:

```powershell
python pipeline\ingest.py --folder C:\path\to\your\pdfs
```

This will:
- Extract text from each PDF page by page using `pypdf`
- Embed each page using `gemini-embedding-001` via Vertex AI
- Store vectors and metadata in Qdrant under `rag_adk_collection`

You will see progress like:
```
Found 5 PDF(s)
Ingesting: document.pdf
  Extracted 10 pages
  Embedded pages 1-5
  Embedded pages 6-10
  Stored 10 pages in Qdrant ✓
Done — 5 document(s) indexed
```

---

## Step 2 — Run the agent locally

```powershell
adk web
```

Open `http://localhost:8000` in your browser.
Select `rag_agent` from the dropdown and ask any question.

The agent calls `search_docs`, searches Qdrant, and returns
a grounded answer with source page and file references.

---

## Step 3 — Deploy to Cloud Run

**Deploy the agent:**
```powershell
adk deploy cloud_run `
  --project=rag-adk-495601 `
  --region=us-central1 `
  --service_name=rag-document-agent `
  --with_ui `
  rag_agent
```

**Set environment variables on Cloud Run** (run after deploy):
```powershell
gcloud run services update rag-document-agent `
  --region=us-central1 `
  --project=rag-adk-495601 `
  --set-env-vars `
GCP_PROJECT_ID=rag-adk-495601,`
GCP_LOCATION=us-central1,`
GOOGLE_GENAI_USE_VERTEXAI=true,`
QDRANT_COLLECTION=rag_adk_collection,`
QDRANT_URL=your-qdrant-cluster-url,`
QDRANT_API_KEY=your-qdrant-api-key
```

**Grant Vertex AI permission to Cloud Run service account:**
```powershell
gcloud projects add-iam-policy-binding rag-adk-495601 `
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" `
  --role="roles/aiplatform.user"
```

Open the Cloud Run URL — the ADK web UI is bundled and ready.

---

## How it works

```
User question
     ↓
ADK root_agent (Gemini 2.5 Flash)
     ↓
search_docs tool
     ↓
gemini-embedding-001 — embeds the question into a vector
     ↓
Qdrant — finds the most semantically relevant document pages
     ↓
gemini-2.5-flash — generates a grounded answer from the pages
     ↓
Answer + source page references back to user
```

---

## Troubleshooting

**PowerShell execution policy error:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**ADC quota project warning:**
```powershell
gcloud auth application-default set-quota-project rag-adk-495601
```

**`qdrant_client` has no attribute `search`:**
Use `query_points()` instead — newer versions of `qdrant-client` changed the API.

**Agent not responding on Cloud Run:**
Make sure environment variables are set on the Cloud Run service
and the service account has `roles/aiplatform.user`.