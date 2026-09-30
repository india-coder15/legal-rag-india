# Legal RAG System — Indian Labour Laws

A production-level Retrieval-Augmented Generation (RAG) system for querying Indian labour laws. Uses hybrid vector + keyword search to find relevant laws and sections, then generates precise legal answers via a self-hosted LLM. Exposes a REST API for MERN frontend integration.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| REST API | FastAPI + Uvicorn |
| UI (optional) | Gradio |
| Database | SQLite + FTS5 |
| Vector Search | FAISS (BAAI/bge-small-en-v1.5 — 384 dimensions) |
| LLM | Self-hosted Gemma 4 (gemma-4-31b-it, 65k context) |
| Data Source | Google Sheets (via Service Account) |
| Language | Python 3.10+ |

---

## Project Structure

```
legal-rag-india/
│
├── run_api.py                # Start the REST API server (port 8080)
├── run.py                    # Start the Gradio UI (port 7860)
├── run_ingestion.py          # Pull from Google Sheets → SQLite + FAISS
├── requirements.txt
├── .env.example              # Copy to .env and fill in your values
│
├── app/
│   └── gradio_app.py         # Gradio UI (optional frontend)
│
├── backend/
│   ├── config.py             # All configuration constants
│   ├── api/
│   │   └── routes.py         # FastAPI REST endpoints
│   ├── embeddings/
│   │   ├── encoder.py        # Embedding model loader (BAAI/bge-small-en-v1.5)
│   │   ├── faiss_index.py    # FAISS index build, save, load, parallel search
│   │   └── text_prep.py      # Prepares law/section text for embedding
│   ├── ingestion/
│   │   ├── excel_reader.py   # Reads 6 Google Sheet tabs into Python dicts
│   │   ├── db_writer.py      # Writes to SQLite, builds FTS5 indexes
│   │   └── batch_processor.py # Orchestrates ingestion + embedding indexing
│   ├── llm/
│   │   └── mistral_client.py # LiteLLM wrapper for self-hosted Gemma LLM
│   ├── logging/
│   │   ├── logger.py         # Central logger
│   │   └── query_logger.py   # Query history (SQLite)
│   └── retrieval/
│       ├── pipeline.py       # Full query pipeline (entry point)
│       ├── doc_selector.py   # Law selection: FAISS + FTS5 + LLM
│       └── section_selector.py # Section selection: FAISS + keyword + LLM
│
└── database/                 # Generated locally — not in GitHub
    ├── legal_db.sqlite       # Built by run_ingestion.py
    └── faiss/                # Built by run_ingestion.py
        ├── laws.index
        └── sections.index
```

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- Google Service Account credentials JSON file (get from team on WhatsApp)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Vinay152003/legal-rag-india
cd legal-rag-india
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: `sentence-transformers` installs PyTorch (~2GB). First install may take a few minutes.

### Step 4 — Add Google Credentials File

Get the file `abyd-legal-ai-f489d309e52d.json` from the team on **WhatsApp** and place it at the **root of the project folder** (same level as `run_api.py`).

```
legal-rag-india/
├── abyd-legal-ai-f489d309e52d.json   ← place here
├── run_api.py
├── run_ingestion.py
...
```

> This file is a Google Service Account key that allows reading the legal dataset from Google Sheets. It is never committed to GitHub.

### Step 5 — Configure Environment

Copy the example env file:

```bash
# Mac / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` and set your values:

```
GOOGLE_SHEET_ID=1GLA_Dy5p8h5DKucN8FWmzqfmmYWexkmJfTQCdgyLvz4
GOOGLE_CREDS_PATH=abyd-legal-ai-f489d309e52d.json
LLM_ENDPOINT=http://44.223.191.230:8000
LLM_MODEL=gemma-4-31b-it
MOCK_LLM=false
```

> `GOOGLE_SHEET_ID` and `GOOGLE_CREDS_PATH` are already set correctly in `.env.example` — no changes needed unless your setup differs.

### Step 6 — Run Ingestion

Pulls all data from Google Sheets → SQLite + FAISS indices:

```bash
python run_ingestion.py
```

Expected output:
```
Source  : Google Sheets (ID: 1GLA_Dy5p8h5DKucN8FWmzqfmmYWexkmJfTQCdgyLvz4)
Database: .../database/legal_db.sqlite
FAISS   : .../database/faiss

=== Legal Dataset Ingestion ===
  Read: 179 laws, 4443 sections, 1587 definitions ...
  Laws       ████████ 100%
  Sections   ████████ 100%
  ...
=== Ingestion Complete ===

=== Embedding Indexing ===
  Law FAISS index built: 179 vectors
  Section FAISS index built: 4443 vectors
=== Embedding Indexing Complete ===
```

> Re-run whenever the Google Sheet data is updated.

### Step 7 — Start the REST API

```bash
python run_api.py
```

API available at: **http://localhost:8080**
Swagger docs at: **http://localhost:8080/docs**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Liveness check |
| POST | `/api/query` | Run a legal query |
| POST | `/api/ingest` | Trigger re-ingestion from Google Sheets (background) |
| GET | `/api/ingest/status` | Check ingestion progress |

### POST /api/query

Request:
```json
{
  "query": "Can an employer deduct wages without notice?"
}
```

Response:
```json
{
  "query": "...",
  "answer": "NO\n\nReason: ...\n\nCitations: ...\n\nConclusion: ...",
  "answer_type": "NO",
  "reason": "...",
  "citations": ["[Payment of Wages Act, Section 7]"],
  "conclusion": "...",
  "laws": [{ "name": "...", "act_number": "...", "jurisdiction": "India", ... }],
  "sections": ["Section 7 — Deductions from wages"],
  "definitions": [{ "term": "wages", "definition": "..." }],
  "crossrefs": [],
  "cases": [{ "name": "...", "citation": "...", "holding": "..." }],
  "version": { "type": "CURRENT", "effective_from": "...", "effective_to": "" },
  "llm_live": true
}
```

---

## Current Working Query Flow

```
MERN Frontend / Gradio UI sends query
         │
         ▼
─────────────────────────────────────────────
STEP 1 — LAW SELECTION
─────────────────────────────────────────────
Score ALL active laws using 4 hybrid signals:

  • FAISS vector search  — semantic similarity via parallel batches (0–30 boost)
  • FTS5 keyword score   — exact keyword match in law name + sections
  • Name-match boost     — query mentions a law name directly (+8 to +20)
  • Specificity boost    — rare domain keywords in query (+4/+8/+15/+25)

All 4 signals merged → total score per law → ranked list (top 20)

LLM always decides:
  "Given this query, which of these laws apply?"
  → Returns only applicable laws (no fixed cap — 1, 3, 7, whatever applies)

         │
         ▼
─────────────────────────────────────────────
STEP 2 — SECTION SELECTION (per applicable law)
─────────────────────────────────────────────
For each applicable law:

  • FAISS section search — parallel batches, filtered to law → top 30
  • Keyword scoring      — label match (+5), keyword match (+3),
                           summary match (+3), text match (+1)
  • Hybrid merge: keyword score + FAISS vector boost (0–20)
    → ranked section list (top 20 candidates)

LLM always decides:
  "Given this query, which of these sections apply?"
  → Returns only applicable sections (no fixed cap)

Enrichment fetched for selected sections:
  • Definitions   — terms linked to selected sections
  • Cross-refs    — REPEAL / OVERRIDE entries only
  • Cases         — top 3: key holding + interpretive effect
  • Version       — CURRENT only (INACTIVE excluded)

         │
         ▼
─────────────────────────────────────────────
STEP 3 — QUERY TYPE DETECTION (inside LLM prompt)
─────────────────────────────────────────────
TYPE A — Compliance query
  Triggers: Can? Is? Are? Must? Should? Does? May?
  Example:  "Can an employer deduct wages?"
  → Output: YES / NO / DEPENDS

TYPE B — Informational query
  Triggers: What is? What are? How much? Define? Explain? List?
  Example:  "What is the minimum age for employment?"
  → Output: INFORMATION

         │
         ▼
─────────────────────────────────────────────
STEP 4 — ANSWER GENERATION
─────────────────────────────────────────────
Prompt sent to self-hosted Gemma 4 LLM (65k context window) contains:
  • System rules (mandatory format)
  • Per-law block for each applicable law:
      - Act name, act number, jurisdiction
      - Version: CURRENT + effective date + overview
      - Sections (exact legal text)
      - Definitions
      - Cross-references (Override / Repeal)
      - Case law (key holdings + interpretive effects)

TYPE A output format:
  YES / NO / DEPENDS
  Reason:     (3–4 lines, verified sources only)
  Citations:  [Law Name, Section X] / [Override: ...] / [Repeal: ...]
  Conclusion: (short, decisive, matches opening word)

TYPE B output format:
  INFORMATION
  Answer:     (direct factual answer from verified sources)
  Citations:  [Law Name, Section X]
  Conclusion: (short summary of key legal facts)

         │
         ▼
─────────────────────────────────────────────
STEP 5 — POST-PROCESSING
─────────────────────────────────────────────
  • YES/NO consistency check:
    If opening word contradicts conclusion → auto-corrected
    Skipped for INFORMATION responses
  • Answer parsed into structured fields:
    answer_type, reason, citations[], conclusion

         │
         ▼
─────────────────────────────────────────────
STEP 6 — LOG + RETURN
─────────────────────────────────────────────
  • Logged to query_history table in SQLite
    (query, laws selected, scores, decision type, answer preview, time)

  • REST API returns structured JSON to MERN frontend
  • Gradio UI displays answer, sources, pipeline trace, history
```

---

## Ingestion Options

```bash
# Full ingestion + FAISS build (default)
python run_ingestion.py

# Ingestion only, skip FAISS build
python run_ingestion.py --no-embed

# Ingestion + FAISS + LLM section summarization
python run_ingestion.py --summarize
```

---

## Files Not in Repository

| File | Reason |
|------|--------|
| `abyd-legal-ai-f489d309e52d.json` | Google Service Account key — get from team on WhatsApp |
| `database/legal_db.sqlite` | Generated by `run_ingestion.py` |
| `database/faiss/` | Generated by `run_ingestion.py` |
| `.env` | Contains secrets — copy from `.env.example` |

---

## LLM Offline / Mock Mode

If the LLM endpoint is unreachable, the system runs in **mock mode** automatically:
- Law and section selection still works (FAISS + FTS5)
- Answer generation returns a placeholder response
- All source data (laws, sections, definitions) still displays correctly

To force mock mode: set `MOCK_LLM=true` in your `.env`.
