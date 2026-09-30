import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent  # D:\Rhett\Annexure B 2\

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH    = BASE_DIR / "database" / "legal_db.sqlite"
FAISS_DIR  = BASE_DIR / "database" / "faiss"
GOOGLE_SHEET_ID   = os.getenv("GOOGLE_SHEET_ID", "1GLA_Dy5p8h5DKucN8FWmzqfmmYWexkmJfTQCdgyLvz4")
GOOGLE_CREDS_PATH = Path(os.getenv("GOOGLE_CREDS_PATH", str(BASE_DIR / "abyd-legal-ai-f489d309e52d.json")))

# ── Ingestion ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 50

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_ENDPOINT     = os.getenv("LLM_ENDPOINT", "http://44.223.191.230:8000")
LLM_MODEL        = os.getenv("LLM_MODEL", "gemma-4-31b-it")
LLM_MAX_TOKENS   = 65536
LLM_ANSWER_TOKENS  = 1500
LLM_SELECT_TOKENS  = 500

# ── Embeddings (FAISS) ─────────────────────────────────────────────────────────
EMBEDDING_MODEL         = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM           = 384
HYBRID_VEC_WEIGHT_LAW     = 30   # max vector boost added to law score
HYBRID_VEC_WEIGHT_SECTION = 20   # max vector boost added to section score

# ── Excel sheet names (must match exactly) ─────────────────────────────────────
SHEET_LAWS      = "Laws"
SHEET_SECTIONS  = "Sections"
SHEET_DEFINITIONS = "Definitions"
SHEET_VERSIONS  = "Versions"
SHEET_CROSSREFS = "Cross References"
SHEET_CASES     = "Cases"
