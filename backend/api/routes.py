"""
FastAPI REST endpoints for MERN frontend integration.

Endpoints:
  GET  /api/health         — liveness check
  POST /api/query          — run a legal query, returns structured JSON
  POST /api/ingest         — trigger Google Sheets → SQLite → FAISS re-ingestion (background)
  GET  /api/ingest/status  — check ingestion progress
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Legal AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production to your React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared ingestion state ────────────────────────────────────────────────────

_ingest_state: dict = {"running": False, "last_status": None, "error": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_answer(answer: str) -> dict:
    """
    Parse the structured LLM answer text into discrete fields so the
    frontend can render components (verdict badge, citations list, etc.)
    without doing its own string parsing.
    """
    if not answer:
        return {"answer_type": "UNKNOWN", "reason": "", "citations": [], "conclusion": ""}

    lines      = answer.strip().split("\n")
    first_line = lines[0].strip()

    if re.match(r"\*{0,2}INFORMATION\*{0,2}\s*$", first_line, re.IGNORECASE):
        answer_type = "INFORMATION"
    elif re.match(r"\*{0,2}YES\*{0,2}\s*$", first_line, re.IGNORECASE):
        answer_type = "YES"
    elif re.match(r"\*{0,2}NO\*{0,2}\s*$", first_line, re.IGNORECASE):
        answer_type = "NO"
    elif re.match(r"\*{0,2}DEPENDS\*{0,2}\s*$", first_line, re.IGNORECASE):
        answer_type = "DEPENDS"
    else:
        answer_type = "UNKNOWN"

    # "Reason:" for FORMAT A, "Answer:" for FORMAT B
    reason_match = re.search(
        r"(?:Reason|Answer)\s*:\s*(.*?)(?=\n\s*Citations\s*:|\n\s*Conclusion\s*:|\Z)",
        answer, re.IGNORECASE | re.DOTALL,
    )
    citations_match = re.search(
        r"Citations\s*:\s*(.*?)(?=\n\s*Conclusion\s*:|\Z)",
        answer, re.IGNORECASE | re.DOTALL,
    )
    conclusion_match = re.search(
        r"Conclusion\s*:\s*(.*?)$",
        answer, re.IGNORECASE | re.DOTALL,
    )

    citations_text = citations_match.group(1).strip() if citations_match else ""
    citations_list = [
        line.lstrip("- ").strip()
        for line in citations_text.split("\n")
        if line.strip().startswith("-")
    ]

    return {
        "answer_type": answer_type,
        "reason":      reason_match.group(1).strip() if reason_match else "",
        "citations":   citations_list,
        "conclusion":  conclusion_match.group(1).strip() if conclusion_match else "",
    }


def _run_ingest_background():
    """Runs in FastAPI BackgroundTask — Excel(GSheets) → SQLite → FAISS."""
    from backend.ingestion.batch_processor import run_ingestion, run_embedding_indexing
    _ingest_state["running"]     = True
    _ingest_state["error"]       = None
    _ingest_state["last_status"] = "running"
    try:
        run_ingestion()
        run_embedding_indexing()
        _ingest_state["last_status"] = "success"
    except Exception as exc:
        _ingest_state["last_status"] = "failed"
        _ingest_state["error"]       = str(exc)
    finally:
        _ingest_state["running"] = False


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/api/query", tags=["Query"])
def query_endpoint(req: QueryRequest):
    """
    Run a legal query through the full pipeline.

    Request body:
        { "query": "What is the minimum age for employment?" }

    Response:
        {
          "query":        string,
          "answer":       string,          // raw LLM answer
          "answer_type":  "YES|NO|DEPENDS|INFORMATION|UNKNOWN",
          "reason":       string,          // parsed Reason / Answer section
          "citations":    [string, ...],   // parsed citation lines
          "conclusion":   string,          // parsed Conclusion section
          "laws":         [...],
          "sections":     [...],
          "definitions":  [...],
          "crossrefs":    [...],
          "cases":        [...],
          "version":      { type, effective_from, effective_to },
          "llm_live":     bool
        }
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    from backend.retrieval.pipeline import run_query
    result = run_query(query)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    parsed = _parse_answer(result.get("answer", ""))
    result.update(parsed)
    return result


@app.post("/api/ingest", tags=["Ingestion"])
def ingest_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger a full re-ingestion from Google Sheets.
    Runs in the background — poll /api/ingest/status for progress.
    """
    if _ingest_state["running"]:
        return {"status": "already_running", "message": "Ingestion is already in progress"}
    background_tasks.add_task(_run_ingest_background)
    return {"status": "started", "message": "Ingestion started in background"}


@app.get("/api/ingest/status", tags=["Ingestion"])
def ingest_status():
    """
    Returns current ingestion state.

    Response:
        {
          "running":     bool,
          "last_status": "running|success|failed|null",
          "error":       string|null
        }
    """
    return _ingest_state
