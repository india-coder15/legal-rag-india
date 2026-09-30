"""
Run this script to import Excel data into SQLite and build FAISS indices.
Always deletes the old database and rebuilds from scratch.

Usage:
    python run_ingestion.py              # ingest + build FAISS indices
    python run_ingestion.py --summarize  # also run LLM section summarization
    python run_ingestion.py --no-embed   # ingest only, skip FAISS build
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backend.ingestion.batch_processor import run_ingestion, run_embedding_indexing, run_llm_summaries
from backend.config import GOOGLE_SHEET_ID, DB_PATH, FAISS_DIR

if __name__ == "__main__":
    print(f"Source  : Google Sheets (ID: {GOOGLE_SHEET_ID})")
    print(f"Database: {DB_PATH}")
    print(f"FAISS   : {FAISS_DIR}\n")

    # Step 1: Delete old database and rebuild from Excel
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Old database deleted. Rebuilding from scratch...")

    run_ingestion()

    # Step 2: Build FAISS embedding indices (unless skipped)
    if "--no-embed" not in sys.argv:
        # Delete old FAISS indices before rebuild
        for f in ["laws.index", "laws_ids.json", "sections.index", "sections_ids.json"]:
            p = FAISS_DIR / f
            if p.exists():
                p.unlink()
        run_embedding_indexing()
    else:
        print("\nSkipping FAISS embedding (--no-embed flag set)")

    # Step 3: Optional LLM summarization
    if "--summarize" in sys.argv:
        run_llm_summaries(batch_size=8)

    print("\nAll done. Run 'python run.py' to start the application.")
