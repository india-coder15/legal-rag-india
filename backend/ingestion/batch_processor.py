"""
Ingestion orchestrator.
  run_ingestion()        — reads Excel → writes SQLite → builds FTS5 indexes
  run_embedding_indexing() — reads SQLite → generates embeddings → builds FAISS indices
  run_llm_summaries()   — optional: LLM summarizes sections without summaries
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import BATCH_SIZE, FAISS_DIR
from backend.ingestion.excel_reader import (
    read_laws, read_sections, read_definitions,
    read_versions, read_crossrefs, read_cases
)
from backend.ingestion.db_writer import (
    init_db, upsert_laws, upsert_sections, upsert_definitions,
    upsert_versions, upsert_crossrefs, upsert_cases,
    get_row_counts, get_sections_without_summary, update_section_summaries
)

try:
    from rich.console import Console
    from rich.progress import track
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False


def _log(msg):
    if USE_RICH:
        console.print(msg)
    else:
        print(msg)


def _batch(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _ingest_sheet(name, rows, upsert_fn, batch_size):
    if not rows:
        _log(f"  {name}: 0 rows — skipping")
        return 0
    total    = len(rows)
    inserted = 0
    if USE_RICH:
        for batch in track(list(_batch(rows, batch_size)), description=f"  {name}"):
            upsert_fn(batch)
            inserted += len(batch)
    else:
        for batch in _batch(rows, batch_size):
            upsert_fn(batch)
            inserted += len(batch)
            print(f"  {name}: {inserted}/{total}", end="\r")
        print()
    return inserted


# ── Step 1: Excel → SQLite ─────────────────────────────────────────────────────

def run_ingestion(batch_size: int = BATCH_SIZE):
    _log("\n=== Legal Dataset Ingestion ===\n")
    _log("Initialising database schema...")
    init_db()
    _log("  Schema ready.\n")

    _log("Reading Excel sheets...")
    laws        = read_laws()
    sections    = read_sections()
    definitions = read_definitions()
    versions    = read_versions()
    crossrefs   = read_crossrefs()
    cases       = read_cases()
    _log(f"  Read: {len(laws)} laws, {len(sections)} sections, "
         f"{len(definitions)} definitions, {len(versions)} versions, "
         f"{len(crossrefs)} cross-refs, {len(cases)} cases\n")

    _log("Writing to database...")
    _ingest_sheet("Laws",            laws,        upsert_laws,        batch_size)
    _ingest_sheet("Sections",        sections,    upsert_sections,    batch_size)
    _ingest_sheet("Definitions",     definitions, upsert_definitions, batch_size)
    _ingest_sheet("Versions",        versions,    upsert_versions,    batch_size)
    _ingest_sheet("Cross References", crossrefs,  upsert_crossrefs,   batch_size)
    _ingest_sheet("Cases",           cases,       upsert_cases,       batch_size)

    counts = get_row_counts()
    _log("\n=== Ingestion Complete ===")
    _log("\nDatabase row counts:")
    for table, count in counts.items():
        _log(f"  {table:<25} {count:>6} rows")
    _log("")


# ── Step 2: SQLite → FAISS Embeddings ──────────────────────────────────────────

def run_embedding_indexing():
    """
    Read all active laws + sections from SQLite, generate embeddings,
    build FAISS indices and save to database/faiss/.

    Called automatically after run_ingestion() in run_ingestion.py.
    Safe to re-run — fully rebuilds indices each time.
    """
    import sqlite3
    import numpy as np
    from backend.config import DB_PATH
    from backend.embeddings.encoder import encode_texts
    from backend.embeddings.text_prep import prepare_law_text, prepare_section_text
    from backend.embeddings.faiss_index import (
        build_law_index, build_section_index, reload_indices
    )

    _log("\n=== Embedding Indexing ===\n")
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    # ── Laws ──
    _log("Loading active laws from DB...")
    cur.execute("""
        SELECT l.law_uid, l.law_name, l.domain, l.jurisdiction, l.act_number
        FROM laws l
        WHERE l.law_uid NOT IN (
            SELECT DISTINCT law_uid FROM versions
            WHERE upper(version_type) = 'INACTIVE'
            AND (section_uid IS NULL OR section_uid = '')
            AND law_uid != ''
        )
        ORDER BY l.law_uid
    """)
    law_rows = cur.fetchall()
    _log(f"  {len(law_rows)} active laws found\n")

    law_uids  = []
    law_texts = []
    for law_uid, law_name, domain, jurisdiction, act_number in law_rows:
        # Get top section labels for this law to enrich law embedding
        cur.execute("""
            SELECT section_label FROM sections
            WHERE law_uid = ? AND section_label IS NOT NULL AND section_label != ''
            ORDER BY section_uid LIMIT 20
        """, (law_uid,))
        labels = [r[0] for r in cur.fetchall()]
        row    = {"law_uid": law_uid, "law_name": law_name, "domain": domain,
                  "jurisdiction": jurisdiction, "act_number": act_number}
        law_uids.append(law_uid)
        law_texts.append(prepare_law_text(row, labels))

    _log(f"Generating law embeddings ({len(law_texts)} laws)...")
    law_embeddings = encode_texts(law_texts, batch_size=32)
    build_law_index(law_uids, law_embeddings)
    _log(f"  Law FAISS index built: {len(law_uids)} vectors\n")

    # ── Sections ──
    _log("Loading active sections from DB...")
    cur.execute("""
        SELECT section_uid, law_uid, section_label, keywords, llm_summary, exact_text
        FROM sections
        WHERE section_uid NOT IN (
            SELECT DISTINCT section_uid FROM versions
            WHERE upper(version_type) = 'INACTIVE'
            AND section_uid != ''
        )
        ORDER BY section_uid
    """)
    sec_rows = cur.fetchall()
    _log(f"  {len(sec_rows)} active sections found\n")

    conn.close()

    sec_uids  = []
    sec_texts = []
    for sec_uid, law_uid, label, keywords, summary, text in sec_rows:
        row = {
            "section_uid":   sec_uid,
            "section_label": label or "",
            "keywords":      keywords or "",
            "llm_summary":   summary or "",
            "exact_text":    text or "",
        }
        sec_uids.append(sec_uid)
        sec_texts.append(prepare_section_text(row))

    _log(f"Generating section embeddings ({len(sec_texts)} sections)...")
    sec_embeddings = encode_texts(sec_texts, batch_size=64)
    build_section_index(sec_uids, sec_embeddings)
    _log(f"  Section FAISS index built: {len(sec_uids)} vectors\n")

    # Reload cached singletons so queries immediately use new indices
    reload_indices()

    _log("=== Embedding Indexing Complete ===\n")
    _log(f"  Law index    : {FAISS_DIR / 'laws.index'}")
    _log(f"  Section index: {FAISS_DIR / 'sections.index'}\n")


# ── Step 3: Optional LLM summaries ────────────────────────────────────────────

def run_llm_summaries(batch_size: int = 8):
    """Run LLM over sections to generate 1-2 line summaries. Safe to re-run."""
    from backend.llm.mistral_client import call_summarize_batch, is_live

    _log("\n=== LLM Section Summarization ===\n")
    pending = get_sections_without_summary()
    if not pending:
        _log("  All sections already have summaries. Nothing to do.")
        return

    _log(f"  Sections to summarize: {len(pending)}")
    _log(f"  LLM endpoint: {'LIVE' if is_live() else 'MOCK'}")
    _log(f"  Batch size: {batch_size} sections per LLM call\n")

    done    = 0
    batches = list(_batch(pending, batch_size))
    for chunk in (track(batches, description="  Summarizing") if USE_RICH else batches):
        summaries = call_summarize_batch(chunk)
        updates   = [
            {"section_uid": sec["section_uid"], "llm_summary": summary.strip()}
            for sec, summary in zip(chunk, summaries)
            if summary and summary.strip()
        ]
        if updates:
            update_section_summaries(updates)
        done += len(chunk)
        if not USE_RICH:
            print(f"  Summarized: {done}/{len(pending)}", end="\r")

    if not USE_RICH:
        print()
    _log(f"\n  Done. Summarized {done} sections.")
