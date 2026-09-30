import sqlite3
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import DB_PATH

_conn = None


def get_connection():
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=OFF")
    return _conn


def init_db():
    conn = get_connection()
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS laws (
        law_uid         TEXT PRIMARY KEY,
        law_name        TEXT NOT NULL,
        domain          TEXT,
        law_type        TEXT,
        jurisdiction    TEXT,
        act_number      TEXT,
        enactment_year  INTEGER,
        status          TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sections (
        section_uid   TEXT PRIMARY KEY,
        law_uid       TEXT NOT NULL,
        section_label TEXT,
        keywords      TEXT,
        exact_text    TEXT,
        llm_summary   TEXT,
        created_at    TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (law_uid) REFERENCES laws(law_uid)
    );

    CREATE TABLE IF NOT EXISTS definitions (
        definition_uid  TEXT PRIMARY KEY,
        term            TEXT,
        sub_section_uid TEXT,
        definition_text TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS versions (
        version_uid        TEXT PRIMARY KEY,
        section_uid        TEXT,
        law_uid            TEXT,
        version_type       TEXT,
        effective_from     TEXT,
        effective_to       TEXT,
        summary_legal_text TEXT,
        created_at         TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS cross_references (
        crossref_uid      TEXT PRIMARY KEY,
        from_section_uid  TEXT,
        from_law_uid      TEXT,
        to_section_uid    TEXT,
        to_law_uid        TEXT,
        relationship_type TEXT,
        created_at        TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS cases (
        case_uid              TEXT PRIMARY KEY,
        case_name             TEXT,
        court                 TEXT,
        citation              TEXT,
        year                  INTEGER,
        governing_law         TEXT,
        sections_interpreted  TEXT,
        key_holding           TEXT,
        interpretive_effect   TEXT,
        created_at            TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS dataset_metadata (
        key          TEXT PRIMARY KEY,
        value        TEXT,
        description  TEXT,
        last_updated TEXT
    );
    """)

    # B-tree indexes
    cur.executescript("""
    CREATE INDEX IF NOT EXISTS idx_sections_law_uid        ON sections(law_uid);
    CREATE INDEX IF NOT EXISTS idx_definitions_sub_sec     ON definitions(sub_section_uid);
    CREATE INDEX IF NOT EXISTS idx_versions_law_uid        ON versions(law_uid);
    CREATE INDEX IF NOT EXISTS idx_versions_section_uid    ON versions(section_uid);
    CREATE INDEX IF NOT EXISTS idx_crossrefs_from_law      ON cross_references(from_law_uid);
    CREATE INDEX IF NOT EXISTS idx_crossrefs_to_law        ON cross_references(to_law_uid);
    """)

    # Migrate: add llm_summary if missing
    try:
        cur.execute("SELECT llm_summary FROM sections LIMIT 1")
    except Exception:
        cur.execute("ALTER TABLE sections ADD COLUMN llm_summary TEXT")

    # FTS5 virtual tables
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS laws_fts USING fts5(
            law_uid UNINDEXED, law_name, domain, jurisdiction, act_number,
            content='laws', content_rowid='rowid'
        )
    """)
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            section_uid UNINDEXED, law_uid UNINDEXED,
            section_label, keywords, exact_text,
            content='sections', content_rowid='rowid'
        )
    """)
    conn.commit()


def _now():
    return datetime.utcnow().isoformat()


def upsert_laws(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO laws
                (law_uid, law_name, domain, law_type, jurisdiction,
                 act_number, enactment_year, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            row.get("law_uid"), row.get("law_name"), row.get("domain"),
            row.get("law_type"), row.get("jurisdiction"), row.get("act_number"),
            row.get("enactment_year"), row.get("status"), _now()
        ))
    conn.commit()
    _rebuild_laws_fts(cur)
    conn.commit()


def upsert_sections(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO sections
                (section_uid, law_uid, section_label, keywords, exact_text, created_at)
            VALUES (?,?,?,?,?,?)
        """, (
            row.get("section_uid"), row.get("law_uid"), row.get("section_label"),
            row.get("keywords"), row.get("exact_text"), _now()
        ))
    conn.commit()
    _rebuild_sections_fts(cur)
    conn.commit()


def upsert_definitions(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO definitions
                (definition_uid, term, sub_section_uid, definition_text, created_at)
            VALUES (?,?,?,?,?)
        """, (
            row.get("definition_uid"), row.get("term"),
            row.get("sub_section_uid"), row.get("definition_text"), _now()
        ))
    conn.commit()


def upsert_versions(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO versions
                (version_uid, section_uid, law_uid, version_type,
                 effective_from, effective_to, summary_legal_text, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            row.get("version_uid"), row.get("section_uid"), row.get("law_uid"),
            row.get("version_type"), row.get("effective_from"), row.get("effective_to"),
            row.get("summary_legal_text"), _now()
        ))
    conn.commit()


def upsert_crossrefs(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO cross_references
                (crossref_uid, from_section_uid, from_law_uid,
                 to_section_uid, to_law_uid, relationship_type, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (
            row.get("crossref_uid"), row.get("from_section_uid"), row.get("from_law_uid"),
            row.get("to_section_uid"), row.get("to_law_uid"), row.get("relationship_type"), _now()
        ))
    conn.commit()


def upsert_cases(rows: list):
    conn = get_connection()
    cur  = conn.cursor()
    for row in rows:
        cur.execute("""
            INSERT OR REPLACE INTO cases
                (case_uid, case_name, court, citation, year,
                 governing_law, sections_interpreted, key_holding,
                 interpretive_effect, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            row.get("case_uid"), row.get("case_name"), row.get("court"),
            row.get("citation"), row.get("year"), row.get("governing_law"),
            row.get("sections_interpreted"), row.get("key_holding"),
            row.get("interpretive_effect"), _now()
        ))
    conn.commit()


def update_section_summaries(summaries: list):
    conn = get_connection()
    cur  = conn.cursor()
    for s in summaries:
        cur.execute("UPDATE sections SET llm_summary=? WHERE section_uid=?",
                    (s["llm_summary"], s["section_uid"]))
    conn.commit()


def get_sections_without_summary(limit: int = 0) -> list:
    conn = get_connection()
    cur  = conn.cursor()
    sql  = """SELECT section_uid, law_uid, section_label, keywords, exact_text
              FROM sections WHERE (llm_summary IS NULL OR llm_summary = '')
              AND exact_text IS NOT NULL AND exact_text != ''"""
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql)
    return [{"section_uid": r[0], "law_uid": r[1], "section_label": r[2],
             "keywords": r[3], "exact_text": r[4]} for r in cur.fetchall()]


def _rebuild_laws_fts(cur):
    cur.execute("INSERT INTO laws_fts(laws_fts) VALUES('rebuild')")


def _rebuild_sections_fts(cur):
    cur.execute("INSERT INTO sections_fts(sections_fts) VALUES('rebuild')")


def get_row_counts() -> dict:
    conn = get_connection()
    cur  = conn.cursor()
    tables = ["laws", "sections", "definitions", "versions", "cross_references", "cases"]
    counts = {}
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        counts[t] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sections WHERE llm_summary IS NOT NULL AND llm_summary != ''")
    counts["sections_with_summary"] = cur.fetchone()[0]
    return counts
