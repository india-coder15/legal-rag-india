"""
Query history logger.
Logs every query processed by the pipeline to a SQLite table.
Provides read functions for the Gradio History tab.
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import DB_PATH


def _conn():
    return sqlite3.connect(str(DB_PATH))


def init_history_table():
    """Create query_history table if it doesn't exist."""
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            query           TEXT NOT NULL,
            laws_selected   TEXT,
            decision_type   TEXT,
            finalists_json  TEXT,
            fts_query_used  TEXT,
            answer_preview  TEXT,
            processing_ms   INTEGER,
            trace_detail    TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE query_history ADD COLUMN trace_detail TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


def log_query(
    query: str,
    laws_selected: list,
    decision_type: str,
    finalists: list,
    fts_query_used: str,
    answer: str,
    processing_ms: int,
    trace: dict = None,
):
    """Write one query record to query_history table."""
    try:
        init_history_table()
        conn = _conn()

        finalists_summary = json.dumps([
            {
                "rank":         i + 1,
                "law_uid":      f.get("law_uid", ""),
                "law_name":     f.get("law_name", ""),
                "score":        f.get("score", 0),
                "fts_raw":      f.get("fts_raw", 0),
                "fts_norm":     f.get("fts_norm", 0),
                "name_boost":   f.get("name_boost", 0),
                "spec_boost":   f.get("spec_boost", 0),
                "vec_boost":    f.get("vec_boost", 0),
                "name_boosted": f.get("name_boosted", False),
            }
            for i, f in enumerate(finalists[:15])
        ])

        trace_detail = None
        if trace:
            all_laws_scored = trace.get("all_laws_scored", [])
            trace_detail = json.dumps({
                "fts_query_used":    fts_query_used,
                "fts_mode":          trace.get("fts_mode", ""),
                "query_keywords":    trace.get("query_keywords", []),
                "tracked_keywords":  trace.get("tracked_keywords", []),
                "total_active_laws": trace.get("total_active_laws", 0),
                "all_laws_scored": [
                    {
                        "law_uid":       l.get("law_uid", ""),
                        "law_name":      l.get("law_name", ""),
                        "fts_raw":       l.get("fts_raw", 0),
                        "fts_norm":      l.get("fts_norm", 0),
                        "norm_factor":   l.get("norm_factor", 1.0),
                        "section_count": l.get("section_count", 0),
                        "name_boost":    l.get("name_boost", 0),
                        "spec_boost":    l.get("spec_boost", 0),
                        "vec_boost":     l.get("vec_boost", 0),
                        "spec_keywords": l.get("spec_keywords", []),
                        "amend_penalty": l.get("amend_penalty", 0),
                        "score":         l.get("score", 0),
                        "name_boosted":  l.get("name_boosted", False),
                    }
                    for l in all_laws_scored
                ],
            })

        conn.execute("""
            INSERT INTO query_history
              (timestamp, query, laws_selected, decision_type,
               finalists_json, fts_query_used, answer_preview, processing_ms, trace_detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            json.dumps(laws_selected),
            decision_type,
            finalists_summary,
            fts_query_used,
            (answer or "")[:300],
            processing_ms,
            trace_detail,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[query_logger] Warning: could not log query — {e}")


def get_history(limit: int = 50) -> list:
    """Return last N query records, newest first."""
    try:
        init_history_table()
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, timestamp, query, laws_selected, decision_type,
                   finalists_json, fts_query_used, answer_preview, processing_ms
            FROM query_history ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id":             r[0],
                "timestamp":      r[1],
                "query":          r[2],
                "laws_selected":  _safe_json(r[3], []),
                "decision_type":  r[4] or "",
                "finalists":      _safe_json(r[5], []),
                "fts_query_used": r[6] or "",
                "answer_preview": r[7] or "",
                "processing_ms":  r[8] or 0,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[query_logger] Warning: could not read history — {e}")
        return []


def get_record(record_id: int) -> dict:
    """Return single query record by ID."""
    try:
        init_history_table()
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, timestamp, query, laws_selected, decision_type,
                   finalists_json, fts_query_used, answer_preview, processing_ms, trace_detail
            FROM query_history WHERE id = ?
        """, (record_id,))
        r = cur.fetchone()
        conn.close()
        if not r:
            return {}
        return {
            "id":             r[0],
            "timestamp":      r[1],
            "query":          r[2],
            "laws_selected":  _safe_json(r[3], []),
            "decision_type":  r[4] or "",
            "finalists":      _safe_json(r[5], []),
            "fts_query_used": r[6] or "",
            "answer_preview": r[7] or "",
            "processing_ms":  r[8] or 0,
            "trace_detail":   _safe_json(r[9], {}),
        }
    except Exception:
        return {}


def delete_query(record_id: int) -> bool:
    """Delete a single query record by ID."""
    try:
        init_history_table()
        conn = _conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM query_history WHERE id = ?", (record_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    except Exception as e:
        print(f"[query_logger] Warning: could not delete record — {e}")
        return False


def _safe_json(val, default):
    try:
        return json.loads(val) if val else default
    except Exception:
        return default
