"""
Step 1 of retrieval: identify all applicable laws for a user query.

Flow:
  1. Embed query once (10ms)
  2. FAISS law search — parallel batches → top 20 by vector similarity
  3. FTS5 search → scores for all laws
  4. Name-match boost → if query mentions a law name
  5. Hybrid merge per law:
       total = fts_norm + name_boost + spec_boost + vec_boost(0-30)
  6. LLM always decides: "Which of these laws apply to the query?"
     → returns only the applicable laws (no fixed cap)
  7. Returns list of applicable law dicts + trace metadata
"""
import sys
import json
import sqlite3
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import DB_PATH, HYBRID_VEC_WEIGHT_LAW
from backend.llm.mistral_client import call_select_laws


def _conn():
    return sqlite3.connect(str(DB_PATH))


_STOP_WORDS = {
    "what","are","the","is","a","an","of","in","for","to","and","or","on",
    "under","as","that","this","which","with","be","by","any","all","do",
    "does","can","could","would","should","how","when","where","who","why",
    "explain","tell","me","give","list","show","describe","define","about",
    "regarding","related","provisions","provision",
}

_TRACK_KEYWORDS = [
    "gratuity", "deduct", "retrenchment", "maternity", "provident",
    "esic", "esi", "epf", "factory", "mine", "hazardous", "contractor",
    "overtime", "bonus", "apprentice", "plantation", "beedi", "canteen",
    "compensation", "termination", "dismissal", "resignation", "layoff",
    "lockout", "arbitration", "conciliation", "tribunal", "strike",
    "probation", "superannuation", "workmen",
    "migrant", "construction", "building", "transport", "cinema",
    "welfare", "safety", "registration", "insurance", "ventilation",
    "discrimination", "remuneration", "accommodation", "allowance",
    "protective", "machinery", "adolescent", "scavenger", "dock",
]

_KEYWORD_LAW_MAP_CACHE = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_for_fts(query: str) -> str:
    tokens = re.sub(r'[^\w\s]', ' ', query).lower().split()
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else re.sub(r'[^\w\s]', ' ', query).strip()


def _query_words(query: str) -> list:
    words = re.sub(r'[^\w\s]', ' ', query).lower().split()
    return [w for w in words if w not in _STOP_WORDS and len(w) >= 3]


# ── Specificity boost ─────────────────────────────────────────────────────────

def _get_keyword_law_map() -> dict:
    global _KEYWORD_LAW_MAP_CACHE
    if _KEYWORD_LAW_MAP_CACHE is not None:
        return _KEYWORD_LAW_MAP_CACHE
    conn = _conn()
    cur  = conn.cursor()
    keyword_law_map = {}
    for kw in _TRACK_KEYWORDS:
        cur.execute("""
            SELECT DISTINCT law_uid FROM sections
            WHERE lower(keywords) LIKE ? OR lower(exact_text) LIKE ?
        """, (f"%{kw}%", f"%{kw}%"))
        keyword_law_map[kw] = {row[0] for row in cur.fetchall()}
    conn.close()
    _KEYWORD_LAW_MAP_CACHE = keyword_law_map
    return keyword_law_map


def _specificity_boost_detail(query_words_list: list, law_uid: str) -> tuple:
    kw_map  = _get_keyword_law_map()
    boost   = 0
    details = []
    seen    = set()
    for word in query_words_list:
        if word not in kw_map or word in seen:
            continue
        seen.add(word)
        law_set = kw_map[word]
        if law_uid not in law_set:
            continue
        count = len(law_set)
        if   count <= 3:  b = 25
        elif count <= 8:  b = 15
        elif count <= 15: b = 8
        elif count <= 25: b = 4
        else:             b = 0
        if b > 0:
            details.append({"keyword": word, "law_count": count, "boost": b})
            boost += b
    return boost, details


# ── FTS scoring ───────────────────────────────────────────────────────────────

def _fts_score_all(query: str, cur) -> tuple:
    clean = _clean_for_fts(query)
    if not clean:
        return {}, "", "empty"

    sql = """
        WITH law_hits AS (
            SELECT law_uid, 3 AS score FROM laws_fts WHERE laws_fts MATCH ?
        ),
        sec_hits AS (
            SELECT law_uid, 1 AS score FROM sections_fts WHERE sections_fts MATCH ?
        ),
        combined AS (
            SELECT law_uid, SUM(score) AS total_score
            FROM (SELECT * FROM law_hits UNION ALL SELECT * FROM sec_hits)
            GROUP BY law_uid
        )
        SELECT law_uid, total_score FROM combined
    """

    def _run(fts_q):
        try:
            cur.execute(sql, (fts_q, fts_q))
            rows = cur.fetchall()
            return {row[0]: row[1] for row in rows} if rows else None
        except Exception:
            return None

    result = _run(clean)
    if result:
        return result, clean, "exact"

    words = clean.split()
    for take in (5, 4, 3, 2):
        if len(words) >= take:
            fts_q  = " OR ".join(words[:take])
            result = _run(fts_q)
            if result:
                return result, fts_q, f"OR-fallback({take})"

    return {}, clean, "no-match"


# ── Name-match boost ──────────────────────────────────────────────────────────

def _name_match_boost(query: str, cur) -> dict:
    query_lower = query.lower()
    cur.execute("SELECT law_uid, law_name FROM laws")
    rows   = cur.fetchall()
    boosts = {}
    for law_uid, law_name in rows:
        if not law_name:
            continue
        name_lower = law_name.lower()
        if len(name_lower) >= 8 and name_lower.rstrip(".") in query_lower:
            boosts[law_uid] = 20
            continue
        simplified = re.sub(r'\(.*?\)', '', name_lower)
        simplified = re.sub(r'\s*,?\s*\d{4}\.?\s*$', '', simplified)
        simplified = re.sub(r'\b(act|code|rules?|regulations?|order|ordinance)\.?\s*$',
                            '', simplified, flags=re.IGNORECASE).strip(" ,.")
        sim_words  = [w for w in simplified.split() if len(w) > 1]
        if len(sim_words) >= 3 and simplified in query_lower:
            boosts[law_uid] = 20
            continue
        primary = re.sub(r'\(.*', '', name_lower).strip()
        primary = re.sub(r'\s*,?\s*\d{4}\.?\s*$', '', primary).strip()
        _GENERIC     = {'labour','wages','laws','rules','workers','employees',
                        'employment','provisions','miscellaneous','regulation'}
        _COMMON_LEGAL = {'code','employee','employer','industrial','payment',
                         'compensation','social','security','central','state'}
        all_words = [w for w in re.split(r'\W+', primary)
                     if len(w) > 3 and w not in _STOP_WORDS and w not in _GENERIC]
        if len(all_words) >= 3:
            matches = sum(1 for w in all_words if w in query_lower)
            ratio   = matches / len(all_words)
            if matches >= 2 and ratio >= 0.75:
                boosts[law_uid] = 18
            elif matches >= 3 and ratio >= 0.5:
                boosts[law_uid] = 12
        elif len(all_words) >= 2:
            matches = sum(1 for w in all_words if w in query_lower)
            matched = [w for w in all_words if w in query_lower]
            if matches == len(all_words):
                boosts[law_uid] = 15
            elif matches >= 2:
                boosts[law_uid] = 12
            elif matches >= 1 and any(w not in _COMMON_LEGAL for w in matched):
                boosts[law_uid] = 10
        elif len(all_words) == 1:
            w = all_words[0]
            if w in query_lower and len(w) >= 6 and w not in _COMMON_LEGAL:
                boosts[law_uid] = 8
    return boosts


# ── FAISS vector scoring (parallel batches) ───────────────────────────────────

def _vector_score_all(query: str) -> dict:
    """
    Embed query → search all law vectors in parallel batches via FAISS.
    Returns {law_uid: normalized_score (0-HYBRID_VEC_WEIGHT_LAW)}.
    Gracefully returns {} if FAISS index not built yet.
    """
    try:
        from backend.embeddings.encoder import encode_query
        from backend.embeddings.faiss_index import search_laws
        query_vec = encode_query(query)
        results   = search_laws(query_vec, top_k=20)
        if not results:
            return {}
        # Normalize: top similarity score → HYBRID_VEC_WEIGHT_LAW, others proportional
        top_score = results[0][1] if results[0][1] > 0 else 1.0
        return {
            uid: int((score / top_score) * HYBRID_VEC_WEIGHT_LAW)
            for uid, score in results
        }
    except Exception:
        return {}  # Fallback: FAISS not built yet, vec_boost = 0 for all


# ── Scoring: score all laws, build ranked list ─────────────────────────────────

def _score_all_laws(query: str) -> tuple:
    """
    Score every active law using hybrid signals:
      total = fts_norm + name_boost + spec_boost + vec_boost
    Returns (all_finalists_sorted, meta_dict).
    """
    conn = _conn()
    cur  = conn.cursor()

    # Active laws only
    cur.execute("""
        SELECT l.law_uid, l.law_name, l.domain, l.jurisdiction, l.status
        FROM laws l
        WHERE l.law_uid NOT IN (
            SELECT DISTINCT law_uid FROM versions
            WHERE upper(version_type) = 'INACTIVE'
            AND (section_uid IS NULL OR section_uid = '')
            AND law_uid != ''
        )
        ORDER BY l.law_uid
    """)
    all_laws = cur.fetchall()

    if not all_laws:
        conn.close()
        return [], {}

    name_boosts                       = _name_match_boost(query, cur)
    fts_scores, fts_q_used, fts_mode  = _fts_score_all(query, cur)

    cur.execute("SELECT law_uid, COUNT(*) FROM sections GROUP BY law_uid")
    section_counts = dict(cur.fetchall())
    conn.close()

    qwords      = _query_words(query)
    vec_boosts  = _vector_score_all(query)   # parallel FAISS search
    kw_map      = _get_keyword_law_map()

    tracked_in_query = []
    seen_kw = set()
    for w in qwords:
        if w in kw_map and w not in seen_kw:
            seen_kw.add(w)
            tracked_in_query.append({"keyword": w, "law_count": len(kw_map[w])})

    all_laws_scored = []
    for law_uid, law_name, domain, jurisdiction, status in all_laws:
        fts_raw  = fts_scores.get(law_uid, 0)
        name     = name_boosts.get(law_uid, 0)
        spec, spec_detail = _specificity_boost_detail(qwords, law_uid)
        vec      = vec_boosts.get(law_uid, 0)

        sec_count   = section_counts.get(law_uid, 30)
        norm_factor = round(min(1.0, 60.0 / max(60, sec_count)), 2)
        fts_norm    = int(fts_raw * norm_factor)

        amend_penalty = 0
        if re.search(r'\b(amendment|amending)\b', law_name.lower()):
            fts_norm      = max(fts_norm - 5, 0)
            amend_penalty = 5

        total = fts_norm + name + spec + vec

        all_laws_scored.append({
            "law_uid":       law_uid,
            "law_name":      law_name,
            "domain":        domain,
            "jurisdiction":  jurisdiction,
            "status":        status,
            "score":         total,
            "name_boosted":  name > 0,
            "fts_raw":       fts_raw,
            "fts_norm":      fts_norm,
            "norm_factor":   norm_factor,
            "section_count": sec_count,
            "name_boost":    name,
            "spec_boost":    spec,
            "vec_boost":     vec,
            "spec_keywords": spec_detail,
            "amend_penalty": amend_penalty,
        })

    all_laws_scored.sort(key=lambda x: x["score"], reverse=True)

    meta = {
        "fts_query_used":    fts_q_used,
        "fts_mode":          fts_mode,
        "query_keywords":    qwords,
        "tracked_keywords":  tracked_in_query,
        "all_laws_scored":   all_laws_scored,
        "total_active_laws": len(all_laws),
    }
    return all_laws_scored, meta


# ── LLM law selection ──────────────────────────────────────────────────────────

def _build_law_select_prompt(query: str, candidates: list) -> str:
    """Build prompt asking LLM to return ALL applicable laws from ranked list."""
    lines = [
        "You are a legal expert specialising in Indian labour law.",
        "From the ranked list of laws below, identify ALL laws that apply to answer the query.",
        "A law applies if the query directly concerns its subject matter.",
        "",
        f'Query: "{query}"',
        "",
        "Candidate laws (ranked by relevance score, highest first):",
    ]
    for c in candidates:
        lines.append(
            f"  [score={c['score']}] {c['law_uid']} | {c['law_name']} "
            f"| {c.get('domain','')} | {c.get('jurisdiction','')}"
        )
    lines += [
        "",
        "Return ONLY a JSON array of law_uid strings for all applicable laws.",
        "If multiple laws apply, include all of them.",
        "If only one applies, return only that one.",
        'Example: ["LAW_001", "LAW_023"]',
        "Do NOT include laws that are irrelevant to the query.",
        "Reply with JSON array only — no explanation, no extra text.",
    ]
    return "\n".join(lines)


def _parse_law_uids(response: str, candidates: list) -> list:
    """Extract list of law_uids from LLM JSON array response."""
    valid_uids = {c["law_uid"] for c in candidates}
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [uid for uid in data if uid in valid_uids]
    except Exception:
        pass
    # Fallback: extract any LAW_NNN patterns from raw text
    found = re.findall(r'LAW_\d+', response)
    result = []
    seen   = set()
    for uid in found:
        if uid in valid_uids and uid not in seen:
            result.append(uid)
            seen.add(uid)
    return result if result else ([candidates[0]["law_uid"]] if candidates else [])


# ── Enrich law_uid → full metadata ────────────────────────────────────────────

def _enrich(law_uid: str, all_scored: list) -> dict:
    """Fetch full law metadata from DB for a given law_uid."""
    conn = _conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT law_uid, law_name, domain, law_type, jurisdiction,
               act_number, enactment_year, status
        FROM laws WHERE law_uid = ?
    """, (law_uid,))
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "law_uid":        row[0],
            "law_name":       row[1],
            "domain":         row[2],
            "law_type":       row[3],
            "jurisdiction":   row[4],
            "act_number":     row[5],
            "enactment_year": row[6],
            "status":         row[7],
            "reasoning":      "Selected by LLM from hybrid-scored ranked list",
        }

    fallback = next((f for f in all_scored if f["law_uid"] == law_uid), None)
    if fallback:
        return {**fallback, "reasoning": "Fallback to scored finalist"}
    return {"law_uid": law_uid, "reasoning": "Unknown"}


# ── Public API ─────────────────────────────────────────────────────────────────

def select_laws(query: str) -> tuple:
    """
    Main entry point.
    Returns (laws_list, trace) where:
      - laws_list: list of applicable law dicts (no fixed cap)
      - trace: full scoring breakdown for history logging

    Flow:
      1. Score all laws → ranked list (FAISS + FTS5 + name-match + specificity)
      2. LLM always selects applicable laws from top-20 ranked list
      3. Return enriched law metadata for each selected law
    """
    query = query.strip()
    if not query:
        return [], {}

    all_scored, meta = _score_all_laws(query)

    if not all_scored:
        return [], {}

    trace = {
        "finalists":         all_scored[:20],
        "decision_type":     "llm-always",
        "fts_query_used":    meta.get("fts_query_used", ""),
        "fts_mode":          meta.get("fts_mode", ""),
        "query_keywords":    meta.get("query_keywords", []),
        "tracked_keywords":  meta.get("tracked_keywords", []),
        "all_laws_scored":   meta.get("all_laws_scored", []),
        "total_active_laws": meta.get("total_active_laws", 0),
    }

    # Send top 20 candidates to LLM for final selection
    top_candidates = [c for c in all_scored[:20] if c["score"] > 0]
    if not top_candidates:
        top_candidates = all_scored[:5]  # fallback: show top 5 even if score=0

    prompt      = _build_law_select_prompt(query, top_candidates)
    raw_response = call_select_laws(prompt)
    selected_uids = _parse_law_uids(raw_response, top_candidates)

    if not selected_uids:
        # Hard fallback: use top scored law
        selected_uids = [all_scored[0]["law_uid"]]
        trace["decision_type"] = "llm-fallback-top1"

    laws = [_enrich(uid, all_scored) for uid in selected_uids]
    return laws, trace
