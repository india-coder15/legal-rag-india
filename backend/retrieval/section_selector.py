"""
Step 2 of retrieval: for each selected law, find all applicable sections.

Flow:
  1. FAISS section search filtered to law_uid → top 30 by vector similarity
  2. Keyword scoring (label + keywords + summary + text matching)
  3. Hybrid merge per section: keyword_score + vec_boost(0-20)
     → ranked section list
  4. LLM always decides: "Which of these sections apply to the query?"
     → returns only applicable sections (no fixed cap)
  5. Fetch enrichment: definitions + cross-refs + cases + version (CURRENT only)
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import DB_PATH, HYBRID_VEC_WEIGHT_SECTION
from backend.llm.mistral_client import call_select_sections

_STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall",
    "should","may","might","can","could","of","in","to","for",
    "with","on","at","by","from","as","into","through","during",
    "before","after","above","below","between","under","what","how",
    "which","who","whom","this","that","these","those","and","or",
    "but","if","not","no","so","than","too","very","just","about",
    "it","its","my","your","we","they","he","she","any","all",
}

_BOILERPLATE = (
    "short title","commencement","repeal","power to make rules",
    "protection of action","cognizance of offence","amendment of",
    "amendments to","laying of","power of central government to give direction",
    "power to remove difficult",
)

_DOMAIN_LABELS = [
    "audio-visual","mine ","mines ","dock ","plantation",
    "newspaper","journalist","cine-worker",
]


def _conn():
    return sqlite3.connect(str(DB_PATH))


def _query_words(query: str) -> list:
    words = re.split(r'\W+', query.lower())
    return [w for w in words if w and len(w) > 2 and w not in _STOP_WORDS]


def _fuzzy_in(word: str, text: str) -> bool:
    if word in text:
        return True
    text_dehyph = text.replace("-", " ")
    if word in text_dehyph:
        return True
    if len(word) >= 5:
        stem = word[:min(len(word) - 1, 6)]
        if stem in text or stem in text_dehyph:
            return True
    if len(word) >= 5:
        for tw in re.split(r'\W+', text):
            if len(tw) >= 4 and word.startswith(tw):
                return True
    return False


def _score_section(sec: dict, qwords: list, law_name_words: set, cur) -> int:
    """Keyword-based section scoring."""
    score       = 0
    label_lower   = sec["section_label"].lower()
    kw_lower      = sec["keywords"].lower()
    summary_lower = sec["llm_summary"].lower()
    text_lower    = sec["exact_text"][:3000].lower()

    if not kw_lower.strip():
        text_words = re.split(r'\W+', text_lower)
        pseudo, seen_pw = [], set()
        for tw in text_words:
            if len(tw) >= 4 and tw not in _STOP_WORDS and tw not in seen_pw:
                pseudo.append(tw)
                seen_pw.add(tw)
                if len(pseudo) >= 30:
                    break
        kw_lower = " ".join(pseudo)

    for w in qwords:
        if _fuzzy_in(w, label_lower):
            score += 1 if w in law_name_words else 5
        if _fuzzy_in(w, kw_lower):
            score += 3
        if _fuzzy_in(w, summary_lower):
            score += 3
        if _fuzzy_in(w, text_lower):
            score += 1

    cur.execute(
        "SELECT term FROM definitions WHERE sub_section_uid LIKE ? || '%'",
        (sec["section_uid"],)
    )
    for (term,) in cur.fetchall():
        if term and any(w in term.lower() for w in qwords):
            score += 2

    if any(bp in label_lower for bp in _BOILERPLATE):
        score = max(0, score - 5)

    label_clean = label_lower.strip().strip(".,;: ")
    if label_clean in ("definitions","defination","definition",
                       "interpretation","definitions and interpretation"):
        score = min(score, 3)

    query_text = " ".join(qwords)
    for domain in _DOMAIN_LABELS:
        dterm = domain.strip()
        if dterm in label_lower and dterm not in query_text:
            score = max(0, score - 8)
            break

    return score


# ── FAISS vector scoring for sections ─────────────────────────────────────────

def _vector_score_sections(query: str, law_uid: str) -> dict:
    """
    Embed query → search all section vectors in parallel batches via FAISS,
    filtered to sections of this specific law.
    Returns {section_uid: vec_boost (0-HYBRID_VEC_WEIGHT_SECTION)}.
    Returns {} gracefully if FAISS index not built yet.
    """
    try:
        from backend.embeddings.encoder import encode_query
        from backend.embeddings.faiss_index import search_sections
        query_vec = encode_query(query)
        results   = search_sections(query_vec, top_k=30, law_uids=[law_uid])
        if not results:
            return {}
        top_score = results[0][1] if results[0][1] > 0 else 1.0
        return {
            uid: int((score / top_score) * HYBRID_VEC_WEIGHT_SECTION)
            for uid, score in results
        }
    except Exception:
        return {}


# ── LLM section selection ──────────────────────────────────────────────────────

def _build_section_select_prompt(query: str, law_name: str, candidates: list) -> str:
    """Build prompt asking LLM to return ALL applicable sections for the query."""
    lines = [
        "You are a legal expert specialising in Indian labour law.",
        f"Law: {law_name}",
        "",
        "From the candidate sections below, identify ALL sections that are relevant",
        "to answer the query. Include a section if it directly addresses the query topic.",
        "",
        f'Query: "{query}"',
        "",
        "Candidate sections (ranked by relevance score, highest first):",
    ]
    for sec in candidates:
        preview = (sec.get("exact_text") or "")[:150].replace("\n", " ")
        lines.append(
            f"  [score={sec['score']}] {sec['section_uid']} | "
            f"{sec['section_label']} | {preview}..."
        )
    lines += [
        "",
        "Return ONLY a JSON array of section_uid strings for all applicable sections.",
        "Include all sections that are relevant — there is no limit.",
        'Example: ["SEC_001_004", "SEC_001_002", "SEC_001_007"]',
        "Reply with JSON array only — no explanation, no extra text.",
    ]
    return "\n".join(lines)


def _parse_section_uids(response: str, candidates: list) -> list:
    """Extract list of section_uids from LLM JSON array response."""
    valid_uids = {s["section_uid"] for s in candidates}
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [uid for uid in data if uid in valid_uids]
    except Exception:
        pass
    found  = re.findall(r'SEC_\w+', response)
    result, seen = [], set()
    for uid in found:
        if uid in valid_uids and uid not in seen:
            result.append(uid)
            seen.add(uid)
    return result if result else ([candidates[0]["section_uid"]] if candidates else [])


# ── Public API ─────────────────────────────────────────────────────────────────

def select_sections(law_uid: str, query: str, max_tokens: int = 8000) -> dict:
    """
    Score and select applicable sections for a law given a query.

    Returns dict:
        sections:    list of selected section dicts (chosen by LLM)
        definitions: list of {term, text}
        crossrefs:   list of {rel, to_law, to_sec}
        cases:       list of {case_name, court, citation, year, key_holding, interpretive_effect}
        version:     dict {version_type, effective_from, effective_to, summary_legal_text}
    """
    conn   = _conn()
    cur    = conn.cursor()
    qwords = _query_words(query)
    if not qwords:
        qwords = re.split(r'\W+', query.lower())[:5]

    # ── 1. Fetch active sections ──────────────────────────────────────────────
    cur.execute("""
        SELECT section_uid, section_label, keywords, exact_text, llm_summary
        FROM sections
        WHERE law_uid = ?
        AND section_uid NOT IN (
            SELECT DISTINCT section_uid FROM versions
            WHERE upper(version_type) = 'INACTIVE' AND section_uid != ''
        )
        ORDER BY section_uid
    """, (law_uid,))
    all_sections = [
        {"section_uid": r[0], "section_label": r[1] or "",
         "keywords": r[2] or "", "exact_text": r[3] or "", "llm_summary": r[4] or ""}
        for r in cur.fetchall()
    ]

    if not all_sections:
        conn.close()
        return {"sections": [], "definitions": [], "crossrefs": [], "cases": [], "version": {}}

    # Law name words (to deprioritize in label scoring)
    cur.execute("SELECT law_name FROM laws WHERE law_uid = ?", (law_uid,))
    law_name_row  = cur.fetchone()
    law_name      = law_name_row[0] if law_name_row else ""
    law_name_words = {
        w.lower() for w in re.split(r'\W+', law_name)
        if len(w) > 3 and w.lower() not in _STOP_WORDS
    } if law_name else set()

    # ── 2. Keyword scoring ────────────────────────────────────────────────────
    for sec in all_sections:
        sec["keyword_score"] = _score_section(sec, qwords, law_name_words, cur)

    # ── 3. FAISS vector boost ─────────────────────────────────────────────────
    vec_boosts = _vector_score_sections(query, law_uid)
    for sec in all_sections:
        sec["vec_boost"] = vec_boosts.get(sec["section_uid"], 0)
        sec["score"]     = sec["keyword_score"] + sec["vec_boost"]

    # ── 4. Sort by hybrid score, keep top candidates for LLM ─────────────────
    all_sections.sort(key=lambda s: s["score"], reverse=True)
    candidates = [s for s in all_sections if s["score"] > 0][:20]

    # Ensure we always have something to show LLM
    if not candidates:
        candidates = [
            s for s in all_sections
            if not any(bp in s["section_label"].lower() for bp in ("short title","commencement","repeal"))
        ][:5]

    # ── 5. LLM always decides which sections apply ────────────────────────────
    if candidates:
        prompt        = _build_section_select_prompt(query, law_name, candidates)
        raw_response  = call_select_sections(prompt)
        selected_uids = _parse_section_uids(raw_response, candidates)

        if not selected_uids and candidates:
            selected_uids = [candidates[0]["section_uid"]]

        uid_to_sec = {s["section_uid"]: s for s in all_sections}
        selected   = [uid_to_sec[uid] for uid in selected_uids if uid in uid_to_sec]
    else:
        selected = []

    # Token budget guard — trim if total text exceeds budget
    char_budget = max_tokens * 4
    final, total_chars = [], 0
    for sec in selected:
        text_len = len(sec.get("exact_text", ""))
        if total_chars + text_len > char_budget and final:
            break
        final.append(sec)
        total_chars += text_len
    selected = final if final else selected[:3]

    # ── 6. Definitions ────────────────────────────────────────────────────────
    definitions = []
    seen_terms  = set()
    for sec in selected:
        cur.execute("""
            SELECT term, definition_text FROM definitions
            WHERE sub_section_uid LIKE ? || '%'
        """, (sec["section_uid"],))
        for r in cur.fetchall():
            if r[0] and r[1] and r[0] not in seen_terms:
                definitions.append({"term": r[0], "text": r[1]})
                seen_terms.add(r[0])

    # Also pull from the law's definition section
    law_num = law_uid.replace("LAW_", "")
    cur.execute("""
        SELECT term, definition_text FROM definitions
        WHERE sub_section_uid LIKE ? || '%'
        ORDER BY definition_uid
    """, (f"SEC_{law_num.zfill(3)}_002",))
    query_lower = query.lower()
    for r in cur.fetchall():
        if r[0] and r[1] and r[0] not in seen_terms:
            if r[0].lower() in query_lower:  # only priority defs from def section
                definitions.append({"term": r[0], "text": r[1]})
                seen_terms.add(r[0])

    # ── 7. Cross-references ───────────────────────────────────────────────────
    cur.execute("""
        SELECT cr.relationship_type, l.law_name, cr.to_section_uid
        FROM cross_references cr
        LEFT JOIN laws l ON l.law_uid = cr.to_law_uid
        WHERE cr.from_law_uid = ?
    """, (law_uid,))
    crossrefs = [
        {"rel": r[0] or "", "to_law": r[1] or "", "to_sec": r[2] or ""}
        for r in cur.fetchall()
    ]

    # ── 8. Cases ──────────────────────────────────────────────────────────────
    cases = []
    if law_name:
        cur.execute("""
            SELECT case_name, court, citation, year, key_holding,
                   interpretive_effect, sections_interpreted
            FROM cases WHERE governing_law LIKE ?
        """, (f"%{law_name[:30]}%",))
        for r in cur.fetchall():
            cases.append({
                "case_name":            r[0] or "",
                "court":                r[1] or "",
                "citation":             r[2] or "",
                "year":                 r[3],
                "key_holding":          r[4] or "",
                "interpretive_effect":  r[5] or "",
                "sections_interpreted": r[6] or "",
            })

    # ── 9. Version (CURRENT only) ─────────────────────────────────────────────
    cur.execute("""
        SELECT version_type, effective_from, effective_to, summary_legal_text
        FROM versions
        WHERE law_uid = ? AND upper(version_type) = 'CURRENT'
        ORDER BY effective_from DESC
        LIMIT 1
    """, (law_uid,))
    ver_row = cur.fetchone()
    version = {}
    if ver_row:
        version = {
            "version_type":      (ver_row[0] or "").strip(),
            "effective_from":    ver_row[1] or "",
            "effective_to":      ver_row[2] or "",
            "summary_legal_text": ver_row[3] or "",
        }

    conn.close()

    return {
        "sections":    selected,
        "definitions": definitions,
        "crossrefs":   crossrefs,
        "cases":       cases,
        "version":     version,
    }
