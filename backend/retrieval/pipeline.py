"""
Full query pipeline:
  1. Law selection  (FAISS + FTS5 + name-match → LLM always decides)
  2. Section selection per law (FAISS + keyword → LLM always decides)
  3. Answer generation (LLM with full context: sections + defs + crossrefs + cases + version)
  4. YES/NO consistency fix
  5. Log to history
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.retrieval.doc_selector import select_laws
from backend.retrieval.section_selector import select_sections
from backend.llm.mistral_client import call_answer, is_live
from backend.logging.query_logger import log_query, init_history_table
from backend.logging.logger import get_logger

log = get_logger("pipeline")
init_history_table()


# ── System prompt (client-defined, mandatory format) ──────────────────────────

_SYSTEM_RULES = """\
You are a legal reasoning and response engine operating under strict citation and verification constraints.

----------------------------------------
STEP 1 — DETECT QUERY TYPE (MANDATORY, DO NOT OUTPUT THIS STEP)
----------------------------------------

Before answering, classify the query into one of two types:

TYPE A — COMPLIANCE / LEGAL OPINION query:
  Triggers: Can...? Is...? Are...? Should...? Must...? Does...? May...?
            Is it legal...? Is it allowed...? Is it mandatory...?
  Examples: "Can an employer deduct wages?" / "Is overtime compulsory?"
  → Use FORMAT A (YES / NO / DEPENDS)

TYPE B — INFORMATIONAL / DEFINITIONAL query:
  Triggers: What is...? What are...? How much...? How many...? Who is...?
            Define...? Explain...? List...? When is...? Which...?
  Examples: "What is the minimum age for employment?" /
            "What are the penalties under the Factories Act?" /
            "How much gratuity is an employee entitled to?"
  → Use FORMAT B (INFORMATION)

Use FORMAT A only when the query explicitly asks for a legal opinion or compliance verdict.
Use FORMAT B for all factual, definitional, or explanatory queries.

----------------------------------------
FORMAT A — COMPLIANCE QUERY OUTPUT (MANDATORY for TYPE A)
----------------------------------------

Start your response with ONE of the following (in capital letters):

YES / NO / DEPENDS

Then strictly follow this structure:

Reason:
Provide a clear, concise legal reasoning based only on verified sources. Do not speculate.

Citations:
List only verified legal sources used in reasoning. Use structured references:
- [Law Name, Section X]
- [Definition: Term, Law name]
- [WEB: Page name, website link]
- [Override: Law A overrides Law B]
- [Repeal: Law A repeals Law B]

Do NOT include any unverified sources.

Conclusion:
Provide a short, decisive legal conclusion aligned with the reasoning above.

----------------------------------------
FORMAT B — INFORMATIONAL QUERY OUTPUT (MANDATORY for TYPE B)
----------------------------------------

Start your response with:

INFORMATION

Then strictly follow this structure:

Answer:
Provide a direct, factual answer based only on verified legal sources. State the facts clearly.
Do not speculate. Do not use YES/NO/DEPENDS.

Citations:
List only verified legal sources used. Use structured references:
- [Law Name, Section X]
- [Definition: Term, Law name]
- [Override: Law A overrides Law B]
- [Repeal: Law A repeals Law B]

Conclusion:
Provide a short summary of the key legal fact(s) stated above.

----------------------------------------
STRICT RULES
----------------------------------------

1. SOURCE CONTROL
- Use VERIFIED_DB sources as primary.
- WEB_DERIVED sources may be used only when VERIFIED_DB is insufficient.
- Clearly distinguish WEB sources in citations.
- If sufficient citations are NOT available:
    -> Output: DEPENDS
    -> Clearly state insufficiency in Reason.

2. NO HALLUCINATION
- Do NOT assume facts, sections, or applicability.
- Do NOT generalize beyond cited law.

3. DETERMINISTIC OUTPUT
- No variation in the format is allowed
- No rephrasing of YES/NO/DEPENDS allowed
- Section labels must match exactly: Reason, Citations, Conclusion

4. NO EXTRA TEXT
- Do NOT add introductions, disclaimers, or explanations outside the format.
- Keep reasoning minimal (3-4 lines max).

5. PRECISION OVER COMPLETENESS
- Keep answer short and sharp.
- Avoid long paragraphs.

6. DEPENDS USAGE
Use DEPENDS when:
- facts are incomplete
- multiple legal outcomes possible
- jurisdictional ambiguity exists
- citations are insufficient

7. CONFLICT HANDLING
- Apply override/repeal logic from verified cross references
- Ignore repealed/inactive provisions

8. HARD FAILURE RULE
- If no valid citations are available:
    -> Output MUST be DEPENDS
    -> Reason must state: "Insufficient verified legal basis"

----------------------------------------
INTERNAL ALIGNMENT RULES (DO NOT OUTPUT)
----------------------------------------

- Only ACTIVE laws/sections must be used
- Repealed/overridden provisions must be ignored
- Definitions must be linked to active parent sections
- Cross-reference logic must already be resolved before answering

----------------------------------------
VERSION & CROSS-REFERENCE USAGE
----------------------------------------

1. Version Filtering (MANDATORY)
- Retrieve all relevant laws/sections.
- Check Versions sheet:
    -> Keep only entries marked CURRENT
    -> Discard all INACTIVE entries

2. Cross-Reference Resolution (MANDATORY)
- Check Cross References for remaining entries:
    -> If REPEAL: remove the repealed law/section
    -> If OVERRIDE: prefer the overriding law/section over the overridden one

3. Final Set
- Use ONLY:
    -> CURRENT laws/sections
    -> After applying override/repeal logic

4. Citations
- If override/repeal applied, include in citations:
    -> [Override: Law A overrides Law B]
    -> [Repeal: Law A repeals Law B]

5. Hard Rule
- Never use:
    -> INACTIVE laws
    -> Repealed sections
    -> Overridden provisions (if a valid override exists)

----------------------------------------
ILLUSTRATIONS (REFERENCE ONLY — NOT RULES)
----------------------------------------

Example 1:
Query: Is an employer required to constitute an ICC under POSH?

YES

Reason:
Employer with 10 or more employees must constitute an ICC.

Citations:
- [Sexual Harassment of Women at Workplace Act, 2013, Section 4]

Conclusion:
ICC constitution is mandatory.

----------------------------------------

Example 2:
Query: Can an RTI application be rejected for personal information?

DEPENDS

Reason:
Personal information is exempt unless larger public interest justifies disclosure.

Citations:
- [Right to Information Act, 2005, Section 8(1)(j)]

Conclusion:
Disclosure depends on public interest override.

----------------------------------------

Example 3:
Query: Are apprentices considered employees under labour laws?

NO

Reason:
Apprentices are trainees and not workers.

Citations:
- [Apprentices Act, 1961, Section 18]

Conclusion:
Apprentices are not employees.

----------------------------------------

Example 4 (FORMAT B — Informational):
Query: What is the minimum age for employment under Indian labour laws?

INFORMATION

Answer:
The minimum age for employment is 14 years. Adolescents between 14 and 18 years may work in non-hazardous occupations only, with restrictions on working hours and conditions. Employment of children below 14 years is prohibited in all establishments.

Citations:
- [Child and Adolescent Labour (Prohibition and Regulation) Act, 1986, Section 3]
- [Occupational Safety, Health and Working Conditions Code, 2020, Section 2]

Conclusion:
Minimum age for employment is 14 years. Children below 14 are completely prohibited from employment.

----------------------------------------

Example 5 (FORMAT B — Informational):
Query: What are the penalties for non-payment of gratuity?

INFORMATION

Answer:
An employer who fails to pay gratuity is liable to imprisonment of up to 1 year or a fine of up to Rs. 20,000 or both. For non-payment of gratuity specifically, the minimum imprisonment is 6 months unless the court gives special reasons.

Citations:
- [Payment of Gratuity Act, 1972, Section 9]

Conclusion:
Non-payment of gratuity attracts imprisonment up to 1 year and/or fine up to Rs. 20,000.

"""


# ── Build LLM prompt ──────────────────────────────────────────────────────────

def _build_law_block(law: dict, search: dict) -> str:
    """Build the legal text block for one law."""
    law_header = (
        f"Act: {law.get('law_name', '')}\n"
        f"Act Number: {law.get('act_number', '')}\n"
        f"Jurisdiction: {law.get('jurisdiction', '')}\n"
    )

    # Version — CURRENT only
    version    = search.get("version", {})
    ver_type   = (version.get("version_type") or "").strip().upper()
    ver_summary = (version.get("summary_legal_text") or "")[:800]
    ver_from   = (version.get("effective_from") or "").split(" ")[0]
    ver_line   = ""
    if ver_type == "CURRENT":
        ver_line = f"Version: CURRENT{f' (effective from {ver_from})' if ver_from else ''}\n"
    overview = f"Overview: {ver_summary}\n" if ver_summary else ""

    # Sections
    section_texts = []
    for sec in search.get("sections", []):
        label = sec.get("section_label", "")
        text  = sec.get("exact_text", "")
        if text:
            section_texts.append(f"### {label}\n{text}")
    sections_block = "\n\n".join(section_texts) if section_texts else "No sections retrieved."

    # Definitions
    defs_text = ""
    if search.get("definitions"):
        defs_text = "DEFINITIONS:\n" + "\n".join(
            f'- {d["term"]}: {d["text"]}' for d in search["definitions"][:5] if d.get("term")
        ) + "\n"

    # Cross-references — REPEAL/OVERRIDE types for LLM citation logic
    crossref_text = ""
    relevant_crs  = [
        cr for cr in search.get("crossrefs", [])
        if (cr.get("rel") or "").strip().upper() in ("REPEAL", "OVERRIDE", "SUPERSEDE", "AMEND")
    ]
    if relevant_crs:
        lines = []
        for cr in relevant_crs[:5]:
            rel    = (cr.get("rel") or "").strip().capitalize()
            to_law = (cr.get("to_law") or "").strip()
            to_sec = (cr.get("to_sec") or "").strip()
            entry  = f"- [{rel}: {law.get('law_name', '')} {rel}s {to_law}"
            if to_sec:
                entry += f" ({to_sec})"
            entry += "]"
            lines.append(entry)
        crossref_text = "CROSS-REFERENCES:\n" + "\n".join(lines) + "\n"

    # Cases — top 3
    cases_text = ""
    case_lines  = []
    for c in search.get("cases", [])[:3]:
        name    = (c.get("case_name") or "").strip()
        holding = (c.get("key_holding") or "").strip()
        effect  = (c.get("interpretive_effect") or "").strip()
        citation = (c.get("citation") or "").strip()
        if name and (holding or effect):
            line = f"- {name}"
            if citation:
                line += f" [{citation}]"
            if holding:
                line += f": {holding}"
            if effect:
                line += f" | Effect: {effect}"
            case_lines.append(line)
    if case_lines:
        cases_text = "CASE LAW:\n" + "\n".join(case_lines) + "\n"

    return (
        f"--- {law.get('law_name', 'Law').upper()} ---\n"
        f"{law_header}{ver_line}{overview}"
        f"SECTIONS:\n{sections_block}\n"
        f"{defs_text}{crossref_text}{cases_text}"
    )


def _build_answer_prompt(query: str, laws: list, searches: list) -> str:
    laws_block = "\n\n".join(
        _build_law_block(law, search)
        for law, search in zip(laws, searches)
    )
    return (
        f"{_SYSTEM_RULES}"
        f"APPLICABLE LEGAL TEXT:\n\n{laws_block}\n\n"
        f"USER QUERY: {query}\n"
    )


# ── YES/NO consistency fix ────────────────────────────────────────────────────

def _fix_yes_no_consistency(answer: str) -> str:
    if not answer:
        return answer
    lines      = answer.strip().split('\n')
    first_line = lines[0].strip()
    # Skip fix for INFORMATION responses — they don't have YES/NO
    if re.match(r'\*{0,2}INFORMATION\*{0,2}\s*$', first_line, re.IGNORECASE):
        return answer
    opening_match = re.match(r'\*{0,2}(YES|NO|DEPENDS)\*{0,2}\s*$', first_line, re.IGNORECASE)
    if not opening_match:
        return answer
    first_word = opening_match.group(1).upper()

    conclusion_match = re.search(
        r'\*{0,2}Conclusion\*{0,2}\s*:\s*(.*?)$',
        answer, re.IGNORECASE | re.DOTALL
    )
    if not conclusion_match:
        return answer

    conclusion = re.sub(r'\*+', '', conclusion_match.group(1)).lower().strip()

    no_patterns = [
        r'\bnot\s+permissible\b', r'\bnot\s+permitted\b', r'\bnot\s+allowed\b',
        r'\bprohibited\b', r'\billegal\b', r'\bunlawful\b', r'\bforbidden\b',
        r'\bcannot\b', r'\bshall\s+not\b', r'\bmust\s+not\b', r'\bdo\s+not\b',
        r'\bnot\s+legally\b', r'\bnot\s+entitled\b', r'\bimpermissible\b',
        r'\bviolates?\b', r'\bin\s+violation\b',
        r'\bare\s+void\b', r'\bis\s+void\b', r'\bnot\s+valid\b',
        r'\bnot\s+comply\b', r'\bnon-complian', r'\bbarred\b',
    ]
    yes_patterns = [
        r'\bis\s+permissible\b', r'\bis\s+permitted\b', r'\bis\s+allowed\b',
        r'\bis\s+mandatory\b', r'\bis\s+required\b',
        r'\blegally\s+valid\b', r'\blegally\s+permissible\b',
        r'\bis\s+entitled\b', r'\bare\s+entitled\b',
        r'\bmust\s+be\s+(?!avoided|stopped|terminated|closed|rejected)\b',
        r'\bis\s+legal\b', r'\bis\s+obligat', r'\blegally\s+bound\b',
    ]

    is_no  = any(re.search(p, conclusion) for p in no_patterns)
    is_yes = any(re.search(p, conclusion) for p in yes_patterns)

    if is_no and not is_yes and first_word == 'YES':
        return re.sub(r'\*{0,2}YES\*{0,2}', 'NO', answer, count=1, flags=re.IGNORECASE)
    if is_yes and not is_no and first_word == 'NO':
        return re.sub(r'\*{0,2}NO\*{0,2}', 'YES', answer, count=1, flags=re.IGNORECASE)
    return answer


# ── Output formatter ──────────────────────────────────────────────────────────

def _format_output(query: str, laws: list, searches: list, answer: str) -> dict:
    primary_law    = laws[0]
    primary_search = searches[0]

    sections_out = []
    multi = len(laws) > 1
    for law, search in zip(laws, searches):
        prefix = f"[{law.get('law_name', '')}] " if multi else ""
        for sec in search.get("sections", []):
            label = sec.get("section_label", "")
            if label:
                sections_out.append(f"{prefix}{label}")

    crossrefs_out = []
    for search in searches:
        for cr in search.get("crossrefs", []):
            rel    = (cr.get("rel") or "").strip().capitalize()
            to_law = (cr.get("to_law") or "").strip()
            to_sec = (cr.get("to_sec") or "").strip()
            if to_law:
                entry = f"{rel}: {to_law}" + (f" ({to_sec})" if to_sec else "")
                if entry not in crossrefs_out:
                    crossrefs_out.append(entry)

    cases_out  = []
    seen_cases = set()
    for search in searches:
        for c in search.get("cases", []):
            name = (c.get("case_name") or "").strip()
            if name and name not in seen_cases:
                seen_cases.add(name)
                cases_out.append({
                    "name":     name,
                    "citation": (c.get("citation") or "").strip(),
                    "court":    (c.get("court") or "").strip(),
                    "effect":   (c.get("interpretive_effect") or "").strip(),
                    "holding":  (c.get("key_holding") or "").strip(),
                })

    query_lower  = query.lower()
    answer_lower = answer.lower()
    all_defs, seen_terms = [], set()
    for search in searches:
        for d in search.get("definitions", []):
            term = (d.get("term") or "").strip()
            if term and term not in seen_terms:
                seen_terms.add(term)
                all_defs.append(d)

    priority, rest = [], []
    for d in all_defs:
        term = (d.get("term") or "").strip()
        text = (d.get("text") or "").strip()
        if not term or not text:
            continue
        if term.lower() in query_lower or term.lower() in answer_lower:
            priority.append({"term": term, "definition": text})
        else:
            rest.append({"term": term, "definition": text})
    defs_out = (priority + rest)[:6]

    version  = primary_search.get("version") or {}
    ver_type = (version.get("version_type") or "").strip().upper()
    eff_from = (version.get("effective_from") or "").split(" ")[0]
    eff_to   = (version.get("effective_to") or "").split(" ")[0]

    pipeline_details = {
        "doc_selection": {
            "reasoning":  primary_law.get("reasoning", ""),
            "selected_law": primary_law.get("law_name", ""),
            "all_laws":   [l.get("law_name", "") for l in laws],
        },
        "section_selection": {
            "method": "FAISS + keyword hybrid scoring → LLM section selection",
            "sections": [
                f"{sec.get('section_label', '')} (score={sec.get('score', 0)})"
                for search in searches for sec in search.get("sections", [])
            ],
        },
    }

    laws_out = [
        {
            "name":        l.get("law_name", ""),
            "act_number":  l.get("act_number", ""),
            "domain":      l.get("domain", ""),
            "type":        (l.get("law_type") or "").capitalize(),
            "jurisdiction": l.get("jurisdiction", ""),
            "year":        l.get("enactment_year", ""),
            "status":      (l.get("status") or "").capitalize(),
        }
        for l in laws
    ]

    return {
        "query":            query,
        "answer":           answer,
        "pipeline_details": pipeline_details,
        "law":              laws_out[0],
        "laws":             laws_out,
        "sections":         sections_out,
        "definitions":      defs_out,
        "crossrefs":        crossrefs_out,
        "cases":            cases_out,
        "version": {
            "type":           ver_type,
            "effective_from": eff_from,
            "effective_to":   eff_to,
        },
        "llm_live": is_live(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(query: str) -> dict:
    """Full pipeline. Returns formatted output dict for Gradio UI."""
    query = query.strip()
    if not query:
        return {"error": "Empty query"}

    t_start = time.time()

    # Step 1: Select applicable laws (FAISS + FTS5 + name-match → LLM decides)
    log.info("STEP 1 >> Law selection started")
    t1   = time.time()
    laws, trace = select_laws(query)
    log.info(f"STEP 1 >> Done | {int((time.time()-t1)*1000)}ms | decision={trace.get('decision_type','?')}")

    if not laws:
        log.error("STEP 1 >> No matching law found")
        return {
            "error": "No matching law found. Try rephrasing with specific act names or legal terms.",
            "query": query,
        }

    for i, l in enumerate(laws, 1):
        log.info(f"STEP 1 >> Law {i}: [{l.get('law_uid','')}] {l.get('law_name','')}")

    # Step 2: Section selection per law (FAISS + keyword → LLM decides)
    log.info("STEP 2 >> Section selection started")
    t2      = time.time()
    searches = []
    for law in laws:
        search = select_sections(law_uid=law["law_uid"], query=query)
        searches.append(search)
        log.info(f"STEP 2 >> [{law.get('law_uid','')}] {len(search.get('sections',[]))} sections selected")
    log.info(f"STEP 2 >> Done | {int((time.time()-t2)*1000)}ms")

    # Check primary law has substantive text
    primary_search = searches[0]
    has_text = any(
        (sec.get("exact_text") or "").strip()
        for sec in primary_search.get("sections", [])
    )
    if not primary_search.get("sections") or not has_text:
        log.warning("STEP 2 >> No section text — returning empty answer")
        answer = (
            "No information available for this query in the provided sections of "
            f"{laws[0].get('law_name', 'the selected law')}."
        )
        _do_log(query, laws, trace, answer, t_start)
        return _format_output(query, laws, searches, answer)

    # Step 3: Generate answer
    log.info("STEP 3 >> Sending prompt to LLM...")
    t3     = time.time()
    prompt = _build_answer_prompt(query, laws, searches)
    log.info(f"STEP 3 >> Prompt length: {len(prompt)} chars")
    answer = call_answer(prompt)
    log.info(f"STEP 3 >> Done | {int((time.time()-t3)*1000)}ms")

    # Step 4: Fix YES/NO consistency
    original_first = (answer or "").strip().split('\n')[0].strip().upper()
    answer         = _fix_yes_no_consistency(answer)
    fixed_first    = (answer or "").strip().split('\n')[0].strip().upper()
    if original_first != fixed_first:
        log.warning(f"STEP 4 >> YES/NO corrected: {original_first} -> {fixed_first}")

    total_ms = int((time.time() - t_start) * 1000)
    log.info(f"DONE >> Total pipeline time: {total_ms}ms")

    # Step 5: Log to history
    _do_log(query, laws, trace, answer, t_start)

    return _format_output(query, laws, searches, answer)


def _do_log(query: str, laws: list, trace: dict, answer: str, t_start: float):
    try:
        processing_ms = int((time.time() - t_start) * 1000)
        log_query(
            query=query,
            laws_selected=[l.get("law_name", "") for l in laws],
            decision_type=trace.get("decision_type", "unknown"),
            finalists=trace.get("finalists", []),
            fts_query_used=trace.get("fts_query_used", ""),
            answer=answer,
            processing_ms=processing_ms,
            trace=trace,
        )
    except Exception as e:
        print(f"[pipeline] History log error: {e}")
