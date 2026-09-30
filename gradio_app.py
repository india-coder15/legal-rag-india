import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import gradio as gr
from backend.retrieval.pipeline import run_query
from backend.llm.mistral_client import is_live, reset_endpoint_check
from backend.logging.query_logger import get_history, get_record, delete_query
from backend.logging.logger import get_logger

log = get_logger("api")


# ── Renderers ─────────────────────────────────────────────────────────────────

def _render_answer(result: dict) -> str:
    if result.get("error"):
        return f"⚠ {result['error']}"
    answer = (result.get("answer") or "").strip()
    if not answer:
        return "No answer generated."
    if not result.get("llm_live"):
        answer += ("\n\n---\n*Note: AI answer engine is currently offline. "
                   "Source references below are real data.*")
    return answer


def _render_sources(result: dict) -> str:
    if result.get("error") or not result.get("law"):
        return ""

    all_laws = result.get("laws") or [result["law"]]
    lines    = []

    if len(all_laws) > 1:
        lines.append(f"### Applicable Laws ({len(all_laws)})")
    else:
        lines.append("### Applicable Law")
    lines.append("")

    ver = result.get("version", {})
    for i, law in enumerate(all_laws):
        name   = law.get("name", "")
        act_no = law.get("act_number", "")
        lines.append(f"**{i+1}. {name}**" + (f"  \n*{act_no}*" if act_no else ""))
        meta = []
        if law.get("domain"):       meta.append(f"**Domain:** {law['domain']}")
        if law.get("type"):         meta.append(f"**Type:** {law['type']}")
        if law.get("jurisdiction"): meta.append(f"**Jurisdiction:** {law['jurisdiction']}")
        if law.get("year"):         meta.append(f"**Year:** {law['year']}")
        if law.get("status"):       meta.append(f"**Status:** {law['status']}")
        if i == 0 and ver.get("type"):
            v = ver["type"].capitalize()
            if ver.get("effective_from"):
                v += f" (from {ver['effective_from']})"
            meta.append(f"**Version:** {v}")
        if meta:
            lines.append("  \n".join(meta))
        lines.append("")

    sections = [s for s in result.get("sections", [])
                if s not in ("Overview", "Sections", "Cross-References", "Case Law")]
    if sections:
        lines.append("**Relevant Sections:**")
        for s in sections[:8]:
            lines.append(f"- {s}")
        lines.append("")

    crossrefs = result.get("crossrefs", [])
    if crossrefs:
        lines.append("**Cross-References:**")
        for cr in crossrefs[:5]:
            lines.append(f"- {cr}")
        lines.append("")

    cases = result.get("cases", [])
    if cases:
        lines.append("**Case Law:**")
        for c in cases[:3]:
            n   = c.get("name", "")
            cit = c.get("citation", "")
            eff = c.get("effect", "")
            lines.append(
                f"- **{n}**" + (f" — {cit}" if cit else "") +
                (f"  \n  *{eff.capitalize()}*" if eff else "")
            )
        lines.append("")

    defs = result.get("definitions", [])
    if defs:
        lines.append("**Definitions:**")
        for d in defs[:5]:
            term = d.get("term", "")
            text = (d.get("definition") or "").strip()
            if len(text) > 200:
                text = text[:200].rsplit(" ", 1)[0] + "…"
            if term:
                lines.append(f"- **{term}** — {text}")

    return "\n".join(lines)


def _render_pipeline(result: dict) -> str:
    if result.get("error"):
        return ""
    pd    = result.get("pipeline_details", {})
    lines = []

    ds = pd.get("doc_selection", {})
    lines.append("### Document Selection")
    if ds.get("reasoning"):
        lines.append(f"- *Thinking:* {ds['reasoning']}")
    all_laws = ds.get("all_laws", [])
    if len(all_laws) > 1:
        lines.append(f"- **Applicable Laws ({len(all_laws)}):**")
        for l in all_laws:
            lines.append(f"  - {l}")
    elif ds.get("selected_law"):
        lines.append(f"- **Selected Law:** {ds['selected_law']}")
    lines.append("")

    ss = pd.get("section_selection", {})
    lines.append("### Section Selection")
    if ss.get("method"):
        lines.append(f"- *Method:* {ss['method']}")
    sections = ss.get("sections", [])
    if sections:
        lines.append(f"- **Sections:** {', '.join(sections[:8])}")

    return "\n".join(lines)


# ── History ───────────────────────────────────────────────────────────────────

def _render_history_table() -> str:
    records = get_history(limit=50)
    if not records:
        return "*No queries processed yet. Run a query first.*"
    lines = [
        "| # | Time | Laws Selected | Decision | Time (ms) | Query |",
        "|---|------|--------------|----------|-----------|-------|",
    ]
    for r in records:
        laws    = ", ".join(r["laws_selected"]) if r["laws_selected"] else "—"
        if len(laws) > 60:
            laws = laws[:57] + "…"
        query   = (r["query"] or "")[:60].replace("|", " ").replace("\n", " ")
        if len(r["query"]) > 60:
            query += "…"
        lines.append(
            f"| {r['id']} | {r['timestamp']} | {laws} | "
            f"{r['decision_type'] or '—'} | {r['processing_ms']} | {query} |"
        )
    return "\n".join(lines)


def _render_history_detail(record_id_str: str) -> str:
    try:
        record_id = int(str(record_id_str).strip())
    except (ValueError, TypeError):
        return "*Enter a valid query # from the table above.*"

    r = get_record(record_id)
    if not r:
        return f"*No record found for # {record_id}.*"

    td    = r.get("trace_detail") or {}
    lines = []

    lines.append(f"## Query #{r['id']} — {r['timestamp']}")
    lines.append(f"\n**Query:**\n> {r['query']}\n")
    lines.append(
        f"**Processing Time:** {r['processing_ms']} ms  |  "
        f"**Decision:** `{r['decision_type']}`  |  "
        f"**Active Laws Scanned:** {td.get('total_active_laws', '?')}\n"
    )

    laws = r["laws_selected"] or []
    lines.append(f"### Laws Selected ({len(laws)})")
    for law in laws:
        lines.append(f"- **{law}**")
    lines.append("")

    lines.append("### Step 1 — Keywords Extracted from Query")
    qkw = td.get("query_keywords", [])
    lines.append(f"`{'` `'.join(qkw) if qkw else 'none'}`\n")

    lines.append("### Step 2 — Domain Keywords Found in Query (Specificity Boost)")
    tracked = td.get("tracked_keywords", [])
    if tracked:
        lines.append("| Keyword | Found in N laws | Boost Given |")
        lines.append("|---------|----------------|-------------|")
        for t in tracked:
            cnt = t.get("law_count", 0)
            b   = "+25" if cnt <= 3 else "+15" if cnt <= 8 else "+8" if cnt <= 15 else "+4" if cnt <= 25 else "0"
            lines.append(f"| `{t['keyword']}` | {cnt} laws | {b} |")
    else:
        lines.append("*No domain keywords matched.*")
    lines.append("")

    lines.append("### Step 3 — FTS (Full-Text Search)")
    lines.append(f"- FTS query: `{td.get('fts_query_used', r.get('fts_query_used', 'n/a'))}`")
    lines.append(f"- Mode: `{td.get('fts_mode', 'n/a')}`\n")

    all_scored = td.get("all_laws_scored", [])
    if all_scored:
        all_scored_sorted = sorted(all_scored, key=lambda x: x.get("score", 0), reverse=True)
        lines.append(f"### Step 4 — Score Breakdown for ALL {len(all_scored)} Laws")
        lines.append(
            "*fts_raw=raw hits | fts_norm=after size penalty | "
            "name=name-match | spec=keyword rarity | vec=FAISS vector | total=final*\n"
        )
        lines.append("| # | Law | Sections | fts_raw | fts_norm | name | spec | vec | **total** |")
        lines.append("|---|-----|----------|---------|----------|------|------|-----|-----------|")
        for i, l in enumerate(all_scored_sorted, 1):
            law_name = (l.get("law_name") or "")[:45]
            lines.append(
                f"| {i} | {law_name} | {l.get('section_count',0)} "
                f"| {l.get('fts_raw',0)} | {l.get('fts_norm',0)} "
                f"| {l.get('name_boost',0)} | {l.get('spec_boost',0)} "
                f"| {l.get('vec_boost',0)} | **{l.get('score',0)}** |"
            )
        lines.append("")

    lines.append("### Step 5 — Final Decision")
    dt = r["decision_type"] or ""
    if "llm-always" in dt:
        lines.append("- **Method:** LLM always decides — top 20 candidates sent to LLM.")
    elif "fallback" in dt:
        lines.append("- **Method:** LLM fallback — top scored law used.")
    lines.append("")

    if r["answer_preview"]:
        lines.append("### Answer Preview")
        lines.append(f"> {r['answer_preview']}{'...' if len(r['answer_preview']) >= 300 else ''}")

    return "\n".join(lines)


def _handle_delete(record_id_str: str):
    try:
        record_id = int(str(record_id_str).strip())
    except (ValueError, TypeError):
        return "*Enter a valid query # to delete.*", _render_history_table()
    deleted = delete_query(record_id)
    if deleted:
        msg = f"*Query #{record_id} deleted successfully.*"
        log.info(f"History record #{record_id} deleted")
    else:
        msg = f"*No record found for # {record_id}.*"
    return msg, _render_history_table()


# ── Query handler ─────────────────────────────────────────────────────────────

def handle_query(query: str):
    if not (query or "").strip():
        return "*Enter a query and press Search.*", "", ""

    reset_endpoint_check()
    t0     = time.time()
    result = run_query(query)
    elapsed = int((time.time() - t0) * 1000)

    if not result.get("error"):
        laws     = result.get("laws") or [result.get("law", {})]
        law_names = [l.get("name", "") for l in laws]
        log.info(f"RESPONSE READY | laws={law_names} | total_time={elapsed}ms")

    return _render_answer(result), _render_sources(result), _render_pipeline(result)


# ── UI ────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="PageIndex Legal Retrieval") as app:

        gr.Markdown("# PageIndex Legal Retrieval — Pilot")
        gr.Markdown("#### Annexure B | Rhett")

        with gr.Row():
            query_input = gr.Textbox(
                label="Your Legal Query",
                placeholder="e.g. What are the minimum wage provisions under the Code on Wages?",
                lines=2, scale=5,
            )
            with gr.Column(scale=1, min_width=130):
                submit_btn = gr.Button("Search", variant="primary", size="lg")
                clear_btn  = gr.Button("Clear", size="sm")

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_out = gr.Markdown(value="*Your answer will appear here.*")
            with gr.TabItem("Sources"):
                sources_out = gr.Markdown(value="*Source details will appear here.*")
            with gr.TabItem("Pipeline Details"):
                pipeline_out = gr.Markdown(value="*Pipeline reasoning will appear here.*")
            with gr.TabItem("Query History"):
                gr.Markdown("### All Processed Queries — Backend Trace")
                refresh_btn   = gr.Button("Refresh History", variant="secondary", size="sm")
                history_table = gr.Markdown(value="*Click Refresh to load history.*")
                gr.Markdown("---\n**Enter a Query # from the table to see details or delete it:**")
                with gr.Row():
                    detail_id_input = gr.Textbox(label="Query #", placeholder="e.g. 5",
                                                 scale=1, max_lines=1)
                    detail_btn  = gr.Button("Show Details", variant="primary", scale=1)
                    delete_btn  = gr.Button("Delete This Query", variant="stop", scale=1)
                delete_status  = gr.Markdown(value="")
                history_detail = gr.Markdown(value="")

        gr.Examples(
            examples=[
                ["For payroll compliance, what counts as wages and how should minimum wages be determined in Rajasthan?"],
                ["What are the penalties for employing contract labour without registration?"],
                ["Under what conditions can an employer make deductions from wages?"],
                ["What is the definition of employee under the Employees State Insurance Act?"],
                ["What is the minimum age for employment under Indian labour laws?"],
            ],
            inputs=query_input,
        )

        outputs = [answer_out, sources_out, pipeline_out]
        submit_btn.click(fn=handle_query, inputs=query_input, outputs=outputs)
        query_input.submit(fn=handle_query, inputs=query_input, outputs=outputs)
        clear_btn.click(
            fn=lambda: ("*Your answer will appear here.*",
                        "*Source details will appear here.*",
                        "*Pipeline reasoning will appear here.*", ""),
            outputs=outputs + [query_input],
        )

        refresh_btn.click(fn=_render_history_table, inputs=[], outputs=history_table)
        detail_btn.click(fn=_render_history_detail, inputs=detail_id_input, outputs=history_detail)
        delete_btn.click(fn=_handle_delete, inputs=detail_id_input,
                         outputs=[delete_status, history_table])

    return app


def launch(share: bool = True):
    from backend.config import LLM_ENDPOINT, LLM_MODEL, DB_PATH, EXCEL_PATH
    log.info("=" * 60)
    log.info("SERVER STARTING — Legal RAG System")
    log.info(f"DB        : {DB_PATH}")
    log.info(f"EXCEL     : {EXCEL_PATH}")
    log.info(f"LLM       : {LLM_ENDPOINT}")
    log.info(f"MODEL     : {LLM_MODEL}")
    log.info("=" * 60)

    app  = build_app()
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    import socket
    for p in range(port, port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", p)) != 0:
                port = p
                break

    log.info(f"SERVER READY — http://127.0.0.1:{port}")
    app.launch(
        share=share,
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        theme=gr.themes.Soft(),
    )
