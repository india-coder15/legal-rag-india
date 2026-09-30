"""
LiteLLM wrapper for the self-hosted Gemma endpoint.
- Mock mode when endpoint is unreachable.
- Token-aware: trims context to stay within 65k window.
- Endpoint: http://44.223.191.230:8000 | Model: gemma-4-31b-it
"""
import os
import sys
import json
import socket
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import LLM_ENDPOINT, LLM_MODEL, LLM_MAX_TOKENS, LLM_ANSWER_TOKENS, LLM_SELECT_TOKENS
from backend.logging.logger import get_logger

_log = get_logger("llm")

_FORCE_MOCK       = os.getenv("MOCK_LLM", "false").lower() == "true"
_ENDPOINT_CHECKED = False
_ENDPOINT_ALIVE   = False


def _check_endpoint() -> bool:
    global _ENDPOINT_CHECKED, _ENDPOINT_ALIVE
    if _ENDPOINT_CHECKED:
        return _ENDPOINT_ALIVE
    _ENDPOINT_CHECKED = True
    if _FORCE_MOCK:
        _ENDPOINT_ALIVE = False
        return False
    try:
        parsed = urlparse(LLM_ENDPOINT)
        host = parsed.hostname
        port = parsed.port or 80
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        _ENDPOINT_ALIVE = True
    except Exception:
        _ENDPOINT_ALIVE = False
    return _ENDPOINT_ALIVE


def reset_endpoint_check():
    global _ENDPOINT_CHECKED, _ENDPOINT_ALIVE
    _ENDPOINT_CHECKED = False
    _ENDPOINT_ALIVE   = False


def is_live() -> bool:
    return _check_endpoint()


# ── Token helpers ──────────────────────────────────────────────────────────────

def token_count(text: str) -> int:
    return max(1, len(text) // 4)


def trim_to_limit(text: str, reserve: int = 2000) -> str:
    max_chars = (LLM_MAX_TOKENS - reserve) * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[trimmed to fit context window]"


# ── Mock responses ─────────────────────────────────────────────────────────────

def _mock_select_laws(prompt: str) -> str:
    import re
    match = re.search(r'LAW_\d+', prompt)
    uid = match.group(0) if match else "LAW_001"
    return json.dumps([uid])


def _mock_select_sections(prompt: str) -> str:
    import re
    matches = re.findall(r'SEC_\w+', prompt)
    return json.dumps(matches[:2] if matches else [])


def _mock_answer(prompt: str) -> str:
    return (
        "[MOCK ANSWER — LLM endpoint offline]\n\n"
        "This is a placeholder response. Activate the LLM endpoint at "
        f"{LLM_ENDPOINT} and call reset_endpoint_check() to get real answers."
    )


# ── LiteLLM call ──────────────────────────────────────────────────────────────

def _litellm_call(prompt: str, max_tokens: int) -> str:
    global _ENDPOINT_ALIVE
    import litellm
    import time
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy")
    t = time.time()
    try:
        response = litellm.completion(
            model=f"openai/{LLM_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            api_base=f"{LLM_ENDPOINT}/v1",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        _log.info(f"LLM >> {int((time.time()-t)*1000)}ms | len={len(result)}")
        return result
    except Exception as e:
        _log.error(f"LLM >> FAILED | {int((time.time()-t)*1000)}ms | {e}")
        _ENDPOINT_ALIVE = False
        raise


# ── Public API ─────────────────────────────────────────────────────────────────

def call_select_laws(prompt: str) -> str:
    """
    LLM selects ALL applicable laws from ranked candidate list.
    Returns raw string expected to be a JSON array: ["LAW_001", "LAW_023"]
    """
    if not _check_endpoint():
        return _mock_select_laws(prompt)
    try:
        return _litellm_call(trim_to_limit(prompt, reserve=LLM_SELECT_TOKENS), LLM_SELECT_TOKENS)
    except Exception:
        return _mock_select_laws(prompt)


def call_select_sections(prompt: str) -> str:
    """
    LLM selects ALL applicable sections from ranked candidate list.
    Returns raw string expected to be a JSON array: ["SEC_001_004", "SEC_001_002"]
    """
    if not _check_endpoint():
        return _mock_select_sections(prompt)
    try:
        return _litellm_call(trim_to_limit(prompt, reserve=LLM_SELECT_TOKENS), LLM_SELECT_TOKENS)
    except Exception:
        return _mock_select_sections(prompt)


def call_answer(prompt: str) -> str:
    """Call LLM to generate final YES/NO/DEPENDS answer."""
    if not _check_endpoint():
        return _mock_answer(prompt)
    try:
        return _litellm_call(trim_to_limit(prompt, reserve=LLM_ANSWER_TOKENS), LLM_ANSWER_TOKENS)
    except Exception:
        return _mock_answer(prompt)


def call_summarize_batch(sections: list) -> list:
    """Summarize a batch of sections in a single LLM call."""
    if not _check_endpoint():
        return [f"[MOCK] Summary for: {s.get('section_label', '')}" for s in sections]

    parts = []
    for i, s in enumerate(sections):
        text  = (s.get("exact_text") or "")[:1200]
        kw    = (s.get("keywords") or "")[:200]
        label = s.get("section_label", "")
        parts.append(f"[{i+1}] Section: {label}\nKeywords: {kw}\nText: {text}")

    prompt = (
        "You are a legal analyst. For each section below, write a 1-2 line summary that captures:\n"
        "- What the section prescribes/regulates (amounts, limits, formulas, rights, penalties)\n"
        "- Key terms or thresholds mentioned\n"
        "- Who it applies to\n\n"
        "Return ONLY a JSON array of strings, one summary per section, in order.\n"
        "Example: [\"Prescribes gratuity formula: 15 days wages per year of service.\", \"Sets penalty...\"]\n\n"
        + "\n\n".join(parts)
    )

    try:
        raw = _litellm_call(trim_to_limit(prompt, reserve=1500), 1500)
        import re
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            summaries = json.loads(match.group(0))
            if isinstance(summaries, list) and len(summaries) == len(sections):
                return [str(s) for s in summaries]
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw.split("\n") if l.strip()]
        return (lines + [""] * len(sections))[:len(sections)]
    except Exception:
        return [f"Summary unavailable: {s.get('section_label', '')}" for s in sections]
