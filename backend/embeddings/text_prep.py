"""
Text preparation for embedding.
Combines relevant fields from DB rows into a single string per law / per section.
Keeps each string within ~512 tokens (model's max sequence length).
"""

_MAX_CHARS = 1800  # ~450 tokens — safe limit for 512-token model


def prepare_law_text(law_row: dict, section_labels: list) -> str:
    """
    Build embedding text for a law.
    Combines: law_name + domain + jurisdiction + act_number + top section labels.

    Example output:
        "Payment of Gratuity Act, 1972. Domain: Labour. Jurisdiction: Central.
         Act: No. 39 of 1972. Sections: Definitions, Gratuity payment,
         Continuous service, Nomination, Controlling authority, ..."
    """
    parts = []

    name = (law_row.get("law_name") or "").strip()
    if name:
        parts.append(name)

    domain = (law_row.get("domain") or "").strip()
    if domain:
        parts.append(f"Domain: {domain}")

    juris = (law_row.get("jurisdiction") or "").strip()
    if juris:
        parts.append(f"Jurisdiction: {juris}")

    act_no = (law_row.get("act_number") or "").strip()
    if act_no:
        parts.append(f"Act: {act_no}")

    if section_labels:
        labels_str = ", ".join(str(l) for l in section_labels[:20])
        parts.append(f"Sections: {labels_str}")

    text = ". ".join(parts)
    return text[:_MAX_CHARS]


def prepare_section_text(section_row: dict) -> str:
    """
    Build embedding text for a section.
    Combines: section_label + keywords + llm_summary + first 500 chars of exact_text.

    Example output:
        "Payment of Gratuity — Section 4: Payment of Gratuity.
         Keywords: gratuity, continuous service, fifteen days, wages.
         Summary: Prescribes gratuity formula: 15 days wages per year of service.
         Text: (1) Gratuity shall be payable to an employee on..."
    """
    parts = []

    label = (section_row.get("section_label") or "").strip()
    if label:
        parts.append(label)

    keywords = (section_row.get("keywords") or "").strip()
    if keywords:
        parts.append(f"Keywords: {keywords}")

    summary = (section_row.get("llm_summary") or "").strip()
    if summary:
        parts.append(f"Summary: {summary}")

    text = (section_row.get("exact_text") or "").strip()
    if text:
        parts.append(f"Text: {text[:500]}")

    combined = ". ".join(parts)
    return combined[:_MAX_CHARS]
