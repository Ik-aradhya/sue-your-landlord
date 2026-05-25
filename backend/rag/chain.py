import os
import re
from typing import Optional
from langchain_groq import ChatGroq
from core.config import settings
from core.session_store import session_store
from models.schemas import RAGResponse, Confidence
from rag.prompts import build_prompt, build_retrieval_queries
from rag.retriever import run_retrieval

# Initialise LLM once at module level
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.LLM_MODEL,
    temperature=0,
    max_tokens=1000
)

def call_llm(messages: list[dict]) -> dict:
    """
    Sends prompt to Groq and returns raw text response.

    Input:  list of message dicts (system + user)
    Output: { llm_output: str, error_type: str | None }

    Guarantee: does not interpret or format the response.
               only handles communication with the API.
    """
    try:
        response = llm.invoke(messages)
        return {
            "llm_output": response.content,
            "error_type": None
        }
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()
        credit_markers = [
            "rate_limit",
            "rate limit",
            "429",
            "too many requests",
            "quota",
            "credit",
            "billing",
            "insufficient",
            "resource exhausted",
        ]
        if any(marker in error_lower for marker in credit_markers):
            return {"llm_output": None, "error_type": "LLM_CREDITS_UNAVAILABLE"}
        return {"llm_output": None, "error_type": f"LLM_ERROR: {error_msg}"}


def format_response(
    llm_output: str,
    confidence: Confidence,
    law_chunks: Optional[list] = None,
    lease_chunks: Optional[list] = None,
    question: str = "",
) -> RAGResponse:
    """
    Parses raw LLM output into structured RAGResponse.

    Confidence is parsed from the model response, then constrained by
    parser-level quality checks.
    """

    def extract_field(label: str, text: str) -> str:
        """Extracts content after a label like 'ANSWER:' """
        pattern = rf"{label}:\s*(.*?)(?=\n[A-Z ]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def clean_lease_reference(raw: str) -> str:
        """Strips wrapping quotes and collapses extra whitespace from lease reference."""
        if not raw:
            return raw
        cleaned = raw.strip().strip('"').strip("'").strip("“”‘’")
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip().strip(',').strip()
        return cleaned

    def parse_confidence(raw: str) -> Optional[Confidence]:
        value = raw.strip().upper()
        if value.startswith("HIGH"):
            return Confidence.HIGH
        if value.startswith("MEDIUM"):
            return Confidence.MEDIUM
        if value.startswith("LOW"):
            return Confidence.LOW
        return None

    answer       = extract_field("ANSWER", llm_output)
    raw_legal_basis = extract_field("LEGAL BASIS", llm_output)
    lease_ref    = clean_lease_reference(extract_field("LEASE REFERENCE", llm_output))
    explanation  = extract_field("EXPLANATION", llm_output)
    conflict_raw = extract_field("CONFLICT", llm_output)
    llm_confidence = parse_confidence(extract_field("CONFIDENCE", llm_output))
    answer = re.sub(r"\s+", " ", format_answer(answer, question, explanation)).strip()
    lease_ref = format_lease_reference(
        raw_lease_reference=lease_ref,
        lease_chunks=lease_chunks or [],
        question=question,
        supporting_notes=explanation,
    )
    legal_basis = format_legal_basis(
        raw_legal_basis,
        law_chunks or [],
        question=question,
        answer=answer,
        explanation=explanation,
    )
    if (not legal_basis or legal_basis.lower() == "lease document") and lease_ref:
        legal_basis = format_lease_legal_basis(lease_chunks or [], question)
    answer = align_answer_with_legal_basis(answer, legal_basis)
    explanation = format_explanation(explanation, legal_basis)

    # If parsing failed — return safe fallback
    if not answer:
        return RAGResponse(
            answer="I was unable to parse the response. Please try again.",
            legal_basis="",
            lease_reference=None,
            explanation="",
            confidence=Confidence.LOW,
            conflict_flag=False,
            error_type="PARSING_FAILED"
        )

    conflict_flag = conflict_raw.strip().upper().startswith("YES")

    # Post-process confidence based on LLM output hedging
    HEDGE_PHRASES = [
        "one would need to refer",
        "does not explicitly",
        "may be entitled",
        "might be able to",
        "under certain conditions",
        "it depends",
        "cannot be determined",
        "unclear",
        "ambiguous",
        "possibly",
        "perhaps",
        "could be",
        "could have",
        "may have",
        "may be able",
        "would depend",
        "depends on",
        "not enough information",
        "only if",
        "it depends on your situation",
        "you may have rights",
        "consider seeking legal advice",
        "the context does not provide enough information",
        "consult a legal professional",
        "review the act",
        "seek legal advice",
        "the next step would be to review",
    ]

    combined_lower = f"{answer} {explanation}".lower()
    hedged = any(phrase in combined_lower for phrase in HEDGE_PHRASES)

    final_confidence = llm_confidence or confidence

    if final_confidence == Confidence.HIGH and not (legal_basis or lease_ref):
        final_confidence = Confidence.MEDIUM

    if final_confidence == Confidence.HIGH and hedged:
        final_confidence = Confidence.MEDIUM

    if final_confidence in (Confidence.HIGH, Confidence.MEDIUM) and hedged and not legal_basis:
        final_confidence = Confidence.LOW

    return RAGResponse(
        answer=answer,
        legal_basis=legal_basis,
        lease_reference=lease_ref if lease_ref else None,
        explanation=explanation,
        confidence=final_confidence,
        conflict_flag=conflict_flag,
        error_type=None
    )


def _clean_lease_text(text: str) -> str:
    cleaned = re.sub(r"\[Page\s+\d+\]", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" \"'")


LEGAL_CITATION_RE = re.compile(
    r"(?:Citation\s*:\s*)?"
    r"(?P<act>[A-Z][A-Za-z0-9 &'().,\-/]+?Act,\s*\d{4})"
    r",?\s*Section\s+(?P<section>\d+[A-Z]?)"
    r"(?P<title>\s*[-–—:]\s*[^;\n]+)?",
    re.IGNORECASE,
)
SECTION_ONLY_RE = re.compile(r"\bSection\s+(\d+[A-Z]?)\b", re.IGNORECASE)


def _clean_legal_basis(raw: str) -> str:
    cleaned = re.sub(r"\bCitation\s*:\s*", "", raw or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(" \"'")
    return cleaned.strip(" ,;")


def _clean_section_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    title = re.sub(r"^[\-–—:.\s]+", "", title)
    title = re.split(r"\s+\(\d+\)\b|\.\s+", title, maxsplit=1)[0]
    return title.strip(" -–—:.")


def _section_number_from_text(text: str) -> Optional[str]:
    match = re.search(r"\bSection\s+(\d+[A-Z]?)\b", text or "", re.IGNORECASE)
    return match.group(1).upper() if match else None


def _act_and_section_from_chunk(chunk) -> tuple[str, str] | None:
    section = getattr(chunk, "section", "") or ""
    match = LEGAL_CITATION_RE.search(section)
    if not match:
        return None
    return _clean_legal_basis(match.group("act")), match.group("section").upper()


def _law_chunk_matches_section(chunk, act: str, section_num: str) -> bool:
    section = getattr(chunk, "section", "") or ""
    text_start = _clean_lease_text(getattr(chunk, "text", ""))[:220]
    section_match = _section_number_from_text(section)
    if section_match and section_match == section_num.upper():
        return True

    section_heading = re.search(
        rf"(?:^|\n|\s)(?:Section\s+)?{re.escape(section_num)}\s*[.:\-–—]",
        getattr(chunk, "text", "") or "",
        re.IGNORECASE,
    )
    return bool(section_heading and (not act or act.lower() in f"{section} {text_start}".lower()))


def _matching_law_chunks(act: str, section_num: str, law_chunks: list) -> list:
    return [
        chunk for chunk in law_chunks
        if _law_chunk_matches_section(chunk, act, section_num)
    ]


def _section_title_from_metadata(section: str, section_num: str) -> Optional[str]:
    match = re.search(
        rf"\bSection\s+{re.escape(section_num)}\b\s*[-–—:]\s*(.+)$",
        section or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    return _clean_section_title(match.group(1)) or None


def _section_title_from_law_text(text: str, section_num: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        rf"(?:^|\n)\s*(?:Section\s+)?{re.escape(section_num)}\s*[.:\-–—]\s*([A-Z][^\n.()]{{4,180}})",
        rf"\b(?:Section\s+)?{re.escape(section_num)}\s*[.:\-–—]\s*([A-Z][^.\n()]{{4,180}})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        title = _clean_section_title(match.group(1))
        if title:
            return title
    return None


def _find_section_title(act: str, section_num: str, law_chunks: list) -> Optional[str]:
    matching_chunks = _matching_law_chunks(act, section_num, law_chunks)
    fallback_chunks = [chunk for chunk in law_chunks if chunk not in matching_chunks]

    for chunk in [*matching_chunks, *fallback_chunks]:
        section = getattr(chunk, "section", "") or ""
        title = _section_title_from_metadata(section, section_num)
        if title:
            return title
        title = _section_title_from_law_text(getattr(chunk, "text", "") or "", section_num)
        if title:
            return title
    return None


def _says_law_does_not_support(answer: str, explanation: str) -> bool:
    combined = f"{answer or ''} {explanation or ''}".lower()
    markers = [
        "legal context does not provide",
        "law does not provide",
        "no legal provision",
        "does not provide specific provisions",
        "does not support",
        "not supported by",
        "primary protection comes from the lease",
        "primary basis comes from the lease",
    ]
    return any(marker in combined for marker in markers)


def _law_citation_matches_question(
    chunks: list,
    question: str,
    answer: str,
    explanation: str,
) -> bool:
    if not question:
        return True
    if not chunks:
        return False
    if _says_law_does_not_support(answer, explanation):
        return False

    topic = _issue_topic(question)
    terms = _question_terms(question)
    haystack = " ".join(
        f"{getattr(chunk, 'section', '')} {getattr(chunk, 'text', '')}"
        for chunk in chunks
    ).lower()

    if topic == "security deposit":
        if "security deposit" in haystack:
            return True
        deposit_terms = {"security", "deposit"}
        protection_terms = {"refund", "refunded", "deduct", "deduction", "interest"}
        return bool(deposit_terms & terms) and bool(protection_terms & terms) and (
            bool(deposit_terms & set(re.findall(r"[a-z0-9]+", haystack)))
            and bool(protection_terms & set(re.findall(r"[a-z0-9]+", haystack)))
        )

    if topic == "repairs or maintenance":
        q = (question or "").lower()
        if "maintenance" in q:
            maintenance_markers = [
                "maintenance",
                "service charge",
                "service charges",
                "society charge",
                "society charges",
            ]
            return any(marker in haystack for marker in maintenance_markers)
        if "amenit" in q:
            return "amenity" in haystack or "amenities" in haystack
        if "repair" in q:
            return "repair" in haystack or "repairs" in haystack

    overlap = {term for term in terms if term in haystack}
    required = 1 if len(terms) <= 2 else 2
    return len(overlap) >= required


def format_legal_basis(
    raw_legal_basis: str,
    law_chunks: list,
    question: str = "",
    answer: str = "",
    explanation: str = "",
) -> str:
    """
    Enrich legal citations with section titles from retrieved law text.
    When law chunks are available, unverified titles are dropped instead of invented.
    """
    raw = _clean_legal_basis(raw_legal_basis)
    if not raw:
        return raw

    formatted: list[str] = []
    seen: set[str] = set()
    matched_citation = False

    def add_citation(act: str, section_num: str, title: Optional[str]) -> None:
        citation = f"{act}, Section {section_num}"
        if title:
            citation = f"{citation} - {title}"
        key = citation.lower()
        if key not in seen:
            formatted.append(citation)
            seen.add(key)

    for match in LEGAL_CITATION_RE.finditer(raw):
        matched_citation = True
        act = _clean_legal_basis(match.group("act"))
        section_num = match.group("section").upper()
        matching_chunks = _matching_law_chunks(act, section_num, law_chunks)
        if law_chunks and not _law_citation_matches_question(
            matching_chunks,
            question,
            answer,
            explanation,
        ):
            continue
        title = _find_section_title(act, section_num, law_chunks)
        if not title and match.group("title") and not law_chunks:
            title = _clean_section_title(match.group("title"))

        add_citation(act, section_num, title)

    mentioned_sections = {
        match.group(1).upper()
        for match in SECTION_ONLY_RE.finditer(f"{answer or ''} {explanation or ''}")
    }
    cited_sections = {
        match.group("section").upper()
        for match in LEGAL_CITATION_RE.finditer(raw)
    }
    for section_num in sorted(mentioned_sections - cited_sections):
        matching_chunks = [
            chunk for chunk in law_chunks
            if (_act_and_section_from_chunk(chunk) or ("", ""))[1] == section_num
        ]
        if not _law_citation_matches_question(matching_chunks, question, answer, explanation):
            continue
        act = ""
        if matching_chunks:
            parsed = _act_and_section_from_chunk(matching_chunks[0])
            act = parsed[0] if parsed else ""
        if not act:
            continue
        add_citation(act, section_num, _find_section_title(act, section_num, law_chunks))

    if formatted:
        return "; ".join(formatted)

    if matched_citation:
        return ""

    return raw


def _issue_topic(question: str) -> str:
    q = (question or "").lower()
    topic_map = [
        (("security deposit", "deposit"), "security deposit"),
        (
            (
                "advance rent",
                "advance payment",
                "advance amount",
                "months advance",
                "month advance",
                "rent additionally",
                "rent additional",
                "additional advance",
            ),
            "additional advance rent",
        ),
        (
            (
                "rent increase",
                "increase rent",
                "rent be increased",
                "rent increased",
                "pay extra",
                "extra rent",
                "extra amount",
                "extra payment",
                "extra this month",
                "additional rent",
                "additional amount",
                "rent hike",
                "escalation",
                "revision",
            ),
            "rent increase",
        ),
        (("notice", "vacate", "termination"), "notice or termination"),
        (("receipt", "cash", "payment proof", "paid rent"), "rent payment receipts"),
        (("repair", "maintenance"), "repairs or maintenance"),
        (("lock", "evict", "possession", "leave"), "possession or eviction"),
    ]
    for keywords, topic in topic_map:
        if any(keyword in q for keyword in keywords):
            return topic
    compact = re.sub(r"[^a-z0-9\s]", " ", q)
    words = [
        word for word in compact.split()
        if len(word) > 3 and word not in {
            "what", "when", "where", "which", "landlord", "tenant",
            "lease", "does", "have", "with", "from", "this", "that",
            "your", "mine", "protected", "allowed", "required",
        }
    ]
    return " ".join(words[:3]) or "this issue"


def _is_extra_charge_question(question: str) -> bool:
    q = (question or "").lower()
    charge_markers = [
        "pay extra",
        "extra rent",
        "extra amount",
        "extra payment",
        "extra this month",
        "additional rent",
        "additional amount",
        "advance rent",
        "advance payment",
        "additional advance",
        "more rent",
        "charge more",
        "charging more",
    ]
    return any(marker in q for marker in charge_markers)


def _starts_with_landlord_entitlement(answer: str) -> bool:
    lower = (answer or "").strip().lower()
    prefixes = (
        "the landlord is entitled",
        "a landlord is entitled",
        "landlord is entitled",
        "the landlord can",
        "a landlord can",
        "landlord can",
        "the landlord may",
        "a landlord may",
        "landlord may",
    )
    return lower.startswith(prefixes)


def _annual_increase_phrase(text: str) -> Optional[str]:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:per\s*cent|percent|%)\s+per\s+annum",
        text or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1)} per cent per annum"


def format_answer(answer: str, question: str, explanation: str = "") -> str:
    """
    Keep rent/extra-charge answers tenant-first instead of landlord-entitlement-first.
    Other topics and already tenant-framed answers pass through unchanged.
    """
    if not answer or _issue_topic(question) != "rent increase":
        return answer
    if not _starts_with_landlord_entitlement(answer):
        return answer

    combined = f"{answer} {explanation}"
    annual = _annual_increase_phrase(combined)
    court_or_dispute = bool(re.search(r"\b(court|dispute|determin)", combined, re.IGNORECASE))

    if annual:
        rule = f"The cited law limits ordinary rent increases to {annual}"
    else:
        rule = "Any extra rent demand must fit the cited law and lease"

    if court_or_dispute:
        rule = f"{rule}, and a disputed or higher permitted increase must follow the cited court/dispute mechanism"

    if _is_extra_charge_question(question):
        return f"You do not have to pay an unsupported extra rent demand. {rule}."

    return f"{rule}."


def format_explanation(explanation: str, legal_basis: str) -> str:
    if not explanation:
        return explanation
    if "Section " in (legal_basis or ""):
        return explanation

    sentences = re.split(r"(?<=[.!?])\s+", explanation)
    filtered = [
        sentence for sentence in sentences
        if not re.search(r"\b(?:Section\s+\d+[A-Z]?|Rent Control Act)\b", sentence, re.IGNORECASE)
    ]
    return " ".join(sentence.strip() for sentence in filtered if sentence.strip()) or explanation


def align_answer_with_legal_basis(answer: str, legal_basis: str) -> str:
    if not answer or "Section " in (legal_basis or ""):
        return answer
    cleaned = re.sub(
        r"\bor\s+the\s+[A-Z][A-Za-z &'().,\-/]+Rent Control Act,\s*\d{4}",
        "or the retrieved legal context",
        answer,
    )
    cleaned = re.sub(
        r"\bunder\s+the\s+[A-Z][A-Za-z &'().,\-/]+Rent Control Act,\s*\d{4}",
        "under the retrieved legal context",
        cleaned,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _question_terms(question: str) -> set[str]:
    topic = _issue_topic(question)
    text = f"{question or ''} {topic}".lower()
    terms = {
        word for word in re.findall(r"[a-z0-9]+", text)
        if len(word) > 3 and word not in {
            "what", "when", "where", "which", "landlord", "tenant",
            "lease", "does", "have", "with", "from", "this", "that",
            "your", "mine", "protected", "allowed", "required",
        }
    }
    if topic == "repairs or maintenance" and "repair" not in (question or "").lower():
        terms.discard("repair")
        terms.discard("repairs")
    if "deposit" in terms:
        terms.update({
            "security",
            "refund",
            "refunded",
            "deduct",
            "deduction",
            "damage",
            "damages",
            "dues",
            "interest",
            "unpaid",
        })
    if topic == "rent increase" or "increase" in terms:
        terms.update({"rent", "increase", "revision", "escalation", "enhance", "enhancement"})
    if topic == "additional advance rent":
        terms.update({
            "advance",
            "rent",
            "monthly",
            "month",
            "months",
            "payment",
            "paid",
            "additional",
        })
    if topic == "repairs or maintenance":
        q = (question or "").lower()
        terms.update({
            "maintenance",
            "charges",
            "charge",
            "society",
            "monthly",
            "service",
            "services",
        })
        if "repair" in q:
            terms.update({"repair", "repairs"})
    if "notice" in terms:
        terms.update({"notice", "terminate", "termination", "vacate"})
    return terms


def _lease_heading(chunk, fallback_index: int) -> str:
    section = (getattr(chunk, "section", "") or "").strip()
    if section:
        return section
    return f"Lease Excerpt {fallback_index}"


def _title_case_clause_label(label: str) -> str:
    words = re.findall(r"[A-Za-z0-9&]+", label or "")
    return " ".join(word if word == "&" else word.capitalize() for word in words)


def _lease_clause_label_from_text(text: str) -> str:
    cleaned = _clean_lease_text(text)
    match = re.search(
        r"(?:(?:Clause|Article)\s+\d+[A-Z]?\s*[-:.\s]*)?([A-Z][A-Z\s/&-]{2,50}):",
        cleaned,
    )
    if not match:
        return ""
    return _title_case_clause_label(match.group(1))


def _is_generic_lease_heading(heading: str) -> bool:
    return bool(re.fullmatch(r"lease document(?:,\s*page\s*\d+)?", (heading or "").strip(), re.IGNORECASE))


def format_lease_legal_basis(lease_chunks: list, question: str) -> str:
    if not lease_chunks:
        return ""

    selected_chunk, _ = _select_lease_chunk("", lease_chunks, question)
    if not selected_chunk:
        return "Lease Document"

    heading = _lease_heading(selected_chunk, 1)
    if not _is_generic_lease_heading(heading):
        return heading

    label = _lease_clause_label_from_text(getattr(selected_chunk, "text", "") or "")
    if label:
        return f"Lease Document - {label} clause"
    return "Lease Document"


def _clean_lease_excerpt_for_display(excerpt: str) -> str:
    def ordinal(value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def ordinal_of(match: re.Match) -> str:
        value = int(match.group(1))
        return f"{ordinal(value)} of"

    text = _clean_lease_text(excerpt)
    label_match = re.search(r"\b[A-Z][A-Z\s/&-]{2,50}:\s*", text)
    if label_match and label_match.start() > 0:
        text = text[label_match.start():]
    text = re.sub(
        r"^(?:Clause|Article)\s+\d+[A-Z]?\s*[-:.\s]*(?:[A-Z][A-Z\s/&-]{2,50})?[.:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^[A-Z][A-Z\s/&-]{2,50}:\s*", "", text)
    text = re.sub(r"(?<!\w)[%€₹]\s*(\d[\d,]*(?:/-)?)", r"Rs. \1", text)
    text = text.replace("‘", "").replace("’", "")
    text = re.sub(r"\b(\d{1,2})[\"%*]\s+of\b", ordinal_of, text)
    text = re.sub(r"\b(\d{1,2})[\"%*]\s+day\b", lambda match: f"{ordinal(int(match.group(1)))} day", text)
    text = re.sub(r"\s+Rent for the first month i\.?e\.?$", "", text, flags=re.IGNORECASE)
    text = text.strip(" \"'.,;:")
    return re.sub(r"\s+", " ", text)


def _score_lease_chunk(chunk, terms: set[str]) -> int:
    haystack = f"{getattr(chunk, 'section', '')} {getattr(chunk, 'text', '')}".lower()
    score = sum(haystack.count(term) for term in terms)
    if "maintenance" in terms and "maintenance:" in haystack:
        score += 8
    if "security" in terms and "security deposit:" in haystack:
        score += 8
    if "rent" in terms and "rent:" in haystack:
        score += 4
    return score


def _select_lease_chunk(raw_reference: str, lease_chunks: list, question: str):
    if not lease_chunks:
        return None, 0

    raw = raw_reference or ""
    clause_match = re.search(r"\b(?:lease\s+)?(?:clause|article)\s+(\d+[a-z]?)\b", raw, re.IGNORECASE)
    if clause_match:
        clause_num = clause_match.group(1)
        clause_pattern = re.compile(rf"\b(?:clause|article)\s+{re.escape(clause_num)}\b", re.IGNORECASE)
        for i, chunk in enumerate(lease_chunks, start=1):
            heading = _lease_heading(chunk, i)
            text_start = _clean_lease_text(getattr(chunk, "text", ""))[:120]
            if clause_pattern.search(f"{heading} {text_start}"):
                return chunk, i
        if clause_num.isdigit():
            synthetic_index = int(clause_num)
            if 1 <= synthetic_index <= len(lease_chunks):
                return lease_chunks[synthetic_index - 1], synthetic_index

    terms = _question_terms(question)
    scored = [
        (_score_lease_chunk(chunk, terms), i, chunk)
        for i, chunk in enumerate(lease_chunks, start=1)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    _, index, chunk = scored[0]
    return chunk, index


def _lease_excerpt(text: str, question: str, max_chars: int = 320) -> str:
    cleaned = _clean_lease_text(text)
    if not cleaned:
        return ""

    terms = _question_terms(question)
    candidates = [
        candidate.strip()
        for candidate in re.split(r"(?<!Rs\.)(?<=[.!?])\s+|\s{2,}", cleaned)
        if len(candidate.strip()) >= 20
    ] or [cleaned]

    def score(candidate: str) -> int:
        lower = candidate.lower()
        return sum(lower.count(term) for term in terms)

    scored = [(score(candidate), i, candidate) for i, candidate in enumerate(candidates)]
    best_score, best_index, best = max(scored, key=lambda item: item[0])
    if best_score == 0:
        best = cleaned
        best_index = -1

    if best_index >= 0:
        parts = [best]
        total_len = len(best)
        for direction in (1, -1):
            index = best_index + direction
            while 0 <= index < len(candidates):
                candidate = candidates[index]
                candidate_score = score(candidate)
                added_len = len(candidate) + 1
                if candidate_score <= 0 or total_len + added_len > max_chars:
                    break
                if direction > 0:
                    parts.append(candidate)
                else:
                    parts.insert(0, candidate)
                total_len += added_len
                index += direction
        best = " ".join(parts)

    if len(best) <= max_chars:
        return best
    truncated = best[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{truncated}..."


def _lease_reference_says_silent(raw_reference: str) -> bool:
    raw = (raw_reference or "").lower()
    silence_markers = [
        "no clause",
        "no relevant",
        "no direct",
        "silent",
        "contains no provision",
        "no provision",
        "does not address",
        "doesn't address",
        "lacks",
        "not mention",
    ]
    return any(marker in raw for marker in silence_markers)


def _augment_lease_excerpt_from_related_chunks(
    excerpt: str,
    selected_chunk,
    lease_chunks: list,
    question: str,
    max_chars: int = 380,
) -> str:
    terms = _question_terms(question)
    present_terms = {term for term in terms if term in (excerpt or "").lower()}
    missing_terms = terms - present_terms
    if not missing_terms or len(excerpt) >= max_chars:
        return excerpt

    related = []
    for chunk in lease_chunks:
        if chunk is selected_chunk:
            continue
        candidate = _clean_lease_excerpt_for_display(
            _lease_excerpt(getattr(chunk, "text", ""), question, max_chars=160)
        )
        if not candidate:
            continue
        lower = candidate.lower()
        if candidate in excerpt:
            continue
        score = sum(lower.count(term) for term in missing_terms)
        if score > 0:
            related.append((score, candidate))

    related.sort(key=lambda item: item[0], reverse=True)
    combined = excerpt
    for _, candidate in related:
        separator = " " if combined.endswith((".", "!", "?")) else ". "
        if len(combined) + len(separator) + len(candidate) > max_chars:
            continue
        if candidate[:1].islower():
            candidate = f"{candidate[:1].upper()}{candidate[1:]}"
        combined = f"{combined}{separator}{candidate}"
        break
    return combined


def format_lease_reference(
    raw_lease_reference: str,
    lease_chunks: list,
    question: str,
    supporting_notes: str = "",
) -> str:
    """
    Render lease references with actual lease text, never clause numbers alone.
    """
    if not lease_chunks:
        return raw_lease_reference

    selected_chunk, index = _select_lease_chunk(raw_lease_reference, lease_chunks, question)
    if not selected_chunk:
        return raw_lease_reference

    heading = _lease_heading(selected_chunk, index)
    excerpt = _clean_lease_excerpt_for_display(
        _lease_excerpt(getattr(selected_chunk, "text", ""), question)
    )
    excerpt = _augment_lease_excerpt_from_related_chunks(
        excerpt,
        selected_chunk,
        lease_chunks,
        question,
    )
    if not excerpt:
        return raw_lease_reference

    if (
        _lease_reference_says_silent(raw_lease_reference)
        or _lease_reference_says_silent(supporting_notes)
    ):
        topic = _issue_topic(question)
        return f"No direct {topic} clause. Closest lease text: {excerpt}"

    if _is_generic_lease_heading(heading):
        label = _lease_clause_label_from_text(getattr(selected_chunk, "text", "") or "")
        if label:
            return f"{label}: {excerpt}"
        return f"Lease text: {excerpt}"
    return f"{heading}: {excerpt}"


def fallback_response(reason: str = "") -> RAGResponse:
    """
    Returns a safe, honest response when confidence is LOW
    or any pipeline stage fails.
    """
    return RAGResponse(
        answer="No legal provision was retrieved that answers this question.",
        legal_basis="",
        lease_reference=None,
        explanation="Missing: a matching statute section or lease clause for the specific issue. Ask again with the issue and state named clearly, such as repairs, rent increase, lockout, possession, notice, or receipts.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=f"FALLBACK: {reason}" if reason else "FALLBACK"
    )


def retrieval_unavailable_response(reason: str = "") -> RAGResponse:
    """
    Returns a clear system-status response when Pinecone/embedding retrieval is
    unavailable. This must not look like a legal conclusion.
    """
    return RAGResponse(
        answer="The legal database is temporarily unavailable, so I can't retrieve the law or lease context right now.",
        legal_basis="",
        lease_reference="Retrieval unavailable — lease and law context could not be checked",
        explanation="This is a system retrieval issue, not a legal answer. Please try again after the vector database is available.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=f"RETRIEVAL_UNAVAILABLE: {reason}" if reason else "RETRIEVAL_UNAVAILABLE"
    )


def llm_unavailable_response(reason: str = "") -> RAGResponse:
    """
    Returns a clear system-status response when the LLM provider cannot answer.
    This must not look like a legal conclusion.
    """
    return RAGResponse(
        answer="AI credits are not available right now. Please try again after some time.",
        legal_basis="",
        lease_reference=None,
        explanation="The language model could not generate an answer because the provider is temporarily unavailable, rate limited, or out of credits.",
        confidence=Confidence.LOW,
        conflict_flag=False,
        error_type=reason or "LLM_UNAVAILABLE"
    )


def is_retrieval_unavailable(reason: str | None) -> bool:
    if not reason:
        return False

    reason_upper = reason.upper()
    unavailable_markers = [
        "EMBEDDING_FAILED",
        "LAW_INDEX_EMPTY_OR_UNAVAILABLE",
        "LAW_RETRIEVAL_FAILED",
        "VECTOR_DB_ERROR",
        "PINECONE",
        "CONNECTION",
        "TIMEOUT",
        "UNAVAILABLE",
        "DNS",
        "SERVICE",
    ]
    return any(marker in reason_upper for marker in unavailable_markers)


def run_rag_chain(
    question: str,
    state: str,
    session_id: str,
    history: Optional[list] = None,
) -> RAGResponse:
    """
    Master function — runs the complete RAG pipeline.

    retrieve → confidence check → prompt → LLM → format

    This is the only function the API route needs to call.
    """

    if history is None:
        history = session_store.get_history(session_id)

    # Rewritten queries are embedded for retrieval; `question` alone is for the LLM.
    retrieval_queries = build_retrieval_queries(question, history)

    # Stage 1 — Retrieve (embedding uses rewritten / history-augmented queries)
    retrieval = run_retrieval(
        question=retrieval_queries[0],
        state=state,
        session_id=session_id,
        retrieval_queries=retrieval_queries,
    )

    if retrieval["error_type"]:
        if is_retrieval_unavailable(retrieval["error_type"]):
            return retrieval_unavailable_response(retrieval["error_type"])
        return fallback_response(retrieval["error_type"])

    context = retrieval["context"]

    # Stage 2 — Confidence gate
    # If retrieval found no usable context, stop. If it found some context, let
    # the prompt decide whether the documents answer the question; this avoids
    # unhelpful fallbacks for valid law questions with weaker vector scores.
    if retrieval["should_fallback"] and not (context.law_chunks or context.lease_chunks):
        return fallback_response("low retrieval confidence")

    # Stage 3 — Build prompt (jurisdiction + conversation history in user message)
    messages = build_prompt(
        question=question,
        law_chunks=context.law_chunks,
        lease_chunks=context.lease_chunks,
        state=state,
        history=history,
    )

    # Stage 4 — Call LLM
    llm_result = call_llm(messages)
    if llm_result["error_type"]:
        return llm_unavailable_response(llm_result["error_type"])

    # Stage 5 — Format and return structured response
    parsed = format_response(
        llm_output=llm_result["llm_output"],
        confidence=context.confidence,
        law_chunks=context.law_chunks,
        lease_chunks=context.lease_chunks,
        question=question,
    )

    session_store.append_turn(
        session_id=session_id,
        question=question,
        answer=parsed.answer,
        metadata={
            "confidence": parsed.confidence.value,
            "conflict_flag": parsed.conflict_flag,
            "state": state,
        },
    )

    return parsed
