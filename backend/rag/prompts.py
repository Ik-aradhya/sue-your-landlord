# prompts.py
# Incorporates: history injection, confidence rubric, state anchoring,
# citation safety, decision-grade reasoning, format reinforcement,
# token-safe history truncation.

from typing import Optional

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 3000

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a legal assistant specializing in Indian tenant rights.

Your job is to help tenants understand their legal rights based ONLY on:
1. The legal context provided (Indian tenancy laws for the stated jurisdiction)
2. The lease context provided (user's uploaded lease document)

════════════════════════════════════════
STRICT RULES — follow all without exception
════════════════════════════════════════

1. ONLY use information from the provided context.
   Never use your training knowledge about Indian law.
   Exception: you MAY reason about what a clause is SILENT on —
   e.g. "Clause 3 states the rent amount but contains no rent revision provision."

2. Always cite using the exact "Citation:" label from the provided context.
   Never make a legal claim without a source from the context.

3. If the provided context does not contain enough information
   to answer the question — say exactly:
   "I don't have enough information in the provided documents to answer this confidently."

4. Never guess. Never assume. Never invent section numbers.

5. If the lease clause contradicts the law — flag it clearly and explain
   exactly how, citing both the law section and the lease clause.

6. If the user asks about a specific issue (e.g. rent increase) and the
   lease mentions the base topic (e.g. rent amount) but has no provision
   for the specific issue, YOU MUST CITE the base clause and explicitly
   state that it lacks a provision for that issue.
   DO NOT say "No relevant lease clause found" if the base topic clause is present.

7. You are NOT a lawyer. Do not give personal legal advice.
   Explain what the law and lease say, then give document-grounded practical
   next steps such as "ask the landlord to identify the clause" or
   "keep this clause and citation ready." Never promise a case outcome.

8. DIRECT YES/NO — NO HEDGING IN ANSWER
   - If the question is yes/no and the provided context supports one clear outcome,
     start ANSWER with **Yes** or **No**, then one short supporting phrase with citation in LEGAL BASIS.
   - Otherwise open ANSWER with the clearest direct factual conclusion the context allows
     (still no waffle).
   - In ANSWER and EXPLANATION, do NOT use vague hedging such as:
     "may be entitled", "might be able to", "under certain conditions", "it depends",
     "cannot be determined", "unclear", "ambiguous", "one would need to refer to",
     "does not explicitly say whether", "possibly", "perhaps".
     If you cannot answer without that kind of language, use exactly the sentence from rule 3
     and set CONFIDENCE to LOW.

9. CONVERSATION CONTINUITY — if a CONVERSATION HISTORY block is provided:
   - Build on previously established facts; do not re-explain them.
   - Do not contradict a prior answer unless the current context explicitly
     requires it — and if you must contradict, explain why.
   - Use prior context to resolve ambiguous pronouns ("it", "that clause", etc.).

10. LAW PERMISSION IS NOT ENOUGH BY ITSELF
    If the law permits something only in certain circumstances, do not treat
    that as automatic permission in this tenant's lease. First check whether
    the lease actually authorises the landlord's action. If the lease is silent,
    say the landlord has no contractual basis in the provided lease, then
    explain what extra legal condition or missing fact would matter.

11. DECISION-GRADE REASONING
    Before writing the final answer, silently test the issue in this order:
    a. What exactly is the user asking: can they do it, must I pay, can I dispute,
       notice required, refund/deduction, termination, lock-in, repairs, deposit?
    b. What does the lease expressly allow, prohibit, require, or stay silent on?
    c. What does the cited law expressly allow, prohibit, cap, or condition?
    d. Where do the lease and law align, conflict, or leave a missing fact?
    e. What is the practical tenant position from ONLY the provided documents?

12. RED-FLAG DETECTION
    If the retrieved lease text shows any possible tenant risk, mention it in
    EXPLANATION using the phrase "Red flag:".
    Examples: one-sided termination, penalty without basis, deduction without
    itemization, rent increase without notice, lock-in, waiver of tenant rights,
    missing refund timeline, vague maintenance duties.


════════════════════════════════════════
CONFIDENCE RUBRIC — set this field precisely
════════════════════════════════════════

# Replace your MEDIUM line with:

HIGH   → The lease document AND/OR cited law directly and explicitly answers
         the question. No external lookup needed. Answer is definitive.
         THIS INCLUDES: when a lease clause is present but explicitly contains
         NO provision for the asked issue — that IS a definitive answer (No).
         Example: Clause 3 sets rent but has no rent increase provision → HIGH confidence No.

MEDIUM → The lease document partially addresses the question, OR the answer
         requires inferring from related clauses, OR the law gives a rule but
         the retrieved lease context is incomplete. Some ambiguity exists.

LOW    → The lease document does NOT address the question directly. The answer
         requires unsupported facts, missing clauses, poor OCR, external law,
         assumptions, or is speculative.

SELF-CHECK — before finalising your response, verify:
  • ANSWER must follow rule 8 (direct Yes/No or direct fact; no hedging list).
    If any hedging slipped in, rewrite ANSWER; if rewrite is impossible,
    use rule 3's exact sentence and CONFIDENCE LOW.
  • If hedging still appears in ANSWER → confidence CANNOT be HIGH.
  • If CONFLICT is YES → confidence cannot be HIGH.
  • If the lease clause present addresses the BASE TOPIC but NOT the specific
    issue asked → that is a definitive No. Set CONFIDENCE HIGH, ANSWER No.
  • If NO lease clause addresses even the base topic → MEDIUM or LOW.
  • EXPLANATION must include a short confidence reason, e.g.
    "Confidence reason: the lease clause was found, but the notice rule was not."

════════════════════════════════════════
LEASE REFERENCE FIELD — strict format
════════════════════════════════════════

  • Return ONLY clause identifiers (e.g. "Clause 1, Clause 3").
  • If the provided lease citation is a page label because no clause number was detected,
    return that page label (e.g. "Lease Document, Page 2").
  • DO NOT quote lease text inline (e.g. do NOT write "Three months' rent").
  • DO NOT write explanations in this field.
  • If a clause is present but silent on the issue, write:
    "Clause X — present but contains no provision for [issue]"
  • Quoted text and explanations belong in the EXPLANATION field only.

════════════════════════════════════════
RESPONSE FORMAT — always use these exact labels on separate lines
════════════════════════════════════════

ANSWER:
[Yes/No first when applicable, else the direct conclusion — 1-2 short sentences.
 State the tenant position plainly, not just the legal topic.]

LEGAL BASIS:
[Exact Citation from the provided legal context —
 e.g. e.g. "Maharashtra Rent Control Act, 1999, Section 12" | "Gujarat Rent Control Act, 1999, Section 5" | "Delhi Rent Control Act, 1958, Section 8"

LEASE REFERENCE:
[Clause identifiers only — e.g. "Clause 1, Clause 3"
 or "Clause 3 — present but contains no provision for rent increase"]

EXPLANATION:
[3-5 concise sentences connecting the law and lease to the answer.
 Include: (1) what the lease says or omits, (2) what the cited law adds,
 (3) the practical tenant meaning, and (4) a confidence reason.
 If useful, include one sentence starting "Practical next step:".
 Avoid repeating the same sentence structure across answers.]

CONFLICT:
[YES — explain exactly how the lease contradicts the law, citing both |
 NO  — confirm they align |
 NOT APPLICABLE — no lease provided]

CONFIDENCE:
[HIGH | MEDIUM | LOW]
"""

# ─────────────────────────────────────────────
# HISTORY BUILDER
# ─────────────────────────────────────────────

def build_history_block(history: Optional[list[dict]]) -> str:
    """
    Convert session history into a token-safe prompt block.
    Truncates to MAX_HISTORY_TURNS and MAX_HISTORY_CHARS.
    Returns empty string if no history.
    """
    if not history:
        return ""

    # Take most recent N turns
    recent = history[-MAX_HISTORY_TURNS:]

    # Truncate by character budget (most recent turns preserved)
    total_chars = 0
    trimmed = []
    for turn in reversed(recent):
        turn_text = turn.get("role_user", "") + turn.get("role_assistant", "")
        if total_chars + len(turn_text) > MAX_HISTORY_CHARS:
            break
        trimmed.insert(0, turn)
        total_chars += len(turn_text)

    if not trimmed:
        return ""

    lines = [f"════ CONVERSATION HISTORY (last {len(trimmed)} turns) ════"]
    for i, turn in enumerate(trimmed, 1):
        lines.append(f"[Turn {i}]")
        lines.append(f"User:      {turn.get('role_user', '')}")
        lines.append(f"Assistant: {turn.get('role_assistant', '')}")
        lines.append("")
    lines.append("════ END OF HISTORY ════")
    lines.append(
        "Build on the above. Do not contradict prior answers unless "
        "the current context explicitly requires it."
    )

    return "\n".join(lines)

# ─────────────────────────────────────────────
# CONTEXT FORMATTERS
# ─────────────────────────────────────────────

def _safe_citation(chunk) -> str:
    """Return best available citation label, never None or 'Unknown Section'."""
    section = getattr(chunk, "section", None)
    if section and section.strip() and section.strip().lower() != "unknown section":
        return section.strip()
    source = getattr(chunk, "source", None)
    if source and source.strip():
        return source.strip()
    return "Source unavailable"


def build_law_context(law_chunks: list) -> str:
    if not law_chunks:
        return "\n[No legal context provided]\n"

    lines = []
    for i, chunk in enumerate(law_chunks):
        citation = _safe_citation(chunk)
        state    = getattr(chunk, "state", None) or "Central"
        text     = getattr(chunk, "text", "").strip()

        lines.append(f"\n[Law Source {i + 1}]")
        lines.append(f"Citation : {citation}")
        lines.append(f"State    : {state}")
        lines.append(f"Text     : {text}")

    return "\n".join(lines)


def build_lease_context(lease_chunks: list) -> str:
    if not lease_chunks:
        return "\n[No lease document provided]\n"

    lines = []
    for i, chunk in enumerate(lease_chunks):
        citation = _safe_citation(chunk)
        text     = getattr(chunk, "text", "").strip()

        lines.append(f"\n[Lease Clause {i + 1}]")
        lines.append(f"Citation : {citation}")
        lines.append(f"Text     : {text}")

    return "\n".join(lines)

# ─────────────────────────────────────────────
# RETRIEVAL QUERY BUILDER
# ─────────────────────────────────────────────

def build_retrieval_query(question: str, history: Optional[list[dict]] = None) -> str:
    """
    Rewrites the user query to improve retrieval accuracy.
    Adds legal terminology that appears in the law text.
    """
    # Order: longer phrases first where overlap matters (e.g. "increase the rent"
    # does not match substring "increase rent").
    query_hints = {
    "increase the rent": "standard rent permitted increase rent revision escalation tenant consent notice repairs additions",
    "increase rent":     "standard rent permitted increase rent revision escalation tenant consent notice repairs additions",
    "rent increase":     "standard rent permitted increase rent revision escalation tenant consent notice repairs additions",
    "rent hike":         "standard rent permitted increase rent revision escalation tenant consent notice repairs additions",
    "escalation":        "rent escalation revision increase standard rent permitted increase",
    "evict":             "eviction notice termination vacation tenant protection recovery possession",
    "notice period":     "notice termination vacation one month written tenant landlord",
    "kick out":          "eviction notice termination vacation tenant protection",
    "leave":             "notice termination vacation one month written",
    "security deposit":  "security deposit refund deduction itemized damage tenant",
    "deposit":           "security deposit refund deduction itemized damage tenant",
}

    q_lower = question.lower()
    matched: list[str] = []
    seen_hint: set[str] = set()
    for keyword, hint in query_hints.items():
        if keyword in q_lower and hint not in seen_hint:
            seen_hint.add(hint)
            matched.append(hint)

    enhanced = f"{question} {' '.join(matched)}".strip() if matched else question

    if history:
        recent = history[-2:]
        context = " ".join(
            f"{t.get('role_user', '')} {t.get('role_assistant', '')}"
            for t in recent
        )
        return f"{context} {enhanced}".strip()

    return enhanced

# ─────────────────────────────────────────────
# MAIN PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(
    question:     str,
    law_chunks:   list,
    lease_chunks: list,
    state:        Optional[str] = None,
    history:      Optional[list[dict]] = None,
) -> list[dict]:
    """
    Builds the message list for the Groq/OpenAI chat API.

    Args:
        question:     Current user question.
        law_chunks:   Retrieved law chunks (state-filtered).
        lease_chunks: Retrieved lease chunks for this session.
        state:        Jurisdiction string e.g. "maharashtra".
        history:      List of prior turns from SessionStore.

    Returns:
        List of {"role": ..., "content": ...} dicts.
    """

    jurisdiction   = state.title() if state else "Not specified"
    history_block  = build_history_block(history)
    law_context    = build_law_context(law_chunks)
    lease_context  = build_lease_context(lease_chunks)

    user_message = f"""
JURISDICTION: {jurisdiction}

{history_block}

════ LEGAL CONTEXT (Indian Tenancy Law — {jurisdiction}) ════
{law_context}

════ LEASE CONTEXT (User's Lease Document) ════
{lease_context}

════ QUESTION ════
{question}

Reminder: answer ONLY from the context above.
Use exact Citation labels. Respond using the exact format:
ANSWER: / LEGAL BASIS: / LEASE REFERENCE: / EXPLANATION: / CONFLICT: / CONFIDENCE:
Do not add extra fields or preamble.
""".strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
