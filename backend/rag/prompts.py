# prompts.py
# Incorporates: history injection, confidence rubric, state anchoring,
# citation safety, format reinforcement, token-safe history truncation.

from typing import Optional

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 3000

RENT_INCREASE_KEYWORDS = (
    "rent increase",
    "increase rent",
    "increase the rent",
    "increased",
    "rent hike",
    "hike rent",
    "escalation",
    "rent revision",
    "permitted increase",
)

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

3. If the provided law context AND lease context together do not contain
   enough information to answer the question — say exactly:
   "I don't have enough information in the provided documents to answer this confidently."
   If the lease is silent but the legal context answers the question, answer
   from the cited legal context and clearly say the lease is silent.

4. Never guess. Never assume. Never invent section numbers.

5. If the lease clause contradicts the law — flag it clearly and explain
   exactly how, citing both the law section and the lease clause.

6. If the user asks about a specific issue (e.g. rent increase) and the
   lease mentions the base topic (e.g. rent amount) but has no provision
   for the specific issue, YOU MUST CITE the base clause and explicitly
   state that it lacks a provision for that issue.
   DO NOT say "No relevant lease clause found" if the base topic clause is present.

7. You are NOT a lawyer. Do not give personal legal advice.
   Only explain what the law and lease say.

8. DIRECT ANSWER — NO UNSOURCED HEDGING
   - If the question is yes/no and the provided context supports one clear outcome,
     start ANSWER with **Yes** or **No**, then one short supporting phrase with citation in LEGAL BASIS.
   - If the legally correct answer is conditional, start ANSWER with **Only if**,
     **Only to the extent**, or **No, unless** and then state the condition from
     the cited context.
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

10. LEASE SILENCE IS A LEASE FINDING, NOT A LAW FINDING
    If the lease has no clause on the issue, say so clearly in LEASE REFERENCE
    and EXPLANATION. Do NOT say the lease permits the action.
    Then apply only the cited legal context to explain whether the law allows,
    limits, or does not answer the action. Never answer "Yes" solely because
    the law has a broad permission if the retrieved legal text gives conditions.

11. RENT / CHARGE INCREASE CATEGORY SAFETY
    For questions about rent increase, annual increase, enhanced rent, permitted
    increase, standard rent, repairs, taxes, improvements, additions, amenities,
    maintenance, or service charges:
    - Do NOT merge different statutory categories into one answer.
    - Distinguish ordinary/annual rent increase from increases for additions,
      improvements, repairs, taxes, amenities, or services.
    - If the question asks for ordinary annual rent increase but the retrieved
      law only discusses additions/improvements/repairs/taxes, say the retrieved
      context does not provide an ordinary annual increase rule; do not present
      that special category as the general yearly rent increase.
    - If multiple retrieved sections apply, identify which section answers which
      category in EXPLANATION.
    This rule is state-neutral and must be applied for every jurisdiction.

12. ACTIONABILITY WITHOUT NEW UI FIELDS
    Keep the exact response labels below. In EXPLANATION, add one short final
    sentence with the next document-grounded step when supported by context,
    such as requesting written basis, checking whether the lease clause exists,
    or disputing an unsupported demand in writing. Do not draft notices unless
    the user asks.

13. CONSISTENCY ACROSS QUESTION PHRASING
    If the current question is about rent increase, rent hike, escalation,
    permitted increase, rent increase notice, or disputing a rent increase,
    treat it as the same legal cluster. Reconcile the same set of retrieved
    facts every time:
    - ordinary or annual rent increase / standard rent rule;
    - additions, improvements, repairs, taxes, amenities, or other special
      categories;
    - court or dispute mechanism, if provided;
    - lease clause or lease silence.
    If the retrieved context contains a relevant numeric rate, percentage, cap,
    or formula, include it in ANSWER. Do not answer only "yes" or only cite a
    special category when the user asked how much ordinary rent can increase.
    If the user asks notice for a rent increase and the retrieved context states
    no notice period but gives a legal mechanism, say no specific notice period
    is stated in the retrieved lease/law context and explain the mechanism.


════════════════════════════════════════
CONFIDENCE RUBRIC — set this field precisely
════════════════════════════════════════

HIGH   → The lease document AND/OR cited law directly and explicitly answers
         the question. No external lookup needed. Answer is definitive.
         This includes when the lease clearly contains no provision on the
         specific issue AND the cited law directly answers the legal rule.

MEDIUM → The lease document partially addresses the question, OR the answer
         requires connecting related clauses/statutory categories. Some
         ambiguity exists, but the answer is still grounded in retrieved text.

LOW    → The retrieved documents do NOT answer the question directly, OR only
         special/adjacent legal categories were retrieved for a general question,
         OR the answer would require outside law, assumptions, or speculation.

SELF-CHECK — before finalising your response, verify:
  • ANSWER must follow rule 8 (direct Yes/No or direct fact; no hedging list).
    If any hedging slipped in, rewrite ANSWER; if rewrite is impossible,
    use rule 3's exact sentence and CONFIDENCE LOW.
  • If hedging still appears in ANSWER → confidence CANNOT be HIGH.
  • If CONFLICT is YES → confidence cannot be HIGH.
  • If the lease clause present addresses the BASE TOPIC but NOT the specific
    issue asked, say the lease is silent on that issue. Then answer only from
    cited law if the law directly answers it.
  • If NO lease clause addresses even the base topic → MEDIUM or LOW unless
    the cited law alone directly answers the question.

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
[Yes/No first when applicable, else the direct conclusion — 1-2 short sentences, no hedging]

LEGAL BASIS:
[Exact Citation labels from the provided legal context only.
 If multiple categories apply, list the exact citation for each category.]

LEASE REFERENCE:
[Clause identifiers only — e.g. "Clause 1, Clause 3"
 or "Clause 3 — present but contains no provision for rent increase"]

EXPLANATION:
[2-4 sentences connecting the law and lease to the answer.
 Quoted lease text may appear here if essential.
 Include one short next-step sentence when supported by the retrieved context.]

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


def is_rent_increase_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in RENT_INCREASE_KEYWORDS)


def build_issue_guidance(question: str) -> str:
    if not is_rent_increase_question(question):
        return ""

    return """
════ ISSUE GUIDANCE: RENT INCREASE CLUSTER ════
Use one consistent rent-increase analysis for this question.
Check the retrieved context for:
1. ordinary/annual rent increase or standard rent rule;
2. special categories such as additions, improvements, repairs, taxes, amenities, or services;
3. court/dispute mechanism;
4. lease clause or lease silence.
If a relevant rate/cap/formula appears in the retrieved context, include it in ANSWER.
If the question asks about notice and the retrieved context has no notice period, do not fallback solely for that reason; say no specific notice period is stated in the retrieved lease/law context and then explain the retrieved legal mechanism.
════ END ISSUE GUIDANCE ════
""".strip()

# ─────────────────────────────────────────────
# RETRIEVAL QUERY BUILDER
# ─────────────────────────────────────────────

def build_retrieval_query(question: str, history: Optional[list[dict]] = None) -> str:
    """
    Rewrites the user query to improve retrieval accuracy.
    Adds legal terminology that appears in the law text.
    """
    rent_increase_canonical_hint = (
        "standard rent permitted increase annual increase ordinary rent increase "
        "rent revision escalation percentage rate cap formula per annum court "
        "fix standard rent permitted increase dispute application additions "
        "improvements repairs taxes amenities services"
    )

    # Order: longer phrases first where overlap matters (e.g. "increase the rent"
    # does not match substring "increase rent").
    query_hints = {
        "increase the rent": rent_increase_canonical_hint,
        "rent increase":     rent_increase_canonical_hint,
        "increase rent":     rent_increase_canonical_hint,
        "rent hike":         rent_increase_canonical_hint,
        "hike rent":         rent_increase_canonical_hint,
        "rent revision":     rent_increase_canonical_hint,
        "permitted increase": rent_increase_canonical_hint,
        "increased":         rent_increase_canonical_hint,
        "escalation":        "rent escalation revision increase standard rent permitted increase",
        "special addition":  "special additions improvements repairs amenities permitted increase expenses",
        "improvement":       "special additions improvements repairs amenities permitted increase expenses",
        "maintenance":       "maintenance service charges amenities repairs permitted increase",
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
    issue_guidance = build_issue_guidance(question)

    user_message = f"""
JURISDICTION: {jurisdiction}

{history_block}

{issue_guidance}

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
