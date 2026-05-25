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
    "pay extra",
    "extra rent",
    "extra amount",
    "extra payment",
    "extra this month",
    "additional rent",
    "additional amount",
    "escalation",
    "rent revision",
    "permitted increase",
)

RECEIPT_PAYMENT_KEYWORDS = (
    "receipt",
    "rent receipt",
    "cash",
    "paid rent",
    "payment proof",
    "proof of payment",
    "no proof",
    "no receipt",
)

LOCKOUT_KEYWORDS = (
    "changed the lock",
    "change the lock",
    "change locks",
    "changed locks",
    "lockout",
    "locked out",
    "locked me out",
    "blocking access",
    "blocked access",
    "physically remove",
    "physically removed",
    "remove my belongings",
    "removed my belongings",
    "tenant belongings",
    "cut electricity",
    "cutting electricity",
    "cut power",
    "cutting power",
    "cut water",
    "cutting water",
    "cut gas",
    "cutting gas",
    "cut utilities",
    "cutting utilities",
    "shut off electricity",
    "shut off water",
    "shut off gas",
    "shut off utilities",
    "threaten",
    "threatened",
    "threatening",
)

ORAL_LEASE_KEYWORDS = (
    "no written lease",
    "without a written lease",
    "never signed",
    "did not sign",
    "didn't sign",
    "oral lease",
    "verbal lease",
    "not signed",
    "no lease agreement",
)

HOLDOVER_KEYWORDS = (
    "lease ended",
    "lease has ended",
    "lease expired",
    "lease has expired",
    "term ended",
    "after my lease",
    "after the lease",
    "keep staying",
    "continue staying",
    "hasn't asked me to leave",
    "has not asked me to leave",
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
   Never make a legal claim without a cited source.
   A wrong section number is worse than no citation. Before finalising, verify
   that the cited section's retrieved text actually supports the answer. If the
   answer relies on a different retrieved section, cite that section instead.
   If no retrieved section directly supports the answer, say the legal provision
   was not retrieved and set CONFIDENCE to LOW.

3. If the lease is silent but the legal context answers the question, answer
   from the cited legal context and clearly say the lease is silent. Lease
   silence does not defeat statutory tenant protection; the law fills the gap.
   If the retrieved law truly contains no provision for the issue, answer:
   "No legal provision exists for this under [JURISDICTION] law."
   Set CONFIDENCE to LOW.

4. Never guess. Never assume. Never invent section numbers.

5. If the lease clause contradicts the law — flag it clearly and explain
   exactly how, citing both the law section and the lease clause.

6. If the user asks about a specific issue (e.g. rent increase) and the
   lease mentions the base topic (e.g. rent amount) but has no provision
   for the specific issue, YOU MUST CITE the base clause and explicitly
   state that it lacks a provision for that issue.
   DO NOT say "No relevant lease clause found" if the base topic clause is present.

7. Do not add disclaimers or referral endings. The product must give the
   tenant the next concrete step from the law and lease, not send them away.

8. DIRECT ANSWER — NO UNSOURCED HEDGING
   - If the question is yes/no and the provided context supports one clear outcome,
     start ANSWER with **Yes** or **No**, then one short supporting phrase with citation in LEGAL BASIS.
   - If the question asks "what notice", "how much", "when", "who", "which
     section", or "what can I do", answer that exact question first. Do not force a Yes/No opening when the user asked for a requirement, amount,
     deadline, actor, section, or next step.
   - If the correct outcome is mixed, start with **Partially** and identify the
     enforceable and unenforceable parts. If facts are missing but the retrieved
     law points strongly in one direction, start with **Likely** and name the
     assumption in EXPLANATION.
   - Never start with "Only if". State the legal result directly. If a statutory
     condition matters, state it as a rule after the direct yes/no answer.
   - Otherwise open ANSWER with the clearest direct factual conclusion the context allows
     (still no waffle).
   - In ANSWER and EXPLANATION, do NOT use banned vague phrases such as:
     "Only if", "It depends on your situation", "You may have rights",
     "Consider seeking legal advice", "The context does not provide enough information",
     "consult a legal professional", "review the Act", "seek legal advice",
     "the next step would be to review",
     "may be entitled", "might be able to", "under certain conditions", "it depends",
     "cannot be determined", "unclear", "ambiguous", "one would need to refer to",
     "does not explicitly say whether", "possibly", "perhaps".
     If you cannot answer without that kind of language, use rule 3's
     no-provision sentence and set CONFIDENCE to LOW.

9. ANSWER WRITING QUALITY — GLOBAL, ALL STATES
   The response must feel like legal analysis, not a search result summary or
   copied law text.
   - Put the most important legal conclusion in the first line of ANSWER.
   - Keep ANSWER to 1-2 sharp sentences. Do not repeat the same point in
     ANSWER and EXPLANATION unless the second mention explains why.
    - LEGAL BASIS supports the answer; it must not replace the answer.
    - Do not copy statutory wording as the answer. Translate the retrieved law
      into a precise conclusion, then cite the section separately.
    - For landlord rent-increase or extra-charge demands, lead with tenant
      protection. Do not start ANSWER with "The landlord is entitled..." or
      "The landlord can...". Prefer: "You do not have to pay an unsupported extra demand..."
      or "A rent increase is limited to...".
    - EXPLANATION should explain why the conclusion follows from the cited law
      and lease, in plain professional language.
   - Lease analysis must be intelligent and contextual: say what the relevant
     clause covers, what it omits, and why that matters. Do not repeatedly write
     "no clause found" when a related clause exists.
   - Use precise legal verbs: "must", "cannot", "is enforceable", "is not
     enforceable", "requires", "does not require", "is protected". Avoid
     vague modal language like "could", "may", or "depends" unless the legal
     text itself creates discretion or a fact is genuinely missing.
   - If facts are unclear, state the assumption briefly in EXPLANATION and then
     give the answer under that assumption.
   - LOW confidence must be useful: say exactly what is missing, such as the
     relevant statutory provision, the lease clause, the rent amount, the notice
     text, possession status, or payment proof.
   - Avoid filler, generic motivation, robotic template phrasing, and obvious
     over-explanation.
   - End with a stronger concrete next step, not a referral or document-review
     placeholder.

10. CONVERSATION CONTINUITY — if a CONVERSATION HISTORY block is provided:
   - Build on previously established facts; do not re-explain them.
   - Do not contradict a prior answer unless the current context explicitly
     requires it — and if you must contradict, explain why.
   - Use prior context to resolve ambiguous pronouns ("it", "that clause", etc.).

11. LEASE SILENCE IS A LEASE FINDING, NOT A LAW FINDING
    If the lease has no clause on the issue, say so clearly in LEASE REFERENCE
    and EXPLANATION. Do NOT say the lease permits the action.
    Then apply only the cited legal context to explain whether the law allows,
    limits, or does not answer the action. Never answer "Yes" solely because
    the law has a broad permission if the retrieved legal text gives conditions.

12. RENT / CHARGE INCREASE CATEGORY SAFETY
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

13. ACTIONABILITY WITHOUT NEW UI FIELDS
    Keep the exact response labels below. In EXPLANATION, add one short final
    sentence with the next document-grounded step when supported by context,
    such as requesting written basis, checking whether the lease clause exists,
    or disputing an unsupported demand in writing. Do not draft notices unless
    the user asks.

14. CONSISTENCY ACROSS QUESTION PHRASING
    If the current question is about rent increase, rent hike, extra rent,
    an extra amount demanded by the landlord, escalation, permitted increase,
    rent increase notice, or disputing a rent increase,
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
    a notice period, state that notice period first. If it gives no notice
    period but gives a legal mechanism, say that no standalone notice period was
    retrieved and then state the mechanism. Do not start such an answer with
    "No" unless the user asked whether a notice is valid or allowed.
    If the retrieved legal mechanism requires court fixation, court
    determination, an application, consent, certificate, or another statutory
    condition, explain that a unilateral landlord notice alone is not enough
    under the retrieved context. The next step should be tenant-protective and
    document-grounded, e.g. ask the landlord for the cited legal basis or
    dispute an unsupported unilateral increase in writing.
    For extra rent or extra-payment demands, the ANSWER must start from the
    tenant's protection against unsupported demands, then explain any permitted
    increase. Do not lead with landlord entitlement wording.
    Do not write that the law "does not provide a specific percentage or
    procedure" when the retrieved context contains any percentage, cap, formula,
    court mechanism, consent requirement, certificate requirement, or other
    procedure. If the retrieved rule is only for a special category, say it is
    limited to that category and cannot by itself validate a general verbal rent
    increase.

15. RENT RECEIPT / CASH PAYMENT SAFETY
    If the current question is about paying rent in cash, missing receipts,
    proof of payment, or rent receipts, treat it as the receipt/payment cluster.
    Check the retrieved context for a statutory duty to issue written receipts,
    punishment/fine for failure, and any lease silence. If the retrieved law says
    the landlord must issue a written receipt, frame the answer as the landlord's
    obligation and possible violation, not as the tenant being unprotected.
    For questions phrased as "am I protected?" or "I paid cash but have no
    receipt", do not start with "No" solely because the tenant lacks a receipt.
    Lead with the statutory duty if it appears in the retrieved context.
    Suggested next steps should be document-grounded: request a written receipt,
    preserve bank/UPI messages, witnesses, chats, or other proof if available,
    and consider filing/raising a complaint if the retrieved law provides a
    penalty for failure to issue receipts.

════════════════════════════════════════
CONFIDENCE RUBRIC — set this field precisely
════════════════════════════════════════

HIGH   → The answer cites a specific statutory section or the lease document
         directly and explicitly answers the question. No external lookup
         needed. Answer is definitive. This includes clear statutory answers
         where the lease is silent, and clear lease-law conflicts.

MEDIUM → The answer is from retrieved law text or lease text but does not cite
         a specific section, OR it requires connecting related clauses/statutory
         categories. Some ambiguity exists, but the answer is still grounded.

LOW    → No retrieved legal provision covers the issue. Use the exact rule 3
         no-provision sentence. Never show LOW confidence on clear statutory
         answers such as lock changes, no written lease, or court-protected
         possession after lease expiry.

SELF-CHECK — before finalising your response, verify:
  • ANSWER must follow rule 8 (direct Yes/No or direct fact; no hedging list).
    If any hedging slipped in, rewrite ANSWER; if rewrite is impossible,
    use rule 3's exact sentence and CONFIDENCE LOW.
  • If hedging still appears in ANSWER → rewrite the answer before finalising.
  • If a specific statutory section answers the question → confidence should be HIGH.
  • The LEGAL BASIS section number must match the statute text used in
    EXPLANATION. If the section text and explanation discuss different legal
    issues, fix the citation or lower confidence to LOW.
  • LEGAL BASIS should include the Act, section number, and the section title
    when that title appears in the provided legal context. Keep it to one
    compact citation line, not a paragraph.
  • If the lease clause present addresses the BASE TOPIC but NOT the specific
    issue asked, say the lease is silent on that issue. Then answer only from
    cited law if the law directly answers it.
  • If NO lease clause addresses even the base topic → MEDIUM or LOW unless
    the cited law alone directly answers the question.

════════════════════════════════════════
LEASE REFERENCE FIELD — strict format
════════════════════════════════════════

  • Never return clause numbers alone. Do NOT write only "Clause 1",
    "Clause 2", "Lease Clause 3", or similar.
  • Always include the relevant lease clause heading/label AND the actual
    lease text that supports the answer.
  • Format direct lease references like:
    "Clause X (HEADING): 'exact or tightly paraphrased lease text relevant
    to the question'"
  • If no lease clause directly addresses the question, say:
    "No clause in this lease directly addresses [topic].
    Closest relevant clause — Clause X (HEADING): 'actual text from the
    closest relevant clause'"
  • If the provided lease citation is a page label because no clause number was
    detected, use the page label plus the relevant text.
  • Keep this field short, but include enough text for the tenant to understand
    what the lease actually says.

════════════════════════════════════════
RESPONSE FORMAT — always use these exact labels on separate lines
════════════════════════════════════════

ANSWER:
[Yes/No/Partially/Likely first when applicable, else the direct conclusion — 1-2 short sentences, no hedging]

LEGAL BASIS:
[Exact Citation labels from the provided legal context only, plus the section
 title when it appears in that context. Keep each citation compact.]

LEASE REFERENCE:
[Clause heading/label plus relevant lease text.
 If no clause directly answers the issue, use:
 "No clause in this lease directly addresses [topic].
 Closest relevant clause — Clause X (HEADING): '[actual text]'"]

EXPLANATION:
[2-3 sentences connecting the law and lease to the answer.
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

        lines.append(f"\n[Lease Excerpt {i + 1}]")
        lines.append(f"Citation : {citation}")
        lines.append(f"Text     : {text}")

    return "\n".join(lines)


def is_rent_increase_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in RENT_INCREASE_KEYWORDS)


def is_receipt_payment_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in RECEIPT_PAYMENT_KEYWORDS)


def is_lockout_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in LOCKOUT_KEYWORDS)


def is_oral_lease_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in ORAL_LEASE_KEYWORDS)


def is_holdover_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return any(keyword in q_lower for keyword in HOLDOVER_KEYWORDS)


def build_issue_guidance(question: str) -> str:
    guidance_blocks: list[str] = []

    if is_lockout_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: LOCKOUT / SELF-HELP EVICTION ════
If the question involves the landlord changing locks, cutting electricity/water/gas, physically removing tenant belongings, threatening the tenant, or blocking access:
- Answer from the retrieved statute for the selected state.
- If the retrieved law prohibits the act or requires court recovery of possession, say it directly and cite that section.
- Never imply a unilateral lock change, forced exclusion, utility cutoff, threat, or removal of belongings is valid unless the retrieved law expressly allows that exact act.
- Give an immediate next step grounded in the retrieved law, such as police complaint, rent court/application, restoration of possession, restoration of essential supply, written dispute, or other remedy actually supported by the retrieved context.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_oral_lease_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: NO WRITTEN LEASE / ORAL TENANCY ════
If the question involves no written lease, no signed lease, an oral lease, or a verbal tenancy:
- Answer from the retrieved statute for the selected state.
- If the retrieved law protects a person in possession, a deemed tenant, licensee, or tenant without a written agreement, say so directly and cite that section.
- Do not write "Only if the Act applies to your situation." Name the actual statutory condition only if the retrieved law makes it necessary.
- If the lease is absent, say the answer is based on statute, not lease-specific facts.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_holdover_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: LEASE ENDED / HOLDOVER POSSESSION ════
If the question involves a lease that ended or expired while the landlord has not recovered possession:
- Answer from the retrieved statute for the selected state.
- If the retrieved law requires court proceedings or limits recovery of possession, say the tenant cannot be removed without that legal process and cite the section.
- If the retrieved law supports continued payment as relevant, tell the tenant to keep paying rent and preserve proof. Do not invent "statutory tenant" language unless the retrieved law supports it.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_rent_increase_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: RENT INCREASE CLUSTER ════
Use one consistent rent-increase analysis for this question.
Check the retrieved context for:
1. ordinary/annual rent increase or standard rent rule;
2. special categories such as additions, improvements, repairs, taxes, amenities, or services;
3. court/dispute mechanism;
4. lease clause or lease silence.
If a relevant rate/cap/formula appears in the retrieved context, include it in ANSWER.
If the question asks what notice is required, answer the notice requirement first. If the retrieved context has no standalone notice period but has a rent-increase mechanism, say that no notice period was retrieved and then state the mechanism. Do not start with "No" unless the user asked a yes/no question.
If the retrieved context includes both a definition section and a substantive rent-increase section, cite the substantive rent-increase section first. Use definition sections only as support.
Do not write that the law lacks a specific percentage or procedure if the retrieved context contains any rent-increase rate, cap, court mechanism, consent requirement, certificate requirement, or other statutory procedure. If the retrieved rule is only for a special category, say it is limited to that category and cannot by itself validate a general verbal rent increase.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_receipt_payment_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: RECEIPT / CASH PAYMENT CLUSTER ════
Use one consistent receipt/payment analysis for this question.
Check the retrieved context for:
1. a landlord duty to issue written receipts for amounts received;
2. penalty or fine for failure to issue receipts;
3. lease clause or lease silence on payment receipts;
4. practical proof of payment the tenant may preserve.
If the retrieved law says the landlord must issue a written receipt, frame the issue as the landlord's statutory obligation and possible violation. Do not answer as though the tenant is simply unprotected because the landlord failed to provide the receipt.
For "am I protected?" wording, do not start with "No" solely because the tenant lacks a receipt; lead with the retrieved statutory duty to issue a written receipt.
════ END ISSUE GUIDANCE ════
""".strip())

    return "\n\n".join(guidance_blocks)

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
        "increase in rent annually rent revision escalation percentage rate cap "
        "formula per annum court fix standard rent permitted increase determine "
        "dispute application special additions improvements structural alterations "
        "heavy repairs additions improvements repairs taxes amenities services "
        "substantive rent increase provision"
    )
    lockout_canonical_hint = (
        "recovery possession landlord tenant lockout changed locks illegal "
        "dispossession restore possession essential supply utilities police "
        "complaint belongings threats court application"
    )
    oral_lease_canonical_hint = (
        "deemed tenant possession paying rent written lease oral verbal agreement "
        "tenant protected tenancy without written agreement"
    )
    holdover_canonical_hint = (
        "statutory tenant lease expired lease ended recovery possession court "
        "order continue paying rent protected possession tenancy after expiry"
    )

    # Order: longer phrases first where overlap matters (e.g. "increase the rent"
    # does not match substring "increase rent").
    query_hints = {
        "increase the rent": rent_increase_canonical_hint,
        "rent increase":     rent_increase_canonical_hint,
        "increase rent":     rent_increase_canonical_hint,
        "pay extra":         rent_increase_canonical_hint,
        "extra rent":        rent_increase_canonical_hint,
        "extra amount":      rent_increase_canonical_hint,
        "extra payment":     rent_increase_canonical_hint,
        "extra this month":  rent_increase_canonical_hint,
        "additional rent":   rent_increase_canonical_hint,
        "additional amount": rent_increase_canonical_hint,
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
        "rent receipt":      "giving receipt amount received compulsory written receipt landlord failure fine rent payment cash proof",
        "receipt":           "giving receipt amount received compulsory written receipt landlord failure fine rent payment cash proof",
        "cash":              "giving receipt amount received compulsory written receipt landlord failure fine rent payment cash proof",
        "paid rent":         "giving receipt amount received compulsory written receipt landlord failure fine rent payment cash proof",
        "payment proof":     "giving receipt amount received compulsory written receipt landlord failure fine rent payment cash proof",
        "changed the lock":  lockout_canonical_hint,
        "change the lock":   lockout_canonical_hint,
        "changed locks":     lockout_canonical_hint,
        "change locks":      lockout_canonical_hint,
        "lockout":           lockout_canonical_hint,
        "locked out":        lockout_canonical_hint,
        "blocking access":   lockout_canonical_hint,
        "belongings":        lockout_canonical_hint,
        "cut electricity":   lockout_canonical_hint,
        "cutting electricity": lockout_canonical_hint,
        "cut water":         lockout_canonical_hint,
        "cutting water":     lockout_canonical_hint,
        "cut gas":           lockout_canonical_hint,
        "cutting gas":       lockout_canonical_hint,
        "cut utilities":     lockout_canonical_hint,
        "cutting utilities": lockout_canonical_hint,
        "shut off electricity": lockout_canonical_hint,
        "shut off water":    lockout_canonical_hint,
        "shut off gas":      lockout_canonical_hint,
        "shut off utilities": lockout_canonical_hint,
        "threat":            lockout_canonical_hint,
        "never signed":      oral_lease_canonical_hint,
        "no written lease":  oral_lease_canonical_hint,
        "written lease":     oral_lease_canonical_hint,
        "oral lease":        oral_lease_canonical_hint,
        "verbal lease":      oral_lease_canonical_hint,
        "lease ended":       holdover_canonical_hint,
        "lease expired":     holdover_canonical_hint,
        "term ended":        holdover_canonical_hint,
        "keep staying":      holdover_canonical_hint,
        "continue staying":  holdover_canonical_hint,
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


def build_retrieval_queries(question: str, history: Optional[list[dict]] = None) -> list[str]:
    """
    Return one or more query strings for retrieval.

    Most issues use a single expanded query. Rent increase questions are split
    into a few targeted queries so one phrasing cannot accidentally retrieve
    only the court/dispute chunk or only the special-improvement chunk.
    """
    primary = build_retrieval_query(question, history)
    if is_lockout_question(question):
        targeted_queries = [
            "recovery possession tenant landlord court process eviction possession",
            "landlord changed locks tenant dispossessed restore possession court application",
            "lockout cutting utilities removing belongings threatening tenant essential supply remedy",
        ]
    elif is_oral_lease_question(question):
        targeted_queries = [
            "tenant deemed possession paying rent oral tenancy written agreement",
            "no written lease oral tenancy tenant rights protected possession rent",
            "person in possession paying rent deemed tenant written agreement not signed",
        ]
    elif is_holdover_question(question):
        targeted_queries = [
            "recovery possession tenant court order lease expired tenancy protected",
            "lease expired statutory tenant cannot evict without court order",
            "continue paying rent evidence tenancy landlord has not asked to leave",
        ]
    elif is_receipt_payment_question(question):
        targeted_queries = [
            "giving receipt for any amount received compulsory written receipt landlord",
            "landlord fails to give written receipt amount received fine default",
            "rent receipt cash payment proof tenant paid rent receipt",
        ]
    elif is_rent_increase_question(question):
        targeted_queries = [
            (
                "increase in rent annually standard rent permitted increase "
                "ordinary annual rent increase percentage rate cap formula per annum "
                "substantive rent increase provision"
            ),
            (
                "court fix standard rent permitted increase dispute application "
                "determine amount permitted increase"
            ),
            (
                "increase in rent on account of improvement special additions improvements "
                "structural alterations special or heavy repairs expenses increase rent"
            ),
        ]
    else:
        return [primary]

    queries: list[str] = []
    seen: set[str] = set()
    for query in [primary, *targeted_queries]:
        normalized = " ".join(query.split())
        if normalized not in seen:
            queries.append(normalized)
            seen.add(normalized)
    return queries

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
