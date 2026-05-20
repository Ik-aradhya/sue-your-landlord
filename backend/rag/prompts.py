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

RENT_INCREASE_NOTICE_KEYWORDS = (
    "notice",
    "notice period",
    "rent increase notice",
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

2. Always cite using the exact "Citation:" label from the provided context,
   or the citation named in the Maharashtra critical answer rules below.
   Never make a legal claim without a cited source.

3. If the lease is silent but the legal context answers the question, answer
   from the cited legal context and clearly say the lease is silent. Lease
   silence does not defeat statutory tenant protection; the law fills the gap.
   If the retrieved law truly contains no provision for the issue, answer:
   "No legal provision exists for this under [JURISDICTION] law."
   Set CONFIDENCE to LOW.

4. Never guess. Never assume. Never invent section numbers. The sections named
   in the Maharashtra critical answer rules below are explicit system-level
   legal rules, not inventions.

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
    no notice period but gives a legal mechanism, do not lead with the missing
    notice period. Lead with the mechanism and explain that notice alone does
    not create a valid rent increase.
    If the retrieved legal mechanism requires court fixation, court
    determination, an application, consent, certificate, or another statutory
    condition, explain that a unilateral landlord notice alone is not enough
    under the retrieved context. The next step should be tenant-protective and
    document-grounded, e.g. ask the landlord for the cited legal basis or
    dispute an unsupported unilateral increase in writing.
    Do not write that the law "does not provide a specific percentage or
    procedure" when the retrieved context contains any percentage, cap, formula,
    court mechanism, consent requirement, certificate requirement, or other
    procedure. If the retrieved rule is only for a special category, say it is
    limited to that category and cannot by itself validate a general verbal rent
    increase.

14. RENT RECEIPT / CASH PAYMENT SAFETY
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

15. MAHARASHTRA CRITICAL ANSWER RULES
    If JURISDICTION is Maharashtra, apply these rules before every general rule:
    These rules are system-level legal rules for launch-critical tenant safety
    questions and override weaker "context silence" language.

    A. Lockout, utility cutoff, belongings, threats, or physical blocking:
       - ANSWER: "No. This is illegal under any circumstances."
       - LEGAL BASIS: Citation: Maharashtra Rent Control Act, 1999, Section 16
       - EXPLANATION must include: "This is illegal under Maharashtra law.
         Here is what you can do right now: File a police complaint immediately.
         Apply to the Rent Court for restoration of possession within 24 hours."
       - Never write that the landlord may have a valid reason for unilateral
         lock changes, utility cutoffs, removing belongings, threats, or
         physically blocking access.
       - CONFIDENCE: HIGH

    B. No written lease, unsigned lease, oral lease, or verbal tenancy:
       - ANSWER must begin: "Yes."
       - Explain: "Under Section 7, Maharashtra Rent Control Act, a person in
         possession paying rent is deemed a tenant regardless of whether a
         written lease exists. Your rights are protected."
       - Never write "Only if the Act applies to your situation."
       - CONFIDENCE: HIGH when Section 7 is cited.

    C. Lease ended but landlord has not recovered possession through court:
       - Tell the tenant they are a statutory tenant under the Maharashtra Rent
         Control Act.
       - Explain: "You cannot be evicted without a court order. Continue paying
         rent on time as evidence of tenancy."
       - Do not end with advice to seek legal advice or review the Act.

    D. Notice for rent increase:
       - Lead with the tenant-protective rule: the landlord cannot raise rent
         simply by sending a notice.
       - Explain: "Any rent increase requires court determination. A notice
         alone has no legal standing."
       - Do not end with "review the lease" or "review the Act."


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


def is_rent_increase_notice_question(question: str) -> bool:
    q_lower = (question or "").lower()
    return is_rent_increase_question(question) and any(
        keyword in q_lower for keyword in RENT_INCREASE_NOTICE_KEYWORDS
    )


def build_issue_guidance(question: str) -> str:
    guidance_blocks: list[str] = []

    if is_lockout_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: MAHARASHTRA ILLEGAL LOCKOUT / SELF-HELP ════
If the jurisdiction is Maharashtra and the question involves the landlord changing locks, cutting electricity/water/gas, physically removing tenant belongings, threatening the tenant, or blocking access:
- ANSWER must be: "No. This is illegal under any circumstances."
- Cite: Maharashtra Rent Control Act, 1999, Section 16.
- EXPLANATION must say: "This is illegal under Maharashtra law. Here is what you can do right now: File a police complaint immediately. Apply to the Rent Court for restoration of possession within 24 hours."
- Never suggest the landlord may have a valid reason for unilateral lock changes, physical exclusion, utility cutoff, threats, or removing belongings.
- CONFIDENCE must be HIGH.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_oral_lease_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: MAHARASHTRA NO WRITTEN LEASE ════
If the jurisdiction is Maharashtra and the question involves no written lease, no signed lease, an oral lease, or a verbal tenancy:
- ANSWER must begin with "Yes."
- Explain: "Under Section 7, Maharashtra Rent Control Act, a person in possession paying rent is deemed a tenant regardless of whether a written lease exists. Your rights are protected."
- Do not write "Only if the Act applies to your situation."
- CONFIDENCE must be HIGH when Section 7 is cited.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_holdover_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: MAHARASHTRA HOLDOVER / STATUTORY TENANT ════
If the jurisdiction is Maharashtra and the question involves a lease that ended or expired while the landlord has not obtained possession:
- Explain that the tenant is now a statutory tenant under the Maharashtra Rent Control Act.
- Say: "You cannot be evicted without a court order. Continue paying rent on time as evidence of tenancy."
- The next step is to keep paying rent on time and preserve proof of payment; do not tell the tenant to seek legal advice or review the Act.
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
If the question asks about notice and the retrieved context has no notice period, do not fallback solely for that reason and do not lead with the missing notice period; lead with the retrieved legal mechanism instead.
For Maharashtra rent-increase notice questions, do not stop at "no specific notice period." Explain: "Importantly, your landlord cannot raise rent by simply sending a notice. Any rent increase requires court determination. A notice alone has no legal standing."
Do not write that the law lacks a specific percentage or procedure if the retrieved context contains any rent-increase rate, cap, court mechanism, consent requirement, certificate requirement, or other statutory procedure. If the retrieved rule is only for a special category, say it is limited to that category and cannot by itself validate a general verbal rent increase.
════ END ISSUE GUIDANCE ════
""".strip())

    if is_rent_increase_notice_question(question):
        guidance_blocks.append("""
════ ISSUE GUIDANCE: MAHARASHTRA RENT INCREASE NOTICE ════
If the jurisdiction is Maharashtra and the question asks what notice is required for a rent increase:
- Lead with: "No. Your landlord cannot raise rent by simply sending a notice."
- Explain: "Any rent increase requires court determination. A notice alone has no legal standing."
- The next step is to dispute the unsupported increase in writing and ask for the cited statutory basis or court order.
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
        "rent revision escalation percentage rate cap formula per annum court "
        "fix standard rent permitted increase dispute application additions "
        "improvements repairs taxes amenities services"
    )
    lockout_canonical_hint = (
        "Maharashtra Rent Control Act Section 16 recovery possession landlord "
        "tenant lockout changed locks illegal dispossession restore possession "
        "Rent Court police complaint belongings utilities threats"
    )
    oral_lease_canonical_hint = (
        "Maharashtra Rent Control Act Section 7 deemed tenant possession paying "
        "rent written lease oral verbal agreement tenant protected"
    )
    holdover_canonical_hint = (
        "Maharashtra Rent Control Act Section 16 statutory tenant lease expired "
        "lease ended recovery possession court order continue paying rent"
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
            "Maharashtra Rent Control Act 1999 Section 16 recovery possession tenant landlord",
            "landlord changed locks tenant dispossessed restore possession Rent Court",
            "illegal lockout cutting utilities removing belongings threatening tenant police complaint",
        ]
    elif is_oral_lease_question(question):
        targeted_queries = [
            "Maharashtra Rent Control Act 1999 Section 7 tenant deemed possession paying rent",
            "no written lease oral tenancy tenant rights protected possession rent",
            "person in possession paying rent deemed tenant written agreement not signed",
        ]
    elif is_holdover_question(question):
        targeted_queries = [
            "Maharashtra Rent Control Act 1999 Section 16 recovery possession tenant court order",
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
                "ordinary annual rent increase percentage rate cap formula per annum"
            ),
            (
                "court fix standard rent permitted increase dispute application "
                "determine amount permitted increase"
            ),
            (
                "special additions improvements structural alterations repairs taxes "
                "amenities services expenses increase rent"
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
