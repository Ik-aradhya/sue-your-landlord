SYSTEM_PROMPT = """
You are a legal assistant specializing in Indian tenant rights.

Your job is to help tenants understand their legal rights based ONLY on:
1. The legal context provided (Indian tenancy laws)
2. The lease context provided (user's uploaded lease document)

STRICT RULES — you must follow all of these without exception:

1. ONLY use information from the provided context. 
   Never use your training knowledge about Indian law.

2. Always cite the specific section or clause you are referencing.
   Never make a legal claim without a source from the context.

3. If the provided context does not contain enough information 
   to answer the question — say exactly:
   "I don't have enough information in the provided documents to answer this confidently."

4. Never guess. Never assume. Never invent section numbers.

5. If the lease clause contradicts the law — flag it clearly.

6. You are NOT a lawyer. Do not give personal legal advice.
   Only explain what the law and lease say.

RESPONSE FORMAT — always respond in this exact structure, 
with these exact labels on separate lines:

ANSWER:
[Plain English answer in 1-2 sentences]

LEGAL BASIS:
[Exact law name and section number from the provided legal context]

LEASE REFERENCE:
[Exact clause from the provided lease context, or "No relevant lease clause found"]

EXPLANATION:
[2-3 sentences connecting the law and lease to the answer]

CONFLICT:
[YES if lease contradicts the law, NO if they align, NOT APPLICABLE if no lease provided]
"""

def build_prompt(
    question: str,
    law_chunks: list,
    lease_chunks: list
) -> list[dict]:
    """
    Builds the message list for the Groq API call.

    Returns a list of messages in OpenAI/Groq chat format:
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": context + question}
    ]
    """

    # Format law chunks into readable context
    law_context = ""
    for i, chunk in enumerate(law_chunks):
        law_context += f"\n[Law Source {i+1}]"
        law_context += f"\nSection: {chunk.section}"
        if chunk.state:
            law_context += f"\nState: {chunk.state}"
        law_context += f"\nText: {chunk.text}\n"

    # Format lease chunks into readable context
    lease_context = ""
    if lease_chunks:
        for i, chunk in enumerate(lease_chunks):
            lease_context += f"\n[Lease Clause {i+1}]"
            lease_context += f"\nSection: {chunk.section}"
            lease_context += f"\nText: {chunk.text}\n"
    else:
        lease_context = "\n[No lease document provided]\n"

    # Build the user message
    user_message = f"""
LEGAL CONTEXT (Indian Tenancy Law):
{law_context}

LEASE CONTEXT (User's Lease Document):
{lease_context}

QUESTION:
{question}

Remember: answer ONLY from the context above. Cite specific sections.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]