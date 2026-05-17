"""
test_pinecone_retrieval.py
--------------------------
Standalone test for Pinecone chunk retrieval.

Run from the project root:
    python test_pinecone_retrieval.py

Tests:
  1. Pinecone connection & index stats
  2. Raw query — checks vectors actually come back with text
  3. State filter   — ensures metadata filter works (maharashtra / gujarat)
  4. Retriever layer — exercises run_retrieval() end-to-end
  5. Chunk quality   — validates every chunk has non-empty text + section
"""

import os, sys, textwrap

# ── allow `from backend.xxx import …` when running from project root ──────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# load .env so Settings() can find the keys
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from pinecone import Pinecone
from backend.core.config import settings
from backend.core.database import get_embedding, get_law_index
from backend.rag.retriever import run_retrieval

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
INFO = f"{CYAN}ℹ INFO{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

def hr():
    print("─" * 60)

# =============================================================================
# TEST 1 — Pinecone connection & index stats
# =============================================================================
def test_connection():
    hr()
    print(f"\n{CYAN}TEST 1 — Pinecone connection & index stats{RESET}\n")
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        print(f"  Index name  : {settings.PINECONE_INDEX_NAME}")
        print(f"  Total vectors: {total}")
        if total == 0:
            print(f"  {WARN} Index is empty — ingest law documents first!")
            return False
        print(f"  {PASS} Connected. {total} vectors found.")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


# =============================================================================
# TEST 2 — Raw Pinecone query
# =============================================================================
def test_raw_query():
    hr()
    print(f"\n{CYAN}TEST 2 — Raw Pinecone query (no state filter){RESET}\n")
    query = "What are the rights of a tenant regarding rent increase?"
    try:
        vec = get_embedding(query)
        index = get_law_index()
        results = index.query(
            vector=vec,
            top_k=5,
            include_metadata=True
        )
        matches = results.get("matches", [])
        if not matches:
            print(f"  {FAIL} No matches returned.")
            return False

        print(f"  Query    : \"{query}\"")
        print(f"  Returned : {len(matches)} chunks\n")
        for i, m in enumerate(matches):
            score    = m.get("score", 0.0)
            meta     = m.get("metadata", {})
            text     = meta.get("text", "")
            section  = meta.get("section", "—")
            state    = meta.get("state", "—")
            snippet  = textwrap.shorten(text, width=120, placeholder="…")

            # Per-chunk verdict
            ok = bool(text.strip()) and bool(section.strip())
            mark = PASS if ok else FAIL

            print(f"  [{i+1}] {mark}  score={score:.4f}  state={state}")
            print(f"       section : {section}")
            print(f"       snippet : {snippet}\n")

        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


# =============================================================================
# TEST 3 — State metadata filter
# =============================================================================
def test_state_filter():
    hr()
    print(f"\n{CYAN}TEST 3 — State metadata filter{RESET}\n")
    query = "Can the landlord evict a tenant without notice?"
    passed = True

    for state in ["maharashtra", "gujarat"]:
        try:
            vec = get_embedding(query)
            index = get_law_index()
            results = index.query(
                vector=vec,
                top_k=3,
                filter={"state": state},
                include_metadata=True
            )
            matches = results.get("matches", [])
            print(f"  State = {state!r}  →  {len(matches)} match(es)")
            if matches:
                for m in matches:
                    returned_state = m.get("metadata", {}).get("state", "—")
                    correct = returned_state.lower() == state.lower()
                    mark = PASS if correct else FAIL
                    print(f"    {mark}  returned state={returned_state!r}  score={m['score']:.4f}")
                    if not correct:
                        passed = False
            else:
                print(f"    {WARN} No results for state={state!r} — no data ingested for this state?")
        except Exception as e:
            print(f"  {FAIL} {e}")
            passed = False

    return passed


# =============================================================================
# TEST 4 — run_retrieval() end-to-end
# =============================================================================
def test_run_retrieval():
    hr()
    print(f"\n{CYAN}TEST 4 — run_retrieval() (retriever layer){RESET}\n")

    # Use a fake session_id so ChromaDB lease collection returns 0 results
    # (we only want to exercise the law / Pinecone path here)
    test_cases = [
        {
            "question": "What notice period must a landlord give before eviction?",
            "state":    "maharashtra",
            "session_id": "test-session-does-not-exist",
        },
        {
            "question": "Can a landlord increase rent arbitrarily?",
            "state":    "gujarat",
            "session_id": "test-session-does-not-exist",
        },
    ]

    all_ok = True
    for tc in test_cases:
        try:
            result = run_retrieval(
                question=tc["question"],
                state=tc["state"],
                session_id=tc["session_id"],
            )
            ctx = result.get("context")
            err = result.get("error_type")
            fallback = result.get("should_fallback")

            if err:
                print(f"  {FAIL}  error_type={err}")
                all_ok = False
                continue

            law_n   = len(ctx.law_chunks)   if ctx else 0
            lease_n = len(ctx.lease_chunks) if ctx else 0
            conf    = ctx.confidence.value  if ctx else "—"
            law_sc  = ctx.law_score         if ctx else "—"

            ok = law_n > 0
            mark = PASS if ok else FAIL
            if not ok:
                all_ok = False

            print(f"  Q: \"{tc['question']}\"")
            print(f"     state={tc['state']}  {mark}")
            print(f"     law_chunks={law_n}  lease_chunks={lease_n}  "
                  f"law_score={law_sc:.4f}  confidence={conf}  fallback={fallback}\n")

        except Exception as e:
            print(f"  {FAIL} {e}")
            all_ok = False

    return all_ok


# =============================================================================
# TEST 5 — Chunk quality check (every stored chunk)
# =============================================================================
def test_chunk_quality():
    hr()
    print(f"\n{CYAN}TEST 5 — Chunk quality (sample 50 vectors){RESET}\n")
    try:
        # Pinecone doesn't support "list all" easily; use a zero-vector fetch
        # as a proxy to grab a page of vectors.
        index = get_law_index()
        # embed a broad phrase to get a diverse top-50
        vec = get_embedding("landlord tenant rent law India")
        results = index.query(vector=vec, top_k=50, include_metadata=True)
        matches = results.get("matches", [])

        if not matches:
            print(f"  {WARN} No vectors to inspect.")
            return True

        missing_text    = 0
        missing_section = 0
        missing_state   = 0
        empty_text      = 0

        for m in matches:
            meta = m.get("metadata", {})
            text    = meta.get("text", "")
            section = meta.get("section", "")
            state   = meta.get("state", "")

            if text is None:        missing_text += 1
            elif not text.strip():  empty_text   += 1
            if not section.strip(): missing_section += 1
            if not state.strip():   missing_state   += 1

        total = len(matches)
        ok = (missing_text + empty_text + missing_section) == 0

        print(f"  Sampled       : {total} chunks")
        print(f"  Missing text  : {missing_text}")
        print(f"  Empty text    : {empty_text}")
        print(f"  Missing section: {missing_section}")
        print(f"  Missing state : {missing_state} (may be ok for global law)")
        print()
        if ok:
            print(f"  {PASS} All sampled chunks have text + section.")
        else:
            print(f"  {FAIL} Some chunks are malformed — check ingestion pipeline.")
        return ok

    except Exception as e:
        print(f"  {FAIL} {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print(f"  Pinecone Retrieval Test Suite")
    print(f"  Index: {settings.PINECONE_INDEX_NAME}")
    print(f"{'═'*60}\n")

    results = {
        "Connection & stats": test_connection(),
        "Raw query":          test_raw_query(),
        "State filter":       test_state_filter(),
        "run_retrieval()":    test_run_retrieval(),
        "Chunk quality":      test_chunk_quality(),
    }

    hr()
    print(f"\n{CYAN}SUMMARY{RESET}\n")
    all_passed = True
    for name, ok in results.items():
        mark = PASS if ok else FAIL
        print(f"  {mark}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}All tests passed!{RESET}")
    else:
        print(f"{RED}Some tests failed — see output above.{RESET}")
    print()
