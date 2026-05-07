import sys
sys.path.append(".")

from backend.rag.retriever import run_retrieval

def run_test():
    print("\n" + "="*50)
    print("TEST — Full retrieval pipeline")
    print("="*50)

    # Use the 3 core use cases from your product definition
    questions = [
        "Can my landlord increase rent during an active lease?",
        "What notice period is required before eviction?",
        "Is my security deposit refundable?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        result = run_retrieval(
            question=question,
            state="maharashtra",
            session_id="test_session_001"
        )

        if result["error_type"]:
            print(f"ERROR: {result['error_type']}")
            continue

        context = result["context"]
        print(f"Should fallback : {result['should_fallback']}")
        print(f"Confidence      : {context.confidence}")
        print(f"Law score       : {context.law_score:.4f}")
        print(f"Law chunks found: {len(context.law_chunks)}")

        if context.law_chunks:
            print(f"\nTop law chunk:")
            print(context.law_chunks[0].text[:300])

    print("\n" + "="*50)
    print("TEST COMPLETE")

if __name__ == "__main__":
    run_test()