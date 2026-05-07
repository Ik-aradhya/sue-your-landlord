
import sys
sys.path.append(".")

from backend.rag.chain import run_rag_chain

def run_test():
    print("\n" + "="*60)
    print("TEST — Full RAG pipeline end to end")
    print("="*60)

    # Your 3 core use cases from Day 1
    test_cases = [
        "Can my landlord increase rent during an active lease?",
        "What notice period is required before eviction?",
        "Is my security deposit refundable?"
    ]

    for question in test_cases:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        response = run_rag_chain(
            question=question,
            state="maharashtra",
            session_id="test_session_001"
        )

        print(f"ANSWER      : {response.answer}")
        print(f"LEGAL BASIS : {response.legal_basis}")
        print(f"LEASE REF   : {response.lease_reference}")
        print(f"EXPLANATION : {response.explanation}")
        print(f"CONFIDENCE  : {response.confidence}")
        print(f"CONFLICT    : {response.conflict_flag}")
        print(f"ERROR       : {response.error_type}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_test()