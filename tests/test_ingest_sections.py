import sys

sys.path.append(".")
sys.path.append("backend")

from backend.models.schemas import DocType
from backend.rag.ingest import chunk_text, find_last_law_section_citation_in_text


def test_law_section_detection_ignores_inline_cross_references():
    text = """
11. Increase in rent annually and on account of improvement etc. special addition etc. and
special or heavy repairs.- (1) A landlord shall be entitled to make an increase of 4 per cent.
per annum in the rent of the premises referred to in sub-section (1) of section 2.
(2) Improvements do not include repairs under sub-section (1) of section 14.
"""

    assert (
        find_last_law_section_citation_in_text(text, "maharashtra")
        == "Maharashtra Rent Control Act, 1999, Section 11"
    )


def test_law_chunking_carries_heading_across_body_chunks():
    text = """
10. Rent in excess of standard rent illegal.- It shall not be lawful to claim increases
above the standard rent and permitted increases.

11. Increase in rent annually and on account of improvement etc. special addition etc. and
special or heavy repairs.- (1) A landlord shall be entitled to make an increase of 4 per cent.
per annum in the rent of the premises referred to in sub-section (1) of section 2.
(2) A landlord shall also be entitled to make such increase for improvement or structural
alterations. Improvements and alterations do not include repairs under sub-section (1) of
section 14.
"""

    result = chunk_text(text, DocType.LAW, state="maharashtra")

    assert result["error_type"] is None
    assert result["chunks"][-1].section == "Maharashtra Rent Control Act, 1999, Section 11"
