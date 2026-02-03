"""
PHI Detection Guardrails

Implement functions to detect Protected Health Information (PHI)
in text for safety checks in the RAG pipeline.
"""

import re
from typing import Dict, List, Optional


def detect_phi(text: str) -> Dict[str, List[str]]:
    """
    Detect potential PHI in text.

    Parameters
    ----------
    text : str
        Text to scan for PHI

    Returns
    -------
    dict
        Dictionary with PHI types as keys and lists of matches as values.
        Empty dict if no PHI detected.

    Examples
    --------
    >>> detect_phi("Call me at 555-123-4567")
    {'phone': ['555-123-4567']}

    >>> detect_phi("No PHI here")
    {}
    """
    # TODO: Implement PHI detection
    #
    # At minimum, detect:
    # - Social Security Numbers (XXX-XX-XXXX)
    # - Phone numbers (various formats)
    # - Email addresses
    # - Medical Record Numbers (MRN patterns like MRN: 12345)
    #
    # Hints:
    # - Use re.findall() to find all matches for each pattern
    # - Return empty dict if no PHI found
    # - Consider case-insensitive matching where appropriate
    #
    # Example patterns:
    # ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    # phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    # email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # mrn_pattern = r'\b(MRN|Medical Record)[\s:#]*\d+\b'

    pass


def scan_for_phi(texts: List[str]) -> List[Dict]:
    """
    Scan multiple texts for PHI.

    Parameters
    ----------
    texts : list of str
        List of text strings to scan

    Returns
    -------
    list of dict
        List of results, one per input text.
        Each dict has 'text_index', 'has_phi', and 'phi_found' keys.
    """
    # TODO: Implement batch PHI scanning
    #
    # For each text:
    # 1. Call detect_phi()
    # 2. Record the index, whether PHI was found, and what was found
    #
    # Example return:
    # [
    #     {"text_index": 0, "has_phi": True, "phi_found": {"phone": ["555-1234"]}},
    #     {"text_index": 1, "has_phi": False, "phi_found": {}}
    # ]

    pass


def redact_phi(text: str, phi_results: Dict[str, List[str]]) -> str:
    """
    Redact detected PHI from text.

    Parameters
    ----------
    text : str
        Original text
    phi_results : dict
        Results from detect_phi()

    Returns
    -------
    str
        Text with PHI replaced by [REDACTED]
    """
    # TODO: Implement PHI redaction
    #
    # For each PHI match found, replace it with [REDACTED]
    #
    # Example:
    # text = "Call 555-123-4567"
    # phi_results = {"phone": ["555-123-4567"]}
    # returns "Call [REDACTED]"

    pass


if __name__ == "__main__":
    # Test PHI detection
    test_texts = [
        "Patient John Doe, MRN: 12345678, DOB: 01/15/1980",
        "Contact: john.doe@email.com or 555-123-4567",
        "SSN: 123-45-6789 for insurance purposes",
        "Blood pressure 120/80, no concerns noted",  # No PHI
    ]

    print("PHI Detection Test")
    print("-" * 50)

    for i, text in enumerate(test_texts):
        print(f"\nText {i + 1}: {text}")
        # result = detect_phi(text)
        # print(f"PHI Found: {result if result else 'None'}")
        #
        # if result:
        #     redacted = redact_phi(text, result)
        #     print(f"Redacted: {redacted}")
