"""
Prompt and parsing definitions for detecting multiple distinct visible documents or separate image items.
Part of the mandatory Multiple-Document / Multiple-Image Detection Rule for the KYC Document Sorter.
"""

from __future__ import annotations

import re
from typing import Optional

# ==============================================================================
# MULTIPLE DOCUMENT DETECTION PROMPT
# ==============================================================================
# Evaluates whether a single image contains two or more separate document/image
# items (such as Aadhaar + PAN together, two ID cards, an ID card next to a bill, etc.).
# Strictly ensures that front/back of same ID, internal tables, and multi-section
# bills are NOT mistaken for multiple documents.
# ==============================================================================

MULTI_DOC_PROMPT = """You are an expert visual document inspector.

Inspect the complete image carefully. Determine whether this image contains:
1. Exactly ONE document or image item (e.g., a single identity card, a single certificate, a single bill, a single bank statement page, a single form, or a single letter).
2. OR MULTIPLE distinct visible documents or separate image items (e.g., two or more separate cards/documents photographed or scanned together in the same file).

EXAMPLES OF MULTIPLE DOCUMENTS:
- An Aadhaar card AND a PAN card visible together in one image.
- An Aadhaar card AND a Voter ID card visible together in one image.
- An ID card AND a utility bill (electricity/gas/phone) visible together in one image.
- Two distinct cards belonging to different ID categories or different people.
- Two or more separate, independent physical document items placed side by side or photographed together.

CRITICAL SINGLE DOCUMENT RULES (MUST RETURN SINGLE_DOCUMENT):
- FRONT AND BACK VIEW OF THE SAME PHYSICAL ID CARD: Showing both the front side and back side of the same identity card (e.g. Aadhaar card front & back, Voter ID front & back, Driving Licence front & back, or e-Aadhaar folded letter) is EXACTLY ONE document => SINGLE_DOCUMENT.
- RATION CARDS / FAMILY CARDS: A ration card containing multi-column tables of household members, family details, and food grain quotas is EXACTLY ONE document => SINGLE_DOCUMENT.
- GAS BILLS / UTILITY INVOICES: A bill sheet that includes an attached refill receipt, cash memo, or consumer card printed on the same paper is EXACTLY ONE document => SINGLE_DOCUMENT.
- ID CARDS WITH INTERNAL BOXES: An ID card containing a portrait photo, signature box, QR code box, emblem, and barcode within its borders is EXACTLY ONE document => SINGLE_DOCUMENT.
- PASSPORT PAGES: A passport booklet page with stamps, seals, or multiple printed sections is EXACTLY ONE document => SINGLE_DOCUMENT.
- Do NOT mistake internal tables, columns, logos, stamps, signatures, or printed sections of a single page for multiple documents.
- Only return MULTIPLE_DOCUMENTS when there are two or more distinct, separate physical documents or cards present in the image.

Return EXACTLY ONE of the following answers:
SINGLE_DOCUMENT
MULTIPLE_DOCUMENTS

No explanation.
No reasoning.
No additional text."""


def parse_multi_document_response(raw_output: Optional[str]) -> Optional[bool]:
    """
    Parses the model response for the multiple-document detection prompt.

    Returns:
        True:  Multiple distinct documents/items detected.
        False: Exactly one document/item detected (or single valid document category returned).
        None:  Could not determine / empty / invalid response.
    """
    if not raw_output:
        return None

    cleaned = str(raw_output).strip().upper()
    for char in ('`', '"', "'", '\n', '\r', '*', '#'):
        cleaned = cleaned.replace(char, " ")
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return None

    # Check for MULTIPLE_DOCUMENTS indicators
    if (
        "MULTIPLE_DOCUMENTS" in cleaned
        or "MULTIPLE DOCUMENTS" in cleaned
        or "MULTIPLE_DOCUMENT" in cleaned
        or "MULTIPLE DOCUMENT" in cleaned
        or re.search(r"\bMULTIPLE\b", cleaned)
    ):
        return True

    # Check for SINGLE_DOCUMENT indicators
    if (
        "SINGLE_DOCUMENT" in cleaned
        or "SINGLE DOCUMENT" in cleaned
        or re.search(r"\bSINGLE\b", cleaned)
    ):
        return False

    # Also check if response is a recognized single KYC category (e.g. from existing unit test mocks)
    from kyc_sorter.sorter.vlm_router import VALID_CATEGORIES
    if cleaned in VALID_CATEGORIES:
        return False

    for cat in VALID_CATEGORIES:
        if cat in cleaned:
            return False

    return None
