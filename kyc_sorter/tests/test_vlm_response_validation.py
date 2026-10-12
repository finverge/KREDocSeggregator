"""
Regression & Robustness Test Suite for VLM Classification Response Validation.

Validates that:
1. Explanatory text (including the exact error response from TRIVENI_BATHALA_1)
   is safely rejected and routed to Unidentified_files rather than guessing or failing.
2. Structured JSON response formats supported by Ollama are parsed correctly.
3. Plain text single-line category labels and recognized aliases are preserved.
4. Explanatory sentences containing incidental category keywords are NOT blindly matched.
5. All 17 canonical categories (plus MULTIPLE_DOCUMENTS) are correctly supported.
6. The Ollama payload includes the strict structured response format schema.
7. Unidentified routing preserves original filenames without appending suffixes.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kyc_sorter.sorter.vlm_router import (
    CANONICAL_CATEGORY_MAP,
    CLASSIFICATION_SUFFIXES,
    STRUCTURED_CATEGORY_SCHEMA,
    VALID_CATEGORIES,
    classify_document,
    normalize_category,
    process_vlm_file,
    route_document,
)


@pytest.fixture
def test_dirs(tmp_path):
    """Creates isolated input and destination directories for test execution."""
    input_dir = tmp_path / "Source Files" / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "Destination files"
    dest_root.mkdir(parents=True, exist_ok=True)

    kyc_dir = dest_root / "KYC"
    non_kyc_dir = dest_root / "NON KYC"
    unidentified_dir = dest_root / "Unidentified_files"

    for d in (kyc_dir, non_kyc_dir, unidentified_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "input": input_dir,
        "dest_root": dest_root,
        "kyc": kyc_dir,
        "non_kyc": non_kyc_dir,
        "unidentified": unidentified_dir,
    }


def _create_sample_file(directory: Path, filename: str, content: bytes = b"DUMMY_IMAGE_DATA") -> Path:
    p = directory / filename
    p.write_bytes(content)
    return p


# ==============================================================================
# 1. Regression Test: Exact Error Response from Failed Batch File
# ==============================================================================

TRIVENI_FAILED_RESPONSE = (
    "INCOME TAX DEPARTMENT GOVT OF INDIA\n\n"
    "This is clearly an official document from the Income Tax Department "
    "of the Government of India, which is a government-issued ID. The layout "
    "includes a photo, name, permanent account number, and official"
)


def test_triveni_explanatory_response_rejected_by_normalizer():
    """
    Verifies that the exact explanatory response from TRIVENI_BATHALA_1
    is NOT matched to any category by normalize_category.
    """
    category = normalize_category(TRIVENI_FAILED_RESPONSE)
    assert category is None, f"Expected None for explanatory text, got '{category}'"


def test_triveni_explanatory_response_routes_to_unidentified(test_dirs):
    """
    Verifies that when the model outputs the TRIVENI explanatory response:
    - classify_document returns None category with error message.
    - process_vlm_file routes the document safely to Unidentified_files.
    - The original filename is preserved without suffixes.
    - The routing status is UNIDENTIFIED (not crashed).
    """
    filename = "20260120_0935_5527_8040_ba62084d2632_TRIVENI_BATHALA_1.jpg"
    src = _create_sample_file(test_dirs["input"], filename)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": TRIVENI_FAILED_RESPONSE}}

    with patch("requests.post", return_value=mock_resp):
        result = process_vlm_file(
            image_file=src,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

    assert result.status == "UNIDENTIFIED"
    assert result.routing_type == "UNIDENTIFIED"
    assert result.category is None
    assert result.final_filename == filename
    assert result.destination_path == test_dirs["unidentified"] / filename
    assert result.destination_path.is_file()
    assert not src.exists()
    assert not (test_dirs["kyc"] / filename).exists()
    assert not (test_dirs["non_kyc"] / filename).exists()


# ==============================================================================
# 2. Strict Structured Response Format (JSON) Parsing
# ==============================================================================

def test_structured_json_response_all_valid_categories():
    """Verifies that structured JSON format {"category": "<CAT>"} parses for all valid categories."""
    for cat in sorted(VALID_CATEGORIES):
        json_str = json.dumps({"category": cat})
        normalized = normalize_category(json_str)
        expected = "NON_KYC" if cat == "NON_KYC" else cat
        assert normalized == expected, f"Failed for category JSON {json_str!r}: expected {expected!r}, got {normalized!r}"


def test_structured_json_with_formatting_variations():
    """Verifies JSON parsing handles formatting variations (markdown blocks, whitespace, casing)."""
    # Markdown block
    assert normalize_category('```json\n{"category": "PAN CARD"}\n```') == "PAN CARD"
    # Multiline indented JSON
    assert normalize_category('{\n  "category": "AADHAAR CARD"\n}') == "AADHAAR CARD"
    # Lowercase value
    assert normalize_category('{"category": "voter id"}') == "VOTER ID"
    # Document type alternate key
    assert normalize_category('{"document_type": "BANK STATEMENT"}') == "BANK STATEMENT"
    # Classification alternate key
    assert normalize_category('{"classification": "ELECTRICITY BILL"}') == "ELECTRICITY BILL"


def test_structured_json_invalid_category_returns_none():
    """Verifies JSON with non-existent or unallowed categories returns None."""
    assert normalize_category('{"category": "UNKNOWN"}') is None
    assert normalize_category('{"category": "DOG_PICTURE"}') is None
    assert normalize_category('{"category": "INCOME TAX DEPARTMENT"}') is None
    assert normalize_category('{"category": ""}') is None
    assert normalize_category('{"category": null}') is None


# ==============================================================================
# 3. Conversational / Explanatory False-Positive Prevention (Anti-Blind Matching)
# ==============================================================================

@pytest.mark.parametrize(
    "prose_text",
    [
        "This document mentions PAN CARD, but it is actually a personal loan agreement.",
        "Could be VOTER ID or PASSPORT",
        "This is clearly an official Aadhaar card from UIDAI.",
        "NOT A PAN CARD",
        "Maybe Aadhaar",
        "The customer provided a DRIVING LICENCE number on this bank application form.",
        "I am not sure if this is a PASSPORT or a GOVT ID.",
        "The document is an INCOME TAX DEPARTMENT receipt.",
        "Vehicle registration number: DL 01 AB 1234, but this is a loan statement.",
        "This is a photo of a card, possibly GOVT ID, but unreadable.",
    ],
)
def test_conversational_text_rejected(prose_text: str):
    """Verifies that conversational or explanatory prose is never blindly matched."""
    assert normalize_category(prose_text) is None, f"Prose '{prose_text}' should return None"


# ==============================================================================
# 4. Backward Compatibility: Single-Line Labels & Canonical Aliases
# ==============================================================================

@pytest.mark.parametrize(
    "raw_input,expected",
    [
        ("PAN CARD", "PAN CARD"),
        ("AADHAAR CARD", "AADHAAR CARD"),
        ("VOTER ID", "VOTER ID"),
        ("BANK STATEMENT", "BANK STATEMENT"),
        ("CKYC CONFIRMATION", "CKYC CONFIRMATION"),
        ("DRIVING LICENCE", "DRIVING LICENCE"),
        ("ELECTRICITY BILL", "ELECTRICITY BILL"),
        ("GAS BILL", "GAS BILL"),
        ("MUNICIPAL TAX RECEIPT", "MUNICIPAL TAX RECEIPT"),
        ("PHONE BILL", "PHONE BILL"),
        ("RATION CARD", "RATION CARD"),
        ("REGISTRATION CERTIFICATE", "REGISTRATION CERTIFICATE"),
        ("PASSPORT", "PASSPORT"),
        ("GOVT ID", "GOVT ID"),
        ("CKYC DECLARATION", "CKYC DECLARATION"),
        ("IT RETURNS", "IT RETURNS"),
        ("NON_KYC", "NON_KYC"),
        ("MULTIPLE_DOCUMENTS", "MULTIPLE_DOCUMENTS"),
        # Standard Aliases
        ("DRIVING LICENSE", "DRIVING LICENCE"),
        ("DRIVING-LICENCE", "DRIVING LICENCE"),
        ("NON-KYC", "NON_KYC"),
        ("NONKYC", "NON_KYC"),
        ("NON KYC", "NON_KYC"),
        ("MUNICIPAL RECEIPT", "MUNICIPAL TAX RECEIPT"),
        ("MULTIPLE DOCUMENTS", "MULTIPLE_DOCUMENTS"),
        ("IT RETURN", "IT RETURNS"),
        ("VOTER ID CARD", "VOTER ID"),
        ("AADHAR CARD", "AADHAAR CARD"),
        # Prefixed single lines
        ("CATEGORY: PAN CARD", "PAN CARD"),
        ("DOCUMENT TYPE: AADHAAR CARD", "AADHAAR CARD"),
        ("DOCUMENT_TYPE: VOTER ID", "VOTER ID"),
        ("Classification: PASSPORT", "PASSPORT"),
    ],
)
def test_single_line_and_alias_normalization(raw_input: str, expected: str):
    """Verifies single-line inputs and aliases map to canonical category names."""
    assert normalize_category(raw_input) == expected


def test_empty_and_whitespace_returns_none():
    """Verifies empty, whitespace, and None inputs safely return None."""
    assert normalize_category("") is None
    assert normalize_category("   ") is None
    assert normalize_category(None) is None
    assert normalize_category("\n\n") is None


# ==============================================================================
# 5. Schema & Payload Validation
# ==============================================================================

def test_structured_schema_contains_all_17_categories():
    """Verifies STRUCTURED_CATEGORY_SCHEMA enum contains all 17 valid categories + MULTIPLE_DOCUMENTS."""
    enum_vals = set(STRUCTURED_CATEGORY_SCHEMA["properties"]["category"]["enum"])
    assert len(CLASSIFICATION_SUFFIXES) == 16
    assert VALID_CATEGORIES == set(CLASSIFICATION_SUFFIXES.keys()) | {"NON_KYC", "MULTIPLE_DOCUMENTS", "UNIDENTIFIED"}
    assert enum_vals == VALID_CATEGORIES


def test_classify_document_includes_format_in_payload(tmp_path):
    """Verifies that classify_document includes the structured format schema in the Ollama request payload."""
    img_path = tmp_path / "test_format_payload.jpg"
    img_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 100)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": '{"category": "PAN CARD"}'}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        cat, err, _ = classify_document(img_path)

    assert cat == "PAN CARD"
    assert err is None
    assert mock_post.call_count == 1
    call_kwargs = mock_post.call_args[1]
    payload = call_kwargs["json"]
    assert "format" in payload
    assert payload["format"] == STRUCTURED_CATEGORY_SCHEMA
    assert payload["options"]["temperature"] == 0


# ==============================================================================
# 6. End-to-End Routing for Structured vs Unidentified Responses
# ==============================================================================

def test_e2e_structured_json_routes_to_kyc_with_suffix(test_dirs):
    """Verifies that a valid structured JSON response routes to KYC with appropriate suffix."""
    src = _create_sample_file(test_dirs["input"], "triveni_fixed.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": '{"category": "PAN CARD"}'}}

    with patch("requests.post", return_value=mock_resp):
        res = process_vlm_file(
            image_file=src,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

    assert res.status == "SUCCESS"
    assert res.category == "PAN CARD"
    assert res.routing_type == "KYC"
    assert res.final_filename == "triveni_fixed_PAN_CARD.jpg"
    assert res.destination_path == test_dirs["kyc"] / "triveni_fixed_PAN_CARD.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


def test_e2e_malformed_response_routes_to_unidentified_without_suffix(test_dirs):
    """Verifies that malformed or unidentifiable responses route to Unidentified_files with original filename."""
    src = _create_sample_file(test_dirs["input"], "unclassifiable.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": "This document contains elements of a bank passbook and a utility bill."}
    }

    with patch("requests.post", return_value=mock_resp):
        res = process_vlm_file(
            image_file=src,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

    assert res.status == "UNIDENTIFIED"
    assert res.category is None
    assert res.routing_type == "UNIDENTIFIED"
    assert res.final_filename == "unclassifiable.jpg"  # Original name, no suffix
    assert res.destination_path == test_dirs["unidentified"] / "unclassifiable.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()
