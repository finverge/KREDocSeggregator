"""
Automated unit and integration tests for the Multiple-Document / Multiple-Image Detection Rule.
Verifies that any image containing more than one distinct visible document/item
is routed to NON KYC, overriding any single-document category or confidence score.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kyc_sorter.detector.multi_doc_prompt import MULTI_DOC_PROMPT, parse_multi_document_response
from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.sorter.journal import ProcessingJournal
from kyc_sorter.sorter.vlm_router import (
    detect_multiple_documents,
    process_vlm_file,
    route_document,
)
from kyc_sorter.utils.config_loader import load_config


@pytest.fixture
def test_dirs(tmp_path):
    """Isolated directory structure for multi-document testing."""
    in_dir = tmp_path / "Source Files" / "Input"
    in_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "Destination files"
    dest_root.mkdir(parents=True, exist_ok=True)
    journal_path = dest_root / "kyc_journal.db"

    cfg = load_config(
        overrides={
            "input": {"directory": str(in_dir)},
            "output": {"directory": str(dest_root), "operation": "move"},
            "processing": {
                "dry_run": False,
                "max_workers": 2,
                "journal_path": str(journal_path),
            },
        }
    )
    return {
        "input": in_dir,
        "dest_root": dest_root,
        "journal_path": journal_path,
        "config": cfg,
    }


def _create_sample_file(dir_path: Path, filename: str, size: int = 1024) -> Path:
    p = dir_path / filename
    p.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * (size - 10))
    return p


# ==============================================================================
# 1. Multi-Doc Parser Unit Tests
# ==============================================================================
class TestMultiDocParser:
    def test_parse_multiple_documents_variants(self):
        assert parse_multi_document_response("MULTIPLE_DOCUMENTS") is True
        assert parse_multi_document_response("MULTIPLE DOCUMENTS") is True
        assert parse_multi_document_response("MULTIPLE_DOCUMENT") is True
        assert parse_multi_document_response("There are MULTIPLE documents in this image.") is True
        assert parse_multi_document_response("`MULTIPLE_DOCUMENTS`") is True

    def test_parse_single_document_variants(self):
        assert parse_multi_document_response("SINGLE_DOCUMENT") is False
        assert parse_multi_document_response("SINGLE DOCUMENT") is False
        assert parse_multi_document_response("SINGLE_DOCUMENT.") is False
        assert parse_multi_document_response("`SINGLE_DOCUMENT`") is False

    def test_parse_single_kyc_category_treated_as_single(self):
        # When mock returns a category name, it represents a single document
        assert parse_multi_document_response("PAN CARD") is False
        assert parse_multi_document_response("AADHAAR CARD") is False
        assert parse_multi_document_response("VOTER ID") is False

    def test_parse_unrecognized_returns_none(self):
        assert parse_multi_document_response("") is None
        assert parse_multi_document_response(None) is None
        assert parse_multi_document_response("I cannot decide what this is.") is None


# ==============================================================================
# 2. Mandatory Multiple-Document Detection Tests
# ==============================================================================
class TestMultipleDocumentRoutingRules:
    def test_01_aadhaar_plus_pan_routes_to_non_kyc(self, test_dirs):
        """Case 1: Aadhaar + PAN in one JPG must route to NON KYC with original filename."""
        src = _create_sample_file(test_dirs["input"], "aadhaar_and_pan.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

        with patch("requests.post", return_value=mock_resp):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.status == "SUCCESS"
        assert res.routing_type == "NON_KYC"
        assert res.is_multiple_document is True
        assert res.final_filename == "aadhaar_and_pan.jpg"  # Original filename unchanged
        assert res.destination_folder == test_dirs["dest_root"] / "NON KYC"
        assert res.destination_path.is_file()
        assert not src.exists()

    def test_02_aadhaar_plus_voter_id_routes_to_non_kyc(self, test_dirs):
        """Case 2: Aadhaar + Voter ID in one JPG must route to NON KYC."""
        src = _create_sample_file(test_dirs["input"], "aadhaar_voter.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

        with patch("requests.post", return_value=mock_resp):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "NON_KYC"
        assert res.is_multiple_document is True
        assert res.destination_folder == test_dirs["dest_root"] / "NON KYC"

    def test_03_kyc_plus_non_kyc_routes_to_non_kyc(self, test_dirs):
        """Case 3: Aadhaar + electricity bill / non-KYC content in one JPG must route to NON KYC."""
        src = _create_sample_file(test_dirs["input"], "pan_plus_bill.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

        with patch("requests.post", return_value=mock_resp):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "NON_KYC"
        assert res.is_multiple_document is True
        assert res.destination_folder == test_dirs["dest_root"] / "NON KYC"

    def test_04_multiple_distinct_items_with_low_confidence(self, test_dirs):
        """Case 4: Multiple items route to NON KYC regardless of confidence score."""
        src = _create_sample_file(test_dirs["input"], "two_low_quality_cards.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

        with patch("requests.post", return_value=mock_resp):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "NON_KYC"
        assert res.destination_folder == test_dirs["dest_root"] / "NON KYC"
        assert res.status == "SUCCESS"

    def test_05_high_confidence_single_kyc_cannot_override_multi_doc(self, test_dirs):
        """
        Case 5: Positive multiple-item detection preempts single classification.
        Single classification is never invoked when multiple items are detected.
        """
        src = _create_sample_file(test_dirs["input"], "two_cards_one_clear_pan.jpg")

        # Even if classify_document would return confident PAN CARD,
        # detect_multiple_documents returns MULTIPLE_DOCUMENTS
        def fake_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            content = payload.get("messages", [{}])[0].get("content", "")
            resp = MagicMock()
            resp.status_code = 200
            if "visual document inspector" in content:
                resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}
            else:
                resp.json.return_value = {"message": {"content": "PAN CARD"}}
            return resp

        with patch("requests.post", side_effect=fake_post) as mock_p:
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "NON_KYC"
        assert res.is_multiple_document is True
        assert res.destination_folder == test_dirs["dest_root"] / "NON KYC"
        # Only 1 call was made to Ollama because multi-doc bypassed single classification
        assert mock_p.call_count == 1

    def test_06_single_aadhaar_remains_classified_normally(self, test_dirs):
        """Case 6: A single Aadhaar document routes to KYC/ with _AADHAAR_CARD suffix."""
        src = _create_sample_file(test_dirs["input"], "pure_aadhaar.jpg")

        def fake_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            content = payload.get("messages", [{}])[0].get("content", "")
            resp = MagicMock()
            resp.status_code = 200
            if "visual document inspector" in content:
                resp.json.return_value = {"message": {"content": "SINGLE_DOCUMENT"}}
            else:
                resp.json.return_value = {"message": {"content": "AADHAAR CARD"}}
            return resp

        with patch("requests.post", side_effect=fake_post):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.status == "SUCCESS"
        assert res.routing_type == "KYC"
        assert res.category == "AADHAAR CARD"
        assert res.is_multiple_document is False
        assert res.final_filename == "pure_aadhaar_AADHAAR_CARD.jpg"
        assert res.destination_folder == test_dirs["dest_root"] / "KYC"

    def test_07_single_pan_remains_classified_normally(self, test_dirs):
        """Case 7: A single PAN document routes to KYC/ with _PAN_CARD suffix."""
        src = _create_sample_file(test_dirs["input"], "pure_pan.jpg")

        def fake_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            content = payload.get("messages", [{}])[0].get("content", "")
            resp = MagicMock()
            resp.status_code = 200
            if "visual document inspector" in content:
                resp.json.return_value = {"message": {"content": "SINGLE_DOCUMENT"}}
            else:
                resp.json.return_value = {"message": {"content": "PAN CARD"}}
            return resp

        with patch("requests.post", side_effect=fake_post):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.status == "SUCCESS"
        assert res.routing_type == "KYC"
        assert res.category == "PAN CARD"
        assert res.is_multiple_document is False
        assert res.final_filename == "pure_pan_PAN_CARD.jpg"
        assert res.destination_folder == test_dirs["dest_root"] / "KYC"

    def test_08_single_document_with_multiple_text_fields_and_logo(self, test_dirs):
        """Case 8: Document with multiple fields/logos is recognized as SINGLE_DOCUMENT."""
        src = _create_sample_file(test_dirs["input"], "bank_statement_many_fields.jpg")

        def fake_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            content = payload.get("messages", [{}])[0].get("content", "")
            resp = MagicMock()
            resp.status_code = 200
            if "visual document inspector" in content:
                # Inspector correctly recognizes internal fields as a single document
                resp.json.return_value = {"message": {"content": "SINGLE_DOCUMENT"}}
            else:
                resp.json.return_value = {"message": {"content": "BANK STATEMENT"}}
            return resp

        with patch("requests.post", side_effect=fake_post):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "KYC"
        assert res.category == "BANK STATEMENT"
        assert res.is_multiple_document is False
        assert res.final_filename == "bank_statement_many_fields_BANK_STATEMENT.jpg"

    def test_09_unreliable_item_count_fallback_to_unidentified(self, test_dirs):
        """Case 9: Unreliable item count or network failure routes safely to Unidentified_files."""
        src = _create_sample_file(test_dirs["input"], "corrupted_card.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            res = process_vlm_file(
                image_file=src,
                destination_root=test_dirs["dest_root"],
            )

        assert res.routing_type == "UNIDENTIFIED"
        assert res.destination_folder == test_dirs["dest_root"] / "Unidentified_files"
        assert "HTTP error 500" in (res.error or "")

    def test_10_multi_doc_batch_processor_and_journal_integration(self, test_dirs):
        """
        Case 10: BatchProcessor tracks identified_multi_doc count and writes
        proper NON_KYC records to SQLite journal without collisions or corruption.
        """
        src1 = _create_sample_file(test_dirs["input"], "doc_multi_1.jpg")
        src2 = _create_sample_file(test_dirs["input"], "doc_multi_2.jpg")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

        processor = BatchProcessor(config=test_dirs["config"])

        with patch("requests.post", return_value=mock_resp):
            summary = processor.process_batch(source_dir=test_dirs["input"])

        assert summary.total_files == 2
        assert summary.identified_multi_doc == 2
        assert summary.counts_by_type.get("NON_KYC") == 2
        assert (test_dirs["dest_root"] / "NON KYC" / "doc_multi_1.jpg").is_file()
        assert (test_dirs["dest_root"] / "NON KYC" / "doc_multi_2.jpg").is_file()

        # Verify journal recorded them as completed NON_KYC
        journal = ProcessingJournal(db_path=test_dirs["journal_path"])
        rec1 = journal.reconcile_file_status(src1, expected_operation="move")
        rec2 = journal.reconcile_file_status(src2, expected_operation="move")
        assert rec1 is not None
        assert rec1[1] == "NON_KYC"
        assert rec2 is not None
        assert rec2[1] == "NON_KYC"
