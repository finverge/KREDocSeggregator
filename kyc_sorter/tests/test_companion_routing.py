"""
Focused Unit and Integration Tests for Companion File Handling and Routing.

Validates Phase 2 specifications:
1. Companion file generation (.txt and .json) for both KYC and NON_KYC.
2. Companion file reuse when .txt or .json already exists in source directory.
3. Synchronized filenames and collision suffix (_1, _2) across image, .txt, and .json.
4. JSON schema completeness and valid JSON syntax.
5. Handling missing/empty OCR text without rerun or crash.
6. Error handling for failed file movements without silent file loss.
7. NON_KYC audit trail preserving rejection category and rejection reason.
8. Dry run safety.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from kyc_sorter.detector.classifier import ClassificationResult, DocumentClassifier
from kyc_sorter.detector.ocr import OCREngine, OCRResult, OCRLine
from kyc_sorter.sorter.batch_processor import BatchProcessor, ProcessedItem
from kyc_sorter.sorter.file_router import FileRouter, RouteDecision
from kyc_sorter.utils.config_loader import Config, InputConfig, OutputConfig, ProcessingConfig


class TestCompanionRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "Destination files"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.router = FileRouter(
            output_directory=self.output_dir,
            kyc_folder="KYC",
            unknown_folder="NON KYC",
            operation="copy",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_kyc_routing_generates_companion_files(self):
        """Routing a KYC document generates matching .txt and .json files with identical stem."""
        src_img = self.input_dir / "aadhaar_doc.jpg"
        src_img.write_text("binary image data")

        decision = self.router.route(
            source_file=src_img,
            document_type="AADHAAR_CARD",
            ocr_text="GOVERNMENT OF INDIA 1234 5678 9012",
            confidence=0.98,
            classification_status="IDENTIFIED",
            rejection_category="KYC",
        )

        self.assertTrue(decision.success)
        self.assertEqual(decision.folder_name, "AADHAAR CARD")
        self.assertEqual(decision.destination_path.parent, self.output_dir / "KYC")

        # Verify synchronized naming
        expected_stem = "aadhaar_doc_AADHAAR_CARD"
        self.assertEqual(decision.destination_path.name, f"{expected_stem}.jpg")
        self.assertIsNotNone(decision.destination_txt_path)
        self.assertEqual(decision.destination_txt_path.name, f"{expected_stem}.txt")
        self.assertIsNotNone(decision.destination_json_path)
        self.assertEqual(decision.destination_json_path.name, f"{expected_stem}.json")

        # Verify files exist on disk
        self.assertTrue(decision.destination_path.is_file())
        self.assertTrue(decision.destination_txt_path.is_file())
        self.assertTrue(decision.destination_json_path.is_file())

        # Verify .txt content
        self.assertEqual(decision.destination_txt_path.read_text(encoding="utf-8"), "GOVERNMENT OF INDIA 1234 5678 9012")

        # Verify JSON schema and data
        metadata = json.loads(decision.destination_json_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_filename"], "aadhaar_doc.jpg")
        self.assertEqual(metadata["final_output_filename"], f"{expected_stem}.jpg")
        self.assertEqual(metadata["document_type"], "AADHAAR_CARD")
        self.assertEqual(metadata["confidence"], 0.98)
        self.assertEqual(metadata["classification_status"], "IDENTIFIED")
        self.assertEqual(metadata["rejection_category"], "KYC")
        self.assertIsNone(metadata["rejection_reason"])
        self.assertEqual(metadata["ocr_text_file"], f"{expected_stem}.txt")
        self.assertEqual(metadata["ocr_text"], "GOVERNMENT OF INDIA 1234 5678 9012")
        self.assertIn("processing_timestamp", metadata)

    def test_non_kyc_routing_includes_rejection_reason_and_audit_data(self):
        """Routing a NON_KYC document generates companion files in NON KYC with rejection reason."""
        src_img = self.input_dir / "receipt.jpg"
        src_img.write_text("grocery receipt")

        decision = self.router.route(
            source_file=src_img,
            document_type="UNKNOWN",
            ocr_text="Store receipt total Rs 500",
            confidence=0.25,
            classification_status="UNKNOWN",
            rejection_category="LOW_CONFIDENCE",
            rejection_reason="Low confidence (0.25 < 0.80)",
        )

        self.assertTrue(decision.success)
        self.assertEqual(decision.destination_path.parent, self.output_dir / "NON KYC")

        expected_stem = "receipt"
        self.assertEqual(decision.destination_path.name, f"{expected_stem}.jpg")
        self.assertEqual(decision.destination_txt_path.name, f"{expected_stem}.txt")
        self.assertEqual(decision.destination_json_path.name, f"{expected_stem}.json")

        metadata = json.loads(decision.destination_json_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["document_type"], "UNKNOWN")
        self.assertEqual(metadata["classification_status"], "UNKNOWN")
        self.assertEqual(metadata["rejection_category"], "LOW_CONFIDENCE")
        self.assertEqual(metadata["rejection_reason"], "Low confidence (0.25 < 0.80)")
        self.assertEqual(metadata["ocr_text_file"], f"{expected_stem}.txt")
        self.assertEqual(metadata["ocr_text"], "Store receipt total Rs 500")

    def test_companion_reuse_existing_source_txt_and_json(self):
        """Existing matching .txt and .json in source are reused without re-running OCR."""
        src_img = self.input_dir / "pan_doc.jpg"
        src_txt = self.input_dir / "pan_doc.txt"
        src_json = self.input_dir / "pan_doc.json"

        src_img.write_text("pan image data")
        src_txt.write_text("EXISTING OCR TEXT IN SOURCE FILE")
        src_json.write_text(json.dumps({"upstream_key": "existing_backend_value", "extra_field": 123}))

        # Route without supplying in-memory ocr_text; router must read from existing src_txt
        decision = self.router.route(
            source_file=src_img,
            document_type="PAN_CARD",
            confidence=0.92,
        )

        self.assertTrue(decision.success)
        self.assertEqual(decision.txt_action, "REUSED_SOURCE_TXT")
        self.assertEqual(decision.json_action, "REUSED_AND_UPDATED_SOURCE_JSON")

        # Destination .txt must contain existing source text
        self.assertEqual(decision.destination_txt_path.read_text(encoding="utf-8"), "EXISTING OCR TEXT IN SOURCE FILE")

        # Destination .json must merge existing source JSON while updating router audit fields
        metadata = json.loads(decision.destination_json_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["upstream_key"], "existing_backend_value")
        self.assertEqual(metadata["extra_field"], 123)
        self.assertEqual(metadata["document_type"], "PAN_CARD")
        self.assertEqual(metadata["ocr_text"], "EXISTING OCR TEXT IN SOURCE FILE")

    def test_companion_reuse_in_move_mode_cleans_source_files(self):
        """In move mode, existing companion files in source directory are moved cleanly."""
        move_router = FileRouter(
            output_directory=self.output_dir,
            operation="move",
        )
        src_img = self.input_dir / "voter.png"
        src_txt = self.input_dir / "voter.txt"
        src_json = self.input_dir / "voter.json"

        src_img.write_text("voter image")
        src_txt.write_text("ELECTION COMMISSION OF INDIA")
        src_json.write_text(json.dumps({"initial": "data"}))

        decision = move_router.route(
            source_file=src_img,
            document_type="VOTER_ID",
            confidence=0.96,
        )

        self.assertTrue(decision.success)
        self.assertFalse(src_img.exists(), "Source image must be moved")
        self.assertFalse(src_txt.exists(), "Source text must be moved")
        self.assertFalse(src_json.exists(), "Source json must be cleaned/moved")
        self.assertTrue(decision.destination_path.exists())
        self.assertTrue(decision.destination_txt_path.exists())
        self.assertTrue(decision.destination_json_path.exists())

    def test_synchronized_filenames_with_collision_counter(self):
        """When destination files collide, image, .txt, and .json receive synchronized _N counter."""
        kyc_dir = self.output_dir / "KYC"
        kyc_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed existing files
        (kyc_dir / "doc_AADHAAR_CARD.jpg").write_text("first image")
        (kyc_dir / "doc_AADHAAR_CARD.txt").write_text("first text")
        (kyc_dir / "doc_AADHAAR_CARD.json").write_text("{}")

        src_img = self.input_dir / "doc.jpg"
        src_img.write_text("second image")

        decision = self.router.route(
            source_file=src_img,
            document_type="AADHAAR_CARD",
            ocr_text="second text",
            confidence=0.95,
        )

        self.assertTrue(decision.success)
        self.assertEqual(decision.destination_path.name, "doc_AADHAAR_CARD_1.jpg")
        self.assertEqual(decision.destination_txt_path.name, "doc_AADHAAR_CARD_1.txt")
        self.assertEqual(decision.destination_json_path.name, "doc_AADHAAR_CARD_1.json")

        # Now simulate a dangling collision where only .json_1 exists; next route must advance to _2
        (kyc_dir / "doc_AADHAAR_CARD_1.json").write_text("{}")
        src_img3 = self.input_dir / "doc.jpg"

        decision2 = self.router.route(
            source_file=src_img3,
            document_type="AADHAAR_CARD",
            ocr_text="third text",
            confidence=0.91,
        )
        self.assertEqual(decision2.destination_path.name, "doc_AADHAAR_CARD_2.jpg")
        self.assertEqual(decision2.destination_txt_path.name, "doc_AADHAAR_CARD_2.txt")
        self.assertEqual(decision2.destination_json_path.name, "doc_AADHAAR_CARD_2.json")

    def test_non_kyc_collision_handling_keeps_companion_stems_synchronized(self):
        """NON_KYC collision handling appends _1, _2 to original stem for image, txt, and json."""
        src_img = self.input_dir / "invoice.jpg"
        src_img.write_text("invoice image")

        d1 = self.router.route(src_img, "UNKNOWN")
        self.assertEqual(d1.destination_path.name, "invoice.jpg")
        self.assertEqual(d1.destination_txt_path.name, "invoice.txt")
        self.assertEqual(d1.destination_json_path.name, "invoice.json")

        d2 = self.router.route(src_img, "UNKNOWN")
        self.assertEqual(d2.destination_path.name, "invoice_1.jpg")
        self.assertEqual(d2.destination_txt_path.name, "invoice_1.txt")
        self.assertEqual(d2.destination_json_path.name, "invoice_1.json")

    def test_missing_ocr_outputs_generates_clean_empty_companions_without_error(self):
        """If OCR yields empty text and no companion files exist, router succeeds without error."""
        src_img = self.input_dir / "empty_doc.jpg"
        src_img.write_text("blank")

        decision = self.router.route(
            source_file=src_img,
            document_type="UNKNOWN",
            ocr_text="",
            confidence=0.0,
            rejection_reason="No text detected",
        )

        self.assertTrue(decision.success)
        self.assertTrue(decision.destination_txt_path.is_file())
        self.assertEqual(decision.destination_txt_path.read_text(encoding="utf-8"), "")

        meta = json.loads(decision.destination_json_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["ocr_text"], "")
        self.assertEqual(meta["rejection_reason"], "No text detected")

    def test_dry_run_does_not_modify_filesystem(self):
        """In dry run mode, no files are copied, moved, or generated."""
        dry_router = FileRouter(
            output_directory=self.output_dir,
            dry_run=True,
        )
        src_img = self.input_dir / "sample.jpg"
        src_img.write_text("sample")

        decision = dry_router.route(
            source_file=src_img,
            document_type="AADHAAR_CARD",
            ocr_text="Sample text",
        )

        self.assertTrue(decision.success)
        self.assertTrue(decision.dry_run)
        self.assertTrue(src_img.exists())
        self.assertFalse(decision.destination_path.exists())
        self.assertFalse(decision.destination_txt_path.exists())
        self.assertFalse(decision.destination_json_path.exists())

    def test_failed_file_movement_handled_gracefully(self):
        """File operation errors are captured in RouteDecision without silent file loss."""
        src_img = self.input_dir / "fail_doc.jpg"
        src_img.write_text("data")

        # Mock shutil.copy2 to raise PermissionError
        with patch("shutil.copy2", side_effect=PermissionError("Permission denied")):
            decision = self.router.route(
                source_file=src_img,
                document_type="AADHAAR_CARD",
            )
            self.assertFalse(decision.success)
            self.assertIn("Permission denied", decision.error)
            self.assertTrue(src_img.exists(), "Source file must not be lost")

    def test_batch_processor_e2e_mocked_companion_flow(self):
        """BatchProcessor coordinates OCR text, rejection reason, and companion generation."""
        # Setup config
        config = Config(
            input=InputConfig(directory=str(self.input_dir)),
            output=OutputConfig(directory=str(self.output_dir), operation="copy", generate_companion_files=True),
            processing=ProcessingConfig(confidence_threshold=0.80, dry_run=False, max_workers=1, adaptive_workers=False),
        )

        # Create two sample image files
        doc1 = self.input_dir / "img1.jpg"
        doc1.write_text("fake img 1")
        doc2 = self.input_dir / "img2.jpg"
        doc2.write_text("fake img 2")

        # Mock OCR engine and classifier (single-pass)
        mock_ocr = MagicMock(spec=OCREngine)
        mock_classifier = MagicMock(spec=DocumentClassifier)

        def mock_extract(file_path):
            p = Path(file_path)
            if "img1" in p.name:
                return OCRResult(
                    text="INCOME TAX DEPARTMENT GOVT OF INDIA ABCDE1234F",
                    lines=[OCRLine(text="INCOME TAX DEPARTMENT GOVT OF INDIA ABCDE1234F", confidence=0.95)],
                    mean_confidence=0.95,
                    success=True,
                )
            else:
                return OCRResult(
                    text="Multiple documents detected on single page",
                    lines=[OCRLine(text="Multiple documents detected on single page", confidence=0.30)],
                    mean_confidence=0.30,
                    success=True,
                )

        def mock_classify(text, lines=None):
            if "INCOME TAX" in text:
                return ClassificationResult(
                    document_type="PAN_CARD",
                    confidence=0.95,
                    status="IDENTIFIED",
                    matched_template="PAN_CARD",
                    rejection_category="KYC",
                )
            else:
                return ClassificationResult(
                    document_type="UNKNOWN",
                    confidence=0.30,
                    status="UNKNOWN",
                    matched_template=None,
                    rejection_category="MULTI_DOC",
                    matched_signals={"reason": "Aadhaar and PAN detected simultaneously"},
                )

        mock_ocr.extract_text.side_effect = mock_extract
        mock_classifier.classify_text.side_effect = mock_classify

        processor = BatchProcessor(
            config=config,
            ocr_engine=mock_ocr,
            classifier=mock_classifier,
            router=self.router,
        )

        summary = processor.process_batch(source_dir=self.input_dir)

        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.failed, 0)

        # Check KYC destination
        kyc_img = self.output_dir / "KYC" / "img1_PAN_CARD.jpg"
        kyc_txt = self.output_dir / "KYC" / "img1_PAN_CARD.txt"
        kyc_json = self.output_dir / "KYC" / "img1_PAN_CARD.json"
        self.assertTrue(kyc_img.is_file())
        self.assertTrue(kyc_txt.is_file())
        self.assertTrue(kyc_json.is_file())
        self.assertIn("INCOME TAX", kyc_txt.read_text(encoding="utf-8"))
        kyc_meta = json.loads(kyc_json.read_text(encoding="utf-8"))
        self.assertEqual(kyc_meta["document_type"], "PAN_CARD")
        self.assertEqual(kyc_meta["classification_status"], "IDENTIFIED")

        # Check NON KYC destination (preserves original filename)
        non_kyc_img = self.output_dir / "NON KYC" / "img2.jpg"
        non_kyc_txt = self.output_dir / "NON KYC" / "img2.txt"
        non_kyc_json = self.output_dir / "NON KYC" / "img2.json"
        self.assertTrue(non_kyc_img.is_file())
        self.assertTrue(non_kyc_txt.is_file())
        self.assertTrue(non_kyc_json.is_file())
        non_kyc_meta = json.loads(non_kyc_json.read_text(encoding="utf-8"))
        self.assertEqual(non_kyc_meta["document_type"], "UNKNOWN")
        self.assertEqual(non_kyc_meta["rejection_category"], "MULTI_DOC")
        self.assertIn("Multi-document detected", non_kyc_meta["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
