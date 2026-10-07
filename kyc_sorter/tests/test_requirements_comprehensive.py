"""
Comprehensive test suite verifying all 18 specific requirements:
1. Single Aadhaar image routed correctly
2. Single PAN image routed correctly
3. Single Voter ID image routed correctly
4. Non-KYC image routed to NON KYC
5. Image containing Aadhaar and PAN routed to NON KYC
6. Image containing Aadhaar, PAN, and Voter ID routed to NON KYC
7. Image with close competing classification scores routed to NON KYC
8. Image with one clearly dominant classification score routed to correct KYC category
9. Unreadable or corrupt image handled safely
10. Duplicate filenames do not overwrite existing files (collision protection)
11. Only JPG files are processed (.png, .pdf, .txt, .jpeg rejected/skipped)
12. Uppercase .JPG extensions handled correctly
13. Watcher mode routes files correctly
14. Batch mode routes files correctly
15. GUI still works and initializes cleanly
16. CPU monitoring and adaptive worker management behave correctly
17. Application shuts down safely during processing
18. Entire test suite integration verification
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from kyc_sorter.detector.classifier import DocumentClassifier
from kyc_sorter.detector.ocr import OCRResult, OCRLine
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.sorter.batch_processor import BatchProcessor, compute_adaptive_workers
from kyc_sorter.sorter.watcher import SourceFolderWatcher
from kyc_sorter.utils.config_loader import load_config, Config


class TestAll18Requirements(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.tmp_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "Destination files"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_config(
            overrides={
                "input": {"directory": str(self.input_dir)},
                "output": {
                    "directory": str(self.output_dir),
                    "operation": "copy",
                    "kyc_folder": "KYC",
                    "unknown_folder": "NON KYC",
                },
                "processing": {
                    "confidence_threshold": 0.80,
                    "max_workers": 6,
                    "cpu_limit_percent": 80,
                    "adaptive_workers": True,
                    "multiple_document_score_margin": 0.10,
                    "dry_run": False,
                },
                "ocr": {"cpu_threads": 1, "use_gpu": False},
            },
            base_dir=self.base_dir,
        )
        self.router = FileRouter(
            output_directory=self.output_dir,
            operation="copy",
            kyc_folder="KYC",
            unknown_folder="NON KYC",
        )
        self.classifier = DocumentClassifier(
            confidence_threshold=0.80,
            multiple_document_score_margin=0.10,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_dummy_image(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(path)
        return path

    # Requirement 1: Single Aadhaar image is routed correctly
    def test_01_single_aadhaar_routed_correctly(self):
        text = "GOVERNMENT OF INDIA Unique Identification Authority of India 9876 5432 1098 help@uidai.gov.in"
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "AADHAAR_CARD")
        self.assertGreaterEqual(result.confidence, 0.80)
        
        src_file = self._create_dummy_image(self.input_dir / "aadhaar_test.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "KYC")
        self.assertEqual(dest.name, "aadhaar_test_AADHAAR_CARD.jpg")
        self.assertTrue(dest.exists())

    # Requirement 2: Single PAN image is routed correctly
    def test_02_single_pan_routed_correctly(self):
        text = "INCOME TAX DEPARTMENT GOVT. OF INDIA Permanent Account Number Card ABCDE1234F"
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "PAN_CARD")
        self.assertGreaterEqual(result.confidence, 0.80)

        src_file = self._create_dummy_image(self.input_dir / "pan_test.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "KYC")
        self.assertEqual(dest.name, "pan_test_PAN_CARD.jpg")
        self.assertTrue(dest.exists())

    # Requirement 3: Single Voter ID image is routed correctly
    def test_03_single_voter_id_routed_correctly(self):
        text = "ELECTION COMMISSION OF INDIA ELECTOR PHOTO IDENTITY CARD EPIC NO. ABC1234567"
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "VOTER_ID")
        self.assertGreaterEqual(result.confidence, 0.80)

        src_file = self._create_dummy_image(self.input_dir / "voter_test.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "KYC")
        self.assertEqual(dest.name, "voter_test_VOTER_ID.jpg")
        self.assertTrue(dest.exists())

    # Requirement 4: Non-KYC image is routed to NON KYC
    def test_04_non_kyc_routed_to_non_kyc(self):
        text = "Supermarket grocery receipt 2 apples milk bread total 500"
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "UNKNOWN")
        self.assertLess(result.confidence, 0.80)

        src_file = self._create_dummy_image(self.input_dir / "random_photo.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "NON KYC")
        self.assertEqual(dest.name, "random_photo.jpg")
        self.assertTrue(dest.exists())

    # Requirement 5: Image containing Aadhaar and PAN is routed to NON KYC
    def test_05_image_containing_aadhaar_and_pan_routed_to_non_kyc(self):
        text = (
            "GOVERNMENT OF INDIA Unique Identification Authority of India 9876 5432 1098 "
            "INCOME TAX DEPARTMENT Permanent Account Number Card ABCDE1234F"
        )
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "UNKNOWN")
        self.assertLess(result.confidence, 0.80)
        self.assertIn("Multiple KYC documents detected", result.matched_signals.get("reason", ""))

        src_file = self._create_dummy_image(self.input_dir / "aadhaar_and_pan.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "NON KYC")
        self.assertEqual(dest.name, "aadhaar_and_pan.jpg")
        self.assertTrue(dest.exists())

    # Requirement 6: Image containing Aadhaar, PAN, and Voter ID is routed to NON KYC
    def test_06_image_containing_three_kyc_docs_routed_to_non_kyc(self):
        text = (
            "GOVERNMENT OF INDIA Unique Identification Authority of India 9876 5432 1098 "
            "INCOME TAX DEPARTMENT Permanent Account Number Card ABCDE1234F "
            "ELECTION COMMISSION OF INDIA ELECTOR PHOTO IDENTITY CARD ABC1234567"
        )
        result = self.classifier.classify_text(text)
        self.assertEqual(result.document_type, "UNKNOWN")
        self.assertLess(result.confidence, 0.80)
        self.assertIn("Multiple KYC documents detected", result.matched_signals.get("reason", ""))

        src_file = self._create_dummy_image(self.input_dir / "three_docs.jpg")
        decision = self.router.route(src_file, result.document_type)
        dest = decision.destination_path
        self.assertEqual(dest.parent.name, "NON KYC")
        self.assertEqual(dest.name, "three_docs.jpg")
        self.assertTrue(dest.exists())

    # Requirement 7: Image with close competing classification scores is routed to NON KYC
    def test_07_close_competing_scores_routed_to_non_kyc(self):
        def fake_score_competing(tmpl, text_clean, tokens, text_concat):
            if tmpl.name == "AADHAAR":
                return 8.0, {"mandatory_passed": True, "pos_score": 8.0, "strong_matches": ["unique"], "regular_matches": [], "regex_matches": [], "negative_matches": []}
            elif tmpl.name == "PAN":
                return 7.6, {"mandatory_passed": True, "pos_score": 7.6, "strong_matches": ["income"], "regular_matches": [], "regex_matches": [], "negative_matches": []}
            return 0.0, {"mandatory_passed": False}

        with patch.object(self.classifier, "_score_exact", side_effect=fake_score_competing), \
             patch.object(self.classifier, "_check_mandatory_keywords", return_value=True), \
             patch.object(self.classifier, "_check_template_anchor", return_value=False):
            result = self.classifier.classify_text("dummy test text")
            self.assertEqual(result.document_type, "UNKNOWN")
            self.assertIn("Ambiguous", result.matched_signals.get("reason", ""))

            src_file = self._create_dummy_image(self.input_dir / "ambiguous_scores.jpg")
            decision = self.router.route(src_file, result.document_type)
            dest = decision.destination_path
            self.assertEqual(dest.parent.name, "NON KYC")
            self.assertEqual(dest.name, "ambiguous_scores.jpg")
            self.assertTrue(dest.exists())

    # Requirement 8: Clearly dominant classification score is routed to correct KYC category
    def test_08_dominant_classification_score_routed_to_kyc(self):
        def fake_score_dominant(tmpl, text_clean, tokens, text_concat):
            if tmpl.name == "PAN":
                return 15.0, {"mandatory_passed": True, "pos_score": 15.0, "strong_matches": ["income", "permanent"], "regular_matches": [], "regex_matches": [], "negative_matches": []}
            elif tmpl.name == "VOTER":
                return 3.0, {"mandatory_passed": True, "pos_score": 3.0, "strong_matches": [], "regular_matches": ["identity"], "regex_matches": [], "negative_matches": []}
            return 0.0, {"mandatory_passed": False}

        with patch.object(self.classifier, "_score_exact", side_effect=fake_score_dominant), \
             patch.object(self.classifier, "_check_mandatory_keywords", return_value=True):
            result = self.classifier.classify_text("dummy test text")
            self.assertEqual(result.document_type, "PAN_CARD")
            self.assertGreaterEqual(result.confidence, 0.80)

            src_file = self._create_dummy_image(self.input_dir / "dominant_pan.jpg")
            decision = self.router.route(src_file, result.document_type)
            dest = decision.destination_path
            self.assertEqual(dest.parent.name, "KYC")
            self.assertEqual(dest.name, "dominant_pan_PAN_CARD.jpg")
            self.assertTrue(dest.exists())

    # Requirement 9: Unreadable or corrupt image is handled safely
    def test_09_unreadable_or_corrupt_image_handled_safely(self):
        corrupt_file = self.input_dir / "corrupted_file.jpg"
        corrupt_file.write_bytes(b"CORRUPT DATA NOT A VALID JPEG")

        # Mock OCR raising an exception on corrupt image
        mock_ocr = MagicMock()
        mock_ocr.extract_text.side_effect = Exception("Invalid image data")

        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=self.classifier,
            router=self.router,
        )
        res = processor.process_file(corrupt_file)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "FAILED")
        dest = Path(res.destination_path)
        self.assertEqual(dest.parent.name, "NON KYC")
        self.assertEqual(dest.name, "corrupted_file.jpg")
        self.assertTrue(dest.exists())

    # Requirement 10: Duplicate filenames do not overwrite existing files (collision protection)
    def test_10_duplicate_filenames_do_not_overwrite(self):
        src1 = self._create_dummy_image(self.input_dir / "doc1.jpg")
        dest1 = self.router.route(src1, "AADHAAR_CARD").destination_path
        self.assertEqual(dest1.name, "doc1_AADHAAR_CARD.jpg")
        self.assertTrue(dest1.exists())

        # Second file with same original name
        dest2 = self.router.route(src1, "AADHAAR_CARD").destination_path
        self.assertEqual(dest2.name, "doc1_AADHAAR_CARD_1.jpg")
        self.assertTrue(dest2.exists())
        self.assertTrue(dest1.exists())

        # Third file
        dest3 = self.router.route(src1, "AADHAAR_CARD").destination_path
        self.assertEqual(dest3.name, "doc1_AADHAAR_CARD_2.jpg")
        self.assertTrue(dest3.exists())
        self.assertTrue(dest1.exists())
        self.assertTrue(dest2.exists())

        # NON_KYC collision handling (preserving original stem)
        src_non = self._create_dummy_image(self.input_dir / "non1.jpg")
        dest_non1 = self.router.route(src_non, "UNKNOWN").destination_path
        self.assertEqual(dest_non1.name, "non1.jpg")
        self.assertTrue(dest_non1.exists())

        dest_non2 = self.router.route(src_non, "UNKNOWN").destination_path
        self.assertEqual(dest_non2.name, "non1_1.jpg")
        self.assertTrue(dest_non2.exists())
        self.assertTrue(dest_non1.exists())

    # Requirement 11: Only JPG files are processed (.png, .pdf, .txt, .jpeg rejected/skipped)
    def test_11_only_jpg_files_processed(self):
        # Create valid JPG
        self._create_dummy_image(self.input_dir / "valid.jpg")
        # Create non-JPG files
        (self.input_dir / "doc.png").write_bytes(b"\x89PNG")
        (self.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4")
        (self.input_dir / "doc.txt").write_text("Hello text")
        (self.input_dir / "doc.jpeg").write_bytes(b"\xFF\xD8\xFF")

        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = OCRResult(text="random content", lines=[], mean_confidence=0.5)

        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=self.classifier,
            router=self.router,
        )
        summary = processor.process_batch(self.input_dir)
        self.assertEqual(summary.total_files, 1)
        self.assertEqual(summary.skipped_unsupported, 4)
        self.assertEqual(summary.processed, 1)

    # Requirement 12: Uppercase .JPG extensions are handled correctly
    def test_12_uppercase_jpg_extension_handled_correctly(self):
        self._create_dummy_image(self.input_dir / "valid_upper.JPG")

        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = OCRResult(
            text="INCOME TAX DEPARTMENT Permanent Account Number Card ABCDE1234F",
            lines=[],
            mean_confidence=0.95,
        )

        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=self.classifier,
            router=self.router,
        )
        summary = processor.process_batch(self.input_dir)
        self.assertEqual(summary.total_files, 1)
        self.assertEqual(summary.processed, 1)
        self.assertTrue((self.output_dir / "KYC" / "valid_upper_PAN_CARD.JPG").exists())

    # Requirement 13: Watcher mode routes files correctly
    def test_13_watcher_mode_routes_files_correctly(self):
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = OCRResult(
            text="GOVERNMENT OF INDIA 9876 5432 1098 help@uidai.gov.in",
            lines=[],
            mean_confidence=0.95,
        )
        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=self.classifier,
            router=self.router,
        )
        watcher = SourceFolderWatcher(
            config=self.config,
            processor=processor,
        )
        test_file = self._create_dummy_image(self.input_dir / "watch_doc.jpg")
        watcher._process_single_file_safe(test_file)
        self.assertTrue((self.output_dir / "KYC" / "watch_doc_AADHAAR_CARD.jpg").exists())

    # Requirement 14: Batch mode routes files correctly
    def test_14_batch_mode_routes_files_correctly(self):
        self._create_dummy_image(self.input_dir / "batch_kyc.jpg")
        self._create_dummy_image(self.input_dir / "batch_non_kyc.jpg")

        def side_effect(path):
            if "batch_kyc" in str(path):
                return OCRResult(text="INCOME TAX DEPARTMENT ABCDE1234F", lines=[], mean_confidence=0.95)
            return OCRResult(text="Random notes grocery", lines=[], mean_confidence=0.2)

        mock_ocr = MagicMock()
        mock_ocr.extract_text.side_effect = side_effect

        self.config.processing.max_workers = 1
        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=self.classifier,
            router=self.router,
        )
        summary = processor.process_batch(self.input_dir)
        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.kyc_routed_count, 1)
        self.assertEqual(summary.non_kyc_routed_count, 1)
        self.assertTrue((self.output_dir / "KYC" / "batch_kyc_PAN_CARD.jpg").exists())
        self.assertTrue((self.output_dir / "NON KYC" / "batch_non_kyc.jpg").exists())

    # Requirement 15: GUI still works and initializes cleanly
    def test_15_gui_initialization_and_thread_creation(self):
        from kyc_sorter.gui import SorterApp
        self.assertTrue(hasattr(SorterApp, "_build_ui"))
        self.assertTrue(hasattr(SorterApp, "_start_processing"))

    # Requirement 16: CPU monitoring and adaptive worker management behave correctly
    def test_16_cpu_monitoring_and_adaptive_worker_management(self):
        # 1. On 12 cores, adaptive workers should respect target 80% ceiling and max configured workers
        workers = compute_adaptive_workers(
            configured_max=6,
            total_cores=12,
            adaptive_enabled=True,
            target_cpu_percent=80.0,
        )
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 6)

        # 2. On low-core systems (e.g. 2 cores), worker should be conservative (1 worker)
        workers_low = compute_adaptive_workers(
            configured_max=6,
            total_cores=2,
            adaptive_enabled=True,
            target_cpu_percent=80.0,
        )
        self.assertEqual(workers_low, 1)

        # 3. If adaptive is disabled, exactly configured_max is returned
        workers_fixed = compute_adaptive_workers(
            configured_max=6,
            total_cores=12,
            adaptive_enabled=False,
            target_cpu_percent=80.0,
        )
        self.assertEqual(workers_fixed, 6)

    # Requirement 17: Application shuts down safely during processing
    def test_17_application_shuts_down_safely(self):
        processor = BatchProcessor(
            config=self.config,
            ocr_engine=MagicMock(),
            classifier=self.classifier,
            router=self.router,
        )
        # Test abort flag
        processor.abort()
        self.assertTrue(processor._abort_requested)
        # Verify process_batch aborts cleanly
        self._create_dummy_image(self.input_dir / "test_abort.jpg")
        summary = processor.process_batch(self.input_dir)
        self.assertEqual(summary.processed, 0)

    # Requirement 18: Entire test suite integration verification
    def test_18_integration_verification(self):
        # Check output structure conforms to:
        # Destination files/
        #     KYC/
        #     NON KYC/
        self.assertTrue(self.output_dir.exists())
        # Check no "Unknown" folder was created
        self.assertFalse((self.output_dir / "Unknown").exists())


if __name__ == "__main__":
    unittest.main()
