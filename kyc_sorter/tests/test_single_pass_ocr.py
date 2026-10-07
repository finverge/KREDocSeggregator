"""
Tests verifying single-pass OCR execution and model configuration.
Guarantees:
1. Exactly ONE OCR invocation per image (no two-stage, fallback, or cascade pass).
2. Exactly matching Finverge backend model configuration (PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec).
3. Profiling output accurately reflects single-pass execution without obsolete Stage 1/2 labels.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from kyc_sorter.detector.classifier import DocumentClassifier
from kyc_sorter.detector.ocr import OCREngine, OCRResult
from kyc_sorter.sorter.batch_processor import BatchProcessor, DocumentProfileTiming
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.utils.config_loader import load_config


class TestSinglePassOCRPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.input_dir = self.tmp_path / "input"
        self.output_dir = self.tmp_path / "output"
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
                    "max_workers": 1,
                    "dry_run": False,
                },
                "ocr": {
                    "language": "en",
                    "use_gpu": False,
                    "cpu_threads": 3,
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                },
            }
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_sample_image(self, name: str = "test.jpg") -> Path:
        img_path = self.input_dir / name
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        img.save(img_path)
        return img_path

    def test_exact_ocr_model_configuration_matches_backend(self):
        """Verify OCREngine loads the exact single-pass models specified in backend config."""
        engine = OCREngine(
            language=self.config.ocr.language,
            use_gpu=self.config.ocr.use_gpu,
            cpu_threads=self.config.ocr.cpu_threads,
            det_model_name=self.config.ocr.text_detection_model_name,
            rec_model_name=self.config.ocr.text_recognition_model_name,
        )

        self.assertEqual(engine.det_model_name, "PP-OCRv5_mobile_det")
        self.assertEqual(engine.rec_model_name, "en_PP-OCRv5_mobile_rec")
        self.assertFalse(engine.use_gpu)
        self.assertEqual(engine.cpu_threads, 3)

    def test_single_ocr_invocation_for_kyc_document(self):
        """Verify that processing a valid KYC image calls OCR exactly once."""
        img_file = self._create_sample_image("pan.jpg")

        mock_ocr = MagicMock(spec=OCREngine)
        mock_ocr.rec_model_name = "en_PP-OCRv5_mobile_rec"
        mock_ocr.extract_text.return_value = OCRResult(
            text="INCOME TAX DEPARTMENT GOVT. OF INDIA Permanent Account Number Card ABCDE1234F",
            lines=[],
            mean_confidence=0.95,
        )

        classifier = DocumentClassifier(confidence_threshold=0.80)
        router = FileRouter(output_directory=self.output_dir)
        processor = BatchProcessor(
            config=self.config,
            ocr_engine=mock_ocr,
            classifier=classifier,
            router=router,
        )

        item = processor.process_file(img_file)

        self.assertEqual(mock_ocr.extract_text.call_count, 1)
        self.assertEqual(item.document_type, "PAN_CARD")
        self.assertEqual(item.status, "IDENTIFIED")
        self.assertEqual(item.rejection_category, "KYC")

    def test_single_ocr_invocation_for_non_kyc_and_low_confidence(self):
        """Verify that low-confidence, unknown, or multi-doc images invoke OCR strictly once with no second pass."""
        test_cases = [
            ("receipt.jpg", "Store receipt Rs 1500 total paid", "UNKNOWN", "NON_KYC"),
            (
                "multi.jpg",
                "GOVERNMENT OF INDIA 9876 5432 1098 help@uidai.gov.in INCOME TAX DEPARTMENT ABCDE1234F",
                "UNKNOWN",
                "MULTI_DOC",
            ),
            (
                "weak.jpg",
                "Name John Doe Father Bob Male",
                "UNKNOWN",
                "LOW_CONFIDENCE",
            ),
        ]

        for fname, text_val, expected_type, expected_cat in test_cases:
            img_file = self._create_sample_image(fname)

            mock_ocr = MagicMock(spec=OCREngine)
            mock_ocr.rec_model_name = "en_PP-OCRv5_mobile_rec"
            mock_ocr.extract_text.return_value = OCRResult(
                text=text_val,
                lines=[],
                mean_confidence=0.50,
            )

            classifier = DocumentClassifier(confidence_threshold=0.80)
            router = FileRouter(output_directory=self.output_dir)
            processor = BatchProcessor(
                config=self.config,
                ocr_engine=mock_ocr,
                classifier=classifier,
                router=router,
            )

            item = processor.process_file(img_file)

            # Strictly 1 invocation: no second pass, no escalation, no fallback
            self.assertEqual(
                mock_ocr.extract_text.call_count,
                1,
                f"Failed for {fname}: expected exactly 1 OCR call, got {mock_ocr.extract_text.call_count}",
            )
            self.assertEqual(item.document_type, expected_type)
            self.assertEqual(item.rejection_category, expected_cat)

    def test_profiling_block_contains_no_obsolete_stage_labels(self):
        """Verify DocumentProfileTiming formats clean single-pass output with no Stage 1/2 or Small/Med rec labels."""
        timing = DocumentProfileTiming(
            file_load_sec=0.05,
            preprocessing_sec=0.01,
            ocr_init_sec=0.0,
            ocr_inference_sec=1.45,
            ocr_processing_sec=0.03,
            classification_sec=0.02,
            confidence_sec=0.01,
            multi_doc_detection_sec=0.01,
            routing_sec=0.01,
            file_operation_sec=0.02,
            logging_sec=0.01,
            total_sec=1.62,
        )

        formatted = timing.format_profiling_block("document001.jpg")

        # Must NOT contain obsolete two-stage labels
        self.assertNotIn("Stage processed", formatted)
        self.assertNotIn("STAGE 1", formatted)
        self.assertNotIn("STAGE 2", formatted)
        self.assertNotIn("Fast Screen", formatted)
        self.assertNotIn("Small rec time", formatted)
        self.assertNotIn("Medium rec time", formatted)

        # Must accurately describe single-pass execution
        self.assertIn("Single-pass (PP-OCRv5_mobile)", formatted)
        self.assertIn("OCR inference:          1.45 sec", formatted)
        self.assertIn("TOTAL:                  1.62 sec", formatted)

    def test_ocr_engine_extract_text_invokes_paddle_ocr_once(self):
        """Verify OCREngine.extract_text invokes paddle_ocr.ocr() exactly once per image."""
        img_file = self._create_sample_image("scan.jpg")
        engine = OCREngine(use_gpu=False, cpu_threads=1)

        fake_paddle = MagicMock()
        fake_paddle.ocr.return_value = [
            [[[[0, 0], [10, 0], [10, 10], [0, 10]], ("INCOME TAX DEPARTMENT", 0.99)]]
        ]

        with patch.object(engine, "_get_ocr_instance", return_value=fake_paddle):
            result = engine.extract_text(img_file)

            # Exactly one call to paddle.ocr(img_arr)
            self.assertEqual(fake_paddle.ocr.call_count, 1)
            self.assertTrue(result.success)
            self.assertIn("INCOME TAX DEPARTMENT", result.text)


if __name__ == "__main__":
    unittest.main()
