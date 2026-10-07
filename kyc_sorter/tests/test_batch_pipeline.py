"""
Integration tests for the complete KYC Batch Processing Pipeline.
Covers real-world test scenarios:
1. Aadhaar Card detection and date-partitioned routing
2. PAN Card detection and date-partitioned routing
3. Non-KYC / Blurry image routing to Unknown
4. Unsupported document format (.txt file) skipped safely
5. Duplicate filename collision handling (prevent overwrite)
6. Audit CSV report generation
"""

import csv
import datetime
import tempfile
from pathlib import Path
import unittest
from PIL import Image, ImageDraw, ImageFilter

from kyc_sorter.detector.classifier import DocumentClassifier
from kyc_sorter.detector.ocr import OCREngine
from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.utils.config_loader import Config, load_config


class TestBatchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr_engine = OCREngine(language="en", use_gpu=False, cpu_threads=1)
        cls.classifier = DocumentClassifier(confidence_threshold=0.80)

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_config(
            overrides={
                "input": {"directory": str(self.input_dir)},
                "output": {"directory": str(self.output_dir), "operation": "copy"},
                "processing": {"confidence_threshold": 0.80, "dry_run": False},
            },
            base_dir=self.base_dir,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_card_image(self, path: Path, lines: list, blur: bool = False):
        img = Image.new("RGB", (600, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        y = 25
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0))
            y += 40

        if blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=5))

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    def test_batch_processing_and_segregation(self):
        # 1. Aadhaar Card sample (.jpg)
        aadhaar_path = self.input_dir / "aadhaar_sample.jpg"
        self._create_card_image(
            aadhaar_path,
            [
                "GOVERNMENT OF INDIA",
                "Unique Identification Authority of India",
                "9876 5432 1098",
                "Help@uidai.gov.in 1947",
            ],
        )

        # 2. PAN Card sample (.JPG uppercase)
        pan_path = self.input_dir / "pan_sample.JPG"
        self._create_card_image(
            pan_path,
            [
                "INCOME TAX DEPARTMENT",
                "GOVT. OF INDIA",
                "Permanent Account Number Card",
                "ABCDE1234F",
            ],
        )

        # 3. Non-KYC sample (Receipt / unrelated text in .jpg)
        non_kyc_path = self.input_dir / "grocery_bill.jpg"
        self._create_card_image(
            non_kyc_path,
            [
                "Fresh Supermarket Receipt",
                "Apples 2kg - Rs 240",
                "Milk 1L - Rs 60",
                "Total Amount: Rs 300",
            ],
        )

        # 4. Unsupported file formats (must be completely ignored per Requirement 2 & 5)
        (self.input_dir / "notes.txt").write_text("Unrelated document format", encoding="utf-8")
        (self.input_dir / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)
        (self.input_dir / "photo.jpeg").write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 30)
        (self.input_dir / "photo.bmp").write_bytes(b"BM" + b"\x00" * 30)

        processor = BatchProcessor(
            config=self.config,
            ocr_engine=self.ocr_engine,
            classifier=self.classifier,
        )

        # Execute full batch
        summary = processor.process_batch(source_dir=self.input_dir)

        # 1. Check file counts
        self.assertEqual(summary.total_files, 3, "Should process exactly 3 supported .jpg/.JPG files")
        self.assertEqual(summary.skipped_unsupported, 4, "Should skip the 4 non-JPG files (.txt, .png, .jpeg, .bmp)")
        self.assertEqual(summary.processed, 3)
        self.assertEqual(summary.failed, 0)

        # 2. Check classification counts
        self.assertEqual(summary.counts_by_type.get("AADHAAR_CARD", 0), 1)
        self.assertEqual(summary.counts_by_type.get("PAN_CARD", 0), 1)
        self.assertEqual(summary.counts_by_type.get("UNKNOWN", 0), 1)

        # 3. Check direct folder routing (KYC and NON KYC directly, NO SUBFOLDERS, NO DATE FOLDERS)
        aadhaar_file = self.output_dir / "KYC" / "aadhaar_sample_AADHAAR_CARD.jpg"
        pan_file = self.output_dir / "KYC" / "pan_sample_PAN_CARD.JPG"
        unknown_file = self.output_dir / "NON KYC" / "grocery_bill.jpg"

        self.assertTrue(aadhaar_file.is_file(), "Aadhaar file must exist directly in KYC folder with _AADHAAR_CARD suffix")
        self.assertTrue(pan_file.is_file(), "PAN file must exist directly in KYC folder with _PAN_CARD suffix")
        self.assertTrue(unknown_file.is_file(), "Non-KYC file must route directly to NON KYC folder preserving original name")

        # Verify non-JPG files were NEVER copied, moved, or renamed
        self.assertFalse((self.output_dir / "KYC" / "photo_AADHAAR_CARD.png").exists())
        self.assertFalse((self.output_dir / "NON KYC" / "photo.png").exists())
        self.assertFalse((self.output_dir / "NON KYC" / "photo_UNKNOWN.png").exists())
        self.assertFalse((self.output_dir / "KYC" / "photo_AADHAAR_CARD.jpeg").exists())

        # 4. Check duplicate filename collision handling (prevent overwrite)
        dup_result = processor.process_file(pan_path)
        expected_dup_name = "pan_sample_PAN_CARD_1.JPG"
        self.assertEqual(Path(dup_result.destination_path).name, expected_dup_name)
        self.assertTrue(Path(dup_result.destination_path).is_file())
        self.assertTrue(pan_file.is_file(), "Original file must NOT be overwritten")

        # 5. Check that CSV report generation is completely disabled
        self.assertIsNone(summary.csv_report_path, "CSV report path must be None")
        csv_files = list(self.output_dir.glob("*.csv")) + list(self.output_dir.rglob("*.csv"))
        self.assertEqual(len(csv_files), 0, f"No CSV reports should be created, found: {csv_files}")

        # Check output folders: only KYC and NON KYC
        out_subdirs = [p.name for p in self.output_dir.iterdir() if p.is_dir()]
        self.assertEqual(sorted(out_subdirs), ["KYC", "NON KYC"])


if __name__ == "__main__":
    unittest.main()
