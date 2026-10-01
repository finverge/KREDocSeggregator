"""
Unit tests for FileRouter.
Validates destination path calculation, strict two-folder routing (KYC and Unknown),
no document subfolders, no date folders, all 16 official naming conventions,
collision deduplication, copy mode, move mode, and dry-run mode.
"""

import datetime
import tempfile
from pathlib import Path
import unittest

from kyc_sorter.sorter.file_router import FileRouter, RouteDecision


class TestFileRouter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)
        self.output_dir = self.base_path / "Destination files"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_date = datetime.date(2026, 9, 30)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_default_routing_no_subfolders_no_date_folders(self):
        """Default router must route directly to KYC or Unknown without subfolders or dates."""
        router = FileRouter(output_directory=self.output_dir)
        src = self.base_path / "IMG0001.jpg"

        dest_path, folder_name, date_str = router.calculate_destination_path(
            source_file=src,
            document_type="AADHAAR_CARD",
            processing_date=self.fixed_date,
        )
        expected = self.output_dir / "KYC" / "IMG0001_AADHAAR_CARD.jpg"
        self.assertEqual(dest_path, expected)
        self.assertIsNone(date_str)
        # Ensure parent is directly KYC
        self.assertEqual(dest_path.parent, self.output_dir / "KYC")

    def test_unknown_routing_directly_to_unknown_folder(self):
        """Unknown documents must route directly to Unknown without subfolders or dates."""
        router = FileRouter(output_directory=self.output_dir)
        src = self.base_path / "blurry_photo.png"

        dest_path, folder_name, _ = router.calculate_destination_path(
            source_file=src,
            document_type="UNKNOWN",
            processing_date=self.fixed_date,
        )
        expected = self.output_dir / "Unknown" / "blurry_photo_UNKNOWN.png"
        self.assertEqual(dest_path, expected)
        self.assertEqual(dest_path.parent, self.output_dir / "Unknown")

    def test_all_16_official_document_naming_and_routing(self):
        """Verify all 16 official document types and suffixes per Section 4 & 5."""
        router = FileRouter(output_directory=self.output_dir)

        test_cases = [
            ("IMG0001.jpg", "AADHAAR CARD", "_AADHAAR_CARD", "KYC"),
            ("IMG0001.jpg", "AADHAAR_CARD", "_AADHAAR_CARD", "KYC"),
            ("IMG0002.jpeg", "PAN CARD", "_PAN_CARD", "KYC"),
            ("IMG0002.jpeg", "PAN_CARD", "_PAN_CARD", "KYC"),
            ("IMG0003.png", "VOTER ID", "_VOTER_ID", "KYC"),
            ("IMG0003.png", "VOTER_ID", "_VOTER_ID", "KYC"),
            ("IMG0004.jpg", "BANK STATEMENT", "_BANK_STATEMENT", "KYC"),
            ("IMG0004.jpg", "BANK_STATEMENT", "_BANK_STATEMENT", "KYC"),
            ("IMG0004.jpg", "BANK_PASSBOOK", "_BANK_STATEMENT", "KYC"),
            ("IMG0005.jpeg", "CKYC CONFIRMATION", "_CKYC_CONFIRMATION", "KYC"),
            ("IMG0005.jpeg", "CKYC_CONFIRMATION", "_CKYC_CONFIRMATION", "KYC"),
            ("IMG0006.jpg", "DRIVING LICENCE", "_DRIVING_LICENCE", "KYC"),
            ("IMG0006.jpg", "DRIVING_LICENCE", "_DRIVING_LICENCE", "KYC"),
            ("IMG0007.jpg", "ELECTRICITY BILL", "_ELECTRICITY_BILL", "KYC"),
            ("IMG0007.jpg", "ELECTRICITY_BILL", "_ELECTRICITY_BILL", "KYC"),
            ("IMG0008.jpg", "GAS BILL", "_GAS_BILL", "KYC"),
            ("IMG0008.jpg", "GAS_BILL", "_GAS_BILL", "KYC"),
            ("IMG0009.jpg", "MUNICIPAL TAX RECEIPT", "_MUNICIPAL_RECEIPT", "KYC"),
            ("IMG0009.jpg", "MUNICIPAL_TAX_RECEIPT", "_MUNICIPAL_RECEIPT", "KYC"),
            ("IMG0010.jpg", "PHONE BILL", "_PHONE_BILL", "KYC"),
            ("IMG0010.jpg", "PHONE_BILL", "_PHONE_BILL", "KYC"),
            ("IMG0011.jpg", "RATION CARD", "_RATION_CARD", "KYC"),
            ("IMG0011.jpg", "RATION_CARD", "_RATION_CARD", "KYC"),
            ("IMG0012.jpg", "REGISTRATION CERTIFICATE", "_REGISTRATION_CERTIFICATE", "KYC"),
            ("IMG0012.jpg", "REGISTRATION_CERTIFICATE", "_REGISTRATION_CERTIFICATE", "KYC"),
            ("IMG0013.jpg", "PASSPORT", "_PASSPORT", "KYC"),
            ("IMG0014.jpg", "GOVT ID", "_GOVT_ID", "KYC"),
            ("IMG0014.jpg", "GOVT_ID", "_GOVT_ID", "KYC"),
            ("IMG0015.jpg", "CKYC DECLARATION", "_CKYC_DECLARATION", "KYC"),
            ("IMG0015.jpg", "CKYC_DECLARATION", "_CKYC_DECLARATION", "KYC"),
            ("IMG0016.jpg", "IT RETURNS", "_IT_RETURNS", "KYC"),
            ("IMG0016.jpg", "IT_RETURNS", "_IT_RETURNS", "KYC"),
            # Non-KYC / Unknown
            ("IMG0020.jpg", "UNKNOWN", "_UNKNOWN", "Unknown"),
            ("IMG0021.jpeg", "UNKNOWN", "_UNKNOWN", "Unknown"),
        ]

        for filename, doc_type, expected_suffix, expected_bucket in test_cases:
            src = self.base_path / filename
            dest_path, _, _ = router.calculate_destination_path(
                source_file=src,
                document_type=doc_type,
                processing_date=self.fixed_date,
            )

            stem = Path(filename).stem
            ext = Path(filename).suffix
            expected_name = f"{stem}{expected_suffix}{ext}"

            self.assertEqual(dest_path.name, expected_name)
            self.assertEqual(dest_path.parent, self.output_dir / expected_bucket)
            # Ensure no subfolders exist between output_dir and the file except KYC or Unknown
            rel_parts = dest_path.relative_to(self.output_dir).parts
            self.assertEqual(len(rel_parts), 2, f"Must have exactly 2 parts: {rel_parts}")
            self.assertEqual(rel_parts[0], expected_bucket)

    def test_collision_deduplication(self):
        """When destination file exists, append _1, _2, etc. before extension (Section 15)."""
        router = FileRouter(output_directory=self.output_dir)
        kyc_dir = self.output_dir / "KYC"
        kyc_dir.mkdir(parents=True, exist_ok=True)

        # Simulate existing file IMG0001_AADHAAR_CARD.jpg
        (kyc_dir / "IMG0001_AADHAAR_CARD.jpg").write_text("file 1 content")

        src = self.base_path / "IMG0001.jpg"
        dest_path1, _, _ = router.calculate_destination_path(src, "AADHAAR_CARD")
        self.assertEqual(dest_path1.name, "IMG0001_AADHAAR_CARD_1.jpg")

        # Simulate IMG0001_AADHAAR_CARD_1.jpg also exists
        (kyc_dir / "IMG0001_AADHAAR_CARD_1.jpg").write_text("file 2 content")
        dest_path2, _, _ = router.calculate_destination_path(src, "AADHAAR_CARD")
        self.assertEqual(dest_path2.name, "IMG0001_AADHAAR_CARD_2.jpg")

    def test_copy_operation(self):
        """In COPY mode, original file remains in source and copy is created in destination."""
        router = FileRouter(output_directory=self.output_dir, operation="copy")
        src = self.base_path / "IMG0001.jpg"
        src.write_text("dummy binary content")

        decision = router.route(src, "AADHAAR_CARD")
        self.assertTrue(decision.success)
        self.assertFalse(decision.dry_run)
        self.assertTrue(src.exists(), "Source file must remain in COPY mode")
        self.assertTrue(decision.destination_path.exists(), "Destination file must be created")
        self.assertEqual(decision.destination_path.read_text(), "dummy binary content")
        self.assertEqual(decision.destination_path.parent, self.output_dir / "KYC")

    def test_move_operation(self):
        """In MOVE mode, original file is moved to destination only after successful processing."""
        router = FileRouter(output_directory=self.output_dir, operation="move")
        src = self.base_path / "IMG0002.jpg"
        src.write_text("pan card binary data")

        decision = router.route(src, "PAN_CARD")
        self.assertTrue(decision.success)
        self.assertFalse(src.exists(), "Source file must be moved")
        self.assertTrue(decision.destination_path.exists())
        self.assertEqual(decision.destination_path.read_text(), "pan card binary data")
        self.assertEqual(decision.destination_path.parent, self.output_dir / "KYC")

    def test_dry_run_mode(self):
        """In dry-run mode, neither source nor destination are modified."""
        router = FileRouter(output_directory=self.output_dir, operation="move", dry_run=True)
        src = self.base_path / "critical_doc.jpg"
        src.write_text("must not be touched")

        decision = router.route(src, "AADHAAR_CARD")
        self.assertTrue(decision.success)
        self.assertTrue(decision.dry_run)
        self.assertTrue(src.exists(), "Source file must NEVER be touched in dry-run mode")
        self.assertFalse(decision.destination_path.exists(), "Destination file must NOT be written in dry-run")

    def test_missing_source_file_error_handling(self):
        """Non-existent source file returns graceful error without throwing unhandled exception."""
        router = FileRouter(output_directory=self.output_dir)
        non_existent = self.base_path / "non_existent.jpg"
        decision = router.route(non_existent, "AADHAAR_CARD")
        self.assertFalse(decision.success)
        self.assertIn("does not exist", str(decision.error))


if __name__ == "__main__":
    unittest.main()
