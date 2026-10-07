"""
Unit tests for configuration loader and path resolution.
"""

import tempfile
from pathlib import Path
import unittest
import yaml

from kyc_sorter.utils.config_loader import load_config, Config


class TestConfigLoader(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.application.name, "KYC Document Sorter")
        self.assertIn(config.output.operation, ("copy", "move"))
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.80)
        self.assertEqual(config.processing.supported_extensions, {".jpg"})
        self.assertFalse(config.security.log_sensitive_text)
        self.assertEqual(config.get_destination_subfolder("AADHAAR_CARD"), "AADHAAR CARD")
        self.assertEqual(config.get_destination_subfolder("PAN_CARD"), "PAN CARD")
        self.assertEqual(config.get_destination_subfolder("BANK_PASSBOOK"), "BANK STATEMENT")
        self.assertEqual(config.get_destination_subfolder("NON_EXISTENT"), "NON KYC")
        self.assertEqual(config.unknown_folder, "NON KYC")
        # Filename suffixes test
        self.assertEqual(config.get_filename_suffix("AADHAAR_CARD"), "AADHAAR_CARD")
        self.assertEqual(config.get_filename_suffix("PAN_CARD"), "PAN_CARD")
        self.assertEqual(config.get_filename_suffix("VOTER_ID"), "VOTER_ID")
        self.assertEqual(config.get_filename_suffix("BANK_PASSBOOK"), "BANK_STATEMENT")
        self.assertEqual(config.get_filename_suffix("PASSPORT"), "PASSPORT")
        self.assertEqual(config.get_filename_suffix("UNKNOWN"), "UNKNOWN")
        self.assertEqual(config.get_filename_suffix("NON_EXISTENT"), "UNKNOWN")
        # Watcher config test
        self.assertEqual(config.watcher.stability_delay_sec, 1.0)
        self.assertEqual(config.watcher.settle_timeout_sec, 30.0)
        # New processing configs
        self.assertEqual(config.processing.cpu_limit_percent, 80.0)
        self.assertTrue(config.processing.adaptive_workers)
        self.assertAlmostEqual(config.processing.multiple_document_score_margin, 0.10)
        self.assertIn(config.ocr.cpu_threads, (1, 3, 4))

    def test_overrides_and_validation(self):
        overrides = {
            "output": {"operation": "MOVE"},
            "processing": {
                "confidence_threshold": 0.95,
                "dry_run": True,
                "cpu_limit_percent": 75.0,
                "multiple_document_score_margin": 0.15,
                "supported_extensions": ["JPG", "pdf", ".TIFF"],
            },
            "security": {"log_sensitive_text": True},
        }
        config = load_config(overrides=overrides)
        self.assertEqual(config.output.operation, "move")
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.95)
        self.assertTrue(config.processing.dry_run)
        self.assertEqual(config.processing.cpu_limit_percent, 75.0)
        self.assertAlmostEqual(config.processing.multiple_document_score_margin, 0.15)
        self.assertTrue(config.security.log_sensitive_text)
        self.assertIn(".jpg", config.processing.supported_extensions)
        self.assertIn(".pdf", config.processing.supported_extensions)
        self.assertIn(".tiff", config.processing.supported_extensions)

    def test_invalid_operation_fallback(self):
        overrides = {"output": {"operation": "delete_all"}}
        config = load_config(overrides=overrides)
        # Should fallback to safe default "copy"
        self.assertEqual(config.output.operation, "copy")

    def test_confidence_threshold_clamping(self):
        config_low = load_config(overrides={"processing": {"confidence_threshold": -0.5}})
        self.assertEqual(config_low.processing.confidence_threshold, 0.0)

        config_high = load_config(overrides={"processing": {"confidence_threshold": 1.5}})
        self.assertEqual(config_high.processing.confidence_threshold, 1.0)

    def test_max_workers_configuration(self):
        config = load_config()
        self.assertIn(config.processing.max_workers, (1, 6))

        custom = load_config(overrides={"processing": {"max_workers": 8}})
        self.assertEqual(custom.processing.max_workers, 8)

    def test_adaptive_worker_calculation(self):
        from kyc_sorter.sorter.batch_processor import compute_adaptive_workers
        self.assertEqual(compute_adaptive_workers(4, total_cores=2), 1)
        self.assertEqual(compute_adaptive_workers(4, total_cores=4), 2)
        self.assertEqual(compute_adaptive_workers(4, total_cores=6), 3)
        self.assertEqual(compute_adaptive_workers(4, total_cores=8), 4)
        self.assertEqual(compute_adaptive_workers(4, total_cores=12), 4)
        self.assertEqual(compute_adaptive_workers(6, total_cores=12), 6)
        self.assertEqual(compute_adaptive_workers(8, total_cores=4), 2)


if __name__ == "__main__":
    unittest.main()

