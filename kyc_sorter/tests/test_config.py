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
        self.assertEqual(config.output.operation, "copy")
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.80)
        self.assertEqual(config.processing.supported_extensions, {".jpg"})
        self.assertFalse(config.security.log_sensitive_text)
        self.assertEqual(config.get_destination_subfolder("AADHAAR_CARD"), "AADHAAR CARD")
        self.assertEqual(config.get_destination_subfolder("PAN_CARD"), "PAN CARD")
        self.assertEqual(config.get_destination_subfolder("BANK_PASSBOOK"), "BANK STATEMENT")
        self.assertEqual(config.get_destination_subfolder("NON_EXISTENT"), "Unknown")
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

    def test_overrides_and_validation(self):
        overrides = {
            "output": {"operation": "MOVE"},
            "processing": {
                "confidence_threshold": 0.95,
                "dry_run": True,
                "supported_extensions": ["JPG", "pdf", ".TIFF"],
            },
            "security": {"log_sensitive_text": True},
        }
        config = load_config(overrides=overrides)
        self.assertEqual(config.output.operation, "move")
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.95)
        self.assertTrue(config.processing.dry_run)
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
        self.assertEqual(config.processing.max_workers, 4)

        custom = load_config(overrides={"processing": {"max_workers": 6}})
        self.assertEqual(custom.processing.max_workers, 6)

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

