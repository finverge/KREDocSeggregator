"""
Unit and integration tests for SourceFolderWatcher and safe file-ready checks.
"""

from __future__ import annotations

import datetime
import tempfile
import threading
import time
from pathlib import Path
import unittest

from kyc_sorter.sorter.batch_processor import BatchProcessor, ProcessedItem
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.sorter.watcher import SourceFolderWatcher, wait_for_file_ready
from kyc_sorter.utils.config_loader import Config, InputConfig, OutputConfig, ProcessingConfig, WatcherConfig


class TestFileReadyCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_ready_success(self):
        test_file = self.tmp_path / "sample.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"dummy image data" * 20)

        ready = wait_for_file_ready(
            file_path=test_file,
            stability_delay_sec=0.1,
            settle_timeout_sec=2.0,
            check_interval_sec=0.05,
        )
        self.assertTrue(ready)

    def test_file_ready_non_existent(self):
        missing = self.tmp_path / "missing.png"
        ready = wait_for_file_ready(
            file_path=missing,
            stability_delay_sec=0.1,
            settle_timeout_sec=0.3,
            check_interval_sec=0.05,
        )
        self.assertFalse(ready)

    def test_file_ready_growing_file(self):
        """Simulate a file that continues to grow before stabilizing."""
        growing_file = self.tmp_path / "growing.jpg"
        growing_file.write_bytes(b"initial chunk")

        def append_later():
            time.sleep(0.15)
            try:
                with open(growing_file, "ab") as f:
                    f.write(b"second larger chunk of data")
            except OSError:
                pass

        writer_thread = threading.Thread(target=append_later)
        writer_thread.start()

        ready = wait_for_file_ready(
            file_path=growing_file,
            stability_delay_sec=0.2,
            settle_timeout_sec=3.0,
            check_interval_sec=0.05,
        )
        writer_thread.join()
        self.assertTrue(ready)
        self.assertGreater(growing_file.stat().st_size, len(b"initial chunk"))


class DummyMockProcessor:
    """Mock processor for testing watcher routing and duplicate prevention."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.processed_calls = []
        self.router = FileRouter(
            output_directory=output_dir,
            create_date_folder=False,
            create_subfolders=False,
            operation="copy",
        )

    def process_file(self, file_path: Path) -> ProcessedItem:
        self.processed_calls.append(file_path.name)
        # Simulate classification as PAN_CARD
        decision = self.router.route(file_path, "PAN_CARD")
        return ProcessedItem(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_file=file_path.name,
            source_path=str(file_path),
            document_type="PAN_CARD",
            confidence=0.95,
            status="IDENTIFIED",
            matched_template="PAN_CARD",
            destination_path=str(decision.destination_path),
            operation="copy",
            dry_run=False,
        )


class TestSourceFolderWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)
        self.input_dir = self.base_dir / "Input"
        self.output_dir = self.base_dir / "Output"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        self.config = Config(
            input=InputConfig(directory=str(self.input_dir)),
            output=OutputConfig(directory=str(self.output_dir), operation="copy", create_date_folder=False),
            processing=ProcessingConfig(confidence_threshold=0.80),
            watcher=WatcherConfig(stability_delay_sec=0.1, settle_timeout_sec=2.0, poll_interval_sec=0.05),
        )
        self.config.input.resolve(self.base_dir)
        self.config.output.resolve(self.base_dir)

        self.mock_processor = DummyMockProcessor(self.output_dir)
        self.watcher = SourceFolderWatcher(
            config=self.config,
            processor=self.mock_processor,
        )

    def tearDown(self):
        self.watcher.stop()
        self.tmp.cleanup()

    def test_duplicate_prevention_on_multiple_events(self):
        """Simulate rapid multiple filesystem events for the same file."""
        sample_img = self.input_dir / "IMG0002.jpg"
        sample_img.write_bytes(b"\xFF\xD8\xFF" + b"pan image data" * 30)

        # Trigger on_filesystem_event multiple times simulating created + modified + closed
        self.watcher.on_filesystem_event(sample_img)
        self.watcher.on_filesystem_event(sample_img)
        self.watcher.on_filesystem_event(sample_img)

        # Wait for worker thread to complete
        time.sleep(0.5)

        # Check: must only be processed exactly once
        self.assertEqual(len(self.mock_processor.processed_calls), 1)
        self.assertEqual(self.mock_processor.processed_calls[0], "IMG0002.jpg")

        # Verify output file with renamed suffix directly in KYC folder (no date or subfolders)
        expected_renamed = self.output_dir / "KYC" / "IMG0002_PAN_CARD.jpg"
        self.assertTrue(expected_renamed.is_file(), f"Renamed file must exist at {expected_renamed}")

    def test_non_jpg_extensions_completely_ignored(self):
        """Non-JPG extensions (.jpeg, .png, .bmp, .tif, .pdf, .txt) must be ignored completely."""
        non_jpg_files = [
            self.input_dir / "sample.jpeg",
            self.input_dir / "sample.png",
            self.input_dir / "sample.bmp",
            self.input_dir / "sample.tif",
            self.input_dir / "document.pdf",
            self.input_dir / "notes.txt",
        ]
        for f in non_jpg_files:
            f.write_bytes(b"\x00" * 50)
            self.watcher.on_filesystem_event(f)

        time.sleep(0.3)
        self.assertEqual(len(self.mock_processor.processed_calls), 0, "Non-JPG files must never be processed")

    def test_uppercase_jpg_extension_accepted(self):
        """Uppercase .JPG must be accepted and processed."""
        jpg_upper = self.input_dir / "DOC0099.JPG"
        jpg_upper.write_bytes(b"\xFF\xD8\xFF" + b"image data" * 20)

        self.watcher.on_filesystem_event(jpg_upper)
        time.sleep(0.4)

        self.assertIn("DOC0099.JPG", self.mock_processor.processed_calls)

    def test_watcher_resilience_on_failing_document(self):
        """Watcher must continue running and process subsequent documents if one document fails."""
        failing_img = self.input_dir / "failing.jpg"
        failing_img.write_bytes(b"\xFF\xD8\xFF" + b"corrupt file data")

        valid_img = self.input_dir / "IMG0003.jpg"
        valid_img.write_bytes(b"\xFF\xD8\xFF" + b"valid image data")

        original_process = self.mock_processor.process_file

        def side_effect(p: Path):
            if "failing" in p.name:
                raise RuntimeError("Simulated OCR failure or corrupt file")
            return original_process(p)

        self.mock_processor.process_file = side_effect

        self.watcher.on_filesystem_event(failing_img)
        time.sleep(0.5)

        self.watcher.on_filesystem_event(valid_img)
        time.sleep(0.5)

        # Valid image must have been processed despite earlier failure
        self.assertIn("IMG0003.jpg", self.mock_processor.processed_calls)

    def test_clean_stop_and_shutdown(self):
        """Watcher starts and shuts down cleanly without lingering threads."""
        self.watcher.start(process_existing=False)
        self.assertTrue(self.watcher.is_running)
        self.watcher.stop()
        self.assertFalse(self.watcher.is_running)


if __name__ == "__main__":
    unittest.main()
