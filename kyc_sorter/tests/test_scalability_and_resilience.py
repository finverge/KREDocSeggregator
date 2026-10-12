"""
Comprehensive unit and integration tests for Scalability, Reliability & Production Readiness.
Covers:
  - SQLite persistent journaling, crash recovery, and filesystem reconciliation
  - Dry-run immutability (no journal mutations)
  - Ollama bounded exponential backoff retries (HTTP 502/503/504 vs HTTP 400)
  - CPU throttling pause & resume without task abandonment
  - Deadlock safety timeout during CPU throttling
  - Batch streaming boundaries (os.scandir chunking)
  - Watcher bounded concurrency, event bursts, deduplication, and overflow backpressure
  - Destination collision handling and safe file routing
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kyc_sorter.sorter.batch_processor import BatchProcessor, ProcessedItem, iter_source_batches
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.sorter.journal import ProcessingJournal
from kyc_sorter.sorter.vlm_router import (
    classify_document,
    process_vlm_batch,
    route_document,
)
from kyc_sorter.sorter.watcher import SourceFolderWatcher
from kyc_sorter.utils.config_loader import (
    Config,
    InputConfig,
    OutputConfig,
    ProcessingConfig,
    VLMConfig,
    WatcherConfig,
)


# ==============================================================================
# 1. SQLite Persistent Journaling & Crash Reconciliation Tests
# ==============================================================================

class TestProcessingJournal:
    """Tests for SQLite WAL-mode journal, crash recovery, and reconciliation."""

    def test_journal_initialization_wal_mode(self, tmp_path):
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)
        assert db_path.is_file()

        # Check WAL mode
        with sqlite3.connect(db_path) as conn:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"

            # Check table exists
            table_info = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='document_journal';"
            ).fetchone()
            assert table_info is not None
        journal.close()

    def test_journal_record_lifecycle(self, tmp_path):
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)

        src = tmp_path / "test.jpg"
        src.write_bytes(b"\xFF\xD8\xFF" + b"image data" * 10)
        dst = tmp_path / "test_PAN_CARD.jpg"
        dst.write_bytes(b"\xFF\xD8\xFF" + b"image data" * 10)

        # 1. Record start
        journal.record_start(src)
        status = journal.get_file_status(src)
        assert status is not None
        assert status["status"] == "IN_PROGRESS"

        # 2. Record completion
        journal.record_completed(
            source_path=src,
            category="PAN_CARD",
            destination_path=dst,
            confidence=0.95,
            operation="move",
        )
        status2 = journal.get_file_status(src)
        assert status2 is not None
        assert status2["status"] == "COMPLETED"
        assert status2["category"] == "PAN_CARD"
        assert status2["destination_path"] == str(dst.resolve())
        journal.close()

    def test_reconcile_skips_when_dest_exists_and_verified(self, tmp_path):
        """When destination exists with valid size, reconcile confirms completion."""
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)

        src = tmp_path / "doc.jpg"
        src.write_bytes(b"content" * 10)
        dst = tmp_path / "doc_PAN_CARD.jpg"
        dst.write_bytes(b"content" * 10)

        journal.record_start(src)
        journal.record_completed(
            source_path=src,
            category="PAN_CARD",
            destination_path=dst,
            confidence=0.98,
            operation="copy",
        )

        # Reconcile should return not None (skip re-processing)
        assert journal.reconcile_file_status(src, is_move_operation=False) is not None
        journal.close()

    def test_reconcile_reprocesses_if_crashed_before_file_moved(self, tmp_path):
        """If journal claims completed or in-progress, but destination file does NOT exist, reconcile returns None."""
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)

        src = tmp_path / "doc_crash.jpg"
        src.write_bytes(b"content" * 10)
        non_existent_dst = tmp_path / "missing_PAN_CARD.jpg"

        journal.record_start(src)
        journal.record_completed(
            source_path=src,
            category="PAN_CARD",
            destination_path=non_existent_dst,
            confidence=0.98,
            operation="move",
        )

        # Reconcile must catch that destination does not exist and flag for reprocessing (None)
        assert journal.reconcile_file_status(src, is_move_operation=True) is None
        journal.close()

    def test_reconcile_reprocesses_if_dest_file_is_empty_corrupt(self, tmp_path):
        """If destination exists but has size 0 (partial write crash), reconcile returns None."""
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)

        src = tmp_path / "doc_partial.jpg"
        src.write_bytes(b"actual content" * 10)
        dst = tmp_path / "partial_PAN_CARD.jpg"
        dst.write_bytes(b"")  # 0 bytes

        journal.record_start(src)
        journal.record_completed(
            source_path=src,
            category="PAN_CARD",
            destination_path=dst,
            confidence=0.98,
            operation="move",
        )

        assert journal.reconcile_file_status(src, is_move_operation=True) is None
        journal.close()

    def test_reconcile_reprocesses_if_source_modified_since_journal(self, tmp_path):
        """If source file mtime/size changed after completion, reprocess."""
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path)

        src = tmp_path / "doc_mod.jpg"
        src.write_bytes(b"initial" * 10)
        dst = tmp_path / "doc_mod_PAN_CARD.jpg"
        dst.write_bytes(b"initial" * 10)

        journal.record_start(src)
        journal.record_completed(
            source_path=src,
            category="PAN_CARD",
            destination_path=dst,
            confidence=0.98,
            operation="copy",
        )

        # Modify source file
        time.sleep(0.05)
        src.write_bytes(b"new modified content that is much larger")

        assert journal.reconcile_file_status(src, is_move_operation=False) is None
        journal.close()

    def test_dry_run_immutability(self, tmp_path):
        """Dry run mode must NEVER write to SQLite journal."""
        db_path = tmp_path / "journal.db"
        journal = ProcessingJournal(db_path=db_path, dry_run=True)

        src = tmp_path / "sample.jpg"
        src.write_bytes(b"data")
        dst = tmp_path / "sample_PAN.jpg"

        journal.record_start(src)
        journal.record_completed(src, "PAN_CARD", dst, 0.99, "copy")
        journal.record_failed(src, "Some error")

        # Database file must not even exist or have 0 rows
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM document_journal;").fetchone()[0]
                assert count == 0
        journal.close()


# ==============================================================================
# 2. Ollama Request Bounded Retries Tests
# ==============================================================================

class TestOllamaRetries:
    """Tests for bounded exponential backoff retries on Ollama requests."""

    def test_retry_transient_http_503_then_succeed(self, tmp_path):
        """Ollama transient 503 error succeeds on 3rd attempt (2 retries)."""
        img_path = tmp_path / "img_503.jpg"
        img_path.write_bytes(b"\xFF\xD8\xFF" + b"image bytes" * 10)

        call_count = 0

        def mock_post(url, json=None, timeout=300):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count < 3:
                resp.status_code = 503
                resp.text = '{"error": "model loading"}'
                return resp
            resp.status_code = 200
            resp.json.return_value = {
                "message": {"content": "DOCUMENT_TYPE: PAN CARD\nCONFIDENCE: 0.95"}
            }
            return resp

        with patch("requests.post", side_effect=mock_post):
            with patch("time.sleep") as mock_sleep:
                res = classify_document(img_path)

        assert res.category == "PAN CARD"
        assert res.error is None
        assert call_count == 3
        # Should have slept with exponential backoff (1.0s, then 2.0s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    def test_retry_exhaustion_on_persistent_503(self, tmp_path):
        """Ollama fails with HTTP 503 permanently -> stops after max_retries=3 (4 total attempts)."""
        img_path = tmp_path / "img_exhaust.jpg"
        img_path.write_bytes(b"\xFF\xD8\xFF" + b"image bytes" * 10)

        call_count = 0

        def mock_post(url, json=None, timeout=300):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 503
            resp.text = '{"error": "server overloaded"}'
            return resp

        with patch("requests.post", side_effect=mock_post):
            with patch("time.sleep"):
                res = classify_document(img_path)

        # Permanent failure returns None category with error message
        assert res.category is None
        assert res.error is not None
        # 1 initial attempt + 3 retries = 4 attempts
        assert call_count == 4

    def test_no_retry_on_client_error_400(self, tmp_path):
        """HTTP 400 Bad Request is permanent and must NOT retry."""
        img_path = tmp_path / "img_400.jpg"
        img_path.write_bytes(b"\xFF\xD8\xFF" + b"image bytes" * 10)

        call_count = 0

        def mock_post(url, json=None, timeout=300):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 400
            resp.text = '{"error": "invalid parameter"}'
            return resp

        with patch("requests.post", side_effect=mock_post):
            with patch("time.sleep") as mock_sleep:
                res = classify_document(img_path)

        assert res.category is None
        assert res.error is not None
        assert call_count == 1  # Exactly 1 call, 0 retries
        mock_sleep.assert_not_called()

    def test_retry_on_transient_connection_error(self, tmp_path):
        """ConnectionError is retried with backoff."""
        img_path = tmp_path / "img_conn.jpg"
        img_path.write_bytes(b"\xFF\xD8\xFF" + b"image bytes" * 10)

        call_count = 0

        def mock_post(url, json=None, timeout=300):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                import requests
                raise requests.exceptions.ConnectionError("Connection refused")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "message": {"content": "DOCUMENT_TYPE: AADHAAR CARD\nCONFIDENCE: 0.99"}
            }
            return resp

        with patch("requests.post", side_effect=mock_post):
            with patch("time.sleep") as mock_sleep:
                res = classify_document(img_path)

        assert res.category == "AADHAAR CARD"
        assert res.error is None
        assert call_count == 2
        mock_sleep.assert_called_once_with(1.0)


# ==============================================================================
# 3. CPU Throttling Pause/Resume & Deadlock Safety Tests
# ==============================================================================

class TestCpuThrottleRecovery:
    """Tests that CPU throttling pauses and resumes without abandoning files."""

    def test_cpu_throttling_pauses_and_resumes_without_dropping_tasks(self, tmp_path):
        """Simulate CPU exceeding throttle threshold for 3 intervals, then dropping."""
        files = []
        for i in range(6):
            f = tmp_path / f"test_{i}.jpg"
            f.write_bytes(b"\xFF\xD8\xFF" + b"content" * 10)
            files.append(f)

        dest_root = tmp_path / "output"
        dest_root.mkdir()

        # Mock CPU monitor: throttled for the first 2 checks, then unthrottled
        class FakeCpuMonitor:
            def __init__(self):
                self.checks = 0
                self.cpu_exceeded = False
                self.last_sample_time = time.time()
                self.samples = [60.0]
                self.peak_cpu = 85.0
                self.peak_mem_percent = 50.0
                self.ever_cpu_limit_exceeded = False
                self.peak_vram_mb = 0.0
                self.gpu_info = {}

            @property
            def throttle_active(self):
                self.checks += 1
                return self.checks <= 2

            def stop(self):
                pass

        fake_monitor = FakeCpuMonitor()

        from kyc_sorter.sorter.vlm_router import RouteResult

        def fake_process_vlm_file(image_file, **kwargs):
            p = Path(image_file)
            dest_p = dest_root / f"PAN_{p.name}"
            return RouteResult(
                source_file=p,
                destination_path=dest_p,
                destination_folder=dest_p.parent,
                category="PAN CARD",
                routing_type="KYC",
                final_filename=dest_p.name,
                action="MOVED",
                source_exists=False,
                status="SUCCESS",
                error=None,
                elapsed_time=0.01,
                timings={},
            )

        with patch("kyc_sorter.sorter.vlm_router.process_vlm_file", side_effect=fake_process_vlm_file):
            summary = process_vlm_batch(
                files=files,
                destination_root=dest_root,
                cpu_monitor=fake_monitor,
                max_workers=2,
            )

        # CRITICAL ASSERTION: All 6 files must be processed! None abandoned!
        assert len(summary.results) == 6
        assert summary.completed_files == 6
        assert summary.total_files == 6
        assert fake_monitor.checks >= 2

    def test_cpu_throttling_deadlock_safety_timeout(self, tmp_path):
        """If CPU monitor remains permanently throttled, worker does NOT hang forever."""
        files = [tmp_path / "test_deadlock.jpg"]
        files[0].write_bytes(b"\xFF\xD8\xFF" + b"content" * 10)
        dest_root = tmp_path / "output"
        dest_root.mkdir()

        # Permanently throttled monitor with stale heartbeat
        class StuckCpuMonitor:
            cpu_exceeded = True
            throttle_active = True
            last_sample_time = 0.0  # Stale heartbeat breaks through deadlock safety
            samples = [95.0]
            peak_cpu = 95.0
            peak_mem_percent = 50.0
            ever_cpu_limit_exceeded = True
            peak_vram_mb = 0.0
            gpu_info = {}

            def stop(self):
                pass

        stuck_monitor = StuckCpuMonitor()
        from kyc_sorter.sorter.vlm_router import RouteResult

        def fake_process_vlm_file(image_file, **kwargs):
            p = Path(image_file)
            dest_p = dest_root / "test_deadlock_PAN_CARD.jpg"
            return RouteResult(
                source_file=p,
                destination_path=dest_p,
                destination_folder=dest_p.parent,
                category="PAN CARD",
                routing_type="KYC",
                final_filename=dest_p.name,
                action="MOVED",
                source_exists=False,
                status="SUCCESS",
                error=None,
                elapsed_time=0.01,
                timings={},
            )

        with patch("time.sleep"):
            with patch("kyc_sorter.sorter.vlm_router.process_vlm_file", side_effect=fake_process_vlm_file):
                summary = process_vlm_batch(
                    files=files,
                    destination_root=dest_root,
                    cpu_monitor=stuck_monitor,
                    max_workers=1,
                )

        # Worker broke through deadlock timeout (stuck monitor sample stale) and processed the file!
        assert len(summary.results) == 1
        assert summary.completed_files == 1


# ==============================================================================
# 4. Batch Streaming Boundaries Tests
# ==============================================================================

class TestBatchStreaming:
    """Tests for incremental os.scandir chunking and boundary conditions."""

    def test_iter_source_batches_chunking(self, tmp_path):
        """25 files with batch_size=10 yields 3 chunks: 10, 10, 5."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(25):
            (input_dir / f"img_{i:03d}.jpg").write_bytes(b"data")

        # Add an unsupported file and a temporary file that must be skipped
        (input_dir / "ignore.png").write_bytes(b"data")
        (input_dir / "temp.jpg.tmp").write_bytes(b"data")

        chunks = list(iter_source_batches(
            source_dir=input_dir,
            batch_size=10,
            supported_extensions={".jpg"},
        ))

        assert len(chunks) == 3
        assert len(chunks[0][0]) == 10
        assert len(chunks[1][0]) == 10
        assert len(chunks[2][0]) == 5

        all_names = [f.name for chunk, idx in chunks for f in chunk]
        assert len(all_names) == 25
        assert "ignore.png" not in all_names
        assert "temp.jpg.tmp" not in all_names

    def test_iter_source_batches_empty_directory(self, tmp_path):
        """Empty input directory yields 0 chunks."""
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        chunks = list(iter_source_batches(source_dir=input_dir, batch_size=50))
        assert len(chunks) == 0

    def test_iter_source_batches_non_existent_directory(self, tmp_path):
        """Non-existent directory yields 0 chunks."""
        missing = tmp_path / "does_not_exist"
        chunks = list(iter_source_batches(source_dir=missing, batch_size=50))
        assert len(chunks) == 0


# ==============================================================================
# 5. Watcher Concurrency Boundedness, Bursts & Backpressure Tests
# ==============================================================================

class TestWatcherConcurrencyAndBackpressure:
    """Tests for bounded watcher thread pool, event bursts, deduplication, and overflow backpressure."""

    def test_watcher_burst_event_deduplication(self, tmp_path):
        """Dropping multiple duplicate events for the same file processes exactly once."""
        input_dir = tmp_path / "Input"
        output_dir = tmp_path / "Output"
        input_dir.mkdir()
        output_dir.mkdir()

        config = Config(
            input=InputConfig(directory=str(input_dir)),
            output=OutputConfig(directory=str(output_dir), operation="copy"),
            processing=ProcessingConfig(max_workers=2, watcher_max_queue=10),
            watcher=WatcherConfig(stability_delay_sec=0.05, settle_timeout_sec=1.0, poll_interval_sec=0.02),
        )
        config.input.resolve(tmp_path)
        config.output.resolve(tmp_path)

        mock_processor = MagicMock()
        mock_processor.process_file.return_value = ProcessedItem(
            timestamp="2026-10-09 12:00:00",
            source_file="test_burst.jpg",
            source_path=str(input_dir / "test_burst.jpg"),
            document_type="PAN_CARD",
            confidence=0.95,
            status="IDENTIFIED",
            matched_template="PAN_CARD",
            destination_path=str(output_dir / "test_burst_PAN_CARD.jpg"),
            operation="copy",
            dry_run=False,
        )

        watcher = SourceFolderWatcher(config=config, processor=mock_processor)
        sample = input_dir / "test_burst.jpg"
        sample.write_bytes(b"\xFF\xD8\xFF" + b"pan data" * 20)

        # Trigger 5 rapid filesystem events for the same file
        for _ in range(5):
            watcher.on_filesystem_event(sample)

        time.sleep(0.3)
        watcher.stop()

        # Processor should only have been called ONCE
        assert mock_processor.process_file.call_count == 1

    def test_watcher_backpressure_overflow_drains_without_file_loss(self, tmp_path):
        """When queue capacity is reached, overflow buffer holds excess and drains them safely."""
        input_dir = tmp_path / "Input"
        output_dir = tmp_path / "Output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Set max_queue_size to 2 with 1 worker to force backpressure overflow
        config = Config(
            input=InputConfig(directory=str(input_dir)),
            output=OutputConfig(directory=str(output_dir), operation="copy"),
            processing=ProcessingConfig(max_workers=1, watcher_max_queue=2),
            watcher=WatcherConfig(stability_delay_sec=0.02, settle_timeout_sec=1.0, poll_interval_sec=0.01),
        )
        config.input.resolve(tmp_path)
        config.output.resolve(tmp_path)

        processed_files = []
        lock = threading.Lock()

        class SlowProcessor:
            def process_file(self, f: Path):
                with lock:
                    processed_files.append(f.name)
                time.sleep(0.05)
                return ProcessedItem(
                    timestamp="2026-10-09 12:00:00",
                    source_file=f.name,
                    source_path=str(f),
                    document_type="PAN_CARD",
                    confidence=0.95,
                    status="IDENTIFIED",
                    matched_template="PAN_CARD",
                    destination_path=str(output_dir / f.name),
                    operation="copy",
                    dry_run=False,
                )

        watcher = SourceFolderWatcher(config=config, processor=SlowProcessor())

        # Create 5 distinct files
        files = []
        for i in range(5):
            f = input_dir / f"doc_{i}.jpg"
            f.write_bytes(b"\xFF\xD8\xFF" + b"img" * 20)
            files.append(f)

        # Rapidly post all 5 file events
        for f in files:
            watcher.on_filesystem_event(f)

        # Wait for all workers and drained queue items to finish
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with lock:
                if len(processed_files) >= 5:
                    break
            time.sleep(0.05)

        watcher.stop()

        # ALL 5 files must be processed, none lost due to queue overflow!
        assert len(processed_files) == 5
        assert set(processed_files) == {f"doc_{i}.jpg" for i in range(5)}


# ==============================================================================
# 6. Destination Collision Handling Tests
# ==============================================================================

class TestCollisionSafety:
    """Verify multiple files with colliding destination names get distinct incremented suffixes."""

    def test_collision_increments_suffix_without_overwrite(self, tmp_path):
        output_dir = tmp_path / "Output"
        output_dir.mkdir()

        router = FileRouter(
            output_directory=output_dir,
            create_date_folder=False,
            create_subfolders=False,
            operation="copy",
        )

        src1 = tmp_path / "invoice.jpg"
        src1.write_bytes(b"content 1")
        res1 = router.route(src1, "PAN_CARD")
        assert res1.destination_path.name == "invoice_PAN_CARD.jpg"
        assert res1.destination_path.read_bytes() == b"content 1"

        # Route second file with identical base name
        src2 = tmp_path / "sub" / "invoice.jpg"
        src2.parent.mkdir()
        src2.write_bytes(b"content 2")
        res2 = router.route(src2, "PAN_CARD")
        assert res2.destination_path.name == "invoice_PAN_CARD_1.jpg"
        assert res2.destination_path.read_bytes() == b"content 2"

        # Route third file with identical base name
        src3 = tmp_path / "sub2" / "invoice.jpg"
        src3.parent.mkdir()
        src3.write_bytes(b"content 3")
        res3 = router.route(src3, "PAN_CARD")
        assert res3.destination_path.name == "invoice_PAN_CARD_2.jpg"
        assert res3.destination_path.read_bytes() == b"content 3"

        # Original files remain intact and not overwritten
        assert res1.destination_path.read_bytes() == b"content 1"
        assert res2.destination_path.read_bytes() == b"content 2"
        assert res3.destination_path.read_bytes() == b"content 3"
