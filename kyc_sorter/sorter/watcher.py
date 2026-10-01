"""
Source Folder Filesystem Event Listener and Watcher for KYC Document Sorter.
Continuously monitors the configured input directory for newly added KYC image files,
safely verifies complete file writes, prevents duplicate processing across OS events,
and routes classified & renamed documents into date-partitioned output folders.
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from kyc_sorter.sorter.batch_processor import BatchProcessor, ProcessedItem
from kyc_sorter.utils.config_loader import Config, load_config
from kyc_sorter.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def wait_for_file_ready(
    file_path: Path,
    stability_delay_sec: float = 1.0,
    settle_timeout_sec: float = 30.0,
    check_interval_sec: float = 0.5,
) -> bool:
    """
    Waits until an incoming image file is completely written, readable, and size-stable.

    Verifications performed:
      1. File exists on disk and is a regular file.
      2. File size is greater than zero.
      3. File can be opened in binary read mode without OS sharing or write locks.
      4. File size remains identical across stability_delay_sec.
      5. File is verified readable again after the delay.

    Returns:
      True if the file is completely written and stable.
      False if settle_timeout_sec expires or file is removed.
    """
    start_time = time.time()
    p = Path(file_path)

    while (time.time() - start_time) < settle_timeout_sec:
        if not p.exists() or not p.is_file():
            time.sleep(check_interval_sec)
            continue

        try:
            size1 = p.stat().st_size
        except OSError:
            time.sleep(check_interval_sec)
            continue

        if size1 <= 0:
            time.sleep(check_interval_sec)
            continue

        # Check if file can be opened without lock contention
        try:
            with open(p, "rb") as f:
                f.read(min(size1, 4096))
        except (PermissionError, OSError):
            time.sleep(check_interval_sec)
            continue

        # Sleep stability delay to detect continuing writes/copies
        time.sleep(stability_delay_sec)

        if not p.exists() or not p.is_file():
            return False

        try:
            size2 = p.stat().st_size
        except OSError:
            time.sleep(check_interval_sec)
            continue

        # If size is identical and positive, perform final verification
        if size1 == size2 and size2 > 0:
            try:
                with open(p, "rb") as f:
                    f.read(min(size2, 4096))
                return True
            except (PermissionError, OSError):
                time.sleep(check_interval_sec)
                continue

    logger.warning(
        "File '%s' failed to stabilize within %.1fs timeout (last size: %s).",
        p.name,
        settle_timeout_sec,
        p.stat().st_size if p.exists() else "deleted",
    )
    return False


TEMPORARY_EXTENSIONS = {
    ".tmp",
    ".part",
    ".crdownload",
    ".download",
    ".partial",
    ".temp",
    ".swp",
}


class KYCFileSystemEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that intercepts creation, modification, and move events.
    Filters by supported image extensions and delegates to SourceFolderWatcher.
    """

    def __init__(self, watcher: SourceFolderWatcher):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            self.watcher.on_filesystem_event(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self.watcher.on_filesystem_event(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            dest = getattr(event, "dest_path", None)
            if dest:
                self.watcher.on_filesystem_event(Path(dest))


class SourceFolderWatcher:
    """
    Continuous filesystem watcher for incoming KYC document images.
    Implements safe file-ready checks, duplicate suppression, and automated routing.
    """

    def __init__(
        self,
        config: Config,
        processor: Optional[BatchProcessor] = None,
        on_event_callback: Optional[Callable[[ProcessedItem], None]] = None,
    ):
        self.config = config
        self.processor = processor or BatchProcessor(config=config)
        self.on_event_callback = on_event_callback

        self.watch_dir = self.config.input.resolved_path
        self.supported_extensions = self.config.processing.supported_extensions

        # Concurrency & duplicate prevention
        self._lock = threading.Lock()
        self.in_progress: Set[Path] = set()
        # Maps canonical Path -> (st_mtime, st_size)
        self.processed_files: Dict[Path, Tuple[float, int]] = {}

        self.observer: Optional[Observer] = None
        self.is_running: bool = False
        self._csv_lock = threading.Lock()

    def on_filesystem_event(self, path: Path):
        """
        Handle incoming event notification from filesystem watcher.
        Performs preliminary checks before spawning safe processing task.
        """
        resolved_path = path.resolve()

        # Ignore temporary / partial download files (.tmp, .part, .crdownload, etc.)
        suffix_lower = resolved_path.suffix.lower()
        if suffix_lower in TEMPORARY_EXTENSIONS or any(
            resolved_path.name.lower().endswith(temp_ext) for temp_ext in TEMPORARY_EXTENSIONS
        ):
            return

        # Check supported file extension
        if suffix_lower not in self.supported_extensions:
            return

        # Ignore hidden/system/lock files
        if resolved_path.name.startswith(".") or resolved_path.name.startswith("~"):
            return

        with self._lock:
            # 1. Check if already in progress
            if resolved_path in self.in_progress:
                return

            # 2. Check if already processed with matching signature
            if resolved_path.exists():
                try:
                    st = resolved_path.stat()
                    if self.processed_files.get(resolved_path) == (st.st_mtime, st.st_size):
                        return
                except OSError:
                    pass

        # Spawn asynchronous worker thread to avoid blocking the watchdog observer loop
        worker = threading.Thread(
            target=self._process_single_file_safe,
            args=(resolved_path,),
            daemon=True,
            name=f"KYCWatcherWorker-{resolved_path.name}",
        )
        worker.start()

    def _process_single_file_safe(self, file_path: Path):
        """
        Thread-safe file processor:
          1. Sets in_progress lock.
          2. Waits until file write is complete and stable.
          3. Checks duplicate processing cache.
          4. Executes OCR, Classification, and Renaming + Routing.
          5. Updates duplicate cache and CSV report.
        """
        with self._lock:
            if file_path in self.in_progress:
                return
            self.in_progress.add(file_path)

        try:
            logger.info("New document event detected: '%s'. Awaiting file write completion...", file_path.name)

            # Wait until file is completely written and readable
            is_ready = wait_for_file_ready(
                file_path=file_path,
                stability_delay_sec=self.config.watcher.stability_delay_sec,
                settle_timeout_sec=self.config.watcher.settle_timeout_sec,
                check_interval_sec=self.config.watcher.poll_interval_sec,
            )

            if not is_ready:
                logger.warning(
                    "Skipping '%s': file was not stable or readable within timeout.",
                    file_path.name,
                )
                return

            # Check if file still exists
            if not file_path.is_file():
                return

            # Obtain stable signature
            try:
                st = file_path.stat()
                stable_sig = (st.st_mtime, st.st_size)
            except OSError:
                return

            # Verify against duplicate tracker with stable signature
            with self._lock:
                if self.processed_files.get(file_path) == stable_sig:
                    logger.debug("Skipping '%s': duplicate event already processed.", file_path.name)
                    return

            logger.info("File '%s' is ready. Starting local OCR extraction...", file_path.name)

            # Process file: OCR -> Classification -> Suffix Renaming -> Date Folder Routing
            item = self.processor.process_file(file_path)

            # Mark file as processed with its stable signature
            with self._lock:
                self.processed_files[file_path] = stable_sig

            # CSV report generation is completely disabled per requirements

            # Notify GUI or consumer callback
            if self.on_event_callback:
                try:
                    self.on_event_callback(item)
                except Exception as cb_err:
                    logger.error("Error invoking watcher callback: %s", cb_err)

        except Exception as e:
            logger.error("Error processing watched file '%s': %s", file_path.name, e, exc_info=True)
            # Route unreadable / corrupted document to Unknown if file exists
            try:
                if file_path.exists() and file_path.is_file():
                    err_decision = self.processor.router.route(file_path, "UNKNOWN")
                    logger.info("Routed failed file '%s' to Unknown: %s", file_path.name, err_decision.destination_path)
            except Exception as routing_err:
                logger.error("Could not route failed file '%s' to Unknown: %s", file_path.name, routing_err)
        finally:
            with self._lock:
                self.in_progress.discard(file_path)

    def _append_csv_report(self, item: ProcessedItem) -> Optional[Path]:
        """CSV report generation is completely disabled per requirements."""
        return None

    def start(self, process_existing: bool = False):
        """
        Start monitoring the configured source directory for incoming files.
        If process_existing is True, existing supported files in the source directory
        are processed prior to or alongside event monitoring.
        """
        if self.is_running:
            logger.warning("Watcher is already running on: %s", self.watch_dir)
            return

        if not self.watch_dir.is_dir():
            raise FileNotFoundError(f"Source watch directory does not exist: {self.watch_dir}")

        event_handler = KYCFileSystemEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        self.is_running = True

        logger.info("=" * 60)
        logger.info("SOURCE FOLDER EVENT LISTENER ACTIVE")
        logger.info("Monitoring Folder   : %s", self.watch_dir)
        logger.info("Supported Extensions: %s", ", ".join(sorted(self.supported_extensions)))
        logger.info("Destination Root    : %s", self.config.output.resolved_path)
        logger.info("Operation Mode      : %s", self.config.output.operation.upper())
        logger.info("Confidence Gate     : %.0f%%", self.config.processing.confidence_threshold * 100)
        logger.info("=" * 60)

        if process_existing:
            logger.info("Scanning for existing files in source folder...")
            supported, _ = self.processor.scan_source_files(self.watch_dir)
            for f in supported:
                self.on_filesystem_event(f)

    def stop(self):
        """Stop filesystem monitoring."""
        if not self.is_running:
            return

        logger.info("Stopping source folder event listener...")
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.observer = None
        self.is_running = False
        logger.info("Source folder event listener stopped.")

    def join(self, timeout: Optional[float] = None):
        """Wait for the observer thread to complete."""
        if self.observer:
            self.observer.join(timeout=timeout)
