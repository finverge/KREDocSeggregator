"""
Batch Processor for Large-Scale KYC Document Segregation.
Scans source directories, coordinates OCR, classification, and routing,
provides per-file fault isolation, generates summary reports and CSV logs.
"""

from __future__ import annotations

import csv
import datetime
import logging
import time
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from kyc_sorter.detector.classifier import ClassificationResult, DocumentClassifier
from kyc_sorter.detector.ocr import OCREngine, OCRResult
from kyc_sorter.sorter.file_router import FileRouter, RouteDecision
from kyc_sorter.utils.config_loader import Config

logger = logging.getLogger(__name__)

# Process-level singletons for worker processes (initialized once per process)
_WORKER_OCR_ENGINE: Optional[OCREngine] = None
_WORKER_CLASSIFIER: Optional[DocumentClassifier] = None


def compute_adaptive_workers(configured_max: Optional[int] = None, total_cores: Optional[int] = None) -> int:
    """
    Determine safe worker process count adaptively based on available CPU cores.
    Leaves headroom for the OS and desktop applications:
      - 1-2 cores: 1 worker
      - 3-4 cores: 2 workers
      - 5-7 cores: 3 workers
      - 8+ cores: min(configured_max, cores // 2)
    """
    cores = total_cores if (total_cores is not None and total_cores >= 1) else (os.cpu_count() or 1)
    if cores <= 2:
        adaptive = 1
    elif cores <= 4:
        adaptive = 2
    elif cores <= 7:
        adaptive = 3
    else:
        adaptive = max(2, cores // 2)

    limit = configured_max if (configured_max is not None and configured_max >= 1) else 4
    return max(1, min(adaptive, limit))


def _worker_process_init(confidence_threshold: float, language: str, cpu_threads: int):
    """
    Worker process initializer.
    Executed ONCE per worker process when spawned.
    Instantiates and pre-initializes OCREngine and DocumentClassifier as process singletons.
    """
    global _WORKER_OCR_ENGINE, _WORKER_CLASSIFIER
    # Configure safety env vars for worker process
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    os.environ["PADDLEX_DISABLE_AUTO_INIT"] = "True"
    os.environ["PADDLEX_LOG_LEVEL"] = "CRITICAL"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        _WORKER_OCR_ENGINE = OCREngine(language=language, cpu_threads=cpu_threads)
        # Pre-initialize PaddleOCR model once in worker memory
        _WORKER_OCR_ENGINE._get_ocr_instance()
    except Exception as e:
        logger.error("Failed to initialize OCREngine in worker process: %s", e)

    _WORKER_CLASSIFIER = DocumentClassifier(confidence_threshold=confidence_threshold)


def _worker_process_task(file_path_str: str) -> Dict[str, Any]:
    """
    Worker task: runs OCR once using the process-level OCREngine,
    runs DocumentClassifier once reusing the OCR text, and returns raw classification results.
    Does not write or route files; safe and picklable.
    """
    global _WORKER_OCR_ENGINE, _WORKER_CLASSIFIER
    t0 = time.time()
    try:
        file_path = Path(file_path_str)

        if _WORKER_OCR_ENGINE is None:
            _WORKER_OCR_ENGINE = OCREngine()
        if _WORKER_CLASSIFIER is None:
            _WORKER_CLASSIFIER = DocumentClassifier()

        ocr_res = _WORKER_OCR_ENGINE.extract_text(file_path)
        duration = time.time() - t0

        if not ocr_res.success:
            return {
                "source_path": file_path_str,
                "document_type": "UNKNOWN",
                "confidence": 0.0,
                "status": "FAILED",
                "matched_template": None,
                "error": ocr_res.error,
                "processing_time_sec": duration,
            }

        class_res = _WORKER_CLASSIFIER.classify_text(ocr_res.text)
        duration = time.time() - t0

        return {
            "source_path": file_path_str,
            "document_type": class_res.document_type,
            "confidence": class_res.confidence,
            "status": class_res.status,
            "matched_template": class_res.matched_template,
            "error": None,
            "processing_time_sec": duration,
        }
    except Exception as e:
        duration = time.time() - t0
        return {
            "source_path": file_path_str,
            "document_type": "UNKNOWN",
            "confidence": 0.0,
            "status": "FAILED",
            "matched_template": None,
            "error": str(e),
            "processing_time_sec": duration,
        }


@dataclass
class ProcessedItem:
    timestamp: str
    source_file: str
    source_path: str
    document_type: str
    confidence: float
    status: str
    matched_template: Optional[str]
    destination_path: str
    operation: str
    dry_run: bool
    error: Optional[str] = None
    processing_time_sec: float = 0.0


@dataclass
class BatchSummary:
    total_files: int = 0
    processed: int = 0
    failed: int = 0
    skipped_unsupported: int = 0
    counts_by_type: Dict[str, int] = field(default_factory=dict)
    items: List[ProcessedItem] = field(default_factory=list)
    csv_report_path: Optional[Path] = None
    elapsed_time_sec: float = 0.0

    def print_summary(self):
        """Display clean terminal summary table."""
        total_kyc = sum(v for k, v in self.counts_by_type.items() if k != "UNKNOWN")
        total_unknown = self.counts_by_type.get("UNKNOWN", 0)

        print("\n" + "=" * 55)
        print("KYC DOCUMENT PROCESSING COMPLETE")
        print("=" * 55)
        print(f"Total files        : {self.total_files}")
        print(f"Processed          : {self.processed}")
        print(f"Routed to KYC      : {total_kyc}")
        print(f"Routed to Unknown  : {total_unknown}")
        print(f"Failed             : {self.failed}")
        if self.skipped_unsupported > 0:
            print(f"Unsupported files  : {self.skipped_unsupported}")
        print("-" * 55)
        print("KYC Document Breakdown:")
        for doc_type, count in sorted(self.counts_by_type.items()):
            if doc_type != "UNKNOWN":
                print(f"  {doc_type:<28}: {count}")
        print(f"Time Elapsed       : {self.elapsed_time_sec:.2f}s")
        print("=" * 55 + "\n")


TEMPORARY_EXTENSIONS = frozenset({
    ".tmp", ".part", ".crdownload", ".download", ".partial",
    ".swp", ".lock", ".temp", ".pending", ".uploading"
})


class BatchProcessor:
    """
    Coordinates end-to-end processing of KYC documents from source to destination.
    """

    def __init__(
        self,
        config: Config,
        ocr_engine: Optional[OCREngine] = None,
        classifier: Optional[DocumentClassifier] = None,
        router: Optional[FileRouter] = None,
    ):
        self.config = config
        self.ocr_engine = ocr_engine or OCREngine(
            language=config.ocr.language,
            use_gpu=config.ocr.use_gpu,
            cpu_threads=config.ocr.cpu_threads,
        )
        self.classifier = classifier or DocumentClassifier(
            confidence_threshold=config.processing.confidence_threshold,
        )
        self.router = router or FileRouter(
            output_directory=config.output.resolved_path,
            folders_mapping=config.folders,
            suffixes_mapping=config.suffixes,
            kyc_folder=config.kyc_folder,
            unknown_folder=config.unknown_folder,
            unknown_suffix=config.unknown_suffix,
            operation=config.output.operation,
            create_date_folder=config.output.create_date_folder,
            create_subfolders=False,
            date_format=config.output.date_format,
            dry_run=config.processing.dry_run,
        )

    def scan_source_files(self, source_dir: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:
        """
        Scan directory for supported image files, returning (supported_files, unsupported_files).
        Excludes temporary and partial files (.tmp, .part, .crdownload).
        """
        src = Path(source_dir) if source_dir else self.config.input.resolved_path
        supported: List[Path] = []
        unsupported: List[Path] = []

        if not src.is_dir():
            logger.warning("Source directory does not exist: %s", src)
            return supported, unsupported

        valid_exts = self.config.processing.supported_extensions

        for item in sorted(src.iterdir()):
            if item.is_file():
                ext = item.suffix.lower()
                # Skip temporary / partial / hidden files
                if ext in TEMPORARY_EXTENSIONS or item.name.startswith(".") or item.name.startswith("~"):
                    continue
                if ext in valid_exts:
                    supported.append(item)
                else:
                    unsupported.append(item)

        return supported, unsupported

    def process_file(
        self,
        file_path: Path,
        processing_date: Optional[datetime.date] = None,
    ) -> ProcessedItem:
        """
        Process a single document through OCR, classification, and routing.
        Guaranteed not to raise uncaught exceptions.
        """
        start_time = time.time()
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ignore non-supported file formats completely (do not OCR, classify, copy, move, or rename)
        ext = file_path.suffix.lower()
        if ext not in self.config.processing.supported_extensions:
            logger.info("Ignoring non-supported file '%s' (only %s supported).", file_path.name, ", ".join(sorted(self.config.processing.supported_extensions)))
            return ProcessedItem(
                timestamp=ts_str,
                source_file=file_path.name,
                source_path=str(file_path.resolve()),
                document_type="IGNORED",
                confidence=0.0,
                status="IGNORED",
                matched_template=None,
                destination_path="",
                operation=self.config.output.operation,
                dry_run=self.config.processing.dry_run,
                error=f"Ignored non-supported format: {file_path.suffix}",
                processing_time_sec=0.0,
            )

        try:
            # 1. OCR Extraction
            ocr_res = self.ocr_engine.extract_text(file_path)

            if not ocr_res.success:
                # OCR failed (e.g. corrupt or unsupported image format)
                decision = self.router.route(file_path, "UNKNOWN", processing_date=processing_date)
                duration = time.time() - start_time
                logger.warning(
                    "[%s] OCR_FAILED: '%s' -> %s (%s)",
                    ts_str,
                    file_path.name,
                    decision.destination_path,
                    ocr_res.error,
                )
                return ProcessedItem(
                    timestamp=ts_str,
                    source_file=file_path.name,
                    source_path=str(file_path.resolve()),
                    document_type="UNKNOWN",
                    confidence=0.0,
                    status="FAILED",
                    matched_template=None,
                    destination_path=str(decision.destination_path),
                    operation=self.config.output.operation,
                    dry_run=self.config.processing.dry_run,
                    error=ocr_res.error,
                    processing_time_sec=duration,
                )

            # 2. Document Classification
            class_res = self.classifier.classify_text(ocr_res.text)

            # 3. File Routing
            decision = self.router.route(
                file_path,
                class_res.document_type,
                processing_date=processing_date,
            )

            duration = time.time() - start_time

            # Log requirement 12
            action_tag = "WOULD_ROUTE" if self.config.processing.dry_run else decision.operation.upper()
            logger.info(
                "%s | %s | %s | conf=%.2f | dest=%s | status=%s",
                ts_str,
                file_path.name,
                class_res.document_type,
                class_res.confidence,
                decision.destination_path,
                class_res.status,
            )

            return ProcessedItem(
                timestamp=ts_str,
                source_file=file_path.name,
                source_path=str(file_path.resolve()),
                document_type=class_res.document_type,
                confidence=class_res.confidence,
                status=class_res.status,
                matched_template=class_res.matched_template,
                destination_path=str(decision.destination_path),
                operation=self.config.output.operation,
                dry_run=self.config.processing.dry_run,
                error=decision.error,
                processing_time_sec=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Unexpected error processing '%s': %s", file_path.name, e, exc_info=True)
            # Route to unknown on unexpected failure
            try:
                decision = self.router.route(file_path, "UNKNOWN", processing_date=processing_date)
                dest_str = str(decision.destination_path)
            except Exception:
                dest_str = str(self.config.output.resolved_path / self.config.unknown_folder / file_path.name)

            return ProcessedItem(
                timestamp=ts_str,
                source_file=file_path.name,
                source_path=str(file_path.resolve()),
                document_type="UNKNOWN",
                confidence=0.0,
                status="FAILED",
                matched_template=None,
                destination_path=dest_str,
                operation=self.config.output.operation,
                dry_run=self.config.processing.dry_run,
                error=str(e),
                processing_time_sec=duration,
            )

    def process_batch(
        self,
        source_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, ProcessedItem], None]] = None,
    ) -> BatchSummary:
        """
        Execute full batch processing across all files in source directory.
        """
        batch_start = time.time()
        supported_files, unsupported_files = self.scan_source_files(source_dir)

        total_count = len(supported_files)
        logger.info("Found %d supported files to process (%d unsupported skipped).", total_count, len(unsupported_files))

        summary = BatchSummary(
            total_files=total_count,
            skipped_unsupported=len(unsupported_files),
        )

        if total_count == 0:
            summary.elapsed_time_sec = time.time() - batch_start
            return summary

        # Determine adaptive worker count
        configured_max = getattr(self.config.processing, "max_workers", 4)
        effective_workers = compute_adaptive_workers(configured_max)
        num_workers = min(effective_workers, total_count)

        if num_workers <= 1:
            logger.info("Processing %d files sequentially (workers=1)...", total_count)
            for i, fpath in enumerate(supported_files, start=1):
                item = self.process_file(fpath)
                summary.items.append(item)
                summary.processed += 1

                if item.status == "FAILED":
                    summary.failed += 1

                doc_type = item.document_type
                summary.counts_by_type[doc_type] = summary.counts_by_type.get(doc_type, 0) + 1

                if progress_callback:
                    progress_callback(i, total_count, item)
        else:
            logger.info(
                "Processing %d files concurrently using %d worker processes (adaptive limit from max %d on %d CPU cores)...",
                total_count,
                num_workers,
                configured_max,
                os.cpu_count() or 1,
            )
            file_path_strs = [str(f.resolve()) for f in supported_files]
            completed_count = 0
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_worker_process_init,
                    initargs=(
                        self.config.processing.confidence_threshold,
                        self.config.ocr.language,
                        self.config.ocr.cpu_threads,
                    ),
                ) as executor:
                    results = executor.map(_worker_process_task, file_path_strs, chunksize=1)
                    for raw_res in results:
                        completed_count += 1
                        fpath = Path(raw_res["source_path"])
                        doc_type = raw_res["document_type"]
                        status = raw_res["status"]
                        error = raw_res["error"]

                        # Execute safe sequential routing & collision handling on main process
                        decision = self.router.route(fpath, doc_type)
                        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        item = ProcessedItem(
                            timestamp=ts_str,
                            source_file=fpath.name,
                            source_path=str(fpath),
                            document_type=doc_type,
                            confidence=raw_res["confidence"],
                            status=status,
                            matched_template=raw_res["matched_template"],
                            destination_path=str(decision.destination_path),
                            operation=self.config.output.operation,
                            dry_run=self.config.processing.dry_run,
                            error=error or decision.error,
                            processing_time_sec=raw_res["processing_time_sec"],
                        )

                        summary.items.append(item)
                        summary.processed += 1
                        if item.status == "FAILED":
                            summary.failed += 1
                        summary.counts_by_type[doc_type] = summary.counts_by_type.get(doc_type, 0) + 1

                        logger.info(
                            "[%s] %s | %s | conf=%.2f | dest=%s | status=%s",
                            item.timestamp,
                            item.source_file,
                            item.document_type,
                            item.confidence,
                            item.destination_path,
                            item.status,
                        )

                        if progress_callback:
                            progress_callback(completed_count, total_count, item)

            except KeyboardInterrupt:
                logger.warning("\nBatch processing interrupted by user (Ctrl+C). Terminating worker processes cleanly...")

        summary.elapsed_time_sec = time.time() - batch_start

        # CSV report generation is completely disabled per requirements
        summary.csv_report_path = None

        return summary

    def _write_csv_report(self, items: List[ProcessedItem]) -> Optional[Path]:
        """CSV report generation is completely disabled per requirements."""
        return None
